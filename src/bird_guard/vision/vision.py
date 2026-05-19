import time
from collections import deque
from enum import Enum

import cv2
import numpy as np
import math

from bird_guard.camera.camera import Frame
from bird_guard.vision.utils.debug_utils import DebugViewer, DebugUtils
from bird_guard.vision.utils.image_utils import ImageUtils, FloatImage, GrayImage
from bird_guard.vision.utils.vision_utils import VisionUtils
from bird_guard.vision.vision_config import ModuleConfig_Vision

# ===============
# Motion Detector
# ===============
class MotionDetector:

    def __init__(self, settings: ModuleConfig_Vision, threshold: int = 400):
        self.settings = settings

        self.prev_gray_blurred = None
        self.threshold = threshold

        self.debug_view = None

        self.diff_method: VisionUtils.DetectionMode = VisionUtils.DetectionMode.BG_REM   # TODO: Make configurable or remove
        self.background_subtractor: cv2.BackgroundSubtractor | None = None

        self.activity_map: FloatImage | None = None

        self.is_first_iteration: bool = True
        self.is_in_warmup: bool = True
        self.warmup_control: WarmupControl = WarmupControl()
        self.current_timestamp: float = time.time()

        self.min_shape_angle = math.atan2(1, 3)     # TODO: depends on cell size ...
        self.max_shape_angle = math.atan2(3, 1)

        self.detection_state_machine = DetectionStateMachine()
        #self.potential_detection_timestamp: float | None = None
        #self.detection_start_timestamp: float | None = None
        #self.potential_detection_end_timestamp: float | None = None

        self._init_components()

    def _init_components(self):
        # disable IPP (Intel Performance Primitives) to get consistent look on all platforms
        cv2.setUseOptimized(False)

        # create MOG2 object, if required
        if self.diff_method == VisionUtils.DetectionMode.BG_REM:
            self.background_subtractor = cv2.createBackgroundSubtractorMOG2(
                history=200,
                varThreshold=25,
                detectShadows=False # False: generates result images with only 0 and 255; True: also 127 (shadow)
            )

        # create DebugViewer object, if debug is enabled
        if self.settings.debug:
            self.debug_view = DebugViewer()
            self._setup_debug_view()

    def _show_debug_window(self):
        if self.debug_view is not None:
            self.debug_view.show_debug_window()

    def _setup_debug_view(self):
        self.debug_view.viewer_config.set_image_matrix_size(2, 2)
        self.debug_view.viewer_config.set_image_matrix_dimensions(1440, 810)

        self.debug_view.viewer_config.register_image(0, 0, "top_left")
        self.debug_view.viewer_config.register_image(0, 1, "top_right")
        self.debug_view.viewer_config.register_image(1, 0, "bottom_left")
        self.debug_view.viewer_config.register_image(1, 1, "bottom_right")


    def detect_movement(self, frame: Frame) -> bool:

        # pre-process
        # -----------
        gray, gray_blurred = self._pre_process(frame)

        # open debug window on first call
        if self.is_first_iteration:
            self._show_debug_window()

        # PROCESS
        # -------
        if self.prev_gray_blurred is None and self.diff_method is VisionUtils.DetectionMode.IM_DIFF:
            movement_detected = False
        else:
            movement_detected = self._process(gray, gray_blurred)

        # post-process
        # ------------
        self._post_process(gray_blurred)

        # return results
        # --------------
        return movement_detected


    def _pre_process(self, frame: Frame) -> tuple[GrayImage, GrayImage]:
        """Extract everything needed from the frame and init iteration"""

        # store timestamp of current frame
        self.current_timestamp = frame.timestamp

        # get current frame as grayscale image
        gray = VisionUtils.get_frame_as_gray_image(frame)

        # BLUR gray image to reduce noise
        gray_blurred = VisionUtils.get_blurred_gray_image(gray, frame.type)

        return gray, gray_blurred


    def _post_process(self, gray_blurred: GrayImage):
        """Update internal states to enable starting over with the next iteration"""

        self.is_first_iteration = False

        # update image/frame history
        self.prev_gray_blurred = gray_blurred



    def _process(self, gray: GrayImage, gray_blurred: GrayImage) -> bool:
        """The actual image processing pipeline"""

        # set gray images to debug viewer
        if self.debug_view: self.debug_view.set_image("top_left", gray)
        if self.debug_view: self.debug_view.set_image("top_right", gray_blurred)

        # get DIFF image by selected method
        match self.diff_method:
            case VisionUtils.DetectionMode.IM_DIFF:
                diff = VisionUtils.get_image_sequence_difference_simple(gray_blurred, self.prev_gray_blurred)
            case VisionUtils.DetectionMode.BG_REM:
                diff = VisionUtils.get_image_sequence_difference_MOG2(gray_blurred, self.background_subtractor)
            case _:
                raise NotImplementedError(f"Diff image generation method {self.diff_method} is not implemented!")

        if self.debug_view: self.debug_view.set_image("bottom_left", diff)


        # get CONTOURS of changed areas (and the corresponding mask and the number of changed pixels)
        contours, changes_mask, changed_pixels = VisionUtils.get_contours_by_threshold(diff)
        big_contours = VisionUtils.filter_big_contours(contours)
        extreme_contours = [] # TODO: Remove this old contour computation

        # update ACTIVITY MAP
        self.activity_map, current_activity_map = VisionUtils.update_activity_map(
            changes_mask,
            self.activity_map,
            self.settings.motion_detector.activity_map.cell_size,
            alpha=self.settings.motion_detector.activity_map.alpha_warmup if self.is_in_warmup else self.settings.motion_detector.activity_map.alpha
        )

        # check finish of WARMUP phase
        if self.is_in_warmup:
            self.warmup_control.put_time_value(self.current_timestamp, float(np.min(self.activity_map)))
            self.is_in_warmup = not self.warmup_control.is_warmup_finished()

        # detect MOVEMENT (primitive)
        if self.is_in_warmup:
            movement_detected = False
        else:
            detection = (len(big_contours) > 0 or len(extreme_contours) > 0)
            self.detection_state_machine.update(detection, self.current_timestamp)
            match self.detection_state_machine.get_state():
                case (DetectionStateMachine.DetectionState.NO_DETECTION |
                      DetectionStateMachine.DetectionState.POTENTIAL_DETECTION):
                    movement_detected = False
                case (DetectionStateMachine.DetectionState.DETECTION |
                      DetectionStateMachine.DetectionState.POTENTIAL_DETECTION_END):
                    movement_detected = True
                case _:
                    raise RuntimeError(f"Unhandled case: {self.detection_state_machine.get_state().name}")

        # draw DEBUG IMAGE, if enabled
        if self.debug_view is not None:
            # update debug image in debug viewer
            self.debug_view.set_image("bottom_right", DebugUtils.draw_debug_image(gray, contours, big_contours, current_activity_map))
            activity_mix_image = DebugUtils.draw_activity_mix_image(self.activity_map, gray_blurred, current_activity_map)

            # test: draw the activity maps
            short_activity_map = current_activity_map
            short_activity_image = ImageUtils.mix_in_binary_image(gray, np.float32(short_activity_map), (0.0, 0.5, 1.0))

            # print min/max info (short activity)
            image_w, image_h = ImageUtils.get_image_size_wh(short_activity_image)
            ImageUtils.draw_text(short_activity_image, f"Min: {np.min(short_activity_map):.4f} | Max: {np.max(short_activity_map):.4f}",
                                 (image_w - 20, image_h - 20), anchor=ImageUtils.TextAnchor.BOTTOM_RIGHT, font_scale=0.5)

            long_activity_map = self.activity_map
            long_activity_image = ImageUtils.mix_in_binary_image(gray, np.float32(long_activity_map) / max(np.max(long_activity_map),0.1), (0.0, 1.0, 0.0))

            # print min/max info (long activity)
            image_w, image_h = ImageUtils.get_image_size_wh(long_activity_image)
            _, text_height = ImageUtils.draw_text(long_activity_image, f"Min: {np.min(long_activity_map):.4f} | Max: {np.max(long_activity_map):.4f}",
                                 (image_w - 20, image_h - 20), anchor=ImageUtils.TextAnchor.BOTTOM_RIGHT, font_scale=0.5)

            # print warmup info
            if self.is_in_warmup:
                ImageUtils.draw_text(long_activity_image, "WARMUP",
                                     (image_w - 20, image_h - text_height - 20), anchor=ImageUtils.TextAnchor.BOTTOM_RIGHT, font_scale=0.5, text_color_bgr=(0, 0, 255))

            # compute activity difference
            activity_diff_map = ImageUtils.float_image_to_gray(np.clip(np.subtract(current_activity_map, self.activity_map), 0.0, 1.0))

            """DEV"""
            num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
                (activity_diff_map > 150).astype(np.uint8), # TODO: Parameter
                connectivity=8
            )

            candidates = []
            for stat in stats[1:,:]:  # row 0 is the background -> skip
                if stat[cv2.CC_STAT_AREA] > 4:
                    shape_angle = math.atan2(stat[cv2.CC_STAT_HEIGHT],stat[cv2.CC_STAT_WIDTH])
                    if self.min_shape_angle < shape_angle < self.max_shape_angle:
                        candidates.append(stat)

            activity_diff_image = ImageUtils.rescale(activity_diff_map, (image_w, image_h))
            activity_candidate_image = ImageUtils.color_image_to_gray(ImageUtils.get_blank_bgr_image((image_w, image_h)))
            for candidate in candidates:
                activity_map_cell_size = self.settings.motion_detector.activity_map.cell_size
                bbx_x = candidate[cv2.CC_STAT_LEFT] * activity_map_cell_size[0]
                bbx_y = candidate[cv2.CC_STAT_TOP] * activity_map_cell_size[1]
                bbx_w = candidate[cv2.CC_STAT_WIDTH] * activity_map_cell_size[0]
                bbx_h = candidate[cv2.CC_STAT_HEIGHT] * activity_map_cell_size[1]
                roi = activity_diff_image[bbx_y:bbx_y+bbx_h, bbx_x:bbx_x+bbx_w]
                activity_candidate_image[bbx_y:bbx_y+bbx_h, bbx_x:bbx_x+bbx_w] = roi


            # generate and draw contours from difference
            contours, *_ = VisionUtils.get_contours_by_threshold(activity_candidate_image, threshold_value=150)    # TODO: Parameter
            big_contours = VisionUtils.filter_big_contours(contours=contours, min_area=900) # TODO: Parameter
            extreme_contours = VisionUtils.filter_big_contours(contours=big_contours, min_area=51200)   # TODO: Parameter

            cv2.drawContours(activity_mix_image, contours, -1, (0, 255, 0), 1)
            cv2.drawContours(activity_mix_image, big_contours, -1, (0, 255, 255), 1)
            cv2.drawContours(activity_mix_image, extreme_contours, -1, (0, 0, 255), 1)

            # compute brightness and print info
            brightness = gray_blurred.mean()
            ImageUtils.draw_text(activity_mix_image, f"Brightness: {brightness:.2f}", (image_w - 20, image_h - 20),
                                 anchor=ImageUtils.TextAnchor.BOTTOM_RIGHT, font_scale=0.5)

            match(self.detection_state_machine.get_state()):
                case DetectionStateMachine.DetectionState.POTENTIAL_DETECTION:
                    ImageUtils.draw_text(activity_mix_image, "POTENTIAL", (20, image_h - 20),
                                         anchor=ImageUtils.TextAnchor.BOTTOM_LEFT, font_scale=0.5,
                                         thickness=2, text_color_bgr=(0, 255, 255))
                case DetectionStateMachine.DetectionState.DETECTION:
                    ImageUtils.draw_text(activity_mix_image, "DETECTION", (20, image_h - 20),
                                         anchor=ImageUtils.TextAnchor.BOTTOM_LEFT, font_scale=0.5,
                                         thickness=2, text_color_bgr=(0, 0, 255))
                case DetectionStateMachine.DetectionState.POTENTIAL_DETECTION_END:
                    ImageUtils.draw_text(activity_mix_image, "POTENTIAL END", (20, image_h - 20),
                                         anchor=ImageUtils.TextAnchor.BOTTOM_LEFT, font_scale=0.5,
                                         thickness=2, text_color_bgr=(0, 255, 0))
                case _:
                    pass

            self.debug_view.set_image("top_right", activity_mix_image)
            self.debug_view.set_image("top_left", short_activity_image)
            self.debug_view.set_image("bottom_left", long_activity_image)


        # update images in the DEBUG VIEW
        if self.debug_view:
            self.debug_view.update_debug_view()


        return movement_detected


# =====================
# DetectionStateMachine
# =====================
class DetectionStateMachine:
    class DetectionState(Enum):
        NO_DETECTION = 0
        POTENTIAL_DETECTION = 1
        DETECTION = 2
        POTENTIAL_DETECTION_END = 3

    def __init__(self):
        self._state = DetectionStateMachine.DetectionState.NO_DETECTION

        self._timestamp_potential: float | None = None
        self._timestamp_detection: float | None = None
        self._timestamp_potential_end: float | None = None

    def get_state(self) -> DetectionState:
        return self._state

    def update(self, detection: bool, timestamp: float):

        match self._state:
            case DetectionStateMachine.DetectionState.NO_DETECTION:
                # reset timestamps
                self._timestamp_potential = None
                self._timestamp_detection = None
                self._timestamp_potential_end = None

                # detection -> potential detection
                if detection:
                    self._timestamp_potential = timestamp
                    print(f"Potential detection at: {timestamp}")
                    self._state = DetectionStateMachine.DetectionState.POTENTIAL_DETECTION

            case DetectionStateMachine.DetectionState.POTENTIAL_DETECTION:
                # detection -> real detection, it at least 2.0 seconds old
                if detection:
                    if timestamp - self._timestamp_potential > 2.0: # TODO: Parameter
                        self._timestamp_detection = timestamp
                        print(f"Detection confirmed at: {timestamp}")
                        self._state = DetectionStateMachine.DetectionState.DETECTION

                # no detection -> reject potential detection, after a total of 2.0 seconds of consideration
                else:
                    if timestamp - self._timestamp_potential > 2.0: # TODO: Parameter
                        self._timestamp_potential = None
                        print(f"Detection rejected at: {timestamp}")
                        self._state = DetectionStateMachine.DetectionState.NO_DETECTION

            case DetectionStateMachine.DetectionState.DETECTION:
                # no detection -> potential detection end
                if not detection:
                    self._timestamp_potential_end = timestamp
                    print(f"Potential detection end at: {timestamp}")
                    self._state = DetectionStateMachine.DetectionState.POTENTIAL_DETECTION_END

            case DetectionStateMachine.DetectionState.POTENTIAL_DETECTION_END:
                # detection -> abort detection end
                if detection:
                    self._timestamp_potential_end = None
                    print(f"Aborted detection end at: {timestamp}")
                    self._state = DetectionStateMachine.DetectionState.DETECTION

                # no detection -> confirm detection end, if no detection for at least 4.0 seconds
                else:
                    if timestamp - self._timestamp_potential_end > 4.0: # TODO: Parameter
                        print(f"Confirmed detection end at: {timestamp}")
                        self._state = DetectionStateMachine.DetectionState.NO_DETECTION

            case _:
                raise RuntimeError(f"Unhandled case: {self._state.name}")


# =============
# WarmupControl
# =============
class WarmupControl:
    def __init__(self):
        self.initial_value_array: list[tuple[float, float]] | None = None
        self.warmup_queue: deque | None = None
        self.warmup_finished: bool = False

    def put_time_value(self, timestamp: float, value: float):
        new_tuple = (timestamp, value)
        if self.warmup_queue is None:
            if self.initial_value_array is None:
                self.initial_value_array = [new_tuple]
            else:
                diff_time = timestamp - self.initial_value_array[0][0]
                if diff_time >= 2.0:    # TODO: Parameter
                    self.warmup_queue = deque(self.initial_value_array, maxlen=len(self.initial_value_array))
                    self.warmup_queue.append(new_tuple)
                else:
                    self.initial_value_array.append(new_tuple)
        else:
            self.warmup_queue.append(new_tuple)

    def is_warmup_finished(self) -> bool:
        if self.warmup_finished:
            return True

        if self.warmup_queue is None:
            return False
        else:
            oldest_element = self.warmup_queue[0]
            newest_element = self.warmup_queue[-1]
            diff_time = newest_element[0] - oldest_element[0]
            diff_val  = newest_element[1] - oldest_element[1]
            if diff_time > 0:
                diff_rate = diff_val / diff_time
                if diff_rate > -0.005 and newest_element[1] < 0.02:   # TODO: Parameters
                    self.warmup_finished = True
                    return True
                else:
                    return False
            else:
                return False

