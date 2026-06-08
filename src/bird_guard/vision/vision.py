import time
from collections import deque
from enum import Enum
from itertools import chain

import cv2
import numpy as np
import math

from bird_guard.camera.camera import Frame
from bird_guard.utils import DebugInfo
from bird_guard.vision.utils.debug_utils import DebugViewer, DebugUtils
from bird_guard.vision.utils.image_utils import ImageUtils, FloatImage, GrayImage, BGRImage
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
        self.brightness_float_map: FloatImage | None = None
        self.brightness_change_map: BGRImage | None = None

        self.image_size_wh: tuple[int, int] | None = None
        self.map_size_wh: tuple[int, int] | None = None

        self.is_first_iteration: bool = True
        self.is_in_warmup: bool = True
        self.warmup_control: WarmupControl = WarmupControl()
        self.current_timestamp: float = time.time()

        self.min_shape_angle = math.atan2(1, 3) # TODO: Parameter?
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
                varThreshold=10,
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


    def detect_movement(self, frame: Frame, debug_info: DebugInfo | None = None) -> bool:

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
            movement_detected = self._process(gray, gray_blurred, debug_info)

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

        # get and store the working image size
        if self.image_size_wh is None:
            self.image_size_wh = ImageUtils.get_image_size_wh(gray)

        # BLUR gray image to reduce noise
        gray_blurred = VisionUtils.get_blurred_gray_image(gray, frame.type)

        return gray, gray_blurred


    def _post_process(self, gray_blurred: GrayImage):
        """Update internal states to enable starting over with the next iteration"""

        self.is_first_iteration = False

        # update image/frame history
        self.prev_gray_blurred = gray_blurred



    def _process(self, gray: GrayImage, gray_blurred: GrayImage, debug_info: DebugInfo | None = None) -> bool:
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


        """DEV"""


        # get CHANGES MASK from the diff image
        _, changes_mask, _ = VisionUtils.get_contours_by_threshold(diff)

        # update ACTIVITY MAP
        self.activity_map, current_activity_map = VisionUtils.update_activity_map(
            changes_mask,
            self.activity_map,
            self.settings.motion_detector.activity_map.cell_size,
            alpha=(self.settings.motion_detector.activity_map.alpha_warmup
                   if self.is_in_warmup else self.settings.motion_detector.activity_map.alpha)
        )
        assert self.activity_map is not None

        # get and store the working map size
        if self.map_size_wh is None:
            self.map_size_wh = ImageUtils.get_image_size_wh(self.activity_map)

        # ensure that map and image sizes are valid from this point on!
        assert self.map_size_wh is not None
        assert self.image_size_wh is not None

        # update brightness map
        self.brightness_float_map, current_brightness_float_map = VisionUtils.update_brightness_map(
            gray_blurred,
            self.brightness_float_map,
            self.settings.motion_detector.activity_map.cell_size,
            alpha=0.1
        )

        # compute float brightness difference (based on maps)
        # brightness_float_map_diff = ((current_brightness_float_map - self.brightness_map) + 1.0) / 2.0
        brightness_float_map_diff = np.subtract(current_brightness_float_map, self.brightness_float_map)

        # remove noise
        relevant_brightness_change_threshold = 0.005  # TODO: Parameter
        brightness_float_map_diff[np.fabs(brightness_float_map_diff) < relevant_brightness_change_threshold] = 0.0

        # blow up brightness areas a bit (keep negative and positive values and combine the result)
        default_kernel = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))
        brightness_float_map_diff = VisionUtils.signed_dilate(brightness_float_map_diff, default_kernel, iterations=1)

        # close small holes
        brightness_float_map_diff = VisionUtils.signed_morphologyEx(
            brightness_float_map_diff,
            cv2.MORPH_CLOSE,
            np.ones((3, 3), np.float32)
        )

        # get abs float map
        brightness_float_map_absdiff = np.fabs(brightness_float_map_diff)

        # create initial brightness change map
        if self.brightness_change_map is None:
            self.brightness_change_map = np.zeros_like(brightness_float_map_diff, dtype=np.uint8)

        # darken the brightness change map and add new brightness blobs
        if not self.is_in_warmup:
            # darken brightness_change_map
            # 255 = 3 * 5 * 17 = 3 * 85 = 15 * 17
            darken_step = int(255 / (5 * 3))    # TODO: Parameter: duration=3s (@5 fps) -> 5 * 3
            self.brightness_change_map[self.brightness_change_map >= darken_step] -= darken_step
            self.brightness_change_map[self.brightness_change_map < darken_step] = 0

            # get areas on the brightness abs-diff float-map
            bla = ImageUtils.float_image_to_gray(brightness_float_map_absdiff * 255.0)
            num_labels, labels, stats, centroids = (
                cv2.connectedComponentsWithStats(bla))

            # add newly detected brightness blobs to self.brightness_change_map
            for i, stat in enumerate(stats[1:]):
                area = stat[cv2.CC_STAT_AREA]
                if area >= 6**2:  # TODO: Param
                    current_label_no = i + 1
                    x = stat[cv2.CC_STAT_LEFT]
                    y = stat[cv2.CC_STAT_TOP]
                    w = stat[cv2.CC_STAT_WIDTH]
                    h = stat[cv2.CC_STAT_HEIGHT]

                    # copy float absdiff bbx to blob
                    brightness_blob = brightness_float_map_absdiff[y:y + h, x:x + w]
                    # set all cells to zero, which do not belong to the current label
                    brightness_blob[labels[y:y + h, x:x + w] != current_label_no] = 0.0

                    # insert non-zero blob cells into self.brightness_change_map
                    brightness_diff_image_blob = self.brightness_change_map[y:y + h, x:x + w]
                    brightness_diff_image_blob[brightness_blob > 0] = 255
                    self.brightness_change_map[y:y + h, x:x + w] = brightness_diff_image_blob



        # compute activity difference (current vs. accumulated history)
        activity_float_map_diff = np.clip(np.subtract(current_activity_map, self.activity_map), 0.0, 1.0)
        activity_map_diff = ImageUtils.float_image_to_gray(activity_float_map_diff)
        activity_map_diff[activity_map_diff < 10] = 0  # ignore small differences TODO: Parameter

        # find connected areas in the activity-map difference
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(activity_map_diff, connectivity=8)

        # filter probably irrelevant areas (by bbx shape and area size)
        candidates = []
        for stat in stats[1:, :]:  # row 0 is the background -> skip
            # filter by area size ...
            if stat[cv2.CC_STAT_AREA] >= 2: # TODO: size param
                # ... and by bbx shape
                shape_angle = math.atan2(stat[cv2.CC_STAT_HEIGHT], stat[cv2.CC_STAT_WIDTH])
                if self.min_shape_angle < shape_angle < self.max_shape_angle:
                    # accept area as object detection candidate
                    candidates.append(stat)

        # upscale the map to the full image size and copy the candidate areas to a new image (activity_candidate_image)
        activity_diff_image = ImageUtils.rescale(activity_map_diff, self.image_size_wh)#, interpolation_method=cv2.INTER_NEAREST) # upscale diff map to image
        activity_candidate_image = np.zeros_like(activity_diff_image, dtype=np.uint8)   # new image to copy to
        activity_map_cell_size = self.settings.motion_detector.activity_map.cell_size   # map cell size
        for candidate in candidates:
            # upscale bbx to full image resolution
            bbx_x = candidate[cv2.CC_STAT_LEFT] * activity_map_cell_size[0]
            bbx_y = candidate[cv2.CC_STAT_TOP] * activity_map_cell_size[1]
            bbx_w = candidate[cv2.CC_STAT_WIDTH] * activity_map_cell_size[0]
            bbx_h = candidate[cv2.CC_STAT_HEIGHT] * activity_map_cell_size[1]
            # copy candidate roi
            roi = activity_diff_image[bbx_y:bbx_y + bbx_h, bbx_x:bbx_x + bbx_w] # FIXME: may contain overlapping regions! Use labels! => re-work and make it similar to brightness blob handling (probably no need for candidate list!)
            activity_candidate_image[bbx_y:bbx_y + bbx_h, bbx_x:bbx_x + bbx_w] = roi

        # -> ignore activity where relevant brightness changes have been detected
        brightness_change_image = ImageUtils.rescale(self.brightness_change_map, self.image_size_wh,
                                                     interpolation_method=cv2.INTER_NEAREST)
        activity_candidate_image[brightness_change_image > 0] = 0

        # generate contours from difference
        contours, *_ = VisionUtils.get_contours_by_threshold(activity_candidate_image, threshold_value=50)  # TODO: Parameter
        big_contours = VisionUtils.filter_contours_by_area(contours=contours, min_area=200, remove_filtered_contours=True)  # TODO: Parameter
        extreme_contours = VisionUtils.filter_contours_by_area(contours=big_contours, min_area=51200)  # TODO: Parameter

        brightness_contours, *_ = VisionUtils.get_contours_by_threshold(brightness_change_image, threshold_value=0)


        """END"""

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


        """DEBUG"""
        # draw DEBUG IMAGE, if enabled
        if self.debug_view is not None:

            # draw the activity maps
            short_activity_map = current_activity_map
            short_activity_image = ImageUtils.mix_in_binary_image(gray,
                                                                  np.float32(short_activity_map),
                                                                  (0.0, 0.5, 1.0))

            # print min/max info (short activity)
            image_w, image_h = ImageUtils.get_image_size_wh(short_activity_image)
            ImageUtils.draw_text(
                            short_activity_image,
                            f"Min: {np.min(short_activity_map):.4f} | Max: {np.max(short_activity_map):.4f}",
                            (image_w - 20, image_h - 20),
                            anchor=ImageUtils.TextAnchor.BOTTOM_RIGHT, font_scale=0.5)

            # draw long-term activity map into gray image (with auto-scaled intensity)
            long_activity_map = self.activity_map
            long_activity_image = ImageUtils.mix_in_binary_image(
                                                    gray,
                                                    np.float32(long_activity_map) / max(np.max(long_activity_map), 0.1),
                                                    (0.0, 1.0, 0.0))

            # print min/max info (long activity)
            image_w, image_h = ImageUtils.get_image_size_wh(long_activity_image)
            _, text_height = ImageUtils.draw_text(long_activity_image,
                                  f"Min: {np.min(long_activity_map):.4f} | Max: {np.max(long_activity_map):.4f}",
                                  (image_w - 20, image_h - 20),
                                  anchor=ImageUtils.TextAnchor.BOTTOM_RIGHT, font_scale=0.5)

            # print warmup info
            if self.is_in_warmup:
                ImageUtils.draw_text(long_activity_image, "WARMUP",
                                     (image_w - 20, image_h - text_height - 20),
                                     anchor=ImageUtils.TextAnchor.BOTTOM_RIGHT, font_scale=0.5,
                                     text_color_bgr=(0, 0, 255))

            # print debug info
            if self.settings.debug and debug_info is not None and debug_info.is_dummy_camera:
                ImageUtils.draw_text(long_activity_image,
                                     [
                                         f"State: {self.detection_state_machine.get_state().name}",
                                         (f"Frame: {debug_info.dummy_video_frame + 1} / {debug_info.dummy_video_num_frames}"
                                          f" ({'step' if debug_info.is_replay_paused else ' >> '})")
                                      ],
                                     (10, image_h - 15),
                                     anchor=ImageUtils.TextAnchor.BOTTOM_LEFT, font_scale=0.4,
                                     text_color_bgr=(255, 255, 255))
                cv2.rectangle(long_activity_image, pt1=(9, image_h - 11 ), pt2=(111, image_h - 7),
                              color=(255, 255, 255), thickness=1)
                cv2.rectangle(long_activity_image, pt1=(10, image_h - 10),
                              pt2=(10 + int(100 * debug_info.dummy_video_frame / (debug_info.dummy_video_num_frames - 1)), image_h - 8),
                              color=(0, 0, 255), thickness=-1)


            activity_mix_image = DebugUtils.draw_activity_mix_image(self.activity_map,
                                                                    gray_blurred,
                                                                    current_activity_map)

            float_map_upscale = ImageUtils.rescale(brightness_float_map_diff, self.image_size_wh, interpolation_method=cv2.INTER_NEAREST)
            for contour in brightness_contours:
                contour_image = VisionUtils.get_contour_image(float_map_upscale, contour)
                max_val = np.max(contour_image)
                min_val = np.min(contour_image)

                cv2.drawContours(activity_mix_image, [contour], -1, (255, 255, 0), 1)

                rect = VisionUtils.Rect.from_contour(contour)
                rect.draw(activity_mix_image, (255, 255,0), 1,
                          [f"({min_val:.03f}, {max_val:.03f})",
                                f"   -> {max(abs(min_val), max_val):.3f}"],
                          0.3, 1)


            all_contours = chain(contours, big_contours, extreme_contours)
            for i, contour in enumerate(all_contours):
                brightness_contour_image = VisionUtils.get_contour_image(float_map_upscale, contour)
                max_val = np.max(brightness_contour_image)
                min_val = np.min(brightness_contour_image)

                if i < len(contours):
                    color = (0, 255, 0)
                elif i < len(contours) + len(big_contours):
                    color = (0, 255, 255)
                else:
                    color = (0, 0, 255)

                cv2.drawContours(activity_mix_image, [contour], -1, color, 1)

                rect = VisionUtils.Rect.from_contour(contour)
                rect.draw(activity_mix_image, color, 1,
                          [f"b: {max(abs(min_val), max_val):.3f}"],
                          0.3, 1)


            #cv2.drawContours(activity_mix_image, contours, -1, (0, 255, 0), 1)
            #cv2.drawContours(activity_mix_image, big_contours, -1, (0, 255, 255), 1)
            #cv2.drawContours(activity_mix_image, extreme_contours, -1, (0, 0, 255), 1)

            # compute brightness and print info
            brightness = self.brightness_float_map.mean() * 100.0
            brightness_change = current_brightness_float_map.mean() * 100 - brightness
            ImageUtils.draw_text(activity_mix_image,
                                 [f"Brightness: {brightness:.2f}",f"Brightness change: {brightness_change: .2f}"],
                                 (image_w - 20, image_h - 20),
                                 anchor=ImageUtils.TextAnchor.BOTTOM_RIGHT,
                                 font_scale=0.5)

            # print current detection state
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


            # assign images to the debug view matrix elements
            self.debug_view.set_image("top_left", changes_mask)

            # self.debug_view.set_image("bottom_right", ImageUtils.rescale(np.clip(brightness_float_map_diff * 10, 0.0, 1.0), ImageUtils.get_image_size_wh(gray)))
            self.debug_view.set_image("bottom_right", ImageUtils.rescale(self.brightness_change_map, ImageUtils.get_image_size_wh(gray), interpolation_method=cv2.INTER_NEAREST))

            self.debug_view.set_image("top_right", activity_mix_image)
            #self.debug_view.set_image("top_left", short_activity_image)
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
            #print(f"diff_time: {diff_time}, diff_val: {diff_val}")
            if diff_time > 0:
                diff_rate = diff_val / diff_time
                #print(f"diff_rate: {diff_rate} > -0.005 and {newest_element[1]} < 0.02")
                if diff_rate > -0.005 and newest_element[1] < 0.02:   # TODO: Parameters
                    self.warmup_finished = True
                    return True
                else:
                    return False
            else:
                return False

