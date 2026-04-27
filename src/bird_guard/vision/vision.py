import cv2
import numpy as np

from bird_guard.camera.camera import Frame
from bird_guard.vision.utils.debug_utils import DebugViewer
from bird_guard.vision.utils.image_utils import *
from bird_guard.vision.utils.vision_utils import VisionUtils, Contours
from bird_guard.vision.vision_config import ModuleConfig_Vision

# ===============
# Motion Detector
# ===============
class MotionDetector:

    def __init__(self, settings: ModuleConfig_Vision, threshold: int = 400):
        self.settings = settings

        self.prev_gray = None
        self.threshold = threshold

        self.debug_view = None

        self.diff_method: VisionUtils.DetectionMode = VisionUtils.DetectionMode.BG_REM   # TODO: Make configurable or remove
        self.background_subtractor: cv2.BackgroundSubtractor | None = None

        self.activity_map: FloatImage | None = None

        self.is_first_iteration: bool = True
        self.warmup_counter: int = 1

        self._init_components()

    def _init_components(self):
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
        self.debug_view.viewer_config.set_image_matrix_dimensions(1920, 1080)

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
        if self.prev_gray is None and self.diff_method is VisionUtils.DetectionMode.IM_DIFF:
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

        # get current frame as grayscale image
        gray = VisionUtils.get_frame_as_gray_image(frame)

        # BLUR gray image to reduce noise
        gray_blurred = VisionUtils.get_blurred_gray_image(gray, frame.type)

        return gray, gray_blurred


    def _post_process(self, gray_blurred: GrayImage):
        """Update internal states to enable starting over with the next iteration"""

        self.is_first_iteration = False

        # update image/frame history
        self.prev_gray = gray_blurred



    def _process(self, gray: GrayImage, gray_blurred: GrayImage) -> bool:
        """The actual image processing pipeline"""

        # set gray images to debug viewer
        if self.debug_view: self.debug_view.set_image("top_left", gray)
        if self.debug_view: self.debug_view.set_image("top_right", gray_blurred)

        # get DIFF image by selected method
        match self.diff_method:
            case VisionUtils.DetectionMode.IM_DIFF:
                diff = VisionUtils.get_image_sequence_difference_simple(gray_blurred, self.prev_gray)
            case VisionUtils.DetectionMode.BG_REM:
                diff = VisionUtils.get_image_sequence_difference_MOG2(gray_blurred, self.background_subtractor)
            case _:
                raise NotImplementedError(f"Diff image generation method {self.diff_method} is not implemented!")

        if self.debug_view: self.debug_view.set_image("bottom_left", diff)


        # get CONTOURS of changed areas (and the corresponding mask and the number of changed pixels)
        contours, changes_mask, changed_pixels = VisionUtils.get_contours_by_threshold(diff)
        big_contours = VisionUtils.filter_big_contours(contours)

        # update ACTIVITY MAP
        self.activity_map, current_activity_map = VisionUtils.update_activity_map(
            changes_mask,
            self.activity_map,
            self.settings.motion_detector.activity_map.cell_size,
            alpha=self.settings.motion_detector.activity_map.alpha
        )


        # draw DEBUG IMAGE, if enabled
        if self.debug_view is not None:
            # update debug image in debug viewer
            if self.debug_view: self.debug_view.set_image("bottom_right", self.draw_debug_image(gray, contours, big_contours, current_activity_map))
            self.debug_view.set_image("top_right", self.draw_activity_mix_image(gray_blurred, current_activity_map))

            # test: draw the activity maps
            short_threshold = 0.5
            short_activity_map = (current_activity_map > short_threshold)
            short_activity_image = ImageUtils.mix_in_binary_image(gray, np.float32(short_activity_map), (0.0, 0.5, 1.0))

            long_threshold_min = 0.0
            long_threshold_max = 0.1
            long_activity_map = (long_threshold_min < self.activity_map) & (self.activity_map < long_threshold_max)
            long_activity_image = ImageUtils.mix_in_binary_image(gray, np.float32(long_activity_map), (0.0, 1.0, 0.0))
            long_activity_image = cv2.subtract(cv2.add(long_activity_image, short_activity_image), ImageUtils.gray_image_to_color(gray))
            image_w, image_h = ImageUtils.get_image_size_wh(long_activity_image)
            ImageUtils.draw_text(long_activity_image,
                                 [f"short threshold > {short_threshold}", f"{long_threshold_max} > long threshold > {long_threshold_min}"],
                                 (image_w - 20, image_h - 20),
                                 anchor=ImageUtils.TextAnchor.BOTTOM_RIGHT, font_scale=0.5)


            self.debug_view.set_image("top_left", short_activity_image)
            self.debug_view.set_image("bottom_left", long_activity_image)


        # update images in the DEBUG VIEW
        if self.debug_view:
            self.debug_view.update_debug_view()


        # detect MOVEMENT (primitive)
        if self.warmup_counter > 0:
            self.warmup_counter -= 1
            movement_detected = False
        else:
            movement_detected = (changed_pixels > self.threshold)


        return movement_detected


    def draw_activity_mix_image(self, gray_base_image: GrayImage, current_activity_map: FloatImage) -> BGRImage:
        # subtract blue for current activity -> yellow
        mix_color = ImageUtils.mix_images_to_color(
            ImageUtils.gray_image_to_color(gray_base_image),
            (1.0,) * 3,
            ImageUtils.gray_image_to_color(ImageUtils.float_image_to_gray(current_activity_map)),
            (-1.0, 0.0, 0.0)
        )

        # subtract cyan for activity map -> red
        mix_color = ImageUtils.mix_images_to_color(
            mix_color,
            (1.0,) * 3,
            ImageUtils.gray_image_to_color(ImageUtils.float_image_to_gray(self.activity_map)),
            (-1.0, -1.0, 0.0)
        )

        return mix_color


    def draw_debug_image(self, gray_base_image: GrayImage, all_contours: Contours, big_contours: Contours, current_activity_map: FloatImage) -> BGRImage:
        # convert to color image for debug drawing in colors
        debug_img = ImageUtils.gray_image_to_color(gray_base_image)


        #test_binary_image = (self.activity_map > 0.05)     # Äste hinterlassen spuren bei 0.005; 0.1 zu viel! 0.05 vielleicht noch ok
        test_binary_image = (current_activity_map > 0.75)
        debug_img = ImageUtils.mix_in_binary_image(debug_img, np.float32(test_binary_image), (1.0, 1.0, 0.0))


        # draw small and big contours
        cv2.drawContours(debug_img, all_contours, -1, (0, 0, 255), 2)
        cv2.drawContours(debug_img, big_contours, -1, (0, 255, 0), 2)

        # get and draw bounding boxes of the big contours
        for contour in big_contours:
            bbx = VisionUtils.Rect.from_contour(contour)
            bbx.draw(debug_img, (0, 255, 255), 2)

        return debug_img
