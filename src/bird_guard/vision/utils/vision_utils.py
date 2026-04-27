from enum import Enum
from typing import Sequence
from dataclasses import dataclass

import cv2
import numpy as np

from bird_guard.camera.camera import Frame
from bird_guard.vision.utils.image_utils import ImageUtils, GrayImage, BGRImage, FloatImage, BGRColor

# define aliases for vision types
Contour = np.ndarray                # individual contour
Contours = Sequence[np.ndarray]     # const list of contours


# ===========
# VisionUtils
# ===========
class VisionUtils:

    class DetectionMode(Enum):
        IM_DIFF = 0
        BG_REM = 1

    @dataclass
    class Rect:
        x: int
        y: int
        w: int
        h: int

        @staticmethod
        def from_contour(contour: Contour) -> "VisionUtils.Rect":
            return VisionUtils.Rect(*cv2.boundingRect(contour))

        def draw(self, image: BGRImage, color_bgr: BGRColor, line_width: int = 1):
            cv2.rectangle(image, (self.x, self.y), (self.x + self.w, self.y + self.h), color_bgr, line_width)

    # --------

    @staticmethod
    def get_frame_as_gray_image(frame: Frame) -> GrayImage:
        # convert color image to gray, if needed
        match frame.type:
            case Frame.FrameType.COLOR:
                gray = ImageUtils.color_image_to_gray(frame.data)
            case Frame.FrameType.GRAY:
                gray = frame.data
            case Frame.FrameType.LORES:
                gray = ImageUtils.get_gray_image_from_yuv420(frame.data)
            case _:
                raise NotImplementedError("Frame type not yet implemented.")

        return gray

    @staticmethod
    def get_blurred_gray_image(gray: GrayImage, kernel_size: int | Frame.FrameType) -> GrayImage:
        """Blur image (to reduce noise)"""
        # TODO: make kernel sizes configurable (also the auto-selected ones!)
        if isinstance(kernel_size, int):
            # kernel_size argument is a number -> create kernel_shape
            kernel_shape = (kernel_size, kernel_size)
        elif isinstance(kernel_size, Frame.FrameType):
            # kernel_size argument is frame type -> auto select kernel_shape by type
            frame_type: Frame.FrameType = kernel_size
            match frame_type:
                case Frame.FrameType.GRAY:
                    kernel_shape = (21, 21)
                case Frame.FrameType.LORES:
                    kernel_shape = (11, 11)
                case _:
                    raise NotImplementedError("Frame type not supported.")
        else:
            raise TypeError("Given kernel_size argument has invalid type")

        # blur image
        return cv2.GaussianBlur(gray, kernel_shape, 0)


    @staticmethod
    def get_image_sequence_difference_simple(current_image: GrayImage, previous_image: GrayImage) -> GrayImage:
        return cv2.absdiff(previous_image, current_image)

    @staticmethod
    def get_image_sequence_difference_MOG2(current_image: GrayImage, bgs: cv2.BackgroundSubtractor) -> GrayImage:
        return bgs.apply(current_image, learningRate=0.005)

    @staticmethod
    def get_contours_by_threshold(diff_image: GrayImage, threshold_value: int = 25) -> tuple[Contours, GrayImage, int]:
        # detect significant differences via threshold
        _, thresh_mask = cv2.threshold(diff_image, threshold_value, 255, cv2.THRESH_BINARY)  # THRESH_BINARY: set everything above threshold_value to maxval, 0 otherwise

        # dilate to connect contours across small gaps
        # (dilate extends white pixels according to the default_kernel size/shape/anchor)
        default_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        thresh_mask_dilated = cv2.dilate(thresh_mask, default_kernel, iterations=2)

        # count number of detection pixels
        changed_pixels = cv2.countNonZero(thresh_mask_dilated)

        # find cluster contours
        contours, _ = cv2.findContours(thresh_mask_dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        return contours, thresh_mask_dilated, changed_pixels

    @staticmethod
    def filter_big_contours(contours: Contours, min_area: int = 500) -> Contours:
        big_contours = [contour for contour in contours if cv2.contourArea(contour) > min_area]
        return big_contours

    @staticmethod
    def update_activity_map(changes_mask: GrayImage,
                            activity_map: FloatImage | None,
                            cell_size_wh: tuple[int, int] = (16,16),
                            alpha: float = 0.02) -> tuple[FloatImage, FloatImage]:
        # convert changes mask to float mask, where 255 equals 1.0 and 0 otherwise
        # (-> will ignore shadows, if activated for MOG2 method!)
        float_mask = np.float32(changes_mask == 255)

        # check if mask is divisible by cell size
        h, w = changes_mask.shape[:2]
        if (w % cell_size_wh[0] != 0) or (h % cell_size_wh[1] != 0):
            raise ValueError(f"Image of size ({w}x{h}) is not divisible into cells of size ({cell_size_wh[0]}x{cell_size_wh[1]})")

        # determine activity map size
        activity_map_size_wh = (w // cell_size_wh[0], h // cell_size_wh[1])

        # generate current activity map from float_mask by scaling it down (using INTER_AREA -> automatically sets the average value of the original cell!)
        current_activity_map = cv2.resize(float_mask, activity_map_size_wh, interpolation=cv2.INTER_AREA)

        if activity_map is None:
            activity_map = current_activity_map.copy()
        else:
            activity_map = cv2.accumulateWeighted(current_activity_map, activity_map, alpha)

        return activity_map, current_activity_map
