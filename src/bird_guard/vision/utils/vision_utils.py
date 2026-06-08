from enum import Enum
from typing import Sequence
from dataclasses import dataclass

import cv2
import numpy as np

import numpy.typing as npt

from bird_guard.camera.camera import Frame
from bird_guard.vision.utils.image_utils import ImageUtils, Image, GrayImage, BGRImage, FloatImage, BGRColor

# define aliases for vision types
Contour = np.ndarray                    # individual contour
Contours = list[np.ndarray]             # const list of contours
BGRMap = npt.NDArray[np.uint8]          # map with 3 color channels
GrayMap = npt.NDArray[np.uint8]         # single channel map
FloatMap = npt.NDArray[np.float32]      # float32 map
BinaryMap = npt.NDArray[np.bool_]       # bool map


# ===========
# VisionUtils
# ===========
class VisionUtils:

    class DetectionMode(Enum):
        IM_DIFF = 0     # use image difference
        BG_REM = 1      # use more advanced background removal method

    @dataclass
    class Rect:
        x: int
        y: int
        w: int
        h: int

        @staticmethod
        def from_contour(contour: Contour) -> "VisionUtils.Rect":
            return VisionUtils.Rect(*cv2.boundingRect(contour))

        def draw(self,
                 image: BGRImage,
                 color_bgr: BGRColor,
                 line_width: int = 1,
                 text: str | list[str] | None = None,
                 font_scale: float = 1.0,
                 thickness: int = 1):

            cv2.rectangle(image, (self.x, self.y), (self.x + self.w, self.y + self.h), color_bgr, line_width)

            if text is not None:
                text_w, text_h, *_ = ImageUtils.get_multiline_text_size(text, font_scale, thickness)

                needed_height = text_h + 1 + line_width
                if self.y > needed_height:
                    ImageUtils.draw_text(image, text, (self.x, self.y - 1 - line_width), color_bgr, ImageUtils.TextAnchor.BOTTOM_LEFT,
                                         font_scale, thickness)
                else:
                    ImageUtils.draw_text(image, text, (self.x, self.y + self.h + 1 + line_width), color_bgr, ImageUtils.TextAnchor.TOP_LEFT,
                                         font_scale, thickness)

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
                gray = ImageUtils.yuv420_image_to_gray(frame.data)
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

        return list(contours), thresh_mask_dilated, changed_pixels

    @staticmethod
    def filter_contours_by_area(contours: Contours, min_area: int = 500, remove_filtered_contours: bool = False) -> Contours:
        big_contours = []
        filtered_indices = [i for i, contour in enumerate(contours) if cv2.contourArea(contour) > min_area]
        for i in filtered_indices[::-1]:
            big_contours.append(contours[i])
            if remove_filtered_contours:
                 del contours[i]

        return big_contours

    @staticmethod
    def get_clustered_image_size_wh(image: Image, cell_size_wh: tuple[int, int]):
        # check if image size is divisible by cell size
        w, h = ImageUtils.get_image_size_wh(image)
        if (w % cell_size_wh[0] != 0) or (h % cell_size_wh[1] != 0):
            raise ValueError(f"Image of size ({w}x{h}) is not divisible into cells of size ({cell_size_wh[0]}x{cell_size_wh[1]})")

        # determine clustered image size
        clustered_image_size_wh = (w // cell_size_wh[0], h // cell_size_wh[1])

        return clustered_image_size_wh


    @staticmethod
    def update_activity_map(changes_mask: GrayImage,
                            activity_map: FloatImage | None,
                            cell_size_wh: tuple[int, int] = (16,16),
                            alpha: float = 0.02) -> tuple[FloatImage, FloatImage]:
        # convert changes mask to float mask, where 255 equals 1.0 and 0 otherwise
        # (-> will ignore shadows, if activated for MOG2 method!)
        float_mask = np.float32(changes_mask == 255)

        # get activity map size
        activity_map_size_wh = VisionUtils.get_clustered_image_size_wh(changes_mask, cell_size_wh)

        # generate current activity map from float_mask by scaling it down (using INTER_AREA -> automatically sets the average value of the original cell!)
        current_activity_map = cv2.resize(float_mask, activity_map_size_wh, interpolation=cv2.INTER_AREA)

        if activity_map is None:
            activity_map = current_activity_map.copy()
        else:
            activity_map = cv2.accumulateWeighted(current_activity_map, activity_map, alpha)    # dst = (1 - alpha) * dst + alpha * src

        return activity_map, current_activity_map


    @staticmethod
    def update_brightness_map(gray_image: GrayImage,
                              brightness_map: FloatImage | None,
                              cell_size_wh: tuple[int, int] = (16, 16),
                              alpha: float = 0.02) -> tuple[FloatImage, FloatImage]:

        clustered_size_wh = VisionUtils.get_clustered_image_size_wh(gray_image, cell_size_wh)

        float_image = ImageUtils.gray_image_to_float(gray_image)
        current_brightness_map = cv2.resize(float_image, clustered_size_wh, interpolation=cv2.INTER_AREA)

        if brightness_map is None:
            brightness_map = current_brightness_map.copy()
        else:
            brightness_map = cv2.accumulateWeighted(current_brightness_map, brightness_map, alpha)

        return brightness_map, current_brightness_map

    @staticmethod
    def get_contour_image(base_image: Image, contour: Contour) -> Image:
        """Return base_image, where everything outside the contour is zero"""
        mask = np.zeros_like(base_image, dtype=np.uint8)
        cv2.fillPoly(mask, pts=[contour], color=255)
        contour_image = cv2.bitwise_and(base_image, base_image, mask=mask)
        return contour_image

    @staticmethod
    def signed_dilate(float_image: FloatImage, kernel: cv2.typing.MatLike, iterations: int = 1) -> FloatImage:
        pos = cv2.dilate(np.maximum(float_image, 0), kernel, iterations=iterations)
        neg = cv2.dilate(np.maximum(-float_image, 0), kernel, iterations=iterations)
        return pos - neg

    @staticmethod
    def signed_morphologyEx(float_image: FloatImage, operation: int, kernel: cv2.typing.MatLike) -> FloatImage:
        pos = cv2.morphologyEx(np.maximum(float_image, 0), operation, kernel)
        neg = cv2.morphologyEx(np.maximum(-float_image, 0), operation, kernel)
        return pos - neg
