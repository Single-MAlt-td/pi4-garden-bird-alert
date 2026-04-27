"""
Image utilities.
"""
from argparse import ArgumentTypeError
from enum import Enum

import numpy as np
import cv2

import numpy.typing as npt

# define aliases for image array types
BGRColor = tuple[int, int, int]             # color definition in BGR format (int from 0 to 255)
BGRFloatColor = tuple[float, float, float]  # color definition in BGR format (float from 0.0 to 1.0)
BGRImage = npt.NDArray[np.uint8]            # color image
GrayImage = npt.NDArray[np.uint8]           # grayscale image
YUV420Image = npt.NDArray[np.uint8]         # image in YUV420 format
FloatImage = npt.NDArray[np.float32]        # grayscale image with values from 0.0 to 1.0 as float32
BinaryImage = npt.NDArray[np.bool_]         # binary/bool image (Note: 'np.bool' is not yet supported for older numpy version like 1.24.2, but 'np.bool_' is)
Image = BGRImage | GrayImage                # color or grayscale image


# ==========
# ImageUtils
# ==========
class ImageUtils:
    """
    Helper class providing several useful functions for images
    """

    class TextAnchor(Enum):
        """Anchor points for placing text"""
        TOP_LEFT = 0
        TOP_CENTER = 1
        TOP_RIGHT = 2
        CENTER_LEFT   = 10
        CENTER_CENTER = 11
        CENTER_RIGHT  = 12
        BOTTOM_LEFT = 20
        BOTTOM_CENTER = 21
        BOTTOM_RIGHT = 22

    # ----------

    def __init__(self):
        pass

    @staticmethod
    def get_blank_image(image_size_wh: tuple[int, int], color_bgr: tuple[int, ...] = (0, 0, 0)) -> Image:
        """
        Get a blank image filled with the given color

        Args:
            image_size_wh: (width, height) specifying the image size
            color_bgr: tuple of dynamic length specifying the color and color dimension (e.g. black is (0,) for grayscale, (0,0,0) for BGR)

        Returns:
             Image array (uint8)
        """
        return np.full((*reversed(image_size_wh), len(color_bgr)), color_bgr, dtype=np.uint8)

    @staticmethod
    def get_color_channels(image: Image) -> int:
        """Return the number of color channels"""
        _, _, *image_channels = image.shape
        if image.ndim == 3:
            return image.shape[2]
        elif image.ndim == 2:
            return 1
        else:
            raise ValueError(f"Unexpected number of image matrix dimensions: {image.ndim}")

    @staticmethod
    def get_image_size_wh(image: Image) -> tuple[int, int]:
        return image.shape[1::-1]

    @staticmethod
    def rescale(image: Image,
                new_size_wh: tuple[int, int],
                keep_aspect: bool = True,
                interpolation_method: int | None = None
                ) -> Image:
        """
        Rescale the given image (with or without keep aspect).
        If needed, padding is performed with black color.

        Args:
            image: Input image
            new_size_wh: (width, height) specifying the new image size
            keep_aspect: If true, the aspect ratio is not changed and if needed, padding is performed with black color
            interpolation_method: The interpolation method to use (if None, the method is auto-selected)

        Returns:
            The rescaled image with size new_size_wh
        """

        # get current image size
        im_height, im_width, *im_channels = image.shape

        # if aspect ratio is irrelevant, just rescale and return the result
        if not keep_aspect:
            if new_size_wh < (im_width, im_height):
                # downscale
                if interpolation_method is None:
                    interpolation_method = cv2.INTER_AREA
                return cv2.resize(image, new_size_wh, interpolation=interpolation_method)
            else:
                # upscale
                if interpolation_method is None:
                    interpolation_method = cv2.INTER_LINEAR
                return cv2.resize(image, new_size_wh, interpolation=interpolation_method)

        # determine new image size
        scale_width = new_size_wh[0] / im_width
        scale_height = new_size_wh[1] / im_height
        scale = min(scale_width, scale_height)

        scaled_image_width = int(im_width * scale)
        scaled_image_height = int(im_height * scale)

        # rescale image
        is_upscale = (scale > 1.0)
        if is_upscale:
            scaled_image = cv2.resize(image, (scaled_image_width, scaled_image_height), interpolation=cv2.INTER_LINEAR)
        else:
            scaled_image = cv2.resize(image, (scaled_image_width, scaled_image_height), interpolation=cv2.INTER_AREA)

        # pad image with black borders (if needed)
        if (scaled_image_width, scaled_image_height) != new_size_wh:

            # determine number of color channels
            if not im_channels:
                num_channels = 1
            else:
                num_channels = im_channels[0]

            add_letterbox = (scale_width < scale_height)
            if add_letterbox:
                # add letterbox
                diff_height = new_size_wh[1] - scaled_image_height
                top_border_height = diff_height // 2
                bottom_border_height = diff_height - top_border_height
                padded_image = cv2.copyMakeBorder(scaled_image,
                                                  top_border_height, bottom_border_height, 0, 0,
                                                  cv2.BORDER_CONSTANT, value=(0,) * num_channels)
            else:
                # add pillarbox
                diff_width = new_size_wh[0] - scaled_image_width
                left_border_width = diff_width // 2
                right_border_width = diff_width - left_border_width
                padded_image = cv2.copyMakeBorder(scaled_image, 0, 0, left_border_width, right_border_width,
                                                  cv2.BORDER_CONSTANT, value=(0,) * num_channels)

            return padded_image

        else:
            return scaled_image

    @staticmethod
    def gray_image_to_color(gray: GrayImage) -> BGRImage:
        return cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)

    @staticmethod
    def color_image_to_gray(image: BGRImage) -> GrayImage:
        return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    @staticmethod
    def get_gray_image_from_yuv420(image: YUV420Image) -> GrayImage:
        if len(image.shape) != 2 or image.shape[0] % 3 != 0:
            raise ValueError("Expected a YUV420 image, but got something else!")

        height = (image.shape[0] * 2) // 3
        return image[:height, :]

    @staticmethod
    def is_float_image(float_image: FloatImage) -> bool:
        if np.issubdtype(float_image.dtype, np.floating):
            if np.all((0.0 <= float_image) & (float_image <= 1.0)):
                return True
        return False

    @staticmethod
    def is_gray_image(gray_image: GrayImage) -> bool:
        if ImageUtils.get_color_channels(gray_image) == 1:
            if np.issubdtype(gray_image.dtype, np.uint8):
                if np.all((0 <= gray_image) & (gray_image <= 255)):
                    return True
        return False

    @staticmethod
    def is_color_image(color_image: BGRImage) -> bool:
        if ImageUtils.get_color_channels(color_image) == 3:
            if np.issubdtype(color_image.dtype, np.uint8):
                if np.all((0 <= color_image) & (color_image <= 255)):
                    return True
        return False

    @staticmethod
    def float_image_to_gray(float_image: FloatImage) -> GrayImage:
        if ImageUtils.is_float_image(float_image):
            return np.uint8(float_image * 255)
        else:
            raise ArgumentTypeError(f"Image is not a float image!")

    @staticmethod
    def mix_images_to_color(base_image: BGRImage,
                            color_weight_base_bgr: tuple[float, float, float],
                            overlay_image: BGRImage,
                            color_weight_overlay_bgr: tuple[float, float, float]
                            ) -> BGRImage:

        # convert overlay image to color image and scale to base image size
        overlay_color = ImageUtils.rescale(overlay_image, ImageUtils.get_image_size_wh(base_image), interpolation_method=cv2.INTER_NEAREST)

        # mix color channels
        mix_result = base_image.copy()
        for channel in range(3):
            mix_result[..., channel] = cv2.addWeighted(mix_result[..., channel],
                                                        float(color_weight_base_bgr[channel]),
                                                        overlay_color[..., channel],
                                                        float(color_weight_overlay_bgr[channel]),
                                                        0)
        return mix_result


    @staticmethod
    def mix_in_binary_image(base_image: Image, binary_image: BinaryImage, color_bgr: BGRFloatColor) -> BGRImage:

        color_subtract = tuple([val - 1.0 for val in color_bgr])

        if ImageUtils.is_gray_image(base_image):
            base_color_image = ImageUtils.gray_image_to_color(base_image)
        elif ImageUtils.is_float_image(base_image):
            base_color_image = ImageUtils.gray_image_to_color(ImageUtils.float_image_to_gray(base_image))
        elif ImageUtils.is_color_image(base_image):
            base_color_image = base_image
        else:
            raise ValueError("Unsupported image type")

        mix_color = ImageUtils.mix_images_to_color(
            base_color_image,
            (1.0,) * 3,
            ImageUtils.gray_image_to_color(ImageUtils.float_image_to_gray(np.float32(binary_image))),
            color_subtract
        )
        return mix_color


    @staticmethod
    def draw_text(color_image: BGRImage,
                  text_list: str | list[str],
                  pos: tuple[int, int],
                  color: BGRColor = (255, 255, 255),
                  anchor: TextAnchor = TextAnchor.BOTTOM_LEFT,
                  font_scale: float = 1.0,
                  thickness: int = 1,
                  vertical_spacing: int = 5
                  ) -> tuple[int, int]:
        """
        Draw single- or multiline text to the given image.

        Args:
            color_image: The target image
            text_list: A text or list of texts to be drawn (each list element will be drawn in a new line)
            pos: Target pixel position of the text anchor
            color: Text color
            anchor: Defines which point on the bounding box of the text corresponds to the target pixel position (pos) (default: BOTTOM_LEFT)
            font_scale: Font scaling factor that acts on the font base size (default: 1.0)
            thickness: Integer line thickness of the font (default: 1)
            vertical_spacing: Vertical space between text lines in pixels (default: 5)

        Returns:
            Tuple (text_width, text_height), which contains the size of the generated text in pixels
        """

        # ensure text_list is a list
        if isinstance(text_list, str):
            text_list = [text_list]

        # determine sizes of individual text lines
        text_widths: list[int] = []
        text_heights: list[int] = []
        baselines: list[int] = []
        for text in text_list:
            # determine the bounding box size of the text
            (text_w, text_h), baseline = cv2.getTextSize(
                text,
                cv2.FONT_HERSHEY_SIMPLEX,
                font_scale,
                thickness,
            )
            text_widths.append(text_w)
            text_heights.append(text_h)
            baselines.append(baseline)

        # compute bounding box dimensions, which includes the entire text
        text_bbx_width = max(text_widths)
        text_bbx_height = sum(text_heights) + sum(baselines) + (len(text_list) - 1) * vertical_spacing

        # extract vertical and horizontal shift factors of the given anchor
        h_anchor_val = anchor.value % 10                    # first digit encodes horizontal shift factor
        v_anchor_val = (anchor.value - h_anchor_val) // 10  # second digit encodes vertical shift factor

        # compute the y-offset to the pos y-coordinate (respecting the given anchor) to correctly position the first text line
        y_bbx_offset = v_anchor_val * (text_bbx_height // 2) - text_heights[0]

        # draw all texts
        # --------------
        y_offset = 0 # iterative shift of the y_bbx_offset after each drawn text line
        for idx, text in enumerate(text_list):
            # get bounding box size of current text line
            text_w = text_widths[idx]
            text_h = text_heights[idx] + baselines[idx] # the text height is not the actual height, since it's missing the baseline component

            # compute needed shift in x-direction to match the desired horizontal anchor position
            x_shift = h_anchor_val * text_w // 2

            # apply the y-offset to the bbx offset to get the correct y-shift
            y_shift = y_bbx_offset - y_offset

            # draw the text
            cv2.putText(
                color_image,
                text,
                (pos[0] - x_shift, pos[1] - y_shift),
                cv2.FONT_HERSHEY_SIMPLEX,
                font_scale,
                color,
                thickness,
                cv2.LINE_AA,
            )

            # apply the y-shift to draw to the next line in the next iteration
            y_offset += text_h + vertical_spacing

        return text_bbx_width, text_bbx_height
