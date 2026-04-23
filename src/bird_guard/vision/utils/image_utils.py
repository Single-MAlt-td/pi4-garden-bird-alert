"""
Image utilities.
"""

import numpy as np
import cv2

import numpy.typing as npt

# define aliases for image array types
BGRImage = npt.NDArray[np.uint8]    # color image
GrayImage = npt.NDArray[np.uint8]   # grayscale image
Image = BGRImage | GrayImage        # color or grayscale image


# ==========
# ImageUtils
# ==========
class ImageUtils:
    """
    Helper class providing several useful functions for images
    """
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
    def rescale(image: Image, new_size_wh: tuple[int, int], keep_aspect: bool = True) -> Image:
        """
        Rescale the given image (with or without keep aspect).
        If needed, padding is performed with black color.

        Args:
            image: Input image
            new_size_wh: (width, height) specifying the new image size
            keep_aspect: If true, the aspect ratio is not changed and if needed, padding is performed with black color

        Returns:
            The rescaled image with size new_size_wh
        """

        # get current image size
        im_height, im_width, *im_channels = image.shape

        # if aspect ratio is irrelevant, just rescale and return the result
        if not keep_aspect:
            if new_size_wh < (im_width, im_height):
                return cv2.resize(image, new_size_wh, interpolation=cv2.INTER_AREA)
            else:
                return cv2.resize(image, new_size_wh, interpolation=cv2.INTER_LINEAR)

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



