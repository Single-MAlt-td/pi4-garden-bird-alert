"""
Script to demonstrate (and visually check) the ImageUtils.draw_text function.
"""

import cv2
from bird_guard.vision.utils.image_utils import ImageUtils

# create black image
image = ImageUtils.get_blank_image((640, 360), (0, 0, 0))

# get image size
image_w, image_h = ImageUtils.get_image_size_wh(image)

# draw a vertical and horizontal line through the image center
cv2.line(image, (0, image_h //2), (image_w, image_h // 2), (0, 0, 255), 1)
cv2.line(image, (image_w // 2, 0), (image_w // 2, image_h), (0, 0, 255), 1)

# font settings
font_scale = 0.5
vertical_space = 10

# draw text
# ---------
# insert text at all image positions that correspond to the available anchor points, using the corresponding anchor point for easy positioning

# yellow: all four edges
ImageUtils.draw_text(image, ["Ag", "BCy"],
                     (0, 0), anchor=ImageUtils.TextAnchor.TOP_LEFT,
                     font_scale=font_scale, color=(0, 255, 255), vertical_spacing=vertical_space)

ImageUtils.draw_text(image, ["Ag", "BCy"],
                     (image_w, 0), anchor=ImageUtils.TextAnchor.TOP_RIGHT,
                     font_scale=font_scale, color=(0, 255, 255), vertical_spacing=vertical_space)

ImageUtils.draw_text(image, ["Ag", "BCy"],
                     (0, image_h), anchor=ImageUtils.TextAnchor.BOTTOM_LEFT,
                     font_scale=font_scale, color=(0, 255, 255), vertical_spacing=vertical_space)

ImageUtils.draw_text(image, ["Ag", "BCy"],
                     (image_w, image_h), anchor=ImageUtils.TextAnchor.BOTTOM_RIGHT,
                     font_scale=font_scale, color=(0, 255, 255), vertical_spacing=vertical_space)

# cyan: left/right center
ImageUtils.draw_text(image, ["Ag", "==", "BCy"],
                     (0, image_h // 2), anchor=ImageUtils.TextAnchor.CENTER_LEFT,
                     font_scale=font_scale, color=(255, 255, 0), vertical_spacing=vertical_space)

ImageUtils.draw_text(image, ["Ag", "==", "BCy"],
                     (image_w, image_h // 2), anchor=ImageUtils.TextAnchor.CENTER_RIGHT,
                     font_scale=font_scale, color=(255, 255, 0), vertical_spacing=vertical_space)

# cyan: center top/bottom
ImageUtils.draw_text(image, ["Ag", "||"],
                     (image_w // 2, 0), anchor=ImageUtils.TextAnchor.TOP_CENTER,
                     font_scale=font_scale, color=(255, 255, 0), vertical_spacing=vertical_space)

ImageUtils.draw_text(image, ["||", "Ag"],
                     (image_w // 2, image_h), anchor=ImageUtils.TextAnchor.BOTTOM_CENTER,
                     font_scale=font_scale, color=(255, 255, 0), vertical_spacing=vertical_space)

# yellow: center
ImageUtils.draw_text(image, ["--", "+", "---"],
                     (image_w // 2, image_h // 2), anchor=ImageUtils.TextAnchor.CENTER_CENTER,
                     font_scale=font_scale, color=(0, 255, 255), vertical_spacing=vertical_space)

# show image and keep window open until any key is pressed
cv2.imshow("Figure", image)
cv2.waitKey(0)
cv2.destroyAllWindows()
