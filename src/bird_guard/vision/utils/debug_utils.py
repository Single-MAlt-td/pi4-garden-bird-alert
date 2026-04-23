"""
Utilities for debugging.
"""

import copy
from dataclasses import dataclass, field

import numpy as np
import cv2

from bird_guard.vision.utils.image_utils import ImageUtils, Image

# ============
# Debug Viewer
# ============
class DebugViewer:
    """
    Helper class to handle the display of images in a window for debugging purposes.
    To show multiple images at once, a matrix size can be defined and individual images can be assigned to the matrix
    elements, which will automatically be displayed and scaled in the debug window.
    """

    @dataclass
    class DebugViewerConfiguration:
        """
        Configuration of the debug viewer
        """
        rows: int = 1       # image matrix rows
        cols: int = 1       # image matrix columns
        width: int = 1920   # width of the image matrix in px
        height: int = 1080  # height of the image matrix in px
        image_assignment: list[list[str]] = field(default_factory=lambda: [[""]])   # matrix element -> image name mapping

        def __post_init__(self):
            self._init_image_assignment_matrix()

        def _init_image_assignment_matrix(self):
            self.image_assignment = [["" for _ in range(self.cols)] for _ in range(self.rows)]

        def set_image_matrix_size(self, rows: int, cols: int):
            """
            Define the size of the image matrix for the debug window.
            A [rows x cols] matrix can display rows * cols many images at once.

            Args:
                rows: Number of images per row
                cols: Number of images per column

            Returns:

            """
            self.rows = rows
            self.cols = cols
            self._init_image_assignment_matrix()

        def set_image_matrix_dimensions(self, width: int, height: int):
            """Define the size of the entire image matrix in pixels"""
            self.width = width
            self.height = height

        def register_image(self, row: int, col: int, name: str):
            """
            Assign the matrix cell [row, col] to the image with the given name.

            Args:
                row: Image matrix row
                col: Image matrix column
                name: Image identifier name
            """
            if row in range(self.rows) and col in range(self.cols):
                self.image_assignment[row][col] = name
            else:
                raise IndexError(f"Element ({row}, {col}) exceeds matrix size")

    # ----------

    def __init__(self, viewer_config: DebugViewerConfiguration = DebugViewerConfiguration(rows=2, cols=2)):
        self.window_name: str = "Debug"
        self.image_storage: dict = {}
        self.viewer_config = viewer_config

    def __del__(self):
        self._hide_debug_window()

    def show_debug_window(self):
        """Open the debug window"""
        cv2.namedWindow(self.window_name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(self.window_name, (self.viewer_config.width, self.viewer_config.height))

    def _hide_debug_window(self):
        cv2.destroyWindow(self.window_name)

    def set_image(self, name: str, image: Image):
        """
        Store/Overwrite image under the given name in the internal image storage

        Args:
            name: Image identifier name
            image: Image to be stored under the identifier name
        """
        self.image_storage[name] = image

    def _get_image(self, name: str) -> Image or None:
        if name in self.image_storage:
            return self.image_storage[name]
        else:
            return None

    def update_debug_view(self):
        """Update the entire image matrix content of the debug window"""

        # determine the target image size of all sub-images
        sub_image_width = self.viewer_config.width // self.viewer_config.cols
        sub_image_height = self.viewer_config.height // self.viewer_config.rows

        # put images, which are referenced by the viewer_config image_assignment, to a 2D matrix (size defined by viewer_config)
        im_matrix_rows = []
        for row in range(self.viewer_config.rows):
            im_matrix_cols = []
            for col in range(self.viewer_config.cols):
                image_name = self.viewer_config.image_assignment[row][col]

                # if an image is assigned and set, get the image
                # (use a blank black image otherwise)
                if image_name:
                    image = self._get_image(image_name)
                    if image is not None:
                        im_channels = ImageUtils.get_color_channels(image)
                        if im_channels == 1:
                            image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
                        elif im_channels == 3:
                            pass
                        else:
                            raise ValueError(f"Expected image with 1 or 3 color channels, but got {im_channels[0]}")
                    else:
                        # no image set -> build black dummy image
                        image = ImageUtils.get_blank_image((sub_image_width, sub_image_height), (0, 0, 0))
                else:
                    # no image assigned -> build black dummy image
                    image = ImageUtils.get_blank_image((sub_image_width, sub_image_height), (0, 0, 0))

                im_matrix_cols.append(copy.copy(image))
            im_matrix_rows.append(im_matrix_cols)

        # rescale all matrix images to the best fitting size and build the image matrix (all images must have 3 color channels!)
        combined = np.vstack(
            [
                np.hstack(
                    [ImageUtils.rescale(image, (sub_image_width, sub_image_height)) for image in im_cols]
                ) for im_cols in im_matrix_rows
            ]
        )

        # show the final image matrix in the debug window
        cv2.imshow(self.window_name, cv2.resize(combined, (self.viewer_config.width, self.viewer_config.height)))
