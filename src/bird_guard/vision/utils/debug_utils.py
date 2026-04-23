import copy
from dataclasses import dataclass, field

import numpy as np
import cv2

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
            self.rows = rows
            self.cols = cols
            self._init_image_assignment_matrix()

        def register_image(self, row: int, col: int, name: str):
            if row in range(self.rows) and col in range(self.cols):
                self.image_assignment[row][col] = name
            else:
                raise IndexError(f"Element ({row}, {col}) exceeds matrix size")


    def __init__(self, viewer_config: DebugViewerConfiguration = DebugViewerConfiguration(rows=2, cols=2)):
        self.window_name: str = "Debug"
        self.image_storage: dict = {}
        self.viewer_config = viewer_config

    def __del__(self):
        self._hide_debug_window()

    def show_debug_window(self):
        cv2.namedWindow(self.window_name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(self.window_name, (self.viewer_config.width, self.viewer_config.height))

    def _hide_debug_window(self):
        cv2.destroyWindow(self.window_name)

    def set_image(self, name: str, image: np.array):
        self.image_storage[name] = image

    def _get_image(self, name: str) -> np.array:
        if name in self.image_storage:
            return self.image_storage[name]
        else:
            raise IndexError(f"Image with name {name} not registered!")

    def update_debug_view(self):
        im_matrix_rows = []
        for row in range(self.viewer_config.rows):
            im_matrix_cols = []
            for col in range(self.viewer_config.cols):
                image_name = self.viewer_config.image_assignment[row][col]
                if image_name:
                    image = self._get_image(image_name)
                    im_height, im_width, *im_channels = image.shape
                    if not im_channels:
                        # assume grayscale
                        image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
                    elif im_channels[0] == 3:
                        pass
                    else:
                        raise ValueError(f"Expected image with 1 or 3 color channels, but got {im_channels[0]}")

                    im_matrix_cols.append(copy.copy(image))
            im_matrix_rows.append(im_matrix_cols)

        # TODO: scale images (keep aspect ratio)

        combined = np.vstack([np.hstack([image for image in im_cols ]) for im_cols in im_matrix_rows])

        cv2.imshow(self.window_name, cv2.resize(combined, (self.viewer_config.width, self.viewer_config.height)))