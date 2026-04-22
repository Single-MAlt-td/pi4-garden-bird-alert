import copy
from dataclasses import dataclass, field

import numpy as np
import cv2
from enum import Enum

from bird_guard.camera.camera import Frame


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

        cv2.imshow(self.window_name, cv2.resize(combined, (1920, 1080)))    # TODO: set window size by config



# ===============
# Motion Detector
# ===============
class MotionDetector:
    class DetectionMode(Enum):
        IM_DIFF = 0
        BG_REM = 1

    def __init__(self, threshold: int = 400, enable_debug = False):
        self.prev_frame = None
        self.prev_gray = None
        self.threshold = threshold

        self.debug_view = None

        self.mode: MotionDetector.DetectionMode = MotionDetector.DetectionMode.BG_REM   # TODO: Make configurable
        self.bg = None

        self._init_components(enable_debug)

    def _init_components(self, enable_debug: bool = False):
        # create MOG2 object, if required
        if self.mode == MotionDetector.DetectionMode.BG_REM:
            self.bg = cv2.createBackgroundSubtractorMOG2(
                history=200,
                varThreshold=25,
                detectShadows=True
            )

        # create DebugViewer object, if debug is enabled
        if enable_debug:
            self.debug_view = DebugViewer()
            self._setup_debug_view()

    def _show_debug_window(self):
        if self.debug_view is not None:
            self.debug_view.show_debug_window()

    def _setup_debug_view(self):
        self.debug_view.viewer_config.set_image_matrix_size(2, 2)
        self.debug_view.viewer_config.register_image(0, 0, "original")
        self.debug_view.viewer_config.register_image(0, 1, "blurred")
        self.debug_view.viewer_config.register_image(1, 0, "diff")
        self.debug_view.viewer_config.register_image(1, 1, "inspect")


    # TODO: Extract the bbx generator code into some own function or class
    def process(self, frame: Frame) -> bool:
        # convert to gray image (if needed) and set kernel size
        match frame.type:
            case Frame.FrameType.COLOR:
                gray = cv2.cvtColor(frame.data, cv2.COLOR_BGR2GRAY)
                kernel_size = 21
            case Frame.FrameType.GRAY:
                gray = frame.data
                kernel_size = 21
            case Frame.FrameType.LORES:
                gray = frame.data[:frame.dim_y, :]
                kernel_size = 5
            case _:
                raise NotImplementedError("Frame type not yet implemented.")

        if self.debug_view: self.debug_view.set_image("original", gray)

        # blur image to reduce noise
        gray_blurred = cv2.GaussianBlur(gray, (kernel_size, kernel_size), 0)
        if self.debug_view: self.debug_view.set_image("blurred", gray_blurred)

        match self.mode:
            case MotionDetector.DetectionMode.IM_DIFF:
                # store previous image (if not already set)
                if self.prev_gray is None:
                    self.prev_frame = frame
                    self.prev_gray = gray_blurred
                    self._show_debug_window()
                    return False
                # compute difference between current and previous image
                diff = cv2.absdiff(self.prev_gray, gray_blurred)
                #diff = cv2.subtract(self.prev_gray, gray)

            case MotionDetector.DetectionMode.BG_REM:
                diff = self.bg.apply(gray_blurred, learningRate=0.005)
                if self.prev_gray is None:
                    self.prev_frame = frame
                    self.prev_gray = gray_blurred
                    self._show_debug_window()
                    return False

            case _:
                raise NotImplementedError("Unknown case")

        if self.debug_view: self.debug_view.set_image("diff", diff)

        # detect significant differences via threshold
        _, thresh_raw = cv2.threshold(diff, 25, 255, cv2.THRESH_BINARY)

        # dilate to connect contours across small gaps
        thresh = cv2.dilate(thresh_raw, None, iterations=2)

        # count number of changed pixels
        changed_pixels = cv2.countNonZero(thresh)

        # detect contours
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        if self.debug_view is not None:
            #contour_areas = [cv2.contourArea(contour) for contour in contours]
            big_contours = [contour for contour in contours if cv2.contourArea(contour) > 500]

            debug_img = cv2.cvtColor(gray_blurred, cv2.COLOR_GRAY2BGR)
            cv2.drawContours(debug_img, contours, -1, (0, 0, 255), 2)
            cv2.drawContours(debug_img, big_contours, -1, (0, 255, 0), 2)
            for contour in big_contours:
                x, y, w, h = cv2.boundingRect(contour)
                cv2.rectangle(debug_img, (x, y), (x + w, y + h), (0, 255, 255), 2)

            # TODO: remove im1 or use it
            if self.mode == MotionDetector.DetectionMode.IM_DIFF:
                im1 = self.prev_gray
            elif self.mode == MotionDetector.DetectionMode.BG_REM:
                im1 = self.bg.getBackgroundImage()
            else:
                raise Exception("Unexpected detection mode.")

            if self.debug_view: self.debug_view.set_image("inspect", debug_img)

            # update images in the debug view and stop until a key press is received
            # TODO: make waitkey configurable
            # TODO: implement FPS for no waitkey
            if self.debug_view:
                self.debug_view.update_debug_view()

                key = cv2.waitKey(0)
                if key == 27 or key == ord('q'):
                    raise Exception("QUIT by user")


        # update previous image
        self.prev_frame = frame
        self.prev_gray = gray_blurred

        return changed_pixels > self.threshold