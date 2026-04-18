from pathlib import Path
import numpy as np
import cv2

from abc import ABC, abstractmethod
from enum import Enum
from typing import Tuple

from bird_guard.camera.camera_config import AppConfig_Camera

CAM_MODULE_BASE_PATH = Path(__file__).resolve().parents[0]

# ============
# CAMERA FRAME
# ============
class Frame:
    class FrameType(Enum):
        UNSET = 0
        LORES = 1
        GRAY = 2
        COLOR = 3

    def __init__(self,
                 frame: np.array = None,
                 frame_type: FrameType = FrameType.UNSET,
                 dimensions_xy: Tuple[int, int] = (None, None)
                 ):
        self.data: np.array = frame
        self.type: Frame.FrameType = frame_type
        self.dim_x, self.dim_y = dimensions_xy

        self.height: int or None = None
        self.width: int or None = None

        if frame is not None:
            self.height, self.width, *_ = frame.data.shape


# =================
# CAMERA SUPERCLASS (abstract)
# =================
class Camera(ABC):
    def __init__(self, settings: AppConfig_Camera):
        self.settings = settings

    @abstractmethod
    def _initialize_camera(self):
        pass

    @abstractmethod
    def get_frame(self, frame_type: Frame.FrameType = Frame.FrameType.COLOR) -> Frame:
        raise NotImplementedError

    @staticmethod
    def save_frame(filename: Path, frame: Frame):
        cv2.imwrite(filename, frame.data)

    @staticmethod
    def show_frame(frame: Frame):
        cv2.imshow("View Frame", frame.data)

# ================
# PICAMERA2 CAMERA
# ================
class PiCam2Camera(Camera):
    def __init__(self, settings: AppConfig_Camera):
        super().__init__(settings)

        self.cam = None

        self._initialize_camera()

    def _initialize_camera(self):
        try:
            from picamera2 import Picamera2, Preview
        except ImportError:
            raise RuntimeError("Missing python package: picamera2 (Note: PiCamera2 is usually only available on RaspberryPi systems)")

        self.cam = Picamera2()
        config = self.cam.create_video_configuration(
            main={"size": self.settings.color_image_size, "format": self.settings.color_image_format},
            lores={"size": self.settings.lores_image_size, "format": self.settings.lores_image_format},
            buffer_count=4
        )
        self.cam.configure(config)
        self.cam.set_controls({"AnalogueGain": self.settings.ISO / 100.0})
        self.cam.start()

    def get_frame(self, frame_type: Frame.FrameType = Frame.FrameType.COLOR) -> Frame:
        if self.cam is not None:
            match frame_type:
                case Frame.FrameType.COLOR:
                    return Frame(self.cam.capture_array("main"), frame_type, self.settings.color_image_size)
                case Frame.FrameType.LORES:
                    return Frame(self.cam.capture_array("lores"), frame_type, self.settings.lores_image_size)
                case _:
                    raise NotImplementedError("Frame type not yet implemented.")

# ============
# DUMMY CAMERA
# ============
class DummyCamera(Camera):
    def __init__(self, settings: AppConfig_Camera):
        super().__init__(settings)

        self.dummy_images: list[np.array] = []
        self.counter: int = 0

        self._initialize_camera()

    def _initialize_camera(self):
        self._load_dummy_images()

    def _load_dummy_images(self):
        self.dummy_images = []
        self.counter = 0
        image_folder = CAM_MODULE_BASE_PATH.parents[2] / "data/dummy_images"
        jpeg_files = list(image_folder.glob("*.jpeg"))
        for image_filename in jpeg_files:
            image = cv2.imread(image_filename, cv2.IMREAD_COLOR)
            self.dummy_images.append(image)
        if len(self.dummy_images) > 0:
            print(f"Loaded {len(self.dummy_images)} dummy images")
        else:
            raise FileNotFoundError("Failed to load the dummy images!")

    def get_frame(self, frame_type: Frame.FrameType = Frame.FrameType.COLOR) -> Frame:
        idx_return = self.counter
        self.counter = (self.counter + 1) % len(self.dummy_images)
        if frame_type == Frame.FrameType.COLOR:
            return Frame(self.dummy_images[idx_return], frame_type, self.settings.color_image_size)
        elif frame_type == Frame.FrameType.LORES:
            lores_img = cv2.resize(self.dummy_images[idx_return], self.settings.lores_image_size)
            return Frame(cv2.cvtColor(lores_img, cv2.COLOR_BGR2YUV_I420), frame_type, self.settings.lores_image_size)
        else:
            raise NotImplementedError("Frame type not yet implemented.")
