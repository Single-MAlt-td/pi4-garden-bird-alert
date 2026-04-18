from pathlib import Path
import numpy as np
import cv2

from abc import ABC, abstractmethod

from bird_guard.camera.config import AppConfig_Camera

CAM_MODULE_BASE_PATH = Path(__file__).resolve().parents[0]

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
    def get_frame(self, source: str = "main"):
        raise NotImplementedError

    @staticmethod
    def save_frame(filename: Path, frame: np.array):
        cv2.imwrite(filename, frame)

    @staticmethod
    def show_frame(frame: np.array):
        cv2.imshow("View Frame", frame)

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
            main={"size": (self.settings.width, self.settings.height), "format": "RGB888"},
            lores={"size": (640, 360), "format": "YUV420"},
            buffer_count=4
        )
        self.cam.configure(config)
        self.cam.set_controls({"AnalogueGain": self.settings.ISO / 100.0})
        self.cam.start()

    def get_frame(self, source: str = "main"):
        if self.cam is not None:
            return self.cam.capture_array(source)

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

    def get_frame(self, source: str = "") -> np.array:
        idx_return = self.counter
        self.counter = (self.counter + 1) % len(self.dummy_images)
        return self.dummy_images[idx_return]
