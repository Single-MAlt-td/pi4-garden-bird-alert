import time
from pathlib import Path
import cv2

from abc import ABC, abstractmethod
from enum import Enum
from typing import Tuple

from bird_guard.camera.camera_config import ModuleConfig_Camera
from bird_guard.vision.utils.image_utils import Image, ImageUtils, BGRImage
from bird_guard.utils import PlatformInfo


CAM_MODULE_NAME = "camera"


# ============
# CAMERA FRAME
# ============
class Frame:
    """
    Hold a camera frame image as Image (np.typing.NDArray[np.uint8]) and metadata.
    """
    class FrameType(Enum):
        """
        Implemented image types.
        """
        UNSET = 0   # undefined type (default)
        LORES = 1   # low-res YUV420 image
        GRAY = 2    # high-res gray image (currently not used)
        COLOR = 3   # high-res BGR image

    def __init__(self,
                 image: Image | None = None,
                 frame_type: FrameType = FrameType.UNSET,
                 image_size: Tuple[int, int] = (None, None),
                 timestamp_override: float | None = None
                 ):
        self.data: Image = image                            # actual image array
        self.type: Frame.FrameType = frame_type             # internal image type
        self.width_image, self.height_image = image_size    # actual main image dimensions (image data array may hold additional data (e.g. YUV420))
        self.timestamp = time.time() if timestamp_override is None else timestamp_override

    @property
    def width_data(self):
        # width of the image data stored in the image array
        return self.data.shape[1] if self.data is not None else None

    @property
    def height_data(self):
        # height of the image data stored in the image array
        return self.data.shape[0] if self.data is not None else None

    def copy(self) -> "Frame":
        return Frame(self.data.copy() if self.data is not None else None,
                     self.type,
                     (self.width_image, self.height_image))

    def convert_to_bgr_color_image(self):
        match self.type:
            case Frame.FrameType.COLOR:
                pass
            case Frame.FrameType.LORES:
                self.data = ImageUtils.yuv420_image_to_color(self.data)
            case Frame.FrameType.GRAY:
                self.data = ImageUtils.gray_image_to_color(self.data)
            case _:
                raise RuntimeError("Unexpected type")

        self.type = Frame.FrameType.COLOR
        self.width_image = self.width_data
        self.height_image = self.height_data


# =================
# CAMERA SUPERCLASS (abstract)
# =================
class Camera(ABC):
    def __init__(self, settings: ModuleConfig_Camera):
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
    def __init__(self, settings: ModuleConfig_Camera):
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
            main={"size": self.settings.color_image_size, "format": "RGB888"},
            lores={"size": self.settings.lores_image_size, "format": "YUV420"},
            buffer_count=4
        )
        self.cam.configure(config)
        self.cam.set_controls({"AnalogueGain": self.settings.picamera2.ISO / 100.0})
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
    def __init__(self, settings: ModuleConfig_Camera, app_name: str):
        super().__init__(settings)

        self.app_name: str = app_name
        self.dummy_images: list[Image] = []
        self.counter: int = 0
        self.simulated_time: float = time.time()    # use the current time as initial time (because why not)

        self._initialize_camera()

    def _initialize_camera(self):
        self._load_dummy_images()

    def _load_dummy_images(self):
        self.dummy_images = []
        self.counter = 0

        # get images in frame folder
        image_folder = PlatformInfo.get_data_path(self.app_name) / "dummy_images" / self.settings.dummy_camera.images_subfolder
        jpeg_files = list(image_folder.glob("*.jp*g"))

        print(f"Loading dummy images from {image_folder} ...")
        for image_filename in jpeg_files:
            image = cv2.imread(image_filename, cv2.IMREAD_COLOR)
            self.dummy_images.append(cv2.resize(image, self.settings.color_image_size))

        if len(self.dummy_images) > 0:
            print(f"Loaded {len(self.dummy_images)} dummy images")
        else:
            raise FileNotFoundError("Failed to load the dummy images!")

    def get_frame(self, frame_type: Frame.FrameType = Frame.FrameType.COLOR) -> Frame:
        idx_return = self.counter
        self.counter = (self.counter + 1) % len(self.dummy_images)
        self.simulated_time += 1.0 / self.settings.fps

        if frame_type == Frame.FrameType.COLOR:

            return Frame(self.dummy_images[idx_return], frame_type, self.settings.color_image_size,
                         timestamp_override=self.simulated_time)

        elif frame_type == Frame.FrameType.LORES:

            lores_img = cv2.resize(self.dummy_images[idx_return], self.settings.lores_image_size)
            return Frame(cv2.cvtColor(lores_img, cv2.COLOR_BGR2YUV_I420), frame_type, self.settings.lores_image_size,
                         timestamp_override=self.simulated_time)

        else:
            raise NotImplementedError("Frame type not yet implemented.")
