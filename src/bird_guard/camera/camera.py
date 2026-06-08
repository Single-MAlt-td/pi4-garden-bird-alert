import time
from collections import deque
from pathlib import Path
import cv2

from abc import ABC, abstractmethod
from enum import Enum
from typing import Tuple

from bird_guard.camera.camera_config import ModuleConfig_Camera
from bird_guard.vision.utils.image_utils import Image, ImageUtils, BGRImage
from bird_guard.utils import PlatformInfo, VideoPlayer


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

        self._app_name: str = app_name
        self._dummy_images: list[Image] = []
        self._current_index: int = 0
        self.simulated_time: float = time.time()    # use the current time as initial time (because why not)

        self._video_player: VideoPlayer | None = None

        # lores and color image can be retrieved for (approx) the same frame; to simulate this behavior queues are used
        # to provide the corresponding image and the next images are only buffered, if a requested queue is empty
        self._lores_queue = deque(maxlen=1)
        self._color_queue = deque(maxlen=1)

        self._initialize_camera()

    def _initialize_camera(self):
        self._load_dummy_images()
        self._buffer_images(0)

    def _buffer_images(self, index: int):
        self._lores_queue.append(self._dummy_images[index])
        self._color_queue.append(self._dummy_images[index])

    def _load_dummy_images(self):
        self._dummy_images = []
        self._current_index = 0

        dummy_cam_data_folder = (PlatformInfo.get_data_path(self._app_name) / "dummy_cam_data"
                                 / self.settings.dummy_camera.dummy_data_subfolder)
        supported_video_files = {".avi", ".mp4"}
        supported_image_files = {".jpg", ".jpeg", ".png"}

        # get supported image and video files in frame folder (sorted by name)
        dummy_files = files = sorted(
            [elem for elem in dummy_cam_data_folder.iterdir()
                                        if elem.is_file() and (elem.suffix.lower() in supported_image_files or
                                                               elem.suffix.lower() in supported_video_files)
            ],
            key=lambda elem: elem.name
        )

        print(f"Loading images and videos from {dummy_cam_data_folder} as dummy camera output ...")
        for file in dummy_files:
            if file.suffix.lower() in supported_image_files:
                # load image file
                image = cv2.imread(file, cv2.IMREAD_COLOR)
                self._dummy_images.append(cv2.resize(image, self.settings.color_image_size))
            else:
                # load video file
                self._video_player = VideoPlayer(file, self.settings.fps)
                for _ in range(self._video_player.get_num_frames()):
                    video_frame = self._video_player.get_next_frame()
                    self._dummy_images.append(ImageUtils.rescale(video_frame, self.settings.color_image_size))
                self._video_player.close()
                self._video_player = None

        # check
        if len(self._dummy_images) > 0:
            print(f"Loaded {len(self._dummy_images)} dummy images")
        else:
            raise FileNotFoundError("Failed to load the dummy image and/or video data!")

    def get_frame(self, frame_type: Frame.FrameType = Frame.FrameType.COLOR) -> Frame:

        next_index = (self._current_index + 1) % len(self._dummy_images)
        next_time = self.simulated_time + 1.0 / self.settings.fps

        if frame_type == Frame.FrameType.COLOR:

            if len(self._color_queue) == 0:
                self._current_index = next_index
                self.simulated_time = next_time
                self._buffer_images(self._current_index)

            dummy_image = self._color_queue.popleft()

            frame = Frame(dummy_image, frame_type, self.settings.color_image_size,
                         timestamp_override=self.simulated_time)

        elif frame_type == Frame.FrameType.LORES:

            if len(self._lores_queue) == 0:
                self._current_index = next_index
                self.simulated_time = next_time
                self._buffer_images(self._current_index)

            dummy_image = self._lores_queue.popleft()

            lores_img = cv2.resize(dummy_image, self.settings.lores_image_size)
            frame = Frame(cv2.cvtColor(lores_img, cv2.COLOR_BGR2YUV_I420), frame_type, self.settings.lores_image_size,
                         timestamp_override=self.simulated_time)

        else:
            raise NotImplementedError("Frame type not yet implemented.")

        return frame

    def get_dummy_video_info(self):
        return self._current_index, len(self._dummy_images)
