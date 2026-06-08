"""
General utilities.
"""
from dataclasses import dataclass
from enum import Enum
import platform
from pathlib import Path
from platformdirs import user_config_path, user_data_path
import time

import cv2

from bird_guard.vision.utils.image_utils import BGRImage, YUV420Image


# ---------
# DebugInfo
# ---------
@dataclass
class DebugInfo:
    """Dataclass to provide debug data to modules from the main program"""
    is_dummy_camera: bool = False
    dummy_video_frame: int = 0          # 0-based index
    dummy_video_num_frames: int = 0
    is_replay_paused: bool = False



# ============
# PlatformInfo
# ============
class PlatformInfo:
    """
    Helper class to query platform and platform specific information
    """
    class OperatingSystem(Enum):
        """
        Enum for known detected platforms
        """
        UNSUPPORTED = 0
        WINDOWS = 1
        LINUX = 2
        PROBABLY_RASPI = 3


    @staticmethod
    def get_platform() -> OperatingSystem:
        """
        Detect platform and return the result.

        Returns:
            Platform Enum
        """
        if platform.system() == "Windows":
            return PlatformInfo.OperatingSystem.WINDOWS
        elif platform.system() == "Linux":
            if "arm" in platform.machine().lower() or "aarch" in platform.machine().lower():
                return PlatformInfo.OperatingSystem.PROBABLY_RASPI
            else:
                return PlatformInfo.OperatingSystem.LINUX
        else:
            return PlatformInfo.OperatingSystem.UNSUPPORTED

    @staticmethod
    def get_platform_name() -> str:
        """Return the name of the OS"""
        return platform.system()

    @staticmethod
    def get_config_path(app_name: str) -> Path:
        """Return path to the config file (returns user folder, if app was installed)"""
        repo_root_path = Path(__file__).resolve().parents[2]
        repo_config_path = repo_root_path / "config"
        if repo_config_path.exists():
            return repo_config_path
        else:
            return user_config_path(app_name, appauthor=False) / "config"

    @staticmethod
    def get_data_path(app_name: str) -> Path:
        """Return path to the data folder (returns user folder, if app was installed)"""
        repo_root_path = Path(__file__).resolve().parents[2]
        repo_data_path = repo_root_path / "data"
        if repo_data_path.exists():
            return repo_data_path
        else:
            return user_data_path(app_name, appauthor=False) / "data"

    @staticmethod
    def get_tests_path(app_name: str) -> Path:
        return PlatformInfo.get_data_path(app_name).parent / "tests"


# =========
# FPSTiming
# =========
class FPSTiming:
    """
    Helper class for time measurements and controlling precise frame times
    """
    def __init__(self, target_delta_time: float | None = None):
        """
        Args:
            target_delta_time: The target delta-time to be reached by wait_remaining_time
        """
        self._target_delta_time: float | None = target_delta_time
        self._start_time: float | None = None

    def set_target_delta_time(self, target_delta_time: float | None):
        """Overwrite the target delta-time"""
        self._target_delta_time = target_delta_time

    def start_measurement(self):
        """Set the current time as measurement start (required for calling get_elapsed_time)"""
        self._start_time = time.process_time()

    def get_elapsed_time(self, verbose=False):
        """
        Return the elapsed time since the last start_measurement call

        Args:
            verbose: Print the elapsed time

        Returns:
            Elapsed time in seconds
        """
        if self._start_time is None:
            raise ValueError("Call start_measurement first!")

        elapsed = time.process_time() - self._start_time

        if verbose:
            print(f"Elapsed time: {elapsed} seconds")

        return elapsed

    def wait_remaining_time(self, verbose=False):
        """
        Measures the elapsed time since the last start_measurement call and halts the program the remaining time
        until the target_delta_time is reached.

        Args:
            verbose: Print sleep time and elapsed time
        """
        if self._target_delta_time is None:
            raise ValueError("Target delta-time is not set!")

        # compute remaining time we need to wait
        elapsed = self.get_elapsed_time()
        sleep_time = max(0.0, self._target_delta_time - elapsed)

        if verbose:
            str_prefix = "SLOW! " if sleep_time == 0 else ""
            print(f"{str_prefix}Waiting: {sleep_time:.3f} seconds (elapsed: {elapsed:.3f} seconds) ...")

        # wait
        time.sleep(sleep_time)

        # set start time None to enforce another start_measurement call
        self._start_time = None


# ===========
# VideoPlayer
# ===========
class VideoPlayer:
    def __init__(self, video_file: Path, target_fps: int | None = None):
        self._cap = None
        self._video_file: Path = video_file
        self._target_fps: int = target_fps

        self._vinfo_fps: int | None = None
        self._vinfo_num_frames: int | None = None
        self._vinfo_size_wh: tuple[int, int] | None = None

        self._current_frame_index: int = 0
        self._frame_skip: int = 0

        self._open_video_file()


    def __del__(self):
        if self._cap is not None:
            self._cap.release()

    def _open_video_file(self):

        self._cap = cv2.VideoCapture(str(self._video_file))
        if not self._cap.isOpened():
            raise RuntimeError(f"Failed to open video file: {self._video_file}")

        self._vinfo_num_frames = int(self._cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self._vinfo_fps = int(self._cap.get(cv2.CAP_PROP_FPS))
        self._vinfo_size_wh = (int(self._cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
                               int(self._cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))

        if self._target_fps is None:
            self._frame_skip = 0
        else:
            if (self._vinfo_fps % self._target_fps) == 0:
                num_original_frames_in_target_frame = int(self._vinfo_fps / self._target_fps)
                self._frame_skip = num_original_frames_in_target_frame - 1
            else:
                raise AttributeError(f"The video frame rate ({self._vinfo_fps}) must be divisible by the specified target frame rate ({self._target_fps})!")

    def get_num_frames(self) -> int | None:
        return self._vinfo_num_frames

    def set_current_frame_index(self, frame_index: int):
        self._current_frame_index = max(0, min(frame_index, self.get_num_frames() - 1))

    def get_next_frame(self) -> BGRImage:
        if self._cap is None:
            raise RuntimeError("Failed to get frame: No video loaded!")

        # get frame
        self._cap.set(cv2.CAP_PROP_POS_FRAMES, self._current_frame_index)
        ok, frame_image = self._cap.read()

        if not ok or frame_image is None:
            raise RuntimeError(f"Failed to read frame no {self._current_frame_index}")

        # increment index
        self._current_frame_index = (self._current_frame_index + self._frame_skip + 1) % self._vinfo_num_frames

        return frame_image

    def close(self):
        if self._cap is not None:
            self._cap.release()
