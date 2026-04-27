"""
General utilities.
"""

from enum import Enum

import platform
from pathlib import Path
from platformdirs import user_config_path, user_data_path
from importlib.resources import files, as_file
import time


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
    def get_module_path(app_module_name: str) -> Path:
        """Return path to the module (named 'app.module.[...]')"""
        return Path(str(files(app_module_name)))


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
            print(f"Waiting: {sleep_time} seconds (elapsed: {elapsed} seconds) ...")

        # wait
        time.sleep(sleep_time)

        # set start time None to enforce another start_measurement call
        self._start_time = None
