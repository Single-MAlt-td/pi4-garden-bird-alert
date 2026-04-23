"""
General utilities.
"""

import platform
from enum import Enum

# ============
# PlatformInfo
# ============
class PlatformInfo:
    """
    Helper class to query platform information
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