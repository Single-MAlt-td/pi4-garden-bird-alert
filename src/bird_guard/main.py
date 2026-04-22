import platform
from pathlib import Path
from enum import Enum

from bird_guard.config import ConfigHandler, AppConfig
from bird_guard.notify.ntfy_client import NtfyHandler
from bird_guard.camera.camera import DummyCamera, PiCam2Camera, Frame
from bird_guard.vision.vision import MotionDetector

APP_BASE_PATH = Path(__file__).resolve().parents[2]

class OperatingSystem(Enum):
    UNSUPPORTED = 0
    WINDOWS = 1
    LINUX = 2
    PROBABLY_RASPI = 3

def get_platform() -> OperatingSystem:
    if platform.system() == "Windows":
        return OperatingSystem.WINDOWS
    elif platform.system() == "Linux":
        if "arm" in platform.machine().lower() or "aarch" in platform.machine().lower():
            return OperatingSystem.PROBABLY_RASPI
        else:
            return OperatingSystem.LINUX
    else:
        return OperatingSystem.UNSUPPORTED

def main():
    print("bird app started")

    # read config
    config_handler = ConfigHandler(APP_BASE_PATH / "config" / "config.toml")
    settings = config_handler.get_config()
    print(settings)
    config_handler.save_config(settings)    # TODO: Remove when config design is finished

    # init notify module
    ntfy_handler = NtfyHandler(settings.ntfy)

    # test notify
    ntfy_handler.send_message("Test", True)

    # auto-select camera depending on detected system
    match get_platform():
        case OperatingSystem.PROBABLY_RASPI:
            cam = PiCam2Camera(settings.camera)
        case OperatingSystem.WINDOWS | OperatingSystem.LINUX:
            cam = DummyCamera(settings.camera)
        case _:
            raise NotImplementedError(f"Platform {platform.system()} is not supported.")

    # init motion detector
    motion_detector = MotionDetector(enable_debug=True)

    # test process frames
    while True:
        # get frame
        frame = cam.get_frame(Frame.FrameType.LORES)

        # detect
        if motion_detector.process(frame):
            print("MOVEMENT detected!")

if __name__ == "__main__":
    main()
