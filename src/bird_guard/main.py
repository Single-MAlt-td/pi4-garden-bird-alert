import platform
from pathlib import Path

from bird_guard.config import ConfigHandler, AppConfig
from bird_guard.notify.ntfy_client import NtfyHandler
from bird_guard.camera.camera import DummyCamera, PiCam2Camera, Frame
from bird_guard.vision.vision import MotionDetector

APP_BASE_PATH = Path(__file__).resolve().parents[2]

def get_platform() -> int:
    if platform.system() == "Windows":
        return 0
    elif platform.system() == "Linux":
        if "arm" in platform.machine().lower() or "aarch" in platform.machine().lower():
            return 1
        else:
            return -1
    else:
        raise NotImplementedError(f"Platform {platform.system()} is not supported.")

def main():
    print("bird app started")

    # read config
    config_handler = ConfigHandler(APP_BASE_PATH / "config" / "config.toml")
    settings = config_handler.get_config()
    print(settings)

    # init notify module
    ntfy_handler = NtfyHandler(settings.ntfy)

    # test notify
    ntfy_handler.send_message("Test", True)

    # auto-select camera depending on detected system
    platform_id = get_platform()
    if platform_id == 0:
        cam = DummyCamera(settings.camera)
    elif platform_id == 1:
        cam = PiCam2Camera(settings.camera)
    else:
        raise Exception("Platform not supported.")

    # init motion detector
    detector = MotionDetector()

    # test process 10 frames
    for i in range(10):
        print("get frame ...")
        frame = cam.get_frame(Frame.FrameType.LORES)
        print("detect ...")
        if detector.detect(frame):
            print("MOVEMENT detected!")

if __name__ == "__main__":
    main()
