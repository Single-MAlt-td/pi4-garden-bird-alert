from pathlib import Path

from bird_guard.config import ConfigHandler, AppConfig
from notify_core.ntfy_client import NtfyHandler
from camera_core.camera import DummyCamera
from vision_core.vision import MotionDetector

APP_BASE_PATH = Path(__file__).resolve().parents[2]



def main():
    print("bird app started")
    config_handler = ConfigHandler(APP_BASE_PATH / "config" / "config.toml")
    print(config_handler.get_config())

    ntfy_handler = NtfyHandler(config_handler.get_config().ntfy)
    ntfy_handler.send_message("Test", True)

    cam = DummyCamera()
    detector = MotionDetector()

    for i in range(10):
        print("get frame ...")
        frame = cam.get_frame()
        print("detect ...")
        if detector.detect(frame):
            print("MOVEMENT detected!")

if __name__ == "__main__":
    main()
