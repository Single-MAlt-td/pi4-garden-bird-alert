from pathlib import Path

from bird_guard.config import ConfigHandler, AppConfig
from bird_guard.notify.ntfy_client import NtfyHandler
from bird_guard.camera.camera import DummyCamera
from bird_guard.vision.vision import MotionDetector

APP_BASE_PATH = Path(__file__).resolve().parents[2]



def main():
    print("bird app started")
    config_handler = ConfigHandler(APP_BASE_PATH / "config" / "config.toml")
    settings = config_handler.get_config()
    print(settings)

    ntfy_handler = NtfyHandler(settings.ntfy)
    ntfy_handler.send_message("Test", True)

    cam = DummyCamera(settings.camera)
    detector = MotionDetector()

    for i in range(10):
        print("get frame ...")
        frame = cam.get_frame()
        print("detect ...")
        if detector.detect(frame):
            print("MOVEMENT detected!")

if __name__ == "__main__":
    main()
