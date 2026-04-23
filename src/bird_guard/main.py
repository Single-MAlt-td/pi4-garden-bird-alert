from pathlib import Path

from bird_guard.utils import PlatformInfo
from bird_guard.config import ConfigHandler
from bird_guard.notify.ntfy_client import NtfyHandler
from bird_guard.camera.camera import DummyCamera, PiCam2Camera, Frame
from bird_guard.vision.vision import MotionDetector


# get the root path of the repo
APP_BASE_PATH = Path(__file__).resolve().parents[2]


# ====
# Main
# ====
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
    match PlatformInfo.get_platform():
        case PlatformInfo.OperatingSystem.PROBABLY_RASPI:
            print("Detected OS is likely Raspi -> Using Picamera2!")
            camera = PiCam2Camera(settings.camera)
        case PlatformInfo.OperatingSystem.WINDOWS | PlatformInfo.OperatingSystem.LINUX:
            print("Detected OS is non-Raspi -> Using dummy camera!")
            camera = DummyCamera(settings.camera)
        case _:
            raise NotImplementedError(f"Platform {PlatformInfo.get_platform_name()} is not supported.")

    # init motion detector
    motion_detector = MotionDetector(enable_debug=True)

    # test process frames
    print("HINT: Press any key to switch to the next image. Press Q to quit.")
    while True:
        # get frame
        frame = camera.get_frame(Frame.FrameType.LORES)

        # detect
        if motion_detector.process(frame):
            print("MOVEMENT detected!")

if __name__ == "__main__":
    main()
