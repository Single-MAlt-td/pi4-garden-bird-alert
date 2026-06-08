import cv2

from bird_guard.utils import PlatformInfo, FPSTiming, DebugInfo
from bird_guard.config import ConfigHandler
from bird_guard.recorder import VideoRecorder
from bird_guard.notify.ntfy_client import NtfyHandler
from bird_guard.camera.camera import DummyCamera, PiCam2Camera, Frame
from bird_guard.vision.vision import MotionDetector


APP_NAME = "bird_guard"


# ====
# Main
# ====
def main():
    print("bird app started")

    # read config
    # -----------
    config_handler = ConfigHandler(PlatformInfo.get_config_path(APP_NAME) / "config.toml")
    settings = config_handler.get_config()
    config_handler.save_config(settings)    # TODO: Remove when config design is finished

    # init debug struct
    # -----------------
    debug_info = DebugInfo()

    # init notify module
    # ------------------
    ntfy_handler = NtfyHandler(settings.ntfy)

    # test notify
    ntfy_handler.send_message("Test", True)

    # auto-select camera depending on detected system
    # ------------------
    speed_factor: int = 1
    match PlatformInfo.get_platform():
        case PlatformInfo.OperatingSystem.PROBABLY_RASPI:
            print("Detected OS is likely Raspi -> Using Picamera2!")
            camera = PiCam2Camera(settings.camera)
        case PlatformInfo.OperatingSystem.WINDOWS | PlatformInfo.OperatingSystem.LINUX:
            print("Detected OS is non-Raspi -> Using dummy camera!")
            camera = DummyCamera(settings.camera, APP_NAME)
            speed_factor = settings.camera.dummy_camera.speed_factor
        case _:
            raise NotImplementedError(f"Platform {PlatformInfo.get_platform_name()} is not supported.")

    # init motion detector
    # --------------------
    motion_detector = MotionDetector(settings.vision)

    # init fps timing
    # ---------------
    fps_timing = FPSTiming(1.0 / (settings.camera.fps * speed_factor))
    wait_key_enabled = False

    # init recorder
    # -------------
    if settings.recorder.enable:
        recorder = VideoRecorder(settings.camera.fps, PlatformInfo.get_data_path(APP_NAME) / "recordings", settings.recorder)
        if not settings.recorder.simulate:
            print("Video recorder enabled!")
        else:
            print("Video recorder is in simulation mode!")
    else:
        recorder = None

    # process frames
    # --------------
    print("HINT: Press SPACE to single step, TAB to continue and Q or ESCAPE to quit.")
    while True:
        # start iteration time measurement
        fps_timing.start_measurement()

        # get frames
        frame = camera.get_frame(Frame.FrameType.LORES)
        frame_hires = None
        if recorder:
            frame_hires = camera.get_frame(Frame.FrameType.COLOR)

        # store dummy camera debug data
        if isinstance(camera, DummyCamera):
            debug_info.is_dummy_camera = True
            debug_info.dummy_video_frame, debug_info.dummy_video_num_frames = camera.get_dummy_video_info()
            debug_info.is_replay_paused = wait_key_enabled

        # store hires color frame in recorder (FIFO)
        if recorder:
            recorder.put_image(frame_hires)

        # DETECT
        # ------
        if motion_detector.detect_movement(frame, debug_info):
            if recorder and not recorder.is_recording():
                rec_file_name = recorder.start_recording()
                if not settings.recorder.simulate:
                    print(f"Started recording to: {rec_file_name}")
                else:
                    print(f"WOULD start recording to: {rec_file_name}")
        else:
            if recorder and recorder.is_recording():
                recorder.stop_recording()
                if not settings.recorder.simulate:
                    print("Recording finished!")
                else:
                    print("WOULD have finished recording!")

        # act on key presses
        key = cv2.waitKey(1 - wait_key_enabled)
        if key == ord(' '):
            # space key -> switch to manual single step
            wait_key_enabled = True
        if key == 9:
            # tab key -> continue auto fps
            wait_key_enabled = False
        if key == 27 or key == ord('q'):
            # q or escape key -> quit
            print("QUIT by user")
            return

        # wait remaining time to reach the configured FPS
        if not wait_key_enabled:
            fps_timing.wait_remaining_time(verbose=False)

if __name__ == "__main__":
    main()
