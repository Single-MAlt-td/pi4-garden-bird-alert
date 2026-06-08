import pytest
from dataclasses import dataclass

import cv2

from tests.utils import load_video_frame_images_from_resources

from bird_guard.utils import FPSTiming
from bird_guard.recorder import VideoRecorder, Config_VideoRecorder
from bird_guard.camera.camera import Frame
from bird_guard.vision.utils.image_utils import ImageUtils

@dataclass
class test_params:
    """Test parameters"""
    use_lores: bool = False     # record lores frames (full-res color otherwise)
    num_replay_frames: int = 0  # number of frames to replay (from test data resources) and record
    idx_record_start: int = 0   # replay frame at which recording shall be started (must be > 0 to generate an image history)
    history_length_s: int = 0   # length of history image storage in (full) seconds
    fps: int = 0                # FPS for replay and recording

@dataclass
class expected_result:
    """Test quantities"""
    num_files: int = 0      # number of video files to be expected in the recording folder
    num_frames: int = 0     # number of expected frames in the saved video file
    fps: int = 0            # expected fps of the saved video file


# Test definitions
# ================
@pytest.mark.parametrize("params, expected", [
    (test_params(False, 20, 10, 1, 10), expected_result(1,20,10)),  # record 10 frames history + 10 frames life @ 10 fps
    (test_params(True, 20, 10, 1, 10), expected_result(1,20,10)),   # lores: record 10 frames history + 10 frames life @ 10 fps
    (test_params(False, 20, 10, 0, 10), expected_result(1,10,10)),  # test what happens for history length = 0
    (test_params(False, 20, 10, 1, 0), expected_result(1,0,10)),    # test what happens for fps = 0
])


# Test implementation (uses temp_dir fixture)
# ===================
def test_recording(temp_dir, params: test_params, expected: expected_result):
    """Test the bird_guard.recorder.VideoRecorder functionality (based on default settings from Config_VideoRecorder)"""

    # get images from tests/resources/video_frames as list of cv2 images
    image_list = load_video_frame_images_from_resources()
    assert len(image_list) > 0

    output_path = temp_dir
    video_path = None

    if params.fps > 0:
        fps_timer = FPSTiming(1.0/params.fps)
    else:
        fps_timer = FPSTiming(0.1)

    recorder_settings = Config_VideoRecorder()  # <- default settings
    recorder_settings.enable = True
    recorder_settings.history_seconds = params.history_length_s

    recorder = VideoRecorder(params.fps, output_path, recorder_settings)    # <- VideoRecorder

    n_frames = params.num_replay_frames
    for i in range(n_frames):
        fps_timer.start_measurement()

        image = image_list[i % len(image_list)].copy()
        image_size = ImageUtils.get_image_size_wh(image)

        # START recording NOW
        if i == params.idx_record_start:
            print("Starting recorder")
            video_path = recorder.start_recording()

        # write frame number to image
        ImageUtils.draw_text(image, str(i + 1), (10, 10),
                             anchor=ImageUtils.TextAnchor.TOP_LEFT,
                             font_scale=1, thickness=3,
                             text_color_bgr=(0, 255, 0))

        # write debug text to the image
        if i >= params.idx_record_start:
            ImageUtils.draw_text(image, "LIVE", (image_size[0] - 10, 10),
                                 anchor=ImageUtils.TextAnchor.TOP_RIGHT,
                                 font_scale=1, thickness=3,
                                 text_color_bgr=(0, 255, 255))
        else:
            ImageUtils.draw_text(image, "HISTORY", (image_size[0] - 10, 10),
                                 anchor=ImageUtils.TextAnchor.TOP_RIGHT,
                                 font_scale=1, thickness=3,
                                 text_color_bgr=(0, 0, 255))

        # put image to recorder
        print(f"Putting frame {i + 1} of {n_frames}")
        if params.use_lores:
            recorder.put_image(Frame(ImageUtils.color_image_to_yuv420(image), Frame.FrameType.LORES, image_size))
        else:
            recorder.put_image(Frame(image, Frame.FrameType.COLOR, image_size))

        # simulate real camera timing (max cpu speed is too fast for the recorder)
        fps_timer.wait_remaining_time()

    # stop recording and write video file
    print("Stop recording")
    recorder.stop_recording()

    # check if there is an output file
    output = list(output_path.glob(f"*.{recorder_settings.file_ext}"))
    assert len(output) is expected.num_files

    # check if the file is empty
    is_empty_file = output[0].stat().st_size == 0

    # check contents of the file
    if not is_empty_file:
        cap = cv2.VideoCapture(str(video_path))
        assert cap.isOpened()

        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        #width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        #height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        cap.release()

        assert total_frames is expected.num_frames
        assert fps is expected.fps
    else:
        # for 0 expected frames we expect an empty file
        assert expected.num_frames is 0
