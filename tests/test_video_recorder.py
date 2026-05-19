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
    use_lores: bool = False
    num_replay_frames: int = 0
    idx_record_start: int = 0
    history_length_s: int = 0
    fps: int = 0

@dataclass
class expected_result:
    num_files: int = 0
    num_frames: int = 0
    fps: int = 0

@pytest.mark.parametrize("params, expected", [
    (test_params(False, 20, 10, 1, 10), expected_result(1,20,10)),
    (test_params(True, 20, 10, 1, 10), expected_result(1,20,10)),
    (test_params(False, 20, 10, 0, 10), expected_result(1,10,10)),
    (test_params(False, 20, 10, 1, 0), expected_result(1,0,10)),
])

def test_recording(temp_dir, params: test_params, expected: expected_result):

    image_list = load_video_frame_images_from_resources()
    assert len(image_list) > 0

    output_path = temp_dir
    video_path = None

    fps_timer = FPSTiming(0.1)
    recorder_settings = Config_VideoRecorder()
    recorder_settings.enable = True

    recorder_settings.history_seconds = params.history_length_s
    recorder = VideoRecorder(params.fps, output_path, recorder_settings)

    n_frames = params.num_replay_frames
    for i in range(n_frames):
        fps_timer.start_measurement()

        image = image_list[i % len(image_list)].copy()
        image_size = ImageUtils.get_image_size_wh(image)

        # write frame number to image
        ImageUtils.draw_text(image, str(i + 1), (10, 10),
                             anchor=ImageUtils.TextAnchor.TOP_LEFT,
                             font_scale=1, thickness=3,
                             text_color_bgr=(0, 255, 0))

        # start recording
        if i == params.idx_record_start:
            print("Starting recorder")
            video_path = recorder.start_recording()

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