import time
from collections import deque
import threading
from datetime import datetime
from pathlib import Path
from dataclasses import dataclass
from typing import Any

import cv2

from bird_guard.camera.camera import Frame


# ====================
# Config VideoRecorder
# ====================
@dataclass
class Config_VideoRecorder:
    """
    VideoRecorder specific settings
    """
    enable: bool = True         # if the recorder is enabled
    simulate: bool = False      # only simulate recording, don't really do it (useful for testing)
    history_seconds: int = 8    # max number of seconds of video history, which shall be included at the video beginning
    codec: str = "XVID"         # codec to be used by cv2.VideoWriter (alternatives: MJPG (big files!), avc1)
    file_ext: str = "avi"       # file name extension of the recorded video file (should correspond to the codec)

    @classmethod
    def from_dict(cls, config_file_data_recorder: dict[str, Any]) -> "Config_VideoRecorder":
        return cls(
            enable=bool(config_file_data_recorder.get("enable", cls.enable)),
            simulate=bool(config_file_data_recorder.get("simulate", cls.simulate)),
            history_seconds=int(config_file_data_recorder.get("history_seconds", cls.history_seconds)),
            codec=str(config_file_data_recorder.get("codec", cls.codec)),
            file_ext=str(config_file_data_recorder.get("file_ext", cls.file_ext)),
        )


# =============
# VideoRecorder
# =============
class VideoRecorder:
    """
    Class to record live videos.
    The output filename is generated automatically by date and time and is stored in the specified output folder.
    """
    def __init__(self, FPS: int, output_path: Path, settings: Config_VideoRecorder):
        self.FPS = FPS
        self.output_path = output_path
        self.settings = settings

        self.recording_queue = deque(maxlen=max(1, self.FPS * self.settings.history_seconds))

        self.stop_event = threading.Event()     # used to stop the recording thread
        self.lock = threading.Lock()            # used to make deque usage thread-safe
        self.thread_rec = None                  # currently running recording thread instance

        # init recorder
        self._initial_preparations()

    def _initial_preparations(self):
        # ensure the output path exists (cv2.VideoWriter won't do this)
        self.output_path.mkdir(parents=True, exist_ok=True)

    def put_image(self, frame: Frame):
        # copy input frame and ensure it's a BGR color image
        frame_copy = frame.copy()
        frame_copy.convert_to_bgr_color_image()

        # store frame in history queue
        with self.lock:
            self.recording_queue.append(frame_copy)

    def is_recording(self) -> bool:
        return self.thread_rec is not None

    def start_recording(self, output_filename_without_extension: str | None = None) -> Path | None:
        # if thread is already running -> ignore start command
        if self.is_recording():
            return None

        # ensure the stop queue is empty
        self.stop_event.clear()

        # clear the recording queue in case the history is disabled (0), because in this case the recording_queue
        # has a size of 1 (otherwise we couldn't use it), which may already contain a previous frame, which is a
        # history frame we don't want (has no practical relevance, but leads to correct behavior)
        if self.settings.history_seconds == 0:
            with self.lock:
                self.recording_queue.clear()

        # generate output filename
        if output_filename_without_extension is None:
            timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
            video_path = self.output_path / f"rec_{timestamp}.{self.settings.file_ext}"
        else:
            video_path = self.output_path / f"{Path(output_filename_without_extension).stem}.{self.settings.file_ext}"

        # START the recording thread
        self.thread_rec = threading.Thread(target=self._fnc_thread_recording, args=(video_path, self.settings.simulate), daemon=True)
        self.thread_rec.start()

        # return the filename
        return video_path

    def stop_recording(self):
        # nothing to do, if the thread is already stopped
        if not self.is_recording():
            return

        # stop recorder thread
        self.stop_event.set()

        # wait until the thread has finished
        self.thread_rec.join()
        self.thread_rec = None


    def _fnc_thread_recording(self, video_path: Path, do_simulate: bool):
        # crate video writer
        fourcc = cv2.VideoWriter_fourcc(*self.settings.codec)
        video = None

        # write the entire recording queue to the video file (handles new frames automatically)
        while not self.stop_event.is_set() or len(self.recording_queue) > 0:
            frame = None

            # get oldest frame from the recording queue
            with self.lock:
                if len(self.recording_queue) > 0:
                    frame = self.recording_queue.popleft()

            # no frame in queue -> sleep for some time to prevent high cpu usage and retry
            if frame is None:
                time.sleep(0.01)
                continue

            # skip the entire video writer section if simulation is enabled
            if do_simulate:
                continue

            # create the video writer, if it does not yet exist
            if video is None:
                video = cv2.VideoWriter(video_path, fourcc, self.FPS, (frame.width_image, frame.height_image))
                if not video.isOpened():
                    raise RuntimeError("Failed to create video writer")

            # WRITE the actual image to the video
            video.write(frame.data)

        # finalize the video
        if video is not None:
            video.release()

        # reset stop event
        self.stop_event.clear()

