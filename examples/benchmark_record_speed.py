"""
Benchmark script to check how fast high-res and lo-res images can be obtained from Picamera2
(requires a Raspberry Pi system with connected camera or some similar setup)
"""

import time
import math
from numpy.ma.extras import average

from bird_guard.camera.camera import PiCam2Camera, ModuleConfig_Camera, Frame
from bird_guard.utils import FPSTiming

num_frame_requests = 30   # number of frames queried for speed testing

# ---------------

def benchmark(cam, frame_type: Frame.FrameType) -> Frame:
    print(f"Starting benchmark for '{frame_type}' ...")
    fps_timing = FPSTiming()
    times = []
    frame = Frame()
    for i in range(num_frame_requests):
        fps_timing.start_measurement()
        frame = cam.get_frame(frame_type)
        times.append(fps_timing.get_elapsed_time())
    print(f"Average frame duration: {average(times):.3f}")
    print(f"Min/Max frame duration: [{min(times):.3f}, {max(times):.3f}]")
    print(f"Average FPS: {1.0 / average(times):.1f}")
    return frame

def main():
    settings = ModuleConfig_Camera()
    fps_timing = FPSTiming()

    fps_timing.start_measurement()
    cam = PiCam2Camera(settings)
    print(f"Init duration: {fps_timing.get_elapsed_time():.2f} seconds")

    fps_timing.start_measurement()
    frame = cam.get_frame()
    print(f"First frame duration: {fps_timing.get_elapsed_time():.3f} seconds")
    print(f"First frame shape: {frame.data.shape}")

    last_main_frame = benchmark(cam, Frame.FrameType.COLOR)
    last_lores_frame = benchmark(cam, Frame.FrameType.LORES)


if __name__ == "__main__":
    main()
