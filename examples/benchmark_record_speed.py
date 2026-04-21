"""
Benchmark script to check how fast high-res and lo-res images can be obtained from Picamera2
(requires a Raspberry Pi system with connected camera or some similar setup)
"""

import time
import math
from numpy.ma.extras import average

from bird_guard.camera.camera import PiCam2Camera, AppConfig_Camera, Frame

num_frame_requests = 30   # number of frames queried for speed testing

# ---------------

t_start = math.inf  # global variable for tic and toc

def tic():
    global t_start
    t_start = time.time()

def toc():
    global t_start
    if math.isinf(t_start):
        return time.time() - t_start
    else:
        raise ValueError("use tic before toc!")

def benchmark(cam, frame_type: Frame.FrameType) -> Frame:
    print(f"Starting benchmark for '{frame_type}' ...")
    times = []
    frame = Frame()
    for i in range(num_frame_requests):
        tic()
        frame = cam.get_frame(frame_type)
        times.append(toc())
    print(f"Average frame duration: {average(times):.3f}")
    print(f"Min/Max frame duration: [{min(times):.3f}, {max(times):.3f}]")
    print(f"Average FPS: {1.0 / average(times):.1f}")
    return frame

def main():
    settings = AppConfig_Camera()

    tic()
    cam = PiCam2Camera(settings)
    print(f"Init duration: {toc():.2f} seconds")

    tic()
    frame = cam.get_frame()
    print(f"First frame duration: {toc():.3f} seconds")
    print(f"First frame shape: {frame.data.shape}")

    last_main_frame = benchmark(cam, Frame.FrameType.COLOR)
    last_lores_frame = benchmark(cam, Frame.FrameType.LORES)


if __name__ == "__main__":
    main()
