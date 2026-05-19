import shutil
from pathlib import Path

import cv2

from bird_guard.main import APP_NAME
from bird_guard.utils import PlatformInfo

def load_video_frame_images_from_resources():
    image_folder = PlatformInfo.get_tests_path(APP_NAME) / "resources/video_frames"
    jpeg_files = list(image_folder.glob("*.jp*g"))

    image_list = []

    print(f"Loading images from {image_folder} ...")
    for image_filename in jpeg_files:
        image = cv2.imread(image_filename, cv2.IMREAD_COLOR)
        image_list.append(cv2.resize(image, (1920, 1080)))

    return image_list

def get_temp_output_path():
    return PlatformInfo.get_tests_path(APP_NAME) / "temp"

def delete_temp_subdir(subdir: Path):
    # delete a subdirectory of the tests/temp dir (and ensure it really is a subdirectory!)
    if subdir.exists() and subdir.is_relative_to(get_temp_output_path()):
        shutil.rmtree(subdir)