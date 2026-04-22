# Dummy images for the dummy camera

The dummy camera is a simulation of a real camera used for development on non-raspi systems.
The directory `data/dummy_images/duck_5fps` must contain frame images for the dummy camera.
Because of the file sizes these images are not included in this repository, but can be generated easily:

## Download test video

See [CREDIT.md](../CREDIT.md) and download the original video (`25829-352978434.mp4`).

## Extract frames

Use the `video_frame_extractor_gui` tool to extract the frames from the downloaded video:

* Start the tool (tkinter UI): `python tools/video_frame_extractor_gui.py`
* Select the downloaded video file
* Adjust settings:
  * Start frame: 110
  * End frame: 1055
  * Frame skip: 5
  * (leave target dimensions and JPG quality as is)
* Click `Start export` and select the target folder: `data/dummy_images/duck_5fps`
* Click `Select folder` and wait until the process is finished (the progress update might get stuck (known bug))

