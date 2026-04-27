# Dummy images for the dummy camera

The dummy camera is a simulation of a real camera used for development on non-raspi systems.
The data directory (see [for users](../../../README.md#user-folders) or [for developers](../../../README.md#dev-folders)) must contain subfolder `dummy_images`, which may contain further subfolders.
Each of these subfolders may contain video frame images for the dummy camera for testing.
The filenames are not relevant, but should be in order when sorted by name.

To use a certain subfolder of `<data>/dummy_images/`, set the corresponding parameter
in the config file to the name of this subfolder: `camera.dummy_camera/images_subfolder="<subfolder_name>"`

The dummy camera requires an image series for output. For this purpose, any video can be used.
At the moment, individual frames must be extracted from a video for usage by the dummy camera.
In the future it might be possible to process videos directly.

Important properties the videos (or sub-parts of them) must fulfill:

* Stationary camera (stable, no camera movements)
* Should contain target objects doing 'target things' (e.g. birds pecking seeds)

## Extracting frames from a video

Use the `video_frame_extractor_gui` tool to extract the frames from the downloaded video:

* Start the tool (tkinter UI): `python tools/video_frame_extractor_gui.py`
* Select the downloaded video file
* Adjust settings
* Click `Select folder` and select the target folder (in `<data>/dummy_images/`)
* Click `Start export` and wait until the process is finished (the progress update might get stuck (known bug))


## Example

As example, a video of ducks in a meadow could be used.
For this, a subdirectory in `<data>/dummy_images/` should be named `duck_5fps` (so the video frames will be stored in `<data>/dummy_images/ducks_5fps`).
Because of the file sizes these images are not included in this repository, but can be generated easily:

**Download test video:**

See [CREDIT.md](../CREDIT.md) and download the original video (`25829-352978434.mp4`).

**Extract video frames:**

Follow the steps from [here](#extracting-frames-from-a-video) and:
* choose `<data>/dummy_images/duck_5fps` as output directory
* Adjust settings:
  * Start frame: 110
  * End frame: 1055
  * Frame skip: 5
  * (leave target dimensions and JPG quality as is)

**Configure:**

Open the config file `<config>/config.toml` and set variables:

* `[camera.dummy_camera]`
  * `images_subfolder="ducks_5fps"`
  * `speed_factor = 1`
* `[vision]`
  * `debug = true`

**Start:**

Start the `bird-guard` app
