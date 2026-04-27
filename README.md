# Bird Guard

The goal of this app is to detect birds pecking grass seeds in the garden and sending alert notifications to the smartphone.
The app is built for a *Raspberry Pi 4* system with a *Pi High Quality Camera*, but is not limited to that and might be extended for other camera types in the future.

💡 For development and testing, a Raspberry Pi and/or a camera is not required, since a dummy camera module can be used to simulate the real camera behavior by using image frames of suitable videos.

⚠️ This is a personal project for lawn protection and learning image processing and camera based object tracking concepts.

## Contents

- [Installation](#installation)
  - [For Users](#for-users)
    - [Starting the App](#starting-the-app) 
    - [User Folders](#user-folders)
  - [For Developers](#for-developers)
    - [Installation using PyCharm](#installation-using-pycharm)
      - [Starting the App (main.py)](#starting-the-app-mainpy)
    - [Manual installation](#manual-installation)
      - [Starting the App (manual)](#starting-the-app-manual)
    - [Dev Folders](#dev-folders)
- [Config File](#config-file)

---

# Installation

## For Users

1. Clone repository: 
```
git clone https://github.com/Single-MAlt-td/pi4-garden-bird-alert.git
cd pi4-garden-bird-alert
```

2. Switch to the desired branch, if needed (e.g. develop):
```
git checkout develop
```

3. Create a virtual environment and activate it (highly recommended):
```
python -m venv .venv
(Windows) -> .venv\Scripts\activate
(Linux/Mac) -> source .venv/bin/activate
```

4. Install dependencies and app modules (ensure your venv is activated):
```
python -m pip install .
```

### Starting the App

Ensure your venv is activated! Then just execute:

```
bird-guard
```


### User Folders

Note the following user file locations:

* Config folder (contains the app configuration file (`config.toml`)):
  * Linux: `/home/<user>/.config/bird_guard/config`
  * Windows: `C:\Users\<user>\AppData\Local\bird_guard\config`

* Data folder (may contain additional data, e.g. video recordings and dummy images): 
  * Linux: `/home/<user>/.local/share/bird_guard/data`
  * Windows: `C:\Users\<user>\AppData\Local\bird_guard\data`


## For Developers

### Installation using PyCharm

Create a new project from the git repository:

* Open PyCharm
* Select: **File** → **Project from Version Control...** 
* Select **Repository URL** in the left sidebar (should be default)
* Ensure **Version control** is set to "Git"
* Enter the repository **URL**: https://github.com/Single-MAlt-td/pi4-garden-bird-alert.git
* Consider changing the project name
* Select **Clone**

Setup PyCharm:

* Switch to the desired branch, if needed (e.g. develop)
* Configure the Python interpreter: 
* File → Settings → Project: <project-name> → Python Interpreter
* Add an Interpreter (Python 3.11 is recommended)
* Open a Terminal in PyCharm and check if everything is correct:
  * Execute: `python -c "import sys; print(sys.executable)"` → should show the `python.exe` of the `.venv` subfolder
  * If something is odd (which happens sometimes), open a new Terminal or reload the project / restart PyCharm
* Install dependencies by executing (Terminal): `python -m pip install -e .`

#### Starting the App (main.py)

* Generate test frame images for the dummy camera (see: [ducks_5fps/README.md](data/dummy_images/ducks_5fps/README.md))
* It should now be possible to open and run `src/bird_guard/main.py` directly in PyCharm


### Manual installation

Follow steps 1 to 3 for users. Then:

4. Install dependencies and link app modules (ensure your venv is activated):
```
python -m pip install -e .
```

#### Starting the App (manual)

Ensure your venv is activated! Then execute:

```
python -m bird_guard.main
```

### Dev Folders

Unlike a user installation, all files remain in the cloned repository for developers:

* Config folder (contains the app configuration file (`config.toml`)):
  * Linux: `<repo_root>/config`
  * Windows: `<repo_root>\config`

* Data folder (may contain additional data, e.g. video recordings and dummy images): 
  * Linux: `<repo_root>/data`
  * Windows: `<repo_root>\data`

---

## Config File

All app settings can be configured in file `config.toml`, which is located in the `config` folder
(see the corresponding section in [Installation](#installation)).
