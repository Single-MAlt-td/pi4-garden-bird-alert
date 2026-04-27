"""
Camera module config definition.
"""

from dataclasses import dataclass, field
from typing import Any


# =====================
# SubConfig DummyCamera
# =====================
@dataclass
class SubConfig_DummyCamera:
    """
    Dummy camera specific settings
    """
    images_subfolder: str = "ducks_5fps"    # name of the subfolder in data/dummy_images/ containing the images to be used
    speed_factor: int = 1                   # multiplicator for the camera fps for faster debugging with the dummy camera

    @classmethod
    def from_dict(cls, config_file_data_camera: dict[str, Any]) -> "SubConfig_DummyCamera":
        return cls(
            images_subfolder=config_file_data_camera.get("images_subfolder", cls.images_subfolder),
            speed_factor=int(config_file_data_camera.get("speed_factor", cls.speed_factor)),
        )


# ===================
# SubConfig Picamera2
# ===================
@dataclass
class SubConfig_Picamera2:
    """
    Picamera2 specific settings
    """
    ISO: int = 100      # ISO setting for the camera

    @classmethod
    def from_dict(cls, config_file_data_camera: dict[str, Any]) -> "SubConfig_Picamera2":
        return cls(
            ISO=int(config_file_data_camera.get("ISO", cls.ISO)),
        )


# ===================
# ModuleConfig Camera
# ===================
@dataclass()
class ModuleConfig_Camera:
    """
    General camera settings
    """
    color_image_size: tuple[int, int] = (1920, 1080)    # size of the highres images
    lores_image_size: tuple[int, int] = (640, 360)      # size of the lores images
    fps: int = 5                                        # desired frames per second
    picamera2: SubConfig_Picamera2 = field(default_factory=SubConfig_Picamera2)
    dummy_camera: SubConfig_DummyCamera = field(default_factory=SubConfig_DummyCamera)

    @classmethod
    def from_dict(cls, config_file_data_camera: dict[str, Any]) -> "ModuleConfig_Camera":
        return cls(
            picamera2=SubConfig_Picamera2.from_dict(config_file_data_camera.get("picamera2", {})),
            dummy_camera=SubConfig_DummyCamera.from_dict(config_file_data_camera.get("dummy_camera", {})),
            color_image_size=tuple(config_file_data_camera.get("color_image_size", cls.color_image_size)),
            lores_image_size=tuple(config_file_data_camera.get("lores_image_size", cls.lores_image_size)),
            fps=int(config_file_data_camera.get("fps", cls.fps)),
        )


