from dataclasses import dataclass, field
from typing import Any, Tuple


@dataclass
class SubConfig_DummyCamera:
    images_subfolder: str = "ducks_5fps"

    @classmethod
    def from_dict(cls, config_file_data_camera: dict[str, Any]) -> "SubConfig_DummyCamera":
        return cls(
            images_subfolder=config_file_data_camera.get("images_subfolder", cls.images_subfolder),
        )

@dataclass
class SubConfig_Picamera2:
    ISO: int = 100

    @classmethod
    def from_dict(cls, config_file_data_camera: dict[str, Any]) -> "SubConfig_Picamera2":
        return cls(
            ISO=config_file_data_camera.get("ISO", cls.ISO),
        )

@dataclass()
class ModuleConfig_Camera:
    color_image_size: Tuple[int, int] = (1920, 1080)
    lores_image_size: Tuple[int, int] = (640, 360)
    picamera2: SubConfig_Picamera2 = field(default_factory=SubConfig_Picamera2)
    dummy_camera: SubConfig_DummyCamera = field(default_factory=SubConfig_DummyCamera)

    @classmethod
    def from_dict(cls, config_file_data_camera: dict[str, Any]) -> "ModuleConfig_Camera":
        return cls(
            color_image_size=tuple(config_file_data_camera.get("color_image_size", cls.color_image_size)),
            lores_image_size=tuple(config_file_data_camera.get("lores_image_size", cls.lores_image_size)),
        )


