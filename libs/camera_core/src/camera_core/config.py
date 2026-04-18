from dataclasses import dataclass
from typing import Any

@dataclass()
class AppConfig_Camera:
    width: int = 1920
    height: int = 1080
    ISO: int = 100

    @classmethod
    def from_dict(cls, config_file_data_camera: dict[str, Any]) -> "AppConfig_Camera":
        return cls(
            width=config_file_data_camera.get("width", cls.width),
            height=config_file_data_camera.get("height", cls.height),
            ISO=config_file_data_camera.get("ISO", cls.ISO),
        )

