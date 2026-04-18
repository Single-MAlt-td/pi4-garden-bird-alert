from dataclasses import dataclass
from typing import Any, Tuple

@dataclass()
class AppConfig_Camera:
    color_image_size: Tuple[int, int] = (1920, 1080)
    color_image_format: str = "RGB888"
    lores_image_size: Tuple[int, int] = (640, 360)
    lores_image_format: str = "YUV420"
    ISO: int = 100

    @classmethod
    def from_dict(cls, config_file_data_camera: dict[str, Any]) -> "AppConfig_Camera":
        return cls(
            color_image_size=tuple(config_file_data_camera.get("color_image_size", cls.color_image_size)),
            color_image_format=config_file_data_camera.get("color_image_format", cls.color_image_format),
            lores_image_size=tuple(config_file_data_camera.get("lores_image_size", cls.lores_image_size)),
            lores_image_format=config_file_data_camera.get("lores_image_format", cls.lores_image_format),
            ISO=config_file_data_camera.get("ISO", cls.ISO),
        )

