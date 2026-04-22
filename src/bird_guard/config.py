from pathlib import Path
import tomllib
import toml

from dataclasses import dataclass, field, asdict
from typing import Any

from bird_guard.notify.notify_config import ModuleConfig_Ntfy
from bird_guard.camera.camera_config import ModuleConfig_Camera

# ======
# CONFIG
# ======
@dataclass()
class AppConfig:
    # define fields/structs of the config
    ntfy: ModuleConfig_Ntfy = field(default_factory=ModuleConfig_Ntfy)
    camera: ModuleConfig_Camera = field(default_factory=ModuleConfig_Camera)

    # implement reader
    @classmethod
    def from_dict(cls, config_file_data: dict[str, Any]) -> "AppConfig":
        return cls(
            ntfy=ModuleConfig_Ntfy.from_dict(config_file_data.get("ntfy", {})),
            camera=ModuleConfig_Camera.from_dict(config_file_data.get("camera", {}))
        )
# ===========


# =============
# CONFIG READER
# =============
class ConfigHandler:
    def __init__(self, config_file: Path):
        self.config_file: Path = config_file
        self.config: AppConfig

        # Load config (creates default config file, if not already existent)
        self.config = self._load_app_config(self.config_file)

    def save_config(self, app_config: AppConfig) -> bool:
        return self._save_app_config(app_config, self.config_file)

    def get_config(self) -> AppConfig:
        return self.config

    def _check_and_create_default_config_file(self, config_file: Path) -> bool:
        if not config_file.exists() or config_file.stat().st_size == 0:
            try:
                config_file.parent.mkdir(parents=True, exist_ok=True)
                self._save_app_config(AppConfig(), config_file)
                print(f"Created config file: {config_file}")
            except Exception as e:
                print(f"Error during creation of config file: {e}")
                return False
        return True


    def _load_app_config(self, config_file: Path) -> AppConfig:

        # Ensure the config file exists
        self._check_and_create_default_config_file(config_file)

        # Create a config object with default data
        default_config = AppConfig()

        # Read and parse the data
        try:
            with open(config_file, "rb") as f:
                config_file_data = tomllib.load(f)

            # parse config data
            config_data = AppConfig.from_dict(config_file_data)

        except Exception as e:
            print(f"Error reading config file: {e} -> Using default configuration!")
            config_data = default_config

        return config_data


    def _save_app_config(self, config_data: AppConfig, config_file: Path) -> bool:
        try:
            with open(config_file, "w") as f:
                toml.dump(asdict(config_data), f)   # type: ignore
            return True
        except (PermissionError, OSError) as e:
            print(f"Error when writing config file: {e}")
            return False
