"""
Notify module config definition.
"""

from dataclasses import dataclass
from typing import Any

@dataclass()
class ModuleConfig_Ntfy:
    """
    General notify settings
    """
    url: str = "https://ntfy.sh"    # NTFY base URL (must usually not be changed)
    topic: str = "NTFY_TOPIC_URL"   # Environment variable name, which holds the user topic name (e.g. 'hans_peter_ipad2')

    @classmethod
    def from_dict(cls, config_file_data_ntfy: dict[str, Any]) -> "ModuleConfig_Ntfy":
        return cls(
            url=config_file_data_ntfy.get("url", cls.url),
            topic=config_file_data_ntfy.get("topic", cls.topic)
        )


