from dataclasses import dataclass
from typing import Any

@dataclass()
class ModuleConfig_Ntfy:
    url: str = "https://ntfy.sh"
    topic: str = "NTFY_TOPIC_URL"

    @classmethod
    def from_dict(cls, config_file_data_ntfy: dict[str, Any]) -> "ModuleConfig_Ntfy":
        return cls(
            url=config_file_data_ntfy.get("url", cls.url),
            topic=config_file_data_ntfy.get("topic", cls.topic)
        )


