import requests
import os
import warnings

from bird_guard.notify.notify_config import AppConfig_Ntfy

class NtfyHandler:
    def __init__(self, config: AppConfig_Ntfy):
        self.config = config
        self.full_url = self._get_full_ntfy_url()

    def _get_topic(self) -> str or None:
        topic_or_env_var = self.config.topic
        topic = os.getenv(topic_or_env_var)
        if topic is None and topic_or_env_var == AppConfig_Ntfy.topic:
            warnings.warn(f"NTFY topic is not set in config.toml and environment variable {topic_or_env_var} is not found! NTFY functionality is thereby disabled!")
            return None
        return topic

    def _get_full_ntfy_url(self) -> str or None:
        topic = self._get_topic()
        if topic is not None:
            return f"{self.config.url.rstrip('/')}/{topic.lstrip('/')}"
        else:
            return None


    def send_message(self, message: str, do_simulate: bool = False):
        if self.full_url is not None:
            if not do_simulate:
                requests.post(self.full_url, data=message.encode(encoding='utf-8'))
            else:
                print(f"Would send NTFY message: '{message}' to url '{self.full_url}'")
        else:
            if do_simulate:
                print("NTFY message sending suppressed due to undefined URL")


