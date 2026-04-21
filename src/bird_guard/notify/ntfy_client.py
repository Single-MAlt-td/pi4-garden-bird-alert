import requests
import os
import warnings

from bird_guard.notify.notify_config import AppConfig_Ntfy

class NtfyHandler:
    def __init__(self, config: AppConfig_Ntfy):
        self.config = config
        self.full_url = self._get_full_ntfy_url()

    def _get_topic(self) -> str or None:
        topic = os.getenv(self.config.topic)
        if topic is None:
            warnings.warn(f"NTFY topic environment variable {self.config.topic} is not found! NTFY functionality is thereby disabled!")
            return None
        return topic

    def _get_full_ntfy_url(self) -> str or None:
        topic = self._get_topic()
        if topic:
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


