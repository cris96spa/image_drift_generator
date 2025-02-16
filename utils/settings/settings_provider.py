import os
from utils.singleton import Singleton
from utils.settings.settings import Settings


class SettingsProvider(metaclass=Singleton):
    """This class provides settings for the application."""

    def __init__(self) -> None:
        self.settings = Settings()  # type: ignore

    def is_debug(self) -> bool:
        """Return whether the application is in debug mode."""
        return self.settings.debug

    def get_default_timestamp(self) -> float:
        """Return the default timestamp."""
        return self.settings.default_timestamp

    def get_default_delta_timestamp(self) -> float:
        """Return the default delta timestamp."""
        return self.settings.default_delta_timestamp
