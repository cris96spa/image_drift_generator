from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """
    All the settings for the application.
    """

    # Debug mode
    debug: bool

    # Default timestamp info
    default_timestamp: float
    default_delta_timestamp: float

    # Model configuration
    model_config = SettingsConfigDict(
        env_file='.settings',
        env_file_encoding='utf-8',
        case_sensitive=False,
        extra='allow',
    )
