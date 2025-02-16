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

    # Default transformation thresholds
    default_transformation_threshold: float

    # Rotate parameters
    rotate__degrees: int

    # Color jitter
    uniform_color_jitter: bool

    # Brightness parameters
    color_jitter__brightness: float

    # Contrast parameters
    color_jitter__contrast: float

    # Saturation parameters
    color_jitter__saturation: float

    # Hue parameters
    color_jitter__hue: float

    # Gaussian blur parameters
    gaussian_blur__sigma: float
    gaussian_blur__kernel_size: int

    # Gaussian noise parameters
    gaussian_noise__sigma: float
    gaussian_noise__mean: float

    # Model configuration
    model_config = SettingsConfigDict(
        env_file='.settings',
        env_file_encoding='utf-8',
        case_sensitive=False,
        extra='allow',
    )
