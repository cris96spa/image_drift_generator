import os
import time
from utils.singleton import Singleton
from utils.settings.settings import Settings
import torch
from dotenv import load_dotenv


class SettingsProvider(metaclass=Singleton):
    """This class provides settings for the application."""

    def __init__(self) -> None:
        self.settings = Settings()  # type: ignore
        load_dotenv()

    def is_debug(self) -> bool:
        """Return whether the application is in debug mode."""
        return self.settings.debug

    def get_default_timestamp(self) -> float:
        """Return the default timestamp."""
        return self.settings.default_timestamp

    def get_default_delta_timestamp(self) -> float:
        """Return the default delta timestamp."""
        return self.settings.default_delta_timestamp

    def is_uniform_color_jitter(self) -> bool:
        """Return whether the color jitter is uniform."""
        return self.settings.uniform_color_jitter

    def get_default_transform_threshold(self) -> float:
        """Return the default transformation threshold."""
        return self.settings.default_transformation_threshold

    def get_rotate_parameters(self) -> dict:
        """Return the parameters for the rotate transformation.

        Returns:
            dict: dictionary containing drift_params for the rotate transformation.
        """
        return {'degrees': self.settings.rotate__degrees}

    def get_brightness_parameters(self) -> dict:
        """Return the parameters for the brightness transformation.

        Returns:
            dict: dictionary containing drift_params for the brightness transformation.
        """
        return {'brightness': self.settings.color_jitter__brightness}

    def get_contrast_parameters(self) -> dict:
        """Return the parameters for the contrast transformation.

        Returns:
            dict: dictionary containing drift_params for the contrast transformation.
        """
        return {'contrast': self.settings.color_jitter__contrast}

    def get_saturation_parameters(self) -> dict:
        """Return the parameters for the saturation transformation.

        Returns:
            dict: dictionary containing drift_params for the saturation transformation.
        """
        return {'saturation': self.settings.color_jitter__saturation}

    def get_hue_parameters(self) -> dict:
        """Return the parameters for the hue transformation.

        Returns:
            dict: dictionary containing drift_params for the hue transformation.
        """
        return {'hue': self.settings.color_jitter__hue}

    def get_gaussian_blur_drift_parameters(self) -> dict:
        """Return the parameters for the gaussian blur transformation.

        Returns:
            dict: dictionary containing drift_params for the gaussian blur transformation.
        """
        return {'sigma': self.settings.gaussian_blur__sigma}

    def get_gaussian_blur_constant_parameters(self) -> dict:
        """Return the constant parameters for the gaussian blur transformation.

        Returns:
            dict: dictionary containing constant_params for the gaussian blur transformation.
        """
        return {'kernel_size': self.settings.gaussian_blur__kernel_size}

    def get_gaussian_noise_drift_parameters(self) -> dict:
        """Return the parameters for the gaussian noise transformation.

        Returns:
            dict: dictionary containing drift_params for the gaussian noise transformation.
        """
        return {'sigma': self.settings.gaussian_noise__sigma}

    def get_gaussian_noise_constant_parameters(self) -> dict:
        """Return the constant parameters for the gaussian noise transformation.

        Returns:
            dict: dictionary containing constant_params for the gaussian noise transformation.
        """
        return {'mean': self.settings.gaussian_noise__mean}

    def get_device(self) -> torch.device:
        """Return the device to be used for computation."""
        device = torch.device('cpu')
        if torch.cuda.is_available():
            device = torch.device('cuda')
        elif torch.backends.mps.is_available():
            device = torch.device('mps')
        return device

    def get_default_input_dim(self) -> tuple[int, int]:
        """Return the default input dimension."""
        return self.settings.default_input_dim, self.settings.default_input_dim

    def get_default_batch_size(self) -> int:
        """Return the default batch size."""
        return self.settings.default_batch_size


if __name__ == '__main__':
    settings_provider = SettingsProvider()
    print(settings_provider.is_debug())
    print(settings_provider.get_default_timestamp())
    print(settings_provider.get_default_delta_timestamp())
    print(settings_provider.get_rotate_parameters())
    print(settings_provider.get_brightness_parameters())
    print(settings_provider.get_contrast_parameters())
    print(settings_provider.get_saturation_parameters())
    print(settings_provider.get_hue_parameters())
    print(settings_provider.get_gaussian_blur_drift_parameters())
    print(settings_provider.get_gaussian_blur_constant_parameters())
    print(settings_provider.get_gaussian_noise_drift_parameters())
    print(settings_provider.get_gaussian_noise_constant_parameters())
    print(settings_provider.get_default_transform_threshold())
    print(settings_provider.get_device())
