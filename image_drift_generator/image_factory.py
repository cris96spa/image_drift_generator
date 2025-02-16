from collections.abc import Sequence
import torchvision.transforms.v2 as transforms
from image_drift_generator.enums import ImageTransform
from image_drift_generator.models import TransformInfo, TransformType
from utils.settings.settings_provider import SettingsProvider


AVAILABLE_TRANSFORM_TYPES = {
    ImageTransform.ROTATE: TransformType(
        transformation=transforms.RandomRotation,
        drift_params=SettingsProvider().get_rotate_parameters(),
    ),
    ImageTransform.BRIGHTNESS: TransformType(
        transformation=transforms.ColorJitter,
        drift_params=SettingsProvider().get_brightness_parameters(),
    ),
    ImageTransform.CONTRAST: TransformType(
        transformation=transforms.ColorJitter,
        drift_params=SettingsProvider().get_contrast_parameters(),
    ),
    ImageTransform.SATURATION: TransformType(
        transformation=transforms.ColorJitter,
        drift_params=SettingsProvider().get_saturation_parameters(),
    ),
    ImageTransform.HUE: TransformType(
        transformation=transforms.ColorJitter,
        drift_params=SettingsProvider().get_hue_parameters(),
    ),
    ImageTransform.GAUSSIAN_BLUR: TransformType(
        transformation=transforms.GaussianBlur,
        constant_params=SettingsProvider().get_gaussian_blur_constant_parameters(),
        drift_params=SettingsProvider().get_gaussian_blur_drift_parameters(),
    ),
    ImageTransform.GAUSSIAN_NOISE: TransformType(
        transformation=transforms.GaussianNoise,
        constant_params=SettingsProvider().get_gaussian_noise_constant_parameters(),
        drift_params=SettingsProvider().get_gaussian_noise_drift_parameters(),
    ),
    ImageTransform.DEFAULT: None,
}


class ImageTransformFactory:
    """
    Factory class to create image transformations
    """

    @staticmethod
    def _make_transform(
        transform_info: TransformInfo,
    ) -> transforms.Transform | transforms.Compose:
        """Create a transformation based on the transform element"

        Args:
            transform_info (TransformInfo): The transformation info to to be created

        Returns:
            transforms.Transform: transformation to be applied to the image dataset.

        Raises:
            ValueError: if the transformation is not supported.
        """
        if transform_info is None:
            raise ValueError('Transform info can not be None.')
        if transform_info.transf_type in AVAILABLE_TRANSFORM_TYPES:
            if transform_info.transf_type == ImageTransform.DEFAULT:
                return ImageTransformFactory._make_default_transform_pipeline(
                    transform_info.drift_level
                )

            # Get the transform class name
            transf_type: TransformType = AVAILABLE_TRANSFORM_TYPES[
                transform_info.transf_type
            ]
            transform_cls = transf_type.transformation

            # Get parameters
            params = transf_type.constant_params.copy()
            drift_params = transf_type.drift_params

            # Update drift parameters with the drift level
            for key, value in drift_params.items():
                if (
                    SettingsProvider().is_uniform_color_jitter()
                    and transform_cls == transforms.ColorJitter
                ):
                    params[key] = (
                        value * transform_info.drift_level,
                        value * transform_info.drift_level,
                    )
                else:
                    params[key] = value * transform_info.drift_level

            transform = transform_cls(**params)
            return transform
        else:
            raise ValueError(
                f'Selected transformation: {transform_info.transf_type} is currently not supported.'
            )

    @staticmethod
    def make_transform_pipeline(
        transform_list: list[TransformInfo],
    ) -> transforms.Compose:
        """Create a transformation pipeline based on the list of transformations

        Args:
            transform_list (list[TransformInfo]): List of transformations to be applied to the image dataset.

        Returns:
            transforms.Compose: transformation pipeline to be applied to the image dataset.
        """
        return transforms.Compose(
            [
                ImageTransformFactory._make_transform(transform_element)
                for transform_element in transform_list
            ]
        )

    @staticmethod
    def _make_default_transform_pipeline(
        drift_level: float,
    ) -> transforms.Compose:
        """Create a default transformation pipeline based on the drift level.
        This transformation pipeline allow to apply a drift level that affects
        the KL divergence between the original and the transformed image in a way that
        is directly proportional to the drift_level parameter.

        The drift_level is used as following:
            - drift_level <= DEFAULT_TRANSFORMATION_THRESHOLD: Gaussian Blur
            - drift_level > DEFAULT_TRANSFORMATION_THRESHOLD: Gaussian Noise.

        Args:
            drift_level (int in [0, 1]): drift level to be applied to the image dataset.

        Returns:
            transforms.Compose: transformation pipeline to be applied to the image dataset.
        """
        if drift_level is None:
            raise ValueError('Drift level must be provided')

        if drift_level < 0 or drift_level > 1:
            raise ValueError('Drift level must be in [0, 1]')

        if drift_level <= SettingsProvider().get_default_transform_threshold():
            transform_list = [
                TransformInfo(
                    transf_type=ImageTransform.GAUSSIAN_BLUR,
                    drift_level=drift_level,
                ),
            ]
        else:
            transform_list = [
                TransformInfo(
                    transf_type=ImageTransform.GAUSSIAN_NOISE,
                    drift_level=drift_level - 0.4,
                ),
            ]
        return ImageTransformFactory.make_transform_pipeline(transform_list)

    @staticmethod
    def make_input_pipeline(
        input_dim: int | Sequence[int] | None = None,
    ) -> transforms.Compose:
        """Create the input transformation pipeline for the image dataset

        parameters:
        -----------
        input_dim: dimension of the input image (Width, Height). Default is (224, 224)

        returns:
        --------
        input transformation pipeline
        """
        if input_dim is None:
            input_dim = SettingsProvider().get_default_input_dim()
        return transforms.Compose(
            [
                transforms.Resize(input_dim),
                transforms.ToTensor(),
            ]
        )
