from enum import Enum


class ExtendedEnum(str, Enum):
    def __str__(self):
        return self.value


class FileType(ExtendedEnum):
    """
    Fields:
    ---
    - CSV
    - JSON
    - PARQUET
    - PNG
    - JPG
    - NPY
    """

    CSV = 'csv'
    JSON = 'json'
    PARQUET = 'parquet'
    PNG = 'png'
    JPG = 'jpg'
    NPY = 'npy'


class FolderType(ExtendedEnum):
    """
    Type of folder

    Fields:
    ---
    - UNCOMPRESSED
    - TAR
    - ZIP
    """

    UNCOMPRESSED = 'uncompressed'
    TAR = 'tar'
    ZIP = 'zip'


class ImageTransform(ExtendedEnum):
    """Available image transformations.

    Fields:
    ---
    - ROTATE
    - BRIGHTNESS
    - CONTRAST
    - SATURATION
    - HUE
    - GAUSSIAN_BLUR
    - GAUSSIAN_NOISE
    """

    ROTATE = 'rotate'
    BRIGHTNESS = 'brightness'
    CONTRAST = 'contrast'
    SATURATION = 'saturation'
    HUE = 'hue'
    GAUSSIAN_BLUR = 'gaussian_blur'
    GAUSSIAN_NOISE = 'gaussian_noise'
    DEFAULT = 'default'
