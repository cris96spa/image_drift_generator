from collections.abc import Sequence

import torchvision.transforms.v2 as transforms
from pydantic import BaseModel
from torchvision.transforms.v2._transform import Transform
from enum import Enum
import polars as pl


class BaseML3Enum(str, Enum):
    """
    Base class for all enums in the ML3 Platform SDK
    """

    def __str__(self):
        return self.value


class ColumnSubRole(BaseML3Enum):
    """
    Column subrole enum
    Describe the subrole of a column
    """

    RAG_USER_INPUT = 'user_input'
    RAG_RETRIEVED_CONTEXT = 'retrieved_context'
    MODEL_PROBABILITY = 'model_probability'
    OBJECT_DETECTION_LABEL_TARGET = 'object_detection_label_target'
    OBJECT_DETECTION_LABEL_PREDICTION = 'object_detection_label_prediction'


class ColumnRole(BaseML3Enum):
    """
    Column role enum
    Describe the role of a column
    """

    INPUT = 'input'
    INPUT_MASK = 'input_mask'
    PREDICTION = 'prediction'
    TARGET = 'target'
    ERROR = 'error'
    ID = 'id'
    TIME_ID = 'time_id'
    KPI = 'kpi'
    INPUT_ADDITIONAL_EMBEDDING = 'input_additional_embedding'
    TARGET_ADDITIONAL_EMBEDDING = 'target_additional_embedding'
    PREDICTION_ADDITIONAL_EMBEDDING = 'prediction_additional_embedding'
    USER_INPUT = 'user_input'
    RETRIEVED_CONTEXT = 'retrieved_context'


class DataType(BaseML3Enum):
    """
    Data type enum
    Describe data type of input
    """

    FLOAT = 'float'
    STRING = 'string'
    CATEGORICAL = 'categorical'

    # array can have multiple dimensions each of them with n elemens
    # for instance, an image is an array with c channels, hence it is
    # an array_3 with [h, w, c] where h is the number of pixels over
    # the height axis, w over the width axis and c is the number of
    # channels (3 for RGB images).

    # array [h]  # noqa
    ARRAY_1 = 'array_1'
    # array [h, w]  # noqa
    ARRAY_2 = 'array_2'
    # array [h, w, c]  # noqa
    ARRAY_3 = 'array_3'


class ImageMode(BaseML3Enum):
    """
    Image mode enumeration

    Fields
    ------
    RGB: Red, Green, Blue
    RGBA: Red, Green, Blue, Alpha
    GRAYSCALE: Grayscale
    """

    RGB = 'rgb'
    RGBA = 'rgba'
    GRAYSCALE = 'grayscale'


class ColumnInfo(BaseModel):
    """
    Column base model

    Attributes:
        name: str
        role: ColumnRole
        is_nullable: bool
        data_type: DataType
        predicted_target: Optional[str] = None
        possible_values: Optional[list[str | int | bool]] = None
        model_id: Optional[str] = None
        dims: Optional[tuple[int]] = None
            it is mandatory when data_type is Array
        classes_names: Optional[list[str]] = None
            it is mandatory when the column is the target
            in multilabel classification tasks
        subrole: Optional[ColumnSubRole] = None
            Indicates the subrole of the column. It's used in
            RAG tasks to define the role of the input columns
            (e.g. user input or retrieved context)
        image_mode: Optional[ImageMode] = None
            Indicates the mode of the image. It must be provided
            when the data type is an image
    """

    name: str
    role: ColumnRole
    is_nullable: bool
    data_type: DataType
    predicted_target: str | None = None
    possible_values: list[str | int | bool] | None = None
    model_id: str | None = None
    dims: tuple[int, ...] | None = None
    classes_names: list[str] | None = None
    subrole: ColumnSubRole | None = None
    image_mode: ImageMode | None = None


class DataSchema(BaseModel):
    """
    Data schema base model

    Attributes:
        columns: List[ColumnInfo]

    """

    columns: list[ColumnInfo]


class FileType(Enum):
    """
    Fields:
    -------
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

    def __str__(self):
        return self.value


class FolderType(Enum):
    """
    Type of folder

    **Fields**

        - UNCOMPRESSED
        - TAR
        - ZIP
    """

    UNCOMPRESSED = 'uncompressed'
    TAR = 'tar'
    ZIP = 'zip'


class ImageDataCategoryInfo(BaseModel):
    """
    Contains all necessary information for a
    data category in a task that supports images
    """

    class Config:
        """
        to allow arbitrary types like dataframe
        """

        arbitrary_types_allowed = True

    # The folder where the images are stored
    input_folder: str
    # The mapping of the images
    # between ids and filenames
    input_mapping: pl.DataFrame
    input_folder_type: FolderType
    input_file_type: FileType
    is_input_folder: bool
    # Input embedding
    input_embedding_file_type: FileType | None = None
    input_embedding_file_path: str | None = None

    # If targets are in a folder
    target_folder: str | None = None
    # Mapping if targets are in a folder
    target_mapping: pl.DataFrame | None = None
    # The target dataframe if
    # targets are scalars
    target: pl.DataFrame | None = None
    target_folder_type: FolderType | None = None
    target_file_type: FileType | None = None
    is_target_folder: bool = False

    prediction_folder: str | None = None
    prediction_mapping: pl.DataFrame | None = None
    prediction: pl.DataFrame | None = None
    prediction_folder_type: FolderType | None = None
    prediction_file_type: FileType | None = None
    is_prediction_folder: bool = False


class DriftTarget(str, Enum):
    """
    Enums that defines the type of drift applied
    """

    INPUT = 'input'
    CONDITIONAL = 'conditional'
    CONCEPT = 'concept'  # both Input and Conditional drift


class InputDriftType(str, Enum):
    """
    Enums that defines how the input drift is applied
    """

    # Input drift types for tabular data
    TABULAR_CONTINUOUS = 'tabular_continuous'
    TABULAR_DISCRETE = 'tabular_discrete'
    TABULAR_BOTH = 'tabular_both'
    TABULAR_RANDOM = 'tabular_random'

    # Input drift types for dataset where distributions
    # are known
    MIXTURE_INPUT = 'mixture_input'
    MIXTURE_PROPORTIONS = 'mixture_proportions'
    MIXTURE_BOTH = 'mixture_both'

    # Input drift types for image data
    IMAGE_AUGMENTATION = 'image_augmentation'


class ImageTransform(str, Enum):
    ROTATE = 'rotate'
    BRIGHTNESS = 'brightness'
    CONTRAST = 'contrast'
    SATURATION = 'saturation'
    HUE = 'hue'
    GAUSSIAN_BLUR = 'gaussian_blur'
    GAUSSIAN_NOISE = 'gaussian_noise'


class TransformType(BaseModel):
    """Base model for image transformations"""

    transformation: type[Transform]
    constant_params: dict = {}
    drift_params: dict


class TransformInfo(BaseModel):
    """Base model for image transformations info.
    parameters:
    ----------
    transf_type: Type of required transformation
    drift_level: required drift level for the transformation
    """

    transf_type: ImageTransform
    drift_level: float


SUPPORTED_TRANSFORM = {
    ImageTransform.ROTATE: TransformType(
        transformation=transforms.RandomRotation, drift_params={'degrees': 90}
    ),
    ImageTransform.BRIGHTNESS: TransformType(
        transformation=transforms.ColorJitter,
        drift_params={'brightness': 5},
    ),
    ImageTransform.CONTRAST: TransformType(
        transformation=transforms.ColorJitter, drift_params={'contrast': 5}
    ),
    ImageTransform.SATURATION: TransformType(
        transformation=transforms.ColorJitter,
        drift_params={'saturation': 15},
    ),
    ImageTransform.HUE: TransformType(
        transformation=transforms.ColorJitter,
        drift_params={'hue': 0.5},
    ),
    ImageTransform.GAUSSIAN_BLUR: TransformType(
        transformation=transforms.GaussianBlur,
        constant_params={'kernel_size': 5},
        drift_params={'sigma': 3.0},
    ),
    ImageTransform.GAUSSIAN_NOISE: TransformType(
        transformation=transforms.GaussianNoise,
        constant_params={'mean': 0.1},
        drift_params={'sigma': 0.5},
    ),
}


class ImageTransformFactory:
    """
    Factory class to create image transformations
    """

    @staticmethod
    def _make_transform(
        transform_element: TransformInfo,
    ) -> transforms.Transform:
        if transform_element is None:
            raise ValueError('Transform element is None')
        if transform_element.transf_type in SUPPORTED_TRANSFORM:
            # Get the transform class name
            transf_info: TransformType = SUPPORTED_TRANSFORM[
                transform_element.transf_type
            ]
            transform_cls = transf_info.transformation

            # Get parameters
            params = transf_info.constant_params.copy()
            drift_params = transf_info.drift_params
            # Update drift parameters with the drift level
            params.update(
                {
                    key: value * transform_element.drift_level
                    for key, value in drift_params.items()
                }
            )
            transform = transform_cls(**params)
            return transform
        else:
            raise ValueError(
                f'Augmentation {transform_element.transf_type} not supported'
            )

    @staticmethod
    def make_transform_pipeline(
        transform_list: list[TransformInfo],
    ) -> transforms.Compose:
        return transforms.Compose(
            [
                ImageTransformFactory._make_transform(transform_element)
                for transform_element in transform_list
            ]
        )

    @staticmethod
    def make_default_transform_pipeline(
        drift_level: float,
    ) -> transforms.Compose:
        """Create a default transformation pipeline based on the drift level.
        Using a combination of Gaussian Blur and Gaussian Noise it is possible to create an
        incremental level of drift that can be controlled by the drift level parameter.
        The rationale is that:
            - Gaussian Blur: introduce a small of drift on the image, thus is suggested for small changes.
            - Gaussian Noise: introduce a bigger amount of drift on the image, thus is suggested for big changes.
        The drift_level is used as following:
            - drift_level <= 0.5: Gaussian Blur
            - drift_level > 0.5: Gaussian Noise.

        parameters:
        -----------
        drift_level: drift level in [0, 1]

        returns:
        --------
        list of transformations
        """
        if drift_level is None:
            raise ValueError('Drift level must be provided')

        if drift_level < 0 or drift_level > 1:
            raise ValueError('Drift level must be in [0, 1]')

        if drift_level <= 0.5:
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
            input_dim = (224, 224)
        return transforms.Compose(
            [
                transforms.Resize(input_dim),
                transforms.ToTensor(),
            ]
        )
