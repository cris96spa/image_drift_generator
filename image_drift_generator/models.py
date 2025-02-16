from image_drift_generator.enums import FileType, FolderType, ImageTransform
from pydantic import BaseModel, Field
import polars as pl
from torchvision.transforms.v2._transform import Transform


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


class TransformType(BaseModel):
    """Base model for image transformations"""

    transformation: type[Transform]
    constant_params: dict = Field(default_factory=dict)
    drift_params: dict


class TransformInfo(BaseModel):
    """Base model for image transformations info.

    Args:
        transf_type (ImageTransform): Type of required transformation
        drift_level (float): drift level required for the given transformation
    """

    transf_type: ImageTransform
    drift_level: float
