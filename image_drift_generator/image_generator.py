# type: ignore
from operator import is_
import os
import shutil
from collections.abc import Sequence
from random import sample
from uuid import uuid4
from time import time
import polars as pl
import torch
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
from torchvision.utils import save_image
from image_drift_generator.image_factory import *
from abc import ABC, abstractmethod
import numpy as np
import random


class DatasetGenerator(ABC):
    def __init__(self, seed: int):
        self.seed = seed

        # Check if GPU is available, default to CPU
        if torch.cuda.is_available():
            self.device = torch.device('cuda')
        elif torch.backends.mps.is_available():
            self.device = torch.device('mps')
        else:
            self.device = torch.device('cpu')

        # Setting seed for reproducibility
        np.random.seed(seed)
        random.seed(seed)
        torch.use_deterministic_algorithms(True)
        torch.set_default_device(self.device)
        self.torch_generator = torch.manual_seed(seed)
        pl.set_random_seed(seed)

        # If GPU is available, set seed for torch.Generator
        if self.device != torch.device('cpu'):
            self.torch_generator = torch.Generator(device=self.device)
            self.torch_generator.manual_seed(seed)

        self.current_id = 0
        self.current_timestamp = 1717232400.0
        self.sample_delta_timestamp = 1800  # 30 minutes
        self.numpy_generator = np.random.default_rng(seed=seed)

    @abstractmethod
    def sample(self, num_samples: int) -> pl.DataFrame:
        raise NotImplementedError

    @abstractmethod
    def add_abrupt_drift(
        self,
        drift_target: DriftTarget,
        drift_level: float | None,
        input_drift_type: InputDriftType | None = None,
        **kwargs,
    ):
        raise NotImplementedError

    @abstractmethod
    def get_dataschema(self):
        raise NotImplementedError


class ImageDatasetGenerator(DatasetGenerator):
    """Image Dataset Generator class. Allow to generate drifted image datasets.

    Attributes:
    -----------
    seed: seed for the random generator
    input_path: path to the input images. The images folder should be organized as follow:
        input_path
            |--- class1
            |--- class2
            |--- ...
    output_path: path to the output images. The images folder will be organized as follow:
        output_path
            |--- sampled_images.zip
            |--- input_mapping.parquet
            |--- target.parquet
    input_dim: dimension of the input images (Width, Height). Default is (224, 224).
    batch_size: batch size for the dataloader. Default is 32
    """

    DATA_FOLDER = 'sampled_images'
    INPUT_MAPPING_FILE = 'input_mapping.parquet'
    TARGET_FILE = 'target.parquet'
    FILE_TYPE = 'png'

    def __init__(
        self,
        seed: int,
        input_path: str,
        input_dim: int | Sequence[int] | None = None,
        batch_size: int | None = None,
        organize_by_class: bool = False,
        sample_delta_timestamp: int | None = None,
        shuffle: bool = True,
    ):
        super().__init__(seed)

        self.input_dim = input_dim or (224, 224)
        self.batch_size = batch_size or 32

        # Get the input transform pipeline. It consists of a resize and a tensorization
        self.input_transform = ImageTransformFactory.make_input_pipeline(
            input_dim=input_dim
        )

        # Create the dataset and the dataloader
        self.dataset = ImageFolder(
            root=input_path, transform=self.input_transform
        )
        self.dataloader = DataLoader(
            dataset=self.dataset,
            batch_size=self.batch_size,
            shuffle=shuffle,
            generator=self.torch_generator,
        )

        self.CLASS_MAPPING = {
            i: self.dataset.classes[i]
            for i in range(len(self.dataset.classes))
        }
        self.LABELS = self.CLASS_MAPPING.keys()

        # Initialize the transformation pipeline
        self.transform_pipeline = None

        # Initialize the sample delta timestamp
        self.sample_delta_timestamp = sample_delta_timestamp or 60

        # Folder definition
        self.input_path = input_path
        self.organize_by_class = organize_by_class

    def sample(
        self, num_samples: int, output_path: str
    ) -> ImageDataCategoryInfo:
        """Sample num_samples images from the dataset.
        This method will sample num_samples images from the dataset and apply the transformation pipeline if defined.

        parameters:
        -----------
        num_samples: number of samples to extract. Must be less than the number of residual samples in the dataset
                - output_path: path to the output images. The images folder will be organized as follow:
        output_path
            |--- sampled_images.zip
            |--- input_mapping.parquet
            |--- target.parquet
        raises:
        -------
        ValueError: if num_samples is greater than the number of residual samples in the dataset

        returns:
        --------
        ImageDataCategoryInfo: object containing the sampled images, the input mapping and the target data"""

        # Check if the number of samples required is greater than the number of residual samples in the dataset
        if num_samples + self.current_id > len(self.dataset):
            raise ValueError(
                f'Number of samples required num_samples: {num_samples}, is greater than the number of residual samples in the dataset (total length - cursor): {len(self.dataset) - self.current_id}'
            )

        # Initialize the sampled count to keep track of the number of samples extracted
        sampled_count = 0
        input_mapping_data = []
        target_data = []

        # Skip the samples already seen
        data_iterator = iter(self.dataloader)
        for _ in range(self.current_id // self.batch_size):
            next(data_iterator)

        # Sample the images and apply the transformation pipeline if defined
        for images, labels in data_iterator:
            if sampled_count >= num_samples:
                break
            tensor_images = images.to(self.device)

            if self.transform_pipeline is not None:
                transformed_images = self.transform_pipeline(tensor_images)
            else:
                transformed_images = tensor_images

            # Get the current batch size as the minimum between the
            # default batch_size and the remaining samples to extract
            curr_batch_size = min(
                len(transformed_images), num_samples - sampled_count
            )

            # Save the images to the output folder
            self._save_samples_wrapper(
                images=transformed_images[:curr_batch_size],
                labels=labels,
                input_mapping_data=input_mapping_data,
                target_data=target_data,
                output_path=output_path,
            )

            # Update the sampled count and the cursor
            sampled_count += curr_batch_size

        # Prepare the ImageDataCategoryInfo object
        # 1. Create the dataframes
        input_mapping = pl.DataFrame(input_mapping_data)
        target = pl.DataFrame(target_data)

        # 2. Save the dataframes
        with open(
            os.path.join(output_path, self.INPUT_MAPPING_FILE), mode='ab'
        ) as f:
            input_mapping.write_parquet(f)
        with open(os.path.join(output_path, self.TARGET_FILE), mode='ab') as f:
            target.write_parquet(f)

        # 3. Zip the sampled images
        input_folder = self._make_archive(output_path)

        # 4. Return the ImageDataCategoryInfo object
        return ImageDataCategoryInfo(
            input_folder=input_folder,
            input_folder_type=FolderType.ZIP,
            input_file_type=self.FILE_TYPE,
            is_input_folder=True,
            input_mapping=input_mapping,
            target=target,
            is_target_folder=False,
        )

    def _save_samples_wrapper(
        self,
        images,
        labels,
        input_mapping_data: list,
        target_data: list,
        output_path: str,
    ) -> None:
        """Save the images to the output folder.

        parameters:
        -----------
        images: tensor of images
        labels: tensor of labels
        input_mapping_data: list of input mapping data that will be updated in place
        target_data: list of target mapping data that will be updated in place
        - output_path: path to the output images. The images folder will be organized as follow:
            output_path
                |--- sampled_images.zip
                |--- input_mapping.parquet
                |--- target.parquet
        """

        for image, label in zip(images, labels, strict=False):
            # Create the temp folder if it doesn't exist
            # Will be deleted after zipping
            temp_folder = os.path.join(output_path, self.DATA_FOLDER)
            # Check if the images should be organized by classes
            if self.organize_by_class:
                temp_folder = os.path.join(
                    temp_folder, self.CLASS_MAPPING[label.item()]
                )

            os.makedirs(temp_folder, exist_ok=True)
            # Save the image
            filename = str(uuid4()) + '.' + self.FILE_TYPE
            save_path = os.path.join(temp_folder, filename)
            save_image(image, save_path)

            # Extend the input mapping data
            input_mapping_data.append(
                {
                    'sample-id': self.current_id,
                    'timestamp': self.current_timestamp,
                    'file_name': filename,
                }
            )

            # Extend the target mapping data
            target_data.append(
                {
                    'sample-id': self.current_id,
                    'timestamp': self.current_timestamp,
                    'label': label,
                }
            )

            # Update the sample id and the timestamp
            self.current_id += 1
            self.current_timestamp += self.sample_delta_timestamp

    def _make_archive(self, output_path: str) -> str:
        """
        Zips the Data folder and deletes the temporary one where images are saved,
        after successful zipping. In case of any error, the original folder is preserved.

        parameters:
        -----------
        - output_path: path to the output images. The images folder will be organized as follow:
            output_path
                |--- sampled_images.zip
                |--- input_mapping.parquet
                |--- target.parquet
        """
        folder_path = os.path.join(output_path, self.DATA_FOLDER)

        try:
            # Create the zip archive
            folder_path_zipped = shutil.make_archive(
                folder_path, 'zip', folder_path
            )
            print(f'Archive created successfully: {folder_path}.zip')

        except Exception as e:
            # Handle errors during the zipping process
            print(f'An error occurred while creating the archive: {e}')
            raise e
        else:
            # Only delete the original folder if no exception was raised
            try:
                shutil.rmtree(folder_path)
                print(
                    f"Original folder '{folder_path}' deleted after zipping."
                )
            except Exception as e:
                print(f'Failed to delete the original folder: {e}')
                raise e

        return folder_path_zipped

    def add_abrupt_drift(
        self,
        drift_target: DriftTarget,
        drift_level: float | None = None,
        input_drift_type: InputDriftType | None = None,
        transform_list: list[TransformInfo] | None = None,
    ):
        """Add abrupt drift.
        The drift can be applied to the input images defining a transformation pipeline.
        When calling add_abrupt_drift, the processing pipeline is updated and will be applied to the next samples.

        parameters:
        -----------
        drift_target: target of the drift
        drift_level: level of the drift in [0, 1]
        input_drift_type: type of the drift
        transform_list: list of transformations to apply to the images

        raises:
        -------
        ValueError: if drift target or input drift type are not supported

        """
        match drift_target:
            case DriftTarget.INPUT:
                match input_drift_type:
                    case InputDriftType.IMAGE_AUGMENTATION:
                        if drift_level is not None:
                            if drift_level == 0:
                                logger.warning(
                                    'Drift level is 0, resetting the transformation pipeline.'
                                )
                                # Reset the transformation pipeline, next samples will be drift-free
                                self.transform_pipeline = None
                                return
                            transform_pipeline = ImageTransformFactory._make_default_transform_pipeline(
                                drift_level=drift_level
                            )
                            if transform_list is not None:
                                logger.warning(
                                    'Both drift_level and transformation are provided, drift_level will be used'
                                )
                        elif transform_list is None:
                            raise ValueError(
                                'Either drift_level or transformation must be provided, now both are None'
                            )
                        else:
                            transform_pipeline = (
                                ImageTransformFactory.make_transform_pipeline(
                                    transform_list=transform_list
                                )
                            )
                        self.transform_pipeline = transform_pipeline
                    case _:
                        raise ValueError(
                            f'Input drift type {input_drift_type} is not supported'
                        )
            case _:
                raise ValueError(
                    f'Drift target {drift_target} is not supported'
                )

    def get_dataschema(self) -> DataSchema:
        """
        Returns the data schema
        """

        columns = [
            ColumnInfo(
                name='timestamp',
                role=ColumnRole.TIME_ID,
                is_nullable=False,
                data_type=DataType.FLOAT,
            ),
            ColumnInfo(
                name='sample-id',
                role=ColumnRole.ID,
                is_nullable=False,
                data_type=DataType.STRING,
            ),
            ColumnInfo(
                name='label',
                role=ColumnRole.TARGET,
                is_nullable=False,
                data_type=DataType.CATEGORICAL,
                possible_values=self.LABELS,
            ),
            ColumnInfo(
                name='image',
                role=ColumnRole.INPUT,
                is_nullable=False,
                data_type=DataType.ARRAY_3,
                dims=(224, 224, 3),
                image_mode=ImageMode.RGB,
            ),
        ]
        return DataSchema(columns=columns)

    def reset(self):
        """Reset the generator to the initial state."""
        self.current_id = 0
        self.current_timestamp = 1717232400.0
        self.transform_pipeline = None
