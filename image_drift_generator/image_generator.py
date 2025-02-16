import os
import shutil
from collections.abc import Sequence
from uuid import uuid4
from typing import cast, Callable
import polars as pl
from tqdm import tqdm
import torch
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
from torchvision.utils import save_image
from image_drift_generator.image_factory import (
    ImageTransformFactory,
    AVAILABLE_TRANSFORM_TYPES,
)
from abc import ABC, abstractmethod
import numpy as np
import random
from utils.settings.settings_provider import SettingsProvider
import sys
from loguru import logger
from image_drift_generator.enums import FileType, FolderType
from image_drift_generator.models import ImageDataCategoryInfo, TransformInfo
from image_drift_generator.embedder import ImageEmbedder
from utils.stats import compute_embeddings_kl_divergence

logger.add(
    sys.stderr,
    format='{time:MMMM D, YYYY > HH:mm:ss} | {level} | {message} | {extra}',
)


class DatasetGenerator(ABC):
    """Abstract class for dataset generators. It provides the basic functionalities to generate datasets."""

    def __init__(self, seed: int):
        self.seed = seed

        self.device = SettingsProvider().get_device()

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

        self._current_id = 0
        self._default_timestamp = SettingsProvider().get_default_timestamp()
        self.sample_delta_timestamp = (
            SettingsProvider().get_default_delta_timestamp()
        )  # 30 minutes
        self.numpy_generator = np.random.default_rng(seed=seed)

    @abstractmethod
    def sample(self, num_samples: int) -> pl.DataFrame:
        raise NotImplementedError

    @abstractmethod
    def add_abrupt_drift(
        self,
        **kwargs,
    ):
        raise NotImplementedError


class ImageDatasetGenerator(DatasetGenerator):
    """
    Image Dataset Generator class. Allow to generate drifted image datasets.

    Parameters:
        seed (int): seed for the random generator
        input_path (str): path to the input images. The images folder should be organized as follow:
            input_path
                |--- class1
                |--- class2
                |--- ...
        input_dim (int or tuple of int): dimension of the input images (Width, Height). Default is (224, 224).
        batch_size (int): batch size for the dataloader. Default is 32
        organize_by_class (bool): if True, the sampled images will be organized by classes in the output folder. Default is False.
        sample_delta_timestamp (int): delta timestamp between two consecutive samples. Default is 60 seconds.
        shuffle (bool): if True, the dataloader will shuffle the samples. Default is True.
        compute_embeddings (bool): if True, the generator will compute the embeddings of the images. Default is False.

    Example:
        1. Instantiate the generator with the input and output paths
        2. Call the `add_abrupt_drift` method to instanciate a transformation pipeline by providing:
            - `transform_list`: a list of transformations to apply to the images
        3. Call the `sample` method to generate the images and store them in the output folder along with the input mapping and target parquet files.
    """

    DATA_FOLDER = 'sampled_images'
    INPUT_MAPPING_FILE = 'input_mapping.parquet'
    TARGET_FILE = 'target.parquet'
    EMBEDDING_FILE = 'embeddings.parquet'
    FILE_TYPE = FileType.PNG
    EMBEDDING_FILE_TYPE = FileType.PARQUET

    TRASFORMATION_METHOD_COLUMN = 'trasformation_method'
    SIMILARITY_COLUMN = 'similarity'
    DRIFT_LEVEL_COLUMN = 'drift_level'

    def __init__(
        self,
        seed: int,
        input_path: str,
        input_dim: int | Sequence[int] | None = None,
        batch_size: int | None = None,
        organize_by_class: bool = False,
        sample_delta_timestamp: int | None = None,
        shuffle: bool = True,
        compute_embeddings: bool = False,
    ):
        super().__init__(seed)

        self.input_dim = (
            input_dim or SettingsProvider().get_default_input_dim()
        )
        self.batch_size = (
            batch_size or SettingsProvider().get_default_batch_size()
        )

        self._current_timestamp = self._default_timestamp

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
        self._transform_list = None

        # Initialize the sample delta timestamp
        self.sample_delta_timestamp = (
            sample_delta_timestamp
            or SettingsProvider().get_default_delta_timestamp()
        )

        # Folder definition
        self.input_path = input_path
        self.organize_by_class = organize_by_class
        self.compute_embeddings = compute_embeddings

    def sample(
        self, num_samples: int, output_path: str
    ) -> ImageDataCategoryInfo:
        """Sample num_samples images from the dataset.
        This method will sample num_samples images from the dataset and apply the transformation pipeline if defined.

        Args:
            num_samples (int): number of samples to extract. Must be less than the number of residual samples in the dataset
            output_path (str): path to the output images. The images folder will be organized as follow:
                output_path
                    |--- sampled_images.zip
                    |--- input_mapping.parquet
                    |--- target.parquet
                    |--- embeddings.parquet (if compute_embeddings is True)

        Returns:
            ImageDataCategoryInfo: object containing the sampled images, the input mapping and the target data

        Raises:
            ValueError: if num_samples is greater than the number of residual samples in the dataset
        """

        # Check if the number of samples required is greater than the number of residual samples in the dataset
        if num_samples + self.current_id > len(self.dataset):
            raise ValueError(
                f'Number of samples required num_samples: {num_samples}, is greater than the number of residual samples in the dataset (total length - cursor): {len(self.dataset) - self.current_id}'
            )

        # Initialize the sampled count to keep track of the number of samples extracted
        sampled_count = 0
        input_mapping_data = []
        target_data = []
        embeddings_data = []

        # Skip the samples already seen
        data_iterator = cast(DataLoader, iter(self.dataloader))
        for i in range(self.current_id // self.batch_size):
            next(data_iterator)  # type: ignore

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
                embeddings_data=embeddings_data,
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
            os.path.join(output_path, self.INPUT_MAPPING_FILE), mode='wb'
        ) as f:
            input_mapping.write_parquet(f)
        with open(os.path.join(output_path, self.TARGET_FILE), mode='wb') as f:
            target.write_parquet(f)
        if self.compute_embeddings:
            embeddings = pl.DataFrame(embeddings_data)
            embeddings = embeddings.with_columns(
                pl.col('embedding')
                .cast(pl.Array(pl.Float64, 512))
                .alias('embedding')
            )
            with open(
                os.path.join(output_path, self.EMBEDDING_FILE), mode='wb'
            ) as f:
                embeddings.write_parquet(f)

            EMBEDDING_FYLE = os.path.join(output_path, self.EMBEDDING_FILE)
            EMBEDDING_FYLE_TYPE = self.EMBEDDING_FILE_TYPE
        else:
            EMBEDDING_FYLE = None
            EMBEDDING_FYLE_TYPE = None

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
            input_embedding_file_path=EMBEDDING_FYLE,
            input_embedding_file_type=EMBEDDING_FYLE_TYPE,
        )

    def _save_samples_wrapper(
        self,
        images,
        labels,
        input_mapping_data: list,
        target_data: list,
        embeddings_data: list,
        output_path: str,
    ) -> None:
        """Save the images to the output folder."""

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
            filename = str(uuid4()) + '.' + self.FILE_TYPE.value
            save_path = os.path.join(temp_folder, filename)
            save_image(image, save_path)

            # Extend the input mapping data
            input_mapping_data.append(
                {
                    'sample-id': self.current_id,
                    'timestamp': self._current_timestamp,
                    'file_name': filename,
                }
            )

            # Extend the target mapping data
            target_data.append(
                {
                    'sample-id': self.current_id,
                    'timestamp': self._current_timestamp,
                    'label': label,
                }
            )
            if self.compute_embeddings:
                embedder = ImageEmbedder(device=self.device.type)
                embedding = (
                    embedder.compute_embeddings(image, preprocess=False)
                    .squeeze()
                    .tolist()
                )
                embeddings_data.append(
                    {
                        'sample-id': self.current_id,
                        'timestamp': self._current_timestamp,
                        'embedding': embedding,
                    }
                )
            # Update the sample id and the timestamp
            self.current_id += 1
            self._current_timestamp += self.sample_delta_timestamp

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
            logger.info(f'Archive created successfully: {folder_path}.zip')

        except Exception as e:
            # Handle errors during the zipping process
            logger.error(f'An error occurred while creating the archive: {e}')
            raise e
        else:
            # Only delete the original folder if no exception was raised
            try:
                shutil.rmtree(folder_path)
                logger.info(
                    f"Original folder '{folder_path}' deleted after zipping."
                )
            except Exception as e:
                logger.error(f'Failed to delete the original folder: {e}')
                raise e

        return folder_path_zipped

    def add_abrupt_drift(
        self,
        transform_list: list[TransformInfo],
    ) -> None:
        """Add an abrupt drift to the input images. The drift can be applied to the input images defining a transformation pipeline.
        When calling add_abrupt_drift, the processing pipeline is updated and will be applied to the next call to samples.

        Args:
            transform_list (list[TransformInfo]): list of transformations to apply to the images

        Raises:
            ValueError: if transform_list is None
        """

        if transform_list is None or len(transform_list) == 0:
            raise ValueError(
                f'The transformation list cannot be None or empty. Got: {transform_list}'
            )
        else:
            transform_pipeline = ImageTransformFactory.make_transform_pipeline(
                transform_list=transform_list
            )
        self.transform_pipeline = transform_pipeline
        self._transform_list = transform_list

    def evaluate_transformation_pipeline(
        self, similarity_func: Callable | None = None
    ) -> list[dict]:
        """Evaluate the transformation pipeline by computing the similarity between the original and transformed images.

        Args:
            similarity_func (Callable): function to compute the similarity between the original and transformed images. Default is KL divergence.

        Returns:
            list[dict]: list of dictionaries containing the transformation methods, parameters, similarity and drift level for each transformation applied.
        """
        metrics = []
        if similarity_func is None:
            similarity_func = compute_embeddings_kl_divergence

        for batch, _ in tqdm(
            self.dataloader,
            desc='Processing batches',
            total=len(self.dataloader),
            leave=False,
        ):
            if self.transform_pipeline is None:
                raise ValueError(
                    'Transformation pipeline is None. Please add a transformation pipeline before evaluating it.'
                )
            else:
                # Compute the embeddings for the original and transformed images
                original_embeddings = cast(
                    torch.Tensor,
                    ImageEmbedder().compute_embeddings(
                        images=batch, preprocess=False, to_numpy=False
                    ),
                )

                # Compute the embeddings for the transformed images
                transformed_images = self.transform_pipeline(batch)
                transormed_embeddings = cast(
                    torch.Tensor,
                    ImageEmbedder().compute_embeddings(
                        images=transformed_images,
                        preprocess=False,
                        to_numpy=False,
                    ),
                )

                # Compute the similarity between the original and transformed images
                similarity = similarity_func(
                    original_features=original_embeddings,
                    transformed_features=transormed_embeddings,
                )
                # Get the parameters for each transformation
                params = {}
                augmentation_methods = []
                for transform_element in self._transform_list:  # type: ignore
                    base_key = transform_element.transf_type
                    augmentation_methods.append(base_key)

                    for key, value in AVAILABLE_TRANSFORM_TYPES[
                        base_key
                    ].drift_params.items():
                        params[f'{base_key}_{key}'] = value
                    for key, value in AVAILABLE_TRANSFORM_TYPES[
                        base_key
                    ].constant_params.items():
                        params[f'{base_key}_{key}'] = value

                drift_levels = [
                    transform_element.drift_level
                    for transform_element in self._transform_list  # type: ignore
                ]

                avg_drift_level = sum(drift_levels) / len(drift_levels)

                row_dict = {
                    self.TRASFORMATION_METHOD_COLUMN: augmentation_methods,
                    **params,
                    self.SIMILARITY_COLUMN: similarity,
                    self.DRIFT_LEVEL_COLUMN: avg_drift_level,
                }

                metrics.append(row_dict)

        return metrics

    def reset(self):
        """Reset the generator to the initial state."""
        self.current_id = 0
        self._current_timestamp = self._default_timestamp
        self.transform_pipeline = None
        self._transform_list = None
