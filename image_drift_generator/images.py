from PIL import Image
from cv2 import line
import polars as pl
from uuid import uuid4
import os
import numpy as np
import torch
import torch.nn as nn
from torchvision import models
from torchvision.utils import save_image
from scipy.stats import entropy
from tqdm.auto import tqdm
import torchvision.transforms.v2 as transforms
import plotly.graph_objects as go

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
torch.set_default_device(device)

# Define some colors to cycle through for each line (you can customize this list)


available_transforms = {
    'rotate': transforms.RandomRotation,
    'flip_horizontal': transforms.RandomHorizontalFlip,
    'brightness': transforms.ColorJitter,
    'contrast': transforms.ColorJitter,
    'saturation': transforms.ColorJitter,
    'hue': transforms.ColorJitter,
    'color_jitter': transforms.ColorJitter,
    'gaussian_blur': transforms.GaussianBlur,
    'gaussian_noise': transforms.GaussianNoise,
}


class TransformType:
    def __init__(
        self,
        transformation,
        drift_params: dict,
        constant_params: dict | None = None,
    ):
        self.transformation = transformation
        self.constant_params = (
            constant_params if constant_params is not None else {}
        )
        self.drift_params = drift_params


class TransformInfo:
    def __init__(self, transf_type: str, drift_level: float):
        self.transf_type = transf_type
        self.drift_level = drift_level


available_transforms = {
    'rotate': TransformType(
        transformation=transforms.RandomRotation, drift_params={'degrees': 90}
    ),
    'brightness': TransformType(
        transformation=transforms.ColorJitter,
        drift_params={'brightness': 5},
    ),
    'contrast': TransformType(
        transformation=transforms.ColorJitter, drift_params={'contrast': 5}
    ),
    'saturation': TransformType(
        transformation=transforms.ColorJitter,
        drift_params={'saturation': 15},
    ),
    'hue': TransformType(
        transformation=transforms.ColorJitter,
        drift_params={'hue': 0.5},
    ),
    'gaussian_blur': TransformType(
        transformation=transforms.GaussianBlur,
        constant_params={'kernel_size': 5},
        drift_params={'sigma': 3.0},
    ),
    'gaussian_noise': TransformType(
        transformation=transforms.GaussianNoise,
        constant_params={'mean': 0.1},
        drift_params={'sigma': 0.5},
    ),
}


# region Augmentation Evaluation


def get_transform(transform_element: TransformInfo):
    if transform_element is None:
        raise ValueError('Transform element is None')
    if transform_element.transf_type in available_transforms:
        # Get the transform class name
        transf_info: TransformType = available_transforms[
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


def get_transform_pipeline(transform_list: list[TransformInfo]):
    return [
        get_transform(transform_element)
        for transform_element in transform_list
    ]


def evaluate_transformations(
    dataloader, transform_list: list[TransformInfo], model
):
    metrics = []
    for batch, _ in tqdm(
        dataloader,
        desc='Processing batches',
        total=len(dataloader),
        leave=False,
    ):
        # Get the transform pipeline
        transform_pipe = get_transform_pipeline(transform_list)
        transform_pipe = transforms.Compose(transform_pipe)

        # Apply the transformation pipeline
        augmented_images = transform_pipe(batch)

        # Compute KL divergence
        # Extract features for both images
        original_features = get_features(batch, model)
        transformed_features = get_features(augmented_images, model)

        # kl_div_color = compute_kl_divergence(batch, augmented_images)
        kl_div = compute_embeddings_kl_divergence(
            original_features, transformed_features
        )
        cos_sim = compute_images_similarity(
            original_features, transformed_features
        )

        # Get the parameters for each transformation
        params = {}
        augmentation_methods = []
        for transform_element in transform_list:
            base_key = transform_element.transf_type
            augmentation_methods.append(base_key)

            for key, value in available_transforms[
                base_key
            ].drift_params.items():
                params[f'{base_key}_{key}'] = value
            for key, value in available_transforms[
                base_key
            ].constant_params.items():
                params[f'{base_key}_{key}'] = value

        drift_levels = [
            transform_element.drift_level
            for transform_element in transform_list
        ]
        avg_drift_level = sum(drift_levels) / len(drift_levels)
        row_dict = {
            'Augmentation Methods': augmentation_methods,
            **params,
            'KL Divergence': kl_div,
            # 'KL Divergence Color': kl_div_color,
            'Cosine Similarity': cos_sim,
            'Drift Level': avg_drift_level,
            # "class": labels[i].item()
        }
        # Store result for each image
        metrics.append(row_dict)
    return metrics


# endregion


# region Saving Helper Functions


def transform_and_save(
    dataloader,
    transform_list: list[TransformInfo],
    output_path: str,
):
    for batch, labels in tqdm(
        dataloader,
        desc='Processing batches',
        total=len(dataloader),
        leave=False,
    ):
        # Get the transform pipeline
        transform_pipe = get_transform_pipeline(transform_list)
        transform_pipe = transforms.Compose(transform_pipe)

        # Apply the transformation pipeline
        augmented_images = transform_pipe(batch)
        save_images(
            output_path=output_path,
            dataset=dataloader.dataset,
            images=augmented_images,
            labels=labels,
        )


def save_images(output_path, dataset, images, labels) -> None:
    """Save the images to the output folder.
    This method will save the images to the output folder, organized by class.

    parameters:
    -----------
    images: tensor of images
    labels: tensor of labels
    """
    for image, label in zip(images, labels):
        # Create the folder if it doesn't exist
        class_folder = os.path.join(output_path, dataset.classes[label])
        os.makedirs(class_folder, exist_ok=True)

        # Save the image
        filename = str(uuid4()) + '.png'
        save_path = os.path.join(class_folder, filename)
        save_image(image, save_path)


def saving_sample_wrapper(
    image_generator,
    output_folder: str,
    initial_sample_id: int,
    initial_timestamp: float,
):
    """this wrapper saves generated images and creates target and
    mapping csv files."""
    sample_id = initial_sample_id
    timestamp = initial_timestamp

    # while True:
    elem = next(image_generator)
    image_name_list = []
    label_list = []
    sample_id_list = []
    timestamp_list = []

    for i, image in enumerate(elem[0]):
        name = str(uuid4())
        pil_image = Image.fromarray(np.uint8(image * 255).squeeze())
        pil_image.save(os.path.join(output_folder, 'images', name + '.png'))
        if int(elem[1][i]) == 0:
            pil_image.save(
                os.path.join(
                    output_folder, 'tf_dirs', 'ok_front', name + '.png'
                )
            )
        else:
            pil_image.save(
                os.path.join(
                    output_folder, 'tf_dirs', 'def_front', name + '.png'
                )
            )
        image_name_list.append(name)
        sample_id_list.append(sample_id)
        timestamp_list.append(timestamp)

        # add 10 seconds
        timestamp += 10
        sample_id += 1

    for label in elem[1]:
        label_list.append(int(label))

    target = pl.DataFrame(
        {
            'label': label_list,
            'sample_id': sample_id_list,
            'timestamp': timestamp_list,
        }
    )
    mapping = pl.DataFrame(
        {
            'file_name': list(map(lambda x: x + '.png', image_name_list)),
            'sample_id': sample_id_list,
            'timestamp': timestamp_list,
        }
    )
    with open(os.path.join(output_folder, 'target.csv'), mode='ab') as f:
        target.write_csv(f, include_header=False)
    with open(os.path.join(output_folder, 'mapping.csv'), mode='ab') as f:
        mapping.write_csv(f, include_header=False)

    yield elem


# endregion
