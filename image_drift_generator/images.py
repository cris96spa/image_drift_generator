from PIL import Image
from cv2 import line
from matplotlib.style import available
import polars as pl
from uuid import uuid4
import os
import numpy as np
import torch
import torch.nn as nn
from torchvision import models
from torchvision.utils import save_image
from scipy.stats import entropy
from matplotlib import pyplot as plt
from tqdm.auto import tqdm
import torchvision.transforms.v2 as transforms
import plotly.graph_objects as go

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
torch.set_default_device(device)

# Define some colors to cycle through for each line (you can customize this list)
colors = [
    '#636EFA',
    '#EF553B',
    '#00CC96',
    '#AB63FA',
    '#FFA15A',
    '#19D3F3',
    '#FF6692',
    '#B6E880',
    '#FF97FF',
    '#FECB52',
    '#FF6E4A',
    '#FFCC96',
    '#B19CD9',
    '#FF6692',
    '#666666',
    '#882E72',
    '#1965B0',
    '#7FC97F',
    '#5B5B5B',
]

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

# region Similarity Metrics


def compute_mean_variance(image_tensor):
    """Compute mean and variance for each channel in the image tensor (CHW format)."""
    mean = torch.mean(
        image_tensor, dim=(1, 2)
    )  # Mean for each channel (C, H, W)
    variance = torch.var(image_tensor, dim=(1, 2))  # Variance for each channel
    return mean, variance


def compute_kl_divergence(original_image, transformed_image):
    """Compute KL divergence between pixel intensity histograms of two images."""
    original_hist = np.histogram(
        original_image.numpy().flatten(), bins=256, range=(0, 1), density=True
    )[0]
    transformed_hist = np.histogram(
        transformed_image.numpy().flatten(),
        bins=256,
        range=(0, 1),
        density=True,
    )[0]

    # Add a small value to avoid division by zero
    original_hist = original_hist + 1e-7
    transformed_hist = transformed_hist + 1e-7

    # Compute KL Divergence
    kl_div = entropy(original_hist, transformed_hist)
    return kl_div


def compute_embeddings_kl_divergence(original_features, transformed_features):
    """
    Compute KL divergence between two sets of image embeddings.
    parameters:
    -----------
        original_features (Tensor): Feature vector for the original image (batch_size, embedding_dim)
        transformed_features (Tensor): Feature vector for the transformed image (batch_size, embedding_dim)
    returns:
    --------
        float: KL divergence between the two feature vectors
    """
    kl_div = nn.KLDivLoss(reduction='batchmean')

    # Apply softmax to convert embeddings to probabilities
    log_probs_orig = torch.log_softmax(
        original_features, dim=1
    )  # Log probabilities for original embeddings
    probs_transformed = torch.softmax(
        transformed_features, dim=1
    )  # Probabilities for transformed embeddings

    # Compute KL divergence
    kl_loss = kl_div(log_probs_orig, probs_transformed)
    return kl_loss.item()


def get_features(images, model):
    with torch.no_grad():
        images = images.to(device)
        features = model(images)
    return features.squeeze()


def compute_cosine_similarity(original_features, transformed_features):
    """
    Compute cosine similarity between two feature vectors.
    parameters:
    -----------
        original_features (Tensor): Feature vector for the original image
        transformed_features (Tensor): Feature vector for the transformed image
    returns:
    --------
        float: Cosine similarity between the two feature vectors
    """
    cosine_similarity = nn.CosineSimilarity()
    similarity = cosine_similarity(original_features, transformed_features)
    return similarity


def compute_images_similarity(original_features, transformed_features):
    """
    Compute the similarity between two images using a pre-trained model.
    parameters:
    -----------
        original_image (Tensor): Original image tensor
        transformed_image (Tensor): Transformed image tensor
    returns:
    --------
        float: Cosine similarity between the two images
    """

    # Compute cosine similarity
    similarity = compute_cosine_similarity(
        original_features, transformed_features
    )
    return similarity.mean().item()


# endregion

# region Plotting Functions


def plot_histogram(image, ax, title='Pixel Intensity Histogram'):
    """Plot histogram of pixel intensities for an image."""
    # Image is assumed to be a tensor in CHW format
    image = image.reshape(-1, 3)  # Flatten the image pixels by channel

    colors = ['r', 'g', 'b']  # Color channels
    for i, color in enumerate(colors):
        ax.hist(
            image[:, i],
            bins=256,
            color=color,
            alpha=0.4,
            label=f'Channel {color.upper()}',
            density=False,
        )
    ax.legend()

    ax.set_title(title)
    ax.set_xlim([0, 1])  # Assuming normalized [0, 1] pixel values
    ax.set_ylim([0, 1000])


def plot_similarity_metric(df, metric_name, labels, title=''):
    plt.figure(figsize=(10, 6))
    for row in df.iter_rows(named=True):
        # Check condition for specific mean and sigma
        values = row[metric_name]
        sequence_index = np.arange(len(values))

        # Compute confidence intervals (Here we simulate using standard deviation)
        # You can replace this with actual error bounds if available
        ci = 1.96  # 95% confidence interval
        std_dev = np.std(values) * ci
        lower_bound = np.array(values) - std_dev
        upper_bound = np.array(values) + std_dev

        plt.plot(
            sequence_index,
            values,
            label=''.join([f'{label} = {row[label]}, ' for label in labels]),
        )
        plt.fill_between(sequence_index, lower_bound, upper_bound, alpha=0.2)

    plt.title(title)
    plt.xlabel('Sequence index (subsampled)')
    plt.ylabel(metric_name)
    plt.legend(title='parameters')
    plt.grid(True)
    plt.show()


def plotly_similarity_metric(
    df, metric_name, labels, title='', width=1200, height=800
):
    fig = go.Figure()
    # Iterate over rows in the DataFrame
    for i, row in enumerate(df.iter_rows(named=True)):
        # Extract values from the row
        values = row[metric_name]
        sequence_index = np.arange(len(values))

        # Compute confidence intervals (Here we simulate using standard deviation)
        ci = 1.96  # 95% confidence interval
        std_dev = np.std(values) * ci
        lower_bound = np.array(values) - std_dev
        upper_bound = np.array(values) + std_dev

        # Generate label string from the `labels` parameter
        label_str = ', '.join([f'{label} = {row[label]}' for label in labels])

        # Set a color for the line trace (cycle through colors list)
        color = colors[i % len(colors)]

        # Plot the main line trace
        line_trace = go.Scatter(
            x=sequence_index,
            y=values,
            mode='lines',
            name=label_str,
            line=dict(color=color),  # Set the color explicitly
            legendgroup=f'group_{i}',  # Group the line with its CI
        )

        # Add the main line trace to the figure
        fig.add_trace(line_trace)

        # Convert hex color to rgba with transparency for the CI fillcolor
        r, g, b = tuple(
            int(color.lstrip('#')[j : j + 2], 16) for j in (0, 2, 4)
        )
        fill_color = f'rgba({r}, {g}, {b}, 0.2)'

        # Add the confidence interval trace with the same color but transparent
        fig.add_trace(
            go.Scatter(
                x=np.concatenate([sequence_index, sequence_index[::-1]]),
                y=np.concatenate([upper_bound, lower_bound[::-1]]),
                fill='toself',
                fillcolor=fill_color,  # Set fill color with transparency
                line=dict(color='rgba(255,255,255,0)'),  # No line for the CI
                showlegend=False,  # Do not show CI in legend
                legendgroup=f'group_{i}',
            )
        )

    # Update the layout of the plot
    fig.update_layout(
        title=title,
        xaxis_title='Sequence index',
        yaxis_title=metric_name,
        legend_title='Parameters',
        hovermode='x',
        template='plotly_white',
        legend=dict(
            font=dict(size=10),  # Reduce legend font size
        ),
        width=width,
        height=height,
    )

    # Show the plot
    fig.show()


def plotly_boxplot_metric(
    df, metric_name, labels, title='', width=1200, height=800
):
    fig = go.Figure()

    # Iterate over rows in the DataFrame
    for i, row in enumerate(df.iter_rows(named=True)):
        # Extract values from the row
        values = row[metric_name]

        # Generate label string from the `labels` parameter
        label_str = ', '.join([f'{label} = {row[label]}' for label in labels])

        # Set a color for the box trace (cycle through colors list)
        color = colors[i % len(colors)]

        # Plot the box plot for the row
        box_trace = go.Box(
            y=values,
            name=label_str,
            marker_color=color,  # Set the color explicitly
            boxmean='sd',  # Option to show mean and standard deviation
        )

        # Add the box plot trace to the figure
        fig.add_trace(box_trace)

    # Update the layout of the plot
    fig.update_layout(
        title=title,
        xaxis_title='Parameters',
        yaxis_title=metric_name,
        legend_title='Parameters',
        hovermode='closest',
        template='plotly_white',
        width=width,
        height=height,
    )

    # Show the plot
    fig.show()


# endregion
