import torch
import numpy as np
from scipy.stats import entropy
import torch.nn as nn
from utils.settings.settings_provider import SettingsProvider


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
        images = images.to(SettingsProvider().get_device())
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
