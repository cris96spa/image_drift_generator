import torch
import numpy as np
from scipy.stats import entropy
import torch.nn as nn


def compute_mean_variance(image: torch.Tensor):
    """Compute mean and variance for each channel in the image tensor (CHW format)."""
    mean = torch.mean(image, dim=(1, 2))  # Mean for each channel (C, H, W)
    variance = torch.var(image, dim=(1, 2))  # Variance for each channel
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


def compute_embeddings_kl_divergence(
    original_features: torch.Tensor, transformed_features: torch.Tensor
) -> float:
    """
    Compute KL divergence between two sets of image embeddings.
    Args:
        original_features (torch.Tensor): Feature vector for the original image (batch_size, embedding_dim)
        transformed_features (torch.Tensor): Feature vector for the transformed image (batch_size, embedding_dim)

    Returns:
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
    return kl_div(log_probs_orig, probs_transformed).item()


def compute_embeddings_cosine_similarity(
    original_features: torch.Tensor, transformed_features: torch.Tensor
):
    """
    Compute the similarity between two images using a pre-trained model.
    Args:
        original_image (torch.Tensor): Original image tensor
        transformed_image (torch.Tensor): Transformed image tensor

    Returns:
        float: Cosine similarity between the two images
    """

    # Compute cosine similarity
    cosine_similarity = nn.CosineSimilarity()
    similarity = cosine_similarity(original_features, transformed_features)

    return similarity.mean().item()
