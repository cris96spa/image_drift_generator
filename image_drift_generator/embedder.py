import numpy as np
import torch
from torchvision.models import ResNet18_Weights, resnet18
from utils.settings.settings_provider import SettingsProvider


class ImageEmbedder:
    """
    Abstract class that defines the interface of an Image Embedder.
    """

    def __init__(
        self, device: str | None = None, model: torch.nn.Module | None = None
    ):
        # Set the device to use
        if device is None:
            self.device = SettingsProvider().get_device()
        else:
            self.device = torch.device(device)

        # Load the model
        if model is None:
            model = resnet18(weights=ResNet18_Weights.DEFAULT)

        # Remove the last layer of the model
        layers = list(model.children())[:-1]
        model = torch.nn.Sequential(*layers)
        self.model = model.to(self.device)
        self.model.eval()
        self.preprocess_fn = ResNet18_Weights.DEFAULT.transforms()

    def compute_embeddings(
        self,
        images: np.ndarray | torch.Tensor,
        preprocess: bool = False,
        to_numpy: bool = True,
    ) -> np.ndarray | torch.Tensor:
        """Compute the embeddings for the input image.

        Args:
            images (np.ndarray | torch.Tensor): Input image to compute the embeddings.
            preprocess (bool): Whether to preprocess the image before computing the embeddings.
            to_numpy (bool): Whether to return the embeddings as a numpy array.

        Returns:
            np.ndarray | torch.Tensor: Embeddings for the input image.

        Raises:
            ValueError: If the input data type is not supported.
        """

        if isinstance(images, np.ndarray):
            images = images.astype(np.float32)
            tensor_data = torch.as_tensor(
                images, device=self.device, dtype=torch.float32
            )
            if len(tensor_data.shape) == 3:
                tensor_data = tensor_data.permute(2, 0, 1)
            if len(tensor_data.shape) == 4:
                tensor_data = tensor_data.permute(0, 3, 1, 2)
        elif isinstance(images, torch.Tensor):
            tensor_data = images.to(self.device)
        else:
            raise ValueError('Input data type not supported')

        # Resnet model requires the input of 4 dimensions
        if tensor_data.ndim == 3:
            tensor_data = tensor_data.unsqueeze(0)

        # preprocess the image according to the model
        if preprocess:
            tensor_data = self.preprocess_fn(tensor_data)

        batch_size = tensor_data.shape[0]
        with torch.no_grad():
            embedding = self.model.forward(tensor_data)
        embedding = embedding.reshape(batch_size, -1)
        return embedding.cpu().numpy() if to_numpy else embedding
