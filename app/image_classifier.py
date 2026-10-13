"""Turn an uploaded image into ranked CIFAR-10 predictions using the trained CNN."""

import io
from pathlib import Path

import torch
from PIL import Image

from app.cnn_model import CIFAR10_CLASSES, PREPROCESS, AssignmentCNN

DEFAULT_WEIGHTS = Path(__file__).parent / "weights" / "cifar10_cnn.pt"


class CifarImagePredictor:
    """Loads the CNN once; `top_predictions` ranks the classes for one image."""

    def __init__(self, weights_path=DEFAULT_WEIGHTS):
        self.model = AssignmentCNN(num_classes=len(CIFAR10_CLASSES))
        self.model.load_state_dict(
            torch.load(weights_path, map_location="cpu", weights_only=True))
        self.model.eval()

    @staticmethod
    def _to_tensor(image_bytes):
        # Raises PIL.UnidentifiedImageError if the bytes are not an image.
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        return PREPROCESS(image).unsqueeze(0)  # add a batch dimension: (1, 3, 64, 64)

    @torch.no_grad()
    def top_predictions(self, image_bytes, k=3):
        """Return the k most likely classes, most likely first."""
        probs = torch.softmax(self.model(self._to_tensor(image_bytes)), dim=1).squeeze(0)
        values, indices = torch.topk(probs, k)
        return [
            {"label": CIFAR10_CLASSES[i], "probability": round(v, 4)}
            for v, i in zip(values.tolist(), indices.tolist())
        ]