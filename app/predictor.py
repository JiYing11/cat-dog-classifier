"""Load a trained checkpoint and predict cat/dog probabilities for one image."""

from dataclasses import dataclass
from pathlib import Path

import torch
from PIL import Image
from torchvision import transforms

from ml.model import CLASS_NAMES, IMAGE_SIZE, IMAGENET_MEAN, IMAGENET_STD, build_model


ROOT = Path(__file__).resolve().parents[1]
MODEL_ROOT = ROOT / "models"


@dataclass
class CatDogPredictor:
    """A ready-to-use model plus the preprocessing and device it expects."""

    model: torch.nn.Module
    device: torch.device
    class_names: tuple[str, ...]
    preprocess: transforms.Compose

    def predict(self, image: Image.Image) -> dict[str, float]:
        """Return one probability per class for a PIL image."""
        image_tensor = self.preprocess(image.convert("RGB")).unsqueeze(0).to(self.device)
        with torch.inference_mode():
            probabilities = torch.softmax(self.model(image_tensor), dim=1)[0]
        return {
            class_name: float(probabilities[index].item())
            for index, class_name in enumerate(self.class_names)
        }


def load_predictor(version: str = "v2") -> CatDogPredictor:
    """Load a saved version checkpoint without downloading pretrained weights."""
    if version not in {"v1", "v2"}:
        raise ValueError("Model version must be 'v1' or 'v2'.")

    checkpoint_path = MODEL_ROOT / version / "best.pth"
    if not checkpoint_path.is_file():
        raise FileNotFoundError(
            f"Model checkpoint not found: {checkpoint_path}. Train this version first."
        )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)
    expected_mapping = {name: index for index, name in enumerate(CLASS_NAMES)}
    if checkpoint.get("class_to_idx") != expected_mapping:
        raise ValueError("Checkpoint class order is not cat=0, dog=1.")

    model = build_model(pretrained=False).to(device)
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()
    class_names = tuple(
        name for name, _ in sorted(checkpoint["class_to_idx"].items(), key=lambda item: item[1])
    )
    preprocess = transforms.Compose(
        [
            transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ]
    )
    return CatDogPredictor(model, device, class_names, preprocess)
