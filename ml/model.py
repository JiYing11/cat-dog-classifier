"""Shared model and image normalization settings for training and inference."""

from torch import nn
from torchvision.models import MobileNet_V3_Small_Weights, mobilenet_v3_small

IMAGE_SIZE = 224
CLASS_NAMES = ("cat", "dog")
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def build_model(*, pretrained: bool = True) -> nn.Module:
    """Create a two-class MobileNetV3 Small model.

    The pretrained weights provide general visual features; the final layer is
    replaced so the model predicts only cat or dog.
    """
    weights = MobileNet_V3_Small_Weights.DEFAULT if pretrained else None
    model = mobilenet_v3_small(weights=weights)
    input_features = model.classifier[-1].in_features
    model.classifier[-1] = nn.Linear(input_features, len(CLASS_NAMES))
    return model
