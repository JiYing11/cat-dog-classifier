"""Evaluate a saved checkpoint once on the held-out Oxford-IIIT Pet test set."""

import argparse
from pathlib import Path

import mlflow
import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from ml.model import CLASS_NAMES, IMAGE_SIZE, IMAGENET_MEAN, IMAGENET_STD, build_model


ROOT = Path(__file__).resolve().parents[1]
TEST_ROOT = ROOT / "data" / "test"
MODEL_ROOT = ROOT / "models"
MLFLOW_DB = ROOT / "mlflow.db"


def evaluate(version: str, batch_size: int = 32) -> None:
    """Load the selected checkpoint and report test-set classification metrics."""
    checkpoint_path = MODEL_ROOT / version / "best.pth"
    if not checkpoint_path.is_file():
        raise FileNotFoundError(
            f"Checkpoint not found: {checkpoint_path}. Train {version} first."
        )

    transform = transforms.Compose(
        [
            transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ]
    )
    test_data = datasets.ImageFolder(TEST_ROOT, transform=transform)
    expected_mapping = {name: index for index, name in enumerate(CLASS_NAMES)}
    if test_data.class_to_idx != expected_mapping:
        raise ValueError(
            f"Expected test folders {expected_mapping}; got {test_data.class_to_idx}."
        )
    if len(test_data) == 0:
        raise ValueError(f"No test images found under {TEST_ROOT}.")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)
    if checkpoint.get("class_to_idx") != expected_mapping:
        raise ValueError("Checkpoint class order does not match the test dataset.")

    model = build_model(pretrained=False).to(device)
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()
    loader = DataLoader(
        test_data,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=torch.cuda.is_available(),
    )

    confusion = torch.zeros((len(CLASS_NAMES), len(CLASS_NAMES)), dtype=torch.int64)
    with torch.inference_mode():
        for images, labels in loader:
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            predictions = model(images).argmax(dim=1)
            batch_counts = torch.bincount(
                labels * len(CLASS_NAMES) + predictions,
                minlength=len(CLASS_NAMES) ** 2,
            ).reshape(len(CLASS_NAMES), len(CLASS_NAMES))
            confusion += batch_counts.cpu()

    total = int(confusion.sum().item())
    accuracy = float(confusion.diag().sum().item()) / total
    class_metrics: dict[str, float] = {}
    for index, class_name in enumerate(CLASS_NAMES):
        actual = int(confusion[index, :].sum().item())
        predicted = int(confusion[:, index].sum().item())
        true_positive = int(confusion[index, index].item())
        class_metrics[f"test_precision_{class_name}"] = (
            true_positive / predicted if predicted else 0.0
        )
        class_metrics[f"test_recall_{class_name}"] = (
            true_positive / actual if actual else 0.0
        )

    mlflow.set_tracking_uri(f"sqlite:///{MLFLOW_DB.as_posix()}")
    mlflow.set_experiment("cat-dog-classifier")
    with mlflow.start_run(run_name=f"{version}-held-out-test"):
        mlflow.log_params(
            {
                "dataset_version": version,
                "checkpoint": checkpoint_path.relative_to(ROOT).as_posix(),
                "test_images": total,
            }
        )
        mlflow.log_metrics({"test_accuracy": accuracy, **class_metrics})

    print(f"Checkpoint: {checkpoint_path}")
    print(f"Test images: {total}")
    print(f"Accuracy: {accuracy:.4f} ({accuracy:.1%})")
    print("Confusion matrix: rows=true [cat, dog], columns=predicted [cat, dog]")
    print(confusion.tolist())
    for class_name in CLASS_NAMES:
        precision = class_metrics[f"test_precision_{class_name}"]
        recall = class_metrics[f"test_recall_{class_name}"]
        print(f"{class_name}: precision={precision:.3f}, recall={recall:.3f}")
    print(f"Test metrics recorded in MLflow run: {version}-held-out-test")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--version", choices=("v1", "v2"), default="v2")
    args = parser.parse_args()
    evaluate(args.version)
