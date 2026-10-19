"""Train one dataset version and record the run in local MLflow tracking."""

import argparse
import random
from pathlib import Path

import mlflow
import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from ml.model import (
    CLASS_NAMES,
    IMAGE_SIZE,
    IMAGENET_MEAN,
    IMAGENET_STD,
    build_model,
)


ROOT = Path(__file__).resolve().parents[1]
VERSIONS_ROOT = ROOT / "data" / "versions"
MODEL_ROOT = ROOT / "models"
SEED = 42


def make_loaders(version: str, batch_size: int) -> tuple[DataLoader, DataLoader]:
    """Read the version's train/val folders and apply consistent transforms."""
    train_transform = transforms.Compose(
        [
            transforms.RandomResizedCrop(IMAGE_SIZE, scale=(0.75, 1.0)),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ]
    )
    eval_transform = transforms.Compose(
        [
            transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ]
    )

    version_root = VERSIONS_ROOT / version
    train_data = datasets.ImageFolder(version_root / "train", transform=train_transform)
    val_data = datasets.ImageFolder(version_root / "val", transform=eval_transform)
    expected_mapping = {name: index for index, name in enumerate(CLASS_NAMES)}
    if train_data.class_to_idx != expected_mapping or val_data.class_to_idx != expected_mapping:
        raise ValueError(
            f"Expected class folders {expected_mapping}; got train={train_data.class_to_idx}, "
            f"val={val_data.class_to_idx}. Check the dataset directory names."
        )

    generator = torch.Generator().manual_seed(SEED)
    train_loader = DataLoader(
        train_data,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
        pin_memory=torch.cuda.is_available(),
        generator=generator,
    )
    val_loader = DataLoader(
        val_data,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=torch.cuda.is_available(),
    )
    return train_loader, val_loader


def run_epoch(
    model: nn.Module,
    loader: DataLoader,
    loss_function: nn.Module,
    device: torch.device,
    optimizer: torch.optim.Optimizer | None = None,
) -> tuple[float, float]:
    """Run one training or validation pass and return mean loss and accuracy."""
    is_training = optimizer is not None
    model.train(is_training)
    total_loss = 0.0
    correct = 0
    total = 0

    with torch.set_grad_enabled(is_training):
        for images, labels in loader:
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            if is_training:
                optimizer.zero_grad(set_to_none=True)

            logits = model(images)
            loss = loss_function(logits, labels)
            if is_training:
                loss.backward()
                optimizer.step()

            batch_size = labels.size(0)
            total_loss += loss.item() * batch_size
            correct += (logits.argmax(dim=1) == labels).sum().item()
            total += batch_size

    if total == 0:
        raise ValueError("The data loader is empty; check that the dataset was pulled.")
    return total_loss / total, correct / total


def train(version: str, epochs: int, batch_size: int, learning_rate: float) -> Path:
    """Train a model for one dataset version and save its best validation state."""
    random.seed(SEED)
    torch.manual_seed(SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(SEED)

    train_loader, val_loader = make_loaders(version, batch_size)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_model(pretrained=True).to(device)
    loss_function = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    output_path = MODEL_ROOT / version / "best.pth"
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # MLflow 3 uses a database backend for experiment metadata. The SQLite
    # database stays local to this project and is excluded from Git.
    tracking_db = ROOT / "mlflow.db"
    mlflow.set_tracking_uri(f"sqlite:///{tracking_db.as_posix()}")
    mlflow.set_experiment("cat-dog-classifier")
    with mlflow.start_run(run_name=f"{version}-mobilenet-v3-small"):
        mlflow.log_params(
            {
                "dataset_version": version,
                "architecture": "mobilenet_v3_small",
                "image_size": IMAGE_SIZE,
                "epochs": epochs,
                "batch_size": batch_size,
                "learning_rate": learning_rate,
                "seed": SEED,
                "device": str(device),
                "train_images": len(train_loader.dataset),
                "validation_images": len(val_loader.dataset),
            }
        )
        best_accuracy = -1.0
        for epoch in range(1, epochs + 1):
            train_loss, train_accuracy = run_epoch(
                model, train_loader, loss_function, device, optimizer
            )
            val_loss, val_accuracy = run_epoch(model, val_loader, loss_function, device)
            mlflow.log_metrics(
                {
                    "train_loss": train_loss,
                    "train_accuracy": train_accuracy,
                    "val_loss": val_loss,
                    "val_accuracy": val_accuracy,
                },
                step=epoch,
            )
            print(
                f"{version} | epoch {epoch}/{epochs} | "
                f"train loss {train_loss:.4f}, accuracy {train_accuracy:.3f} | "
                f"val loss {val_loss:.4f}, accuracy {val_accuracy:.3f}"
            )

            if val_accuracy >= best_accuracy:
                best_accuracy = val_accuracy
                torch.save(
                    {
                        "architecture": "mobilenet_v3_small",
                        "state_dict": model.state_dict(),
                        "class_to_idx": train_loader.dataset.class_to_idx,
                        "image_size": IMAGE_SIZE,
                        "mean": IMAGENET_MEAN,
                        "std": IMAGENET_STD,
                        "dataset_version": version,
                    },
                    output_path,
                )

        mlflow.log_metric("best_val_accuracy", best_accuracy)
        mlflow.log_artifact(str(output_path), artifact_path="checkpoint")

    print(f"Best checkpoint saved to: {output_path}")
    print(f"MLflow run recorded for dataset {version}.")
    return output_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--version", choices=("v1", "v2"), required=True)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    train(args.version, args.epochs, args.batch_size, args.learning_rate)
