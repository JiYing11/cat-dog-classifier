"""Download Oxford-IIIT Pet and create reproducible balanced dataset versions."""

from pathlib import Path
import random

from torchvision.datasets import OxfordIIITPet


ROOT = Path(__file__).resolve().parents[1]
DATA_ROOT = ROOT / "data"
SOURCE_ROOT = DATA_ROOT / "source"
VERSIONS_ROOT = DATA_ROOT / "versions"
SEED = 42
VERSION_SIZES = {"v1": 150, "v2": 500}


def _copy_image(image, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    image.convert("RGB").save(destination, format="JPEG", quality=95)


def prepare_datasets() -> None:
    """Download the source data and write train/validation/test image folders."""
    for version in VERSION_SIZES:
        target = VERSIONS_ROOT / version
        if target.exists():
            raise FileExistsError(
                f"{target} already exists. Move it aside before regenerating datasets."
            )

    trainval = OxfordIIITPet(
        root=SOURCE_ROOT,
        split="trainval",
        target_types="binary-category",
        download=True,
    )
    test = OxfordIIITPet(
        root=SOURCE_ROOT,
        split="test",
        target_types="binary-category",
        download=True,
    )

    by_class: dict[str, list[tuple[int, object]]] = {"cat": [], "dog": []}
    for index in range(len(trainval)):
        image, label = trainval[index]
        class_name = "cat" if int(label) == 0 else "dog"
        by_class[class_name].append((index, image))

    rng = random.Random(SEED)
    for images in by_class.values():
        rng.shuffle(images)

    for version, per_class in VERSION_SIZES.items():
        version_root = VERSIONS_ROOT / version
        for class_name, images in by_class.items():
            selected = images[:per_class]
            if len(selected) < per_class:
                raise RuntimeError(f"Not enough {class_name} images to create {version}.")
            train_count = int(per_class * 0.8)
            for position, (source_index, image) in enumerate(selected):
                split = "train" if position < train_count else "val"
                _copy_image(
                    image,
                    version_root / split / class_name / f"{source_index:05d}.jpg",
                )

    test_root = DATA_ROOT / "test"
    if test_root.exists():
        raise FileExistsError(f"{test_root} already exists; refusing to overwrite it.")
    for index in range(len(test)):
        image, label = test[index]
        class_name = "cat" if int(label) == 0 else "dog"
        _copy_image(image, test_root / class_name / f"{index:05d}.jpg")

    print("Created balanced v1 and v2 datasets plus the held-out test set.")
    print("v1: 150 images per class; v2: 500 images per class.")
    print(f"Dataset files are under: {DATA_ROOT}")


if __name__ == "__main__":
    prepare_datasets()
