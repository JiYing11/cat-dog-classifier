# Dataset versions

## Source

The project uses the Oxford-IIIT Pet dataset through TorchVision's `OxfordIIITPet` loader. Its `binary-category` target provides the cat/dog labels directly. The loader downloads the `trainval` and `test` splits into the ignored `data/source/` directory.

Source documentation: <https://docs.pytorch.org/vision/main/generated/torchvision.datasets.OxfordIIITPet.html>

## Versions

- **v1:** a deterministic, balanced sample of 150 training/validation images per class, with an 80/20 train/validation split.
- **v2:** a deterministic, balanced sample of 500 training/validation images per class, with the same split. This version adds more training examples while keeping class counts equal.
- **Test set:** the source dataset's separate test split, kept outside the training and validation folders.

`python -m ml.data_pipeline` creates these directories. Raw data and generated images are ignored by Git and will be tracked with DVC. Dataset DVC hashes and the Git commit that records each version will be added here after DVC is initialized and the data is generated.
