# Dataset versions

## Source

The project uses the Oxford-IIIT Pet dataset through TorchVision's `OxfordIIITPet` loader. Its `binary-category` target provides the cat/dog labels directly. The loader downloads the `trainval` and `test` splits into the ignored `data/source/` directory.

Source documentation: <https://docs.pytorch.org/vision/main/generated/torchvision.datasets.OxfordIIITPet.html>

## Versions

- **v1:** a deterministic, balanced sample of 150 training/validation images per class, with an 80/20 train/validation split.
- **v2:** a deterministic, balanced sample of 500 training/validation images per class, with the same split. This version adds more training examples while keeping class counts equal.
- **Test set:** the source dataset's separate test split, kept outside the training and validation folders.

`python -m ml.data_pipeline` creates these directories. Raw data and generated images are ignored by Git; DVC stores their contents in the configured Azure Blob remote. Git stores only the small pointer files.

## DVC identities and Git history

Both dataset snapshots and the held-out test pointer were added in commit [`9991150`](https://github.com/JiYing11/cat-dog-classifier/commit/9991150329bf27d9953fedf14abe926fdd2966d1).

| Snapshot | DVC directory hash | Pointer file | Git commit |
|---|---|---|---|
| v1 | `75bc3f1e504eae121ef915692acd54d8.dir` | `data/versions/v1.dvc` | `9991150` |
| v2 | `ffdd790cb7e8efa14b9187d585e89ead.dir` | `data/versions/v2.dvc` | `9991150` |
| Held-out test | `a68dca8d58ea1f8b0e0c13a9bec2ab3b.dir` | `data/test.dvc` | `9991150` |

These DVC hashes identify the exact directory contents. Training runs created before automatic provenance logging was added recorded the dataset name (`v1` or `v2`); rerun training with the current `ml/train.py` to also record the Git commit and DVC directory hash in MLflow and in each checkpoint.
