# Dataset documentation

## Source dataset

The project uses the Oxford-IIIT Pet dataset, loaded with TorchVision's `OxfordIIITPet` dataset class. The `binary-category` target maps each image to one of two labels: `cat` or `dog`. The preparation script downloads the source `trainval` and `test` splits into `data/source/`. The official test split is kept separate from training and validation.

Source: [Oxford-IIIT Pet dataset loader documentation](https://docs.pytorch.org/vision/main/generated/torchvision.datasets.OxfordIIITPet.html).

## Project snapshots

`python -m ml.data_pipeline` creates balanced, reproducible snapshots using seed `42`:

| Snapshot | Images per class | Total images | Train / validation split |
|---|---:|---:|---|
| v1 | 150 | 300 | 80% / 20% within each class |
| v2 | 500 | 1,000 | 80% / 20% within each class |
| Held-out test | Source test split | 3,669 | Not used for fitting or model selection |

v2 increases the number of training/validation examples while retaining equal class counts. Since the v1 sample is contained in the larger deterministic selection used for v2, these versions are dataset-size snapshots, not independent samples. Keep the test set out of training and model selection.

## Storage and retrieval

Image files are excluded from Git. Git tracks the small DVC pointer files; DVC stores the corresponding image contents in the Azure Blob container configured for this project. The remote URL is `azure://dvc-data/cat-dog-classifier` in storage account `catdoggstorage`. Credentials must remain outside Git and must never be placed in this document.

From the repository root, `dvc pull` retrieves the tracked datasets and `dvc push` uploads changed data. The local source and generated folders are ignored by Git; the pointer files are the versioned records.

## Snapshot identities

The dataset pointers were committed in Git commit [`9991150`](https://github.com/JiYing11/cat-dog-classifier/commit/9991150329bf27d9953fedf14abe926fdd2966d1).

| Data snapshot | DVC directory hash | Git pointer | Pointer commit |
|---|---|---|---|
| v1 | `75bc3f1e504eae121ef915692acd54d8.dir` | `data/versions/v1.dvc` | `9991150` |
| v2 | `ffdd790cb7e8efa14b9187d585e89ead.dir` | `data/versions/v2.dvc` | `9991150` |
| Held-out test | `a68dca8d58ea1f8b0e0c13a9bec2ab3b.dir` | `data/test.dvc` | `9991150` |

The DVC directory hash identifies the exact contents represented by each pointer. The training runs recorded in MLflow also include their selected snapshot hash.
