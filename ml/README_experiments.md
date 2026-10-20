# Experiment report

## Objective

Compare fine-tuning the same pretrained MobileNetV3 Small model with two balanced training/validation snapshots: v1 (150 images per class) and v2 (500 images per class). The separate 3,669-image test split is reserved for the final v2 evaluation.

## Method

Both training runs used the same configuration so the dataset version was the main comparison variable:

| Setting | Value |
|---|---|
| Architecture | TorchVision MobileNetV3 Small, initialized with ImageNet pretrained weights |
| Input size | 224 × 224 pixels |
| Optimizer | AdamW |
| Learning rate | 0.0001 |
| Batch size | 32 |
| Epochs | 5 |
| Random seed | 42 |
| Training device | CUDA (as recorded in MLflow) |
| Selection rule | Checkpoint with the best validation accuracy |

Training applies random resized crops and horizontal flips. Validation and test images are resized and normalized without random augmentation.

## Results

These are the latest provenance-recorded runs in the local MLflow database:

| Run | Dataset | Best validation accuracy | MLflow run ID |
|---|---|---:|---|
| `v1-mobilenet-v3-small` | v1 | 93.33% | `fd428d89882d474892c1c787f35af0ff` |
| `v2-mobilenet-v3-small` | v2 | 94.50% | `48d9a3e853ae4c30b44832d78f0e8d9e` |

The held-out v2 evaluation is recorded as run `bdc00b40fa88445e90b61c1bda320d41`:

| Metric | Result |
|---|---:|
| Test images | 3,669 |
| Accuracy | 94.06% |
| Cat precision / recall | 89.01% / 93.07% |
| Dog precision / recall | 96.63% / 94.53% |

Confusion matrix (rows are actual labels; columns are predicted labels; order is cat, dog):

```text
[[1101,  82],
 [ 136, 2350]]
```

## Interpretation and selected model

The v2 checkpoint is selected because it used the larger balanced snapshot and achieved higher validation accuracy than v1. Its held-out test accuracy is 94.06%. Per-class precision and recall are included because overall accuracy alone does not show which class is more often missed. These figures describe this held-out dataset and do not guarantee the same performance on images from other sources.

## Tracking and reproduction

MLflow tracking metadata is stored locally in `mlflow.db`; checkpoint artifacts are stored under `models/`. Both are local project artifacts and are not committed to Git. The training runs above record the code commit, clean/dirty worktree state, DVC snapshot hash, configuration, device, and metrics.

To reproduce the training runs from the repository root after configuring DVC access:

```powershell
dvc pull
python -m ml.train --version v1 --epochs 5
python -m ml.train --version v2 --epochs 5
python -m ml.evaluate --version v2
```

The evaluation script records test metrics in MLflow. Do not use the held-out test results to tune the model or choose between checkpoints.
