# Current model registry entry

## Selected model

| Field | Value |
|---|---|
| Status | Selected model for the local demo |
| Model version | v2 |
| Architecture | TorchVision MobileNetV3 Small, fine-tuned from ImageNet pretrained weights |
| Task / labels | Image classification: `cat`, `dog` |
| Input | RGB image, resized and normalized to the training configuration (224 × 224) |
| Training dataset | DVC v2, `ffdd790cb7e8efa14b9187d585e89ead.dir` |
| Training code commit | `a10b081e0d0ff6091416f09f7e0f220f544f981a` |
| Training MLflow run | `48d9a3e853ae4c30b44832d78f0e8d9e` (`v2-mobilenet-v3-small`) |
| Checkpoint | `models/v2/best.pth` (local; intentionally excluded from Git) |
| Best validation accuracy | 94.50% |
| Held-out test accuracy | 94.06% on 3,669 images |
| Test MLflow run | `bdc00b40fa88445e90b61c1bda320d41` (`v2-held-out-test`) |
| Inference code | `app/predictor.py` |

## Selection rationale

The v2 run used 500 images per class, compared with 150 per class for v1, and achieved higher best validation accuracy (94.50% versus 93.33%). The v2 checkpoint then achieved 94.06% accuracy on the held-out test split. See [`README_experiments.md`](../README_experiments.md) for class-level metrics and the confusion matrix.

## Availability and reproduction

The checkpoint is a local artifact and is not included in Git or the Docker image. For a fresh checkout, retrieve the dataset with DVC, install project dependencies, and train v2 using the command in [`PROVENANCE.md`](../PROVENANCE.md). The Streamlit app expects the checkpoint at `models/v2/best.pth` when v2 is selected.

This file is a lightweight project registry entry, not an MLflow Model Registry deployment. The model is selected for the course demonstration; no external production service is deployed.
