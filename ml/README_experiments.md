# Experiment summary

## Experiment design

Both runs fine-tuned TorchVision MobileNetV3 Small with ImageNet pretrained weights, 224 x 224 inputs, AdamW, learning rate `0.0001`, batch size `32`, seed `42`, and five epochs. The comparison changes the training dataset snapshot: v1 has 300 training/validation images total and v2 has 1,000. MLflow logs hyperparameters, per-epoch training/validation loss and accuracy, the best validation accuracy, and the saved checkpoint artifact.

The training runs are named `v1-mobilenet-v3-small` and `v2-mobilenet-v3-small` in the local `cat-dog-classifier` MLflow experiment. The held-out evaluation run is named `v2-held-out-test`.

## Results and selected model

- v1 best validation accuracy: `93.3%`.
- v2 best validation accuracy: `95.0%`.
- v2 held-out test accuracy: `94.06%` (3,669 images).
- v2 cat precision/recall: `89.1%` / `93.0%`.
- v2 dog precision/recall: `96.6%` / `94.6%`.

**Selected model: v2.** It uses the larger balanced training snapshot and achieved higher best validation accuracy than v1. Its held-out test accuracy is strong, and class-specific recall shows that it recognizes both classes well. Accuracy is the main summary metric because the app returns one of two labels; per-class precision and recall are included because the held-out test split is not class balanced and overall accuracy alone can hide class-specific errors.

## Reproducibility

MLflow metadata is stored locally in `mlflow.db`; start the UI with `mlflow server --backend-store-uri sqlite:///mlflow.db --host 127.0.0.1 --port 5000`. Checkpoints are stored under the ignored `models/` directory and are not included in Git. The current training script also records the Git commit, whether the Git working tree was dirty, and the DVC directory hash. The original v1/v2 runs predate those provenance fields; after committing the provenance update, rerun both training commands to create fully traceable runs.
