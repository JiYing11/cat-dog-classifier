# Model and data provenance

This document records the lineage of the selected v2 model from source data to local inference. It describes the latest training and evaluation runs that were verified in the project's local MLflow database.

## 1. Source data and preparation

- **Source:** Oxford-IIIT Pet, loaded with TorchVision `OxfordIIITPet` and `target_types="binary-category"`.
- **Source partitions:** `trainval` is used to construct training/validation snapshots; the dataset's separate `test` partition is kept held out.
- **Preparation code:** `ml/data_pipeline.py`.
- **Sampling:** deterministic seed `42`; balanced cat/dog snapshots; 80/20 training/validation split.
- **Snapshots:** v1 has 150 images per class; v2 has 500 images per class; the held-out test set contains 3,669 images.
- **Git pointer commit:** `9991150329bf27d9953fedf14abe926fdd2966d1`.
- **DVC hashes:** v1 `75bc3f1e504eae121ef915692acd54d8.dir`; v2 `ffdd790cb7e8efa14b9187d585e89ead.dir`; test `a68dca8d58ea1f8b0e0c13a9bec2ab3b.dir`.
- **Remote storage:** DVC remote `azure://dvc-data/cat-dog-classifier` in account `catdoggstorage`. Credentials are deliberately not recorded here.

See [`README_data.md`](README_data.md) for dataset details and pointer paths.

## 2. Training implementation and configuration

- **Code revision:** `a10b081e0d0ff6091416f09f7e0f220f544f981a`.
- **Worktree state:** clean (`git_worktree_dirty=False`) for both verified training runs.
- **Training code:** `ml/train.py`; model definition: `ml/model.py`.
- **Architecture:** TorchVision MobileNetV3 Small initialized from ImageNet weights, with a two-class output layer.
- **Framework:** PyTorch and TorchVision.
- **Configuration:** 224 × 224 inputs, AdamW, learning rate `0.0001`, batch size `32`, 5 epochs, seed `42`.
- **Execution device:** CUDA, as recorded in MLflow.
- **Checkpoint selection:** save the checkpoint with the best validation accuracy during training.

## 3. Verified training runs

| Snapshot | MLflow run name / ID | Best validation accuracy | DVC hash |
|---|---|---:|---|
| v1 | `v1-mobilenet-v3-small` / `fd428d89882d474892c1c787f35af0ff` | 93.33% | `75bc3f1e504eae121ef915692acd54d8.dir` |
| v2 | `v2-mobilenet-v3-small` / `48d9a3e853ae4c30b44832d78f0e8d9e` | 94.50% | `ffdd790cb7e8efa14b9187d585e89ead.dir` |

The selected artifact is `models/v2/best.pth`. The checkpoint stores the architecture, weights, class mapping, preprocessing values, dataset version, Git commit, and DVC dataset hash.

## 4. Held-out evaluation

- **Evaluation code:** `ml/evaluate.py`.
- **Checkpoint:** `models/v2/best.pth`.
- **Test pointer:** `data/test.dvc`, DVC hash `a68dca8d58ea1f8b0e0c13a9bec2ab3b.dir`.
- **MLflow run:** `v2-held-out-test`, ID `bdc00b40fa88445e90b61c1bda320d41`.
- **Results:** accuracy 94.06% on 3,669 images; cat precision/recall 89.01%/93.07%; dog precision/recall 96.63%/94.53%.
- **Confusion matrix:** `[[1101, 82], [136, 2350]]`, rows actual `[cat, dog]`, columns predicted `[cat, dog]`.

The evaluation script logs the dataset version, checkpoint path, test image count, and test metrics. It does not currently log the evaluation code commit, DVC test hash, or checkpoint file hash as MLflow parameters; those identities are linked here from the verified project files and run context. The test set must remain excluded from fitting and model selection.

## 5. Local inference and demo

`app/predictor.py` loads the selected checkpoint and applies the recorded preprocessing. `app/main.py` provides the Streamlit upload interface. The Docker image packages the application runtime; model checkpoints are mounted from the local `models/` directory, so Docker does not train the model or retrieve DVC data.

## 6. Reproduction

From the repository root, after configuring Azure access for DVC and installing `requirements.txt`:

```powershell
dvc pull
python -m ml.train --version v2 --epochs 5
python -m ml.evaluate --version v2
```

Training and evaluation metadata is stored in the local `mlflow.db`; checkpoint files are stored locally under `models/`. Do not commit credentials, the local MLflow database, or checkpoint files to Git.
