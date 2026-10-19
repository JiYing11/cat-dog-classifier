# Model and data provenance

## Dataset

Images come from the Oxford-IIIT Pet dataset via TorchVision's `OxfordIIITPet` loader. `binary-category` supplies the cat/dog labels. The v1 and v2 training snapshots are balanced samples from the train/validation source split; the official test split is held out. Snapshot details, hashes, and the Git commit that added the DVC pointers are in [`README_data.md`](README_data.md).

## Model

The classifier uses TorchVision MobileNetV3 Small initialized with the official ImageNet pretrained weights, then replaces the final classifier layer with two outputs (`cat`, `dog`) and fine-tunes on the selected DVC dataset version. The checkpoint stores the architecture, weights, class mapping, preprocessing parameters, dataset version, and (for runs made with the current script) the code commit and DVC hash.

## Reproduction

1. Check out the recorded Git commit and run `dvc pull` to retrieve the selected data snapshot from Azure.
2. Install `requirements.txt` in the project virtual environment.
3. Run `python -m ml.train --version v2 --epochs 5`.
4. Use `python -m ml.evaluate --version v2` to evaluate once on the held-out test split.

Training logs parameters, epoch metrics, and a checkpoint artifact to the local MLflow database. Do not use the held-out test split for fitting or model selection.
