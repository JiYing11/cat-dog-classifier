# Current model

- **Selected version:** v2
- **Architecture:** MobileNetV3 Small, fine-tuned from ImageNet pretrained weights
- **Dataset snapshot:** DVC v2 (`ffdd790cb7e8efa14b9187d585e89ead.dir`)
- **Selection evidence:** 95.0% best validation accuracy; 94.06% accuracy on the held-out test set of 3,669 images
- **Checkpoint path:** `models/v2/best.pth` (local and ignored by Git)
- **Inference:** `app/predictor.py` loads this checkpoint; the Streamlit selector also supports v1

The checkpoint is not committed to Git. Recreate it by pulling v2 with DVC and running the training command in `ml/PROVENANCE.md`.
