# Cat and Dog Image Classifier

A course project for classifying an uploaded image as a cat or a dog. The project will demonstrate application development, testing, Git-based collaboration, Docker, GitHub Actions, and basic MLOps with DVC and MLflow.

## Project plan

- `app/` — the image classification application
- `ml/` — data preparation, training, and experiment notes
- `tests/` — application and prediction tests
- `.github/workflows/` — automated checks and image builds

Training data and generated model files will not be committed directly to Git. Dataset versions will be tracked with DVC.

## Run locally

Create and activate a virtual environment, install `requirements.txt`, then run:

```powershell
python -m streamlit run app/main.py
```

The current milestone validates and previews JPEG/PNG uploads. Model training and predictions are the next step.
