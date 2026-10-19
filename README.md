# Cat and Dog Image Classifier

A course project that classifies an uploaded image as a cat or dog. It demonstrates a Streamlit application, image validation, model training and evaluation, Git, Docker, GitHub Actions, DVC dataset versions, and MLflow experiment tracking.

## Project contents

- `app/` — Streamlit upload and prediction interface.
- `ml/` — dataset preparation, MobileNetV3 training, and held-out test evaluation.
- `tests/` — upload validation tests, including invalid, corrupt, unsupported, and oversized images.
- `data/*.dvc` — DVC metadata for dataset versions and the held-out test set. The image files live in the configured Azure Blob remote.
- `models/` — locally trained checkpoints. They are intentionally excluded from Git.
- `.github/workflows/ci.yml` — runs application tests and builds the Docker image on pushes and pull requests.

## Local setup

Run these commands in PowerShell from the repository folder:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
dvc pull
```

The first command creates an isolated Python environment for this project. The second activates it in the current PowerShell window. The next two install project dependencies. `dvc pull` downloads the dataset files described by the DVC metadata; it requires a working Azure remote login/configuration.

If the model checkpoints are not present, train them before using the app:

```powershell
python -m ml.train --version v1 --epochs 5
python -m ml.train --version v2 --epochs 5
python -m ml.evaluate --version v2
```

Training records runs in the local MLflow database (`mlflow.db`) and writes checkpoints under `models/`. Evaluation uses the held-out test set and records test metrics in MLflow.

Start the web application:

```powershell
python -m streamlit run app/main.py
```

Open the local address Streamlit prints (normally `http://localhost:8501`). Select a model version, upload a JPEG or PNG, and choose **Classify**. The displayed confidence is the model's softmax probability, not a guarantee that the prediction is correct.

## View MLflow experiments

From the repository folder, run:

```powershell
mlflow server --backend-store-uri sqlite:///mlflow.db --host 127.0.0.1 --port 5000
```

Open `http://127.0.0.1:5000`. The tracking database and UI are local to this computer; dataset blobs are stored in Azure through DVC.

## Docker demonstration

Docker runs the app in a Linux container using CPU inference. First make sure the v2 checkpoint exists locally, then build and start:

```powershell
docker compose up --build
```

Open `http://localhost:8501`. Compose mounts the local `models/` directory into the container, so the checkpoints do not need to be copied into the image. Stop the app with `Ctrl+C`, then remove the stopped container with:

```powershell
docker compose down
```

Docker does not download datasets or train models. Run DVC and training on the host first. Never put Azure keys or other credentials in the image or Git.

## Dataset versions and experiments

The project tracks balanced dataset snapshots `v1` and `v2`, plus a separate held-out test set, using DVC. Training `v1` and `v2` creates separate MLflow runs for comparison. Evaluation of `v2` creates a held-out-test MLflow run with accuracy and per-class metrics. Do not use the held-out test set for training or model selection.

## Git workflow

Use `dev` for ongoing work, feature branches for isolated changes, and `staging` for a demonstration-ready integration. Merge reviewed work into `main` when it is ready to submit. Push branches and open pull requests on GitHub so Actions can report CI status.
