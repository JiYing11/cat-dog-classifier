# Build, test, and delivery flow

## Branch path

Use short-lived feature branches for individual changes. Open a pull request from a feature branch into `dev`; after CI succeeds, merge it to `dev`. Open a pull request from `dev` to `staging` for demonstration review, and from `staging` to `main` for the final release:

`feature/*` -> `dev` -> `staging` -> `main`

## GitHub Actions

The workflow in `.github/workflows/ci.yml` runs on pull requests whose target is `dev`, `staging`, or `main`. It installs Python dependencies, runs the full current pytest suite, checks lint and formatting with Ruff, and builds the Docker image. Pushes to `dev`, `staging`, and `main` run the same checks and publish an image tagged with the branch name to `ghcr.io/jiying11/cat-dog-classifier` using the workflow's short-lived `GITHUB_TOKEN`.

The app currently has no runtime secrets, so staging and production do not need separate `.env` files. The branch tags are `:dev`, `:staging`, and `:main`; the selected model checkpoint remains outside the image and is mounted at run time.

## Local demonstration

The published image can be pulled from GHCR and run with the local checkpoint directory mounted. On the project computer, use `docker pull ghcr.io/jiying11/cat-dog-classifier:dev`, then `docker run -d --name cat-dog-demo -p 127.0.0.1:8502:8501 --mount "type=bind,source=$($PWD.Path)\models,target=/app/models,readonly" ghcr.io/jiying11/cat-dog-classifier:dev`. Open `http://localhost:8502`. Stop it with `docker stop cat-dog-demo`.

## Repository settings still required

In GitHub repository settings, enable branch protection or a ruleset for `dev`, `staging`, and `main`. Require the CI checks to pass before merging pull requests. This setting is managed by GitHub and cannot be enforced by the workflow file alone.
