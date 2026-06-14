# System Scope

## Current implementation

This repository implements a modular credit risk modelling pipeline under active
hardening. The current codebase includes:

- A canonical CR-3 entry point at `scripts/train_credit_risk_model.py`, also
  exposed through `main.py`.
- CSV ingestion with configurable local data paths and train/test splitting.
- Feature preprocessing for numeric, categorical, high-cardinality, and
  date-like fields.
- Baseline model training and hyperparameter tuning scaffolding.
- A tracked fitted logistic regression model and preprocessor from a
  reproducible sample-scale run.
- Holdout metrics, evaluation plots, threshold analysis, and measured batch
  inference evidence.
- A prediction pipeline that can load a preprocessor and model when both
  artifacts are available.
- Repository integrity and transformer tests that do not require raw data.

Earlier modules under `src/components/` are retained experimentation
scaffolding. They are not the source of the committed CR-3 metrics or artifacts.

## Current boundaries

The repository does not currently include:

- Reproducible full-scale execution logs.
- A verified serving API, web interface, container image, or live deployment.
- Operational monitoring, drift detection, or automated retraining.

No production deployment is claimed, and no full-scale benchmark is claimed.
The raw LendingClub accepted-loan data is excluded from the public repository.

The current goal is a practical, modular credit risk modelling pipeline with a
clear senior-review upgrade path. Validation evidence, scale measurements,
threshold selection, and deployment hardening must be added through
reproducible runs before stronger operational claims are made.
