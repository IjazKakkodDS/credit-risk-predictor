# System Scope

## Current implementation

This repository implements a modular credit risk modelling pipeline under active
hardening. The current codebase includes:

- CSV ingestion with configurable local data paths and train/test splitting.
- Feature preprocessing for numeric, categorical, high-cardinality, and
  date-like fields.
- Baseline model training and hyperparameter tuning scaffolding.
- A tracked fitted preprocessor artifact.
- A prediction pipeline that can load a preprocessor and model when both
  artifacts are available.
- Repository integrity and transformer tests that do not require raw data.

## Current boundaries

The repository does not currently include:

- A committed trained model artifact.
- Reproducible full-scale execution logs.
- Committed model evaluation plots or metric reports.
- A verified serving API, web interface, container image, or live deployment.
- Operational monitoring, drift detection, or automated retraining.

No production deployment is claimed, and no full-scale benchmark is claimed.
The raw LendingClub accepted-loan data is excluded from the public repository.

The current goal is a practical, modular credit risk modelling pipeline with a
clear senior-review upgrade path. Validation evidence, scale measurements,
threshold selection, and deployment hardening must be added through
reproducible runs before stronger operational claims are made.
