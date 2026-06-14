# Credit Risk Predictor

Credit Risk Predictor is a modular credit default modelling pipeline built
around LendingClub-style loan data. The public repo contains a leakage-aware
baseline training path, fitted model and preprocessing artifacts, sample-scale
holdout evaluation evidence, and an inference benchmark. Full-source execution
and deployment hardening remain planned work.

![Python](https://img.shields.io/badge/Python-3.10%2B-blue?logo=python&logoColor=white)
![Pandas](https://img.shields.io/badge/Pandas-Data_Processing-green?logo=pandas)
![Scikit-learn](https://img.shields.io/badge/Scikit--learn-ML_Pipeline-F7931E?logo=scikit-learn)
![XGBoost](https://img.shields.io/badge/XGBoost-Modeling-FF7043?logo=xgboost)
![CatBoost](https://img.shields.io/badge/CatBoost-Modeling-8A2BE2?logo=catboost)
![Optuna](https://img.shields.io/badge/Optuna-Tuning-00C49A?logo=optuna)
![AWS](https://img.shields.io/badge/AWS-Elastic_Beanstalk_Config-F29111?logo=amazon-aws)
![License](https://img.shields.io/badge/License-MIT-lightgrey)

## Overview

The project focuses on credit default modelling from LendingClub-style accepted-loan data. It includes modular ingestion, preprocessing, feature transformation, model training code, and saved preprocessing support for later inference work.

The original LendingClub source contains approximately 2.26M accepted-loan records. The public repo excludes raw data and does not yet include a committed full-scale run log. Full-scale benchmarking is a planned next step.

## Senior Review Status

This repo is a modular credit risk modelling pipeline under active hardening. It
is public-safe after contamination cleanup and now includes a reproducible
temporal training, calibration and untouched-test workflow, segment diagnostics,
scale evidence through a 1M source-row prefix, and a corrected batch benchmark.

It does not yet claim full-scale benchmark execution or production deployment.
The approximately 2.26M-record figure describes the original source scale only.

## Current Evidence Snapshot

- [System scope and boundaries](docs/evidence/system_scope.md)
- [Data scale context](docs/evidence/data_scale_context.md)
- [Model validation requirements](docs/evidence/model_validation_requirements.md)
- [Class imbalance policy](docs/evidence/class_imbalance_policy.md)
- [Feature availability policy](docs/evidence/feature_availability_policy.md)
- [Model card](docs/evidence/model_card.md)
- [Reproducibility profile](docs/evidence/reproducibility_profile.md)
- [Threshold policy](docs/evidence/threshold_policy.md)
- [Measured resource profile](docs/evidence/resource_profile.md)
- [Validation report index](reports/model_validation/README.md)
- Tracked fitted model and preprocessor artifacts under `artifact/`.

The canonical run loaded a deterministic 100,000-row prefix and retained 87,892
resolved outcomes. It used 52,735 training rows, 17,578 calibration rows, and
17,579 untouched test rows with a 20.03% overall positive-class rate.

## CR-3 Scale and Validation Evidence

The CR-3 baseline excludes LendingClub grade, sub-grade, and interest rate from
pre-decision model inputs, applies Platt calibration on a separate calibration
split, and selects its analytical threshold without using the final test split.

| Metric | Value |
| --- | ---: |
| Untouched-test ROC-AUC | 0.7322 |
| Untouched-test PR-AUC | 0.4437 |
| Calibrated Brier score | 0.1470 |
| Precision at selected 0.20 threshold | 0.3432 |
| Recall at selected 0.20 threshold | 0.6843 |
| F1 at selected 0.20 threshold | 0.4572 |

The committed evidence includes a confusion matrix, ROC curve, precision-recall
curve, calibration curve, classification report, feature contract, and
reproducible provenance and checksums.

The original source file contains approximately 2.26M rows, but scale claims
are based only on completed benchmark runs. CR-3 completed these temporal
training stages:

| Requested rows | Usable resolved rows | Training seconds | Peak RSS | ROC-AUC | PR-AUC |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 100K | 87,892 | 1.0432 | 505.54 MB | 0.7322 | 0.4437 |
| 500K | 391,168 | 12.0387 | 1,790.68 MB | 0.7195 | 0.3980 |
| 1M | 571,494 | 17.2885 | 3,241.73 MB | 0.6955 | 0.3727 |

The full dataset was not run. Larger 2M, 5M, and 10M targets remain planned.

## Threshold and Class Imbalance Policy

The baseline uses balanced class weights rather than optimizing accuracy alone.
The calibration grid selected threshold 0.20 by F1, then evaluated it once on
the untouched test split. This remains analytical evidence for reviewer
discussion. Selection still requires approved loss, false-rejection, and review
capacity assumptions.

## Batch Benchmark Evidence

A corrected synthetic benchmark draws category values from fitted transformer
vocabularies and measures preprocessing plus prediction across five runs:

| Rows | p50 | p95 | Rows per second | Peak RSS |
| ---: | ---: | ---: | ---: | ---: |
| 10K | 0.0283 s | 0.0318 s | 353,865 | 156.97 MB |
| 100K | 0.1888 s | 0.1912 s | 529,658 | 178.43 MB |
| 500K | 1.2857 s | 1.8073 s | 388,899 | 264.12 MB |

This is a local artifact benchmark, not a deployed-service or full-source
performance claim.

Segment diagnostics cover grade, purpose, home ownership, state, application
type, and term where groups contain enough rows and both target classes. They
are not fairness certification.

## Current System Scope

- Modular ingestion, transformation, and training code under `src/`.
- Custom transformers for frequency encoding and date-derived feature extraction.
- Tracked logistic regression and preprocessing artifacts under `artifact/`.
- Reproducible sample-scale validation evidence under `reports/`.
- Credit risk EDA notebook at `notebook/eda_credit_risk.ipynb`.
- GitHub Actions runs tests and compilation without raw data.
- No production service is claimed.

## Repository Structure

```text
credit-risk-predictor/
|-- artifact/
|   |-- model.pkl
|   `-- preprocessor.pkl
|-- docs/evidence/
|-- .github/workflows/
|-- notebook/
|   `-- eda_credit_risk.ipynb
|-- reports/
|   |-- benchmark_summary.json
|   |-- batch_benchmark_results.json
|   |-- scale_benchmark_results.json
|   `-- model_validation/
|-- scripts/
|   |-- benchmark_inference.py
|   |-- run_scale_benchmarks.py
|   `-- train_credit_risk_model.py
|-- src/
|   |-- components/
|   |   |-- data_ingestion.py
|   |   |-- data_transformation.py
|   |   `-- model_trainer.py
|   |-- pipeline/
|   |   `-- predict_pipeline.py
|   |-- exception.py
|   |-- logger.py
|   `-- utils.py
|-- tests/
|-- requirements.txt
|-- setup.py
`-- README.md
```

## Local Setup

```bash
git clone https://github.com/IjazKakkodDS/credit-risk-predictor.git
cd credit-risk-predictor
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

Raw data is not committed. By default, ingestion looks for:

```text
data/lending_club_accepted_loans.csv
```

To use a different local data file, set:

```bash
set CREDIT_RISK_DATA_PATH=path\to\lending_club_accepted_loans.csv
```

## Pipeline Components

- `scripts/train_credit_risk_model.py` is the canonical CR-3 training,
  calibration, validation, and report-generation path.
- `main.py` delegates to the canonical trainer.
- `src/components/` retains earlier modular ingestion, transformation, and
  multi-model experimentation scaffolding for reference. It is not the source
  of the committed CR-3 evidence.
- `src/pipeline/predict_pipeline.py` provides a prediction pipeline entry point for future inference integration.

## Validation

Fast repository checks are included under `tests/`:

```bash
python -m pytest tests/ -v --tb=short
python -m compileall . -q
```

These tests focus on repo integrity, path hygiene, transformer behavior, and pipeline file expectations. They do not require raw data, retraining, network access, or a live service.

## Scale Roadmap

CR-3 completed 100K, 500K, and 1M source-prefix stages. The next possible target
is the full local source, but it should only be attempted with sufficient memory
headroom after the measured 3.24 GB peak at 1M.

`scripts/benchmark_inference.py` provides a synthetic batch inference scaffold.
It runs only when both fitted preprocessor and model artifacts are available and
writes `reports/benchmark_summary.json` only after prediction succeeds.

`scripts/train_credit_risk_model.py` regenerates the model, metrics, plots, and
threshold analysis from an explicitly supplied local dataset path.

## Current Gaps Before Senior Review

- No full-source training benchmark.
- No deployment cost or concurrency benchmark.
- No approved business cost matrix or operating threshold.
- No containerized API or web UI layer currently exposed.
- The canonical 100K temporal window is narrow and not broad multi-vintage proof.
- Segment diagnostics are not protected-class fairness validation.

## Roadmap

- Add temporal validation and out-of-time stability evidence.
- Calibrate model probabilities on an appropriate validation split.
- Add segment analysis and a model card tied to committed artifacts.
- Measure memory, concurrency behavior, and larger-scale runtime.
- Harden inference packaging after evaluation evidence is committed.
- Expand tests around pipeline contracts and serialization compatibility.

## Contact

- Email: [ijazkakkod@gmail.com](mailto:ijazkakkod@gmail.com)
- LinkedIn: [linkedin.com/in/ijazkakkod](https://linkedin.com/in/ijazkakkod)
- GitHub: [github.com/IjazKakkodDS](https://github.com/IjazKakkodDS)

## License

This project is licensed under the [MIT License](LICENSE.txt).
