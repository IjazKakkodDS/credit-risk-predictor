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
sample-scale training run, holdout validation artifacts, threshold analysis,
class imbalance policy, and a genuine batch inference benchmark.

It does not yet claim full-scale benchmark execution or production deployment.
The approximately 2.26M-record figure describes the original source scale only.

## Current Evidence Snapshot

- [System scope and boundaries](docs/evidence/system_scope.md)
- [Data scale context](docs/evidence/data_scale_context.md)
- [Model validation requirements](docs/evidence/model_validation_requirements.md)
- [Class imbalance policy](docs/evidence/class_imbalance_policy.md)
- [Threshold policy](docs/evidence/threshold_policy.md)
- [Measured resource profile](docs/evidence/resource_profile.md)
- [Validation report index](reports/model_validation/README.md)
- Tracked fitted model and preprocessor artifacts under `artifact/`.

The committed run loaded a deterministic 100,000-row prefix sample and retained
87,892 resolved outcomes. It used 70,313 training rows and 17,579 test rows with
a 20.03% positive-class rate.

## Model Validation Evidence

The class-weighted logistic regression baseline produced these holdout results:

| Metric | Value |
| --- | ---: |
| ROC-AUC | 0.7359 |
| PR-AUC | 0.4209 |
| Precision at 0.50 | 0.3340 |
| Recall at 0.50 | 0.6705 |
| F1 at 0.50 | 0.4459 |

The committed evidence includes a confusion matrix, ROC curve, precision-recall
curve, calibration curve, classification report, feature contract, and
reproducible training metadata. The calibration curve indicates systematic
risk overprediction, so probability recalibration remains required.

## Threshold and Class Imbalance Policy

The baseline uses balanced class weights rather than optimizing accuracy alone.
The threshold grid found its highest observed F1 at 0.55, with F1 0.4506. This
is an analytical result for reviewer discussion, not a business-optimal
threshold. Selection still requires approved loss, false-rejection, and review
capacity assumptions.

## Batch Benchmark Evidence

A 10,000-row synthetic end-to-end benchmark measured preprocessing plus
prediction across five runs:

- Median batch time: 36.27 ms.
- Throughput at the median batch time: approximately 275,724 rows per second.

This is a local artifact benchmark, not a deployed-service or full-source
performance claim.

## Current System Scope

- Modular ingestion, transformation, and training code under `src/`.
- Custom transformers for frequency encoding and date-derived feature extraction.
- Tracked logistic regression and preprocessing artifacts under `artifact/`.
- Reproducible sample-scale validation evidence under `reports/`.
- Credit risk EDA notebook at `notebook/eda_credit_risk.ipynb`.
- Elastic Beanstalk configuration is present, but live deployment is not currently verified.
- No production service is claimed.

## Repository Structure

```text
credit-risk-predictor/
|-- artifact/
|   |-- model.pkl
|   `-- preprocessor.pkl
|-- docs/evidence/
|-- notebook/
|   `-- eda_credit_risk.ipynb
|-- reports/
|   |-- benchmark_summary.json
|   `-- model_validation/
|-- scripts/
|   |-- benchmark_inference.py
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

- `src/components/data_ingestion.py` loads the configured CSV, removes high-null columns, and writes train/test splits.
- `src/components/data_transformation.py` builds preprocessing pipelines for numeric, categorical, high-cardinality categorical, and date-like features.
- `src/components/model_trainer.py` contains baseline model training and Optuna-backed tuning scaffolding.
- `src/pipeline/predict_pipeline.py` provides a prediction pipeline entry point for future inference integration.

## Validation

Fast repository checks are included under `tests/`:

```bash
python -m pytest tests/ -v --tb=short
python -m compileall . -q
```

These tests focus on repo integrity, path hygiene, transformer behavior, and pipeline file expectations. They do not require raw data, retraining, network access, or a live service.

## Scale Roadmap

The current evidence is sample-scale. The next targets are 100K resolved rows,
500K rows, and a reproducible 1M-row run. A 2M+ run should only be presented
with runtime, memory, environment, and cost evidence.

`scripts/benchmark_inference.py` provides a synthetic batch inference scaffold.
It runs only when both fitted preprocessor and model artifacts are available and
writes `reports/benchmark_summary.json` only after prediction succeeds.

`scripts/train_credit_risk_model.py` regenerates the model, metrics, plots, and
threshold analysis from an explicitly supplied local dataset path.

## Current Gaps Before Senior Review

- No full-source training or scale benchmark.
- No memory profile, cost profile, or concurrency benchmark.
- No approved business cost matrix or operating threshold.
- No containerized API or web UI layer currently exposed.
- No temporal holdout or segment-level fairness and stability analysis yet.

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
