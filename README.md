# Credit Risk Predictor

Credit Risk Predictor is a modular credit default modelling pipeline built around LendingClub-style loan data, with preprocessing, feature engineering, model training scaffolding, and a deployment-oriented project structure. The current public repo contains the cleaned pipeline skeleton and preprocessing artifact, while committed full-scale training logs, evaluation artifacts, and deployment hardening remain planned next steps.

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

## Current System Scope

- Modular ingestion, transformation, and training code under `src/`.
- Custom transformers for frequency encoding and date-derived feature extraction.
- Tracked preprocessing artifact at `artifact/preprocessor.pkl`.
- Credit risk EDA notebook at `notebook/eda_credit_risk.ipynb`.
- Elastic Beanstalk configuration is present, but live deployment is not currently verified.
- No production service is claimed.

## Repository Structure

```text
credit-risk-predictor/
|-- artifact/
|   `-- preprocessor.pkl
|-- notebook/
|   `-- eda_credit_risk.ipynb
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

## Current Gaps Before Senior Review

- No committed full-scale training log.
- No committed confusion matrix, ROC, PR, calibration, or threshold artifacts.
- No class imbalance and threshold policy documentation yet.
- No containerized API or web UI layer currently exposed.
- Limited tests added for repo integrity and transformer validation.

## Roadmap

- Add traceable evaluation artifacts from reproducible training runs.
- Document class imbalance handling, threshold policy, and calibration strategy.
- Add model cards or experiment summaries tied to committed artifacts.
- Harden inference packaging after evaluation evidence is committed.
- Expand tests around pipeline contracts and serialization compatibility.

## Contact

- Email: [ijazkakkod@gmail.com](mailto:ijazkakkod@gmail.com)
- LinkedIn: [linkedin.com/in/ijazkakkod](https://linkedin.com/in/ijazkakkod)
- GitHub: [github.com/IjazKakkodDS](https://github.com/IjazKakkodDS)

## License

This project is licensed under the [MIT License](LICENSE.txt).
