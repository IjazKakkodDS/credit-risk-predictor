
# Credit Risk Predictor

## Scalable Machine Learning Pipeline for Lending Club Loan Risk Classification

> A professional-grade ML pipeline for credit risk classification using Lending Club data, featuring scalable preprocessing, advanced model tuning, and deployment-ready structure.

![Python](https://img.shields.io/badge/Python-3.10%2B-blue?logo=python&logoColor=white)
![Pandas](https://img.shields.io/badge/Pandas-Data_Processing-green?logo=pandas)
![Scikit-learn](https://img.shields.io/badge/Scikit--learn-Baseline_Models-F7931E?logo=scikit-learn)
![XGBoost](https://img.shields.io/badge/XGBoost-Core_Model-FF7043?logo=xgboost)
![CatBoost](https://img.shields.io/badge/CatBoost-Boosting_Model-8A2BE2?logo=catboost)
![Optuna](https://img.shields.io/badge/Optuna-Hyperparameter_Tuning-00C49A?logo=optuna)
![Streamlit](https://img.shields.io/badge/Streamlit-App_UI-ff4b4b?logo=streamlit)
![FastAPI](https://img.shields.io/badge/FastAPI-REST_API-009688?logo=fastapi)
![Docker](https://img.shields.io/badge/Docker-Container_Ready-2496ED?logo=docker)
![AWS](https://img.shields.io/badge/AWS-Elastic_Beanstalk-F29111?logo=amazon-aws)
![License](https://img.shields.io/badge/License-MIT-lightgrey)

---

## Overview

This project delivers a full-scale, production-ready machine learning pipeline designed to predict **credit risk** using  **Lending Club loan data** . The pipeline streamlines ingestion, preprocessing, training (baseline and tuned), evaluation, and deployment into a web-ready application.

---

## Problem Statement

Lending Club datasets contain detailed loan and borrower attributes. The key objective is to build a high-performing binary classifier that predicts loan default risk—critical for lenders, risk analysts, and investors.

---

## Project Objectives

* ✅  **Data Ingestion** : Load and sanitize data, removing high-null features.
* ✅  **Preprocessing** : Handle missing values, outliers, and encode categorical variables (incl. frequency encoding).
* ✅  **Modeling** : Train and benchmark multiple classifiers:
  * Logistic Regression
  * Decision Tree
  * Random Forest
  * XGBoost
  * CatBoost
  * AdaBoost
* ✅  **Hyperparameter Tuning** : Leverage **Optuna** for scalable tuning with early stopping and pruning.
* ✅  **Evaluation** : Benchmark baseline vs tuned models.
* ✅  **Deployment** : Deploy the optimal model through a **Streamlit or FastAPI** interface.

---

## Dataset Summary

* **Source** : Lending Club Accepted Loans Dataset
* **Size** : ~2.26M rows × 151 columns
* **Post-Cleaning** : ~1.8M training samples, 452K test samples
* **Features** : ~177 engineered after transformations

---

## Key Technical Challenges & Solutions

| Challenge                         | Solution                                                                 |
| --------------------------------- | ------------------------------------------------------------------------ |
| Large Dataset Size                | Subsampling, chunked loading, and memory profiling                       |
| WinError 1450 (Windows Multiproc) | Implemented fallback to sequential mode                                  |
| Tuning Overhead                   | Reduced sample ratio, trial caps, pruning, and lightweight baseline mode |

---

## Accomplishments

### Modular Data Ingestion

* Loads, sanitizes, and splits raw CSV data into structured subsets.

### Feature Transformation Pipeline

* Handles:
  * Outlier capping
  * Null imputation
  * One-hot & frequency encoding
  * Temporal feature extraction

### Model Training Engine

* Scikit-learn for baseline benchmarking
* XGBoost & CatBoost for advanced boosting
* Optuna integration for tuning

### Evaluation Suite

* Accuracy, AUC, F1, and Confusion Matrix logged for every model
* Performance plotted and summarized for reproducibility

### Deployment (In Progress)

* Code structured for app integration (Streamlit / FastAPI)
* Docker-compatible structure for local/cloud deploy

---

## Next Steps

* ✅ Expand tuning to full dataset (incrementally)
* ✅ Integrate SHAP/XAI module for post-hoc model interpretation
* ✅ Complete deployment with containerized API & frontend app
* ✅ Monitor model drift & establish retraining pipeline

---

## Deployment Strategy (In Progress)

| Layer       | Stack                       | Purpose                            |
| ----------- | --------------------------- | ---------------------------------- |
| App UI      | Streamlit / FastAPI         | Collect input, display predictions |
| Backend API | Flask / FastAPI             | Model inference & preprocessing    |
| Container   | Docker                      | Platform-agnostic packaging        |
| Cloud Host  | AWS Elastic Beanstalk / GCP | Web-accessible deployment          |
| Monitoring  | Logging + Alerts            | For errors, usage, & performance   |

---

## Performance Snapshot

| Model             | Tuning        | Accuracy        | ROC-AUC        | Training Time | Notes                      |
| ----------------- | ------------- | --------------- | -------------- | ------------- | -------------------------- |
| Logistic Reg      | No            | 82.3%           | 0.71           | Fast          | Strong, simple baseline    |
| Random Forest     | No            | 84.2%           | 0.76           | Slow          | Good performance, slow     |
| **XGBoost** | **Yes** | **85.9%** | **0.79** | Medium        | **Best performance** |

---

## Tech Stack

**Languages & Frameworks:**

* Python 3.10+
* Scikit-learn, XGBoost, CatBoost, Optuna
* Streamlit / FastAPI / Flask (optional backend)

**DevOps & Deployment:**

* Docker
* AWS (Elastic Beanstalk), Render, or Heroku

**Logging & Monitoring:**

* Python logger
* Optional: Prometheus, Sentry

**Version Control:**

* Git & GitHub

---

## Repository Structure

```bash
credit-risk-predictor/
├── data/                  # Raw and processed datasets
├── src/                   # Ingestion, transformation, training scripts
├── models/                # Saved model binaries
├── logs/                  # Pipeline & evaluation logs
├── app/                   # Streamlit or FastAPI web interface
├── Dockerfile             # For containerized deployment
├── requirements.txt       # Python dependencies
└── README.md              # Documentation
```

---

## Run Locally (Streamlit)

```bash
# Clone repo
$ git clone https://github.com/IjazKakkodDS/credit-risk-predictor.git
$ cd credit-risk-predictor

# Setup environment
$ conda create -n loan-xai-env python=3.10 -y
$ conda activate loan-xai-env
$ pip install -r requirements.txt

# Launch web app
$ cd app
$ streamlit run app.py
```

---

## Contact

* **Email** : [ijazkakkod@gmail.com](mailto:ijazkakkod@gmail.com)
* **LinkedIn** : [linkedin.com/in/ijazkakkod](https://linkedin.com/in/ijazkakkod)
* **GitHub** : [github.com/IjazKakkodDS](https://github.com/IjazKakkodDS)

---

## TODO (Project Roadmap)

* [X] Build modular ingestion & transformation pipelines
* [X] Implement baseline ML models
* [X] Integrate Optuna-based hyperparameter tuning
* [X] Compare & log model performances
* [ ] Integrate SHAP/XAI for interpretability
* [ ] Complete deployment with Streamlit/FastAPI
* [ ] Containerize using Docker for local/cloud deploy
* [ ] Deploy to AWS (Elastic Beanstalk) or Render
* [ ] Set up monitoring, logging, and CI/CD

---

## License

This project is licensed under the [MIT License](https://chatgpt.com/c/LICENSE).
