# src/model_trainer.py

import os
import sys
import warnings
from dataclasses import dataclass
from sklearn.base import clone
from sklearn.metrics import accuracy_score
from catboost import CatBoostClassifier
from xgboost import XGBClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.exceptions import ConvergenceWarning
from src.exception import CustomException
from src.logger import logging
from src.utils import save_object, evaluate_models, tune_single_model

@dataclass
class ModelTrainerConfig:
    model_path: str = os.path.join("artifact", "model.pkl")

class ModelTrainer:
    def __init__(self):
        self.config = ModelTrainerConfig()

    def initiate_model_trainer(self, train_arr, test_arr):
        try:
            logging.info("Starting model training process.")

            # Split full arrays into features (X) and target (y)
            X_train, y_train = train_arr[:, :-1], train_arr[:, -1]
            X_test, y_test = test_arr[:, :-1], test_arr[:, -1]

            # Optional: suppress convergence warnings if desired.
            # warnings.filterwarnings("ignore", category=ConvergenceWarning)

            logging.info("Defining candidate models and parameter grids.")

            models = {
                "LogisticRegression": LogisticRegression(solver='saga', max_iter=3000, n_jobs=-1),
                "DecisionTree": DecisionTreeClassifier(),
                "RandomForest": RandomForestClassifier(n_jobs=-1),
                "GradientBoosting": GradientBoostingClassifier(),
                "XGBoost": XGBClassifier(use_label_encoder=False, eval_metric='logloss', n_jobs=-1),
                "CatBoost": CatBoostClassifier(verbose=False),
                "AdaBoost": AdaBoostClassifier()
            }

            params = {
                "LogisticRegression": {
                    'C': [0.1, 1.0],
                    'max_iter': [1000, 2000, 3000]
                },
                "DecisionTree": {
                    'criterion': ['gini', 'entropy'],
                    'max_depth': [4, 6, 8]
                },
                "RandomForest": {
                    'n_estimators': [50, 100],
                    'max_depth': [5, 10]
                },
                "GradientBoosting": {
                    'learning_rate': [0.01, 0.1],
                    'n_estimators': [50, 100]
                },
                "XGBoost": {
                    'learning_rate': [0.01, 0.1],
                    'n_estimators': [50, 100]
                },
                "CatBoost": {
                    'depth': [4, 6],
                    'learning_rate': [0.01, 0.1],
                    'iterations': [50, 100]
                },
                "AdaBoost": {
                    'learning_rate': [0.01, 0.1],
                    'n_estimators': [50, 100]
                }
            }

            # First, compute baseline (untuned) performance for each model.
            baseline_results = {}
            for model_name, model in models.items():
                logging.info(f"Evaluating baseline performance for model {model_name}")
                baseline_model = clone(model)
                baseline_model.fit(X_train, y_train)
                y_train_pred = baseline_model.predict(X_train)
                y_test_pred = baseline_model.predict(X_test)
                base_train_acc = accuracy_score(y_train, y_train_pred)
                base_test_acc = accuracy_score(y_test, y_test_pred)
                logging.info(f"Baseline {model_name}: Train Acc: {base_train_acc:.4f}, Test Acc: {base_test_acc:.4f}")
                baseline_results[model_name] = (base_train_acc, base_test_acc)

            # Use baseline mode settings for tuning.
            baseline_mode = True
            if baseline_mode:
                sample_ratio = 0.01  # Use 1% of data
                n_trials = 2         # Use 2 trials per model for a quick baseline
                logging.info("Baseline mode ON: Using 1% sample and 2 trials per model for tuning.")
            else:
                sample_ratio = 0.1
                n_trials = 10

            # Perform hyperparameter tuning using Optuna.
            logging.info("Starting hyperparameter tuning with Optuna (sequential).")
            tuned_report, tuned_estimators = evaluate_models(
                X_train=X_train, y_train=y_train,
                X_test=X_test, y_test=y_test,
                models=models, param=params,
                sample_ratio=sample_ratio,
                n_trials=n_trials
            )

            tuned_results = {}
            for model_name, tuned_model in tuned_estimators.items():
                y_train_pred_tuned = tuned_model.predict(X_train)
                y_test_pred_tuned = tuned_model.predict(X_test)
                tuned_train_acc = accuracy_score(y_train, y_train_pred_tuned)
                tuned_test_acc = accuracy_score(y_test, y_test_pred_tuned)
                logging.info(f"Tuned {model_name}: Train Acc: {tuned_train_acc:.4f}, Test Acc: {tuned_test_acc:.4f}")
                tuned_results[model_name] = (tuned_train_acc, tuned_test_acc)

            # Log a side-by-side comparison of baseline and tuned performance.
            logging.info("Comparison of Baseline vs. Tuned Performance:")
            for m in models.keys():
                base_train, base_test = baseline_results.get(m, (None, None))
                tuned_train, tuned_test = tuned_results.get(m, (None, None))
                logging.info(f"{m}: Baseline -> Train: {base_train:.4f}, Test: {base_test:.4f} | Tuned -> Train: {tuned_train:.4f}, Test: {tuned_test:.4f}")

            # Select best model based on tuned test accuracy.
            best_model_name = max(tuned_results, key=lambda k: tuned_results[k][1])
            best_model_score = tuned_results[best_model_name][1]
            best_model = tuned_estimators[best_model_name]
            logging.info(f"Selected Best Model (Tuned): {best_model_name} with Test Accuracy: {best_model_score:.4f}")

            if best_model_score < 0.65:
                logging.warning("No model found above 0.65 accuracy threshold.")

            save_object(self.config.model_path, best_model)
            logging.info(f"Saved best model to {self.config.model_path}")

            return best_model_score

        except Exception as e:
            raise CustomException(e, sys)
