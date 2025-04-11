# src/utils.py

import os
import sys
import pickle
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score
from src.logger import logging
from src.exception import CustomException
import optuna

def save_object(file_path, obj):
    try:
        dir_path = os.path.dirname(file_path)
        os.makedirs(dir_path, exist_ok=True)
        with open(file_path, "wb") as file_obj:
            pickle.dump(obj, file_obj)
    except Exception as e:
        raise CustomException(e, sys)

def load_object(file_path):
    try:
        with open(file_path, "rb") as file_obj:
            return pickle.load(file_obj)
    except Exception as e:
        raise CustomException(e, sys)

def tune_single_model(model_name, model, param_dict, 
                      X_train_sample, y_train_sample, 
                      X_test, y_test, full_X_train, full_y_train, n_trials):
    """
    Tune hyperparameters for a single model using Optuna with a MedianPruner.
    Returns a tuple: (model_name, best_test_accuracy, tuned_estimator).
    """
    try:
        def objective(trial):
            logging.info(f"Model {model_name}: Starting trial {trial.number}")
            # Build the parameter dictionary using trial suggestions.
            current_params = {key: trial.suggest_categorical(key, values)
                              for key, values in param_dict.items()}
            model.set_params(**current_params)
            model.fit(X_train_sample, y_train_sample)
            y_pred = model.predict(X_test)
            test_acc = accuracy_score(y_test, y_pred)
            logging.info(f"Model {model_name}: Trial {trial.number} finished with test accuracy {test_acc:.4f}")
            return 1 - test_acc  # We minimize 1 - accuracy

        # Create a study with a MedianPruner to stop unpromising trials.
        pruner = optuna.pruners.MedianPruner(n_startup_trials=1, n_warmup_steps=1)
        study = optuna.create_study(direction="minimize", pruner=pruner)
        study.optimize(objective, n_trials=n_trials)
        best_params = study.best_trial.params
        logging.info(f"Model {model_name}: Best parameters found: {best_params}")

        # Retrain on the full training data with the best parameters.
        model.set_params(**best_params)
        model.fit(full_X_train, full_y_train)

        # Evaluate on full training and test sets.
        y_train_pred = model.predict(full_X_train)
        y_test_pred = model.predict(X_test)
        train_acc = accuracy_score(full_y_train, y_train_pred)
        test_acc = accuracy_score(y_test, y_test_pred)
        weighted_f1 = f1_score(y_test, y_test_pred, average="weighted")
        logging.info(
            f"Model {model_name}: Tuned final evaluation -> Train Accuracy: {train_acc:.4f}, "
            f"Test Accuracy: {test_acc:.4f}, Weighted F1: {weighted_f1:.4f}"
        )
        return (model_name, test_acc, model)
    except Exception as e:
        logging.error(f"Model {model_name} failed during tuning: {e}")
        return (model_name, None, None)

def evaluate_models(X_train, y_train, X_test, y_test, models, param, 
                    sample_ratio=1.0, n_trials=10):
    """
    Sequentially evaluate candidate models using hyperparameter tuning with Optuna.
    Uses a subset of the training data (controlled by sample_ratio) for fast tuning.
    
    Returns:
      - report: dict mapping model names to tuned test accuracy.
      - best_estimators: dict mapping model names to the final, retrained estimator.
    """
    try:
        report = {}
        best_estimators = {}
        
        if sample_ratio < 1.0:
            sample_size = int(sample_ratio * X_train.shape[0])
            sample_indices = np.random.choice(X_train.shape[0], sample_size, replace=False)
            X_train_sample = X_train[sample_indices]
            y_train_sample = y_train[sample_indices]
            logging.info(f"Using {sample_size} samples ({sample_ratio*100:.0f}%) of training data for tuning.")
        else:
            X_train_sample = X_train
            y_train_sample = y_train

        for model_name, model in models.items():
            logging.info(f"Starting hyperparameter tuning for model: {model_name}")
            name, test_acc, tuned_model = tune_single_model(
                model_name, model, param[model_name],
                X_train_sample, y_train_sample,
                X_test, y_test,
                X_train, y_train,
                n_trials
            )
            if test_acc is not None:
                report[name] = test_acc
                best_estimators[name] = tuned_model
            else:
                logging.error(f"Model {model_name} did not return a valid test accuracy.")
        return report, best_estimators

    except Exception as e:
        raise CustomException(e, sys)

def detect_outliers_iqr(df, col):
    """
    Detect outliers in a numeric column using the IQR method.
    Returns indices of outlier rows.
    """
    try:
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        outliers_index = df[(df[col] < lower_bound) | (df[col] > upper_bound)].index
        return outliers_index
    except Exception as e:
        raise CustomException(e, sys)
