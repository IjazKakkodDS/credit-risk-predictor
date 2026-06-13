"""Train and evaluate a leakage-aware credit risk baseline."""

from __future__ import annotations

import argparse
import json
import pickle
import sys
from datetime import datetime, timezone
from pathlib import Path
from time import perf_counter
from typing import Any

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.calibration import calibration_curve
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    average_precision_score,
    brier_score_loss,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA_PATH = REPO_ROOT / "data" / "lending_club_accepted_loans.csv"
DEFAULT_OUTPUT_DIR = REPO_ROOT / "reports" / "model_validation"
DEFAULT_ARTIFACT_DIR = REPO_ROOT / "artifact"

POSITIVE_LABELS = {"Charged Off", "Default"}
NEGATIVE_LABELS = {"Fully Paid"}

# Application-time fields only. Post-outcome payment, recovery, hardship, and
# settlement fields are intentionally excluded.
CANDIDATE_FEATURES = [
    "loan_amnt",
    "term",
    "int_rate",
    "installment",
    "grade",
    "sub_grade",
    "emp_length",
    "home_ownership",
    "annual_inc",
    "verification_status",
    "purpose",
    "addr_state",
    "dti",
    "delinq_2yrs",
    "fico_range_low",
    "fico_range_high",
    "inq_last_6mths",
    "open_acc",
    "pub_rec",
    "revol_bal",
    "revol_util",
    "total_acc",
    "initial_list_status",
    "collections_12_mths_ex_med",
    "application_type",
    "acc_now_delinq",
    "tot_coll_amt",
    "tot_cur_bal",
    "mort_acc",
    "pub_rec_bankruptcies",
]


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Train a reproducible LendingClub-style binary default-risk "
            "baseline and generate holdout validation evidence."
        )
    )
    parser.add_argument("--data-path", type=Path, default=DEFAULT_DATA_PATH)
    parser.add_argument("--target-column", default="loan_status")
    parser.add_argument("--sample-rows", type=int)
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--artifact-dir", type=Path, default=DEFAULT_ARTIFACT_DIR)
    return parser.parse_args(argv)


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def validate_args(args: argparse.Namespace) -> None:
    if not args.data_path.is_file():
        raise FileNotFoundError(
            f"Dataset not found: {args.data_path}. Pass --data-path explicitly "
            "or set up the documented default path."
        )
    if args.sample_rows is not None and args.sample_rows <= 0:
        raise ValueError("--sample-rows must be greater than zero.")
    if not 0 < args.test_size < 1:
        raise ValueError("--test-size must be between zero and one.")


def load_training_frame(
    data_path: Path, target_column: str, sample_rows: int | None
) -> tuple[pd.DataFrame, list[str]]:
    header = pd.read_csv(data_path, nrows=0)
    if target_column not in header.columns:
        raise ValueError(
            f"Target column '{target_column}' was not found in the dataset."
        )

    feature_columns = [
        column for column in CANDIDATE_FEATURES if column in header.columns
    ]
    if not feature_columns:
        raise ValueError("No supported application-time feature columns were found.")

    frame = pd.read_csv(
        data_path,
        usecols=feature_columns + [target_column],
        nrows=sample_rows,
        low_memory=False,
    )
    return frame, feature_columns


def map_target(target: pd.Series) -> pd.Series:
    normalized = target.astype("string").str.strip()
    mapped = pd.Series(np.nan, index=target.index, dtype="float64")
    mapped.loc[normalized.isin(POSITIVE_LABELS)] = 1
    mapped.loc[normalized.isin(NEGATIVE_LABELS)] = 0
    return mapped


def build_preprocessor(
    frame: pd.DataFrame, feature_columns: list[str]
) -> tuple[ColumnTransformer, list[str], list[str]]:
    numeric_columns = [
        column
        for column in feature_columns
        if pd.api.types.is_numeric_dtype(frame[column])
    ]
    categorical_columns = [
        column for column in feature_columns if column not in numeric_columns
    ]

    numeric_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )
    categorical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            (
                "one_hot",
                OneHotEncoder(handle_unknown="ignore", min_frequency=5),
            ),
        ]
    )
    preprocessor = ColumnTransformer(
        transformers=[
            ("numeric", numeric_pipeline, numeric_columns),
            ("categorical", categorical_pipeline, categorical_columns),
        ]
    )
    return preprocessor, numeric_columns, categorical_columns


def threshold_metrics(
    y_true: pd.Series, probabilities: np.ndarray
) -> dict[str, Any]:
    evaluations = []
    for threshold in np.arange(0.05, 1.0, 0.05):
        predictions = (probabilities >= threshold).astype(int)
        tn, fp, fn, tp = confusion_matrix(
            y_true, predictions, labels=[0, 1]
        ).ravel()
        evaluations.append(
            {
                "threshold": round(float(threshold), 2),
                "precision": float(
                    precision_score(y_true, predictions, zero_division=0)
                ),
                "recall": float(recall_score(y_true, predictions, zero_division=0)),
                "f1": float(f1_score(y_true, predictions, zero_division=0)),
                "false_positive_count": int(fp),
                "false_negative_count": int(fn),
                "predicted_positive_rate": float(predictions.mean()),
            }
        )

    best_f1 = max(evaluations, key=lambda item: item["f1"])
    high_recall_candidates = [
        item for item in evaluations if item["recall"] >= 0.80
    ]
    high_recall = (
        max(high_recall_candidates, key=lambda item: item["threshold"])
        if high_recall_candidates
        else None
    )
    balanced = min(
        evaluations,
        key=lambda item: abs(item["precision"] - item["recall"]),
    )
    return {
        "policy_status": (
            "Analytical threshold evidence only. A business-optimal threshold "
            "requires an approved loss and review-cost matrix."
        ),
        "threshold_grid": evaluations,
        "best_f1_threshold": best_f1,
        "high_recall_threshold": high_recall,
        "balanced_precision_recall_threshold": balanced,
    }


def save_plots(
    output_dir: Path,
    y_test: pd.Series,
    probabilities: np.ndarray,
    predictions: np.ndarray,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    fig, axis = plt.subplots(figsize=(6, 5))
    ConfusionMatrixDisplay.from_predictions(
        y_test,
        predictions,
        display_labels=["Fully Paid", "Default"],
        cmap="Blues",
        colorbar=False,
        ax=axis,
    )
    axis.set_title("Holdout Confusion Matrix at Threshold 0.50")
    fig.tight_layout()
    fig.savefig(output_dir / "confusion_matrix.png", dpi=160)
    plt.close(fig)

    fpr, tpr, _ = roc_curve(y_test, probabilities)
    fig, axis = plt.subplots(figsize=(6, 5))
    axis.plot(fpr, tpr, label=f"ROC-AUC = {roc_auc_score(y_test, probabilities):.3f}")
    axis.plot([0, 1], [0, 1], linestyle="--", color="gray")
    axis.set(
        xlabel="False Positive Rate",
        ylabel="True Positive Rate",
        title="Holdout ROC Curve",
    )
    axis.legend(loc="lower right")
    fig.tight_layout()
    fig.savefig(output_dir / "roc_curve.png", dpi=160)
    plt.close(fig)

    precision, recall, _ = precision_recall_curve(y_test, probabilities)
    fig, axis = plt.subplots(figsize=(6, 5))
    axis.plot(
        recall,
        precision,
        label=f"PR-AUC = {average_precision_score(y_test, probabilities):.3f}",
    )
    axis.axhline(float(y_test.mean()), linestyle="--", color="gray")
    axis.set(
        xlabel="Recall",
        ylabel="Precision",
        title="Holdout Precision-Recall Curve",
    )
    axis.legend(loc="upper right")
    fig.tight_layout()
    fig.savefig(output_dir / "pr_curve.png", dpi=160)
    plt.close(fig)

    observed, predicted = calibration_curve(
        y_test, probabilities, n_bins=10, strategy="quantile"
    )
    fig, axis = plt.subplots(figsize=(6, 5))
    axis.plot(predicted, observed, marker="o", label="Logistic regression")
    axis.plot([0, 1], [0, 1], linestyle="--", color="gray", label="Ideal")
    axis.set(
        xlabel="Mean Predicted Probability",
        ylabel="Observed Positive Rate",
        title="Holdout Calibration Curve",
    )
    axis.legend(loc="upper left")
    fig.tight_layout()
    fig.savefig(output_dir / "calibration_curve.png", dpi=160)
    plt.close(fig)


def run_training(args: argparse.Namespace) -> dict[str, Any]:
    validate_args(args)
    created_at = datetime.now(timezone.utc).isoformat()
    frame, feature_columns = load_training_frame(
        args.data_path, args.target_column, args.sample_rows
    )
    rows_loaded = len(frame)

    mapped_target = map_target(frame[args.target_column])
    eligible = mapped_target.notna()
    filtered = frame.loc[eligible, feature_columns].copy()
    target = mapped_target.loc[eligible].astype(int)
    if target.nunique() != 2:
        raise ValueError(
            "Filtered target must contain both resolved classes: Fully Paid "
            "and Charged Off or Default."
        )

    X_train, X_test, y_train, y_test = train_test_split(
        filtered,
        target,
        test_size=args.test_size,
        random_state=args.random_state,
        stratify=target,
    )
    preprocessor, numeric_columns, categorical_columns = build_preprocessor(
        X_train, feature_columns
    )
    model = LogisticRegression(
        class_weight="balanced",
        solver="liblinear",
        max_iter=1000,
        random_state=args.random_state,
    )

    training_started = perf_counter()
    X_train_processed = preprocessor.fit_transform(X_train)
    model.fit(X_train_processed, y_train)
    training_seconds = perf_counter() - training_started

    inference_started = perf_counter()
    X_test_processed = preprocessor.transform(X_test)
    probabilities = model.predict_proba(X_test_processed)[:, 1]
    predictions = (probabilities >= 0.50).astype(int)
    inference_seconds = perf_counter() - inference_started

    tn, fp, fn, tp = confusion_matrix(
        y_test, predictions, labels=[0, 1]
    ).ravel()
    metrics = {
        "accuracy": float(accuracy_score(y_test, predictions)),
        "precision": float(precision_score(y_test, predictions, zero_division=0)),
        "recall": float(recall_score(y_test, predictions, zero_division=0)),
        "f1": float(f1_score(y_test, predictions, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_test, probabilities)),
        "pr_auc": float(average_precision_score(y_test, probabilities)),
        "brier_score": float(brier_score_loss(y_test, probabilities)),
        "default_threshold": 0.50,
        "positive_rate": float(target.mean()),
        "test_positive_rate": float(y_test.mean()),
        "true_negative_count": int(tn),
        "false_positive_count": int(fp),
        "false_negative_count": int(fn),
        "true_positive_count": int(tp),
    }
    report = classification_report(
        y_test,
        predictions,
        target_names=["fully_paid", "default_risk"],
        output_dict=True,
        zero_division=0,
    )
    thresholds = threshold_metrics(y_test, probabilities)

    args.artifact_dir.mkdir(parents=True, exist_ok=True)
    with (args.artifact_dir / "preprocessor.pkl").open("wb") as file_obj:
        pickle.dump(preprocessor, file_obj)
    with (args.artifact_dir / "model.pkl").open("wb") as file_obj:
        pickle.dump(model, file_obj)

    feature_payload = {
        "feature_columns": feature_columns,
        "numeric_columns": numeric_columns,
        "categorical_columns": categorical_columns,
        "transformed_feature_count": int(X_train_processed.shape[1]),
        "excluded_feature_policy": (
            "Only curated application-time fields are used. Post-outcome "
            "payment, recovery, hardship, settlement, and status fields are "
            "excluded."
        ),
    }
    training_run = {
        "rows_loaded": rows_loaded,
        "rows_after_target_filter": int(len(target)),
        "rows_used": int(len(target)),
        "train_rows": int(len(y_train)),
        "test_rows": int(len(y_test)),
        "positive_rate": float(target.mean()),
        "feature_count": len(feature_columns),
        "transformed_feature_count": int(X_train_processed.shape[1]),
        "model_name": "LogisticRegression(class_weight='balanced')",
        "random_state": args.random_state,
        "test_size": args.test_size,
        "training_seconds": float(training_seconds),
        "inference_seconds_test": float(inference_seconds),
        "created_at_utc": created_at,
        "sample_mode": args.sample_rows is not None,
        "sample_rows_requested": args.sample_rows,
        "sampling_strategy": (
            "Deterministic prefix sample from the local ignored source CSV."
            if args.sample_rows is not None
            else "All rows from the local ignored source CSV."
        ),
        "data_path_mode": "local_ignored_data",
        "target_column": args.target_column,
        "eligible_positive_labels": sorted(POSITIVE_LABELS),
        "eligible_negative_labels": sorted(NEGATIVE_LABELS),
    }

    write_json(args.output_dir / "metrics.json", metrics)
    write_json(args.output_dir / "classification_report.json", report)
    write_json(args.output_dir / "training_run.json", training_run)
    write_json(args.output_dir / "feature_columns.json", feature_payload)
    write_json(args.output_dir / "threshold_analysis.json", thresholds)
    save_plots(args.output_dir, y_test, probabilities, predictions)

    print(
        json.dumps(
            {
                "status": "completed",
                "rows_used": training_run["rows_used"],
                "sample_mode": training_run["sample_mode"],
                "roc_auc": metrics["roc_auc"],
                "pr_auc": metrics["pr_auc"],
                "training_seconds": training_seconds,
                "inference_seconds_test": inference_seconds,
            },
            indent=2,
        )
    )
    return {
        "metrics": metrics,
        "training_run": training_run,
        "threshold_analysis": thresholds,
    }


def main(argv: list[str] | None = None) -> int:
    try:
        run_training(parse_args(argv))
        return 0
    except Exception as exc:
        print(f"Training failed: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
