"""Train, calibrate, and validate a leakage-aware credit risk baseline."""

from __future__ import annotations

import argparse
import hashlib
import json
import pickle
import platform
import subprocess
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
import sklearn
from sklearn.calibration import CalibratedClassifierCV, calibration_curve
from sklearn.compose import ColumnTransformer
from sklearn.frozen import FrozenEstimator
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

# Pre-decision application and credit-file fields. LendingClub grade, sub-grade,
# interest rate, payment, recovery, hardship, and settlement fields are not
# model inputs.
MODEL_FEATURES = [
    "loan_amnt",
    "term",
    "installment",
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
SEGMENT_COLUMNS = [
    "grade",
    "purpose",
    "home_ownership",
    "addr_state",
    "application_type",
    "term",
]
KNOWN_LEAKAGE_FIELDS = [
    "recoveries",
    "collection_recovery_fee",
    "total_pymnt",
    "total_rec_prncp",
    "total_rec_int",
    "last_pymnt_amnt",
    "out_prncp",
    "settlement_status",
    "hardship_status",
    "debt_settlement_flag",
    "grade",
    "sub_grade",
    "int_rate",
]


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train a calibrated credit-risk baseline with independent splits."
    )
    parser.add_argument("--data-path", type=Path, default=DEFAULT_DATA_PATH)
    parser.add_argument("--target-column", default="loan_status")
    parser.add_argument("--sample-rows", type=int)
    parser.add_argument(
        "--split-strategy", choices=["random", "temporal"], default="random"
    )
    parser.add_argument("--date-column", default="issue_d")
    parser.add_argument("--calibration-size", type=float, default=0.2)
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--artifact-dir", type=Path, default=DEFAULT_ARTIFACT_DIR)
    parser.add_argument("--skip-dataset-hash", action="store_true")
    return parser.parse_args(argv)


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def hash_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file_obj:
        for chunk in iter(lambda: file_obj.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def current_commit() -> str:
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    return result.stdout.strip() if result.returncode == 0 else "unknown"


def working_tree_dirty() -> bool:
    result = subprocess.run(
        ["git", "status", "--porcelain"],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    return bool(result.stdout.strip()) if result.returncode == 0 else True


def validate_args(args: argparse.Namespace) -> None:
    if not args.data_path.is_file():
        raise FileNotFoundError(f"Dataset not found: {args.data_path}")
    if args.sample_rows is not None and args.sample_rows <= 0:
        raise ValueError("--sample-rows must be greater than zero.")
    if args.calibration_size <= 0 or args.test_size <= 0:
        raise ValueError("Calibration and test sizes must be positive.")
    if args.calibration_size + args.test_size >= 1:
        raise ValueError("Calibration plus test size must be less than one.")


def load_frame(args: argparse.Namespace) -> tuple[pd.DataFrame, list[str], list[str]]:
    header = pd.read_csv(args.data_path, nrows=0)
    if args.target_column not in header.columns:
        raise ValueError(f"Target column '{args.target_column}' was not found.")
    if args.split_strategy == "temporal" and args.date_column not in header.columns:
        raise ValueError(
            f"Temporal split requires date column '{args.date_column}'."
        )

    features = [column for column in MODEL_FEATURES if column in header.columns]
    segments = [column for column in SEGMENT_COLUMNS if column in header.columns]
    extra = [args.target_column]
    if args.date_column in header.columns:
        extra.append(args.date_column)
    usecols = list(dict.fromkeys(features + segments + extra))
    frame = pd.read_csv(
        args.data_path,
        usecols=usecols,
        nrows=args.sample_rows,
        low_memory=False,
    )
    return frame, features, segments


def map_target(target: pd.Series) -> pd.Series:
    normalized = target.astype("string").str.strip()
    mapped = pd.Series(np.nan, index=target.index, dtype="float64")
    mapped.loc[normalized.isin(POSITIVE_LABELS)] = 1
    mapped.loc[normalized.isin(NEGATIVE_LABELS)] = 0
    return mapped


def parse_dates(values: pd.Series) -> pd.Series:
    parsed = pd.to_datetime(values, format="%b-%Y", errors="coerce")
    if parsed.notna().sum() == 0:
        parsed = pd.to_datetime(values, errors="coerce")
    return parsed


def split_indices(
    frame: pd.DataFrame,
    target: pd.Series,
    args: argparse.Namespace,
) -> tuple[pd.Index, pd.Index, pd.Index, dict[str, Any]]:
    if args.split_strategy == "temporal":
        dates = parse_dates(frame[args.date_column])
        if dates.isna().any():
            raise ValueError(
                f"Temporal split could not parse {int(dates.isna().sum())} date values."
            )
        ordered = dates.sort_values(kind="stable").index
        train_end = int(len(ordered) * (1 - args.calibration_size - args.test_size))
        calibration_end = int(len(ordered) * (1 - args.test_size))
        train_idx = ordered[:train_end]
        calibration_idx = ordered[train_end:calibration_end]
        test_idx = ordered[calibration_end:]
        summary = {
            "strategy": "temporal",
            "date_column": args.date_column,
            "train_date_min": dates.loc[train_idx].min().date().isoformat(),
            "train_date_max": dates.loc[train_idx].max().date().isoformat(),
            "calibration_date_min": dates.loc[calibration_idx].min().date().isoformat(),
            "calibration_date_max": dates.loc[calibration_idx].max().date().isoformat(),
            "test_date_min": dates.loc[test_idx].min().date().isoformat(),
            "test_date_max": dates.loc[test_idx].max().date().isoformat(),
        }
    else:
        train_cal_idx, test_idx = train_test_split(
            frame.index,
            test_size=args.test_size,
            random_state=args.random_state,
            stratify=target,
        )
        relative_calibration_size = args.calibration_size / (1 - args.test_size)
        train_idx, calibration_idx = train_test_split(
            train_cal_idx,
            test_size=relative_calibration_size,
            random_state=args.random_state,
            stratify=target.loc[train_cal_idx],
        )
        summary = {"strategy": "random", "date_column": None}

    for name, indices in [
        ("train", train_idx),
        ("calibration", calibration_idx),
        ("test", test_idx),
    ]:
        if target.loc[indices].nunique() != 2:
            raise ValueError(f"{name} split does not contain both target classes.")
    summary.update(
        {
            "train_rows": len(train_idx),
            "calibration_rows": len(calibration_idx),
            "test_rows": len(test_idx),
            "train_positive_rate": float(target.loc[train_idx].mean()),
            "calibration_positive_rate": float(target.loc[calibration_idx].mean()),
            "test_positive_rate": float(target.loc[test_idx].mean()),
        }
    )
    return train_idx, calibration_idx, test_idx, summary


def build_preprocessor(
    frame: pd.DataFrame, feature_columns: list[str]
) -> tuple[ColumnTransformer, list[str], list[str]]:
    numeric = [
        column
        for column in feature_columns
        if pd.api.types.is_numeric_dtype(frame[column])
    ]
    categorical = [column for column in feature_columns if column not in numeric]
    preprocessor = ColumnTransformer(
        transformers=[
            (
                "numeric",
                Pipeline(
                    [
                        ("imputer", SimpleImputer(strategy="median")),
                        ("scaler", StandardScaler()),
                    ]
                ),
                numeric,
            ),
            (
                "categorical",
                Pipeline(
                    [
                        ("imputer", SimpleImputer(strategy="most_frequent")),
                        (
                            "one_hot",
                            OneHotEncoder(handle_unknown="ignore", min_frequency=5),
                        ),
                    ]
                ),
                categorical,
            ),
        ]
    )
    return preprocessor, numeric, categorical


def metric_set(
    y_true: pd.Series, probabilities: np.ndarray, threshold: float
) -> dict[str, Any]:
    predictions = (probabilities >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, predictions, labels=[0, 1]).ravel()
    return {
        "threshold": float(threshold),
        "accuracy": float(accuracy_score(y_true, predictions)),
        "precision": float(precision_score(y_true, predictions, zero_division=0)),
        "recall": float(recall_score(y_true, predictions, zero_division=0)),
        "f1": float(f1_score(y_true, predictions, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_true, probabilities)),
        "pr_auc": float(average_precision_score(y_true, probabilities)),
        "brier_score": float(brier_score_loss(y_true, probabilities)),
        "true_negative_count": int(tn),
        "false_positive_count": int(fp),
        "false_negative_count": int(fn),
        "true_positive_count": int(tp),
    }


def threshold_analysis(
    y_true: pd.Series, probabilities: np.ndarray
) -> dict[str, Any]:
    grid = []
    for threshold in np.arange(0.05, 1.0, 0.05):
        metrics = metric_set(y_true, probabilities, float(threshold))
        grid.append(
            {
                "threshold": round(float(threshold), 2),
                "precision": metrics["precision"],
                "recall": metrics["recall"],
                "f1": metrics["f1"],
                "false_positive_count": metrics["false_positive_count"],
                "false_negative_count": metrics["false_negative_count"],
                "predicted_positive_rate": float(
                    (probabilities >= threshold).mean()
                ),
            }
        )
    high_recall = [item for item in grid if item["recall"] >= 0.80]
    return {
        "selection_split": "calibration",
        "policy_status": (
            "Analytical threshold evidence only. Business deployment requires "
            "an approved loss and review-cost matrix."
        ),
        "threshold_grid": grid,
        "best_f1_threshold": max(grid, key=lambda item: item["f1"]),
        "high_recall_threshold": (
            max(high_recall, key=lambda item: item["threshold"])
            if high_recall
            else None
        ),
        "balanced_precision_recall_threshold": min(
            grid, key=lambda item: abs(item["precision"] - item["recall"])
        ),
    }


def segment_analysis(
    test_frame: pd.DataFrame,
    y_test: pd.Series,
    probabilities: np.ndarray,
    threshold: float,
    segment_columns: list[str],
) -> tuple[dict[str, Any], str]:
    results: dict[str, Any] = {}
    predictions = (probabilities >= threshold).astype(int)
    probability_series = pd.Series(probabilities, index=test_frame.index)
    prediction_series = pd.Series(predictions, index=test_frame.index)
    lines = ["# Segment Stability", "", f"Analytical threshold: {threshold:.2f}", ""]

    for column in SEGMENT_COLUMNS:
        if column not in segment_columns:
            results[column] = {"status": "skipped", "reason": "column unavailable"}
            continue
        groups = []
        values = test_frame[column].fillna("MISSING").astype(str)
        for value, indices in values.groupby(values).groups.items():
            segment_y = y_test.loc[indices]
            item: dict[str, Any] = {
                "segment": value,
                "count": len(indices),
                "positive_rate": float(segment_y.mean()),
            }
            if len(indices) < 100:
                item.update(status="skipped", reason="fewer than 100 rows")
            elif segment_y.nunique() < 2:
                item.update(status="skipped", reason="one target class")
            else:
                segment_probabilities = probability_series.loc[indices].to_numpy()
                segment_predictions = prediction_series.loc[indices].to_numpy()
                item.update(
                    status="measured",
                    roc_auc=float(
                        roc_auc_score(segment_y, segment_probabilities)
                    ),
                    pr_auc=float(
                        average_precision_score(segment_y, segment_probabilities)
                    ),
                    precision=float(
                        precision_score(
                            segment_y, segment_predictions, zero_division=0
                        )
                    ),
                    recall=float(
                        recall_score(segment_y, segment_predictions, zero_division=0)
                    ),
                    f1=float(
                        f1_score(segment_y, segment_predictions, zero_division=0)
                    ),
                )
            groups.append(item)
        results[column] = {"status": "completed", "groups": groups}
        measured = sum(item.get("status") == "measured" for item in groups)
        lines.append(f"- `{column}`: {measured} measured groups of {len(groups)}")
    lines.extend(
        [
            "",
            "Small or one-class segments are explicitly skipped. Segment metrics "
            "are diagnostic and are not fairness certification.",
        ]
    )
    return results, "\n".join(lines) + "\n"


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
    axis.set_title("Untouched Test Confusion Matrix")
    fig.tight_layout()
    fig.savefig(output_dir / "confusion_matrix.png", dpi=160)
    plt.close(fig)

    fpr, tpr, _ = roc_curve(y_test, probabilities)
    fig, axis = plt.subplots(figsize=(6, 5))
    axis.plot(fpr, tpr, label=f"ROC-AUC = {roc_auc_score(y_test, probabilities):.3f}")
    axis.plot([0, 1], [0, 1], linestyle="--", color="gray")
    axis.set(xlabel="False Positive Rate", ylabel="True Positive Rate", title="Test ROC Curve")
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
    axis.set(xlabel="Recall", ylabel="Precision", title="Test Precision-Recall Curve")
    axis.legend(loc="upper right")
    fig.tight_layout()
    fig.savefig(output_dir / "pr_curve.png", dpi=160)
    plt.close(fig)

    observed, predicted = calibration_curve(
        y_test, probabilities, n_bins=10, strategy="quantile"
    )
    fig, axis = plt.subplots(figsize=(6, 5))
    axis.plot(predicted, observed, marker="o", label="Calibrated model")
    axis.plot([0, 1], [0, 1], linestyle="--", color="gray", label="Ideal")
    axis.set(
        xlabel="Mean Predicted Probability",
        ylabel="Observed Positive Rate",
        title="Test Calibration Curve",
    )
    axis.legend(loc="upper left")
    fig.tight_layout()
    fig.savefig(output_dir / "calibration_curve.png", dpi=160)
    plt.close(fig)


def run_training(args: argparse.Namespace) -> dict[str, Any]:
    validate_args(args)
    total_started = perf_counter()
    created_at = datetime.now(timezone.utc).isoformat()
    frame, features, segments = load_frame(args)
    rows_loaded = len(frame)
    mapped = map_target(frame[args.target_column])
    eligible = mapped.notna()
    frame = frame.loc[eligible].copy()
    target = mapped.loc[eligible].astype(int)
    if target.nunique() != 2:
        raise ValueError("Resolved target does not contain both classes.")

    train_idx, calibration_idx, test_idx, split_summary = split_indices(
        frame, target, args
    )
    preprocessor, numeric, categorical = build_preprocessor(
        frame.loc[train_idx], features
    )

    training_started = perf_counter()
    X_train = preprocessor.fit_transform(frame.loc[train_idx, features])
    base_model = LogisticRegression(
        class_weight="balanced",
        solver="liblinear",
        max_iter=1000,
        random_state=args.random_state,
    )
    base_model.fit(X_train, target.loc[train_idx])
    training_seconds = perf_counter() - training_started

    X_calibration = preprocessor.transform(frame.loc[calibration_idx, features])
    X_test = preprocessor.transform(frame.loc[test_idx, features])
    calibration_started = perf_counter()
    calibrated_model = CalibratedClassifierCV(
        FrozenEstimator(base_model), method="sigmoid"
    )
    calibrated_model.fit(X_calibration, target.loc[calibration_idx])
    calibration_seconds = perf_counter() - calibration_started

    base_calibration_probabilities = base_model.predict_proba(X_calibration)[:, 1]
    calibrated_calibration_probabilities = calibrated_model.predict_proba(
        X_calibration
    )[:, 1]
    thresholds = threshold_analysis(
        target.loc[calibration_idx], calibrated_calibration_probabilities
    )
    selected_threshold = thresholds["best_f1_threshold"]["threshold"]

    inference_started = perf_counter()
    base_test_probabilities = base_model.predict_proba(X_test)[:, 1]
    calibrated_test_probabilities = calibrated_model.predict_proba(X_test)[:, 1]
    inference_seconds = perf_counter() - inference_started

    uncalibrated_metrics = metric_set(
        target.loc[test_idx], base_test_probabilities, 0.5
    )
    calibrated_default = metric_set(
        target.loc[test_idx], calibrated_test_probabilities, 0.5
    )
    calibrated_selected = metric_set(
        target.loc[test_idx], calibrated_test_probabilities, selected_threshold
    )
    metrics = {
        **uncalibrated_metrics,
        "default_threshold": 0.5,
        "positive_rate": float(target.mean()),
        "test_positive_rate": float(target.loc[test_idx].mean()),
        "evaluation_split": "untouched_test",
        "probability_status": "uncalibrated",
    }
    calibrated_metrics = {
        "calibration_method": "platt_sigmoid",
        "calibration_split": "calibration",
        "default_threshold_metrics": calibrated_default,
        "selected_threshold_metrics": calibrated_selected,
        "selected_threshold_source": "calibration_best_f1",
        "calibration_brier_before": float(
            brier_score_loss(
                target.loc[calibration_idx], base_calibration_probabilities
            )
        ),
        "calibration_brier_after": float(
            brier_score_loss(
                target.loc[calibration_idx], calibrated_calibration_probabilities
            )
        ),
    }
    thresholds["untouched_test_evaluation"] = calibrated_selected

    segment_results, segment_markdown = segment_analysis(
        frame.loc[test_idx],
        target.loc[test_idx],
        calibrated_test_probabilities,
        selected_threshold,
        segments,
    )

    args.artifact_dir.mkdir(parents=True, exist_ok=True)
    preprocessor_path = args.artifact_dir / "preprocessor.pkl"
    model_path = args.artifact_dir / "model.pkl"
    with preprocessor_path.open("wb") as file_obj:
        pickle.dump(preprocessor, file_obj)
    with model_path.open("wb") as file_obj:
        pickle.dump(calibrated_model, file_obj)

    feature_payload = {
        "feature_columns": features,
        "numeric_columns": numeric,
        "categorical_columns": categorical,
        "segment_columns": segments,
        "transformed_feature_count": int(X_train.shape[1]),
        "excluded_leakage_fields": KNOWN_LEAKAGE_FIELDS,
        "prediction_moment": "pre-decision application-time",
    }
    training_run = {
        "run_status": "success",
        "failure_reason": None,
        "rows_loaded": rows_loaded,
        "rows_after_target_filter": len(frame),
        "rows_used": len(frame),
        "train_rows": len(train_idx),
        "calibration_rows": len(calibration_idx),
        "test_rows": len(test_idx),
        "positive_rate": float(target.mean()),
        "feature_count": len(features),
        "transformed_feature_count": int(X_train.shape[1]),
        "model_name": "LogisticRegression(class_weight='balanced') + Platt calibration",
        "random_state": args.random_state,
        "split_strategy": args.split_strategy,
        "training_seconds": float(training_seconds),
        "calibration_seconds": float(calibration_seconds),
        "inference_seconds_test": float(inference_seconds),
        "total_pipeline_seconds": float(perf_counter() - total_started),
        "created_at_utc": created_at,
        "sample_mode": args.sample_rows is not None,
        "sample_rows_requested": args.sample_rows,
        "data_path_mode": "local_ignored_data",
        "target_column": args.target_column,
    }
    provenance = {
        "created_at_utc": created_at,
        "source_parent_commit": current_commit(),
        "working_tree_dirty_at_generation": working_tree_dirty(),
        "training_script_sha256": hash_file(Path(__file__)),
        "python_version": platform.python_version(),
        "platform": platform.platform(),
        "scikit_learn_version": sklearn.__version__,
        "numpy_version": np.__version__,
        "pandas_version": pd.__version__,
        "command": (
            "python scripts/train_credit_risk_model.py "
            f"--sample-rows {args.sample_rows or 'all'} "
            f"--split-strategy {args.split_strategy}"
        ),
        "dataset_fingerprint": {
            "filename": args.data_path.name,
            "size_bytes": args.data_path.stat().st_size,
            "sha256": (
                "not_measured" if args.skip_dataset_hash else hash_file(args.data_path)
            ),
        },
    }
    feature_policy_summary = {
        "prediction_moment": "pre-decision application-time",
        "grade_sub_grade_int_rate": "excluded from model; diagnostic use only",
        "excluded_leakage_fields": KNOWN_LEAKAGE_FIELDS,
        "policy_document": "docs/evidence/feature_availability_policy.md",
    }

    write_json(args.output_dir / "metrics.json", metrics)
    write_json(args.output_dir / "calibrated_metrics.json", calibrated_metrics)
    write_json(
        args.output_dir / "classification_report.json",
        classification_report(
            target.loc[test_idx],
            calibrated_test_probabilities >= selected_threshold,
            target_names=["fully_paid", "default_risk"],
            output_dict=True,
            zero_division=0,
        ),
    )
    write_json(args.output_dir / "training_run.json", training_run)
    write_json(args.output_dir / "feature_columns.json", feature_payload)
    write_json(args.output_dir / "threshold_analysis.json", thresholds)
    write_json(args.output_dir / "split_summary.json", split_summary)
    write_json(args.output_dir / "provenance.json", provenance)
    write_json(args.output_dir / "feature_policy_summary.json", feature_policy_summary)
    write_json(args.output_dir / "segment_stability.json", segment_results)
    (args.output_dir / "segment_stability.md").write_text(
        segment_markdown, encoding="utf-8"
    )
    save_plots(
        args.output_dir,
        target.loc[test_idx],
        calibrated_test_probabilities,
        calibrated_test_probabilities >= selected_threshold,
    )

    checksums = {
        "artifact/model.pkl": hash_file(model_path),
        "artifact/preprocessor.pkl": hash_file(preprocessor_path),
    }
    for name in [
        "metrics.json",
        "calibrated_metrics.json",
        "training_run.json",
        "split_summary.json",
        "threshold_analysis.json",
    ]:
        checksums[f"reports/model_validation/{name}"] = hash_file(
            args.output_dir / name
        )
    write_json(args.output_dir / "artifact_checksums.json", checksums)

    result = {
        "status": "completed",
        "rows_used": len(frame),
        "split_strategy": args.split_strategy,
        "selected_threshold": selected_threshold,
        "test_roc_auc": calibrated_selected["roc_auc"],
        "test_pr_auc": calibrated_selected["pr_auc"],
        "test_brier": calibrated_selected["brier_score"],
        "training_seconds": training_seconds,
        "total_pipeline_seconds": training_run["total_pipeline_seconds"],
    }
    print(json.dumps(result, indent=2))
    return {
        "metrics": metrics,
        "calibrated_metrics": calibrated_metrics,
        "training_run": training_run,
        "split_summary": split_summary,
        "threshold_analysis": thresholds,
    }


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        run_training(args)
        return 0
    except Exception as exc:
        args.output_dir.mkdir(parents=True, exist_ok=True)
        write_json(
            args.output_dir / "training_run.json",
            {
                "run_status": "failed",
                "failure_reason": str(exc),
                "created_at_utc": datetime.now(timezone.utc).isoformat(),
                "split_strategy": args.split_strategy,
                "sample_rows_requested": args.sample_rows,
            },
        )
        print(f"Training failed: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
