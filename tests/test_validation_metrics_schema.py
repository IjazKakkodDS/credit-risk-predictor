import json
import math
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
VALIDATION_DIR = ROOT / "reports" / "model_validation"


def load_json(filename):
    return json.loads((VALIDATION_DIR / filename).read_text(encoding="utf-8"))


def test_metrics_schema_and_ranges():
    metrics = load_json("metrics.json")
    probability_metrics = [
        "accuracy",
        "precision",
        "recall",
        "f1",
        "roc_auc",
        "pr_auc",
        "brier_score",
        "positive_rate",
        "test_positive_rate",
    ]

    for key in probability_metrics:
        assert key in metrics
        assert isinstance(metrics[key], (int, float))
        assert math.isfinite(metrics[key])
        assert 0 <= metrics[key] <= 1

    assert metrics["default_threshold"] == 0.5


def test_training_run_schema_and_counts():
    run = load_json("training_run.json")

    assert isinstance(run["sample_mode"], bool)
    for key in [
        "rows_loaded",
        "rows_after_target_filter",
        "rows_used",
        "train_rows",
        "test_rows",
        "feature_count",
    ]:
        assert isinstance(run[key], int)
        assert run[key] > 0

    assert (
        run["train_rows"] + run["calibration_rows"] + run["test_rows"]
        == run["rows_used"]
    )
    assert 0 <= run["positive_rate"] <= 1
    assert run["training_seconds"] > 0
    assert run["inference_seconds_test"] > 0
    assert run["data_path_mode"] == "local_ignored_data"
    assert run["run_status"] == "success"


def test_threshold_analysis_has_expected_grid_and_selections():
    analysis = load_json("threshold_analysis.json")

    assert len(analysis["threshold_grid"]) == 19
    assert analysis["selection_split"] == "calibration"
    assert analysis["best_f1_threshold"]["threshold"] in {
        item["threshold"] for item in analysis["threshold_grid"]
    }
    assert analysis["balanced_precision_recall_threshold"]["threshold"] in {
        item["threshold"] for item in analysis["threshold_grid"]
    }
