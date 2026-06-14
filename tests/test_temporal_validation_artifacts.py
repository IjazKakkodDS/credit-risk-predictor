import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
VALIDATION_DIR = ROOT / "reports" / "model_validation"


def load_json(name):
    return json.loads((VALIDATION_DIR / name).read_text(encoding="utf-8"))


def test_temporal_split_summary_has_independent_partitions():
    split = load_json("split_summary.json")

    assert split["strategy"] == "temporal"
    assert split["train_rows"] > 0
    assert split["calibration_rows"] > 0
    assert split["test_rows"] > 0
    assert split["train_date_min"] <= split["train_date_max"]
    assert split["train_date_max"] <= split["calibration_date_max"]
    assert split["calibration_date_max"] <= split["test_date_max"]


def test_threshold_selection_and_final_evaluation_are_separate():
    analysis = load_json("threshold_analysis.json")
    metrics = load_json("calibrated_metrics.json")

    assert analysis["selection_split"] == "calibration"
    assert metrics["calibration_split"] == "calibration"
    assert metrics["selected_threshold_source"] == "calibration_best_f1"
    assert "untouched_test_evaluation" in analysis
