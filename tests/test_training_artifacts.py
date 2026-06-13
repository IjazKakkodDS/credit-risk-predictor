import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
VALIDATION_DIR = ROOT / "reports" / "model_validation"


def test_training_outputs_exist_and_are_non_empty():
    required_outputs = [
        VALIDATION_DIR / "metrics.json",
        VALIDATION_DIR / "classification_report.json",
        VALIDATION_DIR / "training_run.json",
        VALIDATION_DIR / "feature_columns.json",
        VALIDATION_DIR / "threshold_analysis.json",
        VALIDATION_DIR / "confusion_matrix.png",
        VALIDATION_DIR / "roc_curve.png",
        VALIDATION_DIR / "pr_curve.png",
        VALIDATION_DIR / "calibration_curve.png",
        ROOT / "artifact" / "model.pkl",
        ROOT / "artifact" / "preprocessor.pkl",
    ]

    for output in required_outputs:
        assert output.is_file()
        assert output.stat().st_size > 0


def test_feature_contract_excludes_post_outcome_leakage_fields():
    payload = json.loads(
        (VALIDATION_DIR / "feature_columns.json").read_text(encoding="utf-8")
    )
    features = set(payload["feature_columns"])
    leakage_fields = {
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
        "issue_d",
    }

    assert features.isdisjoint(leakage_fields)


def test_report_json_does_not_disclose_local_absolute_paths():
    for report in VALIDATION_DIR.glob("*.json"):
        content = report.read_text(encoding="utf-8")
        assert "C:\\" not in content
        assert "OneDrive" not in content
        assert "Users\\" not in content
