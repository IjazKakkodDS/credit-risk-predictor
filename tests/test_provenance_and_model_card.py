import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
VALIDATION_DIR = ROOT / "reports" / "model_validation"


def test_provenance_and_checksums_exist():
    provenance_path = VALIDATION_DIR / "provenance.json"
    checksums_path = VALIDATION_DIR / "artifact_checksums.json"
    assert provenance_path.is_file()
    assert checksums_path.is_file()

    provenance = json.loads(provenance_path.read_text(encoding="utf-8"))
    checksums = json.loads(checksums_path.read_text(encoding="utf-8"))
    assert len(provenance["source_parent_commit"]) == 40
    assert len(provenance["training_script_sha256"]) == 64
    assert isinstance(provenance["working_tree_dirty_at_generation"], bool)
    assert provenance["dataset_fingerprint"]["sha256"] != "not_measured"
    assert all(len(value) == 64 for value in checksums.values())


def test_model_card_and_feature_policy_are_non_empty_and_claim_safe():
    paths = [
        ROOT / "docs" / "evidence" / "model_card.md",
        ROOT / "docs" / "evidence" / "feature_availability_policy.md",
    ]
    forbidden = [
        "production-ready",
        "is a business-optimal threshold",
        "validated customer usage",
    ]
    for path in paths:
        content = path.read_text(encoding="utf-8").lower()
        assert content.strip()
        for phrase in forbidden:
            assert phrase not in content
