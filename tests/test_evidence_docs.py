from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
EVIDENCE_DOCS = [
    ROOT / "docs/evidence/system_scope.md",
    ROOT / "docs/evidence/data_scale_context.md",
    ROOT / "docs/evidence/model_validation_requirements.md",
    ROOT / "docs/evidence/class_imbalance_policy.md",
    ROOT / "docs/evidence/feature_availability_policy.md",
    ROOT / "docs/evidence/model_card.md",
    ROOT / "docs/evidence/reproducibility_profile.md",
    ROOT / "docs/evidence/threshold_policy.md",
    ROOT / "docs/evidence/resource_profile.md",
]


def test_evidence_documents_exist_and_are_non_empty():
    for document in EVIDENCE_DOCS:
        assert document.is_file()
        assert document.read_text(encoding="utf-8").strip()


def test_evidence_documents_avoid_unsupported_claims():
    forbidden_phrases = [
        "production-ready",
        "enterprise deployed",
        "customer usage",
        "guaranteed model quality",
    ]

    for document in EVIDENCE_DOCS:
        content = document.read_text(encoding="utf-8").lower()
        for phrase in forbidden_phrases:
            assert phrase not in content


def test_evidence_documents_do_not_claim_completed_full_scale_benchmark():
    prohibited_claims = [
        "full-scale benchmark has been completed",
        "full-scale benchmark completed",
        "completed full-scale benchmark",
    ]

    for document in EVIDENCE_DOCS:
        content = document.read_text(encoding="utf-8").lower()
        for claim in prohibited_claims:
            assert claim not in content
