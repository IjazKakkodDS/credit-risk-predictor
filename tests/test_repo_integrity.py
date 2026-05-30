from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_required_public_files_exist():
    assert (ROOT / "README.md").is_file()
    assert (ROOT / "LICENSE.txt").is_file() or (ROOT / "LICENSE").is_file()


def test_wrong_project_files_are_absent():
    removed_paths = [
        "application.py",
        "templates/home.html",
        "templates/index.html",
        "notebook/eda_student_performance.ipynb",
        "notebook/model_training_student_perfomance.ipynb",
    ]

    for relative_path in removed_paths:
        assert not (ROOT / relative_path).exists()


def test_setup_package_identity_is_credit_risk_only():
    setup_text = (ROOT / "setup.py").read_text(encoding="utf-8")

    assert "credit_risk_predictor" in setup_text
    assert "score_predictor" not in setup_text


def test_data_ingestion_has_no_local_machine_path():
    ingestion_text = (ROOT / "src/components/data_ingestion.py").read_text(encoding="utf-8")
    forbidden_fragments = ["C:\\", "OneDrive", "Desktop", "Users\\", "ijazk"]

    for fragment in forbidden_fragments:
        assert fragment not in ingestion_text


def test_readme_does_not_make_unsupported_claims():
    readme = (ROOT / "README.md").read_text(encoding="utf-8").lower()
    forbidden_phrases = [
        "fastapi",
        "streamlit",
        "docker",
        "production-ready",
        "production-grade",
        "enterprise deployed",
        "cloud deployed",
        "customer usage",
        "guaranteed",
        "fully autonomous",
        "regulatory approval",
    ]

    for phrase in forbidden_phrases:
        assert phrase not in readme
