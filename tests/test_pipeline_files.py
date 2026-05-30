from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def tracked_file_exists(relative_path):
    return (ROOT / relative_path).is_file()


def test_core_pipeline_files_exist():
    required_files = [
        "src/components/data_ingestion.py",
        "src/components/data_transformation.py",
        "src/components/model_trainer.py",
    ]

    for relative_path in required_files:
        assert tracked_file_exists(relative_path)

    predict_pipeline = ROOT / "src/pipeline/predict_pipeline.py"
    if predict_pipeline.exists():
        assert predict_pipeline.is_file()


def test_preprocessor_artifact_is_present_and_non_empty():
    artifact = ROOT / "artifact/preprocessor.pkl"

    if artifact.exists():
        assert artifact.is_file()
        assert artifact.stat().st_size > 0


def test_readme_does_not_claim_untracked_model_artifact():
    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    model_artifact = ROOT / "artifact/model.pkl"

    if not model_artifact.exists():
        assert "model.pkl" not in readme


def test_wrong_project_notebooks_are_absent():
    wrong_notebooks = [
        "notebook/eda_student_performance.ipynb",
        "notebook/model_training_student_perfomance.ipynb",
    ]

    for relative_path in wrong_notebooks:
        assert not (ROOT / relative_path).exists()
