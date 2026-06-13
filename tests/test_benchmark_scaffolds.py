import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
BENCHMARK_SCRIPT = ROOT / "scripts/benchmark_inference.py"
VALIDATION_SCRIPT = ROOT / "scripts/generate_validation_report.py"


def run_script(script: Path, *arguments: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, str(script), *arguments],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )


def test_scaffold_scripts_exist():
    assert BENCHMARK_SCRIPT.is_file()
    assert VALIDATION_SCRIPT.is_file()


def test_scaffold_scripts_expose_help():
    for script in [BENCHMARK_SCRIPT, VALIDATION_SCRIPT]:
        result = run_script(script, "--help")
        assert result.returncode == 0
        assert "usage:" in result.stdout.lower()


def test_benchmark_handles_missing_model_without_writing_report():
    output_path = ROOT / "reports" / "__test_benchmark_summary.json"
    result = run_script(
        BENCHMARK_SCRIPT,
        "--model-path",
        str(ROOT / "artifact" / "__missing_model_for_test__.pkl"),
        "--output-path",
        str(output_path),
    )

    assert result.returncode == 0
    assert "Model artifact not found" in result.stdout
    assert not output_path.exists()


def test_validation_handles_missing_inputs_without_fake_outputs():
    result = run_script(
        VALIDATION_SCRIPT,
        "--model-path",
        str(ROOT / "artifact" / "__missing_model_for_test__.pkl"),
        "--test-data-path",
        str(ROOT / "artifact" / "__missing_test_for_test__.csv"),
    )

    assert result.returncode == 0
    assert "required inputs are missing" in result.stdout
    for expected_output in [
        "confusion_matrix.png",
        "roc_curve.png",
        "pr_curve.png",
        "calibration_curve.png",
        "classification_report.json",
        "threshold_analysis.json",
    ]:
        assert expected_output in result.stdout
        assert not (ROOT / "reports" / expected_output).exists()
