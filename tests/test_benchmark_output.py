import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_benchmark_summary_contains_real_positive_measurements():
    path = ROOT / "reports" / "benchmark_summary.json"
    assert path.is_file()

    summary = json.loads(path.read_text(encoding="utf-8"))
    assert summary["rows"] > 0
    assert summary["elapsed_seconds"] > 0
    assert summary["p50_seconds"] > 0
    assert summary["rows_per_second"] > 0

    artifact = ROOT / summary["artifact_used"]
    assert artifact.is_file()
    assert artifact.stat().st_size > 0
