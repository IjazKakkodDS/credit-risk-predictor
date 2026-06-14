import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_scale_results_have_success_or_documented_failure():
    path = ROOT / "reports" / "scale_benchmark_results.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    results = payload["results"]

    assert any(item["status"] == "success" for item in results)
    for item in results:
        if item["status"] == "success":
            assert item["usable_rows"] > 0
            assert item["training_seconds"] > 0
            assert item["total_pipeline_seconds"] > 0
            assert item["peak_process_tree_rss_mb"] > 0
        else:
            assert item["failure_reason"]


def test_scale_report_does_not_claim_unexecuted_large_targets():
    payload = json.loads(
        (ROOT / "reports" / "scale_benchmark_results.json").read_text(
            encoding="utf-8"
        )
    )
    completed = {
        item["requested_rows"]
        for item in payload["results"]
        if item["status"] == "success"
    }
    report = (ROOT / "reports" / "scale_benchmark_report.md").read_text(
        encoding="utf-8"
    )

    if 2_000_000 not in completed:
        assert "2M execution completed" not in report
    if 5_000_000 not in completed:
        assert "5M execution completed" not in report
    if 10_000_000 not in completed:
        assert "10M execution completed" not in report
