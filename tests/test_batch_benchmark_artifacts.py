import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_batch_ladder_has_real_successful_measurements():
    payload = json.loads(
        (ROOT / "reports" / "batch_benchmark_results.json").read_text(
            encoding="utf-8"
        )
    )
    results = payload["results"]
    assert any(item["status"] == "success" for item in results)

    for item in results:
        assert "categorical_generation_status" in item
        if item["status"] == "success":
            assert item["rows"] > 0
            assert item["rows_per_second"] > 0
            assert item["p50_seconds"] > 0
            assert item["p95_seconds"] > 0
            assert item["categorical_generation_status"] == "realistic_categories"
        else:
            assert item["failure_reason"]
