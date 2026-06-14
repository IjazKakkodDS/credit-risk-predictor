"""Run a measured training scale ladder without claiming failed levels."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from time import perf_counter, sleep
from typing import Any

import psutil


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA_PATH = REPO_ROOT / "data" / "lending_club_accepted_loans.csv"
DEFAULT_RESULTS = REPO_ROOT / "reports" / "scale_benchmark_results.json"
DEFAULT_REPORT = REPO_ROOT / "reports" / "scale_benchmark_report.md"


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run measured credit-risk scale stages.")
    parser.add_argument("--data-path", type=Path, default=DEFAULT_DATA_PATH)
    parser.add_argument("--sizes", type=int, nargs="+", default=[100000, 500000, 1000000])
    parser.add_argument("--include-full", action="store_true")
    parser.add_argument("--max-minutes-per-run", type=float, default=30)
    parser.add_argument("--split-strategy", choices=["random", "temporal"], default="temporal")
    parser.add_argument("--output", type=Path, default=DEFAULT_RESULTS)
    parser.add_argument("--report-output", type=Path, default=DEFAULT_REPORT)
    return parser.parse_args(argv)


def monitor(command: list[str], timeout_seconds: float) -> tuple[int, str, str, float, float]:
    started = perf_counter()
    process = subprocess.Popen(
        command,
        cwd=REPO_ROOT,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    ps_process = psutil.Process(process.pid)
    peak_rss = 0
    timed_out = False
    while process.poll() is None:
        try:
            current_rss = ps_process.memory_info().rss
            for child in ps_process.children(recursive=True):
                current_rss += child.memory_info().rss
            peak_rss = max(peak_rss, current_rss)
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            pass
        if perf_counter() - started > timeout_seconds:
            timed_out = True
            process.kill()
            break
        sleep(0.1)
    stdout, stderr = process.communicate()
    if timed_out:
        stderr = (stderr + "\nTimed out.").strip()
    return process.returncode or (124 if timed_out else 0), stdout, stderr, perf_counter() - started, peak_rss / (1024 * 1024)


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def stage_result(
    size: int | None, args: argparse.Namespace
) -> dict[str, Any]:
    label = "full" if size is None else str(size)
    output_dir = REPO_ROOT / "artifact" / "scale_runs" / label / "reports"
    artifact_dir = REPO_ROOT / "artifact" / "scale_runs" / label / "artifacts"
    command = [
        sys.executable,
        str(REPO_ROOT / "scripts" / "train_credit_risk_model.py"),
        "--data-path",
        str(args.data_path),
        "--split-strategy",
        args.split_strategy,
        "--output-dir",
        str(output_dir),
        "--artifact-dir",
        str(artifact_dir),
        "--skip-dataset-hash",
    ]
    if size is not None:
        command.extend(["--sample-rows", str(size)])
    return_code, stdout, stderr, elapsed, peak_memory = monitor(
        command, args.max_minutes_per_run * 60
    )
    base = {
        "requested_rows": "full" if size is None else size,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "total_wall_seconds": elapsed,
        "peak_process_tree_rss_mb": peak_memory,
    }
    if return_code != 0:
        return {
            **base,
            "status": "failed",
            "failure_reason": stderr.strip() or stdout.strip() or f"exit code {return_code}",
        }
    run = read_json(output_dir / "training_run.json")
    calibrated = read_json(output_dir / "calibrated_metrics.json")
    selected = calibrated["selected_threshold_metrics"]
    model_path = artifact_dir / "model.pkl"
    return {
        **base,
        "status": "success",
        "failure_reason": None,
        "usable_rows": run["rows_used"],
        "train_rows": run["train_rows"],
        "calibration_rows": run["calibration_rows"],
        "test_rows": run["test_rows"],
        "split_strategy": run["split_strategy"],
        "model": run["model_name"],
        "training_seconds": run["training_seconds"],
        "total_pipeline_seconds": run["total_pipeline_seconds"],
        "test_inference_seconds": run["inference_seconds_test"],
        "artifact_size_kb": model_path.stat().st_size / 1024,
        "roc_auc": selected["roc_auc"],
        "pr_auc": selected["pr_auc"],
        "brier_score": read_json(output_dir / "metrics.json")["brier_score"],
        "calibrated_brier": selected["brier_score"],
    }


def render_report(results: list[dict[str, Any]]) -> str:
    lines = [
        "# Scale Benchmark Report",
        "",
        "| Requested rows | Status | Usable rows | Training sec | Total sec | Peak RSS MB | ROC-AUC | PR-AUC | Calibrated Brier |",
        "| ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for item in results:
        if item["status"] == "success":
            lines.append(
                f"| {item['requested_rows']} | success | {item['usable_rows']:,} | "
                f"{item['training_seconds']:.4f} | {item['total_pipeline_seconds']:.4f} | "
                f"{item['peak_process_tree_rss_mb']:.2f} | {item['roc_auc']:.4f} | "
                f"{item['pr_auc']:.4f} | {item['calibrated_brier']:.4f} |"
            )
        else:
            lines.append(
                f"| {item['requested_rows']} | failed | | | | "
                f"{item['peak_process_tree_rss_mb']:.2f} | | | |"
            )
            lines.append(f"\nFailure: {item['failure_reason']}\n")
    lines.extend(
        [
            "",
            "Only successful rows are scale evidence. Full-source, 2M, 5M, and "
            "10M execution remain unclaimed unless explicitly present above.",
        ]
    )
    return "\n".join(lines) + "\n"


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if not args.data_path.is_file():
        print(f"Dataset not found: {args.data_path}", file=sys.stderr)
        return 1
    sizes: list[int | None] = list(args.sizes)
    if args.include_full:
        sizes.append(None)
    results = [stage_result(size, args) for size in sizes]
    payload = {
        "split_strategy": args.split_strategy,
        "full_dataset_requested": args.include_full,
        "results": results,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    args.report_output.write_text(render_report(results), encoding="utf-8")
    print(json.dumps(payload, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
