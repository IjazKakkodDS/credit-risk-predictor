"""Run an artifact-gated synthetic batch inference benchmark."""

from __future__ import annotations

import argparse
import json
import pickle
import sys
from datetime import datetime, timezone
from pathlib import Path
from statistics import median
from time import perf_counter
from typing import Any, Iterable

import numpy as np
import pandas as pd
import psutil


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

DEFAULT_ARTIFACT_DIR = REPO_ROOT / "artifact"
DEFAULT_OUTPUT_PATH = REPO_ROOT / "reports" / "benchmark_summary.json"
DEFAULT_RESULTS_PATH = REPO_ROOT / "reports" / "batch_benchmark_results.json"
DEFAULT_REPORT_PATH = REPO_ROOT / "reports" / "batch_benchmark_report.md"
MISSING_MODEL_MESSAGE = (
    "Model artifact not found. Benchmark scaffold is ready, but full benchmark "
    "requires a trained model artifact."
)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Benchmark synthetic batch inference without raw data, network "
            "calls, or retraining."
        )
    )
    parser.add_argument("--rows", type=int, nargs="+", default=[1000])
    parser.add_argument("--runs", type=int, default=5)
    parser.add_argument("--artifact-dir", type=Path, default=DEFAULT_ARTIFACT_DIR)
    parser.add_argument("--model-path", type=Path)
    parser.add_argument(
        "--preprocessor-path", type=Path
    )
    parser.add_argument(
        "--output",
        "--output-path",
        dest="output_path",
        type=Path,
        default=DEFAULT_OUTPUT_PATH,
    )
    parser.add_argument("--results-output", type=Path, default=DEFAULT_RESULTS_PATH)
    parser.add_argument("--report-output", type=Path, default=DEFAULT_REPORT_PATH)
    return parser.parse_args(argv)


def load_pickle(path: Path) -> Any:
    with path.open("rb") as file_obj:
        return pickle.load(file_obj)


def _clean_choices(values: Iterable[Any]) -> list[Any]:
    choices = []
    for value in values:
        try:
            if pd.isna(value):
                continue
        except (TypeError, ValueError):
            pass
        choices.append(value)
    return choices


def _numeric_values(column: str, rows: int, rng: np.random.Generator) -> np.ndarray:
    name = column.lower()
    if "loan_amnt" in name or "funded_amnt" in name:
        return rng.integers(1000, 40001, size=rows)
    if "annual_inc" in name:
        return rng.lognormal(mean=11.0, sigma=0.5, size=rows)
    if name in {"int_rate", "revol_util", "all_util", "bc_util"}:
        return rng.uniform(0, 100, size=rows)
    if "fico" in name:
        return rng.integers(580, 851, size=rows)
    if name == "dti":
        return rng.uniform(0, 45, size=rows)
    if "installment" in name or "pymnt" in name or "bal" in name:
        return rng.uniform(0, 50000, size=rows)
    if name == "policy_code":
        return np.ones(rows)
    return rng.integers(0, 25, size=rows)


def _repeat_choices(
    values: Iterable[Any], rows: int, fallback: str = "UNKNOWN"
) -> np.ndarray:
    choices = _clean_choices(values)
    if not choices:
        choices = [fallback]
    return np.resize(np.asarray(choices, dtype=object), rows)


def generate_synthetic_rows(preprocessor: Any, rows: int) -> pd.DataFrame:
    """Generate input rows using the fitted transformer's real column contract."""
    if rows <= 0:
        raise ValueError("--rows must be greater than zero")

    feature_names = list(getattr(preprocessor, "feature_names_in_", []))
    if not feature_names:
        raise ValueError("Preprocessor does not expose fitted input feature names.")

    rng = np.random.default_rng(42)
    data: dict[str, Any] = {
        str(column): np.zeros(rows) for column in feature_names
    }

    for name, transformer, columns in getattr(preprocessor, "transformers_", []):
        if name == "remainder" or transformer == "drop":
            continue
        column_names = [str(column) for column in columns]
        if name == "numeric":
            for column in column_names:
                data[column] = _numeric_values(column, rows, rng)
        elif name in {"small_cat", "categorical"}:
            encoder = transformer.named_steps.get("one_hot")
            categories = getattr(encoder, "categories_", [])
            for index, column in enumerate(column_names):
                values = categories[index] if index < len(categories) else []
                data[column] = _repeat_choices(values, rows)
        elif name == "large_cat":
            encoder = transformer.named_steps.get("freq_encoder")
            frequency_maps = getattr(encoder, "freq_maps_", {})
            for column in column_names:
                data[column] = _repeat_choices(frequency_maps.get(column, {}), rows)
        elif name == "date_cols":
            for column in column_names:
                data[column] = np.resize(
                    np.asarray(["2018-01-01", "2019-06-15", "2020-12-01"]),
                    rows,
                )

    return pd.DataFrame(data, columns=[str(column) for column in feature_names])


def percentile(values: list[float], percentile_value: float) -> float:
    return float(np.percentile(np.asarray(values), percentile_value))


def benchmark_size(
    rows: int,
    runs: int,
    preprocessor: Any,
    model: Any,
    artifact_used: str,
) -> dict[str, Any]:
    process = psutil.Process()
    try:
        batch = generate_synthetic_rows(preprocessor, rows)
        categorical_columns = []
        for name, _, columns in getattr(preprocessor, "transformers_", []):
            if name in {"small_cat", "large_cat", "categorical"}:
                categorical_columns.extend(str(column) for column in columns)
        categorical_ok = bool(categorical_columns) and all(
            not (
                pd.api.types.is_numeric_dtype(batch[column])
                and (batch[column] == 0).all()
            )
            for column in categorical_columns
        )
        model.predict(preprocessor.transform(batch))
        timings = []
        peak_rss = process.memory_info().rss
        for _ in range(runs):
            started = perf_counter()
            predictions = model.predict(preprocessor.transform(batch))
            timings.append(perf_counter() - started)
            peak_rss = max(peak_rss, process.memory_info().rss)
        if len(predictions) != rows:
            raise ValueError(
                f"Prediction count mismatch: expected {rows}, got {len(predictions)}."
            )
        p50 = median(timings)
        return {
            "rows": rows,
            "runs": runs,
            "elapsed_seconds": float(sum(timings)),
            "p50_seconds": float(p50),
            "p95_seconds": percentile(timings, 95),
            "rows_per_second": float(rows / p50),
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "artifact_used": artifact_used,
            "categorical_generation_status": (
                "realistic_categories" if categorical_ok else "failed_validation"
            ),
            "peak_process_rss_mb": float(peak_rss / (1024 * 1024)),
            "status": "success",
            "failure_reason": None,
        }
    except Exception as exc:
        return {
            "rows": rows,
            "runs": runs,
            "status": "failed",
            "failure_reason": str(exc),
            "categorical_generation_status": "failed",
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "artifact_used": artifact_used,
        }


def render_report(results: list[dict[str, Any]]) -> str:
    lines = [
        "# Batch Benchmark Report",
        "",
        "| Rows | Status | p50 seconds | p95 seconds | Rows/sec | Peak RSS MB |",
        "| ---: | --- | ---: | ---: | ---: | ---: |",
    ]
    for item in results:
        if item["status"] == "success":
            lines.append(
                f"| {item['rows']:,} | success | {item['p50_seconds']:.6f} | "
                f"{item['p95_seconds']:.6f} | {item['rows_per_second']:.2f} | "
                f"{item['peak_process_rss_mb']:.2f} |"
            )
        else:
            lines.append(
                f"| {item['rows']:,} | failed: {item['failure_reason']} | | | | |"
            )
    lines.extend(
        [
            "",
            "Benchmarks include preprocessing and prediction with generated values "
            "drawn from fitted categorical vocabularies. They are local batch "
            "measurements, not deployed-service claims.",
        ]
    )
    return "\n".join(lines) + "\n"


def run_benchmark(args: argparse.Namespace) -> int:
    if any(rows <= 0 for rows in args.rows):
        print("--rows values must be greater than zero.", file=sys.stderr)
        return 2
    if args.runs <= 0:
        print("--runs must be greater than zero.", file=sys.stderr)
        return 2
    model_path = args.model_path or args.artifact_dir / "model.pkl"
    preprocessor_path = (
        args.preprocessor_path or args.artifact_dir / "preprocessor.pkl"
    )
    if not model_path.is_file():
        print(MISSING_MODEL_MESSAGE)
        return 0
    if not preprocessor_path.is_file():
        print(
            "Preprocessor artifact not found. Benchmark requires both the fitted "
            "preprocessor and trained model artifacts."
        )
        return 0

    try:
        preprocessor = load_pickle(preprocessor_path)
        model = load_pickle(model_path)
        try:
            artifact_used = str(model_path.resolve().relative_to(REPO_ROOT))
        except ValueError:
            artifact_used = model_path.name
        artifact_used = artifact_used.replace("\\", "/")
        results = [
            benchmark_size(rows, args.runs, preprocessor, model, artifact_used)
            for rows in args.rows
        ]
        successful = [item for item in results if item["status"] == "success"]
        if not successful:
            raise RuntimeError("All requested benchmark sizes failed.")
        summary = {
            "benchmark": "synthetic_batch_inference",
            **successful[-1],
            "p50_milliseconds": successful[-1]["p50_seconds"] * 1000,
            "rows_per_second_at_p50": successful[-1]["rows_per_second"],
            "notes": "Latest successful CR-3 batch measurement.",
        }
        args.output_path.parent.mkdir(parents=True, exist_ok=True)
        args.output_path.write_text(
            json.dumps(summary, indent=2) + "\n", encoding="utf-8"
        )
        write_payload = {
            "benchmark": "synthetic_batch_inference_ladder",
            "results": results,
        }
        args.results_output.parent.mkdir(parents=True, exist_ok=True)
        args.results_output.write_text(
            json.dumps(write_payload, indent=2) + "\n", encoding="utf-8"
        )
        args.report_output.write_text(render_report(results), encoding="utf-8")
        print(
            f"Benchmark completed. Results written to {args.results_output}"
        )
        return 0 if all(item["status"] == "success" for item in results) else 1
    except Exception as exc:
        print(f"Benchmark could not execute: {exc}", file=sys.stderr)
        return 1


def main(argv: list[str] | None = None) -> int:
    return run_benchmark(parse_args(argv))


if __name__ == "__main__":
    raise SystemExit(main())
