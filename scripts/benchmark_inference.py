"""Run an artifact-gated synthetic batch inference benchmark."""

from __future__ import annotations

import argparse
import json
import pickle
import sys
from pathlib import Path
from statistics import median
from time import perf_counter
from typing import Any, Iterable

import numpy as np
import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

DEFAULT_MODEL_PATH = REPO_ROOT / "artifact" / "model.pkl"
DEFAULT_PREPROCESSOR_PATH = REPO_ROOT / "artifact" / "preprocessor.pkl"
DEFAULT_OUTPUT_PATH = REPO_ROOT / "reports" / "benchmark_summary.json"
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
    parser.add_argument("--rows", type=int, default=1000)
    parser.add_argument("--runs", type=int, default=5)
    parser.add_argument("--model-path", type=Path, default=DEFAULT_MODEL_PATH)
    parser.add_argument(
        "--preprocessor-path", type=Path, default=DEFAULT_PREPROCESSOR_PATH
    )
    parser.add_argument("--output-path", type=Path, default=DEFAULT_OUTPUT_PATH)
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
        elif name == "small_cat":
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


def run_benchmark(args: argparse.Namespace) -> int:
    if args.rows <= 0:
        print("--rows must be greater than zero.", file=sys.stderr)
        return 2
    if args.runs <= 0:
        print("--runs must be greater than zero.", file=sys.stderr)
        return 2
    if not args.model_path.is_file():
        print(MISSING_MODEL_MESSAGE)
        return 0
    if not args.preprocessor_path.is_file():
        print(
            "Preprocessor artifact not found. Benchmark requires both the fitted "
            "preprocessor and trained model artifacts."
        )
        return 0

    try:
        preprocessor = load_pickle(args.preprocessor_path)
        model = load_pickle(args.model_path)
        batch = generate_synthetic_rows(preprocessor, args.rows)
        transformed = preprocessor.transform(batch)

        model.predict(transformed)
        timings = []
        for _ in range(args.runs):
            started = perf_counter()
            predictions = model.predict(transformed)
            timings.append(perf_counter() - started)

        if len(predictions) != args.rows:
            raise ValueError(
                f"Prediction count mismatch: expected {args.rows}, got "
                f"{len(predictions)}."
            )

        p50_seconds = median(timings)
        summary = {
            "benchmark": "synthetic_batch_inference",
            "rows": args.rows,
            "runs": args.runs,
            "p50_seconds": p50_seconds,
            "p50_milliseconds": p50_seconds * 1000,
            "rows_per_second_at_p50": args.rows / p50_seconds,
            "model_path": str(args.model_path),
            "preprocessor_path": str(args.preprocessor_path),
            "notes": (
                "Synthetic pipeline smoke benchmark; not a raw-data or "
                "full-scale performance claim."
            ),
        }
        args.output_path.parent.mkdir(parents=True, exist_ok=True)
        args.output_path.write_text(
            json.dumps(summary, indent=2) + "\n", encoding="utf-8"
        )
        print(f"Benchmark completed. Summary written to {args.output_path}")
        return 0
    except Exception as exc:
        print(f"Benchmark could not execute: {exc}", file=sys.stderr)
        return 1


def main(argv: list[str] | None = None) -> int:
    return run_benchmark(parse_args(argv))


if __name__ == "__main__":
    raise SystemExit(main())
