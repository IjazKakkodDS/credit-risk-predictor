"""Check readiness for future credit risk validation artifact generation."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MODEL_PATH = REPO_ROOT / "artifact" / "model.pkl"
DEFAULT_TEST_DATA_PATH = REPO_ROOT / "artifact" / "test.csv"
EXPECTED_OUTPUTS = (
    "confusion_matrix.png",
    "roc_curve.png",
    "pr_curve.png",
    "calibration_curve.png",
    "classification_report.json",
    "threshold_analysis.json",
)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Artifact-gated scaffold for future model validation reporting. "
            "Expected outputs: " + ", ".join(EXPECTED_OUTPUTS)
        )
    )
    parser.add_argument("--model-path", type=Path, default=DEFAULT_MODEL_PATH)
    parser.add_argument("--test-data-path", type=Path, default=DEFAULT_TEST_DATA_PATH)
    return parser.parse_args(argv)


def check_readiness(args: argparse.Namespace) -> int:
    missing = []
    if not args.model_path.is_file():
        missing.append(f"model artifact: {args.model_path}")
    if not args.test_data_path.is_file():
        missing.append(f"test data: {args.test_data_path}")

    if missing:
        print(
            "Validation artifacts were not generated because required inputs "
            "are missing:"
        )
        for item in missing:
            print(f"- {item}")
        print("Expected future outputs:")
        for output in EXPECTED_OUTPUTS:
            print(f"- {output}")
        return 0

    print("Required model and test data inputs were found.")
    print(
        "Validation generation remains intentionally gated until the target "
        "contract, probability interface, and positive-class definition are "
        "documented. No report files were created."
    )
    print("Expected future outputs:")
    for output in EXPECTED_OUTPUTS:
        print(f"- {output}")
    return 0


def main(argv: list[str] | None = None) -> int:
    return check_readiness(parse_args(argv))


if __name__ == "__main__":
    raise SystemExit(main())
