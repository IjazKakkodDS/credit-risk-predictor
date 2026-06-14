import pickle
from pathlib import Path

from scripts.benchmark_inference import generate_synthetic_rows


ROOT = Path(__file__).resolve().parents[1]


def test_generated_benchmark_frame_matches_fitted_features():
    with (ROOT / "artifact" / "preprocessor.pkl").open("rb") as file_obj:
        preprocessor = pickle.load(file_obj)

    frame = generate_synthetic_rows(preprocessor, 25)

    assert frame.columns.tolist() == preprocessor.feature_names_in_.tolist()
    assert len(frame) == 25


def test_generated_categorical_values_are_not_numeric_zero_placeholders():
    with (ROOT / "artifact" / "preprocessor.pkl").open("rb") as file_obj:
        preprocessor = pickle.load(file_obj)

    frame = generate_synthetic_rows(preprocessor, 25)
    categorical_columns = []
    for name, _, columns in preprocessor.transformers_:
        if name in {"categorical", "small_cat", "large_cat"}:
            categorical_columns.extend(columns)

    assert categorical_columns
    for column in categorical_columns:
        assert frame[column].dtype == object
        assert not (frame[column] == 0).all()
