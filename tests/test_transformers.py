import numpy as np
import pandas as pd

from src.components.data_transformation import DateFeatureExtractor, FrequencyEncoder


def test_frequency_encoder_fit_transform_counts_categories():
    encoder = FrequencyEncoder()
    train = pd.DataFrame({"grade": ["A", "B", "A", None]})

    transformed = encoder.fit(train).transform(train)

    assert transformed.shape == (4, 1)
    assert transformed["grade"].tolist() == [2, 1, 2, 1]


def test_frequency_encoder_unknown_category_does_not_crash():
    encoder = FrequencyEncoder().fit(pd.DataFrame({"grade": ["A", "B", "A"]}))

    transformed = encoder.transform(pd.DataFrame({"grade": ["C", "A"]}))

    assert transformed["grade"].tolist() == [0, 2]


def test_date_feature_extractor_returns_year_and_month_columns():
    extractor = DateFeatureExtractor()
    values = np.array([["2024-01-15"], ["2025-12-02"], ["not-a-date"]], dtype=object)

    transformed = extractor.fit(values).transform(values)

    assert transformed.shape == (3, 2)
    assert transformed.tolist() == [[2024, 1], [2025, 12], [0, 0]]
