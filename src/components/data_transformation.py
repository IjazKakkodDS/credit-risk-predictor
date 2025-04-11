# src/data_transformation.py

import sys
import os
from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.exception import CustomException
from src.logger import logging
from src.utils import save_object, detect_outliers_iqr

@dataclass
class DataTransformationConfig:
    preprocessor_obj_file_path: str = os.path.join("artifact", "preprocessor.pkl")

# ----------------- FREQUENCY ENCODER FOR LARGE-CARD COLUMNS --------------------
class FrequencyEncoder(BaseEstimator, TransformerMixin):
    """
    Replaces each category with the frequency (count) of that category in the training set.
    Example:
      if 'RENT' appears 8000 times, 'RENT' -> 8000
      if 'MORTGAGE' appears 10000 times, 'MORTGAGE' -> 10000
    """
    def __init__(self):
        pass

    def fit(self, X, y=None):
        """
        X is a DataFrame with shape (n_samples, n_columns)
        We'll compute a freq map for each column separately.
        """
        self.freq_maps_ = {}
        for col in X.columns:
            counts = X[col].value_counts(dropna=False)
            self.freq_maps_[col] = counts.to_dict()
        return self

    def transform(self, X):
        X_transformed = pd.DataFrame()
        for col in X.columns:
            freq_map = self.freq_maps_[col]
            # map each value to freq, unknown => 0
            X_transformed[col] = X[col].map(lambda v: freq_map.get(v, 0)).fillna(0)
        return X_transformed


# ----------------- DATE FEATURE EXTRACTOR --------------------
class DateFeatureExtractor(BaseEstimator, TransformerMixin):
    """
    Convert date columns to year & month numeric features.
    If no date columns exist, or they're not properly parsed as datetime64,
    this will yield zero-based features.
    """
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        # X is a 2D array: shape (n_samples, n_date_cols).
        # We'll produce shape (n_samples, 2*n_date_cols).
        transformed_arrays = []
        for i in range(X.shape[1]):
            col_data = pd.Series(X[:, i]).apply(lambda v: pd.to_datetime(v, errors='coerce'))
            year = col_data.dt.year.fillna(0).astype(int)
            month = col_data.dt.month.fillna(0).astype(int)
            transformed_arrays.append(year.values.reshape(-1,1))
            transformed_arrays.append(month.values.reshape(-1,1))
        return np.hstack(transformed_arrays)


class DataTransformation:
    def __init__(self):
        self.config = DataTransformationConfig()

    def handle_outliers(self, df, numeric_cols):
        """
        Cap outliers in numeric columns using IQR method.
        """
        for col in numeric_cols:
            if col not in df.columns:
                continue
            outlier_index = detect_outliers_iqr(df, col)
            if len(outlier_index) > 0:
                Q1 = df[col].quantile(0.25)
                Q3 = df[col].quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - 1.5 * IQR
                upper_bound = Q3 + 1.5 * IQR
                df.loc[df[col] < lower_bound, col] = lower_bound
                df.loc[df[col] > upper_bound, col] = upper_bound
        return df

    def build_numeric_pipeline(self):
        """
        Numeric pipeline: median imputation + standard scaler
        """
        return Pipeline(steps=[
            ("num_imputer", SimpleImputer(strategy="median")),
            ("num_scaler", StandardScaler())
        ])

    def build_small_cat_pipeline(self):
        """
        For columns with <= max_cat_threshold unique categories:
        frequent imputer + one-hot + standard scaler
        """
        return Pipeline(steps=[
            ("cat_imputer", SimpleImputer(strategy="most_frequent")),
            ("one_hot", OneHotEncoder(handle_unknown='ignore')),
            ("cat_scaler", StandardScaler(with_mean=False))
        ])

    def build_large_cat_pipeline(self):
        """
        For columns with > max_cat_threshold unique categories:
        frequency encoding + median imputer (in case freq=0 for unknown) + standard scaler
        """
        return Pipeline(steps=[
            ("freq_encoder", FrequencyEncoder()),
            ("freq_imputer", SimpleImputer(strategy="median")),  # in case we end up with NaN
            ("freq_scaler", StandardScaler())
        ])

    def build_date_pipeline(self):
        """
        Convert each date col to (year, month).
        """
        return Pipeline(steps=[
            ("date_imputer", SimpleImputer(strategy="most_frequent")),
            ("date_extract", DateFeatureExtractor())
        ])

    def initiate_data_transformation(self, train_path, test_path):
        try:
            train_df = pd.read_csv(train_path, low_memory=False)
            test_df = pd.read_csv(test_path, low_memory=False)

            logging.info(f"Reading training set: {train_path}, shape={train_df.shape}")
            logging.info(f"Reading testing set: {test_path}, shape={test_df.shape}")

            target_col = "loan_status"

            # Drop rows missing the target
            logging.info(f"train_df before dropna: {train_df.shape}")
            train_df.dropna(subset=[target_col], inplace=True)
            logging.info(f"train_df after dropna: {train_df.shape}")

            logging.info(f"test_df before dropna: {test_df.shape}")
            test_df.dropna(subset=[target_col], inplace=True)
            logging.info(f"test_df after dropna: {test_df.shape}")

            # Split features & target
            X_train = train_df.drop(columns=[target_col], axis=1)
            y_train = train_df[target_col]

            X_test = test_df.drop(columns=[target_col], axis=1)
            y_test = test_df[target_col]

            # Identify numeric, categorical, date columns
            numeric_cols = X_train.select_dtypes(include=["int64", "float64"]).columns.tolist()
            cat_cols = X_train.select_dtypes(include=["object", "category"]).columns.tolist()
            date_cols = [
                col for col in X_train.columns
                if np.issubdtype(X_train[col].dtype, np.datetime64)
            ]

            # Optional: remove known ID-like columns
            for col in ["id", "emp_title", "title", "url"]:
                if col in cat_cols:
                    cat_cols.remove(col)
                if col in numeric_cols:
                    numeric_cols.remove(col)

            # -------------- Outlier Capping --------------
            X_train = self.handle_outliers(X_train, numeric_cols)
            X_test = self.handle_outliers(X_test, numeric_cols)

            # -------------- Split cat_cols by cardinality --------------
            max_cat_threshold = 50  # You can adjust
            small_cat_cols = []
            large_cat_cols = []
            for col in cat_cols:
                n_uniq = X_train[col].nunique()
                if n_uniq <= max_cat_threshold:
                    small_cat_cols.append(col)
                else:
                    large_cat_cols.append(col)

            logging.info(
                f"Small-cat columns (<= {max_cat_threshold} unique): {small_cat_cols}"
            )
            logging.info(
                f"Large-cat columns (> {max_cat_threshold} unique): {large_cat_cols}"
            )

            # -------------- Build ColumnTransformer --------------
            numeric_pipeline = self.build_numeric_pipeline()
            small_cat_pipeline = self.build_small_cat_pipeline()
            large_cat_pipeline = self.build_large_cat_pipeline()
            date_pipeline = self.build_date_pipeline()

            # For multiple categorical groups, we can do a FeatureUnion or separate branches:
            column_transformers = []
            if numeric_cols:
                column_transformers.append(("numeric", numeric_pipeline, numeric_cols))
            if small_cat_cols:
                column_transformers.append(("small_cat", small_cat_pipeline, small_cat_cols))
            if large_cat_cols:
                column_transformers.append(("large_cat", large_cat_pipeline, large_cat_cols))
            if date_cols:
                column_transformers.append(("date_cols", date_pipeline, date_cols))

            preprocessor = ColumnTransformer(column_transformers)

            # Fit & transform
            X_train_processed = preprocessor.fit_transform(X_train)
            X_test_processed = preprocessor.transform(X_test)

            logging.info(f"X_train_processed shape={X_train_processed.shape}, "
                         f"y_train shape={y_train.shape}")
            logging.info(f"X_test_processed shape={X_test_processed.shape}, "
                         f"y_test shape={y_test.shape}")

            # Final shape check
            if X_train_processed.shape[0] != y_train.shape[0]:
                raise CustomException(
                    f"Row mismatch in train: X={X_train_processed.shape[0]}, y={y_train.shape[0]}",
                    sys
                )
            if X_test_processed.shape[0] != y_test.shape[0]:
                raise CustomException(
                    f"Row mismatch in test: X={X_test_processed.shape[0]}, y={y_test.shape[0]}",
                    sys
                )

            train_arr = np.c_[X_train_processed, y_train.values]
            test_arr = np.c_[X_test_processed, y_test.values]

            # Save the preprocessor
            save_object(self.config.preprocessor_obj_file_path, preprocessor)
            logging.info("Saved preprocessor object successfully.")

            return train_arr, test_arr, self.config.preprocessor_obj_file_path
        
        except Exception as e:
            raise CustomException(e, sys)
