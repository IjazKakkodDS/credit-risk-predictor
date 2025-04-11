# src/data_ingestion.py

import os
import sys
import pandas as pd
from dataclasses import dataclass
from sklearn.model_selection import train_test_split

from src.exception import CustomException
from src.logger import logging

@dataclass
class DataIngestionConfig:
    raw_data_path: str = os.path.join("artifact", "data_raw.csv")
    train_data_path: str = os.path.join("artifact", "train.csv")
    test_data_path: str = os.path.join("artifact", "test.csv")

class DataIngestion:
    def __init__(self):
        self.config = DataIngestionConfig()

    def initiate_data_ingestion(self):
        logging.info("Starting data ingestion")
        try:
            # Update with your actual CSV path
            csv_path = r"C:\Users\ijazk\OneDrive\Desktop\ML_AI_Portfolio\credit-risk-predictor-lending-club\notebook\data\lending_club_accepted_loans.csv"
            df = pd.read_csv(csv_path, low_memory=False)
            logging.info(f"Loaded data from {csv_path}, shape={df.shape}")

            # Drop columns with >40% missing
            threshold = 40.0
            total_rows = len(df)
            missing_percent = df.isnull().sum() * 100 / total_rows
            cols_to_drop = missing_percent[missing_percent > threshold].index.tolist()
            df.drop(columns=cols_to_drop, inplace=True, errors="ignore")
            logging.info(f"Dropped columns with >{threshold}% missing: {cols_to_drop}")

            # Save cleaned raw
            os.makedirs(os.path.dirname(self.config.raw_data_path), exist_ok=True)
            df.to_csv(self.config.raw_data_path, index=False)
            logging.info(f"Saved cleaned raw data to {self.config.raw_data_path}")

            # 80/20 split
            train_df, test_df = train_test_split(df, test_size=0.2, random_state=42)
            logging.info(f"Train shape={train_df.shape}, Test shape={test_df.shape}")

            train_df.to_csv(self.config.train_data_path, index=False)
            test_df.to_csv(self.config.test_data_path, index=False)

            return (self.config.train_data_path, self.config.test_data_path)
        except Exception as e:
            raise CustomException(e, sys)
