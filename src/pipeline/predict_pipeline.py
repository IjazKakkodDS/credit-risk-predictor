# src/predict_pipeline.py

import os
import sys
import pandas as pd
import numpy as np

from src.exception import CustomException
from src.logger import logging
from src.utils import load_object

class PredictPipeline:
    def __init__(self):
        self.preprocessor_path = os.path.join("artifact", "preprocessor.pkl")
        self.model_path = os.path.join("artifact", "model.pkl")

    def predict(self, input_df: pd.DataFrame):
        try:
            preprocessor = load_object(self.preprocessor_path)
            model = load_object(self.model_path)

            transformed = preprocessor.transform(input_df)
            preds = model.predict(transformed)
            return preds
        except Exception as e:
            raise CustomException(e, sys)

class CustomData:
    """
    Example usage:
      cd = CustomData(loan_amnt=5000, annual_inc=60000, ...)
      df = cd.get_data_as_dataframe()
      pipeline = PredictPipeline()
      preds = pipeline.predict(df)
    """
    def __init__(self, **kwargs):
        for k,v in kwargs.items():
            setattr(self, k, v)
    
    def get_data_as_dataframe(self):
        try:
            data = {k: [v] for k, v in self.__dict__.items()}
            return pd.DataFrame(data)
        except Exception as e:
            raise CustomException(e, sys)
