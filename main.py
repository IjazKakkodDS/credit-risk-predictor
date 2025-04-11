# main.py

from src.components.data_ingestion import DataIngestion
from src.components.data_transformation import DataTransformation
from src.components.model_trainer import ModelTrainer
from src.logger import logging

def run_pipeline():
    ingestion = DataIngestion()
    train_path, test_path = ingestion.initiate_data_ingestion()

    transformer = DataTransformation()
    train_arr, test_arr, _ = transformer.initiate_data_transformation(train_path, test_path)

    trainer = ModelTrainer()
    best_score = trainer.initiate_model_trainer(train_arr, test_arr)

    logging.info(f"Pipeline complete. Best model test accuracy: {best_score}")

if __name__ == "__main__":
    run_pipeline()
