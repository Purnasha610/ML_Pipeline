import logging
from src.components.data_ingestion import DataIngestion
from src.components.model_trainer import ModelTrainer

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

    try:
        logging.info("🚀 Starting training pipeline...")

        # Step 1: Data Ingestion
        ingestion = DataIngestion()
        train_path, test_path = ingestion.initiate_data_ingestion()
        logging.info(f"✅ Data ingestion completed.\nTrain path: {train_path}\nTest path: {test_path}")

        # Step 2: Model Training
        trainer = ModelTrainer()
        trainer.train(train_path, test_path)   # ✅ Only pass 2 args now
        logging.info("✅ Model training completed.")

    except Exception as e:
        logging.error(f"❌ Training pipeline failed: {e}")
