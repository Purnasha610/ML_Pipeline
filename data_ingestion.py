import pandas as pd
import os
from sklearn.model_selection import train_test_split
import logging

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

class DataIngestion:
    def __init__(self):
        self.raw_data_path = os.path.join("dataset", "Combined_Flights_2022.csv")
        self.artifacts_dir = os.path.join("artifacts")
        self.train_path = os.path.join(self.artifacts_dir, "train.csv")
        self.test_path = os.path.join(self.artifacts_dir, "test.csv")

    def initiate_data_ingestion(self):
        try:
            logging.info("📥 Reading only first 5000 rows from selected columns")

            # Read only specific columns and only first 5000 rows
            df = pd.read_csv(self.raw_data_path, usecols=[
                'Month', 'DayOfWeek', 'DepTime', 'DepDelayMinutes',
                'ArrDelayMinutes', 'Cancelled', 'AirTime', 'Distance',
                'DepDel15', 'ArrDel15'
            ], nrows=5000)

            logging.info(f"✅ Data loaded. Shape: {df.shape}")

            os.makedirs(self.artifacts_dir, exist_ok=True)

            train_df, test_df = train_test_split(df, test_size=0.2, random_state=42)
            train_df.to_csv(self.train_path, index=False)
            test_df.to_csv(self.test_path, index=False)

            logging.info("✅ Train and test CSVs saved.")
            return self.train_path, self.test_path

        except Exception as e:
            logging.error(f"❌ Data ingestion failed: {e}")
            raise
