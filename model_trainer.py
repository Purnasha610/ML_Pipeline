import pandas as pd
import os
import logging
import pickle
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

class ModelTrainer:
    def train(self, train_path, test_path):
        try:
            logging.info("📦 Loading training and testing data from CSV files...")
            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)

            # ✅ Check for target column
            if 'DepDel15' not in train_df.columns or 'DepDel15' not in test_df.columns:
                raise ValueError("❌ Target column 'DepDel15' not found in data.")

            # ✅ Split into X and y
            y_train = train_df['DepDel15']
            X_train = train_df.drop(columns=['DepDel15'])

            y_test = test_df['DepDel15']
            X_test = test_df.drop(columns=['DepDel15'])

            # ✅ Drop rows with NaN values
            logging.info("🧹 Dropping rows with missing values...")
            train_clean = pd.concat([X_train, y_train], axis=1).dropna()
            test_clean = pd.concat([X_test, y_test], axis=1).dropna()

            X_train = train_clean.drop(columns=['DepDel15'])
            y_train = train_clean['DepDel15']
            X_test = test_clean.drop(columns=['DepDel15'])
            y_test = test_clean['DepDel15']

            # ✅ Train model
            logging.info("🧠 Training RandomForestClassifier...")
            model = RandomForestClassifier(n_jobs=-1, random_state=42)
            model.fit(X_train, y_train)

            # ✅ Evaluate
            y_pred = model.predict(X_test)
            acc = accuracy_score(y_test, y_pred)
            logging.info(f"✅ Accuracy: {acc}")
            logging.info("📊 Classification Report:\n" + classification_report(y_test, y_pred))
            logging.info("📉 Confusion Matrix:\n" + str(confusion_matrix(y_test, y_pred)))

            # ✅ Save model
            model_path = os.path.join("artifacts", "model.pkl")
            with open(model_path, "wb") as f:
                pickle.dump(model, f)
            logging.info(f"💾 Model saved to: {model_path}")

        except Exception as e:
            logging.error(f"❌ Error during model training: {e}")
            raise e
