from pathlib import Path
import joblib
import pandas as pd

# Project root directory
BASE_DIR = Path(__file__).resolve().parent.parent

# Models directory
MODEL_DIR = BASE_DIR / "models"

# Load model files
model = joblib.load(MODEL_DIR / "churn_model.pkl")
scaler = joblib.load(MODEL_DIR / "scaler.pkl")
feature_columns = joblib.load(MODEL_DIR / "feature_columns.pkl")


def predict_customer(customer_data):

    df = pd.DataFrame([customer_data])

    df = df[feature_columns]

    # Scale input
    df_scaled = scaler.transform(df)

    prediction = model.predict(df_scaled)[0]
    probability = model.predict_proba(df_scaled)[0][1]

    return prediction, probability