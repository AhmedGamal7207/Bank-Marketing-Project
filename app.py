from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import pandas as pd
import numpy as np

app = FastAPI()
artifacts = joblib.load("models/bank_model.pkl")

pipeline = artifacts['pipeline']
cat_cols = artifacts['cat_cols']
num_cols = artifacts['num_cols']
threshold = artifacts.get('optimal_threshold', 0.5)


class InputData(BaseModel):
    age: int
    job: str
    marital: str
    education: str
    balance: int
    housing: str
    loan: str
    contact: str
    month: str
    campaign: int
    previous: int
    poutcome: str
    pdays: int


@app.post("/predict")
def predict(data: InputData):
    row = data.model_dump()

    # Feature engineering (same as DC_FE + model_training notebooks)
    high_conv_months = ['mar', 'sep', 'oct', 'dec']
    row['high_month'] = 1 if row['month'] in high_conv_months else 0

    age = row['age']
    if age <= 30:
        row['age_group'] = 'Young (<30)'
    elif age <= 45:
        row['age_group'] = 'Adult (30-45)'
    elif age <= 60:
        row['age_group'] = 'Senior (45-60)'
    else:
        row['age_group'] = 'Elder (>60)'

    balance = row['balance']
    if balance <= 0:
        row['balance_group'] = 'Negative/Zero'
    elif balance <= 1000:
        row['balance_group'] = 'Low (1-1k)'
    elif balance <= 5000:
        row['balance_group'] = 'Medium (1k-5k)'
    else:
        row['balance_group'] = 'High (>5k)'

    row['campaign'] = min(row['campaign'], 10)
    row['was_contacted_before'] = 0 if row.pop('pdays') == -1 else 1

    df = pd.DataFrame([row])[cat_cols + num_cols]

    probability = float(pipeline.predict_proba(df)[0][1])
    prediction = 1 if probability >= threshold else 0

    return {
        "prediction": prediction,
        "probability": round(probability, 4),
        "label": "Yes — likely to subscribe" if prediction == 1 else "No — unlikely to subscribe"
    }


@app.get("/home")
def home():
    return {
        "message": "Welcome to the Bank Marketing Prediction API",
        "model": artifacts['best_model_name'],
        "test_auc": round(artifacts['test_auc'], 4),
        "test_f1": round(artifacts['test_f1'], 4),
        "authors": ["Abdallah Mohamed", "Ahmed Gamal", "Ahmed Saleem"],
        "Track": "AI - Alexandria Branch - Intake 46"
    }
