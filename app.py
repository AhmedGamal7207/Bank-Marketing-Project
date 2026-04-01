from fastapi import FastAPI
from pydantic import BaseModel
from sklearn.pipeline import Pipeline

import joblib

app = FastAPI()
model = joblib.load("models/dummy_model.pkl")

class InputData(BaseModel):
    dummy_parm1: float
    dummy_parm2: float
    dummy_parm3: float

@app.post("/predict")
def predict(data: InputData):
    features = [[data.dummy_parm1, data.dummy_parm2, data.dummy_parm3]]
    pipeline = Pipeline([
        # ("preprocessor", None),  # Placeholder for any preprocessing steps
        ("model", model)]
        )

    prediction = pipeline.predict(features)[0]
    return {"prediction": float(prediction)}

@app.get("/home")
def home():
    return {"message": "Welcome to the bank marketing API. Wish us luck!", 
            "authors": ["Abdallah Mohamed", "Ahmed Gamal", "Ahmed Saleem"],
            "Track": "AI - Alexandria Branch - Intake 46"}