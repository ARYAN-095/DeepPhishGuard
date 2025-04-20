from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import joblib
import uvicorn
import requests

import numpy as np
from src.feature_extractor.feature_builder import build_feature_vector

app = FastAPI(title="DeepPhishGuard API")

# Load model
MODEL_PATH = "models/xgboost_best.pkl"
model = joblib.load(MODEL_PATH)

# Request schema
class PredictionRequest(BaseModel):
    url: str = None
    html: str = None

# Response schema
class PredictionResponse(BaseModel):
    prediction: str
    confidence: float

@app.post("/predict", response_model=PredictionResponse)
def predict(request: PredictionRequest):
    if not request.url and not request.html:
        raise HTTPException(status_code=400, detail="Either 'url' or 'html' must be provided.")

    try:
        if request.url:
            response = requests.get(request.url, timeout=10)
            html = response.text
            base_url = request.url
        else:
            html = request.html
            base_url = "https://example.com"

        vector = build_feature_vector(base_url, html).reshape(1, -1)
        prob = model.predict_proba(vector)[0][1]
        is_phish = prob > 0.5

        return PredictionResponse(
            prediction="phishing" if is_phish else "benign",
            confidence=round(prob, 4)
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")


@app.get("/")
def root():
    return {"status": "DeepPhishGuard API is running"}
