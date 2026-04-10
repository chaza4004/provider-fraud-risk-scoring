from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import pandas as pd
import numpy as np
import json

app = FastAPI()

# Load artifacts
xgb = joblib.load("models/improved_xgb.pkl")
brf = joblib.load("models/brf.pkl")
columns = joblib.load("models/model_columns.pkl")

with open("models/ensemble_config.json") as f:
    config = json.load(f)

w_xgb = config["w_xgb"]
w_brf = config["w_brf"]
threshold = config["threshold"]


class ProviderInput(BaseModel):
    allowed_to_submitted_ratio: float
    standardized_to_payment_ratio: float
    services_per_beneficiary: float
    payment_per_beneficiary: float
    submitted_per_beneficiary :float
    benes_vs_specialty: float
    services_vs_specialty: float
    Tot_Benes: float
    Tot_Srvcs: float



@app.get("/")
def home():
    return {"message": "Fraud Risk API is running"}


@app.post("/predict")
def predict(data: ProviderInput):
    try:
        df = pd.DataFrame([data.dict()])

        for col in columns:
            if col not in df.columns:
                df[col] = 0

        df = df[columns]

        xgb_prob = xgb.predict_proba(df)[:, 1]
        brf_prob = brf.predict_proba(df)[:, 1]

        final_prob = (w_xgb * xgb_prob) + (w_brf * brf_prob)
        pred = (final_prob >= threshold).astype(int)

        return {
            "xgb_probability": round(float(xgb_prob[0]), 4),
            "brf_probability": round(float(brf_prob[0]), 4),
            "fraud_risk_score": round(float(final_prob[0] * 100), 2),
            "prediction": int(pred[0])
        }

    except Exception as e:
        return {"error": str(e)}