import os
import json
import joblib
import numpy as np
import pandas as pd


MODEL_XGB_PATH = "models/improved_xgb.pkl"
MODEL_BRF_PATH = "models/brf.pkl"
COLUMNS_PATH = "models/model_columns.pkl"
CONFIG_PATH = "models/ensemble_config.json"

INPUT_PATH = r"C:\Users\User\OneDrive\Desktop\MLProject\src\data\processed\new_provider_data.csv"     # replace with new data later
OUTPUT_PATH = "outputs/fraud_predictions.csv"


def load_artifacts():
    improved_xgb = joblib.load(MODEL_XGB_PATH)
    brf = joblib.load(MODEL_BRF_PATH)
    model_columns = joblib.load(COLUMNS_PATH)

    with open(CONFIG_PATH, "r") as f:
        config = json.load(f)

    return improved_xgb, brf, model_columns, config


def prepare_features(df: pd.DataFrame, model_columns: list[str]) -> tuple[pd.DataFrame, pd.DataFrame]:
    # Keep identifier columns if available
    output_df = pd.DataFrame()
    for candidate in ["NPI", "Rndrng_NPI", "provider_id"]:
        if candidate in df.columns:
            output_df[candidate] = df[candidate]
            break

    # Remove target if present
    if "fraud_label" in df.columns:
        df = df.drop(columns=["fraud_label"])

    # Add missing columns with 0
    missing_cols = [col for col in model_columns if col not in df.columns]
    for col in missing_cols:
        df[col] = 0

    # Drop extra columns not used by model
    df = df[model_columns]

    return df, output_df


def main():
    os.makedirs("outputs", exist_ok=True)

    improved_xgb, brf, model_columns, config = load_artifacts()

    w_xgb = config["w_xgb"]
    w_brf = config["w_brf"]
    threshold = config["threshold"]

    print("Loaded ensemble config:")
    print(config)

    df = pd.read_csv(INPUT_PATH)
    X, output_df = prepare_features(df, model_columns)

    # Predict probabilities from both models
    xgb_probs = improved_xgb.predict_proba(X)[:, 1]
    brf_probs = brf.predict_proba(X)[:, 1]

    # Weighted ensemble
    final_probs = (w_xgb * xgb_probs) + (w_brf * brf_probs)
    final_preds = (final_probs >= threshold).astype(int)

    # Build output
    output_df["xgb_prob"] = np.round(xgb_probs, 6)
    output_df["brf_prob"] = np.round(brf_probs, 6)
    output_df["fraud_risk_score"] = np.round(final_probs * 100, 2)
    output_df["predicted_label"] = final_preds

    output_df = output_df.sort_values("fraud_risk_score", ascending=False)
    output_df.to_csv(OUTPUT_PATH, index=False)

    print(f"Saved predictions to: {OUTPUT_PATH}")
    print(output_df.head(10))


if __name__ == "__main__":
    main()