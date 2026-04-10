import os
import json
import joblib
import numpy as np
import pandas as pd
#we prepared this code to see if the new data have a big diffrence with the training data that we tained our model on. This is important to check because if the new data is very different from the training data, our model may not perform well. We use a common metric called Population Stability Index (PSI) to measure the difference between the two datasets. The PSI helps us understand if there has been a significant shift in the distribution of features, which could indicate that our model may need to be retrained or updated to maintain its performance.
#in our case we are comparing the same data so as we expected the drift is low but when we will do the prediction on new data we can use this code to check if there is a drift or not and how much it is.
TRAIN_PATH = r"C:\Users\User\OneDrive\Desktop\MLProject\data\processed\model_dataset.csv"
NEW_PATH = r"C:\Users\User\OneDrive\Desktop\MLProject\src\data\processed\new_provider_data.csv"   #we should replace this by the file that we will do the new prediction on
COLUMNS_PATH = r"C:\Users\User\OneDrive\Desktop\MLProject\models\model_columns.pkl"
OUTPUT_PATH = r"C:\Users\User\OneDrive\Desktop\MLProject\outputs\drift_report.csv"


def population_stability_index(expected, actual, bins=10):
    expected = np.asarray(expected, dtype=float)
    actual = np.asarray(actual, dtype=float)

    expected = expected[~np.isnan(expected)]
    actual = actual[~np.isnan(actual)]

    if len(expected) == 0 or len(actual) == 0:
        return np.nan

    # Use expected distribution to define bins
    breakpoints = np.percentile(expected, np.arange(0, bins + 1) * 100 / bins)
    breakpoints = np.unique(breakpoints)

    if len(breakpoints) < 2:
        return 0.0

    expected_counts, _ = np.histogram(expected, bins=breakpoints)
    actual_counts, _ = np.histogram(actual, bins=breakpoints)

    expected_perc = expected_counts / max(expected_counts.sum(), 1)
    actual_perc = actual_counts / max(actual_counts.sum(), 1)

    expected_perc = np.where(expected_perc == 0, 0.0001, expected_perc)
    actual_perc = np.where(actual_perc == 0, 0.0001, actual_perc)

    psi = np.sum((actual_perc - expected_perc) * np.log(actual_perc / expected_perc))
    return psi


def classify_drift(psi):
    if pd.isna(psi):
        return "unknown"
    if psi < 0.10:
        return "low"
    if psi < 0.25:
        return "moderate"
    return "high"


def main():
    os.makedirs("outputs", exist_ok=True)

    model_columns = joblib.load(COLUMNS_PATH)

    train_df = pd.read_csv(TRAIN_PATH)
    new_df = pd.read_csv(NEW_PATH)

    if "fraud_label" in train_df.columns:
        train_df = train_df.drop(columns=["fraud_label"])
    if "fraud_label" in new_df.columns:
        new_df = new_df.drop(columns=["fraud_label"])

    # Keep only model columns that exist in both
    common_cols = [col for col in model_columns if col in train_df.columns and col in new_df.columns]

    numeric_cols = [
        col for col in common_cols
        if pd.api.types.is_numeric_dtype(train_df[col]) and pd.api.types.is_numeric_dtype(new_df[col])
    ]

    rows = []
    for col in numeric_cols:
        train_series = train_df[col]
        new_series = new_df[col]

        psi = population_stability_index(train_series, new_series)

        rows.append({
            "feature": col,
            "train_mean": train_series.mean(),
            "new_mean": new_series.mean(),
            "train_median": train_series.median(),
            "new_median": new_series.median(),
            "train_std": train_series.std(),
            "new_std": new_series.std(),
            "train_missing_rate": train_series.isna().mean(),
            "new_missing_rate": new_series.isna().mean(),
            "psi": psi,
            "drift_flag": classify_drift(psi)
        })

    drift_report = pd.DataFrame(rows).sort_values("psi", ascending=False)
    drift_report.to_csv(OUTPUT_PATH, index=False)

    print(f"Saved drift report to: {OUTPUT_PATH}")
    print(drift_report.head(15))


if __name__ == "__main__":
    main()