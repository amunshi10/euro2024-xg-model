"""
Load the saved best xG model and predict goal probability for new shots.

Usage:
    python src/predict.py --model models/best_model.pkl --input data/new_shots.csv
"""

import argparse

import joblib
import pandas as pd

from data_preprocessing import clean_data, engineer_features, get_X_y


def predict(model_path: str, input_path: str) -> pd.Series:
    pipeline = joblib.load(model_path)
    df = pd.read_csv(input_path)
    df = engineer_features(df)
    df = clean_data(df)
    X, _ = get_X_y(df)
    preds = pipeline.predict_proba(X)[:, 1]
    return pd.Series(preds, name="xg")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="models/best_model.pkl")
    parser.add_argument("--input", required=True)
    args = parser.parse_args()
    results = predict(args.model, args.input)
    print(results.to_string())
