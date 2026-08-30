"""
Export a Power BI-ready CSV from the trained v2 (360 freeze-frame) xG model.

Loads the 360-enriched shot dataset, applies the same feature engineering used
at training time, scores every shot with the saved v2 model, and writes a flat
CSV with just the columns a Power BI dashboard needs: who, when, the shot's
geometry/context, the actual outcome, and the model's predicted xG.

Usage:
    python src/export_powerbi.py --output ../euro2024-xg-powerbi-dashboard/data/euro2024_shots_powerbi.csv
"""

import argparse

import joblib
import pandas as pd

from data_preprocessing import clean_data, engineer_features, engineer_freeze_frame_features, get_X_y_360, load_data

DEFAULT_DATA = "data/euro2024_shots_360.csv"
DEFAULT_MODEL = "models/best_model_360.pkl"
DEFAULT_OUTPUT = "../euro2024-xg-powerbi-dashboard/data/euro2024_shots_powerbi.csv"


def build_export(data_path: str, model_path: str) -> pd.DataFrame:
    df = load_data(data_path)
    df = engineer_features(df)
    df = clean_data(df)
    df = engineer_freeze_frame_features(df)

    X, y = get_X_y_360(df)
    model = joblib.load(model_path)
    df = df.loc[X.index].copy()
    df["predicted_xg"] = model.predict_proba(X)[:, 1]

    out = pd.DataFrame({
        "Player": df["player_name"],
        "Team": df["team_name"],
        "Minute": df["minute"],
        "ShotAngle": df["angle"].round(2),
        "ShotDistance": df["distance"].round(2),
        "DefendersInCone": df["defenders_in_cone"],
        "ActualGoal": df["goal"],
        "Outcome": df["goal"].map({1: "Goal", 0: "No Goal"}),
        "PredictedXG": df["predicted_xg"].round(4),
        "StatsBombXG": df["shot_statsbomb_xg"].round(4),
    })
    return out


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default=DEFAULT_DATA)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--output", default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    export_df = build_export(args.data, args.model)
    export_df.to_csv(args.output, index=False, encoding="utf-8")
    print(f"Exported {len(export_df)} shots to {args.output}")
    print(export_df.head(3).to_string(index=False))
