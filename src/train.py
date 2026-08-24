"""
Train xG classifiers on Euro 2024 shot data, log experiments with MLflow,
and save the best model (lowest test log loss).

Usage:
    python src/train.py --data data/euro2024_shots.csv
"""

import argparse
import os

import joblib
import mlflow
import mlflow.sklearn
from sklearn.metrics import brier_score_loss, log_loss, roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split

from data_preprocessing import clean_data, engineer_features, get_X_y, load_data
from models import build_pipelines


def evaluate(y_true, y_proba) -> dict:
    return {
        "log_loss": log_loss(y_true, y_proba),
        "roc_auc": roc_auc_score(y_true, y_proba),
        "brier": brier_score_loss(y_true, y_proba),
    }


def train_and_log(data_path: str, output_dir: str = "models") -> str:
    os.makedirs(output_dir, exist_ok=True)

    df = load_data(data_path)
    df = engineer_features(df)
    df = clean_data(df)
    X, y = get_X_y(df)

    # Stratified split: goals are a minority class, so preserve their ratio in the test set
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    pipelines = build_pipelines()

    best_name, best_loss, best_pipeline = None, float("inf"), None

    mlflow.set_experiment("euro2024_xg_model")

    for name, pipeline in pipelines.items():
        with mlflow.start_run(run_name=name):
            cv_auc = cross_val_score(
                pipeline, X_train, y_train,
                cv=skf, scoring="roc_auc", n_jobs=-1
            )
            mlflow.log_param("model", name)
            mlflow.log_metric("cv_auc_mean", cv_auc.mean())
            mlflow.log_metric("cv_auc_std", cv_auc.std())

            pipeline.fit(X_train, y_train)
            y_proba = pipeline.predict_proba(X_test)[:, 1]
            metrics = evaluate(y_test, y_proba)

            for k, v in metrics.items():
                mlflow.log_metric(f"test_{k}", v)

            mlflow.sklearn.log_model(pipeline, artifact_path="model")

            print(f"{name:20s}  CV AUC={cv_auc.mean():.3f}  "
                  f"Test LogLoss={metrics['log_loss']:.3f}  AUC={metrics['roc_auc']:.3f}  Brier={metrics['brier']:.3f}")

            if metrics["log_loss"] < best_loss:
                best_loss, best_name, best_pipeline = metrics["log_loss"], name, pipeline

    best_path = os.path.join(output_dir, "best_model.pkl")
    joblib.dump(best_pipeline, best_path)
    print(f"\nBest model: {best_name} (LogLoss={best_loss:.3f}) saved to {best_path}")
    return best_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="data/euro2024_shots.csv")
    parser.add_argument("--output", default="models")
    args = parser.parse_args()
    train_and_log(args.data, args.output)
