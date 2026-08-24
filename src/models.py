from sklearn.ensemble import GradientBoostingClassifier, HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


def build_pipelines() -> dict:
    """Return a dict of named sklearn Pipelines ready for fit/predict_proba."""
    return {
        "LogisticRegression": Pipeline([
            ("scaler", StandardScaler()),
            ("model", LogisticRegression(max_iter=2000)),
        ]),
        "RandomForest": Pipeline([
            ("scaler", StandardScaler()),
            ("model", RandomForestClassifier(n_estimators=300, max_depth=6, random_state=42, n_jobs=-1)),
        ]),
        "GradientBoosting": Pipeline([
            ("scaler", StandardScaler()),
            ("model", GradientBoostingClassifier(n_estimators=200, learning_rate=0.05, max_depth=3, random_state=42)),
        ]),
        "HistGradientBoosting": Pipeline([
            ("scaler", StandardScaler()),
            ("model", HistGradientBoostingClassifier(max_iter=200, learning_rate=0.05, random_state=42)),
        ]),
    }
