import math

import numpy as np
import pandas as pd

# StatsBomb pitch is 120 x 80; the goal spans y in [36, 44] at x = 120
GOAL_X = 120
GOAL_Y_LEFT = 36
GOAL_Y_RIGHT = 44


def load_data(filepath: str) -> pd.DataFrame:
    return pd.read_csv(filepath)


def calculate_angle(x: float, y: float) -> float:
    """Angle (degrees) subtended by the goal mouth from the shot location."""
    g0 = np.array([GOAL_X, GOAL_Y_RIGHT]) - np.array([x, y])
    g1 = np.array([GOAL_X, GOAL_Y_LEFT]) - np.array([x, y])
    angle = math.atan2(np.linalg.det([g0, g1]), np.dot(g0, g1))
    return abs(math.degrees(angle))


def calculate_distance(x: float, y: float) -> float:
    """Euclidean distance from the shot location to the nearest point on the goal line."""
    x_dist = GOAL_X - x
    if y < GOAL_Y_LEFT:
        y_dist = GOAL_Y_LEFT - y
    elif y > GOAL_Y_RIGHT:
        y_dist = y - GOAL_Y_RIGHT
    else:
        y_dist = 0
    return math.sqrt(x_dist ** 2 + y_dist ** 2)


def _preferable_side(row) -> int:
    # A right-footed shot from the left side of the box (and vice versa) allows a
    # finesse shot across the body, generally considered a higher-quality strike.
    side = "left" if row["location_y"] < 40 else ("right" if row["location_y"] > 40 else "center")
    foot = row["shot_body_part_name"]
    if (side == "left" and foot == "Right Foot") or (side == "right" and foot == "Left Foot"):
        return 1
    return 0


BOOL_FLAG_COLS = ["shot_first_time", "shot_one_on_one", "shot_open_goal", "shot_aerial_won", "under_pressure"]


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["angle"] = df.apply(lambda r: calculate_angle(r["location_x"], r["location_y"]), axis=1)
    df["distance"] = df.apply(lambda r: calculate_distance(r["location_x"], r["location_y"]), axis=1)
    df["header"] = (df["shot_body_part_name"] == "Head").astype(int)
    df["preferable_side"] = df.apply(_preferable_side, axis=1)

    for col in BOOL_FLAG_COLS:
        if col not in df.columns:
            df[col] = False
        df[col] = df[col].fillna(False).infer_objects(copy=False).astype(int)

    df = pd.get_dummies(df, columns=["shot_technique_name", "shot_type_name"], prefix=["technique", "type"])

    df["goal"] = (df["shot_outcome_name"] == "Goal").astype(int)
    return df


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    return df.dropna(subset=["location_x", "location_y", "shot_outcome_name"])


BASE_FEATURE_COLS = ["angle", "distance", "header", "preferable_side"] + BOOL_FLAG_COLS

TARGET_COL = "goal"


def get_X_y(df: pd.DataFrame):
    dummy_cols = [c for c in df.columns if c.startswith("technique_") or c.startswith("type_")]
    feature_cols = [c for c in BASE_FEATURE_COLS if c in df.columns] + dummy_cols

    subset = feature_cols + ([TARGET_COL] if TARGET_COL in df.columns else [])
    df_clean = df.dropna(subset=subset)

    y = df_clean[TARGET_COL] if TARGET_COL in df_clean.columns else None
    return df_clean[feature_cols], y
