"""
Fetch StatsBomb 360 freeze-frame data for Euro 2024, join it with shot events,
and engineer defender/goalkeeper positioning features.

360 freeze frames capture every visible player's location at the moment of a shot,
tagged as teammate/opponent/keeper (no identity). This lets us build features that
raw event data can't: how many defenders are between the shooter and goal, how far
the keeper is off their line, how close the nearest defender is.

Usage:
    python src/fetch_360.py --output data/euro2024_shots_360.csv
"""

import argparse
import math

import numpy as np
import pandas as pd

from fetch_data import _get_json, fetch_all_shots, fetch_matches

GOAL_POST_TOP = (120, 44)
GOAL_POST_BOTTOM = (120, 36)


def _sign(p1, p2, p3):
    return (p1[0] - p3[0]) * (p2[1] - p3[1]) - (p2[0] - p3[0]) * (p1[1] - p3[1])


def _point_in_triangle(pt, a, b, c) -> bool:
    d1, d2, d3 = _sign(pt, a, b), _sign(pt, b, c), _sign(pt, c, a)
    has_neg = (d1 < 0) or (d2 < 0) or (d3 < 0)
    has_pos = (d1 > 0) or (d2 > 0) or (d3 > 0)
    return not (has_neg and has_pos)


def fetch_360_for_match(match_id: int) -> dict:
    try:
        frames = _get_json(f"three-sixty/{match_id}.json")
    except Exception:
        return {}
    return {f["event_uuid"]: f for f in frames}


def compute_freeze_frame_features(shot_x: float, shot_y: float, freeze_frame) -> dict:
    if not freeze_frame:
        return {
            "defenders_in_cone": np.nan,
            "nearest_opponent_distance": np.nan,
            "opponents_within_5m": np.nan,
            "keeper_distance_to_shot": np.nan,
            "keeper_distance_to_goal_line": np.nan,
            "has_360": 0,
        }

    shot_pt = (shot_x, shot_y)
    opponents = [p for p in freeze_frame if not p.get("teammate", True)]
    opponent_outfield = [p["location"] for p in opponents if not p.get("keeper", False)]
    keeper = next((p["location"] for p in opponents if p.get("keeper", False)), None)

    defenders_in_cone = sum(
        1 for loc in opponent_outfield
        if loc[0] > shot_x and _point_in_triangle(loc, shot_pt, GOAL_POST_TOP, GOAL_POST_BOTTOM)
    )
    distances = [math.dist(shot_pt, loc) for loc in opponent_outfield]

    return {
        "defenders_in_cone": defenders_in_cone,
        "nearest_opponent_distance": min(distances) if distances else np.nan,
        "opponents_within_5m": sum(1 for d in distances if d <= 5),
        "keeper_distance_to_shot": math.dist(shot_pt, keeper) if keeper else np.nan,
        "keeper_distance_to_goal_line": (120 - keeper[0]) if keeper else np.nan,
        "has_360": 1,
    }


def build_360_dataset() -> pd.DataFrame:
    df_shots = fetch_all_shots()
    df_matches = fetch_matches()

    feature_rows = []
    for i, match_id in enumerate(df_matches["match_id"]):
        lookup = fetch_360_for_match(match_id)
        match_shots = df_shots[df_shots["match_id"] == match_id]

        n_with_frame = 0
        for _, row in match_shots.iterrows():
            frame_entry = lookup.get(row["id"])
            freeze_frame = frame_entry["freeze_frame"] if frame_entry else None
            feats = compute_freeze_frame_features(row["location_x"], row["location_y"], freeze_frame)
            feats["id"] = row["id"]
            feature_rows.append(feats)
            n_with_frame += feats["has_360"]

        print(f"[{i + 1}/{len(df_matches)}] match {match_id}: "
              f"{n_with_frame}/{len(match_shots)} shots with 360 data")

    df_features = pd.DataFrame(feature_rows)
    return df_shots.merge(df_features, on="id", how="left")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="data/euro2024_shots_360.csv")
    args = parser.parse_args()

    df = build_360_dataset()
    df.to_csv(args.output, index=False, encoding="utf-8")
    print(f"\nSaved {len(df)} shots to {args.output}")
    print(f"360 coverage: {df['has_360'].mean():.1%}")
