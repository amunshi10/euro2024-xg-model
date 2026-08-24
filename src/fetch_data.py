"""
Download Euro 2024 shot event data from StatsBomb's free open-data repo
and cache it as a flat CSV of shots.

Usage:
    python src/fetch_data.py --output data/euro2024_shots.csv
"""

import argparse

import pandas as pd
import requests

RAW_BASE = "https://raw.githubusercontent.com/statsbomb/open-data/master/data"
COMPETITION_ID = 55  # UEFA Euro
SEASON_ID = 282  # 2024

KEEP_COLUMNS = [
    "id", "match_id", "team_name", "player_name", "minute", "second", "period",
    "location_x", "location_y", "play_pattern_name",
    "shot_statsbomb_xg", "shot_outcome_name", "shot_type_name",
    "shot_technique_name", "shot_body_part_name",
    "shot_first_time", "shot_one_on_one", "shot_open_goal", "shot_aerial_won",
    "under_pressure",
]


def _get_json(path: str):
    resp = requests.get(f"{RAW_BASE}/{path}", timeout=30)
    resp.raise_for_status()
    return resp.json()


def fetch_matches() -> pd.DataFrame:
    matches = _get_json(f"matches/{COMPETITION_ID}/{SEASON_ID}.json")
    return pd.json_normalize(matches, sep="_")


def fetch_shots_for_match(match_id: int) -> pd.DataFrame:
    events = _get_json(f"events/{match_id}.json")
    # period <= 4 excludes penalty shoot-outs, which have very different scoring odds
    shots = [e for e in events if e.get("type", {}).get("name") == "Shot" and e.get("period", 0) <= 4]
    if not shots:
        return pd.DataFrame()
    df = pd.json_normalize(shots, sep="_")
    df["match_id"] = match_id
    return df


def fetch_all_shots() -> pd.DataFrame:
    df_matches = fetch_matches()
    frames = []
    for i, match_id in enumerate(df_matches["match_id"]):
        df = fetch_shots_for_match(match_id)
        if not df.empty:
            frames.append(df)
        print(f"[{i + 1}/{len(df_matches)}] match {match_id}: {len(df)} shots")
    df_shots = pd.concat(frames, ignore_index=True)

    df_shots["location_x"] = df_shots["location"].apply(lambda loc: loc[0] if isinstance(loc, list) else None)
    df_shots["location_y"] = df_shots["location"].apply(lambda loc: loc[1] if isinstance(loc, list) else None)

    keep = [c for c in KEEP_COLUMNS if c in df_shots.columns]
    return df_shots[keep]


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="data/euro2024_shots.csv")
    args = parser.parse_args()
    df = fetch_all_shots()
    df.to_csv(args.output, index=False, encoding="utf-8")
    print(f"\nSaved {len(df)} shots to {args.output}")
