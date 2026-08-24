# Euro 2024 Expected Goals (xG) Model

A machine learning project that builds an expected goals (xG) model from scratch using
StatsBomb's free open shot event data for UEFA Euro 2024, and compares it against
StatsBomb's own published xG values.

## Overview

- **Dataset:** 1,316 shots from every UEFA Euro 2024 match (StatsBomb open data,
  `competition_id=55`, `season_id=282`)
- **Target:** Probability a shot results in a goal (binary classification)
- **Models:** Logistic Regression, Random Forest, Gradient Boosting, Hist Gradient Boosting
- **Best model:** Random Forest — Log Loss 0.254, ROC-AUC 0.728, Brier 0.069

## Repository Structure

```
├── data/
│   └── euro2024_shots.csv
├── notebooks/
│   └── euro2024_xg_model.ipynb
├── src/
│   ├── fetch_data.py
│   ├── data_preprocessing.py
│   ├── models.py
│   ├── train.py
│   └── predict.py
├── outputs/
├── models/
└── requirements.txt
```

## Features

**Geometric shot features:**
- Angle subtended by the goal mouth from the shot location
- Distance to the nearest point on the goal line
- "Preferable side" — whether the shooting foot matches the finesse angle for that side

**Shot context features:**
- Body part (header vs. foot), shot technique, shot type (open play / free kick / penalty / corner)
- First-time shot, one-on-one, open goal, aerial ball won, under pressure

**Model evaluation:**
- Stratified 5-fold cross-validation (ROC-AUC)
- Held-out test metrics: log loss, ROC-AUC, Brier score
- Reliability/calibration curve against StatsBomb's own `shot_statsbomb_xg`
- All runs logged to MLflow for reproducibility

## Results

| Model | CV AUC | Test Log Loss | Test ROC AUC | Test Brier |
|-------|--------|----------------|--------------|------------|
| Random Forest | 0.766 | 0.254 | 0.728 | 0.069 |
| Logistic Regression | 0.751 | 0.256 | 0.737 | 0.071 |
| Gradient Boosting | 0.732 | 0.283 | 0.669 | 0.076 |
| Hist Gradient Boosting | 0.742 | 0.430 | 0.627 | 0.088 |

## Data Source

Shot events are pulled directly from StatsBomb's free
[open-data](https://github.com/statsbomb/open-data) GitHub repository via
`src/fetch_data.py`. No API key or account is required — this is one of the few
non-historical competitions StatsBomb has released in full for free.

## Usage

```bash
pip install -r requirements.txt
python src/fetch_data.py                          # cache shot data to data/euro2024_shots.csv
python src/train.py                                # train + log models, save the best one
python src/predict.py --input data/new_shots.csv   # score new shots
```

## Tech Stack

`Python` `scikit-learn` `pandas` `numpy` `MLflow` `matplotlib` `seaborn` `mplsoccer`
