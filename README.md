# Euro 2024 Expected Goals (xG) Model

Building an expected goals model from scratch, using StatsBomb's free open shot event
data for **UEFA Euro 2024**, and checking it against StatsBomb's own published xG for
the same shots.

Methodology follows [Alfian Hakim's "How to Build Your Own Expected Goals (xG) Model"](https://medium.com/@alf.19x/how-to-build-your-own-expected-goals-xg-model-2bd186dccdf7)
(which used World Cup 2022 data) — same geometric-feature approach, applied instead to
a full European Championship.

## Background

xG measures the quality of a chance rather than just counting shots: it's the
probability, between 0 and 1, that a given shot results in a goal, estimated from
thousands of historically similar shots. A tap-in from two yards might carry an xG of
0.8; a speculative strike from 30 yards might be 0.02. Every analytics provider
(StatsBomb, Opta, FBref) runs its own model trained on its own data and features, which
is why the same shot gets a different xG number depending on where you look — there's
no single "correct" xG, only models of varying sophistication. This project builds one
from first principles to see how far a handful of geometric and contextual features get
you toward matching a commercial model's output.

## Data

1,316 shots from all 51 Euro 2024 matches, pulled directly from StatsBomb's free
[open-data](https://github.com/statsbomb/open-data) repository
(`competition_id=55`, `season_id=282`) via `src/fetch_data.py` — no API key needed.

| Outcome | Count |
|---|---|
| Off Target | 397 |
| Blocked | 386 |
| Saved | 312 |
| **Goal** | **107** |
| Wayward | 81 |
| Post / Saved to Post / Saved Off Target | 33 |

Only an **8.1% conversion rate** — goals are the minority class by a wide margin, which
is why evaluation leans on log loss / Brier score / ROC-AUC rather than accuracy.

![All Euro 2024 shots — red for no goal, gold for goal](outputs/shot_map.png)

## Feature Engineering

Raw `(x, y)` shot locations aren't useful to a model directly — they need to become
geometry:

- **Angle** — the angle subtended by the goal mouth from the shot location (wide angle
  = clean sight of a big target)
- **Distance** — distance to the nearest point on the goal line

```python
def calculate_angle(x, y):
    g0 = np.array([120, 44]) - np.array([x, y])
    g1 = np.array([120, 36]) - np.array([x, y])
    angle = np.arctan2(np.linalg.det([g0, g1]), np.dot(g0, g1))
    return abs(np.degrees(angle))
```

Binning shots by angle and distance and plotting the observed goal rate per bin
confirms the intuition immediately — wider angle and shorter distance both push
scoring probability up, though not perfectly monotonically (defenders, keeper
positioning, and shot technique all add noise geometry alone can't capture):

![Goal probability vs angle and distance](outputs/feature_scatter.png)

On top of geometry, shot context is added as boolean and one-hot features: body part
(header vs. foot), "preferable side" (does the shooting foot match a finesse angle for
that side of the box), shot technique, shot type (open play / free kick / penalty /
corner), and situational flags (first-time, one-on-one, open goal, aerial ball won,
under pressure).

Correlating every feature against the goal outcome shows what actually predicts
scoring in this dataset:

![Feature correlation with goal](outputs/correlation_analysis.png)

Penalties (`type_Penalty`) and `angle` are the two strongest positive predictors;
`distance` is by far the strongest negative one — exactly what the geometric model
above predicts. Interestingly, `preferable_side` and `header` show almost no
correlation here, which pushes back a little on the intuitive assumption that
finesse-angle shots are meaningfully more clinical at this shot volume.

## Modelling

Four classifiers are trained as scikit-learn Pipelines (`StandardScaler` → classifier),
evaluated with stratified 5-fold cross-validation, and logged to MLflow:

| Model | CV AUC | Test Log Loss | Test ROC AUC | Test Brier |
|-------|--------|----------------|--------------|------------|
| **Random Forest** | **0.766** | **0.254** | **0.728** | **0.069** |
| Logistic Regression | 0.751 | 0.256 | 0.737 | 0.071 |
| Gradient Boosting | 0.732 | 0.283 | 0.669 | 0.076 |
| Hist Gradient Boosting | 0.742 | 0.430 | 0.627 | 0.088 |

![Model comparison across log loss, Brier score, and ROC-AUC](outputs/model_comparison.png)

Random Forest comes out on top on log loss and Brier score (the metrics that matter
most for a *probability* estimator, as opposed to a classifier that just needs to rank
shots correctly). Logistic Regression is a close second and, being linear/interpretable,
is arguably the more useful model if you want to reason about *why* a shot got its xG
value rather than just trust the number.

### Best Model: Discrimination & Calibration

![ROC curve and calibration curve for the Random Forest model](outputs/best_model_predictions.png)

The calibration curve tracks the diagonal reasonably well at low predicted xG (where
most shots live) and starts to run hot at the high end — the model's highest-confidence
predictions (penalties, open goals) slightly overstate the true goal rate in this
sample, likely just a small-sample effect given how few such shots exist in one
tournament.

### How Does It Compare to StatsBomb's Own xG?

| Metric | Our Model | StatsBomb xG |
|---|---|---|
| Log Loss | 0.254 | **0.230** |
| ROC AUC | 0.728 | **0.808** |
| Brier | 0.069 | **0.065** |

StatsBomb's model wins on every metric, which is expected — their xG is trained on
millions of shots across competitions and, crucially, uses **360 freeze-frame data**
(defender and goalkeeper positions at the moment of the shot) that this project doesn't
use. What's notable is *how close* a model built from nine features and 1,316 shots
gets to a commercial model with orders of magnitude more data behind it — geometry
alone explains most of what separates a good chance from a bad one.

## Team Over/Underperformance

Summing each team's shot xG across the tournament and comparing it to actual goals
scored separates "created good chances" from "actually took them":

![Team goals minus xG across Euro 2024](outputs/team_overperformance.png)

**Spain overperformed the most** (+5.2 — 14 goals from 8.79 xG), consistent with
winning the tournament on the back of clinical finishing from a young, attacking side.
**France underperformed the most** by a wide margin, scoring far fewer goals than their
chance quality suggested — a narrative that matched a lot of the tournament coverage
around their toothless attack despite reaching the semi-finals. Portugal and Croatia
also finished well below their xG, while Switzerland, Germany, and the Netherlands
outperformed theirs.

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

## Usage

```bash
pip install -r requirements.txt
python src/fetch_data.py                          # cache shot data to data/euro2024_shots.csv
python src/train.py                                # train + log models, save the best one
python src/predict.py --input data/new_shots.csv   # score new shots
```

## Key Findings

- **Angle and distance dominate** as predictors — the pure geometry of a chance
  explains most of its quality, exactly as the underlying physics of shooting suggests.
- **Shot type matters a lot**: penalties carry a much higher baseline scoring
  probability than any open-play feature combination can replicate, which is why
  `type_Penalty` is the single strongest correlate with scoring.
- A from-scratch model trained on ~1,300 shots tracks a commercial provider's
  proprietary xG reasonably closely on log loss and calibration, despite using far
  fewer features — the gap is mostly explained by StatsBomb's use of defender/keeper
  freeze-frame positioning, which this project doesn't have access to.
- **Team over/underperformance vs. xG** is a useful lens independent of shot volume —
  it separates teams that created chances from teams that actually converted them,
  and lines up with the eye-test from the tournament (Spain clinical, France wasteful).

## Tech Stack

`Python` `scikit-learn` `pandas` `numpy` `MLflow` `matplotlib` `seaborn` `mplsoccer`
