# RaceTimePredictor

Predicting the race times of runs based on historical Strava data for 5K, 10K, Half-Marathon, and Marathon.

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
```

## Project layout

```
src/race_predictor/
├── data/        # Strava CSV loader, run normalization
├── features/    # Trailing fitness index (10 wk for 5K/10K, 12 wk for Half/Marathon)
├── models/      # VDOT, Riegel, ML corrector
├── confidence/  # Intervals and scores
└── evaluate/    # Backtest harness
```

Place your Strava bulk export in `data/`.

## Data loader

```python
from race_predictor.data import load_runs

runs = load_runs("data/activities.csv")
print(f"{len(runs)} runs loaded")
print(runs[-1].distance_mi, runs[-1].pace_min_per_mi)
```

All distances are in miles, elevation in feet, temperature in °F, pace in min/mile.

## Fitness features

```python
from race_predictor.data import load_runs
from race_predictor.features import compute_fitness_features

runs = load_runs("data/activities.csv")
features = compute_fitness_features(runs, runs[-1].date, "Marathon")
# 12-week window for Half/Marathon; 10-week for 5K/10K
```

## Baseline predictions (VDOT + Riegel)

```python
from race_predictor.data import load_runs
from race_predictor.models.baseline import predict_all_baselines
from race_predictor.formatting import format_time, format_pace

runs = load_runs("data/activities.csv")
predictions = predict_all_baselines(
    runs,
    as_of=runs[-1].date,
    elev_gain_ft=492,
    elev_loss_ft=492,
    temp_f=72,
)
for p in predictions:
    print(p.distance_label, format_time(p.predicted_time_sec), format_pace(p.pace_min_per_mi))
```

Blends **60% VDOT** (from weighted recent fitness) + **40% Riegel** (from best recent effort), then adjusts for elevation (ft) and temperature (°F).

## Hybrid predictions (baseline + ML residual)

```python
from race_predictor.data import load_runs
from race_predictor.models.predictor import train, predict_all
from race_predictor.formatting import format_time, format_pace

runs = load_runs("data/activities.csv")
model = train(runs, "models/trained_model.pkl")

predictions = predict_all(
    runs,
    model,
    as_of=runs[-1].date,
    elev_gain_ft=492,
    elev_loss_ft=492,
    temp_f=72,
)
for p in predictions:
    print(
        p.distance_label,
        format_time(p.predicted_time_sec),
        format_pace(p.pace_min_per_mi),
        f"conf {p.confidence}",
        f"{format_time(p.interval_low_sec)}–{format_time(p.interval_high_sec)}",
    )
```

## Backtest evaluation

```python
from race_predictor.data import load_runs
from race_predictor.evaluate import run_backtest, backtest_to_dict

runs = load_runs("data/activities.csv")
result = run_backtest(runs)
for m in result.metrics:
    if m.count:
        print(m.label, f"MAPE {m.mape:.1%}", f"±5% {m.within_5_pct:.0%}", f"80% cov {m.interval_coverage_80:.0%}")
```

Confidence (0–100) combines data sufficiency, model agreement, backtest reliability, and extrapolation penalty. Each prediction includes an 80% interval.

## Tests

```bash
pytest
```
