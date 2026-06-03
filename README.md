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
├── features/    # 10-week window + fitness index
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

## Tests

```bash
pytest
```
