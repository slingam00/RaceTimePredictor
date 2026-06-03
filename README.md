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

## Tests

```bash
pytest
```
