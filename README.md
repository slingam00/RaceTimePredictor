# RaceTimePredictor

Predict race times for **5K**, **10K**, **Half-Marathon**, and **Marathon** from trailing Strava training data. All units are US customary (miles, feet, °F, min/mile).

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
```

Place your Strava bulk export in `data/` (must include `activities.csv`).

## CLI

```bash
# Train hybrid model (VDOT/Riegel baseline + ML residual)
race-predictor train --data-dir data/

# Predict all four distances for a race
race-predictor predict \
  --elev-gain-ft 492 \
  --elev-loss-ft 492 \
  --temp-f 72 \
  --as-of 2026-06-01

# Run time-series backtest (single athlete)
race-predictor evaluate --data-dir data/ --output reports/backtest.json

# Phase 2: athlete-level public benchmark
race-predictor benchmark \
  --corpus benchmarks/sample_corpus.csv \
  --holdout-frac 0.25 \
  --output reports/benchmark.json
```

### Example output

```
Distance   Predicted    Pace         80% Interval           Conf
--------------------------------------------------------------------
5K         29:49        9:36/mi      29:05 – 30:34          82
10K        1:05:45      10:35/mi     1:05:09 – 1:06:20      71
Half       2:21:30      10:48/mi     2:20:33 – 2:22:28      58
Marathon   5:06:12      11:41/mi     4:10:58 – 6:01:26      42
```

## How it works

1. **Data loader** — Parses Strava CSV, converts to miles / feet / °F
2. **Fitness index** — 10-week window for 5K/10K, 12-week for Half/Marathon
3. **Baseline** — 60% VDOT + 40% Riegel, adjusted for elevation and temperature
4. **ML residual** — Gradient Boosting correction trained on time-series holdouts
5. **Confidence** — Score (0–100) and 80% prediction intervals from backtest residuals

## Phase 2 — Public benchmark

Validate generalization across athletes using a multi-athlete CSV corpus:

1. **Corpus loader** — `athlete_id` + activity rows (see `load_benchmark_corpus`)
2. **Athlete split** — Train on 75% of athletes, hold out 25% (no cross-athlete leakage)
3. **Variants** — Riegel-only, VDOT-only, UltraSignup-style rank, hybrid (baseline + ML)
4. **Report** — Per-distance MAPE comparison + generalization verdict in `reports/benchmark.json`

Add public datasets (NYRR, RunSignup exports, research corpora) as CSV files under `benchmarks/` using the same column schema as `benchmarks/sample_corpus.csv`.

## Project layout

```
src/race_predictor/
├── data/        # Strava CSV loader
├── features/    # Trailing fitness index
├── models/      # VDOT, Riegel, ML residual, predictor
├── confidence/  # Confidence scores and intervals
├── evaluate/    # Backtest + athlete-level benchmark
└── cli.py       # train / predict / evaluate / benchmark commands
```

## Python API

```python
from race_predictor.data import load_runs
from race_predictor.models.predictor import train, predict_all
from race_predictor.data import load_benchmark_corpus
from race_predictor.evaluate import run_backtest, run_benchmark

runs = load_runs("data/activities.csv")
model = train(runs)
predictions = predict_all(runs, model, runs[-1].date, elev_gain_ft=492, elev_loss_ft=492, temp_f=72)
result = run_backtest(runs, model)

corpus = load_benchmark_corpus("benchmarks/sample_corpus.csv")
benchmark = run_benchmark(corpus, holdout_frac=0.25, seed=42)
```

## Tests

```bash
pytest
```
