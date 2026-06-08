# RaceTimePredictor

Predict race times for **5K**, **10K**, **Half-Marathon**, and **Marathon** from trailing Strava training data. All units are US customary (miles, feet, °F, min/mile).

The model supports two training modes:

- **Athlete** — train the ML residual on your own Strava history (time-series holdouts).
- **Population** — train the residual on thousands of public race finishers from RunSignup, then predict using your Strava fitness baseline.

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
```

Place your Strava bulk export in `data/` (must include `activities.csv`).

Optional: copy `.env.example` to `.env` if you plan to use RunSignup OAuth or partner API keys. Public race results can be synced without credentials.

## CLI

```bash
# Athlete mode: train on your Strava export
race-predictor train --athlete --data-dir data/

# Population mode: train on RunSignup finisher corpus (default when corpus exists)
race-predictor sync-corpus
race-predictor train --population

# Predict all four distances for a race (uses your Strava history + trained model)
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

# Optional: RunSignup OAuth login (not required for public results)
race-predictor runsignup-login
```

`train` auto-selects population mode when `benchmarks/runsignup_corpus.csv` exists; pass `--athlete` to force Strava-only training.

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

1. **Data loader** — Parses Strava CSV (or benchmark corpus CSV), converts to miles / feet / °F
2. **Fitness index** — 10-week window for 5K/10K, 12-week for Half/Marathon
3. **Baseline** — 60% VDOT + 40% Riegel, adjusted for elevation and temperature
4. **ML residual** — Gradient Boosting correction:
   - *Athlete mode*: trained on your time-series holdouts
   - *Population mode*: trained on finisher residuals (actual time vs VDOT + course-adjusted expectation)
5. **Confidence** — Score (0–100) and 80% prediction intervals from backtest residuals

At prediction time, your Strava history always supplies the fitness baseline; a population-trained model only replaces the residual layer with corrections learned from many athletes.

## Phase 2 — Public benchmark

Validate generalization across athletes using a multi-athlete CSV corpus:

1. **Corpus loader** — `athlete_id` + activity rows (see `load_benchmark_corpus`)
2. **Athlete split** — Train on 75% of athletes, hold out 25% (no cross-athlete leakage)
3. **Variants** — Riegel-only, VDOT-only, UltraSignup-style rank, hybrid (baseline + ML)
4. **Report** — Per-distance MAPE comparison + generalization verdict in `reports/benchmark.json`

Add public datasets (NYRR, RunSignup exports, research corpora) as CSV files under `benchmarks/` using the same column schema as `benchmarks/sample_corpus.csv`.

## Phase 3.5 — RunSignup population corpus

Build a diverse multi-athlete training set from public RunSignup finisher results.

### Workflow

```bash
# 1. Sync curated races → benchmarks/runsignup_corpus.csv
race-predictor sync-corpus

# 2. Review coverage (distance, terrain, month, temperature)
cat reports/corpus_coverage.json

# 3. Train population residual model
race-predictor train --population

# 4. Predict with your Strava data + population model
race-predictor predict --elev-gain-ft 200 --elev-loss-ft 200 --temp-f 55
```

`sync-corpus` reads `catalog/races.json`, fetches race metadata and finisher times via the RunSignup REST API, geocodes each race city/state for historical weather, and writes a benchmark-style CSV. A coverage report validates:

| Dimension | Threshold |
|-----------|-----------|
| Distances | ≥2 races and ≥50 finishers each (5K, 10K, Half, Marathon) |
| Course types | flat, rolling, uphill, downhill all represented |
| Calendar | finishers in ≥8 distinct months |
| Temperature | cold (&lt;45°F), mild (45–65°F), warm (&gt;65°F) |

The bundled corpus has **~7,000 finishers** across **32 races**, with finishers in all 12 months and mixed terrain/climate conditions.

### Catalog format

Each entry in `catalog/races.json` specifies:

- `runsignup_race_id` — RunSignup race ID
- `distance_label` — `5K`, `10K`, `Half`, or `Marathon`
- `course_type` — `flat`, `rolling`, `uphill`, or `downhill`
- `elev_gain_ft` / `elev_loss_ft` — course elevation metadata
- `event_name_contains` / `event_name_excludes` — match the correct event within a multi-distance race
- `runsignup_event_id` — optional, when multiple events share a name
- `typical_temp_f` — optional fallback if geocoding/weather lookup fails

### RunSignup client

`src/race_predictor/data/runsignup_client.py` fetches public race detail and paginated finisher results. OAuth (`runsignup-login`) and partner API keys (`.env`) are optional — published results are readable without authentication.

### Weather

Race-day temperatures come from [Open-Meteo](https://open-meteo.com/): city/state is geocoded when RunSignup does not provide coordinates, then historical archive data is used for past race dates.

## Web UI

A Next.js frontend (`web/`) talks to a FastAPI backend (`api/`):

```bash
# Terminal 1 — API (requires data/activities.csv)
uvicorn api.main:app --reload --port 8000

# Terminal 2 — frontend
cd web && npm install && npm run dev
```

Open `http://localhost:3000`, enter elevation and temperature, and get predictions for all four distances. Train a population model via CLI first (`sync-corpus` + `train --population`); the API auto-trains athlete mode only when no model file exists.

## Project layout

```
catalog/
  races.json              # Curated RunSignup race catalog
benchmarks/
  runsignup_corpus.csv    # Synced finisher training corpus
  sample_corpus.csv       # Small example for Phase 2 benchmark
reports/
  corpus_coverage.json    # Sync coverage validation
api/                      # FastAPI prediction service
web/                      # Next.js UI
src/race_predictor/
├── data/                 # Strava loader, RunSignup client, corpus sync, weather
├── features/             # Trailing fitness index
├── models/               # VDOT, Riegel, residual, population trainer, predictor
├── confidence/           # Confidence scores and intervals
├── evaluate/             # Backtest + athlete-level benchmark
└── cli.py                # train / predict / sync-corpus / evaluate / benchmark
```

## Python API

```python
from race_predictor.data import load_runs
from race_predictor.data.benchmark_loader import load_benchmark_corpus
from race_predictor.models.predictor import train, train_population, predict_all, load_model
from race_predictor.evaluate import run_backtest, run_benchmark

# Athlete training
runs = load_runs("data/activities.csv")
model = train(runs)
predictions = predict_all(runs, model, runs[-1].date, elev_gain_ft=492, elev_loss_ft=492, temp_f=72)

# Population training
corpus_runs = load_benchmark_corpus("benchmarks/runsignup_corpus.csv")
pop_model = train_population(corpus_runs)
predictions = predict_all(runs, pop_model, runs[-1].date, elev_gain_ft=200, elev_loss_ft=200, temp_f=55)

result = run_backtest(runs, model)
benchmark = run_benchmark(corpus_runs, holdout_frac=0.25, seed=42)
```

## Tests

```bash
pytest
```
