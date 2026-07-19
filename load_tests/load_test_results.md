# Load Testing — Rossmann Forecasting API

Load testing for the API using [Locust](https://locust.io/). Simulates concurrent users hitting the real endpoints — most importantly `POST /api/v1/predict`, the only endpoint that runs actual feature engineering and XGBoost inference rather than just reading in-memory state.

## Why this exists

Model accuracy (RMSPE) was rigorously validated elsewhere in this project — the rolling multi-origin backtest, root-cause error analysis, etc. What that work doesn't tell you is how the API *behaves under concurrent traffic*: response time as load increases, whether the model-artifact singleton in `app/model_loader.py` holds up under concurrent access, and where throughput actually tops out on real hardware. That's what this directory is for.

## Contents

```
load_test/
├── load_test_results.md   — this file
└── locustfile.py           — the Locust test definition
```

## Prerequisites

- `locust` installed (already in `requirements-dev.txt`: `pip install -r requirements-dev.txt`)
- The API **actually running** with real loaded model artifacts — this hits real HTTP endpoints, not mocks. Unlike `tests/`, there's nothing to fake here; if `artifacts/models/global_model.pkl` doesn't exist yet, run `notebook/Time_Series_Save_Production_Model.py` first.

## What gets tested

`locustfile.py` simulates a realistic mix of traffic:

| Endpoint | Weight | Why |
|---|---|---|
| `POST /api/v1/predict` | 10 | The real workload — feature engineering + XGBoost inference. This is what matters. |
| `GET /health` | 2 | Cheap, but a realistic client would still poll it. |
| `GET /version` | 1 | Cheap. |
| `GET /` | 1 | Cheap. |

Every `/predict` request is built with a **randomized but always-valid** payload — `store_id` (1–1115), `horizon_days` (1–90, sweeping across all four horizon buckets), and a genuinely random calendar date (so `year`/`month`/`day`/`day_of_week` are always mutually consistent). About 30% of requests also include lag-feature overrides, exercising that code path separately from the default per-store lookup path. The goal is measuring real inference latency under load — not re-testing request validation, which `tests/test_api_integration.py` already covers.

## Running it

All commands below assume you're in the **project root** (not inside `load_test/`).

**1. Start the API** in one terminal:
```bash
uvicorn app.main:app --host 0.0.0.0 --port 8000
```

**2. Run Locust** in a second terminal.

Interactive (recommended first time — a web UI where you can watch live and adjust user count on the fly):
```bash
locust -f load_test/locustfile.py --host http://localhost:8000
```
Then open <http://localhost:8089>, set the number of users and spawn rate, and start.

Headless (for a scripted run with a saved report — useful for CI or repeatable benchmarking):
```bash
locust -f load_test/locustfile.py --host http://localhost:8000 \
    --headless -u 50 -r 5 -t 2m \
    --csv=load_test/results --html=load_test/report.html
```

| Flag | Meaning |
|---|---|
| `-u 50` | Simulate 50 concurrent users |
| `-r 5` | Ramp up 5 new users/second until reaching `-u` |
| `-t 2m` | Run for 2 minutes, then stop automatically |
| `--csv` | Writes `results_stats.csv`, `results_failures.csv`, etc. |
| `--html` | Writes a single-file HTML report |

`load_test/results*.csv` and `load_test/report.html` are test output, not source — gitignored, not something to commit.

## What to look at

- **p95 / p99 response time for `POST /api/v1/predict` specifically** — the endpoint doing real work, so the one that matters most.
- **Failure rate** — should be 0% at normal load. A non-zero rate under concurrency (not at low load) can reveal thread-safety issues in how the model artifact singleton is accessed, which a single manual request would never surface.
- **Requests/sec at the point response time starts climbing sharply** — that's your practical throughput ceiling on the current hardware.
- Push `-u` up incrementally (50 → 100 → 200) across separate runs to find where that ceiling actually is, rather than guessing from a single run.

## Reference baseline (sandbox verification run)

This is **not** a benchmark of your production hardware — it's the result of a quick verification run in a lightweight sandbox environment, included only to show the tool is wired correctly end-to-end and to give a rough shape of what output to expect. Re-run this yourself on real hardware before drawing any real conclusions.

```
locust -f load_test/locustfile.py --host http://localhost:8000 --headless -u 20 -r 5 -t 20s
```

| Endpoint | Requests | Failures | Median | p95 | p99 |
|---|---|---|---|---|---|
| `POST /api/v1/predict` | 211 | 0 (0.00%) | 3 ms | 15 ms | 21 ms |
| `GET /health` | 46 | 0 (0.00%) | 2 ms | 10 ms | 13 ms |
| `GET /version` | 21 | 0 (0.00%) | 2 ms | 3 ms | 17 ms |
| `GET /` | 24 | 0 (0.00%) | 2 ms | 2 ms | 3 ms |
| **Aggregated** | **302** | **0 (0.00%)** | **3 ms** | **12 ms** | **20 ms** |

Zero failures at 20 concurrent users is a reasonable sign, but far from a real ceiling — it wasn't pushed anywhere near hard enough to find one.
