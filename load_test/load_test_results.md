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

This is **not** a benchmark of your production hardware — it's the result of a quick verification run in a lightweight sandbox environment, included only to show the tool is wired correctly end-to-end and to give a rough shape of what output to expect.

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

## Production results — real deployment (GHCR image, before multi-worker fix)

Run against the actual deployed container. Full percentile breakdown for `POST /api/v1/predict`, the endpoint that matters:

| Percentile | 50 users | 200 users |
|---|---|---|
| 50% (median) | 28 ms | 1400 ms |
| 66% | 35 ms | 1400 ms |
| 75% | 41 ms | 1500 ms |
| 80% | 45 ms | 1500 ms |
| 90% | 60 ms | 1600 ms |
| 95% | 79 ms | 1600 ms |
| 98% | 130 ms | 1700 ms |
| 99% | **1000 ms** | 1700 ms |
| 99.9% | 1200 ms | 2700 ms |
| max | 1300 ms | 2800 ms |
| Total requests | 4417 | 9467 |
| `/predict` requests | 3156 | 6759 |

### Diagnosis

At 50 users, most requests are genuinely fast (p50=28ms, p95=79ms) — but there's a sharp cliff between p98 (130ms) and p99 (1000ms). At 200 users, the *entire* distribution shifted to ~1.4–1.7s — not just the tail. This pattern (fast baseline + a sudden cliff at moderate load, collapsing to uniformly slow at higher load) is the signature of requests **queueing for a limited thread pool**, not the model itself being slow — a single prediction takes low-single-digit milliseconds (see the sandbox baseline above).

**Root cause:** `POST /api/v1/predict` is a synchronous `def` route, so FastAPI/Starlette runs it in a per-process thread pool capped at 40 concurrent threads by default. The Dockerfile ran a **single Uvicorn process**, meaning that 40-thread cap was the entire container's capacity, regardless of the host's actual CPU core count.

### Fix applied

`Dockerfile` now runs multiple Uvicorn worker processes (`--workers`, default 4, overridable via the `WORKERS` env var at `docker run` time). Multiple processes — not just more threads within one process — give real parallelism, since each has its own thread pool and Python's GIL is per-process, letting CPU-bound XGBoost inference actually run in parallel across cores.

## Post-fix results — same deployment, multi-worker image

Re-run against the rebuilt image at the same two concurrency levels, for a direct before/after comparison on `POST /api/v1/predict`:

| Percentile | 50 users (before) | 50 users (after) | 200 users (before) | 200 users (after) |
|---|---|---|---|---|
| 50% (median) | 28 ms | **4 ms** | 1400 ms | **9 ms** |
| 66% | 35 ms | **5 ms** | 1400 ms | **13 ms** |
| 75% | 41 ms | **6 ms** | 1500 ms | **18 ms** |
| 80% | 45 ms | **6 ms** | 1500 ms | **22 ms** |
| 90% | 60 ms | **9 ms** | 1600 ms | **42 ms** |
| 95% | 79 ms | **13 ms** | 1600 ms | **68 ms** |
| 98% | 130 ms | **18 ms** | 1700 ms | **99 ms** |
| 99% | 1000 ms | **24 ms** | 1700 ms | **120 ms** |
| 99.9% | 1200 ms | **37 ms** | 2700 ms | **190 ms** |
| max | 1300 ms | **38 ms** | 2800 ms | **260 ms** |
| `/predict` requests served | 3156 | **3322** | 6759 | **13014** |

### Result

The fix resolved the problem completely, at both concurrency levels tested:

- **The p99 cliff at 50 users is gone.** Before: a sharp jump from p98=130ms to p99=1000ms — a small fraction of requests hitting the thread-pool wall. After: a smooth, low curve topping out at p99=24ms. No cliff.
- **At 200 users, the whole distribution collapsed back to fast.** Before: every percentile from p50 to max sat in the 1.4–2.8s range — total saturation. After: p50=9ms, p99=120ms, max=260ms — an order of magnitude (or more) faster across the board.
- **Throughput nearly doubled at 200 users** (6,759 → 13,014 `/predict` requests served in the same window) — direct evidence requests were previously stuck queueing rather than the model being the bottleneck, exactly as diagnosed.

Request mix ratios stayed consistent with the configured task weights across all four runs (`/predict` ≈ 71% of traffic in every case, matching the 10:2:1:1 weighting in `locustfile.py`), which is a reasonable indicator no requests were silently dropped or erroring out during either run — though confirming the raw failure count from Locust's own summary (not just the percentile table) is worth doing on any future run, since that number wasn't captured here.