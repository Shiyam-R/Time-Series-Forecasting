# Rossmann Store Sales Forecasting API

A production-quality, multi-horizon sales forecasting system built on the [Rossmann Store Sales](https://www.kaggle.com/c/rossmann-store-sales) dataset — from raw-data preprocessing through a containerized, CI-tested FastAPI inference service.

[![CI](https://github.com/<your-username>/Time-Series-Forecasting/actions/workflows/ci.yml/badge.svg)](https://github.com/<your-username>/Time-Series-Forecasting/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

> Replace `<your-username>` in the badge URL above with your actual GitHub username/org once this is pushed, or the badge won't resolve.

---

## Overview

Given a store ID and a target date, the API returns a predicted daily sales figure (EUR). Under the hood, forecasts are routed to one of four independently trained XGBoost models based on how far ahead the target date is:

| Horizon bucket | Days ahead | Model |
|---|---|---|
| `near` | 1–14 | `near_model.pkl` |
| `mid` | 15–30 | `mid_model.pkl` |
| `far` | 31–60 | `far_model.pkl` |
| `extended` | 61–90 | `extended_model.pkl` |

This four-bucket, direct-forecasting design exists because sales-driving patterns (promotions, holidays, day-of-week effects) behave differently depending on how far out you're forecasting — a single model tuned across the full 1–90 day range underperforms four models each specialized to their own horizon.

### What makes this more than a training script

- **Fair, multi-origin backtesting.** Early evaluation used a single test window that happened to concentrate around Easter 2015, making the `near` horizon look artificially worse than it really was — a calendar-composition artifact, not a genuine model deficiency. The evaluation was rebuilt as a rolling, weekly-origin backtest to remove that bias.
- **Root-cause-driven feature engineering.** Rather than adding features speculatively, errors were traced to their sources first — e.g., September-origin forecasts loading disproportionately onto October due to German Unity Day and Herbstferien school holidays, and November errors concentrated in Store Type A weekend observations (with Store 652 as the single largest contributor). Interaction features were then added only where they addressed a diagnosed error source.
- **Explainability.** SHAP-based analysis of feature contributions backs the error analysis above, rather than treating the model as a black box.

## Project Structure

```
Time-Series-Forecasting/
├── app/
│   ├── main.py                  # FastAPI application factory + lifespan
│   ├── config.py                # paths, constants, env-var configuration
│   ├── exceptions.py            # custom exception hierarchy
│   ├── model_loader.py          # artifact loading singleton
│   ├── api/routes.py            # /, /health, /version, /api/v1/predict
│   ├── schemas/{request,response}.py
│   ├── services/prediction_service.py
│   └── utils/{preprocessing,feature_engineering,logger}.py
├── artifacts/
│   └── models/                  # production model artifacts (near/mid/far/extended)
├── data/                        # raw + processed data (gitignored — see below)
├── figures/                     # root-cause analysis plots
├── notebook/                    # numbered pipeline scripts (preprocessing → training → evaluation)
├── tests/                       # pytest suite
├── .github/workflows/ci.yml     # CI: install, test, syntax-check, Docker build
├── Dockerfile                   # multi-stage production image
├── requirements.txt             # production dependencies
├── requirements-dev.txt         # + testing tooling
└── .env.example                 # documented environment variables
```

`data/` isn't tracked in git (see `.gitignore`) — it's the raw/processed Kaggle CSVs, which are large and regenerable via the numbered scripts in `notebook/`, not source code.

## Getting Started

### Prerequisites

- Python 3.13
- Docker (optional, for containerized runs)

### Local setup

```bash
git clone https://github.com/<your-username>/Time-Series-Forecasting.git
cd Time-Series-Forecasting

# Production + testing dependencies
pip install -r requirements-dev.txt

# Optional: local environment overrides
cp .env.example .env
```

### Run locally

```bash
uvicorn app.main:app --reload
```

- Swagger UI: <http://localhost:8000/docs>
- ReDoc: <http://localhost:8000/redoc>

### Run with Docker

```bash
docker build -t rossmann-forecasting-api .
docker run -p 8000:8000 rossmann-forecasting-api
```

## API Reference

| Method | Endpoint | Purpose |
|---|---|---|
| `GET` | `/` | Project metadata and available endpoints |
| `GET` | `/health` | Reports whether all four models are loaded (`healthy` / `degraded`) |
| `GET` | `/version` | Reports API version, model version, and deployment environment |
| `POST` | `/api/v1/predict` | Generates a sales forecast |

The full request/response schema — including every field, validation rule, and example — is generated automatically from the Pydantic models and is browsable at `/docs`. That's the authoritative reference; a couple of the core fields for `POST /api/v1/predict`:

```json
{
  "store_id": 652,
  "horizon_days": 90,
  "year": 2015,
  "month": 11,
  "day": 14,
  "day_of_week": 6,
  "promo": 1
}
```

```json
{
  "status": "success",
  "prediction": {
    "predicted_sales": 7072.26,
    "target_date": "2015-11-14"
  },
  "forecast_details": {
    "horizon_days": 90,
    "horizon_bucket": "extended"
  },
  "metadata": {
    "store_id": 652,
    "model": "xgboost_extended",
    "model_version": "v1.0.0",
    "prediction_timestamp": "2026-07-05T06:47:56Z"
  }
}
```

Lag/rolling-window features (`lag_7`, `roll_7_mean`, etc.) are optional — if omitted, the API falls back to precomputed per-store training defaults. Supplying recent actual sales as overrides improves accuracy.

## Configuration

All configuration lives in `app/config.py`, with optional environment-variable overrides (see `.env.example`):

| Variable | Default | Purpose |
|---|---|---|
| `ENVIRONMENT` | `development` | Surfaced via `/version` |
| `LOG_LEVEL` | `INFO` | Console + file logging verbosity |
| `CORS_ALLOWED_ORIGINS` | `*` | Comma-separated allow-list, or `*` for all origins |

None of these need to be set — omitting all of them preserves the original hardcoded behavior exactly.

## Testing

```bash
pytest
```

Configuration lives in `pytest.ini` (test discovery, `asyncio_mode`).

## Load Testing

Concurrency and latency testing lives in `load_test/` (Locust) — separate from `tests/` since it exercises a live running instance of the API rather than mocked components. See [`load_test/load_test_results.md`](load_test/load_test_results.md) for setup, how to run it, and how to interpret results.

## CI/CD

Every push and pull request to `main` runs `.github/workflows/ci.yml`:

1. Checkout → Python 3.13 setup (with pip caching)
2. Verify `requirements.txt` installs cleanly on its own
3. Install `requirements-dev.txt`
4. Validate Python syntax (`compileall`)
5. Run the `pytest` suite
6. Build the production Docker image (only if the above all pass)

## Logging

Structured logs are written to console and to `logs/rossmann_api.log` (rotated at 10 MB, 5 backups retained). Level is controlled via `LOG_LEVEL`.

## License

[MIT](LICENSE)