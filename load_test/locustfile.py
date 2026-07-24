"""
load_test/locustfile.py
─────────────────────────────────────────────────────────────────────────────
Load test for the Rossmann Forecasting API.

See load_test/load_test_results.md for full documentation, sample results,
and how to interpret them. Quick reference below.

The API must be actually RUNNING with real loaded model artifacts for this
to be meaningful — it hits real HTTP endpoints, not mocks. Start the API
first (locally or against a deployed instance), then point Locust at it.

Run from the PROJECT ROOT (interactive web UI — open http://localhost:8089
to configure & watch):
    locust -f load_test/locustfile.py --host http://localhost:8000

Run (headless, e.g. for a scripted load test with a report):
    locust -f load_test/locustfile.py --host http://localhost:8000 \
        --headless -u 50 -r 5 -t 2m \
        --csv=load_test/results --html=load_test/report.html

    -u 50   : simulate 50 concurrent users
    -r 5    : ramp up 5 new users/second until reaching -u
    -t 2m   : run for 2 minutes then stop automatically
    --csv   : writes results_stats.csv, results_failures.csv, etc.
    --html  : writes a single-file HTML report

What to look at afterward:
    - Median / p95 / p99 response time for POST /api/v1/predict specifically
      — this is the endpoint that actually runs feature engineering +
      XGBoost inference, so it's the one that matters most under load.
    - Failure rate — should be 0% at normal load; a non-zero rate under
      concurrency can reveal thread-safety issues in how the model artifact
      singleton is accessed (see app/model_loader.py) that a single-request
      manual test would never surface.
    - Requests/sec at the point response time starts climbing sharply —
      that's your practical throughput ceiling on the current hardware.
"""

import random
from datetime import date, timedelta

from locust import HttpUser, task, between


# ── Valid ranges — kept in sync with app/schemas/request.py ──────────────────
STORE_ID_MIN, STORE_ID_MAX = 1, 1115
HORIZON_MIN, HORIZON_MAX   = 1, 90
DATE_MIN = date(2013, 1, 1)
DATE_MAX = date(2030, 12, 31)
STATE_HOLIDAYS = ["0", "0", "0", "0", "a", "b", "c"]  # weighted toward "none"


def _random_target_date() -> date:
    span_days = (DATE_MAX - DATE_MIN).days
    return DATE_MIN + timedelta(days=random.randint(0, span_days))


def _random_predict_payload() -> dict:
    """
    Build an always-VALID, randomized request body — every field satisfies
    the constraints in app/schemas/request.py, so every request should
    reach the model and return 200, not 422. That matters here: the whole
    point is measuring inference latency under load, not re-testing
    validation (tests/test_api_integration.py already covers that).
    """
    target_date = _random_target_date()

    payload = {
        "store_id":       random.randint(STORE_ID_MIN, STORE_ID_MAX),
        "horizon_days":   random.randint(HORIZON_MIN, HORIZON_MAX),
        "year":           target_date.year,
        "month":          target_date.month,
        "day":            target_date.day,
        "day_of_week":    target_date.isoweekday(),  # 1=Mon..7=Sun, matches schema
        "promo":          random.choice([0, 1]),
        "state_holiday":  random.choice(STATE_HOLIDAYS),
        "school_holiday": random.choice([0, 1]),
    }

    # ~30% of requests exercise the lag-override code path instead of the
    # default per-store lookup — a distinct branch in prediction_service.py
    # that deserves its own coverage under load.
    if random.random() < 0.3:
        payload.update({
            "lag_7":       round(random.uniform(2000, 9000), 2),
            "lag_14":      round(random.uniform(2000, 9000), 2),
            "roll_7_mean": round(random.uniform(2000, 9000), 2),
        })

    return payload


class RossmannAPIUser(HttpUser):
    """Simulates a client of the Rossmann Forecasting API."""

    # Random think-time between requests per simulated user, in seconds —
    # avoids every virtual user firing in lockstep, which would understate
    # real-world request spacing.
    wait_time = between(0.5, 2.0)

    @task(10)
    def predict(self):
        """POST /api/v1/predict — the primary, latency-critical endpoint."""
        payload = _random_predict_payload()
        with self.client.post(
            "/api/v1/predict", json=payload, catch_response=True,
        ) as response:
            if response.status_code != 200:
                response.failure(
                    f"Expected 200, got {response.status_code}: {response.text[:200]}"
                )

    @task(2)
    def health(self):
        """GET /health — cheap, but worth including in a realistic mix."""
        self.client.get("/health")

    @task(1)
    def version(self):
        """GET /version."""
        self.client.get("/version")

    @task(1)
    def root(self):
        """GET /."""
        self.client.get("/")