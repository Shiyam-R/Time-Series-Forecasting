"""
tests/test_api_integration.py
─────────────────────────────────────────────────────────────────────────────
End-to-end integration tests using FastAPI's TestClient.

Unlike tests/test_prediction_service.py — which calls PredictionService
directly and never touches routing — these tests send real HTTP-shaped
requests through the full stack: route wiring, Pydantic request validation,
the prediction service, and the custom exception handler that converts
RossmannAPIError subclasses into structured JSON. This closes the gap that
service-level unit tests can't cover: a broken route decorator, a schema
field rename, or an exception that no longer maps to the expected response
shape would all be invisible to test_prediction_service.py but caught here.

Run:
    pytest tests/test_api_integration.py -v

All model artifacts are mocked so tests run without actual .pkl files —
the FastAPI lifespan's real load_artifacts() call is patched to a no-op,
and the shared `artifacts` singleton is populated directly instead.
"""

from datetime import datetime, timezone
from unittest.mock import MagicMock, patch

import pytest
from fastapi.testclient import TestClient

from app.exceptions import ModelNotLoadedError, PredictionError
from app.model_loader import artifacts as shared_artifacts
from app.schemas.response import (
    ForecastDetails,
    MetadataSection,
    PredictionResponse,
    PredictionSection,
)


# ── Fixtures ──────────────────────────────────────────────────────────────────
@pytest.fixture
def client():
    """
    A TestClient wired to the real FastAPI app, with artifact *loading*
    mocked out (no real .pkl/.json files needed) but the shared artifacts
    singleton populated with realistic-shaped mock data so route handlers
    that read it directly (e.g. /health) behave as if startup succeeded.
    """
    mock_model = MagicMock()
    mock_model.predict.return_value = [5000.0]

    shared_artifacts.models = {
        "near": mock_model, "mid": mock_model,
        "far": mock_model, "extended": mock_model,
    }
    shared_artifacts.feature_columns = ["month", "day", "year", "day_of_week"]
    shared_artifacts.store_metadata  = {"1": {"StoreType": "a", "Assortment": "a"}}
    shared_artifacts.lag_defaults    = {"1": {"lag_7": 4500.0}}
    shared_artifacts.loaded          = True

    # Prevent the lifespan from attempting to read real artifact files from
    # disk on startup — see app/main.py's lifespan(), which calls this.
    with patch("app.main.load_artifacts", return_value=shared_artifacts):
        with TestClient(app_under_test()) as test_client:
            yield test_client

    # Reset the singleton so tests don't leak state into each other.
    shared_artifacts.models = {}
    shared_artifacts.feature_columns = []
    shared_artifacts.store_metadata = {}
    shared_artifacts.lag_defaults = {}
    shared_artifacts.loaded = False


def app_under_test():
    """Import lazily so the load_artifacts patch above is active first."""
    from app.main import app
    return app


def _valid_payload(**overrides) -> dict:
    """A valid POST /api/v1/predict request body, with optional overrides."""
    payload = {
        "store_id":       1,
        "horizon_days":   7,
        "year":           2015,
        "month":          6,
        "day":            15,
        "day_of_week":    1,
        "promo":          0,
        "state_holiday":  "0",
        "school_holiday": 0,
    }
    payload.update(overrides)
    return payload


def _mock_prediction_response() -> PredictionResponse:
    """A canned successful PredictionResponse, matching the real schema."""
    return PredictionResponse(
        prediction=PredictionSection(predicted_sales=4321.5, target_date="2015-06-15"),
        forecast_details=ForecastDetails(horizon_days=7, horizon_bucket="near"),
        metadata=MetadataSection(
            store_id=1,
            model="xgboost_near",
            model_version="v1.0.0",
            prediction_timestamp=datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        ),
    )


# ── GET / ─────────────────────────────────────────────────────────────────────
class TestRootEndpoint:
    """GET / returns project metadata."""

    def test_returns_200(self, client) -> None:
        response = client.get("/")
        assert response.status_code == 200

    def test_response_shape(self, client) -> None:
        body = client.get("/").json()
        assert "name" in body
        assert "version" in body
        assert "endpoints" in body
        assert isinstance(body["endpoints"], list)
        assert len(body["endpoints"]) >= 4


# ── GET /health ───────────────────────────────────────────────────────────────
class TestHealthEndpoint:
    """GET /health reports model load status."""

    def test_healthy_when_all_buckets_loaded(self, client) -> None:
        response = client.get("/health")
        assert response.status_code == 200
        body = response.json()
        assert body["status"] == "healthy"
        assert len(body["models_loaded"]) == 4
        assert all(m["loaded"] for m in body["models_loaded"])

    def test_degraded_when_a_bucket_is_missing(self, client) -> None:
        # Simulate a partial artifact-load failure at startup.
        del shared_artifacts.models["extended"]
        try:
            response = client.get("/health")
            body = response.json()
            assert body["status"] == "degraded"
            extended_status = next(
                m for m in body["models_loaded"] if m["bucket"] == "extended"
            )
            assert extended_status["loaded"] is False
        finally:
            # Restore for any subsequent test using this fixture instance.
            shared_artifacts.models["extended"] = MagicMock()


# ── GET /version ──────────────────────────────────────────────────────────────
class TestVersionEndpoint:
    """GET /version reports build/deploy identity, distinct from /health."""

    def test_returns_200(self, client) -> None:
        response = client.get("/version")
        assert response.status_code == 200

    def test_response_contains_expected_fields(self, client) -> None:
        body = client.get("/version").json()
        assert "api_version" in body
        assert "model_version" in body
        assert "environment" in body
        assert isinstance(body["api_version"], str)
        assert isinstance(body["model_version"], str)


# ── POST /api/v1/predict — success path ───────────────────────────────────────
class TestPredictEndpointSuccess:
    """A valid request flows through routing, validation, and the service."""

    def test_valid_request_returns_200(self, client) -> None:
        with patch(
            "app.api.routes.prediction_service.predict",
            return_value=_mock_prediction_response(),
        ) as mock_predict:
            response = client.post("/api/v1/predict", json=_valid_payload())

        assert response.status_code == 200
        mock_predict.assert_called_once()

    def test_response_matches_prediction_response_shape(self, client) -> None:
        with patch(
            "app.api.routes.prediction_service.predict",
            return_value=_mock_prediction_response(),
        ):
            body = client.post("/api/v1/predict", json=_valid_payload()).json()

        assert body["status"] == "success"
        assert body["prediction"]["predicted_sales"] == 4321.5
        assert body["prediction"]["target_date"] == "2015-06-15"
        assert body["forecast_details"]["horizon_bucket"] == "near"
        assert body["metadata"]["store_id"] == 1


# ── POST /api/v1/predict — request validation (422, handled by FastAPI/Pydantic)
class TestPredictEndpointValidation:
    """
    Malformed requests are rejected by Pydantic's Field constraints before
    ever reaching the prediction service. These return FastAPI's default
    422 Unprocessable Entity — NOT the custom RossmannAPIError handler,
    since store_id/horizon_days validity is enforced declaratively via
    Field(ge=..., le=...) in app/schemas/request.py, not raised as
    InvalidStoreIDError/InvalidHorizonError from within a route handler.
    """

    def test_store_id_zero_returns_422(self, client) -> None:
        with patch("app.api.routes.prediction_service.predict") as mock_predict:
            response = client.post(
                "/api/v1/predict", json=_valid_payload(store_id=0)
            )
        assert response.status_code == 422
        mock_predict.assert_not_called()

    def test_store_id_1116_returns_422(self, client) -> None:
        response = client.post(
            "/api/v1/predict", json=_valid_payload(store_id=1116)
        )
        assert response.status_code == 422

    def test_horizon_zero_returns_422(self, client) -> None:
        response = client.post(
            "/api/v1/predict", json=_valid_payload(horizon_days=0)
        )
        assert response.status_code == 422

    def test_horizon_91_returns_422(self, client) -> None:
        response = client.post(
            "/api/v1/predict", json=_valid_payload(horizon_days=91)
        )
        assert response.status_code == 422

    def test_invalid_state_holiday_returns_422(self, client) -> None:
        response = client.post(
            "/api/v1/predict", json=_valid_payload(state_holiday="z")
        )
        assert response.status_code == 422

    def test_missing_required_field_returns_422(self, client) -> None:
        payload = _valid_payload()
        del payload["store_id"]
        response = client.post("/api/v1/predict", json=payload)
        assert response.status_code == 422

    def test_invalid_calendar_date_returns_422(self, client) -> None:
        # April has 30 days — the 31st is not a valid calendar date.
        response = client.post(
            "/api/v1/predict",
            json=_valid_payload(month=4, day=31),
        )
        assert response.status_code == 422


# ── POST /api/v1/predict — RossmannAPIError → structured JSON (via the
# custom exception handler registered in app/main.py) ─────────────────────────
class TestPredictEndpointServiceErrors:
    """
    Errors raised from within the service layer are caught by the custom
    RossmannAPIError handler and converted into the structured
    {status, message, detail, code} JSON shape defined by ErrorDetail —
    this is the behavior that only an HTTP-level integration test can verify.
    """

    def test_model_not_loaded_returns_503_with_structured_body(self, client) -> None:
        with patch(
            "app.api.routes.prediction_service.predict",
            side_effect=ModelNotLoadedError("near"),
        ):
            response = client.post("/api/v1/predict", json=_valid_payload())

        assert response.status_code == 503
        body = response.json()
        assert body["status"] == "error"
        assert body["code"] == 503
        assert "message" in body
        assert "detail" in body

    def test_prediction_error_returns_500_with_structured_body(self, client) -> None:
        with patch(
            "app.api.routes.prediction_service.predict",
            side_effect=PredictionError("XGBoost inference failed unexpectedly."),
        ):
            response = client.post("/api/v1/predict", json=_valid_payload())

        assert response.status_code == 500
        body = response.json()
        assert body["status"] == "error"
        assert body["code"] == 500
