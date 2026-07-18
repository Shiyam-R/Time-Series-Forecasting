"""
tests/test_prediction_service.py
─────────────────────────────────────────────────────────────────────────────
Unit tests for the core prediction service.

Run:
    pytest tests/ -v

All model artifacts are mocked so tests run without actual .pkl files.
"""

import numpy as np
import pytest
from unittest.mock import MagicMock, patch

from app.exceptions import (
    InvalidStoreIDError,
    InvalidHorizonError,
    InvalidCategoricalValueError,
    ModelNotLoadedError,
)
from app.schemas.request import PredictionRequest
from app.utils.preprocessing import get_horizon_bucket


# ── Helpers ───────────────────────────────────────────────────────────────────
def _make_request(**overrides) -> PredictionRequest:
    """Return a valid PredictionRequest with optional field overrides."""
    defaults = {
        "store_id":      1,
        "horizon_days":  7,
        "year":          2015,
        "month":         6,
        "day":           15,
        "day_of_week":   1,
        "promo":         0,
        "state_holiday": "0",
        "school_holiday": 0,
    }
    defaults.update(overrides)
    return PredictionRequest(**defaults)


def _mock_artifacts(bucket: str = "near", prediction_value: float = 5000.0):
    """
    Build a mock ModelArtifacts object that returns ``prediction_value``
    regardless of the input feature vector.
    """
    mock_model = MagicMock()
    mock_model.predict.return_value = np.array([prediction_value])

    mock_arts = MagicMock()
    mock_arts.get_model.return_value     = mock_model
    mock_arts.get_store_meta.return_value = {
        "StoreType": "a", "Assortment": "a",
        "CompetitionDistance": 1000, "Promo2": 0,
        "CompetitionOpenSinceYear": 2010,
        "CompetitionOpenSinceMonth": 1,
    }
    mock_arts.get_lag_defaults.return_value = {}
    mock_arts.feature_columns = ["month", "day", "year", "day_of_week",
                                  "is_promo", "is_weekend"]
    return mock_arts


# ── Horizon bucket routing ────────────────────────────────────────────────────
class TestHorizonBucketRouting:
    """get_horizon_bucket() maps horizon_days to the correct bucket."""

    @pytest.mark.parametrize("days,expected", [
        (1,  "near"),
        (14, "near"),
        (15, "mid"),
        (30, "mid"),
        (31, "far"),
        (60, "far"),
        (61, "extended"),
        (90, "extended"),
    ])
    def test_valid_horizons(self, days: int, expected: str) -> None:
        assert get_horizon_bucket(days) == expected

    def test_horizon_zero_raises(self) -> None:
        with pytest.raises(InvalidHorizonError):
            get_horizon_bucket(0)

    def test_horizon_91_raises(self) -> None:
        with pytest.raises(InvalidHorizonError):
            get_horizon_bucket(91)


# ── Valid prediction ──────────────────────────────────────────────────────────
class TestValidPrediction:
    """PredictionService.predict() returns a valid response for good input."""

    def test_predict_returns_positive_sales(self) -> None:
        from app.services.prediction_service import PredictionService

        mock_arts = _mock_artifacts(prediction_value=4321.5)
        with patch("app.services.prediction_service.artifacts", mock_arts):
            service  = PredictionService()
            request  = _make_request(store_id=1, horizon_days=7)
            response = service.predict(request)

        assert response.prediction.predicted_sales      == 4321.5
        assert response.forecast_details.horizon_bucket == "near"
        assert response.metadata.store_id               == 1
        assert response.status                          == "success"

    def test_predict_clips_negative_to_zero(self) -> None:
        from app.services.prediction_service import PredictionService

        mock_arts = _mock_artifacts(prediction_value=-100.0)
        with patch("app.services.prediction_service.artifacts", mock_arts):
            service  = PredictionService()
            response = service.predict(_make_request())

        assert response.prediction.predicted_sales == 0.0

    @pytest.mark.parametrize("horizon,bucket", [
        (7,  "near"),
        (20, "mid"),
        (45, "far"),
        (75, "extended"),
    ])
    def test_correct_bucket_selected(self, horizon: int, bucket: str) -> None:
        from app.services.prediction_service import PredictionService

        mock_arts = _mock_artifacts()
        with patch("app.services.prediction_service.artifacts", mock_arts):
            service  = PredictionService()
            response = service.predict(_make_request(horizon_days=horizon))

        assert response.forecast_details.horizon_bucket == bucket

    def test_target_date_formatted_correctly(self) -> None:
        from app.services.prediction_service import PredictionService

        mock_arts = _mock_artifacts()
        with patch("app.services.prediction_service.artifacts", mock_arts):
            service  = PredictionService()
            response = service.predict(
                _make_request(year=2015, month=11, day=7)
            )

        assert response.prediction.target_date == "2015-11-07"


# ── Invalid store ID ──────────────────────────────────────────────────────────
class TestInvalidStoreID:
    """Requests with out-of-range store_id are rejected before inference."""

    def test_store_id_zero_raises_validation_error(self) -> None:
        with pytest.raises(Exception):   # Pydantic raises ValidationError
            _make_request(store_id=0)

    def test_store_id_1116_raises_validation_error(self) -> None:
        with pytest.raises(Exception):
            _make_request(store_id=1116)

    def test_store_id_999_is_valid(self) -> None:
        req = _make_request(store_id=999)
        assert req.store_id == 999


# ── Invalid horizon ───────────────────────────────────────────────────────────
class TestInvalidHorizon:
    """Requests with out-of-range horizon_days are rejected."""

    def test_horizon_zero_validation_error(self) -> None:
        with pytest.raises(Exception):
            _make_request(horizon_days=0)

    def test_horizon_91_validation_error(self) -> None:
        with pytest.raises(Exception):
            _make_request(horizon_days=91)


# ── Invalid categorical values ────────────────────────────────────────────────
class TestInvalidCategoricals:
    """Invalid categorical values are caught at the Pydantic layer."""

    def test_invalid_state_holiday_raises(self) -> None:
        with pytest.raises(Exception):
            _make_request(state_holiday="z")

    def test_valid_state_holiday_a(self) -> None:
        req = _make_request(state_holiday="a")
        assert req.state_holiday == "a"

    def test_promo_out_of_range_raises(self) -> None:
        with pytest.raises(Exception):
            _make_request(promo=2)


# ── Missing required fields ───────────────────────────────────────────────────
class TestMissingFields:
    """Required fields that are absent cause Pydantic ValidationError."""

    def test_missing_store_id_raises(self) -> None:
        with pytest.raises(Exception):
            PredictionRequest(
                horizon_days=7, year=2015, month=6,
                day=15, day_of_week=1,
            )

    def test_missing_horizon_raises(self) -> None:
        with pytest.raises(Exception):
            PredictionRequest(
                store_id=1, year=2015, month=6,
                day=15, day_of_week=1,
            )

    def test_missing_date_fields_raises(self) -> None:
        with pytest.raises(Exception):
            PredictionRequest(store_id=1, horizon_days=7)


# ── Model not loaded ──────────────────────────────────────────────────────────
class TestModelNotLoaded:
    """ModelNotLoadedError is raised when a bucket model is missing."""

    def test_missing_model_raises(self) -> None:
        from app.services.prediction_service import PredictionService

        mock_arts = _mock_artifacts()
        mock_arts.get_model.side_effect = ModelNotLoadedError("near")

        with patch("app.services.prediction_service.artifacts", mock_arts):
            service = PredictionService()
            with pytest.raises(ModelNotLoadedError):
                service.predict(_make_request(horizon_days=7))


# ── Feature engineering ───────────────────────────────────────────────────────
class TestFeatureEngineering:
    """Core feature engineering functions produce expected outputs."""

    def test_calendar_features_adds_weekend(self) -> None:
        from app.utils.feature_engineering import add_calendar_features
        f = {"year": 2015, "month": 11, "day": 14, "day_of_week": 6}
        out = add_calendar_features(f)
        assert out["is_weekend"] == 1

    def test_calendar_features_weekday_not_weekend(self) -> None:
        from app.utils.feature_engineering import add_calendar_features
        f = {"year": 2015, "month": 6, "day": 15, "day_of_week": 1}
        out = add_calendar_features(f)
        assert out["is_weekend"] == 0

    def test_interaction_store_652_detected(self) -> None:
        from app.utils.feature_engineering import add_interaction_features
        f = {
            "store_id": 652, "month": 11, "day": 14, "year": 2015,
            "StoreType": "a", "is_promo": 0,
            "is_school_holiday": 0, "is_state_holiday": 0, "is_weekend": 1,
        }
        out = add_interaction_features(f)
        assert out["is_store_652"]        == 1
        assert out["store652_november"]   == 1
        assert out["november_weekend"]    == 1
        assert out["store652_november_weekend"] == 1

    def test_interaction_type_a_november(self) -> None:
        from app.utils.feature_engineering import add_interaction_features
        f = {
            "store_id": 1, "month": 11, "day": 7, "year": 2015,
            "StoreType": "a", "is_promo": 0,
            "is_school_holiday": 0, "is_state_holiday": 0, "is_weekend": 0,
        }
        out = add_interaction_features(f)
        assert out["is_store_type_a"] == 1
        assert out["storeA_november"] == 1
        assert out["storeA_november_weekend"] == 0  # not a weekend