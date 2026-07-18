"""
app/services/prediction_service.py
─────────────────────────────────────────────────────────────────────────────
Core prediction service. Coordinates:
  1. Request validation
  2. Horizon-to-bucket mapping
  3. Feature vector construction
  4. XGBoost inference
  5. Response assembly (new nested structure)

Prediction logic and model loading are unchanged.
Only the response assembly reflects the redesigned schema.
"""

import datetime
from app.config import MODEL_VERSION
from app.exceptions import PredictionError
from app.model_loader import artifacts
from app.schemas.request import PredictionRequest
from app.schemas.response import (
    PredictionResponse,
    PredictionSection,
    ForecastDetails,
    MetadataSection,
)
from app.utils.preprocessing import (
    build_feature_vector,
    get_horizon_bucket,
    validate_store_id,
)
from app.utils.logger import get_logger

logger = get_logger(__name__)


class PredictionService:
    """
    Stateless service for generating sales predictions.

    All state (models, feature columns, metadata) lives in the ``artifacts``
    singleton. The service can be instantiated freely without overhead.
    """

    def predict(self, request: PredictionRequest) -> PredictionResponse:
        """
        Generate a sales forecast for a single (store, date) pair.

        Args:
            request: Validated :class:`~app.schemas.request.PredictionRequest`.

        Returns:
            :class:`~app.schemas.response.PredictionResponse` with nested
            prediction, forecast_details, and metadata sections.

        Raises:
            :class:`~app.exceptions.InvalidStoreIDError`
            :class:`~app.exceptions.InvalidHorizonError`
            :class:`~app.exceptions.PreprocessingError`
            :class:`~app.exceptions.PredictionError`
        """
        logger.info(
            "Prediction request — store_id=%s  horizon=%s days  target=%s-%02d-%02d",
            request.store_id, request.horizon_days,
            request.year, request.month, request.day,
        )

        # ── 1. Validate store ID ──────────────────────────────────────────────
        validate_store_id(request.store_id)

        # ── 2. Determine horizon bucket and select model ──────────────────────
        bucket = get_horizon_bucket(request.horizon_days)
        model  = artifacts.get_model(bucket)

        logger.info(
            "Routing to %s model (days %s → bucket '%s')",
            bucket.upper(), request.horizon_days, bucket,
        )

        # ── 3. Retrieve store metadata and lag defaults ───────────────────────
        store_meta   = artifacts.get_store_meta(request.store_id)
        lag_defaults = artifacts.get_lag_defaults(request.store_id)

        # ── 4. Build feature vector ───────────────────────────────────────────
        feature_vector = build_feature_vector(
            request_dict    = request.to_feature_dict(),
            store_meta      = store_meta,
            lag_defaults    = lag_defaults,
            feature_columns = artifacts.feature_columns,
        )

        # ── 5. Model inference ────────────────────────────────────────────────
        try:
            raw_prediction  = float(model.predict(feature_vector)[0])
            predicted_sales = max(0.0, round(raw_prediction, 2))
        except Exception as exc:
            logger.error(
                "Inference failed — store_id=%s  bucket=%s  error=%s",
                request.store_id, bucket, exc, exc_info=True,
            )
            raise PredictionError(str(exc)) from exc

        target_date = f"{request.year}-{request.month:02d}-{request.day:02d}"
        timestamp   = datetime.datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ")

        logger.info(
            "Prediction complete — store=%s  date=%s  bucket=%s  predicted_sales=%.2f",
            request.store_id, target_date, bucket, predicted_sales,
        )

        # ── 6. Assemble nested response ───────────────────────────────────────
        return PredictionResponse(
            prediction=PredictionSection(
                predicted_sales=predicted_sales,
                target_date=target_date,
            ),
            forecast_details=ForecastDetails(
                horizon_days=request.horizon_days,
                horizon_bucket=bucket,
            ),
            metadata=MetadataSection(
                store_id=request.store_id,
                model=f"xgboost_{bucket}",
                model_version=MODEL_VERSION,
                prediction_timestamp=timestamp,
            ),
        )


# Module-level singleton — instantiated once and reused per request
prediction_service = PredictionService()