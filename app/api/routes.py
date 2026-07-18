"""
app/api/routes.py
─────────────────────────────────────────────────────────────────────────────
FastAPI router containing all API endpoints.

Endpoints:
  GET  /                    — Project info and status.
  GET  /health              — Health check; confirms all models are loaded.
  GET  /version             — Build/deploy version identity.
  POST /api/v1/predict      — Generate a sales forecast.
"""

from datetime import datetime
from typing import List

from fastapi import APIRouter

from app.config import (
    API_TITLE, API_VERSION, API_DESCRIPTION, BUCKET_ORDER, MODEL_FILES,
    MODEL_VERSION, ENVIRONMENT,
)
from app.model_loader import artifacts
from app.schemas.request import PredictionRequest
from app.schemas.response import (
    ProjectInfoResponse,
    HealthResponse,
    ModelStatus,
    VersionResponse,
    PredictionResponse,
    ErrorDetail,
)
from app.services.prediction_service import prediction_service
from app.utils.logger import get_logger

logger = get_logger(__name__)
router = APIRouter()


# ── GET / ─────────────────────────────────────────────────────────────────────
@router.get(
    "/",
    response_model=ProjectInfoResponse,
    summary="Project information",
    description="Returns project metadata and a list of available endpoints. "
                "Use this to confirm the API is running.",
    tags=["Info"],
)
def root() -> ProjectInfoResponse:
    return ProjectInfoResponse(
        name        = API_TITLE,
        version     = API_VERSION,
        description = API_DESCRIPTION,
        endpoints   = [
            "GET  /               — Project info",
            "GET  /health         — Health check",
            "GET  /version        — Build/deploy version",
            "POST /api/v1/predict — Generate a sales forecast",
        ],
    )


# ── GET /health ───────────────────────────────────────────────────────────────
@router.get(
    "/health",
    response_model=HealthResponse,
    summary="Application health check",
    description=(
        "Reports the load status of all four horizon-specific XGBoost models "
        "and associated artifacts. "
        "`status` is `'healthy'` when all models are loaded, `'degraded'` otherwise."
    ),
    tags=["Info"],
)
def health() -> HealthResponse:
    model_statuses: List[ModelStatus] = [
        ModelStatus(
            bucket = bucket,
            loaded = bucket in artifacts.models,
            file   = MODEL_FILES.get(bucket, "unknown"),
        )
        for bucket in BUCKET_ORDER
    ]
    status = "healthy" if all(ms.loaded for ms in model_statuses) else "degraded"
    logger.info("Health check — status=%s", status)
    return HealthResponse(
        status               = status,
        models_loaded        = model_statuses,
        feature_columns      = len(artifacts.feature_columns),
        store_count          = len(artifacts.store_metadata),
        lag_defaults_loaded  = bool(artifacts.lag_defaults),
        timestamp            = datetime.utcnow(),
    )


# ── GET /version ──────────────────────────────────────────────────────────────
@router.get(
    "/version",
    response_model=VersionResponse,
    summary="Build and deploy version",
    description=(
        "Reports API and model version identity, and the deployment "
        "environment. Distinct from /health, which reports runtime model "
        "load status rather than build identity — use this endpoint to "
        "confirm exactly what's deployed."
    ),
    tags=["Info"],
)
def version() -> VersionResponse:
    return VersionResponse(
        api_version   = API_VERSION,
        model_version = MODEL_VERSION,
        environment   = ENVIRONMENT,
        trained_at    = artifacts.model_metadata.get("trained_at"),
        git_commit    = artifacts.model_metadata.get("git_commit"),
    )


# ── POST /api/v1/predict ──────────────────────────────────────────────────────
@router.post(
    "/api/v1/predict",
    response_model=PredictionResponse,
    summary="Generate a sales forecast",
    description=(
        "Generates a daily sales forecast for a single Rossmann store on a specific "
        "target date.\n\n"
        "**Horizon routing** — `horizon_days` determines which XGBoost model is used:\n\n"
        "| `horizon_days` | Bucket | Model |\n"
        "|---|---|---|\n"
        "| 1 – 14 | `near` | xgboost_near |\n"
        "| 15 – 30 | `mid` | xgboost_mid |\n"
        "| 31 – 60 | `far` | xgboost_far |\n"
        "| 61 – 90 | `extended` | xgboost_extended |\n\n"
        "**Lag features** — If `lag_7`, `roll_7_mean`, etc. are not supplied, "
        "the API uses precomputed per-store defaults from training. "
        "Supplying recent actual sales as lag overrides will improve accuracy.\n\n"
        "**Units** — `predicted_sales` is in EUR (Rossmann dataset currency). "
        "No currency conversion is applied."
    ),
    tags=["Prediction"],
    responses={
        200: {
            "description": "Forecast generated successfully.",
            "content": {
                "application/json": {
                    "example": {
                        "status": "success",
                        "prediction": {
                            "predicted_sales": 7072.26,
                            "target_date": "2015-11-14",
                        },
                        "forecast_details": {
                            "horizon_days": 90,
                            "horizon_bucket": "extended",
                        },
                        "metadata": {
                            "store_id": 652,
                            "model": "xgboost_extended",
                            "model_version": "v1.0.0",
                            "prediction_timestamp": "2026-07-05T06:47:56Z",
                        },
                    }
                }
            },
        },
        400: {"model": ErrorDetail, "description": "Invalid store ID, horizon, or categorical value."},
        422: {"model": ErrorDetail, "description": "Missing or malformed request fields."},
        500: {"model": ErrorDetail, "description": "Internal preprocessing or inference error."},
        503: {"model": ErrorDetail, "description": "Model artifact not loaded."},
    },
)
def predict(request: PredictionRequest) -> PredictionResponse:
    logger.info(
        "POST /api/v1/predict — store_id=%s  horizon=%s",
        request.store_id, request.horizon_days,
    )
    return prediction_service.predict(request)