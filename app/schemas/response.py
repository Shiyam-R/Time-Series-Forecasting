"""
app/schemas/response.py
─────────────────────────────────────────────────────────────────────────────
Pydantic response models for all API endpoints.

The prediction response is split into three logical sections:
  - prediction       : the sales forecast and the date it applies to.
  - forecast_details : horizon metadata that describes the forecast window.
  - metadata         : operational context — store, model, and timestamp.

Every field carries a description, unit/format note, and example value so
that the auto-generated OpenAPI docs (/docs) are self-explanatory.
"""

from datetime import datetime
from typing import List, Literal, Optional
from pydantic import BaseModel, Field


# ── Sub-models — prediction response ─────────────────────────────────────────
class PredictionSection(BaseModel):
    """The core forecast output."""

    predicted_sales: float = Field(
        ...,
        description=(
            "Predicted daily sales for the requested store on the target date. "
            "Unit: the currency used in the Rossmann dataset (EUR). "
            "This value is not converted to any other currency."
        ),
        examples=[7072.26],
        ge=0.0,
    )
    target_date: str = Field(
        ...,
        description=(
            "The calendar date for which sales are forecast. "
            "Format: ISO 8601 (YYYY-MM-DD)."
        ),
        examples=["2015-11-14"],
        pattern=r"^\d{4}-\d{2}-\d{2}$",
    )


class ForecastDetails(BaseModel):
    """Describes the forecast horizon and the model bucket selected."""

    horizon_days: int = Field(
        ...,
        description=(
            "Number of days ahead from the origin date that this forecast covers. "
            "Unit: days. Range: 1–90."
        ),
        examples=[90],
        ge=1,
        le=90,
    )
    horizon_bucket: Literal["near", "mid", "far", "extended"] = Field(
        ...,
        description=(
            "Forecast horizon bucket automatically determined from horizon_days. "
            "Allowed values: "
            "'near' (1–14 days), "
            "'mid' (15–30 days), "
            "'far' (31–60 days), "
            "'extended' (61–90 days). "
            "Each bucket is served by a dedicated XGBoost model."
        ),
        examples=["extended"],
    )


class MetadataSection(BaseModel):
    """Operational metadata about the store, model, and timing."""

    store_id: int = Field(
        ...,
        description="Rossmann store ID for which the forecast was generated. Range: 1–1115.",
        examples=[652],
        ge=1,
        le=1115,
    )
    model: str = Field(
        ...,
        description=(
            "Identifier of the XGBoost model used to generate this forecast. "
            "Format: xgboost_{bucket}."
        ),
        examples=["xgboost_extended"],
    )
    model_version: str = Field(
        ...,
        description="Version string of the deployed model artifact.",
        examples=["v1.0.0"],
    )
    prediction_timestamp: str = Field(
        ...,
        description=(
            "UTC timestamp at which this prediction was generated. "
            "Format: ISO 8601 UTC (YYYY-MM-DDTHH:MM:SSZ)."
        ),
        examples=["2026-07-05T06:47:56Z"],
    )


# ── Top-level prediction response ─────────────────────────────────────────────
class PredictionResponse(BaseModel):
    """
    Structured response for POST /api/v1/predict.

    Fields are organised into three sections by purpose:
    - **prediction** — the sales figure and the date it applies to.
    - **forecast_details** — horizon metadata (days ahead, bucket name).
    - **metadata** — store, model, version, and timestamp.
    """

    status: Literal["success"] = Field(
        default="success",
        description="Response status. Always 'success' for HTTP 200 responses.",
        examples=["success"],
    )
    prediction: PredictionSection = Field(
        ...,
        description="Core forecast output: predicted sales and target date.",
    )
    forecast_details: ForecastDetails = Field(
        ...,
        description="Horizon context: days ahead and the bucket used for model routing.",
    )
    metadata: MetadataSection = Field(
        ...,
        description="Operational context: store, model identifier, version, and timestamp.",
    )

    model_config = {
        "json_schema_extra": {
            "examples": [{
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
            }]
        }
    }


# ── Shared error response ─────────────────────────────────────────────────────
class ErrorDetail(BaseModel):
    """Returned for all handled errors (4xx / 5xx)."""

    status: Literal["error"] = Field(default="error")
    message: str = Field(
        ...,
        description="Short human-readable summary of the error.",
        examples=["Store ID 9999 is not valid."],
    )
    detail: str = Field(
        ...,
        description="Detailed explanation or corrective suggestion.",
        examples=["Store ID must be between 1 and 1115. Received: 9999."],
    )
    code: int = Field(
        ...,
        description="HTTP status code.",
        examples=[400],
    )


# ── GET / response ────────────────────────────────────────────────────────────
class ProjectInfoResponse(BaseModel):
    """Response for the root endpoint."""

    name:        str
    version:     str
    description: str
    endpoints:   List[str]
    status:      str = Field(default="running")


# ── GET /health response ──────────────────────────────────────────────────────
class ModelStatus(BaseModel):
    """Load status for a single horizon bucket model."""

    bucket:  str  = Field(..., description="Horizon bucket name.", examples=["near"])
    loaded:  bool = Field(..., description="True if the model artifact is loaded.")
    file:    str  = Field(..., description="Artifact filename.", examples=["near_model.pkl"])


class HealthResponse(BaseModel):
    """Response for GET /health."""

    status: str = Field(
        ...,
        description="'healthy' if all four models are loaded, 'degraded' otherwise.",
        examples=["healthy"],
    )
    models_loaded:       List[ModelStatus]
    feature_columns:     int  = Field(..., description="Number of feature columns loaded.")
    store_count:         int  = Field(..., description="Number of stores in metadata.")
    lag_defaults_loaded: bool = Field(..., description="True if lag_defaults.json is loaded.")
    timestamp:           datetime = Field(..., description="UTC time of the health check.")


# ── GET /version response ─────────────────────────────────────────────────────
class VersionResponse(BaseModel):
    """
    Response for GET /version.

    Distinct from /health: this reports build/deploy identity (what code and
    model version is running), not runtime load status. This is the
    convention deploy tooling and monitoring dashboards typically expect.
    """

    api_version:   str = Field(..., description="Semantic version of the API itself.", examples=["1.0.0"])
    model_version: str = Field(..., description="Version string of the deployed model artifacts.", examples=["v1.0.0"])
    environment:   str = Field(
        ...,
        description="Deployment environment: 'development', 'staging', or 'production'.",
        examples=["production"],
    )
    trained_at: Optional[str] = Field(
        default=None,
        description=(
            "UTC timestamp when the currently deployed model was trained. "
            "None if artifacts/models/model_metadata.json was not found "
            "(older artifact sets predating this field)."
        ),
        examples=["2026-07-18T14:32:00Z"],
    )
    git_commit: Optional[str] = Field(
        default=None,
        description=(
            "Short git commit hash the model was trained from, if known. "
            "'unknown' if training happened outside a git repository."
        ),
        examples=["a1b2c3d"],
    )