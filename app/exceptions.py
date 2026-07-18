"""
app/exceptions.py
─────────────────────────────────────────────────────────────────────────────
Custom exception hierarchy for the Rossmann Forecasting API.
All exceptions carry an HTTP status code so that the exception handler
in routes.py can return consistent, structured error responses without
try/except boilerplate in every endpoint.
"""

from typing import Optional


class RossmannAPIError(Exception):
    """Base class for all Rossmann API errors."""

    def __init__(
        self,
        message: str,
        status_code: int = 500,
        detail: Optional[str] = None,
    ) -> None:
        super().__init__(message)
        self.message     = message
        self.status_code = status_code
        self.detail      = detail or message


# ── Validation Errors (400) ───────────────────────────────────────────────────
class InvalidStoreIDError(RossmannAPIError):
    """Raised when the requested Store ID is outside the valid range."""

    def __init__(self, store_id: int) -> None:
        super().__init__(
            message=f"Store ID {store_id} is not valid.",
            status_code=400,
            detail=(
                f"Store ID must be between 1 and 1115. "
                f"Received: {store_id}."
            ),
        )


class InvalidHorizonError(RossmannAPIError):
    """Raised when horizon_days is outside the supported range (1–90)."""

    def __init__(self, horizon_days: int) -> None:
        super().__init__(
            message=f"Forecast horizon {horizon_days} is not valid.",
            status_code=400,
            detail=(
                f"horizon_days must be between 1 and 90. "
                f"Received: {horizon_days}."
            ),
        )


class InvalidCategoricalValueError(RossmannAPIError):
    """Raised when a categorical field contains an unrecognised value."""

    def __init__(self, field: str, value: str, valid_values: set) -> None:
        super().__init__(
            message=f"Invalid value for field '{field}': '{value}'.",
            status_code=400,
            detail=(
                f"Field '{field}' received value '{value}'. "
                f"Valid values are: {sorted(valid_values)}."
            ),
        )


class MissingFieldError(RossmannAPIError):
    """Raised when a required request field is absent."""

    def __init__(self, field: str) -> None:
        super().__init__(
            message=f"Required field '{field}' is missing.",
            status_code=422,
            detail=f"The field '{field}' is required but was not provided.",
        )


# ── Artifact / Model Errors (503) ────────────────────────────────────────────
class ModelNotLoadedError(RossmannAPIError):
    """Raised when the requested horizon model has not been loaded."""

    def __init__(self, bucket: str) -> None:
        super().__init__(
            message=f"Model for bucket '{bucket}' is not loaded.",
            status_code=503,
            detail=(
                f"The '{bucket}' XGBoost model artifact was not found "
                f"or failed to load during startup."
            ),
        )


class ArtifactLoadError(RossmannAPIError):
    """Raised when any artifact fails to load at startup."""

    def __init__(self, artifact: str, reason: str) -> None:
        super().__init__(
            message=f"Failed to load artifact: {artifact}.",
            status_code=503,
            detail=f"Artifact '{artifact}' could not be loaded. Reason: {reason}.",
        )


# ── Inference Errors (500) ────────────────────────────────────────────────────
class PreprocessingError(RossmannAPIError):
    """Raised when feature engineering or preprocessing fails."""

    def __init__(self, reason: str) -> None:
        super().__init__(
            message="Preprocessing pipeline failed.",
            status_code=500,
            detail=f"Feature engineering error: {reason}.",
        )


class PredictionError(RossmannAPIError):
    """Raised when model inference fails."""

    def __init__(self, reason: str) -> None:
        super().__init__(
            message="Prediction failed.",
            status_code=500,
            detail=f"Model inference error: {reason}.",
        )
