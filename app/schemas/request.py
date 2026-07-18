"""
app/schemas/request.py
─────────────────────────────────────────────────────────────────────────────
Pydantic request model for the POST /api/v1/predict endpoint.
Field-level validation catches malformed input before it reaches the
feature engineering pipeline.
"""

from typing import Optional
from pydantic import BaseModel, Field, field_validator, model_validator


class PredictionRequest(BaseModel):
    """
    Prediction request payload.

    All required fields map directly to features used during training.
    Lag feature fields are optional; when omitted the API uses
    precomputed per-store training defaults stored in ``lag_defaults.json``.
    """

    # ── Store identification ──────────────────────────────────────────────────
    store_id: int = Field(
        ...,
        ge=1,
        le=1115,
        description="Rossmann store ID (1–1115).",
        examples=[652],
    )

    # ── Forecast horizon ──────────────────────────────────────────────────────
    horizon_days: int = Field(
        ...,
        ge=1,
        le=90,
        description=(
            "Number of days ahead to forecast. "
            "Determines the horizon bucket and model: "
            "1–14=near, 15–30=mid, 31–60=far, 61–90=extended."
        ),
        examples=[7],
    )

    # ── Target date ───────────────────────────────────────────────────────────
    year: int = Field(
        ...,
        ge=2013,
        le=2030,
        description="Year of the forecast target date.",
        examples=[2015],
    )
    month: int = Field(
        ...,
        ge=1,
        le=12,
        description="Month of the forecast target date (1–12).",
        examples=[11],
    )
    day: int = Field(
        ...,
        ge=1,
        le=31,
        description="Day of the forecast target date (1–31).",
        examples=[14],
    )
    day_of_week: int = Field(
        ...,
        ge=1,
        le=7,
        description="Day of the week for the target date (1=Mon, 7=Sun).",
        examples=[6],
    )

    # ── Known future features ─────────────────────────────────────────────────
    promo: int = Field(
        default=0,
        ge=0,
        le=1,
        description="Whether a promotion is active on the target date (0 or 1).",
        examples=[1],
    )
    state_holiday: str = Field(
        default="0",
        description=(
            "State holiday type on the target date. "
            "'0'=none, 'a'=public holiday, 'b'=Easter, 'c'=Christmas."
        ),
        examples=["0"],
    )
    school_holiday: int = Field(
        default=0,
        ge=0,
        le=1,
        description="Whether schools are on holiday on the target date (0 or 1).",
        examples=[0],
    )

    # ── Optional lag feature overrides ───────────────────────────────────────
    # If provided, these override the stored defaults for this request.
    # The variable names match training column names exactly.
    lag_1:        Optional[float] = Field(default=None, description="Sales 1 day before origin.")
    lag_7:        Optional[float] = Field(default=None, description="Sales 7 days before origin.")
    lag_14:       Optional[float] = Field(default=None, description="Sales 14 days before origin.")
    lag_21:       Optional[float] = Field(default=None, description="Sales 21 days before origin.")
    lag_28:       Optional[float] = Field(default=None, description="Sales 28 days before origin.")
    lag_56:       Optional[float] = Field(default=None, description="Sales 56 days before origin.")
    lag_91:       Optional[float] = Field(default=None, description="Sales 91 days before origin.")
    lag_182:      Optional[float] = Field(default=None, description="Sales 182 days before origin.")
    lag_364:      Optional[float] = Field(default=None, description="Sales 364 days before origin.")
    roll_7_mean:  Optional[float] = Field(default=None, description="7-day rolling mean before origin.")
    roll_14_mean: Optional[float] = Field(default=None, description="14-day rolling mean before origin.")
    roll_28_mean: Optional[float] = Field(default=None, description="28-day rolling mean before origin.")
    roll_7_std:   Optional[float] = Field(default=None, description="7-day rolling std before origin.")
    roll_14_std:  Optional[float] = Field(default=None, description="14-day rolling std before origin.")
    roll_28_std:  Optional[float] = Field(default=None, description="28-day rolling std before origin.")

    # ── Validators ────────────────────────────────────────────────────────────
    @field_validator("state_holiday")
    @classmethod
    def validate_state_holiday(cls, v: str) -> str:
        valid = {"0", "a", "b", "c"}
        if str(v) not in valid:
            raise ValueError(
                f"state_holiday must be one of {sorted(valid)}. Got: '{v}'."
            )
        return str(v)

    @model_validator(mode="after")
    def validate_date(self) -> "PredictionRequest":
        """Check that year/month/day form a valid calendar date."""
        import datetime
        try:
            datetime.date(self.year, self.month, self.day)
        except ValueError as exc:
            raise ValueError(
                f"Invalid date: {self.year}-{self.month:02d}-{self.day:02d}. "
                f"Reason: {exc}"
            ) from exc
        return self

    def to_feature_dict(self) -> dict:
        """
        Export the request as a plain dict for the preprocessing pipeline.
        Lag overrides are included only when explicitly provided.
        """
        base = {
            "store_id":      self.store_id,
            "year":          self.year,
            "month":         self.month,
            "day":           self.day,
            "day_of_week":   self.day_of_week,
            "promo":         self.promo,
            "state_holiday": self.state_holiday,
            "school_holiday": self.school_holiday,
        }
        # Include lag overrides where explicitly set
        lag_fields = [
            "lag_1", "lag_7", "lag_14", "lag_21", "lag_28",
            "lag_56", "lag_91", "lag_182", "lag_364",
            "roll_7_mean", "roll_14_mean", "roll_28_mean",
            "roll_7_std", "roll_14_std", "roll_28_std",
        ]
        for fld in lag_fields:
            val = getattr(self, fld)
            if val is not None:
                base[fld] = val
        return base

    model_config = {"json_schema_extra": {
        "examples": [{
            "store_id":      652,
            "horizon_days":  7,
            "year":          2015,
            "month":         11,
            "day":           14,
            "day_of_week":   6,
            "promo":         1,
            "state_holiday": "0",
            "school_holiday": 0,
        }]
    }}
