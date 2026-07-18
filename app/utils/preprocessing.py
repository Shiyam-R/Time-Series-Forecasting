"""
app/utils/preprocessing.py
─────────────────────────────────────────────────────────────────────────────
Orchestrates the complete preprocessing pipeline for a single prediction
request. This module is the single source of truth for how raw request
fields become a model-ready numpy array.

Pipeline order (must match training exactly):
  1. Calendar features
  2. Store metadata features
  3. Lag features (from artifacts or request overrides)
  4. Interaction features
  5. Column alignment (reindex to training feature column order)
  6. Null filling with -9999 (matches training)
"""

from typing import Dict, Any, List
import numpy as np

from app.config import BUCKET_RANGES, BUCKET_ORDER, MIN_HORIZON, MAX_HORIZON
from app.exceptions import (
    InvalidStoreIDError,
    InvalidHorizonError,
    InvalidCategoricalValueError,
    PreprocessingError,
)
from app.utils.feature_engineering import (
    add_calendar_features,
    add_store_features,
    add_interaction_features,
)
from app.utils.logger import get_logger

logger = get_logger(__name__)


def get_horizon_bucket(horizon_days: int) -> str:
    """
    Map an integer horizon to its named bucket.

    Args:
        horizon_days: Number of days ahead to forecast (1–90).

    Returns:
        One of ``'near'``, ``'mid'``, ``'far'``, ``'extended'``.

    Raises:
        :class:`~app.exceptions.InvalidHorizonError` if outside 1–90.
    """
    if not (MIN_HORIZON <= horizon_days <= MAX_HORIZON):
        raise InvalidHorizonError(horizon_days)

    for bucket, (lo, hi) in BUCKET_RANGES.items():
        if lo <= horizon_days <= hi:
            return bucket

    # Should never reach here given the range check above
    raise InvalidHorizonError(horizon_days)


def validate_store_id(store_id: int) -> None:
    """
    Raise :class:`~app.exceptions.InvalidStoreIDError` if store_id is
    outside the Rossmann dataset range of 1–1115.
    """
    if not (1 <= store_id <= 1115):
        raise InvalidStoreIDError(store_id)


def validate_categorical(field: str, value: str, valid: set) -> None:
    """
    Raise :class:`~app.exceptions.InvalidCategoricalValueError` if ``value``
    is not in ``valid``.
    """
    if str(value) not in valid:
        raise InvalidCategoricalValueError(field, str(value), valid)


def build_feature_vector(
    request_dict: Dict[str, Any],
    store_meta:   Dict[str, Any],
    lag_defaults: Dict[str, float],
    feature_columns: List[str],
) -> np.ndarray:
    """
    Transform a raw request dict into a 2-D numpy array ready for XGBoost.

    Args:
        request_dict:    Validated request fields as a plain dict.
        store_meta:      Store metadata for the requested store_id.
        lag_defaults:    Default lag values for the store, keyed by column name.
        feature_columns: Ordered list of feature column names used at training.

    Returns:
        ``np.ndarray`` of shape ``(1, len(feature_columns))``.

    Raises:
        :class:`~app.exceptions.PreprocessingError` on any failure.
    """
    try:
        features: Dict[str, Any] = dict(request_dict)

        # Step 1: calendar features
        features = add_calendar_features(features)

        # Step 2: store metadata features
        features = add_store_features(features, store_meta)

        # Step 3: lag features — request overrides take priority,
        #         then stored defaults, then -9999 (training fill value)
        for col, default_val in lag_defaults.items():
            if col not in features or features[col] is None:
                features[col] = default_val

        # Step 4: interaction features
        features = add_interaction_features(features)

        # Step 5: align to training column order
        vector = [features.get(col, -9999) for col in feature_columns]
        vector = [v if v is not None else -9999 for v in vector]

        arr = np.array(vector, dtype=np.float32).reshape(1, -1)

    except (InvalidStoreIDError, InvalidHorizonError,
            InvalidCategoricalValueError):
        raise  # re-raise validation errors as-is

    except Exception as exc:
        logger.error("Preprocessing failed: %s", exc, exc_info=True)
        raise PreprocessingError(str(exc)) from exc

    logger.debug(
        "Feature vector built — store_id=%s  shape=%s",
        request_dict.get("store_id"),
        arr.shape,
    )
    return arr
