"""
app/utils/feature_engineering.py
─────────────────────────────────────────────────────────────────────────────
Pure feature engineering functions that mirror the training pipeline
in script 15 exactly. Any change here MUST be reflected in training, and
any change to the training pipeline MUST be reflected here.

All functions accept a plain dict and return a dict with additional keys,
keeping them stateless and easy to test.
"""

from typing import Dict, Any
import pandas as pd
import math

from app.config import (
    STORETYPE_ENCODING,
    ASSORTMENT_ENCODING,
    STATEHOLIDAY_ENCODING,
    SPIKE_MONTHS,
    STORE_652_SPIKE_MONTH_WEIGHTS,
    STORE_A_SPIKE_MONTH_WEIGHTS,
)


def add_calendar_features(features: Dict[str, Any]) -> Dict[str, Any]:
    """
    Add calendar-derived features from year / month / day / day_of_week.
    Mirrors ``add_calendar_store_features`` in script 15.
    """
    month = features["month"]
    day   = features["day"]
    year  = features["year"]

    features["week_of_year"]  = pd.Timestamp(year, month, day).isocalendar().week
    features["quarter"]       = (month - 1) // 3 + 1
    features["is_month_start"]= int(day <= 7)
    features["is_month_end"]  = int(day >= 24)
    features["is_weekend"]    = int(features["day_of_week"] >= 6)

    return features

def _safe_int(val, default: int) -> int:
    """Return int(val), falling back to default if val is None or NaN."""
    try:
        if val is None or (isinstance(val, float) and math.isnan(val)):
            return default
        return int(val)
    except (ValueError, TypeError):
        return default

def add_store_features(
    features: Dict[str, Any],
    store_meta: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Attach store-level metadata and derive encoded/compound features.
    ``store_meta`` is a single store's metadata dict from artifacts.
    """
    store_type   = str(store_meta.get("StoreType", "a")).lower()
    assortment   = str(store_meta.get("Assortment", "a")).lower()
    comp_dist    = float(store_meta.get("CompetitionDistance", 0) or 0)
    promo2       = int(store_meta.get("Promo2", 0) or 0)
    comp_open_yr = _safe_int(store_meta.get("CompetitionOpenSinceYear"), features["year"])
    comp_open_mo = _safe_int(store_meta.get("CompetitionOpenSinceMonth"), features["month"])

    features["StoreType"]             = store_type
    features["Assortment"]            = assortment
    features["storetype_enc"]         = STORETYPE_ENCODING.get(store_type, -1)
    features["assortment_enc"]        = ASSORTMENT_ENCODING.get(assortment, -1)
    features["competition_distance"]  = comp_dist
    features["promo2"]                = promo2
    features["comp_months_open"]      = max(
        0,
        (features["year"] - comp_open_yr) * 12
        + (features["month"] - comp_open_mo),
    )
    features["stateholiday_type"] = STATEHOLIDAY_ENCODING.get(
        str(features.get("state_holiday", "0")), 0
    )
    features["is_state_holiday"]  = int(str(features.get("state_holiday", "0")) != "0")
    features["is_school_holiday"] = int(features.get("school_holiday", 0))
    features["is_promo"]          = int(features.get("promo", 0))
    features["is_open"]           = 1   # closed days are excluded at training time

    return features


def _days_to_unity_day(year: int, month: int, day: int) -> int:
    """
    Minimum absolute distance in days from any German Unity Day (Oct 3).
    Considers the nearest occurrence (previous, current, or next year).
    """
    target = pd.Timestamp(year, month, day)
    candidates = [
        abs((target - pd.Timestamp(year - 1, 10, 3)).days),
        abs((target - pd.Timestamp(year,     10, 3)).days),
        abs((target - pd.Timestamp(year + 1, 10, 3)).days),
    ]
    return int(min(candidates))


def add_interaction_features(features: Dict[str, Any]) -> Dict[str, Any]:
    """
    Add all targeted interaction features derived from root cause analysis.
    Mirrors ``add_new_interaction_features`` in script 15.

    Root cause findings driving each feature group:
      - Store Type A  → highest RMSPE and error contribution overall.
      - Store 652     → largest single-store contributor within Type A.
      - November      → highest RMSPE month, especially Type A on weekends.
      - October       → German Unity Day / Herbstferien composition artifact.
      - School hols   → secondary driver for Dec and Feb spikes.
    """
    store_id   = int(features["store_id"])
    month      = int(features["month"])
    day        = int(features["day"])
    year       = int(features["year"])
    store_type = str(features.get("StoreType", "a"))
    is_promo   = int(features.get("is_promo", 0))
    is_school  = int(features.get("is_school_holiday", 0))
    is_state   = int(features.get("is_state_holiday", 0))
    is_weekend = int(features.get("is_weekend", 0))

    # ── Core indicators ───────────────────────────────────────────────────────
    is_type_a   = int(store_type == "a")
    is_store652 = int(store_id == 652)
    is_nov      = int(month == 11)
    is_oct      = int(month == 10)
    is_feb      = int(month == 2)
    is_dec      = int(month == 12)
    is_apr      = int(month == 4)

    features["is_store_type_a"]     = is_type_a
    features["is_store_652"]        = is_store652
    features["is_november"]         = is_nov
    features["is_october"]          = is_oct
    features["is_february"]         = is_feb
    features["is_december"]         = is_dec
    features["is_april"]            = is_apr
    features["is_high_error_month"] = int(month in SPIKE_MONTHS)

    # ── Two-way interactions (explicitly engineered) ───────────────────────────
    features["store652_weekend"]    = is_store652 * is_weekend
    features["november_weekend"]    = is_nov      * is_weekend
    features["storeA_november"]     = is_type_a   * is_nov
    features["storeA_weekend"]      = is_type_a   * is_weekend
    features["store652_november"]   = is_store652 * is_nov
    features["storeA_october"]      = is_type_a   * is_oct

    # ── Two-way interactions (from analysis) ──────────────────────────────────
    features["store652_school_hol"] = is_store652 * is_school
    features["november_school_hol"] = is_nov      * is_school
    features["storeA_school_hol"]   = is_type_a   * is_school
    features["december_school_hol"] = is_dec      * is_school
    features["february_school_hol"] = is_feb      * is_school
    features["april_school_hol"]    = is_apr      * is_school
    features["promo_weekend"]       = is_promo     * is_weekend
    features["promo_november"]      = is_promo     * is_nov
    features["storeA_promo"]        = is_type_a   * is_promo
    features["store652_promo"]      = is_store652 * is_promo
    features["october_weekend"]     = is_oct       * is_weekend
    features["october_school_hol"]  = is_oct       * is_school
    features["november_state_hol"]  = is_nov       * is_state
    features["storeA_state_hol"]    = is_type_a   * is_state

    # ── Three-way interactions ────────────────────────────────────────────────
    features["store652_november_weekend"] = is_store652 * is_nov * is_weekend
    features["storeA_november_weekend"]   = is_type_a   * is_nov * is_weekend
    features["storeA_november_school"]    = is_type_a   * is_nov * is_school
    features["storeA_october_weekend"]    = is_type_a   * is_oct * is_weekend

    # ── Calendar proximity features ───────────────────────────────────────────
    days_unity = _days_to_unity_day(year, month, day)
    features["days_to_unity_day"]   = days_unity
    features["is_unity_day_window"] = int(days_unity <= 7)
    features["is_unity_day_week"]   = int(days_unity <= 3)

    features["is_herbstferien"] = int(
        month == 10 and 6 <= day <= 25 and is_school == 1
    )
    features["nov_or_unity_school"] = max(
        features["november_school_hol"],
        features["is_herbstferien"],
    )

    # ── Store severity scores ─────────────────────────────────────────────────
    features["store652_spike_month"] = (
        is_store652 * STORE_652_SPIKE_MONTH_WEIGHTS.get(month, 0)
    )
    features["storeA_spike_month"] = (
        is_type_a * STORE_A_SPIKE_MONTH_WEIGHTS.get(month, 0)
    )

    return features
