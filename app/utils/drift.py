"""
app/utils/drift.py
─────────────────────────────────────────────────────────────────────────────
Feature drift monitoring for GET /drift.

Computes Population Stability Index (PSI) per feature, comparing the
distribution of recent live prediction requests against the training data's
reference distribution (see notebook/Time_Series_Save_Production_Model.py's
build_reference_distributions(), saved as artifacts/feature_reference_stats.json).

PSI interpretation (standard industry thresholds — see app/config.py):
    < 0.10   : no significant distribution change
    0.10-0.25: moderate change, worth investigating
    >= 0.25  : significant change, feature distribution has genuinely shifted

IMPORTANT SCOPE CAVEAT — read before treating /drift as authoritative:
The rolling buffer below is IN-PROCESS MEMORY. Under a multi-worker
deployment (see Dockerfile's WORKERS env var), each worker process has its
own separate buffer. GET /drift reflects only the traffic that happened to
land on whichever worker handled that specific /drift request — not the
whole container's traffic, and the buffer resets on every restart. A
process-wide view would need shared storage (Redis, a small DB), which is
a deliberate scope decision for this project's current size, not an
oversight. See app/config.py's DRIFT_* constants for full context.

TWO ADDITIONAL CHARACTERISTICS TO KNOW ABOUT, found during verification
(both expected, not bugs):

1. SMALL-SAMPLE NOISE: PSI is inherently noisy with few samples relative
   to the number of bins per feature. DRIFT_MIN_SAMPLES gates the
   "insufficient_data" status, but even at the minimum, expect PSI to
   settle down further as the buffer fills toward DRIFT_WINDOW_SIZE.

2. LAG/ROLLING FEATURES SHOW AN ELEVATED PSI BASELINE UNDER DEFAULT USAGE:
   when a caller doesn't supply explicit lag_7/roll_7_mean/etc. overrides
   (see app/utils/preprocessing.py), every prediction for a given store
   reuses that store's single static historical average from
   lag_defaults.json. Repeatedly reusing one fixed per-store mean has
   inherently less spread than the raw day-by-day values the training
   reference distribution was built from — so lag_*/roll_* features will
   show a persistent PSI baseline under default (non-override) traffic,
   independent of any genuine drift. This is a structural property of the
   serving design, not something this module tries to correct for. Traffic
   that supplies real recent sales as lag overrides (as the API's own docs
   recommend for accuracy) will show more representative PSI readings for
   these specific features.

3. LOCUST LOAD-TEST TRAFFIC IS A POOR /drift BASELINE, BY DESIGN: verified
   directly — running load_test/locustfile.py's traffic through GET /drift
   showed "significant" drift dominated by `year` (PSI > 12). This is
   correct, not a bug: that script's date range (2013-2030) and randomized
   promo/state_holiday rates were deliberately chosen to stress-test input
   variety and validation robustness under concurrency, not to mimic
   realistic production traffic — the model was only trained on 2013 to
   mid-2015 data, so load-test requests with e.g. year=2028 are genuinely,
   correctly flagged as out-of-distribution. Use real application traffic,
   or a separate realistic-traffic sample, to get a meaningful drift
   baseline — not load-test output.
"""

import threading
from collections import deque
from typing import Any, Dict, List, Optional

import numpy as np

from app.config import (
    DRIFT_WINDOW_SIZE,
    DRIFT_MIN_SAMPLES,
    DRIFT_PSI_MODERATE,
    DRIFT_PSI_SIGNIFICANT,
    DRIFT_TOP_N_IN_RESPONSE,
)
from app.utils.logger import get_logger

logger = get_logger(__name__)


class RequestFeatureBuffer:
    """
    Thread-safe rolling buffer of recent live-request feature vectors.
    Fixed capacity (DRIFT_WINDOW_SIZE) — oldest entries drop automatically
    once full, via collections.deque's maxlen.
    """

    def __init__(self, maxlen: int = DRIFT_WINDOW_SIZE):
        self._buffer: deque = deque(maxlen=maxlen)
        self._lock = threading.Lock()

    def record(self, feature_columns: List[str], vector: List[float]) -> None:
        """Append one request's feature values, keyed by column name."""
        with self._lock:
            self._buffer.append(dict(zip(feature_columns, vector)))

    def snapshot(self) -> List[Dict[str, float]]:
        """A copy of the current buffer contents, safe to iterate outside the lock."""
        with self._lock:
            return list(self._buffer)

    def __len__(self) -> int:
        with self._lock:
            return len(self._buffer)


# Module-level singleton — one buffer per worker PROCESS (see scope caveat
# in the module docstring above).
request_buffer = RequestFeatureBuffer()


def record_request_features(feature_columns: List[str], vector: List[float]) -> None:
    """Called by prediction_service.py after every successful prediction."""
    request_buffer.record(feature_columns, vector)


def _compute_psi_categorical(reference: Dict[str, Any], live_values: List[float]) -> Optional[float]:
    """PSI for a categorical/low-cardinality feature via exact-value matching."""
    ref_values = reference["values"]
    ref_props  = reference["proportions"]

    live_arr = np.array(live_values, dtype=np.float64)
    live_counts = np.array([(live_arr == v).sum() for v in ref_values], dtype=np.float64)

    # Live values not present in the reference's category set (a genuinely
    # new category showing up) count as drift too — track them separately
    # rather than silently dropping them.
    matched = live_counts.sum()
    unmatched = len(live_arr) - matched
    if unmatched > 0:
        live_counts = np.append(live_counts, unmatched)
        ref_props = ref_props + [1e-6]  # near-zero reference presence for "unseen category"

    return _psi_from_proportions(np.array(ref_props), live_counts / live_counts.sum())


def _compute_psi_continuous(reference: Dict[str, Any], live_values: List[float]) -> Optional[float]:
    """PSI for a continuous feature via the training data's own quantile bins."""
    edges = np.array(reference["bin_edges"])
    ref_props = np.array(reference["bin_proportions"])

    live_arr = np.array(live_values, dtype=np.float64)
    # Clip to the reference range so out-of-range live values land in the
    # nearest edge bin rather than being silently dropped by np.histogram.
    clipped = np.clip(live_arr, edges[0], edges[-1])
    live_counts, _ = np.histogram(clipped, bins=edges)

    return _psi_from_proportions(ref_props, live_counts / live_counts.sum())


def _psi_from_proportions(ref_props: np.ndarray, live_props: np.ndarray) -> float:
    """
    Standard PSI formula: sum[(live% - ref%) * ln(live% / ref%)] per bin.
    A small epsilon guards against log(0) / division-by-zero for empty bins
    — a standard, widely-used PSI implementation detail, not an approximation
    that changes the metric's meaning.
    """
    eps = 1e-6
    ref_props  = np.clip(ref_props, eps, None)
    live_props = np.clip(live_props, eps, None)
    return float(np.sum((live_props - ref_props) * np.log(live_props / ref_props)))


def _drift_level(psi: float) -> str:
    if psi >= DRIFT_PSI_SIGNIFICANT:
        return "significant"
    if psi >= DRIFT_PSI_MODERATE:
        return "moderate"
    return "none"


def generate_drift_report(reference_stats: Dict[str, Any]) -> Dict[str, Any]:
    """
    Build the full GET /drift response payload.

    Returns a dict matching app.schemas.response.DriftResponse. If the
    buffer doesn't yet hold DRIFT_MIN_SAMPLES requests, PSI isn't computed
    (too noisy to be meaningful) and status is "insufficient_data".
    """
    snapshot = request_buffer.snapshot()
    sample_size = len(snapshot)

    if not reference_stats:
        return {
            "status": "unavailable",
            "sample_size": sample_size,
            "window_capacity": DRIFT_WINDOW_SIZE,
            "overall_drift_level": None,
            "features": [],
            "features_evaluated": 0,
        }

    if sample_size < DRIFT_MIN_SAMPLES:
        return {
            "status": "insufficient_data",
            "sample_size": sample_size,
            "window_capacity": DRIFT_WINDOW_SIZE,
            "overall_drift_level": None,
            "features": [],
            "features_evaluated": 0,
        }

    results = []
    for feature, ref in reference_stats.items():
        live_values = [row.get(feature) for row in snapshot if feature in row]
        if not live_values:
            continue

        try:
            if ref["type"] == "categorical":
                psi = _compute_psi_categorical(ref, live_values)
            else:
                psi = _compute_psi_continuous(ref, live_values)
        except Exception as exc:
            logger.warning("PSI computation failed for feature '%s': %s", feature, exc)
            continue

        if psi is not None:
            results.append({"feature": feature, "psi": round(psi, 4), "drift_level": _drift_level(psi)})

    results.sort(key=lambda r: r["psi"], reverse=True)

    overall = "none"
    if any(r["drift_level"] == "significant" for r in results):
        overall = "significant"
    elif any(r["drift_level"] == "moderate" for r in results):
        overall = "moderate"

    return {
        "status": "ok",
        "sample_size": sample_size,
        "window_capacity": DRIFT_WINDOW_SIZE,
        "overall_drift_level": overall,
        "features": results[:DRIFT_TOP_N_IN_RESPONSE],
        "features_evaluated": len(results),
    }