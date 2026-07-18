"""
app/config.py
─────────────────────────────────────────────────────────────────────────────
Centralised configuration for the Rossmann Forecasting API.
All artifact paths, model settings, and feature constants live here.
Update ARTIFACTS_DIR when deploying to a different environment.

Environment variables (all optional — sensible defaults preserve prior
behavior exactly if none are set):
    ENVIRONMENT            "development" | "staging" | "production"
                            (default: "development")
    LOG_LEVEL               Python logging level name, e.g. "INFO", "DEBUG"
                            (default: "INFO")
    CORS_ALLOWED_ORIGINS    Comma-separated list of allowed origins, or "*"
                            (default: "*")

A local .env file (gitignored) is auto-loaded if present via python-dotenv.
This is a no-op when no .env file exists, so Docker/production behavior is
unaffected unless you explicitly create one. See .env.example for the
documented list of variables.
"""

import os
from pathlib import Path
from typing import Dict, List, Tuple

from dotenv import load_dotenv

load_dotenv()  # no-op if no .env file is present


# ── Project Paths ─────────────────────────────────────────────────────────────
BASE_DIR      = Path(__file__).resolve().parent.parent
ARTIFACTS_DIR = BASE_DIR / "artifacts"
MODELS_DIR    = ARTIFACTS_DIR / "models"
LOGS_DIR      = BASE_DIR / "logs"
MODEL_VERSION = "v1.0.0"
LOGS_DIR.mkdir(parents=True, exist_ok=True)


# ── Environment / Runtime Configuration ───────────────────────────────────────
ENVIRONMENT: str = os.environ.get("ENVIRONMENT", "development")

LOG_LEVEL_NAME: str = os.environ.get("LOG_LEVEL", "INFO").upper()

_cors_raw = os.environ.get("CORS_ALLOWED_ORIGINS", "*")
CORS_ALLOWED_ORIGINS: List[str] = (
    ["*"] if _cors_raw.strip() == "*"
    else [origin.strip() for origin in _cors_raw.split(",") if origin.strip()]
)


# ── API Metadata ──────────────────────────────────────────────────────────────
API_TITLE       = "Rossmann Store Sales Forecasting API"
API_VERSION     = "1.0.0"
API_DESCRIPTION = (
    "Production-quality multi-horizon sales forecasting API built on XGBoost. "
    "Supports four forecast horizon buckets: Near (1–14 days), Mid (15–30), "
    "Far (31–60), and Extended (61–90). "
    "Feature engineering mirrors the fair multi-origin training pipeline exactly."
)


# ── Horizon Buckets ───────────────────────────────────────────────────────────
BUCKET_RANGES: Dict[str, Tuple[int, int]] = {
    "near":     (1,  14),
    "mid":      (15, 30),
    "far":      (31, 60),
    "extended": (61, 90),
}
BUCKET_ORDER = ["near", "mid", "far", "extended"]
MAX_HORIZON  = 90
MIN_HORIZON  = 1


# ── Artifact File Names ───────────────────────────────────────────────────────
MODEL_FILES: Dict[str, str] = {
    "near":     "near_model.pkl",
    "mid":      "mid_model.pkl",
    "far":      "far_model.pkl",
    "extended": "extended_model.pkl",
}
FEATURE_COLUMNS_FILE = ARTIFACTS_DIR / "feature_columns.json"
STORE_METADATA_FILE  = ARTIFACTS_DIR / "store_metadata.json"
LAG_DEFAULTS_FILE    = ARTIFACTS_DIR / "lag_defaults.json"


# ── Valid Categorical Values ──────────────────────────────────────────────────
VALID_STATE_HOLIDAYS = {"0", "a", "b", "c"}
VALID_STORE_TYPES    = {"a", "b", "c", "d"}
VALID_ASSORTMENTS    = {"a", "b", "c"}
VALID_DAY_OF_WEEK    = set(range(1, 8))       # 1=Mon … 7=Sun
STORE_ID_RANGE       = (1, 1115)


# ── Lag Feature Columns ───────────────────────────────────────────────────────
LAG_DAYS    = [1, 7, 14, 21, 28, 56, 91, 182, 364]
ROLL_WINDOWS = [7, 14, 28]
LAG_COLS = (
    [f"lag_{d}" for d in LAG_DAYS]
    + [f"roll_{w}_mean" for w in ROLL_WINDOWS]
    + [f"roll_{w}_std"  for w in ROLL_WINDOWS]
)


# ── Store Type Encoding ───────────────────────────────────────────────────────
STORETYPE_ENCODING  = {"a": 0, "b": 1, "c": 2, "d": 3}
ASSORTMENT_ENCODING = {"a": 0, "b": 1, "c": 2}
STATEHOLIDAY_ENCODING = {"0": 0, "a": 1, "b": 2, "c": 3}


# ── High-Error Spike Months (from root cause analysis) ───────────────────────
SPIKE_MONTHS = {4, 10, 11, 12}   # Apr, Oct, Nov, Dec
STORE_652_SPIKE_MONTH_WEIGHTS = {11: 3, 12: 2, 2: 1}
STORE_A_SPIKE_MONTH_WEIGHTS   = {11: 3, 12: 2, 10: 2, 2: 1, 4: 1}