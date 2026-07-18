"""
app/model_loader.py
─────────────────────────────────────────────────────────────────────────────
Loads all model artifacts exactly once at application startup and exposes a
global ``artifacts`` singleton. Every prediction request reuses these objects
without reloading from disk.

Expected artifact layout (configure paths in app/config.py):

    artifacts/
    ├── models/
    │   ├── near_model.pkl
    │   ├── mid_model.pkl
    │   ├── far_model.pkl
    │   └── extended_model.pkl
    ├── feature_columns.json   — list of feature column names (training order)
    ├── store_metadata.json    — {store_id: {StoreType, Assortment, …}}
    └── lag_defaults.json      — {store_id: {lag_7: …, roll_7_mean: …, …}}

Save these from your training pipeline (script 16) before starting the API.
"""

import json
import pickle
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

from app.config import (
    MODELS_DIR,
    MODEL_FILES,
    FEATURE_COLUMNS_FILE,
    STORE_METADATA_FILE,
    LAG_DEFAULTS_FILE,
    BUCKET_ORDER,
)
from app.exceptions import ArtifactLoadError, ModelNotLoadedError
from app.utils.logger import get_logger

logger = get_logger(__name__)


# ── Artifact Container ────────────────────────────────────────────────────────
@dataclass
class ModelArtifacts:
    """
    Holds all loaded artifacts. Instantiated once and stored as a module-level
    singleton so that every request handler shares the same objects.
    """
    models:          Dict[str, Any]       = field(default_factory=dict)
    feature_columns: List[str]            = field(default_factory=list)
    store_metadata:  Dict[str, Dict]      = field(default_factory=dict)
    lag_defaults:    Dict[str, Dict]      = field(default_factory=dict)
    loaded:          bool                 = False

    def get_model(self, bucket: str):
        """Return the XGBoost model for the given horizon bucket."""
        model = self.models.get(bucket)
        if model is None:
            raise ModelNotLoadedError(bucket)
        return model

    def get_store_meta(self, store_id: int) -> Dict:
        """Return store metadata dict, falling back to sensible defaults."""
        key = str(store_id)
        if key not in self.store_metadata:
            logger.warning(
                "Store %s not found in metadata — using defaults.", store_id
            )
            return {"StoreType": "a", "Assortment": "a",
                    "CompetitionDistance": 1000, "Promo2": 0,
                    "CompetitionOpenSinceYear": 2010,
                    "CompetitionOpenSinceMonth": 1}
        return self.store_metadata[key]

    def get_lag_defaults(self, store_id: int) -> Dict:
        """Return lag feature defaults for the store."""
        key = str(store_id)
        return self.lag_defaults.get(key, {})

    @property
    def loaded_buckets(self) -> List[str]:
        return [b for b in BUCKET_ORDER if b in self.models]


# Module-level singleton — populated by load_artifacts()
artifacts = ModelArtifacts()


# ── Loader ────────────────────────────────────────────────────────────────────
def _load_pickle(path: Path, label: str) -> Any:
    """Load a pickle file; raise ArtifactLoadError on failure."""
    if not path.exists():
        raise ArtifactLoadError(label, f"File not found: {path}")
    try:
        with open(path, "rb") as f:
            return pickle.load(f)
    except Exception as exc:
        raise ArtifactLoadError(label, str(exc)) from exc


def _load_json(path: Path, label: str) -> Any:
    """Load a JSON file; raise ArtifactLoadError on failure."""
    if not path.exists():
        raise ArtifactLoadError(label, f"File not found: {path}")
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as exc:
        raise ArtifactLoadError(label, str(exc)) from exc


def load_artifacts() -> ModelArtifacts:
    """
    Load all model artifacts into the global ``artifacts`` singleton.
    Called once from the FastAPI lifespan context manager in ``main.py``.

    Returns:
        The populated :class:`ModelArtifacts` instance.

    Raises:
        :class:`~app.exceptions.ArtifactLoadError` if any critical artifact
        fails to load.
    """
    global artifacts

    logger.info("Loading model artifacts …")

    # ── Horizon-specific XGBoost models ──────────────────────────────────────
    loaded_models = {}
    for bucket, filename in MODEL_FILES.items():
        path = MODELS_DIR / filename
        try:
            model = _load_pickle(path, f"{bucket} model")
            loaded_models[bucket] = model
            logger.info("  ✓ %s model loaded  (%s)", bucket.upper(), path.name)
        except ArtifactLoadError as exc:
            # Non-fatal per bucket — warn and continue; health check will report
            logger.warning("  ✗ %s model not loaded: %s", bucket, exc.detail)

    if not loaded_models:
        raise ArtifactLoadError(
            "all horizon models",
            "No XGBoost models could be loaded. "
            "Run script 16 (save_production_model.py) first.",
        )

    # ── Feature columns ───────────────────────────────────────────────────────
    feat_cols = _load_json(FEATURE_COLUMNS_FILE, "feature_columns.json")
    logger.info("  ✓ Feature columns loaded  (%d columns)", len(feat_cols))

    # ── Store metadata ────────────────────────────────────────────────────────
    store_meta = _load_json(STORE_METADATA_FILE, "store_metadata.json")
    logger.info("  ✓ Store metadata loaded  (%d stores)", len(store_meta))

    # ── Lag defaults ──────────────────────────────────────────────────────────
    lag_defs: Dict = {}
    try:
        lag_defs = _load_json(LAG_DEFAULTS_FILE, "lag_defaults.json")
        logger.info("  ✓ Lag defaults loaded  (%d stores)", len(lag_defs))
    except ArtifactLoadError:
        logger.warning(
            "  ✗ lag_defaults.json not found — lag features will be -9999."
        )

    # ── Populate singleton ────────────────────────────────────────────────────
    artifacts.models          = loaded_models
    artifacts.feature_columns = feat_cols
    artifacts.store_metadata  = store_meta
    artifacts.lag_defaults    = lag_defs
    artifacts.loaded          = True

    logger.info(
        "Artifact loading complete. Buckets available: %s",
        artifacts.loaded_buckets,
    )
    return artifacts
