"""
16_save_production_model.py
─────────────────────────────────────────────────────────────────────────────
Trains the final production XGBoost model on the full available dataset and
saves all artifacts required by the FastAPI application.

ARCHITECTURE NOTE (read before modifying):
    This trains ONE global XGBoost model — not four separate per-bucket
    models. This mirrors script 15 (Feature_Engineering_Experiment.py)
    exactly, whose interaction features (store652_weekend, storeA_november,
    is_herbstferien, etc.) are validated in that script by training a single
    model and evaluating it *sliced* by horizon bucket after the fact — the
    bucket never changes what the model is trained on.

    This is a deliberate choice, not a shortcut: an earlier version of this
    script trained inside a `for bucket in BUCKET_RANGES` loop, but every
    iteration used identical X/y and identical random_state, so all four
    "different" models came out byte-for-byte identical anyway — the loop
    gave the illusion of per-bucket differentiation without ever actually
    producing it. Removing the loop makes the codebase honest about what's
    actually being trained.

    A GENUINELY different per-bucket model exists in
    notebook/Time_Series_Fair_Horizon_Evaluation.py (script 12) — it uses a
    different data construction entirely (anchor/target date pairs, horizon
    as an input feature, log-transformed targets, a fitted StandardScaler).
    That approach was intentionally kept experimental and out of production
    (saved as artifacts/xgb_fair_eval_*.pkl). If that architecture is ever
    promoted to production instead, app/utils/preprocessing.py and
    app/utils/feature_engineering.py both need a corresponding rewrite to
    add anchor/horizon/log-target/scaler support — they currently mirror
    script 15's methodology, not script 12's.

Run ONCE before starting the API:
    python 16_save_production_model.py

Artifacts saved to artifacts/:
    models/global_model.pkl      — The single production XGBoost model,
                                    used for all four horizon buckets
                                    (near/mid/far/extended route to the
                                    same model — see app/config.py
                                    MODEL_FILES).
    feature_columns.json         — Ordered feature column list
    store_metadata.json          — Per-store metadata dict
    lag_defaults.json            — Per-store lag feature defaults
"""

import json
import pickle
import subprocess
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from xgboost import XGBRegressor

# ── Paths ─────────────────────────────────────────────────────────────────────
TRAIN_CSV    = Path("data/raw/train.csv")
STORE_CSV    = Path("data/raw/store.csv")
ARTIFACTS    = Path("artifacts")
MODELS_DIR   = ARTIFACTS / "models"
MODELS_DIR.mkdir(parents=True, exist_ok=True)

# ── Training window ────────────────────────────────────────────────────────────
TRAIN_END    = pd.Timestamp("2015-07-31")   # last date used for training
BUCKET_RANGES = {   # kept only for documentation / evaluation slicing elsewhere
    "near":     (1,  14),
    "mid":      (15, 30),
    "far":      (31, 60),
    "extended": (61, 90),
}
LAG_DAYS     = [1, 7, 14, 21, 28, 56, 91, 182, 364]
ROLL_WINDOWS = [7, 14, 28]

# ── XGBoost params — keep in sync with script 15 ─────────────────────────────
XGBOOST_PARAMS = {
    "n_estimators":     500,
    "max_depth":        6,
    "learning_rate":    0.05,
    "subsample":        0.8,
    "colsample_bytree": 0.8,
    "min_child_weight": 20,
    "reg_alpha":        0.1,
    "reg_lambda":       1.0,
    "random_state":     42,
    "n_jobs":           -1,
    "tree_method":      "hist",
    "objective":        "reg:squarederror",
    "verbosity":        0,
}

STORETYPE_ENC  = {"a": 0, "b": 1, "c": 2, "d": 3}
ASSORTMENT_ENC = {"a": 0, "b": 1, "c": 2}
SPIKE_MONTHS   = {4, 10, 11, 12}
STORE652_MONTH_W = {11: 3, 12: 2, 2: 1}
STOREA_MONTH_W   = {11: 3, 12: 2, 10: 2, 2: 1, 4: 1}


# ── Feature engineering (mirrors script 15 / app/utils/feature_engineering.py)
def _days_to_unity(year, month, day):
    t = pd.Timestamp(year, month, day)
    return min(
        abs((t - pd.Timestamp(year - 1, 10, 3)).days),
        abs((t - pd.Timestamp(year,     10, 3)).days),
        abs((t - pd.Timestamp(year + 1, 10, 3)).days),
    )


def build_features(df: pd.DataFrame, store_df: pd.DataFrame) -> pd.DataFrame:
    df = df.merge(store_df, on="Store", how="left")

    # Calendar
    df["month"]          = df["Date"].dt.month
    df["day"]            = df["Date"].dt.day
    df["year"]           = df["Date"].dt.year
    df["week_of_year"]   = df["Date"].dt.isocalendar().week.astype(int)
    df["day_of_week"]    = df["DayOfWeek"]
    df["quarter"]        = df["Date"].dt.quarter
    df["is_month_start"] = (df["day"] <= 7).astype(int)
    df["is_month_end"]   = (df["day"] >= 24).astype(int)
    df["is_weekend"]     = (df["DayOfWeek"] >= 6).astype(int)

    # Store metadata
    df["storetype_enc"]       = df["StoreType"].str.lower().map(STORETYPE_ENC).fillna(-1)
    df["assortment_enc"]      = df["Assortment"].str.lower().map(ASSORTMENT_ENC).fillna(-1)
    df["competition_distance"]= df["CompetitionDistance"].fillna(df["CompetitionDistance"].median())
    df["promo2"]              = df["Promo2"].fillna(0).astype(int)
    df["comp_months_open"]    = np.maximum(0, (
        (df["year"]  - df["CompetitionOpenSinceYear"].fillna(df["year"])) * 12 +
        (df["month"] - df["CompetitionOpenSinceMonth"].fillna(df["month"]))
    ))

    # Flags
    df["is_state_holiday"]  = (df["StateHoliday"].astype(str).str.strip() != "0").astype(int)
    df["is_school_holiday"] = df["SchoolHoliday"].fillna(0).astype(int)
    df["is_promo"]          = df["Promo"].fillna(0).astype(int)
    df["is_open"]           = df["Open"].fillna(1).astype(int)
    df["stateholiday_type"] = df["StateHoliday"].astype(str).str.strip().map(
        {"0": 0, "a": 1, "b": 2, "c": 3}).fillna(0)

    # Lag features (computed using actual history — no leakage)
    df = df.sort_values(["Store", "Date"])
    grp = df.groupby("Store")["Sales"]
    for d in LAG_DAYS:
        df[f"lag_{d}"] = grp.shift(d)
    for w in ROLL_WINDOWS:
        shifted = grp.shift(1)
        df[f"roll_{w}_mean"] = shifted.transform(
            lambda x: x.rolling(w, min_periods=1).mean())
        df[f"roll_{w}_std"]  = shifted.transform(
            lambda x: x.rolling(w, min_periods=1).std().fillna(0))

    # Interaction features (mirrors add_new_interaction_features)
    df["is_store_type_a"]    = (df["StoreType"].str.lower() == "a").astype(int)
    df["is_store_652"]       = (df["Store"] == 652).astype(int)
    df["is_november"]        = (df["month"] == 11).astype(int)
    df["is_october"]         = (df["month"] == 10).astype(int)
    df["is_february"]        = (df["month"] == 2).astype(int)
    df["is_december"]        = (df["month"] == 12).astype(int)
    df["is_april"]           = (df["month"] == 4).astype(int)
    df["is_high_error_month"]= df["month"].isin(SPIKE_MONTHS).astype(int)

    is_a = df["is_store_type_a"]
    is_652 = df["is_store_652"]
    is_nov = df["is_november"]
    is_oct = df["is_october"]
    is_sch = df["is_school_holiday"]
    is_sta = df["is_state_holiday"]
    is_prm = df["is_promo"]
    is_wkd = df["is_weekend"]
    is_feb = df["is_february"]
    is_dec = df["is_december"]
    is_apr = df["is_april"]

    df["store652_weekend"]         = is_652 * is_wkd
    df["november_weekend"]         = is_nov * is_wkd
    df["storeA_november"]          = is_a * is_nov
    df["storeA_weekend"]           = is_a * is_wkd
    df["store652_november"]        = is_652 * is_nov
    df["storeA_october"]           = is_a * is_oct
    df["store652_school_hol"]      = is_652 * is_sch
    df["november_school_hol"]      = is_nov * is_sch
    df["storeA_school_hol"]        = is_a * is_sch
    df["december_school_hol"]      = is_dec * is_sch
    df["february_school_hol"]      = is_feb * is_sch
    df["april_school_hol"]         = is_apr * is_sch
    df["promo_weekend"]            = is_prm * is_wkd
    df["promo_november"]           = is_prm * is_nov
    df["storeA_promo"]             = is_a * is_prm
    df["store652_promo"]           = is_652 * is_prm
    df["october_weekend"]          = is_oct * is_wkd
    df["october_school_hol"]       = is_oct * is_sch
    df["november_state_hol"]       = is_nov * is_sta
    df["storeA_state_hol"]         = is_a * is_sta
    df["store652_november_weekend"]= is_652 * is_nov * is_wkd
    df["storeA_november_weekend"]  = is_a * is_nov * is_wkd
    df["storeA_november_school"]   = is_a * is_nov * is_sch
    df["storeA_october_weekend"]   = is_a * is_oct * is_wkd

    df["days_to_unity_day"]  = df.apply(
        lambda r: _days_to_unity(int(r["year"]), int(r["month"]), int(r["day"])),
        axis=1)
    df["is_unity_day_window"]= (df["days_to_unity_day"] <= 7).astype(int)
    df["is_unity_day_week"]  = (df["days_to_unity_day"] <= 3).astype(int)
    df["is_herbstferien"]    = (
        (df["month"] == 10) & df["day"].between(6, 25) & (is_sch == 1)
    ).astype(int)
    df["nov_or_unity_school"]= np.maximum(
        df["november_school_hol"], df["is_herbstferien"])
    df["store652_spike_month"]= is_652 * df["month"].map(STORE652_MONTH_W).fillna(0)
    df["storeA_spike_month"]  = is_a   * df["month"].map(STOREA_MONTH_W).fillna(0)

    return df


def get_feature_cols(df: pd.DataFrame) -> list:
    exclude = {
        "Date", "Sales", "Customers", "Open", "Store",
        "StoreType", "Assortment", "StateHoliday",
        "CompetitionOpenSinceMonth", "CompetitionOpenSinceYear",
        "Promo2SinceWeek", "Promo2SinceYear", "PromoInterval",
        # Raw Rossmann CSV columns duplicated by properly-engineered
        # lowercase equivalents below. These must be excluded, not just
        # left in: the live API (app/utils/feature_engineering.py) has no
        # raw CSV row to copy them from at inference time — it only ever
        # populates the engineered versions — so if these stay in the
        # training feature set, every live prediction silently sends
        # -9999 (missing sentinel) for all five, regardless of what the
        # caller actually requested. This was a real, universal bug:
        # "DayOfWeek" -> use "day_of_week" instead
        # "Promo"     -> use "is_promo" instead
        # "SchoolHoliday" -> use "is_school_holiday" instead
        # "CompetitionDistance" -> use "competition_distance" instead
        # "Promo2"    -> use "promo2" instead
        "DayOfWeek", "Promo", "SchoolHoliday",
        "CompetitionDistance", "Promo2",
    }
    return [c for c in df.columns if c not in exclude]


def _get_git_commit() -> str:
    """
    Best-effort short git commit hash for training provenance.
    Returns "unknown" if not in a git repo, git isn't installed, or the
    lookup fails for any reason — this must never break training.
    """
    try:
        result = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True, text=True, timeout=5, check=True,
        )
        return result.stdout.strip() or "unknown"
    except Exception:
        return "unknown"


# ── Main ──────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("=" * 60)
    print("Script 16 — Save Production Model Artifacts")
    print("=" * 60)

    # Load raw data
    print("\nLoading data …")
    train = pd.read_csv(TRAIN_CSV, parse_dates=["Date"],
                        dtype={"StateHoliday": str}, low_memory=False)
    train = train[(train["Open"] == 1) & (train["Sales"] > 0)].copy()
    store = pd.read_csv(STORE_CSV, low_memory=False)

    # Use all data up to TRAIN_END
    train = train[train["Date"] <= TRAIN_END].copy()
    print(f"  Training rows: {len(train):,}  |  "
          f"Date range: {train['Date'].min().date()} → {train['Date'].max().date()}")

    # Build features
    print("\nBuilding features …")
    df_feat = build_features(train, store)
    feat_cols = get_feature_cols(df_feat)
    df_feat = df_feat.dropna(subset=["Sales"])
    X = df_feat[feat_cols].fillna(-9999)
    y = df_feat["Sales"].values
    print(f"  Feature matrix: {X.shape}")

    # ── Train ONE global model ────────────────────────────────────────────────
    # See the module docstring for why this is a single model rather than a
    # per-bucket loop: this matches script 15's validated methodology, where
    # `horizon_bucket` is an evaluation-time slice, not a training-time split.
    print("\nTraining production model …")
    model = XGBRegressor(**XGBOOST_PARAMS)
    model.fit(X, y, verbose=False)

    model_path = MODELS_DIR / "global_model.pkl"
    with open(model_path, "wb") as f:
        pickle.dump(model, f)
    print(f"  ✓ Global model saved  →  {model_path}")
    print("    (used for all four horizon buckets — see app/config.py MODEL_FILES)")

    # ── Model metadata sidecar (training provenance) ──────────────────────────
    # No RMSPE is recorded here: this script trains on the full dataset with
    # no holdout split, so it has no honest accuracy figure to report.
    # RMSPE is measured separately in the fair multi-origin backtest
    # (script 12) — see that script's output for validated accuracy numbers.
    model_metadata = {
        "trained_at":          datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "git_commit":          _get_git_commit(),
        "training_data_start": str(train["Date"].min().date()),
        "training_data_end":   str(train["Date"].max().date()),
        "training_rows":       int(len(train)),
        "feature_count":       len(feat_cols),
        "xgboost_params":      XGBOOST_PARAMS,
        "methodology": (
            "Single global XGBoost model serving all four horizon buckets "
            "(near/mid/far/extended route to the same model — see "
            "app/config.py MODEL_FILES and this script's module docstring)."
        ),
    }
    metadata_path = MODELS_DIR / "model_metadata.json"
    with open(metadata_path, "w") as f:
        json.dump(model_metadata, f, indent=2)
    print(f"  ✓ Model metadata saved  →  {metadata_path}")

    # Save feature columns
    feat_cols_path = ARTIFACTS / "feature_columns.json"
    with open(feat_cols_path, "w") as f:
        json.dump(feat_cols, f, indent=2)
    print(f"\n  ✓ Feature columns saved ({len(feat_cols)} cols)  →  {feat_cols_path}")

    # Save store metadata
    meta_cols = [
        "Store", "StoreType", "Assortment", "CompetitionDistance",
        "CompetitionOpenSinceMonth", "CompetitionOpenSinceYear",
        "Promo2", "Promo2SinceWeek", "Promo2SinceYear", "PromoInterval",
    ]
    meta_cols = [c for c in meta_cols if c in store.columns]
    store_meta = (
        store[meta_cols]
        .set_index("Store")
        .where(pd.notnull(store.set_index("Store")[meta_cols.copy()[1:]]), other=None)
        .to_dict(orient="index")
    )
    store_meta = {str(k): v for k, v in store_meta.items()}
    meta_path = ARTIFACTS / "store_metadata.json"
    with open(meta_path, "w") as f:
        json.dump(store_meta, f, indent=2)
    print(f"  ✓ Store metadata saved ({len(store_meta)} stores)  →  {meta_path}")

    # Compute per-store lag defaults from the most recent available data
    print("\nComputing lag defaults from most recent data …")
    max_date = train["Date"].max()
    lag_defaults = {}
    for store_id, grp in train.groupby("Store"):
        grp = grp.sort_values("Date")
        defs = {}
        for d in [1, 7, 14, 21, 28, 56, 91, 182, 364]:
            lag_date = max_date - pd.Timedelta(days=d)
            row = grp[grp["Date"] == lag_date]
            defs[f"lag_{d}"] = float(row["Sales"].iloc[0]) if len(row) else float(
                grp["Sales"].mean())
        for w in [7, 14, 28]:
            tail = grp.tail(w)["Sales"]
            defs[f"roll_{w}_mean"] = float(tail.mean())
            defs[f"roll_{w}_std"]  = float(tail.std()) if len(tail) > 1 else 0.0
        lag_defaults[str(store_id)] = defs
    lag_path = ARTIFACTS / "lag_defaults.json"
    with open(lag_path, "w") as f:
        json.dump(lag_defaults, f, indent=2)
    print(f"  ✓ Lag defaults saved ({len(lag_defaults)} stores)  →  {lag_path}")

    print("\n" + "=" * 60)
    print("All artifacts saved.  Start the API with:")
    print("  uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload")
    print("=" * 60)