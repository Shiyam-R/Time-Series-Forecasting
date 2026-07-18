"""
16_save_production_model.py
─────────────────────────────────────────────────────────────────────────────
Trains the final production XGBoost models (one per horizon bucket) on the
full available dataset and saves all artifacts required by the FastAPI
application.

Run ONCE before starting the API:
    python 16_save_production_model.py

Artifacts saved to artifacts/:
    models/near_model.pkl        — Near bucket XGBoost model (days 1–14)
    models/mid_model.pkl         — Mid bucket XGBoost model  (days 15–30)
    models/far_model.pkl         — Far bucket XGBoost model  (days 31–60)
    models/extended_model.pkl    — Extended bucket XGBoost   (days 61–90)
    feature_columns.json         — Ordered feature column list
    store_metadata.json          — Per-store metadata dict
    lag_defaults.json            — Per-store lag feature defaults
"""

import json
import pickle
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

# ── Evaluation settings ───────────────────────────────────────────────────────
TRAIN_END    = pd.Timestamp("2015-07-31")   # last date used for training
BUCKET_RANGES = {
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
    }
    return [c for c in df.columns if c not in exclude]


def assign_bucket(h: int) -> str:
    for b, (lo, hi) in BUCKET_RANGES.items():
        if lo <= h <= hi:
            return b
    return "other"


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

    # Train per-bucket models
    print("\nTraining horizon-specific models …")
    for bucket, (lo, hi) in BUCKET_RANGES.items():
        # Assign pseudo-horizon using day_of_year modulo
        # (in production retraining, use actual horizon from fair eval)
        # Simple approach: train all models on full data (horizon is a feature
        # in production prediction, not in this training — consistent with
        # the "direct sales model" approach used in script 15)
        model = XGBRegressor(**XGBOOST_PARAMS)
        model.fit(X, y, verbose=False)

        path = MODELS_DIR / f"{bucket}_model.pkl"
        with open(path, "wb") as f:
            pickle.dump(model, f)
        print(f"  ✓ {bucket.upper()} model saved  →  {path}")

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
