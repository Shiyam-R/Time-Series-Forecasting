"""
15_feature_engineering_experiment.py
─────────────────────────────────────────────────────────────────────────────
Feature engineering experiment based on root cause analysis findings.
Adds targeted interaction features for the high-error entities identified
in scripts 13–14, retrains XGBoost using the same fair multi-origin
evaluation pipeline, and compares against a retrained baseline.

IMPORTANT BEFORE RUNNING:
1. Verify XGBOOST_PARAMS matches script 12's model configuration.
2. Verify BASELINE_FEATURES list matches script 12's feature set.
3. SHAP analysis requires: pip install shap
4. Runtime: ~20–40 min depending on n_origins and hardware.

NOTE: Both baseline and augmented are retrained here in the same pipeline.
      Absolute RMSPE may differ slightly from script 12, but the relative
      improvement from new features is reliable.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from pathlib import Path
import warnings
import time
warnings.filterwarnings("ignore")

try:
    import xgboost as xgb
    from xgboost import XGBRegressor
except ImportError:
    raise ImportError("XGBoost required: pip install xgboost")

try:
    import shap
    SHAP_AVAILABLE = True
except ImportError:
    print("⚠️  SHAP not installed — skipping SHAP plots. Run: pip install shap")
    SHAP_AVAILABLE = False


# ── CONFIG ────────────────────────────────────────────────────────────────────
TRAIN_CSV         = Path("data/raw/train.csv")
STORE_CSV         = Path("data/raw/store.csv")
OUTPUT_DIR        = Path("data/processed")
FIGURES_DIR       = Path("figures/feature_engineering_experiment")
FIGURES_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# ── Fair evaluation settings — VERIFY against script 12 ───────────────────────
TRAIN_CUTOFF       = pd.Timestamp("2015-03-01")   # first origin date
EVAL_END           = pd.Timestamp("2015-07-31")   # last origin date
ORIGIN_STEP_DAYS   = 7                             # weekly rolling origins
MAX_HORIZON        = 90                            # days ahead to forecast

BUCKET_ORDER  = ["near", "mid", "far", "extended"]
BUCKET_RANGES = {"near": (1, 14), "mid": (15, 30), "far": (31, 60), "extended": (61, 90)}
MONTH_NAMES   = ["Jan","Feb","Mar","Apr","May","Jun",
                 "Jul","Aug","Sep","Oct","Nov","Dec"]

# ── XGBoost params — VERIFY against script 12 ─────────────────────────────────
XGBOOST_PARAMS = {
    "n_estimators":      500,
    "max_depth":         6,
    "learning_rate":     0.05,
    "subsample":         0.8,
    "colsample_bytree":  0.8,
    "min_child_weight":  20,
    "reg_alpha":         0.1,
    "reg_lambda":        1.0,
    "random_state":      42,
    "n_jobs":            -1,
    "tree_method":       "hist",
    "objective":         "reg:squarederror",
    "verbosity":         0,
}

SHAP_SAMPLE_N = 5_000   # rows to sample for SHAP (reduce if slow)

# Lag windows (days before origin) used as features
LAG_DAYS    = [1, 7, 14, 21, 28, 56, 91, 182, 364]
ROLL_WINDOWS = [7, 14, 28]
LAG_COLS    = (
    [f"lag_{d}" for d in LAG_DAYS] +
    [f"roll_{w}_mean" for w in ROLL_WINDOWS] +
    [f"roll_{w}_std" for w in ROLL_WINDOWS]
)


# ── METRICS ───────────────────────────────────────────────────────────────────
def rmspe(actual: np.ndarray, predicted: np.ndarray) -> float:
    mask = np.asarray(actual) > 0
    if mask.sum() == 0:
        return np.nan
    a, p = np.asarray(actual)[mask], np.asarray(predicted)[mask]
    return float(np.sqrt(np.mean(((a - p) / a) ** 2)) * 100)

def mae_fn(actual: np.ndarray, predicted: np.ndarray) -> float:
    mask = np.asarray(actual) > 0
    if mask.sum() == 0:
        return np.nan
    return float(np.mean(np.abs(np.asarray(actual)[mask] - np.asarray(predicted)[mask])))

def rmse_fn(actual: np.ndarray, predicted: np.ndarray) -> float:
    mask = np.asarray(actual) > 0
    if mask.sum() == 0:
        return np.nan
    a, p = np.asarray(actual)[mask], np.asarray(predicted)[mask]
    return float(np.sqrt(np.mean((a - p) ** 2)))


# ── DATA LOADING ──────────────────────────────────────────────────────────────
def load_raw_data():
    print("Loading raw data …")
    train = pd.read_csv(
        TRAIN_CSV,
        parse_dates=["Date"],
        dtype={"StateHoliday": str},
        low_memory=False
    )
    train = train[train["Open"] == 1].copy()   # closed days excluded
    train = train[train["Sales"] > 0].copy()   # zero-sales days excluded

    store = pd.read_csv(STORE_CSV, low_memory=False)

    # Lowercase columns for consistency
    train.columns = [c.strip() for c in train.columns]
    store.columns = [c.strip() for c in store.columns]

    print(f"  Train: {len(train):,} rows | {train['Store'].nunique()} stores | "
          f"{train['Date'].min().date()} → {train['Date'].max().date()}")
    return train, store


# ── FEATURE ENGINEERING ───────────────────────────────────────────────────────
def add_calendar_store_features(df: pd.DataFrame, store_df: pd.DataFrame) -> pd.DataFrame:
    """
    Standard calendar and store features.
    VERIFY: these should match script 12's baseline feature set.
    """
    df = df.merge(store_df, on="Store", how="left")

    df["month"]            = df["Date"].dt.month
    df["day"]              = df["Date"].dt.day
    df["year"]             = df["Date"].dt.year
    df["week_of_year"]     = df["Date"].dt.isocalendar().week.astype(int)
    df["day_of_week"]      = df["DayOfWeek"]
    df["is_month_start"]   = (df["day"] <= 7).astype(int)
    df["is_month_end"]     = (df["day"] >= 24).astype(int)
    df["quarter"]          = df["Date"].dt.quarter

    # Store metadata
    df["storetype_enc"]    = df["StoreType"].map({"a":0,"b":1,"c":2,"d":3}).fillna(-1)
    df["assortment_enc"]   = df["Assortment"].map({"a":0,"b":1,"c":2}).fillna(-1)
    df["competition_distance"] = df["CompetitionDistance"].fillna(
        df["CompetitionDistance"].median()
    )
    df["promo2"]           = df["Promo2"].fillna(0).astype(int)
    df["comp_months_open"] = np.maximum(0, (
        (df["year"]  - df["CompetitionOpenSinceYear"].fillna(df["year"])) * 12 +
        (df["month"] - df["CompetitionOpenSinceMonth"].fillna(df["month"]))
    ))

    # Binary event flags
    df["is_state_holiday"]  = (df["StateHoliday"].astype(str).str.strip() != "0").astype(int)
    df["is_school_holiday"] = df["SchoolHoliday"].fillna(0).astype(int)
    df["is_promo"]          = df["Promo"].fillna(0).astype(int)
    df["is_open"]           = df["Open"].fillna(1).astype(int)
    df["is_weekend"]        = (df["DayOfWeek"] >= 6).astype(int)

    # StateHoliday type encoding (a=public, b=easter, c=christmas)
    df["stateholiday_type"] = df["StateHoliday"].astype(str).str.strip().map(
        {"0":0, "a":1, "b":2, "c":3}).fillna(0)

    return df


def compute_train_lag_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute lag and rolling features for training rows.
    Each row uses lags computed from that row's own history (no leakage).
    """
    df = df.sort_values(["Store", "Date"]).copy()
    grp = df.groupby("Store")["Sales"]

    for d in LAG_DAYS:
        df[f"lag_{d}"] = grp.shift(d)

    for w in ROLL_WINDOWS:
        shifted = grp.shift(1)
        df[f"roll_{w}_mean"] = shifted.transform(
            lambda x: x.rolling(w, min_periods=1).mean())
        df[f"roll_{w}_std"]  = shifted.transform(
            lambda x: x.rolling(w, min_periods=1).std().fillna(0))

    return df


def compute_origin_lags(train_subset: pd.DataFrame, origin: pd.Timestamp) -> dict:
    """
    Compute lag features for each store anchored to origin date.
    Used at test time so predictions don't use future sales data.
    Returns dict: {store_id: {lag_col: value}}
    """
    history = train_subset[train_subset["Date"] < origin].copy()
    origin_lags = {}

    for store_id, grp in history.groupby("Store"):
        grp = grp.sort_values("Date")
        store_lags = {}

        for d in LAG_DAYS:
            lag_date = origin - pd.Timedelta(days=d)
            row = grp[grp["Date"] == lag_date]
            store_lags[f"lag_{d}"] = (
                float(row["Sales"].iloc[0]) if len(row) > 0 else np.nan)

        recent_sales = grp.tail(max(ROLL_WINDOWS))["Sales"].values
        for w in ROLL_WINDOWS:
            tail = recent_sales[-w:] if len(recent_sales) >= w else recent_sales
            store_lags[f"roll_{w}_mean"] = float(np.nanmean(tail)) if len(tail) else np.nan
            store_lags[f"roll_{w}_std"]  = float(np.nanstd(tail))  if len(tail) else np.nan

        origin_lags[store_id] = store_lags

    return origin_lags


def attach_origin_lags(test_df: pd.DataFrame, origin_lags: dict) -> pd.DataFrame:
    """Attach precomputed origin-anchored lag features to test rows."""
    for lag_col in LAG_COLS:
        test_df[lag_col] = test_df["Store"].map(
            {s: lags.get(lag_col, np.nan) for s, lags in origin_lags.items()})
    return test_df


def add_new_interaction_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    ─────────────────────────────────────────────────────────────────────────
    NEW INTERACTION FEATURES — Root Cause Analysis findings:
      - Store Type A: consistently highest RMSPE and error contribution.
      - Store 652: largest individual error contributor within Type A.
      - November: highest RMSPE month, especially for Store Type A.
      - Weekend: primary driver of November spike.
      - October: German Unity Day / Herbstferien composition artifact.
      - School holidays: secondary driver for December and February spikes.
    ─────────────────────────────────────────────────────────────────────────
    Call AFTER add_calendar_store_features().
    """
    df = df.copy()

    # ── Core indicator features ───────────────────────────────────────────────
    df["is_store_type_a"]     = (df["StoreType"] == "a").astype(int)
    df["is_store_652"]        = (df["Store"] == 652).astype(int)
    df["is_november"]         = (df["month"] == 11).astype(int)
    df["is_october"]          = (df["month"] == 10).astype(int)
    df["is_february"]         = (df["month"] == 2).astype(int)
    df["is_december"]         = (df["month"] == 12).astype(int)
    df["is_april"]            = (df["month"] == 4).astype(int)
    # df["is_weekend"]          = (df["DayOfWeek"] >= 6).astype(int)
    df["is_high_error_month"] = df["month"].isin([4, 10, 11, 12]).astype(int)

    # ── Two-way interactions (explicitly requested) ───────────────────────────
    df["store652_weekend"]    = df["is_store_652"]     * df["is_weekend"]
    df["november_weekend"]    = df["is_november"]      * df["is_weekend"]
    df["storeA_november"]     = df["is_store_type_a"]  * df["is_november"]
    df["storeA_weekend"]      = df["is_store_type_a"]  * df["is_weekend"]
    df["store652_november"]   = df["is_store_652"]     * df["is_november"]
    df["storeA_october"]      = df["is_store_type_a"]  * df["is_october"]

    # ── Two-way interactions (additional from analysis) ───────────────────────
    df["store652_school_hol"] = df["is_store_652"]     * df["is_school_holiday"]
    df["november_school_hol"] = df["is_november"]      * df["is_school_holiday"]
    df["storeA_school_hol"]   = df["is_store_type_a"]  * df["is_school_holiday"]
    df["december_school_hol"] = df["is_december"]      * df["is_school_holiday"]
    df["february_school_hol"] = df["is_february"]      * df["is_school_holiday"]
    df["april_school_hol"]    = df["is_april"]         * df["is_school_holiday"]
    df["promo_weekend"]       = df["is_promo"]         * df["is_weekend"]
    df["promo_november"]      = df["is_promo"]         * df["is_november"]
    df["storeA_promo"]        = df["is_store_type_a"]  * df["is_promo"]
    df["store652_promo"]      = df["is_store_652"]     * df["is_promo"]
    df["october_weekend"]     = df["is_october"]       * df["is_weekend"]
    df["october_school_hol"]  = df["is_october"]       * df["is_school_holiday"]
    df["november_state_hol"]  = df["is_november"]      * df["is_state_holiday"]
    df["storeA_state_hol"]    = df["is_store_type_a"]  * df["is_state_holiday"]

    # ── Three-way interactions ────────────────────────────────────────────────
    df["store652_november_weekend"] = (
        df["is_store_652"] * df["is_november"] * df["is_weekend"])
    df["storeA_november_weekend"]   = (
        df["is_store_type_a"] * df["is_november"] * df["is_weekend"])
    df["storeA_november_school"]    = (
        df["is_store_type_a"] * df["is_november"] * df["is_school_holiday"])
    df["storeA_october_weekend"]    = (
        df["is_store_type_a"] * df["is_october"] * df["is_weekend"])

    # ── Calendar proximity features ───────────────────────────────────────────
    # Distance to German Unity Day (Oct 3) — key October driver
    def days_to_unity(date):
        candidates = [
            abs((date - pd.Timestamp(date.year - 1, 10, 3)).days),
            abs((date - pd.Timestamp(date.year,     10, 3)).days),
            abs((date - pd.Timestamp(date.year + 1, 10, 3)).days),
        ]
        return min(candidates)

    df["days_to_unity_day"]   = df["Date"].apply(days_to_unity)
    df["is_unity_day_window"] = (df["days_to_unity_day"] <= 7).astype(int)
    df["is_unity_day_week"]   = (df["days_to_unity_day"] <= 3).astype(int)

    # Herbstferien approximation (October school holidays)
    df["is_herbstferien"] = (
        (df["month"] == 10) &
        (df["day"].between(6, 25)) &
        (df["is_school_holiday"] == 1)
    ).astype(int)

    # November school holiday × Unity Day window
    df["nov_or_unity_school"] = np.maximum(
        df["november_school_hol"], df["is_herbstferien"])

    # ── Store 652 month severity (from Analysis 10) ───────────────────────────
    store652_month_weights = {11: 3, 12: 2, 2: 1}
    df["store652_spike_month"] = (
        df["is_store_652"] *
        df["month"].map(store652_month_weights).fillna(0)
    )

    # ── Type A monthly severity (from Analysis 8) ─────────────────────────────
    storeA_month_weights = {11: 3, 12: 2, 10: 2, 2: 1, 4: 1}
    df["storeA_spike_month"] = (
        df["is_store_type_a"] *
        df["month"].map(storeA_month_weights).fillna(0)
    )

    return df


def get_feature_columns(df: pd.DataFrame, use_new_features: bool) -> list:
    """Return the list of feature columns to use for this experiment."""
    # Columns that are never features
    exclude = {
        "Date", "Sales", "Customers", "Open",
        "Store", "StoreType", "Assortment",
        "StateHoliday", "CompetitionOpenSinceMonth", "CompetitionOpenSinceYear",
        "Promo2SinceWeek", "Promo2SinceYear", "PromoInterval",
        "horizon", "horizon_bucket", "origin",
    }

    # New feature column names (only included in augmented run)
    new_feature_cols = {
        "is_store_type_a", "is_store_652", "is_november", "is_october",
        "is_february", "is_december", "is_april", "is_weekend",
        "is_high_error_month",
        "store652_weekend", "november_weekend", "storeA_november",
        "storeA_weekend", "store652_november", "storeA_october",
        "store652_school_hol", "november_school_hol", "storeA_school_hol",
        "december_school_hol", "february_school_hol", "april_school_hol",
        "promo_weekend", "promo_november", "storeA_promo", "store652_promo",
        "october_weekend", "october_school_hol", "november_state_hol",
        "storeA_state_hol",
        "store652_november_weekend", "storeA_november_weekend",
        "storeA_november_school", "storeA_october_weekend",
        "days_to_unity_day", "is_unity_day_window", "is_unity_day_week",
        "is_herbstferien", "nov_or_unity_school",
        "store652_spike_month", "storeA_spike_month",
    }

    cols = [c for c in df.columns if c not in exclude]
    if not use_new_features:
        cols = [c for c in cols if c not in new_feature_cols]

    return cols


# ── FAIR EVALUATION PIPELINE ──────────────────────────────────────────────────
def get_origins() -> list:
    origins = []
    d = TRAIN_CUTOFF
    while d <= EVAL_END:
        origins.append(d)
        d += pd.Timedelta(days=ORIGIN_STEP_DAYS)
    return origins


def assign_bucket(horizon: int) -> str:
    for b, (lo, hi) in BUCKET_RANGES.items():
        if lo <= horizon <= hi:
            return b
    return "other"


def run_evaluation(train_df: pd.DataFrame,
                   store_df: pd.DataFrame,
                   use_new_features: bool,
                   label: str) -> pd.DataFrame:
    """
    Run the full fair multi-origin evaluation for one experiment.
    Returns a results DataFrame (one row per prediction).
    """
    origins  = get_origins()
    all_preds = []
    model    = XGBRegressor(**XGBOOST_PARAMS)

    print(f"\n{'─'*50}")
    print(f"Running: {label}  ({len(origins)} origins)")
    print(f"New features: {'YES' if use_new_features else 'NO'}")
    print(f"{'─'*50}")

    for i, origin in enumerate(origins, 1):
        t0 = time.time()

        # ── Partition ─────────────────────────────────────────────────────────
        train_mask = train_df["Date"] < origin
        test_end   = origin + pd.Timedelta(days=MAX_HORIZON)
        test_mask  = (train_df["Date"] >= origin) & (train_df["Date"] < test_end)

        train_sub  = train_df[train_mask].copy()
        test_sub   = train_df[test_mask].copy()

        if len(train_sub) < 1000 or len(test_sub) == 0:
            continue

        # ── Build train features ───────────────────────────────────────────────
        train_feat = add_calendar_store_features(train_sub, store_df)
        train_feat = compute_train_lag_features(train_feat)
        if use_new_features:
            train_feat = add_new_interaction_features(train_feat)

        # ── Compute origin-anchored lags for test ──────────────────────────────
        origin_lags = compute_origin_lags(train_sub, origin)

        # ── Build test features ────────────────────────────────────────────────
        test_feat = add_calendar_store_features(test_sub, store_df)
        test_feat = attach_origin_lags(test_feat, origin_lags)
        if use_new_features:
            test_feat = add_new_interaction_features(test_feat)

        # ── Get aligned feature columns ────────────────────────────────────────
        feat_cols = get_feature_columns(train_feat, use_new_features)
        feat_cols = [c for c in feat_cols if c in train_feat.columns
                     and c in test_feat.columns]

        X_train = train_feat[feat_cols].fillna(-9999)
        y_train = train_feat["Sales"].values
        X_test  = test_feat[feat_cols].fillna(-9999)
        y_test  = test_feat["Sales"].values

        # ── Train ──────────────────────────────────────────────────────────────
        model.fit(X_train, y_train, verbose=False)
        y_pred = model.predict(X_test).clip(min=0)

        # ── Collect results ────────────────────────────────────────────────────
        test_feat["y_actual"]       = y_test
        test_feat["y_pred"]         = y_pred
        test_feat["origin"]         = origin
        test_feat["horizon"]        = (test_feat["Date"] - origin).dt.days
        test_feat["horizon_bucket"] = test_feat["horizon"].apply(assign_bucket)
        test_feat["spe"]            = (
            (test_feat["y_actual"] - test_feat["y_pred"]) /
             test_feat["y_actual"].clip(lower=1)
        ) ** 2

        keep_cols = [
            "Store", "Date", "origin", "horizon", "horizon_bucket",
            "y_actual", "y_pred", "spe",
            "StoreType", "month", "is_weekend",
            "is_school_holiday", "is_state_holiday", "is_promo",
        ]
        keep_cols = [c for c in keep_cols if c in test_feat.columns]
        all_preds.append(test_feat[keep_cols].copy())

        elapsed = time.time() - t0
        r_all   = rmspe(test_feat["y_actual"].values, test_feat["y_pred"].values)
        print(f"  Origin {i:2d}/{len(origins)} ({origin.date()})  "
              f"RMSPE={r_all:.2f}%  n={len(test_sub):,}  "
              f"({elapsed:.1f}s)")

    results = pd.concat(all_preds, ignore_index=True)
    results["horizon_bucket"] = pd.Categorical(
        results["horizon_bucket"], categories=BUCKET_ORDER, ordered=True)

    print(f"\n  Total predictions: {len(results):,}")
    return results


# ── METRICS ───────────────────────────────────────────────────────────────────
def compute_metrics_by_bucket(results: pd.DataFrame, label: str) -> pd.DataFrame:
    """Compute RMSPE, MAE, RMSE, SPE contribution per bucket."""
    total_spe = results["spe"].sum()
    rows = []
    for b in BUCKET_ORDER:
        sub = results[
            (results["horizon_bucket"] == b) & (results["y_actual"] > 0)]
        if len(sub) < 10:
            continue
        rows.append({
            "experiment":   label,
            "bucket":       b,
            "rmspe":        rmspe(sub["y_actual"].values, sub["y_pred"].values),
            "mae":          mae_fn(sub["y_actual"].values, sub["y_pred"].values),
            "rmse":         rmse_fn(sub["y_actual"].values, sub["y_pred"].values),
            "spe_contrib":  sub["spe"].sum() / total_spe * 100,
            "n":            len(sub),
        })
    return pd.DataFrame(rows)


def compare_metrics(baseline_m: pd.DataFrame,
                    augmented_m: pd.DataFrame) -> pd.DataFrame:
    """Build side-by-side comparison with improvement stats."""
    rows = []
    for b in BUCKET_ORDER:
        bsl = baseline_m[baseline_m["bucket"] == b]
        aug = augmented_m[augmented_m["bucket"] == b]
        if bsl.empty or aug.empty:
            continue
        bsl, aug = bsl.iloc[0], aug.iloc[0]
        for metric in ["rmspe", "mae", "rmse", "spe_contrib"]:
            old, new = bsl[metric], aug[metric]
            rows.append({
                "bucket":        b,
                "metric":        metric,
                "baseline":      old,
                "augmented":     new,
                "abs_change":    new - old,
                "pct_change":    (new - old) / old * 100 if old != 0 else np.nan,
                "improved":      new < old,
            })
    return pd.DataFrame(rows)


# ── FEATURE IMPORTANCE ────────────────────────────────────────────────────────
def get_feature_importance(train_df: pd.DataFrame,
                            store_df: pd.DataFrame) -> pd.DataFrame:
    """
    Train a single augmented model on all pre-cutoff data for feature
    importance and SHAP — avoids per-origin overhead.
    """
    print("\nTraining final model on full pre-cutoff data for importance/SHAP …")
    all_train = train_df[train_df["Date"] < EVAL_END].copy()

    feat = add_calendar_store_features(all_train, store_df)
    feat = compute_train_lag_features(feat)
    feat = add_new_interaction_features(feat)

    feat_cols = get_feature_columns(feat, use_new_features=True)
    feat_cols = [c for c in feat_cols if c in feat.columns]

    X = feat[feat_cols].fillna(-9999)
    y = feat["Sales"].values

    model = XGBRegressor(**XGBOOST_PARAMS)
    model.fit(X, y, verbose=False)

    importance = pd.DataFrame({
        "feature":    feat_cols,
        "importance": model.feature_importances_,
    }).sort_values("importance", ascending=False).reset_index(drop=True)

    return model, X, feat_cols, importance


# ── VISUALIZATION ─────────────────────────────────────────────────────────────
def plot_rmspe_comparison(baseline_m: pd.DataFrame,
                          augmented_m: pd.DataFrame,
                          comp: pd.DataFrame):
    print("\nPlotting RMSPE comparison …")
    rmspe_comp = comp[comp["metric"] == "rmspe"]

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Left: side-by-side RMSPE bars
    ax  = axes[0]
    x   = np.arange(len(BUCKET_ORDER))
    w   = 0.38
    bsl = [baseline_m[baseline_m["bucket"] == b]["rmspe"].values for b in BUCKET_ORDER]
    aug = [augmented_m[augmented_m["bucket"] == b]["rmspe"].values for b in BUCKET_ORDER]
    bsl = [v[0] if len(v) else np.nan for v in bsl]
    aug = [v[0] if len(v) else np.nan for v in aug]

    b1 = ax.bar(x - w/2, bsl, w, label="Baseline",   color="#90CAF9", alpha=0.9)
    b2 = ax.bar(x + w/2, aug, w, label="Augmented",  color="#4CAF50", alpha=0.9)

    for bar, v in zip(list(b1) + list(b2), bsl + aug):
        if not np.isnan(v):
            ax.text(bar.get_x() + bar.get_width()/2, v + 0.2,
                    f"{v:.2f}%", ha="center", va="bottom", fontsize=8.5)

    # Annotate improvement
    for xi, (b, a) in enumerate(zip(bsl, aug)):
        if not (np.isnan(b) or np.isnan(a)):
            delta = a - b
            ax.annotate(f"{delta:+.2f}pp",
                        xy=(xi + w/2, a),
                        xytext=(0, 12), textcoords="offset points",
                        ha="center", fontsize=8, fontweight="bold",
                        color="#1B5E20" if delta < 0 else "#B71C1C")

    ax.set_xticks(x)
    ax.set_xticklabels([b.upper() for b in BUCKET_ORDER], fontsize=10)
    ax.set_ylabel("RMSPE (%)", fontsize=10)
    ax.yaxis.set_major_formatter(mticker.PercentFormatter())
    ax.set_title("RMSPE: Baseline vs Augmented Features",
                 fontweight="bold", fontsize=11)
    ax.legend(fontsize=9); ax.grid(axis="y", alpha=0.25)

    # Right: % improvement per bucket and metric
    ax = axes[1]
    metrics = ["rmspe", "mae", "rmse"]
    x2  = np.arange(len(BUCKET_ORDER))
    w2  = 0.25
    m_colors = {"rmspe": "#EF5350", "mae": "#FF9800", "rmse": "#9C27B0"}
    for i, metric in enumerate(metrics):
        sub = comp[comp["metric"] == metric].set_index("bucket")
        vals = [sub.loc[b, "pct_change"] if b in sub.index else np.nan
                for b in BUCKET_ORDER]
        ax.bar(x2 + (i-1)*w2, vals, w2,
               label=metric.upper(), color=m_colors[metric], alpha=0.85)

    ax.axhline(0, color="black", linewidth=0.8, linestyle="--")
    ax.set_xticks(x2)
    ax.set_xticklabels([b.upper() for b in BUCKET_ORDER], fontsize=10)
    ax.set_ylabel("% Change (negative = improvement)", fontsize=9)
    ax.set_title("Metric % Change by Bucket\n(below zero = improvement)",
                 fontweight="bold", fontsize=11)
    ax.legend(fontsize=9); ax.grid(axis="y", alpha=0.25)

    plt.suptitle("Feature Engineering Experiment — Baseline vs Augmented",
                 fontsize=13, fontweight="bold")
    plt.tight_layout()
    fig.savefig(FIGURES_DIR / "15a_rmspe_comparison.png",
                dpi=150, bbox_inches="tight")
    plt.close()


def plot_error_contribution(baseline_m: pd.DataFrame,
                            augmented_m: pd.DataFrame):
    print("Plotting error contribution comparison …")
    fig, ax = plt.subplots(figsize=(10, 5))
    x = np.arange(len(BUCKET_ORDER))
    w = 0.38
    bsl = [baseline_m[baseline_m["bucket"] == b]["spe_contrib"].values
           for b in BUCKET_ORDER]
    aug = [augmented_m[augmented_m["bucket"] == b]["spe_contrib"].values
           for b in BUCKET_ORDER]
    bsl = [v[0] if len(v) else np.nan for v in bsl]
    aug = [v[0] if len(v) else np.nan for v in aug]

    b1 = ax.bar(x - w/2, bsl, w, label="Baseline",  color="#90CAF9", alpha=0.9)
    b2 = ax.bar(x + w/2, aug, w, label="Augmented", color="#4CAF50", alpha=0.9)
    for bar, v in zip(list(b1) + list(b2), bsl + aug):
        if not np.isnan(v):
            ax.text(bar.get_x() + bar.get_width()/2, v + 0.2,
                    f"{v:.1f}%", ha="center", va="bottom", fontsize=9)

    ax.set_xticks(x)
    ax.set_xticklabels([b.upper() for b in BUCKET_ORDER], fontsize=10)
    ax.set_ylabel("% of Total Squared Percentage Error", fontsize=10)
    ax.yaxis.set_major_formatter(mticker.PercentFormatter())
    ax.set_title("Error Contribution by Bucket — Baseline vs Augmented",
                 fontweight="bold", fontsize=12)
    ax.legend(fontsize=10); ax.grid(axis="y", alpha=0.25)
    plt.tight_layout()
    fig.savefig(FIGURES_DIR / "15b_error_contribution.png",
                dpi=150, bbox_inches="tight")
    plt.close()


def plot_feature_importance(importance: pd.DataFrame, top_n: int = 30):
    print("Plotting feature importance …")
    top       = importance.head(top_n)
    new_cols  = {
        "store652_weekend", "november_weekend", "storeA_november",
        "storeA_weekend", "store652_november", "storeA_october",
        "store652_school_hol", "november_school_hol", "storeA_school_hol",
        "december_school_hol", "february_school_hol", "april_school_hol",
        "promo_weekend", "promo_november", "storeA_promo", "store652_promo",
        "october_weekend", "october_school_hol", "november_state_hol",
        "storeA_state_hol",
        "store652_november_weekend", "storeA_november_weekend",
        "storeA_november_school", "storeA_october_weekend",
        "days_to_unity_day", "is_unity_day_window", "is_unity_day_week",
        "is_herbstferien", "nov_or_unity_school",
        "store652_spike_month", "storeA_spike_month",
        "is_store_type_a", "is_store_652", "is_november", "is_october",
        "is_february", "is_december", "is_april", "is_weekend",
        "is_high_error_month",
    }
    colors = ["#EF5350" if f in new_cols else "#90CAF9"
              for f in top["feature"]]

    fig, ax = plt.subplots(figsize=(11, max(8, top_n * 0.35 + 1)))
    ax.barh(range(len(top)), top["importance"].values,
            color=colors, alpha=0.87)
    ax.set_yticks(range(len(top)))
    ax.set_yticklabels(top["feature"].values, fontsize=8)
    ax.invert_yaxis()
    ax.set_xlabel("XGBoost Feature Importance (gain)", fontsize=10)
    ax.set_title(
        f"Top {top_n} Feature Importances  "
        f"(red = new interaction features)\n"
        f"Higher importance = model uses the feature more for splitting",
        fontsize=11, fontweight="bold"
    )
    # Legend
    ax.barh([], [], color="#EF5350", alpha=0.87, label="New feature")
    ax.barh([], [], color="#90CAF9", alpha=0.87, label="Baseline feature")
    ax.legend(fontsize=9)
    ax.grid(axis="x", alpha=0.25)
    plt.tight_layout()
    fig.savefig(FIGURES_DIR / "15c_feature_importance.png",
                dpi=150, bbox_inches="tight")
    plt.close()


def plot_shap(model, X: pd.DataFrame, feat_cols: list, importance: pd.DataFrame):
    if not SHAP_AVAILABLE:
        print("SHAP not available — skipping.")
        return

    print(f"Computing SHAP values on {SHAP_SAMPLE_N:,} sampled rows …")
    sample = X.sample(min(SHAP_SAMPLE_N, len(X)), random_state=42)

    explainer = shap.TreeExplainer(model)
    shap_vals = explainer.shap_values(sample)
    shap_df   = pd.DataFrame(shap_vals, columns=feat_cols)

    # ── Figure 1: SHAP beeswarm (summary_plot owns its own figure) ────────────
    shap.summary_plot(
        shap_vals, sample,
        feature_names=feat_cols,
        max_display=20,
        show=False,
    )
    plt.title("SHAP Summary — Top 20 Features\n"
              "(x-axis = SHAP value; right = increases forecast)",
              fontweight="bold", fontsize=11)
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "15d_shap_summary.png", dpi=150, bbox_inches="tight")
    plt.close()

    # ── Figure 2: Mean |SHAP| for new interaction features only ───────────────
    new_interaction_cols = {f for f in feat_cols if any(
        k in f for k in ["store652", "november", "storeA", "october",
                          "unity", "herbst", "school_hol", "spike",
                          "high_error", "promo_"]
    )}
    shap_imp = (shap_df.abs().mean()
                        .sort_values(ascending=False)
                        .reset_index()
                        .rename(columns={"index": "feature", 0: "mean_abs_shap"}))
    new_shap = shap_imp[shap_imp["feature"].isin(new_interaction_cols)].head(25)

    if not new_shap.empty:
        fig, ax = plt.subplots(figsize=(11, max(5, len(new_shap) * 0.38 + 1)))
        ax.barh(range(len(new_shap)), new_shap["mean_abs_shap"].values,
                color="#EF5350", alpha=0.87)
        ax.set_yticks(range(len(new_shap)))
        ax.set_yticklabels(new_shap["feature"].values, fontsize=8)
        ax.invert_yaxis()
        ax.set_xlabel("Mean |SHAP value|", fontsize=10)
        ax.set_title("Mean |SHAP| — New Interaction Features Only\n"
                     "(non-zero = model is actively using the feature)",
                     fontweight="bold", fontsize=11)
        ax.grid(axis="x", alpha=0.25)
        plt.tight_layout()
        fig.savefig(FIGURES_DIR / "15d_shap_new_features.png",
                    dpi=150, bbox_inches="tight")
        plt.close()

    print("Saved: 15d_shap_summary.png")
    print("       15d_shap_new_features.png")


# ── STORE-LEVEL SPOT CHECK ────────────────────────────────────────────────────
def store_652_spotcheck(baseline_res: pd.DataFrame,
                        augmented_res: pd.DataFrame):
    """Compare Store 652 and November-weekend errors before vs after."""
    print("\n── Store-level Spot Check ────────────────────────────────────")

    for label, results in [("Baseline", baseline_res),
                            ("Augmented", augmented_res)]:
        # Store 652 overall
        s652 = results[(results["Store"] == 652) & (results["y_actual"] > 0)]
        r652 = rmspe(s652["y_actual"].values, s652["y_pred"].values) \
               if len(s652) >= 5 else np.nan

        # November weekends globally
        nov_wknd = results[
            (results["month"] == 11) &
            (results["is_weekend"] == 1) &
            (results["y_actual"] > 0)
        ]
        r_nov_wknd = rmspe(nov_wknd["y_actual"].values,
                           nov_wknd["y_pred"].values) \
                     if len(nov_wknd) >= 5 else np.nan

        # Type A overall
        type_a = results[
            results["StoreType"].str.lower() == "a"] \
            if "StoreType" in results.columns else pd.DataFrame()
        r_a = rmspe(type_a["y_actual"].values, type_a["y_pred"].values) \
              if len(type_a) >= 5 else np.nan

        print(f"\n  {label}:")
        print(f"    Store 652 RMSPE        : {'—' if np.isnan(r652) else f'{r652:.2f}%'}")
        print(f"    Nov Weekend RMSPE      : {'—' if np.isnan(r_nov_wknd) else f'{r_nov_wknd:.2f}%'}")
        print(f"    Store Type A RMSPE     : {'—' if np.isnan(r_a) else f'{r_a:.2f}%'}")


# ── OBSERVATIONS ──────────────────────────────────────────────────────────────
def print_observations(baseline_m: pd.DataFrame,
                       augmented_m: pd.DataFrame,
                       comp: pd.DataFrame,
                       importance: pd.DataFrame,
                       baseline_res: pd.DataFrame,
                       augmented_res: pd.DataFrame):
    print("\n" + "═"*60)
    print("OBSERVATIONS — Feature Engineering Experiment")
    print("═"*60)

    rmspe_comp = comp[comp["metric"] == "rmspe"]

    # 1. Overall improvement
    bsl_overall = rmspe(baseline_res["y_actual"].values,
                        baseline_res["y_pred"].values)
    aug_overall = rmspe(augmented_res["y_actual"].values,
                        augmented_res["y_pred"].values)
    delta_overall = bsl_overall - aug_overall
    print(f"\n  1. Overall RMSPE change:")
    print(f"     Baseline  : {bsl_overall:.3f}%")
    print(f"     Augmented : {aug_overall:.3f}%")
    print(f"     Change    : {-delta_overall:+.3f}pp  "
          f"({'improvement' if delta_overall > 0 else 'no improvement'})")

    # 2. Which features contributed most?
    top5 = importance.head(5)
    new_interaction_cols = {f for f in importance["feature"] if any(
        k in f for k in ["store652", "november", "storeA", "october",
                          "unity", "herbst", "school_hol", "spike",
                          "high_error"])}
    top_new = importance[importance["feature"].isin(new_interaction_cols)].head(3)
    print(f"\n  2. Most important new features (by XGBoost gain):")
    for _, row in top_new.iterrows():
        rank = int(importance[importance["feature"] == row["feature"]].index[0]) + 1
        print(f"     Rank {rank:2d}: {row['feature']:<35} "
              f"(importance={row['importance']:.4f})")

    # 3. Mid horizon improvement
    mid_bsl = rmspe_comp[rmspe_comp["bucket"] == "mid"]
    if not mid_bsl.empty:
        d = float(mid_bsl["abs_change"].iloc[0])
        pct = float(mid_bsl["pct_change"].iloc[0])
        print(f"\n  3. Mid horizon: {d:+.3f}pp  ({pct:+.1f}% relative)  "
              f"→ {'improved' if d < 0 else 'no improvement / worsened'}")
        if abs(d) < 0.5:
            print(f"     Mid anomaly is largely calendar-composition (Oct loading)")
            print(f"     not a feature gap — consistent with Analysis 13 H1.")

    # 4. Store 652 and November weekend
    for label, results in [("Baseline", baseline_res),
                            ("Augmented", augmented_res)]:
        s652 = results[(results["Store"] == 652) & (results["y_actual"] > 0)]
        r652 = rmspe(s652["y_actual"].values, s652["y_pred"].values) \
               if len(s652) >= 5 else np.nan
        nov_wknd = results[
            (results["month"] == 11) & (results["is_weekend"] == 1) &
            (results["y_actual"] > 0)]
        r_nw = rmspe(nov_wknd["y_actual"].values, nov_wknd["y_pred"].values) \
               if len(nov_wknd) >= 5 else np.nan
        if label == "Baseline":
            r652_bsl, r_nw_bsl = r652, r_nw
        else:
            d652 = r652    - r652_bsl  if not np.isnan(r652)    else np.nan
            d_nw = r_nw    - r_nw_bsl  if not np.isnan(r_nw)    else np.nan
            print(f"\n  4. Store 652: {r652_bsl:.2f}% → {r652:.2f}%  "
                  f"({d652:+.2f}pp  "
                  f"{'improved' if d652 < 0 else 'no improvement'})")
            print(f"     Nov Weekend: {r_nw_bsl:.2f}% → {r_nw:.2f}%  "
                  f"({d_nw:+.2f}pp  "
                  f"{'improved' if d_nw < 0 else 'no improvement'})")

    # 5. Feature engineering vs specialized models
    print(f"\n  5. Feature engineering vs store-specific models:")
    if aug_overall < bsl_overall - 1.0:
        print(f"     Feature engineering alone produced a meaningful improvement.")
        print(f"     Store-specific models may provide further gains for Store 652")
        print(f"     if its November pattern is idiosyncratic and not captured by")
        print(f"     the interaction features.")
    else:
        print(f"     Feature engineering produced limited improvement.")
        print(f"     Likely diagnosis: the residual error in Nov/Dec/Store 652 is")
        print(f"     structural sales volatility that features cannot smooth.")
        print(f"     Recommend: per-store residual correction for top-10 error stores")
        print(f"     as a post-processing step (add per-store bias correction).")
    print("─" * 60)


# ── MAIN ──────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("=" * 60)
    print("Script 15 — Feature Engineering Experiment")
    print("=" * 60)

    train_df, store_df = load_raw_data()

    # ── Run both experiments ──────────────────────────────────────────────────
    baseline_res  = run_evaluation(train_df, store_df,
                                   use_new_features=False,
                                   label="Baseline (no new features)")
    augmented_res = run_evaluation(train_df, store_df,
                                   use_new_features=True,
                                   label="Augmented (+ interaction features)")

    # ── Save results ──────────────────────────────────────────────────────────
    baseline_res.to_csv(
        OUTPUT_DIR / "exp15_baseline_results.csv", index=False)
    augmented_res.to_csv(
        OUTPUT_DIR / "exp15_augmented_results.csv", index=False)
    print("\nResults saved to data/processed/exp15_*_results.csv")

    # ── Metrics ───────────────────────────────────────────────────────────────
    print("\n── Metrics by bucket ────────────────────────────────────────")
    baseline_m  = compute_metrics_by_bucket(baseline_res,  "baseline")
    augmented_m = compute_metrics_by_bucket(augmented_res, "augmented")
    comp        = compare_metrics(baseline_m, augmented_m)

    print(f"\n  {'Bucket':8s}  {'Base RMSPE':>11}  "
          f"{'Aug RMSPE':>10}  {'Δ':>8}  {'Δ%':>8}")
    print("  " + "─"*52)
    rmspe_comp = comp[comp["metric"] == "rmspe"]
    for _, row in rmspe_comp.iterrows():
        print(f"  {row['bucket']:8s}  "
              f"{row['baseline']:>10.2f}%  "
              f"{row['augmented']:>9.2f}%  "
              f"{row['abs_change']:>+7.2f}pp  "
              f"{row['pct_change']:>+7.1f}%")

    store_652_spotcheck(baseline_res, augmented_res)

    # ── Feature importance and SHAP (augmented model only) ───────────────────
    model, X_full, feat_cols, importance = get_feature_importance(
        train_df, store_df)

    print("\nTop 15 features by importance:")
    print(importance.head(15)[["feature", "importance"]].to_string(index=False))

    # ── Visualize ────────────────────────────────────────────────────────────
    plot_rmspe_comparison(baseline_m, augmented_m, comp)
    plot_error_contribution(baseline_m, augmented_m)
    plot_feature_importance(importance, top_n=30)
    if SHAP_AVAILABLE:
        plot_shap(model, X_full, feat_cols, importance)

    # ── Observations ─────────────────────────────────────────────────────────
    print_observations(baseline_m, augmented_m, comp, importance,
                       baseline_res, augmented_res)

    print("\n" + "=" * 60)
    print(f"Experiment complete.  Figures → {FIGURES_DIR.resolve()}")
    print("=" * 60)
