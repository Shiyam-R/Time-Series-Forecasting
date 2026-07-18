# =============================================================================
# ROSSMANN STORE SALES — SECTION 1: FORECASTING BASELINES (FULL COMPARISON)
# File   : notebook/10_baselines.py
# Input  : data/processed/features.csv
#          data/processed/train_horizon.csv
#          data/processed/test_horizon.csv
#          artifacts/xgb_horizon_*.pkl
#          artifacts/last_known_per_store.pkl
# =============================================================================
#
# SEVEN METHODS COMPARED, PER BUCKET:
#   1. Seasonal Naive (weekly)   — same day last week, period=7
#   2. Seasonal Naive (monthly)  — same "day count" last month, period=30
#   3. Seasonal Naive (yearly)   — same day last year, period=365
#   4. Moving Average            — anchor's rolling_mean_30, held flat
#   5. Historical Mean           — store's all-time average
#   6. Linear Regression         — trained fresh per bucket, no tuning
#   7. LightGBM                  — trained fresh per bucket, RandomizedSearchCV
#   8. XGBoost                   — reused from 08_horizon_modelling.py
#
# RUNTIME WARNING:
#   LightGBM gets the SAME tuning treatment XGBoost already received —
#   10 iterations x 3 folds x 4 buckets = 120 more fits, comparable
#   runtime to the original XGBoost training session. Linear
#   Regression is fast (seconds). Budget for another long run.
#
# WHY MULTIPLE SEASONAL PERIODS:
#   Weekly  -> captures the DayOfWeek/Promo retail cadence
#   Monthly -> a 30-day period naive, consistent with how lag_30 /
#              rolling_mean_30 are already defined elsewhere in this
#              project (day-count based, not calendar-month based)
#   Yearly  -> captures Christmas/Easter-style annual repetition that
#              a 7-day-only naive could never see
#
# Each is built with the SAME leakage-safe principle: the reference
# date a naive method looks up must NEVER fall later than the anchor
# (today) date — walking back in multiples of the period until that
# holds, exactly like the lag features themselves.
# =============================================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import joblib
import warnings
warnings.filterwarnings("ignore")

from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import TimeSeriesSplit, RandomizedSearchCV
from sklearn.metrics import mean_absolute_error, mean_squared_error
from lightgbm import LGBMRegressor


plt.rcParams.update({
    "figure.facecolor" : "#0f1117",
    "axes.facecolor"   : "#161b27",
    "axes.edgecolor"   : "#2a3550",
    "axes.labelcolor"  : "#a0b8d0",
    "xtick.color"      : "#607080",
    "ytick.color"      : "#607080",
    "text.color"       : "#c8d8e8",
    "grid.color"       : "#1e2d40",
    "grid.linestyle"   : "--",
    "grid.alpha"       : 0.5,
    "font.family"      : "monospace",
})

ACCENT  = "#4a9eff"
ACCENT2 = "#ff6b6b"
ACCENT3 = "#06ffa5"
ACCENT4 = "#ffd93d"
ACCENT5 = "#c77dff"
ACCENT6 = "#ff9a3c"
ACCENT7 = "#5ad1e6"

METHOD_COLORS = {
    "Naive (weekly)"  : ACCENT2,
    "Naive (monthly)" : "#c97a7a",
    "Naive (yearly)"  : "#8a4f4f",
    "Moving Average"  : ACCENT4,
    "Historical Mean" : ACCENT5,
    "Linear Regression": ACCENT7,
    "LightGBM"        : ACCENT6,
    "XGBoost"         : ACCENT3,
}
METHOD_ORDER = ["Naive (weekly)","Naive (monthly)","Naive (yearly)",
                 "Moving Average","Historical Mean",
                 "Linear Regression","LightGBM","XGBoost"]
BUCKET_ORDER = ["near", "mid", "far", "extended"]


# =============================================================================
# METRIC FUNCTIONS
# =============================================================================

def mape(y_true, y_pred):
    mask = y_true != 0
    return np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100

def rmspe(y_true, y_pred):
    mask = y_true != 0
    if mask.sum() == 0:
        return np.nan
    return np.sqrt(np.mean(((y_true[mask] - y_pred[mask]) / y_true[mask]) ** 2)) * 100

def compute_all_metrics(y_true, y_pred):
    y_true_arr = np.asarray(y_true, dtype=float)
    y_pred_arr = np.asarray(y_pred, dtype=float)
    if len(y_true_arr) == 0:
        return {"MAE": np.nan, "RMSE": np.nan, "MAPE": np.nan, "RMSPE": np.nan, "n": 0}
    return {
        "MAE"  : round(mean_absolute_error(y_true_arr, y_pred_arr), 2),
        "RMSE" : round(np.sqrt(mean_squared_error(y_true_arr, y_pred_arr)), 2),
        "MAPE" : round(mape(y_true_arr, y_pred_arr), 2),
        "RMSPE": round(rmspe(y_true_arr, y_pred_arr), 2),
        "n"    : len(y_true_arr),
    }


# =============================================================================
# CUSTOM PER-STORE CV SPLITTER  (same as 04 / 08)
# =============================================================================

def per_store_time_series_cv(df, n_splits=3, store_col="Store", date_col="Date"):
    train_folds = [[] for _ in range(n_splits)]
    val_folds   = [[] for _ in range(n_splits)]
    skipped = 0
    for store_id, group in df.groupby(store_col):
        group_sorted = group.sort_values(date_col)
        positions = group_sorted.index.to_numpy()
        if len(positions) < n_splits + 1:
            skipped += 1
            continue
        tscv = TimeSeriesSplit(n_splits=n_splits)
        for fold_num, (tr_rel, va_rel) in enumerate(tscv.split(positions)):
            train_folds[fold_num].extend(positions[tr_rel])
            val_folds[fold_num].extend(positions[va_rel])
    splits = [(np.array(train_folds[i]), np.array(val_folds[i])) for i in range(n_splits)]
    return splits, skipped


# =============================================================================
# LEAKAGE-SAFE SEASONAL NAIVE  (reusable for any period)
# =============================================================================

def compute_seasonal_naive(test, day_level, period, store_historical_mean, horizon_col="horizon_actual"):
    k = np.ceil(test[horizon_col] / period).astype(int)
    ref_date = test["target_date"] - pd.to_timedelta(period * k, unit="D")

    violations = (ref_date > test["Date"]).sum()
    assert violations == 0, f"Seasonal Naive (period={period}) leaks into the future! ({violations} violations)"

    lookup_keys = list(zip(test["Store"], ref_date))
    pred = pd.Series(day_level.reindex(lookup_keys).values, index=test.index)

    missing_mask = pred.isna()
    pred[missing_mask] = test.loc[missing_mask, "Store"].map(store_historical_mean)

    return pred, missing_mask.sum()


# =============================================================================
# STEP 0 — LOAD DATA
# =============================================================================

features = pd.read_csv("data/processed/features.csv", parse_dates=["Date"])
train    = pd.read_csv("data/processed/train_horizon.csv", parse_dates=["Date","target_date"])
test     = pd.read_csv("data/processed/test_horizon.csv", parse_dates=["Date","target_date"])

ALL_FEATURES = joblib.load("artifacts/horizon_all_features.pkl")
last_known   = joblib.load("artifacts/last_known_per_store.pkl")
TARGET = "target_log_sales"

xgb_models = {
    "near"     : joblib.load("artifacts/xgb_horizon_near.pkl"),
    "mid"      : joblib.load("artifacts/xgb_horizon_mid.pkl"),
    "far"      : joblib.load("artifacts/xgb_horizon_far.pkl"),
    "extended" : joblib.load("artifacts/xgb_horizon_extended.pkl"),
}

print("=" * 60)
print("STEP 0 — DATA LOADED")
print("=" * 60)
print(f"Train pairs : {len(train):,}")
print(f"Test pairs  : {len(test):,}\n")

train_boundary = features["Date"].max() - pd.DateOffset(months=4)
train_features = features[features["Date"] <= train_boundary]
day_level = features.set_index(["Store", "Date"])["Sales"]
store_historical_mean = train_features.groupby("Store")["Sales"].mean()


# =============================================================================
# STEP 0b — RECOVER THE REAL HORIZON  (the "horizon" column is SCALED)
# =============================================================================
#
# "horizon" was included as a continuous model feature in
# 07_horizon_dataset.py and StandardScaler-transformed along with
# the other continuous features before being saved to CSV. For date
# arithmetic (like Seasonal Naive lookups) we need the REAL day
# count, not the scaled version. target_date and Date were never
# scaled — recompute horizon directly from their difference instead
# of trusting the scaled "horizon" column.
# =============================================================================

test["horizon_actual"] = (test["target_date"] - test["Date"]).dt.days

print("=" * 60)
print("STEP 0b — RECOVERED REAL HORIZON FROM DATES")
print("=" * 60)
print(f"horizon column (scaled, DO NOT USE for date math) range: "
      f"{test['horizon'].min():.2f} -> {test['horizon'].max():.2f}")
print(f"horizon_actual (real days, recomputed)            range: "
      f"{test['horizon_actual'].min()} -> {test['horizon_actual'].max()}")

bucket_check = {
    "near": (1,14), "mid": (15,30), "far": (31,60), "extended": (61,90)
}
for bucket, (lo, hi) in bucket_check.items():
    sub = test[test["horizon_bucket"] == bucket]["horizon_actual"]
    in_range = sub.between(lo, hi).all()
    print(f"  {bucket:<10} horizon_actual in [{lo},{hi}]: {'OK' if in_range else 'MISMATCH!!'}")


# =============================================================================
# STEP 1 — THREE SEASONAL NAIVE VARIANTS
# =============================================================================

print("=" * 60)
print("STEP 1 — SEASONAL NAIVE (weekly / monthly / yearly)")
print("=" * 60)

for period, label, col in [(7, "weekly", "naive_weekly_pred"),
                            (30, "monthly", "naive_monthly_pred"),
                            (365, "yearly", "naive_yearly_pred")]:
    pred, n_fallback = compute_seasonal_naive(test, day_level, period, store_historical_mean)
    test[col] = pred
    print(f"  {label:<8} (period={period:>3}) -> fallback used on {n_fallback:,} rows "
          f"({n_fallback/len(test)*100:.2f}%)")


# =============================================================================
# STEP 2 — MOVING AVERAGE & HISTORICAL MEAN
# =============================================================================

print("\n" + "=" * 60)
print("STEP 2 — MOVING AVERAGE & HISTORICAL MEAN")
print("=" * 60)

test["moving_avg_pred"]      = test["Store"].map(last_known["rolling_mean_30"])
test["historical_mean_pred"] = test["Store"].map(store_historical_mean)
print("Both computed — constant per store, regardless of horizon.")


# =============================================================================
# STEP 3 — LINEAR REGRESSION PER BUCKET  (fast, no tuning)
# =============================================================================

print("\n" + "=" * 60)
print("STEP 3 — LINEAR REGRESSION PER BUCKET")
print("=" * 60)

linear_models = {}
for bucket in BUCKET_ORDER:
    bucket_train = train[train["horizon_bucket"] == bucket]
    X_train = bucket_train[ALL_FEATURES]
    y_train = bucket_train[TARGET]

    lr = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("model", LinearRegression())])
    
    lr.fit(X_train, y_train)
    linear_models[bucket] = lr
    print(f"  {bucket:<10} trained on {len(bucket_train):,} rows")

test["linear_pred"] = np.nan
for bucket, model in linear_models.items():
    mask = test["horizon_bucket"] == bucket
    pred_log = model.predict(test.loc[mask, ALL_FEATURES])
    test.loc[mask, "linear_pred"] = np.clip(np.expm1(pred_log), 0, None)


# =============================================================================
# STEP 4 — LIGHTGBM PER BUCKET  (tuned, same rigor as XGBoost)
# =============================================================================

print("\n" + "=" * 60)
print("STEP 4 — LIGHTGBM PER BUCKET  (RandomizedSearchCV, 10 iter x 3 folds)")
print("=" * 60)
print("This is the long step — comparable runtime to the original")
print("XGBoost bucket training session. Treat as a background job.\n")

lgbm_param_dist = {
    "n_estimators"  : [200, 300, 400, 500],
    "max_depth"     : [4, 6, 8, -1],
    "learning_rate" : [0.01, 0.03, 0.05, 0.1],
    "num_leaves"    : [31, 50, 70, 100],
    "subsample"     : [0.7, 0.8, 0.9, 1.0],
}

lgbm_models = {}
for bucket in BUCKET_ORDER:
    print(f"\n--- {bucket.upper()} ---")
    bucket_train = train[train["horizon_bucket"] == bucket].reset_index(drop=True)
    X_train = bucket_train[ALL_FEATURES]
    y_train = bucket_train[TARGET]

    cv_splits, skipped = per_store_time_series_cv(bucket_train, n_splits=3)
    print(f"CV folds built — stores skipped: {skipped}")

    lgbm_base = LGBMRegressor(objective="regression", random_state=42,
                               verbosity=-1, n_jobs=-1)
    search = RandomizedSearchCV(
        estimator=lgbm_base, param_distributions=lgbm_param_dist,
        n_iter=10, cv=cv_splits, scoring="neg_root_mean_squared_error",
        random_state=42, n_jobs=-1, verbose=0
    )
    search.fit(X_train, y_train)
    print(f"Best params: {search.best_params_}")

    final_model = LGBMRegressor(**search.best_params_, objective="regression",
                                 random_state=42, verbosity=-1, n_jobs=-1)
    final_model.fit(X_train, y_train)
    lgbm_models[bucket] = final_model
    joblib.dump(final_model, f"artifacts/lgbm_horizon_{bucket}.pkl")
    print(f"Saved -> artifacts/lgbm_horizon_{bucket}.pkl")

test["lgbm_pred"] = np.nan
for bucket, model in lgbm_models.items():
    mask = test["horizon_bucket"] == bucket
    pred_log = model.predict(test.loc[mask, ALL_FEATURES])
    test.loc[mask, "lgbm_pred"] = np.clip(np.expm1(pred_log), 0, None)


# =============================================================================
# STEP 5 — XGBOOST PREDICTIONS  (already trained, just predict)
# =============================================================================

print("\n" + "=" * 60)
print("STEP 5 — XGBOOST BUCKET PREDICTIONS (reused, no retraining)")
print("=" * 60)

test["xgb_pred"] = np.nan
for bucket, model in xgb_models.items():
    mask = test["horizon_bucket"] == bucket
    pred_log = model.predict(test.loc[mask, ALL_FEATURES])
    test.loc[mask, "xgb_pred"] = np.clip(np.expm1(pred_log), 0, None)
print("Done.")


# =============================================================================
# STEP 6 — FULL COMPARISON TABLE
# =============================================================================

method_cols = {
    "Naive (weekly)"   : "naive_weekly_pred",
    "Naive (monthly)"  : "naive_monthly_pred",
    "Naive (yearly)"   : "naive_yearly_pred",
    "Moving Average"   : "moving_avg_pred",
    "Historical Mean"  : "historical_mean_pred",
    "Linear Regression": "linear_pred",
    "LightGBM"         : "lgbm_pred",
    "XGBoost"          : "xgb_pred",
}

results = []
for bucket in BUCKET_ORDER:
    sub = test[test["horizon_bucket"] == bucket]
    for method_name, col in method_cols.items():
        m = compute_all_metrics(sub["target_Sales"].values, sub[col].values)
        results.append({"bucket": bucket, "method": method_name, **m})

results_df = pd.DataFrame(results)

print("\n" + "=" * 60)
print("STEP 6 — FULL COMPARISON TABLE (per bucket)")
print("=" * 60)
for bucket in BUCKET_ORDER:
    sub = results_df[results_df["bucket"] == bucket].set_index("method").loc[METHOD_ORDER]
    best_method = sub["RMSPE"].idxmin()
    print(f"\n{bucket.upper()}")
    print(f"{'Method':<20} {'MAE':>10} {'RMSE':>10} {'MAPE':>8} {'RMSPE':>8}")
    print("-" * 60)
    for method, row in sub.iterrows():
        mark = " <- BEST" if method == best_method else ""
        print(f"{method:<20} {row['MAE']:>10.2f} {row['RMSE']:>10.2f} "
              f"{row['MAPE']:>7.2f}% {row['RMSPE']:>7.2f}%{mark}")


# =============================================================================
# STEP 7 — INTERPRETATION
# =============================================================================

print("\n" + "=" * 60)
print("STEP 7 — INTERPRETATION: IS THE COMPLEXITY WORTH IT?")
print("=" * 60)

simple_methods = ["Naive (weekly)","Naive (monthly)","Naive (yearly)",
                   "Moving Average","Historical Mean"]

for bucket in BUCKET_ORDER:
    sub = results_df[results_df["bucket"] == bucket].set_index("method")
    best_naive = sub.loc[simple_methods, "RMSPE"]
    best_naive_name, best_naive_val = best_naive.idxmin(), best_naive.min()

    linear_rmspe = sub.loc["Linear Regression", "RMSPE"]
    lgbm_rmspe   = sub.loc["LightGBM", "RMSPE"]
    xgb_rmspe    = sub.loc["XGBoost", "RMSPE"]

    print(f"\n{bucket.upper()}:")
    print(f"  Best simple baseline : {best_naive_name} ({best_naive_val:.2f}%)")
    print(f"  Linear Regression    : {linear_rmspe:.2f}%  "
          f"({best_naive_val - linear_rmspe:+.2f}pp vs best naive, seconds to train)")
    print(f"  LightGBM (tuned)     : {lgbm_rmspe:.2f}%  "
          f"({linear_rmspe - lgbm_rmspe:+.2f}pp vs Linear, ~hour to train)")
    print(f"  XGBoost (tuned)      : {xgb_rmspe:.2f}%  "
          f"({linear_rmspe - xgb_rmspe:+.2f}pp vs Linear, ~hour to train)")

    tree_vs_linear = linear_rmspe - min(lgbm_rmspe, xgb_rmspe)
    if tree_vs_linear < 1.0:
        print(f"  -> Tree models barely beat Linear Regression here ({tree_vs_linear:+.2f}pp)."
              f" The hours of tuning may not be earning their cost for this bucket.")
    else:
        print(f"  -> Tree models meaningfully beat Linear Regression ({tree_vs_linear:+.2f}pp)."
              f" The added complexity is justified here.")


# =============================================================================
# VISUALISATION 1 — Full Method Comparison, All Buckets
# =============================================================================

fig, axes = plt.subplots(1, 4, figsize=(22, 6))
fig.suptitle("RMSPE — All 8 Methods, per Bucket", fontsize=13, color="#e0f0ff", y=1.03)

for ax, bucket in zip(axes, BUCKET_ORDER):
    sub = results_df[results_df["bucket"] == bucket].set_index("method").loc[METHOD_ORDER]
    colors = [METHOD_COLORS[m] for m in sub.index]
    bars = ax.barh(sub.index, sub["RMSPE"], color=colors, edgecolor="#0f1117")
    for bar, val in zip(bars, sub["RMSPE"]):
        ax.text(bar.get_width()+0.5, bar.get_y()+bar.get_height()/2,
                f"{val:.1f}%", va="center", fontsize=8, color="#c0d8e8")
    ax.set_title(bucket, fontsize=11, color="#c0d8e8")
    ax.set_xlabel("RMSPE (%)")
    ax.grid(True, axis="x")
    ax.invert_yaxis()

plt.tight_layout()
plt.savefig("baseline_full_comparison.png", dpi=150, bbox_inches="tight")
plt.show()


# =============================================================================
# VISUALISATION 2 — Cost vs Benefit: Training Time vs Accuracy
# =============================================================================

approx_train_minutes = {
    "Naive (weekly)": 0, "Naive (monthly)": 0, "Naive (yearly)": 0,
    "Moving Average": 0, "Historical Mean": 0,
    "Linear Regression": 1, "LightGBM": 90, "XGBoost": 90,
}

fig, ax = plt.subplots(figsize=(11, 7))
for bucket, marker in zip(BUCKET_ORDER, ["o","s","^","D"]):
    sub = results_df[results_df["bucket"] == bucket].set_index("method")
    x = [approx_train_minutes[m] for m in METHOD_ORDER]
    y = [sub.loc[m, "RMSPE"] for m in METHOD_ORDER]
    ax.scatter(x, y, label=bucket, s=70, marker=marker, alpha=0.85)

ax.set_xscale("symlog")
ax.set_title("Cost vs Benefit — Approx. Training Time vs RMSPE",
             fontsize=13, color="#e0f0ff", pad=12)
ax.set_xlabel("Approx. training time (minutes, log scale)")
ax.set_ylabel("RMSPE (%)")
ax.legend(fontsize=9, title="Bucket")
ax.grid(True)

plt.tight_layout()
plt.savefig("baseline_cost_vs_benefit.png", dpi=150, bbox_inches="tight")
plt.show()


# =============================================================================
# SAVE RESULTS
# =============================================================================

results_df.to_csv("data/processed/baseline_comparison_results.csv", index=False)
test.to_csv("data/processed/test_horizon_with_baselines.csv", index=False)

print(f"""
================================================================
       SECTION 1 — FULL BASELINE COMPARISON — DONE
================================================================
  8 methods compared per bucket:
    3x Seasonal Naive (weekly/monthly/yearly), Moving Average,
    Historical Mean, Linear Regression, LightGBM, XGBoost

  Saved
    data/processed/baseline_comparison_results.csv
    data/processed/test_horizon_with_baselines.csv
    artifacts/lgbm_horizon_*.pkl  (newly trained)

  Plots
    baseline_full_comparison.png
    baseline_cost_vs_benefit.png

  Next -> Section 2: Horizon-Level Error Analysis (refresh)
================================================================
""")