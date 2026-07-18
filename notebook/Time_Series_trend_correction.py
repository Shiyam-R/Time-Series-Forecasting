# =============================================================================
# ROSSMANN STORE SALES — TREND CORRECTION
# File   : notebook/06_trend_correction.py
# Input  : data/processed/train.csv
#          data/processed/test.csv
#          artifacts/xgb_model.pkl
#          artifacts/lgbm_model.pkl
#          artifacts/all_features.pkl
# Output : artifacts/store_trend_models.pkl
#          artifacts/global_trend_model.pkl
#          artifacts/reference_date.pkl
#          artifacts/xgb_trend_model.pkl
#          artifacts/lgbm_trend_model.pkl
# =============================================================================
#
# THE PROBLEM:
#   Tree-based models (XGBoost, LightGBM) cannot predict a value higher
#   than what they saw during training. They split data into bins and
#   average within each bin — they never extrapolate beyond the range
#   of values used to train them.
#
#   If a store's sales keep growing past what training data showed,
#   the tree model will systematically UNDERPREDICT — exactly the
#   error that causes real stockouts in a real business.
#
# THE FIX (used by the 1st place Kaggle winner):
#   Step 1 — Fit a simple Ridge regression PER STORE on
#            (days since start) → log_sales. This captures the
#            store's own linear growth trend. Straight lines CAN
#            extrapolate beyond training range, unlike trees.
#   Step 2 — Compute the residual: what the trend line could NOT
#            explain. residual = actual_log_sales - trend_prediction
#   Step 3 — Train XGBoost/LightGBM to predict the RESIDUAL instead
#            of raw log_sales. The tree models now only need to
#            capture non-linear patterns (Promo, holidays, DayOfWeek)
#            on top of an already-correct trend baseline.
#   Step 4 — Final prediction = trend_prediction + residual_prediction
# =============================================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import joblib
import warnings
warnings.filterwarnings("ignore")

from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error
from xgboost import XGBRegressor
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

MIN_ROWS_FOR_TREND = 10   # stores with fewer rows use the global fallback trend


# =============================================================================
# STEP 0 — LOAD DATA, FEATURES, AND PREVIOUSLY TUNED MODELS
# =============================================================================

train = pd.read_csv("data/processed/train.csv", parse_dates=["Date"]).reset_index(drop=True)
test  = pd.read_csv("data/processed/test.csv",  parse_dates=["Date"]).reset_index(drop=True)

ALL_FEATURES = joblib.load("artifacts/all_features.pkl")

# Load previously tuned models — we reuse their hyperparameters
# instead of re-running RandomizedSearchCV, since tuning was already
# confirmed not to be the highest-leverage next step.
xgb_prev_model  = joblib.load("artifacts/xgb_model.pkl")
lgbm_prev_model = joblib.load("artifacts/lgbm_model.pkl")

xgb_best_params  = xgb_prev_model.get_params()
lgbm_best_params = lgbm_prev_model.get_params()

X_train = train[ALL_FEATURES]
X_test  = test[ALL_FEATURES]

print("=" * 55)
print("STEP 0 — DATA AND PRIOR MODELS LOADED")
print("=" * 55)
print(f"Train : {train.shape}")
print(f"Test  : {test.shape}")
print(f"Reusing tuned hyperparameters from 04_modelling.py\n")


# =============================================================================
# STEP 1 — BUILD DAY COUNTER FEATURE
# =============================================================================
#
# reference_date = the very first date in the entire training set.
# Using ONE global reference date (not per-store) keeps the day
# counter on the same real-calendar-time scale for every store,
# so growth rates between stores stay comparable.
#
# days_since_start = 0 on the first day of training, increasing by
# 1 each day after. This is what the Ridge model learns a slope on.
# =============================================================================

reference_date = train["Date"].min()

train["days_since_start"] = (train["Date"] - reference_date).dt.days
test["days_since_start"]  = (test["Date"]  - reference_date).dt.days

print("=" * 55)
print("STEP 1 — DAY COUNTER")
print("=" * 55)
print(f"Reference date (day 0) : {reference_date.date()}")
print(f"Train day range : {train['days_since_start'].min()} → {train['days_since_start'].max()}")
print(f"Test  day range : {test['days_since_start'].min()} → {test['days_since_start'].max()}")
print(f"\nTest days are HIGHER than any day seen in train — this is exactly")
print(f"the extrapolation zone where tree models alone would underpredict.\n")


# =============================================================================
# STEP 2 — FIT PER-STORE RIDGE TREND MODELS
# =============================================================================
#
# For each store: Ridge(days_since_start → log_sales)
# Stores with fewer than MIN_ROWS_FOR_TREND rows fall back to a
# single GLOBAL ridge model fit on all stores combined — avoids
# an unstable trend line from a handful of noisy points.
# =============================================================================

global_ridge = Ridge(alpha=1.0)
global_ridge.fit(train[["days_since_start"]], train["log_sales"])

store_trend_models = {}
fallback_stores = []

for store_id, group in train.groupby("Store"):
    if len(group) < MIN_ROWS_FOR_TREND:
        store_trend_models[store_id] = global_ridge
        fallback_stores.append(store_id)
        continue
    ridge = Ridge(alpha=1.0)
    ridge.fit(group[["days_since_start"]], group["log_sales"])
    store_trend_models[store_id] = ridge

print("=" * 55)
print("STEP 2 — PER-STORE RIDGE TREND MODELS")
print("=" * 55)
print(f"Stores fitted individually : {train['Store'].nunique() - len(fallback_stores)}")
print(f"Stores using global fallback (< {MIN_ROWS_FOR_TREND} rows) : {len(fallback_stores)}")

# Show slope distribution — positive = growing, negative = declining
slopes = pd.Series({
    sid: model.coef_[0] for sid, model in store_trend_models.items()
    if sid not in fallback_stores
})
print(f"\nTrend slope distribution across stores (log_sales per day):")
print(slopes.describe().round(5).to_string())
print(f"\nGrowing stores (positive slope)  : {(slopes > 0).sum()}")
print(f"Declining stores (negative slope): {(slopes < 0).sum()}")


# =============================================================================
# STEP 3 — COMPUTE TREND PREDICTIONS FOR TRAIN AND TEST
# =============================================================================

def predict_trend(df, store_trend_models, global_ridge):
    """Apply each store's own ridge model to predict its trend line."""
    trend_pred = pd.Series(index=df.index, dtype=float)
    for store_id, group in df.groupby("Store"):
        model = store_trend_models.get(store_id, global_ridge)
        trend_pred.loc[group.index] = model.predict(group[["days_since_start"]])
    return trend_pred.values

train["trend_pred_log"] = predict_trend(train, store_trend_models, global_ridge)
test["trend_pred_log"]  = predict_trend(test,  store_trend_models, global_ridge)

print("\n" + "=" * 55)
print("STEP 3 — TREND PREDICTIONS COMPUTED")
print("=" * 55)
print("Sample (Store 1, last 3 train rows + first 3 test rows):")
print(train[train["Store"]==1][["Date","log_sales","trend_pred_log"]].tail(3).to_string())
print(test[test["Store"]==1][["Date","trend_pred_log"]].head(3).to_string())


# =============================================================================
# STEP 4 — COMPUTE RESIDUAL TARGET
# =============================================================================
#
# residual = what's LEFT after removing the trend line.
# This is what XGBoost/LightGBM will now be trained to predict —
# a much "flatter" target that doesn't require extrapolation,
# since the trend already explains the growth direction.
# =============================================================================

train["residual_log"] = train["log_sales"] - train["trend_pred_log"]

print("\n" + "=" * 55)
print("STEP 4 — RESIDUAL TARGET")
print("=" * 55)
print(f"Original log_sales range : {train['log_sales'].min():.3f} → {train['log_sales'].max():.3f}")
print(f"Residual range           : {train['residual_log'].min():.3f} → {train['residual_log'].max():.3f}")
print(f"Residual mean (should be near 0): {train['residual_log'].mean():.4f}")


# =============================================================================
# STEP 5 — RETRAIN XGBOOST AND LIGHTGBM ON THE RESIDUAL
# =============================================================================
#
# Same hyperparameters already tuned in 04_modelling.py — only the
# TARGET changes, from log_sales to residual_log.
# =============================================================================

print("\n" + "=" * 55)
print("STEP 5 — RETRAINING ON RESIDUAL TARGET")
print("=" * 55)

xgb_trend = XGBRegressor(**xgb_best_params)
xgb_trend.fit(X_train, train["residual_log"])
print("XGBoost retrained on residual_log.")

lgbm_trend = LGBMRegressor(**lgbm_best_params)
lgbm_trend.fit(X_train, train["residual_log"])
print("LightGBM retrained on residual_log.")


# =============================================================================
# STEP 6 — FINAL PREDICTIONS  (trend + residual)
# =============================================================================

xgb_residual_pred  = xgb_trend.predict(X_test)
lgbm_residual_pred = lgbm_trend.predict(X_test)

xgb_final_log  = test["trend_pred_log"].values + xgb_residual_pred
lgbm_final_log = test["trend_pred_log"].values + lgbm_residual_pred

xgb_trend_pred  = np.clip(np.expm1(xgb_final_log),  0, None)
lgbm_trend_pred = np.clip(np.expm1(lgbm_final_log), 0, None)

# Original (non-trend-corrected) predictions for comparison
xgb_orig_pred  = np.clip(np.expm1(xgb_prev_model.predict(X_test)),  0, None)
lgbm_orig_pred = np.clip(np.expm1(lgbm_prev_model.predict(X_test)), 0, None)

y_test_raw = test["Sales"].values

print("\n" + "=" * 55)
print("STEP 6 — FINAL PREDICTIONS GENERATED")
print("=" * 55)
print(pd.DataFrame({
    "Actual"          : y_test_raw[:5],
    "XGBoost_orig"    : xgb_orig_pred[:5].round(1),
    "XGBoost_trend"   : xgb_trend_pred[:5].round(1),
    "LightGBM_orig"   : lgbm_orig_pred[:5].round(1),
    "LightGBM_trend"  : lgbm_trend_pred[:5].round(1),
}).to_string())


# =============================================================================
# STEP 7 — METRICS COMPARISON  (Before vs After Trend Correction)
# =============================================================================

def mape(y_true, y_pred):
    mask = y_true != 0
    return np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100

def rmspe(y_true, y_pred):
    mask = y_true != 0
    if mask.sum() == 0:
        return np.nan
    return np.sqrt(np.mean(((y_true[mask] - y_pred[mask]) / y_true[mask]) ** 2)) * 100

def compute_all_metrics(y_true, y_pred, model_name):
    mae  = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mp   = mape(y_true, y_pred)
    rp   = rmspe(y_true, y_pred)
    return {"Model": model_name, "MAE": round(mae,2), "RMSE": round(rmse,2),
            "MAPE": round(mp,2), "RMSPE": round(rp,2)}

results = [
    compute_all_metrics(y_test_raw, xgb_orig_pred,   "XGBoost (original)"),
    compute_all_metrics(y_test_raw, xgb_trend_pred,  "XGBoost (trend-corrected)"),
    compute_all_metrics(y_test_raw, lgbm_orig_pred,  "LightGBM (original)"),
    compute_all_metrics(y_test_raw, lgbm_trend_pred, "LightGBM (trend-corrected)"),
]
metrics_df = pd.DataFrame(results).set_index("Model")

print("\n" + "=" * 55)
print("STEP 7 — METRICS: BEFORE vs AFTER TREND CORRECTION")
print("=" * 55)
print(f"\n{'Model':<28} {'MAE':>10} {'RMSE':>10} {'MAPE':>8} {'RMSPE':>8}")
print("─" * 68)
for model, row in metrics_df.iterrows():
    print(f"{model:<28} {row['MAE']:>10.2f} {row['RMSE']:>10.2f} "
          f"{row['MAPE']:>7.2f}% {row['RMSPE']:>7.2f}%")

xgb_improvement  = metrics_df.loc["XGBoost (original)","RMSPE"]  - metrics_df.loc["XGBoost (trend-corrected)","RMSPE"]
lgbm_improvement = metrics_df.loc["LightGBM (original)","RMSPE"] - metrics_df.loc["LightGBM (trend-corrected)","RMSPE"]

print(f"\nXGBoost  RMSPE change  : {xgb_improvement:+.2f} percentage points")
print(f"LightGBM RMSPE change  : {lgbm_improvement:+.2f} percentage points")

best_model_name = metrics_df["RMSPE"].idxmin()
print(f"\nBest overall model: {best_model_name}")


# =============================================================================
# VISUALISATION 1 — Trend Extrapolation on Top Growing Stores
# =============================================================================
#
# This is the plot that proves the FIX actually works — shows the
# trend line correctly continuing to rise into the test period,
# something the original tree model alone could not do.
# =============================================================================

top_growing_stores = slopes.sort_values(ascending=False).head(3).index.tolist()

fig, axes = plt.subplots(3, 1, figsize=(13, 12), sharex=False)
fig.suptitle("Trend Extrapolation — Top 3 Fastest-Growing Stores",
             fontsize=13, color="#e0f0ff", y=1.01)

for ax, store_id in zip(axes, top_growing_stores):
    store_train = train[train["Store"]==store_id].sort_values("Date")
    store_test  = test[test["Store"]==store_id].sort_values("Date")

    ax.plot(store_train["Date"], store_train["Sales"],
            color="white", lw=1, alpha=0.4, label="Actual (train)")
    ax.plot(store_test["Date"], store_test["Sales"],
            color="white", lw=2, label="Actual (test)")

    # Original XGBoost prediction (capped by training max)
    orig_pred_store = xgb_orig_pred[test["Store"].values == store_id]
    ax.plot(store_test["Date"], orig_pred_store,
            color=ACCENT2, lw=1.5, linestyle="--",
            label="XGBoost original (capped)")

    # Trend-corrected prediction (follows the growth)
    trend_pred_store = xgb_trend_pred[test["Store"].values == store_id]
    ax.plot(store_test["Date"], trend_pred_store,
            color=ACCENT3, lw=1.5, linestyle="--",
            label="XGBoost trend-corrected")

    slope_val = slopes[store_id]
    ax.set_title(f"Store {store_id}  (trend slope: {slope_val:+.5f} log-sales/day)",
                 fontsize=10, color="#c0d8e8")
    ax.set_ylabel("Sales (€)")
    ax.legend(fontsize=8, loc="upper left")
    ax.grid(True)

axes[-1].set_xlabel("Date")
plt.tight_layout()
plt.savefig("trend_extrapolation_examples.png", dpi=150, bbox_inches="tight")
plt.show()


# =============================================================================
# VISUALISATION 2 — Before vs After RMSPE Comparison
# =============================================================================

fig, ax = plt.subplots(figsize=(10, 6))

model_pairs = [
    ("XGBoost",  "XGBoost (original)",  "XGBoost (trend-corrected)"),
    ("LightGBM", "LightGBM (original)", "LightGBM (trend-corrected)"),
]

x = np.arange(len(model_pairs))
width = 0.35

orig_vals  = [metrics_df.loc[o, "RMSPE"] for _, o, t in model_pairs]
trend_vals = [metrics_df.loc[t, "RMSPE"] for _, o, t in model_pairs]

bars1 = ax.bar(x - width/2, orig_vals, width, color=ACCENT2,
               edgecolor="#0f1117", label="Original (no trend correction)")
bars2 = ax.bar(x + width/2, trend_vals, width, color=ACCENT3,
               edgecolor="#0f1117", label="Trend-corrected")

for bars in [bars1, bars2]:
    for bar in bars:
        ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.1,
                f"{bar.get_height():.2f}%", ha="center", fontsize=9, color="#c0d8e8")

ax.set_xticks(x)
ax.set_xticklabels([name for name, _, _ in model_pairs])
ax.set_title("RMSPE — Before vs After Trend Correction",
             fontsize=13, color="#e0f0ff", pad=12)
ax.set_ylabel("RMSPE (%)")
ax.legend(fontsize=10)
ax.grid(True, axis="y")

plt.tight_layout()
plt.savefig("trend_correction_comparison.png", dpi=150, bbox_inches="tight")
plt.show()


# =============================================================================
# SAVE ARTIFACTS
# =============================================================================

joblib.dump(store_trend_models, "artifacts/store_trend_models.pkl")
joblib.dump(global_ridge,       "artifacts/global_trend_model.pkl")
joblib.dump(reference_date,     "artifacts/reference_date.pkl")
joblib.dump(xgb_trend,          "artifacts/xgb_trend_model.pkl")
joblib.dump(lgbm_trend,         "artifacts/lgbm_trend_model.pkl")

print("\n" + "=" * 55)
print("ARTIFACTS SAVED")
print("=" * 55)
print("artifacts/store_trend_models.pkl  → per-store Ridge models")
print("artifacts/global_trend_model.pkl  → fallback Ridge model")
print("artifacts/reference_date.pkl      → day 0 reference date")
print("artifacts/xgb_trend_model.pkl     → XGBoost trained on residuals")
print("artifacts/lgbm_trend_model.pkl    → LightGBM trained on residuals")


# =============================================================================
# FINAL SUMMARY
# =============================================================================

print(f"""
╔═════════════════════════════════════════════════════╗
║         TREND CORRECTION — DONE                      ║
╠═════════════════════════════════════════════════════╣
║  Method                                               ║
║    Per-store Ridge(days_since_start → log_sales)      ║
║    Fallback to global Ridge for stores < {MIN_ROWS_FOR_TREND} rows    ║
║    XGBoost/LightGBM retrained on residual, not raw    ║
║    log_sales — reused prior tuned hyperparameters     ║
║                                                       ║
║  Results                                              ║
║    XGBoost  RMSPE change  : {xgb_improvement:+.2f} pp                  ║
║    LightGBM RMSPE change  : {lgbm_improvement:+.2f} pp                  ║
║    Best model overall     : {best_model_name:<25} ║
║                                                       ║
║  Why this matters for the business                   ║
║    Fixes systematic underprediction during growth —  ║
║    exactly when stockouts and lost sales occur.       ║
║    This targets the asymmetric cost case, not just    ║
║    average accuracy.                                  ║
║                                                       ║
║  Next step → API serving the trend-corrected model    ║
╚═════════════════════════════════════════════════════╝
""")
