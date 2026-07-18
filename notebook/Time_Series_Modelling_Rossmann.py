# =============================================================================
# ROSSMANN STORE SALES — MODEL SELECTION & TRAINING 
# File   : notebook/04_modelling.py
# Input  : data/processed/train.csv
#          data/processed/test.csv
#          artifacts/continuous_features.pkl
#          artifacts/all_features.pkl
# Output : artifacts/linear_model.pkl
#          artifacts/xgb_model.pkl
#          artifacts/lgbm_model.pkl
# =============================================================================
#
# MODELS:
#   1. Linear Regression → baseline sanity check
#   2. XGBoost           → tuned with RandomizedSearchCV
#   3. LightGBM          → tuned with RandomizedSearchCV
#
# WHY NOT SARIMA / ETS HERE:
#   Both only work on a single series. We have 1115 stores and
#   external features (Promo, Holiday) that they cannot use at all.
#   Tree-based ML models are the right fit for this dataset.
#
# CV STRATEGY — CUSTOM PER-STORE SPLITTER:
#   sklearn's TimeSeriesSplit alone would cut by date GLOBALLY —
#   meaning a fold's train/val boundary is the same calendar date
#   for every store, but does not guarantee every store is properly
#   represented within each fold the same way.
#
#   Our custom splitter builds TimeSeriesSplit INDEPENDENTLY for
#   EACH store, then combines fold 1 from every store into one
#   training set, fold 2 from every store into the next, and so on.
#
#   This guarantees every single store contributes proportionally
#   to every fold — critical since we want to evaluate and predict
#   performance per store, not just in aggregate.
# =============================================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import joblib
import os
import warnings
warnings.filterwarnings("ignore")

from sklearn.linear_model import LinearRegression
from sklearn.model_selection import TimeSeriesSplit, RandomizedSearchCV
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


# =============================================================================
# STEP 0 — LOAD DATA AND FEATURE LISTS
# =============================================================================

train = pd.read_csv("data/processed/train.csv", parse_dates=["Date"])
test  = pd.read_csv("data/processed/test.csv",  parse_dates=["Date"])

ALL_FEATURES = joblib.load("artifacts/all_features.pkl")
TARGET       = "log_sales"
TARGET_RAW   = "Sales"

# Reset index — REQUIRED for the custom splitter below.
# The splitter works with integer positions (0, 1, 2...) not the
# original pandas index, so positions must be clean and sequential.
train = train.reset_index(drop=True)
test  = test.reset_index(drop=True)

X_train = train[ALL_FEATURES]
y_train = train[TARGET]
X_test  = test[ALL_FEATURES]
y_test  = test[TARGET]
y_test_raw = test[TARGET_RAW]

print("=" * 55)
print("DATA LOADED")
print("=" * 55)
print(f"Train : {train.shape}  {train['Date'].min().date()} → {train['Date'].max().date()}")
print(f"Test  : {test.shape}   {test['Date'].min().date()} → {test['Date'].max().date()}")
print(f"Features: {len(ALL_FEATURES)}\n")

print("Columns with NaN in X_train:")
print(X_train.columns[X_train.isna().any()].tolist())

print("\nColumns with NaN in X_test:")
print(X_test.columns[X_test.isna().any()].tolist())

print("\nNaN counts (train):")
print(X_train.isna().sum()[X_train.isna().sum() > 0])


# =============================================================================
# STEP 1 — CUSTOM PER-STORE TIME SERIES CV SPLITTER
# =============================================================================
#
# For EACH store independently:
#   Sort that store's rows by date
#   Apply TimeSeriesSplit to get expanding-window folds
#
# Then COMBINE across stores:
#   Fold 1 train  = Store1's fold1 train + Store2's fold1 train + ...
#   Fold 1 val    = Store1's fold1 val   + Store2's fold1 val   + ...
#
# Visually for 3 stores, 3 folds:
#
#   Store 1:  [Tr][Tr][Va]              → fold 1
#             [Tr][Tr][Tr][Va]          → fold 2
#             [Tr][Tr][Tr][Tr][Va]      → fold 3
#
#   Store 2:  [Tr][Tr][Va]              → fold 1
#             [Tr][Tr][Tr][Va]          → fold 2
#             [Tr][Tr][Tr][Tr][Va]      → fold 3
#
#   Combined Fold 1 = Store1 fold1 + Store2 fold1 + Store3 fold1 ...
#   Combined Fold 2 = Store1 fold2 + Store2 fold2 + Store3 fold2 ...
#
# This way every store contributes to every fold — not just
# whichever stores happen to have data in a particular date window.
# =============================================================================

def per_store_time_series_cv(df, n_splits=3, store_col="Store", date_col="Date"):
    """
    Build CV folds independently per store, then combine by fold number.

    Returns a list of (train_indices, val_indices) tuples — the format
    RandomizedSearchCV expects when cv is passed as a list of splits.
    """
    train_folds = [[] for _ in range(n_splits)]
    val_folds   = [[] for _ in range(n_splits)]
    skipped_stores = 0

    for store_id, group in df.groupby(store_col):
        group_sorted = group.sort_values(date_col)
        positions = group_sorted.index.to_numpy()

        # Skip stores with too little history to form n_splits folds
        if len(positions) < n_splits + 1:
            skipped_stores += 1
            continue

        tscv = TimeSeriesSplit(n_splits=n_splits)
        for fold_num, (tr_rel, va_rel) in enumerate(tscv.split(positions)):
            train_folds[fold_num].extend(positions[tr_rel])
            val_folds[fold_num].extend(positions[va_rel])

    splits = [
        (np.array(train_folds[i]), np.array(val_folds[i]))
        for i in range(n_splits)
    ]

    print(f"Custom per-store CV — {n_splits} folds built")
    print(f"Stores skipped (too little data): {skipped_stores}")
    for i, (tr, va) in enumerate(splits, 1):
        print(f"  Fold {i}: train={len(tr):,} rows  val={len(va):,} rows")

    return splits


print("=" * 55)
print("STEP 1 — CUSTOM PER-STORE CV SPLITTER")
print("=" * 55)

# n_splits=3 used here (not 5) to keep RandomizedSearchCV runtime
# reasonable given 800K+ training rows. Increase if your hardware
# allows more compute time.
cv_splits = per_store_time_series_cv(train, n_splits=3)


# =============================================================================
# MODEL 1 — LINEAR REGRESSION (BASELINE)
# =============================================================================
#
# Trained directly on full train data — no tuning needed for a baseline.
# Purpose: sanity check. If XGBoost/LightGBM cannot beat this simple
# model, something is wrong with feature engineering.
#
# NOTE: Linear Regression treats label-encoded categoricals
# (StoreType_enc, Assortment_enc) as continuous numbers, which is not
# fully correct for a linear model. This is acceptable for a baseline
# only — XGBoost and LightGBM (tree-based) do not have this limitation
# since they split on thresholds, not magnitudes.
# =============================================================================

print("\n" + "=" * 55)
print("MODEL 1 — LINEAR REGRESSION (BASELINE)")
print("=" * 55)

linear = LinearRegression()
linear.fit(X_train, y_train)

linear_pred_log = linear.predict(X_test)
linear_pred     = np.expm1(linear_pred_log)   # reverse log1p

print("Baseline trained.")
print(f"Sample predictions vs actual:")
print(pd.DataFrame({
    "Actual"  : y_test_raw.values[:5],
    "Linear"  : linear_pred[:5].round(1)
}).to_string())


# =============================================================================
# MODEL 2 — XGBOOST WITH RANDOMIZEDSEARCHCV
# =============================================================================

print("\n" + "=" * 55)
print("MODEL 2 — XGBOOST + RANDOMIZEDSEARCHCV")
print("=" * 55)

xgb_param_dist = {
    "n_estimators"  : [200, 300, 400, 500],
    "max_depth"     : [4, 6, 8, 10],
    "learning_rate" : [0.01, 0.03, 0.05, 0.1],
    "subsample"     : [0.7, 0.8, 0.9, 1.0],
    "colsample_bytree" : [0.7, 0.8, 0.9, 1.0],
}

xgb_base = XGBRegressor(
    objective="reg:squarederror",
    random_state=42,
    verbosity=0,
    n_jobs=-1
)

print("Running RandomizedSearchCV (10 iterations × 3 folds = 30 fits)...")
print("This may take several minutes depending on hardware.\n")

xgb_search = RandomizedSearchCV(
    estimator=xgb_base,
    param_distributions=xgb_param_dist,
    n_iter=10,
    cv=cv_splits,
    scoring="neg_root_mean_squared_error",
    random_state=42,
    n_jobs=-1,
    verbose=1
)
xgb_search.fit(X_train, y_train)

xgb_best_params = xgb_search.best_params_
print(f"\nBest XGBoost params:")
for k, v in xgb_best_params.items():
    print(f"  {k:18} : {v}")
print(f"Best CV RMSE (log scale): {-xgb_search.best_score_:.4f}")

# Train final model on full train data with best params
xgb_best = XGBRegressor(
    **xgb_best_params,
    objective="reg:squarederror",
    random_state=42,
    verbosity=0,
    n_jobs=-1
)
xgb_best.fit(X_train, y_train)

xgb_pred_log = xgb_best.predict(X_test)
xgb_pred     = np.expm1(xgb_pred_log)

print(f"\nXGBoost final model trained on full train set.")
print(pd.DataFrame({
    "Actual"  : y_test_raw.values[:5],
    "XGBoost" : xgb_pred[:5].round(1)
}).to_string())


# =============================================================================
# MODEL 3 — LIGHTGBM WITH RANDOMIZEDSEARCHCV
# =============================================================================
#
# LightGBM is generally faster than XGBoost on large datasets and
# handles categorical-like integer features efficiently — a good
# fit for our 800K+ row dataset with many label-encoded columns.
# =============================================================================

print("\n" + "=" * 55)
print("MODEL 3 — LIGHTGBM + RANDOMIZEDSEARCHCV")
print("=" * 55)

lgbm_param_dist = {
    "n_estimators"  : [200, 300, 400, 500],
    "max_depth"     : [4, 6, 8, -1],
    "learning_rate" : [0.01, 0.03, 0.05, 0.1],
    "num_leaves"    : [31, 50, 70, 100],
    "subsample"     : [0.7, 0.8, 0.9, 1.0],
}

lgbm_base = LGBMRegressor(
    objective="regression",
    random_state=42,
    verbosity=-1,
    n_jobs=-1
)

print("Running RandomizedSearchCV (10 iterations × 3 folds = 30 fits)...\n")

lgbm_search = RandomizedSearchCV(
    estimator=lgbm_base,
    param_distributions=lgbm_param_dist,
    n_iter=10,
    cv=cv_splits,
    scoring="neg_root_mean_squared_error",
    random_state=42,
    n_jobs=-1,
    verbose=1
)
lgbm_search.fit(X_train, y_train)

lgbm_best_params = lgbm_search.best_params_
print(f"\nBest LightGBM params:")
for k, v in lgbm_best_params.items():
    print(f"  {k:18} : {v}")
print(f"Best CV RMSE (log scale): {-lgbm_search.best_score_:.4f}")

# Train final model on full train data with best params
lgbm_best = LGBMRegressor(
    **lgbm_best_params,
    objective="regression",
    random_state=42,
    verbosity=-1,
    n_jobs=-1
)
lgbm_best.fit(X_train, y_train)

lgbm_pred_log = lgbm_best.predict(X_test)
lgbm_pred     = np.expm1(lgbm_pred_log)

print(f"\nLightGBM final model trained on full train set.")
print(pd.DataFrame({
    "Actual"  : y_test_raw.values[:5],
    "LightGBM": lgbm_pred[:5].round(1)
}).to_string())


# =============================================================================
# SAVE ALL MODEL ARTIFACTS
# =============================================================================

os.makedirs("artifacts", exist_ok=True)

joblib.dump(linear,   "artifacts/linear_model.pkl")
joblib.dump(xgb_best, "artifacts/xgb_model.pkl")
joblib.dump(lgbm_best,"artifacts/lgbm_model.pkl")

print("\n" + "=" * 55)
print("ARTIFACTS SAVED")
print("=" * 55)
print("artifacts/linear_model.pkl")
print("artifacts/xgb_model.pkl")
print("artifacts/lgbm_model.pkl")


# =============================================================================
# VISUALISATION 1 — Aggregated Predictions vs Actual (All Stores Summed)
# =============================================================================

pred_df = test[["Date", "Store"]].copy()
pred_df["Actual"]   = y_test_raw.values
pred_df["Linear"]   = linear_pred
pred_df["XGBoost"]  = xgb_pred
pred_df["LightGBM"] = lgbm_pred

daily_agg = pred_df.groupby("Date")[["Actual","Linear","XGBoost","LightGBM"]].sum()

fig, ax = plt.subplots(figsize=(14, 6))
ax.plot(daily_agg.index, daily_agg["Actual"],
        color="white", lw=2.5, label="Actual", zorder=5)
ax.plot(daily_agg.index, daily_agg["Linear"],
        color=ACCENT2, lw=1.5, linestyle="--", label="Linear Regression")
ax.plot(daily_agg.index, daily_agg["XGBoost"],
        color=ACCENT,  lw=1.5, linestyle="--", label="XGBoost")
ax.plot(daily_agg.index, daily_agg["LightGBM"],
        color=ACCENT3, lw=1.5, linestyle="--", label="LightGBM")

ax.set_title("All Models — Aggregated Daily Sales vs Actual (Test Period)",
             fontsize=13, color="#e0f0ff", pad=12)
ax.set_xlabel("Date")
ax.set_ylabel("Total Daily Sales (€, all stores)")
ax.legend(fontsize=10)
ax.grid(True)

plt.tight_layout()
plt.savefig("model_predictions_aggregated.png", dpi=150, bbox_inches="tight")
plt.show()


# =============================================================================
# VISUALISATION 2 — Single Store Comparison (Store 1)
# =============================================================================

store1_pred = pred_df[pred_df["Store"] == 1].sort_values("Date")

fig, ax = plt.subplots(figsize=(14, 5))
ax.plot(store1_pred["Date"], store1_pred["Actual"],
        color="white", lw=2, label="Actual", zorder=5)
ax.plot(store1_pred["Date"], store1_pred["Linear"],
        color=ACCENT2, lw=1.5, linestyle="--", marker="o", markersize=3, label="Linear")
ax.plot(store1_pred["Date"], store1_pred["XGBoost"],
        color=ACCENT,  lw=1.5, linestyle="--", marker="s", markersize=3, label="XGBoost")
ax.plot(store1_pred["Date"], store1_pred["LightGBM"],
        color=ACCENT3, lw=1.5, linestyle="--", marker="^", markersize=3, label="LightGBM")

ax.set_title("Single Store Comparison — Store 1",
             fontsize=13, color="#e0f0ff", pad=12)
ax.set_xlabel("Date")
ax.set_ylabel("Sales (€)")
ax.legend(fontsize=10)
ax.grid(True)

plt.tight_layout()
plt.savefig("model_predictions_store1.png", dpi=150, bbox_inches="tight")
plt.show()


# =============================================================================
# VISUALISATION 3 — Feature Importance (XGBoost vs LightGBM)
# =============================================================================

xgb_importance = pd.DataFrame({
    "feature"   : ALL_FEATURES,
    "importance": xgb_best.feature_importances_
}).sort_values("importance", ascending=True).tail(15)

lgbm_importance = pd.DataFrame({
    "feature"   : ALL_FEATURES,
    "importance": lgbm_best.feature_importances_
}).sort_values("importance", ascending=True).tail(15)

fig, axes = plt.subplots(1, 2, figsize=(16, 7))
fig.suptitle("Top 15 Feature Importance — XGBoost vs LightGBM",
             fontsize=13, color="#e0f0ff", y=1.02)

axes[0].barh(xgb_importance["feature"], xgb_importance["importance"],
             color=ACCENT, edgecolor="#0f1117")
axes[0].set_title("XGBoost", fontsize=11, color="#c0d8e8")
axes[0].grid(True, axis="x")

axes[1].barh(lgbm_importance["feature"], lgbm_importance["importance"],
             color=ACCENT3, edgecolor="#0f1117")
axes[1].set_title("LightGBM", fontsize=11, color="#c0d8e8")
axes[1].grid(True, axis="x")

plt.tight_layout()
plt.savefig("model_feature_importance.png", dpi=150, bbox_inches="tight")
plt.show()


# =============================================================================
# FINAL SUMMARY
# =============================================================================

print("""
╔═════════════════════════════════════════════════════╗
║         MODELLING — DONE                            ║
╠═════════════════════════════════════════════════════╣
║  Model 1 — Linear Regression (baseline)              ║
║    No tuning — trained directly on full train data   ║
║                                                      ║
║  Model 2 — XGBoost                                   ║
║    Tuned with RandomizedSearchCV (10 iter × 3 folds) ║
║    CV  : custom per-store splitter                   ║
║                                                      ║
║  Model 3 — LightGBM                                  ║
║    Tuned with RandomizedSearchCV (10 iter × 3 folds) ║
║    CV  : custom per-store splitter                   ║
║                                                      ║
║  CV Strategy                                         ║
║    Per-store TimeSeriesSplit, combined by fold        ║
║    Guarantees every store contributes to every fold  ║
║                                                      ║
║  Dropped                                             ║
║    SARIMA, ETS — cannot use external features or     ║
║    scale to 1115 stores                              ║
║                                                      ║
║  Saved                                               ║
║    artifacts/linear_model.pkl                        ║
║    artifacts/xgb_model.pkl                           ║
║    artifacts/lgbm_model.pkl                          ║
║                                                      ║
║  Next step → 05_evaluation.py                        ║
║  (MAE, RMSE, MAPE, RMSPE — per-store breakdown)      ║
╚═════════════════════════════════════════════════════╝
""")
