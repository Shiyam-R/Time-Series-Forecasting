# =============================================================================
# ROSSMANN STORE SALES — MULTI-HORIZON MODELLING
# File   : notebook/08_horizon_modelling.py
# Input  : data/processed/train_horizon.csv
#          data/processed/test_horizon.csv
#          artifacts/horizon_all_features.pkl
#          artifacts/horizon_boundary_dates.pkl
# Output : artifacts/xgb_horizon_near.pkl
#          artifacts/xgb_horizon_mid.pkl
#          artifacts/xgb_horizon_far.pkl
#          artifacts/xgb_horizon_extended.pkl
#          artifacts/horizon_best_params.pkl
# =============================================================================
#
# WHY 4 SEPARATE MODELS, EACH INDEPENDENTLY TUNED:
#   Near-horizon predictions lean heavily on lag/rolling features —
#   last week's sales strongly predicts next week's sales.
#   Extended-horizon predictions can barely use that signal at all —
#   90 days out, last week's actual numbers are nearly irrelevant,
#   and the model has to rely on seasonality, Promo and calendar
#   patterns instead. These are structurally different problems,
#   so each bucket gets its OWN RandomizedSearchCV — sharing one
#   set of hyperparameters across all 4 would assume a similarity
#   that doesn't actually exist.
#
# RUNTIME WARNING:
#   Each bucket has 400K-580K training rows. RandomizedSearchCV here
#   runs 10 iterations x 3 folds = 30 fits PER BUCKET, x 4 buckets
#   = 120 total XGBoost fits. This can take a long time depending on
#   hardware — treat this as a "run it and come back later" job,
#   similar to how the Kaggle winners trained their solutions
#   overnight.
# =============================================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import joblib
import warnings
warnings.filterwarnings("ignore")

from sklearn.model_selection import TimeSeriesSplit, RandomizedSearchCV
from sklearn.metrics import mean_absolute_error, mean_squared_error
from xgboost import XGBRegressor

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
BUCKET_COLORS = {"near": ACCENT3, "mid": ACCENT4, "far": ACCENT2, "extended": ACCENT5}


# =============================================================================
# STEP 0 — LOAD DATA
# =============================================================================

train = pd.read_csv("data/processed/train_horizon.csv", parse_dates=["Date","target_date"])
test  = pd.read_csv("data/processed/test_horizon.csv",  parse_dates=["Date","target_date"])

ALL_FEATURES = joblib.load("artifacts/horizon_all_features.pkl")
boundary_info = joblib.load("artifacts/horizon_boundary_dates.pkl")
HORIZON_BUCKETS = boundary_info["horizon_buckets"]

TARGET = "target_log_sales"

print("=" * 60)
print("STEP 0 — DATA LOADED")
print("=" * 60)
print(f"Train pairs : {len(train):,}")
print(f"Test  pairs : {len(test):,}")
print(f"Features    : {len(ALL_FEATURES)}")
print(f"Buckets     : {list(HORIZON_BUCKETS.keys())}\n")


# =============================================================================
# STEP 1 — CUSTOM PER-STORE TIME SERIES CV SPLITTER
# =============================================================================
#
# Same logic as 04_modelling.py — builds folds independently per
# store (by anchor Date), then combines fold-by-fold across stores.
# Re-applied here per bucket subset.
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
    y_true_arr = np.asarray(y_true)
    y_pred_arr = np.asarray(y_pred)
    return {
        "MAE"  : round(mean_absolute_error(y_true_arr, y_pred_arr), 2),
        "RMSE" : round(np.sqrt(mean_squared_error(y_true_arr, y_pred_arr)), 2),
        "MAPE" : round(mape(y_true_arr, y_pred_arr), 2),
        "RMSPE": round(rmspe(y_true_arr, y_pred_arr), 2),
    }


# =============================================================================
# STEP 2 — TRAIN ONE TUNED MODEL PER BUCKET
# =============================================================================

xgb_param_dist = {
    "n_estimators"     : [200, 300, 400, 500],
    "max_depth"        : [4, 6, 8, 10],
    "learning_rate"    : [0.01, 0.03, 0.05, 0.1],
    "subsample"        : [0.7, 0.8, 0.9, 1.0],
    "colsample_bytree" : [0.7, 0.8, 0.9, 1.0],
}

bucket_models       = {}
bucket_best_params  = {}
bucket_metrics      = {}
bucket_predictions  = {}
bucket_feature_imp  = {}

for bucket in HORIZON_BUCKETS.keys():
    print("\n" + "=" * 60)
    print(f"BUCKET: {bucket.upper()}  (horizons: {HORIZON_BUCKETS[bucket]})")
    print("=" * 60)

    bucket_train = train[train["horizon_bucket"] == bucket].reset_index(drop=True)
    bucket_test  = test[test["horizon_bucket"] == bucket].reset_index(drop=True)

    X_train = bucket_train[ALL_FEATURES]
    y_train = bucket_train[TARGET]
    X_test  = bucket_test[ALL_FEATURES]
    y_test_raw = bucket_test["target_Sales"]

    print(f"Train rows: {len(bucket_train):,}   Test rows: {len(bucket_test):,}")

    # Build CV splits for this bucket
    cv_splits, skipped = per_store_time_series_cv(bucket_train, n_splits=3)
    print(f"CV folds built — stores skipped (too little data): {skipped}")

    # RandomizedSearchCV — independently tuned for this bucket
    xgb_base = XGBRegressor(objective="reg:squarederror", random_state=42,
                             verbosity=0, n_jobs=-1)

    print(f"Running RandomizedSearchCV for {bucket} bucket (10 iter x 3 folds = 30 fits)...")
    search = RandomizedSearchCV(
        estimator=xgb_base,
        param_distributions=xgb_param_dist,
        n_iter=10,
        cv=cv_splits,
        scoring="neg_root_mean_squared_error",
        random_state=42,
        n_jobs=-1,
        verbose=0
    )
    search.fit(X_train, y_train)

    best_params = search.best_params_
    print(f"Best params for {bucket}:")
    for k, v in best_params.items():
        print(f"  {k:18} : {v}")

    # Train final model on full bucket train data
    final_model = XGBRegressor(**best_params, objective="reg:squarederror",
                                random_state=42, verbosity=0, n_jobs=-1)
    final_model.fit(X_train, y_train)

    # Predict and evaluate
    pred_log = final_model.predict(X_test)
    pred     = np.clip(np.expm1(pred_log), 0, None)

    metrics = compute_all_metrics(y_test_raw.values, pred)
    print(f"\n{bucket.upper()} bucket test metrics:")
    for k, v in metrics.items():
        print(f"  {k:6} : {v}")

    bucket_models[bucket]      = final_model
    bucket_best_params[bucket] = best_params
    bucket_metrics[bucket]     = metrics
    bucket_predictions[bucket] = {
        "horizon"  : bucket_test["horizon"].values,
        "actual"   : y_test_raw.values,
        "predicted": pred,
    }
    bucket_feature_imp[bucket] = pd.Series(
        final_model.feature_importances_, index=ALL_FEATURES
    ).sort_values(ascending=False)

    joblib.dump(final_model, f"artifacts/xgb_horizon_{bucket}.pkl")
    print(f"Saved -> artifacts/xgb_horizon_{bucket}.pkl")

joblib.dump(bucket_best_params, "artifacts/horizon_best_params.pkl")


# =============================================================================
# STEP 3 — BUCKET-LEVEL METRICS SUMMARY
# =============================================================================

metrics_df = pd.DataFrame(bucket_metrics).T
metrics_df = metrics_df.loc[["near","mid","far","extended"]]

print("\n" + "=" * 60)
print("STEP 3 — METRICS BY BUCKET")
print("=" * 60)
print(f"\n{'Bucket':<10} {'MAE':>10} {'RMSE':>10} {'MAPE':>8} {'RMSPE':>8}")
print("-" * 50)
for bucket, row in metrics_df.iterrows():
    print(f"{bucket:<10} {row['MAE']:>10.2f} {row['RMSE']:>10.2f} "
          f"{row['MAPE']:>7.2f}% {row['RMSPE']:>7.2f}%")

print(f"""
Expected pattern: RMSPE should rise from Near to Extended — this is
NATURAL forecasting uncertainty, not a flaw. The important thing is
that this degradation is now smooth and honest, not an artificial
recursive-error spike.
""")


# =============================================================================
# STEP 4 — PER-INDIVIDUAL-HORIZON RMSPE  (Full Degradation Curve)
# =============================================================================

horizon_level_rows = []
for bucket, preds in bucket_predictions.items():
    df_b = pd.DataFrame({
        "horizon" : preds["horizon"],
        "actual"  : preds["actual"],
        "predicted": preds["predicted"],
    })
    for h, group in df_b.groupby("horizon"):
        m = compute_all_metrics(group["actual"].values, group["predicted"].values)
        horizon_level_rows.append({"horizon": h, "bucket": bucket, **m})

horizon_level_df = pd.DataFrame(horizon_level_rows).sort_values("horizon")

print("=" * 60)
print("STEP 4 — RMSPE BY INDIVIDUAL HORIZON")
print("=" * 60)
print(horizon_level_df[["horizon","bucket","RMSPE"]].to_string(index=False))


# =============================================================================
# VISUALISATION 1 — RMSPE Degradation Curve Across All Horizons
# =============================================================================

fig, ax = plt.subplots(figsize=(13, 6))

for bucket in HORIZON_BUCKETS.keys():
    sub = horizon_level_df[horizon_level_df["bucket"] == bucket]
    ax.plot(sub["horizon"], sub["RMSPE"], marker="o", markersize=7,
            color=BUCKET_COLORS[bucket], lw=2, label=bucket)

ax.set_title("Forecast Accuracy vs Horizon — RMSPE Degradation Curve",
             fontsize=13, color="#e0f0ff", pad=12)
ax.set_xlabel("Horizon (days ahead)")
ax.set_ylabel("RMSPE (%)")
ax.legend(fontsize=10, title="Bucket")
ax.grid(True)

plt.tight_layout()
plt.savefig("horizon_rmspe_degradation.png", dpi=150, bbox_inches="tight")
plt.show()


# =============================================================================
# VISUALISATION 2 — Feature Importance: Near vs Extended
# =============================================================================
#
# This validates the core hypothesis behind building 4 separate
# models — lag/rolling features should dominate for Near, while
# calendar/Promo features should dominate for Extended.
# =============================================================================

fig, axes = plt.subplots(1, 2, figsize=(16, 7))
fig.suptitle("Feature Importance — Near vs Extended Bucket",
             fontsize=13, color="#e0f0ff", y=1.02)

near_top = bucket_feature_imp["near"].head(12).sort_values()
ext_top  = bucket_feature_imp["extended"].head(12).sort_values()

axes[0].barh(near_top.index, near_top.values, color=ACCENT3, edgecolor="#0f1117")
axes[0].set_title("Near (1-14 days)", fontsize=11, color="#c0d8e8")
axes[0].grid(True, axis="x")

axes[1].barh(ext_top.index, ext_top.values, color=ACCENT5, edgecolor="#0f1117")
axes[1].set_title("Extended (61-90 days)", fontsize=11, color="#c0d8e8")
axes[1].grid(True, axis="x")

plt.tight_layout()
plt.savefig("horizon_feature_importance_comparison.png", dpi=150, bbox_inches="tight")
plt.show()


# =============================================================================
# VISUALISATION 3 — Bucket Metrics Comparison
# =============================================================================

fig, axes = plt.subplots(1, 4, figsize=(18, 5))
fig.suptitle("Metrics by Bucket", fontsize=13, color="#e0f0ff", y=1.02)

buckets_order = ["near","mid","far","extended"]
colors_order  = [BUCKET_COLORS[b] for b in buckets_order]

for i, metric in enumerate(["MAE","RMSE","MAPE","RMSPE"]):
    vals = [bucket_metrics[b][metric] for b in buckets_order]
    bars = axes[i].bar(buckets_order, vals, color=colors_order, edgecolor="#0f1117")
    for bar, val in zip(bars, vals):
        axes[i].text(bar.get_x()+bar.get_width()/2, bar.get_height()+max(vals)*0.01,
                    f"{val:.1f}", ha="center", fontsize=8, color="#c0d8e8")
    axes[i].set_title(metric, fontsize=11, color="#c0d8e8")
    axes[i].grid(True, axis="y")

plt.tight_layout()
plt.savefig("horizon_bucket_metrics.png", dpi=150, bbox_inches="tight")
plt.show()


# =============================================================================
# FINAL SUMMARY
# =============================================================================

print(f"""
================================================================
       MULTI-HORIZON MODELLING — DONE
================================================================
  4 independently tuned XGBoost models, one per bucket:
""")
for bucket in buckets_order:
    print(f"    {bucket:<10} RMSPE: {bucket_metrics[bucket]['RMSPE']:.2f}%  "
          f"(horizons {HORIZON_BUCKETS[bucket]})")

print(f"""
  Saved
    artifacts/xgb_horizon_near.pkl
    artifacts/xgb_horizon_mid.pkl
    artifacts/xgb_horizon_far.pkl
    artifacts/xgb_horizon_extended.pkl
    artifacts/horizon_best_params.pkl

  Next step -> API update
  (route requests to the correct bucket model based on
   how many days ahead the request is for)
================================================================
""")
