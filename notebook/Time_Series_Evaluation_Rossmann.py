# =============================================================================
# ROSSMANN STORE SALES — EVALUATION
# File   : notebook/05_evaluation.py
# Input  : data/processed/test.csv
#          artifacts/linear_model.pkl
#          artifacts/xgb_model.pkl
#          artifacts/lgbm_model.pkl
#          artifacts/all_features.pkl
# =============================================================================
#
# METRICS:
#   MAE   — average error in € units
#   RMSE  — penalises large errors more heavily
#   MAPE  — percentage error, scale free
#   RMSPE — Root Mean Squared Percentage Error (official Kaggle metric
#           for this competition) — zero-actual rows excluded
#
# ANALYSIS LEVELS:
#   1. Global  — one number per model
#   2. Per-store — does the model work equally well across all 1115 stores?
#   3. By segment — StoreType, Promo, DayOfWeek breakdowns
#   4. Worst stores — bottom 10 stores by error, evidence for tuning need
# =============================================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import joblib
import warnings
warnings.filterwarnings("ignore")

from sklearn.metrics import mean_absolute_error, mean_squared_error

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
# STEP 0 — LOAD DATA, MODELS, GENERATE PREDICTIONS
# =============================================================================

test = pd.read_csv("data/processed/test.csv", parse_dates=["Date"])
ALL_FEATURES = joblib.load("artifacts/all_features.pkl")

linear = joblib.load("artifacts/linear_model.pkl")
xgb    = joblib.load("artifacts/xgb_model.pkl")
lgbm   = joblib.load("artifacts/lgbm_model.pkl")

X_test = test[ALL_FEATURES]
y_test_raw = test["Sales"]

# Generate predictions — reverse log1p transform for all models
linear_pred = np.expm1(linear.predict(X_test))
xgb_pred    = np.expm1(xgb.predict(X_test))
lgbm_pred   = np.expm1(lgbm.predict(X_test))

# Clip negative predictions to 0 — log1p reversal can occasionally
# produce small negative values for very low sales days
linear_pred = np.clip(linear_pred, 0, None)
xgb_pred    = np.clip(xgb_pred, 0, None)
lgbm_pred   = np.clip(lgbm_pred, 0, None)

pred_df = test[["Date","Store","DayOfWeek","Promo","StoreType_enc"]].copy()
pred_df["Actual"]   = y_test_raw.values
pred_df["Linear"]   = linear_pred
pred_df["XGBoost"]  = xgb_pred
pred_df["LightGBM"] = lgbm_pred

print("=" * 55)
print("STEP 0 — PREDICTIONS GENERATED")
print("=" * 55)
print(f"Test rows : {len(test):,}")
print(f"Stores    : {test['Store'].nunique()}")
print(f"\nSample predictions:")
print(pred_df[["Date","Store","Actual","Linear","XGBoost","LightGBM"]].head(5).to_string())


# =============================================================================
# STEP 1 — METRIC FUNCTIONS
# =============================================================================
#
# RMSPE excludes rows where Actual=0 to avoid division by zero.
# Even though Open=0 rows were removed in feature engineering,
# Rossmann data has known rows where Open=1 but Sales=0 due to
# data quality issues — these must be excluded from RMSPE only.
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
    y_true_arr = np.asarray(y_true)
    y_pred_arr = np.asarray(y_pred)
    mae   = mean_absolute_error(y_true_arr, y_pred_arr)
    rmse  = np.sqrt(mean_squared_error(y_true_arr, y_pred_arr))
    mp    = mape(y_true_arr, y_pred_arr)
    rmspe_val = rmspe(y_true_arr, y_pred_arr)
    return {
        "Model": model_name, "MAE": round(mae, 2), "RMSE": round(rmse, 2),
        "MAPE": round(mp, 2), "RMSPE": round(rmspe_val, 2)
    }

zero_sales_rows = (pred_df["Actual"] == 0).sum()
print("\n" + "=" * 55)
print("STEP 1 — METRIC SETUP")
print("=" * 55)
print(f"Rows with Actual Sales = 0 (excluded from RMSPE/MAPE only): {zero_sales_rows}")


# =============================================================================
# STEP 2 — GLOBAL METRICS
# =============================================================================

results = [
    compute_all_metrics(pred_df["Actual"], pred_df["Linear"],   "Linear Regression"),
    compute_all_metrics(pred_df["Actual"], pred_df["XGBoost"],  "XGBoost"),
    compute_all_metrics(pred_df["Actual"], pred_df["LightGBM"], "LightGBM"),
]
metrics_df = pd.DataFrame(results).set_index("Model")

print("\n" + "=" * 55)
print("STEP 2 — GLOBAL METRICS (Test Set: Last 2 Months)")
print("=" * 55)
print(f"\n{'Model':<20} {'MAE':>10} {'RMSE':>10} {'MAPE':>8} {'RMSPE':>8}")
print("─" * 60)

best_rmspe = metrics_df["RMSPE"].idxmin()
for model, row in metrics_df.iterrows():
    mark = " ← WINNER" if model == best_rmspe else ""
    print(f"{model:<20} {row['MAE']:>10.2f} {row['RMSE']:>10.2f} "
          f"{row['MAPE']:>7.2f}% {row['RMSPE']:>7.2f}%{mark}")

print(f"""
Ranked by RMSPE — the official Kaggle metric for this competition.
RMSPE squares percentage errors before averaging, so it penalises
large relative misses (e.g. predicting €1000 when actual is €100)
more heavily than MAPE does.
""")


# =============================================================================
# STEP 3 — PER-STORE METRICS  (Winning Model Only)
# =============================================================================

winner_col = best_rmspe
print("=" * 55)
print(f"STEP 3 — PER-STORE METRICS  (Model: {winner_col})")
print("=" * 55)

def per_store_metrics(df, model_col):
    rows = []
    for store_id, group in df.groupby("Store"):
        m = compute_all_metrics(group["Actual"], group[model_col], store_id)
        rows.append(m)
    return pd.DataFrame(rows).rename(columns={"Model": "Store"}).set_index("Store")

store_metrics = per_store_metrics(pred_df, winner_col)

print(f"\nPer-store RMSPE distribution ({winner_col}):")
print(store_metrics["RMSPE"].describe().round(2).to_string())

print(f"\nStores with RMSPE > 20% (potentially problematic): "
      f"{(store_metrics['RMSPE'] > 20).sum()} out of {len(store_metrics)}")


# =============================================================================
# STEP 4 — ERROR BREAKDOWN BY SEGMENT
# =============================================================================

pred_df["abs_pct_error"] = np.where(
    pred_df["Actual"] != 0,
    np.abs(pred_df["Actual"] - pred_df[winner_col]) / pred_df["Actual"] * 100,
    np.nan
)

print("\n" + "=" * 55)
print(f"STEP 4 — ERROR BREAKDOWN BY SEGMENT  (Model: {winner_col})")
print("=" * 55)

# By StoreType
storetype_err = pred_df.groupby("StoreType_enc")["abs_pct_error"].mean().round(2)
print(f"\nMean Absolute % Error by StoreType:")
print(storetype_err.to_string())

# By Promo
promo_err = pred_df.groupby("Promo")["abs_pct_error"].mean().round(2)
print(f"\nMean Absolute % Error by Promo:")
print(promo_err.to_string())

# By DayOfWeek
dow_err = pred_df.groupby("DayOfWeek")["abs_pct_error"].mean().round(2)
print(f"\nMean Absolute % Error by DayOfWeek:")
print(dow_err.to_string())


# =============================================================================
# STEP 5 — WORST PERFORMING STORES
# =============================================================================

worst_stores = store_metrics.sort_values("RMSPE", ascending=False).head(10)

print("\n" + "=" * 55)
print(f"STEP 5 — TOP 10 WORST PERFORMING STORES  (Model: {winner_col})")
print("=" * 55)
print(worst_stores.to_string())

best_stores = store_metrics.sort_values("RMSPE", ascending=True).head(5)
print(f"\nTop 5 BEST performing stores (for comparison):")
print(best_stores.to_string())


# =============================================================================
# VISUALISATION 1 — Global Metrics Comparison
# =============================================================================

fig, axes = plt.subplots(1, 4, figsize=(18, 5))
fig.suptitle("Global Model Comparison — All 4 Metrics",
             fontsize=13, color="#e0f0ff", y=1.02)

models  = metrics_df.index.tolist()
colors  = [ACCENT2, ACCENT, ACCENT3]
metric_names = ["MAE", "RMSE", "MAPE", "RMSPE"]

for i, metric in enumerate(metric_names):
    vals = metrics_df[metric].values
    bars = axes[i].bar(models, vals, color=colors, edgecolor="#0f1117", width=0.5)
    best_idx = np.argmin(vals)
    bars[best_idx].set_edgecolor("white")
    bars[best_idx].set_linewidth(2)
    for bar, val in zip(bars, vals):
        axes[i].text(bar.get_x()+bar.get_width()/2, bar.get_height()+max(vals)*0.01,
                    f"{val:.1f}", ha="center", fontsize=8, color="#c0d8e8")
    axes[i].set_title(metric, fontsize=11, color="#c0d8e8")
    axes[i].tick_params(axis="x", rotation=20)
    axes[i].grid(True, axis="y")

plt.tight_layout()
plt.savefig("eval_global_metrics.png", dpi=150, bbox_inches="tight")
plt.show()


# =============================================================================
# VISUALISATION 2 — Per-Store RMSPE Distribution
# =============================================================================

fig, axes = plt.subplots(1, 2, figsize=(15, 5))
fig.suptitle(f"Per-Store RMSPE Distribution — {winner_col}",
             fontsize=13, color="#e0f0ff", y=1.02)

axes[0].hist(store_metrics["RMSPE"], bins=40,
             color=ACCENT3, edgecolor="#0f1117", alpha=0.85)
axes[0].axvline(store_metrics["RMSPE"].median(), color=ACCENT4,
                lw=2, linestyle="--",
                label=f"Median: {store_metrics['RMSPE'].median():.1f}%")
axes[0].axvline(store_metrics["RMSPE"].mean(), color=ACCENT2,
                lw=2, linestyle="--",
                label=f"Mean: {store_metrics['RMSPE'].mean():.1f}%")
axes[0].set_title("Distribution Across 1115 Stores", fontsize=10, color="#c0d8e8")
axes[0].set_xlabel("RMSPE (%)")
axes[0].set_ylabel("Number of Stores")
axes[0].legend(fontsize=9)
axes[0].grid(True)

axes[1].boxplot(store_metrics["RMSPE"], vert=True, patch_artist=True,
                boxprops=dict(facecolor=ACCENT3+"44", color=ACCENT3),
                medianprops=dict(color=ACCENT4, linewidth=2),
                flierprops=dict(markerfacecolor=ACCENT2, markersize=4, alpha=0.5))
axes[1].set_title("RMSPE Box Plot (outlier stores visible)",
                  fontsize=10, color="#c0d8e8")
axes[1].set_ylabel("RMSPE (%)")
axes[1].grid(True, axis="y")

plt.tight_layout()
plt.savefig("eval_per_store_distribution.png", dpi=150, bbox_inches="tight")
plt.show()


# =============================================================================
# VISUALISATION 3 — Error Breakdown by Segment
# =============================================================================

fig, axes = plt.subplots(1, 3, figsize=(16, 5))
fig.suptitle(f"Error Breakdown by Segment — {winner_col}",
             fontsize=13, color="#e0f0ff", y=1.02)

storetype_labels = {0:"Type a", 1:"Type b", 2:"Type c", 3:"Type d"}
axes[0].bar([storetype_labels[i] for i in storetype_err.index],
            storetype_err.values, color=ACCENT, edgecolor="#0f1117")
axes[0].set_title("By Store Type", fontsize=10, color="#c0d8e8")
axes[0].set_ylabel("Mean Absolute % Error")
axes[0].grid(True, axis="y")

axes[1].bar(["No Promo","Promo"], promo_err.values,
            color=[ACCENT2, ACCENT3], edgecolor="#0f1117", width=0.5)
axes[1].set_title("By Promo", fontsize=10, color="#c0d8e8")
axes[1].set_ylabel("Mean Absolute % Error")
axes[1].grid(True, axis="y")

dow_labels = ["Mon","Tue","Wed","Thu","Fri","Sat","Sun"]
axes[2].bar(dow_labels, dow_err.values,
            color=ACCENT5, edgecolor="#0f1117")
axes[2].set_title("By Day of Week", fontsize=10, color="#c0d8e8")
axes[2].set_ylabel("Mean Absolute % Error")
axes[2].grid(True, axis="y")

plt.tight_layout()
plt.savefig("eval_error_by_segment.png", dpi=150, bbox_inches="tight")
plt.show()


# =============================================================================
# VISUALISATION 4 — Worst 10 Stores
# =============================================================================

fig, ax = plt.subplots(figsize=(12, 6))
bars = ax.barh(
    [f"Store {s}" for s in worst_stores.index[::-1]],
    worst_stores["RMSPE"].values[::-1],
    color=ACCENT2, edgecolor="#0f1117"
)
ax.axvline(store_metrics["RMSPE"].median(), color="white", lw=1,
           linestyle="--", alpha=0.5,
           label=f"Overall median: {store_metrics['RMSPE'].median():.1f}%")
ax.set_title(f"Top 10 Worst Performing Stores — {winner_col}",
             fontsize=13, color="#e0f0ff", pad=12)
ax.set_xlabel("RMSPE (%)")
ax.legend(fontsize=9)
ax.grid(True, axis="x")

plt.tight_layout()
plt.savefig("eval_worst_stores.png", dpi=150, bbox_inches="tight")
plt.show()


# =============================================================================
# STEP 6 — TUNING RECOMMENDATION
# =============================================================================

global_rmspe = metrics_df.loc[winner_col, "RMSPE"]
pct_problematic = (store_metrics["RMSPE"] > 20).sum() / len(store_metrics) * 100

print("\n" + "=" * 55)
print("STEP 6 — TUNING RECOMMENDATION")
print("=" * 55)
print(f"\nWinning model        : {winner_col}")
print(f"Global RMSPE          : {global_rmspe:.2f}%")
print(f"Stores with RMSPE>20% : {pct_problematic:.1f}% of all stores")

if global_rmspe < 15 and pct_problematic < 10:
    verdict = "Performance looks strong. Deep hyperparameter tuning may yield only marginal gains."
elif global_rmspe < 20:
    verdict = "Performance is reasonable. Deep tuning could help, focus on worst-performing stores/segments first."
else:
    verdict = "Performance has room to improve. Deep hyperparameter tuning is recommended."

print(f"\nVerdict: {verdict}")

print(f"""
Reference — Kaggle Rossmann competition winning entries achieved
RMSPE around 10-12%. Use this as a rough benchmark.
""")

print("""
╔═════════════════════════════════════════════════════╗
║         EVALUATION — DONE                            ║
╠═════════════════════════════════════════════════════╣
║  Global metrics  : MAE, RMSE, MAPE, RMSPE             ║
║  Per-store       : RMSPE distribution across 1115     ║
║                    stores                              ║
║  Segment errors  : StoreType, Promo, DayOfWeek         ║
║  Worst stores    : Top 10 identified                  ║
║                                                        ║
║  Plots saved                                          ║
║    eval_global_metrics.png                            ║
║    eval_per_store_distribution.png                    ║
║    eval_error_by_segment.png                           ║
║    eval_worst_stores.png                               ║
║                                                        ║
║  Next → decide tuning scope based on verdict above     ║
╚═════════════════════════════════════════════════════╝
""")
