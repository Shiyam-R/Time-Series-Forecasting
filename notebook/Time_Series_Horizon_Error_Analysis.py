# =============================================================================
# ROSSMANN STORE SALES — SECTION 2: HORIZON-LEVEL ERROR ANALYSIS
# File   : notebook/11_horizon_error_analysis.py
# Input  : data/processed/test_horizon_with_baselines.csv
#          data/processed/baseline_comparison_results.csv
# =============================================================================
#
# WHAT THIS FIXES FROM EARLIER:
#   08_horizon_modelling.py and 09_horizon_diagnostic.py both grouped
#   by the "horizon" column loaded directly from test_horizon.csv —
#   but that column was StandardScaler-transformed in
#   07_horizon_dataset.py (it's also a model FEATURE, so it had to
#   be scaled for training). The bucket-level RMSPE numbers from
#   those scripts are still valid (bucket assignment used a string
#   label, untouched by scaling) — but the "RMSPE by individual
#   horizon" x-axis labels in those two scripts were wrong.
#
#   This script recomputes the REAL horizon directly from
#   (target_date - Date).dt.days — both are genuine timestamps,
#   never touched by any scaler — and uses that everywhere below.
#
# WHAT THIS ADDS BEYOND THE EARLIER VERSION:
#   - MAE and RMSE by exact horizon, not just RMSPE
#   - XGBoost, LightGBM, Linear Regression AND the best simple
#     baseline plotted together — shows whether XGBoost's edge over
#     simpler methods grows, shrinks, or disappears with horizon
#   - A direct re-check of the Easter-window effect on Near-bucket
#     horizons, now that days_since/until_holiday exists as a
#     feature — validates whether that fix actually helped
# =============================================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
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
ACCENT6 = "#ff9a3c"
ACCENT7 = "#5ad1e6"

EASTER_START = pd.Timestamp("2015-04-03")
EASTER_END   = pd.Timestamp("2015-04-06")


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
# STEP 0 — LOAD DATA AND RECOMPUTE THE REAL HORIZON
# =============================================================================

test = pd.read_csv(
    "data/processed/test_horizon_with_baselines.csv",
    parse_dates=["Date", "target_date"]
)
baseline_results = pd.read_csv("data/processed/baseline_comparison_results.csv")

# Recompute fresh — never trust the scaled "horizon" column for date math
test["horizon_actual"] = (test["target_date"] - test["Date"]).dt.days

print("=" * 60)
print("STEP 0 — DATA LOADED, REAL HORIZON RECOMPUTED")
print("=" * 60)
print(f"Test pairs    : {len(test):,}")
print(f"Horizon range : {test['horizon_actual'].min()} -> {test['horizon_actual'].max()} days")
print(f"Unique horizons sampled: {sorted(test['horizon_actual'].unique())}\n")

# Identify the best simple baseline overall (for comparison line)
simple_methods = ["Naive (weekly)","Naive (monthly)","Naive (yearly)",
                   "Moving Average","Historical Mean"]
overall_simple = baseline_results[baseline_results["method"].isin(simple_methods)]
best_simple_overall = overall_simple.groupby("method")["RMSPE"].mean().idxmin()
print(f"Best simple baseline overall (by mean RMSPE across buckets): {best_simple_overall}")

method_to_col = {
    "Naive (weekly)"  : "naive_weekly_pred",
    "Naive (monthly)" : "naive_monthly_pred",
    "Naive (yearly)"  : "naive_yearly_pred",
    "Moving Average"  : "moving_avg_pred",
    "Historical Mean" : "historical_mean_pred",
}
best_simple_col = method_to_col[best_simple_overall]


# =============================================================================
# STEP 1 — METRICS BY EXACT HORIZON, ALL 4 MODELS
# =============================================================================

models_to_compare = {
    "XGBoost"           : "xgb_pred",
    "LightGBM"          : "lgbm_pred",
    "Linear Regression" : "linear_pred",
    f"Best Baseline ({best_simple_overall})": best_simple_col,
}

print("=" * 60)
print("STEP 1 — METRICS BY EXACT HORIZON")
print("=" * 60)

horizon_results = []
for h in sorted(test["horizon_actual"].unique()):
    sub = test[test["horizon_actual"] == h]
    bucket = sub["horizon_bucket"].iloc[0]
    for model_name, col in models_to_compare.items():
        m = compute_all_metrics(sub["target_Sales"].values, sub[col].values)
        horizon_results.append({"horizon": h, "bucket": bucket, "model": model_name, **m})

horizon_df = pd.DataFrame(horizon_results)

# Print XGBoost row for every horizon — the production model's curve
xgb_view = horizon_df[horizon_df["model"] == "XGBoost"].sort_values("horizon")
print(f"\n{'Horizon':>8} {'Bucket':<10} {'MAE':>10} {'RMSE':>10} {'MAPE':>8} {'RMSPE':>8}")
print("-" * 60)
for _, row in xgb_view.iterrows():
    print(f"{int(row['horizon']):>8} {row['bucket']:<10} {row['MAE']:>10.2f} "
          f"{row['RMSE']:>10.2f} {row['MAPE']:>7.2f}% {row['RMSPE']:>7.2f}%")


# =============================================================================
# STEP 2 — EASIEST / HARDEST HORIZONS
# =============================================================================

print("\n" + "=" * 60)
print("STEP 2 — EASIEST / HARDEST HORIZONS (XGBoost, by RMSPE)")
print("=" * 60)

xgb_sorted = xgb_view.sort_values("RMSPE")
print("\nEasiest 5 horizons:")
print(xgb_sorted[["horizon","bucket","RMSPE"]].head(5).to_string(index=False))
print("\nHardest 5 horizons:")
print(xgb_sorted[["horizon","bucket","RMSPE"]].tail(5).to_string(index=False))


# =============================================================================
# STEP 3 — IS DEGRADATION SMOOTH? CORRELATION CHECK + RESIDUAL FROM TREND
# =============================================================================

print("\n" + "=" * 60)
print("STEP 3 — IS PERFORMANCE DEGRADATION SMOOTH?")
print("=" * 60)

corr = xgb_view["horizon"].corr(xgb_view["RMSPE"])
print(f"Correlation between horizon and RMSPE: {corr:.3f}")
print(f"  (closer to +1 = cleanly monotonic degradation, "
      f"closer to 0 = noisy/non-monotonic)")

# Fit a simple linear trend to RMSPE vs horizon, flag outlier horizons
z = np.polyfit(xgb_view["horizon"], xgb_view["RMSPE"], 1)
trend_line = np.poly1d(z)
xgb_view = xgb_view.copy()
xgb_view["trend_predicted"] = trend_line(xgb_view["horizon"])
xgb_view["deviation_from_trend"] = xgb_view["RMSPE"] - xgb_view["trend_predicted"]

print(f"\nHorizons deviating most from the smooth trend line (|deviation| > 3pp):")
outliers = xgb_view[xgb_view["deviation_from_trend"].abs() > 3].sort_values(
    "deviation_from_trend", ascending=False)
if len(outliers) > 0:
    print(outliers[["horizon","bucket","RMSPE","trend_predicted","deviation_from_trend"]].to_string(index=False))
else:
    print("  None — degradation is smooth with no horizon standing out.")


# =============================================================================
# STEP 4 — DID THE HOLIDAY-DISTANCE FIX ACTUALLY HELP? (Easter re-check)
# =============================================================================

print("\n" + "=" * 60)
print("STEP 4 — EASTER WINDOW RE-CHECK (post holiday-distance feature)")
print("=" * 60)

near_horizons = sorted(test[test["horizon_bucket"] == "near"]["horizon_actual"].unique())
for h in near_horizons:
    sub = test[test["horizon_actual"] == h]
    in_easter = sub["target_date"].between(EASTER_START, EASTER_END)

    overall_m = compute_all_metrics(sub["target_Sales"].values, sub["xgb_pred"].values)
    if in_easter.sum() > 0:
        easter_m = compute_all_metrics(sub.loc[in_easter,"target_Sales"].values, sub.loc[in_easter,"xgb_pred"].values)
        print(f"  h={h:<3} overall RMSPE={overall_m['RMSPE']:6.2f}%  "
              f"Easter rows RMSPE={easter_m['RMSPE']:6.2f}%  (n_easter={easter_m['n']})")
    else:
        print(f"  h={h:<3} overall RMSPE={overall_m['RMSPE']:6.2f}%  (no Easter rows at this horizon)")

print(f"""
For reference, BEFORE the days_since/until_holiday feature was added,
the near bucket's Easter-window RMSPE was 36.74% (see
09_horizon_diagnostic.py output). Compare that to the Easter-row
RMSPE values above — a drop indicates the new feature genuinely
helped the model anticipate the holiday rather than just react to
the exact holiday date.
""")


# =============================================================================
# VISUALISATION 1 — Degradation Curves, 3 Metrics, 4 Models
# =============================================================================

fig, axes = plt.subplots(3, 1, figsize=(14, 14), sharex=True)
fig.suptitle("Forecast Accuracy vs Exact Horizon — RMSPE / MAE / RMSE",
             fontsize=13, color="#e0f0ff", y=1.01)

model_colors = {
    "XGBoost": ACCENT3, "LightGBM": ACCENT6,
    "Linear Regression": ACCENT7,
    f"Best Baseline ({best_simple_overall})": ACCENT2,
}

for ax, metric in zip(axes, ["RMSPE","MAE","RMSE"]):
    for model_name in models_to_compare.keys():
        sub = horizon_df[horizon_df["model"] == model_name].sort_values("horizon")
        ax.plot(sub["horizon"], sub[metric], marker="o", markersize=5,
                color=model_colors[model_name], lw=1.8, label=model_name)
    ax.set_ylabel(metric + (" (%)" if metric in ["RMSPE","MAPE"] else " (€)"))
    ax.legend(fontsize=8, loc="upper left")
    ax.grid(True)

axes[-1].set_xlabel("Horizon (days ahead)")
plt.tight_layout()
plt.savefig("horizon_error_degradation_curves.png", dpi=150, bbox_inches="tight")
plt.show()


# =============================================================================
# VISUALISATION 2 — XGBoost's Edge Over Best Baseline, by Horizon
# =============================================================================

fig, ax = plt.subplots(figsize=(13, 6))

xgb_sub = horizon_df[horizon_df["model"] == "XGBoost"].sort_values("horizon")
baseline_sub = horizon_df[horizon_df["model"] == f"Best Baseline ({best_simple_overall})"].sort_values("horizon")

edge = baseline_sub["RMSPE"].values - xgb_sub["RMSPE"].values
colors = [ACCENT3 if e > 0 else ACCENT2 for e in edge]

bars = ax.bar(xgb_sub["horizon"].astype(str), edge, color=colors, edgecolor="#0f1117")
ax.axhline(0, color="white", lw=1, alpha=0.5)
ax.set_title(f"XGBoost's Advantage Over Best Baseline ({best_simple_overall}), by Horizon",
             fontsize=13, color="#e0f0ff", pad=12)
ax.set_xlabel("Horizon (days ahead)")
ax.set_ylabel("RMSPE improvement (pp) — positive = XGBoost wins")
ax.tick_params(axis="x", rotation=45)
ax.grid(True, axis="y")

plt.tight_layout()
plt.savefig("horizon_xgb_edge_over_baseline.png", dpi=150, bbox_inches="tight")
plt.show()


# =============================================================================
# VISUALISATION 3 — Trend Line with Deviation Highlighted
# =============================================================================

fig, ax = plt.subplots(figsize=(13, 6))

ax.scatter(xgb_view["horizon"], xgb_view["RMSPE"], color=ACCENT3, s=70, zorder=5, label="Actual RMSPE")
x_line = np.linspace(xgb_view["horizon"].min(), xgb_view["horizon"].max(), 100)
ax.plot(x_line, trend_line(x_line), color="white", lw=1.5, linestyle="--", alpha=0.6, label="Linear trend")

for _, row in xgb_view.iterrows():
    if abs(row["deviation_from_trend"]) > 3:
        ax.annotate(f"h={int(row['horizon'])}", xy=(row["horizon"], row["RMSPE"]),
                    xytext=(0,10), textcoords="offset points", fontsize=8,
                    color=ACCENT2, ha="center")
        ax.scatter([row["horizon"]], [row["RMSPE"]], color=ACCENT2, s=100, zorder=6)

ax.set_title(f"RMSPE vs Horizon — Trend Line  (correlation r={corr:.2f})",
             fontsize=13, color="#e0f0ff", pad=12)
ax.set_xlabel("Horizon (days ahead)")
ax.set_ylabel("RMSPE (%)")
ax.legend(fontsize=9)
ax.grid(True)

plt.tight_layout()
plt.savefig("horizon_trend_and_outliers.png", dpi=150, bbox_inches="tight")
plt.show()


# =============================================================================
# SAVE RESULTS
# =============================================================================

horizon_df.to_csv("data/processed/horizon_level_error_results.csv", index=False)

print(f"""
================================================================
       SECTION 2 — HORIZON-LEVEL ERROR ANALYSIS — DONE
================================================================
  Fixed: real horizon recomputed from dates, not the scaled
  "horizon" column — earlier 08/09 per-horizon x-axis labels
  are now corrected here.

  Degradation correlation (horizon vs RMSPE): {corr:.3f}

  Saved
    data/processed/horizon_level_error_results.csv

  Plots
    horizon_error_degradation_curves.png
    horizon_xgb_edge_over_baseline.png
    horizon_trend_and_outliers.png

  Next -> Section 3: Store-Level Error Analysis
================================================================
""")
