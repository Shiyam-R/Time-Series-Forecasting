# =============================================================================
# ROSSMANN STORE SALES — HORIZON DIAGNOSTIC
# File   : notebook/09_horizon_diagnostic.py
# Input  : data/processed/test_horizon.csv
#          artifacts/xgb_horizon_*.pkl
#          artifacts/horizon_all_features.pkl
# =============================================================================
#
# PURPOSE:
#   Investigate why the Near bucket (RMSPE 22.68%) performed WORSE
#   than Extended (RMSPE 14.91%) — the opposite of normal forecasting
#   behaviour, where accuracy should degrade as horizon grows.
#
# HYPOTHESIS:
#   The Near bucket's test target dates are squeezed into a narrow
#   13-day window (Apr 1-14, 2015) by the leakage-prevention boundary
#   logic, and that window happens to directly contain Easter 2015
#   (Good Friday Apr 3, Easter Sunday Apr 5, Easter Monday Apr 6).
#   Far/Extended buckets span 58 days, diluting the same holiday's
#   relative impact on their overall error.
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
BUCKET_COLORS = {"near": ACCENT3, "mid": ACCENT4, "far": ACCENT2, "extended": ACCENT5}

EASTER_START = pd.Timestamp("2015-04-03")   # Good Friday
EASTER_END   = pd.Timestamp("2015-04-06")   # Easter Monday


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

    # Guard against empty subsets — e.g. the Extended bucket's target
    # dates (May-June) never overlap the Easter window at all, so
    # "Easter window" metrics for that bucket have zero rows by
    # construction. That's expected, not an error — report it as N/A.
    if len(y_true_arr) == 0:
        return {"MAE": np.nan, "RMSE": np.nan, "MAPE": np.nan,
                "RMSPE": np.nan, "n": 0}

    return {
        "MAE"  : round(mean_absolute_error(y_true_arr, y_pred_arr), 2),
        "RMSE" : round(np.sqrt(mean_squared_error(y_true_arr, y_pred_arr)), 2),
        "MAPE" : round(mape(y_true_arr, y_pred_arr), 2),
        "RMSPE": round(rmspe(y_true_arr, y_pred_arr), 2),
        "n"    : len(y_true_arr),
    }


# =============================================================================
# STEP 0 — LOAD TEST DATA AND ALL 4 MODELS
# =============================================================================

test = pd.read_csv("data/processed/test_horizon.csv", parse_dates=["Date","target_date"])
ALL_FEATURES = joblib.load("artifacts/horizon_all_features.pkl")

models = {
    "near"     : joblib.load("artifacts/xgb_horizon_near.pkl"),
    "mid"      : joblib.load("artifacts/xgb_horizon_mid.pkl"),
    "far"      : joblib.load("artifacts/xgb_horizon_far.pkl"),
    "extended" : joblib.load("artifacts/xgb_horizon_extended.pkl"),
}

print("=" * 60)
print("STEP 0 — DATA AND MODELS LOADED")
print("=" * 60)
print(f"Test pairs: {len(test):,}\n")


# =============================================================================
# STEP 1 — TARGET DATE RANGE PER BUCKET
# =============================================================================

print("=" * 60)
print("STEP 1 — TARGET DATE RANGE PER BUCKET")
print("=" * 60)

for bucket in ["near","mid","far","extended"]:
    sub = test[test["horizon_bucket"] == bucket]
    span_days = (sub["target_date"].max() - sub["target_date"].min()).days
    print(f"{bucket:<10} : {sub['target_date'].min().date()} -> "
          f"{sub['target_date'].max().date()}  (span: {span_days} days, "
          f"{len(sub):,} rows)")


# =============================================================================
# STEP 2 — WHAT % OF EACH BUCKET'S TEST ROWS FALL IN THE EASTER WINDOW
# =============================================================================

print("\n" + "=" * 60)
print(f"STEP 2 — EASTER WINDOW OVERLAP ({EASTER_START.date()} to {EASTER_END.date()})")
print("=" * 60)

for bucket in ["near","mid","far","extended"]:
    sub = test[test["horizon_bucket"] == bucket]
    in_easter = sub["target_date"].between(EASTER_START, EASTER_END)
    pct = in_easter.mean() * 100
    print(f"{bucket:<10} : {in_easter.sum():>6,} / {len(sub):>6,} rows "
          f"land in Easter window  ({pct:5.1f}%)")


# =============================================================================
# STEP 3 — GENERATE PREDICTIONS FOR ALL BUCKETS
# =============================================================================

test["predicted"] = np.nan
for bucket, model in models.items():
    mask = test["horizon_bucket"] == bucket
    X = test.loc[mask, ALL_FEATURES]
    pred_log = model.predict(X)
    test.loc[mask, "predicted"] = np.clip(np.expm1(pred_log), 0, None)

print("\n" + "=" * 60)
print("STEP 3 — PREDICTIONS GENERATED FOR ALL BUCKETS")
print("=" * 60)


# =============================================================================
# STEP 4 — RMSPE: HOLIDAY ROWS vs NON-HOLIDAY ROWS, PER BUCKET
# =============================================================================
#
# This is the direct test of the hypothesis. If RMSPE is much higher
# specifically on holiday-window rows — and the Near bucket has a
# disproportionate SHARE of its rows in that window — that's the
# smoking gun, not "near-term forecasting is fundamentally harder."
# =============================================================================

print("\n" + "=" * 60)
print("STEP 4 — RMSPE: EASTER WINDOW vs NON-EASTER, PER BUCKET")
print("=" * 60)

diagnostic_rows = []
for bucket in ["near","mid","far","extended"]:
    sub = test[test["horizon_bucket"] == bucket].copy()
    in_easter = sub["target_date"].between(EASTER_START, EASTER_END)

    easter_metrics     = compute_all_metrics(sub.loc[in_easter, "target_Sales"], sub.loc[in_easter, "predicted"])
    non_easter_metrics = compute_all_metrics(sub.loc[~in_easter, "target_Sales"], sub.loc[~in_easter, "predicted"])
    overall_metrics    = compute_all_metrics(sub["target_Sales"], sub["predicted"])

    print(f"\n{bucket.upper()}")
    print(f"  Overall RMSPE      : {overall_metrics['RMSPE']:6.2f}%  (n={overall_metrics['n']:,})")
    print(f"  Easter window RMSPE: {easter_metrics['RMSPE']:6.2f}%  (n={easter_metrics['n']:,})")
    print(f"  Non-Easter RMSPE   : {non_easter_metrics['RMSPE']:6.2f}%  (n={non_easter_metrics['n']:,})")

    diagnostic_rows.append({"bucket": bucket, "type": "Easter window",     "RMSPE": easter_metrics["RMSPE"],     "n": easter_metrics["n"]})
    diagnostic_rows.append({"bucket": bucket, "type": "Non-Easter",        "RMSPE": non_easter_metrics["RMSPE"], "n": non_easter_metrics["n"]})
    diagnostic_rows.append({"bucket": bucket, "type": "Overall (as reported)", "RMSPE": overall_metrics["RMSPE"], "n": overall_metrics["n"]})

diag_df = pd.DataFrame(diagnostic_rows)


# =============================================================================
# STEP 5 — RMSPE BY INDIVIDUAL HORIZON, WITHIN NEAR BUCKET ONLY
# =============================================================================
#
# If h=1 and h=3 (closest to Easter) are the worst offenders within
# the near bucket itself, that further confirms the holiday is
# driving the result rather than horizon length itself.
# =============================================================================

print("\n" + "=" * 60)
print("STEP 5 — RMSPE BY HORIZON, NEAR BUCKET ONLY")
print("=" * 60)

near_sub = test[test["horizon_bucket"] == "near"]
for h in sorted(near_sub["horizon"].unique()):
    h_sub = near_sub[near_sub["horizon"] == h]
    m = compute_all_metrics(h_sub["target_Sales"], h_sub["predicted"])
    target_range = f"{h_sub['target_date'].min().date()} -> {h_sub['target_date'].max().date()}"
    print(f"  h={int(h):<3} RMSPE={m['RMSPE']:6.2f}%  n={m['n']:>5,}  target dates: {target_range}")


# =============================================================================
# VISUALISATION 1 — Target Date Histogram per Bucket, Easter Highlighted
# =============================================================================

fig, axes = plt.subplots(4, 1, figsize=(13, 12), sharex=True)
fig.suptitle("Target Date Distribution per Bucket — Easter Window Highlighted",
             fontsize=13, color="#e0f0ff", y=1.01)

for ax, bucket in zip(axes, ["near","mid","far","extended"]):
    sub = test[test["horizon_bucket"] == bucket]
    ax.hist(sub["target_date"], bins=60, color=BUCKET_COLORS[bucket],
            edgecolor="#0f1117", alpha=0.85)
    ax.axvspan(EASTER_START, EASTER_END, color=ACCENT2, alpha=0.25, label="Easter (Apr 3-6)")
    ax.set_title(f"{bucket} bucket", fontsize=10, color="#c0d8e8")
    ax.set_ylabel("Pairs")
    ax.legend(fontsize=8, loc="upper right")
    ax.grid(True)

axes[-1].set_xlabel("Target Date")
plt.tight_layout()
plt.savefig("diagnostic_target_date_distribution.png", dpi=150, bbox_inches="tight")
plt.show()


# =============================================================================
# VISUALISATION 2 — RMSPE: Easter vs Non-Easter, All Buckets
# =============================================================================

fig, ax = plt.subplots(figsize=(11, 6))

buckets_order = ["near","mid","far","extended"]
x = np.arange(len(buckets_order))
width = 0.25

easter_vals     = [diag_df[(diag_df.bucket==b)&(diag_df.type=="Easter window")]["RMSPE"].values[0] for b in buckets_order]
non_easter_vals = [diag_df[(diag_df.bucket==b)&(diag_df.type=="Non-Easter")]["RMSPE"].values[0] for b in buckets_order]
overall_vals    = [diag_df[(diag_df.bucket==b)&(diag_df.type=="Overall (as reported)")]["RMSPE"].values[0] for b in buckets_order]

# Keep the raw (possibly NaN) values for labeling, use 0-filled versions for bar heights
easter_bar_heights     = [0 if np.isnan(v) else v for v in easter_vals]
non_easter_bar_heights = [0 if np.isnan(v) else v for v in non_easter_vals]
overall_bar_heights    = [0 if np.isnan(v) else v for v in overall_vals]

ax.bar(x - width, easter_bar_heights,     width, color=ACCENT2, edgecolor="#0f1117", label="Easter window")
ax.bar(x,          non_easter_bar_heights, width, color=ACCENT3, edgecolor="#0f1117", label="Non-Easter")
ax.bar(x + width,  overall_bar_heights,    width, color=ACCENT,  edgecolor="#0f1117", label="Overall (originally reported)")

for i, (e, ne, o) in enumerate(zip(easter_vals, non_easter_vals, overall_vals)):
    if not np.isnan(e):
        ax.text(i-width, e+0.3, f"{e:.1f}", ha="center", fontsize=8, color="#c0d8e8")
    else:
        ax.text(i-width, 0.3, "N/A", ha="center", fontsize=8, color="#607080")
    if not np.isnan(ne):
        ax.text(i, ne+0.3, f"{ne:.1f}", ha="center", fontsize=8, color="#c0d8e8")
    if not np.isnan(o):
        ax.text(i+width, o+0.3, f"{o:.1f}", ha="center", fontsize=8, color="#c0d8e8")

ax.set_xticks(x)
ax.set_xticklabels(buckets_order)
ax.set_title("RMSPE — Easter Window vs Non-Easter Rows, by Bucket",
             fontsize=13, color="#e0f0ff", pad=12)
ax.set_ylabel("RMSPE (%)")
ax.legend(fontsize=9)
ax.grid(True, axis="y")

plt.tight_layout()
plt.savefig("diagnostic_easter_rmspe_comparison.png", dpi=150, bbox_inches="tight")
plt.show()


# =============================================================================
# FINAL SUMMARY
# =============================================================================

near_overall = diag_df[(diag_df.bucket=="near")&(diag_df.type=="Overall (as reported)")]["RMSPE"].values[0]
near_non_easter = diag_df[(diag_df.bucket=="near")&(diag_df.type=="Non-Easter")]["RMSPE"].values[0]
near_easter_pct = (test[(test.horizon_bucket=="near") & test["target_date"].between(EASTER_START,EASTER_END)].shape[0] /
                   test[test.horizon_bucket=="near"].shape[0] * 100)

print(f"""
================================================================
       DIAGNOSTIC CONCLUSION
================================================================
  Near bucket originally reported RMSPE : {near_overall:.2f}%
  Near bucket Non-Easter RMSPE only     : {near_non_easter:.2f}%
  Share of Near bucket inside Easter    : {near_easter_pct:.1f}% of all rows

  If Non-Easter RMSPE is substantially lower than the originally
  reported overall figure, the hypothesis is confirmed: the Near
  bucket's "worse than Extended" result was a TEST-SET ARTIFACT
  caused by its narrow 13-day window landing on Easter — not a
  genuine finding that near-term forecasting is harder than
  long-term forecasting.

  This does NOT mean the Near model is bad. It means the Near
  bucket's CURRENT test sample is not representative, and the
  reported 22.68% RMSPE should not be trusted as this model's
  true accuracy until re-evaluated on a holiday-free window too.
================================================================
""")