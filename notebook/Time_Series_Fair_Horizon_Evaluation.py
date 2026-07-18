# =============================================================================
# ROSSMANN STORE SALES — FAIR MULTI-ORIGIN ROLLING-ORIGIN BACKTEST
# File   : notebook/12_fair_horizon_evaluation.py
# Input  : data/processed/features.csv
# Output : Separate "_fair_eval" artifacts — does NOT touch the
#          production API's models (xgb_horizon_*.pkl, scaler_horizon.pkl,
#          last_known_per_store.pkl remain untouched)
# =============================================================================
#
# WHY A SEPARATE TRACK:
#   This is an EXPERIMENT to answer "is 90 days actually needed", not
#   a production deployment. Everything here is saved under distinct
#   *_fair_eval filenames so the live API is never silently affected
#   by this analysis run.
#
# THE PROBLEM BEING FIXED:
#   The original single-origin test window meant each bucket's target
#   dates landed in DIFFERENT, non-overlapping parts of the calendar
#   (Near: only Apr 1-14, hit Easter hard. Extended: May-June, missed
#   April's holidays entirely). That confounded "horizon length" with
#   "which holidays this bucket happened to land near" — an unfair
#   comparison.
#
# THE FIX — ROLLING-ORIGIN BACKTESTING:
#   Instead of ONE anchor window, use MANY anchor points ("origins")
#   spread every 7 days across a FULL YEAR of test data. For every
#   origin, generate near/mid/far/extended forecasts. Aggregate each
#   bucket's metrics across ALL origins — so every bucket now sees
#   the SAME mix of seasons, holidays, and ordinary weeks. This is
#   the same technique production forecasting systems (e.g. Amazon)
#   use to validate models before shipping.
#
# WHY USING ORIGINS DEEP INTO THE "TEST" PERIOD IS NOT LEAKAGE:
#   The model's PARAMETERS are fixed once, trained only on data
#   through train_boundary. Using a later origin (e.g. March 2015)
#   just means feeding the SAME static model fresh, real INPUT
#   features computed from real historical sales at that point — no
#   different from how this model would actually be used in
#   production as time moves forward without retraining. What
#   actually matters for leakage is the TARGET being forecast: every
#   target date here is strictly after train_boundary, so no target
#   value was ever seen by the model during training.

#  Things to do next - Create month bucket and see rmpse for each month, RMSPE by horizon and Holiday vs non-holiday
# =============================================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import joblib
import os
import warnings
warnings.filterwarnings("ignore")

from sklearn.preprocessing import StandardScaler
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
BUCKET_ORDER = ["near","mid","far","extended"]

OLD_BUCKET_RMSPE = {"near": 22.68, "mid": 18.79, "far": 15.81, "extended": 14.91}


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
# STEP 0 — LOAD DATA, DEFINE FEATURE ROLES  (same as 07_horizon_dataset.py)
# =============================================================================

df = pd.read_csv("data/processed/features.csv", parse_dates=["Date"])
df.sort_values(["Store", "Date"], inplace=True)
df.reset_index(drop=True, inplace=True)

ANCHOR_CONTINUOUS = [
    "lag_7", "lag_14", "lag_30",
    "rolling_mean_7", "rolling_std_7",
    "rolling_mean_30", "rolling_std_30",
    "CompetitionDistance",
]
ANCHOR_STATIC = [
    "StoreType_enc", "Assortment_enc",
    "CompetitionDistance_missing", "Promo2",
]
TARGET_DAY_FEATURES = [
    "DayOfWeek", "Month", "Year", "WeekOfMonth",
    "IsWeekend", "IsMonthStart", "IsMonthEnd",
    "dow_sin", "dow_cos", "month_sin", "month_cos",
    "Promo", "StateHoliday_enc", "SchoolHoliday", "Promo_DayOfWeek",
    "days_since_holiday", "days_until_holiday",
]
CONTINUOUS_FEATURES = ANCHOR_CONTINUOUS + ["horizon", "days_since_holiday", "days_until_holiday"]
CATEGORICAL_BINARY_FEATURES = ANCHOR_STATIC + [
    "DayOfWeek", "Month", "Year", "WeekOfMonth",
    "IsWeekend", "IsMonthStart", "IsMonthEnd",
    "Promo", "StateHoliday_enc", "SchoolHoliday", "Promo_DayOfWeek",
]
CYCLICAL_FEATURES = ["dow_sin", "dow_cos", "month_sin", "month_cos"]
ALL_FEATURES = CONTINUOUS_FEATURES + [
    f for f in CATEGORICAL_BINARY_FEATURES if f not in ANCHOR_STATIC
] + ANCHOR_STATIC + CYCLICAL_FEATURES
ALL_FEATURES = list(dict.fromkeys(ALL_FEATURES))

HORIZON_BUCKETS = {
    "near"     : [1, 3, 7, 10, 14],
    "mid"      : [15, 18, 21, 25, 30],
    "far"      : [31, 38, 45, 52, 60],
    "extended" : [61, 70, 80, 90],
}
ALL_HORIZONS = sorted(sum(HORIZON_BUCKETS.values(), []))

def horizon_to_bucket(h):
    for bucket, horizons in HORIZON_BUCKETS.items():
        if h in horizons:
            return bucket
    return "unknown"

print("=" * 60)
print("STEP 0 — DATA LOADED")
print("=" * 60)
print(f"Shape: {df.shape}  Range: {df['Date'].min().date()} -> {df['Date'].max().date()}\n")


# =============================================================================
# STEP 1 — NEW BOUNDARY: FULL-YEAR TEST PERIOD
# =============================================================================

max_date = df["Date"].max()
NEW_TRAIN_BOUNDARY = max_date - pd.DateOffset(months=12)   # 2014-07-31
NEW_TEST_START      = NEW_TRAIN_BOUNDARY + pd.Timedelta(days=1)
NEW_TEST_END         = max_date

print("=" * 60)
print("STEP 1 — NEW TRAIN/TEST BOUNDARY (full-year test)")
print("=" * 60)
print(f"Train      : {df['Date'].min().date()} -> {NEW_TRAIN_BOUNDARY.date()}")
print(f"Test       : {NEW_TEST_START.date()} -> {NEW_TEST_END.date()}  "
      f"({(NEW_TEST_END-NEW_TEST_START).days+1} days)")
print(f"(Was previously a single ~120 day origin window — this is now "
      f"{(NEW_TEST_END-NEW_TEST_START).days+1} days, multiple rolling origins)\n")


# =============================================================================
# STEP 2 — TRAIN PAIRS  (same logic as 07_horizon_dataset.py, new boundary)
# =============================================================================

day_lookup = df[["Store", "Date"] + TARGET_DAY_FEATURES + ["Sales", "log_sales"]].rename(
    columns={"Date": "target_date", "Sales": "target_Sales", "log_sales": "target_log_sales"}
)

def build_pairs(anchor_df, horizons, boundary_check):
    anchor_cols = ["Store", "Date"] + ANCHOR_CONTINUOUS + ANCHOR_STATIC
    anchor_slim = anchor_df[anchor_cols].copy()
    all_pairs = []
    for h in horizons:
        temp = anchor_slim.copy()
        temp["target_date"] = temp["Date"] + pd.Timedelta(days=h)
        temp["horizon"] = h
        temp = temp[boundary_check(temp["target_date"])]
        if len(temp) == 0:
            continue
        merged = temp.merge(day_lookup, on=["Store", "target_date"], how="inner")
        all_pairs.append(merged)
    result = pd.concat(all_pairs, ignore_index=True)
    result["horizon_bucket"] = result["horizon"].apply(horizon_to_bucket)
    return result

train_dates_all = sorted(df[df["Date"] <= NEW_TRAIN_BOUNDARY]["Date"].unique())
train_anchor_dates = set(train_dates_all[::5])
train_anchor_df = df[(df["Date"] <= NEW_TRAIN_BOUNDARY) & (df["Date"].isin(train_anchor_dates))]

train_pairs = build_pairs(
    train_anchor_df, ALL_HORIZONS,
    boundary_check=lambda target_date: target_date <= NEW_TRAIN_BOUNDARY
)

print("=" * 60)
print("STEP 2 — TRAIN PAIRS BUILT")
print("=" * 60)
print(f"Train anchor days sampled (every 5th): {len(train_anchor_dates)} of {len(train_dates_all)}")
print(f"Train pairs: {len(train_pairs):,}\n")


# =============================================================================
# STEP 3 — MULTI-ORIGIN TEST PAIRS  (the core fix)
# =============================================================================
#
# Origins spaced every 7 days from the train boundary up through the
# last point that still leaves room for the full 90-day horizon
# inside available data. Every bucket draws from the SAME set of
# origins — so every bucket sees the same mix of seasons/holidays.
# =============================================================================

max_horizon = max(ALL_HORIZONS)
last_valid_origin = NEW_TEST_END - pd.Timedelta(days=max_horizon)

origins = pd.date_range(start=NEW_TRAIN_BOUNDARY, end=last_valid_origin, freq="7D")

print("=" * 60)
print("STEP 3 — MULTI-ORIGIN TEST SET")
print("=" * 60)
print(f"Origins: every 7 days, {origins.min().date()} -> {origins.max().date()}")
print(f"Number of origins: {len(origins)}")
print(f"Each origin's last horizon (h=90) lands by: "
      f"{(origins.max() + pd.Timedelta(days=90)).date()}  (must be <= {NEW_TEST_END.date()})")

origin_anchor_df = df[df["Date"].isin(origins)]
print(f"\nOrigin anchor rows available in data: {origin_anchor_df['Date'].nunique()} "
      f"of {len(origins)} requested dates "
      f"(some origins may be non-trading days for the global calendar but each "
      f"store's OWN trading calendar is what's actually used per row)")

test_pairs = build_pairs(
    origin_anchor_df, ALL_HORIZONS,
    boundary_check=lambda target_date: (target_date > NEW_TRAIN_BOUNDARY) & (target_date <= NEW_TEST_END)
)

print(f"\nTest pairs (multi-origin): {len(test_pairs):,}")
print(f"\nTest pairs by bucket:")
print(test_pairs["horizon_bucket"].value_counts().reindex(BUCKET_ORDER).to_string())


# =============================================================================
# STEP 4 — LEAKAGE CHECKS
# =============================================================================

print("\n" + "=" * 60)
print("STEP 4 — LEAKAGE CHECKS")
print("=" * 60)
assert (train_pairs["target_date"] <= NEW_TRAIN_BOUNDARY).all(), "Train target leaked!"
print("Check 1 passed: all train targets <= train boundary.")
assert (test_pairs["target_date"] > NEW_TRAIN_BOUNDARY).all(), "Test target before boundary!"
assert (test_pairs["target_date"] <= NEW_TEST_END).all(), "Test target beyond available data!"
print("Check 2 passed: every test target is strictly after train boundary,")
print("                meaning the model never saw any of these values as a")
print("                training label, regardless of how far into the test")
print("                period the ANCHOR itself falls.")


# =============================================================================
# STEP 5 — CALENDAR EXPOSURE CHECK  (did the fix actually work?)
# =============================================================================

print("\n" + "=" * 60)
print("STEP 5 — CALENDAR EXPOSURE PER BUCKET (the actual fix being validated)")
print("=" * 60)
for bucket in BUCKET_ORDER:
    sub = test_pairs[test_pairs["horizon_bucket"] == bucket]
    months_covered = sorted(sub["target_date"].dt.to_period("M").unique().astype(str))
    print(f"\n{bucket:<10} target dates span {sub['target_date'].min().date()} -> "
          f"{sub['target_date'].max().date()}")
    print(f"           calendar months covered: {months_covered}")


# =============================================================================
# STEP 6 — SCALE  (fresh scaler, fit on NEW train only)
# =============================================================================

scaler = StandardScaler()
scaler.fit(train_pairs[CONTINUOUS_FEATURES])
train_pairs[CONTINUOUS_FEATURES] = scaler.transform(train_pairs[CONTINUOUS_FEATURES])
test_pairs[CONTINUOUS_FEATURES]  = scaler.transform(test_pairs[CONTINUOUS_FEATURES])

os.makedirs("artifacts", exist_ok=True)
joblib.dump(scaler, "artifacts/scaler_fair_eval.pkl")

print("\n" + "=" * 60)
print("STEP 6 — SCALED (fresh scaler saved as scaler_fair_eval.pkl)")
print("=" * 60)


# =============================================================================
# STEP 7 — TRAIN XGBOOST PER BUCKET ON THE NEW (SMALLER) TRAINING SET
# =============================================================================

print("\n" + "=" * 60)
print("STEP 7 — TRAINING XGBOOST PER BUCKET (new boundary)")
print("=" * 60)
print("Runtime: comparable to the original 08_horizon_modelling.py session.")
print("Saved as xgb_fair_eval_<bucket>.pkl — production artifacts untouched.\n")

xgb_param_dist = {
    "n_estimators"     : [200, 300, 400, 500],
    "max_depth"        : [4, 6, 8, 10],
    "learning_rate"    : [0.01, 0.03, 0.05, 0.1],
    "subsample"        : [0.7, 0.8, 0.9, 1.0],
    "colsample_bytree" : [0.7, 0.8, 0.9, 1.0],
}

fair_models  = {}
fair_metrics = {}
fair_predictions = {}

for bucket in BUCKET_ORDER:
    print(f"\n--- {bucket.upper()} ---")
    bucket_train = train_pairs[train_pairs["horizon_bucket"] == bucket].reset_index(drop=True)
    bucket_test  = test_pairs[test_pairs["horizon_bucket"] == bucket].reset_index(drop=True)

    X_train = bucket_train[ALL_FEATURES]
    y_train = bucket_train["target_log_sales"]
    X_test  = bucket_test[ALL_FEATURES]
    y_test_raw = bucket_test["target_Sales"]

    print(f"Train rows: {len(bucket_train):,}   Test rows (multi-origin): {len(bucket_test):,}")

    cv_splits, skipped = per_store_time_series_cv(bucket_train, n_splits=3)
    print(f"CV folds built — stores skipped: {skipped}")

    xgb_base = XGBRegressor(objective="reg:squarederror", random_state=42, verbosity=0, n_jobs=-1)
    search = RandomizedSearchCV(
        estimator=xgb_base, param_distributions=xgb_param_dist,
        n_iter=10, cv=cv_splits, scoring="neg_root_mean_squared_error",
        random_state=42, n_jobs=-1, verbose=0
    )
    search.fit(X_train, y_train)
    print(f"Best params: {search.best_params_}")

    final_model = XGBRegressor(**search.best_params_, objective="reg:squarederror",
                                random_state=42, verbosity=0, n_jobs=-1)
    final_model.fit(X_train, y_train)
    joblib.dump(final_model, f"artifacts/xgb_fair_eval_{bucket}.pkl")

    pred_log = final_model.predict(X_test)
    pred = np.clip(np.expm1(pred_log), 0, None)

    metrics = compute_all_metrics(y_test_raw.values, pred)
    print(f"{bucket.upper()} fair-eval RMSPE: {metrics['RMSPE']:.2f}%  "
          f"(was {OLD_BUCKET_RMSPE[bucket]:.2f}% under the old single-origin test)")

    fair_models[bucket] = final_model
    fair_metrics[bucket] = metrics
    fair_predictions[bucket] = {
        "store"      : bucket_test["Store"].values,   
        "origin"     : bucket_test["Date"].values,
        "horizon"    : (bucket_test["target_date"] - bucket_test["Date"]).dt.days.values,
        "target_date": bucket_test["target_date"].values,
        "actual"     : y_test_raw.values,
        "predicted"  : pred,
    }


# =============================================================================
# STEP 8 — OLD vs NEW BUCKET COMPARISON (the headline result)
# =============================================================================

print("\n" + "=" * 60)
print("STEP 8 — DOES THE NEAR-vs-EXTENDED ANOMALY SURVIVE FAIR TESTING?")
print("=" * 60)
print(f"\n{'Bucket':<10} {'OLD RMSPE':>12} {'NEW RMSPE (fair)':>18} {'Change':>10}")
print("-" * 55)
for bucket in BUCKET_ORDER:
    old = OLD_BUCKET_RMSPE[bucket]
    new = fair_metrics[bucket]["RMSPE"]
    change = new - old
    print(f"{bucket:<10} {old:>11.2f}% {new:>17.2f}% {change:>+9.2f}pp")

near_new = fair_metrics["near"]["RMSPE"]
extended_new = fair_metrics["extended"]["RMSPE"]
print(f"\nNear  vs Extended (fair, multi-origin): {near_new:.2f}% vs {extended_new:.2f}%")
if near_new < extended_new:
    print("-> Near-term forecasts are now MORE accurate than long-term, as expected.")
    print("   The original 'near is worse' finding was a test-set artifact, confirmed.")
elif abs(near_new - extended_new) < 1.5:
    print("-> Near and Extended are now roughly comparable — horizon length alone")
    print("   does not strongly predict accuracy once calendar exposure is fair.")
else:
    print("-> Near-term is STILL worse even under fair testing — this is a genuine")
    print("   finding, not a test-set artifact. Worth investigating further.")

# =============================================================================
# STEP 9 — Deep Analysis Dataset 
# =============================================================================

print("\n" + "=" * 60)
print("STEP 8 — Deep Analysis Dataset")
print("=" * 60)

all_prediction_rows = []

for bucket, data in fair_predictions.items():
    print(list(data.keys()))
    bucket_df = pd.DataFrame({
    "store":          data["store"],        
    "origin_date":    data["origin"],
    "forecast_date":  data["target_date"],  
    "horizon":        data["horizon"],
    "horizon_bucket": bucket,
    "y_actual":       data["actual"],
    "y_pred":         data["predicted"],
})

    all_prediction_rows.append(bucket_df)

fair_eval_results = pd.concat(
    all_prediction_rows,
    ignore_index=True
)

fair_eval_results.to_csv(
    "data/processed/fair_eval_results.csv",
    index=False
)


# =============================================================================
# VISUALISATION 1 — Old vs New Bucket RMSPE
# =============================================================================

fig, ax = plt.subplots(figsize=(11, 6))
x = np.arange(len(BUCKET_ORDER))
width = 0.35

old_vals = [OLD_BUCKET_RMSPE[b] for b in BUCKET_ORDER]
new_vals = [fair_metrics[b]["RMSPE"] for b in BUCKET_ORDER]

ax.bar(x - width/2, old_vals, width, color=ACCENT2, edgecolor="#0f1117", label="OLD (single origin, confounded)")
ax.bar(x + width/2, new_vals, width, color=ACCENT3, edgecolor="#0f1117", label="NEW (multi-origin, fair)")
for i, (o, n) in enumerate(zip(old_vals, new_vals)):
    ax.text(i-width/2, o+0.3, f"{o:.1f}%", ha="center", fontsize=8, color="#c0d8e8")
    ax.text(i+width/2, n+0.3, f"{n:.1f}%", ha="center", fontsize=8, color="#c0d8e8")

ax.set_xticks(x)
ax.set_xticklabels(BUCKET_ORDER)
ax.set_title("Bucket RMSPE — Before vs After Controlling for Calendar Exposure",
             fontsize=13, color="#e0f0ff", pad=12)
ax.set_ylabel("RMSPE (%)")
ax.legend(fontsize=9)
ax.grid(True, axis="y")

plt.tight_layout()
plt.savefig("fair_eval_old_vs_new.png", dpi=150, bbox_inches="tight")
plt.show()


# =============================================================================
# VISUALISATION 2 — RMSPE by Origin, per Bucket (remaining variability)
# =============================================================================

fig, axes = plt.subplots(4, 1, figsize=(14, 13), sharex=False)
fig.suptitle("RMSPE by Origin Date — Within Each Bucket (multi-origin)",
             fontsize=13, color="#e0f0ff", y=1.01)

for ax, bucket in zip(axes, BUCKET_ORDER):
    preds = fair_predictions[bucket]
    df_b = pd.DataFrame({
        "origin": preds["origin"], "actual": preds["actual"], "predicted": preds["predicted"]
    })
    origin_rmspe = df_b.groupby("origin").apply(
        lambda g: rmspe(g["actual"].values, g["predicted"].values)
    ).reset_index(name="RMSPE")
    origin_rmspe["origin"] = pd.to_datetime(origin_rmspe["origin"])
    origin_rmspe.sort_values("origin", inplace=True)

    ax.plot(origin_rmspe["origin"], origin_rmspe["RMSPE"], color=BUCKET_COLORS[bucket],
            marker="o", markersize=4, lw=1.2)
    ax.axhline(fair_metrics[bucket]["RMSPE"], color="white", lw=1, linestyle="--",
               alpha=0.5, label=f"Bucket mean: {fair_metrics[bucket]['RMSPE']:.1f}%")
    ax.set_title(f"{bucket}", fontsize=10, color="#c0d8e8")
    ax.set_ylabel("RMSPE (%)")
    ax.legend(fontsize=8)
    ax.grid(True)

axes[-1].set_xlabel("Origin date")
plt.tight_layout()
plt.savefig("fair_eval_rmspe_by_origin.png", dpi=150, bbox_inches="tight")
plt.show()


# =============================================================================
# SAVE RESULTS
# =============================================================================

fair_results_df = pd.DataFrame(fair_metrics).T
fair_results_df["old_rmspe"] = [OLD_BUCKET_RMSPE[b] for b in fair_results_df.index]
fair_results_df.to_csv("data/processed/fair_eval_bucket_results.csv")

print(f"""
================================================================
       FAIR MULTI-ORIGIN BACKTEST — DONE
================================================================
  {len(origins)} origins, every 7 days, {origins.min().date()} -> {origins.max().date()}
  Every bucket evaluated across the SAME calendar exposure.

  Saved (experimental track, production untouched)
    artifacts/xgb_fair_eval_<bucket>.pkl
    artifacts/scaler_fair_eval.pkl
    data/processed/fair_eval_bucket_results.csv

  Plots
    fair_eval_old_vs_new.png
    fair_eval_rmspe_by_origin.png

  Decision point: review Step 8's verdict above before deciding
  whether to promote this fair-eval model into production, or
  whether to extend this analysis to LightGBM/Linear as Phase 2.
================================================================
""")
