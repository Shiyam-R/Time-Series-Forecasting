# =============================================================================
# ROSSMANN STORE SALES — DIRECT MULTI-HORIZON DATASET BUILDER
# File   : notebook/07_horizon_dataset.py
# Input  : data/processed/features.csv   (day-level, from 02_feature_engineering.py)
# Output : data/processed/train_horizon.csv
#          data/processed/test_horizon.csv
#          artifacts/scaler_horizon.pkl
#          artifacts/horizon_continuous_features.pkl
#          artifacts/horizon_all_features.pkl
#          artifacts/horizon_boundary_dates.pkl
# =============================================================================
#
# WHY THIS EXISTS:
#   Our original model implicitly assumed "today's lag data is always
#   available" — fine for 1-day-ahead, but breaks down for longer
#   horizons because it forces RECURSIVE forecasting (feeding
#   predictions back in as fake lag values, compounding error).
#
#   This script builds a DIRECT multi-horizon dataset instead. Every
#   training row is an (anchor_day, target_day) PAIR where:
#
#     ANCHOR features  (known as of "today")
#       lag_7, lag_14, lag_30, rolling stats, store metadata
#
#     TARGET-DAY features  (knowable in advance about the future day)
#       DayOfWeek, Month, cyclical encodings, Promo, StateHoliday,
#       SchoolHoliday  — a business PLANS these, they are not derived
#       from history
#
#     horizon  (NEW explicit feature)
#       how many days ahead the target day is from the anchor day —
#       the model directly learns how the relationship changes the
#       further out it is asked to predict
#
#     target   = the target day's actual Sales / log_sales
#
#   No model ever sees another model's prediction as an input.
#   This eliminates compounding recursive error entirely.
# =============================================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import joblib
import os
import warnings
warnings.filterwarnings("ignore")

from sklearn.preprocessing import StandardScaler

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
# STEP 0 — LOAD DAY-LEVEL FEATURES
# =============================================================================

df = pd.read_csv("data/processed/features.csv", parse_dates=["Date"])
df.sort_values(["Store", "Date"], inplace=True)
df.reset_index(drop=True, inplace=True)

print("=" * 60)
print("STEP 0 — DAY-LEVEL FEATURES LOADED")
print("=" * 60)
print(f"Shape  : {df.shape}")
print(f"Range  : {df['Date'].min().date()} → {df['Date'].max().date()}")
print(f"Stores : {df['Store'].nunique()}\n")


# =============================================================================
# STEP 1 — DEFINE FEATURE ROLES
# =============================================================================
#
# ANCHOR features  → pulled from the anchor (today) row
# TARGET features  → pulled from the target (future) row — these are
#                    the things a business plans or already knows in
#                    advance about a future calendar day
# =============================================================================

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
# dedupe while preserving order
ALL_FEATURES = list(dict.fromkeys(ALL_FEATURES))

print("=" * 60)
print("STEP 1 — FEATURE ROLES")
print("=" * 60)
print(f"Anchor (known today)      : {ANCHOR_CONTINUOUS + ANCHOR_STATIC}")
print(f"Target-day (known ahead)  : {TARGET_DAY_FEATURES}")
print(f"New feature               : horizon")
print(f"Total model features      : {len(ALL_FEATURES)}\n")


# =============================================================================
# STEP 2 — HORIZON BUCKETS
# =============================================================================

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
print("STEP 2 — HORIZON BUCKETS")
print("=" * 60)
for bucket, horizons in HORIZON_BUCKETS.items():
    print(f"  {bucket:<10} : {horizons}")
print(f"\nTotal sampled horizons: {len(ALL_HORIZONS)} → {ALL_HORIZONS}\n")


# =============================================================================
# STEP 3 — DEFINE TRAIN / TEST BOUNDARY  (Extended Test Window)
# =============================================================================
#
# Test must be long enough to fully validate the Extended bucket
# (horizon up to 90 days). The test-anchor window itself spans the
# last 30 days of train, so the EARLIEST possible test anchor still
# needs 90 more days of runway inside the test period.
#
#   required test length >= anchor_window (30) + max_horizon (90) = 120 days
#
# 4 months (~122 days) covers this with a small safety margin.
# =============================================================================

max_date = df["Date"].max()
TEST_MONTHS = 4
test_start    = max_date - pd.DateOffset(months=TEST_MONTHS) + pd.Timedelta(days=1)
train_boundary = test_start - pd.Timedelta(days=1)
test_end      = max_date

TEST_ANCHOR_WINDOW_DAYS = 30
test_anchor_window_start = train_boundary - pd.Timedelta(days=TEST_ANCHOR_WINDOW_DAYS - 1)

print("=" * 60)
print("STEP 3 — TRAIN / TEST BOUNDARY")
print("=" * 60)
print(f"Train period       : {df['Date'].min().date()} → {train_boundary.date()}")
print(f"Test period        : {test_start.date()} → {test_end.date()}")
print(f"Test length        : {(test_end - test_start).days + 1} days")
print(f"Test-anchor window : {test_anchor_window_start.date()} → {train_boundary.date()}"
      f"  ({TEST_ANCHOR_WINDOW_DAYS} days)")

required_days = TEST_ANCHOR_WINDOW_DAYS + max(ALL_HORIZONS)
actual_days   = (test_end - test_start).days + 1
print(f"\nSanity check — required test length for full horizon coverage: {required_days} days")
print(f"Actual test length                                            : {actual_days} days")
assert actual_days >= required_days, "Test window too short for max horizon — increase TEST_MONTHS"
print("Test window is long enough to validate every horizon up to 90 days.\n")


# =============================================================================
# STEP 4 — BUILD ANCHOR-TARGET PAIRS  (Vectorised Self-Merge)
# =============================================================================
#
# For each horizon h:
#   target_date = anchor_date + h days
#   merge anchor rows with the row AT target_date for the SAME store
#   inner join → pairs where the target day didn't exist (closed day)
#                are automatically and correctly dropped
# =============================================================================

day_lookup = df[["Store", "Date"] + TARGET_DAY_FEATURES + ["Sales", "log_sales"]].rename(
    columns={"Date": "target_date", "Sales": "target_Sales", "log_sales": "target_log_sales"}
)

def build_pairs(anchor_df, horizons, boundary_check):
    """
    anchor_df       : rows eligible to be anchors (Store, Date, anchor features)
    horizons        : list of horizon values to build
    boundary_check  : function(target_date) -> bool, keeps only valid target dates
    """
    anchor_cols = ["Store", "Date"] + ANCHOR_CONTINUOUS + ANCHOR_STATIC
    anchor_slim = anchor_df[anchor_cols].copy()

    all_pairs = []
    for h in horizons:
        temp = anchor_slim.copy()
        temp["target_date"] = temp["Date"] + pd.Timedelta(days=h)
        temp["horizon"] = h

        # Keep only target dates allowed for this split (train vs test)
        temp = temp[boundary_check(temp["target_date"])]
        if len(temp) == 0:
            continue

        merged = temp.merge(day_lookup, on=["Store", "target_date"], how="inner")
        all_pairs.append(merged)

    result = pd.concat(all_pairs, ignore_index=True)
    result["horizon_bucket"] = result["horizon"].apply(horizon_to_bucket)
    return result


# ── TRAIN anchors: every 5th day across the train period (sampling) ────────
train_dates_all = sorted(df[df["Date"] <= train_boundary]["Date"].unique())
train_anchor_dates = set(train_dates_all[::5])   # every 5th day

train_anchor_df = df[
    (df["Date"] <= train_boundary) & (df["Date"].isin(train_anchor_dates))
]

train_pairs = build_pairs(
    train_anchor_df,
    ALL_HORIZONS,
    boundary_check=lambda target_date: target_date <= train_boundary
)

# ── TEST anchors: full daily density in the last 30 days of train ──────────
test_anchor_df = df[
    (df["Date"] >= test_anchor_window_start) & (df["Date"] <= train_boundary)
]

test_pairs = build_pairs(
    test_anchor_df,
    ALL_HORIZONS,
    boundary_check=lambda target_date: (target_date >= test_start) & (target_date <= test_end)
)

print("=" * 60)
print("STEP 4 — ANCHOR-TARGET PAIRS BUILT")
print("=" * 60)
print(f"Train anchor days sampled (every 5th) : {len(train_anchor_dates)} "
      f"out of {len(train_dates_all)} available")
print(f"Test anchor days (full density)       : {test_anchor_df['Date'].nunique()}")
print(f"\nRow counts:")
print(f"  Original day-level rows : {len(df):,}")
print(f"  Train pairs (stacked)   : {len(train_pairs):,}")
print(f"  Test  pairs (stacked)   : {len(test_pairs):,}")

print(f"\nTrain pairs by horizon bucket:")
print(train_pairs["horizon_bucket"].value_counts().to_string())
print(f"\nTest pairs by horizon bucket:")
print(test_pairs["horizon_bucket"].value_counts().to_string())


# =============================================================================
# STEP 5 — LEAKAGE PREVENTION ASSERTIONS
# =============================================================================
#
# These checks are not optional — they are the proof that no
# training example uses information from the future, and no test
# example uses information it shouldn't have. This is the kind of
# rigor a production forecasting system MUST have before shipping.
# =============================================================================

print("\n" + "=" * 60)
print("STEP 5 — LEAKAGE PREVENTION CHECKS")
print("=" * 60)

assert (train_pairs["Date"] <= train_boundary).all(), "Train anchor leaked past boundary!"
assert (train_pairs["target_date"] <= train_boundary).all(), "Train target leaked past boundary!"
print("Check 1 passed — all train anchors AND targets fall within the train period.")

assert (test_pairs["Date"] <= train_boundary).all(), "Test anchor leaked — anchor must be known data!"
assert (test_pairs["target_date"] >= test_start).all(), "Test target falls before test period!"
assert (test_pairs["target_date"] <= test_end).all(), "Test target leaked past test period!"
print("Check 2 passed — all test anchors are known data (<= train boundary).")
print("Check 3 passed — all test targets fall strictly within the held-out test period.")
print("\nNo training example uses a future outcome. No test example")
print("uses information that wouldn't be available at forecast time.")


# =============================================================================
# STEP 6 — SCALE CONTINUOUS FEATURES
# =============================================================================
#
# Same principle as before — fit on train only, apply to both.
# 'horizon' is included here since it's a magnitude-meaningful
# continuous number (1 to 90), not a category.
# =============================================================================

scaler = StandardScaler()
scaler.fit(train_pairs[CONTINUOUS_FEATURES])

train_pairs[CONTINUOUS_FEATURES] = scaler.transform(train_pairs[CONTINUOUS_FEATURES])
test_pairs[CONTINUOUS_FEATURES]  = scaler.transform(test_pairs[CONTINUOUS_FEATURES])

print("\n" + "=" * 60)
print("STEP 6 — SCALING (Continuous Features + Horizon)")
print("=" * 60)
print(f"Scaled columns: {CONTINUOUS_FEATURES}")
print(f"\nTrain scaled sample:")
print(train_pairs[CONTINUOUS_FEATURES].head(3).to_string())


# =============================================================================
# STEP 7 — SAVE ARTIFACTS
# =============================================================================

os.makedirs("data/processed", exist_ok=True)
os.makedirs("artifacts", exist_ok=True)

keep_cols = ["Store", "Date", "target_date", "horizon", "horizon_bucket"] + \
            [f for f in ALL_FEATURES if f != "horizon"] + \
            ["target_Sales", "target_log_sales"]

train_pairs[keep_cols].to_csv("data/processed/train_horizon.csv", index=False)
test_pairs[keep_cols].to_csv("data/processed/test_horizon.csv", index=False)

joblib.dump(scaler, "artifacts/scaler_horizon.pkl")
joblib.dump(CONTINUOUS_FEATURES, "artifacts/horizon_continuous_features.pkl")
joblib.dump(ALL_FEATURES, "artifacts/horizon_all_features.pkl")
joblib.dump({
    "train_boundary"          : train_boundary,
    "test_start"               : test_start,
    "test_end"                 : test_end,
    "test_anchor_window_start" : test_anchor_window_start,
    "horizon_buckets"          : HORIZON_BUCKETS,
}, "artifacts/horizon_boundary_dates.pkl")

print("\n" + "=" * 60)
print("STEP 7 — SAVED")
print("=" * 60)
print(f"data/processed/train_horizon.csv -> {train_pairs.shape[0]:,} rows")
print(f"data/processed/test_horizon.csv  -> {test_pairs.shape[0]:,} rows")
print("artifacts/scaler_horizon.pkl")
print("artifacts/horizon_continuous_features.pkl")
print("artifacts/horizon_all_features.pkl")
print("artifacts/horizon_boundary_dates.pkl")


# =============================================================================
# STEP 7b — LAST-KNOWN-PER-STORE SNAPSHOT  (for live API inference)
# =============================================================================
#
# At inference time, "today" is always the most recent real date we
# have data for. This snapshot is simply the LAST row per store from
# the full day-level dataset — it already contains exactly the
# anchor features (lag_7/14/30, rolling stats, static store info)
# needed to forecast ANY future horizon from that single anchor point.
#
# No recomputation needed — features.csv already built these via
# shift() during feature engineering.
# =============================================================================

anchor_cols = ["Store", "Date"] + ANCHOR_CONTINUOUS + ANCHOR_STATIC
last_known_per_store = (
    df.sort_values(["Store", "Date"])
      .groupby("Store")
      .tail(1)[anchor_cols]
      .set_index("Store")
)

joblib.dump(last_known_per_store, "artifacts/last_known_per_store.pkl")

print("\n" + "=" * 60)
print("STEP 7b — LAST-KNOWN-PER-STORE SNAPSHOT SAVED")
print("=" * 60)
print(f"artifacts/last_known_per_store.pkl -> {last_known_per_store.shape[0]} stores")
print(f"Anchor (today) date: {last_known_per_store['Date'].iloc[0].date()}")
print(f"\nSample (Store 1):")
print(last_known_per_store.loc[[1]].to_string())


# =============================================================================
# VISUALISATION 1 — Anchor/Target Structure for One Example Store
# =============================================================================

example_store = train_pairs["Store"].iloc[0]
example_anchor = train_pairs[train_pairs["Store"] == example_store]["Date"].iloc[0]
example_rows = train_pairs[
    (train_pairs["Store"] == example_store) & (train_pairs["Date"] == example_anchor)
].sort_values("horizon")

fig, ax = plt.subplots(figsize=(13, 4))
ax.scatter([example_anchor]*len(example_rows), [0]*len(example_rows),
           color=ACCENT4, s=120, zorder=5, label="Anchor day (today)")
ax.scatter(example_rows["target_date"], [0]*len(example_rows),
           color=ACCENT3, s=60, zorder=4, label="Target days (forecasted)")

for _, row in example_rows.iterrows():
    ax.annotate(f"h={int(row['horizon'])}",
                xy=(row["target_date"], 0), xytext=(0, 15),
                textcoords="offset points", fontsize=7,
                color="#8ab4d4", ha="center")
    ax.plot([example_anchor, row["target_date"]], [0, 0],
            color=ACCENT, lw=0.8, alpha=0.4, zorder=1)

ax.axhline(0, color="#2a3550", lw=1)
ax.set_yticks([])
ax.set_title(f"One Anchor Day -> Multiple Target Days (Store {example_store})",
             fontsize=12, color="#e0f0ff", pad=12)
ax.set_xlabel("Date")
ax.legend(fontsize=9, loc="upper left")
ax.grid(True, axis="x")

plt.tight_layout()
plt.savefig("horizon_anchor_target_structure.png", dpi=150, bbox_inches="tight")
plt.show()


# =============================================================================
# VISUALISATION 2 — Row Count by Horizon (Train vs Test)
# =============================================================================

fig, axes = plt.subplots(1, 2, figsize=(15, 5))
fig.suptitle("Pair Count by Horizon — Train vs Test",
             fontsize=13, color="#e0f0ff", y=1.02)

train_horizon_counts = train_pairs["horizon"].value_counts().sort_index()
test_horizon_counts  = test_pairs["horizon"].value_counts().sort_index()

bucket_colors = {"near": ACCENT3, "mid": ACCENT4, "far": ACCENT2, "extended": ACCENT5}
train_colors = [bucket_colors.get(horizon_to_bucket(h), "#888888") for h in train_horizon_counts.index]
test_colors  = [bucket_colors.get(horizon_to_bucket(h), "#888888") for h in test_horizon_counts.index]

axes[0].bar(train_horizon_counts.index.astype(str), train_horizon_counts.values,
            color=train_colors, edgecolor="#0f1117")
axes[0].set_title("Train Pairs by Horizon (days)", fontsize=10, color="#c0d8e8")
axes[0].set_xlabel("Horizon (days ahead)")
axes[0].set_ylabel("Number of Pairs")
axes[0].tick_params(axis="x", rotation=45)
axes[0].grid(True, axis="y")

axes[1].bar(test_horizon_counts.index.astype(str), test_horizon_counts.values,
            color=test_colors, edgecolor="#0f1117")
axes[1].set_title("Test Pairs by Horizon (days)", fontsize=10, color="#c0d8e8")
axes[1].set_xlabel("Horizon (days ahead)")
axes[1].tick_params(axis="x", rotation=45)
axes[1].grid(True, axis="y")

plt.tight_layout()
plt.savefig("horizon_pair_counts.png", dpi=150, bbox_inches="tight")
plt.show()


# =============================================================================
# FINAL SUMMARY
# =============================================================================

print(f"""
================================================================
       DIRECT MULTI-HORIZON DATASET — DONE
================================================================
  Structure
    Each row = (anchor_day, target_day) pair
    Anchor features : lag/rolling/store metadata (today)
    Target features : calendar/Promo/Holiday (future day)
    New feature      : horizon (1 to 90 days)

  Train
    Anchors sampled every 5th day
    {len(train_pairs):,} pairs across {len(ALL_HORIZONS)} horizons

  Test
    Anchors at full density (last 30 days of train)
    {len(test_pairs):,} pairs, targets strictly in held-out
    test period ({test_start.date()} -> {test_end.date()})

  Leakage prevention
    All assertions passed

  Saved
    data/processed/train_horizon.csv
    data/processed/test_horizon.csv
    artifacts/scaler_horizon.pkl
    artifacts/horizon_*.pkl

  Next step -> 08_horizon_modelling.py
  (4 XGBoost models, one per horizon bucket)
================================================================
""")