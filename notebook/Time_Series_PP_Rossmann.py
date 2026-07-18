# =============================================================================
# ROSSMANN STORE SALES — PREPROCESSING
# File   : notebook/03_preprocessing.py
# Input  : data/processed/features.csv
# Output : data/processed/train.csv
#          data/processed/test.csv
#          artifacts/scaler.pkl
# =============================================================================
#
# STEPS:
#   1. Load engineered features
#   2. Define feature groups (continuous vs categorical/binary/cyclical)
#   3. Train / Test split — last 2 months as test
#   4. StandardScaler — fit on train, apply to both, continuous features only
#   5. Save artifacts
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
# STEP 0 — LOAD ENGINEERED FEATURES
# =============================================================================

df = pd.read_csv(
    "data/processed/features.csv",
    parse_dates=["Date"]
)

print("=" * 55)
print("FEATURES LOADED")
print("=" * 55)
print(f"Shape  : {df.shape}")
print(f"Range  : {df['Date'].min().date()} → {df['Date'].max().date()}")
print(f"Stores : {df['Store'].nunique()}\n")


# =============================================================================
# STEP 1 — DEFINE FEATURE GROUPS
# =============================================================================
#
# Only CONTINUOUS features get scaled with StandardScaler.
# Categorical, binary, cyclical and ID columns are left untouched.
#
# WHY:
#   StandardScaler centres data around mean=0, std=1.
#   For continuous values like lag_7 or rolling_mean_30 this puts
#   all features on a comparable scale.
#
#   For binary flags (0/1), categorical codes (0,1,2,3) or
#   cyclical sin/cos (-1 to 1) — scaling would distort their meaning.
#   A category like StoreType_enc=2 does not mean "more" than 1,
#   it is just a different category. Scaling implies a magnitude
#   relationship that does not exist.
# =============================================================================

CONTINUOUS_FEATURES = [
    "lag_7", "lag_14", "lag_30",
    "rolling_mean_7", "rolling_std_7",
    "rolling_mean_30", "rolling_std_30",
    "CompetitionDistance",
]

CATEGORICAL_BINARY_FEATURES = [
    "DayOfWeek", "Month", "Year", "WeekOfMonth",
    "IsWeekend", "IsMonthStart", "IsMonthEnd",
    "Promo", "StateHoliday_enc", "SchoolHoliday",
    "Promo_DayOfWeek",
    "StoreType_enc", "Assortment_enc",
    "CompetitionDistance_missing", "Promo2",
]

CYCLICAL_FEATURES = [
    "dow_sin", "dow_cos", "month_sin", "month_cos"
]

# Store is kept for reference (grouping in evaluation) but NOT a model feature
ID_COLUMN = "Store"

ALL_FEATURES = CONTINUOUS_FEATURES + CATEGORICAL_BINARY_FEATURES + CYCLICAL_FEATURES
TARGET = "log_sales"
TARGET_RAW = "Sales"

print("=" * 55)
print("STEP 1 — FEATURE GROUPS")
print("=" * 55)
print(f"Continuous ({len(CONTINUOUS_FEATURES)}) — WILL be scaled:")
for f in CONTINUOUS_FEATURES:
    print(f"   {f}")
print(f"\nCategorical/Binary ({len(CATEGORICAL_BINARY_FEATURES)}) — NOT scaled:")
for f in CATEGORICAL_BINARY_FEATURES:
    print(f"   {f}")
print(f"\nCyclical ({len(CYCLICAL_FEATURES)}) — NOT scaled (already -1 to 1):")
for f in CYCLICAL_FEATURES:
    print(f"   {f}")
print(f"\nID column (kept for reference, not a feature): {ID_COLUMN}")
print(f"\nTotal model features: {len(ALL_FEATURES)}")
print(f"Target (log scale)  : {TARGET}")
print(f"Target (raw scale)  : {TARGET_RAW}\n")


# =============================================================================
# STEP 2 — TRAIN / TEST SPLIT  (Last 2 Months as Test)
# =============================================================================
#
# RULE: Never shuffle. Always split by time, across all stores together.
#
# Last 2 months covers all days of week, all weeks of month,
# both month-start and month-end patterns — good test coverage.
# =============================================================================

max_date   = df["Date"].max()
split_date = max_date - pd.DateOffset(months=2)

train = df[df["Date"] <  split_date].copy()
test  = df[df["Date"] >= split_date].copy()

print("=" * 55)
print("STEP 2 — TRAIN / TEST SPLIT")
print("=" * 55)
print(f"Split date : {split_date.date()}")
print(f"Train : {len(train):,} rows  "
      f"({train['Date'].min().date()} → {train['Date'].max().date()})")
print(f"Test  : {len(test):,} rows   "
      f"({test['Date'].min().date()} → {test['Date'].max().date()})")
print(f"\nTrain stores : {train['Store'].nunique()}")
print(f"Test  stores : {test['Store'].nunique()}")

# Verify test set covers all days of week
print(f"\nTest set DayOfWeek coverage:")
print(test["DayOfWeek"].value_counts().sort_index().to_string())
print(f"\nTest set WeekOfMonth coverage:")
print(test["WeekOfMonth"].value_counts().sort_index().to_string())


# =============================================================================
# STEP 3 — STANDARD SCALER  (Continuous Features Only)
# =============================================================================
#
# Fit ONLY on train data — same principle as air passengers project.
# Apply the same fitted scaler to both train and test.
#
# Only CONTINUOUS_FEATURES go through the scaler.
# Categorical, binary and cyclical features pass through unchanged.
# =============================================================================

scaler = StandardScaler()

# Fit on train continuous features only
scaler.fit(train[CONTINUOUS_FEATURES])

# Transform both train and test
train_scaled_continuous = pd.DataFrame(
    scaler.transform(train[CONTINUOUS_FEATURES]),
    columns=CONTINUOUS_FEATURES,
    index=train.index
)
test_scaled_continuous = pd.DataFrame(
    scaler.transform(test[CONTINUOUS_FEATURES]),
    columns=CONTINUOUS_FEATURES,
    index=test.index
)

# Replace continuous columns with scaled versions
train_final = train.copy()
test_final  = test.copy()
train_final[CONTINUOUS_FEATURES] = train_scaled_continuous
test_final[CONTINUOUS_FEATURES]  = test_scaled_continuous

print("\n" + "=" * 55)
print("STEP 3 — STANDARD SCALER (Continuous Only)")
print("=" * 55)

scale_info = pd.DataFrame({
    "feature" : CONTINUOUS_FEATURES,
    "mean"    : scaler.mean_.round(3),
    "std"     : scaler.scale_.round(3)
})
print("\nScaler learned from train data:")
print(scale_info.to_string(index=False))

print(f"\nBefore scaling (train sample):")
print(train[["lag_7","rolling_mean_30","CompetitionDistance"]].head(3).to_string())

print(f"\nAfter scaling (train sample):")
print(train_final[["lag_7","rolling_mean_30","CompetitionDistance"]].head(3).to_string())

print(f"\nUnchanged — categorical example (train sample):")
print(train_final[["DayOfWeek","StoreType_enc","Promo"]].head(3).to_string())

print(f"\nUnchanged — cyclical example (train sample):")
print(train_final[["dow_sin","dow_cos"]].head(3).to_string())


# =============================================================================
# STEP 4 — SAVE ALL ARTIFACTS
# =============================================================================

# os.makedirs("data/processed", exist_ok=True)
# os.makedirs("artifacts", exist_ok=True)

# Keep Date and Store for reference, plus all features and both targets
keep_cols = ["Date", "Store"] + ALL_FEATURES + [TARGET, TARGET_RAW]

train_out = train_final[keep_cols]
test_out  = test_final[keep_cols]

train_out.to_csv("data/processed/train.csv", index=False)
test_out.to_csv("data/processed/test.csv", index=False)

joblib.dump(scaler, "artifacts/scaler.pkl")
joblib.dump(CONTINUOUS_FEATURES, "artifacts/continuous_features.pkl")
joblib.dump(ALL_FEATURES, "artifacts/all_features.pkl")

print("\n" + "=" * 55)
print("STEP 4 — SAVED")
print("=" * 55)
print(f"data/processed/train.csv        → {train_out.shape}")
print(f"data/processed/test.csv         → {test_out.shape}")
print(f"artifacts/scaler.pkl            → StandardScaler (continuous features)")
print(f"artifacts/continuous_features.pkl → list of scaled feature names")
print(f"artifacts/all_features.pkl      → list of all model feature names")


# =============================================================================
# VISUALISATION 1 — Train / Test Split on Aggregated Sales
# =============================================================================

daily_train = train.groupby("Date")["Sales"].sum()
daily_test  = test.groupby("Date")["Sales"].sum()

fig, ax = plt.subplots(figsize=(14, 5))

ax.plot(daily_train.index, daily_train.values,
        color=ACCENT, lw=1.2, label=f"Train  ({len(train):,} rows)")
ax.plot(daily_test.index, daily_test.values,
        color=ACCENT2, lw=1.5, label=f"Test   ({len(test):,} rows)")

ax.axvline(split_date, color="white", lw=1.2,
           linestyle="--", alpha=0.5)
ax.annotate("← Train                    Test →",
            xy=(split_date, daily_train.max() * 0.85),
            fontsize=10, color="white", alpha=0.6, ha="center")

ax.set_title("Train / Test Split — Last 2 Months as Test",
             fontsize=13, color="#e0f0ff", pad=12)
ax.set_ylabel("Total Daily Sales (€)")
ax.set_xlabel("Date")
ax.legend(fontsize=10)
ax.grid(True)

plt.tight_layout()
plt.savefig("pp_train_test_split.png", dpi=150, bbox_inches="tight")
plt.show()


# =============================================================================
# VISUALISATION 2 — Before vs After Scaling (Continuous Features)
# =============================================================================

features_to_show = ["lag_7", "rolling_mean_30", "CompetitionDistance"]

fig, axes = plt.subplots(len(features_to_show), 2,
                         figsize=(13, 9))
fig.suptitle("Before vs After StandardScaler — Continuous Features (Store 1)",
             fontsize=13, color="#e0f0ff", y=1.01)

store1_train = train[train["Store"]==1]
store1_train_scaled = train_final[train_final["Store"]==1]

for i, feat in enumerate(features_to_show):
    axes[i][0].plot(store1_train["Date"], store1_train[feat],
                    color=ACCENT2, lw=1.2)
    axes[i][0].set_title(f"{feat} — before scaling",
                          fontsize=9, color="#c0d8e8")
    axes[i][0].grid(True)

    axes[i][1].plot(store1_train_scaled["Date"], store1_train_scaled[feat],
                    color=ACCENT3, lw=1.2)
    axes[i][1].axhline(0, color="white", lw=0.8,
                        linestyle="--", alpha=0.3)
    axes[i][1].set_title(f"{feat} — after scaling (mean≈0, std≈1)",
                          fontsize=9, color="#c0d8e8")
    axes[i][1].grid(True)

plt.tight_layout()
plt.savefig("pp_scaling_comparison.png", dpi=150, bbox_inches="tight")
plt.show()


# =============================================================================
# FINAL SUMMARY
# =============================================================================

print("""
╔═════════════════════════════════════════════════════╗
║         PREPROCESSING — DONE                       ║
╠═════════════════════════════════════════════════════╣
║  Split                                              ║
║    Method : Last 2 months as test (date cutoff)    ║
║    Covers : all DayOfWeek, all WeekOfMonth values   ║
║                                                     ║
║  Scaling                                            ║
║    Scaler   : StandardScaler                        ║
║    Fit on   : Train continuous features only         ║
║    Applied  : Continuous features in train + test   ║
║    Untouched: categorical, binary, cyclical, Store  ║
║                                                     ║
║  Saved                                              ║
║    data/processed/train.csv                         ║
║    data/processed/test.csv                          ║
║    artifacts/scaler.pkl                             ║
║    artifacts/continuous_features.pkl                ║
║    artifacts/all_features.pkl                       ║
║                                                     ║
║  Next step → 04_modelling.py                        ║
║  (LightGBM / XGBoost training and evaluation)       ║
╚═════════════════════════════════════════════════════╝
""")
