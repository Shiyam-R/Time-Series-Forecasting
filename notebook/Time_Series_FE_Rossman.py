# =============================================================================
# ROSSMANN STORE SALES — FEATURE ENGINEERING
# File   : notebook/02_feature_engineering.py
# Input  : data/raw/train.csv
#          data/raw/store.csv
# Output : data/processed/features.csv
# =============================================================================
#
# FEATURES CREATED:
#   Lag        → lag_7, lag_14, lag_30
#   Rolling    → mean & std for window 7 and 30
#   Calendar   → DayOfWeek, Month, Year, WeekOfMonth,
#                IsWeekend, IsMonthStart, IsMonthEnd
#   Cyclical   → sin/cos of DayOfWeek and Month
#   External   → Promo, StateHoliday (encoded), SchoolHoliday,
#                Promo × DayOfWeek interaction
#   Store      → StoreType (encoded), Assortment (encoded),
#                CompetitionDistance, CompetitionDistance_missing, Promo2
#   Target     → log_sales
#
# DROPPED:
#   Customers  → data leakage (not known at prediction time)
#   lag_365    → loses too many rows (365 per store × 1115 stores)
# =============================================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import warnings
warnings.filterwarnings("ignore")

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
# STEP 0 — LOAD AND MERGE RAW DATA
# =============================================================================

train = pd.read_csv(
    "data/raw/train.csv",
    parse_dates=["Date"],
    dtype={"StateHoliday": str}
)
store = pd.read_csv("data/raw/store.csv")

# Merge store metadata
df = train.merge(store, on="Store", how="left")

# Remove closed store days — Sales=0 on closed days is not real behaviour
df = df[df["Open"] == 1].copy()

# Sort by Store then Date — critical for lag features
# Lag features shift rows DOWN within each store
# If not sorted, lag_7 for Store 2 might accidentally use Store 1 values
df.sort_values(["Store", "Date"], inplace=True)
df.reset_index(drop=True, inplace=True)

# Fill CompetitionDistance NaN with median before any feature creation
df["CompetitionDistance"].fillna(
    df["CompetitionDistance"].median(), inplace=True
)

print("=" * 55)
print("STEP 0 — DATA LOADED AND MERGED")
print("=" * 55)
print(f"Shape  : {df.shape}")
print(f"Stores : {df['Store'].nunique()}")
print(f"Range  : {df['Date'].min().date()} → {df['Date'].max().date()}\n")


# =============================================================================
# STEP 1 — DROP CUSTOMERS COLUMN (DATA LEAKAGE)
# =============================================================================
#
# WHY: At prediction time we do not know how many customers will visit.
#      Customers has the highest correlation with Sales (from EDA).
#      If we keep it, the model will rely on it heavily and fail
#      in production where this value is unavailable.
#
# This is called DATA LEAKAGE — using information that would not be
# available at the time of making a real prediction.
# =============================================================================

df.drop(columns=["Customers"], inplace=True)

print("=" * 55)
print("STEP 1 — DROPPED CUSTOMERS COLUMN (LEAKAGE)")
print("=" * 55)
print("Customers removed — not available at prediction time.\n")


# =============================================================================
# STEP 2 — LAG FEATURES  (lag_7, lag_14, lag_30)
# =============================================================================
#
# CRITICAL: Use groupby("Store") before shift.
#
# WHY: Our data has 1115 stores stacked on top of each other.
#      Without groupby, lag_7 at the first row of Store 2
#      would look back into the last rows of Store 1 — completely wrong.
#
#      With groupby("Store"), each store's lags only look within
#      that store's own history.
#
# lag_7  → same day last week    (strongest, ACF confirmed)
# lag_14 → same day 2 weeks ago
# lag_30 → approximately same day last month
# =============================================================================

df["lag_7"]  = df.groupby("Store")["Sales"].shift(7)
df["lag_14"] = df.groupby("Store")["Sales"].shift(14)
df["lag_30"] = df.groupby("Store")["Sales"].shift(30)

print("=" * 55)
print("STEP 2 — LAG FEATURES")
print("=" * 55)
print("Sample (Store 1, first 35 rows):")
store1 = df[df["Store"] == 1][["Date","Sales","lag_7","lag_14","lag_30"]].head(35).tail(8)
print(store1.to_string())
print("""
lag_7  at row 8  = Sales from row 1  (7 days ago) ✓
lag_14 at row 15 = Sales from row 1  (14 days ago) ✓
NaN in early rows is correct — no history available yet
""")


# =============================================================================
# STEP 3 — ROLLING FEATURES  (window 7 and 30)
# =============================================================================
#
# Same groupby("Store") logic applies here.
# shift(1) before rolling to prevent data leakage —
# rolling window must only use PAST values, never the current row.
#
# rolling_mean_7  → average of last 7 days  (short term trend)
# rolling_std_7   → spread of last 7 days   (recent volatility)
# rolling_mean_30 → average of last 30 days (medium term trend)
# rolling_std_30  → spread of last 30 days
# =============================================================================

def rolling_feature(series, window, func="mean"):
    shifted = series.shift(1)
    if func == "mean":
        return shifted.rolling(window=window, min_periods=1).mean()
    elif func == "std":
        return shifted.rolling(window=window, min_periods=1).std()

df["rolling_mean_7"]  = df.groupby("Store")["Sales"].transform(
    lambda x: rolling_feature(x, 7, "mean"))
df["rolling_std_7"]   = df.groupby("Store")["Sales"].transform(
    lambda x: rolling_feature(x, 7, "std"))
df["rolling_mean_30"] = df.groupby("Store")["Sales"].transform(
    lambda x: rolling_feature(x, 30, "mean"))
df["rolling_std_30"]  = df.groupby("Store")["Sales"].transform(
    lambda x: rolling_feature(x, 30, "std"))

print("=" * 55)
print("STEP 3 — ROLLING FEATURES")
print("=" * 55)
print(df[df["Store"]==1][["Date","Sales",
      "rolling_mean_7","rolling_std_7",
      "rolling_mean_30","rolling_std_30"]].head(10).to_string())
print("""
min_periods=1 → allows rolling to compute even with fewer than
               window rows (avoids extra NaN in early rows)
""")


# =============================================================================
# STEP 4 — CALENDAR FEATURES
# =============================================================================
#
# DayOfWeek  → already exists in raw data (1=Mon to 7=Sun)
# Month      → extract from Date (1–12)
# Year       → extract from Date
# WeekOfMonth → which week within the month (1st, 2nd, 3rd, 4th)
#               first week of month has different sales than last week
# IsWeekend  → Saturday=6, Sunday=7 in Rossmann encoding
# IsMonthStart → first 3 days of month (paycheck effect — people spend more)
# IsMonthEnd   → last 3 days of month
# =============================================================================

df["Month"]       = df["Date"].dt.month
df["Year"]        = df["Date"].dt.year
df["WeekOfMonth"] = (df["Date"].dt.day - 1) // 7 + 1   # 1, 2, 3, 4
df["IsWeekend"]   = (df["DayOfWeek"] >= 6).astype(int)  # 1 if Sat or Sun
df["IsMonthStart"] = (df["Date"].dt.day <= 3).astype(int)
df["IsMonthEnd"]   = (df["Date"].dt.day >= df["Date"].dt.days_in_month - 2).astype(int)

print("=" * 55)
print("STEP 4 — CALENDAR FEATURES")
print("=" * 55)
print(df[["Date","DayOfWeek","Month","Year",
          "WeekOfMonth","IsWeekend",
          "IsMonthStart","IsMonthEnd"]].head(10).to_string())
print(f"""
WeekOfMonth distribution:
{df['WeekOfMonth'].value_counts().sort_index().to_string()}

IsWeekend    : {df['IsWeekend'].sum():,} weekend rows
IsMonthStart : {df['IsMonthStart'].sum():,} month-start rows
IsMonthEnd   : {df['IsMonthEnd'].sum():,} month-end rows
""")


# =============================================================================
# STEP 5 — CYCLICAL ENCODING  (sin/cos of DayOfWeek and Month)
# =============================================================================
#
# DayOfWeek: Sunday(7) and Monday(1) are neighbours but
#            plain numbers treat them as far apart (gap of 6).
#            Sin/cos wraps them onto a circle so they are close.
#
# Month: December(12) and January(1) are neighbours —
#        same fix needed as air passengers project.
#
# Both need sin AND cos together to uniquely identify each value.
# =============================================================================

# DayOfWeek cyclical (1–7)
df["dow_sin"] = np.sin(2 * np.pi * df["DayOfWeek"] / 7)
df["dow_cos"] = np.cos(2 * np.pi * df["DayOfWeek"] / 7)

# Month cyclical (1–12)
df["month_sin"] = np.sin(2 * np.pi * df["Month"] / 12)
df["month_cos"] = np.cos(2 * np.pi * df["Month"] / 12)

print("=" * 55)
print("STEP 5 — CYCLICAL ENCODING")
print("=" * 55)
print("DayOfWeek  →  sin     cos")
dow_sample = df[["DayOfWeek","dow_sin","dow_cos"]].drop_duplicates().sort_values("DayOfWeek")
for _, row in dow_sample.iterrows():
    day_name = ["Mon","Tue","Wed","Thu","Fri","Sat","Sun"][int(row["DayOfWeek"])-1]
    print(f"  {day_name}({int(row['DayOfWeek'])})   "
          f"{row['dow_sin']:+.3f}   {row['dow_cos']:+.3f}")
print("""
Sun(7) and Mon(1) have similar cos values → model sees them as neighbours ✓
""")


# =============================================================================
# STEP 6 — EXTERNAL FEATURES
# =============================================================================
#
# Promo         → already exists as binary (0/1)
# SchoolHoliday → already exists as binary (0/1)
#
# StateHoliday  → has string values: "0", "a", "b", "c"
#                 Label encode: 0=none, 1=public, 2=Easter, 3=Christmas
#                 LightGBM handles label encoded categoricals well
#
# Promo × DayOfWeek interaction:
#   From EDA, promo effect differs by day of week.
#   A promo on Monday drives more sales than a promo on Sunday.
#   Multiplying them creates a feature that captures this interaction.
# =============================================================================

# StateHoliday label encoding
holiday_map = {"0": 0, "a": 1, "b": 2, "c": 3}
df["StateHoliday_enc"] = df["StateHoliday"].map(holiday_map).fillna(0).astype(int)


# =============================================================================
# STEP 6b — HOLIDAY DISTANCE FEATURES  (days_since / days_until)
# =============================================================================
#
# WHY: From the multi-horizon diagnostic, RMSPE spiked to ~36-37% on
# Easter days across EVERY horizon bucket, regardless of how far
# ahead the forecast was made. A single same-day StateHoliday flag
# does not tell the model that sales typically ramp DOWN approaching
# a multi-day holiday cluster and ramp back UP afterward — it only
# fires on the exact holiday date itself.
#
# These two features give the model a smooth signal of "how close
# are we to a holiday" in both directions, capped at 14 days so a
# handful of very isolated holidays don't create extreme outlier
# values that dominate the feature's scale.
#
# IMPLEMENTATION: vectorised per-store using np.searchsorted — for
# each day, binary-search the sorted list of that store's holiday
# dates to find the nearest one before and after.
# =============================================================================

HOLIDAY_DISTANCE_CAP = 14

def compute_holiday_distance(dates, holiday_dates, cap=HOLIDAY_DISTANCE_CAP):
    dates = dates.values.astype("datetime64[D]")
    if len(holiday_dates) == 0:
        return (np.full(len(dates), cap), np.full(len(dates), cap))

    holiday_dates = np.sort(np.asarray(holiday_dates, dtype="datetime64[D]"))
    idx = np.searchsorted(holiday_dates, dates)

    days_since = np.full(len(dates), cap)
    days_until = np.full(len(dates), cap)

    has_prev = idx > 0
    days_since[has_prev] = (dates[has_prev] - holiday_dates[idx[has_prev] - 1]).astype(int)

    has_next = idx < len(holiday_dates)
    days_until[has_next] = (holiday_dates[idx[has_next]] - dates[has_next]).astype(int)

    return np.minimum(days_since, cap), np.minimum(days_until, cap)

df.sort_values(["Store", "Date"], inplace=True)
df["days_since_holiday"] = 0
df["days_until_holiday"] = 0

for store_id, group in df.groupby("Store"):
    holiday_dates = group.loc[group["StateHoliday_enc"] != 0, "Date"]
    since, until = compute_holiday_distance(group["Date"], holiday_dates)
    df.loc[group.index, "days_since_holiday"] = since
    df.loc[group.index, "days_until_holiday"] = until

print("=" * 55)
print("STEP 6b — HOLIDAY DISTANCE FEATURES")
print("=" * 55)
print(f"Cap: {HOLIDAY_DISTANCE_CAP} days in each direction")
print(f"\nSample around a holiday (Store 1, days_since/until):")
sample_store = df[df["Store"]==1].sort_values("Date")
near_holiday = sample_store[sample_store["days_since_holiday"] <= 5]
print(near_holiday[["Date","StateHoliday_enc","days_since_holiday","days_until_holiday"]].head(10).to_string())
print(f"""
Example reading: days_since_holiday=0 means TODAY is a holiday.
days_until_holiday=2 means a holiday is coming in 2 days — sales
may already be ramping up (panic buying) or down (closures) ahead
of it, which a same-day-only flag could never capture.
""")


# Promo × DayOfWeek interaction
df["Promo_DayOfWeek"] = df["Promo"] * df["DayOfWeek"]

print("=" * 55)
print("STEP 6 — EXTERNAL FEATURES")
print("=" * 55)
print("StateHoliday encoding:")
for k, v in holiday_map.items():
    label = {"0":"None","a":"Public Holiday",
             "b":"Easter","c":"Christmas"}[k]
    print(f"  '{k}' → {v}  ({label})")

print(f"\nPromo × DayOfWeek sample:")
print(df[["Promo","DayOfWeek","Promo_DayOfWeek"]].drop_duplicates().sort_values(
    ["Promo","DayOfWeek"]).head(10).to_string())
print("""
Promo=0, any day  → Promo_DayOfWeek = 0 (no promo)
Promo=1, Mon(1)   → Promo_DayOfWeek = 1
Promo=1, Fri(5)   → Promo_DayOfWeek = 5
Promo=1, Sun(7)   → Promo_DayOfWeek = 7
Model learns different weights for each promo-day combination ✓
""")


# =============================================================================
# STEP 7 — STORE METADATA FEATURES
# =============================================================================
#
# StoreType   → "a","b","c","d" → label encode to 0,1,2,3
# Assortment  → "a","b","c"     → label encode to 0,1,2
#
# CompetitionDistance → already filled with median in Step 0
#
# CompetitionDistance_missing → binary flag
#   WHY: We filled NaN with median but the model should know
#        which rows had missing data originally.
#        Missing competition distance may mean no nearby competitor
#        which is itself useful information.
#
# Promo2 → already exists as binary (0/1)
# =============================================================================

# Store competition distance missing flag BEFORE fillna was applied
# Since fillna was applied in Step 0, recreate from original
train_raw = pd.read_csv("data/raw/train.csv", parse_dates=["Date"],
                         dtype={"StateHoliday": str})
store_raw  = pd.read_csv("data/raw/store.csv")
merged_raw = train_raw.merge(store_raw, on="Store", how="left")
merged_raw = merged_raw[merged_raw["Open"] == 1]
merged_raw.sort_values(["Store","Date"], inplace=True)
df["CompetitionDistance_missing"] = merged_raw["CompetitionDistance"].isna().astype(int).values

# StoreType label encoding
storetype_map  = {"a": 0, "b": 1, "c": 2, "d": 3}
assortment_map = {"a": 0, "b": 1, "c": 2}

df["StoreType_enc"]  = df["StoreType"].map(storetype_map)
df["Assortment_enc"] = df["Assortment"].map(assortment_map)

print("=" * 55)
print("STEP 7 — STORE METADATA FEATURES")
print("=" * 55)
print("StoreType encoding :")
for k, v in storetype_map.items():
    print(f"  '{k}' → {v}")
print("\nAssortment encoding:")
for k, v in assortment_map.items():
    label = {"a":"Basic","b":"Extra","c":"Extended"}[k]
    print(f"  '{k}' → {v}  ({label})")
print(f"\nCompetitionDistance_missing: "
      f"{df['CompetitionDistance_missing'].sum():,} rows originally had NaN")


# =============================================================================
# STEP 8 — LOG TRANSFORM TARGET
# =============================================================================
#
# From EDA: Sales is right skewed and multiplicative.
# log1p = log(1 + Sales) — safer than log(Sales) because:
#   log(0) = -infinity → breaks the model
#   log1p(0) = 0       → safe
#
# To reverse after prediction: np.expm1(predicted) = exp(predicted) - 1
# =============================================================================

df["log_sales"] = np.log1p(df["Sales"])

print("\n" + "=" * 55)
print("STEP 8 — LOG TRANSFORM TARGET")
print("=" * 55)
print(f"Sales range     : {df['Sales'].min()} → {df['Sales'].max():,}")
print(f"log_sales range : {df['log_sales'].min():.3f} → {df['log_sales'].max():.3f}")
print("""
log1p used instead of log because:
  log(0)    = -infinity → breaks model if any Sales = 0
  log1p(0)  = 0         → safe

To reverse predictions:
  predicted_sales = np.expm1(predicted_log_sales)
""")


# =============================================================================
# STEP 9 — DROP NaN ROWS AND UNUSED COLUMNS
# =============================================================================

# Columns no longer needed
drop_cols = [
    "Open",           # all rows are Open=1 after filtering
    "StateHoliday",   # replaced by StateHoliday_enc
    "StoreType",      # replaced by StoreType_enc
    "Assortment",     # replaced by Assortment_enc
    "PromoInterval",  # complex to use, dropping for now
]
df.drop(columns=drop_cols, inplace=True, errors="ignore")

before = len(df)
df.dropna(subset=["lag_7","lag_14","lag_30"], inplace=True)
after  = len(df)

print("=" * 55)
print("STEP 9 — DROP NaN AND UNUSED COLUMNS")
print("=" * 55)
print(f"Rows before dropna : {before:,}")
print(f"Rows dropped       : {before - after:,}  (lag_30 needs 30 days history)")
print(f"Rows after dropna  : {after:,}")
print(f"\nFinal columns ({len(df.columns)}):")
for col in df.columns:
    print(f"  {col}")


# =============================================================================
# STEP 10 — FINAL FEATURE MATRIX
# =============================================================================

FEATURES = [c for c in df.columns if c not in ["Sales", "log_sales", "Date"]]
TARGET   = "log_sales"

print("\n" + "=" * 55)
print("STEP 10 — FINAL FEATURE MATRIX")
print("=" * 55)
print(f"Features : {len(FEATURES)}")
print(f"Target   : {TARGET}")
print(f"Shape    : {df[FEATURES].shape}")
print(f"\nSample (last 3 rows):")
print(df[FEATURES + [TARGET]].tail(3).to_string())


# =============================================================================
# VISUALISATION 1 — Lag Features vs Sales (Store 1)
# =============================================================================

store1_df = df[df["Store"] == 1].copy()

fig, axes = plt.subplots(3, 1, figsize=(13, 11), sharex=True)
fig.suptitle("Lag Features vs Actual Sales — Store 1",
             fontsize=13, color="#e0f0ff", y=1.01)

lag_info = [
    ("lag_7",  "lag t-7  (same day last week)",    ACCENT),
    ("lag_14", "lag t-14 (same day 2 weeks ago)",  ACCENT4),
    ("lag_30", "lag t-30 (same day last month)",   ACCENT3),
]

for ax, (col, label, color) in zip(axes, lag_info):
    ax.plot(store1_df["Date"], store1_df["Sales"],
            color="white", lw=1.5, alpha=0.4, label="Actual Sales")
    ax.plot(store1_df["Date"], store1_df[col],
            color=color, lw=1.2, label=label)
    corr = store1_df["Sales"].corr(store1_df[col])
    ax.fill_between(store1_df["Date"],
                    store1_df["Sales"], store1_df[col],
                    alpha=0.07, color=color)
    ax.set_ylabel("Sales (€)")
    ax.legend(fontsize=8, loc="upper left")
    ax.set_title(f"r = {corr:.3f}", fontsize=9, color="#8ab4d4")
    ax.grid(True)

axes[-1].set_xlabel("Date")
plt.tight_layout()
plt.savefig("fe_lag_features.png", dpi=150, bbox_inches="tight")
plt.show()


# =============================================================================
# VISUALISATION 2 — Rolling Features (Store 1)
# =============================================================================

fig, axes = plt.subplots(2, 1, figsize=(13, 8), sharex=True)
fig.suptitle("Rolling Features — Mean & Std (Store 1)",
             fontsize=13, color="#e0f0ff", y=1.01)

axes[0].plot(store1_df["Date"], store1_df["Sales"],
             color="white", lw=1.2, alpha=0.35, label="Actual")
axes[0].plot(store1_df["Date"], store1_df["rolling_mean_7"],
             color=ACCENT, lw=1.5, label="rolling_mean_7  (fast)")
axes[0].plot(store1_df["Date"], store1_df["rolling_mean_30"],
             color=ACCENT4, lw=1.5, label="rolling_mean_30 (smooth)")
axes[0].set_ylabel("Sales (€)")
axes[0].legend(fontsize=9)
axes[0].grid(True)
axes[0].set_title("7-day reacts fast to spikes, 30-day shows smoother trend",
                  fontsize=9, color="#8ab4d4")

axes[1].plot(store1_df["Date"], store1_df["rolling_std_7"],
             color=ACCENT2, lw=1.5, label="rolling_std_7")
axes[1].plot(store1_df["Date"], store1_df["rolling_std_30"],
             color=ACCENT5, lw=1.5, label="rolling_std_30")
axes[1].fill_between(store1_df["Date"],
                     store1_df["rolling_std_30"],
                     alpha=0.1, color=ACCENT5)
axes[1].set_ylabel("Std Deviation")
axes[1].set_xlabel("Date")
axes[1].legend(fontsize=9)
axes[1].grid(True)
axes[1].set_title("Std spikes around promotions and holidays",
                  fontsize=9, color="#8ab4d4")

plt.tight_layout()
plt.savefig("fe_rolling_features.png", dpi=150, bbox_inches="tight")
plt.show()


# =============================================================================
# VISUALISATION 3 — Log Transform Effect
# =============================================================================

fig, axes = plt.subplots(1, 2, figsize=(13, 5))
fig.suptitle("Log Transform — Compressing Right Skew",
             fontsize=13, color="#e0f0ff")

axes[0].hist(df["Sales"], bins=60,
             color=ACCENT2, edgecolor="#0f1117", alpha=0.85)
axes[0].set_title("Original Sales\n(right skewed)",
                  fontsize=10, color="#c0d8e8")
axes[0].set_xlabel("Sales (€)")
axes[0].set_ylabel("Frequency")
axes[0].grid(True)

axes[1].hist(df["log_sales"], bins=60,
             color=ACCENT3, edgecolor="#0f1117", alpha=0.85)
axes[1].set_title("log1p(Sales)\n(approximately normal)",
                  fontsize=10, color="#c0d8e8")
axes[1].set_xlabel("log(Sales)")
axes[1].set_ylabel("Frequency")
axes[1].grid(True)

plt.tight_layout()
plt.savefig("fe_log_transform.png", dpi=150, bbox_inches="tight")
plt.show()


# =============================================================================
# SAVE PROCESSED DATA
# =============================================================================

os.makedirs("data/processed", exist_ok=True)
df.to_csv("data/processed/features.csv", index=False)

print("\n" + "=" * 55)
print("SAVED → data/processed/features.csv")
print("=" * 55)
print(f"Rows    : {df.shape[0]:,}")
print(f"Columns : {df.shape[1]}")

print("""
╔═════════════════════════════════════════════════════╗
║       FEATURE ENGINEERING — DONE                   ║
╠═════════════════════════════════════════════════════╣
║  Lag features                                       ║
║    lag_7  → same day last week                      ║
║    lag_14 → same day 2 weeks ago                    ║
║    lag_30 → same day last month                     ║
║                                                     ║
║  Rolling features                                   ║
║    rolling_mean_7,  rolling_std_7                   ║
║    rolling_mean_30, rolling_std_30                  ║
║                                                     ║
║  Calendar features                                  ║
║    DayOfWeek, Month, Year, WeekOfMonth              ║
║    IsWeekend, IsMonthStart, IsMonthEnd               ║
║                                                     ║
║  Cyclical features                                  ║
║    dow_sin, dow_cos, month_sin, month_cos            ║
║                                                     ║
║  External features                                  ║
║    Promo, StateHoliday_enc, SchoolHoliday           ║
║    Promo_DayOfWeek (interaction)                    ║
║    days_since_holiday, days_until_holiday (capped 14)║
║                                                     ║
║  Store features                                     ║
║    StoreType_enc, Assortment_enc                    ║
║    CompetitionDistance                              ║
║    CompetitionDistance_missing                      ║
║    Promo2                                           ║
║                                                     ║
║  Target                                             ║
║    log_sales → np.expm1() to reverse               ║
║                                                     ║
║  Dropped                                            ║
║    Customers (leakage), lag_365 (data loss)         ║
║    Open, StateHoliday, StoreType, Assortment        ║
║    PromoInterval                                    ║
║                                                     ║
║  Next step → 03_preprocessing.py                   ║
╚═════════════════════════════════════════════════════╝
""")