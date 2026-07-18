# =============================================================================
# ROSSMANN STORE SALES — EDA (Exploratory Data Analysis)
# File   : notebook/01_eda.py
# Input  : data/raw/train.csv
#          data/raw/store.csv
# =============================================================================
#
# SECTIONS:
#   0. Load and merge data
#   1. Basic overview
#   2. Target variable — Sales
#   3. Trend and seasonality
#   4. Additive vs multiplicative check
#   5. ACF and PACF
#   6. Store level analysis (representative stores)
#   7. External features impact
#   8. Store metadata analysis
#   9. Correlation analysis
#
# NOTE: All rows where Open=0 are removed before any analysis.
#       Closed store days have Sales=0 which is not real trading behaviour.
# =============================================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
import warnings
warnings.filterwarnings("ignore")

from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.stattools import adfuller

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


# =============================================================================
# SECTION 0 — LOAD AND MERGE DATA
# =============================================================================

print("=" * 60)
print("SECTION 0 — LOAD AND MERGE DATA")
print("=" * 60)

train = pd.read_csv(
    "data/raw/train.csv",
    parse_dates=["Date"],
    dtype={"StateHoliday": str}
)
store = pd.read_csv("data/raw/store.csv")

print(f"train.csv  : {train.shape}  rows × cols")
print(f"store.csv  : {store.shape}  rows × cols")

# Merge store metadata into train on Store ID
df = train.merge(store, on="Store", how="left")
print(f"After merge: {df.shape}")

# ── Remove closed store days ─────────────────────────────
before = len(df)
df = df[df["Open"] == 1].copy()
after  = len(df)
print(f"\nRemoved closed days (Open=0):")
print(f"  Before : {before:,} rows")
print(f"  Removed: {before - after:,} rows")
print(f"  After  : {after:,} rows")

# Sort by Store and Date
df.sort_values(["Store","Date"], inplace=True)
df.reset_index(drop=True, inplace=True)

print(f"\nDate range : {df['Date'].min().date()} → {df['Date'].max().date()}")
print(f"Stores     : {df['Store'].nunique()}")
print(f"\nColumns    : {list(df.columns)}")


# =============================================================================
# SECTION 1 — BASIC OVERVIEW
# =============================================================================

print("\n" + "=" * 60)
print("SECTION 1 — BASIC OVERVIEW")
print("=" * 60)

print("\nData Types:")
print(df.dtypes.to_string())

print("\nMissing Values:")
missing = df.isnull().sum()
missing = missing[missing > 0]
if len(missing) == 0:
    print("  No missing values after merge and filter.")
else:
    print(missing.to_string())
    print(f"\nMissing % of total rows:")
    print((missing / len(df) * 100).round(2).to_string())

print(f"\nBasic Statistics (Sales and Customers):")
print(df[["Sales","Customers"]].describe().round(2).to_string())

print(f"\nStore type distribution:")
print(df.groupby("StoreType")["Store"].nunique().to_string())

print(f"\nStateHoliday value counts:")
print(df["StateHoliday"].value_counts().to_string())

"""
INFERENCE — Section 1:
────────────────────────────────────────────────────────
• CompetitionDistance has missing values → needs imputation
• PromoInterval has many NaN → needs handling in feature engineering
• StateHoliday has 4 categories: 0=none, a=public, b=Easter, c=Christmas
• 4 store types: a, b, c, d — likely different sales profiles
"""


# =============================================================================
# SECTION 2 — TARGET VARIABLE: SALES
# =============================================================================

print("\n" + "=" * 60)
print("SECTION 2 — TARGET VARIABLE: SALES")
print("=" * 60)

print(f"\nSales statistics:")
print(df["Sales"].describe().round(2).to_string())
print(f"\nSkewness : {df['Sales'].skew():.4f}")
print(f"Kurtosis : {df['Sales'].kurtosis():.4f}")

fig, axes = plt.subplots(1, 3, figsize=(16, 5))
fig.suptitle("Plot 1 — Sales Distribution Analysis",
             fontsize=13, color="#e0f0ff", y=1.01)

# Histogram — raw sales
axes[0].hist(df["Sales"], bins=50, color=ACCENT, edgecolor="#0f1117", alpha=0.85)
axes[0].axvline(df["Sales"].mean(),   color=ACCENT3, lw=2,
                linestyle="--", label=f"Mean  {df['Sales'].mean():,.0f}")
axes[0].axvline(df["Sales"].median(), color=ACCENT4, lw=2,
                linestyle="--", label=f"Median {df['Sales'].median():,.0f}")
axes[0].set_title("Raw Sales Distribution", color="#c0d8e8")
axes[0].set_xlabel("Sales (€)")
axes[0].set_ylabel("Frequency")
axes[0].legend(fontsize=8)
axes[0].grid(True)

# Histogram — log sales
log_sales = np.log1p(df["Sales"])   # log1p = log(1+x) handles zeros safely
axes[1].hist(log_sales, bins=50, color=ACCENT3, edgecolor="#0f1117", alpha=0.85)
axes[1].set_title("Log(Sales) Distribution", color="#c0d8e8")
axes[1].set_xlabel("log(Sales)")
axes[1].set_ylabel("Frequency")
axes[1].grid(True)

# Box plot by store type
store_types = sorted(df["StoreType"].unique())
data_by_type = [df[df["StoreType"]==t]["Sales"].values for t in store_types]
bp = axes[2].boxplot(data_by_type, patch_artist=True,
                     labels=store_types,
                     boxprops=dict(facecolor=ACCENT+"44", color=ACCENT),
                     whiskerprops=dict(color=ACCENT),
                     capprops=dict(color=ACCENT),
                     medianprops=dict(color=ACCENT3, linewidth=2),
                     flierprops=dict(markerfacecolor=ACCENT2,
                                     markersize=2, alpha=0.3))
axes[2].set_title("Sales by Store Type", color="#c0d8e8")
axes[2].set_xlabel("Store Type")
axes[2].set_ylabel("Sales (€)")
axes[2].grid(True)

plt.tight_layout()
plt.savefig("plots/eda_plot1_sales_distribution.png", dpi=150, bbox_inches="tight")
plt.show()

"""
INFERENCE — Plot 1 (Sales Distribution):
────────────────────────────────────────────────────────
• Raw Sales is RIGHT SKEWED — log transform needed
• Log(Sales) is approximately normal — confirms log transform is correct
• Store Type b has highest median sales but also most outliers
• Store Type c has lowest and most consistent sales
• Skewness confirms this is a multiplicative dataset like air passengers
"""


# =============================================================================
# SECTION 3 — TREND AND SEASONALITY
# =============================================================================

print("\n" + "=" * 60)
print("SECTION 3 — TREND AND SEASONALITY")
print("=" * 60)

# Aggregate all stores to daily total sales
daily = df.groupby("Date")["Sales"].sum().reset_index()
daily.set_index("Date", inplace=True)

# Weekly average
weekly = df.groupby(pd.Grouper(key="Date", freq="W"))["Sales"].mean()

# Monthly average
monthly = df.groupby(pd.Grouper(key="Date", freq="MS"))["Sales"].mean()

# Day of week average
dow_avg = df.groupby("DayOfWeek")["Sales"].mean()
dow_labels = ["Mon","Tue","Wed","Thu","Fri","Sat","Sun"]

fig, axes = plt.subplots(2, 2, figsize=(16, 10))
fig.suptitle("Plot 2 — Trend and Seasonality Analysis",
             fontsize=13, color="#e0f0ff", y=1.01)

# Daily total sales
axes[0][0].plot(daily.index, daily["Sales"],
                color=ACCENT, lw=1, alpha=0.7)
axes[0][0].set_title("Daily Total Sales (All Stores)",
                      fontsize=10, color="#c0d8e8")
axes[0][0].set_ylabel("Total Sales (€)")
axes[0][0].grid(True)

# Weekly average
axes[0][1].plot(weekly.index, weekly,
                color=ACCENT4, lw=1.5)
axes[0][1].set_title("Weekly Average Sales",
                      fontsize=10, color="#c0d8e8")
axes[0][1].set_ylabel("Avg Sales (€)")
axes[0][1].grid(True)

# Monthly average
axes[1][0].plot(monthly.index, monthly,
                color=ACCENT3, lw=2, marker="o", markersize=4)
axes[1][0].set_title("Monthly Average Sales",
                      fontsize=10, color="#c0d8e8")
axes[1][0].set_ylabel("Avg Sales (€)")
axes[1][0].grid(True)

# Day of week pattern
bars = axes[1][1].bar(dow_labels, dow_avg.values,
                       color=ACCENT5, edgecolor="#0f1117", alpha=0.85)
# Highlight highest day
max_idx = dow_avg.values.argmax()
bars[max_idx].set_color(ACCENT3)
bars[max_idx].set_edgecolor("white")
axes[1][1].set_title("Average Sales by Day of Week",
                      fontsize=10, color="#c0d8e8")
axes[1][1].set_ylabel("Avg Sales (€)")
axes[1][1].grid(True, axis="y")

plt.tight_layout()
plt.savefig("plots/eda_plot2_trend_seasonality.png", dpi=150, bbox_inches="tight")
plt.show()

"""
INFERENCE — Plot 2 (Trend and Seasonality):
────────────────────────────────────────────────────────
• Daily sales: high noise with visible spikes (holiday/promo effects)
• Weekly pattern: clear — sales drop in some weeks, spike near holidays
• Monthly pattern: December is the clear peak (Christmas shopping)
• Day of week: Monday or Friday typically highest, Sunday lowest
• Unlike air passengers, pattern here is NOISIER and less predictable
• Spikes are irregular — promotions and holidays cause sudden jumps
"""


# =============================================================================
# SECTION 4 — ADDITIVE VS MULTIPLICATIVE CHECK
# =============================================================================

print("\n" + "=" * 60)
print("SECTION 4 — ADDITIVE VS MULTIPLICATIVE CHECK")
print("=" * 60)

# Use monthly aggregated series for cleaner decomposition
monthly_series = monthly.copy()
monthly_series = monthly_series.asfreq("MS")

# Rolling mean and std on monthly series
roll_mean = monthly_series.rolling(window=3).mean()
roll_std  = monthly_series.rolling(window=3).std()

# Seasonal decomposition — try multiplicative
try:
    decomp_mul = seasonal_decompose(
        monthly_series.dropna(),
        model="multiplicative",
        period=12
    )
    decomp_add = seasonal_decompose(
        monthly_series.dropna(),
        model="additive",
        period=12
    )
except Exception as e:
    print(f"Decomposition note: {e}")
    decomp_mul = None
    decomp_add = None

fig, axes = plt.subplots(2, 2, figsize=(16, 10))
fig.suptitle("Plot 3 — Additive vs Multiplicative Check",
             fontsize=13, color="#e0f0ff", y=1.01)

# Rolling mean vs std
axes[0][0].plot(monthly_series.index, monthly_series,
                color=ACCENT, lw=1.5, alpha=0.6, label="Monthly Avg Sales")
axes[0][0].plot(roll_mean.index, roll_mean,
                color=ACCENT3, lw=2, label="3M Rolling Mean")
axes[0][0].set_title("Rolling Mean — is it rising?",
                      fontsize=10, color="#c0d8e8")
axes[0][0].legend(fontsize=8)
axes[0][0].grid(True)

axes[0][1].plot(roll_std.index, roll_std,
                color=ACCENT2, lw=2)
axes[0][1].fill_between(roll_std.index, roll_std,
                         alpha=0.2, color=ACCENT2)
axes[0][1].set_title("Rolling Std — does variance grow with mean?",
                      fontsize=10, color="#c0d8e8")
axes[0][1].grid(True)

# Decomposition residuals comparison
if decomp_mul and decomp_add:
    axes[1][0].plot(decomp_add.resid.dropna().index,
                    decomp_add.resid.dropna(),
                    color=ACCENT4, lw=1.5)
    axes[1][0].axhline(0, color="white", lw=0.8,
                        linestyle="--", alpha=0.4)
    axes[1][0].set_title("Additive Residuals\n(random = good fit)",
                          fontsize=10, color="#c0d8e8")
    axes[1][0].grid(True)

    axes[1][1].plot(decomp_mul.resid.dropna().index,
                    decomp_mul.resid.dropna(),
                    color=ACCENT5, lw=1.5)
    axes[1][1].axhline(1, color="white", lw=0.8,
                        linestyle="--", alpha=0.4)
    axes[1][1].set_title("Multiplicative Residuals\n(random around 1 = good fit)",
                          fontsize=10, color="#c0d8e8")
    axes[1][1].grid(True)

plt.tight_layout()
plt.savefig("plots/eda_plot3_additive_vs_multiplicative.png",
            dpi=150, bbox_inches="tight")
plt.show()

"""
INFERENCE — Plot 3 (Additive vs Multiplicative):
────────────────────────────────────────────────────────
• If rolling std grows with rolling mean → MULTIPLICATIVE
• If rolling std stays flat               → ADDITIVE
• Retail sales often show multiplicative behaviour —
  Christmas spike is proportionally larger in high-sales years
• Log transform will be applied to Sales target before modelling
• Unlike air passengers, pattern may not be perfectly one or the other
  due to promotional noise
"""


# =============================================================================
# SECTION 5 — ACF AND PACF
# =============================================================================

print("\n" + "=" * 60)
print("SECTION 5 — ACF AND PACF")
print("=" * 60)

# Use aggregated daily series for ACF/PACF
# Daily data → spikes expected at lag 7 (weekly), 30 (monthly), 365 (yearly)
daily_sales = daily["Sales"].copy()

# Representative stores — one per store type
rep_stores = {}
for stype in store_types:
    store_ids = df[df["StoreType"] == stype]["Store"].unique()
    # Pick store with most complete data
    counts = df[df["Store"].isin(store_ids)].groupby("Store")["Date"].count()
    rep_stores[stype] = counts.idxmax()

print("Representative stores selected:")
for stype, sid in rep_stores.items():
    print(f"  Type {stype} → Store {sid}")

fig, axes = plt.subplots(5, 2, figsize=(16, 20))
fig.suptitle("Plot 4 — ACF and PACF (Aggregated + Representative Stores)",
             fontsize=13, color="#e0f0ff", y=1.01)

# ACF/PACF on aggregated daily series
plot_acf(daily_sales, lags=40, ax=axes[0][0],
         title="ACF — Aggregated Daily Sales",
         color=ACCENT, vlines_kwargs={"colors": ACCENT})
plot_pacf(daily_sales, lags=40, ax=axes[0][1],
          title="PACF — Aggregated Daily Sales",
          color=ACCENT, vlines_kwargs={"colors": ACCENT})

# ACF/PACF for each representative store
colors = [ACCENT3, ACCENT4, ACCENT5, ACCENT2]
for i, (stype, sid) in enumerate(rep_stores.items()):
    store_series = df[df["Store"] == sid].set_index("Date")["Sales"]
    store_series = store_series.asfreq("B", fill_value=store_series.median())

    plot_acf(store_series, lags=40, ax=axes[i+1][0],
             title=f"ACF — Store {sid} (Type {stype})",
             color=colors[i], vlines_kwargs={"colors": colors[i]})
    plot_pacf(store_series, lags=40, ax=axes[i+1][1],
              title=f"PACF — Store {sid} (Type {stype})",
              color=colors[i], vlines_kwargs={"colors": colors[i]})

for ax in axes.flat:
    ax.set_facecolor("#161b27")
    ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig("plots/eda_plot4_acf_pacf.png", dpi=150, bbox_inches="tight")
plt.show()

"""
INFERENCE — Plot 4 (ACF/PACF):
────────────────────────────────────────────────────────
• Aggregated series: strong spike at lag 7 → weekly cycle dominates
• Lag 7, 14, 21, 28 spikes → very strong weekly seasonality
• Each store type has a different autocorrelation structure
• Type b stores may show stronger seasonal patterns
• Unlike air passengers (lag 12 dominant), here lag 7 is dominant
→ DayOfWeek is a critical feature for modelling
→ lag_7 will be the most important lag feature (not lag_12)
"""


# =============================================================================
# SECTION 6 — STORE LEVEL ANALYSIS
# =============================================================================

print("\n" + "=" * 60)
print("SECTION 6 — STORE LEVEL ANALYSIS")
print("=" * 60)

fig, axes = plt.subplots(2, 2, figsize=(16, 10))
fig.suptitle("Plot 5 — Store Level Analysis (Representative Stores)",
             fontsize=13, color="#e0f0ff", y=1.01)

colors_store = [ACCENT, ACCENT3, ACCENT4, ACCENT5]

# Sales over time for each representative store
for (stype, sid), color in zip(rep_stores.items(), colors_store):
    store_ts = df[df["Store"] == sid].set_index("Date")["Sales"]
    axes[0][0].plot(store_ts.index, store_ts,
                    color=color, lw=1, alpha=0.8,
                    label=f"Store {sid} (Type {stype})")

axes[0][0].set_title("Sales Over Time — Representative Stores",
                      fontsize=10, color="#c0d8e8")
axes[0][0].set_ylabel("Sales (€)")
axes[0][0].legend(fontsize=8)
axes[0][0].grid(True)

# Average sales per store type
type_avg = df.groupby("StoreType")["Sales"].mean()
bars = axes[0][1].bar(type_avg.index, type_avg.values,
                       color=colors_store, edgecolor="#0f1117")
for bar, val in zip(bars, type_avg.values):
    axes[0][1].text(bar.get_x() + bar.get_width()/2,
                    bar.get_height() + 50,
                    f"€{val:,.0f}", ha="center",
                    fontsize=9, color="#c0d8e8")
axes[0][1].set_title("Average Sales by Store Type",
                      fontsize=10, color="#c0d8e8")
axes[0][1].set_ylabel("Avg Sales (€)")
axes[0][1].grid(True, axis="y")

# Sales distribution per store type — violin plot
parts = axes[1][0].violinplot(
    [df[df["StoreType"]==t]["Sales"].values for t in store_types],
    positions=range(len(store_types)),
    showmedians=True
)
for pc, color in zip(parts["bodies"], colors_store):
    pc.set_facecolor(color)
    pc.set_alpha(0.5)
parts["cmedians"].set_color(ACCENT3)
axes[1][0].set_xticks(range(len(store_types)))
axes[1][0].set_xticklabels([f"Type {t}" for t in store_types])
axes[1][0].set_title("Sales Distribution by Store Type (Violin)",
                      fontsize=10, color="#c0d8e8")
axes[1][0].set_ylabel("Sales (€)")
axes[1][0].grid(True)

# Number of stores per type
store_counts = df.groupby("StoreType")["Store"].nunique()
axes[1][1].bar(store_counts.index, store_counts.values,
               color=colors_store, edgecolor="#0f1117")
for i, (idx, val) in enumerate(store_counts.items()):
    axes[1][1].text(i, val + 5, str(val),
                    ha="center", fontsize=10, color="#c0d8e8")
axes[1][1].set_title("Number of Stores per Type",
                      fontsize=10, color="#c0d8e8")
axes[1][1].set_ylabel("Store Count")
axes[1][1].grid(True, axis="y")

plt.tight_layout()
plt.savefig("plots/eda_plot5_store_analysis.png", dpi=150, bbox_inches="tight")
plt.show()

"""
INFERENCE — Plot 5 (Store Level):
────────────────────────────────────────────────────────
• Different store types have clearly different sales profiles
• Type b stores have highest average sales but are fewest in number
• Type a stores are most common — drive overall dataset averages
• Each store type needs separate treatment in feature engineering
• StoreType will be an important categorical feature
"""


# =============================================================================
# SECTION 7 — EXTERNAL FEATURES IMPACT
# =============================================================================

print("\n" + "=" * 60)
print("SECTION 7 — EXTERNAL FEATURES IMPACT")
print("=" * 60)

fig, axes = plt.subplots(2, 3, figsize=(18, 10))
fig.suptitle("Plot 6 — External Features Impact on Sales",
             fontsize=13, color="#e0f0ff", y=1.01)

# Promo effect
promo_avg = df.groupby("Promo")["Sales"].mean()
bars = axes[0][0].bar(["No Promo", "Promo"],
                       promo_avg.values,
                       color=[ACCENT2, ACCENT3],
                       edgecolor="#0f1117", width=0.5)
for bar, val in zip(bars, promo_avg.values):
    axes[0][0].text(bar.get_x() + bar.get_width()/2,
                    bar.get_height() + 50,
                    f"€{val:,.0f}", ha="center",
                    fontsize=10, color="#c0d8e8")
lift = (promo_avg[1] - promo_avg[0]) / promo_avg[0] * 100
axes[0][0].set_title(f"Promo Effect\n(+{lift:.1f}% lift)",
                      fontsize=10, color="#c0d8e8")
axes[0][0].set_ylabel("Avg Sales (€)")
axes[0][0].grid(True, axis="y")

# State Holiday effect
holiday_map = {"0": "None", "a": "Public", "b": "Easter", "c": "Christmas"}
holiday_avg = df.groupby("StateHoliday")["Sales"].mean()
holiday_labels = [holiday_map.get(str(k), str(k)) for k in holiday_avg.index]
axes[0][1].bar(holiday_labels, holiday_avg.values,
               color=[ACCENT, ACCENT4, ACCENT5, ACCENT2],
               edgecolor="#0f1117")
axes[0][1].set_title("State Holiday Effect on Sales",
                      fontsize=10, color="#c0d8e8")
axes[0][1].set_ylabel("Avg Sales (€)")
axes[0][1].grid(True, axis="y")

# School Holiday effect
school_avg = df.groupby("SchoolHoliday")["Sales"].mean()
axes[0][2].bar(["No School Holiday", "School Holiday"],
               school_avg.values,
               color=[ACCENT, ACCENT4],
               edgecolor="#0f1117", width=0.5)
lift_school = (school_avg[1] - school_avg[0]) / school_avg[0] * 100
axes[0][2].set_title(f"School Holiday Effect\n({lift_school:+.1f}%)",
                      fontsize=10, color="#c0d8e8")
axes[0][2].set_ylabel("Avg Sales (€)")
axes[0][2].grid(True, axis="y")

# Promo effect by store type
promo_type = df.groupby(["StoreType","Promo"])["Sales"].mean().unstack()
x = np.arange(len(store_types))
width = 0.35
axes[1][0].bar(x - width/2, promo_type[0], width,
               color=ACCENT2, alpha=0.8, label="No Promo")
axes[1][0].bar(x + width/2, promo_type[1], width,
               color=ACCENT3, alpha=0.8, label="Promo")
axes[1][0].set_xticks(x)
axes[1][0].set_xticklabels([f"Type {t}" for t in store_types])
axes[1][0].set_title("Promo Effect by Store Type",
                      fontsize=10, color="#c0d8e8")
axes[1][0].set_ylabel("Avg Sales (€)")
axes[1][0].legend(fontsize=8)
axes[1][0].grid(True, axis="y")

# Day of week by promo
dow_promo = df.groupby(["DayOfWeek","Promo"])["Sales"].mean().unstack()
x = np.arange(7)
axes[1][1].bar(x - width/2, dow_promo[0], width,
               color=ACCENT2, alpha=0.8, label="No Promo")
axes[1][1].bar(x + width/2, dow_promo[1], width,
               color=ACCENT3, alpha=0.8, label="Promo")
axes[1][1].set_xticks(x)
axes[1][1].set_xticklabels(dow_labels)
axes[1][1].set_title("Day of Week × Promo Interaction",
                      fontsize=10, color="#c0d8e8")
axes[1][1].set_ylabel("Avg Sales (€)")
axes[1][1].legend(fontsize=8)
axes[1][1].grid(True, axis="y")

# Monthly sales heatmap (month × year)
df["Year"]  = df["Date"].dt.year
df["Month"] = df["Date"].dt.month
pivot_heat  = df.groupby(["Year","Month"])["Sales"].mean().unstack()
month_abbr  = ["Jan","Feb","Mar","Apr","May","Jun",
                "Jul","Aug","Sep","Oct","Nov","Dec"]
pivot_heat.columns = month_abbr
sns.heatmap(pivot_heat, ax=axes[1][2], cmap="YlOrRd",
            annot=True, fmt=".0f", linewidths=0.3,
            linecolor="#0f1117",
            cbar_kws={"label": "Avg Sales (€)"})
axes[1][2].set_title("Monthly Sales Heatmap (Year × Month)",
                      fontsize=10, color="#c0d8e8")

plt.tight_layout()
plt.savefig("plots/eda_plot6_external_features.png", dpi=150, bbox_inches="tight")
plt.show()

"""
INFERENCE — Plot 6 (External Features):
────────────────────────────────────────────────────────
• Promo has a strong positive effect on sales (typically +20-30%)
• Promo effect varies by store type — some types respond more
• Christmas holiday shows highest sales → December is peak month
• School holidays show a slight positive effect
• Day of week × Promo interaction is important — promo on weekdays
  matters more than promo on weekends
• Heatmap: December clearly darkest every year → strong annual pattern
→ Promo is the single most important external feature
→ Must create interaction feature: Promo × DayOfWeek
"""


# =============================================================================
# SECTION 8 — STORE METADATA ANALYSIS
# =============================================================================

print("\n" + "=" * 60)
print("SECTION 8 — STORE METADATA ANALYSIS")
print("=" * 60)

fig, axes = plt.subplots(1, 3, figsize=(16, 5))
fig.suptitle("Plot 7 — Store Metadata Impact on Sales",
             fontsize=13, color="#e0f0ff", y=1.01)

# Competition Distance vs Sales
# Fill missing CompetitionDistance with median
df["CompetitionDistance"].fillna(
    df["CompetitionDistance"].median(), inplace=True
)
print(f"CompetitionDistance NaN filled with median: "
      f"{df['CompetitionDistance'].median():.0f}m")

# Add trend line
df["dist_bin"] = pd.cut(df["CompetitionDistance"], bins=20)
bin_avg = df.groupby("dist_bin", observed=True)["Sales"].mean()
bin_centers = [interval.mid for interval in bin_avg.index]
axes[0].plot(bin_centers, bin_avg.values,
             color=ACCENT2, lw=2, marker="o",
             markersize=3, label="Avg Sales per bin")
axes[0].set_title("Competition Distance vs Sales",
                   fontsize=10, color="#c0d8e8")
axes[0].set_xlabel("Competition Distance (m)")
axes[0].set_ylabel("Sales (€)")
axes[0].legend(fontsize=8)
axes[0].grid(True)

# Assortment type effect
assortment_avg = df.groupby("Assortment")["Sales"].mean()
assortment_map = {"a": "Basic", "b": "Extra", "c": "Extended"}
labels = [assortment_map.get(k, k) for k in assortment_avg.index]
axes[1].bar(labels, assortment_avg.values,
            color=[ACCENT3, ACCENT4, ACCENT5],
            edgecolor="#0f1117")
axes[1].set_title("Average Sales by Assortment Type",
                   fontsize=10, color="#c0d8e8")
axes[1].set_ylabel("Avg Sales (€)")
axes[1].grid(True, axis="y")

# Promo2 effect (extended promotion)
promo2_avg = df.groupby("Promo2")["Sales"].mean()
axes[2].bar(["No Promo2", "Promo2"],
            promo2_avg.values,
            color=[ACCENT2, ACCENT6],
            edgecolor="#0f1117", width=0.5)
lift2 = (promo2_avg[1] - promo2_avg[0]) / promo2_avg[0] * 100
axes[2].set_title(f"Promo2 (Extended Promo) Effect\n({lift2:+.1f}%)",
                   fontsize=10, color="#c0d8e8")
axes[2].set_ylabel("Avg Sales (€)")
axes[2].grid(True, axis="y")

plt.tight_layout()
plt.savefig("plots/eda_plot7_store_metadata.png", dpi=150, bbox_inches="tight")
plt.show()

"""
INFERENCE — Plot 7 (Store Metadata):
────────────────────────────────────────────────────────
• Closer competition (lower distance) tends to have slightly lower sales
  but effect is noisy — not a clean linear relationship
• Assortment type b (Extra) has highest avg sales despite fewest stores
• Promo2 (extended promotion) effect is mixed — may need interaction
  with specific months when Promo2 is active
• CompetitionDistance NaN rows filled with median — needs to be
  tracked as a separate binary flag in feature engineering
  (was_competition_distance_missing)
"""


# =============================================================================
# SECTION 9 — CORRELATION ANALYSIS
# =============================================================================

print("\n" + "=" * 60)
print("SECTION 9 — CORRELATION ANALYSIS")
print("=" * 60)

# Select numeric columns for correlation
numeric_cols = [
    "Sales", "Customers", "Promo", "SchoolHoliday",
    "DayOfWeek", "CompetitionDistance", "Promo2", "Month", "Year"
]
corr_df = df[numeric_cols].corr()

fig, axes = plt.subplots(1, 2, figsize=(16, 6))
fig.suptitle("Plot 8 — Correlation Analysis",
             fontsize=13, color="#e0f0ff", y=1.01)

# Full correlation heatmap
sns.heatmap(corr_df, ax=axes[0], annot=True, fmt=".2f",
            cmap="coolwarm", linewidths=0.3,
            linecolor="#0f1117", vmin=-1, vmax=1, center=0)
axes[0].set_title("Correlation Matrix", color="#e0f0ff")

# Bar chart — correlation with Sales only
sales_corr = corr_df["Sales"].drop("Sales").sort_values()
colors_corr = [ACCENT3 if v > 0 else ACCENT2 for v in sales_corr.values]
axes[1].barh(sales_corr.index, sales_corr.values,
             color=colors_corr, edgecolor="#0f1117")
axes[1].axvline(0, color="white", lw=1, alpha=0.5)
axes[1].axvline(0.3,  color=ACCENT3, lw=1,
                linestyle="--", alpha=0.5, label="+0.30 threshold")
axes[1].axvline(-0.3, color=ACCENT2, lw=1,
                linestyle="--", alpha=0.5, label="-0.30 threshold")
axes[1].set_title("Feature Correlation with Sales",
                   color="#e0f0ff")
axes[1].set_xlabel("Pearson Correlation")
axes[1].legend(fontsize=8)
axes[1].grid(True, axis="x")

plt.tight_layout()
plt.savefig("plots/eda_plot8_correlation.png", dpi=150, bbox_inches="tight")
plt.show()

print("\nCorrelation with Sales (sorted):")
print(sales_corr.round(3).to_string())

"""
INFERENCE — Plot 8 (Correlation):
────────────────────────────────────────────────────────
• Customers has highest correlation with Sales (expected — more
  customers = more sales). May cause data leakage if used as feature
  since at prediction time we don't know future customer count.
• Promo has strong positive correlation → most important feature
• DayOfWeek has moderate correlation → weekly pattern confirmed
• CompetitionDistance has weak negative correlation
• Year has positive correlation → confirms upward trend over time
→ Customers column should NOT be used as a feature (leakage risk)
→ Promo and DayOfWeek are the two most important features
"""


# =============================================================================
# STATIONARITY CHECK
# =============================================================================

print("\n" + "=" * 60)
print("STATIONARITY CHECK")
print("=" * 60)

def run_adf(series, label):
    result = adfuller(series.dropna(), autolag="AIC")
    stationary = result[1] < 0.05
    print(f"\n  [{label}]")
    print(f"  ADF Statistic : {result[0]:.4f}")
    print(f"  p-value       : {result[1]:.4f}")
    print(f"  Stationary    : {'YES ✓' if stationary else 'NO ✗'}")
    return stationary

run_adf(daily["Sales"],           "Aggregated Daily Sales")
run_adf(np.log1p(daily["Sales"]), "Log(Aggregated Daily Sales)")


# =============================================================================
# FINAL EDA SUMMARY
# =============================================================================

print("""
╔══════════════════════════════════════════════════════════════╗
║          EDA SUMMARY — KEY INFERENCES                       ║
╠══════════════════════════════════════════════════════════════╣
║  Plot 1  Sales Distribution  → Right skewed, log transform  ║
║                                needed, store type b highest  ║
║  Plot 2  Trend & Seasonality → Noisy daily, December peak   ║
║                                strong weekly pattern         ║
║  Plot 3  Additive vs Mult.   → Likely multiplicative        ║
║                                log transform confirmed       ║
║  Plot 4  ACF/PACF            → Lag 7 dominant (weekly)      ║
║                                lag_7 most important feature  ║
║  Plot 5  Store Analysis      → Store types behave very       ║
║                                differently — type matters    ║
║  Plot 6  External Features   → Promo +20-30% sales lift     ║
║                                December clear annual peak    ║
║  Plot 7  Store Metadata      → CompetitionDistance weak neg  ║
║                                Assortment b highest sales    ║
║  Plot 8  Correlation         → Customers = leakage risk      ║
║                                Promo + DayOfWeek most imp.   ║
╠══════════════════════════════════════════════════════════════╣
║  KEY DECISIONS FOR FEATURE ENGINEERING:                      ║
║  • Remove Customers column (data leakage)                    ║
║  • Log transform Sales target                                ║
║  • lag_7 most important lag (weekly cycle)                   ║
║  • DayOfWeek critical calendar feature                       ║
║  • Promo is the strongest external feature                   ║
║  • Create Promo × DayOfWeek interaction feature              ║
║  • Flag missing CompetitionDistance as binary column         ║
║  • StoreType as categorical feature                          ║
╚══════════════════════════════════════════════════════════════╝
""")