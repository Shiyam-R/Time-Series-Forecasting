"""
14_october_spike_analysis.py
─────────────────────────────────────────────────────────────────────────────
Root-cause drill-down of the October RMSPE spike across six lenses,
plus a StoreType × Month analysis:

  1. Store-level contribution   (which stores are driving October error?)
  2. Day-of-week profile        (weekend / long-weekend effect?)
  3. School holiday coincidence (Herbstferien overlap?)
  4. Promo association          (does Promo interact with holiday-period error?)
  5. State holiday alignment    (German Unity Day Oct-3 footprint)
  6. StoreType × Assortment     (which store segments are most affected?)
  7. Combined profile           (top-N worst October stores vs rest)
  8. StoreType × Month          (are spikes concentrated in specific store types?)

Requires:
  RESULTS_CSV   : fair-eval results from script 12
  TRAIN_CSV     : original Rossmann train.csv  (Promo / StateHoliday / SchoolHoliday)
  STORE_CSV     : original Rossmann store.csv  (StoreType / Assortment)
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from pathlib import Path
import warnings
warnings.filterwarnings("ignore")


# ── CONFIG ───────────────────────────────────────────────────────────────────
RESULTS_CSV  = Path("data/processed/fair_eval_results.csv")
TRAIN_CSV    = Path("data/raw/train.csv")
STORE_CSV    = Path("data/raw/store.csv")
FIGURES_DIR  = Path("figures/october_drill_down")
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

BUCKET_ORDER = ["near", "mid", "far", "extended"]
BUCKET_RANGES = {"near": (1, 14), "mid": (15, 30), "far": (31, 60), "extended": (61, 90)}
TOP_N_STORES = 20

DOW_LABELS = {1: "Mon", 2: "Tue", 3: "Wed", 4: "Thu", 5: "Fri", 6: "Sat", 7: "Sun"}
MONTH_NAMES = ["Jan","Feb","Mar","Apr","May","Jun","Jul","Aug","Sep","Oct","Nov","Dec"]


# ── HELPERS ──────────────────────────────────────────────────────────────────
def rmspe(actual: np.ndarray, predicted: np.ndarray) -> float:
    """RMSPE %, closed days excluded. Returns nan if no valid rows."""
    mask = np.asarray(actual) > 0
    if mask.sum() == 0:
        return np.nan
    a, p = np.asarray(actual)[mask], np.asarray(predicted)[mask]
    return float(np.sqrt(np.mean(((a - p) / a) ** 2)) * 100)


def oct_df(df: pd.DataFrame) -> pd.DataFrame:
    return df[df["month"] == 10].copy()


# ── DATA LOADING + FEATURE JOIN ───────────────────────────────────────────────
def load_data() -> pd.DataFrame:
    # ── Results CSV ──────────────────────────────────────────────────────────
    print("Loading results …")
    res = pd.read_csv(RESULTS_CSV, low_memory=False)
    res.columns = res.columns.str.lower().str.strip().str.replace(" ", "_")

    aliases = {
        "actual":          "y_actual",
        "sales":           "y_actual",
        "actual_sales":    "y_actual",
        "predicted":       "y_pred",
        "pred":            "y_pred",
        "predicted_sales": "y_pred",
        "bucket":          "horizon_bucket",
        "date":            "forecast_date",
        "target_date":     "forecast_date",
    }
    res = res.rename(columns={k: v for k, v in aliases.items() if k in res.columns})

    res["forecast_date"] = pd.to_datetime(res["forecast_date"])
    res["store"]         = res["store"].astype(int)
    res["horizon_bucket"] = pd.Categorical(
        res["horizon_bucket"].str.lower().str.strip(),
        categories=BUCKET_ORDER, ordered=True
    )
    res = res[res["y_actual"] > 0].copy()

    # ── Join Rossmann train features ─────────────────────────────────────────
    print("Joining train features (Promo / StateHoliday / SchoolHoliday) …")
    train = pd.read_csv(TRAIN_CSV, parse_dates=["Date"], low_memory=False)
    train.columns = train.columns.str.lower()
    train = train.rename(columns={"store": "store", "date": "forecast_date"})
    train_feats = train[["store", "forecast_date",
                          "promo", "stateholiday", "schoolholiday"]].copy()
    res = res.merge(train_feats, on=["store", "forecast_date"], how="left")

    res["stateholiday"] = (
        res["stateholiday"]
        .astype(str).str.strip()
        .replace({"0": "none", "0.0": "none", "nan": "none",
                  "a": "public", "b": "easter", "c": "christmas"})
        .fillna("none")
    )
    res["is_state_holiday"]  = res["stateholiday"] != "none"
    res["is_public_holiday"] = res["stateholiday"] == "public"
    res["is_school_holiday"] = res["schoolholiday"].fillna(0).astype(bool)
    res["is_promo"]          = res["promo"].fillna(0).astype(bool)

    res["dow"]        = res["forecast_date"].dt.dayofweek + 1   # 1=Mon … 7=Sun
    res["dow_label"]  = res["dow"].map(DOW_LABELS)
    res["is_weekend"] = res["dow"] >= 6
    res["month"]      = res["forecast_date"].dt.month
    res["month_name"] = res["forecast_date"].dt.strftime("%b")

    # ── Join store metadata ───────────────────────────────────────────────────
    print("Joining store metadata (StoreType / Assortment) …")
    store = pd.read_csv(STORE_CSV, low_memory=False)
    store.columns = store.columns.str.lower()
    store = store.rename(columns={"store": "store"})
    res = res.merge(store[["store", "storetype", "assortment"]],
                    on="store", how="left")

    res["spe"] = ((res["y_actual"] - res["y_pred"]) / res["y_actual"]) ** 2

    n_oct = (res["month"] == 10).sum()
    print(f"\nReady: {len(res):,} rows total  |  {n_oct:,} in October")
    print("StateHoliday distribution in October:")
    print(res[res["month"] == 10]["stateholiday"].value_counts().to_string())
    return res


# ── ANALYSIS 1: Store-Level Contribution ─────────────────────────────────────
def analysis_1_store_contribution(df: pd.DataFrame) -> pd.DataFrame:
    print("\n" + "═"*60)
    print("ANALYSIS 1 — Store-Level October Error Contribution")
    print("═"*60)

    oct = oct_df(df)
    total_oct_spe = oct["spe"].sum()

    store_stats = (
        oct.groupby("store")
           .apply(lambda g: pd.Series({
               "n":          len(g),
               "spe_sum":    g["spe"].sum(),
               "rmspe_pct":  rmspe(g["y_actual"].values, g["y_pred"].values),
               "storetype":  g["storetype"].iloc[0] if "storetype" in g.columns else None,
               "assortment": g["assortment"].iloc[0] if "assortment" in g.columns else None,
           }))
           .reset_index()
    )
    store_stats["contrib_pct"] = store_stats["spe_sum"] / total_oct_spe * 100
    store_stats = store_stats.sort_values("contrib_pct", ascending=False).reset_index(drop=True)

    top = store_stats.head(TOP_N_STORES)
    print(f"\nTop {TOP_N_STORES} stores by October error contribution:")
    print(top[["store", "storetype", "assortment", "rmspe_pct", "contrib_pct", "n"]]
          .rename(columns={"rmspe_pct": "OctRMSPE%", "contrib_pct": "Contrib%"})
          .to_string(index=False))

    cum_top = store_stats.head(TOP_N_STORES)["contrib_pct"].sum()
    cum_50  = store_stats.head(50)["contrib_pct"].sum()
    print(f"\nTop-{TOP_N_STORES} stores → {cum_top:.1f}% of October squared error")
    print(f"Top-50  stores → {cum_50:.1f}%")

    type_colors = {"a": "#2196F3", "b": "#EF5350", "c": "#4CAF50", "d": "#FF9800"}
    fig, axes = plt.subplots(1, 2, figsize=(16, 8))

    ax = axes[0]
    bar_colors = [type_colors.get(str(t), "#9E9E9E") for t in top["storetype"]]
    ax.barh(range(TOP_N_STORES), top["contrib_pct"].values, color=bar_colors, alpha=0.85)
    ax.set_yticks(range(TOP_N_STORES))
    ax.set_yticklabels([f"Store {int(s)}" for s in top["store"]], fontsize=8)
    ax.invert_yaxis()
    ax.set_xlabel("% of Total October Squared Error", fontsize=10)
    ax.set_title(f"Top {TOP_N_STORES} Stores — October Error Contribution\n"
                 "(colour = StoreType)", fontweight="bold")
    for t, c in type_colors.items():
        ax.barh([], [], color=c, label=f"Type {t}")
    ax.legend(fontsize=9, title="StoreType")
    ax.grid(axis="x", alpha=0.25)

    ax = axes[1]
    bottom_n = store_stats.tail(len(store_stats) - TOP_N_STORES)
    types = sorted(type_colors.keys())
    x_pos = np.arange(len(types))
    for offset, subset, label, color in [
        (0.0, top,      f"Top {TOP_N_STORES}", "#EF5350"),
        (0.4, bottom_n, "Rest",                "#42A5F5"),
    ]:
        vc     = subset["storetype"].value_counts()
        counts = [vc.get(t, 0) for t in types]
        total  = max(sum(counts), 1)
        pcts   = [c / total * 100 for c in counts]
        ax.bar(x_pos + offset * 0.4, pcts, 0.38,
               label=label, color=color, alpha=0.85)
    ax.set_xticks(x_pos + 0.2)
    ax.set_xticklabels([f"Type {t}" for t in types], fontsize=10)
    ax.set_ylabel("% of stores in group")
    ax.yaxis.set_major_formatter(mticker.PercentFormatter())
    ax.set_title("StoreType Mix: Top-N vs Rest of Stores", fontweight="bold")
    ax.legend(fontsize=9); ax.grid(axis="y", alpha=0.25)

    plt.suptitle("Store-Level October Error Contribution", fontsize=13, fontweight="bold")
    plt.tight_layout()
    fig.savefig(FIGURES_DIR / "1_store_october_contribution.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("Saved: 1_store_october_contribution.png")
    return store_stats


# ── ANALYSIS 2: Day-of-Week Effect ───────────────────────────────────────────
def analysis_2_dayofweek(df: pd.DataFrame):
    print("\n" + "═"*60)
    print("ANALYSIS 2 — Day-of-Week Profile in October")
    print("═"*60)
    print("\n  German Unity Day 2014 = Friday Oct 3")

    rows = []
    for month, month_label in [(9, "September"), (10, "October"), (11, "November")]:
        for dow in range(1, 8):
            sub = df[(df["month"] == month) & (df["dow"] == dow)]
            if len(sub) >= 10:
                rows.append({
                    "month":       month,
                    "month_label": month_label,
                    "dow":         dow,
                    "dow_label":   DOW_LABELS[dow],
                    "rmspe":       rmspe(sub["y_actual"].values, sub["y_pred"].values),
                    "n":           len(sub),
                })
    res = pd.DataFrame(rows)

    print("\nRMSPE by Day-of-Week — Sep / Oct / Nov:")
    pivot = (res.pivot(index="dow_label", columns="month_label", values="rmspe")
               .reindex([DOW_LABELS[i] for i in range(1, 8)]))
    print(pivot.round(1).to_string())

    fig, ax = plt.subplots(figsize=(11, 5))
    colors_m = {"September": "#42A5F5", "October": "#EF5350", "November": "#66BB6A"}
    x = np.arange(7)
    w = 0.25
    for i, (month_label, color) in enumerate(colors_m.items()):
        sub = res[res["month_label"] == month_label].sort_values("dow")
        if sub.empty:
            continue
        x_pos = sub["dow"].values - 1
        ax.bar(x_pos + (i-1)*w, sub["rmspe"].values, w,
               label=month_label, color=color, alpha=0.85)
    ax.set_xticks(x)
    ax.set_xticklabels([DOW_LABELS[i] for i in range(1, 8)], fontsize=10)
    ax.set_ylabel("RMSPE (%)", fontsize=11)
    ax.yaxis.set_major_formatter(mticker.PercentFormatter())
    ax.set_title("RMSPE by Day of Week — Sep / Oct / Nov\n"
                 "(Unity Day 2014 = Friday Oct 3 → watch Thu / Fri / Sat in Oct)",
                 fontsize=12, fontweight="bold")
    ax.legend(fontsize=10); ax.grid(axis="y", alpha=0.25)
    plt.tight_layout()
    fig.savefig(FIGURES_DIR / "2_dayofweek_october.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("Saved: 2_dayofweek_october.png")
    return res


# ── ANALYSIS 3: School Holiday Coincidence ────────────────────────────────────
def analysis_3_schoolholiday(df: pd.DataFrame):
    print("\n" + "═"*60)
    print("ANALYSIS 3 — School Holiday Coincidence (Herbstferien)")
    print("═"*60)

    oct_sch_pct = df[df["month"] == 10]["is_school_holiday"].mean() * 100
    all_sch_pct = df["is_school_holiday"].mean() * 100
    print(f"\nSchool holiday prevalence: October = {oct_sch_pct:.1f}%  |  "
          f"Annual mean = {all_sch_pct:.1f}%")

    rows = []
    for m_idx, m_name in enumerate(MONTH_NAMES, start=1):
        for is_sch in [False, True]:
            sub = df[(df["month"] == m_idx) & (df["is_school_holiday"] == is_sch)]
            if len(sub) >= 20:
                rows.append({
                    "month":          m_idx,
                    "month_name":     m_name,
                    "school_holiday": is_sch,
                    "label":  "SchoolHoliday=1" if is_sch else "SchoolHoliday=0",
                    "rmspe":  rmspe(sub["y_actual"].values, sub["y_pred"].values),
                    "n":      len(sub),
                })
    res = pd.DataFrame(rows)

    print("\nOctober breakdown:")
    for _, row in res[res["month"] == 10].iterrows():
        print(f"  {row['label']:<22}: {row['rmspe']:.1f}%  (n={row['n']:,})")

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    ax = axes[0]
    for is_sch, label, color in [(False, "No school holiday", "#42A5F5"),
                                   (True,  "School holiday",    "#EF5350")]:
        sub = res[res["school_holiday"] == is_sch].sort_values("month")
        if not sub.empty:
            ax.plot(sub["month_name"], sub["rmspe"], marker="o",
                    linewidth=2, color=color, label=label, markersize=6)
    ax.set_ylabel("RMSPE (%)", fontsize=11)
    ax.yaxis.set_major_formatter(mticker.PercentFormatter())
    ax.set_title("RMSPE by Month: School Holiday vs Not", fontweight="bold")
    ax.legend(fontsize=9); ax.grid(alpha=0.22)
    ax.tick_params(axis="x", rotation=45, labelsize=8)

    ax = axes[1]
    sch_rate = df.groupby("month")["is_school_holiday"].mean().reset_index()
    sch_rate["month_name"] = sch_rate["month"].map(
        {i+1: m for i, m in enumerate(MONTH_NAMES)}
    )
    sch_rate = sch_rate.sort_values("month")
    ax.bar(sch_rate["month_name"], sch_rate["is_school_holiday"] * 100,
           color="#EF5350", alpha=0.75)
    ax.axhline(all_sch_pct, color="gray", linestyle="--",
               linewidth=1.2, label=f"Annual mean {all_sch_pct:.1f}%")
    ax.set_ylabel("% of days with SchoolHoliday=1", fontsize=10)
    ax.yaxis.set_major_formatter(mticker.PercentFormatter())
    ax.set_title("School Holiday Prevalence per Month", fontweight="bold")
    ax.legend(fontsize=9)
    ax.tick_params(axis="x", rotation=45, labelsize=8)
    ax.grid(axis="y", alpha=0.25)

    plt.suptitle("School Holiday (Herbstferien) Coincidence", fontsize=13, fontweight="bold")
    plt.tight_layout()
    fig.savefig(FIGURES_DIR / "3_schoolholiday.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("Saved: 3_schoolholiday.png")
    return res


# ── ANALYSIS 4: Promo Association ────────────────────────────────────────────
def analysis_4_promo(df: pd.DataFrame):
    print("\n" + "═"*60)
    print("ANALYSIS 4 — Promo Association with October Spike")
    print("═"*60)

    oct_promo_pct = df[df["month"] == 10]["is_promo"].mean() * 100
    all_promo_pct = df["is_promo"].mean() * 100
    print(f"\nPromo prevalence: October = {oct_promo_pct:.1f}%  |  "
          f"Annual mean = {all_promo_pct:.1f}%")

    rows = []
    for m_idx, m_name in enumerate(MONTH_NAMES, start=1):
        for is_promo in [False, True]:
            sub = df[(df["month"] == m_idx) & (df["is_promo"] == is_promo)]
            if len(sub) >= 20:
                rows.append({
                    "month":      m_idx,
                    "month_name": m_name,
                    "promo":      is_promo,
                    "label":      "Promo=1" if is_promo else "Promo=0",
                    "rmspe":      rmspe(sub["y_actual"].values, sub["y_pred"].values),
                    "n":          len(sub),
                })
    res = pd.DataFrame(rows)

    print("\nOctober Promo breakdown:")
    for _, row in res[res["month"] == 10].iterrows():
        print(f"  {row['label']:<10}: {row['rmspe']:.1f}%  (n={row['n']:,})")

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    ax = axes[0]
    for is_promo, label, color in [(False, "Promo=0", "#42A5F5"),
                                    (True,  "Promo=1", "#FF9800")]:
        sub = res[res["promo"] == is_promo].sort_values("month")
        if not sub.empty:
            ax.plot(sub["month_name"], sub["rmspe"], marker="o",
                    linewidth=2, color=color, label=label, markersize=6)
    ax.set_ylabel("RMSPE (%)", fontsize=11)
    ax.yaxis.set_major_formatter(mticker.PercentFormatter())
    ax.set_title("RMSPE by Month: Promo vs No Promo", fontweight="bold")
    ax.legend(fontsize=9); ax.grid(alpha=0.22)
    ax.tick_params(axis="x", rotation=45, labelsize=8)

    ax = axes[1]
    promo_rate = df.groupby("month")["is_promo"].mean().reset_index()
    promo_rate["month_name"] = promo_rate["month"].map(
        {i+1: m for i, m in enumerate(MONTH_NAMES)}
    )
    promo_rate = promo_rate.sort_values("month")
    ax.bar(promo_rate["month_name"], promo_rate["is_promo"] * 100,
           color="#FF9800", alpha=0.78)
    ax.axhline(all_promo_pct, color="gray", linestyle="--",
               linewidth=1.2, label=f"Annual mean {all_promo_pct:.1f}%")
    ax.set_ylabel("% of days with Promo=1", fontsize=10)
    ax.yaxis.set_major_formatter(mticker.PercentFormatter())
    ax.set_title("Promo Prevalence per Month", fontweight="bold")
    ax.legend(fontsize=9)
    ax.tick_params(axis="x", rotation=45, labelsize=8)
    ax.grid(axis="y", alpha=0.25)

    plt.suptitle("Promo Association with October Error", fontsize=13, fontweight="bold")
    plt.tight_layout()
    fig.savefig(FIGURES_DIR / "4_promo_association.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("Saved: 4_promo_association.png")
    return res


# ── ANALYSIS 5: State Holiday Alignment ──────────────────────────────────────
def analysis_5_stateholiday(df: pd.DataFrame):
    print("\n" + "═"*60)
    print("ANALYSIS 5 — State Holiday Alignment (German Unity Day Focus)")
    print("═"*60)

    print("\nRMSPE by StateHoliday type — all months:")
    for hol_type in ["none", "public", "easter", "christmas"]:
        sub = df[df["stateholiday"] == hol_type]
        if len(sub) >= 10:
            print(f"  {hol_type:10s}: {rmspe(sub['y_actual'].values, sub['y_pred'].values):6.1f}%  "
                  f"(n={len(sub):,})")

    print("\nOctober StateHoliday breakdown:")
    oct = oct_df(df)
    for hol_type in sorted(oct["stateholiday"].unique()):
        sub = oct[oct["stateholiday"] == hol_type]
        if len(sub) >= 5:
            print(f"  {hol_type:10s}: {rmspe(sub['y_actual'].values, sub['y_pred'].values):6.1f}%  "
                  f"(n={len(sub):,})")

    oct = oct.copy()
    oct["day_of_month"] = oct["forecast_date"].dt.day
    day_stats = []
    for d in range(1, 32):
        sub = oct[oct["day_of_month"] == d]
        if len(sub) >= 5:
            day_stats.append({
                "day":          d,
                "rmspe":        rmspe(sub["y_actual"].values, sub["y_pred"].values),
                "stateholiday": sub["stateholiday"].mode().iloc[0],
                "n":            len(sub),
            })
    day_df = pd.DataFrame(day_stats)

    if not day_df.empty:
        print("\nTop 5 worst days in October:")
        print(day_df.nlargest(5, "rmspe")[["day", "rmspe", "stateholiday", "n"]]
              .to_string(index=False))

    fig, axes = plt.subplots(1, 2, figsize=(15, 5))

    if not day_df.empty:
        ax = axes[0]
        hol_color_map = {"none": "#42A5F5", "public": "#EF5350",
                         "easter": "#FF9800", "christmas": "#9C27B0"}
        colors_day = [hol_color_map.get(h, "#42A5F5") for h in day_df["stateholiday"]]
        ax.bar(day_df["day"], day_df["rmspe"], color=colors_day, alpha=0.85)
        if 3 in day_df["day"].values:
            d3 = day_df[day_df["day"] == 3]["rmspe"].iloc[0]
            ax.axvline(3, color="darkred", linestyle="--", linewidth=1.5)
            ax.text(3, d3 + 1, "Oct 3\n(Unity Day)", ha="center",
                    fontsize=8, color="darkred", fontweight="bold")
        ax.set_xlabel("Day of October", fontsize=10)
        ax.set_ylabel("RMSPE (%)", fontsize=10)
        ax.yaxis.set_major_formatter(mticker.PercentFormatter())
        ax.set_title("RMSPE per Day of October\n(red bars = public holiday)",
                     fontweight="bold")
        ax.grid(axis="y", alpha=0.25)
        for label, color in [("Normal day", "#42A5F5"), ("Public holiday", "#EF5350")]:
            ax.bar([], [], color=color, label=label)
        ax.legend(fontsize=9)

    ax = axes[1]
    oct["days_from_unity"] = oct["day_of_month"] - 3
    window_stats = []
    for d in range(-7, 8):
        sub = oct[oct["days_from_unity"] == d]
        if len(sub) >= 5:
            window_stats.append({
                "offset": d,
                "rmspe":  rmspe(sub["y_actual"].values, sub["y_pred"].values),
                "n":      len(sub),
            })
    if window_stats:
        ws = pd.DataFrame(window_stats)
        bar_clr = ["#EF5350" if d == 0 else "#90CAF9" for d in ws["offset"]]
        ax.bar(ws["offset"], ws["rmspe"], color=bar_clr, alpha=0.88)
        ax.axvline(0, color="darkred", linestyle="--", linewidth=1.5)
        ax.set_xlabel("Days from German Unity Day (Oct 3 = 0)", fontsize=10)
        ax.set_ylabel("RMSPE (%)", fontsize=10)
        ax.yaxis.set_major_formatter(mticker.PercentFormatter())
        ax.set_title("RMSPE in ±7-Day Window Around German Unity Day",
                     fontweight="bold")
        ax.grid(axis="y", alpha=0.25)

    plt.suptitle("State Holiday Alignment — German Unity Day Oct 3",
                 fontsize=13, fontweight="bold")
    plt.tight_layout()
    fig.savefig(FIGURES_DIR / "5_stateholiday_alignment.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("Saved: 5_stateholiday_alignment.png")
    return day_df


# ── ANALYSIS 6: StoreType × Assortment ───────────────────────────────────────
def analysis_6_storetype_assortment(df: pd.DataFrame):
    print("\n" + "═"*60)
    print("ANALYSIS 6 — StoreType × Assortment Breakdown")
    print("═"*60)

    oct = oct_df(df)
    store_types = sorted(df["storetype"].dropna().unique())
    assortments = sorted(df["assortment"].dropna().unique())

    print("\nOctober vs Annual RMSPE by StoreType:")
    print(f"  {'Type':<6} {'Oct RMSPE':>10} {'Annual':>8} {'Oct Premium':>12}")
    print("  " + "─"*40)
    for st in store_types:
        sub_o = oct[oct["storetype"] == st]
        sub_a = df[df["storetype"] == st]
        if len(sub_o) < 10:
            continue
        r_oct = rmspe(sub_o["y_actual"].values, sub_o["y_pred"].values)
        r_ann = rmspe(sub_a["y_actual"].values, sub_a["y_pred"].values)
        print(f"  Type {st}   {r_oct:>9.1f}%  {r_ann:>7.1f}%  {r_oct - r_ann:>+10.1f}pp")

    records = []
    for st in store_types:
        for asmnt in assortments:
            sub_o = oct[(oct["storetype"] == st) & (oct["assortment"] == asmnt)]
            sub_a = df[(df["storetype"] == st)  & (df["assortment"]  == asmnt)]
            if len(sub_o) >= 20:
                r_o = rmspe(sub_o["y_actual"].values, sub_o["y_pred"].values)
                r_a = rmspe(sub_a["y_actual"].values, sub_a["y_pred"].values)
                records.append({
                    "storetype":    st,
                    "assortment":   asmnt,
                    "rmspe_oct":    r_o,
                    "rmspe_annual": r_a,
                    "oct_premium":  r_o - r_a,
                    "n_oct":        len(sub_o),
                })
    res = pd.DataFrame(records)
    if not res.empty:
        print("\nStoreType × Assortment — October premium (worst first):")
        print(res[["storetype", "assortment", "rmspe_oct",
                   "rmspe_annual", "oct_premium", "n_oct"]]
              .sort_values("oct_premium", ascending=False)
              .to_string(index=False))

    fig, axes = plt.subplots(1, 2, figsize=(15, 5))

    ax = axes[0]
    x = np.arange(len(store_types))
    w = 0.38
    oct_vals, ann_vals = [], []
    for st in store_types:
        sub_o = oct[oct["storetype"] == st]
        sub_a = df[df["storetype"] == st]
        oct_vals.append(rmspe(sub_o["y_actual"].values, sub_o["y_pred"].values) if len(sub_o) >= 5 else np.nan)
        ann_vals.append(rmspe(sub_a["y_actual"].values, sub_a["y_pred"].values) if len(sub_a) >= 5 else np.nan)
    b1 = ax.bar(x - w/2, ann_vals, w, label="Annual",  color="#90CAF9", alpha=0.9)
    b2 = ax.bar(x + w/2, oct_vals, w, label="October", color="#EF5350", alpha=0.9)
    for bar in list(b1) + list(b2):
        v = bar.get_height()
        if not np.isnan(v):
            ax.text(bar.get_x() + bar.get_width()/2, v + 0.3,
                    f"{v:.0f}%", ha="center", va="bottom", fontsize=8)
    ax.set_xticks(x)
    ax.set_xticklabels([f"Type {t}" for t in store_types], fontsize=10)
    ax.set_ylabel("RMSPE (%)", fontsize=10)
    ax.yaxis.set_major_formatter(mticker.PercentFormatter())
    ax.set_title("Annual vs October RMSPE by StoreType", fontweight="bold")
    ax.legend(fontsize=9); ax.grid(axis="y", alpha=0.25)

    ax = axes[1]
    if not res.empty:
        pivot = res.pivot(index="storetype", columns="assortment", values="rmspe_oct")
        im = ax.imshow(pivot.values, aspect="auto", cmap="RdYlGn_r", vmin=5, vmax=60)
        ax.set_xticks(range(len(pivot.columns)))
        ax.set_xticklabels([f"Assortment {a}" for a in pivot.columns], fontsize=9)
        ax.set_yticks(range(len(pivot.index)))
        ax.set_yticklabels([f"Type {t}" for t in pivot.index], fontsize=9)
        for i in range(pivot.values.shape[0]):
            for j in range(pivot.values.shape[1]):
                v = pivot.values[i, j]
                if not np.isnan(v):
                    ax.text(j, i, f"{v:.0f}%", ha="center", va="center",
                            fontsize=10, fontweight="bold",
                            color="white" if v > 42 else "black")
        plt.colorbar(im, ax=ax, label="October RMSPE (%)")
        ax.set_title("October RMSPE: StoreType × Assortment", fontweight="bold")

    plt.suptitle("StoreType × Assortment — October Breakdown",
                 fontsize=13, fontweight="bold")
    plt.tight_layout()
    fig.savefig(FIGURES_DIR / "6_storetype_assortment.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("Saved: 6_storetype_assortment.png")
    return res


# ── ANALYSIS 7: Combined Profile — Worst-N October Stores ────────────────────
def combined_profile(df: pd.DataFrame, store_stats: pd.DataFrame):
    print("\n" + "═"*60)
    print(f"COMBINED PROFILE — Top-{TOP_N_STORES} October Stores Across All Months")
    print("═"*60)

    top_ids = set(store_stats.head(TOP_N_STORES)["store"])
    df = df.copy()
    df["is_top_oct"] = df["store"].isin(top_ids)

    print(f"\nOctober feature rates — Top-{TOP_N_STORES} vs Rest:")
    print(f"  {'Feature':<24} {'Top-N':>8} {'Rest':>8}")
    print("  " + "─"*44)
    for feat, label in [
        ("is_promo",          "Promo rate"),
        ("is_school_holiday", "SchoolHoliday rate"),
        ("is_public_holiday", "StateHoliday=public"),
        ("is_weekend",        "Weekend rate"),
    ]:
        oct_top  = df[df["is_top_oct"]  & (df["month"] == 10)][feat].mean() * 100
        oct_rest = df[~df["is_top_oct"] & (df["month"] == 10)][feat].mean() * 100
        print(f"  {label:<24} {oct_top:>7.1f}%  {oct_rest:>7.1f}%")

    fig, ax = plt.subplots(figsize=(13, 5))
    for flag, label, color, lw in [
        (True,  f"Top-{TOP_N_STORES} October stores", "#EF5350", 2.5),
        (False, "Rest of stores",                      "#42A5F5", 2.0),
    ]:
        pts = []
        for m_idx, m_name in enumerate(MONTH_NAMES, start=1):
            sub = df[(df["is_top_oct"] == flag) & (df["month"] == m_idx)]
            if len(sub) >= 20:
                pts.append({
                    "m_idx": m_idx, "m": m_name,
                    "rmspe": rmspe(sub["y_actual"].values, sub["y_pred"].values),
                })
        if pts:
            tmp = pd.DataFrame(pts).sort_values("m_idx")
            ax.plot(tmp["m"], tmp["rmspe"], marker="o", linewidth=lw,
                    color=color, label=label, markersize=7)

    ax.set_ylabel("RMSPE (%)", fontsize=11)
    ax.yaxis.set_major_formatter(mticker.PercentFormatter())
    ax.set_title(
        f"Monthly RMSPE: Top-{TOP_N_STORES} October Stores vs Rest\n"
        "Gap only in October → calendar/feature fix.  "
        "Gap year-round → those stores need store-specific treatment.",
        fontsize=11, fontweight="bold"
    )
    ax.legend(fontsize=10); ax.grid(alpha=0.22)
    ax.tick_params(axis="x", rotation=45, labelsize=9)
    plt.tight_layout()
    fig.savefig(FIGURES_DIR / "7_worst_stores_monthly_profile.png",
                dpi=150, bbox_inches="tight")
    plt.close()
    print("Saved: 7_worst_stores_monthly_profile.png")


# ── ANALYSIS 8: StoreType × Month RMSPE and Error Contribution ───────────────
def analysis_storetype_monthly(df: pd.DataFrame):
    """
    Investigates whether monthly error spikes (Apr / Oct / Nov / Dec) are
    concentrated in specific store types or broadly uniform across all types.

    Key distinction:
      RMSPE heatmap       → model quality per prediction (volume-neutral)
      Contribution heatmap → total error mass (amplified by store count)

    If both heatmaps agree on Type A → genuine model quality problem.
    If only contribution is high → volume effect, not accuracy problem.
    """
    print("\n" + "═"*60)
    print("ANALYSIS 8 — StoreType × Month RMSPE and Error Contribution")
    print("═"*60)

    if "storetype" not in df.columns:
        print("⚠️  'storetype' column missing — was the store.csv join successful?")
        return None

    store_types = sorted(df["storetype"].dropna().unique())
    total_spe   = df["spe"].sum()
    SPIKE_MONTHS = {4: "Apr", 10: "Oct", 11: "Nov", 12: "Dec"}

    # ── Build StoreType × Month records ───────────────────────────────────────
    records = []
    for st in store_types:
        for m_idx, m_name in enumerate(MONTH_NAMES, start=1):
            sub = df[(df["storetype"] == st) & (df["month"] == m_idx)]
            if len(sub) >= 20:
                records.append({
                    "storetype":   st,
                    "month":       m_idx,
                    "month_name":  m_name,
                    "rmspe":       rmspe(sub["y_actual"].values, sub["y_pred"].values),
                    "contrib_pct": sub["spe"].sum() / total_spe * 100,
                    "n":           len(sub),
                })
    res = pd.DataFrame(records)

    if res.empty:
        print("⚠️  No data returned — check that storetype column is populated.")
        return None

    month_cols    = [m for m in MONTH_NAMES if m in res["month_name"].unique()]
    pivot_rmspe   = (res.pivot(index="storetype", columns="month_name", values="rmspe")
                       .reindex(index=store_types)[month_cols])
    pivot_contrib = (res.pivot(index="storetype", columns="month_name", values="contrib_pct")
                       .reindex(index=store_types)[month_cols])

    # ── Overall ranking ────────────────────────────────────────────────────────
    ranking_rows = []
    for st in store_types:
        sub = df[df["storetype"] == st]
        r   = rmspe(sub["y_actual"].values, sub["y_pred"].values)
        c   = sub["spe"].sum() / total_spe * 100
        n_stores     = sub["store"].nunique() if "store" in df.columns else None
        worst_row    = res[res["storetype"] == st].nlargest(1, "rmspe")
        ranking_rows.append({
            "storetype":     st,
            "overall_rmspe": r,
            "contrib_pct":   c,
            "n_stores":      n_stores,
            "worst_month":   worst_row["month_name"].iloc[0] if not worst_row.empty else "—",
            "worst_rmspe":   worst_row["rmspe"].iloc[0]      if not worst_row.empty else np.nan,
        })
    ranking = (pd.DataFrame(ranking_rows)
                 .sort_values("overall_rmspe", ascending=False)
                 .reset_index(drop=True))

    # ── Print: overall ranking ─────────────────────────────────────────────────
    print("\nOverall StoreType ranking (worst → best RMSPE):")
    print(f"  {'Type':<6} {'RMSPE':>9} {'Contrib%':>10} {'#Stores':>8} "
          f"{'Worst Month':>13} {'Worst RMSPE':>12}")
    print("  " + "─"*62)
    for _, row in ranking.iterrows():
        n_str = f"{int(row['n_stores'])}" if row["n_stores"] is not None else "—"
        print(f"  Type {row['storetype']}  "
              f"{row['overall_rmspe']:>8.1f}%  "
              f"{row['contrib_pct']:>9.1f}%  "
              f"{n_str:>8}  "
              f"{row['worst_month']:>12}  "
              f"{row['worst_rmspe']:>10.1f}%")

    # ── Print: spike month breakdown ───────────────────────────────────────────
    print(f"\nRMSPE in spike months vs annual average by StoreType:")
    header = f"  {'Type':<6} {'Annual':>8}"
    for abbrev in SPIKE_MONTHS.values():
        header += f" {abbrev:>9}"
    print(header)
    print("  " + "─"*50)
    for st in store_types:
        sub_all = df[df["storetype"] == st]
        annual  = rmspe(sub_all["y_actual"].values, sub_all["y_pred"].values)
        line    = f"  Type {st}  {annual:>7.1f}%"
        for m_idx in SPIKE_MONTHS:
            sub_m = df[(df["storetype"] == st) & (df["month"] == m_idx)]
            val   = rmspe(sub_m["y_actual"].values, sub_m["y_pred"].values) if len(sub_m) >= 10 else np.nan
            line += f" {val:>9.1f}%" if not np.isnan(val) else f"         —"
        print(line)

    # ── Print: observations ────────────────────────────────────────────────────
    print("\n── Observations ──────────────────────────────────────────────")

    worst_type = ranking.iloc[0]["storetype"]
    print(f"\n  Highest overall RMSPE : Type {worst_type} "
          f"({ranking.iloc[0]['overall_rmspe']:.1f}%, "
          f"{ranking.iloc[0]['contrib_pct']:.1f}% of total error)")

    if "a" in store_types:
        a_overall = rmspe(df[df["storetype"] == "a"]["y_actual"].values,
                          df[df["storetype"] == "a"]["y_pred"].values)
        a_pos     = list(ranking["storetype"]).index("a") + 1

        months_a_worst = []
        for m_idx, m_name in enumerate(MONTH_NAMES, start=1):
            month_vals = {}
            for st in store_types:
                sub = df[(df["storetype"] == st) & (df["month"] == m_idx)]
                if len(sub) >= 10:
                    month_vals[st] = rmspe(sub["y_actual"].values, sub["y_pred"].values)
            if "a" in month_vals and month_vals["a"] == max(month_vals.values()):
                months_a_worst.append(m_name)

        print(f"\n  Type A rank           : {a_pos} of {len(store_types)} "
              f"({a_overall:.1f}% annual RMSPE)")
        print(f"  Months where Type A is worst: {', '.join(months_a_worst) if months_a_worst else 'none'}")

        if len(months_a_worst) == 12:
            print("  → CONSISTENTLY worst across all months — structural gap, not seasonal.")
        elif len(months_a_worst) >= 7:
            print(f"  → Worst in {len(months_a_worst)}/12 months — predominantly structural,")
            print("    with some seasonal amplification in spike months.")
        else:
            print(f"  → Worst in only {len(months_a_worst)}/12 months — seasonal, not structural.")

        print(f"\n  Type A RMSPE premium in spike months (vs annual {a_overall:.1f}%):")
        for m_idx, abbrev in SPIKE_MONTHS.items():
            sub_m = df[(df["storetype"] == "a") & (df["month"] == m_idx)]
            if len(sub_m) >= 10:
                spike_r = rmspe(sub_m["y_actual"].values, sub_m["y_pred"].values)
                print(f"    {abbrev} : {spike_r:.1f}%  ({spike_r - a_overall:+.1f}pp)")

    print(f"\n  Spike concentration — uniform across types or Type-A-specific?")
    for m_idx, abbrev in SPIKE_MONTHS.items():
        month_vals = {}
        for st in store_types:
            sub = df[(df["storetype"] == st) & (df["month"] == m_idx)]
            if len(sub) >= 10:
                month_vals[st] = rmspe(sub["y_actual"].values, sub["y_pred"].values)
        if month_vals:
            spread     = max(month_vals.values()) - min(month_vals.values())
            worst_in_m = max(month_vals, key=month_vals.get)
            label      = "concentrated" if spread > 10 else "broadly uniform"
            print(f"    {abbrev} : spread={spread:.1f}pp, "
                  f"worst=Type {worst_in_m} ({month_vals[worst_in_m]:.1f}%)  → {label}")

    print("\n  Reminder — contribution vs RMSPE:")
    print("    High contribution + high RMSPE → genuine model quality problem.")
    print("    High contribution + similar RMSPE to other types → volume effect only.")
    print("─" * 60)

    # ── FIGURE 1: Stacked heatmaps ─────────────────────────────────────────────
    spike_abbrevs = set(SPIKE_MONTHS.values())
    fig, axes = plt.subplots(2, 1, figsize=(15, 9))

    for ax, pivot, cmap, vmin, cb_label, title in [
        (axes[0], pivot_rmspe,   "RdYlGn_r", 5,
         "RMSPE (%)",
         "StoreType × Month RMSPE  "
         "(navy border = confirmed spike months: Apr / Oct / Nov / Dec)"),
        (axes[1], pivot_contrib, "YlOrRd",   0,
         "% of Total Squared Error",
         "StoreType × Month Error Contribution  "
         "(reflects both RMSPE and number of stores in each type)"),
    ]:
        vals = pivot.values.astype(float)
        vmax = vals[~np.isnan(vals)].max() if not np.all(np.isnan(vals)) else 1
        im   = ax.imshow(vals, aspect="auto", cmap=cmap, vmin=vmin, vmax=vmax)
        ax.set_xticks(range(len(pivot.columns)))
        ax.set_xticklabels(pivot.columns, fontsize=10)
        ax.set_yticks(range(len(store_types)))
        ax.set_yticklabels([f"Type {t}" for t in pivot.index], fontsize=11)

        for i in range(vals.shape[0]):
            for j in range(vals.shape[1]):
                v = vals[i, j]
                if not np.isnan(v):
                    dark = (cmap == "RdYlGn_r" and (v > 42 or v < 9)) or \
                           (cmap == "YlOrRd"   and v > vmax * 0.6)
                    fmt  = f"{v:.0f}%" if cmap == "RdYlGn_r" else f"{v:.1f}%"
                    ax.text(j, i, fmt, ha="center", va="center",
                            fontsize=9, fontweight="bold",
                            color="white" if dark else "black")

        for j, col in enumerate(pivot.columns):
            if col in spike_abbrevs:
                ax.add_patch(plt.Rectangle(
                    (j - 0.5, -0.5), 1, len(store_types),
                    fill=False, edgecolor="navy",
                    linewidth=2.5, clip_on=False, zorder=5,
                ))

        plt.colorbar(im, ax=ax, label=cb_label, shrink=0.75)
        ax.set_title(title, fontsize=11, fontweight="bold", pad=8)

    plt.suptitle("StoreType × Month Error Analysis", fontsize=13, fontweight="bold")
    plt.tight_layout()
    fig.savefig(FIGURES_DIR / "8a_storetype_month_heatmaps.png",
                dpi=150, bbox_inches="tight")
    plt.close()

    # ── FIGURE 2: Overall ranking bars ────────────────────────────────────────
    type_colors = {"a": "#2196F3", "b": "#EF5350", "c": "#4CAF50", "d": "#FF9800"}
    bar_colors  = [type_colors.get(str(t), "#9E9E9E") for t in ranking["storetype"]]
    x_labels    = [f"Type {t}" for t in ranking["storetype"]]

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    for ax, col, ylabel, note in [
        (axes[0], "overall_rmspe", "RMSPE (%)",
         "Model quality per prediction\n(volume-neutral)"),
        (axes[1], "contrib_pct",   "% of Total Squared Error",
         "Total error mass\n(amplified by store count)"),
    ]:
        bars = ax.bar(x_labels, ranking[col].values, color=bar_colors, alpha=0.88)
        for bar, val in zip(bars, ranking[col].values):
            ax.text(bar.get_x() + bar.get_width()/2, val + 0.3,
                    f"{val:.1f}%", ha="center", va="bottom",
                    fontsize=10, fontweight="bold")
        ax.set_ylabel(ylabel, fontsize=10)
        ax.yaxis.set_major_formatter(mticker.PercentFormatter())
        ax.set_title(note, fontsize=10, fontweight="bold")
        ax.grid(axis="y", alpha=0.25)

    plt.suptitle("StoreType Overall Ranking — RMSPE vs Error Contribution\n"
                 "(both charts agree → genuine accuracy problem, "
                 "not just a volume effect)",
                 fontsize=11, fontweight="bold")
    plt.tight_layout()
    fig.savefig(FIGURES_DIR / "8b_storetype_overall_ranking.png",
                dpi=150, bbox_inches="tight")
    plt.close()

    print("\nSaved: 8a_storetype_month_heatmaps.png")
    print("       8b_storetype_overall_ranking.png")
    return res

def analysis_store_drilldown(df: pd.DataFrame,
                              focus_type: str = "a",
                              top_n:      int  = 20):
    """
    Drills from store-type level to individual stores.
    Determines whether high Type A error is systemic across all stores
    or concentrated in a small subset.

    Produces:
      9a_type_a_store_rankings.png       — top-N bars by RMSPE and contribution
      9b_type_a_month_store_heatmaps.png — month × store RMSPE + contribution heatmaps
    """
    print("\n" + "═"*60)
    print(f"ANALYSIS 9 — Individual Store Drill-Down  "
          f"(Focus: Type {focus_type.upper()})")
    print("═"*60)

    if "storetype" not in df.columns:
        print("⚠️  storetype column missing — was the store.csv join successful?")
        return None
    if "store" not in df.columns:
        print("⚠️  store column missing — check script 12 output.")
        return None

    total_spe = df["spe"].sum()
    type_df   = df[df["storetype"] == focus_type].copy()

    if type_df.empty:
        print(f"⚠️  No rows found for store type '{focus_type}'.")
        return None

    type_total_spe = type_df["spe"].sum()

    # ── Per-store stats ───────────────────────────────────────────────────────
    store_stats = (
        type_df.groupby("store")
               .apply(lambda g: pd.Series({
                   "rmspe":       rmspe(g["y_actual"].values, g["y_pred"].values),
                   "spe_sum":     g["spe"].sum(),
                   "n":           len(g),
               }))
               .reset_index()
    )
    store_stats["contrib_pct"]        = store_stats["spe_sum"] / total_spe * 100
    store_stats["contrib_within_type"]= store_stats["spe_sum"] / type_total_spe * 100

    by_rmspe   = store_stats.sort_values("rmspe",       ascending=False).reset_index(drop=True)
    by_contrib = store_stats.sort_values("contrib_pct", ascending=False).reset_index(drop=True)
    top_rmspe  = by_rmspe.head(top_n)
    top_contrib= by_contrib.head(top_n)

    n_stores         = len(store_stats)
    type_total_contrib = store_stats["contrib_pct"].sum()

    # ── Print: summary tables ─────────────────────────────────────────────────
    print(f"\nType {focus_type.upper()}: {n_stores} stores, "
          f"{type_total_contrib:.1f}% of total all-store error\n")

    print(f"Top {top_n} by RMSPE:")
    print(f"  {'Store':>7} {'RMSPE%':>8} {'Contrib%':>10} {'n':>7}")
    print("  " + "─"*36)
    for _, row in top_rmspe.iterrows():
        print(f"  {int(row['store']):>7}  {row['rmspe']:>7.1f}%  "
              f"{row['contrib_pct']:>9.2f}%  {int(row['n']):>7,}")

    print(f"\nTop {top_n} by Error Contribution:")
    print(f"  {'Store':>7} {'Contrib%':>9} {'Within-Type%':>14} {'RMSPE%':>8}")
    print("  " + "─"*42)
    for _, row in top_contrib.iterrows():
        print(f"  {int(row['store']):>7}  {row['contrib_pct']:>8.2f}%  "
              f"{row['contrib_within_type']:>13.1f}%  {row['rmspe']:>7.1f}%")

    # ── Concentration analysis ────────────────────────────────────────────────
    cum = by_contrib.copy()
    cum["cumsum"] = cum["contrib_pct"].cumsum()
    stores_50 = int((cum["cumsum"] <= type_total_contrib * 0.50).sum()) + 1
    stores_80 = int((cum["cumsum"] <= type_total_contrib * 0.80).sum()) + 1
    pct_50    = stores_50 / n_stores * 100
    pct_80    = stores_80 / n_stores * 100

    # ── Spike month concentration ─────────────────────────────────────────────
    SPIKE_MONTHS = {11: "Nov", 12: "Dec", 2: "Feb"}
    spike_rows = []
    for m_idx, m_abbrev in SPIKE_MONTHS.items():
        sub_m = type_df[type_df["month"] == m_idx]
        if len(sub_m) < 20:
            continue
        store_spe   = sub_m.groupby("store")["spe"].sum().sort_values(ascending=False)
        top5_pct    = store_spe.head(5).sum() / sub_m["spe"].sum() * 100
        n_with_data = sub_m["store"].nunique()
        spike_rows.append({
            "month": m_abbrev, "top5_pct": top5_pct,
            "n_stores": n_with_data,
            "concentrated": top5_pct > 50,
        })

    # ── Print: observations ───────────────────────────────────────────────────
    print(f"\n── Observations ──────────────────────────────────────────────")
    print(f"\n  Concentration (% of Type {focus_type.upper()} error driven by subset):")
    print(f"    {stores_50:3d} of {n_stores} stores ({pct_50:4.1f}%) → 50% of Type "
          f"{focus_type.upper()} error")
    print(f"    {stores_80:3d} of {n_stores} stores ({pct_80:4.1f}%) → 80% of Type "
          f"{focus_type.upper()} error")

    if pct_50 < 15:
        verdict = "HIGHLY CONCENTRATED"
        action  = ("A small subset drives most error. Store-specific treatment "
                   "(residual correction, targeted features) will have the "
                   "highest ROI.")
    elif pct_50 < 30:
        verdict = "MODERATELY CONCENTRATED"
        action  = ("Some concentration. Both feature engineering improvements "
                   "and targeted fixes for the worst stores are warranted.")
    else:
        verdict = "BROADLY SYSTEMIC"
        action  = ("Error is spread across most Type A stores. Feature "
                   "engineering fixes (Unity Day, Nov school holidays) will "
                   "benefit all stores rather than targeted per-store treatment.")

    print(f"\n  Verdict: {verdict}")
    print(f"  → {action}")

    if spike_rows:
        print(f"\n  Spike month concentration (Nov / Dec / Feb):")
        for row in spike_rows:
            label = "concentrated in a few stores" if row["concentrated"] \
                    else "broadly spread across stores"
            print(f"    {row['month']}: top-5 stores → {row['top5_pct']:.1f}% of "
                  f"Type {focus_type.upper()} {row['month']} error  "
                  f"({row['n_stores']} stores with data)  → {label}")

    print("─" * 60)

    # ── Month × Store data for heatmaps ───────────────────────────────────────
    # Use top-N by RMSPE as the reference set and ordering for both heatmaps
    top_store_ids = list(top_rmspe["store"])
    month_records = []
    for s in top_store_ids:
        for m_idx, m_name in enumerate(MONTH_NAMES, start=1):
            sub = type_df[(type_df["store"] == s) & (type_df["month"] == m_idx)]
            if len(sub) >= 5:
                month_records.append({
                    "store":       s,
                    "month":       m_idx,
                    "month_name":  m_name,
                    "rmspe":       rmspe(sub["y_actual"].values, sub["y_pred"].values),
                    "contrib_pct": sub["spe"].sum() / total_spe * 100,
                })
    month_df = pd.DataFrame(month_records)

    # ── FIGURE 1: Ranking bars ────────────────────────────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(16, 8))

    for ax, data, xcol, xlabel, color, title in [
        (axes[0], top_rmspe,   "rmspe",       "RMSPE (%)",
         "#EF5350", f"Top {top_n} Type {focus_type.upper()} Stores by RMSPE"),
        (axes[1], top_contrib, "contrib_pct",  "% of Total Squared Error",
         "#FF9800", f"Top {top_n} Type {focus_type.upper()} Stores by Error Contribution"),
    ]:
        ax.barh(range(len(data)), data[xcol].values, color=color, alpha=0.85)
        ax.set_yticks(range(len(data)))
        ax.set_yticklabels([f"Store {int(s)}" for s in data["store"]], fontsize=8)
        ax.invert_yaxis()
        ax.set_xlabel(xlabel, fontsize=10)
        ax.xaxis.set_major_formatter(mticker.PercentFormatter())
        ax.set_title(title, fontweight="bold")
        ax.grid(axis="x", alpha=0.25)
        # Annotate bars
        for i, val in enumerate(data[xcol].values):
            ax.text(val + data[xcol].max() * 0.01, i,
                    f"{val:.1f}%", va="center", fontsize=7)

    plt.suptitle(f"Store Type {focus_type.upper()} — Individual Store Rankings",
                 fontsize=13, fontweight="bold")
    plt.tight_layout()
    fig.savefig(FIGURES_DIR / f"9a_type_{focus_type}_store_rankings.png",
                dpi=150, bbox_inches="tight")
    plt.close()

    # ── FIGURE 2: Month × Store heatmaps ─────────────────────────────────────
    if not month_df.empty:
        month_cols  = [m for m in MONTH_NAMES if m in month_df["month_name"].unique()]
        spike_abbrevs = {"Nov", "Dec", "Feb"}

        pivot_r = (month_df.pivot(index="store", columns="month_name", values="rmspe")
                            .reindex(index=[s for s in top_store_ids
                                            if s in month_df["store"].values])
                            [month_cols])
        pivot_c = (month_df.pivot(index="store", columns="month_name", values="contrib_pct")
                            .reindex(index=pivot_r.index)
                            [month_cols])

        fig, axes = plt.subplots(2, 1, figsize=(16, max(12, len(pivot_r) * 0.55 + 3)))

        for ax, pivot, cmap, vmin, fmt_str, cb_label, title in [
            (axes[0], pivot_r, "RdYlGn_r", 5,  "{:.0f}",
             "RMSPE (%)",
             f"Type {focus_type.upper()} — Month × Store RMSPE  "
             "(navy border = Nov / Dec / Feb spike months)"),
            (axes[1], pivot_c, "YlOrRd",   0,  "{:.2f}",
             "% of Total Squared Error",
             f"Type {focus_type.upper()} — Month × Store Error Contribution"),
        ]:
            vals = pivot.values.astype(float)
            finite_vals = vals[~np.isnan(vals)]
            vmax = finite_vals.max() if len(finite_vals) > 0 else 1

            im = ax.imshow(vals, aspect="auto", cmap=cmap, vmin=vmin, vmax=vmax)
            ax.set_xticks(range(len(pivot.columns)))
            ax.set_xticklabels(pivot.columns, fontsize=9)
            ax.set_yticks(range(len(pivot.index)))
            ax.set_yticklabels([f"Store {int(s)}" for s in pivot.index], fontsize=7)

            for i in range(vals.shape[0]):
                for j in range(vals.shape[1]):
                    v = vals[i, j]
                    if not np.isnan(v):
                        dark = (cmap == "RdYlGn_r" and (v > 42 or v < 9)) or \
                               (cmap == "YlOrRd"   and v > vmax * 0.6)
                        ax.text(j, i, fmt_str.format(v),
                                ha="center", va="center", fontsize=6,
                                fontweight="bold",
                                color="white" if dark else "black")

            for j, col in enumerate(pivot.columns):
                if col in spike_abbrevs:
                    ax.add_patch(plt.Rectangle(
                        (j - 0.5, -0.5), 1, len(pivot.index),
                        fill=False, edgecolor="navy",
                        linewidth=2.0, clip_on=False, zorder=5,
                    ))

            plt.colorbar(im, ax=ax, label=cb_label, shrink=0.75)
            ax.set_title(title, fontsize=11, fontweight="bold", pad=8)

        plt.suptitle(
            f"Type {focus_type.upper()} — Month × Store Error Heatmaps  "
            f"(top {top_n} worst stores by RMSPE, worst-first top to bottom)",
            fontsize=12, fontweight="bold"
        )
        plt.tight_layout()
        fig.savefig(FIGURES_DIR / f"9b_type_{focus_type}_month_store_heatmaps.png",
                    dpi=150, bbox_inches="tight")
        plt.close()

    print(f"\nSaved: 9a_type_{focus_type}_store_rankings.png")
    print(f"       9b_type_{focus_type}_month_store_heatmaps.png")
    return store_stats

def analysis_single_store(df: pd.DataFrame, store_id: int = 652):
    """
    Drills into a single store to identify which calendar and business
    events are responsible for forecasting failures.

    Investigates: StateHoliday, SchoolHoliday, Promo, DayOfWeek,
    Weekend vs Weekday, Month — globally and within each spike month.

    Produces:
      10a_store_{id}_rmspe_profiles.png      — 6-panel RMSPE breakdown
      10b_store_{id}_error_contributions.png — contribution by each feature
      10c_store_{id}_spike_month_factors.png — factor premiums for Nov/Dec/Feb
    """
    print("\n" + "═"*60)
    print(f"ANALYSIS 10 — Calendar Event Drill-Down  (Store {store_id})")
    print("═"*60)

    if "store" not in df.columns:
        print("⚠️  store column missing.")
        return None

    store_df = df[df["store"] == store_id].copy()
    if store_df.empty:
        print(f"⚠️  No data found for store {store_id}.")
        return None

    total_spe      = df["spe"].sum()
    store_total_spe= store_df["spe"].sum()
    store_rmspe    = rmspe(store_df["y_actual"].values, store_df["y_pred"].values)
    store_contrib  = store_total_spe / total_spe * 100
    store_type     = store_df["storetype"].iloc[0] \
                     if "storetype" in store_df.columns else "?"
    n              = len(store_df)

    print(f"\n  Store {store_id} (Type {str(store_type).upper()}) — "
          f"{n:,} predictions")
    print(f"  Overall RMSPE={store_rmspe:.1f}%  |  "
          f"Error contribution={store_contrib:.2f}% of total")

    # ── Helper: RMSPE + contribution per category ─────────────────────────────
    def feature_breakdown(col, values_order=None):
        vals = values_order if values_order is not None \
               else sorted(store_df[col].dropna().unique())
        rows = []
        for v in vals:
            sub = store_df[store_df[col] == v]
            if len(sub) >= 5:
                rows.append({
                    "value":          v,
                    "rmspe":          rmspe(sub["y_actual"].values, sub["y_pred"].values),
                    "contrib_pct":    sub["spe"].sum() / total_spe * 100,
                    "n":              len(sub),
                    "pct_of_store":   len(sub) / n * 100,
                })
        return pd.DataFrame(rows)

    # ── Feature breakdowns ────────────────────────────────────────────────────
    bd_state  = feature_breakdown("stateholiday",
                                   ["none", "public", "easter", "christmas"])
    bd_state["label"] = bd_state["value"]   
    bd_school = feature_breakdown("is_school_holiday", [False, True])
    bd_promo  = feature_breakdown("is_promo",          [False, True])
    bd_dow    = feature_breakdown("dow", list(range(1, 8)))
    bd_wknd   = feature_breakdown("is_weekend", [False, True])
    bd_month  = feature_breakdown("month", list(range(1, 13)))

    # Readable labels
    bd_school = bd_school.copy()
    bd_school["label"] = bd_school["value"].map(
        {False: "No school holiday", True: "School holiday"})
    bd_promo  = bd_promo.copy()
    bd_promo["label"]  = bd_promo["value"].map({False: "No Promo", True: "Promo"})
    bd_wknd   = bd_wknd.copy()
    bd_wknd["label"]   = bd_wknd["value"].map({False: "Weekday", True: "Weekend"})
    bd_dow    = bd_dow.copy()
    bd_dow["label"]    = bd_dow["value"].map(DOW_LABELS)
    bd_month  = bd_month.copy()
    bd_month["label"]  = bd_month["value"].map(
        {i+1: m for i, m in enumerate(MONTH_NAMES)})

    # ── Spike month factor premium computation ────────────────────────────────
    SPIKE_MONTHS = [(11, "November"), (12, "December"), (2, "February")]
    FACTORS = [
        ("is_public_holiday", True,  "StateHoliday"),
        ("is_school_holiday", True,  "SchoolHoliday"),
        ("is_promo",          True,  "Promo"),
        ("is_weekend",        True,  "Weekend"),
    ]

    def factor_premium(sub_df, col, active_val):
        """RMSPE for (active, inactive, premium) within sub_df."""
        act = sub_df[sub_df[col] == active_val]
        ina = sub_df[sub_df[col] != active_val]
        r_a = rmspe(act["y_actual"].values, act["y_pred"].values) \
              if len(act) >= 5 else np.nan
        r_i = rmspe(ina["y_actual"].values, ina["y_pred"].values) \
              if len(ina) >= 5 else np.nan
        prem = (r_a - r_i) if not (np.isnan(r_a) or np.isnan(r_i)) else np.nan
        return r_a, r_i, prem

    spike_results = {}
    for m_idx, m_name in SPIKE_MONTHS:
        sub_m = store_df[store_df["month"] == m_idx]
        m_rmspe = rmspe(sub_m["y_actual"].values, sub_m["y_pred"].values) \
                  if len(sub_m) >= 5 else np.nan
        spike_results[m_name] = {"overall_rmspe": m_rmspe, "factors": {}}
        for col, val, label in FACTORS:
            r_a, r_i, prem = factor_premium(sub_m, col, val)
            spike_results[m_name]["factors"][label] = {
                "active": r_a, "inactive": r_i, "premium": prem
            }

    # ── Overall driver ranking ─────────────────────────────────────────────────
    overall_premiums = []
    for col, val, label in FACTORS:
        r_a, r_i, prem = factor_premium(store_df, col, val)
        if not np.isnan(prem):
            overall_premiums.append({
                "driver": label, "active_rmspe": r_a,
                "inactive_rmspe": r_i, "premium": prem,
            })
    overall_premiums.sort(key=lambda x: x["premium"], reverse=True)

    # ── Print observations ────────────────────────────────────────────────────
    print("\n── Calendar Event RMSPE (all dates) ─────────────────────────")

    for data, col_label in [
        (bd_state,  "StateHoliday"),
        (bd_school, "SchoolHoliday"),
        (bd_promo,  "Promo"),
    ]:
        if not data.empty:
            print(f"\n  {col_label}:")
            lbl_col = "label" if "label" in data.columns else "value"
            for _, row in data.iterrows():
                print(f"    {str(row[lbl_col]):22s}: "
                      f"{row['rmspe']:6.1f}%  "
                      f"({row['pct_of_store']:4.1f}% of store days)")

    print("\n  Overall error driver ranking (RMSPE premium when active):")
    for i, d in enumerate(overall_premiums, 1):
        print(f"    {i}. {d['driver']:<16}: {d['active_rmspe']:.1f}% vs "
              f"{d['inactive_rmspe']:.1f}% when inactive  "
              f"({d['premium']:+.1f}pp premium)")

    print("\n── Spike Month Analysis (Nov / Dec / Feb) ────────────────────")
    for m_name, res_m in spike_results.items():
        o = res_m["overall_rmspe"]
        print(f"\n  {m_name}  (overall RMSPE: "
              f"{'—' if np.isnan(o) else f'{o:.1f}%'})")
        sorted_f = sorted(
            [(f, v) for f, v in res_m["factors"].items()
             if not np.isnan(v["premium"])],
            key=lambda x: x[1]["premium"], reverse=True
        )
        for f_name, v in sorted_f:
            print(f"    {f_name:<16}: active={v['active']:.1f}%  "
                  f"inactive={v['inactive']:.1f}%  "
                  f"premium={v['premium']:+.1f}pp")
        if sorted_f:
            print(f"    → Primary driver: {sorted_f[0][0]}")

    print("\n  Are the same factors driving all three spike months?")
    top_drivers = {}
    for m_name, res_m in spike_results.items():
        valid = [(f, v["premium"]) for f, v in res_m["factors"].items()
                 if not np.isnan(v["premium"])]
        if valid:
            top = max(valid, key=lambda x: x[1])
            top_drivers[m_name] = top[0]
            print(f"    {m_name:<12}: primary driver = {top[0]}  "
                  f"({top[1]:+.1f}pp premium)")
    unique_drivers = set(top_drivers.values())
    if len(unique_drivers) == 1:
        print(f"  → YES — all three spike months share the same primary driver: "
              f"{list(unique_drivers)[0]}.")
    else:
        print(f"  → NO — spike months have different primary drivers: "
              f"{', '.join(f'{m}={d}' for m, d in top_drivers.items())}.")

    print("─" * 60)

    # ── FIGURE 1: 3×2 RMSPE profiles ─────────────────────────────────────────
    fig, axes = plt.subplots(3, 2, figsize=(14, 13))
    plt.suptitle(
        f"Store {store_id} (Type {str(store_type).upper()}) — "
        f"Calendar Feature RMSPE Profiles\n"
        f"Overall RMSPE={store_rmspe:.1f}%   "
        f"Error contribution={store_contrib:.2f}% of total",
        fontsize=12, fontweight="bold"
    )

    def bar_rmspe(ax, data, xcol="label", title="", color="#42A5F5"):
        if data.empty:
            ax.set_visible(False)
            return
        vals = data[xcol].tolist()
        heights = data["rmspe"].tolist()
        colors  = [color] * len(vals)
        # Highlight max in red
        if len(heights) > 1:
            colors[heights.index(max(heights))] = "#EF5350"
        bars = ax.bar(vals, heights, color=colors, alpha=0.87)
        for bar, h in zip(bars, heights):
            ax.text(bar.get_x() + bar.get_width()/2, h + max(heights)*0.01,
                    f"{h:.0f}%", ha="center", va="bottom", fontsize=8)
        ax.set_ylabel("RMSPE (%)", fontsize=9)
        ax.yaxis.set_major_formatter(mticker.PercentFormatter())
        ax.set_title(title, fontweight="bold", fontsize=10)
        ax.tick_params(axis="x", labelsize=8, rotation=15)
        ax.grid(axis="y", alpha=0.25)

    bar_rmspe(axes[0,0], bd_state,  title="RMSPE by StateHoliday",  color="#9C27B0")
    bar_rmspe(axes[0,1], bd_school, title="RMSPE by SchoolHoliday", color="#FF9800")
    bar_rmspe(axes[1,0], bd_promo,  title="RMSPE by Promo",         color="#4CAF50")
    bar_rmspe(axes[1,1], bd_dow,    title="RMSPE by Day of Week",   color="#2196F3")
    bar_rmspe(axes[2,0], bd_wknd,   title="Weekend vs Weekday RMSPE", color="#607D8B")

    ax = axes[2,1]
    if not bd_month.empty:
        ax.plot(bd_month["label"], bd_month["rmspe"],
                marker="o", linewidth=2, color="#EF5350", markersize=7)
        spike_abbrevs = {MONTH_NAMES[m-1] for m, _ in SPIKE_MONTHS}
        for _, row in bd_month.iterrows():
            if row["label"] in spike_abbrevs:
                ax.annotate(
                    f"← {row['label']}",
                    xy=(list(bd_month["label"]).index(row["label"]),
                        row["rmspe"]),
                    xytext=(5, 0), textcoords="offset points",
                    ha="left", fontsize=7, color="darkred", fontweight="bold",
                )
        ax.set_ylabel("RMSPE (%)", fontsize=9)
        ax.yaxis.set_major_formatter(mticker.PercentFormatter())
        ax.set_title("Monthly RMSPE  (spike months annotated)",
                     fontweight="bold", fontsize=10)
        ax.tick_params(axis="x", rotation=45, labelsize=8)
        ax.grid(alpha=0.22)
    else:
        axes[2,1].set_visible(False)

    plt.tight_layout()
    fig.savefig(FIGURES_DIR / f"10a_store_{store_id}_rmspe_profiles.png",
                dpi=150, bbox_inches="tight")
    plt.close()

    # ── FIGURE 2: Error contribution breakdown ────────────────────────────────
    fig, axes = plt.subplots(2, 2, figsize=(13, 9))
    plt.suptitle(
        f"Store {store_id} — Error Contribution by Calendar Feature\n"
        f"(% of total all-store squared percentage error)",
        fontsize=12, fontweight="bold"
    )

    def bar_contrib(ax, data, xcol="label", title="", color="#EF5350"):
        if data.empty:
            ax.set_visible(False)
            return
        vals    = data[xcol].tolist()
        heights = data["contrib_pct"].tolist()
        mx      = max(heights) if heights else 1
        bars = ax.bar(vals, heights, color=color, alpha=0.85)
        for bar, h in zip(bars, heights):
            ax.text(bar.get_x() + bar.get_width()/2, h + mx * 0.01,
                    f"{h:.3f}%", ha="center", va="bottom", fontsize=7)
        ax.set_ylabel("% of Total Squared Error", fontsize=9)
        ax.yaxis.set_major_formatter(
            mticker.FuncFormatter(lambda x, _: f"{x:.3f}%"))
        ax.set_title(title, fontweight="bold", fontsize=10)
        ax.tick_params(axis="x", labelsize=8, rotation=15)
        ax.grid(axis="y", alpha=0.25)

    bar_contrib(axes[0,0], bd_state,  title="Contribution by StateHoliday",  color="#9C27B0")
    bar_contrib(axes[0,1], bd_school, title="Contribution by SchoolHoliday", color="#FF9800")
    bar_contrib(axes[1,0], bd_promo,  title="Contribution by Promo",         color="#4CAF50")

    ax = axes[1,1]
    if not bd_month.empty:
        spike_abbrevs = {MONTH_NAMES[m-1] for m, _ in SPIKE_MONTHS}
        colors_m = ["#EF5350" if lbl in spike_abbrevs else "#90CAF9"
                    for lbl in bd_month["label"]]
        mx_c = bd_month["contrib_pct"].max()
        bars = ax.bar(bd_month["label"], bd_month["contrib_pct"],
                      color=colors_m, alpha=0.85)
        for bar, h in zip(bars, bd_month["contrib_pct"]):
            ax.text(bar.get_x() + bar.get_width()/2, h + mx_c * 0.01,
                    f"{h:.3f}%", ha="center", va="bottom", fontsize=6, rotation=45)
        ax.set_ylabel("% of Total Squared Error", fontsize=9)
        ax.yaxis.set_major_formatter(
            mticker.FuncFormatter(lambda x, _: f"{x:.3f}%"))
        ax.set_title("Contribution by Month  (red = spike months)",
                     fontweight="bold", fontsize=10)
        ax.tick_params(axis="x", rotation=45, labelsize=8)
        ax.grid(axis="y", alpha=0.25)
    else:
        axes[1,1].set_visible(False)

    plt.tight_layout()
    fig.savefig(FIGURES_DIR / f"10b_store_{store_id}_error_contributions.png",
                dpi=150, bbox_inches="tight")
    plt.close()

    # ── FIGURE 3: Spike month factor premiums ─────────────────────────────────
    factor_labels = [f for _, _, f in FACTORS]
    spike_names   = [m for _, m in SPIKE_MONTHS]
    colors_spike  = ["#EF5350", "#FF9800", "#42A5F5"]

    fig, ax = plt.subplots(figsize=(12, 5))
    x = np.arange(len(factor_labels))
    w = 0.25
    for i, (m_name, color) in enumerate(zip(spike_names, colors_spike)):
        premiums = [
            spike_results[m_name]["factors"].get(f, {}).get("premium", np.nan)
            for f in factor_labels
        ]
        bars = ax.bar(x + (i-1)*w, premiums, w,
                      label=m_name, color=color, alpha=0.87)
        for bar, val in zip(bars, premiums):
            if not np.isnan(val):
                offset = 0.4 if val >= 0 else -2.0
                ax.text(bar.get_x() + bar.get_width()/2, val + offset,
                        f"{val:+.0f}pp", ha="center", va="bottom", fontsize=8)

    ax.axhline(0, color="black", linewidth=0.8, linestyle="--")
    ax.set_xticks(x)
    ax.set_xticklabels(factor_labels, fontsize=11)
    ax.set_ylabel("RMSPE Premium (percentage points)\n[active − inactive]",
                  fontsize=10)
    ax.set_title(
        f"Store {store_id} — Calendar Factor RMSPE Premium by Spike Month\n"
        "Tallest bar per month = primary driver of that month's spike  |  "
        "Same pattern across months = shared root cause",
        fontsize=11, fontweight="bold"
    )
    ax.legend(fontsize=10, title="Spike month")
    ax.grid(axis="y", alpha=0.25)
    plt.tight_layout()
    fig.savefig(FIGURES_DIR / f"10c_store_{store_id}_spike_month_factors.png",
                dpi=150, bbox_inches="tight")
    plt.close()

    print(f"\nSaved: 10a_store_{store_id}_rmspe_profiles.png")
    print(f"       10b_store_{store_id}_error_contributions.png")
    print(f"       10c_store_{store_id}_spike_month_factors.png")
    return store_df

def analysis_ablation(df: pd.DataFrame):
    """
    Quantitatively validates identified error drivers by re-computing RMSPE
    and total error mass after independently excluding each factor.
 
    Experiments:
      1. Exclude Store 652
      2. Exclude all Type A stores
      3. Exclude November
      4. Exclude weekends
      5. Exclude school holidays
      6. Exclude November weekends only
      7. Exclude Store 652 during November only
      8. Joint ablation  — all factors simultaneously
 
    Produces:
      11a_ablation_rmspe_comparison.png  — RMSPE before/after per experiment
      11b_ablation_error_mass.png        — SPE reduction + data excluded overlay
      11c_ablation_importance_ranking.png — ranked horizontal bar chart
    """
    print("\n" + "═"*60)
    print("ANALYSIS 11 — Counterfactual (Ablation) Validation")
    print("═"*60)
 
    # ── Baseline ──────────────────────────────────────────────────────────────
    baseline_n     = len(df)
    baseline_rmspe = rmspe(df["y_actual"].values, df["y_pred"].values)
    baseline_spe   = df["spe"].sum()
 
    print(f"\nBaseline — {baseline_n:,} predictions")
    print(f"  RMSPE = {baseline_rmspe:.3f}%  |  Total SPE = {baseline_spe:.4f}")
 
    # ── Experiment definitions  (label, keep_mask) ────────────────────────────
    # keep_mask = True means the row is KEPT in the ablated dataset
    expts = []
 
    if "store" in df.columns:
        expts.append(("Excl. Store 652",
                       df["store"] != 652))
 
    if "storetype" in df.columns:
        expts.append(("Excl. Type A",
                       df["storetype"] != "a"))
 
    expts.append(("Excl. November",
                   df["month"] != 11))
 
    if "is_weekend" in df.columns:
        expts.append(("Excl. Weekends",
                       ~df["is_weekend"]))
 
    if "is_school_holiday" in df.columns:
        expts.append(("Excl. School Hols",
                       ~df["is_school_holiday"]))
 
    if "is_weekend" in df.columns:
        expts.append(("Excl. Nov Weekends",
                       ~(df["month"].eq(11) & df["is_weekend"])))
 
    if "store" in df.columns:
        expts.append(("Excl. S652 × Nov",
                       ~(df["store"].eq(652) & df["month"].eq(11))))
 
    # Joint ablation — exclude every identified driver simultaneously
    if all(c in df.columns for c in ["store", "is_weekend", "is_school_holiday"]):
        expts.append(("Joint Ablation",
                       (df["store"]            != 652) &
                       (df["month"]            != 11)  &
                       (~df["is_weekend"])             &
                       (~df["is_school_holiday"])))
 
    # ── Run experiments ───────────────────────────────────────────────────────
    rows = []
    for label, keep in expts:
        sub = df[keep]
        n   = len(sub)
        if n < 10:
            continue
        r        = rmspe(sub["y_actual"].values, sub["y_pred"].values)
        spe_rem  = sub["spe"].sum()
        n_excl   = baseline_n - n
        pct_excl = n_excl / baseline_n * 100
        rr       = baseline_rmspe - r
        rows.append({
            "experiment":       label,
            "n_remaining":      n,
            "n_excluded":       n_excl,
            "pct_excluded":     pct_excl,
            "rmspe":            r,
            "rmspe_reduction":  rr,
            "pct_rmspe_improv": rr / baseline_rmspe * 100,
            "spe_remaining":    spe_rem,
            "pct_spe_reduction":(baseline_spe - spe_rem) / baseline_spe * 100,
            "efficiency":       rr / pct_excl if pct_excl > 0 else np.nan,
        })
 
    res = pd.DataFrame(rows)
 
    # ── Print comparison table ────────────────────────────────────────────────
    print(f"\n  {'Experiment':<22} {'N kept':>9} {'Excl%':>6} "
          f"{'RMSPE':>7} {'ΔRMSPE':>8} {'%Improv':>8} "
          f"{'SPEred%':>8} {'Effic.':>7}")
    print("  " + "─"*83)
    for _, row in res.iterrows():
        marker = "▶ " if "Joint" in row["experiment"] else "  "
        print(f"{marker}{row['experiment']:<22} "
              f"{int(row['n_remaining']):>9,} "
              f"{row['pct_excluded']:>5.1f}% "
              f"{row['rmspe']:>6.2f}% "
              f"{row['rmspe_reduction']:>+7.2f}pp "
              f"{row['pct_rmspe_improv']:>7.1f}% "
              f"{row['pct_spe_reduction']:>7.1f}% "
              f"{row['efficiency']:>6.3f}")
    print(f"\n  Efficiency = pp RMSPE gained per 1% of predictions excluded.")
    print(f"  Higher efficiency = more impact from removing fewer predictions.")
 
    # ── Observations ──────────────────────────────────────────────────────────
    print("\n── Observations ──────────────────────────────────────────────")
 
    non_joint  = res[~res["experiment"].str.contains("Joint")]
    joint_rows = res[res["experiment"].str.contains("Joint")]
 
    # 1. Which factor contributes most?
    best_raw = non_joint.loc[non_joint["rmspe_reduction"].idxmax()]
    best_eff = non_joint.loc[non_joint["efficiency"].idxmax()]
    print(f"\n  1. Largest single contributor (raw RMSPE reduction):")
    print(f"     '{best_raw['experiment']}' → "
          f"{best_raw['rmspe_reduction']:+.2f}pp RMSPE reduction  "
          f"(excludes only {best_raw['pct_excluded']:.1f}% of data)")
    if best_eff["experiment"] != best_raw["experiment"]:
        print(f"     Most efficient: '{best_eff['experiment']}' → "
              f"{best_eff['efficiency']:.3f} pp per 1% excluded")
 
    # 2. Store 652 significance
    s652 = non_joint[non_joint["experiment"].str.contains("652") &
                     ~non_joint["experiment"].str.contains("Nov")]
    if not s652.empty:
        r = s652.iloc[0]
        print(f"\n  2. Store 652 significance:")
        print(f"     Removing it ({r['pct_excluded']:.2f}% of all predictions) "
              f"→ RMSPE improves by {r['rmspe_reduction']:+.2f}pp "
              f"({r['pct_rmspe_improv']:.1f}% relative).")
        print(f"     Efficiency = {r['efficiency']:.2f}  "
              f"({'disproportionately high impact' if r['efficiency'] > 1.5 else 'moderate impact'}"
              f" relative to its data share).")
 
    # 3. November vs November weekends
    nov_r = non_joint[non_joint["experiment"] == "Excl. November"]
    nwk_r = non_joint[non_joint["experiment"] == "Excl. Nov Weekends"]
    if not nov_r.empty and not nwk_r.empty:
        n_rr = nov_r.iloc[0]
        w_rr = nwk_r.iloc[0]
        share = w_rr["rmspe_reduction"] / n_rr["rmspe_reduction"] * 100
        print(f"\n  3. November vs November Weekends:")
        print(f"     Excl. November      → {n_rr['rmspe_reduction']:+.2f}pp  "
              f"(excl. {n_rr['pct_excluded']:.1f}% of data)")
        print(f"     Excl. Nov Weekends  → {w_rr['rmspe_reduction']:+.2f}pp  "
              f"(excl. {w_rr['pct_excluded']:.1f}% of data)")
        print(f"     Nov weekends capture {share:.0f}% of November's full impact.")
        if share > 70:
            print("     → November weekends are the dominant within-November driver.")
            print("       Non-weekend November days are relatively well-forecast.")
        elif share > 40:
            print("     → November weekends are significant but non-weekend")
            print("       November days also contribute meaningfully.")
        else:
            print("     → November's error is spread across all days, not")
            print("       concentrated in weekends.")
 
    # 4. Total explainability
    if not joint_rows.empty:
        jr = joint_rows.iloc[0]
        print(f"\n  4. Total explainability (Joint Ablation):")
        print(f"     All factors removed simultaneously:")
        print(f"     {baseline_rmspe:.2f}% → {jr['rmspe']:.2f}%  "
              f"({jr['rmspe_reduction']:+.2f}pp,  "
              f"{jr['pct_rmspe_improv']:.1f}% relative improvement)")
        print(f"     {jr['pct_excluded']:.1f}% of predictions excluded.")
        print(f"     → Identified factors explain {jr['pct_rmspe_improv']:.1f}% "
              f"of baseline RMSPE.")
        print(f"     → Residual RMSPE on clean data: {jr['rmspe']:.2f}%")
 
    # 5. Remaining unknowns
    if not joint_rows.empty:
        jr = joint_rows.iloc[0]
        unexplained = jr["rmspe"] / baseline_rmspe * 100
        print(f"\n  5. Remaining errors requiring further investigation:")
        print(f"     {unexplained:.0f}% of baseline RMSPE is unexplained "
              f"by the identified factors.")
        print(f"     Likely sources:")
        print(f"       - Store-level structural trends not captured in features")
        print(f"       - Promo2 / continuous promotion interaction effects")
        print(f"       - Competition effects (CompetitionDistance, new entrants)")
        print(f"       - Regional demand shocks beyond national holiday calendar")
        print(f"       - Mid-bucket horizon anomaly (days 30–40 October window)")
    print("─" * 60)
 
    # ── Shared helpers for figures ────────────────────────────────────────────
    sorted_res = (res.sort_values("rmspe_reduction", ascending=False)
                     .reset_index(drop=True))
    labels     = sorted_res["experiment"].tolist()
    n_e        = len(labels)
    x          = np.arange(n_e)
    w          = 0.38
    bar_blue   = ["#4CAF50" if "Joint" in l else "#2196F3" for l in labels]
    fig_w      = max(12, n_e * 1.5)
 
    # ── FIGURE 11a: RMSPE before vs after ────────────────────────────────────
    fig, ax = plt.subplots(figsize=(fig_w, 6))
    ax.bar(x - w/2, [baseline_rmspe]*n_e, w,
           label=f"Baseline ({baseline_rmspe:.2f}%)",
           color="#BDBDBD", alpha=0.9)
    bars_after = ax.bar(x + w/2, sorted_res["rmspe"], w,
                        label="After exclusion",
                        color=bar_blue, alpha=0.88)
    for bar, val in zip(bars_after, sorted_res["rmspe"]):
        ax.text(bar.get_x() + bar.get_width()/2, val + 0.08,
                f"{val:.2f}%", ha="center", va="bottom", fontsize=7.5)
    for xi, red in enumerate(sorted_res["rmspe_reduction"]):
        ax.annotate(f"{red:+.2f}pp",
                    xy=(xi + w/2, sorted_res.loc[xi, "rmspe"]),
                    xytext=(0, 14), textcoords="offset points",
                    ha="center", fontsize=7, fontweight="bold",
                    color="#1B5E20" if red > 0 else "#B71C1C")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=25, ha="right", fontsize=9)
    ax.set_ylabel("RMSPE (%)", fontsize=11)
    ax.yaxis.set_major_formatter(mticker.PercentFormatter())
    ax.set_title(
        "Ablation Analysis — RMSPE Before and After Each Exclusion\n"
        "(sorted by impact, largest first  |  "
        "green = joint ablation, blue = individual)",
        fontsize=12, fontweight="bold"
    )
    ax.legend(fontsize=10)
    ax.grid(axis="y", alpha=0.25)
    plt.tight_layout()
    fig.savefig(FIGURES_DIR / "11a_ablation_rmspe_comparison.png",
                dpi=150, bbox_inches="tight")
    plt.close()
 
    # ── FIGURE 11b: Error mass reduction + data excluded overlay ──────────────
    bar_red = ["#4CAF50" if "Joint" in l else "#EF5350" for l in labels]
    fig, ax = plt.subplots(figsize=(fig_w, 5))
    bars_m = ax.bar(x, sorted_res["pct_spe_reduction"],
                    color=bar_red, alpha=0.85)
    for bar, val in zip(bars_m, sorted_res["pct_spe_reduction"]):
        ax.text(bar.get_x() + bar.get_width()/2, val + 0.3,
                f"{val:.1f}%", ha="center", va="bottom", fontsize=8)
    ax2 = ax.twinx()
    ax2.plot(x, sorted_res["pct_excluded"], marker="D",
             color="#FF9800", linewidth=1.5, markersize=6,
             linestyle="--", label="% predictions excluded", alpha=0.9)
    for xi, val in enumerate(sorted_res["pct_excluded"]):
        ax2.annotate(f"{val:.1f}%",
                     xy=(xi, val), xytext=(0, 6), textcoords="offset points",
                     ha="center", fontsize=6.5, color="#E65100")
    ax2.set_ylabel("% Predictions Excluded", fontsize=10, color="#FF9800")
    ax2.tick_params(axis="y", colors="#FF9800")
    ax2.yaxis.set_major_formatter(mticker.PercentFormatter())
    ax2.legend(loc="upper right", fontsize=9)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=25, ha="right", fontsize=9)
    ax.set_ylabel("% Reduction in Total Squared Error", fontsize=11)
    ax.yaxis.set_major_formatter(mticker.PercentFormatter())
    ax.set_title(
        "Ablation Analysis — Error Mass Reduction\n"
        "(bars = SPE reduction  |  orange diamonds = share of data removed  |  "
        "context: large SPE reduction from small exclusion = high-leverage driver)",
        fontsize=11, fontweight="bold"
    )
    ax.grid(axis="y", alpha=0.25)
    plt.tight_layout()
    fig.savefig(FIGURES_DIR / "11b_ablation_error_mass.png",
                dpi=150, bbox_inches="tight")
    plt.close()
 
    # ── FIGURE 11c: Ranked importance (horizontal bars) ───────────────────────
    rank_df    = res.sort_values("rmspe_reduction", ascending=True).reset_index(drop=True)
    bar_clr_h  = ["#4CAF50" if "Joint" in l else "#2196F3"
                  for l in rank_df["experiment"]]
    fig_h      = max(5, len(rank_df) * 0.6 + 1.5)
 
    fig, ax = plt.subplots(figsize=(11, fig_h))
    bars_h = ax.barh(rank_df["experiment"], rank_df["rmspe_reduction"],
                     color=bar_clr_h, alpha=0.87)
    x_max = rank_df["rmspe_reduction"].max()
    for bar, (_, row) in zip(bars_h, rank_df.iterrows()):
        w_b = bar.get_width()
        ax.text(w_b + x_max * 0.02,
                bar.get_y() + bar.get_height()/2,
                f"{w_b:+.2f}pp   effic={row['efficiency']:.2f}",
                va="center", fontsize=8)
    ax.axvline(0, color="black", linewidth=0.8, linestyle="--")
    ax.set_xlabel("RMSPE Reduction (percentage points)", fontsize=11)
    ax.set_xlim(0, x_max * 1.55)
    ax.set_title(
        "Ablation Importance Ranking\n"
        "(efficiency = pp RMSPE gained per 1% of predictions excluded  |  "
        "higher efficiency = more targeted fix)",
        fontsize=12, fontweight="bold"
    )
    ax.grid(axis="x", alpha=0.25)
    plt.tight_layout()
    fig.savefig(FIGURES_DIR / "11c_ablation_importance_ranking.png",
                dpi=150, bbox_inches="tight")
    plt.close()
 
    print("\nSaved: 11a_ablation_rmspe_comparison.png")
    print("       11b_ablation_error_mass.png")
    print("       11c_ablation_importance_ranking.png")
    return res

def analysis_october_contradiction(df: pd.DataFrame):
    """
    Resolves the apparent contradiction between:
      - October appearing as a high-RMSPE month in diagnostic heatmaps
      - Excluding October producing only marginal improvement in overall metrics

    Produces:
      12a_oct_monthly_context.png        — monthly RMSPE + mini ablation
      12b_oct_bucket_breakdown.png       — October vs other months by bucket
      12c_oct_feature_interactions.png   — Promo / Weekend / Holiday / DOW
      12d_oct_store_concentration.png    — scatter + contribution by store
    """
    print("\n" + "═"*60)
    print("ANALYSIS 12 — October Contradiction Investigation")
    print("═"*60)

    # ── Baseline ──────────────────────────────────────────────────────────────
    total_spe      = df["spe"].sum()
    baseline_rmspe = rmspe(df["y_actual"].values, df["y_pred"].values)
    n_total        = len(df)

    oct             = df[df["month"] == 10].copy()
    non_oct         = df[df["month"] != 10].copy()
    n_oct           = len(oct)
    oct_pct         = n_oct / n_total * 100
    oct_spe         = oct["spe"].sum()
    oct_spe_pct     = oct_spe / total_spe * 100
    oct_rmspe       = rmspe(oct["y_actual"].values, oct["y_pred"].values)
    non_oct_rmspe   = rmspe(non_oct["y_actual"].values, non_oct["y_pred"].values)

    # Mini ablation — exclude October
    rmspe_no_oct  = non_oct_rmspe
    delta_no_oct  = baseline_rmspe - rmspe_no_oct

    # Perfect-October scenario — zero out October SPE, keep row count
    perf_spe      = np.concatenate([non_oct["spe"].values,
                                     np.zeros(n_oct)])
    rmspe_perf    = float(np.sqrt(perf_spe.mean()) * 100)
    delta_perf    = baseline_rmspe - rmspe_perf

    print(f"\nOctober snapshot:")
    print(f"  Predictions : {n_oct:,} of {n_total:,} ({oct_pct:.1f}%)")
    print(f"  Error share : {oct_spe_pct:.1f}% of total SPE")
    print(f"  Oct RMSPE   : {oct_rmspe:.2f}%")
    print(f"  Non-Oct RMSPE: {non_oct_rmspe:.2f}%")
    print(f"  Baseline RMSPE: {baseline_rmspe:.2f}%")
    print(f"\nMini ablation:")
    print(f"  Excl. October       → {rmspe_no_oct:.2f}%  ({delta_no_oct:+.2f}pp)")
    print(f"  Perfect Oct forecast → {rmspe_perf:.2f}%  ({delta_perf:+.2f}pp)")

    # ── Monthly context ───────────────────────────────────────────────────────
    monthly_rows = []
    for m in range(1, 13):
        sub = df[df["month"] == m]
        if len(sub) >= 20:
            monthly_rows.append({
                "month":       m,
                "month_name":  MONTH_NAMES[m - 1],
                "rmspe":       rmspe(sub["y_actual"].values, sub["y_pred"].values),
                "n":           len(sub),
                "pct_total":   len(sub) / n_total * 100,
                "spe_pct":     sub["spe"].sum() / total_spe * 100,
            })
    monthly_df = pd.DataFrame(monthly_rows)

    # ── Bucket-level comparison: October vs other months ──────────────────────
    bucket_rows = []
    for bucket in BUCKET_ORDER:
        for is_oct, label in [(True, "October"), (False, "Other months")]:
            mask = (df["horizon_bucket"] == bucket) & \
                   ((df["month"] == 10) if is_oct else (df["month"] != 10))
            sub = df[mask]
            if len(sub) >= 10:
                bucket_rows.append({
                    "bucket":   bucket,
                    "period":   label,
                    "rmspe":    rmspe(sub["y_actual"].values, sub["y_pred"].values),
                    "spe_pct":  sub["spe"].sum() / total_spe * 100,
                    "n":        len(sub),
                })
    bucket_df = pd.DataFrame(bucket_rows)

    print("\nOctober vs other months RMSPE by bucket:")
    oct_premiums = {}
    for bucket in BUCKET_ORDER:
        bsub = bucket_df[bucket_df["bucket"] == bucket]
        r_o  = bsub[bsub["period"] == "October"]["rmspe"].values
        r_n  = bsub[bsub["period"] == "Other months"]["rmspe"].values
        if r_o.size and r_n.size:
            diff = float(r_o[0] - r_n[0])
            oct_premiums[bucket] = diff
            print(f"  {bucket:8s}: Oct={r_o[0]:.1f}%  "
                  f"Other={r_n[0]:.1f}%  Δ={diff:+.1f}pp")

    # ── Feature interactions within October ───────────────────────────────────
    feat_rows = []
    for col, vals, label_map in [
        ("is_promo",          [False, True],
         {False: "No Promo",    True: "Promo"}),
        ("is_weekend",        [False, True],
         {False: "Weekday",     True: "Weekend"}),
        ("is_school_holiday", [False, True],
         {False: "No School H", True: "School H"}),
        ("is_public_holiday", [False, True],
         {False: "Normal day",  True: "Public H"}),
    ]:
        if col not in df.columns:
            continue
        for v in vals:
            sub_o = oct[oct[col] == v]
            sub_a = df[df[col] == v]
            if len(sub_o) >= 5 and len(sub_a) >= 5:
                feat_rows.append({
                    "feature":       col.replace("is_", "").replace("_", " ").title(),
                    "value":         label_map.get(v, str(v)),
                    "rmspe_oct":     rmspe(sub_o["y_actual"].values,
                                           sub_o["y_pred"].values),
                    "rmspe_annual":  rmspe(sub_a["y_actual"].values,
                                           sub_a["y_pred"].values),
                    "spe_pct_oct":   sub_o["spe"].sum() / oct_spe * 100,
                    "n_oct":         len(sub_o),
                })
    feat_df = pd.DataFrame(feat_rows)

    # Day-of-week within October
    dow_rows = []
    for d in range(1, 8):
        sub = oct[oct["dow"] == d] if "dow" in oct.columns else pd.DataFrame()
        if len(sub) >= 5:
            dow_rows.append({
                "dow":   d,
                "label": DOW_LABELS[d],
                "rmspe": rmspe(sub["y_actual"].values, sub["y_pred"].values),
                "n":     len(sub),
            })
    dow_df = pd.DataFrame(dow_rows)

    # ── Store-level concentration ─────────────────────────────────────────────
    store_rows = []
    if "store" in df.columns:
        for sid in oct["store"].unique():
            so = oct[oct["store"] == sid]
            sa = df[df["store"] == sid]
            if len(so) >= 5 and len(sa) >= 20:
                r_o = rmspe(so["y_actual"].values, so["y_pred"].values)
                r_a = rmspe(sa["y_actual"].values, sa["y_pred"].values)
                store_rows.append({
                    "store":       sid,
                    "rmspe_oct":   r_o,
                    "rmspe_annual":r_a,
                    "oct_premium": r_o - r_a,
                    "spe_oct":     so["spe"].sum(),
                    "spe_pct_oct": so["spe"].sum() / oct_spe * 100,
                    "storetype":   so["storetype"].iloc[0]
                                   if "storetype" in so.columns else "?",
                })
    store_oct = pd.DataFrame(store_rows)

    top5_spe_pct = np.nan
    if not store_oct.empty:
        top5_spe_pct = (store_oct.nlargest(5, "spe_oct")["spe_oct"].sum()
                        / oct_spe * 100)
        n_elevated   = (store_oct["oct_premium"] > 5).sum()
        print(f"\nStore concentration in October ({len(store_oct)} stores):")
        print(f"  Top-5 stores account for {top5_spe_pct:.0f}% of October SPE")
        print(f"  {n_elevated} stores ({n_elevated/len(store_oct)*100:.0f}%) "
              f"have Oct RMSPE >5pp above their annual average")
        print(f"  Median Oct premium: {store_oct['oct_premium'].median():+.1f}pp")

    # ── Print observations ─────────────────────────────────────────────────────
    print("\n── Why the Contradiction? ────────────────────────────────────")

    print(f"\n  H1. Volume:")
    spe_ratio = oct_spe_pct / oct_pct if oct_pct > 0 else 1.0
    print(f"      October = {oct_pct:.1f}% of predictions, "
          f"{oct_spe_pct:.1f}% of error mass (ratio {spe_ratio:.2f})")
    if spe_ratio < 1.15:
        print("      → October's SPE share is close to its prediction share.")
        print("        Elevated RMSPE rate does not translate to dominant error mass.")
    else:
        print("      → October error mass is disproportionate to its prediction share.")

    print(f"\n  H2. Ambient noise (other months nearly as bad):")
    print(f"      Non-Oct RMSPE = {non_oct_rmspe:.2f}% (baseline = {baseline_rmspe:.2f}%)")
    print(f"      Gap = {baseline_rmspe - non_oct_rmspe:.2f}pp — "
          f"{'small: other months pull the average up almost as much' if abs(delta_no_oct) < 1.5 else 'meaningful gap'}")
    if not monthly_df.empty:
        months_above_oct = monthly_df[monthly_df["rmspe"] >= oct_rmspe]["month_name"].tolist()
        if months_above_oct:
            print(f"      Months with RMSPE ≥ October: {', '.join(months_above_oct)}")
            print(f"      → October is NOT uniquely the worst month; "
                  f"removing it leaves similarly noisy months behind.")

    print(f"\n  H3. Bucket specificity:")
    if oct_premiums:
        worst_b  = max(oct_premiums, key=oct_premiums.get)
        best_b   = min(oct_premiums, key=oct_premiums.get)
        spread   = max(oct_premiums.values()) - min(oct_premiums.values())
        print(f"      October premium ranges from {min(oct_premiums.values()):+.1f}pp "
              f"({best_b}) to {max(oct_premiums.values()):+.1f}pp ({worst_b})")
        if spread > 8:
            print(f"      → October is a BUCKET-SPECIFIC problem (worst in {worst_b}).")
            print(f"        In other buckets October is barely elevated — diagnostic")
            print(f"        heatmaps of individual buckets amplify this local spike.")

    if not store_oct.empty:
        print(f"\n  H4. Store concentration:")
        print(f"      Top-5 stores = {top5_spe_pct:.0f}% of October error mass.")
        if top5_spe_pct > 55:
            print(f"      → October error is concentrated in stores that also have")
            print(f"        elevated annual RMSPE. October amplifies an existing")
            print(f"        store-level problem — it is not a calendar effect per se.")

    print(f"\n  Summary:")
    reasons = []
    if abs(delta_no_oct) < 1.5:
        reasons.append("other months are nearly as noisy (ambient error floor)")
    if spe_ratio < 1.2:
        reasons.append("October's prediction share is small")
    if oct_premiums and (max(oct_premiums.values()) - min(oct_premiums.values())) > 6:
        reasons.append(f"October elevation is bucket-specific ({worst_b} only)")
    if not store_oct.empty and top5_spe_pct > 50:
        reasons.append("October error is concentrated in a handful of stores")
    for i, r in enumerate(reasons, 1):
        print(f"    {i}. {r.capitalize()}.")
    print("─" * 60)

    # ── FIGURE 12a: Monthly RMSPE context + mini ablation ────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(15, 5))

    ax = axes[0]
    if not monthly_df.empty:
        colors_m = ["#EF5350" if m == 10 else "#90CAF9"
                    for m in monthly_df["month"]]
        bars = ax.bar(monthly_df["month_name"], monthly_df["rmspe"],
                      color=colors_m, alpha=0.87)
        ax.axhline(baseline_rmspe, color="black", linewidth=1.2,
                   linestyle="--", label=f"Baseline {baseline_rmspe:.2f}%")
        ax.axhline(non_oct_rmspe, color="#FF9800", linewidth=1.2,
                   linestyle=":", label=f"Non-Oct {non_oct_rmspe:.2f}%")
        for bar, val in zip(bars, monthly_df["rmspe"]):
            ax.text(bar.get_x() + bar.get_width()/2, val + 0.2,
                    f"{val:.0f}%", ha="center", va="bottom", fontsize=7)
        ax.set_ylabel("RMSPE (%)", fontsize=10)
        ax.yaxis.set_major_formatter(mticker.PercentFormatter())
        ax.set_title("Monthly RMSPE  (October highlighted in red)\n"
                     "Orange dotted = non-October baseline",
                     fontweight="bold", fontsize=10)
        ax.legend(fontsize=8)
        ax.tick_params(axis="x", rotation=45, labelsize=8)
        ax.grid(axis="y", alpha=0.25)

    ax = axes[1]
    scenarios = ["Baseline", "Excl. October", "Perfect\nOct Forecast"]
    values    = [baseline_rmspe, rmspe_no_oct, rmspe_perf]
    colors_ab = ["#9E9E9E", "#2196F3", "#4CAF50"]
    bars2 = ax.bar(scenarios, values, color=colors_ab, alpha=0.88, width=0.5)
    for bar, val in zip(bars2, values):
        ax.text(bar.get_x() + bar.get_width()/2, val + 0.05,
                f"{val:.3f}%", ha="center", va="bottom",
                fontsize=9, fontweight="bold")
    # Annotate deltas
    for xi, (val, base) in enumerate(zip(values[1:], [baseline_rmspe]*2), 1):
        delta = val - base
        ax.annotate(f"{delta:+.2f}pp",
                    xy=(xi, val), xytext=(0, 12), textcoords="offset points",
                    ha="center", fontsize=9, fontweight="bold",
                    color="#1B5E20" if delta < 0 else "#B71C1C")
    ax.set_ylabel("RMSPE (%)", fontsize=10)
    ax.yaxis.set_major_formatter(mticker.PercentFormatter())
    ax.set_title("Mini Ablation — October Impact\n"
                 "'Perfect' = zero error on all October predictions",
                 fontweight="bold", fontsize=10)
    ax.grid(axis="y", alpha=0.25)
    y_min = min(values) * 0.97
    y_max = max(values) * 1.06
    ax.set_ylim(y_min, y_max)

    plt.suptitle("October Context — Monthly Profile and Ablation Impact",
                 fontsize=12, fontweight="bold")
    plt.tight_layout()
    fig.savefig(FIGURES_DIR / "12a_oct_monthly_context.png",
                dpi=150, bbox_inches="tight")
    plt.close()

    # ── FIGURE 12b: Bucket breakdown ──────────────────────────────────────────
    if not bucket_df.empty:
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))

        ax = axes[0]
        x  = np.arange(len(BUCKET_ORDER))
        w  = 0.38
        for offset, period, color in [
            (-w/2, "Other months", "#90CAF9"),
            ( w/2, "October",      "#EF5350"),
        ]:
            vals = [bucket_df[(bucket_df["bucket"] == b) &
                               (bucket_df["period"] == period)]["rmspe"].values
                    for b in BUCKET_ORDER]
            vals = [v[0] if len(v) else np.nan for v in vals]
            bars = ax.bar(x + offset, vals, w, label=period, color=color, alpha=0.87)
            for bar, v in zip(bars, vals):
                if not np.isnan(v):
                    ax.text(bar.get_x() + bar.get_width()/2, v + 0.2,
                            f"{v:.0f}%", ha="center", va="bottom", fontsize=8)
        ax.set_xticks(x)
        ax.set_xticklabels([b.upper() for b in BUCKET_ORDER], fontsize=10)
        ax.set_ylabel("RMSPE (%)", fontsize=10)
        ax.yaxis.set_major_formatter(mticker.PercentFormatter())
        ax.set_title("RMSPE by Bucket: October vs Other Months\n"
                     "(shows whether October elevation is bucket-specific)",
                     fontweight="bold", fontsize=10)
        ax.legend(fontsize=9)
        ax.grid(axis="y", alpha=0.25)

        ax = axes[1]
        oct_bucket_spe = [
            bucket_df[(bucket_df["bucket"] == b) &
                       (bucket_df["period"] == "October")]["spe_pct"].values
            for b in BUCKET_ORDER
        ]
        oct_bucket_spe = [v[0] if len(v) else 0.0 for v in oct_bucket_spe]
        bars3 = ax.bar([b.upper() for b in BUCKET_ORDER], oct_bucket_spe,
                       color="#EF5350", alpha=0.85)
        for bar, v in zip(bars3, oct_bucket_spe):
            ax.text(bar.get_x() + bar.get_width()/2, v + 0.01,
                    f"{v:.2f}%", ha="center", va="bottom", fontsize=9)
        ax.set_ylabel("% of Total Squared Error", fontsize=10)
        ax.yaxis.set_major_formatter(
            mticker.FuncFormatter(lambda x, _: f"{x:.2f}%"))
        ax.set_title("October SPE Contribution by Bucket\n"
                     "(context: which bucket drives October error mass)",
                     fontweight="bold", fontsize=10)
        ax.grid(axis="y", alpha=0.25)

        plt.suptitle("October — Bucket-Level Breakdown",
                     fontsize=12, fontweight="bold")
        plt.tight_layout()
        fig.savefig(FIGURES_DIR / "12b_oct_bucket_breakdown.png",
                    dpi=150, bbox_inches="tight")
        plt.close()

    # ── FIGURE 12c: Feature interactions ─────────────────────────────────────
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    plt.suptitle("October — Feature Interactions\n"
                 "(orange = October RMSPE, blue = annual RMSPE for same condition)",
                 fontsize=12, fontweight="bold")

    def grouped_feat_bars(ax, feature_name, title):
        sub = feat_df[feat_df["feature"] == feature_name]
        if sub.empty:
            ax.set_visible(False)
            return
        x   = np.arange(len(sub))
        w   = 0.38
        b1  = ax.bar(x - w/2, sub["rmspe_annual"].values, w,
                     label="Annual", color="#90CAF9", alpha=0.9)
        b2  = ax.bar(x + w/2, sub["rmspe_oct"].values, w,
                     label="October", color="#FF9800", alpha=0.9)
        for bar, val in zip(list(b1) + list(b2), 
                             list(sub["rmspe_annual"]) + list(sub["rmspe_oct"])):
            ax.text(bar.get_x() + bar.get_width()/2, val + 0.2,
                    f"{val:.0f}%", ha="center", va="bottom", fontsize=8)
        ax.set_xticks(x)
        ax.set_xticklabels(sub["value"].tolist(), fontsize=9)
        ax.set_ylabel("RMSPE (%)", fontsize=9)
        ax.yaxis.set_major_formatter(mticker.PercentFormatter())
        ax.set_title(title, fontweight="bold", fontsize=10)
        ax.legend(fontsize=8)
        ax.grid(axis="y", alpha=0.25)

    grouped_feat_bars(axes[0, 0], "Promo",          "Promo: October vs Annual")
    grouped_feat_bars(axes[0, 1], "Weekend",         "Weekend: October vs Annual")
    grouped_feat_bars(axes[1, 0], "School Holiday",  "School Holiday: Oct vs Annual")
    grouped_feat_bars(axes[1, 1], "Public Holiday",  "Public Holiday: Oct vs Annual")

    # Override (1,1) with DayOfWeek if DOW data is available and Public H is sparse
    if not dow_df.empty:
        ax = axes[1, 1]
        ax.clear()
        ax.bar(dow_df["label"], dow_df["rmspe"], color="#9C27B0", alpha=0.85)
        for _, row in dow_df.iterrows():
            ax.text(list(dow_df["label"]).index(row["label"]),
                    row["rmspe"] + 0.2, f"{row['rmspe']:.0f}%",
                    ha="center", va="bottom", fontsize=8)
        # Re-annotate x-ticks properly
        ax.set_xticks(range(len(dow_df)))
        ax.set_xticklabels(dow_df["label"].tolist(), fontsize=9)
        ax.set_ylabel("RMSPE (%)", fontsize=9)
        ax.yaxis.set_major_formatter(mticker.PercentFormatter())
        ax.set_title("Day of Week RMSPE within October\n"
                     "(Unity Day 2014 = Friday Oct 3)",
                     fontweight="bold", fontsize=10)
        ax.grid(axis="y", alpha=0.25)

    plt.tight_layout()
    fig.savefig(FIGURES_DIR / "12c_oct_feature_interactions.png",
                dpi=150, bbox_inches="tight")
    plt.close()

    # ── FIGURE 12d: Store concentration ───────────────────────────────────────
    if not store_oct.empty:
        fig, axes = plt.subplots(1, 2, figsize=(15, 6))

        # Left: scatter — annual RMSPE vs October RMSPE per store
        ax = axes[0]
        type_colors = {"a": "#2196F3", "b": "#EF5350",
                       "c": "#4CAF50", "d": "#FF9800"}
        sc_colors = [type_colors.get(str(t), "#9E9E9E")
                     for t in store_oct["storetype"]]
        ax.scatter(store_oct["rmspe_annual"], store_oct["rmspe_oct"],
                   c=sc_colors, alpha=0.65, s=35, edgecolors="none")
        # Diagonal — y = x (no October premium)
        diag_max = max(store_oct["rmspe_annual"].max(),
                       store_oct["rmspe_oct"].max()) * 1.05
        ax.plot([0, diag_max], [0, diag_max], color="gray",
                linestyle="--", linewidth=1, label="y = x (no premium)")
        # Annotate worst 5 stores
        worst5 = store_oct.nlargest(5, "oct_premium")
        for _, row in worst5.iterrows():
            ax.annotate(f"S{int(row['store'])}",
                        xy=(row["rmspe_annual"], row["rmspe_oct"]),
                        xytext=(4, 4), textcoords="offset points",
                        fontsize=7, color="darkred")
        # Legend for store types
        for t, c in type_colors.items():
            ax.scatter([], [], c=c, label=f"Type {t}")
        ax.set_xlabel("Annual RMSPE (%)", fontsize=10)
        ax.set_ylabel("October RMSPE (%)", fontsize=10)
        ax.set_title("Store Scatter: Annual vs October RMSPE\n"
                     "Points above diagonal = worse in October than usual",
                     fontweight="bold", fontsize=10)
        ax.legend(fontsize=8, title="StoreType", loc="upper left")
        ax.grid(alpha=0.22)

        # Right: top-20 stores by October SPE contribution
        ax = axes[1]
        top20 = store_oct.nlargest(20, "spe_oct").sort_values("spe_pct_oct")
        bar_clr = [type_colors.get(str(t), "#9E9E9E")
                   for t in top20["storetype"]]
        ax.barh(range(len(top20)), top20["spe_pct_oct"],
                color=bar_clr, alpha=0.85)
        ax.set_yticks(range(len(top20)))
        ax.set_yticklabels([f"Store {int(s)}" for s in top20["store"]],
                            fontsize=7)
        ax.set_xlabel("% of October Squared Error", fontsize=10)
        ax.xaxis.set_major_formatter(
            mticker.FuncFormatter(lambda x, _: f"{x:.1f}%"))
        ax.set_title("Top-20 Stores — Share of October Error Mass\n"
                     "(colour = StoreType)",
                     fontweight="bold", fontsize=10)
        for t, c in type_colors.items():
            ax.barh([], [], color=c, label=f"Type {t}")
        ax.legend(fontsize=7, title="StoreType")
        ax.grid(axis="x", alpha=0.25)

        plt.suptitle("October — Store-Level Concentration",
                     fontsize=12, fontweight="bold")
        plt.tight_layout()
        fig.savefig(FIGURES_DIR / "12d_oct_store_concentration.png",
                    dpi=150, bbox_inches="tight")
        plt.close()

    print("\nSaved: 12a_oct_monthly_context.png")
    print("       12b_oct_bucket_breakdown.png")
    print("       12c_oct_feature_interactions.png")
    if not store_oct.empty:
        print("       12d_oct_store_concentration.png")

    return {
        "monthly":     monthly_df,
        "buckets":     bucket_df,
        "features":    feat_df,
        "store_oct":   store_oct,
        "ablation": {
            "baseline":    baseline_rmspe,
            "no_october":  rmspe_no_oct,
            "perfect_oct": rmspe_perf,
        },
    }

def analysis_mid_horizon_anomaly(df: pd.DataFrame):
    """
    Determines why the Mid horizon bucket (days 15–30) shows the highest
    RMSPE despite calendar and store analyses not fully explaining it.

    Produces:
      13a_mid_bucket_month_heatmap.png       — Horizon × Month RMSPE
      13b_mid_bucket_storetype_heatmap.png   — Horizon × StoreType RMSPE
      13c_mid_feature_interactions.png       — Promo / Holiday / Weekend by bucket
      13d_mid_error_distributions.png        — SPE distribution per bucket
      13e_mid_calendar_composition.png       — Calendar loading per bucket
      13f_mid_ablation.png                   — Mid RMSPE after stripping October
    """
    print("\n" + "═"*60)
    print("ANALYSIS 13 — Mid Horizon Anomaly Investigation")
    print("═"*60)

    # ── Baseline per bucket ───────────────────────────────────────────────────
    total_spe = df["spe"].sum()
    n_total   = len(df)

    print(f"\nBaseline RMSPE and composition per bucket:")
    print(f"  {'Bucket':8s} {'RMSPE':>7} {'N':>9} {'N%':>6} {'SPE%':>7}")
    print("  " + "─"*42)
    bucket_baseline = {}
    for b in BUCKET_ORDER:
        sub = df[df["horizon_bucket"] == b]
        r   = rmspe(sub["y_actual"].values, sub["y_pred"].values)
        n   = len(sub)
        bucket_baseline[b] = {
            "rmspe": r, "n": n,
            "n_pct": n / n_total * 100,
            "spe_pct": sub["spe"].sum() / total_spe * 100,
        }
        print(f"  {b:8s} {r:>6.2f}%  {n:>9,}  "
              f"{n/n_total*100:>5.1f}%  "
              f"{sub['spe'].sum()/total_spe*100:>6.1f}%")

    mid_rmspe = bucket_baseline["mid"]["rmspe"]

    # ── H1: Calendar composition — month loading per bucket ──────────────────
    print("\nH1 — Calendar composition: month loading per bucket (% of bucket N)")
    print(f"  {'Month':>5}", end="")
    for b in BUCKET_ORDER:
        print(f"  {b:>10}", end="")
    print()
    print("  " + "─"*50)

    month_load = {}
    for m_idx, m_name in enumerate(MONTH_NAMES, start=1):
        row_vals = {}
        print(f"  {m_name:>5}", end="")
        for b in BUCKET_ORDER:
            sub = df[(df["horizon_bucket"] == b) & (df["month"] == m_idx)]
            n_b = bucket_baseline[b]["n"]
            pct = len(sub) / n_b * 100 if n_b > 0 else 0.0
            row_vals[b] = pct
            print(f"  {pct:>9.1f}%", end="")
        month_load[m_name] = row_vals
        print()

    # October loading in mid vs other buckets
    oct_load = {b: month_load["Oct"][b] for b in BUCKET_ORDER}
    print(f"\n  October loading: {' | '.join(f'{b}={v:.1f}%' for b, v in oct_load.items())}")
    if oct_load["mid"] > max(oct_load[b] for b in BUCKET_ORDER if b != "mid") * 1.15:
        print("  → October is DISPROPORTIONATELY loaded into Mid (confirms H1)")
    else:
        print("  → October load is roughly equal across buckets (H1 weak)")

    # ── H1 falsification: strip October from mid ─────────────────────────────
    mid_no_oct  = df[(df["horizon_bucket"] == "mid") & (df["month"] != 10)]
    mid_oct     = df[(df["horizon_bucket"] == "mid") & (df["month"] == 10)]
    rmspe_mid_no_oct = rmspe(mid_no_oct["y_actual"].values, mid_no_oct["y_pred"].values)
    rmspe_mid_oct    = rmspe(mid_oct["y_actual"].values, mid_oct["y_pred"].values) \
                       if len(mid_oct) >= 5 else np.nan

    far_rmspe  = bucket_baseline["far"]["rmspe"]
    ext_rmspe  = bucket_baseline["extended"]["rmspe"]

    print(f"\n  Mid RMSPE (all months)    : {mid_rmspe:.2f}%")
    print(f"  Mid RMSPE (Oct only)      : "
          f"{'—' if np.isnan(rmspe_mid_oct) else f'{rmspe_mid_oct:.2f}%'}")
    print(f"  Mid RMSPE (excl. October) : {rmspe_mid_no_oct:.2f}%")
    print(f"  Far RMSPE                 : {far_rmspe:.2f}%")
    print(f"  Extended RMSPE            : {ext_rmspe:.2f}%")

    delta_h1 = mid_rmspe - rmspe_mid_no_oct
    if rmspe_mid_no_oct <= max(far_rmspe, ext_rmspe) + 1.0:
        h1_verdict = ("CONFIRMED — Mid RMSPE collapses to Far/Extended level "
                      "after removing October. Anomaly is entirely compositional.")
    elif delta_h1 > 2.0:
        h1_verdict = ("PARTIAL — October explains most of the gap. "
                      "Residual gap suggests secondary factors also present.")
    else:
        h1_verdict = ("WEAK — Removing October barely changes Mid RMSPE. "
                      "Model genuinely struggles at this horizon range.")
    print(f"\n  H1 verdict: {h1_verdict}")

    # ── H2: Store composition ─────────────────────────────────────────────────
    print("\nH2 — Store composition: do hard stores appear more in Mid?")
    if "store" in df.columns:
        # Per-store annual RMSPE
        store_rmspe_map = (
            df.groupby("store")
              .apply(lambda g: rmspe(g["y_actual"].values, g["y_pred"].values))
              .to_dict()
        )
        # Map back into df
        df = df.copy()
        df["store_annual_rmspe"] = df["store"].map(store_rmspe_map)

        store_comp = {}
        for b in BUCKET_ORDER:
            sub   = df[df["horizon_bucket"] == b]
            mean_r= sub["store_annual_rmspe"].mean()
            store_comp[b] = mean_r
            print(f"  {b:8s}: mean annual store RMSPE in bucket = {mean_r:.2f}%")

        mid_store_mean = store_comp["mid"]
        other_store_means = {b: v for b, v in store_comp.items() if b != "mid"}
        if mid_store_mean > max(other_store_means.values()) * 1.05:
            print("  → Mid contains disproportionately harder stores (H2 partial)")
        else:
            print("  → Store composition is similar across buckets (H2 weak)")
    else:
        print("  ⚠️  store column missing — skipping H2")

    # ── H3: Genuine model decay ───────────────────────────────────────────────
    print("\nH3 — Genuine model decay: per-day RMSPE monotonicity check")
    day_rmspe = {}
    for h in range(1, 91):
        sub = df[df["horizon"] == h]
        if len(sub) >= 10:
            day_rmspe[h] = rmspe(sub["y_actual"].values, sub["y_pred"].values)

    # Bucket-level average of daily RMSPE (separates composition from decay)
    for b, (lo, hi) in BUCKET_RANGES.items():
        vals = [v for h, v in day_rmspe.items() if lo <= h <= hi]
        if vals:
            print(f"  {b:8s} (days {lo:2d}–{hi:2d}): "
                  f"mean per-day RMSPE = {np.mean(vals):.2f}%  "
                  f"trend = "
                  f"{'↑' if vals[-1] > vals[0] else '↓' if vals[-1] < vals[0] else '→'}")

    # ── Horizon × Month RMSPE table ───────────────────────────────────────────
    bm_records = []
    for b in BUCKET_ORDER:
        for m_idx, m_name in enumerate(MONTH_NAMES, start=1):
            sub = df[(df["horizon_bucket"] == b) & (df["month"] == m_idx)]
            if len(sub) >= 20:
                bm_records.append({
                    "bucket":      b,
                    "month":       m_idx,
                    "month_name":  m_name,
                    "rmspe":       rmspe(sub["y_actual"].values, sub["y_pred"].values),
                    "spe_pct":     sub["spe"].sum() / total_spe * 100,
                    "n":           len(sub),
                })
    bm_df = pd.DataFrame(bm_records)

    # ── Horizon × StoreType table ─────────────────────────────────────────────
    bt_records = []
    if "storetype" in df.columns:
        store_types = sorted(df["storetype"].dropna().unique())
        for b in BUCKET_ORDER:
            for st in store_types:
                sub = df[(df["horizon_bucket"] == b) & (df["storetype"] == st)]
                if len(sub) >= 20:
                    bt_records.append({
                        "bucket":    b,
                        "storetype": st,
                        "rmspe":     rmspe(sub["y_actual"].values, sub["y_pred"].values),
                        "spe_pct":   sub["spe"].sum() / total_spe * 100,
                        "n":         len(sub),
                    })
    bt_df = pd.DataFrame(bt_records)

    # ── Feature interaction tables ────────────────────────────────────────────
    feature_specs = []
    for col, vals, label_map, feat_name in [
        ("is_promo",          [False, True],
         {False: "No Promo", True: "Promo"},       "Promo"),
        ("is_weekend",        [False, True],
         {False: "Weekday",  True: "Weekend"},      "Weekend"),
        ("is_school_holiday", [False, True],
         {False: "No SchH",  True: "School Hol"},   "SchoolH"),
        ("is_public_holiday", [False, True],
         {False: "Normal",   True: "State Hol"},    "StateH"),
    ]:
        if col not in df.columns:
            continue
        for b in BUCKET_ORDER:
            for v in vals:
                sub = df[(df["horizon_bucket"] == b) & (df[col] == v)]
                if len(sub) >= 10:
                    feature_specs.append({
                        "feature": feat_name,
                        "bucket":  b,
                        "value":   label_map.get(v, str(v)),
                        "active":  v,
                        "rmspe":   rmspe(sub["y_actual"].values, sub["y_pred"].values),
                        "spe_pct": sub["spe"].sum() / total_spe * 100,
                        "n":       len(sub),
                    })
    feat_df = pd.DataFrame(feature_specs)

    # ── Calendar composition per bucket (month × bucket % loading) ────────────
    comp_records = []
    for b in BUCKET_ORDER:
        n_b = bucket_baseline[b]["n"]
        for m_idx, m_name in enumerate(MONTH_NAMES, start=1):
            n_bm = len(df[(df["horizon_bucket"] == b) & (df["month"] == m_idx)])
            comp_records.append({
                "bucket":     b,
                "month":      m_idx,
                "month_name": m_name,
                "pct_in_bucket": n_bm / n_b * 100 if n_b > 0 else 0.0,
            })
    comp_df = pd.DataFrame(comp_records)

    # ── Observations ──────────────────────────────────────────────────────────
    print("\n── Summary Observations ──────────────────────────────────────")
    print(f"\n  Mid RMSPE = {mid_rmspe:.2f}%  "
          f"(Far={far_rmspe:.2f}%  Extended={ext_rmspe:.2f}%)")
    print(f"  Mid without October = {rmspe_mid_no_oct:.2f}%  "
          f"(reduction: {delta_h1:.2f}pp = "
          f"{delta_h1/mid_rmspe*100:.0f}% of anomaly explained by H1)")

    if not bm_df.empty:
        mid_month = bm_df[bm_df["bucket"] == "mid"]
        if not mid_month.empty:
            worst_m = mid_month.loc[mid_month["rmspe"].idxmax()]
            print(f"\n  Worst month for Mid : {worst_m['month_name']} "
                  f"({worst_m['rmspe']:.1f}%)")

    if not bt_df.empty:
        mid_type = bt_df[bt_df["bucket"] == "mid"]
        if not mid_type.empty:
            worst_t = mid_type.loc[mid_type["rmspe"].idxmax()]
            print(f"  Worst StoreType for Mid : Type {worst_t['storetype'].upper()} "
                  f"({worst_t['rmspe']:.1f}%)")

    print(f"\n  Root cause determination:")
    if delta_h1 >= 3.0:
        print(f"    PRIMARY: Calendar composition (H1). October falls")
        print(f"    disproportionately in the Mid window due to September")
        print(f"    origins. Fix: origin-aware resampling or October-specific")
        print(f"    features (days_to_unity_day, herbstferien_flag).")
    elif delta_h1 >= 1.5:
        print(f"    PARTIAL: H1 explains some of the gap. Investigate H2 and H3.")
    else:
        print(f"    H1 is WEAK. Mid anomaly is not primarily calendar composition.")
        print(f"    Check H3 (per-day RMSPE chart in script 13 Fig 2).")
        print(f"    Model may genuinely under-use medium-lag features (14–30 days).")
    print("─" * 60)

    # ── FIGURE 13a: Horizon × Month heatmap ──────────────────────────────────
    if not bm_df.empty:
        month_cols = [m for m in MONTH_NAMES if m in bm_df["month_name"].unique()]
        pivot_bm   = (bm_df.pivot(index="bucket", columns="month_name", values="rmspe")
                           .reindex(index=BUCKET_ORDER)[month_cols])

        fig, ax = plt.subplots(figsize=(15, 4))
        vals = pivot_bm.values.astype(float)
        im   = ax.imshow(vals, aspect="auto", cmap="RdYlGn_r", vmin=5, vmax=55)
        ax.set_xticks(range(len(pivot_bm.columns)))
        ax.set_xticklabels(pivot_bm.columns, fontsize=10)
        ax.set_yticks(range(len(BUCKET_ORDER)))
        ax.set_yticklabels(BUCKET_ORDER, fontsize=11)

        for i in range(vals.shape[0]):
            for j in range(vals.shape[1]):
                v = vals[i, j]
                if not np.isnan(v):
                    ax.text(j, i, f"{v:.0f}%", ha="center", va="center",
                            fontsize=9, fontweight="bold",
                            color="white" if v > 42 or v < 9 else "black")

        # Highlight October column
        for j, col in enumerate(pivot_bm.columns):
            if col == "Oct":
                ax.add_patch(plt.Rectangle(
                    (j - 0.5, -0.5), 1, len(BUCKET_ORDER),
                    fill=False, edgecolor="navy",
                    linewidth=2.5, clip_on=False, zorder=5))
        plt.colorbar(im, ax=ax, label="RMSPE (%)", shrink=0.8)
        ax.set_title(
            "Horizon Bucket × Month RMSPE Heatmap\n"
            "(navy border = October  |  Mid row hottest = composition artifact?)",
            fontsize=12, fontweight="bold", pad=8
        )
        plt.tight_layout()
        fig.savefig(FIGURES_DIR / "13a_mid_bucket_month_heatmap.png",
                    dpi=150, bbox_inches="tight")
        plt.close()

    # ── FIGURE 13b: Horizon × StoreType heatmap ──────────────────────────────
    if not bt_df.empty:
        store_types = sorted(bt_df["storetype"].unique())
        pivot_bt    = (bt_df.pivot(index="storetype", columns="bucket", values="rmspe")
                            .reindex(columns=BUCKET_ORDER))

        fig, ax = plt.subplots(figsize=(10, max(4, len(store_types) * 1.1 + 1)))
        vals = pivot_bt.values.astype(float)
        im   = ax.imshow(vals, aspect="auto", cmap="RdYlGn_r", vmin=5, vmax=55)
        ax.set_xticks(range(len(BUCKET_ORDER)))
        ax.set_xticklabels(BUCKET_ORDER, fontsize=11)
        ax.set_yticks(range(len(store_types)))
        ax.set_yticklabels([f"Type {t}" for t in store_types], fontsize=11)
        for i in range(vals.shape[0]):
            for j in range(vals.shape[1]):
                v = vals[i, j]
                if not np.isnan(v):
                    ax.text(j, i, f"{v:.0f}%", ha="center", va="center",
                            fontsize=11, fontweight="bold",
                            color="white" if v > 42 or v < 9 else "black")
        # Highlight Mid column
        for j, b in enumerate(BUCKET_ORDER):
            if b == "mid":
                ax.add_patch(plt.Rectangle(
                    (j - 0.5, -0.5), 1, len(store_types),
                    fill=False, edgecolor="navy",
                    linewidth=2.5, clip_on=False, zorder=5))
        plt.colorbar(im, ax=ax, label="RMSPE (%)", shrink=0.8)
        ax.set_title(
            "StoreType × Horizon Bucket RMSPE\n"
            "(navy = Mid  |  if Mid column uniformly hot across types → "
            "calendar composition, not store-specific)",
            fontsize=11, fontweight="bold", pad=8
        )
        plt.tight_layout()
        fig.savefig(FIGURES_DIR / "13b_mid_bucket_storetype_heatmap.png",
                    dpi=150, bbox_inches="tight")
        plt.close()

    # ── FIGURE 13c: Feature interactions by bucket ────────────────────────────
    if not feat_df.empty:
        features = feat_df["feature"].unique()
        n_feat   = len(features)
        fig, axes = plt.subplots(2, 2, figsize=(14, 9))
        plt.suptitle(
            "Horizon Bucket × Calendar Feature RMSPE\n"
            "(if Mid bar is consistently highest within each feature group "
            "→ model horizon effect; if not → composition)",
            fontsize=12, fontweight="bold"
        )
        bucket_colors = {b: c for b, c in zip(
            BUCKET_ORDER, ["#2196F3", "#4CAF50", "#FF9800", "#9C27B0"])}

        for ax, feat_name in zip(axes.flatten(), features):
            sub_f  = feat_df[feat_df["feature"] == feat_name]
            # Only keep "active=True" for the 2-value features
            sub_f  = sub_f[sub_f["active"] == True]
            if sub_f.empty:
                ax.set_visible(False)
                continue
            x      = np.arange(len(BUCKET_ORDER))
            vals_f = [sub_f[sub_f["bucket"] == b]["rmspe"].values for b in BUCKET_ORDER]
            vals_f = [v[0] if len(v) else np.nan for v in vals_f]
            clr    = [bucket_colors.get(b, "#9E9E9E") for b in BUCKET_ORDER]
            bars   = ax.bar(x, vals_f, color=clr, alpha=0.87)
            for bar, v in zip(bars, vals_f):
                if not np.isnan(v):
                    ax.text(bar.get_x() + bar.get_width()/2, v + 0.3,
                            f"{v:.0f}%", ha="center", va="bottom", fontsize=8)
            ax.set_xticks(x)
            ax.set_xticklabels(BUCKET_ORDER, fontsize=9)
            ax.set_ylabel("RMSPE (%)", fontsize=9)
            ax.yaxis.set_major_formatter(mticker.PercentFormatter())
            ax.set_title(f"RMSPE when {feat_name} = active\n"
                         "(bars per bucket, active observations only)",
                         fontweight="bold", fontsize=10)
            ax.grid(axis="y", alpha=0.25)

        # Hide unused subplots
        for ax in axes.flatten()[n_feat:]:
            ax.set_visible(False)

        plt.tight_layout()
        fig.savefig(FIGURES_DIR / "13c_mid_feature_interactions.png",
                    dpi=150, bbox_inches="tight")
        plt.close()

    # ── FIGURE 13d: Error distribution by bucket ──────────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    plt.suptitle("Forecast Error Distribution by Horizon Bucket",
                 fontsize=12, fontweight="bold")

    # Log-SPE violin
    ax = axes[0]
    clip_99 = df["spe"].quantile(0.99)
    plot_data = []
    plot_labels = []
    for b in BUCKET_ORDER:
        sub = df[(df["horizon_bucket"] == b) & (df["spe"] > 0) &
                 (df["spe"] <= clip_99)]
        if len(sub) > 0:
            plot_data.append(np.log10(sub["spe"].values + 1e-8))
            plot_labels.append(b)
    if plot_data:
        vp = ax.violinplot(plot_data, showmedians=True, showextrema=True)
        for i, (pc, b) in enumerate(zip(vp["bodies"], plot_labels)):
            pc.set_facecolor(list(bucket_colors.values())[i])
            pc.set_alpha(0.75)
        ax.set_xticks(range(1, len(plot_labels) + 1))
        ax.set_xticklabels(plot_labels, fontsize=10)
        ax.set_ylabel("log₁₀(SPE)  [clipped at 99th pct]", fontsize=10)
        ax.set_title("SPE Distribution (log scale)\n"
                     "(wider/higher = more error mass and variance)",
                     fontweight="bold", fontsize=10)
        ax.grid(axis="y", alpha=0.25)

    # Cumulative SPE concentration (Lorenz-style)
    ax = axes[1]
    for b, color in zip(BUCKET_ORDER, bucket_colors.values()):
        sub = df[df["horizon_bucket"] == b]["spe"].sort_values()
        if len(sub) < 2:
            continue
        cum  = np.cumsum(sub.values)
        norm = cum / cum[-1]
        pct  = np.arange(1, len(sub) + 1) / len(sub) * 100
        ax.plot(pct, norm * 100, linewidth=2, color=color, label=b)
    ax.plot([0, 100], [0, 100], color="gray", linestyle="--",
            linewidth=1, label="Perfect equality")
    ax.set_xlabel("% of Predictions (sorted by SPE)", fontsize=10)
    ax.set_ylabel("% of Cumulative SPE", fontsize=10)
    ax.yaxis.set_major_formatter(mticker.PercentFormatter())
    ax.xaxis.set_major_formatter(mticker.PercentFormatter())
    ax.set_title("Cumulative SPE Concentration\n"
                 "(curve bowing left = error concentrated in a few predictions)",
                 fontweight="bold", fontsize=10)
    ax.legend(fontsize=9)
    ax.grid(alpha=0.22)

    plt.tight_layout()
    fig.savefig(FIGURES_DIR / "13d_mid_error_distributions.png",
                dpi=150, bbox_inches="tight")
    plt.close()

    # ── FIGURE 13e: Calendar composition per bucket ───────────────────────────
    if not comp_df.empty:
        month_cols = [m for m in MONTH_NAMES if m in comp_df["month_name"].unique()]
        pivot_comp = (comp_df.pivot(index="bucket", columns="month_name",
                                    values="pct_in_bucket")
                             .reindex(index=BUCKET_ORDER)[month_cols])

        fig, ax = plt.subplots(figsize=(15, 4))
        vals = pivot_comp.values.astype(float)
        im   = ax.imshow(vals, aspect="auto", cmap="YlOrRd")
        ax.set_xticks(range(len(pivot_comp.columns)))
        ax.set_xticklabels(pivot_comp.columns, fontsize=10)
        ax.set_yticks(range(len(BUCKET_ORDER)))
        ax.set_yticklabels(BUCKET_ORDER, fontsize=11)
        v_max = vals[~np.isnan(vals)].max() if not np.all(np.isnan(vals)) else 1
        for i in range(vals.shape[0]):
            for j in range(vals.shape[1]):
                v = vals[i, j]
                if not np.isnan(v):
                    ax.text(j, i, f"{v:.0f}%", ha="center", va="center",
                            fontsize=8, fontweight="bold",
                            color="white" if v > v_max * 0.65 else "black")
        # Highlight October
        for j, col in enumerate(pivot_comp.columns):
            if col == "Oct":
                ax.add_patch(plt.Rectangle(
                    (j - 0.5, -0.5), 1, len(BUCKET_ORDER),
                    fill=False, edgecolor="navy",
                    linewidth=2.5, clip_on=False, zorder=5))
        plt.colorbar(im, ax=ax,
                     label="% of Bucket Predictions in that Month", shrink=0.8)
        ax.set_title(
            "Calendar Composition per Bucket  "
            "(% of each bucket's predictions that fall in each month)\n"
            "If Oct cell is hottest in Mid row → September origins "
            "are loading the Mid window onto October",
            fontsize=11, fontweight="bold", pad=8
        )
        plt.tight_layout()
        fig.savefig(FIGURES_DIR / "13e_mid_calendar_composition.png",
                    dpi=150, bbox_inches="tight")
        plt.close()

    # ── FIGURE 13f: Mid ablation — stripping October ──────────────────────────
    ablation_rows = []
    for b in BUCKET_ORDER:
        sub_all = df[df["horizon_bucket"] == b]
        sub_noct = df[(df["horizon_bucket"] == b) & (df["month"] != 10)]
        ablation_rows.append({
            "bucket":         b,
            "rmspe_all":      rmspe(sub_all["y_actual"].values,
                                    sub_all["y_pred"].values),
            "rmspe_no_oct":   rmspe(sub_noct["y_actual"].values,
                                    sub_noct["y_pred"].values)
                              if len(sub_noct) >= 10 else np.nan,
            "n_all":          len(sub_all),
            "n_no_oct":       len(sub_noct),
        })
    abl_df = pd.DataFrame(ablation_rows)

    fig, ax = plt.subplots(figsize=(10, 5))
    x = np.arange(len(BUCKET_ORDER))
    w = 0.38
    b1 = ax.bar(x - w/2, abl_df["rmspe_all"].values, w,
                label="All months",    color="#90CAF9", alpha=0.9)
    b2 = ax.bar(x + w/2, abl_df["rmspe_no_oct"].values, w,
                label="Excl. October", color="#EF5350", alpha=0.9)
    for bar, val in zip(list(b1) + list(b2),
                         list(abl_df["rmspe_all"]) + list(abl_df["rmspe_no_oct"])):
        if not np.isnan(val):
            ax.text(bar.get_x() + bar.get_width()/2, val + 0.2,
                    f"{val:.1f}%", ha="center", va="bottom", fontsize=9)
    # Annotate delta for each bucket
    for xi, (r_all, r_nooct) in enumerate(zip(abl_df["rmspe_all"],
                                               abl_df["rmspe_no_oct"])):
        if not np.isnan(r_nooct):
            delta = r_nooct - r_all
            ax.annotate(f"{delta:+.1f}pp",
                        xy=(xi + w/2, r_nooct),
                        xytext=(0, 10), textcoords="offset points",
                        ha="center", fontsize=9, fontweight="bold",
                        color="#1B5E20" if delta < 0 else "#B71C1C")
    ax.set_xticks(x)
    ax.set_xticklabels([b.upper() for b in BUCKET_ORDER], fontsize=11)
    ax.set_ylabel("RMSPE (%)", fontsize=11)
    ax.yaxis.set_major_formatter(mticker.PercentFormatter())
    ax.set_title(
        "Impact of Removing October on Each Bucket's RMSPE\n"
        "Key question: does Mid drop to Far/Extended level after removing October?\n"
        f"Mid without Oct = {rmspe_mid_no_oct:.2f}%  |  "
        f"Far = {far_rmspe:.2f}%  |  Extended = {ext_rmspe:.2f}%",
        fontsize=11, fontweight="bold"
    )
    ax.legend(fontsize=10)
    ax.grid(axis="y", alpha=0.25)
    plt.tight_layout()
    fig.savefig(FIGURES_DIR / "13f_mid_ablation.png",
                dpi=150, bbox_inches="tight")
    plt.close()

    print("\nSaved: 13a_mid_bucket_month_heatmap.png")
    print("       13b_mid_bucket_storetype_heatmap.png")
    print("       13c_mid_feature_interactions.png")
    print("       13d_mid_error_distributions.png")
    print("       13e_mid_calendar_composition.png")
    print("       13f_mid_ablation.png")

    return {
        "bucket_month":    bm_df,
        "bucket_storetype": bt_df,
        "features":        feat_df,
        "composition":     comp_df,
        "ablation":        abl_df,
        "hypotheses": {
            "H1_delta_pp":         delta_h1,
            "H1_verdict":          h1_verdict,
            "mid_rmspe_no_oct":    rmspe_mid_no_oct,
            "far_rmspe":           far_rmspe,
            "extended_rmspe":      ext_rmspe,
        },
    }


# ── MAIN ─────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("=" * 60)
    print("Script 14 — October Spike Root-Cause Drill-Down")
    print("=" * 60)

    df = load_data()

    # store_stats = analysis_1_store_contribution(df)
    # analysis_2_dayofweek(df)
    # analysis_3_schoolholiday(df)
    # analysis_4_promo(df)
    # analysis_5_stateholiday(df)
    # analysis_6_storetype_assortment(df)
    # combined_profile(df, store_stats)
    # analysis_storetype_monthly(df)
    # analysis_store_drilldown(df, focus_type="a", top_n=20)
    # analysis_single_store(df, store_id=652)
    # analysis_ablation(df)
    # analysis_october_contradiction(df)
    analysis_mid_horizon_anomaly(df)

    print("\n" + "=" * 60)
    print(f"All analyses complete.  Figures → {FIGURES_DIR.resolve()}")
    print("=" * 60)