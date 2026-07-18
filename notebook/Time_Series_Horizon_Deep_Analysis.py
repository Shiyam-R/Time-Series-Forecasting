"""
13_horizon_deep_analysis.py
─────────────────────────────────────────────────────────────────────────────
Four diagnostic analyses on fair-horizon evaluation results:
  1. RMSPE by Month × Bucket        (heatmap + per-bucket monthly bars)
  2. RMSPE by Horizon Day (1–90)    (line chart with bucket means + 7-day MA)
  3. Holiday vs Non-Holiday          (grouped bars + monthly MID-bucket breakdown)
  4. Error Contribution              (bucket totals + month × bucket heatmap)

Assumes the results CSV saved by script 12 contains (at minimum):
  forecast_date, horizon, horizon_bucket, y_actual (or alias), y_pred (or alias)
  Optional but useful: is_holiday / StateHoliday, store_type / StoreType, origin_date

Update RESULTS_CSV below if your path differs.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from pathlib import Path
import warnings

warnings.filterwarnings("ignore")

# ── CONFIG ───────────────────────────────────────────────────────────────────
RESULTS_CSV  = Path("data/processed/fair_eval_results.csv")   # ← update if needed
FIGURES_DIR  = Path("figures/horizon_analysis")
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

BUCKET_ORDER  = ["near", "mid", "far", "extended"]
BUCKET_RANGES = {"near": (1, 14), "mid": (15, 30), "far": (31, 60), "extended": (61, 90)}
BUCKET_COLORS = {
    "near":     "#2196F3",
    "mid":      "#4CAF50",
    "far":      "#FF9800",
    "extended": "#9C27B0",
}
MONTH_NAMES = ["Jan","Feb","Mar","Apr","May","Jun",
               "Jul","Aug","Sep","Oct","Nov","Dec"]


# ── HELPERS ──────────────────────────────────────────────────────────────────
def rmspe(actual: np.ndarray, predicted: np.ndarray) -> float:
    """RMSPE %, excluding zero-sales days. Returns np.nan if no valid rows."""
    mask = np.asarray(actual) > 0
    if mask.sum() == 0:
        return np.nan
    a, p = np.asarray(actual)[mask], np.asarray(predicted)[mask]
    return float(np.sqrt(np.mean(((a - p) / a) ** 2)) * 100)


def load_and_normalise(path: Path) -> pd.DataFrame:
    """
    Load results CSV from script 12, normalise column names, validate,
    add month helpers, and drop closed-day rows.
    """
    df = pd.read_csv(path, low_memory=False)
    df.columns = df.columns.str.lower().str.strip().str.replace(" ", "_")

    # ── Column aliases ───────────────────────────────────────────────────────
    aliases = {
        # actual sales
        "actual":         "y_actual",
        "sales":          "y_actual",
        "actual_sales":   "y_actual",
        "true_sales":     "y_actual",
        "target":         "y_actual",
        # predicted sales
        "predicted":      "y_pred",
        "pred":           "y_pred",
        "predicted_sales":"y_pred",
        "forecast":       "y_pred",
        "yhat":           "y_pred",
        # horizon
        "days_ahead":     "horizon",
        "day":            "horizon",
        "h":              "horizon",
        # bucket
        "bucket":         "horizon_bucket",
        # dates
        "date":           "forecast_date",
        "target_date":    "forecast_date",
        # holiday
        "stateholiday":   "is_holiday",
        "state_holiday":  "is_holiday",
        "holiday":        "is_holiday",
        "is_state_holiday":"is_holiday",
        # store type
        "storetype":      "store_type",
    }
    df = df.rename(columns={k: v for k, v in aliases.items() if k in df.columns})

    # ── Validate required columns ────────────────────────────────────────────
    required = ["forecast_date", "horizon", "horizon_bucket", "y_actual", "y_pred"]
    missing  = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(
            f"Missing required columns: {missing}\n"
            f"Available columns: {list(df.columns)}\n"
            "Update the 'aliases' dict in load_and_normalise() to map your column names."
        )

    # ── Parse horizon and sanity-check for scaler corruption ────────────────
    df["horizon"] = pd.to_numeric(df["horizon"], errors="coerce").round().astype("Int64")
    h_min, h_max = df["horizon"].min(), df["horizon"].max()
    if h_min < 0 or h_max < 5:
        raise ValueError(
            f"CRITICAL: 'horizon' range [{h_min}, {h_max}] — looks like StandardScaler "
            "was applied to this column. Re-run script 12 with the scaler fix."
        )
    if h_max > 90:
        print(f"⚠️  Max horizon = {h_max} (expected ≤90). Clipping to 1–90.")
        df = df[(df["horizon"] >= 1) & (df["horizon"] <= 90)].copy()

    # ── Parse dates ──────────────────────────────────────────────────────────
    df["forecast_date"] = pd.to_datetime(df["forecast_date"])
    if "origin_date" in df.columns:
        df["origin_date"] = pd.to_datetime(df["origin_date"])

    # ── Normalise holiday column (Rossmann StateHoliday: '0'/'a'/'b'/'c') ───
    if "is_holiday" in df.columns:
        df["is_holiday"] = (
            df["is_holiday"]
            .replace({"0": False, "": False, 0: False,
                      "a": True,  "b": True, "c": True,
                      1: True,    "1": True})
            .fillna(False)
            .astype(bool)
        )
    else:
        print("⚠️  No holiday column found — Analysis 3 will show all as Non-Holiday.")
        df["is_holiday"] = False

    # ── Categorical bucket order ─────────────────────────────────────────────
    df["horizon_bucket"] = pd.Categorical(
        df["horizon_bucket"].str.lower().str.strip(),
        categories=BUCKET_ORDER, ordered=True,
    )

    # ── Helper columns ───────────────────────────────────────────────────────
    df["month"]      = df["forecast_date"].dt.month
    df["month_name"] = df["forecast_date"].dt.strftime("%b")

    # ── Drop closed days ─────────────────────────────────────────────────────
    n_before = len(df)
    df = df[df["y_actual"] > 0].copy()
    dropped  = n_before - len(df)

    print(f"\nLoaded {n_before:,} rows — dropped {dropped:,} closed-day rows")
    print(f"Analysing {len(df):,} open-day rows")
    print(f"Date range : {df['forecast_date'].min().date()} → "
          f"{df['forecast_date'].max().date()}")

    print("\nSample counts per bucket:")
    counts = df.groupby("horizon_bucket", observed=True).size().rename("n")
    for_day = counts.copy()
    for_day.index = for_day.index.map(
        lambda b: f"{b} ({BUCKET_RANGES[b][1]-BUCKET_RANGES[b][0]+1} days)"
    )
    for b, n in counts.items():
        lo, hi = BUCKET_RANGES[b]
        span = hi - lo + 1
        print(f"  {b:8s}: {n:8,}  ({n/span:,.0f} samples/day)")

    # ── Diagnostic: origin × bucket density (flags far undercount issue) ─────
    if "origin_date" in df.columns:
        print("\nDiagnostic — mean samples per unique origin per bucket:")
        origin_density = (
            df.groupby(["origin_date", "horizon_bucket"], observed=True)
              .size()
              .reset_index(name="n")
              .groupby("horizon_bucket", observed=True)["n"]
              .agg(n_origins="count", mean_per_origin="mean",
                   median_per_origin="median")
              .round(1)
        )
        print(origin_density)
        print("  If 'far' shows far fewer n_origins than other buckets → "
              "boundary constraint in script 12 is culling far origins.")

    return df


# ── ANALYSIS 1: RMSPE by Month × Bucket ──────────────────────────────────────
def analysis_1_month_bucket(df: pd.DataFrame) -> pd.DataFrame:
    print("\n" + "═"*60)
    print("ANALYSIS 1 — RMSPE by Month × Bucket")
    print("═"*60)

    MIN_SAMPLES = 20

    records = []
    for bucket in BUCKET_ORDER:
        for m_idx, m_name in enumerate(MONTH_NAMES, start=1):
            sub = df[(df["horizon_bucket"] == bucket) & (df["month"] == m_idx)]
            if len(sub) >= MIN_SAMPLES:
                records.append({
                    "bucket":     bucket,
                    "month":      m_idx,
                    "month_name": m_name,
                    "rmspe":      rmspe(sub["y_actual"].values, sub["y_pred"].values),
                    "n":          len(sub),
                })

    res = pd.DataFrame(records)

    pivot = (
        res.pivot(index="bucket", columns="month_name", values="rmspe")
           .reindex(BUCKET_ORDER)
           [[m for m in MONTH_NAMES if m in res["month_name"].unique()]]
    )

    print("\nRMSPE % by Month × Bucket:\n")
    print(pivot.round(1).to_string())
    print("\nWorst month per bucket:")
    for bucket in BUCKET_ORDER:
        row = res[res["bucket"] == bucket]
        if not row.empty:
            worst = row.loc[row["rmspe"].idxmax()]
            print(f"  {bucket:8s}: {worst['month_name']} = {worst['rmspe']:.1f}%")

    # ── Heatmap ───────────────────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(14, 4))
    vals = pivot.values.astype(float)
    im = ax.imshow(vals, aspect="auto", cmap="RdYlGn_r", vmin=5, vmax=55)
    ax.set_xticks(range(len(pivot.columns)))
    ax.set_xticklabels(pivot.columns, fontsize=10)
    ax.set_yticks(range(len(BUCKET_ORDER)))
    ax.set_yticklabels(BUCKET_ORDER, fontsize=11)
    for i in range(len(BUCKET_ORDER)):
        for j in range(len(pivot.columns)):
            v = vals[i, j]
            if not np.isnan(v):
                color = "white" if (v > 40 or v < 9) else "black"
                ax.text(j, i, f"{v:.0f}%", ha="center", va="center",
                        fontsize=9, fontweight="bold", color=color)
    plt.colorbar(im, ax=ax, label="RMSPE (%)", shrink=0.8)
    ax.set_title("RMSPE by Month × Bucket\n"
                 "(October spike in MID confirms holiday-composition hypothesis)",
                 fontsize=12, fontweight="bold", pad=10)
    plt.tight_layout()
    fig.savefig(FIGURES_DIR / "1a_rmspe_month_bucket_heatmap.png",
                dpi=150, bbox_inches="tight")
    plt.close()

    # ── Per-bucket monthly bar charts ─────────────────────────────────────────
    fig, axes = plt.subplots(2, 2, figsize=(14, 7), sharey=False)
    for ax, bucket in zip(axes.flatten(), BUCKET_ORDER):
        sub = res[res["bucket"] == bucket].sort_values("month")
        if sub.empty:
            ax.set_visible(False)
            continue
        bars = ax.bar(sub["month_name"], sub["rmspe"],
                      color=BUCKET_COLORS[bucket], alpha=0.82)
        mean_val = sub["rmspe"].mean()
        ax.axhline(mean_val, color="red", linestyle="--", linewidth=1.3,
                   label=f"Mean {mean_val:.1f}%")
        ax.set_title(bucket.upper(), fontweight="bold", fontsize=11)
        ax.set_ylabel("RMSPE (%)")
        ax.yaxis.set_major_formatter(mticker.PercentFormatter())
        ax.tick_params(axis="x", rotation=45, labelsize=8)
        ax.legend(fontsize=9)
        ax.grid(axis="y", alpha=0.25)
        # Annotate worst month
        worst = sub.loc[sub["rmspe"].idxmax()]
        worst_x = list(sub["month_name"]).index(worst["month_name"])
        ax.annotate(
            f"↑ {worst['month_name']}\n{worst['rmspe']:.0f}%",
            xy=(worst_x, worst["rmspe"]),
            xytext=(0, 6), textcoords="offset points",
            ha="center", fontsize=8, color="darkred", fontweight="bold",
        )
    plt.suptitle("Monthly RMSPE Profile per Bucket", fontsize=13, fontweight="bold")
    plt.tight_layout()
    fig.savefig(FIGURES_DIR / "1b_monthly_rmspe_per_bucket.png",
                dpi=150, bbox_inches="tight")
    plt.close()

    print("\nSaved: 1a_rmspe_month_bucket_heatmap.png")
    print("       1b_monthly_rmspe_per_bucket.png")
    return res


# ── ANALYSIS 2: RMSPE by Horizon Day (1–90) ──────────────────────────────────
def analysis_2_by_day(df: pd.DataFrame) -> pd.DataFrame:
    print("\n" + "═"*60)
    print("ANALYSIS 2 — RMSPE by Horizon Day (1–90)")
    print("═"*60)

    records = []
    for h in range(1, 91):
        sub = df[df["horizon"] == h]
        if len(sub) >= 5:
            records.append({
                "horizon": h,
                "rmspe":   rmspe(sub["y_actual"].values, sub["y_pred"].values),
                "n":       len(sub),
            })
    res = pd.DataFrame(records)

    print("\nBucket-level stats from day-level view:")
    print(f"  {'Bucket':8s} {'Range':>7}  {'Mean':>6}  {'Max':>6}  {'Max day':>8}")
    print("  " + "-"*46)
    for bucket, (lo, hi) in BUCKET_RANGES.items():
        sub = res[(res["horizon"] >= lo) & (res["horizon"] <= hi)]
        if sub.empty:
            continue
        worst_day = int(sub.loc[sub["rmspe"].idxmax(), "horizon"])
        print(f"  {bucket:8s} {lo:3d}–{hi:3d}   "
              f"{sub['rmspe'].mean():5.1f}%  "
              f"{sub['rmspe'].max():5.1f}%  "
              f"day {worst_day:3d}")

    print("\n  Key: if mid's peak days (15–30) are systematically higher than far's (31–60)")
    print("       it confirms the October-calendar-alignment hypothesis.")

    # ── Plot ──────────────────────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(14, 5))

    # Background bucket shading
    for bucket, (lo, hi) in BUCKET_RANGES.items():
        ax.axvspan(lo, hi, alpha=0.07, color=BUCKET_COLORS[bucket])

    # Bucket boundary lines
    for x_val in [14.5, 30.5, 60.5]:
        ax.axvline(x_val, color="gray", linestyle="--", linewidth=0.9, alpha=0.6)

    # Raw per-day RMSPE
    ax.plot(res["horizon"], res["rmspe"],
            color="#888888", linewidth=0.9, alpha=0.45, label="Daily RMSPE")

    # 7-day rolling average
    ma = res["rmspe"].rolling(7, center=True, min_periods=3).mean()
    ax.plot(res["horizon"], ma,
            color="#D32F2F", linewidth=2.5, label="7-day MA")

    # Dotted bucket-mean lines + labels
    for bucket, (lo, hi) in BUCKET_RANGES.items():
        sub = res[(res["horizon"] >= lo) & (res["horizon"] <= hi)]
        if sub.empty:
            continue
        mean_r = sub["rmspe"].mean()
        ax.hlines(mean_r, lo, hi,
                  colors=BUCKET_COLORS[bucket], linewidth=2.0,
                  linestyle=":", alpha=0.9)
        ax.text((lo + hi) / 2, mean_r + 0.9,
                f"{bucket}\n{mean_r:.1f}%",
                ha="center", fontsize=8,
                color=BUCKET_COLORS[bucket], fontweight="bold")

    ax.set_xlabel("Forecast Horizon (days ahead)", fontsize=11)
    ax.set_ylabel("RMSPE (%)", fontsize=11)
    ax.yaxis.set_major_formatter(mticker.PercentFormatter())
    ax.set_xlim(1, 90)
    ax.set_title("RMSPE by Forecast Horizon Day  "
                 "(dotted = bucket mean, red = 7-day MA)",
                 fontsize=12, fontweight="bold")
    ax.legend(fontsize=10, loc="upper right")
    ax.grid(axis="y", alpha=0.22)
    plt.tight_layout()
    fig.savefig(FIGURES_DIR / "2_rmspe_by_horizon_day.png",
                dpi=150, bbox_inches="tight")
    plt.close()
    print("\nSaved: 2_rmspe_by_horizon_day.png")
    return res


# ── ANALYSIS 3: Holiday vs Non-Holiday ───────────────────────────────────────
def analysis_3_holiday(df: pd.DataFrame) -> pd.DataFrame:
    print("\n" + "═"*60)
    print("ANALYSIS 3 — Holiday vs Non-Holiday RMSPE")
    print("═"*60)

    n_hol = df["is_holiday"].sum()
    print(f"\nHoliday rows: {n_hol:,} ({n_hol/len(df)*100:.1f}% of open days)")

    records = []
    for bucket in BUCKET_ORDER:
        for is_hol in [False, True]:
            sub = df[(df["horizon_bucket"] == bucket) & (df["is_holiday"] == is_hol)]
            if len(sub) >= 10:
                records.append({
                    "bucket":     bucket,
                    "is_holiday": is_hol,
                    "label":      "Holiday" if is_hol else "Non-Holiday",
                    "rmspe":      rmspe(sub["y_actual"].values, sub["y_pred"].values),
                    "n":          len(sub),
                })
    res = pd.DataFrame(records)

    print(f"\n{'Bucket':<10} {'Non-Holiday':>12} {'Holiday':>10} "
          f"{'Premium':>10} {'Hol%':>8}")
    print("  " + "─"*50)
    for bucket in BUCKET_ORDER:
        nh = res[(res["bucket"] == bucket) & (~res["is_holiday"])]
        h  = res[(res["bucket"] == bucket) & (res["is_holiday"])]
        if nh.empty or h.empty:
            print(f"  {bucket:<10} — insufficient holiday data")
            continue
        total = nh["n"].iloc[0] + h["n"].iloc[0]
        hol_pct = h["n"].iloc[0] / total * 100
        premium = h["rmspe"].iloc[0] - nh["rmspe"].iloc[0]
        print(f"  {bucket:<10} {nh['rmspe'].iloc[0]:>11.1f}%  "
              f"{h['rmspe'].iloc[0]:>9.1f}%  "
              f"{premium:>+9.1f}pp  {hol_pct:>6.1f}%")

    print("\n  'Premium' = additional RMSPE on holiday days.")
    print("  High premium in MID supports the October-holiday-window hypothesis.")

    # ── Grouped bar: all buckets ──────────────────────────────────────────────
    nh_vals = [res[(res["bucket"] == b) & (~res["is_holiday"])]["rmspe"].values
               for b in BUCKET_ORDER]
    h_vals  = [res[(res["bucket"] == b) & (res["is_holiday"])]["rmspe"].values
               for b in BUCKET_ORDER]
    nh_vals = [v[0] if len(v) else np.nan for v in nh_vals]
    h_vals  = [v[0] if len(v) else np.nan for v in h_vals]

    x, w = np.arange(len(BUCKET_ORDER)), 0.38
    fig, ax = plt.subplots(figsize=(10, 5))
    b1 = ax.bar(x - w/2, nh_vals, w, label="Non-Holiday", color="#42A5F5", alpha=0.9)
    b2 = ax.bar(x + w/2, h_vals,  w, label="Holiday",     color="#EF5350", alpha=0.9)
    for bar in list(b1) + list(b2):
        v = bar.get_height()
        if not np.isnan(v):
            ax.text(bar.get_x() + bar.get_width()/2, v + 0.4,
                    f"{v:.1f}%", ha="center", va="bottom", fontsize=9)
    ax.set_xticks(x)
    ax.set_xticklabels([b.upper() for b in BUCKET_ORDER], fontsize=11)
    ax.set_ylabel("RMSPE (%)", fontsize=11)
    ax.yaxis.set_major_formatter(mticker.PercentFormatter())
    ax.set_title("Holiday vs Non-Holiday RMSPE by Bucket",
                 fontsize=13, fontweight="bold")
    ax.legend(fontsize=10)
    ax.grid(axis="y", alpha=0.25)
    plt.tight_layout()
    fig.savefig(FIGURES_DIR / "3a_holiday_vs_nonholiday.png",
                dpi=150, bbox_inches="tight")
    plt.close()

    # ── Monthly breakdown for MID (highest anomaly bucket) ───────────────────
    fig, ax = plt.subplots(figsize=(12, 4))
    for is_hol, label, color in [(False, "Non-Holiday", "#42A5F5"),
                                   (True,  "Holiday",     "#EF5350")]:
        pts = []
        for m_idx, m_name in enumerate(MONTH_NAMES, start=1):
            sub = df[
                (df["horizon_bucket"] == "mid") &
                (df["month"] == m_idx) &
                (df["is_holiday"] == is_hol)
            ]
            if len(sub) >= 10:
                pts.append({"m_idx": m_idx, "m": m_name,
                             "rmspe": rmspe(sub["y_actual"].values,
                                            sub["y_pred"].values)})
        if pts:
            tmp = pd.DataFrame(pts).sort_values("m_idx")
            ax.plot(tmp["m"], tmp["rmspe"], marker="o",
                    linewidth=2, color=color, label=label, markersize=6)

    # Highlight October
    ax.axvspan(8.5, 9.5, alpha=0.12, color="red")   # index of Oct in MONTH_NAMES plot
    ax.set_ylabel("RMSPE (%)", fontsize=11)
    ax.yaxis.set_major_formatter(mticker.PercentFormatter())
    ax.set_title("Holiday vs Non-Holiday RMSPE by Month — MID Bucket\n"
                 "(red band = October: German Unity Day + Herbstferien)",
                 fontsize=11, fontweight="bold")
    ax.legend(fontsize=10)
    ax.grid(alpha=0.25)
    plt.tight_layout()
    fig.savefig(FIGURES_DIR / "3b_holiday_monthly_mid.png",
                dpi=150, bbox_inches="tight")
    plt.close()

    print("\nSaved: 3a_holiday_vs_nonholiday.png")
    print("       3b_holiday_monthly_mid.png")
    return res


# ── ANALYSIS 4: Error Contribution ───────────────────────────────────────────
def analysis_4_contribution(df: pd.DataFrame) -> pd.DataFrame:
    print("\n" + "═"*60)
    print("ANALYSIS 4 — Error Contribution Breakdown")
    print("═"*60)

    df = df.copy()
    df["spe"] = ((df["y_actual"] - df["y_pred"]) / df["y_actual"]) ** 2
    total_spe = df["spe"].sum()

    records = []
    for bucket in BUCKET_ORDER:
        for m_idx, m_name in enumerate(MONTH_NAMES, start=1):
            sub = df[(df["horizon_bucket"] == bucket) & (df["month"] == m_idx)]
            if len(sub) > 0:
                records.append({
                    "bucket":           bucket,
                    "month":            m_idx,
                    "month_name":       m_name,
                    "contribution_pct": sub["spe"].sum() / total_spe * 100,
                    "rmspe":            rmspe(sub["y_actual"].values, sub["y_pred"].values),
                    "n":                len(sub),
                })
    res = pd.DataFrame(records)

    # Bucket totals
    print("\nTotal error contribution by bucket:")
    for bucket in BUCKET_ORDER:
        pct = res[res["bucket"] == bucket]["contribution_pct"].sum()
        print(f"  {bucket:8s}: {pct:5.1f}% of total squared error")

    print("\nTop 8 highest-contributing (Month × Bucket) cells:")
    top = (res.nlargest(8, "contribution_pct")
              [["bucket", "month_name", "contribution_pct", "rmspe", "n"]]
              .rename(columns={"contribution_pct": "contrib_%", "rmspe": "rmspe_%"}))
    print(top.to_string(index=False))

    # ── Figure: bucket bar + month×bucket heatmap ─────────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(16, 5))

    # Left: bucket-level contribution bar
    ax = axes[0]
    bucket_totals = (res.groupby("bucket")["contribution_pct"]
                        .sum().reindex(BUCKET_ORDER))
    bars = ax.bar(BUCKET_ORDER, bucket_totals.values,
                  color=[BUCKET_COLORS[b] for b in BUCKET_ORDER], alpha=0.85)
    for bar, val in zip(bars, bucket_totals.values):
        ax.text(bar.get_x() + bar.get_width()/2, val + 0.4,
                f"{val:.1f}%", ha="center", va="bottom",
                fontsize=10, fontweight="bold")
    ax.set_ylabel("% of Total Squared Error", fontsize=11)
    ax.yaxis.set_major_formatter(mticker.PercentFormatter())
    ax.set_title("Error Contribution by Bucket", fontweight="bold")
    ax.grid(axis="y", alpha=0.25)

    # Right: month × bucket heatmap
    ax = axes[1]
    pivot = (
        res.pivot(index="bucket", columns="month_name", values="contribution_pct")
           .reindex(BUCKET_ORDER)
           [[m for m in MONTH_NAMES if m in res["month_name"].unique()]]
    )
    vals = pivot.fillna(0).values.astype(float)
    im = ax.imshow(vals, aspect="auto", cmap="YlOrRd")
    ax.set_xticks(range(len(pivot.columns)))
    ax.set_xticklabels(pivot.columns, fontsize=9, rotation=45)
    ax.set_yticks(range(len(BUCKET_ORDER)))
    ax.set_yticklabels(BUCKET_ORDER, fontsize=9)
    for i in range(vals.shape[0]):
        for j in range(vals.shape[1]):
            v = vals[i, j]
            if v > 0.2:
                ax.text(j, i, f"{v:.1f}",
                        ha="center", va="center", fontsize=8,
                        color="white" if v > 6 else "black")
    plt.colorbar(im, ax=ax, label="% of Total Squared Error")
    ax.set_title("Error Contribution: Month × Bucket\n"
                 "(high mid×Oct = root cause of ordering anomaly)",
                 fontweight="bold", fontsize=10)

    plt.suptitle("Error Contribution Analysis", fontsize=13, fontweight="bold")
    plt.tight_layout()
    fig.savefig(FIGURES_DIR / "4a_error_contribution.png",
                dpi=150, bbox_inches="tight")
    plt.close()

    # ── Optional: StoreType breakdown ─────────────────────────────────────────
    if "store_type" in df.columns:
        print("\n4b. RMSPE by StoreType × Bucket:")
        store_types = sorted(df["store_type"].dropna().unique())
        rows_st = []
        for bucket in BUCKET_ORDER:
            for st in store_types:
                sub = df[(df["horizon_bucket"] == bucket) & (df["store_type"] == st)]
                if len(sub) >= 20:
                    rows_st.append({
                        "bucket":     bucket,
                        "store_type": st,
                        "rmspe":      rmspe(sub["y_actual"].values, sub["y_pred"].values),
                        "n":          len(sub),
                    })
        st_df = pd.DataFrame(rows_st)
        pivot_st = (st_df.pivot(index="store_type", columns="bucket", values="rmspe")
                        .reindex(columns=BUCKET_ORDER))
        print(pivot_st.round(1).to_string())

        colors_st = ["#FF6B6B", "#4ECDC4", "#45B7D1", "#96CEB4"]
        x = np.arange(len(BUCKET_ORDER))
        w = 0.8 / len(store_types)
        fig, ax = plt.subplots(figsize=(10, 5))
        for i, st in enumerate(store_types):
            sub_st = st_df[st_df["store_type"] == st]
            vals_st = [sub_st[sub_st["bucket"] == b]["rmspe"].values for b in BUCKET_ORDER]
            vals_st = [v[0] if len(v) else np.nan for v in vals_st]
            ax.bar(x + i*w - (len(store_types)-1)*w/2, vals_st, w,
                   label=f"Type {st}",
                   color=colors_st[i % len(colors_st)], alpha=0.85)
        ax.set_xticks(x)
        ax.set_xticklabels([b.upper() for b in BUCKET_ORDER])
        ax.set_ylabel("RMSPE (%)")
        ax.yaxis.set_major_formatter(mticker.PercentFormatter())
        ax.set_title("RMSPE by StoreType × Bucket", fontweight="bold")
        ax.legend(title="StoreType")
        ax.grid(axis="y", alpha=0.25)
        plt.tight_layout()
        fig.savefig(FIGURES_DIR / "4b_storetype_bucket.png",
                    dpi=150, bbox_inches="tight")
        plt.close()
        print("Saved: 4b_storetype_bucket.png")
    else:
        print("ℹ️  store_type not in results — skipping 4b StoreType breakdown.")

    print("\nSaved: 4a_error_contribution.png")
    return res


# ── MAIN ─────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("=" * 60)
    print("Script 13 — Fair-Horizon Diagnostic Analysis")
    print("=" * 60)

    df = load_and_normalise(RESULTS_CSV)

    r1 = analysis_1_month_bucket(df)
    r2 = analysis_2_by_day(df)
    r3 = analysis_3_holiday(df)
    r4 = analysis_4_contribution(df)

    print("\n" + "=" * 60)
    print(f"All analyses complete.  Figures → {FIGURES_DIR.resolve()}")
    print("=" * 60)

    # ── Quick diagnostic summary ───────────────────────────────────────────
    print("\n── What to look for in each output ──")
    print("1a heatmap    : October column should be the hottest cell in MID row")
    print("1b bars       : MID's worst month should be Oct or adjacent; FAR/EXT should peak Dec")
    print("2 horizon-day : RMSPE should NOT drop cleanly from day 30→31; if it does, model")
    print("                skill is genuine; if noisy it's data composition")
    print("3a holiday bar: MID holiday premium should be largest; quantifies Oct-holiday drag")
    print("3b mid monthly: Holiday line should spike Oct, non-holiday line should be flatter")
    print("4a heatmap    : High mid×Oct cell = October is the root cause of ordering anomaly")
    print("4b store types: Type b (extra assortment) and a typically hardest to forecast")
