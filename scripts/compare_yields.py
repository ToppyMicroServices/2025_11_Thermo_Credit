"""Compare US 10Y Treasury (DGS10) vs Japan Long-term Yield (IRLTLT01JPM156N).

Usage:
  export FRED_API_KEY=XXXX
  python scripts/compare_yields.py --start 1995-01-01 --png

Outputs:
  site/yield_compare.html (interactive Plotly)
  site/yield_compare.png  (optional static fallback if --png)

The script normalizes both yields to their first valid observation for visual
shape comparison (index=100). It also computes a simple rolling differential
and correlation diagnostics.
"""
import argparse
import os
import sys

import pandas as pd
import plotly.express as px

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

FRED_KEY = os.getenv("FRED_API_KEY", "")

def fred_series(series_id: str, start: str = "1990-01-01") -> pd.DataFrame:
    if not FRED_KEY:
        raise RuntimeError("FRED_API_KEY not set; cannot fetch online.")
    url = (
        "https://api.stlouisfed.org/fred/series/observations"
        f"?series_id={series_id}&api_key={FRED_KEY}&file_type=json&observation_start={start}"
    )
    import requests
    r = requests.get(url, timeout=30); r.raise_for_status()
    obs = r.json()["observations"]
    df = pd.DataFrame(obs)[["date", "value"]]
    df["value"] = pd.to_numeric(df["value"], errors="coerce")
    df = df.dropna().rename(columns={"value": series_id})
    df["date"] = pd.to_datetime(df["date"])
    return df.sort_values("date")

def normalize_first(df: pd.DataFrame, cols) -> pd.DataFrame:
    out = df.copy()
    for c in cols:
        series = out[c].dropna()
        if series.empty:
            continue
        base = series.iloc[0]
        if base == 0:
            continue
        out[c + "_norm"] = 100.0 * out[c] / base
    return out

def build_plot(df: pd.DataFrame) -> px.line:
    plot_cols = [c for c in df.columns if c.endswith("_norm")] or [c for c in df.columns if c in ("DGS10","IRLTLT01JPM156N")]
    return px.line(
        df, x="date", y=plot_cols,
        title="US vs Japan Long-term Yield (Normalized to 100 at First Observation)",
        labels={"value": "Index (=100 at first obs)", "date": "Date", "variable": "Yield"},
    )

def rolling_metrics(df: pd.DataFrame) -> pd.DataFrame:
    """Compute monthly-style rolling metrics. If the data are monthly, use window=12.
    If data are daily, this remains a rough approximation but still informative.
    """
    roll = df.set_index("date")[["DGS10","IRLTLT01JPM156N"]].astype(float)
    # Try to infer frequency: if median delta > 3 days, assume monthly
    idx = roll.index.to_series().sort_values()
    if len(idx) >= 3:
        dt_med = (idx.diff().dt.days.median()) or 0
    else:
        dt_med = 30
    window = 12 if dt_med >= 15 else 252  # monthly vs daily-ish fallback

    metrics = pd.DataFrame({
        "date": roll.index,
        "spread_US_minus_JP": roll["DGS10"] - roll["IRLTLT01JPM156N"],
    })
    metrics[f"roll_corr_{window}"] = roll["DGS10"].rolling(window).corr(roll["IRLTLT01JPM156N"])  # monthly=12 or daily=252
    return metrics

def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Compare US vs Japan long-term yields.")
    ap.add_argument("--start", default="1990-01-01", help="Observation start date (YYYY-MM-DD)")
    ap.add_argument("--png", action="store_true", help="Write PNG fallback")
    return ap.parse_args()

def main() -> None:
    args = parse_args()
    os.makedirs("site", exist_ok=True)
    try:
        us = fred_series("DGS10", args.start)
        jp = fred_series("IRLTLT01JPM156N", args.start)
    except Exception as e:
        print("Fetch failed:", e)
        return
    df = pd.merge(us, jp, on="date", how="inner")
    df = normalize_first(df, ["DGS10","IRLTLT01JPM156N"])  # adds *_norm columns
    fig = build_plot(df)

    # Write interactive HTML snippet
    html = fig.to_html(full_html=True, include_plotlyjs="cdn")
    with open("site/yield_compare.html", "w", encoding="utf-8") as f:
        f.write(html)
    print("Wrote site/yield_compare.html")

    if args.png:
        try:
            fig.write_image("site/yield_compare.png", scale=2, width=1280, height=720)
            print("Wrote site/yield_compare.png")
        except Exception as e:
            print("PNG export failed:", e)

    # Simple metrics CSV
    metrics = rolling_metrics(df)
    metrics.to_csv("site/yield_compare_metrics.csv", index=False)
    print("Wrote site/yield_compare_metrics.csv")

if __name__ == "__main__":
    main()
