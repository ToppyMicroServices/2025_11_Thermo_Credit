"""Create a paper-ready summary figure from the regional dashboard."""
from __future__ import annotations

import os
import tempfile
from pathlib import Path
from typing import Iterable


_CACHE_ROOT = Path(tempfile.gettempdir()) / "thermo_credit_takeaways_cache"
_CACHE_ROOT.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(_CACHE_ROOT / "mplconfig"))
os.environ.setdefault("XDG_CACHE_HOME", str(_CACHE_ROOT / "xdg-cache"))
os.environ.setdefault("SOURCE_DATE_EPOCH", "1704067200")

import matplotlib

matplotlib.use("Agg")

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from lib.public_api import REGION_SPECS, load_region_frame
from lib.report_helpers import filter_dashboard_events, load_dashboard_events


NAVY = "#02213b"
CYAN = "#2a9dca"
MINT = "#2a9d78"
AMBER = "#d59245"
GRID = "#d8e3ea"
EVENT_COLORS = {
    "bubble": "#e7b966",
    "crisis": "#d9786f",
    "pandemic": "#8296b6",
    "policy": "#75a9c2",
}
EVENT_LABELS = {
    "jp_ycc": "YCC",
    "pandemic": "COVID-19",
    "tightening": "Rate shock",
    "us_regional_banks": "US banks",
    "eu_qe": "ECB QE",
}


def _style() -> None:
    sns.set_theme(
        style="whitegrid",
        context="paper",
        font="DejaVu Sans",
        rc={
            "axes.facecolor": "#fbfdfe",
            "axes.edgecolor": "#a8bdc9",
            "axes.labelcolor": NAVY,
            "axes.titlecolor": NAVY,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "figure.facecolor": "white",
            "grid.color": GRID,
            "grid.linewidth": 0.7,
            "legend.frameon": False,
            "pdf.fonttype": 42,
            "svg.fonttype": "none",
            "svg.hashsalt": "thermo-credit-dashboard-takeaways",
        },
    )


def _event_bands(
    axes: Iterable[plt.Axes],
    events: list[dict],
    *,
    region: str,
    start: pd.Timestamp,
    end: pd.Timestamp,
) -> None:
    visible = filter_dashboard_events(
        events,
        region_key=region,
        start_date=start,
        end_date=end,
    )
    axes = list(axes)
    for index, event in enumerate(visible):
        band_start = pd.to_datetime(event.get("visible_start"), errors="coerce")
        band_end = pd.to_datetime(event.get("visible_end"), errors="coerce")
        if pd.isna(band_start) or pd.isna(band_end):
            continue
        color = EVENT_COLORS.get(str(event.get("category") or ""), "#9aadb8")
        for axis in axes:
            axis.axvspan(band_start, band_end, color=color, alpha=0.12, linewidth=0)
        short_label = EVENT_LABELS.get(str(event.get("key") or ""))
        if short_label:
            axes[0].text(
                band_start,
                0.98 - 0.08 * (index % 2),
                short_label,
                transform=axes[0].get_xaxis_transform(),
                ha="left",
                va="top",
                fontsize=7,
                color=NAVY,
            )


def build_dashboard_takeaways(
    *,
    site_dir: Path,
    output_dir: Path,
    events_path: Path,
    start_date: str = "2015-01-01",
) -> list[Path]:
    """Write PNG, PDF, SVG, and a compact LaTeX include snippet."""
    _style()
    output_dir.mkdir(parents=True, exist_ok=True)
    start = pd.Timestamp(start_date)
    events = load_dashboard_events(str(events_path))
    figure, axes = plt.subplots(2, 3, figsize=(15, 7.6), sharex="col")

    for column, region in enumerate(("jp", "eu", "us")):
        frame, _ = load_region_frame(site_dir, region)
        frame = frame[frame["date"] >= start].copy()
        if frame.empty:
            raise ValueError(f"No {region} observations on or after {start.date()}")
        top = axes[0, column]
        bottom = axes[1, column]
        spec = REGION_SPECS[region]
        q_t = pd.to_numeric(frame.get("q_t"), errors="coerce")
        x_c = (
            pd.to_numeric(frame.get("X_C"), errors="coerce")
            .rolling(4, min_periods=1)
            .median()
            .div(1_000_000.0)
        )
        q_valid = q_t.dropna()
        x_valid = x_c.dropna()
        if q_valid.empty or x_valid.empty:
            raise ValueError(f"{region} lacks usable q_t or X_C observations")
        q_last_index = q_valid.index[-1]
        x_last_index = x_valid.index[-1]

        top.plot(frame["date"], q_t, color=MINT, linewidth=2.1)
        top.scatter(frame.loc[q_last_index, "date"], q_valid.iloc[-1], color=MINT, s=28, zorder=4)
        top.set_ylim(0, 1)
        top.set_ylabel("Share")
        top.set_title(spec["label"], loc="left", fontsize=13, fontweight="bold")
        evidence_label = "JP measurement bridge" if region == "jp" else "Regional proxy panel"
        top.text(
            0.01,
            0.08,
            evidence_label,
            transform=top.transAxes,
            fontsize=8,
            color=NAVY,
            bbox={"boxstyle": "round,pad=0.28", "facecolor": "white", "edgecolor": GRID, "alpha": 0.92},
        )
        top.text(
            0.99,
            0.08,
            f"last {q_valid.iloc[-1]:.3f} ({frame.loc[q_last_index, 'date']:%Y-%m})",
            transform=top.transAxes,
            ha="right",
            fontsize=8,
            color=MINT,
            fontweight="bold",
        )

        bottom.plot(frame["date"], x_c, color=CYAN, linewidth=2.1)
        bottom.scatter(frame.loc[x_last_index, "date"], x_valid.iloc[-1], color=CYAN, s=28, zorder=4)
        bottom.axhline(0, color=NAVY, alpha=0.3, linewidth=0.9)
        bottom.set_ylabel(r"$X_C$ / $10^6$ (4Q median)")
        bottom.text(
            0.99,
            0.08,
            f"last {x_valid.iloc[-1]:.3f} ({frame.loc[x_last_index, 'date']:%Y-%m})",
            transform=bottom.transAxes,
            ha="right",
            fontsize=8,
            color=CYAN,
            fontweight="bold",
        )
        bottom.xaxis.set_major_locator(mdates.YearLocator(2))
        bottom.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
        bottom.tick_params(axis="x", labelrotation=0)
        _event_bands((top, bottom), events, region=region, start=start, end=frame["date"].max())

    figure.suptitle(
        "Regional credit composition and the experimental $X_C$ diagnostic",
        x=0.055,
        y=0.985,
        ha="left",
        fontsize=18,
        fontweight="bold",
        color=NAVY,
    )
    figure.text(
        0.055,
        0.94,
        "Raw allocation shares above; only $X_C$ uses a four-quarter rolling median below.",
        ha="left",
        fontsize=10,
        color="#436579",
    )
    figure.text(
        0.055,
        0.015,
        "Evidence boundary: JP is a borrower-composition measurement; EU/US are proxies. "
        "Event bands mark timing. $X_C$ scales are region-specific and are not forecasts or policy thresholds.",
        ha="left",
        fontsize=8.5,
        color="#436579",
    )
    figure.subplots_adjust(left=0.06, right=0.985, top=0.88, bottom=0.10, hspace=0.22, wspace=0.20)

    base = output_dir / "dashboard_takeaways"
    outputs = [base.with_suffix(".png"), base.with_suffix(".pdf"), base.with_suffix(".svg")]
    figure.savefig(outputs[0], dpi=220, bbox_inches="tight", facecolor="white")
    figure.savefig(outputs[1], bbox_inches="tight", facecolor="white", metadata={"Creator": "Thermo Credit"})
    figure.savefig(outputs[2], bbox_inches="tight", facecolor="white", metadata={"Creator": "Thermo Credit"})
    plt.close(figure)
    svg_text = outputs[2].read_text(encoding="utf-8")
    outputs[2].write_text(
        "\n".join(line.rstrip() for line in svg_text.splitlines()) + "\n",
        encoding="utf-8",
    )

    snippet = output_dir / "dashboard_takeaways.tex"
    snippet.write_text(
        "\\begin{figure}[htbp]\n"
        "  \\centering\n"
        "  \\includegraphics[width=\\textwidth]{generated/dashboard_takeaways.pdf}\n"
        "  \\caption{Regional allocation shares and region-specific $X_C$ diagnostics.}\n"
        "  \\label{fig:dashboard_takeaways}\n"
        "\\end{figure}\n",
        encoding="utf-8",
    )
    outputs.append(snippet)
    return outputs


__all__ = ["build_dashboard_takeaways"]
