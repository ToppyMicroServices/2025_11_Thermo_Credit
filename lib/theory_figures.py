from __future__ import annotations

import os
import io
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence

_PLOT_CACHE_ROOT = Path(tempfile.gettempdir()) / "thermo_credit_plot_cache"
_PLOT_CACHE_ROOT.mkdir(parents=True, exist_ok=True)
_MPLCONFIGDIR = _PLOT_CACHE_ROOT / "mplconfig"
_XDG_CACHE_HOME = _PLOT_CACHE_ROOT / "xdg-cache"
_MPLCONFIGDIR.mkdir(parents=True, exist_ok=True)
_XDG_CACHE_HOME.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(_MPLCONFIGDIR))
os.environ.setdefault("XDG_CACHE_HOME", str(_XDG_CACHE_HOME))

import matplotlib

matplotlib.use("Agg")

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from lib.report_helpers import filter_dashboard_events, load_dashboard_events

REGION_SPECS: Sequence[tuple[str, str, Sequence[str]]] = (
    ("jp", "Japan (JP)", ("indicators_jp.csv", "indicators.csv")),
    ("eu", "Euro Area (EU)", ("indicators_eu.csv",)),
    ("us", "United States (US)", ("indicators_us.csv",)),
)

METRIC_LABELS = {
    "S_M": "S_M",
    "T_L": "T_L",
    "X_C": "X_C",
    "loop_area": "Loop area",
}

EVENT_SHORT_LABELS = {
    "dotcom": "IT Bubble",
    "lehman": "Lehman",
    "jp_bank_cleanup": "JP Cleanup",
    "us_housing_boom": "US Housing",
    "euro_debt": "Euro Debt",
    "jp_quake": "3/11",
    "eu_omt": "OMT",
    "jp_qqe": "QQE",
    "us_qe1": "QE1",
    "eu_qe": "ECB QE",
    "jp_ycc": "YCC",
    "pandemic": "COVID",
    "tightening": "Rate Shock",
    "us_regional_banks": "US Banks",
}


@dataclass
class RegionFrame:
    key: str
    label: str
    frame: pd.DataFrame


def configure_theory_plot_style() -> None:
    """Set a paper-friendly plotting style with modern defaults."""
    sns.set_theme(
        style="whitegrid",
        context="paper",
        palette="colorblind",
        font="DejaVu Sans",
        rc={
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.titleweight": "semibold",
            "axes.labelsize": 10,
            "axes.titlesize": 11,
            "legend.frameon": False,
            "figure.facecolor": "white",
            "axes.facecolor": "#fbfcfe",
            "grid.color": "#d9e2ef",
            "grid.linewidth": 0.8,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "svg.fonttype": "none",
        },
    )


def _coerce_indicator_frame(frame: pd.DataFrame) -> Optional[pd.DataFrame]:
    if frame.empty or "date" not in frame.columns:
        return None
    frame = frame.copy().assign(date=pd.to_datetime(frame["date"], errors="coerce"))
    frame = frame.dropna(subset=["date"]).sort_values("date").reset_index(drop=True)
    return frame if not frame.empty else None


def _load_indicator_csv(path: Path) -> Optional[pd.DataFrame]:
    if not path.exists():
        return None
    try:
        frame = pd.read_csv(path)
    except Exception:
        return None
    return _coerce_indicator_frame(frame)


def _load_indicator_csv_from_ref(repo_root: Path, relative_path: str, source_ref: str) -> Optional[pd.DataFrame]:
    try:
        raw = subprocess.check_output(
            ["git", "-C", str(repo_root), "show", f"{source_ref}:{relative_path}"],
            text=True,
            stderr=subprocess.DEVNULL,
        )
    except Exception:
        return None
    try:
        frame = pd.read_csv(io.StringIO(raw))
    except Exception:
        return None
    return _coerce_indicator_frame(frame)


def load_region_frames(site_dir: Path, source_ref: str | None = None) -> List[RegionFrame]:
    """Load the best available indicator frame for each region."""
    frames: List[RegionFrame] = []
    repo_root = site_dir.parent
    for key, label, candidates in REGION_SPECS:
        frame: Optional[pd.DataFrame] = None
        for candidate in candidates:
            local_frame = _load_indicator_csv(site_dir / candidate)
            ref_frame = None
            if source_ref:
                ref_frame = _load_indicator_csv_from_ref(repo_root, f"{site_dir.name}/{candidate}", source_ref)
            if local_frame is None:
                maybe = ref_frame
            elif ref_frame is None:
                maybe = local_frame
            else:
                local_end = pd.to_datetime(local_frame["date"].max(), errors="coerce")
                ref_end = pd.to_datetime(ref_frame["date"].max(), errors="coerce")
                maybe = ref_frame if pd.notna(ref_end) and (pd.isna(local_end) or ref_end > local_end) else local_frame
            if maybe is not None and not maybe.empty:
                frame = maybe
                break
        if frame is not None:
            frames.append(RegionFrame(key=key, label=label, frame=frame))
    return frames


def _robust_score_series(series: pd.Series) -> pd.Series:
    """Return a winsorized median/MAD score with smooth compression."""
    numeric = pd.to_numeric(series, errors="coerce")
    transformed = pd.Series(np.arcsinh(numeric), index=series.index, dtype=float)
    valid = transformed.dropna()
    if valid.empty:
        return pd.Series(np.nan, index=series.index, dtype=float)

    lower = float(valid.quantile(0.05))
    upper = float(valid.quantile(0.95))
    winsorized = valid.clip(lower=lower, upper=upper)
    center = float(winsorized.median())
    mad = float((winsorized - center).abs().median())

    if np.isfinite(mad) and mad > 0:
        scale = 1.4826 * mad
    else:
        q75 = float(winsorized.quantile(0.75))
        q25 = float(winsorized.quantile(0.25))
        iqr = q75 - q25
        if np.isfinite(iqr) and iqr > 0:
            scale = iqr / 1.349
        else:
            std = float(winsorized.std(ddof=0))
            if not np.isfinite(std) or std <= 0:
                return pd.Series(np.nan, index=series.index, dtype=float)
            scale = std

    bounded = transformed.clip(lower=lower, upper=upper)
    scored = (bounded - center) / scale
    return np.arcsinh(scored)


def _prepare_metric_series(metric: str, series: pd.Series) -> pd.Series:
    numeric = pd.to_numeric(series, errors="coerce")
    if metric == "X_C":
        smoothed = numeric.rolling(window=4, min_periods=1).median()
        return smoothed.rolling(window=2, min_periods=1).mean()
    return numeric


def build_metric_long_frame(
    region_frames: Iterable[RegionFrame],
    metrics: Sequence[str],
    *,
    start_date: pd.Timestamp,
) -> pd.DataFrame:
    """Convert region panels into a long-form robust-score frame."""
    parts: List[pd.DataFrame] = []
    for region in region_frames:
        local = region.frame.copy()
        local = local[local["date"] >= start_date].copy()
        if local.empty:
            continue
        for metric in metrics:
            if metric not in local.columns:
                continue
            prepared = _prepare_metric_series(metric, local[metric])
            values = _robust_score_series(prepared)
            part = pd.DataFrame(
                {
                    "date": local["date"],
                    "value": values,
                    "metric": metric,
                    "metric_label": METRIC_LABELS.get(metric, metric),
                    "region_key": region.key,
                    "region_label": region.label,
                }
            )
            part = part.dropna(subset=["value"])
            if not part.empty:
                parts.append(part)
    if not parts:
        return pd.DataFrame(columns=["date", "value", "metric", "metric_label", "region_key", "region_label"])
    return pd.concat(parts, ignore_index=True)


def _event_color(event: Dict[str, Any]) -> str:
    category = str(event.get("category") or "").strip().lower()
    return {
        "bubble": "#f59e0b",
        "crisis": "#ef4444",
        "pandemic": "#0ea5e9",
        "policy": "#8b5cf6",
    }.get(category, "#94a3b8")


def _annotate_events(ax: Any, events: Sequence[Dict[str, Any]], y_min: float, y_max: float) -> None:
    if not events:
        return
    span = max(y_max - y_min, 1e-6)
    label_levels = (
        y_max - 0.06 * span,
        y_max - 0.15 * span,
        y_max - 0.24 * span,
    )
    for idx, event in enumerate(events):
        start = pd.to_datetime(event.get("visible_start"), errors="coerce")
        end = pd.to_datetime(event.get("visible_end"), errors="coerce")
        if pd.isna(start) or pd.isna(end):
            continue
        midpoint = start + (end - start) / 2
        short = EVENT_SHORT_LABELS.get(str(event.get("key") or "").strip(), str(event.get("label") or "").strip())
        ax.text(
            midpoint,
            label_levels[idx % len(label_levels)],
            short,
            ha="center",
            va="top",
            fontsize=7.0,
            color="#334155",
            rotation=90,
            rotation_mode="anchor",
            alpha=0.92,
            bbox={
                "boxstyle": "round,pad=0.12",
                "facecolor": "white",
                "edgecolor": "none",
                "alpha": 0.68,
            },
        )


def draw_metric_panels(
    region_frames: Sequence[RegionFrame],
    events: Sequence[Dict[str, Any]],
    metrics: Sequence[str],
    *,
    title: str,
    subtitle: str,
    output_stem: Path,
    start_date: pd.Timestamp,
) -> List[Path]:
    """Render a multi-region metric panel and save it as PDF + SVG."""
    long_df = build_metric_long_frame(region_frames, metrics, start_date=start_date)
    if long_df.empty:
        return []

    configure_theory_plot_style()
    fig, axes = plt.subplots(
        nrows=len(region_frames),
        ncols=1,
        figsize=(10.9, 2.8 * len(region_frames) + 1.0),
        sharex=True,
        constrained_layout=False,
    )
    if len(region_frames) == 1:
        axes = [axes]
    fig.subplots_adjust(left=0.085, right=0.975, top=0.92, bottom=0.08, hspace=0.14)

    palette = dict(zip([METRIC_LABELS.get(metric, metric) for metric in metrics], sns.color_palette("colorblind", n_colors=len(metrics))))
    legend_handles = None
    legend_labels = None

    for ax, region in zip(axes, region_frames):
        region_df = long_df[long_df["region_key"] == region.key].copy()
        if region_df.empty:
            ax.set_visible(False)
            continue
        for metric in metrics:
            metric_label = METRIC_LABELS.get(metric, metric)
            metric_df = region_df[region_df["metric"] == metric].copy()
            if metric_df.empty:
                continue
            ax.plot(
                metric_df["date"],
                metric_df["value"],
                label=metric_label,
                color=palette[metric_label],
                linewidth=2.0,
                alpha=0.98,
            )
        ax.axhline(0.0, color="#94a3b8", linewidth=1.0, alpha=0.7, zorder=0)
        local_events = filter_dashboard_events(
            events,
            region_key=region.key,
            start_date=region_df["date"].min(),
            end_date=region_df["date"].max(),
        )
        for event in local_events:
            start = pd.to_datetime(event.get("visible_start"), errors="coerce")
            end = pd.to_datetime(event.get("visible_end"), errors="coerce")
            if pd.isna(start) or pd.isna(end):
                continue
            ax.axvspan(start, end, color=_event_color(event), alpha=0.09, lw=0, zorder=0)

        ax.set_ylabel("Within-region robust score")
        ax.set_title(region.label, loc="left", pad=4)
        ax.tick_params(axis="y", labelsize=8.5)
        ax.tick_params(axis="x", labelsize=8.5)
        ax.xaxis.set_major_locator(mdates.YearLocator(base=4))
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
        latest_date = pd.to_datetime(region_df["date"].max()).strftime("%Y-%m-%d")
        ax.text(
            0.995,
            0.97,
            f"latest {latest_date}",
            transform=ax.transAxes,
            ha="right",
            va="top",
            fontsize=8,
            color="#64748b",
            bbox={
                "boxstyle": "round,pad=0.18",
                "facecolor": "white",
                "edgecolor": "none",
                "alpha": 0.82,
            },
        )
        y_min, y_max = ax.get_ylim()
        _annotate_events(ax, local_events, y_min, y_max)
        if legend_handles is None:
            legend_handles, legend_labels = ax.get_legend_handles_labels()
        if ax.get_legend() is not None:
            ax.get_legend().remove()

    axes[-1].set_xlabel("Date")
    fig.suptitle(title, y=0.97, fontsize=14, fontweight="semibold")
    fig.text(0.5, 0.02, subtitle, ha="center", va="bottom", fontsize=8.8, color="#475569")
    if legend_handles and legend_labels:
        fig.legend(
            legend_handles,
            legend_labels,
            loc="upper center",
            bbox_to_anchor=(0.5, 0.955),
            ncol=max(2, len(metrics)),
            columnspacing=1.2,
            handlelength=2.2,
        )

    output_stem.parent.mkdir(parents=True, exist_ok=True)
    outputs = [output_stem.with_suffix(".pdf"), output_stem.with_suffix(".svg")]
    fig.savefig(outputs[0], format="pdf", bbox_inches="tight")
    fig.savefig(outputs[1], format="svg", bbox_inches="tight")
    plt.close(fig)
    return outputs


def build_theory_figures(
    *,
    site_dir: Path,
    output_dir: Path,
    events_path: Path,
    start_date: str = "1998-01-01",
    source_ref: str | None = None,
) -> List[Path]:
    """Generate all paper-ready figures used by the LaTeX note."""
    region_frames = load_region_frames(site_dir, source_ref=source_ref)
    if not region_frames:
        return []
    events = load_dashboard_events(str(events_path))
    plot_start = pd.to_datetime(start_date, errors="coerce")
    if pd.isna(plot_start):
        plot_start = pd.Timestamp("1998-01-01")

    outputs: List[Path] = []
    outputs.extend(
        draw_metric_panels(
            region_frames,
            events,
            ("S_M", "T_L"),
            title="Thermo-Credit Regime Lines from Dashboard Data",
            subtitle="S_M and T_L use a winsorized median/MAD score with smooth asinh compression within region; shaded bands follow the shared event registry.",
            output_stem=output_dir / "theory_sm_tl_panels",
            start_date=plot_start,
        )
    )
    outputs.extend(
        draw_metric_panels(
            region_frames,
            events,
            ("X_C", "loop_area"),
            title="Thermo-Credit Capacity and Dissipation Panels",
            subtitle="X_C here is the smoothed dashboard proxy, shown with the same winsorized median/MAD score and smooth asinh compression; the tuned implicit headroom score is summarized separately in the theory snapshot.",
            output_stem=output_dir / "theory_capacity_panels",
            start_date=plot_start,
        )
    )
    return outputs
