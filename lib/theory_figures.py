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
os.environ.setdefault("SOURCE_DATE_EPOCH", "1704067200")

import matplotlib

matplotlib.use("Agg")

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
try:
    import seaborn as sns
except ModuleNotFoundError:
    sns = None

from lib.report_helpers import filter_dashboard_events, load_dashboard_events

REGION_SPECS: Sequence[tuple[str, str, Sequence[str]]] = (
    ("jp", "Japan (JP)", ("indicators_jp.csv", "indicators.csv")),
    ("eu", "Euro Area (EU)", ("indicators_eu.csv",)),
    ("us", "United States (US)", ("indicators_us.csv",)),
)

METRIC_LABELS = {
    "S_M": "S_M",
    "T_L": "T_L (liquidity state)",
    "X_C": "X_C",
    "loop_area": "Streaming loop area",
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
    source_path: str = ""
    panel_mode: str = "dashboard"


def configure_theory_plot_style() -> None:
    """Set a paper-friendly plotting style with modern defaults."""
    if sns is None:
        try:
            plt.style.use("seaborn-v0_8-whitegrid")
        except OSError:
            plt.style.use("default")
        plt.rcParams.update(
            {
                "axes.spines.top": False,
                "axes.spines.right": False,
                "figure.dpi": 160,
                "savefig.dpi": 220,
                "svg.hashsalt": "thermo-credit",
            }
        )
        return
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
            "svg.hashsalt": "thermo-credit",
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


def _realtime_candidate(candidate: str) -> str:
    if candidate == "indicators.csv":
        return "indicators_realtime.csv"
    return candidate.replace(".csv", "_realtime.csv")


def _candidate_sequence(candidates: Sequence[str], mode: str) -> list[str]:
    if mode == "realtime":
        out: list[str] = []
        for candidate in candidates:
            rt = _realtime_candidate(candidate)
            if rt not in out:
                out.append(rt)
            if candidate not in out:
                out.append(candidate)
        return out
    return list(candidates)


def load_region_frames(site_dir: Path, source_ref: str | None = None, *, mode: str = "dashboard") -> List[RegionFrame]:
    """Load the best available indicator frame for each region."""
    frames: List[RegionFrame] = []
    repo_root = site_dir.parent
    mode = (mode or "dashboard").strip().lower()
    for key, label, candidates in REGION_SPECS:
        frame: Optional[pd.DataFrame] = None
        source_path = ""
        for candidate in _candidate_sequence(candidates, mode):
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
                source_path = f"{site_dir.name}/{candidate}"
                break
        if frame is not None:
            panel_mode = "realtime" if "realtime" in source_path else "dashboard"
            frames.append(RegionFrame(key=key, label=label, frame=frame, source_path=source_path, panel_mode=panel_mode))
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


def _prepare_metric_series(metric: str, series: pd.Series, *, smooth: bool = True) -> pd.Series:
    numeric = pd.to_numeric(series, errors="coerce")
    if smooth and metric == "X_C":
        return numeric.rolling(window=4, min_periods=1).median()
    return numeric


def build_metric_long_frame(
    region_frames: Iterable[RegionFrame],
    metrics: Sequence[str],
    *,
    start_date: pd.Timestamp,
    transform: str = "robust",
) -> pd.DataFrame:
    """Convert region panels into a long-form raw or robust-score frame."""
    parts: List[pd.DataFrame] = []
    use_raw = transform == "raw"
    for region in region_frames:
        local = region.frame.copy()
        local = local[local["date"] >= start_date].copy()
        if local.empty:
            continue
        for metric in metrics:
            if metric not in local.columns:
                continue
            prepared = _prepare_metric_series(metric, local[metric], smooth=not use_raw)
            values = prepared if use_raw else _robust_score_series(prepared)
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


def _apply_event_spans(ax: Any, events: Sequence[Dict[str, Any]]) -> None:
    for event in events:
        start = pd.to_datetime(event.get("visible_start"), errors="coerce")
        end = pd.to_datetime(event.get("visible_end"), errors="coerce")
        if pd.isna(start) or pd.isna(end):
            continue
        ax.axvspan(start, end, color=_event_color(event), alpha=0.09, lw=0, zorder=0)


def _plot_raw_metric_rows(
    ax: Any,
    raw_df: pd.DataFrame,
    metrics: Sequence[str],
    palette: Dict[str, Any],
) -> List[Any]:
    handles: List[Any] = []
    if raw_df.empty:
        return handles
    if len(metrics) == 2:
        axes = [ax, ax.twinx()]
    else:
        axes = [ax for _ in metrics]
    for metric, local_ax in zip(metrics, axes):
        metric_label = METRIC_LABELS.get(metric, metric)
        metric_df = raw_df[raw_df["metric"] == metric].copy()
        if metric_df.empty:
            continue
        line = local_ax.plot(
            metric_df["date"],
            metric_df["value"],
            label=f"{metric_label} raw",
            color=palette[metric_label],
            linewidth=1.55,
            alpha=0.94,
        )[0]
        handles.append(line)
        if len(metrics) == 2:
            local_ax.set_ylabel(f"{metric_label} raw", color=palette[metric_label])
            local_ax.tick_params(axis="y", labelcolor=palette[metric_label], labelsize=8.0)
    if len(metrics) != 2:
        ax.set_ylabel("Raw reported units")
    return handles


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
    score_df = build_metric_long_frame(region_frames, metrics, start_date=start_date, transform="robust")
    raw_df = build_metric_long_frame(region_frames, metrics, start_date=start_date, transform="raw")
    if score_df.empty and raw_df.empty:
        return []

    configure_theory_plot_style()
    row_count = 2 * len(region_frames)
    fig, axes = plt.subplots(
        nrows=row_count,
        ncols=1,
        figsize=(10.9, 4.45 * len(region_frames) + 1.1),
        sharex=True,
        constrained_layout=False,
    )
    if row_count == 1:
        axes = [axes]
    fig.subplots_adjust(left=0.09, right=0.91, top=0.88, bottom=0.065, hspace=0.18)

    labels = [METRIC_LABELS.get(metric, metric) for metric in metrics]
    if sns is not None:
        colors = sns.color_palette("colorblind", n_colors=len(metrics))
    else:
        color_cycle = plt.rcParams["axes.prop_cycle"].by_key().get("color", [])
        if not color_cycle:
            color_cycle = ["C0", "C1", "C2", "C3", "C4", "C5"]
        colors = [color_cycle[idx % len(color_cycle)] for idx in range(len(metrics))]
    palette = dict(zip(labels, colors))
    legend_handles: List[Any] = []
    legend_labels: List[str] = []

    for idx, region in enumerate(region_frames):
        raw_ax = axes[2 * idx]
        score_ax = axes[2 * idx + 1]
        region_raw = raw_df[raw_df["region_key"] == region.key].copy()
        region_score = score_df[score_df["region_key"] == region.key].copy()
        region_dates = pd.concat(
            [region_raw.get("date", pd.Series(dtype="datetime64[ns]")), region_score.get("date", pd.Series(dtype="datetime64[ns]"))],
            ignore_index=True,
        ).dropna()
        if region_dates.empty:
            raw_ax.set_visible(False)
            score_ax.set_visible(False)
            continue

        local_events = filter_dashboard_events(
            events,
            region_key=region.key,
            start_date=region_dates.min(),
            end_date=region_dates.max(),
        )
        _apply_event_spans(raw_ax, local_events)
        _apply_event_spans(score_ax, local_events)

        raw_handles = _plot_raw_metric_rows(raw_ax, region_raw, metrics, palette)
        raw_ax.set_title(f"{region.label} - raw reported units", loc="left", pad=4)
        raw_ax.tick_params(axis="x", labelsize=8.5)
        raw_ax.xaxis.set_major_locator(mdates.YearLocator(base=4))
        raw_ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
        for handle in raw_handles:
            label = handle.get_label()
            if label not in legend_labels:
                legend_handles.append(handle)
                legend_labels.append(label)

        for metric in metrics:
            metric_label = METRIC_LABELS.get(metric, metric)
            metric_df = region_score[region_score["metric"] == metric].copy()
            if metric_df.empty:
                continue
            line = score_ax.plot(
                metric_df["date"],
                metric_df["value"],
                label=f"{metric_label} score",
                color=palette[metric_label],
                linewidth=2.0,
                alpha=0.98,
            )[0]
            label = line.get_label()
            if label not in legend_labels:
                legend_handles.append(line)
                legend_labels.append(label)
        score_ax.axhline(0.0, color="#94a3b8", linewidth=1.0, alpha=0.7, zorder=0)
        score_ax.set_ylabel("Within-region robust score")
        score_ax.set_title(f"{region.label} - normalized historical position", loc="left", pad=4)
        score_ax.tick_params(axis="y", labelsize=8.5)
        score_ax.tick_params(axis="x", labelsize=8.5)
        score_ax.xaxis.set_major_locator(mdates.YearLocator(base=4))
        score_ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
        latest_date = pd.to_datetime(region_dates.max()).strftime("%Y-%m-%d")
        raw_ax.text(
            0.995,
            0.97,
            f"latest {latest_date}",
            transform=raw_ax.transAxes,
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
        y_min, y_max = score_ax.get_ylim()
        _annotate_events(score_ax, local_events, y_min, y_max)
        for local_ax in (raw_ax, score_ax):
            if local_ax.get_legend() is not None:
                local_ax.get_legend().remove()

    axes[-1].set_xlabel("Date")
    fig.suptitle(title, y=0.985, fontsize=14, fontweight="semibold")
    fig.text(0.5, 0.017, subtitle, ha="center", va="bottom", fontsize=8.8, color="#475569")
    if legend_handles:
        fig.legend(
            legend_handles,
            legend_labels,
            loc="upper center",
            bbox_to_anchor=(0.5, 0.948),
            ncol=max(2, min(4, len(legend_handles))),
            columnspacing=1.2,
            handlelength=2.2,
        )

    output_stem.parent.mkdir(parents=True, exist_ok=True)
    outputs = [output_stem.with_suffix(".pdf"), output_stem.with_suffix(".svg")]
    fig.savefig(
        outputs[0],
        format="pdf",
        bbox_inches="tight",
        metadata={"CreationDate": None, "ModDate": None},
    )
    fig.savefig(
        outputs[1],
        format="svg",
        bbox_inches="tight",
        metadata={"Date": None},
    )
    plt.close(fig)
    return outputs


def _safe_numeric(frame: pd.DataFrame, column: str) -> pd.Series:
    if column not in frame.columns:
        return pd.Series(np.nan, index=frame.index, dtype=float)
    return pd.to_numeric(frame[column], errors="coerce")


def _positive_log(series: pd.Series) -> pd.Series:
    numeric = pd.to_numeric(series, errors="coerce")
    return pd.Series(np.log(numeric.where(numeric > 0)), index=series.index, dtype=float)


def _forward_acceleration(level: pd.Series, horizon: int) -> pd.Series:
    logged = _positive_log(level)
    future = logged.shift(-horizon) - logged
    trailing = logged - logged.shift(horizon)
    return future - trailing


def _forward_change(series: pd.Series, horizon: int) -> pd.Series:
    numeric = pd.to_numeric(series, errors="coerce")
    return numeric.shift(-horizon) - numeric


def _first_available_column(frame: pd.DataFrame, candidates: Sequence[str], min_rows: int = 12) -> str | None:
    for column in candidates:
        if column not in frame.columns:
            continue
        values = pd.to_numeric(frame[column], errors="coerce").dropna()
        if values.size >= min_rows and float(values.std(ddof=0)) > 1e-12:
            return column
    return None


def _complement_borrower_share(frame: pd.DataFrame) -> pd.Series:
    share = _safe_numeric(frame, "one_minus_q_t")
    if share.dropna().empty:
        c_t = _safe_numeric(frame, "C_t").where(lambda s: s > 0)
        share = 1.0 - (_safe_numeric(frame, "C_NFB") / c_t)
    return share


def _fit_line(ax: Any, x: pd.Series, y: pd.Series, color: str) -> None:
    local = pd.DataFrame({"x": x, "y": y}).replace([np.inf, -np.inf], np.nan).dropna()
    if local.shape[0] < 8 or float(local["x"].std(ddof=0)) <= 1e-12:
        return
    beta = np.polyfit(local["x"].to_numpy(dtype=float), local["y"].to_numpy(dtype=float), deg=1)
    xs = np.linspace(float(local["x"].min()), float(local["x"].max()), 50)
    ys = beta[0] * xs + beta[1]
    ax.plot(xs, ys, color=color, linewidth=1.6, alpha=0.82)


def draw_jp_bridge_state_panel(
    region_frames: Sequence[RegionFrame],
    *,
    output_stem: Path,
    start_date: pd.Timestamp,
    common_start: pd.Timestamp = pd.Timestamp("2009-06-30"),
    lambda_b: float = 0.5,
) -> List[Path]:
    """Render the audited BOJ bridge state and low-denominator observations."""
    jp_region = next((region for region in region_frames if region.key == "jp"), None)
    if jp_region is None:
        return []

    frame = jp_region.frame.copy()
    if "date" not in frame.columns:
        return []
    frame = frame.assign(date=pd.to_datetime(frame["date"], errors="coerce"))
    frame = frame.dropna(subset=["date"]).sort_values("date")
    effective_start = max(pd.Timestamp(start_date), common_start)
    frame = frame[frame["date"] >= effective_start].copy()
    required = {"q_t", "C_t"}
    if frame.empty or not required.issubset(frame.columns):
        return []

    c_t = _safe_numeric(frame, "C_t")
    c_t_4q = c_t.rolling(window=4, min_periods=4).sum()
    q_t = _safe_numeric(frame, "q_t")
    q_1q = _safe_numeric(frame, "borrower_composition_NFB_1q")
    if q_1q.dropna().empty:
        q_1q = _safe_numeric(frame, "borrower_composition_G_1q")
    if q_1q.dropna().empty:
        q_1q = _safe_numeric(frame, "operating_borrower_share_1q")
    if q_1q.dropna().empty:
        q_1q = _safe_numeric(frame, "share_G_direct")
    local = pd.DataFrame(
        {
            "date": frame["date"],
            "q_t": q_t,
            "q_1q": q_1q,
            "positive_flow": c_t_4q,
        }
    ).replace([np.inf, -np.inf], np.nan)
    local = local.dropna(subset=["q_t", "positive_flow"])
    local = local[local["positive_flow"] > 0].copy()
    if len(local) < 4:
        return []

    low_flow_threshold = float(local["positive_flow"].quantile(0.20))
    low_flow = local["positive_flow"] <= low_flow_threshold
    flow_trillion_yen = local["positive_flow"] / 10000.0
    threshold_trillion_yen = low_flow_threshold / 10000.0

    configure_theory_plot_style()
    fig, axes = plt.subplots(
        nrows=2,
        ncols=1,
        figsize=(9.2, 5.8),
        sharex=True,
        constrained_layout=False,
        gridspec_kw={"height_ratios": [1.3, 0.8]},
    )
    fig.subplots_adjust(left=0.10, right=0.98, top=0.88, bottom=0.14, hspace=0.18)
    state_ax, flow_ax = axes

    state_ax.plot(
        local["date"],
        local["q_1q"],
        color="#94a3b8",
        linewidth=1.0,
        alpha=0.85,
        label="One-quarter sensitivity",
    )
    state_ax.plot(
        local["date"],
        local["q_t"],
        color="#1d4ed8",
        linewidth=2.2,
        label="Primary four-quarter NFB share",
    )
    state_ax.scatter(
        local.loc[low_flow, "date"],
        local.loc[low_flow, "q_t"],
        s=34,
        marker="D",
        color="#d97706",
        edgecolor="white",
        linewidth=0.45,
        zorder=4,
        label="Low-denominator quarter",
    )
    state_ax.set_ylabel("NFB borrower share, $q_t$")
    state_ax.set_ylim(0.0, 1.0)
    state_ax.legend(loc="upper right", frameon=False, ncol=3, fontsize=8.4)
    state_ax.set_title("Borrower-composition bridge", loc="left")

    flow_ax.plot(
        local["date"],
        flow_trillion_yen,
        color="#0f766e",
        linewidth=1.7,
    )
    flow_ax.fill_between(
        local["date"],
        0.0,
        flow_trillion_yen,
        color="#99f6e4",
        alpha=0.35,
    )
    flow_ax.axhline(
        threshold_trillion_yen,
        color="#d97706",
        linewidth=1.0,
        linestyle="--",
        label="20th-percentile threshold",
    )
    flow_ax.set_ylabel("Four-quarter positive\nchanges (JPY trillion)")
    flow_ax.set_title("Mapped four-quarter denominator", loc="left")
    flow_ax.legend(loc="upper right", frameon=False, fontsize=8.4)
    flow_ax.xaxis.set_major_locator(mdates.YearLocator(base=2))
    flow_ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))

    latest_date = pd.to_datetime(local["date"].max()).strftime("%Y-%m-%d")
    fig.suptitle(
        "BOJ Borrower-Composition Bridge after the Classification Break",
        y=0.965,
        fontsize=13.5,
        fontweight="semibold",
    )
    fig.text(
        0.5,
        0.035,
        "Borrower-sector stock changes, not gross loan-purpose flows. "
        f"Low-denominator marks the bottom sample quintile; latest observation: {latest_date}.",
        ha="center",
        va="bottom",
        fontsize=8.6,
        color="#475569",
    )

    output_stem.parent.mkdir(parents=True, exist_ok=True)
    outputs = [output_stem.with_suffix(".pdf"), output_stem.with_suffix(".svg")]
    fig.savefig(outputs[0], format="pdf", bbox_inches="tight", metadata={"CreationDate": None, "ModDate": None})
    fig.savefig(outputs[1], format="svg", bbox_inches="tight", metadata={"Date": None})
    plt.close(fig)
    return outputs


def draw_jp_destination_target_panel(
    region_frames: Sequence[RegionFrame],
    events: Sequence[Dict[str, Any]],
    *,
    output_stem: Path,
    start_date: pd.Timestamp,
    horizon: int = 4,
) -> List[Path]:
    """Render Japan borrower shares against the recorded application series."""
    jp_region = next((region for region in region_frames if region.key == "jp"), None)
    if jp_region is None:
        return []

    frame = jp_region.frame.copy()
    if "date" not in frame.columns:
        return []
    frame = frame.assign(date=pd.to_datetime(frame["date"], errors="coerce"))
    frame = frame.dropna(subset=["date"]).sort_values("date")
    frame = frame[frame["date"] >= start_date].copy()
    if frame.empty or "q_t" not in frame.columns:
        return []

    q_t = _safe_numeric(frame, "q_t")
    complement_share = _complement_borrower_share(frame)
    asset_col = _first_available_column(frame, ("house_price", "land_price", "equity_price", "asset_price", "A", "L_asset"))
    spread_col = _first_available_column(frame, ("spread", "hy_oas", "credit_spread"))
    if asset_col is None or spread_col is None:
        return []

    asset_accel = _forward_acceleration(frame[asset_col], horizon)
    spread_change = _forward_change(frame[spread_col], horizon)
    time_panel = pd.DataFrame(
        {"date": frame["date"], "q_t": q_t, "complement_share": complement_share}
    ).dropna(subset=["q_t"])
    target_panel = pd.DataFrame(
        {
            "date": frame["date"],
            "q_t": q_t,
            "complement_share": complement_share,
            "asset_accel": asset_accel,
            "spread_change": spread_change,
        }
    )
    if time_panel.empty or target_panel[["asset_accel", "spread_change"]].dropna(how="all").empty:
        return []

    configure_theory_plot_style()
    fig, axes = plt.subplots(
        nrows=2,
        ncols=2,
        figsize=(10.2, 6.8),
        constrained_layout=False,
        gridspec_kw={"height_ratios": [1.08, 1.0]},
    )
    fig.subplots_adjust(left=0.08, right=0.98, top=0.88, bottom=0.11, hspace=0.38, wspace=0.28)
    top_ax = axes[0, 0]
    top_ax2 = axes[0, 1]
    scatter_asset = axes[1, 0]
    scatter_spread = axes[1, 1]

    local_events = filter_dashboard_events(
        events,
        region_key="jp",
        start_date=time_panel["date"].min(),
        end_date=time_panel["date"].max(),
    )
    for ax in (top_ax, top_ax2):
        _apply_event_spans(ax, local_events)

    top_ax.plot(time_panel["date"], time_panel["q_t"], color="#2563eb", linewidth=1.9, label="$q_t$")
    top_ax.set_title("Japan borrower composition", loc="left")
    top_ax.set_ylabel("$q_t$")
    top_ax.set_ylim(max(0.0, float(time_panel["q_t"].quantile(0.01)) - 0.04), min(1.0, float(time_panel["q_t"].quantile(0.99)) + 0.04))
    top_ax.xaxis.set_major_locator(mdates.YearLocator(base=4))
    top_ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))

    share_panel = time_panel.dropna(subset=["complement_share"])
    top_ax2.plot(
        share_panel["date"],
        share_panel["complement_share"],
        color="#c2410c",
        linewidth=1.9,
        label="$1-q_t$",
    )
    top_ax2.set_title("Other included borrowers", loc="left")
    top_ax2.set_ylabel("$1-q_t$")
    if not share_panel.empty:
        top_ax2.set_ylim(
            max(0.0, float(share_panel["complement_share"].quantile(0.01)) - 0.04),
            min(1.0, float(share_panel["complement_share"].quantile(0.99)) + 0.04),
        )
    top_ax2.xaxis.set_major_locator(mdates.YearLocator(base=4))
    top_ax2.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))

    asset_local = target_panel.dropna(subset=["q_t", "asset_accel"])
    scatter_asset.scatter(asset_local["q_t"], asset_local["asset_accel"], s=20, alpha=0.72, color="#2563eb", edgecolor="white", linewidth=0.35)
    _fit_line(scatter_asset, asset_local["q_t"], asset_local["asset_accel"], "#1d4ed8")
    scatter_asset.axhline(0.0, color="#94a3b8", linewidth=0.9)
    scatter_asset.set_title(f"Next {horizon}Q: BOJ balance-sheet acceleration", loc="left")
    scatter_asset.set_xlabel("$q_t$")
    scatter_asset.set_ylabel("Forward balance-sheet acceleration")

    spread_local = target_panel.dropna(subset=["complement_share", "spread_change"])
    scatter_spread.scatter(
        spread_local["complement_share"],
        spread_local["spread_change"],
        s=20,
        alpha=0.72,
        color="#c2410c",
        edgecolor="white",
        linewidth=0.35,
    )
    _fit_line(
        scatter_spread,
        spread_local["complement_share"],
        spread_local["spread_change"],
        "#9a3412",
    )
    scatter_spread.axhline(0.0, color="#94a3b8", linewidth=0.9)
    scatter_spread.set_title(f"Next {horizon}Q: long-term JGB yield change", loc="left")
    scatter_spread.set_xlabel("$1-q_t$")
    scatter_spread.set_ylabel("Forward JGB yield change")

    latest_date = pd.to_datetime(time_panel["date"].max()).strftime("%Y-%m-%d")
    fig.suptitle("Japan Borrower Composition and Application Series", y=0.965, fontsize=14, fontweight="semibold")
    fig.text(
        0.5,
        0.025,
        f"Release-lagged Japan panel; lower axes use BOJ total assets and the long-term JGB yield. Latest observation: {latest_date}.",
        ha="center",
        va="bottom",
        fontsize=8.8,
        color="#475569",
    )

    output_stem.parent.mkdir(parents=True, exist_ok=True)
    outputs = [output_stem.with_suffix(".pdf"), output_stem.with_suffix(".svg")]
    fig.savefig(outputs[0], format="pdf", bbox_inches="tight", metadata={"CreationDate": None, "ModDate": None})
    fig.savefig(outputs[1], format="svg", bbox_inches="tight", metadata={"Date": None})
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
    realtime_frames = load_region_frames(site_dir, source_ref=source_ref, mode="realtime")
    if not realtime_frames:
        realtime_frames = region_frames
    events = load_dashboard_events(str(events_path))
    plot_start = pd.to_datetime(start_date, errors="coerce")
    if pd.isna(plot_start):
        plot_start = pd.Timestamp("1998-01-01")

    outputs: List[Path] = []
    outputs.extend(
        draw_jp_bridge_state_panel(
            realtime_frames,
            output_stem=output_dir / "theory_boj_bridge_state",
            start_date=plot_start,
        )
    )
    outputs.extend(
        draw_jp_destination_target_panel(
            realtime_frames,
            events,
            output_stem=output_dir / "theory_jp_destination_targets",
            start_date=plot_start,
        )
    )
    outputs.extend(
        draw_metric_panels(
            region_frames,
            events,
            ("S_M", "T_L"),
            title="Prototype Monitoring Lines from Dashboard Data",
            subtitle="Each region shows raw reported units and a within-region winsorized median/MAD score with smooth asinh compression; shaded bands follow the shared event registry.",
            output_stem=output_dir / "theory_sm_tl_panels",
            start_date=plot_start,
        )
    )
    outputs.extend(
        draw_metric_panels(
            region_frames,
            events,
            ("X_C", "loop_area"),
            title="Prototype Headroom and Loop-Path Panels",
            subtitle="Raw panels preserve reported dashboard units; normalized panels show within-region historical position. The tuned implicit headroom score is summarized separately in the calibration table.",
            output_stem=output_dir / "theory_capacity_panels",
            start_date=plot_start,
        )
    )
    return outputs
