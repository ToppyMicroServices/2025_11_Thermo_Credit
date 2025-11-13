import os
import sys
import json
import shutil
import html as html_lib
from datetime import datetime
import base64
from io import BytesIO
try:
    from PIL import Image  # type: ignore
except Exception:
    Image = None  # Pillow optional; fallback to raw bytes
from typing import Any, Dict, List, Optional, Tuple
from urllib.parse import urlparse

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from lib.indicators import compute_diagnostics
from lib.raw_inputs import load_sources, enabled_sources, load_and_normalize

SITE_DIR = os.path.join(ROOT, "site")
DATA_DIR = os.path.join(ROOT, "data")
DEFAULT_BASE_URL = "https://toppymicros.com/2025_11_Thermo_Credit"
REQUIRED_THERMO_COLS = ["S_M", "T_L", "p_C", "V_C", "U"]
DERIVATIVE_COLS = ["dS_dV_at_T", "dp_dT_at_V", "maxwell_gap"]
FIRSTLAW_COLS = ["dU", "Q_like", "W_like", "dU_pred", "firstlaw_resid"]
CATEGORY_LABELS = {
    "q_productive": "Productive",
    "q_housing": "Housing",
    "q_consumption": "Consumption",
    "q_financial": "Financial",
    "q_government": "Government",
}
CATEGORY_LABELS = {
    "q_productive": "Productive",
    "q_housing": "Housing",
    "q_consumption": "Consumption",
    "q_financial": "Financial",
    "q_government": "Government",
}

# Expose raw_inputs_df at module level so tests can import this module and verify normalization
raw_inputs_df = None
try:
    # Strategy: try CWD/data first (honors monkeypatched chdir in tests),
    # then fall back to repo DATA_DIR. Take the first that yields any frames.
    for src_path in (os.path.join("data", "sources.json"), os.path.join(DATA_DIR, "sources.json")):
        try:
            srcs = load_sources(src_path)
            if srcs:
                cand = load_and_normalize(enabled_sources(srcs))
                if cand is not None:
                    raw_inputs_df = cand
                    break
        except Exception:
            continue
except Exception as exc:
    # Swallow but emit minimal diagnostic to aid CI debugging
    print("[report] raw_inputs module-level init failed:", exc)
    raw_inputs_df = None


def _load_csv(path: str) -> Optional[pd.DataFrame]:
    if not os.path.exists(path):
        return None
    try:
        df = pd.read_csv(path)
    except Exception:
        return None
    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"], errors="coerce")
        df = df.dropna(subset=["date"])
    return df


def _plot_start_date() -> pd.Timestamp:
    raw = os.getenv("REPORT_PLOT_START") or os.getenv("PLOT_START") or "2010-01-01"
    try:
        return pd.to_datetime(raw)
    except Exception:
        return pd.Timestamp("2010-01-01")


def _load_json(path: str) -> Optional[Dict[str, Any]]:
    if not os.path.exists(path):
        return None
    try:
        with open(path, "r", encoding="utf-8") as fp:
            data = json.load(fp)
    except Exception:
        return None
    return data if isinstance(data, dict) else None


def _resolve_diag_window(default: int = 24) -> int:
    raw = os.getenv("REPORT_DIAG_WINDOW") or os.getenv("DIAG_WINDOW")
    if raw:
        try:
            val = int(raw)
            if val >= 3:
                return val
        except ValueError:
            pass
    return default


def _calc_effective_window(frame: pd.DataFrame, requested: int) -> Tuple[int, str]:
    if not all(c in frame.columns for c in REQUIRED_THERMO_COLS):
        return 0, ""
    available = frame[REQUIRED_THERMO_COLS].dropna().shape[0]
    if requested and requested >= 3:
        if available >= requested:
            return requested, f" (window={requested})"
        if available >= 3:
            return available, f" (requested {requested}, using {available})"
        return 0, " (insufficient data for diagnostics)"
    if available >= 6:
        eff = min(24, available)
        return eff, f" (auto={eff})"
    if available >= 3:
        return available, f" (auto={available})"
    return 0, ""


def _style_figure(fig) -> None:
    fig.update_layout(
        template="plotly_white",
        hovermode="x unified",
        legend=dict(orientation="h", y=1.02, yanchor="bottom", x=1.0, xanchor="right"),
        margin=dict(t=60, b=40, l=40, r=20),
    )
    fig.update_xaxes(showgrid=False)
    fig.update_yaxes(showgrid=True, zeroline=True)
    # Serif stack for a LaTeX-like feel (applies to non-math text inside figures)
    fig.update_layout(font=dict(
        family="STIX Two Text, Times New Roman, Times, Georgia, serif",
        size=12,
    ))


def _apply_hover(fig, fmt: str) -> None:
    fig.update_traces(hovertemplate="%{x|%Y-%m-%d}<br>%{y:" + fmt + "}<extra>%{fullData.name}</extra>")


def make_dual_axis_sm_tl(plot_df: pd.DataFrame, title: str) -> go.Figure:
    """Build an S_M (left axis) and T_L (right axis) dual‑axis line chart.

    - Left y-axis: S_M (dispersion)
    - Right y-axis: T_L (liquidity temperature)
    """
    from plotly.subplots import make_subplots
    df = plot_df.copy()
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df = df.dropna(subset=["date"])
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    # Colors aligned with Plotly's default qualitative palette
    col_sm = "#1f77b4"  # blue
    col_tl = "#ff7f0e"  # orange
    # Primary axis: S_M
    if "S_M" in df.columns:
        fig.add_trace(
            go.Scatter(
                x=df["date"],
                y=pd.to_numeric(df["S_M"], errors="coerce"),
                name="S_M (dispersion)",
                mode="lines",
                line=dict(color=col_sm, width=2.0),
            ),
            secondary_y=False,
        )
    # Secondary axis: T_L
    if "T_L" in df.columns:
        fig.add_trace(
            go.Scatter(
                x=df["date"],
                y=pd.to_numeric(df["T_L"], errors="coerce"),
                name="T_L (liquidity temperature)",
                mode="lines",
                line=dict(color=col_tl, width=2.0, dash="solid"),
            ),
            secondary_y=True,
        )
    fig.update_layout(title=title, legend=dict(orientation="h", y=1.02, yanchor="bottom", x=1.0, xanchor="right"))
    fig.update_xaxes(title_text="Date")
    fig.update_yaxes(title_text="S_M (dispersion)", secondary_y=False)
    fig.update_yaxes(title_text="T_L (liquidity temperature)", secondary_y=True)
    # Refined visual touches
    fig.update_layout(plot_bgcolor="#fbfbfc")
    fig.update_yaxes(showgrid=True, gridcolor="#e9ecef", zeroline=True)
    return fig


def _filter_placeholders(df: pd.DataFrame) -> pd.DataFrame:
    """If a 'placeholder' column exists, filter out rows marked as placeholders.
    This hides provisional data from charts/tables while keeping them in source files.
    """
    if "placeholder" in df.columns:
        try:
            mask = ~(df["placeholder"].astype(bool))
            return df[mask].copy()
        except Exception:
            return df
    return df


def _out_of_spec_mask(df: pd.DataFrame) -> pd.Series:
    """Return a boolean mask for dates where diagnostics are out-of-spec.
    Uses robust thresholds based on MAD for maxwell_gap and firstlaw_resid.
    """
    import numpy as np
    idx = df.index
    mask = pd.Series(False, index=idx)
    for col in ("maxwell_gap", "firstlaw_resid"):
        if col not in df.columns:
            continue
        s = pd.to_numeric(df[col], errors="coerce")
        a = s.abs().dropna()
        if a.empty:
            continue
        if len(a) >= 12:
            mad = (a - a.median()).abs().median()
            thresh = float(a.median() + 6.0 * mad) if mad and mad > 0 else float(a.quantile(0.99))
        else:
            thresh = float(a.quantile(0.99))
        if not np.isfinite(thresh) or thresh <= 0:
            continue
        mask = mask | (s.abs() > thresh)
    return mask


def _mask_to_ranges(dates: pd.Series, mask: pd.Series) -> List[Tuple[pd.Timestamp, pd.Timestamp]]:
    """Convert a boolean mask to contiguous date ranges.
    Returns list of (start, end) inclusive timestamps.
    """
    ranges: List[Tuple[pd.Timestamp, pd.Timestamp]] = []
    if dates.empty or mask.empty or len(dates) != len(mask):
        return ranges
    current_start = None
    for dt_val, flag in zip(dates, mask):
        if flag and current_start is None:
            current_start = dt_val
        elif not flag and current_start is not None:
            ranges.append((current_start, prev_dt))
            current_start = None
        prev_dt = dt_val
    if current_start is not None:
        ranges.append((current_start, prev_dt))
    return ranges


def _augment_region_frame(frame: pd.DataFrame, effective_window: int, has_thermo: bool) -> Tuple[pd.DataFrame, bool]:
    local = frame.copy()
    for col in local.columns:
        if col != "date":
            local[col] = pd.to_numeric(local[col], errors="coerce")
    has_derivatives = all(c in local.columns for c in DERIVATIVE_COLS) and not local[DERIVATIVE_COLS].dropna(how="all").empty
    if has_thermo and effective_window >= 3 and not has_derivatives:
        local = compute_diagnostics(local.copy(), window=effective_window)
        has_derivatives = all(c in local.columns for c in DERIVATIVE_COLS) and not local[DERIVATIVE_COLS].dropna(how="all").empty
    needed_extra = DERIVATIVE_COLS + FIRSTLAW_COLS + ["Q_like", "W_like", "dU_pred"]
    for col in needed_extra:
        if col not in local.columns:
            local[col] = np.nan
    return local, has_derivatives


def _figs_html(specs: List[Tuple[Any, str, str]]) -> str:
    parts: List[str] = []
    for fig, title, alt in specs:
        html = fig.to_html(full_html=False, include_plotlyjs="cdn")
        parts.append(
            f"<figure aria-label=\"{html_lib.escape(alt)}\">{html}<figcaption>{html_lib.escape(title)}</figcaption></figure>"
        )
    return "".join(parts)


def _selected_table(meta: Optional[Dict[str, Any]], header: str) -> str:
    if not isinstance(meta, dict):
        return ""
    rows: List[Dict[str, Any]] = []
    for role, entry in meta.items():
        if isinstance(entry, dict):
            rows.append({
                "Role": role,
                "Series ID": entry.get("id", ""),
                "Source": entry.get("source", ""),
                "Start": entry.get("start", ""),
                "Title": entry.get("title", ""),
            })
    if not rows:
        return ""
    table = pd.DataFrame(rows).to_html(index=False, border=0, classes="mini", escape=True)
    return f"<h2>{html_lib.escape(header)} Selected Input Series</h2>{table}"


def _build_compare_context(region_ctxs: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    """Create a combined comparison section across regions for core metrics.

    Metrics compared: S_M, T_L, loop_area, X_C (when present per region).
    """
    if not region_ctxs:
        return None
    # Collect frames with data
    items: List[Tuple[str, pd.DataFrame]] = []
    for ctx in region_ctxs:
        label = ctx.get("label")
        frame = ctx.get("frame")
        if not isinstance(label, str) or not isinstance(frame, pd.DataFrame) or frame.empty:
            continue
        items.append((label, frame))
    if len(items) < 2:
        return None

    metric_specs = [
        ("S_M", "Compare – S_M", "Money entropy"),
        ("T_L", "Compare – T_L", "Liquidity temperature"),
        ("loop_area", "Compare – Policy Loop Dissipation", "Loop area"),
        ("X_C", "Compare – Credit Exergy Ceiling", "X_C"),
    ]

    raw_figs: List[Tuple[Any, str, str]] = []
    # Build a latest summary table
    latest_rows: List[Dict[str, Any]] = []
    for label, df in items:
        row: Dict[str, Any] = {"Region": label}
        if "date" in df.columns:
            dlast = pd.to_datetime(df["date"], errors="coerce").dropna()
            row["Latest date"] = dlast.iloc[-1].strftime("%Y-%m-%d") if not dlast.empty else ""
        for m, _, _ in metric_specs:
            if m in df.columns:
                try:
                    row[m] = float(pd.to_numeric(df[m], errors="coerce").dropna().iloc[-1])
                except Exception:
                    row[m] = None
            else:
                row[m] = None
        latest_rows.append(row)

    # Build figures per metric when at least one region has the metric
    start = _plot_start_date()
    for met, title, alt in metric_specs:
        long_parts: List[pd.DataFrame] = []
        for label, df in items:
            if "date" not in df.columns:
                continue
            # For X_C, fall back to F_C if missing or all NaN
            if met == "X_C":
                col = None
                if "X_C" in df.columns and pd.to_numeric(df["X_C"], errors="coerce").dropna().size > 0:
                    col = "X_C"
                elif "F_C" in df.columns and pd.to_numeric(df["F_C"], errors="coerce").dropna().size > 0:
                    col = "F_C"
                if col is None:
                    continue
                part = df[["date", col]].copy()
            else:
                if met not in df.columns:
                    continue
                part = df[["date", met]].copy()
            part = part[part["date"] >= start]
            part = part.rename(columns={part.columns[1]: "value"})
            part["Region"] = label
            long_parts.append(part)
        if not long_parts:
            continue
        long_df = pd.concat(long_parts, ignore_index=True)
        long_df["date"] = pd.to_datetime(long_df["date"], errors="coerce")
        long_df = long_df.dropna(subset=["date", "value"]).sort_values("date")
        if long_df.empty:
            continue
        # Use metric-specific y-axis label with plain-English hints (avoid math in Plotly SVG)
        y_label = {
            "S_M": "S_M (dispersion)",
            "T_L": "T_L (liquidity temperature)",
            "loop_area": "Loop area (dissipation)",
            "X_C": "X_C (credit exergy ceiling)",
        }.get(met, "Value")
        fig = px.line(
            long_df,
            x="date",
            y="value",
            color="Region",
            title=title,
            render_mode="svg",
            labels={"value": y_label, "date": "Date"},
        )
        _style_figure(fig)
        _apply_hover(fig, ".3f")
        raw_figs.append((fig, title.replace("Compare – ", "Compare: "), alt))

    raw_charts_html = _figs_html(raw_figs)

    # Standardized comparison (per-region z-scores) and normalized entropy
    std_figs: List[Tuple[Any, str, str]] = []
    def _z_of(series: pd.Series) -> Optional[pd.Series]:
        s = pd.to_numeric(series, errors="coerce")
        s = s.dropna()
        if s.empty:
            return None
        m = float(s.mean())
        sd = float(s.std())
        if not np.isfinite(sd) or sd <= 0:
            return None
        return (series.astype(float) - m) / sd

    # S_M_hat (already normalized 0..1 when K fixed)
    try:
        start = _plot_start_date()
        long_parts_hat: List[pd.DataFrame] = []
        for label, df in items:
            if "date" not in df.columns or "S_M_hat" not in df.columns:
                continue
            part = df[["date", "S_M_hat"]].copy()
            part = part[part["date"] >= start]
            part = part.rename(columns={"S_M_hat": "value"})
            part["Region"] = label
            long_parts_hat.append(part)
        if long_parts_hat:
            long_df_hat = pd.concat(long_parts_hat, ignore_index=True)
            long_df_hat["date"] = pd.to_datetime(long_df_hat["date"], errors="coerce")
            long_df_hat = long_df_hat.dropna(subset=["date", "value"]).sort_values("date")
            if not long_df_hat.empty:
                fig_hat = px.line(
                    long_df_hat,
                    x="date",
                    y="value",
                    color="Region",
                    title="Compare – S_M_hat (normalized entropy)",
                    render_mode="svg",
                    labels={"value": "S_M_hat", "date": "Date"},
                )
                _style_figure(fig_hat)
                _apply_hover(fig_hat, ".3f")
                std_figs.append((fig_hat, "Compare: S_M_hat", "S_M_hat"))
    except Exception:
        pass

    # Standardize T_L, loop_area, F_C, X_C (per-region z-score)
    for met, title, alt in [("T_L", "Compare – T_L (standardized)", "T_L z"),
                             ("loop_area", "Compare – Loop area (standardized)", "Loop area z"),
                             ("U", "Compare – Internal Energy (standardized)", "U z"),
                             ("dU", "Compare – ΔU (standardized)", "dU z"),
                             ("dF_C", "Compare – ΔF_C (standardized)", "dF_C z"),
                             ("F_C", "Compare – Free Energy (standardized)", "F_C z"),
                             ("X_C", "Compare – X_C (standardized)", "X_C z")]:
        long_parts_z: List[pd.DataFrame] = []
        for label, df in items:
            if "date" not in df.columns or met not in df.columns:
                continue
            series = pd.to_numeric(df[met], errors="coerce")
            z = _z_of(series)
            if z is None:
                continue
            part = pd.DataFrame({
                "date": pd.to_datetime(df["date"], errors="coerce"),
                "value": z,
                "Region": label,
            })
            part = part.dropna(subset=["date", "value"])  # filter invalid dates/values
            part = part[part["date"] >= start]
            long_parts_z.append(part)
        if not long_parts_z:
            continue
        long_df_z = pd.concat(long_parts_z, ignore_index=True).sort_values("date")
        if long_df_z.empty:
            continue
        figz = px.line(
            long_df_z,
            x="date",
            y="value",
            color="Region",
            title=title,
            render_mode="svg",
            labels={"value": "z-score (within region)", "date": "Date"},
        )
        _style_figure(figz)
        _apply_hover(figz, ".3f")
        std_figs.append((figz, title.replace("Compare – ", "Compare: "), alt))

    std_charts_html = _figs_html(std_figs) if std_figs else ""

    summary_html = ""
    if latest_rows:
        latest_df = pd.DataFrame(latest_rows)
        # Order columns nicely
        cols = [c for c in ["Region", "Latest date", "S_M", "T_L", "loop_area", "X_C"] if c in latest_df.columns]
        latest_df = latest_df[cols]
        # One-line headline to orient first-time readers
        try:
            # Pick a common latest date if present
            dates = pd.to_datetime(latest_df.get("Latest date"), errors="coerce").dropna()
            latest_dt_str = dates.max().strftime("%Y-%m-%d") if not dates.empty else ""
        except Exception:
            latest_dt_str = ""
        headline = (
            f"<p><strong>At the latest date</strong>{' (' + latest_dt_str + ')' if latest_dt_str else ''}, this section compares dispersion (S<sub>M</sub>), liquidity temperature (T<sub>L</sub>), loop dissipation, and remaining credit exergy (X<sub>C</sub>) across regions. The table below gives exact values.</p>"
        )
        summary_html = headline + "<h2>Compare – Latest snapshot</h2>" + latest_df.to_html(index=False, border=0, classes="mini", float_format=lambda x: f"{x:.4g}")

    # Build a toggle UI (Standardized default) and panes for raw vs standardized
    toggle_html = (
        '<div class="subtabs compare-toggle" role="tablist">'
        '<button class="active" data-mode="std" aria-pressed="true">Standardized</button>'
        '<button data-mode="raw" aria-pressed="false">Raw</button>'
        '</div>'
    )
    _std_inner = std_charts_html if std_charts_html else "<p class=\"note small\">No standardized charts available.</p>"
    _raw_inner = raw_charts_html if raw_charts_html else "<p class=\"note small\">No raw charts available.</p>"
    panes_html = (
        '<div class="compare-block">'
        f'<div class="pane std active">{_std_inner}</div>'
        f'<div class="pane raw">{_raw_inner}</div>'
        '</div>'
    )
    region_html = (
        f"<section class=\"region-summary\"><h2>Compare (JP/EU/US)</h2>{summary_html}{toggle_html}</section>" + panes_html
    )

    return {
        "key": "compare",
        "label": "Compare",
        "html": region_html,
        "fig_specs": raw_figs + std_figs,
        "summary_line": None,
        "summary_items": [],
        "has_maxwell_fig": False,
        "has_firstlaw_fig": False,
        "has_raw_inputs_fig": False,
        "last_date": max((pd.to_datetime(ctx.get("last_date")) for ctx in region_ctxs if ctx.get("last_date")), default=datetime.utcnow()),
        "frame": pd.DataFrame(),
    }


def _selected_summary_line(prefix: str, meta: Optional[Dict[str, Any]]) -> Optional[str]:
    if not isinstance(meta, dict):
        return None
    pieces: List[str] = []
    for role, entry in meta.items():
        if isinstance(entry, dict):
            sid = entry.get("id", "")
            start = entry.get("start", "")
            if sid:
                suffix = f"({start})" if start else ""
                pieces.append(f"{role}={sid}{suffix}")
    if not pieces:
        return None
    return f"{prefix} Selected: " + ", ".join(pieces)


def _role_label(role: str) -> str:
    mapping = {
        "money_scale": "Money scale",
        "base_proxy": "Base",
        "yield_proxy": "Long-term yield",
        "credit_volume": "Credit volume",
        "loan_spread": "Loan spread",
        "gov_yield": "Government yield",
        "corp_yield": "Corporate yield",
        "turnover": "Turnover",
    }
    # allow *_jp/_eu/_us suffixes
    base = role
    for suf in ("_jp", "_eu", "_us"):
        if role.endswith(suf):
            base = role[: -len(suf)]
            break
    return mapping.get(base, role)


def _build_inputs_summary(region_ctxs: List[Dict[str, Any]]) -> str:
    rows: List[str] = []
    for ctx in region_ctxs:
        label = ctx.get("label", "")
        meta = ctx.get("selected_meta")
        if not isinstance(meta, dict) or not meta:
            continue
        pills: List[str] = []
        for role, entry in meta.items():
            if not isinstance(entry, dict):
                continue
            title = entry.get("title") or entry.get("id", "")
            provider = entry.get("provider") or entry.get("source") or ""
            start = entry.get("start") or ""
            start_y = start[:4] if isinstance(start, str) and len(start) >= 4 else ""
            parts: List[str] = [f"<strong>{html_lib.escape(_role_label(role))}</strong>: {html_lib.escape(title)}"]
            tail: List[str] = []
            if provider:
                tail.append(html_lib.escape(provider))
            if start_y:
                tail.append(f"since {start_y}")
            if tail:
                parts.append(" (" + ", ".join(tail) + ")")
            pills.append('<span class="pill">' + "".join(parts) + "</span>")
        if pills:
            row_html = (
                '<div class="inputs-row">'
                f"<span class=\"region-tag\">{html_lib.escape(label)}</span> "
                + '<span class="pill-list">' + " ".join(pills) + "</span>"
                + "</div>"
            )
            rows.append(row_html)
    if not rows:
        return ""
    return '<section class="inputs-summary"><h2>Inputs summary</h2>' + "".join(rows) + "</section>"


def _selected_summary_sentence(prefix: str, meta: Optional[Dict[str, Any]]) -> Optional[str]:
    if not isinstance(meta, dict) or not meta:
        return None
    def pick(keys: List[str]) -> Optional[Dict[str, Any]]:
        # allow *_jp/_eu/_us suffixes
        for k in keys:
            if k in meta and isinstance(meta[k], dict):
                return meta[k]
            for suf in ("_jp", "_eu", "_us"):
                ks = k + suf
                if ks in meta and isinstance(meta[ks], dict):
                    return meta[ks]
        return None
    roles = [
        ("money_scale", "Money scale"),
        ("base_proxy", "Base"),
        ("yield_proxy", "Long-term yield"),
    ]
    parts: List[str] = []
    for key, label in roles:
        ent = pick([key])
        if ent:
            title = ent.get("title") or ent.get("id", "")
            start = ent.get("start") or ""
            start_y = start[:4] if isinstance(start, str) and len(start) >= 4 else ""
            tail = f" (since {start_y})" if start_y else ""
            parts.append(f"{label}: {html_lib.escape(title)}{tail}")
    if not parts:
        return None
    return f"{html_lib.escape(prefix)} — " + " | ".join(parts)


def _definitions_table(ref_df: pd.DataFrame) -> str:
    defs = {
        "date": ("Date", "End-of-month timestamp", "YYYY-MM-DD"),
        "S_M": ("Money entropy", "Mixing entropy", "index"),
        "T_L": ("Liquidity temperature", "Composite flow proxy", "index"),
        "p_C": ("Credit pressure", "Conjugate to V_C", "index"),
        "V_C": ("Credit volume", "Capacity proxy", "index"),
        "U": ("Internal energy", "Stored potential", "index"),
        "F_C": ("Free energy F_C", "Helmholtz proxy", "index"),
        "X_C": ("Exergy ceiling X_C", "Usable potential", "index"),
        "loop_area": ("Loop area", "Streaming dissipation", "index^2"),
    }
    cols = [c for c in defs if c == "date" or c in ref_df.columns]
    rows = [
        {"Column": col, "Name": defs[col][0], "Meaning": defs[col][1], "Unit/Scale": defs[col][2]}
        for col in cols
    ]
    if not rows:
        return ""
    table = pd.DataFrame(rows).to_html(index=False, border=0, classes="mini", escape=True)
    return "<h2>Data &amp; Definitions</h2>" + table


def _sources_table(sources_meta: List[Dict[str, Any]]) -> str:
    rows: List[Dict[str, Any]] = []
    for entry in sources_meta:
        if not isinstance(entry, dict):
            continue
        rows.append({
            "ID": entry.get("id", ""),
            "Title": entry.get("title", ""),
            "Provider": entry.get("provider", ""),
            "Country": entry.get("country", ""),
            "Freq": entry.get("frequency", ""),
            "Units": entry.get("units", ""),
            "Enabled": "yes" if entry.get("enabled") else "no",
        })
    if not rows:
        return ""
    table = pd.DataFrame(rows).to_html(index=False, border=0, classes="mini", escape=True)
    # Fold large sources table by default for first-time readers
    return "<details><summary>Data sources</summary>" + table + "</details>"


def _build_raw_inputs_fig(raw_df: Optional[pd.DataFrame]):
    if raw_df is None or raw_df.empty or "date" not in raw_df.columns:
        return None
    value_cols = [c for c in raw_df.columns if c != "date"]
    start = _plot_start_date()
    raw_df = raw_df[raw_df["date"] >= start]
    if not value_cols:
        return None
    long_df = raw_df.melt(id_vars="date", value_vars=value_cols, var_name="Series", value_name="Value")
    color_map = raw_df.attrs.get("series_country_map", {})
    palette = {"JP": "#1f77b4", "JPN": "#1f77b4", "EU": "#ff7f0e", "EZ": "#ff7f0e", "US": "#2ca02c", "USA": "#2ca02c"}
    discrete_map = {series: palette.get(country, "#6c757d") for series, country in color_map.items()}
    fig = px.line(
        long_df,
        x="date",
        y="Value",
        color="Series",
        title="Raw Inputs (normalized first=100)",
        color_discrete_map=discrete_map,
        render_mode="svg",
        labels={"Value": "Index (first=100)", "date": "Date", "Series": "Series"},
    )
    _style_figure(fig)
    _apply_hover(fig, ".2f")
    return fig


def _build_region_context(
    key: str,
    label: str,
    frame: Optional[pd.DataFrame],
    *,
    diag_window: int,
    selected_meta: Optional[Dict[str, Any]] = None,
    include_raw_inputs: bool = False,
    raw_inputs_fig=None,
) -> Optional[Dict[str, Any]]:
    if frame is None:
        return None
    local = frame.copy()
    def _empty_context() -> Dict[str, Any]:
        summary_items = ["No indicator data available yet."]
        plot_start = _plot_start_date()
        summary_html = "<p class=\"note\">No indicator data available yet.</p>"
        selected_table_html = _selected_table(selected_meta, label)
        region_html = (
            f"<section class=\"region-summary\"><h2>{html_lib.escape(label)}</h2>"
            f"{summary_html}{selected_table_html}</section>"
        )
        return {
            "key": key,
            "label": label,
            "html": region_html,
            "fig_specs": [],
            "summary_line": _selected_summary_line(label, selected_meta),
            "summary_items": summary_items,
            "has_maxwell_fig": False,
            "has_firstlaw_fig": False,
            "has_raw_inputs_fig": False,
            "last_date": datetime.utcnow(),
            "frame": local,
        }
    if local.empty:
        return _empty_context()
    if "date" in local.columns:
        local["date"] = pd.to_datetime(local["date"])
        local = local.sort_values("date").reset_index(drop=True)
    if local.empty:
        return _empty_context()
    # Hide provisional placeholders if marked
    local = _filter_placeholders(local)
    has_thermo = all(c in local.columns for c in REQUIRED_THERMO_COLS)
    effective_window, eff_note = _calc_effective_window(local, diag_window)
    local, has_derivatives = _augment_region_frame(local, effective_window, has_thermo)
    # Plot subset filtered by start date
    plot_start = _plot_start_date()
    plot_df = local[local["date"] >= plot_start].copy() if "date" in local.columns else local.copy()

    fig_specs: List[Tuple[Any, str, str]] = []
    if {"S_M", "T_L"}.issubset(local.columns) and not plot_df.empty:
        # Dual-axis layout for very different scales
        fig = make_dual_axis_sm_tl(plot_df, title=f"{label} – S_M & T_L")
        _style_figure(fig)
        _apply_hover(fig, ".3f")
        fig_specs.append((fig, "S_M & T_L", "Entropy & temperature"))
    # Stacked MECE entropy view when per-category columns exist
    cat_cols = [c for c in plot_df.columns if c.startswith("S_M_in_")]
    cat_cols = [c for c in cat_cols if pd.to_numeric(plot_df[c], errors="coerce").dropna().abs().sum() > 0]
    if cat_cols:
        long_df = plot_df[["date"] + cat_cols].melt(id_vars="date", var_name="category", value_name="value")
        long_df = long_df.dropna(subset=["date", "value"])
        if not long_df.empty:
            long_df["category_key"] = long_df["category"].str.replace("S_M_in_", "", n=1)
            long_df["Category"] = long_df["category_key"].map(CATEGORY_LABELS).fillna(
                long_df["category_key"].str.replace("_", " ").str.title()
            )
            fig_cat = px.area(
                long_df,
                x="date",
                y="value",
                color="Category",
                title=f"{label} – S_M by category",
                labels={"value": "S_M_in (per category)", "date": "Date", "Category": "Category"},
            )
            _style_figure(fig_cat)
            _apply_hover(fig_cat, ".3f")
            fig_specs.append((fig_cat, "S_M by category", "Entropy by MECE categories"))
    if "loop_area" in local.columns and not plot_df.empty:
        fig = px.line(
            plot_df,
            x="date",
            y="loop_area",
            title=f"{label} – Policy Loop Dissipation",
            labels={"loop_area": "Loop area (dissipation)", "date": "Date"},
        )
        _style_figure(fig)
        _apply_hover(fig, ".3f")
        fig_specs.append((fig, "Policy Loop Dissipation", "Loop area"))
    # Exergy, free energy, internal energy, change in free energy, and surplus/shortage figures
    if not plot_df.empty:
        # Exergy X_C (if available)
        if "X_C" in plot_df.columns and pd.to_numeric(plot_df["X_C"], errors="coerce").dropna().size > 0:
            fig_xc = px.line(
                plot_df,
                x="date",
                y="X_C",
                title=f"{label} – Credit Exergy Ceiling",
                labels={"X_C": "X_C (credit exergy ceiling)", "date": "Date"},
            )
            _style_figure(fig_xc)
            _apply_hover(fig_xc, ".3f")
            fig_specs.append((fig_xc, "Credit Exergy Ceiling", "X_C"))
        # Free energy F_C (always show if present)
        if "F_C" in plot_df.columns and pd.to_numeric(plot_df["F_C"], errors="coerce").dropna().size > 0:
            fig_fc = px.line(
                plot_df,
                x="date",
                y="F_C",
                title=f"{label} – Free Energy (F_C)",
                labels={"F_C": "F_C (free energy)", "date": "Date"},
            )
            _style_figure(fig_fc)
            _apply_hover(fig_fc, ".3f")
            fig_specs.append((fig_fc, "Free Energy (F_C)", "F_C"))
        # Change in free energy dF_C
        if "dF_C" in plot_df.columns and pd.to_numeric(plot_df["dF_C"], errors="coerce").dropna().size > 0:
            fig_dfc = px.line(
                plot_df,
                x="date",
                y="dF_C",
                title=f"{label} – ΔF_C (change in free energy)",
                labels={"dF_C": "ΔF_C", "date": "Date"},
            )
            _style_figure(fig_dfc)
            _apply_hover(fig_dfc, ".3f")
            fig_specs.append((fig_dfc, "ΔF_C (change)", "dF_C"))
        # Internal energy U
        if "U" in plot_df.columns and pd.to_numeric(plot_df["U"], errors="coerce").dropna().size > 0:
            fig_u = px.line(
                plot_df,
                x="date",
                y="U",
                title=f"{label} – Internal Energy (U)",
                labels={"U": "U (internal energy)", "date": "Date"},
            )
            _style_figure(fig_u)
            _apply_hover(fig_u, ".3f")
            fig_specs.append((fig_u, "Internal Energy (U)", "U"))
        # Surplus/Shortage split from ΔF_C
        plus_ok = "X_C_plus" in plot_df.columns and pd.to_numeric(plot_df["X_C_plus"], errors="coerce").dropna().size > 0
        minus_ok = "X_C_minus" in plot_df.columns and pd.to_numeric(plot_df["X_C_minus"], errors="coerce").dropna().size > 0
        if plus_ok or minus_ok:
            df_pm = plot_df[["date"]].copy()
            if plus_ok:
                df_pm["Surplus (X_C+)"] = pd.to_numeric(plot_df["X_C_plus"], errors="coerce")
            if minus_ok:
                df_pm["Shortage (X_C−)"] = pd.to_numeric(plot_df["X_C_minus"], errors="coerce")
            y_cols = [c for c in ["Surplus (X_C+)", "Shortage (X_C−)"] if c in df_pm.columns]
            if y_cols:
                fig_pm = px.area(
                    df_pm,
                    x="date",
                    y=y_cols,
                    title=f"{label} – Surplus/Shortage (ΔF_C split)",
                    labels={"value": "ΔF_C components (surplus/shortage)", "variable": "Component", "date": "Date"},
                )
                _style_figure(fig_pm)
                _apply_hover(fig_pm, ".3f")
                fig_specs.append((fig_pm, "Surplus/Shortage (ΔF_C)", "X_C_plus / X_C_minus"))

    deriv_cols_present = [c for c in DERIVATIVE_COLS if c in local.columns]
    out_of_spec_note = ""
    out_of_spec_ranges: List[Tuple[pd.Timestamp, pd.Timestamp]] = []
    if has_derivatives and effective_window >= 3 and deriv_cols_present and not plot_df.empty:
        title = f"{label} – Maxwell-like Relation"
        if eff_note:
            title += eff_note
        fig = px.line(
            plot_df,
            x="date",
            y=deriv_cols_present,
            title=title,
            markers=True,
            labels={"value": "Coefficient", "variable": "Series", "date": "Date"},
        )
        _style_figure(fig)
        _apply_hover(fig, ".3f")
        # Shade out-of-spec zones across the full plot if diagnostics spike
        try:
            mask = _out_of_spec_mask(plot_df)
            if mask.any():
                out_of_spec_ranges = _mask_to_ranges(plot_df["date"], mask)
                for (x0, x1) in out_of_spec_ranges:
                    fig.add_vrect(x0=x0, x1=x1, fillcolor="gray", opacity=0.12, line_width=0, layer="below")
        except Exception:
            pass
        fig_specs.append((fig, "Maxwell-like Test", "Derivatives"))
    firstlaw_cols = [c for c in ["dU", "dU_pred", "firstlaw_resid"] if c in local.columns]
    if has_thermo and firstlaw_cols and not plot_df.empty:
        fig = px.line(
            plot_df,
            x="date",
            y=firstlaw_cols,
            title=f"{label} – First-law Decomposition",
            markers=True,
            labels={"value": "Change", "variable": "Component", "date": "Date"},
        )
        _style_figure(fig)
        _apply_hover(fig, ".3f")
        # Mirror shading on first-law plot for same out-of-spec windows
        try:
            if not out_of_spec_ranges:
                mask2 = _out_of_spec_mask(plot_df)
                if mask2.any():
                    out_of_spec_ranges = _mask_to_ranges(plot_df["date"], mask2)
            for (x0, x1) in out_of_spec_ranges:
                fig.add_vrect(x0=x0, x1=x1, fillcolor="gray", opacity=0.12, line_width=0, layer="below")
        except Exception:
            pass
        fig_specs.append((fig, "First-law Decomposition", "ΔU vs predicted"))
    if include_raw_inputs and raw_inputs_fig is not None:
        fig_specs.append((raw_inputs_fig, "Raw Inputs (first=100)", "Normalized raw inputs"))

    charts_html = _figs_html(fig_specs)

    last_row = local.iloc[-1]
    last_ts = pd.to_datetime(last_row.get("date"), errors="coerce")
    last_date = last_ts.to_pydatetime() if not pd.isna(last_ts) else datetime.utcnow()
    fmt = lambda v: f"{float(v):.4g}" if pd.notna(v) else "n/a"
    summary_items: List[str] = []
    summary_items.append(f"Latest date: {last_date.strftime('%Y-%m-%d')}")
    if "S_M" in local.columns:
        summary_items.append(f"S_M: {fmt(last_row.get('S_M'))}")
    if "T_L" in local.columns:
        summary_items.append(f"T_L: {fmt(last_row.get('T_L'))}")
    if "loop_area" in local.columns:
        summary_items.append(f"Loop area: {fmt(last_row.get('loop_area'))}")
    if "U" in local.columns:
        summary_items.append(f"U: {fmt(last_row.get('U'))}")
    # Summary: show X_C if present; otherwise F_C label it accordingly
    # Also collect X_C behavior for interpretation and possible suppression
    xc_series = None
    if "X_C" in local.columns and pd.to_numeric(local["X_C"], errors="coerce").dropna().size > 0:
        summary_items.append(f"X_C: {fmt(last_row.get('X_C'))}")
        xc_series = pd.to_numeric(local["X_C"], errors="coerce").dropna()
    elif "F_C" in local.columns and pd.to_numeric(local["F_C"], errors="coerce").dropna().size > 0:
        summary_items.append(f"F_C: {fmt(last_row.get('F_C'))}")
    if has_derivatives and "maxwell_gap" in local.columns:
        summary_items.append(f"Maxwell gap: {fmt(last_row.get('maxwell_gap'))}")
    if has_thermo and "firstlaw_resid" in local.columns:
        summary_items.append(f"First-law resid: {fmt(last_row.get('firstlaw_resid'))}")
    summary_html = "<ul>" + "".join(f"<li>{html_lib.escape(item)}</li>" for item in summary_items) + "</ul>"

    # Add a short state comment to orient readers (low/mid/high by intra-series quantiles)
    def _bucket(val: Optional[float], series: pd.Series) -> Optional[str]:
        try:
            s = pd.to_numeric(series, errors="coerce").dropna()
            if s.size < 6 or val is None or not np.isfinite(val):
                return None
            q1, q2 = float(s.quantile(0.33)), float(s.quantile(0.66))
            if val <= q1:
                return "low"
            if val >= q2:
                return "high"
            return "mid-range"
        except Exception:
            return None

    try:
        last_sm = float(pd.to_numeric(local.get("S_M"), errors="coerce").dropna().iloc[-1]) if "S_M" in local.columns else None
    except Exception:
        last_sm = None
    try:
        last_tl = float(pd.to_numeric(local.get("T_L"), errors="coerce").dropna().iloc[-1]) if "T_L" in local.columns else None
    except Exception:
        last_tl = None
    try:
        last_la = float(pd.to_numeric(local.get("loop_area"), errors="coerce").dropna().iloc[-1]) if "loop_area" in local.columns else None
    except Exception:
        last_la = None
    try:
        last_xc = float(pd.to_numeric(local.get("X_C"), errors="coerce").dropna().iloc[-1]) if "X_C" in local.columns else None
    except Exception:
        last_xc = None

    sm_bucket = _bucket(last_sm, local.get("S_M", pd.Series(dtype=float))) if "S_M" in local.columns else None
    tl_bucket = _bucket(last_tl, local.get("T_L", pd.Series(dtype=float))) if "T_L" in local.columns else None
    la_desc = None
    if last_la is not None and np.isfinite(last_la):
        la_desc = "non-zero" if abs(last_la) > 1e-12 else "near zero"
    xc_desc = None
    if last_xc is not None and np.isfinite(last_xc):
        if last_xc <= 1e-9:
            xc_desc = "≈0 (limited remaining room)"
        else:
            xc_desc = "positive (some room remains)"

    parts: List[str] = []
    if sm_bucket and tl_bucket:
        parts.append(f"{label} sits in a <strong>{sm_bucket}-dispersion, {tl_bucket}-temperature</strong> regime.")
    elif sm_bucket or tl_bucket:
        if sm_bucket:
            parts.append(f"Dispersion is <strong>{sm_bucket}</strong>.")
        if tl_bucket:
            parts.append(f"Liquidity temperature is <strong>{tl_bucket}</strong>.")
    if la_desc:
        parts.append(f"Loop area is <strong>{la_desc}</strong>, indicating {'ongoing dissipation' if la_desc=='non-zero' else 'a quiet loop'}.")
    if xc_desc:
        parts.append(f"X<sub>C</sub> is <strong>{xc_desc}</strong>.")
    comment_html = ("<p>" + " ".join(parts) + "</p>") if parts else ""

    # Mini table columns with fallback: include F_C if X_C absent
    mini_cols_base = ["S_M", "T_L", "loop_area", "U", "dF_C"]
    suppress_xc_numeric = False
    if xc_series is not None and not xc_series.empty:
        try:
            # Suppress numeric table if X_C is deeply negative across the board
            med = float(xc_series.median())
            mad = float((xc_series - med).abs().median()) if xc_series.size >= 8 else float(xc_series.mad()) if hasattr(xc_series, 'mad') else 0.0
            neg95 = float(xc_series.quantile(0.95))
            if neg95 < 0 and med < -(3.0 * mad + 1e-6):
                suppress_xc_numeric = True
        except Exception:
            suppress_xc_numeric = False
    if not suppress_xc_numeric and "X_C" in local.columns and pd.to_numeric(local["X_C"], errors="coerce").dropna().size > 0:
        mini_cols_base.append("X_C")
    elif "F_C" in local.columns and pd.to_numeric(local["F_C"], errors="coerce").dropna().size > 0:
        mini_cols_base.append("F_C")
    mini_cols = [col for col in mini_cols_base if col in local.columns]
    mini_html = ""
    if mini_cols:
        mini_tail = local[["date"] + mini_cols].tail(6).copy()
        mini_tail["date"] = mini_tail["date"].dt.strftime("%Y-%m-%d")
        mini_html = mini_tail.to_html(index=False, border=0, classes="mini", escape=True)

    diagnostics_html = ""
    if has_derivatives and effective_window >= 3 and deriv_cols_present:
        diag_subset = local[["date"] + deriv_cols_present].dropna().tail(6)
        if not diag_subset.empty:
            diag_subset["date"] = diag_subset["date"].dt.strftime("%Y-%m-%d")
            diagnostics_html += f"<h2>Diagnostics – Maxwell-like (window={effective_window})</h2>" + diag_subset.to_html(index=False, border=0, classes="mini", escape=True)
            if out_of_spec_ranges:
                spans = ", ".join([f"{s.strftime('%Y-%m-%d')} → {e.strftime('%Y-%m-%d')}" for s, e in out_of_spec_ranges])
                diagnostics_html += f"<p class=\"note\"><strong>Out-of-spec / crisis / proxy invalid zone</strong>: {html_lib.escape(spans)}</p>"
    elif has_thermo and diag_window:
        diagnostics_html += f"<h2>Diagnostics – Maxwell-like</h2><p class=\"note\">Insufficient data (requested window={diag_window}).</p>"

    firstlaw_table_cols = [c for c in ["dU", "Q_like", "W_like", "dU_pred", "firstlaw_resid"] if c in local.columns]
    if has_thermo and firstlaw_table_cols:
        fl = local[["date"] + firstlaw_table_cols].dropna().tail(6)
        if not fl.empty:
            fl = fl.rename(columns={"W_like": "minus_pV"})
            fl["date"] = fl["date"].dt.strftime("%Y-%m-%d")
            diagnostics_html += "<h2>Diagnostics – First-law</h2>" + fl.to_html(index=False, border=0, classes="mini", escape=True)

    selected_table_html = _selected_table(selected_meta, label)

    # Interpretation notes section (X_C sign)
    interpret_notes = ""
    if xc_series is not None and not xc_series.empty:
        interpret_notes = (
            "<p class=\"note\"><strong>X_C sign interpretation</strong>: above zero suggests some usable potential remains; large negative values imply limited room."
        )
        if suppress_xc_numeric:
            interpret_notes += " Numeric table suppressed for X_C (estimation logic under review)."
        interpret_notes += "</p>"

    # Fold advanced diagnostics by default
    if diagnostics_html:
        diagnostics_html = f"<details><summary>Advanced diagnostics</summary>{diagnostics_html}</details>"

    region_html = (
        f"<section class=\"region-summary\"><h2>{html_lib.escape(label)}</h2>{summary_html}{comment_html}"
        f"<h2>Recent values</h2>{mini_html}{diagnostics_html}{interpret_notes}{selected_table_html}</section>"
        + charts_html
    )

    return {
        "key": key,
        "label": label,
        "html": region_html,
        "fig_specs": fig_specs,
        "summary_line": _selected_summary_line(label, selected_meta),
        "summary_items": summary_items,
        "has_maxwell_fig": any(spec[1] == "Maxwell-like Test" for spec in fig_specs),
        "has_firstlaw_fig": any(spec[1] == "First-law Decomposition" for spec in fig_specs),
        "has_raw_inputs_fig": any(spec[1] == "Raw Inputs (first=100)" for spec in fig_specs),
        "last_date": last_date,
        "frame": local,
        "selected_meta": selected_meta,
    }


def _validated_base_url(raw: str) -> str:
    try:
        parsed = urlparse((raw or "").strip())
    except Exception:
        return DEFAULT_BASE_URL
    if parsed.scheme != "https":
        return DEFAULT_BASE_URL
    host = (parsed.netloc or "").lower()
    allowed = {"toppymicros.com", "toppymicroservices.github.io"}
    if host not in allowed:
        return DEFAULT_BASE_URL
    path = parsed.path.rstrip("/")
    if not path.endswith("/2025_11_Thermo_Credit"):
        path = "/2025_11_Thermo_Credit"
    return f"https://{host}{path}"


def rss_escape(text: str) -> str:
    return text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def _logo_data_uri() -> str:
    # Prefer pre-compressed logo if present.
    candidates = [
        os.path.join(ROOT, "scripts", "og-brand-clean.min.png"),
        os.path.join(ROOT, "scripts", "og-brand-clean.png"),
        os.path.join(ROOT, "og-brand-clean.png"),
    ]
    for p in candidates:
        if os.path.exists(p):
            try:
                with open(p, "rb") as fh:
                    raw = fh.read()
                # If Pillow is available, resize and quantize to reduce size
                if Image is not None:
                    try:
                        im = Image.open(BytesIO(raw)).convert("RGBA")
                        # target height ~80px (header shows 40px; x2 for HiDPI)
                        max_h = 80
                        w, h = im.size
                        if h > max_h and h > 0:
                            new_w = max(1, int(w * max_h / h))
                            im = im.resize((new_w, max_h), Image.LANCZOS)
                        # adaptive palette to 128 colors then save optimized PNG
                        im_q = im.convert("P", palette=Image.ADAPTIVE, colors=128)
                        buf = BytesIO()
                        im_q.save(buf, format="PNG", optimize=True, compress_level=9)
                        data = buf.getvalue()
                    except Exception:
                        data = raw
                else:
                    data = raw
                encoded = base64.b64encode(data).decode("ascii")
                return f"data:image/png;base64,{encoded}"
            except Exception:
                continue
    return ""

def main() -> None:
    os.makedirs(SITE_DIR, exist_ok=True)

    diag_window = _resolve_diag_window()

    jp_df = _load_csv(os.path.join(SITE_DIR, "indicators.csv"))
    eu_df = _load_csv(os.path.join(SITE_DIR, "indicators_eu.csv"))
    us_df = _load_csv(os.path.join(SITE_DIR, "indicators_us.csv"))
    if eu_df is None:
        eu_df = pd.DataFrame()
    if us_df is None:
        us_df = pd.DataFrame()

    selected_meta = _load_json(os.path.join(DATA_DIR, "series_selected.json"))
    eu_selected_meta = _load_json(os.path.join(DATA_DIR, "series_selected_eu.json"))
    us_selected_meta = _load_json(os.path.join(DATA_DIR, "series_selected_us.json"))

    # Prefer module-level preloaded raw_inputs_df if available; otherwise attempt repo data path
    sources_meta = load_sources(os.path.join(DATA_DIR, "sources.json"))
    global raw_inputs_df  # reuse module variable
    if raw_inputs_df is None:
        raw_inputs_df = load_and_normalize(enabled_sources(sources_meta))
        # Fallback: if still None (e.g. tests chdir into temp dir with alternative data set), try CWD-relative sources.json
        if raw_inputs_df is None:
            alt_sources = load_sources(os.path.join("data", "sources.json"))
            if alt_sources:
                raw_inputs_df = load_and_normalize(enabled_sources(alt_sources))
    raw_inputs_fig = _build_raw_inputs_fig(raw_inputs_df)

    regions: List[Dict[str, Any]] = []

    jp_ctx = _build_region_context(
        "jp",
        "Japan (JP)",
        jp_df,
        diag_window=diag_window,
        selected_meta=selected_meta,
        include_raw_inputs=raw_inputs_fig is not None,
        raw_inputs_fig=raw_inputs_fig,
    )
    if jp_ctx:
        regions.append(jp_ctx)

    eu_ctx = _build_region_context(
        "eu",
        "Euro Area (EU)",
        eu_df,
        diag_window=diag_window,
        selected_meta=eu_selected_meta,
        include_raw_inputs=raw_inputs_fig is not None,
        raw_inputs_fig=raw_inputs_fig,
    )
    if eu_ctx:
        regions.append(eu_ctx)

    us_ctx = _build_region_context(
        "us",
        "United States (US)",
        us_df,
        diag_window=diag_window,
        selected_meta=us_selected_meta,
        include_raw_inputs=raw_inputs_fig is not None,
        raw_inputs_fig=raw_inputs_fig,
    )
    if us_ctx:
        regions.append(us_ctx)

    if not regions:
        raise SystemExit("No region data available to render report.")

    primary_ctx = regions[0]
    defs_html = _definitions_table(primary_ctx["frame"])
    # Optional formulas block (rendered via MathJax)
    formulas_html = (
        "<h2>Formulas</h2>"
        "<ul>"
        "<li>Free energy: $F_C = U - T_0\\, S_M$</li>"
        "<li>Change in free energy: $\\Delta F_C(t) = F_C(t) - F_C^{\\mathrm{ref}}$</li>"
        "<li>Surplus/shortage split: $X_C^{+}(t) = \\max(0,\\, \\Delta F_C(t)),\\; X_C^{-}(t) = \\max(0,\\, -\\Delta F_C(t))$</li>"
        "<li>First-law (discrete approximation): $\\Delta U \\approx \\bar T\\, \\Delta S - \\bar p\\, \\Delta V$</li>"
        "<li>Maxwell-like relation (rolling OLS): $\\left. \\partial S / \\partial V \\right|_T \\approx \\left. \\partial p / \\partial T \\right|_V$</li>"
        "</ul>"
    )
    sources_html = _sources_table(sources_meta)

    selected_summary_html = ""
    inputs_summary_html = _build_inputs_summary(regions)

    # Optional: add a Compare tab if at least two regions have frames (even if one is placeholder, charts are gated by data presence)
    compare_ctx = _build_compare_context([ctx for ctx in regions if isinstance(ctx.get("frame"), pd.DataFrame)])
    if compare_ctx and compare_ctx.get("html"):
        regions_with_compare = [compare_ctx] + regions
    else:
        regions_with_compare = regions

    if len(regions_with_compare) > 1:
        buttons: List[str] = []
        region_divs: List[str] = []
        for idx, ctx in enumerate(regions_with_compare):
            active_cls = " active" if idx == 0 else ""
            buttons.append(f"<button class=\"tab{active_cls}\" data-target=\"{ctx['key']}\">{html_lib.escape(ctx['label'])}</button>")
            region_divs.append(f"<div id=\"region-{ctx['key']}\" class=\"region{active_cls}\">{ctx['html']}</div>")
        tabs_html = '<div class="tabs" role="tablist">' + ''.join(buttons) + '</div>'
        regions_html = ''.join(region_divs)
    else:
        tabs_html = ""
        regions_html = regions_with_compare[0]["html"]

    label_to_filename = {
        "S_M & T_L": "fig1.png",
        "Policy Loop Dissipation": "fig2.png",
        "Credit Exergy Ceiling": "fig3.png",
        "Maxwell-like Test": "fig4.png",
        "First-law Decomposition": "fig5.png",
        "Raw Inputs (first=100)": "fig_raw_inputs.png",
    }

    png_fallback_ok = False
    if jp_ctx:
        png_targets: List[Tuple[Any, str]] = []
        for fig, short_label, _ in jp_ctx["fig_specs"]:
            filename = label_to_filename.get(short_label)
            if filename:
                png_targets.append((fig, filename))
        if png_targets:
            try:
                for fig, filename in png_targets:
                    fig.write_image(os.path.join(SITE_DIR, filename), scale=2, width=1280, height=720)
                png_fallback_ok = True
            except Exception as exc:
                print("PNG export skipped:", exc)

    extra_png = ""
    if png_fallback_ok and jp_ctx:
        if jp_ctx.get("has_maxwell_fig") and os.path.exists(os.path.join(SITE_DIR, "fig4.png")):
            extra_png += '<figure><img src="fig4.png" alt="Maxwell-like" width="100%"/><figcaption>Maxwell-like</figcaption></figure>'
        if jp_ctx.get("has_firstlaw_fig") and os.path.exists(os.path.join(SITE_DIR, "fig5.png")):
            extra_png += '<figure><img src="fig5.png" alt="First-law" width="100%"/><figcaption>First-law</figcaption></figure>'
        if jp_ctx.get("has_raw_inputs_fig") and os.path.exists(os.path.join(SITE_DIR, "fig_raw_inputs.png")):
            extra_png += '<figure><img src="fig_raw_inputs.png" alt="Raw inputs" width="100%"/><figcaption>Raw Inputs</figcaption></figure>'
    if png_fallback_ok:
        noscript = ("<noscript><h2>Static Images</h2>"
                    "<figure><img src='fig1.png' alt='S_M & T_L' width='100%'/><figcaption>S_M & T_L</figcaption></figure>"
                    "<figure><img src='fig2.png' alt='Loop area' width='100%'/><figcaption>Loop area</figcaption></figure>"
                    "<figure><img src='fig3.png' alt='X_C' width='100%'/><figcaption>X_C</figcaption></figure>"
                    + extra_png + "</noscript>")
    else:
        noscript = "<noscript><p>No static images this run.</p></noscript>"

    logo_uri = _logo_data_uri()
    # Brand colors can be overridden by env vars
    BRAND_BG = os.getenv("BRAND_BG", "#0d1b2a")
    BRAND_BG2 = os.getenv("BRAND_BG2", "#1b263b")
    BRAND_TEXT = os.getenv("BRAND_TEXT", "#ffffff")

    style_block = (
        "body{font-family:system-ui,-apple-system,Segoe UI,Roboto,Helvetica,Arial,sans-serif;line-height:1.5;margin:1.25rem;background:#f6f8fb}"
        "h1{font-size:1.6rem;margin:0 0 .5rem}h2{font-size:1.1rem;margin:1.25rem 0 .5rem}.wrap{max-width:1100px;margin:0 auto}"
        ".note{color:#333;margin:.5rem 0 1rem}.note.small{font-size:.85rem;color:#666}figure{margin:1rem 0}figcaption{font-size:.8rem;color:#555}"
        ".region-summary{background:#fff;border:1px solid #eee;border-radius:8px;padding:.85rem 1rem}"
        "table.mini{border-collapse:collapse;margin:.5rem 0}table.mini td,table.mini th{padding:.25rem .5rem;border-bottom:1px solid:#ddd;text-align:right}table.mini th:first-child,table.mini td:first-child{text-align:left}"
        ".tabs{display:flex;gap:.5rem;margin:.75rem 0 1rem}.tabs button{border:1px solid #888;background:#f8f8f8;padding:.4rem .75rem;cursor:pointer;font-size:.8rem;border-radius:4px}.tabs button.active{background:#333;color:#fff}"
        ".subtabs{display:flex;gap:.4rem;margin:.5rem 0 .75rem}.subtabs button{border:1px solid #aaa;background:#f6f7f9;padding:.3rem .6rem;font-size:.78rem;border-radius:999px;cursor:pointer}.subtabs button.active{background:#333;color:#fff;border-color:#333}"
        ".compare-block .pane{display:none}.compare-block .pane.active{display:block}"
        ".region{display:none}.region.active{display:block}"
        ".intro{background:#eef2f7;border:1px solid #dde4ee;padding:.85rem 1rem;border-radius:8px;margin:1rem 0}"
        ".intro ul{margin:.5rem 0 .75rem;padding-left:1.1rem}"
        ".intro li{margin:.3rem 0}"
        "details{margin:.5rem 0}details>summary{cursor:pointer;list-style:none;font-weight:600}details>summary::-webkit-details-marker{display:none}"
        ".inputs-summary{background:#fafafa;border:1px solid #eee;padding:.75rem;border-radius:6px;margin:.75rem 0 1rem}"
        ".inputs-summary .inputs-row{margin:.35rem 0}.inputs-summary .region-tag{display:inline-block;background:#333;color:#fff;border-radius:3px;padding:.15rem .4rem;font-size:.75rem;margin-right:.4rem}"
        ".inputs-summary .pill-list{display:inline}.inputs-summary .pill{display:inline-block;border:1px solid #ddd;background:#fff;border-radius:999px;padding:.15rem .5rem;margin:.15rem .25rem;font-size:.75rem}"
        + f":root{{--brand-bg:{BRAND_BG};--brand-bg2:{BRAND_BG2};--brand-text:{BRAND_TEXT};}}"
        ".brandbar{display:flex;align-items:center;gap:10px;margin-bottom:1rem;padding:.5rem .75rem;border-radius:8px;background:linear-gradient(90deg,var(--brand-bg),var(--brand-bg2));color:var(--brand-text)}"
        ".brandbar img{height:40px;width:auto;border-radius:6px;box-shadow:0 0 0 1px rgba(255,255,255,.2)}"
        ".brandbar .brand-name{font-weight:600;font-size:1rem;color:var(--brand-text)}"
        ".footer-brand{margin-top:2rem;padding:.75rem;border-top:none;border-radius:8px;background:linear-gradient(90deg,var(--brand-bg),var(--brand-bg2));font-size:.75rem;color:var(--brand-text);display:flex;align-items:center;gap:10px}"
        ".footer-brand img{height:32px;width:auto;border-radius:6px;box-shadow:0 0 0 1px rgba(255,255,255,.2)}"
    )

    head = ("<!doctype html><html lang=\"en\"><head><meta charset=\"utf-8\"><meta name=\"viewport\" "
            "content=\"width=device-width,initial-scale=1\"><title>Thermo-Credit Monitor</title><meta name=\"description\" "
            "content=\"Monthly thermo-credit indicators.\"><style>" + style_block + "</style>"
            + "</head><body><div class=\"wrap\"><div class=\"brandbar\">"
            + (f'<img src="{logo_uri}" alt="Company Logo"/>' if logo_uri else "")
            + '<span class="brand-name">ToppyMicroServices</span></div><h1>Thermo-Credit Monitor</h1><p class="note">Interactive charts with summary & fallbacks.</p>')

    intro_html = (
        '<section class="intro">'
        '<h2>What this page shows</h2>'
        '<p>This dashboard tracks monthly thermo-credit indicators for Japan, the Euro Area, and the US. '
        'It is meant to answer very simple questions:</p>'
        '<ul>'
        '<li>Is credit currently <strong>tight or loose</strong> in each region?</li>'
        '<li>How much <strong>room is left</strong> for non-disruptive adjustment?</li>'
        '<li>Where do we see signs of <strong>stress or overheating</strong> in the loop?</li>'
        '</ul>'
        '<p>Under the hood, the framework uses four core metrics:</p>'
        '<ul>'
        '<li><strong>S<sub>M</sub></strong> – dispersion of money/credit (size × allocation spread)</li>'
        '<li><strong>T<sub>L</sub></strong> – liquidity “temperature” (funding &amp; market conditions)</li>'
        '<li><strong>Loop area</strong> – dissipation along the policy/regulatory loop</li>'
        '<li><strong>X<sub>C</sub></strong> – remaining “credit exergy”, i.e. safe room to adjust</li>'
        '</ul>'
        '<p>Values here are <strong>experimental</strong> and follow the Thermo-Credit v0.x spec. '
        'They are for research and discussion, not for trading or regulatory use.</p>'
        '</section>'
    )

    page_body = intro_html + selected_summary_html + inputs_summary_html + tabs_html + regions_html + noscript + sources_html + defs_html + formulas_html
    script_block = ("\n<script>(function(){const tabs=[...document.querySelectorAll('.tabs button')];if(tabs.length){"
                    "tabs.forEach(btn=>btn.addEventListener('click',()=>{tabs.forEach(x=>x.classList.remove('active'));btn.classList.add('active');"
                    "const tgt=btn.getAttribute('data-target');document.querySelectorAll('.region').forEach(r=>r.classList.remove('active'));"
                    "const el=document.getElementById('region-'+tgt);if(el)el.classList.add('active');}));}"
                    "document.querySelectorAll('.compare-toggle').forEach(ct=>{const btns=[...ct.querySelectorAll('button')];const block=ct.parentElement.nextElementSibling;"
                    "btns.forEach(btn=>btn.addEventListener('click',()=>{btns.forEach(x=>x.classList.remove('active'));btn.classList.add('active');const mode=btn.getAttribute('data-mode');"
                    "if(block){block.querySelectorAll('.pane').forEach(p=>p.classList.remove('active'));const target=block.querySelector('.pane.'+(mode==='std'?'std':'raw'));if(target)target.classList.add('active');}"
                    "}));});})();</script></body></html>")

    final_html = head + page_body + '<div class="footer-brand">' + (f'<img src="{logo_uri}" alt="Company Logo"/>' if logo_uri else "") + '<span>© ' + datetime.utcnow().strftime('%Y') + ' ToppyMicroServices</span></div></div>' + script_block
    with open(os.path.join(SITE_DIR, "report.html"), "w", encoding="utf-8") as fp:
        fp.write(final_html)
    print("Wrote site/report.html")

    base_url = _validated_base_url(os.getenv("TMS_BASE_URL", DEFAULT_BASE_URL))
    month_key = primary_ctx["last_date"].strftime("%Y-%m")
    month_dir = os.path.join(SITE_DIR, month_key)
    os.makedirs(month_dir, exist_ok=True)

    if png_fallback_ok:
        for filename in label_to_filename.values():
            src = os.path.join(SITE_DIR, filename)
            if os.path.exists(src):
                try:
                    shutil.copyfile(src, os.path.join(month_dir, filename))
                except Exception:
                    pass

    month_head = ("<!doctype html><html lang=\"en\"><head><meta charset=\"utf-8\"><meta name=\"viewport\" "
                  f"content=\"width=device-width,initial-scale=1\"><title>Thermo-Credit Monitor – {month_key}</title><meta name=\"description\" "
                  "content=\"Monthly thermo-credit indicators.\"><style>" + style_block + "</style>"
                  + "</head><body><div class=\"wrap\"><div class=\"brandbar\">"
                  + (f'<img src="{logo_uri}" alt="Company Logo"/>' if logo_uri else "")
                  + '<span class="brand-name">ToppyMicroServices</span></div><h1>Thermo-Credit Monitor</h1><p class="note">Interactive charts with summary & fallbacks.</p>')
    month_html = month_head + page_body + '<div class="footer-brand">' + (f'<img src="{logo_uri}" alt="Company Logo"/>' if logo_uri else "") + '<span>© ' + datetime.utcnow().strftime('%Y') + ' ToppyMicroServices</span></div></div>' + script_block
    with open(os.path.join(month_dir, "index.html"), "w", encoding="utf-8") as fp:
        fp.write(month_html)

    archive_path = os.path.join(SITE_DIR, "archive.json")
    try:
        archive = json.load(open(archive_path, "r", encoding="utf-8")) if os.path.exists(archive_path) else []
    except Exception:
        archive = []
    if not isinstance(archive, list):
        archive = []

    entry = {
        "month": month_key,
        "url": f"{base_url}/{month_key}/",
        "lastmod": primary_ctx["last_date"].strftime("%Y-%m-%d"),
        "title": f"Thermo-Credit Monitor {month_key}",
        "summary": primary_ctx["summary_items"],
    }

    archive_by_month = {e.get("month"): e for e in archive if isinstance(e, dict)}
    archive_by_month[month_key] = entry
    archive = sorted(archive_by_month.values(), key=lambda e: e.get("month", ""), reverse=True)
    with open(archive_path, "w", encoding="utf-8") as fp:
        json.dump(archive, fp, ensure_ascii=False, indent=2)

    rss_items: List[str] = []
    for item in archive[:24]:
        try:
            pub = datetime.strptime(item["month"] + "-01", "%Y-%m-%d")
        except Exception:
            continue
        pub_rfc822 = pub.strftime("%a, %d %b %Y 00:00:00 +0000")
        summary_text = " – ".join(map(str, item.get("summary", [])[:2]))
        rss_items.append(
            f"<item><title>{rss_escape(item['title'])}</title><link>{rss_escape(item['url'])}</link><guid>{rss_escape(item['url'])}</guid><pubDate>{rss_escape(pub_rfc822)}</pubDate><description>{rss_escape(summary_text)}</description></item>"
        )

    rss_xml = ("<?xml version='1.0' encoding='UTF-8'?><rss version='2.0'><channel><title>Thermo-Credit Monitor</title>"
               f"<link>{base_url}/</link><description>Monthly thermo-credit indicators: S_M, T_L, loop dissipation, X_C.</description>"
               "<language>en</language>" + ''.join(rss_items) + "</channel></rss>")
    with open(os.path.join(SITE_DIR, "feed.xml"), "w", encoding="utf-8") as fp:
        fp.write(rss_xml)

    urls = [f"{base_url}/", f"{base_url}/report.html", f"{base_url}/feed.xml"] + [f"{base_url}/{item['month']}/" for item in archive]
    today = datetime.utcnow().strftime("%Y-%m-%d")
    urlset = ''.join(f"<url><loc>{rss_escape(u)}</loc><lastmod>{today}</lastmod></url>" for u in urls)
    sitemap_xml = f"<?xml version='1.0' encoding='UTF-8'?><urlset xmlns='http://www.sitemaps.org/schemas/sitemap/0.9'>{urlset}</urlset>"
    with open(os.path.join(SITE_DIR, "sitemap.xml"), "w", encoding="utf-8") as fp:
        fp.write(sitemap_xml)
    with open(os.path.join(SITE_DIR, "robots.txt"), "w", encoding="utf-8") as fp:
        fp.write(f"User-agent: *\nAllow: /\nSitemap: {base_url}/sitemap.xml\n")
    print("Wrote monthly archive, feed.xml, sitemap.xml, and robots.txt")


if __name__ == "__main__":
    main()
