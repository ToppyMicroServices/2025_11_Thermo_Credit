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


def _apply_hover(fig, fmt: str) -> None:
    fig.update_traces(hovertemplate="%{x|%Y-%m-%d}<br>%{y:" + fmt + "}<extra>%{fullData.name}</extra>")


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

    figs: List[Tuple[Any, str, str]] = []
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
        fig = px.line(long_df, x="date", y="value", color="Region", title=title, render_mode="svg")
        _style_figure(fig)
        _apply_hover(fig, ".3f")
        figs.append((fig, title.replace("Compare – ", "Compare: "), alt))

    charts_html = _figs_html(figs)

    summary_html = ""
    if latest_rows:
        latest_df = pd.DataFrame(latest_rows)
        # Order columns nicely
        cols = [c for c in ["Region", "Latest date", "S_M", "T_L", "loop_area", "X_C"] if c in latest_df.columns]
        latest_df = latest_df[cols]
        summary_html = "<h2>Compare – Latest snapshot</h2>" + latest_df.to_html(index=False, border=0, classes="mini", float_format=lambda x: f"{x:.4g}")

    region_html = (
        f"<section class=\"region-summary\"><h2>Compare (JP/EU/US)</h2>{summary_html}</section>" + charts_html
    )

    return {
        "key": "compare",
        "label": "Compare",
        "html": region_html,
        "fig_specs": figs,
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
        ("money_scale", "マネースケール"),
        ("base_proxy", "ベース"),
        ("yield_proxy", "長期金利"),
    ]
    parts: List[str] = []
    for key, label in roles:
        ent = pick([key])
        if ent:
            title = ent.get("title") or ent.get("id", "")
            start = ent.get("start") or ""
            start_y = start[:4] if isinstance(start, str) and len(start) >= 4 else ""
            tail = f" (開始 {start_y})" if start_y else ""
            parts.append(f"{label}: {html_lib.escape(title)}{tail}")
    if not parts:
        return None
    return f"{html_lib.escape(prefix)} — " + " ・ ".join(parts)


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
    return "<h2>Sources</h2>" + table


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
    fig = px.line(long_df, x="date", y="Value", color="Series", title="Raw Inputs (normalized first=100)", color_discrete_map=discrete_map, render_mode="svg")
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
    has_thermo = all(c in local.columns for c in REQUIRED_THERMO_COLS)
    effective_window, eff_note = _calc_effective_window(local, diag_window)
    local, has_derivatives = _augment_region_frame(local, effective_window, has_thermo)
    # Plot subset filtered by start date
    plot_start = _plot_start_date()
    plot_df = local[local["date"] >= plot_start].copy() if "date" in local.columns else local.copy()

    fig_specs: List[Tuple[Any, str, str]] = []
    if {"S_M", "T_L"}.issubset(local.columns) and not plot_df.empty:
        fig = px.line(plot_df, x="date", y=["S_M", "T_L"], title=f"{label} – S_M & T_L")
        _style_figure(fig)
        _apply_hover(fig, ".3f")
        fig_specs.append((fig, "S_M & T_L", "Entropy & temperature"))
    if "loop_area" in local.columns and not plot_df.empty:
        fig = px.line(plot_df, x="date", y="loop_area", title=f"{label} – Policy Loop Dissipation")
        _style_figure(fig)
        _apply_hover(fig, ".3f")
        fig_specs.append((fig, "Policy Loop Dissipation", "Loop area"))
    # Exergy / free-energy figure: use X_C if available (non-all-NaN), else fall back to F_C so US still gets a plot
    if not plot_df.empty:
        x_col = None
        if "X_C" in plot_df.columns and pd.to_numeric(plot_df["X_C"], errors="coerce").dropna().size > 0:
            x_col = "X_C"
            title_x = "Credit Exergy Ceiling"
        elif "F_C" in plot_df.columns and pd.to_numeric(plot_df["F_C"], errors="coerce").dropna().size > 0:
            x_col = "F_C"
            title_x = "Free Energy (F_C)"
        if x_col is not None:
            fig = px.line(plot_df, x="date", y=x_col, title=f"{label} – {title_x}")
            _style_figure(fig)
            _apply_hover(fig, ".3f")
            fig_specs.append((fig, title_x, x_col))

    deriv_cols_present = [c for c in DERIVATIVE_COLS if c in local.columns]
    if has_derivatives and effective_window >= 3 and deriv_cols_present and not plot_df.empty:
        title = f"{label} – Maxwell-like Relation"
        if eff_note:
            title += eff_note
        fig = px.line(plot_df, x="date", y=deriv_cols_present, title=title, markers=True)
        _style_figure(fig)
        _apply_hover(fig, ".3f")
        fig_specs.append((fig, "Maxwell-like Test", "Derivatives"))
    firstlaw_cols = [c for c in ["dU", "dU_pred", "firstlaw_resid"] if c in local.columns]
    if has_thermo and firstlaw_cols and not plot_df.empty:
        fig = px.line(plot_df, x="date", y=firstlaw_cols, title=f"{label} – First-law Decomposition", markers=True)
        _style_figure(fig)
        _apply_hover(fig, ".3f")
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
    # Summary: show X_C if present; otherwise F_C label it accordingly
    if "X_C" in local.columns and pd.to_numeric(local["X_C"], errors="coerce").dropna().size > 0:
        summary_items.append(f"X_C: {fmt(last_row.get('X_C'))}")
    elif "F_C" in local.columns and pd.to_numeric(local["F_C"], errors="coerce").dropna().size > 0:
        summary_items.append(f"F_C: {fmt(last_row.get('F_C'))}")
    if has_derivatives and "maxwell_gap" in local.columns:
        summary_items.append(f"Maxwell gap: {fmt(last_row.get('maxwell_gap'))}")
    if has_thermo and "firstlaw_resid" in local.columns:
        summary_items.append(f"First-law resid: {fmt(last_row.get('firstlaw_resid'))}")
    summary_html = "<ul>" + "".join(f"<li>{html_lib.escape(item)}</li>" for item in summary_items) + "</ul>"

    # Mini table columns with fallback: include F_C if X_C absent
    mini_cols_base = ["S_M", "T_L", "loop_area"]
    if "X_C" in local.columns and pd.to_numeric(local["X_C"], errors="coerce").dropna().size > 0:
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

    region_html = (
        f"<section class=\"region-summary\"><h2>{html_lib.escape(label)}</h2>{summary_html}"
        f"<h2>Recent values</h2>{mini_html}{diagnostics_html}{selected_table_html}</section>"
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

    head = ("<!doctype html><html lang=\"en\"><head><meta charset=\"utf-8\"><meta name=\"viewport\" "
        "content=\"width=device-width,initial-scale=1\"><title>Thermo-Credit Monitor</title><meta name=\"description\" "
    "content=\"Monthly thermo-credit indicators.\"><style>"
    # Base
    "body{font-family:system-ui,-apple-system,Segoe UI,Roboto,Helvetica,Arial,sans-serif;line-height:1.5;margin:1.25rem}"
    "h1{font-size:1.6rem;margin:0 0 .5rem}h2{font-size:1.1rem;margin:1.25rem 0 .5rem}.wrap{max-width:1100px;margin:0 auto}"
    ".note{color:#333;margin:.5rem 0 1rem}figure{margin:1rem 0}figcaption{font-size:.8rem;color:#555}"
    "table.mini{border-collapse:collapse;margin:.5rem 0}table.mini td,table.mini th{padding:.25rem .5rem;border-bottom:1px solid #ddd;text-align:right}table.mini th:first-child,table.mini td:first-child{text-align:left}"
    ".tabs{display:flex;gap:.5rem;margin:.75rem 0 1rem}.tabs button{border:1px solid #888;background:#f8f8f8;padding:.4rem .75rem;cursor:pointer;font-size:.8rem;border-radius:4px}.tabs button.active{background:#333;color:#fff}"
    ".region{display:none}.region.active{display:block}"
    # Inputs summary
    ".inputs-summary{background:#fafafa;border:1px solid #eee;padding:.75rem;border-radius:6px;margin:.75rem 0 1rem}"
    ".inputs-summary .inputs-row{margin:.35rem 0}.inputs-summary .region-tag{display:inline-block;background:#333;color:#fff;border-radius:3px;padding:.15rem .4rem;font-size:.75rem;margin-right:.4rem}"
    ".inputs-summary .pill-list{display:inline}.inputs-summary .pill{display:inline-block;border:1px solid #ddd;background:#fff;border-radius:999px;padding:.15rem .5rem;margin:.15rem .25rem;font-size:.75rem}"
    # Brand colors via CSS variables
    + f":root{{--brand-bg:{BRAND_BG};--brand-bg2:{BRAND_BG2};--brand-text:{BRAND_TEXT};}}"
    # Brandbar (header)
    ".brandbar{display:flex;align-items:center;gap:10px;margin-bottom:1rem;padding:.5rem .75rem;border-radius:8px;background:linear-gradient(90deg,var(--brand-bg),var(--brand-bg2));color:var(--brand-text)}"
    ".brandbar img{height:40px;width:auto;border-radius:6px;box-shadow:0 0 0 1px rgba(255,255,255,.2)}"
    ".brandbar .brand-name{font-weight:600;font-size:1rem;color:var(--brand-text)}"
    # Footer brand
    ".footer-brand{margin-top:2rem;padding:.75rem;border-top:none;border-radius:8px;background:linear-gradient(90deg,var(--brand-bg),var(--brand-bg2));font-size:.75rem;color:var(--brand-text);display:flex;align-items:center;gap:10px}"
    ".footer-brand img{height:32px;width:auto;border-radius:6px;box-shadow:0 0 0 1px rgba(255,255,255,.2)}"
    "</style></head><body><div class=\"wrap\"><div class=\"brandbar\">"
    + (f'<img src="{logo_uri}" alt="Company Logo"/>' if logo_uri else "")
    + '<span class="brand-name">ToppyMicroServices</span></div><h1>Thermo-Credit Monitor</h1><p class="note">Interactive charts with summary & fallbacks.</p>')

    page_body = selected_summary_html + inputs_summary_html + tabs_html + regions_html + noscript + sources_html + defs_html
    script_block = ("\n<script>(function(){const b=[...document.querySelectorAll('.tabs button')];if(!b.length)return;"
                    "b.forEach(btn=>btn.addEventListener('click',()=>{b.forEach(x=>x.classList.remove('active'));btn.classList.add('active');"
                    "const tgt=btn.getAttribute('data-target');document.querySelectorAll('.region').forEach(r=>r.classList.remove('active'));"
                    "const el=document.getElementById('region-'+tgt);if(el)el.classList.add('active');}));})();</script></body></html>")

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

    month_head = head.replace("<title>Thermo-Credit Monitor</title>", f"<title>Thermo-Credit Monitor – {month_key}</title>")
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
