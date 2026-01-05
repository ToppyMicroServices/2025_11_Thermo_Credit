import base64
import html as html_lib
import json
import os
import shutil
import sys
from datetime import datetime
from io import BytesIO
from typing import Any
from urllib.parse import urlparse

import numpy as np
import pandas as pd
import plotly.express as px

try:
    from PIL import Image
except Exception:
    Image = None

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import contextlib

from lib.raw_inputs import enabled_sources, load_and_normalize, load_sources
from lib.report_helpers import (
    CATEGORY_LABELS,
    DERIVATIVE_COLS,
    REQUIRED_THERMO_COLS,
    ChartSpec,
    CompareBuilder,
    _apply_hover,
    _augment_region_frame,
    _calc_effective_window,
    _chart_interpretation,
    _figs_html,
    _filter_placeholders,
    _load_csv,
    _load_json,
    _mask_to_ranges,
    _out_of_spec_mask,
    _plot_start_date,
    _resolve_diag_window,
    _selected_table,
    _series_bucket,
    _series_trend,
    _style_figure,
    make_dual_axis_sm_tl,
)

SITE_DIR = os.path.join(ROOT, "site")
DATA_DIR = os.path.join(ROOT, "data")
DEFAULT_BASE_URL = "https://toppymicros.com/2025_11_Thermo_Credit"
raw_inputs_df: pd.DataFrame | None = None

# Try preloading normalized raw inputs so tests can assert against module state.
try:
    for candidate in (os.path.join("data", "sources.json"), os.path.join(DATA_DIR, "sources.json")):
        if not os.path.exists(candidate):
            continue
        sources_cfg = load_sources(candidate)
        if not sources_cfg:
            continue
        maybe_df = load_and_normalize(enabled_sources(sources_cfg))
        if maybe_df is not None:
            raw_inputs_df = maybe_df
            break
except Exception as exc:
    print("[report] raw_inputs preload failed:", exc)


def _build_compare_context(region_ctxs: list[dict[str, Any]]) -> dict[str, Any] | None:
    builder = CompareBuilder(region_ctxs)
    compare_data = builder.build()
    if compare_data is None:
        return None

    summary_html = ""
    if not compare_data.latest_rows.empty:
        latest_df = compare_data.latest_rows.copy()
        cols = [c for c in ["Region", "Latest date", "S_M", "T_L", "loop_area", "X_C"] if c in latest_df.columns]
        if cols:
            latest_df = latest_df[cols]
        try:
            dates = pd.to_datetime(latest_df.get("Latest date"), errors="coerce").dropna()
            latest_dt_str = dates.max().strftime("%Y-%m-%d") if not dates.empty else ""
        except Exception:
            latest_dt_str = ""
        headline = (
            f"<p><strong>At the latest date</strong>{' (' + latest_dt_str + ')' if latest_dt_str else ''}, this section compares dispersion (S<sub>M</sub>), liquidity temperature (T<sub>L</sub>), loop dissipation, and remaining credit exergy (X<sub>C</sub>) across regions. The table below gives exact values.</p>"
        )
        summary_html = headline + "<h2>Compare – Latest snapshot</h2>" + latest_df.to_html(index=False, border=0, classes="mini", float_format=lambda x: f"{x:.4g}")

    raw_charts_html = _figs_html(compare_data.raw_figs)
    std_charts_html = _figs_html(compare_data.std_figs) if compare_data.std_figs else ""

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
        "fig_specs": compare_data.raw_figs + compare_data.std_figs,
        "summary_line": None,
        "summary_items": [],
        "has_maxwell_fig": False,
        "has_firstlaw_fig": False,
        "has_raw_inputs_fig": False,
        "last_date": max((pd.to_datetime(ctx.get("last_date")) for ctx in region_ctxs if ctx.get("last_date")), default=datetime.utcnow()),
        "frame": pd.DataFrame(),
    }


def _selected_summary_line(prefix: str, meta: dict[str, Any] | None) -> str | None:
    if not isinstance(meta, dict):
        return None
    pieces: list[str] = []
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


def _build_inputs_summary(region_ctxs: list[dict[str, Any]]) -> str:
    rows: list[str] = []
    for ctx in region_ctxs:
        label = ctx.get("label", "")
        meta = ctx.get("selected_meta")
        if not isinstance(meta, dict) or not meta:
            continue
        pills: list[str] = []
        for role, entry in meta.items():
            if not isinstance(entry, dict):
                continue
            title = entry.get("title") or entry.get("id", "")
            provider = entry.get("provider") or entry.get("source") or ""
            start = entry.get("start") or ""
            start_y = start[:4] if isinstance(start, str) and len(start) >= 4 else ""
            parts: list[str] = [f"<strong>{html_lib.escape(_role_label(role))}</strong>: {html_lib.escape(title)}"]
            tail: list[str] = []
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


def _selected_summary_sentence(prefix: str, meta: dict[str, Any] | None) -> str | None:
    if not isinstance(meta, dict) or not meta:
        return None
    def pick(keys: list[str]) -> dict[str, Any] | None:
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
    parts: list[str] = []
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


def _sources_table(sources_meta: list[dict[str, Any]]) -> str:
    rows: list[dict[str, Any]] = []
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


def _build_raw_inputs_fig(raw_df: pd.DataFrame | None):
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
    frame: pd.DataFrame | None,
    *,
    diag_window: int,
    selected_meta: dict[str, Any] | None = None,
    include_raw_inputs: bool = False,
    raw_inputs_fig=None,
) -> dict[str, Any] | None:
    if frame is None:
        return None
    local = frame.copy()
    def _empty_context() -> dict[str, Any]:
        summary_items = ["No indicator data available yet."]
        # Note: plot start date not used when no data available
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

    fig_specs: list[ChartSpec] = []
    if {"S_M", "T_L"}.issubset(local.columns) and not plot_df.empty:
        # Dual-axis layout for very different scales
        fig = make_dual_axis_sm_tl(plot_df, title=f"{label} – S_M & T_L")
        _style_figure(fig)
        _apply_hover(fig, ".3f")
        interp = _chart_interpretation("S_M & T_L", plot_df)
        fig_specs.append((fig, "S_M & T_L", "Entropy & temperature", interp))
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
            interp = _chart_interpretation("S_M by category", plot_df)
            fig_specs.append((fig_cat, "S_M by category", "Entropy by MECE categories", interp))
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
    interp = _chart_interpretation("Policy Loop Dissipation", plot_df)
    fig_specs.append((fig, "Policy Loop Dissipation", "Loop area", interp))
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
            interp = _chart_interpretation("Credit Exergy Ceiling", plot_df)
            fig_specs.append((fig_xc, "Credit Exergy Ceiling", "X_C", interp))
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
            interp = _chart_interpretation("Free Energy (F_C)", plot_df)
            fig_specs.append((fig_fc, "Free Energy (F_C)", "F_C", interp))
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
            interp = _chart_interpretation("ΔF_C (change)", plot_df)
            fig_specs.append((fig_dfc, "ΔF_C (change)", "dF_C", interp))
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
            interp = _chart_interpretation("Internal Energy (U)", plot_df)
            fig_specs.append((fig_u, "Internal Energy (U)", "U", interp))
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
                interp = _chart_interpretation("Surplus/Shortage (ΔF_C)", df_pm)
                fig_specs.append((fig_pm, "Surplus/Shortage (ΔF_C)", "X_C_plus / X_C_minus", interp))

    deriv_cols_present = [c for c in DERIVATIVE_COLS if c in local.columns]
    out_of_spec_ranges: list[tuple[pd.Timestamp, pd.Timestamp]] = []
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
    interp = _chart_interpretation("Maxwell-like Test", plot_df)
    fig_specs.append((fig, "Maxwell-like Test", "Derivatives", interp))
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
        interp = _chart_interpretation("First-law Decomposition", plot_df)
        fig_specs.append((fig, "First-law Decomposition", "ΔU vs predicted", interp))
    if include_raw_inputs and raw_inputs_fig is not None:
        interp = _chart_interpretation("Raw Inputs (first=100)", None)
        fig_specs.append((raw_inputs_fig, "Raw Inputs (first=100)", "Normalized raw inputs", interp))

    charts_html = _figs_html(fig_specs)

    last_row = local.iloc[-1]
    last_ts = pd.to_datetime(last_row.get("date"), errors="coerce")
    last_date = last_ts.to_pydatetime() if not pd.isna(last_ts) else datetime.utcnow()
    def fmt(v):
        return f"{float(v):.4g}" if pd.notna(v) else "n/a"
    summary_items: list[str] = []
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

    sm_bucket = _series_bucket(local.get("S_M"), last_sm) if "S_M" in local.columns else None
    tl_bucket = _series_bucket(local.get("T_L"), last_tl) if "T_L" in local.columns else None
    la_desc = None
    if last_la is not None and np.isfinite(last_la):
        la_desc = "non-zero" if abs(last_la) > 1e-12 else "near zero"
    xc_desc = None
    if last_xc is not None and np.isfinite(last_xc):
        if last_xc <= 1e-9:
            xc_desc = "≈0 (limited remaining room)"
        else:
            xc_desc = "positive (some room remains)"

    parts: list[str] = []
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

    chart_lines: list[tuple[str, str]] = []
    if "S_M" in local.columns or "T_L" in local.columns:
        msg_parts: list[str] = []
        if "S_M" in local.columns and last_sm is not None:
            sm_desc = sm_bucket or f"{fmt(last_sm)}"
            msg_parts.append(f"S_M is {sm_desc}")
        if "T_L" in local.columns and last_tl is not None:
            tl_desc = tl_bucket or f"{fmt(last_tl)}"
            msg_parts.append(f"T_L is {tl_desc}")
        if msg_parts:
            chart_lines.append(("S_M & T_L", ", ".join(msg_parts) + f" as of {last_date.strftime('%Y-%m-%d')}"))
    if "loop_area" in local.columns and last_la is not None:
        loop_trend = _series_trend(local.get("loop_area"))
        trend_txt = f" and {loop_trend}" if loop_trend else ""
        chart_lines.append(("Policy Loop Dissipation", f"Loop area is {la_desc or fmt(last_la)}{trend_txt}."))
    if last_xc is not None:
        xc_trend = _series_trend(xc_series) if xc_series is not None else None
        xc_text = xc_desc or f"{fmt(last_xc)}"
        suffix = f" and {xc_trend}" if xc_trend else ""
        chart_lines.append(("Credit Exergy Ceiling", f"X_C is {xc_text}{suffix}."))
    if has_derivatives and "maxwell_gap" in local.columns:
        gap_desc = fmt(last_row.get("maxwell_gap"))
        spec = "alerts active" if out_of_spec_ranges else "inside spec"
        chart_lines.append(("Maxwell-like Test", f"Gap is {gap_desc} ({spec})."))
    if has_thermo and "firstlaw_resid" in local.columns:
        resid_desc = fmt(last_row.get("firstlaw_resid"))
        chart_lines.append(("First-law Decomposition", f"Residual is {resid_desc} (ΔU minus predicted)."))

    chart_notes_html = ""
    if chart_lines:
        items = "".join(
            f"<div class=\"chart-note\"><strong>{html_lib.escape(title)}</strong><span>{html_lib.escape(text)}</span></div>"
            for title, text in chart_lines
        )
        chart_notes_html = f"<div class=\"chart-notes\"><h3>Interpretation</h3>{items}</div>"

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
        f"<section class=\"region-summary\"><h2>{html_lib.escape(label)}</h2>{summary_html}{comment_html}{chart_notes_html}"
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

    # Prefer deterministic JP path if present, fallback to legacy dump when empty
    jp_df = _load_csv(os.path.join(SITE_DIR, "indicators_jp.csv"))
    if jp_df is None or (isinstance(jp_df, pd.DataFrame) and jp_df.empty):
        legacy_jp = _load_csv(os.path.join(SITE_DIR, "indicators.csv"))
        if isinstance(legacy_jp, pd.DataFrame) and not legacy_jp.empty:
            jp_df = legacy_jp
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

    regions: list[dict[str, Any]] = []

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
        buttons: list[str] = []
        region_divs: list[str] = []
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
        png_targets: list[tuple[Any, str]] = []
        for fig, short_label, _, _ in jp_ctx["fig_specs"]:
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
    ".chart-notes{background:#f1f4fb;border:1px solid #dce3f1;border-radius:6px;padding:.4rem .7rem;margin:.8rem 0}"
    ".chart-note{display:flex;flex-direction:column;margin:.2rem 0;font-size:.82rem}"
    ".chart-note strong{font-weight:600;color:#1b2a43}"
    ".chart-note span{color:#333;font-size:.78rem}"
    ".chart-note-inline{display:block;font-size:.78rem;color:#444;margin-top:.2rem}"
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
                with contextlib.suppress(Exception):
                    shutil.copyfile(src, os.path.join(month_dir, filename))

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
        archive = json.load(open(archive_path, encoding="utf-8")) if os.path.exists(archive_path) else []
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

    rss_items: list[str] = []
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
