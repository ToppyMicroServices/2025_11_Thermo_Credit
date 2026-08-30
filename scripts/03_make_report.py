import os
import sys
import json
import shutil
import html as html_lib
from datetime import datetime, timezone
import base64
from io import BytesIO
from urllib.parse import urlparse
from typing import Any, Dict, Iterable, List, Optional, Tuple

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

from lib.raw_inputs import load_sources, enabled_sources, load_and_normalize
from lib.report_helpers import (
    CATEGORY_LABELS,
    ChartSpec,
    CompareBuilder,
    DERIVATIVE_COLS,
    FIRSTLAW_COLS,
    REQUIRED_THERMO_COLS,
    _apply_hover,
    _augment_region_frame,
    _calc_effective_window,
    _chart_interpretation,
    _figs_html,
    _filter_placeholders,
    _latest_numeric,
    filter_dashboard_events,
    load_dashboard_events,
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
raw_inputs_df: Optional[pd.DataFrame] = None

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


def _build_compare_context(region_ctxs: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    builder = CompareBuilder(region_ctxs)
    compare_data = builder.build()
    if compare_data is None:
        return None

    summary_html = ""
    if not compare_data.latest_rows.empty:
        latest_df = compare_data.latest_rows.copy()
        metric_labels = {
            "S_M": "S_M",
            "T_L": "T_L",
            "loop_area": "Loop",
            "X_C": "X_C",
        }
        cols: List[str] = [c for c in ["Region", "Latest date"] if c in latest_df.columns]
        for metric in metric_labels:
            for suffix in ("pctile", "rank"):
                col = f"{metric} {suffix}"
                if col in latest_df.columns:
                    cols.append(col)
        if cols:
            latest_df = latest_df[cols]
        latest_df = latest_df.rename(
            columns={
                key: value
                for metric, label in metric_labels.items()
                for key, value in (
                    (f"{metric} pctile", f"{label} pctile"),
                    (f"{metric} rank", f"{label} rank"),
                )
            }
        )
        try:
            dates = pd.to_datetime(latest_df.get("Latest date"), errors="coerce").dropna()
            latest_dt_str = dates.max().strftime("%Y-%m-%d") if not dates.empty else ""
        except Exception:
            latest_dt_str = ""
        headline = (
            f"<p><strong>At the latest date</strong>{' (' + latest_dt_str + ')' if latest_dt_str else ''}, this section compares only within-region historical position. Percentiles and ranks are computed inside each regional history; raw indicator levels are not compared across JP/EU/US.</p>"
        )
        snapshot = latest_df.to_html(index=False, border=0, classes="mini", float_format=lambda x: f"{x:.4g}")
        highlight_rows: List[str] = []
        for metric, label in [
            ("S_M", "Highest S_M rank"),
            ("T_L", "Highest T_L rank"),
            ("loop_area", "Highest loop rank"),
            ("X_C", "Highest X_C rank"),
        ]:
            pct_col = f"{metric_labels[metric]} pctile"
            rank_col = f"{metric_labels[metric]} rank"
            if pct_col in latest_df.columns:
                vals = pd.to_numeric(latest_df[pct_col], errors="coerce")
                if vals.notna().any():
                    row = latest_df.loc[vals.idxmax()]
                    detail = f"within-region pctile = {float(row[pct_col]):.1f}"
                    if rank_col in latest_df.columns and pd.notna(row.get(rank_col)):
                        detail += f", rank {row.get(rank_col)}"
                    highlight_rows.append(_summary_card(label, str(row.get("Region", "")), detail))
        highlights = '<div class="summary-grid compare-highlights">' + "".join(highlight_rows) + "</div>" if highlight_rows else ""
        summary_html = headline + highlights + "<h2>Compare – Latest within-region ranks</h2>" + _table_scroll(snapshot)

    std_charts_html = _figs_html(compare_data.std_figs) if compare_data.std_figs else ""

    _std_inner = std_charts_html if std_charts_html else "<p class=\"note small\">No standardized charts available.</p>"
    panes_html = (
        '<div class="compare-block">'
        f'<div class="pane std active">{_std_inner}</div>'
        '</div>'
    )
    region_html = (
        f"<section class=\"region-summary\"><h2>Compare (within-region ranks)</h2>{summary_html}</section>" + panes_html
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


def _summary_card(label: str, value: str, detail: str, tone: str = "neutral") -> str:
    return (
        f'<article class="summary-card tone-{html_lib.escape(tone)}">'
        f'<span class="summary-label">{html_lib.escape(label)}</span>'
        f'<strong class="summary-value">{html_lib.escape(value)}</strong>'
        f'<span class="summary-detail">{html_lib.escape(detail)}</span>'
        "</article>"
    )


def _summary_cards_html(items: List[str]) -> str:
    details = {
        "Latest date": "Most recent observation in this panel.",
        "S_M": "Dispersion and allocation spread.",
        "T_L": "Liquidity state index.",
        "Loop area": "Streaming open-path area in the policy-credit state plane.",
        "U": "Stored potential.",
        "X_C": "Remaining adjustment room.",
        "F_C": "Free-energy proxy.",
        "Maxwell curl Ω": "Proxy consistency diagnostic.",
        "First-law resid": "Energy-balance residual.",
    }
    cards: List[str] = []
    for item in items:
        label, _, value = item.partition(":")
        label = label.strip()
        value = value.strip() or "n/a"
        display_label = "Latest" if label == "Latest date" else label
        cards.append(_summary_card(display_label, value, details.get(label, "Latest reading.")))
    if not cards:
        return ""
    return '<div class="summary-grid">' + "".join(cards) + "</div>"


def _table_scroll(html: str) -> str:
    return f'<div class="table-scroll">{html}</div>' if html else ""


def _last_metric_row(ctx: Dict[str, Any]) -> Dict[str, Any]:
    frame = ctx.get("frame")
    if not isinstance(frame, pd.DataFrame) or frame.empty:
        return {}
    row = frame.iloc[-1]
    out: Dict[str, Any] = {
        "label": ctx.get("label", ""),
        "date": row.get("date"),
    }
    for col in ("S_M", "T_L", "loop_area", "X_C", "F_C"):
        if col in frame.columns:
            try:
                val = float(pd.to_numeric(frame[col], errors="coerce").dropna().iloc[-1])
            except Exception:
                val = np.nan
            out[col] = val if np.isfinite(val) else np.nan
    return out


def _build_dashboard_summary(region_ctxs: List[Dict[str, Any]]) -> str:
    rows = [_last_metric_row(ctx) for ctx in region_ctxs]
    rows = [row for row in rows if row]
    if not rows:
        return ""

    def leader(metric: str) -> Optional[Dict[str, Any]]:
        vals = [row for row in rows if metric in row and pd.notna(row.get(metric))]
        if not vals:
            return None
        return max(vals, key=lambda row: float(row[metric]))

    def fmt(v: Any) -> str:
        return f"{float(v):.4g}" if pd.notna(v) else "n/a"

    latest_dates = [pd.to_datetime(row.get("date"), errors="coerce") for row in rows]
    latest_dates = [d for d in latest_dates if pd.notna(d)]
    latest_date = max(latest_dates).strftime("%Y-%m-%d") if latest_dates else "n/a"
    stale = []
    if latest_dates:
        freshest = max(latest_dates)
        for row in rows:
            dt = pd.to_datetime(row.get("date"), errors="coerce")
            if pd.notna(dt) and (freshest - dt).days > 365:
                stale.append(str(row.get("label", "")))

    cards = [_summary_card("Latest observation", latest_date, "Freshest regional panel.")]
    for metric, label, detail in [
        ("S_M", "Dispersion peak", "Largest money/credit dispersion."),
        ("T_L", "Liquidity peak", "Highest liquidity state index."),
        ("loop_area", "Loop peak", "Largest streaming loop area."),
        ("X_C", "Headroom peak", "Largest remaining adjustment room."),
    ]:
        item = leader(metric)
        if item:
            cards.append(_summary_card(label, str(item.get("label", "")), f"{metric} = {fmt(item.get(metric))}"))
    stale_note = ""
    if stale:
        stale_note = (
            '<p class="note small"><strong>Data freshness note:</strong> '
            + html_lib.escape(", ".join(stale))
            + " is more than one year behind the freshest regional panel.</p>"
        )
    return (
        '<section class="decision-summary">'
        '<div class="section-heading"><span class="section-kicker">Overview</span><h2>Current signal</h2></div>'
        '<div class="summary-grid kpi-grid">' + "".join(cards) + "</div>" + stale_note + "</section>"
    )


def _build_coverage_summary(region_ctxs: List[Dict[str, Any]]) -> str:
    dated: List[Tuple[Dict[str, Any], pd.Timestamp]] = []
    for ctx in region_ctxs:
        dt = pd.to_datetime(ctx.get("last_date"), errors="coerce")
        if pd.notna(dt):
            dated.append((ctx, dt))
    if not dated:
        return ""
    freshest = max(dt for _, dt in dated)
    cards: List[str] = []
    for ctx, dt in dated:
        lag_days = int((freshest - dt).days)
        if lag_days <= 45:
            tone, label, note = "current", "Current", "Freshest panel."
        elif lag_days <= 365:
            tone, label, note = "delayed", "Delayed", f"{lag_days} days lag."
        else:
            tone, label, note = "stale", "Stale", f"{lag_days} days lag."
        cards.append(
            '<article class="coverage-card tone-' + tone + '">'
            '<div class="coverage-head">'
            f'<span class="coverage-region">{html_lib.escape(str(ctx.get("label", "")))}</span>'
            f'<span class="status-badge tone-{tone}">{label}</span>'
            "</div>"
            f'<strong class="coverage-date">{dt.strftime("%Y-%m-%d")}</strong>'
            f'<p class="coverage-note">{html_lib.escape(note)}</p>'
            "</article>"
        )
    return (
        '<section class="coverage-summary">'
        '<div class="section-heading"><span class="section-kicker">Coverage</span><h2>Data window</h2></div>'
        '<div class="coverage-grid">' + "".join(cards) + "</div></section>"
    )


def _build_event_summary(region_ctxs: List[Dict[str, Any]]) -> str:
    events = load_dashboard_events(os.path.join(DATA_DIR, "report_events.csv"))
    if not events:
        return ""
    starts: List[pd.Timestamp] = []
    ends: List[pd.Timestamp] = []
    for ctx in region_ctxs:
        frame = ctx.get("frame")
        if isinstance(frame, pd.DataFrame) and "date" in frame.columns and not frame.empty:
            dates = pd.to_datetime(frame["date"], errors="coerce").dropna()
            if not dates.empty:
                starts.append(dates.min())
                ends.append(dates.max())
    visible = filter_dashboard_events(
        events,
        start_date=min(starts) if starts else None,
        end_date=max(ends) if ends else None,
    )
    if not visible:
        return ""
    cards: List[str] = []
    for event in visible[:8]:
        category = str(event.get("category") or "event").lower()
        tone = category if category in {"bubble", "crisis", "pandemic", "policy"} else "event"
        regions = ", ".join(str(x).upper() for x in event.get("regions", []) if x)
        cards.append(
            '<article class="event-card">'
            '<div class="event-head">'
            f'<span class="event-chip tone-{html_lib.escape(tone)}">{html_lib.escape(regions or "ALL")}</span>'
            f'<span class="event-chip tone-{html_lib.escape(tone)}">{html_lib.escape(category.title())}</span>'
            "</div>"
            f'<strong class="event-title">{html_lib.escape(str(event.get("label", "")))}</strong>'
            f'<span class="event-date">{event["start_date"].strftime("%Y-%m-%d")} to {event["end_date"].strftime("%Y-%m-%d")}</span>'
            f'<p class="event-note">{html_lib.escape(str(event.get("description", "")))}</p>'
            "</article>"
        )
    return (
        '<details class="event-summary"><summary>Reference events</summary>'
        '<p class="note small">Event bands are used as reading context for the visible chart window.</p>'
        '<div class="event-grid">' + "".join(cards) + "</div></details>"
    )


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
    return '<details class="inputs-summary"><summary>Inputs summary</summary>' + "".join(rows) + "</details>"


def _build_freshness_summary(region_ctxs: List[Dict[str, Any]]) -> str:
    rows: List[Tuple[str, Optional[datetime]]] = []
    dated_rows: List[datetime] = []

    for ctx in region_ctxs:
        label = str(ctx.get("label") or "")
        no_indicator_data = any("No indicator data" in str(item) for item in ctx.get("summary_items", []))
        parsed = pd.to_datetime(ctx.get("last_date"), errors="coerce")
        last_date = None if pd.isna(parsed) or no_indicator_data else parsed.to_pydatetime()
        if last_date is not None:
            dated_rows.append(last_date)
        rows.append((label, last_date))

    if not rows:
        return ""

    latest_across_regions = max(dated_rows) if dated_rows else None
    checked_at = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    cards: List[str] = []

    for label, last_date in rows:
        safe_label = html_lib.escape(label)
        if last_date is None:
            badge = "Data gap"
            badge_cls = "badge-gap"
            date_text = "No data"
            detail = "No indicator rows are available for this region in this run."
        else:
            date_text = last_date.strftime("%Y-%m-%d")
            lag_days = (latest_across_regions - last_date).days if latest_across_regions else 0
            if lag_days > 92:
                badge = "Data gap"
                badge_cls = "badge-gap"
                detail = f"{lag_days} days behind the latest regional observation."
            else:
                badge = "Latest available"
                badge_cls = "badge-ok"
                detail = "Aligned with the latest regional observation." if lag_days == 0 else f"{lag_days} days behind the latest regional observation."

        cards.append(
            '<div class="freshness-card">'
            f"<strong>{safe_label}</strong>"
            f"<span class=\"freshness-date\">{html_lib.escape(date_text)}</span>"
            f"<span class=\"freshness-badge {badge_cls}\">{html_lib.escape(badge)}</span>"
            f"<span class=\"freshness-detail\">{html_lib.escape(detail)}</span>"
            "</div>"
        )

    return (
        '<section class="freshness-panel" aria-label="Data freshness">'
        "<h2>Data freshness</h2>"
        '<p class="note small">Latest available differs by region. Cross-region comparisons use each region\'s latest observation, not necessarily the same calendar month.</p>'
        '<div class="freshness-grid">' + "".join(cards) + "</div>"
        f'<p class="freshness-meta">Check date: {checked_at} UTC</p>'
        "</section>"
    )


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
        "T_L": ("Liquidity state", "Composite flow proxy", "index"),
        "p_C": ("Credit pressure", "Conjugate to V_C", "index"),
        "V_C": ("Credit volume", "Capacity proxy", "index"),
        "U": ("Internal energy", "Stored potential", "index"),
        "F_C": ("Free energy F_C", "Helmholtz proxy", "index"),
        "X_C": ("Exergy ceiling X_C", "Usable potential", "index"),
        "loop_area": ("Loop area", "Streaming open-path area", "index^2"),
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
    palette = {"JP": "#62d2ff", "JPN": "#62d2ff", "EU": "#ffbd69", "EZ": "#ffbd69", "US": "#70e1c8", "USA": "#70e1c8"}
    discrete_map = {series: palette.get(country, "#b4cad9") for series, country in color_map.items()}
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

    fig_specs: List[ChartSpec] = []
    if {"S_M", "T_L"}.issubset(local.columns) and not plot_df.empty:
        # Dual-axis layout for very different scales
        fig = make_dual_axis_sm_tl(plot_df, title=f"{label} – S_M & T_L")
        _style_figure(fig)
        _apply_hover(fig, ".3f")
        interp = _chart_interpretation("S_M & T_L", plot_df)
        fig_specs.append((fig, "S_M & T_L", "Entropy & liquidity state", interp))
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
            title=f"{label} – Loop Path Monitor",
            labels={"loop_area": "Streaming loop area", "date": "Date"},
        )
        _style_figure(fig)
        _apply_hover(fig, ".3f")
    interp = _chart_interpretation("Loop Path Monitor", plot_df)
    fig_specs.append((fig, "Loop Path Monitor", "Loop area", interp))
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
    fmt = lambda v: f"{float(v):.4g}" if pd.notna(v) else "n/a"
    summary_items: List[str] = []
    summary_items.append(f"Latest date: {last_date.strftime('%Y-%m-%d')}")
    if "S_M" in local.columns:
        summary_items.append(f"S_M: {fmt(last_row.get('S_M'))}")
    if "T_L" in local.columns:
        summary_items.append(f"T_L: {fmt(last_row.get('T_L'))}")
    if "loop_area" in local.columns:
        summary_items.append(f"Streaming loop area: {fmt(last_row.get('loop_area'))}")
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
    maxwell_col = "maxwell_curl" if "maxwell_curl" in local.columns else "maxwell_gap"
    if has_derivatives and maxwell_col in local.columns:
        summary_items.append(f"Maxwell curl Ω: {fmt(last_row.get(maxwell_col))}")
    if has_thermo and "firstlaw_resid" in local.columns:
        summary_items.append(f"First-law resid: {fmt(last_row.get('firstlaw_resid'))}")
    summary_html = _summary_cards_html(summary_items)

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

    parts: List[str] = []
    if sm_bucket and tl_bucket:
        parts.append(f"{label} sits in a <strong>{sm_bucket}-dispersion, {tl_bucket}-liquidity</strong> regime.")
    elif sm_bucket or tl_bucket:
        if sm_bucket:
            parts.append(f"Dispersion is <strong>{sm_bucket}</strong>.")
        if tl_bucket:
            parts.append(f"Liquidity state is <strong>{tl_bucket}</strong>.")
    if la_desc:
        parts.append(f"Streaming loop area is <strong>{la_desc}</strong>, indicating {'an active loop path' if la_desc=='non-zero' else 'a quiet loop path'}.")
    if xc_desc:
        parts.append(f"X<sub>C</sub> is <strong>{xc_desc}</strong>.")
    comment_html = ("<p>" + " ".join(parts) + "</p>") if parts else ""

    chart_lines: List[Tuple[str, str]] = []
    if "S_M" in local.columns or "T_L" in local.columns:
        msg_parts: List[str] = []
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
        chart_lines.append(("Loop Path Monitor", f"Streaming loop area is {la_desc or fmt(last_la)}{trend_txt}."))
    if last_xc is not None:
        xc_trend = _series_trend(xc_series) if xc_series is not None else None
        xc_text = xc_desc or f"{fmt(last_xc)}"
        suffix = f" and {xc_trend}" if xc_trend else ""
        chart_lines.append(("Credit Exergy Ceiling", f"X_C is {xc_text}{suffix}."))
    maxwell_col = "maxwell_curl" if "maxwell_curl" in local.columns else "maxwell_gap"
    if has_derivatives and maxwell_col in local.columns:
        gap_desc = fmt(last_row.get(maxwell_col))
        spec = "alerts active" if out_of_spec_ranges else "inside spec"
        chart_lines.append(("Maxwell-like Test", f"Curl Ω is {gap_desc} ({spec})."))
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
        mini_html = _table_scroll(mini_tail.to_html(index=False, border=0, classes="mini", escape=True))

    diagnostics_html = ""
    if has_derivatives and effective_window >= 3 and deriv_cols_present:
        diag_subset = local[["date"] + deriv_cols_present].dropna().tail(6)
        if not diag_subset.empty:
            diag_subset["date"] = diag_subset["date"].dt.strftime("%Y-%m-%d")
            diagnostics_html += f"<h2>Diagnostics – Maxwell-like (window={effective_window})</h2>" + _table_scroll(diag_subset.to_html(index=False, border=0, classes="mini", escape=True))
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
            diagnostics_html += "<h2>Diagnostics – First-law</h2>" + _table_scroll(fl.to_html(index=False, border=0, classes="mini", escape=True))

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
        "<li>Maxwell-like relation (rolling local linear regression): $\\left. \\partial T / \\partial V \\right|_S + \\left. \\partial p / \\partial S \\right|_V \\approx 0$</li>"
        "</ul>"
    )
    sources_html = _sources_table(sources_meta)

    inputs_summary_html = _build_inputs_summary(regions)
    freshness_html = _build_freshness_summary(regions)
    dashboard_summary_html = _build_dashboard_summary(regions)
    coverage_summary_html = _build_coverage_summary(regions)
    event_summary_html = _build_event_summary(regions)

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
        "Loop Path Monitor": "fig2.png",
        "Credit Exergy Ceiling": "fig3.png",
        "Maxwell-like Test": "fig4.png",
        "First-law Decomposition": "fig5.png",
        "Raw Inputs (first=100)": "fig_raw_inputs.png",
    }

    png_fallback_ok = False
    skip_png_fallback = os.getenv("TMS_SKIP_PNG", "").strip().lower() in {"1", "true", "yes"}
    if jp_ctx and not skip_png_fallback:
        png_targets: List[Tuple[Any, str]] = []
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
        ".freshness-panel{background:#fff;border:1px solid #dde4ee;border-radius:8px;padding:.85rem 1rem;margin:1rem 0}"
        ".freshness-panel h2{margin-top:0}.freshness-grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(180px,1fr));gap:.6rem}"
        ".freshness-card{border:1px solid #e1e6ef;border-radius:6px;background:#fafcff;padding:.6rem;display:flex;flex-direction:column;gap:.2rem}"
        ".freshness-date{font-size:.9rem;color:#1b2a43}.freshness-detail{font-size:.78rem;color:#555}.freshness-meta{font-size:.78rem;color:#666;margin:.6rem 0 0}"
        ".freshness-badge{display:inline-block;width:max-content;border-radius:999px;padding:.15rem .45rem;font-size:.72rem;font-weight:600}"
        ".freshness-badge.badge-ok{background:#e9f7ef;color:#166534}.freshness-badge.badge-gap{background:#fff2cc;color:#92400e}"
        "details{margin:.5rem 0}details>summary{cursor:pointer;list-style:none;font-weight:600}details>summary::-webkit-details-marker{display:none}"
        ".inputs-summary{background:#fafafa;border:1px solid #eee;padding:.75rem;border-radius:6px;margin:.75rem 0 1rem}"
        ".inputs-summary .inputs-row{margin:.35rem 0}.inputs-summary .region-tag{display:inline-block;background:#333;color:#fff;border-radius:3px;padding:.15rem .4rem;font-size:.75rem;margin-right:.4rem}"
        ".inputs-summary .pill-list{display:inline}.inputs-summary .pill{display:inline-block;border:1px solid #ddd;background:#fff;border-radius:999px;padding:.15rem .5rem;margin:.15rem .25rem;font-size:.75rem}"
        """
@import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700;800&display=swap');
:root{color-scheme:dark;--page-bg:#02213b;--surface:#072d4a;--surface-muted:#0a3857;--fg:#f5fbff;--ink:#f5fbff;--ink-soft:#d9edf8;--accent:#62d2ff;--accent-cool:#62d2ff;--sand:#62d2ff;--border:rgba(185,226,255,.2);--border-strong:rgba(185,226,255,.34);--text:#f5fbff;--muted:#b4cad9;--soft:#8fb1c5;--shadow:0 22px 60px rgba(0,10,24,.28);--shadow-soft:0 10px 30px rgba(0,10,24,.2);--radius:20px;--radius-sm:12px}
*,*::before,*::after{box-sizing:border-box}
html{scroll-behavior:smooth;background:var(--page-bg)}
body{margin:0;overflow-x:hidden;background:radial-gradient(circle at 8% 0%,rgba(23,126,115,.14),transparent 27rem),radial-gradient(circle at 100% 30%,rgba(241,199,109,.15),transparent 30rem),var(--page-bg);color:var(--text);font-family:"Avenir Next","Gill Sans",Candara,"Trebuchet MS",sans-serif;line-height:1.58}
h1,h2,h3{font-family:"Iowan Old Style","Palatino Linotype",Palatino,Georgia,serif;color:var(--ink)}h1{max-width:780px;margin:0;font-size:clamp(2.65rem,5.2vw,5.15rem);font-weight:600;letter-spacing:-.045em;line-height:.96}h2{margin:0;font-size:clamp(1.35rem,2vw,1.85rem);font-weight:600;line-height:1.16}h3{margin:.2rem 0 .5rem;font-size:1rem}p,ul{margin-top:0}.wrap{max-width:1320px;margin:0 auto;padding:1.5rem}.page-header,.page-content{display:grid;gap:1.25rem}.page-header{position:relative;isolation:isolate;overflow:hidden;margin-bottom:1.4rem;border:1px solid rgba(255,255,255,.14);border-radius:30px;background:linear-gradient(135deg,#0b2526 0%,#123b39 57%,#1d514a 100%);box-shadow:0 30px 80px rgba(9,39,39,.2);color:#fff}.page-header::before{position:absolute;z-index:-1;inset:0;content:"";opacity:.22;background-image:linear-gradient(rgba(255,255,255,.08) 1px,transparent 1px),linear-gradient(90deg,rgba(255,255,255,.08) 1px,transparent 1px);background-size:42px 42px;mask-image:linear-gradient(110deg,transparent 10%,#000 85%)}.page-header::after{position:absolute;z-index:-1;right:-8rem;bottom:-15rem;width:34rem;height:34rem;border:1px solid rgba(241,199,109,.35);border-radius:50%;box-shadow:0 0 0 4rem rgba(241,199,109,.035),0 0 0 9rem rgba(241,199,109,.025);content:""}.page-hero{display:grid;grid-template-columns:minmax(0,1fr) auto;align-items:end;gap:2rem;padding:3.8rem 3rem 3.4rem;border:0}.page-hero::after{align-self:end;max-width:230px;padding:1rem 1.1rem;border:1px solid rgba(255,255,255,.18);border-radius:16px;background:rgba(4,22,22,.24);box-shadow:inset 0 1px rgba(255,255,255,.08);color:#d9ece7;content:"3 REGIONS\\A 4 CORE METRICS\\A RECALCULABLE MODEL";font-size:.72rem;font-weight:750;letter-spacing:.12em;line-height:1.9;white-space:pre}.page-header h1{color:#fff}.page-kicker,.section-kicker{display:block;color:var(--accent);font-size:.7rem;font-weight:800;letter-spacing:.16em;text-transform:uppercase}.page-kicker{margin-bottom:1rem;color:var(--sand)}.page-subtitle,.note{max-width:760px;margin:.85rem 0 0;color:var(--muted)}.page-subtitle{color:#c7dad6;font-size:1.02rem}.note.small{color:var(--soft);font-size:.84rem}
.intro,.coverage-summary,.region-summary,.inputs-summary,.event-summary,.reference-panel{padding:1.45rem;background:rgba(255,253,248,.94);border:1px solid rgba(190,200,191,.7);border-radius:var(--radius);box-shadow:var(--shadow-soft)}.decision-summary{position:relative;overflow:hidden;padding:1.65rem;background:linear-gradient(125deg,#f7ecd2,#fff9ea 58%,#f5e2bb);border:1px solid #e5d1a4;border-radius:var(--radius);box-shadow:var(--shadow);color:var(--ink)}.decision-summary::after{position:absolute;right:-2rem;top:-3.5rem;width:12rem;height:12rem;border:1px solid rgba(230,109,50,.22);border-radius:50%;content:""}.intro{background:#e9f3ef;border-color:#c8ddd6}.intro ul{margin:.5rem 0 .8rem;padding-left:1.15rem}.intro li{margin:.28rem 0}.section-heading{display:grid;gap:.25rem;margin-bottom:1rem}.decision-summary .section-kicker{color:#a44a25}.decision-summary h2{color:var(--ink)}
.summary-grid,.coverage-grid,.event-grid,.chart-grid{display:grid;gap:1rem}.summary-grid{grid-template-columns:repeat(auto-fit,minmax(160px,1fr));margin:0}.kpi-grid{grid-template-columns:repeat(5,minmax(0,1fr))}.coverage-grid{grid-template-columns:repeat(3,minmax(0,1fr))}.event-grid{grid-template-columns:repeat(auto-fit,minmax(240px,1fr))}
.summary-card,.coverage-card,.event-card{min-width:0;padding:1rem;border:1px solid var(--border);border-radius:var(--radius-sm);background:var(--surface)}.summary-card{transition:transform .2s ease,box-shadow .2s ease}.summary-card:hover{transform:translateY(-2px);box-shadow:var(--shadow-soft)}.decision-summary .summary-card{background:rgba(255,255,255,.58);border-color:rgba(165,120,57,.2);box-shadow:inset 0 1px rgba(255,255,255,.8)}.summary-label,.coverage-region{display:block;color:var(--muted);font-size:.68rem;font-weight:800;letter-spacing:.1em;text-transform:uppercase}.decision-summary .summary-label{color:#87552d}.summary-value,.coverage-date,.event-title{display:block;margin-top:.48rem;color:var(--text);font-size:1.05rem;font-weight:800;line-height:1.2;overflow-wrap:anywhere}.decision-summary .summary-value{color:var(--ink);font-size:1.18rem}.summary-detail,.coverage-note,.event-note,.event-date{display:block;margin-top:.32rem;color:var(--soft);font-size:.78rem;line-height:1.45}.decision-summary .summary-detail{color:#705f48}
.coverage-card{position:relative;overflow:hidden;padding:1.15rem 1.2rem!important;background:var(--surface)!important;color:var(--text)!important}.coverage-card::before{position:absolute;inset:0 auto 0 0;width:4px;background:var(--border-strong);content:""}.coverage-card.tone-current::before{background:#2b9a77}.coverage-card.tone-delayed::before{background:#dc9b2f}.coverage-card.tone-stale::before{background:#d55245}.coverage-head,.event-head{display:flex;flex-wrap:wrap;align-items:center;justify-content:space-between;gap:.45rem;margin-bottom:.45rem}.status-badge,.event-chip{display:inline-flex;align-items:center;border-radius:999px;padding:.2rem .58rem;border:1px solid transparent;font-size:.66rem;font-weight:800;letter-spacing:.03em}.tone-current{border-color:#9cd5bf!important;background:#eef8f3!important;color:#166649!important}.tone-delayed{border-color:#e9c06e!important;background:#fff7e6!important;color:#855817!important}.tone-stale,.tone-crisis{border-color:#e8aaa4!important;background:#fff0ee!important;color:#8e2e27!important}.tone-bubble{border-color:#e9c06e!important;background:#fff5d8!important;color:#855817!important}.tone-pandemic{border-color:#9ebedc!important;background:#edf5fb!important;color:#275d87!important}.tone-policy{border-color:#99cfc5!important;background:#edf8f5!important;color:#17695e!important}
.table-scroll{width:100%;margin-top:1rem;overflow-x:auto;border:1px solid var(--border);border-radius:var(--radius-sm);-webkit-overflow-scrolling:touch}table.mini{width:100%;border-collapse:collapse;margin:0;background:var(--surface);font-size:.82rem}table.mini th,table.mini td{padding:.72rem .78rem;border-bottom:1px solid #e6e9e3;text-align:right;vertical-align:top;white-space:nowrap}table.mini thead th{position:sticky;top:0;background:#e9efeb;color:var(--ink);font-size:.68rem;letter-spacing:.06em;text-transform:uppercase}table.mini tbody tr:nth-child(even){background:#faf8f1}table.mini tbody tr:last-child td{border-bottom:0}table.mini th:first-child,table.mini td:first-child{text-align:left}
.tabs,.subtabs{display:flex;flex-wrap:wrap;gap:.45rem}.tabs{position:sticky;top:.7rem;z-index:8;justify-content:center;width:max-content;max-width:100%;margin:.15rem auto .35rem;padding:.42rem;background:rgba(16,42,42,.92);backdrop-filter:blur(16px);border:1px solid rgba(255,255,255,.15);border-radius:999px;box-shadow:0 14px 35px rgba(16,42,42,.18)}.tabs button,.subtabs button{border:1px solid var(--border-strong);background:var(--surface);color:var(--text);border-radius:999px;cursor:pointer;transition:background .18s ease,color .18s ease,border-color .18s ease,transform .18s ease}.tabs button{padding:.58rem 1.05rem;border-color:transparent;background:transparent;color:#d6e4e0;font-size:.78rem;font-weight:750}.subtabs button{padding:.38rem .76rem;font-size:.76rem;font-weight:700}.tabs button:hover{color:#fff;transform:translateY(-1px)}.tabs button.active{border-color:rgba(241,199,109,.5);background:var(--sand);color:#18312f}.subtabs button.active{border-color:var(--ink);background:var(--ink);color:#fff}.compare-block .pane,.region{display:none}.compare-block .pane.active,.region.active{display:block;animation:panel-in .36s ease both}
.region-summary{padding:1.65rem}.region-summary>h2{margin-bottom:.85rem}.region-summary>p{max-width:900px;color:var(--muted)}.chart-grid{grid-template-columns:1fr;margin-top:1.15rem}.compare-block .chart-grid{grid-template-columns:repeat(2,minmax(0,1fr));gap:1.1rem}figure.chart-card{position:relative;margin:0;padding:1.1rem;background:rgba(255,253,248,.96);border:1px solid rgba(190,200,191,.75);border-radius:var(--radius);box-shadow:var(--shadow-soft);overflow:hidden;transition:transform .2s ease,box-shadow .2s ease}figure.chart-card:hover{transform:translateY(-2px);box-shadow:var(--shadow)}figure.chart-card::before{position:absolute;top:0;left:1.1rem;width:3.6rem;height:3px;border-radius:0 0 4px 4px;background:var(--accent-cool);content:""}.chart-card .plotly-graph-div{min-height:420px}figcaption{display:grid;gap:.25rem;padding:.8rem .2rem .1rem;border-top:1px solid #e8e9e2;color:var(--muted);font-size:.82rem}figcaption strong{color:var(--ink);font-size:.9rem}.chart-notes{margin:.9rem 0;padding:.7rem .85rem;background:#eef4f0;border:1px solid #d1ded7;border-radius:var(--radius-sm)}.chart-note{display:flex;flex-direction:column;gap:.12rem;margin:.28rem 0;font-size:.82rem}.chart-note strong{color:var(--ink);font-weight:700}.chart-note span,.chart-note-inline{display:block;color:var(--muted);font-size:.78rem}
details{margin:.55rem 0}details>summary{position:relative;padding-right:2.2rem;cursor:pointer;color:var(--ink);font-weight:800;list-style:none}details>summary::after{position:absolute;right:.15rem;top:50%;width:1.5rem;height:1.5rem;border:1px solid var(--border);border-radius:50%;content:"+";font-size:1rem;font-weight:500;line-height:1.35rem;text-align:center;transform:translateY(-50%);transition:transform .18s ease}details[open]>summary::after{transform:translateY(-50%) rotate(45deg)}details>summary::-webkit-details-marker{display:none}.inputs-row{margin:.7rem 0;padding:.75rem 0;border-bottom:1px solid var(--border)}.inputs-row:last-child{border-bottom:0}.inputs-summary .region-tag{display:inline-block;margin-right:.45rem;padding:.22rem .55rem;border-radius:999px;background:var(--ink);color:#fff;font-size:.7rem;font-weight:700}.inputs-summary .pill-list{display:inline}.inputs-summary .pill{display:inline-block;margin:.18rem .22rem;padding:.25rem .62rem;border:1px solid var(--border);border-radius:999px;background:#f9f7f0;font-size:.74rem}.event-card{border-left:3px solid var(--accent);box-shadow:0 5px 16px rgba(17,42,42,.04)}
/* Toppy main-site theme */
body{background:radial-gradient(circle at 8% 0%,rgba(98,210,255,.12),transparent 28rem),radial-gradient(circle at 100% 28%,rgba(12,113,155,.18),transparent 32rem),var(--page-bg);font-family:"Plus Jakarta Sans",-apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif}
h1,h2,h3{font-family:"Plus Jakarta Sans",-apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif;color:var(--fg)}
.page-header{border-color:var(--border);background:linear-gradient(135deg,#02213b 0%,#072d4a 58%,#0c719b 145%);box-shadow:0 30px 80px rgba(0,10,24,.34)}.page-header::before{background-image:linear-gradient(rgba(98,210,255,.08) 1px,transparent 1px),linear-gradient(90deg,rgba(98,210,255,.08) 1px,transparent 1px)}.page-header::after{border-color:rgba(98,210,255,.3);box-shadow:0 0 0 4rem rgba(98,210,255,.035),0 0 0 9rem rgba(98,210,255,.02)}.page-hero::after{border-color:var(--border);background:rgba(2,33,59,.5);color:var(--muted)}.page-kicker,.section-kicker{color:var(--accent)}.page-subtitle{color:var(--muted)}
.intro,.coverage-summary,.region-summary,.inputs-summary,.event-summary,.reference-panel{background:rgba(7,45,74,.94);border-color:var(--border);box-shadow:var(--shadow-soft)}.decision-summary{background:linear-gradient(125deg,#072d4a,#093957 62%,#0b4968);border-color:var(--border);color:var(--fg)}.decision-summary::after{border-color:rgba(98,210,255,.24)}.decision-summary .section-kicker,.decision-summary h2{color:var(--accent)}.intro{background:#083852;border-color:var(--border)}
.summary-card,.coverage-card,.event-card{border-color:var(--border);background:rgba(2,33,59,.52)}.decision-summary .summary-card{background:rgba(2,33,59,.42);border-color:var(--border);box-shadow:inset 0 1px rgba(255,255,255,.04)}.decision-summary .summary-label{color:var(--accent)}.decision-summary .summary-value{color:var(--fg)}.decision-summary .summary-detail{color:var(--muted)}
.coverage-card{background:var(--surface)!important;color:var(--fg)!important}.tone-current{border-color:rgba(98,210,255,.42)!important;background:rgba(98,210,255,.1)!important;color:#bcecff!important}.tone-delayed{border-color:rgba(180,202,217,.42)!important;background:rgba(180,202,217,.08)!important;color:#dcebf4!important}.tone-stale,.tone-crisis{border-color:rgba(255,137,137,.4)!important;background:rgba(255,91,91,.09)!important;color:#ffbaba!important}.tone-bubble{border-color:rgba(98,210,255,.35)!important;background:rgba(98,210,255,.08)!important;color:#bcecff!important}.tone-pandemic{border-color:rgba(125,184,255,.4)!important;background:rgba(125,184,255,.09)!important;color:#c7e0ff!important}.tone-policy{border-color:rgba(98,210,255,.42)!important;background:rgba(98,210,255,.1)!important;color:#bcecff!important}
.table-scroll{border-color:var(--border)}table.mini{background:var(--surface)}table.mini th,table.mini td{border-bottom-color:var(--border)}table.mini thead th{background:#0a3a59;color:var(--fg)}table.mini tbody tr:nth-child(even){background:rgba(98,210,255,.035)}
.tabs{background:rgba(2,33,59,.94);border-color:var(--border);box-shadow:0 14px 35px rgba(0,10,24,.28)}.tabs button{color:var(--muted)}.tabs button.active{border-color:var(--accent);background:var(--accent);color:var(--page-bg)}.subtabs button{border-color:var(--border);background:var(--surface);color:var(--muted)}.subtabs button.active{border-color:var(--accent-strong);background:var(--accent-strong);color:#fff}
figure.chart-card{background:rgba(7,45,74,.96);border-color:var(--border);box-shadow:var(--shadow-soft)}figure.chart-card::before{background:var(--accent)}figcaption{border-top-color:var(--border);color:var(--muted)}figcaption strong{color:var(--fg)}.chart-notes{background:rgba(98,210,255,.06);border-color:var(--border)}.chart-note strong{color:var(--fg)}.event-card{border-left-color:var(--accent);box-shadow:0 5px 16px rgba(0,10,24,.18)}
details>summary{color:var(--fg)}details>summary::after{border-color:var(--border)}.inputs-row{border-bottom-color:var(--border)}.inputs-summary .region-tag{background:var(--accent-strong);color:#fff}.inputs-summary .pill{border-color:var(--border);background:rgba(2,33,59,.5)}
@keyframes panel-in{from{opacity:0;transform:translateY(8px)}to{opacity:1;transform:none}}@keyframes rise-in{from{opacity:0;transform:translateY(14px)}to{opacity:1;transform:none}}.page-header,.decision-summary,.coverage-summary{animation:rise-in .55s ease both}.decision-summary{animation-delay:.08s}.coverage-summary{animation-delay:.14s}
""".strip()
        + f":root{{--brand-bg:{BRAND_BG};--brand-bg2:{BRAND_BG2};--brand-text:{BRAND_TEXT};}}"
        + ".brandbar,.footer-brand{display:flex;align-items:center;gap:11px;color:var(--brand-text)}.brandbar{min-height:58px;padding:.65rem 1rem;border-bottom:1px solid rgba(255,255,255,.12);background:transparent}.brandbar img,.footer-brand img{width:auto;border-radius:8px;box-shadow:0 0 0 1px rgba(255,255,255,.22)}.brandbar img{height:34px}.brandbar .brand-name{font-size:.9rem;font-weight:800;letter-spacing:.03em;color:var(--brand-text)}.brand-tag{margin-left:auto;border:1px solid rgba(255,255,255,.2);border-radius:999px;padding:.24rem .62rem;color:#d4e7e2;font-size:.66rem;font-weight:800;letter-spacing:.08em;text-transform:uppercase}.footer-brand{justify-content:center;margin-top:1.6rem;padding:1rem;color:#c9d9d5;background:var(--ink);border-radius:var(--radius);font-size:.74rem}.footer-brand img{height:28px}@media(max-width:1000px){.kpi-grid{grid-template-columns:repeat(3,minmax(0,1fr))}.compare-block .chart-grid{grid-template-columns:1fr}}@media(max-width:760px){.wrap{padding:.7rem}.page-header{border-radius:22px}.brandbar{padding:.55rem .75rem}.brand-tag{display:none}.page-hero{display:block;padding:2.4rem 1.35rem 2.2rem}.page-hero::after{display:block;margin-top:1.6rem;max-width:none}.decision-summary,.coverage-summary,.region-summary,.intro,.inputs-summary,.event-summary,.reference-panel{padding:1.1rem}.kpi-grid,.coverage-grid,.chart-grid{grid-template-columns:1fr}.tabs{top:.35rem;width:100%;border-radius:16px}.tabs button{flex:1 1 42%;min-width:0;padding:.5rem .35rem}.chart-card .plotly-graph-div{min-height:360px}figure.chart-card{padding:.7rem}}@media(prefers-reduced-motion:reduce){html{scroll-behavior:auto}.page-header,.decision-summary,.coverage-summary,.region.active,.pane.active{animation:none!important}.summary-card,figure.chart-card,.tabs button{transition:none!important}}"
    )

    head = ("<!doctype html><html lang=\"en\"><head><meta charset=\"utf-8\"><meta name=\"viewport\" "
            "content=\"width=device-width,initial-scale=1\"><title>Thermo-Credit Monitor</title><meta name=\"description\" "
            "content=\"Monthly thermo-credit indicators.\"><style>" + style_block + "</style>"
            + "</head><body><div class=\"wrap\"><header class=\"page-header\"><div class=\"brandbar\">"
            + (f'<img src="{logo_uri}" alt="Company Logo"/>' if logo_uri else "")
            + '<span class="brand-name">ToppyMicroServices</span><span class="brand-tag">Research dashboard</span></div><div class="page-hero"><div><span class="page-kicker">Thermo-credit monitor</span><h1>Regional Credit Thermodynamics</h1><p class="page-subtitle">Monthly indicators for dispersion, liquidity state, loop-path monitoring, and credit exergy.</p></div></div></header>')

    intro_html = (
        '<details class="intro">'
        '<summary>How to read this dashboard</summary>'
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
        '<li><strong>T<sub>L</sub></strong> – liquidity state index (funding &amp; market conditions)</li>'
        '<li><strong>Loop area</strong> – streaming open-path area in the policy/regulatory state plane</li>'
        '<li><strong>X<sub>C</sub></strong> – remaining “credit exergy”, i.e. safe room to adjust</li>'
        '</ul>'
        '<p>Values here are <strong>experimental</strong> and follow the Thermo-Credit v0.x spec. '
        'They are for research and discussion, not for trading or regulatory use.</p>'
        '</details>'
    )

    page_body = (
        '<main class="page-content">'
        + dashboard_summary_html
        + freshness_html
        + coverage_summary_html
        + tabs_html
        + regions_html
        + intro_html
        + inputs_summary_html
        + event_summary_html
        + noscript
        + sources_html
        + '<section class="reference-panel">'
        + defs_html
        + formulas_html
        + '</section></main>'
    )
    script_block = ("\n<script>(function(){const resizePlots=(root)=>{if(!window.Plotly||!root)return;root.querySelectorAll('.plotly-graph-div').forEach(plot=>window.Plotly.Plots.resize(plot));};const tabs=[...document.querySelectorAll('.tabs button')];if(tabs.length){"
                    "tabs.forEach(btn=>btn.addEventListener('click',()=>{tabs.forEach(x=>x.classList.remove('active'));btn.classList.add('active');"
                    "const tgt=btn.getAttribute('data-target');document.querySelectorAll('.region').forEach(r=>r.classList.remove('active'));"
                    "const el=document.getElementById('region-'+tgt);if(el){el.classList.add('active');requestAnimationFrame(()=>resizePlots(el));}}));}"
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
                  + "</head><body><div class=\"wrap\"><header class=\"page-header\"><div class=\"brandbar\">"
                  + (f'<img src="{logo_uri}" alt="Company Logo"/>' if logo_uri else "")
                  + '<span class="brand-name">ToppyMicroServices</span><span class="brand-tag">Research dashboard</span></div><div class="page-hero"><div><span class="page-kicker">Thermo-credit monitor</span><h1>Regional Credit Thermodynamics</h1><p class="page-subtitle">Monthly indicators for dispersion, liquidity state, loop-path monitoring, and credit exergy.</p></div></div></header>')
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
               f"<link>{base_url}/</link><description>Monthly thermo-credit indicators: S_M, T_L, streaming loop area, X_C.</description>"
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
