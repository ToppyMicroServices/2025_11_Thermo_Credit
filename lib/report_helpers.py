import os
import json
import html as html_lib
import textwrap
from typing import Any, Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from lib.indicators import compute_diagnostics

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
EVENT_CATEGORY_COLORS = {
    "bubble": "#f59e0b",
    "crisis": "#ef4444",
    "pandemic": "#0ea5e9",
    "policy": "#8b5cf6",
}

ChartSpec = Tuple[Any, str, str, Optional[str]]


class CompareData:
    """Structured container for compare-tab inputs (test friendly)."""

    def __init__(self, latest_rows: pd.DataFrame, raw_figs: List[ChartSpec], std_figs: List[ChartSpec]):
        self.latest_rows = latest_rows
        self.raw_figs = raw_figs
        self.std_figs = std_figs


class CompareBuilder:
    """Build reusable compare data for JP/EU/US dashboards."""

    def __init__(
        self,
        region_ctxs: Iterable[Dict[str, Any]],
        *,
        start_date: Optional[pd.Timestamp] = None,
        events: Optional[Iterable[Dict[str, Any]]] = None,
    ):
        self.region_ctxs = [ctx for ctx in region_ctxs if isinstance(ctx.get("frame"), pd.DataFrame)]
        self.start_date = start_date or _plot_start_date()
        self.events = list(events or [])

    def build(self) -> Optional[CompareData]:
        data = self._collect()
        if data is None:
            return None
        latest_rows, items = data
        raw_figs = self._build_raw_figs(items)
        std_figs = self._build_std_figs(items)
        latest_df = pd.DataFrame(latest_rows)
        return CompareData(latest_df, raw_figs, std_figs)

    def _collect(self) -> Optional[Tuple[List[Dict[str, Any]], List[Tuple[str, pd.DataFrame]]]]:
        if not self.region_ctxs:
            return None
        items: List[Tuple[str, pd.DataFrame]] = []
        latest_rows: List[Dict[str, Any]] = []
        metric_specs = [
            ("S_M", "S_M"),
            ("T_L", "T_L"),
            ("loop_area", "loop_area"),
            ("X_C", "X_C"),
        ]
        for ctx in self.region_ctxs:
            label = ctx.get("label")
            frame = ctx.get("frame")
            if not isinstance(label, str) or not isinstance(frame, pd.DataFrame) or frame.empty:
                continue
            items.append((label, frame))
            row: Dict[str, Any] = {"Region": label}
            if "date" in frame.columns:
                dlast = pd.to_datetime(frame["date"], errors="coerce").dropna()
                row["Latest date"] = dlast.iloc[-1].strftime("%Y-%m-%d") if not dlast.empty else ""
            for col, _ in metric_specs:
                if col not in frame.columns:
                    row[col] = None
                    continue
                try:
                    row[col] = float(pd.to_numeric(frame[col], errors="coerce").dropna().iloc[-1])
                except Exception:
                    row[col] = None
            latest_rows.append(row)
        if len(items) < 2:
            return None
        return latest_rows, items

    def _build_raw_figs(self, items: List[Tuple[str, pd.DataFrame]]) -> List[ChartSpec]:
        metric_specs = [
            ("S_M", "Compare – S_M", "Money entropy"),
            ("T_L", "Compare – T_L", "Liquidity temperature"),
            ("loop_area", "Compare – Policy Loop Dissipation", "Loop area"),
            ("X_C", "Compare – Credit Exergy Ceiling", "X_C"),
        ]
        raw_figs: List[ChartSpec] = []
        for met, title, alt in metric_specs:
            long_parts: List[pd.DataFrame] = []
            for label, df in items:
                if "date" not in df.columns:
                    continue
                col = met
                if met == "X_C":
                    col = None
                    if "X_C" in df.columns and pd.to_numeric(df["X_C"], errors="coerce").dropna().size > 0:
                        col = "X_C"
                    elif "F_C" in df.columns and pd.to_numeric(df["F_C"], errors="coerce").dropna().size > 0:
                        col = "F_C"
                    if col is None:
                        continue
                elif met not in df.columns:
                    continue
                part = df[["date", col]].copy()
                part = part[part["date"] >= self.start_date]
                part = part.rename(columns={col: "value"})
                part["Region"] = label
                long_parts.append(part)
            if not long_parts:
                continue
            long_df = pd.concat(long_parts, ignore_index=True)
            long_df["date"] = pd.to_datetime(long_df["date"], errors="coerce")
            long_df = long_df.dropna(subset=["date", "value"]).sort_values("date")
            if long_df.empty:
                continue
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
            apply_event_overlays(
                fig,
                filter_dashboard_events(
                    self.events,
                    start_date=long_df["date"].min(),
                    end_date=long_df["date"].max(),
                    global_only=True,
                ),
            )
            short_label = title.replace("Compare – ", "Compare: ")
            interp = _chart_interpretation(short_label, long_df)
            raw_figs.append((fig, short_label, alt, interp))
        return raw_figs

    def _build_std_figs(self, items: List[Tuple[str, pd.DataFrame]]) -> List[ChartSpec]:
        std_figs: List[ChartSpec] = []

        def _z_of(series: pd.Series) -> Optional[pd.Series]:
            s = pd.to_numeric(series, errors="coerce").dropna()
            if s.empty:
                return None
            m = float(s.mean())
            sd = float(s.std())
            if not np.isfinite(sd) or sd <= 0:
                return None
            return (series.astype(float) - m) / sd

        long_parts_hat: List[pd.DataFrame] = []
        for label, df in items:
            if "date" not in df.columns or "S_M_hat" not in df.columns:
                continue
            part = df[["date", "S_M_hat"]].copy()
            part = part[part["date"] >= self.start_date]
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
                apply_event_overlays(
                    fig_hat,
                    filter_dashboard_events(
                        self.events,
                        start_date=long_df_hat["date"].min(),
                        end_date=long_df_hat["date"].max(),
                        global_only=True,
                    ),
                )
                interp = _chart_interpretation("Compare: S_M_hat", long_df_hat)
                std_figs.append((fig_hat, "Compare: S_M_hat", "S_M_hat", interp))

        for met, title, alt in [
            ("T_L", "Compare – T_L (standardized)", "T_L z"),
            ("loop_area", "Compare – Loop area (standardized)", "Loop area z"),
            ("U", "Compare – Internal Energy (standardized)", "U z"),
            ("dU", "Compare – ΔU (standardized)", "dU z"),
            ("dF_C", "Compare – ΔF_C (standardized)", "dF_C z"),
            ("F_C", "Compare – Free Energy (standardized)", "F_C z"),
            ("X_C", "Compare – X_C (standardized)", "X_C z"),
        ]:
            long_parts_z: List[pd.DataFrame] = []
            for label, df in items:
                if "date" not in df.columns or met not in df.columns:
                    continue
                z = _z_of(df[met])
                if z is None:
                    continue
                part = pd.DataFrame({
                    "date": pd.to_datetime(df["date"], errors="coerce"),
                    "value": z,
                    "Region": label,
                })
                part = part.dropna(subset=["date", "value"])
                part = part[part["date"] >= self.start_date]
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
            apply_event_overlays(
                figz,
                filter_dashboard_events(
                    self.events,
                    start_date=long_df_z["date"].min(),
                    end_date=long_df_z["date"].max(),
                    global_only=True,
                ),
            )
            short_label = title.replace("Compare – ", "Compare: ")
            interp = _chart_interpretation(short_label, long_df_z)
            std_figs.append((figz, short_label, alt, interp))
        return std_figs


def _plot_start_date() -> pd.Timestamp:
    raw = os.getenv("REPORT_PLOT_START") or os.getenv("PLOT_START") or "1998-01-01"
    try:
        return pd.to_datetime(raw)
    except Exception:
        return pd.Timestamp("1998-01-01")


def load_dashboard_events(path: str) -> List[Dict[str, Any]]:
    """Load canonical dashboard event windows from CSV."""
    if not path or not os.path.exists(path):
        return []
    try:
        frame = pd.read_csv(path)
    except Exception:
        return []
    events: List[Dict[str, Any]] = []
    for row in frame.to_dict(orient="records"):
        label = str(row.get("label") or "").strip()
        if not label:
            continue
        start = pd.to_datetime(row.get("start_date"), errors="coerce")
        end = pd.to_datetime(row.get("end_date"), errors="coerce")
        if pd.isna(start):
            continue
        if pd.isna(end):
            end = start
        if end < start:
            start, end = end, start
        regions_raw = str(row.get("regions") or "all").strip().lower()
        regions = [part.strip() for part in regions_raw.split(",") if part.strip()] or ["all"]
        events.append(
            {
                "key": str(row.get("key") or "").strip(),
                "label": label,
                "start_date": start,
                "end_date": end,
                "regions": regions,
                "category": str(row.get("category") or "crisis").strip().lower(),
                "description": str(row.get("description") or "").strip(),
            }
        )
    return events


def filter_dashboard_events(
    events: Iterable[Dict[str, Any]],
    *,
    region_key: str = "",
    start_date: Optional[pd.Timestamp] = None,
    end_date: Optional[pd.Timestamp] = None,
    global_only: bool = False,
) -> List[Dict[str, Any]]:
    """Return events that overlap the requested region and date window."""
    filtered: List[Dict[str, Any]] = []
    norm_region = str(region_key or "").strip().lower()
    norm_start = pd.to_datetime(start_date, errors="coerce") if start_date is not None else None
    norm_end = pd.to_datetime(end_date, errors="coerce") if end_date is not None else None
    for raw in events:
        regions = [str(part).strip().lower() for part in raw.get("regions") or [] if str(part).strip()]
        if not regions:
            regions = ["all"]
        is_global = "all" in regions
        if global_only and not is_global:
            continue
        if not global_only and norm_region and not (is_global or norm_region in regions):
            continue
        start = pd.to_datetime(raw.get("start_date"), errors="coerce")
        end = pd.to_datetime(raw.get("end_date"), errors="coerce")
        if pd.isna(start) or pd.isna(end):
            continue
        if norm_start is not None and end < norm_start:
            continue
        if norm_end is not None and start > norm_end:
            continue
        visible_start = max(start, norm_start) if norm_start is not None else start
        visible_end = min(end, norm_end) if norm_end is not None else end
        enriched = dict(raw)
        enriched["visible_start"] = visible_start
        enriched["visible_end"] = visible_end
        filtered.append(enriched)
    return filtered


def apply_event_overlays(fig: Any, events: Iterable[Dict[str, Any]]) -> None:
    """Shade canonical event windows on a Plotly figure."""
    for event in events:
        x0 = pd.to_datetime(event.get("visible_start"), errors="coerce")
        x1 = pd.to_datetime(event.get("visible_end"), errors="coerce")
        if pd.isna(x0) or pd.isna(x1):
            continue
        category = str(event.get("category") or "crisis").strip().lower()
        color = EVENT_CATEGORY_COLORS.get(category, "#94a3b8")
        try:
            fig.add_vrect(
                x0=x0,
                x1=x1,
                fillcolor=color,
                opacity=0.10,
                line_width=0,
                layer="below",
                annotation_text=str(event.get("label") or ""),
                annotation_position="top left",
                annotation=dict(font=dict(size=10, color="#334155")),
            )
        except Exception:
            continue


def build_event_summary_html(events: Iterable[Dict[str, Any]], *, plot_start: Optional[pd.Timestamp] = None) -> str:
    """Render a compact HTML summary of the canonical event registry."""
    normalized = list(events)
    if not normalized:
        return ""
    cards: List[str] = []
    for event in normalized:
        start = pd.to_datetime(event.get("start_date"), errors="coerce")
        end = pd.to_datetime(event.get("end_date"), errors="coerce")
        if pd.isna(start) or pd.isna(end):
            continue
        regions = event.get("regions") or ["all"]
        scope = "Global" if "all" in regions else "/".join(str(part).upper() for part in regions)
        category = str(event.get("category") or "crisis").strip().lower()
        tone = html_lib.escape(category)
        date_text = start.strftime("%Y-%m-%d")
        if end != start:
            date_text += " to " + end.strftime("%Y-%m-%d")
        description = str(event.get("description") or "").strip()
        cards.append(
            "<article class=\"event-card\">"
            f"<div class=\"event-head\"><span class=\"event-chip tone-{tone}\">{html_lib.escape(scope)}</span>"
            f"<span class=\"event-chip tone-{tone}\">{html_lib.escape(category.title())}</span></div>"
            f"<strong class=\"event-title\">{html_lib.escape(str(event.get('label') or ''))}</strong>"
            f"<span class=\"event-date\">{html_lib.escape(date_text)}</span>"
            f"<p class=\"event-note\">{html_lib.escape(description)}</p>"
            "</article>"
        )
    if not cards:
        return ""
    lead = "<p class=\"note small\">Named event bands are drawn from a shared registry and overlaid on charts whenever the event window overlaps the visible period.</p>"
    if plot_start is not None:
        try:
            lead = (
                "<p class=\"note small\">Named event bands are drawn from a shared registry and overlaid on charts whenever the event window overlaps the visible period. "
                f"This report currently starts at {html_lib.escape(pd.to_datetime(plot_start).strftime('%Y-%m-%d'))}.</p>"
            )
        except Exception:
            pass
    return '<section class="event-summary"><h2>Reference events</h2>' + lead + f'<div class="event-grid">{"".join(cards)}</div></section>'


def render_dashboard_events_tex(
    events: Iterable[Dict[str, Any]],
    *,
    source_path: str = "data/report_events.csv",
) -> str:
    """Render the canonical event registry as a LaTeX snippet."""
    normalized = list(events)
    lines = [
        "% Auto-generated by scripts/03_make_report.py. Do not edit by hand.",
        r"\noindent\textit{Canonical event windows used by the dashboard overlay layer. Source: \texttt{"
        + latex_escape(source_path)
        + "}.}",
        "",
        r"\begin{itemize}",
    ]
    for event in normalized:
        start = pd.to_datetime(event.get("start_date"), errors="coerce")
        end = pd.to_datetime(event.get("end_date"), errors="coerce")
        if pd.isna(start) or pd.isna(end):
            continue
        scope = "global" if "all" in (event.get("regions") or []) else "/".join(str(part).upper() for part in event.get("regions") or [])
        window = start.strftime("%Y-%m-%d")
        if end != start:
            window += " to " + end.strftime("%Y-%m-%d")
        body = f"{event.get('label', '')} [{scope}; {window}]"
        description = str(event.get("description") or "").strip()
        if description:
            body += f": {description}"
        lines.append("  \\item " + latex_escape(body))
    lines.append(r"\end{itemize}")
    return "\n".join(lines).strip() + "\n"


def _style_figure(fig) -> None:
    fig.update_layout(
        template="plotly_white",
        hovermode="x unified",
        legend=dict(orientation="h", y=1.02, yanchor="bottom", x=1.0, xanchor="right"),
        margin=dict(t=60, b=40, l=40, r=20),
    )
    fig.update_xaxes(showgrid=False)
    fig.update_yaxes(showgrid=True, zeroline=True)
    fig.update_layout(font=dict(
        family="STIX Two Text, Times New Roman, Times, Georgia, serif",
        size=12,
    ))


def _apply_hover(fig, fmt: str) -> None:
    fig.update_traces(hovertemplate="%{x|%Y-%m-%d}<br>%{y:" + fmt + "}<extra>%{fullData.name}</extra>")


def build_report_style_block(brand_bg: str, brand_bg2: str, brand_text: str) -> str:
    """Return readable inline CSS for the generated dashboard HTML."""
    return textwrap.dedent(
        f"""
        :root {{
          --page-bg: #f6f8fb;
          --surface-bg: #ffffff;
          --surface-muted: #fafafa;
          --surface-emphasis: #eef2f7;
          --border-soft: #e5e7eb;
          --border-strong: #dce3f1;
          --text-main: #18212b;
          --text-muted: #4b5563;
          --text-soft: #6b7280;
          --tab-bg: #f8fafc;
          --tab-border: #94a3b8;
          --tab-active-bg: #1f2937;
          --tab-active-text: #ffffff;
          --shadow-soft: 0 10px 30px rgba(15, 23, 42, 0.05);
          --radius-lg: 14px;
          --radius-md: 10px;
          --radius-sm: 8px;
          --brand-bg: {brand_bg};
          --brand-bg2: {brand_bg2};
          --brand-text: {brand_text};
        }}

        /* Base */
        html {{
          background: var(--page-bg);
        }}

        body {{
          margin: 1.25rem;
          background: var(--page-bg);
          color: var(--text-main);
          font-family: system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
          line-height: 1.55;
        }}

        h1 {{
          margin: 0;
          font-size: 1.9rem;
          line-height: 1.15;
        }}

        h2 {{
          margin: 0 0 0.75rem;
          font-size: 1.08rem;
          line-height: 1.25;
        }}

        p, ul {{
          margin-top: 0;
        }}

        figure {{
          margin: 1rem 0;
        }}

        figcaption {{
          color: var(--text-muted);
          font-size: 0.8rem;
        }}

        /* Layout */
        .wrap {{
          max-width: 1120px;
          margin: 0 auto;
        }}

        .page-header {{
          display: grid;
          gap: 0.85rem;
          margin-bottom: 1rem;
        }}

        .page-hero {{
          display: grid;
          gap: 0.35rem;
        }}

        .page-subtitle {{
          margin: 0;
          color: var(--text-muted);
        }}

        .page-content {{
          display: grid;
          gap: 1rem;
        }}

        /* Shared panel styling */
        .region-summary,
        .intro,
        .inputs-summary,
        .coverage-summary,
        .event-summary {{
          background: var(--surface-bg);
          border: 1px solid var(--border-soft);
          border-radius: var(--radius-lg);
          box-shadow: var(--shadow-soft);
          padding: 0.95rem 1rem;
        }}

        .intro {{
          background: var(--surface-emphasis);
          border-color: #dde4ee;
        }}

        .note {{
          margin: 0.5rem 0 1rem;
          color: var(--text-muted);
        }}

        .note.small {{
          color: var(--text-soft);
          font-size: 0.85rem;
        }}

        .coverage-grid,
        .event-grid,
        .summary-grid,
        .chart-grid {{
          display: grid;
          gap: 0.8rem;
        }}

        .coverage-grid {{
          grid-template-columns: repeat(auto-fit, minmax(220px, 1fr));
          margin-top: 0.85rem;
        }}

        .event-grid {{
          grid-template-columns: repeat(auto-fit, minmax(220px, 1fr));
          margin-top: 0.85rem;
        }}

        .summary-grid {{
          grid-template-columns: repeat(auto-fit, minmax(148px, 1fr));
          margin: 0 0 0.9rem;
        }}

        .coverage-card,
        .event-card,
        .summary-card {{
          border: 1px solid var(--border-soft);
          border-radius: var(--radius-md);
          background: linear-gradient(180deg, #ffffff, #f8fafc);
          padding: 0.8rem 0.9rem;
        }}

        .coverage-card.tone-current,
        .summary-card.tone-current {{
          border-color: #b7e4c7;
          background: linear-gradient(180deg, #ffffff, #f2fbf5);
        }}

        .coverage-card.tone-delayed,
        .summary-card.tone-delayed {{
          border-color: #fde68a;
          background: linear-gradient(180deg, #ffffff, #fffaf0);
        }}

        .coverage-card.tone-stale,
        .summary-card.tone-stale {{
          border-color: #fecaca;
          background: linear-gradient(180deg, #ffffff, #fff3f3);
        }}

        .coverage-head {{
          display: flex;
          align-items: center;
          justify-content: space-between;
          gap: 0.75rem;
          margin-bottom: 0.45rem;
        }}

        .coverage-region,
        .summary-label {{
          color: var(--text-muted);
          font-size: 0.76rem;
          font-weight: 600;
          letter-spacing: 0.02em;
          text-transform: uppercase;
        }}

        .coverage-date,
        .summary-value {{
          display: block;
          color: var(--text-main);
          font-size: 1.02rem;
          line-height: 1.2;
        }}

        .coverage-note,
        .event-note,
        .summary-detail {{
          display: block;
          margin-top: 0.25rem;
          color: var(--text-soft);
          font-size: 0.78rem;
          line-height: 1.4;
        }}

        .event-head {{
          display: flex;
          flex-wrap: wrap;
          gap: 0.35rem;
          margin-bottom: 0.45rem;
        }}

        .event-title {{
          display: block;
          color: var(--text-main);
          font-size: 0.98rem;
          line-height: 1.3;
        }}

        .event-date {{
          display: block;
          margin-top: 0.2rem;
          color: var(--text-muted);
          font-size: 0.8rem;
        }}

        .event-chip {{
          display: inline-flex;
          align-items: center;
          border-radius: 999px;
          padding: 0.16rem 0.52rem;
          font-size: 0.7rem;
          font-weight: 700;
          border: 1px solid transparent;
          background: #e2e8f0;
          color: #334155;
        }}

        .event-chip.tone-bubble {{
          background: #fef3c7;
          border-color: #fcd34d;
          color: #92400e;
        }}

        .event-chip.tone-crisis {{
          background: #fee2e2;
          border-color: #fca5a5;
          color: #991b1b;
        }}

        .event-chip.tone-pandemic {{
          background: #dbeafe;
          border-color: #93c5fd;
          color: #1d4ed8;
        }}

        .event-chip.tone-policy {{
          background: #ede9fe;
          border-color: #c4b5fd;
          color: #6d28d9;
        }}

        .status-badge {{
          display: inline-flex;
          align-items: center;
          border-radius: 999px;
          padding: 0.18rem 0.55rem;
          font-size: 0.72rem;
          font-weight: 700;
          border: 1px solid transparent;
          background: #e5e7eb;
          color: #374151;
        }}

        .status-badge.tone-current {{
          background: #dcfce7;
          border-color: #86efac;
          color: #166534;
        }}

        .status-badge.tone-delayed {{
          background: #fef3c7;
          border-color: #fcd34d;
          color: #92400e;
        }}

        .status-badge.tone-stale {{
          background: #fee2e2;
          border-color: #fca5a5;
          color: #991b1b;
        }}

        /* Data tables */
        table.mini {{
          width: 100%;
          border-collapse: collapse;
          margin: 0.5rem 0;
          background: transparent;
        }}

        table.mini th,
        table.mini td {{
          padding: 0.42rem 0.58rem;
          border-bottom: 1px solid var(--border-soft);
          text-align: right;
          vertical-align: top;
        }}

        table.mini th:first-child,
        table.mini td:first-child {{
          text-align: left;
        }}

        /* Navigation controls */
        .tabs,
        .subtabs {{
          display: flex;
          flex-wrap: wrap;
          gap: 0.5rem;
        }}

        .tabs {{
          margin: 0.25rem 0 0.75rem;
        }}

        .subtabs {{
          margin: 0.35rem 0 0.8rem;
        }}

        .tabs button,
        .subtabs button {{
          border: 1px solid var(--tab-border);
          background: var(--tab-bg);
          color: var(--text-main);
          border-radius: 999px;
          cursor: pointer;
          transition: background 120ms ease, color 120ms ease, border-color 120ms ease;
        }}

        .tabs button {{
          padding: 0.48rem 0.9rem;
          font-size: 0.82rem;
        }}

        .subtabs button {{
          padding: 0.34rem 0.72rem;
          font-size: 0.78rem;
        }}

        .tabs button.active,
        .subtabs button.active {{
          border-color: var(--tab-active-bg);
          background: var(--tab-active-bg);
          color: var(--tab-active-text);
        }}

        .compare-block .pane,
        .region {{
          display: none;
        }}

        .compare-block .pane.active,
        .region.active {{
          display: block;
        }}

        .chart-grid {{
          grid-template-columns: repeat(auto-fit, minmax(320px, 1fr));
          margin-top: 1rem;
        }}

        figure.chart-card {{
          margin: 0;
          padding: 0.75rem;
          background: var(--surface-bg);
          border: 1px solid var(--border-soft);
          border-radius: var(--radius-lg);
          box-shadow: var(--shadow-soft);
        }}

        .chart-card .plotly-graph-div {{
          min-height: 320px;
        }}

        .chart-card figcaption {{
          margin-top: 0.55rem;
        }}

        /* Narrative helpers */
        .chart-notes {{
          margin: 0.8rem 0;
          padding: 0.5rem 0.72rem;
          background: #f1f4fb;
          border: 1px solid var(--border-strong);
          border-radius: var(--radius-sm);
        }}

        .chart-note {{
          display: flex;
          flex-direction: column;
          gap: 0.12rem;
          margin: 0.28rem 0;
          font-size: 0.82rem;
        }}

        .chart-note strong {{
          color: #1b2a43;
          font-weight: 600;
        }}

        .chart-note span,
        .chart-note-inline {{
          color: var(--text-muted);
          font-size: 0.78rem;
        }}

        details {{
          margin: 0.6rem 0;
        }}

        details > summary {{
          cursor: pointer;
          font-weight: 600;
          list-style: none;
        }}

        details > summary::-webkit-details-marker {{
          display: none;
        }}

        /* Input pills */
        .inputs-row {{
          margin: 0.4rem 0;
        }}

        .inputs-summary .region-tag {{
          display: inline-block;
          margin-right: 0.45rem;
          padding: 0.18rem 0.45rem;
          border-radius: 999px;
          background: #1f2937;
          color: #ffffff;
          font-size: 0.74rem;
          font-weight: 600;
        }}

        .inputs-summary .pill-list {{
          display: inline;
        }}

        .inputs-summary .pill {{
          display: inline-block;
          margin: 0.15rem 0.25rem;
          padding: 0.18rem 0.56rem;
          border: 1px solid var(--border-soft);
          border-radius: 999px;
          background: var(--surface-bg);
          font-size: 0.75rem;
        }}

        /* Branding */
        .brandbar,
        .footer-brand {{
          display: flex;
          align-items: center;
          gap: 10px;
          color: var(--brand-text);
          background: linear-gradient(90deg, var(--brand-bg), var(--brand-bg2));
        }}

        .brandbar {{
          padding: 0.6rem 0.85rem;
          border-radius: var(--radius-lg);
        }}

        .brandbar img,
        .footer-brand img {{
          width: auto;
          border-radius: 6px;
          box-shadow: 0 0 0 1px rgba(255, 255, 255, 0.2);
        }}

        .brandbar img {{
          height: 40px;
        }}

        .brandbar .brand-name {{
          font-size: 1rem;
          font-weight: 600;
          color: var(--brand-text);
        }}

        .footer-brand {{
          margin-top: 1.5rem;
          padding: 0.75rem 0.85rem;
          border-radius: var(--radius-lg);
          font-size: 0.75rem;
        }}

        .footer-brand img {{
          height: 32px;
        }}

        /* Responsive */
        @media (max-width: 720px) {{
          body {{
            margin: 0.7rem;
          }}

          h1 {{
            font-size: 1.55rem;
          }}

          .region-summary,
          .intro,
          .inputs-summary,
          .coverage-summary,
          .event-summary,
          .brandbar,
          .footer-brand {{
            padding-left: 0.8rem;
            padding-right: 0.8rem;
          }}

          .chart-grid {{
            grid-template-columns: 1fr;
          }}

          table.mini {{
            display: block;
            overflow-x: auto;
          }}
        }}
        """
    ).strip()


def build_report_script_block() -> str:
    """Return readable inline JS for dashboard tab toggles."""
    return textwrap.dedent(
        """
        <script>
        (function () {
          const tabs = Array.from(document.querySelectorAll(".tabs button"));
          if (tabs.length) {
            tabs.forEach((button) => {
              button.addEventListener("click", () => {
                tabs.forEach((item) => item.classList.remove("active"));
                button.classList.add("active");
                const targetId = button.getAttribute("data-target");
                document.querySelectorAll(".region").forEach((region) => {
                  region.classList.remove("active");
                });
                const target = document.getElementById("region-" + targetId);
                if (target) {
                  target.classList.add("active");
                }
              });
            });
          }

          document.querySelectorAll(".compare-toggle").forEach((toggle) => {
            const buttons = Array.from(toggle.querySelectorAll("button"));
            const block = toggle.parentElement ? toggle.parentElement.nextElementSibling : null;
            buttons.forEach((button) => {
              button.addEventListener("click", () => {
                buttons.forEach((item) => item.classList.remove("active"));
                button.classList.add("active");
                const mode = button.getAttribute("data-mode") === "std" ? "std" : "raw";
                if (!block) {
                  return;
                }
                block.querySelectorAll(".pane").forEach((pane) => {
                  pane.classList.remove("active");
                });
                const target = block.querySelector(".pane." + mode);
                if (target) {
                  target.classList.add("active");
                }
              });
            });
          });
        })();
        </script>
        """
    ).strip()


def latex_escape(text: Any) -> str:
    """Escape plain text for safe inclusion in LaTeX snippets."""
    replacements = {
        "\\": r"\textbackslash{}",
        "&": r"\&",
        "%": r"\%",
        "$": r"\$",
        "#": r"\#",
        "_": r"\_",
        "{": r"\{",
        "}": r"\}",
        "~": r"\textasciitilde{}",
        "^": r"\textasciicircum{}",
        "Δ": "Delta ",
        "≈": "approx. ",
        "−": "-",
        "→": "to",
    }
    return "".join(replacements.get(ch, ch) for ch in str(text))


def build_dashboard_takeaway_sections(region_ctxs: Iterable[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Normalize dashboard takeaway lines into reusable structured sections."""
    sections: List[Dict[str, Any]] = []
    for ctx in region_ctxs:
        label = str(ctx.get("label") or "").strip()
        if not label:
            continue
        latest_date = ""
        last_date = ctx.get("last_date")
        if hasattr(last_date, "strftime"):
            try:
                latest_date = last_date.strftime("%Y-%m-%d")
            except Exception:
                latest_date = ""
        bullets: List[str] = []
        for raw in ctx.get("takeaway_lines") or []:
            text = " ".join(str(raw).split())
            if text and text not in bullets:
                bullets.append(text)
        if not bullets:
            continue
        sections.append(
            {
                "key": str(ctx.get("key") or "").strip(),
                "label": label,
                "latest_date": latest_date,
                "bullets": bullets,
            }
        )
    return sections


def render_dashboard_takeaways_tex(
    sections: Iterable[Dict[str, Any]],
    *,
    source_path: str = "site/report.html",
    report_month: str = "",
) -> str:
    """Render dashboard takeaways as a LaTeX snippet that can be \\input{}."""
    normalized = list(sections)
    lines = [
        "% Auto-generated by scripts/03_make_report.py. Do not edit by hand.",
        r"\noindent\textit{Source: \texttt{"
        + latex_escape(source_path)
        + "}"
        + (f", snapshot {latex_escape(report_month)}" if report_month else "")
        + ".}",
        "",
    ]
    for section in normalized:
        label = latex_escape(section.get("label", ""))
        latest_date = latex_escape(section.get("latest_date", ""))
        heading = label + (f" (latest: {latest_date})" if latest_date else "")
        lines.append(r"\paragraph{" + heading + "}")
        lines.append(r"\begin{itemize}")
        for bullet in section.get("bullets", []):
            lines.append("  \\item " + latex_escape(bullet))
        lines.append(r"\end{itemize}")
        lines.append("")
    return "\n".join(lines).strip() + "\n"


def write_dashboard_takeaways_png(
    path: str,
    sections: Iterable[Dict[str, Any]],
    *,
    title: str = "Thermo-Credit Dashboard Takeaways",
    subtitle: str = "",
) -> bool:
    """Write a simple PNG summary card for dashboard takeaways."""
    try:
        from PIL import Image, ImageDraw, ImageFont
    except Exception:
        return False

    normalized = list(sections)
    if not normalized:
        return False

    def _load_font(size: int, *, bold: bool = False):
        candidates = [
            "/System/Library/Fonts/Supplemental/Helvetica.ttc",
            "/System/Library/Fonts/Supplemental/Arial Unicode.ttf",
            "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf" if bold else "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        ]
        if bold:
            candidates.insert(0, "/System/Library/Fonts/Supplemental/Arial Bold.ttf")
        else:
            candidates.insert(0, "/System/Library/Fonts/Supplemental/Arial.ttf")
        for candidate in candidates:
            if candidate and os.path.exists(candidate):
                try:
                    return ImageFont.truetype(candidate, size=size)
                except Exception:
                    continue
        return ImageFont.load_default()

    title_font = _load_font(34, bold=True)
    section_font = _load_font(24, bold=True)
    body_font = _load_font(20)
    meta_font = _load_font(18)

    width = 1600
    padding_x = 64
    padding_y = 54
    gutter = 16
    section_gap = 24
    line_height = 30

    blocks: List[Tuple[str, str]] = [("title", title)]
    if subtitle:
        blocks.append(("meta", subtitle))
    for section in normalized:
        heading = str(section.get("label", "")).strip()
        latest_date = str(section.get("latest_date", "")).strip()
        if latest_date:
            heading += f" ({latest_date})"
        blocks.append(("section", heading))
        for bullet in section.get("bullets", []):
            wrapped = textwrap.wrap(str(bullet), width=92) or [str(bullet)]
            first = True
            for line in wrapped:
                prefix = u"\u2022 " if first else "  "
                blocks.append(("body", prefix + line))
                first = False

    height = padding_y * 2
    for kind, _ in blocks:
        if kind == "title":
            height += 48 + gutter
        elif kind == "section":
            height += 34 + gutter
        elif kind == "meta":
            height += 28 + gutter
        else:
            height += line_height + 6
    height += section_gap * max(len(normalized) - 1, 0)

    image = Image.new("RGB", (width, height), color="#f6f8fb")
    draw = ImageDraw.Draw(image)

    # Header band
    draw.rounded_rectangle(
        (24, 24, width - 24, 172),
        radius=28,
        fill="#10243d",
    )
    y = 48
    x = padding_x
    for kind, text in blocks:
        if kind == "title":
            draw.text((x, y), text, fill="#ffffff", font=title_font)
            y += 48 + gutter
            continue
        if kind == "meta":
            draw.text((x, y), text, fill="#d8e4f2", font=meta_font)
            y = max(y + 28 + gutter, 188)
            continue
        break

    card_top = max(y + 8, 188)
    draw.rounded_rectangle(
        (24, card_top, width - 24, height - 24),
        radius=28,
        fill="#ffffff",
        outline="#d9e2ef",
        width=2,
    )

    y = card_top + 36
    for kind, text in blocks:
        if kind in {"title", "meta"}:
            continue
        if kind == "section":
            draw.text((x, y), text, fill="#10243d", font=section_font)
            y += 34 + gutter
            continue
        draw.text((x + 10, y), text, fill="#213042", font=body_font)
        y += line_height + 6
        if text.startswith("  "):
            continue
        # Small breathing room after each bullet block.
        y += 2

    os.makedirs(os.path.dirname(path), exist_ok=True)
    image.save(path, format="PNG", optimize=True)
    return True


def _latest_numeric(frame: Optional[pd.DataFrame], column: str) -> Optional[float]:
    if frame is None or not isinstance(frame, pd.DataFrame) or column not in frame.columns:
        return None
    try:
        vals = pd.to_numeric(frame[column], errors="coerce").dropna()
    except Exception:
        return None
    if vals.empty:
        return None
    val = float(vals.iloc[-1])
    return val if np.isfinite(val) else None


def _series_bucket(series: Optional[pd.Series], value: Optional[float] = None) -> Optional[str]:
    if series is None:
        return None
    try:
        s = pd.to_numeric(series, errors="coerce").dropna()
    except Exception:
        return None
    if s.size < 6:
        return None
    val = float(s.iloc[-1]) if value is None else float(value)
    if val is None or not np.isfinite(val):
        return None
    q1, q2 = float(s.quantile(0.33)), float(s.quantile(0.66))
    if not np.isfinite(q1) or not np.isfinite(q2):
        return None
    if val <= q1:
        return "low"
    if val >= q2:
        return "high"
    return "mid-range"


def _series_trend(series: Optional[pd.Series]) -> Optional[str]:
    if series is None:
        return None
    try:
        s = pd.to_numeric(series, errors="coerce").dropna()
    except Exception:
        return None
    if s.size < 3:
        return None
    tail = s.tail(4)
    if tail.size < 2:
        return None
    delta = float(tail.iloc[-1] - tail.iloc[0])
    if not np.isfinite(delta):
        return None
    if abs(delta) < 1e-6:
        return "flat"
    return "rising" if delta > 0 else "falling"


def _metric_phrase(metric: str) -> str:
    mapping = {
        "S_M": "dispersion",
        "T_L": "liquidity temperature",
        "loop_area": "loop dissipation",
        "X_C": "credit exergy",
        "S_M_hat": "normalized dispersion",
        "U": "internal energy",
        "dU": "ΔU",
        "dF_C": "ΔF_C",
        "F_C": "free energy",
    }
    key = metric.replace("(standardized)", "").strip()
    return mapping.get(key, metric)


def _compare_interpretation(short_label: str, frame: Optional[pd.DataFrame]) -> Optional[str]:
    if frame is None or not isinstance(frame, pd.DataFrame):
        return "Cross-region view; tighter clustering implies similar regimes."
    if not {"value", "Region"}.issubset(frame.columns):
        return "Cross-region view; tighter clustering implies similar regimes."
    data = frame.copy()
    data["value"] = pd.to_numeric(data["value"], errors="coerce")
    data = data.dropna(subset=["value"])
    if "date" in data.columns:
        data = data.sort_values("date")
    if data.empty:
        return "Cross-region view; tighter clustering implies similar regimes."
    latest = data.groupby("Region").tail(1)
    latest = latest.dropna(subset=["value"])
    if latest.empty or latest["Region"].nunique() < 2:
        return "Cross-region view; watch relative slopes."
    leader = latest.loc[latest["value"].idxmax()]
    laggard = latest.loc[latest["value"].idxmin()]
    if leader.get("Region") == laggard.get("Region"):
        return "Cross-region view; watch relative slopes."
    metric = short_label.replace("Compare:", "").strip()
    phrase = _metric_phrase(metric)
    return (
        f"{phrase.capitalize()} highest in {leader['Region']} (≈{leader['value']:.2f}) "
        f"vs {laggard['Region']} (≈{laggard['value']:.2f})."
    )


def _chart_interpretation(short_label: str, frame: Optional[pd.DataFrame]) -> Optional[str]:
    label = (short_label or "").strip()
    if not label:
        return None
    if label.startswith("Compare"):
        return _compare_interpretation(label, frame)
    if label == "Raw Inputs (first=100)":
        return "Each input series is rebased to 100 at its start; steep slopes flag faster money/credit growth."
    if frame is None or not isinstance(frame, pd.DataFrame) or frame.empty:
        return None

    def _bucket_text(col: str, val: Optional[float]) -> Optional[str]:
        bucket = _series_bucket(frame.get(col), val)
        return f"{val:.2f} ({bucket})" if val is not None and bucket else (f"{val:.2f}" if val is not None else None)

    if label == "S_M & T_L":
        sm = _latest_numeric(frame, "S_M")
        tl = _latest_numeric(frame, "T_L")
        if sm is None and tl is None:
            return None
        parts: List[str] = []
        sm_txt = _bucket_text("S_M", sm)
        if sm_txt:
            parts.append(f"S_M≈{sm_txt}")
        tl_txt = _bucket_text("T_L", tl)
        if tl_txt:
            parts.append(f"T_L≈{tl_txt}")
        suffix = " Balanced readings mean policy has room; high/high combos often precede overheating."
        return ", ".join(parts) + suffix if parts else None

    if label == "S_M by category":
        cat_cols = [c for c in frame.columns if c.startswith("S_M_in_")]
        if not cat_cols:
            return "Category stacking shows which lending blocks drive dispersion."
        try:
            latest = frame[cat_cols].apply(pd.to_numeric, errors="coerce").dropna(how="all").tail(1)
        except Exception:
            latest = pd.DataFrame()
        if latest.empty:
            return "Category stacking shows which lending blocks drive dispersion."
        row = latest.iloc[0]
        contribs: List[Tuple[str, float]] = []
        total = 0.0
        for col, val in row.items():
            if pd.isna(val):
                continue
            total += abs(float(val))
            key = col.replace("S_M_in_", "")
            contribs.append((CATEGORY_LABELS.get(key, key.replace("_", " ").title()), float(val)))
        if total <= 0 or not contribs:
            return "Category stacking shows which lending blocks drive dispersion."
        contribs.sort(key=lambda kv: abs(kv[1]), reverse=True)
        top_parts = [f"{name} {abs(val)/total:.0%}" for name, val in contribs[:2]]
        return "Latest dispersion split: " + ", ".join(top_parts) + "."

    if label == "Policy Loop Dissipation":
        val = _latest_numeric(frame, "loop_area")
        if val is None:
            return None
        trend = _series_trend(frame.get("loop_area"))
        state = "dissipating" if val > 0 else ("amplifying" if val < 0 else "quiet")
        tail = f" and {trend}" if trend else ""
        return f"Loop area ≈{val:.3f} ({state}{tail})."

    if label == "Credit Exergy Ceiling":
        val = _latest_numeric(frame, "X_C")
        if val is None:
            return "Tracks remaining credit headroom; above zero means slack remains."
        bucket = _series_bucket(frame.get("X_C"), val)
        trend = _series_trend(frame.get("X_C"))
        tone = "slack" if val > 0 else "tight"
        trend_txt = f", {trend}" if trend else ""
        bucket_txt = f" ({bucket})" if bucket else ""
        return f"X_C≈{val:.2f}{bucket_txt}{trend_txt} so headroom looks {tone}."

    if label == "Free Energy (F_C)":
        val = _latest_numeric(frame, "F_C")
        if val is None:
            return None
        trend = _series_trend(frame.get("F_C"))
        bucket = _series_bucket(frame.get("F_C"), val)
        bucket_txt = f" ({bucket})" if bucket else ""
        trend_txt = f" and {trend}" if trend else ""
        return f"F_C≈{val:.2f}{bucket_txt}{trend_txt}; falling values hint at demand destruction."

    if label == "ΔF_C (change)":
        val = _latest_numeric(frame, "dF_C")
        if val is None:
            return None
        direction = "releasing" if val > 0 else "absorbing" if val < 0 else "stable"
        trend = _series_trend(frame.get("dF_C"))
        tail = f" and {trend}" if trend else ""
        return f"ΔF_C≈{val:.3f}, so the system is {direction}{tail}."

    if label == "Internal Energy (U)":
        val = _latest_numeric(frame, "U")
        if val is None:
            return None
        bucket = _series_bucket(frame.get("U"), val)
        trend = _series_trend(frame.get("U"))
        parts = [f"U≈{val:.2f}"]
        if bucket:
            parts.append(f"{bucket}")
        if trend:
            parts.append(trend)
        return " / ".join(parts) + " potential stored in the system."

    if label == "Surplus/Shortage (ΔF_C)":
        plus = _latest_numeric(frame, "Surplus (X_C+)")
        minus = _latest_numeric(frame, "Shortage (X_C−)")
        if plus is None and minus is None:
            return None
        if plus is not None and minus is not None:
            dominance = "surplus" if plus > minus else "shortage" if minus > plus else "balanced"
            return f"Surplus≈{plus:.2f}, shortage≈{minus:.2f}; {dominance} dominates."
        if plus is not None:
            return f"Surplus≈{plus:.2f}; shortages muted."
        return f"Shortage≈{minus:.2f}; little positive slack left." if minus is not None else None

    if label == "Maxwell-like Test":
        gap = _latest_numeric(frame, "maxwell_gap")
        if gap is None:
            return "Comparing ∂S/∂V|T and ∂p/∂T|V; overlap means proxies agree."
        try:
            series = pd.to_numeric(frame.get("maxwell_gap"), errors="coerce").dropna()
        except Exception:
            series = pd.Series(dtype=float)
        mad = float((series - series.median()).abs().median()) if not series.empty else 0.0
        spec = "inside spec" if mad == 0 or abs(gap) <= 3 * mad else "out-of-spec"
        return f"Maxwell gap≈{gap:.3f} ({spec})."

    if label == "First-law Decomposition":
        resid = _latest_numeric(frame, "firstlaw_resid")
        if resid is None:
            return "Tracks ΔU versus predicted TΔS−pΔV contributions."
        trend = _series_trend(frame.get("firstlaw_resid"))
        trend_txt = f" trending {trend}" if trend and trend != "flat" else ""
        return f"Residual≈{resid:.3f}{trend_txt}; near zero means the proxies close the energy balance."

    return None


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


def make_dual_axis_sm_tl(plot_df: pd.DataFrame, title: str) -> go.Figure:
    df = plot_df.copy()
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df = df.dropna(subset=["date"])
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    col_sm = "#1f77b4"
    col_tl = "#ff7f0e"
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
    fig.update_layout(plot_bgcolor="#fbfbfc")
    fig.update_yaxes(showgrid=True, gridcolor="#e9ecef", zeroline=True)
    return fig


def _filter_placeholders(df: pd.DataFrame) -> pd.DataFrame:
    if "placeholder" in df.columns:
        try:
            mask = ~(df["placeholder"].astype(bool))
            return df[mask].copy()
        except Exception:
            return df
    return df


def _out_of_spec_mask(df: pd.DataFrame) -> pd.Series:
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
    ranges: List[Tuple[pd.Timestamp, pd.Timestamp]] = []
    if dates.empty or mask.empty or len(dates) != len(mask):
        return ranges
    current_start = None
    prev_dt = None
    for dt_val, flag in zip(dates, mask):
        if flag and current_start is None:
            current_start = dt_val
        elif not flag and current_start is not None:
            ranges.append((current_start, prev_dt))
            current_start = None
        prev_dt = dt_val
    if current_start is not None and prev_dt is not None:
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


def _figs_html(specs: List[ChartSpec]) -> str:
    if not specs:
        return ""
    parts: List[str] = []
    for fig, title, alt, interp in specs:
        html = fig.to_html(full_html=False, include_plotlyjs="cdn")
        caption = f"<strong>{html_lib.escape(title)}</strong>"
        if interp:
            caption += f"<span class=\"chart-note-inline\">{html_lib.escape(interp)}</span>"
        parts.append(
            f"<figure class=\"chart-card\" aria-label=\"{html_lib.escape(alt)}\">{html}<figcaption>{caption}</figcaption></figure>"
        )
    return "<div class=\"chart-grid\">" + "".join(parts) + "</div>"


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
