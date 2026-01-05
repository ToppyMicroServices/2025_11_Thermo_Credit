import html as html_lib
import json
import os
from collections.abc import Iterable
from typing import Any

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

ChartSpec = tuple[Any, str, str, str | None]


class CompareData:
    """Structured container for compare-tab inputs (test friendly)."""

    def __init__(self, latest_rows: pd.DataFrame, raw_figs: list[ChartSpec], std_figs: list[ChartSpec]):
        self.latest_rows = latest_rows
        self.raw_figs = raw_figs
        self.std_figs = std_figs


class CompareBuilder:
    """Build reusable compare data for JP/EU/US dashboards."""

    def __init__(self, region_ctxs: Iterable[dict[str, Any]], *, start_date: pd.Timestamp | None = None):
        self.region_ctxs = [ctx for ctx in region_ctxs if isinstance(ctx.get("frame"), pd.DataFrame)]
        self.start_date = start_date or _plot_start_date()

    def build(self) -> CompareData | None:
        data = self._collect()
        if data is None:
            return None
        latest_rows, items = data
        raw_figs = self._build_raw_figs(items)
        std_figs = self._build_std_figs(items)
        latest_df = pd.DataFrame(latest_rows)
        return CompareData(latest_df, raw_figs, std_figs)

    def _collect(self) -> tuple[list[dict[str, Any]], list[tuple[str, pd.DataFrame]]] | None:
        if not self.region_ctxs:
            return None
        items: list[tuple[str, pd.DataFrame]] = []
        latest_rows: list[dict[str, Any]] = []
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
            row: dict[str, Any] = {"Region": label}
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

    def _build_raw_figs(self, items: list[tuple[str, pd.DataFrame]]) -> list[ChartSpec]:
        metric_specs = [
            ("S_M", "Compare – S_M", "Money entropy"),
            ("T_L", "Compare – T_L", "Liquidity temperature"),
            ("loop_area", "Compare – Policy Loop Dissipation", "Loop area"),
            ("X_C", "Compare – Credit Exergy Ceiling", "X_C"),
        ]
        raw_figs: list[ChartSpec] = []
        for met, title, alt in metric_specs:
            long_parts: list[pd.DataFrame] = []
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
            short_label = title.replace("Compare – ", "Compare: ")
            interp = _chart_interpretation(short_label, long_df)
            raw_figs.append((fig, short_label, alt, interp))
        return raw_figs

    def _build_std_figs(self, items: list[tuple[str, pd.DataFrame]]) -> list[ChartSpec]:
        std_figs: list[ChartSpec] = []

        def _z_of(series: pd.Series) -> pd.Series | None:
            s = pd.to_numeric(series, errors="coerce").dropna()
            if s.empty:
                return None
            m = float(s.mean())
            sd = float(s.std())
            if not np.isfinite(sd) or sd <= 0:
                return None
            return (series.astype(float) - m) / sd

        long_parts_hat: list[pd.DataFrame] = []
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
            long_parts_z: list[pd.DataFrame] = []
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
            short_label = title.replace("Compare – ", "Compare: ")
            interp = _chart_interpretation(short_label, long_df_z)
            std_figs.append((figz, short_label, alt, interp))
        return std_figs


def _plot_start_date() -> pd.Timestamp:
    raw = os.getenv("REPORT_PLOT_START") or os.getenv("PLOT_START") or "2010-01-01"
    try:
        return pd.to_datetime(raw)
    except Exception:
        return pd.Timestamp("2010-01-01")


def _style_figure(fig) -> None:
    fig.update_layout(
        template="plotly_white",
        hovermode="x unified",
        legend={"orientation": "h", "y": 1.02, "yanchor": "bottom", "x": 1.0, "xanchor": "right"},
        margin={"t": 60, "b": 40, "l": 40, "r": 20},
    )
    fig.update_xaxes(showgrid=False)
    fig.update_yaxes(showgrid=True, zeroline=True)
    fig.update_layout(font={
        "family": "STIX Two Text, Times New Roman, Times, Georgia, serif",
        "size": 12,
    })


def _apply_hover(fig, fmt: str) -> None:
    fig.update_traces(hovertemplate="%{x|%Y-%m-%d}<br>%{y:" + fmt + "}<extra>%{fullData.name}</extra>")


def _latest_numeric(frame: pd.DataFrame | None, column: str) -> float | None:
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


def _series_bucket(series: pd.Series | None, value: float | None = None) -> str | None:
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


def _series_trend(series: pd.Series | None) -> str | None:
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


def _compare_interpretation(short_label: str, frame: pd.DataFrame | None) -> str | None:
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


def _chart_interpretation(short_label: str, frame: pd.DataFrame | None) -> str | None:
    label = (short_label or "").strip()
    if not label:
        return None
    if label.startswith("Compare"):
        return _compare_interpretation(label, frame)
    if label == "Raw Inputs (first=100)":
        return "Each input series is rebased to 100 at its start; steep slopes flag faster money/credit growth."
    if frame is None or not isinstance(frame, pd.DataFrame) or frame.empty:
        return None

    def _bucket_text(col: str, val: float | None) -> str | None:
        bucket = _series_bucket(frame.get(col), val)
        return f"{val:.2f} ({bucket})" if val is not None and bucket else (f"{val:.2f}" if val is not None else None)

    if label == "S_M & T_L":
        sm = _latest_numeric(frame, "S_M")
        tl = _latest_numeric(frame, "T_L")
        if sm is None and tl is None:
            return None
        parts: list[str] = []
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
        contribs: list[tuple[str, float]] = []
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


def _load_csv(path: str) -> pd.DataFrame | None:
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


def _load_json(path: str) -> dict[str, Any] | None:
    if not os.path.exists(path):
        return None
    try:
        with open(path, encoding="utf-8") as fp:
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


def _calc_effective_window(frame: pd.DataFrame, requested: int) -> tuple[int, str]:
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
                line={"color": col_sm, "width": 2.0},
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
                line={"color": col_tl, "width": 2.0, "dash": "solid"},
            ),
            secondary_y=True,
        )
    fig.update_layout(title=title, legend={"orientation": "h", "y": 1.02, "yanchor": "bottom", "x": 1.0, "xanchor": "right"})
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


def _mask_to_ranges(dates: pd.Series, mask: pd.Series) -> list[tuple[pd.Timestamp, pd.Timestamp]]:
    ranges: list[tuple[pd.Timestamp, pd.Timestamp]] = []
    if dates.empty or mask.empty or len(dates) != len(mask):
        return ranges
    current_start = None
    prev_dt = None
    for dt_val, flag in zip(dates, mask, strict=False):
        if flag and current_start is None:
            current_start = dt_val
        elif not flag and current_start is not None:
            ranges.append((current_start, prev_dt))
            current_start = None
        prev_dt = dt_val
    if current_start is not None and prev_dt is not None:
        ranges.append((current_start, prev_dt))
    return ranges


def _augment_region_frame(frame: pd.DataFrame, effective_window: int, has_thermo: bool) -> tuple[pd.DataFrame, bool]:
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


def _figs_html(specs: list[ChartSpec]) -> str:
    parts: list[str] = []
    for fig, title, alt, interp in specs:
        html = fig.to_html(full_html=False, include_plotlyjs="cdn")
        caption = f"<strong>{html_lib.escape(title)}</strong>"
        if interp:
            caption += f"<span class=\"chart-note-inline\">{html_lib.escape(interp)}</span>"
        parts.append(
            f"<figure aria-label=\"{html_lib.escape(alt)}\">{html}<figcaption>{caption}</figcaption></figure>"
        )
    return "".join(parts)


def _selected_table(meta: dict[str, Any] | None, header: str) -> str:
    if not isinstance(meta, dict):
        return ""
    rows: list[dict[str, Any]] = []
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


