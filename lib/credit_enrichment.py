"""Shared credit enrichment logic for Thermo-Credit.

Centralizes computation of depth and turnover enrichment metrics so region scripts (JP/EU/US) can apply consistent heuristics
and warning diagnostics.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

DEFAULT_DEPTH_FALLBACK = 1000.0
DEFAULT_TURNOVER_FALLBACK = 1.0

def _cal(value, default):
    return default if value is None else value


def compute_enrichment(
    df: pd.DataFrame,
    depth_source: pd.Series | None = None,
    turnover_source: pd.Series | None = None,
    l_real_col: str = "L_real",
    u_col: str = "U",
    depth_col: str = "depth",
    turnover_col: str = "turnover",
    warnings: list | None = None,
    depth_scale: float | None = None,
    depth_fallback: float | None = None,
    turnover_min: float | None = None,
    turnover_max: float | None = None,
    turnover_fallback: float | None = None,
    clip_warn_threshold: float | None = None,
) -> pd.DataFrame:
    """Compute depth & turnover columns with fallback and clipping diagnostics.

    Parameters
    ----------
    df : DataFrame containing at least L_real and U columns.
    depth_source : Optional series with depth stock values aligned by date.
    turnover_source : Optional series with turnover values or liquidity proxy.
    l_real_col : Name of real credit stock column.
    u_col : Name of internal energy / capacity proxy column.
    warnings : list collecting warning strings.

    Returns
    -------
    Updated DataFrame with depth and turnover columns.
    """
    out = df.copy()
    if warnings is None:
        warnings = []

    # Resolve calibration parameters
    depth_scale = _cal(depth_scale, 4.0)
    depth_fallback = _cal(depth_fallback, DEFAULT_DEPTH_FALLBACK)
    turnover_min = _cal(turnover_min, 0.1)
    turnover_max = _cal(turnover_max, 10.0)
    turnover_fallback = _cal(turnover_fallback, DEFAULT_TURNOVER_FALLBACK)
    clip_warn_threshold = _cal(clip_warn_threshold, 0.15)

    # Treat all-NaN sources as missing
    if depth_source is not None and getattr(depth_source, 'notna', None) is not None:
        if depth_source.notna().sum() == 0:
            depth_source = None
    if turnover_source is not None and getattr(turnover_source, 'notna', None) is not None:
        if turnover_source.notna().sum() == 0:
            turnover_source = None

    # Depth
    if depth_source is not None:
        out[depth_col] = depth_source
    else:
        # heuristic scaling fallback using depth_scale
        l_series = out[l_real_col].astype(float)
        # If all zeros or NaN, assign uniform DEFAULT_DEPTH_FALLBACK
        if l_series.notna().sum() == 0 or l_series.abs().sum() == 0:
            out[depth_col] = depth_fallback
        else:
            med = l_series.median() or 1.0
            scaled = l_series * depth_fallback / med
            out[depth_col] = scaled.fillna(depth_fallback)

    # Turnover heuristic: U / L_real (capacity over stock)
    if turnover_source is not None:
        out[turnover_col] = turnover_source
    else:
        with np.errstate(divide="ignore", invalid="ignore"):
            base_ratio = out[u_col].astype(float) / out[l_real_col].replace({0: np.nan}).astype(float)
    out[turnover_col] = base_ratio.replace([np.inf, -np.inf], np.nan).fillna(turnover_fallback)

    # Clip turnover to sane bounds
    clipped_low = out[turnover_col] < turnover_min
    clipped_high = out[turnover_col] > turnover_max
    out.loc[clipped_low, turnover_col] = turnover_min
    out.loc[clipped_high, turnover_col] = turnover_max
    total_rows = len(out)
    if total_rows:
        frac_clipped = (clipped_low.sum() + clipped_high.sum()) / total_rows
        if frac_clipped > clip_warn_threshold:
            warnings.append(
                f"Turnover clipping applied to {frac_clipped:.1%} of rows (>{clip_warn_threshold:.0%} threshold)."
            )
    return out


def merge_depth_turnover(
    credit_df: pd.DataFrame,
    depth_df: pd.DataFrame | None = None,
    turnover_df: pd.DataFrame | None = None,
    date_col: str = "date",
    depth_value_col: str = "value",
    turnover_value_col: str = "value",
    warnings: list | None = None,
) -> pd.DataFrame:
    """Merge external depth/turnover raw series (already quarterly) if provided."""
    out = credit_df.copy()
    if depth_df is not None:
        d = depth_df[[date_col, depth_value_col]].rename(columns={depth_value_col: "depth_real"})
        out = out.merge(d, on=date_col, how="left")
    if turnover_df is not None:
        t = turnover_df[[date_col, turnover_value_col]].rename(columns={turnover_value_col: "turnover_real"})
        out = out.merge(t, on=date_col, how="left")
    return out

__all__ = [
    "compute_enrichment",
    "merge_depth_turnover",
]
