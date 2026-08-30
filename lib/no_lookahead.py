from __future__ import annotations

import re
from typing import Any, Mapping

import pandas as pd


DEFAULT_RELEASE_LAG_GROUPS: dict[str, int] = {
    "gdp": 90,
    "credit": 120,
    "spread": 1,
    "market": 1,
    "money": 30,
    "allocation": 90,
    "destination": 90,
    "regulatory": 90,
}

GROUP_COLUMNS: dict[str, tuple[str, ...]] = {
    "gdp": ("U", "Y", "U_gdp_only"),
    "credit": ("L_real", "L_asset", "depth", "turnover"),
    "spread": ("spread",),
    "market": ("E_p", "E_T"),
    "money": ("M_in", "M_out"),
    "destination": (
        "C_t",
        "C_t_raw_delta",
        "C_G",
        "C_B",
        "C_E",
        "C_R",
        "C_A",
        "q_t",
        "one_minus_q_t",
        "destination_coverage",
        "destination_coverage_observed",
    ),
    "regulatory": ("p_R", "p_C", "V_R", "V_C", "capital_headroom", "lcr_headroom", "nsfr_headroom"),
}

PATTERN_GROUPS: tuple[tuple[str, str], ...] = (
    (r"^q_", "allocation"),
    (r"^share_.*_(direct|observed|proxy)$", "destination"),
    (r"^fixed_investment_", "destination"),
    (r"^(classified|total|unclassified)_positive_flow$", "destination"),
    (r".*_headroom$", "regulatory"),
)


def realtime_preprocessing_config(cfg: Mapping[str, Any] | None) -> dict[str, Any]:
    """Return the real-time preprocessing config with conservative defaults."""
    root = cfg if isinstance(cfg, Mapping) else {}
    pre = root.get("preprocessing", {})
    pre = pre if isinstance(pre, Mapping) else {}
    rt = pre.get("real_time_forecast", {})
    rt = rt if isinstance(rt, Mapping) else {}
    return {
        "enabled": bool(rt.get("enabled", True)),
        "default_lag_days": int(rt.get("default_lag_days", 0) or 0),
        "release_lags_days": rt.get("release_lags_days", {}),
        "profile": str(rt.get("profile", "default_release_lags")),
    }


def resolve_release_lags(cfg: Mapping[str, Any] | None) -> tuple[dict[str, int], int, str]:
    """Expand group-level lag settings into a column-level lag map."""
    rt = realtime_preprocessing_config(cfg)
    group_lags = dict(DEFAULT_RELEASE_LAG_GROUPS)
    column_lags: dict[str, int] = {}
    raw = rt.get("release_lags_days", {})
    if isinstance(raw, Mapping):
        for key, value in raw.items():
            try:
                lag = max(0, int(value))
            except Exception:
                continue
            key_str = str(key)
            if key_str in GROUP_COLUMNS or key_str in group_lags:
                group_lags[key_str] = lag
            else:
                column_lags[key_str] = lag
    for group, lag in group_lags.items():
        column_lags[f"__group__{group}"] = lag
    for group, cols in GROUP_COLUMNS.items():
        lag = int(group_lags.get(group, 0))
        for col in cols:
            column_lags.setdefault(col, lag)
    return column_lags, int(rt["default_lag_days"]), str(rt["profile"])


def release_lag_for_column(column: str, column_lags: Mapping[str, int], default_lag_days: int = 0) -> int:
    if column in column_lags:
        return max(0, int(column_lags[column]))
    for pattern, group in PATTERN_GROUPS:
        if re.match(pattern, column):
            return max(0, int(column_lags.get(f"__group__{group}", DEFAULT_RELEASE_LAG_GROUPS.get(group, default_lag_days))))
    return max(0, int(default_lag_days))


def apply_release_lags(
    frame: pd.DataFrame,
    column_lags: Mapping[str, int],
    *,
    default_lag_days: int = 0,
) -> pd.DataFrame:
    """Convert a dated panel into an as-of panel available at each row date.

    For a column with lag L, the value stamped at date d becomes usable only at
    d + L days. The returned row at forecast origin t therefore contains the
    most recent value whose availability date is <= t.
    """
    if frame is None or frame.empty or "date" not in frame.columns:
        return frame

    src = frame.copy(deep=True)
    src = src.assign(date=pd.to_datetime(src["date"], errors="coerce"))
    src = src.dropna(subset=["date"]).sort_values("date").reset_index(drop=True)
    if src.empty:
        return src

    origins = pd.DataFrame({"date": src["date"].drop_duplicates().sort_values()})
    out = origins.copy()
    for col in src.columns:
        if col == "date":
            continue
        lag_days = release_lag_for_column(col, column_lags, default_lag_days)
        if lag_days <= 0:
            values = src[["date", col]].drop_duplicates("date", keep="last")
            out = out.merge(values, on="date", how="left")
            continue

        available = src[["date", col]].dropna(subset=[col]).copy()
        if available.empty:
            out[col] = pd.NA
            continue
        available = available.assign(available_date=available["date"] + pd.to_timedelta(lag_days, unit="D"))
        available = available.sort_values("available_date").drop_duplicates("available_date", keep="last")
        merged = pd.merge_asof(
            origins,
            available[["available_date", col]],
            left_on="date",
            right_on="available_date",
            direction="backward",
        )
        out[col] = merged[col].to_numpy()
    return out


__all__ = [
    "apply_release_lags",
    "realtime_preprocessing_config",
    "release_lag_for_column",
    "resolve_release_lags",
]
