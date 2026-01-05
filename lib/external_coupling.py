"""Utilities for constructing external pressure/temperature indices.

The helper builds monthly indices by z-scoring individual driver series and
averaging them with equal weights (skip NaNs so missing drivers do not backfill).
"""
from __future__ import annotations

import logging
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)
SeriesFetcher = Callable[[str, Optional[str]], pd.DataFrame]


def _to_monthly(series_df: pd.DataFrame, freq: str) -> pd.Series:
    if series_df is None or series_df.empty:
        return pd.Series(dtype=float)
    df = series_df.copy()
    if "date" not in df.columns:
        raise ValueError("series dataframe must contain a 'date' column")
    df["date"] = pd.to_datetime(df["date"])
    values = pd.to_numeric(df["value"], errors="coerce")
    ser = pd.Series(values.values, index=df["date"])
    monthly = ser.resample(freq).mean()
    return monthly


def _zscore(series: pd.Series) -> pd.Series:
    if series.empty:
        return series
    mean = series.mean(skipna=True)
    std = series.std(skipna=True, ddof=0)
    if std is None or np.isnan(std) or std == 0:
        return series * np.nan
    return (series - mean) / std


def _transform_series(
    primary: pd.Series,
    transform: str,
    secondary: Optional[pd.Series] = None,
) -> pd.Series:
    transform = (transform or "value").strip().lower()
    ser = primary.copy()
    if transform == "spread":
        if secondary is None:
            raise ValueError("spread transform requires secondary series")
        ser = ser - secondary
    elif transform == "diff":
        ser = ser.diff()
    elif transform == "pct_change":
        ser = ser.pct_change()
    elif transform == "log_return":
        ser = np.log(ser.where(ser > 0)).diff()
    else:  # value
        ser = ser
    return ser


def _component_series(
    spec: Dict[str, Any],
    fetcher: SeriesFetcher,
    freq: str,
    cache: Dict[Tuple[str, Optional[str]], pd.DataFrame],
) -> pd.Series:
    series_id = spec.get("id")
    if not series_id:
        return pd.Series(dtype=float)
    start_override = spec.get("start")
    cache_key = (series_id, start_override)
    if cache_key not in cache:
        try:
            cache[cache_key] = fetcher(series_id, start_override)
        except Exception as exc:
            logger.warning("Failed to fetch %s: %s", series_id, exc)
            cache[cache_key] = pd.DataFrame()
    primary_df = cache[cache_key]
    primary = _to_monthly(primary_df, freq)
    secondary = None
    secondary_id = spec.get("id_b")
    if secondary_id:
        sec_key = (secondary_id, spec.get("start_b") or start_override)
        if sec_key not in cache:
            try:
                cache[sec_key] = fetcher(secondary_id, spec.get("start_b") or start_override)
            except Exception as exc:
                logger.warning("Failed to fetch %s: %s", secondary_id, exc)
                cache[sec_key] = pd.DataFrame()
        secondary = _to_monthly(cache[sec_key], freq)
    transform = spec.get("transform", "value")
    ser = _transform_series(primary, transform, secondary)
    if spec.get("scale") is not None:
        ser = ser * float(spec.get("scale"))
    key = spec.get("key") or series_id
    ser.name = str(key)
    return _zscore(ser)


def _build_group(
    specs: List[Dict[str, Any]],
    fetcher: SeriesFetcher,
    freq: str,
    cache: Dict[Tuple[str, Optional[str]], pd.DataFrame],
) -> Tuple[pd.DataFrame, List[str]]:
    frames = []
    keys: List[str] = []
    for spec in specs or []:
        series = _component_series(spec, fetcher, freq, cache)
        if series.empty:
            continue
        frames.append(series)
        keys.append(series.name)
    if not frames:
        return pd.DataFrame(), []
    df = pd.concat(frames, axis=1, join="outer")
    # Keep a plain DatetimeIndex to avoid pandas Period freq limitations (e.g., 'MS').
    df.index = pd.to_datetime(df.index).tz_localize(None)
    return df.sort_index(), keys


def build_external_coupling_indices(
    ext_cfg: Dict[str, Any],
    fetcher: SeriesFetcher,
) -> pd.DataFrame:
    """Return monthly external pressure/temperature indices.

    Parameters
    ----------
    ext_cfg : Dict
        Configuration dictionary (typically config["external_coupling"]).
    fetcher : Callable[[series_id, start], DataFrame]
        Function returning a DataFrame with columns ["date", "value"].
    """
    if not isinstance(ext_cfg, dict) or not ext_cfg.get("enabled"):
        return pd.DataFrame()
    freq = str(ext_cfg.get("frequency", "MS")).upper()
    cache: Dict[Tuple[str, Optional[str]], pd.DataFrame] = {}
    pressure_specs = ext_cfg.get("pressure_components", [])
    temperature_specs = ext_cfg.get("temperature_components", [])
    pressure_df, pressure_keys = _build_group(pressure_specs, fetcher, freq, cache)
    temp_df, temp_keys = _build_group(temperature_specs, fetcher, freq, cache)
    frames = [df for df in (pressure_df, temp_df) if not df.empty]
    if not frames:
        return pd.DataFrame()
    monthly = pd.concat(frames, axis=1, join="outer").sort_index()
    if pressure_keys:
        monthly["E_p"] = monthly[pressure_keys].mean(axis=1, skipna=True)
    if temp_keys:
        monthly["E_T"] = monthly[temp_keys].mean(axis=1, skipna=True)
    monthly = monthly.reset_index().rename(columns={"index": "date"})
    monthly["date"] = pd.to_datetime(monthly["date"]).dt.tz_localize(None)
    return monthly

__all__ = ["build_external_coupling_indices"]
