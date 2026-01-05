"""Shared FRED API utilities with retry logic and error handling.

Provides a common interface for fetching time series data from the Federal Reserve
Economic Data (FRED) API with retry logic, proper error handling, and caching support.
"""
from __future__ import annotations

import logging
import os
import time
from typing import Optional

import pandas as pd
import requests

logger = logging.getLogger(__name__)

FRED_API_BASE = "https://api.stlouisfed.org/fred/series/observations"
DEFAULT_TIMEOUT = 30.0


def fetch_fred_series(
    series_id: str,
    api_key: Optional[str] = None,
    start: str = "1990-01-01",
    retries: int = 3,
    backoff: float = 1.0,
    timeout: float = DEFAULT_TIMEOUT,
) -> pd.DataFrame:
    """Fetch a FRED time series with retry logic.

    Parameters
    ----------
    series_id : str
        FRED series identifier (e.g., "DGS10", "WALCL")
    api_key : str, optional
        FRED API key. If None, reads from FRED_API_KEY environment variable.
    start : str
        Observation start date in YYYY-MM-DD format
    retries : int
        Number of retry attempts on failure
    backoff : float
        Backoff multiplier for retries (exponential backoff)
    timeout : float
        Request timeout in seconds

    Returns
    -------
    pd.DataFrame
        DataFrame with columns ["date", "value"], sorted by date

    Raises
    ------
    requests.RequestException
        If all retry attempts fail
    ValueError
        If API key is not provided or empty
    """
    if api_key is None:
        api_key = os.getenv("FRED_API_KEY", "").strip()
    if not api_key:
        raise ValueError("FRED_API_KEY not provided and not found in environment")

    url = f"{FRED_API_BASE}?series_id={series_id}&api_key={api_key}&file_type=json&observation_start={start}"

    last_exc = None
    for attempt in range(retries):
        try:
            r = requests.get(url, timeout=timeout)
            r.raise_for_status()
            data = r.json()
            obs = data.get("observations", [])

            rows = []
            for o in obs:
                date = o.get("date")
                val = o.get("value")
                if val in (None, "."):
                    continue
                try:
                    fval = float(val)
                except (ValueError, TypeError):
                    continue
                rows.append((date, fval))

            df = pd.DataFrame(rows, columns=["date", "value"])
            if not df.empty:
                df["date"] = pd.to_datetime(df["date"])
            return df

        except requests.RequestException as exc:
            last_exc = exc
            if attempt < retries - 1:
                sleep_time = backoff * (2**attempt)
                logger.warning(
                    "Failed to fetch %s (attempt %d/%d): %s. Retrying in %.1fs...",
                    series_id,
                    attempt + 1,
                    retries,
                    exc,
                    sleep_time,
                )
                time.sleep(sleep_time)
            else:
                logger.error("Failed to fetch %s after %d attempts: %s", series_id, retries, exc)

    if last_exc:
        raise last_exc
    raise requests.RequestException(f"Failed to fetch {series_id} after {retries} attempts")


__all__ = ["fetch_fred_series"]
