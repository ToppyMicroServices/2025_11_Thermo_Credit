"""Helpers for fetching ECB Data Portal time series via the official API."""

from __future__ import annotations

import io
import re
import time
from typing import Optional

import pandas as pd
import requests

BASE_URL = "https://data-api.ecb.europa.eu/service/data"
USER_AGENT = "TQTC-Research/1.0 (+https://toppymicros.com)"

_QUARTER_RE = re.compile(r"^(?P<year>\d{4})-Q(?P<quarter>[1-4])$")


def _parse_time_period(value: str) -> pd.Timestamp:
    text = str(value).strip()
    match = _QUARTER_RE.match(text)
    if match:
        year = int(match.group("year"))
        quarter = int(match.group("quarter"))
        return pd.Period(f"{year}Q{quarter}", freq="Q-DEC").to_timestamp(how="end").normalize()
    return pd.Timestamp(text)


def normalize_ecb_csv(payload: str) -> pd.DataFrame:
    """Normalize ECB csvdata payloads to columns ['date', 'value']."""
    df = pd.read_csv(io.StringIO(payload))
    lowered = {str(col).strip().lower(): col for col in df.columns}
    date_col = lowered.get("time_period") or lowered.get("date")
    value_col = lowered.get("obs_value") or lowered.get("value")
    if not date_col or not value_col:
        raise ValueError("ECB CSV payload missing TIME_PERIOD/OBS_VALUE columns")
    out = pd.DataFrame({"date": df[date_col], "value": df[value_col]})
    out["date"] = out["date"].map(_parse_time_period)
    out["value"] = pd.to_numeric(out["value"], errors="coerce")
    out = out.dropna(subset=["date", "value"]).sort_values("date").reset_index(drop=True)
    return out


def fetch_ecb_series(
    flow_id: str,
    series_key: str,
    *,
    start_period: Optional[str] = None,
    end_period: Optional[str] = None,
    timeout: float = 30.0,
    retries: int = 3,
    backoff: float = 1.5,
) -> pd.DataFrame:
    params = {"format": "csvdata"}
    if start_period:
        params["startPeriod"] = start_period
    if end_period:
        params["endPeriod"] = end_period
    url = f"{BASE_URL}/{flow_id}/{series_key}"
    last_exc: Optional[Exception] = None
    for attempt in range(retries):
        try:
            response = requests.get(url, params=params, timeout=timeout, headers={"User-Agent": USER_AGENT})
            response.raise_for_status()
            return normalize_ecb_csv(response.text)
        except Exception as exc:
            last_exc = exc
            if attempt < retries - 1:
                time.sleep(backoff ** attempt)
            else:
                break
    if last_exc:
        raise last_exc
    raise RuntimeError(f"ECB fetch failed for {flow_id}/{series_key}")


__all__ = ["fetch_ecb_series", "normalize_ecb_csv"]
