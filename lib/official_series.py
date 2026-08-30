"""Fetch and normalize public macro-financial series from official APIs."""

from __future__ import annotations

import io
import time
from typing import Any, Optional

import pandas as pd
import requests


FRED_API_URL = "https://api.stlouisfed.org/fred/series/observations"
FRED_CSV_URL = "https://fred.stlouisfed.org/graph/fredgraph.csv"
ECB_API_URL = "https://data-api.ecb.europa.eu/service/data"
USER_AGENT = "Thermo-Credit/2.2"


def _request(
    session: Any,
    url: str,
    *,
    params: dict[str, Any],
    timeout: float,
    retries: int,
    backoff: float,
) -> Any:
    last_error: Optional[Exception] = None
    for attempt in range(max(1, retries)):
        try:
            response = session.get(
                url,
                params=params,
                timeout=timeout,
                headers={"User-Agent": USER_AGENT},
            )
            response.raise_for_status()
            return response
        except Exception as exc:
            last_error = exc
            if attempt + 1 < max(1, retries):
                time.sleep(backoff**attempt)
    if last_error is not None:
        raise last_error
    raise RuntimeError(f"No response from {url}")


def _normalize_frame(
    frame: pd.DataFrame,
    *,
    date_column: str,
    value_column: str,
) -> pd.DataFrame:
    out = pd.DataFrame(
        {
            "date": pd.to_datetime(frame[date_column], errors="coerce"),
            "value": pd.to_numeric(frame[value_column], errors="coerce"),
        }
    )
    out = out.dropna(subset=["date", "value"]).drop_duplicates("date", keep="last")
    return out.sort_values("date").reset_index(drop=True)


def fetch_fred_series(
    series_id: str,
    *,
    start: str = "1990-01-01",
    end: str = "",
    api_key: str = "",
    timeout: float = 30.0,
    retries: int = 3,
    backoff: float = 1.5,
    session: Any = requests,
) -> pd.DataFrame:
    """Fetch FRED observations, using the public graph CSV when no key exists."""
    if api_key:
        params: dict[str, Any] = {
            "series_id": series_id,
            "api_key": api_key,
            "file_type": "json",
            "observation_start": start,
        }
        if end:
            params["observation_end"] = end
        response = _request(
            session,
            FRED_API_URL,
            params=params,
            timeout=timeout,
            retries=retries,
            backoff=backoff,
        )
        observations = response.json().get("observations", [])
        frame = pd.DataFrame(observations)
        if frame.empty or not {"date", "value"}.issubset(frame.columns):
            raise ValueError(f"FRED returned no observations for {series_id}")
        out = _normalize_frame(frame, date_column="date", value_column="value")
    else:
        params = {"id": series_id, "cosd": start}
        if end:
            params["coed"] = end
        response = _request(
            session,
            FRED_CSV_URL,
            params=params,
            timeout=timeout,
            retries=retries,
            backoff=backoff,
        )
        frame = pd.read_csv(io.StringIO(response.text))
        if frame.empty or len(frame.columns) < 2:
            raise ValueError(f"FRED public CSV returned no observations for {series_id}")
        date_column = next(
            (column for column in frame.columns if str(column).strip().upper() in {"DATE", "OBSERVATION_DATE"}),
            frame.columns[0],
        )
        value_column = series_id if series_id in frame.columns else frame.columns[1]
        out = _normalize_frame(frame, date_column=date_column, value_column=value_column)

    if out.empty:
        raise ValueError(f"FRED returned no numeric observations for {series_id}")
    return out


def fetch_ecb_series(
    dataflow: str,
    series_key: str,
    *,
    start: str = "",
    end: str = "",
    timeout: float = 60.0,
    retries: int = 3,
    backoff: float = 1.5,
    session: Any = requests,
) -> pd.DataFrame:
    """Fetch one ECB Data Portal series in csvdata format."""
    params: dict[str, Any] = {"format": "csvdata"}
    if start:
        params["startPeriod"] = start
    if end:
        params["endPeriod"] = end
    response = _request(
        session,
        f"{ECB_API_URL}/{dataflow}/{series_key}",
        params=params,
        timeout=timeout,
        retries=retries,
        backoff=backoff,
    )
    frame = pd.read_csv(io.StringIO(response.text))
    required = {"TIME_PERIOD", "OBS_VALUE"}
    if frame.empty or not required.issubset(frame.columns):
        raise ValueError(f"ECB returned no observations for {dataflow}/{series_key}")
    out = _normalize_frame(frame, date_column="TIME_PERIOD", value_column="OBS_VALUE")
    if out.empty:
        raise ValueError(f"ECB returned no numeric observations for {dataflow}/{series_key}")
    return out


__all__ = ["fetch_ecb_series", "fetch_fred_series"]
