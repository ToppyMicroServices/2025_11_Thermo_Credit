"""Shared World Bank API fetch with caching + fallback.

Normalizes responses to a DataFrame with columns [date, value] and quarterly-aligned dates (+ QuarterEnd(0)).
"""
from __future__ import annotations
import os, json, time
import pandas as pd
import requests
from typing import Optional, Sequence

USER_AGENT = "TQTC-Research/1.0 (+https://toppymicros.com)"
BASE = "https://api.worldbank.org/v2"


def fetch_worldbank_series(
    country: str,
    indicator: str,
    cache_dir: str = "data",
    per_page: int = 20000,
    retries: int = 4,
    backoff: float = 2.0,
    fallback_csvs: Optional[Sequence[str]] = None,
    timeout: float = 45.0,
) -> pd.DataFrame:
    cache_name = f"worldbank_cache_{country}_{indicator}.json".replace("/", "_")
    os.makedirs(cache_dir, exist_ok=True)
    cache_path = os.path.join(cache_dir, cache_name)

    # Cache hit
    if os.path.exists(cache_path):
        try:
            payload = json.load(open(cache_path, "r", encoding="utf-8"))
            df = _normalize_payload(payload)
            if not df.empty:
                return df
        except Exception:
            pass

    # Live request with jitter
    url = f"{BASE}/country/{country}/indicator/{indicator}?format=json&per_page={per_page}"
    headers = {"User-Agent": USER_AGENT}
    last_exc = None
    for i in range(retries):
        try:
            time.sleep(0.05 + 0.2 * (i / max(1, retries - 1)))
            r = requests.get(url, timeout=timeout, headers=headers)
            r.raise_for_status()
            payload = r.json()
            data_block = payload[1]
            df_raw = [{"date": row.get("date"), "value": row.get("value")} for row in data_block]
            json.dump(df_raw, open(cache_path, "w", encoding="utf-8"))
            df = _normalize_payload(df_raw)
            if not df.empty:
                return df
        except Exception as exc:
            last_exc = exc
            if i < retries - 1:
                time.sleep(backoff ** i)
            else:
                break

    # Fallback CSVs (if provided)
    if fallback_csvs:
        for path in fallback_csvs:
            if os.path.exists(path):
                try:
                    df_local = pd.read_csv(path)
                    dcol = next((c for c in df_local.columns if str(c).lower() == "date"), None)
                    vcol = next((c for c in df_local.columns if str(c).lower() == "value"), None)
                    if not dcol or not vcol:
                        continue
                    df = pd.DataFrame({"date": df_local[dcol], "value": df_local[vcol]})
                    df = _align_quarters(df)
                    df["value"] = pd.to_numeric(df["value"], errors="coerce")
                    out = df.dropna().sort_values("date")
                    if not out.empty:
                        return out
                except Exception:
                    continue

    if last_exc:
        raise last_exc
    raise RuntimeError("World Bank fetch failed (no cache, live data, or fallback CSV succeeded)")


def _normalize_payload(payload) -> pd.DataFrame:
    if not payload:
        return pd.DataFrame(columns=["date", "value"])
    df = pd.DataFrame(payload)[["date", "value"]]
    df = _align_quarters(df)
    df["value"] = pd.to_numeric(df["value"], errors="coerce")
    return df.dropna().sort_values("date")


def _align_quarters(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["date"] = pd.to_datetime(df["date"]) + pd.offsets.QuarterEnd(0)
    return df

__all__ = ["fetch_worldbank_series"]
