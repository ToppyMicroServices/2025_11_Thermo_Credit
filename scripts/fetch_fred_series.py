"""Fetch selected EU/US raw series from FRED and write CSVs under data/.

Reads IDs from data/series_selected_eu.json and data/series_selected_us.json
and downloads observations via FRED API. Requires env FRED_API_KEY.

Usage (optional):
  python scripts/fetch_fred_series.py --start 1990-01-01

Writes files: data/<SERIES_ID>.csv with columns: date,value
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from typing import List, Set

import pandas as pd
import requests

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
DATA_DIR = os.path.join(ROOT, "data")


def _collect_ids(paths: List[str]) -> List[str]:
    ids: Set[str] = set()
    for p in paths:
        fp = os.path.join(DATA_DIR, p)
        if not os.path.exists(fp):
            continue
        try:
            meta = json.load(open(fp, "r", encoding="utf-8"))
        except Exception:
            continue
        if isinstance(meta, dict):
            for v in meta.values():
                if isinstance(v, dict):
                    sid = v.get("id")
                    if isinstance(sid, str) and sid.strip():
                        ids.add(sid.strip())
    return sorted(ids)


def fetch_fred_series(series_id: str, api_key: str, start: str) -> pd.DataFrame:
    url = (
        "https://api.stlouisfed.org/fred/series/observations"
        f"?series_id={series_id}&api_key={api_key}&file_type=json&observation_start={start}"
    )
    r = requests.get(url, timeout=30)
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
        except Exception:
            continue
        rows.append((date, fval))
    return pd.DataFrame(rows, columns=["date", "value"]).assign(date=lambda d: pd.to_datetime(d["date"]))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--start", default=os.getenv("OBS_START", "1990-01-01"))
    ap.add_argument("--sleep", type=float, default=float(os.getenv("FRED_SLEEP", "0.5")))
    args = ap.parse_args()

    api_key = os.getenv("FRED_API_KEY", "").strip()
    if not api_key:
        print("[fetch_fred] FRED_API_KEY not set; skipping fetch.")
        return 0

    os.makedirs(DATA_DIR, exist_ok=True)
    ids = _collect_ids(["series_selected_eu.json", "series_selected_us.json"])
    if not ids:
        print("[fetch_fred] No IDs found in series_selected_{eu,us}.json; nothing to do.")
        return 0

    print(f"[fetch_fred] Fetching {len(ids)} series from FRED starting {args.start} ...")
    ok = 0
    for idx, sid in enumerate(ids, 1):
        try:
            df = fetch_fred_series(sid, api_key, args.start)
            if not df.empty:
                out = os.path.join(DATA_DIR, f"{sid}.csv")
                df.to_csv(out, index=False)
                ok += 1
                print(f"[fetch_fred] {idx}/{len(ids)} wrote {sid}.csv ({len(df)} rows)")
            else:
                print(f"[fetch_fred] {idx}/{len(ids)} {sid}: empty dataset")
        except Exception as e:
            print(f"[fetch_fred] {idx}/{len(ids)} {sid}: ERROR {e}")
        time.sleep(max(0.0, args.sleep))
    print(f"[fetch_fred] Done. Wrote {ok}/{len(ids)} series.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
