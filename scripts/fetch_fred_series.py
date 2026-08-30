"""Fetch selected EU/US raw series from FRED and write CSVs under data/.

Reads IDs from data/series_selected_eu.json and data/series_selected_us.json.
Uses the authenticated JSON API when FRED_API_KEY is set and the official
public graph CSV otherwise.

Usage (optional):
  python scripts/fetch_fred_series.py --start 1990-01-01

Writes files: data/<SERIES_ID>.csv with columns: date,value
"""
from __future__ import annotations
import os, sys, json, time
import argparse
from typing import List, Set

import pandas as pd

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
DATA_DIR = os.path.join(ROOT, "data")
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from lib.official_series import fetch_fred_series as fetch_official_fred_series


NON_FRED_SERIES = {"ECB_BSI_TOTAL_ASSETS"}


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
                    if isinstance(sid, str) and sid.strip() and sid.strip() not in NON_FRED_SERIES:
                        ids.add(sid.strip())
    return sorted(ids)


def fetch_fred_series(series_id: str, api_key: str, start: str) -> pd.DataFrame:
    return fetch_official_fred_series(series_id, api_key=api_key, start=start)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--start", default=os.getenv("OBS_START", "1990-01-01"))
    ap.add_argument("--sleep", type=float, default=float(os.getenv("FRED_SLEEP", "0.5")))
    ap.add_argument(
        "--allow-partial",
        action="store_true",
        help="Return success when only some requested series were updated.",
    )
    args = ap.parse_args()

    api_key = os.getenv("FRED_API_KEY", "").strip()
    os.makedirs(DATA_DIR, exist_ok=True)
    ids = _collect_ids(["series_selected_eu.json", "series_selected_us.json"])
    if not ids:
        print("[fetch_fred] No IDs found in series_selected_{eu,us}.json; nothing to do.")
        return 0

    mode = "JSON API" if api_key else "public CSV"
    print(f"[fetch_fred] Fetching {len(ids)} series from FRED {mode} starting {args.start} ...")
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
    if ok != len(ids) and not args.allow_partial:
        print("[fetch_fred] Refusing to treat a partial refresh as successful.")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
