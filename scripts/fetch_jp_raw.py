#!/usr/bin/env python3
"""Fetch the public FRED mirrors used by the Japan panel."""

from __future__ import annotations

import os
import sys
import time
from pathlib import Path
from typing import Tuple


ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from lib.official_series import fetch_fred_series


SERIES = [
    ("JPNASSETS", "BoJ Total Assets"),
    ("MYAGM2JPM189S", "Japan M2 (Monthly)"),
    ("IRLTLT01JPM156N", "Long-term JGB Yield"),
]

FRED_API_KEY = os.getenv("FRED_API_KEY", "")
DATA.mkdir(parents=True, exist_ok=True)


def fetch_and_save(
    series_id: str,
    start: str = "2012-01-01",
    sleep_sec: float = 0.6,
) -> Tuple[bool, str]:
    try:
        frame = fetch_fred_series(
            series_id,
            start=start,
            end=os.getenv("FRED_OBS_END", ""),
            api_key=FRED_API_KEY,
        )
        frame.to_csv(DATA / f"{series_id}.csv", index=False)
        time.sleep(max(0.0, sleep_sec))
        return True, f"Saved data/{series_id}.csv ({len(frame)} rows)"
    except Exception as exc:
        return False, f"Fetch failed for {series_id}: {exc}"


def main() -> int:
    start = os.getenv("JP_START", "2012-01-01")
    sleep_sec = float(os.getenv("FRED_SLEEP", "0.6"))
    ok_count = 0
    for series_id, title in SERIES:
        ok, message = fetch_and_save(series_id, start=start, sleep_sec=sleep_sec)
        print(("[OK]" if ok else "[ERR]"), series_id, title, "-", message)
        ok_count += int(ok)
    if ok_count != len(SERIES):
        print(f"[ERR] Refusing partial JP refresh ({ok_count}/{len(SERIES)} series updated).")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
