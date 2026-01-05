import os
import sys
import time

import pandas as pd
import requests

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
DATA = os.path.join(ROOT, "data")

SERIES = [
    ("JPNASSETS", "BoJ Total Assets"),
    ("MYAGM2JPM189S", "Japan M2 (Monthly)"),
    ("IRLTLT01JPM156N", "Long-term JGB Yield"),
]

FRED_API_KEY = os.getenv("FRED_API_KEY")
FRED_API_URL = "https://api.stlouisfed.org/fred/series/observations"

os.makedirs(DATA, exist_ok=True)

def fetch_and_save(sid: str, start: str = "2012-01-01", sleep_sec: float = 0.6) -> tuple[bool, str]:
    if not FRED_API_KEY:
        return False, "FRED_API_KEY not set"
    params = {
        "series_id": sid,
        "api_key": FRED_API_KEY,
        "file_type": "json",
        "observation_start": start,
        "observation_end": os.getenv("FRED_OBS_END", ""),
    }
    try:
        r = requests.get(FRED_API_URL, params=params, timeout=30)
        r.raise_for_status()
        js = r.json()
        obs = js.get("observations", [])
        out = pd.DataFrame(obs)
        if out.empty:
            return False, f"No observations for {sid}"
        out = out.rename(columns={"date": "date", "value": "value"})
        out["date"] = pd.to_datetime(out["date"], errors="coerce")
        out["value"] = pd.to_numeric(out["value"], errors="coerce")
        out = out.dropna(subset=["date", "value"]).sort_values("date")[['date','value']]
        out.to_csv(os.path.join(DATA, f"{sid}.csv"), index=False)
        time.sleep(sleep_sec)
        return True, f"Saved data/{sid}.csv ({len(out)} rows)"
    except Exception as e:
        return False, f"Fetch failed for {sid}: {e}"

if __name__ == "__main__":
    start = os.getenv("JP_START", "2012-01-01")
    sleep_sec = float(os.getenv("FRED_SLEEP", "0.6"))
    ok_any = False
    for sid, title in SERIES:
        ok, msg = fetch_and_save(sid, start=start, sleep_sec=sleep_sec)
        print(("[OK]" if ok else "[ERR]"), sid, title, "-", msg)
        ok_any = ok_any or ok
    if not ok_any:
        sys.exit(1)
