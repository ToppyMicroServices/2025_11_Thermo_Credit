#!/usr/bin/env python3
"""Fetch the current Eurosystem total-assets series from the ECB Data Portal."""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from lib.official_series import fetch_ecb_series


SERIES_ID = "ECB_BSI_TOTAL_ASSETS"
DATAFLOW = "BSI"
SERIES_KEY = "M.U2.N.C.T00.A.1.Z5.0000.Z01.E"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--start", default="1999-01")
    parser.add_argument("--end", default="")
    args = parser.parse_args()

    frame = fetch_ecb_series(DATAFLOW, SERIES_KEY, start=args.start, end=args.end)
    data_dir = ROOT / "data"
    data_dir.mkdir(parents=True, exist_ok=True)
    output_path = data_dir / f"{SERIES_ID}.csv"
    frame.to_csv(output_path, index=False)

    metadata = {
        "id": SERIES_ID,
        "provider": "European Central Bank",
        "dataflow": DATAFLOW,
        "series_key": SERIES_KEY,
        "title": "Total assets of the Eurosystem, stocks",
        "units": "EUR millions",
        "frequency": "monthly",
        "retrieved_at": datetime.now(timezone.utc).isoformat(),
        "first_observation": frame["date"].min().date().isoformat(),
        "latest_observation": frame["date"].max().date().isoformat(),
        "rows": int(len(frame)),
        "source_url": f"https://data.ecb.europa.eu/data/datasets/{DATAFLOW}/{DATAFLOW}.{SERIES_KEY}",
    }
    (data_dir / f"{SERIES_ID}.metadata.json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(f"Wrote {output_path} ({len(frame)} rows; latest {metadata['latest_observation']})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
