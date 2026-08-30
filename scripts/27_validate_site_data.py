#!/usr/bin/env python3
"""Validate published regional CSVs and write a machine-readable manifest."""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from lib.data_quality import validate_site_data


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--site-dir", type=Path, default=ROOT / "site")
    parser.add_argument("--output", type=Path, default=ROOT / "site" / "data_manifest.json")
    parser.add_argument("--min-rows", type=int, default=8)
    parser.add_argument(
        "--regions",
        nargs="+",
        choices=("jp", "eu", "us"),
        help="Validate only the selected regions. The default validates all three.",
    )
    parser.add_argument(
        "--max-age-days",
        type=int,
        default=None,
        help="Fail when a regional latest observation is older than this many days.",
    )
    args = parser.parse_args()

    source_files = [
        ROOT / "data" / "series_selected.json",
        ROOT / "data" / "series_selected_eu.json",
        ROOT / "data" / "series_selected_us.json",
        ROOT / "data" / "ECB_BSI_TOTAL_ASSETS.metadata.json",
        ROOT / "data" / "credit_destination_jp_metadata.json",
    ]
    manifest = validate_site_data(
        args.site_dir,
        reference_time=datetime.now(timezone.utc),
        min_rows=args.min_rows,
        max_age_days=args.max_age_days,
        regions=args.regions,
        source_files=source_files,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    for region in manifest["regions"]:
        latest = region.get("latest_observation", "unknown")
        print(f"{region['region'].upper()}: rows={region.get('rows', 0)} latest={latest} valid={region['valid']}")
        for error in region["errors"]:
            print(f"  ERROR: {error}")
    print(f"Wrote {args.output}")
    return 0 if manifest["valid"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
