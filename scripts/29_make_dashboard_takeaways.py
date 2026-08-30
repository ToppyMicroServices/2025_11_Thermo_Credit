#!/usr/bin/env python3
"""Write citation-ready dashboard takeaway figures."""
from __future__ import annotations

import os
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from lib.dashboard_takeaways import build_dashboard_takeaways


def main() -> int:
    outputs = build_dashboard_takeaways(
        site_dir=ROOT / "site",
        output_dir=ROOT / "tex" / "generated",
        events_path=ROOT / "data" / "report_events.csv",
        start_date=os.getenv("TAKEAWAYS_START", "2015-01-01"),
    )
    for path in outputs:
        print(f"Wrote {path.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
