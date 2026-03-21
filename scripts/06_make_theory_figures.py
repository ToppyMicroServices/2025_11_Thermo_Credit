from __future__ import annotations

import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from lib.theory_figures import build_theory_figures


def main() -> None:
    outputs = build_theory_figures(
        site_dir=ROOT / "site",
        output_dir=ROOT / "tex" / "generated",
        events_path=ROOT / "data" / "report_events.csv",
        start_date=os.getenv("THEORY_PLOT_START") or os.getenv("REPORT_PLOT_START") or "1998-01-01",
        source_ref=os.getenv("THEORY_SOURCE_REF") or None,
    )
    if not outputs:
        raise SystemExit("No theory figures were generated.")
    for path in outputs:
        print(f"Wrote {path.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
