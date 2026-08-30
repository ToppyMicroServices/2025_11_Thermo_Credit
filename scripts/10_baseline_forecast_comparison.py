from __future__ import annotations

import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from lib.baseline_forecast import run_baseline_forecast_comparison, write_baseline_forecast_outputs


def main() -> None:
    source_ref = os.getenv("THEORY_SOURCE_REF") or None
    panel_mode = os.getenv("BASELINE_FORECAST_PANEL_MODE", os.getenv("CALIBRATION_PANEL_MODE", "realtime"))
    results, coverage, summary = run_baseline_forecast_comparison(
        ROOT / "site",
        source_ref=source_ref,
        panel_mode=panel_mode,
    )
    if results.empty:
        raise SystemExit("No baseline forecast comparison results were produced.")
    outputs = write_baseline_forecast_outputs(results, coverage, summary, root=ROOT)
    for path in outputs:
        print(f"Wrote {path.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
