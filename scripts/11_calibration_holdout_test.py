from __future__ import annotations

import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from lib.calibration_holdout import run_calibration_holdout_tests, write_calibration_holdout_outputs


def main() -> None:
    source_ref = os.getenv("THEORY_SOURCE_REF") or None
    panel_mode = os.getenv("CALIBRATION_HOLDOUT_PANEL_MODE", os.getenv("CALIBRATION_PANEL_MODE", "realtime"))
    results = run_calibration_holdout_tests(ROOT / "site", source_ref=source_ref, panel_mode=panel_mode)
    if results.empty:
        raise SystemExit("No calibration holdout results were produced.")
    outputs = write_calibration_holdout_outputs(results, root=ROOT)
    for path in outputs:
        print(f"Wrote {path.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
