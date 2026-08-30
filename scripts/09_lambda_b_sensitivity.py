from __future__ import annotations

import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from lib.lambda_sensitivity import run_lambda_sensitivity, write_lambda_sensitivity_outputs


def main() -> None:
    source_ref = os.getenv("THEORY_SOURCE_REF") or None
    panel_mode = os.getenv("LAMBDA_SENSITIVITY_PANEL_MODE", os.getenv("CALIBRATION_PANEL_MODE", "realtime"))
    metrics, panel, summary = run_lambda_sensitivity(
        ROOT / "site",
        source_ref=source_ref,
        panel_mode=panel_mode,
    )
    if metrics.empty:
        raise SystemExit("No lambda_B sensitivity results were produced.")
    outputs = write_lambda_sensitivity_outputs(metrics, panel, summary, root=ROOT)
    for path in outputs:
        print(f"Wrote {path.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
