from __future__ import annotations

import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from lib.theory_calibration import calibrate_regions, write_calibration_outputs


def main() -> None:
    source_ref = os.getenv("THEORY_SOURCE_REF") or None
    results = calibrate_regions(ROOT / "site", source_ref=source_ref)
    if not results:
        raise SystemExit("No calibration results were produced.")
    outputs = write_calibration_outputs(
        results,
        output_dir=ROOT / "tex" / "generated",
        data_dir=ROOT / "data",
        source_ref=source_ref,
    )
    for path in outputs:
        print(f"Wrote {path.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
