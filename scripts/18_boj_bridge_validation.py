from __future__ import annotations

import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from lib.boj_bridge_validation import write_boj_bridge_validation_outputs


def main() -> int:
    outputs = write_boj_bridge_validation_outputs(ROOT)
    for path in outputs.values():
        print(f"Wrote {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
