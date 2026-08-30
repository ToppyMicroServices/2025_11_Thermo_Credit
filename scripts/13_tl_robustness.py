from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from lib.tl_robustness import run_tl_robustness, write_tl_robustness_outputs  # noqa: E402


def main() -> int:
    results = run_tl_robustness(ROOT / "site")
    if results.empty:
        raise SystemExit("No indicator panels were available for TL robustness.")
    for path in write_tl_robustness_outputs(results, root=ROOT):
        print(f"Wrote {path.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
