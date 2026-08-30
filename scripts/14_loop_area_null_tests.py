from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from lib.loop_null_tests import run_loop_area_null_tests, write_loop_area_null_test_outputs  # noqa: E402


def main() -> int:
    results = run_loop_area_null_tests(ROOT / "site", events_path=ROOT / "data" / "report_events.csv")
    if results.empty:
        raise SystemExit("No loop-area null-test results were produced.")
    for path in write_loop_area_null_test_outputs(results, root=ROOT):
        print(f"Wrote {path.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
