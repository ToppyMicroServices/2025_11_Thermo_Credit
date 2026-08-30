from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from lib.integrability_synthetic import (  # noqa: E402
    run_integrability_synthetic_test,
    write_integrability_synthetic_outputs,
)


def main() -> int:
    results = run_integrability_synthetic_test()
    if results.empty:
        raise SystemExit("No integrability synthetic-test results were produced.")
    for path in write_integrability_synthetic_outputs(results, root=ROOT):
        print(f"Wrote {path.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
