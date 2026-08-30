from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from lib.submission_readiness import evaluate_submission_readiness, write_submission_readiness_outputs  # noqa: E402


def main() -> int:
    results = evaluate_submission_readiness(ROOT)
    if results.empty:
        raise SystemExit("No submission-readiness results were produced.")
    for path in write_submission_readiness_outputs(results, root=ROOT):
        print(f"Wrote {path.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
