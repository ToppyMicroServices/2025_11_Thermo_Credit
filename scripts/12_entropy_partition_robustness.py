from __future__ import annotations

import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from lib.entropy_robustness import (  # noqa: E402
    run_entropy_partition_robustness,
    write_entropy_partition_robustness_outputs,
)


def main() -> int:
    seed = int(os.getenv("ENTROPY_ROBUSTNESS_SEED", "7311"))
    results = run_entropy_partition_robustness(ROOT / "data", seed=seed)
    if results.empty:
        raise SystemExit("No allocation_q*.csv files were available for entropy robustness.")
    for path in write_entropy_partition_robustness_outputs(results, root=ROOT):
        print(f"Wrote {path.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
