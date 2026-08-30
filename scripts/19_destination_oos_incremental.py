from __future__ import annotations

import os
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from lib.destination_oos import (
    ALLOCATION_MEASURES,
    BOJ_RELEASE_LAG_DAYS,
    MIN_TRAINING_ROWS_SETTINGS,
    PRIMARY_ALLOCATION_MEASURE,
    PRIMARY_MIN_TRAINING_ROWS,
    run_destination_oos,
    write_destination_oos_outputs,
)


def main() -> None:
    source_ref = os.getenv("THEORY_SOURCE_REF") or None
    panel_mode = os.getenv("DESTINATION_OOS_PANEL_MODE", os.getenv("CALIBRATION_PANEL_MODE", "realtime"))
    raw_measures = os.getenv(
        "DESTINATION_OOS_ALLOCATION_MEASURES",
        ",".join(ALLOCATION_MEASURES),
    )
    allocation_measures = tuple(
        dict.fromkeys(value.strip() for value in raw_measures.split(",") if value.strip())
    )
    primary_allocation_measure = os.getenv(
        "DESTINATION_OOS_PRIMARY_ALLOCATION_MEASURE",
        PRIMARY_ALLOCATION_MEASURE,
    ).strip()
    release_lag_days = int(
        os.getenv("DESTINATION_OOS_RELEASE_LAG_DAYS", str(BOJ_RELEASE_LAG_DAYS))
    )
    raw_training_settings = os.getenv(
        "DESTINATION_OOS_MIN_TRAINING_ROWS_SETTINGS",
        ",".join(str(value) for value in MIN_TRAINING_ROWS_SETTINGS),
    )
    min_training_rows_settings = tuple(
        dict.fromkeys(
            int(value.strip())
            for value in raw_training_settings.split(",")
            if value.strip()
        )
    )
    primary_min_training_rows = int(
        os.getenv(
            "DESTINATION_OOS_PRIMARY_MIN_TRAINING_ROWS",
            str(PRIMARY_MIN_TRAINING_ROWS),
        )
    )
    raw_boj_path = os.getenv("DESTINATION_OOS_BOJ_DATA_PATH", "").strip()
    boj_data_path = Path(raw_boj_path) if raw_boj_path else ROOT / "data" / "credit_destination_jp.csv"
    results = run_destination_oos(
        ROOT / "site",
        source_ref=source_ref,
        panel_mode=panel_mode,
        boj_data_path=boj_data_path,
        allocation_measures=allocation_measures,
        primary_allocation_measure=primary_allocation_measure,
        release_lag_days=release_lag_days,
        min_training_rows_settings=min_training_rows_settings,
        primary_min_training_rows=primary_min_training_rows,
    )
    if results.empty:
        raise SystemExit("No JP borrower-composition OOS results were produced.")
    outputs = write_destination_oos_outputs(results, root=ROOT)
    for path in outputs:
        print(f"Wrote {path.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
