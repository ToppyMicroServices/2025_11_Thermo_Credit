from __future__ import annotations

import numpy as np
import pandas as pd

from lib.entropy_robustness import (
    build_partition_shares,
    evaluate_entropy_partition_region,
    run_entropy_partition_robustness,
    summarize_entropy_robustness,
)


def _varying_allocation_frame() -> pd.DataFrame:
    dates = pd.date_range("2020-03-31", periods=6, freq="QE-DEC")
    return pd.DataFrame(
        {
            "date": dates,
            "q_pay": [0.30, 0.32, 0.29, 0.31, 0.34, 0.28],
            "q_firm": [0.30, 0.28, 0.31, 0.27, 0.26, 0.33],
            "q_asset": [0.25, 0.24, 0.26, 0.29, 0.27, 0.24],
            "q_reserve": [0.15, 0.16, 0.14, 0.13, 0.13, 0.15],
            "q_productive": [0.30, 0.28, 0.31, 0.27, 0.26, 0.33],
            "q_housing": [0.12, 0.13, 0.11, 0.12, 0.14, 0.11],
            "q_consumption": [0.18, 0.19, 0.18, 0.19, 0.20, 0.17],
            "q_financial": [0.25, 0.24, 0.26, 0.29, 0.27, 0.24],
            "q_government": [0.15, 0.16, 0.14, 0.13, 0.13, 0.15],
        }
    )


def test_build_partition_shares_supports_requested_bucket_counts() -> None:
    frame = _varying_allocation_frame()

    for family in ("borrower_label", "loan_purpose"):
        for bucket_count in (3, 5, 7):
            shares = build_partition_shares(frame, family=family, bucket_count=bucket_count)

            assert shares.shape[1] == bucket_count
            assert np.allclose(shares.sum(axis=1), 1.0)


def test_entropy_robustness_reports_controls_and_flat_decision() -> None:
    frame = _varying_allocation_frame()
    flat = frame.copy()
    q_cols = [col for col in flat.columns if col.startswith("q_")]
    for col in q_cols:
        flat.loc[:, col] = flat[col].iloc[0]

    results = evaluate_entropy_partition_region(flat, region_key="xx", region_label="Test Region")
    observed = results[results["control"].eq("observed")]
    controls = set(results["control"])
    summary = summarize_entropy_robustness(results)

    assert controls == {"observed", "shuffled_shares", "fixed_shares", "random_walk_shares"}
    assert len(observed) == 6
    assert observed["flat_flag"].all()
    assert set(observed["main_text_use"]) == {"exclude_entropy_result"}
    assert summary["regions"]["xx"]["exclude_entropy_from_main_text"] is True


def test_entropy_robustness_prefers_jp_destination_panel(tmp_path) -> None:
    dates = pd.date_range("2020-03-31", periods=6, freq="QE-DEC")
    pd.DataFrame(
        {
            "date": dates,
            "q_productive": [0.3] * 6,
            "q_housing": [0.2] * 6,
            "q_consumption": [0.1] * 6,
            "q_financial": [0.2] * 6,
            "q_government": [0.2] * 6,
        }
    ).to_csv(tmp_path / "allocation_q.csv", index=False)
    fallback = pd.DataFrame({"date": dates, "q_productive": [0.3] * 6, "q_financial": [0.7] * 6})
    fallback.to_csv(tmp_path / "allocation_q_eu.csv", index=False)
    fallback.to_csv(tmp_path / "allocation_q_us.csv", index=False)
    pd.DataFrame(
        {
            "date": dates,
            "C_G": [8, 5, 7, 3, 9, 4],
            "C_B": [1, 2, 1, 4, 1, 3],
            "C_E": [1, 3, 2, 3, 0.5, 3],
        }
    ).to_csv(tmp_path / "credit_destination_jp.csv", index=False)

    results = run_entropy_partition_robustness(tmp_path)
    jp_observed = results[results["region_key"].eq("jp") & results["control"].eq("observed")]

    assert set(jp_observed["partition_input_source"]) == {"credit_destination_jp"}
    assert not jp_observed["flat_flag"].all()
