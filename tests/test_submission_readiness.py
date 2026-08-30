from __future__ import annotations

import json

import pandas as pd

from lib.submission_readiness import (
    REPRODUCIBILITY_REQUIRED_OUTPUTS,
    evaluate_submission_readiness,
    render_submission_readiness_tex,
    summarize_submission_readiness,
    write_submission_readiness_outputs,
)


def _write_fixture(root):
    site = root / "site"
    data = root / "data"
    rep = root / "replication"
    site.mkdir()
    data.mkdir()
    rep.mkdir()
    pd.DataFrame(
        [
            {
                "region_key": "jp",
                "baseline": "boj_mapped_stock_growth",
                "model": "matched_credit_plus_q_t",
                "is_primary_allocation_measure": True,
                "horizon_quarters": 4,
                "target": "asset_acceleration",
                "metric_loss_diff": -0.1,
                "block_ci_low": -0.2,
                "block_ci_high": -0.01,
            },
            {
                "region_key": "jp",
                "baseline": "boj_mapped_stock_growth",
                "model": "matched_credit_plus_q_t",
                "is_primary_allocation_measure": True,
                "horizon_quarters": 8,
                "target": "asset_acceleration",
                "metric_loss_diff": -0.1,
                "block_ci_low": -0.2,
                "block_ci_high": -0.01,
            },
            {
                "region_key": "jp",
                "baseline": "boj_mapped_stock_growth",
                "model": "matched_credit_plus_q_t",
                "is_primary_allocation_measure": True,
                "horizon_quarters": 4,
                "target": "spread_widening",
                "metric_loss_diff": -0.1,
                "block_ci_low": -0.2,
                "block_ci_high": -0.01,
            },
            {
                "region_key": "jp",
                "baseline": "boj_mapped_stock_growth",
                "model": "matched_credit_plus_q_t",
                "is_primary_allocation_measure": True,
                "horizon_quarters": 8,
                "target": "spread_widening",
                "metric_loss_diff": -0.1,
                "block_ci_low": -0.2,
                "block_ci_high": -0.01,
            },
        ]
    ).to_csv(site / "destination_oos_incremental.csv", index=False)
    pd.DataFrame({"winner_rmse": ["tuned_XC", "tuned_XC"]}).to_csv(site / "calibration_holdout_test.csv", index=False)
    pd.DataFrame({"control": ["observed"], "flat_flag": [False]}).to_csv(site / "entropy_partition_robustness.csv", index=False)
    pd.DataFrame({"credit_destination_source": ["loan_purpose_direct"]}).to_csv(site / "credit_destination.csv", index=False)
    pd.DataFrame({"monotone_pass": [True, True]}).to_csv(site / "tl_robustness.csv", index=False)
    pd.DataFrame(
        {
            "window_family": ["latest_rolling"] * 5,
            "region_key": ["jp"] * 5,
            "segmentation_window": [12] * 5,
            "null_method": ["block", "phase", "ar", "event", "placebo"],
            "null_status": ["top_5pct"] * 5,
        }
    ).to_csv(site / "loop_area_null_tests.csv", index=False)
    (rep / "reproducibility_log.md").write_text("- Status: PASS\n", encoding="utf-8")
    (rep / "reproducibility_manifest.json").write_text(
        json.dumps(
            {
                "generated_at_utc": "2026-06-19T00:00:00+00:00",
                "pass": True,
                "outputs": {rel: {"sha256": "test"} for rel in REPRODUCIBILITY_REQUIRED_OUTPUTS},
            }
        ),
        encoding="utf-8",
    )


def test_submission_readiness_passes_when_all_gates_are_met(tmp_path) -> None:
    _write_fixture(tmp_path)

    results = evaluate_submission_readiness(tmp_path)
    summary = summarize_submission_readiness(results)
    tex = render_submission_readiness_tex(results)

    assert summary["submit_now"] is True
    assert results["status"].eq("pass").all()
    assert "Submission-readiness gates" in tex


def test_submission_readiness_outputs_json_and_csv(tmp_path) -> None:
    _write_fixture(tmp_path)
    results = evaluate_submission_readiness(tmp_path)

    paths = write_submission_readiness_outputs(results, root=tmp_path)

    assert all(path.exists() for path in paths)
    summary = json.loads((tmp_path / "data" / "submission_readiness_summary.json").read_text())
    assert summary["passed"] == 5


def test_submission_readiness_rejects_stale_reproducibility_log(tmp_path) -> None:
    _write_fixture(tmp_path)
    (tmp_path / "replication" / "reproducibility_manifest.json").write_text(
        json.dumps({"generated_at_utc": "2026-06-01T00:00:00+00:00", "pass": True, "outputs": {}}),
        encoding="utf-8",
    )

    results = evaluate_submission_readiness(tmp_path)
    row = results[results["criterion_id"].eq("full_reproducibility")].iloc[0]

    assert row["status"] == "not_yet"
    assert "stale" in row["current_read"]
