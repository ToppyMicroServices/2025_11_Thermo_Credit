import json
import subprocess
import sys
from pathlib import Path

import pytest

from lib.thermo_credit_tools import (
    build_repo_compute_payload,
    compare_regimes,
    compute_thermo_credit_metrics,
    evaluate_scenario,
    get_theory_overview,
    get_variable_definitions,
)


ROOT = Path(__file__).resolve().parents[1]


def test_overview_keeps_measurement_and_structural_share_separate() -> None:
    result = get_theory_overview({"detail_level": "full"})
    assert result["current_measurement"]["q_t"].startswith("Japan four-quarter")
    assert "q_t^use" in result["proposed_structural_variables"]
    assert "incremental predictability" in result["limitations"][-1]


def test_variable_definitions_preserve_evidence_and_claim_limit() -> None:
    result = get_variable_definitions({"symbols": ["q_t", "q_t^use"]})
    variables = {row["symbol"]: row for row in result["variables"]}
    assert variables["q_t"]["evidence_class"] == "derived_measurement"
    assert variables["q_t^use"]["status"] == "not_identified"
    assert variables["q_t"]["claim_limit"]


def test_compute_metrics_is_an_arithmetic_partition() -> None:
    result = compute_thermo_credit_metrics(
        {
            "region": "jp",
            "allocation_semantics": "borrower composition",
            "allocation_evidence_class": "derived_measurement",
            "observations": [
                {"date": "2024-12-31", "credit_scale": 100, "allocation_share": 0.60, "output_proxy": 500},
                {"date": "2025-03-31", "credit_scale": 120, "allocation_share": 0.55, "output_proxy": 506},
            ],
        }
    )
    latest = result["latest"]
    assert latest["allocated_scale"] == 66.0
    assert latest["complement_scale"] == 54.0
    assert latest["allocation_change"] == -0.05
    assert latest["output_change_per_credit"] == 0.05
    assert "arithmetic decompositions" in result["limitations"][1]


def test_compute_metrics_rejects_invalid_share() -> None:
    with pytest.raises(ValueError, match="between 0 and 1"):
        compute_thermo_credit_metrics(
            {"observations": [{"date": "2025-03-31", "credit_scale": 1, "allocation_share": 1.2}]}
        )


def test_scenario_does_not_claim_dynamics() -> None:
    result = evaluate_scenario(
        {
            "baseline_state": {"credit_scale": 100, "allocation_share": 0.6},
            "scenario_shocks": {"delta_credit_scale": 20, "delta_allocation_share": -0.1},
            "horizon_periods": 4,
        }
    )
    assert result["delta"]["allocated_scale"] == 0.0
    assert result["delta"]["complement_scale"] == 20.0
    assert "No behavioral response" in result["limitations"][0]


def test_compare_regimes_reports_plain_period_difference() -> None:
    result = compare_regimes(
        {
            "allocation_semantics": "borrower composition",
            "period_a": {"label": "A", "observations": [{"date": "2020-03-31", "credit_scale": 100, "allocation_share": 0.6}]},
            "period_b": {"label": "B", "observations": [{"date": "2021-03-31", "credit_scale": 120, "allocation_share": 0.5}]},
        }
    )
    assert result["change_b_minus_a"]["allocation_share"] == -0.1
    assert "not a causal estimate" in result["limitations"][1]


def test_repo_payload_and_cli_use_current_jp_semantics() -> None:
    payload = build_repo_compute_payload("jp", limit=2)
    assert payload["allocation_evidence_class"] == "derived_measurement"
    assert "borrower-composition" in payload["allocation_semantics"]
    command = [
        sys.executable,
        str(ROOT / "scripts" / "thermo_credit_cli.py"),
        "compute_thermo_credit_metrics",
        "--repo-region",
        "jp",
        "--limit",
        "2",
    ]
    completed = subprocess.run(command, cwd=ROOT, capture_output=True, text=True, check=True)
    result = json.loads(completed.stdout)
    assert len(result["metrics"]) == 2
    assert result["allocation_evidence_class"] == "derived_measurement"
