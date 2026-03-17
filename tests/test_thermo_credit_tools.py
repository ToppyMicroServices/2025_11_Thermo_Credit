import json
import subprocess
import sys
from pathlib import Path

from lib.thermo_credit_tools import (
    build_repo_compute_payload,
    compare_regimes,
    compute_thermo_credit_metrics,
    evaluate_scenario,
    get_theory_overview,
    get_variable_definitions,
)


ROOT = Path(__file__).resolve().parents[1]


def test_get_theory_overview_short_is_structured():
    result = get_theory_overview({"detail_level": "short"})

    assert result["model_version"] == "thermo-credit-v2-draft"
    assert len(result["purpose"]) == 2
    assert "q_t" in result["state_vector"]
    assert result["interpretation_status"] == "research"


def test_get_variable_definitions_filters_existing_repo_metrics():
    result = get_variable_definitions(
        {
            "symbols": ["C_t", "q_t", "F_C"],
            "include_existing_repo_metrics": False,
        }
    )

    symbols = {row["symbol"] for row in result["variables"]}
    assert symbols == {"C_t", "q_t"}
    assert all(row["category"] != "existing_repo_metric" for row in result["variables"])
    assert any(row["current_repo_artifacts"] for row in result["variables"])


def test_compute_metrics_from_structured_input():
    payload = {
        "region": "jp",
        "frequency": "quarterly",
        "observations": [
            {
                "date": "2024-12-31",
                "credit_total": 100.0,
                "credit_real_share": 0.65,
                "nominal_output": 510.0,
                "asset_price_index": 103.0,
                "stress_proxy": 0.18,
            },
            {
                "date": "2025-03-31",
                "credit_total": 120.0,
                "credit_real_share": 0.62,
                "nominal_output": 515.0,
                "asset_price_index": 106.0,
                "stress_proxy": 0.27,
            },
        ],
    }

    result = compute_thermo_credit_metrics(payload)
    latest = result["metrics"][-1]

    assert latest["credit_real_flow"] == 74.4
    assert latest["credit_asset_flow"] == 45.6
    assert latest["credit_efficiency"] == 0.041667
    assert latest["asset_bias"] == 0.025
    assert latest["dissipation_proxy"] == 0.958333
    assert result["summary"]["latest_credit_real_share"] == 0.62


def test_compute_metrics_from_repo_payload():
    payload = build_repo_compute_payload("jp", limit=3)
    result = compute_thermo_credit_metrics(payload)

    assert len(result["metrics"]) == 3
    assert result["source_ids"] == ["site/indicators.csv"]
    assert 0.0 <= result["metrics"][-1]["credit_real_share"] <= 1.0
    assert any("L_real" in limitation for limitation in result["limitations"])


def test_evaluate_scenario_surfaces_asset_shift():
    result = evaluate_scenario(
        {
            "region": "jp",
            "frequency": "quarterly",
            "baseline_state": {
                "credit_total": 120.0,
                "credit_real_share": 0.62,
                "nominal_output": 515.0,
                "asset_price_index": 106.0,
                "stress_proxy": 0.27,
            },
            "scenario_shocks": {
                "delta_credit_total": 20.0,
                "delta_credit_real_share": -0.12,
            },
            "horizon_periods": 2,
        }
    )

    assert result["delta"]["credit_asset_flow"] > 0
    assert result["delta"]["credit_real_flow"] < 0
    assert "Asset-directed flow rises" in result["summary"]


def test_compare_regimes_reports_more_asset_bias():
    payload = {
        "region": "custom",
        "frequency": "quarterly",
        "period_a": {
            "label": "Period A",
            "observations": [
                {
                    "date": "2024-06-30",
                    "credit_total": 100.0,
                    "credit_real_share": 0.70,
                    "nominal_output": 500.0,
                    "stress_proxy": 0.10,
                },
                {
                    "date": "2024-09-30",
                    "credit_total": 110.0,
                    "credit_real_share": 0.68,
                    "nominal_output": 508.0,
                    "stress_proxy": 0.12,
                },
            ],
        },
        "period_b": {
            "label": "Period B",
            "observations": [
                {
                    "date": "2025-06-30",
                    "credit_total": 120.0,
                    "credit_real_share": 0.52,
                    "nominal_output": 510.0,
                    "stress_proxy": 0.30,
                },
                {
                    "date": "2025-09-30",
                    "credit_total": 130.0,
                    "credit_real_share": 0.48,
                    "nominal_output": 514.0,
                    "stress_proxy": 0.34,
                },
            ],
        },
    }

    result = compare_regimes(payload)

    assert result["comparison"]["avg_credit_real_share"] < 0
    assert result["comparison"]["avg_stress_proxy"] > 0
    assert "more asset-biased" in result["summary"]


def test_cli_compute_from_repo_region():
    cmd = [
        sys.executable,
        str(ROOT / "scripts" / "thermo_credit_cli.py"),
        "compute_thermo_credit_metrics",
        "--repo-region",
        "jp",
        "--limit",
        "2",
    ]
    completed = subprocess.run(cmd, capture_output=True, text=True, check=True, cwd=ROOT)
    payload = json.loads(completed.stdout)

    assert payload["region"] == "jp"
    assert len(payload["metrics"]) == 2
