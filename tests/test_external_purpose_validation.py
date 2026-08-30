from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from lib.external_purpose_validation import (
    load_metadata,
    render_validation_table,
    run_external_validation,
    validate_mlit_housing,
    validate_mof_manufacturing,
    verify_snapshot_checksums,
)


ROOT = Path(__file__).resolve().parent.parent


def _mof_metadata() -> dict[str, object]:
    return {
        "validation_type": "borrower-side convergence",
        "supported_primary_branch": "NFB / manufacturing",
        "purpose_status": "not purpose validation",
        "estimand": "4Q stock-change association",
        "population": "manufacturing corporations",
        "frequency": "quarterly",
        "sample_change_control": "use 4Q differences",
        "does_not_validate": ["loan purpose", "entire NFB"],
        "official_file_url": "https://example.test/mof.xls",
    }


def _mlit_metadata() -> dict[str, object]:
    return {
        "validation_type": "direct-purpose partial validation",
        "supported_primary_branch": "PROP / household housing",
        "purpose_status": "housing branch only",
        "estimand": "published purpose composition",
        "population": "private housing-loan institutions",
        "frequency": "annual",
        "does_not_validate": ["corporate real estate", "quarterly stock changes"],
        "official_file_url": "https://example.test/mlit.pdf",
    }


def test_mof_validation_uses_four_quarter_changes() -> None:
    dates = pd.date_range("2020-03-31", periods=8, freq="QE-DEC")
    mof = pd.DataFrame(
        {
            "date": dates,
            "total_bank_borrowing_100m_yen": [100, 105, 103, 110, 120, 117, 130, 150],
        }
    )
    boj = pd.DataFrame(
        {
            "date": dates,
            "stock_manufacturing": [200, 210, 208, 215, 240, 235, 260, 300],
        }
    )

    result = validate_mof_manufacturing(mof, boj, _mof_metadata())

    assert result["measurement_interval"] == "four-quarter change"
    assert result["n_common_changes"] == 4
    assert result["direction_agreement"] == 1.0
    assert result["purpose_status"] == "not purpose validation"
    assert "loan purpose" in result["does_not_validate"]


def test_mlit_validation_is_limited_to_housing_branch() -> None:
    frame = pd.DataFrame(
        {
            "reference_period": ["FY2024"] * 5,
            "series_id": [
                "new_housing_purpose_share",
                "existing_housing_purpose_share",
                "refinancing_purpose_share",
                "apartment_loan_new_lending",
                "apartment_loan_outstanding",
            ],
            "value": [0.704, 0.231, 0.064, 38183.70, 367063.24],
        }
    )

    result = validate_mlit_housing(frame, _mlit_metadata())

    assert result["status"] == "published_cross_section"
    assert result["acquisition_purpose_share"] == pytest.approx(0.935)
    assert result["classified_share_rounding_gap"] == pytest.approx(0.001)
    assert "corporate real estate" in result["does_not_validate"]


def test_committed_snapshots_match_recorded_checksums() -> None:
    metadata = load_metadata(ROOT)

    checks = verify_snapshot_checksums(ROOT, metadata)

    assert set(checks) == {"mof_manufacturing", "mlit_private_housing"}
    assert all(item["verified"] for item in checks.values())


def test_repository_pipeline_keeps_partial_claim_boundary() -> None:
    summary = run_external_validation(ROOT)
    table = render_validation_table(summary)

    assert summary["mof_manufacturing"]["n_common_changes"] > 0
    assert summary["mof_manufacturing"]["purpose_status"] == "not purpose validation"
    assert summary["mlit_private_housing"]["status"] == "published_cross_section"
    assert "Partial external validation" in table
    assert "NFB: manufacturing" in table
    assert "PROP: household housing" in table
    assert "does not validate corporate real-estate loans" in table
