from __future__ import annotations

import pandas as pd

from lib.boj_bridge_validation import (
    main_mapping_rows,
    mapping_rows,
    primary_mapping_rows,
    render_main_mapping_table,
    render_primary_mapping_table,
    render_taxonomy_robustness_table,
    render_validation_table,
    taxonomy_robustness_rows,
    validation_rows,
)
from lib.boj_credit_taxonomies import (
    PRIMARY_TAXONOMY_ID,
    TAXONOMY_METADATA,
    TAXONOMY_SELECTION_RULE,
)


def test_boj_bridge_validation_reports_core_audits() -> None:
    frame = pd.DataFrame(
        {
            "C_t": [10.0, 20.0, 30.0],
            "C_G": [4.0, 8.0, 12.0],
            "C_B": [2.0, 4.0, 6.0],
            "C_E": [4.0, 8.0, 12.0],
            "C_G_series_positive": [5.0, 10.0, 15.0],
            "C_B_series_positive": [2.0, 4.0, 6.0],
            "C_E_series_positive": [4.0, 8.0, 12.0],
            "C_G_net": [4.0, -2.0, 12.0],
            "C_B_net": [2.0, 1.0, 6.0],
            "C_E_net": [4.0, 5.0, 12.0],
            "total_positive_flow": [10.0, 20.0, 30.0],
            "fixed_investment_new_G": [2.0, 4.0, 6.0],
            "fixed_investment_new_B": [1.0, 2.0, 3.0],
            "fixed_investment_new_E": [1.0, 2.0, 3.0],
            "delta_net_real_estate": [1.0, 2.0, 3.0],
            "delta_net_finance_insurance": [1.0, -1.0, 2.0],
            "delta_net_households_housing_consumer_tax": [2.0, 3.0, 4.0],
        }
    )

    rows = validation_rows(frame, lambda_b=0.5)

    assert [row["audit"] for row in rows] == [
        "One-quarter signed-change diagnostic",
        "Aggregation-order sensitivity",
        "Quarter-of-year denominator availability",
        "Low-denominator stress",
        "Flow-floor sensitivity",
        "Four-quarter measurement interval",
        "Four-quarter within-BOJ purpose-coded comparator",
        "Construction-borrower reclassification",
        "Real-estate borrower reclassification",
        "Finance and household borrower alternatives",
        "Series availability and aggregate reconciliation",
        "Refinancing, rollovers, write-offs, and reclassification",
    ]
    aggregation = next(row for row in rows if row["audit"] == "Aggregation-order sensitivity")
    assert "median |delta q|" in aggregation["readout"]


def test_signed_audit_keeps_negative_aggregate_denominators() -> None:
    frame = pd.DataFrame(
        {
            "C_t": [10.0, 10.0, 10.0, 10.0],
            "C_G": [4.0, 4.0, 4.0, 4.0],
            "C_B": [2.0, 2.0, 2.0, 2.0],
            "C_E": [4.0, 4.0, 4.0, 4.0],
            "C_G_net": [4.0, -4.0, 2.0, 1.0],
            "C_B_net": [2.0, -2.0, -1.0, -1.0],
            "C_E_net": [4.0, -4.0, 1.0, 1e-12],
        }
    )

    signed = next(
        row
        for row in validation_rows(frame)
        if row["audit"] == "One-quarter signed-change diagnostic"
    )

    assert "1Q signed N=3" in signed["readout"]
    assert "aggregate contractions=1" in signed["readout"]
    assert "paired N=3" in signed["readout"]
    assert "not the primary four-quarter q_t" in signed["interpretation"]


def test_alternative_allocation_uses_four_quarter_primary_coordinate() -> None:
    group_b = [0.0, 0.0, 0.0, 8.0, 0.0, 0.0, 0.0, 0.0]
    frame = pd.DataFrame(
        {
            "date": pd.date_range("2020-03-31", periods=8, freq="QE-DEC"),
            "C_t": [2.0 + value for value in group_b],
            "C_G": [1.0] * 8,
            "C_B": group_b,
            "C_E": [1.0] * 8,
            "C_G_net": [1.0] * 8,
            "C_B_net": group_b,
            "C_E_net": [1.0] * 8,
        }
    )

    construction = next(
        row
        for row in validation_rows(frame)
        if row["audit"] == "Construction-borrower reclassification"
    )

    assert "4Q median |share shift|" in construction["readout"]
    assert "is 0.50" in construction["readout"]


def test_boj_bridge_validation_reports_net_change_reconciliation() -> None:
    frame = pd.DataFrame(
        {
            "date": pd.date_range("2020-03-31", periods=4, freq="QE-DEC"),
            "C_t": [10.0, 20.0, 30.0, 40.0],
            "C_G": [4.0, 8.0, 12.0, 16.0],
            "C_B": [2.0, 4.0, 6.0, 8.0],
            "C_E": [4.0, 8.0, 12.0, 16.0],
            "stock_sector_a": [40.0, 50.0, 65.0, 80.0],
            "stock_sector_b": [60.0, 70.0, 85.0, 100.0],
            "stock_total": [100.0, 120.0, 151.0, 181.0],
        }
    )

    rows = validation_rows(frame, lambda_b=0.5)
    reconciliation = next(
        row for row in rows if row["audit"] == "Series availability and aggregate reconciliation"
    )

    assert "net-change N=3" in reconciliation["readout"]
    assert "median gap=0 (JPY 100m)" in reconciliation["readout"]


def test_boj_mapping_rows_include_required_columns() -> None:
    frame = pd.DataFrame(
        {
            "C_G": [4.0],
            "C_B": [1.0],
            "C_E": [5.0],
            "unclassified_positive_flow": [0.5],
            "total_positive_flow": [10.5],
            "fixed_investment_new_total": [3.0],
        }
    )
    metadata = {
        "bucket_mapping": [
            {"bucket": "G", "sector": "manufacturing", "stock_code": "G1"},
            {"bucket": "B", "sector": "construction", "stock_code": "B1"},
            {"bucket": "E", "sector": "real_estate", "stock_code": "E1"},
            {"bucket": "U", "sector": "overseas", "stock_code": "U1"},
        ]
    }

    row = mapping_rows(frame, metadata)[0]

    assert set(row) == {
        "source",
        "borrower",
        "bucket",
        "rationale",
        "ambiguous",
        "type",
        "bias",
        "negative",
        "lag",
        "coverage",
    }


def test_boj_main_mapping_table_is_compact() -> None:
    frame = pd.DataFrame(
        {
            "C_G": [4.0],
            "C_B": [1.0],
            "C_E": [5.0],
            "unclassified_positive_flow": [0.5],
            "total_positive_flow": [10.5],
            "fixed_investment_new_total": [3.0],
        }
    )
    metadata = {
        "bucket_mapping": [
            {"bucket": "G", "sector": "manufacturing", "stock_code": "G1"},
            {"bucket": "B", "sector": "construction", "stock_code": "B1"},
            {"bucket": "E", "sector": "real_estate", "stock_code": "E1"},
            {"bucket": "U", "sector": "overseas", "stock_code": "U1"},
        ]
    }

    rows = main_mapping_rows(frame, metadata)
    table = render_main_mapping_table(frame, metadata)

    assert len(rows) == 5
    assert "Author-defined BOJ borrower-composition groups" in table
    assert "Group G: broad sectors, local governments, and other organisations" in table
    assert "Group B: construction" in table
    assert "Group E: finance, real estate, households" in table
    assert "$P_t^G$" in table
    assert "$P_t^B$" in table
    assert "$P_t^E$" in table
    assert rows[-1]["share"] == "claim-boundary audit"
    assert "operating" not in table.lower()
    assert "Ambiguous cases" not in table
    assert "Detailed BOJ" not in table


def test_rendered_validation_table_keeps_reviewer_priority_rows_only() -> None:
    frame = pd.DataFrame(
        {
            "date": pd.date_range("2020-03-31", periods=4, freq="QE-DEC"),
            "C_t": [10.0, 20.0, 30.0, 40.0],
            "C_G": [4.0, 8.0, 12.0, 16.0],
            "C_B": [2.0, 4.0, 6.0, 8.0],
            "C_E": [4.0, 8.0, 12.0, 16.0],
            "C_G_net": [4.0, 8.0, 12.0, 16.0],
            "C_B_net": [2.0, 4.0, 6.0, 8.0],
            "C_E_net": [4.0, 8.0, 12.0, 16.0],
            "C_G_series_positive": [5.0, 9.0, 13.0, 17.0],
            "C_B_series_positive": [2.0, 4.0, 6.0, 8.0],
            "C_E_series_positive": [4.0, 8.0, 12.0, 16.0],
            "fixed_investment_new_G": [2.0, 4.0, 6.0, 8.0],
            "fixed_investment_new_B": [1.0, 2.0, 3.0, 4.0],
            "fixed_investment_new_E": [1.0, 2.0, 3.0, 4.0],
        }
    )

    table = render_validation_table(frame)

    assert "BOJ borrower-composition audits" in table
    assert "Aggregation-order sensitivity" in table
    assert "Four-quarter within-BOJ purpose-coded comparator" in table
    assert "Series availability and aggregate reconciliation" in table
    assert "Low-denominator stress" not in table
    assert "Flow-floor sensitivity" not in table
    assert "Four-quarter measurement interval" not in table
    assert "Real-estate borrower reclassification" in table
    assert "Refinancing, rollovers, write-offs" not in table
    assert "criterion" not in table.lower()


def test_published_taxonomy_tables_and_audits_keep_population_identities_visible() -> None:
    steps = pd.Series(range(5), dtype=float)
    frame = pd.DataFrame(
        {
            "date": pd.date_range("2020-03-31", periods=5, freq="QE-DEC"),
            "C_t": [20.0] * 5,
            "C_t_primary": [20.0] * 5,
            "C_t_raw_delta": [20.0] * 5,
            "primary_included_net_flow": [20.0] * 5,
            "C_NFB": [6.0] * 5,
            "C_FIN": [2.0] * 5,
            "C_PROP": [8.0] * 5,
            "C_HH_NONHOUSING": [4.0] * 5,
            "q_t": [float("nan"), float("nan"), float("nan"), float("nan"), 0.3],
            "primary_taxonomy_delta_valid": [False, True, True, True, True],
            "stock_primary_nfb": 100.0 + 6.0 * steps,
            "stock_primary_fin": 20.0 + 2.0 * steps,
            "stock_primary_prop": 50.0 + 8.0 * steps,
            "stock_primary_hh_nonhousing": 30.0 + 4.0 * steps,
            "primary_included_stock": 200.0 + 20.0 * steps,
            "stock_primary_nfb_mapped_detail": 90.0 + 5.0 * steps,
            "stock_overseas_explicit": [8.0] * 5,
            "stock_unresolved_residual": 2.0 + steps,
            "primary_nfb_residual_identity_gap_stock": [0.0] * 5,
            "stock_household_total": 70.0 + 10.0 * steps,
            "stock_household_housing": 40.0 + 6.0 * steps,
            "stock_household_nonhousing": 30.0 + 4.0 * steps,
            "stock_household_consumer_purpose": 20.0 + 3.0 * steps,
            "stock_total": 210.0 + 21.0 * steps,
            "stock_local_governments_explicit": 10.0 + steps,
            "explicit_scope_gap_to_official_stock": [0.0] * 5,
            "primary_positive_flow": [20.0] * 5,
            "explicit_scope_positive_flow": [21.0] * 5,
            "local_governments_positive_flow": [1.0] * 5,
            "overseas_positive_flow": [0.0] * 5,
            "unresolved_residual_positive_flow": [1.0] * 5,
            "werner_population_gap_stock": [0.0] * 5,
            "muller_verner_population_gap_stock": [0.0] * 5,
        }
    )
    metadata = {
        "primary_taxonomy_id": PRIMARY_TAXONOMY_ID,
        "taxonomies": TAXONOMY_METADATA,
        "taxonomy_selection": TAXONOMY_SELECTION_RULE,
    }

    mapping = primary_mapping_rows(frame, metadata)
    main = main_mapping_rows(frame, metadata)
    audits = validation_rows(frame)
    main_table = render_main_mapping_table(frame, metadata)
    detail_table = render_primary_mapping_table(frame, metadata)
    robustness_table = render_taxonomy_robustness_table(metadata)
    robustness = taxonomy_robustness_rows(metadata)

    assert mapping[:7] == main
    assert len(mapping) == 9
    assert len(main) == 7
    assert any(row["role"] == r"$P_t^{NFB}$" for row in mapping)
    assert any(row["source"] == "Overseas-linked loans" for row in mapping)
    assert any(row["source"] == "Unresolved residual" for row in mapping)
    assert [row["audit"] for row in audits[:6]] == [
        "Primary same-population identity",
        "Published residual NFB decomposition",
        "Household non-housing split",
        "Explicit official-population coverage",
        "Literature-anchored taxonomy population identity",
        "Primary four-quarter vector availability",
    ]
    identity = audits[0]["readout"]
    assert "max |C_t-sum four buckets|=0.000000" in identity
    assert "max |included stock-sum four stocks|=0.000000" in identity
    assert "Literature-anchored BOJ credit-allocation crosswalk" in main_table
    assert "Published-taxonomy BOJ crosswalk" in detail_table
    assert "not an externally time-stamped preregistration" in robustness_table
    assert len(robustness) == 4
    assert TAXONOMY_SELECTION_RULE["oos_results_used"] is False
