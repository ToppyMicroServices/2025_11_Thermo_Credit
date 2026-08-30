from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parent.parent
SPEC = importlib.util.spec_from_file_location(
    "boj_credit_destination_fetch",
    ROOT / "scripts" / "17_fetch_boj_jp_credit_destination.py",
)
assert SPEC is not None and SPEC.loader is not None
fetch = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(fetch)


def _raw_panel() -> pd.DataFrame:
    dates = pd.to_datetime(
        ["2009-06-30", "2009-09-30", "2009-12-31", "2010-03-31", "2010-06-30"]
    )
    return pd.DataFrame(
        {
            "date": dates,
            "G1_DLLI5DS2_X": [50.0, 60.0, 50.0, 55.0, 54.0],
            "G2_DLLI5DS2_X": [50.0, 40.0, 60.0, 53.0, 52.0],
            "GSUM_DLLI5DS2_X": [100.0, 100.0, 110.0, 108.0, 106.0],
            "B1_DLLI5DS2_X": [10.0, 12.0, 14.0, 16.0, 18.0],
            "E1_DLLI5DS2_X": [20.0, 22.0, 24.0, 26.0, 28.0],
            "U1_DLLI5DS2_X": [5.0, 5.0, 5.0, 5.0, 5.0],
            "G1_DLLI5DS5_X": [2.0, 3.0, np.nan, 4.0, 4.0],
            "G2_DLLI5DS5_X": [1.0, 1.0, 2.0, 2.0, 2.0],
            "GSUM_DLLI5DS5_X": [3.0, 4.0, 5.0, 6.0, 6.0],
            "B1_DLLI5DS5_X": [1.0, 1.0, 1.0, 1.0, 1.0],
            "E1_DLLI5DS5_X": [1.0, 1.0, 1.0, 1.0, 1.0],
            "U1_DLLI5DS5_X": [0.0, 0.0, 0.0, 0.0, 0.0],
            "DLLILKG90_DLLI5DS2T": [135.0, 139.0, 153.0, 155.0, 157.0],
            "DLLILKG90_DLLI5DS5T": [6.0, 7.0, 8.0, 9.0, 9.0],
        }
    )


def test_bucket_positive_measure_is_invariant_to_within_bucket_subdivision() -> None:
    raw = _raw_panel()
    split_map = (
        {"bucket": "G", "sector": "g1", "stock_code": "G1_DLLI5DS2_X"},
        {"bucket": "G", "sector": "g2", "stock_code": "G2_DLLI5DS2_X"},
        {"bucket": "B", "sector": "b1", "stock_code": "B1_DLLI5DS2_X"},
        {"bucket": "E", "sector": "e1", "stock_code": "E1_DLLI5DS2_X"},
        {"bucket": "U", "sector": "u1", "stock_code": "U1_DLLI5DS2_X"},
    )
    coarse_map = (
        {"bucket": "G", "sector": "g_sum", "stock_code": "GSUM_DLLI5DS2_X"},
        {"bucket": "B", "sector": "b1", "stock_code": "B1_DLLI5DS2_X"},
        {"bucket": "E", "sector": "e1", "stock_code": "E1_DLLI5DS2_X"},
        {"bucket": "U", "sector": "u1", "stock_code": "U1_DLLI5DS2_X"},
    )

    split = fetch._construct_destination_panel(raw, sector_map=split_map)
    coarse = fetch._construct_destination_panel(raw, sector_map=coarse_map)

    central = [
        "C_G",
        "C_B",
        "C_E",
        "C_t",
        "borrower_composition_G_1q",
        "borrower_composition_B_1q",
        "borrower_composition_E_1q",
        "borrower_composition_G_4q",
        "borrower_composition_B_4q",
        "borrower_composition_E_4q",
        "operating_borrower_share_1q",
        "operating_borrower_share_4q",
        "operating_borrower_share",
        "q_t",
        "mapped_domestic_stock",
    ]
    pd.testing.assert_frame_equal(split[central], coarse[central])
    assert split.loc[1, "C_G"] == 0.0
    assert split.loc[1, "C_G_series_positive"] == 10.0
    assert coarse.loc[1, "C_G_series_positive"] == 0.0
    assert split.loc[1, "delta_positive_g1"] == 10.0
    assert np.isclose(split.loc[4, "borrower_composition_G_4q"], 10.0 / 26.0)
    assert np.isclose(split.loc[4, "borrower_composition_B_4q"], 8.0 / 26.0)
    assert np.isclose(split.loc[4, "borrower_composition_E_4q"], 8.0 / 26.0)
    assert np.isclose(
        split.loc[
            4,
            [
                "borrower_composition_G_4q",
                "borrower_composition_B_4q",
                "borrower_composition_E_4q",
            ],
        ].sum(),
        1.0,
    )
    assert split.loc[4, "q_t"] == split.loc[4, "borrower_composition_G_4q"]
    assert (
        split.loc[4, "operating_borrower_share_4q"]
        == split.loc[4, "borrower_composition_G_4q"]
    )


def test_common_taxonomy_break_and_incomplete_purpose_cells_are_not_imputed() -> None:
    split_map = (
        {"bucket": "G", "sector": "g1", "stock_code": "G1_DLLI5DS2_X"},
        {"bucket": "G", "sector": "g2", "stock_code": "G2_DLLI5DS2_X"},
        {"bucket": "B", "sector": "b1", "stock_code": "B1_DLLI5DS2_X"},
        {"bucket": "E", "sector": "e1", "stock_code": "E1_DLLI5DS2_X"},
        {"bucket": "U", "sector": "u1", "stock_code": "U1_DLLI5DS2_X"},
    )
    panel = fetch._construct_destination_panel(_raw_panel(), sector_map=split_map)

    assert not bool(panel.loc[0, "common_taxonomy_delta_valid"])
    assert panel.loc[0, ["C_G", "C_B", "C_E", "C_t"]].isna().all()
    assert bool(panel.loc[1, "common_taxonomy_delta_valid"])
    assert np.isnan(panel.loc[2, "fixed_investment_new_G"])
    assert panel.loc[2, "mapped_domestic_stock"] == 148.0


def test_published_primary_taxonomy_uses_one_population_and_keeps_residuals_explicit() -> None:
    dates = pd.to_datetime(
        ["2009-06-30", "2009-09-30", "2009-12-31", "2010-03-31", "2010-06-30"]
    )
    steps = np.arange(5, dtype=float)
    sector_map = (
        {
            "bucket": "G",
            "sector": "manufacturing",
            "stock_code": "MAN_DLLI5DS2_X",
            "primary_bucket": "NFB",
            "werner_bucket": "COMPLEMENT",
            "muller_verner_bucket": "TRADABLE",
        },
        {
            "bucket": "G",
            "sector": "other_services",
            "stock_code": "SVC_DLLI5DS2_X",
            "primary_bucket": "NFB",
            "werner_bucket": "COMPLEMENT",
            "muller_verner_bucket": "OTHER_NFB",
        },
        {
            "bucket": "B",
            "sector": "construction",
            "stock_code": "CON_DLLI5DS2_X",
            "primary_bucket": "NFB",
            "werner_bucket": "FCP",
            "muller_verner_bucket": "NONTRADABLE",
        },
        {
            "bucket": "E",
            "sector": "finance_insurance",
            "stock_code": "FIN_DLLI5DS2_X",
            "primary_bucket": "FIN",
            "werner_bucket": "FCP",
            "muller_verner_bucket": "FIN",
        },
        {
            "bucket": "E",
            "sector": "real_estate",
            "stock_code": "REA_DLLI5DS2_X",
            "primary_bucket": "PROP",
            "werner_bucket": "FCP",
            "muller_verner_bucket": "NONTRADABLE",
        },
        {
            "bucket": "E",
            "sector": "households_housing_consumer_tax",
            "stock_code": fetch.HOUSEHOLD_TOTAL_STOCK_CODE,
            "primary_bucket": "HOUSEHOLD_SPLIT",
            "werner_bucket": "COMPLEMENT",
            "muller_verner_bucket": "HH",
        },
        {
            "bucket": "G",
            "sector": "other_organizations",
            "stock_code": "ORG_DLLI5DS2_X",
            "primary_bucket": "NFB_RESIDUAL_COMPONENT",
            "werner_bucket": "COMPLEMENT",
            "muller_verner_bucket": "UNRESOLVED",
        },
        {
            "bucket": "G",
            "sector": "local_governments",
            "stock_code": "LOC_DLLI5DS2_X",
            "primary_bucket": "EXCLUDE_LOCAL_GOVERNMENT",
            "werner_bucket": "EXCLUDE_LOCAL_GOVERNMENT",
            "muller_verner_bucket": "EXCLUDE_LOCAL_GOVERNMENT",
        },
        {
            "bucket": "U",
            "sector": "overseas_yen_and_transferred_loans",
            "stock_code": "OVR_DLLI5DS2_X",
            "primary_bucket": "NFB_RESIDUAL_COMPONENT",
            "werner_bucket": "COMPLEMENT",
            "muller_verner_bucket": "UNRESOLVED",
        },
    )
    raw = pd.DataFrame(
        {
            "date": dates,
            "MAN_DLLI5DS2_X": 100.0 + 10.0 * steps,
            "SVC_DLLI5DS2_X": 10.0 + steps,
            "CON_DLLI5DS2_X": 20.0 + 2.0 * steps,
            "FIN_DLLI5DS2_X": 10.0 + steps,
            "REA_DLLI5DS2_X": 30.0 + 3.0 * steps,
            fetch.HOUSEHOLD_TOTAL_STOCK_CODE: 80.0 + 4.0 * steps,
            fetch.HOUSEHOLD_HOUSING_STOCK_CODE: 50.0 + 3.0 * steps,
            fetch.HOUSEHOLD_CONSUMER_STOCK_CODE: 20.0 + steps,
            "ORG_DLLI5DS2_X": 2.0,
            "LOC_DLLI5DS2_X": 7.0 + steps,
            "OVR_DLLI5DS2_X": 5.0,
            fetch.OFFICIAL_TOTAL_STOCK_CODE: 265.0 + 22.0 * steps,
        }
    )

    panel = fetch._construct_destination_panel(raw, sector_map=sector_map)

    assert panel.loc[1, "C_NFB"] == 13.0
    assert panel.loc[1, "C_FIN"] == 1.0
    assert panel.loc[1, "C_PROP"] == 6.0
    assert panel.loc[1, "C_HH_NONHOUSING"] == 1.0
    assert panel.loc[1, "C_t"] == 21.0
    assert panel.loc[1, "C_t_raw_delta"] == 21.0
    assert panel.loc[1, "primary_included_stock"] == panel.loc[1, "stock_total"] - 8.0
    assert panel.loc[1, "mapped_domestic_stock"] == panel.loc[1, "legacy_mapped_domestic_stock"]
    assert panel.loc[1, "mapped_domestic_stock"] != panel.loc[1, "primary_included_stock"]
    assert panel.loc[1, "stock_overseas_explicit"] == 5.0
    assert panel.loc[1, "stock_unresolved_residual"] == 3.0
    assert panel["primary_nfb_residual_identity_gap_stock"].abs().max() == 0.0
    assert panel["werner_population_gap_stock"].abs().max() == 0.0
    assert panel["muller_verner_population_gap_stock"].abs().max() == 0.0
    assert panel["explicit_scope_gap_to_official_stock"].abs().max() == 0.0
    assert np.isclose(panel.loc[4, "q_t"], 13.0 / 21.0)
    assert panel.loc[4, "q_t"] == panel.loc[4, "borrower_composition_NFB_4q"]
    assert panel.loc[4, "legacy_q_t"] == panel.loc[4, "borrower_composition_G_4q"]
