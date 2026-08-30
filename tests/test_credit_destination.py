from __future__ import annotations

import numpy as np
import pandas as pd

from lib.credit_destination import build_credit_destination_panel


def test_credit_destination_constructs_core_split_from_mece_proxy() -> None:
    dates = pd.date_range("2020-03-31", periods=3, freq="QE-DEC")
    frame = pd.DataFrame(
        {
            "date": dates,
            "L_real": [100.0, 120.0, 150.0],
            "q_productive": [0.30, 0.30, 0.30],
            "q_housing": [0.20, 0.20, 0.20],
            "q_financial": [0.25, 0.25, 0.25],
        }
    )
    cfg = {"credit_destination": {"lambda_B": 0.5, "housing_construction_share": 0.4}}

    out = build_credit_destination_panel(frame, cfg)

    assert list(out.columns)[:9] == ["date", "C_t", "C_t_raw_delta", "C_G", "C_B", "C_E", "C_R", "C_A", "q_t"]
    assert out.loc[1, "C_t"] == 20.0
    assert out.loc[1, "C_G"] == 6.0
    assert np.isclose(out.loc[1, "C_B"], 1.6)
    assert np.isclose(out.loc[1, "C_E"], 7.4)
    assert np.isclose(out.loc[1, "q_t"], 0.30 + 0.5 * 0.20 * 0.4)
    assert out.loc[1, "credit_destination_source"] == "allocation_proxy"


def test_credit_destination_falls_back_from_legacy_allocation_columns() -> None:
    frame = pd.DataFrame(
        {
            "date": pd.date_range("2020-03-31", periods=3, freq="QE-DEC"),
            "L_real": [100.0, 110.0, 130.0],
            "q_pay": [0.35, 0.35, 0.35],
            "q_firm": [0.30, 0.30, 0.30],
            "q_asset": [0.20, 0.20, 0.20],
            "q_reserve": [0.15, 0.15, 0.15],
        }
    )

    out = build_credit_destination_panel(frame, {"credit_destination": {"household_housing_share": 0.4}})

    expected_housing = 0.35 * 0.4
    assert np.isclose(out.loc[2, "share_G_proxy"], 0.30)
    assert np.isclose(out.loc[2, "share_B_proxy"], expected_housing * 0.35)
    assert np.isclose(out.loc[2, "share_E_proxy"], 0.20 + expected_housing * 0.65)
    assert np.isfinite(out.loc[2, "q_t"])


def test_credit_destination_uses_only_positive_new_credit_flow() -> None:
    frame = pd.DataFrame(
        {
            "date": pd.date_range("2020-03-31", periods=3, freq="QE-DEC"),
            "L_real": [100.0, 90.0, 95.0],
            "q_productive": [0.30, 0.30, 0.30],
        }
    )

    out = build_credit_destination_panel(frame, {})

    assert out.loc[1, "C_t"] == 0.0
    assert np.isnan(out.loc[1, "q_t"])
    assert out.loc[2, "C_t"] == 5.0


def test_credit_destination_prefers_observed_cgbe_components() -> None:
    frame = pd.DataFrame(
        {
            "date": pd.date_range("2020-03-31", periods=5, freq="QE-DEC"),
            "L_real": [100.0, 1000.0, 1010.0, 1020.0, 1030.0],
            "q_productive": [0.10] * 5,
            "C_t": [10.0, 20.0, 20.0, 20.0, 20.0],
            "C_G": [4.0, 6.0, 6.0, 6.0, 6.0],
            "C_B": [2.0, 4.0, 4.0, 4.0, 4.0],
            "C_E": [4.0, 10.0, 10.0, 10.0, 10.0],
            "common_taxonomy_delta_valid": [0.0, 1.0, 1.0, 1.0, 1.0],
            "destination_coverage_observed": [0.8, 0.9, 0.9, 0.9, 0.9],
        }
    )

    out = build_credit_destination_panel(
        frame,
        {"credit_destination": {"source": "jp_boj_sector_observed", "lambda_B": 0.5}},
    )

    assert out.loc[1, "C_G"] == 6.0
    assert out.loc[1, "C_B"] == 4.0
    assert out.loc[1, "C_E"] == 10.0
    assert np.isnan(out.loc[1, "q_t"])
    assert np.isclose(out.loc[4, "q_t"], 6.0 / 20.0)
    assert np.isclose(out.loc[1, "borrower_composition_G_1q"], 6.0 / 20.0)
    assert np.isclose(out.loc[1, "borrower_composition_B_1q"], 4.0 / 20.0)
    assert np.isclose(out.loc[1, "borrower_composition_E_1q"], 10.0 / 20.0)
    assert np.isclose(
        out.loc[1, ["borrower_composition_G_1q", "borrower_composition_B_1q", "borrower_composition_E_1q"]].sum(),
        1.0,
    )
    assert np.isclose(out.loc[4, "borrower_composition_G_4q"], 6.0 / 20.0)
    assert np.isclose(out.loc[4, "borrower_composition_B_4q"], 4.0 / 20.0)
    assert np.isclose(out.loc[4, "borrower_composition_E_4q"], 10.0 / 20.0)
    assert np.isclose(
        out.loc[4, ["borrower_composition_G_4q", "borrower_composition_B_4q", "borrower_composition_E_4q"]].sum(),
        1.0,
    )
    assert out.loc[4, "q_t"] == out.loc[4, "borrower_composition_G_4q"]
    assert out.loc[1, "operating_borrower_share_1q"] == out.loc[1, "borrower_composition_G_1q"]
    assert out.loc[4, "operating_borrower_share_4q"] == out.loc[4, "borrower_composition_G_4q"]
    assert np.isclose(out.loc[1, "C_R"], 6.0)
    assert np.isclose(out.loc[1, "C_A"], 14.0)
    assert np.isnan(out.loc[1, "lambda_B"])
    assert np.isclose(out.loc[1, "share_E_direct"], 0.5)
    assert out.loc[1, "credit_destination_source"] == "jp_boj_sector_observed"
    assert np.isclose(out.loc[1, "destination_coverage"], 0.9)


def test_credit_destination_prefers_published_four_bucket_taxonomy() -> None:
    frame = pd.DataFrame(
        {
            "date": pd.date_range("2020-03-31", periods=5, freq="QE-DEC"),
            "C_t": [20.0] * 5,
            "C_t_raw_delta": [15.0] * 5,
            "C_NFB": [6.0] * 5,
            "C_FIN": [2.0] * 5,
            "C_PROP": [8.0] * 5,
            "C_HH_NONHOUSING": [4.0] * 5,
            "C_G": [4.0] * 5,
            "C_B": [3.0] * 5,
            "C_E": [13.0] * 5,
            "primary_taxonomy_delta_valid": [False, True, True, True, True],
            "common_taxonomy_delta_valid": [False, True, True, True, True],
            "primary_flow_coverage_observed": [0.95] * 5,
            "legacy_q_t": [0.2] * 5,
            "werner_population_gap_stock": [0.0] * 5,
            "muller_verner_population_gap_stock": [0.0] * 5,
        }
    )

    out = build_credit_destination_panel(
        frame,
        {"credit_destination": {"source": "jp_boj_sector_observed"}},
    )

    assert out.loc[1, "C_t"] == 20.0
    assert out.loc[1, "C_NFB"] == 6.0
    assert out.loc[1, "C_FIN"] == 2.0
    assert out.loc[1, "C_PROP"] == 8.0
    assert out.loc[1, "C_HH_NONHOUSING"] == 4.0
    assert np.isnan(out.loc[1, "q_t"])
    assert np.isclose(out.loc[4, "q_t"], 0.3)
    assert out.loc[4, "q_t"] == out.loc[4, "borrower_composition_NFB_4q"]
    assert out.loc[1, "legacy_q_t"] == 0.2
    assert out.loc[1, "destination_coverage"] == 0.95
    assert (
        out.loc[1, "credit_destination_taxonomy_id"]
        == "bezemer_samarina_zhang_2020_japan_v1"
    )
    assert not bool(out["primary_component_total_mismatch"].any())
