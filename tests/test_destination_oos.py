from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

from lib.destination_oos import (
    ALLOCATION_FEATURE_COLUMN,
    BOJ_COMMON_TAXONOMY_START,
    BOJ_BASELINE_UNIVERSE,
    BOJ_MATCHED_STOCK_COLUMN,
    LEGACY_ALLOCATION_FEATURE_COLUMN,
    _build_boj_universe_asof,
    _score_destination_model,
    render_destination_oos_asset_auxiliary_tex,
    render_destination_oos_tex,
    run_destination_oos,
    write_destination_oos_outputs,
)
from lib.baseline_forecast import ForecastTarget


def _jp_frame(periods: int = 96) -> pd.DataFrame:
    dates = pd.date_range("2000-03-31", periods=periods, freq="QE-DEC")
    t = np.arange(periods, dtype=float)
    q_t = 0.55 - 0.08 * np.sin(t / 8.0)
    credit = 100.0 * np.exp(0.010 * t)
    c_t = np.maximum(np.diff(np.r_[credit[0], credit]), 0.1)
    c_a = (1.0 - q_t) * c_t
    return pd.DataFrame(
        {
            "date": dates,
            "Y": 100.0 * np.exp(0.006 * t),
            "L_real": credit,
            "L_asset": 80.0 * np.exp(0.008 * t + 0.015 * (1.0 - q_t)),
            "spread": 1.5 + 0.3 * (1.0 - q_t) + 0.05 * np.sin(t / 3.0),
            "q_t": q_t,
            "C_t": c_t,
            "C_A": c_a,
            "one_minus_q_t": 1.0 - q_t,
            "T_L": 0.4 + 0.1 * np.sin(t / 5.0),
            "p_C": 0.2 + 0.03 * np.cos(t / 6.0),
            "V_C": 20.0 + 0.2 * np.cos(t / 7.0),
        }
    )


def _boj_direct_frame(periods: int = 96) -> pd.DataFrame:
    dates = pd.date_range("2000-03-31", periods=periods, freq="QE-DEC")
    t = np.arange(periods, dtype=float)
    stock_nfb = 400.0 * np.exp(0.009 * t + 0.006 * np.sin(t / 4.0))
    stock_fin = 90.0 * np.exp(0.006 * t + 0.004 * np.cos(t / 5.0))
    stock_prop = 220.0 * np.exp(0.011 * t - 0.005 * np.sin(t / 6.0))
    stock_hh = 110.0 * np.exp(0.008 * t + 0.003 * np.cos(t / 7.0))
    c_nfb = pd.Series(stock_nfb).diff().clip(lower=0.0)
    c_fin = pd.Series(stock_fin).diff().clip(lower=0.0)
    c_prop = pd.Series(stock_prop).diff().clip(lower=0.0)
    c_hh = pd.Series(stock_hh).diff().clip(lower=0.0)
    c_fcp = c_fin + 0.55 * c_prop + 0.15 * c_nfb
    c_complement = c_nfb + c_prop + c_hh - c_fcp
    c_nontradable = 0.65 * c_nfb + 0.70 * c_prop
    c_tradable = 0.25 * c_nfb
    c_other_nfb = 0.10 * c_nfb
    c_mv_fin = c_fin
    c_mv_hh = c_hh
    c_mv_unresolved = 0.30 * c_prop
    return pd.DataFrame(
        {
            "date": dates,
            "primary_included_stock": stock_nfb + stock_fin + stock_prop + stock_hh,
            "C_NFB": c_nfb,
            "C_FIN": c_fin,
            "C_PROP": c_prop,
            "C_HH_NONHOUSING": c_hh,
            "C_WERNER_FCP": c_fcp,
            "C_WERNER_COMPLEMENT": c_complement,
            "C_MV_NONTRADABLE": c_nontradable,
            "C_MV_TRADABLE": c_tradable,
            "C_MV_OTHER_NFB": c_other_nfb,
            "C_MV_FIN": c_mv_fin,
            "C_MV_HH": c_mv_hh,
            "C_MV_UNRESOLVED": c_mv_unresolved,
        }
    )


def test_destination_oos_reports_focused_jp_models(tmp_path) -> None:
    site = tmp_path / "site"
    site.mkdir()
    _jp_frame().to_csv(site / "indicators_realtime.csv", index=False)
    data = tmp_path / "data"
    data.mkdir()
    _boj_direct_frame().to_csv(data / "credit_destination_jp.csv", index=False)

    results = run_destination_oos(site)
    tex = render_destination_oos_tex(results)
    auxiliary_tex = render_destination_oos_asset_auxiliary_tex(results)

    assert not results.empty
    assert set(results["horizon_quarters"]) == {4, 8}
    assert {
        "q_t_only",
        "matched_credit_plus_q_t",
        "matched_credit_plus_complement_identity",
    }.issubset(set(results["model"]))
    assert set(results["target"]) == {"asset_acceleration", "spread_widening"}
    assert set(results["baseline"]) == {"boj_primary_included_stock_growth"}
    assert set(results["baseline_universe"]) == {BOJ_BASELINE_UNIVERSE}
    assert not results["model_features"].str.contains("operating").any()
    assert not results["effect_feature"].str.contains("operating").any()
    assert set(
        results.loc[
            results["allocation_measure"].eq("bezemer_nfb_4q"),
            "allocation_definition",
        ]
    ) == {"sum_4Q(NFB)/sum_4Q(NFB+FIN+PROP+HH_NONHOUSING)"}
    assert set(
        results.loc[
            results["allocation_measure"].eq("werner_fcp_4q"),
            "allocation_definition",
        ]
    ) == {"sum_4Q(FCP)/sum_4Q(FCP+COMPLEMENT)"}
    assert set(
        results.loc[
            results["allocation_measure"].eq(
                "muller_verner_nontradable_4q"
            ),
            "allocation_definition",
        ]
    ) == {
        "sum_4Q(NONTRADABLE)/"
        "sum_4Q(NONTRADABLE+TRADABLE+OTHER_NFB+FIN+HH+UNRESOLVED)"
    }
    assert set(results["allocation_measure"]) == {
        "bezemer_nfb_4q",
        "werner_fcp_4q",
        "muller_verner_nontradable_4q",
        "bezemer_nfb_1q",
        "werner_fcp_1q",
        "muller_verner_nontradable_1q",
    }
    assert set(results["min_training_rows_setting"]) == {20, 24, 28}
    assert set(
        results.loc[
            results["is_primary_allocation_measure"],
            "allocation_measure",
        ]
    ) == {"bezemer_nfb_4q"}
    assert set(
        results.loc[
            results["is_primary_training_window"],
            "min_training_rows_setting",
        ]
    ) == {28}
    assert set(results["boj_data_source"]) == {"data/credit_destination_jp.csv"}
    assert "Bridge application: JP borrower-composition pseudo-OOS" in tex
    assert "Bezemer: NFB (primary)" in tex
    assert "Werner-inspired BOJ proxy" in tex
    assert r'M\"uller--Verner: non-tradable' in tex
    assert "Matched-stock RMSE" in tex
    assert "Mean $\\Delta$ squared loss" in tex
    assert "not an RMSE difference" in tex
    assert "$\\Delta$ RMSE" not in tex
    assert "20-case minimum" in tex
    assert "24-case minimum" in tex
    assert "long-term JGB yield change" in tex
    assert "& Outcome &" not in tex
    assert "n/a" not in tex
    assert "BOJ balance-sheet acceleration" not in tex
    assert "nominal-GDP" not in tex
    assert "Clark" not in tex
    assert "Auxiliary BOJ balance-sheet acceleration" in auxiliary_tex
    assert "20-case minimum" not in auxiliary_tex

    write_destination_oos_outputs(results, root=tmp_path)
    written = pd.read_csv(site / "destination_oos_incremental.csv")
    payload = json.loads(
        (data / "destination_oos_incremental_summary.json").read_text(
            encoding="utf-8"
        )
    )
    assert set(written["min_training_rows_setting"]) == {20, 24, 28}
    assert set(written["allocation_measure"]) == {
        "bezemer_nfb_4q",
        "werner_fcp_4q",
        "muller_verner_nontradable_4q",
        "bezemer_nfb_1q",
        "werner_fcp_1q",
        "muller_verner_nontradable_1q",
    }
    assert payload["primary_allocation_measure"] == "bezemer_nfb_4q"
    assert payload["allocation_definition"] == (
        "sum_4Q(NFB)/sum_4Q(NFB+FIN+PROP+HH_NONHOUSING)"
    )
    assert payload["borrower_composition_feature"] == "borrower_composition_coordinate"
    assert payload["legacy_feature_aliases"] == {
        "borrower_composition_G": "borrower_composition_coordinate"
    }
    assert payload["min_training_rows_settings"] == [20, 24, 28]
    assert payload["primary_min_training_rows"] == 28

    paired = results[
        results["model"].isin(
            {"matched_credit_plus_q_t", "matched_credit_plus_complement_identity"}
        )
        & results["is_primary_allocation_measure"]
        & results["is_primary_training_window"]
    ].pivot(
        index=["horizon_quarters", "target"],
        columns="model",
        values="metric_loss_diff",
    )
    np.testing.assert_allclose(
        paired["matched_credit_plus_q_t"],
        paired["matched_credit_plus_complement_identity"],
        rtol=0.0,
        atol=1e-12,
    )


def test_boj_universe_respects_taxonomy_break_and_release_lag() -> None:
    direct = pd.DataFrame(
        {
            "date": pd.to_datetime(
                ["2009-03-31", "2009-06-30", "2009-09-30", "2009-12-31"]
            ),
            "primary_included_stock": [300.0, 500.0, 520.0, 532.0],
            # The 2009Q2 values are numerical but cross the taxonomy break.
            "C_NFB": [1.0, 100.0, 12.0, 6.0],
            "C_FIN": [1.0, 30.0, 2.0, 2.0],
            "C_PROP": [1.0, 50.0, 4.0, 2.0],
            "C_HH_NONHOUSING": [1.0, 20.0, 2.0, 2.0],
            "C_WERNER_FCP": [1.0, 80.0, 8.0, 4.0],
            "C_WERNER_COMPLEMENT": [1.0, 120.0, 12.0, 8.0],
        }
    )
    origins = pd.Series(
        pd.to_datetime(["2009-06-30", "2009-09-30", "2009-12-31"])
    )

    asof = _build_boj_universe_asof(
        origins,
        direct,
        allocation_measures=("bezemer_nfb_1q", "werner_fcp_1q"),
        primary_allocation_measure="bezemer_nfb_1q",
        release_lag_days=90,
    )

    assert pd.isna(asof.loc[0, BOJ_MATCHED_STOCK_COLUMN])
    assert asof.loc[1, BOJ_MATCHED_STOCK_COLUMN] == pytest.approx(500.0)
    primary_column = f"{ALLOCATION_FEATURE_COLUMN}__bezemer_nfb_1q"
    legacy_column = f"{LEGACY_ALLOCATION_FEATURE_COLUMN}__bezemer_nfb_1q"
    assert pd.isna(asof.loc[1, primary_column])
    assert asof.loc[2, primary_column] == pytest.approx(
        12.0 / 20.0
    )
    assert asof.loc[
        2,
        "borrower_composition_coordinate__werner_fcp_1q",
    ] == pytest.approx(8.0 / 20.0)
    pd.testing.assert_series_equal(
        asof[legacy_column],
        asof[primary_column],
        check_names=False,
    )
    assert asof.loc[1, "boj_source_date"] == BOJ_COMMON_TAXONOMY_START
    assert asof.loc[2, "boj_source_date"] == pd.Timestamp("2009-09-30")


def test_four_quarter_share_uses_four_valid_post_break_flows() -> None:
    direct = pd.DataFrame(
        {
            "date": pd.to_datetime(
                [
                    "2009-06-30",
                    "2009-09-30",
                    "2009-12-31",
                    "2010-03-31",
                    "2010-06-30",
                ]
            ),
            "primary_included_stock": [500.0, 520.0, 532.0, 548.0, 568.0],
            # The large first row is a cross-break value and must not enter
            # the first four-quarter construction.
            "C_NFB": [100.0, 12.0, 6.0, 8.0, 10.0],
            "C_FIN": [30.0, 2.0, 2.0, 2.0, 2.0],
            "C_PROP": [50.0, 4.0, 2.0, 4.0, 6.0],
            "C_HH_NONHOUSING": [20.0, 2.0, 2.0, 2.0, 2.0],
        }
    )
    origins = pd.Series(pd.to_datetime(["2010-06-30", "2010-09-30"]))

    asof = _build_boj_universe_asof(
        origins,
        direct,
        allocation_measures=("bezemer_nfb_4q",),
        release_lag_days=90,
    )

    column = f"{ALLOCATION_FEATURE_COLUMN}__bezemer_nfb_4q"
    assert pd.isna(asof.loc[0, column])
    assert asof.loc[1, column] == pytest.approx(
        (12.0 + 6.0 + 8.0 + 10.0)
        / (
            (12.0 + 6.0 + 8.0 + 10.0)
            + (2.0 + 2.0 + 2.0 + 2.0)
            + (4.0 + 2.0 + 4.0 + 6.0)
            + (2.0 + 2.0 + 2.0 + 2.0)
        )
    )
    assert asof.loc[0, "raw_allocation_available__bezemer_nfb_4q"] == 1
    assert asof.loc[0, "raw_allocation_total__bezemer_nfb_4q"] == 4


def test_destination_oos_purges_unrealized_forward_labels(monkeypatch: pytest.MonkeyPatch) -> None:
    periods = 30
    horizon = 4
    dates = pd.date_range("2010-03-31", periods=periods, freq="QE-DEC")
    outcome = pd.Series(np.linspace(-0.2, 0.3, periods))
    target = ForecastTarget(
        key="spread_widening",
        label="yield change",
        target_type="continuous",
        source_column="spread",
        outcome=outcome,
        ar1_feature=outcome.shift(1),
    )
    frame = pd.DataFrame({"date": dates})
    baseline = pd.DataFrame({"total_credit_growth": np.linspace(0.1, 0.4, periods)})
    model = baseline.assign(q_t=np.linspace(0.2, 0.8, periods))
    training_lengths: list[int] = []

    def fake_fit(x_train, y_train, x_test, *, effect_feature):
        training_lengths.append(len(y_train))
        return float(pd.to_numeric(y_train, errors="coerce").mean()), 0.0

    monkeypatch.setattr("lib.destination_oos._fit_predict_with_effect", fake_fit)
    result = _score_destination_model(
        frame,
        target,
        baseline,
        model,
        effect_feature="q_t",
        horizon=horizon,
        min_training_rows=8,
    )

    assert result["n"] > 0
    assert training_lengths[:2] == [8, 8]
    assert result["required_min_training_cases"] == 8
    assert result["minimum_common_training_cases"] >= 8


def test_destination_oos_requires_complete_training_cases(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    periods = 36
    horizon = 4
    dates = pd.date_range("2010-03-31", periods=periods, freq="QE-DEC")
    outcome = pd.Series(np.linspace(-0.2, 0.3, periods))
    target = ForecastTarget(
        key="spread_widening",
        label="yield change",
        target_type="continuous",
        source_column="spread",
        outcome=outcome,
        ar1_feature=outcome.shift(1),
    )
    frame = pd.DataFrame({"date": dates})
    baseline = pd.DataFrame(
        {"boj_primary_included_stock_growth": np.linspace(0.1, 0.4, periods)}
    )
    model = baseline.assign(
        borrower_composition_coordinate=np.linspace(0.2, 0.8, periods)
    )
    model.loc[:4, "borrower_composition_coordinate"] = np.nan
    training_indices: list[tuple[int, ...]] = []

    def fake_fit(x_train, y_train, x_test, *, effect_feature):
        assert tuple(x_train.index) == tuple(y_train.index)
        training_indices.append(tuple(int(value) for value in x_train.index))
        return float(pd.to_numeric(y_train, errors="coerce").mean()), 0.0

    monkeypatch.setattr("lib.destination_oos._fit_predict_with_effect", fake_fit)
    result = _score_destination_model(
        frame,
        target,
        baseline,
        model,
        effect_feature=ALLOCATION_FEATURE_COLUMN,
        horizon=horizon,
        min_training_rows=8,
    )

    assert result["n"] > 0
    # Five incomplete early candidate rows are removed from both fits. The
    # candidate and baseline calls therefore receive identical common indices.
    assert training_indices[:2] == [tuple(range(5, 13)), tuple(range(5, 13))]
    assert result["minimum_common_training_cases"] >= 8
