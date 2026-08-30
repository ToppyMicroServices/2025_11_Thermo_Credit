from __future__ import annotations

import numpy as np
import pandas as pd

from lib.external_coupling import build_external_coupling_indices
from lib.no_lookahead import apply_release_lags, resolve_release_lags
from lib.theory_calibration import calibrate_regions


def test_release_lag_panel_uses_only_available_observations() -> None:
    dates = pd.to_datetime(["2020-03-31", "2020-06-30", "2020-09-30"])
    frame = pd.DataFrame({"date": dates, "U": [100.0, 200.0, 999.0], "spread": [1.0, 2.0, 3.0]})

    out = apply_release_lags(frame, {"U": 90, "spread": 1})

    assert np.isnan(out.loc[0, "U"])
    assert out.loc[1, "U"] == 100.0
    assert out.loc[2, "U"] == 200.0
    assert out.loc[2, "spread"] == 2.0


def test_destination_columns_use_release_lag_group_default() -> None:
    lags, default, _ = resolve_release_lags({})

    assert lags["C_G"] == 90
    assert lags["q_t"] == 90
    assert lags["destination_coverage"] == 90
    assert lags["destination_coverage_observed"] == 90
    assert default == 0


def test_zero_destination_vector_is_not_mixed_with_prior_total() -> None:
    dates = pd.to_datetime(["2020-03-31", "2020-06-30", "2020-09-30"])
    frame = pd.DataFrame(
        {
            "date": dates,
            "C_t": [10.0, 0.0, 8.0],
            "C_G": [4.0, 0.0, 3.0],
            "C_B": [2.0, 0.0, 1.0],
            "C_E": [4.0, 0.0, 4.0],
        }
    )

    out = apply_release_lags(
        frame,
        {"C_t": 90, "C_G": 90, "C_B": 90, "C_E": 90},
    )

    assert out.loc[2, ["C_t", "C_G", "C_B", "C_E"]].tolist() == [0.0, 0.0, 0.0, 0.0]


def test_external_coupling_expanding_zscore_has_no_future_lookahead() -> None:
    dates = pd.date_range("2020-01-01", periods=24, freq="MS")
    values = np.linspace(1.0, 2.0, len(dates))

    def fetcher_base(series_id: str, start: str | None = None) -> pd.DataFrame:
        return pd.DataFrame({"date": dates, "value": values})

    changed_values = values.copy()
    changed_values[18:] = 100.0

    def fetcher_changed(series_id: str, start: str | None = None) -> pd.DataFrame:
        return pd.DataFrame({"date": dates, "value": changed_values})

    cfg = {
        "enabled": True,
        "frequency": "MS",
        "zscore_min_periods": 3,
        "pressure_components": [{"id": "X", "key": "x", "transform": "value"}],
    }
    base = build_external_coupling_indices(cfg, fetcher_base)
    changed = build_external_coupling_indices(cfg, fetcher_changed)

    np.testing.assert_allclose(base["E_p"].iloc[:18], changed["E_p"].iloc[:18], equal_nan=True)


def test_calibration_prefers_realtime_indicator_panel(tmp_path) -> None:
    site = tmp_path / "site"
    site.mkdir()
    dates = pd.date_range("2015-03-31", periods=32, freq="QE-DEC")
    base = pd.DataFrame(
        {
            "date": dates,
            "U": np.linspace(100.0, 180.0, len(dates)),
            "V_C": np.linspace(40.0, 55.0, len(dates)),
            "S_M": np.linspace(20.0, 30.0, len(dates)),
            "spread": np.linspace(3.0, 2.0, len(dates)),
            "loop_area": np.linspace(12.0, 7.0, len(dates)),
            "T_L": np.linspace(0.25, 0.65, len(dates)),
            "X_C": np.linspace(15.0, 42.0, len(dates)),
        }
    )
    dashboard = base.assign(preprocessing_mode="dashboard_retrospective")
    realtime = base.assign(U=base["U"] + 10.0, preprocessing_mode="real_time_release_lagged")
    dashboard.to_csv(site / "indicators.csv", index=False)
    realtime.to_csv(site / "indicators_realtime.csv", index=False)

    results = calibrate_regions(site, panel_mode="realtime")

    assert len(results) == 1
    assert results[0].panel_source == "site/indicators_realtime.csv"
    assert results[0].preprocessing_mode == "real_time_release_lagged"
