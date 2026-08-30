from __future__ import annotations

import numpy as np
import pandas as pd

from lib.baseline_forecast import (
    build_base_features,
    build_forecast_targets,
    evaluate_region_fe_panel,
    evaluate_region_forecasts,
    render_baseline_forecast_tex,
    summarize_baseline_forecasts,
)
from lib.forecast_frames import RegionFrame


def _synthetic_frame(periods: int = 56) -> pd.DataFrame:
    dates = pd.date_range("2010-03-31", periods=periods, freq="QE-DEC")
    t = np.arange(periods, dtype=float)
    y = 100.0 * np.exp(0.01 * t + 0.002 * np.sin(t / 3.0))
    credit = 90.0 * np.exp(0.012 * t)
    spread = 2.0 + 0.2 * np.sin(t / 4.0)
    return pd.DataFrame(
        {
            "date": dates,
            "Y": y,
            "U": y,
            "L_real": credit,
            "L_asset": 50.0 * np.exp(0.014 * t + 0.003 * np.cos(t / 2.0)),
            "spread": spread,
            "M_in": 80.0 * np.exp(0.009 * t),
            "q_t": 0.35 + 0.03 * np.sin(t / 5.0),
            "S_M_hat": 0.9 + 0.01 * np.sin(t / 6.0),
            "T_L": 0.5 + 0.1 * np.sin(t / 4.0),
            "X_C": 1.0 + 0.2 * np.cos(t / 7.0),
            "loop_area": np.cumsum(np.sin(t / 8.0)),
            "C_R": 3.0 + 0.1 * np.sin(t / 5.0),
            "C_A": 2.0 + 0.2 * np.cos(t / 5.0),
            "destination_coverage": 0.8,
            "p_C": spread,
            "V_C": 40.0 - 0.1 * spread,
        }
    )


def test_build_forecast_targets_records_unavailable_inflation() -> None:
    targets, coverage = build_forecast_targets(_synthetic_frame())

    assert "real_growth" in {t.key for t in targets}
    assert "spread_widening" in {t.key for t in targets}
    inflation = coverage[coverage["target"] == "inflation"].iloc[0]
    assert inflation["status"] == "unavailable"


def test_credit_to_gdp_gap_does_not_fallback_to_activity_or_credit() -> None:
    frame = _synthetic_frame().drop(columns=["Y"])
    frame["U_gdp_only"] = np.nan

    features = build_base_features(frame)

    assert features["credit_to_gdp_gap"].isna().all()


def test_evaluate_region_forecasts_outputs_full_comparison() -> None:
    region = RegionFrame(key="jp", label="Japan (JP)", frame=_synthetic_frame(), source_path="site/indicators_realtime.csv")

    results, coverage = evaluate_region_forecasts(region)
    summary = summarize_baseline_forecasts(results)
    tex = render_baseline_forecast_tex(summary, coverage)

    assert not results.empty
    assert {"baseline_only", "full_thermo_credit"}.issubset(set(results["candidate"]))
    assert "total_credit_growth" in set(results["baseline"])
    assert "Auxiliary proxy-panel" in tex


def test_region_fixed_effect_panel_baseline_is_reported() -> None:
    regions = [
        RegionFrame(key="jp", label="Japan (JP)", frame=_synthetic_frame(), source_path="jp.csv"),
        RegionFrame(key="us", label="United States (US)", frame=_synthetic_frame(), source_path="us.csv"),
    ]

    results = evaluate_region_fe_panel(regions)

    assert "region_fixed_effect_panel" in set(results["baseline"])
    assert "full_thermo_credit" in set(results["candidate"])
    assert results["model_features"].str.contains("fe_jp").any()
