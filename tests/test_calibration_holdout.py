from __future__ import annotations

import numpy as np
import pandas as pd

from lib.calibration_holdout import evaluate_calibration_holdout_region, render_calibration_holdout_tex
from lib.theory_figures import RegionFrame


def _synthetic_holdout_frame(periods: int = 92) -> pd.DataFrame:
    dates = pd.date_range("2000-03-31", periods=periods, freq="QE-DEC")
    t = np.arange(periods, dtype=float)
    stress = np.sin(t / 6.0)
    return pd.DataFrame(
        {
            "date": dates,
            "U": 100.0 + 1.5 * t + 2.0 * np.sin(t / 8.0),
            "Y": 100.0 + 1.5 * t + 2.0 * np.sin(t / 8.0),
            "V_C": 50.0 + 0.4 * t - stress,
            "S_M": 20.0 + 0.3 * t + stress,
            "spread": 2.0 + 0.15 * stress + 0.02 * np.cos(t / 3.0),
            "T_L": 0.7 - 0.05 * stress,
            "X_C": 1.0 - stress + 0.1 * np.cos(t / 5.0),
        }
    )


def test_calibration_holdout_reports_fixed_and_rolling_rows() -> None:
    region = RegionFrame(key="jp", label="Japan (JP)", frame=_synthetic_holdout_frame(), source_path="site/indicators_realtime.csv")

    results = evaluate_calibration_holdout_region(region)

    assert {"fixed_2000_2015", "rolling_10y"}.issubset(set(results["strategy"]))
    assert "spread_widening" in set(results["target"])
    assert {"tuned_rmse", "raw_rmse", "simple_rmse"}.issubset(results.columns)


def test_calibration_holdout_tex_mentions_train_only_theta() -> None:
    region = RegionFrame(key="us", label="United States (US)", frame=_synthetic_holdout_frame(), source_path="site/indicators_us_realtime.csv")
    results = evaluate_calibration_holdout_region(region)

    tex = render_calibration_holdout_tex(results)

    assert "Diagnostic holdout test" in tex
    assert "theta=(T0,p0,U0,V0,S0)" in tex
