from __future__ import annotations

import numpy as np
import pandas as pd

from lib.lambda_sensitivity import (
    lambda_destination_panel,
    render_lambda_sensitivity_tex,
    sensitivity_rows_for_region,
    summarize_lambda_sensitivity,
)
from lib.theory_figures import RegionFrame


def _synthetic_destination_frame(periods: int = 40) -> pd.DataFrame:
    dates = pd.date_range("2015-03-31", periods=periods, freq="QE-DEC")
    c_t = np.full(periods, 10.0)
    c_g = np.full(periods, 3.0)
    c_b = np.full(periods, 2.0)
    c_e = np.full(periods, 4.0)
    y = 100.0 * np.exp(np.linspace(0.0, 0.2, periods))
    l_asset = 60.0 * np.exp(np.linspace(0.0, 0.35, periods))
    return pd.DataFrame(
        {
            "date": dates,
            "C_t": c_t,
            "C_G": c_g,
            "C_B": c_b,
            "C_E": c_e,
            "Y": y,
            "U": y,
            "L_asset": l_asset,
            "destination_coverage": 0.9,
            "preprocessing_mode": "real_time_release_lagged",
        }
    )


def test_lambda_destination_panel_recomputes_weighted_split() -> None:
    frame = _synthetic_destination_frame(4)

    out = lambda_destination_panel(frame, 0.75)

    assert np.isclose(out.loc[0, "C_R"], 3.0 + 0.75 * 2.0)
    assert np.isclose(out.loc[0, "C_A"], 4.0 + 0.25 * 2.0)
    assert np.isclose(out.loc[0, "q_t"], (3.0 + 0.75 * 2.0) / 10.0)


def test_lambda_sensitivity_emits_all_grid_targets() -> None:
    region = RegionFrame(key="jp", label="Japan (JP)", frame=_synthetic_destination_frame(), source_path="site/indicators_realtime.csv")

    metrics, panel = sensitivity_rows_for_region(region, (0.0, 0.5, 1.0), horizon=4)
    summary = summarize_lambda_sensitivity(metrics)
    tex = render_lambda_sensitivity_tex(summary)

    assert set(metrics["lambda_B"]) == {0.0, 0.5, 1.0}
    assert set(metrics["target"]) == {"asset_acceleration", "real_growth"}
    assert panel["lambda_B"].nunique() == 3
    assert "lambda_B" in tex
