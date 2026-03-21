from __future__ import annotations

import pandas as pd

from lib.theory_calibration import calibrate_region_frame, render_calibration_tex, render_snapshot_tex


def _synthetic_region_frame() -> pd.DataFrame:
    dates = pd.date_range("2018-03-31", periods=12, freq="QE-DEC")
    frame = pd.DataFrame(
        {
            "date": dates,
            "U": [100, 103, 105, 107, 110, 114, 117, 119, 122, 124, 127, 130],
            "V_C": [40, 41, 42, 43, 44, 44.5, 45, 45.5, 46, 46.5, 47, 47.5],
            "S_M": [20, 20.5, 21, 21.5, 22, 22.4, 22.7, 23, 23.2, 23.5, 23.7, 24],
            "spread": [3.5, 3.4, 3.3, 3.2, 3.0, 2.9, 2.8, 2.7, 2.7, 2.6, 2.5, 2.4],
            "loop_area": [12, 11, 10.5, 10, 9.5, 9.0, 8.8, 8.5, 8.1, 7.8, 7.5, 7.2],
            "T_L": [0.25, 0.28, 0.31, 0.34, 0.39, 0.42, 0.45, 0.49, 0.52, 0.56, 0.59, 0.63],
            "X_C": [15, 18, 19, 22, 25, 28, 30, 33, 35, 37, 39, 42],
        }
    )
    return frame


def test_calibrate_region_frame_returns_finite_objective() -> None:
    result = calibrate_region_frame(_synthetic_region_frame(), "jp")
    assert result.coverage_end == "2020-12-31"
    assert result.objective == result.objective
    assert result.params["T0"] > 0
    assert result.scales["U"] > 0
    assert "growth_corr" in result.diagnostics
    assert "growth_corr" in result.baseline_diagnostics


def test_rendered_tex_mentions_regions_and_objective_inputs() -> None:
    result = calibrate_region_frame(_synthetic_region_frame(), "us")
    calib_tex = render_calibration_tex([result], source_ref="origin/main")
    snap_tex = render_snapshot_tex([result], source_ref="origin/main")
    assert "origin/main" in calib_tex
    assert "US" in calib_tex
    assert "coverage" in snap_tex.lower()
    assert "implicit headroom" in snap_tex.lower()
