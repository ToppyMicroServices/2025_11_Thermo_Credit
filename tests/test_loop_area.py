import numpy as np
import pandas as pd

from lib.loop_area import (
    absolute_path_area,
    closed_loop_signed_area,
    event_window_loop_diagnostics,
    loop_area_null_distribution,
    loop_area_p_value,
    open_path_integral,
    rolling_loop_diagnostics,
)


def test_closed_loop_area_separates_signed_and_absolute_metrics():
    v = [0.0, 1.0, 1.0, 0.0]
    p = [0.0, 0.0, 1.0, 1.0]

    assert np.isclose(closed_loop_signed_area(p, v), 1.0)
    assert np.isclose(abs(closed_loop_signed_area(list(reversed(p)), list(reversed(v)))), 1.0)
    assert absolute_path_area(p, v) >= abs(open_path_integral(p, v))


def test_rolling_loop_diagnostics_marks_only_complete_windows():
    p = [0.0, 0.0, 1.0, 1.0, 0.0]
    v = [0.0, 1.0, 1.0, 0.0, 0.0]
    out = rolling_loop_diagnostics(p, v, window=4)

    assert np.isnan(out["loop_closed_area"][2])
    assert np.isclose(out["loop_closed_area"][3], 1.0)
    assert np.isfinite(out["loop_abs_area"][4])


def test_loop_null_distribution_and_p_value_are_deterministic():
    p = np.sin(np.linspace(0, 2 * np.pi, 16, endpoint=False))
    v = np.cos(np.linspace(0, 2 * np.pi, 16, endpoint=False))
    null_a = loop_area_null_distribution(p, v, samples=8, method="phase", seed=7)
    null_b = loop_area_null_distribution(p, v, samples=8, method="phase", seed=7)

    np.testing.assert_allclose(null_a, null_b)
    p_value = loop_area_p_value(closed_loop_signed_area(p, v), null_a)
    assert 0.0 <= p_value <= 1.0


def test_loop_null_distribution_supports_ar_surrogates_and_block_shuffle():
    p = np.sin(np.linspace(0, 4 * np.pi, 24, endpoint=False))
    v = np.cos(np.linspace(0, 4 * np.pi, 24, endpoint=False))

    ar_null = loop_area_null_distribution(p, v, samples=6, method="ar1", seed=11)
    shuffle_null = loop_area_null_distribution(p, v, samples=6, method="block_shuffle", block_size=4, seed=11)

    assert ar_null.shape == (6,)
    assert shuffle_null.shape == (6,)
    assert np.isfinite(ar_null).all()
    assert np.isfinite(shuffle_null).all()


def test_event_window_loop_diagnostics_uses_explicit_cycles():
    dates = pd.date_range("2020-03-31", periods=5, freq="QE-DEC")
    frame = pd.DataFrame({
        "date": dates,
        "p_C": [0.0, 0.0, 1.0, 1.0, 0.0],
        "V_C": [0.0, 1.0, 1.0, 0.0, 0.0],
    })
    windows = [{
        "key": "cycle_a",
        "label": "Cycle A",
        "start": "2020-03-31",
        "end": "2020-12-31",
        "regions": ["jp"],
    }]

    result = event_window_loop_diagnostics(frame, windows, region_key="jp", null_samples=4, seed=3)

    assert result.loc[0, "cycle_key"] == "cycle_a"
    assert result.loc[0, "n_obs"] == 4
    assert np.isclose(result.loc[0, "loop_closed_area"], 1.0)
    assert 0.0 <= result.loc[0, "loop_phase_p"] <= 1.0
