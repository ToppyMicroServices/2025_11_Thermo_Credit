import numpy as np
import pandas as pd

from lib.indicators import build_indicators_core


def _inputs():
    dates = pd.date_range("2023-01-01", periods=3, freq="QE-DEC")
    money = pd.DataFrame({
        "date": dates,
        "M_in": [100.0, 110.0, 120.0],
        "M_out": [90.0, 95.0, 105.0],
    })
    q = pd.DataFrame({
        "date": dates,
        "q_a": [0.5, 0.5, 0.5],
        "q_b": [0.5, 0.5, 0.5],
    })
    cred = pd.DataFrame({
        "date": dates,
        "U": [200.0, 210.0, 230.0],
        "L_real": 1.0,
        "L_asset": 1.0,
        "Y": 1.0,
        "spread": 0.3,
        "depth": 900.0,
        "turnover": 1.0,
    })
    reg = pd.DataFrame({
        "date": dates,
        "p_R": 0.4,
        "V_R": 80.0,
    })
    return money, q, cred, reg


def _cfg():
    return {
        "T0": 2.0,
        "k": 1.0,
        "q_cols": ["q_a", "q_b"],
        "entropy_per_category": False,
        "V_C_formula": "legacy",
        "U_detrend": {"enabled": False},
    }


def test_free_energy_matches_definition():
    money, q, cred, reg = _inputs()
    df = build_indicators_core(money, q, cred, reg, _cfg())

    assert "F_C" in df.columns
    assert "S_M" in df.columns
    assert "U" in df.columns

    expected = pd.to_numeric(df["U"], errors="coerce") - 2.0 * pd.to_numeric(df["S_M"], errors="coerce")
    np.testing.assert_allclose(pd.to_numeric(df["F_C"], errors="coerce"), expected.values)

    # U detrending disabled -> no U_star/U_trend columns
    assert "U_star" not in df.columns
    assert "U_trend" not in df.columns
