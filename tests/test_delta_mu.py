import numpy as np
import pandas as pd

from lib.indicators import build_indicators_core


def _inputs():
    dates = pd.date_range("2022-12-31", periods=4, freq="QE-DEC")
    money = pd.DataFrame({
        "date": dates,
        "M_in": np.linspace(100.0, 130.0, len(dates)),
        "M_out": np.linspace(80.0, 100.0, len(dates)),
    })
    q = pd.DataFrame({
        "date": dates,
        "q_a": [0.7, 0.6, 0.4, 0.25],
        "q_b": [0.3, 0.4, 0.6, 0.75],
    })
    cred = pd.DataFrame({
        "date": dates,
        "L_real": np.linspace(900, 1100, len(dates)),
        "L_asset": np.linspace(800, 1000, len(dates)),
        "U": np.linspace(600, 650, len(dates)),
        "Y": np.linspace(500, 550, len(dates)),
        "spread": np.linspace(0.4, 0.5, len(dates)),
        "depth": np.linspace(950, 1000, len(dates)),
        "turnover": np.linspace(1.0, 1.1, len(dates)),
    })
    reg = pd.DataFrame({
        "date": dates,
        "p_R": np.linspace(0.3, 0.4, len(dates)),
        "V_R": np.linspace(70, 90, len(dates)),
    })
    return money, q, cred, reg


def _cfg():
    return {
        "T0": 1.0,
        "k": 1.0,
        "q_cols": ["q_a", "q_b"],
        "entropy_per_category": False,
        "V_C_formula": "legacy",
    }


def test_delta_mu_columns_present_and_centered():
    money, q, cred, reg = _inputs()
    df = build_indicators_core(money, q, cred, reg, _cfg())

    # Basic column presence
    assert "mu_q_a" in df.columns
    assert "mu_q_b" in df.columns
    assert "dmu_q_a" in df.columns
    assert "dmu_q_b" in df.columns
    assert "mu_mean" in df.columns

    # Row-wise centering: Δμ_a + Δμ_b = 0 (within numerical tolerance)
    centered = (df["dmu_q_a"] + df["dmu_q_b"]).dropna()
    assert not centered.empty
    assert np.allclose(centered.values, 0.0, atol=1e-10)


def test_delta_mu_sign_matches_mu_deviation():
    money, q, cred, reg = _inputs()
    df = build_indicators_core(money, q, cred, reg, _cfg())

    # Focus on the last row where shares are most imbalanced
    row = df.iloc[-1]
    assert row["q_b"] > row["q_a"]

    # Higher μ should correspond to positive Δμ relative to the mean
    mu_a = row["mu_q_a"]
    mu_b = row["mu_q_b"]
    dmu_a = row["dmu_q_a"]
    dmu_b = row["dmu_q_b"]

    # They should deviate from the mean in opposite directions
    assert dmu_a * dmu_b <= 0

    if mu_b > mu_a:
        assert dmu_b > dmu_a
    elif mu_a > mu_b:
        assert dmu_a > dmu_b
