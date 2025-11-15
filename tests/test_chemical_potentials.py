import numpy as np
import pandas as pd

from lib.indicators import build_indicators_core


def _inputs():
    dates = pd.date_range("2022-12-31", periods=3, freq="QE-DEC")
    money = pd.DataFrame({
        "date": dates,
        "M_in": [100.0, 120.0, 140.0],
        "M_out": [80.0, 90.0, 95.0],
    })
    q = pd.DataFrame({
        "date": dates,
        "q_risky": [0.3, 0.5, 0.75],
        "q_safe": [0.7, 0.5, 0.25],
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
        "q_cols": ["q_risky", "q_safe"],
        "entropy_per_category": False,
        "V_C_formula": "legacy",
    }


def test_mu_columns_present():
    money, q, cred, reg = _inputs()
    df = build_indicators_core(money, q, cred, reg, _cfg())
    assert "mu_q_risky" in df.columns
    assert "mu_q_safe" in df.columns
    assert df[["mu_q_risky", "mu_q_safe"]].notna().any().all()


def test_mu_ranks_follow_shares():
    money, q, cred, reg = _inputs()
    df = build_indicators_core(money, q, cred, reg, _cfg())
    row = df.iloc[-1]
    assert row["q_risky"] > row["q_safe"]
    assert row["mu_q_risky"] > row["mu_q_safe"]