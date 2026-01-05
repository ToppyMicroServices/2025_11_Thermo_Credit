import numpy as np
import pandas as pd

from lib.indicators import build_indicators_core


def _toy_cfg(overrides=None):
    cfg = {
        "T0": 1.0,
        "q_cols": [
            "q_productive",
            "q_housing",
            "q_consumption",
            "q_financial",
            "q_government",
        ],
        "U_detrend": {"enabled": True, "method": "rolling", "window": 3, "min_periods": 2},
        "V_C_formula": "legacy",
    }
    if overrides:
        cfg.update(overrides)
    return cfg


def test_u_star_uses_only_past_data():
    dates = pd.date_range("2022-01-01", periods=6, freq="QE-DEC")
    money = pd.DataFrame({
        "date": dates,
        "M_in": np.linspace(100, 150, len(dates)),
        "M_out": np.linspace(80, 120, len(dates)),
    })
    q = pd.DataFrame({
        "date": dates,
    })
    for col in ["q_productive", "q_housing", "q_consumption", "q_financial", "q_government"]:
        q[col] = 0.2
    cred = pd.DataFrame({
        "date": dates,
        "spread": np.linspace(0.5, 0.6, len(dates)),
        "depth": np.linspace(1000, 1050, len(dates)),
        "turnover": np.linspace(1.0, 1.1, len(dates)),
        "U": [100, 102, 104, 110, 115, 118],
    })
    reg = pd.DataFrame({
        "date": dates,
        "p_R": np.linspace(0.4, 0.5, len(dates)),
        "V_R": np.linspace(80, 90, len(dates)),
        "capital_headroom": np.linspace(70, 75, len(dates)),
        "lcr_headroom": np.linspace(69, 74, len(dates)),
        "nsfr_headroom": np.linspace(68, 73, len(dates)),
    })

    df = build_indicators_core(money, q, cred, reg, _toy_cfg())

    assert "U_star" in df.columns
    assert "U_trend" in df.columns
    assert df["U_star"].isna().sum() >= 1  # early rows needing history

    # With window=3 rolling mean, index 3 uses U[1:4]
    expected_trend_idx3 = np.mean([102, 104, 110])
    assert np.isclose(df.loc[3, "U_trend"], expected_trend_idx3, atol=1e-9)
    assert np.isclose(df.loc[3, "U_star"], 110 - expected_trend_idx3, atol=1e-9)

    # Ensure no lookahead: rolling mean at index 2 cannot use values beyond index 2
    expected_trend_idx2 = np.mean([100, 102, 104])
    assert np.isclose(df.loc[2, "U_trend"], expected_trend_idx2, atol=1e-9)
