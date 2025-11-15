import numpy as np
import pandas as pd

from lib.indicators import build_indicators_core
from lib.config_loader import load_config


MECE_COLS = [
    "q_productive",
    "q_housing",
    "q_consumption",
    "q_financial",
    "q_government",
]


def _sample_inputs():
    dates = pd.date_range("2020-03-31", periods=4, freq="QE-DEC")
    money = pd.DataFrame({
        "date": dates,
        "M_in": np.linspace(100, 160, len(dates)),
        "M_out": np.linspace(90, 150, len(dates)),
    })
    q = pd.DataFrame({"date": dates})
    for col in MECE_COLS:
        q[col] = 1.0 / len(MECE_COLS)
    cred = pd.DataFrame({
        "date": dates,
        "L_real": np.linspace(1000, 1300, len(dates)),
        "L_asset": np.linspace(800, 1100, len(dates)),
        "U": np.linspace(500, 800, len(dates)),
        "Y": np.linspace(400, 700, len(dates)),
        "spread": np.linspace(0.5, 0.8, len(dates)),
        "depth": np.linspace(900, 1200, len(dates)),
        "turnover": np.linspace(1.0, 1.3, len(dates)),
    })
    reg = pd.DataFrame({
        "date": dates,
        "p_R": np.linspace(0.4, 0.7, len(dates)),
        "V_R": np.linspace(80, 110, len(dates)),
        "E_p": np.array([-1.0, -0.5, 0.5, 1.0]),
        "E_T": np.array([0.2, 0.4, 0.6, 0.8]),
    })
    return money, q, cred, reg


def _cfg(alpha: float, delta: float, enabled: bool = True):
    return {
        "T0": 1.0,
        "k": 1.0,
        "q_cols": MECE_COLS,
        "V_C_formula": "legacy",
        "external_coupling": {
            "enabled": enabled,
            "alpha": alpha,
            "delta": delta,
        },
    }


def test_pressure_coupling_adjusts_p_c():
    money, q, cred, reg = _sample_inputs()
    cfg = _cfg(alpha=0.5, delta=0.0)
    df = build_indicators_core(money, q, cred, reg, cfg)
    assert "p_C_baseline" in df.columns
    assert "E_p_contrib" in df.columns
    expected = df["p_C_baseline"] + df["E_p_contrib"].fillna(0.0)
    assert np.allclose(df["p_C"].values, expected.values, equal_nan=True)
    ep = pd.to_numeric(df["E_p"], errors="coerce")
    assert np.allclose(df["E_p_contrib"].values, 0.5 * ep.values, equal_nan=True)


def test_temperature_coupling_uses_delta():
    money, q, cred, reg = _sample_inputs()
    cfg = _cfg(alpha=0.0, delta=0.25)
    df = build_indicators_core(money, q, cred, reg, cfg)
    assert "T_L_baseline" in df.columns
    assert "E_T_contrib" in df.columns
    expected = df["T_L_baseline"] + df["E_T_contrib"].fillna(0.0)
    assert np.allclose(df["T_L"].values, expected.values, equal_nan=True)
    et = pd.to_numeric(df["E_T"], errors="coerce")
    assert np.allclose(df["E_T_contrib"].values, 0.25 * et.values, equal_nan=True)


def test_zero_coefficients_match_disabled_behavior():
    money, q, cred, reg = _sample_inputs()
    df_disabled = build_indicators_core(money, q, cred, reg, _cfg(alpha=0.0, delta=0.0, enabled=False))
    df_zero = build_indicators_core(money, q, cred, reg, _cfg(alpha=0.0, delta=0.0, enabled=True))
    assert np.allclose(df_zero["p_C"].values, df_disabled["p_C"].values, equal_nan=True)
    assert np.allclose(df_zero["T_L"].values, df_disabled["T_L"].values, equal_nan=True)


def test_region_configs_enable_pressure_coupling_eu():
    cfg = load_config("eu")
    money, q, cred, reg = _sample_inputs()
    df = build_indicators_core(money, q, cred, reg, cfg)
    alpha = float(cfg.get("external_coupling", {}).get("alpha", 0.0))
    assert alpha > 0
    ep = pd.to_numeric(df["E_p"], errors="coerce")
    expected = df["p_C_baseline"] + alpha * ep
    assert np.allclose(df["p_C"].values, expected.values, equal_nan=True)


def test_region_configs_enable_pressure_coupling_us():
    cfg = load_config("us")
    money, q, cred, reg = _sample_inputs()
    df = build_indicators_core(money, q, cred, reg, cfg)
    alpha = float(cfg.get("external_coupling", {}).get("alpha", 0.0))
    assert alpha > 0
    ep = pd.to_numeric(df["E_p"], errors="coerce")
    expected = df["p_C_baseline"] + alpha * ep
    assert np.allclose(df["p_C"].values, expected.values, equal_nan=True)
