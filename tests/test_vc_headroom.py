import numpy as np
import pandas as pd

from lib.indicators import build_indicators_core

Q_COLS = [
    "q_productive",
    "q_housing",
    "q_consumption",
    "q_financial",
    "q_government",
]


def _base_frames():
    dates = pd.date_range("2021-01-01", periods=5, freq="QE-DEC")
    money = pd.DataFrame({
        "date": dates,
        "M_in": np.linspace(100, 140, len(dates)),
        "M_out": np.linspace(80, 120, len(dates)),
    })
    q = pd.DataFrame({"date": dates})
    for col in Q_COLS:
        q[col] = 0.2
    cred = pd.DataFrame({
        "date": dates,
        "spread": np.linspace(0.4, 0.6, len(dates)),
        "depth": np.linspace(900, 950, len(dates)),
        "turnover": np.linspace(1.0, 1.05, len(dates)),
        "U": np.linspace(100, 120, len(dates)),
    })
    reg = pd.DataFrame({
        "date": dates,
        "p_R": np.linspace(0.3, 0.5, len(dates)),
        "V_R": np.linspace(70, 74, len(dates)),
        "capital_headroom": np.linspace(60, 63, len(dates)),
        "lcr_headroom": np.linspace(65, 62, len(dates)),
        "nsfr_headroom": np.linspace(58, 61, len(dates)),
    })
    return money, q, cred, reg


def _cfg(extra=None):
    cfg = {
        "T0": 1.0,
        "q_cols": Q_COLS,
        "U_detrend": {"enabled": False},
        "V_C_formula": "min_headroom",
        "V_C_headroom_cols": ["capital_headroom", "lcr_headroom", "nsfr_headroom"],
    }
    if extra:
        cfg.update(extra)
    return cfg


def test_vc_min_headroom_overrides_legacy_values():
    money, q, cred, reg = _base_frames()
    df = build_indicators_core(money, q, cred, reg, _cfg())
    expected_min = np.minimum.reduce([
        reg["capital_headroom"].values,
        reg["lcr_headroom"].values,
        reg["nsfr_headroom"].values,
    ])
    np.testing.assert_allclose(df["V_C"].values, expected_min)
    np.testing.assert_allclose(df["V_C_legacy"].values, reg["V_R"].values)
    assert (df["V_C_formula_used"] == "min_headroom").any()


def test_vc_legacy_leaves_original_values():
    money, q, cred, reg = _base_frames()
    cfg = _cfg({"V_C_formula": "legacy"})
    df = build_indicators_core(money, q, cred, reg, cfg)
    np.testing.assert_allclose(df["V_C"].values, reg["V_R"].values)
    assert "V_C_formula_used" not in df.columns


def test_vc_min_headroom_handles_missing_columns():
    money, q, cred, reg = _base_frames()
    reg = reg.drop(columns=["lcr_headroom"])  # simulate data gap
    df = build_indicators_core(money, q, cred, reg, _cfg())
    expected = np.minimum(reg["capital_headroom"].values, reg["nsfr_headroom"].values)
    np.testing.assert_allclose(df["V_C"].values, expected)


def test_vc_headroom_derives_from_base_when_columns_absent():
    money, q, cred, reg = _base_frames()
    reg = reg.drop(columns=["capital_headroom", "lcr_headroom", "nsfr_headroom"])
    scales = [1.0, 0.95, 0.90]
    cfg = _cfg({
        "V_C_headroom_cols": ["capital_headroom", "lcr_headroom", "nsfr_headroom"],
        "V_C_headroom_scales": scales,
    })
    df = build_indicators_core(money, q, cred, reg, cfg)
    expected = reg["V_R"].values * min(scales)
    np.testing.assert_allclose(df["V_C"].values, expected)
    assert "V_C_headroom" in df.columns


def test_vc_headroom_leaves_legacy_when_all_nan():
    money, q, cred, reg = _base_frames()
    reg[["capital_headroom", "lcr_headroom", "nsfr_headroom"]] = np.nan
    df = build_indicators_core(money, q, cred, reg, _cfg())
    np.testing.assert_allclose(df["V_C"].values, reg["V_R"].values)
    assert (df["V_C_formula_used"] == "min_headroom").all()
