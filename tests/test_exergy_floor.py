import numpy as np
import pandas as pd

from lib.indicators import build_indicators_core


Q_COLS = ["q_a", "q_b"]


def _inputs():
    dates = pd.date_range("2023-01-01", periods=3, freq="QE-DEC")
    money = pd.DataFrame({
        "date": dates,
        "M_in": np.linspace(100, 120, len(dates)),
        "M_out": np.linspace(90, 110, len(dates)),
    })
    q = pd.DataFrame({
        "date": dates,
        "q_a": [0.6, 0.5, 0.4],
        "q_b": [0.4, 0.5, 0.6],
    })
    cred = pd.DataFrame({
        "date": dates,
        "U": [10.0, 5.0, 12.0],
        "L_real": 1.0,
        "L_asset": 1.0,
        "Y": 1.0,
        "spread": 1.0,
        "depth": 1.0,
        "turnover": 1.0,
    })
    reg = pd.DataFrame({
        "date": dates,
        "p_R": np.linspace(0.2, 0.4, len(dates)),
        "V_R": np.linspace(50, 60, len(dates)),
    })
    return money, q, cred, reg


def _base_cfg(extra=None):
    cfg = {
        "T0": 1.0,
        "k": 1.0,
        "q_cols": Q_COLS,
        "entropy_per_category": False,
        "V_C_formula": "legacy",
    }
    if extra:
        cfg.update(extra)
    return cfg


def test_exergy_floor_clips_negative_values():
    money, q, cred, reg = _inputs()
    cfg = _base_cfg({
        "exergy_floor_zero": True,
        "exergy_floor_mode": "clip",
    })
    df = build_indicators_core(money, q, cred, reg, cfg)
    fc = pd.to_numeric(df["F_C"], errors="coerce")
    xc = pd.to_numeric(df["X_C"], errors="coerce")
    assert np.allclose(xc.values, np.clip(fc.values, 0, None), equal_nan=True)
    assert (xc >= 0).all()


def test_exergy_floor_can_be_disabled():
    money, q, cred, reg = _inputs()
    cfg = _base_cfg({
        "exergy_floor_zero": False,  # leave negatives untouched
    })
    df = build_indicators_core(money, q, cred, reg, cfg)
    fc = pd.to_numeric(df["F_C"], errors="coerce")
    xc = pd.to_numeric(df["X_C"], errors="coerce")
    assert np.allclose(xc.values, fc.values, equal_nan=True)
    assert (xc < 0).any(), "Expect at least one negative exergy when floor disabled"
