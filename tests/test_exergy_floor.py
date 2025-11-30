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


def test_free_energy_baseline_min_shifts_upward():
    money, q, cred, reg = _inputs()
    raw_df = build_indicators_core(
        money,
        q,
        cred,
        reg,
        _base_cfg({
            "exergy_floor_zero": False,
            "F_C_baseline_mode": "none",
        }),
    )
    raw_fc = pd.to_numeric(raw_df["F_C"], errors="coerce")
    raw_xc = pd.to_numeric(raw_df["X_C"], errors="coerce")

    shifted_cfg = _base_cfg({
        "exergy_floor_zero": False,  # isolate the baseline shift from clipping
        "F_C_baseline_mode": "min",
        "F_C_baseline_eps": 0.1,
    })
    shifted_df = build_indicators_core(money, q, cred, reg, shifted_cfg)
    fc_shift = pd.to_numeric(shifted_df["F_C"], errors="coerce")
    xc_shift = pd.to_numeric(shifted_df["X_C"], errors="coerce")

    offset = max(0.0, -float(raw_fc.min()) + 0.1)
    np.testing.assert_allclose(fc_shift.values, raw_fc.values + offset)
    np.testing.assert_allclose(xc_shift.values, raw_xc.values + offset)
    assert (fc_shift > 0).all()
    assert (xc_shift > 0).all()
    assert shifted_df["F_C_baseline_mode"].iloc[0] == "min"
    assert np.isclose(float(shifted_df["F_C_baseline_offset"].iloc[0]), offset)
