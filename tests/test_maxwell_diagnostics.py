import numpy as np
import pandas as pd

from lib.indicators import compute_diagnostics


def test_maxwell_curl_is_zero_for_known_quadratic_potential():
    t = np.arange(60, dtype=float)
    s = 10.0 + 0.35 * t + np.sin(t / 4.0)
    v = 5.0 + 0.22 * t + np.cos(t / 5.0)
    a, b, c = 0.40, -0.70, 0.30

    # U(S,V)=aS^2+bSV+cV^2 implies T=dU/dS and p=-dU/dV.
    u = a * s**2 + b * s * v + c * v**2
    tl = 2.0 * a * s + b * v
    pc = -(b * s + 2.0 * c * v)

    df = pd.DataFrame(
        {
            "date": pd.date_range("2000-01-01", periods=len(t), freq="QE-DEC"),
            "S_M": s,
            "V_C": v,
            "T_L": tl,
            "p_C": pc,
            "U": u,
        }
    )

    out = compute_diagnostics(df, window=12)
    valid = out[["dT_dV_at_S", "dp_dS_at_V", "maxwell_curl"]].dropna()

    assert len(valid) == len(df) - 11
    np.testing.assert_allclose(valid["dT_dV_at_S"], b, atol=1e-10)
    np.testing.assert_allclose(valid["dp_dS_at_V"], -b, atol=1e-10)
    np.testing.assert_allclose(valid["maxwell_curl"], 0.0, atol=1e-10)
    np.testing.assert_allclose(out["maxwell_gap"].dropna(), valid["maxwell_curl"], atol=1e-12)
