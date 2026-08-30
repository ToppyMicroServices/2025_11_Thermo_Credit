import numpy as np
import pandas as pd

from lib.temperature import liquidity_state_index, liquidity_temperature


def _frame(spread, depth, turnover):
    return pd.DataFrame(
        {
            "date": pd.date_range("2020-01-01", periods=len(spread), freq="QE-DEC"),
            "spread": spread,
            "depth": depth,
            "turnover": turnover,
        }
    )


def test_liquidity_index_is_monotone_at_current_observation():
    n_hist = 16
    hist_spread = np.linspace(0.8, 1.2, n_hist)
    hist_depth = np.linspace(900.0, 1100.0, n_hist)
    hist_turnover = np.linspace(0.8, 1.2, n_hist)

    bad = _frame(
        np.r_[hist_spread, 1.8],
        np.r_[hist_depth, 700.0],
        np.r_[hist_turnover, 0.5],
    )
    good = _frame(
        np.r_[hist_spread, 0.4],
        np.r_[hist_depth, 1400.0],
        np.r_[hist_turnover, 1.8],
    )

    bad_t = float(liquidity_state_index(bad)["T_L"].iloc[-1])
    good_t = float(liquidity_state_index(good)["T_L"].iloc[-1])

    assert good_t > bad_t
    assert bad_t < 0.5


def test_liquidity_index_has_no_future_lookahead():
    n = 24
    base = _frame(
        np.linspace(0.8, 1.2, n),
        np.linspace(900.0, 1100.0, n),
        np.linspace(0.8, 1.2, n),
    )
    changed_future = base.copy()
    changed_future.loc[18:, "spread"] = 10.0
    changed_future.loc[18:, "depth"] = 100.0
    changed_future.loc[18:, "turnover"] = 0.1

    base_t = liquidity_temperature(base)["T_L"]
    changed_t = liquidity_temperature(changed_future)["T_L"]

    np.testing.assert_allclose(base_t.iloc[:18], changed_t.iloc[:18], atol=1e-12)


def test_liquidity_index_sorts_dates_before_expanding_scores():
    frame = _frame(
        np.linspace(0.8, 1.2, 20),
        np.linspace(900.0, 1100.0, 20),
        np.linspace(0.8, 1.2, 20),
    )
    shuffled = frame.sample(frac=1.0, random_state=42).reset_index(drop=True)

    expected = liquidity_temperature(frame)
    actual = liquidity_temperature(shuffled)

    pd.testing.assert_series_equal(actual["date"], expected["date"])
    np.testing.assert_allclose(actual["T_L"], expected["T_L"], atol=1e-12)
