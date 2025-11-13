import pandas as pd
from lib.credit_enrichment import compute_enrichment


def test_turnover_clipping_warning(capsys):
    # Construct synthetic data causing >15% clipping
    # L_real constant, U spans very small and very large values to trigger both low/high clipping
    dates = pd.date_range("2020-01-01", periods=40, freq="QE-DEC")
    l_real = pd.Series([100.0] * len(dates))
    # First 10 extremely small (ratio < 0.1), next 10 very large (ratio > 10), rest moderate
    u_values = ([1.0] * 10) + ([2000.0] * 10) + ([500.0] * 20)
    df = pd.DataFrame({"date": dates, "L_real": l_real, "L_asset": l_real * 0.4, "U": u_values, "Y": u_values, "spread": 0.0})
    warnings = []
    out = compute_enrichment(df, warnings=warnings)
    # Capture printed warnings (none printed here unless caller prints) so we only assert list content.
    assert any("Turnover clipping" in w for w in warnings), "Expected turnover clipping warning when >15% rows clipped"
    clipped_min_count = (out["turnover"] == 0.1).sum()
    clipped_max_count = (out["turnover"] == 10.0).sum()
    assert clipped_min_count >= 1 and clipped_max_count >= 1, "Should clip both low and high extremes"
