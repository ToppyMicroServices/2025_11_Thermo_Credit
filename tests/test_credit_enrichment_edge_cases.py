import pandas as pd
from lib.credit_enrichment import compute_enrichment


def test_compute_enrichment_empty_df():
    df = pd.DataFrame(columns=["date", "L_real", "U"])  # empty
    warnings = []
    out = compute_enrichment(df, warnings=warnings)
    # Columns should exist even if empty
    assert "depth" in out.columns and "turnover" in out.columns
    assert len(out) == 0
    # No clipping warning for empty frame
    assert not warnings, f"Unexpected warnings for empty frame: {warnings}"


def test_compute_enrichment_partial_nans():
    dates = pd.date_range("2021-01-01", periods=8, freq="QE-DEC")
    # Introduce NaNs in alternating positions
    l_real = pd.Series([100.0, None, 120.0, 0.0, None, 150.0, 0.0, 200.0])
    u_vals = pd.Series([50.0, 40.0, None, 10.0, 5.0, None, 0.0, 3000.0])
    df = pd.DataFrame({"date": dates, "L_real": l_real, "U": u_vals})
    warnings = []
    out = compute_enrichment(df, warnings=warnings)
    # Turnover should never be inf or negative
    assert (out["turnover"].replace([float("inf"), float("-inf")], pd.NA).notna()).all(), "Inf values present"
    assert (out["turnover"] >= 0).all(), "Negative turnover produced"
    # Depth fallback should fill NaNs
    assert out["depth"].isna().sum() == 0, "Depth contains NaNs after fallback"


def test_compute_enrichment_all_zero_l_real():
    dates = pd.date_range("2022-01-01", periods=6, freq="QE-DEC")
    df = pd.DataFrame({"date": dates, "L_real": [0.0]*6, "U": [10.0, 0.0, 5.0, 1.0, 2.0, 100.0]})
    warnings = []
    out = compute_enrichment(df, warnings=warnings)
    # Turnover computation should fallback for all-zero denominator and then clip within bounds
    assert (out["turnover"] >= 0.1).all() and (out["turnover"] <= 10.0).all(), "Turnover not within clipping bounds"
    # Depth should fallback (since median of zeros is 0 -> guarded by or 1.0 path, still yields numeric)
    assert (out["depth"] > 0).all(), "Depth fallback failed for zero L_real"
    # Clipping fraction may exceed threshold; warning optional. Ensure no crash and at most one warning message.
    assert len(warnings) <= 1, f"Unexpected multiple warnings: {warnings}"


def test_compute_enrichment_all_nan_sources():
    # Depth and turnover sources entirely NaN should trigger heuristic fallbacks
    dates = pd.date_range("2023-01-01", periods=5, freq="QE-DEC")
    base = pd.DataFrame({
        "date": dates,
        "L_real": [100.0, 120.0, 140.0, 160.0, 180.0],
        "U": [50.0, 60.0, 70.0, 80.0, 90.0],
    })
    depth_src = pd.Series([float('nan')] * 5, index=dates)
    turn_src = pd.Series([float('nan')] * 5, index=dates)
    warnings = []
    out = compute_enrichment(base, depth_source=depth_src, turnover_source=turn_src, warnings=warnings)
    # Both depth & turnover should be heuristic (non-NaN, within clipping bounds)
    assert out['depth'].isna().sum() == 0, "Depth fallback failed with all-NaN source"
    assert out['turnover'].isna().sum() == 0, "Turnover fallback failed with all-NaN source"
    assert (out['turnover'] >= 0.1).all() and (out['turnover'] <= 10.0).all(), "Turnover outside clipping bounds"
