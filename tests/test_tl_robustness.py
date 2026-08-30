from __future__ import annotations

import numpy as np
import pandas as pd

from lib.temperature import liquidity_state_index
from lib.tl_robustness import (
    VARIANT_COLUMNS,
    compute_tl_variants,
    legacy_signed_product_violation_count,
    run_tl_monotonicity_checks,
)


def _frame() -> pd.DataFrame:
    n = 28
    return pd.DataFrame(
        {
            "date": pd.date_range("2015-03-31", periods=n, freq="QE-DEC"),
            "spread": np.r_[np.linspace(1.0, 1.4, n - 1), 0.7],
            "depth": np.r_[np.linspace(900.0, 1_100.0, n - 1), 1_350.0],
            "turnover": np.r_[np.linspace(0.8, 1.2, n - 1), 1.7],
        }
    )


def test_tl_variants_include_requested_specifications_and_match_current_tl() -> None:
    frame = _frame()
    variants = compute_tl_variants(frame)
    current = liquidity_state_index(frame)["T_L"]

    for column in VARIANT_COLUMNS.values():
        assert column in variants.columns
        assert variants[column].notna().any()
    np.testing.assert_allclose(variants["TL_additive_zscore"], current, atol=1e-12)


def test_requested_tl_variants_are_monotone_but_signed_product_is_rejected() -> None:
    checks = run_tl_monotonicity_checks()

    for variant in VARIANT_COLUMNS:
        assert checks[variant]["monotone_pass"] is True
        assert checks[variant]["all_good_minus_all_bad"] > 0
    assert legacy_signed_product_violation_count() > 0
    assert checks["legacy_signed_product"]["monotone_pass"] is False
