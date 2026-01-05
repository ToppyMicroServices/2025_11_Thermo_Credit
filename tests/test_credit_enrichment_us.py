import os

import pandas as pd

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data")
REQUIRED_COLUMNS = ["L_asset", "depth", "turnover", "L_real", "U", "Y", "spread"]


def test_us_credit_enrichment_columns_present():
    path = os.path.join(DATA_DIR, "credit_us.csv")
    assert os.path.exists(path), "credit_us.csv should exist after US feature build"
    df = pd.read_csv(path)
    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    assert not missing, f"Missing US enrichment columns: {missing}"
    assert df.shape[0] > 0, "US credit dataframe should have rows"
    for c in ["L_asset", "depth", "turnover"]:
        assert df[c].notna().any(), f"Column {c} should contain non-null values"
