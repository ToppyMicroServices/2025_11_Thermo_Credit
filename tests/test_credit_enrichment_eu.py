import os
import pandas as pd

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data")

REQUIRED_COLUMNS = ["L_asset", "depth", "turnover", "L_real", "U", "Y", "spread"]


def test_eu_credit_enrichment_columns_present():
    path = os.path.join(DATA_DIR, "credit_eu.csv")
    assert os.path.exists(path), "credit_eu.csv should exist after EU feature build"
    df = pd.read_csv(path)
    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    assert not missing, f"Missing EU enrichment columns: {missing}"
    # Basic sanity: columns are non-empty
    assert df.shape[0] > 0, "EU credit dataframe should have rows"
    for c in ["L_asset", "depth", "turnover"]:
        assert df[c].notna().any(), f"Column {c} should contain non-null values"
