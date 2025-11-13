import os
import pandas as pd

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data")
REQUIRED_COLUMNS = ["L_asset", "depth", "turnover", "L_real", "U", "Y", "spread"]
TOY_COLUMNS = ["L_asset_toy", "depth_toy", "turnover_toy"]

def test_jp_credit_enrichment_columns_present():
    path = os.path.join(DATA_DIR, "credit.csv")
    assert os.path.exists(path), "credit.csv should exist after JP feature build"
    df = pd.read_csv(path)
    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    assert not missing, f"Missing JP enrichment columns: {missing}"
    toy_missing = [c for c in TOY_COLUMNS if c not in df.columns]
    assert not toy_missing, f"Missing JP toy baseline columns: {toy_missing}"
    assert df.shape[0] > 0, "JP credit dataframe should have rows"
    for c in ["L_asset", "depth", "turnover"]:
        assert df[c].notna().any(), f"Column {c} should contain non-null values"
