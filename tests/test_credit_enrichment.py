import pandas as pd
import os

def test_credit_enrichment_columns_present():
    path = os.path.join("site", "indicators.csv")
    assert os.path.exists(path), "indicators.csv missing; run compute script first"
    df = pd.read_csv(path)
    required = ["L_asset", "depth", "turnover", "L_asset_toy", "depth_toy", "turnover_toy"]
    missing = [c for c in required if c not in df.columns]
    assert not missing, f"Missing enrichment columns: {missing}"
