import os
import importlib.util
from pathlib import Path
import pandas as pd

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data")
REQUIRED_COLUMNS = ["L_asset", "depth", "turnover", "L_real", "U", "Y", "spread"]


def _load_indicator_script():
    path = Path(__file__).resolve().parents[1] / "scripts" / "02_compute_indicators.py"
    spec = importlib.util.spec_from_file_location("compute_indicators_for_test", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_us_credit_enrichment_columns_present():
    path = os.path.join(DATA_DIR, "credit_us.csv")
    assert os.path.exists(path), "credit_us.csv should exist after US feature build"
    df = pd.read_csv(path)
    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    assert not missing, f"Missing US enrichment columns: {missing}"
    assert df.shape[0] > 0, "US credit dataframe should have rows"
    for c in ["L_asset", "depth", "turnover"]:
        assert df[c].notna().any(), f"Column {c} should contain non-null values"


def test_us_gdp_like_levels_are_harmonized_to_credit_scale():
    module = _load_indicator_script()
    frame = pd.DataFrame(
        {
            "date": pd.to_datetime(["2024-03-31", "2024-06-30"]),
            "L_real": [40_000.0, 41_000.0],
            "Y": [29_184_890_000_000.0, None],
            "U": [29_184_890_000_000.0, None],
        }
    )

    out = module._ensure_credit_inputs(frame, pd.DataFrame())

    assert out["Y"].max() < 50_000
    assert out["U"].max() < 50_000
