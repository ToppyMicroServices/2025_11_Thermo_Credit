import os
import pandas as pd
import pytest


@pytest.mark.parametrize("path", ["site/indicators.csv", "site/indicators_eu.csv"]) 
def test_core_columns_if_present(path):
    if not os.path.exists(path):
        pytest.skip(f"{path} not present")
    df = pd.read_csv(path)
    core = {"date", "S_M", "T_L", "loop_area", "X_C"}
    assert core.issubset(set(df.columns)), f"{path} missing required columns"


def test_jp_eu_shared_subset_if_both():
    jp_path = "site/indicators.csv"
    eu_path = "site/indicators_eu.csv"
    if not (os.path.exists(jp_path) and os.path.exists(eu_path)):
        pytest.skip("JP/EU indicators not both present")
    jp = pd.read_csv(jp_path)
    eu = pd.read_csv(eu_path)
    common = set(jp.columns).intersection(set(eu.columns))
    required = {"date", "S_M", "T_L", "loop_area", "X_C"}
    assert required.issubset(common), "Shared required subset missing across JP/EU"
