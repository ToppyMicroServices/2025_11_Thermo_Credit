import pandas as pd

from lib.ecb_data import normalize_ecb_csv


def test_normalize_ecb_csv_handles_monthly_periods():
    payload = "TIME_PERIOD,OBS_VALUE\n2024-11,150.5\n2024-12,151.0\n"

    df = normalize_ecb_csv(payload)

    assert df["date"].dt.strftime("%Y-%m-%d").tolist() == ["2024-11-01", "2024-12-01"]
    assert df["value"].tolist() == [150.5, 151.0]


def test_normalize_ecb_csv_handles_quarterly_periods():
    payload = "TIME_PERIOD,OBS_VALUE\n2024-Q3,123.0\n2024-Q4,125.0\n"

    df = normalize_ecb_csv(payload)

    assert df["date"].dt.strftime("%Y-%m-%d").tolist() == ["2024-09-30", "2024-12-31"]
    assert df["value"].tolist() == [123.0, 125.0]
