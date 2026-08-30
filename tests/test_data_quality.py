from datetime import datetime, timezone

import pandas as pd

from lib.data_quality import REGION_FILES, validate_site_data


def _write_panel(path, latest="2025-03-31", rows=8):
    dates = pd.date_range(end=latest, periods=rows, freq="QE-DEC")
    pd.DataFrame(
        {
            "date": dates,
            "S_M": range(rows),
            "T_L": range(rows),
            "p_C": range(rows),
            "X_C": range(rows),
            "loop_area": range(rows),
            "preprocessing_mode": "dashboard_retrospective",
        }
    ).to_csv(path, index=False)


def test_validate_site_data_accepts_complete_panels(tmp_path):
    for filename in REGION_FILES.values():
        _write_panel(tmp_path / filename)

    manifest = validate_site_data(
        tmp_path,
        reference_time=datetime(2026, 1, 1, tzinfo=timezone.utc),
        min_rows=8,
        max_age_days=400,
    )

    assert manifest["valid"] is True
    assert all(region["sha256"] for region in manifest["regions"])


def test_validate_site_data_rejects_header_only_and_stale_panels(tmp_path):
    for region, filename in REGION_FILES.items():
        if region == "eu":
            pd.DataFrame(columns=["date", "S_M", "T_L", "p_C", "X_C", "loop_area", "preprocessing_mode"]).to_csv(
                tmp_path / filename, index=False
            )
        else:
            _write_panel(tmp_path / filename, latest="2020-03-31")

    manifest = validate_site_data(
        tmp_path,
        reference_time=datetime(2026, 1, 1, tzinfo=timezone.utc),
        min_rows=8,
        max_age_days=550,
    )

    assert manifest["valid"] is False
    eu = next(region for region in manifest["regions"] if region["region"] == "eu")
    assert any("row count" in error for error in eu["errors"])


def test_validate_site_data_can_limit_regions(tmp_path):
    _write_panel(tmp_path / "indicators.csv")

    manifest = validate_site_data(
        tmp_path,
        reference_time=datetime(2026, 1, 1, tzinfo=timezone.utc),
        min_rows=8,
        regions=("jp",),
    )

    assert manifest["valid"] is True
    assert [region["region"] for region in manifest["regions"]] == ["jp"]
