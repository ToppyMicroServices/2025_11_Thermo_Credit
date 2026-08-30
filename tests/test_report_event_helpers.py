from __future__ import annotations

from pathlib import Path

import pandas as pd

from lib.report_helpers import _augment_region_frame, filter_dashboard_events, load_dashboard_events


def test_augment_region_frame_converts_string_columns_to_numeric() -> None:
    frame = pd.DataFrame(
        {
            "date": pd.to_datetime(["2025-03-31", "2025-06-30"]),
            "X_C": pd.Series(["1.5", "invalid"], dtype="str"),
        }
    )

    converted, _ = _augment_region_frame(frame, effective_window=2, has_thermo=False)

    assert converted["X_C"].iloc[0] == 1.5
    assert pd.isna(converted["X_C"].iloc[1])
    assert pd.api.types.is_numeric_dtype(converted["X_C"])


def test_load_dashboard_events_reads_registry() -> None:
    events = load_dashboard_events(str(Path("data/report_events.csv")))
    assert events
    dotcom = next(event for event in events if event["key"] == "dotcom")
    assert "all" in dotcom["regions"]


def test_filter_dashboard_events_applies_region_and_window() -> None:
    events = load_dashboard_events(str(Path("data/report_events.csv")))
    filtered = filter_dashboard_events(
        events,
        region_key="eu",
        start_date=pd.Timestamp("2020-01-01"),
        end_date=pd.Timestamp("2020-12-31"),
    )
    keys = {event["key"] for event in filtered}
    assert "pandemic" in keys
    assert "jp_quake" not in keys
    pandemic = next(event for event in filtered if event["key"] == "pandemic")
    assert pandemic["visible_start"] == pd.Timestamp("2020-02-01")
    assert pandemic["visible_end"] == pd.Timestamp("2020-12-31")
