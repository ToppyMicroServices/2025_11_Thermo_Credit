from __future__ import annotations

from pathlib import Path

import pandas as pd

from lib.report_helpers import filter_dashboard_events, load_dashboard_events


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
