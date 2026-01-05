import json
import os

import pandas as pd


def test_raw_inputs_normalization(tmp_path):
    # Prepare a temp data dir
    d_data = tmp_path / "data"
    d_data.mkdir()

    # Minimal sources.json with two enabled series
    sources = [
        {"id": "SER_A", "title": "A", "enabled": True},
        {"id": "SER_B", "title": "B", "enabled": True},
    ]
    (d_data / "sources.json").write_text(json.dumps(sources), encoding="utf-8")

    # Create raw CSVs
    pd.DataFrame({
        "date": ["2020-01-01", "2020-02-01"],
        "value": [50.0, 75.0],
    }).to_csv(d_data / "SER_A.csv", index=False)
    pd.DataFrame({
        "date": ["2020-01-01", "2020-02-01"],
        "value": [200.0, 180.0],
    }).to_csv(d_data / "SER_B.csv", index=False)

    # Import helper and run
    import sys
    sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
    from lib.raw_inputs import enabled_sources, load_and_normalize, load_sources

    srcs = load_sources(str(d_data / "sources.json"))
    df = load_and_normalize(enabled_sources(srcs))

    assert df is not None
    # First row of each series should be 100
    row0 = df.iloc[0]
    assert abs(float(row0["SER_A"]) - 100.0) < 1e-6
    assert abs(float(row0["SER_B"]) - 100.0) < 1e-6

from pathlib import Path


def write_csv(path: Path, rows):
    path.write_text("date,value\n" + "\n".join(f"{d},{v}" for d, v in rows), encoding="utf-8")


def test_sources_normalization(tmp_path, monkeypatch):
    # Arrange: create a temporary data directory structure
    data_dir = tmp_path / "data"
    site_dir = tmp_path / "site"
    data_dir.mkdir()
    site_dir.mkdir()

    # Two mock series with different starting levels
    write_csv(data_dir / "AAA.csv", [("2024-01-01", 200), ("2024-02-01", 250)])
    write_csv(data_dir / "BBB.csv", [("2024-01-01", 5), ("2024-02-01", 10)])

    sources = [
        {"id": "AAA", "enabled": True},
        {"id": "BBB", "enabled": True},
        {"id": "CCC", "enabled": False},  # disabled should be ignored
    ]
    (data_dir / "sources.json").write_text(json.dumps(sources), encoding="utf-8")

    # Copy minimal indicators.csv needed by report script
    indicators = pd.DataFrame({
        "date": pd.to_datetime(["2024-01-01", "2024-02-01"]),
        "S_M": [1, 2],
        "T_L": [1.0, 1.5],
        "loop_area": [0.0, 0.1],
        "X_C": [1.0, 1.1],
        "p_C": [1, 1.1],
        "V_C": [1, 1.1],
        "U": [1, 1.2],
    })
    indicators.to_csv(site_dir / "indicators.csv", index=False)

    # Monkeypatch CWD so script reads our temp data
    monkeypatch.chdir(tmp_path)

    # Act: import the script module (executes report generation)
    import importlib.util
    spec = importlib.util.spec_from_file_location("report", str(Path(__file__).parent.parent / "scripts" / "03_make_report.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)  # type: ignore

    # Assert: raw_inputs figure data normalization
    # The module variable raw_inputs_df should exist
    raw_df = getattr(mod, "raw_inputs_df", None)
    assert raw_df is not None, "raw_inputs_df should be created when enabled sources exist"
    # First row should be 100 for both AAA and BBB
    first = raw_df.iloc[0]
    assert first["AAA"] == 100.0
    assert first["BBB"] == 100.0
    # Second row ratios: AAA 250/200*100=125, BBB 10/5*100=200
    second = raw_df.iloc[1]
    assert abs(second["AAA"] - 125.0) < 1e-9
    assert abs(second["BBB"] - 200.0) < 1e-9

