from __future__ import annotations

import json
import importlib.util
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parent.parent
SPEC = importlib.util.spec_from_file_location("reproducibility_check", ROOT / "scripts" / "08_reproducibility_check.py")
assert SPEC and SPEC.loader
repro = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(repro)


def test_compare_csv_accepts_numeric_tolerance(tmp_path):
    left = tmp_path / "left.csv"
    right = tmp_path / "right.csv"
    pd.DataFrame({"date": ["2025-03-31"], "value": [1.0]}).to_csv(left, index=False)
    pd.DataFrame({"date": ["2025-03-31"], "value": [1.0 + 1e-10]}).to_csv(right, index=False)

    result = repro.compare_csv(left, right, rtol=1e-8, atol=1e-8)

    assert result["pass"]


def test_compare_json_reports_large_numeric_difference(tmp_path):
    left = tmp_path / "left.json"
    right = tmp_path / "right.json"
    left.write_text(json.dumps({"x": 1.0}), encoding="utf-8")
    right.write_text(json.dumps({"x": 1.1}), encoding="utf-8")

    result = repro.compare_json(left, right, rtol=1e-9, atol=1e-9)

    assert not result["pass"]
    assert result["failure_count"] == 1
