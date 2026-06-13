import importlib.util
from datetime import datetime
from pathlib import Path


def _load_report_module():
    script_path = Path(__file__).parent.parent / "scripts" / "03_make_report.py"
    spec = importlib.util.spec_from_file_location("report_freshness_for_tests", str(script_path))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)  # type: ignore[attr-defined]
    return module


def test_freshness_summary_flags_stale_region():
    mod = _load_report_module()

    html = mod._build_freshness_summary([
        {"label": "Japan (JP)", "last_date": datetime(2025, 3, 31), "summary_items": []},
        {"label": "Euro Area (EU)", "last_date": datetime(2020, 3, 31), "summary_items": []},
        {"label": "United States (US)", "last_date": datetime(2025, 3, 31), "summary_items": []},
    ])

    assert "Data freshness" in html
    assert "Check date:" in html
    assert "2020-03-31" in html
    assert "Data gap" in html
    assert "Latest available" in html
