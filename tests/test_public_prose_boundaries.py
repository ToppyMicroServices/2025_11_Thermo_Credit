import importlib.util
from pathlib import Path

import pandas as pd


def test_diagnostic_outliers_are_not_described_as_crises():
    path = Path(__file__).resolve().parents[1] / "scripts" / "03_make_report.py"
    spec = importlib.util.spec_from_file_location("report_prose_boundaries", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    frame = pd.DataFrame({
        "date": pd.date_range("2020-03-31", periods=12, freq="QE-DEC"),
        "S_M": range(12),
        "T_L": [0.5] * 12,
        "p_C": [1.0] * 12,
        "V_C": range(100, 112),
        "U": range(200, 212),
        "dT_dV_at_S": [0.1] * 12,
        "dp_dS_at_V": [0.1] * 12,
        "maxwell_curl": [0.2] * 11 + [10.0],
    })

    context = module._build_region_context("test", "Test", frame, diag_window=3)

    assert "Model diagnostic flags (not crisis classifications)" in context["html"]
    assert "Out-of-spec / crisis / proxy invalid zone" not in context["html"]
