from __future__ import annotations

import numpy as np
import pandas as pd

from lib.loop_null_tests import (
    NULL_METHOD_LABELS,
    evaluate_loop_null_tests_region,
    render_loop_area_null_tests_tex,
    summarize_loop_area_null_tests,
)


def _frame(n: int = 28) -> pd.DataFrame:
    theta = np.linspace(0, 3 * np.pi, n)
    return pd.DataFrame(
        {
            "date": pd.date_range("2018-03-31", periods=n, freq="QE-DEC"),
            "p_C": np.sin(theta) + np.linspace(0.0, 0.2, n),
            "V_C": np.cos(theta) + np.linspace(0.0, 0.3, n),
        }
    )


def test_loop_null_tests_emit_requested_methods_and_segmentation_windows() -> None:
    frame = _frame()
    events = [
        {
            "key": "stress_a",
            "label": "Stress A",
            "category": "test",
            "start": pd.Timestamp("2019-03-31"),
            "end": pd.Timestamp("2020-03-31"),
        }
    ]

    results = evaluate_loop_null_tests_region(
        frame,
        region_key="jp",
        region_label="Japan (JP)",
        panel_source="site/test.csv",
        events=events,
        latest_windows=(8, 12),
        null_samples=12,
        seed=5,
    )

    assert set(NULL_METHOD_LABELS).issubset(set(results["null_method"]))
    assert {8, 12}.issubset(set(results["segmentation_window"]))
    assert "registered_event" in set(results["window_family"])
    percentiles = pd.to_numeric(results["actual_null_percentile"], errors="coerce").dropna()
    assert ((percentiles >= 0.0) & (percentiles <= 1.0)).all()


def test_loop_null_summary_and_tex_render_latest_table() -> None:
    results = evaluate_loop_null_tests_region(
        _frame(),
        region_key="us",
        region_label="United States (US)",
        panel_source="site/test.csv",
        latest_windows=(8, 12, 16),
        null_samples=8,
        seed=7,
    )

    summary = summarize_loop_area_null_tests(results)
    tex = render_loop_area_null_tests_tex(results)

    assert summary["regions"]["us"]["latest_rows"] == 15
    assert "Loop-area null tests" in tex
    assert "phase randomization" in tex
