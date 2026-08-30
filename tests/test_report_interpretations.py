import importlib.util
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go


def _load_report_module():
    script_path = Path(__file__).parent.parent / "scripts" / "03_make_report.py"
    spec = importlib.util.spec_from_file_location("report_for_tests", str(script_path))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)  # type: ignore[attr-defined]
    return module


def test_chart_interpretation_sm_tl_includes_both_axes():
    mod = _load_report_module()
    dates = pd.date_range("2023-01-01", periods=8, freq="ME")
    df = pd.DataFrame({
        "date": dates,
        "S_M": [0.4, 0.45, 0.5, 0.65, 0.7, 0.75, 0.78, 0.8],
        "T_L": [0.8, 0.85, 0.9, 1.0, 1.05, 1.1, 1.2, 1.25],
    })

    text = mod._chart_interpretation("S_M & T_L", df)

    assert text is not None
    assert "S_M≈" in text
    assert "T_L≈" in text


def test_compare_chart_interpretation_highlights_regions():
    mod = _load_report_module()
    df = pd.DataFrame({
        "date": pd.to_datetime(["2023-01-01", "2023-01-01", "2023-02-01", "2023-02-01"]),
        "value": [0.5, 0.7, 0.9, 0.4],
        "Region": ["JP", "US", "JP", "US"],
    })

    text = mod._chart_interpretation("Compare: S_M", df)

    assert text is not None
    assert "highest in" in text
    assert "vs" in text


def test_raw_inputs_interpretation_is_static_message():
    mod = _load_report_module()

    text = mod._chart_interpretation("Raw Inputs (first=100)", None)

    assert "rebased to 100" in text


def test_region_summary_items_render_as_cards():
    mod = _load_report_module()

    html = mod._summary_cards_html(["Latest date: 2025-03-31", "S_M: 12.3"])

    assert 'class="summary-grid"' in html
    assert 'class="summary-card tone-neutral"' in html
    assert "Most recent observation" in html


def test_coverage_summary_flags_stale_region():
    mod = _load_report_module()
    contexts = [
        {"label": "Japan (JP)", "last_date": pd.Timestamp("2025-03-31")},
        {"label": "Euro Area (EU)", "last_date": pd.Timestamp("2020-03-31")},
    ]

    html = mod._build_coverage_summary(contexts)

    assert "Data window" in html
    assert "tone-current" in html
    assert "tone-stale" in html


def test_figures_render_with_readable_height():
    mod = _load_report_module()
    fig = go.Figure(data=go.Scatter(x=[1, 2], y=[3, 4]))

    html = mod._figs_html([(fig, "Example chart", "Example", "Short note")])

    assert 'class="chart-grid"' in html
    assert "height:420px" in html
    assert '"displaylogo": false' in html


def test_registered_event_bands_add_window_and_optional_label():
    mod = _load_report_module()
    fig = go.Figure(data=go.Scatter(x=pd.date_range("2020-01-01", periods=4), y=[1, 2, 3, 4]))
    events = [{
        "key": "pandemic",
        "label": "COVID-19 pandemic shock",
        "category": "pandemic",
        "visible_start": pd.Timestamp("2020-02-01"),
        "visible_end": pd.Timestamp("2020-03-31"),
    }]

    mod._add_registered_event_bands(fig, events, show_labels=True)

    assert len(fig.layout.shapes) == 1
    assert fig.layout.annotations[0].text == "COVID-19"
