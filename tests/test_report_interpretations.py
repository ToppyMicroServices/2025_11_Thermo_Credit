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
    assert "historical positions are model-specific" in text
    assert "No policy or overheating threshold is validated." in text


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
    assert "Compare historical position, not raw levels." in text


def test_compare_interpretation_keeps_raw_level_caveat_without_data():
    mod = _load_report_module()

    text = mod._chart_interpretation("Compare: T_L", None)

    assert "standardized within each region" in text
    assert "Raw levels cannot be compared across regions." in text


def test_compare_normalized_entropy_caption_uses_plotted_quantity():
    mod = _load_report_module()
    frame = pd.DataFrame({
        "date": pd.to_datetime(["2023-01-01", "2023-01-01", "2023-02-01", "2023-02-01"]),
        "value": [0.5, 0.99, 0.97, 0.6],
        "Region": ["JP", "US", "JP", "US"],
    })
    original = frame.copy(deep=True)

    text = mod._chart_interpretation("Compare: S_M_hat", frame)

    assert text == "Normalized entropy is highest in JP (S_M_hat≈0.97) vs US (S_M_hat≈0.60)."
    pd.testing.assert_frame_equal(frame, original)


def test_compare_standardized_caption_is_unchanged():
    mod = _load_report_module()
    frame = pd.DataFrame({"value": [0.97, 0.6], "Region": ["JP", "US"]})

    text = mod._chart_interpretation("Compare: T_L (standardized)", frame)

    assert text == (
        "Liquidity state: the standardized position is highest in JP "
        "(z≈0.97) vs US (z≈0.60). Compare historical position, not raw levels."
    )


def test_compare_normalized_entropy_fallbacks_do_not_claim_standardization():
    mod = _load_report_module()
    frames = [
        None,
        pd.DataFrame(),
        pd.DataFrame({"value": [None], "Region": ["JP"]}),
        pd.DataFrame({"value": [0.97], "Region": ["JP"]}),
        pd.DataFrame({"value": [0.97, 0.97], "Region": ["JP", "US"]}),
    ]

    for frame in frames:
        text = mod._chart_interpretation("Compare: S_M_hat", frame)
        assert text == "S_M_hat shows normalized entropy, not within-region z-scores."


def test_xc_interpretation_keeps_experimental_scope():
    mod = _load_report_module()

    text = mod._chart_interpretation("X_C Diagnostic", pd.DataFrame({"X_C": [0.25]}))
    missing = mod._chart_interpretation("X_C Diagnostic", pd.DataFrame({"X_C": [None]}))

    assert "X_C≈0.25" in text
    assert "experimental diagnostic, not validated headroom" in text
    assert "exergy-like transformation is experimental" in missing
    assert "No threshold has been validated." in missing


def test_firstlaw_interpretation_keeps_value_trend_and_model_scope():
    mod = _load_report_module()

    text = mod._chart_interpretation(
        "First-law Decomposition", pd.DataFrame({"firstlaw_resid": [0.1, 0.2, 0.3]})
    )

    assert "Residual≈0.300 and rising" in text
    assert "in the first-law-like model" in text


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
    assert "Credit scale times allocation dispersion." in html


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


def test_chart_style_separates_title_and_wrapped_legend():
    mod = _load_report_module()
    fig = go.Figure(data=go.Scatter(x=[1, 2], y=[3, 4]), layout={"title": "Example"})

    mod._style_figure(fig)

    assert fig.layout.margin.t == 112
    assert fig.layout.title.x == 0.0
    assert fig.layout.title.xanchor == "left"
    assert fig.layout.title.yanchor == "top"
    assert fig.layout.legend.title.text == ""


def test_mobile_tabs_do_not_cover_charts():
    css = (Path(__file__).parent.parent / "assets" / "report.css").read_text(encoding="utf-8")
    mobile_css = css.split("@media (max-width: 760px)", 1)[1].split(
        "@media (prefers-reduced-motion: reduce)", 1
    )[0]

    assert ".tabs {" in mobile_css
    assert "position: static;" in mobile_css
    assert ".hero-button:last-child" in mobile_css


def test_missing_optional_diagnostics_do_not_reuse_previous_chart():
    mod = _load_report_module()
    dates = pd.date_range("2023-01-01", periods=8, freq="QE-DEC")
    frame = pd.DataFrame({
        "date": dates,
        "S_M": range(8),
        "T_L": [0.5 + value / 100 for value in range(8)],
    })

    context = mod._build_region_context("test", "Test", frame, diag_window=24)
    labels = [spec[1] for spec in context["fig_specs"]]

    assert labels == ["S_M & T_L"]


def test_dashboard_entrypoints_have_identical_content(tmp_path, monkeypatch):
    mod = _load_report_module()
    monkeypatch.setattr(mod, "SITE_DIR", str(tmp_path))

    mod._write_dashboard_entrypoints("<html>dashboard</html>")

    assert (tmp_path / "index.html").read_bytes() == (tmp_path / "report.html").read_bytes()


def test_page_header_explains_scope_and_links_public_outputs():
    mod = _load_report_module()

    html = mod._build_page_header("body{}", "")

    assert "Credit scale is only half the story." in html
    assert "borrower composition" in html
    assert "broader proxies" in html
    assert 'href="#dashboard-tabs"' in html
    assert "releases/latest/download/theory.pdf" in html
    assert "10.5281/zenodo.17563220" in html


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
