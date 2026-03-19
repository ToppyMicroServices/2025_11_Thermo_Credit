import importlib.util
from pathlib import Path

import pandas as pd

from lib.report_helpers import (
    apply_event_overlays,
    build_dashboard_takeaway_sections,
    build_event_summary_html,
    filter_dashboard_events,
    load_dashboard_events,
    render_dashboard_events_tex,
    render_dashboard_takeaways_tex,
    write_dashboard_takeaways_png,
)


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


def test_build_dashboard_takeaway_sections_deduplicates_lines():
    sections = build_dashboard_takeaway_sections(
        [
            {
                "key": "jp",
                "label": "Japan (JP)",
                "last_date": pd.Timestamp("2025-03-31"),
                "takeaway_lines": ["Latest snapshot", "Latest snapshot", "  ", "Loop area is non-zero."],
            }
        ]
    )

    assert sections == [
        {
            "key": "jp",
            "label": "Japan (JP)",
            "latest_date": "2025-03-31",
            "bullets": ["Latest snapshot", "Loop area is non-zero."],
        }
    ]


def test_render_dashboard_takeaways_tex_escapes_special_chars():
    text = render_dashboard_takeaways_tex(
        [
            {
                "label": "Japan (JP)",
                "latest_date": "2025-03-31",
                "bullets": ["S_M & T_L: q_t=0.45, stress_proxy=0.10_1"],
            }
        ],
        source_path="site/report.html",
        report_month="2025-03",
    )

    assert "\\texttt{site/report.html}" in text
    assert "Japan (JP) (latest: 2025-03-31)" in text
    assert "S\\_M \\& T\\_L" in text
    assert "stress\\_proxy=0.10\\_1" in text


def test_write_dashboard_takeaways_png_writes_file(tmp_path):
    out = tmp_path / "dashboard_takeaways.png"

    ok = write_dashboard_takeaways_png(
        str(out),
        [
            {
                "label": "Japan (JP)",
                "latest_date": "2025-03-31",
                "bullets": ["Latest snapshot (2025-03-31): S_M=0.8, T_L=1.2, loop area=0.3, X_C=0.1."],
            }
        ],
        subtitle="Auto-generated snapshot for 2025-03",
    )

    assert ok is True
    assert out.exists()
    assert out.stat().st_size > 0


def test_build_coverage_summary_marks_stale_regions():
    mod = _load_report_module()

    html = mod._build_coverage_summary(
        [
            {"label": "Japan (JP)", "last_date": pd.Timestamp("2025-03-31")},
            {"label": "Euro Area (EU)", "last_date": pd.Timestamp("2020-03-31")},
        ]
    )

    assert "Coverage Summary" in html
    assert "Japan (JP)" in html
    assert "Euro Area (EU)" in html
    assert "Current" in html
    assert "Stale" in html


def test_load_and_filter_dashboard_events(tmp_path):
    path = tmp_path / "events.csv"
    path.write_text(
        "\n".join(
            [
                "key,label,start_date,end_date,regions,category,description",
                "dotcom,IT Bubble unwind,2000-03-01,2002-10-31,all,bubble,Example",
                "euro,Euro sovereign debt crisis,2010-04-01,2012-07-31,eu,crisis,Example",
            ]
        ),
        encoding="utf-8",
    )

    events = load_dashboard_events(str(path))
    visible = filter_dashboard_events(
        events,
        region_key="eu",
        start_date=pd.Timestamp("2000-01-01"),
        end_date=pd.Timestamp("2011-01-01"),
    )

    assert [event["key"] for event in visible] == ["dotcom", "euro"]
    assert visible[0]["visible_start"].strftime("%Y-%m-%d") == "2000-03-01"
    assert visible[0]["visible_end"].strftime("%Y-%m-%d") == "2002-10-31"
    assert visible[1]["visible_start"].strftime("%Y-%m-%d") == "2010-04-01"
    assert visible[1]["visible_end"].strftime("%Y-%m-%d") == "2011-01-01"


def test_render_dashboard_events_tex_and_html():
    events = [
        {
            "key": "lehman",
            "label": "Lehman / Global Financial Crisis",
            "start_date": pd.Timestamp("2008-09-15"),
            "end_date": pd.Timestamp("2009-06-30"),
            "regions": ["all"],
            "category": "crisis",
            "description": "Global funding stress after the Lehman Brothers collapse.",
        }
    ]

    tex = render_dashboard_events_tex(events)
    html = build_event_summary_html(events, plot_start=pd.Timestamp("1998-01-01"))

    assert "Lehman / Global Financial Crisis" in tex
    assert "2008-09-15 to 2009-06-30" in tex
    assert "Reference events" in html
    assert "1998-01-01" in html


def test_apply_event_overlays_adds_shapes_and_annotations():
    mod = _load_report_module()
    fig = mod.px.line(
        pd.DataFrame(
            {
                "date": pd.to_datetime(["2020-01-01", "2020-02-01"]),
                "value": [1.0, 2.0],
            }
        ),
        x="date",
        y="value",
    )

    apply_event_overlays(
        fig,
        [
            {
                "label": "COVID-19 pandemic shock",
                "visible_start": pd.Timestamp("2020-02-01"),
                "visible_end": pd.Timestamp("2020-03-31"),
                "category": "pandemic",
            }
        ],
    )

    assert len(fig.layout.shapes) == 1
    assert len(fig.layout.annotations) == 1
