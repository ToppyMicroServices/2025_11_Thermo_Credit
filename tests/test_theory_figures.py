from pathlib import Path

import pandas as pd

from lib.theory_figures import (
    _prepare_metric_series,
    _robust_score_series,
    build_theory_figures,
    load_region_frames,
)


def _write_region_csv(path: Path, start: str, periods: int, *, scale: float) -> None:
    dates = pd.date_range(start, periods=periods, freq="QE")
    frame = pd.DataFrame(
        {
            "date": dates,
            "S_M": [scale * (i + 1) for i in range(periods)],
            "T_L": [0.2 * scale + 0.03 * i for i in range(periods)],
            "X_C": [0.5 * scale + 0.05 * i for i in range(periods)],
            "loop_area": [0.1 * scale + 0.02 * i for i in range(periods)],
        }
    )
    frame.to_csv(path, index=False)


def test_load_region_frames_uses_jp_fallback(tmp_path):
    site_dir = tmp_path / "site"
    site_dir.mkdir()
    _write_region_csv(site_dir / "indicators.csv", "2000-03-31", 6, scale=1.0)
    _write_region_csv(site_dir / "indicators_eu.csv", "2000-03-31", 6, scale=2.0)
    _write_region_csv(site_dir / "indicators_us.csv", "2000-03-31", 6, scale=3.0)

    frames = load_region_frames(site_dir)

    assert [item.key for item in frames] == ["jp", "eu", "us"]
    assert len(frames[0].frame) == 6


def test_build_theory_figures_writes_pdf_and_svg(tmp_path):
    site_dir = tmp_path / "site"
    output_dir = tmp_path / "tex" / "generated"
    data_dir = tmp_path / "data"
    site_dir.mkdir(parents=True)
    output_dir.mkdir(parents=True)
    data_dir.mkdir(parents=True)

    _write_region_csv(site_dir / "indicators.csv", "2000-03-31", 8, scale=1.0)
    _write_region_csv(site_dir / "indicators_eu.csv", "2000-03-31", 8, scale=2.0)
    _write_region_csv(site_dir / "indicators_us.csv", "2000-03-31", 8, scale=3.0)
    (data_dir / "report_events.csv").write_text(
        "\n".join(
            [
                "key,label,start_date,end_date,regions,category,description",
                "dotcom,IT Bubble unwind,2000-03-01,2002-10-31,all,bubble,Example",
                "jp_qqe,Abenomics and QQE launch,2013-04-04,2014-12-31,jp,policy,Example",
            ]
        ),
        encoding="utf-8",
    )

    outputs = build_theory_figures(
        site_dir=site_dir,
        output_dir=output_dir,
        events_path=data_dir / "report_events.csv",
        start_date="2000-01-01",
    )

    expected = {
        output_dir / "theory_sm_tl_panels.pdf",
        output_dir / "theory_sm_tl_panels.svg",
        output_dir / "theory_capacity_panels.pdf",
        output_dir / "theory_capacity_panels.svg",
    }
    assert set(outputs) == expected
    assert all(path.exists() and path.stat().st_size > 0 for path in expected)


def test_robust_score_series_handles_outlier_without_blowing_up():
    series = pd.Series([1.0, 1.2, 1.4, 1.6, 12.0])

    scored = _robust_score_series(series)

    assert scored.notna().all()
    assert float(scored.iloc[-1]) > 2.5
    assert abs(float(scored.iloc[2])) < 0.2


def test_prepare_metric_series_applies_trailing_smoothing_to_xc():
    series = pd.Series([0.0, 0.0, 1.0, 100.0, 2.0])

    prepared = _prepare_metric_series("X_C", series)
    untouched = _prepare_metric_series("S_M", series)

    assert list(prepared.round(2)) == [0.0, 0.0, 0.0, 0.25, 1.0]
    assert list(untouched) == list(series)
