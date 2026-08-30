from pathlib import Path

import pandas as pd
from PIL import Image

from lib.dashboard_takeaways import build_dashboard_takeaways


def test_dashboard_takeaways_write_citation_ready_formats(tmp_path: Path) -> None:
    site = tmp_path / "site"
    data = tmp_path / "data"
    output = tmp_path / "tex" / "generated"
    site.mkdir()
    data.mkdir()
    dates = pd.date_range("2015-03-31", periods=20, freq="QE-DEC")
    frame = pd.DataFrame(
        {
            "date": dates,
            "q_t": [0.6 - 0.005 * index for index in range(20)],
            "X_C": [1.0 - 0.04 * index for index in range(20)],
        }
    )
    frame.to_csv(site / "indicators_jp.csv", index=False)
    frame.to_csv(site / "indicators_eu.csv", index=False)
    frame.to_csv(site / "indicators_us.csv", index=False)
    (data / "report_events.csv").write_text(
        "key,label,start_date,end_date,regions,category,description\n"
        "pandemic,COVID-19,2020-02-01,2021-03-31,all,pandemic,test\n",
        encoding="utf-8",
    )

    outputs = build_dashboard_takeaways(site_dir=site, output_dir=output, events_path=data / "report_events.csv")

    assert output / "dashboard_takeaways.png" in outputs
    assert output / "dashboard_takeaways.pdf" in outputs
    assert output / "dashboard_takeaways.svg" in outputs
    with Image.open(output / "dashboard_takeaways.png") as image:
        assert image.width > 2000
        assert image.height > 900
    snippet = (output / "dashboard_takeaways.tex").read_text(encoding="utf-8")
    assert "generated/dashboard_takeaways.pdf" in snippet
    assert "fig:dashboard_takeaways" in snippet
