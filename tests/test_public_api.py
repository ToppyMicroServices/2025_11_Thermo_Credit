import json
from pathlib import Path

import pandas as pd

from lib.public_api import build_public_api


def _indicator_frame() -> pd.DataFrame:
    dates = pd.date_range("2019-03-31", periods=16, freq="QE-DEC")
    return pd.DataFrame(
        {
            "date": dates,
            "C_t": range(100, 116),
            "q_t": [0.6 - 0.01 * index for index in range(16)],
            "one_minus_q_t": [0.4 + 0.01 * index for index in range(16)],
            "S_M": [0.2 + 0.02 * index for index in range(16)],
            "T_L": [0.8 + 0.01 * index for index in range(16)],
            "p_C": [0.1 + 0.01 * index for index in range(16)],
            "X_C": [1.0 - 0.03 * index for index in range(16)],
            "loop_area": [0.02 * index for index in range(16)],
        }
    )


def test_build_public_api_labels_evidence_and_event_limits(tmp_path: Path) -> None:
    site = tmp_path / "site"
    data = tmp_path / "data"
    site.mkdir()
    data.mkdir()
    frame = _indicator_frame()
    frame.to_csv(site / "indicators_jp.csv", index=False)
    frame.to_csv(site / "indicators_eu.csv", index=False)
    frame.to_csv(site / "indicators_us.csv", index=False)
    (data / "data_dictionary.csv").write_text(
        "variable,symbol,evidence_class,status,region,frequency,units,repo_source,construction,claim_limit\n"
        "primary_q,q_t,derived_measurement,current,JP,Quarterly,share,site/indicators_jp.csv,test,borrower composition only\n",
        encoding="utf-8",
    )
    (data / "report_events.csv").write_text(
        "key,label,start_date,end_date,regions,category,description\n"
        "pandemic,COVID-19,2020-02-01,2021-03-31,all,pandemic,Registered test window.\n",
        encoding="utf-8",
    )

    outputs = build_public_api(tmp_path)

    assert site / "api" / "v1" / "manifest.json" in outputs
    jp = json.loads((site / "api" / "v1" / "regions" / "jp" / "latest.json").read_text())
    eu = json.loads((site / "api" / "v1" / "regions" / "eu" / "latest.json").read_text())
    cases = json.loads((site / "api" / "v1" / "case-studies.json").read_text())
    assert jp["allocation_evidence_class"] == "derived_measurement"
    assert "loan purpose" in jp["claim_limit"]
    assert eu["allocation_evidence_class"] == "proxy"
    assert {row["region"] for row in cases["case_studies"]} == {"jp", "eu", "us"}
    assert all(row["interpretation_status"] == "descriptive" for row in cases["case_studies"])
    assert all("does not identify causality" in row["claim_limit"] for row in cases["case_studies"])
    assert "Treat JP q_t as borrower composition" in (site / "llms.txt").read_text()
