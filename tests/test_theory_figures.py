from pathlib import Path

import numpy as np
import pandas as pd

from lib.theory_figures import build_theory_figures


def test_build_theory_figures_includes_jp_destination_targets(tmp_path: Path) -> None:
    site = tmp_path / "site"
    data = tmp_path / "data"
    out = tmp_path / "tex" / "generated"
    site.mkdir()
    data.mkdir()
    dates = pd.date_range("2010-03-31", periods=40, freq="QE-DEC")
    t = np.arange(len(dates), dtype=float)
    q_t = 0.58 - 0.06 * np.sin(t / 5.0)
    c_t = 100.0 + t
    frame = pd.DataFrame(
        {
            "date": dates,
            "q_t": q_t,
            "one_minus_q_t": 1.0 - q_t,
            "C_t": c_t,
            "C_G": q_t * c_t,
            "C_B": np.zeros_like(c_t),
            "C_A": (1.0 - q_t) * c_t,
            "L_asset": 90.0 * np.exp(0.01 * t + 0.03 * (1.0 - q_t)),
            "spread": 0.8 + 0.2 * (1.0 - q_t),
            "S_M": 0.5 + 0.02 * t,
            "T_L": 0.7 + 0.01 * t,
            "X_C": 2.0 - 0.01 * t,
            "loop_area": np.sin(t / 4.0),
        }
    )
    frame.to_csv(site / "indicators_realtime.csv", index=False)
    frame.to_csv(site / "indicators.csv", index=False)
    (data / "report_events.csv").write_text("key,region,start,end,label,category\n", encoding="utf-8")

    outputs = build_theory_figures(site_dir=site, output_dir=out, events_path=data / "report_events.csv")

    assert out / "theory_jp_destination_targets.pdf" in outputs
    assert out / "theory_jp_destination_targets.svg" in outputs
    assert out / "theory_boj_bridge_state.pdf" in outputs
    assert out / "theory_boj_bridge_state.svg" in outputs
