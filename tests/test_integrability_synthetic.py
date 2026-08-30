from __future__ import annotations

import numpy as np

from lib.integrability_synthetic import (
    NONINTEGRABLE_OMEGA,
    evaluate_integrability_synthetic_case,
    render_integrability_synthetic_tex,
    run_integrability_synthetic_test,
    summarize_integrability_synthetic,
)


def test_clean_integrable_potential_returns_zero_curl() -> None:
    row = evaluate_integrability_synthetic_case(
        scenario="integrable_potential",
        noise_level=0.0,
        sampling_step=2,
        window=16,
        seed=3,
    )

    assert row["n_valid"] > 0
    assert row["mean_abs_omega"] < 1e-10
    assert row["decision"] == "passes_zero_curl"


def test_clean_nonintegrable_case_recovers_known_vorticity() -> None:
    row = evaluate_integrability_synthetic_case(
        scenario="nonintegrable_vorticity",
        noise_level=0.0,
        sampling_step=1,
        window=16,
        seed=5,
    )

    assert np.isclose(row["mean_omega"], NONINTEGRABLE_OMEGA, atol=1e-10)
    assert row["decision"] == "detects_nonintegrable"


def test_proxy_misspecification_flags_consistency_warning() -> None:
    row = evaluate_integrability_synthetic_case(
        scenario="proxy_misspecified",
        noise_level=0.0,
        sampling_step=1,
        window=16,
        seed=7,
    )

    assert row["proxy_misspecified"] is True
    assert row["mean_abs_omega"] > 0.05
    assert row["decision"] == "proxy_warning"


def test_synthetic_grid_summary_and_tex_are_renderable() -> None:
    results = run_integrability_synthetic_test(
        scenarios=("integrable_potential", "nonintegrable_vorticity"),
        noise_levels=(0.0, 0.05),
        sampling_steps=(1, 4),
        window=16,
        seed=11,
    )
    summary = summarize_integrability_synthetic(results)
    tex = render_integrability_synthetic_tex(results)

    assert len(results) == 8
    assert summary["overall"]["clean_integrable_zero_curl_pass"] is True
    assert summary["overall"]["clean_nonintegrable_detection_pass"] is True
    assert "Synthetic integrability" in tex
