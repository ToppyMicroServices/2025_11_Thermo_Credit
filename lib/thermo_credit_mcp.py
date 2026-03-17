"""Minimal MCP server wrapper for Thermo Credit."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Mapping

from lib.thermo_credit_tools import (
    build_repo_compute_payload,
    compare_regimes as compare_regimes_core,
    compute_thermo_credit_metrics as compute_metrics_core,
    evaluate_scenario as evaluate_scenario_core,
    get_theory_overview as get_theory_overview_core,
    get_variable_definitions as get_variable_definitions_core,
)


ROOT = Path(__file__).resolve().parents[1]
EXAMPLES_PATH = ROOT / "examples" / "thermo_credit_mcp_examples.jsonl"
MCP_SPEC_PATH = ROOT / "docs" / "thermo_credit_mcp_spec.md"
V2_SPEC_PATH = ROOT / "docs" / "thermo_credit_v2_spec.md"
DEFINITIONS_PATH = ROOT / "docs" / "definitions.md"
DATA_DICTIONARY_PATH = ROOT / "data" / "data_dictionary.csv"

INSTALL_HINT = 'Install the MCP SDK first: pip install "mcp[cli]"'


def _pretty_json(data: Mapping[str, Any]) -> str:
    return json.dumps(data, indent=2, ensure_ascii=False)


def get_resource_text(uri: str) -> str:
    if uri == "thermo-credit://overview":
        return _pretty_json(get_theory_overview_core({"detail_level": "full"}))
    if uri == "thermo-credit://definitions":
        return _pretty_json(
            get_variable_definitions_core({"include_existing_repo_metrics": True})
        )
    if uri == "thermo-credit://data-dictionary":
        return DATA_DICTIONARY_PATH.read_text(encoding="utf-8")
    if uri == "thermo-credit://limitations":
        overview = get_theory_overview_core({"detail_level": "full"})
        return _pretty_json({"limitations": overview["limitations"], "source_ids": overview["source_ids"]})
    if uri == "thermo-credit://worked-examples":
        return EXAMPLES_PATH.read_text(encoding="utf-8")
    raise KeyError(f"Unknown Thermo Credit resource: {uri}")


def _repo_metrics(region: str, limit: int) -> Dict[str, Any]:
    payload = build_repo_compute_payload(region, limit=limit)
    return compute_metrics_core(payload)


def build_explain_current_regime_prompt(region: str = "jp", limit: int = 4) -> str:
    metrics = _repo_metrics(region, limit)
    latest = metrics["metrics"][-1]
    return (
        f"Explain the current Thermo Credit regime for {region.upper()} using the latest measurable state. "
        f"Focus on q_t, asset bias, dissipation, and stress. Latest metrics: {json.dumps(latest, ensure_ascii=False)}. "
        "State explicitly which quantities are proxies and mention at least one falsifiability condition."
    )


def build_compare_periods_prompt(region: str = "jp", limit: int = 8) -> str:
    repo_payload = build_repo_compute_payload(region, limit=limit)
    observations = repo_payload["observations"]
    if len(observations) < 2:
        return (
            f"Compare two Thermo Credit regimes for {region.upper()}, but first ask for a longer observation window "
            "because fewer than two observations are available."
        )
    split = max(1, len(observations) // 2)
    period_a = {"label": "Earlier window", "observations": observations[:split]}
    period_b = {"label": "Later window", "observations": observations[split:]}
    comparison = compare_regimes_core(
        {
            "region": region,
            "frequency": repo_payload["frequency"],
            "period_a": period_a,
            "period_b": period_b,
            "__source_ids": repo_payload["__source_ids"],
        }
    )
    return (
        f"Compare the earlier and later Thermo Credit regimes for {region.upper()}. "
        f"Use this comparison summary as ground truth: {json.dumps(comparison, ensure_ascii=False)}. "
        "Describe whether the system became more asset-biased, more fragile, or less efficient."
    )


def build_stress_test_credit_mix_prompt(
    region: str = "jp",
    limit: int = 4,
    delta_credit_total: float = 20.0,
    delta_credit_real_share: float = -0.1,
    horizon_periods: int = 2,
) -> str:
    repo_metrics = _repo_metrics(region, limit)
    latest = repo_metrics["metrics"][-1]
    scenario = evaluate_scenario_core(
        {
            "region": region,
            "frequency": repo_metrics["frequency"],
            "baseline_state": {
                "credit_total": latest["credit_total"],
                "credit_real_share": latest["credit_real_share"],
                "stress_proxy": latest["stress_proxy"],
            },
            "scenario_shocks": {
                "delta_credit_total": delta_credit_total,
                "delta_credit_real_share": delta_credit_real_share,
            },
            "horizon_periods": horizon_periods,
        }
    )
    return (
        f"Stress-test the Thermo Credit mix for {region.upper()} by shocking total credit by {delta_credit_total} "
        f"and q_t by {delta_credit_real_share}. Use this scenario result: {json.dumps(scenario, ensure_ascii=False)}. "
        "Explain the trade-off between headline credit expansion, asset-directed flow, and fragility."
    )


def create_mcp_server() -> Any:
    try:
        from mcp.server.fastmcp import FastMCP
    except ImportError as exc:
        raise RuntimeError(INSTALL_HINT) from exc

    mcp = FastMCP(
        "Thermo Credit",
        instructions=(
            "Thermo Credit is a research-grade macro-financial model. Prefer structured JSON outputs, "
            "call get_variable_definitions before computing metrics when symbols may be ambiguous, "
            "and state clearly when q_t, asset_bias, or stress are proxy-derived."
        ),
        json_response=True,
    )

    @mcp.resource("thermo-credit://overview")
    def resource_overview() -> str:
        """Read the model overview as structured JSON text."""
        return get_resource_text("thermo-credit://overview")

    @mcp.resource("thermo-credit://definitions")
    def resource_definitions() -> str:
        """Read machine-friendly Thermo Credit variable definitions."""
        return get_resource_text("thermo-credit://definitions")

    @mcp.resource("thermo-credit://data-dictionary")
    def resource_data_dictionary() -> str:
        """Read the current data dictionary backing the measurement layer."""
        return get_resource_text("thermo-credit://data-dictionary")

    @mcp.resource("thermo-credit://limitations")
    def resource_limitations() -> str:
        """Read the current limitations and proxy notes."""
        return get_resource_text("thermo-credit://limitations")

    @mcp.resource("thermo-credit://worked-examples")
    def resource_worked_examples() -> str:
        """Read example tool calls and output excerpts."""
        return get_resource_text("thermo-credit://worked-examples")

    @mcp.tool()
    def get_theory_overview(detail_level: str = "standard") -> Dict[str, Any]:
        """Return the shortest reliable summary of the model and its failure modes."""
        return get_theory_overview_core({"detail_level": detail_level})

    @mcp.tool()
    def get_variable_definitions(
        symbols: list[str] | None = None,
        include_existing_repo_metrics: bool = False,
    ) -> Dict[str, Any]:
        """Return machine-readable definitions for Thermo Credit variables."""
        return get_variable_definitions_core(
            {
                "symbols": symbols or [],
                "include_existing_repo_metrics": include_existing_repo_metrics,
            }
        )

    @mcp.tool()
    def compute_thermo_credit_metrics(
        region: str,
        frequency: str,
        observations: list[dict[str, Any]],
        options: dict[str, Any] | None = None,
    ) -> Dict[str, Any]:
        """Compute baseline Thermo Credit metrics from structured observations."""
        return compute_metrics_core(
            {
                "region": region,
                "frequency": frequency,
                "observations": observations,
                "options": options or {},
            }
        )

    @mcp.tool()
    def evaluate_scenario(
        region: str,
        frequency: str,
        baseline_state: dict[str, Any],
        scenario_shocks: dict[str, Any],
        horizon_periods: int,
    ) -> Dict[str, Any]:
        """Evaluate a shocked Thermo Credit state against the baseline."""
        return evaluate_scenario_core(
            {
                "region": region,
                "frequency": frequency,
                "baseline_state": baseline_state,
                "scenario_shocks": scenario_shocks,
                "horizon_periods": horizon_periods,
            }
        )

    @mcp.tool()
    def compare_regimes(
        region: str,
        frequency: str,
        period_a: dict[str, Any],
        period_b: dict[str, Any],
        options: dict[str, Any] | None = None,
    ) -> Dict[str, Any]:
        """Compare two historical windows and diagnose the regime shift."""
        return compare_regimes_core(
            {
                "region": region,
                "frequency": frequency,
                "period_a": period_a,
                "period_b": period_b,
                "options": options or {},
            }
        )

    @mcp.prompt(title="Explain Current Regime")
    def explain_current_regime(region: str = "jp", limit: int = 4) -> str:
        """Generate a reusable prompt for explaining the latest measurable regime."""
        return build_explain_current_regime_prompt(region=region, limit=limit)

    @mcp.prompt(title="Compare Periods")
    def compare_periods(region: str = "jp", limit: int = 8) -> str:
        """Generate a reusable prompt for comparing two recent windows."""
        return build_compare_periods_prompt(region=region, limit=limit)

    @mcp.prompt(title="Stress-Test Credit Mix")
    def stress_test_credit_mix(
        region: str = "jp",
        limit: int = 4,
        delta_credit_total: float = 20.0,
        delta_credit_real_share: float = -0.1,
        horizon_periods: int = 2,
    ) -> str:
        """Generate a reusable prompt for a credit-mix stress test."""
        return build_stress_test_credit_mix_prompt(
            region=region,
            limit=limit,
            delta_credit_total=delta_credit_total,
            delta_credit_real_share=delta_credit_real_share,
            horizon_periods=horizon_periods,
        )

    return mcp


__all__ = [
    "INSTALL_HINT",
    "build_compare_periods_prompt",
    "build_explain_current_regime_prompt",
    "build_stress_test_credit_mix_prompt",
    "create_mcp_server",
    "get_resource_text",
]
