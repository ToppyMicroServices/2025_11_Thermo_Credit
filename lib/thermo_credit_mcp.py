"""MCP resources, tools, and prompts for Thermo Credit."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping

from lib.thermo_credit_tools import (
    build_repo_compute_payload,
    compare_regimes as compare_regimes_core,
    compute_thermo_credit_metrics as compute_metrics_core,
    evaluate_scenario as evaluate_scenario_core,
    get_theory_overview as get_theory_overview_core,
    get_variable_definitions as get_variable_definitions_core,
)


ROOT = Path(__file__).resolve().parents[1]
INSTALL_HINT = 'Install the MCP SDK first: pip install "mcp[cli]>=2.1,<3"'


def _pretty(payload: Mapping[str, Any]) -> str:
    return json.dumps(payload, indent=2, ensure_ascii=False, sort_keys=True)


def get_resource_text(uri: str) -> str:
    if uri == "thermo-credit://overview":
        return _pretty(get_theory_overview_core({"detail_level": "full"}))
    if uri == "thermo-credit://definitions":
        return _pretty(get_variable_definitions_core({"include_model_diagnostics": True}))
    if uri == "thermo-credit://data-dictionary":
        return (ROOT / "data" / "data_dictionary.csv").read_text(encoding="utf-8")
    if uri == "thermo-credit://limitations":
        overview = get_theory_overview_core({"detail_level": "full"})
        return _pretty(
            {
                "interpretation_status": overview["interpretation_status"],
                "limitations": overview["limitations"],
                "falsification_rules": overview["falsification_rules"],
            }
        )
    if uri == "thermo-credit://worked-examples":
        return (ROOT / "examples" / "thermo_credit_mcp_examples.jsonl").read_text(encoding="utf-8")
    if uri == "thermo-credit://public-api-manifest":
        return (ROOT / "site" / "api" / "v1" / "manifest.json").read_text(encoding="utf-8")
    if uri == "thermo-credit://case-studies":
        return (ROOT / "site" / "api" / "v1" / "case-studies.json").read_text(encoding="utf-8")
    raise KeyError(f"Unknown Thermo Credit resource: {uri}")


def build_explain_current_regime_prompt(region: str = "jp", limit: int = 4) -> str:
    computed = compute_metrics_core(build_repo_compute_payload(region, limit=limit))
    return (
        f"Describe the latest {region.upper()} Thermo Credit measurement using this JSON: "
        f"{json.dumps(computed['latest'], ensure_ascii=False)}. "
        "Use the supplied allocation_semantics. State the evidence class and claim limits. "
        "Do not call q_t a loan-purpose share, policy threshold, or forecast."
    )


def build_compare_periods_prompt(region: str = "jp", limit: int = 8) -> str:
    payload = build_repo_compute_payload(region, limit=limit)
    observations = payload["observations"]
    if len(observations) < 2:
        return f"Ask for at least two observations before comparing {region.upper()} periods."
    split = max(1, len(observations) // 2)
    comparison = compare_regimes_core(
        {
            **payload,
            "period_a": {"label": "Earlier window", "observations": observations[:split]},
            "period_b": {"label": "Later window", "observations": observations[split:]},
        }
    )
    return (
        f"Compare the two {region.upper()} windows using this descriptive result: "
        f"{json.dumps(comparison, ensure_ascii=False)}. "
        "Report differences in period means without causal or predictive language."
    )


def build_stress_test_prompt(
    region: str = "jp",
    limit: int = 4,
    delta_credit_scale: float = 20.0,
    delta_allocation_share: float = -0.1,
) -> str:
    computed = compute_metrics_core(build_repo_compute_payload(region, limit=limit))
    latest = computed["latest"]
    scenario = evaluate_scenario_core(
        {
            "region": region,
            "baseline_state": {
                "credit_scale": latest["credit_scale"],
                "allocation_share": latest["allocation_share"],
            },
            "scenario_shocks": {
                "delta_credit_scale": delta_credit_scale,
                "delta_allocation_share": delta_allocation_share,
            },
            "horizon_periods": 1,
        }
    )
    return (
        f"Explain this arithmetic {region.upper()} scenario: {json.dumps(scenario, ensure_ascii=False)}. "
        "Do not infer behavioral dynamics, prices, fragility, or future outcomes."
    )


def create_mcp_server() -> Any:
    try:
        from mcp.server import MCPServer
    except ImportError as exc:
        raise RuntimeError(INSTALL_HINT) from exc

    server = MCPServer(
        "Thermo Credit",
        version="1.0.0",
        instructions=(
            "Use evidence_class and claim_limit in every explanation. The current JP q_t is a "
            "borrower-composition coordinate, while q_t^use is not identified. Tool outputs are "
            "research measurements or arithmetic scenarios, not production forecasts."
        ),
    )

    @server.resource("thermo-credit://overview")
    def resource_overview() -> str:
        """Read the model purpose, current evidence, and failure conditions."""
        return get_resource_text("thermo-credit://overview")

    @server.resource("thermo-credit://definitions")
    def resource_definitions() -> str:
        """Read evidence-labelled variable definitions."""
        return get_resource_text("thermo-credit://definitions")

    @server.resource("thermo-credit://data-dictionary")
    def resource_data_dictionary() -> str:
        """Read source, unit, construction, and claim-limit metadata."""
        return get_resource_text("thermo-credit://data-dictionary")

    @server.resource("thermo-credit://limitations")
    def resource_limitations() -> str:
        """Read limitations and falsification rules."""
        return get_resource_text("thermo-credit://limitations")

    @server.resource("thermo-credit://worked-examples")
    def resource_examples() -> str:
        """Read compact structured call examples."""
        return get_resource_text("thermo-credit://worked-examples")

    @server.resource("thermo-credit://public-api-manifest")
    def resource_manifest() -> str:
        """Read the current static API manifest."""
        return get_resource_text("thermo-credit://public-api-manifest")

    @server.resource("thermo-credit://case-studies")
    def resource_case_studies() -> str:
        """Read descriptive registered-event comparisons."""
        return get_resource_text("thermo-credit://case-studies")

    @server.tool()
    def get_theory_overview(detail_level: str = "standard") -> dict[str, Any]:
        """Return model purpose, current evidence, limitations, and falsification rules."""
        return get_theory_overview_core({"detail_level": detail_level})

    @server.tool()
    def get_variable_definitions(
        symbols: list[str] | None = None,
        include_model_diagnostics: bool = False,
    ) -> dict[str, Any]:
        """Return variable definitions with evidence classes and claim limits."""
        return get_variable_definitions_core(
            {
                "symbols": symbols or [],
                "include_model_diagnostics": include_model_diagnostics,
            }
        )

    @server.tool()
    def compute_thermo_credit_metrics(
        region: str,
        frequency: str,
        allocation_semantics: str,
        allocation_evidence_class: str,
        observations: list[dict[str, Any]],
    ) -> dict[str, Any]:
        """Compute an evidence-labelled arithmetic credit partition."""
        return compute_metrics_core(
            {
                "region": region,
                "frequency": frequency,
                "allocation_semantics": allocation_semantics,
                "allocation_evidence_class": allocation_evidence_class,
                "observations": observations,
            }
        )

    @server.tool()
    def evaluate_scenario(
        region: str,
        baseline_state: dict[str, Any],
        scenario_shocks: dict[str, Any],
        horizon_periods: int = 1,
    ) -> dict[str, Any]:
        """Apply user-supplied arithmetic shocks without estimating dynamics."""
        return evaluate_scenario_core(
            {
                "region": region,
                "baseline_state": baseline_state,
                "scenario_shocks": scenario_shocks,
                "horizon_periods": horizon_periods,
            }
        )

    @server.tool()
    def compare_regimes(
        region: str,
        allocation_semantics: str,
        allocation_evidence_class: str,
        period_a: dict[str, Any],
        period_b: dict[str, Any],
    ) -> dict[str, Any]:
        """Compare period means without assigning causes or forecasts."""
        return compare_regimes_core(
            {
                "region": region,
                "allocation_semantics": allocation_semantics,
                "allocation_evidence_class": allocation_evidence_class,
                "period_a": period_a,
                "period_b": period_b,
            }
        )

    @server.prompt(title="Explain current measurement")
    def explain_current_measurement(region: str = "jp", limit: int = 4) -> str:
        """Prepare an evidence-bounded explanation of the latest measurement."""
        return build_explain_current_regime_prompt(region, limit)

    @server.prompt(title="Compare two windows")
    def compare_two_windows(region: str = "jp", limit: int = 8) -> str:
        """Prepare a descriptive comparison of two recent windows."""
        return build_compare_periods_prompt(region, limit)

    @server.prompt(title="Apply an arithmetic credit shock")
    def arithmetic_credit_shock(
        region: str = "jp",
        limit: int = 4,
        delta_credit_scale: float = 20.0,
        delta_allocation_share: float = -0.1,
    ) -> str:
        """Prepare a bounded arithmetic scenario prompt."""
        return build_stress_test_prompt(region, limit, delta_credit_scale, delta_allocation_share)

    return server


__all__ = [
    "INSTALL_HINT",
    "build_compare_periods_prompt",
    "build_explain_current_regime_prompt",
    "build_stress_test_prompt",
    "create_mcp_server",
    "get_resource_text",
]
