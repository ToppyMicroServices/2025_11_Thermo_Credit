import json
from pathlib import Path

import anyio
import pytest

from lib.thermo_credit_mcp import (
    build_compare_periods_prompt,
    build_explain_current_regime_prompt,
    build_stress_test_prompt,
    create_mcp_server,
    get_resource_text,
)


ROOT = Path(__file__).resolve().parents[1]


def test_resources_keep_q_t_claim_boundary() -> None:
    overview = json.loads(get_resource_text("thermo-credit://overview"))
    definitions = json.loads(get_resource_text("thermo-credit://definitions"))
    variables = {row["symbol"]: row for row in definitions["variables"]}
    assert overview["current_measurement"]["evidence_class"] == "derived_measurement"
    assert "loan-purpose" in variables["q_t"]["claim_limit"]
    assert variables["q_t^use"]["status"] == "not_identified"


def test_prompt_builders_use_descriptive_language() -> None:
    explain = build_explain_current_regime_prompt("jp", 2)
    compare = build_compare_periods_prompt("jp", 4)
    scenario = build_stress_test_prompt("jp", 2)
    assert "Do not call q_t a loan-purpose share" in explain
    assert "without causal or predictive language" in compare
    assert "Do not infer behavioral dynamics" in scenario


def test_mcp_assets_parse_as_json() -> None:
    schema_dir = ROOT / "schemas" / "thermo_credit"
    for path in schema_dir.glob("*.json"):
        json.loads(path.read_text(encoding="utf-8"))
    for line in (ROOT / "examples" / "thermo_credit_mcp_examples.jsonl").read_text(encoding="utf-8").splitlines():
        if line.strip():
            json.loads(line)


def test_server_can_be_created() -> None:
    pytest.importorskip("mcp.server")
    server = create_mcp_server()

    async def list_surfaces():
        return (
            await server.list_tools(),
            await server.list_resources(),
            await server.list_prompts(),
        )

    tools, resources, prompts = anyio.run(list_surfaces)
    assert {tool.name for tool in tools} == {
        "get_theory_overview",
        "get_variable_definitions",
        "compute_thermo_credit_metrics",
        "evaluate_scenario",
        "compare_regimes",
    }
    assert "thermo-credit://case-studies" in {str(resource.uri) for resource in resources}
    assert "explain_current_measurement" in {prompt.name for prompt in prompts}
