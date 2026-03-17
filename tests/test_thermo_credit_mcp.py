from lib.thermo_credit_mcp import (
    build_compare_periods_prompt,
    build_explain_current_regime_prompt,
    build_stress_test_credit_mix_prompt,
    get_resource_text,
)


def test_resources_are_readable():
    overview = get_resource_text("thermo-credit://overview")
    definitions = get_resource_text("thermo-credit://definitions")
    dictionary = get_resource_text("thermo-credit://data-dictionary")

    assert "thermo-credit-v2-draft" in overview
    assert '"symbol": "C_t"' in definitions
    assert "credit_total_flow" in dictionary


def test_prompt_builders_embed_actionable_context():
    explain_prompt = build_explain_current_regime_prompt(region="jp", limit=2)
    compare_prompt = build_compare_periods_prompt(region="jp", limit=4)
    stress_prompt = build_stress_test_credit_mix_prompt(region="jp", limit=2)

    assert "JP" in explain_prompt
    assert "q_t" in explain_prompt
    assert "asset-biased" in compare_prompt or "real-credit share" in compare_prompt
    assert "stress-test" in stress_prompt.lower()
    assert "fragility" in stress_prompt


def test_can_create_server_when_mcp_is_installed():
    fastmcp = __import__("pytest").importorskip("mcp.server.fastmcp")
    assert fastmcp is not None

    from lib.thermo_credit_mcp import create_mcp_server

    server = create_mcp_server()
    assert server is not None
