import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_claude_desktop_example_is_valid_json():
    path = ROOT / "examples" / "mcp" / "claude_desktop_config.thermo_credit.json"
    payload = json.loads(path.read_text(encoding="utf-8"))

    server = payload["mcpServers"]["thermo-credit"]
    assert server["type"] == "stdio"
    assert server["args"][-1] == "stdio"


def test_claude_code_project_example_is_valid_json():
    path = ROOT / "examples" / "mcp" / "claude_code_project.mcp.json"
    payload = json.loads(path.read_text(encoding="utf-8"))

    server = payload["mcpServers"]["thermo-credit"]
    assert server["command"].endswith("/.venv-mcp/bin/python")
    assert "thermo_credit_mcp_server.py" in server["args"][0]


def test_connection_guide_mentions_chatgpt_remote_requirement():
    guide = (ROOT / "docs" / "mcp_connection_guide.md").read_text(encoding="utf-8")

    assert "does not currently support local `stdio` MCP servers" in guide
    assert "Claude Desktop" in guide
    assert "ChatGPT" in guide
