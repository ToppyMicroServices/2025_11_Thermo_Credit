# Thermo Credit MCP Connection Guide

Verified against official OpenAI and Anthropic documentation on 2026-03-17.

This guide explains how to connect the Thermo Credit MCP server to:

- Claude Desktop / Claude Code for local `stdio` use
- ChatGPT developer mode for remote MCP use

The repo now contains two separate layers:

- `lib/thermo_credit_tools.py`: transport-agnostic model logic
- `lib/thermo_credit_mcp.py`: FastMCP wrapper exposing resources, tools, and prompts

## 1. Python requirement

The MCP SDK dependency in this repo is enabled for Python `3.10` to `3.13`.
If your main project `.venv` is older, create a dedicated MCP environment.

Example:

```bash
python3.11 -m venv .venv-mcp
./.venv-mcp/bin/pip install -U pip
./.venv-mcp/bin/pip install -r requirements.txt -c constraints.txt
```

You can also use `python3.13` if that is what your machine provides.

## 2. Claude Desktop

Claude Desktop can use a local `stdio` MCP server. This is the easiest path for
local testing because it does not require public hosting.

Template config:

- `examples/mcp/claude_desktop_config.thermo_credit.json`

You need to replace the placeholder paths with absolute paths on your machine.
The most important fields are:

- `type: "stdio"`
- `command`: a compatible Python interpreter, typically `.venv-mcp/bin/python`
- `args`: the Thermo Credit MCP server script and `--transport stdio`

The Anthropic docs also note that if the executable is not on your `PATH`, you
should provide the full path explicitly.

## 3. Claude Code

Claude Code can use project-scoped MCP configuration via `.mcp.json`.

Template config:

- `examples/mcp/claude_code_project.mcp.json`

This file is appropriate when you want a team-visible project config that can
be checked into source control after replacing the placeholder paths.

Anthropic also documents:

- `claude mcp add --scope project ...` for project-scoped config
- `claude mcp add-from-claude-desktop` to import existing Claude Desktop MCPs

## 4. ChatGPT

ChatGPT developer mode is different from Claude Desktop in one important way:
it does not currently support local `stdio` MCP servers. The MCP server must be
publicly reachable over HTTPS.

Practical implication for this repo:

1. Run Thermo Credit with a remote MCP transport:

```bash
./.venv-mcp/bin/python scripts/thermo_credit_mcp_server.py --transport streamable-http
```

2. Put that server behind a public HTTPS endpoint such as:

```text
https://your-domain.example.com/mcp
```

3. In ChatGPT web, enable developer mode and create an app pointing at that
remote MCP endpoint.

Current OpenAI guidance says:

- Developer mode is enabled in `Settings -> Apps -> Advanced settings -> Developer mode`
- Apps are created from a remote MCP server in ChatGPT app settings
- Supported MCP protocols are `SSE` and `streaming HTTP`
- Supported auth modes are `OAuth`, `No Authentication`, and `Mixed Authentication`

Important current limits:

- ChatGPT custom MCP apps are web-only, not mobile
- Local MCP servers are not supported
- Search/fetch tools are no longer required
- As of 2026-03-17, Pro users can connect MCPs with read/fetch permissions in
  developer mode, while full MCP write support is documented for Business and
  Enterprise/Edu

## 5. Repo-specific recommendation

For Thermo Credit, the cleanest split is:

- Local research and testing: Claude Desktop or Claude Code with `stdio`
- External demo or broader AI access: ChatGPT with a public `streamable-http`
  endpoint

That matches the current repo structure well:

- local private workflows can use the repo directly
- remote AI-facing access can reuse the same tool layer without rewriting logic

## 6. Official references

- OpenAI developer mode guide:
  [developers.openai.com/api/docs/guides/developer-mode](https://developers.openai.com/api/docs/guides/developer-mode)
- OpenAI help article for apps and MCP in ChatGPT:
  [help.openai.com developer mode beta](https://help.openai.com/en/articles/12584461-developer-mode-apps-and-full-mcp-connectors-in-chatgpt-beta)
- Anthropic Claude Code MCP docs:
  [code.claude.com/docs/en/mcp](https://code.claude.com/docs/en/mcp)
