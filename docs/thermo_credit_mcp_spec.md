# Thermo Credit MCP interface

Status: research interface, version 1

The MCP server makes the current measurement layer callable. It does not turn
Thermo Credit into a validated forecast or make the server discoverable by
itself. The same calculations are available through the JSON CLI and the
versioned static API.

## Evidence rule

The current Japanese `q_t` is a four-quarter borrower-composition coordinate.
It is not the proposed credit-use share `q_t^use`. The EU and US fields with the
same display name are coarser proxies. Every response therefore carries an
evidence class, allocation semantics, source IDs, or a claim limit as
appropriate.

## Resources

- `thermo-credit://overview`
- `thermo-credit://definitions`
- `thermo-credit://data-dictionary`
- `thermo-credit://limitations`
- `thermo-credit://worked-examples`
- `thermo-credit://public-api-manifest`
- `thermo-credit://case-studies`

The last two resources require `python scripts/28_build_public_api.py` to have
run in the current checkout.

## Tools

- `get_theory_overview` returns the purpose, current evidence, limits, and
  falsification rules.
- `get_variable_definitions` returns source and claim metadata for selected
  symbols.
- `compute_thermo_credit_metrics` applies an arithmetic partition to supplied
  credit scale and allocation share observations.
- `evaluate_scenario` applies user-supplied shocks without estimating dynamics.
- `compare_regimes` compares period means without assigning causes.

Schemas are under `schemas/thermo_credit/`. Worked calls are in
`examples/thermo_credit_mcp_examples.jsonl`.

## Run locally

```bash
python scripts/28_build_public_api.py
python scripts/thermo_credit_mcp_server.py --transport stdio
```

For a separately secured HTTP deployment:

```bash
python scripts/thermo_credit_mcp_server.py --transport streamable-http
```

The default endpoint is `/mcp`. Public deployment, authentication, rate
limits, logging, and availability are outside this repository's static Pages
workflow. Streamable HTTP must not be exposed with sensitive inputs unless
those controls are supplied by the deployment environment.

## Non-MCP access

```bash
python scripts/thermo_credit_cli.py get_theory_overview
python scripts/thermo_credit_cli.py compute_thermo_credit_metrics --repo-region jp --limit 8
```

The static files begin at `site/api/v1/manifest.json`. GitHub Pages can publish
these read-only JSON files, but it cannot run the MCP server.
