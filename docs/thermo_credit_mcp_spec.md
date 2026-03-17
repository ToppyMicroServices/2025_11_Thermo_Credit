# Thermo Credit MCP Minimal Spec

Status: draft for implementation

This document defines the minimum MCP-facing interface for Thermo Credit.
The goal is to make Thermo Credit callable by AI systems, not just readable by
humans.

It is designed to sit on top of:

- `docs/thermo_credit_v2_spec.md`
- `docs/definitions.md`
- `data/data_dictionary.csv`

## 1. Design Goal

The MCP layer should let an AI system do three things reliably:

1. fetch definitions and limits,
2. compute core Thermo Credit metrics from structured inputs,
3. explain differences across scenarios and regimes in a reusable way.

This is a distribution layer, not a validation layer. It helps models access
Thermo Credit correctly, but it does not prove the theory is right.

## 2. Server Surfaces

The server should expose three surface types.

### Resources

- `thermo-credit://overview`
- `thermo-credit://definitions`
- `thermo-credit://data-dictionary`
- `thermo-credit://limitations`
- `thermo-credit://worked-examples`

These are read-oriented resources for grounding and reuse.

### Tools

Phase 1 core tools:

1. `get_theory_overview`
2. `get_variable_definitions`
3. `compute_thermo_credit_metrics`

Phase 2 extension tools:

4. `evaluate_scenario`
5. `compare_regimes`

### Prompt Templates

- `explain_current_regime`
- `compare_periods`
- `stress-test_credit_mix`

Prompts are optional at first, but the tool responses should already be shaped
so that a prompt layer can consume them without extra parsing.

## 3. Tool Contract Principles

Every tool response should include:

- `model_version`
- `definitions_version`
- `interpretation_status`
- `limitations`
- `source_ids`

This keeps the output explainable and prevents silent drift.

Recommended status values:

- `research`
- `experimental`
- `production_candidate`

## 4. Tool Specs

### 4.1 `get_theory_overview`

Purpose:

Return the shortest reliable summary of the model, its purpose, and how it can
fail.

Inputs:

- optional `detail_level`: `short`, `standard`, `full`

Output schema:

- `schemas/thermo_credit/get_theory_overview.response.schema.json`

Key output fields:

- model purpose
- state vector
- hypotheses `H1` to `H3`
- falsifiability conditions
- implementation status

### 4.2 `get_variable_definitions`

Purpose:

Return machine-readable definitions for core variables and derived metrics.

Inputs:

- optional `symbols`
- optional `include_existing_repo_metrics`

Output schema:

- `schemas/thermo_credit/get_variable_definitions.response.schema.json`

Expected use:

- grounding before metric computation,
- resolving symbol meaning,
- preventing confusion between `q_t` and current entropy allocation shares.

### 4.3 `compute_thermo_credit_metrics`

Purpose:

Compute the baseline Thermo Credit v2 metrics from structured time-series input.

Input schema:

- `schemas/thermo_credit/compute_thermo_credit_metrics.request.schema.json`

Output schema:

- `schemas/thermo_credit/compute_thermo_credit_metrics.response.schema.json`

Core computed fields:

- `credit_real_share`
- `credit_asset_share`
- `credit_real_flow`
- `credit_asset_flow`
- `credit_efficiency`
- `asset_bias`
- `dissipation_proxy`
- `stress_proxy`

### 4.4 `evaluate_scenario`

Purpose:

Compare a baseline state with a hypothetical change such as:

- higher total credit,
- lower real-credit share,
- higher prior stress.

Input schema:

- `schemas/thermo_credit/evaluate_scenario.request.schema.json`

Output schema:

- `schemas/thermo_credit/evaluate_scenario.response.schema.json`

The response should combine numeric deltas with a short narrative explanation.

### 4.5 `compare_regimes`

Purpose:

Compare two historical windows or two precomputed metric sets and return a
diagnostic summary.

Input schema:

- `schemas/thermo_credit/compare_regimes.request.schema.json`

Output schema:

- `schemas/thermo_credit/compare_regimes.response.schema.json`

The response should say:

- what changed,
- whether the system became more asset-biased,
- whether fragility increased,
- which limitations matter.

## 5. Structured Output Conventions

The server should prefer structured JSON over mixed prose.

Recommended conventions:

- dates in ISO `YYYY-MM-DD`
- region codes: `jp`, `eu`, `us`, `custom`
- frequency: `monthly` or `quarterly`
- no hidden units
- narrative summaries limited to short paragraphs or bullet-like strings

If a metric is only a proxy, the response should say so directly.

## 6. Example Response Shape

Illustrative metric response:

```json
{
  "model_version": "thermo-credit-v2-draft",
  "definitions_version": "2026-03-draft",
  "interpretation_status": "research",
  "region": "jp",
  "frequency": "quarterly",
  "metrics": [
    {
      "date": "2025-03-31",
      "credit_total": 120.0,
      "credit_real_share": 0.62,
      "credit_asset_share": 0.38,
      "credit_real_flow": 74.4,
      "credit_asset_flow": 45.6,
      "credit_efficiency": 0.41,
      "asset_bias": 0.18,
      "dissipation_proxy": 0.59,
      "stress_proxy": 0.27
    }
  ],
  "limitations": [
    "credit_real_share is proxy-estimated",
    "asset_bias uses an index-based proxy"
  ],
  "source_ids": ["CRDQJPAPABIS", "JPNASSETS", "VIXCLS"]
}
```

## 7. Publication Stack

For Toppy, the best order is:

1. publish spec and schemas,
2. ship a minimal MCP server,
3. add a small ChatGPT-facing or web-facing demo,
4. add public examples and worked cases,
5. only then push directory discovery.

## 8. Suggested Repo Layout

```text
docs/
  thermo_credit_mcp_spec.md
  thermo_credit_v2_spec.md
  definitions.md

schemas/
  thermo_credit/
    *.schema.json

examples/
  thermo_credit_mcp_examples.jsonl

llms.txt
```

## 9. Immediate Next Step

The first implementation target should be:

1. `get_variable_definitions`
2. `compute_thermo_credit_metrics`

Those two tools are enough to move Thermo Credit from a static report toward an
AI-usable interface.
