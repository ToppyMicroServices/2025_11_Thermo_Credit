"""Transport-neutral tools for the Thermo Credit measurement layer."""
from __future__ import annotations

import math
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

import pandas as pd

from lib.public_api import (
    MODEL_VERSION,
    REGION_SPECS,
    load_definition_records,
    load_region_frame,
)


ROOT = Path(__file__).resolve().parents[1]
INTERPRETATION_STATUS = "research"
TOOL_REGISTRY: dict[str, Callable[..., dict[str, Any]]] = {}


def _register(name: str) -> Callable[[Callable[..., dict[str, Any]]], Callable[..., dict[str, Any]]]:
    def decorator(function: Callable[..., dict[str, Any]]) -> Callable[..., dict[str, Any]]:
        TOOL_REGISTRY[name] = function
        return function

    return decorator


def _finite(value: Any, *, name: str) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be numeric") from exc
    if not math.isfinite(number):
        raise ValueError(f"{name} must be finite")
    return number


def _optional_finite(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return round(number, 8) if math.isfinite(number) else None


def _base(source_ids: Sequence[str]) -> dict[str, Any]:
    return {
        "model_version": MODEL_VERSION,
        "interpretation_status": INTERPRETATION_STATUS,
        "source_ids": list(dict.fromkeys(str(item) for item in source_ids if str(item))),
    }


@_register("get_theory_overview")
def get_theory_overview(payload: Mapping[str, Any] | None = None) -> dict[str, Any]:
    detail = str((payload or {}).get("detail_level", "standard")).lower()
    limit = {"short": 2, "standard": 3, "full": 99}.get(detail, 3)
    response = _base(
        [
            "tex/theory.tex",
            "docs/definitions.md",
            "docs/identification_strategy.md",
            "docs/calibration_protocol.md",
        ]
    )
    response.update(
        {
            "purpose": [
                "Measure credit scale and borrower composition on a reconciled population.",
                "Keep current measurements separate from the proposed credit-use partition.",
                "Test whether a composition coordinate adds information beyond matched credit-stock growth.",
            ][:limit],
            "current_measurement": {
                "q_t": "Japan four-quarter non-financial-business borrower-composition coordinate",
                "evidence_class": "derived_measurement",
                "claim_limit": "It does not identify loan purpose, GDP-linked use, or final expenditure.",
            },
            "proposed_structural_variables": ["C_t", "C_t^R", "C_t^A", "q_t^use", "S_t"],
            "falsification_rules": [
                "Do not claim forecasting value unless composition beats a matched-scale untouched baseline.",
                "Keep the structural Real/Asset partition unidentified until purpose-coded evidence is available.",
                "Keep dashboard quantities descriptive if calibration does not improve untouched holdout loss.",
            ][:limit],
            "limitations": [
                "EU and US allocation panels are proxies, not cross-country validation.",
                "Dashboard thermodynamic terms are model diagnostics, not physical quantities.",
                "Current pseudo-OOS results do not establish incremental predictability.",
            ][:limit],
        }
    )
    return response


@_register("get_variable_definitions")
def get_variable_definitions(payload: Mapping[str, Any] | None = None) -> dict[str, Any]:
    payload = dict(payload or {})
    requested = {str(item) for item in payload.get("symbols", []) if str(item)}
    include_diagnostics = bool(payload.get("include_model_diagnostics", False))
    rows = load_definition_records(ROOT / "data" / "data_dictionary.csv")
    variables = []
    for row in rows:
        if requested and row.get("symbol") not in requested:
            continue
        if not include_diagnostics and row.get("evidence_class") == "model_diagnostic":
            continue
        variables.append(row)
    response = _base(["docs/definitions.md", "data/data_dictionary.csv"])
    response.update(
        {
            "variables": variables,
            "limitations": [
                "The current q_t and proposed q_t^use are different variables.",
                "Evidence class and claim_limit must be preserved in downstream use.",
            ],
        }
    )
    return response


def build_repo_compute_payload(region: str, limit: int | None = None) -> dict[str, Any]:
    region = str(region).lower()
    if region not in REGION_SPECS:
        raise ValueError(f"Unsupported repo region: {region}")
    frame, source = load_region_frame(ROOT / "site", region)
    if limit is not None and limit > 0:
        frame = frame.tail(limit)
    scale_candidates = (
        ("primary_positive_flow", "C_t_primary", "C_t")
        if region == "jp"
        else ("C_t", "C_t_primary")
    )
    scale_column = next((column for column in scale_candidates if column in frame.columns), None)
    if scale_column is None or "q_t" not in frame.columns:
        raise ValueError(f"{region} panel lacks a credit scale or q_t column")

    observations: list[dict[str, Any]] = []
    carry_columns = ("Y", "S_M", "T_L", "p_C", "X_C", "loop_area")
    for _, row in frame.iterrows():
        observation: dict[str, Any] = {
            "date": pd.Timestamp(row["date"]).date().isoformat(),
            "credit_scale": _optional_finite(row.get(scale_column)),
            "allocation_share": _optional_finite(row.get("q_t")),
        }
        for column in carry_columns:
            if column in frame.columns:
                value = _optional_finite(row.get(column))
                if value is not None:
                    observation["output_proxy" if column == "Y" else column] = value
        if observation["credit_scale"] is not None and observation["allocation_share"] is not None:
            observations.append(observation)

    spec = REGION_SPECS[region]
    return {
        "region": region,
        "frequency": "quarterly",
        "allocation_semantics": spec["allocation_semantics"],
        "allocation_evidence_class": spec["evidence_class"],
        "observations": observations,
        "__source_ids": [source],
        "__claim_limit": spec["claim_limit"],
    }


@_register("compute_thermo_credit_metrics")
def compute_thermo_credit_metrics(payload: Mapping[str, Any]) -> dict[str, Any]:
    payload = dict(payload or {})
    observations = list(payload.get("observations") or [])
    if not observations:
        raise ValueError("A non-empty observations array is required")
    semantics = str(payload.get("allocation_semantics") or "unspecified allocation proxy")
    evidence_class = str(payload.get("allocation_evidence_class") or "unspecified")
    rows = sorted(observations, key=lambda row: str(row.get("date") or ""))
    metrics: list[dict[str, Any]] = []
    previous_share: float | None = None
    previous_output: float | None = None

    for row in rows:
        date = pd.to_datetime(row.get("date"), errors="coerce")
        if pd.isna(date):
            raise ValueError("Each observation needs a valid date")
        credit_scale = _finite(row.get("credit_scale"), name="credit_scale")
        share = _finite(row.get("allocation_share"), name="allocation_share")
        if not 0.0 <= share <= 1.0:
            raise ValueError("allocation_share must be between 0 and 1")
        output = _optional_finite(row.get("output_proxy"))
        metric = {
            "date": pd.Timestamp(date).date().isoformat(),
            "credit_scale": round(credit_scale, 8),
            "allocation_share": round(share, 8),
            "complement_share": round(1.0 - share, 8),
            "allocated_scale": round(credit_scale * share, 8),
            "complement_scale": round(credit_scale * (1.0 - share), 8),
            "allocation_change": None if previous_share is None else round(share - previous_share, 8),
            "output_change_per_credit": None,
        }
        if output is not None and previous_output is not None and abs(credit_scale) > 1e-12:
            metric["output_change_per_credit"] = round((output - previous_output) / credit_scale, 8)
        for diagnostic in ("S_M", "T_L", "p_C", "X_C", "loop_area"):
            value = _optional_finite(row.get(diagnostic))
            if value is not None:
                metric[diagnostic] = value
        metrics.append(metric)
        previous_share = share
        if output is not None:
            previous_output = output

    response = _base(payload.get("__source_ids") or ["custom_input"])
    response.update(
        {
            "region": str(payload.get("region") or "custom").lower(),
            "frequency": str(payload.get("frequency") or "quarterly").lower(),
            "allocation_semantics": semantics,
            "allocation_evidence_class": evidence_class,
            "metrics": metrics,
            "latest": metrics[-1],
            "limitations": [
                str(payload.get("__claim_limit") or "The allocation share is interpreted only as supplied."),
                "Allocated and complement scales are arithmetic decompositions, not causal effects.",
                "output_change_per_credit is a descriptive ratio and not an efficiency estimate unless separately identified.",
            ],
        }
    )
    return response


@_register("evaluate_scenario")
def evaluate_scenario(payload: Mapping[str, Any]) -> dict[str, Any]:
    payload = dict(payload or {})
    baseline = dict(payload.get("baseline_state") or {})
    shocks = dict(payload.get("scenario_shocks") or {})
    credit = _finite(baseline.get("credit_scale"), name="baseline credit_scale")
    share = _finite(baseline.get("allocation_share"), name="baseline allocation_share")
    shocked_credit = credit + _finite(shocks.get("delta_credit_scale", 0.0), name="delta_credit_scale")
    shocked_share = share + _finite(shocks.get("delta_allocation_share", 0.0), name="delta_allocation_share")
    if not 0.0 <= share <= 1.0 or not 0.0 <= shocked_share <= 1.0:
        raise ValueError("baseline and shocked allocation shares must be between 0 and 1")

    def state(scale: float, allocation: float) -> dict[str, float]:
        return {
            "credit_scale": round(scale, 8),
            "allocation_share": round(allocation, 8),
            "allocated_scale": round(scale * allocation, 8),
            "complement_scale": round(scale * (1.0 - allocation), 8),
        }

    before = state(credit, share)
    after = state(shocked_credit, shocked_share)
    delta = {key: round(after[key] - before[key], 8) for key in before}
    response = _base(payload.get("__source_ids") or ["user_supplied_scenario"])
    response.update(
        {
            "region": str(payload.get("region") or "custom").lower(),
            "horizon_periods": int(payload.get("horizon_periods") or 1),
            "baseline": before,
            "scenario": after,
            "delta": delta,
            "summary": "Arithmetic response to the user-supplied credit-scale and allocation-share shocks.",
            "limitations": [
                "No behavioral response, lag structure, price response, stress recursion, or forecast is estimated.",
                "The allocation meaning and evidence class must be supplied by the caller.",
            ],
        }
    )
    return response


def _regime_summary(payload: Mapping[str, Any], period: Mapping[str, Any]) -> dict[str, Any]:
    computed = compute_thermo_credit_metrics(
        {
            **payload,
            "observations": period.get("observations") or [],
            "__source_ids": payload.get("__source_ids") or ["custom_input"],
        }
    )
    frame = pd.DataFrame(computed["metrics"])
    means: dict[str, float] = {}
    for column in ("credit_scale", "allocation_share", "allocated_scale", "complement_scale", "S_M", "T_L", "p_C", "X_C", "loop_area"):
        if column in frame.columns:
            numeric = pd.to_numeric(frame[column], errors="coerce").dropna()
            if not numeric.empty:
                means[column] = round(float(numeric.mean()), 8)
    return {
        "label": str(period.get("label") or "period"),
        "observations": int(len(frame)),
        "mean": means,
    }


@_register("compare_regimes")
def compare_regimes(payload: Mapping[str, Any]) -> dict[str, Any]:
    payload = dict(payload or {})
    period_a = _regime_summary(payload, dict(payload.get("period_a") or {}))
    period_b = _regime_summary(payload, dict(payload.get("period_b") or {}))
    shared = sorted(set(period_a["mean"]) & set(period_b["mean"]))
    comparison = {
        key: round(period_b["mean"][key] - period_a["mean"][key], 8)
        for key in shared
    }
    response = _base(payload.get("__source_ids") or ["custom_input"])
    response.update(
        {
            "region": str(payload.get("region") or "custom").lower(),
            "period_a": period_a,
            "period_b": period_b,
            "change_b_minus_a": comparison,
            "summary": "Descriptive difference in period means; positive values mean period B is higher.",
            "limitations": [
                "Window selection can change the comparison.",
                "The result is not a causal estimate, regime classifier, or forecast.",
            ],
        }
    )
    return response


def run_tool(name: str, payload: Mapping[str, Any] | None = None) -> dict[str, Any]:
    if name not in TOOL_REGISTRY:
        raise KeyError(f"Unknown Thermo Credit tool: {name}")
    return TOOL_REGISTRY[name](payload or {})


__all__ = [
    "TOOL_REGISTRY",
    "build_repo_compute_payload",
    "compare_regimes",
    "compute_thermo_credit_metrics",
    "evaluate_scenario",
    "get_theory_overview",
    "get_variable_definitions",
    "run_tool",
]
