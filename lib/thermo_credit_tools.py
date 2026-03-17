"""MCP-ready Thermo Credit v2 core tools.

This module keeps the model logic transport-agnostic so it can be exposed via
CLI, MCP, HTTP, or notebooks without rewriting the underlying calculations.
"""
from __future__ import annotations

import csv
import math
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Sequence

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
DEFINITIONS_PATH = ROOT / "docs" / "definitions.md"
DATA_DICTIONARY_PATH = ROOT / "data" / "data_dictionary.csv"

MODEL_VERSION = "thermo-credit-v2-draft"
DEFINITIONS_VERSION = "2026-03-draft"
INTERPRETATION_STATUS = "research"

DEFAULT_THEORY_OVERVIEW = {
    "purpose": [
        "Describe how the destination of new credit changes the translation from credit expansion into growth, prices, asset inflation, and fragility.",
        "Separate real-directed credit from asset-directed credit instead of treating aggregate credit volume as sufficient.",
        "Create a measurable bridge from thermodynamic language to predictive macro-financial diagnostics.",
    ],
    "state_vector": [
        "C_t",
        "q_t",
        "C_t^R",
        "C_t^A",
        "Y_t^N",
        "Y_t^R",
        "P_t",
        "A_t",
        "S_t",
    ],
    "hypotheses": [
        "H1: Lower q_t should lead asset-price acceleration before it shows up in real activity.",
        "H2: Higher C_t with low q_t should have weaker pass-through into growth and broad prices.",
        "H3: High C_t^A combined with high S_t should predict future volatility or credit contraction risk.",
    ],
    "falsifiability": [
        "If separating C_t^R from C_t^A does not improve explanatory or predictive performance, the partition is too weak.",
        "If falling q_t does not precede higher asset inflation or fragility, the destination channel is misspecified.",
        "If adding S_t does not improve downside diagnostics beyond existing indicators, the thermo extension should be simplified.",
    ],
    "limitations": [
        "q_t is not directly observed in most datasets and currently relies on proxy or latent-state design choices.",
        "Asset bias and dissipation are provisional proxies, not fully structural estimates.",
        "The current implementation is research-grade and designed for transparent recalculation rather than production forecasting.",
    ],
}

SECTION_CATEGORY_MAP = {
    "Core Variables": "core_variable",
    "Derived Thermo v2 Metrics": "derived_metric",
    "Existing Repo Thermo Diagnostics": "existing_repo_metric",
}

DETAIL_LEVEL_LIMITS = {
    "short": 2,
    "standard": 3,
    "full": 99,
}

TOOL_REGISTRY: Dict[str, Any] = {}


def _register(name: str):
    def decorator(func: Any) -> Any:
        TOOL_REGISTRY[name] = func
        return func

    return decorator


def _dedupe(items: Iterable[str]) -> List[str]:
    seen: set[str] = set()
    ordered: List[str] = []
    for item in items:
        text = str(item).strip()
        if not text or text in seen:
            continue
        seen.add(text)
        ordered.append(text)
    return ordered


def _finite_or_none(value: Any, digits: int = 6) -> float | None:
    try:
        num = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(num):
        return None
    return round(num, digits)


def _float_or_nan(value: Any) -> float:
    try:
        num = float(value)
    except (TypeError, ValueError):
        return math.nan
    return num if math.isfinite(num) else math.nan


def _clip_share(value: Any, fallback: float = 0.5) -> float:
    num = _float_or_nan(value)
    if math.isnan(num):
        return fallback
    return min(1.0, max(0.0, num))


def _markdown_row(line: str) -> List[str]:
    stripped = line.strip()
    if not stripped.startswith("|") or not stripped.endswith("|"):
        return []
    return [cell.strip() for cell in stripped.strip("|").split("|")]


def _strip_markdown_code(text: str) -> str:
    cleaned = str(text).strip()
    if cleaned.startswith("`") and cleaned.endswith("`") and len(cleaned) >= 2:
        return cleaned[1:-1]
    return cleaned


@lru_cache(maxsize=1)
def _definition_catalog() -> List[Dict[str, Any]]:
    lines = DEFINITIONS_PATH.read_text(encoding="utf-8").splitlines()
    current_section = ""
    rows: List[Dict[str, Any]] = []
    idx = 0
    while idx < len(lines):
        line = lines[idx]
        if line.startswith("## "):
            current_section = line[3:].strip()
        is_table = line.strip().startswith("|")
        next_is_separator = idx + 1 < len(lines) and set(lines[idx + 1].replace("|", "").strip()) <= {"-", " ", ":"}
        if is_table and next_is_separator and current_section in SECTION_CATEGORY_MAP:
            header = _markdown_row(lines[idx])
            idx += 2
            while idx < len(lines) and lines[idx].strip().startswith("|"):
                data = _markdown_row(lines[idx])
                if len(data) == len(header):
                    row = dict(zip(header, data))
                    rows.append(
                        {
                            "section": current_section,
                            "category": SECTION_CATEGORY_MAP[current_section],
                            "symbol": _strip_markdown_code(row.get("Symbol", "")),
                            "name": _strip_markdown_code(row.get("Name", "")),
                            "type": _strip_markdown_code(row.get("Type", "")),
                            "units": _strip_markdown_code(row.get("Units", "")),
                            "definition": _strip_markdown_code(row.get("Operational definition", "")),
                            "status": _strip_markdown_code(row.get("Current repo status", "")),
                        }
                    )
                idx += 1
            continue
        idx += 1
    return rows


@lru_cache(maxsize=1)
def _data_dictionary_by_symbol() -> Dict[str, Dict[str, str]]:
    with DATA_DICTIONARY_PATH.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        return {row["symbol"]: row for row in reader if row.get("symbol")}


def _definition_rows() -> List[Dict[str, Any]]:
    catalog = _definition_catalog()
    dictionary = _data_dictionary_by_symbol()
    out: List[Dict[str, Any]] = []
    for entry in catalog:
        dict_row = dictionary.get(entry["symbol"], {})
        artifacts = dict_row.get("repo_source", "")
        artifact_list = [
            part.strip().strip('"')
            for part in artifacts.split(";")
            if part.strip()
        ]
        out.append(
            {
                "symbol": entry["symbol"],
                "name": entry["name"],
                "category": entry["category"],
                "type": entry["type"],
                "units": entry["units"],
                "definition": entry["definition"] or dict_row.get("construction", ""),
                "status": entry["status"] or dict_row.get("status", ""),
                "empirical_role": dict_row.get("role", ""),
                "current_repo_artifacts": artifact_list,
            }
        )
    return out


def _region_indicator_path(region: str) -> Path:
    region = (region or "").strip().lower()
    path_map = {
        "jp": ROOT / "site" / "indicators.csv",
        "eu": ROOT / "site" / "indicators_eu.csv",
        "us": ROOT / "site" / "indicators_us.csv",
    }
    if region not in path_map:
        raise ValueError(f"Unsupported repo region: {region}")
    return path_map[region]


def build_repo_compute_payload(region: str, limit: int | None = None) -> Dict[str, Any]:
    path = _region_indicator_path(region)
    df = pd.read_csv(path)
    if limit is not None and limit > 0:
        df = df.tail(limit)
    return {
        "region": region,
        "frequency": "quarterly",
        "observations": df.to_dict(orient="records"),
        "__source_ids": [str(path.relative_to(ROOT))],
    }


def _derive_credit_total(df: pd.DataFrame, limitations: List[str]) -> pd.Series:
    if "credit_total" in df.columns:
        return pd.to_numeric(df["credit_total"], errors="coerce")
    if {"L_real", "L_asset"}.issubset(df.columns):
        limitations.append(
            "credit_total was derived from repo columns L_real + L_asset."
        )
        return pd.to_numeric(df["L_real"], errors="coerce").fillna(0.0) + pd.to_numeric(
            df["L_asset"], errors="coerce"
        ).fillna(0.0)
    raise ValueError("observations must provide credit_total or both L_real and L_asset")


def _derive_real_share(
    df: pd.DataFrame, carry_forward: bool, limitations: List[str]
) -> pd.Series:
    if "credit_real_share" in df.columns:
        series = pd.to_numeric(df["credit_real_share"], errors="coerce")
    elif {"L_real", "L_asset"}.issubset(df.columns):
        total = (
            pd.to_numeric(df["L_real"], errors="coerce").fillna(0.0)
            + pd.to_numeric(df["L_asset"], errors="coerce").fillna(0.0)
        )
        l_real = pd.to_numeric(df["L_real"], errors="coerce")
        series = l_real.div(total.where(total != 0))
        limitations.append(
            "credit_real_share was derived from repo columns L_real / (L_real + L_asset)."
        )
    else:
        series = pd.Series(index=df.index, dtype=float)
    if carry_forward:
        series = series.ffill()
    missing_mask = series.isna()
    if missing_mask.any():
        series.loc[missing_mask] = 0.5
        limitations.append(
            "Missing credit_real_share values were filled with a neutral 0.5 fallback."
        )
    return series.clip(lower=0.0, upper=1.0)


def _derive_nominal_output(df: pd.DataFrame, limitations: List[str]) -> pd.Series:
    if "nominal_output" in df.columns:
        return pd.to_numeric(df["nominal_output"], errors="coerce")
    if "Y" in df.columns:
        limitations.append("nominal_output reuses repo column Y as the nominal output proxy.")
        return pd.to_numeric(df["Y"], errors="coerce")
    if "U" in df.columns:
        limitations.append("nominal_output reuses repo column U as a bookkeeping output proxy.")
        return pd.to_numeric(df["U"], errors="coerce")
    return pd.Series(index=df.index, dtype=float)


def _derive_asset_proxy(df: pd.DataFrame, limitations: List[str]) -> pd.Series:
    if "asset_price_index" in df.columns:
        return pd.to_numeric(df["asset_price_index"], errors="coerce")
    if "L_asset" in df.columns:
        limitations.append(
            "asset_price_index reuses repo column L_asset as an asset-allocation proxy."
        )
        return pd.to_numeric(df["L_asset"], errors="coerce")
    return pd.Series(index=df.index, dtype=float)


def _structural_stress(
    prev_stress: float,
    credit_asset_flow: float,
    credit_real_flow: float,
    nominal_output: float,
) -> float:
    rho = 0.8
    gamma = 0.35
    delta = 0.15
    denom = nominal_output if math.isfinite(nominal_output) and abs(nominal_output) > 1e-12 else 1.0
    asset_term = credit_asset_flow / denom
    real_term = credit_real_flow / denom
    return rho * prev_stress + gamma * asset_term - delta * real_term


def _structural_efficiency(credit_real_share: float, stress_proxy: float) -> float:
    eta0 = 0.45
    alpha = 0.25
    beta = 0.35
    stress_term = stress_proxy if math.isfinite(stress_proxy) else 0.0
    return eta0 - alpha * (1.0 - credit_real_share) - beta * stress_term


def _derive_stress_proxy(df: pd.DataFrame, limitations: List[str]) -> pd.Series:
    if "stress_proxy" in df.columns:
        return pd.to_numeric(df["stress_proxy"], errors="coerce")
    if "p_C" in df.columns:
        limitations.append("stress_proxy reuses repo column p_C as a raw fragility proxy.")
        return pd.to_numeric(df["p_C"], errors="coerce")
    return pd.Series(index=df.index, dtype=float)


def _metric_rows(
    df: pd.DataFrame,
    normalize_by_output: bool,
    limitations: List[str],
) -> List[Dict[str, Any]]:
    metrics: List[Dict[str, Any]] = []
    prev_output = math.nan
    prev_real_output = math.nan
    prev_asset_proxy = math.nan
    prev_stress = 0.0
    used_structural_efficiency = False
    used_structural_stress = False

    for row in df.itertuples(index=False):
        credit_total = _float_or_nan(row.credit_total)
        credit_real_share = _clip_share(row.credit_real_share)
        credit_asset_share = 1.0 - credit_real_share
        credit_real_flow = credit_total * credit_real_share if math.isfinite(credit_total) else math.nan
        credit_asset_flow = credit_total * credit_asset_share if math.isfinite(credit_total) else math.nan
        nominal_output = _float_or_nan(getattr(row, "nominal_output"))
        real_output = _float_or_nan(getattr(row, "real_output_index"))
        asset_proxy = _float_or_nan(getattr(row, "asset_price_index"))
        stress_proxy = _float_or_nan(getattr(row, "stress_proxy"))

        if not math.isfinite(stress_proxy):
            stress_proxy = _structural_stress(prev_stress, credit_asset_flow, credit_real_flow, nominal_output)
            used_structural_stress = True

        if math.isfinite(nominal_output) and math.isfinite(prev_output) and abs(credit_total) > 1e-12:
            credit_efficiency = (nominal_output - prev_output) / credit_total
        elif math.isfinite(real_output) and math.isfinite(prev_real_output) and abs(credit_total) > 1e-12:
            credit_efficiency = (real_output - prev_real_output) / credit_total
        else:
            credit_efficiency = _structural_efficiency(credit_real_share, stress_proxy)
            used_structural_efficiency = True

        if math.isfinite(asset_proxy) and math.isfinite(prev_asset_proxy) and abs(credit_total) > 1e-12:
            if (
                normalize_by_output
                and math.isfinite(nominal_output)
                and nominal_output > 0
                and asset_proxy > 0
                and prev_asset_proxy > 0
            ):
                asset_bias = math.log(asset_proxy / prev_asset_proxy) / (credit_total / nominal_output)
            else:
                asset_bias = (asset_proxy - prev_asset_proxy) / credit_total
        else:
            if math.isfinite(nominal_output) and abs(nominal_output) > 1e-12:
                asset_bias = credit_asset_flow / nominal_output
            else:
                asset_bias = credit_asset_share

        metrics.append(
            {
                "date": pd.Timestamp(row.date).strftime("%Y-%m-%d"),
                "credit_total": _finite_or_none(credit_total) or 0.0,
                "credit_real_share": _finite_or_none(credit_real_share) or 0.0,
                "credit_asset_share": _finite_or_none(credit_asset_share) or 0.0,
                "credit_real_flow": _finite_or_none(credit_real_flow) or 0.0,
                "credit_asset_flow": _finite_or_none(credit_asset_flow) or 0.0,
                "credit_efficiency": _finite_or_none(credit_efficiency) or 0.0,
                "asset_bias": _finite_or_none(asset_bias) or 0.0,
                "dissipation_proxy": _finite_or_none(max(0.0, 1.0 - credit_efficiency)) or 0.0,
                "stress_proxy": _finite_or_none(stress_proxy) or 0.0,
            }
        )

        if math.isfinite(nominal_output):
            prev_output = nominal_output
        if math.isfinite(real_output):
            prev_real_output = real_output
        if math.isfinite(asset_proxy):
            prev_asset_proxy = asset_proxy
        prev_stress = stress_proxy if math.isfinite(stress_proxy) else prev_stress

    if used_structural_efficiency:
        limitations.append(
            "credit_efficiency falls back to the reduced-form eta_t equation when no lagged output series is available."
        )
    if used_structural_stress:
        limitations.append(
            "stress_proxy falls back to the reduced-form S_t recursion when no observed stress input is available."
        )
    return metrics


def _base_response(source_ids: Sequence[str] | None = None) -> Dict[str, Any]:
    sources = list(source_ids or [])
    if not sources:
        sources = ["custom_input"]
    return {
        "model_version": MODEL_VERSION,
        "definitions_version": DEFINITIONS_VERSION,
        "interpretation_status": INTERPRETATION_STATUS,
        "source_ids": _dedupe(sources),
    }


@_register("get_theory_overview")
def get_theory_overview(payload: Mapping[str, Any] | None = None) -> Dict[str, Any]:
    payload = dict(payload or {})
    detail_level = str(payload.get("detail_level", "standard")).strip().lower()
    item_limit = DETAIL_LEVEL_LIMITS.get(detail_level, DETAIL_LEVEL_LIMITS["standard"])
    overview = _base_response(
        ["docs/thermo_credit_v2_spec.md", "docs/definitions.md", "data/data_dictionary.csv"]
    )
    overview.update(
        {
            "purpose": DEFAULT_THEORY_OVERVIEW["purpose"][:item_limit],
            "state_vector": DEFAULT_THEORY_OVERVIEW["state_vector"],
            "hypotheses": DEFAULT_THEORY_OVERVIEW["hypotheses"][:item_limit],
            "falsifiability": DEFAULT_THEORY_OVERVIEW["falsifiability"][:item_limit],
            "limitations": DEFAULT_THEORY_OVERVIEW["limitations"][:item_limit],
        }
    )
    return overview


@_register("get_variable_definitions")
def get_variable_definitions(payload: Mapping[str, Any] | None = None) -> Dict[str, Any]:
    payload = dict(payload or {})
    requested_symbols = payload.get("symbols")
    symbols = {str(symbol).strip() for symbol in requested_symbols or [] if str(symbol).strip()}
    include_existing = bool(payload.get("include_existing_repo_metrics", False))

    variables: List[Dict[str, Any]] = []
    for row in _definition_rows():
        if not include_existing and row["category"] == "existing_repo_metric":
            continue
        if symbols and row["symbol"] not in symbols:
            continue
        variables.append(row)

    response = _base_response(["docs/definitions.md", "data/data_dictionary.csv"])
    response.update(
        {
            "variables": variables,
            "limitations": [
                "Variable meanings are fixed, but several v2 quantities remain proxy or planned implementations.",
                "Current repo thermo diagnostics are legacy-compatible and should not be conflated with q_t or S_t.",
            ],
        }
    )
    return response


@_register("compute_thermo_credit_metrics")
def compute_thermo_credit_metrics(payload: Mapping[str, Any]) -> Dict[str, Any]:
    payload = dict(payload or {})
    observations = payload.get("observations")
    if not observations:
        raise ValueError("compute_thermo_credit_metrics requires a non-empty observations array")

    region = str(payload.get("region", "custom")).strip().lower()
    frequency = str(payload.get("frequency", "quarterly")).strip().lower()
    options = payload.get("options") or {}
    normalize_by_output = bool(options.get("normalize_by_output", False))
    carry_forward = bool(options.get("carry_forward_missing_real_share", True))

    df = pd.DataFrame(list(observations)).copy()
    if "date" not in df.columns:
        raise ValueError("Each observation must include a date field.")
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date").reset_index(drop=True)

    limitations: List[str] = []
    df["credit_total"] = _derive_credit_total(df, limitations)
    df["credit_real_share"] = _derive_real_share(df, carry_forward, limitations)
    df["nominal_output"] = _derive_nominal_output(df, limitations)
    if "real_output_index" in df.columns:
        df["real_output_index"] = pd.to_numeric(df["real_output_index"], errors="coerce")
    else:
        df["real_output_index"] = math.nan
    df["asset_price_index"] = _derive_asset_proxy(df, limitations)
    df["stress_proxy"] = _derive_stress_proxy(df, limitations)

    metrics = _metric_rows(df, normalize_by_output, limitations)
    latest = metrics[-1]

    response = _base_response(payload.get("__source_ids"))
    response.update(
        {
            "region": region,
            "frequency": frequency,
            "metrics": metrics,
            "summary": {
                "latest_date": latest["date"],
                "latest_credit_real_share": latest["credit_real_share"],
                "latest_stress_proxy": latest["stress_proxy"],
            },
            "limitations": _dedupe(
                limitations
                + [
                    "Thermo Credit v2 metrics are provisional proxies and should be used as research diagnostics.",
                ]
            ),
        }
    )
    return response


def _state_to_metrics(state: Mapping[str, Any], prev_state: Mapping[str, Any] | None = None) -> Dict[str, float]:
    credit_total = _float_or_nan(state.get("credit_total"))
    credit_real_share = _clip_share(state.get("credit_real_share"))
    credit_asset_share = 1.0 - credit_real_share
    credit_real_flow = credit_total * credit_real_share
    credit_asset_flow = credit_total * credit_asset_share
    nominal_output = _float_or_nan(state.get("nominal_output"))
    asset_price_index = _float_or_nan(state.get("asset_price_index"))
    stress_proxy = _float_or_nan(state.get("stress_proxy"))
    if not math.isfinite(stress_proxy):
        prev_stress = _float_or_nan((prev_state or {}).get("stress_proxy"))
        stress_proxy = _structural_stress(
            0.0 if math.isnan(prev_stress) else prev_stress,
            credit_asset_flow,
            credit_real_flow,
            nominal_output,
        )
    credit_efficiency = _structural_efficiency(credit_real_share, stress_proxy)
    if math.isfinite(nominal_output) and abs(nominal_output) > 1e-12:
        asset_bias = credit_asset_flow / nominal_output
    else:
        asset_bias = credit_asset_share
    return {
        "credit_total": credit_total,
        "credit_real_share": credit_real_share,
        "credit_asset_share": credit_asset_share,
        "credit_real_flow": credit_real_flow,
        "credit_asset_flow": credit_asset_flow,
        "credit_efficiency": credit_efficiency,
        "asset_bias": asset_bias,
        "dissipation_proxy": max(0.0, 1.0 - credit_efficiency),
        "stress_proxy": stress_proxy,
        "nominal_output": nominal_output,
        "asset_price_index": asset_price_index,
    }


def _project_state(
    baseline_state: Mapping[str, Any],
    horizon_periods: int,
) -> Dict[str, float]:
    current = dict(baseline_state)
    metrics = _state_to_metrics(current)
    for _ in range(max(1, horizon_periods)):
        nominal_output = _float_or_nan(current.get("nominal_output"))
        if math.isfinite(nominal_output):
            current["nominal_output"] = nominal_output + metrics["credit_efficiency"] * metrics["credit_total"]
        asset_price_index = _float_or_nan(current.get("asset_price_index"))
        if math.isfinite(asset_price_index):
            current["asset_price_index"] = asset_price_index * (1.0 + metrics["asset_bias"])
        current["stress_proxy"] = _structural_stress(
            metrics["stress_proxy"],
            metrics["credit_asset_flow"],
            metrics["credit_real_flow"],
            _float_or_nan(current.get("nominal_output")),
        )
        metrics = _state_to_metrics(current, metrics)
    return metrics


@_register("evaluate_scenario")
def evaluate_scenario(payload: Mapping[str, Any]) -> Dict[str, Any]:
    payload = dict(payload or {})
    baseline_state = payload.get("baseline_state")
    if not isinstance(baseline_state, Mapping):
        raise ValueError("evaluate_scenario requires baseline_state")

    scenario_shocks = dict(payload.get("scenario_shocks") or {})
    horizon_periods = int(payload.get("horizon_periods", 1))

    shocked_state = dict(baseline_state)
    shocked_state["credit_total"] = _float_or_nan(baseline_state.get("credit_total")) + _float_or_nan(
        scenario_shocks.get("delta_credit_total", 0.0)
    )
    shocked_state["credit_real_share"] = _clip_share(
        _float_or_nan(baseline_state.get("credit_real_share"))
        + _float_or_nan(scenario_shocks.get("delta_credit_real_share", 0.0))
    )
    if "stress_proxy" in baseline_state or "delta_stress_proxy" in scenario_shocks:
        shocked_state["stress_proxy"] = _float_or_nan(baseline_state.get("stress_proxy")) + _float_or_nan(
            scenario_shocks.get("delta_stress_proxy", 0.0)
        )

    baseline_metrics = _project_state(baseline_state, horizon_periods)
    scenario_metrics = _project_state(shocked_state, horizon_periods)

    delta = {
        key: _finite_or_none(scenario_metrics[key] - baseline_metrics[key]) or 0.0
        for key in (
            "credit_total",
            "credit_real_share",
            "credit_asset_share",
            "credit_real_flow",
            "credit_asset_flow",
            "credit_efficiency",
            "asset_bias",
            "dissipation_proxy",
            "stress_proxy",
        )
    }

    summary_parts = []
    if delta["credit_asset_flow"] > 0:
        summary_parts.append("Asset-directed flow rises under the scenario.")
    if delta["credit_real_flow"] < 0:
        summary_parts.append("Real-directed flow weakens despite higher headline credit.")
    if delta["stress_proxy"] > 0:
        summary_parts.append("Projected stress is higher, which lowers the quality of additional credit expansion.")
    if not summary_parts:
        summary_parts.append("The scenario is close to baseline under the current reduced-form assumptions.")

    response = _base_response(["docs/thermo_credit_v2_spec.md"])
    response.update(
        {
            "baseline": {key: _finite_or_none(value) or 0.0 for key, value in baseline_metrics.items()},
            "scenario": {key: _finite_or_none(value) or 0.0 for key, value in scenario_metrics.items()},
            "delta": delta,
            "summary": " ".join(summary_parts),
            "limitations": [
                "Scenario evaluation uses the reduced-form eta_t and S_t recursions rather than a fully estimated state-space model.",
                "Nominal output and asset-price projections remain heuristic placeholders until dedicated forecast blocks are fitted.",
            ],
        }
    )
    return response


def _period_summary(payload: Mapping[str, Any], period: Mapping[str, Any]) -> Dict[str, float]:
    result = compute_thermo_credit_metrics(
        {
            "region": payload.get("region", "custom"),
            "frequency": payload.get("frequency", "quarterly"),
            "observations": period.get("observations", []),
            "options": payload.get("options", {}),
            "__source_ids": payload.get("__source_ids", []),
        }
    )
    df = pd.DataFrame(result["metrics"])
    return {
        "avg_credit_real_share": float(df["credit_real_share"].mean()),
        "avg_credit_asset_share": float(df["credit_asset_share"].mean()),
        "avg_credit_efficiency": float(df["credit_efficiency"].mean()),
        "avg_asset_bias": float(df["asset_bias"].mean()),
        "avg_dissipation_proxy": float(df["dissipation_proxy"].mean()),
        "avg_stress_proxy": float(df["stress_proxy"].mean()),
        "latest_credit_real_share": float(df["credit_real_share"].iloc[-1]),
        "latest_stress_proxy": float(df["stress_proxy"].iloc[-1]),
    }


@_register("compare_regimes")
def compare_regimes(payload: Mapping[str, Any]) -> Dict[str, Any]:
    payload = dict(payload or {})
    period_a = payload.get("period_a")
    period_b = payload.get("period_b")
    if not isinstance(period_a, Mapping) or not isinstance(period_b, Mapping):
        raise ValueError("compare_regimes requires period_a and period_b")

    summary_a = _period_summary(payload, period_a)
    summary_b = _period_summary(payload, period_b)
    comparison = {
        key: _finite_or_none(summary_b[key] - summary_a[key]) or 0.0
        for key in summary_a
    }

    summary_parts = []
    if comparison["avg_credit_real_share"] < 0:
        summary_parts.append(f"{period_b['label']} is more asset-biased than {period_a['label']}.")
    else:
        summary_parts.append(f"{period_b['label']} keeps at least as much real-credit share as {period_a['label']}.")
    if comparison["avg_stress_proxy"] > 0:
        summary_parts.append("Fragility rises on the stress proxy.")
    if comparison["avg_dissipation_proxy"] > 0:
        summary_parts.append("Credit conversion becomes less efficient on average.")

    response = _base_response(payload.get("__source_ids"))
    response.update(
        {
            "period_a_label": str(period_a.get("label", "period_a")),
            "period_b_label": str(period_b.get("label", "period_b")),
            "comparison": comparison,
            "summary": " ".join(summary_parts),
            "limitations": [
                "Regime comparison aggregates provisional proxy metrics.",
                "Interpretation should be checked against direct lending classifications when available.",
            ],
        }
    )
    return response


def run_tool(name: str, payload: Mapping[str, Any] | None = None) -> Dict[str, Any]:
    if name not in TOOL_REGISTRY:
        raise KeyError(f"Unknown Thermo Credit tool: {name}")
    return TOOL_REGISTRY[name](payload or {})


__all__ = [
    "MODEL_VERSION",
    "DEFINITIONS_VERSION",
    "INTERPRETATION_STATUS",
    "TOOL_REGISTRY",
    "build_repo_compute_payload",
    "compare_regimes",
    "compute_thermo_credit_metrics",
    "evaluate_scenario",
    "get_theory_overview",
    "get_variable_definitions",
    "run_tool",
]
