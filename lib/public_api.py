"""Build the static, evidence-labelled Thermo Credit API."""
from __future__ import annotations

import csv
import json
import math
from pathlib import Path
from typing import Any, Iterable, Mapping

import pandas as pd


API_VERSION = "v1"
MODEL_VERSION = "thermo-credit-measurement-bridge-v1"
REGION_SPECS: dict[str, dict[str, Any]] = {
    "jp": {
        "label": "Japan",
        "files": ("indicators_jp.csv", "indicators.csv"),
        "evidence_class": "derived_measurement",
        "allocation_semantics": "four-quarter non-financial-business borrower-composition coordinate",
        "claim_limit": "Borrower composition only; it does not identify loan purpose or GDP-linked use.",
    },
    "eu": {
        "label": "Euro Area",
        "files": ("indicators_eu.csv",),
        "evidence_class": "proxy",
        "allocation_semantics": "configured regional allocation proxy",
        "claim_limit": "Schema-portability proxy; it is not validation of the Japanese measure.",
    },
    "us": {
        "label": "United States",
        "files": ("indicators_us.csv",),
        "evidence_class": "proxy",
        "allocation_semantics": "configured regional allocation proxy",
        "claim_limit": "Schema-portability proxy; it is not validation of the Japanese measure.",
    },
}
PUBLIC_METRICS = (
    "q_t",
    "one_minus_q_t",
    "C_t",
    "C_t_primary",
    "S_M",
    "T_L",
    "p_C",
    "U",
    "F_C",
    "X_C",
    "loop_area",
)
CASE_STUDY_METRICS = ("q_t", "X_C", "p_C", "S_M", "T_L", "loop_area")


def _json_number(value: Any, digits: int = 8) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(number):
        return None
    return round(number, digits)


def _write_json(path: Path, payload: Any) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return path


def load_region_frame(site_dir: Path, region: str) -> tuple[pd.DataFrame, str]:
    spec = REGION_SPECS[region]
    for filename in spec["files"]:
        path = site_dir / filename
        if not path.exists():
            continue
        frame = pd.read_csv(path)
        if frame.empty or "date" not in frame.columns:
            continue
        frame = frame.assign(date=pd.to_datetime(frame["date"], errors="coerce"))
        frame = frame.dropna(subset=["date"]).sort_values("date").reset_index(drop=True)
        if not frame.empty:
            return frame, f"site/{filename}"
    raise FileNotFoundError(f"No usable indicator panel found for {region} in {site_dir}")


def latest_region_payload(site_dir: Path, region: str) -> dict[str, Any]:
    frame, source = load_region_frame(site_dir, region)
    latest = frame.iloc[-1]
    metrics = {
        metric: number
        for metric in PUBLIC_METRICS
        if metric in frame.columns
        and (number := _json_number(latest.get(metric))) is not None
    }
    spec = REGION_SPECS[region]
    return {
        "api_version": API_VERSION,
        "model_version": MODEL_VERSION,
        "region": region,
        "region_label": spec["label"],
        "latest_date": pd.Timestamp(latest["date"]).date().isoformat(),
        "frequency": "quarterly",
        "source_id": source,
        "allocation_evidence_class": spec["evidence_class"],
        "allocation_semantics": spec["allocation_semantics"],
        "metrics": metrics,
        "claim_limit": spec["claim_limit"],
        "diagnostic_limit": (
            "S_M, T_L, p_C, U, F_C, X_C, and loop_area are model diagnostics, "
            "not validated policy thresholds or forecasts."
        ),
    }


def load_definition_records(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return [{key: value for key, value in row.items()} for row in csv.DictReader(handle)]


def _window_snapshot(
    frame: pd.DataFrame,
    start: pd.Timestamp,
    end: pd.Timestamp,
    metrics: Iterable[str],
) -> dict[str, Any]:
    selected = frame[(frame["date"] >= start) & (frame["date"] <= end)]
    values: dict[str, float] = {}
    for metric in metrics:
        if metric not in selected.columns:
            continue
        numeric = pd.to_numeric(selected[metric], errors="coerce").dropna()
        if not numeric.empty:
            values[metric] = round(float(numeric.mean()), 8)
    return {
        "start": start.date().isoformat(),
        "end": end.date().isoformat(),
        "observations": int(len(selected)),
        "mean": values,
    }


def _metric_delta(left: Mapping[str, Any], right: Mapping[str, Any]) -> dict[str, float]:
    left_means = left.get("mean", {})
    right_means = right.get("mean", {})
    shared = sorted(set(left_means) & set(right_means))
    return {
        metric: round(float(right_means[metric]) - float(left_means[metric]), 8)
        for metric in shared
    }


def build_case_studies(site_dir: Path, events_path: Path) -> list[dict[str, Any]]:
    events = pd.read_csv(events_path)
    studies: list[dict[str, Any]] = []
    frames = {region: load_region_frame(site_dir, region)[0] for region in REGION_SPECS}

    for event in events.to_dict(orient="records"):
        start = pd.to_datetime(event.get("start_date"), errors="coerce")
        end = pd.to_datetime(event.get("end_date"), errors="coerce")
        if pd.isna(start) or pd.isna(end) or end < start:
            continue
        region_field = str(event.get("regions") or "").strip().lower()
        event_regions = list(REGION_SPECS) if region_field == "all" else [
            part.strip() for part in region_field.replace(";", ",").split(",") if part.strip()
        ]
        duration = end - start
        pre_start = start - duration - pd.Timedelta(days=1)
        pre_end = start - pd.Timedelta(days=1)
        post_start = end + pd.Timedelta(days=1)
        post_end = end + duration + pd.Timedelta(days=1)

        for region in event_regions:
            if region not in frames:
                continue
            frame = frames[region]
            during = _window_snapshot(frame, start, end, CASE_STUDY_METRICS)
            if during["observations"] == 0 or not during["mean"]:
                continue
            before = _window_snapshot(frame, pre_start, pre_end, CASE_STUDY_METRICS)
            after = _window_snapshot(frame, post_start, post_end, CASE_STUDY_METRICS)
            studies.append(
                {
                    "key": str(event.get("key") or ""),
                    "label": str(event.get("label") or ""),
                    "region": region,
                    "category": str(event.get("category") or "event"),
                    "registered_window": {
                        "start": start.date().isoformat(),
                        "end": end.date().isoformat(),
                    },
                    "description": str(event.get("description") or ""),
                    "before": before,
                    "during": during,
                    "after": after,
                    "change_during_minus_before": _metric_delta(before, during),
                    "change_after_minus_during": _metric_delta(during, after),
                    "interpretation_status": "descriptive",
                    "claim_limit": (
                        "Registered-window comparison only. Timing overlap does not identify "
                        "causality, treatment effects, or predictive performance."
                    ),
                }
            )
    return sorted(studies, key=lambda row: (row["registered_window"]["start"], row["key"], row["region"]))


def build_public_api(root: Path, *, site_dir: Path | None = None) -> list[Path]:
    root = root.resolve()
    site_dir = (site_dir or root / "site").resolve()
    output_dir = site_dir / "api" / API_VERSION
    definitions = load_definition_records(root / "data" / "data_dictionary.csv")
    latest = {region: latest_region_payload(site_dir, region) for region in REGION_SPECS}
    case_studies = build_case_studies(site_dir, root / "data" / "report_events.csv")
    as_of = max(payload["latest_date"] for payload in latest.values())

    outputs = [
        _write_json(
            output_dir / "manifest.json",
            {
                "api_version": API_VERSION,
                "model_version": MODEL_VERSION,
                "as_of_latest_observation": as_of,
                "interpretation_status": "research",
                "endpoints": {
                    "definitions": "definitions.json",
                    "case_studies": "case-studies.json",
                    "jp_latest": "regions/jp/latest.json",
                    "eu_latest": "regions/eu/latest.json",
                    "us_latest": "regions/us/latest.json",
                },
                "claim_limit": (
                    "This static API distributes reproducible measurements and diagnostics. "
                    "It does not validate the theory or provide a production forecast."
                ),
            },
        ),
        _write_json(output_dir / "definitions.json", {"variables": definitions}),
        _write_json(output_dir / "case-studies.json", {"case_studies": case_studies}),
    ]
    for region, payload in latest.items():
        outputs.append(_write_json(output_dir / "regions" / region / "latest.json", payload))

    llms_text = "\n".join(
        [
            "# Thermo Credit static interface",
            "",
            f"API version: {API_VERSION}",
            f"Model version: {MODEL_VERSION}",
            "",
            "Start with api/v1/manifest.json and api/v1/definitions.json.",
            "Treat JP q_t as borrower composition, not loan purpose.",
            "Treat EU and US allocation fields as proxies.",
            "Carry claim_limit and interpretation_status into every downstream explanation.",
            "The event cases are descriptive registered-window comparisons, not causal estimates.",
            "",
        ]
    )
    llms_path = site_dir / "llms.txt"
    llms_path.write_text(llms_text, encoding="utf-8")
    outputs.append(llms_path)
    return outputs


__all__ = [
    "API_VERSION",
    "MODEL_VERSION",
    "REGION_SPECS",
    "build_case_studies",
    "build_public_api",
    "latest_region_payload",
    "load_definition_records",
    "load_region_frame",
]
