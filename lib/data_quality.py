"""Structural and freshness checks for published regional indicator panels."""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import pandas as pd


MODEL_VERSION = "2.2.0"
REGION_FILES = {
    "jp": "indicators.csv",
    "eu": "indicators_eu.csv",
    "us": "indicators_us.csv",
}
REQUIRED_COLUMNS = {
    "date",
    "S_M",
    "T_L",
    "p_C",
    "X_C",
    "loop_area",
    "preprocessing_mode",
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _iso_utc(value: datetime) -> str:
    if value.tzinfo is None:
        value = value.replace(tzinfo=timezone.utc)
    return value.astimezone(timezone.utc).isoformat().replace("+00:00", "Z")


def validate_indicator_file(
    path: Path,
    *,
    region: str,
    reference_time: datetime,
    min_rows: int,
    max_age_days: int | None,
) -> dict[str, Any]:
    errors: list[str] = []
    warnings: list[str] = []
    result: dict[str, Any] = {
        "region": region,
        "path": path.as_posix(),
        "errors": errors,
        "warnings": warnings,
    }

    if not path.exists():
        errors.append("file is missing")
        result["valid"] = False
        return result
    if path.stat().st_size == 0:
        errors.append("file is empty")
        result["valid"] = False
        return result

    try:
        frame = pd.read_csv(path)
    except Exception as exc:
        errors.append(f"CSV could not be read: {exc}")
        result["valid"] = False
        return result

    result["bytes"] = int(path.stat().st_size)
    result["sha256"] = _sha256(path)
    result["rows"] = int(len(frame))
    result["columns"] = int(len(frame.columns))
    if len(frame) < min_rows:
        errors.append(f"row count {len(frame)} is below required minimum {min_rows}")

    missing = sorted(REQUIRED_COLUMNS - set(frame.columns))
    if missing:
        errors.append("missing required columns: " + ", ".join(missing))

    if "date" in frame.columns:
        dates = pd.to_datetime(frame["date"], errors="coerce")
        if dates.isna().any():
            errors.append(f"{int(dates.isna().sum())} date values are invalid")
        valid_dates = dates.dropna()
        if not valid_dates.empty:
            latest = pd.Timestamp(valid_dates.max())
            earliest = pd.Timestamp(valid_dates.min())
            result["earliest_observation"] = earliest.date().isoformat()
            result["latest_observation"] = latest.date().isoformat()
            result["duplicate_dates"] = int(valid_dates.duplicated().sum())
            if result["duplicate_dates"]:
                errors.append(f"{result['duplicate_dates']} duplicate dates")
            if not valid_dates.is_monotonic_increasing:
                errors.append("dates are not monotonic increasing")
            age_days = int((pd.Timestamp(reference_time).tz_localize(None) - latest).days)
            result["age_days"] = age_days
            if max_age_days is not None and age_days > max_age_days:
                errors.append(f"latest observation is {age_days} days old; maximum is {max_age_days}")

    for column in sorted(REQUIRED_COLUMNS - {"date", "preprocessing_mode"}):
        if column in frame.columns and pd.to_numeric(frame[column], errors="coerce").notna().sum() == 0:
            errors.append(f"required metric {column} has no numeric observations")

    if "preprocessing_mode" in frame.columns:
        modes = sorted(str(value) for value in frame["preprocessing_mode"].dropna().unique())
        result["preprocessing_modes"] = modes
        if modes != ["dashboard_retrospective"]:
            warnings.append("published panel contains an unexpected preprocessing mode")

    result["valid"] = not errors
    return result


def validate_site_data(
    site_dir: Path,
    *,
    reference_time: datetime | None = None,
    min_rows: int = 8,
    max_age_days: int | None = None,
    regions: Iterable[str] | None = None,
    source_files: Iterable[Path] = (),
) -> dict[str, Any]:
    reference = reference_time or datetime.now(timezone.utc)
    selected_regions = tuple(regions) if regions is not None else tuple(REGION_FILES)
    region_results = [
        validate_indicator_file(
            site_dir / REGION_FILES[region],
            region=region,
            reference_time=reference,
            min_rows=min_rows,
            max_age_days=max_age_days,
        )
        for region in selected_regions
    ]

    sources: list[dict[str, Any]] = []
    for path in source_files:
        if not path.exists():
            continue
        record: dict[str, Any] = {"path": path.as_posix(), "sha256": _sha256(path)}
        try:
            record["selection"] = json.loads(path.read_text(encoding="utf-8"))
        except Exception as exc:
            record["parse_error"] = str(exc)
        sources.append(record)

    return {
        "schema_version": "1.0",
        "model_version": MODEL_VERSION,
        "generated_at": _iso_utc(reference),
        "valid": all(region["valid"] for region in region_results),
        "regions": region_results,
        "source_manifests": sources,
    }


__all__ = [
    "MODEL_VERSION",
    "REGION_FILES",
    "REQUIRED_COLUMNS",
    "validate_indicator_file",
    "validate_site_data",
]
