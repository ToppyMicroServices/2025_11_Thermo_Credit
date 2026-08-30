"""Append-only release-vintage archive for the BOJ measurement bridge.

The archive deliberately separates a current-vintage protocol seed from
prospective release vintages.  It never overwrites a completed record:
recapturing identical content is idempotent, while changed content creates a
new, linked vintage.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
import shutil
import subprocess
import tempfile
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence


ARCHIVE_SCHEMA_VERSION = "1.0"
SEED_CAPTURE_CLASS = "protocol_baseline_current_vintage_seed"
PROSPECTIVE_CAPTURE_CLASS = "prospective_release_vintage"
CAPTURE_CLASSES = frozenset({SEED_CAPTURE_CLASS, PROSPECTIVE_CAPTURE_CLASS})
MAPPING_STABLE_FIELDS = (
    "db",
    "bucket_mapping",
    "construction",
    "primary_taxonomy_id",
    "primary_vector_columns",
    "primary_scale_stock_column",
    "taxonomy_selection",
    "taxonomies",
    "legacy_aliases",
    "common_taxonomy_stock_start",
    "first_valid_common_taxonomy_flow",
)
PROTOCOL_MAPPING_IDENTITY_FIELDS = (
    "primary_taxonomy_id",
    "primary_vector_columns",
    "primary_scale_stock_column",
    "taxonomy_selection",
)


class VintageArchiveError(RuntimeError):
    """Base error for invalid or unsafe archive operations."""


class ArchiveIntegrityError(VintageArchiveError):
    """Raised when an existing immutable record fails checksum verification."""


class ProspectiveEligibilityError(VintageArchiveError):
    """Raised when a record is not eligible for prospective scoring."""


@dataclass(frozen=True)
class CaptureResult:
    path: Path
    manifest: dict[str, Any]
    created: bool


@dataclass(frozen=True)
class ScoreResult:
    path: Path
    record: dict[str, Any]
    created: bool


def canonical_json_bytes(value: Any) -> bytes:
    """Return deterministic UTF-8 JSON suitable for hashing."""

    return (
        json.dumps(
            value,
            ensure_ascii=False,
            allow_nan=False,
            sort_keys=True,
            separators=(",", ":"),
        )
        + "\n"
    ).encode("utf-8")


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _parse_utc(value: str, *, field: str) -> datetime:
    text = str(value).strip()
    if not text:
        raise VintageArchiveError(f"{field} must not be empty.")
    if text.endswith("Z"):
        text = text[:-1] + "+00:00"
    try:
        parsed = datetime.fromisoformat(text)
    except ValueError as exc:
        raise VintageArchiveError(
            f"{field} must be an ISO-8601 timestamp with a UTC offset."
        ) from exc
    if parsed.tzinfo is None:
        raise VintageArchiveError(f"{field} must include a UTC offset.")
    return parsed.astimezone(timezone.utc)


def normalize_utc(value: str, *, field: str) -> str:
    parsed = _parse_utc(value, field=field)
    if parsed.microsecond:
        return parsed.isoformat(timespec="microseconds").replace("+00:00", "Z")
    return parsed.isoformat(timespec="seconds").replace("+00:00", "Z")


def _required_mapping_snapshot(
    metadata: Mapping[str, Any],
    *,
    mapping_version: str,
) -> dict[str, Any]:
    if not mapping_version.strip():
        raise VintageArchiveError("mapping.version must not be empty.")
    missing = [field for field in MAPPING_STABLE_FIELDS if field not in metadata]
    if missing:
        raise VintageArchiveError(
            "Mapping metadata is missing stable fields: " + ", ".join(missing)
        )
    series_metadata = metadata.get("series_metadata")
    if not isinstance(series_metadata, Mapping) or not series_metadata:
        raise VintageArchiveError(
            "Mapping metadata must contain non-empty series_metadata."
        )
    primary_taxonomy_id = str(metadata["primary_taxonomy_id"]).strip()
    primary_vector = metadata["primary_vector_columns"]
    primary_scale = str(metadata["primary_scale_stock_column"]).strip()
    selection = metadata["taxonomy_selection"]
    taxonomies = metadata["taxonomies"]
    if not primary_taxonomy_id:
        raise VintageArchiveError("primary_taxonomy_id must not be empty.")
    if (
        not isinstance(primary_vector, Sequence)
        or isinstance(primary_vector, (str, bytes))
        or not primary_vector
    ):
        raise VintageArchiveError("primary_vector_columns must be a non-empty list.")
    if not primary_scale:
        raise VintageArchiveError("primary_scale_stock_column must not be empty.")
    if not isinstance(selection, Mapping):
        raise VintageArchiveError("taxonomy_selection must be an object.")
    if not isinstance(taxonomies, Mapping) or primary_taxonomy_id not in taxonomies:
        raise VintageArchiveError(
            "taxonomies must contain the declared primary_taxonomy_id."
        )
    if str(selection.get("primary_taxonomy_id", "")) != primary_taxonomy_id:
        raise VintageArchiveError(
            "taxonomy_selection.primary_taxonomy_id must match primary_taxonomy_id."
        )
    if selection.get("oos_results_used") is not False:
        raise VintageArchiveError(
            "taxonomy_selection must state that OOS results were not used."
        )
    outcome_columns = selection.get("outcome_columns_used")
    if not isinstance(outcome_columns, Sequence) or isinstance(
        outcome_columns,
        (str, bytes),
    ) or list(outcome_columns):
        raise VintageArchiveError(
            "taxonomy_selection.outcome_columns_used must be an empty list."
        )
    declared_taxonomy_ids = [
        primary_taxonomy_id,
        *[str(value) for value in selection.get("robustness_taxonomy_ids", [])],
        str(selection.get("legacy_taxonomy_id", "")),
    ]
    missing_taxonomies = [
        value for value in declared_taxonomy_ids if value and value not in taxonomies
    ]
    if missing_taxonomies:
        raise VintageArchiveError(
            "taxonomies is missing selected IDs: " + ", ".join(missing_taxonomies)
        )
    series_ids = sorted(str(value) for value in series_metadata)
    snapshot = {
        "mapping_version": mapping_version,
        "series_ids": series_ids,
    }
    snapshot.update({field: metadata[field] for field in MAPPING_STABLE_FIELDS})
    return snapshot


def _validate_protocol(protocol: Mapping[str, Any]) -> None:
    required = ("protocol_id", "protocol_version", "mapping", "source", "evaluation")
    missing = [field for field in required if field not in protocol]
    if missing:
        raise VintageArchiveError(
            "Protocol is missing required fields: " + ", ".join(missing)
        )
    mapping = protocol["mapping"]
    if not isinstance(mapping, Mapping) or not str(mapping.get("version", "")).strip():
        raise VintageArchiveError("Protocol mapping.version must not be empty.")
    missing_mapping = [
        field for field in PROTOCOL_MAPPING_IDENTITY_FIELDS if field not in mapping
    ]
    if missing_mapping:
        raise VintageArchiveError(
            "Protocol mapping is missing identity fields: "
            + ", ".join(missing_mapping)
        )
    evaluation = protocol["evaluation"]
    if not isinstance(evaluation, Mapping):
        raise VintageArchiveError("Protocol evaluation must be an object.")
    for field in ("outcomes", "horizons_quarters", "benchmark", "loss", "revision_policy"):
        if field not in evaluation:
            raise VintageArchiveError(f"Protocol evaluation.{field} is required.")


def git_code_state(
    repo_root: Path,
    *,
    code_files: Sequence[Path] = (),
) -> dict[str, Any]:
    """Record an anonymous code identity without paths or author metadata."""

    try:
        commit = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=repo_root,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
        dirty = bool(
            subprocess.run(
                ["git", "status", "--porcelain", "--untracked-files=normal"],
                cwd=repo_root,
                check=True,
                capture_output=True,
                text=True,
            ).stdout.strip()
        )
    except (OSError, subprocess.CalledProcessError) as exc:
        raise VintageArchiveError("Unable to resolve the repository code state.") from exc

    file_hashes = {
        path.name: sha256_file(path)
        for path in code_files
        if path.is_file()
    }
    return {
        "git_commit": commit,
        "worktree_dirty": dirty,
        "capture_file_sha256": file_hashes,
    }


def _safe_member_name(name: str) -> str:
    if not name or Path(name).name != name or name in {".", ".."}:
        raise ArchiveIntegrityError(f"Unsafe archive member name: {name!r}")
    return name


def _read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ArchiveIntegrityError(f"Cannot read valid JSON from {path.name}.") from exc
    if not isinstance(value, dict):
        raise ArchiveIntegrityError(f"{path.name} must contain a JSON object.")
    return value


def _snapshot_directories(archive_root: Path) -> list[Path]:
    root = archive_root / "snapshots"
    if not root.exists():
        return []
    return sorted(
        path
        for path in root.iterdir()
        if path.is_dir() and not path.name.startswith(".")
    )


def list_vintages(archive_root: Path) -> list[dict[str, Any]]:
    manifests: list[dict[str, Any]] = []
    for path in _snapshot_directories(archive_root):
        manifest_path = path / "manifest.json"
        if manifest_path.is_file():
            manifests.append(_read_json(manifest_path))
    return sorted(
        manifests,
        key=lambda item: (
            str(item.get("release", {}).get("timestamp_utc", "")),
            str(item.get("vintage_id", "")),
        ),
    )


def verify_vintage(path: Path) -> dict[str, Any]:
    manifest_path = path / "manifest.json"
    checksum_path = path / "manifest.sha256"
    manifest = _read_json(manifest_path)

    try:
        expected_manifest_hash = checksum_path.read_text(encoding="ascii").split()[0]
    except (OSError, IndexError) as exc:
        raise ArchiveIntegrityError(
            f"Missing or invalid manifest checksum for {path.name}."
        ) from exc
    actual_manifest_hash = sha256_file(manifest_path)
    if expected_manifest_hash != actual_manifest_hash:
        raise ArchiveIntegrityError(f"Manifest checksum mismatch for {path.name}.")

    checks = (
        ("raw_payload", "sha256"),
        ("mapping", "sha256"),
        ("protocol", "sha256"),
    )
    for section_name, checksum_name in checks:
        section = manifest.get(section_name)
        if not isinstance(section, Mapping):
            raise ArchiveIntegrityError(
                f"Manifest section {section_name} is missing for {path.name}."
            )
        member_name = _safe_member_name(str(section.get("file", "")))
        member_path = path / member_name
        if not member_path.is_file():
            raise ArchiveIntegrityError(
                f"Archive member {member_name} is missing for {path.name}."
            )
        if sha256_file(member_path) != str(section.get(checksum_name, "")):
            raise ArchiveIntegrityError(
                f"Checksum mismatch for {member_name} in {path.name}."
            )
    if manifest.get("vintage_id") != path.name:
        raise ArchiveIntegrityError(f"Vintage ID/path mismatch for {path.name}.")
    return manifest


def verify_archive(archive_root: Path) -> list[dict[str, Any]]:
    return [verify_vintage(path) for path in _snapshot_directories(archive_root)]


def _payload_filename(media_type: str, content_encoding: str | None) -> str:
    if content_encoding and content_encoding.lower() == "gzip":
        return "raw_payload.gz"
    lowered = media_type.lower()
    if "json" in lowered:
        return "raw_payload.json"
    if "csv" in lowered:
        return "raw_payload.csv"
    return "raw_payload.bin"


def _write_completed_directory(
    *,
    parent: Path,
    destination: Path,
    members: Mapping[str, bytes],
    checksum_member: str,
    checksum_target: str,
) -> None:
    parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=".capture-", dir=parent))
    try:
        for name, payload in members.items():
            _safe_member_name(name)
            (temporary / name).write_bytes(payload)
        checksum = sha256_file(temporary / checksum_target)
        (temporary / checksum_member).write_text(
            f"{checksum}  {checksum_target}\n",
            encoding="ascii",
        )
        os.rename(temporary, destination)
    except Exception:
        if temporary.exists():
            shutil.rmtree(temporary)
        raise


def capture_vintage(
    *,
    archive_root: Path,
    raw_payload: bytes,
    protocol: Mapping[str, Any],
    mapping_metadata: Mapping[str, Any],
    mapping_metadata_bytes: bytes,
    capture_class: str,
    release_timestamp: str,
    retrieved_at: str,
    release_timestamp_source: str,
    release_timestamp_precision: str,
    source_url: str,
    media_type: str,
    content_encoding: str | None,
    capture_method: str,
    code_state: Mapping[str, Any],
) -> CaptureResult:
    """Capture a seed or future release without overwriting prior records."""

    _validate_protocol(protocol)
    if capture_class not in CAPTURE_CLASSES:
        raise VintageArchiveError(
            f"capture_class must be one of {', '.join(sorted(CAPTURE_CLASSES))}."
        )
    if not raw_payload:
        raise VintageArchiveError("raw_payload must not be empty.")
    if not source_url.strip():
        raise VintageArchiveError("source_url must not be empty.")
    if not release_timestamp_source.strip():
        raise VintageArchiveError("release_timestamp_source must not be empty.")
    if not release_timestamp_precision.strip():
        raise VintageArchiveError("release_timestamp_precision must not be empty.")

    release_utc = normalize_utc(release_timestamp, field="release_timestamp")
    retrieved_utc = normalize_utc(retrieved_at, field="retrieved_at")
    release_dt = _parse_utc(release_utc, field="release_timestamp")
    retrieved_dt = _parse_utc(retrieved_utc, field="retrieved_at")
    if retrieved_dt < release_dt:
        raise VintageArchiveError("retrieved_at cannot precede release_timestamp.")

    registration_value = protocol.get("registration_timestamp_utc")
    registration_utc: str | None = None
    if registration_value:
        registration_utc = normalize_utc(
            str(registration_value),
            field="registration_timestamp_utc",
        )
    if capture_class == PROSPECTIVE_CAPTURE_CLASS:
        if registration_utc is None:
            raise ProspectiveEligibilityError(
                "Prospective capture is disabled until registration_timestamp_utc "
                "is fixed in the protocol."
            )
        registration_dt = _parse_utc(
            registration_utc,
            field="registration_timestamp_utc",
        )
        if release_dt <= registration_dt:
            raise ProspectiveEligibilityError(
                "A prospective release must be strictly later than protocol registration."
            )

    mapping_version = str(protocol["mapping"]["version"])
    mapping_snapshot = _required_mapping_snapshot(
        mapping_metadata,
        mapping_version=mapping_version,
    )
    protocol_mapping = protocol["mapping"]
    for field in PROTOCOL_MAPPING_IDENTITY_FIELDS:
        if canonical_json_bytes(protocol_mapping[field]) != canonical_json_bytes(
            mapping_snapshot[field]
        ):
            raise VintageArchiveError(
                f"Protocol mapping.{field} does not match mapping metadata."
            )
    mapping_bytes = canonical_json_bytes(mapping_snapshot)
    protocol_bytes = canonical_json_bytes(dict(protocol))
    raw_hash = sha256_bytes(raw_payload)
    mapping_hash = sha256_bytes(mapping_bytes)
    protocol_hash = sha256_bytes(protocol_bytes)
    series_ids = mapping_snapshot["series_ids"]

    identity = {
        "capture_class": capture_class,
        "mapping_sha256": mapping_hash,
        "protocol_sha256": protocol_hash,
        "raw_payload_sha256": raw_hash,
        "release_timestamp_utc": release_utc,
        "series_ids": series_ids,
    }
    identity_hash = sha256_bytes(canonical_json_bytes(identity))
    vintage_id = (
        "boj-"
        + release_dt.strftime("%Y%m%dT%H%M%SZ")
        + "-"
        + identity_hash[:20]
    )

    prior_same_release: list[dict[str, Any]] = []
    for prior in verify_archive(archive_root):
        if prior.get("release", {}).get("timestamp_utc") != release_utc:
            continue
        prior_same_release.append(prior)
        prior_identity = prior.get("capture_identity_sha256")
        if prior_identity == identity_hash:
            prior_path = archive_root / "snapshots" / str(prior["vintage_id"])
            verified = verify_vintage(prior_path)
            return CaptureResult(path=prior_path, manifest=verified, created=False)

    raw_changed = any(
        prior.get("raw_payload", {}).get("sha256") != raw_hash
        for prior in prior_same_release
    )
    prior_ids = [str(prior.get("vintage_id")) for prior in prior_same_release]
    payload_name = _payload_filename(media_type, content_encoding)
    eligible = capture_class == PROSPECTIVE_CAPTURE_CLASS
    manifest = {
        "schema_version": ARCHIVE_SCHEMA_VERSION,
        "record_type": "boj_release_vintage",
        "vintage_id": vintage_id,
        "capture_class": capture_class,
        "eligible_for_prospective_scoring": eligible,
        "evidence_boundary": (
            "Future release-vintage evidence eligible under the registered protocol."
            if eligible
            else "Current-vintage seed for protocol and audit only; not preregistered "
            "evidence and never eligible for prospective scoring."
        ),
        "capture_identity_sha256": identity_hash,
        "release": {
            "timestamp_utc": release_utc,
            "timestamp_source": release_timestamp_source,
            "timestamp_precision": release_timestamp_precision,
            "retrieved_at_utc": retrieved_utc,
        },
        "source": {
            "name": str(protocol["source"].get("name", "")),
            "url": source_url,
            "api_endpoint": str(protocol["source"].get("api_endpoint", "")),
            "database": str(protocol["source"].get("database", "")),
            "series_ids": series_ids,
            "capture_method": capture_method,
            "content_encoding": content_encoding or "identity",
        },
        "raw_payload": {
            "file": payload_name,
            "sha256": raw_hash,
            "bytes": len(raw_payload),
            "media_type": media_type,
        },
        "mapping": {
            "file": "mapping.json",
            "version": mapping_version,
            "sha256": mapping_hash,
            "metadata_sha256": sha256_bytes(mapping_metadata_bytes),
            "primary_taxonomy_id": mapping_snapshot["primary_taxonomy_id"],
            "primary_vector_columns": mapping_snapshot["primary_vector_columns"],
            "primary_scale_stock_column": mapping_snapshot[
                "primary_scale_stock_column"
            ],
            "taxonomy_selection": mapping_snapshot["taxonomy_selection"],
            "taxonomy_ids": sorted(mapping_snapshot["taxonomies"]),
        },
        "protocol": {
            "file": "protocol.json",
            "id": str(protocol["protocol_id"]),
            "version": str(protocol["protocol_version"]),
            "sha256": protocol_hash,
            "registration_timestamp_utc": registration_utc,
        },
        "code": dict(code_state),
        "evaluation": dict(protocol["evaluation"]),
        "revision": {
            "sequence_for_release": len(prior_same_release) + 1,
            "prior_vintage_ids_for_release": prior_ids,
            "raw_payload_changed_from_prior_capture": raw_changed,
            "policy": (
                "No prior record is overwritten. A changed payload, mapping, or "
                "protocol creates a new linked vintage."
            ),
        },
    }
    manifest_bytes = canonical_json_bytes(manifest)
    destination = archive_root / "snapshots" / vintage_id
    if destination.exists():
        verified = verify_vintage(destination)
        if verified.get("capture_identity_sha256") == identity_hash:
            return CaptureResult(path=destination, manifest=verified, created=False)
        raise ArchiveIntegrityError(f"Conflicting existing destination {vintage_id}.")

    _write_completed_directory(
        parent=archive_root / "snapshots",
        destination=destination,
        members={
            payload_name: raw_payload,
            "mapping.json": mapping_bytes,
            "protocol.json": protocol_bytes,
            "manifest.json": manifest_bytes,
        },
        checksum_member="manifest.sha256",
        checksum_target="manifest.json",
    )
    return CaptureResult(path=destination, manifest=verify_vintage(destination), created=True)


def _declared_outcome_ids(evaluation: Mapping[str, Any]) -> set[str]:
    outcomes = evaluation.get("outcomes", [])
    if not isinstance(outcomes, Sequence) or isinstance(outcomes, (str, bytes)):
        return set()
    result: set[str] = set()
    for item in outcomes:
        if isinstance(item, Mapping) and item.get("id"):
            result.add(str(item["id"]))
        elif isinstance(item, str):
            result.add(item)
    return result


def _score_directories(archive_root: Path) -> list[Path]:
    root = archive_root / "scores"
    if not root.exists():
        return []
    return sorted(
        path
        for path in root.iterdir()
        if path.is_dir() and not path.name.startswith(".")
    )


def _verify_score(path: Path) -> dict[str, Any]:
    record_path = path / "score.json"
    checksum_path = path / "score.sha256"
    record = _read_json(record_path)
    try:
        expected = checksum_path.read_text(encoding="ascii").split()[0]
    except (OSError, IndexError) as exc:
        raise ArchiveIntegrityError(f"Missing score checksum for {path.name}.") from exc
    if sha256_file(record_path) != expected:
        raise ArchiveIntegrityError(f"Score checksum mismatch for {path.name}.")
    if record.get("score_id") != path.name:
        raise ArchiveIntegrityError(f"Score ID/path mismatch for {path.name}.")
    return record


def verify_scores(archive_root: Path) -> list[dict[str, Any]]:
    return [_verify_score(path) for path in _score_directories(archive_root)]


def append_score(
    *,
    archive_root: Path,
    vintage_id: str,
    outcome_id: str,
    horizon_quarters: int,
    target_period: str,
    benchmark_forecast: float,
    candidate_forecast: float,
    realized_value: float,
    realization_release_timestamp: str,
    realization_source_url: str,
    realization_payload_sha256: str,
    recorded_at: str,
    code_state: Mapping[str, Any],
) -> ScoreResult:
    """Append a squared-loss score against a frozen prospective vintage.

    The values must already be on the scale declared by the protocol.  Seed
    vintages are rejected mechanically.
    """

    vintage_path = archive_root / "snapshots" / vintage_id
    manifest = verify_vintage(vintage_path)
    if not manifest.get("eligible_for_prospective_scoring", False):
        raise ProspectiveEligibilityError(
            "Protocol baseline/current-vintage seeds cannot be scored as "
            "prospective evidence."
        )
    evaluation = manifest.get("evaluation", {})
    if outcome_id not in _declared_outcome_ids(evaluation):
        raise ProspectiveEligibilityError(
            f"Outcome {outcome_id!r} was not declared in the frozen protocol."
        )
    declared_horizons = {int(value) for value in evaluation.get("horizons_quarters", [])}
    if int(horizon_quarters) not in declared_horizons:
        raise ProspectiveEligibilityError(
            f"Horizon {horizon_quarters} was not declared in the frozen protocol."
        )
    loss = evaluation.get("loss", {})
    if not isinstance(loss, Mapping):
        raise ProspectiveEligibilityError("The frozen protocol has no valid loss rule.")
    outcome_loss = str(
        loss.get("by_outcome", {}).get(outcome_id, loss.get("id", ""))
        if isinstance(loss.get("by_outcome", {}), Mapping)
        else loss.get("id", "")
    )
    if outcome_loss not in {"squared_error", "brier"}:
        raise ProspectiveEligibilityError(
            "Automatic scoring supports only declared squared_error or brier loss."
        )
    numeric_values = {
        "benchmark_forecast": float(benchmark_forecast),
        "candidate_forecast": float(candidate_forecast),
        "realized_value": float(realized_value),
    }
    if not all(math.isfinite(value) for value in numeric_values.values()):
        raise VintageArchiveError("Forecast and realization values must be finite.")
    if outcome_loss == "brier":
        if numeric_values["realized_value"] not in {0.0, 1.0}:
            raise VintageArchiveError("A Brier realization must be exactly 0 or 1.")
        if not (
            0.0 <= numeric_values["benchmark_forecast"] <= 1.0
            and 0.0 <= numeric_values["candidate_forecast"] <= 1.0
        ):
            raise VintageArchiveError("Brier forecasts must be probabilities in [0, 1].")
    if not target_period.strip():
        raise VintageArchiveError("target_period must not be empty.")
    if not realization_source_url.strip():
        raise VintageArchiveError("realization_source_url must not be empty.")
    if not re.fullmatch(r"[0-9a-f]{64}", realization_payload_sha256):
        raise VintageArchiveError(
            "realization_payload_sha256 must be a lowercase SHA-256 digest."
        )

    release_utc = normalize_utc(
        realization_release_timestamp,
        field="realization_release_timestamp",
    )
    recorded_utc = normalize_utc(recorded_at, field="recorded_at")
    information_release = _parse_utc(
        str(manifest["release"]["timestamp_utc"]),
        field="vintage release timestamp",
    )
    realization_release = _parse_utc(
        release_utc,
        field="realization_release_timestamp",
    )
    if realization_release <= information_release:
        raise VintageArchiveError(
            "The realization release must follow the forecast information vintage."
        )
    if _parse_utc(recorded_utc, field="recorded_at") < realization_release:
        raise VintageArchiveError("recorded_at cannot precede realization release.")

    baseline_loss = (numeric_values["benchmark_forecast"] - numeric_values["realized_value"]) ** 2
    candidate_loss = (numeric_values["candidate_forecast"] - numeric_values["realized_value"]) ** 2
    score_key = {
        "vintage_id": vintage_id,
        "outcome_id": outcome_id,
        "horizon_quarters": int(horizon_quarters),
        "target_period": target_period,
    }
    content = {
        **score_key,
        **numeric_values,
        "realization_release_timestamp_utc": release_utc,
        "realization_source_url": realization_source_url,
        "realization_payload_sha256": realization_payload_sha256,
    }
    content_hash = sha256_bytes(canonical_json_bytes(content))
    prior_records: list[dict[str, Any]] = []
    for path in _score_directories(archive_root):
        prior = _verify_score(path)
        if all(prior.get(key) == value for key, value in score_key.items()):
            prior_records.append(prior)
            if prior.get("score_content_sha256") == content_hash:
                return ScoreResult(path=path, record=prior, created=False)

    score_id = "score-" + content_hash[:24]
    record = {
        "schema_version": ARCHIVE_SCHEMA_VERSION,
        "record_type": "prospective_score",
        "score_id": score_id,
        "score_content_sha256": content_hash,
        **score_key,
        "forecast_information_set": {
            "vintage_id": vintage_id,
            "release_timestamp_utc": manifest["release"]["timestamp_utc"],
            "raw_payload_sha256": manifest["raw_payload"]["sha256"],
        },
        "outcome_realization": {
            "value": numeric_values["realized_value"],
            "release_timestamp_utc": release_utc,
            "source_url": realization_source_url,
            "raw_payload_sha256": realization_payload_sha256,
        },
        "forecast_values": {
            "benchmark": numeric_values["benchmark_forecast"],
            "candidate": numeric_values["candidate_forecast"],
            "scale": loss.get("scale", ""),
        },
        "loss": {
            "id": outcome_loss,
            "benchmark": baseline_loss,
            "candidate": candidate_loss,
            "candidate_minus_benchmark": candidate_loss - baseline_loss,
        },
        "recorded_at_utc": recorded_utc,
        "code": dict(code_state),
        "revision": {
            "sequence_for_target": len(prior_records) + 1,
            "prior_score_ids_for_target": [
                str(item["score_id"]) for item in prior_records
            ],
            "policy": evaluation.get("revision_policy"),
        },
    }
    score_bytes = canonical_json_bytes(record)
    destination = archive_root / "scores" / score_id
    if destination.exists():
        verified = _verify_score(destination)
        if verified.get("score_content_sha256") == content_hash:
            return ScoreResult(path=destination, record=verified, created=False)
        raise ArchiveIntegrityError(f"Conflicting existing score {score_id}.")
    _write_completed_directory(
        parent=archive_root / "scores",
        destination=destination,
        members={"score.json": score_bytes},
        checksum_member="score.sha256",
        checksum_target="score.json",
    )
    return ScoreResult(path=destination, record=_verify_score(destination), created=True)
