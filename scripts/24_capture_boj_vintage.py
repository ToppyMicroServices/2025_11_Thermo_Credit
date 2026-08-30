#!/usr/bin/env python3
"""Capture an immutable BOJ release vintage under the prospective protocol."""

from __future__ import annotations

import argparse
import gzip
import json
import sys
import urllib.parse
import urllib.request
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from lib.vintage_archive import (  # noqa: E402
    PROSPECTIVE_CAPTURE_CLASS,
    SEED_CAPTURE_CLASS,
    VintageArchiveError,
    capture_vintage,
    git_code_state,
)


def _load_json_object(path: Path, *, label: str) -> tuple[dict[str, Any], bytes]:
    try:
        payload = path.read_bytes()
        value = json.loads(payload.decode("utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise VintageArchiveError(f"{label} must be a readable UTF-8 JSON file.") from exc
    if not isinstance(value, dict):
        raise VintageArchiveError(f"{label} must contain a JSON object.")
    return value, payload


def _series_ids(metadata: Mapping[str, Any]) -> list[str]:
    series_metadata = metadata.get("series_metadata")
    if not isinstance(series_metadata, Mapping) or not series_metadata:
        raise VintageArchiveError(
            "Mapping metadata must contain non-empty series_metadata."
        )
    return sorted(str(code) for code in series_metadata)


def _request_url(
    protocol: Mapping[str, Any],
    metadata: Mapping[str, Any],
    *,
    end_date: str | None,
) -> str:
    source = protocol.get("source", {})
    endpoint = str(source.get("api_endpoint", "")).strip()
    if not endpoint:
        raise VintageArchiveError("Protocol source.api_endpoint must not be empty.")
    resolved_end = end_date or str(metadata.get("end_date", "")).strip()
    if not resolved_end:
        raise VintageArchiveError(
            "An end date is required (CLI --end-date or mapping metadata end_date)."
        )
    params = {
        "format": str(source.get("format", "json")),
        "lang": str(source.get("language", "en")),
        "db": str(source.get("database", metadata.get("db", ""))),
        "startDate": str(source.get("start_date", metadata.get("start_date", "197701"))),
        "endDate": resolved_end,
        "code": ",".join(_series_ids(metadata)),
    }
    return endpoint + "?" + urllib.parse.urlencode(params)


def _validate_boj_response(
    payload: bytes,
    *,
    content_encoding: str | None,
) -> None:
    decoded = payload
    if content_encoding and content_encoding.lower() == "gzip":
        try:
            decoded = gzip.decompress(payload)
        except OSError as exc:
            raise VintageArchiveError("BOJ response claimed gzip but was invalid.") from exc
    try:
        value = json.loads(decoded.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise VintageArchiveError("BOJ API did not return valid UTF-8 JSON.") from exc
    if not isinstance(value, Mapping) or int(value.get("STATUS", 500)) != 200:
        raise VintageArchiveError(f"BOJ API returned a non-success payload: {value!r}")


def _fetch_payload(url: str) -> tuple[bytes, str, str, str | None]:
    request = urllib.request.Request(
        url,
        headers={
            "Accept": "application/json",
            "Accept-Encoding": "identity",
            "User-Agent": "anonymous-release-vintage-archive/1.0",
        },
    )
    try:
        with urllib.request.urlopen(request, timeout=90) as response:
            payload = response.read()
            final_url = response.geturl()
            media_type = response.headers.get_content_type()
            content_encoding = response.headers.get("Content-Encoding")
    except OSError as exc:
        raise VintageArchiveError("Unable to retrieve the BOJ API payload.") from exc
    _validate_boj_response(payload, content_encoding=content_encoding)
    return payload, final_url, media_type, content_encoding


def _infer_seed_release(metadata: Mapping[str, Any]) -> tuple[str, str, str]:
    series_metadata = metadata.get("series_metadata", {})
    last_updates: list[str] = []
    if isinstance(series_metadata, Mapping):
        for value in series_metadata.values():
            if isinstance(value, Mapping) and value.get("last_update") is not None:
                digits = re_digits(str(value["last_update"]))
                if len(digits) == 8:
                    last_updates.append(digits)
    if not last_updates:
        raise VintageArchiveError(
            "Seed release date could not be inferred; pass --release-timestamp."
        )
    latest = max(last_updates)
    timestamp = (
        f"{latest[:4]}-{latest[4:6]}-{latest[6:8]}T00:00:00Z"
    )
    return timestamp, "BOJ series_metadata LAST_UPDATE", "date"


def re_digits(value: str) -> str:
    return "".join(character for character in value if character.isdigit())


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Capture a BOJ current-vintage seed or post-registration release "
            "as an immutable archive record."
        )
    )
    parser.add_argument(
        "--protocol",
        default="prospective/protocol.json",
        help="Frozen prospective protocol JSON.",
    )
    parser.add_argument(
        "--mapping-metadata",
        default="data/credit_destination_jp_metadata.json",
        help="BOJ mapping metadata used to enumerate series and freeze the mapping.",
    )
    parser.add_argument(
        "--archive",
        default="prospective/archive",
        help="Append-only archive root.",
    )
    parser.add_argument(
        "--capture-class",
        required=True,
        choices=(SEED_CAPTURE_CLASS, PROSPECTIVE_CAPTURE_CLASS),
    )
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument(
        "--input",
        help="Exact local payload bytes to store (intended for the baseline seed).",
    )
    source.add_argument(
        "--fetch",
        action="store_true",
        help="Fetch the exact BOJ API response bytes.",
    )
    parser.add_argument(
        "--release-timestamp",
        help=(
            "Official release timestamp in ISO-8601 form. Required for prospective "
            "captures; a date-only LAST_UPDATE is inferred for a seed if omitted."
        ),
    )
    parser.add_argument(
        "--release-timestamp-source",
        default="publisher release timestamp",
    )
    parser.add_argument(
        "--release-timestamp-precision",
        default="second",
        choices=("date", "minute", "second", "microsecond", "observed_upper_bound"),
    )
    parser.add_argument(
        "--retrieved-at",
        help="Retrieval timestamp override for deterministic testing; defaults to now.",
    )
    parser.add_argument(
        "--source-url",
        help="Exact upstream URL for a local input; defaults to the reconstructed BOJ query.",
    )
    parser.add_argument(
        "--media-type",
        help="Media type for a local input; inferred from its suffix when omitted.",
    )
    parser.add_argument(
        "--end-date",
        help="BOJ API quarterly end date (YYYYQQ); defaults to metadata end_date.",
    )
    return parser


def _local_media_type(path: Path, override: str | None) -> str:
    if override:
        return override
    return {
        ".csv": "text/csv",
        ".json": "application/json",
        ".gz": "application/gzip",
    }.get(path.suffix.lower(), "application/octet-stream")


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    protocol_path = (ROOT / args.protocol).resolve()
    metadata_path = (ROOT / args.mapping_metadata).resolve()
    archive_path = (ROOT / args.archive).resolve()
    protocol, _ = _load_json_object(protocol_path, label="Protocol")
    metadata, metadata_bytes = _load_json_object(
        metadata_path,
        label="Mapping metadata",
    )

    release_timestamp = args.release_timestamp
    release_source = args.release_timestamp_source
    release_precision = args.release_timestamp_precision
    if not release_timestamp:
        if args.capture_class != SEED_CAPTURE_CLASS:
            raise VintageArchiveError(
                "--release-timestamp is required for prospective captures."
            )
        release_timestamp, release_source, release_precision = _infer_seed_release(
            metadata
        )

    reconstructed_url = _request_url(
        protocol,
        metadata,
        end_date=args.end_date,
    )
    if args.fetch:
        raw_payload, source_url, media_type, content_encoding = _fetch_payload(
            reconstructed_url
        )
        capture_method = "boj_api_transport_response"
    else:
        input_path = (ROOT / args.input).resolve()
        try:
            raw_payload = input_path.read_bytes()
        except OSError as exc:
            raise VintageArchiveError(f"Unable to read local input {input_path.name}.") from exc
        source_url = args.source_url or reconstructed_url
        media_type = _local_media_type(input_path, args.media_type)
        content_encoding = "gzip" if input_path.suffix.lower() == ".gz" else None
        capture_method = "local_current_vintage_seed"
        if args.capture_class != SEED_CAPTURE_CLASS:
            raise VintageArchiveError(
                "Prospective captures must use --fetch so the release response is frozen."
            )

    code_state = git_code_state(
        ROOT,
        code_files=(
            Path(__file__).resolve(),
            ROOT / "lib" / "vintage_archive.py",
        ),
    )
    result = capture_vintage(
        archive_root=archive_path,
        raw_payload=raw_payload,
        protocol=protocol,
        mapping_metadata=metadata,
        mapping_metadata_bytes=metadata_bytes,
        capture_class=args.capture_class,
        release_timestamp=release_timestamp,
        retrieved_at=args.retrieved_at or _utc_now(),
        release_timestamp_source=release_source,
        release_timestamp_precision=release_precision,
        source_url=source_url,
        media_type=media_type,
        content_encoding=content_encoding,
        capture_method=capture_method,
        code_state=code_state,
    )
    output = {
        "created": result.created,
        "vintage_id": result.manifest["vintage_id"],
        "capture_class": result.manifest["capture_class"],
        "eligible_for_prospective_scoring": result.manifest[
            "eligible_for_prospective_scoring"
        ],
        "archive_path": str(result.path.relative_to(ROOT)),
        "raw_payload_sha256": result.manifest["raw_payload"]["sha256"],
    }
    print(json.dumps(output, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main(sys.argv[1:]))
    except VintageArchiveError as exc:
        print(f"error: {exc}", file=sys.stderr)
        raise SystemExit(2)
