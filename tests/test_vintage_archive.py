from __future__ import annotations

import json
from pathlib import Path

import pytest

from lib.vintage_archive import (
    ArchiveIntegrityError,
    PROSPECTIVE_CAPTURE_CLASS,
    SEED_CAPTURE_CLASS,
    ProspectiveEligibilityError,
    VintageArchiveError,
    append_score,
    canonical_json_bytes,
    capture_vintage,
    verify_archive,
    verify_scores,
)


def _mapping() -> dict[str, object]:
    return {
        "db": "LA01",
        "bucket_mapping": [
            {
                "bucket": "legacy_G",
                "sector": "manufacturing",
                "stock_code": "SERIES_NFB",
                "primary_bucket": "NFB",
            },
            {
                "bucket": "legacy_E",
                "sector": "finance",
                "stock_code": "SERIES_FIN",
                "primary_bucket": "FIN",
            },
            {
                "bucket": "legacy_E",
                "sector": "real_estate",
                "stock_code": "SERIES_PROP",
                "primary_bucket": "PROP",
            },
            {
                "bucket": "legacy_E",
                "sector": "household_nonhousing",
                "stock_code": "SERIES_HH_NONHOUSING",
                "primary_bucket": "HH_NONHOUSING",
            },
        ],
        "construction": "test frozen published four-way borrower mapping",
        "primary_taxonomy_id": "bezemer_samarina_zhang_2020_japan_v1",
        "primary_vector_columns": [
            "borrower_composition_NFB_4q",
            "borrower_composition_FIN_4q",
            "borrower_composition_PROP_4q",
            "borrower_composition_HH_NONHOUSING_4q",
        ],
        "primary_scale_stock_column": "primary_included_stock",
        "taxonomy_selection": {
            "primary_taxonomy_id": "bezemer_samarina_zhang_2020_japan_v1",
            "robustness_taxonomy_ids": [
                "werner_1997_financial_circulation_v1",
                "muller_verner_2024_sector_v1",
            ],
            "legacy_taxonomy_id": "author_gbe_legacy_v1",
            "outcome_columns_used": [],
            "oos_results_used": False,
        },
        "taxonomies": {
            "bezemer_samarina_zhang_2020_japan_v1": {"role": "primary"},
            "werner_1997_financial_circulation_v1": {
                "role": "pre_specified_robustness"
            },
            "muller_verner_2024_sector_v1": {
                "role": "pre_specified_robustness"
            },
            "author_gbe_legacy_v1": {"role": "appendix_legacy"},
        },
        "legacy_aliases": {"legacy_q_t": "borrower_composition_G_4q"},
        "common_taxonomy_stock_start": "2009-06-30",
        "first_valid_common_taxonomy_flow": "2009-09-30",
        "series_metadata": {
            "SERIES_NFB": {"last_update": 20260820},
            "SERIES_FIN": {"last_update": 20260820},
            "SERIES_PROP": {"last_update": 20260820},
            "SERIES_HH_NONHOUSING": {"last_update": 20260820},
        },
    }


def _protocol(registration: str | None = None) -> dict[str, object]:
    return {
        "schema_version": "1.0",
        "protocol_id": "test-boj-protocol",
        "protocol_version": "1.0.0",
        "registration_timestamp_utc": registration,
        "mapping": {
            "version": "test-bezemer-four-way-v1",
            "primary_taxonomy_id": "bezemer_samarina_zhang_2020_japan_v1",
            "primary_vector_columns": [
                "borrower_composition_NFB_4q",
                "borrower_composition_FIN_4q",
                "borrower_composition_PROP_4q",
                "borrower_composition_HH_NONHOUSING_4q",
            ],
            "primary_scale_stock_column": "primary_included_stock",
            "taxonomy_selection": {
                "primary_taxonomy_id": "bezemer_samarina_zhang_2020_japan_v1",
                "robustness_taxonomy_ids": [
                    "werner_1997_financial_circulation_v1",
                    "muller_verner_2024_sector_v1",
                ],
                "legacy_taxonomy_id": "author_gbe_legacy_v1",
                "outcome_columns_used": [],
                "oos_results_used": False,
            },
        },
        "source": {
            "name": "Bank of Japan Time-Series Data Search API",
            "api_endpoint": "https://example.test/boj",
            "database": "LA01",
        },
        "evaluation": {
            "outcomes": [
                {"id": "jgb_yield_widening_event", "loss": "brier"},
                {"id": "nominal_gdp_growth", "loss": "squared_error"},
            ],
            "horizons_quarters": [4, 8],
            "benchmark": {"id": "primary_included_stock_growth"},
            "candidate": {"id": "matched_stock_plus_composition"},
            "loss": {
                "id": "outcome_specific_squared_loss",
                "by_outcome": {
                    "jgb_yield_widening_event": "brier",
                    "nominal_gdp_growth": "squared_error",
                },
                "scale": "frozen test scale",
            },
            "revision_policy": {
                "predictor": "first capture primary",
                "outcome": "first release primary",
            },
        },
    }


def _code_state() -> dict[str, object]:
    return {
        "git_commit": "a" * 40,
        "worktree_dirty": False,
        "capture_file_sha256": {"capture.py": "b" * 64},
    }


def _capture(
    archive: Path,
    *,
    payload: bytes = b'{"STATUS":200,"value":1}',
    capture_class: str = SEED_CAPTURE_CLASS,
    registration: str | None = None,
    release: str = "2026-08-20T00:00:00Z",
) -> object:
    mapping = _mapping()
    return capture_vintage(
        archive_root=archive,
        raw_payload=payload,
        protocol=_protocol(registration),
        mapping_metadata=mapping,
        mapping_metadata_bytes=canonical_json_bytes(mapping),
        capture_class=capture_class,
        release_timestamp=release,
        retrieved_at="2026-08-20T00:05:00Z",
        release_timestamp_source="publisher calendar",
        release_timestamp_precision="minute",
        source_url="https://example.test/boj?db=LA01&code=SERIES_NFB",
        media_type="application/json",
        content_encoding=None,
        capture_method="test_fixture",
        code_state=_code_state(),
    )


def test_seed_manifest_is_explicitly_nonprospective_and_complete(tmp_path: Path) -> None:
    result = _capture(tmp_path)

    assert result.created
    manifest = result.manifest
    assert manifest["capture_class"] == SEED_CAPTURE_CLASS
    assert manifest["eligible_for_prospective_scoring"] is False
    assert "not preregistered" in manifest["evidence_boundary"]
    assert manifest["source"]["series_ids"] == [
        "SERIES_FIN",
        "SERIES_HH_NONHOUSING",
        "SERIES_NFB",
        "SERIES_PROP",
    ]
    assert manifest["source"]["url"].startswith("https://example.test/boj")
    assert manifest["release"] == {
        "timestamp_utc": "2026-08-20T00:00:00Z",
        "timestamp_source": "publisher calendar",
        "timestamp_precision": "minute",
        "retrieved_at_utc": "2026-08-20T00:05:00Z",
    }
    assert manifest["mapping"]["version"] == "test-bezemer-four-way-v1"
    assert manifest["mapping"]["primary_taxonomy_id"] == (
        "bezemer_samarina_zhang_2020_japan_v1"
    )
    assert manifest["mapping"]["primary_vector_columns"] == [
        "borrower_composition_NFB_4q",
        "borrower_composition_FIN_4q",
        "borrower_composition_PROP_4q",
        "borrower_composition_HH_NONHOUSING_4q",
    ]
    assert manifest["mapping"]["primary_scale_stock_column"] == (
        "primary_included_stock"
    )
    assert manifest["mapping"]["taxonomy_selection"]["oos_results_used"] is False
    assert manifest["mapping"]["taxonomy_ids"] == [
        "author_gbe_legacy_v1",
        "bezemer_samarina_zhang_2020_japan_v1",
        "muller_verner_2024_sector_v1",
        "werner_1997_financial_circulation_v1",
    ]
    assert manifest["code"]["git_commit"] == "a" * 40
    assert manifest["evaluation"]["horizons_quarters"] == [4, 8]
    assert manifest["evaluation"]["benchmark"]["id"] == (
        "primary_included_stock_growth"
    )
    assert manifest["evaluation"]["loss"]["by_outcome"]["jgb_yield_widening_event"] == "brier"
    assert len(manifest["raw_payload"]["sha256"]) == 64
    assert verify_archive(tmp_path) == [manifest]
    serialized = json.dumps(manifest)
    assert str(tmp_path) not in serialized
    assert "/Users/" not in serialized
    assert "akira" not in serialized.lower()
    assert "author_name" not in serialized.lower()


def test_duplicate_capture_is_idempotent(tmp_path: Path) -> None:
    first = _capture(tmp_path)
    second = _capture(tmp_path)

    assert first.created
    assert not second.created
    assert second.path == first.path
    assert len(verify_archive(tmp_path)) == 1


def test_changed_content_at_same_release_creates_linked_vintage(tmp_path: Path) -> None:
    first = _capture(tmp_path, payload=b'{"STATUS":200,"value":1}')
    second = _capture(tmp_path, payload=b'{"STATUS":200,"value":2}')

    assert first.path != second.path
    assert second.created
    assert second.manifest["revision"]["sequence_for_release"] == 2
    assert second.manifest["revision"]["raw_payload_changed_from_prior_capture"] is True
    assert second.manifest["revision"]["prior_vintage_ids_for_release"] == [
        first.manifest["vintage_id"]
    ]
    assert len(verify_archive(tmp_path)) == 2


def test_prospective_capture_requires_prior_registration(tmp_path: Path) -> None:
    with pytest.raises(ProspectiveEligibilityError, match="disabled"):
        _capture(
            tmp_path,
            capture_class=PROSPECTIVE_CAPTURE_CLASS,
            registration=None,
        )

    with pytest.raises(ProspectiveEligibilityError, match="strictly later"):
        _capture(
            tmp_path,
            capture_class=PROSPECTIVE_CAPTURE_CLASS,
            registration="2026-08-20T00:00:00Z",
            release="2026-08-20T00:00:00Z",
        )


def test_protocol_cannot_silently_switch_primary_taxonomy(tmp_path: Path) -> None:
    mapping = _mapping()
    protocol = _protocol()
    protocol["mapping"]["primary_taxonomy_id"] = "post_hoc_mapping"

    with pytest.raises(
        VintageArchiveError,
        match=r"mapping\.primary_taxonomy_id does not match",
    ):
        capture_vintage(
            archive_root=tmp_path,
            raw_payload=b'{"STATUS":200}',
            protocol=protocol,
            mapping_metadata=mapping,
            mapping_metadata_bytes=canonical_json_bytes(mapping),
            capture_class=SEED_CAPTURE_CLASS,
            release_timestamp="2026-08-20T00:00:00Z",
            retrieved_at="2026-08-20T00:05:00Z",
            release_timestamp_source="publisher calendar",
            release_timestamp_precision="minute",
            source_url="https://example.test/boj",
            media_type="application/json",
            content_encoding=None,
            capture_method="test_fixture",
            code_state=_code_state(),
        )


def test_future_postregistration_capture_is_score_eligible(tmp_path: Path) -> None:
    result = _capture(
        tmp_path,
        capture_class=PROSPECTIVE_CAPTURE_CLASS,
        registration="2026-08-19T00:00:00Z",
    )

    assert result.manifest["eligible_for_prospective_scoring"] is True
    assert result.manifest["protocol"]["registration_timestamp_utc"] == (
        "2026-08-19T00:00:00Z"
    )


def test_seed_is_rejected_by_score_writer(tmp_path: Path) -> None:
    seed = _capture(tmp_path)

    with pytest.raises(ProspectiveEligibilityError, match="cannot be scored"):
        append_score(
            archive_root=tmp_path,
            vintage_id=seed.manifest["vintage_id"],
            outcome_id="jgb_yield_widening_event",
            horizon_quarters=4,
            target_period="2027Q2",
            benchmark_forecast=0.4,
            candidate_forecast=0.3,
            realized_value=0.0,
            realization_release_timestamp="2027-08-20T00:00:00Z",
            realization_source_url="https://example.test/outcome",
            realization_payload_sha256="c" * 64,
            recorded_at="2027-08-20T00:05:00Z",
            code_state=_code_state(),
        )


def test_score_is_append_only_idempotent_and_revision_aware(tmp_path: Path) -> None:
    vintage = _capture(
        tmp_path,
        capture_class=PROSPECTIVE_CAPTURE_CLASS,
        registration="2026-08-19T00:00:00Z",
    )
    kwargs = {
        "archive_root": tmp_path,
        "vintage_id": vintage.manifest["vintage_id"],
        "outcome_id": "jgb_yield_widening_event",
        "horizon_quarters": 4,
        "target_period": "2027Q2",
        "benchmark_forecast": 0.6,
        "candidate_forecast": 0.4,
        "realized_value": 0.0,
        "realization_release_timestamp": "2027-08-20T00:00:00Z",
        "realization_source_url": "https://example.test/outcome",
        "realization_payload_sha256": "c" * 64,
        "recorded_at": "2027-08-20T00:05:00Z",
        "code_state": _code_state(),
    }

    first = append_score(**kwargs)
    duplicate = append_score(**kwargs)
    revised = append_score(
        **{
            **kwargs,
            "realized_value": 1.0,
            "realization_payload_sha256": "d" * 64,
            "recorded_at": "2027-09-20T00:05:00Z",
        }
    )

    assert first.created
    assert not duplicate.created
    assert duplicate.path == first.path
    assert first.record["loss"]["id"] == "brier"
    assert first.record["loss"]["candidate_minus_benchmark"] == pytest.approx(-0.2)
    assert revised.record["revision"]["sequence_for_target"] == 2
    assert revised.record["revision"]["prior_score_ids_for_target"] == [
        first.record["score_id"]
    ]
    assert len(verify_scores(tmp_path)) == 2


def test_tampering_is_detected_before_reuse(tmp_path: Path) -> None:
    result = _capture(tmp_path)
    raw_name = result.manifest["raw_payload"]["file"]
    (result.path / raw_name).write_bytes(b"tampered")

    with pytest.raises(ArchiveIntegrityError, match="Checksum mismatch"):
        verify_archive(tmp_path)
    with pytest.raises(ArchiveIntegrityError, match="Checksum mismatch"):
        _capture(tmp_path)
