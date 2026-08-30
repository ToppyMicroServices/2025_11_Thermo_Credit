from __future__ import annotations

from lib.boj_credit_taxonomies import (
    MAPPING_VERSION,
    MULLER_VERNER_TAXONOMY_ID,
    PRIMARY_TAXONOMY_ID,
    SECTOR_MAP,
    TAXONOMY_METADATA,
    TAXONOMY_SELECTION_RULE,
    WERNER_TAXONOMY_ID,
)


def test_mapping_version_is_explicit() -> None:
    assert MAPPING_VERSION == "boj-bezemer-four-way-common-population-2026-07-30-v1"


def _sector(name: str) -> dict[str, str]:
    return next(row for row in SECTOR_MAP if row["sector"] == name)


def test_taxonomies_are_fixed_without_outcome_or_oos_selection() -> None:
    assert TAXONOMY_SELECTION_RULE["primary_taxonomy_id"] == PRIMARY_TAXONOMY_ID
    assert TAXONOMY_SELECTION_RULE["robustness_taxonomy_ids"] == [
        WERNER_TAXONOMY_ID,
        MULLER_VERNER_TAXONOMY_ID,
    ]
    assert TAXONOMY_SELECTION_RULE["outcome_columns_used"] == []
    assert TAXONOMY_SELECTION_RULE["oos_results_used"] is False


def test_construction_placements_follow_each_cited_crosswalk() -> None:
    construction = _sector("construction")

    assert construction["primary_bucket"] == "NFB"
    assert construction["werner_bucket"] == "FCP"
    assert construction["muller_verner_bucket"] == "NONTRADABLE"
    assert "Construction is an NFB detail" in TAXONOMY_METADATA[
        PRIMARY_TAXONOMY_ID
    ]["construction_placement"]
    assert "financial-circulation proxy" in TAXONOMY_METADATA[
        WERNER_TAXONOMY_ID
    ]["construction_placement"]
    assert "Construction is nontradable" in TAXONOMY_METADATA[
        MULLER_VERNER_TAXONOMY_ID
    ]["construction_placement"]


def test_published_residual_components_are_explicit() -> None:
    assert _sector("other_organizations")["primary_bucket"] == "NFB_RESIDUAL_COMPONENT"
    assert (
        _sector("overseas_yen_and_transferred_loans")["primary_bucket"]
        == "NFB_RESIDUAL_COMPONENT"
    )
    assert _sector("local_governments")["primary_bucket"] == "EXCLUDE_LOCAL_GOVERNMENT"
    assert (
        _sector("overseas_yen_and_transferred_loans")["muller_verner_bucket"]
        == "UNRESOLVED"
    )
    assert _sector("other_organizations")["werner_bucket"] == "COMPLEMENT"
    assert (
        _sector("overseas_yen_and_transferred_loans")["werner_bucket"]
        == "COMPLEMENT"
    )
