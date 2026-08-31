import importlib.util
from pathlib import Path

import pytest
import requests


SCRIPT_PATH = Path(__file__).resolve().parent.parent / "scripts" / "release_to_zenodo.py"


def _load_module():
    spec = importlib.util.spec_from_file_location("release_to_zenodo_for_tests", SCRIPT_PATH)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_extract_public_concept_record_id_from_top_level_field():
    module = _load_module()
    record = {"id": 17778342, "conceptrecid": "17563220"}
    assert module._extract_public_concept_record_id(record) == "17563220"


def test_resource_deposition_id_prefers_links():
    module = _load_module()
    resource = {
        "id": 17778342,
        "links": {"self": "https://zenodo.org/api/deposit/depositions/18888888"},
    }
    assert module._resource_deposition_id(resource) == 18888888


def test_resource_deposition_id_falls_back_to_numeric_id():
    module = _load_module()
    assert module._resource_deposition_id({"id": "17778342"}) == 17778342


def test_extract_public_concept_record_id_from_parent_field():
    module = _load_module()
    record = {"id": 17778342, "parent": {"id": "17563220"}}
    assert module._extract_public_concept_record_id(record) == "17563220"


def test_resolve_concept_record_id_uses_public_records_endpoint(monkeypatch):
    module = _load_module()
    calls = []

    def fake_request_json(session, method, url, *, expected=None, **kwargs):
        calls.append((method, url, expected, kwargs))
        return {"id": 17778342, "conceptrecid": "17563220"}

    monkeypatch.setattr(module, "_request_json", fake_request_json)
    session = requests.Session()
    concept_record_id = module._resolve_concept_record_id(
        session,
        "https://zenodo.org/api",
        explicit_concept_record_id="",
        seed_record_id_or_doi="10.5281/zenodo.17778342",
    )

    assert concept_record_id == "17563220"
    assert calls == [
        ("GET", "https://zenodo.org/api/records/17778342", [200], {})
    ]


def test_resolve_concept_record_id_prefers_explicit_value():
    module = _load_module()
    session = requests.Session()
    assert (
        module._resolve_concept_record_id(
            session,
            "https://zenodo.org/api",
            explicit_concept_record_id="17563220",
            seed_record_id_or_doi="10.5281/zenodo.17778342",
        )
        == "17563220"
    )


def test_sanitize_metadata_for_update_drops_invalid_dates():
    module = _load_module()
    metadata = {
        "title": "Example",
        "dates": [
            {"type": "issued"},
            {"type": "accepted", "date": ""},
            {"type": "collected", "date": "2026-03-20"},
        ],
    }
    cleaned = module._sanitize_metadata_for_update(metadata)
    assert cleaned["dates"] == [{"type": "collected", "date": "2026-03-20"}]


def test_sanitize_metadata_for_update_removes_dates_when_all_invalid():
    module = _load_module()
    metadata = {"title": "Example", "dates": [{"type": "issued"}]}
    cleaned = module._sanitize_metadata_for_update(metadata)
    assert "dates" not in cleaned


def test_release_description_replaces_stale_version_text():
    module = _load_module()

    description = module._release_description(
        "release/v2.3.0",
        "https://github.com/ToppyMicroServices/2025_11_Thermo_Credit/releases/tag/release/v2.3.0",
    )

    assert "Version v2.3.0" in description
    assert "borrower-composition measure" in description
    assert "not loan purpose" in description
    assert "v2.1.7" not in description


def test_related_identifiers_replace_old_project_links_and_keep_external_ones():
    module = _load_module()
    external = {
        "identifier": "https://doi.org/10.1234/example",
        "relation": "cites",
    }
    old_release = {
        "identifier": (
            "https://github.com/ToppyMicroServices/2025_11_Thermo_Credit/"
            "releases/tag/release/v2.1.7"
        ),
        "relation": "isSupplementedBy",
    }

    related = module._merge_project_related_identifiers(
        [external, old_release],
        (
            "https://github.com/ToppyMicroServices/2025_11_Thermo_Credit/"
            "releases/tag/release/v2.3.0"
        ),
    )

    assert external in related
    assert old_release not in related
    assert any(item["identifier"].endswith("release/v2.3.0") for item in related)
    assert any(item["identifier"].endswith("2025_11_Thermo_Credit/") for item in related)


def test_update_metadata_writes_canonical_release_record(monkeypatch):
    module = _load_module()
    captured = {}

    def fake_request_json(session, method, url, *, expected=None, **kwargs):
        captured.update(kwargs["json"]["metadata"])
        return {"id": 18888888, "metadata": captured}

    monkeypatch.setattr(module, "_request_json", fake_request_json)
    draft = {
        "id": 18888888,
        "metadata": {
            "title": "Old title",
            "description": "Version 2.0 under development",
            "creators": [{"name": "Okutomi, Akira"}],
        },
    }

    module._update_metadata(
        requests.Session(),
        "https://zenodo.org/api",
        draft,
        tag="release/v2.3.0",
        release_url=(
            "https://github.com/ToppyMicroServices/2025_11_Thermo_Credit/"
            "releases/tag/release/v2.3.0"
        ),
    )

    assert captured["title"] == module.PAPER_TITLE
    assert captured["version"] == "v2.3.0"
    assert captured["creators"] == [{"name": "Okutomi, Akira"}]
    assert "under development" not in captured["description"]
    assert captured["language"] == "eng"
    assert len(captured["related_identifiers"]) == 3


def test_linked_latest_draft_ignores_same_id(monkeypatch):
    module = _load_module()
    latest = {
        "id": 17778342,
        "links": {"latest_draft": "https://zenodo.org/api/deposit/depositions/17778342"},
    }
    assert module._linked_latest_draft(requests.Session(), "https://zenodo.org/api", latest) is None


def test_linked_latest_draft_fetches_distinct_draft(monkeypatch):
    module = _load_module()
    calls = []

    def fake_request_json(session, method, url, *, expected=None, **kwargs):
        calls.append((method, url, expected, kwargs))
        return {"id": 18888888, "submitted": False}

    monkeypatch.setattr(module, "_request_json", fake_request_json)
    latest = {
        "id": 17778342,
        "links": {"latest_draft": "https://zenodo.org/api/deposit/depositions/18888888"},
    }
    draft = module._linked_latest_draft(requests.Session(), "https://zenodo.org/api", latest)
    assert draft == {"id": 18888888, "submitted": False}
    assert calls == [
        ("GET", "https://zenodo.org/api/deposit/depositions/18888888", [200], {})
    ]


def test_extract_public_concept_record_id_fails_cleanly():
    module = _load_module()
    with pytest.raises(SystemExit, match="did not expose a concept record id"):
        module._extract_public_concept_record_id({"id": 17778342})
