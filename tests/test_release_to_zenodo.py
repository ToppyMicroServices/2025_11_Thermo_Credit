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
