from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parent.parent
CONCEPT_DOI = "10.5281/zenodo.17563220"
VERSION_DOI = "10.5281/zenodo.22177533"


def test_citation_file_identifies_current_release_and_concept():
    citation = yaml.safe_load((ROOT / "CITATION.cff").read_text(encoding="utf-8"))

    assert citation["cff-version"] == "1.2.0"
    assert citation["version"] == "2.2.1"
    assert citation["doi"] == VERSION_DOI
    assert CONCEPT_DOI in citation["message"]
    assert citation["preferred-citation"]["doi"] == VERSION_DOI
    assert citation["license"] == "Apache-2.0"


def test_readmes_point_to_the_same_current_dois():
    for filename in ("README.md", "README_JP.md"):
        text = (ROOT / filename).read_text(encoding="utf-8")
        assert CONCEPT_DOI in text
        assert VERSION_DOI in text
        assert "10.5281/zenodo.17778342" not in text


def test_public_entrypoints_are_present_in_both_readmes():
    expected = (
        "https://toppymicros.com/2025_11_Thermo_Credit/",
        "releases/latest/download/theory.pdf",
        "docs/identification_strategy.md",
    )
    for filename in ("README.md", "README_JP.md"):
        text = (ROOT / filename).read_text(encoding="utf-8")
        for target in expected:
            assert target in text
