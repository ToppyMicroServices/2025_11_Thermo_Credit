import importlib.util
from pathlib import Path

import pytest


SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "02_compute_all_regions.py"


def _load_module():
    spec = importlib.util.spec_from_file_location("compute_all_regions_for_tests", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_all_region_builder_calls_shared_builder_for_each_region(monkeypatch):
    module = _load_module()
    calls = []
    monkeypatch.setattr(module, "_load_compute_region", lambda: lambda region: calls.append(region) or f"{region}.csv")

    assert module.main() == 0
    assert calls == ["jp", "eu", "us"]


def test_all_region_builder_propagates_region_failure(monkeypatch):
    module = _load_module()

    def fail_on_eu(region):
        if region == "eu":
            raise RuntimeError("EU failed")
        return f"{region}.csv"

    monkeypatch.setattr(module, "_load_compute_region", lambda: fail_on_eu)
    with pytest.raises(RuntimeError, match="EU failed"):
        module.main()
