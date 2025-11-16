"""Compatibility wrapper invoking the unified indicator builder for US data."""
from __future__ import annotations

import importlib.util
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
CORE_SCRIPT = ROOT / "scripts" / "02_compute_indicators.py"


def _load_core_module():
    spec = importlib.util.spec_from_file_location("compute_core", CORE_SCRIPT)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load core script from {CORE_SCRIPT}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)  # type: ignore[attr-defined]
    return module


def main() -> None:
    module = _load_core_module()
    compute_region = getattr(module, "compute_region", None)
    if not callable(compute_region):
        raise RuntimeError("compute_region() not found in 02_compute_indicators.py")
    compute_region("us")


if __name__ == "__main__":
    main()