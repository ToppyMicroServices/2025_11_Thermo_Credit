#!/usr/bin/env python3
"""Build all regional indicator panels with the shared implementation."""

from __future__ import annotations

import importlib.util
from pathlib import Path
from typing import Callable


ROOT = Path(__file__).resolve().parents[1]
CORE_SCRIPT = ROOT / "scripts" / "02_compute_indicators.py"
REGIONS = ("jp", "eu", "us")


def _load_compute_region() -> Callable[[str], str]:
    spec = importlib.util.spec_from_file_location("thermo_credit_compute_indicators", CORE_SCRIPT)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load {CORE_SCRIPT}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    compute_region = getattr(module, "compute_region", None)
    if not callable(compute_region):
        raise RuntimeError("compute_region() not found in scripts/02_compute_indicators.py")
    return compute_region


def main() -> int:
    (ROOT / "site").mkdir(parents=True, exist_ok=True)
    compute_region = _load_compute_region()
    outputs = [compute_region(region) for region in REGIONS]
    print("Wrote:", ", ".join(outputs))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
