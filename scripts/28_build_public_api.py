#!/usr/bin/env python3
"""Build the versioned static JSON API and LLM index."""
from __future__ import annotations

import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from lib.public_api import build_public_api


def main() -> int:
    outputs = build_public_api(ROOT)
    for path in outputs:
        print(f"Wrote {path.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
