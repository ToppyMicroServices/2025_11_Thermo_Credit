#!/usr/bin/env python3
"""Call a Thermo Credit tool and print structured JSON."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from lib.thermo_credit_tools import build_repo_compute_payload, run_tool


def _payload(args: argparse.Namespace) -> dict[str, Any]:
    if args.repo_region:
        if args.tool != "compute_thermo_credit_metrics":
            raise SystemExit("--repo-region is available only for compute_thermo_credit_metrics")
        return build_repo_compute_payload(args.repo_region, limit=args.limit)
    if args.input:
        return json.loads(Path(args.input).read_text(encoding="utf-8"))
    if not sys.stdin.isatty():
        raw = sys.stdin.read().strip()
        return json.loads(raw) if raw else {}
    return {}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("tool")
    parser.add_argument("--input", help="JSON input path")
    parser.add_argument("--repo-region", choices=["jp", "eu", "us"])
    parser.add_argument("--limit", type=int)
    args = parser.parse_args()
    result = run_tool(args.tool, _payload(args))
    print(json.dumps(result, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
