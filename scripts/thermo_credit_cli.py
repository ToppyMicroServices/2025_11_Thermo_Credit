#!/usr/bin/env python3
"""JSON CLI for the Thermo Credit tool surface."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from lib.thermo_credit_tools import build_repo_compute_payload, run_tool


def _load_payload(args: argparse.Namespace) -> Dict[str, Any]:
    if args.repo_region:
        if args.tool != "compute_thermo_credit_metrics":
            raise SystemExit("--repo-region is only supported for compute_thermo_credit_metrics")
        return build_repo_compute_payload(args.repo_region, limit=args.limit)
    if args.input:
        return json.loads(Path(args.input).read_text(encoding="utf-8"))
    if not sys.stdin.isatty():
        raw = sys.stdin.read().strip()
        return json.loads(raw) if raw else {}
    return {}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("tool", help="Thermo Credit tool name")
    parser.add_argument("--input", help="Path to a JSON payload")
    parser.add_argument(
        "--repo-region",
        choices=["jp", "eu", "us"],
        help="Build a compute_thermo_credit_metrics payload from site/indicators*.csv",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Limit repo-derived observations to the last N rows",
    )
    args = parser.parse_args()

    payload = _load_payload(args)
    result = run_tool(args.tool, payload)
    sys.stdout.write(json.dumps(result, indent=2, ensure_ascii=False))
    sys.stdout.write("\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
