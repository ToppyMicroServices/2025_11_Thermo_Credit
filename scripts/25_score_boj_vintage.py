#!/usr/bin/env python3
"""Append a score for a frozen, prospectively eligible BOJ vintage."""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from lib.vintage_archive import (  # noqa: E402
    VintageArchiveError,
    append_score,
    git_code_state,
)


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Append the declared squared-error or Brier loss for forecasts "
            "based on one frozen prospective BOJ vintage."
        )
    )
    parser.add_argument("--archive", default="prospective/archive")
    parser.add_argument("--vintage-id", required=True)
    parser.add_argument("--outcome-id", required=True)
    parser.add_argument("--horizon-quarters", required=True, type=int)
    parser.add_argument("--target-period", required=True)
    parser.add_argument("--benchmark-forecast", required=True, type=float)
    parser.add_argument("--candidate-forecast", required=True, type=float)
    parser.add_argument("--realized-value", required=True, type=float)
    parser.add_argument("--realization-release-timestamp", required=True)
    parser.add_argument("--realization-source-url", required=True)
    parser.add_argument("--realization-payload-sha256", required=True)
    parser.add_argument(
        "--recorded-at",
        help="Timestamp override for deterministic testing; defaults to now.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    archive_path = (ROOT / args.archive).resolve()
    code_state = git_code_state(
        ROOT,
        code_files=(
            Path(__file__).resolve(),
            ROOT / "lib" / "vintage_archive.py",
        ),
    )
    result = append_score(
        archive_root=archive_path,
        vintage_id=args.vintage_id,
        outcome_id=args.outcome_id,
        horizon_quarters=args.horizon_quarters,
        target_period=args.target_period,
        benchmark_forecast=args.benchmark_forecast,
        candidate_forecast=args.candidate_forecast,
        realized_value=args.realized_value,
        realization_release_timestamp=args.realization_release_timestamp,
        realization_source_url=args.realization_source_url,
        realization_payload_sha256=args.realization_payload_sha256,
        recorded_at=args.recorded_at or _utc_now(),
        code_state=code_state,
    )
    print(
        json.dumps(
            {
                "created": result.created,
                "score_id": result.record["score_id"],
                "vintage_id": result.record["vintage_id"],
                "candidate_minus_benchmark": result.record["loss"][
                    "candidate_minus_benchmark"
                ],
                "archive_path": str(result.path.relative_to(ROOT)),
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main(sys.argv[1:]))
    except VintageArchiveError as exc:
        print(f"error: {exc}", file=sys.stderr)
        raise SystemExit(2)
