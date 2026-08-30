#!/usr/bin/env python3
"""Verify every immutable BOJ vintage and prospective score checksum."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from lib.vintage_archive import (  # noqa: E402
    ArchiveIntegrityError,
    verify_archive,
    verify_scores,
)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Verify checksums and record/path identities in a BOJ vintage archive."
    )
    parser.add_argument("--archive", default="prospective/archive")
    args = parser.parse_args(argv)
    archive = (ROOT / args.archive).resolve()
    vintages = verify_archive(archive)
    scores = verify_scores(archive)
    print(
        json.dumps(
            {
                "archive": str(archive.relative_to(ROOT)),
                "valid": True,
                "vintage_count": len(vintages),
                "prospective_vintage_count": sum(
                    bool(item.get("eligible_for_prospective_scoring"))
                    for item in vintages
                ),
                "seed_count": sum(
                    not bool(item.get("eligible_for_prospective_scoring"))
                    for item in vintages
                ),
                "score_count": len(scores),
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main(sys.argv[1:]))
    except ArchiveIntegrityError as exc:
        print(f"integrity error: {exc}", file=sys.stderr)
        raise SystemExit(2)
