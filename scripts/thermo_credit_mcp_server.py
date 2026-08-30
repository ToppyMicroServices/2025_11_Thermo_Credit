#!/usr/bin/env python3
"""Run the Thermo Credit MCP server over stdio or Streamable HTTP."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from lib.thermo_credit_mcp import create_mcp_server


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--transport",
        choices=["stdio", "streamable-http"],
        default="stdio",
        help="Use stdio locally or Streamable HTTP for a separately secured deployment.",
    )
    args = parser.parse_args()
    server = create_mcp_server()
    if args.transport == "streamable-http":
        server.run(
            transport="streamable-http",
            stateless_http=True,
            json_response=True,
        )
    else:
        server.run(transport="stdio")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
