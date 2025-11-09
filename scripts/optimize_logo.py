"""Optimize the branding logo og-brand-clean.png into a smaller PNG.

Usage:
  python scripts/optimize_logo.py [--height 80] [--colors 128]

It will read scripts/og-brand-clean.png (or ./og-brand-clean.png) and write a
pre-compressed version to scripts/og-brand-clean.min.png. Report builder can
embed this file directly (base64) without re-running compression logic each time.

This is useful when running in CI to keep HTML size smaller and build faster.
"""
from __future__ import annotations

import argparse
import os
from io import BytesIO
import sys

try:
    from PIL import Image  # type: ignore
except Exception:  # pragma: no cover
    Image = None

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

SOURCE_CANDIDATES = [
    os.path.join(ROOT, "scripts", "og-brand-clean.png"),
    os.path.join(ROOT, "og-brand-clean.png"),
]
TARGET_PATH = os.path.join(ROOT, "scripts", "og-brand-clean.min.png")


def optimize(height: int, colors: int) -> int:
    if Image is None:
        print("Pillow not available; cannot optimize.", file=sys.stderr)
        return 1
    src = next((p for p in SOURCE_CANDIDATES if os.path.exists(p)), None)
    if not src:
        print("Source logo not found.", file=sys.stderr)
        return 2
    try:
        im = Image.open(src).convert("RGBA")
    except Exception as exc:
        print(f"Failed to open source: {exc}", file=sys.stderr)
        return 3
    w, h = im.size
    if h > height and h > 0:
        new_w = max(1, int(w * height / h))
        im = im.resize((new_w, height), Image.LANCZOS)
    # Quantize
    try:
        im_q = im.convert("P", palette=Image.ADAPTIVE, colors=colors)
    except Exception:
        im_q = im
    buf = BytesIO()
    try:
        im_q.save(buf, format="PNG", optimize=True, compress_level=9)
    except Exception:
        im.save(buf, format="PNG")
    data = buf.getvalue()
    with open(TARGET_PATH, "wb") as fp:
        fp.write(data)
    orig_size = os.path.getsize(src)
    new_size = len(data)
    ratio = (new_size / orig_size) if orig_size else 0
    print(f"Wrote {TARGET_PATH} ({new_size} bytes, {ratio:.2%} of original {orig_size} bytes)")
    return 0


def main(argv: list[str]) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--height", type=int, default=80, help="Target max height (px)")
    ap.add_argument("--colors", type=int, default=128, help="Palette size (1-256)")
    args = ap.parse_args(argv)
    if args.colors < 1 or args.colors > 256:
        print("--colors must be between 1 and 256", file=sys.stderr)
        return 1
    return optimize(args.height, args.colors)


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
