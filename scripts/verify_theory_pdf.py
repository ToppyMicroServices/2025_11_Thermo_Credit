#!/usr/bin/env python3
"""Validate and independently render the final theory PDF."""
from __future__ import annotations

import argparse
import hashlib
import json
import platform
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any, Sequence

from PIL import Image, ImageDraw


ROOT = Path(__file__).resolve().parents[1]
FORBIDDEN_QPDF_MARKERS = ("warning", "repaired", "incorrect xref", "recursive dict")


class VerificationError(RuntimeError):
    pass


def _run(command: Sequence[str]) -> dict[str, Any]:
    completed = subprocess.run(command, capture_output=True, text=True, check=False)
    output = "\n".join(part.strip() for part in (completed.stdout, completed.stderr) if part.strip())
    return {
        "command": list(command),
        "exit_code": completed.returncode,
        "output": output,
    }


def _require_tool(name: str) -> str:
    path = shutil.which(name)
    if not path:
        raise VerificationError(f"Required tool is unavailable: {name}")
    return path


def _version(command: Sequence[str]) -> str:
    result = _run(command)
    first_line = result["output"].splitlines()[0] if result["output"] else "unknown"
    return first_line.strip()


def _page_count(pdfinfo_output: str) -> int:
    match = re.search(r"^Pages:\s+(\d+)\s*$", pdfinfo_output, flags=re.MULTILINE)
    if not match:
        raise VerificationError("pdfinfo did not report a page count")
    return int(match.group(1))


def _prepare_directory(path: Path) -> None:
    if path.exists():
        shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)


def _inspect_rendered_pages(directory: Path, expected_pages: int) -> dict[str, Any]:
    paths = sorted(directory.glob("page-*.png"))
    if len(paths) != expected_pages:
        raise VerificationError(
            f"{directory.name} rendered {len(paths)} pages; expected {expected_pages}"
        )
    dimensions: list[list[int]] = []
    minimum_ink_fraction = 1.0
    for path in paths:
        with Image.open(path) as image:
            image.load()
            width, height = image.size
            if width < 600 or height < 700:
                raise VerificationError(f"Rendered page is unexpectedly small: {path} {image.size}")
            grayscale = image.convert("L")
            histogram = grayscale.histogram()
            total = width * height
            ink = sum(histogram[:245])
            ink_fraction = ink / total
            if ink_fraction < 0.001:
                raise VerificationError(f"Rendered page appears blank: {path}")
            dimensions.append([width, height])
            minimum_ink_fraction = min(minimum_ink_fraction, ink_fraction)
    return {
        "status": "pass",
        "page_count": len(paths),
        "dimensions": dimensions,
        "minimum_nonwhite_fraction": round(minimum_ink_fraction, 8),
        "files": [path.name for path in paths],
    }


def _write_contact_sheet(directory: Path, *, columns: int = 4) -> Path:
    paths = sorted(directory.glob("page-*.png"))
    if not paths:
        raise VerificationError(f"No rendered pages for contact sheet: {directory}")
    thumb_width = 245
    thumb_height = 317
    gutter = 16
    label_height = 22
    rows = (len(paths) + columns - 1) // columns
    sheet = Image.new(
        "RGB",
        (
            gutter + columns * (thumb_width + gutter),
            gutter + rows * (thumb_height + label_height + gutter),
        ),
        "#e8eef2",
    )
    draw = ImageDraw.Draw(sheet)
    for index, path in enumerate(paths):
        with Image.open(path) as page:
            thumbnail = page.convert("RGB")
            thumbnail.thumbnail((thumb_width, thumb_height))
        column = index % columns
        row = index // columns
        x = gutter + column * (thumb_width + gutter)
        y = gutter + row * (thumb_height + label_height + gutter)
        sheet.paste(thumbnail, (x, y))
        draw.text((x, y + thumb_height + 3), f"Page {index + 1}", fill="#02213b")
    path = directory / "contact-sheet.png"
    sheet.save(path)
    return path


def _render_pdfium(pdf_path: Path, output_dir: Path, dpi: int) -> tuple[str, dict[str, Any]]:
    try:
        import pypdfium2 as pdfium
    except ImportError as exc:
        raise VerificationError("pypdfium2 is required for PDFium rendering") from exc
    _prepare_directory(output_dir)
    document = pdfium.PdfDocument(str(pdf_path))
    try:
        for index in range(len(document)):
            page = document[index]
            try:
                bitmap = page.render(scale=dpi / 72.0)
                try:
                    bitmap.to_pil().save(output_dir / f"page-{index + 1:03d}.png")
                finally:
                    bitmap.close()
            finally:
                page.close()
    finally:
        document.close()
    version = getattr(pdfium, "__version__", None)
    if version is None:
        try:
            from importlib.metadata import version as package_version

            version = package_version("pypdfium2")
        except Exception:
            version = "unknown"
    return f"pypdfium2 {version}", {}


def verify_pdf(
    pdf_path: Path,
    output_dir: Path,
    *,
    dpi: int = 144,
    require_pdfkit: bool = False,
) -> dict[str, Any]:
    pdf_path = pdf_path.resolve()
    output_dir = output_dir.resolve()
    if not pdf_path.is_file() or pdf_path.stat().st_size == 0:
        raise VerificationError(f"PDF is missing or empty: {pdf_path}")
    _prepare_directory(output_dir)

    qpdf = _require_tool("qpdf")
    gs = _require_tool("gs")
    pdfinfo = _require_tool("pdfinfo")
    pdftoppm = _require_tool("pdftoppm")
    structural: dict[str, Any] = {}
    renderers: dict[str, Any] = {}

    qpdf_result = _run([qpdf, "--check", str(pdf_path)])
    lowered = qpdf_result["output"].lower()
    bad_markers = [marker for marker in FORBIDDEN_QPDF_MARKERS if marker in lowered]
    if qpdf_result["exit_code"] != 0 or bad_markers:
        raise VerificationError(
            f"qpdf structural check failed: markers={bad_markers}; {qpdf_result['output']}"
        )
    structural["qpdf"] = {"status": "pass", **qpdf_result}

    gs_result = _run(
        [
            gs,
            "-q",
            "-dNOPAUSE",
            "-dBATCH",
            "-sDEVICE=nullpage",
            str(pdf_path),
        ]
    )
    if gs_result["exit_code"] != 0 or gs_result["output"]:
        raise VerificationError(f"Ghostscript check failed or warned: {gs_result['output']}")
    structural["ghostscript"] = {"status": "pass", **gs_result}

    info_result = _run([pdfinfo, str(pdf_path)])
    if info_result["exit_code"] != 0:
        raise VerificationError(f"pdfinfo failed: {info_result['output']}")
    pages = _page_count(info_result["output"])

    poppler_dir = output_dir / "poppler"
    _prepare_directory(poppler_dir)
    poppler_result = _run(
        [pdftoppm, "-png", "-r", str(dpi), str(pdf_path), str(poppler_dir / "page")]
    )
    if poppler_result["exit_code"] != 0 or poppler_result["output"]:
        raise VerificationError(f"Poppler rendering failed or warned: {poppler_result['output']}")
    renderers["poppler"] = {
        "version": _version([pdftoppm, "-v"]),
        **_inspect_rendered_pages(poppler_dir, pages),
        "contact_sheet": _write_contact_sheet(poppler_dir).name,
    }

    pdfium_dir = output_dir / "pdfium"
    pdfium_version, _ = _render_pdfium(pdf_path, pdfium_dir, dpi)
    renderers["pdfium"] = {
        "version": pdfium_version,
        **_inspect_rendered_pages(pdfium_dir, pages),
        "contact_sheet": _write_contact_sheet(pdfium_dir).name,
    }

    clang = shutil.which("clang")
    pdfkit_source = ROOT / "scripts" / "render_pdf_pdfkit.m"
    if platform.system() == "Darwin" and clang and pdfkit_source.exists():
        pdfkit_dir = output_dir / "pdfkit"
        _prepare_directory(pdfkit_dir)
        renderer_path = Path(tempfile.gettempdir()) / "thermo-credit-pdfkit-renderer"
        module_cache = output_dir / "clang-module-cache"
        module_cache.mkdir(parents=True, exist_ok=True)
        compile_result = _run(
            [
                clang,
                "-fobjc-arc",
                f"-fmodules-cache-path={module_cache}",
                "-framework",
                "Foundation",
                "-framework",
                "AppKit",
                "-framework",
                "PDFKit",
                str(pdfkit_source),
                "-o",
                str(renderer_path),
            ]
        )
        if compile_result["exit_code"] != 0:
            raise VerificationError(f"PDFKit renderer compilation failed: {compile_result['output']}")
        pdfkit_result = _run(
            [str(renderer_path), str(pdf_path), str(pdfkit_dir), str(dpi)]
        )
        if pdfkit_result["exit_code"] != 0:
            raise VerificationError(f"PDFKit rendering failed: {pdfkit_result['output']}")
        renderers["pdfkit"] = {
            "version": f"PDFKit via {_version([clang, '--version'])}",
            "output": pdfkit_result["output"],
            **_inspect_rendered_pages(pdfkit_dir, pages),
            "contact_sheet": _write_contact_sheet(pdfkit_dir).name,
        }
    elif require_pdfkit:
        raise VerificationError("PDFKit rendering was required but is unavailable")
    else:
        renderers["pdfkit"] = {"status": "not_applicable", "reason": "non-macOS runner"}

    digest = hashlib.sha256(pdf_path.read_bytes()).hexdigest()
    report = {
        "status": "pass",
        "pdf": str(pdf_path),
        "sha256": digest,
        "bytes": pdf_path.stat().st_size,
        "page_count": pages,
        "dpi": dpi,
        "structural_checks": structural,
        "renderers": renderers,
        "versions": {
            "qpdf": _version([qpdf, "--version"]),
            "ghostscript": _version([gs, "--version"]),
            "pdfinfo": _version([pdfinfo, "-v"]),
            "python": sys.version.split()[0],
            "platform": platform.platform(),
        },
    }
    (output_dir / "qa-report.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    (output_dir / "theory.pdf.sha256").write_text(
        f"{digest}  {pdf_path.name}\n",
        encoding="ascii",
    )
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pdf", type=Path, default=ROOT / "output" / "pdf" / "theory.pdf")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "output" / "pdf" / "qa" / "theory")
    parser.add_argument("--dpi", type=int, default=144)
    parser.add_argument("--require-pdfkit", action="store_true")
    args = parser.parse_args()
    try:
        report = verify_pdf(
            args.pdf,
            args.output_dir,
            dpi=args.dpi,
            require_pdfkit=args.require_pdfkit,
        )
    except VerificationError as exc:
        args.output_dir.mkdir(parents=True, exist_ok=True)
        failure = {"status": "fail", "error": str(exc), "pdf": str(args.pdf)}
        (args.output_dir / "qa-report.json").write_text(
            json.dumps(failure, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        print(f"PDF verification failed: {exc}", file=sys.stderr)
        return 1
    print(
        f"Verified {report['page_count']} pages; sha256={report['sha256']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
