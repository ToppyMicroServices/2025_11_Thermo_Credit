from __future__ import annotations

import hashlib
import os
import re
import tempfile
import zipfile
from pathlib import Path, PurePosixPath


ROOT = Path(__file__).resolve().parent.parent
ARCHIVE_NAME = "jfs_anonymous_manuscript_source.zip"
ARCHIVE_ROOT = "jfs_anonymous_manuscript_source"
SOURCE_RELATIVE = Path("theory.tex")
GENERATED_DIRECTORY = "generated"
GRAPHICS_EXTENSIONS = (".pdf", ".png", ".jpg", ".jpeg", ".eps")

REFERENCE_PATTERN = re.compile(
    r"""
    \\(?P<command>input|include|includegraphics)
    (?:\s*\[[^\]]*\])?
    \s*\{(?P<target>[^{}]+)\}
    """,
    re.VERBOSE | re.MULTILINE,
)
OPTIONAL_FILE_PATTERN = re.compile(
    r"\\IfFileExists\s*\{(?P<target>[^{}]+)\}",
    re.MULTILINE,
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def strip_tex_comments(text: str) -> str:
    """Remove TeX comments while preserving escaped percent signs."""
    uncommented: list[str] = []
    for line in text.splitlines(keepends=True):
        cut = len(line)
        for index, character in enumerate(line):
            if character != "%":
                continue
            preceding_backslashes = 0
            cursor = index - 1
            while cursor >= 0 and line[cursor] == "\\":
                preceding_backslashes += 1
                cursor -= 1
            if preceding_backslashes % 2 == 0:
                cut = index
                break
        uncommented.append(line[:cut])
        if cut < len(line) and line.endswith("\n"):
            uncommented.append("\n")
    return "".join(uncommented)


def _safe_generated_path(target: str) -> PurePosixPath:
    normalized = target.strip()
    if not normalized or "\\" in normalized:
        raise ValueError(f"Unsupported dynamic or empty TeX reference: {target!r}")
    relative = PurePosixPath(normalized)
    if (
        relative.is_absolute()
        or relative.parts[0] != GENERATED_DIRECTORY
        or any(part in {"", ".", ".."} for part in relative.parts)
    ):
        raise ValueError(
            "Anonymous source references must stay under generated/: "
            f"{target!r}"
        )
    return relative


def _resolve_reference(tex_directory: Path, command: str, target: str) -> Path:
    relative = _safe_generated_path(target)
    if command in {"input", "include"}:
        candidates = [relative if relative.suffix else relative.with_suffix(".tex")]
    elif relative.suffix:
        candidates = [relative]
    else:
        candidates = [
            PurePosixPath(f"{relative.as_posix()}{extension}")
            for extension in GRAPHICS_EXTENSIONS
        ]

    generated_root = (tex_directory / GENERATED_DIRECTORY).resolve()
    for candidate in candidates:
        path = tex_directory.joinpath(*candidate.parts)
        if not path.is_file():
            continue
        resolved = path.resolve()
        try:
            resolved.relative_to(generated_root)
        except ValueError as error:
            raise ValueError(
                f"Generated reference resolves outside generated/: {target!r}"
            ) from error
        return path

    choices = ", ".join(path.as_posix() for path in candidates)
    raise FileNotFoundError(f"Missing referenced manuscript input: {choices}")


def discover_source_files(root: Path = ROOT) -> tuple[Path, ...]:
    """Return archive-relative source files needed to compile theory.tex."""
    tex_directory = root / "tex"
    source = tex_directory / SOURCE_RELATIVE
    if not source.is_file():
        raise FileNotFoundError(source)

    pending = [source]
    discovered: set[Path] = {source}
    while pending:
        current = pending.pop()
        if current.suffix.lower() != ".tex":
            continue
        text = strip_tex_comments(current.read_text(encoding="utf-8"))
        optional_targets = {
            match.group("target").strip()
            for match in OPTIONAL_FILE_PATTERN.finditer(text)
        }
        for match in REFERENCE_PATTERN.finditer(text):
            target = match.group("target")
            try:
                dependency = _resolve_reference(
                    tex_directory,
                    match.group("command"),
                    target,
                )
            except FileNotFoundError:
                if target.strip() in optional_targets:
                    continue
                raise
            if dependency not in discovered:
                discovered.add(dependency)
                pending.append(dependency)

    return tuple(
        sorted(path.relative_to(tex_directory) for path in discovered)
    )


def _write_checksum(checksum_path: Path, archive_path: Path) -> None:
    checksum_path.parent.mkdir(parents=True, exist_ok=True)
    payload = f"{sha256(archive_path)}  {archive_path.name}\n"
    with tempfile.NamedTemporaryFile(
        mode="w",
        encoding="utf-8",
        prefix=f".{checksum_path.name}.",
        dir=checksum_path.parent,
        delete=False,
    ) as stream:
        stream.write(payload)
        temporary_path = Path(stream.name)
    try:
        os.replace(temporary_path, checksum_path)
        checksum_path.chmod(0o644)
    finally:
        if temporary_path.exists():
            temporary_path.unlink()


def build_archive(
    root: Path = ROOT,
    archive_path: Path | None = None,
) -> tuple[Path, Path, tuple[Path, ...]]:
    """Build a bit-for-bit deterministic anonymous manuscript source archive."""
    root = root.resolve()
    archive_path = (
        archive_path
        if archive_path is not None
        else root / "submission" / ARCHIVE_NAME
    )
    archive_path = archive_path.resolve()
    archive_path.parent.mkdir(parents=True, exist_ok=True)
    source_files = discover_source_files(root)
    tex_directory = root / "tex"

    with tempfile.NamedTemporaryFile(
        prefix=f".{archive_path.name}.",
        suffix=".tmp",
        dir=archive_path.parent,
        delete=False,
    ) as stream:
        temporary_path = Path(stream.name)
    try:
        with zipfile.ZipFile(temporary_path, "w", compression=zipfile.ZIP_STORED) as archive:
            for relative in source_files:
                info = zipfile.ZipInfo(
                    filename=f"{ARCHIVE_ROOT}/{relative.as_posix()}",
                    date_time=(2024, 1, 1, 0, 0, 0),
                )
                info.compress_type = zipfile.ZIP_STORED
                info.create_system = 3
                info.external_attr = 0o100644 << 16
                archive.writestr(info, (tex_directory / relative).read_bytes())
        os.replace(temporary_path, archive_path)
        archive_path.chmod(0o644)
    finally:
        if temporary_path.exists():
            temporary_path.unlink()

    checksum_path = Path(f"{archive_path}.sha256")
    _write_checksum(checksum_path, archive_path)
    return archive_path, checksum_path, source_files


def main() -> int:
    archive_path, checksum_path, source_files = build_archive()
    print(f"Wrote {archive_path.relative_to(ROOT)}")
    print(f"Wrote {checksum_path.relative_to(ROOT)}")
    print("Included:")
    for path in source_files:
        print(f"  tex/{path.as_posix()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
