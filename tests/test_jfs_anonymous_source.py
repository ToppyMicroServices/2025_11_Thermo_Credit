from __future__ import annotations

import hashlib
import importlib.util
import os
import zipfile
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parent.parent
SPEC = importlib.util.spec_from_file_location(
    "jfs_anonymous_source",
    ROOT / "scripts" / "21_build_jfs_anonymous_source.py",
)
assert SPEC and SPEC.loader
builder = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(builder)


def _write_fixture(root: Path) -> None:
    generated = root / "tex" / "generated"
    generated.mkdir(parents=True)
    (root / "tex" / "theory.tex").write_text(
        r"""
\documentclass{article}
\usepackage{graphicx}
\begin{document}
\input{generated/table}
\includegraphics[width=\textwidth]{generated/bridge.pdf}
\IfFileExists{generated/optional.pdf}{
  \includegraphics{generated/optional.pdf}
}{}
% \input{generated/commented_identity}
\end{document}
""".lstrip(),
        encoding="utf-8",
    )
    (generated / "table.tex").write_text(
        "\\input{generated/note.tex}\n",
        encoding="utf-8",
    )
    (generated / "note.tex").write_text("Anonymous note.\n", encoding="utf-8")
    (generated / "bridge.pdf").write_bytes(b"%PDF-1.4\nfixture\n%%EOF\n")
    (root / "tex" / "title_page.tex").write_text(
        "IDENTITY MUST NOT ENTER ARCHIVE\n",
        encoding="utf-8",
    )


def test_discovers_only_transitive_generated_dependencies(tmp_path: Path) -> None:
    _write_fixture(tmp_path)

    discovered = builder.discover_source_files(tmp_path)

    assert discovered == (
        Path("generated/bridge.pdf"),
        Path("generated/note.tex"),
        Path("generated/table.tex"),
        Path("theory.tex"),
    )
    assert Path("title_page.tex") not in discovered
    assert Path("generated/commented_identity.tex") not in discovered


def test_includes_guarded_generated_asset_when_it_exists(tmp_path: Path) -> None:
    _write_fixture(tmp_path)
    (tmp_path / "tex" / "generated" / "optional.pdf").write_bytes(
        b"%PDF-1.4\noptional\n%%EOF\n"
    )

    discovered = builder.discover_source_files(tmp_path)

    assert Path("generated/optional.pdf") in discovered


def test_archive_is_deterministic_and_excludes_title_page(tmp_path: Path) -> None:
    _write_fixture(tmp_path)
    archive = tmp_path / "submission" / builder.ARCHIVE_NAME

    first_archive, checksum, _ = builder.build_archive(tmp_path, archive)
    first_bytes = first_archive.read_bytes()
    os.utime(tmp_path / "tex" / "theory.tex", (1_900_000_000, 1_900_000_000))
    second_archive, _, _ = builder.build_archive(tmp_path, archive)

    assert second_archive.read_bytes() == first_bytes
    expected_hash = hashlib.sha256(first_bytes).hexdigest()
    assert checksum.read_text(encoding="utf-8") == (
        f"{expected_hash}  {builder.ARCHIVE_NAME}\n"
    )

    with zipfile.ZipFile(second_archive) as source_zip:
        assert source_zip.namelist() == [
            f"{builder.ARCHIVE_ROOT}/generated/bridge.pdf",
            f"{builder.ARCHIVE_ROOT}/generated/note.tex",
            f"{builder.ARCHIVE_ROOT}/generated/table.tex",
            f"{builder.ARCHIVE_ROOT}/theory.tex",
        ]
        assert all(
            info.date_time == (2024, 1, 1, 0, 0, 0)
            for info in source_zip.infolist()
        )
        assert b"IDENTITY MUST NOT ENTER ARCHIVE" not in b"".join(
            source_zip.read(name) for name in source_zip.namelist()
        )


def test_rejects_reference_outside_generated_directory(tmp_path: Path) -> None:
    (tmp_path / "tex").mkdir()
    (tmp_path / "tex" / "theory.tex").write_text(
        "\\input{title_page}\n",
        encoding="utf-8",
    )
    (tmp_path / "tex" / "title_page.tex").write_text(
        "Identifying author details.\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="must stay under generated"):
        builder.discover_source_files(tmp_path)
