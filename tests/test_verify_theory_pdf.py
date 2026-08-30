import importlib.util
from pathlib import Path

import pytest
from PIL import Image


SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "verify_theory_pdf.py"


def _module():
    spec = importlib.util.spec_from_file_location("verify_theory_pdf", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_page_count_parser() -> None:
    module = _module()
    assert module._page_count("Title: Test\nPages:          17\nEncrypted: no\n") == 17
    with pytest.raises(module.VerificationError, match="page count"):
        module._page_count("Title: Test\n")


def test_rendered_page_inspection_rejects_blank_page(tmp_path: Path) -> None:
    module = _module()
    image = Image.new("RGB", (800, 900), "white")
    image.save(tmp_path / "page-001.png")
    with pytest.raises(module.VerificationError, match="appears blank"):
        module._inspect_rendered_pages(tmp_path, 1)


def test_rendered_page_inspection_records_nonblank_page(tmp_path: Path) -> None:
    module = _module()
    image = Image.new("RGB", (800, 900), "white")
    for x in range(100, 700):
        for y in range(100, 110):
            image.putpixel((x, y), (0, 0, 0))
    image.save(tmp_path / "page-001.png")
    result = module._inspect_rendered_pages(tmp_path, 1)
    assert result["status"] == "pass"
    assert result["page_count"] == 1
