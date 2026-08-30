from __future__ import annotations

import hashlib
import json
import os
import shutil
import tempfile
import zipfile
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent
SUBMISSION = ROOT / "submission"
ARCHIVE_NAME = "anonymous_replication_archive.zip"
ARCHIVE_ROOT = "borrower_composition_replication"

FILES = (
    ("submission/anonymous_replication_README.md", "README.md"),
    ("submission/anonymous_replication_requirements.txt", "requirements.txt"),
    ("submission/anonymous_replication_verify.py", "verify.py"),
    ("LICENSE", "LICENSE"),
    ("data/credit_destination_jp.csv", "data/credit_destination_jp.csv"),
    ("data/credit_destination_jp_metadata.json", "data/credit_destination_jp_metadata.json"),
    (
        "data/external_validation/mof_manufacturing_bank_borrowing.csv",
        "data/external_validation/mof_manufacturing_bank_borrowing.csv",
    ),
    (
        "data/external_validation/mlit_private_housing_2024.csv",
        "data/external_validation/mlit_private_housing_2024.csv",
    ),
    ("data/external_validation/metadata.json", "data/external_validation/metadata.json"),
    ("data/external_validation/README.md", "data/external_validation/README.md"),
    ("site/indicators_realtime.csv", "site/indicators_realtime.csv"),
    ("prospective/protocol.json", "prospective/protocol.json"),
    ("prospective/OSF_PROTOCOL.md", "prospective/OSF_PROTOCOL.md"),
    ("lib/__init__.py", "lib/__init__.py"),
    ("lib/forecast_frames.py", "lib/forecast_frames.py"),
    ("lib/baseline_forecast.py", "lib/baseline_forecast.py"),
    ("lib/destination_oos.py", "lib/destination_oos.py"),
    ("lib/boj_credit_taxonomies.py", "lib/boj_credit_taxonomies.py"),
    ("lib/boj_bridge_validation.py", "lib/boj_bridge_validation.py"),
    ("lib/external_purpose_validation.py", "lib/external_purpose_validation.py"),
    ("lib/vintage_archive.py", "lib/vintage_archive.py"),
    ("scripts/18_boj_bridge_validation.py", "scripts/18_boj_bridge_validation.py"),
    ("scripts/19_destination_oos_incremental.py", "scripts/19_destination_oos_incremental.py"),
    ("scripts/23_external_purpose_validation.py", "scripts/23_external_purpose_validation.py"),
    ("scripts/24_capture_boj_vintage.py", "scripts/24_capture_boj_vintage.py"),
    ("scripts/25_score_boj_vintage.py", "scripts/25_score_boj_vintage.py"),
    ("scripts/26_verify_boj_vintage_archive.py", "scripts/26_verify_boj_vintage_archive.py"),
)

EXPECTED_OUTPUTS = (
    "data/boj_bridge_validation_summary.json",
    "data/external_validation_summary.json",
    "data/destination_oos_incremental_summary.json",
    "site/destination_oos_incremental.csv",
    "tex/generated/theory_boj_bridge_mapping_main.tex",
    "tex/generated/theory_boj_primary_mapping.tex",
    "tex/generated/theory_boj_taxonomy_robustness.tex",
    "tex/generated/theory_boj_bridge_mapping.tex",
    "tex/generated/theory_boj_bridge_validation.tex",
    "tex/generated/theory_external_partial_validation.tex",
    "tex/generated/theory_destination_oos_incremental.tex",
    "tex/generated/theory_destination_oos_asset_auxiliary.tex",
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def copy_file(source: Path, destination: Path) -> None:
    if not source.is_file():
        raise FileNotFoundError(source)
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, destination)


def write_deterministic_zip(source_root: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        prefix="anonymous_replication_",
        suffix=".zip",
        dir=destination.parent,
        delete=False,
    ) as temporary:
        temporary_path = Path(temporary.name)
    try:
        with zipfile.ZipFile(
            temporary_path,
            "w",
            compression=zipfile.ZIP_DEFLATED,
            compresslevel=9,
        ) as archive:
            for path in sorted(source_root.rglob("*")):
                if not path.is_file():
                    continue
                relative = path.relative_to(source_root).as_posix()
                info = zipfile.ZipInfo(
                    filename=f"{ARCHIVE_ROOT}/{relative}",
                    date_time=(2024, 1, 1, 0, 0, 0),
                )
                info.compress_type = zipfile.ZIP_DEFLATED
                info.external_attr = 0o100644 << 16
                archive.writestr(info, path.read_bytes())
        os.replace(temporary_path, destination)
        destination.chmod(0o644)
    finally:
        if temporary_path.exists():
            temporary_path.unlink()


def main() -> int:
    with tempfile.TemporaryDirectory(prefix="anonymous_replication_stage_") as directory:
        stage = Path(directory)
        for source, destination in FILES:
            copy_file(ROOT / source, stage / destination)
        for output in EXPECTED_OUTPUTS:
            copy_file(ROOT / output, stage / "expected" / output)

        inventory = {
            path.relative_to(stage).as_posix(): {
                "sha256": sha256(path),
                "bytes": path.stat().st_size,
            }
            for path in sorted(stage.rglob("*"))
            if path.is_file()
        }
        (stage / "manifest_sha256.json").write_text(
            json.dumps(inventory, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        archive_path = SUBMISSION / ARCHIVE_NAME
        write_deterministic_zip(stage, archive_path)

    checksum_path = SUBMISSION / f"{ARCHIVE_NAME}.sha256"
    checksum_path.write_text(
        f"{sha256(archive_path)}  {ARCHIVE_NAME}\n",
        encoding="utf-8",
    )
    print(f"Wrote {archive_path.relative_to(ROOT)}")
    print(f"Wrote {checksum_path.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
