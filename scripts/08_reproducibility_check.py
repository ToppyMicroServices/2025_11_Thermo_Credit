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
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent

INPUT_CSVS = (
    "data/*.csv",
)

OUTPUTS = (
    "site/indicators.csv",
    "site/indicators_realtime.csv",
    "site/credit_destination.csv",
    "site/credit_destination_realtime.csv",
    "site/indicators_eu.csv",
    "site/indicators_eu_realtime.csv",
    "site/credit_destination_eu.csv",
    "site/credit_destination_eu_realtime.csv",
    "site/indicators_us.csv",
    "site/indicators_us_realtime.csv",
    "site/credit_destination_us.csv",
    "site/credit_destination_us_realtime.csv",
    "site/lambda_b_sensitivity.csv",
    "site/credit_destination_lambda_b_sweep.csv",
    "site/destination_oos_incremental.csv",
    "site/baseline_forecast_comparison.csv",
    "site/baseline_forecast_target_coverage.csv",
    "site/calibration_holdout_test.csv",
    "site/entropy_partition_robustness.csv",
    "site/tl_robustness.csv",
    "site/loop_area_null_tests.csv",
    "site/integrability_synthetic_test.csv",
    "site/submission_readiness.csv",
    "site/realtime_release_lags.json",
    "site/realtime_release_lags_eu.json",
    "site/realtime_release_lags_us.json",
    "data/calibrated_theory_params.json",
    "data/lambda_b_sensitivity_summary.json",
    "data/boj_bridge_validation_summary.json",
    "data/external_validation_summary.json",
    "data/destination_oos_incremental_summary.json",
    "data/baseline_forecast_summary.json",
    "data/calibration_holdout_summary.json",
    "data/entropy_partition_robustness_summary.json",
    "data/tl_robustness_summary.json",
    "data/loop_area_null_summary.json",
    "data/integrability_synthetic_summary.json",
    "data/submission_readiness_summary.json",
    "tex/generated/theory_calibration.tex",
    "tex/generated/theory_empirical_snapshot.tex",
    "tex/generated/theory_boj_bridge_mapping_main.tex",
    "tex/generated/theory_boj_primary_mapping.tex",
    "tex/generated/theory_boj_taxonomy_robustness.tex",
    "tex/generated/theory_boj_bridge_mapping.tex",
    "tex/generated/theory_boj_bridge_validation.tex",
    "tex/generated/theory_external_partial_validation.tex",
    "tex/generated/theory_lambda_b_sensitivity.tex",
    "tex/generated/theory_destination_oos_incremental.tex",
    "tex/generated/theory_destination_oos_asset_auxiliary.tex",
    "tex/generated/theory_baseline_forecast_comparison.tex",
    "tex/generated/theory_calibration_holdout.tex",
    "tex/generated/theory_entropy_partition_robustness.tex",
    "tex/generated/theory_tl_robustness.tex",
    "tex/generated/theory_loop_area_null_tests.tex",
    "tex/generated/theory_integrability_synthetic_test.tex",
    "tex/generated/theory_submission_readiness.tex",
    "tex/generated/theory_jp_destination_targets.pdf",
    "tex/generated/theory_jp_destination_targets.svg",
    "tex/generated/theory_sm_tl_panels.pdf",
    "tex/generated/theory_sm_tl_panels.svg",
    "tex/generated/theory_capacity_panels.pdf",
    "tex/generated/theory_capacity_panels.svg",
)


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def discover_input_csvs(root: Path) -> list[Path]:
    paths: list[Path] = []
    for pattern in INPUT_CSVS:
        paths.extend(root.glob(pattern))
    return sorted({p for p in paths if p.is_file()})


def hash_manifest(paths: list[Path], root: Path) -> dict[str, dict[str, Any]]:
    out: dict[str, dict[str, Any]] = {}
    for path in paths:
        rel = path.relative_to(root).as_posix()
        out[rel] = {
            "sha256": sha256_file(path),
            "bytes": path.stat().st_size,
        }
    return out


def run_command(cmd: list[str], *, cwd: Path) -> dict[str, Any]:
    proc = subprocess.run(cmd, cwd=cwd, text=True, capture_output=True)
    entry = {
        "command": " ".join(cmd),
        "returncode": proc.returncode,
        "stdout_tail": proc.stdout[-4000:],
        "stderr_tail": proc.stderr[-4000:],
    }
    if proc.returncode != 0:
        raise RuntimeError(f"Command failed: {' '.join(cmd)}\n{proc.stderr[-4000:]}")
    return entry


def generation_commands() -> list[list[str]]:
    py = sys.executable
    return [
        [py, "scripts/02_compute_indicators.py", "jp"],
        [py, "scripts/02_compute_indicators.py", "eu"],
        [py, "scripts/02_compute_indicators.py", "us"],
        [py, "scripts/03_make_report.py"],
        [py, "scripts/07_calibrate_theory_model.py"],
        [py, "scripts/18_boj_bridge_validation.py"],
        [py, "scripts/23_external_purpose_validation.py"],
        [py, "scripts/09_lambda_b_sensitivity.py"],
        [py, "scripts/19_destination_oos_incremental.py"],
        [py, "scripts/10_baseline_forecast_comparison.py"],
        [py, "scripts/11_calibration_holdout_test.py"],
        [py, "scripts/12_entropy_partition_robustness.py"],
        [py, "scripts/13_tl_robustness.py"],
        [py, "scripts/14_loop_area_null_tests.py"],
        [py, "scripts/15_integrability_synthetic_test.py"],
        [py, "scripts/16_submission_readiness.py"],
        [py, "scripts/06_make_theory_figures.py"],
    ]


def run_generation(label: str) -> list[dict[str, Any]]:
    entries: list[dict[str, Any]] = []
    for cmd in generation_commands():
        entry = run_command(cmd, cwd=ROOT)
        entry["phase"] = label
        entries.append(entry)
    return entries


def copy_outputs(target_dir: Path) -> None:
    target_dir.mkdir(parents=True, exist_ok=True)
    for rel in OUTPUTS:
        src = ROOT / rel
        if not src.exists():
            raise FileNotFoundError(f"Expected output was not generated: {rel}")
        dst = target_dir / rel
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)


def canonical_text(path: Path) -> str:
    text = path.read_text(encoding="utf-8", errors="replace").replace("\r\n", "\n")
    if path.suffix == ".svg":
        text = re.sub(r"<dc:date>.*?</dc:date>\n?", "", text)
    return text


def compare_text(ref: Path, cur: Path) -> dict[str, Any]:
    ref_text = canonical_text(ref)
    cur_text = canonical_text(cur)
    return {
        "mode": "canonical_text_exact",
        "pass": ref_text == cur_text,
        "reference_sha256": hashlib.sha256(ref_text.encode("utf-8")).hexdigest(),
        "current_sha256": hashlib.sha256(cur_text.encode("utf-8")).hexdigest(),
    }


def _numeric_series(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce")


def compare_csv(ref: Path, cur: Path, *, rtol: float, atol: float) -> dict[str, Any]:
    left = pd.read_csv(ref)
    right = pd.read_csv(cur)
    failures: list[str] = []
    max_abs_diff = 0.0
    max_rel_diff = 0.0

    if list(left.columns) != list(right.columns):
        failures.append("columns differ")
    if left.shape != right.shape:
        failures.append(f"shape differs: {left.shape} != {right.shape}")

    if not failures:
        for col in left.columns:
            l_raw = left[col]
            r_raw = right[col]
            l_num = _numeric_series(l_raw)
            r_num = _numeric_series(r_raw)
            numeric = l_num.notna().any() or r_num.notna().any()
            if numeric:
                l = l_num.to_numpy(dtype=float)
                r = r_num.to_numpy(dtype=float)
                same_nan = np.isnan(l) & np.isnan(r)
                close = np.isclose(l, r, rtol=rtol, atol=atol, equal_nan=True)
                bad = ~(close | same_nan)
                if bad.any():
                    failures.append(f"{col}: {int(bad.sum())} numeric differences")
                finite = np.isfinite(l) & np.isfinite(r)
                if finite.any():
                    abs_diff = np.abs(l[finite] - r[finite])
                    max_abs_diff = max(max_abs_diff, float(abs_diff.max(initial=0.0)))
                    denom = np.maximum(np.abs(l[finite]), atol)
                    rel_diff = abs_diff / denom
                    max_rel_diff = max(max_rel_diff, float(rel_diff.max(initial=0.0)))
            else:
                l = l_raw.fillna("").astype(str).to_list()
                r = r_raw.fillna("").astype(str).to_list()
                if l != r:
                    failures.append(f"{col}: text values differ")

    return {
        "mode": "csv_numeric_tolerance",
        "pass": not failures,
        "rtol": rtol,
        "atol": atol,
        "max_abs_diff": max_abs_diff,
        "max_rel_diff": max_rel_diff,
        "failures": failures,
    }


def compare_json_values(left: Any, right: Any, *, path: str, rtol: float, atol: float, failures: list[str]) -> None:
    if isinstance(left, dict) and isinstance(right, dict):
        if set(left) != set(right):
            failures.append(f"{path}: keys differ")
            return
        for key in sorted(left):
            compare_json_values(left[key], right[key], path=f"{path}.{key}", rtol=rtol, atol=atol, failures=failures)
        return
    if isinstance(left, list) and isinstance(right, list):
        if len(left) != len(right):
            failures.append(f"{path}: list length differs")
            return
        for idx, (l_item, r_item) in enumerate(zip(left, right)):
            compare_json_values(l_item, r_item, path=f"{path}[{idx}]", rtol=rtol, atol=atol, failures=failures)
        return
    if isinstance(left, (int, float)) or isinstance(right, (int, float)):
        try:
            l_num = float(left)
            r_num = float(right)
        except Exception:
            failures.append(f"{path}: numeric coercion differs")
            return
        if not np.isclose(l_num, r_num, rtol=rtol, atol=atol, equal_nan=True):
            failures.append(f"{path}: {l_num!r} != {r_num!r}")
        return
    if left != right:
        failures.append(f"{path}: value differs")


def compare_json(ref: Path, cur: Path, *, rtol: float, atol: float) -> dict[str, Any]:
    left = json.loads(ref.read_text(encoding="utf-8"))
    right = json.loads(cur.read_text(encoding="utf-8"))
    failures: list[str] = []
    compare_json_values(left, right, path="$", rtol=rtol, atol=atol, failures=failures)
    return {
        "mode": "json_numeric_tolerance",
        "pass": not failures,
        "rtol": rtol,
        "atol": atol,
        "failures": failures[:50],
        "failure_count": len(failures),
    }


def compare_binary(ref: Path, cur: Path) -> dict[str, Any]:
    ref_hash = sha256_file(ref)
    cur_hash = sha256_file(cur)
    return {
        "mode": "binary_exact",
        "pass": ref_hash == cur_hash,
        "reference_sha256": ref_hash,
        "current_sha256": cur_hash,
        "reference_bytes": ref.stat().st_size,
        "current_bytes": cur.stat().st_size,
    }


def compare_outputs(reference_dir: Path, *, rtol: float, atol: float) -> dict[str, dict[str, Any]]:
    comparisons: dict[str, dict[str, Any]] = {}
    for rel in OUTPUTS:
        ref = reference_dir / rel
        cur = ROOT / rel
        if rel.endswith(".csv"):
            comparisons[rel] = compare_csv(ref, cur, rtol=rtol, atol=atol)
        elif rel.endswith(".json"):
            comparisons[rel] = compare_json(ref, cur, rtol=rtol, atol=atol)
        elif rel.endswith((".tex", ".svg")):
            comparisons[rel] = compare_text(ref, cur)
        else:
            comparisons[rel] = compare_binary(ref, cur)
    return comparisons


def write_log(report_dir: Path, manifest: dict[str, Any]) -> None:
    report_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = report_dir / "reproducibility_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
    lines = [
        "# Reproducibility Check",
        "",
        f"- Status: {'PASS' if manifest['pass'] else 'FAIL'}",
        f"- Generated at: {manifest['generated_at_utc']}",
        f"- Tolerance: rtol={manifest['tolerance']['rtol']}, atol={manifest['tolerance']['atol']}",
        f"- Input CSV files hashed: {len(manifest['input_csv_hashes'])}",
        "",
        "## Output Comparisons",
        "",
    ]
    for rel, result in manifest["comparisons"].items():
        status = "PASS" if result.get("pass") else "FAIL"
        lines.append(f"- `{rel}`: {status} ({result.get('mode')})")
        if result.get("failures"):
            lines.append(f"  - failures: {result['failures'][:3]}")
    lines.extend(["", "## Input CSV Hashes", ""])
    for rel, item in manifest["input_csv_hashes"].items():
        lines.append(f"- `{rel}`: `{item['sha256']}` ({item['bytes']} bytes)")
    (report_dir / "reproducibility_log.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a two-pass raw-data reproducibility check.")
    parser.add_argument("--rtol", type=float, default=1e-9, help="Relative tolerance for numeric outputs.")
    parser.add_argument("--atol", type=float, default=1e-8, help="Absolute tolerance for numeric outputs.")
    parser.add_argument("--report-dir", default="replication", help="Directory for the manifest and markdown log.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    report_dir = ROOT / args.report_dir
    input_paths = discover_input_csvs(ROOT)
    input_before = hash_manifest(input_paths, ROOT)
    commands: list[dict[str, Any]] = []
    with tempfile.TemporaryDirectory(prefix="thermo_repro_") as tmp:
        reference_dir = Path(tmp) / "reference"
        commands.extend(run_generation("reference"))
        input_after_reference = hash_manifest(input_paths, ROOT)
        copy_outputs(reference_dir)
        commands.extend(run_generation("rebuild"))
        input_after_rebuild = hash_manifest(input_paths, ROOT)
        comparisons = compare_outputs(reference_dir, rtol=args.rtol, atol=args.atol)

    input_hashes_stable = input_before == input_after_reference == input_after_rebuild
    outputs_pass = all(item.get("pass") for item in comparisons.values())
    manifest = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "python": sys.version,
        "platform": platform.platform(),
        "tolerance": {"rtol": args.rtol, "atol": args.atol},
        "input_csv_hashes": input_before,
        "input_hashes_stable": input_hashes_stable,
        "commands": commands,
        "outputs": hash_manifest([ROOT / rel for rel in OUTPUTS], ROOT),
        "comparisons": comparisons,
        "pass": bool(input_hashes_stable and outputs_pass),
    }
    write_log(report_dir, manifest)
    if not manifest["pass"]:
        print(f"Reproducibility check FAILED. See {report_dir / 'reproducibility_log.md'}")
        return 1
    print(f"Reproducibility check PASSED. See {report_dir / 'reproducibility_log.md'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
