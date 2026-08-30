from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


TARGET_LABELS = {
    "asset_acceleration": "asset acceleration",
    "spread_widening": "spread widening",
}

REPRODUCIBILITY_REQUIRED_OUTPUTS = {
    "site/destination_oos_incremental.csv",
    "data/destination_oos_incremental_summary.json",
    "data/boj_bridge_validation_summary.json",
    "tex/generated/theory_boj_bridge_mapping_main.tex",
    "tex/generated/theory_destination_oos_incremental.tex",
    "tex/generated/theory_boj_bridge_mapping.tex",
    "tex/generated/theory_boj_bridge_validation.tex",
    "tex/generated/theory_submission_readiness.tex",
    "tex/generated/theory_jp_destination_targets.pdf",
    "tex/generated/theory_jp_destination_targets.svg",
}


def _read_csv(path: Path) -> pd.DataFrame:
    return pd.read_csv(path) if path.exists() else pd.DataFrame()


def _fmt_bool(value: bool) -> str:
    return "pass" if bool(value) else "not yet"


def _metric_improves(row: pd.Series, *, require_ci: bool = True) -> bool:
    diff = pd.to_numeric(pd.Series([row.get("metric_loss_diff")]), errors="coerce").iloc[0]
    if not np.isfinite(diff) or diff >= 0:
        return False
    if require_ci:
        hi = pd.to_numeric(pd.Series([row.get("block_ci_high")]), errors="coerce").iloc[0]
        return bool(np.isfinite(hi) and hi < 0)
    return True


def _criterion_destination_oos(root: Path) -> dict[str, Any]:
    path = root / "site" / "destination_oos_incremental.csv"
    frame = _read_csv(path)
    rows: list[dict[str, Any]] = []
    complete_main = False
    if not frame.empty:
        local = frame[
            frame["region_key"].eq("jp")
            & frame["baseline"].eq("boj_mapped_stock_growth")
            & frame["model"].eq("matched_credit_plus_q_t")
            & frame["target"].isin(TARGET_LABELS)
        ].copy()
        if "is_primary_allocation_measure" in local.columns:
            primary = local["is_primary_allocation_measure"]
            if primary.dtype != bool:
                primary = primary.astype(str).str.lower().isin({"1", "true", "yes"})
            local = local[primary].copy()
        if "is_primary_training_window" in local.columns:
            primary_window = local["is_primary_training_window"]
            if primary_window.dtype != bool:
                primary_window = primary_window.astype(str).str.lower().isin(
                    {"1", "true", "yes"}
                )
            local = local[primary_window].copy()
        for target in TARGET_LABELS:
            target_rows = local[local["target"].eq(target)]
            if target_rows.empty:
                rows.append({"target": target, "status": "missing", "metric_loss_diff": None, "block_ci": ""})
                continue
            best = target_rows.sort_values("metric_loss_diff").iloc[0]
            lower_point_rows = int(
                pd.to_numeric(target_rows["metric_loss_diff"], errors="coerce").lt(0).sum()
            )
            horizon_bits = []
            for _, hrow in target_rows.sort_values("horizon_quarters").iterrows():
                horizon_bits.append(
                    f"{int(hrow['horizon_quarters'])}Q [{float(hrow['block_ci_low']):.3f}, {float(hrow['block_ci_high']):.3f}]"
                )
            rows.append(
                {
                    "target": target,
                    "status": f"{lower_point_rows}/{len(target_rows)} lower point-loss rows",
                    "metric_loss_diff": float(best["metric_loss_diff"]),
                    "block_ci": "; ".join(horizon_bits),
                }
            )
        main_horizons = set(
            pd.to_numeric(
                local.loc[local["target"].eq("spread_widening"), "horizon_quarters"],
                errors="coerce",
            ).dropna().astype(int)
        )
        complete_main = {4, 8}.issubset(main_horizons)
    passed = bool(complete_main)
    detail = "; ".join(
        f"{TARGET_LABELS[item['target']]}: {item['status']} ({item['block_ci']})"
        for item in rows
    )
    return {
        "criterion_id": "borrower_composition_oos",
        "criterion": "The JP use example matches the BOJ scale and composition populations and reports both point loss and uncertainty.",
        "status": "pass" if passed else "not_yet",
        "current_read": detail or "matched-population application unavailable",
        "blocking_issue": (
            "The 4Q and 8Q primary borrower-composition rows against matched BOJ stock growth are incomplete."
            if not passed
            else ""
        ),
        "next_action": (
            "Treat the table as a use example; require an independent, pre-specified sample before making a predictive claim."
        ),
    }


def _criterion_calibration_holdout(root: Path) -> dict[str, Any]:
    path = root / "site" / "calibration_holdout_test.csv"
    frame = _read_csv(path)
    if frame.empty:
        return {
            "criterion_id": "calibrated_xc_oos",
            "criterion": "Calibrated X_C beats raw X_C and a simple baseline OOS.",
            "status": "not_yet",
            "current_read": "calibration holdout table unavailable",
            "blocking_issue": "No holdout comparison is available.",
            "next_action": "Regenerate calibration holdout tests with fixed and rolling train/test splits.",
        }
    wins = frame["winner_rmse"].astype(str).eq("tuned_XC")
    passed = bool(wins.all())
    tuned = int(wins.sum())
    total = int(len(frame))
    return {
        "criterion_id": "calibrated_xc_oos",
        "criterion": "Calibrated X_C beats raw X_C and a simple baseline OOS.",
        "status": "pass" if passed else "not_yet",
        "current_read": f"tuned X_C is RMSE winner in {tuned}/{total} region-strategy rows",
        "blocking_issue": (
            "Raw X_C or the simple trailing baseline still wins in most holdout rows."
            if not passed else ""
        ),
        "next_action": "Keep calibrated X_C as a diagnostic overlay until it dominates raw X_C and the simple baseline OOS.",
    }


def _criterion_sm_richer_data(root: Path) -> dict[str, Any]:
    entropy = _read_csv(root / "site" / "entropy_partition_robustness.csv")
    credit = _read_csv(root / "site" / "credit_destination.csv")
    observed = entropy[entropy["control"].eq("observed")] if not entropy.empty else pd.DataFrame()
    all_flat = bool(not observed.empty and observed["flat_flag"].astype(bool).all())
    source = ""
    if not credit.empty and "credit_destination_source" in credit.columns:
        values = sorted(str(x) for x in credit["credit_destination_source"].dropna().unique())
        source = ",".join(values)
    passed = bool(source and source != "allocation_proxy" and not all_flat)
    return {
        "criterion_id": "sm_richer_loan_purpose",
        "criterion": "S_M_hat moves under richer sectoral or loan-purpose data; otherwise S_M is redefined as a scale indicator.",
        "status": "pass" if passed else "not_yet",
        "current_read": f"source={source or 'unknown'}; observed entropy partitions all_flat={all_flat}",
        "blocking_issue": (
            "Current panels use coarse allocation proxies and all observed S_M_hat robustness partitions are flat."
            if not passed else ""
        ),
        "next_action": "Keep S_M as auxiliary unless the sectoral bridge is replaced or confirmed by direct loan-purpose shares.",
    }


def _criterion_robustness(root: Path) -> dict[str, Any]:
    tl = _read_csv(root / "site" / "tl_robustness.csv")
    loop = _read_csv(root / "site" / "loop_area_null_tests.csv")
    tl_pass = bool(not tl.empty and tl["monotone_pass"].astype(bool).all())
    loop_latest = loop[loop["window_family"].eq("latest_rolling")] if not loop.empty else pd.DataFrame()
    loop_methods = set(loop_latest["null_method"].dropna()) if not loop_latest.empty else set()
    loop_extreme = (
        loop_latest.groupby(["region_key", "segmentation_window"])["null_status"]
        .apply(lambda values: set(values).issubset({"top_5pct", "top_10pct"}))
        if not loop_latest.empty else pd.Series(dtype=bool)
    )
    loop_pass = bool(not loop_extreme.empty and loop_extreme.all() and len(loop_methods) >= 5)
    passed = tl_pass and loop_pass
    return {
        "criterion_id": "tl_loop_robustness",
        "criterion": "T_L and loop-area results survive normalization, window, bucket, and event-definition changes.",
        "status": "pass" if passed else "mixed",
        "current_read": f"T_L monotonicity pass={tl_pass}; loop robust across null/window definitions={loop_pass}",
        "blocking_issue": (
            "Liquidity-state variants pass monotonicity checks, but loop-area evidence is mixed across phase, AR, block, event, and placebo nulls."
            if not passed else ""
        ),
        "next_action": "Use T_L as a monotone monitoring convention and loop area as an audit trigger unless loop extremes survive all null designs.",
    }


def _criterion_reproducibility(root: Path) -> dict[str, Any]:
    log_path = root / "replication" / "reproducibility_log.md"
    manifest_path = root / "replication" / "reproducibility_manifest.json"
    text = log_path.read_text(encoding="utf-8", errors="replace") if log_path.exists() else ""
    log_passed = "- Status: PASS" in text
    manifest_passed = False
    missing_outputs = sorted(REPRODUCIBILITY_REQUIRED_OUTPUTS)
    generated = "unknown"
    if manifest_path.exists():
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            generated = str(manifest.get("generated_at_utc") or "unknown")
            outputs = set((manifest.get("outputs") or {}).keys())
            missing_outputs = sorted(REPRODUCIBILITY_REQUIRED_OUTPUTS - outputs)
            manifest_passed = bool(manifest.get("pass")) and not missing_outputs
        except Exception:
            manifest_passed = False
    passed = bool(log_passed and manifest_passed)
    if passed:
        current_read = f"two-pass reproducibility check PASS ({generated})"
    elif log_passed and missing_outputs:
        current_read = "PASS log is stale; missing current BOJ bridge or destination-OOS outputs"
    else:
        current_read = "current two-pass reproducibility PASS not found"
    return {
        "criterion_id": "full_reproducibility",
        "criterion": "Figures and numeric tables regenerate from raw public data with fixed hashes.",
        "status": "pass" if passed else "not_yet",
        "current_read": current_read,
        "blocking_issue": "" if passed else "Rerun scripts/08_reproducibility_check.py after the BOJ bridge and destination-OOS outputs are in place.",
        "next_action": (
            "Keep the reproduction log and input SHA-256 manifest in the review package."
            if passed else
            "Rerun the two-pass reproducibility check and update the review-package manifest."
        ),
    }


def evaluate_submission_readiness(root: Path) -> pd.DataFrame:
    rows = [
        _criterion_destination_oos(root),
        _criterion_calibration_holdout(root),
        _criterion_sm_richer_data(root),
        _criterion_robustness(root),
        _criterion_reproducibility(root),
    ]
    out = pd.DataFrame(rows)
    out.insert(0, "priority", range(1, len(out) + 1))
    return out


def summarize_submission_readiness(results: pd.DataFrame) -> dict[str, Any]:
    passed = int(results["status"].eq("pass").sum()) if not results.empty else 0
    total = int(len(results))
    blockers = results[~results["status"].eq("pass")]["criterion_id"].to_list() if not results.empty else []
    return {
        "passed": passed,
        "total": total,
        "blockers": blockers,
        "submit_now": bool(total > 0 and passed == total),
        "recommended_positioning": (
            "Position the submission as a borrower-composition measurement paper. "
            "The BOJ mapping and validation audits are the main evidence; the OOS table is a bounded use example."
        ),
    }


def _latex_escape(value: Any) -> str:
    return (
        str(value)
        .replace("\\", "\\textbackslash{}")
        .replace("_", "\\_")
        .replace("&", "\\&")
        .replace("%", "\\%")
    )


def render_submission_readiness_tex(results: pd.DataFrame) -> str:
    if results.empty:
        return "% Submission-readiness table unavailable.\n"
    lines = [
        "\\begin{table}[htbp]",
        "  \\centering",
        "  \\small",
        "  \\setlength{\\tabcolsep}{3pt}",
        "  \\caption{Submission-readiness gates for the next revision.}",
        "  \\label{tab:submission_readiness}",
        "  \\resizebox{\\textwidth}{!}{%",
        "  \\begin{tabular}{@{}lllll@{}}",
        "    \\toprule",
        "    Priority & Gate & Status & Current read & Next action \\\\",
        "    \\midrule",
    ]
    for _, row in results.iterrows():
        lines.append(
            "    "
            + " & ".join(
                [
                    str(int(row["priority"])),
                    _latex_escape(row["criterion"]),
                    _latex_escape(row["status"].replace("_", " ")),
                    _latex_escape(row["current_read"]),
                    _latex_escape(row["next_action"]),
                ]
            )
            + " \\\\"
        )
    lines.extend(
        [
            "    \\bottomrule",
            "  \\end{tabular}",
            "  }",
            "  \\par\\smallskip\\raggedright\\footnotesize "
            "These gates are deliberately stricter than the current prototype evidence. "
            "The protected claim is an auditable borrower-composition measure, "
            "not a validated loan-purpose or forecasting model.",
            "\\end{table}",
        ]
    )
    return "\n".join(lines) + "\n"


def write_submission_readiness_outputs(results: pd.DataFrame, *, root: Path) -> list[Path]:
    site_dir = root / "site"
    data_dir = root / "data"
    tex_dir = root / "tex" / "generated"
    site_dir.mkdir(parents=True, exist_ok=True)
    data_dir.mkdir(parents=True, exist_ok=True)
    tex_dir.mkdir(parents=True, exist_ok=True)
    csv_path = site_dir / "submission_readiness.csv"
    json_path = data_dir / "submission_readiness_summary.json"
    tex_path = tex_dir / "theory_submission_readiness.tex"
    results.to_csv(csv_path, index=False)
    json_path.write_text(
        json.dumps(summarize_submission_readiness(results), indent=2, sort_keys=True, allow_nan=False),
        encoding="utf-8",
    )
    tex_path.write_text(render_submission_readiness_tex(results), encoding="utf-8")
    return [csv_path, json_path, tex_path]
