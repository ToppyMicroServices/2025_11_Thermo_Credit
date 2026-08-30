from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import pandas as pd


REGION_SOURCES = {
    "jp": ("Japan (JP)", "allocation_q.csv"),
    "eu": ("Euro Area (EU)", "allocation_q_eu.csv"),
    "us": ("United States (US)", "allocation_q_us.csv"),
}

DESTINATION_SOURCES = {
    "jp": "credit_destination_jp.csv",
}

PARTITION_FAMILIES = ("borrower_label", "loan_purpose")
BUCKET_COUNTS = (3, 5, 7)
CONTROLS = ("observed", "shuffled_shares", "fixed_shares", "random_walk_shares")

FLAT_RANGE_TOL = 1e-4
FLAT_SD_TOL = 1e-5


def _stable_seed(*parts: Any, base_seed: int = 7_311) -> int:
    payload = "|".join(str(part) for part in parts)
    digest = hashlib.sha256(payload.encode("utf-8")).hexdigest()
    return (int(digest[:12], 16) + int(base_seed)) % (2**32 - 1)


def _normalize_shares(frame: pd.DataFrame) -> pd.DataFrame:
    shares = frame.apply(pd.to_numeric, errors="coerce").clip(lower=0.0)
    total = shares.sum(axis=1)
    out = shares.div(total.replace(0.0, np.nan), axis=0)
    return out


def _entropy_hat(shares: pd.DataFrame) -> pd.Series:
    norm = _normalize_shares(shares)
    k = len(norm.columns)
    if k <= 1:
        return pd.Series(np.nan, index=norm.index)
    arr = norm.to_numpy(dtype=float)
    mask = arr > 0
    terms = np.zeros_like(arr, dtype=float)
    terms[mask] = arr[mask] * np.log(arr[mask])
    values = -terms.sum(axis=1) / math.log(k)
    invalid = ~np.isfinite(arr).any(axis=1)
    values[invalid] = np.nan
    return pd.Series(values, index=norm.index)


def _has_columns(frame: pd.DataFrame, columns: tuple[str, ...]) -> bool:
    return all(col in frame.columns for col in columns)


def _borrower_base(frame: pd.DataFrame) -> pd.DataFrame:
    legacy = ("q_pay", "q_firm", "q_asset", "q_reserve")
    if _has_columns(frame, legacy):
        return frame.loc[:, legacy].rename(
            columns={
                "q_pay": "households_pay",
                "q_firm": "firms",
                "q_asset": "asset_finance",
                "q_reserve": "public_reserve",
            }
        )
    purpose = _loan_purpose_base(frame)
    out = pd.DataFrame(index=frame.index)
    out["households_pay"] = purpose.get("housing", 0.0) + purpose.get("consumption", 0.0)
    out["firms"] = purpose.get("productive", 0.0)
    out["asset_finance"] = purpose.get("financial", 0.0)
    out["public_reserve"] = purpose.get("government", 0.0)
    return out


def _loan_purpose_base(
    frame: pd.DataFrame,
    *,
    household_housing_share: float = 0.4,
) -> pd.DataFrame:
    purpose = ("q_productive", "q_housing", "q_consumption", "q_financial", "q_government")
    purpose_columns = {
        "q_productive": "productive",
        "q_housing": "housing",
        "q_consumption": "consumption",
        "q_financial": "financial",
        "q_government": "government",
    }
    present = [column for column in purpose if column in frame.columns]
    if len(present) >= 2:
        out = pd.DataFrame(index=frame.index)
        for source, target in purpose_columns.items():
            out[target] = (
                pd.to_numeric(frame[source], errors="coerce")
                if source in frame.columns
                else 0.0
            )
        return out
    legacy = _borrower_base_from_legacy_only(frame)
    pay = legacy["households_pay"]
    out = pd.DataFrame(index=frame.index)
    out["productive"] = legacy["firms"]
    out["housing"] = pay * float(household_housing_share)
    out["consumption"] = pay * (1.0 - float(household_housing_share))
    out["financial"] = legacy["asset_finance"]
    out["government"] = legacy["public_reserve"]
    return out


def _borrower_base_from_legacy_only(frame: pd.DataFrame) -> pd.DataFrame:
    legacy = ("q_pay", "q_firm", "q_asset", "q_reserve")
    if _has_columns(frame, legacy):
        return frame.loc[:, legacy].rename(
            columns={
                "q_pay": "households_pay",
                "q_firm": "firms",
                "q_asset": "asset_finance",
                "q_reserve": "public_reserve",
            }
        )
    fallback_cols = [col for col in frame.columns if col.startswith("q_")]
    if len(fallback_cols) < 2:
        raise ValueError("Need at least two q_* columns for entropy robustness.")
    out = frame.loc[:, fallback_cols].copy()
    out.columns = [f"bucket_{idx + 1}" for idx in range(len(out.columns))]
    return out


def build_partition_shares(
    frame: pd.DataFrame,
    *,
    family: str,
    bucket_count: int,
    household_housing_share: float = 0.4,
    housing_construction_share: float = 0.35,
) -> pd.DataFrame:
    """Return deterministic shares for a robustness partition.

    The 5- and 7-bucket variants may split coarse observed categories. They
    are sensitivity partitions, not claims of directly observed sub-buckets.
    """
    family = family.strip().lower()
    if family not in PARTITION_FAMILIES:
        raise ValueError(f"Unknown partition family: {family}")
    if bucket_count not in BUCKET_COUNTS:
        raise ValueError(f"Unsupported bucket count: {bucket_count}")

    if family == "borrower_label":
        base = _borrower_base(frame)
        if bucket_count == 3:
            out = pd.DataFrame(index=base.index)
            out["households_pay"] = base["households_pay"]
            out["firms"] = base["firms"]
            out["asset_and_public"] = base["asset_finance"] + base["public_reserve"]
        elif bucket_count == 5:
            out = pd.DataFrame(index=base.index)
            out["households_pay"] = base["households_pay"]
            out["firms"] = base["firms"]
            out["asset_finance"] = base["asset_finance"]
            out["public_reserve_core"] = base["public_reserve"] * 0.7
            out["public_reserve_other"] = base["public_reserve"] * 0.3
        else:
            out = pd.DataFrame(index=base.index)
            out["household_housing"] = base["households_pay"] * household_housing_share
            out["household_other"] = base["households_pay"] * (1.0 - household_housing_share)
            out["firm_working_capital"] = base["firms"] * 0.5
            out["firm_investment"] = base["firms"] * 0.5
            out["asset_finance"] = base["asset_finance"]
            out["public_reserve_core"] = base["public_reserve"] * 0.7
            out["public_reserve_other"] = base["public_reserve"] * 0.3
    else:
        base = _loan_purpose_base(frame, household_housing_share=household_housing_share)
        if bucket_count == 3:
            out = pd.DataFrame(index=base.index)
            out["productive"] = base["productive"]
            out["household_uses"] = base["housing"] + base["consumption"]
            out["asset_public_uses"] = base["financial"] + base["government"]
        elif bucket_count == 5:
            out = base.copy()
        else:
            out = pd.DataFrame(index=base.index)
            out["productive"] = base["productive"]
            out["housing_construction"] = base["housing"] * housing_construction_share
            out["existing_housing"] = base["housing"] * (1.0 - housing_construction_share)
            out["consumption"] = base["consumption"]
            out["existing_financial_claims"] = base["financial"] * 0.5
            out["market_finance"] = base["financial"] * 0.5
            out["government"] = base["government"]

    return _normalize_shares(out)


def _control_shares(
    shares: pd.DataFrame,
    *,
    control: str,
    seed: int,
    random_walk_sigma: float = 0.03,
) -> pd.DataFrame:
    control = control.strip().lower()
    norm = _normalize_shares(shares)
    if control == "observed":
        return norm
    if control == "fixed_shares":
        mean = norm.mean(axis=0, skipna=True)
        fixed = pd.DataFrame([mean.to_numpy()] * len(norm), columns=norm.columns, index=norm.index)
        return _normalize_shares(fixed)

    rng = np.random.default_rng(seed)
    if control == "shuffled_shares":
        shuffled = pd.DataFrame(index=norm.index)
        for col in norm.columns:
            values = norm[col].to_numpy(dtype=float)
            shuffled[col] = rng.permutation(values)
        return _normalize_shares(shuffled)

    if control == "random_walk_shares":
        k = len(norm.columns)
        if k == 0:
            return norm
        start = norm.mean(axis=0, skipna=True).fillna(1.0 / k).clip(lower=1e-6)
        logits = np.log(start.to_numpy(dtype=float))
        rows = []
        for idx in range(len(norm)):
            if idx > 0:
                logits = logits + rng.normal(0.0, random_walk_sigma, size=k)
            shifted = logits - np.nanmax(logits)
            probs = np.exp(shifted)
            probs = probs / probs.sum()
            rows.append(probs)
        return pd.DataFrame(rows, columns=norm.columns, index=norm.index)

    raise ValueError(f"Unknown control: {control}")


def _safe_corr(left: pd.Series, right: pd.Series) -> float:
    pair = pd.concat([left, right], axis=1).dropna()
    if len(pair) < 3:
        return float("nan")
    if pair.iloc[:, 0].std(ddof=0) == 0 or pair.iloc[:, 1].std(ddof=0) == 0:
        return float("nan")
    return float(pair.iloc[:, 0].corr(pair.iloc[:, 1]))


def _evaluate_series(
    values: pd.Series,
    *,
    observed: pd.Series,
    dates: pd.Series,
) -> dict[str, Any]:
    clean = pd.to_numeric(values, errors="coerce")
    obs = pd.to_numeric(observed, errors="coerce")
    valid = clean.dropna()
    diff = clean.diff().abs().dropna()
    if valid.empty:
        start = end = ""
        mean = sd = val_range = mean_abs_delta = max_abs_delta = float("nan")
    else:
        idx = valid.index
        start = pd.to_datetime(dates.loc[idx[0]]).date().isoformat()
        end = pd.to_datetime(dates.loc[idx[-1]]).date().isoformat()
        mean = float(valid.mean())
        sd = float(valid.std(ddof=0))
        val_range = float(valid.max() - valid.min())
        mean_abs_delta = float(diff.mean()) if not diff.empty else 0.0
        max_abs_delta = float(diff.max()) if not diff.empty else 0.0
    pair = pd.concat([clean, obs], axis=1).dropna()
    rmse = float(np.sqrt(np.mean((pair.iloc[:, 0] - pair.iloc[:, 1]) ** 2))) if len(pair) else float("nan")
    return {
        "n": int(valid.size),
        "start": start,
        "end": end,
        "sm_hat_mean": mean,
        "sm_hat_sd": sd,
        "sm_hat_range": val_range,
        "mean_abs_delta": mean_abs_delta,
        "max_abs_delta": max_abs_delta,
        "corr_with_observed": _safe_corr(clean, obs),
        "rmse_vs_observed": rmse,
        "flat_flag": bool(
            np.isfinite(sd)
            and np.isfinite(val_range)
            and (sd <= FLAT_SD_TOL or val_range <= FLAT_RANGE_TOL)
        ),
    }


def evaluate_entropy_partition_region(
    frame: pd.DataFrame,
    *,
    region_key: str,
    region_label: str,
    input_source: str = "allocation_q",
    seed: int = 7_311,
    household_housing_share: float = 0.4,
    housing_construction_share: float = 0.35,
) -> pd.DataFrame:
    if "date" not in frame.columns:
        raise ValueError("allocation frame must include a date column")
    df = frame.copy(deep=True).sort_values("date").reset_index(drop=True).copy()
    df.loc[:, "date"] = pd.to_datetime(df["date"])
    rows: list[dict[str, Any]] = []
    for family in PARTITION_FAMILIES:
        for bucket_count in BUCKET_COUNTS:
            shares = build_partition_shares(
                df,
                family=family,
                bucket_count=bucket_count,
                household_housing_share=household_housing_share,
                housing_construction_share=housing_construction_share,
            )
            observed = _entropy_hat(shares)
            for control in CONTROLS:
                control_seed = _stable_seed(region_key, family, bucket_count, control, base_seed=seed)
                control_values = _entropy_hat(_control_shares(shares, control=control, seed=control_seed))
                metrics = _evaluate_series(control_values, observed=observed, dates=df["date"])
                rows.append(
                    {
                        "region_key": region_key,
                        "region_label": region_label,
                        "partition_family": family,
                        "bucket_count": int(bucket_count),
                        "control": control,
                        "negative_control": control != "observed",
                        "partition_input_source": input_source,
                        "source_q_columns": ",".join([col for col in df.columns if col.startswith("q_")]),
                        **metrics,
                    }
                )
    out = pd.DataFrame(rows).copy()
    observed_rows = out["control"].eq("observed")
    out.loc[:, "main_text_use"] = np.where(
        observed_rows & out["flat_flag"],
        "exclude_entropy_result",
        np.where(observed_rows, "eligible_as_auxiliary", "negative_control"),
    )
    return out


def summarize_entropy_robustness(results: pd.DataFrame) -> dict[str, Any]:
    observed = results[results["control"].eq("observed")].copy()
    region_summary: dict[str, Any] = {}
    for region_key, group in observed.groupby("region_key", sort=True):
        region_label = str(group["region_label"].iloc[0])
        flat_count = int(group["flat_flag"].sum())
        total = int(len(group))
        max_range = float(pd.to_numeric(group["sm_hat_range"], errors="coerce").max())
        max_sd = float(pd.to_numeric(group["sm_hat_sd"], errors="coerce").max())
        region_summary[region_key] = {
            "region_label": region_label,
            "observed_partitions": total,
            "flat_observed_partitions": flat_count,
            "exclude_entropy_from_main_text": bool(flat_count == total),
            "max_observed_range": max_range,
            "max_observed_sd": max_sd,
        }

    random_walk = results[results["control"].eq("random_walk_shares")].copy()
    rw_sd = pd.to_numeric(random_walk["sm_hat_sd"], errors="coerce")
    return {
        "flat_range_tolerance": FLAT_RANGE_TOL,
        "flat_sd_tolerance": FLAT_SD_TOL,
        "regions": region_summary,
        "overall": {
            "observed_partitions": int(len(observed)),
            "flat_observed_partitions": int(observed["flat_flag"].sum()),
            "all_observed_flat": bool(observed["flat_flag"].all()) if len(observed) else False,
            "max_random_walk_sd": float(rw_sd.max()) if rw_sd.notna().any() else float("nan"),
        },
        "interpretation": (
            "If all observed partitions for a region are flat, S_M_hat is treated "
            "as an allocation-table diagnostic rather than main empirical evidence."
        ),
    }


def run_entropy_partition_robustness(
    data_dir: Path,
    *,
    seed: int = 7_311,
) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    for region_key, (region_label, filename) in REGION_SOURCES.items():
        path = data_dir / filename
        if not path.exists():
            continue
        frame = pd.read_csv(path, parse_dates=["date"])
        input_source = "allocation_q"
        dest_filename = DESTINATION_SOURCES.get(region_key)
        dest_path = data_dir / dest_filename if dest_filename else None
        if dest_path and dest_path.exists():
            destination = pd.read_csv(dest_path, parse_dates=["date"])
            direct_cols = {"C_G", "C_B", "C_E"}
            if direct_cols.issubset(destination.columns):
                shares = destination[["date", "C_G", "C_B", "C_E"]].copy()
                shares.loc[:, "q_productive"] = pd.to_numeric(shares["C_G"], errors="coerce").clip(lower=0.0)
                shares.loc[:, "q_housing"] = pd.to_numeric(shares["C_B"], errors="coerce").clip(lower=0.0)
                shares.loc[:, "q_consumption"] = 0.0
                shares.loc[:, "q_financial"] = pd.to_numeric(shares["C_E"], errors="coerce").clip(lower=0.0)
                shares.loc[:, "q_government"] = 0.0
                frame = shares[
                    ["date", "q_productive", "q_housing", "q_consumption", "q_financial", "q_government"]
                ].copy()
                input_source = "credit_destination_jp"
        frames.append(
            evaluate_entropy_partition_region(
                frame,
                region_key=region_key,
                region_label=region_label,
                input_source=input_source,
                seed=seed,
            )
        )
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


def _fmt_num(value: Any, digits: int = 4) -> str:
    try:
        val = float(value)
    except Exception:
        return ""
    if not np.isfinite(val):
        return ""
    return f"{val:.{digits}f}"


def _latex_escape(value: Any) -> str:
    text = str(value)
    return (
        text.replace("\\", "\\textbackslash{}")
        .replace("_", "\\_")
        .replace("&", "\\&")
        .replace("%", "\\%")
    )


def render_entropy_partition_robustness_tex(results: pd.DataFrame) -> str:
    if results.empty:
        return "% Entropy partition robustness table unavailable.\n"
    observed = results[results["control"].eq("observed")].copy()
    rw = results[results["control"].eq("random_walk_shares")][
        ["region_key", "partition_family", "bucket_count", "sm_hat_sd"]
    ].rename(columns={"sm_hat_sd": "random_walk_sd"})
    table = observed.merge(rw, on=["region_key", "partition_family", "bucket_count"], how="left")
    table = table.sort_values(["region_label", "partition_family", "bucket_count"])

    lines = [
        "\\begin{table}[htbp]",
        "  \\centering",
        "  \\small",
        "  \\setlength{\\tabcolsep}{4pt}",
        "  \\caption{Entropy partition robustness for normalized allocation entropy.}",
        "  \\label{tab:entropy_partition_robustness}",
        "  \\resizebox{\\textwidth}{!}{%",
        "  \\begin{tabular}{@{}lllllll@{}}",
        "    \\toprule",
        "    Region & Partition & $K$ & Obs. mean & Obs. sd & RW-control sd & Decision \\\\",
        "    \\midrule",
    ]
    for _, row in table.iterrows():
        decision = "drop from main text" if bool(row["flat_flag"]) else "auxiliary only"
        family = str(row["partition_family"]).replace("_", " ")
        lines.append(
            "    "
            + " & ".join(
                [
                    _latex_escape(row["region_label"]),
                    _latex_escape(family),
                    str(int(row["bucket_count"])),
                    _fmt_num(row["sm_hat_mean"]),
                    _fmt_num(row["sm_hat_sd"], digits=6),
                    _fmt_num(row["random_walk_sd"], digits=6),
                    _latex_escape(decision),
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
            "The table recomputes $S_{M,\\mathrm{hat}}=H(q)/\\log K$ under borrower-label and "
            "loan-purpose partitions with 3, 5, and 7 buckets. Five- and seven-bucket variants "
            "use deterministic splits of currently coarse categories where direct sub-bucket data "
            "are unavailable. RW-control sd is the standard deviation from a random-walk share "
            "negative control using the same bucket count. Flat observed partitions are excluded "
            "from main empirical evidence.",
            "\\end{table}",
        ]
    )
    return "\n".join(lines) + "\n"


def write_entropy_partition_robustness_outputs(results: pd.DataFrame, *, root: Path) -> list[Path]:
    site_dir = root / "site"
    data_dir = root / "data"
    tex_dir = root / "tex" / "generated"
    site_dir.mkdir(parents=True, exist_ok=True)
    data_dir.mkdir(parents=True, exist_ok=True)
    tex_dir.mkdir(parents=True, exist_ok=True)

    csv_path = site_dir / "entropy_partition_robustness.csv"
    json_path = data_dir / "entropy_partition_robustness_summary.json"
    tex_path = tex_dir / "theory_entropy_partition_robustness.tex"

    results.to_csv(csv_path, index=False)
    summary = summarize_entropy_robustness(results)
    json_path.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    tex_path.write_text(render_entropy_partition_robustness_tex(results), encoding="utf-8")
    return [csv_path, json_path, tex_path]
