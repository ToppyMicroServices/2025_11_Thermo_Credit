from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import pandas as pd

from lib.temperature import EPS, expanding_zscore


REGION_PANELS = {
    "jp": ("Japan (JP)", "indicators_realtime.csv"),
    "eu": ("Euro Area (EU)", "indicators_eu_realtime.csv"),
    "us": ("United States (US)", "indicators_us_realtime.csv"),
}

VARIANT_COLUMNS = {
    "multiplicative": "TL_multiplicative",
    "additive_zscore": "TL_additive_zscore",
    "soft_min": "TL_soft_min",
    "harmonic_mean": "TL_harmonic_mean",
    "spread_only": "TL_spread_only",
    "turnover_excluded": "TL_turnover_excluded",
    "depth_excluded": "TL_depth_excluded",
}

VARIANT_LABELS = {
    "multiplicative": "multiplicative",
    "additive_zscore": "additive z-score",
    "soft_min": "soft-min",
    "harmonic_mean": "harmonic mean",
    "spread_only": "spread-only",
    "turnover_excluded": "turnover-excluded",
    "depth_excluded": "depth-excluded",
}

VARIANT_INPUTS = {
    "multiplicative": ("spread", "depth", "turnover"),
    "additive_zscore": ("spread", "depth", "turnover"),
    "soft_min": ("spread", "depth", "turnover"),
    "harmonic_mean": ("spread", "depth", "turnover"),
    "spread_only": ("spread",),
    "turnover_excluded": ("spread", "depth"),
    "depth_excluded": ("spread", "turnover"),
}


def _positive_log(series: pd.Series) -> pd.Series:
    numeric = pd.to_numeric(series, errors="coerce").astype(float)
    return np.log(numeric.clip(lower=EPS))


def _logistic(series: pd.Series) -> pd.Series:
    x = pd.to_numeric(series, errors="coerce").astype(float).clip(lower=-40.0, upper=40.0)
    return 1.0 / (1.0 + np.exp(-x))


def _component_zscores(frame: pd.DataFrame, *, min_periods: int = 8) -> pd.DataFrame:
    out = pd.DataFrame(index=frame.index)
    if "spread" in frame.columns:
        out["spread"] = -expanding_zscore(_positive_log(frame["spread"]), min_periods=min_periods)
    if "depth" in frame.columns:
        out["depth"] = expanding_zscore(_positive_log(frame["depth"]), min_periods=min_periods)
    if "turnover" in frame.columns:
        out["turnover"] = expanding_zscore(_positive_log(frame["turnover"]), min_periods=min_periods)
    return out


def _require_columns(frame: pd.DataFrame, columns: tuple[str, ...]) -> pd.DataFrame | None:
    if not set(columns).issubset(frame.columns):
        return None
    return frame.loc[:, list(columns)].apply(pd.to_numeric, errors="coerce")


def _geometric_mean(frame: pd.DataFrame, columns: tuple[str, ...]) -> pd.Series:
    values = _require_columns(frame, columns)
    if values is None:
        return pd.Series(np.nan, index=frame.index)
    arr = values.to_numpy(dtype=float)
    valid = np.isfinite(arr).all(axis=1)
    out = np.full(len(values), np.nan)
    out[valid] = np.exp(np.log(np.clip(arr[valid], EPS, None)).mean(axis=1))
    return pd.Series(out, index=frame.index)


def _harmonic_mean(frame: pd.DataFrame, columns: tuple[str, ...]) -> pd.Series:
    values = _require_columns(frame, columns)
    if values is None:
        return pd.Series(np.nan, index=frame.index)
    arr = values.to_numpy(dtype=float)
    valid = np.isfinite(arr).all(axis=1)
    out = np.full(len(values), np.nan)
    clipped = np.clip(arr[valid], EPS, None)
    out[valid] = len(columns) / (1.0 / clipped).sum(axis=1)
    return pd.Series(out, index=frame.index)


def _soft_min(frame: pd.DataFrame, columns: tuple[str, ...], *, tau: float = 0.15) -> pd.Series:
    values = _require_columns(frame, columns)
    if values is None:
        return pd.Series(np.nan, index=frame.index)
    arr = values.to_numpy(dtype=float)
    valid = np.isfinite(arr).all(axis=1)
    out = np.full(len(values), np.nan)
    local = arr[valid]
    # Normalized log-sum-exp soft minimum: equal inputs return that same input.
    out[valid] = -tau * np.log(np.exp(-local / tau).mean(axis=1))
    return pd.Series(out, index=frame.index)


def compute_tl_variants(frame: pd.DataFrame, *, min_periods: int = 8) -> pd.DataFrame:
    """Compute no-lookahead liquidity-state alternatives from spread/depth/turnover.

    All variants are oriented so higher means more liquid: lower spreads, deeper
    markets, and higher turnover.
    """
    df = frame.copy()
    if "date" in df.columns:
        df = df.assign(date=pd.to_datetime(df["date"], errors="coerce")).sort_values("date").reset_index(drop=True)
    out = df[["date"]].copy() if "date" in df.columns else pd.DataFrame(index=df.index)
    z = _component_zscores(df, min_periods=min_periods)
    scores = z.apply(_logistic)

    all_cols = ("spread", "depth", "turnover")
    out["TL_additive_zscore"] = (
        _logistic(z.loc[:, list(all_cols)].mean(axis=1)) if set(all_cols).issubset(z.columns) else np.nan
    )
    out["TL_multiplicative"] = _geometric_mean(scores, all_cols)
    out["TL_soft_min"] = _soft_min(scores, all_cols)
    out["TL_harmonic_mean"] = _harmonic_mean(scores, all_cols)
    out["TL_spread_only"] = _logistic(z["spread"]) if "spread" in z.columns else np.nan
    out["TL_turnover_excluded"] = (
        _logistic(z.loc[:, ["spread", "depth"]].mean(axis=1))
        if {"spread", "depth"}.issubset(z.columns)
        else np.nan
    )
    out["TL_depth_excluded"] = (
        _logistic(z.loc[:, ["spread", "turnover"]].mean(axis=1))
        if {"spread", "turnover"}.issubset(z.columns)
        else np.nan
    )

    for component in ("spread", "depth", "turnover"):
        if component in z.columns:
            out[f"TL_z_{component}"] = z[component]
            out[f"TL_component_{component}"] = scores[component]
    return out


def legacy_signed_multiplicative_score(
    spread_z: float | np.ndarray,
    depth_z: float | np.ndarray,
    turnover_z: float | np.ndarray,
) -> float | np.ndarray:
    """Diagnostic for the rejected signed-product liquidity formula."""
    return np.asarray(spread_z) * np.asarray(depth_z) * (1.0 + 0.5 * np.asarray(turnover_z))


def legacy_signed_product_violation_count() -> int:
    grid = np.array([-2.0, -1.0, 0.0, 1.0, 2.0])
    violations = 0
    for spread_z in grid:
        for depth_z in grid:
            for turnover_z in grid:
                base = float(legacy_signed_multiplicative_score(spread_z, depth_z, turnover_z))
                for delta in ((1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0)):
                    improved = float(
                        legacy_signed_multiplicative_score(
                            spread_z + delta[0],
                            depth_z + delta[1],
                            turnover_z + delta[2],
                        )
                    )
                    if improved < base - 1e-12:
                        violations += 1
    return violations


def _synthetic_liquidity_frame(
    *,
    spread_last: float = 1.0,
    depth_last: float = 1_000.0,
    turnover_last: float = 1.0,
    n_history: int = 24,
) -> pd.DataFrame:
    dates = pd.date_range("2010-03-31", periods=n_history + 1, freq="QE-DEC")
    return pd.DataFrame(
        {
            "date": dates,
            "spread": np.r_[np.linspace(0.9, 1.1, n_history), spread_last],
            "depth": np.r_[np.linspace(950.0, 1_050.0, n_history), depth_last],
            "turnover": np.r_[np.linspace(0.9, 1.1, n_history), turnover_last],
        }
    )


def run_tl_monotonicity_checks() -> dict[str, dict[str, Any]]:
    scenarios = {
        "base": _synthetic_liquidity_frame(),
        "lower_spread": _synthetic_liquidity_frame(spread_last=0.45),
        "higher_depth": _synthetic_liquidity_frame(depth_last=1_450.0),
        "higher_turnover": _synthetic_liquidity_frame(turnover_last=1.75),
        "all_bad": _synthetic_liquidity_frame(spread_last=1.9, depth_last=650.0, turnover_last=0.45),
        "all_good": _synthetic_liquidity_frame(spread_last=0.45, depth_last=1_450.0, turnover_last=1.75),
    }
    values: dict[str, dict[str, float]] = {}
    for scenario, frame in scenarios.items():
        scores = compute_tl_variants(frame)
        values[scenario] = {
            variant: float(scores[column].iloc[-1])
            for variant, column in VARIANT_COLUMNS.items()
            if column in scores.columns
        }

    checks: dict[str, dict[str, Any]] = {}
    for variant in VARIANT_COLUMNS:
        base = values["base"][variant]
        deltas = {
            "lower_spread_delta": values["lower_spread"][variant] - base,
            "higher_depth_delta": values["higher_depth"][variant] - base,
            "higher_turnover_delta": values["higher_turnover"][variant] - base,
            "all_good_minus_all_bad": values["all_good"][variant] - values["all_bad"][variant],
        }
        checks[variant] = {
            **{key: float(val) for key, val in deltas.items()},
            "monotone_pass": bool(
                deltas["lower_spread_delta"] >= -1e-12
                and deltas["higher_depth_delta"] >= -1e-12
                and deltas["higher_turnover_delta"] >= -1e-12
                and deltas["all_good_minus_all_bad"] > 0
            ),
        }

    checks["legacy_signed_product"] = {
        "violations": legacy_signed_product_violation_count(),
        "monotone_pass": False,
        "note": "Rejected diagnostic: signed raw-product z-score formula is not coordinate-wise monotone.",
    }
    return checks


def _safe_corr(left: pd.Series, right: pd.Series, *, method: str = "pearson") -> float:
    pair = pd.concat([left, right], axis=1).dropna()
    if len(pair) < 3:
        return float("nan")
    if method == "spearman":
        pair = pair.rank(method="average")
    if pair.iloc[:, 0].std(ddof=0) <= 0 or pair.iloc[:, 1].std(ddof=0) <= 0:
        return float("nan")
    return float(pair.iloc[:, 0].corr(pair.iloc[:, 1]))


def _low_liquidity_agreement(left: pd.Series, right: pd.Series) -> float:
    pair = pd.concat([left, right], axis=1).dropna()
    if len(pair) < 8:
        return float("nan")
    l_thr = pair.iloc[:, 0].quantile(0.25)
    r_thr = pair.iloc[:, 1].quantile(0.25)
    return float((pair.iloc[:, 0].le(l_thr) == pair.iloc[:, 1].le(r_thr)).mean())


def evaluate_tl_robustness_region(
    frame: pd.DataFrame,
    *,
    region_key: str,
    region_label: str,
    panel_source: str,
    monotonicity: Mapping[str, Mapping[str, Any]] | None = None,
) -> pd.DataFrame:
    if "date" not in frame.columns:
        raise ValueError("TL robustness frame must include a date column")
    df = frame.copy().sort_values("date").reset_index(drop=True)
    df.loc[:, "date"] = pd.to_datetime(df["date"])
    variants = compute_tl_variants(df)
    current = pd.to_numeric(df["T_L"], errors="coerce") if "T_L" in df.columns else variants["TL_additive_zscore"]
    additive = variants["TL_additive_zscore"]
    checks = dict(monotonicity or run_tl_monotonicity_checks())

    rows: list[dict[str, Any]] = []
    for variant, column in VARIANT_COLUMNS.items():
        series = pd.to_numeric(variants[column], errors="coerce")
        valid = series.dropna()
        start = ""
        end = ""
        latest = float("nan")
        if not valid.empty:
            start = pd.to_datetime(variants.loc[valid.index[0], "date"]).date().isoformat()
            end = pd.to_datetime(variants.loc[valid.index[-1], "date"]).date().isoformat()
            latest = float(valid.iloc[-1])
        pair_current = pd.concat([series, current], axis=1).dropna()
        rmse_vs_current = (
            float(np.sqrt(np.mean((pair_current.iloc[:, 0] - pair_current.iloc[:, 1]) ** 2)))
            if len(pair_current)
            else float("nan")
        )
        pair_add = pd.concat([series, additive], axis=1).dropna()
        mean_abs_diff = (
            float((pair_add.iloc[:, 0] - pair_add.iloc[:, 1]).abs().mean()) if len(pair_add) else float("nan")
        )
        check = checks.get(variant, {})
        rows.append(
            {
                "region_key": region_key,
                "region_label": region_label,
                "panel_source": panel_source,
                "variant": variant,
                "variant_label": VARIANT_LABELS[variant],
                "inputs": ",".join(VARIANT_INPUTS[variant]),
                "n": int(valid.size),
                "start": start,
                "end": end,
                "latest": latest,
                "mean": float(valid.mean()) if not valid.empty else float("nan"),
                "sd": float(valid.std(ddof=0)) if not valid.empty else float("nan"),
                "range": float(valid.max() - valid.min()) if not valid.empty else float("nan"),
                "corr_vs_additive": _safe_corr(series, additive),
                "rank_corr_vs_additive": _safe_corr(series, additive, method="spearman"),
                "mean_abs_diff_vs_additive": mean_abs_diff,
                "low_liquidity_agreement_vs_additive": _low_liquidity_agreement(series, additive),
                "rmse_vs_current_T_L": rmse_vs_current,
                "corr_vs_spread": _safe_corr(series, pd.to_numeric(df.get("spread"), errors="coerce")),
                "corr_vs_depth": _safe_corr(series, pd.to_numeric(df.get("depth"), errors="coerce")),
                "corr_vs_turnover": _safe_corr(series, pd.to_numeric(df.get("turnover"), errors="coerce")),
                "lower_spread_delta": float(check.get("lower_spread_delta", np.nan)),
                "higher_depth_delta": float(check.get("higher_depth_delta", np.nan)),
                "higher_turnover_delta": float(check.get("higher_turnover_delta", np.nan)),
                "all_good_minus_all_bad": float(check.get("all_good_minus_all_bad", np.nan)),
                "monotone_pass": bool(check.get("monotone_pass", False)),
                "main_spec_status": "main_specification" if variant == "additive_zscore" else "robustness_check",
            }
        )
    return pd.DataFrame(rows)


def run_tl_robustness(site_dir: Path) -> pd.DataFrame:
    monotonicity = run_tl_monotonicity_checks()
    frames: list[pd.DataFrame] = []
    for region_key, (region_label, filename) in REGION_PANELS.items():
        path = site_dir / filename
        if not path.exists() and region_key == "jp":
            path = site_dir / "indicators.csv"
        if not path.exists():
            continue
        frame = pd.read_csv(path, parse_dates=["date"])
        frames.append(
            evaluate_tl_robustness_region(
                frame,
                region_key=region_key,
                region_label=region_label,
                panel_source=path.relative_to(site_dir.parent).as_posix(),
                monotonicity=monotonicity,
            )
        )
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


def summarize_tl_robustness(results: pd.DataFrame) -> dict[str, Any]:
    checks = run_tl_monotonicity_checks()
    region_summary: dict[str, Any] = {}
    for region_key, group in results.groupby("region_key", sort=True):
        label = str(group["region_label"].iloc[0])
        additive_row = group[group["variant"].eq("additive_zscore")]
        min_corr = pd.to_numeric(group["corr_vs_additive"], errors="coerce").dropna()
        min_agree = pd.to_numeric(group["low_liquidity_agreement_vs_additive"], errors="coerce").dropna()
        region_summary[region_key] = {
            "region_label": label,
            "variants": int(len(group)),
            "additive_latest": float(additive_row["latest"].iloc[0]) if not additive_row.empty else float("nan"),
            "minimum_corr_vs_additive": float(min_corr.min()) if not min_corr.empty else float("nan"),
            "minimum_low_liquidity_agreement": float(min_agree.min()) if not min_agree.empty else float("nan"),
            "all_requested_variants_monotone": bool(group["monotone_pass"].all()),
        }
    return {
        "regions": region_summary,
        "requested_variants": list(VARIANT_COLUMNS.keys()),
        "all_requested_variants_monotone": bool(results["monotone_pass"].all()) if not results.empty else False,
        "legacy_signed_product": checks["legacy_signed_product"],
        "interpretation": (
            "The additive z-score liquidity index remains the main specification. "
            "The rejected signed raw-product diagnostic has monotonicity violations; "
            "monotone multiplicative, soft-min, harmonic, and exclusion variants are robustness checks."
        ),
    }


def _fmt(value: Any, digits: int = 3) -> str:
    try:
        val = float(value)
    except Exception:
        return ""
    if not np.isfinite(val):
        return ""
    return f"{val:.{digits}f}"


def _latex_escape(value: Any) -> str:
    return str(value).replace("_", "\\_").replace("&", "\\&").replace("%", "\\%")


def render_tl_robustness_tex(results: pd.DataFrame) -> str:
    if results.empty:
        return "% TL robustness table unavailable.\n"
    table = results.sort_values(["region_label", "variant"])
    lines = [
        "\\begin{table}[htbp]",
        "  \\centering",
        "  \\small",
        "  \\setlength{\\tabcolsep}{3pt}",
        "  \\caption{Liquidity-state index robustness.}",
        "  \\label{tab:tl_robustness}",
        "  \\resizebox{\\textwidth}{!}{%",
        "  \\begin{tabular}{@{}llllllll@{}}",
        "    \\toprule",
        "    Region & Variant & Inputs & $N$ & Latest & SD & Corr. vs add. & Low-liq. agree \\\\",
        "    \\midrule",
    ]
    for _, row in table.iterrows():
        lines.append(
            "    "
            + " & ".join(
                [
                    _latex_escape(row["region_label"]),
                    _latex_escape(row["variant_label"]),
                    _latex_escape(row["inputs"]),
                    str(int(row["n"])),
                    _fmt(row["latest"]),
                    _fmt(row["sd"]),
                    _fmt(row["corr_vs_additive"]),
                    _fmt(row["low_liquidity_agreement_vs_additive"]),
                ]
            )
            + " \\\\"
        )
    legacy = run_tl_monotonicity_checks()["legacy_signed_product"]
    lines.extend(
        [
            "    \\bottomrule",
            "  \\end{tabular}",
            "  }",
            "  \\par\\smallskip\\raggedright\\footnotesize "
            "All reported variants use expanding-window log z-scores and are oriented so that lower "
            "spreads, deeper markets, and higher turnover raise the index. The additive z-score "
            "variant is the main specification. Multiplicative means the geometric mean of monotone "
            "logistic component scores, not the rejected signed raw-product formula. The signed "
            f"raw-product diagnostic has {int(legacy['violations'])} grid monotonicity violations "
            "and is therefore not used as a main specification.",
            "\\end{table}",
        ]
    )
    return "\n".join(lines) + "\n"


def write_tl_robustness_outputs(results: pd.DataFrame, *, root: Path) -> list[Path]:
    site_dir = root / "site"
    data_dir = root / "data"
    tex_dir = root / "tex" / "generated"
    site_dir.mkdir(parents=True, exist_ok=True)
    data_dir.mkdir(parents=True, exist_ok=True)
    tex_dir.mkdir(parents=True, exist_ok=True)

    csv_path = site_dir / "tl_robustness.csv"
    json_path = data_dir / "tl_robustness_summary.json"
    tex_path = tex_dir / "theory_tl_robustness.tex"

    results.to_csv(csv_path, index=False)
    json_path.write_text(json.dumps(summarize_tl_robustness(results), indent=2, sort_keys=True), encoding="utf-8")
    tex_path.write_text(render_tl_robustness_tex(results), encoding="utf-8")
    return [csv_path, json_path, tex_path]
