from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import pandas as pd

from lib.theory_figures import RegionFrame, load_region_frames


LAMBDA_GRID: Sequence[float] = (0.0, 0.25, 0.5, 0.75, 1.0)
MIN_TRAINING_ROWS = 24


@dataclass(frozen=True)
class TargetSpec:
    key: str
    label: str
    source_column: str
    series: pd.Series


def _normal_cdf(value: float) -> float:
    import math

    return 0.5 * math.erfc(-value / math.sqrt(2.0))


def _safe_numeric(frame: pd.DataFrame, column: str, *, default: float = np.nan) -> pd.Series:
    if column not in frame.columns:
        return pd.Series(default, index=frame.index, dtype=float)
    return pd.to_numeric(frame[column], errors="coerce")


def _pick_numeric_column(frame: pd.DataFrame, candidates: Sequence[str], min_rows: int = 12) -> str | None:
    for column in candidates:
        if column not in frame.columns:
            continue
        series = pd.to_numeric(frame[column], errors="coerce")
        if series.dropna().size >= min_rows and float(series.dropna().std(ddof=0)) > 1e-12:
            return column
    return None


def _positive_log(series: pd.Series) -> pd.Series:
    numeric = pd.to_numeric(series, errors="coerce")
    return pd.Series(np.log(numeric.where(numeric > 0)), index=series.index, dtype=float)


def _forward_log_growth(level: pd.Series, horizon: int) -> pd.Series:
    logged = _positive_log(level)
    return logged.shift(-horizon) - logged


def _forward_log_acceleration(level: pd.Series, horizon: int) -> pd.Series:
    logged = _positive_log(level)
    future = logged.shift(-horizon) - logged
    trailing = logged - logged.shift(horizon)
    return future - trailing


def _first_valid(series: pd.Series) -> float:
    valid = pd.to_numeric(series, errors="coerce").dropna()
    if valid.empty:
        return float("nan")
    return float(valid.iloc[-1])


def _mean_valid(series: pd.Series) -> float:
    valid = pd.to_numeric(series, errors="coerce").dropna()
    if valid.empty:
        return float("nan")
    return float(valid.mean())


def _last_text(series: pd.Series, default: str = "") -> str:
    valid = series.dropna().astype(str)
    if valid.empty:
        return default
    return str(valid.iloc[-1])


def _auc_score(scores: Sequence[float], events: Sequence[int]) -> float:
    positives = [score for score, event in zip(scores, events) if event == 1 and np.isfinite(score)]
    negatives = [score for score, event in zip(scores, events) if event == 0 and np.isfinite(score)]
    if not positives or not negatives:
        return float("nan")
    total = 0.0
    for pos in positives:
        for neg in negatives:
            if pos > neg:
                total += 1.0
            elif pos == neg:
                total += 0.5
    return float(total / (len(positives) * len(negatives)))


def _linear_predict(train_score: pd.Series, train_target: pd.Series, test_score: float) -> tuple[float, float, float, float]:
    pair = pd.concat(
        [
            pd.to_numeric(train_score, errors="coerce"),
            pd.to_numeric(train_target, errors="coerce"),
        ],
        axis=1,
    ).dropna()
    if pair.shape[0] < 8 or not np.isfinite(test_score):
        return float("nan"), float("nan"), float("nan"), float("nan")
    x = pair.iloc[:, 0]
    y = pair.iloc[:, 1]
    y_mean = float(y.mean())
    y_std = float(y.std(ddof=0))
    if not np.isfinite(y_std) or y_std <= 1e-12:
        return float("nan"), float("nan"), float("nan"), float(pair.shape[0])
    x_std = float(x.std(ddof=0))
    if not np.isfinite(x_std) or x_std <= 1e-12:
        pred = y_mean
    else:
        slope, intercept = np.polyfit(x.to_numpy(dtype=float), y.to_numpy(dtype=float), deg=1)
        pred = float(intercept + slope * test_score)
    pred_z = (pred - y_mean) / y_std
    threshold_z = (0.0 - y_mean) / y_std
    return float(pred_z), float(threshold_z), y_mean, float(pair.shape[0])


def _expanding_oos_metrics(
    score: pd.Series,
    baseline_score: pd.Series,
    target: pd.Series,
    *,
    min_training_rows: int = MIN_TRAINING_ROWS,
) -> dict[str, float]:
    tuned_preds: list[float] = []
    baseline_preds: list[float] = []
    actuals: list[float] = []
    thresholds: list[float] = []

    for idx in range(min_training_rows, len(target)):
        actual = float(target.iloc[idx]) if idx < len(target) and np.isfinite(target.iloc[idx]) else float("nan")
        if not np.isfinite(actual):
            continue
        train_target = target.iloc[:idx]
        tuned_pred, threshold_z, y_mean, _ = _linear_predict(score.iloc[:idx], train_target, float(score.iloc[idx]))
        baseline_pred, _, _, _ = _linear_predict(
            baseline_score.iloc[:idx],
            train_target,
            float(baseline_score.iloc[idx]),
        )
        if not (np.isfinite(tuned_pred) and np.isfinite(baseline_pred) and np.isfinite(threshold_z)):
            continue
        y_std = float(pd.to_numeric(train_target, errors="coerce").dropna().std(ddof=0))
        if not np.isfinite(y_std) or y_std <= 1e-12:
            continue
        actuals.append(float((actual - y_mean) / y_std))
        tuned_preds.append(float(tuned_pred))
        baseline_preds.append(float(baseline_pred))
        thresholds.append(float(threshold_z))

    n = len(actuals)
    if n < 8:
        return {
            "n": float(n),
            "rmse": float("nan"),
            "mae": float("nan"),
            "baseline_rmse": float("nan"),
            "baseline_mae": float("nan"),
            "auc": float("nan"),
            "log_score": float("nan"),
            "dm_stat": float("nan"),
            "dm_p": float("nan"),
        }

    actual_arr = np.asarray(actuals, dtype=float)
    tuned_arr = np.asarray(tuned_preds, dtype=float)
    baseline_arr = np.asarray(baseline_preds, dtype=float)
    threshold_arr = np.asarray(thresholds, dtype=float)
    tuned_errors = tuned_arr - actual_arr
    baseline_errors = baseline_arr - actual_arr
    loss_diff = tuned_errors**2 - baseline_errors**2
    if n >= 8 and float(np.std(loss_diff, ddof=1)) > 1e-12:
        dm_stat = float(np.mean(loss_diff) / (np.std(loss_diff, ddof=1) / np.sqrt(n)))
        dm_p = float(2.0 * (1.0 - _normal_cdf(abs(dm_stat))))
    else:
        dm_stat = float("nan")
        dm_p = float("nan")

    events = (actual_arr > threshold_arr).astype(int)
    event_scores = tuned_arr - threshold_arr
    probs = 1.0 / (1.0 + np.exp(-np.clip(event_scores, -30, 30)))
    probs = np.clip(probs, 1e-6, 1.0 - 1e-6)
    log_score = float(-np.mean(events * np.log(probs) + (1 - events) * np.log(1 - probs)))

    return {
        "n": float(n),
        "rmse": float(np.sqrt(np.mean(tuned_errors**2))),
        "mae": float(np.mean(np.abs(tuned_errors))),
        "baseline_rmse": float(np.sqrt(np.mean(baseline_errors**2))),
        "baseline_mae": float(np.mean(np.abs(baseline_errors))),
        "auc": _auc_score(event_scores.tolist(), events.tolist()),
        "log_score": log_score,
        "dm_stat": dm_stat,
        "dm_p": dm_p,
    }


def lambda_destination_panel(frame: pd.DataFrame, lambda_b: float) -> pd.DataFrame:
    out = pd.DataFrame({"date": pd.to_datetime(frame["date"], errors="coerce")})
    c_t = _safe_numeric(frame, "C_t", default=0.0)
    c_g = _safe_numeric(frame, "C_G", default=0.0).fillna(0.0)
    c_b = _safe_numeric(frame, "C_B", default=0.0).fillna(0.0)
    c_e = _safe_numeric(frame, "C_E", default=0.0).fillna(0.0)
    c_r = c_g + float(lambda_b) * c_b
    c_a = c_e + (1.0 - float(lambda_b)) * c_b
    q_t = pd.Series(np.where(c_t > 0, c_r / c_t, np.nan), index=frame.index, dtype=float)
    out["lambda_B"] = float(lambda_b)
    out["C_t"] = c_t
    out["C_G"] = c_g
    out["C_B"] = c_b
    out["C_E"] = c_e
    out["C_R"] = c_r
    out["C_A"] = c_a
    out["q_t"] = q_t
    out["one_minus_q_t"] = np.where(c_t > 0, 1.0 - q_t, np.nan)
    for column in ("destination_coverage", "credit_destination_source", "preprocessing_mode", "release_lag_profile"):
        if column in frame.columns:
            out[column] = frame[column]
    return out


def _target_specs(frame: pd.DataFrame, horizon: int) -> list[TargetSpec]:
    specs: list[TargetSpec] = []
    growth_col = _pick_numeric_column(frame, ("Y", "U_gdp_only", "U", "L_real"))
    if growth_col:
        specs.append(
            TargetSpec(
                key="real_growth",
                label="real growth",
                source_column=growth_col,
                series=_forward_log_growth(frame[growth_col], horizon),
            )
        )
    asset_col = _pick_numeric_column(frame, ("asset_price", "A", "L_asset", "L_asset_toy"))
    if asset_col:
        specs.append(
            TargetSpec(
                key="asset_acceleration",
                label="asset acceleration",
                source_column=asset_col,
                series=_forward_log_acceleration(frame[asset_col], horizon),
            )
        )
    return specs


def _scale_series(frame: pd.DataFrame, target: TargetSpec) -> pd.Series:
    if target.key == "asset_acceleration":
        col = _pick_numeric_column(frame, ("asset_price", "A", "L_asset", "L_asset_toy")) or target.source_column
    else:
        col = _pick_numeric_column(frame, ("Y", "U_gdp_only", "U", "L_real")) or target.source_column
    scale = pd.to_numeric(frame[col], errors="coerce").where(lambda s: s > 0)
    fallback = float(scale.dropna().median()) if not scale.dropna().empty else 1.0
    return scale.fillna(fallback).replace(0.0, fallback)


def _score_for_target(panel: pd.DataFrame, target: TargetSpec, scale: pd.Series) -> pd.Series:
    if target.key == "asset_acceleration":
        numerator = pd.to_numeric(panel["C_A"], errors="coerce")
    else:
        numerator = pd.to_numeric(panel["C_R"], errors="coerce")
    return numerator / scale


def sensitivity_rows_for_region(region: RegionFrame, lambda_grid: Sequence[float], *, horizon: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    frame = region.frame.copy().sort_values("date").reset_index(drop=True)
    targets = _target_specs(frame, horizon)
    metric_rows: list[dict[str, Any]] = []
    panel_rows: list[pd.DataFrame] = []
    for lambda_b in lambda_grid:
        panel = lambda_destination_panel(frame, float(lambda_b))
        panel.insert(0, "region_key", region.key)
        panel.insert(1, "region_label", region.label)
        panel["panel_source"] = region.source_path
        panel_rows.append(panel)
        c_t = pd.to_numeric(panel["C_t"], errors="coerce").fillna(0.0)
        for target in targets:
            scale = _scale_series(frame, target)
            score = _score_for_target(panel, target, scale)
            baseline_score = c_t / scale
            metrics = _expanding_oos_metrics(score, baseline_score, target.series)
            metric_rows.append(
                {
                    "region_key": region.key,
                    "region_label": region.label,
                    "panel_source": region.source_path,
                    "preprocessing_mode": _last_text(
                        frame.get("preprocessing_mode", pd.Series(index=frame.index, dtype=object))
                    ),
                    "lambda_B": float(lambda_b),
                    "horizon_quarters": int(horizon),
                    "target": target.key,
                    "target_label": target.label,
                    "target_source": target.source_column,
                    "predictor": "C_A / asset_scale" if target.key == "asset_acceleration" else "C_R / activity_scale",
                    "baseline": "C_t / same_scale",
                    "latest_q_t": _first_valid(panel["q_t"]),
                    "mean_q_t": _mean_valid(panel["q_t"]),
                    "latest_destination_coverage": _first_valid(panel.get("destination_coverage", pd.Series(index=panel.index, dtype=float))),
                    **metrics,
                }
            )
    metrics_df = pd.DataFrame(metric_rows)
    panel_df = pd.concat(panel_rows, ignore_index=True) if panel_rows else pd.DataFrame()
    return metrics_df, panel_df


def run_lambda_sensitivity(
    site_dir: Path,
    *,
    source_ref: str | None = None,
    panel_mode: str = "realtime",
    lambda_grid: Sequence[float] = LAMBDA_GRID,
    horizon: int = 4,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    metric_parts: list[pd.DataFrame] = []
    panel_parts: list[pd.DataFrame] = []
    for region in load_region_frames(site_dir, source_ref=source_ref, mode=panel_mode):
        metrics, panel = sensitivity_rows_for_region(region, lambda_grid, horizon=horizon)
        if not metrics.empty:
            metric_parts.append(metrics)
        if not panel.empty:
            panel_parts.append(panel)
    metrics_df = pd.concat(metric_parts, ignore_index=True) if metric_parts else pd.DataFrame()
    panel_df = pd.concat(panel_parts, ignore_index=True) if panel_parts else pd.DataFrame()
    summary_df = summarize_lambda_sensitivity(metrics_df)
    return metrics_df, panel_df, summary_df


def summarize_lambda_sensitivity(metrics: pd.DataFrame) -> pd.DataFrame:
    if metrics.empty:
        return pd.DataFrame()
    rows: list[dict[str, Any]] = []
    group_cols = ["region_key", "region_label", "target", "target_label", "target_source"]
    for keys, group in metrics.groupby(group_cols, dropna=False):
        region_key, region_label, target, target_label, target_source = keys
        valid = group[pd.to_numeric(group["rmse"], errors="coerce").notna()].copy()
        valid = valid[pd.to_numeric(valid["n"], errors="coerce") >= 8]
        if valid.empty:
            rows.append(
                {
                    "region_key": region_key,
                    "region_label": region_label,
                    "target": target,
                    "target_label": target_label,
                    "target_source": target_source,
                    "best_lambda_B": float("nan"),
                    "best_rmse": float("nan"),
                    "baseline_rmse": float("nan"),
                    "rmse_range": float("nan"),
                    "rmse_range_pct": float("nan"),
                    "best_improvement_vs_total_credit": float("nan"),
                    "improvement_sign_flip": False,
                    "lambda_sensitive": True,
                    "target_unstable": True,
                    "main_claim_use": "insufficient coverage",
                }
            )
            continue
        best_idx = pd.to_numeric(valid["rmse"], errors="coerce").idxmin()
        best = valid.loc[best_idx]
        rmse = pd.to_numeric(valid["rmse"], errors="coerce")
        baseline = pd.to_numeric(valid["baseline_rmse"], errors="coerce")
        rmse_range = float(rmse.max() - rmse.min())
        best_rmse = float(best["rmse"])
        rmse_range_pct = float(rmse_range / max(abs(best_rmse), 1e-12))
        improvement = baseline - rmse
        improvement_tol = max(abs(best_rmse), 1.0) * 1e-8
        has_better = bool((improvement > improvement_tol).any())
        has_worse = bool((improvement < -improvement_tol).any())
        sign_flip = has_better and has_worse
        lambda_sensitive = bool(sign_flip or rmse_range_pct > 0.10)
        best_improvement = float(best["baseline_rmse"] - best["rmse"])
        target_unstable = bool(best_rmse > 5.0)
        if target_unstable or lambda_sensitive:
            main_claim_use = "do not use as main claim"
        elif best_improvement <= improvement_tol:
            main_claim_use = "lambda-robust; no gain"
        else:
            main_claim_use = "robust within grid"
        rows.append(
            {
                "region_key": region_key,
                "region_label": region_label,
                "target": target,
                "target_label": target_label,
                "target_source": target_source,
                "best_lambda_B": float(best["lambda_B"]),
                "best_rmse": best_rmse,
                "baseline_rmse": float(best["baseline_rmse"]),
                "rmse_range": rmse_range,
                "rmse_range_pct": rmse_range_pct,
                "best_improvement_vs_total_credit": best_improvement,
                "improvement_sign_flip": sign_flip,
                "lambda_sensitive": lambda_sensitive,
                "target_unstable": target_unstable,
                "main_claim_use": main_claim_use,
            }
        )
    return pd.DataFrame(rows)


def _format_float(value: Any, digits: int = 3) -> str:
    try:
        numeric = float(value)
    except Exception:
        return "n/a"
    if not np.isfinite(numeric):
        return "n/a"
    return f"{numeric:.{digits}f}"


def _format_lambda_read(row: Mapping[str, Any]) -> str:
    try:
        rmse_range = float(row.get("rmse_range", float("nan")))
    except Exception:
        rmse_range = float("nan")
    if np.isfinite(rmse_range) and rmse_range <= 1e-8:
        return "flat"
    return _format_float(row.get("best_lambda_B", float("nan")), 2)


def _latex_escape(text: str) -> str:
    replacements = {
        "\\": r"\textbackslash{}",
        "&": r"\&",
        "%": r"\%",
        "$": r"\$",
        "#": r"\#",
        "_": r"\_",
        "{": r"\{",
        "}": r"\}",
    }
    out = str(text)
    for src, dst in replacements.items():
        out = out.replace(src, dst)
    return out


def render_lambda_sensitivity_tex(summary: pd.DataFrame) -> str:
    tex_summary = summary.copy()
    if not tex_summary.empty and "region_key" in tex_summary.columns:
        jp_only = tex_summary[tex_summary["region_key"].astype(str).str.upper().eq("JP")]
        if not jp_only.empty:
            tex_summary = jp_only
    lines = [
        r"\begin{table}[htbp]",
        r"  \centering",
        r"  \small",
        r"  \setlength{\tabcolsep}{3pt}",
        r"  \caption{Japan $\lambda_B$ sensitivity for the BOJ bridge.}",
        r"  \label{tab:lambda_b_sensitivity}",
        r"  \resizebox{\textwidth}{!}{%",
        r"  \begin{tabular}{@{}llllllll@{}}",
        r"    \toprule",
        r"    Region & Target & Target proxy & Best $\lambda_B$ & Best RMSE & Total-credit RMSE & RMSE range & Main-claim use \\",
        r"    \midrule",
    ]
    if tex_summary.empty:
        lines.append(r"    No sensitivity results were generated. \\")
    else:
        for _, row in tex_summary.sort_values(["region_key", "target"]).iterrows():
            lines.append(
                "    "
                + _latex_escape(row["region_label"])
                + " & "
                + _latex_escape(row["target_label"])
                + " & "
                + _latex_escape(row["target_source"])
                + " & "
                + _latex_escape(_format_lambda_read(row))
                + " & "
                + _format_float(row["best_rmse"])
                + " & "
                + _format_float(row["baseline_rmse"])
                + " & "
                + _format_float(row["rmse_range"])
                + " & "
                + _latex_escape(row["main_claim_use"])
                + r" \\"
            )
    lines.extend(
        [
            r"    \bottomrule",
            r"  \end{tabular}",
            r"  }",
            r"  \par\smallskip\raggedright\footnotesize "
            + _latex_escape(
                "The grid is fixed at lambda_B in {0, 0.25, 0.5, 0.75, 1}; no region-specific lambda_B is estimated. The submitted table reports Japan only because the main empirical bridge is Japan-only. EU/US proxy-panel sweeps remain in the companion CSV as portability checks. Lambda-robust no gain means the destination score does not improve on the total-credit baseline."
            ),
            r"\end{table}",
        ]
    )
    return "\n".join(lines) + "\n"


def write_lambda_sensitivity_outputs(
    metrics: pd.DataFrame,
    panel: pd.DataFrame,
    summary: pd.DataFrame,
    *,
    root: Path,
) -> list[Path]:
    outputs: list[Path] = []
    site_dir = root / "site"
    data_dir = root / "data"
    tex_dir = root / "tex" / "generated"
    site_dir.mkdir(parents=True, exist_ok=True)
    data_dir.mkdir(parents=True, exist_ok=True)
    tex_dir.mkdir(parents=True, exist_ok=True)

    metrics_path = site_dir / "lambda_b_sensitivity.csv"
    panel_path = site_dir / "credit_destination_lambda_b_sweep.csv"
    summary_json_path = data_dir / "lambda_b_sensitivity_summary.json"
    tex_path = tex_dir / "theory_lambda_b_sensitivity.tex"

    metrics.to_csv(metrics_path, index=False)
    panel.to_csv(panel_path, index=False)
    payload: Mapping[str, Any] = {
        "lambda_grid": list(LAMBDA_GRID),
        "summary": json.loads(summary.to_json(orient="records")),
    }
    summary_json_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    tex_path.write_text(render_lambda_sensitivity_tex(summary), encoding="utf-8")

    outputs.extend([metrics_path, panel_path, summary_json_path, tex_path])
    return outputs


__all__ = [
    "LAMBDA_GRID",
    "lambda_destination_panel",
    "render_lambda_sensitivity_tex",
    "run_lambda_sensitivity",
    "summarize_lambda_sensitivity",
    "write_lambda_sensitivity_outputs",
]
