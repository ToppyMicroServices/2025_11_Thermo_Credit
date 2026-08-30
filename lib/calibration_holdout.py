from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd

from lib.theory_calibration import (
    MIN_TRAINING_ROWS,
    PARAMETER_NAMES,
    _auc_score,
    _filled_numeric,
    _fit_params,
    _format_float,
    _latex_escape,
    _normal_cdf,
    _pick_numeric_column,
    _x_c_from_params,
)
from lib.theory_figures import RegionFrame, load_region_frames


HORIZON_QUARTERS = 4
FIXED_TRAIN_START = "2000-01-01"
FIXED_TRAIN_END = "2015-12-31"
FIXED_TEST_START = "2016-01-01"
FIXED_TEST_END = "2025-12-31"
ROLLING_TRAIN_QUARTERS = 40
HOLDOUT_ITERATIONS = 4
ROLLING_ITERATIONS = 3


@dataclass(frozen=True)
class HoldoutTarget:
    key: str
    label: str
    source_column: str
    outcome: pd.Series
    simple_score: pd.Series


def _choose_holdout_target(frame: pd.DataFrame, *, horizon: int = HORIZON_QUARTERS) -> HoldoutTarget:
    spread_col = _pick_numeric_column(frame, ("spread", "hy_oas", "credit_spread"))
    if spread_col:
        spread = _filled_numeric(frame[spread_col])
        outcome = spread.shift(-horizon) - spread
        if outcome.dropna().size >= 16 and float(outcome.dropna().std(ddof=0)) > 1e-12:
            return HoldoutTarget(
                key="spread_widening",
                label="credit spread widening",
                source_column=spread_col,
                outcome=outcome,
                simple_score=spread - spread.shift(horizon),
            )

    loop_col = _pick_numeric_column(frame, ("loop_area",))
    if loop_col:
        loop = _filled_numeric(frame[loop_col]).abs()
        outcome = loop.shift(-horizon) - loop
        if outcome.dropna().size >= 16 and float(outcome.dropna().std(ddof=0)) > 1e-12:
            return HoldoutTarget(
                key="loop_area_change",
                label="loop-area change",
                source_column=loop_col,
                outcome=outcome,
                simple_score=loop - loop.shift(horizon),
            )

    tl_col = _pick_numeric_column(frame, ("T_L",))
    if tl_col:
        tl = _filled_numeric(frame[tl_col])
        outcome = -(tl.shift(-horizon) - tl)
        if outcome.dropna().size >= 16 and float(outcome.dropna().std(ddof=0)) > 1e-12:
            return HoldoutTarget(
                key="liquidity_deterioration",
                label="liquidity deterioration",
                source_column=tl_col,
                outcome=outcome,
                simple_score=-(tl - tl.shift(horizon)),
            )

    return HoldoutTarget(
        key="unavailable",
        label="unavailable",
        source_column="",
        outcome=pd.Series(np.nan, index=frame.index, dtype=float),
        simple_score=pd.Series(np.nan, index=frame.index, dtype=float),
    )


def _linear_predict_z(train_score: pd.Series, train_target: pd.Series, test_score: float) -> tuple[float, float, float]:
    pair = pd.concat(
        [
            pd.to_numeric(train_score, errors="coerce").reset_index(drop=True).rename("score"),
            pd.to_numeric(train_target, errors="coerce").reset_index(drop=True).rename("target"),
        ],
        axis=1,
    ).dropna()
    if pair.shape[0] < 8 or not np.isfinite(test_score):
        return float("nan"), float("nan"), float("nan")
    y = pair["target"]
    y_mean = float(y.mean())
    y_std = float(y.std(ddof=0))
    if not np.isfinite(y_std) or y_std <= 1e-12:
        return float("nan"), float("nan"), float(pair.shape[0])
    y_z = (y - y_mean) / y_std
    threshold_z = (0.0 - y_mean) / y_std
    x = pair["score"]
    x_std = float(x.std(ddof=0))
    if not np.isfinite(x_std) or x_std <= 1e-12:
        return float(y_z.mean()), float(threshold_z), float(pair.shape[0])
    slope, intercept = np.polyfit(x.to_numpy(dtype=float), y_z.to_numpy(dtype=float), deg=1)
    return float(intercept + slope * test_score), float(threshold_z), float(pair.shape[0])


def _dm_pvalue(loss_diff: np.ndarray) -> float:
    values = np.asarray(loss_diff, dtype=float)
    values = values[np.isfinite(values)]
    n = len(values)
    if n < 8 or float(np.std(values, ddof=1)) <= 1e-12:
        return float("nan")
    stat = float(np.mean(values) / (np.std(values, ddof=1) / np.sqrt(n)))
    return float(2.0 * (1.0 - _normal_cdf(abs(stat))))


def _model_metrics(pred: np.ndarray, actual: np.ndarray, threshold: np.ndarray) -> dict[str, float]:
    errors = pred - actual
    events = (actual > threshold).astype(int)
    score = pred - threshold
    probs = np.clip(1.0 / (1.0 + np.exp(-np.clip(score, -30, 30))), 1e-6, 1.0 - 1e-6)
    return {
        "rmse": float(np.sqrt(np.mean(errors**2))),
        "mae": float(np.mean(np.abs(errors))),
        "brier": float(np.mean((probs - events) ** 2)),
        "auc": _auc_score(score.tolist(), events.tolist()),
        "log_score": float(-np.mean(events * np.log(probs) + (1 - events) * np.log(1 - probs))),
    }


def _summarize_predictions(
    actuals: Sequence[float],
    thresholds: Sequence[float],
    tuned_preds: Sequence[float],
    raw_preds: Sequence[float],
    simple_preds: Sequence[float],
) -> dict[str, Any]:
    n = len(actuals)
    if n < 8:
        return {"n": float(n), "status": "insufficient coverage"}
    actual = np.asarray(actuals, dtype=float)
    threshold = np.asarray(thresholds, dtype=float)
    tuned = np.asarray(tuned_preds, dtype=float)
    raw = np.asarray(raw_preds, dtype=float)
    simple = np.asarray(simple_preds, dtype=float)
    tuned_metrics = _model_metrics(tuned, actual, threshold)
    raw_metrics = _model_metrics(raw, actual, threshold)
    simple_metrics = _model_metrics(simple, actual, threshold)
    tuned_loss = (tuned - actual) ** 2
    raw_loss = (raw - actual) ** 2
    simple_loss = (simple - actual) ** 2
    rmse_values = {
        "tuned_XC": tuned_metrics["rmse"],
        "raw_XC": raw_metrics["rmse"],
        "simple_baseline": simple_metrics["rmse"],
    }
    winner = min(rmse_values, key=rmse_values.get)
    if winner == "tuned_XC":
        status = "tuned wins"
    elif winner == "raw_XC":
        status = "raw X_C wins"
    else:
        status = "simple baseline wins"
    return {
        "n": float(n),
        "winner_rmse": winner,
        "status": status,
        "tuned_rmse": tuned_metrics["rmse"],
        "raw_rmse": raw_metrics["rmse"],
        "simple_rmse": simple_metrics["rmse"],
        "tuned_mae": tuned_metrics["mae"],
        "raw_mae": raw_metrics["mae"],
        "simple_mae": simple_metrics["mae"],
        "tuned_brier": tuned_metrics["brier"],
        "raw_brier": raw_metrics["brier"],
        "simple_brier": simple_metrics["brier"],
        "tuned_auc": tuned_metrics["auc"],
        "raw_auc": raw_metrics["auc"],
        "simple_auc": simple_metrics["auc"],
        "tuned_log_score": tuned_metrics["log_score"],
        "raw_log_score": raw_metrics["log_score"],
        "simple_log_score": simple_metrics["log_score"],
        "tuned_minus_raw_loss": float(np.mean(tuned_loss - raw_loss)),
        "tuned_minus_simple_loss": float(np.mean(tuned_loss - simple_loss)),
        "dm_p_vs_raw": _dm_pvalue(tuned_loss - raw_loss),
        "dm_p_vs_simple": _dm_pvalue(tuned_loss - simple_loss),
    }


def _actual_z(target: pd.Series, train_target: pd.Series, idx: int) -> float:
    actual = float(target.iloc[idx]) if idx < len(target) and np.isfinite(target.iloc[idx]) else float("nan")
    train_valid = pd.to_numeric(train_target, errors="coerce").dropna()
    if train_valid.size < 8 or not np.isfinite(actual):
        return float("nan")
    mean = float(train_valid.mean())
    std = float(train_valid.std(ddof=0))
    if not np.isfinite(std) or std <= 1e-12:
        return float("nan")
    return float((actual - mean) / std)


def _known_target_indices(indices: Sequence[int], forecast_idx: int, horizon: int) -> np.ndarray:
    return np.asarray([int(idx) for idx in indices if int(idx) + horizon < forecast_idx], dtype=int)


def _params_payload(params: Mapping[str, float], prefix: str = "") -> dict[str, float]:
    return {f"{prefix}{name}": float(params.get(name, float("nan"))) for name in PARAMETER_NAMES}


def _evaluate_fixed_holdout(
    frame: pd.DataFrame,
    region_key: str,
    target: HoldoutTarget,
    *,
    horizon: int,
    iterations: int,
) -> dict[str, Any]:
    dates = pd.to_datetime(frame["date"], errors="coerce")
    train_mask = (dates >= pd.Timestamp(FIXED_TRAIN_START)) & (dates <= pd.Timestamp(FIXED_TRAIN_END))
    test_mask = (dates >= pd.Timestamp(FIXED_TEST_START)) & (dates <= pd.Timestamp(FIXED_TEST_END))
    train = frame.loc[train_mask].copy().reset_index(drop=True)
    if train.shape[0] < MIN_TRAINING_ROWS:
        return {"strategy": "fixed_2000_2015", "n": 0.0, "status": "insufficient training"}
    try:
        params, _defaults, scales, u_col, v_col, s_col, _best = _fit_params(
            train,
            region_key,
            horizon=horizon,
            iterations=iterations,
        )
    except ValueError:
        return {"strategy": "fixed_2000_2015", "n": 0.0, "status": "missing calibration inputs"}

    tuned_score = _x_c_from_params(frame, params, u_col=u_col, v_col=v_col, s_col=s_col, scales=scales)
    raw_score = _filled_numeric(frame.get("X_C", pd.Series(index=frame.index, dtype=float)))
    simple_score = target.simple_score
    target_values = target.outcome
    train_idx = frame.index[train_mask].to_numpy()
    test_idx = frame.index[test_mask].to_numpy()
    if len(test_idx) == 0:
        return {"strategy": "fixed_2000_2015", "n": 0.0, "status": "insufficient test coverage", **_params_payload(params)}
    map_train_idx = _known_target_indices(train_idx, int(test_idx.min()), horizon)
    train_target = target_values.iloc[map_train_idx]

    actuals: list[float] = []
    thresholds: list[float] = []
    tuned_preds: list[float] = []
    raw_preds: list[float] = []
    simple_preds: list[float] = []
    for idx in test_idx:
        actual = _actual_z(target_values, train_target, int(idx))
        tuned_pred, threshold_z, _ = _linear_predict_z(tuned_score.iloc[map_train_idx], train_target, float(tuned_score.iloc[idx]))
        raw_pred, _, _ = _linear_predict_z(raw_score.iloc[map_train_idx], train_target, float(raw_score.iloc[idx]))
        simple_pred, _, _ = _linear_predict_z(simple_score.iloc[map_train_idx], train_target, float(simple_score.iloc[idx]))
        if all(np.isfinite(v) for v in (actual, threshold_z, tuned_pred, raw_pred, simple_pred)):
            actuals.append(actual)
            thresholds.append(threshold_z)
            tuned_preds.append(tuned_pred)
            raw_preds.append(raw_pred)
            simple_preds.append(simple_pred)

    result = _summarize_predictions(actuals, thresholds, tuned_preds, raw_preds, simple_preds)
    result.update(
        {
            "strategy": "fixed_2000_2015",
            "train_start": str(pd.to_datetime(train["date"]).min().date()),
            "train_end": str(pd.to_datetime(train["date"]).max().date()),
            "test_start": FIXED_TEST_START,
            "test_end": FIXED_TEST_END,
            "train_windows": 1.0,
            **_params_payload(params),
        }
    )
    return result


def _evaluate_rolling_holdout(
    frame: pd.DataFrame,
    region_key: str,
    target: HoldoutTarget,
    *,
    horizon: int,
    iterations: int,
    train_quarters: int,
) -> dict[str, Any]:
    dates = pd.to_datetime(frame["date"], errors="coerce")
    raw_score = _filled_numeric(frame.get("X_C", pd.Series(index=frame.index, dtype=float)))
    target_values = target.outcome
    simple_score = target.simple_score
    actuals: list[float] = []
    thresholds: list[float] = []
    tuned_preds: list[float] = []
    raw_preds: list[float] = []
    simple_preds: list[float] = []
    param_rows: list[dict[str, float]] = []

    for idx, current_date in enumerate(dates):
        if pd.isna(current_date) or current_date < pd.Timestamp(FIXED_TEST_START) or current_date > pd.Timestamp(FIXED_TEST_END):
            continue
        start_idx = max(0, idx - train_quarters)
        train = frame.iloc[start_idx:idx].copy().reset_index(drop=True)
        if train.shape[0] < MIN_TRAINING_ROWS:
            continue
        train_indices = np.arange(start_idx, idx)
        map_train_idx = _known_target_indices(train_indices, idx, horizon)
        train_target = target_values.iloc[map_train_idx]
        actual = _actual_z(target_values, train_target, idx)
        if not np.isfinite(actual):
            continue
        try:
            params, _defaults, scales, u_col, v_col, s_col, _best = _fit_params(
                train,
                region_key,
                horizon=horizon,
                iterations=iterations,
            )
        except ValueError:
            continue
        tuned_train = _x_c_from_params(train, params, u_col=u_col, v_col=v_col, s_col=s_col, scales=scales)
        tuned_test = _x_c_from_params(frame.iloc[[idx]].copy(), params, u_col=u_col, v_col=v_col, s_col=s_col, scales=scales)
        relative_map_idx = map_train_idx - start_idx
        tuned_pred, threshold_z, _ = _linear_predict_z(tuned_train.iloc[relative_map_idx], train_target, float(tuned_test.iloc[0]))
        raw_pred, _, _ = _linear_predict_z(raw_score.iloc[map_train_idx], train_target, float(raw_score.iloc[idx]))
        simple_pred, _, _ = _linear_predict_z(simple_score.iloc[map_train_idx], train_target, float(simple_score.iloc[idx]))
        if all(np.isfinite(v) for v in (threshold_z, tuned_pred, raw_pred, simple_pred)):
            actuals.append(actual)
            thresholds.append(threshold_z)
            tuned_preds.append(tuned_pred)
            raw_preds.append(raw_pred)
            simple_preds.append(simple_pred)
            param_rows.append(_params_payload(params))

    result = _summarize_predictions(actuals, thresholds, tuned_preds, raw_preds, simple_preds)
    result.update(
        {
            "strategy": "rolling_10y",
            "train_start": "rolling",
            "train_end": "rolling",
            "test_start": FIXED_TEST_START,
            "test_end": FIXED_TEST_END,
            "train_windows": float(len(param_rows)),
        }
    )
    if param_rows:
        params_df = pd.DataFrame(param_rows)
        for name in PARAMETER_NAMES:
            result[name] = float(params_df[name].mean())
    return result


def evaluate_calibration_holdout_region(
    region: RegionFrame,
    *,
    horizon: int = HORIZON_QUARTERS,
    fixed_iterations: int = HOLDOUT_ITERATIONS,
    rolling_iterations: int = ROLLING_ITERATIONS,
    rolling_train_quarters: int = ROLLING_TRAIN_QUARTERS,
) -> pd.DataFrame:
    frame = region.frame.copy().sort_values("date").reset_index(drop=True)
    target = _choose_holdout_target(frame, horizon=horizon)
    rows: list[dict[str, Any]] = []
    for result in (
        _evaluate_fixed_holdout(frame, region.key, target, horizon=horizon, iterations=fixed_iterations),
        _evaluate_rolling_holdout(
            frame,
            region.key,
            target,
            horizon=horizon,
            iterations=rolling_iterations,
            train_quarters=rolling_train_quarters,
        ),
    ):
        result.update(
            {
                "region_key": region.key,
                "region_label": region.label,
                "panel_source": region.source_path,
                "horizon_quarters": int(horizon),
                "target": target.key,
                "target_label": target.label,
                "target_source": target.source_column,
            }
        )
        rows.append(result)
    return pd.DataFrame(rows)


def run_calibration_holdout_tests(
    site_dir: Path,
    *,
    source_ref: str | None = None,
    panel_mode: str = "realtime",
    horizon: int = HORIZON_QUARTERS,
) -> pd.DataFrame:
    rows: list[pd.DataFrame] = []
    for region in load_region_frames(site_dir, source_ref=source_ref, mode=panel_mode):
        rows.append(evaluate_calibration_holdout_region(region, horizon=horizon))
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()


def _render_metric(row: pd.Series, column: str) -> str:
    return _format_float(float(row.get(column, float("nan"))), 3)


def render_calibration_holdout_tex(results: pd.DataFrame) -> str:
    lines = [
        r"\begin{table}[htbp]",
        r"  \centering",
        r"  \small",
        r"  \setlength{\tabcolsep}{3pt}",
        r"  \caption{Diagnostic holdout test for the implicit headroom score.}",
        r"  \label{tab:calibration_holdout}",
        r"  \resizebox{\textwidth}{!}{%",
        r"  \begin{tabular}{@{}llllllllll@{}}",
        r"    \toprule",
        r"    Region & Strategy & Target & $N$ & Tuned RMSE & Raw $X_C$ RMSE & Simple RMSE & Tuned Brier & Winner & DM $p$ vs raw \\",
        r"    \midrule",
    ]
    if results.empty:
        lines.append(r"    No calibration holdout results were generated. \\")
    else:
        order = {"fixed_2000_2015": 0, "rolling_10y": 1}
        work = results.copy()
        work.loc[:, "_order"] = work["strategy"].map(order).fillna(9)
        for _, row in work.sort_values(["region_key", "_order"]).iterrows():
            lines.append(
                "    "
                + _latex_escape(str(row["region_label"]))
                + " & "
                + _latex_escape(str(row["strategy"]))
                + " & "
                + _latex_escape(str(row["target_label"]))
                + " & "
                + _format_float(float(row.get("n", float("nan"))), 0)
                + " & "
                + _render_metric(row, "tuned_rmse")
                + " & "
                + _render_metric(row, "raw_rmse")
                + " & "
                + _render_metric(row, "simple_rmse")
                + " & "
                + _render_metric(row, "tuned_brier")
                + " & "
                + _latex_escape(str(row.get("winner_rmse", row.get("status", "n/a"))))
                + " & "
                + _render_metric(row, "dm_p_vs_raw")
                + r" \\"
            )
    lines.extend(
        [
            r"    \bottomrule",
            r"  \end{tabular}",
            r"  }",
            r"  \par\smallskip\raggedright\footnotesize "
            + _latex_escape(
                "The fixed split estimates theta=(T0,p0,U0,V0,S0) on observations available through 2015 and evaluates 2016--2025 where the h=4 target is observed. The rolling strategy refits theta on the previous 40 quarters before each forecast origin. The train-side forecast mapping uses only target realizations known before the forecast origin. Negative tuned-minus-raw loss in the CSV favors the calibrated score."
            ),
            r"  \par\smallskip\raggedright\footnotesize "
            + _latex_escape(
                "The simple baseline is the target source's trailing four-quarter change. Test targets prefer future spread widening; loop-area and liquidity-deterioration targets are used only when spread coverage is insufficient."
            ),
            r"\end{table}",
        ]
    )
    return "\n".join(lines) + "\n"


def write_calibration_holdout_outputs(results: pd.DataFrame, *, root: Path) -> list[Path]:
    site_dir = root / "site"
    data_dir = root / "data"
    tex_dir = root / "tex" / "generated"
    site_dir.mkdir(parents=True, exist_ok=True)
    data_dir.mkdir(parents=True, exist_ok=True)
    tex_dir.mkdir(parents=True, exist_ok=True)

    csv_path = site_dir / "calibration_holdout_test.csv"
    json_path = data_dir / "calibration_holdout_summary.json"
    tex_path = tex_dir / "theory_calibration_holdout.tex"
    results.to_csv(csv_path, index=False)
    payload = {
        "horizon_quarters": HORIZON_QUARTERS,
        "fixed_train_window": [FIXED_TRAIN_START, FIXED_TRAIN_END],
        "fixed_test_window": [FIXED_TEST_START, FIXED_TEST_END],
        "rolling_train_quarters": ROLLING_TRAIN_QUARTERS,
        "results": json.loads(results.to_json(orient="records")),
    }
    json_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    tex_path.write_text(render_calibration_holdout_tex(results), encoding="utf-8")
    return [csv_path, json_path, tex_path]


__all__ = [
    "evaluate_calibration_holdout_region",
    "render_calibration_holdout_tex",
    "run_calibration_holdout_tests",
    "write_calibration_holdout_outputs",
]
