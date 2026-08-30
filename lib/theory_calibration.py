from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence

import numpy as np
import pandas as pd

from lib.config_loader import load_config
from lib.theory_figures import RegionFrame, load_region_frames


PARAMETER_NAMES: Sequence[str] = ("T0", "p0", "U0", "V0", "S0")
OBJECTIVE_WEIGHTS: Dict[str, float] = {
    "stress": 1.05,
    "loop": 0.85,
    "liquidity": 0.45,
    "negative_share": 0.70,
    "regularization": 0.20,
    "range": 0.35,
}
GRID_MULTIPLIERS: Sequence[float] = (-1.0, -0.5, 0.0, 0.5, 1.0)
MIN_TRAINING_ROWS = 24


@dataclass
class CalibrationResult:
    region_key: str
    region_label: str
    coverage_start: str
    coverage_end: str
    params: Dict[str, float]
    defaults: Dict[str, float]
    scales: Dict[str, float]
    objective: float
    diagnostics: Dict[str, float]
    baseline_diagnostics: Dict[str, float]
    oos_diagnostics: Dict[str, Any]
    latest: Dict[str, float]
    deltas_4q: Dict[str, float]
    notes: List[str]
    preprocessing_mode: str
    panel_source: str


def _pick_numeric_column(frame: pd.DataFrame, candidates: Sequence[str]) -> Optional[str]:
    for candidate in candidates:
        if candidate not in frame.columns:
            continue
        series = pd.to_numeric(frame[candidate], errors="coerce")
        if series.dropna().size >= 8:
            return candidate
    return None


def _robust_scale(series: pd.Series, fallback: float = 1.0) -> float:
    numeric = pd.to_numeric(series, errors="coerce").dropna()
    if numeric.empty:
        return fallback
    q75 = float(numeric.quantile(0.75))
    q25 = float(numeric.quantile(0.25))
    scale = q75 - q25
    if np.isfinite(scale) and scale > 0:
        return scale
    std = float(numeric.std(ddof=0))
    if np.isfinite(std) and std > 0:
        return std
    return fallback


def _corr_stats(left: pd.Series, right: pd.Series) -> tuple[float, int]:
    pair = pd.concat(
        [
            pd.to_numeric(left, errors="coerce"),
            pd.to_numeric(right, errors="coerce"),
        ],
        axis=1,
    ).dropna()
    if pair.shape[0] < 8:
        return 0.0, int(pair.shape[0])
    if float(pair.iloc[:, 0].std(ddof=0)) <= 1e-12 or float(pair.iloc[:, 1].std(ddof=0)) <= 1e-12:
        return 0.0, int(pair.shape[0])
    value = float(pair.iloc[:, 0].corr(pair.iloc[:, 1]))
    return (value if np.isfinite(value) else 0.0), int(pair.shape[0])


def _safe_corr(left: pd.Series, right: pd.Series) -> float:
    value, _ = _corr_stats(left, right)
    return value


def _filled_numeric(series: pd.Series) -> pd.Series:
    numeric = pd.to_numeric(series, errors="coerce")
    if numeric.dropna().empty:
        return numeric
    return numeric.interpolate(limit_direction="both")


def _latest_value(frame: pd.DataFrame, column: str) -> float:
    if column not in frame.columns:
        return float("nan")
    numeric = pd.to_numeric(frame[column], errors="coerce").dropna()
    if numeric.empty:
        return float("nan")
    return float(numeric.iloc[-1])


def _delta_4q(frame: pd.DataFrame, column: str) -> float:
    if column not in frame.columns:
        return float("nan")
    numeric = pd.to_numeric(frame[column], errors="coerce").dropna()
    if numeric.size < 5:
        return float("nan")
    return float(numeric.iloc[-1] - numeric.iloc[-5])


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
    out = text
    for src, dst in replacements.items():
        out = out.replace(src, dst)
    return out


def _latex_note_text(text: str) -> str:
    """Escape a prose note while keeping common metric names readable."""
    out = _latex_escape(text)
    replacements = {
        r"S\_M": r"$S_M$",
        r"T\_L": r"$T_L$",
        r"X\_C": r"$X_C$",
    }
    for src, dst in replacements.items():
        out = out.replace(src, dst)
    return out


def _format_float(value: float, digits: int = 3) -> str:
    if not np.isfinite(value):
        return "n/a"
    return f"{value:,.{digits}f}"


def _format_diagnostic(result: "CalibrationResult", key: str, pairs_key: str, digits: int = 3) -> str:
    if result.diagnostics.get(pairs_key, 0.0) < 8:
        return "n/a"
    return _format_float(result.diagnostics[key], digits)


def _format_oos(result: "CalibrationResult", key: str, digits: int = 3) -> str:
    if result.oos_diagnostics.get("n", 0.0) < 8:
        return "n/a"
    return _format_float(result.oos_diagnostics.get(key, float("nan")), digits)


def _format_pct(value: float) -> str:
    if not np.isfinite(value):
        return "n/a"
    return f"P{value:.0f}"


def _loop_area_audit_notes(results: Sequence["CalibrationResult"]) -> List[str]:
    groups: Dict[tuple[str, int], List[str]] = {}
    values: Dict[tuple[str, int], float] = {}
    for result in results:
        value = result.latest.get("loop_area", float("nan"))
        if not np.isfinite(value) or abs(value) <= 1e-9:
            continue
        key = (result.coverage_end, int(round(float(value))))
        groups.setdefault(key, []).append(result.region_label)
        values[key] = float(value)
    notes: List[str] = []
    for key, labels in groups.items():
        if len(labels) < 2:
            continue
        label_text = f"{labels[0]} and {labels[1]}" if len(labels) == 2 else ", ".join(labels)
        notes.append(
            "Identical latest loop-area values for "
            + label_text
            + f" ({_format_float(values[key], 0)} on {key[0]}) require a source/fallback audit and are not interpreted as a cross-region empirical finding."
        )
    return notes


def _missingness_note(result: "CalibrationResult") -> str:
    notes = " ".join(result.notes).lower()
    bits: List[str] = []
    if not np.isfinite(result.latest.get("T_L", float("nan"))):
        bits.append("T_L unavailable")
    if "spread coverage is too thin" in notes:
        bits.append("spread coverage thin")
    if "clipped at zero" in notes:
        bits.append("raw X_C clipped")
    if result.oos_diagnostics.get("n", 0.0) < 8:
        bits.append("OOS coverage thin")
    return "; ".join(bits) if bits else "reported columns available"


def _render_calibration_metadata_panel_tex(results: Sequence["CalibrationResult"]) -> str:
    header = [
        r"  \par\medskip",
        r"  \textit{Panel B. Comparability metadata.}\par\smallskip",
        r"  \setlength{\tabcolsep}{3pt}",
        r"  \resizebox{\textwidth}{!}{%",
        r"  \begin{tabular}{@{}lllll@{}}",
        r"    \toprule",
        r"    Region & Estimation window & Units/readout & Scaling & Missingness/read limits \\",
        r"    \midrule",
    ]
    rows: List[str] = []
    for result in results:
        window = f"{result.coverage_start}--{result.coverage_end}"
        readout = "within-region index; raw source units not comparable"
        scaling = "U,V,S median-centered and IQR-scaled by region"
        rows.append(
            "    "
            + _latex_escape(result.region_label)
            + f" & {_latex_escape(window)}"
            + f" & {_latex_escape(readout)}"
            + f" & {_latex_escape(scaling)}"
            + f" & {_latex_escape(_missingness_note(result))} \\\\"
        )
    footer = [
        r"    \bottomrule",
        r"  \end{tabular}",
        r"  }",
        r"  \par\smallskip\raggedright\footnotesize "
        + _latex_escape(
            "Because reference anchors and scales are region-specific, JP/EU/US calibrated score levels are not compared directly. Cross-region statements use only within-region percentiles, z-scores, historical ranks, and OOS diagnostics."
        ),
        r"  \par\smallskip\raggedright\footnotesize "
        + _latex_escape(
            "Calibration loads the real-time release-lagged indicator panels when available; dashboard-retrospective panels are used only as a fallback."
        ),
    ]
    return "\n".join(header + rows + footer) + "\n"


def _describe_level(series: pd.Series, latest: float) -> str:
    valid = pd.to_numeric(series, errors="coerce").dropna()
    if valid.empty or not np.isfinite(latest):
        return "unclassified"
    low = float(valid.quantile(0.33))
    high = float(valid.quantile(0.67))
    if latest <= low:
        return "low"
    if latest >= high:
        return "high"
    return "mid"


def _describe_delta(delta: float, scale: float, *, inverse: bool = False) -> str:
    if not np.isfinite(delta):
        return "unclear"
    threshold = max(scale * 0.05, 1e-9)
    if abs(delta) <= threshold:
        return "roughly flat"
    positive = "easing" if inverse else "rising"
    negative = "worsening" if inverse else "falling"
    if inverse:
        return positive if delta < 0 else negative
    return positive if delta > 0 else negative


def _delta_from_series(series: pd.Series) -> float:
    numeric = pd.to_numeric(series, errors="coerce").dropna()
    if numeric.size < 5:
        return float("nan")
    return float(numeric.iloc[-1] - numeric.iloc[-5])


def _latest_percentile(series: pd.Series) -> float:
    numeric = pd.to_numeric(series, errors="coerce").dropna()
    if numeric.empty:
        return float("nan")
    latest = float(numeric.iloc[-1])
    return float(100.0 * (numeric <= latest).sum() / numeric.size)


def _x_c_from_params(
    frame: pd.DataFrame,
    params: Mapping[str, float],
    *,
    u_col: str,
    v_col: str,
    s_col: str,
    scales: Mapping[str, float],
) -> pd.Series:
    u = _filled_numeric(frame[u_col])
    v = _filled_numeric(frame[v_col])
    s = _filled_numeric(frame[s_col])
    u_score = (u - params["U0"]) / max(scales["U"], 1.0)
    v_score = (v - params["V0"]) / max(scales["V"], 1.0)
    s_score = (s - params["S0"]) / max(scales["S"], 1.0)
    raw = u_score + params["p0"] * v_score - params["T0"] * s_score
    return pd.Series(np.arcsinh(raw), index=frame.index, dtype=float)


def _objective_components(
    frame: pd.DataFrame,
    params: Mapping[str, float],
    defaults: Mapping[str, float],
    u_col: str,
    v_col: str,
    s_col: str,
    scales: Mapping[str, float],
    horizon: int,
) -> Dict[str, float]:
    x_c = _x_c_from_params(frame, params, u_col=u_col, v_col=v_col, s_col=s_col, scales=scales)
    u_series = _filled_numeric(frame[u_col])
    spread_series = _filled_numeric(frame.get("spread", pd.Series(index=frame.index, dtype=float)))
    loop_series = _filled_numeric(frame.get("loop_area", pd.Series(index=frame.index, dtype=float))).abs()
    tl = _filled_numeric(frame.get("T_L", pd.Series(index=frame.index, dtype=float)))

    future_u = u_series.shift(-horizon) - u_series
    future_spread = spread_series.shift(-horizon) - spread_series
    future_loop = loop_series.shift(-horizon) - loop_series

    growth_corr, growth_n = _corr_stats(x_c, future_u)
    stress_corr, stress_n = _corr_stats(x_c, future_spread)
    loop_corr, loop_n = _corr_stats(x_c, future_loop)
    liq_corr, liq_n = _corr_stats(x_c, tl)

    x_valid = pd.to_numeric(x_c, errors="coerce").dropna()
    neg_share = float((x_valid < 0).mean()) if not x_valid.empty else 0.5
    dispersion = float(x_valid.std(ddof=0)) if x_valid.size >= 2 else 0.0
    range_penalty = 0.0 if dispersion > 1e-9 else 1.0

    reg_penalty = 0.0
    reg_scales = {
        "T0": max(abs(defaults["T0"]), 1.0),
        "p0": max(abs(defaults["p0"]), 1.0),
        "U0": max(_robust_scale(pd.to_numeric(frame[u_col], errors="coerce")), 1.0),
        "V0": max(_robust_scale(pd.to_numeric(frame[v_col], errors="coerce")), 1.0),
        "S0": max(_robust_scale(pd.to_numeric(frame[s_col], errors="coerce")), 1.0),
    }
    for name in PARAMETER_NAMES:
        reg_penalty += ((params[name] - defaults[name]) / reg_scales[name]) ** 2

    # Growth correlation is reported as a diagnostic only. It is excluded from
    # the search objective because U can be GDP-like in some panels, making
    # future U growth an overlap-prone target for the same constructed signal.
    objective = (
        OBJECTIVE_WEIGHTS["stress"] * (1.0 + stress_corr)
        + OBJECTIVE_WEIGHTS["loop"] * (1.0 + loop_corr)
        + OBJECTIVE_WEIGHTS["liquidity"] * (1.0 - liq_corr)
        + OBJECTIVE_WEIGHTS["negative_share"] * neg_share
        + OBJECTIVE_WEIGHTS["regularization"] * reg_penalty
        + OBJECTIVE_WEIGHTS["range"] * range_penalty
    )
    return {
        "objective": float(objective),
        "growth_corr": float(growth_corr),
        "stress_corr": float(stress_corr),
        "loop_corr": float(loop_corr),
        "liquidity_corr": float(liq_corr),
        "negative_share": float(neg_share),
        "dispersion": float(dispersion),
        "regularization": float(reg_penalty),
        "growth_pairs": float(growth_n),
        "stress_pairs": float(stress_n),
        "loop_pairs": float(loop_n),
        "liquidity_pairs": float(liq_n),
    }


def _baseline_diagnostics(frame: pd.DataFrame, horizon: int) -> Dict[str, float]:
    pipeline = _filled_numeric(frame.get("X_C", pd.Series(index=frame.index, dtype=float)))
    u_series = _filled_numeric(frame.get("U", pd.Series(index=frame.index, dtype=float)))
    spread_series = _filled_numeric(frame.get("spread", pd.Series(index=frame.index, dtype=float)))
    loop_series = _filled_numeric(frame.get("loop_area", pd.Series(index=frame.index, dtype=float))).abs()
    future_u = u_series.shift(-horizon) - u_series
    future_spread = spread_series.shift(-horizon) - spread_series
    future_loop = loop_series.shift(-horizon) - loop_series
    growth_corr, growth_n = _corr_stats(pipeline, future_u)
    stress_corr, stress_n = _corr_stats(pipeline, future_spread)
    loop_corr, loop_n = _corr_stats(pipeline, future_loop)
    return {
        "growth_corr": float(growth_corr),
        "stress_corr": float(stress_corr),
        "loop_corr": float(loop_corr),
        "growth_pairs": float(growth_n),
        "stress_pairs": float(stress_n),
        "loop_pairs": float(loop_n),
    }


def _calibration_context(work: pd.DataFrame, region_key: str) -> tuple[str, str, str, Dict[str, float], Dict[str, float]]:
    u_col = _pick_numeric_column(work, ("U", "Y", "L_real"))
    v_col = _pick_numeric_column(work, ("V_C", "V_C_headroom", "V_C_legacy"))
    s_col = _pick_numeric_column(work, ("S_M",))
    if u_col is None or v_col is None or s_col is None:
        raise ValueError(f"Missing calibration inputs for region {region_key}")

    cfg = load_config(region_key)
    u_series = _filled_numeric(work[u_col])
    v_series = _filled_numeric(work[v_col])
    s_series = _filled_numeric(work[s_col])

    scales = {
        "U": max(_robust_scale(u_series), 1.0),
        "V": max(_robust_scale(v_series), 1.0),
        "S": max(_robust_scale(s_series), 1.0),
    }

    defaults = {
        "T0": float(cfg.get("T0", 1.0)),
        "p0": float(cfg.get("p0", 1.0)),
        "U0": float(u_series.median(skipna=True)),
        "V0": float(v_series.median(skipna=True)),
        "S0": float(s_series.median(skipna=True)),
    }
    return u_col, v_col, s_col, scales, defaults


def _fit_params(
    work: pd.DataFrame,
    region_key: str,
    *,
    horizon: int,
    iterations: int,
) -> tuple[Dict[str, float], Dict[str, float], Dict[str, float], str, str, str, Dict[str, float]]:
    u_col, v_col, s_col, scales, defaults = _calibration_context(work, region_key)
    params = dict(defaults)
    step_sizes = {
        "T0": max(abs(defaults["T0"]) * 0.35, 0.15),
        "p0": max(abs(defaults["p0"]) * 0.35, 0.15),
        "U0": scales["U"],
        "V0": scales["V"],
        "S0": scales["S"],
    }

    best = _objective_components(
        work,
        params,
        defaults,
        u_col=u_col,
        v_col=v_col,
        s_col=s_col,
        scales=scales,
        horizon=horizon,
    )
    for _ in range(iterations):
        for name in PARAMETER_NAMES:
            current = params[name]
            candidates = []
            for multiplier in GRID_MULTIPLIERS:
                candidate = current + multiplier * step_sizes[name]
                if name in {"T0", "p0"}:
                    candidate = max(candidate, 0.05)
                candidates.append(float(candidate))
            local_best = dict(best)
            local_value = current
            for candidate in candidates:
                trial = dict(params)
                trial[name] = float(candidate)
                score = _objective_components(
                    work,
                    trial,
                    defaults,
                    u_col=u_col,
                    v_col=v_col,
                    s_col=s_col,
                    scales=scales,
                    horizon=horizon,
                )
                if score["objective"] < local_best["objective"]:
                    local_best = score
                    local_value = float(candidate)
            params[name] = local_value
            best = local_best
        for key in step_sizes:
            step_sizes[key] *= 0.55

    return params, defaults, scales, u_col, v_col, s_col, best


def _normal_cdf(value: float) -> float:
    import math

    return 0.5 * math.erfc(-value / math.sqrt(2.0))


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


def _linear_prediction(train_score: pd.Series, train_target: pd.Series, test_score: float) -> tuple[float, float, float]:
    pair = pd.concat(
        [
            pd.to_numeric(train_score, errors="coerce"),
            pd.to_numeric(train_target, errors="coerce"),
        ],
        axis=1,
    ).dropna()
    if pair.shape[0] < 8 or not np.isfinite(test_score):
        return float("nan"), float("nan"), float("nan")
    y = pair.iloc[:, 1]
    y_mean = float(y.mean())
    y_std = float(y.std(ddof=0))
    if not np.isfinite(y_std) or y_std <= 1e-12:
        return float("nan"), float("nan"), float("nan")
    x = pair.iloc[:, 0]
    x_std = float(x.std(ddof=0))
    y_z = (y - y_mean) / y_std
    threshold_z = (0.0 - y_mean) / y_std
    if not np.isfinite(x_std) or x_std <= 1e-12:
        return float(y_z.mean()), threshold_z, float(pair.shape[0])
    slope, intercept = np.polyfit(x.to_numpy(dtype=float), y_z.to_numpy(dtype=float), deg=1)
    return float(intercept + slope * test_score), float(threshold_z), float(pair.shape[0])


def _choose_oos_target(work: pd.DataFrame, horizon: int) -> tuple[str, pd.Series]:
    spread = _filled_numeric(work.get("spread", pd.Series(index=work.index, dtype=float)))
    spread_target = spread.shift(-horizon) - spread
    if spread_target.dropna().size >= 12 and float(spread_target.dropna().std(ddof=0)) > 1e-12:
        return "spread widening", spread_target
    loop = _filled_numeric(work.get("loop_area", pd.Series(index=work.index, dtype=float))).abs()
    loop_target = loop.shift(-horizon) - loop
    if loop_target.dropna().size >= 12 and float(loop_target.dropna().std(ddof=0)) > 1e-12:
        return "loop-area change", loop_target
    return "unavailable", pd.Series(index=work.index, dtype=float)


def _oos_diagnostics(work: pd.DataFrame, region_key: str, *, horizon: int, iterations: int) -> Dict[str, Any]:
    target_label, target = _choose_oos_target(work, horizon)
    if target_label == "unavailable":
        return {"target": target_label, "n": 0.0}

    min_train = max(MIN_TRAINING_ROWS, horizon * 4)
    tuned_preds: List[float] = []
    baseline_preds: List[float] = []
    actuals: List[float] = []
    thresholds: List[float] = []

    for idx in range(min_train, max(min_train, len(work) - horizon)):
        actual = float(target.iloc[idx]) if idx < len(target) and np.isfinite(target.iloc[idx]) else float("nan")
        if not np.isfinite(actual):
            continue
        train = work.iloc[:idx].copy()
        try:
            params, _, scales, u_col, v_col, s_col, _ = _fit_params(
                train,
                region_key,
                horizon=horizon,
                iterations=max(1, min(iterations, 3)),
            )
        except ValueError:
            continue
        train_target = target.iloc[:idx]
        train_x = _x_c_from_params(train, params, u_col=u_col, v_col=v_col, s_col=s_col, scales=scales)
        test_x = _x_c_from_params(work.iloc[[idx]].copy(), params, u_col=u_col, v_col=v_col, s_col=s_col, scales=scales)
        tuned_pred, threshold_z, _ = _linear_prediction(train_x, train_target, float(test_x.iloc[0]))

        pipeline_train = _filled_numeric(train.get("X_C", pd.Series(index=train.index, dtype=float)))
        pipeline_test = _filled_numeric(work.iloc[[idx]].get("X_C", pd.Series(index=work.iloc[[idx]].index, dtype=float)))
        baseline_pred, _, _ = _linear_prediction(pipeline_train, train_target, float(pipeline_test.iloc[0]) if len(pipeline_test) else float("nan"))

        if np.isfinite(tuned_pred) and np.isfinite(baseline_pred) and np.isfinite(threshold_z):
            actual_mean = float(pd.to_numeric(train_target, errors="coerce").dropna().mean())
            actual_std = float(pd.to_numeric(train_target, errors="coerce").dropna().std(ddof=0))
            if np.isfinite(actual_std) and actual_std > 1e-12:
                actual_z = (actual - actual_mean) / actual_std
                tuned_preds.append(float(tuned_pred))
                baseline_preds.append(float(baseline_pred))
                actuals.append(float(actual_z))
                thresholds.append(float(threshold_z))

    n = len(actuals)
    if n < 8:
        return {"target": target_label, "n": float(n)}
    actual_arr = np.asarray(actuals, dtype=float)
    tuned_arr = np.asarray(tuned_preds, dtype=float)
    baseline_arr = np.asarray(baseline_preds, dtype=float)
    threshold_arr = np.asarray(thresholds, dtype=float)
    tuned_errors = tuned_arr - actual_arr
    baseline_errors = baseline_arr - actual_arr
    rmse = float(np.sqrt(np.mean(tuned_errors**2)))
    mae = float(np.mean(np.abs(tuned_errors)))
    baseline_rmse = float(np.sqrt(np.mean(baseline_errors**2)))
    baseline_mae = float(np.mean(np.abs(baseline_errors)))
    events = (actual_arr > threshold_arr).astype(int)
    stress_scores = tuned_arr - threshold_arr
    auc = _auc_score(stress_scores.tolist(), events.tolist())
    probs = 1.0 / (1.0 + np.exp(-np.clip(stress_scores, -30, 30)))
    probs = np.clip(probs, 1e-6, 1.0 - 1e-6)
    log_score = float(-np.mean(events * np.log(probs) + (1 - events) * np.log(1 - probs)))
    loss_diff = tuned_errors**2 - baseline_errors**2
    if n >= 8 and float(np.std(loss_diff, ddof=1)) > 1e-12:
        dm_stat = float(np.mean(loss_diff) / (np.std(loss_diff, ddof=1) / np.sqrt(n)))
        dm_p = float(2.0 * (1.0 - _normal_cdf(abs(dm_stat))))
    else:
        dm_stat = float("nan")
        dm_p = float("nan")
    return {
        "target": target_label,
        "n": float(n),
        "rmse": rmse,
        "mae": mae,
        "baseline_rmse": baseline_rmse,
        "baseline_mae": baseline_mae,
        "auc": auc,
        "log_score": log_score,
        "dm_stat": dm_stat,
        "dm_p": dm_p,
    }


def calibrate_region_frame(frame: pd.DataFrame, region_key: str, *, horizon: int = 4, iterations: int = 4) -> CalibrationResult:
    work = frame.copy().sort_values("date").reset_index(drop=True)
    preprocessing_mode = "dashboard_retrospective"
    if "preprocessing_mode" in work.columns:
        mode_values = work["preprocessing_mode"].dropna().astype(str)
        if not mode_values.empty:
            preprocessing_mode = str(mode_values.iloc[-1])
    params, defaults, scales, u_col, v_col, s_col, best = _fit_params(
        work,
        region_key,
        horizon=horizon,
        iterations=iterations,
    )
    x_c = _x_c_from_params(work, params, u_col=u_col, v_col=v_col, s_col=s_col, scales=scales)
    baseline = _baseline_diagnostics(work, horizon=horizon)
    oos = _oos_diagnostics(work, region_key, horizon=horizon, iterations=iterations)
    latest = {
        "S_M": _latest_value(work, "S_M"),
        "T_L": _latest_value(work, "T_L"),
        "X_C_calibrated": float(pd.to_numeric(x_c, errors="coerce").dropna().iloc[-1]) if pd.to_numeric(x_c, errors="coerce").dropna().size else float("nan"),
        "X_C_pipeline": _latest_value(work, "X_C"),
        "loop_area": _latest_value(work, "loop_area"),
        "q_t": _latest_value(work, "q_t"),
        "destination_coverage": _latest_value(work, "destination_coverage"),
        "S_M_pctile": _latest_percentile(work.get("S_M", pd.Series(index=work.index, dtype=float))),
        "T_L_pctile": _latest_percentile(work.get("T_L", pd.Series(index=work.index, dtype=float))),
        "X_C_calibrated_pctile": _latest_percentile(x_c),
        "X_C_pipeline_pctile": _latest_percentile(work.get("X_C", pd.Series(index=work.index, dtype=float))),
        "loop_area_pctile": _latest_percentile(work.get("loop_area", pd.Series(index=work.index, dtype=float))),
        "q_t_pctile": _latest_percentile(work.get("q_t", pd.Series(index=work.index, dtype=float))),
    }
    deltas_4q = {
        "S_M": _delta_4q(work, "S_M"),
        "T_L": _delta_4q(work, "T_L"),
        "X_C_calibrated": _delta_from_series(x_c),
        "loop_area": _delta_4q(work, "loop_area"),
    }

    tl_series = pd.to_numeric(work.get("T_L", pd.Series(index=work.index, dtype=float)), errors="coerce")
    tl_level = _describe_level(tl_series, latest["T_L"])
    xc_delta_desc = _describe_delta(deltas_4q["X_C_calibrated"], _robust_scale(pd.to_numeric(x_c, errors="coerce")), inverse=False)
    loop_desc = _describe_delta(deltas_4q["loop_area"], _robust_scale(pd.to_numeric(work.get("loop_area"), errors="coerce")), inverse=True)
    notes = [
        f"coverage through {pd.to_datetime(work['date']).max().date()}",
        f"implicit headroom is {xc_delta_desc} over the last four quarters",
        f"streaming loop area is {loop_desc} over the last four quarters",
    ]
    if np.isfinite(latest["q_t"]):
        notes.insert(
            1,
            f"credit-destination q_t proxy is {_format_float(latest['q_t'])} with {_format_float(latest['destination_coverage'])} destination coverage",
        )
    if tl_series.dropna().size >= 8:
        notes.insert(1, f"T_L is currently {tl_level} in its own regional history")
    else:
        notes.insert(1, "T_L is unavailable in the current indicator panel")
    stress_delta = best["stress_corr"] - baseline["stress_corr"]
    if oos.get("n", 0.0) >= 8:
        rmse = float(oos.get("rmse", float("nan")))
        baseline_rmse = float(oos.get("baseline_rmse", float("nan")))
        comparison = "beats" if np.isfinite(rmse) and np.isfinite(baseline_rmse) and rmse < baseline_rmse else "does not beat"
        notes.append(
            f"OOS {oos.get('target', 'stress')} RMSE {comparison} pipeline ({_format_float(rmse)} versus {_format_float(baseline_rmse)}, DM p={_format_float(oos.get('dm_p', float('nan')))})"
        )
    else:
        notes.append("OOS stress validation coverage is still too thin for a strong calibration read")
    if np.isfinite(latest["X_C_pipeline"]) and abs(latest["X_C_pipeline"]) < 1e-9 and abs(latest["X_C_calibrated"]) >= 0.5:
        notes.append("pipeline X_C is clipped at zero at the sample end, so the calibrated score remains a diagnostic rather than confirmed OOS evidence")
    elif not np.isfinite(latest["X_C_pipeline"]):
        notes.append("pipeline X_C is unavailable at the sample end, so the calibrated implicit score is the usable signal")
    if stress_delta < -0.05:
        notes.append(f"the tuned signal weakens the link to future spread widening from {_format_float(baseline['stress_corr'])} to {_format_float(best['stress_corr'])}")
    if best["stress_pairs"] < 8:
        notes.append("spread coverage is too thin for a forward-spread read")
    if best["loop_pairs"] < 8:
        notes.append("loop-area coverage is still too thin for a strong path-dependence read")

    return CalibrationResult(
        region_key=region_key,
        region_label=str(region_key).upper(),
        coverage_start=str(pd.to_datetime(work["date"]).min().date()),
        coverage_end=str(pd.to_datetime(work["date"]).max().date()),
        params={k: float(v) for k, v in params.items()},
        defaults={k: float(v) for k, v in defaults.items()},
        scales={k: float(v) for k, v in scales.items()},
        objective=float(best["objective"]),
        diagnostics={k: float(v) for k, v in best.items() if k != "objective"},
        baseline_diagnostics=baseline,
        oos_diagnostics=oos,
        latest=latest,
        deltas_4q=deltas_4q,
        notes=notes,
        preprocessing_mode=preprocessing_mode,
        panel_source="direct_frame",
    )


def calibrate_regions(site_dir: Path, *, source_ref: str | None = None, horizon: int = 4, panel_mode: str = "realtime") -> List[CalibrationResult]:
    results: List[CalibrationResult] = []
    for region in load_region_frames(site_dir, source_ref=source_ref, mode=panel_mode):
        result = calibrate_region_frame(region.frame, region.key, horizon=horizon)
        result.region_label = region.label
        result.panel_source = region.source_path
        if result.preprocessing_mode == "dashboard_retrospective" and region.panel_mode == "realtime":
            result.preprocessing_mode = "real_time_release_lagged"
        results.append(result)
    return results


def _write_text(path: Path, text: str) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    return path


def render_calibration_tex(results: Sequence[CalibrationResult], *, source_ref: str | None = None) -> str:
    header = [
        r"\begin{table}[htbp]",
        r"  \centering",
        r"  \small",
        r"  \setlength{\tabcolsep}{4pt}",
        r"  \caption{Out-of-sample diagnostics and comparability metadata for the calibrated implicit headroom score.}",
        r"  \label{tab:theory_calibration}",
        r"  \textit{Panel A. Expanding-window OOS diagnostics.}\par\smallskip",
        r"  \resizebox{\textwidth}{!}{%",
        r"  \begin{tabular}{@{}llrrrrrrrr@{}}",
        r"    \toprule",
        r"    Region & OOS target & End date & $N$ & RMSE & Pipe. RMSE & MAE & AUC & Log score & DM $p$ \\",
        r"    \midrule",
    ]
    rows: List[str] = []
    for result in results:
        rows.append(
            "    "
            + _latex_escape(result.region_label)
            + f" & {_latex_escape(str(result.oos_diagnostics.get('target', 'n/a')))}"
            + f" & {_latex_escape(result.coverage_end)}"
            + f" & {_format_float(result.oos_diagnostics.get('n', float('nan')), 0)}"
            + f" & {_format_oos(result, 'rmse')}"
            + f" & {_format_oos(result, 'baseline_rmse')}"
            + f" & {_format_oos(result, 'mae')}"
            + f" & {_format_oos(result, 'auc')}"
            + f" & {_format_oos(result, 'log_score')}"
            + f" & {_format_oos(result, 'dm_p')} \\\\"
        )
    footer = [
        r"    \bottomrule",
        r"  \end{tabular}",
        r"  }",
        _render_calibration_metadata_panel_tex(results).rstrip(),
    ]
    footer.append(
        r"  \par\smallskip\raggedright\footnotesize "
        + _latex_escape(
            "Each OOS point is generated by expanding-window calibration on observations available before the forecast origin, then evaluated h=4 quarters ahead. RMSE, MAE, AUC, and log score use the tuned score; the Diebold-Mariano p-value compares squared OOS errors against the raw pipeline X_C baseline. Future U growth is not used as the search target because U can be GDP-like."
        )
    )
    if source_ref:
        footer.append(
            "  \\par\\smallskip\\raggedright\\footnotesize "
            + _latex_escape(f"Input indicator panels were loaded from the fresher of the current worktree and {source_ref}.")
        )
    footer.append(r"\end{table}")
    return "\n".join(header + rows + footer) + "\n"


def render_snapshot_tex(results: Sequence[CalibrationResult], *, source_ref: str | None = None) -> str:
    lines = [r"\begin{itemize}"]
    if results:
        coverage = ", ".join(f"{_latex_escape(r.region_label)} through {_latex_escape(r.coverage_end)}" for r in results)
        source_text = f" using the fresher of the worktree and {source_ref}" if source_ref else ""
        lines.append(rf"  \item Coverage in the current theory build{_latex_escape(source_text)}: {coverage}.")
        modes = ", ".join(sorted({_latex_escape(r.preprocessing_mode) for r in results}))
        if modes:
            lines.append(
                rf"  \item Calibration diagnostics use {modes} preprocessing; source panel paths are recorded in the calibration JSON."
            )
    for note in _loop_area_audit_notes(results):
        lines.append(r"  \item " + _latex_escape(note))
    for result in results:
        xc_rank = _format_pct(result.latest.get("X_C_calibrated_pctile", float("nan")))
        pipeline_rank = _format_pct(result.latest.get("X_C_pipeline_pctile", float("nan")))
        loop_rank = _format_pct(result.latest.get("loop_area_pctile", float("nan")))
        note_text = "; ".join(_latex_note_text(note) for note in result.notes[1:])
        limit_text = _missingness_note(result)
        limit_clause = "" if limit_text == "reported columns available" else f" Read limited by {limit_text}."
        lines.append(
            "  \\item "
            + _latex_escape(result.region_label)
            + f": within-region calibrated headroom is {xc_rank}, pipeline $X_C$ is {pipeline_rank}, streaming loop area is {loop_rank}, and {note_text}.{_latex_escape(limit_clause)} No cross-region score-level conclusion is drawn."
        )
    lines.append(r"\end{itemize}")
    return "\n".join(lines) + "\n"


def render_calibration_json(results: Sequence[CalibrationResult], *, source_ref: str | None = None) -> List[Dict[str, Any]]:
    payload: List[Dict[str, Any]] = []
    audit_notes = _loop_area_audit_notes(results)
    for result in results:
        payload.append(
            {
                "region_key": result.region_key,
                "region_label": result.region_label,
                "coverage_start": result.coverage_start,
                "coverage_end": result.coverage_end,
                "source_ref": source_ref,
                "params": result.params,
                "defaults": result.defaults,
                "scales": result.scales,
                "objective": result.objective,
                "diagnostics": result.diagnostics,
                "baseline_diagnostics": result.baseline_diagnostics,
                "oos_diagnostics": result.oos_diagnostics,
                "latest": result.latest,
                "deltas_4q": result.deltas_4q,
                "notes": result.notes,
                "audit_notes": audit_notes,
                "comparison_basis": "within-region percentiles/z-scores only; calibrated levels are not cross-region comparable",
                "preprocessing_mode": result.preprocessing_mode,
                "panel_source": result.panel_source,
            }
        )
    return payload


def write_calibration_outputs(
    results: Sequence[CalibrationResult],
    *,
    output_dir: Path,
    data_dir: Path,
    source_ref: str | None = None,
) -> List[Path]:
    import json

    outputs: List[Path] = []
    outputs.append(_write_text(output_dir / "theory_calibration.tex", render_calibration_tex(results, source_ref=source_ref)))
    outputs.append(_write_text(output_dir / "theory_empirical_snapshot.tex", render_snapshot_tex(results, source_ref=source_ref)))
    payload = render_calibration_json(results, source_ref=source_ref)
    json_path = data_dir / "calibrated_theory_params.json"
    json_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    outputs.append(json_path)
    return outputs


__all__ = [
    "CalibrationResult",
    "PARAMETER_NAMES",
    "calibrate_region_frame",
    "calibrate_regions",
    "render_calibration_tex",
    "render_snapshot_tex",
    "write_calibration_outputs",
]
