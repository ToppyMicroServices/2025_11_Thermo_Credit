from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence

import numpy as np
import pandas as pd

from lib.config_loader import load_config
from lib.theory_figures import RegionFrame, load_region_frames


PARAMETER_NAMES: Sequence[str] = ("T0", "p0", "U0", "V0", "S0")


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
    latest: Dict[str, float]
    deltas_4q: Dict[str, float]
    notes: List[str]


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


def _format_float(value: float, digits: int = 3) -> str:
    if not np.isfinite(value):
        return "n/a"
    return f"{value:,.{digits}f}"


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

    objective = (
        1.40 * (1.0 - growth_corr)
        + 1.05 * (1.0 + stress_corr)
        + 0.85 * (1.0 + loop_corr)
        + 0.45 * (1.0 - liq_corr)
        + 0.70 * neg_share
        + 0.20 * reg_penalty
        + 0.35 * range_penalty
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


def calibrate_region_frame(frame: pd.DataFrame, region_key: str, *, horizon: int = 4, iterations: int = 4) -> CalibrationResult:
    work = frame.copy().sort_values("date").reset_index(drop=True)
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
            for multiplier in (-1.0, -0.5, 0.0, 0.5, 1.0):
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

    x_c = _x_c_from_params(work, params, u_col=u_col, v_col=v_col, s_col=s_col, scales=scales)
    baseline = _baseline_diagnostics(work, horizon=horizon)
    pipeline_series = pd.to_numeric(work.get("X_C"), errors="coerce")
    latest = {
        "S_M": _latest_value(work, "S_M"),
        "T_L": _latest_value(work, "T_L"),
        "X_C_calibrated": float(pd.to_numeric(x_c, errors="coerce").dropna().iloc[-1]) if pd.to_numeric(x_c, errors="coerce").dropna().size else float("nan"),
        "X_C_pipeline": _latest_value(work, "X_C"),
        "loop_area": _latest_value(work, "loop_area"),
    }
    deltas_4q = {
        "S_M": _delta_4q(work, "S_M"),
        "T_L": _delta_4q(work, "T_L"),
        "X_C_calibrated": _delta_from_series(x_c),
        "loop_area": _delta_4q(work, "loop_area"),
    }

    tl_level = _describe_level(pd.to_numeric(work.get("T_L"), errors="coerce"), latest["T_L"])
    xc_delta_desc = _describe_delta(deltas_4q["X_C_calibrated"], _robust_scale(pd.to_numeric(x_c, errors="coerce")), inverse=False)
    loop_desc = _describe_delta(deltas_4q["loop_area"], _robust_scale(pd.to_numeric(work.get("loop_area"), errors="coerce")), inverse=True)
    notes = [
        f"coverage through {pd.to_datetime(work['date']).max().date()}",
        f"T_L is currently {tl_level} in its own regional history",
        f"implicit headroom is {xc_delta_desc} over the last four quarters",
        f"loop dissipation is {loop_desc} over the last four quarters",
    ]
    growth_delta = best["growth_corr"] - baseline["growth_corr"]
    stress_delta = best["stress_corr"] - baseline["stress_corr"]
    if baseline["growth_pairs"] < 8:
        notes.append("pipeline X_C has limited forward-growth coverage, so the implicit score carries the current empirical read")
    elif growth_delta > 0.05:
        notes.append(f"forward-growth correlation improves from {_format_float(baseline['growth_corr'])} to {_format_float(best['growth_corr'])}")
    elif growth_delta < -0.05:
        notes.append(f"forward-growth correlation does not yet improve over pipeline X_C ({_format_float(best['growth_corr'])} versus {_format_float(baseline['growth_corr'])})")
    if np.isfinite(latest["X_C_pipeline"]) and abs(latest["X_C_pipeline"]) < 1e-9 and abs(latest["X_C_calibrated"]) >= 0.5:
        notes.append("pipeline X_C is clipped at zero at the sample end, so calibration is doing real work")
    elif not np.isfinite(latest["X_C_pipeline"]):
        notes.append("pipeline X_C is unavailable at the sample end, so the calibrated implicit score is the usable signal")
    if stress_delta < -0.05:
        notes.append(f"the tuned signal weakens the link to future spread widening from {_format_float(baseline['stress_corr'])} to {_format_float(best['stress_corr'])}")
    if best["loop_pairs"] < 8:
        notes.append("loop-area coverage is still too thin for a strong dissipation read")

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
        latest=latest,
        deltas_4q=deltas_4q,
        notes=notes,
    )


def calibrate_regions(site_dir: Path, *, source_ref: str | None = None, horizon: int = 4) -> List[CalibrationResult]:
    results: List[CalibrationResult] = []
    for region in load_region_frames(site_dir, source_ref=source_ref):
        result = calibrate_region_frame(region.frame, region.key, horizon=horizon)
        result.region_label = region.label
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
        r"  \caption{Illustrative region-by-region calibration of $(T_0,p_0,U_0,V_0,S_0)$.}",
        r"  \label{tab:theory_calibration}",
        r"  \resizebox{\textwidth}{!}{%",
        r"  \begin{tabular}{@{}lrrrrrrrr@{}}",
        r"    \toprule",
        r"    Region & End date & $T_0$ & $p_0$ & $U_0$ & $V_0$ & $S_0$ & Corr. to $\Delta U_{t+4}$ & Corr. to $\Delta \mathrm{spread}_{t+4}$ \\",
        r"    \midrule",
    ]
    rows: List[str] = []
    for result in results:
        rows.append(
            "    "
            + _latex_escape(result.region_label)
            + f" & {_latex_escape(result.coverage_end)}"
            + f" & {_format_float(result.params['T0'])}"
            + f" & {_format_float(result.params['p0'])}"
            + f" & {_format_float(result.params['U0'], 0)}"
            + f" & {_format_float(result.params['V0'], 0)}"
            + f" & {_format_float(result.params['S0'], 0)}"
            + f" & {_format_float(result.diagnostics['growth_corr'])}"
            + f" & {_format_float(result.diagnostics['stress_corr'])} \\\\"
        )
    footer = [
        r"    \bottomrule",
        r"  \end{tabular}",
        r"  }",
    ]
    footer.append(
        r"  \par\smallskip\raggedright\footnotesize "
        + _latex_escape(
            "The search is deliberately lightweight. Level anchors are first centered on each region's sample medians, so the tuned score should be read as an implicit local-headroom index rather than as a raw nominal level."
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
    for result in results:
        xc_latest = _format_float(result.latest["X_C_calibrated"])
        xc_pipeline = _format_float(result.latest["X_C_pipeline"])
        loop_latest = _format_float(result.latest["loop_area"], 0)
        note_text = "; ".join(_latex_escape(note) for note in result.notes[1:])
        lines.append(
            "  \\item "
            + _latex_escape(result.region_label)
            + f": the calibrated implicit headroom score is {xc_latest}, pipeline $X_C$ is {xc_pipeline}, latest loop area is {loop_latest}, and {note_text}."
        )
    lines.append(r"\end{itemize}")
    return "\n".join(lines) + "\n"


def render_calibration_json(results: Sequence[CalibrationResult], *, source_ref: str | None = None) -> List[Dict[str, Any]]:
    payload: List[Dict[str, Any]] = []
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
                "latest": result.latest,
                "deltas_4q": result.deltas_4q,
                "notes": result.notes,
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
