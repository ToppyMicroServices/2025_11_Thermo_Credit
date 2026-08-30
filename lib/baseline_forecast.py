from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd

from lib.forecast_frames import RegionFrame, load_region_frames


HORIZON_QUARTERS = 4
MIN_TRAINING_ROWS = 28
BOOTSTRAP_REPS = 400
BLOCK_LENGTH = 4


BASELINE_NAMES: Sequence[str] = (
    "AR(1)",
    "total_credit_growth",
    "credit_to_gdp_gap",
    "spread_only",
    "money_growth",
    "simple_fci",
)

PANEL_BASELINE_NAME = "region_fixed_effect_panel"

CANDIDATE_NAMES: Sequence[str] = (
    "baseline_only",
    "baseline_plus_q_t",
    "baseline_plus_SM",
    "baseline_plus_TL",
    "baseline_plus_XC",
    "baseline_plus_loop_area",
    "full_thermo_credit",
)


@dataclass(frozen=True)
class ForecastTarget:
    key: str
    label: str
    target_type: str
    source_column: str
    outcome: pd.Series
    ar1_feature: pd.Series
    event_mode: str = ""
    event_quantile: float = float("nan")


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
        series = pd.to_numeric(frame[column], errors="coerce").dropna()
        if series.size >= min_rows and float(series.std(ddof=0)) > 1e-12:
            return column
    return None


def _positive_log(series: pd.Series) -> pd.Series:
    numeric = pd.to_numeric(series, errors="coerce")
    return pd.Series(np.log(numeric.where(numeric > 0)), index=series.index, dtype=float)


def _forward_growth(level: pd.Series, horizon: int) -> pd.Series:
    logged = _positive_log(level)
    return logged.shift(-horizon) - logged


def _trailing_growth(level: pd.Series, horizon: int) -> pd.Series:
    logged = _positive_log(level)
    return logged - logged.shift(horizon)


def _forward_acceleration(level: pd.Series, horizon: int) -> pd.Series:
    logged = _positive_log(level)
    future = logged.shift(-horizon) - logged
    trailing = logged - logged.shift(horizon)
    return future - trailing


def _trailing_acceleration(level: pd.Series, horizon: int) -> pd.Series:
    logged = _positive_log(level)
    now = logged - logged.shift(horizon)
    prev = logged.shift(horizon) - logged.shift(2 * horizon)
    return now - prev


def _expanding_z(series: pd.Series, min_periods: int = 8) -> pd.Series:
    numeric = pd.to_numeric(series, errors="coerce")
    mean = numeric.expanding(min_periods=min_periods).mean()
    std = numeric.expanding(min_periods=min_periods).std(ddof=0)
    return (numeric - mean) / std.replace(0.0, np.nan)


def _trailing_volatility(series: pd.Series, horizon: int) -> pd.Series:
    numeric = pd.to_numeric(series, errors="coerce")
    change = numeric.diff()
    return change.rolling(window=horizon, min_periods=max(3, horizon // 2)).std()


def _stress_score(frame: pd.DataFrame) -> pd.Series:
    parts: list[pd.Series] = []
    for column in ("spread", "T_L", "p_C", "loop_area"):
        if column in frame.columns:
            parts.append(_expanding_z(_safe_numeric(frame, column)))
    if "V_C" in frame.columns:
        parts.append(-_expanding_z(_safe_numeric(frame, "V_C")))
    if not parts:
        return pd.Series(np.nan, index=frame.index, dtype=float)
    return pd.concat(parts, axis=1).mean(axis=1, skipna=True)


def build_forecast_targets(frame: pd.DataFrame, *, horizon: int = HORIZON_QUARTERS) -> tuple[list[ForecastTarget], pd.DataFrame]:
    targets: list[ForecastTarget] = []
    coverage: list[dict[str, Any]] = []

    growth_col = _pick_numeric_column(frame, ("Y", "industrial_production", "IP", "U_gdp_only", "U", "L_real"))
    if growth_col:
        targets.append(
            ForecastTarget(
                key="real_growth",
                label="real GDP / activity growth",
                target_type="continuous",
                source_column=growth_col,
                outcome=_forward_growth(frame[growth_col], horizon),
                ar1_feature=_trailing_growth(frame[growth_col], horizon),
            )
        )
        coverage.append({"target": "real_growth", "status": "available", "source_column": growth_col})
    else:
        coverage.append({"target": "real_growth", "status": "unavailable", "source_column": ""})

    price_col = _pick_numeric_column(frame, ("inflation", "CPI", "P", "price_level", "deflator"))
    if price_col:
        price = _safe_numeric(frame, price_col)
        outcome = price.shift(-horizon) - price if price_col == "inflation" else _forward_growth(price, horizon)
        ar1 = price - price.shift(horizon) if price_col == "inflation" else _trailing_growth(price, horizon)
        targets.append(
            ForecastTarget(
                key="inflation",
                label="inflation",
                target_type="continuous",
                source_column=price_col,
                outcome=outcome,
                ar1_feature=ar1,
            )
        )
        coverage.append({"target": "inflation", "status": "available", "source_column": price_col})
    else:
        coverage.append({"target": "inflation", "status": "unavailable", "source_column": ""})

    asset_col = _pick_numeric_column(frame, ("house_price", "land_price", "equity_price", "asset_price", "A", "L_asset"))
    if asset_col:
        targets.append(
            ForecastTarget(
                key="asset_acceleration",
                label="house/land/equity acceleration",
                target_type="continuous",
                source_column=asset_col,
                outcome=_forward_acceleration(frame[asset_col], horizon),
                ar1_feature=_trailing_acceleration(frame[asset_col], horizon),
            )
        )
        coverage.append({"target": "asset_acceleration", "status": "available", "source_column": asset_col})
    else:
        coverage.append({"target": "asset_acceleration", "status": "unavailable", "source_column": ""})

    spread_col = _pick_numeric_column(frame, ("spread", "hy_oas", "credit_spread"))
    if spread_col:
        spread = _safe_numeric(frame, spread_col)
        targets.append(
            ForecastTarget(
                key="spread_widening",
                label="credit spread widening",
                target_type="binary",
                source_column=spread_col,
                outcome=spread.shift(-horizon) - spread,
                ar1_feature=spread - spread.shift(horizon),
                event_mode="fixed_zero_high",
            )
        )
        coverage.append({"target": "spread_widening", "status": "available", "source_column": spread_col})
    else:
        coverage.append({"target": "spread_widening", "status": "unavailable", "source_column": ""})

    stress = _stress_score(frame)
    if stress.dropna().size >= 16 and float(stress.dropna().std(ddof=0)) > 1e-12:
        targets.append(
            ForecastTarget(
                key="stress_regime",
                label="recession / stress regime proxy",
                target_type="binary",
                source_column="stress_score_proxy",
                outcome=stress.shift(-horizon),
                ar1_feature=stress,
                event_mode="upper_quantile",
                event_quantile=0.75,
            )
        )
        coverage.append({"target": "stress_regime", "status": "available", "source_column": "stress_score_proxy"})
    else:
        coverage.append({"target": "stress_regime", "status": "unavailable", "source_column": ""})

    if growth_col:
        growth = _forward_growth(frame[growth_col], horizon)
        targets.append(
            ForecastTarget(
                key="downside_growth",
                label="downside lower-tail growth",
                target_type="binary",
                source_column=growth_col,
                outcome=growth,
                ar1_feature=_trailing_growth(frame[growth_col], horizon),
                event_mode="lower_quantile",
                event_quantile=0.20,
            )
        )
        coverage.append({"target": "downside_growth", "status": "available", "source_column": growth_col})
    else:
        coverage.append({"target": "downside_growth", "status": "unavailable", "source_column": ""})

    vol_source = spread_col or asset_col
    if vol_source:
        vol = _trailing_volatility(_safe_numeric(frame, vol_source), horizon)
        if vol.dropna().size >= 16 and float(vol.dropna().std(ddof=0)) > 1e-12:
            targets.append(
                ForecastTarget(
                    key="volatility_spike",
                    label="volatility spike",
                    target_type="binary",
                    source_column=vol_source,
                    outcome=vol.shift(-horizon),
                    ar1_feature=vol,
                    event_mode="upper_quantile",
                    event_quantile=0.80,
                )
            )
            coverage.append({"target": "volatility_spike", "status": "available", "source_column": vol_source})
        else:
            coverage.append({"target": "volatility_spike", "status": "unavailable", "source_column": vol_source or ""})
    else:
        coverage.append({"target": "volatility_spike", "status": "unavailable", "source_column": ""})

    return targets, pd.DataFrame(coverage)


def _credit_to_gdp_gap(frame: pd.DataFrame) -> pd.Series:
    credit = _safe_numeric(frame, "L_real")
    gdp_col = _pick_numeric_column(frame, ("Y", "U_gdp_only"))
    if gdp_col is None:
        return pd.Series(np.nan, index=frame.index, dtype=float)
    gdp = _safe_numeric(frame, gdp_col).where(lambda s: s > 0)
    ratio = credit / gdp
    trend = ratio.expanding(min_periods=12).mean()
    return ratio - trend


def build_base_features(frame: pd.DataFrame, *, horizon: int = HORIZON_QUARTERS) -> pd.DataFrame:
    out = pd.DataFrame(index=frame.index)
    out["total_credit_growth"] = _trailing_growth(_safe_numeric(frame, "L_real"), horizon)
    out["credit_to_gdp_gap"] = _credit_to_gdp_gap(frame)
    spread_col = _pick_numeric_column(frame, ("spread", "hy_oas", "credit_spread"))
    out["spread_only"] = _safe_numeric(frame, spread_col) if spread_col else np.nan
    money_col = _pick_numeric_column(frame, ("M_in", "M2", "money"))
    out["money_growth"] = _trailing_growth(_safe_numeric(frame, money_col), horizon) if money_col else np.nan
    fci_parts = []
    for column in ("spread", "T_L", "p_C"):
        if column in frame.columns:
            fci_parts.append(_expanding_z(_safe_numeric(frame, column)))
    if "V_C" in frame.columns:
        fci_parts.append(-_expanding_z(_safe_numeric(frame, "V_C")))
    out["simple_fci"] = pd.concat(fci_parts, axis=1).mean(axis=1, skipna=True) if fci_parts else np.nan
    out["q_t"] = _safe_numeric(frame, "q_t")
    out["SM"] = _safe_numeric(frame, "S_M_hat") if "S_M_hat" in frame.columns else _safe_numeric(frame, "S_M")
    out["TL"] = _safe_numeric(frame, "T_L")
    out["XC"] = _safe_numeric(frame, "X_C")
    out["loop_area"] = _safe_numeric(frame, "loop_area")
    out["C_R"] = _safe_numeric(frame, "C_R")
    out["C_A"] = _safe_numeric(frame, "C_A")
    out["destination_coverage"] = _safe_numeric(frame, "destination_coverage")
    return out


def _baseline_columns(name: str, target: ForecastTarget) -> list[str]:
    if name == "AR(1)":
        return ["AR1"]
    return [name]


def _candidate_extra_columns(name: str) -> list[str]:
    mapping = {
        "baseline_only": [],
        "baseline_plus_q_t": ["q_t"],
        "baseline_plus_SM": ["SM"],
        "baseline_plus_TL": ["TL"],
        "baseline_plus_XC": ["XC"],
        "baseline_plus_loop_area": ["loop_area"],
        "full_thermo_credit": ["q_t", "SM", "TL", "XC", "loop_area", "C_R", "C_A", "destination_coverage"],
    }
    return mapping[name]


def _feature_matrix(features: pd.DataFrame, columns: Sequence[str]) -> pd.DataFrame:
    if not columns:
        return pd.DataFrame(index=features.index)
    return features.loc[:, list(columns)].copy()


def _prune_sparse_full_features(features: pd.DataFrame, columns: Sequence[str]) -> tuple[list[str], list[str]]:
    kept: list[str] = []
    dropped: list[str] = []
    for column in columns:
        if column not in features.columns:
            dropped.append(column)
            continue
        valid_count = int(pd.to_numeric(features[column], errors="coerce").notna().sum())
        if valid_count >= MIN_TRAINING_ROWS:
            kept.append(column)
        else:
            dropped.append(column)
    return kept, dropped


def _resolved_model_columns(
    features: pd.DataFrame,
    baseline_columns: Sequence[str],
    candidate_name: str,
) -> tuple[list[str], list[str]]:
    extras = _candidate_extra_columns(candidate_name)
    model_cols = list(dict.fromkeys(list(baseline_columns) + extras))
    if candidate_name != "full_thermo_credit":
        return model_cols, []
    baseline_set = set(baseline_columns)
    kept_extras, dropped_extras = _prune_sparse_full_features(features, extras)
    return list(dict.fromkeys(list(baseline_columns) + kept_extras)), [
        col for col in dropped_extras if col not in baseline_set
    ]


def _fit_predict_ridge(x_train: pd.DataFrame, y_train: pd.Series, x_test: pd.Series) -> float:
    test_raw = pd.to_numeric(x_test, errors="coerce")
    if test_raw.isna().any():
        return float("nan")
    train = pd.concat([x_train, pd.to_numeric(y_train, errors="coerce").rename("__target__")], axis=1).dropna()
    if train.shape[0] < 8:
        return float("nan")
    y = train["__target__"].to_numpy(dtype=float)
    x = train.drop(columns=["__target__"])
    means = x.mean(axis=0)
    stds = x.std(axis=0, ddof=0).replace(0.0, np.nan)
    x_std = ((x - means) / stds).replace([np.inf, -np.inf], np.nan).fillna(0.0)
    test = ((test_raw - means) / stds).replace([np.inf, -np.inf], np.nan).fillna(0.0)
    if test.isna().any():
        return float("nan")
    design = np.column_stack([np.ones(len(x_std)), x_std.to_numpy(dtype=float)])
    alpha = 1e-6
    penalty = np.eye(design.shape[1]) * alpha
    penalty[0, 0] = 0.0
    beta = np.linalg.pinv(design.T @ design + penalty) @ design.T @ y
    test_design = np.concatenate([[1.0], test.to_numpy(dtype=float)])
    return float(test_design @ beta)


def _threshold_from_train(target: ForecastTarget, train_target: pd.Series) -> float:
    valid = pd.to_numeric(train_target, errors="coerce").dropna()
    if valid.empty:
        return float("nan")
    if target.event_mode == "fixed_zero_high":
        return 0.0
    if target.event_mode == "upper_quantile":
        return float(valid.quantile(target.event_quantile))
    if target.event_mode == "lower_quantile":
        return float(valid.quantile(target.event_quantile))
    return 0.0


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


def _block_bootstrap_ci(loss_diff: np.ndarray, *, block_length: int = BLOCK_LENGTH, reps: int = BOOTSTRAP_REPS) -> tuple[float, float]:
    values = np.asarray(loss_diff, dtype=float)
    values = values[np.isfinite(values)]
    n = len(values)
    if n < 8:
        return float("nan"), float("nan")
    rng = np.random.default_rng(20260529)
    means = []
    starts = np.arange(n)
    for _ in range(reps):
        sample: list[float] = []
        while len(sample) < n:
            start = int(rng.choice(starts))
            block = [values[(start + offset) % n] for offset in range(block_length)]
            sample.extend(block)
        means.append(float(np.mean(sample[:n])))
    return float(np.quantile(means, 0.025)), float(np.quantile(means, 0.975))


def _score_oos(
    frame: pd.DataFrame,
    target: ForecastTarget,
    baseline_x: pd.DataFrame,
    model_x: pd.DataFrame,
    *,
    min_training_rows: int = MIN_TRAINING_ROWS,
    dates: pd.Series | None = None,
) -> dict[str, Any]:
    actual_z: list[float] = []
    pred_z: list[float] = []
    base_z: list[float] = []
    events: list[int] = []
    probs: list[float] = []
    base_probs: list[float] = []
    event_scores: list[float] = []

    y = pd.to_numeric(target.outcome, errors="coerce")
    date_values = pd.to_datetime(dates, errors="coerce") if dates is not None else None
    candidate_indices = range(len(frame)) if date_values is not None else range(min_training_rows, len(frame))
    for idx in candidate_indices:
        actual = float(y.iloc[idx]) if idx < len(y) and np.isfinite(y.iloc[idx]) else float("nan")
        if not np.isfinite(actual):
            continue
        if date_values is not None:
            current_date = date_values.iloc[idx]
            if pd.isna(current_date):
                continue
            train_mask = date_values < current_date
            train_idx = np.flatnonzero(train_mask.to_numpy())
            if len(train_idx) < min_training_rows:
                continue
            train_y = y.iloc[train_idx]
            model_pred = _fit_predict_ridge(model_x.iloc[train_idx], train_y, model_x.iloc[idx])
            base_pred = _fit_predict_ridge(baseline_x.iloc[train_idx], train_y, baseline_x.iloc[idx])
        else:
            train_y = y.iloc[:idx]
            model_pred = _fit_predict_ridge(model_x.iloc[:idx], train_y, model_x.iloc[idx])
            base_pred = _fit_predict_ridge(baseline_x.iloc[:idx], train_y, baseline_x.iloc[idx])
        valid_train = train_y.dropna()
        if valid_train.size < 8 or not (np.isfinite(model_pred) and np.isfinite(base_pred)):
            continue
        y_mean = float(valid_train.mean())
        y_std = float(valid_train.std(ddof=0))
        if not np.isfinite(y_std) or y_std <= 1e-12:
            continue
        actual_std = (actual - y_mean) / y_std
        model_std = (model_pred - y_mean) / y_std
        base_std = (base_pred - y_mean) / y_std
        actual_z.append(float(actual_std))
        pred_z.append(float(model_std))
        base_z.append(float(base_std))
        if target.target_type == "binary":
            threshold = _threshold_from_train(target, train_y)
            if not np.isfinite(threshold):
                continue
            threshold_z = (threshold - y_mean) / y_std
            if target.event_mode == "lower_quantile":
                event = int(actual_std < threshold_z)
                score = threshold_z - model_std
                base_score = threshold_z - base_std
            else:
                event = int(actual_std > threshold_z)
                score = model_std - threshold_z
                base_score = base_std - threshold_z
            events.append(event)
            event_scores.append(float(score))
            probs.append(float(1.0 / (1.0 + np.exp(-np.clip(score, -30, 30)))))
            base_probs.append(float(1.0 / (1.0 + np.exp(-np.clip(base_score, -30, 30)))))

    n = len(actual_z)
    if n < 8:
        return {"n": float(n), "metric_loss_diff": float("nan")}
    actual_arr = np.asarray(actual_z, dtype=float)
    pred_arr = np.asarray(pred_z, dtype=float)
    base_arr = np.asarray(base_z, dtype=float)

    if target.target_type == "binary" and len(events) >= 8:
        event_arr = np.asarray(events, dtype=float)
        prob_arr = np.clip(np.asarray(probs, dtype=float), 1e-6, 1.0 - 1e-6)
        base_prob_arr = np.clip(np.asarray(base_probs, dtype=float), 1e-6, 1.0 - 1e-6)
        brier = float(np.mean((prob_arr - event_arr) ** 2))
        base_brier = float(np.mean((base_prob_arr - event_arr) ** 2))
        loss_diff = (prob_arr - event_arr) ** 2 - (base_prob_arr - event_arr) ** 2
        log_score = float(-np.mean(event_arr * np.log(prob_arr) + (1 - event_arr) * np.log(1 - prob_arr)))
        base_log_score = float(-np.mean(event_arr * np.log(base_prob_arr) + (1 - event_arr) * np.log(1 - base_prob_arr)))
        rmse = float(np.sqrt(brier))
        mae = float(np.mean(np.abs(prob_arr - event_arr)))
        auc = _auc_score(event_scores, events)
        ci_low, ci_high = _block_bootstrap_ci(loss_diff)
        dm_p = _dm_pvalue(loss_diff)
        return {
            "n": float(len(events)),
            "rmse": rmse,
            "mae": mae,
            "auc": auc,
            "brier": brier,
            "log_score": log_score,
            "baseline_rmse": float(np.sqrt(base_brier)),
            "baseline_mae": float(np.mean(np.abs(base_prob_arr - event_arr))),
            "baseline_brier": base_brier,
            "baseline_log_score": base_log_score,
            "metric_loss_diff": float(np.mean(loss_diff)),
            "dm_p": dm_p,
            "block_ci_low": ci_low,
            "block_ci_high": ci_high,
        }

    errors = pred_arr - actual_arr
    base_errors = base_arr - actual_arr
    loss_diff = errors**2 - base_errors**2
    ci_low, ci_high = _block_bootstrap_ci(loss_diff)
    return {
        "n": float(n),
        "rmse": float(np.sqrt(np.mean(errors**2))),
        "mae": float(np.mean(np.abs(errors))),
        "auc": float("nan"),
        "brier": float("nan"),
        "log_score": float("nan"),
        "baseline_rmse": float(np.sqrt(np.mean(base_errors**2))),
        "baseline_mae": float(np.mean(np.abs(base_errors))),
        "baseline_brier": float("nan"),
        "baseline_log_score": float("nan"),
        "metric_loss_diff": float(np.mean(loss_diff)),
        "dm_p": _dm_pvalue(loss_diff),
        "block_ci_low": ci_low,
        "block_ci_high": ci_high,
    }


def _dm_pvalue(loss_diff: np.ndarray) -> float:
    values = np.asarray(loss_diff, dtype=float)
    values = values[np.isfinite(values)]
    n = len(values)
    if n < 8 or float(np.std(values, ddof=1)) <= 1e-12:
        return float("nan")
    stat = float(np.mean(values) / (np.std(values, ddof=1) / np.sqrt(n)))
    return float(2.0 * (1.0 - _normal_cdf(abs(stat))))


def evaluate_region_forecasts(region: RegionFrame, *, horizon: int = HORIZON_QUARTERS) -> tuple[pd.DataFrame, pd.DataFrame]:
    frame = region.frame.copy().sort_values("date").reset_index(drop=True)
    targets, coverage = build_forecast_targets(frame, horizon=horizon)
    coverage.insert(0, "region_key", region.key)
    coverage.insert(1, "region_label", region.label)
    coverage["panel_source"] = region.source_path
    features = build_base_features(frame, horizon=horizon)
    rows: list[dict[str, Any]] = []
    for target in targets:
        local_features = features.copy()
        local_features["AR1"] = target.ar1_feature
        for baseline_name in BASELINE_NAMES:
            base_cols = _baseline_columns(baseline_name, target)
            baseline_x = _feature_matrix(local_features, base_cols)
            for candidate_name in CANDIDATE_NAMES:
                model_cols, dropped_cols = _resolved_model_columns(local_features, base_cols, candidate_name)
                model_x = _feature_matrix(local_features, model_cols)
                metrics = _score_oos(frame, target, baseline_x, model_x)
                rows.append(
                    {
                        "region_key": region.key,
                        "region_label": region.label,
                        "panel_source": region.source_path,
                        "horizon_quarters": int(horizon),
                        "target": target.key,
                        "target_label": target.label,
                        "target_type": target.target_type,
                        "target_source": target.source_column,
                        "baseline": baseline_name,
                        "candidate": candidate_name,
                        "model_features": ",".join(model_cols),
                        "dropped_model_features": ",".join(dropped_cols),
                        **metrics,
                    }
                )
    return pd.DataFrame(rows), coverage


def _panel_baseline_columns(features: pd.DataFrame) -> list[str]:
    region_cols = [column for column in features.columns if column.startswith("fe_")]
    plain_cols = ["AR1", "total_credit_growth", "credit_to_gdp_gap", "spread_only", "money_growth", "simple_fci"]
    return [column for column in region_cols + plain_cols if column in features.columns]


def _within_region_panel_features(features: pd.DataFrame) -> pd.DataFrame:
    out = features.copy()
    columns = [
        "AR1",
        "total_credit_growth",
        "credit_to_gdp_gap",
        "spread_only",
        "money_growth",
        "simple_fci",
        "q_t",
        "SM",
        "TL",
        "XC",
        "loop_area",
        "C_R",
        "C_A",
        "destination_coverage",
    ]
    for column in columns:
        if column in out.columns:
            out.loc[:, column] = _expanding_z(out[column]).to_numpy()
    return out


def evaluate_region_fe_panel(regions: Sequence[RegionFrame], *, horizon: int = HORIZON_QUARTERS) -> pd.DataFrame:
    prepared: list[tuple[RegionFrame, pd.DataFrame, dict[str, ForecastTarget], pd.DataFrame]] = []
    for region in regions:
        frame = region.frame.copy().sort_values("date").reset_index(drop=True)
        targets, _coverage = build_forecast_targets(frame, horizon=horizon)
        features = build_base_features(frame, horizon=horizon)
        prepared.append((region, frame, {target.key: target for target in targets}, features))

    target_keys = sorted({target_key for _region, _frame, targets, _features in prepared for target_key in targets})
    rows: list[dict[str, Any]] = []
    for target_key in target_keys:
        panel_parts: list[pd.DataFrame] = []
        target_labels: list[str] = []
        target_types: list[str] = []
        target_sources: list[str] = []
        for region, frame, targets, features in prepared:
            target = targets.get(target_key)
            if target is None:
                continue
            local = features.copy()
            local["AR1"] = target.ar1_feature
            local = _within_region_panel_features(local)
            local["date"] = pd.to_datetime(frame["date"], errors="coerce") if "date" in frame else pd.NaT
            local["region_key"] = region.key
            local["region_label"] = region.label
            local["__outcome__"] = target.outcome
            panel_parts.append(local)
            target_labels.append(target.label)
            target_types.append(target.target_type)
            target_sources.append(target.source_column)

        if not panel_parts:
            continue
        panel = pd.concat(panel_parts, ignore_index=True).sort_values(["date", "region_key"]).reset_index(drop=True)
        for region_key in sorted(panel["region_key"].dropna().astype(str).unique()):
            panel.loc[:, f"fe_{region_key}"] = (panel["region_key"].astype(str) == region_key).astype(float).to_numpy()
        base_cols = _panel_baseline_columns(panel)
        if not base_cols:
            continue
        target = ForecastTarget(
            key=target_key,
            label=target_labels[0],
            target_type=target_types[0],
            source_column=";".join(sorted(set(target_sources))),
            outcome=pd.to_numeric(panel["__outcome__"], errors="coerce"),
            ar1_feature=pd.to_numeric(panel["AR1"], errors="coerce"),
        )
        # Restore event metadata from the first contributing target.
        for _region, _frame, targets, _features in prepared:
            source_target = targets.get(target_key)
            if source_target is not None:
                target = ForecastTarget(
                    key=target.key,
                    label=target.label,
                    target_type=target.target_type,
                    source_column=target.source_column,
                    outcome=target.outcome,
                    ar1_feature=target.ar1_feature,
                    event_mode=source_target.event_mode,
                    event_quantile=source_target.event_quantile,
                )
                break
        baseline_x = _feature_matrix(panel, base_cols)
        for candidate_name in CANDIDATE_NAMES:
            model_cols, dropped_cols = _resolved_model_columns(panel, base_cols, candidate_name)
            model_x = _feature_matrix(panel, model_cols)
            metrics = _score_oos(panel, target, baseline_x, model_x, dates=panel["date"])
            rows.append(
                {
                    "region_key": "panel",
                    "region_label": "Pooled region FE",
                    "panel_source": ";".join(sorted({region.source_path for region, *_ in prepared})),
                    "horizon_quarters": int(horizon),
                    "target": target.key,
                    "target_label": target.label,
                    "target_type": target.target_type,
                    "target_source": target.source_column,
                    "baseline": PANEL_BASELINE_NAME,
                    "candidate": candidate_name,
                    "model_features": ",".join(model_cols),
                    "dropped_model_features": ",".join(dropped_cols),
                    **metrics,
                }
            )
    return pd.DataFrame(rows)


def run_baseline_forecast_comparison(
    site_dir: Path,
    *,
    source_ref: str | None = None,
    panel_mode: str = "realtime",
    horizon: int = HORIZON_QUARTERS,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    result_parts: list[pd.DataFrame] = []
    coverage_parts: list[pd.DataFrame] = []
    regions = list(load_region_frames(site_dir, source_ref=source_ref, mode=panel_mode))
    for region in regions:
        results, coverage = evaluate_region_forecasts(region, horizon=horizon)
        if not results.empty:
            result_parts.append(results)
        coverage_parts.append(coverage)
    panel_results = evaluate_region_fe_panel(regions, horizon=horizon)
    if not panel_results.empty:
        result_parts.append(panel_results)
    results_df = pd.concat(result_parts, ignore_index=True) if result_parts else pd.DataFrame()
    coverage_df = pd.concat(coverage_parts, ignore_index=True) if coverage_parts else pd.DataFrame()
    summary_df = summarize_baseline_forecasts(results_df)
    return results_df, coverage_df, summary_df


def summarize_baseline_forecasts(results: pd.DataFrame) -> pd.DataFrame:
    if results.empty:
        return pd.DataFrame()
    rows: list[dict[str, Any]] = []
    keys = ["region_key", "region_label", "target", "target_label", "target_type", "target_source"]
    for group_key, group in results.groupby(keys, dropna=False):
        region_key, region_label, target, target_label, target_type, target_source = group_key
        tc = group[(group["baseline"] == "total_credit_growth") & (group["candidate"] == "full_thermo_credit")]
        if tc.empty:
            continue
        row = tc.iloc[0]
        metric_name = "brier" if target_type == "binary" else "rmse"
        baseline_metric_name = "baseline_brier" if target_type == "binary" else "baseline_rmse"
        model_metric = float(row.get(metric_name, float("nan")))
        baseline_metric = float(row.get(baseline_metric_name, float("nan")))
        loss_diff = float(row.get("metric_loss_diff", float("nan")))
        ci_low = float(row.get("block_ci_low", float("nan")))
        ci_high = float(row.get("block_ci_high", float("nan")))
        n = float(row.get("n", float("nan")))
        if not np.isfinite(model_metric) or n < 8:
            status = "insufficient coverage"
        elif np.isfinite(ci_high) and ci_high < 0:
            status = "beats total credit"
        elif np.isfinite(loss_diff) and loss_diff < 0:
            status = "weak improvement"
        else:
            status = "no improvement"
        rows.append(
            {
                "region_key": region_key,
                "region_label": region_label,
                "target": target,
                "target_label": target_label,
                "target_type": target_type,
                "target_source": target_source,
                "n": n,
                "full_metric": model_metric,
                "total_credit_metric": baseline_metric,
                "metric": metric_name,
                "loss_diff_vs_total_credit": loss_diff,
                "block_ci_low": ci_low,
                "block_ci_high": ci_high,
                "dm_p": float(row.get("dm_p", float("nan"))),
                "status": status,
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


def render_baseline_forecast_tex(summary: pd.DataFrame, coverage: pd.DataFrame) -> str:
    lines = [
        r"\begin{table}[htbp]",
        r"  \centering",
        r"  \small",
        r"  \setlength{\tabcolsep}{3pt}",
        r"  \caption{Auxiliary proxy-panel forecast comparison for the monitoring set.}",
        r"  \label{tab:baseline_forecast_comparison}",
        r"  \resizebox{\textwidth}{!}{%",
        r"  \begin{tabular}{@{}lllllllll@{}}",
        r"    \toprule",
        r"    Region & Target & Source & Metric & $N$ & Full & Total-credit & $\Delta$ loss CI & Status \\",
        r"    \midrule",
    ]
    if summary.empty:
        lines.append(r"    No forecast-comparison results were generated. \\")
    else:
        for _, row in summary.sort_values(["region_key", "target"]).iterrows():
            ci_text = f"[{_format_float(row['block_ci_low'])}, {_format_float(row['block_ci_high'])}]"
            lines.append(
                "    "
                + _latex_escape(row["region_label"])
                + " & "
                + _latex_escape(row["target_label"])
                + " & "
                + _latex_escape(row["target_source"])
                + " & "
                + _latex_escape(row["metric"])
                + " & "
                + _format_float(row["n"], 0)
                + " & "
                + _format_float(row["full_metric"])
                + " & "
                + _format_float(row["total_credit_metric"])
                + " & "
                + _latex_escape(ci_text)
                + " & "
                + _latex_escape(row["status"])
                + r" \\"
            )
    unavailable = coverage[(coverage["status"] == "unavailable")] if not coverage.empty else pd.DataFrame()
    missing_targets = ", ".join(sorted(set(unavailable["target"].astype(str)))) if not unavailable.empty else "none"
    lines.extend(
        [
            r"    \bottomrule",
            r"  \end{tabular}",
            r"  }",
            r"  \par\smallskip\raggedright\footnotesize "
            + _latex_escape(
                "Rows are auxiliary dashboard checks, not cross-country evidence for the credit-destination claim. EU/US rows use allocation-proxy panels; the main empirical test is the Japan destination-share OOS table. Continuous targets report RMSE; binary targets report Brier score. The CI is a block-bootstrap interval for candidate loss minus baseline loss."
            ),
            r"  \par\smallskip\raggedright\footnotesize "
            + _latex_escape(
                "The companion CSV also reports AR(1), credit-to-GDP-gap, spread-only, money-growth, simple-FCI, and pooled region fixed-effect baselines; pooled rows use within-region expanding z-scores and should be read as a schema check."
            ),
            r"  \par\smallskip\raggedright\footnotesize "
            + _latex_escape(f"Unavailable target families in at least one region: {missing_targets}."),
            r"\end{table}",
        ]
    )
    return "\n".join(lines) + "\n"


def write_baseline_forecast_outputs(
    results: pd.DataFrame,
    coverage: pd.DataFrame,
    summary: pd.DataFrame,
    *,
    root: Path,
) -> list[Path]:
    site_dir = root / "site"
    data_dir = root / "data"
    tex_dir = root / "tex" / "generated"
    site_dir.mkdir(parents=True, exist_ok=True)
    data_dir.mkdir(parents=True, exist_ok=True)
    tex_dir.mkdir(parents=True, exist_ok=True)

    results_path = site_dir / "baseline_forecast_comparison.csv"
    coverage_path = site_dir / "baseline_forecast_target_coverage.csv"
    summary_path = data_dir / "baseline_forecast_summary.json"
    tex_path = tex_dir / "theory_baseline_forecast_comparison.tex"
    results.to_csv(results_path, index=False)
    coverage.to_csv(coverage_path, index=False)
    payload = {
        "horizon_quarters": HORIZON_QUARTERS,
        "summary": json.loads(summary.to_json(orient="records")),
        "coverage": json.loads(coverage.to_json(orient="records")),
    }
    summary_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    tex_path.write_text(render_baseline_forecast_tex(summary, coverage), encoding="utf-8")
    return [results_path, coverage_path, summary_path, tex_path]


__all__ = [
    "build_forecast_targets",
    "evaluate_region_forecasts",
    "evaluate_region_fe_panel",
    "render_baseline_forecast_tex",
    "run_baseline_forecast_comparison",
    "summarize_baseline_forecasts",
    "write_baseline_forecast_outputs",
]
