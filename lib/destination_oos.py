from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd

from lib.baseline_forecast import (
    BLOCK_LENGTH,
    ForecastTarget,
    MIN_TRAINING_ROWS,
    _block_bootstrap_ci,
    _feature_matrix,
    _safe_numeric,
    _threshold_from_train,
    build_base_features,
    build_forecast_targets,
)
from lib.forecast_frames import RegionFrame, load_region_frames
from lib.boj_credit_taxonomies import (
    MULLER_VERNER_TAXONOMY_ID,
    PRIMARY_TAXONOMY_ID,
    WERNER_TAXONOMY_ID,
)


TARGET_KEYS: Sequence[str] = ("asset_acceleration", "spread_widening")
HORIZONS: Sequence[int] = (4, 8)
ANALYSIS_START = pd.Timestamp("2009-09-30")
BOJ_COMMON_TAXONOMY_START = pd.Timestamp("2009-06-30")
BOJ_RELEASE_LAG_DAYS = 90
BOJ_DIRECT_STOCK_COLUMN = "primary_included_stock"
BOJ_MATCHED_STOCK_COLUMN = "boj_primary_included_stock"
BOJ_MATCHED_GROWTH_COLUMN = "boj_primary_included_stock_growth"
BOJ_BASELINE_UNIVERSE = "primary_included_stock"
ALLOCATION_FEATURE_COLUMN = "borrower_composition_coordinate"
LEGACY_ALLOCATION_FEATURE_COLUMN = "borrower_composition_G"
PRIMARY_ALLOCATION_MEASURE = "bezemer_nfb_4q"
ALLOCATION_MEASURES: Sequence[str] = (
    "bezemer_nfb_4q",
    "werner_fcp_4q",
    "muller_verner_nontradable_4q",
    "bezemer_nfb_1q",
    "werner_fcp_1q",
    "muller_verner_nontradable_1q",
)
MIN_TRAINING_ROWS_SETTINGS: Sequence[int] = (20, 24, 28)
PRIMARY_MIN_TRAINING_ROWS = MIN_TRAINING_ROWS
OOS_BOOTSTRAP_REPS = 2_000

MODEL_SPECS: Mapping[str, tuple[str, tuple[str, ...], str]] = {
    "AR(1)": ("AR(1)", ("AR1",), ""),
    "spread_or_fci": ("Spread / FCI", ("spread_or_fci",), ""),
    "q_t_only": ("Borrower-composition q_t only", (ALLOCATION_FEATURE_COLUMN,), ALLOCATION_FEATURE_COLUMN),
    "complement_share_only": (
        "1-q_t identity check",
        ("one_minus_allocation_share",),
        "one_minus_allocation_share",
    ),
    "matched_credit_plus_q_t": (
        "Matched BOJ stock growth + q_t",
        (BOJ_MATCHED_GROWTH_COLUMN, ALLOCATION_FEATURE_COLUMN),
        ALLOCATION_FEATURE_COLUMN,
    ),
    "matched_credit_plus_complement_identity": (
        "Matched BOJ stock growth + 1-q_t identity check",
        (BOJ_MATCHED_GROWTH_COLUMN, "one_minus_allocation_share"),
        "one_minus_allocation_share",
    ),
}

ALLOCATION_COMPONENTS: Mapping[str, tuple[str, ...]] = {
    "bezemer_nfb_4q": ("C_NFB", "C_FIN", "C_PROP", "C_HH_NONHOUSING"),
    "bezemer_nfb_1q": ("C_NFB", "C_FIN", "C_PROP", "C_HH_NONHOUSING"),
    "werner_fcp_4q": ("C_WERNER_FCP", "C_WERNER_COMPLEMENT"),
    "werner_fcp_1q": ("C_WERNER_FCP", "C_WERNER_COMPLEMENT"),
    "muller_verner_nontradable_4q": (
        "C_MV_NONTRADABLE",
        "C_MV_TRADABLE",
        "C_MV_OTHER_NFB",
        "C_MV_FIN",
        "C_MV_HH",
        "C_MV_UNRESOLVED",
    ),
    "muller_verner_nontradable_1q": (
        "C_MV_NONTRADABLE",
        "C_MV_TRADABLE",
        "C_MV_OTHER_NFB",
        "C_MV_FIN",
        "C_MV_HH",
        "C_MV_UNRESOLVED",
    ),
}

ALLOCATION_LABELS: Mapping[str, str] = {
    "bezemer_nfb_4q": "Bezemer non-financial-business share, four-quarter",
    "bezemer_nfb_1q": "Bezemer non-financial-business share, one-quarter",
    "werner_fcp_4q": (
        "Werner-inspired BOJ borrower-sector proxy share, four-quarter"
    ),
    "werner_fcp_1q": (
        "Werner-inspired BOJ borrower-sector proxy share, one-quarter"
    ),
    "muller_verner_nontradable_4q": (
        "Muller-Verner non-tradable share, four-quarter"
    ),
    "muller_verner_nontradable_1q": (
        "Muller-Verner non-tradable share, one-quarter"
    ),
}

ALLOCATION_DEFINITIONS: Mapping[str, str] = {
    "bezemer_nfb_4q": "sum_4Q(NFB)/sum_4Q(NFB+FIN+PROP+HH_NONHOUSING)",
    "bezemer_nfb_1q": "NFB/(NFB+FIN+PROP+HH_NONHOUSING)",
    "werner_fcp_4q": "sum_4Q(FCP)/sum_4Q(FCP+COMPLEMENT)",
    "werner_fcp_1q": "FCP/(FCP+COMPLEMENT)",
    "muller_verner_nontradable_4q": (
        "sum_4Q(NONTRADABLE)/"
        "sum_4Q(NONTRADABLE+TRADABLE+OTHER_NFB+FIN+HH+UNRESOLVED)"
    ),
    "muller_verner_nontradable_1q": (
        "NONTRADABLE/(NONTRADABLE+TRADABLE+OTHER_NFB+FIN+HH+UNRESOLVED)"
    ),
}

ALLOCATION_ROLLING_WINDOWS: Mapping[str, int] = {
    "bezemer_nfb_4q": 4,
    "werner_fcp_4q": 4,
    "muller_verner_nontradable_4q": 4,
}

ALLOCATION_TAXONOMY_IDS: Mapping[str, str] = {
    "bezemer_nfb_4q": PRIMARY_TAXONOMY_ID,
    "bezemer_nfb_1q": PRIMARY_TAXONOMY_ID,
    "werner_fcp_4q": WERNER_TAXONOMY_ID,
    "werner_fcp_1q": WERNER_TAXONOMY_ID,
    "muller_verner_nontradable_4q": MULLER_VERNER_TAXONOMY_ID,
    "muller_verner_nontradable_1q": MULLER_VERNER_TAXONOMY_ID,
}

ALLOCATION_COORDINATES: Mapping[str, str] = {
    "bezemer_nfb_4q": "NFB",
    "bezemer_nfb_1q": "NFB",
    "werner_fcp_4q": "FCP",
    "werner_fcp_1q": "FCP",
    "muller_verner_nontradable_4q": "NONTRADABLE",
    "muller_verner_nontradable_1q": "NONTRADABLE",
}

FOUR_QUARTER_TAXONOMY_MEASURES: Sequence[str] = (
    "bezemer_nfb_4q",
    "werner_fcp_4q",
    "muller_verner_nontradable_4q",
)


def _latex_escape(value: Any) -> str:
    text = str(value)
    replacements = {
        "\\": r"\textbackslash{}",
        "&": r"\&",
        "%": r"\%",
        "$": r"\$",
        "#": r"\#",
        "_": r"\_",
        "{": r"\{",
        "}": r"\}",
        "~": r"\textasciitilde{}",
        "^": r"\textasciicircum{}",
    }
    return "".join(replacements.get(ch, ch) for ch in text)


def _format_float(value: Any, digits: int = 3) -> str:
    try:
        numeric = float(value)
    except Exception:
        return "n/a"
    if not np.isfinite(numeric):
        return "n/a"
    return f"{numeric:.{digits}f}"


def _target_label(key: str) -> str:
    return {
        "asset_acceleration": "BOJ balance-sheet acceleration",
        "spread_widening": "long-term JGB yield change",
        "downside_growth": "activity-proxy lower tail",
    }.get(key, key)


def _strict_row_sum(frame: pd.DataFrame, columns: Sequence[str]) -> pd.Series:
    numeric = frame.loc[:, list(columns)].apply(pd.to_numeric, errors="coerce")
    return numeric.sum(axis=1, min_count=len(columns))


def _allocation_share(
    frame: pd.DataFrame,
    columns: Sequence[str],
    *,
    rolling_window: int = 1,
) -> pd.Series:
    if len(columns) < 2:
        raise ValueError("An allocation measure must provide at least two components.")
    components = frame.loc[:, list(columns)].apply(pd.to_numeric, errors="coerce")
    if int(rolling_window) > 1:
        components = components.rolling(
            window=int(rolling_window),
            min_periods=int(rolling_window),
        ).sum()
    denominator = components.sum(axis=1, min_count=len(columns))
    numerator = components.iloc[:, 0]
    return (numerator / denominator.where(denominator > 0.0)).astype(float)


def _build_boj_universe_asof(
    origins: pd.Series,
    direct: pd.DataFrame,
    *,
    allocation_measures: Sequence[str] = ALLOCATION_MEASURES,
    primary_allocation_measure: str = PRIMARY_ALLOCATION_MEASURE,
    release_lag_days: int = BOJ_RELEASE_LAG_DAYS,
) -> pd.DataFrame:
    """Build matched BOJ scale/allocation inputs available at each origin.

    The common-taxonomy start level is retained for stock growth, but its
    cross-break flow/allocation observation is invalidated. Release lags are
    applied here from raw BOJ dates; callers must not pass an already lagged
    direct panel.
    """
    if release_lag_days < 0:
        raise ValueError("release_lag_days must be non-negative.")

    requested = tuple(dict.fromkeys(str(value).strip() for value in allocation_measures if str(value).strip()))
    if not requested:
        raise ValueError("At least one allocation measure is required.")
    if primary_allocation_measure not in requested:
        raise ValueError("primary_allocation_measure must be included in allocation_measures.")

    required = {"date", BOJ_DIRECT_STOCK_COLUMN}
    missing = sorted(required.difference(direct.columns))
    if missing:
        raise ValueError(f"BOJ direct panel is missing matched-universe columns: {', '.join(missing)}")

    source = direct.copy().assign(
        date=pd.to_datetime(direct["date"], errors="coerce")
    )
    source = (
        source.dropna(subset=["date"])
        .sort_values("date")
        .drop_duplicates("date", keep="last")
    )
    source = source[source["date"] >= BOJ_COMMON_TAXONOMY_START].reset_index(drop=True)
    if source.empty:
        raise ValueError("BOJ direct panel has no observations in the common-taxonomy window.")

    source.loc[:, BOJ_MATCHED_STOCK_COLUMN] = pd.to_numeric(
        source[BOJ_DIRECT_STOCK_COLUMN],
        errors="coerce",
    )

    available_measures: list[str] = []
    coverage_metadata: dict[str, Any] = {}
    for measure in requested:
        columns = ALLOCATION_COMPONENTS.get(measure)
        if columns is None:
            raise ValueError(
                f"Unknown allocation measure '{measure}'. "
                f"Choose from {', '.join(sorted(ALLOCATION_COMPONENTS))}."
            )
        if not set(columns).issubset(source.columns):
            if measure == primary_allocation_measure:
                missing_columns = ", ".join(sorted(set(columns).difference(source.columns)))
                raise ValueError(
                    f"Primary allocation measure '{measure}' is unavailable; "
                    f"missing {missing_columns}."
                )
            continue
        output_column = f"{ALLOCATION_FEATURE_COLUMN}__{measure}"
        legacy_output_column = f"{LEGACY_ALLOCATION_FEATURE_COLUMN}__{measure}"
        allocation_source = source.copy()
        allocation_source.loc[
            allocation_source["date"].eq(BOJ_COMMON_TAXONOMY_START),
            list(columns),
        ] = np.nan
        source.loc[:, output_column] = _allocation_share(
            allocation_source,
            columns,
            rolling_window=ALLOCATION_ROLLING_WINDOWS.get(measure, 1),
        )
        # 2009Q2 components are cross-break changes even when the raw panel
        # contains a numerical value. They are not a valid allocation input.
        source.loc[source["date"].eq(BOJ_COMMON_TAXONOMY_START), output_column] = np.nan
        # Preserve the former external column name as an explicit legacy alias.
        source.loc[:, legacy_output_column] = source[output_column]
        coverage_window = source["date"] > BOJ_COMMON_TAXONOMY_START
        coverage_values = pd.to_numeric(source.loc[coverage_window, output_column], errors="coerce")
        missing_source_dates = source.loc[
            coverage_window & source[output_column].isna(),
            "date",
        ]
        missing_source_quarters = (
            pd.to_datetime(missing_source_dates, errors="coerce")
            .dt.quarter.value_counts()
            .sort_index()
        )
        coverage_metadata.update(
            {
                f"raw_allocation_available__{measure}": int(coverage_values.notna().sum()),
                f"raw_allocation_total__{measure}": int(coverage_values.size),
                f"raw_allocation_missing_source_quarters__{measure}": ",".join(
                    f"Q{int(quarter)}:{int(count)}"
                    for quarter, count in missing_source_quarters.items()
                ),
            }
        )
        available_measures.append(measure)

    if primary_allocation_measure not in available_measures:
        raise ValueError(f"Primary allocation measure '{primary_allocation_measure}' is unavailable.")

    value_columns = [BOJ_MATCHED_STOCK_COLUMN]
    value_columns.extend(
        f"{ALLOCATION_FEATURE_COLUMN}__{measure}" for measure in available_measures
    )
    value_columns.extend(
        f"{LEGACY_ALLOCATION_FEATURE_COLUMN}__{measure}"
        for measure in available_measures
    )
    source = source.loc[:, ["date", *value_columns]].copy()
    source.loc[:, "boj_source_date"] = source["date"]
    source.loc[:, "boj_available_date"] = source["date"] + pd.to_timedelta(
        int(release_lag_days),
        unit="D",
    )

    origin_frame = pd.DataFrame({"date": pd.to_datetime(origins, errors="coerce")})
    origin_frame = origin_frame.sort_values("date").reset_index()
    asof = pd.merge_asof(
        origin_frame,
        source.drop(columns=["date"]).sort_values("boj_available_date"),
        left_on="date",
        right_on="boj_available_date",
        direction="backward",
    )
    asof = asof.sort_values("index").drop(columns=["index"]).reset_index(drop=True)
    asof.loc[:, "boj_release_lag_days"] = int(release_lag_days)
    asof.loc[:, "boj_common_taxonomy_start"] = BOJ_COMMON_TAXONOMY_START
    asof.loc[:, "available_allocation_measures"] = ",".join(available_measures)
    for key, value in coverage_metadata.items():
        asof.loc[:, key] = value
    return asof


def _load_boj_direct_panel(site_dir: Path, boj_data_path: Path | None) -> tuple[pd.DataFrame, Path]:
    path = Path(boj_data_path) if boj_data_path is not None else site_dir.parent / "data" / "credit_destination_jp.csv"
    if not path.exists():
        raise FileNotFoundError(
            "Matched-universe OOS requires the raw BOJ bridge panel at "
            f"{path}; the BIS L_real series is not used as a fallback."
        )
    return pd.read_csv(path), path


def _attach_boj_universe(
    frame: pd.DataFrame,
    *,
    direct: pd.DataFrame,
    allocation_measures: Sequence[str],
    primary_allocation_measure: str,
    release_lag_days: int,
) -> pd.DataFrame:
    if "date" not in frame.columns:
        raise ValueError("JP forecast panel must contain a date column.")
    out = frame.copy().assign(
        date=pd.to_datetime(frame["date"], errors="coerce")
    )
    matched = _build_boj_universe_asof(
        out["date"],
        direct,
        allocation_measures=allocation_measures,
        primary_allocation_measure=primary_allocation_measure,
        release_lag_days=release_lag_days,
    )
    for column in matched.columns:
        if column == "date":
            continue
        out.loc[:, column] = matched[column].to_numpy()
    return out


def _trailing_log_growth(level: pd.Series, horizon: int) -> pd.Series:
    numeric = pd.to_numeric(level, errors="coerce")
    logged = np.log(numeric.where(numeric > 0.0))
    return logged - logged.shift(int(horizon))


def _destination_features(
    frame: pd.DataFrame,
    *,
    horizon: int,
    allocation_column: str,
) -> pd.DataFrame:
    features = build_base_features(frame, horizon=horizon).copy()
    features.loc[:, BOJ_MATCHED_GROWTH_COLUMN] = _trailing_log_growth(
        _safe_numeric(frame, BOJ_MATCHED_STOCK_COLUMN),
        horizon,
    )
    features.loc[:, ALLOCATION_FEATURE_COLUMN] = _safe_numeric(frame, allocation_column)
    # External callers of this helper may still reference the earlier name.
    features.loc[:, LEGACY_ALLOCATION_FEATURE_COLUMN] = features[ALLOCATION_FEATURE_COLUMN]
    features.loc[:, "one_minus_allocation_share"] = 1.0 - features[ALLOCATION_FEATURE_COLUMN]
    features.loc[:, "spread_or_fci"] = features["spread_only"].combine_first(features["simple_fci"])
    return features


def _fit_predict_with_effect(
    x_train: pd.DataFrame,
    y_train: pd.Series,
    x_test: pd.Series,
    *,
    effect_feature: str,
) -> tuple[float, float]:
    test_raw = pd.to_numeric(x_test, errors="coerce")
    if test_raw.isna().any():
        return float("nan"), float("nan")
    train = pd.concat([x_train, pd.to_numeric(y_train, errors="coerce").rename("__target__")], axis=1).dropna()
    if train.shape[0] < 8:
        return float("nan"), float("nan")
    y = train["__target__"].to_numpy(dtype=float)
    x = train.drop(columns=["__target__"])
    means = x.mean(axis=0)
    stds = x.std(axis=0, ddof=0).replace(0.0, np.nan)
    x_std = ((x - means) / stds).replace([np.inf, -np.inf], np.nan).fillna(0.0)
    test = ((test_raw - means) / stds).replace([np.inf, -np.inf], np.nan).fillna(0.0)
    design = np.column_stack([np.ones(len(x_std)), x_std.to_numpy(dtype=float)])
    penalty = np.eye(design.shape[1]) * 1e-6
    penalty[0, 0] = 0.0
    beta = np.linalg.pinv(design.T @ design + penalty) @ design.T @ y
    prediction = float(np.concatenate([[1.0], test.to_numpy(dtype=float)]) @ beta)

    effect = float("nan")
    if effect_feature and effect_feature in x.columns:
        position = list(x.columns).index(effect_feature) + 1
        effect = float(beta[position])
    return prediction, effect


def _score_destination_model(
    frame: pd.DataFrame,
    target: ForecastTarget,
    baseline_x: pd.DataFrame,
    model_x: pd.DataFrame,
    *,
    effect_feature: str,
    horizon: int,
    min_training_rows: int = MIN_TRAINING_ROWS,
) -> dict[str, Any]:
    y = pd.to_numeric(target.outcome, errors="coerce")
    actual_z: list[float] = []
    model_z: list[float] = []
    base_z: list[float] = []
    loss_diff_values: list[float] = []
    events: list[int] = []
    model_probs: list[float] = []
    base_probs: list[float] = []
    event_scores: list[float] = []
    effects: list[float] = []
    forecast_dates: list[str] = []
    common_training_cases: list[int] = []

    first_origin = min_training_rows + horizon - 1
    for idx in range(first_origin, len(frame)):
        actual = float(y.iloc[idx]) if idx < len(y) and np.isfinite(y.iloc[idx]) else float("nan")
        if not np.isfinite(actual):
            continue
        training_stop = idx - horizon + 1
        train_y = y.iloc[:training_stop]
        common_complete = pd.concat(
            [
                model_x.iloc[:training_stop].add_prefix("__model__"),
                baseline_x.iloc[:training_stop].add_prefix("__baseline__"),
                train_y.rename("__target__"),
            ],
            axis=1,
        ).dropna()
        if common_complete.shape[0] < min_training_rows:
            continue
        common_index = common_complete.index
        common_y = train_y.loc[common_index]
        model_pred, effect = _fit_predict_with_effect(
            model_x.loc[common_index],
            common_y,
            model_x.iloc[idx],
            effect_feature=effect_feature,
        )
        base_pred, _base_effect = _fit_predict_with_effect(
            baseline_x.loc[common_index],
            common_y,
            baseline_x.iloc[idx],
            effect_feature="",
        )
        if not (np.isfinite(model_pred) and np.isfinite(base_pred)):
            continue
        common_training_cases.append(int(common_complete.shape[0]))
        y_mean = float(common_y.mean())
        y_std = float(common_y.std(ddof=0))
        if not np.isfinite(y_std) or y_std <= 1e-12:
            continue
        actual_std = (actual - y_mean) / y_std
        model_std = (model_pred - y_mean) / y_std
        base_std = (base_pred - y_mean) / y_std
        actual_z.append(float(actual_std))
        model_z.append(float(model_std))
        base_z.append(float(base_std))

        if np.isfinite(effect):
            direction = -1.0 if effect_feature == ALLOCATION_FEATURE_COLUMN else 1.0
            effects.append(float(direction * effect))

        if target.target_type == "binary":
            threshold = _threshold_from_train(target, common_y)
            if not np.isfinite(threshold):
                continue
            threshold_z = (threshold - y_mean) / y_std
            if target.event_mode == "lower_quantile":
                event = int(actual_std < threshold_z)
                model_score = threshold_z - model_std
                base_score = threshold_z - base_std
            else:
                event = int(actual_std > threshold_z)
                model_score = model_std - threshold_z
                base_score = base_std - threshold_z
            model_prob = float(1.0 / (1.0 + np.exp(-np.clip(model_score, -30, 30))))
            base_prob = float(1.0 / (1.0 + np.exp(-np.clip(base_score, -30, 30))))
            events.append(event)
            model_probs.append(model_prob)
            base_probs.append(base_prob)
            event_scores.append(float(model_score))
            loss_diff_values.append((model_prob - event) ** 2 - (base_prob - event) ** 2)
        else:
            loss_diff_values.append((model_std - actual_std) ** 2 - (base_std - actual_std) ** 2)
        forecast_dates.append(str(pd.Timestamp(frame["date"].iloc[idx]).date()))

    n = len(loss_diff_values)
    if n < 8:
        return {
            "n": float(n),
            "metric_loss_diff": float("nan"),
            "required_min_training_cases": int(min_training_rows),
            "minimum_common_training_cases": (
                min(common_training_cases) if common_training_cases else float("nan")
            ),
        }

    actual_arr = np.asarray(actual_z, dtype=float)
    model_arr = np.asarray(model_z, dtype=float)
    base_arr = np.asarray(base_z, dtype=float)
    loss_diff = np.asarray(loss_diff_values, dtype=float)
    ci_low, ci_high = _block_bootstrap_ci(
        loss_diff,
        block_length=max(BLOCK_LENGTH, int(horizon)),
        reps=OOS_BOOTSTRAP_REPS,
    )
    effect_mean = float(np.mean(effects)) if effects else float("nan")
    date_fields = {
        "first_forecast_origin": forecast_dates[0] if forecast_dates else "",
        "last_forecast_origin": forecast_dates[-1] if forecast_dates else "",
        "required_min_training_cases": int(min_training_rows),
        "minimum_common_training_cases": min(common_training_cases),
    }

    if target.target_type == "binary" and len(events) >= 8:
        event_arr = np.asarray(events, dtype=float)
        model_prob_arr = np.clip(np.asarray(model_probs, dtype=float), 1e-6, 1.0 - 1e-6)
        base_prob_arr = np.clip(np.asarray(base_probs, dtype=float), 1e-6, 1.0 - 1e-6)
        return {
            "n": float(len(events)),
            "metric": "brier",
            "model_metric": float(np.mean((model_prob_arr - event_arr) ** 2)),
            "baseline_metric": float(np.mean((base_prob_arr - event_arr) ** 2)),
            "metric_loss_diff": float(np.mean(loss_diff)),
            "loss_differential_estimand": "candidate_brier_loss_minus_baseline_brier_loss",
            "block_ci_low": ci_low,
            "block_ci_high": ci_high,
            "effect_1sd": effect_mean,
            "event_count": int(event_arr.sum()),
            **date_fields,
        }

    errors = model_arr - actual_arr
    base_errors = base_arr - actual_arr
    return {
        "n": float(n),
        "metric": "rmse",
        "model_metric": float(np.sqrt(np.mean(errors**2))),
        "baseline_metric": float(np.sqrt(np.mean(base_errors**2))),
        "metric_loss_diff": float(np.mean(loss_diff)),
        "loss_differential_estimand": (
            "mean_standardized_squared_error_candidate_minus_baseline"
        ),
        "block_ci_low": ci_low,
        "block_ci_high": ci_high,
        "effect_1sd": effect_mean,
        **date_fields,
    }


def _status(row: Mapping[str, Any]) -> str:
    diff = float(row.get("metric_loss_diff", float("nan")))
    ci_hi = float(row.get("block_ci_high", float("nan")))
    if not np.isfinite(diff):
        return "insufficient coverage"
    if np.isfinite(ci_hi) and ci_hi < 0:
        return "interval below zero"
    if diff < 0:
        return "lower point loss; interval includes zero"
    return "no lower point loss"


def _focused_rows_for_region(
    region: RegionFrame,
    *,
    horizon: int,
    allocation_measure: str,
    primary_allocation_measure: str,
    min_training_rows: int,
    primary_min_training_rows: int,
    boj_data_source: str,
    release_lag_days: int,
) -> list[dict[str, Any]]:
    frame = region.frame.copy().sort_values("date").reset_index(drop=True)
    frame = frame[pd.to_datetime(frame["date"], errors="coerce") >= ANALYSIS_START].reset_index(drop=True)
    targets, _coverage = build_forecast_targets(frame, horizon=horizon)
    target_by_key = {target.key: target for target in targets}
    allocation_column = f"{ALLOCATION_FEATURE_COLUMN}__{allocation_measure}"
    if allocation_column not in frame.columns:
        return []
    allocation_values = pd.to_numeric(frame[allocation_column], errors="coerce")
    missing_origin_quarters = (
        pd.to_datetime(frame.loc[allocation_values.isna(), "date"], errors="coerce")
        .dt.quarter.value_counts()
        .sort_index()
    )
    missing_quarter_text = ",".join(
        f"Q{int(quarter)}:{int(count)}"
        for quarter, count in missing_origin_quarters.items()
    )
    raw_available_column = f"raw_allocation_available__{allocation_measure}"
    raw_total_column = f"raw_allocation_total__{allocation_measure}"
    raw_missing_column = (
        f"raw_allocation_missing_source_quarters__{allocation_measure}"
    )
    raw_available = int(
        pd.to_numeric(frame.get(raw_available_column), errors="coerce").dropna().iloc[0]
    )
    raw_total = int(
        pd.to_numeric(frame.get(raw_total_column), errors="coerce").dropna().iloc[0]
    )
    raw_missing = str(frame.get(raw_missing_column).dropna().iloc[0])
    features = _destination_features(
        frame,
        horizon=horizon,
        allocation_column=allocation_column,
    )
    rows: list[dict[str, Any]] = []
    for target_key in TARGET_KEYS:
        target = target_by_key.get(target_key)
        if target is None:
            continue
        if target_key == "spread_widening":
            target = replace(target, target_type="continuous", event_mode="", event_quantile=float("nan"))
        local = features.copy()
        local["AR1"] = target.ar1_feature
        baseline_x = _feature_matrix(local, [BOJ_MATCHED_GROWTH_COLUMN])
        for model_name, (model_label, model_cols, effect_feature) in MODEL_SPECS.items():
            model_x = _feature_matrix(local, model_cols)
            metrics = _score_destination_model(
                frame,
                target,
                baseline_x,
                model_x,
                effect_feature=effect_feature,
                horizon=horizon,
                min_training_rows=int(min_training_rows),
            )
            row = {
                "region_key": region.key,
                "region_label": region.label,
                "panel_source": region.source_path,
                "horizon_quarters": int(horizon),
                "target": target.key,
                "target_label": _target_label(target.key),
                "target_type": target.target_type,
                "target_source": target.source_column,
                "baseline": BOJ_MATCHED_GROWTH_COLUMN,
                "baseline_label": "matched BOJ primary-population stock growth",
                "baseline_universe": BOJ_BASELINE_UNIVERSE,
                "model": model_name,
                "model_label": model_label,
                "model_features": ",".join(model_cols),
                "effect_feature": effect_feature,
                "allocation_measure": allocation_measure,
                "taxonomy_id": ALLOCATION_TAXONOMY_IDS.get(
                    allocation_measure,
                    "",
                ),
                "taxonomy_coordinate": ALLOCATION_COORDINATES.get(
                    allocation_measure,
                    "",
                ),
                "allocation_measure_label": ALLOCATION_LABELS.get(
                    allocation_measure,
                    allocation_measure,
                ),
                "allocation_definition": ALLOCATION_DEFINITIONS.get(
                    allocation_measure,
                    allocation_measure,
                ),
                "is_primary_allocation_measure": (
                    allocation_measure == primary_allocation_measure
                ),
                "min_training_rows_setting": int(min_training_rows),
                "is_primary_training_window": (
                    int(min_training_rows) == int(primary_min_training_rows)
                ),
                "allocation_available_origins": int(allocation_values.notna().sum()),
                "allocation_total_origins": int(len(allocation_values)),
                "allocation_availability_rate": float(allocation_values.notna().mean()),
                "allocation_missing_origin_quarters": missing_quarter_text,
                "raw_allocation_available_quarters": raw_available,
                "raw_allocation_total_quarters": raw_total,
                "raw_allocation_availability_rate": (
                    float(raw_available / raw_total) if raw_total else float("nan")
                ),
                "raw_allocation_missing_source_quarters": raw_missing,
                "boj_data_source": boj_data_source,
                "boj_common_taxonomy_start": str(BOJ_COMMON_TAXONOMY_START.date()),
                "release_lag_days": int(release_lag_days),
                **metrics,
            }
            row["status"] = _status(row)
            rows.append(row)
    return rows


def run_destination_oos(
    site_dir: Path,
    *,
    source_ref: str | None = None,
    panel_mode: str = "realtime",
    horizons: Sequence[int] = HORIZONS,
    boj_data_path: Path | None = None,
    allocation_measures: Sequence[str] = ALLOCATION_MEASURES,
    primary_allocation_measure: str = PRIMARY_ALLOCATION_MEASURE,
    release_lag_days: int = BOJ_RELEASE_LAG_DAYS,
    min_training_rows_settings: Sequence[int] = MIN_TRAINING_ROWS_SETTINGS,
    primary_min_training_rows: int = PRIMARY_MIN_TRAINING_ROWS,
) -> pd.DataFrame:
    training_settings = tuple(
        dict.fromkeys(int(value) for value in min_training_rows_settings)
    )
    if not training_settings or any(value < 8 for value in training_settings):
        raise ValueError("min_training_rows_settings must contain values of at least 8.")
    if int(primary_min_training_rows) not in training_settings:
        raise ValueError(
            "primary_min_training_rows must be included in min_training_rows_settings."
        )
    regions = list(load_region_frames(site_dir, source_ref=source_ref, mode=panel_mode))
    jp_regions = [region for region in regions if region.key == "jp"]
    if not jp_regions:
        return pd.DataFrame()
    direct, resolved_boj_path = _load_boj_direct_panel(site_dir, boj_data_path)
    attached = _attach_boj_universe(
        jp_regions[0].frame,
        direct=direct,
        allocation_measures=allocation_measures,
        primary_allocation_measure=primary_allocation_measure,
        release_lag_days=release_lag_days,
    )
    region = RegionFrame(
        key=jp_regions[0].key,
        label=jp_regions[0].label,
        frame=attached,
        source_path=jp_regions[0].source_path,
        panel_mode=jp_regions[0].panel_mode,
    )
    try:
        boj_data_source = resolved_boj_path.resolve().relative_to(
            site_dir.parent.resolve()
        ).as_posix()
    except ValueError:
        boj_data_source = resolved_boj_path.name
    available_series = attached.get(
        "available_allocation_measures",
        pd.Series(dtype=object),
    ).dropna()
    available_raw = str(available_series.iloc[-1]) if not available_series.empty else ""
    available = set(filter(None, available_raw.split(",")))
    rows: list[dict[str, Any]] = []
    for horizon in horizons:
        for measure in allocation_measures:
            if measure not in available:
                continue
            for min_training_rows in training_settings:
                rows.extend(
                    _focused_rows_for_region(
                        region,
                        horizon=int(horizon),
                        allocation_measure=str(measure),
                        primary_allocation_measure=primary_allocation_measure,
                        min_training_rows=int(min_training_rows),
                        primary_min_training_rows=int(primary_min_training_rows),
                        boj_data_source=boj_data_source,
                        release_lag_days=release_lag_days,
                    )
                )
    return pd.DataFrame(rows)


def _summary_rows(
    results: pd.DataFrame,
    *,
    target_keys: set[str] | None = None,
    model_keys: set[str] | None = None,
) -> pd.DataFrame:
    if results.empty:
        return results
    keep_models = {
        "matched_credit_plus_q_t",
        "matched_credit_plus_complement_identity",
    }
    keep = results[results["model"].isin(keep_models)].copy()
    if "is_primary_allocation_measure" in keep.columns:
        primary = keep["is_primary_allocation_measure"]
        if primary.dtype != bool:
            primary = primary.astype(str).str.lower().isin({"1", "true", "yes"})
        keep = keep[primary].copy()
    if "is_primary_training_window" in keep.columns:
        primary_training = keep["is_primary_training_window"]
        if primary_training.dtype != bool:
            primary_training = primary_training.astype(str).str.lower().isin(
                {"1", "true", "yes"}
            )
        keep = keep[primary_training].copy()
    if model_keys is not None:
        keep = keep[keep["model"].isin(model_keys)].copy()
    if target_keys is not None:
        keep = keep[keep["target"].isin(target_keys)].copy()
    order = {
        "asset_acceleration": 0,
        "spread_widening": 1,
        "downside_growth": 2,
        "matched_credit_plus_q_t": 0,
        "matched_credit_plus_complement_identity": 1,
    }
    keep.loc[:, "target_order"] = keep["target"].map(order).fillna(99)
    keep.loc[:, "model_order"] = keep["model"].map(order).fillna(99)
    return keep.sort_values(
        ["horizon_quarters", "target_order", "model_order"]
    ).drop(columns=["target_order", "model_order"])


def _model_label_tex(row: Mapping[str, Any]) -> str:
    model = str(row.get("model", ""))
    if model == "matched_credit_plus_q_t":
        if str(row.get("allocation_measure", "")) == PRIMARY_ALLOCATION_MEASURE:
            return r"Mapped-stock growth + $q_t$"
        return r"Mapped-stock growth + coordinate"
    if model == "matched_credit_plus_complement_identity":
        return r"Mapped-stock growth + $(1-q_t)$ (identity check)"
    return _latex_escape(row.get("model_label", ""))


def _allocation_label_tex(row: Mapping[str, Any]) -> str:
    measure = str(row.get("allocation_measure", ""))
    if measure == "bezemer_nfb_4q":
        return r"Bezemer: NFB (primary)"
    if measure == "werner_fcp_4q":
        return r"Werner-inspired BOJ proxy"
    if measure == "muller_verner_nontradable_4q":
        return r"M\"uller--Verner: non-tradable (BOJ adaptation)"
    if measure == "bezemer_nfb_1q":
        return r"Bezemer: NFB, 1Q"
    if measure == "werner_fcp_1q":
        return r"Werner-inspired BOJ proxy, 1Q"
    if measure == "muller_verner_nontradable_1q":
        return r"M\"uller--Verner: non-tradable, 1Q"
    return _latex_escape(row.get("allocation_measure_label", measure))


def _truthy(series: pd.Series) -> pd.Series:
    if series.dtype == bool:
        return series
    return series.astype(str).str.lower().isin({"1", "true", "yes"})


def _reportable_rows(results: pd.DataFrame) -> pd.DataFrame:
    if results.empty:
        return results
    required_numeric = (
        "n",
        "model_metric",
        "baseline_metric",
        "metric_loss_diff",
        "block_ci_low",
        "block_ci_high",
    )
    keep = pd.Series(True, index=results.index)
    for column in required_numeric:
        if column not in results.columns:
            return results.iloc[0:0].copy()
        values = pd.to_numeric(results[column], errors="coerce")
        keep &= values.notna() & np.isfinite(values)
    keep &= pd.to_numeric(results["n"], errors="coerce").ge(8)
    return results[keep].copy()


def _main_table_rows(results: pd.DataFrame) -> pd.DataFrame:
    if results.empty:
        return results
    keep = results[
        results["target"].eq("spread_widening")
        & results["model"].eq("matched_credit_plus_q_t")
        & _truthy(results["is_primary_training_window"])
        & results["allocation_measure"].isin(FOUR_QUARTER_TAXONOMY_MEASURES)
    ].copy()
    keep = _reportable_rows(keep)
    measure_order = {
        "bezemer_nfb_4q": 0,
        "werner_fcp_4q": 1,
        "muller_verner_nontradable_4q": 2,
    }
    keep.loc[:, "__measure_order"] = keep["allocation_measure"].map(
        measure_order
    ).fillna(99)
    return keep.sort_values(
        ["horizon_quarters", "__measure_order"]
    ).drop(columns=["__measure_order"])


def _auxiliary_table_rows(results: pd.DataFrame) -> pd.DataFrame:
    if results.empty:
        return results
    keep = results[
        results["target"].eq("asset_acceleration")
        & results["model"].eq("matched_credit_plus_q_t")
        & _truthy(results["is_primary_training_window"])
        & results["allocation_measure"].isin(FOUR_QUARTER_TAXONOMY_MEASURES)
    ].copy()
    keep = _reportable_rows(keep)
    measure_order = {
        "bezemer_nfb_4q": 0,
        "werner_fcp_4q": 1,
        "muller_verner_nontradable_4q": 2,
    }
    keep.loc[:, "__measure_order"] = keep["allocation_measure"].map(
        measure_order
    ).fillna(99)
    return keep.sort_values(
        ["horizon_quarters", "__measure_order"]
    ).drop(columns=["__measure_order"])


def _training_sensitivity_note(
    results: pd.DataFrame,
    *,
    target_key: str,
) -> str:
    if results.empty:
        return ""
    local = results[
        results["target"].eq(target_key)
        & results["model"].eq("matched_credit_plus_q_t")
        & results["allocation_measure"].eq(PRIMARY_ALLOCATION_MEASURE)
        & results["horizon_quarters"].eq(4)
        & results["min_training_rows_setting"].isin({20, 24})
    ].copy()
    values: dict[int, float] = {}
    for _, row in local.iterrows():
        setting = int(row["min_training_rows_setting"])
        value = float(row.get("metric_loss_diff", float("nan")))
        if np.isfinite(value):
            values[setting] = value
    if set(values) != {20, 24}:
        return ""
    selected = results[
        results["target"].eq(target_key)
        & results["model"].eq("matched_credit_plus_q_t")
        & results["allocation_measure"].eq(PRIMARY_ALLOCATION_MEASURE)
        & results["horizon_quarters"].eq(4)
        & results["min_training_rows_setting"].eq(PRIMARY_MIN_TRAINING_ROWS)
    ]
    selected_value = (
        float(selected.iloc[0]["metric_loss_diff"])
        if not selected.empty
        else float("nan")
    )
    values_with_selected = [values[20], values[24], selected_value]
    if all(np.isfinite(value) and value > 0.0 for value in values_with_selected):
        interpretation = (
            "all three signs are above zero, so none of these window choices "
            "shows a point-loss improvement"
        )
    elif all(np.isfinite(value) and value < 0.0 for value in values_with_selected):
        interpretation = (
            "all three signs are below zero, although this remains a selected, "
            "descriptive sensitivity"
        )
    else:
        interpretation = "the sign changes across training-window choices"
    return (
        "For the primary Bezemer four-quarter row, the 4Q mean loss differential is "
        f"{values[20]:.3f} with a 20-case minimum and {values[24]:.3f} with a "
        f"24-case minimum; {interpretation}."
    )


def _effect_tex(row: Mapping[str, Any]) -> str:
    try:
        value = float(row.get("effect_1sd", float("nan")))
    except Exception:
        value = float("nan")
    if not np.isfinite(value):
        return "n/a"
    feature = row.get("effect_feature", "")
    if feature == ALLOCATION_FEATURE_COLUMN:
        return rf"1 s.d. fall $q_t$: {_format_float(value)}"
    if feature == "one_minus_allocation_share":
        return rf"1 s.d. rise $(1-q_t)$: {_format_float(value)}"
    return "n/a"


def _render_destination_oos_table(
    summary: pd.DataFrame,
    *,
    caption: str,
    label: str,
    scope_note: str,
    sensitivity_note: str = "",
) -> str:
    lines = [
        r"\begin{table}[htbp]",
        r"  \centering",
        r"  \small",
        r"  \setlength{\tabcolsep}{3pt}",
        rf"  \caption{{{caption}}}",
        rf"  \label{{{label}}}",
        r"  \resizebox{\textwidth}{!}{%",
        r"  \begin{tabular}{@{}llllllll@{}}",
        r"    \toprule",
        r"    $h$ & Outcome & Literature-anchored coordinate & Model & $N$ & Coordinate-aug. RMSE & Matched-stock RMSE & Mean $\Delta$ squared loss [95\% CI] \\",
        r"    \midrule",
    ]
    if summary.empty:
        lines.append(r"    No JP borrower-composition OOS results were generated. \\")
    else:
        for _, row in summary.iterrows():
            loss_diff_text = (
                f"{_format_float(row['metric_loss_diff'])} "
                f"[{_format_float(row['block_ci_low'])}, {_format_float(row['block_ci_high'])}]"
            )
            lines.append(
                "    "
                + f"{int(row['horizon_quarters'])}Q"
                + " & "
                + _latex_escape(row["target_label"])
                + " & "
                + _allocation_label_tex(row)
                + " & "
                + _model_label_tex(row)
                + " & "
                + _format_float(row["n"], 0)
                + " & "
                + _format_float(row["model_metric"])
                + " & "
                + _format_float(row["baseline_metric"])
                + " & "
                + _latex_escape(loss_diff_text)
                + r" \\"
            )
    first_origin_values = (
        summary["first_forecast_origin"]
        if "first_forecast_origin" in summary.columns
        else pd.Series(dtype=object)
    )
    last_origin_values = (
        summary["last_forecast_origin"]
        if "last_forecast_origin" in summary.columns
        else pd.Series(dtype=object)
    )
    first_origins = pd.to_datetime(first_origin_values, errors="coerce").dropna()
    last_origins = pd.to_datetime(last_origin_values, errors="coerce").dropna()
    origin_note = ""
    if not first_origins.empty and not last_origins.empty:
        origin_note = (
            f" Reported forecast origins range from {first_origins.min().date()} to "
            f"{last_origins.max().date()}."
        )
    coverage_bits: list[str] = []
    if not summary.empty and "allocation_measure" in summary.columns:
        for measure in FOUR_QUARTER_TAXONOMY_MEASURES:
            rows = summary[summary["allocation_measure"].eq(measure)]
            if rows.empty:
                continue
            row = rows.iloc[0]
            available = int(row.get("raw_allocation_available_quarters", 0))
            total = int(row.get("raw_allocation_total_quarters", 0))
            missing_pattern = str(
                row.get("raw_allocation_missing_source_quarters", "")
            )
            label_text = ALLOCATION_LABELS.get(measure, measure)
            bit = (
                f"The {label_text} coordinate is available in "
                f"{available}/{total} post-break source quarters"
            )
            if missing_pattern:
                bit += f" (missing-source pattern {missing_pattern})"
            coverage_bits.append(bit + ".")
    coverage_note = " ".join(coverage_bits)
    lines.extend(
        [
            r"    \bottomrule",
            r"  \end{tabular}",
            r"  }",
            r"  \par\smallskip\raggedright\footnotesize "
            + _latex_escape(
                "Rows use the current-vintage Japan pseudo-OOS panel with an assumed "
                "fixed 90-day lag for BOJ inputs. "
                "Scale is log growth of the exact Bezemer Japan-crosswalk population "
                "(BOJ total less local-government loans) in primary_included_stock; "
                "the lender population is domestically licensed banks, but the residual "
                "NFB bucket includes the disclosed overseas-linked series. Each displayed four-quarter "
                "coordinate uses the same population. The three literature-anchored "
                "mappings are reported jointly rather than choosing among their displayed "
                "loss results: Bezemer non-financial business, a Werner-inspired BOJ "
                "borrower-sector proxy, or a BOJ-sector adaptation of Muller-Verner "
                "non-tradables. Positive parts are "
                "taken after aggregation within each reported taxonomy bucket. "
                "The 2009Q2 cross-taxonomy flow is invalid; dated BOJ inputs are shifted "
                "by that assumed fixed lag. Targets report standardized "
                "RMSE. Each h-quarter regression excludes training labels whose outcomes "
                "are not realized by the forecast origin and requires at least "
                f"{MIN_TRAINING_ROWS} complete "
                "training cases in both models. The moving-block length equals the horizon, "
                "with 2,000 fixed-seed replications. The mean loss differential is candidate "
                "standardized squared loss minus matched-stock baseline loss; it is not an "
                "RMSE difference. Its block-bootstrap interval is descriptive and conditional "
                "on the displayed construction and training-window choice, not confirmatory "
                "post-selection inference. Negative values favor the candidate."
                + origin_note
            ),
            r"  \par\smallskip\raggedright\footnotesize "
            + _latex_escape(coverage_note),
            r"  \par\smallskip\raggedright\footnotesize "
            + _latex_escape(
                "Only models augmenting matched-stock growth are shown; the companion CSV "
                "contains standalone predictors, identity checks, simple benchmarks, and "
                "alternative borrower-composition constructions when available. No "
                "p-values or multiplicity-adjusted inference are reported."
            ),
            *(
                [
                    r"  \par\smallskip\raggedright\footnotesize "
                    + _latex_escape(sensitivity_note)
                ]
                if sensitivity_note
                else []
            ),
            r"  \par\smallskip\raggedright\footnotesize "
            + _latex_escape(scope_note),
            r"\end{table}",
        ]
    )
    return "\n".join(lines) + "\n"


def render_destination_oos_tex(results: pd.DataFrame) -> str:
    summary = _main_table_rows(results)
    return _render_destination_oos_table(
        summary,
        caption="Bridge application: JP borrower-composition pseudo-OOS loss comparisons.",
        label="tab:destination_oos_incremental",
        scope_note=(
            "The table is a use case for the BOJ borrower-composition bridge, not a validated forecasting model."
        ),
        sensitivity_note=_training_sensitivity_note(
            results,
            target_key="spread_widening",
        ),
    )


def render_destination_oos_asset_auxiliary_tex(results: pd.DataFrame) -> str:
    summary = _auxiliary_table_rows(results)
    return _render_destination_oos_table(
        summary,
        caption="Auxiliary BOJ balance-sheet acceleration pseudo-OOS check.",
        label="tab:destination_oos_asset_auxiliary",
        scope_note=(
            "This appendix-only specification records the asset-acceleration check "
            "separately from the main borrower-composition application."
        ),
    )


def write_destination_oos_outputs(results: pd.DataFrame, *, root: Path) -> list[Path]:
    site_dir = root / "site"
    data_dir = root / "data"
    tex_dir = root / "tex" / "generated"
    site_dir.mkdir(parents=True, exist_ok=True)
    data_dir.mkdir(parents=True, exist_ok=True)
    tex_dir.mkdir(parents=True, exist_ok=True)

    results_path = site_dir / "destination_oos_incremental.csv"
    summary_path = data_dir / "destination_oos_incremental_summary.json"
    tex_path = tex_dir / "theory_destination_oos_incremental.tex"
    auxiliary_tex_path = tex_dir / "theory_destination_oos_asset_auxiliary.tex"
    results.to_csv(results_path, index=False)
    primary_rows = results.copy()
    if "is_primary_allocation_measure" in primary_rows.columns:
        primary_rows = primary_rows[
            _truthy(primary_rows["is_primary_allocation_measure"])
        ].copy()
    if "is_primary_training_window" in primary_rows.columns:
        primary_rows = primary_rows[
            _truthy(primary_rows["is_primary_training_window"])
        ].copy()
    primary_measures = (
        sorted(str(value) for value in primary_rows["allocation_measure"].dropna().unique())
        if "allocation_measure" in primary_rows.columns
        else []
    )
    payload = {
        "horizons": (
            sorted(int(value) for value in results["horizon_quarters"].dropna().unique())
            if "horizon_quarters" in results.columns
            else list(HORIZONS)
        ),
        "targets": (
            sorted(str(value) for value in results["target"].dropna().unique())
            if "target" in results.columns
            else list(TARGET_KEYS)
        ),
        "baseline": BOJ_MATCHED_GROWTH_COLUMN,
        "baseline_universe": BOJ_BASELINE_UNIVERSE,
        "borrower_composition_feature": ALLOCATION_FEATURE_COLUMN,
        "legacy_feature_aliases": {
            LEGACY_ALLOCATION_FEATURE_COLUMN: ALLOCATION_FEATURE_COLUMN,
        },
        "primary_taxonomy_id": PRIMARY_TAXONOMY_ID,
        "taxonomy_ids": {
            measure: ALLOCATION_TAXONOMY_IDS.get(measure, "")
            for measure in sorted(
                str(value)
                for value in results.get(
                    "allocation_measure",
                    pd.Series(dtype=object),
                ).dropna().unique()
            )
        },
        "taxonomy_coordinates": {
            measure: ALLOCATION_COORDINATES.get(measure, "")
            for measure in sorted(
                str(value)
                for value in results.get(
                    "allocation_measure",
                    pd.Series(dtype=object),
                ).dropna().unique()
            )
        },
        "primary_allocation_measure": (
            primary_measures[0] if len(primary_measures) == 1 else primary_measures
        ),
        "allocation_definition": ALLOCATION_DEFINITIONS.get(
            primary_measures[0],
            primary_measures[0],
        ) if len(primary_measures) == 1 else {
            measure: ALLOCATION_DEFINITIONS.get(measure, measure)
            for measure in primary_measures
        },
        "allocation_measures": (
            sorted(str(value) for value in results["allocation_measure"].dropna().unique())
            if "allocation_measure" in results.columns
            else []
        ),
        "min_training_rows_settings": (
            sorted(
                int(value)
                for value in results["min_training_rows_setting"].dropna().unique()
            )
            if "min_training_rows_setting" in results.columns
            else list(MIN_TRAINING_ROWS_SETTINGS)
        ),
        "primary_min_training_rows": (
            int(primary_rows["min_training_rows_setting"].dropna().iloc[0])
            if (
                "min_training_rows_setting" in primary_rows.columns
                and not primary_rows["min_training_rows_setting"].dropna().empty
            )
            else PRIMARY_MIN_TRAINING_ROWS
        ),
        "loss_differential_estimand": (
            "mean_standardized_squared_error_candidate_minus_baseline"
        ),
        "summary": json.loads(_summary_rows(results).to_json(orient="records")),
        "all_rows": json.loads(results.to_json(orient="records")),
    }
    summary_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    tex_path.write_text(render_destination_oos_tex(results), encoding="utf-8")
    auxiliary_tex_path.write_text(
        render_destination_oos_asset_auxiliary_tex(results),
        encoding="utf-8",
    )
    return [results_path, summary_path, tex_path, auxiliary_tex_path]


__all__ = [
    "ALLOCATION_FEATURE_COLUMN",
    "ALLOCATION_MEASURES",
    "BOJ_BASELINE_UNIVERSE",
    "BOJ_DIRECT_STOCK_COLUMN",
    "BOJ_RELEASE_LAG_DAYS",
    "LEGACY_ALLOCATION_FEATURE_COLUMN",
    "MIN_TRAINING_ROWS_SETTINGS",
    "PRIMARY_ALLOCATION_MEASURE",
    "PRIMARY_MIN_TRAINING_ROWS",
    "_build_boj_universe_asof",
    "render_destination_oos_asset_auxiliary_tex",
    "render_destination_oos_tex",
    "run_destination_oos",
    "write_destination_oos_outputs",
]
