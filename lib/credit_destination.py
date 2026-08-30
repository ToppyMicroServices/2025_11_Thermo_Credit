from __future__ import annotations

from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd

from lib.boj_credit_taxonomies import PRIMARY_BUCKETS, PRIMARY_TAXONOMY_ID


DEFAULT_LAMBDA_B = 0.5
DEFAULT_HOUSING_CONSTRUCTION_SHARE = 0.35
DEFAULT_HOUSEHOLD_HOUSING_SHARE = 0.40


def _first_numeric(frame: pd.DataFrame, candidates: Sequence[str], *, default: float = 0.0) -> pd.Series:
    for col in candidates:
        if col not in frame.columns:
            continue
        series = pd.to_numeric(frame[col], errors="coerce")
        if series.notna().any():
            return series.fillna(default)
    return pd.Series(default, index=frame.index, dtype=float)


def _bounded_config_float(cfg: Mapping[str, Any], key: str, default: float) -> float:
    try:
        value = float(cfg.get(key, default))
    except Exception:
        value = default
    return min(max(value, 0.0), 1.0)


def credit_destination_config(cfg: Mapping[str, Any] | None) -> dict[str, Any]:
    root = cfg if isinstance(cfg, Mapping) else {}
    raw = root.get("credit_destination", {})
    dest_cfg = raw if isinstance(raw, Mapping) else {}
    return {
        "enabled": bool(dest_cfg.get("enabled", True)),
        "lambda_B": _bounded_config_float(dest_cfg, "lambda_B", DEFAULT_LAMBDA_B),
        "housing_construction_share": _bounded_config_float(
            dest_cfg,
            "housing_construction_share",
            DEFAULT_HOUSING_CONSTRUCTION_SHARE,
        ),
        "household_housing_share": _bounded_config_float(
            dest_cfg,
            "household_housing_share",
            DEFAULT_HOUSEHOLD_HOUSING_SHARE,
        ),
        "source": str(dest_cfg.get("source", "allocation_proxy")),
        "direct_panel_path": str(dest_cfg.get("direct_panel_path", "") or ""),
    }


def _series_with_any(frame: pd.DataFrame, candidates: Sequence[str]) -> pd.Series | None:
    for col in candidates:
        if col not in frame.columns:
            continue
        series = pd.to_numeric(frame[col], errors="coerce")
        if series.notna().any():
            return series
    return None


def _validity_mask(frame: pd.DataFrame, column: str) -> pd.Series:
    if column not in frame.columns:
        return pd.Series(True, index=frame.index, dtype=bool)
    raw = frame[column]
    if raw.dtype == bool:
        return raw.fillna(False)
    numeric = pd.to_numeric(raw, errors="coerce")
    text = raw.astype(str).str.strip().str.lower().isin({"1", "1.0", "true", "yes"})
    return numeric.gt(0).fillna(False) | text


def _build_primary_direct_destination_panel(
    frame: pd.DataFrame,
    dest_cfg: Mapping[str, Any],
) -> pd.DataFrame:
    raw_components = {
        bucket: _series_with_any(frame, (f"C_{bucket}",))
        for bucket in PRIMARY_BUCKETS
    }
    if any(series is None for series in raw_components.values()):
        return pd.DataFrame()

    valid = _validity_mask(frame, "primary_taxonomy_delta_valid")
    components = {
        bucket: raw_components[bucket].clip(lower=0.0).where(valid)  # type: ignore[union-attr]
        for bucket in PRIMARY_BUCKETS
    }
    component_sum = sum(components.values())
    c_t_source = _series_with_any(frame, ("C_t_primary", "C_t"))
    c_t_mismatch = (
        (c_t_source - component_sum).abs() > 1e-9
        if c_t_source is not None
        else pd.Series(False, index=frame.index)
    )
    c_t = component_sum.replace(0.0, np.nan)
    denominator_1q = component_sum.replace(0.0, np.nan)
    composition_1q = {
        bucket: components[bucket] / denominator_1q
        for bucket in PRIMARY_BUCKETS
    }
    rolling = {
        bucket: components[bucket].rolling(window=4, min_periods=4).sum()
        for bucket in PRIMARY_BUCKETS
    }
    rolling_total = sum(rolling.values())
    composition_4q = {
        bucket: rolling[bucket] / rolling_total.replace(0.0, np.nan)
        for bucket in PRIMARY_BUCKETS
    }
    q_t = composition_4q["NFB"]

    legacy_components = {
        bucket: _series_with_any(frame, (f"C_{bucket}",))
        for bucket in ("G", "B", "E")
    }
    common_valid = _validity_mask(frame, "common_taxonomy_delta_valid")
    legacy = {
        bucket: (
            legacy_components[bucket].clip(lower=0.0).where(common_valid)
            if legacy_components[bucket] is not None
            else pd.Series(np.nan, index=frame.index, dtype=float)
        )
        for bucket in ("G", "B", "E")
    }
    coverage_source = _series_with_any(
        frame,
        (
            "primary_flow_coverage_observed",
            "destination_coverage_observed",
            "destination_coverage",
        ),
    )
    coverage = (
        coverage_source
        if coverage_source is not None
        else pd.Series(np.nan, index=frame.index, dtype=float)
    )
    c_t_raw_delta = _first_numeric(
        frame,
        ("primary_included_net_flow", "C_t_raw_delta"),
        default=np.nan,
    )

    result = pd.DataFrame(
        {
            "date": pd.to_datetime(frame["date"], errors="coerce"),
            "C_t": c_t,
            "C_t_raw_delta": c_t_raw_delta,
            "C_G": legacy["G"],
            "C_B": legacy["B"],
            "C_E": legacy["E"],
            # Compatibility aliases only: these are borrower-composition
            # complements, not identified loan-purpose flows.
            "C_R": components["NFB"],
            "C_A": (
                components["FIN"]
                + components["PROP"]
                + components["HH_NONHOUSING"]
            ),
            "q_t": q_t,
            "one_minus_q_t": 1.0 - q_t,
            **{
                f"C_{bucket}": components[bucket]
                for bucket in PRIMARY_BUCKETS
            },
            **{
                f"borrower_composition_{bucket}_1q": composition_1q[bucket]
                for bucket in PRIMARY_BUCKETS
            },
            **{
                f"borrower_composition_{bucket}_4q": composition_4q[bucket]
                for bucket in PRIMARY_BUCKETS
            },
            "q_t_primary": q_t,
            "destination_coverage": pd.to_numeric(coverage, errors="coerce").clip(
                lower=0.0,
                upper=1.0,
            ),
            "lambda_B": float("nan"),
            "housing_construction_share": float(dest_cfg["housing_construction_share"]),
            "credit_destination_source": str(dest_cfg["source"]),
            "credit_destination_taxonomy_id": PRIMARY_TAXONOMY_ID,
            "credit_destination_overflow_scaled": False,
            "primary_component_total_mismatch": c_t_mismatch.fillna(False).astype(bool),
        }
    )

    retained_prefixes = (
        "legacy_",
        "primary_",
        "werner_",
        "muller_verner_",
        "C_WERNER_",
        "C_MV_",
        "stock_primary_",
        "stock_household_",
        "household_",
    )
    retained_exact = {
        "common_taxonomy_delta_valid",
        "primary_taxonomy_delta_valid",
        "mapped_domestic_stock",
        "stock_local_governments_explicit",
        "stock_overseas_explicit",
        "stock_unresolved_residual",
        "explicit_scope_stock",
        "explicit_scope_gap_to_official_stock",
    }
    for column in frame.columns:
        if column in result.columns:
            continue
        if column in retained_exact or column.startswith(retained_prefixes):
            result[column] = frame[column].to_numpy()
    return result


def _build_direct_destination_panel(
    frame: pd.DataFrame,
    dest_cfg: Mapping[str, Any],
) -> pd.DataFrame:
    primary = _build_primary_direct_destination_panel(frame, dest_cfg)
    if not primary.empty:
        return primary

    index = frame.index
    c_g_raw = _series_with_any(frame, ("C_G", "C_G_observed", "credit_G", "productive_credit_flow"))
    c_b_raw = _series_with_any(frame, ("C_B", "C_B_observed", "credit_B", "construction_credit_flow"))
    c_e_raw = _series_with_any(frame, ("C_E", "C_E_observed", "credit_E", "existing_asset_credit_flow"))
    if c_g_raw is None or c_b_raw is None or c_e_raw is None:
        return pd.DataFrame()

    c_g = c_g_raw.clip(lower=0.0)
    c_b = c_b_raw.clip(lower=0.0)
    c_e = c_e_raw.clip(lower=0.0)
    if "common_taxonomy_delta_valid" in frame.columns:
        valid = _validity_mask(frame, "common_taxonomy_delta_valid")
        c_g = c_g.where(valid)
        c_b = c_b.where(valid)
        c_e = c_e.where(valid)
    component_sum = c_g + c_b + c_e

    c_t_raw = _series_with_any(frame, ("C_t", "C_t_observed", "credit_destination_total"))
    if c_t_raw is None:
        c_t = component_sum
    else:
        c_t = c_t_raw.combine_first(component_sum).fillna(0.0).clip(lower=0.0)
    zero_total = c_t <= 0
    c_t = c_t.where(~zero_total, component_sum)

    overflow = component_sum > c_t.replace(0.0, np.nan)
    if overflow.any():
        scale = (component_sum / c_t.replace(0.0, np.nan)).where(overflow, 1.0).replace(0.0, 1.0)
        c_g = c_g / scale
        c_b = c_b / scale
        c_e = c_e / scale
        component_sum = c_g + c_b + c_e

    # Direct BOJ components are borrower groups, not loan-purpose buckets.
    # Keep the legacy C_R/C_A columns for downstream compatibility, but define
    # them transparently as Group G versus the other mapped borrower groups.
    # Applying a construction weight here would reintroduce an unvalidated
    # destination interpretation.
    lambda_b = float("nan")
    c_r = c_g
    c_a = c_b + c_e
    denominator_1q = component_sum.where(component_sum > 0)
    composition_g_1q = c_g / denominator_1q
    composition_b_1q = c_b / denominator_1q
    composition_e_1q = c_e / denominator_1q
    rolling_g = c_g.rolling(window=4, min_periods=4).sum()
    rolling_b = c_b.rolling(window=4, min_periods=4).sum()
    rolling_e = c_e.rolling(window=4, min_periods=4).sum()
    rolling_total = rolling_g + rolling_b + rolling_e
    composition_g_4q = rolling_g / rolling_total.where(rolling_total > 0)
    composition_b_4q = rolling_b / rolling_total.where(rolling_total > 0)
    composition_e_4q = rolling_e / rolling_total.where(rolling_total > 0)
    is_boj_direct = str(dest_cfg.get("source", "")).startswith("jp_boj")
    q_t = composition_g_4q if is_boj_direct else composition_g_1q

    coverage_source = _series_with_any(frame, ("destination_coverage", "destination_coverage_observed"))
    if coverage_source is None:
        coverage = pd.Series(np.where(c_t > 0, component_sum / c_t, np.nan), index=index, dtype=float)
    else:
        coverage = coverage_source

    return pd.DataFrame(
        {
            "date": pd.to_datetime(frame["date"], errors="coerce"),
            "C_t": c_t,
            "C_t_raw_delta": _first_numeric(frame, ("C_t_raw_delta", "C_t_total_raw_delta"), default=np.nan),
            "C_G": c_g,
            "C_B": c_b,
            "C_E": c_e,
            "C_R": c_r,
            "C_A": c_a,
            "q_t": q_t,
            "one_minus_q_t": 1.0 - q_t,
            "borrower_composition_G_1q": composition_g_1q,
            "borrower_composition_B_1q": composition_b_1q,
            "borrower_composition_E_1q": composition_e_1q,
            "borrower_composition_G_4q": composition_g_4q,
            "borrower_composition_B_4q": composition_b_4q,
            "borrower_composition_E_4q": composition_e_4q,
            # Legacy aliases retained for existing output consumers.
            "operating_borrower_share_1q": composition_g_1q,
            "operating_borrower_share_4q": composition_g_4q,
            "share_G_direct": composition_g_1q,
            "share_B_direct": composition_b_1q,
            "share_E_direct": composition_e_1q,
            "destination_coverage": pd.to_numeric(coverage, errors="coerce").clip(lower=0.0),
            "lambda_B": lambda_b,
            "housing_construction_share": float(dest_cfg["housing_construction_share"]),
            "credit_destination_source": str(dest_cfg["source"]),
            "credit_destination_overflow_scaled": overflow.fillna(False).astype(bool).to_numpy(),
        }
    )


def build_credit_destination_panel(frame: pd.DataFrame, cfg: Mapping[str, Any] | None = None) -> pd.DataFrame:
    """Construct a proxy panel for the core credit-destination variables.

    The preferred future input is loan-purpose data. Until that is available,
    this function maps the existing allocation shares onto the theoretical
    three-way split and scales them by a nonnegative new-credit proxy from
    changes in the real credit stock.
    """
    if frame is None or frame.empty or "date" not in frame.columns:
        return pd.DataFrame()
    dest_cfg = credit_destination_config(cfg)
    if not dest_cfg["enabled"]:
        return pd.DataFrame({"date": pd.to_datetime(frame["date"], errors="coerce")})

    direct = _build_direct_destination_panel(frame, dest_cfg)
    if not direct.empty:
        return direct

    out = pd.DataFrame({"date": pd.to_datetime(frame["date"], errors="coerce")})
    work = frame.copy()
    index = work.index
    l_real = _first_numeric(work, ("L_real", "C_t_stock", "credit_stock"), default=np.nan)
    c_delta = l_real.diff()
    c_t = c_delta.clip(lower=0.0)
    if c_t.notna().any():
        c_t = c_t.fillna(0.0)
    else:
        c_t = pd.Series(np.nan, index=index, dtype=float)

    household_share = _first_numeric(work, ("q_pay", "q_households"), default=0.0).clip(lower=0.0)
    inferred_housing = household_share * float(dest_cfg["household_housing_share"])
    housing_share = _first_numeric(work, ("q_housing", "q_mortgage", "q_real_estate"), default=np.nan)
    housing_share = housing_share.combine_first(inferred_housing).fillna(0.0).clip(lower=0.0)

    productive_share = _first_numeric(
        work,
        ("q_productive", "q_firm", "q_corporates", "q_business", "q_working_capital"),
        default=0.0,
    ).clip(lower=0.0)
    financial_share = _first_numeric(
        work,
        ("q_financial", "q_asset", "q_securities", "q_margin", "q_existing_assets"),
        default=0.0,
    ).clip(lower=0.0)

    construction_share = (housing_share * float(dest_cfg["housing_construction_share"])).clip(lower=0.0)
    existing_asset_share = (
        financial_share + housing_share * (1.0 - float(dest_cfg["housing_construction_share"]))
    ).clip(lower=0.0)
    destination_sum = productive_share + construction_share + existing_asset_share
    overflow = destination_sum > 1.0
    if overflow.any():
        scale = destination_sum.where(destination_sum > 1.0, 1.0)
        productive_share = productive_share / scale
        construction_share = construction_share / scale
        existing_asset_share = existing_asset_share / scale
        destination_sum = productive_share + construction_share + existing_asset_share

    lambda_b = float(dest_cfg["lambda_B"])
    source = str(dest_cfg["source"])
    if source != "allocation_proxy":
        source = "allocation_proxy"
    c_g = c_t * productive_share
    c_b = c_t * construction_share
    c_e = c_t * existing_asset_share
    c_r = c_g + lambda_b * c_b
    c_a = c_e + (1.0 - lambda_b) * c_b
    q_t = np.where(c_t > 0, c_r / c_t, np.nan)

    return pd.DataFrame(
        {
            "date": out["date"],
            "C_t": c_t,
            "C_t_raw_delta": c_delta,
            "C_G": c_g,
            "C_B": c_b,
            "C_E": c_e,
            "C_R": c_r,
            "C_A": c_a,
            "q_t": q_t,
            "one_minus_q_t": np.where(c_t > 0, 1.0 - q_t, np.nan),
            "share_G_proxy": productive_share,
            "share_B_proxy": construction_share,
            "share_E_proxy": existing_asset_share,
            "destination_coverage": destination_sum.clip(lower=0.0, upper=1.0),
            "lambda_B": lambda_b,
            "housing_construction_share": float(dest_cfg["housing_construction_share"]),
            "credit_destination_source": source,
            "credit_destination_overflow_scaled": overflow.astype(bool).to_numpy(),
        }
    )


__all__ = ["build_credit_destination_panel", "credit_destination_config"]
