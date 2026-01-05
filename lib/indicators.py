"""Shared indicator construction helpers (JP & EU).

Functions:
  build_indicators_core(money, q, cred, reg, cfg) -> DataFrame
    - merges entropy & temperature with credit & regulatory tables
    - canonical column renames (p_R->p_C, V_R->V_C, T->T_L)
    - loop area streaming estimator
    - free energy F_C and exergy X_C (fallback to F_C)
  compute_diagnostics(df) -> DataFrame
    - adds Maxwell-like rolling derivatives and first-law decomposition columns

All inputs must have a 'date' column (datetime) and expected numeric columns.
Missing optional pieces (e.g., U or S_M for F_C) will result in NaNs instead of raising.
"""
from __future__ import annotations
import pandas as pd
import numpy as np
from typing import Dict, Any, Optional, Tuple

from lib.loop_area import LoopArea
from lib.entropy import money_entropy
from lib.temperature import liquidity_temperature


DEFAULT_HEADROOM_COLS = ("capital_headroom", "lcr_headroom", "nsfr_headroom")


def _sanitize_numeric(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce") if isinstance(series, pd.Series) else pd.Series(dtype=float)


def _detrend_no_lookahead(series: pd.Series,
                           method: str = "rolling",
                           window: int = 12,
                           min_periods: Optional[int] = None) -> Tuple[pd.Series, pd.Series]:
    """Return (trend, detrended) using only past observations.

    Supported methods:
      - "rolling" (default): simple moving average with given window
      - "ema": exponential moving average (span=window)
    """
    s = _sanitize_numeric(series)
    if window <= 1:
        window = 2
    if min_periods is None or min_periods < 1:
        min_periods = max(1, min(window, 3))
    method = (method or "rolling").strip().lower()
    if method == "ema":
        trend = s.ewm(span=window, adjust=False, min_periods=min_periods).mean()
    else:
        trend = s.rolling(window=window, min_periods=min_periods).mean()
    detrended = s - trend
    return trend, detrended


def _apply_u_detrend(df: pd.DataFrame, cfg: Dict[str, Any]) -> pd.DataFrame:
    if "U" not in df.columns:
        return df
    u_cfg = cfg.get("U_detrend", {}) if isinstance(cfg, dict) else {}
    enabled = u_cfg.get("enabled")
    if enabled is None:
        enabled = True
    if not enabled:
        return df
    try:
        window = int(u_cfg.get("window", 12))
    except Exception:
        window = 12
    try:
        min_periods = u_cfg.get("min_periods")
        min_periods = int(min_periods) if min_periods is not None else None
    except Exception:
        min_periods = None
    method = str(u_cfg.get("method", "rolling")).strip().lower()
    try:
        trend, detrended = _detrend_no_lookahead(df["U"], method=method, window=window, min_periods=min_periods)
        df["U_trend"] = trend
        df["U_star"] = detrended
    except Exception:
        df["U_trend"] = np.nan
        df["U_star"] = np.nan
    return df


def _headroom_columns(cfg: Dict[str, Any]) -> Tuple[str, ...]:
    cols = cfg.get("V_C_headroom_cols") if isinstance(cfg, dict) else None
    if cols and isinstance(cols, (list, tuple)):
        return tuple(str(c) for c in cols)
    return DEFAULT_HEADROOM_COLS


def _derive_headroom(df: pd.DataFrame, cfg: Dict[str, Any]) -> Tuple[pd.DataFrame, Tuple[str, ...]]:
    base = None
    for cand in ("V_C", "V_R"):
        if cand in df.columns:
            ser = _sanitize_numeric(df[cand])
            if ser.notna().any():
                base = ser
                break
    if base is None:
        return df, tuple()
    scales = cfg.get("V_C_headroom_scales") if isinstance(cfg, dict) else None
    if not scales or not isinstance(scales, (list, tuple)):
        scales = (1.0, 1.0, 1.0)
    cols = _headroom_columns(cfg)
    derived_cols = []
    for idx, col in enumerate(cols):
        scale = float(scales[idx]) if idx < len(scales) else float(scales[-1])
        col_name = col or f"headroom_{idx}"
        df[col_name] = base * scale
        derived_cols.append(col_name)
    return df, tuple(derived_cols)


def _apply_vc_formula(df: pd.DataFrame, cfg: Dict[str, Any]) -> pd.DataFrame:
    mode = str(cfg.get("V_C_formula", "legacy")).strip().lower() if isinstance(cfg, dict) else "legacy"
    if mode != "min_headroom":
        return df
    candidate_cols = list(_headroom_columns(cfg))
    available = [c for c in candidate_cols if c in df.columns]
    if not available:
        inferred = [c for c in df.columns if str(c).endswith("_headroom")]
        available = inferred
    if not available:
        df, inferred_tuple = _derive_headroom(df, cfg)
        available = list(inferred_tuple)
    if not available:
        return df
    headroom = df[available].apply(pd.to_numeric, errors="coerce")
    vc_head = headroom.min(axis=1, skipna=True)
    if "V_C" in df.columns:
        legacy = _sanitize_numeric(df["V_C"])
    elif "V_R" in df.columns:
        legacy = _sanitize_numeric(df["V_R"])
    else:
        legacy = pd.Series(np.nan, index=df.index)
    df["V_C_legacy"] = legacy
    df["V_C_headroom"] = vc_head
    mask = vc_head.notna()
    if mask.any():
        df.loc[mask, "V_C"] = vc_head[mask]
    df["V_C_formula_used"] = "min_headroom"
    return df


def _apply_free_energy_baseline(df: pd.DataFrame, cfg: Dict[str, Any]) -> pd.DataFrame:
    if "F_C" not in df.columns or not isinstance(cfg, dict):
        return df
    mode = str(cfg.get("F_C_baseline_mode", "none") or "").strip().lower()
    if mode in ("", "none", "raw"):
        return df
    fnum = pd.to_numeric(df["F_C"], errors="coerce")
    if fnum.dropna().empty:
        return df

    ref = None
    if mode == "min":
        ref = fnum.min()
    elif mode == "quantile":
        try:
            q = float(cfg.get("F_C_baseline_quantile", 0.05))
        except Exception:
            q = 0.05
        q = min(max(q, 0.0), 1.0)
        ref = fnum.quantile(q)
    elif mode == "first":
        idx = fnum.first_valid_index()
        ref = fnum.loc[idx] if idx is not None else np.nan
    elif mode == "value":
        ref = cfg.get("F_C_baseline_value")
    else:
        return df
    try:
        ref_val = float(ref)
    except Exception:
        return df
    if not np.isfinite(ref_val):
        return df
    try:
        eps = float(cfg.get("F_C_baseline_eps", 0.0) or 0.0)
    except Exception:
        eps = 0.0

    offset = max(0.0, -ref_val + eps)
    df["F_C_baseline"] = ref_val
    df["F_C_baseline_mode"] = mode
    df["F_C_baseline_offset"] = offset
    if offset:
        df["F_C"] = fnum + offset
        if "X_C" in df.columns:
            try:
                df["X_C"] = pd.to_numeric(df["X_C"], errors="coerce") + offset
            except Exception:
                pass
    return df


def _apply_external_coupling(df: pd.DataFrame, cfg: Dict[str, Any]) -> pd.DataFrame:
    ext_cfg = cfg.get("external_coupling") if isinstance(cfg, dict) else None
    if not isinstance(ext_cfg, dict) or not ext_cfg.get("enabled"):
        return df
    alpha = float(ext_cfg.get("alpha", 0.0) or 0.0)
    delta = float(ext_cfg.get("delta", 0.0) or 0.0)
    if alpha and "E_p" in df.columns and "p_C" in df.columns:
        baseline = pd.to_numeric(df["p_C"], errors="coerce")
        ep = pd.to_numeric(df["E_p"], errors="coerce")
        contrib = alpha * ep
        df["p_C_baseline"] = baseline
        df["E_p_contrib"] = contrib
        df["p_C_coupling_alpha"] = alpha
        df["p_C"] = baseline + contrib.fillna(0.0)
    if delta and "E_T" in df.columns and "T_L" in df.columns:
        baseline_t = pd.to_numeric(df["T_L"], errors="coerce")
        et = pd.to_numeric(df["E_T"], errors="coerce")
        contrib_t = delta * et
        df["T_L_baseline"] = baseline_t
        df["E_T_contrib"] = contrib_t
        df["T_L_coupling_delta"] = delta
        df["T_L"] = baseline_t + contrib_t.fillna(0.0)
    return df


def _apply_chemical_potentials(df: pd.DataFrame,
                               cfg: Dict[str, Any],
                               q_cols: Optional[Tuple[str, ...]] = None) -> pd.DataFrame:
    if not q_cols:
        return df
    if "M_in" not in df.columns:
        return df
    q_cols_present = [c for c in q_cols if c in df.columns]
    if not q_cols_present:
        return df
    try:
        k_val = float(cfg.get("k", 1.0))
    except Exception:
        k_val = 1.0
    if k_val == 0:
        return df
    try:
        t0 = float(cfg.get("T0", 1.0))
    except Exception:
        t0 = 1.0
    eps = cfg.get("mu_share_floor", 1e-6)
    try:
        eps = float(eps)
    except Exception:
        eps = 1e-6
    shares = df[q_cols_present].astype(float).clip(lower=eps)
    M_in = pd.to_numeric(df["M_in"], errors="coerce")
    factor = t0 * k_val * M_in
    log_term = np.log(shares)
    for col in q_cols_present:
        df[f"mu_{col}"] = factor * (log_term[col] + 1.0)

    # Relative chemical potentials Δμ_i = μ_i − μ̄ (per row, across buckets).
    # These remain diagnostics only and are not yet used to drive any flows.
    mu_cols = [f"mu_{col}" for col in q_cols_present if f"mu_{col}" in df.columns]
    if mu_cols:
        mu_vals = df[mu_cols].apply(pd.to_numeric, errors="coerce")
        mu_bar = mu_vals.mean(axis=1)
        df["mu_mean"] = mu_bar
        for col in mu_cols:
            dcol = col.replace("mu_", "dmu_", 1)
            df[dcol] = mu_vals[col] - mu_bar
    return df


def build_indicators_core(money: pd.DataFrame,
                           q: pd.DataFrame,
                           cred: pd.DataFrame,
                           reg: pd.DataFrame,
                           cfg: Dict[str, Any]) -> pd.DataFrame:
    """Merge sources and compute loop area, F_C, X_C.
    Expects money columns (M_in, M_out), q shares, cred columns (U, S_M inputs after entropy merge), reg (p_R,V_R).
    """
    t0 = float(cfg.get("T0", 1.0))
    lam = float(cfg.get("loop_forgetting", 0.98))
    k_val = float(cfg.get("k", 1.0))
    q_cols_cfg = cfg.get("q_cols")  # optional MECE categories override
    per_cat = bool(cfg.get("entropy_per_category", False))
    if q_cols_cfg and isinstance(q_cols_cfg, (list, tuple)):
        S = money_entropy(money, q, k=k_val, q_cols=q_cols_cfg, per_category=per_cat)
    else:
        S = money_entropy(money, q, k=k_val, per_category=per_cat)  # provides S_M
    T = liquidity_temperature(cred)       # provides T (later T_L)

    df = cred.merge(S, on="date").merge(T, on="date").merge(reg, on="date")
    df = df.sort_values("date").reset_index(drop=True)

    # Attach underlying allocation share columns (q_*) so downstream tests and
    # diagnostics can recompute normalized entropy independently. These are NOT
    # needed for S_M computation itself (already done) but provide transparency.
    q_share_cols = [c for c in q.columns if c.startswith("q_")]
    if q_share_cols:
        df_q = q[["date"] + q_share_cols].drop_duplicates("date")
        df = df.merge(df_q, on="date", how="left")

    money_cols = [c for c in ("M_in", "M_out") if c in money.columns]
    if money_cols:
        df_money = money[["date"] + money_cols].drop_duplicates("date")
        df = df.merge(df_money, on="date", how="left")

    la = LoopArea(lam=lam)
    areas = []
    for _, r in df.iterrows():
        areas.append(la.update(r.get("p_R"), r.get("V_R")))
    df["loop_area"] = areas

    rename_map = {}
    if "p_R" in df.columns and "p_C" not in df.columns:
        rename_map["p_R"] = "p_C"
    if "V_R" in df.columns and "V_C" not in df.columns:
        rename_map["V_R"] = "V_C"
    if "T" in df.columns and "T_L" not in df.columns:
        rename_map["T"] = "T_L"
    if rename_map:
        df = df.rename(columns=rename_map)

    df = _apply_external_coupling(df, cfg)
    chem_cols = tuple(q_cols_cfg) if q_cols_cfg else tuple(q_share_cols)
    df = _apply_chemical_potentials(df, cfg, chem_cols)
    df = _apply_vc_formula(df, cfg)
    df = _apply_u_detrend(df, cfg)

    # Choose which U-series to use for free energy / exergy.
    # Default: raw U. If cfg["U_energy_mode"] asks for a detrended series and
    # U_star (cycle) or U_trend exist, use those instead.
    u_eff_col = "U"
    if "U" not in df.columns:
        # If U is missing but detrended variants exist, fall back to U_star.
        if "U_star" in df.columns:
            u_eff_col = "U_star"
        elif "U_trend" in df.columns:
            u_eff_col = "U_trend"
    else:
        mode_u = str(cfg.get("U_energy_mode", "") or "").strip().lower()
        if mode_u in ("detrended", "cycle", "star") and "U_star" in df.columns:
            u_eff_col = "U_star"
        elif mode_u in ("trend", "baseline") and "U_trend" in df.columns:
            u_eff_col = "U_trend"

    df["U_used_for_energy"] = u_eff_col

    if all(c in df.columns for c in [u_eff_col, "S_M"]):
        u_series = pd.to_numeric(df[u_eff_col], errors="coerce")
        s_series = pd.to_numeric(df["S_M"], errors="coerce")
        df["F_C"] = u_series - t0 * s_series
    else:
        df["F_C"] = np.nan

    p0 = cfg.get("p0"); V0 = cfg.get("V0"); U0 = cfg.get("U0"); S0 = cfg.get("S0")
    if all(v is not None for v in (p0, V0, U0, S0)) and "V_C" in df.columns and "S_M" in df.columns and u_eff_col in df.columns:
        try:
            p0f, V0f, U0f, S0f = float(p0), float(V0), float(U0), float(S0)
            u_series = pd.to_numeric(df[u_eff_col], errors="coerce")
            v_series = pd.to_numeric(df["V_C"], errors="coerce")
            s_series = pd.to_numeric(df["S_M"], errors="coerce")
            df["X_C"] = (u_series - U0f) + p0f * (v_series - V0f) - t0 * (s_series - S0f)
        except Exception:
            df["X_C"] = df["F_C"]
    else:
        df["X_C"] = df["F_C"]

    # Shift F_C/X_C upward using a configurable baseline (e.g., min or quantile)
    df = _apply_free_energy_baseline(df, cfg)

    # Enforce non-negative exergy by baseline adjustment or clipping.
    # Default behavior: clip negatives at 0 (shift, if used, is for visualization only).
    # Configure via cfg:
    #   exergy_floor_zero: bool (default True)
    #   exergy_floor_mode: 'clip' (default) or 'shift'
    try:
        if bool(cfg.get("exergy_floor_zero", True)) and "X_C" in df.columns:
            mode = str(cfg.get("exergy_floor_mode", "clip")).strip().lower()
            xnum = pd.to_numeric(df["X_C"], errors="coerce")
            if mode == "clip":
                df["X_C"] = xnum.clip(lower=0)
            else:
                xmin = float(xnum.min()) if np.isfinite(xnum.min()) else np.nan
                if np.isfinite(xmin) and xmin < 0:
                    df["X_C"] = xnum - xmin
    except Exception:
        # If anything goes wrong, leave X_C as-is
        pass

    # Fixed-reference split of free energy into surplus/shortage components
    # ΔF_C(t) = F_C(t) - F_C_ref; X_C_plus = max(0, ΔF_C); X_C_minus = max(0, -ΔF_C)
    try:
        fc = pd.to_numeric(df.get("F_C", pd.Series([], dtype=float)), errors="coerce")
        f_ref = None
        # Preference order for reference: explicit cfg value -> cfg date pick -> first valid
        if "F_C_ref" in cfg:
            try:
                f_ref = float(cfg.get("F_C_ref"))
            except Exception:
                f_ref = None
        if f_ref is None and "F_C_ref_date" in cfg:
            try:
                ref_dt = pd.to_datetime(cfg.get("F_C_ref_date"))
                # exact match only; if multiple, take first
                match = df.loc[df["date"] == ref_dt, "F_C"]
                if not match.empty:
                    f_ref = float(pd.to_numeric(match, errors="coerce").dropna().iloc[0])
            except Exception:
                f_ref = None
        if f_ref is None:
            idx0 = fc.first_valid_index()
            if idx0 is not None:
                f_ref = float(fc.loc[idx0])
        if f_ref is not None and np.isfinite(f_ref):
            dF = fc.astype(float) - f_ref
            df["Delta_F_C"] = dF
            df["X_C_plus"] = dF.clip(lower=0)
            df["X_C_minus"] = (-dF).clip(lower=0)
        else:
            # If we cannot resolve a fixed reference, emit NaNs for the split
            df["Delta_F_C"] = np.nan
            df["X_C_plus"] = np.nan
            df["X_C_minus"] = np.nan
    except Exception:
        # Do not fail the pipeline if any issue occurs; keep base columns
        df["Delta_F_C"] = np.nan
        df["X_C_plus"] = np.nan
        df["X_C_minus"] = np.nan

    return df


def compute_diagnostics(df: pd.DataFrame, window: int = 24) -> pd.DataFrame:
    """Add Maxwell-like and first-law diagnostics.

    Drops leading rows until at least window non-missing observations exist for core variables
    to stabilize OLS. Window can be overridden by DIAG_WINDOW env variable.
    """
    import os
    try:
        env_w = int(os.getenv("DIAG_WINDOW", window))
        if env_w >= 8:  # sanity lower bound
            window = env_w
    except Exception:
        pass
    required = ["S_M", "T_L", "p_C", "V_C", "U"]
    if not all(c in df.columns for c in required):
        return df

    # Drop initial rows with NA in core columns to reduce distortion
    core_subset = df[required].dropna()
    if len(core_subset) < window:
        # not enough data; skip diagnostics
        return df
    first_valid_date = core_subset.iloc[0]["S_M"], core_subset.index[0]
    # where all required present
    mask_valid = df[required].notna().all(axis=1)
    # keep all rows but we only compute for rolling windows; early rows remain NaN
    n = len(df)
    def _rolling_partial_beta(y, x_main, x_cond):
        out = np.full(n, np.nan)
        cols = [y, x_main, x_cond]
        X = df[cols].astype(float).to_numpy()
        for i in range(window - 1, n):
            sl = X[i - window + 1:i + 1, :]
            if np.isnan(sl).any():
                continue
            Yw = sl[:, 0]
            Xm = np.column_stack([np.ones(window), sl[:, 1], sl[:, 2]])
            try:
                beta, *_ = np.linalg.lstsq(Xm, Yw, rcond=None)
                out[i] = beta[1]
            except Exception:
                out[i] = np.nan
        return out

    df["dS_dV_at_T"] = _rolling_partial_beta("S_M", "V_C", "T_L")
    df["dp_dT_at_V"] = _rolling_partial_beta("p_C", "T_L", "V_C")
    df["maxwell_gap"] = df["dS_dV_at_T"] - df["dp_dT_at_V"]

    df["dU"] = df["U"].astype(float).diff()
    df["dS"] = df["S_M"].astype(float).diff()
    df["dV"] = df["V_C"].astype(float).diff()
    # Also expose dF_C (helps standardized cross-region comparison of changes)
    if "F_C" in df.columns:
        try:
            df["dF_C"] = df["F_C"].astype(float).diff()
        except Exception:
            df["dF_C"] = np.nan
    df["T_bar"] = df["T_L"].astype(float).rolling(2).mean()
    df["p_bar"] = df["p_C"].astype(float).rolling(2).mean()
    df["Q_like"] = df["T_bar"] * df["dS"]
    df["W_like"] = df["p_bar"] * df["dV"]
    df["dU_pred"] = df["Q_like"] - df["W_like"]
    df["firstlaw_resid"] = df["dU"] - df["dU_pred"]
    return df
