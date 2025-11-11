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
from typing import Dict, Any

from lib.loop_area import LoopArea
from lib.entropy import money_entropy
from lib.temperature import liquidity_temperature


def build_indicators_core(money: pd.DataFrame,
                           q: pd.DataFrame,
                           cred: pd.DataFrame,
                           reg: pd.DataFrame,
                           cfg: Dict[str, Any]) -> pd.DataFrame:
    """Merge sources and compute loop area, F_C, X_C.
    Expects money columns (M_in, M_out), q shares, cred columns (U, S_M inputs after entropy merge), reg (p_R,V_R).
    """
    T0 = float(cfg.get("T0", 1.0))
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

    if all(c in df.columns for c in ["U", "S_M"]):
        df["F_C"] = df["U"].astype(float) - T0 * df["S_M"].astype(float)
    else:
        df["F_C"] = np.nan

    p0 = cfg.get("p0"); V0 = cfg.get("V0"); U0 = cfg.get("U0"); S0 = cfg.get("S0")
    if all(v is not None for v in (p0, V0, U0, S0)) and "V_C" in df.columns and "S_M" in df.columns and "U" in df.columns:
        try:
            p0f, V0f, U0f, S0f = float(p0), float(V0), float(U0), float(S0)
            df["X_C"] = (df["U"].astype(float) - U0f) + p0f * (df["V_C"].astype(float) - V0f) - T0 * (df["S_M"].astype(float) - S0f)
        except Exception:
            df["X_C"] = df["F_C"]
    else:
        df["X_C"] = df["F_C"]

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
    df["T_bar"] = df["T_L"].astype(float).rolling(2).mean()
    df["p_bar"] = df["p_C"].astype(float).rolling(2).mean()
    df["Q_like"] = df["T_bar"] * df["dS"]
    df["W_like"] = df["p_bar"] * df["dV"]
    df["dU_pred"] = df["Q_like"] - df["W_like"]
    df["firstlaw_resid"] = df["dU"] - df["dU_pred"]
    return df
