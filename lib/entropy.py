import math
from collections.abc import Iterable, Sequence

import numpy as np
import pandas as pd


def shannon_H_from_probs(p: np.ndarray) -> float:
    p = p.astype(float)
    # Align with test behavior: clamp negatives to 0 before normalization
    p[p < 0] = 0.0
    s = p.sum()
    if s <= 0 or not np.isfinite(s):
        return float("nan")
    p = p / s
    mask = p > 0
    if not np.any(mask):
        return 0.0
    return float(-(p[mask] * np.log(p[mask])).sum())

def shannon_H(row: pd.Series, cols: Iterable[str]) -> float:
    probs = row[list(cols)].values.astype(float)
    return shannon_H_from_probs(probs)

def money_entropy(
    df_money: pd.DataFrame,
    df_q: pd.DataFrame,
    k: float = 1.0,
    q_cols: Sequence[str] = ("q_pay","q_firm","q_asset","q_reserve"),
    per_category: bool = False,
) -> pd.DataFrame:
    """Compute monetary entropy and optional per-category contributions.

    Parameters
    ----------
    df_money : DataFrame with columns ['date','M_in','M_out']
    df_q     : DataFrame with allocation share columns summing (approximately) to 1.
    k        : scaling constant.
    q_cols   : columns to use for entropy computation (configurable; MECE categories).
    per_category : if True, include S_M_in_<col> for each category (k * M_in * p_i).
    """
    if not set(q_cols).issubset(df_q.columns):
        # fallback to intersection that exists
        q_cols = [c for c in q_cols if c in df_q.columns]
    df = df_money.merge(df_q, on="date", how="inner").copy()
    if not q_cols:
        df["Hq"] = np.nan
    else:
        df["Hq"] = df.apply(lambda r: shannon_H(r, q_cols), axis=1)
    df["S_M_in"]  = k * df["M_in"].astype(float)  * df["Hq"]
    df["S_M_out"] = k * df["M_out"].astype(float) * df["Hq"]
    df["S_M"] = df["S_M_in"]
    # Expose normalized entropy (scale-free) dividing by log(K)
    K = len(q_cols) if q_cols else 0
    if K > 0:
        df["S_M_hat"] = df["Hq"].astype(float) / math.log(K)
    else:
        df["S_M_hat"] = np.nan
    cols = ["date","Hq","S_M_hat","S_M","S_M_in","S_M_out"]
    if per_category and q_cols:
        # raw probability vector
        for c in q_cols:
            df[f"S_M_in_{c}"] = k * df["M_in"].astype(float) * df[c].astype(float).clip(0, 1)
            cols.append(f"S_M_in_{c}")
    return df[cols]
