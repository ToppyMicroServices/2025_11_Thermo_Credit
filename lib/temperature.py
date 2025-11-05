import pandas as pd
import numpy as np

def zscore(s: pd.Series) -> pd.Series:
    s = s.astype(float)
    return (s - s.mean()) / (s.std(ddof=0) + 1e-12)

def liquidity_temperature(df_credit: pd.DataFrame) -> pd.DataFrame:
    z_inv_spread = zscore(1.0 / (df_credit["spread"].astype(float) + 1e-9))
    z_inv_depth  = zscore(1.0 / (df_credit["depth"].astype(float) + 1e-9))
    z_turn       = zscore(df_credit["turnover"].astype(float))
    T_hat = (z_inv_spread * z_inv_depth * (1 + 0.5*z_turn))
    out = df_credit[["date"]].copy()
    # 0-1 normalize
    th_min = T_hat.min()
    th_max = T_hat.max()
    out["T_L"] = (T_hat - th_min) / (th_max - th_min + 1e-12)
    return out
