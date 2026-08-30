import pandas as pd
import numpy as np

EPS = 1e-12


def _as_positive_log(series: pd.Series) -> pd.Series:
    s = pd.to_numeric(series, errors="coerce").astype(float)
    return np.log(s.clip(lower=EPS))


def expanding_zscore(s: pd.Series, min_periods: int = 8) -> pd.Series:
    s = pd.to_numeric(s, errors="coerce").astype(float)
    expanding = s.expanding(min_periods=max(2, min_periods))
    mean = expanding.mean()
    std = expanding.std(ddof=0)
    z = (s - mean) / (std + EPS)
    return z.replace([np.inf, -np.inf], np.nan).fillna(0.0)


def zscore(s: pd.Series) -> pd.Series:
    """Retrospective full-sample z-score for visualization checks only."""
    s = pd.to_numeric(s, errors="coerce").astype(float)
    return (s - s.mean()) / (s.std(ddof=0) + EPS)


def _logistic(s: pd.Series) -> pd.Series:
    x = pd.to_numeric(s, errors="coerce").astype(float).clip(lower=-40.0, upper=40.0)
    return 1.0 / (1.0 + np.exp(-x))


def liquidity_state_index(df_credit: pd.DataFrame, min_periods: int = 8) -> pd.DataFrame:
    df = df_credit.copy()
    if "date" in df.columns:
        df = df.assign(date=pd.to_datetime(df["date"], errors="coerce")).sort_values("date").reset_index(drop=True)

    spread_z = -expanding_zscore(_as_positive_log(df["spread"]), min_periods=min_periods)
    depth_z = expanding_zscore(_as_positive_log(df["depth"]), min_periods=min_periods)
    turnover_z = expanding_zscore(_as_positive_log(df["turnover"]), min_periods=min_periods)

    additive_score = (spread_z + depth_z + turnover_z) / 3.0
    spread_idx = _logistic(spread_z)
    depth_idx = _logistic(depth_z)
    turnover_idx = _logistic(turnover_z)

    out = df[["date"]].copy()
    out["T_L"] = _logistic(additive_score)
    out["T_L_multiplicative_check"] = (spread_idx * depth_idx * turnover_idx).pow(1.0 / 3.0)
    return out

def liquidity_temperature(df_credit: pd.DataFrame) -> pd.DataFrame:
    return liquidity_state_index(df_credit)
