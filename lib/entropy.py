import numpy as np
import pandas as pd

def shannon_H(row, cols):
    p = np.clip(row[list(cols)].values.astype(float), 1e-12, 1.0)
    p = p / p.sum()
    return float(-(p * np.log(p)).sum())

def money_entropy(df_money: pd.DataFrame, df_q: pd.DataFrame, k: float = 1.0,
                  q_cols=("q_pay","q_firm","q_asset","q_reserve")) -> pd.DataFrame:
    df = df_money.merge(df_q, on="date", how="inner").copy()
    df["Hq"] = df.apply(lambda r: shannon_H(r, q_cols), axis=1)
    df["S_M_in"]  = k * df["M_in"].astype(float)  * df["Hq"]
    df["S_M_out"] = k * df["M_out"].astype(float) * df["Hq"]
    df["S_M"] = df["S_M_in"]
    return df[["date","Hq","S_M","S_M_in","S_M_out"]]
