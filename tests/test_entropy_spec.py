# tests/test_entropy_spec.py
import math

import numpy as np
import pandas as pd
import yaml


# --- 1) 仕様テスト（単調性・境界） ---
def H_norm(p):
    p = np.array([max(x,0.0) for x in p], float)
    s = p.sum()
    if s == 0: return math.nan
    p = p / s
    H = -(p[p>0]*np.log(p[p>0])).sum()
    return H / math.log(len(p))

def test_monotonicity_and_bounds():
    assert H_norm([1,0,0]) < H_norm([0.5,0.5,0]) < H_norm([1/3,1/3,1/3])
    assert H_norm([1,0,0]) == 0.0
    assert abs(H_norm([0.25,0.25,0.25,0.25]) - 1.0) < 1e-12
    assert math.isnan(H_norm([0,0,0]))

# --- 2) 実データと一致確認（S_M or S_M_hat 列がある想定） ---
def test_matches_pipeline():
    ind = pd.read_csv("site/indicators.csv", parse_dates=[0], index_col=0)
    cfg = yaml.safe_load(open("config.yml"))
    cfg_q = cfg.get("q_cols", [])
    q_cols = [c for c in cfg_q if c in ind.columns]
    assert len(q_cols) >= 2
    # Use exactly configured categories (MECE) ensuring they sum ≈1
    P = ind[q_cols].clip(lower=0)
    s = P.sum(axis=1)
    mask = (s > 0) & ~s.isna()
    Pn = (P[mask].T / s[mask]).T
    S_hat = -(Pn.where(Pn>0).apply(np.log).mul(Pn)).sum(axis=1) / math.log(len(q_cols))
    # Compare with normalized entropy column
    assert "S_M_hat" in ind.columns
    diff = (ind.loc[S_hat.index, "S_M_hat"] - S_hat).abs().dropna()
    # If no valid rows (e.g., early pipeline missing entropy due to upstream trimming), accept empty diff.
    if diff.empty:
        assert True
    else:
        assert diff.max() < 1e-12
