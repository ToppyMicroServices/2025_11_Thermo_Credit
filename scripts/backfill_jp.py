import os
import sys

import pandas as pd

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
DATA = os.path.join(ROOT, "data")

# Simple synthetic backfill: linear interpolation backwards with mild trend.
# Only fills if earliest date > 2015-01-01, generates quarterly rows from 2010Q1.

def _load(path, parse_dates=True):
    if not os.path.exists(path):
        return None
    df = pd.read_csv(path)
    if parse_dates and "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"], errors="coerce")
        df = df.dropna(subset=["date"]).sort_values("date").reset_index(drop=True)
    return df

money_path = os.path.join(DATA, "money.csv")
alloc_path = os.path.join(DATA, "allocation_q.csv")
cred_path  = os.path.join(DATA, "credit.csv")
reg_path   = os.path.join(DATA, "reg_pressure.csv")

money = _load(money_path)
alloc = _load(alloc_path)
cred  = _load(cred_path)
reg   = _load(reg_path)

START_TARGET = pd.Timestamp("2010-01-01")
CUT = pd.Timestamp("2015-01-01")

if any(df is None or df.empty for df in [money, alloc, cred, reg]):
    print("[backfill_jp] Missing one or more JP base files; abort.")
    sys.exit(0)

# Only backfill if current earliest date is after CUT
if money["date"].min() <= CUT:
    print("[backfill_jp] Existing JP data already starts before or at 2015; no backfill.")
    sys.exit(0)

# Build quarterly range
quarters = pd.date_range(START_TARGET, money["date"].min() - pd.offsets.QuarterEnd(0), freq="Q")
quarters = quarters[quarters < money["date"].min()]  # strictly earlier
if quarters.empty:
    print("[backfill_jp] No quarters to backfill.")
    sys.exit(0)

# Helper to extrapolate backwards with small annual growth (~2%)
def backward_extrap(series_latest: float, n: int, step: float):
    # produce n values older, reverse chronological
    vals = []
    val = series_latest
    for _i in range(n):
        val /= (1.0 + step)  # step growth forward so divide for backwards
        vals.append(val)
    return list(reversed(vals))

latest_M_in = float(money.iloc[0]["M_in"])  # earliest row after sort is oldest? we sorted ascending
latest_M_out = float(money.iloc[0]["M_out"])  # use earliest as anchor; apply growth forward to reach present
nq = len(quarters)
back_M_in = backward_extrap(latest_M_in, nq, 0.02/4)
back_M_out = backward_extrap(latest_M_out, nq, 0.02/4)

# Allocation shares: assume gradual shift; take earliest allocation row
alloc_row = alloc.iloc[0]
q_pay0, q_firm0, q_asset0, q_reserve0 = [float(alloc_row[c]) for c in ["q_pay","q_firm","q_asset","q_reserve"]]
# drift small toward a stable mix
back_q_pay    = [q_pay0 + (i* -0.0005) for i in range(nq)]
back_q_firm   = [q_firm0 + (i* 0.0003) for i in range(nq)]
back_q_asset  = [q_asset0 + (i* 0.0004) for i in range(nq)]
back_q_res    = [q_reserve0 for _ in range(nq)]
# normalize each quarter row to sum ~1
def _norm_row(a,b,c,d):
    s=a+b+c+d
    return a/s, b/s, c/s, d/s
norm_alloc = [ _norm_row(a,b,c,d) for a,b,c,d in zip(back_q_pay,back_q_firm,back_q_asset,back_q_res) ]
back_q_pay, back_q_firm, back_q_asset, back_q_res = zip(*norm_alloc)

# Credit: extrapolate U, Y, L_real, L_asset, spread, depth, turnover backwards similarly
cred_row = cred.iloc[0]
U0 = float(cred_row["U"]); Y0 = float(cred_row["Y"])
Lr0 = float(cred_row["L_real"]); La0 = float(cred_row["L_asset"])
spread0 = float(cred_row["spread"]); depth0 = float(cred_row["depth"]); turn0 = float(cred_row["turnover"])
back_U = backward_extrap(U0, nq, 0.01/4)
back_Y = backward_extrap(Y0, nq, 0.01/4)
back_Lr = backward_extrap(Lr0, nq, 0.015/4)
back_La = backward_extrap(La0, nq, 0.015/4)
back_spread = backward_extrap(spread0, nq, -0.005/4)  # allow wider spreads earlier (so negative growth forward)
back_depth = backward_extrap(depth0, nq, 0.01/4)
back_turn = backward_extrap(turn0, nq, 0.02/4)

# Regulatory pressure: assume p_R slightly higher earlier, V_R lower
reg_row = reg.iloc[0]
pR0 = float(reg_row["p_R"]); VR0 = float(reg_row["V_R"])
back_pR = backward_extrap(pR0, nq, -0.01/4)  # a bit higher earlier
back_VR = backward_extrap(VR0, nq, 0.01/4)

# Build DataFrames
back_money = pd.DataFrame({"date": quarters, "M_in": back_M_in, "M_out": back_M_out})
back_alloc = pd.DataFrame({"date": quarters, "q_pay": back_q_pay, "q_firm": back_q_firm, "q_asset": back_q_asset, "q_reserve": back_q_res})
back_cred  = pd.DataFrame({"date": quarters, "L_real": back_Lr, "L_asset": back_La, "U": back_U, "Y": back_Y, "spread": back_spread, "depth": back_depth, "turnover": back_turn})
back_reg   = pd.DataFrame({"date": quarters, "p_R": back_pR, "V_R": back_VR})

# Concatenate and save
money_full = pd.concat([back_money, money], ignore_index=True).sort_values("date")
alloc_full = pd.concat([back_alloc, alloc], ignore_index=True).sort_values("date")
cred_full  = pd.concat([back_cred, cred], ignore_index=True).sort_values("date")
reg_full   = pd.concat([back_reg, reg], ignore_index=True).sort_values("date")

money_full.to_csv(money_path, index=False)
alloc_full.to_csv(alloc_path, index=False)
cred_full.to_csv(cred_path, index=False)
reg_full.to_csv(reg_path, index=False)
print(f"[backfill_jp] Added {nq} quarterly rows from {START_TARGET.date()} to {money['date'].min().date()}.")
