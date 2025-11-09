import os, sys
import pandas as pd, yaml

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)
from lib.indicators import build_indicators_core
from lib.indicators import compute_diagnostics

try:
    from lib.config_loader import load_config
    cfg = load_config("jp")
except Exception:
    cfg = yaml.safe_load(open("config.yml"))

# Build JP money from raw series when available
def _read_raw(sid: str) -> pd.DataFrame:
    path = os.path.join("data", f"{sid}.csv")
    if not os.path.exists(path):
        return pd.DataFrame()
    df = pd.read_csv(path, parse_dates=["date"]).dropna()
    return df.sort_values("date")

boj = _read_raw("JPNASSETS")
m2  = _read_raw("MYAGM2JPM189S")
yld = _read_raw("IRLTLT01JPM156N")

# Allow environment to define earliest JP date (e.g. JP_START=2012-01-01)
JP_START = os.getenv("JP_START", "2012-01-01")
try:
    jp_start_ts = pd.Timestamp(JP_START)
    for df in (boj, m2, yld):
        if not df.empty:
            df.drop(df[df["date"] < jp_start_ts].index, inplace=True)
except Exception:
    pass

def _qe_dec(dfm: pd.DataFrame) -> pd.DataFrame:
    if dfm.empty:
        return dfm
    dfm = dfm.set_index("date").resample("QE-DEC").last().reset_index()
    return dfm

# Money scale from M2 if present, else BoJ assets; base from BoJ assets
money_scale = m2 if not m2.empty else boj
base = boj
if money_scale.empty or base.empty:
    # fallback to existing files if raw not present
    money = pd.read_csv("data/money.csv", parse_dates=["date"]).sort_values("date")
else:
    ms_q = _qe_dec(money_scale)
    bs_q = _qe_dec(base)
    money = ms_q.merge(bs_q, on="date", how="outer", suffixes=("_ms","_bs")).sort_values("date")
    # derive M_in/M_out: treat money scale as inflow proxy, base as outflow proxy (or vice versa)
    money = money.rename(columns={"value_ms":"M_in","value_bs":"M_out"})

# Allocation: extend back to money start if needed by forward-filling earliest row
q = pd.read_csv("data/allocation_q.csv", parse_dates=["date"]).sort_values("date")
if not money.empty and not q.empty and money["date"].min() < q["date"].min():
    first_row = q.iloc[0]
    ext_idx = pd.date_range(money["date"].min(), q["date"].min() - pd.offsets.QuarterEnd(0), freq="QE-DEC")
    if len(ext_idx) > 0:
        q_ext = pd.DataFrame({
            "date": ext_idx,
            "q_pay": first_row["q_pay"],
            "q_firm": first_row["q_firm"],
            "q_asset": first_row["q_asset"],
            "q_reserve": first_row["q_reserve"],
        })
        q = pd.concat([q_ext, q], ignore_index=True).sort_values("date").reset_index(drop=True)

# Credit & regulatory pressure: if existing quarterly files cover earlier period, keep; otherwise leave as-is
cred  = pd.read_csv("data/credit.csv", parse_dates=["date"]).sort_values("date")
reg   = pd.read_csv("data/reg_pressure.csv", parse_dates=["date"]).sort_values("date")

df = build_indicators_core(money, q, cred, reg, cfg)
df = compute_diagnostics(df)

os.makedirs("site", exist_ok=True)
df.to_csv("site/indicators.csv", index=False)
print("Wrote site/indicators.csv")
