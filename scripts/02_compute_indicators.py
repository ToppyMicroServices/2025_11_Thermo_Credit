import pandas as pd, yaml
from lib.entropy import money_entropy
from lib.temperature import liquidity_temperature
from lib.loop_area import LoopArea

cfg = yaml.safe_load(open("config.yml"))
T0  = float(cfg.get("T0", 1.0))
lam = float(cfg.get("loop_forgetting", 0.98))

money = pd.read_csv("data/money.csv", parse_dates=["date"])
q     = pd.read_csv("data/allocation_q.csv", parse_dates=["date"])
cred  = pd.read_csv("data/credit.csv", parse_dates=["date"])
reg   = pd.read_csv("data/reg_pressure.csv", parse_dates=["date"])

S = money_entropy(money, q, k=float(cfg.get("k", 1.0)))
T = liquidity_temperature(cred)

df = cred.merge(S, on="date").merge(T, on="date").merge(reg, on="date")
df = df.sort_values("date").reset_index(drop=True)

la = LoopArea(lam=lam)
areas = []
for _, r in df.iterrows():
    areas.append(la.update(r["p_R"], r["V_R"]))
df["loop_area"] = areas

df["X_C"] = df["U"].astype(float) - T0 * df["S_M"].astype(float)

df.to_csv("site/indicators.csv", index=False)
print("Wrote site/indicators.csv")
