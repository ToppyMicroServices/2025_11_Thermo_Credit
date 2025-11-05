# scripts/01_build_features.py
import os, json, time, math, datetime as dt
import pandas as pd
import requests

FRED_KEY = os.getenv("FRED_API_KEY", "")
WB_BASE  = "https://api.worldbank.org/v2"

if not FRED_KEY:
    print("No FRED_API_KEY; skip online fetch and keep local CSVs.")
    raise SystemExit(0)
    
def fred_series(series_id, start="1990-01-01"):
    url = ("https://api.stlouisfed.org/fred/series/observations"
           f"?series_id={series_id}&api_key={FRED_KEY}&file_type=json&observation_start={start}")
    r = requests.get(url, timeout=30); r.raise_for_status()
    obs = r.json()["observations"]
    df = pd.DataFrame(obs)[["date","value"]]
    df["value"] = pd.to_numeric(df["value"], errors="coerce")
    return df.dropna()

def worldbank_series(country="JPN", indicator="NY.GDP.MKTP.CN"):  # 名目GDP（円）
    url = f"{WB_BASE}/country/{country}/indicator/{indicator}?format=json&per_page=20000"
    r = requests.get(url, timeout=30); r.raise_for_status()
    data = r.json()[1]
    df = pd.DataFrame(data)[["date","value"]]
    df["date"] = pd.to_datetime(df["date"]) + pd.offsets.QuarterEnd(0)
    df["value"] = pd.to_numeric(df["value"], errors="coerce")
    return df.dropna().sort_values("date")

# --- 1) money.csv（M_in=M2, M_out=マネタリーベースの素朴割当） ---
m2 = fred_series("MYAGM2JPM189S")       # M2（日本）月次  ※FRED再配信
base = fred_series("JPNASSETS")         # （便宜）日銀総資産をベースの代理に
money = (pd.merge(pd.to_datetime(m2["date"]).to_frame(name="date"), m2["value"].rename("M_in"), left_index=True, right_index=True)
          .merge(base.rename(columns={"date":"_d","value":"_base"}), left_on="date", right_on="_d", how="left")
          .drop(columns=["_d"]).rename(columns={"_base":"M_out"}))
money.to_csv("data/money.csv", index=False)

# --- 2) credit.csv（クレジット＋GDP＋金利・株式近似） ---
credit = fred_series("CRDQJPAPABIS")    # BIS由来：民間非金融向け信用（日本, 四半期/ドル建等）
gdp = worldbank_series("JPN", "NY.GDP.MKTP.CN")  # 名目GDP（円）
# 粗い近似：spread = 10年JGB利回り（月次）→四半期平均へ
jgb10 = fred_series("DGS10")  # 代用：必要ならECB/日銀に差し替え
jgb10["date"] = pd.to_datetime(jgb10["date"])
jgb10q = jgb10.resample("Q", on="date").mean().reset_index().rename(columns={"value":"spread"})
# turnover 近似：Nikkei225 出来高で代理（AlphaVantage等に差替え可）。ここではダミー0.9。
credit["date"] = pd.to_datetime(credit["date"])
credit_q = (credit.merge(gdp, on="date", how="left")
                  .merge(jgb10q[["date","spread"]], on="date", how="left"))
credit_q = credit_q.rename(columns={"value_x":"L_real", "value_y":"Y"})
credit_q["L_asset"] = credit_q["L_real"] * 0.4  # 暫定比率（後で実データに置換）
credit_q["U"] = credit_q["Y"]                   # モデル都合のポテンシャル代理
credit_q["depth"] = 1000                        # 近似・後で置換
credit_q["turnover"] = 1.0                      # 近似・後で置換
credit_q = credit_q[["date","L_real","L_asset","U","Y","spread","depth","turnover"]]
credit_q.to_csv("data/credit.csv", index=False)

# --- 3) reg_pressure.csv（政策金利＋中銀総資産） ---
policy = fred_series("DFF")  # ひとまずFF金利→後で日銀コール翌日物に置換（公式XLSXあり）
policy = policy.rename(columns={"value":"p_R"}); policy["date"] = pd.to_datetime(policy["date"])
assets = fred_series("JPNASSETS").rename(columns={"value":"V_R"}); assets["date"] = pd.to_datetime(assets["date"])
reg = (policy.merge(assets, on="date", how="outer")
             .sort_values("date").dropna(subset=["p_R","V_R"]))
reg.to_csv("data/reg_pressure.csv", index=False)

# --- 4) allocation_q.csv（初期はフラットでOK：後で半自動化に差し替え） ---
dates = credit_q["date"].drop_duplicates().sort_values()
q = pd.DataFrame({"date": dates})
q["q_pay"]=0.30; q["q_firm"]=0.30; q["q_asset"]=0.25; q["q_reserve"]=0.15
q.to_csv("data/allocation_q.csv", index=False)

print("API fetch -> CSV 書き出し完了")
