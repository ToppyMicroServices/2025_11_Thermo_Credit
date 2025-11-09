# scripts/01_build_features.py
import os, json, time, math, datetime as dt
import argparse
from typing import Optional
import pandas as pd
import requests
import sys
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)
# -----------------------------------
FRED_KEY = os.getenv("FRED_API_KEY", "")
WB_BASE  = "https://api.worldbank.org/v2"
CONFIG_PATH = os.path.join(ROOT, "config.yml")
ROLE_ENV = {
    "money_scale": "MONEY_SERIES",
    "base_proxy": "BASE_SERIES",
    "yield_proxy": "YIELD_SERIES",
}

from lib.series_selector import DEFAULT_SERIES, DEFAULT_START, candidate_queue, load_series_preferences, select_series
    
def fred_series(series_id, start="1990-01-01", retries: int = 3, backoff: float = 1.5):
    """Fetch a FRED series with simple retry/backoff."""
    url = ("https://api.stlouisfed.org/fred/series/observations"
           f"?series_id={series_id}&api_key={FRED_KEY}&file_type=json&observation_start={start}")
    last_exc = None
    for attempt in range(retries):
        try:
            r = requests.get(url, timeout=30); r.raise_for_status()
            obs = r.json()["observations"]
            df = pd.DataFrame(obs)[["date","value"]]
            df["value"] = pd.to_numeric(df["value"], errors="coerce")
            return df.dropna()
        except Exception as e:
            last_exc = e
            if attempt < retries - 1:
                time.sleep(backoff ** attempt)
            else:
                raise

def worldbank_series(country="JPN", indicator="NY.GDP.MKTP.CN"):  # 名目GDP（円）
    url = f"{WB_BASE}/country/{country}/indicator/{indicator}?format=json&per_page=20000"
    r = requests.get(url, timeout=30); r.raise_for_status()
    data = r.json()[1]
    df = pd.DataFrame(data)[["date","value"]]
    df["date"] = pd.to_datetime(df["date"]) + pd.offsets.QuarterEnd(0)
    df["value"] = pd.to_numeric(df["value"], errors="coerce")
    return df.dropna().sort_values("date")
def _log_selection(role: str, info: dict) -> None:
    """Emit a concise line describing which series was picked."""
    title = info.get("title") or ""
    start = info.get("start") or DEFAULT_START
    source = info.get("source") or "default"
    suffix = f" - {title}" if title else ""
    print(f"[series] {role}: {info['id']}{suffix} (source={source}, start={start})")


def build_features(series_prefs: dict) -> None:
    money_choice = select_series(
        "money_scale",
        ROLE_ENV.get("money_scale"),
        fred_series,
        preferences=series_prefs,
        defaults=DEFAULT_SERIES,
    )
    base_choice = select_series(
        "base_proxy",
        ROLE_ENV.get("base_proxy"),
        fred_series,
        preferences=series_prefs,
        defaults=DEFAULT_SERIES,
    )
    yield_choice = select_series(
        "yield_proxy",
        ROLE_ENV.get("yield_proxy"),
        fred_series,
        preferences=series_prefs,
        defaults=DEFAULT_SERIES,
    )

    for role, info in (
        ("money_scale", money_choice),
        ("base_proxy", base_choice),
        ("yield_proxy", yield_choice),
    ):
        _log_selection(role, info)

    selected_meta = {
        "money_scale": {k: money_choice.get(k) for k in ("id", "source", "start", "title")},
        "base_proxy": {k: base_choice.get(k) for k in ("id", "source", "start", "title")},
        "yield_proxy": {k: yield_choice.get(k) for k in ("id", "source", "start", "title")},
    }
    with open("data/series_selected.json", "w", encoding="utf-8") as fp:
        json.dump(selected_meta, fp, ensure_ascii=False, indent=2)

    # Save raw selected series to data/<id>.csv for provenance / raw plotting
    try:
        os.makedirs("data", exist_ok=True)
        for info in (money_choice, base_choice, yield_choice):
            sid = info.get("id")
            df_raw = info.get("data")
            if sid and isinstance(df_raw, pd.DataFrame):
                out_path = os.path.join("data", f"{sid}.csv")
                df_raw.to_csv(out_path, index=False)
    except Exception:
        pass

    # --- 1) money.csv（M_in=M2, M_out=マネタリーベースの素朴割当） ---
    m2 = money_choice["data"].copy()
    m2["date"] = pd.to_datetime(m2["date"])
    m2 = m2.rename(columns={"value": "M_in"})

    base = base_choice["data"].copy()
    base["date"] = pd.to_datetime(base["date"])
    base = base.rename(columns={"value": "M_out"})

    money = m2.merge(base, on="date", how="left")
    money.to_csv("data/money.csv", index=False)

    # --- 2) credit.csv（クレジット＋GDP＋金利・株式近似） ---
    credit = fred_series("CRDQJPAPABIS")    # BIS由来：民間非金融向け信用（日本, 四半期/ドル建等）
    gdp = worldbank_series("JPN", "NY.GDP.MKTP.CN")  # 名目GDP（円）

    jgb = yield_choice["data"].copy()
    jgb["date"] = pd.to_datetime(jgb["date"])
    jgbq = jgb.resample("Q", on="date").mean().reset_index().rename(columns={"value": "spread"})

    credit["date"] = pd.to_datetime(credit["date"])
    credit_q = (credit.merge(gdp, on="date", how="left")
                      .merge(jgbq[["date", "spread"]], on="date", how="left"))
    credit_q = credit_q.rename(columns={"value_x": "L_real", "value_y": "Y"})
    credit_q["L_asset"] = credit_q["L_real"] * 0.4  # 暫定比率（後で実データに置換）
    credit_q["U"] = credit_q["Y"]                   # モデル都合のポテンシャル代理
    credit_q["depth"] = 1000                        # 近似・後で置換
    credit_q["turnover"] = 1.0                      # 近似・後で置換
    credit_q = credit_q[["date", "L_real", "L_asset", "U", "Y", "spread", "depth", "turnover"]]
    credit_q.to_csv("data/credit.csv", index=False)

    # --- 3) reg_pressure.csv（政策金利＋中銀総資産） ---
    policy = fred_series("DFF")  # ひとまずFF金利→後で日銀コール翌日物に置換（公式XLSXあり）
    policy = policy.rename(columns={"value": "p_R"})
    policy["date"] = pd.to_datetime(policy["date"])

    assets = base.copy().rename(columns={"M_out": "V_R"})
    reg = (policy.merge(assets, on="date", how="outer")
                 .sort_values("date").dropna(subset=["p_R", "V_R"]))
    reg.to_csv("data/reg_pressure.csv", index=False)

    # --- 4) allocation_q.csv（初期はフラットでOK：後で半自動化に差し替え） ---
    dates = credit_q["date"].drop_duplicates().sort_values()
    q = pd.DataFrame({"date": dates})
    q["q_pay"] = 0.30
    q["q_firm"] = 0.30
    q["q_asset"] = 0.25
    q["q_reserve"] = 0.15
    q.to_csv("data/allocation_q.csv", index=False)

    print("API fetch -> CSV 書き出し完了")


def list_series(series_prefs: dict, roles: Optional[list] = None) -> None:
    roles_to_show = roles or sorted(set(list(DEFAULT_SERIES.keys()) + list(series_prefs.keys())))
    for role in roles_to_show:
        env_var = ROLE_ENV.get(role)
        queue = candidate_queue(role, env_var, series_prefs, DEFAULT_SERIES)
        if not queue:
            print(f"[series] {role}: (no candidates)")
            continue
        print(f"[series] {role} candidates:")
        for item in queue:
            start = item.get("start") or DEFAULT_START
            title = item.get("title")
            note = item.get("note")
            suffix = f" - {title}" if title else ""
            suffix += f" ({note})" if note else ""
            print(f"  - {item['id']}{suffix} [{item['source']}, start={start}]")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fetch series and build Thermo-Credit feature tables.")
    parser.add_argument(
        "--list-series",
        action="store_true",
        help="List candidate series (including env/config overrides) and exit.",
    )
    parser.add_argument(
        "--role",
        action="append",
        help="When used with --list-series, limit the output to a specific role (can repeat).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    series_prefs = load_series_preferences(CONFIG_PATH)

    if args.list_series:
        list_series(series_prefs, args.role)
        return

    if not FRED_KEY:
        print("No FRED_API_KEY; skip online fetch and keep local CSVs.")
        return

    build_features(series_prefs)


if __name__ == "__main__":
    main()
