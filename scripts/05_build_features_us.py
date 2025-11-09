import os
import json
import time
import argparse
from typing import Optional

import pandas as pd
import requests
import sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from lib.series_selector import (
    DEFAULT_SERIES,
    DEFAULT_START,
    candidate_queue,
    load_series_preferences,
    select_series,
)

FRED_KEY = os.getenv("FRED_API_KEY", "")
CONFIG_PATH = os.path.join(ROOT, "config.yml")

ROLE_ENV_US = {
    "money_scale_us": "MONEY_SERIES_US",
    "base_proxy_us": "BASE_SERIES_US",
    "yield_proxy_us": "YIELD_SERIES_US",
}


def fred_series(series_id: str, start: str = DEFAULT_START, retries: int = 3, backoff: float = 1.5) -> pd.DataFrame:
    if not FRED_KEY:
        raise RuntimeError("FRED_API_KEY not set; cannot fetch online.")
    url = (
        "https://api.stlouisfed.org/fred/series/observations"
        f"?series_id={series_id}&api_key={FRED_KEY}&file_type=json&observation_start={start}"
    )
    last = None
    for i in range(retries):
        try:
            r = requests.get(url, timeout=30)
            r.raise_for_status()
            obs = r.json()["observations"]
            df = pd.DataFrame(obs)[["date", "value"]]
            df["value"] = pd.to_numeric(df["value"], errors="coerce")
            df = df.dropna()
            return df
        except Exception as e:
            last = e
            if i < retries - 1:
                time.sleep(backoff ** i)
            else:
                raise


def worldbank_series(country: str = "USA", indicator: str = "NY.GDP.MKTP.CN", retries: int = 4, backoff: float = 2.0) -> pd.DataFrame:
    """Fetch World Bank indicator with cache + local CSV fallback for the US."""
    cache_name = f"worldbank_cache_{country}_{indicator}.json".replace("/", "_")
    cache_path = os.path.join("data", cache_name)

    if os.path.exists(cache_path):
        try:
            payload = json.load(open(cache_path, "r", encoding="utf-8"))
            df = pd.DataFrame(payload)[["date", "value"]]
            df["date"] = pd.to_datetime(df["date"]) + pd.offsets.QuarterEnd(0)
            df["value"] = pd.to_numeric(df["value"], errors="coerce")
            out = df.dropna().sort_values("date")
            if not out.empty:
                return out
        except Exception:
            pass

    base = "https://api.worldbank.org/v2"
    url = f"{base}/country/{country}/indicator/{indicator}?format=json&per_page=20000"
    headers = {"User-Agent": "TQTC-Research/1.0 (+https://toppymicros.com)"}
    last_exc = None
    for i in range(retries):
        try:
            time.sleep(0.05 + 0.2 * (i / max(1, retries - 1)))
            r = requests.get(url, timeout=45, headers=headers)
            r.raise_for_status()
            payload = r.json()
            data = payload[1]
            df = pd.DataFrame(data)[["date", "value"]]
            try:
                os.makedirs("data", exist_ok=True)
                json.dump(df.to_dict(orient="records"), open(cache_path, "w", encoding="utf-8"))
            except Exception:
                pass
            df["date"] = pd.to_datetime(df["date"]) + pd.offsets.QuarterEnd(0)
            df["value"] = pd.to_numeric(df["value"], errors="coerce")
            return df.dropna().sort_values("date")
        except Exception as exc:
            last_exc = exc
            if i < retries - 1:
                time.sleep(backoff ** i)
            else:
                break

    for local in [os.path.join("data", "gdp_us.csv"), os.path.join("data", "worldbank_gdp_us.csv")]:
        if os.path.exists(local):
            try:
                df = pd.read_csv(local)
                dcol = next((c for c in df.columns if str(c).lower() == "date"), None)
                vcol = next((c for c in df.columns if str(c).lower() == "value"), None)
                if not dcol or not vcol:
                    continue
                df = df[[dcol, vcol]].rename(columns={dcol: "date", vcol: "value"})
                df["date"] = pd.to_datetime(df["date"]) + pd.offsets.QuarterEnd(0)
                df["value"] = pd.to_numeric(df["value"], errors="coerce")
                out = df.dropna().sort_values("date")
                if not out.empty:
                    return out
            except Exception:
                continue

    if last_exc:
        raise last_exc
    raise RuntimeError("World Bank GDP fetch and fallbacks failed for US")


def _log_selection(role: str, info: dict) -> None:
    title = info.get("title") or ""
    start = info.get("start") or DEFAULT_START
    source = info.get("source") or "default"
    suffix = f" - {title}" if title else ""
    print(f"[US series] {role}: {info['id']}{suffix} (source={source}, start={start})")


def list_series(series_prefs: dict, roles: Optional[list] = None) -> None:
    roles_to_show = roles or ["money_scale_us", "base_proxy_us", "yield_proxy_us"]
    for role in roles_to_show:
        env_var = ROLE_ENV_US.get(role)
        queue = candidate_queue(role, env_var, series_prefs, DEFAULT_SERIES)
        if not queue:
            print(f"[US series] {role}: (no candidates)")
            continue
        print(f"[US series] {role} candidates:")
        for item in queue:
            start = item.get("start") or DEFAULT_START
            title = item.get("title")
            note = item.get("note")
            suffix = f" - {title}" if title else ""
            suffix += f" ({note})" if note else ""
            print(f"  - {item['id']}{suffix} [{item['source']}, start={start}]")


def build_us(series_prefs: dict) -> None:
    money_choice = select_series(
        "money_scale_us",
        ROLE_ENV_US.get("money_scale_us"),
        fred_series,
        preferences=series_prefs,
        defaults=DEFAULT_SERIES,
    )
    base_choice = select_series(
        "base_proxy_us",
        ROLE_ENV_US.get("base_proxy_us"),
        fred_series,
        preferences=series_prefs,
        defaults=DEFAULT_SERIES,
    )
    yield_choice = select_series(
        "yield_proxy_us",
        ROLE_ENV_US.get("yield_proxy_us"),
        fred_series,
        preferences=series_prefs,
        defaults=DEFAULT_SERIES,
    )

    for role, info in (
        ("money_scale_us", money_choice),
        ("base_proxy_us", base_choice),
        ("yield_proxy_us", yield_choice),
    ):
        _log_selection(role, info)

    os.makedirs("data", exist_ok=True)
    selected_meta = {
        "money_scale_us": {k: money_choice.get(k) for k in ("id", "source", "start", "title")},
        "base_proxy_us": {k: base_choice.get(k) for k in ("id", "source", "start", "title")},
        "yield_proxy_us": {k: yield_choice.get(k) for k in ("id", "source", "start", "title")},
    }
    with open("data/series_selected_us.json", "w", encoding="utf-8") as fp:
        json.dump(selected_meta, fp, ensure_ascii=False, indent=2)

    for info in (money_choice, base_choice, yield_choice):
        sid = info.get("id")
        df_raw = info.get("data")
        if sid and isinstance(df_raw, pd.DataFrame):
            df_raw.to_csv(os.path.join("data", f"{sid}.csv"), index=False)

    # money_us.csv (monthly)
    m_in = money_choice.get("data").copy()
    m_out = base_choice.get("data").copy()
    for df in (m_in, m_out):
        df["date"] = pd.to_datetime(df["date"])
    m_in = m_in.rename(columns={"value": "M_in"})
    m_out = m_out.rename(columns={"value": "M_out"})
    money = m_in.merge(m_out, on="date", how="outer").sort_values("date")
    money.to_csv("data/money_us.csv", index=False)

    # credit_us.csv (quarterly)
    try:
        bis = fred_series("CRDQUSAPABIS", start="1990-01-01")
    except Exception:
        # Fallback to total loans (weekly); resample later
        bis = fred_series("TOTLL", start="1990-01-01")
    bis["date"] = pd.to_datetime(bis["date"])
    bis = bis.rename(columns={"value": "L_real"})
    bis_q = bis.resample("Q-DEC", on="date").mean().reset_index()

    gdp = worldbank_series("USA", "NY.GDP.MKTP.CN").rename(columns={"value": "Y"})
    yld = yield_choice.get("data").copy()
    yld["date"] = pd.to_datetime(yld["date"])
    yq = yld.resample("Q-DEC", on="date").mean().reset_index().rename(columns={"value": "spread"})

    cred = (
        bis_q.merge(gdp, on="date", how="left")
        .merge(yq[["date", "spread"]], on="date", how="left")
        .sort_values("date")
    )
    cred["L_asset"] = cred["L_real"].astype(float) * 0.4
    cred["U"] = cred["Y"].astype(float)
    cred["depth"] = 1200
    cred["turnover"] = 1.0
    cred = cred[["date", "L_real", "L_asset", "U", "Y", "spread", "depth", "turnover"]]
    cred.to_csv("data/credit_us.csv", index=False)

    # reg_pressure_us.csv (monthly)
    try:
        policy = fred_series("FEDFUNDS", start="1990-01-01")
    except Exception:
        policy = fred_series("DFF", start="1990-01-01")
    policy["date"] = pd.to_datetime(policy["date"])
    policy = policy.rename(columns={"value": "p_R"})
    assets = m_out.rename(columns={"M_out": "V_R"})[["date", "V_R"]]
    reg = (
        policy.merge(assets, on="date", how="outer")
        .sort_values("date")
        .dropna(subset=["p_R", "V_R"], how="any")
    )
    reg.to_csv("data/reg_pressure_us.csv", index=False)

    # allocation_q_us.csv (flat weights for now)
    alloc_path = "data/allocation_q_us.csv"
    if not os.path.exists(alloc_path):
        dates = cred["date"].drop_duplicates().sort_values()
        qdf = pd.DataFrame({"date": dates})
        qdf["q_households"] = 0.30
        qdf["q_corporates"] = 0.35
        qdf["q_public"] = 0.20
        qdf["q_rest"] = 0.15
        qdf.to_csv(alloc_path, index=False)

    print("US feature CSVs built: money_us.csv, credit_us.csv, reg_pressure_us.csv (+ allocation_q_us.csv if new)")


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Fetch US series and build feature CSVs")
    ap.add_argument("--list-series", action="store_true", help="List US candidate series and exit")
    ap.add_argument("--role", action="append", help="Limit roles (money_scale_us, base_proxy_us, yield_proxy_us)")
    return ap.parse_args()


def main() -> None:
    args = parse_args()
    series_prefs = load_series_preferences(CONFIG_PATH)

    if args.list_series:
        list_series(series_prefs, args.role)
        return

    if not FRED_KEY:
        print("No FRED_API_KEY; skip US online fetch and keep local CSVs.")
        return

    build_us(series_prefs)


if __name__ == "__main__":
    main()