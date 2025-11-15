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
from lib.worldbank import fetch_worldbank_series
from lib.credit_enrichment import compute_enrichment
from lib.config_params import allocation_weights, leverage_share
from lib.config_loader import load_config
from lib.external_coupling import build_external_coupling_indices

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


def worldbank_series(country: str = "USA", indicator: str = "NY.GDP.MKTP.CN") -> pd.DataFrame:
    return fetch_worldbank_series(country, indicator, fallback_csvs=[
        os.path.join("data", "gdp_us.csv"),
        os.path.join("data", "worldbank_gdp_us.csv"),
    ])


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


def _external_series_fetcher(series_id: str, start: Optional[str] = None) -> pd.DataFrame:
    df = fred_series(series_id, start or DEFAULT_START)
    df = df.copy()
    df["date"] = pd.to_datetime(df["date"])
    os.makedirs("data", exist_ok=True)
    df.to_csv(os.path.join("data", f"{series_id}.csv"), index=False)
    return df


def _attach_headrooms(reg: pd.DataFrame) -> pd.DataFrame:
    df = reg.copy()
    if "V_R" not in df.columns:
        return df
    base = pd.to_numeric(df["V_R"], errors="coerce")
    pressure = pd.to_numeric(df.get("p_R"), errors="coerce").fillna(0).clip(lower=0)
    df["capital_headroom"] = (base * (1 - 0.04 * pressure)).clip(lower=0)
    df["lcr_headroom"] = (base * (1 - 0.05 * pressure)).clip(lower=0)
    df["nsfr_headroom"] = (base * (1 - 0.06 * pressure)).clip(lower=0)
    return df


def build_us(series_prefs: dict, project_config: dict) -> None:
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
    bis_q = bis.resample("QE-DEC", on="date").mean().reset_index()

    gdp = worldbank_series("USA", "NY.GDP.MKTP.CN").rename(columns={"value": "Y"})
    yld = yield_choice.get("data").copy()
    yld["date"] = pd.to_datetime(yld["date"])
    yq = yld.resample("QE-DEC", on="date").mean().reset_index().rename(columns={"value": "spread"})

    cred = (
        bis_q.merge(gdp, on="date", how="left")
        .merge(yq[["date", "spread"]], on="date", how="left")
        .sort_values("date")
    )
    leverage_ratio = leverage_share(project_config, "us", 0.4)
    cred["L_asset"] = cred["L_real"].astype(float) * leverage_ratio
    cred["U"] = cred["Y"].astype(float)
    warnings: list[str] = []
    enrich_cfg = project_config.get("enrichment", {}) if isinstance(project_config, dict) else {}
    cred = compute_enrichment(
        cred,
        depth_source=None,
        turnover_source=None,
        warnings=warnings,
        depth_scale=enrich_cfg.get("depth_scale"),
        depth_fallback=enrich_cfg.get("depth_fallback"),
        turnover_min=enrich_cfg.get("turnover_min"),
        turnover_max=enrich_cfg.get("turnover_max"),
        turnover_fallback=enrich_cfg.get("turnover_fallback"),
        clip_warn_threshold=enrich_cfg.get("turnover_clip_warn_threshold"),
    )
    for w in warnings:
        print(f"[US enrichment] WARNING: {w}")
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
    reg = _attach_headrooms(reg)
    ext_cfg = project_config.get("external_coupling", {}) if isinstance(project_config, dict) else {}
    ext_df = build_external_coupling_indices(ext_cfg, _external_series_fetcher)
    if not ext_df.empty:
        ext_path = os.path.join("data", "external_coupling_us.csv")
        ext_df.to_csv(ext_path, index=False)
        reg = reg.merge(ext_df, on="date", how="left")
    reg.to_csv("data/reg_pressure_us.csv", index=False)

    # allocation_q_us.csv (flat weights for now)
    alloc_path = "data/allocation_q_us.csv"
    if not os.path.exists(alloc_path):
        dates = cred["date"].drop_duplicates().sort_values()
        qdf = pd.DataFrame({"date": dates})
        default_weights = {
            "q_households": 0.30,
            "q_corporates": 0.35,
            "q_public": 0.20,
            "q_rest": 0.15,
        }
        region_weights = allocation_weights(project_config, "us", default_weights)
        for col, value in region_weights.items():
            qdf[col] = float(value)
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
    project_config = load_config("us")

    if args.list_series:
        list_series(series_prefs, args.role)
        return

    if not FRED_KEY:
        print("No FRED_API_KEY; skip US online fetch and keep local CSVs.")
        return

    build_us(series_prefs, project_config)


if __name__ == "__main__":
    main()