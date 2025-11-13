# scripts/01_build_features.py
import os, json, time, math, datetime as dt
import numpy as np
import numpy as np
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
    load_project_config,
    load_series_preferences,
    select_series,
    candidate_queue,
)
from lib.credit_enrichment import compute_enrichment
from lib.worldbank import fetch_worldbank_series
from lib.config_params import allocation_weights, leverage_share
# -----------------------------------
FRED_KEY = os.getenv("FRED_API_KEY", "")
WB_BASE  = "https://api.worldbank.org/v2"
CONFIG_PATH = os.path.join(ROOT, "config.yml")
ROLE_ENV = {
    "money_scale": "MONEY_SERIES",
    "base_proxy": "BASE_SERIES",
    "yield_proxy": "YIELD_SERIES",
    # enrichment roles currently no env override wiring
    "asset_proxy": None,
    "energy_proxy": None,
    "depth_proxy": None,
    "turnover_proxy": None,
}

def build_features(series_prefs: dict, project_config: dict) -> None:
    """Fetch source series and build feature tables.
    Real data replaces prior placeholders for L_asset, U, depth, turnover; toy baselines retained with *_toy suffix.
    """
    # Helper: some roles use World Bank indicator style IDs.
    def unified_fetch(series_id: str, start: str) -> pd.DataFrame:
        sid = series_id.upper()
        if sid.startswith("NY.GDP") or sid.startswith("NY.") or sid.startswith("GDPN"):
            return worldbank_series("JPN", "NY.GDP.MKTP.CN")
        return fred_series(series_id, start)

    money_choice = select_series("money_scale", ROLE_ENV.get("money_scale"), fred_series,
                                 preferences=series_prefs, defaults=DEFAULT_SERIES)
    base_choice = select_series("base_proxy", ROLE_ENV.get("base_proxy"), fred_series,
                                preferences=series_prefs, defaults=DEFAULT_SERIES)
    yield_choice = select_series("yield_proxy", ROLE_ENV.get("yield_proxy"), fred_series,
                                 preferences=series_prefs, defaults=DEFAULT_SERIES)
    asset_choice = select_series("asset_proxy", None, fred_series,
                                 preferences=series_prefs, defaults=DEFAULT_SERIES)
    energy_choice = select_series("energy_proxy", None, unified_fetch,
                                  preferences=series_prefs, defaults=DEFAULT_SERIES)
    depth_choice = select_series("depth_proxy", None, fred_series,
                                 preferences=series_prefs, defaults=DEFAULT_SERIES)
    turnover_choice = select_series("turnover_proxy", None, fred_series,
                                    preferences=series_prefs, defaults=DEFAULT_SERIES)

    for role, info in (
        ("money_scale", money_choice),
        ("base_proxy", base_choice),
        ("yield_proxy", yield_choice),
        ("asset_proxy", asset_choice),
        ("energy_proxy", energy_choice),
        ("depth_proxy", depth_choice),
        ("turnover_proxy", turnover_choice),
    ):
        _log_selection(role, info)

    selected_meta = {r: {k: info.get(k) for k in ("id", "source", "start", "title")}
                     for r, info in (
                         ("money_scale", money_choice),
                         ("base_proxy", base_choice),
                         ("yield_proxy", yield_choice),
                         ("asset_proxy", asset_choice),
                         ("energy_proxy", energy_choice),
                         ("depth_proxy", depth_choice),
                         ("turnover_proxy", turnover_choice),
                     )}
    os.makedirs("data", exist_ok=True)
    with open("data/series_selected.json", "w", encoding="utf-8") as fp:
        json.dump(selected_meta, fp, ensure_ascii=False, indent=2)

    # Persist raw series
    for info in (money_choice, base_choice, yield_choice, asset_choice, energy_choice, depth_choice, turnover_choice):
        sid = info.get("id"); raw = info.get("data")
        if sid and isinstance(raw, pd.DataFrame):
            raw.to_csv(os.path.join("data", f"{sid}.csv"), index=False)

    # 1) money.csv
    m2 = money_choice["data"].copy(); m2["date"] = pd.to_datetime(m2["date"]); m2 = m2.rename(columns={"value": "M_in"})
    base = base_choice["data"].copy(); base["date"] = pd.to_datetime(base["date"]); base = base.rename(columns={"value": "M_out"})
    money = m2.merge(base, on="date", how="left")
    money.to_csv("data/money.csv", index=False)

    # 2) credit.csv
    credit = fred_series("CRDQJPAPABIS"); credit["date"] = pd.to_datetime(credit["date"])  # real private credit
    gdp = worldbank_series("JPN", "NY.GDP.MKTP.CN")
    jgb = yield_choice["data"].copy(); jgb["date"] = pd.to_datetime(jgb["date"])
    # Use explicit quarter ending convention (December) with new alias 'QE-DEC'
    jgbq = jgb.resample("QE-DEC", on="date").mean().reset_index().rename(columns={"value": "spread"})
    credit_q = (credit.merge(gdp, on="date", how="left")
                      .merge(jgbq[["date", "spread"]], on="date", how="left")
                      .rename(columns={"value_x": "L_real", "value_y": "Y"}))

    def _to_q(df: pd.DataFrame, how: str, col: str) -> pd.DataFrame:
        df = df.copy(); df["date"] = pd.to_datetime(df["date"])
        if how == "mean":
            out = df.resample("QE-DEC", on="date").mean().reset_index()
        else:
            out = df.resample("QE-DEC", on="date").last().reset_index()
        return out.rename(columns={"value": col})

    asset_q = _to_q(asset_choice["data"], "last", "L_asset_real")
    energy_q = _to_q(energy_choice["data"], "mean", "U_energy")
    depth_q = _to_q(depth_choice["data"], "last", "depth_real")
    turnover_q = _to_q(turnover_choice["data"], "mean", "turnover_real")

    credit_q = (credit_q.merge(asset_q, on="date", how="left")
                        .merge(energy_q, on="date", how="left")
                        .merge(depth_q, on="date", how="left")
                        .merge(turnover_q, on="date", how="left"))

    leverage_ratio = leverage_share(project_config, "jp", 0.4)
    credit_q["L_asset"] = credit_q["L_asset_real"].fillna(credit_q["L_real"] * leverage_ratio)
    credit_q["U"] = credit_q["U_energy"].fillna(credit_q["Y"])
    # Shared enrichment (merge real depth/turnover if present; else heuristic)
    warnings: list[str] = []
    # Provide depth/turnover sources if present as Series aligned by date
    depth_src = credit_q["depth_real"] if "depth_real" in credit_q.columns else None
    turnover_src = credit_q["turnover_real"] if "turnover_real" in credit_q.columns else None
    # If depth_src/turnover_src are entirely NaN or missing, pass None to trigger heuristic fallback
    if depth_src is not None and depth_src.notna().sum() == 0:
        depth_src = None
    if turnover_src is not None and turnover_src.notna().sum() == 0:
        turnover_src = None
    enrich_cfg = project_config.get("enrichment", {}) if isinstance(project_config, dict) else {}
    enriched = compute_enrichment(
        credit_q[["date", "L_real", "L_asset", "U", "Y", "spread"]],
        depth_source=depth_src,
        turnover_source=turnover_src,
        warnings=warnings,
        depth_scale=enrich_cfg.get("depth_scale"),
        turnover_min=enrich_cfg.get("turnover_min"),
        turnover_max=enrich_cfg.get("turnover_max"),
        clip_warn_threshold=enrich_cfg.get("turnover_clip_warn_threshold"),
    )
    for w in warnings:
        print(f"[JP enrichment] WARNING: {w}")
    credit_q = credit_q.merge(enriched[["date", "depth", "turnover"]], on="date", how="left")
    # Toy baselines retained for regression protection
    credit_q["L_asset_toy"] = credit_q["L_real"] * leverage_ratio
    credit_q["U_gdp_only"] = credit_q["Y"]
    credit_q["depth_toy"] = 1000
    credit_q["turnover_toy"] = 1.0
    credit_q = credit_q[["date", "L_real", "L_asset", "U", "Y", "spread", "depth", "turnover",
                         "L_asset_toy", "U_gdp_only", "depth_toy", "turnover_toy"]]
    credit_q.to_csv("data/credit.csv", index=False)

    policy = fred_series("DFF"); policy = policy.rename(columns={"value": "p_R"}); policy["date"] = pd.to_datetime(policy["date"])
    assets = base.copy().rename(columns={"M_out": "V_R"})
    reg = (policy.merge(assets, on="date", how="outer")
                 .sort_values("date").dropna(subset=["p_R", "V_R"]))
    reg.to_csv("data/reg_pressure.csv", index=False)

    dates = credit_q["date"].drop_duplicates().sort_values()
    q = pd.DataFrame({"date": dates})
    default_weights = {"q_pay": 0.30, "q_firm": 0.30, "q_asset": 0.25, "q_reserve": 0.15}
    region_weights = allocation_weights(project_config, "jp", default_weights)
    for col, value in region_weights.items():
        q[col] = float(value)
    housing_ratio = float(os.getenv("JP_Q_HOUSING_SHARE", 0.4)); housing_ratio = min(max(housing_ratio, 0.0), 1.0)
    q["q_productive"] = q["q_firm"]; q["q_financial"] = q["q_asset"]; q["q_government"] = q["q_reserve"]
    q["q_housing"] = q["q_pay"] * housing_ratio
    q["q_consumption"] = (q["q_pay"] - q["q_housing"]).clip(lower=0)
    mece_cols = ["q_productive", "q_housing", "q_consumption", "q_financial", "q_government"]
    total = q[mece_cols].sum(axis=1).replace({0: np.nan})
    q.loc[total.notna(), mece_cols] = (q.loc[total.notna(), mece_cols].div(total[total.notna()], axis=0))
    ordered_cols = ["date", "q_pay", "q_firm", "q_asset", "q_reserve", *mece_cols]
    q[ordered_cols].to_csv("data/allocation_q.csv", index=False)

    print("API fetch -> CSV write complete")

# --- helpers (JP) ---
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

def worldbank_series(country: str = "JPN", indicator: str = "NY.GDP.MKTP.CN") -> pd.DataFrame:
    return fetch_worldbank_series(country, indicator)

def _log_selection(role: str, info: dict) -> None:
    title = info.get("title") or ""; start = info.get("start") or DEFAULT_START
    source = info.get("source") or "default"; suffix = f" - {title}" if title else ""
    print(f"[JP series] {role}: {info['id']}{suffix} (source={source}, start={start})")


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
            title = item.get("title"); note = item.get("note")
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
    project_config = load_project_config(CONFIG_PATH)

    if args.list_series:
        list_series(series_prefs, args.role)
        return

    if not FRED_KEY:
        print("No FRED_API_KEY; skip online fetch and keep local CSVs.")
        return

    build_features(series_prefs, project_config)


if __name__ == "__main__":
    main()
