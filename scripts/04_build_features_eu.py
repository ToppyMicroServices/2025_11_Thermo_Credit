import os, json, time
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
    load_project_config,
    load_series_preferences,
    select_series,
)
from lib.credit_enrichment import compute_enrichment
from lib.worldbank import fetch_worldbank_series
from lib.config_params import allocation_weights, leverage_share

FRED_KEY = os.getenv("FRED_API_KEY", "")
CONFIG_PATH = os.path.join(ROOT, "config.yml")

# Environment overrides for EU roles
ROLE_ENV_EU = {
    "money_scale_eu": "MONEY_SERIES_EU",
    "base_proxy_eu": "BASE_SERIES_EU",
    "yield_proxy_eu": "YIELD_SERIES_EU",
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

def worldbank_series(country: str = "EMU", indicator: str = "NY.GDP.MKTP.CN") -> pd.DataFrame:
    return fetch_worldbank_series(country, indicator, fallback_csvs=[
        os.path.join("data", "gdp_eu.csv"),
        os.path.join("data", "eurostat_gdp_eu.csv"),
    ])


def _log_selection(role: str, info: dict) -> None:
    title = info.get("title") or ""
    start = info.get("start") or DEFAULT_START
    source = info.get("source") or "default"
    suffix = f" - {title}" if title else ""
    print(f"[EU series] {role}: {info['id']}{suffix} (source={source}, start={start})")


def list_series(series_prefs: dict, roles: Optional[list] = None) -> None:
    roles_to_show = roles or ["money_scale_eu", "base_proxy_eu", "yield_proxy_eu"]
    for role in roles_to_show:
        env_var = ROLE_ENV_EU.get(role)
        queue = candidate_queue(role, env_var, series_prefs, DEFAULT_SERIES)
        if not queue:
            print(f"[EU series] {role}: (no candidates)")
            continue
        print(f"[EU series] {role} candidates:")
        for item in queue:
            start = item.get("start") or DEFAULT_START
            title = item.get("title")
            note = item.get("note")
            suffix = f" - {title}" if title else ""
            suffix += f" ({note})" if note else ""
            print(f"  - {item['id']}{suffix} [{item['source']}, start={start}]")


def build_eu(series_prefs: dict, project_config: dict) -> None:
    money_choice = select_series(
        "money_scale_eu",
        ROLE_ENV_EU.get("money_scale_eu"),
        fred_series,
        preferences=series_prefs,
        defaults=DEFAULT_SERIES,
    )
    base_choice = select_series(
        "base_proxy_eu",
        ROLE_ENV_EU.get("base_proxy_eu"),
        fred_series,
        preferences=series_prefs,
        defaults=DEFAULT_SERIES,
    )
    yield_choice = select_series(
        "yield_proxy_eu",
        ROLE_ENV_EU.get("yield_proxy_eu"),
        fred_series,
        preferences=series_prefs,
        defaults=DEFAULT_SERIES,
    )

    for role, info in (
        ("money_scale_eu", money_choice),
        ("base_proxy_eu", base_choice),
        ("yield_proxy_eu", yield_choice),
    ):
        _log_selection(role, info)

    os.makedirs("data", exist_ok=True)
    selected_meta = {
        "money_scale_eu": {k: money_choice.get(k) for k in ("id", "source", "start", "title")},
        "base_proxy_eu": {k: base_choice.get(k) for k in ("id", "source", "start", "title")},
        "yield_proxy_eu": {k: yield_choice.get(k) for k in ("id", "source", "start", "title")},
    }
    with open("data/series_selected_eu.json", "w", encoding="utf-8") as fp:
        json.dump(selected_meta, fp, ensure_ascii=False, indent=2)

    # Save raw CSVs for provenance / Raw Inputs plot
    for info in (money_choice, base_choice, yield_choice):
        sid = info.get("id")
        df_raw = info.get("data")
        if sid and isinstance(df_raw, pd.DataFrame):
            df_raw.to_csv(os.path.join("data", f"{sid}.csv"), index=False)

    print("EU fetch -> CSV write complete (series_selected_eu.json, raw series CSV)")

    # --- Build EU feature input CSVs from real series (replace placeholders) ---
    try:
        # money_eu.csv (monthly): M_in from money_scale_eu, M_out from base_proxy_eu
        m_in = money_choice.get("data").copy()
        m_out = base_choice.get("data").copy()
        for df in (m_in, m_out):
            df["date"] = pd.to_datetime(df["date"])
        m_in = m_in.rename(columns={"value": "M_in"})
        m_out = m_out.rename(columns={"value": "M_out"})
        money = m_in.merge(m_out, on="date", how="left").sort_values("date")
        money.to_csv(os.path.join("data", "money_eu.csv"), index=False)

        # credit_eu.csv (quarterly): BIS credit + GDP + yield spread
        try:
            bis = fred_series("CRDQEZAPABIS")  # BIS credit (private non-financial), Euro Area
        except Exception:
            # Fallback candidate (long-term series) - adjust if not available
            bis = fred_series("QUSN628BIS")  # Placeholder: will likely differ; user should replace
        bis["date"] = pd.to_datetime(bis["date"])  # quarterly or monthly depending on source
        bis["date"] = pd.to_datetime(bis["date"])  # quarterly
        bis = bis.rename(columns={"value": "L_real"})
        gdp = worldbank_series("EMU", "NY.GDP.MKTP.CN").rename(columns={"value": "Y"})
        yld = yield_choice.get("data").copy()
        yld["date"] = pd.to_datetime(yld["date"])  # monthly
        # Use explicit quarter ending alias (December) 'QE-DEC'
        yq = yld.resample("QE-DEC", on="date").mean().reset_index().rename(columns={"value": "spread"})
        cred = (bis.merge(gdp, on="date", how="left")
                    .merge(yq[["date", "spread"]], on="date", how="left")
                    .sort_values("date"))
        leverage_ratio = leverage_share(project_config, "eu", 0.4)
        cred["L_asset"] = cred["L_real"].astype(float) * leverage_ratio
        cred["U"] = cred["Y"].astype(float)  # proxy
        # Shared enrichment computation (no external depth/turnover quarterly series yet; heuristics apply)
        warnings: list[str] = []
        enrich_cfg = project_config.get("enrichment", {}) if isinstance(project_config, dict) else {}
        cred = compute_enrichment(
            cred,
            depth_source=None,
            turnover_source=None,
            warnings=warnings,
            depth_scale=enrich_cfg.get("depth_scale"),
            turnover_min=enrich_cfg.get("turnover_min"),
            turnover_max=enrich_cfg.get("turnover_max"),
            clip_warn_threshold=enrich_cfg.get("turnover_clip_warn_threshold"),
        )
        for w in warnings:
            print(f"[EU enrichment] WARNING: {w}")
        cred = cred[["date", "L_real", "L_asset", "U", "Y", "spread", "depth", "turnover"]]
        cred.to_csv(os.path.join("data", "credit_eu.csv"), index=False)

        # reg_pressure_eu.csv (monthly): policy proxy p_R and V_R
        pR_df = yld.rename(columns={"value": "p_R"})
        V_df = m_out.rename(columns={"M_out": "V_R"})[["date", "V_R"]]
        reg = (pR_df.merge(V_df, on="date", how="outer")
                      .sort_values("date").dropna(subset=["p_R", "V_R"], how="any"))
        reg.to_csv(os.path.join("data", "reg_pressure_eu.csv"), index=False)

        # allocation_q_eu.csv (only create if absent): EU-specific buckets
        alloc_path = os.path.join("data", "allocation_q_eu.csv")
        if not os.path.exists(alloc_path):
            dates = cred["date"].drop_duplicates().sort_values()
            qdf = pd.DataFrame({"date": dates})
            default_weights = {
                "q_households": 0.30,
                "q_corporates": 0.35,
                "q_government": 0.20,
                "q_row": 0.15,
            }
            region_weights = allocation_weights(project_config, "eu", default_weights)
            for col, value in region_weights.items():
                qdf[col] = float(value)
            qdf.to_csv(alloc_path, index=False)
        print("EU feature CSVs built: money_eu.csv, credit_eu.csv, reg_pressure_eu.csv (+ allocation_q_eu.csv if missing)")
    except Exception as e:
        print("[EU build] Skipped building EU feature CSVs:", e)


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Fetch EU series and write raw CSV + selection metadata.")
    ap.add_argument("--list-series", action="store_true", help="List EU candidates and exit")
    ap.add_argument("--role", action="append", help="Limit roles (money_scale_eu, base_proxy_eu, yield_proxy_eu)")
    return ap.parse_args()


def main() -> None:
    args = parse_args()
    series_prefs = load_series_preferences(CONFIG_PATH)
    project_config = load_project_config(CONFIG_PATH)

    if args.list_series:
        list_series(series_prefs, args.role)
        return

    if not FRED_KEY:
        print("No FRED_API_KEY; skip EU online fetch and keep local CSVs.")
        return

    build_eu(series_prefs, project_config)


if __name__ == "__main__":
    main()
