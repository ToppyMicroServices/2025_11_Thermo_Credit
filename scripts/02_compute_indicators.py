import os, sys
import numpy as np
import pandas as pd

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)
from lib.indicators import build_indicators_core, compute_diagnostics, DEFAULT_HEADROOM_COLS
from lib.config_loader import load_config

DEFAULT_REGIONS = ("jp", "us", "eu")
MULTI_REGION_TOKENS = {"all", "*", "multi", "all_regions"}


def _output_path_for_region(region: str) -> str:
    region = region.strip().lower()
    return "site/indicators.csv" if region == "jp" else f"site/indicators_{region}.csv"


def _bootstrap_region_env() -> str:
    raw = os.getenv("REGION", "").strip().lower()
    if not raw or raw in MULTI_REGION_TOKENS or "," in raw:
        return "jp"
    return raw


REGION = _bootstrap_region_env()

HEADROOM_DECAY = dict(zip(DEFAULT_HEADROOM_COLS, (0.04, 0.05, 0.06)))


def _ensure_headrooms(reg: pd.DataFrame) -> pd.DataFrame:
    if reg is None or reg.empty:
        return reg
    df = reg.copy()
    base_col = "V_R" if "V_R" in df.columns else "V_C" if "V_C" in df.columns else None
    if base_col is None or "p_R" not in df.columns:
        return df
    base = pd.to_numeric(df[base_col], errors="coerce")
    pressure = pd.to_numeric(df.get("p_R"), errors="coerce").fillna(0).clip(lower=0)
    for col, coeff in HEADROOM_DECAY.items():
        if col in df.columns:
            continue
        df[col] = (base * (1 - coeff * pressure)).clip(lower=0)
    return df


def _ensure_minimal_inputs() -> None:
    """Ensure minimal CSV inputs exist for JP runs when raw sources are absent.

    This mirrors the CI helper but is deliberately tiny and only kicks in
    when files are completely missing, so it won't overwrite any real data.
    """
    data_dir = os.path.join(ROOT, "data")
    os.makedirs(data_dir, exist_ok=True)

    def _write_if_missing(path: str, header, rows) -> None:
        if os.path.exists(path):
            return
        import csv

        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(header)
            w.writerows(rows)

    # Minimal money series
    _write_if_missing(
        os.path.join(data_dir, "money.csv"),
        ["date", "M_in", "M_out"],
        [
            ["2023-01-01", 100.0, 80.0],
            ["2023-04-01", 110.0, 82.0],
        ],
    )

    # Minimal allocation_q (MECE) – keep consistent with config.yml categories
    _write_if_missing(
        os.path.join(data_dir, "allocation_q.csv"),
        [
            "date",
            "q_pay",
            "q_firm",
            "q_asset",
            "q_reserve",
            "q_productive",
            "q_housing",
            "q_consumption",
            "q_financial",
            "q_government",
        ],
        [
            ["2023-01-01", 0.30, 0.30, 0.25, 0.15, 0.30, 0.12, 0.18, 0.25, 0.15],
        ],
    )

    # Minimal credit and regulatory pressure needed by enrichment
    _write_if_missing(
        os.path.join(data_dir, "credit.csv"),
        [
            "date",
            "L_real",
            "L_asset",
            "U",
            "Y",
            "spread",
            "depth",
            "turnover",
            "L_asset_toy",
            "U_gdp_only",
            "depth_toy",
            "turnover_toy",
        ],
        [
            [
                "2023-01-01",
                1000,
                400,
                500,
                500,
                0.5,
                1200,
                1.2,
                400,
                500,
                1000,
                1.0,
            ],
        ],
    )

    _write_if_missing(
        os.path.join(data_dir, "reg_pressure.csv"),
        ["date", "p_R", "V_R"],
        [["2023-01-01", 0.5, 80.0]],
    )


def _prepare_yield_fallback(yield_df: pd.DataFrame) -> pd.DataFrame:
    if yield_df is None or yield_df.empty:
        return pd.DataFrame()
    try:
        y = yield_df.copy()
        y["date"] = pd.to_datetime(y["date"])
    except Exception:
        return pd.DataFrame()
    value_cols = [c for c in y.columns if c != "date"]
    if not value_cols:
        return pd.DataFrame()
    col = value_cols[0]
    try:
        y[col] = pd.to_numeric(y[col], errors="coerce")
    except Exception:
        pass
    y["quarter"] = y["date"].dt.to_period("Q-DEC")
    out = (
        y.groupby("quarter")[col]
        .mean()
        .reset_index()
        .rename(columns={col: "spread_fallback"})
    )
    return out


def _ensure_credit_inputs(cred: pd.DataFrame, yield_df: pd.DataFrame) -> pd.DataFrame:
    if cred is None or cred.empty:
        return cred
    df = cred.copy()
    try:
        df["date"] = pd.to_datetime(df["date"])
    except Exception:
        pass

    if "spread" not in df.columns:
        df["spread"] = np.nan
    df["spread"] = pd.to_numeric(df["spread"], errors="coerce")
    df["quarter"] = df["date"].dt.to_period("Q-DEC")
    y_fallback = _prepare_yield_fallback(yield_df)
    if not y_fallback.empty:
        df = df.merge(y_fallback, on="quarter", how="left")
        df["spread"] = df["spread"].combine_first(df["spread_fallback"])
        df = df.drop(columns=["spread_fallback"])

    if "U" not in df.columns:
        df["U"] = np.nan
    u_series = pd.to_numeric(df["U"], errors="coerce")
    for fallback in ("U_gdp_only", "Y", "L_real"):
        if fallback in df.columns:
            u_series = u_series.combine_first(pd.to_numeric(df[fallback], errors="coerce"))
    df["U"] = u_series

    if "Y" not in df.columns:
        df["Y"] = np.nan
    y_series = pd.to_numeric(df["Y"], errors="coerce")
    for fallback in ("U", "U_gdp_only", "L_real"):
        if fallback in df.columns:
            y_series = y_series.combine_first(pd.to_numeric(df[fallback], errors="coerce"))
    df["Y"] = y_series

    return df.drop(columns=["quarter"], errors="ignore")


def _tokenize_regions(values) -> list[str]:
    tokens: list[str] = []
    if not values:
        return tokens
    if isinstance(values, str):
        chunks = values.split(",")
    else:
        chunks = []
        for item in values:
            chunks.extend(str(item).split(","))
    for chunk in chunks:
        name = chunk.strip().lower()
        if name:
            tokens.append(name)
    return tokens


def _consume_regions(preferred: list[str]) -> list[str]:
    if not preferred:
        return list(DEFAULT_REGIONS)
    if any(tok in MULTI_REGION_TOKENS for tok in preferred):
        return list(DEFAULT_REGIONS)
    deduped: list[str] = []
    for token in preferred:
        if token not in DEFAULT_REGIONS:
            continue
        if token not in deduped:
            deduped.append(token)
    return deduped or list(DEFAULT_REGIONS)


def _resolve_requested_regions(argv: list[str]) -> list[str]:
    cli_tokens = _tokenize_regions(argv)
    if cli_tokens:
        return _consume_regions(cli_tokens)
    env_value = os.getenv("REGION", "").strip().lower()
    env_tokens = _tokenize_regions(env_value)
    return _consume_regions(env_tokens)

cfg = load_config(REGION)

# When running in JP mode, ensure minimal inputs exist so CI or local runs
# that don't have raw data available don't fail with FileNotFoundError.
if REGION == "jp":
    _ensure_minimal_inputs()

# Build JP money from raw series when available
def _read_raw(sid: str) -> pd.DataFrame:
    path = os.path.join("data", f"{sid}.csv")
    if not os.path.exists(path):
        return pd.DataFrame()
    df = pd.read_csv(path, parse_dates=["date"]).dropna()
    return df.sort_values("date")

def _pick_first_available(series_list):
    """Given a list of dicts with id fields, return first DataFrame found under data/<id>.csv."""
    if not series_list:
        return pd.DataFrame()
    for s in series_list:
        sid = s.get("id") if isinstance(s, dict) else None
        if not sid:
            continue
        df = _read_raw(str(sid))
        if not df.empty:
            return df
    return pd.DataFrame()

# Resolve region-specific series preferences from config if available
series_cfg = cfg.get("series", {}) if isinstance(cfg, dict) else {}
if REGION == "eu":
    ms_pref = (series_cfg.get("money_scale_eu", {}) or {}).get("preferred")
    base_pref = series_cfg.get("base_proxy_eu")
    y_pref = series_cfg.get("yield_proxy_eu")
elif REGION == "us":
    ms_pref = (series_cfg.get("money_scale_us", {}) or {}).get("preferred")
    base_pref = series_cfg.get("base_proxy_us")
    y_pref = series_cfg.get("yield_proxy_us")
else:
    ms_pref = (series_cfg.get("money_scale", {}) or {}).get("preferred")
    base_pref = series_cfg.get("base_proxy")
    y_pref = series_cfg.get("yield_proxy")

boj = _pick_first_available(base_pref)
m2  = _pick_first_available(ms_pref)
yld = _pick_first_available(y_pref)

# Allow environment to define earliest JP date (e.g. JP_START=2012-01-01)
def _apply_start(df: pd.DataFrame, start_ts: pd.Timestamp) -> pd.DataFrame:
    if df.empty:
        return df
    return df[df["date"] >= start_ts].copy()

# JP_START applies only to JP region. If not provided, do not trim.
_JP_START_ENV = os.getenv("JP_START", "").strip()
if REGION == "jp" and _JP_START_ENV:
    try:
        jp_start_ts = pd.Timestamp(_JP_START_ENV)
        boj = _apply_start(boj, jp_start_ts)
        m2  = _apply_start(m2, jp_start_ts)
        yld = _apply_start(yld, jp_start_ts)
        print(f"[info] Applied JP_START={jp_start_ts.date()} to raw JP series")
    except Exception as e:
        print(f"[warn] Could not apply JP_START ({_JP_START_ENV}): {e}")

def _qe_dec(dfm: pd.DataFrame) -> pd.DataFrame:
    if dfm.empty:
        return dfm
    dfm = dfm.set_index("date").resample("QE-DEC").last().reset_index()
    return dfm

# Money scale from M2 if present, else base proxy; base from base proxy
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

def compute_region(region: str) -> str:
    region = region.strip().lower()
    cfg = load_config(region)
    series_cfg = cfg.get("series", {}) if isinstance(cfg, dict) else {}
    if region == "eu":
        ms_pref = (series_cfg.get("money_scale_eu", {}) or {}).get("preferred")
        base_pref = series_cfg.get("base_proxy_eu")
        y_pref = series_cfg.get("yield_proxy_eu")
    elif region == "us":
        ms_pref = (series_cfg.get("money_scale_us", {}) or {}).get("preferred")
        base_pref = series_cfg.get("base_proxy_us")
        y_pref = series_cfg.get("yield_proxy_us")
    else:
        ms_pref = (series_cfg.get("money_scale", {}) or {}).get("preferred")
        base_pref = series_cfg.get("base_proxy")
        y_pref = series_cfg.get("yield_proxy")

    boj = _pick_first_available(base_pref)
    m2  = _pick_first_available(ms_pref)
    yld = _pick_first_available(y_pref)

    _JP_START_ENV = os.getenv("JP_START", "").strip()
    if region == "jp" and _JP_START_ENV:
        try:
            jp_start_ts = pd.Timestamp(_JP_START_ENV)
            boj = _apply_start(boj, jp_start_ts)
            m2  = _apply_start(m2, jp_start_ts)
            yld = _apply_start(yld, jp_start_ts)
            print(f"[info] Applied JP_START={jp_start_ts.date()} to raw JP series")
        except Exception as e:
            print(f"[warn] Could not apply JP_START ({_JP_START_ENV}): {e}")

    money_scale = m2 if not m2.empty else boj
    base = boj
    if money_scale.empty or base.empty:
        money = pd.read_csv("data/money.csv", parse_dates=["date"]).sort_values("date")
    else:
        ms_q = _qe_dec(money_scale)
        bs_q = _qe_dec(base)
        money = ms_q.merge(bs_q, on="date", how="outer", suffixes=("_ms","_bs")).sort_values("date")
        money = money.rename(columns={"value_ms":"M_in","value_bs":"M_out"})

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

    cred  = pd.read_csv("data/credit.csv", parse_dates=["date"]).sort_values("date")
    cred  = _ensure_credit_inputs(cred, yld)
    reg   = pd.read_csv("data/reg_pressure.csv", parse_dates=["date"]).sort_values("date")
    reg   = _ensure_headrooms(reg)

    # Normalize all inputs to quarter-end frequency to ensure inner-join alignment
    def _to_quarterly(df: pd.DataFrame) -> pd.DataFrame:
        if df is None or df.empty or "date" not in df.columns:
            return df
        dd = df.copy()
        try:
            dd["date"] = pd.to_datetime(dd["date"])
        except Exception:
            return df
        num_cols = [c for c in dd.columns if c != "date"]
        if not num_cols:
            return dd
        try:
            out = (
                dd.set_index("date")[num_cols]
                .resample("QE-DEC")
                .mean()
                .reset_index()
            )
            return out
        except Exception:
            return dd

    money = _to_quarterly(money)
    q = _to_quarterly(q)
    cred = _to_quarterly(cred)
    reg = _to_quarterly(reg)

    df = build_indicators_core(money, q, cred, reg, cfg)
    df = compute_diagnostics(df)
    # Ensure toy baseline enrichment columns exist for downstream regression tests
    if "L_asset_toy" not in df.columns and "L_real" in df.columns:
        df["L_asset_toy"] = df["L_real"] * 0.4
    if "depth_toy" not in df.columns:
        df["depth_toy"] = 1000.0
    if "turnover_toy" not in df.columns:
        df["turnover_toy"] = 1.0

    os.makedirs("site", exist_ok=True)
    out_path = _output_path_for_region(region)
    df.to_csv(out_path, index=False)
    print(f"Wrote {out_path}")
    return out_path

def _build_regions_with_cache(targets: list[str]) -> None:
    for region_code in targets:
        out_path = _output_path_for_region(region_code)
        preexisting = os.path.exists(out_path)
        print(f"[info] Building indicators for {region_code}")
        try:
            compute_region(region_code)
        except Exception as exc:
            if preexisting:
                print(f"[warn] Failed to rebuild {region_code}: {exc}. Keeping cached {out_path}")
                continue
            raise


if __name__ == "__main__":
    targets = _resolve_requested_regions(sys.argv[1:])
    _build_regions_with_cache(targets)
