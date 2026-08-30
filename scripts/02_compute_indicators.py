import json
import os, sys
import numpy as np
import pandas as pd

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)
from lib.indicators import build_indicators_core, compute_diagnostics, DEFAULT_HEADROOM_COLS
from lib.config_loader import load_config
from lib.no_lookahead import apply_release_lags, realtime_preprocessing_config, resolve_release_lags

DEFAULT_REGIONS = ("jp", "us", "eu")
MULTI_REGION_TOKENS = {"all", "*", "multi", "all_regions"}


def _output_path_for_region(region: str, *, realtime: bool = False) -> str:
    region = region.strip().lower()
    if realtime:
        return "site/indicators_realtime.csv" if region == "jp" else f"site/indicators_{region}_realtime.csv"
    return "site/indicators.csv" if region == "jp" else f"site/indicators_{region}.csv"


def _lag_manifest_path_for_region(region: str) -> str:
    region = region.strip().lower()
    return "site/realtime_release_lags.json" if region == "jp" else f"site/realtime_release_lags_{region}.json"


def _destination_path_for_region(region: str, *, realtime: bool = False) -> str:
    region = region.strip().lower()
    if realtime:
        return "site/credit_destination_realtime.csv" if region == "jp" else f"site/credit_destination_{region}_realtime.csv"
    return "site/credit_destination.csv" if region == "jp" else f"site/credit_destination_{region}.csv"


def _data_path_for_region(stem: str, region: str) -> str:
    region = region.strip().lower()
    if region and region != "jp":
        candidate = os.path.join("data", f"{stem}_{region}.csv")
        if os.path.exists(candidate):
            return candidate
    return os.path.join("data", f"{stem}.csv")


def _read_region_csv(stem: str, region: str) -> pd.DataFrame:
    return pd.read_csv(_data_path_for_region(stem, region), parse_dates=["date"]).sort_values("date")


def _merge_direct_credit_destination(cred: pd.DataFrame, cfg: dict, region: str) -> pd.DataFrame:
    dest_cfg = cfg.get("credit_destination", {}) if isinstance(cfg, dict) else {}
    if not isinstance(dest_cfg, dict):
        return cred
    raw_path = str(dest_cfg.get("direct_panel_path", "") or "").strip()
    if not raw_path:
        return cred
    path = raw_path
    if not os.path.isabs(path):
        path = os.path.join(ROOT, path)
    if not os.path.exists(path):
        print(f"[warn] credit_destination.direct_panel_path missing for {region}: {raw_path}")
        return cred
    direct = pd.read_csv(path, parse_dates=["date"]).sort_values("date")
    if direct.empty or "date" not in direct.columns:
        return cred
    direct_cols = [
        col
        for col in direct.columns
        if col == "date"
        or col.startswith("C_")
        or col.startswith("share_")
        or col.startswith("borrower_composition_")
        or col.startswith("fixed_investment_")
        or col.startswith("legacy_")
        or col.startswith("primary_")
        or col.startswith("werner_")
        or col.startswith("muller_verner_")
        or col.startswith("stock_primary_")
        or col.startswith("stock_household_")
        or col.startswith("household_")
        or col
        in {
            "destination_coverage",
            "destination_coverage_observed",
            "classified_positive_flow",
            "total_positive_flow",
            "unclassified_positive_flow",
            "common_taxonomy_delta_valid",
            "mapped_domestic_stock",
            "stock_local_governments_explicit",
            "stock_overseas_explicit",
            "stock_unresolved_residual",
            "explicit_scope_stock",
            "explicit_scope_gap_to_official_stock",
        }
    ]
    direct = direct.loc[:, direct_cols].drop_duplicates("date", keep="last")
    if cred is None or cred.empty:
        return direct
    left = cred.copy()
    left["date"] = pd.to_datetime(left["date"], errors="coerce")
    direct["date"] = pd.to_datetime(direct["date"], errors="coerce")
    left["__quarter"] = left["date"].dt.to_period("Q-DEC")
    direct["__quarter"] = direct["date"].dt.to_period("Q-DEC")
    direct = direct.drop(columns=["date"]).drop_duplicates("__quarter", keep="last")
    merged = left.merge(direct, on="__quarter", how="left").drop(columns=["__quarter"])
    if "C_t" in merged.columns and "classified_positive_flow" in merged.columns:
        merged.loc[:, "C_t"] = pd.to_numeric(merged["C_t"], errors="coerce").combine_first(
            pd.to_numeric(merged["classified_positive_flow"], errors="coerce")
        )
    return merged


def _write_lag_manifest(region: str, profile: str, default_lag_days: int, column_lags: dict[str, int]) -> None:
    payload = {
        "region": region,
        "profile": profile,
        "default_lag_days": int(default_lag_days),
        "group_lags_days": {
            k.replace("__group__", ""): int(v)
            for k, v in sorted(column_lags.items())
            if k.startswith("__group__")
        },
        "release_lags_days": {k: int(v) for k, v in sorted(column_lags.items()) if not k.startswith("__group__")},
        "note": "Lagged panels are as-of transforms: a dated value becomes usable only after date + lag_days.",
    }
    path = _lag_manifest_path_for_region(region)
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2, sort_keys=True)


def _write_credit_destination_panel(frame: pd.DataFrame, region: str, *, realtime: bool = False) -> None:
    cols = [
        "date",
        "C_t",
        "C_t_raw_delta",
        "C_G",
        "C_B",
        "C_E",
        "C_NFB",
        "C_FIN",
        "C_PROP",
        "C_HH_NONHOUSING",
        "C_R",
        "C_A",
        "q_t",
        "q_t_primary",
        "one_minus_q_t",
        "borrower_composition_NFB_1q",
        "borrower_composition_FIN_1q",
        "borrower_composition_PROP_1q",
        "borrower_composition_HH_NONHOUSING_1q",
        "borrower_composition_NFB_4q",
        "borrower_composition_FIN_4q",
        "borrower_composition_PROP_4q",
        "borrower_composition_HH_NONHOUSING_4q",
        "borrower_composition_G_1q",
        "borrower_composition_B_1q",
        "borrower_composition_E_1q",
        "borrower_composition_G_4q",
        "borrower_composition_B_4q",
        "borrower_composition_E_4q",
        "operating_borrower_share_1q",
        "operating_borrower_share_4q",
        "share_G_proxy",
        "share_B_proxy",
        "share_E_proxy",
        "share_G_direct",
        "share_B_direct",
        "share_E_direct",
        "destination_coverage",
        "lambda_B",
        "housing_construction_share",
        "credit_destination_source",
        "credit_destination_taxonomy_id",
        "credit_destination_overflow_scaled",
        "primary_component_total_mismatch",
        "preprocessing_mode",
        "release_lag_profile",
    ]
    retained_prefixes = (
        "legacy_",
        "primary_",
        "werner_",
        "muller_verner_",
        "C_WERNER_",
        "C_MV_",
        "stock_primary_",
        "stock_household_",
        "household_",
    )
    present = list(
        dict.fromkeys(
            [c for c in cols if c in frame.columns]
            + [
                c
                for c in frame.columns
                if c.startswith(retained_prefixes)
                and not c.endswith("_destination")
            ]
        )
    )
    if "date" not in present:
        return
    frame[present].to_csv(_destination_path_for_region(region, realtime=realtime), index=False)


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
        df.loc[:, "date"] = pd.to_datetime(df["date"])
    except Exception:
        pass

    if "spread" not in df.columns:
        df.loc[:, "spread"] = np.nan
    df.loc[:, "spread"] = pd.to_numeric(df["spread"], errors="coerce")
    df.loc[:, "quarter"] = df["date"].dt.to_period("Q-DEC")
    y_fallback = _prepare_yield_fallback(yield_df)
    if not y_fallback.empty:
        df = df.merge(y_fallback, on="quarter", how="left")
        df.loc[:, "spread"] = df["spread"].combine_first(df["spread_fallback"])
        df = df.drop(columns=["spread_fallback"])

    _harmonize_activity_scale(df)

    if "U" not in df.columns:
        df.loc[:, "U"] = np.nan
    u_series = pd.to_numeric(df["U"], errors="coerce")
    for fallback in ("U_gdp_only", "Y", "L_real"):
        if fallback in df.columns:
            u_series = u_series.combine_first(pd.to_numeric(df[fallback], errors="coerce"))
    df.loc[:, "U"] = u_series

    if "Y" not in df.columns:
        df.loc[:, "Y"] = np.nan
    y_series = pd.to_numeric(df["Y"], errors="coerce")
    for fallback in ("U_gdp_only",):
        if fallback in df.columns:
            y_series = y_series.combine_first(pd.to_numeric(df[fallback], errors="coerce"))
    df.loc[:, "Y"] = y_series

    return df.drop(columns=["quarter"], errors="ignore")


def _harmonize_activity_scale(df: pd.DataFrame) -> None:
    if "L_real" not in df.columns:
        return
    credit = pd.to_numeric(df["L_real"], errors="coerce").dropna()
    if credit.empty:
        return
    credit_median = float(credit.abs().median())
    if not np.isfinite(credit_median) or credit_median <= 0:
        return
    for column in ("Y", "U", "U_gdp_only"):
        if column not in df.columns:
            continue
        values = pd.to_numeric(df[column], errors="coerce")
        valid = values.dropna()
        if valid.empty:
            continue
        ratio = float(valid.abs().median() / credit_median)
        if not np.isfinite(ratio) or ratio <= 1_000:
            continue
        scale = 1_000_000_000.0 if ratio > 1_000_000 else 1_000_000.0
        df.loc[:, column] = values / scale


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
        q_ext = pd.DataFrame({"date": ext_idx})
        for col in [c for c in q.columns if c.startswith("q_")]:
            q_ext[col] = first_row[col]
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
        money = _read_region_csv("money", region)
    else:
        ms_q = _qe_dec(money_scale)
        bs_q = _qe_dec(base)
        money = ms_q.merge(bs_q, on="date", how="outer", suffixes=("_ms","_bs")).sort_values("date")
        money = money.rename(columns={"value_ms":"M_in","value_bs":"M_out"})

    q = _read_region_csv("allocation_q", region)
    q_share_cols = [c for c in q.columns if c.startswith("q_")]
    cfg_q_cols = cfg.get("q_cols") if isinstance(cfg, dict) else None
    if q_share_cols and (not isinstance(cfg_q_cols, (list, tuple)) or not set(cfg_q_cols).intersection(q_share_cols)):
        cfg = dict(cfg)
        cfg["q_cols"] = q_share_cols
    if not money.empty and not q.empty and money["date"].min() < q["date"].min():
        first_row = q.iloc[0]
        ext_idx = pd.date_range(money["date"].min(), q["date"].min() - pd.offsets.QuarterEnd(0), freq="QE-DEC")
        if len(ext_idx) > 0:
            q_ext = pd.DataFrame({"date": ext_idx})
            for col in [c for c in q.columns if c.startswith("q_")]:
                q_ext[col] = first_row[col]
            q = pd.concat([q_ext, q], ignore_index=True).sort_values("date").reset_index(drop=True)

    cred  = _read_region_csv("credit", region)
    cred  = _ensure_credit_inputs(cred, yld)
    cred  = _merge_direct_credit_destination(cred, cfg, region)
    reg   = _read_region_csv("reg_pressure", region)
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

    def _build_output_frame(
        money_in: pd.DataFrame,
        q_in: pd.DataFrame,
        cred_in: pd.DataFrame,
        reg_in: pd.DataFrame,
        *,
        preprocessing_mode: str,
        lag_profile: str,
    ) -> pd.DataFrame:
        out = build_indicators_core(money_in, q_in, cred_in, reg_in, cfg)
        out = compute_diagnostics(out)
        # Ensure toy baseline enrichment columns exist for downstream regression tests
        if "L_asset_toy" not in out.columns and "L_real" in out.columns:
            out["L_asset_toy"] = out["L_real"] * 0.4
        if "depth_toy" not in out.columns:
            out["depth_toy"] = 1000.0
        if "turnover_toy" not in out.columns:
            out["turnover_toy"] = 1.0
        out["preprocessing_mode"] = preprocessing_mode
        out["release_lag_profile"] = lag_profile
        return out

    df = _build_output_frame(
        money,
        q,
        cred,
        reg,
        preprocessing_mode="dashboard_retrospective",
        lag_profile="none",
    )

    os.makedirs("site", exist_ok=True)
    out_path = _output_path_for_region(region)
    df.to_csv(out_path, index=False)
    _write_credit_destination_panel(df, region, realtime=False)
    print(f"Wrote {out_path}")
    rt_cfg = realtime_preprocessing_config(cfg)
    if rt_cfg.get("enabled", True):
        column_lags, default_lag_days, lag_profile = resolve_release_lags(cfg)
        realtime_df = _build_output_frame(
            apply_release_lags(money, column_lags, default_lag_days=default_lag_days),
            apply_release_lags(q, column_lags, default_lag_days=default_lag_days),
            apply_release_lags(cred, column_lags, default_lag_days=default_lag_days),
            apply_release_lags(reg, column_lags, default_lag_days=default_lag_days),
            preprocessing_mode="real_time_release_lagged",
            lag_profile=lag_profile,
        )
        rt_path = _output_path_for_region(region, realtime=True)
        realtime_df.to_csv(rt_path, index=False)
        _write_credit_destination_panel(realtime_df, region, realtime=True)
        _write_lag_manifest(region, lag_profile, default_lag_days, column_lags)
        print(f"Wrote {rt_path}")
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
