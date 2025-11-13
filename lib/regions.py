import os
import pandas as pd
from typing import Dict, Any, Sequence, Optional

from lib.indicators import build_indicators_core, compute_diagnostics


def _load_cfg(region: str) -> Dict[str, Any]:
    import yaml
    base: Dict[str, Any] = {}
    reg: Dict[str, Any] = {}
    try:
        with open("config.yml", "r") as f:
            base = yaml.safe_load(f) or {}
            if not isinstance(base, dict):
                base = {}
    except Exception:
        base = {}
    if region in ("jp", "eu", "us"):
        p = f"config_{region}.yml"
        if os.path.exists(p):
            try:
                with open(p, "r") as f:
                    reg = yaml.safe_load(f) or {}
                    if not isinstance(reg, dict):
                        reg = {}
            except Exception:
                reg = {}
    return {**base, **reg}


def _read_raw(sid: str) -> pd.DataFrame:
    path = os.path.join("data", f"{sid}.csv")
    if not os.path.exists(path):
        return pd.DataFrame()
    df = pd.read_csv(path, parse_dates=["date"]).dropna()
    return df.sort_values("date")


def _pick_first_available(series_list: Optional[Sequence[Dict[str, Any]]]) -> pd.DataFrame:
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


def _apply_start(df: pd.DataFrame, start_ts: pd.Timestamp) -> pd.DataFrame:
    if df.empty:
        return df
    return df[df["date"] >= start_ts].copy()


def _qe_dec(dfm: pd.DataFrame) -> pd.DataFrame:
    if dfm.empty:
        return dfm
    dfm = dfm.set_index("date").resample("QE-DEC").last().reset_index()
    return dfm


def compute_region(region: str) -> str:
    region = (region or "jp").strip().lower()
    cfg = _load_cfg(region)

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
    m2 = _pick_first_available(ms_pref)
    yld = _pick_first_available(y_pref)

    jp_env = os.getenv("JP_START", "").strip()
    if region == "jp" and jp_env:
        try:
            ts = pd.Timestamp(jp_env)
            boj = _apply_start(boj, ts)
            m2 = _apply_start(m2, ts)
            yld = _apply_start(yld, ts)
            print(f"[info] Applied JP_START={ts.date()} to raw JP series")
        except Exception as e:
            print(f"[warn] Could not apply JP_START ({jp_env}): {e}")

    money_scale = m2 if not m2.empty else boj
    base = boj
    if money_scale.empty or base.empty:
        money = pd.read_csv("data/money.csv", parse_dates=["date"]).sort_values("date")
    else:
        ms_q = _qe_dec(money_scale)
        bs_q = _qe_dec(base)
        money = ms_q.merge(bs_q, on="date", how="outer", suffixes=("_ms", "_bs")).sort_values("date")
        money = money.rename(columns={"value_ms": "M_in", "value_bs": "M_out"})

    q = pd.read_csv("data/allocation_q.csv", parse_dates=["date"]).sort_values("date")
    if not money.empty and not q.empty and money["date"].min() < q["date"].min():
        first_row = q.iloc[0]
        ext_idx = pd.date_range(money["date"].min(), q["date"].min() - pd.offsets.QuarterEnd(0), freq="QE-DEC")
        if len(ext_idx) > 0:
            q_ext = pd.DataFrame(
                {
                    "date": ext_idx,
                    "q_pay": first_row["q_pay"],
                    "q_firm": first_row["q_firm"],
                    "q_asset": first_row["q_asset"],
                    "q_reserve": first_row["q_reserve"],
                }
            )
            q = pd.concat([q_ext, q], ignore_index=True).sort_values("date").reset_index(drop=True)

    cred = pd.read_csv("data/credit.csv", parse_dates=["date"]).sort_values("date")
    reg = pd.read_csv("data/reg_pressure.csv", parse_dates=["date"]).sort_values("date")

    df = build_indicators_core(money, q, cred, reg, cfg)
    df = compute_diagnostics(df)

    os.makedirs("site", exist_ok=True)
    out_path = "site/indicators.csv" if region == "jp" else f"site/indicators_{region}.csv"
    df.to_csv(out_path, index=False)
    print(f"Wrote {out_path}")
    return out_path
