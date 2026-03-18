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
    candidates = []
    for s in series_list:
        sid = s.get("id") if isinstance(s, dict) else None
        if not sid:
            continue
        df = _read_raw(str(sid))
        if not df.empty:
            candidates.append((str(sid), df))
    if not candidates:
        return pd.DataFrame()
    first_sid, first_df = candidates[0]
    try:
        first_latest = pd.to_datetime(first_df["date"], errors="coerce").dropna().max()
    except Exception:
        first_latest = pd.NaT
    best_sid, best_df = candidates[0]
    best_latest = first_latest
    for sid, df in candidates[1:]:
        try:
            latest = pd.to_datetime(df["date"], errors="coerce").dropna().max()
        except Exception:
            latest = pd.NaT
        if pd.isna(best_latest) or (pd.notna(latest) and latest > best_latest):
            best_sid, best_df, best_latest = sid, df, latest
    if pd.notna(first_latest) and pd.notna(best_latest) and best_latest - first_latest > pd.Timedelta(days=365):
        print(f"[info] Switching raw series from {first_sid} to fresher fallback {best_sid}")
        return best_df
    return first_df
    

def _data_path(kind: str, region: str) -> str:
    suffix = "" if region == "jp" else f"_{region}"
    return os.path.join("data", f"{kind}{suffix}.csv")


def _read_region_csv(kind: str, region: str) -> pd.DataFrame:
    preferred = _data_path(kind, region)
    fallback = os.path.join("data", f"{kind}.csv")
    path = preferred if os.path.exists(preferred) else fallback
    return pd.read_csv(path, parse_dates=["date"]).sort_values("date")


def _ensure_allocation_view(q: pd.DataFrame, housing_share: float = 0.4) -> pd.DataFrame:
    if q is None or q.empty:
        return q
    out = q.copy()
    for col in ("q_pay", "q_firm", "q_asset", "q_reserve"):
        if col not in out.columns:
            out[col] = pd.NA
    if "q_productive" not in out.columns:
        out["q_productive"] = pd.to_numeric(out["q_firm"], errors="coerce")
    if "q_financial" not in out.columns:
        out["q_financial"] = pd.to_numeric(out["q_asset"], errors="coerce")
    if "q_government" not in out.columns:
        out["q_government"] = pd.to_numeric(out["q_reserve"], errors="coerce")
    if "q_housing" not in out.columns:
        out["q_housing"] = pd.to_numeric(out["q_pay"], errors="coerce") * housing_share
    if "q_consumption" not in out.columns:
        pay = pd.to_numeric(out["q_pay"], errors="coerce")
        housing = pd.to_numeric(out["q_housing"], errors="coerce")
        out["q_consumption"] = (pay - housing).clip(lower=0)
    mece_cols = ["q_productive", "q_housing", "q_consumption", "q_financial", "q_government"]
    total = out[mece_cols].apply(pd.to_numeric, errors="coerce").sum(axis=1).replace({0: pd.NA})
    valid = total.notna()
    if valid.any():
        out.loc[valid, mece_cols] = out.loc[valid, mece_cols].div(total[valid], axis=0)
    return out


def _apply_start(df: pd.DataFrame, start_ts: pd.Timestamp) -> pd.DataFrame:
    if df.empty:
        return df
    return df[df["date"] >= start_ts].copy()


def _qe_dec(dfm: pd.DataFrame) -> pd.DataFrame:
    if dfm.empty:
        return dfm
    dfm = dfm.set_index("date").resample("QE-DEC").last().reset_index()
    return dfm


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
        money = _read_region_csv("money", region)
    else:
        ms_q = _qe_dec(money_scale)
        bs_q = _qe_dec(base)
        money = ms_q.merge(bs_q, on="date", how="outer", suffixes=("_ms", "_bs")).sort_values("date")
        money = money.rename(columns={"value_ms": "M_in", "value_bs": "M_out"})

    q = _ensure_allocation_view(_read_region_csv("allocation_q", region))
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

    cred = _read_region_csv("credit", region)
    reg = _read_region_csv("reg_pressure", region)

    money = _to_quarterly(money)
    q = _to_quarterly(q)
    cred = _to_quarterly(cred)
    reg = _to_quarterly(reg)

    df = build_indicators_core(money, q, cred, reg, cfg)
    df = compute_diagnostics(df)

    os.makedirs("site", exist_ok=True)
    # Deterministic per-region output path. Also keep JP legacy alias.
    out_path_region = f"site/indicators_{region}.csv"
    df.to_csv(out_path_region, index=False)
    print(f"Wrote {out_path_region}")
    if region == "jp":
        legacy = "site/indicators.csv"
        try:
            df.to_csv(legacy, index=False)
            print(f"Wrote {legacy}")
        except Exception:
            pass
    return out_path_region
