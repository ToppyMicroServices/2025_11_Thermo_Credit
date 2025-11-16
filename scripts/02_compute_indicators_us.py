"""Compute United States thermo-credit indicators and write site/indicators_us.csv.

This mirrors scripts/02_compute_indicators.py (JP) and 02_compute_indicators_eu.py (EU)
with US-specific filenames and placeholder synthesis when inputs are missing.
"""
import os
import sys
import json
import argparse
from typing import Optional

import pandas as pd
import yaml

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from lib.indicators import build_indicators_core, compute_diagnostics, DEFAULT_HEADROOM_COLS

try:
    from lib.config_loader import load_config
    CFG = load_config("us")
except Exception:
    import yaml
    CFG = yaml.safe_load(open(os.path.join(ROOT, "config.yml")))
DATA_DIR = os.path.join(ROOT, "data")
SITE_DIR = os.path.join(ROOT, "site")
os.makedirs(SITE_DIR, exist_ok=True)

HEADROOM_DECAY = dict(zip(DEFAULT_HEADROOM_COLS, (0.04, 0.05, 0.06)))


def _with_headrooms(reg: Optional[pd.DataFrame]) -> Optional[pd.DataFrame]:
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


def _load_csv(path: str) -> Optional[pd.DataFrame]:
    if not os.path.exists(path):
        return None
    try:
        df = pd.read_csv(path)
        if "date" in df.columns:
            df["date"] = pd.to_datetime(df["date"])
        return df
    except Exception:
        return None


def _ensure_placeholders():
    """Create minimal US input CSVs if missing, using selected raw series."""
    sel_path = os.path.join(DATA_DIR, "series_selected_us.json")
    if not os.path.exists(sel_path):
        print("[US] series_selected_us.json missing; cannot build placeholders.")
        return
    try:
        meta = json.load(open(sel_path, "r", encoding="utf-8"))
    except Exception:
        print("[US] Could not parse series_selected_us.json; abort placeholders.")
        return

    money_id = (meta.get("money_scale_us") or {}).get("id")
    base_id = (meta.get("base_proxy_us") or {}).get("id")
    yield_id = (meta.get("yield_proxy_us") or {}).get("id")

    def _load_raw(sid: Optional[str]) -> Optional[pd.DataFrame]:
        if not sid:
            return None
        path = os.path.join(DATA_DIR, f"{sid}.csv")
        if not os.path.exists(path):
            return None
        try:
            df = pd.read_csv(path)
            dcol = next((c for c in df.columns if str(c).lower() == "date"), None)
            vcol = next((c for c in df.columns if str(c).lower() == "value"), None)
            if not dcol or not vcol:
                return None
            df = df[[dcol, vcol]].rename(columns={dcol: "date", vcol: "value"})
            df["date"] = pd.to_datetime(df["date"])
            return df.dropna()
        except Exception:
            return None

    # money_us.csv placeholder
    money_csv = os.path.join(DATA_DIR, "money_us.csv")
    if not os.path.exists(money_csv) and money_id and base_id:
        m_in = _load_raw(money_id)
        m_out = _load_raw(base_id)
        if m_in is not None and m_out is not None:
            m_in = m_in.rename(columns={"value": "M_in"})
            m_out = m_out.rename(columns={"value": "M_out"})
            money = m_in.merge(m_out, on="date", how="outer").sort_values("date").dropna()
            if not money.empty:
                money = (
                        money.set_index("date")
                        .resample("QE-DEC")
                        .mean()
                        .reset_index()
                    )
                money.to_csv(money_csv, index=False)
                print("[US] Placeholder money_us.csv built from raw series.")

    # allocation_q_us.csv placeholder
    alloc_csv = os.path.join(DATA_DIR, "allocation_q_us.csv")
    regen_alloc = False
    alloc_existed = os.path.exists(alloc_csv)
    if alloc_existed:
        try:
            alloc = pd.read_csv(alloc_csv, nrows=1)
            needed_cols = {"q_pay", "q_firm", "q_asset", "q_reserve"}
            if not needed_cols.issubset(set(alloc.columns)):
                regen_alloc = True
        except Exception:
            regen_alloc = True
    else:
        regen_alloc = True

    if regen_alloc and os.path.exists(money_csv):
        try:
            money = pd.read_csv(money_csv, parse_dates=["date"])
            dates = money["date"].drop_duplicates().sort_values()
            qdf = pd.DataFrame({"date": dates})
            qdf["q_pay"] = 0.35
            qdf["q_firm"] = 0.30
            qdf["q_asset"] = 0.20
            qdf["q_reserve"] = 0.15
            qdf.to_csv(alloc_csv, index=False)
            if alloc_existed:
                print("[US] Placeholder allocation_q_us.csv refreshed with q_* columns.")
            else:
                print("[US] Placeholder allocation_q_us.csv created (flat shares).")
        except Exception as exc:
            print(f"[US] Failed to build allocation_q_us.csv: {exc}")

    # credit_us.csv placeholder from yield
    credit_csv = os.path.join(DATA_DIR, "credit_us.csv")
    if not os.path.exists(credit_csv) and yield_id:
        ydf = _load_raw(yield_id)
        if ydf is not None:
            try:
                yq = ydf.resample("QE-DEC", on="date").mean().reset_index().rename(columns={"value": "spread"})
                yq["U"] = yq["spread"] * 12.0
                yq["Y"] = yq["U"]
                yq["depth"] = 900
                yq["turnover"] = 1.0
                yq["L_real"] = yq["spread"] * 6.0
                yq["L_asset"] = yq["L_real"] * 0.4
                yq = yq[["date", "L_real", "L_asset", "U", "Y", "spread", "depth", "turnover"]]
                yq.to_csv(credit_csv, index=False)
                print("[US] Placeholder credit_us.csv created from yield series.")
            except Exception as exc:
                print(f"[US] Failed to build credit_us.csv: {exc}")

    # reg_pressure_us.csv placeholder from yield/assets
    reg_csv = os.path.join(DATA_DIR, "reg_pressure_us.csv")
    if not os.path.exists(reg_csv) and yield_id and base_id:
        ydf = _load_raw(yield_id)
        bdf = _load_raw(base_id)
        if ydf is not None and bdf is not None:
            try:
                ydf = ydf.rename(columns={"value": "p_R"})
                bdf = bdf.rename(columns={"value": "V_R"})
                reg = ydf.merge(bdf, on="date", how="outer").sort_values("date").dropna()
                if not reg.empty:
                    reg = (
                        reg.set_index("date")
                        .resample("QE-DEC")
                        .mean()
                        .reset_index()
                    )
                reg = _with_headrooms(reg)
                reg.to_csv(reg_csv, index=False)
                print("[US] Placeholder reg_pressure_us.csv created (p_R from yield, V_R from assets).")
            except Exception as exc:
                print(f"[US] Failed to build reg_pressure_us.csv: {exc}")


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Compute US thermo-credit indicators")
    ap.add_argument("--strict", action="store_true", help="Do not create placeholders; fail if inputs missing")
    return ap.parse_args()


def main() -> None:
    args = parse_args()
    if not args.strict:
        _ensure_placeholders()

    money = _load_csv(os.path.join(DATA_DIR, "money_us.csv"))
    q = _load_csv(os.path.join(DATA_DIR, "allocation_q_us.csv"))
    cred = _load_csv(os.path.join(DATA_DIR, "credit_us.csv"))
    reg = _with_headrooms(_load_csv(os.path.join(DATA_DIR, "reg_pressure_us.csv")))

    # Normalize to quarterly frequency to ensure inner-joins align.
    def _to_quarterly(df: Optional[pd.DataFrame]) -> Optional[pd.DataFrame]:
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
            out = dd.set_index("date")[num_cols].resample("QE-DEC").mean().reset_index()
            return out
        except Exception:
            return dd

    money = _to_quarterly(money)
    q = _to_quarterly(q)
    cred = _to_quarterly(cred)
    reg = _to_quarterly(reg)

    missing = []
    if money is None:
        missing.append("money_us.csv")
    if q is None:
        missing.append("allocation_q_us.csv")
    if cred is None:
        missing.append("credit_us.csv")
    if reg is None:
        missing.append("reg_pressure_us.csv")

    if missing:
        print(f"[US] Missing required inputs for indicator computation: {', '.join(missing)}")
        return

    df = build_indicators_core(money, q, cred, reg, CFG)
    df = compute_diagnostics(df)

    out_path = os.path.join(SITE_DIR, "indicators_us.csv")
    df.to_csv(out_path, index=False)
    print(f"[US] Wrote {out_path}")


if __name__ == "__main__":
    main()