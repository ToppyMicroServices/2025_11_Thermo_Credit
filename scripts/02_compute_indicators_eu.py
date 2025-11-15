"""Compute Euro Area (EU) thermo-credit indicators and write site/indicators_eu.csv.

This mirrors scripts/02_compute_indicators.py for Japan, sourcing EU inputs if present.
If EU CSV inputs are missing, it synthesizes minimal placeholders from
data/series_selected_eu.json and the raw series CSVs produced by
scripts/04_build_features_eu.py.

Inputs (optional, recommended to provide real datasets):
  data/money_eu.csv          -> columns: date, M_in, M_out
  data/allocation_q_eu.csv   -> columns: date, q_<bucket1>, q_<bucket2>, ... (sums to 1)
  data/credit_eu.csv         -> columns: date, U, Y, spread, depth, turnover, L_real, L_asset
  data/reg_pressure_eu.csv   -> columns: date, p_R, V_R

Outputs:
  site/indicators_eu.csv
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
LIB = os.path.join(ROOT, "lib")
if LIB not in sys.path:
    sys.path.insert(0, LIB)

from lib.indicators import build_indicators_core, compute_diagnostics, DEFAULT_HEADROOM_COLS

try:
    from lib.config_loader import load_config
    CFG = load_config("eu")
except Exception:
    CFG = yaml.safe_load(open(os.path.join(ROOT, "config.yml")))
T0 = float(CFG.get("T0", 1.0))
lam = float(CFG.get("loop_forgetting", 0.98))
k_val = float(CFG.get("k", 1.0))

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
            df["date"] = pd.to_datetime(df["date"])  # robust
        return df
    except Exception:
        return None


def _ensure_placeholders():
    """If EU input CSVs are missing, create minimal placeholders using selected raw series.

    Placeholders:
      money_eu.csv: M_in from money_scale_eu, M_out from base_proxy_eu (align monthly)
      allocation_q_eu.csv: flat shares across 4 buckets summing to 1
      credit_eu.csv: quarterly average of yield_proxy_eu -> spread; synthetic columns
      reg_pressure_eu.csv: p_R from yield (as proxy), V_R from base_proxy_eu
    """
    sel_path = os.path.join(DATA_DIR, "series_selected_eu.json")
    if not os.path.exists(sel_path):
        print("[EU] series_selected_eu.json missing; cannot build placeholders.")
        return
    try:
        meta = json.load(open(sel_path, "r", encoding="utf-8"))
    except Exception:
        print("[EU] Could not parse series_selected_eu.json; abort placeholders.")
        return
    money_id = (meta.get("money_scale_eu") or {}).get("id")
    base_id = (meta.get("base_proxy_eu") or {}).get("id")
    yield_id = (meta.get("yield_proxy_eu") or {}).get("id")

    # money_eu.csv
    money_csv = os.path.join(DATA_DIR, "money_eu.csv")
    if not os.path.exists(money_csv) and money_id and base_id:
        m_in_path = os.path.join(DATA_DIR, f"{money_id}.csv")
        m_out_path = os.path.join(DATA_DIR, f"{base_id}.csv")
        if os.path.exists(m_in_path) and os.path.exists(m_out_path):
            try:
                m_in = pd.read_csv(m_in_path)
                m_out = pd.read_csv(m_out_path)
                # Normalize column names
                for df in (m_in, m_out):
                    dcol = next((c for c in df.columns if str(c).lower() == "date"), None)
                    vcol = next((c for c in df.columns if str(c).lower() == "value"), None)
                    if dcol and dcol != "date":
                        df.rename(columns={dcol: "date"}, inplace=True)
                    if vcol and vcol != "value":
                        df.rename(columns={vcol: "value"}, inplace=True)
                m_in["date"] = pd.to_datetime(m_in["date"]) 
                m_out["date"] = pd.to_datetime(m_out["date"]) 
                m_in = m_in.rename(columns={"value": "M_in"})
                m_out = m_out.rename(columns={"value": "M_out"})
                money = m_in.merge(m_out, on="date", how="outer").sort_values("date").dropna()
                money.to_csv(money_csv, index=False)
                print("[EU] Placeholder money_eu.csv built from raw series.")
            except Exception as e:
                print(f"[EU] Failed to build money_eu.csv: {e}")

    # allocation_q_eu.csv (flat shares)
    alloc_csv = os.path.join(DATA_DIR, "allocation_q_eu.csv")
    if not os.path.exists(alloc_csv) and os.path.exists(money_csv):
        try:
            money = pd.read_csv(money_csv, parse_dates=["date"]) 
            dates = money["date"].drop_duplicates().sort_values()
            q = pd.DataFrame({"date": dates})
            q["q_pay"] = 0.25
            q["q_firm"] = 0.25
            q["q_asset"] = 0.25
            q["q_reserve"] = 0.25
            q.to_csv(alloc_csv, index=False)
            print("[EU] Placeholder allocation_q_eu.csv created (flat shares).")
        except Exception as e:
            print(f"[EU] Failed to build allocation_q_eu.csv: {e}")

    # credit_eu.csv (enhanced): prefer corp-gov yield spread or MIR loan-deposit spread; depth from loans outstanding if available
    credit_csv = os.path.join(DATA_DIR, "credit_eu.csv")
    def _load_raw_by_id(sid: str) -> Optional[pd.DataFrame]:
        if not sid:
            return None
        p = os.path.join(DATA_DIR, f"{sid}.csv")
        if not os.path.exists(p):
            return None
        try:
            df = pd.read_csv(p)
            dcol = next((c for c in df.columns if str(c).lower() == "date"), None)
            vcol = next((c for c in df.columns if str(c).lower() == "value"), None)
            if not dcol or not vcol:
                return None
            df = df[[dcol, vcol]].rename(columns={dcol: "date", vcol: "value"})
            df["date"] = pd.to_datetime(df["date"]) 
            return df.dropna()
        except Exception:
            return None

    corp_id = (meta.get("eu_corp_yield") or {}).get("id")
    gov_id  = (meta.get("eu_gov_yield") or {}).get("id")
    mir_loan_id = (meta.get("eu_mir_loan") or {}).get("id")
    mir_dep_id  = (meta.get("eu_mir_deposit") or {}).get("id")
    loans_out_id = (meta.get("eu_loans_outstanding") or {}).get("id")
    bonds_out_id = (meta.get("eu_bonds_outstanding") or {}).get("id")
    turnover_id  = (meta.get("eu_turnover") or {}).get("id")

    advanced_available = (
        (corp_id and gov_id and _load_raw_by_id(corp_id) is not None and _load_raw_by_id(gov_id) is not None) or
        (mir_loan_id and mir_dep_id and _load_raw_by_id(mir_loan_id) is not None and _load_raw_by_id(mir_dep_id) is not None)
    )

    if advanced_available or (not os.path.exists(credit_csv) and yield_id):
        try:
            # Base quarterly index from spread
            spread_df = None
            if corp_id and gov_id:
                corp = _load_raw_by_id(corp_id)
                gov = _load_raw_by_id(gov_id)
                if corp is not None and gov is not None:
                    base = corp.merge(gov, on="date", how="inner", suffixes=("_corp","_gov")).dropna()
                    base["spread"] = base["value_corp"].astype(float) - base["value_gov"].astype(float)
                    spread_df = base[["date","spread"]]
            if spread_df is None and mir_loan_id and mir_dep_id:
                loan = _load_raw_by_id(mir_loan_id)
                dep  = _load_raw_by_id(mir_dep_id)
                if loan is not None and dep is not None:
                    base = loan.merge(dep, on="date", how="inner", suffixes=("_loan","_dep")).dropna()
                    base["spread"] = base["value_loan"].astype(float) - base["value_dep"].astype(float)
                    spread_df = base[["date","spread"]]
            if spread_df is None and yield_id:
                ydf = _load_raw_by_id(yield_id)
                if ydf is not None:
                    spread_df = ydf.rename(columns={"value":"spread"})[["date","spread"]]

            if spread_df is None or spread_df.empty:
                raise RuntimeError("No spread source available for EU credit")

            # Depth
            depth_df = None
            if loans_out_id:
                ldf = _load_raw_by_id(loans_out_id)
                if ldf is not None:
                    depth_df = ldf.rename(columns={"value":"depth"})[["date","depth"]]
            if depth_df is None and bonds_out_id:
                bdf = _load_raw_by_id(bonds_out_id)
                if bdf is not None:
                    depth_df = bdf.rename(columns={"value":"depth"})[["date","depth"]]

            # Turnover
            turn_df = None
            if turnover_id:
                tdf = _load_raw_by_id(turnover_id)
                if tdf is not None:
                    turn_df = tdf.rename(columns={"value":"turnover"})[["date","turnover"]]

            # Build quarterly table
            q_spread = spread_df.resample("QE-DEC", on="date").mean().reset_index()
            out = q_spread.copy()
            if depth_df is not None:
                q_depth = depth_df.resample("QE-DEC", on="date").mean().reset_index()
                out = out.merge(q_depth, on="date", how="left")
            else:
                out["depth"] = 1000.0
            if turn_df is not None:
                q_turn = turn_df.resample("QE-DEC", on="date").mean().reset_index()
                out = out.merge(q_turn, on="date", how="left")
            else:
                out["turnover"] = 1.0

            # Synthetic auxiliaries
            out["U"] = out["spread"].astype(float) * 10.0
            out["Y"] = out["U"]
            out["L_real"] = out["spread"].astype(float) * 5.0
            out["L_asset"] = out["L_real"] * 0.4
            out = out[["date","L_real","L_asset","U","Y","spread","depth","turnover"]].dropna(subset=["date","spread"])
            out.to_csv(credit_csv, index=False)
            print("[EU] credit_eu.csv built (enhanced sources if available).")
        except Exception as e:
            print(f"[EU] Failed to build enhanced credit_eu.csv: {e}")

    # reg_pressure_eu.csv
    reg_csv = os.path.join(DATA_DIR, "reg_pressure_eu.csv")
    if not os.path.exists(reg_csv) and yield_id and base_id:
        y_path = os.path.join(DATA_DIR, f"{yield_id}.csv")
        b_path = os.path.join(DATA_DIR, f"{base_id}.csv")
        if os.path.exists(y_path) and os.path.exists(b_path):
            try:
                ydf = pd.read_csv(y_path)
                bdf = pd.read_csv(b_path)
                for df in (ydf, bdf):
                    dcol = next((c for c in df.columns if str(c).lower() == "date"), None)
                    vcol = next((c for c in df.columns if str(c).lower() == "value"), None)
                    if dcol and dcol != "date":
                        df.rename(columns={dcol: "date"}, inplace=True)
                    if vcol and vcol != "value":
                        df.rename(columns={vcol: "value"}, inplace=True)
                    df["date"] = pd.to_datetime(df["date"]) 
                ydf = ydf.rename(columns={"value": "p_R"})
                bdf = bdf.rename(columns={"value": "V_R"})
                reg = ydf.merge(bdf, on="date", how="outer").sort_values("date").dropna()
                reg = _with_headrooms(reg)
                reg.to_csv(reg_csv, index=False)
                print("[EU] Placeholder reg_pressure_eu.csv created (p_R from yield, V_R from assets).")
            except Exception as e:
                print(f"[EU] Failed to build reg_pressure_eu.csv: {e}")


def parse_args():
    ap = argparse.ArgumentParser(description="Compute EU thermo-credit indicators")
    ap.add_argument("--strict", action="store_true", help="Do not synthesize placeholders; fail if inputs missing")
    return ap.parse_args()


def main():
    args = parse_args()
    if not args.strict:
        _ensure_placeholders()
    money = _load_csv(os.path.join(DATA_DIR, "money_eu.csv"))
    q = _load_csv(os.path.join(DATA_DIR, "allocation_q_eu.csv"))
    cred = _load_csv(os.path.join(DATA_DIR, "credit_eu.csv"))
    reg = _with_headrooms(_load_csv(os.path.join(DATA_DIR, "reg_pressure_eu.csv")))

    missing = []
    if money is None: missing.append("money_eu.csv")
    if q is None: missing.append("allocation_q_eu.csv")
    if cred is None: missing.append("credit_eu.csv")
    if reg is None: missing.append("reg_pressure_eu.csv")
    if missing:
        print(f"[EU] Missing required inputs for indicator computation: {', '.join(missing)}")
        return

    # Align all inputs to quarter-end to ensure inner merges retain rows
    def _resample_quarter_end(df: pd.DataFrame, keep_cols):
        if df is None or df.empty:
            return df
        cols = [c for c in keep_cols if c in df.columns]
        if not cols:
            return df
        out = (
            df.set_index("date")[cols]
              .resample("QE-DEC").mean()
              .reset_index()
        )
        return out

    _tmp = _resample_quarter_end(money, ["M_in", "M_out"])
    money = _tmp if _tmp is not None else money
    _tmp = _resample_quarter_end(reg, ["p_R", "V_R"] + list(DEFAULT_HEADROOM_COLS))
    reg = _tmp if _tmp is not None else reg
    # allocation shares: resample by mean in quarter
    if q is not None and not q.empty:
        q_cols = [c for c in q.columns if c != "date"]
        _tmp = _resample_quarter_end(q, q_cols)
        q = _tmp if _tmp is not None else q
        # re-normalize to 1.0 per row (safety)
        if not q.empty:
            vals = q[q_cols].astype(float)
            s = vals.sum(axis=1).replace(0.0, 1.0)
            q[q_cols] = vals.div(s, axis=0)
    # cred should already be quarterly; ensure quarter-end index
    _tmp = _resample_quarter_end(cred, [c for c in cred.columns if c != "date"])
    cred = _tmp if _tmp is not None else cred

    df = build_indicators_core(money, q, cred, reg, CFG)
    df = compute_diagnostics(df)

    out_path = os.path.join(SITE_DIR, "indicators_eu.csv")
    df.to_csv(out_path, index=False)
    print(f"[EU] Wrote {out_path}")


if __name__ == "__main__":
    main()
