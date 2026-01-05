"""
Create minimal placeholder CSV files needed by tests when running in CI.

This script avoids external API calls and only writes tiny synthetic datasets
when the expected input files are missing:

  - data/money.csv (date, M_in, M_out)
  - data/reg_pressure.csv (date, p_R, V_R)
  - data/credit.csv (JP) with enrichment-required columns
  - data/credit_us.csv (US) with enrichment-required columns
  - data/credit_eu.csv (EU) with enrichment-required columns
    - data/allocation_q.csv (if missing) with MECE categories used by config.yml
    - EU/US equivalents for CI-only minimal runs:
            * data/money_eu.csv, data/reg_pressure_eu.csv, data/allocation_q_eu.csv, data/credit_eu.csv
            * data/money_us.csv, data/reg_pressure_us.csv, data/allocation_q_us.csv, data/credit_us.csv

After seeding inputs, it invokes scripts/02_compute_indicators.py to produce
site/indicators.csv required by entropy and enrichment tests, and also ensures
site/indicators_eu.csv and site/indicators_us.csv are produced so workflows can
rely on them without network access.
"""
from __future__ import annotations

import os

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
DATA = os.path.join(ROOT, "data")
SITE = os.path.join(ROOT, "site")


def ensure_dir(p: str) -> None:
    os.makedirs(p, exist_ok=True)


def write_csv(path: str, header: list[str], rows: list[list]) -> None:
    import csv
    ensure_dir(os.path.dirname(path))
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(header)
        w.writerows(rows)


def ensure_money():
    path = os.path.join(DATA, "money.csv")
    if os.path.exists(path):
        return
    rows = [
        ["2023-01-01", 100.0, 80.0],
        ["2023-04-01", 110.0, 82.0],
        ["2023-07-01", 120.0, 84.0],
        ["2023-10-01", 130.0, 86.0],
        ["2024-01-01", 140.0, 88.0],
    ]
    write_csv(path, ["date", "M_in", "M_out"], rows)


def ensure_reg_pressure():
    path = os.path.join(DATA, "reg_pressure.csv")
    if os.path.exists(path):
        return
    rows = [
        ["2023-01-01", 0.5, 80.0],
        ["2023-04-01", 0.6, 82.0],
        ["2023-07-01", 0.55, 84.0],
        ["2023-10-01", 0.65, 86.0],
        ["2024-01-01", 0.7, 88.0],
    ]
    enriched = []
    for date, p_r, v_r in rows:
        cap = v_r * (1 - 0.04 * p_r)
        lcr = v_r * (1 - 0.05 * p_r)
        nsfr = v_r * (1 - 0.06 * p_r)
        enriched.append([date, p_r, v_r, cap, lcr, nsfr])
    write_csv(path, ["date", "p_R", "V_R", "capital_headroom", "lcr_headroom", "nsfr_headroom"], enriched)


def ensure_credit_jp():
    path = os.path.join(DATA, "credit.csv")
    if os.path.exists(path):
        return
    rows = [
        ["2023-01-01", 1000, 400, 500, 500, 0.5, 1200, 1.2, 400, 500, 1000, 1.0],
        ["2023-04-01", 1100, 440, 520, 520, 0.6, 1210, 1.1, 440, 520, 1000, 1.0],
        ["2023-07-01", 1200, 480, 540, 540, 0.55, 1220, 1.05, 480, 540, 1000, 1.0],
        ["2023-10-01", 1300, 520, 560, 560, 0.65, 1230, 1.0, 520, 560, 1000, 1.0],
        ["2024-01-01", 1400, 560, 580, 580, 0.7, 1240, 0.95, 560, 580, 1000, 1.0],
    ]
    header = [
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
    ]
    write_csv(path, header, rows)


def ensure_credit_generic(path: str):
    if os.path.exists(path):
        return
    rows = [
        ["2023-01-01", 2000, 800, 1500, 1500, 2.0, 300.0, 1.0],
        ["2023-04-01", 2100, 840, 1520, 1520, 2.1, 305.0, 1.1],
        ["2023-07-01", 2200, 880, 1540, 1540, 2.2, 310.0, 1.05],
    ]
    header = ["date", "L_real", "L_asset", "U", "Y", "spread", "depth", "turnover"]
    write_csv(path, header, rows)


def ensure_money_region(region: str):
    suffix = region.lower()
    assert suffix in ("eu", "us")
    path = os.path.join(DATA, f"money_{suffix}.csv")
    if os.path.exists(path):
        return
    # Simple quarterly series matching JP placeholder shape
    rows = [
        ["2023-01-01", 200.0, 160.0],
        ["2023-04-01", 210.0, 162.0],
        ["2023-07-01", 220.0, 164.0],
        ["2023-10-01", 230.0, 166.0],
        ["2024-01-01", 240.0, 168.0],
    ]
    write_csv(path, ["date", "M_in", "M_out"], rows)


def ensure_reg_pressure_region(region: str):
    suffix = region.lower()
    assert suffix in ("eu", "us")
    path = os.path.join(DATA, f"reg_pressure_{suffix}.csv")
    if os.path.exists(path):
        return
    # Mirror JP logic with region-specific filename
    base_rows = [
        ["2023-01-01", 0.5, 160.0],
        ["2023-04-01", 0.6, 162.0],
        ["2023-07-01", 0.55, 164.0],
        ["2023-10-01", 0.65, 166.0],
        ["2024-01-01", 0.7, 168.0],
    ]
    enriched = []
    for date, p_r, v_r in base_rows:
        cap = v_r * (1 - 0.04 * p_r)
        lcr = v_r * (1 - 0.05 * p_r)
        nsfr = v_r * (1 - 0.06 * p_r)
        enriched.append([date, p_r, v_r, cap, lcr, nsfr])
    write_csv(path, ["date", "p_R", "V_R", "capital_headroom", "lcr_headroom", "nsfr_headroom"], enriched)


def ensure_allocation_q_region(region: str):
    suffix = region.lower()
    assert suffix in ("eu", "us")
    path = os.path.join(DATA, f"allocation_q_{suffix}.csv")
    if os.path.exists(path):
        return
    # Keep it minimal with the four core q_* columns expected by regional scripts
    rows = [
        ["2023-01-01", 0.25, 0.25, 0.25, 0.25],
        ["2023-04-01", 0.25, 0.25, 0.25, 0.25],
    ]
    header = ["date", "q_pay", "q_firm", "q_asset", "q_reserve"]
    write_csv(path, header, rows)


def ensure_allocation_q():
    path = os.path.join(DATA, "allocation_q.csv")
    if os.path.exists(path):
        return
    # Minimal MECE allocation aligned with config.yml (normalized per row)
    rows = [
        ["2023-01-01", 0.30, 0.30, 0.25, 0.15, 0.30, 0.12, 0.18, 0.25, 0.15],
        ["2023-04-01", 0.30, 0.30, 0.25, 0.15, 0.30, 0.12, 0.18, 0.25, 0.15],
    ]
    header = [
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
    ]
    write_csv(path, header, rows)


def main():
    ensure_dir(DATA)
    ensure_dir(SITE)
    ensure_money()
    ensure_reg_pressure()
    ensure_credit_jp()
    ensure_credit_generic(os.path.join(DATA, "credit_us.csv"))
    ensure_credit_generic(os.path.join(DATA, "credit_eu.csv"))
    ensure_allocation_q()

    # EU/US minimal inputs for their regional indicator scripts
    ensure_money_region("eu")
    ensure_money_region("us")
    ensure_reg_pressure_region("eu")
    ensure_reg_pressure_region("us")
    ensure_allocation_q_region("eu")
    ensure_allocation_q_region("us")

    # Build indicators (JP default) so that site/indicators.csv exists
    import importlib.util
    ind_path = os.path.join(SITE, "indicators.csv")
    if not os.path.exists(ind_path):
        spec = importlib.util.spec_from_file_location(
            "compute_ind", os.path.join(ROOT, "scripts", "02_compute_indicators.py")
        )
        mod = importlib.util.module_from_spec(spec)
        assert spec.loader is not None
        spec.loader.exec_module(mod)  # type: ignore[attr-defined]
        compute_region = getattr(mod, "compute_region", None)
        if callable(compute_region):
            compute_region("jp")
        else:
            raise RuntimeError("compute_region function not found in 02_compute_indicators.py")

    # Build EU/US indicators so that site/indicators_{eu,us}.csv exist in CI
    def _run_indicator_script(script_name: str):
        sp = importlib.util.spec_from_file_location(
            script_name.replace(".py", ""), os.path.join(ROOT, "scripts", script_name)
        )
        mod2 = importlib.util.module_from_spec(sp)
        assert sp and sp.loader is not None
        sp.loader.exec_module(mod2)  # type: ignore[attr-defined]
        main_fn = getattr(mod2, "main", None)
        if callable(main_fn):
            main_fn()

    eu_out = os.path.join(SITE, "indicators_eu.csv")
    if not os.path.exists(eu_out):
        try:
            _run_indicator_script("02_compute_indicators_eu.py")
        except Exception as e:
            print(f"[ci-prepare] EU indicators build failed: {e}")

    us_out = os.path.join(SITE, "indicators_us.csv")
    if not os.path.exists(us_out):
        try:
            _run_indicator_script("02_compute_indicators_us.py")
        except Exception as e:
            print(f"[ci-prepare] US indicators build failed: {e}")

    print("[ci-prepare] Minimal data prepared for tests.")


if __name__ == "__main__":
    main()
