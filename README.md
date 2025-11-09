- ### Yield comparison (US vs JP)
    To visually compare the placeholder `DGS10` and Japanese long-term yield `IRLTLT01JPM156N`:
    - Ensure `FRED_API_KEY` is set.
    - Run:
        ```
        python scripts/compare_yields.py --start 1995-01-01 --png
        ```
    - Outputs: `site/yield_compare.html` (interactive), `site/yield_compare.png` (optional), and `site/yield_compare_metrics.csv` (spread and rolling correlation diagnostics).
# Thermo‑Credit Monitor (TQTC) — Minimal MVP

This repository computes four thermo‑credit indicators from public statistics (local CSVs or FRED API) and publishes a monthly report to `site/`:

- **S_M — Monetary Dispersion Entropy (entropy‑like, extensive)**
- **T_L — Liquidity “Temperature” (normalized liquidity score)**
- **PLD — Policy/Regulatory Loop Dissipation (`loop_area`)**
- **F_C — Helmholtz Free Energy (`F_C = U − T0 · S_M`)**
- **X_C — Credit Exergy Ceiling (`X_C = (U−U0) + p0·(V_C−V0) − T0·(S_M−S0)`; falls back to F_C if baselines are not configured)**

Artifacts: `site/report.html` (interactive Plotly + PNG fallbacks), `archive.json`, `feed.xml`, `sitemap.xml`, `robots.txt`.

> **Note on formulas**: GitHub does not render LaTeX by default. All formulas below use plain ASCII.

---

## Concept in 60 seconds
**We do not claim “money = entropy.”** We design an **entropy‑like, extensive index** of how money is distributed **and** how much actually moved:

```
S_M = k * M_in * H(q)
```
- `M_in` — money‑in‑circulation inside the chosen system and period (actual flow total; currency units)
- `q = (q1, …, qK)` — composition shares of that flow over K categories (non‑negative, sum to 1)
- `H(q) = - Σ q_i * log(q_i)` — Shannon entropy (nat), measures **dispersion** independent of scale
- `k` — scale constant (role similar to Boltzmann constant). Use `k = 1` or calibrate so a base month equals 100.

**Intuition**: same dispersion but **twice the scale** → twice the impact. Hence `scale × dispersion`.

**Analogy to ideal mixing (isomorphic, not identical):**
- mole fractions `x_i` ↔ shares `q_i`
- particle count `N` ↔ `M_in`
- `k_B` ↔ `k`
- mixing entropy: `ΔS_mix = k_B * N * H(x)` ↔ `S_M = k * M_in * H(q)`

We call it **Monetary Dispersion Entropy** (or **Money Dispersion Index**) to avoid the misleading claim that money *is* entropy.

---

## Foundations (minimal axioms)
- **F1. System & boundary.** Define market, period, and category universe; treat it as a closed system for bookkeeping.
- **F2. Extensive quantity.** `M_in` grows additively with system size/flow volume.
- **F3. Composition.** Split the period flow into non‑negative parts `f_i`; set `M_in = Σ f_i`, `q_i = f_i / M_in`.
- **F4. Entropy functional.** Under the usual continuity/maximum/additivity axioms, uncertainty reduces to Shannon `H(q)`.
- **F5. Scale constant.** Choose `k` so that doubling `M_in` doubles `S_M` (extensivity).

**First‑law style decomposition (bookkeeping):**
```
ΔU = W_policy + T0 * ΔS_M + ε
```
`W_policy` = structured work by policy/regulation/balance‑sheet moves; `T0*ΔS_M` = dispersive contribution; `ε` = residual.

**Second‑law style monotonicity:** in a closed system, random mixing raises `S_M` (ΔS_M ≥ 0). To lower `S_M` deliberately (concentration), some structured work `W_policy` is required.

---

## Indicators (ASCII definitions)
- **S_M**: `S_M = k * M_in * H(q)`
- **T_L**: normalized liquidity score (e.g., using turnover, spreads, depth; z‑scores → [0,1])
- **PLD (loop_area)**: streaming estimate of a policy/regulatory loop’s dissipative area
- **F_C**: `F_C = U - T0 * S_M`  # Helmholtz‑like free energy (fixed environment T0)
- **X_C**: `X_C = (U-U0) + p0*(V_C-V0) - T0*(S_M-S0)`  # Exergy‑like ceiling; optional (requires baselines)

> Implementation notes: see `lib/entropy.py`, `lib/temperature.py`, `lib/loop_area.py`.

---

## Category design (what is `q`?)
Pick **one** axis and keep it stable (MECE, reproducible):
- **Purpose** (e.g., real estate, capex, working capital, inventories, durables, services, export, import)
- **Counterparty** (households, non‑financial corporates, government, rest‑of‑world)
- **Instrument** (cash, deposits, loans, mortgages, consumer credit, CP/bonds)
- **Region** or **Maturity buckets**

Practical tips: fix K; if you change K, report `H/ln(K)` as a 0–1 normalized dispersion. Group tiny buckets into “other.”

---

## Data pipeline
```
 data/*.csv  ──▶  scripts/02_compute_indicators.py  ──▶  site/indicators.csv
                    scripts/03_make_report.py       ──▶  site/report.html (+ PNG / archive / RSS / sitemap)
(optional) scripts/01_build_features.py  # JP online fetch if FRED_API_KEY is present
(optional) scripts/04_build_features_eu.py  # EU online fetch (writes data/series_selected_eu.json)
scripts/02_compute_indicators_eu.py  # EU indicators -> site/indicators_eu.csv
(optional) scripts/05_build_features_us.py  # US online fetch (writes data/series_selected_us.json)
scripts/02_compute_indicators_us.py  # US indicators -> site/indicators_us.csv
```
- Without `FRED_API_KEY`, the fetch step is skipped and local CSVs are used (you will see: `No FRED_API_KEY; skip online fetch ...`).
- Outputs: `indicators.csv`, `report.html`, `fig1.png..fig3.png`, `archive.json`, `feed.xml`, `sitemap.xml`, `robots.txt`.

### Current FRED series used
`scripts/01_build_features.py` now selects the first usable candidate for each role, in the order: environment variable → `config.yml` → built-in defaults.

| role         | Default series_id | Description                                      | Env override |
|--------------|-------------------|--------------------------------------------------|--------------|
| money_scale  | `MYAGM2JPM189S`   | Japan M2, monthly (via FRED)                     | `MONEY_SERIES` |
| base_proxy   | `JPNASSETS`       | Bank of Japan total assets                       | `BASE_SERIES` |
| yield_proxy  | `DGS10`           | 10Y Treasury yield (placeholder until JGB added) | `YIELD_SERIES` |

Run `python scripts/01_build_features.py --list-series` to see the resolved candidate list (including overrides).
### EU (Euro Area) support
We added a parallel EU pipeline and a simple tab switcher in the report (JP / EU).

1) Export your FRED API key and fetch EU series (roles: money_scale_eu, base_proxy_eu, yield_proxy_eu):

```bash
export FRED_API_KEY=YOUR_REAL_KEY
python scripts/04_build_features_eu.py --list-series   # optional preview
python scripts/04_build_features_eu.py                # writes data/series_selected_eu.json and raw CSVs
```

2) Compute EU indicators (writes `site/indicators_eu.csv`):

```bash
python scripts/02_compute_indicators_eu.py
```

3) Rebuild the report with tabs:

```bash
python scripts/03_make_report.py
open site/report.html
```

If `site/indicators_eu.csv` exists, the report renders two tabs. Otherwise it renders JP only.

Notes:
- If you don’t yet have EU-specific CSVs (`money_eu.csv`, `allocation_q_eu.csv`, `credit_eu.csv`, `reg_pressure_eu.csv`), the EU indicator script will synthesize placeholders from the fetched raw series to let you preview the flow.
- PNG fallbacks need `kaleido`. It’s listed in `requirements.txt`, but ensure it’s installed in your active venv:
    ```bash
    pip install kaleido
    ```

### US support
United States support mirrors the EU flow. A third tab appears automatically when `site/indicators_us.csv` exists.

1) Export your FRED key and resolve US series (roles: money_scale_us, base_proxy_us, yield_proxy_us). Environment overrides are `MONEY_SERIES_US`, `BASE_SERIES_US`, `YIELD_SERIES_US`.

```bash
export FRED_API_KEY=YOUR_REAL_KEY
python scripts/05_build_features_us.py --list-series   # optional preview
python scripts/05_build_features_us.py                # writes data/series_selected_us.json and raw CSVs
```

2) Compute US indicators (`site/indicators_us.csv`). Omit `--strict` if you want the script to synthesize placeholders whenever inputs are missing.

```bash
python scripts/02_compute_indicators_us.py
```

3) Rebuild the report. Tabs cycle JP / EU / US depending on available regions.

```bash
python scripts/03_make_report.py
open site/report.html
```

Notes:
- Placeholder generators require the raw series CSVs dropped by the US build script. Run the build step at least once (or provide equivalent local CSVs) before using `--strict`.
- The raw inputs plot now colors US-enabled series in green for quick visual comparison across regions.


## What is downloaded vs. what is shown (Japan)
**Downloaded (online fetch, only if `FRED_API_KEY` is set):**
- `MYAGM2JPM189S` — *Japan M2, monthly* (via FRED).  
  Provider: Board of Governors of the Federal Reserve System (FRED mirror). Country: **JP**.

**Local inputs (if no API key, or in addition to the above):**
- `data/*.csv` — your Japan‑specific tables (e.g., sectoral loans/outstandings, spreads, turnover/depth). The repo treats these as **Japan** by default unless you replace them.

**What the report shows (site/report.html):**
- Core indicators for **Japan**: `S_M`, `T_L`, `loop_area (PLD)`, `F_C` and optionally `X_C`.
- Diagnostic pages (if columns exist): Maxwell‑like test and First‑law decomposition.
- A **Data & Definitions** table that explains each column.
- (Optional) A **Sources** table, if you provide `data/sources.json` as described below.

**Optional provenance file (`data/sources.json`):**
Provide a JSON array describing each input series so the report can render a Sources table automatically. Example:
```json
[
  {
    "id": "MYAGM2JPM189S",
    "title": "Japan M2 (Monthly, SA/NSA per FRED)",
    "provider": "FRED (BoJ/FRB mirror)",
    "country": "JP",
    "frequency": "Monthly",
    "units": "JPY (level or index as provided)",
    "url": "https://fred.stlouisfed.org/series/MYAGM2JPM189S",
    "note": "Used in liquidity/scale features; exact transformation documented in scripts/01_build_features.py"
  },
  {
    "id": "loans_by_sector_jp.csv",
    "title": "Loans Outstanding by Sector (Japan)",
    "provider": "Local CSV (BoJ original tables pre‑processed)",
    "country": "JP",
    "frequency": "Monthly/Quarterly",
    "units": "JPY",
    "url": "",
    "note": "Feeds q (composition) and M_in (scale) after preprocessing."
  }
]
```

### Selecting which series to display
- To pin a specific FRED series without editing code, either export an environment variable (e.g. ``MONEY_SERIES=JPNASSETS`` or ``MONEY_SERIES=JPNASSETS@1995-01-01`` to override the start date) before running the build step, or declare preferences in `config.yml`.
- Example `config.yml` fragment:
    ```yaml
    series:
        money_scale:
            preferred:
                - id: JPNASSETS
                    start: 1995-01-01
                - id: MYAGM2JPM189S
        yield_proxy:
            - DGS10
            - id: IRLTLT01JPM156N
                note: Long-term JGB yield (FRED)
    ```
    The first candidate returning data is used; duplicates are ignored.
- `python scripts/01_build_features.py --list-series --role money_scale` shows the priority list for a single role. Results are also saved to `data/series_selected.json` whenever the fetch step runs.
- The optional `data/sources.json` works as before for report provenance. Set `"enabled": true` for any entries you want to **plot** in the report’s “Raw inputs (enabled)” section, and optionally customize `"path"`.
- Example `data/sources.json` entry (minimal):
    ```json
    {"id":"JPNASSETS","title":"BoJ Total Assets","enabled":true}
    ```
    This assumes a CSV at `data/JPNASSETS.csv` with columns `date,value`. If you need a different filename, add `"path"`.
- Raw inputs plot normalization: each enabled series is scaled to 100 at its first valid observation for comparability.
- Expected CSV schema: columns `date` and `value` (case-insensitive). Dates must be parseable; values numeric.
- The report normalizes each series to **100 at its first non-missing observation** for visual comparability. Use the legend to toggle series.
- You can keep many candidates in `sources.json` (enabled=false by default) and flip them on/off without touching the code.

---

## Local run (quick start)
```bash
python3 -m venv .venv && source .venv/bin/activate
python -m pip install -U pip
pip install -r requirements.txt
# For PNG fallbacks:
pip install kaleido

python scripts/02_compute_indicators.py
python scripts/03_make_report.py
open site/report.html   # macOS
```

---

## GitHub Pages (Actions)
- Workflow: `.github/workflows/build-and-publish.yml`
- Runs on `main` (and only this repo), monthly cron (1st day 04:00 JST), and on demand
- Publishes `site/` to Pages with interactive charts and fallbacks

### Secrets / Variables
- `FRED_API_KEY` (optional) — FRED API key; if missing, we use local CSVs
- `TMS_BASE_URL` (optional) — defaults to `https://toppymicros.com/2025_11_Thermo_Credit`

---

## Security & robustness
- Remote `archive.json` merge is **https‑only**, **allow‑listed domains**, **≤ 512 KiB**, and content‑type checked; otherwise we fall back to a fresh local file.
- Plotly currently uses `include_plotlyjs="cdn"`. To remove external dependencies, bundle a fixed `plotly.min.js` under `site/vendor/` and switch to `include_plotlyjs=False`.
- Dependabot monitors `pip` and `github-actions` weekly.

---

## Directory layout
```
.
├── data/        # input CSVs
├── lib/         # entropy / temperature / loop_area
├── scripts/     # 01: fetch (optional), 02: indicators, 03: report
├── site/        # outputs (report.html, PNGs, archive.json, feed.xml, sitemap.xml, robots.txt)
└── .github/workflows/  # Build & Publish workflow
```

---

## FAQ
**Q. Isn’t this claiming “money is entropy”?**  
A. No. `S_M` is an **entropy‑like extensive index** (`scale × dispersion`). It is **isomorphic** to ideal mixing but **not physically identical** to thermodynamic entropy.

**Q. What if I don’t like the product `M_in * H(q)`?**  
A. Report `(M_in, H(q))` as a 2‑axis dashboard, or use a normalized variant `S*_M = M_in * H(q)/ln(K)`.

**Q. What categories should I use?**  
A. Start with 6–12 purpose buckets (e.g., real estate, capex, working capital, inventories, durables, services, export, import). Keep them stable and MECE.

---

## License
See `LICENSE`.


# Content of 2025_11_Thermo_Credit/scripts/03_make_report.py after the block that builds data_defs_html:

# --- Optional Sources / Provenance table ---
sources_html = ""
for path in ["data/sources.json", "sources.json"]:
  if os.path.exists(path):
      try:
          with open(path, "r", encoding="utf-8") as f:
              sources = json.load(f)
          if isinstance(sources, list) and len(sources) > 0:
              src_df = pd.DataFrame(sources)
              keep_cols = [c for c in ["id","title","provider","country","frequency","units","url","note"] if c in src_df.columns]
              if keep_cols:
                  sources_html = "<h2>Sources</h2>" + src_df[keep_cols].to_html(index=False, border=0, classes="mini", escape=True)
          break
      except Exception as e:
          sources_html = "<h2>Sources</h2><p>Could not load sources.json: " + html_lib.escape(str(e)) + "</p>"
          break

# --- Optional: plot enabled raw input series (from sources.json) ---
raw_fig_html = ""
raw_png_path = None
try:
    sources2 = None
    for path in ["data/sources.json", "sources.json"]:
        if os.path.exists(path):
            with open(path, "r", encoding="utf-8") as f:
                cand = json.load(f)
                if isinstance(cand, list):
                    sources2 = cand
                    break
    raw_series = []
    if sources2:
        for src in sources2:
            if not isinstance(src, dict):
                continue
            if not src.get("enabled", False):
                continue
            sid = src.get("id") or src.get("title") or "series"
            title = src.get("title") or sid
            # Prefer explicit path; otherwise assume data/<id>.csv
            path = src.get("path") or os.path.join("data", f"{sid}.csv")
            if not os.path.exists(path):
                continue
            try:
                df_raw = pd.read_csv(path)
                # Expect columns: date, value (case-insensitive)
                date_col = next((c for c in df_raw.columns if str(c).lower() == "date"), None)
                val_col = next((c for c in df_raw.columns if str(c).lower() == "value"), None)
                if date_col is None or val_col is None:
                    continue
                ser = df_raw[[date_col, val_col]].dropna()
                ser[date_col] = pd.to_datetime(ser[date_col])
                ser = ser.sort_values(date_col)
                ser = ser.rename(columns={date_col: "date", val_col: title})
                raw_series.append(ser.set_index("date")[title])
            except Exception:
                continue
    if raw_series:
        raw_df = pd.concat(raw_series, axis=1).sort_index()
        # Normalize each series to 100 at its first non-NA observation for comparability
        norm_df = raw_df.apply(lambda s: 100.0 * s / s.dropna().iloc[0] if s.dropna().size else s)
        norm_df = norm_df.reset_index().rename(columns={"index": "date"})
        # Build interactive figure
        import plotly.express as px  # already imported above; safe
        fig_raw = px.line(norm_df, x="date", y=[c for c in norm_df.columns if c != "date"],
                          title="Raw Input Series (Enabled) — normalized to 100 at first observation",
                          labels={"value": "Index (=100 at first obs)", "date": "Date", "variable": "Series"})
        raw_fig_html = (
            f"<h2>Raw inputs (enabled)</h2>"
            f"<figure aria-label='Normalized raw input series'>"
            f"{fig_raw.to_html(full_html=False, include_plotlyjs='cdn')}"
            f"<figcaption>Toggle legend to select series.</figcaption></figure>"
        )
        # Try writing a PNG fallback
        try:
            raw_png_path = "site/fig_raw.png"
            fig_raw.write_image(raw_png_path, scale=2, width=1280, height=720)
        except Exception:
            raw_png_path = None
except Exception:
    pass

# In the section where figs_html is built (joining figs_html list), add this line after joining:
figs_html = figs_html + ("\n" + raw_fig_html if raw_fig_html else "")

# In the noscript PNG fallback block, after {extra_png} insert:
#         {('<figure><img src="fig_raw.png" alt="Raw input series (normalized)" width="100%"/><figcaption>Raw inputs (enabled)</figcaption></figure>' if raw_png_path else '')}

# In the section that duplicates PNGs into the month folder, after copying fig5.png add:
#     if raw_png_path and os.path.exists("site/fig_raw.png"):
#         shutil.copyfile("site/fig_raw.png", os.path.join(month_dir, "fig_raw.png"))
