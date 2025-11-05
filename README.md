# Thermo‑Credit Monitor (TQTC) — Minimal MVP

This repository computes four thermo‑credit indicators from public statistics (local CSVs or FRED API) and publishes a monthly report to `site/`:

- **S_M — Monetary Dispersion Entropy (entropy‑like, extensive)**
- **T_L — Liquidity “Temperature” (normalized liquidity score)**
- **PLD — Policy/Regulatory Loop Dissipation (`loop_area`)**
- **X_C — Credit Exergy Ceiling**

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
- **X_C**: `X_C = U - T0 * S_M` (exergy‑like ceiling under background liquidity `T0`)

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
 (optional) scripts/01_build_features.py  # online fetch if FRED_API_KEY is present
```
- Without `FRED_API_KEY`, the fetch step is skipped and local CSVs are used (you will see: `No FRED_API_KEY; skip online fetch ...`).
- Outputs: `indicators.csv`, `report.html`, `fig1.png..fig3.png`, `archive.json`, `feed.xml`, `sitemap.xml`, `robots.txt`.

### Current FRED series used
Defined in `scripts/01_build_features.py`.

| series_id       | Description                 | Note |
|-----------------|-----------------------------|------|
| MYAGM2JPM189S   | Japan M2, monthly (via FRED)| Skipped if no API key; falls back to `data/` CSVs |

Add more series in `01_build_features.py` and update this table accordingly.

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
