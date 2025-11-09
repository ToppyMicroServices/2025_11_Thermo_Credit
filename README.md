# Thermo‑Credit Monitor (TQTC)

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.17563221.svg)](https://doi.org/10.5281/zenodo.17563221)

Compute thermo‑credit indicators from public statistics (local CSVs or FRED API) and render a monthly, multi‑region report.

> This dashboard is an experimental implementation of the Thermo‑Credit framework. All values are prototype indicators; see the Zenodo technical note for definitions and limitations.

Core metrics:
- S_M — Monetary Dispersion Entropy (entropy‑like, extensive)
- T_L — Liquidity “Temperature”
- loop_area — Policy/Regulatory Loop Dissipation (PLD)
- F_C — Helmholtz‑like Free Energy (F_C = U − T0 · S_M)
- X_C — Credit Exergy Ceiling (needs baselines; falls back to F_C when absent)

Artifacts: `site/report.html` (Plotly interactive + PNG fallbacks), `archive.json`, `feed.xml`, `sitemap.xml`, `robots.txt`.

---

## Citation
If you use this repository, please cite the Zenodo record:

- DOI: https://doi.org/10.5281/zenodo.17563221

---

## Quick start
```bash
python3 -m venv .venv && source .venv/bin/activate
python -m pip install -U pip
pip install -r requirements.txt

# (Optional) PNG fallbacks for charts
pip install kaleido

# Build JP indicators and report
python scripts/02_compute_indicators.py
python scripts/03_make_report.py
open site/report.html   # macOS
```

---

## Multi‑region (JP / EU / US)
- EU:
  1) (Optional) Fetch/assemble inputs: `python scripts/04_build_features_eu.py`
  2) Compute: `python scripts/02_compute_indicators_eu.py` → `site/indicators_eu.csv`
- US:
  1) (Optional) Fetch/assemble inputs: `python scripts/05_build_features_us.py`
  2) Compute: `python scripts/02_compute_indicators_us.py` → `site/indicators_us.csv`

If `site/indicators_eu.csv` / `site/indicators_us.csv` exist, region tabs (and the Compare tab) are added to the report automatically.

Notes
- While some regions lack calibrated `X_C`, charts automatically fall back to `F_C`.
- The Raw Inputs figure overlays all `enabled: true` series from `data/sources.json` normalized by first=100 (shared tab across regions).

---

## Configuration
- Base config: `config.yml`
- Region overrides: `config_jp.yml`, `config_eu.yml`, `config_us.yml`
- Environment variables
  - `FRED_API_KEY` (optional): FRED API key (if absent, fall back to local CSVs)
  - `REPORT_PLOT_START` (optional): Start date for plot range (example: `2010-01-01`)
  - `CONFIG_REGION` (optional): Region override (`jp` / `eu` / `us`)
  - Branding: `BRAND_BG`, `BRAND_BG2`, `BRAND_TEXT` (header/footer brand colors)

---

## Branding & logo
- The logo is embedded inline as a Base64 data URI.
- To pre-optimize (recommended):
  ```bash
  python scripts/optimize_logo.py --height 80 --colors 96
  ```
  Output: `scripts/og-brand-clean.min.png` (used preferentially if present)
- Brand colors can be overridden via environment variables:
  ```bash
  export BRAND_BG="#0d1b2a"
  export BRAND_BG2="#1b263b"
  export BRAND_TEXT="#ffffff"
  python scripts/03_make_report.py
  ```

---

## Data & sources
- By default the scripts read `data/*.csv`; `scripts/02_compute_indicators*.py` compute indicators.
- Provide `data/sources.json` to show a Sources table and Raw Inputs figure.
  - Example entry: `{"id":"JPNASSETS","title":"BoJ Total Assets","enabled":true}`
  - Default CSV path is `data/<id>.csv` (columns: `date`, `value`). Use `path` to override.

---

## CI
- Workflow: `.github/workflows/build_report.yml`
  - Dependencies install → logo optimization → report build → upload `site/` as artifact
- (Optional) Add `.github/workflows/update_data.yml` for daily raw data refresh + PR creation.

---

## Tips
- PNG fallback export requires `kaleido`.
- VS Code: commit shared settings (`.vscode/extensions.json`) as needed; ignore others.
- Recommend excluding generated artifacts (`site/`) and large CSVs from Git (CI regenerates them).

---

## License
See `LICENSE`.

---

### Yield comparison (US vs JP)
Helper script to quickly compare US vs JP long yields:
```bash
python scripts/compare_yields.py --start 1995-01-01 --png
```
Outputs: `site/yield_compare.html` (interactive), optional `site/yield_compare.png`, `site/yield_compare_metrics.csv` (spread & rolling correlation).
