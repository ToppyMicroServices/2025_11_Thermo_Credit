# Thermo‑Credit Monitor (TQTC)

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.17563221.svg)](https://doi.org/10.5281/zenodo.17563221)

Compute thermo‑credit indicators from public statistics (local CSVs or FRED API) and render a monthly, multi‑region report.

> This dashboard is an experimental implementation of the Thermo‑Credit framework. All values are prototype indicators; see the Zenodo technical note for definitions and limitations.

Dashboard(Active):
https://toppymicros.com/2025_11_Thermo_Credit/report.html


Core metrics:
- S_M — Monetary Dispersion Entropy (entropy‑like, extensive)
- T_L — Liquidity “Temperature”
- loop_area — Policy/Regulatory Loop Dissipation (PLD)
- F_C — Helmholtz‑like Free Energy (F_C = U − T0 · S_M)
- X_C — Credit Exergy Ceiling (needs baselines; falls back to F_C when absent). By default, X_C is made non‑negative by clipping negatives at 0 (configurable via `exergy_floor_zero` and `exergy_floor_mode`).
- ΔF_C, X_C_plus, X_C_minus — Fixed‑reference split of free energy: ΔF_C(t)=F_C(t)−F_C_ref, X_C_plus=max(0,ΔF_C) (surplus/room), X_C_minus=max(0,−ΔF_C) (shortage). By design X_C_plus·X_C_minus=0.

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

### JP_START and REPORT_PLOT_START

- JP_START (env var) trims the raw Japan time series before any aggregation. Use it to align the model start with the policy regime of interest.
  - Example: set `JP_START=2012-01-01` to exclude earlier observations when building indicators.
  - Scope: affects raw series ingestion only (BoJ assets, M2, yields when available). If missing, script falls back to prebuilt `data/money.csv` and no trim is applied.
- REPORT_PLOT_START controls visualization range only. It does not change computed indicators; it just limits what plots show.
- Interaction: If both are set, JP_START changes the underlying series and derived indicators; REPORT_PLOT_START simply hides earlier computed points in charts.

### Entropy outputs (MECE)

- The allocation shares `q_*` used for entropy are now emitted to `site/indicators.csv` for transparency and recomputation.
- Normalized entropy `S_M_hat` (Shannon H divided by log(K)) is included for scale-free comparisons alongside `S_M` (= k · M_in · H).
- Categories are configured via `config.yml:q_cols` (default: `q_productive`, `q_housing`, `q_consumption`, `q_financial`, `q_government`).
- Housing split can be adjusted with `JP_Q_HOUSING_SHARE` if you need a different ratio for the housing component.

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
  - Credit enrichment keys (JP baseline + EU/US extensions) live in `config.yml`:
    - `asset_proxy`, `energy_proxy`, `depth_proxy`, `turnover_proxy` (Japan defaults)
    - `depth_proxy_eu`, `turnover_proxy_eu` (Euro area overrides)
    - Planned: `depth_proxy_us`, `turnover_proxy_us` (add when US enrichment sources defined)
  - Scripts select these roles (env > config > defaults). Depth/turnover currently computed with heuristic scaling when real series absent; tests guard column presence per region.
  - Exergy floor controls (optional):
    - `exergy_floor_zero` (bool, default true): enforce non‑negative X_C.
    - `exergy_floor_mode` (string, `clip`|`shift`, default `clip`):
      - `clip`: clamp negative values to 0 (default; preferred for operations).
      - `shift`: add a constant offset so that min(X_C)=0 (use for visualization only).
  - Enrichment edge cases & fallbacks:
    - All‑NaN or entirely missing depth/turnover sources: heuristic fallback engaged (depth scaled by median credit stock; turnover from `U / L_real` with safe division).
    - All‑zero `L_real`: depth defaults to a constant (1000) and turnover falls back to 1.0 before clipping to bounds.
    - Clipping diagnostics: if > `turnover_clip_warn_threshold` fraction of rows are clipped (default 15%), a warning is collected.
    - Toy baselines (`L_asset_toy`, `depth_toy`, `turnover_toy`) are ensured during indicator build for regression protection.
- Environment variables
  - `FRED_API_KEY` (optional): FRED API key (if absent, fall back to local CSVs)
  - `JP_START` (optional): Earliest JP date for raw series when building indicators (default `2012-01-01`). Example:
    ```bash
    export JP_START=2008-01-01
    python scripts/02_compute_indicators.py
    ```
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
  - World Bank integration (GDP indicator fetch):
    - Shared helper: `lib.worldbank.fetch_worldbank_series(country, indicator)` centralizes caching & retries.
    - Caching: JSON cache file per (country, indicator) in `data/worldbank_cache_<country>_<indicator>.json`.
    - Retry logic: exponential backoff; if live API & cache fail, scripts attempt configured fallback CSV list.
    - Alignment: series are converted to quarterly using explicit `QE-DEC` resampling for consistency with other data.
    - Indicator string may include an optional `@YYYY-MM-DD` suffix to specify a custom observation start.

---

## Entropy categories (MECE)
- Monetary entropy now defaults to a five-way MECE split stored in `data/allocation_q.csv`:
  - `q_productive` (non-financial firms)
  - `q_housing` (household housing allocation)
  - `q_consumption` (household consumption allocation)
  - `q_financial` (financial system assets)
  - `q_government` (public balance sheet)
- The split is derived from legacy columns (`q_pay`, `q_firm`, `q_asset`, `q_reserve`). By default, household share is split 40% housing / 60% consumption. Override with `JP_Q_HOUSING_SHARE=0.35` (for example) before rebuilding features if you prefer a different ratio.
- `config.yml` sets `q_cols` to the MECE columns and enables `entropy_per_category: true`, which creates per-category entropy flows (`S_M_in_<category>`). These show up in the report as a stacked chart.
- If you want to experiment with a different schema, edit `data/allocation_q.csv` and update `config.yml` accordingly. Ensure the selected columns are positive and sum to ~1 per quarter.

---

## CI
- Workflow: `.github/workflows/build_report.yml`
  - Dependencies install → logo optimization → report build → upload `site/` as artifact
- (Optional) Add `.github/workflows/update_data.yml` for daily raw data refresh + PR creation.

### Dependency & Update Strategy

This repository tests two dependency modes to balance stability with forward compatibility:

1. Pinned (reproducible) mode
   - Exact versions recorded in `constraints.txt` (includes tooling: pytest, pip-audit, cyclonedx-bom).
   - CI installs with: `pip install -r requirements.txt -c constraints.txt`.
   - Security audit (`pip-audit`) runs in strict mode against pinned set; build fails on actionable vulnerabilities.
   - SBOM generated: `cyclonedx-py requirements constraints.txt -o sbom-<py>.json`.

2. Latest (exploratory) mode
   - Uses `requirements.txt` (unpinned top-level libraries) to catch upstream changes early.
   - Non-strict `pip-audit` (informational) allows temporary issues without failing the PR.

CI Matrix (`.github/workflows/matrix-ci.yml`):
- Python versions: 3.10, 3.11
- Modes: `pinned`, `latest`
- Ensures tests & entropy normalization stay stable under new upstream releases.

Renovate (`renovate.json`):
- Weekly schedule (before 05:00 JST Monday) for dependency update PRs.
- Group rules:
  - `plotly` + `kaleido` under `plotly-stack`
  - `pandas` + `numpy` under `core-numeric` (longer stabilityDays)
- Regex manager surfaces pinned versions in `constraints.txt` for bump proposals.
- Dashboard enabled for visibility; rate limits prevent PR spam.

Local workflows:
```bash
# Reproducible environment
python -m pip install -U pip
pip install -r requirements.txt -c constraints.txt

# Forward-compat (latest) check
python -m pip install -U pip
pip install -r requirements.txt
pytest -q
```

Security artifacts:
- SBOM: uploaded as workflow artifact (`sbom-*.json`).
- Secret scanning: regex scan + gitleaks v8 in CI.
- Vulnerability gate: pinned mode only; latest mode is advisory.

Upgrade guidance:
- For major version bumps producing test or audit failures: adjust code/tests first, then update `constraints.txt`. Avoid merging broken pinned builds—stability first.
- Keep `requirements.txt` minimal; add new runtime libs there, and mirror a pinned version in `constraints.txt`.

To freeze current working set after adding a new library:
```bash
pip install <newlib>
python -c 'import importlib, pkgutil; import pkg_resources; print("pandas=="+pkg_resources.get_distribution("pandas").version)'  # example introspection
# Manually append exact version to constraints.txt
```

Troubleshooting:
- If matrix latest fails only due to upstream regression, open an issue and optionally add a temporary ignore rule in Renovate or constraints override.
- If SBOM generation fails, verify `cyclonedx-bom` is present in `constraints.txt` and installed.

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
