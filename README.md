# Thermo‑Credit Monitor (TQTC)

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.17778342.svg)](https://doi.org/10.5281/zenodo.17778342)

Current version: v2.0 (Dec 2025)

Compute thermo‑credit indicators from public statistics (local CSVs or FRED API) and render a monthly, multi‑region dashboard.

> Experimental prototype of the Thermo‑Credit framework. Indicators are for research only; see the Zenodo note for theory and limitations.
Live dashboard:
https://toppymicros.com/2025_11_Thermo_Credit/report.html

Key outputs (per region):
- `site/report.html` — interactive dashboard (Plotly)
- `site/indicators*.csv` — indicator time series
- `site/credit_destination*.csv` — borrower-composition panels (JP primary: Bezemer–Samarina–Zhang Japan crosswalk; legacy G/B/E remains Appendix-only)
- `site/lambda_b_sensitivity.csv` — fixed-grid `lambda_B` forecast sensitivity results
- `site/destination_oos_incremental.csv` — focused JP borrower-composition OOS use example against matched BOJ mapped-stock growth
- `site/baseline_forecast_comparison.csv` — auxiliary OOS baseline comparison against AR(1), total-credit, spread, money, FCI, and pooled FE baselines
- `site/calibration_holdout_test.csv` — fixed and rolling holdout tests for calibrated headroom vs raw `X_C`
- `site/entropy_partition_robustness.csv` — 3/5/7 bucket entropy robustness with negative controls
- `site/tl_robustness.csv` — liquidity-state specification robustness and monotonicity checks
- `site/loop_area_null_tests.csv` — loop-area block, phase, AR surrogate, event-date, and placebo null tests
- `site/integrability_synthetic_test.csv` — Maxwell/vorticity synthetic estimator validation
- `site/submission_readiness.csv` — strict next-revision acceptance gates and current pass/fail status
- `data/*.csv` — intermediate feature tables
---

## Citation
If you use this repository, please cite the Zenodo record:

- DOI: https://doi.org/10.5281/zenodo.17563221
---

## Quick start

```bash
python3 -m venv .venv && source .venv/bin/activate
python -m pip install -U pip
### JP‑specific knobs

- `JP_START` (env var): earliest JP date for raw series when building indicators (e.g. `2012-01-01`).
  - Affects raw series ingestion only (BoJ assets, M2, yields, etc.).
  - If missing, scripts reuse committed CSVs under `data/`.
- `REPORT_PLOT_START`: plot window start only (does **not** change computed indicators).
## Multi‑region usage (JP / EU / US)

JP (default):

```bash
python scripts/02_compute_indicators.py
python scripts/03_make_report.py
---

## Configuration overview

Base config: `config.yml`
Region overrides: `config_jp.yml`, `config_eu.yml`, `config_us.yml`
  - Enrichment edge cases & fallbacks:
    - All‑NaN or entirely missing depth/turnover sources: heuristic fallback engaged (depth scaled by median credit stock; turnover from `U / L_real` with safe division).
    - All‑zero `L_real`: depth defaults to a constant (1000) and turnover falls back to 1.0 before clipping to bounds.
    - Clipping diagnostics: if > `turnover_clip_warn_threshold` fraction of rows are clipped (default 15%), a warning is collected.
    - Fallback constants and toy regression guards live under `enrichment` (`depth_fallback`, `turnover_fallback`, `depth_toy`, `turnover_toy`) so you can tune them per region (override in `config_jp.yml`, etc.).
    - Toy baselines (`L_asset_toy`, `depth_toy`, `turnover_toy`) are ensured during indicator build for regression protection.
  - Branding: `BRAND_BG`, `BRAND_BG2`, `BRAND_TEXT` (header/footer brand colors).

Detailed descriptions of external coupling (`E_p`, `E_T`) and chemical potentials (μ, Δμ) have been moved to `docs/external_coupling.md`.

---

## Data & sources
---

## CI

Workflow: `.github/workflows/build_report.yml`
  - Dependencies install → logo optimization → report build → upload `site/` as artifact
## Tips
- PNG fallback export requires `kaleido`.
- Recommend excluding generated artifacts (`site/`) and large CSVs from Git (CI regenerates them).

---

## License
See `LICENSE`.

# Thermo‑Credit Monitor (TQTC)

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.17563221.svg)](https://doi.org/10.5281/zenodo.17563221)

Compute thermo‑credit indicators from public statistics (local CSVs or FRED API) and render a monthly, multi‑region report.

> This dashboard is an experimental implementation of the Thermo‑Credit framework. All values are prototype indicators; see the Zenodo technical note for definitions and limitations.

Dashboard(Active):
https://toppymicros.com/2025_11_Thermo_Credit/report.html


Core metrics:
- S_M — Monetary Dispersion Entropy (entropy‑like, extensive)
- T_L — Liquidity state index
- loop_area — Policy/Regulatory Loop Dissipation (PLD)
- F_C — Helmholtz‑like Free Energy (F_C = U − T0 · S_M)
- X_C — Credit Exergy Ceiling (needs baselines; falls back to F_C when absent). By default, X_C is made non‑negative by clipping negatives at 0 (configurable via `exergy_floor_zero` and `exergy_floor_mode`).
- ΔF_C, X_C_plus, X_C_minus — Fixed‑reference split of free energy: ΔF_C(t)=F_C(t)−F_C_ref, X_C_plus=max(0,ΔF_C) (surplus/room), X_C_minus=max(0,−ΔF_C) (shortage). By design X_C_plus·X_C_minus=0.

Artifacts: `site/report.html` (Plotly interactive + PNG fallbacks), `archive.json`, `feed.xml`, `sitemap.xml`, `robots.txt`.
The indicator build also emits `site/credit_destination*.csv` panels. For JP,
`q_t` is the four-quarter non-financial-business coordinate of the
Bezemer–Samarina–Zhang Japan crosswalk. Signed stock changes are aggregated
within each of its four buckets before the positive part is applied.

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

### Borrower-composition panels

- `scripts/02_compute_indicators.py` now writes destination panels for every region: `site/credit_destination.csv`, `site/credit_destination_eu.csv`, `site/credit_destination_us.csv`, plus matching `_realtime.csv` versions.
- JP reads `data/credit_destination_jp.csv`, generated from BOJ Time-Series Data Search database LA01. The primary published crosswalk has four buckets: non-financial business (NFB), financial business (FIN), property/mortgage (PROP), and household non-housing (HH_NONHOUSING). Net changes, the former series-first construction, sector-level deltas, and BOJ purpose-coded comparisons are retained as audit columns.
- EU/US panels still estimate `C_t` from nonnegative quarterly changes in `L_real`, then map coarse allocation shares into `C_G` (productive lending), `C_B` (construction/development), and `C_E` (existing real estate, land, and financial claims). Treat them as schema-portability and negative-control panels, not cross-country evidence.
- The JP primary measure is `q_t = sum_4q(C_NFB) / sum_4q(C_NFB + C_FIN + C_PROP + C_HH_NONHOUSING)`. The one-quarter share remains an audit column. It is a borrower-composition measure, not loan purpose or destination. Legacy G/B/E, `C_R`, `C_A`, and `lambda_B` outputs are retained only for Appendix and compatibility paths.
- Scale is `primary_included_stock`, the same population as the four allocation buckets: BOJ total loans of domestically licensed banks less local-government loans. “Domestically licensed” describes lenders, not borrower geography; the residual NFB bucket includes a disclosed overseas-linked component. `mapped_domestic_stock` is a legacy alias and is not the primary scale.
- The Werner-inspired BOJ proxy and the Müller–Verner tradable/non-tradable adaptation are reported jointly over that same included population. Construction is NFB in the primary crosswalk, part of the Werner-inspired proxy, and non-tradable in the Müller–Verner adaptation.

### BOJ bridge internal audit

- `scripts/18_boj_bridge_validation.py` generates the manuscript BOJ source mapping and internal-audit tables.
- Outputs:
  - `tex/generated/theory_boj_bridge_mapping_main.tex`: compact main-text BOJ source-group bridge table.
  - `tex/generated/theory_boj_bridge_mapping.tex`: detailed appendix table with BOJ source series, borrower sectors, bucket mapping, ambiguous cases, bias direction, negative-change treatment, release lag, and coverage/audit share.
  - `tex/generated/theory_boj_bridge_validation.tex`: classification-granularity, positive-versus-signed, quarter-availability, within-BOJ purpose-comparator, alternative-allocation, common-series, and aggregate-reconciliation audits.
  - `data/boj_bridge_validation_summary.json`: machine-readable table backing.

### `lambda_B` sensitivity

- `scripts/09_lambda_b_sensitivity.py` recomputes the destination split for `lambda_B` in `{0, 0.25, 0.5, 0.75, 1}` without estimating a region-specific construction weight. In the manuscript this is treated as an internal BOJ bridge sensitivity audit, not as a main forecast claim.
- Outputs:
  - `site/credit_destination_lambda_b_sweep.csv`: per-date, per-region `C_R`, `C_A`, and `q_t` under each grid value.
  - `site/lambda_b_sensitivity.csv`: expanding-window OOS forecast diagnostics for real-growth and asset-acceleration proxy targets.
  - `tex/generated/theory_lambda_b_sensitivity.tex`: manuscript table.
- Asset acceleration uses `L_asset` when no direct asset-price series is available. Results that are `lambda_B`-sensitive, target-unstable, or fail to improve on the total-credit baseline should not be used as main claims.

### Borrower-composition pseudo-OOS application

- `scripts/19_destination_oos_incremental.py` is a use example for the bridge, not a forecast-improvement claim.
- The manuscript table reports matched BOJ included-stock growth plus all three four-quarter literature-anchored coordinates; `1-q_t` is retained only as an algebraic identity check in the companion CSV.
- The main target is the long-term JGB yield change at 4Q and 8Q horizons. BOJ total-assets acceleration is appendix-only; nominal GDP is omitted because the supplied input is annual.
- Training labels are purged until their forward outcomes are realised. The 2009Q2 cross-classification change is invalid; the first valid flow is 2009Q3 and is usable after its release lag. The companion CSV also reports AR(1), spread/FCI, standalone, and aggregation-order sensitivity rows.
- Outputs:
  - `site/destination_oos_incremental.csv`: full focused OOS grid, including benchmark and destination rows.
  - `data/destination_oos_incremental_summary.json`: machine-readable summary.
  - `tex/generated/theory_destination_oos_incremental.tex`: manuscript table.
- None of the main JGB-yield four-quarter rows lowers point loss. Selected exploratory one-quarter or Appendix rows do, with intervals including zero; the current-vintage sample does not establish incremental predictability.

### Auxiliary baseline forecast comparison

- `scripts/10_baseline_forecast_comparison.py` evaluates whether the broader monitoring indicator set improves OOS forecasts beyond simple baselines.
- Targets include real activity growth, inflation when present, asset-acceleration proxies, spread widening, stress-regime events, lower-tail growth, and volatility spikes.
- Baselines include AR(1), total-credit growth, credit-to-GDP gap, spread-only, money growth, simple FCI, and a pooled region fixed-effect panel using within-region expanding z-scores.
- The indicator builder harmonizes GDP-like `Y`/`U` levels to the local credit scale when raw source units are many orders of magnitude apart, which prevents US World Bank dollar levels from being mixed with billion-scale credit panels.
- Outputs:
  - `site/baseline_forecast_comparison.csv`: full baseline/candidate grid with RMSE, MAE, AUC, Brier score, log score, Diebold-Mariano p-values, and block-bootstrap CIs.
  - `site/baseline_forecast_target_coverage.csv`: target availability and source columns.
  - `data/baseline_forecast_summary.json`: manuscript-ready summary.
  - `tex/generated/theory_baseline_forecast_comparison.tex`: manuscript table.
- Current panels support downside-risk monitoring better than broad growth or asset-acceleration forecasting; lack of robust gains is a reason to keep theory claims modest.

### Calibration holdout test

- `scripts/11_calibration_holdout_test.py` tests whether the calibrated implicit headroom score beats raw pipeline `X_C`.
- The fixed split estimates `theta=(T0,p0,U0,V0,S0)` on observations available through 2015 and evaluates 2016-2025 forecast origins.
- The rolling split refits `theta` on the previous 40 quarters before each forecast origin.
- Outputs:
  - `site/calibration_holdout_test.csv`: tuned `X_C`, raw `X_C`, and simple trailing-change baseline metrics by region and strategy.
  - `data/calibration_holdout_summary.json`: machine-readable summary.
  - `tex/generated/theory_calibration_holdout.tex`: manuscript table.
- Current holdout results do not support a strong validation claim for the calibrated score; use it as a diagnostic overlay unless it beats raw `X_C` and the simple baseline in future panels.

### Entropy partition robustness

- `scripts/12_entropy_partition_robustness.py` tests whether normalized allocation entropy is driven by bucket design.
- It recomputes `S_M_hat` for borrower-label and loan-purpose partitions with 3, 5, and 7 buckets.
- Negative controls include shuffled shares, fixed shares, and random-walk shares.
- Outputs:
  - `site/entropy_partition_robustness.csv`: all observed/control metrics by region, partition, and bucket count.
  - `data/entropy_partition_robustness_summary.json`: region-level decision flags.
  - `tex/generated/theory_entropy_partition_robustness.tex`: manuscript table.
- Current panels are flat under every observed partition, so entropy remains a dashboard diagnostic and is excluded from main empirical evidence.

### TL robustness

- `scripts/13_tl_robustness.py` tests liquidity-state design dependence.
- Variants include multiplicative, additive z-score, soft-min, harmonic mean, spread-only, turnover-excluded, and depth-excluded indices.
- All reported variants use expanding-window log z-scores and are oriented so lower spreads, deeper markets, and higher turnover raise the score.
- Outputs:
  - `site/tl_robustness.csv`: variant metrics by region.
  - `data/tl_robustness_summary.json`: monotonicity and region summary.
  - `tex/generated/theory_tl_robustness.tex`: manuscript table.
- The signed raw-product multiplicative formula is rejected by monotonicity checks; the additive z-score index remains the main specification.

### Loop-area null tests

- `scripts/14_loop_area_null_tests.py` tests whether closed-loop area is larger than trend/autocorrelation null paths.
- Nulls include block shuffle, phase randomization, AR(1) surrogates, event-date permutation, and placebo periods outside registered events.
- The script reports latest 8-, 12-, and 16-quarter segmentation sensitivity plus registered event-window rows.
- Outputs:
  - `site/loop_area_null_tests.csv`: full null-test grid by region, window, and method.
  - `data/loop_area_null_summary.json`: region and overall counts.
  - `tex/generated/theory_loop_area_null_tests.tex`: manuscript table.
- Current panels do not support a standalone hysteresis claim; loop area remains a path-stress monitor unless it is extreme across null designs.

### Integrability synthetic test

- `scripts/15_integrability_synthetic_test.py` validates the Maxwell-like curl estimator on synthetic data.
- It generates `T_L` and `p_C` from a known quadratic potential, then varies noise, sampling frequency, a designed non-integrable field, and proxy misspecification.
- Outputs:
  - `site/integrability_synthetic_test.csv`: full scenario/noise/sampling grid.
  - `data/integrability_synthetic_summary.json`: clean-case pass/fail summary.
  - `tex/generated/theory_integrability_synthetic_test.tex`: manuscript table.
- Clean potential data return `Omega` near zero; designed vorticity is detected; proxy contamination raises curl as a consistency warning.

### Submission-readiness gates

- `scripts/16_submission_readiness.py` evaluates the next-revision acceptance line.
- Gates cover JP `q_t` versus total-credit OOS forecasts, calibrated `X_C` holdouts, richer sectoral/loan-purpose movement in `S_M_hat`, TL/loop robustness, and full reproducibility.
- Outputs:
  - `site/submission_readiness.csv`: gate-level status, blockers, and next actions.
  - `data/submission_readiness_summary.json`: pass count and recommended positioning.
  - `tex/generated/theory_submission_readiness.tex`: manuscript table.
- Current manuscript positioning is limited to a BOJ borrower-composition measure and its verifiable application. The thermodynamic extension is outside the maintained empirical argument.

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
  - `depth_proxy_us`, `turnover_proxy_us` (US overrides)
  - Scripts select these roles (env > config > defaults). Depth/turnover currently computed with heuristic scaling when real series absent; tests guard column presence per region.
  - Free-energy baseline (keeps `F_C` / `X_C` above zero without flattening):
    - `F_C_baseline_mode` (default `min`): `none`/`raw` to disable, or choose `min`/`quantile`/`value`/`first` to set the baseline reference.
    - `F_C_baseline_quantile` (default `0.05`): lower-tail quantile when `mode=quantile`.
    - `F_C_baseline_eps` (default `1e-6`): tiny lift added after shifting so the minimum is strictly positive.
  - Exergy floor controls (optional):
    - `exergy_floor_zero` (bool, default true): enforce non‑negative X_C.
    - `exergy_floor_mode` (string, `clip`|`shift`, default `clip`):
      - `clip`: clamp negative values to 0 (default; preferred for operations).
      - `shift`: add a constant offset so that min(X_C)=0 (use for visualization only).
  - Internal energy detrending (`U_detrend`):
    - `enabled` (default `true`): compute `U_trend` and `U_star = U - trend` without lookahead.
    - `method`: `rolling` (default) or `ema`.
    - `window` / `min_periods`: quarter-length smoothing window and minimum sample count.
    - These series propagate into diagnostics/tests so sudden jumps in U can be compared on a stationary baseline.
  - No-lookahead forecasting panels (`preprocessing.real_time_forecast`):
    - `enabled` (default `true`): in addition to the retrospective dashboard CSVs, emit release-lagged real-time panels such as `site/indicators_realtime.csv`, `site/indicators_eu_realtime.csv`, and `site/indicators_us_realtime.csv`.
    - `release_lags_days`: group or column lags applied before indicator construction. Defaults are GDP/U `90` days, credit/depth/turnover `120` days, spreads and market drivers `1` day, money `30` days, allocation shares `90` days, and regulatory/headroom inputs `90` days.
    - Each real-time build also writes `site/realtime_release_lags*.json` so the release-lag assumptions are auditable in a replication package.
    - Calibration reads the real-time panels by default (`CALIBRATION_PANEL_MODE=realtime`), while paper figures and the public dashboard continue to use the retrospective panels unless explicitly marked otherwise.
  - Credit destination proxy (`credit_destination`):
    - `enabled` (default `true`): emit `site/credit_destination*.csv` and merge the JP primary `C_NFB`, `C_FIN`, `C_PROP`, `C_HH_NONHOUSING`, `q_t`, and audit columns into `site/indicators*.csv`; legacy G/B/E fields remain for Appendix compatibility.
    - `lambda_B` (default `0.5`): construction/development weight assigned to the GDP-linked credit side.
    - `housing_construction_share` and `household_housing_share`: proxy shares used when direct loan-purpose data are absent.
    - `source` is set to `allocation_proxy` until richer sectoral or loan-purpose mappings are available.
  - Credit capacity ceiling (`V_C_formula`):
    - `min_headroom` (default) replaces `V_C` with the tightest regulatory buffer among `capital_headroom`, `lcr_headroom`, and `nsfr_headroom`.
    - `V_C_headroom_cols`: override the column names if your CSVs use different labels.
    - `V_C_headroom_scales`: fallback multipliers applied when explicit columns are missing (defaults to `[1.0, 1.0, 1.0]`).
    - Regional builders (`scripts/04_build_features_eu.py`, `scripts/05_build_features_us.py`) and placeholder generators now emit these headroom columns; when absent, the indicator build derives heuristic values from `p_R` and `V_R` so the pipeline stays stable.
  - Enrichment edge cases & fallbacks:
    - All‑NaN or entirely missing depth/turnover sources: heuristic fallback engaged (depth scaled by median credit stock; turnover from `U / L_real` with safe division).
    - All‑zero `L_real`: depth defaults to a constant (1000) and turnover falls back to 1.0 before clipping to bounds.
    - Clipping diagnostics: if > `turnover_clip_warn_threshold` fraction of rows are clipped (default 15%), a warning is collected.
    - Fallback constants and toy regression guards live under `enrichment` (`depth_fallback`, `turnover_fallback`, `depth_toy`, `turnover_toy`) so you can tune them per region (override in `config_jp.yml`, etc.).
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

### External pressure / liquidity-state coupling
- Configure under `external_coupling` in `config.yml` (override per region via `config_<region>.yml`). Keys:
  - `enabled`: master switch for computing the monthly driver composites.
  - `alpha` / `delta`: coupling coefficients applied to credit pressure (`p_C`) and the liquidity state index (`T_L`). Japan currently sets `alpha=0.2` while every region keeps `delta=0.0` until the liquidity coupling is validated.
  - `frequency`: monthly aggregation frequency (default `MS`).
  - `pressure_components` / `temperature_components`: driver specs with `id`, optional `id_b` (for spreads), `transform`, `scale`, and `key`. Defaults pull US stress proxies: HY OAS, US–JP 10Y yield spread, USDJPY log returns, and VIX. The MOVE index is defined in `data/sources.json` but is temporarily disabled in `config.yml` because it is not available via the FRED JSON API; you can re-enable it once you wire a local `data/MOVE.csv` or alternate data source.
- `scripts/01_build_features.py` invokes `lib.external_coupling.build_external_coupling_indices` to fetch the configured drivers (FRED IDs), expanding-window z-score them monthly, and compute composite indices `E_p` / `E_T`. Raw driver CSVs plus `data/external_coupling_<region>.csv` are persisted for reproducibility.
- The resulting `E_p` / `E_T` columns are merged into `data/reg_pressure.csv`. During indicator construction, `lib.indicators.build_indicators_core` records baseline `p_C` / `T_L`, then applies the coupling contributions (`p_C ← p_C + α·E_p`, `T_L ← T_L + δ·E_T`). Diagnostic columns (`p_C_baseline`, `E_p_contrib`, `T_L_baseline`, `E_T_contrib`) remain in `site/indicators*.csv` so you can audit the effect or dial coefficients back to zero.
- If coupling is enabled but both coefficients equal zero, the build is equivalent to the disabled state (guarded by `tests/test_external_coupling.py`).

### Chemical potentials per allocation bucket
  - `q_cols` determines which allocation buckets receive potentials (defaults to the MECE set in `config.yml`).
  - `mu_share_floor` (optional, default `1e-6`) clips very small shares before taking logarithms to keep `\mu` finite.
  - The build also derives a time-varying cross-bucket mean `mu_mean` and relative spreads `dmu_<bucket> = mu_<bucket> - mu_mean`. These `\Delta\mu_i` columns are centered by construction (they sum to zero across buckets each date) and act as dimensionless drivers for future flow experiments.

---

## Data & sources
- By default the scripts read `data/*.csv`; `scripts/02_compute_indicators*.py` compute both retrospective dashboard indicators and release-lagged real-time forecast indicators.
- JP credit-destination outputs use `data/credit_destination_jp.csv`, a BOJ LA01 sectoral-credit bridge generated by `scripts/17_fetch_boj_jp_credit_destination.py`; EU/US outputs remain allocation-proxy negative-control panels until replaced or validated with direct loan-purpose data.
- BOJ source citation: Bank of Japan, Time-Series Data Search, database LA01, “Loans and Bills Discounted by Sector (Outstanding, Loans for Fixed Investment),” Domestically Licensed Banks.
- Provide `data/sources.json` to show a Sources table and Raw Inputs figure.
- Provide BIS private credit series directly: `CRDQJPAPABIS`, `CRDQEZAPABIS`, and `CRDQUSAPABIS` are configured under `data/sources.json` (Quarterly, billions of local currency) so enrichment depth proxies point to real BIS tables out of the box.
  - Source citation: Bank for International Settlements (BIS), “Credit to the Private Non-Financial Sector,” series CRDQJPAPABIS / CRDQEZAPABIS / CRDQUSAPABIS (BIS Data Portal). If you download the same identifiers via FRED, cite both the original BIS source and FRED as the distribution host (BIS also recommends dual attribution / FRED経由取得時は一次出所=BIS・ホスト=FREDを併記)。
- Turnover proxies rely on existing liquidity series (JP: `MYAGM2JPM189S`, EU: `ECBASSETS`, US: `WALCL`) so no placeholder entries remain.
- A compact `data/credit.csv` (JP) placeholder is committed so tests work out-of-the-box. Replace it with fresh features from `scripts/01_build_features.py` when running the full pipeline; the file only contains a handful of rows covering the enrichment-required columns (`L_asset`, `depth`, `turnover`, toy baselines, etc.)
- Matching stubs for `data/credit_eu.csv` and `data/credit_us.csv` now ship with the repo so EU/US enrichment suites also run without pulling the full historical datasets. Overwrite them with outputs from `scripts/04_build_features_eu.py` / `scripts/05_build_features_us.py` when you need live data.
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

# Two-pass replication check: input CSV hashes + regenerated indicators,
# calibration table, BOJ bridge validation, lambda_B sensitivity, focused
# borrower-composition OOS tests, auxiliary baseline forecasts, calibration holdouts,
# entropy/TL robustness, and paper figures
# with numeric tolerances.
python scripts/08_reproducibility_check.py --report-dir replication

# Forward-compat (latest) check
python -m pip install -U pip
pip install -r requirements.txt
pytest -q
```

The reproducibility check writes `replication/reproducibility_manifest.json`
and `replication/reproducibility_log.md`. It hashes all input CSVs, regenerates
the retrospective and real-time `site/indicators*.csv` panels,
`site/credit_destination*.csv`, `data/calibrated_theory_params.json`, Table 3
snippets, BOJ bridge-validation outputs, the `lambda_B` sensitivity outputs,
focused borrower-composition OOS outputs, auxiliary baseline forecast outputs,
calibration holdout outputs,
entropy/TL/loop/integrability robustness outputs, submission-readiness outputs,
and paper figure assets twice, then
fails if numeric outputs differ beyond `rtol=1e-9, atol=1e-8` or deterministic
text/binary artifacts differ.

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
