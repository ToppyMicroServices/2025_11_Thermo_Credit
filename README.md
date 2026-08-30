# Thermo Credit

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.17563220.svg)](https://doi.org/10.5281/zenodo.17563220)

Thermo Credit is a reproducible research project on credit scale, borrower
composition, and experimental macro-financial diagnostics. It publishes a
quarterly dashboard for Japan, the euro area, and the United States, together
with the paper, source mappings, robustness tests, and recalculation code.

[Open the dashboard](https://toppymicros.com/2025_11_Thermo_Credit/report.html)
| [日本語](README_JP.md)

## Evidence boundary

The strongest current result is a Japanese borrower-composition measurement
bridge built from Bank of Japan sectoral loan stocks. Its primary four buckets
follow the published Japan crosswalk of Bezemer, Samarina, and Zhang. Scale and
composition use the same reconciled loan population.

The scalar `q_t` is the four-quarter non-financial-business coordinate of that
borrower composition. It is not a direct measure of loan purpose, GDP-linked
credit, or final expenditure. EU and US panels use coarser proxies and test
schema portability; they are not cross-country validation of the JP measure.

`S_M`, `T_L`, `p_C`, `U`, `F_C`, `X_C`, and `loop_area` are experimental model
diagnostics. In particular, `X_C` is not a validated safety margin, policy
threshold, or forecast. The current pseudo-out-of-sample results do not
establish incremental predictability over the matched credit-stock baseline.

See [definitions](docs/definitions.md), the
[identification strategy](docs/identification_strategy.md), and the
[calibration protocol](docs/calibration_protocol.md) before interpreting the
outputs.

## Quick start

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt -c constraints.txt

python scripts/02_compute_all_regions.py
python scripts/27_validate_site_data.py --min-rows 8 --max-age-days 550
TMS_SKIP_PNG=1 python scripts/03_make_report.py
pytest -q
```

The generated dashboard is `site/report.html`. Set `TMS_SKIP_PNG=1` when an
optional Plotly image backend is unavailable; the interactive HTML charts are
still generated.

## Machine-facing interfaces

Build the versioned static JSON API, or call the same evidence-labelled core
through the CLI:

```bash
python scripts/28_build_public_api.py
python scripts/thermo_credit_cli.py get_theory_overview
python scripts/thermo_credit_cli.py compute_thermo_credit_metrics --repo-region jp --limit 8
```

The static entry point is `site/api/v1/manifest.json`. An MCP server exposes
the definitions, tools, registered-event cases, and prompt templates:

```bash
python scripts/thermo_credit_mcp_server.py --transport stdio
```

See [the MCP interface](docs/thermo_credit_mcp_spec.md). GitHub Pages publishes
the read-only JSON API but cannot run MCP. HTTP deployment needs separate
authentication and operational controls.

## Data refresh

The all-region refresh fails if a regional build or validation step fails. The
main commands are:

```bash
python scripts/fetch_ecb_series.py
python scripts/fetch_fred_series.py
python scripts/01_build_features.py
python scripts/04_build_features_eu.py
python scripts/05_build_features_us.py
python scripts/02_compute_all_regions.py
python scripts/27_validate_site_data.py --min-rows 8 --max-age-days 550
```

`scripts/fetch_ecb_series.py` reads the current ECB Data Portal BSI total-assets
series. FRED downloads work with the public CSV endpoint and can use
`FRED_API_KEY` when configured. Do not use `--allow-partial` for a publication
refresh.

Source roles and units are listed in `data/data_dictionary.csv` and
`data/sources.json`. The machine-readable site manifest is generated at
`site/data_manifest.json`.

## Research outputs

- `tex/theory.tex`: source of record for the measurement paper.
- `site/report.html`: interactive regional dashboard.
- `site/indicators*.csv`: quarterly regional diagnostics.
- `site/credit_destination*.csv`: borrower-composition or regional proxy panels.
- `site/destination_oos_incremental.csv`: matched-scale JP pseudo-OOS use example.
- `site/calibration_holdout_test.csv`: train/holdout test for calibrated `X_C`.
- `site/submission_readiness.csv`: explicit research gates and blockers.
- `site/api/v1/`: versioned static definitions, latest states, and descriptive cases.
- `schemas/thermo_credit/`: JSON contracts for the MCP and CLI tool surface.
- `prospective/`: frozen protocol and append-only BOJ vintage archive tools.
- `replication/`: reproducibility manifest and logs.

Most files under `site/` and `tex/generated/` are generated and intentionally
ignored by Git.

## Paper and figures

Regenerate the empirical tables and modern Matplotlib/Seaborn figures with:

```bash
python scripts/18_boj_bridge_validation.py
python scripts/23_external_purpose_validation.py
python scripts/19_destination_oos_incremental.py
python scripts/06_make_theory_figures.py
python scripts/29_make_dashboard_takeaways.py
latexmk -cd -pdf -interaction=nonstopmode -halt-on-error tex/theory.tex
```

The theory figure builder writes PDF and SVG. The dashboard takeaway builder
writes PNG, PDF, SVG, and `tex/generated/dashboard_takeaways.tex`, which can be
included in another LaTeX document. Release automation validates the same final
PDF with qpdf, Ghostscript, Poppler, PDFium, and macOS PDFKit before publication.

## Automation

- `CI` runs tests on Python 3.10 and 3.11, a strict all-region build, a
  reproducibility check, dependency auditing, SBOM generation, and secret scans.
- `Update all regional data` refreshes one fixed automation branch and opens one
  update PR only after all regions pass validation.
- `Build & Publish` rebuilds validated regional data and deploys the generated
  static site from `main`.
- `Release theory.pdf` rebuilds and verifies the tagged paper, attaches the PDF,
  checksum, QA report, and takeaway figures to the GitHub release, then creates
  a Zenodo version when the required secret and repository variables exist.

Local validation is not evidence that GitHub Pages, a release, or Zenodo was
updated. Those external states must be checked after each publication run.

## Citation

Use the concept DOI for all versions:

- <https://doi.org/10.5281/zenodo.17563220>

The latest previously published version record is:

- <https://doi.org/10.5281/zenodo.17778342>

## License

See [LICENSE](LICENSE).
