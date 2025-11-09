# Thermo‑Credit Monitor (TQTC)

Compute thermo‑credit indicators from public statistics (local CSVs or FRED API) and render a monthly, multi‑region report.

Core metrics:
- S_M — Monetary Dispersion Entropy (entropy‑like, extensive)
- T_L — Liquidity “Temperature”
- loop_area — Policy/Regulatory Loop Dissipation (PLD)
- F_C — Helmholtz‑like Free Energy (F_C = U − T0 · S_M)
- X_C — Credit Exergy Ceiling (needs baselines; falls back to F_C when absent)

Artifacts: `site/report.html` (Plotly interactive + PNG fallbacks), `archive.json`, `feed.xml`, `sitemap.xml`, `robots.txt`.

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

`site/indicators_eu.csv` / `site/indicators_us.csv` が存在すると、レポートに自動でタブが追加されます（Compare タブも表示）。

Notes
- 一部地域の `X_C` が未校正の間は、自動的に `F_C` へフォールバックしてグラフ表示されます。
- Raw Inputs 図は `data/sources.json` の `enabled: true` な系列を first=100 正規化で重ね描きします（各地域共通タブ）。

---

## Configuration
- Base config: `config.yml`
- Region overrides: `config_jp.yml`, `config_eu.yml`, `config_us.yml`
- 環境変数
  - `FRED_API_KEY`（任意）: FRED の API キー（取得できない場合はローカル CSV を使用）
  - `REPORT_PLOT_START`（任意）: 可視化の開始日（例: `2010-01-01`）
  - `CONFIG_REGION`（任意）: 地域設定の切り替え（`jp`/`eu`/`us`）
  - Branding: `BRAND_BG`, `BRAND_BG2`, `BRAND_TEXT`（ヘッダー/フッターのブランド色）

---

## Branding & logo
- ロゴは Base64 データ URI として HTML にインライン埋め込みされます。
- 事前に軽量化するには（推奨）:
  ```bash
  python scripts/optimize_logo.py --height 80 --colors 96
  ```
  生成: `scripts/og-brand-clean.min.png`（存在すればレポート側が優先的に埋め込み）
- ブランドカラーは環境変数で変更可能:
  ```bash
  export BRAND_BG="#0d1b2a"
  export BRAND_BG2="#1b263b"
  export BRAND_TEXT="#ffffff"
  python scripts/03_make_report.py
  ```

---

## Data & sources
- 既定では `data/*.csv` を読み込み、`scripts/02_compute_indicators*.py` が指標を算出します。
- `data/sources.json` を用意すると、レポートに Sources 表と Raw Inputs 図が表示されます。
  - `{"id":"JPNASSETS","title":"BoJ Total Assets","enabled":true}`
  - CSV の既定パスは `data/<id>.csv`（列: `date`, `value`）。別名を使う場合は `path` を指定。

---

## CI
- Workflow: `.github/workflows/build_report.yml`
  - 依存インストール → ロゴ最適化 → レポート生成 → `site/` をアーティファクトとしてアップロード
- （任意）`.github/workflows/update_data.yml` を用意して日次の生データ更新→PR作成も可能です。

---

## Tips
- PNG 替え玉出力には `kaleido` が必要です。
- VS Code: `.vscode/extensions.json` 等の共有設定はコミット可（他は ignore）。
- 生成物（`site/`）や大きな CSV は Git から除外する運用を推奨（CI で都度生成）。

---

## License
See `LICENSE`.

---

### Yield comparison (US vs JP)
手元で日米長期金利をざっと比較する補助スクリプト:
```bash
python scripts/compare_yields.py --start 1995-01-01 --png
```
生成: `site/yield_compare.html`（インタラクティブ）, `site/yield_compare.png`（任意）, `site/yield_compare_metrics.csv`（スプレッドとローリング相関）。
