# Thermo‑Credit Monitor (TQTC) — 日本語版 README
# ============================================

このリポジトリは、公開統計（主に CSV / FRED API）からサーモクレジット指標を計算
月次レポートを生成するための実験的なツールセットです。

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.17563221.svg)](https://doi.org/10.5281/zenodo.17563221)

- 英語版 README（詳細説明）: `README.md`
- 主な出力: `site/report.html`（ダッシュボード）、`site/indicators*.csv`（dashboard 版と real-time 版の指標時系列）、`site/credit_destination*.csv`（JP は BOJ borrower-composition panel、EU/US は従来の proxy panel）、`site/destination_oos_incremental.csv`（同一BOJ母集団を用いた OOS 利用例）、および各種補助診断
---
## インストール

前提:

- Python 3.10 または 3.11 がインストールされていること
- macOS / Linux を想定（Windows の場合は `source` の部分を PowerShell 用に読み替えてください）

手順:

```bash
# リポジトリを取得
git clone https://github.com/ToppyMicroServices/2025_11_Thermo_Credit.git
cd 2025_11_Thermo_Credit

# 仮想環境を作成して有効化（任意だが推奨）
python3 -m venv .venv
source .venv/bin/activate

# pip を更新して依存ライブラリをインストール
python -m pip install -U pip
pip install -r requirements.txt -c constraints.txt
```


## 基本的な使い方

```bash
python3 -m venv .venv && source .venv/bin/activate
python -m pip install -U pip
pip install -r requirements.txt

# 日本の指標とレポートをビルド
python [02_compute_indicators.py](http://_vscodecontentref_/2)
python [03_make_report.py](http://_vscodecontentref_/3)
open [report.html](http://_vscodecontentref_/4)  # 
```



## 設定ファイルのメンテナンス

主な設定は YAML ファイルで管理されています:

- `config.yml` : 共通のデフォルト設定（指標のパラメータ、エントロピーのカテゴリ、外部カップリングなど）
- `config_jp.yml` : 日本向けの上書き設定
- `config_eu.yml` : EU 向けの上書き設定
- `config_us.yml` : US 向けの上書き設定

よく触る項目の例:

- `q_cols` : エントロピー計算に使うバケット名（MECE のカテゴリ）
- `external_coupling` : 外部圧力 / 流動性状態インデックス（E_p, E_T）の設定
  - `enabled` / `alpha` / `delta`
  - `pressure_components` / `temperature_components`（FRED の series_id や transform など）
- `preprocessing.real_time_forecast` : 予測用の no-lookahead 前処理
  - dashboard 用の `site/indicators*.csv` とは別に、release lag を入れた `site/indicators*_realtime.csv` を出力
  - デフォルト lag は GDP/U 90日、credit/depth/turnover 120日、spread/market 1日、money 30日、allocation/regulatory 90日
  - lag の仮定は `site/realtime_release_lags*.json` にも保存されます
  - calibration は既定で real-time panel を優先して読みます（`CALIBRATION_PANEL_MODE=realtime`）
- `credit_destination` : borrower-composition bridge
  - `enabled: true` で `site/credit_destination*.csv` を出力し、JP では `C_NFB`, `C_FIN`, `C_PROP`, `C_HH_NONHOUSING`, `q_t` と監査列を `site/indicators*.csv` に統合
  - JP の主指標は `q_t = sum_4q(C_NFB) / sum_4q(C_NFB + C_FIN + C_PROP + C_HH_NONHOUSING)`。旧 G/B/E と `lambda_B` 経路は Appendix・互換性用です
  - JP は `data/credit_destination_jp.csv` の BOJ sectoral bridge を優先し、EU/US は allocation proxy のままです
- `enrichment` : depth / turnover などの拡張指標の係数・フォールバック値
- `F_C_baseline_*` : F_C / X_C を系列の最小値や分位点でシフトして常に正に保つための基準（`mode` は `min` / `quantile` / `value` / `first`、`eps` でゼロよりわずかに持ち上げ）
- `exergy_floor_zero` / `exergy_floor_mode` : X_C の負値をどう扱うか（0 クリップかシフトか）

運用的には:

- 共通の調整 → `config.yml` を更新
- 特定地域だけ変えたい → `config_<region>.yml` 側で同じキーを上書き

設定を変えた後は、少なくとも該当地域のインジケータを再計算してください
（例: `python scripts/02_compute_indicators_eu.py`）

---

## スクリプトの目的

よく使うスクリプトの役割は以下の通りです。

**特徴量ビルド（元データ → data/*.csv）**

- `scripts/01_build_features.py`  
  日本の元データ取得＋特徴量テーブル (`data/money.csv`, `data/credit.csv` など) を構築。
- `scripts/04_build_features_eu.py`  
  EU 向けの特徴量テーブル構築。
- `scripts/05_build_features_us.py`  
  US 向けの特徴量テーブル構築。

**指標の計算（data/*.csv → site/indicators*.csv）**

- `scripts/02_compute_indicators.py`  
  日本の指標 (`site/indicators.csv`) と予測用 real-time 指標 (`site/indicators_realtime.csv`) を計算。あわせて `site/credit_destination.csv` と `site/credit_destination_realtime.csv` を出力。
- `scripts/02_compute_indicators_eu.py`  
  EU 指標 (`site/indicators_eu.csv`, `site/indicators_eu_realtime.csv`) と destination panel を計算。
- `scripts/02_compute_indicators_us.py`  
  US 指標 (`site/indicators_us.csv`, `site/indicators_us_realtime.csv`) と destination panel を計算。

**Borrower-composition panel**

- JP は `scripts/17_fetch_boj_jp_credit_destination.py` で BOJ Time-Series Data Search LA01 の sectoral outstanding loans から `data/credit_destination_jp.csv` を作ります。
- JP primary は Bezemer–Samarina–Zhang の日本crosswalkに従う NFB、金融、property/mortgage、家計非住宅の四区分です。各bucket内で signed stock changes を合計してから正値化し、`q_t` は4四半期NFB coordinateとして表示します。loan purpose や destination の直接測定ではありません。
- scale は四区分と同じ `primary_included_stock`（国内免許銀行の公式totalから地方公共団体向けを除いた残高）です。「国内免許」は貸手を指し、NFB residualには明示した海外関連成分が含まれます。旧 `mapped_domestic_stock` はprimaryではありません。
- 建設は primary ではNFB、Werner-inspired BOJ proxyではfinancial-circulation側、Müller–Verner adaptationではnon-tradableに置き、三つを同一母集団で併記します。著者定義G/B/EはAppendixに残します。
- BOJ の purpose-coded 系列は同一提供者の比較です。財務省・国土交通省の公刊統計による検証は部分的triangulationであり、全bucketやloan purpose全般を独立検証するものではありません。
- EU/US はまだ `L_real` の正の四半期差分と coarse allocation shares から作る proxy です。
- coverage は同じ母集団への包含と official aggregate への accounting reconciliation を分けて報告します。
- 本文の internal audit は、24系列の共通利用開始時点と official aggregate への stock reconciliation を別に報告します。

**`lambda_B` 感度分析**

- `scripts/09_lambda_b_sensitivity.py` は `lambda_B ∈ {0, 0.25, 0.5, 0.75, 1}` で `C_R`, `C_A`, `q_t` を再計算します。
- region-specific な `lambda_B` は推定せず、固定グリッドだけを使います。
- 出力は `site/credit_destination_lambda_b_sweep.csv`、`site/lambda_b_sensitivity.csv`、`tex/generated/theory_lambda_b_sensitivity.tex` です。
- asset acceleration は直接の asset-price series がない場合 `L_asset` proxy を使います。
- `lambda_B` に敏感な結果、target が不安定な結果、total-credit baseline に勝たない結果は main claim にしません。

**Baseline forecast comparison**

- `scripts/10_baseline_forecast_comparison.py` は Thermo-Credit 指標が単純な baseline を OOS で上回るかを検証します。
- Targets は real activity growth、inflation（利用可能な場合）、asset-acceleration proxy、spread widening、stress regime、lower-tail growth、volatility spike です。
- Baselines は AR(1)、total-credit growth、credit-to-GDP gap、spread-only、money growth、simple FCI、within-region expanding z-score を使う pooled region fixed-effect panel です。
- GDP-like な `Y` / `U` が credit panel と桁違いの単位で入る場合は、指標生成時に credit scale へ揃えます。これにより US の World Bank dollar level と billion-scale credit の混在を避けます。
- 出力は `site/baseline_forecast_comparison.csv`、`site/baseline_forecast_target_coverage.csv`、`data/baseline_forecast_summary.json`、`tex/generated/theory_baseline_forecast_comparison.tex` です。
- 現在の panel では downside-risk monitor としての改善は一部ありますが、growth / asset acceleration / stress の広い予測力は一貫していません。ここで勝てない結果は本文の理論主張を弱める根拠として扱います。

**Calibration holdout test**

- `scripts/11_calibration_holdout_test.py` は calibrated implicit headroom score が raw pipeline `X_C` を本当に上回るかを検証します。
- fixed split は 2015 年までの観測だけで `theta=(T0,p0,U0,V0,S0)` を推定し、2016-2025 の forecast origin で評価します。
- rolling split は各 forecast origin の直前 40 四半期だけで `theta` を再推定します。
- 出力は `site/calibration_holdout_test.csv`、`data/calibration_holdout_summary.json`、`tex/generated/theory_calibration_holdout.tex` です。
- 現在の holdout 結果では calibrated score は raw `X_C` と simple trailing-change baseline を安定して上回りません。したがって本文では validation ではなく diagnostic overlay として扱います。

**Entropy partition robustness**

- `scripts/12_entropy_partition_robustness.py` は `S_M_hat` が bucket 設計の人工物かどうかを検証します。
- borrower-label partition と loan-purpose partition について、3 / 5 / 7 buckets を再計算します。
- negative control は shuffled shares、fixed shares、random-walk shares です。
- 出力は `site/entropy_partition_robustness.csv`、`data/entropy_partition_robustness_summary.json`、`tex/generated/theory_entropy_partition_robustness.tex` です。
- 現在の panel では全 observed partition が flat なので、entropy result は main empirical evidence から外し、dashboard diagnostic として扱います。

**TL robustness**

- `scripts/13_tl_robustness.py` は liquidity-state index の設計依存性を検証します。
- multiplicative、additive z-score、soft-min、harmonic mean、spread-only、turnover-excluded、depth-excluded を比較します。
- すべて expanding-window log z-score を使い、spread 低下、depth 上昇、turnover 上昇で score が上がる向きに揃えます。
- 出力は `site/tl_robustness.csv`、`data/tl_robustness_summary.json`、`tex/generated/theory_tl_robustness.tex` です。
- signed raw-product 型の multiplicative formula は単調性テストで棄却し、additive z-score を main specification として残します。

**Loop-area null tests**

- `scripts/14_loop_area_null_tests.py` は closed-loop area が単なる trend / autocorrelation で出ていないかを検証します。
- null は block shuffle、phase randomization、AR(1) surrogate、event-date permutation、registered event 外の placebo periods です。
- 最新 8 / 12 / 16 quarter window の segmentation sensitivity と、event registry に基づく event-window 行を出します。
- 出力は `site/loop_area_null_tests.csv`、`data/loop_area_null_summary.json`、`tex/generated/theory_loop_area_null_tests.tex` です。
- 現在の panel では hysteresis claim は単独では支えられず、loop area は path-stress monitor / audit trigger として扱います。

**Integrability synthetic test**

- `scripts/15_integrability_synthetic_test.py` は Maxwell-like curl 推定器を synthetic data で検証します。
- 既知の quadratic potential から `T_L` と `p_C` を作り、noise level、sampling frequency、非可積分場、proxy misspecification を変えます。
- 出力は `site/integrability_synthetic_test.csv`、`data/integrability_synthetic_summary.json`、`tex/generated/theory_integrability_synthetic_test.tex` です。
- clean potential では `Omega` がほぼ 0、設計した vorticity では大きくなり、proxy 汚染でも consistency warning として上がります。

**Submission-readiness gates**

- `scripts/16_submission_readiness.py` は次版の採択可能ラインを gate として判定します。
- gate は JP `q_t` と total-credit baseline の OOS 対決、calibrated `X_C` holdout、richer sectoral / loan-purpose data での `S_M_hat` の動き、TL / loop robustness、完全再現性です。
- 出力は `site/submission_readiness.csv`、`data/submission_readiness_summary.json`、`tex/generated/theory_submission_readiness.tex` です。
- 現在の原稿の主張は BOJ borrower-composition measure と検証可能な利用例に限定します。loan destination と未検証の thermodynamic structure は中核的主張から外しています。

**レポート生成**

- `scripts/03_make_report.py`  
  各地域の `site/indicators*.csv` を読み込み、HTML レポート (`site/report.html`) を生成。

**その他**

- `scripts/ci_prepare_minimal_data.py`  
  CI 用の最小データを用意するためのスクリプト（ネットワーク無しでテストを動かすためのもの）。
- `scripts/fetch_fred_series.py` など  
  生データの取得・バックフィル用のユーティリティ。

テスト関連:

- `pytest` を実行すると `tests/` 以下のユニットテストが走ります。  
  指標の形、外部カップリング、エントロピーの仕様など壊れやすい部分をカバーしています。

---

## CI（GitHub Actions）の概要

GitHub 上では、いくつかのワークフローが自動／手動で動きます（リポジトリの「Actions」タブで確認できます）。

主なもの:

- **CI (`.github/workflows/ci.yml`)**
  - `main` ブランチへの push / PR で起動。
  - Python 3.10 / 3.11 で:
    - `pip install -r requirements.txt -c constraints.txt`
    - `pytest -q`
  - 一部のスクリプト（JP/EU ビルド）も回して、パイプライン全体が壊れていないかをチェック。

- **Matrix CI (Pinned vs Latest)**
  - pinned（`constraints.txt` を使った固定バージョン）と latest（`requirements.txt` のみ）をマトリクスでテスト。
  - 依存ライブラリのアップデートによる影響を早めに検知するためのワークフロー。

- **Build & Publish / Build report**
  - レポート `site/` をビルドし、Artifacts としてアップロードするジョブ。

- **Update JP Data / Update All Regions Data**
  - `schedule`（cron）で定期実行されるデータ更新ジョブ。
  - FRED / World Bank などから新しいデータを取りに行き、特徴量を更新する。

ローカルで CI に近いチェックをしたい場合は、次のように実行すれば大体同じことができます:

```bash
python -m pip install -U pip
pip install -r requirements.txt -c constraints.txt
pytest -q
```

査読用の完全再現チェックは次で実行できます:

```bash
python scripts/08_reproducibility_check.py --report-dir replication
```

このチェックは入力 CSV の SHA-256 を記録し、dashboard 版と real-time 版の
`site/indicators*.csv`、`site/credit_destination*.csv`、`lambda_B` 感度分析、baseline forecast comparison、calibration holdout test、entropy partition robustness、TL robustness、loop-area null tests、integrability synthetic test、submission-readiness gates、calibration JSON、Table 3 用 LaTeX、Figure 1/2 を2回再生成して比較します。
数値出力は `rtol=1e-9, atol=1e-8` で判定し、結果は
`replication/reproducibility_manifest.json` と
`replication/reproducibility_log.md` に保存されます。

## 引用情報

研究やレポートなどで本リポジトリを利用する場合は、次の DOI を引用してください:

- DOI: https://doi.org/10.5281/zenodo.17563221
