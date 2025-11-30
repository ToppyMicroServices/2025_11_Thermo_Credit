# Thermo‑Credit Monitor (TQTC) — 日本語版 README
# ============================================

このリポジトリは、公開統計（主に CSV / FRED API）からサーモクレジット指標を計算
月次レポートを生成するための実験的なツールセットです。

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.17563221.svg)](https://doi.org/10.5281/zenodo.17563221)

- 英語版 README（詳細説明）: `README.md`
- 主な出力: `site/report.html`（ダッシュボード）、`site/indicators*.csv`（指標時系列）
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
- `external_coupling` : 外部圧力 / 温度インデックス（E_p, E_T）の設定
  - `enabled` / `alpha` / `delta`
  - `pressure_components` / `temperature_components`（FRED の series_id や transform など）
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
  日本の指標 (`site/indicators.csv`) を計算。
- `scripts/02_compute_indicators_eu.py`  
  EU 指標 (`site/indicators_eu.csv`) を計算。
- `scripts/02_compute_indicators_us.py`  
  US 指標 (`site/indicators_us.csv`) を計算。

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

## 引用情報

研究やレポートなどで本リポジトリを利用する場合は、次の DOI を引用してください:

- DOI: https://doi.org/10.5281/zenodo.17563221
