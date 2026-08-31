# Thermo Credit

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.17563220.svg)](https://doi.org/10.5281/zenodo.17563220)

銀行貸出は、総額が同じでも、どの部門が借り手になるかによって意味が変わります。Thermo Creditは「貸出の規模」と「借り手構成」を分けて測り、誰でも検証・再計算できる形で公開する研究プロジェクトです。

中心となる成果は、日本の借り手構成を測る方法です。これに、ユーロ圏と米国の試験的な代理系列、論文、データ対応表、ダッシュボード、頑健性検証、再現コードを組み合わせています。

[ダッシュボードを見る](https://toppymicros.com/2025_11_Thermo_Credit/)
| [論文を読む](https://github.com/ToppyMicroServices/2025_11_Thermo_Credit/releases/latest/download/theory.pdf)
| [測定方法を確認する](docs/identification_strategy.md)
| [引用方法](#引用)
| [English](README.md)

## 最初に見るもの

1. ダッシュボードでは、各地域の現在値を、その地域自身の過去と比較します。
2. 論文では、日本の測定方法と検証結果を確認できます。
3. 下記の手順を実行すると、指標と公開データを手元で再計算できます。

## 現時点で分かっていること

| 確認したいこと | 現在の答え |
| --- | --- |
| 日本の業種別貸出残高から借り手構成を測れるか | 4区分の測定値として再現できます。対象範囲と会計上の整合性も検査しています。 |
| `q_t`は貸出金の最終的な使途を示すか | 示しません。4四半期の借り手構成に占める非金融法人の比率です。 |
| ユーロ圏と米国は日本の測定結果を検証しているか | いいえ。より粗い代理系列を使い、同じ計算枠組みを適用できるか確認しています。 |
| 借り手構成を加えると予測が良くなるか | 現在の疑似将来時点検証では、貸出残高だけを使う基準モデルを上回っていません。 |
| 熱力学的な指標を判断基準として使えるか | まだ使えません。現段階では、状態を記述するための実験的な変換です。 |

現在もっとも根拠が強い成果は、日本銀行の業種別貸出残高から作成した借り手構成の測定値です。主要な4区分には、Bezemer、Samarina、Zhangが公表した日本向け対応表を採用しています。貸出規模と構成比は、同じ貸出母集団から計算します。

`S_M`、`T_L`、`p_C`、`U`、`F_C`、`X_C`、`loop_area`は実験的な診断指標です。売買、政策判断、安全性評価のために検証された指標ではありません。解釈の前に、[変数定義](docs/definitions.md)、[識別戦略](docs/identification_strategy.md)、[パラメータ調整の手順](docs/calibration_protocol.md)を確認してください。

## 実行方法

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

生成されるダッシュボードは`site/report.html`です。Plotlyの画像出力環境が使えない場合は`TMS_SKIP_PNG=1`を指定します。この場合も対話型HTMLは生成されます。

## AI・外部ツール向けインターフェース

バージョン付きの静的JSON APIを生成し、同じ計算処理をコマンドラインから呼び出せます。

```bash
python scripts/28_build_public_api.py
python scripts/thermo_credit_cli.py get_theory_overview
python scripts/thermo_credit_cli.py compute_thermo_credit_metrics --repo-region jp --limit 8
```

静的APIの入口は`site/api/v1/manifest.json`です。MCPサーバーは、変数定義、計算機能、登録済みイベントの記述的な比較、プロンプトのひな型を提供します。

```bash
python scripts/thermo_credit_mcp_server.py --transport stdio
```

詳しくは[MCPインターフェース](docs/thermo_credit_mcp_spec.md)を参照してください。GitHub Pagesで公開するのは読み取り専用のJSONです。MCPをHTTPで運用する場合は、認証や呼び出し回数の制限を別途設ける必要があります。

## データ更新

地域別の更新や検査が一つでも失敗すると、全地域更新は失敗として終了します。主なコマンドは次のとおりです。

```bash
python scripts/fetch_ecb_series.py
python scripts/fetch_fred_series.py
python scripts/01_build_features.py
python scripts/04_build_features_eu.py
python scripts/05_build_features_us.py
python scripts/02_compute_all_regions.py
python scripts/27_validate_site_data.py --min-rows 8 --max-age-days 550
```

`scripts/fetch_ecb_series.py`は、ECB Data Portalの現行BSI総資産系列を取得します。公開用の更新では、途中までの取得を成功扱いにする`--allow-partial`を使いません。

系列の出所、単位、代理系列としての限界は、`data/data_dictionary.csv`と`data/sources.json`にまとめています。サイト用の検査結果は`site/data_manifest.json`に出力されます。

## 主な成果物

- `tex/theory.tex`: 測定論文の原稿。
- `site/report.html`: 地域別の対話型ダッシュボード。
- `site/indicators*.csv`: 地域別の四半期診断指標。
- `site/credit_destination*.csv`: 借り手構成または地域別の代理系列。
- `site/destination_oos_incremental.csv`: 日本の貸出規模指標と比較する疑似将来時点検証。
- `site/calibration_holdout_test.csv`: 調整済み`X_C`の学習期間・検証期間テスト。
- `site/submission_readiness.csv`: 研究上の判定条件と未解決項目。
- `site/api/v1/`: バージョン付きの変数定義、最新状態、記述的な事例比較。
- `schemas/thermo_credit/`: MCPとコマンドライン機能のJSONスキーマ。
- `prospective/`: 固定済みの検証手順とBOJの過去時点データを保存するツール。
- `replication/`: 再現性検査の記録。

`site/`と`tex/generated/`の大半は再生成できるため、Gitの管理対象外です。

## 論文と図

本文用の表とMatplotlib/Seaborn図は、次の手順で再生成できます。

```bash
python scripts/18_boj_bridge_validation.py
python scripts/23_external_purpose_validation.py
python scripts/19_destination_oos_incremental.py
python scripts/06_make_theory_figures.py
python scripts/29_make_dashboard_takeaways.py
latexmk -cd -pdf -interaction=nonstopmode -halt-on-error tex/theory.tex
```

本文用の図はPDFとSVGで出力します。ダッシュボードの要点図はPNG、PDF、SVGに加え、他のLaTeX文書で使える`tex/generated/dashboard_takeaways.tex`も生成します。リリース時は、qpdf、Ghostscript、Poppler、PDFium、macOS PDFKitを使って最終PDFそのものを検査します。

## 自動化

- `CI`はテスト、全地域の厳格なビルド、再現性検査、依存関係の監査、SBOM、機密情報の検査を実行します。
- `Update all regional data`は固定した自動更新用ブランチを更新し、全地域が検査を通った場合だけ一つのPRを作ります。
- `Build & Publish`は`main`からデータとサイトを再生成し、GitHub Pagesへ配置します。
- `Release theory.pdf`はタグ付けしたソースから論文を再生成して検査し、PDF、チェックサム、検査報告、要点図をGitHub Releaseへ追加します。必要なシークレットとリポジトリ変数が設定されている場合は、Zenodoにも新しい版を作成します。

ローカルでの成功は、Pages、GitHub Release、Zenodoの更新完了を意味しません。公開後は、それぞれの外部状態を確認する必要があります。

## 引用

すべてのversionをまとめて引用する場合は、concept DOIを使用してください。

- <https://doi.org/10.5281/zenodo.17563220>

現在の最新版は次のとおりです。

- <https://doi.org/10.5281/zenodo.22177533>

## ライセンス

[LICENSE](LICENSE)を参照してください。
