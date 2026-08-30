# Thermo Credit

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.17563220.svg)](https://doi.org/10.5281/zenodo.17563220)

Thermo Creditは、信用の規模と借り手構成を測り、マクロ金融の状態を再計算できる形で公開する研究プロジェクトです。日本、ユーロ圏、米国の四半期ダッシュボードに加え、論文、データ対応表、頑健性検証、再現コードを収録しています。

[ダッシュボードを開く](https://toppymicros.com/2025_11_Thermo_Credit/report.html)
| [English](README.md)

## 現時点で言えること

現在もっとも根拠が強い成果は、日本銀行の業種別貸出残高から作成した「借り手構成」の測定ブリッジです。主要な4区分には、Bezemer、Samarina、Zhangが公表した日本向けcrosswalkを採用しています。信用規模と構成比は、同じ貸出母集団から計算します。

`q_t`は、4四半期の借り手構成に占める非金融法人の座標です。資金使途、GDPに結び付く信用、最終的な支出先を直接測った値ではありません。EUとUSは、より粗いproxyを使った移植性確認用のpanelです。日本の測定結果を国際比較で検証したものではありません。

`S_M`、`T_L`、`p_C`、`U`、`F_C`、`X_C`、`loop_area`は実験的な診断指標です。とくに`X_C`は、安全余力、政策上の閾値、予測値として検証済みではありません。現時点の疑似OOS検証では、借り手構成を加えたモデルが、同じ母集団の信用残高だけを使うbaselineを安定して上回るとは言えません。

解釈の前に、[変数定義](docs/definitions.md)、[識別戦略](docs/identification_strategy.md)、[parameter calibrationの手順](docs/calibration_protocol.md)を確認してください。

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

生成されるダッシュボードは`site/report.html`です。Plotlyの画像出力環境が使えない場合は`TMS_SKIP_PNG=1`を指定します。この場合もinteractive HTMLは生成されます。

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

系列の出所、単位、proxyとしての限界は、`data/data_dictionary.csv`と`data/sources.json`にまとめています。サイト用の検査結果は`site/data_manifest.json`に出力されます。

## 主な成果物

- `tex/theory.tex`: 測定論文の原稿。
- `site/report.html`: 地域別のinteractive dashboard。
- `site/indicators*.csv`: 地域別の四半期診断指標。
- `site/credit_destination*.csv`: 借り手構成または地域別proxy panel。
- `site/destination_oos_incremental.csv`: 日本のmatched-scale疑似OOS利用例。
- `site/calibration_holdout_test.csv`: calibrated `X_C`のtrain/holdout検証。
- `site/submission_readiness.csv`: 研究上のgateと未解決項目。
- `site/api/v1/`: バージョン付きの変数定義、最新状態、記述的な事例比較。
- `schemas/thermo_credit/`: MCPとコマンドライン機能のJSONスキーマ。
- `prospective/`: 固定済みprotocolとBOJ vintage保存ツール。
- `replication/`: 再現性のmanifestとlog。

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

直前に公開済みのversion recordは次のとおりです。

- <https://doi.org/10.5281/zenodo.17778342>

## ライセンス

[LICENSE](LICENSE)を参照してください。
