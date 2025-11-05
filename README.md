# 2025_11_Thermo_Credit
最小構成のMVP。
公開統計CSVから、**流動性温度 (TMS-LT)**、**貨幣エントロピー (S_M)**、**規制ループ散逸面積 (PLD)**、**信用エクセルギー上限 (X_C)** を計算して、`site/report.html` を自動生成します。

## 使い方（最短）
1. GitHubリポジトリを作成し、このプロジェクトをクローンします。
2. 必要に応じて設定を変更します。
3. GitHubのSettings → Pages → **Build and deployment = GitHub Actions** を選択します。
4. Actionsタブで **Build & Publish** のワークフローを選択し、`Run workflow` をクリックして手動実行します（または毎月自動実行されます）。
5. `https://<yourname>.github.io/<repo>/report.html` にアクセスして月次レポートを確認します。

## 指標
- **S_M** = k · M_in · H(q)（貨幣エントロピー）
- **T_L (TMS-LT)** = （低スプレッド×薄板×高回転）を z-score 合成して 0-1 正規化
- **PLD** ≈ Σ p_R(t-1)·ΔV_R(t)（指数忘却付きのストリーミング近似）
- **X_C** = U − T0 · S_M（信用エクセルギー上限）

## ディレクトリ
- `data/` 入力CSV（サンプルデータを含む）
- `lib/`  指標計算用の関数群
- `scripts/` データ処理パイプライン（前処理→指標計算→レポート生成）
- `site/` 出力ディレクトリ（GitHub Actionsが上書きします）
- `.github/workflows/` GitHub Actionsの定義ファイル

## 注意
- このプロジェクトは最低限の実装であり、数式や定義はプロジェクトの規約に合わせて修正してください。
- 大規模データやAPI取得の処理が必要な場合は、`scripts/01_build_features.py` を拡張してください。
