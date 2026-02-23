# 日本株 AI スクリーニングシステム

スイングトレード向けに、**AIが毎日自動で有望な日本株をスクリーニング**するシステムです。

**ハイブリッド予測**に対応：全銘柄結合の「グローバルモデル」と、225銘柄それぞれの「ローカルモデル」の予測確率を重み付け合成し、高精度なスクリーニングを実現します。

---

## 概要

| ステップ | 内容 |
|---|---|
| データ収集 | yfinance で日経225銘柄の株価・出来高を取得 |
| 特徴量生成 | 移動平均、RSI、MACD、ボリンジャーバンドなど20種類以上のテクニカル指標を計算 |
| モデル学習 | LightGBM で「5営業日後に5%以上上昇するか」を学習（グローバル + ローカル225個） |
| スクリーニング | ハイブリッド予測（グローバル×ローカルの重み付け合成）で有望銘柄を抽出 |
| センチメント分析 | Gemini API でニュース見出しを分析し、ポジティブ銘柄のみに絞り込み |
| 通知 | LINE Messaging API で有望銘柄を自動通知 |
| 自動化 | GitHub Actions で平日毎日自動実行 |

---

## ディレクトリ構成

```
自動スクリーニング/
├── config.py                  # 全体設定（期間、閾値、ハイブリッド重み、API キー等）
├── main.py                    # メインパイプライン（従来 / ハイブリッド対応）
├── requirements.txt           # 依存パッケージ
├── nikkei225_tickers.csv      # 監視対象の銘柄コード一覧
├── data/
│   ├── ingest_yfinance.py     # yfinance で株価データ取得
│   └── nikkei225_tickers.py   # 銘柄リスト自動取得スクリプト
├── features/
│   ├── make_features.py       # テクニカル指標の計算
│   └── make_dataset.py        # 学習データセットの作成
├── models/
│   ├── train_model.py         # 従来の汎用モデル学習・保存
│   ├── train_hybrid.py        # ハイブリッド学習（グローバル + ローカル225個）
│   ├── model.pkl              # 従来モデル（自動生成）
│   ├── global_model.pkl       # グローバルモデル（自動生成）
│   └── local/                 # ローカルモデル置き場（自動生成）
│       └── local_model_*.pkl  # 銘柄ごとの個別モデル
├── screening/
│   ├── screen.py              # 従来スクリーニング推論
│   └── screen_hybrid.py       # ハイブリッドスクリーニング推論
├── sentiment/
│   └── news_sentiment.py      # ニュース取得 + Gemini センチメント分析
├── notify/
│   └── line_notify.py         # LINE Messaging API で通知（ハイブリッド対応）
└── .github/
    └── workflows/
        └── screening.yml      # GitHub Actions 自動実行設定
```

---

## セットアップ

### 1. 依存パッケージのインストール

```bash
pip install -r requirements.txt
```

### 2. 銘柄リストの準備

`nikkei225_tickers.csv` をプロジェクトルートに配置します。

**自動取得する場合:**
```bash
python data/nikkei225_tickers.py
```

**手動で作成する場合:**

以下のような形式で、1行1銘柄のCSVファイルを作成してください。

```csv
7203.T
6758.T
9984.T
8306.T
...
```

> Yahoo Finance の日本株ティッカーは `銘柄コード.T` の形式です。

### 3. LINE Messaging API の設定（通知を使う場合）

1. [LINE Developers](https://developers.line.biz/) でプロバイダー・チャネルを作成
2. **Messaging API チャネル** を選択
3. 「チャネルアクセストークン（長期）」を発行
4. 自分のユーザーIDを確認（チャネル基本設定の「あなたのユーザーID」）
5. 環境変数を設定:

```bash
# Windows (PowerShell)
$env:LINE_CHANNEL_ACCESS_TOKEN = "あなたのチャネルアクセストークン"
$env:LINE_USER_ID = "あなたのユーザーID"

# Mac / Linux
export LINE_CHANNEL_ACCESS_TOKEN="あなたのチャネルアクセストークン"
export LINE_USER_ID="あなたのユーザーID"
```

> LINE の設定がなくても、コンソール出力でスクリーニング結果を確認できます。

---

## 使い方

### ハイブリッドモード（推奨）

#### 初回: ハイブリッド学習 → スクリーニング → 通知

```bash
python main.py --full-hybrid
```

グローバルモデル + 225銘柄のローカルモデルを学習し、そのままスクリーニングを実行します。  
初回は **30分〜1時間程度** かかります（PC負荷軽減のためスリープを挟んでいます）。

#### 日次運用: ハイブリッドスクリーニング + 通知

```bash
python main.py --hybrid
```

学習済みのグローバル・ローカルモデルでハイブリッド予測を行い、結果をLINEで通知します。  
**日々の運用ではこのモードを使います。**

#### 日次運用（推奨）: ハイブリッド + ニュースセンチメント分析

```bash
python main.py --hybrid --sentiment
```

ハイブリッド予測の上位30銘柄について、Yahoo!ファイナンスからニュース見出しを取得し、Gemini API でスイングトレード目線のセンチメント（1-5）を判定します。スコア4以上（ポジティブ）の銘柄のみを最終通知リストとして出力します。

> `GEMINI_API_KEY` 環境変数の設定が必要です。

#### 週次: ハイブリッドモデル再学習

```bash
python main.py --hybrid-train
```

グローバルモデルとローカルモデルを再学習します。

### 従来モード（グローバルモデルのみ）

#### フル実行（学習 → スクリーニング → 通知）

```bash
python main.py
```

初回実行時はモデル学習から始まります。  
日経225全銘柄のデータ取得に **10〜20分程度** かかります。

#### モデル学習のみ

```bash
python main.py --train
```

過去2年分のデータで学習し、`models/model.pkl` を保存します。

#### スクリーニングのみ（学習済みモデル使用）

```bash
python main.py --screen
```

直近3ヶ月のデータでスクリーニングし、結果をLINEで通知します。

### 共通

スクリーニング結果は `outputs/` フォルダに `screening_YYYYMMDD_HHMMSS.txt` 形式で保存されます（ログ兼バックアップ）。

---

## 設定のカスタマイズ

`config.py` で以下の設定を変更できます。

| 設定 | デフォルト | 説明 |
|---|---|---|
| `TRAIN_PERIOD` | `"2y"` | 学習データの取得期間 |
| `SCREEN_PERIOD` | `"3mo"` | スクリーニング用データの取得期間 |
| `UP_DAYS` | `5` | 何営業日後の変化を予測するか |
| `UP_RATE` | `0.05` | 上昇率の閾値（5%） |
| `MIN_VOLUME` | `500000` | フィルタリング用最低出来高 |
| `TOP_N` | `6` | 通知する上位銘柄数 |
| `MIN_PROB` | `0.5` | 最低上昇確率 |
| `GLOBAL_WEIGHT` | `0.6` | ハイブリッド合成時のグローバル重み |
| `LOCAL_WEIGHT` | `0.4` | ハイブリッド合成時のローカル重み |
| `LOCAL_MIN_SAMPLES` | `100` | ローカルモデル学習に必要な最低サンプル数 |
| `TRAIN_SLEEP_SEC` | `0.5` | ローカル学習ループ間のスリープ（秒） |
| `LGB_N_JOBS` | `2` | LightGBM の並列コア数制限 |
| `GEMINI_API_KEY` | (環境変数) | Gemini API キー |
| `GEMINI_MODEL` | `"gemini-1.5-flash"` | 使用する Gemini モデル名 |
| `SENTIMENT_TOP_N` | `30` | センチメント分析対象の上位銘柄数 |
| `SENTIMENT_MIN_SCORE` | `4` | 最終通知に残す最低センチメントスコア |
| `SENTIMENT_DEFAULT_SCORE` | `3` | ニュース取得失敗時のデフォルトスコア |
| `SENTIMENT_API_SLEEP` | `2.0` | Gemini API コール間のスリープ（秒） |
| `NEWS_LOOKBACK_DAYS` | `7` | ニュース取得の遡り日数 |

---

## 特徴量一覧

| カテゴリ | 特徴量 | 説明 |
|---|---|---|
| 価格 | `Close`, `Volume` | 終値、出来高 |
| 移動平均 | `ma5`, `ma25`, `ma75` | 5/25/75日移動平均 |
| 乖離率 | `ma5_ratio`, `ma25_ratio`, `ma75_ratio` | 各移動平均からの乖離率 |
| RSI | `rsi14` | 14日RSI |
| MACD | `macd`, `macd_signal`, `macd_hist` | MACD, シグナル, ヒストグラム |
| ボリンジャーバンド | `bb_upper`, `bb_lower`, `bb_width` | 上限, 下限, バンド幅 |
| 出来高 | `volume_ma5_ratio` | 出来高の5日平均との比 |
| 騰落率 | `return_1d`, `return_5d`, `return_10d` | 1/5/10日騰落率 |
| ボラティリティ | `volatility_20d` | 20日ボラティリティ |

---

## GitHub Actions で自動化

### 設定手順

1. プロジェクトを GitHub にプッシュ
2. GitHub リポジトリの **Settings → Secrets and variables → Actions** で以下を追加:
   - `LINE_CHANNEL_ACCESS_TOKEN`
   - `LINE_USER_ID`
   - `GEMINI_API_KEY`（センチメント分析を使う場合）
3. `.github/workflows/screening.yml` が自動で有効になります

### スケジュール

| 実行タイミング | 内容 |
|---|---|
| 平日 16:00 JST | ハイブリッドスクリーニング + 通知 |
| 月曜 16:00 JST | ハイブリッドモデル再学習 |

手動実行も可能です（GitHub Actions → workflow_dispatch）。

---

## ハイブリッド予測の仕組み

```
最終スコア = GLOBAL_WEIGHT × グローバル予測確率 + LOCAL_WEIGHT × ローカル予測確率
```

| モデル | 役割 | 特徴 |
|---|---|---|
| グローバル | 市場全体の傾向を捕捉 | 全銘柄結合データで学習、汎化性能が高い |
| ローカル | 銘柄固有の癖を捕捉 | 銘柄ごとに個別学習、特有の値動きパターンに強い |

ローカルモデルが存在しない銘柄（データ不足等）は、グローバルモデルのみで判定されます。

### PCリソースへの配慮

225銘柄のローカルモデル学習はリソースを消費するため、以下の対策を実装しています:

| 対策 | 設定 | 説明 |
|---|---|---|
| CPUコア数制限 | `LGB_N_JOBS = 2` | LightGBM の並列数を制限 |
| スリープ | `TRAIN_SLEEP_SEC = 0.5` | ループごとに休止しCPU温度を下げる |
| メモリ解放 | 毎ループ `gc.collect()` | 不要変数を削除しメモリリークを防止 |
| 軽量化 | ローカル `n_estimators=150` | グローバルの半分のイテレーション数 |
| スキップ | `LOCAL_MIN_SAMPLES = 100` | データ不足の銘柄は学習しない |

---

## ニュース・センチメント分析フィルターの仕組み

ハイブリッド予測の後処理として、ニュース見出しを基にしたセンチメント分析フィルターを適用できます。

```
LightGBMハイブリッド予測（上位30銘柄）
  ↓
Yahoo!ファイナンスからニュース見出し取得（RSS / HTMLスクレイピング）
  ↓
Gemini API でスイングトレード目線のセンチメント判定（1-5）
  ↓
スコア 4 以上（ポジティブ）の銘柄のみを最終通知
```

| スコア | 意味 |
|---|---|
| 1 | 非常にネガティブ（大幅下落リスク） |
| 2 | ネガティブ（下落圧力あり） |
| 3 | 中立（材料乏しい / 判断困難） |
| 4 | ポジティブ（上昇期待あり） |
| 5 | 非常にポジティブ（強い上昇材料） |

ニュースが見つからない場合や API エラー時はデフォルトスコア 3（中立）が適用され、フィルターで除外されます。

### Gemini API のセットアップ

1. [Google AI Studio](https://aistudio.google.com/) で API キーを取得
2. 環境変数を設定:

```bash
# Windows (PowerShell)
$env:GEMINI_API_KEY = "あなたのAPIキー"

# Mac / Linux
export GEMINI_API_KEY="あなたのAPIキー"
```

> Gemini API キーが未設定でも、センチメント分析なしの従来モードで動作します。

---

## 注意事項

- **投資は自己責任**で行ってください。本システムの出力は投資助言ではありません。
- yfinance のデータ取得には Yahoo Finance の利用規約が適用されます。
- 大量の銘柄を短時間に取得すると、レートリミットに引っかかる場合があります。
- モデルの精度は市場環境によって変動します。定期的な再学習を推奨します。
- GitHub Actions の無料枠には月間の実行時間制限があります（2000分/月）。

---

## トラブルシューティング

| 問題 | 対処 |
|---|---|
| `ModuleNotFoundError` | `pip install -r requirements.txt` を実行 |
| `FileNotFoundError: nikkei225_tickers.csv` | 銘柄リストを作成（上記「銘柄リストの準備」参照） |
| `yfinance` でデータ取得失敗 | ネット接続を確認。時間を置いて再実行 |
| LINE 通知が届かない | 環境変数、トークン、ユーザーIDを確認 |
| モデル精度が低い | `config.py` の `TRAIN_PERIOD` を長くする、特徴量を追加 |
| Gemini API エラー | `GEMINI_API_KEY` 環境変数を確認。Rate Limit の場合は `SENTIMENT_API_SLEEP` を増やす |
| センチメント分析で全銘柄が除外される | `SENTIMENT_MIN_SCORE` を 3 に下げる、または `--sentiment` なしで実行 |

---

## ライセンス

個人利用を想定しています。
