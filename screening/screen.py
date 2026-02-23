
"""
学習済みモデルで最新データを推論し、有望銘柄をスクリーニング
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import joblib
import pandas as pd
from data.ingest_yfinance import fetch_stock_data
from features.make_features import make_features
from config import MODEL_PATH, SCREEN_PERIOD, INTERVAL, MIN_VOLUME, TOP_N, MIN_PROB

def screen_stocks(model_path=None, min_volume=None, top_n=None, min_prob=None):
    if model_path is None:
        model_path = MODEL_PATH
    if min_volume is None:
        min_volume = MIN_VOLUME
    if top_n is None:
        top_n = TOP_N
    if min_prob is None:
        min_prob = MIN_PROB

    # モデル読み込み
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"モデルファイルが見つかりません: {model_path}")
    bundle = joblib.load(model_path)
    model = bundle["model"]
    feature_cols = bundle["feature_cols"]

    # 最新データ取得 → 特徴量生成
    print("スクリーニング用データ取得中...")
    stock_data = fetch_stock_data(period=SCREEN_PERIOD, interval=INTERVAL)
    features_dict = make_features(stock_data)

    # 各銘柄の最新日データを取得
    rows = []
    for code, df in features_dict.items():
        if df.empty:
            continue
        row = df.iloc[-1].copy()
        row["code"] = code
        rows.append(row)

    if not rows:
        print("推論対象データがありません")
        return pd.DataFrame()

    latest_df = pd.DataFrame(rows)

    # 特徴量が揃っている行だけ推論
    missing = [c for c in feature_cols if c not in latest_df.columns]
    if missing:
        print(f"警告: 以下の特徴量が不足: {missing}")
        for c in missing:
            latest_df[c] = 0.0

    X = latest_df[feature_cols]
    latest_df["prob"] = model.predict_proba(X)[:, 1]

    # フィルタ: 出来高 & 上昇確率
    filtered = latest_df[
        (latest_df["Volume"] >= min_volume) &
        (latest_df["prob"] >= min_prob)
    ].copy()
    filtered = filtered.sort_values("prob", ascending=False).head(top_n)

    result = filtered[["code", "Close", "Volume", "prob"]].reset_index(drop=True)
    print(f"スクリーニング結果: {len(result)} 銘柄")
    return result

if __name__ == "__main__":
    result = screen_stocks()
    print(result)
