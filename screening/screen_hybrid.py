"""
ハイブリッド推論スクリーニング
  - グローバルモデル + ローカルモデルの予測確率を重み付け合成
  - ローカルモデルが存在しない銘柄はグローバルのみで判定
"""
import sys
import os
import gc

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import joblib
import pandas as pd
import numpy as np
from data.ingest_yfinance import fetch_stock_data
from features.make_features import make_features
from config import (
    GLOBAL_MODEL_PATH,
    LOCAL_MODEL_DIR,
    GLOBAL_WEIGHT,
    LOCAL_WEIGHT,
    SCREEN_PERIOD,
    INTERVAL,
    MIN_VOLUME,
    TOP_N,
    MIN_PROB,
    SENTIMENT_TOP_N,
)


def _load_global_model(path=None):
    """グローバルモデルをロード"""
    if path is None:
        path = GLOBAL_MODEL_PATH
    if not os.path.exists(path):
        raise FileNotFoundError(f"グローバルモデルが見つかりません: {path}")
    bundle = joblib.load(path)
    return bundle["model"], bundle["feature_cols"]


def _load_local_model(ticker, local_dir=None):
    """
    ローカルモデルをロード。存在しなければ (None, None) を返す。
    """
    if local_dir is None:
        local_dir = LOCAL_MODEL_DIR
    safe_ticker = ticker.replace(".", "_")
    path = os.path.join(local_dir, f"local_model_{safe_ticker}.pkl")
    if not os.path.exists(path):
        return None, None
    bundle = joblib.load(path)
    return bundle["model"], bundle["feature_cols"]


def screen_hybrid(
    global_model_path=None,
    local_model_dir=None,
    global_weight=None,
    local_weight=None,
    min_volume=None,
    top_n=None,
    min_prob=None,
    use_sentiment=False,
):
    """
    ハイブリッドスクリーニング

    Parameters
    ----------
    use_sentiment : bool
        True の場合、Gemini API によるニュース・センチメント分析フィルターを
        最終段階で適用する。対象は prob_hybrid 上位 SENTIMENT_TOP_N 銘柄。

    Returns
    -------
    pd.DataFrame
        columns: code, Close, Volume, prob_global, prob_local, prob_hybrid
        （use_sentiment=True の場合、sentiment_score, sentiment_reason も追加）
    """
    if global_weight is None:
        global_weight = GLOBAL_WEIGHT
    if local_weight is None:
        local_weight = LOCAL_WEIGHT
    if min_volume is None:
        min_volume = MIN_VOLUME
    if top_n is None:
        top_n = TOP_N
    if min_prob is None:
        min_prob = MIN_PROB

    # --- グローバルモデルロード ---
    global_model, g_feature_cols = _load_global_model(global_model_path)

    # --- 最新データ取得 → 特徴量生成 ---
    print("ハイブリッドスクリーニング用データ取得中...")
    stock_data = fetch_stock_data(period=SCREEN_PERIOD, interval=INTERVAL)
    features_dict = make_features(stock_data)

    # メモリ節約: 生データは不要
    del stock_data
    gc.collect()

    # --- 各銘柄の最新日データを収集 ---
    rows = []
    for code, df in features_dict.items():
        if df.empty:
            continue
        row = df.iloc[-1].copy()
        row["code"] = code
        rows.append(row)

    del features_dict
    gc.collect()

    if not rows:
        print("推論対象データがありません")
        return pd.DataFrame()

    latest_df = pd.DataFrame(rows)

    # --- 欠損特徴量の補完 ---
    missing_g = [c for c in g_feature_cols if c not in latest_df.columns]
    if missing_g:
        print(f"警告: グローバル特徴量不足 → 0 埋め: {missing_g}")
        for c in missing_g:
            latest_df[c] = 0.0

    # ================================================================ #
    #  グローバル推論（一括）
    # ================================================================ #
    X_global = latest_df[g_feature_cols]
    latest_df["prob_global"] = global_model.predict_proba(X_global)[:, 1]

    # ================================================================ #
    #  ローカル推論（銘柄ごと）
    # ================================================================ #
    local_probs = np.full(len(latest_df), np.nan)

    local_hit = 0
    for idx in range(len(latest_df)):
        row = latest_df.iloc[idx]
        ticker = row["code"]
        local_model, l_feature_cols = _load_local_model(ticker, local_model_dir)
        if local_model is None:
            continue

        # 特徴量を揃える
        x = {}
        for c in l_feature_cols:
            x[c] = row.get(c, 0.0)
        x_df = pd.DataFrame([x])
        try:
            prob = local_model.predict_proba(x_df)[:, 1][0]
            local_probs[idx] = prob
            local_hit += 1
        except Exception:
            pass

        # メモリ解放
        del local_model, l_feature_cols, x_df
        gc.collect()

    latest_df["prob_local"] = local_probs

    print(f"ローカルモデル適用: {local_hit}/{len(latest_df)} 銘柄")

    # ================================================================ #
    #  ハイブリッドスコア計算
    # ================================================================ #
    def _hybrid_score(row):
        pg = row["prob_global"]
        pl = row["prob_local"]
        if np.isnan(pl):
            # ローカルモデルなし → グローバルのみ
            return pg
        # 重み付け平均
        return global_weight * pg + local_weight * pl

    latest_df["prob_hybrid"] = latest_df.apply(_hybrid_score, axis=1)

    # --- フィルタ ---
    filtered = latest_df[
        (latest_df["Volume"] >= min_volume)
        & (latest_df["prob_hybrid"] >= min_prob)
    ].copy()

    if use_sentiment:
        # センチメント分析: 上位 SENTIMENT_TOP_N 銘柄を広めに取って分析
        sentiment_top = max(top_n, SENTIMENT_TOP_N)
        pre_sentiment = filtered.sort_values("prob_hybrid", ascending=False).head(sentiment_top)
        pre_sentiment = pre_sentiment[
            ["code", "Close", "Volume", "prob_global", "prob_local", "prob_hybrid"]
        ].reset_index(drop=True)

        try:
            from sentiment.news_sentiment import apply_sentiment_filter
            result = apply_sentiment_filter(pre_sentiment)
            # センチメントフィルタ後、最終的に top_n 件に絞る
            result = result.head(top_n).reset_index(drop=True)
            print(f"センチメントフィルター適用後: {len(result)} 銘柄")
            return result
        except Exception as e:
            print(f"センチメント分析エラー（フォールバック: 従来フィルタ）: {e}")
            # フォールバック: センチメント分析なしの従来結果を返す

    filtered = filtered.sort_values("prob_hybrid", ascending=False).head(top_n)
    result = filtered[
        ["code", "Close", "Volume", "prob_global", "prob_local", "prob_hybrid"]
    ].reset_index(drop=True)

    print(f"ハイブリッドスクリーニング結果: {len(result)} 銘柄")
    return result


if __name__ == "__main__":
    result = screen_hybrid()
    if not result.empty:
        print(result.to_string(index=False))
    else:
        print("該当銘柄なし")
