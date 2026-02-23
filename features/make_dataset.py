"""
特徴量とラベル（N営業日後に一定以上上昇したか）を結合した学習用データセットを作成
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pandas as pd
from config import FEATURE_COLS, UP_DAYS, UP_RATE

def make_dataset(features_dict, up_days=None, up_rate=None):
    """
    features_dict: dict[ticker, 特徴量付き DataFrame]
    戻り値: 学習用 DataFrame (特徴量 + target)
    """
    if up_days is None:
        up_days = UP_DAYS
    if up_rate is None:
        up_rate = UP_RATE

    dfs = []
    for code, df in features_dict.items():
        if df is None or df.empty:
            continue
        df = df.copy()
        # ラベル: N営業日後の騰落率
        df["future_close"]  = df["Close"].shift(-up_days)
        df["future_return"]  = (df["future_close"] - df["Close"]) / df["Close"]
        df["target"] = (df["future_return"] >= up_rate).astype(int)
        df["code"] = code
        dfs.append(df)

    if not dfs:
        raise ValueError("特徴量データが空です。データ取得・特徴量生成を確認してください。")

    dataset = pd.concat(dfs, ignore_index=True)

    # 使用する特徴量カラムだけ + target + code を残す
    keep_cols = [c for c in FEATURE_COLS if c in dataset.columns]
    keep_cols += ["target", "code"]
    dataset = dataset[keep_cols].dropna()

    print(f"データセット: {len(dataset)} 行, target=1 比率: {dataset['target'].mean():.3f}")
    return dataset

if __name__ == "__main__":
    import numpy as np
    dates = pd.date_range("2023-01-01", periods=100)
    dummy_df = pd.DataFrame({
        "Date": dates, "Close": np.linspace(1000, 1200, 100),
        "Volume": np.random.randint(100000, 1000000, 100),
    })
    from features.make_features import add_features
    dummy_df = add_features(dummy_df)
    ds = make_dataset({"TEST.T": dummy_df})
    print(ds.tail())