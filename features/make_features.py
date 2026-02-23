"""
テクニカル指標を計算し、特徴量付き DataFrame を返す。
"""
import pandas as pd
import numpy as np

def calc_rsi(series, period=14):
    delta = series.diff()
    up = delta.clip(lower=0)
    down = -delta.clip(upper=0)
    ma_up = up.rolling(window=period, min_periods=period).mean()
    ma_down = down.rolling(window=period, min_periods=period).mean()
    rs = ma_up / (ma_down + 1e-9)
    return 100 - (100 / (1 + rs))

def calc_macd(series, fast=12, slow=26, signal=9):
    ema_fast = series.ewm(span=fast, adjust=False).mean()
    ema_slow = series.ewm(span=slow, adjust=False).mean()
    macd = ema_fast - ema_slow
    macd_signal = macd.ewm(span=signal, adjust=False).mean()
    macd_hist = macd - macd_signal
    return macd, macd_signal, macd_hist

def add_features(df):
    """1銘柄分の DataFrame に特徴量カラムを追加して返す"""
    c = df["Close"]
    v = df["Volume"]

    # --- 移動平均 ---
    df["ma5"]  = c.rolling(5).mean()
    df["ma25"] = c.rolling(25).mean()
    df["ma75"] = c.rolling(75).mean()

    # --- 移動平均乖離率 ---
    df["ma5_ratio"]  = (c - df["ma5"])  / (df["ma5"]  + 1e-9)
    df["ma25_ratio"] = (c - df["ma25"]) / (df["ma25"] + 1e-9)
    df["ma75_ratio"] = (c - df["ma75"]) / (df["ma75"] + 1e-9)

    # --- RSI ---
    df["rsi14"] = calc_rsi(c, 14)

    # --- MACD ---
    df["macd"], df["macd_signal"], df["macd_hist"] = calc_macd(c)

    # --- ボリンジャーバンド ---
    bb_mean = c.rolling(20).mean()
    bb_std  = c.rolling(20).std()
    df["bb_upper"] = bb_mean + 2 * bb_std
    df["bb_lower"] = bb_mean - 2 * bb_std
    df["bb_width"] = (df["bb_upper"] - df["bb_lower"]) / (bb_mean + 1e-9)

    # --- 出来高移動平均比 ---
    df["volume_ma5_ratio"] = v / (v.rolling(5).mean() + 1)

    # --- 騰落率 ---
    df["return_1d"]  = c.pct_change(1)
    df["return_5d"]  = c.pct_change(5)
    df["return_10d"] = c.pct_change(10)

    # --- ボラティリティ ---
    df["volatility_20d"] = c.pct_change().rolling(20).std()

    return df

def make_features(stock_data: dict) -> dict:
    """
    stock_data: dict[ticker, DataFrame]  (ingest_yfinance の戻り値)
    戻り値:     dict[ticker, 特徴量付き DataFrame]
    """
    features = {}
    for code, df in stock_data.items():
        if df is None or df.empty:
            continue
        df = df.copy()
        if "Close" not in df.columns:
            continue
        df = add_features(df)
        features[code] = df
    print(f"特徴量生成完了: {len(features)} 銘柄")
    return features

if __name__ == "__main__":
    dates = pd.date_range("2023-01-01", periods=100)
    dummy = {"TEST.T": pd.DataFrame({
        "Date": dates,
        "Open": np.random.rand(100)*100+1000,
        "High": np.random.rand(100)*100+1050,
        "Low":  np.random.rand(100)*100+950,
        "Close": np.random.rand(100)*100+1000,
        "Volume": np.random.randint(100000, 1000000, 100),
    })}
    feats = make_features(dummy)
    print(feats["TEST.T"].tail())