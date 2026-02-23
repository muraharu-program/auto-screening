"""
yfinanceで日本株の株価データを銘柄ごとに取得し、dict[ticker, DataFrame] で返す。
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import yfinance as yf
import pandas as pd
from config import TICKER_CSV

def load_tickers(csv_path=None):
    """ティッカーリスト(CSV)を読み込む"""
    if csv_path is None:
        csv_path = TICKER_CSV
    if os.path.exists(csv_path):
        return pd.read_csv(csv_path, header=None)[0].tolist()
    raise FileNotFoundError(
        f"{csv_path} が見つかりません。nikkei225_tickers.csv を作成してください。"
    )

def fetch_stock_data(tickers=None, period="1y", interval="1d"):
    """
    銘柄ごとに yf.download して dict[ticker, DataFrame] を返す。
    MultiIndex 問題を回避するため、1銘柄ずつダウンロードする。
    """
    if tickers is None:
        tickers = load_tickers()
    print(f"{len(tickers)} 銘柄のデータをダウンロード中...")
    data = {}
    failed = []
    for i, code in enumerate(tickers, 1):
        try:
            df = yf.download(code, period=period, interval=interval,
                             auto_adjust=True, progress=False)
            if df.empty:
                failed.append(code)
                continue
            # カラムが MultiIndex になる場合があるのでフラット化
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.get_level_values(0)
            df = df.reset_index()
            data[code] = df
        except Exception as e:
            failed.append(code)
        if i % 50 == 0:
            print(f"  {i}/{len(tickers)} 完了")
    print(f"取得成功: {len(data)} 銘柄 / 失敗: {len(failed)} 銘柄")
    return data

if __name__ == "__main__":
    stock_data = fetch_stock_data(period="3mo")
    for code, df in list(stock_data.items())[:3]:
        print(f"\n{code}:\n{df.tail(3)}")