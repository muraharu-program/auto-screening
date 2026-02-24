"""
yfinanceで日本株の株価データを銘柄ごとに取得し、dict[ticker, DataFrame] で返す。
売買代金（終値×出来高）上位N銘柄のプレフィルタリング機能を含む。
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import time
import yfinance as yf
import pandas as pd
import numpy as np
from config import TICKER_CSV, TURNOVER_TOP_N, PREFILTER_PERIOD


def load_tickers(csv_path=None):
    """ティッカーリスト(CSV)を読み込む"""
    if csv_path is None:
        csv_path = TICKER_CSV
    if os.path.exists(csv_path):
        return pd.read_csv(csv_path, header=None)[0].tolist()
    raise FileNotFoundError(
        f"{csv_path} が見つかりません。prime.csv を作成してください。"
    )


def filter_top_by_turnover(tickers=None, top_n=None, period=None):
    """
    売買代金（終値 × 出来高）の直近平均が高い上位 top_n 銘柄を選別する。

    Parameters
    ----------
    tickers : list[str], optional
        対象銘柄リスト。省略時は prime.csv から読み込む。
    top_n : int, optional
        選別する上位銘柄数。省略時は config.TURNOVER_TOP_N (500)。
    period : str, optional
        売買代金計算用のデータ取得期間。省略時は config.PREFILTER_PERIOD ("1mo")。

    Returns
    -------
    list[str]
        売買代金上位 top_n 銘柄のティッカーリスト
    """
    if tickers is None:
        tickers = load_tickers()
    if top_n is None:
        top_n = TURNOVER_TOP_N
    if period is None:
        period = PREFILTER_PERIOD

    print(f"\n{'=' * 50}")
    print(f"[売買代金フィルタ] {len(tickers)} 銘柄 → 上位 {top_n} 銘柄を選別")
    print(f"  計算期間: 直近 {period}")
    print(f"{'=' * 50}")

    # --- バッチダウンロード（高速化）---
    # yfinance は複数銘柄をまとめてダウンロード可能
    BATCH_SIZE = 50
    turnover_list = []  # [(ticker, avg_turnover), ...]
    failed = []

    for batch_start in range(0, len(tickers), BATCH_SIZE):
        batch = tickers[batch_start : batch_start + BATCH_SIZE]
        batch_str = " ".join(batch)

        try:
            df = yf.download(
                batch_str, period=period, interval="1d",
                auto_adjust=True, progress=False, threads=True,
            )

            if df.empty:
                failed.extend(batch)
                continue

            # 複数銘柄の場合は MultiIndex (columns: (Close, ticker), ...)
            if isinstance(df.columns, pd.MultiIndex):
                for ticker in batch:
                    try:
                        close = df[("Close", ticker)]
                        volume = df[("Volume", ticker)]
                        # 売買代金 = 終値 × 出来高
                        daily_turnover = close * volume
                        avg_turnover = daily_turnover.mean()
                        if not np.isnan(avg_turnover) and avg_turnover > 0:
                            turnover_list.append((ticker, avg_turnover))
                        else:
                            failed.append(ticker)
                    except (KeyError, Exception):
                        failed.append(ticker)
            else:
                # 1銘柄のみの場合
                ticker = batch[0]
                try:
                    close = df["Close"]
                    volume = df["Volume"]
                    daily_turnover = close * volume
                    avg_turnover = daily_turnover.mean()
                    if not np.isnan(avg_turnover) and avg_turnover > 0:
                        turnover_list.append((ticker, avg_turnover))
                    else:
                        failed.append(ticker)
                except (KeyError, Exception):
                    failed.append(ticker)

        except Exception as e:
            failed.extend(batch)

        processed = min(batch_start + BATCH_SIZE, len(tickers))
        if processed % 200 == 0 or processed == len(tickers):
            print(f"  売買代金計算: {processed}/{len(tickers)} 銘柄処理済")

        # API負荷軽減
        time.sleep(0.3)

    print(f"  売買代金取得成功: {len(turnover_list)} 銘柄 / 失敗: {len(failed)} 銘柄")

    # --- 売買代金でソート → 上位 top_n を返す ---
    turnover_list.sort(key=lambda x: x[1], reverse=True)
    selected = turnover_list[:top_n]

    if selected:
        top_turnover = selected[0][1]
        bottom_turnover = selected[-1][1]
        print(f"  売買代金 Top1: {selected[0][0]} ({top_turnover:,.0f} 円)")
        print(f"  売買代金 #{len(selected)}: {selected[-1][0]} ({bottom_turnover:,.0f} 円)")

    selected_tickers = [t[0] for t in selected]
    print(f"  → {len(selected_tickers)} 銘柄を選別完了\n")
    return selected_tickers


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
    # 売買代金上位500銘柄を選別してからデータ取得するテスト
    top_tickers = filter_top_by_turnover()
    print(f"\n選別銘柄数: {len(top_tickers)}")
    print(f"先頭10銘柄: {top_tickers[:10]}")

    # 選別した銘柄でデータ取得テスト（3銘柄だけ）
    stock_data = fetch_stock_data(tickers=top_tickers[:3], period="3mo")
    for code, df in stock_data.items():
        print(f"\n{code}:\n{df.tail(3)}")