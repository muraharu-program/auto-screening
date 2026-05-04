"""
長期バックテスト: 学習済みグローバルモデルを使った過去シミュレーション

手法:
  - 直近1年分の株価データを取得
  - 毎週月曜日（または翌営業日）をシミュレーション日として設定
  - 各日の直近データで特徴量を生成 → グローバルモデルで確率を予測
  - 予測確率 >= MIN_PROB (0.62) の銘柄を選出し、5営業日後のリターンを記録

NOTE: 現在の学習済みモデルを全期間に適用するため軽微な未来情報漏洩があるが、
      スクリーニングシステムの方向性を検証するには十分な精度。
"""

import sys
import io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")

import os
import joblib
import numpy as np
import pandas as pd
import yfinance as yf
import warnings
from datetime import datetime, timedelta, date
import gc

warnings.filterwarnings("ignore")

# プロジェクトルートをパスに追加
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, BASE_DIR)

from features.make_features import make_features
from data.ingest_yfinance import filter_top_by_turnover

# ===================== 設定 =====================
GLOBAL_MODEL_PATH = os.path.join(BASE_DIR, "models", "global_model.pkl")
MIN_PROB_NEW = 0.62     # 改善後の閾値
MIN_PROB_OLD = 0.50     # 改善前の閾値（比較用）
MIN_VOLUME   = 500_000
TOP_N        = 6
HOLD_DAYS    = 5        # 5営業日後にエグジット
UP_RATE      = 0.05     # 的中の定義
LOOKBACK_MONTHS = 9     # 何ヶ月前まで遡るか

# 過学習検出設定
LOCAL_OVERFIT_MAX_LOCAL  = 0.90
LOCAL_OVERFIT_MIN_GLOBAL = 0.50


def business_days_range(start: date, end: date, step_weeks: int = 1) -> list[date]:
    """start〜end の範囲で step_weeks 週ごとの月曜日を返す"""
    days = []
    cur = start
    # 月曜日に合わせる
    while cur.weekday() != 0:
        cur += timedelta(days=1)
    while cur <= end:
        days.append(cur)
        cur += timedelta(weeks=step_weeks)
    return days


def get_close_n_days_after(series: pd.Series, ref_date: date, n: int):
    """ref_date 以降で n 番目の終値を返す。データ不足なら None。"""
    subset = series[series.index.date >= ref_date]
    if len(subset) < n + 1:
        return None, None
    return float(subset.iloc[n]), subset.index[n].date()


def simulate_one_day(
    sim_date: date,
    all_prices: dict[str, pd.DataFrame],
    model,
    feature_cols: list[str],
    min_prob: float,
) -> pd.DataFrame:
    """
    sim_date 時点での特徴量を生成し、推薦銘柄を返す。
    all_prices: {ticker: OHLCV DataFrame (全期間)} の辞書
    """
    rows = []
    for ticker, df in all_prices.items():
        # sim_date 以前のデータのみを使う（未来情報を使わないよう制限）
        hist = df[df.index.date <= sim_date]
        if len(hist) < 80:  # 特徴量生成に最低80日必要（MA75など）
            continue
        rows.append((ticker, hist))

    if not rows:
        return pd.DataFrame()

    # make_features に渡す形式に変換
    stock_dict = {ticker: hist for ticker, hist in rows}
    features_dict = make_features(stock_dict)

    latest_rows = []
    for code, feat_df in features_dict.items():
        if feat_df.empty:
            continue
        r = feat_df.iloc[-1].copy()
        r["code"] = code
        latest_rows.append(r)

    if not latest_rows:
        return pd.DataFrame()

    latest_df = pd.DataFrame(latest_rows)

    # 特徴量不足を補完
    for c in feature_cols:
        if c not in latest_df.columns:
            latest_df[c] = 0.0

    X = latest_df[feature_cols].astype(float)
    latest_df["prob"] = model.predict_proba(X)[:, 1]

    # 出来高・確率フィルタ
    result = latest_df[
        (latest_df["Volume"] >= MIN_VOLUME) &
        (latest_df["prob"] >= min_prob)
    ].sort_values("prob", ascending=False).head(TOP_N)

    return result[["code", "Close", "Volume", "prob"]].reset_index(drop=True)


def main():
    # ===================== モデルロード =====================
    if not os.path.exists(GLOBAL_MODEL_PATH):
        print(f"エラー: グローバルモデルが見つかりません: {GLOBAL_MODEL_PATH}")
        print("先に python main.py --hybrid-train を実行してください。")
        return

    print("グローバルモデルをロード中...")
    bundle = joblib.load(GLOBAL_MODEL_PATH)
    model = bundle["model"]
    feature_cols = bundle["feature_cols"]
    print(f"  特徴量数: {len(feature_cols)}")

    # ===================== 銘柄リスト取得 =====================
    print("\n売買代金上位銘柄リストを取得中...")
    tickers = filter_top_by_turnover()
    print(f"  対象銘柄数: {len(tickers)}")

    # ===================== 価格データ一括取得 =====================
    end_date   = date.today()
    start_date = end_date - timedelta(days=int(LOOKBACK_MONTHS * 30.5) + 60)
    print(f"\n価格データ取得: {start_date} ～ {end_date}")

    all_prices: dict[str, pd.DataFrame] = {}
    batch_size = 50
    for i in range(0, len(tickers), batch_size):
        batch = tickers[i:i + batch_size]
        df_batch = yf.download(
            batch,
            start=start_date.strftime("%Y-%m-%d"),
            end=(end_date + timedelta(days=1)).strftime("%Y-%m-%d"),
            interval="1d",
            auto_adjust=True,
            progress=False,
            group_by="ticker",
        )
        for ticker in batch:
            try:
                if isinstance(df_batch.columns, pd.MultiIndex):
                    sub = df_batch[ticker].dropna(subset=["Close"])
                else:
                    sub = df_batch.dropna(subset=["Close"])
                if len(sub) >= 80:
                    all_prices[ticker] = sub
            except Exception:
                pass
        gc.collect()
        print(f"  取得済み: {min(i + batch_size, len(tickers))}/{len(tickers)}", end="\r")

    print(f"\n  有効銘柄数: {len(all_prices)}")

    # 出来高が常に必要なので Close と Volume を持つ辞書に整理
    # (all_prices は既に DataFrame なのでそのまま使用)

    # ===================== シミュレーション日生成 =====================
    sim_start = end_date - timedelta(days=int(LOOKBACK_MONTHS * 30.5))
    sim_dates = business_days_range(sim_start, end_date - timedelta(weeks=1), step_weeks=1)
    print(f"\nシミュレーション日数: {len(sim_dates)} 日 ({sim_dates[0]} ～ {sim_dates[-1]})")

    # Close 系列を別途キャッシュ（リターン計算用）
    close_cache: dict[str, pd.Series] = {
        t: df["Close"].dropna()
        for t, df in all_prices.items()
    }

    # ===================== シミュレーション実行 =====================
    records_new = []  # 改善後 (MIN_PROB=0.62)
    records_old = []  # 改善前 (MIN_PROB=0.50) 比較用

    for sim_date in sim_dates:
        # 改善後
        cands_new = simulate_one_day(sim_date, all_prices, model, feature_cols, MIN_PROB_NEW)
        # 改善前
        cands_old = simulate_one_day(sim_date, all_prices, model, feature_cols, MIN_PROB_OLD)

        for cands, records in [(cands_new, records_new), (cands_old, records_old)]:
            for _, row in cands.iterrows():
                code = row["code"]
                prob = row["prob"]
                entry_price = row["Close"]

                if code not in close_cache:
                    continue
                exit_price, exit_date = get_close_n_days_after(
                    close_cache[code], sim_date, HOLD_DAYS
                )
                if exit_price is None:
                    continue

                ret = (exit_price - entry_price) / entry_price
                records.append({
                    "sim_date": sim_date,
                    "code": code,
                    "prob": round(prob * 100, 1),
                    "entry": round(entry_price, 0),
                    "exit": round(exit_price, 0),
                    "exit_date": exit_date,
                    "return_pct": round(ret * 100, 2),
                    "hit": ret >= UP_RATE,
                })

    # ===================== 集計 =====================
    def summarize(records, label):
        if not records:
            print(f"\n{label}: データなし")
            return
        df = pd.DataFrame(records)
        total = len(df)
        hits  = df["hit"].sum()
        wr    = hits / total * 100
        avg   = df["return_pct"].mean()
        avg_w = df.loc[df["hit"],  "return_pct"].mean() if hits > 0 else 0
        avg_l = df.loc[~df["hit"], "return_pct"].mean() if (total - hits) > 0 else 0
        gains  = df.loc[df["return_pct"] > 0, "return_pct"].sum()
        losses = abs(df.loc[df["return_pct"] <= 0, "return_pct"].sum())
        pf = gains / losses if losses > 0 else float("inf")

        print(f"\n{'='*60}")
        print(f"【{label}】")
        print(f"{'='*60}")
        print(f"  推薦件数    : {total}")
        print(f"  的中率      : {hits}/{total} ({wr:.1f}%)")
        print(f"  平均リターン: {avg:+.2f}%")
        print(f"  勝ち平均    : {avg_w:+.2f}%")
        print(f"  負け平均    : {avg_l:+.2f}%")
        print(f"  プロフィットファクター: {pf:.2f}")

        # 月別集計
        df["month"] = pd.to_datetime(df["sim_date"]).dt.to_period("M")
        print(f"\n  【月別成績】")
        for month, grp in df.groupby("month"):
            g_hits = grp["hit"].sum()
            g_tot  = len(grp)
            g_wr   = g_hits / g_tot * 100
            g_ret  = grp["return_pct"].mean()
            print(f"    {month}: {g_hits}/{g_tot} ({g_wr:.0f}%)  平均{g_ret:+.2f}%")

        # 確率帯別
        print(f"\n  【確率帯別成績】")
        for lo, hi in [(50, 62), (62, 70), (70, 80), (80, 100)]:
            grp = df[(df["prob"] >= lo) & (df["prob"] < hi)]
            if len(grp) == 0:
                continue
            g_wr  = grp["hit"].sum() / len(grp) * 100
            g_ret = grp["return_pct"].mean()
            print(f"    {lo}%〜{hi}%: {grp['hit'].sum()}/{len(grp)} ({g_wr:.0f}%)  平均{g_ret:+.2f}%")

        # CSV 保存
        safe_label = label.replace(" ", "_").replace("(", "").replace(")", "")
        path = os.path.join(BASE_DIR, "outputs", f"backtest_long_{safe_label}.csv")
        df.to_csv(path, index=False, encoding="utf-8-sig")
        print(f"\n  → {path} に保存しました。")

    print(f"\n{'='*60}")
    print("長期バックテスト結果")
    print(f"期間: {sim_dates[0]} ～ {sim_dates[-1]}  ({len(sim_dates)}週)")
    print(f"評価基準: {HOLD_DAYS}営業日後 +{UP_RATE*100:.0f}%以上を「的中」")
    print(f"{'='*60}")

    summarize(records_new, f"改善後 MIN_PROB={MIN_PROB_NEW}")
    summarize(records_old, f"改善前 MIN_PROB={MIN_PROB_OLD} (比較)")


if __name__ == "__main__":
    main()
