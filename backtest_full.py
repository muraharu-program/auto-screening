"""
全スクリーニング出力ファイルを解析して有効性を包括評価するスクリプト
"""

import sys, io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")

import os, re
import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime, timedelta, date
import warnings
warnings.filterwarnings("ignore")

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = os.path.join(BASE_DIR, "outputs")
HOLD_DAYS = 5
UP_RATE = 0.05


# ──────────────────────────────────────────
# 1. ファイル解析
# ──────────────────────────────────────────

def parse_screening_file(path: str) -> dict:
    """
    screening_YYYYMMDD_HHMMSS.txt を解析して推薦銘柄リストと地合い情報を返す
    Returns:
      {
        "datetime": datetime,
        "date": date,
        "regime": str | None,
        "regime_score": float | None,
        "candidates": [ {"code":str, "close":float, "prob":float, "model":str}, ... ]
      }
    """
    fname = os.path.basename(path)
    m = re.match(r"screening_(\d{8})_(\d{6})\.txt", fname)
    if not m:
        return None
    dt = datetime.strptime(m.group(1) + m.group(2), "%Y%m%d%H%M%S")

    try:
        with open(path, encoding="utf-8", errors="replace") as f:
            text = f.read()
    except Exception:
        return None

    result = {
        "datetime": dt,
        "date": dt.date(),
        "regime": None,
        "regime_score": None,
        "candidates": [],
    }

    # 地合い判定
    regime_m = re.search(r"地合い[:：]\s*(\w+)", text)
    if regime_m:
        result["regime"] = regime_m.group(1)
    score_m = re.search(r"総合スコア[:：]\s*([+-]?\d+\.\d+)", text)
    if score_m:
        result["regime_score"] = float(score_m.group(1))

    # 推薦なし
    if "有望銘柄はありませんでした" in text or "買い推奨銘柄はありません" in text:
        return result

    # 銘柄ブロック抽出
    # パターン: 銘柄: XXXX.T または 銘柄: XXXX.T (URL)
    blocks = re.split(r"銘柄[:：]", text)
    for block in blocks[1:]:
        lines = block.strip().splitlines()
        if not lines:
            continue

        # コード
        code_m = re.match(r"\s*(\d{4}[A-Z]?\.T)", lines[0])
        if not code_m:
            continue
        code = code_m.group(1)

        # 終値
        close_m = re.search(r"終値[:：]\s*([\d,]+)", block)
        if not close_m:
            continue
        close = float(close_m.group(1).replace(",", ""))

        # 確率: ハイブリッドは「総合スコア」、通常は「上昇確率」
        hybrid_m = re.search(r"総合スコア[:：]\s*([\d.]+)%", block)
        global_m  = re.search(r"上昇確率[:：]\s*([\d.]+)%", block)
        local_raw = re.search(r"Local[:：]\s*([\d.]+)%", block)
        global_raw = re.search(r"Global[:：]\s*([\d.]+)%", block)

        if hybrid_m:
            prob = float(hybrid_m.group(1))
            model = "hybrid+sentiment" if "センチメント" in block else "hybrid"
        elif global_m:
            prob = float(global_m.group(1))
            model = "global"
        else:
            continue

        local_prob  = float(local_raw.group(1))  if local_raw  else None
        global_prob = float(global_raw.group(1)) if global_raw else None

        result["candidates"].append({
            "code": code,
            "close": close,
            "prob": prob,
            "model": model,
            "local_prob": local_prob,
            "global_prob": global_prob,
        })

    return result


def load_all_screenings(output_dir: str) -> list[dict]:
    files = sorted(
        [f for f in os.listdir(output_dir) if re.match(r"screening_\d{8}_\d{6}\.txt", f)]
    )
    results = []
    for fname in files:
        r = parse_screening_file(os.path.join(output_dir, fname))
        if r:
            results.append(r)
    return results


def dedup_by_date(screenings: list[dict]) -> list[dict]:
    """1日に複数回実行された場合、最後の1回だけ残す"""
    by_date = {}
    for s in screenings:
        d = s["date"]
        if d not in by_date or s["datetime"] > by_date[d]["datetime"]:
            by_date[d] = s
    return sorted(by_date.values(), key=lambda x: x["date"])


# ──────────────────────────────────────────
# 2. 株価取得
# ──────────────────────────────────────────

def fetch_close_series(tickers: list[str], start: date, end: date) -> dict[str, pd.Series]:
    """一括取得して銘柄ごとの終値 Series を返す"""
    cache = {}
    batch = 50
    s_str = start.strftime("%Y-%m-%d")
    e_str = (end + timedelta(days=10)).strftime("%Y-%m-%d")
    for i in range(0, len(tickers), batch):
        b = tickers[i:i+batch]
        df = yf.download(b, start=s_str, end=e_str, interval="1d",
                         auto_adjust=True, progress=False, group_by="ticker")
        for t in b:
            try:
                if isinstance(df.columns, pd.MultiIndex):
                    sub = df[t]["Close"].dropna()
                else:
                    sub = df["Close"].dropna()
                if len(sub) > 0:
                    cache[t] = sub
            except Exception:
                pass
    return cache


def get_exit_return(series: pd.Series, entry_date: date, n: int):
    """entry_date 以降の n 営業日後の終値リターンを返す"""
    sub = series[series.index.date >= entry_date]
    if len(sub) < n + 1:
        return None, None, None
    entry = float(sub.iloc[0])
    exit_ = float(sub.iloc[n])
    ret   = (exit_ - entry) / entry
    exit_date = sub.index[n].date()
    return entry, exit_, ret


# ──────────────────────────────────────────
# 3. バックテスト評価
# ──────────────────────────────────────────

def main():
    print("=" * 65)
    print("全スクリーニング結果 包括バックテスト評価")
    print("=" * 65)

    # ファイル解析
    all_sc = load_all_screenings(OUTPUT_DIR)
    daily_sc = dedup_by_date(all_sc)
    print(f"\n解析ファイル数: {len(all_sc)}件  →  1日1件に集約: {len(daily_sc)}日分")
    print(f"期間: {daily_sc[0]['date']} ～ {daily_sc[-1]['date']}")

    # 候補集計
    all_cands = []
    for sc in daily_sc:
        for c in sc["candidates"]:
            all_cands.append({
                **c,
                "screen_date": sc["date"],
                "regime": sc["regime"],
                "regime_score": sc["regime_score"],
            })

    print(f"推薦候補総数: {len(all_cands)} 銘柄 (地合いブロック後の実推薦のみ)")

    if not all_cands:
        print("推薦なし。終了。")
        return

    df_cands = pd.DataFrame(all_cands)

    # 株価取得
    tickers = list(df_cands["code"].unique())
    start_d = df_cands["screen_date"].min()
    end_d   = df_cands["screen_date"].max()
    print(f"\n{len(tickers)} 銘柄の株価データ取得中...")
    close_cache = fetch_close_series(tickers, start_d, end_d)
    print(f"取得成功: {len(close_cache)} 銘柄")

    # リターン計算
    rows = []
    for _, row in df_cands.iterrows():
        code = row["code"]
        if code not in close_cache:
            continue
        entry, exit_, ret = get_exit_return(close_cache[code], row["screen_date"], HOLD_DAYS)
        if ret is None:
            continue
        rows.append({
            "推薦日": row["screen_date"],
            "銘柄": code,
            "モデル": row["model"],
            "確率(%)": row["prob"],
            "Global(%)": row["global_prob"],
            "Local(%)": row["local_prob"],
            "地合い": row["regime"],
            "地合いスコア": row["regime_score"],
            "推薦時終値": round(entry, 0),
            "5日後終値": round(exit_, 0),
            "リターン(%)": round(ret * 100, 2),
            "的中": ret >= UP_RATE,
        })

    df = pd.DataFrame(rows)
    total = len(df)
    hits  = df["的中"].sum()
    wr    = hits / total * 100
    avg   = df["リターン(%)"].mean()
    w_avg = df.loc[df["的中"],  "リターン(%)"].mean() if hits > 0 else 0
    l_avg = df.loc[~df["的中"], "リターン(%)"].mean() if (total-hits)>0 else 0
    gains  = df.loc[df["リターン(%)"]>0, "リターン(%)"].sum()
    losses = abs(df.loc[df["リターン(%)"]<=0, "リターン(%)"].sum())
    pf = gains/losses if losses>0 else float("inf")

    print(f"\n{'='*65}")
    print("【総合成績】")
    print(f"{'='*65}")
    print(f"  評価件数    : {total}")
    print(f"  的中率      : {hits}/{total} ({wr:.1f}%)")
    print(f"  平均リターン: {avg:+.2f}%")
    print(f"  勝ち平均    : {w_avg:+.2f}%")
    print(f"  負け平均    : {l_avg:+.2f}%")
    print(f"  プロフィットファクター: {pf:.2f}")

    # モデル別
    print(f"\n【モデル別成績】")
    for m, g in df.groupby("モデル"):
        g_wr = g["的中"].sum()/len(g)*100
        g_ret= g["リターン(%)"].mean()
        print(f"  {m:<25} {g['的中'].sum()}/{len(g)} ({g_wr:.0f}%)  平均{g_ret:+.2f}%")

    # 確率帯別
    print(f"\n【確率帯別成績】")
    for lo, hi in [(50,55),(55,60),(60,65),(65,70),(70,80),(80,100)]:
        g = df[(df["確率(%)"]>=lo)&(df["確率(%)"]<hi)]
        if len(g)==0: continue
        g_wr = g["的中"].sum()/len(g)*100
        print(f"  {lo}〜{hi}%: {g['的中'].sum()}/{len(g)} ({g_wr:.0f}%)  平均{g['リターン(%)'].mean():+.2f}%")

    # 月別
    print(f"\n【月別成績】")
    df["月"] = pd.to_datetime(df["推薦日"]).dt.to_period("M")
    for mo, g in df.groupby("月"):
        g_wr = g["的中"].sum()/len(g)*100
        print(f"  {mo}: {g['的中'].sum()}/{len(g)} ({g_wr:.0f}%)  平均{g['リターン(%)'].mean():+.2f}%")

    # 過学習銘柄（Local≥90% かつ Global<50%）の成績
    overfit = df[(df["Local(%)"]>=90) & (df["Global(%)"]<50)]
    if len(overfit) > 0:
        o_wr  = overfit["的中"].sum()/len(overfit)*100
        o_ret = overfit["リターン(%)"].mean()
        print(f"\n【過学習疑い銘柄（Local≥90% & Global<50%）の成績】")
        print(f"  件数: {len(overfit)}  的中率: {o_wr:.0f}%  平均: {o_ret:+.2f}%")
        non_of = df[~((df["Local(%)"]>=90) & (df["Global(%)"]<50))]
        if len(non_of)>0:
            n_wr  = non_of["的中"].sum()/len(non_of)*100
            n_ret = non_of["リターン(%)"].mean()
            print(f"  除外した場合: 件数{len(non_of)}  的中率{n_wr:.0f}%  平均{n_ret:+.2f}%")

    # MIN_PROB フィルター別比較
    print(f"\n【MIN_PROB 閾値シミュレーション比較】")
    for thresh in [0.50, 0.55, 0.60, 0.62, 0.65, 0.70]:
        g = df[df["確率(%)"] >= thresh*100]
        if len(g)==0: continue
        g_wr = g["的中"].sum()/len(g)*100
        g_ret= g["リターン(%)"].mean()
        g_gains = g.loc[g["リターン(%)"]>0,"リターン(%)"].sum()
        g_loss  = abs(g.loc[g["リターン(%)"]<=0,"リターン(%)"].sum())
        g_pf = g_gains/g_loss if g_loss>0 else float("inf")
        print(f"  ≥{thresh*100:.0f}%: {g['的中'].sum()}/{len(g)}件 ({g_wr:.0f}%)  平均{g_ret:+.2f}%  PF={g_pf:.2f}")

    # 地合い別成績
    print(f"\n【地合いレベル別成績】")
    for regime, g in df.groupby("地合い"):
        if pd.isna(regime): continue
        g_wr = g["的中"].sum()/len(g)*100
        print(f"  {regime:<12} {g['的中'].sum()}/{len(g)} ({g_wr:.0f}%)  平均{g['リターン(%)'].mean():+.2f}%")

    # 最良・最悪銘柄
    print(f"\n【リターン上位10銘柄】")
    top10 = df.nlargest(10, "リターン(%)")
    for _, r in top10.iterrows():
        print(f"  {r['推薦日']} {r['銘柄']} {r['リターン(%)']:+.2f}%  確率{r['確率(%)']:.0f}%")

    print(f"\n【リターン下位10銘柄】")
    bot10 = df.nsmallest(10, "リターン(%)")
    for _, r in bot10.iterrows():
        print(f"  {r['推薦日']} {r['銘柄']} {r['リターン(%)']:+.2f}%  確率{r['確率(%)']:.0f}%")

    # CSV 保存
    out = os.path.join(OUTPUT_DIR, "backtest_full_result.csv")
    df.to_csv(out, index=False, encoding="utf-8-sig")
    print(f"\n詳細結果を {out} に保存しました。")


if __name__ == "__main__":
    main()
