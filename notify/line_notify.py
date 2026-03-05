"""
LINE Messaging API で Broadcast 通知を送信（友だち全員に配信）
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import requests
import pandas as pd
import numpy as np
from datetime import datetime
from config import LINE_CHANNEL_ACCESS_TOKEN, OUTPUT_DIR


# ========================================================================
#  地合い通知 重複抑制（last_regime.txt）
# ========================================================================

_LAST_REGIME_PATH = os.path.join(OUTPUT_DIR, "last_regime.txt")


def _load_last_regime() -> str:
    """前回の地合い判定サマリーを読み込む"""
    try:
        with open(_LAST_REGIME_PATH, "r", encoding="utf-8") as fr:
            return fr.read().strip()
    except Exception:
        return ""


def _save_last_regime(text: str):
    """地合い判定サマリーをファイルに保存"""
    try:
        os.makedirs(OUTPUT_DIR, exist_ok=True)
        with open(_LAST_REGIME_PATH, "w", encoding="utf-8") as fw:
            fw.write(text)
    except Exception:
        pass

def format_message(candidates_df, regime=None):
    """スクリーニング結果を読みやすいメッセージに整形（LINE通知用）

    地合い情報は常に簡潔な5段階サマリーのみ表示。
    詳細な指標情報はファイル保存のみ（_build_file_message）。
    """
    lines = []

    # --- 地合い情報を先頭に表示（常に簡潔サマリー）---
    if regime is not None:
        try:
            from market.market_regime import format_regime_summary
            lines.append(format_regime_summary(regime))
            lines.append("\n" + "─" * 30 + "\n")
        except Exception:
            pass

    if candidates_df.empty:
        if regime is not None and not regime.should_buy:
            if regime.should_exit_all:
                lines.append("⚠ 地合いが非常に悪いため、今日の買い推奨銘柄はありません。")
                lines.append("保有中の全ポジションの決済（損切り含む）を検討してください。")
            elif regime.should_reduce:
                lines.append("⚠ 地合いが悪化しているため、今日の買い推奨銘柄はありません。")
                lines.append("保有中の銘柄の利確・ポジション縮小を検討してください。")
        else:
            lines.append("本日の有望銘柄はありませんでした。")
        return "\n".join(lines)

    # ハイブリッドモードか従来モードかを判定
    is_hybrid = "prob_hybrid" in candidates_df.columns
    has_sentiment = "sentiment_score" in candidates_df.columns

    if is_hybrid and has_sentiment:
        lines.append("📈 本日のスクリーニング結果（ハイブリッド＋センチメント）\n")
    elif is_hybrid:
        lines.append("📈 本日のスクリーニング結果（ハイブリッド）\n")
    else:
        lines.append("📈 本日のスクリーニング結果\n")

    for _, row in candidates_df.iterrows():
        code = row["code"]
        close = row["Close"]
        vol = row["Volume"]
        yahoo_finance_url = f"https://finance.yahoo.co.jp/quote/{code}"

        if is_hybrid:
            prob_g = row.get("prob_global", np.nan)
            prob_l = row.get("prob_local", np.nan)
            prob_h = row["prob_hybrid"]
            local_str = f"{prob_l:.1%}" if not np.isnan(prob_l) else "N/A"

            block = (
                f"銘柄: {code} ({yahoo_finance_url})\n"
                f"  終値: {close:,.0f}円\n"
                f"  出来高: {vol:,.0f}\n"
                f"  Global: {prob_g:.1%} / Local: {local_str}\n"
                f"  ▶ 総合スコア: {prob_h:.1%}\n"
            )

            # センチメント情報があれば追記
            if has_sentiment:
                s_score = row.get("sentiment_score", np.nan)
                s_reason = row.get("sentiment_reason", "")
                if not (isinstance(s_score, float) and np.isnan(s_score)):
                    emoji = {1: "🔴", 2: "🟠", 3: "⚪", 4: "🟢", 5: "🟢🟢"}.get(
                        int(s_score), "❓"
                    )
                    block += f"  {emoji} センチメント: {int(s_score)}/5 — {s_reason}\n"

            lines.append(block)
        else:
            prob = row["prob"]
            lines.append(
                f"銘柄: {code} ({yahoo_finance_url})\n"
                f"  終値: {close:,.0f}円\n"
                f"  出来高: {vol:,.0f}\n"
                f"  上昇確率: {prob:.1%}\n"
            )
    return "\n".join(lines)

def _build_file_message(candidates, regime):
    """ファイル保存用メッセージ（LINE通知メッセージ＋地合い詳細を追記）"""
    if isinstance(candidates, pd.DataFrame):
        msg = format_message(candidates, regime=regime)
    elif isinstance(candidates, str):
        msg = candidates
    else:
        msg = str(candidates)

    # 地合い詳細をファイルにのみ追記
    if regime is not None:
        try:
            from market.market_regime import format_regime_message
            msg += "\n\n" + "=" * 40 + "\n"
            msg += "【地合い判定 詳細ログ】\n"
            msg += "=" * 40 + "\n"
            msg += format_regime_message(regime)
        except Exception:
            pass

    return msg


def send_line_message(candidates, token=None, user_id=None, regime=None):
    """
    LINE Messaging API で Broadcast メッセージを送信（友だち全員に配信）
    candidates: DataFrame or str
    regime: MarketRegime or None
    """
    if token is None:
        token = LINE_CHANNEL_ACCESS_TOKEN

    # LINE通知用メッセージ（簡潔版）
    if isinstance(candidates, pd.DataFrame):
        line_msg = format_message(candidates, regime=regime)
    elif isinstance(candidates, str):
        line_msg = candidates
    else:
        line_msg = str(candidates)

    # LINE Messaging API の上限は 5000 文字
    line_msg = line_msg[:5000]

    # ファイル保存用メッセージ（詳細版）
    file_msg = _build_file_message(candidates, regime)

    # 結果をテキストファイルに保存（LINE送信の有無に関わらず行う）
    try:
        os.makedirs(OUTPUT_DIR, exist_ok=True)
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        fname = os.path.join(OUTPUT_DIR, f"screening_{ts}.txt")
        with open(fname, "w", encoding="utf-8") as fw:
            fw.write(file_msg)
        print(f"結果をファイル出力: {fname}")
    except Exception as e:
        print(f"結果ファイル保存エラー: {e}")

    # --- 最終地合い通知の記録管理 ---
    prev_regime = _load_last_regime()
    cur_regime = "" if regime is None else regime.summary

    # 通知をスキップすべきか判定
    if isinstance(candidates, pd.DataFrame) and candidates.empty:
        if prev_regime == cur_regime and cur_regime != "":
            print("前回と地合い判定が同じためLINE通知をスキップします。")
            return None
    # （候補が存在する場合は常に送信）

    # 保存するのは地合いサマリーのみ
    if cur_regime:
        _save_last_regime(cur_regime)

    # LINE送信はトークンが設定されている場合のみ実行
    if not token:
        print("警告: LINE トークンが未設定です。LINE送信をスキップしました。")
        return None

    url = "https://api.line.me/v2/bot/message/broadcast"
    headers = {
        "Authorization": f"Bearer {token}",
        "Content-Type": "application/json",
    }
    payload = {
        "messages": [{"type": "text", "text": line_msg}],
    }
    try:
        r = requests.post(url, headers=headers, json=payload)
        print(f"LINE Broadcast 通知ステータス: {r.status_code}")
        if r.status_code != 200:
            print(f"LINE エラー: {r.text}")
        return r.status_code
    except Exception as e:
        print(f"LINE 送信エラー: {e}")
        return None

if __name__ == "__main__":
    # テスト（従来形式）
    dummy = pd.DataFrame({
        "code": ["7203.T", "6758.T"],
        "Close": [2500, 13000],
        "Volume": [5000000, 3000000],
        "prob": [0.82, 0.75],
    })
    print(format_message(dummy))

    # テスト（ハイブリッド形式）
    print("\n--- ハイブリッド形式 ---")
    dummy_h = pd.DataFrame({
        "code": ["7203.T", "6758.T"],
        "Close": [2500, 13000],
        "Volume": [5000000, 3000000],
        "prob_global": [0.82, 0.75],
        "prob_local": [0.88, float("nan")],
        "prob_hybrid": [0.844, 0.75],
    })
    print(format_message(dummy_h))
