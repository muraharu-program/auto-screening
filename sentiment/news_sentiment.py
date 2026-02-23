"""
ニュース・センチメント分析フィルター
  - Yahoo!ファイナンス(日本)から銘柄別ニュース見出しを取得
  - Gemini API でスイングトレード目線のセンチメントを 1-5 で判定
  - スコア 4 以上の銘柄のみを最終通知リストとして返す

必要パッケージ:
  pip install feedparser google-generativeai requests beautifulsoup4
"""
import sys
import os
import re
import json
import time
from datetime import datetime, timedelta

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pandas as pd
import numpy as np

# --- オプショナル依存 ---------------------------------------------------
try:
    import feedparser
except ImportError:
    feedparser = None
    print("警告: feedparser が未インストールです。pip install feedparser を実行してください。")

try:
    import requests
    from bs4 import BeautifulSoup
except ImportError:
    requests = None
    BeautifulSoup = None
    print("警告: requests / beautifulsoup4 が未インストールです。")

try:
    import google.generativeai as genai
except ImportError:
    genai = None
    print("警告: google-generativeai が未インストールです。pip install google-generativeai を実行してください。")

from config import (
    GEMINI_API_KEY,
    GEMINI_MODEL,
    SENTIMENT_TOP_N,
    SENTIMENT_MIN_SCORE,
    SENTIMENT_DEFAULT_SCORE,
    SENTIMENT_API_SLEEP,
    NEWS_LOOKBACK_DAYS,
)


# ====================================================================== #
#  1. ニュース見出し取得
# ====================================================================== #

def _ticker_to_code(ticker: str) -> str:
    """
    '7203.T' → '7203' のように末尾の .T を除去して証券コードを返す。
    """
    return ticker.replace(".T", "").replace(".t", "")


def fetch_news_rss(ticker: str, lookback_days: int = None) -> list[str]:
    """
    Yahoo!ファイナンス(日本)の銘柄別ニュース RSS から
    直近 lookback_days 日分の見出しを取得する。

    Parameters
    ----------
    ticker : str  例: '7203.T'
    lookback_days : int  遡り日数（デフォルト: config.NEWS_LOOKBACK_DAYS）

    Returns
    -------
    list[str]  ニュース見出しのリスト（空リストの場合あり）
    """
    if lookback_days is None:
        lookback_days = NEWS_LOOKBACK_DAYS

    code = _ticker_to_code(ticker)
    headlines: list[str] = []

    # --- 方法 1: RSS フィード (feedparser) ---
    if feedparser is not None:
        try:
            rss_url = f"https://finance.yahoo.co.jp/rss/company/{code}"
            feed = feedparser.parse(rss_url)
            cutoff = datetime.now() - timedelta(days=lookback_days)
            for entry in feed.entries:
                # 日付フィルタ
                published = entry.get("published_parsed") or entry.get("updated_parsed")
                if published:
                    pub_dt = datetime(*published[:6])
                    if pub_dt < cutoff:
                        continue
                title = entry.get("title", "").strip()
                if title:
                    headlines.append(title)
        except Exception as e:
            print(f"  RSS 取得エラー ({ticker}): {e}")

    # --- 方法 2: フォールバック — HTML スクレイピング ---
    if not headlines and requests is not None and BeautifulSoup is not None:
        try:
            page_url = f"https://finance.yahoo.co.jp/quote/{code}.T/news"
            headers = {
                "User-Agent": (
                    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                    "AppleWebKit/537.36 (KHTML, like Gecko) "
                    "Chrome/120.0.0.0 Safari/537.36"
                ),
            }
            resp = requests.get(page_url, headers=headers, timeout=15)
            resp.raise_for_status()
            soup = BeautifulSoup(resp.text, "lxml")

            # Yahoo!ファイナンス のニュース見出しリンクを探索
            # セレクターは DOM 変更に伴い調整が必要な場合がある
            for a_tag in soup.select("a[href*='/news/']"):
                text = a_tag.get_text(strip=True)
                if text and len(text) > 5:
                    headlines.append(text)

            # 重複排除
            headlines = list(dict.fromkeys(headlines))
        except Exception as e:
            print(f"  HTML スクレイピングエラー ({ticker}): {e}")

    return headlines


# ====================================================================== #
#  2. Gemini API センチメント判定
# ====================================================================== #

_SENTIMENT_PROMPT_TEMPLATE = """\
あなたは日本株のスイングトレード（数日〜2週間）の専門アナリストです。
以下は銘柄コード {ticker} に関する直近のニュース見出しです。

--- ニュース見出し ---
{headlines_text}
--- ここまで ---

上記のニュースを踏まえ、この銘柄の今後1〜2週間の株価への影響を
スイングトレード目線で評価してください。

必ず以下の JSON 形式のみで回答してください。それ以外のテキストは一切出力しないでください。
{{"score": <1〜5の整数>, "reason": "<50文字以内の日本語の理由>"}}

スコアの基準:
  1 = 非常にネガティブ（大幅下落リスク）
  2 = ネガティブ（下落圧力あり）
  3 = 中立（材料乏しい / 判断困難）
  4 = ポジティブ（上昇期待あり）
  5 = 非常にポジティブ（強い上昇材料）
"""


def _parse_gemini_response(text: str) -> dict:
    """
    Gemini の応答テキストから JSON を抽出してパースする。
    応答にコードフェンスや余計なテキストが含まれる場合も対応する。

    Returns
    -------
    dict  {"score": int, "reason": str}  失敗時は None
    """
    if not text:
        return None

    # コードフェンスの除去
    text = re.sub(r"```json\s*", "", text)
    text = re.sub(r"```\s*", "", text)
    text = text.strip()

    # JSON ブロックの抽出（{...} を探す）
    match = re.search(r"\{[^}]+\}", text, re.DOTALL)
    if not match:
        return None

    try:
        data = json.loads(match.group())
        score = int(data.get("score", 0))
        reason = str(data.get("reason", ""))[:50]
        if 1 <= score <= 5:
            return {"score": score, "reason": reason}
    except (json.JSONDecodeError, ValueError, TypeError):
        pass

    return None


def analyze_sentiment_gemini(
    ticker: str,
    headlines: list[str],
    api_key: str = None,
    model_name: str = None,
) -> dict:
    """
    Gemini API を使って銘柄のセンチメントを判定する。

    Parameters
    ----------
    ticker : str
    headlines : list[str]
    api_key : str  省略時は config.GEMINI_API_KEY
    model_name : str  省略時は config.GEMINI_MODEL

    Returns
    -------
    dict  {"score": int, "reason": str}
    """
    if api_key is None:
        api_key = GEMINI_API_KEY
    if model_name is None:
        model_name = GEMINI_MODEL

    default_result = {"score": SENTIMENT_DEFAULT_SCORE, "reason": "判定不能"}

    if genai is None:
        print(f"  [{ticker}] google-generativeai 未インストール → デフォルトスコア")
        return default_result

    if not api_key:
        print(f"  [{ticker}] GEMINI_API_KEY 未設定 → デフォルトスコア")
        return default_result

    if not headlines:
        return {"score": SENTIMENT_DEFAULT_SCORE, "reason": "ニュースなし"}

    # 見出しを最大 20 件に制限（トークン節約）
    headlines_trimmed = headlines[:20]
    headlines_text = "\n".join(f"・{h}" for h in headlines_trimmed)

    prompt = _SENTIMENT_PROMPT_TEMPLATE.format(
        ticker=ticker,
        headlines_text=headlines_text,
    )

    try:
        genai.configure(api_key=api_key)
        model = genai.GenerativeModel(model_name)
        response = model.generate_content(
            prompt,
            generation_config=genai.types.GenerationConfig(
                temperature=0.1,
                max_output_tokens=256,
            ),
        )
        raw_text = response.text
        result = _parse_gemini_response(raw_text)
        if result is not None:
            return result
        else:
            print(f"  [{ticker}] Gemini 応答パース失敗: {raw_text[:100]}")
            return default_result

    except Exception as e:
        print(f"  [{ticker}] Gemini API エラー: {e}")
        return default_result


# ====================================================================== #
#  3. メインフィルター関数
# ====================================================================== #

def apply_sentiment_filter(
    candidates_df: pd.DataFrame,
    top_n: int = None,
    min_score: int = None,
    api_sleep: float = None,
) -> pd.DataFrame:
    """
    ハイブリッドスクリーニング結果に対してニュース・センチメントフィルターを適用する。

    処理フロー:
      1. prob_hybrid 上位 top_n 銘柄を抽出
      2. 各銘柄のニュース見出しを取得
      3. Gemini API でセンチメントスコア（1-5）を判定
      4. スコア min_score 以上の銘柄のみを返す

    Parameters
    ----------
    candidates_df : pd.DataFrame
        screen_hybrid() の出力。columns に 'code', 'prob_hybrid' を含むこと。
    top_n : int
        センチメント分析対象の上位銘柄数（デフォルト: config.SENTIMENT_TOP_N）
    min_score : int
        最終通知に残す最低センチメントスコア（デフォルト: config.SENTIMENT_MIN_SCORE）
    api_sleep : float
        Gemini API コール間のスリープ秒数（デフォルト: config.SENTIMENT_API_SLEEP）

    Returns
    -------
    pd.DataFrame
        フィルタ後の DataFrame。追加カラム: sentiment_score, sentiment_reason
    """
    if top_n is None:
        top_n = SENTIMENT_TOP_N
    if min_score is None:
        min_score = SENTIMENT_MIN_SCORE
    if api_sleep is None:
        api_sleep = SENTIMENT_API_SLEEP

    if candidates_df.empty:
        print("センチメントフィルター: 入力が空です")
        return candidates_df

    # --- Step 1: 上位 top_n 銘柄を抽出 ---
    df = candidates_df.sort_values("prob_hybrid", ascending=False).head(top_n).copy()
    print(f"\n{'='*50}")
    print(f"[センチメント分析] 対象: 上位 {len(df)} 銘柄")
    print(f"{'='*50}")

    sentiment_scores = []
    sentiment_reasons = []

    for idx, row in df.iterrows():
        ticker = row["code"]
        print(f"\n--- {ticker} ---")

        # --- Step 2: ニュース取得 ---
        try:
            headlines = fetch_news_rss(ticker)
            if headlines:
                print(f"  ニュース {len(headlines)} 件取得")
                for h in headlines[:3]:
                    print(f"    • {h}")
                if len(headlines) > 3:
                    print(f"    ... 他 {len(headlines) - 3} 件")
            else:
                print(f"  ニュースが見つかりませんでした → デフォルトスコア {SENTIMENT_DEFAULT_SCORE}")
        except Exception as e:
            print(f"  ニュース取得エラー: {e}")
            headlines = []

        # --- Step 3: Gemini でセンチメント判定 ---
        result = analyze_sentiment_gemini(ticker, headlines)
        score = result["score"]
        reason = result["reason"]
        sentiment_scores.append(score)
        sentiment_reasons.append(reason)

        emoji = {1: "🔴", 2: "🟠", 3: "⚪", 4: "🟢", 5: "🟢🟢"}.get(score, "❓")
        print(f"  センチメント: {emoji} {score}/5 — {reason}")

        # --- Rate Limit 配慮 ---
        if api_sleep > 0:
            time.sleep(api_sleep)

    df["sentiment_score"] = sentiment_scores
    df["sentiment_reason"] = sentiment_reasons

    # --- Step 4: スコアでフィルタ ---
    filtered = df[df["sentiment_score"] >= min_score].copy()
    filtered = filtered.sort_values("prob_hybrid", ascending=False).reset_index(drop=True)

    print(f"\n{'='*50}")
    print(f"[センチメント分析完了]")
    print(f"  分析対象: {len(df)} 銘柄")
    print(f"  スコア {min_score} 以上: {len(filtered)} 銘柄")
    print(f"{'='*50}")

    return filtered


# ====================================================================== #
#  4. スタンドアロン実行（テスト用）
# ====================================================================== #

if __name__ == "__main__":
    # テスト用ダミーデータ（screen_hybrid の出力形式を模倣）
    dummy_df = pd.DataFrame({
        "code": ["7203.T", "6758.T", "9984.T", "8306.T", "6701.T"],
        "Close": [2500, 13000, 4300, 1800, 3900],
        "Volume": [5000000, 3000000, 29000000, 15000000, 11000000],
        "prob_global": [0.72, 0.68, 0.55, 0.50, 0.72],
        "prob_local": [0.85, 0.70, 0.99, 0.80, 0.81],
        "prob_hybrid": [0.77, 0.69, 0.73, 0.62, 0.76],
    })

    print("=== ニュース取得テスト ===")
    for ticker in dummy_df["code"][:2]:
        headlines = fetch_news_rss(ticker)
        print(f"{ticker}: {len(headlines)} 件")
        for h in headlines[:3]:
            print(f"  • {h}")

    print("\n=== センチメントフィルター テスト ===")
    result = apply_sentiment_filter(dummy_df, top_n=5, min_score=4)
    if not result.empty:
        print(result.to_string(index=False))
    else:
        print("フィルタ後の銘柄なし")
