"""
マーケット地合い判定モジュール

日経平均・TOPIX・VIX・ドル円など複数指標を組み合わせて、
日本株市場全体の地合いをリアルタイムに判定する。

判定レベル:
  STRONG_BUY  … 地合い良好。積極的に買いエントリー可
  BUY         … 通常の環境。買いエントリー可
  CAUTION     … 注意。新規買いは控えめに／ポジション縮小推奨
  DANGER      … 危険。新規買い停止／利確推奨
  CRISIS      … 暴落懸念。全ポジション決済（損切り含む）推奨
"""
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import pandas as pd
import yfinance as yf
from dataclasses import dataclass, field
from typing import List, Optional
import sys


def safe_print(*args, **kwargs):
    """print() wrapper that ignores characters the console encoding can't handle"""
    try:
        print(*args, **kwargs)
    except UnicodeEncodeError:
        # strip out unencodable characters
        enc = sys.stdout.encoding or "utf-8"
        filtered = []
        for a in args:
            s = str(a)
            try:
                s.encode(enc)
                filtered.append(s)
            except UnicodeEncodeError:
                filtered.append(s.encode(enc, errors="ignore").decode(enc))
        print(*filtered, **kwargs)


# ========================================================================
#  データクラス
# ========================================================================

@dataclass
class IndicatorResult:
    """個別指標の分析結果"""
    name: str           # 指標名
    score: float        # -2 (非常に悪い) ～ +2 (非常に良い)
    value: float        # 指標の実数値
    comment: str        # 判定コメント


@dataclass
class MarketRegime:
    """マーケット地合い判定の総合結果"""
    level: str                          # STRONG_BUY / BUY / CAUTION / DANGER / CRISIS
    composite_score: float              # 総合スコア (-2 ～ +2)
    action: str                         # 推奨アクション（日本語）
    summary: str                        # 1行サマリー
    indicators: List[IndicatorResult] = field(default_factory=list)

    @property
    def should_buy(self) -> bool:
        """新規エントリー可能か"""
        return self.level in ("STRONG_BUY", "BUY")

    @property
    def should_reduce(self) -> bool:
        """ポジション縮小すべきか"""
        return self.level in ("CAUTION", "DANGER", "CRISIS")

    @property
    def should_exit_all(self) -> bool:
        """全ポジション決済すべきか"""
        return self.level == "CRISIS"


# ========================================================================
#  閾値定義（config.py からインポートする前にデフォルトを用意）
# ========================================================================

def _load_config_value(name, default):
    """config から動的に読む。未定義ならデフォルト値を返す"""
    try:
        from config import __dict__ as cfg
        return cfg.get(name, default)
    except Exception:
        return default


# ========================================================================
#  データ取得
# ========================================================================

def _fetch_index(ticker: str, period: str = "6mo") -> Optional[pd.DataFrame]:
    """yfinance で指数データを取得"""
    try:
        df = yf.download(ticker, period=period, interval="1d",
                         auto_adjust=True, progress=False)
        if df.empty:
            return None
        # MultiIndex 対応
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        return df
    except Exception as e:
        print(f"  警告: {ticker} データ取得失敗: {e}")
        return None


# ========================================================================
#  個別指標の計算
# ========================================================================

def _calc_trend_score(df: pd.DataFrame) -> IndicatorResult:
    """
    トレンド判定: 終値 vs MA25, MA75, MA200
    全MA上 → +2, MA25上のみ → +1, 全MA下 → -2 など
    """
    close = df["Close"]
    c = float(close.iloc[-1])

    ma25 = float(close.rolling(25).mean().iloc[-1])
    ma75 = float(close.rolling(75).mean().iloc[-1])

    # MA200 が取れなければ MA75 で代用
    if len(close) >= 200:
        ma200 = float(close.rolling(200).mean().iloc[-1])
    else:
        ma200 = ma75

    above_ma25 = c > ma25
    above_ma75 = c > ma75
    above_ma200 = c > ma200

    count_above = sum([above_ma25, above_ma75, above_ma200])

    # MA25 の傾き（直近5日 vs 10日前）で加点/減点
    ma25_series = close.rolling(25).mean().dropna()
    if len(ma25_series) >= 10:
        ma25_slope = float(ma25_series.iloc[-1] - ma25_series.iloc[-10])
    else:
        ma25_slope = 0.0

    if count_above == 3:
        score = 2.0 if ma25_slope > 0 else 1.5
        comment = "全移動平均線の上 → 強い上昇トレンド"
    elif count_above == 2:
        score = 1.0
        comment = "移動平均2本の上 → 上昇基調"
    elif count_above == 1:
        score = -0.5
        comment = "移動平均1本の上のみ → トレンド弱い"
    else:
        score = -2.0 if ma25_slope < 0 else -1.5
        comment = "全移動平均線の下 → 下降トレンド"

    return IndicatorResult(
        name="トレンド（日経平均 vs MA）",
        score=score,
        value=c,
        comment=comment,
    )


def _calc_momentum_score(df: pd.DataFrame) -> IndicatorResult:
    """
    モメンタム: 20日騰落率
    """
    close = df["Close"]
    roc_20 = float((close.iloc[-1] / close.iloc[-20] - 1) * 100) if len(close) >= 21 else 0.0

    if roc_20 > 5:
        score = 2.0
        comment = f"20日騰落率 {roc_20:+.1f}% → 強い上昇モメンタム"
    elif roc_20 > 2:
        score = 1.0
        comment = f"20日騰落率 {roc_20:+.1f}% → 上昇モメンタム"
    elif roc_20 > -2:
        score = 0.0
        comment = f"20日騰落率 {roc_20:+.1f}% → 中立"
    elif roc_20 > -5:
        score = -1.0
        comment = f"20日騰落率 {roc_20:+.1f}% → 下落モメンタム"
    else:
        score = -2.0
        comment = f"20日騰落率 {roc_20:+.1f}% → 強い下落モメンタム"

    return IndicatorResult(name="モメンタム（20日騰落率）", score=score,
                           value=roc_20, comment=comment)


def _calc_volatility_score(df: pd.DataFrame) -> IndicatorResult:
    """
    ボラティリティ: 20日ヒストリカルボラティリティ（年率換算）
    低ボラ → 安定、高ボラ → 不安定
    """
    close = df["Close"]
    returns = close.pct_change().dropna()
    if len(returns) < 20:
        return IndicatorResult(name="ボラティリティ", score=0.0,
                               value=0.0, comment="データ不足")

    hv_20 = float(returns.tail(20).std() * np.sqrt(252) * 100)  # 年率%

    if hv_20 < 15:
        score = 1.5
        comment = f"HV20={hv_20:.1f}% → 低ボラティリティ（安定相場）"
    elif hv_20 < 20:
        score = 0.5
        comment = f"HV20={hv_20:.1f}% → 通常のボラティリティ"
    elif hv_20 < 30:
        score = -1.0
        comment = f"HV20={hv_20:.1f}% → 高ボラティリティ（不安定）"
    elif hv_20 < 40:
        score = -1.5
        comment = f"HV20={hv_20:.1f}% → 非常に高いボラティリティ"
    else:
        score = -2.0
        comment = f"HV20={hv_20:.1f}% → 危機的ボラティリティ"

    return IndicatorResult(name="ボラティリティ（HV20）", score=score,
                           value=hv_20, comment=comment)


def _calc_rsi_score(df: pd.DataFrame) -> IndicatorResult:
    """
    RSI(14): 過熱/売られすぎ判定
    """
    close = df["Close"]
    delta = close.diff()
    up = delta.clip(lower=0)
    down = -delta.clip(upper=0)
    ma_up = up.rolling(14).mean()
    ma_down = down.rolling(14).mean()
    rs = ma_up / (ma_down + 1e-9)
    rsi = 100 - (100 / (1 + rs))
    rsi_val = float(rsi.iloc[-1])

    if rsi_val > 70:
        score = -1.0
        comment = f"RSI={rsi_val:.1f} → 買われすぎ（調整リスク）"
    elif rsi_val > 60:
        score = 0.5
        comment = f"RSI={rsi_val:.1f} → やや強い"
    elif rsi_val > 40:
        score = 0.5
        comment = f"RSI={rsi_val:.1f} → 中立圏"
    elif rsi_val > 30:
        score = -0.5
        comment = f"RSI={rsi_val:.1f} → やや弱い"
    else:
        score = -1.5
        comment = f"RSI={rsi_val:.1f} → 売られすぎ（パニック懸念）"

    return IndicatorResult(name="RSI(14)", score=score,
                           value=rsi_val, comment=comment)


def _calc_macd_score(df: pd.DataFrame) -> IndicatorResult:
    """
    MACD ヒストグラムのトレンド
    """
    close = df["Close"]
    ema12 = close.ewm(span=12, adjust=False).mean()
    ema26 = close.ewm(span=26, adjust=False).mean()
    macd = ema12 - ema26
    signal = macd.ewm(span=9, adjust=False).mean()
    hist = macd - signal

    h_now = float(hist.iloc[-1])
    h_prev = float(hist.iloc[-2]) if len(hist) >= 2 else h_now

    if h_now > 0 and h_now > h_prev:
        score = 1.5
        comment = "MACD ヒスト正＆拡大 → 上昇加速"
    elif h_now > 0:
        score = 0.5
        comment = "MACD ヒスト正だが縮小 → 上昇減速"
    elif h_now < 0 and h_now < h_prev:
        score = -1.5
        comment = "MACD ヒスト負＆拡大 → 下落加速"
    else:
        score = -0.5
        comment = "MACD ヒスト負だが縮小 → 下落減速"

    return IndicatorResult(name="MACD ヒストグラム", score=score,
                           value=h_now, comment=comment)


def _calc_drawdown_score(df: pd.DataFrame) -> IndicatorResult:
    """
    直近高値からのドローダウン
    """
    close = df["Close"]
    rolling_max = close.rolling(window=min(len(close), 126), min_periods=1).max()  # 約半年
    dd = float((close.iloc[-1] / rolling_max.iloc[-1] - 1) * 100)

    if dd > -3:
        score = 1.5
        comment = f"高値から {dd:+.1f}% → 高値圏維持"
    elif dd > -5:
        score = 0.5
        comment = f"高値から {dd:+.1f}% → 軽微な調整"
    elif dd > -10:
        score = -0.5
        comment = f"高値から {dd:+.1f}% → 調整局面"
    elif dd > -15:
        score = -1.5
        comment = f"高値から {dd:+.1f}% → 本格的な下落"
    else:
        score = -2.0
        comment = f"高値から {dd:+.1f}% → 暴落水準"

    return IndicatorResult(name="ドローダウン（半年高値比）", score=score,
                           value=dd, comment=comment)


def _calc_usdjpy_score() -> Optional[IndicatorResult]:
    """
    ドル円: 急激な円高はリスクオフの兆候
    """
    df = _fetch_index("JPY=X", period="3mo")
    if df is None or len(df) < 21:
        return None

    close = df["Close"]
    c = float(close.iloc[-1])
    c_20 = float(close.iloc[-20])
    change_pct = (c / c_20 - 1) * 100  # 負 = 円高

    if change_pct < -5:
        score = -2.0
        comment = f"USD/JPY 20日変化 {change_pct:+.1f}% → 急激な円高（リスクオフ）"
    elif change_pct < -2:
        score = -1.0
        comment = f"USD/JPY 20日変化 {change_pct:+.1f}% → 円高傾向（注意）"
    elif change_pct < 2:
        score = 0.0
        comment = f"USD/JPY 20日変化 {change_pct:+.1f}% → 安定"
    else:
        score = 0.5
        comment = f"USD/JPY 20日変化 {change_pct:+.1f}% → 円安（株価サポート）"

    return IndicatorResult(name="ドル円（リスクオフ指標）", score=score,
                           value=c, comment=comment)


def _calc_vix_score() -> Optional[IndicatorResult]:
    """
    VIX (恐怖指数): 世界的なリスクセンチメント
    日本版 VIX (^JNIV) が取れなければ米国 VIX (^VIX) を使用
    """
    # まず日経VI を試す（複数ティッカー候補）
    df = None
    label = "日経VI"
    for jvi_ticker in ["^JNIV", "^JNV", "1552.T"]:
        df = _fetch_index(jvi_ticker, period="3mo")
        if df is not None and not df.empty:
            break
        df = None

    if df is None or df.empty:
        df = _fetch_index("^VIX", period="3mo")
        label = "VIX (米国)"

    if df is None or df.empty:
        return None

    vix_val = float(df["Close"].iloc[-1])

    if vix_val < 15:
        score = 1.5
        comment = f"{label}={vix_val:.1f} → 低恐怖（楽観相場）"
    elif vix_val < 20:
        score = 0.5
        comment = f"{label}={vix_val:.1f} → 通常水準"
    elif vix_val < 30:
        score = -1.0
        comment = f"{label}={vix_val:.1f} → やや高い恐怖"
    elif vix_val < 40:
        score = -1.5
        comment = f"{label}={vix_val:.1f} → 高恐怖（市場不安定）"
    else:
        score = -2.0
        comment = f"{label}={vix_val:.1f} → 極度の恐怖（パニック水準）"

    return IndicatorResult(name=f"{label}（恐怖指数）", score=score,
                           value=vix_val, comment=comment)


# ========================================================================
#  総合判定
# ========================================================================

# 各指標の重み
_DEFAULT_WEIGHTS = {
    "トレンド（日経平均 vs MA）": 2.0,     # トレンドは最重要
    "モメンタム（20日騰落率）": 1.5,
    "ボラティリティ（HV20）": 1.5,
    "RSI(14)": 1.0,
    "MACD ヒストグラム": 1.0,
    "ドローダウン（半年高値比）": 2.0,     # ドローダウンも最重要
    "ドル円（リスクオフ指標）": 1.0,
}

# VIX 系指標の重みはラベルが動的なため特別処理
_VIX_WEIGHT = 1.5

# レベル判定の閾値
_LEVEL_THRESHOLDS = [
    (1.0, "STRONG_BUY"),
    (0.3, "BUY"),
    (-0.3, "CAUTION"),
    (-1.0, "DANGER"),
    (float("-inf"), "CRISIS"),
]

# レベルごとの推奨アクション
_ACTIONS = {
    "STRONG_BUY": "🟢🟢 地合い良好 → 積極的に買いエントリー可",
    "BUY":        "🟢 通常の環境 → 通常通り買いエントリー可",
    "CAUTION":    "🟡 注意 → 新規買い控えめ／ポジション縮小推奨",
    "DANGER":     "🔴 危険 → 新規買い停止／保有銘柄の利確推奨",
    "CRISIS":     "🔴🔴 暴落懸念 → 全ポジション決済（損切り含む）推奨",
}


def assess_market_regime(nikkei_period: str = "6mo") -> MarketRegime:
    """
    マーケット全体の地合いを総合判定する。

    Parameters
    ----------
    nikkei_period : str
        日経平均のデータ取得期間 (default: "6mo")

    Returns
    -------
    MarketRegime
        判定結果
    """
    safe_print("\n" + "=" * 50)
    safe_print("[マーケット地合い判定]")
    safe_print("=" * 50)

    indicators: List[IndicatorResult] = []

    # --- 日経平均データ取得 ---
    safe_print("  日経平均データ取得中...")
    nikkei_df = _fetch_index("^N225", period=nikkei_period)

    if nikkei_df is not None and len(nikkei_df) >= 30:
        safe_print(f"  日経平均: {float(nikkei_df['Close'].iloc[-1]):,.0f}円 "
              f"({len(nikkei_df)}日分)")

        # 日経平均ベースの指標
        indicators.append(_calc_trend_score(nikkei_df))
        indicators.append(_calc_momentum_score(nikkei_df))
        indicators.append(_calc_volatility_score(nikkei_df))
        indicators.append(_calc_rsi_score(nikkei_df))
        indicators.append(_calc_macd_score(nikkei_df))
        indicators.append(_calc_drawdown_score(nikkei_df))
    else:
        safe_print("  警告: 日経平均データ取得失敗 → 利用可能な指標で判定")

    # --- ドル円 ---
    safe_print("  ドル円データ取得中...")
    usdjpy = _calc_usdjpy_score()
    if usdjpy is not None:
        indicators.append(usdjpy)

    # --- VIX ---
    safe_print("  VIX データ取得中...")
    vix = _calc_vix_score()
    if vix is not None:
        indicators.append(vix)

    # --- 総合スコア計算（重み付き平均）---
    if not indicators:
        return MarketRegime(
            level="BUY",
            composite_score=0.0,
            action=_ACTIONS["BUY"],
            summary="指標データ取得不可 → デフォルトで通常判定",
            indicators=[],
        )

    weighted_sum = 0.0
    total_weight = 0.0
    for ind in indicators:
        # VIX は名前が動的なので特別処理
        if "恐怖指数" in ind.name:
            w = _VIX_WEIGHT
        else:
            w = _DEFAULT_WEIGHTS.get(ind.name, 1.0)
        weighted_sum += ind.score * w
        total_weight += w

    composite = weighted_sum / total_weight if total_weight > 0 else 0.0

    # --- レベル判定 ---
    level = "CRISIS"  # fallback
    for threshold, lv in _LEVEL_THRESHOLDS:
        if composite >= threshold:
            level = lv
            break

    action = _ACTIONS[level]

    # --- サマリー生成 ---
    summary = (
        f"総合スコア: {composite:+.2f} → {level}\n"
        f"  {action}"
    )

    # --- 結果表示 ---
    safe_print(f"\n{'─' * 50}")
    safe_print(f"  【地合い判定結果】")
    for ind in indicators:
        emoji = "🟢" if ind.score > 0 else ("🔴" if ind.score < 0 else "⚪")
        safe_print(f"  {emoji} {ind.name}: {ind.comment} (スコア: {ind.score:+.1f})")
    safe_print(f"{'─' * 50}")
    safe_print(f"  ▶ {summary}")
    safe_print(f"{'─' * 50}\n")

    return MarketRegime(
        level=level,
        composite_score=composite,
        action=action,
        summary=summary,
        indicators=indicators,
    )


def format_regime_summary(regime: MarketRegime) -> str:
    """MarketRegime を LINE 通知用の簡潔サマリーに整形（4行のみ）"""
    level_emoji = {
        "STRONG_BUY": "🟢🟢",
        "BUY":        "🟢",
        "CAUTION":    "🟡",
        "DANGER":     "🔴",
        "CRISIS":     "🔴🔴",
    }
    emoji = level_emoji.get(regime.level, "❓")
    return (
        f"📊 マーケット地合い判定\n\n"
        f"{emoji} 地合い: {regime.level}\n"
        f"総合スコア: {regime.composite_score:+.2f}\n"
        f"→ {regime.action}"
    )


def format_regime_message(regime: MarketRegime) -> str:
    """MarketRegime を詳細テキストに整形（ファイル保存用）"""
    lines = ["📊 マーケット地合い判定\n"]

    level_emoji = {
        "STRONG_BUY": "🟢🟢",
        "BUY":        "🟢",
        "CAUTION":    "🟡",
        "DANGER":     "🔴",
        "CRISIS":     "🔴🔴",
    }
    emoji = level_emoji.get(regime.level, "❓")

    lines.append(f"{emoji} 地合い: {regime.level}")
    lines.append(f"総合スコア: {regime.composite_score:+.2f}")
    lines.append(f"→ {regime.action}\n")

    lines.append("【各指標の詳細】")
    for ind in regime.indicators:
        ind_emoji = "🟢" if ind.score > 0 else ("🔴" if ind.score < 0 else "⚪")
        lines.append(f"{ind_emoji} {ind.name}")
        lines.append(f"  {ind.comment}")

    return "\n".join(lines)


# ========================================================================
#  テスト実行
# ========================================================================

if __name__ == "__main__":
    regime = assess_market_regime()
    safe_print("\n=== LINE 通知メッセージプレビュー ===")
    safe_print(format_regime_message(regime))
    safe_print(f"\n新規買い可: {regime.should_buy}")
    safe_print(f"ポジション縮小推奨: {regime.should_reduce}")
    safe_print(f"全決済推奨: {regime.should_exit_all}")
