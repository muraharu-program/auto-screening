"""
日本株スイングトレード AI スクリーニング — メインパイプライン

使い方:
  python main.py              … 学習 → スクリーニング → 通知（フル実行・従来モード）
  python main.py --train      … モデル学習のみ（従来の汎用モデル）
  python main.py --screen     … スクリーニング＋通知のみ（従来の汎用モデル）
  python main.py --hybrid-train … ハイブリッド学習（グローバル + ローカル225銘柄）
  python main.py --hybrid       … ハイブリッドスクリーニング＋通知
  python main.py --full-hybrid  … ハイブリッド学習 → スクリーニング → 通知
  python main.py --hybrid --sentiment  … ハイブリッド＋ニュースセンチメント分析
  python main.py --full-hybrid --sentiment … フル学習→ハイブリッド＋センチメント
"""
import sys
import os
import argparse

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import TRAIN_PERIOD, SCREEN_PERIOD, INTERVAL, MODEL_PATH
from data.ingest_yfinance import fetch_stock_data
from features.make_features import make_features
from features.make_dataset import make_dataset
from models.train_model import train_model
from screening.screen import screen_stocks
from notify.line_notify import send_line_message


def run_train():
    """過去データで学習してモデルを保存（従来の汎用モデル）"""
    print("=" * 50)
    print("[1/4] データ取得（学習用）...")
    stock_data = fetch_stock_data(period=TRAIN_PERIOD, interval=INTERVAL)

    print("[2/4] 特徴量生成...")
    features = make_features(stock_data)

    print("[3/4] 学習データセット作成...")
    dataset = make_dataset(features)

    print("[4/4] モデル学習...")
    train_model(dataset, model_path=MODEL_PATH)
    print("学習完了 ✓")


def run_screen():
    """学習済みモデルでスクリーニング → LINE 通知（従来モード）"""
    print("=" * 50)
    print("[スクリーニング]")
    candidates = screen_stocks(model_path=MODEL_PATH)

    if candidates.empty:
        print("有望銘柄はありませんでした。")
    else:
        print(candidates.to_string(index=False))

    print("[通知]")
    send_line_message(candidates)
    print("完了 ✓")


def run_hybrid_train():
    """ハイブリッドモデル学習（グローバル + ローカル225銘柄）"""
    from models.train_hybrid import train_hybrid
    import gc

    print("=" * 50)
    print("[1/3] データ取得（ハイブリッド学習用）...")
    stock_data = fetch_stock_data(period=TRAIN_PERIOD, interval=INTERVAL)

    print("[2/3] 特徴量生成...")
    features = make_features(stock_data)

    print("[3/3] 学習データセット作成...")
    dataset = make_dataset(features)

    # メモリ解放
    del stock_data, features
    gc.collect()

    print("[ハイブリッド学習開始]")
    train_hybrid(dataset)


def run_hybrid_screen():
    """ハイブリッドスクリーニング → LINE 通知"""
    from screening.screen_hybrid import screen_hybrid

    print("=" * 50)
    print("[ハイブリッドスクリーニング]")
    candidates = screen_hybrid()

    if candidates.empty:
        print("有望銘柄はありませんでした。")
    else:
        print(candidates.to_string(index=False))

    print("[通知]")
    send_line_message(candidates)
    print("完了 ✓")


def run_hybrid_screen_with_sentiment():
    """ハイブリッドスクリーニング + ニュースセンチメントフィルター → LINE 通知"""
    from screening.screen_hybrid import screen_hybrid

    print("=" * 50)
    print("[ハイブリッドスクリーニング + センチメント分析]")
    candidates = screen_hybrid(use_sentiment=True)

    if candidates.empty:
        print("有望銘柄はありませんでした。（センチメントフィルター適用後）")
    else:
        print(candidates.to_string(index=False))

    print("[通知]")
    send_line_message(candidates)
    print("完了 ✓")


def main():
    parser = argparse.ArgumentParser(description="AI スクリーニングシステム")
    parser.add_argument("--train",        action="store_true", help="モデル学習のみ（従来）")
    parser.add_argument("--screen",       action="store_true", help="スクリーニング＋通知（従来）")
    parser.add_argument("--hybrid-train", action="store_true", help="ハイブリッド学習（Global+Local）")
    parser.add_argument("--hybrid",       action="store_true", help="ハイブリッドスクリーニング＋通知")
    parser.add_argument("--full-hybrid",  action="store_true", help="ハイブリッド学習→スクリーニング→通知")
    parser.add_argument("--sentiment",    action="store_true", help="ニュースセンチメント分析フィルターを追加（--hybrid/--full-hybrid と併用）")
    args = parser.parse_args()

    if args.hybrid_train:
        run_hybrid_train()
    elif args.hybrid:
        if args.sentiment:
            run_hybrid_screen_with_sentiment()
        else:
            run_hybrid_screen()
    elif args.full_hybrid:
        run_hybrid_train()
        if args.sentiment:
            run_hybrid_screen_with_sentiment()
        else:
            run_hybrid_screen()
    elif args.train:
        run_train()
    elif args.screen:
        run_screen()
    else:
        # デフォルト: 従来の学習 → スクリーニング
        run_train()
        run_screen()


if __name__ == "__main__":
    main()
