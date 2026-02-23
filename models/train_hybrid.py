"""
ハイブリッドモデル学習
  - グローバルモデル: 全銘柄結合データで学習
  - ローカルモデル: 銘柄ごとに個別学習（225ループ）
PCリソースに配慮し、メモリ解放・スリープ・コア数制限を実装
"""
import sys
import os
import gc
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import lightgbm as lgb
import joblib
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, roc_auc_score

from config import (
    FEATURE_COLS,
    GLOBAL_MODEL_PATH,
    LOCAL_MODEL_DIR,
    LOCAL_MIN_SAMPLES,
    TRAIN_SLEEP_SEC,
    LGB_N_JOBS,
)


def _get_lgbm_params():
    """LightGBM 共通ハイパーパラメータ"""
    return dict(
        n_estimators=300,
        learning_rate=0.05,
        max_depth=6,
        num_leaves=31,
        min_child_samples=20,
        random_state=42,
        verbose=-1,
        n_jobs=LGB_N_JOBS,          # コア数を制限
        force_col_wise=True,         # メモリ効率向上
    )


def _evaluate(model, X_val, y_val):
    """Accuracy / AUC を返す"""
    y_pred = model.predict(X_val)
    y_proba = model.predict_proba(X_val)[:, 1]
    acc = accuracy_score(y_val, y_pred)
    try:
        auc = roc_auc_score(y_val, y_proba)
    except ValueError:
        auc = float("nan")
    return acc, auc


# ------------------------------------------------------------------ #
#  グローバルモデル学習
# ------------------------------------------------------------------ #
def train_global_model(dataset, model_path=None):
    """
    全銘柄結合データでグローバルモデルを学習・保存する。

    Parameters
    ----------
    dataset : pd.DataFrame
        make_dataset() の戻り値（FEATURE_COLS + target + code を含む）
    model_path : str, optional
        保存先パス。省略時は config.GLOBAL_MODEL_PATH

    Returns
    -------
    model : LGBMClassifier
    feature_cols : list[str]
    """
    if model_path is None:
        model_path = GLOBAL_MODEL_PATH

    feature_cols = [c for c in FEATURE_COLS if c in dataset.columns]
    X = dataset[feature_cols]
    y = dataset["target"]

    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, shuffle=False
    )

    model = lgb.LGBMClassifier(**_get_lgbm_params())
    model.fit(X_train, y_train, eval_set=[(X_val, y_val)])

    acc, auc = _evaluate(model, X_val, y_val)
    print(f"[Global] Accuracy: {acc:.4f}, AUC: {auc:.4f}")

    # 特徴量重要度 Top10
    imp = sorted(
        zip(feature_cols, model.feature_importances_),
        key=lambda x: x[1], reverse=True,
    )
    print("[Global] 特徴量重要度 Top10:")
    for name, score in imp[:10]:
        print(f"  {name}: {score}")

    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    joblib.dump({"model": model, "feature_cols": feature_cols}, model_path)
    print(f"グローバルモデル保存: {model_path}")

    return model, feature_cols


# ------------------------------------------------------------------ #
#  ローカル（個別銘柄）モデル学習 — 225 銘柄ループ
# ------------------------------------------------------------------ #
def train_local_models(dataset, local_dir=None, min_samples=None, sleep_sec=None):
    """
    銘柄ごとにローカルモデルを学習し、local_dir 以下に保存する。
    データが少ない銘柄はスキップされる。

    Parameters
    ----------
    dataset : pd.DataFrame
        make_dataset() の戻り値
    local_dir : str, optional
    min_samples : int, optional
    sleep_sec : float, optional

    Returns
    -------
    trained_tickers : list[str]  学習できた銘柄一覧
    skipped_tickers : list[str]  スキップされた銘柄一覧
    """
    if local_dir is None:
        local_dir = LOCAL_MODEL_DIR
    if min_samples is None:
        min_samples = LOCAL_MIN_SAMPLES
    if sleep_sec is None:
        sleep_sec = TRAIN_SLEEP_SEC

    os.makedirs(local_dir, exist_ok=True)

    feature_cols = [c for c in FEATURE_COLS if c in dataset.columns]
    tickers = sorted(dataset["code"].unique())
    total = len(tickers)

    trained_tickers = []
    skipped_tickers = []

    print(f"\n{'='*50}")
    print(f"ローカルモデル学習開始: {total} 銘柄")
    print(f"最低サンプル数: {min_samples}, スリープ: {sleep_sec}s, コア数: {LGB_N_JOBS}")
    print(f"{'='*50}")

    for i, ticker in enumerate(tickers, 1):
        ticker_data = dataset[dataset["code"] == ticker]

        # --- サンプル数チェック ---
        if len(ticker_data) < min_samples:
            skipped_tickers.append(ticker)
            if i % 50 == 0 or i == total:
                print(f"  進捗: {i}/{total} (学習済: {len(trained_tickers)}, スキップ: {len(skipped_tickers)})")
            continue

        X = ticker_data[feature_cols]
        y = ticker_data["target"]

        # クラスが 1 種類しかない場合はスキップ
        if y.nunique() < 2:
            skipped_tickers.append(ticker)
            continue

        try:
            X_train, X_val, y_train, y_val = train_test_split(
                X, y, test_size=0.2, shuffle=False
            )

            # 検証データのクラスが 1 種類の場合でも学習自体は行う
            local_params = _get_lgbm_params()
            local_params["n_estimators"] = 150       # ローカルは軽めに
            local_params["min_child_samples"] = 10   # サンプル少ないので緩和

            model = lgb.LGBMClassifier(**local_params)
            model.fit(X_train, y_train, eval_set=[(X_val, y_val)])

            # 保存
            safe_ticker = ticker.replace(".", "_")    # ファイル名安全化
            path = os.path.join(local_dir, f"local_model_{safe_ticker}.pkl")
            joblib.dump({"model": model, "feature_cols": feature_cols}, path)
            trained_tickers.append(ticker)

        except Exception as e:
            print(f"  [WARNING] {ticker} の学習に失敗: {e}")
            skipped_tickers.append(ticker)

        # --- リソース解放 ---
        del ticker_data, X, y
        try:
            del X_train, X_val, y_train, y_val, model
        except NameError:
            pass
        gc.collect()

        # --- CPU 温度対策: スリープ ---
        if sleep_sec > 0:
            time.sleep(sleep_sec)

        # --- 進捗表示 ---
        if i % 25 == 0 or i == total:
            print(f"  進捗: {i}/{total} (学習済: {len(trained_tickers)}, スキップ: {len(skipped_tickers)})")

    print(f"\nローカルモデル学習完了: 成功 {len(trained_tickers)} / スキップ {len(skipped_tickers)}")
    return trained_tickers, skipped_tickers


# ------------------------------------------------------------------ #
#  ハイブリッド学習（グローバル + ローカル を一括実行）
# ------------------------------------------------------------------ #
def train_hybrid(dataset):
    """
    1. グローバルモデル学習
    2. ローカルモデル学習（225 銘柄ループ）
    """
    print("\n" + "=" * 60)
    print("  ハイブリッドモデル学習パイプライン")
    print("=" * 60)

    # --- グローバル ---
    print("\n[Phase 1] グローバルモデル学習")
    global_model, feature_cols = train_global_model(dataset)

    # グローバル学習後にメモリ解放
    del global_model
    gc.collect()
    time.sleep(1.0)

    # --- ローカル ---
    print("\n[Phase 2] ローカルモデル学習")
    trained, skipped = train_local_models(dataset)

    print("\n" + "=" * 60)
    print("  ハイブリッド学習完了")
    print(f"  グローバル: {GLOBAL_MODEL_PATH}")
    print(f"  ローカル: {len(trained)} 銘柄 → {LOCAL_MODEL_DIR}/")
    print("=" * 60)

    return trained, skipped


# ------------------------------------------------------------------ #
if __name__ == "__main__":
    # テスト用ダミーデータ
    import pandas as pd

    n_per_ticker = 200
    tickers_test = ["7203.T", "6758.T", "9984.T"]
    dfs = []
    for t in tickers_test:
        df = pd.DataFrame({c: np.random.rand(n_per_ticker) for c in FEATURE_COLS})
        df["target"] = np.random.randint(0, 2, n_per_ticker)
        df["code"] = t
        dfs.append(df)
    dataset = pd.concat(dfs, ignore_index=True)

    train_hybrid(dataset)
