"""
LightGBM による分類モデルの学習・保存
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import lightgbm as lgb
import joblib
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, roc_auc_score
from config import FEATURE_COLS, MODEL_PATH

def train_model(dataset, model_path=None):
    """
    dataset: make_dataset の戻り値 (特徴量 + target + code)
    """
    if model_path is None:
        model_path = MODEL_PATH

    # 特徴量カラム
    feature_cols = [c for c in FEATURE_COLS if c in dataset.columns]
    X = dataset[feature_cols]
    y = dataset["target"]

    # 学習 / 検証分割
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, shuffle=False   # 時系列なので shuffle=False
    )

    model = lgb.LGBMClassifier(
        n_estimators=300,
        learning_rate=0.05,
        max_depth=6,
        num_leaves=31,
        min_child_samples=20,
        random_state=42,
        verbose=-1,
    )
    model.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
    )

    # 評価
    y_pred = model.predict(X_val)
    y_proba = model.predict_proba(X_val)[:, 1]
    acc = accuracy_score(y_val, y_pred)
    try:
        auc = roc_auc_score(y_val, y_proba)
    except ValueError:
        auc = float("nan")
    print(f"検証 Accuracy: {acc:.4f}, AUC: {auc:.4f}")

    # 特徴量重要度 Top10
    imp = sorted(zip(feature_cols, model.feature_importances_),
                 key=lambda x: x[1], reverse=True)
    print("特徴量重要度 Top10:")
    for name, score in imp[:10]:
        print(f"  {name}: {score}")

    # 保存
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    joblib.dump({"model": model, "feature_cols": feature_cols}, model_path)
    print(f"モデル保存: {model_path}")
    return model

if __name__ == "__main__":
    import pandas as pd, numpy as np
    n = 500
    df = pd.DataFrame({c: np.random.rand(n) for c in FEATURE_COLS})
    df["target"] = np.random.randint(0, 2, n)
    df["code"] = "TEST.T"
    train_model(df)
