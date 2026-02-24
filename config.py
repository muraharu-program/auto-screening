"""
スクリーニングシステム共通設定
"""
from dotenv import load_dotenv
import os

load_dotenv()

# ===== パス設定 =====
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
TICKER_CSV = os.path.join(BASE_DIR, "prime.csv")
MODEL_PATH = os.path.join(BASE_DIR, "models", "model.pkl")
DATA_DIR = os.path.join(BASE_DIR, "data")
# 出力先（通知テキスト等）
OUTPUT_DIR = os.path.join(BASE_DIR, "outputs")

# ===== ハイブリッドモデル設定 =====
GLOBAL_MODEL_PATH = os.path.join(BASE_DIR, "models", "global_model.pkl")
LOCAL_MODEL_DIR = os.path.join(BASE_DIR, "models", "local")
LOCAL_MIN_SAMPLES = 100       # 個別モデル学習に必要な最低サンプル数
GLOBAL_WEIGHT = 0.6           # ハイブリッド合成時のグローバルモデル重み
LOCAL_WEIGHT = 0.4            # ハイブリッド合成時のローカルモデル重み
TRAIN_SLEEP_SEC = 0.5         # 個別モデル学習ループ間のスリープ（秒）
LGB_N_JOBS = 2                # LightGBM の並列コア数制限

# ===== データ取得設定 =====
TRAIN_PERIOD = "2y"        # 学習用データ取得期間
SCREEN_PERIOD = "3mo"      # スクリーニング用データ取得期間
INTERVAL = "1d"            # 日足

# ===== 売買代金フィルタリング設定 =====
TURNOVER_TOP_N = 500           # 売買代金上位N銘柄を選別
PREFILTER_PERIOD = "1mo"       # 売買代金計算用の取得期間（直近1ヶ月）

# ===== 特徴量設定 =====
FEATURE_COLS = [
    "Close", "Volume",
    "ma5", "ma25", "ma75",
    "ma5_ratio", "ma25_ratio", "ma75_ratio",
    "rsi14",
    "macd", "macd_signal", "macd_hist",
    "bb_upper", "bb_lower", "bb_width",
    "volume_ma5_ratio",
    "return_1d", "return_5d", "return_10d",
    "volatility_20d",
]

# ===== ラベル設定 =====
UP_DAYS = 5          # 何営業日後の変化を予測するか
UP_RATE = 0.05       # 上昇率の閾値（5%）

# ===== フィルタリング設定 =====
MIN_VOLUME = 500000  # 最低出来高（直近）
TOP_N = 6           # 通知する上位銘柄数
MIN_PROB = 0.5       # 最低上昇確率

# ===== センチメント分析設定 =====
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY", "")
GEMINI_MODEL = "gemini-2.5-flash-lite"       # 使用する Gemini モデル名
SENTIMENT_TOP_N = 30                         # センチメント分析対象の上位銘柄数
SENTIMENT_MIN_SCORE = 4                      # 最終通知に残す最低センチメントスコア（1-5）
SENTIMENT_DEFAULT_SCORE = 3                  # ニュース取得失敗時のデフォルトスコア
SENTIMENT_API_SLEEP = 2.0                    # Gemini API 呼び出し間のスリープ（秒）
NEWS_LOOKBACK_DAYS = 7                       # ニュース取得の遡り日数

# ===== LINE Messaging API =====
LINE_CHANNEL_ACCESS_TOKEN = os.environ.get("LINE_CHANNEL_ACCESS_TOKEN", "")
LINE_USER_ID = os.environ.get("LINE_USER_ID", "")
