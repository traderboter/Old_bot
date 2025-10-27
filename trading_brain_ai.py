"""
TradingBrainAI: Advanced Machine Learning Trading Enhancement

This system extends the existing signal generator with powerful ML capabilities,
enabling adaptive learning, pattern recognition, and predictive analytics for
improved trading decisions.

Key features:
1. Multiple neural network models (LSTM, Transformer, CNN) for diverse market insights
2. Online learning for real-time adaptation to market conditions
3. Feature engineering pipeline with technical indicator optimization
4. Reinforcement learning for position sizing and risk management
5. Ensemble predictions with confidence scoring
6. Automated hyperparameter optimization
7. Market regime detection using unsupervised learning
"""
import traceback
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Sequential, Model, load_model
from tensorflow.keras.layers import (LSTM, Dense, Dropout, BatchNormalization,
                                     Input, Conv1D, MaxPooling1D, Flatten,
                                     Bidirectional, TimeDistributed, Attention,
                                     MultiHeadAttention, LayerNormalization, GlobalAveragePooling1D)
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.regularizers import l1_l2
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
from sklearn.model_selection import TimeSeriesSplit
from sklearn.ensemble import RandomForestClassifier, GradientBoostingRegressor
from sklearn.metrics import precision_score, recall_score, accuracy_score, mean_squared_error, mean_absolute_error
import xgboost as xgb
import lightgbm as lgb
import optuna
import joblib
import os
import json
import logging
import datetime
from typing import Dict, List, Any, Optional, Tuple, Union, Set, TypeVar, cast, Callable
from dataclasses import dataclass, field, asdict
from collections import deque
import talib
import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import adfuller
import gym
from gym import spaces
import torch
import torch.nn as nn
import torch.optim as optim
from stable_baselines3 import PPO, A2C, SAC
from stable_baselines3.common.vec_env import DummyVecEnv
import warnings
import copy
import time
from datetime import datetime, timedelta

warnings.filterwarnings('ignore')

# جایگزینی کامل کد پیکربندی لاگر

# Configure logger
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

# حذف تمام هندلرهای موجود
for handler in logger.handlers[:]:
    logger.removeHandler(handler)

# غیرفعال کردن انتشار به لاگر والد
logger.propagate = False

# افزودن یک هندلر جدید
handler = logging.StreamHandler()
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
handler.setFormatter(formatter)
logger.addHandler(handler)

# Ensure TensorFlow uses GPU if available
physical_devices = tf.config.list_physical_devices('GPU')
if len(physical_devices) > 0:
    tf.config.experimental.set_memory_growth(physical_devices[0], True)
    logger.info(f"Using GPU: {physical_devices[0]}")
else:
    logger.info("No GPU available, using CPU")

try:
    import bottleneck as bn
    use_bottleneck_available = True
    logger.info("Bottleneck library found, using optimized rolling calculations.")
except ImportError:
    use_bottleneck_available = False
    logger.info("Bottleneck library not found, using standard pandas rolling calculations.")

# -----------------------------------------------------------------------------
# Configuration Management System
# -----------------------------------------------------------------------------

class ConfigManager:
    """
    مدیریت پیکربندی با قابلیت به‌روزرسانی پویا و انتشار تغییرات به اجزای وابسته
    """

    def __init__(self, initial_config: Dict[str, Any]):
        """مقداردهی اولیه با پیکربندی اولیه"""
        # نسخه‌ای برای استفاده داخلی
        self._config = copy.deepcopy(initial_config)
        # تاریخچه تغییرات
        self._change_history = []
        # شنوندگان تغییرات
        self._update_listeners = []
        # آخرین زمان به‌روزرسانی
        self._last_update_time = datetime.now()

        logger.info("سیستم مدیریت پیکربندی راه‌اندازی شد")

    def get_config(self) -> Dict[str, Any]:
        """دریافت کل پیکربندی فعلی"""
        return copy.deepcopy(self._config)

    def get(self, section: str, default: Any = None) -> Any:
        """
        دریافت بخش خاصی از پیکربندی با پشتیبانی از نقطه
        مثال: config_manager.get('trading_brain_ai.feature_engineering.lookback_window', 20)
        """
        if '.' in section:
            parts = section.split('.')
            value = self._config
            for part in parts:
                if isinstance(value, dict) and part in value:
                    value = value[part]
                else:
                    return default
            return value
        else:
            return self._config.get(section, default)

    def set(self, path: str, value: Any) -> bool:
        """
        تنظیم مقدار در یک مسیر مشخص با پشتیبانی از نقطه
        """
        if not path:
            return False

        # ثبت تغییر در تاریخچه
        self._change_history.append({
            'path': path,
            'old_value': self.get(path),
            'new_value': value,
            'time': datetime.now().isoformat()
        })

        # به‌روزرسانی مقدار
        if '.' in path:
            parts = path.split('.')
            target = self._config
            for i, part in enumerate(parts[:-1]):
                if part not in target:
                    target[part] = {}
                target = target[part]
            target[parts[-1]] = value
        else:
            self._config[path] = value

        self._last_update_time = datetime.now()

        # اطلاع‌رسانی به شنوندگان
        self._notify_listeners(path, value)

        return True

    def update_section(self, section: str, values: Dict[str, Any]) -> bool:
        """به‌روزرسانی یک بخش کامل از پیکربندی"""
        if not section or not isinstance(values, dict):
            return False

        # ثبت تغییر در تاریخچه
        self._change_history.append({
            'section': section,
            'old_values': copy.deepcopy(self.get(section, {})),
            'new_values': copy.deepcopy(values),
            'time': datetime.now().isoformat()
        })

        # اگر بخش شامل نقطه باشد، به صورت سلسله مراتبی به‌روزرسانی می‌کنیم
        if '.' in section:
            parts = section.split('.')
            target = self._config
            for i, part in enumerate(parts[:-1]):
                if part not in target:
                    target[part] = {}
                target = target[part]

            # به‌روزرسانی یا ایجاد بخش نهایی
            if parts[-1] not in target:
                target[parts[-1]] = values
            else:
                if isinstance(target[parts[-1]], dict):
                    target[parts[-1]].update(values)
                else:
                    target[parts[-1]] = values
        else:
            # به‌روزرسانی یا ایجاد بخش اصلی
            if section not in self._config:
                self._config[section] = values
            else:
                if isinstance(self._config[section], dict):
                    self._config[section].update(values)
                else:
                    self._config[section] = values

        self._last_update_time = datetime.now()

        # اطلاع‌رسانی به شنوندگان
        for key, value in values.items():
            path = f"{section}.{key}" if section else key
            self._notify_listeners(path, value)

        return True

    def register_listener(self, callback: Callable[[str, Any], None]) -> None:
        """ثبت یک شنونده برای دریافت تغییرات پیکربندی"""
        if callback not in self._update_listeners:
            self._update_listeners.append(callback)

    def unregister_listener(self, callback: Callable[[str, Any], None]) -> None:
        """حذف یک شنونده از لیست شنوندگان"""
        if callback in self._update_listeners:
            self._update_listeners.remove(callback)

    def _notify_listeners(self, path: str, value: Any) -> None:
        """آگاه‌سازی تمامی شنوندگان از تغییر"""
        for listener in self._update_listeners:
            try:
                listener(path, value)
            except Exception as e:
                logger.error(f"خطای شنونده در مسیر {path}: {e}")

    def save_to_file(self, filepath: str) -> bool:
        """ذخیره پیکربندی فعلی در یک فایل"""
        try:
            # ایجاد دایرکتوری در صورت نیاز
            os.makedirs(os.path.dirname(os.path.abspath(filepath)), exist_ok=True)

            with open(filepath, 'w', encoding='utf-8') as f:
                if filepath.endswith(('.yaml', '.yml')):
                    import yaml
                    yaml.dump(self._config, f, default_flow_style=False, allow_unicode=True)
                else:  # json
                    json.dump(self._config, f, ensure_ascii=False, indent=2)

            logger.info(f"پیکربندی با موفقیت در {filepath} ذخیره شد")
            return True

        except Exception as e:
            logger.error(f"خطا در ذخیره پیکربندی: {e}")
            return False

    def load_from_file(self, filepath: str) -> bool:
        """بارگذاری پیکربندی از یک فایل"""
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                if filepath.endswith(('.yaml', '.yml')):
                    import yaml
                    new_config = yaml.safe_load(f)
                else:  # json
                    new_config = json.load(f)

            # ثبت تاریخچه
            self._change_history.append({
                'action': 'load_from_file',
                'filepath': filepath,
                'time': datetime.now().isoformat()
            })

            # به‌روزرسانی پیکربندی
            self._config = new_config
            self._last_update_time = datetime.now()

            # آگاه‌سازی شنوندگان از تغییر کل پیکربندی
            self._notify_listeners('', new_config)

            logger.info(f"پیکربندی با موفقیت از {filepath} بارگذاری شد")
            return True

        except Exception as e:
            logger.error(f"خطا در بارگذاری پیکربندی: {e}")
            return False

    def get_change_history(self) -> List[Dict[str, Any]]:
        """دریافت تاریخچه تغییرات"""
        return copy.deepcopy(self._change_history)

# -----------------------------------------------------------------------------
# Feature Engineering and Data Preparation
# -----------------------------------------------------------------------------

class FeatureEngineer:
    """Advanced feature engineering for market data"""

    def __init__(self, config_manager: ConfigManager):
        """Initialize with configuration manager"""
        self.config_manager = config_manager
        self.config_manager.register_listener(self._handle_config_update)

        # دریافت پیکربندی اصلی
        self._load_config()

        # مقداردهی اولیه سایر متغیرهای کلاس
        self.scalers = {}
        self.selected_features = set()
        self.feature_importance = {}

        logger.info(
            f"FeatureEngineer initialized with {self.lookback_window} lookback window. Bottleneck enabled: {self.use_bottleneck}")

    def _load_config(self):
        """بارگذاری تنظیمات از مدیریت پیکربندی"""
        # دریافت بخش feature_engineering از تنظیمات
        feature_config = self.config_manager.get('trading_brain_ai.feature_engineering', {})

        # بارگذاری تنظیمات با مقادیر پیش‌فرض
        self.use_ta_features = feature_config.get("use_technical_indicators", True)
        self.use_price_patterns = feature_config.get("use_price_patterns", True)
        self.use_volatility_features = feature_config.get("use_volatility_features", True)
        self.use_volume_features = feature_config.get("use_volume_features", True)
        self.use_market_indicators = feature_config.get("use_market_indicators", True)
        self.lookback_window = feature_config.get("lookback_window", 20)
        self.target_horizons = feature_config.get("target_horizons", [1, 5])
        self.scaler_type = feature_config.get("scaler_type", "robust")
        self.feature_selection = feature_config.get("feature_selection", True)
        self.feature_selection_method = feature_config.get("feature_selection_method", "importance")
        self.importance_threshold = feature_config.get("importance_threshold", 0.01)
        self.use_bottleneck = use_bottleneck_available

    def _handle_config_update(self, path: str, value: Any) -> None:
        """پردازش به‌روزرسانی پیکربندی"""
        # بررسی اینکه آیا مسیر مربوط به تنظیمات این کلاس است
        if path.startswith('trading_brain_ai.feature_engineering.'):
            # استخراج نام پارامتر
            param = path.split('.')[-1]

            # به‌روزرسانی مقدار پارامتر
            if hasattr(self, param):
                old_value = getattr(self, param)
                setattr(self, param, value)
                logger.info(f"تنظیم {param} از {old_value} به {value} تغییر یافت")

        # اگر کل بخش به‌روزرسانی شده باشد
        elif path == 'trading_brain_ai.feature_engineering' and isinstance(value, dict):
            self._load_config()
            logger.info("تمام تنظیمات feature_engineering به‌روزرسانی شد")

    def create_features(self, df: pd.DataFrame) -> Optional[pd.DataFrame]:  # Return type Optional
        """Generate comprehensive feature set from raw OHLCV data - without final dropna"""
        try:
            logger.debug(f"Starting create_features with DataFrame shape: {df.shape if df is not None else 'None'}")
            # افزایش طول مورد نیاز برای اطمینان از کارکرد تمام اندیکاتورها
            min_required_length = max(self.lookback_window + 50, 100)  # حداقل 100 کندل یا lookback+50
            if df is None or len(df) < min_required_length:
                logger.error(
                    f"Insufficient data for feature creation. Need at least {min_required_length} rows, got {len(df) if df is not None else 0}")
                return None

            features_df = df.copy()

            required_columns = ['open', 'high', 'low', 'close', 'volume']
            missing_cols = [col for col in required_columns if col not in features_df.columns]
            if missing_cols:
                logger.error(f"Missing required columns: {', '.join(missing_cols)}")
                return None

            logger.debug("Converting required columns to numeric...")
            for col in required_columns:
                features_df[col] = pd.to_numeric(features_df[col], errors='coerce')
            logger.debug("Numeric conversion done.")

            if features_df[required_columns].isnull().any().any():
                nan_info = features_df[required_columns].isnull().sum()
                logger.warning(f"NaNs found in input columns after numeric conversion:\n{nan_info[nan_info > 0]}")
                features_df[required_columns] = features_df[required_columns].ffill().bfill()
                logger.warning("Filled NaNs in input columns using ffill and bfill.")
                if features_df[required_columns].isnull().any().any():
                    logger.error("NaNs still present after ffill/bfill - input data might be entirely invalid.")
                    return None

            logger.debug("Calculating basic price features...")
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", category=RuntimeWarning)
                features_df['returns'] = features_df['close'].pct_change()
                features_df['log_returns'] = np.log(
                    features_df['close'].replace(0, 1e-9) / features_df['close'].shift(1).replace(0, 1e-9)).replace(
                    [np.inf, -np.inf], 0)
                features_df['hl_ratio'] = (features_df['high'] / features_df['low'].replace(0, 1e-9)).replace(
                    [np.inf, -np.inf], 1)
                features_df['co_ratio'] = (features_df['close'] / features_df['open'].replace(0, 1e-9)).replace(
                    [np.inf, -np.inf], 1)
            logger.debug("Basic price features calculated.")

            logger.debug("Calculating price movement features (rolling stats)...")
            for window in [3, 5, 10, 20, 50]:
                if len(features_df) >= window:
                    if self.use_bottleneck:
                        close_vals = features_df['close'].values
                        returns_vals = features_df['returns'].fillna(0).values
                        log_returns_vals = features_df['log_returns'].fillna(0).values
                        features_df[f'close_prev{window}_mean'] = bn.move_mean(close_vals, window=window,
                                                                               min_count=window)
                        features_df[f'returns_prev{window}_mean'] = bn.move_mean(returns_vals, window=window,
                                                                                 min_count=window)
                        features_df[f'returns_prev{window}_std'] = bn.move_std(returns_vals, window=window,
                                                                               min_count=window, ddof=1)
                        features_df[f'log_returns_prev{window}_mean'] = bn.move_mean(log_returns_vals,
                                                                                     window=window,
                                                                                     min_count=window)
                        features_df[f'log_returns_prev{window}_std'] = bn.move_std(log_returns_vals, window=window,
                                                                                   min_count=window, ddof=1)
                    else:
                        features_df[f'close_prev{window}_mean'] = features_df['close'].rolling(window=window,
                                                                                               min_periods=window).mean()
                        features_df[f'returns_prev{window}_mean'] = features_df['returns'].rolling(window=window,
                                                                                                   min_periods=window).mean()
                        features_df[f'returns_prev{window}_std'] = features_df['returns'].rolling(window=window,
                                                                                                  min_periods=window).std()
                        features_df[f'log_returns_prev{window}_mean'] = features_df['log_returns'].rolling(
                            window=window, min_periods=window).mean()
                        features_df[f'log_returns_prev{window}_std'] = features_df['log_returns'].rolling(
                            window=window, min_periods=window).std()

                    ma_col = features_df[f'close_prev{window}_mean']
                    features_df[f'price_rel_to_ma{window}'] = np.where(ma_col.notna() & (np.abs(ma_col) > 1e-9),
                                                                       features_df['close'] / ma_col, 1.0)
                    features_df[f'roc_{window}'] = features_df['close'].pct_change(periods=window)
                else:
                    logger.warning(
                        f"Skipping window {window} calculations due to insufficient data length ({len(features_df)}).")
            logger.debug("Price movement features calculated.")

            if self.use_ta_features:
                logger.debug("Adding technical indicators...")
                self._add_technical_indicators(features_df)
                logger.debug("Technical indicators added.")

            if self.use_price_patterns:
                logger.debug("Adding price patterns...")
                self._add_price_patterns(features_df)
                logger.debug("Price patterns added.")

            if self.use_volatility_features:
                logger.debug("Adding volatility features...")
                self._add_volatility_features(features_df)
                logger.debug("Volatility features added.")

            if self.use_volume_features and 'volume' in features_df.columns:
                logger.debug("Adding volume features...")
                self._add_volume_features(features_df)
                logger.debug("Volume features added.")
            elif self.use_volume_features:
                logger.warning("Volume features enabled but 'volume' column missing.")

            if isinstance(features_df.index, pd.DatetimeIndex):
                logger.debug("Adding time features...")
                self._add_time_features(features_df)
                logger.debug("Time features added.")
            else:
                logger.warning("Cannot add time features: DataFrame index is not DatetimeIndex.")

            logger.debug("Creating target variables...")
            for horizon in self.target_horizons:
                features_df[f'target_price_change_{horizon}'] = features_df['close'].shift(-horizon) / features_df[
                    'close'].replace(0, 1e-9) - 1
                features_df[f'target_direction_{horizon}'] = (
                            features_df[f'target_price_change_{horizon}'] > 0).astype(int)
            logger.debug("Target variables created.")

            logger.info("Skipping final dropna() in create_features. NaN handling deferred.")

            # Final check if empty (should not happen if dropna is removed unless input was bad)
            if features_df.empty:
                logger.error("Features DataFrame became empty even without final dropna.")
                return None

            final_feature_count = len(features_df.columns)
            logger.info(
                f"Created {final_feature_count} features from input data. Final shape (before potential NaNs): {features_df.shape}")
            if final_feature_count != 147:
                logger.warning(f"Expected 147 features based on original log, but created {final_feature_count}.")

            logger.debug("Returning DataFrame with potential NaNs from create_features.")
            return features_df  # << بازگرداندن DataFrame با NaN ها

        except Exception as e:
            logger.error(f"Error during overall feature creation process: {str(e)}", exc_info=True)
            logger.error("Failed to create features")
            return None

    def _add_technical_indicators(self, df: pd.DataFrame) -> None:
        """Add comprehensive technical indicators"""
        try:
            # Convert to numpy arrays for talib
            open_prices = df['open'].values.astype(np.float64)
            high_prices = df['high'].values.astype(np.float64)
            low_prices = df['low'].values.astype(np.float64)
            close_prices = df['close'].values.astype(np.float64)
            volumes = df['volume'].values.astype(np.float64) if 'volume' in df.columns else None

            # Check if we have enough data for talib functions
            if len(close_prices) < 50:  # Most talib functions need at least 50 data points
                logger.warning(
                    f"Not enough data for talib indicators (need at least 50 points, have {len(close_prices)})")
                return

            # Trend indicators
            df['sma_10'] = talib.SMA(close_prices, timeperiod=10)
            df['sma_20'] = talib.SMA(close_prices, timeperiod=20)
            df['sma_50'] = talib.SMA(close_prices, timeperiod=50)
            df['ema_10'] = talib.EMA(close_prices, timeperiod=10)
            df['ema_20'] = talib.EMA(close_prices, timeperiod=20)
            df['ema_50'] = talib.EMA(close_prices, timeperiod=50)

            # MACD
            macd, macd_signal, macd_hist = talib.MACD(close_prices, fastperiod=12, slowperiod=26, signalperiod=9)
            df['macd'] = macd
            df['macd_signal'] = macd_signal
            df['macd_hist'] = macd_hist
            df['macd_diff'] = macd - macd_signal

            # RSI
            for period in [6, 14, 21]:
                df[f'rsi_{period}'] = talib.RSI(close_prices, timeperiod=period)

            # Stochastic
            slowk, slowd = talib.STOCH(high_prices, low_prices, close_prices,
                                       fastk_period=14, slowk_period=3, slowk_matype=0,
                                       slowd_period=3, slowd_matype=0)
            df['stoch_k'] = slowk
            df['stoch_d'] = slowd
            df['stoch_diff'] = slowk - slowd

            # Bollinger Bands
            upper, middle, lower = talib.BBANDS(close_prices, timeperiod=20, nbdevup=2, nbdevdn=2, matype=0)
            df['bb_upper'] = upper
            df['bb_middle'] = middle
            df['bb_lower'] = lower
            df['bb_width'] = (upper - lower) / middle
            # Handle division by zero
            df['bb_position'] = (close_prices - lower) / (upper - lower)
            df['bb_position'] = df['bb_position'].fillna(0.5)  # Fill NaN with neutral position

            # ADX (Trend strength)
            df['adx'] = talib.ADX(high_prices, low_prices, close_prices, timeperiod=14)
            df['adx_trend'] = ((df['adx'] > 25) & (df['adx'] > df['adx'].shift(1))).astype(int)

            # Add more indicators as needed...

            logger.debug(f"Added technical indicators to DataFrame")

        except Exception as e:
            logger.error(f"Error adding technical indicators: {str(e)}", exc_info=True)

    def _add_price_patterns(self, df: pd.DataFrame) -> None:
        """Add price pattern recognition features"""
        try:
            # Convert to numpy arrays for talib
            open_prices = df['open'].values.astype(np.float64)
            high_prices = df['high'].values.astype(np.float64)
            low_prices = df['low'].values.astype(np.float64)
            close_prices = df['close'].values.astype(np.float64)

            # Candlestick patterns
            pattern_functions = [
                (talib.CDLHAMMER, 'hammer'),
                (talib.CDLINVERTEDHAMMER, 'inverted_hammer'),
                (talib.CDLENGULFING, 'engulfing'),
                (talib.CDLMORNINGSTAR, 'morning_star'),
                (talib.CDLEVENINGSTAR, 'evening_star'),
                (talib.CDLHARAMI, 'harami'),
                (talib.CDLHARAMICROSS, 'harami_cross'),
                (talib.CDLDOJI, 'doji'),
                (talib.CDLSHOOTINGSTAR, 'shooting_star'),
                (talib.CDLMARUBOZU, 'marubozu'),
                (talib.CDLHANGINGMAN, 'hanging_man'),
                (talib.CDLSPINNINGTOP, 'spinning_top'),
                (talib.CDL3WHITESOLDIERS, 'three_white_soldiers'),
                (talib.CDL3BLACKCROWS, 'three_black_crows'),
                (talib.CDLDRAGONFLYDOJI, 'dragonfly_doji'),
                (talib.CDLGRAVESTONEDOJI, 'gravestone_doji'),
                (talib.CDL3INSIDE, 'three_inside'),
                (talib.CDL3OUTSIDE, 'three_outside'),
                (talib.CDLBELTHOLD, 'belt_hold'),
                (talib.CDLCOUNTERATTACK, 'counterattack'),
                (talib.CDLPIERCING, 'piercing')
            ]

            for func, name in pattern_functions:
                df[f'pattern_{name}'] = func(open_prices, high_prices, low_prices, close_prices)

            # Custom pattern features
            df['body_size'] = abs(df['close'] - df['open']) / (df['high'] - df['low'])
            df['upper_shadow'] = (df['high'] - df[['open', 'close']].max(axis=1)) / (df['high'] - df['low'])
            df['lower_shadow'] = (df[['open', 'close']].min(axis=1) - df['low']) / (df['high'] - df['low'])

            # Price gaps
            df['gap_up'] = (df['low'] > df['high'].shift(1)).astype(int)
            df['gap_down'] = (df['high'] < df['low'].shift(1)).astype(int)

            # Higher highs/lower lows detection
            for i in range(2, 6):
                df[f'higher_high_{i}'] = (df['high'] > df['high'].shift(1).rolling(i).max()).astype(int)
                df[f'lower_low_{i}'] = (df['low'] < df['low'].shift(1).rolling(i).min()).astype(int)

            logger.debug(f"Added price pattern features")

        except Exception as e:
            logger.error(f"Error adding price patterns: {e}", exc_info=True)

    def _add_volatility_features(self, df: pd.DataFrame) -> None:
        """Add volatility and regime detection features"""
        try:
            # Historical volatility
            for window in [5, 10, 20, 50]:
                df[f'volatility_{window}'] = df['returns'].rolling(window=window).std() * (252 ** 0.5)  # Annualized

            # Volatility ratios
            df['volatility_ratio_5_20'] = df['volatility_5'] / df['volatility_20']
            df['volatility_ratio_10_50'] = df['volatility_10'] / df['volatility_50']

            # Volatility trend
            df['volatility_trend'] = (df['volatility_10'] > df['volatility_10'].shift(5)).astype(int)

            # Garman-Klass volatility
            log_hl = np.log(df['high'] / df['low'])
            log_co = np.log(df['close'] / df['open'])
            df['gk_vol'] = 0.5 * log_hl ** 2 - (2 * np.log(2) - 1) * log_co ** 2

            # Parkinson volatility
            df['parkinson_vol'] = np.log(df['high'] / df['low']) ** 2 / (4 * np.log(2))

            # GARCH-like simple implementation
            returns = df['returns'].fillna(0)
            alpha, beta = 0.1, 0.9  # GARCH(1,1) parameters
            vol = np.zeros_like(returns)
            vol[0] = returns[0] ** 2

            for i in range(1, len(returns)):
                vol[i] = alpha * returns[i - 1] ** 2 + beta * vol[i - 1]

            df['garch_vol'] = np.sqrt(vol)

            # Volatility regimes
            for window in [20, 50]:
                vol_series = df[f'volatility_{window}']
                vol_mean = vol_series.rolling(window=window * 2).mean()
                vol_std = vol_series.rolling(window=window * 2).std()

                df[f'vol_regime_{window}'] = np.select(
                    [vol_series > (vol_mean + vol_std), vol_series < (vol_mean - vol_std)],
                    [1, -1],  # 1 for high volatility, -1 for low volatility
                    default=0  # 0 for normal volatility
                )

            logger.debug(f"Added volatility features")

        except Exception as e:
            logger.error(f"Error adding volatility features: {e}", exc_info=True)

    def _add_volume_features(self, df: pd.DataFrame) -> None:
        """Add volume-based features"""
        try:
            if 'volume' not in df.columns:
                return

            # Basic volume features
            for window in [5, 10, 20, 50]:
                df[f'volume_ma_{window}'] = df['volume'].rolling(window=window).mean()
                df[f'volume_ratio_{window}'] = df['volume'] / df[f'volume_ma_{window}']

            # Volume trend
            df['volume_trend'] = (df['volume'] > df['volume_ma_20']).astype(int)

            # Volume volatility
            df['volume_volatility'] = df['volume'].rolling(window=10).std() / df['volume_ma_10']

            # Price-volume divergence
            df['price_up_vol_up'] = ((df['returns'] > 0) & (df['volume'] > df['volume'].shift(1))).astype(int)
            df['price_down_vol_up'] = ((df['returns'] < 0) & (df['volume'] > df['volume'].shift(1))).astype(int)
            df['price_up_vol_down'] = ((df['returns'] > 0) & (df['volume'] < df['volume'].shift(1))).astype(int)
            df['price_down_vol_down'] = ((df['returns'] < 0) & (df['volume'] < df['volume'].shift(1))).astype(int)

            # Volume-weighted average price (VWAP)
            df['vwap'] = (df['volume'] * (df['high'] + df['low'] + df['close']) / 3).cumsum() / df[
                'volume'].cumsum()
            df['vwap_diff'] = df['close'] - df['vwap']

            # Volume-weighted RSI
            rsi = talib.RSI(df['close'].values, timeperiod=14)
            volume_sma = df['volume'].rolling(window=14).mean().values
            vol_rsi = rsi * volume_sma
            df['volume_rsi'] = vol_rsi / vol_rsi.max() * 100

            logger.debug(f"Added volume features")

        except Exception as e:
            logger.error(f"Error adding volume features: {e}", exc_info=True)

    def _add_time_features(self, df: pd.DataFrame) -> None:
        """Add time-based features"""
        try:
            if not isinstance(df.index, pd.DatetimeIndex):
                return

            # Day of week, hour of day, etc.
            df['day_of_week'] = df.index.dayofweek
            df['hour_of_day'] = df.index.hour
            df['month'] = df.index.month
            df['is_month_start'] = df.index.is_month_start.astype(int)
            df['is_month_end'] = df.index.is_month_end.astype(int)

            # Cyclical encoding of time features
            df['day_sin'] = np.sin(df['day_of_week'] * (2 * np.pi / 7))
            df['day_cos'] = np.cos(df['day_of_week'] * (2 * np.pi / 7))
            df['hour_sin'] = np.sin(df['hour_of_day'] * (2 * np.pi / 24))
            df['hour_cos'] = np.cos(df['hour_of_day'] * (2 * np.pi / 24))
            df['month_sin'] = np.sin((df['month'] - 1) * (2 * np.pi / 12))
            df['month_cos'] = np.cos((df['month'] - 1) * (2 * np.pi / 12))

            logger.debug(f"Added time features")

        except Exception as e:
            logger.error(f"Error adding time features: {e}", exc_info=True)

    def prepare_data_for_ml(self, features_df: pd.DataFrame, target_horizon: int = 1, train_ratio: float = 0.8,
                            scale_features: bool = True, feature_selection: bool = True) -> Dict[str, Any]:
        """آماده‌سازی ویژگی‌ها برای مدل‌های یادگیری ماشین با مدیریت صحیح NaN"""
        try:
            # کپی داده‌ها برای جلوگیری از تغییر ناخواسته داده‌های اصلی
            df = features_df.copy()

            logger.debug(
                f"Starting prepare_data_for_ml with DataFrame shape: {df.shape if df is not None else 'None'}")

            # معالجه مقادیر NaN به جای حذف ردیف‌ها
            target_col = f'target_direction_{target_horizon}'
            if target_col not in df.columns:
                logger.error(f"Target column {target_col} not found in dataframe")
                return None

            # بررسی تعداد مقادیر NaN قبل از پر کردن
            nan_counts_before = df.isnull().sum()
            cols_with_nan = nan_counts_before[nan_counts_before > 0]
            logger.info(f"Columns with NaN values before filling: {len(cols_with_nan)} columns")
            logger.debug(f"Total NaN values before filling: {nan_counts_before.sum()}")

            # جداسازی داده‌های هدف
            y = df[target_col].copy()

            # استخراج ویژگی‌ها بدون ستون‌های هدف
            feature_cols = [col for col in df.columns if not col.startswith('target_')]
            X = df[feature_cols]

            # حذف ردیف‌هایی که هدف آن‌ها NaN است
            valid_indices = ~y.isna()
            X = X[valid_indices]
            y = y[valid_indices].values

            if len(X) == 0:
                logger.error("No data left after removing rows with NaN target values")
                return None

            logger.info(f"After handling NaN target values, {len(X)} valid samples remain for ML")

            # ===== تغییر مهم: تقسیم داده به آموزش و آزمایش قبل از مدیریت NaN =====
            # Time-based train-test split
            train_size = int(len(X) * train_ratio)
            X_train, X_test = X.iloc[:train_size], X.iloc[train_size:]
            y_train, y_test = y[:train_size], y[train_size:]

            # Check if we have enough data for training
            if len(X_train) < 100:  # Minimum required for training
                logger.error(f"Insufficient training data: {len(X_train)} samples")
                return None

            # ===== تغییر مهم: پر کردن NaN در ویژگی‌ها با میانگین داده‌های آموزشی =====
            # پر کردن مقادیر NaN در ویژگی‌ها با میانگین داده‌های آموزشی
            numeric_cols = X_train.select_dtypes(include=['number']).columns

            # محاسبه میانگین فقط از داده‌های آموزشی
            column_means = {}
            for col in numeric_cols:
                # برای هر ستون ویژگی، میانگین آن را محاسبه می‌کنیم
                # اگر همه مقادیر NaN باشند، از 0 استفاده می‌کنیم
                if X_train[col].isna().all():
                    column_means[col] = 0
                else:
                    column_means[col] = X_train[col].mean()

            # پر کردن NaN در داده‌های آموزشی و آزمایشی با میانگین‌های محاسبه شده
            for col in numeric_cols:
                X_train[col] = X_train[col].fillna(column_means[col])
                X_test[col] = X_test[col].fillna(column_means[col])

            # برای ستون‌های غیر عددی (اگر وجود داشته باشد)
            cat_cols = X_train.select_dtypes(exclude=['number']).columns
            for col in cat_cols:
                # برای ستون‌های دسته‌ای، مد (رایج‌ترین مقدار) از داده‌های آموزشی محاسبه می‌شود
                if not X_train[col].mode().empty:
                    most_common = X_train[col].mode()[0]
                else:
                    most_common = "unknown"

                X_train[col] = X_train[col].fillna(most_common)
                X_test[col] = X_test[col].fillna(most_common)

            # اطمینان از نداشتن مقادیر NaN باقیمانده
            if X_train.isnull().any().any() or X_test.isnull().any().any():
                logger.warning("Still have NaN values after filling. Using 0 for remaining NaNs.")
                X_train = X_train.fillna(0)
                X_test = X_test.fillna(0)

            # Feature scaling
            if scale_features:
                X_train_scaled, X_test_scaled = self._scale_features(X_train, X_test)
            else:
                X_train_scaled, X_test_scaled = X_train, X_test

            # Feature selection
            if feature_selection:
                X_train_selected, X_test_selected, selected_features = self._select_features(X_train_scaled,
                                                                                             X_test_scaled, y_train)
                self.selected_features = selected_features
                if len(self.selected_features) == 0:
                    logger.warning("No features selected during feature selection")
                    self.selected_features = set(X_train.columns)
                    X_train_selected, X_test_selected = X_train_scaled, X_test_scaled
            else:
                X_train_selected, X_test_selected = X_train_scaled, X_test_scaled
                self.selected_features = set(X_train.columns)

            # Convert to numpy arrays for models that require it
            X_train_array = X_train_selected.values if hasattr(X_train_selected, 'values') else X_train_selected
            X_test_array = X_test_selected.values if hasattr(X_test_selected, 'values') else X_test_selected

            logger.info(f"Prepared data with {X_train_array.shape[1]} features, "
                        f"train: {X_train_array.shape[0]} samples, test: {X_test_array.shape[0]} samples")

            return {
                'X_train': X_train_array,
                'X_test': X_test_array,
                'y_train': y_train,
                'y_test': y_test,
                'feature_names': list(self.selected_features),
                'train_dates': X_train.index if hasattr(X_train, 'index') else None,
                'test_dates': X_test.index if hasattr(X_test, 'index') else None,
                'scaler': self.scalers.get('features'),
                'column_means': column_means  # ذخیره میانگین‌ها برای استفاده در زمان پیش‌بینی
            }

        except Exception as e:
            logger.error(f"Error preparing data for ML: {str(e)}", exc_info=True)
            return None

    def _scale_features(self, X_train: pd.DataFrame, X_test: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Scale features using the specified scaler"""
        try:
            # جایگزینی مقادیر بی‌نهایت و بسیار بزرگ قبل از مقیاس‌دهی
            # ابتدا کپی را ایجاد می‌کنیم تا از تغییر ناخواسته داده‌های اصلی جلوگیری شود
            X_train_clean = X_train.copy()
            X_test_clean = X_test.copy()

            # جایگزینی مقادیر بی‌نهایت
            X_train_clean = X_train_clean.replace([np.inf, -np.inf], np.nan)
            X_test_clean = X_test_clean.replace([np.inf, -np.inf], np.nan)

            # محدود کردن مقادیر بسیار بزرگ (clipping)
            # مقادیر بزرگتر از 1e10 به 1e10 محدود می‌شوند
            max_value = 1e10
            min_value = -1e10

            for col in X_train_clean.columns:
                if X_train_clean[col].dtype in ['float64', 'float32']:
                    X_train_clean[col] = X_train_clean[col].clip(lower=min_value, upper=max_value)
                    X_test_clean[col] = X_test_clean[col].clip(lower=min_value, upper=max_value)

            # پر کردن هر مقدار NaN باقی‌مانده با میانگین ستون
            for col in X_train_clean.columns:
                col_mean = X_train_clean[col].mean()
                # اگر میانگین NaN باشد (تمام مقادیر NaN هستند)، از 0 استفاده کن
                if pd.isna(col_mean):
                    col_mean = 0
                X_train_clean[col] = X_train_clean[col].fillna(col_mean)
                X_test_clean[col] = X_test_clean[col].fillna(col_mean)

            # بررسی نهایی برای اطمینان از عدم وجود مقادیر بی‌نهایت یا NaN
            if np.any(np.isnan(X_train_clean.values)) or np.any(np.isinf(X_train_clean.values)):
                logger.warning("Still have NaN or infinite values after cleaning. Using more aggressive cleaning.")
                # جایگزینی تمام مقادیر غیرعادی با صفر
                X_train_clean = X_train_clean.clip(lower=min_value, upper=max_value).fillna(0)
                X_test_clean = X_test_clean.clip(lower=min_value, upper=max_value).fillna(0)

            # انتخاب scaler بر اساس تنظیمات
            if self.scaler_type == 'standard':
                scaler = StandardScaler()
            elif self.scaler_type == 'minmax':
                scaler = MinMaxScaler()
            else:  # default to robust
                scaler = RobustScaler()

            # اعمال fit و transform
            X_train_scaled = scaler.fit_transform(X_train_clean)
            X_test_scaled = scaler.transform(X_test_clean)

            # ذخیره scaler برای استفاده آینده
            self.scalers['features'] = scaler

            # تبدیل به DataFrame برای حفظ نام ستون‌ها
            X_train_scaled_df = pd.DataFrame(X_train_scaled, columns=X_train.columns, index=X_train.index)
            X_test_scaled_df = pd.DataFrame(X_test_scaled, columns=X_test.columns, index=X_test.index)

            return X_train_scaled_df, X_test_scaled_df

        except Exception as e:
            logger.error(f"Error scaling features: {e}", exc_info=True)
            # در صورت خطا، داده‌های اولیه را برگردان
            return X_train, X_test

    def _select_features(self, X_train: pd.DataFrame, X_test: pd.DataFrame, y_train: np.ndarray) -> Tuple[
        pd.DataFrame, pd.DataFrame, Set[str]]:
        """Select most important features"""
        try:
            if self.feature_selection_method == 'importance':
                # بررسی مجدد برای اطمینان از عدم وجود مقادیر بی‌نهایت یا NaN
                X_train_clean = X_train.copy()
                X_test_clean = X_test.copy()

                # محدود کردن اعداد بزرگ و جایگزینی مقادیر غیرعادی
                max_value = 1e10
                min_value = -1e10
                X_train_clean = X_train_clean.clip(lower=min_value, upper=max_value)
                X_test_clean = X_test_clean.clip(lower=min_value, upper=max_value)
                X_train_clean = X_train_clean.fillna(0)
                X_test_clean = X_test_clean.fillna(0)

                # Use Random Forest for feature importance
                rf = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
                rf.fit(X_train_clean, y_train)

                # Get feature importances
                importances = rf.feature_importances_
                feature_names = X_train.columns

                # Store feature importances
                self.feature_importance = {name: importance for name, importance in zip(feature_names, importances)}

                # Select features with importance above threshold
                selected_indices = np.where(importances > self.importance_threshold)[0]
                selected_features = set([feature_names[i] for i in selected_indices])

                # Filter features
                X_train_selected = X_train.iloc[:, selected_indices]
                X_test_selected = X_test.iloc[:, selected_indices]

                logger.info(f"Selected {len(selected_features)} features based on importance")

                return X_train_selected, X_test_selected, selected_features

            else:
                # No feature selection, return all features
                return X_train, X_test, set(X_train.columns)

        except Exception as e:
            logger.error(f"Error selecting features: {e}", exc_info=True)
            return X_train, X_test, set(X_train.columns)

    def prepare_sequence_data(self, features_df: pd.DataFrame, target_horizon: int = 1,
                              sequence_length: int = 10, train_ratio: float = 0.8) -> Dict[str, Any]:
        """آماده‌سازی داده‌های توالی برای شبکه‌های عصبی بازگشتی"""
        try:
            # کپی داده‌ها برای جلوگیری از تغییر ناخواسته داده‌های اصلی
            df = features_df.copy()

            if len(df) < sequence_length + target_horizon:
                logger.warning(
                    f"Not enough data for sequence preparation (need at least {sequence_length + target_horizon} samples)")
                return None

            # انتخاب هدف
            target_col = f'target_direction_{target_horizon}'
            if target_col not in df.columns:
                logger.warning(f"Target column {target_col} not found in dataframe")
                return None

            # بررسی تعداد مقادیر NaN قبل از پر کردن
            nan_counts_before = df.isnull().sum()
            cols_with_nan = nan_counts_before[nan_counts_before > 0]
            logger.info(f"Sequence data: Columns with NaN values before filling: {len(cols_with_nan)} columns")

            # جداسازی داده‌های هدف
            y = df[target_col].copy()

            # استخراج ویژگی‌ها بدون ستون‌های هدف
            feature_cols = [col for col in df.columns if not col.startswith('target_')]
            X = df[feature_cols]

            # پاکسازی داده‌ها: جایگزینی مقادیر بی‌نهایت و بسیار بزرگ
            # جایگزینی مقادیر بی‌نهایت با NaN
            X = X.replace([np.inf, -np.inf], np.nan)

            # محدود کردن مقادیر بسیار بزرگ
            max_value = 1e10
            min_value = -1e10
            for col in X.columns:
                if X[col].dtype in ['float64', 'float32']:
                    X[col] = X[col].clip(lower=min_value, upper=max_value)

            # پر کردن مقادیر NaN در ویژگی‌ها با میانگین یا 0
            numeric_cols = X.select_dtypes(include=['number']).columns
            for col in numeric_cols:
                if X[col].isna().all():
                    X[col] = X[col].fillna(0)
                else:
                    X[col] = X[col].fillna(X[col].mean())

            # برای ستون‌های غیر عددی (اگر وجود داشته باشد)
            cat_cols = X.select_dtypes(exclude=['number']).columns
            for col in cat_cols:
                X[col] = X[col].fillna(X[col].mode()[0] if not X[col].mode().empty else "unknown")

            # حذف ردیف‌هایی که هدف آن‌ها NaN است
            valid_indices = ~y.isna()
            X = X[valid_indices]
            y = y[valid_indices].values

            # اطمینان از داشتن ایندکس‌های متوالی بعد از فیلتر کردن
            X = X.reset_index(drop=True)
            y = pd.Series(y).reset_index(drop=True).values

            if len(X) < sequence_length + target_horizon:
                logger.error(f"After removing NaN targets, not enough data left: {len(X)} rows")
                return None

            logger.info(f"After handling NaN values, {len(X)} valid samples remain for sequence data")

            # بررسی نهایی برای اطمینان از عدم وجود مقادیر بی‌نهایت یا NaN
            if np.any(np.isnan(X.values)) or np.any(np.isinf(X.values)):
                logger.warning("Still have NaN or infinite values after cleaning. Using more aggressive cleaning.")
                # جایگزینی تمام مقادیر غیرعادی با صفر
                X = X.clip(lower=min_value, upper=max_value).fillna(0)

            # Scale features
            scaler = RobustScaler()
            X_scaled = scaler.fit_transform(X)
            X_scaled_df = pd.DataFrame(X_scaled, columns=X.columns, index=X.index)

            # Create sequences
            X_sequences = []
            y_sequences = []
            dates = []

            for i in range(len(X_scaled_df) - sequence_length - target_horizon + 1):
                X_sequences.append(X_scaled_df.iloc[i:i + sequence_length].values)
                y_sequences.append(y[i + sequence_length + target_horizon - 1])
                dates.append(X_scaled_df.index[i + sequence_length - 1])

            X_sequences = np.array(X_sequences)
            y_sequences = np.array(y_sequences)

            # Time-based train-test split
            train_size = int(len(X_sequences) * train_ratio)
            X_train = X_sequences[:train_size]
            X_test = X_sequences[train_size:]
            y_train = y_sequences[:train_size]
            y_test = y_sequences[train_size:]
            train_dates = dates[:train_size]
            test_dates = dates[train_size:]

            logger.info(f"Prepared sequence data with {X_sequences.shape[1:]} shape per sequence, "
                        f"train: {X_train.shape[0]} sequences, test: {X_test.shape[0]} sequences")

            return {
                'X_train': X_train,
                'X_test': X_test,
                'y_train': y_train,
                'y_test': y_test,
                'feature_names': list(X.columns),
                'sequence_length': sequence_length,
                'train_dates': train_dates,
                'test_dates': test_dates,
                'scaler': scaler
            }

        except Exception as e:
            logger.error(f"Error preparing sequence data: {e}", exc_info=True)
            return None

def save_state(self, filepath: str) -> bool:
    """Save feature engineer state to disk"""
    try:
        state = {
            'scalers': {name: joblib.dumps(scaler) for name, scaler in self.scalers.items()},
            'selected_features': list(self.selected_features),
            'feature_importance': self.feature_importance,
            'config': self.config_manager.get_config()
        }

        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(os.path.abspath(filepath)), exist_ok=True)

        # Save to file
        with open(filepath, 'wb') as f:
            joblib.dump(state, f)

        logger.info(f"ذخیره وضعیت مهندسی ویژگی در {filepath}")
        return True

    except Exception as e:
        logger.error(f"خطا در ذخیره وضعیت مهندسی ویژگی: {e}", exc_info=True)
        return False

def load_state(self, filepath: str) -> bool:
    """Load feature engineer state from disk"""
    try:
        with open(filepath, 'rb') as f:
            state = joblib.load(f)

        # Load scalers
        self.scalers = {name: joblib.loads(scaler_bytes) for name, scaler_bytes in state['scalers'].items()}

        # Load selected features
        self.selected_features = set(state['selected_features'])

        # Load feature importance
        self.feature_importance = state['feature_importance']

        # Load config (only for backward compatibility, not actually used with config_manager)
        loaded_config = state.get('config', {})

        logger.info(f"بارگذاری وضعیت مهندسی ویژگی از {filepath}")
        return True

    except Exception as e:
        logger.error(f"خطا در بارگذاری وضعیت مهندسی ویژگی: {e}", exc_info=True)
        return False

# -----------------------------------------------------------------------------
# Neural Network Models for Market Prediction
# -----------------------------------------------------------------------------

class LSTMModel:
    """LSTM model for time series prediction"""

    def __init__(self, config_manager: ConfigManager):
        """Initialize with configuration manager"""
        self.config_manager = config_manager
        self.config_manager.register_listener(self._handle_config_update)

        # دریافت تنظیمات اولیه
        self._load_config()

        self.model = None
        self.history = None

        logger.info(f"مدل LSTM با {len(self.units)} لایه مقداردهی شد")

    def _load_config(self):
        """بارگذاری تنظیمات از مدیریت پیکربندی"""
        # دریافت تنظیمات LSTM
        lstm_config = self.config_manager.get('lstm_model', {})

        # بارگذاری تنظیمات با مقادیر پیش‌فرض
        self.units = lstm_config.get("units", [64, 32])
        self.dropout = lstm_config.get("dropout", 0.2)
        self.learning_rate = lstm_config.get("learning_rate", 0.001)
        self.batch_size = lstm_config.get("batch_size", 32)
        self.epochs = lstm_config.get("epochs", 100)
        self.patience = lstm_config.get("patience", 10)
        self.bidirectional = lstm_config.get("bidirectional", True)

    def _handle_config_update(self, path: str, value: Any) -> None:
        """پردازش به‌روزرسانی پیکربندی"""
        # بررسی اینکه آیا مسیر مربوط به تنظیمات این کلاس است
        if path.startswith('lstm_model.'):
            # استخراج نام پارامتر
            param = path.split('.')[-1]

            # به‌روزرسانی مقدار پارامتر
            if hasattr(self, param):
                old_value = getattr(self, param)
                setattr(self, param, value)
                logger.info(f"تنظیم {param} در مدل LSTM از {old_value} به {value} تغییر یافت")

        # اگر کل بخش به‌روزرسانی شده باشد
        elif path == 'lstm_model' and isinstance(value, dict):
            self._load_config()
            logger.info("تمام تنظیمات مدل LSTM به‌روزرسانی شد")

    def build_model(self, input_shape: Tuple[int, int], output_shape: int = 1) -> None:
        """Build LSTM model"""
        try:
            # Clear session to avoid memory leak
            tf.keras.backend.clear_session()

            # Create model
            model = Sequential()

            # Add LSTM layers
            for i, units in enumerate(self.units):
                return_sequences = i < len(self.units) - 1  # Return sequences for all but last layer

                if i == 0:
                    # First layer
                    if self.bidirectional:
                        model.add(Bidirectional(LSTM(units, return_sequences=return_sequences,
                                                     activation='tanh', recurrent_activation='sigmoid',
                                                     kernel_regularizer=l1_l2(l1=1e-5, l2=1e-4)),
                                                input_shape=input_shape))
                    else:
                        model.add(LSTM(units, return_sequences=return_sequences,
                                       activation='tanh', recurrent_activation='sigmoid',
                                       kernel_regularizer=l1_l2(l1=1e-5, l2=1e-4),
                                       input_shape=input_shape))
                else:
                    # Subsequent layers
                    if self.bidirectional:
                        model.add(Bidirectional(LSTM(units, return_sequences=return_sequences,
                                                     activation='tanh', recurrent_activation='sigmoid',
                                                     kernel_regularizer=l1_l2(l1=1e-5, l2=1e-4))))
                    else:
                        model.add(LSTM(units, return_sequences=return_sequences,
                                       activation='tanh', recurrent_activation='sigmoid',
                                       kernel_regularizer=l1_l2(l1=1e-5, l2=1e-4)))

                # Add BatchNorm and Dropout after each LSTM layer
                model.add(BatchNormalization())
                model.add(Dropout(self.dropout))

            # Add Dense layers for output
            model.add(Dense(16, activation='relu', kernel_regularizer=l1_l2(l1=1e-5, l2=1e-4)))
            model.add(BatchNormalization())
            model.add(Dropout(self.dropout / 2))

            # Final output layer
            if output_shape == 1:
                model.add(Dense(1, activation='sigmoid'))  # Binary classification
            else:
                model.add(Dense(output_shape, activation='softmax'))  # Multi-class classification

            # Compile model
            optimizer = Adam(learning_rate=self.learning_rate)
            model.compile(optimizer=optimizer,
                          loss='binary_crossentropy' if output_shape == 1 else 'categorical_crossentropy',
                          metrics=['accuracy'])

            # Save model
            self.model = model

            logger.info(f"مدل LSTM با {model.count_params()} پارامتر ساخته شد")

        except Exception as e:
            logger.error(f"خطا در ساخت مدل LSTM: {e}", exc_info=True)

    def train(self, X_train: np.ndarray, y_train: np.ndarray, X_val: np.ndarray = None, y_val: np.ndarray = None) -> \
            Dict[str, Any]:
        """Train the LSTM model"""
        try:
            if self.model is None:
                input_shape = X_train.shape[1:]
                output_shape = 1 if len(y_train.shape) == 1 else y_train.shape[1]
                self.build_model(input_shape, output_shape)

            # Callbacks
            callbacks = [
                EarlyStopping(monitor='val_loss', patience=self.patience, restore_best_weights=True),
                ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=self.patience // 2, min_lr=1e-6)
            ]

            # Use validation data if provided, otherwise use 10% of training data
            if X_val is None or y_val is None:
                validation_split = 0.1
                validation_data = None
            else:
                validation_split = 0.0
                validation_data = (X_val, y_val)

            # Train model
            history = self.model.fit(
                X_train, y_train,
                epochs=self.epochs,
                batch_size=self.batch_size,
                validation_split=validation_split,
                validation_data=validation_data,
                callbacks=callbacks,
                verbose=0  # Silence output for cleaner logs
            )

            self.history = history.history

            # Get best epoch
            best_epoch = np.argmin(history.history['val_loss']) + 1

            # Calculate training metrics
            train_loss = history.history['loss'][best_epoch - 1]
            train_accuracy = history.history['accuracy'][best_epoch - 1]
            val_loss = history.history['val_loss'][best_epoch - 1]
            val_accuracy = history.history['val_accuracy'][best_epoch - 1]

            logger.info(
                f"آموزش مدل LSTM برای {best_epoch} دوره، دقت اعتبارسنجی: {val_accuracy:.4f}، خطای اعتبارسنجی: {val_loss:.4f}")

            return {
                'best_epoch': best_epoch,
                'train_loss': train_loss,
                'train_accuracy': train_accuracy,
                'val_loss': val_loss,
                'val_accuracy': val_accuracy
            }

        except Exception as e:
            logger.error(f"خطا در آموزش مدل LSTM: {e}", exc_info=True)
            return {'error': str(e)}

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Make predictions with the trained model"""
        try:
            if self.model is None:
                raise ValueError("مدل هنوز ساخته یا آموزش داده نشده است")

            predictions = self.model.predict(X, verbose=0)

            # Convert probabilities to binary predictions
            if predictions.shape[1] == 1:
                binary_predictions = (predictions > 0.5).astype(int).reshape(-1)
            else:
                binary_predictions = np.argmax(predictions, axis=1)

            return binary_predictions

        except Exception as e:
            logger.error(f"خطا در پیش‌بینی با مدل LSTM: {e}", exc_info=True)
            return np.array([])

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Get probability predictions"""
        try:
            if self.model is None:
                raise ValueError("مدل هنوز ساخته یا آموزش داده نشده است")

            return self.model.predict(X, verbose=0)

        except Exception as e:
            logger.error(f"خطا در دریافت احتمالات پیش‌بینی با مدل LSTM: {e}", exc_info=True)
            return np.array([])

    def evaluate(self, X_test: np.ndarray, y_test: np.ndarray) -> Dict[str, float]:
        """Evaluate model performance"""
        try:
            if self.model is None:
                raise ValueError("مدل هنوز ساخته یا آموزش داده نشده است")

            # Make predictions
            y_pred = self.predict(X_test)
            y_pred_proba = self.predict_proba(X_test)

            # Calculate metrics
            accuracy = accuracy_score(y_test, y_pred)

            # Precision and recall (handle multi-class case)
            if len(y_pred_proba.shape) > 1 and y_pred_proba.shape[1] > 1:
                precision = precision_score(y_test, y_pred, average='weighted')
                recall = recall_score(y_test, y_pred, average='weighted')
            else:
                precision = precision_score(y_test, y_pred)
                recall = recall_score(y_test, y_pred)

            # Loss
            loss, _ = self.model.evaluate(X_test, y_test, verbose=0)

            logger.info(
                f"ارزیابی مدل LSTM: دقت={accuracy:.4f}، صحت={precision:.4f}، فراخوانی={recall:.4f}")

            return {
                'accuracy': float(accuracy),
                'precision': float(precision),
                'recall': float(recall),
                'loss': float(loss)
            }

        except Exception as e:
            logger.error(f"خطا در ارزیابی مدل LSTM: {e}", exc_info=True)
            return {'error': str(e)}

    def save_model(self, filepath: str) -> bool:
        """Save model to disk"""
        try:
            if self.model is None:
                raise ValueError("مدل هنوز ساخته یا آموزش داده نشده است")

            # Create directory if it doesn't exist
            os.makedirs(os.path.dirname(os.path.abspath(filepath)), exist_ok=True)

            # Save model and history
            self.model.save(filepath)

            # Save history separately
            if self.history is not None:
                with open(f"{filepath}_history.json", 'w') as f:
                    json.dump(self.history, f)

            logger.info(f"مدل LSTM در {filepath} ذخیره شد")
            return True

        except Exception as e:
            logger.error(f"خطا در ذخیره مدل LSTM: {e}", exc_info=True)
            return False

    def load_model(self, filepath: str) -> bool:
        """Load model from disk"""
        try:
            # Load model
            self.model = load_model(filepath)

            # Load history if available
            try:
                with open(f"{filepath}_history.json", 'r') as f:
                    self.history = json.load(f)
            except (FileNotFoundError, json.JSONDecodeError):
                logger.warning(f"تاریخچه از {filepath}_history.json قابل بارگذاری نیست")

            logger.info(f"مدل LSTM از {filepath} بارگذاری شد")
            return True

        except Exception as e:
            logger.error(f"خطا در بارگذاری مدل LSTM: {e}", exc_info=True)
            return False

class TransformerModel:
    """Transformer model for time series prediction"""

    def __init__(self, config_manager: ConfigManager):
        """Initialize with configuration manager"""
        self.config_manager = config_manager
        self.config_manager.register_listener(self._handle_config_update)

        # دریافت تنظیمات اولیه
        self._load_config()

        self.model = None
        self.history = None

        logger.info(f"مدل Transformer با {self.num_transformer_blocks} بلوک مقداردهی شد")

    def _load_config(self):
        """بارگذاری تنظیمات از مدیریت پیکربندی"""
        # دریافت تنظیمات Transformer
        transformer_config = self.config_manager.get('transformer_model', {})

        # بارگذاری تنظیمات با مقادیر پیش‌فرض
        self.embedding_dim = transformer_config.get("embedding_dim", 64)
        self.num_heads = transformer_config.get("num_heads", 4)
        self.ff_dim = transformer_config.get("ff_dim", 64)
        self.num_transformer_blocks = transformer_config.get("num_transformer_blocks", 2)
        self.mlp_units = transformer_config.get("mlp_units", [32])
        self.dropout_rate = transformer_config.get("dropout_rate", 0.25)
        self.learning_rate = transformer_config.get("learning_rate", 0.001)
        self.batch_size = transformer_config.get("batch_size", 32)
        self.epochs = transformer_config.get("epochs", 100)
        self.patience = transformer_config.get("patience", 10)

    def _handle_config_update(self, path: str, value: Any) -> None:
        """پردازش به‌روزرسانی پیکربندی"""
        # بررسی اینکه آیا مسیر مربوط به تنظیمات این کلاس است
        if path.startswith('transformer_model.'):
            # استخراج نام پارامتر
            param = path.split('.')[-1]

            # به‌روزرسانی مقدار پارامتر
            if hasattr(self, param):
                old_value = getattr(self, param)
                setattr(self, param, value)
                logger.info(f"تنظیم {param} در مدل Transformer از {old_value} به {value} تغییر یافت")

        # اگر کل بخش به‌روزرسانی شده باشد
        elif path == 'transformer_model' and isinstance(value, dict):
            self._load_config()
            logger.info("تمام تنظیمات مدل Transformer به‌روزرسانی شد")

    def build_model(self, input_shape: Tuple[int, int], output_shape: int = 1) -> None:
        """Build Transformer model"""
        try:
            # Clear session to avoid memory leak
            tf.keras.backend.clear_session()

            # Define input
            inputs = Input(shape=input_shape)

            # Use 1D convolution for patch embedding
            x = Conv1D(filters=self.embedding_dim, kernel_size=1, padding="same")(inputs)
            x = BatchNormalization()(x)
            x = tf.keras.layers.Activation("relu")(x)

            # Transformer blocks
            for _ in range(self.num_transformer_blocks):
                # Attention block
                attention_output = MultiHeadAttention(
                    num_heads=self.num_heads, key_dim=self.embedding_dim // self.num_heads)(x, x)
                x = LayerNormalization(epsilon=1e-6)(attention_output + x)

                # Feed-forward block
                ffn = Dense(self.ff_dim, activation="relu")(x)
                ffn = Dropout(self.dropout_rate)(ffn)
                ffn = Dense(self.embedding_dim)(ffn)
                x = LayerNormalization(epsilon=1e-6)(x + ffn)

            # Global pooling
            x = GlobalAveragePooling1D()(x)

            # MLP head
            for dim in self.mlp_units:
                x = Dense(dim, activation="relu")(x)
                x = Dropout(self.dropout_rate)(x)

            # Output layer
            if output_shape == 1:
                outputs = Dense(1, activation="sigmoid")(x)
            else:
                outputs = Dense(output_shape, activation="softmax")(x)

            # Create model
            model = Model(inputs=inputs, outputs=outputs)

            # Compile model
            optimizer = Adam(learning_rate=self.learning_rate)
            model.compile(optimizer=optimizer,
                          loss='binary_crossentropy' if output_shape == 1 else 'categorical_crossentropy',
                          metrics=['accuracy'])

            # Save model
            self.model = model

            logger.info(f"مدل Transformer با {model.count_params()} پارامتر ساخته شد")

        except Exception as e:
            logger.error(f"خطا در ساخت مدل Transformer: {e}", exc_info=True)

    def train(self, X_train: np.ndarray, y_train: np.ndarray, X_val: np.ndarray = None, y_val: np.ndarray = None) -> \
            Dict[str, Any]:
        """Train the Transformer model"""
        try:
            if self.model is None:
                input_shape = X_train.shape[1:]
                output_shape = 1 if len(y_train.shape) == 1 else y_train.shape[1]
                self.build_model(input_shape, output_shape)

            # Callbacks
            callbacks = [
                EarlyStopping(monitor='val_loss', patience=self.patience, restore_best_weights=True),
                ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=self.patience // 2, min_lr=1e-6)
            ]

            # Use validation data if provided, otherwise use 10% of training data
            if X_val is None or y_val is None:
                validation_split = 0.1
                validation_data = None
            else:
                validation_split = 0.0
                validation_data = (X_val, y_val)

            # Train model
            history = self.model.fit(
                X_train, y_train,
                epochs=self.epochs,
                batch_size=self.batch_size,
                validation_split=validation_split,
                validation_data=validation_data,
                callbacks=callbacks,
                verbose=0  # Silence output for cleaner logs
            )

            self.history = history.history

            # Get best epoch
            best_epoch = np.argmin(history.history['val_loss']) + 1

            # Calculate training metrics
            train_loss = history.history['loss'][best_epoch - 1]
            train_accuracy = history.history['accuracy'][best_epoch - 1]
            val_loss = history.history['val_loss'][best_epoch - 1]
            val_accuracy = history.history['val_accuracy'][best_epoch - 1]

            logger.info(
                f"آموزش مدل Transformer برای {best_epoch} دوره، دقت اعتبارسنجی: {val_accuracy:.4f}، خطای اعتبارسنجی: {val_loss:.4f}")

            return {
                'best_epoch': best_epoch,
                'train_loss': train_loss,
                'train_accuracy': train_accuracy,
                'val_loss': val_loss,
                'val_accuracy': val_accuracy
            }

        except Exception as e:
            logger.error(f"خطا در آموزش مدل Transformer: {e}", exc_info=True)
            return {'error': str(e)}

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Make predictions with the trained model"""
        try:
            if self.model is None:
                raise ValueError("مدل هنوز ساخته یا آموزش داده نشده است")

            predictions = self.model.predict(X, verbose=0)

            # Convert probabilities to binary predictions
            if predictions.shape[1] == 1:
                binary_predictions = (predictions > 0.5).astype(int).reshape(-1)
            else:
                binary_predictions = np.argmax(predictions, axis=1)

            return binary_predictions

        except Exception as e:
            logger.error(f"خطا در پیش‌بینی با مدل Transformer: {e}", exc_info=True)
            return np.array([])

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Get probability predictions"""
        try:
            if self.model is None:
                raise ValueError("مدل هنوز ساخته یا آموزش داده نشده است")

            return self.model.predict(X, verbose=0)

        except Exception as e:
            logger.error(f"خطا در دریافت احتمالات پیش‌بینی با مدل Transformer: {e}", exc_info=True)
            return np.array([])

    def evaluate(self, X_test: np.ndarray, y_test: np.ndarray) -> Dict[str, float]:
        """Evaluate model performance"""
        try:
            if self.model is None:
                raise ValueError("مدل هنوز ساخته یا آموزش داده نشده است")

            # Make predictions
            y_pred = self.predict(X_test)
            y_pred_proba = self.predict_proba(X_test)

            # Calculate metrics
            accuracy = accuracy_score(y_test, y_pred)

            # Precision and recall (handle multi-class case)
            if len(y_pred_proba.shape) > 1 and y_pred_proba.shape[1] > 1:
                precision = precision_score(y_test, y_pred, average='weighted')
                recall = recall_score(y_test, y_pred, average='weighted')
            else:
                precision = precision_score(y_test, y_pred)
                recall = recall_score(y_test, y_pred)

            # Loss
            loss, _ = self.model.evaluate(X_test, y_test, verbose=0)

            logger.info(
                f"ارزیابی مدل Transformer: دقت={accuracy:.4f}، صحت={precision:.4f}، فراخوانی={recall:.4f}")

            return {
                'accuracy': float(accuracy),
                'precision': float(precision),
                'recall': float(recall),
                'loss': float(loss)
            }

        except Exception as e:
            logger.error(f"خطا در ارزیابی مدل Transformer: {e}", exc_info=True)
            return {'error': str(e)}

    def save_model(self, filepath: str) -> bool:
        """Save model to disk"""
        try:
            if self.model is None:
                raise ValueError("مدل هنوز ساخته یا آموزش داده نشده است")

            # Create directory if it doesn't exist
            os.makedirs(os.path.dirname(os.path.abspath(filepath)), exist_ok=True)

            # Save model and history
            self.model.save(filepath)

            # Save history separately
            if self.history is not None:
                with open(f"{filepath}_history.json", 'w') as f:
                    json.dump(self.history, f)

            logger.info(f"مدل Transformer در {filepath} ذخیره شد")
            return True

        except Exception as e:
            logger.error(f"خطا در ذخیره مدل Transformer: {e}", exc_info=True)
            return False

    def load_model(self, filepath: str) -> bool:
        """Load model from disk"""
        try:
            # Load model
            self.model = load_model(filepath)

            # Load history if available
            try:
                with open(f"{filepath}_history.json", 'r') as f:
                    self.history = json.load(f)
            except (FileNotFoundError, json.JSONDecodeError):
                logger.warning(f"تاریخچه از {filepath}_history.json قابل بارگذاری نیست")

            logger.info(f"مدل Transformer از {filepath} بارگذاری شد")
            return True

        except Exception as e:
            logger.error(f"خطا در بارگذاری مدل Transformer: {e}", exc_info=True)
            return False

class CNNModel:
    """CNN model for time series prediction"""

    def __init__(self, config_manager: ConfigManager):
        """Initialize with configuration manager"""
        self.config_manager = config_manager
        self.config_manager.register_listener(self._handle_config_update)

        # دریافت تنظیمات اولیه
        self._load_config()

        self.model = None
        self.history = None

        logger.info(f"مدل CNN با {len(self.filters)} لایه کانولوشنی مقداردهی شد")

    def _load_config(self):
        """بارگذاری تنظیمات از مدیریت پیکربندی"""
        # دریافت تنظیمات CNN
        cnn_config = self.config_manager.get('cnn_model', {})

        # بارگذاری تنظیمات با مقادیر پیش‌فرض
        self.filters = cnn_config.get("filters", [64, 128, 128])
        self.kernel_sizes = cnn_config.get("kernel_sizes", [3, 3, 3])
        self.pool_sizes = cnn_config.get("pool_sizes", [2, 2, 2])
        self.dense_units = cnn_config.get("dense_units", [64, 32])
        self.dropout_rate = cnn_config.get("dropout_rate", 0.3)
        self.learning_rate = cnn_config.get("learning_rate", 0.001)
        self.batch_size = cnn_config.get("batch_size", 32)
        self.epochs = cnn_config.get("epochs", 100)
        self.patience = cnn_config.get("patience", 10)

    def _handle_config_update(self, path: str, value: Any) -> None:
        """پردازش به‌روزرسانی پیکربندی"""
        # بررسی اینکه آیا مسیر مربوط به تنظیمات این کلاس است
        if path.startswith('cnn_model.'):
            # استخراج نام پارامتر
            param = path.split('.')[-1]

            # به‌روزرسانی مقدار پارامتر
            if hasattr(self, param):
                old_value = getattr(self, param)
                setattr(self, param, value)
                logger.info(f"تنظیم {param} در مدل CNN از {old_value} به {value} تغییر یافت")

        # اگر کل بخش به‌روزرسانی شده باشد
        elif path == 'cnn_model' and isinstance(value, dict):
            self._load_config()
            logger.info("تمام تنظیمات مدل CNN به‌روزرسانی شد")

    def build_model(self, input_shape: Tuple[int, int], output_shape: int = 1) -> None:
        """Build CNN model"""
        try:
            # Clear session to avoid memory leak
            tf.keras.backend.clear_session()

            # Create model
            model = Sequential()

            # Add CNN layers
            for i, (filters, kernel_size, pool_size) in enumerate(
                    zip(self.filters, self.kernel_sizes, self.pool_sizes)):
                if i == 0:
                    # First layer
                    model.add(Conv1D(filters=filters, kernel_size=kernel_size, activation='relu',
                                     padding='same', input_shape=input_shape))
                else:
                    # Subsequent layers
                    model.add(Conv1D(filters=filters, kernel_size=kernel_size, activation='relu', padding='same'))

                # Add BatchNorm, MaxPooling, and Dropout
                model.add(BatchNormalization())
                model.add(MaxPooling1D(pool_size=pool_size))
                model.add(Dropout(self.dropout_rate))

            # Flatten and dense layers
            model.add(Flatten())

            # Add Dense layers
            for units in self.dense_units:
                model.add(Dense(units, activation='relu'))
                model.add(BatchNormalization())
                model.add(Dropout(self.dropout_rate))

            # Output layer
            if output_shape == 1:
                model.add(Dense(1, activation='sigmoid'))  # Binary classification
            else:
                model.add(Dense(output_shape, activation='softmax'))  # Multi-class classification

            # Compile model
            optimizer = Adam(learning_rate=self.learning_rate)
            model.compile(optimizer=optimizer,
                          loss='binary_crossentropy' if output_shape == 1 else 'categorical_crossentropy',
                          metrics=['accuracy'])

            # Save model
            self.model = model

            logger.info(f"مدل CNN با {model.count_params()} پارامتر ساخته شد")

        except Exception as e:
            logger.error(f"خطا در ساخت مدل CNN: {e}", exc_info=True)

    def train(self, X_train: np.ndarray, y_train: np.ndarray, X_val: np.ndarray = None, y_val: np.ndarray = None) -> \
            Dict[str, Any]:
        """Train the CNN model"""
        try:
            if self.model is None:
                input_shape = X_train.shape[1:]
                output_shape = 1 if len(y_train.shape) == 1 else y_train.shape[1]
                self.build_model(input_shape, output_shape)

            # Callbacks
            callbacks = [
                EarlyStopping(monitor='val_loss', patience=self.patience, restore_best_weights=True),
                ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=self.patience // 2, min_lr=1e-6)
            ]

            # Use validation data if provided, otherwise use 10% of training data
            if X_val is None or y_val is None:
                validation_split = 0.1
                validation_data = None
            else:
                validation_split = 0.0
                validation_data = (X_val, y_val)

            # Train model
            history = self.model.fit(
                X_train, y_train,
                epochs=self.epochs,
                batch_size=self.batch_size,
                validation_split=validation_split,
                validation_data=validation_data,
                callbacks=callbacks,
                verbose=0  # Silence output for cleaner logs
            )

            self.history = history.history

            # Get best epoch
            best_epoch = np.argmin(history.history['val_loss']) + 1

            # Calculate training metrics
            train_loss = history.history['loss'][best_epoch - 1]
            train_accuracy = history.history['accuracy'][best_epoch - 1]
            val_loss = history.history['val_loss'][best_epoch - 1]
            val_accuracy = history.history['val_accuracy'][best_epoch - 1]

            logger.info(
                f"آموزش مدل CNN برای {best_epoch} دوره، دقت اعتبارسنجی: {val_accuracy:.4f}، خطای اعتبارسنجی: {val_loss:.4f}")

            return {
                'best_epoch': best_epoch,
                'train_loss': train_loss,
                'train_accuracy': train_accuracy,
                'val_loss': val_loss,
                'val_accuracy': val_accuracy
            }

        except Exception as e:
            logger.error(f"خطا در آموزش مدل CNN: {e}", exc_info=True)
            return {'error': str(e)}

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Make predictions with the trained model"""
        try:
            if self.model is None:
                raise ValueError("مدل هنوز ساخته یا آموزش داده نشده است")

            predictions = self.model.predict(X, verbose=0)

            # Convert probabilities to binary predictions
            if predictions.shape[1] == 1:
                binary_predictions = (predictions > 0.5).astype(int).reshape(-1)
            else:
                binary_predictions = np.argmax(predictions, axis=1)

            return binary_predictions

        except Exception as e:
            logger.error(f"خطا در پیش‌بینی با مدل CNN: {e}", exc_info=True)
            return np.array([])

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Get probability predictions"""
        try:
            if self.model is None:
                raise ValueError("مدل هنوز ساخته یا آموزش داده نشده است")

            return self.model.predict(X, verbose=0)

        except Exception as e:
            logger.error(f"خطا در دریافت احتمالات پیش‌بینی با مدل CNN: {e}", exc_info=True)
            return np.array([])

    def evaluate(self, X_test: np.ndarray, y_test: np.ndarray) -> Dict[str, float]:
        """Evaluate model performance"""
        try:
            if self.model is None:
                raise ValueError("مدل هنوز ساخته یا آموزش داده نشده است")

            # Make predictions
            y_pred = self.predict(X_test)
            y_pred_proba = self.predict_proba(X_test)

            # Calculate metrics
            accuracy = accuracy_score(y_test, y_pred)

            # Precision and recall (handle multi-class case)
            if len(y_pred_proba.shape) > 1 and y_pred_proba.shape[1] > 1:
                precision = precision_score(y_test, y_pred, average='weighted')
                recall = recall_score(y_test, y_pred, average='weighted')
            else:
                precision = precision_score(y_test, y_pred)
                recall = recall_score(y_test, y_pred)

            # Loss
            loss, _ = self.model.evaluate(X_test, y_test, verbose=0)

            logger.info(
                f"ارزیابی مدل CNN: دقت={accuracy:.4f}، صحت={precision:.4f}، فراخوانی={recall:.4f}")

            return {
                'accuracy': float(accuracy),
                'precision': float(precision),
                'recall': float(recall),
                'loss': float(loss)
            }

        except Exception as e:
            logger.error(f"خطا در ارزیابی مدل CNN: {e}", exc_info=True)
            return {'error': str(e)}

    def save_model(self, filepath: str) -> bool:
        """Save model to disk"""
        try:
            if self.model is None:
                raise ValueError("مدل هنوز ساخته یا آموزش داده نشده است")

            # Create directory if it doesn't exist
            os.makedirs(os.path.dirname(os.path.abspath(filepath)), exist_ok=True)

            # Save model and history
            self.model.save(filepath)

            # Save history separately
            if self.history is not None:
                with open(f"{filepath}_history.json", 'w') as f:
                    json.dump(self.history, f)

            logger.info(f"مدل CNN در {filepath} ذخیره شد")
            return True

        except Exception as e:
            logger.error(f"خطا در ذخیره مدل CNN: {e}", exc_info=True)
            return False

    def load_model(self, filepath: str) -> bool:
        """Load model from disk"""
        try:
            # Load model
            self.model = load_model(filepath)

            # Load history if available
            try:
                with open(f"{filepath}_history.json", 'r') as f:
                    self.history = json.load(f)
            except (FileNotFoundError, json.JSONDecodeError):
                logger.warning(f"تاریخچه از {filepath}_history.json قابل بارگذاری نیست")

            logger.info(f"مدل CNN از {filepath} بارگذاری شد")
            return True

        except Exception as e:
            logger.error(f"خطا در بارگذاری مدل CNN: {e}", exc_info=True)
            return False

# -----------------------------------------------------------------------------
# Ensemble Model and Prediction Integration
# -----------------------------------------------------------------------------

class EnsembleModel:
    """Ensemble model for trading signal prediction and confidence scoring"""

    def __init__(self, config_manager: ConfigManager):
        """Initialize with configuration manager"""
        self.config_manager = config_manager
        self.config_manager.register_listener(self._handle_config_update)

        # دریافت تنظیمات اولیه
        self._load_config()

        self.models = {}
        self.models_weights = {}
        self.models_metrics = {}
        self.calibrators = {}  # کالیبراتورها برای هر مدل
        self.scaler = None
        self.feature_map = {}  # نگاشت ویژگی‌ها

        logger.info(f"مدل Ensemble با روش {self.ensemble_method} مقداردهی شد")

    def _load_config(self):
        """بارگذاری تنظیمات از مدیریت پیکربندی"""
        # دریافت تنظیمات Ensemble
        ensemble_config = self.config_manager.get('ensemble', {})

        # بارگذاری تنظیمات با مقادیر پیش‌فرض
        self.ensemble_method = ensemble_config.get("method", "weighted")
        self.weights = ensemble_config.get("weights", {
            "lstm": 0.35,
            "transformer": 0.35,
            "cnn": 0.15,
            "xgboost": 0.15
        })
        self.calibrate_probabilities = ensemble_config.get("calibrate_probabilities", True)
        self.min_confidence_threshold = ensemble_config.get("min_confidence_threshold", 0.65)

    def _handle_config_update(self, path: str, value: Any) -> None:
        """پردازش به‌روزرسانی پیکربندی"""
        # بررسی اینکه آیا مسیر مربوط به تنظیمات این کلاس است
        if path.startswith('ensemble.'):
            # استخراج نام پارامتر
            param = path.split('.')[-1]

            # به‌روزرسانی مقدار پارامتر
            if param == "weights" and isinstance(value, dict):
                self.weights = value
                # به‌روزرسانی weights در مدل‌های فعلی
                for model_name in self.models:
                    self.models_weights[model_name] = self.weights.get(model_name, 1.0)
                logger.info(f"وزن‌های مدل‌ها به‌روزرسانی شد: {self.weights}")
            elif hasattr(self, param):
                old_value = getattr(self, param)
                setattr(self, param, value)
                logger.info(f"تنظیم {param} در مدل Ensemble از {old_value} به {value} تغییر یافت")

        # اگر کل بخش به‌روزرسانی شده باشد
        elif path == 'ensemble' and isinstance(value, dict):
            self._load_config()
            # به‌روزرسانی weights در مدل‌های فعلی
            for model_name in self.models:
                self.models_weights[model_name] = self.weights.get(model_name, 1.0)
            logger.info("تمام تنظیمات مدل Ensemble به‌روزرسانی شد")

    def add_model(self, model_name: str, model: Any, weight: float = None) -> None:
        """Add a model to the ensemble"""
        self.models[model_name] = model

        # Use provided weight or default from config
        if weight is not None:
            self.models_weights[model_name] = weight
        else:
            self.models_weights[model_name] = self.weights.get(model_name, 1.0)

        logger.info(f"مدل {model_name} با وزن {self.models_weights[model_name]} به Ensemble اضافه شد")

    def remove_model(self, model_name: str) -> None:
        """Remove a model from the ensemble"""
        if model_name in self.models:
            del self.models[model_name]
            del self.models_weights[model_name]

            logger.info(f"مدل {model_name} از Ensemble حذف شد")

    def update_weights_based_on_performance(self, metrics: Dict[str, Dict[str, float]]) -> None:
        """Update model weights based on performance metrics"""
        if not metrics:
            logger.warning("متریک برای به‌روزرسانی وزن ارائه نشده است")
            return

        # Store metrics
        self.models_metrics = metrics

        total_accuracy = 0
        accuracies = {}

        # Calculate total accuracy
        for model_name, model_metrics in metrics.items():
            if model_name in self.models and 'accuracy' in model_metrics:
                accuracy = model_metrics['accuracy']
                accuracies[model_name] = accuracy
                total_accuracy += accuracy

        if total_accuracy <= 0:
            logger.warning("دقت کل صفر یا منفی است، وزن‌های اصلی را نگه می‌داریم")
            return

        # Update weights based on accuracy contribution
        for model_name, accuracy in accuracies.items():
            self.models_weights[model_name] = accuracy / total_accuracy

        logger.info(f"وزن‌های Ensemble بر اساس عملکرد به‌روزرسانی شد: {self.models_weights}")

    def _calibrate_model_probabilities(self, model_name: str, X_val: np.ndarray, y_val: np.ndarray) -> None:
        """کالیبراسیون احتمالات مدل با استفاده از روش Platt Scaling"""
        try:
            from sklearn.calibration import CalibratedClassifierCV
            import numpy as np

            if model_name not in self.models:
                logger.warning(f"مدل {model_name} برای کالیبراسیون یافت نشد")
                return

            # ایجاد یک wrapper بهبود یافته برای مدل
            class ModelWrapper:
                def __init__(self, model):
                    self.model = model
                    self.classes_ = np.array([0, 1])  # تنظیم کلاس‌ها برای مسئله دودویی
                    self._estimator_type = "classifier"  # تعیین نوع مدل به عنوان classifier

                def predict(self, X):
                    try:
                        preds = self.model.predict(X)
                        # تبدیل به آرایه یک بعدی و بررسی صحت مقادیر
                        if hasattr(preds, 'shape') and len(preds.shape) > 1:
                            preds = preds.flatten()
                        # تبدیل به 0 و 1
                        return (preds > 0.5).astype(int)
                    except Exception as e:
                        logger.error(f"خطا در ModelWrapper.predict: {str(e)}")
                        # در صورت خطا، یک آرایه پیش‌فرض برگردان
                        return np.zeros(len(X), dtype=int)

                def predict_proba(self, X):
                    try:
                        proba = self.model.predict_proba(X)

                        # بررسی شکل خروجی و اصلاح آن برای اطمینان از فرمت (n_samples, n_classes=2)
                        if len(proba.shape) == 1:  # اگر آرایه یک بعدی است (n_samples,)
                            proba_reshaped = np.zeros((len(proba), 2))
                            proba_reshaped[:, 1] = proba
                            proba_reshaped[:, 0] = 1 - proba
                            return proba_reshaped
                        elif proba.shape[1] == 1:  # اگر آرایه (n_samples, 1) است
                            proba_reshaped = np.zeros((proba.shape[0], 2))
                            proba_reshaped[:, 1] = proba.flatten()
                            proba_reshaped[:, 0] = 1 - proba.flatten()
                            return proba_reshaped
                        elif proba.shape[1] == 2:  # اگر قبلاً در فرمت صحیح (n_samples, 2) است
                            return proba
                        else:
                            # اگر شکل غیرمنتظره است، لاگ کنیم و یک آرایه پیش‌فرض برگردانیم
                            logger.warning(f"شکل غیرمنتظره برای predict_proba: {proba.shape}")
                            default_proba = np.zeros((len(X), 2))
                            default_proba[:, 1] = 0.5  # مقادیر پیش‌فرض 50-50
                            default_proba[:, 0] = 0.5
                            return default_proba

                    except Exception as e:
                        logger.error(f"خطا در ModelWrapper.predict_proba: {str(e)}")
                        # در صورت خطا، یک آرایه احتمالات پیش‌فرض برگردان
                        default_proba = np.zeros((len(X), 2))
                        default_proba[:, 1] = 0.5  # مقادیر پیش‌فرض 50-50
                        default_proba[:, 0] = 0.5
                        return default_proba

                def fit(self, X, y):
                    # تنظیم مجدد classes_ بر اساس داده‌های ورودی
                    self.classes_ = np.unique(y)
                    # اطمینان از وجود حداقل دو کلاس
                    if len(self.classes_) == 1:
                        self.classes_ = np.array([0, 1])
                    return self

            # ایجاد wrapper برای مدل
            model_wrapper = ModelWrapper(self.models[model_name])

            # فراخوانی صریح fit برای تنظیم ویژگی‌های مورد نیاز
            model_wrapper.fit(X_val, y_val)

            # تنظیم response_method به 'predict' به جای پیش‌فرض
            calibrator = CalibratedClassifierCV(
                estimator=model_wrapper,
                cv='prefit',  # مدل از قبل آموزش دیده است
                method='sigmoid',  # Platt scaling
                ensemble=False,  # برای جلوگیری از ایجاد مدل‌های متعدد
                n_jobs=1  # برای جلوگیری از مشکلات موازی‌سازی
            )

            # برازش کالیبراتور با داده‌های اعتبارسنجی
            calibrator.fit(X_val, y_val)

            # ذخیره کالیبراتور
            self.calibrators[model_name] = calibrator

            logger.info(f"کالیبراسیون احتمالات برای مدل {model_name} تکمیل شد")

        except Exception as e:
            logger.error(f"خطا در کالیبراسیون احتمالات برای مدل {model_name}: {e}", exc_info=True)

    def calibrate_models(self, X_seq_val: np.ndarray, y_val: np.ndarray, X_flat_val: np.ndarray = None) -> None:
        """کالیبراسیون تمام مدل‌های موجود در مجموعه"""
        if not self.calibrate_probabilities:
            logger.info("کالیبراسیون احتمالات در تنظیمات غیرفعال است")
            return

        logger.info("شروع کالیبراسیون احتمالات برای مدل‌های Ensemble...")

        # کالیبراسیون مدل‌های توالی (LSTM, Transformer, CNN)
        if 'lstm' in self.models:
            self._calibrate_model_probabilities('lstm', X_seq_val, y_val)

        if 'transformer' in self.models:
            self._calibrate_model_probabilities('transformer', X_seq_val, y_val)

        if 'cnn' in self.models:
            self._calibrate_model_probabilities('cnn', X_seq_val, y_val)

        # کالیبراسیون XGBoost (اگر داده مسطح موجود باشد)
        if 'xgboost' in self.models and X_flat_val is not None:
            self._calibrate_model_probabilities('xgboost', X_flat_val, y_val)

        logger.info("کالیبراسیون احتمالات برای تمام مدل‌ها تکمیل شد")

    def set_feature_map(self, feature_names: List[str]) -> None:
        """ایجاد نگاشت ویژگی‌ها برای استفاده در پیش‌بینی‌های آینده"""
        self.feature_map = {name: idx for idx, name in enumerate(feature_names)}
        logger.info(f"نگاشت ویژگی با {len(self.feature_map)} ویژگی ایجاد شد")

    def predict_proba(self, X_seq: np.ndarray, X_flat: np.ndarray = None) -> Tuple[
        np.ndarray, Dict[str, np.ndarray]]:
        """Get weighted probability predictions from all models with calibration"""
        model_probas = {}
        # Initialize ensemble_proba with correct shape based on the first model's output
        first_model_output_shape = None
        num_samples = X_seq.shape[0]  # Get number of samples from input

        # Collect probabilities from each model
        if 'lstm' in self.models:
            try:
                # استفاده از کالیبراتور اگر موجود باشد
                if 'lstm' in self.calibrators and self.calibrate_probabilities:
                    lstm_proba = self.calibrators['lstm'].predict_proba(X_seq)
                else:
                    lstm_proba = self.models['lstm'].predict_proba(X_seq)
                model_probas['lstm'] = lstm_proba
                if first_model_output_shape is None:
                    first_model_output_shape = lstm_proba.shape
            except Exception as e:
                logger.error(f"خطا در دریافت پیش‌بینی از مدل LSTM: {e}", exc_info=True)

        if 'transformer' in self.models:
            try:
                # استفاده از کالیبراتور اگر موجود باشد
                if 'transformer' in self.calibrators and self.calibrate_probabilities:
                    transformer_proba = self.calibrators['transformer'].predict_proba(X_seq)
                else:
                    transformer_proba = self.models['transformer'].predict_proba(X_seq)
                model_probas['transformer'] = transformer_proba
                if first_model_output_shape is None:
                    first_model_output_shape = transformer_proba.shape
            except Exception as e:
                logger.error(f"خطا در دریافت پیش‌بینی از مدل Transformer: {e}", exc_info=True)

        if 'cnn' in self.models:
            try:
                # استفاده از کالیبراتور اگر موجود باشد
                if 'cnn' in self.calibrators and self.calibrate_probabilities:
                    cnn_proba = self.calibrators['cnn'].predict_proba(X_seq)
                else:
                    cnn_proba = self.models['cnn'].predict_proba(X_seq)
                model_probas['cnn'] = cnn_proba
                if first_model_output_shape is None:
                    first_model_output_shape = cnn_proba.shape
            except Exception as e:
                logger.error(f"خطا در دریافت پیش‌بینی از مدل CNN: {e}", exc_info=True)

        # --- XGBoost Prediction and Shape Correction ---
        xgb_proba_corrected = None
        if 'xgboost' in self.models and X_flat is not None:
            try:
                # بهبود تطابق ویژگی‌ها
                feature_names = getattr(self.models['xgboost'], 'feature_names_in_', None)
                if feature_names is not None and hasattr(self, 'feature_map') and self.feature_map:
                    # اگر نقشه ویژگی داریم، استفاده از آن برای مرتب‌سازی/انتخاب ویژگی‌ها
                    if hasattr(X_flat, 'columns'):  # اگر DataFrame است
                        X_flat_adj = X_flat[feature_names].values
                    else:
                        # اگر numpy array است، باید از نقشه ویژگی استفاده کنیم
                        if len(self.feature_map) != X_flat.shape[1]:
                            logger.warning(
                                f"عدم تطابق ابعاد ویژگی: انتظار {len(self.feature_map)}، دریافت شده {X_flat.shape[1]}")

                        # استفاده از feature_map برای انتخاب/ترتیب‌بندی ویژگی‌ها
                        indices = [self.feature_map.get(name, None) for name in feature_names]
                        valid_indices = [i for i in indices if i is not None and i < X_flat.shape[1]]

                        if len(valid_indices) < len(feature_names):
                            logger.warning(f"تنها {len(valid_indices)}/{len(feature_names)} ویژگی یافت شد")

                        if valid_indices:
                            X_flat_adj = X_flat[:, valid_indices]
                        else:
                            logger.error("هیچ ویژگی معتبری برای پیش‌بینی XGBoost یافت نشد")
                            # خط continue حذف شد و با یک exception جایگزین شد
                            raise ValueError("هیچ ویژگی معتبری برای پیش‌بینی XGBoost یافت نشد")
                else:
                    # روش قبلی: بررسی و تطبیق تعداد ویژگی‌ها
                    expected_features = self.models['xgboost'].n_features_in_ if hasattr(self.models['xgboost'],
                                                                                         'n_features_in_') else None
                    actual_features = X_flat.shape[1] if len(X_flat.shape) > 1 else 1
                    X_flat_adj = X_flat  # پیش‌فرض به اصل آرایه

                    if expected_features is not None and expected_features != actual_features:
                        logger.warning(
                            f"عدم تطابق تعداد ویژگی‌های XGBoost: انتظار {expected_features}، دریافت شده {actual_features}. در حال تنظیم...")
                        if actual_features > expected_features:
                            X_flat_adj = X_flat[:, :expected_features]
                        elif actual_features < expected_features:
                            padding = np.zeros((X_flat.shape[0], expected_features - actual_features))
                            X_flat_adj = np.hstack([X_flat, padding])

                # Get probabilities from XGBoost with calibration if available
                if 'xgboost' in self.calibrators and self.calibrate_probabilities:
                    xgb_proba_raw = self.calibrators['xgboost'].predict_proba(X_flat_adj)
                else:
                    xgb_proba_raw = self.models['xgboost'].predict_proba(X_flat_adj)

                model_probas['xgboost'] = xgb_proba_raw  # Store the raw probabilities

                # *** SHAPE CORRECTION LOGIC ***
                if xgb_proba_raw.ndim == 1:
                    # If output is 1D (e.g., only prob of class 1), reshape and calculate class 0 prob
                    xgb_proba_corrected = np.vstack((1 - xgb_proba_raw, xgb_proba_raw)).T  # Shape (n, 2)
                elif xgb_proba_raw.shape[1] == 1:
                    # If output is (n, 1), reshape and calculate class 0 prob
                    xgb_proba_corrected = np.hstack((1 - xgb_proba_raw, xgb_proba_raw))  # Shape (n, 2)
                elif xgb_proba_raw.shape[1] == 2:
                    # Already in correct shape (n, 2)
                    xgb_proba_corrected = xgb_proba_raw
                else:
                    logger.error(f"شکل غیرمنتظره خروجی predict_proba برای XGBoost: {xgb_proba_raw.shape}")
                    xgb_proba_corrected = None  # Indicate error

                if xgb_proba_corrected is not None and first_model_output_shape is None:
                    first_model_output_shape = xgb_proba_corrected.shape

            except Exception as e:
                logger.error(f"خطا در دریافت پیش‌بینی از مدل XGBoost: {e}", exc_info=True)
                xgb_proba_corrected = None  # Ensure it's None on error

                # Initialize ensemble_proba only if we have a valid shape
            if first_model_output_shape is None or len(first_model_output_shape) != 2:
                logger.warning("نمی‌توان شکل خروجی ensemble را از هیچ مدلی تعیین کرد.")
                # Return empty results if no model output probabilities
                return np.array([]), {}

                # Initialize ensemble_proba with zeros based on the determined shape
            ensemble_proba = np.zeros((num_samples, first_model_output_shape[1]))

            # Accumulate weighted probabilities
            if 'lstm' in model_probas:
                if model_probas['lstm'].shape == ensemble_proba.shape:
                    ensemble_proba += model_probas['lstm'] * self.models_weights.get('lstm', 0)
                else:
                    logger.warning(
                        f"عدم تطابق شکل LSTM: {model_probas['lstm'].shape} در مقابل {ensemble_proba.shape}. در حال رد کردن.")

            if 'transformer' in model_probas:
                if model_probas['transformer'].shape == ensemble_proba.shape:
                    ensemble_proba += model_probas['transformer'] * self.models_weights.get('transformer', 0)
                else:
                    logger.warning(
                        f"عدم تطابق شکل Transformer: {model_probas['transformer'].shape} در مقابل {ensemble_proba.shape}. در حال رد کردن.")

            if 'cnn' in model_probas:
                if model_probas['cnn'].shape == ensemble_proba.shape:
                    ensemble_proba += model_probas['cnn'] * self.models_weights.get('cnn', 0)
                else:
                    logger.warning(
                        f"عدم تطابق شکل CNN: {model_probas['cnn'].shape} در مقابل {ensemble_proba.shape}. در حال رد کردن.")

            # Add corrected XGBoost probabilities
            if xgb_proba_corrected is not None:
                if xgb_proba_corrected.shape == ensemble_proba.shape:
                    ensemble_proba += xgb_proba_corrected * self.models_weights.get('xgboost', 0)
                else:
                    logger.warning(
                        f"عدم تطابق شکل XGBoost اصلاح شده: {xgb_proba_corrected.shape} در مقابل {ensemble_proba.shape}. در حال رد کردن XGBoost.")

            # Normalize weights if needed
            total_weight_used = sum(w for name, w in self.models_weights.items() if
                                    name in model_probas and (name != 'xgboost' or xgb_proba_corrected is not None))
            if total_weight_used > 1e-6 and abs(total_weight_used - 1.0) > 1e-3:
                logger.debug(f"نرمال‌سازی احتمالات ensemble با مجموع وزن: {total_weight_used:.4f}")
                ensemble_proba /= total_weight_used

            return ensemble_proba, model_probas

    def predict(self, X_seq: np.ndarray, X_flat: np.ndarray = None) -> Tuple[
        np.ndarray, np.ndarray, Dict[str, Any]]:
        """Make ensemble predictions with confidence scores"""
        # Get weighted probabilities
        ensemble_proba, model_probas = self.predict_proba(X_seq, X_flat)

        if ensemble_proba.size == 0:
            return np.array([]), np.array([]), {}

        # Convert to binary predictions
        if ensemble_proba.shape[1] == 1:
            ensemble_pred = (ensemble_proba > 0.5).astype(int).reshape(-1)
            # Confidence calculation
            confidence = np.where(ensemble_proba >= 0.5, ensemble_proba, 1 - ensemble_proba).reshape(-1)
        else:
            ensemble_pred = np.argmax(ensemble_proba, axis=1)
            # Confidence calculation
            confidence = np.max(ensemble_proba, axis=1)

        # Create detailed output
        details = {
            'model_predictions': {},
            'model_confidences': {},
            'model_weights': self.models_weights,
            'model_disagreement': np.zeros(ensemble_proba.shape[0])
        }

        # Calculate individual model predictions and disagreement
        for model_name, proba in model_probas.items():
            if proba.shape[1] == 1:
                pred = (proba > 0.5).astype(int).reshape(-1)
                model_confidence = np.where(proba >= 0.5, proba, 1 - proba).reshape(-1)
            else:
                pred = np.argmax(proba, axis=1)
                model_confidence = np.max(proba, axis=1)

            details['model_predictions'][model_name] = pred
            details['model_confidences'][model_name] = model_confidence

            # Update disagreement measure
            details['model_disagreement'] += (pred != ensemble_pred).astype(float) * self.models_weights.get(
                model_name,
                0)

        # Normalize disagreement to 0-1 range
        total_weight = sum(self.models_weights.values())
        if total_weight > 0:
            details['model_disagreement'] /= total_weight

        # Adjust confidence based on disagreement
        adjusted_confidence = confidence * (1 - details['model_disagreement'])

        logger.debug(f"پیش‌بینی ensemble با {len(model_probas)} مدل تکمیل شد")

        return ensemble_pred, adjusted_confidence, details

    def evaluate_models(self, X_test_seq: np.ndarray, y_test: np.ndarray,
                        X_test_flat: np.ndarray = None) -> Dict[str, Dict[str, float]]:
        """ارزیابی همه مدل‌های موجود در مجموعه"""
        evaluation_results = {}

        # بررسی ناهماهنگی ابعاد و تنظیم آنها
        seq_samples = len(y_test)
        flat_samples = len(X_test_flat) if X_test_flat is not None else 0

        # اگر ابعاد متفاوت باشند، باید آنها را هماهنگ کنیم
        if flat_samples > 0 and flat_samples != seq_samples:
            logger.warning(
                f"ابعاد نمونه‌ها ناسازگار است: توالی={seq_samples}، مسطح={flat_samples}. در حال تنظیم برای تطابق.")

            # انتخاب حداقل اندازه برای هماهنگی
            min_samples = min(seq_samples, flat_samples)

            # کوتاه کردن داده‌ها به همان اندازه
            y_test = y_test[:min_samples]
            X_test_seq = X_test_seq[:min_samples]
            if X_test_flat is not None:
                X_test_flat = X_test_flat[:min_samples]

        if 'lstm' in self.models:
            logger.info("در حال ارزیابی مدل LSTM...")
            lstm_results = self.models['lstm'].evaluate(X_test_seq, y_test)
            evaluation_results['lstm'] = lstm_results

        if 'transformer' in self.models:
            logger.info("در حال ارزیابی مدل Transformer...")
            transformer_results = self.models['transformer'].evaluate(X_test_seq, y_test)
            evaluation_results['transformer'] = transformer_results

        if 'cnn' in self.models:
            logger.info("در حال ارزیابی مدل CNN...")
            cnn_results = self.models['cnn'].evaluate(X_test_seq, y_test)
            evaluation_results['cnn'] = cnn_results

        # اضافه کردن بررسی برای اطمینان از اینکه مدل XGBoost وجود دارد و None نیست
        if 'xgboost' in self.models and self.models['xgboost'] is not None and X_test_flat is not None:
            logger.info("در حال ارزیابی مدل XGBoost...")
            try:
                xgb_preds = self.models['xgboost'].predict(X_test_flat)

                # اطمینان از تطابق ابعاد
                if len(xgb_preds) != len(y_test):
                    logger.warning(
                        f"اندازه پیش‌بینی‌های XGBoost ({len(xgb_preds)}) با اندازه y_test ({len(y_test)}) مطابقت ندارد. در حال تنظیم...")
                    xgb_preds = xgb_preds[:len(y_test)]

                xgb_accuracy = accuracy_score(y_test, xgb_preds)
                xgb_precision = precision_score(y_test, xgb_preds)
                xgb_recall = recall_score(y_test, xgb_preds)

                evaluation_results['xgboost'] = {
                    'accuracy': float(xgb_accuracy),
                    'precision': float(xgb_precision),
                    'recall': float(xgb_recall)
                }
            except Exception as e:
                logger.error(f"خطا در ارزیابی مدل XGBoost: {e}", exc_info=True)
                evaluation_results['xgboost'] = {'error': str(e)}

        # به‌روزرسانی وزن‌ها براساس عملکرد
        self.update_weights_based_on_performance(evaluation_results)

        logger.info(f"ارزیابی {len(evaluation_results)} مدل در ensemble تکمیل شد")
        return evaluation_results

    def get_trading_signals(self, X_seq: np.ndarray, X_flat: np.ndarray = None,
                            recent_data: pd.DataFrame = None,
                            price_data: pd.DataFrame = None) -> List[Dict[str, Any]]:
        """Generate trading signals with confidence, SL/TP levels"""
        try:
            # Make predictions
            predictions, confidences, details = self.predict(X_seq, X_flat)

            if len(predictions) == 0:
                logger.warning("هیچ پیش‌بینی تولید نشد")
                return []

            # Create signals for each positive prediction with sufficient confidence
            signals = []

            for i, (prediction, confidence) in enumerate(zip(predictions, confidences)):
                # Skip if confidence below threshold
                if confidence < self.min_confidence_threshold:
                    continue

                # Basic signal
                signal = {
                    'direction': 'long' if prediction == 1 else 'short',
                    'confidence': float(confidence),
                    'model_agreement': 1.0 - float(details['model_disagreement'][i])
                }

                # Add model details
                signal['model_details'] = {
                    model_name: {
                        'prediction': int(details['model_predictions'][model_name][i]),
                        'confidence': float(details['model_confidences'][model_name][i]),
                        'weight': float(details['model_weights'].get(model_name, 0))
                    }
                    for model_name in details['model_predictions']
                }

                # Add price data if available
                if recent_data is not None and i < len(recent_data) and price_data is not None:
                    # Get current price and ATR
                    current_price = price_data['close'].iloc[-1]

                    try:
                        atr = talib.ATR(
                            price_data['high'].values.astype(np.float64),
                            price_data['low'].values.astype(np.float64),
                            price_data['close'].values.astype(np.float64),
                            timeperiod=14
                        )[-1]
                    except:
                        # Fallback if ATR calculation fails
                        atr = price_data['close'].std() * 0.5

                    # محاسبه حد ضرر و سود با استفاده از مقادیر risk management از ConfigManager
                    # اینجا بهبود داده شده است تا از تنظیمات پیکربندی استفاده کند
                    sl_atr_mult = self.config_manager.get('risk_management.atr_trailing_multiplier', 2.0)
                    default_rr_ratio = self.config_manager.get('risk_management.preferred_risk_reward_ratio', 2.5)
                    tp_atr_mult = sl_atr_mult * default_rr_ratio  # RR ratio

                    if signal['direction'] == 'long':
                        stop_loss = current_price - (atr * sl_atr_mult)
                        take_profit = current_price + (atr * tp_atr_mult)
                    else:
                        stop_loss = current_price + (atr * sl_atr_mult)
                        take_profit = current_price - (atr * tp_atr_mult)

                    signal.update({
                        'entry_price': float(current_price),
                        'stop_loss': float(stop_loss),
                        'take_profit': float(take_profit),
                        'risk_reward_ratio': float(default_rr_ratio),
                        'atr': float(atr)
                    })

                signals.append(signal)

            # Sort by confidence
            signals.sort(key=lambda x: x['confidence'], reverse=True)

            logger.info(
                f"{len(signals)} سیگنال معاملاتی با ضریب اطمینان بالاتر از {self.min_confidence_threshold} تولید شد")
            return signals

        except Exception as e:
            logger.error(f"خطا در تولید سیگنال‌های معاملاتی: {e}", exc_info=True)
            return []

    def save_ensemble(self, base_filepath: str) -> bool:
        """Save ensemble models to disk"""
        try:
            # Create directory if it doesn't exist
            os.makedirs(os.path.dirname(os.path.abspath(base_filepath)), exist_ok=True)

            # Save each model
            saved_models = []

            if 'lstm' in self.models:
                lstm_path = f"{base_filepath}_lstm"
                if self.models['lstm'].save_model(lstm_path):
                    saved_models.append('lstm')

            if 'transformer' in self.models:
                transformer_path = f"{base_filepath}_transformer"
                if self.models['transformer'].save_model(transformer_path):
                    saved_models.append('transformer')

            if 'cnn' in self.models:
                cnn_path = f"{base_filepath}_cnn"
                if self.models['cnn'].save_model(cnn_path):
                    saved_models.append('cnn')

            if 'xgboost' in self.models:
                xgb_path = f"{base_filepath}_xgboost.joblib"
                joblib.dump(self.models['xgboost'], xgb_path)
                saved_models.append('xgboost')

            # Save calibrators if exist
            if self.calibrators:
                calibrators_path = f"{base_filepath}_calibrators.joblib"
                joblib.dump(self.calibrators, calibrators_path)

            # Save feature map if exists
            if self.feature_map:
                feature_map_path = f"{base_filepath}_feature_map.json"
                with open(feature_map_path, 'w') as f:
                    json.dump(self.feature_map, f)

            # Save ensemble metadata
            metadata = {
                'models': saved_models,
                'weights': self.models_weights,
                'metrics': self.models_metrics,
                'ensemble_method': self.ensemble_method,
                'min_confidence_threshold': self.min_confidence_threshold,
                'calibrate_probabilities': self.calibrate_probabilities,
                'has_calibrators': bool(self.calibrators),
                'has_feature_map': bool(self.feature_map)
            }

            with open(f"{base_filepath}_metadata.json", 'w') as f:
                json.dump(metadata, f)

            logger.info(f"مدل‌های ensemble در {base_filepath} ذخیره شدند")
            return True

        except Exception as e:
            logger.error(f"خطا در ذخیره مدل‌های ensemble: {e}", exc_info=True)
            return False

    def load_ensemble(self, base_filepath: str) -> bool:
        """Load ensemble models from disk"""
        try:
            # Load metadata
            with open(f"{base_filepath}_metadata.json", 'r') as f:
                metadata = json.load(f)

            # Update ensemble configuration
            self.models_weights = metadata['weights']
            self.models_metrics = metadata['metrics']
            self.ensemble_method = metadata['ensemble_method']
            self.min_confidence_threshold = metadata['min_confidence_threshold']
            self.calibrate_probabilities = metadata.get('calibrate_probabilities', True)

            # Load models
            for model_name in metadata['models']:
                if model_name == 'lstm':
                    lstm = LSTMModel(self.config_manager)
                    lstm_path = f"{base_filepath}_lstm"
                    if lstm.load_model(lstm_path):
                        self.models['lstm'] = lstm

                elif model_name == 'transformer':
                    transformer = TransformerModel(self.config_manager)
                    transformer_path = f"{base_filepath}_transformer"
                    if transformer.load_model(transformer_path):
                        self.models['transformer'] = transformer

                elif model_name == 'cnn':
                    cnn = CNNModel(self.config_manager)
                    cnn_path = f"{base_filepath}_cnn"
                    if cnn.load_model(cnn_path):
                        self.models['cnn'] = cnn

                elif model_name == 'xgboost':
                    xgb_path = f"{base_filepath}_xgboost.joblib"
                    self.models['xgboost'] = joblib.load(xgb_path)

            # Load calibrators if they exist
            if metadata.get('has_calibrators', False):
                calibrators_path = f"{base_filepath}_calibrators.joblib"
                if os.path.exists(calibrators_path):
                    self.calibrators = joblib.load(calibrators_path)

            # Load feature map if it exists
            if metadata.get('has_feature_map', False):
                feature_map_path = f"{base_filepath}_feature_map.json"
                if os.path.exists(feature_map_path):
                    with open(feature_map_path, 'r') as f:
                        self.feature_map = json.load(f)

            logger.info(f"مدل‌های ensemble از {base_filepath} بارگذاری شدند")
            return True

        except Exception as e:
            logger.error(f"خطا در بارگذاری مدل‌های ensemble: {e}", exc_info=True)
            return False

# -----------------------------------------------------------------------------
# TradingBrainAI: Main Integration Class
# -----------------------------------------------------------------------------

class TradingBrainAI:
    """Integrated trading brain with ML, RL, and adaptive learning"""

    def __init__(self, config: Dict[str, Any] = None, config_path: str = None):
        """Initialize with configuration"""
        # Manage configuration
        self._initialize_config(config, config_path)

        # Initialize components with config manager
        self._initialize_components()

        # Status tracking
        self.models_trained = False
        self.position_sizer_trained = False
        self.last_signal_time = None
        self.last_update_time = None
        self.current_positions = {}

        # Performance tracking
        self.performance_metrics = {
            'trade_history': [],
            'model_accuracy': {},
            'backtest_results': {}
        }

        # Create model directories
        models_dir = self.config_manager.get('models_dir', 'models')
        os.makedirs(models_dir, exist_ok=True)

        logger.info(f"TradingBrainAI با {len(self.config_manager.get_config())} مؤلفه پیکربندی مقداردهی شد")

    def _initialize_config(self, config: Dict[str, Any] = None, config_path: str = None):
        """Initialize configuration management"""
        # Load configuration
        if config is not None:
            loaded_config = config
        elif config_path is not None:
            loaded_config = self._load_config_from_path(config_path)
        else:
            # Default configuration
            loaded_config = self._get_default_config()

        # Create config manager with the loaded configuration
        self.config_manager = ConfigManager(loaded_config)

    def _initialize_components(self):
        """Initialize all components with the config manager"""
        # Initialize main components
        self.feature_engineer = FeatureEngineer(self.config_manager)

        # Create ML models using config manager
        self.lstm_model = None
        self.transformer_model = None
        self.cnn_model = None
        self.xgboost_model = None

        # Initialize ensemble with config manager
        self.ensemble = EnsembleModel(self.config_manager)

        # Initialize other components
        # Adding stubs for now - these will be implemented properly
        self.position_sizer = None
        self.online_learning = None
        self.hyperopt = None

    def _load_config_from_path(self, config_path: str) -> Dict[str, Any]:
        """
        Load configuration from file with proper error handling

        Args:
            config_path: Path to configuration file

        Returns:
            Configuration dictionary
        """
        try:
            # Check if file exists
            if not os.path.exists(config_path):
                raise FileNotFoundError(f"فایل پیکربندی یافت نشد: {config_path}")

            with open(config_path, 'r', encoding='utf-8') as f:
                if config_path.endswith(('.yaml', '.yml')):
                    import yaml
                    config = yaml.safe_load(f)
                elif config_path.endswith('.json'):
                    import json
                    config = json.load(f)
                else:
                    raise ValueError(f"فرمت فایل پیکربندی پشتیبانی نمی‌شود: {config_path}")

            # Basic configuration validation
            required_sections = ['trading_brain_ai', 'lstm_model', 'transformer_model', 'ensemble']
            missing_sections = [section for section in required_sections if section not in config]

            if missing_sections:
                logger.warning(f"بخش‌های تنظیمات مورد نیاز یافت نشد: {', '.join(missing_sections)}")
                # Don't raise exception, just fill in defaults later

            logger.info(f"پیکربندی با موفقیت از {config_path} بارگذاری شد")
            return config

        except FileNotFoundError:
            logger.critical(f"خطای بحرانی: فایل پیکربندی در {config_path} یافت نشد.")
            raise SystemExit(f"فایل پیکربندی یافت نشد: {config_path}")
        except (json.JSONDecodeError, yaml.YAMLError, ValueError) as e:
            logger.critical(f"خطای بحرانی: فایل پیکربندی {config_path} نامعتبر یا ناقص است: {e}.")
            raise SystemExit(f"فایل پیکربندی نامعتبر: {e}")
        except Exception as e:
            logger.critical(f"خطای بحرانی غیرمنتظره در بارگذاری پیکربندی {config_path}: {e}.", exc_info=True)
            raise SystemExit(f"خطا در بارگذاری پیکربندی: {e}")

    def _get_default_config(self) -> Dict[str, Any]:
        """
        Get default configuration settings

        Returns:
            Default configuration dictionary
        """
        return {
            'trading_brain_ai': {
                'feature_engineering': {
                    'use_technical_indicators': True,
                    'use_price_patterns': True,
                    'use_volatility_features': True,
                    'use_volume_features': True,
                    'lookback_window': 20,
                    'target_horizons': [1, 5, 10],
                    'scaler_type': 'robust',
                    'feature_selection': True,
                    'feature_selection_method': 'importance',
                    'importance_threshold': 0.01
                },
                'signal_generation': {
                    'minimum_signal_score': 180.0,
                    'pattern_scores': {}
                },
                'risk_management': {
                    'min_risk_reward_ratio': 1.8,
                    'preferred_risk_reward_ratio': 2.5,
                    'default_stop_loss_percent': 1.5,
                    'max_risk_per_trade_percent': 1.5
                }
            },
            'lstm_model': {
                'units': [64, 32],
                'dropout': 0.2,
                'learning_rate': 0.001,
                'batch_size': 32,
                'epochs': 100,
                'patience': 10,
                'bidirectional': True
            },
            'transformer_model': {
                'embedding_dim': 64,
                'num_heads': 4,
                'ff_dim': 64,
                'num_transformer_blocks': 2,
                'mlp_units': [32],
                'dropout_rate': 0.25,
                'learning_rate': 0.001,
                'batch_size': 32,
                'epochs': 100,
                'patience': 10
            },
            'cnn_model': {
                'filters': [64, 128, 128],
                'kernel_sizes': [3, 3, 3],
                'pool_sizes': [2, 2, 2],
                'dense_units': [64, 32],
                'dropout_rate': 0.3,
                'learning_rate': 0.001,
                'batch_size': 32,
                'epochs': 100,
                'patience': 10
            },
            'ensemble': {
                'weights': {
                    'lstm': 0.35,
                    'transformer': 0.35,
                    'cnn': 0.15,
                    'xgboost': 0.15
                },
                'calibrate_probabilities': True,
                'min_confidence_threshold': 0.65
            },
            'position_sizer': {
                'model_type': 'ppo',
                'train_steps': 50000,
                'learning_rate': 3e-4,
                'ent_coef': 0.01,
                'batch_size': 64,
                'gamma': 0.99
            },
            'online_learning': {
                'enabled': True,
                'adaptation_frequency': 'daily',
                'min_samples_for_update': 100,
                'learning_rate': 0.01
            },
            'hyperparameter_optimization': {
                'n_trials': 50,
                'timeout': 7200,
                'optimization_metric': 'accuracy',
                'optimization_direction': 'maximize'
            },
            'models_dir': 'models',
            'data_dir': 'data'
        }

    def update_config(self, section: str, values: Dict[str, Any]) -> bool:
        """
        Update configuration settings at runtime

        Args:
            section: Section to update (can use dot notation for nested sections)
            values: Dictionary of values to update

        Returns:
            True if update was successful, False otherwise
        """
        try:
            # Use config manager to update settings
            result = self.config_manager.update_section(section, values)

            if result:
                logger.info(f"پیکربندی بخش '{section}' با موفقیت به‌روزرسانی شد")
                # The update listeners will automatically propagate changes to components
            else:
                logger.warning(f"به‌روزرسانی بخش '{section}' ناموفق بود")

            return result

        except Exception as e:
            logger.error(f"خطا در به‌روزرسانی پیکربندی: {e}", exc_info=True)
            return False

    def prepare_data(self, data: pd.DataFrame, test_size: float = 0.2, return_dict: bool = False) -> Optional[
        Union[Dict[str, Any], Tuple]]:
        """Prepare data for training and testing with enhanced logging"""
        try:
            logger.info(f"آماده‌سازی داده با {len(data)} نمونه...")

            # Create features
            logger.debug("فراخوانی feature_engineer.create_features...")
            features_df = self.feature_engineer.create_features(data)
            logger.debug("بازگشت از feature_engineer.create_features.")

            # --- بررسی دقیق نتیجه create_features ---
            if features_df is None:
                logger.error("TradingBrainAI.prepare_data: feature_engineer.create_features مقدار None برگرداند.")
                # لاگ پیام خطای مشخص
                logger.error("خطا در ایجاد ویژگی‌ها")
                return None
            elif features_df.empty:
                logger.error(
                    "TradingBrainAI.prepare_data: feature_engineer.create_features یک DataFrame خالی برگرداند (احتمالاً به دلیل dropna).")
                # لاگ پیام خطای احتمالی دیگر
                logger.error("خطا در ایجاد ویژگی‌ها - نتیجه خالی است")
                logger.error("خطا در ایجاد ویژگی‌ها")
                return None
            else:
                logger.info(
                    f"ایجاد ویژگی‌ها موفقیت‌آمیز به نظر می‌رسد، شکل DataFrame بعد از create_features: {features_df.shape}")
            # --- پایان بررسی دقیق ---

            # بررسی کافی بودن داده‌ها بعد از ایجاد ویژگی و dropna احتمالی
            logger.debug("بررسی کافی بودن داده‌ها بعد از ایجاد ویژگی...")
            # تنظیم حداقل مورد نیاز بر اساس lookback + target horizon + buffer
            lookback = self.config_manager.get('trading_brain_ai.feature_engineering.lookback_window', 20)
            target_horizons = self.config_manager.get('trading_brain_ai.feature_engineering.target_horizons', [1])
            max_target_horizon = max(target_horizons)
            min_required_after_features = lookback + max_target_horizon + 50  # Buffer برای محاسبات

            if len(features_df) < min_required_after_features:
                logger.error(
                    f"تعداد ردیف‌های داده بعد از ایجاد ویژگی‌ها کافی نیست ({len(features_df)} ردیف). حداقل {min_required_after_features} ردیف برای مراحل بعدی ML مورد نیاز است.")
                logger.error("خطا در ایجاد ویژگی‌ها")
                return None

            # آماده‌سازی داده برای ML
            logger.debug("فراخوانی feature_engineer.prepare_data_for_ml...")
            target_horizon_for_ml = target_horizons[0] if target_horizons else 1  # استفاده از اولین افق

            # دریافت تنظیمات مقیاس‌دهی و انتخاب ویژگی از مدیریت پیکربندی
            scale_features = self.config_manager.get('trading_brain_ai.feature_engineering.scaler_type',
                                                     'robust') != 'none'
            feature_selection = self.config_manager.get('trading_brain_ai.feature_engineering.feature_selection', True)

            ml_data = self.feature_engineer.prepare_data_for_ml(
                features_df,
                target_horizon=target_horizon_for_ml,
                train_ratio=1.0 - test_size,
                scale_features=scale_features,
                feature_selection=feature_selection
            )
            logger.debug("بازگشت از feature_engineer.prepare_data_for_ml.")

            if ml_data is None:
                logger.error("خطا در آماده‌سازی داده ML (ml_data مقدار None است)")
                logger.error("خطا در ایجاد ویژگی‌ها")
                return None
            else:
                # بررسی صحت شکل داده‌ها
                if 'X_train' not in ml_data or 'y_train' not in ml_data or \
                        'X_test' not in ml_data or 'y_test' not in ml_data:
                    logger.error("دیکشنری داده ML فاقد کلیدهای مورد نیاز است (X_train, y_train و غیره).")
                    logger.error("خطا در ایجاد ویژگی‌ها")
                    return None
                logger.info(
                    f"آماده‌سازی داده ML موفقیت‌آمیز بود. شکل آموزش: {ml_data['X_train'].shape}، شکل آزمایش: {ml_data['X_test'].shape}")

            # آماده‌سازی داده توالی برای مدل‌های RNN
            logger.debug("فراخوانی feature_engineer.prepare_sequence_data...")
            seq_data = self.feature_engineer.prepare_sequence_data(
                features_df,  # ارسال DataFrame ویژگی‌ها
                target_horizon=target_horizon_for_ml,  # استفاده از همان افق هدف
                sequence_length=self.config_manager.get('trading_brain_ai.feature_engineering.lookback_window', 20),
                # lookback سازگار
                train_ratio=1.0 - test_size
            )
            logger.debug("بازگشت از feature_engineer.prepare_sequence_data.")

            if seq_data is None:
                logger.error("خطا در آماده‌سازی داده توالی (seq_data مقدار None است)")
                logger.error("خطا در ایجاد ویژگی‌ها")
                return None
            else:
                # بررسی صحت شکل داده‌ها
                if 'X_train' not in seq_data or 'y_train' not in seq_data or \
                        'X_test' not in seq_data or 'y_test' not in seq_data:
                    logger.error("دیکشنری داده توالی فاقد کلیدهای مورد نیاز است (X_train, y_train و غیره).")
                    logger.error("خطا در ایجاد ویژگی‌ها")
                    return None
                logger.info(
                    f"آماده‌سازی داده توالی موفقیت‌آمیز بود. شکل آموزش: {seq_data['X_train'].shape}، شکل آزمایش: {seq_data['X_test'].shape}")

            logger.info(f"آماده‌سازی داده کامل شد: {ml_data['X_train'].shape[0]} نمونه آموزش ML، "
                        f"{seq_data['X_train'].shape[0]} نمونه آموزش توالی.")

            if return_dict:
                return {
                    'features_df': features_df,  # بازگرداندن DataFrame ویژگی‌ها نیز
                    'ml_data': ml_data,
                    'seq_data': seq_data
                }

            # بازگرداندن به صورت tuple اگر return_dict نیست
            return ml_data, seq_data

        except Exception as e:
            logger.error(f"خطای بحرانی در TradingBrainAI.prepare_data: {str(e)}", exc_info=True)
            logger.error("خطا در ایجاد ویژگی‌ها")
            return None

    async def train_models(self, data: pd.DataFrame = None, ml_data: Dict[str, Any] = None,
                           seq_data: Dict[str, Any] = None, optimize: bool = False) -> Dict[str, Any]:
        """آموزش مدل‌های ML برای پیش‌بینی سیگنال‌های معاملاتی با جلوگیری از فراخوانی دوباره prepare_data"""

        # بهبود منطق بررسی داده‌ها با لاگ دقیق‌تر
        if data is not None and ml_data is None and seq_data is None:
            logger.debug("فراخوانی prepare_data برای آماده‌سازی داده‌های ورودی")
            prepared_data = self.prepare_data(data, return_dict=True)
            if prepared_data is None:
                logger.error("آماده‌سازی داده با شکست مواجه شد")
                return {'status': 'error', 'message': 'آماده‌سازی داده با شکست مواجه شد'}

            ml_data = prepared_data['ml_data']
            seq_data = prepared_data['seq_data']
        else:
            logger.debug("استفاده از داده‌های ml_data و seq_data ارائه شده بدون فراخوانی prepare_data")

        # بررسی کامل‌تر داده‌ها
        if ml_data is None:
            logger.error("داده ml_data ارائه نشده یا تهی است")
            return {'status': 'error', 'message': 'داده ml_data ارائه نشده است'}
        if seq_data is None:
            logger.error("داده seq_data ارائه نشده یا تهی است")
            return {'status': 'error', 'message': 'داده seq_data ارائه نشده است'}

        try:
            logger.debug("شروع فرآیند آموزش مدل با داده‌های معتبر")
            training_results = {}

            # هماهنگ‌سازی تعداد نمونه‌های آزمایشی
            ml_test_samples = len(ml_data['X_test'])
            seq_test_samples = len(seq_data['X_test'])

            if ml_test_samples != seq_test_samples:
                logger.warning(
                    f"ابعاد نمونه‌های آزمایش ناسازگار است: ml_data={ml_test_samples}، seq_data={seq_test_samples}. در حال تنظیم...")

                # انتخاب حداقل اندازه
                min_test_samples = min(ml_test_samples, seq_test_samples)

                # تنظیم داده‌های آزمایشی به همان اندازه
                ml_data['X_test'] = ml_data['X_test'][:min_test_samples]
                ml_data['y_test'] = ml_data['y_test'][:min_test_samples]

                seq_data['X_test'] = seq_data['X_test'][:min_test_samples]
                seq_data['y_test'] = seq_data['y_test'][:min_test_samples]

                logger.info(f"مجموعه‌های آزمایش به {min_test_samples} نمونه تنظیم شدند")

            # بهینه‌سازی هایپرپارامترها در صورت درخواست
            if optimize:
                logger.info("شروع بهینه‌سازی هایپرپارامترها...")
                # ... [کد بهینه‌سازی موجود] ...
                logger.info("بهینه‌سازی هایپرپارامترها کامل شد")

            # آماده‌سازی و آموزش مدل‌ها
            logger.info("آموزش مدل LSTM...")
            self.lstm_model = LSTMModel(self.config_manager)
            self.lstm_model.build_model(seq_data['X_train'].shape[1:])
            lstm_results = self.lstm_model.train(
                seq_data['X_train'], seq_data['y_train'],
                seq_data['X_test'], seq_data['y_test']
            )
            training_results['lstm'] = lstm_results

            # آموزش مدل Transformer
            logger.info("آموزش مدل Transformer...")
            self.transformer_model = TransformerModel(self.config_manager)
            self.transformer_model.build_model(seq_data['X_train'].shape[1:])
            transformer_results = self.transformer_model.train(
                seq_data['X_train'], seq_data['y_train'],
                seq_data['X_test'], seq_data['y_test']
            )
            training_results['transformer'] = transformer_results

            # آموزش مدل CNN
            logger.info("آموزش مدل CNN...")
            self.cnn_model = CNNModel(self.config_manager)
            self.cnn_model.build_model(seq_data['X_train'].shape[1:])
            cnn_results = self.cnn_model.train(
                seq_data['X_train'], seq_data['y_train'],
                seq_data['X_test'], seq_data['y_test']
            )
            training_results['cnn'] = cnn_results

            # آموزش مدل XGBoost اگر قبلاً بهینه‌سازی نشده باشد
            if self.xgboost_model is None:
                logger.info("آموزش مدل XGBoost...")
                try:
                    from xgboost import XGBClassifier
                    import xgboost as xgb

                    # تنظیم پارامترهای XGBoost از مدیریت پیکربندی
                    xgb_config = self.config_manager.get('xgboost_model', {})
                    n_estimators = xgb_config.get('n_estimators', 100)
                    max_depth = xgb_config.get('max_depth', 5)
                    learning_rate = xgb_config.get('learning_rate', 0.1)
                    subsample = xgb_config.get('subsample', 0.8)
                    colsample_bytree = xgb_config.get('colsample_bytree', 0.8)

                    # ایجاد و آموزش مدل XGBoost - سازگار با نسخه‌های جدید
                    self.xgboost_model = XGBClassifier(
                        n_estimators=n_estimators,
                        max_depth=max_depth,
                        learning_rate=learning_rate,
                        subsample=subsample,
                        colsample_bytree=colsample_bytree,
                        use_label_encoder=False,
                        eval_metric='logloss',
                        random_state=42
                    )

                    # آموزش مدل با داده‌های مسطح (ML) - سازگار با نسخه‌های جدید
                    # به جای استفاده از early_stopping_rounds در fit
                    eval_set = [(ml_data['X_test'], ml_data['y_test'])]

                    # سعی با استفاده از روش جدید (نسخه‌های جدید XGBoost)
                    try:
                        # روش جدید: با استفاده از callbacks
                        logger.debug("تلاش برای آموزش XGBoost با روش callbacks")
                        self.xgboost_model.fit(
                            ml_data['X_train'], ml_data['y_train'],
                            eval_set=eval_set,
                            callbacks=[xgb.callback.EarlyStopping(rounds=10)]
                        )
                        logger.debug("آموزش XGBoost با روش callbacks موفقیت‌آمیز بود")
                    except (TypeError, AttributeError) as e1:
                        logger.debug(f"آموزش XGBoost با روش callbacks با خطا مواجه شد: {e1}")
                        # اگر روش بالا کار نکرد، سعی با روش میانی
                        try:
                            # روش میانی: با استفاده از early_stopping_rounds
                            logger.debug("تلاش برای آموزش XGBoost با روش early_stopping_rounds")
                            self.xgboost_model.fit(
                                ml_data['X_train'], ml_data['y_train'],
                                eval_set=eval_set,
                                early_stopping_rounds=10,
                                verbose=False
                            )
                            logger.debug("آموزش XGBoost با روش early_stopping_rounds موفقیت‌آمیز بود")
                        except TypeError as e2:
                            logger.debug(f"آموزش XGBoost با روش early_stopping_rounds با خطا مواجه شد: {e2}")
                            # اگر روش میانی هم کار نکرد، بدون توقف زودهنگام آموزش دهیم
                            logger.debug("تلاش برای آموزش XGBoost بدون early stopping")
                            self.xgboost_model.fit(
                                ml_data['X_train'], ml_data['y_train'],
                                eval_set=eval_set,
                                verbose=False
                            )
                            logger.debug("آموزش XGBoost بدون early stopping موفقیت‌آمیز بود")

                    # نتایج آموزش
                    xgb_train_score = self.xgboost_model.score(ml_data['X_train'], ml_data['y_train'])
                    xgb_test_score = self.xgboost_model.score(ml_data['X_test'], ml_data['y_test'])

                    training_results['xgboost'] = {
                        'train_accuracy': float(xgb_train_score),
                        'val_accuracy': float(xgb_test_score)
                    }

                    logger.info(f"آموزش مدل XGBoost با دقت اعتبارسنجی {xgb_test_score:.4f} کامل شد")

                    # ذخیره نام ویژگی‌ها برای استفاده در آینده
                    if 'feature_names' in ml_data:
                        self.feature_names = ml_data['feature_names']

                except Exception as e:
                    logger.error(f"خطا در آموزش مدل XGBoost: {e}", exc_info=True)
                    self.xgboost_model = None  # مطمئن شویم که به درستی مقدار None را داراست
                    training_results['xgboost'] = {'error': str(e)}

            # افزودن مدل‌ها به مجموعه
            self.ensemble = EnsembleModel(self.config_manager)
            self.ensemble.add_model('lstm', self.lstm_model)
            self.ensemble.add_model('transformer', self.transformer_model)
            self.ensemble.add_model('cnn', self.cnn_model)

            # فقط در صورتی که XGBoost با موفقیت آموزش دیده باشد، آن را به مجموعه اضافه کنیم
            if self.xgboost_model is not None:
                self.ensemble.add_model('xgboost', self.xgboost_model)

            # تنظیم نگاشت ویژگی‌ها برای استفاده در XGBoost
            if hasattr(self, 'feature_names') and self.feature_names:
                self.ensemble.set_feature_map(self.feature_names)

            # ارزیابی مدل‌ها
            logger.info("ارزیابی مجموعه مدل‌ها...")
            evaluation_results = self.ensemble.evaluate_models(
                seq_data['X_test'], seq_data['y_test'],
                ml_data['X_test']
            )
            training_results['ensemble_evaluation'] = evaluation_results

            # کالیبراسیون احتمال
            if self.ensemble.calibrate_probabilities:
                logger.info("کالیبراسیون احتمالات مدل‌ها...")
                self.ensemble.calibrate_models(
                    seq_data['X_test'], seq_data['y_test'],
                    ml_data['X_test']
                )
                training_results['probability_calibration'] = {
                    'status': 'completed',
                    'models_calibrated': list(self.ensemble.calibrators.keys())
                }

            # به‌روزرسانی متریک‌های عملکرد
            self.performance_metrics['model_accuracy'] = evaluation_results

            # به‌روزرسانی وضعیت
            self.models_trained = True

            logger.info(f"آموزش مدل با ارزیابی ensemble کامل شد: {evaluation_results}")

            return {
                'status': 'success',
                'training_results': training_results
            }

        except Exception as e:
            logger.error(f"خطا در آموزش مدل‌ها: {e}", exc_info=True)
            return {
                'status': 'error',
                'message': str(e)
            }

    def predict_signal(self, timeframes_data: Dict[str, pd.DataFrame]) -> Dict[str, Any]:
        """
        Generate trading signal based on ML/AI prediction for the latest data point.
        Now accepts multi-timeframe data.

        Args:
            timeframes_data: Dictionary of multi-timeframe data {timeframe: df}.

        Returns:
            A dictionary containing the prediction results, confidence, etc.
        """
        if not self.models_trained:
            return {'status': 'error', 'message': 'مدل‌ها آموزش داده نشده‌اند'}

        try:
            # --- 1. انتخاب تایم‌فریم اصلی و دریافت آخرین داده‌ها ---
            # تعیین تایم‌فریم اصلی (مثلاً، آن با بالاترین وزن یا کوتاه‌ترین زمان)
            # برای سادگی، از اولین تایم‌فریم ارائه شده که داده معتبر دارد استفاده می‌کنیم
            primary_tf = None
            primary_df = None

            # دریافت لیست تایم‌فریم‌ها از تنظیمات
            timeframe_list = self.config_manager.get('data_fetching.timeframes', [])

            # مرتب‌سازی تایم‌فریم‌ها بر اساس ترتیب پیکربندی شده
            valid_tfs = sorted(
                timeframes_data.keys(),
                key=lambda tf: timeframe_list.index(tf) if tf in timeframe_list else len(timeframe_list)
            )

            for tf in valid_tfs:
                if tf in timeframes_data and timeframes_data[tf] is not None and not timeframes_data[tf].empty:
                    primary_tf = tf
                    primary_df = timeframes_data[tf]
                    break

            if primary_df is None or primary_df.empty:
                return {'status': 'error', 'message': 'هیچ داده تایم‌فریم معتبری برای پیش‌بینی ارائه نشده است.'}

            # اطمینان از کافی بودن داده‌ها برای استخراج ویژگی
            lookback_window = self.config_manager.get('trading_brain_ai.feature_engineering.lookback_window', 20)

            if len(primary_df) < lookback_window:
                return {'status': 'error',
                        'message': f'طول داده‌ها کافی نیست ({len(primary_df)}) برای استخراج ویژگی (حداقل {lookback_window} مورد نیاز است).'}

            # --- 2. آماده‌سازی ویژگی‌ها برای آخرین نقطه داده ---
            features_df = self.feature_engineer.create_features(primary_df)

            if features_df is None or features_df.empty:
                return {'status': 'error', 'message': 'استخراج ویژگی‌ها برای پیش‌بینی با شکست مواجه شد.'}

            # دریافت آخرین ویژگی‌ها (آخرین ردیف)
            # حذف ستون‌های هدف چون برای پیش‌بینی در دسترس نیستند
            latest_features = features_df.iloc[-1:].drop(
                [col for col in features_df.columns if col.startswith('target_')], axis=1)

            if latest_features.empty:
                return {'status': 'error', 'message': 'هیچ ویژگی آخری بعد از پردازش در دسترس نیست.'}

            # --- 3. آماده‌سازی داده در قالب مورد انتظار مدل Ensemble ---
            # مدل Ensemble داده‌های توالی (برای RNN‌ها) و احتمالاً داده مسطح (برای مدل‌های درختی) را انتظار دارد

            # آماده‌سازی داده توالی برای RNN‌ها (LSTM، Transformer، CNN)
            sequence_length = self.config_manager.get('trading_brain_ai.feature_engineering.lookback_window', 20)

            # دریافت آخرین `sequence_length` بردار ویژگی
            seq_features_df = features_df.iloc[-sequence_length:].drop(
                [col for col in features_df.columns if col.startswith('target_')], axis=1)

            # اطمینان از اینکه به اندازه کافی نقطه داده برای توالی داریم
            if len(seq_features_df) < sequence_length:
                # پر کردن با صفر اگر تاریخچه کافی نیست
                padding = np.zeros((sequence_length - len(seq_features_df), seq_features_df.shape[1]))
                seq_features_array = np.vstack([padding, seq_features_df.values])
                symbol = primary_df.attrs.get('symbol', 'unknown')
                logger.warning(
                    f"پر کردن داده توالی برای {symbol} {primary_tf}: تاریخچه کافی برای lookback {sequence_length} وجود ندارد.")
            else:
                seq_features_array = seq_features_df.values

            # تغییر شکل برای مدل‌های توالی (نمونه‌ها، گام‌های زمانی، ویژگی‌ها)
            X_seq = seq_features_array.reshape(1, sequence_length, seq_features_array.shape[-1])

            # آماده‌سازی داده مسطح برای مدل‌هایی مانند XGBoost که به ویژگی‌های یکسان استفاده شده در آموزش نیاز دارند
            if hasattr(self.feature_engineer, 'selected_features') and self.feature_engineer.selected_features:
                # دریافت لیست ویژگی‌های انتخاب شده از آموزش
                selected_features = list(self.feature_engineer.selected_features)

                # بررسی اینکه آیا این ویژگی‌ها در ویژگی‌های فعلی ما وجود دارند
                available_features = latest_features.columns.tolist()

                # یافتن ویژگی‌های مشترک بین selected_features و available_features
                common_features = [f for f in selected_features if f in available_features]

                if not common_features:
                    logger.warning(f"هیچ یک از {len(selected_features)} ویژگی انتخاب شده در داده فعلی یافت نشد.")
                    # استفاده از همه ویژگی‌ها به عنوان پشتیبان
                    X_flat = latest_features.values
                else:
                    logger.debug(f"استفاده از {len(common_features)}/{len(selected_features)} ویژگی برای XGBoost.")
                    # انتخاب فقط ویژگی‌های مورد نیاز
                    X_flat = latest_features[common_features].values

                    # بررسی اینکه آیا نیاز است ستون‌های ساختگی را برای تطابق با ابعاد آموزش اضافه کنیم
                    if len(common_features) < len(selected_features):
                        # این یک روش پشتیبان است - بهتر است یک نگاشت ویژگی پیاده‌سازی شود
                        missing_count = len(selected_features) - len(common_features)
                        dummy_features = np.zeros((X_flat.shape[0], missing_count))
                        X_flat = np.hstack([X_flat, dummy_features])
                        logger.warning(f"افزودن {missing_count} ویژگی ساختگی برای تطابق با ابعاد آموزش.")
            else:
                # اطلاعات انتخاب ویژگی در دسترس نیست، از همه ویژگی‌ها استفاده کن اما هشدار بده
                logger.warning("اطلاعات انتخاب ویژگی در دسترس نیست. استفاده از همه ویژگی‌ها.")
                X_flat = latest_features.values

            # --- 4. دریافت پیش‌بینی و اعتماد از مدل Ensemble ---
            predictions, confidences, details = self.ensemble.predict(X_seq, X_flat)

            if len(predictions) == 0:
                return {'status': 'no_prediction', 'message': 'مدل Ensemble پیش‌بینی تولید نکرد.'}

            # استخراج پیش‌بینی برای آخرین نقطه داده
            latest_prediction = predictions[-1]
            latest_confidence = confidences[-1]
            latest_details = {k: v[-1] if isinstance(v, np.ndarray) else v for k, v in
                              details.items()}  # دریافت جزئیات برای آخرین نمونه

            # --- 5. دریافت توصیه تعیین اندازه پوزیشن از Position Sizer (در صورت فعال بودن) ---
            position_sizing_rec = None
            if self.position_sizer_trained:
                try:
                    position_sizing_info = self.position_sizer.predict(seq_features_array)  # ارسال آرایه توالی

                    if position_sizing_info:
                        position_sizing_rec = position_sizing_info
                        latest_details['position_sizing_recommendation'] = position_sizing_rec

                except Exception as ps_error:
                    logger.error(f"خطا در دریافت توصیه تعیین اندازه پوزیشن: {ps_error}", exc_info=True)
                    latest_details['position_sizing_recommendation_error'] = str(ps_error)

            # --- 6. قالب‌بندی نتایج پیش‌بینی به عنوان یک سیگنال ساده‌شده ---
            predicted_direction = 'long' if latest_prediction == 1 else 'short'

            # محاسبه ورودی، SL، TP و RR پیش‌بینی شده
            latest_price = primary_df['close'].iloc[-1]

            # دریافت تنظیمات مدیریت ریسک از مدیریت پیکربندی
            sl_atr_mult = self.config_manager.get('risk_management.atr_trailing_multiplier', 2.0)
            default_rr_ratio = self.config_manager.get('risk_management.preferred_risk_reward_ratio', 2.5)

            # محاسبه حد ضرر و سود
            try:
                # محاسبه ATR
                atr = talib.ATR(
                    primary_df['high'].values.astype(np.float64),
                    primary_df['low'].values.astype(np.float64),
                    primary_df['close'].values.astype(np.float64),
                    timeperiod=14
                )[-1]
            except:
                # جایگزین اگر محاسبه ATR شکست بخورد
                atr = primary_df['close'].std() * 0.5

            tp_atr_mult = sl_atr_mult * default_rr_ratio

            if predicted_direction == 'long':
                stop_loss = latest_price - (atr * sl_atr_mult)
                take_profit = latest_price + (atr * tp_atr_mult)
            else:
                stop_loss = latest_price + (atr * sl_atr_mult)
                take_profit = latest_price - (atr * tp_atr_mult)

            # ساخت نتیجه پیش‌بینی
            prediction_result = {
                'status': 'success',
                'symbol': primary_df.attrs.get('symbol', 'unknown'),  # دریافت نماد از attrs
                'timeframe': primary_tf,
                'direction': predicted_direction,
                'confidence': float(latest_confidence),
                'entry_price': float(latest_price),
                'stop_loss': float(stop_loss),
                'take_profit': float(take_profit),
                'risk_reward_ratio': float(default_rr_ratio),
                'atr': float(atr),
                'model_agreement': float(latest_details.get('model_agreement', 1.0)),
                'model_details': latest_details,  # شامل پیش‌بینی‌های دقیق از مدل‌ها
                'position_sizing': position_sizing_rec,  # شامل توصیه اندازه پوزیشن
                'timestamp': datetime.now().isoformat()  # اضافه کردن زمان برای ردیابی
            }

            # به‌روزرسانی زمان آخرین سیگنال
            self.last_signal_time = datetime.now()

            return {'status': 'success', 'signal': prediction_result}

        except Exception as e:
            symbol = "unknown"
            if primary_df is not None and hasattr(primary_df, 'attrs'):
                symbol = primary_df.attrs.get('symbol', 'unknown')
            logger.error(f"خطا در تولید پیش‌بینی هوش مصنوعی برای {symbol}: {e}", exc_info=True)
            return {
                'status': 'error',
                'message': f'خطا در تولید پیش‌بینی هوش مصنوعی: {str(e)}',
                'error_details': traceback.format_exc()
            }

    def register_trade_result(self, symbol: str, entry_time: str, exit_time: str,
                              direction: str, entry_price: float, exit_price: float,
                              position_size: float = 1.0, profit_pct: float = None,
                              profit_amount: float = None) -> Dict[str, Any]:
        """ثبت نتیجه معامله برای یادگیری مداوم"""
        try:
            # محاسبه سود اگر ارائه نشده است
            if profit_pct is None:
                if direction == 'long':
                    profit_pct = (exit_price / entry_price - 1.0) * 100
                else:  # short
                    profit_pct = (entry_price / exit_price - 1.0) * 100

            # ایجاد نتیجه معامله
            trade_result = {
                'symbol': symbol,
                'entry_time': entry_time,
                'exit_time': exit_time,
                'direction': direction,
                'entry_price': entry_price,
                'exit_price': exit_price,
                'position_size': position_size,
                'profit_pct': profit_pct,
                'profit_amount': profit_amount
            }

            # افزودن به ردیابی عملکرد
            self.performance_metrics['trade_history'].append(trade_result)

            # افزودن به بافر یادگیری آنلاین
            if hasattr(self, 'online_learning') and self.online_learning and hasattr(self.online_learning,
                                                                                     'enabled') and self.online_learning.enabled:
                # تنظیم وزن بر اساس سود (وزن بالاتر برای سودها/ضررهای بزرگتر)
                weight = min(3.0, max(0.5, abs(profit_pct) / 2))

                # افزودن به سیستم یادگیری آنلاین
                self.online_learning.add_trade(trade_result, weight=weight)

                # بررسی اینکه آیا مدل‌ها باید به‌روزرسانی شوند
                if hasattr(self.online_learning,
                           'should_adapt_models') and self.online_learning.should_adapt_models():
                    self._update_models()

            logger.info(f"نتیجه معامله برای {symbol} ثبت شد: {direction} با {profit_pct:.2f}% سود")

            return {
                'status': 'success',
                'trade_id': len(self.performance_metrics['trade_history'])
            }

        except Exception as e:
            logger.error(f"خطا در ثبت نتیجه معامله: {e}", exc_info=True)
            return {
                'status': 'error',
                'message': str(e)
            }

    def _update_models(self) -> Dict[str, Any]:
        """به‌روزرسانی مدل‌ها با یادگیری آنلاین"""
        update_results = {}

        try:
            # بررسی و اطمینان از اینکه سیستم یادگیری آنلاین آماده است
            if not hasattr(self, 'online_learning') or not self.online_learning:
                logger.warning("سیستم یادگیری آنلاین هنوز مقداردهی نشده است")
                return {'status': 'error', 'message': 'سیستم یادگیری آنلاین آماده نیست'}

            # به‌روزرسانی مدل‌های ensemble
            if hasattr(self.online_learning, 'update_ensemble') and self.online_learning.update_ensemble:
                logger.info("به‌روزرسانی مدل‌های ensemble با یادگیری آنلاین...")
                ensemble_update = self.online_learning.update_ensemble_models(self.ensemble)
                update_results['ensemble'] = ensemble_update

            # به‌روزرسانی مدل position sizer
            if hasattr(self.online_learning,
                       'update_position_sizer') and self.online_learning.update_position_sizer and self.position_sizer_trained:
                logger.info("به‌روزرسانی مدل position sizer با یادگیری آنلاین...")
                # ایجاد یک محیط با داده‌های اخیر (ساده‌سازی شده در اینجا)
                env = self.position_sizer.env.envs[0] if hasattr(self.position_sizer,
                                                                 'env') and self.position_sizer.env else None

                if env:
                    position_sizer_update = self.online_learning.update_position_sizer_model(self.position_sizer,
                                                                                             env)
                    update_results['position_sizer'] = position_sizer_update

            self.last_update_time = datetime.now()

            logger.info(f"به‌روزرسانی مدل‌ها با یادگیری آنلاین تکمیل شد")

            return {
                'status': 'success',
                'update_results': update_results,
                'timestamp': self.last_update_time.isoformat()
            }

        except Exception as e:
            logger.error(f"خطا در به‌روزرسانی مدل‌ها: {e}", exc_info=True)
            return {
                'status': 'error',
                'message': str(e)
            }

    def save_state(self, base_filepath: str) -> Dict[str, bool]:
        """ذخیره وضعیت کامل trading brain"""
        try:
            # ایجاد دایرکتوری‌ها در صورت عدم وجود
            os.makedirs(os.path.dirname(os.path.abspath(base_filepath)), exist_ok=True)

            # ذخیره مدل‌ها
            save_results = {}

            if self.models_trained:
                # ذخیره ensemble
                ensemble_path = f"{base_filepath}_ensemble"
                save_results['ensemble'] = self.ensemble.save_ensemble(ensemble_path)

                # ذخیره feature engineer
                fe_path = f"{base_filepath}_feature_engineer.joblib"
                save_results['feature_engineer'] = self.feature_engineer.save_state(fe_path)

            if self.position_sizer_trained and hasattr(self, 'position_sizer') and self.position_sizer:
                # ذخیره position sizer
                ps_path = f"{base_filepath}_position_sizer"
                save_results['position_sizer'] = self.position_sizer.save(ps_path)

            # ذخیره سیستم یادگیری آنلاین
            if hasattr(self, 'online_learning') and self.online_learning and hasattr(self.online_learning,
                                                                                     'enabled') and self.online_learning.enabled:
                ol_path = f"{base_filepath}_online_learning.joblib"
                save_results['online_learning'] = self.online_learning.save_state(ol_path)

            # ذخیره پیکربندی و متریک‌ها
            config_path = f"{base_filepath}_config.json"
            with open(config_path, 'w', encoding='utf-8') as f:
                json.dump(self.config_manager.get_config(), f, ensure_ascii=False, indent=2)

            metrics_path = f"{base_filepath}_metrics.json"
            with open(metrics_path, 'w', encoding='utf-8') as f:
                # تبدیل هر شیء غیرقابل تبدیل به رشته
                metrics_json = json.dumps(self.performance_metrics, default=lambda o: str(o), ensure_ascii=False,
                                          indent=2)
                f.write(metrics_json)

            save_results['config'] = True
            save_results['metrics'] = True

            # ذخیره متادیتا
            metadata = {
                'models_trained': self.models_trained,
                'position_sizer_trained': self.position_sizer_trained,
                'last_signal_time': self.last_signal_time.isoformat() if self.last_signal_time else None,
                'last_update_time': self.last_update_time.isoformat() if self.last_update_time else None,
                'version': '2.0',  # به‌روزرسانی شده به 2.0 برای نسخه با مدیریت پیکربندی
                'save_time': datetime.now().isoformat(),
                'saved_components': list(save_results.keys())
            }

            metadata_path = f"{base_filepath}_metadata.json"
            with open(metadata_path, 'w', encoding='utf-8') as f:
                json.dump(metadata, f, ensure_ascii=False, indent=2)

            logger.info(f"وضعیت TradingBrainAI در {base_filepath} ذخیره شد")

            return save_results

        except Exception as e:
            logger.error(f"خطا در ذخیره وضعیت TradingBrainAI: {e}", exc_info=True)
            return {'error': str(e)}

    def load_state(self, base_filepath: str) -> Dict[str, bool]:
        """بارگذاری وضعیت کامل trading brain"""
        try:
            # بررسی وجود متادیتا
            metadata_path = f"{base_filepath}_metadata.json"
            if not os.path.exists(metadata_path):
                raise FileNotFoundError(f"فایل متادیتا یافت نشد: {metadata_path}")

            # بارگذاری متادیتا
            with open(metadata_path, 'r', encoding='utf-8') as f:
                metadata = json.load(f)

            # بارگذاری پیکربندی
            config_path = f"{base_filepath}_config.json"

            # مقداردهی مجدد مدیریت پیکربندی
            if os.path.exists(config_path):
                with open(config_path, 'r', encoding='utf-8') as f:
                    loaded_config = json.load(f)
                    # ایجاد مدیریت پیکربندی جدید با تنظیمات بارگذاری شده
                    self.config_manager = ConfigManager(loaded_config)
                    logger.info("مدیریت پیکربندی با تنظیمات بارگذاری شده مقداردهی مجدد شد")

            # مقداردهی مجدد اجزای اصلی با مدیریت پیکربندی به‌روز شده
            self._initialize_components()

            # بارگذاری متریک‌ها
            metrics_path = f"{base_filepath}_metrics.json"
            if os.path.exists(metrics_path):
                with open(metrics_path, 'r', encoding='utf-8') as f:
                    self.performance_metrics = json.load(f)

            # بارگذاری feature engineer
            fe_path = f"{base_filepath}_feature_engineer.joblib"
            if os.path.exists(fe_path) and 'feature_engineer' in metadata.get('saved_components', []):
                load_results = {'feature_engineer': self.feature_engineer.load_state(fe_path)}
            else:
                load_results = {}

            # بارگذاری ensemble اگر آموزش دیده شده
            if metadata.get('models_trained', False):
                ensemble_path = f"{base_filepath}_ensemble"
                load_results['ensemble'] = self.ensemble.load_ensemble(ensemble_path)

                # تنظیم مدل‌ها از ensemble
                if 'lstm' in self.ensemble.models:
                    self.lstm_model = self.ensemble.models['lstm']
                if 'transformer' in self.ensemble.models:
                    self.transformer_model = self.ensemble.models['transformer']
                if 'cnn' in self.ensemble.models:
                    self.cnn_model = self.ensemble.models['cnn']
                if 'xgboost' in self.ensemble.models:
                    self.xgboost_model = self.ensemble.models['xgboost']

                self.models_trained = True

            # بارگذاری position sizer اگر آموزش دیده شده و کلاس آن مقداردهی شده
            # (در نسخه فعلی ما فقط استاب داریم)
            if metadata.get('position_sizer_trained', False) and hasattr(self,
                                                                         'position_sizer') and self.position_sizer:
                ps_path = f"{base_filepath}_position_sizer"
                if hasattr(self.position_sizer, 'load') and os.path.exists(ps_path):
                    load_results['position_sizer'] = self.position_sizer.load(ps_path)
                    self.position_sizer_trained = True

            # بارگذاری سیستم یادگیری آنلاین (اگر موجود است)
            if 'online_learning' in metadata.get('saved_components', []) and hasattr(self,
                                                                                     'online_learning') and self.online_learning:
                ol_path = f"{base_filepath}_online_learning.joblib"
                if os.path.exists(ol_path) and hasattr(self.online_learning, 'load_state'):
                    load_results['online_learning'] = self.online_learning.load_state(ol_path)

            # تنظیم مجدد ردیابی زمان
            if metadata.get('last_signal_time'):
                self.last_signal_time = datetime.fromisoformat(metadata['last_signal_time'])
            if metadata.get('last_update_time'):
                self.last_update_time = datetime.fromisoformat(metadata['last_update_time'])

            logger.info(f"وضعیت TradingBrainAI از {base_filepath} بارگذاری شد")

            return load_results

        except Exception as e:
            logger.error(f"خطا در بارگذاری وضعیت TradingBrainAI: {e}", exc_info=True)
            return {'error': str(e)}

    def get_model_performance(self) -> Dict[str, Any]:
        """دریافت متریک‌های عملکرد مدل"""
        return self.performance_metrics

    def get_system_status(self) -> Dict[str, Any]:
        """دریافت وضعیت فعلی سیستم"""
        status = {
            'models_trained': self.models_trained,
            'position_sizer_trained': self.position_sizer_trained,
            'last_signal_time': self.last_signal_time.isoformat() if self.last_signal_time else None,
            'last_update_time': self.last_update_time.isoformat() if self.last_update_time else None,
            'total_trades_recorded': len(self.performance_metrics.get('trade_history', [])),
            'active_positions': len(self.current_positions),
            'model_accuracy': self.performance_metrics.get('model_accuracy', {}),
            'config_status': {
                'last_update': self.config_manager._last_update_time.isoformat(),
                'enabled_features': {
                    'technical_indicators': self.config_manager.get(
                        'trading_brain_ai.feature_engineering.use_technical_indicators', True),
                    'price_patterns': self.config_manager.get(
                        'trading_brain_ai.feature_engineering.use_price_patterns', True),
                    'volatility': self.config_manager.get(
                        'trading_brain_ai.feature_engineering.use_volatility_features', True),
                    'volume': self.config_manager.get('trading_brain_ai.feature_engineering.use_volume_features',
                                                      True)
                },
                'risk_management': {
                    'risk_reward_ratio': self.config_manager.get('risk_management.preferred_risk_reward_ratio',
                                                                 2.5),
                    'stop_loss_percent': self.config_manager.get('risk_management.default_stop_loss_percent', 1.5),
                    'trailing_stop': self.config_manager.get('risk_management.use_trailing_stop', True),
                    'max_position_size': self.config_manager.get('risk_management.max_position_size', 200)
                }
            }
        }

        # اضافه کردن وضعیت یادگیری آنلاین اگر موجود باشد
        if hasattr(self, 'online_learning') and self.online_learning:
            status['online_learning_enabled'] = getattr(self.online_learning, 'enabled', False)
            status['online_learning_buffer_size'] = len(
                getattr(self.online_learning, 'feature_buffer', [])) if hasattr(self.online_learning,
                                                                                'feature_buffer') else 0

        return status

    def visualize_performance(self, output_path: str = None) -> Dict[str, str]:
        """تولید نمودارهای بصری عملکرد"""
        try:
            # ایجاد مسیر نمودارها
            if output_path is None:
                output_path = os.path.join(self.config_manager.get('models_dir', 'models'), 'visualizations')

            os.makedirs(output_path, exist_ok=True)

            # مسیرهای نمودار
            paths = {}

            # نمودار عملکرد معاملات
            if self.performance_metrics.get('trade_history'):
                trade_history = self.performance_metrics['trade_history']

                # بازده تجمعی
                if len(trade_history) > 0:
                    plt.figure(figsize=(10, 6))

                    # استخراج داده‌های سود
                    profits = [trade.get('profit_pct', 0) for trade in trade_history]
                    cumulative_returns = np.cumsum(profits)

                    # رسم نمودار
                    plt.plot(cumulative_returns)
                    plt.title('بازده تجمعی')
                    plt.xlabel('شماره معامله')
                    plt.ylabel('درصد سود تجمعی')
                    plt.grid(True)

                    # ذخیره نمودار
                    returns_path = os.path.join(output_path, 'cumulative_returns.png')
                    plt.savefig(returns_path)
                    plt.close()

                    paths['cumulative_returns'] = returns_path

                # توزیع سود/زیان
                if len(trade_history) > 0:
                    plt.figure(figsize=(10, 6))

                    # استخراج داده‌های سود
                    profits = [trade.get('profit_pct', 0) for trade in trade_history]

                    # رسم هیستوگرام
                    plt.hist(profits, bins=20, alpha=0.7)
                    plt.axvline(x=0, color='r', linestyle='--')
                    plt.title('توزیع سود/زیان')
                    plt.xlabel('درصد سود')
                    plt.ylabel('فراوانی')
                    plt.grid(True)

                    # ذخیره نمودار
                    dist_path = os.path.join(output_path, 'profit_distribution.png')
                    plt.savefig(dist_path)
                    plt.close()

                    paths['profit_distribution'] = dist_path

            # دقت مدل در طول زمان (اگر موجود باشد)
            if hasattr(self, 'online_learning') and self.online_learning and hasattr(self.online_learning,
                                                                                     'performance_metrics') and self.online_learning.performance_metrics.get(
                    'accuracy_history'):
                accuracy_history = self.online_learning.performance_metrics['accuracy_history']

                if len(accuracy_history) > 0:
                    plt.figure(figsize=(10, 6))

                    # استخراج داده‌های دقت برای هر مدل
                    dates = [datetime.fromisoformat(entry['time']) for entry in accuracy_history]

                    # رسم نمودار برای هر مدل
                    for model_name in ['lstm', 'transformer', 'cnn', 'xgboost']:
                        accuracies = [entry['metrics'].get(model_name, {}).get('accuracy', None) for entry in
                                      accuracy_history]
                        valid_data = [(d, a) for d, a in zip(dates, accuracies) if a is not None]

                        if valid_data:
                            valid_dates, valid_accs = zip(*valid_data)
                            plt.plot(valid_dates, valid_accs, label=model_name)

                    plt.title('دقت مدل در طول زمان')
                    plt.xlabel('تاریخ')
                    plt.ylabel('دقت')
                    plt.legend()
                    plt.grid(True)

                    # ذخیره نمودار
                    accuracy_path = os.path.join(output_path, 'accuracy_history.png')
                    plt.savefig(accuracy_path)
                    plt.close()

                    paths['accuracy_history'] = accuracy_path

            # نمودار وضعیت سرمایه backtest (اگر موجود باشد)
            if self.performance_metrics.get('backtest_results') and 'trades' in self.performance_metrics[
                'backtest_results']:
                backtest = self.performance_metrics['backtest_results']
                trades = backtest.get('trades', [])

                if len(trades) > 0:
                    plt.figure(figsize=(10, 6))

                    # بازسازی منحنی وضعیت سرمایه
                    initial_balance = 10000  # پیش‌فرض
                    balances = [initial_balance]

                    current_balance = initial_balance
                    for trade in trades:
                        current_balance += trade.get('pnl_amount', 0)
                        balances.append(current_balance)

                    # رسم نمودار
                    plt.plot(balances)
                    plt.title('منحنی وضعیت سرمایه در بک‌تست')
                    plt.xlabel('شماره معامله')
                    plt.ylabel('موجودی حساب')
                    plt.grid(True)

                    # ذخیره نمودار
                    equity_path = os.path.join(output_path, 'backtest_equity.png')
                    plt.savefig(equity_path)
                    plt.close()

                    paths['backtest_equity'] = equity_path

            logger.info(f"نمودارهای عملکرد در {output_path} تولید شدند")

            return {
                'status': 'success',
                'visualization_paths': paths
            }

        except Exception as e:
            logger.error(f"خطا در تولید نمودارهای عملکرد: {e}", exc_info=True)
            return {
                'status': 'error',
                'message': str(e)
            }

# Example usage
if __name__ == "__main__":
    # Create and initialize TradingBrainAI
    trading_brain = TradingBrainAI()

    # Load sample data (you would replace this with your actual data loading)
    import yfinance as yf

    # Download historical data for a ticker
    symbol = "BTC-USD"
    data = yf.download(symbol, start="2022-01-01", end="2023-01-01", interval="1d")

    # Train models
    training_result = trading_brain.train_models(data)
    print(f"نتیجه آموزش: {training_result['status']}")

    # Run a runtime configuration update (test functionality)
    trading_brain.update_config('risk_management', {
        'preferred_risk_reward_ratio': 3.0,
        'atr_trailing_multiplier': 2.5
    })

    # Generate latest trading signal
    latest_data = data.iloc[-30:]  # Use last 30 days of data
    timeframes_data = {'1d': latest_data}  # Format using timeframe dictionary
    signal_result = trading_brain.predict_signal(timeframes_data)

    if signal_result['status'] == 'success':
        signal = signal_result['signal']
        print(f"سیگنال {signal['direction']} با ضریب اطمینان {signal.get('confidence', 0):.2f} تولید شد")
        print(f"ورود: {signal.get('entry_price')}, حد ضرر: {signal.get('stop_loss')}, "
              f"حد سود: {signal.get('take_profit')}")
    else:
        print(f"سیگنالی تولید نشد: {signal_result.get('message', 'دلیل نامشخص')}")

    # Save state
    trading_brain.save_state("models/trading_brain_btc")