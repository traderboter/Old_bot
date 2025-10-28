"""
ماژول market_regime_detector.py: تشخیص شرایط بازار و تطبیق استراتژی معاملاتی
این ماژول به تشخیص شرایط مختلف بازار (روند، نوسان و غیره) و تنظیم پارامترهای استراتژی می‌پردازد.
بهینه‌سازی شده با قابلیت به‌روزرسانی پیکربندی در زمان اجرا و مدیریت کش هوشمند
"""

import logging
import numpy as np
import pandas as pd
import talib
import copy
import time
from enum import Enum, auto
from typing import Dict, Any, Optional, List, Tuple, Union, TypedDict
from functools import lru_cache

# تنظیم لاگر
logger = logging.getLogger(__name__)


class TrendStrength(Enum):
    """وضعیت‌های مختلف قدرت روند"""
    STRONG = auto()
    WEAK = auto()
    NO_TREND = auto()
    UNKNOWN = auto()


class TrendDirection(Enum):
    """جهت‌های مختلف روند"""
    BULLISH = auto()
    BEARISH = auto()
    NEUTRAL = auto()
    UNKNOWN = auto()


class VolatilityLevel(Enum):
    """سطوح مختلف نوسان"""
    HIGH = auto()
    NORMAL = auto()
    LOW = auto()
    UNKNOWN = auto()


class MarketRegime(Enum):
    """رژیم‌های مختلف بازار"""
    STRONG_TREND = auto()
    STRONG_TREND_HIGH_VOLATILITY = auto()
    WEAK_TREND = auto()
    WEAK_TREND_HIGH_VOLATILITY = auto()
    RANGE = auto()
    RANGE_HIGH_VOLATILITY = auto()
    TIGHT_RANGE = auto()
    CHOPPY = auto()  # بازار آشفته با نوسانات غیرقابل پیش‌بینی
    BREAKOUT = auto()  # شکست از یک محدوده مشخص
    UNKNOWN = auto()


class RegimeDetails(TypedDict, total=False):
    """جزئیات تشخیص رژیم بازار"""
    adx: float
    plus_di: float
    minus_di: float
    atr_percent: float
    adx_stability: float
    bollinger_width: float
    rsi: float
    volume_change: float
    error: str


class RegimeResult(TypedDict):
    """نتیجه تشخیص رژیم بازار"""
    regime: str
    trend_strength: str
    trend_direction: str
    volatility: str
    confidence: float
    details: RegimeDetails


class MarketRegimeDetector:
    """
    تشخیص شرایط بازار و تطبیق استراتژی بر اساس آن

    این کلاس با استفاده از اندیکاتورهای تکنیکال، وضعیت بازار را تشخیص داده
    و پارامترهای استراتژی را متناسب با آن تنظیم می‌کند.
    """

    def __init__(self, config: Dict[str, Any]):
        """
        مقداردهی اولیه تشخیص‌دهنده رژیم بازار

        Args:
            config: دیکشنری تنظیمات
                - market_regime.adx_period: دوره محاسبه ADX
                - market_regime.volatility_period: دوره محاسبه نوسان (ATR)
                - market_regime.strong_trend_threshold: آستانه روند قوی (ADX)
                - market_regime.weak_trend_threshold: آستانه روند ضعیف (ADX)
                - market_regime.high_volatility_threshold: آستانه نوسان بالا (ATR%)
                - market_regime.low_volatility_threshold: آستانه نوسان پایین (ATR%)
                - market_regime.bollinger_period: دوره باندهای بولینگر
                - market_regime.bollinger_std: انحراف معیار باندهای بولینگر
                - market_regime.rsi_period: دوره RSI
                - market_regime.breakout_lookback: دوره بررسی برای تشخیص شکست
                - market_regime.max_lookback: حداکثر دوره بررسی برای تضمین کارکرد درست
        """
        self.config = config
        self.config_version = int(time.time())  # برای ردیابی تغییرات پیکربندی
        self._load_config_values()

        # وضعیت داخلی
        self._last_regime: Optional[RegimeResult] = None
        self._regime_change_count: int = 0

        logger.info(f"تشخیص‌دهنده رژیم بازار راه‌اندازی شد با دوره ADX: {self.adx_period}, "
                   f"دوره نوسان: {self.volatility_period}")

    def _load_config_values(self) -> None:
        """بارگذاری مقادیر از پیکربندی به متغیرهای داخلی"""
        market_regime_config = self.config.get('market_regime', {})

        # دوره‌های لازم برای محاسبه اندیکاتورها
        self.adx_period = market_regime_config.get('adx_period', 14)
        self.volatility_period = market_regime_config.get('volatility_period', 20)
        self.bollinger_period = market_regime_config.get('bollinger_period', 20)
        self.bollinger_std = market_regime_config.get('bollinger_std', 2.0)
        self.rsi_period = market_regime_config.get('rsi_period', 14)
        self.breakout_lookback = market_regime_config.get('breakout_lookback', 10)
        self.max_lookback = market_regime_config.get('max_lookback', 50)

        # آستانه‌های تشخیص
        self.strong_trend_threshold = market_regime_config.get('strong_trend_threshold', 25)
        self.weak_trend_threshold = market_regime_config.get('weak_trend_threshold', 20)
        self.high_volatility_threshold = market_regime_config.get('high_volatility_threshold', 1.5)
        self.low_volatility_threshold = market_regime_config.get('low_volatility_threshold', 0.5)
        self.breakout_threshold = market_regime_config.get('breakout_threshold', 2.0)
        self.choppy_threshold = market_regime_config.get('choppy_threshold', 0.3)

        # تنظیمات کشینگ
        self.use_cache = market_regime_config.get('use_cache', True)
        self._cache_ttl = market_regime_config.get('cache_ttl', 100)  # تعداد فراخوانی‌ها قبل از پاک شدن کش

        # تنظیمات اضافی
        self.adapt_strategy = market_regime_config.get('adapt_strategy', True)
        self.minimum_confidence = market_regime_config.get('minimum_confidence', 0.4)
        self.use_volume_analysis = market_regime_config.get('use_volume_analysis', True)
        self.debug_mode = market_regime_config.get('debug_mode', False)

        # بررسی لاگ دیباگ تنظیمات
        if self.debug_mode:
            logger.debug(f"تنظیمات تشخیص رژیم بازار بارگذاری شد: ADX={self.adx_period}, "
                        f"نوسان={self.volatility_period}, روند قوی={self.strong_trend_threshold}")

    def update_config(self, new_config: Dict[str, Any]) -> None:
        """
        به‌روزرسانی پیکربندی در زمان اجرا

        Args:
            new_config: پیکربندی جدید
        """
        old_values = {
            'adx_period': self.adx_period,
            'volatility_period': self.volatility_period,
            'strong_trend_threshold': self.strong_trend_threshold,
            'weak_trend_threshold': self.weak_trend_threshold,
            'high_volatility_threshold': self.high_volatility_threshold,
            'low_volatility_threshold': self.low_volatility_threshold
        }

        # ذخیره پیکربندی جدید
        self.config = new_config
        self.config_version = int(time.time())

        # بارگذاری مقادیر جدید
        self._load_config_values()

        # بررسی تغییرات کلیدی
        changes = []
        for key, old_value in old_values.items():
            new_value = getattr(self, key, old_value)
            if new_value != old_value:
                changes.append(f"{key}: {old_value} -> {new_value}")

        # پاکسازی کش در صورت تغییر پارامترهای کلیدی
        if changes:
            self.reset_cache()
            changes_str = ", ".join(changes)
            logger.info(f"پیکربندی تشخیص رژیم بازار به‌روزرسانی شد: {changes_str}")
        else:
            logger.debug("پیکربندی تشخیص رژیم بازار به‌روزرسانی شد بدون تغییر در پارامترهای کلیدی")

    def _calculate_indicators(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, bool]:
        """
        محاسبه اندیکاتورهای فنی مورد نیاز

        Args:
            df: دیتافریم قیمت (OHLCV)

        Returns:
            دیتافریم با اندیکاتورهای اضافه شده و پرچم موفقیت
        """
        if len(df) < self.max_lookback:
            logger.warning(f"طول داده ({len(df)}) کمتر از مقدار مورد نیاز ({self.max_lookback}) است")
            return df, False

        try:
            # کپی برای جلوگیری از تغییر دیتافریم اصلی
            df_result = df.copy()

            # محاسبه ADX (شاخص قدرت روند)
            df_result['adx'] = talib.ADX(
                df_result['high'].values,
                df_result['low'].values,
                df_result['close'].values,
                timeperiod=self.adx_period
            )

            # محاسبه +DI و -DI (نشان‌دهنده‌های جهت روند)
            df_result['plus_di'] = talib.PLUS_DI(
                df_result['high'].values,
                df_result['low'].values,
                df_result['close'].values,
                timeperiod=self.adx_period
            )
            df_result['minus_di'] = talib.MINUS_DI(
                df_result['high'].values,
                df_result['low'].values,
                df_result['close'].values,
                timeperiod=self.adx_period
            )

            # محاسبه ATR (متوسط دامنه واقعی - شاخص نوسان)
            df_result['atr'] = talib.ATR(
                df_result['high'].values,
                df_result['low'].values,
                df_result['close'].values,
                timeperiod=self.volatility_period
            )

            # محاسبه ATR نسبی (ATR به عنوان درصدی از قیمت)
            df_result['atr_percent'] = (df_result['atr'] / df_result['close']) * 100

            # محاسبه باندهای بولینگر
            upper, middle, lower = talib.BBANDS(
                df_result['close'].values,
                timeperiod=self.bollinger_period,
                nbdevup=self.bollinger_std,
                nbdevdn=self.bollinger_std
            )
            df_result['bb_upper'] = upper
            df_result['bb_middle'] = middle
            df_result['bb_lower'] = lower
            df_result['bb_width'] = (upper - lower) / middle * 100

            # محاسبه RSI
            df_result['rsi'] = talib.RSI(df_result['close'].values, timeperiod=self.rsi_period)

            # محاسبه تغییرات حجم (اگر ستون حجم وجود داشته باشد و تحلیل حجم فعال باشد)
            if 'volume' in df_result.columns and self.use_volume_analysis:
                df_result['volume_change'] = df_result['volume'].pct_change(5) * 100
                # محاسبه میانگین متحرک حجم
                df_result['volume_sma'] = talib.SMA(df_result['volume'].values, timeperiod=20)
                # نسبت حجم فعلی به میانگین
                df_result['volume_ratio'] = df_result['volume'] / df_result['volume_sma']

            # محاسبه‌های اضافی برای تشخیص بهتر رژیم بازار
            # شاخص ثبات قیمت (Price Stability Index)
            close_std = df_result['close'].rolling(window=14).std()
            close_mean = df_result['close'].rolling(window=14).mean()
            df_result['price_stability'] = 1 - (close_std / close_mean)

            # نسبت روندها - مقایسه میانگین‌های متحرک مختلف
            df_result['sma_5'] = talib.SMA(df_result['close'].values, timeperiod=5)
            df_result['sma_20'] = talib.SMA(df_result['close'].values, timeperiod=20)
            df_result['trend_ratio'] = df_result['sma_5'] / df_result['sma_20']

            return df_result, True

        except Exception as e:
            logger.error(f"خطا در محاسبه اندیکاتورها: {e}")
            return df, False

    def _detect_breakout(self, df: pd.DataFrame) -> Tuple[bool, str]:
        """
        تشخیص شکست قیمت از محدوده (Breakout)

        Args:
            df: دیتافریم با اندیکاتورها

        Returns:
            آیا شکست اتفاق افتاده و جهت آن
        """
        try:
            if 'bb_upper' not in df.columns or 'bb_lower' not in df.columns:
                return False, "neutral"

            # بررسی شکست بالا یا پایین باندهای بولینگر
            close_values = df['close'].iloc[-self.breakout_lookback:]
            upper_values = df['bb_upper'].iloc[-self.breakout_lookback:]
            lower_values = df['bb_lower'].iloc[-self.breakout_lookback:]

            # شکست به بالا: قیمت بالاتر از باند بالایی
            if close_values.iloc[-1] > upper_values.iloc[-1] and all(close_values.iloc[-3:-1] <= upper_values.iloc[-3:-1]):
                # بررسی شدت شکست
                breakout_strength = (close_values.iloc[-1] - upper_values.iloc[-1]) / df['atr'].iloc[-1]

                if breakout_strength > self.breakout_threshold:
                    return True, "bullish"
                else:
                    return False, "neutral"

            # شکست به پایین: قیمت پایین‌تر از باند پایینی
            if close_values.iloc[-1] < lower_values.iloc[-1] and all(close_values.iloc[-3:-1] >= lower_values.iloc[-3:-1]):
                # بررسی شدت شکست
                breakout_strength = (lower_values.iloc[-1] - close_values.iloc[-1]) / df['atr'].iloc[-1]

                if breakout_strength > self.breakout_threshold:
                    return True, "bearish"
                else:
                    return False, "neutral"

            return False, "neutral"

        except Exception as e:
            logger.error(f"خطا در تشخیص شکست: {e}")
            return False, "neutral"

    def _is_choppy_market(self, df: pd.DataFrame) -> bool:
        """
        تشخیص بازار آشفته و بی‌ثبات

        Args:
            df: دیتافریم با اندیکاتورها

        Returns:
            آیا بازار آشفته است
        """
        try:
            if 'adx' not in df.columns or 'rsi' not in df.columns:
                return False

            # ADX پایین نشان‌دهنده عدم وجود روند است
            low_adx = df['adx'].iloc[-1] < self.weak_trend_threshold

            # تغییرات سریع در RSI نشان‌دهنده بازار آشفته است
            rsi_changes = abs(df['rsi'].diff(1).iloc[-5:])
            high_rsi_changes = (rsi_changes > 10).sum() >= 3

            # نوسان قیمت بالا در محدوده کوچک
            if 'atr_percent' in df.columns:
                price_changes = abs(df['close'].pct_change(1).iloc[-5:]) * 100
                avg_change = price_changes.mean()
                direction_changes = (np.sign(df['close'].diff(1).iloc[-6:]).diff(1) != 0).sum()

                # تعداد زیاد تغییر جهت + تغییرات قیمت بالا = بازار آشفته
                if low_adx and (high_rsi_changes or (direction_changes >= 3 and avg_change >= self.choppy_threshold)):
                    if self.debug_mode:
                        logger.debug(f"بازار آشفته تشخیص داده شد: ADX={df['adx'].iloc[-1]:.2f}, "
                                     f"تغییرات جهت={direction_changes}, تغییر قیمت میانگین={avg_change:.2f}%")
                    return True

            return low_adx and high_rsi_changes

        except Exception as e:
            logger.error(f"خطا در تشخیص بازار آشفته: {e}")
            return False

    @lru_cache(maxsize=32)
    def _cached_detect_regime(self, df_hash: str, config_hash: str) -> RegimeResult:
        """
        نسخه کش شده تشخیص رژیم بازار

        Args:
            df_hash: هش دیتافریم برای کش
            config_hash: هش تنظیمات برای کش

        Returns:
            نتیجه تشخیص رژیم بازار
        """
        # این متد فقط برای کشینگ است و فراخوانی مستقیم نمی‌شود
        # پارامترهای هش فقط برای کشینگ استفاده می‌شوند
        if self.debug_mode:
            logger.debug(f"خطای کش برای df_hash={df_hash[:8]}... config_hash={config_hash[:8]}...")
        return self._detect_regime_internal(None)

    def _get_hash_for_caching(self, df: pd.DataFrame) -> Tuple[str, str]:
        """
        ایجاد هش از دیتافریم و تنظیمات برای کشینگ

        Args:
            df: دیتافریم قیمت

        Returns:
            هش دیتافریم و هش تنظیمات
        """
        # هش از آخرین N ردیف دیتافریم
        last_rows = min(20, len(df))
        df_values = df.iloc[-last_rows:].to_numpy().tobytes()
        df_hash = str(hash(df_values))

        # هش از تنظیمات مهم و نسخه پیکربندی
        config_values = (
            self.adx_period,
            self.volatility_period,
            self.strong_trend_threshold,
            self.weak_trend_threshold,
            self.high_volatility_threshold,
            self.low_volatility_threshold,
            self.config_version  # اضافه کردن نسخه پیکربندی برای اطمینان از بروز بودن
        )
        config_hash = str(hash(config_values))

        return df_hash, config_hash

    def _detect_regime_internal(self, df: Optional[pd.DataFrame] = None) -> RegimeResult:
        """
        پیاده‌سازی داخلی تشخیص رژیم بازار

        Args:
            df: دیتافریم با اندیکاتورها (اگر None باشد، یک نتیجه خطا برمی‌گرداند)

        Returns:
            نتیجه تشخیص رژیم بازار
        """
        # نتیجه پیش‌فرض در صورت خطا
        default_result: RegimeResult = {
            'regime': MarketRegime.UNKNOWN.name.lower(),
            'trend_strength': TrendStrength.UNKNOWN.name.lower(),
            'trend_direction': TrendDirection.UNKNOWN.name.lower(),
            'volatility': VolatilityLevel.UNKNOWN.name.lower(),
            'confidence': 0.0,
            'details': {}
        }

        if df is None:
            default_result['details'] = {'error': 'داده‌ای ارائه نشده است'}
            return default_result

        if len(df) < 5:
            default_result['details'] = {'error': f'داده‌ها کافی نیستند: {len(df)} ردیف'}
            return default_result

        try:
            # داده‌های فعلی (آخرین ردیف)
            current_adx = float(df['adx'].iloc[-1])
            current_plus_di = float(df['plus_di'].iloc[-1])
            current_minus_di = float(df['minus_di'].iloc[-1])
            current_atr_percent = float(df['atr_percent'].iloc[-1])
            current_bb_width = float(df['bb_width'].iloc[-1]) if 'bb_width' in df.columns else 0.0
            current_rsi = float(df['rsi'].iloc[-1]) if 'rsi' in df.columns else 50.0

            # مقادیر volume_change اگر وجود داشته باشد
            current_volume_change = float(df['volume_change'].iloc[-1]) if 'volume_change' in df.columns else 0.0
            # نسبت حجم اگر وجود داشته باشد
            current_volume_ratio = float(df['volume_ratio'].iloc[-1]) if 'volume_ratio' in df.columns else 1.0

            # تعیین قدرت و جهت روند
            trend_strength = (
                TrendStrength.STRONG.name.lower() if current_adx > self.strong_trend_threshold else
                TrendStrength.WEAK.name.lower() if current_adx > self.weak_trend_threshold else
                TrendStrength.NO_TREND.name.lower()
            )

            trend_direction = (
                TrendDirection.BULLISH.name.lower() if current_plus_di > current_minus_di else
                TrendDirection.BEARISH.name.lower() if current_minus_di > current_plus_di else
                TrendDirection.NEUTRAL.name.lower()  # اگر هر دو برابر باشند
            )

            # تعیین سطح نوسان
            volatility_level = (
                VolatilityLevel.HIGH.name.lower() if current_atr_percent > self.high_volatility_threshold else
                VolatilityLevel.LOW.name.lower() if current_atr_percent < self.low_volatility_threshold else
                VolatilityLevel.NORMAL.name.lower()
            )

            # بررسی شکست قیمت
            is_breakout, breakout_direction = self._detect_breakout(df)

            # بررسی بازار آشفته
            is_choppy = self._is_choppy_market(df)

            # شناسایی رژیم بازار
            regime = MarketRegime.UNKNOWN

            if is_breakout:
                regime = MarketRegime.BREAKOUT
            elif is_choppy:
                regime = MarketRegime.CHOPPY
            elif trend_strength == TrendStrength.STRONG.name.lower():
                if volatility_level == VolatilityLevel.HIGH.name.lower():
                    regime = MarketRegime.STRONG_TREND_HIGH_VOLATILITY
                else:
                    regime = MarketRegime.STRONG_TREND
            elif trend_strength == TrendStrength.WEAK.name.lower():
                if volatility_level == VolatilityLevel.HIGH.name.lower():
                    regime = MarketRegime.WEAK_TREND_HIGH_VOLATILITY
                else:
                    regime = MarketRegime.WEAK_TREND
            else:  # no_trend
                if volatility_level == VolatilityLevel.HIGH.name.lower():
                    regime = MarketRegime.RANGE_HIGH_VOLATILITY
                elif volatility_level == VolatilityLevel.LOW.name.lower():
                    regime = MarketRegime.TIGHT_RANGE
                else:
                    regime = MarketRegime.RANGE

            # محاسبه سطح اطمینان تشخیص
            # مثلاً بر اساس ثبات ADX در چند دوره اخیر و پارامترهای دیگر
            adx_stability = 1.0
            try:
                recent_adx = df['adx'].iloc[-5:]
                adx_stability = 1.0 - min(1.0, recent_adx.std() / max(0.1, recent_adx.mean()))
            except:
                pass

            # بررسی همبستگی حرکت قیمت با حجم
            volume_price_correlation = 0.5  # مقدار پیش‌فرض
            if self.use_volume_analysis and 'volume' in df.columns:
                try:
                    # همبستگی بین تغییرات قیمت و حجم در 20 کندل اخیر
                    correlation = df['close'].pct_change().iloc[-20:].corr(df['volume'].pct_change().iloc[-20:])
                    volume_price_correlation = abs(correlation)  # استفاده از مقدار مطلق همبستگی
                except:
                    pass

            # ترکیب عوامل مختلف در محاسبه اطمینان
            confidence_factors = [
                adx_stability * 0.5,  # ثبات ADX (وزن: 50%)
                0.3,  # پایه اطمینان (30%)
            ]

            # اگر شکست تشخیص داده شده، اطمینان بالاتر (اگر با جهت روند هماهنگ باشد)
            if is_breakout and ((breakout_direction == "bullish" and trend_direction == TrendDirection.BULLISH.name.lower()) or
                               (breakout_direction == "bearish" and trend_direction == TrendDirection.BEARISH.name.lower())):
                confidence_factors.append(0.2)

            # اگر تحلیل حجم فعال است، از آن هم در محاسبه اطمینان استفاده کنیم
            if self.use_volume_analysis and current_volume_ratio > 1.5:
                confidence_factors.append(0.1 * volume_price_correlation)  # افزایش اطمینان بر اساس همبستگی حجم و قیمت

            # اطمینان نهایی
            confidence = min(1.0, sum(confidence_factors))

            # اضافه کردن جزئیات بیشتر برای تشخیص بهتر رژیم بازار
            details_dict = {
                'adx': current_adx,
                'plus_di': current_plus_di,
                'minus_di': current_minus_di,
                'atr_percent': current_atr_percent,
                'adx_stability': adx_stability,
                'bollinger_width': current_bb_width,
                'rsi': current_rsi,
                'volume_change': current_volume_change
            }

            # اضافه کردن پارامترهای اضافی اگر موجود باشند
            if self.use_volume_analysis and 'volume_ratio' in df.columns:
                details_dict['volume_ratio'] = current_volume_ratio
                details_dict['volume_price_correlation'] = volume_price_correlation

            if 'price_stability' in df.columns:
                details_dict['price_stability'] = float(df['price_stability'].iloc[-1])

            if 'trend_ratio' in df.columns:
                details_dict['trend_ratio'] = float(df['trend_ratio'].iloc[-1])

            # برگرداندن نتایج
            result: RegimeResult = {
                'regime': regime.name.lower(),
                'trend_strength': trend_strength,
                'trend_direction': trend_direction,
                'volatility': volatility_level,
                'confidence': confidence,
                'details': details_dict
            }

            # بررسی تغییر رژیم نسبت به تشخیص قبلی
            if self._last_regime and self._last_regime['regime'] != result['regime']:
                self._regime_change_count += 1
                logger.info(f"رژیم بازار از {self._last_regime['regime']} به {result['regime']} تغییر کرد "
                           f"(تغییر شماره {self._regime_change_count})")

            self._last_regime = result
            return result

        except Exception as e:
            logger.error(f"خطا در تشخیص رژیم بازار: {e}", exc_info=True)
            default_result['details'] = {'error': str(e)}
            return default_result

    def detect_regime(self, df: pd.DataFrame) -> RegimeResult:
        """
        تشخیص شرایط بازار

        Args:
            df: دیتافریم قیمت (OHLCV)

        Returns:
            دیکشنری نتایج تشخیص
        """
        if df is None or len(df) < self.adx_period * 2:
            return {
                'regime': MarketRegime.UNKNOWN.name.lower(),
                'trend_strength': TrendStrength.UNKNOWN.name.lower(),
                'trend_direction': TrendDirection.UNKNOWN.name.lower(),
                'volatility': VolatilityLevel.UNKNOWN.name.lower(),
                'confidence': 0.0,
                'details': {'error': f'داده‌های ناکافی: {len(df) if df is not None else 0} ردیف'}
            }

        try:
            # محاسبه اندیکاتورها روی دیتافریم
            df_with_indicators, success = self._calculate_indicators(df)

            if not success:
                return {
                    'regime': MarketRegime.UNKNOWN.name.lower(),
                    'trend_strength': TrendStrength.UNKNOWN.name.lower(),
                    'trend_direction': TrendDirection.UNKNOWN.name.lower(),
                    'volatility': VolatilityLevel.UNKNOWN.name.lower(),
                    'confidence': 0.0,
                    'details': {'error': 'محاسبه اندیکاتورها ناموفق بود'}
                }

            # حذف NaN های اولیه
            df_with_indicators = df_with_indicators.dropna()

            # استفاده از کش یا تشخیص مستقیم
            if self.use_cache:
                df_hash, config_hash = self._get_hash_for_caching(df_with_indicators)
                return self._cached_detect_regime(df_hash, config_hash)
            else:
                return self._detect_regime_internal(df_with_indicators)

        except Exception as e:
            logger.error(f"خطا در detect_regime: {e}", exc_info=True)
            return {
                'regime': MarketRegime.UNKNOWN.name.lower(),
                'trend_strength': TrendStrength.UNKNOWN.name.lower(),
                'trend_direction': TrendDirection.UNKNOWN.name.lower(),
                'volatility': VolatilityLevel.UNKNOWN.name.lower(),
                'confidence': 0.0,
                'details': {'error': str(e)}
            }

    def get_strategy_parameters(self, regime: Dict[str, Any], base_config: Dict[str, Any]) -> Dict[str, Any]:
        """
        تنظیم پارامترهای استراتژی بر اساس شرایط بازار

        Args:
            regime: نتیجه تشخیص شرایط بازار
            base_config: تنظیمات پایه

        Returns:
            تنظیمات به‌روزشده
        """
        if regime is None or base_config is None:
            return copy.deepcopy(base_config) if base_config else {}

        # عدم تطبیق استراتژی اگر در تنظیمات غیرفعال شده باشد
        if not self.adapt_strategy:
            if self.debug_mode:
                logger.debug("تطبیق استراتژی غیرفعال است، استفاده از پارامترهای پیش‌فرض")
            return copy.deepcopy(base_config)

        config_copy = copy.deepcopy(base_config)
        regime_type = regime.get('regime', MarketRegime.UNKNOWN.name.lower())
        trend_direction = regime.get('trend_direction', TrendDirection.UNKNOWN.name.lower())
        confidence = regime.get('confidence', 0.0)

        if regime_type == MarketRegime.UNKNOWN.name.lower() or confidence < self.minimum_confidence:
            logger.info(f"اطمینان پایین در تشخیص رژیم بازار ({confidence:.2f})، استفاده از پارامترهای پیش‌فرض")
            return config_copy

        # تنظیمات مدیریت ریسک و تولید سیگنال
        risk_management = config_copy.get('risk_management', {})
        signal_generation = config_copy.get('signal_generation', {})
        position_sizing = config_copy.get('position_sizing', {})

        # پارامترهای پایه
        default_stop_loss = risk_management.get('default_stop_loss_percent', 1.5)
        default_risk_reward = risk_management.get('preferred_risk_reward_ratio', 2.5)
        default_max_risk = risk_management.get('max_risk_per_trade_percent', 1.0)

        # ضرایب تعدیل بر اساس رژیم بازار
        adjustment_factors = {
            # روند قوی
            MarketRegime.STRONG_TREND.name.lower(): {
                'stop_loss': 0.9,  # استاپ نزدیک‌تر
                'risk_reward': 1.2,  # هدف دورتر
                'max_risk': 1.0,  # ریسک عادی
                'trend_following_weight': 1.5,  # تقویت سیگنال‌های روند
                'reversal_weight': 0.6,  # تضعیف سیگنال‌های برگشت
                'position_size': 1.1,  # اندازه معامله بزرگتر
                'trailing_stop_activation_percent': 0.9,  # فعال‌سازی سریع‌تر تریلینگ استاپ
                'trailing_stop_distance_percent': 0.9,  # فاصله کمتر تریلینگ استاپ
                'use_trailing_stop': True  # فعال کردن تریلینگ استاپ
            },
            # روند قوی با نوسان بالا
            MarketRegime.STRONG_TREND_HIGH_VOLATILITY.name.lower(): {
                'stop_loss': 1.2,  # استاپ دورتر به دلیل نوسان بالا
                'risk_reward': 1.1,  # هدف کمی دورتر
                'max_risk': 0.85,  # کاهش ریسک
                'trend_following_weight': 1.3,  # تقویت سیگنال‌های روند
                'reversal_weight': 0.5,  # تضعیف شدید سیگنال‌های برگشت
                'position_size': 0.9,  # اندازه معامله کوچکتر به دلیل ریسک بالاتر
                'trailing_stop_activation_percent': 1.0,  # فعال‌سازی عادی تریلینگ استاپ
                'trailing_stop_distance_percent': 1.1,  # فاصله بیشتر تریلینگ استاپ
                'use_trailing_stop': True  # فعال کردن تریلینگ استاپ
            },
            # روند ضعیف
            MarketRegime.WEAK_TREND.name.lower(): {
                'stop_loss': 0.95,  # استاپ کمی نزدیک‌تر
                'risk_reward': 1.05,  # هدف کمی دورتر
                'max_risk': 0.9,  # کاهش کم ریسک
                'trend_following_weight': 1.2,  # تقویت سیگنال‌های روند
                'reversal_weight': 0.8,  # تضعیف سیگنال‌های برگشت
                'position_size': 1.0,  # اندازه معامله عادی
                'trailing_stop_activation_percent': 1.0,  # فعال‌سازی عادی تریلینگ استاپ
                'trailing_stop_distance_percent': 1.0,  # فاصله عادی تریلینگ استاپ
                'use_trailing_stop': True  # فعال کردن تریلینگ استاپ
            },
            # روند ضعیف با نوسان بالا
            MarketRegime.WEAK_TREND_HIGH_VOLATILITY.name.lower(): {
                'stop_loss': 1.15,  # استاپ دورتر به دلیل نوسان بالا
                'risk_reward': 1.0,  # هدف عادی
                'max_risk': 0.8,  # کاهش ریسک
                'trend_following_weight': 1.1,  # کمی تقویت سیگنال‌های روند
                'reversal_weight': 0.7,  # تضعیف سیگنال‌های برگشت
                'position_size': 0.85,  # اندازه معامله کوچکتر به دلیل ریسک بالاتر
                'trailing_stop_activation_percent': 1.1,  # فعال‌سازی کندتر تریلینگ استاپ
                'trailing_stop_distance_percent': 1.2,  # فاصله بیشتر تریلینگ استاپ
                'use_trailing_stop': True  # فعال کردن تریلینگ استاپ
            },
            # رنج
            MarketRegime.RANGE.name.lower(): {
                'stop_loss': 1.1,  # استاپ دورتر
                'risk_reward': 0.9,  # هدف نزدیک‌تر
                'max_risk': 0.9,  # کاهش کم ریسک
                'trend_following_weight': 0.8,  # تضعیف سیگنال‌های روند
                'reversal_weight': 1.3,  # تقویت سیگنال‌های برگشت
                'position_size': 0.95,  # اندازه معامله کمی کوچکتر
                'trailing_stop_activation_percent': 1.0,  # فعال‌سازی عادی تریلینگ استاپ
                'trailing_stop_distance_percent': 0.9,  # فاصله کمتر تریلینگ استاپ
                'use_trailing_stop': True  # فعال کردن تریلینگ استاپ
            },
            # رنج با نوسان بالا
            MarketRegime.RANGE_HIGH_VOLATILITY.name.lower(): {
                'stop_loss': 1.3,  # استاپ خیلی دورتر به دلیل نوسان بالا
                'risk_reward': 0.8,  # هدف نزدیک‌تر
                'max_risk': 0.75,  # کاهش زیاد ریسک
                'trend_following_weight': 0.7,  # تضعیف سیگنال‌های روند
                'reversal_weight': 1.2,  # تقویت سیگنال‌های برگشت
                'position_size': 0.8,  # اندازه معامله کوچکتر
                'trailing_stop_activation_percent': 1.1,  # فعال‌سازی کندتر تریلینگ استاپ
                'trailing_stop_distance_percent': 1.3,  # فاصله بیشتر تریلینگ استاپ
                'use_trailing_stop': True  # فعال کردن تریلینگ استاپ
            },
            # رنج محکم (نوسان کم)
            MarketRegime.TIGHT_RANGE.name.lower(): {
                'stop_loss': 1.05,  # استاپ کمی دورتر
                'risk_reward': 0.95,  # هدف کمی نزدیک‌تر
                'max_risk': 0.95,  # کاهش کم ریسک
                'trend_following_weight': 0.7,  # تضعیف سیگنال‌های روند
                'reversal_weight': 1.4,  # تقویت قوی سیگنال‌های برگشت
                'position_size': 1.0,  # اندازه معامله عادی
                'trailing_stop_activation_percent': 1.0,  # فعال‌سازی عادی تریلینگ استاپ
                'trailing_stop_distance_percent': 0.8,  # فاصله کمتر تریلینگ استاپ
                'use_trailing_stop': False  # غیرفعال کردن تریلینگ استاپ
            },
            # بازار آشفته
            MarketRegime.CHOPPY.name.lower(): {
                'stop_loss': 1.25,  # استاپ دورتر به دلیل نوسانات غیرقابل پیش‌بینی
                'risk_reward': 0.85,  # هدف نزدیک‌تر
                'max_risk': 0.7,  # کاهش زیاد ریسک
                'trend_following_weight': 0.5,  # تضعیف شدید سیگنال‌های روند
                'reversal_weight': 0.8,  # تضعیف سیگنال‌های برگشت
                'position_size': 0.7,  # اندازه معامله خیلی کوچکتر
                'trailing_stop_activation_percent': 1.5,  # فعال‌سازی خیلی کندتر تریلینگ استاپ
                'trailing_stop_distance_percent': 1.5,  # فاصله خیلی بیشتر تریلینگ استاپ
                'use_trailing_stop': False  # غیرفعال کردن تریلینگ استاپ
            },
            # شکست (Breakout)
            MarketRegime.BREAKOUT.name.lower(): {
                'stop_loss': 0.85,  # استاپ نزدیک‌تر برای حفظ سود
                'risk_reward': 1.3,  # هدف دورتر
                'max_risk': 1.1,  # افزایش کم ریسک
                'trend_following_weight': 1.6,  # تقویت قوی سیگنال‌های روند
                'reversal_weight': 0.5,  # تضعیف شدید سیگنال‌های برگشت
                'position_size': 1.2,  # اندازه معامله بزرگتر
                'trailing_stop_activation_percent': 0.8,  # فعال‌سازی سریع‌تر تریلینگ استاپ
                'trailing_stop_distance_percent': 0.8,  # فاصله کمتر تریلینگ استاپ
                'use_trailing_stop': True  # فعال کردن تریلینگ استاپ
            }
        }

        # اعمال ضرایب تعدیل برای رژیم فعلی
        if regime_type in adjustment_factors:
            factors = adjustment_factors[regime_type]
            changes = []

            # تنظیم پارامترهای مدیریت ریسک
            if 'default_stop_loss_percent' in risk_management:
                old_value = risk_management['default_stop_loss_percent']
                risk_management['default_stop_loss_percent'] = default_stop_loss * factors['stop_loss']
                changes.append(f"default_stop_loss_percent: {old_value:.2f} -> {risk_management['default_stop_loss_percent']:.2f}")

            if 'preferred_risk_reward_ratio' in risk_management:
                old_value = risk_management['preferred_risk_reward_ratio']
                risk_management['preferred_risk_reward_ratio'] = default_risk_reward * factors['risk_reward']
                changes.append(f"preferred_risk_reward_ratio: {old_value:.2f} -> {risk_management['preferred_risk_reward_ratio']:.2f}")

            if 'max_risk_per_trade_percent' in risk_management:
                old_value = risk_management['max_risk_per_trade_percent']
                risk_management['max_risk_per_trade_percent'] = default_max_risk * factors['max_risk']
                changes.append(f"max_risk_per_trade_percent: {old_value:.2f} -> {risk_management['max_risk_per_trade_percent']:.2f}")

            # تنظیم پارامترهای تریلینگ استاپ
            if 'use_trailing_stop' in risk_management:
                old_value = risk_management['use_trailing_stop']
                risk_management['use_trailing_stop'] = factors['use_trailing_stop']
                if old_value != risk_management['use_trailing_stop']:
                    changes.append(f"use_trailing_stop: {old_value} -> {risk_management['use_trailing_stop']}")

            if 'trailing_stop_activation_percent' in risk_management:
                old_value = risk_management['trailing_stop_activation_percent']
                risk_management['trailing_stop_activation_percent'] = risk_management.get('trailing_stop_activation_percent', 3.0) * factors.get('trailing_stop_activation_percent', 1.0)
                changes.append(f"trailing_stop_activation_percent: {old_value:.2f} -> {risk_management['trailing_stop_activation_percent']:.2f}")

            if 'trailing_stop_distance_percent' in risk_management:
                old_value = risk_management['trailing_stop_distance_percent']
                risk_management['trailing_stop_distance_percent'] = risk_management.get('trailing_stop_distance_percent', 2.25) * factors.get('trailing_stop_distance_percent', 1.0)
                changes.append(f"trailing_stop_distance_percent: {old_value:.2f} -> {risk_management['trailing_stop_distance_percent']:.2f}")

            # تنظیم وزن سیگنال‌ها
            if 'trend_following_weight' in factors:
                signal_generation['trend_following_weight'] = factors['trend_following_weight']
                changes.append(f"trend_following_weight: {signal_generation['trend_following_weight']:.2f}")

            if 'reversal_weight' in factors:
                signal_generation['reversal_weight'] = factors['reversal_weight']
                changes.append(f"reversal_weight: {signal_generation['reversal_weight']:.2f}")

            # تنظیم اندازه موقعیت
            if 'position_size_multiplier' in position_sizing:
                old_value = position_sizing['position_size_multiplier']
                position_sizing['position_size_multiplier'] = position_sizing.get('position_size_multiplier', 1.0) * factors['position_size']
                changes.append(f"position_size_multiplier: {old_value:.2f} -> {position_sizing['position_size_multiplier']:.2f}")

            # لاگ تغییرات مهم
            if changes and self.debug_mode:
                changes_str = ", ".join(changes)
                logger.info(f"پارامترهای استراتژی برای رژیم {regime_type} با جهت {trend_direction} تنظیم شدند: {changes_str}")

        # تعدیل‌های اضافی بر اساس جهت روند
        if trend_direction == TrendDirection.BULLISH.name.lower():
            # در روند صعودی، ممکن است استراتژی خاصی را تقویت کنید
            if 'bull_market_adjustments' in config_copy:
                bull_adjustments = config_copy['bull_market_adjustments']
                for param_path, adjustment in bull_adjustments.items():
                    # مثال: 'risk_management.trailing_stop_activation' -> 1.2
                    path_parts = param_path.split('.')
                    if len(path_parts) == 2:
                        section, param = path_parts
                        if section in config_copy and param in config_copy[section]:
                            old_value = config_copy[section][param]
                            config_copy[section][param] *= adjustment
                            if self.debug_mode:
                                logger.debug(f"تنظیم پارامتر بازار صعودی: {param_path}: {old_value:.2f} -> {config_copy[section][param]:.2f}")

        elif trend_direction == TrendDirection.BEARISH.name.lower():
            # در روند نزولی، ممکن است استراتژی خاصی را تقویت کنید
            if 'bear_market_adjustments' in config_copy:
                bear_adjustments = config_copy['bear_market_adjustments']
                for param_path, adjustment in bear_adjustments.items():
                    path_parts = param_path.split('.')
                    if len(path_parts) == 2:
                        section, param = path_parts
                        if section in config_copy and param in config_copy[section]:
                            old_value = config_copy[section][param]
                            config_copy[section][param] *= adjustment
                            if self.debug_mode:
                                logger.debug(f"تنظیم پارامتر بازار نزولی: {param_path}: {old_value:.2f} -> {config_copy[section][param]:.2f}")

        # به‌روزرسانی کانفیگ
        config_copy['risk_management'] = risk_management
        config_copy['signal_generation'] = signal_generation
        if position_sizing:
            config_copy['position_sizing'] = position_sizing

        # ذخیره اطلاعات رژیم فعلی در کانفیگ
        if 'current_market_conditions' not in config_copy:
            config_copy['current_market_conditions'] = {}

        config_copy['current_market_conditions'] = {
            'regime': regime_type,
            'trend_direction': trend_direction,
            'confidence': confidence,
            'detection_time': pd.Timestamp.now().isoformat(),
            'details': regime.get('details', {})
        }

        logger.info(f"استراتژی برای بازار {regime_type} با روند {trend_direction} و اطمینان {confidence:.2f} تنظیم شد")

        return config_copy

    def reset_cache(self) -> None:
        """پاک کردن کش"""
        if hasattr(self, '_cached_detect_regime'):
            self._cached_detect_regime.cache_clear()
            logger.info("کش تشخیص رژیم بازار پاک شد")

    def get_current_regime(self) -> Optional[RegimeResult]:
        """
        دریافت آخرین رژیم بازار تشخیص داده شده

        Returns:
            آخرین رژیم بازار تشخیص داده شده یا None
        """
        return self._last_regime

    def get_config_params(self) -> Dict[str, Any]:
        """
        دریافت پارامترهای فعلی پیکربندی

        Returns:
            دیکشنری پارامترهای پیکربندی
        """
        return {
            'adx_period': self.adx_period,
            'volatility_period': self.volatility_period,
            'strong_trend_threshold': self.strong_trend_threshold,
            'weak_trend_threshold': self.weak_trend_threshold,
            'high_volatility_threshold': self.high_volatility_threshold,
            'low_volatility_threshold': self.low_volatility_threshold,
            'use_cache': self.use_cache,
            'adapt_strategy': self.adapt_strategy,
            'minimum_confidence': self.minimum_confidence,
            'use_volume_analysis': self.use_volume_analysis,
            'debug_mode': self.debug_mode,
            'config_version': self.config_version
        }