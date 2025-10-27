"""
ماژول ensemble_strategy.py: پیاده‌سازی سیستم ترکیبی استراتژی‌های معاملاتی
این ماژول از چندین استراتژی مختلف برای تولید سیگنال‌های معاملاتی قوی‌تر استفاده می‌کند.
بهینه‌سازی شده با قابلیت تنظیم پارامترها در زمان اجرا و بازسازی خودکار استراتژی‌ها
"""

import logging
import pandas as pd
import asyncio
import copy
import time
import traceback
from enum import Enum, auto
from typing import Dict, Any, Optional, List, Tuple, Union, TypedDict, Callable, Protocol, Set
from dataclasses import dataclass
from signal_generator import SignalGenerator, SignalInfo, SignalScore

# تنظیم لاگر
logger = logging.getLogger(__name__)


class StrategyType(Enum):
    """انواع استراتژی‌های قابل استفاده در سیستم ترکیبی"""
    BASE = auto()
    TREND_FOLLOWING = auto()
    MEAN_REVERSION = auto()
    BREAKOUT = auto()
    MOMENTUM = auto()
    RANGE_TRADING = auto()
    VOLATILITY_BASED = auto()
    CUSTOM = auto()


class StrategyInfo(TypedDict, total=False):
    """اطلاعات مربوط به یک استراتژی در سیستم ترکیبی"""
    name: str
    generator: SignalGenerator
    weight: float
    type: StrategyType
    enabled: bool
    custom_config: Dict[str, Any]
    description: str
    strategy_id: str  # شناسه یکتا برای استراتژی


class EnsembleSignalResult(TypedDict, total=False):
    """نتیجه تولید سیگنال ترکیبی"""
    signal: Optional[SignalInfo]
    voting_metrics: Dict[str, Any]
    contributing_strategies: List[str]
    agreement_ratio: float
    generation_time: float
    error: Optional[str]


class StrategyConfigBuilder(Protocol):
    """پروتکل برای سازندگان پیکربندی استراتژی"""
    def __call__(self, base_config: Dict[str, Any]) -> Dict[str, Any]:
        ...


class StrategyEnsemble:
    """
    کلاس ترکیب‌کننده استراتژی‌های مختلف برای تولید سیگنال‌های قوی‌تر

    این کلاس از چندین استراتژی مختلف برای تحلیل بازار و تولید سیگنال‌های معاملاتی
    استفاده می‌کند. سیگنال نهایی بر اساس رأی‌گیری وزن‌دار بین استراتژی‌ها تعیین می‌شود.
    """

    def __init__(self, config: Dict[str, Any]):
        """
        مقداردهی اولیه کلاس ترکیب‌کننده استراتژی

        Args:
            config: دیکشنری تنظیمات شامل:
                - ensemble_strategy: تنظیمات مخصوص سیستم ترکیبی
                    - voting_threshold: آستانه توافق برای تولید سیگنال
                    - min_strategies_agreement: حداقل تعداد استراتژی‌های موافق
                    - trend_strategy_weight: وزن استراتژی دنبال‌کننده روند
                    - mean_reversion_strategy_weight: وزن استراتژی بازگشت به میانگین
                    - breakout_strategy_weight: وزن استراتژی شکست
                    - enabled_strategies: لیست استراتژی‌های فعال
                    - dynamic_weights: فعال‌سازی وزن‌دهی پویا
        """
        # ذخیره تنظیمات اصلی
        self.config = copy.deepcopy(config)
        self.ensemble_config = self.config.get('ensemble_strategy', {})
        self.last_config_hash = self._calculate_config_hash(self.ensemble_config)

        # تنظیم اطلاع‌رسانی
        notification = self.config.get('notification', {})
        self.events = notification.get('events', {})
        self.signal_generated = self.events.get('signal_generated')

        # پارامترهای سیستم ترکیبی (دسترسی مستقیم برای بهینه‌سازی عملکرد)
        self._init_core_parameters()

        # تنظیم استراتژی‌های فعال
        self.enabled_strategy_types = self._parse_enabled_strategies()

        # مقداردهی متریک‌های کارایی
        self.performance_metrics = {
            'strategy_type': {},
            'symbol': {},
            'timeframe': {}
        }

        # ایجاد استراتژی‌های مختلف
        self.strategies: List[StrategyInfo] = []
        self.strategy_builders: Dict[StrategyType, Tuple[str, StrategyConfigBuilder, float]] = self._setup_strategy_builders()
        self._create_strategies()

        logger.info(f"سیستم ترکیبی استراتژی‌ها با {len(self.strategies)} استراتژی راه‌اندازی شد")

        # اطلاعات استراتژی‌های فعال
        self._log_active_strategies()

    def _init_core_parameters(self) -> None:
        """مقداردهی پارامترهای اصلی از تنظیمات"""
        self.voting_threshold = self.ensemble_config.get('voting_threshold', 0.6)
        self.min_strategies_agreement = self.ensemble_config.get('min_strategies_agreement', 2)
        self.dynamic_weights = self.ensemble_config.get('dynamic_weights', False)
        self.adaptive_threshold = self.ensemble_config.get('adaptive_threshold', False)
        self.bias_correction = self.ensemble_config.get('bias_correction', False)

        # پارامترهای جدید برای کنترل دقیق‌تر
        self.market_regime_awareness = self.ensemble_config.get('market_regime_awareness', True)
        self.strategy_refresh_on_update = self.ensemble_config.get('strategy_refresh_on_update', True)
        self.log_decision_metrics = self.ensemble_config.get('log_decision_metrics', False)
        self.smart_weight_adjustment = self.ensemble_config.get('smart_weight_adjustment', True)
        self.confidence_threshold = self.ensemble_config.get('confidence_threshold', 0.0)

    def _calculate_config_hash(self, config: Dict[str, Any]) -> int:
        """محاسبه هش از تنظیمات برای تشخیص تغییرات"""
        # استفاده از ساختار ساده‌تر برای محاسبه هش
        hash_items = []

        # پارامترهای اصلی و مهم
        keys_to_hash = [
            'voting_threshold', 'min_strategies_agreement', 'dynamic_weights',
            'trend_strategy_weight', 'mean_reversion_strategy_weight', 'breakout_strategy_weight',
            'momentum_strategy_weight', 'range_trading_strategy_weight', 'volatility_strategy_weight',
            'enabled_strategies', 'adaptive_threshold', 'bias_correction',
            'market_regime_awareness', 'strategy_refresh_on_update'
        ]

        # اضافه کردن کلیدهای مهم به هش
        for key in keys_to_hash:
            if key in config:
                hash_items.append(f"{key}:{config[key]}")

        # افزودن اطلاعات استراتژی‌های سفارشی
        if 'custom_strategies' in config:
            for i, strat in enumerate(config.get('custom_strategies', [])):
                if isinstance(strat, dict):
                    hash_items.append(f"custom_{i}:{strat.get('name')}:{strat.get('weight')}")

        # ایجاد هش ساده
        return hash("".join(hash_items))

    def update_config(self, new_config: Dict[str, Any]) -> bool:
        """
        به‌روزرسانی تنظیمات در زمان اجرا

        Args:
            new_config: تنظیمات جدید

        Returns:
            bool: نتیجه موفقیت‌آمیز بودن به‌روزرسانی
        """
        try:
            # بررسی وجود بخش ensemble_strategy در تنظیمات جدید
            if 'ensemble_strategy' not in new_config:
                logger.warning("بخش ensemble_strategy در تنظیمات جدید یافت نشد")
                return False

            # ذخیره تنظیمات قبلی برای بازگشت در صورت خطا
            old_config = copy.deepcopy(self.config)
            old_ensemble_config = copy.deepcopy(self.ensemble_config)

            # به‌روزرسانی تنظیمات
            self.config = copy.deepcopy(new_config)
            self.ensemble_config = self.config.get('ensemble_strategy', {})

            # محاسبه هش تنظیمات جدید
            new_config_hash = self._calculate_config_hash(self.ensemble_config)

            # اگر تنظیمات تغییر کرده‌اند
            if new_config_hash != self.last_config_hash:
                logger.info("تنظیمات سیستم ترکیبی استراتژی‌ها تغییر کرده‌اند - به‌روزرسانی پارامترها")

                # به‌روزرسانی پارامترهای اصلی
                self._init_core_parameters()

                # تنظیم مجدد استراتژی‌های فعال
                old_strategy_types = self.enabled_strategy_types.copy()
                self.enabled_strategy_types = self._parse_enabled_strategies()

                # مقایسه لیست استراتژی‌ها
                strategy_types_changed = set(old_strategy_types) != set(self.enabled_strategy_types)

                # بررسی نیاز به بازسازی استراتژی‌ها
                strategy_weights_changed = self._check_strategy_weights_changed(old_ensemble_config)
                custom_strategies_changed = self._check_custom_strategies_changed(old_ensemble_config)

                # بازسازی استراتژی‌ها در صورت نیاز
                if (strategy_types_changed or strategy_weights_changed or
                    custom_strategies_changed or self.strategy_refresh_on_update):
                    logger.info("بازسازی استراتژی‌ها به دلیل تغییر در لیست یا پارامترهای استراتژی‌ها")
                    self.refresh_strategies()

                # به‌روزرسانی هش تنظیمات
                self.last_config_hash = new_config_hash

                logger.info("تنظیمات سیستم ترکیبی استراتژی‌ها با موفقیت به‌روزرسانی شدند")
                return True
            else:
                logger.info("تنظیمات سیستم ترکیبی استراتژی‌ها تغییر نکرده‌اند - نیازی به به‌روزرسانی نیست")
                return True

        except Exception as e:
            logger.error(f"خطا در به‌روزرسانی تنظیمات سیستم ترکیبی استراتژی‌ها: {e}", exc_info=True)
            # بازگرداندن تنظیمات قبلی در صورت خطا
            self.config = old_config
            self.ensemble_config = old_ensemble_config
            return False

    def _check_strategy_weights_changed(self, old_config: Dict[str, Any]) -> bool:
        """بررسی تغییر در وزن‌های استراتژی‌ها"""
        weight_params = [
            'trend_strategy_weight', 'mean_reversion_strategy_weight',
            'breakout_strategy_weight', 'momentum_strategy_weight',
            'range_trading_strategy_weight', 'volatility_strategy_weight'
        ]

        for param in weight_params:
            old_value = old_config.get(param, 0)
            new_value = self.ensemble_config.get(param, 0)
            if abs(old_value - new_value) > 0.001:  # مقایسه با تلرانس کم برای اعداد اعشاری
                return True

        return False

    def _check_custom_strategies_changed(self, old_config: Dict[str, Any]) -> bool:
        """بررسی تغییر در استراتژی‌های سفارشی"""
        old_custom = old_config.get('custom_strategies', [])
        new_custom = self.ensemble_config.get('custom_strategies', [])

        # مقایسه تعداد
        if len(old_custom) != len(new_custom):
            return True

        # مقایسه محتوا
        for i, (old_strat, new_strat) in enumerate(zip(old_custom, new_custom)):
            if not isinstance(old_strat, dict) or not isinstance(new_strat, dict):
                return True

            if (old_strat.get('name') != new_strat.get('name') or
                abs(old_strat.get('weight', 0) - new_strat.get('weight', 0)) > 0.001):
                return True

            # مقایسه تغییرات تنظیمات
            old_changes = old_strat.get('config_changes', {})
            new_changes = new_strat.get('config_changes', {})
            if old_changes != new_changes:
                return True

        return False

    def refresh_strategies(self) -> None:
        """بازسازی استراتژی‌ها بر اساس تنظیمات فعلی"""
        try:
            # پاکسازی لیست استراتژی‌ها
            self.strategies = []

            # تنظیم مجدد سازندگان استراتژی
            self.strategy_builders = self._setup_strategy_builders()

            # ایجاد مجدد استراتژی‌ها
            self._create_strategies()

            # لاگ کردن استراتژی‌های فعال
            self._log_active_strategies()

            logger.info(f"استراتژی‌ها با موفقیت بازسازی شدند - تعداد استراتژی‌های فعال: {len(self.strategies)}")

        except Exception as e:
            logger.error(f"خطا در بازسازی استراتژی‌ها: {e}", exc_info=True)

    def _log_active_strategies(self) -> None:
        """لاگ کردن اطلاعات استراتژی‌های فعال"""
        try:
            active_strategies = [f"{s['name']}(w={s['weight']:.1f})" for s in self.strategies if s.get('enabled', True)]
            if active_strategies:
                logger.info(f"استراتژی‌های فعال: {', '.join(active_strategies)}")
            else:
                logger.warning("هیچ استراتژی فعالی یافت نشد!")
        except Exception as e:
            logger.error(f"خطا در لاگ کردن استراتژی‌های فعال: {e}")

    def _parse_enabled_strategies(self) -> List[StrategyType]:
        """تحلیل لیست استراتژی‌های فعال از پیکربندی"""
        enabled_str = self.ensemble_config.get('enabled_strategies', ['base', 'trend_following', 'mean_reversion', 'breakout'])

        if not enabled_str:
            logger.warning("لیست استراتژی‌های فعال خالی است - استفاده از استراتژی پایه")
            return [StrategyType.BASE]

        result = []
        str_to_type = {
            'base': StrategyType.BASE,
            'trend_following': StrategyType.TREND_FOLLOWING,
            'mean_reversion': StrategyType.MEAN_REVERSION,
            'breakout': StrategyType.BREAKOUT,
            'momentum': StrategyType.MOMENTUM,
            'range_trading': StrategyType.RANGE_TRADING,
            'volatility_based': StrategyType.VOLATILITY_BASED,
            'custom': StrategyType.CUSTOM
        }

        for strategy_name in enabled_str:
            strategy_name_lower = strategy_name.lower()
            if strategy_name_lower in str_to_type:
                result.append(str_to_type[strategy_name_lower])
            else:
                logger.warning(f"استراتژی ناشناخته: {strategy_name} - نادیده گرفته شد")

        # اگر هیچ استراتژی‌ای فعال نباشد، حداقل استراتژی پایه را اضافه کن
        if not result:
            logger.warning("هیچ استراتژی معتبری فعال نیست - استفاده از استراتژی پایه")
            result.append(StrategyType.BASE)

        return result

    def _setup_strategy_builders(self) -> Dict[StrategyType, Tuple[str, StrategyConfigBuilder, float]]:
        """تنظیم سازندگان استراتژی‌ها با وزن‌های تنظیم شده"""
        return {
            StrategyType.BASE: (
                'base_strategy',
                lambda c: c,
                1.0
            ),
            StrategyType.TREND_FOLLOWING: (
                'trend_following',
                self._create_trend_following_config,
                self.ensemble_config.get('trend_strategy_weight', 1.0)
            ),
            StrategyType.MEAN_REVERSION: (
                'mean_reversion',
                self._create_mean_reversion_config,
                self.ensemble_config.get('mean_reversion_strategy_weight', 1.0)
            ),
            StrategyType.BREAKOUT: (
                'breakout',
                self._create_breakout_config,
                self.ensemble_config.get('breakout_strategy_weight', 1.0)
            ),
            StrategyType.MOMENTUM: (
                'momentum',
                self._create_momentum_config,
                self.ensemble_config.get('momentum_strategy_weight', 0.9)
            ),
            StrategyType.RANGE_TRADING: (
                'range_trading',
                self._create_range_trading_config,
                self.ensemble_config.get('range_trading_strategy_weight', 0.8)
            ),
            StrategyType.VOLATILITY_BASED: (
                'volatility_based',
                self._create_volatility_based_config,
                self.ensemble_config.get('volatility_strategy_weight', 0.7)
            )
        }

    def _create_strategies(self) -> None:
        """ایجاد استراتژی‌های مختلف با تنظیمات متفاوت"""
        # ایجاد استراتژی‌های فعال
        created_count = 0
        error_count = 0

        for strategy_type in self.enabled_strategy_types:
            if strategy_type in self.strategy_builders:
                name, config_builder, weight = self.strategy_builders[strategy_type]

                try:
                    # ایجاد پیکربندی خاص این استراتژی
                    custom_config = config_builder(self.config)

                    # ایجاد شناسه یکتا برای استراتژی
                    strategy_id = f"{name}_{id(custom_config)}"

                    # ساخت استراتژی
                    strategy: StrategyInfo = {
                        'name': name,
                        'generator': SignalGenerator(custom_config),
                        'weight': weight,
                        'type': strategy_type,
                        'enabled': True,
                        'description': f"استراتژی برای رویکرد {name}",
                        'custom_config': custom_config,
                        'strategy_id': strategy_id
                    }

                    self.strategies.append(strategy)
                    created_count += 1
                    logger.debug(f"استراتژی ایجاد شد: {name}، وزن: {weight:.2f}")

                except Exception as e:
                    error_count += 1
                    logger.error(f"خطا در ایجاد استراتژی {name}: {e}", exc_info=True)

        # اضافه کردن استراتژی‌های سفارشی از پیکربندی
        custom_strategies = self.ensemble_config.get('custom_strategies', [])
        for index, custom_strat in enumerate(custom_strategies):
            if not isinstance(custom_strat, dict):
                logger.warning(f"استراتژی سفارشی با ساختار نامعتبر در ایندکس {index} - نادیده گرفته شد")
                continue

            try:
                name = custom_strat.get('name', f'custom_{index}')
                weight = custom_strat.get('weight', 1.0)
                custom_config = copy.deepcopy(self.config)

                # اعمال تغییرات روی پیکربندی
                if 'config_changes' in custom_strat:
                    self._apply_config_changes(custom_config, custom_strat['config_changes'])

                # ایجاد شناسه یکتا برای استراتژی سفارشی
                strategy_id = f"custom_{name}_{index}_{id(custom_config)}"

                strategy: StrategyInfo = {
                    'name': name,
                    'generator': SignalGenerator(custom_config),
                    'weight': weight,
                    'type': StrategyType.CUSTOM,
                    'enabled': True,
                    'description': custom_strat.get('description', f"استراتژی سفارشی: {name}"),
                    'custom_config': custom_config,
                    'strategy_id': strategy_id
                }

                self.strategies.append(strategy)
                created_count += 1
                logger.debug(f"استراتژی سفارشی ایجاد شد: {name}، وزن: {weight:.2f}")

            except Exception as e:
                error_count += 1
                logger.error(f"خطا در ایجاد استراتژی سفارشی {custom_strat.get('name', f'custom_{index}')}: {e}", exc_info=True)

        # لاگ خلاصه ایجاد استراتژی‌ها
        logger.info(f"ایجاد استراتژی‌ها: {created_count} ایجاد شده، {error_count} خطا")

    def _apply_config_changes(self, config: Dict[str, Any], changes: Dict[str, Any]) -> None:
        """اعمال تغییرات پیکربندی به صورت بازگشتی"""
        for key, value in changes.items():
            if isinstance(value, dict) and key in config and isinstance(config[key], dict):
                # اعمال تغییرات بازگشتی برای دیکشنری‌های تودرتو
                self._apply_config_changes(config[key], value)
            else:
                # اعمال مستقیم تغییر
                config[key] = value

    def _create_trend_following_config(self, base_config: Dict[str, Any]) -> Dict[str, Any]:
        """ساخت کانفیگ استراتژی دنبال‌کننده روند"""
        config_copy = copy.deepcopy(base_config)

        # تغییر پارامترهای Signal Generator
        signal_config = config_copy.get('signal_generation', {})

        # تقویت الگوهای دنبال‌کننده روند
        pattern_scores = signal_config.get('pattern_scores', {})
        trend_patterns = [
            'three_white_soldiers', 'three_black_crows', 'bullish_marubozu', 'bearish_marubozu',
            'macd_bullish_crossover', 'macd_bearish_crossover', 'trendline_break', 'higher_highs_higher_lows',
            'lower_highs_lower_lows', 'atr_breakout', 'big_candle_breakout', 'vwap_cross'
        ]

        for pattern in trend_patterns:
            if pattern in pattern_scores:
                pattern_scores[pattern] *= 1.5  # افزایش وزن الگوهای روند

        # کاهش وزن الگوهای بازگشتی
        reversal_patterns = [
            'hammer', 'inverted_hammer', 'morning_star', 'evening_star',
            'bullish_engulfing', 'bearish_engulfing', 'rsi_oversold_reversal',
            'rsi_overbought_reversal', 'doji', 'hanging_man', 'shooting_star'
        ]

        for pattern in reversal_patterns:
            if pattern in pattern_scores:
                pattern_scores[pattern] *= 0.7  # کاهش وزن الگوهای بازگشتی

        # تنظیمات تشخیص روند
        signal_config['trend_alignment_weight'] = 1.5  # افزایش اهمیت همراستایی روند
        signal_config['divergence_sensitivity'] = 0.6  # کاهش حساسیت به واگرایی‌ها
        signal_config['trend_detection_lookback'] = signal_config.get('trend_detection_lookback', 50) * 1.25  # افزایش دوره نگاه به عقب
        signal_config['trend_strength_threshold'] = signal_config.get('trend_strength_threshold', 25) * 0.95  # کاهش آستانه تشخیص روند

        # به‌روزرسانی تنظیمات
        signal_config['pattern_scores'] = pattern_scores
        config_copy['signal_generation'] = signal_config

        # اضافه کردن توضیح به تنظیمات
        if 'strategy_metadata' not in config_copy:
            config_copy['strategy_metadata'] = {}
        config_copy['strategy_metadata']['description'] = "استراتژی دنبال‌کننده روند - بهینه‌سازی شده برای بازارهای روندی"
        config_copy['strategy_metadata']['type'] = "trend_following"

        return config_copy

    def _create_mean_reversion_config(self, base_config: Dict[str, Any]) -> Dict[str, Any]:
        """ساخت کانفیگ استراتژی بازگشت به میانگین"""
        config_copy = copy.deepcopy(base_config)

        # تغییر پارامترهای Signal Generator
        signal_config = config_copy.get('signal_generation', {})

        # تقویت الگوهای بازگشتی
        pattern_scores = signal_config.get('pattern_scores', {})
        reversal_patterns = [
            'hammer', 'inverted_hammer', 'morning_star', 'evening_star',
            'bullish_engulfing', 'bearish_engulfing', 'rsi_oversold_reversal',
            'rsi_overbought_reversal', 'doji', 'hanging_man', 'shooting_star',
            'tweezer_top', 'tweezer_bottom', 'piercing_line', 'dark_cloud_cover'
        ]

        for pattern in reversal_patterns:
            if pattern in pattern_scores:
                pattern_scores[pattern] *= 1.5  # افزایش وزن الگوهای بازگشتی

        # تقویت الگوهای واگرایی
        divergence_patterns = [
            'rsi_bullish_divergence', 'rsi_bearish_divergence',
            'macd_bullish_divergence', 'macd_bearish_divergence',
            'stoch_bullish_divergence', 'stoch_bearish_divergence',
            'cci_bullish_divergence', 'cci_bearish_divergence'
        ]

        for pattern in divergence_patterns:
            if pattern in pattern_scores:
                pattern_scores[pattern] *= 1.7  # افزایش وزن واگرایی‌ها

        # تنظیمات حساسیت واگرایی
        signal_config['divergence_sensitivity'] = 0.5  # افزایش حساسیت به واگرایی‌ها
        signal_config['trend_alignment_weight'] = 0.8  # کاهش اهمیت همراستایی روند
        signal_config['overbought_threshold'] = signal_config.get('overbought_threshold', 70) - 3  # کاهش آستانه اشباع خرید
        signal_config['oversold_threshold'] = signal_config.get('oversold_threshold', 30) + 3  # افزایش آستانه اشباع فروش

        # اضافه کردن اندیکاتورهای مخصوص بازگشت به میانگین
        indicators = config_copy.get('indicators', {})
        indicators['bollinger_bands'] = {
            'enabled': True,
            'timeperiod': 20,
            'nbdevup': 2.0,
            'nbdevdn': 2.0,
            'matype': 0
        }
        config_copy['indicators'] = indicators

        # به‌روزرسانی تنظیمات
        signal_config['pattern_scores'] = pattern_scores
        config_copy['signal_generation'] = signal_config

        # اضافه کردن توضیح به تنظیمات
        if 'strategy_metadata' not in config_copy:
            config_copy['strategy_metadata'] = {}
        config_copy['strategy_metadata']['description'] = "استراتژی بازگشت به میانگین - بهینه‌سازی شده برای بازارهای رنجی"
        config_copy['strategy_metadata']['type'] = "mean_reversion"

        return config_copy

    def _create_breakout_config(self, base_config: Dict[str, Any]) -> Dict[str, Any]:
        """ساخت کانفیگ استراتژی شکست"""
        config_copy = copy.deepcopy(base_config)

        # تغییر پارامترهای Signal Generator
        signal_config = config_copy.get('signal_generation', {})

        # تقویت الگوهای شکست
        pattern_scores = signal_config.get('pattern_scores', {})
        breakout_patterns = [
            'broken_resistance', 'broken_support', 'horizontal_level_break',
            'trendline_break', 'large_bullish_candle', 'large_bearish_candle',
            'atr_breakout', 'vwap_breakout', 'triangle_break', 'flag_pole_break',
            'channel_break', 'consolidation_break'
        ]

        for pattern in breakout_patterns:
            if pattern in pattern_scores:
                pattern_scores[pattern] *= 1.8  # افزایش وزن الگوهای شکست

        # تقویت تأیید حجم
        signal_config['volume_multiplier_threshold'] = 1.1  # کاهش آستانه برای تشخیص افزایش حجم
        signal_config['volume_confirmation_weight'] = 1.5  # افزایش اهمیت تأیید حجم
        signal_config['lookback_periods'] = signal_config.get('lookback_periods', {})
        signal_config['lookback_periods']['resistance_support'] = 100  # افزایش دوره بررسی برای سطوح مقاومت و حمایت

        # تنظیمات اختصاصی شکست
        signal_config['breakout_atr_multiplier'] = 0.7  # حساسیت بیشتر به شکست‌ها
        signal_config['consolidation_detection_period'] = 14  # دوره تشخیص تثبیت قیمت

        # به‌روزرسانی تنظیمات
        signal_config['pattern_scores'] = pattern_scores
        config_copy['signal_generation'] = signal_config

        # اضافه کردن توضیح به تنظیمات
        if 'strategy_metadata' not in config_copy:
            config_copy['strategy_metadata'] = {}
        config_copy['strategy_metadata']['description'] = "استراتژی شکست - بهینه‌سازی شده برای گذار بازار"
        config_copy['strategy_metadata']['type'] = "breakout"

        return config_copy

    def _create_momentum_config(self, base_config: Dict[str, Any]) -> Dict[str, Any]:
        """ساخت کانفیگ استراتژی مومنتوم"""
        config_copy = copy.deepcopy(base_config)

        # تغییر پارامترهای Signal Generator
        signal_config = config_copy.get('signal_generation', {})

        # افزایش وزن الگوهای مومنتوم
        pattern_scores = signal_config.get('pattern_scores', {})
        momentum_patterns = [
            'macd_bullish_crossover', 'macd_bearish_crossover',
            'rsi_bullish_cross_50', 'rsi_bearish_cross_50',
            'stoch_bullish_cross', 'stoch_bearish_cross',
            'adx_strong_trend', 'consecutive_candles',
            'high_momentum_bar', 'accelerating_bars'
        ]

        for pattern in momentum_patterns:
            if pattern in pattern_scores:
                pattern_scores[pattern] *= 1.6  # افزایش وزن الگوهای مومنتوم

        # تغییر تنظیمات تشخیص مومنتوم
        signal_config['momentum_sensitivity'] = 0.8
        signal_config['adx_threshold'] = signal_config.get('adx_threshold', 25) * 0.9  # کاهش آستانه ADX

        # ضرایب شتاب قیمت
        signal_config['price_acceleration_factor'] = 1.2
        signal_config['momentum_lookback'] = 5

        # به‌روزرسانی تنظیمات
        signal_config['pattern_scores'] = pattern_scores
        config_copy['signal_generation'] = signal_config

        # اضافه کردن توضیح به تنظیمات
        if 'strategy_metadata' not in config_copy:
            config_copy['strategy_metadata'] = {}
        config_copy['strategy_metadata']['description'] = "استراتژی مومنتوم - بهینه‌سازی شده برای حرکات شتابدار قیمت"
        config_copy['strategy_metadata']['type'] = "momentum"

        return config_copy

    def _create_range_trading_config(self, base_config: Dict[str, Any]) -> Dict[str, Any]:
        """ساخت کانفیگ استراتژی معامله در محدوده"""
        config_copy = copy.deepcopy(base_config)

        # تغییر پارامترهای Signal Generator
        signal_config = config_copy.get('signal_generation', {})

        # الگوهای مناسب برای معامله در محدوده
        pattern_scores = signal_config.get('pattern_scores', {})
        range_patterns = [
            'overbought_pullback', 'oversold_bounce',
            'bollinger_bounce', 'range_rejection',
            'double_top', 'double_bottom',
            'upper_band_rejection', 'lower_band_bounce',
            'stoch_overbought_reversal', 'stoch_oversold_reversal'
        ]

        for pattern in range_patterns:
            if pattern in pattern_scores:
                pattern_scores[pattern] *= 1.7

        # کاهش وزن الگوهای شکست و روند
        trend_patterns = [
            'three_white_soldiers', 'three_black_crows',
            'trendline_break', 'broken_resistance', 'broken_support'
        ]

        for pattern in trend_patterns:
            if pattern in pattern_scores:
                pattern_scores[pattern] *= 0.6

        # تنظیمات محدوده قیمت
        signal_config['range_detection_period'] = 30
        signal_config['range_threshold_percent'] = 3.0
        signal_config['mean_reversion_emphasis'] = 1.5

        # به‌روزرسانی تنظیمات
        signal_config['pattern_scores'] = pattern_scores
        config_copy['signal_generation'] = signal_config

        # اضافه کردن توضیح به تنظیمات
        if 'strategy_metadata' not in config_copy:
            config_copy['strategy_metadata'] = {}
        config_copy['strategy_metadata']['description'] = "استراتژی معامله در محدوده - بهینه‌سازی شده برای بازارهای خنثی"
        config_copy['strategy_metadata']['type'] = "range_trading"

        return config_copy

    def _create_volatility_based_config(self, base_config: Dict[str, Any]) -> Dict[str, Any]:
        """ساخت کانفیگ استراتژی مبتنی بر نوسان"""
        config_copy = copy.deepcopy(base_config)

        # تغییر پارامترهای Signal Generator
        signal_config = config_copy.get('signal_generation', {})

        # الگوهای مناسب برای بازارهای پرنوسان
        pattern_scores = signal_config.get('pattern_scores', {})
        volatility_patterns = [
            'atr_breakout', 'wide_range_candle',
            'volatility_squeeze_release', 'keltner_breakout',
            'atr_expansion', 'bollinger_expansion',
            'high_volume_spike', 'gap_and_go'
        ]

        for pattern in volatility_patterns:
            if pattern in pattern_scores:
                pattern_scores[pattern] *= 1.8

        # تنظیمات نوسان
        signal_config['atr_period'] = 14
        signal_config['volatility_lookback'] = 30
        signal_config['volatility_expansion_threshold'] = 1.5
        signal_config['bollinger_band_width_threshold'] = 2.0

        # پارامترهای ریسک برای بازارهای پرنوسان
        risk_management = config_copy.get('risk_management', {})
        risk_management['default_stop_loss_atr_multiplier'] = risk_management.get('default_stop_loss_atr_multiplier', 1.5) * 1.2
        risk_management['volatility_adjustment_factor'] = 1.3
        config_copy['risk_management'] = risk_management

        # به‌روزرسانی تنظیمات
        signal_config['pattern_scores'] = pattern_scores
        config_copy['signal_generation'] = signal_config

        # اضافه کردن توضیح به تنظیمات
        if 'strategy_metadata' not in config_copy:
            config_copy['strategy_metadata'] = {}
        config_copy['strategy_metadata']['description'] = "استراتژی مبتنی بر نوسان - بهینه‌سازی شده برای دوره‌های پرنوسان"
        config_copy['strategy_metadata']['type'] = "volatility_based"

        return config_copy

    def update_strategy_weights(self, performance_data: Optional[Dict[str, Any]] = None) -> None:
        """
        به‌روزرسانی پویای وزن استراتژی‌ها بر اساس عملکرد قبلی

        Args:
            performance_data: داده‌های عملکرد استراتژی‌ها (اختیاری)
        """
        if not self.dynamic_weights:
            return

        if not performance_data and not self.performance_metrics['strategy_type']:
            logger.debug("داده‌های عملکرد برای تنظیم وزن‌ها در دسترس نیست")
            return

        try:
            # استفاده از داده‌های عملکرد خارجی یا داخلی
            data = performance_data if performance_data else self.performance_metrics['strategy_type']

            # اطمینان از وجود داده‌های کافی
            if len(data) < 2:
                return

            # محاسبه وزن‌های جدید بر اساس عملکرد نسبی
            total_performance = sum(data.values())
            if total_performance <= 0:
                logger.warning("مجموع عملکرد استراتژی‌ها صفر یا منفی است - بدون تغییر وزن")
                return

            # ذخیره وزن‌های قبلی برای مقایسه
            old_weights = {s['name']: s['weight'] for s in self.strategies}

            for strategy in self.strategies:
                strategy_type = strategy['type'].name.lower()
                if strategy_type in data:
                    # تنظیم وزن بر اساس عملکرد نسبی با محدودیت تغییرات
                    performance_ratio = data[strategy_type] / total_performance
                    current_weight = strategy['weight']

                    # تنظیم هوشمندانه وزن‌ها (اگر فعال باشد)
                    if self.smart_weight_adjustment:
                        # محاسبه وزن جدید با تأکید بیشتر بر استراتژی‌های موفق
                        # استفاده از یک تابع غیرخطی برای افزایش تأثیر استراتژی‌های موفق‌تر
                        performance_factor = performance_ratio ** 1.5  # توان برای افزایش تأثیر تفاوت‌های بزرگتر

                        # محدود کردن تغییرات به 20% وزن فعلی
                        max_change = current_weight * 0.2
                        weight_change = (performance_factor - 0.5) * max_change * 2  # تنظیم دامنه تغییرات
                        new_weight = current_weight + weight_change

                        # محدود کردن حداقل و حداکثر وزن
                        if new_weight < 0.3:  # حداقل وزن
                            new_weight = 0.3
                        elif new_weight > 2.0:  # حداکثر وزن
                            new_weight = 2.0
                    else:
                        # روش ساده قبلی
                        max_change = current_weight * 0.2
                        new_weight = current_weight * (0.8 + 0.4 * performance_ratio)

                        # اعمال محدودیت تغییرات
                        if new_weight > current_weight + max_change:
                            new_weight = current_weight + max_change
                        elif new_weight < current_weight - max_change:
                            new_weight = current_weight - max_change

                    # تنظیم وزن جدید
                    strategy['weight'] = new_weight

            # لاگ تغییرات وزن‌ها
            weight_changes = []
            for s in self.strategies:
                old_weight = old_weights.get(s['name'], 0)
                new_weight = s['weight']
                if abs(new_weight - old_weight) > 0.01:  # فقط تغییرات قابل توجه
                    change_pct = ((new_weight / old_weight) - 1) * 100 if old_weight > 0 else 0
                    weight_changes.append(f"{s['name']}: {old_weight:.2f} → {new_weight:.2f} ({change_pct:+.1f}%)")

            if weight_changes:
                logger.info(f"وزن‌های استراتژی بر اساس معیارهای عملکرد به‌روز شدند: {', '.join(weight_changes)}")

                # ذخیره وزن‌های جدید در تنظیمات برای استفاده در بازسازی‌های آینده
                strategy_type_to_config_key = {
                    'trend_following': 'trend_strategy_weight',
                    'mean_reversion': 'mean_reversion_strategy_weight',
                    'breakout': 'breakout_strategy_weight',
                    'momentum': 'momentum_strategy_weight',
                    'range_trading': 'range_trading_strategy_weight',
                    'volatility_based': 'volatility_strategy_weight'
                }

                # به‌روزرسانی وزن‌ها در strategy_builders
                for strategy in self.strategies:
                    strategy_type = strategy['type']
                    if strategy_type in self.strategy_builders:
                        name, config_builder, old_weight = self.strategy_builders[strategy_type]
                        # به‌روزرسانی وزن در سازنده استراتژی
                        self.strategy_builders[strategy_type] = (name, config_builder, strategy['weight'])
            else:
                logger.debug("هیچ تغییر قابل توجهی در وزن‌های استراتژی انجام نشد")

        except Exception as e:
            logger.error(f"خطا در به‌روزرسانی وزن‌های استراتژی: {e}", exc_info=True)

    def _update_performance_metrics(self, symbol: str, timeframe: str, strategy_name: str,
                                  signal_accuracy: float) -> None:
        """
        به‌روزرسانی متریک‌های عملکرد استراتژی

        Args:
            symbol: نماد ارز
            timeframe: تایم‌فریم
            strategy_name: نام استراتژی
            signal_accuracy: دقت سیگنال
        """
        # پیدا کردن استراتژی مربوطه
        strategy_type = None
        for strategy in self.strategies:
            if strategy['name'] == strategy_name:
                strategy_type = strategy['type'].name.lower()
                break

        if not strategy_type:
            return

        # به‌روزرسانی متریک‌ها
        try:
            # به‌روزرسانی متریک‌های مربوط به نوع استراتژی
            if strategy_type not in self.performance_metrics['strategy_type']:
                self.performance_metrics['strategy_type'][strategy_type] = signal_accuracy
            else:
                # میانگین متحرک نمایی
                prev_value = self.performance_metrics['strategy_type'][strategy_type]
                alpha = 0.2  # ضریب میانگین متحرک نمایی
                self.performance_metrics['strategy_type'][strategy_type] = \
                    alpha * signal_accuracy + (1 - alpha) * prev_value

            # به‌روزرسانی متریک‌های مربوط به نماد
            if symbol not in self.performance_metrics['symbol']:
                self.performance_metrics['symbol'][symbol] = {}

            if strategy_type not in self.performance_metrics['symbol'][symbol]:
                self.performance_metrics['symbol'][symbol][strategy_type] = signal_accuracy
            else:
                # میانگین متحرک نمایی
                prev_value = self.performance_metrics['symbol'][symbol][strategy_type]
                alpha = 0.3  # ضریب میانگین متحرک نمایی (بزرگتر برای داده‌های خاص نماد)
                self.performance_metrics['symbol'][symbol][strategy_type] = \
                    alpha * signal_accuracy + (1 - alpha) * prev_value

            # به‌روزرسانی متریک‌های مربوط به تایم‌فریم
            if timeframe not in self.performance_metrics['timeframe']:
                self.performance_metrics['timeframe'][timeframe] = {}

            if strategy_type not in self.performance_metrics['timeframe'][timeframe]:
                self.performance_metrics['timeframe'][timeframe][strategy_type] = signal_accuracy
            else:
                # میانگین متحرک نمایی
                prev_value = self.performance_metrics['timeframe'][timeframe][strategy_type]
                alpha = 0.25  # ضریب میانگین متحرک نمایی
                self.performance_metrics['timeframe'][timeframe][strategy_type] = \
                    alpha * signal_accuracy + (1 - alpha) * prev_value

            # لاگ کردن متریک‌های عملکرد اگر لاگ تصمیم‌گیری فعال باشد
            if self.log_decision_metrics:
                logger.debug(f"متریک عملکرد به‌روزرسانی شد: استراتژی={strategy_name}, نماد={symbol}, "
                            f"تایم‌فریم={timeframe}, دقت={signal_accuracy:.3f}")

        except Exception as e:
            logger.error(f"خطا در به‌روزرسانی متریک‌های عملکرد: {e}")

    def _get_dynamic_threshold(self, symbol: str, timeframe: str) -> float:
        """
        محاسبه آستانه پویا برای توافق بین استراتژی‌ها

        Args:
            symbol: نماد ارز
            timeframe: تایم‌فریم

        Returns:
            آستانه پویا
        """
        if not self.adaptive_threshold:
            return self.voting_threshold

        # بررسی وجود متریک‌های عملکرد برای این نماد
        threshold_adjustment = 0.0

        if symbol in self.performance_metrics['symbol']:
            # تنظیم آستانه بر اساس پراکندگی عملکرد استراتژی‌ها
            strategy_metrics = self.performance_metrics['symbol'][symbol]

            if len(strategy_metrics) >= 2:
                values = list(strategy_metrics.values())
                max_val = max(values)
                min_val = min(values)
                spread = max_val - min_val

                # آستانه پویا بر اساس انتشار عملکرد
                if spread > 0.4:  # اختلاف عملکرد زیاد (رقابت شدید)
                    threshold_adjustment = 0.15  # افزایش قابل توجه آستانه
                elif spread > 0.3:
                    threshold_adjustment = 0.1
                elif spread > 0.2:
                    threshold_adjustment = 0.05
                elif spread < 0.1:  # اختلاف عملکرد کم (اجماع)
                    threshold_adjustment = -0.05
                elif spread < 0.05:
                    threshold_adjustment = -0.1

        # تنظیم بر اساس تایم‌فریم
        if timeframe in self.performance_metrics['timeframe']:
            # تایم‌فریم‌های بزرگتر معمولاً معتبرتر هستند، آستانه را کاهش دهید
            if timeframe in ['1d', '4h']:
                threshold_adjustment -= 0.05
            # تایم‌فریم‌های کوچکتر اغلب پر نویزتر هستند، آستانه را افزایش دهید
            elif timeframe in ['5m', '1m']:
                threshold_adjustment += 0.05

        # محدود کردن آستانه نهایی به بازه معقول
        adjusted_threshold = self.voting_threshold + threshold_adjustment

        # محدود کردن آستانه به حداقل 0.4 و حداکثر 0.8
        if adjusted_threshold < 0.4:
            adjusted_threshold = 0.4
        elif adjusted_threshold > 0.8:
            adjusted_threshold = 0.8

        # لاگ کردن اطلاعات آستانه پویا اگر لاگ تصمیم‌گیری فعال باشد
        if self.log_decision_metrics and abs(threshold_adjustment) > 0.01:
            logger.debug(f"آستانه پویا برای {symbol}/{timeframe}: {self.voting_threshold:.2f} → {adjusted_threshold:.2f} "
                        f"(تنظیم: {threshold_adjustment:+.2f})")

        return adjusted_threshold

    def _apply_bias_correction(self, signals: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        اصلاح تورش سیگنال‌ها با توجه به عملکرد گذشته

        Args:
            signals: لیست سیگنال‌ها

        Returns:
            لیست سیگنال‌های اصلاح‌شده
        """
        if not self.bias_correction or not signals:
            return signals

        # بررسی داده‌های کافی برای اصلاح تورش
        if not self.performance_metrics['strategy_type']:
            return signals

        corrected_signals = []
        corrections_applied = 0

        # محاسبه عملکرد میانگین
        avg_performance = sum(self.performance_metrics['strategy_type'].values()) / \
                        len(self.performance_metrics['strategy_type']) if self.performance_metrics['strategy_type'] else 0

        for signal_data in signals:
            strategy_name = signal_data['strategy_name']
            strategy_type = None

            # پیدا کردن نوع استراتژی
            for strategy in self.strategies:
                if strategy['name'] == strategy_name:
                    strategy_type = strategy['type'].name.lower()
                    break

            if not strategy_type or strategy_type not in self.performance_metrics['strategy_type'] or avg_performance <= 0:
                corrected_signals.append(signal_data)
                continue

            # محاسبه ضریب اصلاح بر اساس عملکرد استراتژی
            strategy_performance = self.performance_metrics['strategy_type'][strategy_type]
            performance_ratio = strategy_performance / avg_performance

            correction_factor = 1.0
            # تنظیم ضریب اصلاح با روش پیشرفته‌تر
            if performance_ratio < 0.7:
                # استراتژی با عملکرد ضعیف
                correction_factor = 0.7 + 0.3 * performance_ratio  # حداقل 0.7
            elif performance_ratio > 1.3:
                # استراتژی با عملکرد عالی
                correction_factor = 1.0 + 0.3 * (performance_ratio - 1.0)  # حداکثر 1.3

            # اعمال ضریب اصلاح به وزن استراتژی
            if abs(correction_factor - 1.0) > 0.01:  # فقط اگر تغییر قابل توجهی وجود دارد
                corrected_signal_data = signal_data.copy()
                original_weight = corrected_signal_data['strategy_weight']
                corrected_signal_data['strategy_weight'] *= correction_factor

                # افزودن اطلاعات اصلاح برای شفافیت
                corrected_signal_data['bias_correction_applied'] = True
                corrected_signal_data['original_weight'] = original_weight
                corrected_signal_data['correction_factor'] = correction_factor

                corrected_signals.append(corrected_signal_data)
                corrections_applied += 1

                # لاگ کردن اصلاح اگر لاگ تصمیم‌گیری فعال باشد
                if self.log_decision_metrics:
                    logger.debug(f"اصلاح تورش اعمال شد: استراتژی={strategy_name}, "
                                f"وزن اصلی={original_weight:.2f}, وزن اصلاح‌شده={corrected_signal_data['strategy_weight']:.2f}, "
                                f"ضریب={correction_factor:.2f}")
            else:
                corrected_signals.append(signal_data)

        # لاگ خلاصه اصلاح‌ها
        if corrections_applied > 0 and self.log_decision_metrics:
            logger.debug(f"اصلاح تورش برای {corrections_applied} از {len(signals)} سیگنال اعمال شد")

        return corrected_signals

    def _apply_market_regime_awareness(self, signals: List[Dict[str, Any]], symbol: str, timeframes_data: Dict[str, pd.DataFrame]) -> List[Dict[str, Any]]:
        """
        اعمال آگاهی از رژیم بازار به سیگنال‌ها

        Args:
            signals: لیست سیگنال‌ها
            symbol: نماد ارز
            timeframes_data: داده‌های تایم‌فریم

        Returns:
            لیست سیگنال‌های تنظیم‌شده
        """
        if not self.market_regime_awareness or not signals or not timeframes_data:
            return signals

        # تشخیص رژیم بازار فعلی (ساده)
        market_regime = self._detect_simple_market_regime(timeframes_data)

        # اگر رژیم بازار تشخیص داده نشد، سیگنال‌ها را بدون تغییر برگردان
        if not market_regime:
            return signals

        # تنظیم وزن سیگنال‌ها بر اساس رژیم بازار
        regime_adjusted_signals = []

        # جدول تنظیم وزن‌ها بر اساس رژیم بازار
        regime_weight_adjustments = {
            'trending_up': {
                'trend_following': 1.3,
                'momentum': 1.2,
                'breakout': 1.1,
                'mean_reversion': 0.7,
                'range_trading': 0.6
            },
            'trending_down': {
                'trend_following': 1.3,
                'momentum': 1.2,
                'breakout': 1.1,
                'mean_reversion': 0.7,
                'range_trading': 0.6
            },
            'ranging': {
                'mean_reversion': 1.3,
                'range_trading': 1.4,
                'trend_following': 0.7,
                'momentum': 0.8,
                'breakout': 0.9
            },
            'volatile': {
                'breakout': 1.3,
                'volatility_based': 1.5,
                'mean_reversion': 0.8,
                'range_trading': 0.6,
                'trend_following': 0.9
            },
            'consolidating': {
                'breakout': 1.2,
                'range_trading': 1.1,
                'volatility_based': 0.9,
                'trend_following': 0.7,
                'momentum': 0.7
            }
        }

        for signal_data in signals:
            adjusted_signal = signal_data.copy()
            strategy_type = None

            # پیدا کردن نوع استراتژی
            for strategy in self.strategies:
                if strategy['name'] == signal_data['strategy_name']:
                    strategy_type = strategy['type'].name.lower()
                    break

            if strategy_type and market_regime in regime_weight_adjustments:
                # اعمال ضریب تنظیم بر اساس رژیم بازار
                adjustment_factor = regime_weight_adjustments[market_regime].get(strategy_type, 1.0)

                if abs(adjustment_factor - 1.0) > 0.01:  # فقط اگر تغییر قابل توجهی وجود دارد
                    original_weight = adjusted_signal['strategy_weight']
                    adjusted_signal['strategy_weight'] *= adjustment_factor

                    # افزودن اطلاعات تنظیم برای شفافیت
                    adjusted_signal['market_regime_adjustment_applied'] = True
                    adjusted_signal['original_weight'] = original_weight
                    adjusted_signal['market_regime'] = market_regime
                    adjusted_signal['regime_adjustment_factor'] = adjustment_factor

                    # لاگ کردن تنظیم اگر لاگ تصمیم‌گیری فعال باشد
                    if self.log_decision_metrics:
                        logger.debug(f"تنظیم رژیم بازار: استراتژی={signal_data['strategy_name']}({strategy_type}), "
                                    f"رژیم={market_regime}, وزن اصلی={original_weight:.2f}, "
                                    f"وزن تنظیم‌شده={adjusted_signal['strategy_weight']:.2f}, ضریب={adjustment_factor:.2f}")

            regime_adjusted_signals.append(adjusted_signal)

        # لاگ کردن رژیم بازار
        if self.log_decision_metrics:
            logger.debug(f"رژیم بازار تشخیص داده شده برای {symbol}: {market_regime}")

        return regime_adjusted_signals

    def _detect_simple_market_regime(self, timeframes_data: Dict[str, pd.DataFrame]) -> Optional[str]:
        """
        تشخیص ساده رژیم بازار بر اساس داده‌های تایم‌فریم بزرگتر

        Args:
            timeframes_data: داده‌های تایم‌فریم

        Returns:
            رشته توصیف‌کننده رژیم بازار یا None
        """
        # ترجیح بر استفاده از تایم‌فریم بزرگتر برای تشخیص رژیم بازار
        preferred_timeframes = ['1d', '4h', '1h', '15m']

        df = None
        selected_tf = None

        # انتخاب بزرگترین تایم‌فریم موجود از لیست ترجیحی
        for tf in preferred_timeframes:
            if tf in timeframes_data and len(timeframes_data[tf]) >= 20:  # حداقل 20 کندل نیاز است
                df = timeframes_data[tf]
                selected_tf = tf
                break

        # اگر هیچکدام از تایم‌فریم‌های ترجیحی موجود نباشند، از اولین تایم‌فریم موجود استفاده کن
        if df is None and timeframes_data:
            selected_tf = next(iter(timeframes_data))
            df = timeframes_data[selected_tf]

        if df is None or len(df) < 20:
            return None

        try:
            # محاسبه شاخص‌های اصلی برای تشخیص رژیم بازار
            df['close'] = df['close'].astype(float)

            # محاسبه میانگین متحرک برای تشخیص روند
            df['ma20'] = df['close'].rolling(window=20).mean()
            df['ma50'] = df['close'].rolling(window=50).mean() if len(df) >= 50 else df['close'].rolling(window=20).mean()

            # محاسبه ATR برای تشخیص نوسان
            df['high'] = df['high'].astype(float)
            df['low'] = df['low'].astype(float)

            df['tr1'] = abs(df['high'] - df['low'])
            df['tr2'] = abs(df['high'] - df['close'].shift())
            df['tr3'] = abs(df['low'] - df['close'].shift())
            df['tr'] = df[['tr1', 'tr2', 'tr3']].max(axis=1)
            df['atr14'] = df['tr'].rolling(window=14).mean()

            # محاسبه نسبت ATR به قیمت برای نرمال‌سازی
            df['atr_percent'] = df['atr14'] / df['close'] * 100

            # محاسبه پهنای باند بولینجر
            df['std20'] = df['close'].rolling(window=20).std()
            df['bb_upper'] = df['ma20'] + 2 * df['std20']
            df['bb_lower'] = df['ma20'] - 2 * df['std20']
            df['bb_width'] = (df['bb_upper'] - df['bb_lower']) / df['ma20'] * 100

            # گرفتن داده‌های اخیر
            recent = df.tail(10).copy()

            # تشخیص روند
            trend_up = recent['ma20'].iloc[-1] > recent['ma20'].iloc[0] * 1.005  # 0.5% افزایش در میانگین متحرک
            trend_down = recent['ma20'].iloc[-1] < recent['ma20'].iloc[0] * 0.995  # 0.5% کاهش در میانگین متحرک

            # تشخیص حرکت قیمت اخیر
            recent_move_up = recent['close'].iloc[-1] > recent['close'].iloc[-4] * 1.01  # 1% افزایش در 4 کندل اخیر
            recent_move_down = recent['close'].iloc[-1] < recent['close'].iloc[-4] * 0.99  # 1% کاهش در 4 کندل اخیر

            # تشخیص نوسان
            high_volatility = recent['atr_percent'].mean() > 3.0  # ATR بزرگتر از 3% قیمت
            low_volatility = recent['atr_percent'].mean() < 1.0  # ATR کمتر از 1% قیمت

            # تشخیص تثبیت
            narrowing_bands = recent['bb_width'].iloc[-1] < recent['bb_width'].iloc[0] * 0.8  # کاهش 20% در پهنای باند بولینجر

            # تصمیم‌گیری رژیم بازار بر اساس شاخص‌ها
            if high_volatility:
                return 'volatile'
            elif low_volatility and narrowing_bands:
                return 'consolidating'
            elif trend_up and recent_move_up:
                return 'trending_up'
            elif trend_down and recent_move_down:
                return 'trending_down'
            else:
                return 'ranging'

        except Exception as e:
            logger.warning(f"خطا در تشخیص رژیم بازار: {e}")
            return None

    async def generate_ensemble_signal(self, symbol: str, timeframes_data: Dict[str, pd.DataFrame]) -> Optional[SignalInfo]:
        """
        تولید سیگنال ترکیبی با استفاده از رأی‌گیری از استراتژی‌های مختلف

        Args:
            symbol: نماد ارز
            timeframes_data: دیکشنری داده‌های تایم‌فریم

        Returns:
            سیگنال نهایی
        """
        start_time = time.time()
        result: EnsembleSignalResult = {
            'signal': None,
            'voting_metrics': {},
            'contributing_strategies': [],
            'agreement_ratio': 0.0,
            'generation_time': 0.0,
            'error': None
        }

        try:
            # بررسی ورودی‌ها
            if not timeframes_data:
                logger.warning(f"هیچ داده‌ای برای {symbol} ارائه نشده است")
                result['error'] = "هیچ داده‌ای ارائه نشده است"
                return None

            if not self.strategies:
                logger.warning(f"هیچ استراتژی فعالی برای تولید سیگنال {symbol} وجود ندارد")
                result['error'] = "هیچ استراتژی فعالی وجود ندارد"
                return None

            # محاسبه آستانه پویا (اگر فعال باشد)
            timeframe = next(iter(timeframes_data.keys())) if timeframes_data else "unknown"
            dynamic_threshold = self._get_dynamic_threshold(symbol, timeframe)

            # لاگ شروع تولید سیگنال
            if self.log_decision_metrics:
                logger.debug(f"شروع تولید سیگنال ترکیبی برای {symbol}/{timeframe}")

            # جمع‌آوری سیگنال‌ها از همه استراتژی‌ها به صورت همزمان
            tasks = []
            for strategy in self.strategies:
                if strategy.get('enabled', True):
                    tasks.append(asyncio.create_task(
                        strategy['generator'].analyze_symbol(symbol, timeframes_data)
                    ))

            # منتظر نتیجه همه تحلیل‌ها بمان
            strategies_results = await asyncio.gather(*tasks, return_exceptions=True)

            # پردازش نتایج و حذف خطاها
            signals = []
            errors = 0
            for i, res in enumerate(strategies_results):
                if isinstance(res, Exception):
                    errors += 1
                    logger.error(f"خطا در استراتژی {self.strategies[i]['name']} برای {symbol}: {res}")
                    continue

                if res:  # اگر سیگنالی تولید شده باشد
                    signals.append({
                        'strategy_name': self.strategies[i]['name'],
                        'strategy_weight': self.strategies[i]['weight'],
                        'strategy_type': self.strategies[i]['type'].name.lower(),
                        'strategy_id': self.strategies[i].get('strategy_id', f"unknown_{i}"),
                        'signal': res
                    })

            if not signals:
                if self.signal_generated:
                    logger.info(f"هیچ سیگنالی توسط هیچ استراتژی برای {symbol} تولید نشد")
                result['error'] = "هیچ سیگنالی تولید نشد"
                return None

            # اصلاح تورش سیگنال‌ها (اگر فعال باشد)
            signals = self._apply_bias_correction(signals)

            # اعمال آگاهی از رژیم بازار (اگر فعال باشد)
            signals = self._apply_market_regime_awareness(signals, symbol, timeframes_data)

            # شمارش رأی‌ها برای جهت معامله
            long_votes = 0.0
            short_votes = 0.0
            long_signals = []
            short_signals = []

            for signal_data in signals:
                strategy_weight = signal_data['strategy_weight']
                signal = signal_data['signal']

                # آستانه اطمینان پایه (اگر فعال باشد)
                signal_confidence = signal.score.final_score
                if signal_confidence < self.confidence_threshold:
                    if self.log_decision_metrics:
                        logger.debug(f"سیگنال {signal_data['strategy_name']} برای {symbol} به دلیل امتیاز پایین ({signal_confidence:.2f} < {self.confidence_threshold}) نادیده گرفته شد")
                    continue

                if signal.direction == 'long':
                    long_votes += strategy_weight
                    long_signals.append(signal_data)
                else:  # 'short'
                    short_votes += strategy_weight
                    short_signals.append(signal_data)

            total_votes = long_votes + short_votes
            if total_votes == 0:
                logger.warning(f"هیچ رأی معتبری برای {symbol} وجود ندارد")
                result['error'] = "هیچ رأی معتبری وجود ندارد"
                return None

            # تعیین جهت نهایی بر اساس اکثریت وزن‌دار
            long_vote_ratio = long_votes / total_votes
            short_vote_ratio = short_votes / total_votes

            final_direction = 'long' if long_vote_ratio >= short_vote_ratio else 'short'
            winning_vote_ratio = max(long_vote_ratio, short_vote_ratio)

            # ذخیره متریک‌های رأی‌گیری
            result['voting_metrics'] = {
                'long_votes': long_votes,
                'short_votes': short_votes,
                'total_votes': total_votes,
                'long_vote_ratio': long_vote_ratio,
                'short_vote_ratio': short_vote_ratio,
                'winning_vote_ratio': winning_vote_ratio,
                'threshold': dynamic_threshold,
                'strategies_count': len(signals),
                'long_strategies': len(long_signals),
                'short_strategies': len(short_signals),
                'confidence_threshold': self.confidence_threshold,
                'errors': errors
            }

            # اطمینان از وجود توافق کافی
            if winning_vote_ratio < dynamic_threshold:
                if self.log_decision_metrics:
                    logger.info(f"توافق ناکافی برای {symbol}: {winning_vote_ratio:.2f} < {dynamic_threshold} (آستانه)")
                result['error'] = f"توافق ناکافی: {winning_vote_ratio:.2f} < {dynamic_threshold}"
                result['agreement_ratio'] = winning_vote_ratio
                return None

            # اطمینان از تعداد کافی استراتژی‌های موافق
            agreeing_strategies = sum(1 for s in signals if s['signal'].direction == final_direction)
            if agreeing_strategies < self.min_strategies_agreement:
                logger.info(f"تعداد استراتژی‌های موافق برای {symbol} کم است: {agreeing_strategies} < {self.min_strategies_agreement}")
                result['error'] = f"تعداد استراتژی‌های موافق کم است: {agreeing_strategies} < {self.min_strategies_agreement}"
                result['agreement_ratio'] = winning_vote_ratio
                return None

            # انتخاب بهترین سیگنال از بین سیگنال‌های هم‌جهت
            matching_signals = [s for s in signals if s['signal'].direction == final_direction]
            best_signal_data = max(matching_signals, key=lambda s: s['signal'].score.final_score)
            best_signal = best_signal_data['signal']
            result['contributing_strategies'] = [s['strategy_name'] for s in matching_signals]

            # بهبود امتیاز سیگنال نهایی با توجه به میزان توافق
            enhanced_score = SignalScore()
            enhanced_score.base_score = best_signal.score.base_score
            enhanced_score.timeframe_weight = best_signal.score.timeframe_weight
            enhanced_score.trend_alignment = best_signal.score.trend_alignment
            enhanced_score.volume_confirmation = best_signal.score.volume_confirmation
            enhanced_score.pattern_quality = best_signal.score.pattern_quality

            # افزودن فاکتور همگرایی بر اساس میزان توافق استراتژی‌ها
            # فرمول بهبودیافته برای محاسبه همگرایی با تأکید بیشتر بر توافق بالا
            agreement_boost = (winning_vote_ratio - 0.5) * 2  # تبدیل به بازه -1 تا 1
            agreement_boost = agreement_boost ** 2 if agreement_boost > 0 else 0  # فقط توافق مثبت را تقویت کن
            enhanced_score.confluence_score = best_signal.score.confluence_score + agreement_boost

            # محاسبه امتیاز نهایی بهبودیافته
            enhanced_score.final_score = enhanced_score.base_score * enhanced_score.timeframe_weight * \
                                        enhanced_score.trend_alignment * enhanced_score.volume_confirmation * \
                                        enhanced_score.pattern_quality * (1.0 + enhanced_score.confluence_score)

            # بررسی برخی ویژگی‌های پیشرفته برای بهبود سیگنال
            # 1. بهبود تایم‌فریم تأیید با توجه به سیگنال‌های همراستا
            confirmation_timeframes = set(best_signal.confirmation_timeframes)
            for s in matching_signals:
                signal = s['signal']
                for tf in signal.confirmation_timeframes:
                    confirmation_timeframes.add(tf)

            # 2. بهبود تخمین نقاط ورود/حد ضرر با میانگین وزن‌دار
            weighted_entry = 0
            weighted_stop = 0
            weighted_tp = 0
            total_weight = 0

            for s in matching_signals:
                signal = s['signal']
                weight = s['strategy_weight'] * signal.score.final_score

                weighted_entry += signal.entry_price * weight
                weighted_stop += signal.stop_loss * weight
                weighted_tp += signal.take_profit * weight
                total_weight += weight

            # اگر وزن کل صفر نباشد، از میانگین وزن‌دار استفاده کن
            improved_entry = weighted_entry / total_weight if total_weight > 0 else best_signal.entry_price
            improved_stop = weighted_stop / total_weight if total_weight > 0 else best_signal.stop_loss
            improved_tp = weighted_tp / total_weight if total_weight > 0 else best_signal.take_profit

            # محاسبه نسبت ریسک به ریوارد بهبودیافته
            improved_rr = 0.0
            if final_direction == 'long' and improved_entry > improved_stop:
                improved_rr = (improved_tp - improved_entry) / (improved_entry - improved_stop)
            elif final_direction == 'short' and improved_entry > improved_stop:
                improved_rr = (improved_entry - improved_tp) / (improved_stop - improved_entry)

            # ساخت سیگنال نهایی با امتیاز و پارامترهای بهبودیافته
            final_signal = SignalInfo(
                symbol=best_signal.symbol,
                timeframe=best_signal.timeframe,
                signal_type=f"ensemble_{best_signal.signal_type}",
                direction=final_direction,
                entry_price=improved_entry,
                stop_loss=improved_stop,
                take_profit=improved_tp,
                risk_reward_ratio=improved_rr if improved_rr > 0 else best_signal.risk_reward_ratio,
                timestamp=best_signal.timestamp,
                pattern_names=best_signal.pattern_names,
                score=enhanced_score,
                confirmation_timeframes=list(confirmation_timeframes)
            )

            result['signal'] = final_signal
            result['agreement_ratio'] = winning_vote_ratio

            # لاگ نتیجه نهایی
            logger.info(f"سیگنال ترکیبی برای {symbol} تولید شد: جهت={final_signal.direction}, "
                       f"امتیاز={final_signal.score.final_score:.2f}, "
                       f"توافق={winning_vote_ratio:.2f}, تعداد استراتژی‌ها={agreeing_strategies}")

            # به‌روزرسانی متریک‌های عملکرد
            for s in matching_signals:
                self._update_performance_metrics(
                    symbol=symbol,
                    timeframe=best_signal.timeframe,
                    strategy_name=s['strategy_name'],
                    signal_accuracy=s['signal'].score.final_score
                )

            # به‌روزرسانی وزن‌ها بر اساس عملکرد
            if self.dynamic_weights and len(signals) > 1:
                self.update_strategy_weights()

            return final_signal

        except Exception as e:
            error_msg = f"خطا در تولید سیگنال ترکیبی برای {symbol}: {e}"
            logger.error(error_msg, exc_info=True)
            result['error'] = str(e)
            return None

        finally:
            # ثبت زمان اجرا
            execution_time = time.time() - start_time
            result['generation_time'] = execution_time

            # لاگ کردن زمان اجرا اگر طولانی بود
            if execution_time > 1.0:
                logger.debug(f"زمان تولید سیگنال ترکیبی برای {symbol}: {execution_time:.3f} ثانیه")

    def get_active_strategies(self) -> List[Dict[str, Any]]:
        """
        دریافت لیست استراتژی‌های فعال

        Returns:
            لیست اطلاعات استراتژی‌های فعال
        """
        active_strategies = []

        for strategy in self.strategies:
            if strategy.get('enabled', True):
                # کپی اطلاعات استراتژی با حذف اشیاء غیرقابل سریالایز
                strategy_info = {
                    'name': strategy['name'],
                    'type': strategy['type'].name,
                    'weight': strategy['weight'],
                    'description': strategy.get('description', ''),
                    'strategy_id': strategy.get('strategy_id', '')
                }

                active_strategies.append(strategy_info)

        return active_strategies

    def get_performance_metrics(self) -> Dict[str, Any]:
        """
        دریافت متریک‌های عملکرد استراتژی‌ها

        Returns:
            دیکشنری متریک‌های عملکرد
        """
        return copy.deepcopy(self.performance_metrics)

    def get_ensemble_status(self) -> Dict[str, Any]:
        """
        دریافت وضعیت کلی سیستم ترکیبی استراتژی‌ها

        Returns:
            دیکشنری اطلاعات وضعیت
        """
        return {
            'active_strategies_count': sum(1 for s in self.strategies if s.get('enabled', True)),
            'total_strategies_count': len(self.strategies),
            'voting_threshold': self.voting_threshold,
            'min_strategies_agreement': self.min_strategies_agreement,
            'dynamic_weights': self.dynamic_weights,
            'adaptive_threshold': self.adaptive_threshold,
            'bias_correction': self.bias_correction,
            'market_regime_awareness': self.market_regime_awareness,
            'strategy_types': [s['type'].name for s in self.strategies],
            'last_config_hash': self.last_config_hash
        }

    def enable_strategy(self, strategy_id: str, enabled: bool = True) -> bool:
        """
        فعال یا غیرفعال کردن استراتژی با شناسه مشخص

        Args:
            strategy_id: شناسه استراتژی
            enabled: وضعیت فعال بودن

        Returns:
            نتیجه موفقیت‌آمیز بودن عملیات
        """
        for strategy in self.strategies:
            if strategy.get('strategy_id') == strategy_id or strategy.get('name') == strategy_id:
                strategy['enabled'] = enabled
                logger.info(f"استراتژی {strategy['name']} {'فعال' if enabled else 'غیرفعال'} شد")
                return True

        logger.warning(f"استراتژی با شناسه {strategy_id} یافت نشد")
        return False

    def reset_performance_metrics(self) -> None:
        """پاکسازی متریک‌های عملکرد و بازگشت به وضعیت اولیه"""
        self.performance_metrics = {
            'strategy_type': {},
            'symbol': {},
            'timeframe': {}
        }
        logger.info("متریک‌های عملکرد استراتژی‌ها بازنشانی شدند")

    async def test_strategies(self, symbol: str, timeframes_data: Dict[str, pd.DataFrame]) -> Dict[str, Any]:
        """
        تست تمام استراتژی‌ها برای یک نماد بدون تولید سیگنال ترکیبی

        Args:
            symbol: نماد ارز
            timeframes_data: داده‌های تایم‌فریم

        Returns:
            نتایج تست استراتژی‌ها
        """
        try:
            # جمع‌آوری سیگنال‌ها از همه استراتژی‌ها به صورت همزمان
            tasks = []
            for strategy in self.strategies:
                if strategy.get('enabled', True):
                    tasks.append(asyncio.create_task(
                        strategy['generator'].analyze_symbol(symbol, timeframes_data)
                    ))

            # منتظر نتیجه همه تحلیل‌ها بمان
            strategies_results = await asyncio.gather(*tasks, return_exceptions=True)

            # پردازش نتایج
            results = {}
            for i, res in enumerate(strategies_results):
                strategy_name = self.strategies[i]['name']
                strategy_type = self.strategies[i]['type'].name

                if isinstance(res, Exception):
                    results[strategy_name] = {
                        'error': str(res),
                        'type': strategy_type,
                        'success': False
                    }
                elif res:
                    results[strategy_name] = {
                        'direction': res.direction,
                        'score': res.score.final_score,
                        'entry_price': res.entry_price,
                        'stop_loss': res.stop_loss,
                        'take_profit': res.take_profit,
                        'risk_reward_ratio': res.risk_reward_ratio,
                        'timeframe': res.timeframe,
                        'type': strategy_type,
                        'success': True
                    }
                else:
                    results[strategy_name] = {
                        'message': 'بدون سیگنال',
                        'type': strategy_type,
                        'success': False
                    }

            return {
                'symbol': symbol,
                'timeframe': next(iter(timeframes_data.keys())) if timeframes_data else None,
                'timestamp': time.time(),
                'results': results
            }

        except Exception as e:
            logger.error(f"خطا در تست استراتژی‌ها برای {symbol}: {e}", exc_info=True)
            return {
                'symbol': symbol,
                'error': str(e),
                'success': False
            }