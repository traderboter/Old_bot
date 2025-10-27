"""
MLSignalIntegration: ماژول یکپارچه‌سازی برای اتصال اجزای هوش مصنوعی و یادگیری ماشین با مولد سیگنال موجود

این ماژول به عنوان یک پل عمل می‌کند و به منطق اصلی معاملاتی (SignalGenerator, SignalProcessor, TradeManager)
امکان استفاده از قابلیت‌های پیشرفته TradingBrainAI برای بهبود سیگنال و یادگیری مداوم را می‌دهد.
"""

import logging
import time
import traceback
import json
import asyncio
from datetime import datetime, timezone
from typing import Dict, Any, Optional, List, Tuple, Callable, Awaitable, Union

import pandas as pd
import numpy as np

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

try:
    from trading_brain_ai import TradingBrainAI
except ImportError:
    logging.error("خطا در وارد کردن TradingBrainAI. اطمینان حاصل کنید که trading_brain_ai.py در دسترس است و کلاس تعریف شده است.")
    # تعریف یک کلاس جایگزین برای جلوگیری از خطای NameError
    class TradingBrainAI:
        def __init__(self, *args, **kwargs):
            self.config = {}
            self.config_manager = type('ConfigManager', (), {'get_config': lambda *args: {}, 'get': lambda *args: {}})()
            pass

        def predict_signal(self, *args, **kwargs):
            logging.error("TradingBrainAI در دسترس نیست، predict_signal یک تابع موقت است.")
            return {'status': 'error', 'message': 'TradingBrainAI در دسترس نیست'}

        def register_trade_result(self, *args, **kwargs):
            logging.error("TradingBrainAI در دسترس نیست، register_trade_result یک تابع موقت است.")

        def get_config(self):
            return {}

# فرض می‌کنیم که SignalInfo و TradeResult در signal_generator.py تعریف شده‌اند
try:
    from signal_generator import SignalInfo, TradeResult, SignalScore
except ImportError:
    logging.error("خطا در وارد کردن SignalInfo یا TradeResult از signal_generator. اطمینان حاصل کنید که signal_generator.py در دسترس است.")
    # تعریف کلاس‌های موقت برای جلوگیری از خطای NameError
    class SignalScore:
        def __init__(self, *args, **kwargs):
            self.final_score = 0.0
            self.base_score = 0.0
            self.timeframe_weight = 1.0
            self.trend_alignment = 1.0
            self.volume_confirmation = 1.0
            self.pattern_quality = 1.0
            self.confluence_score = 0.0
            self.macd_analysis_score = 1.0

        def to_dict(self):
            return {}

    class SignalInfo:
        def __init__(self, *args, **kwargs):
            self.symbol = kwargs.get('symbol', "UNKNOWN")
            self.timeframe = kwargs.get('timeframe', "UNKNOWN")
            self.direction = kwargs.get('direction', "UNKNOWN")
            self.entry_price = kwargs.get('entry_price', 0.0)
            self.stop_loss = kwargs.get('stop_loss', 0.0)
            self.take_profit = kwargs.get('take_profit', 0.0)
            self.risk_reward_ratio = kwargs.get('risk_reward_ratio', 0.0)
            self.timestamp = kwargs.get('timestamp', datetime.now())
            self.pattern_names = kwargs.get('pattern_names', [])
            self.score = kwargs.get('score', SignalScore())
            self.confirmation_timeframes = kwargs.get('confirmation_timeframes', [])
            self.regime = kwargs.get('regime', None)
            self.is_reversal = kwargs.get('is_reversal', False)
            self.adapted_config = kwargs.get('adapted_config', None)
            self.correlated_symbols = kwargs.get('correlated_symbols', [])
            self.signal_id = kwargs.get('signal_id', "")
            self.market_context = kwargs.get('market_context', {})

        def to_dict(self):
            return {}

        def ensure_aware_timestamp(self):
            pass

        def generate_signal_id(self):
            pass

    class TradeResult:
        def __init__(self, *args, **kwargs):
            self.symbol = "UNKNOWN"
            self.direction = "UNKNOWN"
            self.entry_price = 0.0
            self.exit_price = 0.0
            self.stop_loss = 0.0
            self.take_profit = 0.0
            self.entry_time = datetime.now()
            self.exit_time = datetime.now()
            self.exit_reason = "UNKNOWN"
            self.profit_pct = 0.0
            self.profit_r = 0.0
            self.market_regime = None
            self.pattern_names = []
            self.timeframe = ""
            self.signal_score = 0.0
            self.trade_duration = None
            self.signal_type = ""
            self.profit_amount = 0.0
            self.position_size = 1.0

        def to_dict(self):
            return {}


class MLSignalIntegration:
    """
    ماژول یکپارچه‌سازی برای اتصال اجزای هوش مصنوعی و یادگیری ماشین با مولد سیگنال موجود

    این ماژول فاصله بین منطق سنتی تولید سیگنال و قابلیت‌های پیشرفته هوش مصنوعی و یادگیری ماشین
    ارائه شده توسط TradingBrainAI را پر می‌کند. سیگنال‌ها را بر اساس پیش‌بینی‌های هوش مصنوعی
    بهبود می‌بخشد و نتایج معاملات را برای یادگیری آنلاین بازخورد می‌دهد.
    """

    def __init__(self, signal_generator, trading_brain: TradingBrainAI):
        """
        مقداردهی اولیه با هر دو سیستم.

        Args:
            signal_generator: یک نمونه از SignalGenerator (یا EnsembleStrategy).
            trading_brain: یک نمونه از TradingBrainAI.
        """
        self.signal_generator = signal_generator
        self.trading_brain = trading_brain
        self.last_sync_time = None

        # دریافت پیکربندی اولیه
        self.config = {}
        self._init_config()

        # تنظیم زمان‌بندی همگام‌سازی
        self._sync_task = None
        self._shutdown_requested = asyncio.Event()

        logger.info("ماژول یکپارچه‌سازی سیگنال‌های هوش مصنوعی راه‌اندازی شد")

    def _init_config(self):
        """دریافت تنظیمات اولیه از TradingBrainAI"""
        try:
            if self.trading_brain and hasattr(self.trading_brain, 'config_manager'):
                self.config = self.trading_brain.config_manager.get_config()
            elif self.trading_brain and hasattr(self.trading_brain, 'config'):
                self.config = self.trading_brain.config
            else:
                logger.warning("دسترسی به پیکربندی TradingBrainAI امکان‌پذیر نیست. استفاده از پیکربندی پیش‌فرض.")
                self.config = {}

            # استخراج بخش تنظیمات مربوط به یکپارچه‌سازی ML
            self.ml_integration_config = self.config.get('ml_signal_integration', {})

            # تنظیم پارامترهای کلیدی از پیکربندی
            self.enabled = self.ml_integration_config.get('enabled', True)
            self.enhance_signals = self.ml_integration_config.get('enhance_signals', True)
            self.register_trade_results = self.ml_integration_config.get('register_trade_results', True)
            self.sync_interval_hours = self.ml_integration_config.get('sync_interval_hours', 1)

            # استخراج حداقل آستانه‌ی اطمینان برای هوش مصنوعی
            ai_config = self.config.get('trading_brain_ai', {})
            self.min_ml_confidence = ai_config.get('ensemble', {}).get('min_confidence_threshold', 0.6)
            self.min_signal_score = ai_config.get('signal_generation', {}).get('minimum_signal_score', 33)

            logger.debug(f"پیکربندی یکپارچه‌سازی ML بارگذاری شد: فعال={self.enabled}, بهبود سیگنال={self.enhance_signals}")

        except Exception as e:
            logger.error(f"خطا در مقداردهی پیکربندی MLSignalIntegration: {e}", exc_info=True)
            self.config = {}
            self.ml_integration_config = {}
            self.enabled = True
            self.enhance_signals = True
            self.register_trade_results = True
            self.sync_interval_hours = 1
            self.min_ml_confidence = 0.6
            self.min_signal_score = 33

    def update_config(self, new_config: Dict[str, Any]) -> None:
        """
        به‌روزرسانی پیکربندی در زمان اجرا

        Args:
            new_config: پیکربندی جدید
        """
        try:
            # ذخیره پیکربندی کامل
            self.config = new_config

            # استخراج بخش تنظیمات مربوط به یکپارچه‌سازی ML
            if 'ml_signal_integration' in new_config:
                old_config = self.ml_integration_config.copy()
                self.ml_integration_config = new_config.get('ml_signal_integration', {})

                # به‌روزرسانی پارامترهای کلیدی
                old_enabled = self.enabled
                self.enabled = self.ml_integration_config.get('enabled', True)
                old_enhance = self.enhance_signals
                self.enhance_signals = self.ml_integration_config.get('enhance_signals', True)
                old_register = self.register_trade_results
                self.register_trade_results = self.ml_integration_config.get('register_trade_results', True)
                old_interval = self.sync_interval_hours
                self.sync_interval_hours = self.ml_integration_config.get('sync_interval_hours', 1)

                # استخراج تنظیمات مربوط به هوش مصنوعی
                ai_config = new_config.get('trading_brain_ai', {})
                self.min_ml_confidence = ai_config.get('ensemble', {}).get('min_confidence_threshold', 0.6)
                self.min_signal_score = ai_config.get('signal_generation', {}).get('minimum_signal_score', 33)

                # لاگ تغییرات مهم
                if old_enabled != self.enabled:
                    logger.info(f"وضعیت فعال بودن یکپارچه‌سازی ML تغییر کرد: {old_enabled} -> {self.enabled}")

                if old_enhance != self.enhance_signals:
                    logger.info(f"وضعیت بهبود سیگنال ML تغییر کرد: {old_enhance} -> {self.enhance_signals}")

                if old_register != self.register_trade_results:
                    logger.info(f"وضعیت ثبت نتایج معامله ML تغییر کرد: {old_register} -> {self.register_trade_results}")

                if old_interval != self.sync_interval_hours:
                    logger.info(f"بازه همگام‌سازی ML تغییر کرد: {old_interval}h -> {self.sync_interval_hours}h")
                    # راه‌اندازی مجدد تسک همگام‌سازی
                    if self._sync_task:
                        # در صورت فعال بودن وضعیت همگام‌سازی، راه‌اندازی مجدد
                        asyncio.create_task(self.start_periodic_sync())

                logger.info("پیکربندی یکپارچه‌سازی سیگنال ML با موفقیت به‌روزرسانی شد")

        except Exception as e:
            logger.error(f"خطا در به‌روزرسانی پیکربندی یکپارچه‌سازی ML: {e}", exc_info=True)

    def enhance_signal(self, signal_info: SignalInfo, timeframes_data: Dict[str, pd.DataFrame]) -> Optional[SignalInfo]:
        """
        بهبود سیگنال از مولد با پیش‌بینی‌های هوش مصنوعی.

        Args:
            signal_info: سیگنال تولید شده توسط SignalGenerator.
            timeframes_data: دیکشنری داده‌های چند تایم‌فریمی {timeframe: df}.

        Returns:
            شیء SignalInfo بهبود یافته یا None اگر سیگنال رد شود.
        """
        # بررسی فعال بودن بهبود سیگنال
        if not self.enabled or not self.enhance_signals:
            logger.debug(f"بهبود سیگنال ML برای {signal_info.symbol} غیرفعال است.")
            return signal_info

        try:
            if not self.trading_brain:
                logger.debug(f"هوش مصنوعی معاملاتی برای بهبود سیگنال {signal_info.symbol} راه‌اندازی نشده یا غیرفعال است.")
                return signal_info

            if not timeframes_data:
                logger.warning(f"داده تایم‌فریم برای بهبود سیگنال {signal_info.symbol} ارائه نشده است. رد کردن بهبود.")
                return signal_info

            # دریافت پیش‌بینی هوش مصنوعی
            signal_result = self.trading_brain.predict_signal(timeframes_data)

            if signal_result['status'] != 'success' or not signal_result.get('signal'):
                logger.debug(f"پیش‌بینی سیگنال ML برای {signal_info.symbol} ناموفق بود: {signal_result.get('message', 'خطای نامشخص')}. رد کردن بهبود.")
                return signal_info

            ml_signal = signal_result.get('signal')
            if not ml_signal:
                logger.warning(f"پیش‌بینی سیگنال ML هیچ سیگنالی برای {signal_info.symbol} {signal_info.timeframe} برنگرداند. رد کردن بهبود.")
                return signal_info

            ml_confidence = ml_signal.get('confidence', 0.0)

            # بررسی آستانه‌ی اطمینان (استفاده از مقادیر محلی به‌روز شده)
            if ml_confidence < self.min_ml_confidence:
                logger.debug(f"اطمینان سیگنال ML ({ml_confidence:.2f}) زیر آستانه ({self.min_ml_confidence}) برای {signal_info.symbol}. رد کردن بهبود.")
                return signal_info

            # فقط در صورت تطابق جهت، بهبود انجام شود
            if signal_info.direction.lower() != ml_signal['direction'].lower():
                logger.debug(f"جهت سیگنال ML ({ml_signal['direction']}) با سیگنال اصلی ({signal_info.direction}) برای {signal_info.symbol} مطابقت ندارد. رد کردن بهبود.")
                return signal_info

            # --- بهبود سیگنال اصلی ---
            # ضریب افزایش بر اساس اطمینان
            confidence_boost_factor = 1.0 + ((ml_confidence - self.min_ml_confidence) / (1.0 - self.min_ml_confidence + 1e-9)) * 0.5
            signal_info.score.final_score *= confidence_boost_factor
            signal_info.score.final_score = min(signal_info.score.final_score, 100.0)

            # ضریب توافق مدل‌ها
            model_agreement_factor = 1.0 + (ml_signal.get('model_agreement', 1.0) - 1.0) * 0.3
            # اعمال به macd_analysis_score در صورت وجود، در غیر این صورت اعمال به base_score
            if hasattr(signal_info.score, 'macd_analysis_score'):
                signal_info.score.macd_analysis_score *= model_agreement_factor
            else:
                signal_info.score.base_score *= model_agreement_factor

            # بهینه‌سازی حد ضرر/سود
            if 'stop_loss' in ml_signal and 'take_profit' in ml_signal:
                ml_sl = ml_signal.get('stop_loss')
                ml_tp = ml_signal.get('take_profit')

                if ml_sl is not None and ml_tp is not None and abs(ml_sl) > 1e-9 and abs(ml_tp) > 1e-9:
                    entry = signal_info.entry_price
                    # بررسی معتبر بودن قیمت ورود
                    if entry is None or abs(entry) < 1e-9:
                        logger.warning(f"عدم امکان اعتبارسنجی SL/TP هوش مصنوعی برای {signal_info.symbol}: قیمت ورود نامعتبر ({entry}). حفظ SL/TP اصلی.")
                    else:
                        is_valid_ml_sl = (signal_info.direction == 'long' and ml_sl < entry) or (signal_info.direction == 'short' and ml_sl > entry)
                        is_valid_ml_tp = (signal_info.direction == 'long' and ml_tp > entry) or (signal_info.direction == 'short' and ml_tp < entry)

                        if is_valid_ml_sl and is_valid_ml_tp:
                            if ml_confidence >= 0.7:
                                # بهبود حد ضرر/سود با میانگین‌گیری
                                signal_info.stop_loss = (signal_info.stop_loss + ml_sl) / 2.0
                                signal_info.take_profit = (signal_info.take_profit + ml_tp) / 2.0

                                # محاسبه مجدد نسبت ریسک به پاداش
                                if signal_info.entry_price is not None and signal_info.stop_loss is not None and signal_info.take_profit is not None:
                                    risk_dist = abs(signal_info.entry_price - signal_info.stop_loss)
                                    reward_dist = abs(signal_info.take_profit - signal_info.entry_price)
                                    if risk_dist > 1e-9:
                                        signal_info.risk_reward_ratio = reward_dist / risk_dist
                                    else:
                                        signal_info.risk_reward_ratio = 0.0
                                logger.debug(f"سیگنال {signal_info.symbol} با SL/TP هوش مصنوعی بهبود یافت.")
                            else:
                                logger.debug(f"اطمینان هوش مصنوعی ({ml_confidence:.2f}) برای تنظیم SL/TP {signal_info.symbol} کافی نیست. حفظ مقادیر اصلی.")
                        else:
                            logger.warning(f"هوش مصنوعی SL/TP نامعتبر برای {signal_info.symbol} ارائه داد. SL: {ml_sl}, TP: {ml_tp}, ورود: {entry}.")

            # افزودن توصیه اندازه پوزیشن
            if 'position_sizing' in ml_signal:
                signal_info.market_context['ml_position_sizing_recommendation'] = ml_signal['position_sizing']
                logger.debug(f"توصیه اندازه پوزیشن هوش مصنوعی به زمینه سیگنال برای {signal_info.symbol} اضافه شد.")

            # افزودن متادیتای هوش مصنوعی
            signal_info.market_context['ml_enhanced'] = True
            signal_info.market_context['ml_confidence'] = float(ml_confidence)
            signal_info.market_context['ml_model_agreement'] = float(ml_signal.get('model_agreement', 1.0))
            signal_info.market_context['ml_details'] = ml_signal.get('model_details', {})

            logger.info(f"سیگنال {signal_info.symbol} {signal_info.timeframe} با پیش‌بینی هوش مصنوعی بهبود یافت (اطمینان: {ml_confidence:.2f}, امتیاز جدید: {signal_info.score.final_score:.2f})")

            # بررسی حداقل امتیاز سیگنال هوش مصنوعی
            if signal_info.score.final_score < self.min_signal_score:
                logger.debug(f"امتیاز سیگنال بهبود یافته برای {signal_info.symbol} ({signal_info.score.final_score:.2f}) زیر حداقل آستانه هوش مصنوعی ({self.min_signal_score}) است. رد سیگنال.")
                return None

            return signal_info

        except Exception as e:
            logger.error(f"خطا در بهبود سیگنال با هوش مصنوعی برای {signal_info.symbol}: {e}", exc_info=True)
            return signal_info

    def register_trade_result(self, trade_result: TradeResult) -> None:
        """
        ثبت نتیجه معامله در هر دو سیستم.

        Args:
            trade_result: نتیجه معامله بسته شده به عنوان یک شیء TradeResult.
        """
        # بررسی فعال بودن ثبت نتایج معامله
        if not self.enabled or not self.register_trade_results:
            logger.debug(f"ثبت نتیجه معامله ML برای {trade_result.symbol} غیرفعال است.")
            return

        try:
            # ثبت در مولد سیگنال
            if self.signal_generator and hasattr(self.signal_generator, 'register_trade_result'):
                self.signal_generator.register_trade_result(trade_result)
                logger.debug(f"نتیجه معامله برای {trade_result.symbol} در SignalGenerator ثبت شد.")

            # ثبت در هوش مصنوعی معاملاتی
            if self.trading_brain and hasattr(self.trading_brain, 'register_trade_result'):
                # آماده‌سازی داده‌ها برای هوش مصنوعی معاملاتی
                trade_result_for_ai = {
                    'symbol': trade_result.symbol,
                    'direction': trade_result.direction,
                    'entry_price': trade_result.entry_price,
                    'exit_price': trade_result.exit_price,
                    'position_size': getattr(trade_result, 'position_size', 1.0),
                    'profit_pct': trade_result.profit_pct,
                    'profit_amount': getattr(trade_result, 'profit_amount', None)
                }

                # تبدیل و استانداردسازی فرمت‌های زمانی
                try:
                    entry_time = trade_result.entry_time
                    exit_time = trade_result.exit_time

                    # تبدیل زمان‌ها به رشته
                    if hasattr(entry_time, 'isoformat'):
                        trade_result_for_ai['entry_time'] = entry_time.isoformat()
                    else:
                        trade_result_for_ai['entry_time'] = str(entry_time)

                    if hasattr(exit_time, 'isoformat'):
                        trade_result_for_ai['exit_time'] = exit_time.isoformat()
                    else:
                        trade_result_for_ai['exit_time'] = str(exit_time)

                except Exception as time_error:
                    logger.warning(f"خطا در تبدیل فرمت زمان برای {trade_result.symbol}: {time_error}. استفاده از مقادیر پیش‌فرض.")
                    trade_result_for_ai['entry_time'] = datetime.now().isoformat()
                    trade_result_for_ai['exit_time'] = datetime.now().isoformat()

                # فراخوانی تابع ثبت نتیجه در هوش مصنوعی
                self.trading_brain.register_trade_result(**trade_result_for_ai)
                logger.debug(f"نتیجه معامله برای {trade_result.symbol} در TradingBrainAI ثبت شد.")

            logger.info(f"نتیجه معامله برای {trade_result.symbol} در سیستم‌های یکپارچه ثبت شد")

        except Exception as e:
            logger.error(f"خطا در ثبت نتیجه معامله: {e}", exc_info=True)

    async def sync_systems(self) -> Dict[str, Any]:
        """
        همگام‌سازی داده‌های تاریخی معاملات بین SignalGenerator و TradingBrainAI.

        Returns:
            یک دیکشنری خلاصه نتیجه همگام‌سازی.
        """
        # بررسی فعال بودن همگام‌سازی
        if not self.enabled or self.sync_interval_hours <= 0:
            logger.debug("همگام‌سازی سیستم‌های ML غیرفعال است.")
            return {'status': 'disabled', 'message': 'همگام‌سازی غیرفعال است'}

        if not self.trading_brain or not self.signal_generator:
            logger.warning("امکان همگام‌سازی سیستم‌ها وجود ندارد: TradingBrainAI یا SignalGenerator راه‌اندازی نشده است.")
            return {'status': 'error', 'message': 'TradingBrainAI یا SignalGenerator راه‌اندازی نشده است'}

        logger.info("شروع همگام‌سازی سیستم‌های ML...")
        num_synced = 0

        try:
            if hasattr(self.signal_generator, 'adaptive_learning') and \
               hasattr(self.signal_generator.adaptive_learning, 'trade_history'):

                all_trades_sg = self.signal_generator.adaptive_learning.trade_history
                logger.debug(f"{len(all_trades_sg)} معامله تاریخی در SignalGenerator یافت شد.")

                trades_to_sync = []
                for trade in all_trades_sg:
                    exit_time = trade.exit_time
                    if isinstance(exit_time, datetime):
                        if exit_time.tzinfo is None:
                            exit_time = exit_time.replace(tzinfo=timezone.utc)
                    else:
                        logger.warning(f"رد همگام‌سازی برای معامله {getattr(trade, 'signal_id', 'نامشخص')}: فرمت زمان خروج نامعتبر.")
                        continue

                    if self.last_sync_time is None or exit_time > self.last_sync_time:
                        trades_to_sync.append(trade)

                logger.info(f"{len(trades_to_sync)} معامله تاریخی برای همگام‌سازی یافت شد.")

                for trade in trades_to_sync:
                    try:
                        trade_result_for_ai = {
                            'symbol': trade.symbol,
                            'direction': trade.direction,
                            'entry_price': trade.entry_price,
                            'exit_price': trade.exit_price,
                            'position_size': getattr(trade, 'position_size', 1.0),
                            'profit_pct': trade.profit_pct,
                            'profit_amount': getattr(trade, 'profit_amount', None)
                        }

                        # تبدیل و استانداردسازی فرمت‌های زمانی
                        entry_time = trade.entry_time
                        exit_time = trade.exit_time

                        if hasattr(entry_time, 'isoformat'):
                            trade_result_for_ai['entry_time'] = entry_time.isoformat()
                        else:
                            trade_result_for_ai['entry_time'] = str(entry_time)

                        if hasattr(exit_time, 'isoformat'):
                            trade_result_for_ai['exit_time'] = exit_time.isoformat()
                        else:
                            trade_result_for_ai['exit_time'] = str(exit_time)

                        self.trading_brain.register_trade_result(**trade_result_for_ai)
                        num_synced += 1
                    except Exception as register_error:
                        logger.error(f"خطا در ثبت معامله تاریخی {getattr(trade, 'signal_id', 'نامشخص')} در TradingBrainAI: {register_error}", exc_info=True)

                self.last_sync_time = datetime.now(timezone.utc)
                logger.info(f"{num_synced} معامله تاریخی با موفقیت از SignalGenerator به TradingBrainAI همگام‌سازی شد.")

                return {
                    'status': 'success',
                    'trades_synced': num_synced,
                    'last_sync_time': self.last_sync_time.isoformat()
                }
            else:
                logger.warning("امکان همگام‌سازی سیستم‌ها وجود ندارد: یادگیری تطبیقی یا تاریخچه معاملات SignalGenerator یافت نشد.")
                return {'status': 'warning', 'message': 'یادگیری تطبیقی یا تاریخچه معاملات SignalGenerator یافت نشد.'}

        except Exception as e:
            logger.error(f"خطای بحرانی در طول همگام‌سازی سیستم‌های ML: {e}", exc_info=True)
            return {
                'status': 'error',
                'message': f'خطای بحرانی در طول همگام‌سازی: {str(e)}',
                'error_details': traceback.format_exc()
            }

    async def start_periodic_sync(self) -> None:
        """شروع همگام‌سازی دوره‌ای بین سیستم‌ها"""
        # لغو تسک قبلی در صورت وجود
        if self._sync_task is not None and not self._sync_task.done():
            self._sync_task.cancel()

        if not self.enabled or self.sync_interval_hours <= 0:
            logger.info("همگام‌سازی دوره‌ای سیستم‌های ML غیرفعال است")
            return

        # ریست کردن رویداد درخواست توقف
        self._shutdown_requested.clear()

        # ایجاد تسک جدید
        self._sync_task = asyncio.create_task(self._sync_loop())
        logger.info(f"همگام‌سازی دوره‌ای سیستم‌های ML هر {self.sync_interval_hours} ساعت فعال شد")

    async def _sync_loop(self) -> None:
        """حلقه اصلی همگام‌سازی دوره‌ای"""
        try:
            while not self._shutdown_requested.is_set():
                # انجام همگام‌سازی
                sync_result = await self.sync_systems()

                if sync_result['status'] == 'success':
                    logger.info(f"همگام‌سازی موفق: {sync_result.get('trades_synced', 0)} معامله همگام شد")
                elif sync_result['status'] == 'warning':
                    logger.warning(f"همگام‌سازی با هشدار: {sync_result.get('message', '')}")
                elif sync_result['status'] == 'error':
                    logger.error(f"خطا در همگام‌سازی: {sync_result.get('message', '')}")

                # انتظار تا همگام‌سازی بعدی
                try:
                    await asyncio.wait_for(
                        self._shutdown_requested.wait(),
                        timeout=self.sync_interval_hours * 3600
                    )
                    # اگر به اینجا برسیم، یعنی رویداد _shutdown_requested تنظیم شده است
                    break
                except asyncio.TimeoutError:
                    # این به معنای پایان مهلت است، بدون تنظیم رویداد _shutdown_requested
                    pass

        except asyncio.CancelledError:
            logger.info("تسک همگام‌سازی دوره‌ای ML لغو شد")
        except Exception as e:
            logger.error(f"خطا در حلقه همگام‌سازی دوره‌ای ML: {e}", exc_info=True)

    async def shutdown(self) -> None:
        """توقف منابع و تسک‌های در حال اجرا"""
        logger.info("در حال توقف ماژول یکپارچه‌سازی ML...")

        # درخواست توقف حلقه همگام‌سازی
        self._shutdown_requested.set()

        # لغو تسک همگام‌سازی
        if self._sync_task is not None and not self._sync_task.done():
            try:
                self._sync_task.cancel()
                await asyncio.sleep(0.1)  # مهلت کوتاه برای لغو تسک
                logger.info("تسک همگام‌سازی ML لغو شد")
            except Exception as e:
                logger.error(f"خطا در لغو تسک همگام‌سازی ML: {e}")

        logger.info("ماژول یکپارچه‌سازی ML با موفقیت متوقف شد")