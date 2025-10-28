"""
ماژول signal_processor.py: پردازش سیگنال‌های معاملاتی چندتایم‌فریمی
این ماژول مسئول گردآوری داده‌ها، اجرای گزارش‌های سیگنال‌ها و مدیریت سیگنال‌های تولید شده است.
بهینه‌سازی شده با قابلیت به‌روزرسانی تنظیمات در زمان اجرا و پردازش هوشمند داده‌ها
"""
import logging
import asyncio
import time
import traceback
from typing import Dict, List, Optional, Any, Tuple, Set, cast, TypeVar
from datetime import datetime, timedelta
import threading
from functools import lru_cache
import random
from dataclasses import dataclass, field
# ایمپورت ماژول یکپارچه‌سازی ML
from ml_signal_integration import MLSignalIntegration
from signal_generator import SignalGenerator, SignalInfo
from market_data_fetcher import MarketDataFetcher

# استفاده شرطی از ماژول بر اساس وجود آن
try:
    from ensemble_strategy import StrategyEnsemble
    HAS_ENSEMBLE = True
except ImportError:
    HAS_ENSEMBLE = False
    class StrategyEnsemble:  # کلاس مجازی
        def __init__(self, *args, **kwargs):
            pass

# تنظیم لاگر
logger = logging.getLogger(__name__)

# تعریف نوع‌های داده
T = TypeVar('T')


@dataclass
class ProcessStats:
    """کلاس آمار پردازش سیگنال"""
    success_count: int = 0
    error_count: int = 0
    total_time: float = 0.0
    total_signals: int = 0

    @property
    def avg_time(self) -> float:
        """محاسبه میانگین زمان پردازش"""
        return self.total_time / max(self.success_count, 1)


@dataclass
class SymbolPriority:
    """کلاس اولویت‌بندی نمادها"""
    symbol: str
    last_process_time: float = 0.0
    has_error: bool = False
    has_signal: bool = False
    is_incomplete: bool = False
    error_count: int = 0

    def calculate_score(self, current_time: float) -> float:
        """محاسبه امتیاز اولویت‌بندی"""
        # زمان سپری شده از آخرین پردازش
        elapsed_time = current_time - self.last_process_time

        # اولویت‌های پایه
        base_score = elapsed_time

        # ضرایب
        incomplete_multiplier = 3.0 if self.is_incomplete else 1.0
        error_multiplier = min(2.0 + (self.error_count * 0.2), 4.0) if self.has_error else 1.0
        signal_penalty = 0.7 if self.has_signal else 1.0

        # امتیاز نهایی
        return base_score * incomplete_multiplier * error_multiplier * signal_penalty


class SignalProcessor:
    """
    پردازشگر سیگنال‌های معاملاتی بهینه‌سازی شده با قابلیت‌های پردازش موازی
    و مدیریت هوشمند منابع و به‌روزرسانی پویای تنظیمات
    """

    def __init__(self, config: Dict[str, Any],
                 market_data_fetcher: MarketDataFetcher,
                 signal_generator: SignalGenerator,
                 ml_integration: Optional[MLSignalIntegration] = None):
        """
        مقداردهی اولیه با تنظیمات و وهله‌های خارجی

        Args:
            config: دیکشنری تنظیمات
            market_data_fetcher: دریافت‌کننده داده‌های بازار
            signal_generator: تولیدکننده سیگنال
            ml_integration: ماژول یکپارچه‌سازی هوش مصنوعی (اختیاری)
        """
        self.ml_integration = ml_integration # ذخیره وهله یکپارچه‌سازی ML
        self.config = config
        self.signal_config = config.get('signal_processing', {})
        self.core_config = config.get('core', {})
        self.signal_generation_config = config.get('signal_generation', {})

        # استفاده از وهله‌های خارجی
        self.market_data_fetcher = market_data_fetcher
        self.signal_generator = signal_generator

        # تایم‌فریم‌ها
        self.timeframes = self.config.get('data_fetching', {}).get('timeframes', ['5m', '15m', '1h', '4h'])
        self.active_symbols: List[str] = []

        # محدودیت پردازش موازی
        self.max_concurrent_symbols = self.core_config.get('max_concurrent_analysis', 8)
        self.processing_semaphore = asyncio.Semaphore(self.max_concurrent_symbols)

        # محدودیت تعداد نمادهای قابل پردازش
        self.max_symbols_per_run = self.signal_config.get('max_symbols_per_run', 1000)

        # تنظیمات اعلان
        self.notification_config = self.signal_config.get('notification', {})

        # مدیریت سیگنال‌ها
        self.signal_history: Dict[str, Dict[str, Any]] = {}
        self.minimum_score = self.signal_generation_config.get('minimum_signal_score', 33.0)
        self.signal_max_age_minutes = self.signal_config.get('signal_max_age_minutes', 30)
        self.check_incomplete_interval = self.signal_config.get('check_incomplete_interval', 60)
        self.auto_forward_signals = self.signal_config.get('auto_forward_signals', True)
        self.ohlcv_limit_per_tf = self.signal_config.get('ohlcv_limit_per_tf', 500)
        self.trade_manager_callback = None

        # مدیریت سیگنال‌های ناقص
        self.incomplete_signals: Dict[str, Dict[str, Any]] = {}

        # قفل‌ها و تسک‌ها
        self._shutdown_requested = asyncio.Event()
        self._periodic_task: Optional[asyncio.Task] = None
        self._signals_lock = threading.RLock()
        self.is_running = False  # وضعیت اجرای فرآیند دوره‌ای

        # زمان گذاری آخرین پردازش نمادها برای مدیریت عادلانه
        self._symbol_priorities: Dict[str, SymbolPriority] = {}

        # آمار
        self.process_stats = ProcessStats()
        self.process_start_time = time.time()

        # مکانیزم backoff برای کنترل خطاها
        self._error_backoff = {
            'count': 0,
            'max_delay': 60,  # حداکثر تاخیر 60 ثانیه
            'base_delay': 2,  # تاخیر پایه 2 ثانیه
        }

        # انتخاب نوع استراتژی
        self.use_ensemble = self.config.get('ensemble_strategy', {}).get('enabled', True)
        self.ensemble_strategy = None

        if self.use_ensemble and HAS_ENSEMBLE:
            try:
                self.ensemble_strategy = StrategyEnsemble(config)
                logger.info("استراتژی ترکیبی با موفقیت راه‌اندازی شد")
            except Exception as e:
                logger.error(f"خطا در راه‌اندازی استراتژی ترکیبی: {e}", exc_info=True)
                self.use_ensemble = False

        logger.info(
            f"پردازشگر سیگنال راه‌اندازی شد. حداکثر نمادهای همزمان: {self.max_concurrent_symbols}, "
            f"حالت ترکیبی: {self.use_ensemble}, حداقل امتیاز سیگنال: {self.minimum_score}"
        )

    async def initialize(self) -> None:
        """راه‌اندازی اولیه"""
        logger.info("پردازشگر سیگنال راه‌اندازی شد (استفاده از کامپوننت‌های خارجی)")
        self.process_start_time = time.time()

    async def shutdown(self) -> None:
        """پاکسازی منابع"""
        logger.info("در حال خاموش کردن پردازشگر سیگنال...")

        self._shutdown_requested.set()
        self.is_running = False

        if self._periodic_task and not self._periodic_task.done():
            self._periodic_task.cancel()
            try:
                await self._periodic_task
            except asyncio.CancelledError:
                pass

        # پاکسازی تسک‌های در حال اجرا
        tasks = [t for t in asyncio.all_tasks() if t is not None and
                 (not t.done()) and t.get_name().startswith('signal_processor_')]

        if tasks:
            logger.info(f"لغو {len(tasks)} تسک پردازش سیگنال باقیمانده")
            for task in tasks:
                task.cancel()
            await asyncio.gather(*tasks, return_exceptions=True)

        logger.info("خاموش کردن پردازشگر سیگنال کامل شد")

        # گزارش آماری نهایی
        self._log_final_stats()

    def _log_final_stats(self) -> None:
        """ثبت آمار نهایی عملکرد"""
        total_runtime = time.time() - self.process_start_time
        hours = int(total_runtime // 3600)
        minutes = int((total_runtime % 3600) // 60)
        seconds = int(total_runtime % 60)

        logger.info(
            f"زمان اجرای پردازشگر سیگنال: {hours:02d}:{minutes:02d}:{seconds:02d} | "
            f"پردازش شده: {self.process_stats.success_count + self.process_stats.error_count} | "
            f"موفق: {self.process_stats.success_count} | "
            f"خطا: {self.process_stats.error_count} | "
            f"سیگنال‌ها: {self.process_stats.total_signals} | "
            f"میانگین زمان: {self.process_stats.avg_time:.2f}s"
        )

    def update_config(self, new_config: Dict[str, Any]) -> None:
        """
        به‌روزرسانی تنظیمات در زمان اجرا

        Args:
            new_config: تنظیمات جدید
        """
        logger.info("به‌روزرسانی تنظیمات پردازشگر سیگنال...")

        # ذخیره تنظیمات قبلی برای مقایسه
        old_signal_config = self.signal_config.copy()
        old_minimum_score = self.minimum_score
        old_max_age = self.signal_max_age_minutes
        old_ensemble_enabled = self.use_ensemble
        old_auto_forward = self.auto_forward_signals
        old_timeframes = self.timeframes.copy()

        # به‌روزرسانی تنظیمات کلی
        self.config = new_config
        self.signal_config = new_config.get('signal_processing', {})
        self.core_config = new_config.get('core', {})
        self.signal_generation_config = new_config.get('signal_generation', {})

        # به‌روزرسانی پارامترهای کلیدی
        self.minimum_score = self.signal_generation_config.get('minimum_signal_score', 33.0)
        self.signal_max_age_minutes = self.signal_config.get('signal_max_age_minutes', 30)
        self.check_incomplete_interval = self.signal_config.get('check_incomplete_interval', 60)
        self.auto_forward_signals = self.signal_config.get('auto_forward_signals', True)
        self.ohlcv_limit_per_tf = self.signal_config.get('ohlcv_limit_per_tf', 500)

        # به‌روزرسانی محدودیت نمادهای همزمان
        new_max_symbols = self.core_config.get('max_concurrent_analysis', 8)
        if new_max_symbols != self.max_concurrent_symbols:
            logger.info(f"به‌روزرسانی حداکثر نمادهای همزمان: {self.max_concurrent_symbols} -> {new_max_symbols}")
            self.max_concurrent_symbols = new_max_symbols
            # ایجاد مجدد سمافور با ظرفیت جدید
            self.processing_semaphore = asyncio.Semaphore(self.max_concurrent_symbols)

        # به‌روزرسانی حداکثر تعداد نمادها در هر اجرا
        self.max_symbols_per_run = self.signal_config.get('max_symbols_per_run', 1000)

        # به‌روزرسانی تایم‌فریم‌ها
        new_timeframes = new_config.get('data_fetching', {}).get('timeframes', ['5m', '15m', '1h', '4h'])
        if set(new_timeframes) != set(old_timeframes):
            logger.info(f"به‌روزرسانی تایم‌فریم‌ها: {old_timeframes} -> {new_timeframes}")
            self.timeframes = new_timeframes

        # به‌روزرسانی تنظیمات استراتژی ترکیبی
        new_ensemble_enabled = new_config.get('ensemble_strategy', {}).get('enabled', True)
        if new_ensemble_enabled != old_ensemble_enabled:
            logger.info(f"وضعیت استراتژی ترکیبی تغییر کرد: {old_ensemble_enabled} -> {new_ensemble_enabled}")
            self.use_ensemble = new_ensemble_enabled

            # بازسازی استراتژی ترکیبی در صورت نیاز
            if self.use_ensemble and HAS_ENSEMBLE and not self.ensemble_strategy:
                try:
                    self.ensemble_strategy = StrategyEnsemble(new_config)
                    logger.info("استراتژی ترکیبی با موفقیت بازسازی شد")
                except Exception as e:
                    logger.error(f"خطا در بازسازی استراتژی ترکیبی: {e}", exc_info=True)
                    self.use_ensemble = False
            elif self.ensemble_strategy and not self.use_ensemble:
                logger.info("استراتژی ترکیبی غیرفعال شد")

        # به‌روزرسانی استراتژی ترکیبی فعلی اگر وجود دارد
        elif self.ensemble_strategy and self.use_ensemble and HAS_ENSEMBLE:
            if hasattr(self.ensemble_strategy, 'update_config'):
                self.ensemble_strategy.update_config(new_config)
                logger.info("تنظیمات استراتژی ترکیبی به‌روزرسانی شد")

        # لاگ تغییرات مهم پارامترها
        if old_minimum_score != self.minimum_score:
            logger.info(f"تغییر حداقل امتیاز سیگنال: {old_minimum_score} -> {self.minimum_score}")

        if old_max_age != self.signal_max_age_minutes:
            logger.info(f"تغییر حداکثر عمر سیگنال: {old_max_age} -> {self.signal_max_age_minutes} دقیقه")

        if old_auto_forward != self.auto_forward_signals:
            logger.info(f"وضعیت ارسال خودکار سیگنال تغییر کرد: {old_auto_forward} -> {self.auto_forward_signals}")

        # پاکسازی سیگنال‌های منقضی با قوانین جدید
        self.clean_expired_incomplete_signals()

        logger.info("به‌روزرسانی تنظیمات پردازشگر سیگنال کامل شد")

    def register_trade_manager_callback(self, callback_function) -> None:
        """
        ثبت تابع callback برای ارسال سیگنال به TradeManager

        Args:
            callback_function: تابع callback که SignalInfo را دریافت می‌کند
        """
        self.trade_manager_callback = callback_function
        logger.info("کالبک مدیریت معاملات با موفقیت ثبت شد")

    def set_active_symbols(self, symbols: List[str]) -> None:
        """
        تنظیم لیست نمادهای فعال

        Args:
            symbols: لیست نمادها
        """
        if not isinstance(symbols, list):
            logger.error(f"فرمت نامعتبر برای نمادها: {type(symbols)}")
            return

        with self._signals_lock:
            # ذخیره تعداد قبلی برای مقایسه
            old_count = len(self.active_symbols)
            self.active_symbols = symbols.copy()  # کپی برای جلوگیری از تغییرات ناخواسته

            # به‌روزرسانی اولویت نمادها
            self._update_symbol_priorities()

            # لاگ تغییرات
            if old_count != len(symbols):
                logger.info(f"لیست نمادهای فعال به‌روزرسانی شد: {old_count} -> {len(symbols)} نماد")
            else:
                logger.info(f"لیست نمادهای فعال بازنویسی شد: {len(symbols)} نماد")

    def _update_symbol_priorities(self) -> None:
        """به‌روزرسانی لیست اولویت نمادها"""
        # حذف نمادهای قدیمی که دیگر فعال نیستند
        active_set = set(self.active_symbols)
        old_keys = set(self._symbol_priorities.keys()) - active_set
        for key in old_keys:
            del self._symbol_priorities[key]

        # اضافه کردن نمادهای جدید
        for symbol in active_set:
            if symbol not in self._symbol_priorities:
                self._symbol_priorities[symbol] = SymbolPriority(symbol=symbol)

        # به‌روزرسانی وضعیت سیگنال‌های ناقص
        for symbol in self.incomplete_signals:
            if symbol in self._symbol_priorities:
                self._symbol_priorities[symbol].is_incomplete = True

        # به‌روزرسانی وضعیت سیگنال‌های موجود
        for symbol in self.signal_history:
            if symbol in self._symbol_priorities:
                self._symbol_priorities[symbol].has_signal = True

    async def _apply_backoff_strategy(self, success: bool = True) -> None:
        """
        اعمال استراتژی backoff برای کنترل خطاها

        Args:
            success: آیا عملیات موفق بوده است
        """
        if success:
            # ریست کانتر در صورت موفقیت
            self._error_backoff['count'] = 0
            return

        # افزایش کانتر خطا
        self._error_backoff['count'] += 1

        # محاسبه تاخیر با الگوریتم exponential backoff
        delay = min(
            self._error_backoff['base_delay'] * (2 ** (self._error_backoff['count'] - 1)),
            self._error_backoff['max_delay']
        )

        # اضافه کردن jitter برای جلوگیری از thundering herd
        jitter = random.uniform(0, 0.1 * delay)
        total_delay = delay + jitter

        logger.warning(f"اعمال تاخیر {total_delay:.2f} ثانیه‌ای پس از خطا (تعداد: {self._error_backoff['count']})")
        await asyncio.sleep(total_delay)

    async def process_symbol(self, symbol: str, force_refresh: bool = False,
                             priority: bool = False) -> Optional[SignalInfo]:
        """
        پردازش کامل یک نماد با اولویت‌بندی و اطلاعات دقیق‌تر خطا

        Args:
            symbol: نماد ارز
            force_refresh: اجبار به دریافت مجدد داده‌ها
            priority: اولویت بالا برای پردازش

        Returns:
            سیگنال تولیدشده یا None
        """
        # نام‌گذاری تسک برای مدیریت بهتر
        current_task = asyncio.current_task()
        if current_task:
            current_task.set_name(f"signal_processor_{symbol}_{int(time.time())}")

        logger.info(f"[پردازشگر] شروع پردازش نماد {symbol} (بازنویسی={force_refresh}, اولویت={priority})")
        start_time = time.time()
        symbol_priority = self._get_symbol_priority(symbol)
        is_successful = False
        signal = None

        try:
            # دریافت داده‌ها با limit مشخص
            limit_needed = self.ohlcv_limit_per_tf

            # استفاده از semaphore برای محدودیت پردازش موازی
            async with self.processing_semaphore:
                logger.debug(f"[پردازشگر] پردازش {symbol} شروع شد (اولویت: {priority})")

                # دریافت داده‌های چندتایم‌فریمی
                timeframes_data = await self.market_data_fetcher.get_multi_timeframe_data(
                    symbol, self.timeframes, force_refresh, limit_per_tf=limit_needed
                )

                # بررسی داده‌های معتبر
                valid_timeframes = [tf for tf, df in timeframes_data.items() if df is not None and not df.empty]
                if not valid_timeframes:
                    # ذخیره در لیست ناقص‌ها برای بررسی بعدی
                    logger.warning(f"[پردازشگر] هیچ داده معتبری برای {symbol} در هیچ تایم‌فریمی یافت نشد")
                    with self._signals_lock:
                        self.incomplete_signals[symbol] = {
                            'reason': 'no_valid_data',
                            'timestamp': datetime.now().astimezone()
                        }
                        if symbol in self._symbol_priorities:
                            self._symbol_priorities[symbol].is_incomplete = True
                            self._symbol_priorities[symbol].has_error = True
                            self._symbol_priorities[symbol].error_count += 1

                    # اعمال استراتژی backoff
                    await self._apply_backoff_strategy(success=False)
                    return None

                if len(valid_timeframes) < len(self.timeframes):
                    missing_tfs = set(self.timeframes) - set(valid_timeframes)
                    logger.debug(f"[پردازشگر] داده‌های ناقص برای {symbol} در تایم‌فریم‌های: {missing_tfs}")

                # تولید سیگنال با روش مناسب
                if self.use_ensemble and self.ensemble_strategy:
                    logger.debug(f"[پردازشگر] تولید سیگنال ترکیبی برای {symbol}...")
                    signal = await self.ensemble_strategy.generate_ensemble_signal(symbol, timeframes_data)
                else:
                    logger.debug(f"[پردازشگر] تولید سیگنال استاندارد برای {symbol}...")
                    signal = await self.signal_generator.analyze_symbol(symbol, timeframes_data)

                if signal:
                    # غنی‌سازی سیگنال توسط MLIntegration
                    if self.ml_integration and self.config.get('ml_signal_integration', {}).get('enhance_signals', True):
                        logger.debug(f"[پردازشگر] غنی‌سازی سیگنال برای {symbol} با ML...")
                        signal = self.ml_integration.enhance_signal(signal, timeframes_data)
                        if signal:
                            logger.debug(
                                f"[پردازشگر] سیگنال برای {symbol} توسط ML غنی‌سازی شد. امتیاز نهایی: {signal.score.final_score:.2f}")
                        else:
                            logger.debug(f"[پردازشگر] سیگنال برای {symbol} در فرآیند غنی‌سازی ML رد شد")
                            return None

                    # ذخیره سیگنال در تاریخچه
                    process_time = time.time() - start_time
                    with self._signals_lock:
                        self.signal_history[symbol] = {
                            'timestamp': datetime.now().astimezone(),
                            'signal': signal,
                            'processing_time': process_time
                        }

                        if symbol in self._symbol_priorities:
                            self._symbol_priorities[symbol].has_signal = True
                            self._symbol_priorities[symbol].has_error = False
                            self._symbol_priorities[symbol].is_incomplete = False
                            self._symbol_priorities[symbol].last_process_time = time.time()

                    # بررسی اعتبار سیگنال برای ارسال
                    if self.auto_forward_signals and self.trade_manager_callback:
                        logger.info(
                            f"[پردازشگر] تلاش برای ارسال سیگنال {symbol} به مدیریت معاملات. "
                            f"امتیاز: {signal.score.final_score:.2f}, جهت: {signal.direction}"
                        )
                        await self._forward_signal_if_valid(signal)

                    is_successful = True
                    self.process_stats.total_signals += 1
                    logger.info(
                        f"[پردازشگر] پردازش نماد {symbol} با موفقیت انجام شد. "
                        f"سیگنال تولید شد: {signal.direction} - امتیاز: {signal.score.final_score:.2f}"
                    )
                    return signal
                else:
                    # سیگنال معتبری تولید نشد
                    logger.debug(f"[پردازشگر] هیچ سیگنال معتبری برای {symbol} تولید نشد")

                    # به‌روزرسانی اطلاعات اولویت نماد
                    with self._signals_lock:
                        if symbol in self._symbol_priorities:
                            self._symbol_priorities[symbol].last_process_time = time.time()
                            self._symbol_priorities[symbol].has_error = False
                            self._symbol_priorities[symbol].is_incomplete = False

                    # حذف از سیگنال‌های ناقص اگر وجود دارد
                    with self._signals_lock:
                        if symbol in self.incomplete_signals:
                            del self.incomplete_signals[symbol]

                    is_successful = True
                    # ریست کانتر خطا در صورت موفقیت
                    await self._apply_backoff_strategy(success=True)
                    return None

        except asyncio.CancelledError:
            logger.debug(f"[پردازشگر] پردازش {symbol} لغو شد")
            raise
        except Exception as e:
            logger.error(f"[پردازشگر] خطا در پردازش نماد {symbol}: {e}", exc_info=True)

            # ذخیره در لیست ناقص‌ها با جزئیات خطا
            with self._signals_lock:
                self.incomplete_signals[symbol] = {
                    'reason': f"خطا: {str(e)}",
                    'timestamp': datetime.now().astimezone(),
                    'error_traceback': traceback.format_exc()
                }

                if symbol in self._symbol_priorities:
                    self._symbol_priorities[symbol].has_error = True
                    self._symbol_priorities[symbol].is_incomplete = True
                    self._symbol_priorities[symbol].error_count += 1

            # اعمال استراتژی backoff
            await self._apply_backoff_strategy(success=False)
            return None
        finally:
            # به‌روزرسانی آمار
            process_time = time.time() - start_time
            if is_successful:
                self.process_stats.success_count += 1
            else:
                self.process_stats.error_count += 1
            self.process_stats.total_time += process_time

            if symbol_priority:
                symbol_priority.last_process_time = time.time()

            logger.debug(
                f"[پردازشگر] پردازش {symbol} در {process_time:.3f} ثانیه به پایان رسید. موفقیت: {is_successful}"
            )

    def _get_symbol_priority(self, symbol: str) -> SymbolPriority:
        """دریافت اطلاعات اولویت یک نماد با ایجاد آن در صورت نیاز"""
        with self._signals_lock:
            if symbol not in self._symbol_priorities:
                self._symbol_priorities[symbol] = SymbolPriority(symbol=symbol)
            return self._symbol_priorities[symbol]

    async def _forward_signal_if_valid(self, signal: SignalInfo) -> bool:
        """
        بررسی اعتبار سیگنال و ارسال به TradeManager در صورت معتبر بودن

        Args:
            signal: سیگنال برای بررسی و ارسال

        Returns:
            True اگر ارسال موفق بود
        """
        try:
            logger.info(
                f"[ارسال] شروع فرآیند بررسی اعتبار و ارسال سیگنال برای {signal.symbol}, "
                f"جهت: {signal.direction}, امتیاز: {signal.score.final_score:.2f}"
            )

            # بررسی وجود callback
            if not self.trade_manager_callback:
                logger.error(
                    f"[ارسال] خطا: تابع callback مدیریت معاملات ثبت نشده است! سیگنال {signal.symbol} ارسال نمی‌شود."
                )
                return False

            # بررسی آستانه امتیاز - استفاده مستقیم از self.minimum_score که در هر بار به‌روزرسانی تنظیمات به‌روز می‌شود
            if signal.score.final_score < self.minimum_score:
                logger.info(
                    f"[ارسال] سیگنال {signal.symbol} به دلیل امتیاز پایین "
                    f"({signal.score.final_score:.2f} < {self.minimum_score}) ارسال نشد"
                )
                return False

            # بررسی اعتبار سیگنال
            is_valid, reason = await self.check_signal_still_valid(signal)
            logger.info(f"[ارسال] نتیجه بررسی اعتبار سیگنال {signal.symbol}: {is_valid}, دلیل: {reason}")

            if not is_valid:
                logger.info(f"[ارسال] سیگنال برای {signal.symbol} معتبر نیست و ارسال نمی‌شود: {reason}")
                return False

            try:
                logger.info(
                    f"[ارسال] ارسال سیگنال {signal.symbol} به مدیریت معاملات. "
                    f"قیمت ورود: {signal.entry_price}, SL: {signal.stop_loss}, TP: {signal.take_profit}"
                )
                # ارسال به TradeManager
                result = await self.trade_manager_callback(signal)

                # بررسی نتیجه ارسال
                if result:
                    logger.info(
                        f"[ارسال] سیگنال برای {signal.symbol} با موفقیت به مدیریت معاملات ارسال شد. شناسه معامله: {result}"
                    )
                    return True
                else:
                    logger.warning(
                        f"[ارسال] مدیریت معاملات سیگنال {signal.symbol} را پذیرش نکرد (نتیجه: {result})"
                    )
                    return False
            except Exception as send_error:
                logger.error(
                    f"[ارسال] خطا در هنگام ارسال سیگنال {signal.symbol} به مدیریت معاملات: {send_error}",
                    exc_info=True
                )
                return False

        except Exception as e:
            logger.error(f"[ارسال] خطای کلی در بررسی و ارسال سیگنال برای {signal.symbol}: {e}", exc_info=True)
            return False

    async def check_signal_still_valid(self, signal: SignalInfo) -> Tuple[bool, str]:
        """
        بررسی اعتبار یک سیگنال با توجه به قیمت فعلی و زمان سپری شده
        (با تنظیمات منعطف‌تر و امکان به‌روزرسانی در زمان اجرا)
        """
        try:
            logger.debug(f"[اعتبارسنجی] بررسی اعتبار سیگنال {signal.symbol}, جهت: {signal.direction}")

            # بررسی زمان انقضا - استفاده از signal_max_age_minutes که در هر به‌روزرسانی تنظیمات به‌روز می‌شود
            current_time = datetime.now().astimezone()

            if not signal.timestamp:
                logger.warning(f"[اعتبارسنجی] سیگنال {signal.symbol} زمان ندارد")
                return False, "سیگنال فاقد زمان است"

            # اطمینان از timezone
            signal_time = signal.timestamp
            if signal_time.tzinfo is None:
                import pytz
                signal_time = pytz.utc.localize(signal_time).astimezone(current_time.tzinfo)

            age_seconds = (current_time - signal_time).total_seconds()
            age_minutes = age_seconds / 60
            # استفاده از signal_max_age_minutes فعلی (که با تغییر تنظیمات به‌روز می‌شود)
            max_age_seconds = max(self.signal_max_age_minutes, 30) * 60

            logger.debug(
                f"[اعتبارسنجی] سن سیگنال {signal.symbol}: {age_minutes:.1f} دقیقه "
                f"(حداکثر مجاز: {max_age_seconds / 60:.1f} دقیقه)"
            )

            if age_seconds > max_age_seconds:
                logger.info(
                    f"[اعتبارسنجی] سیگنال {signal.symbol} منقضی شده است "
                    f"({age_minutes:.1f} دقیقه > {max_age_seconds / 60:.1f} دقیقه)"
                )
                return False, f"سیگنال منقضی شده است ({age_minutes:.1f} دقیقه)"

            # دریافت قیمت فعلی
            current_price = await self.market_data_fetcher.get_current_price(signal.symbol)
            logger.debug(f"[اعتبارسنجی] قیمت فعلی برای {signal.symbol}: {current_price}")

            if current_price is None:
                logger.warning(
                    f"[اعتبارسنجی] امکان دریافت قیمت فعلی برای {signal.symbol} وجود ندارد. فرض بر اعتبار سیگنال"
                )
                return True, "داده قیمتی در دسترس نیست"  # فرض بر اعتبار تا زمان دریافت قیمت

            # بررسی عبور از حد ضرر با انعطاف بیشتر (1% آستانه)
            if signal.direction == 'long' and current_price <= signal.stop_loss * 0.99:  # 1% انعطاف
                logger.info(
                    f"[اعتبارسنجی] سیگنال {signal.symbol} نامعتبر است: قیمت زیر حد ضرر "
                    f"({current_price} <= {signal.stop_loss * 0.99})"
                )
                return False, f"قیمت زیر حد ضرر ({current_price} <= {signal.stop_loss * 0.99})"
            elif signal.direction == 'short' and current_price >= signal.stop_loss * 1.01:  # 1% انعطاف
                logger.info(
                    f"[اعتبارسنجی] سیگنال {signal.symbol} نامعتبر است: قیمت بالای حد ضرر "
                    f"({current_price} >= {signal.stop_loss * 1.01})"
                )
                return False, f"قیمت بالای حد ضرر ({current_price} >= {signal.stop_loss * 1.01})"

            # بررسی حرکت قیمت در جهت مخالف - با انعطاف بیشتر
            entry_price = signal.entry_price
            stop_loss = signal.stop_loss
            # افزایش فاکتور از 0.5 به 0.8 برای انعطاف‌پذیری بیشتر
            allowed_adverse_movement_factor = 0.8  # بیشتر شده

            if signal.direction == 'long':
                adverse_movement_limit = entry_price - abs(entry_price - stop_loss) * allowed_adverse_movement_factor
                logger.debug(
                    f"[اعتبارسنجی] محدوده مجاز حرکت معکوس برای {signal.symbol} (long): "
                    f"{adverse_movement_limit} (فعلی: {current_price})"
                )
                if current_price < adverse_movement_limit:
                    logger.info(
                        f"[اعتبارسنجی] سیگنال {signal.symbol} نامعتبر است: قیمت بیش از حد مجاز در جهت مخالف "
                        f"({current_price} < {adverse_movement_limit})"
                    )
                    return False, f"حرکت قیمت بیش از حد در جهت مخالف ({current_price} < {adverse_movement_limit})"
            elif signal.direction == 'short':
                adverse_movement_limit = entry_price + abs(entry_price - stop_loss) * allowed_adverse_movement_factor
                logger.debug(
                    f"[اعتبارسنجی] محدوده مجاز حرکت معکوس برای {signal.symbol} (short): "
                    f"{adverse_movement_limit} (فعلی: {current_price})"
                )
                if current_price > adverse_movement_limit:
                    logger.info(
                        f"[اعتبارسنجی] سیگنال {signal.symbol} نامعتبر است: قیمت بیش از حد مجاز در جهت مخالف "
                        f"({current_price} > {adverse_movement_limit})"
                    )
                    return False, f"حرکت قیمت بیش از حد در جهت مخالف ({current_price} > {adverse_movement_limit})"

            # اگر تمام بررسی‌ها موفق بود، سیگنال معتبر است
            logger.debug(f"[اعتبارسنجی] سیگنال {signal.symbol} معتبر است")
            return True, "سیگنال معتبر است"

        except Exception as e:
            logger.error(f"[اعتبارسنجی] خطا در بررسی اعتبار سیگنال برای {signal.symbol}: {e}", exc_info=True)
            return False, f"خطا در فرآیند اعتبارسنجی: {str(e)}"

    @staticmethod
    def _get_tf_minutes_util(timeframe: str) -> int:
        """تبدیل تایم‌فریم به دقیقه برای مرتب‌سازی."""
        try:
            if 'm' in timeframe:
                return int(timeframe.replace('m', ''))
            elif 'h' in timeframe:
                return int(timeframe.replace('h', '')) * 60
            elif 'd' in timeframe:
                return int(timeframe.replace('d', '')) * 1440
            elif 'w' in timeframe:
                return int(timeframe.replace('w', '')) * 10080
            else:
                return 99999  # بزرگ برای نامعتبر
        except ValueError:
            return 99999

    def store_incomplete_signal(self, symbol: str, timeframe: str, data: Dict[str, Any]) -> None:
        """
        ذخیره اطلاعات سیگنال ناقص برای بررسی بعدی

        Args:
            symbol: نماد ارز
            timeframe: تایم‌فریم
            data: داده‌های سیگنال ناقص
        """
        with self._signals_lock:
            if symbol not in self.incomplete_signals:
                self.incomplete_signals[symbol] = {}

            self.incomplete_signals[symbol][timeframe] = {
                'data': data,
                'timestamp': datetime.now().astimezone()
            }

            # به‌روزرسانی اولویت نماد
            if symbol in self._symbol_priorities:
                self._symbol_priorities[symbol].is_incomplete = True

            logger.info(f"سیگنال ناقص برای {symbol} در تایم‌فریم {timeframe} ذخیره شد")

    def clean_expired_incomplete_signals(self) -> None:
        """پاکسازی سیگنال‌های ناقص منقضی شده با استراتژی مناسب"""
        with self._signals_lock:
            current_time = datetime.now().astimezone()
            symbols_to_remove = []

            for symbol, data in self.incomplete_signals.items():
                # اگر داده یک دیکشنری با تایم‌فریم‌هاست
                if isinstance(data, dict) and 'timestamp' in data:
                    timestamp = data['timestamp']
                    if not hasattr(timestamp, 'tzinfo') or timestamp.tzinfo is None:
                        # تبدیل naive به aware (با فرض UTC)
                        try:
                            import pytz
                            timestamp = pytz.utc.localize(timestamp).astimezone(current_time.tzinfo)
                        except Exception:
                            # در صورت خطا، فرض بر منقضی بودن
                            symbols_to_remove.append(symbol)
                            continue

                    age = (current_time - timestamp).total_seconds() / 60

                    # استفاده از signal_max_age_minutes فعلی که با تغییر تنظیمات به‌روز می‌شود
                    if age > self.signal_max_age_minutes:
                        symbols_to_remove.append(symbol)
                        logger.debug(f"حذف سیگنال ناقص منقضی شده برای {symbol}")

                # ساختار قدیمی - دیکشنری‌های تایم‌فریم‌ها
                elif isinstance(data, dict):
                    timeframes_to_remove = []

                    for timeframe, tf_data in data.items():
                        if isinstance(tf_data, dict) and 'timestamp' in tf_data:
                            timestamp = tf_data['timestamp']
                            if not hasattr(timestamp, 'tzinfo') or timestamp.tzinfo is None:
                                # تبدیل naive به aware (با فرض UTC)
                                try:
                                    import pytz
                                    timestamp = pytz.utc.localize(timestamp).astimezone(current_time.tzinfo)
                                except Exception:
                                    # در صورت خطا، فرض بر منقضی بودن
                                    timeframes_to_remove.append(timeframe)
                                    continue

                            age = (current_time - timestamp).total_seconds() / 60

                            # استفاده از signal_max_age_minutes فعلی که با تغییر تنظیمات به‌روز می‌شود
                            if age > self.signal_max_age_minutes:
                                timeframes_to_remove.append(timeframe)
                                logger.debug(f"حذف سیگنال ناقص منقضی شده برای {symbol} در تایم‌فریم {timeframe}")

                    # پاکسازی تایم‌فریم‌های منقضی شده
                    for tf in timeframes_to_remove:
                        del data[tf]

                    # اگر همه تایم‌فریم‌ها برای یک نماد پاک شدند
                    if not data:
                        symbols_to_remove.append(symbol)

            # پاکسازی نمادهای بدون تایم‌فریم
            for symbol in symbols_to_remove:
                del self.incomplete_signals[symbol]
                # به‌روزرسانی اولویت نماد
                if symbol in self._symbol_priorities:
                    self._symbol_priorities[symbol].is_incomplete = False

            if symbols_to_remove:
                logger.debug(f"پاکسازی {len(symbols_to_remove)} سیگنال ناقص منقضی شده")

    async def check_incomplete_signals(self) -> None:
        """بررسی سیگنال‌های ناقص ذخیره شده برای تکمیل آنها با اولویت بالا"""
        try:
            # پاکسازی سیگنال‌های منقضی شده
            self.clean_expired_incomplete_signals()

            with self._signals_lock:
                # کپی لیست نمادها برای جلوگیری از خطای تغییر طول دیکشنری حین تکرار
                symbols_to_check = list(self.incomplete_signals.keys())

            if not symbols_to_check:
                return

            logger.info(f"بررسی {len(symbols_to_check)} نماد با سیگنال‌های ناقص")

            # محدود کردن تعداد نمادها در هر اجرا
            if len(symbols_to_check) > self.max_concurrent_symbols * 2:
                logger.info(f"محدود کردن بررسی به {self.max_concurrent_symbols * 2} نماد از {len(symbols_to_check)}")
                symbols_to_check = symbols_to_check[:self.max_concurrent_symbols * 2]

            # بررسی مجدد نمادها با اولویت بالا
            tasks = []
            for symbol in symbols_to_check:
                task = asyncio.create_task(self.process_symbol(symbol, force_refresh=True, priority=True))
                tasks.append((symbol, task))

            # منتظر ماندن برای تکمیل همه تسک‌ها
            signals_completed = 0
            for symbol, task in tasks:
                try:
                    result = await task

                    if result is not None:
                        signals_completed += 1
                        # حذف از سیگنال‌های ناقص
                        with self._signals_lock:
                            if symbol in self.incomplete_signals:
                                del self.incomplete_signals[symbol]
                                if symbol in self._symbol_priorities:
                                    self._symbol_priorities[symbol].is_incomplete = False
                                logger.info(f"سیگنال ناقص برای {symbol} تکمیل شد")
                except Exception as e:
                    logger.error(f"خطا در پردازش سیگنال ناقص برای {symbol}: {e}")

            logger.info(f"تکمیل {signals_completed} از {len(symbols_to_check)} سیگنال ناقص")

        except Exception as e:
            logger.error(f"خطا در بررسی سیگنال‌های ناقص: {e}", exc_info=True)

    async def process_all_symbols(self, symbols: Optional[List[str]] = None, force_refresh: bool = False) -> List[SignalInfo]:
        """
        پردازش تمام نمادها به صورت موازی با اولویت‌بندی هوشمند

        Args:
            symbols: لیست نمادها (اگر None باشد از لیست فعال استفاده می‌شود)
            force_refresh: دریافت مجدد داده‌ها

        Returns:
            لیست سیگنال‌های معتبر
        """
        if symbols is None:
            with self._signals_lock:
                symbols = self.active_symbols.copy()

        if not symbols:
            logger.warning("هیچ نمادی برای پردازش وجود ندارد")
            return []

        # محدود کردن تعداد نمادها در هر اجرا - استفاده از تنظیمات فعلی
        max_symbols = self.max_symbols_per_run
        if len(symbols) > max_symbols:
            logger.info(f"محدود کردن پردازش به {max_symbols} نماد از {len(symbols)}")
            symbols = symbols[:max_symbols]

        logger.info(f"پردازش {len(symbols)} نماد (بازنویسی={force_refresh})")
        start_time = time.time()

        # اولویت‌بندی نمادها بر اساس زمان آخرین پردازش
        symbols_sorted = self._prioritize_symbols(symbols)

        # اجرای موازی با محدودیت semaphore
        valid_signals = []
        error_count = 0

        # استفاده از استراتژی اجرای گروهی با اندازه متغیر - تنظیم مجدد بر اساس تنظیمات فعلی
        max_batch_size = min(self.max_concurrent_symbols * 3, len(symbols_sorted))
        min_batch_size = max(self.max_concurrent_symbols, 5)

        # محاسبه اندازه گروه بر اساس تعداد نمادها
        batch_size = max(min_batch_size, min(max_batch_size, len(symbols_sorted) // 5))

        # اجرای پردازش در گروه‌های کوچک‌تر
        for i in range(0, len(symbols_sorted), batch_size):
            batch = symbols_sorted[i:i + batch_size]
            logger.debug(
                f"پردازش گروه {i // batch_size + 1}/{(len(symbols_sorted) + batch_size - 1) // batch_size}: {len(batch)} نماد"
            )

            # ایجاد تسک‌ها برای گروه فعلی
            tasks = []
            for symbol in batch:
                task = asyncio.create_task(self.process_symbol(symbol, force_refresh))
                tasks.append((symbol, task))

            # انتظار برای تکمیل تمام تسک‌های گروه
            for symbol, task in tasks:
                try:
                    result = await task

                    if result is not None:
                        valid_signals.append(result)
                        with self._signals_lock:
                            if symbol in self._symbol_priorities:
                                self._symbol_priorities[symbol].last_process_time = time.time()
                                self._symbol_priorities[symbol].has_signal = True
                        logger.debug(f"سیگنال معتبر برای {symbol} به نتایج اضافه شد")
                    else:
                        with self._signals_lock:
                            if symbol in self._symbol_priorities:
                                self._symbol_priorities[symbol].last_process_time = time.time()

                except Exception as e:
                    logger.error(f"خطای کنترل نشده در پردازش {symbol}: {e}")
                    with self._signals_lock:
                        if symbol in self._symbol_priorities:
                            self._symbol_priorities[symbol].has_error = True
                            self._symbol_priorities[symbol].error_count += 1
                    error_count += 1

            # تاخیر بین گروه‌ها به صورت متغیر برای کاهش فشار
            if i + batch_size < len(symbols_sorted):
                # تاخیر کمتر برای گروه‌های کوچک‌تر، بیشتر برای گروه‌های بزرگ‌تر
                delay = min(0.1 + (batch_size / 50), 1.0)
                await asyncio.sleep(delay)

        processing_time = time.time() - start_time
        logger.info(
            f"پردازش {len(symbols)} نماد در {processing_time:.2f} ثانیه انجام شد. "
            f"سیگنال‌های معتبر: {len(valid_signals)}, خطاها: {error_count}"
        )

        return valid_signals

    def _prioritize_symbols(self, symbols: List[str]) -> List[str]:
        """
        اولویت‌بندی نمادها بر اساس معیارهای هوشمند

        Args:
            symbols: لیست نمادهای ورودی

        Returns:
            لیست نمادها به ترتیب اولویت
        """
        current_time = time.time()

        # محاسبه امتیاز برای هر نماد
        symbol_scores = []
        for symbol in symbols:
            priority_obj = self._get_symbol_priority(symbol)
            score = priority_obj.calculate_score(current_time)
            symbol_scores.append((symbol, score))

        # مرتب‌سازی بر اساس امتیاز نزولی
        sorted_symbols = [s[0] for s in sorted(symbol_scores, key=lambda x: x[1], reverse=True)]

        # بررسی توازن نمادها - مخلوط کردن برخی نمادها با اولویت پایین
        # برای جلوگیری از گرسنگی نمادهایی که همیشه اولویت پایین دارند
        if len(sorted_symbols) > 10:
            # افزودن برخی نمادهای تصادفی با اولویت پایین به ابتدای لیست
            low_priority_count = min(int(len(sorted_symbols) * 0.1), 20)  # حداکثر 10% یا 20 نماد
            if low_priority_count > 0:
                low_priority_symbols = sorted_symbols[-low_priority_count:]
                random.shuffle(low_priority_symbols)

                # افزودن 3 نماد با اولویت پایین به ابتدای لیست
                random_samples = low_priority_symbols[:3]
                for s in random_samples:
                    sorted_symbols.remove(s)

                # قرار دادن نمادهای تصادفی در اولویت‌های بالاتر
                insert_positions = [int(len(sorted_symbols) * p) for p in [0.1, 0.3, 0.5]]
                for i, s in enumerate(random_samples):
                    if i < len(insert_positions):
                        sorted_symbols.insert(insert_positions[i], s)
                    else:
                        sorted_symbols.insert(0, s)

        return sorted_symbols

    async def periodic_processing(self) -> None:
        """اجرای دوره‌ای پردازش سیگنال‌ها با برنامه زمانی هوشمند"""
        try:
            # بررسی سیگنال‌های ناقص هر چند ثانیه
            check_interval = self.check_incomplete_interval
            last_full_process_time = 0
            full_process_interval = 300  # فاصله زمانی پردازش کامل (ثانیه)

            # تنظیم پرچم وضعیت اجرا
            self.is_running = True

            while not self._shutdown_requested.is_set():
                current_time = time.time()
                try:
                    # بررسی سیگنال‌های ناقص در هر چرخه
                    await self.check_incomplete_signals()

                    # پردازش کامل دوره‌ای با فاصله زمانی متغیر
                    if current_time - last_full_process_time > full_process_interval:
                        # تنظیم فاصله زمانی بعدی بر اساس تعداد نمادها
                        symbol_count = len(self.active_symbols)
                        if symbol_count < 20:
                            next_interval = 180  # 3 دقیقه برای نمادهای کم
                        elif symbol_count < 50:
                            next_interval = 300  # 5 دقیقه برای تعداد متوسط
                        elif symbol_count < 100:
                            next_interval = 600  # 10 دقیقه برای تعداد زیاد
                        else:
                            next_interval = 900  # 15 دقیقه برای تعداد بسیار زیاد

                        logger.info(f"شروع پردازش دوره‌ای کامل نمادها (هر {next_interval//60} دقیقه برای {symbol_count} نماد)")
                        await self.process_all_symbols()
                        last_full_process_time = time.time()
                        full_process_interval = next_interval

                except Exception as e:
                    logger.error(f"خطا در چرخه پردازش دوره‌ای: {e}", exc_info=True)

                # انتظار تا چرخه بعدی با امکان توقف میان‌راه
                try:
                    await asyncio.wait_for(self._shutdown_requested.wait(), timeout=check_interval)
                    if self._shutdown_requested.is_set():
                        logger.info("پردازش دوره‌ای به دلیل درخواست خاموش شدن متوقف شد")
                        break
                except asyncio.TimeoutError:
                    # زمان انتظار به پایان رسید، ادامه به چرخه بعدی
                    pass

            # تنظیم پرچم وضعیت اجرا
            self.is_running = False

        except asyncio.CancelledError:
            self.is_running = False
            logger.info("تسک پردازش دوره‌ای لغو شد")
        except Exception as e:
            self.is_running = False
            logger.error(f"خطای غیرمنتظره در پردازش دوره‌ای: {e}", exc_info=True)

    async def start_periodic_processing(self) -> None:
        """شروع پردازش دوره‌ای به صورت تسک پس‌زمینه"""
        if self._periodic_task is not None and not self._periodic_task.done():
            logger.warning("پردازش دوره‌ای قبلاً در حال اجرا است")
            return

        self._shutdown_requested.clear()
        self._periodic_task = asyncio.create_task(self.periodic_processing())
        self._periodic_task.set_name("signal_processor_periodic")
        logger.info("پردازش دوره‌ای آغاز شد")

    async def stop_periodic_processing(self) -> None:
        """توقف پردازش دوره‌ای"""
        if self._periodic_task is None or self._periodic_task.done():
            logger.debug("هیچ تسک پردازش دوره‌ای در حال اجرا نیست")
            return

        self._shutdown_requested.set()

        try:
            await asyncio.wait_for(self._periodic_task, timeout=10)
            logger.info("پردازش دوره‌ای با موفقیت متوقف شد")
        except asyncio.TimeoutError:
            logger.warning("زمان انتظار برای توقف تسک دوره‌ای به پایان رسید، اجبار به لغو")
            self._periodic_task.cancel()
            try:
                await self._periodic_task
            except asyncio.CancelledError:
                pass

        self._periodic_task = None
        self.is_running = False

    def get_signals_summary(self) -> Dict[str, Any]:
        """
        ارائه خلاصه‌ای از سیگنال‌های فعلی

        Returns:
            دیکشنری حاوی خلاصه سیگنال‌ها
        """
        with self._signals_lock:
            # استفاده از زمان aware
            current_time = datetime.now().astimezone()

            long_signals = []
            short_signals = []
            expired_signals = []
            processing_times = []  # برای محاسبه میانگین زمان پردازش

            # کپی کردن آیتم‌ها برای جلوگیری از خطای تغییر دیکشنری حین تکرار
            history_items = list(self.signal_history.items())

            for symbol, signal_data in history_items:
                # بررسی کنید که signal_data یک دیکشنری معتبر است و کلیدهای لازم را دارد
                if not isinstance(signal_data, dict) or 'timestamp' not in signal_data or 'signal' not in signal_data:
                    logger.warning(f"فرمت نامعتبر داده سیگنال برای نماد {symbol} در تاریخچه. رد شد.")
                    continue

                signal_time = signal_data['timestamp']
                signal = signal_data['signal']

                # اطمینان از اینکه signal_time از نوع datetime و aware است
                if not isinstance(signal_time, datetime):
                    # تلاش برای تبدیل اگر رشته است (هرچند باید از قبل datetime باشد)
                    try:
                        signal_time = datetime.fromisoformat(str(signal_time))
                    except (TypeError, ValueError):
                        logger.warning(f"فرمت زمان نامعتبر برای سیگنال {symbol}. رد کردن بررسی سن.")
                        continue  # نمی‌توان سن را محاسبه کرد

                # اگر signal_time از نوع datetime است ولی naive، آن را aware کنید
                if signal_time.tzinfo is None:
                    try:
                        import pytz
                        signal_time = pytz.utc.localize(signal_time).astimezone(current_time.tzinfo)
                    except Exception as tz_err:
                        logger.warning(
                            f"خطا در تبدیل زمان به timezone aware برای {symbol}: {tz_err}. استفاده از UTC."
                        )
                        try:
                            # تلاش دوباره با فرض UTC
                            import pytz
                            signal_time = pytz.utc.localize(signal_time)
                        except Exception:
                            continue

                # محاسبه سن سیگنال (حالا هر دو زمان باید aware باشند)
                try:
                    age_seconds = (current_time - signal_time).total_seconds()
                    age_minutes = age_seconds / 60
                except TypeError as e:
                    logger.error(
                        f"خطای TypeError در محاسبه سن برای {symbol}: {e}. فعلی: {current_time}, سیگنال: {signal_time}"
                    )
                    continue  # رد کردن این سیگنال

                # افزودن زمان پردازش به لیست (اگر وجود دارد)
                processing_time = signal_data.get('processing_time')
                if processing_time is not None:
                    processing_times.append(float(processing_time))

                # بررسی تاریخ انقضا - استفاده از signal_max_age_minutes فعلی
                if age_minutes > self.signal_max_age_minutes:
                    expired_signals.append(symbol)
                # اطمینان از وجود شی signal و ویژگی direction
                elif signal and hasattr(signal, 'direction'):
                    # تبدیل دیکشنری خلاصه سیگنال
                    signal_summary = {
                        'symbol': symbol,
                        'score': getattr(signal, 'score', None),
                        'final_score': getattr(signal.score, 'final_score', 0.0) if hasattr(signal, 'score') else 0.0,
                        'age_minutes': round(age_minutes, 1),
                        'entry_price': getattr(signal, 'entry_price', None),
                        'stop_loss': getattr(signal, 'stop_loss', None),
                        'take_profit': getattr(signal, 'take_profit', None),
                        'risk_reward_ratio': getattr(signal, 'risk_reward_ratio', None),
                        'processing_time': round(processing_time, 3) if processing_time is not None else None
                    }
                    if signal.direction == 'long':
                        long_signals.append(signal_summary)
                    elif signal.direction == 'short':
                        short_signals.append(signal_summary)

            # مرتب‌سازی بر اساس امتیاز (نزولی)
            long_signals.sort(key=lambda x: x['final_score'], reverse=True)
            short_signals.sort(key=lambda x: x['final_score'], reverse=True)

            # حذف سیگنال‌های منقضی شده از تاریخچه اصلی
            for symbol in expired_signals:
                if symbol in self.signal_history:  # دوباره چک کنید چون ممکن است در تکرار قبلی حذف شده باشد
                    del self.signal_history[symbol]
                    # به‌روزرسانی اولویت نماد
                    if symbol in self._symbol_priorities:
                        self._symbol_priorities[symbol].has_signal = False

            # محاسبه میانگین زمان پردازش
            avg_processing_time = sum(processing_times) / len(processing_times) if processing_times else 0.0

            processing_stats = {
                'avg_processing_time': round(avg_processing_time, 3),
                'symbols_processed_count': len(self._symbol_priorities),
                'active_symbols_in_list': len(self.active_symbols),
                'success_count': self.process_stats.success_count,
                'error_count': self.process_stats.error_count,
                'success_rate': round(self.process_stats.success_count /
                                   max(self.process_stats.success_count + self.process_stats.error_count, 1) * 100, 1),
                'total_signals_generated': self.process_stats.total_signals
            }

            # اضافه کردن حداقل امتیاز فعلی به خلاصه
            summary = {
                'long_signals': long_signals,
                'short_signals': short_signals,
                'total_signals': len(long_signals) + len(short_signals),
                'expired_signals_count': len(expired_signals),
                'incomplete_signals_count': len(self.incomplete_signals),
                'timestamp': current_time.isoformat(),
                'processing_stats': processing_stats,
                'current_minimum_score': self.minimum_score,
                'signal_max_age_minutes': self.signal_max_age_minutes,
                'auto_forward_signals': self.auto_forward_signals,
                'ensemble_mode': self.use_ensemble
            }

            return summary

    # متدهای اضافی برای مدیریت بهتر

    async def process_selected_symbols(self, symbols: List[str], force_refresh: bool = True) -> List[SignalInfo]:
        """
        پردازش مجدد لیستی از نمادهای خاص با اولویت بالا

        Args:
            symbols: لیست نمادهای مورد نظر
            force_refresh: دریافت مجدد داده‌ها

        Returns:
            لیست سیگنال‌های تولید شده
        """
        if not symbols:
            return []

        logger.info(f"پردازش {len(symbols)} نماد انتخاب شده با اولویت بالا")

        # اجرای همزمان با محدودیت
        tasks = []
        for symbol in symbols:
            task = asyncio.create_task(self.process_symbol(symbol, force_refresh=force_refresh, priority=True))
            tasks.append((symbol, task))

        # جمع‌آوری نتایج
        valid_signals = []
        for symbol, task in tasks:
            try:
                signal = await task
                if signal:
                    valid_signals.append(signal)
            except Exception as e:
                logger.error(f"خطا در پردازش نماد انتخاب شده {symbol}: {e}")

        logger.info(f"پردازش {len(symbols)} نماد انتخاب شده، تولید {len(valid_signals)} سیگنال معتبر")
        return valid_signals

    def get_signal_for_symbol(self, symbol: str) -> Optional[SignalInfo]:
        """
        دریافت سیگنال فعلی برای یک نماد

        Args:
            symbol: نماد مورد نظر

        Returns:
            اطلاعات سیگنال یا None
        """
        with self._signals_lock:
            if symbol in self.signal_history:
                signal_data = self.signal_history[symbol]

                # بررسی اینکه سیگنال منقضی نشده باشد
                timestamp = signal_data.get('timestamp')
                if timestamp:
                    current_time = datetime.now().astimezone()

                    # اگر timestamp naive است آن را aware کنیم
                    if not hasattr(timestamp, 'tzinfo') or timestamp.tzinfo is None:
                        try:
                            import pytz
                            timestamp = pytz.utc.localize(timestamp).astimezone(current_time.tzinfo)
                        except Exception:
                            # در صورت خطا، فرض بر منقضی بودن
                            return None

                    age_minutes = (current_time - timestamp).total_seconds() / 60
                    if age_minutes <= self.signal_max_age_minutes:
                        return signal_data.get('signal')

        return None

    def get_active_signals_count(self) -> int:
        """
        دریافت تعداد کل سیگنال‌های فعال

        Returns:
            تعداد سیگنال‌های فعال
        """
        with self._signals_lock:
            current_time = datetime.now().astimezone()
            active_count = 0

            for symbol, data in self.signal_history.items():
                if 'timestamp' in data:
                    timestamp = data['timestamp']

                    # اگر timestamp naive است آن را aware کنیم
                    if not hasattr(timestamp, 'tzinfo') or timestamp.tzinfo is None:
                        try:
                            import pytz
                            timestamp = pytz.utc.localize(timestamp).astimezone(current_time.tzinfo)
                        except Exception:
                            # در صورت خطا، فرض بر منقضی بودن
                            continue

                    age_minutes = (current_time - timestamp).total_seconds() / 60
                    if age_minutes <= self.signal_max_age_minutes:
                        active_count += 1

        return active_count

    async def process_all(self, force: bool = False) -> None:
        """
        پردازش همه نمادهای فعال - متد ساده‌تر برای فراخوانی از بیرون

        Args:
            force: آیا باید داده‌ها مجدداً دریافت شوند
        """
        logger.info(f"شروع پردازش همه نمادهای فعال (مجبور به بازنویسی: {force})")
        await self.process_all_symbols(force_refresh=force)
        logger.info("پردازش همه نمادهای فعال به پایان رسید")