# trading/trade_manager.py
"""
ماژول trade_manager.py: مدیریت معاملات، ریسک، و اجرای سفارشات (شبیه‌سازی شده).
شامل مدیریت ریسک پیشرفته: اندازه پوزیشن پویا، کنترل همبستگی، حد سود چند مرحله‌ای، استاپ متحرک ATR.

تغییرات و بهینه‌سازی‌ها:
- استفاده از cachetools برای ذخیره موقت نتایج محاسبات پرتکرار
- بهبود مدیریت پایگاه داده با استفاده از context manager
- افزودن سیستم تشخیص آنومالی در داده‌های قیمت برای جلوگیری از عملیات نادرست
- اضافه کردن قابلیت ثبت و پیگیری بهتر معاملات با دسته‌بندی و تگ‌گذاری
- بهینه‌سازی الگوریتم استاپ متحرک با پشتیبانی از استراتژی‌های ترکیبی
- محاسبات بهینه‌تر همبستگی و آمار با الگوریتم‌های کارآمدتر
- بهبود مدیریت حافظه و کاهش فشار بر GC با مدیریت بهتر اشیاء
- پشتیبانی از به‌روزرسانی پویای تنظیمات در زمان اجرا
"""

import logging
import asyncio
import json
from datetime import datetime
from typing import Optional
# ... سایر ایمپورت‌ها ...
from signal_generator import SignalInfo, SignalScore
from multi_tp_trade import Trade
import math
from typing import Dict, List, Optional, Any, Tuple, Callable, Awaitable, Union
import logging
import asyncio
import time
from typing import Dict, List, Optional, Any, Tuple, Callable, Awaitable
from datetime import datetime
import json
import sqlite3
import threading
import uuid
import math
import os
import pandas as pd
import numpy as np
import talib
from cachetools import TTLCache, LRUCache
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
import copy  # برای استفاده از deepcopy
# فرض: SignalInfo از ماژول signal_generator وارد می‌شود
from signal_generator import SignalInfo
import json
import random  # برای برخی توابع دیگر
import math  # برای توابع sanitize_float

# وارد کردن ماژول‌های مرتبط با trade_extensions
try:
    from trade_extensions import CorrelationManager, PositionSizeOptimizer, TradeAnalyzer, DynamicStopManager, \
        RiskCalculator

    EXTENSIONS_AVAILABLE = True
except ImportError:
    EXTENSIONS_AVAILABLE = False
from multi_tp_trade import Trade
# ایمپورت کلاس TradeResult و SignalInfo از signal_generator
from signal_generator import TradeResult, SignalInfo
# ایمپورت برای تایپ هینتینگ کالبک ML integration
from ml_signal_integration import MLSignalIntegration

# تنظیم لاگر
logger = logging.getLogger(__name__)

# ========================================
#        تنظیمات و ثابت‌های سیستم
# ========================================
# حداقل تعداد داده برای محاسبات معتبر همبستگی
MIN_CORRELATION_DATA_POINTS = 20

# حداکثر تعداد تلاش برای عملیات‌های دیتابیس
MAX_DB_RETRIES = 3

# حداکثر تعداد ترد‌های همزمان در ThreadPoolExecutor
MAX_THREAD_WORKERS = 4

# زمان (به ثانیه) نگهداری نتایج کش شده
CACHE_TTL_SECONDS = 300

# زمان انقضا برای کش قیمت‌ها (به ثانیه)
PRICE_CACHE_TTL = 10

# حداکثر اندازه کش LRU
MAX_LRU_CACHE_SIZE = 100

# حداکثر زمان اجرای پروسه‌های همزمان (به ثانیه)
ASYNC_OPERATION_TIMEOUT = 30


# ===============================================
#      دیتاکلاس‌های کمکی
# ===============================================
@dataclass
class TradeSummary:
    """خلاصه اطلاعات معامله برای گزارش‌گیری"""
    trade_id: str
    symbol: str
    direction: str
    status: str
    entry_price: float
    current_price: float = None
    profit_loss: float = None
    profit_loss_percent: float = None
    entry_time: datetime = None
    age_hours: float = 0.0
    remaining_quantity: float = 0.0
    risk_reward_ratio: float = None
    tags: List[str] = field(default_factory=list)
    strategy_name: str = None

    @classmethod
    def from_trade(cls, trade: Trade) -> 'TradeSummary':
        """ایجاد خلاصه از یک معامله"""
        with trade._lock:
            age = trade.get_age()
            pnl = trade.profit_loss
            pnl_percent = trade.profit_loss_percent

            # برای معاملات باز، از سود/زیان شناور استفاده می‌کنیم
            if trade.status != 'closed':
                pnl = trade.get_floating_pnl()
                pnl_percent = trade.get_net_pnl_percent()

            return cls(
                trade_id=trade.trade_id,
                symbol=trade.symbol,
                direction=trade.direction,
                status=trade.status,
                entry_price=trade.entry_price,
                current_price=trade.current_price,
                profit_loss=pnl,
                profit_loss_percent=pnl_percent,
                entry_time=trade.timestamp,
                age_hours=age,
                remaining_quantity=trade.remaining_quantity,
                risk_reward_ratio=trade.risk_reward_ratio,
                tags=trade.tags.copy() if trade.tags else [],
                strategy_name=trade.strategy_name
            )


@dataclass
class PortfolioStats:
    """آمار جامع پورتفولیو معاملات"""
    total_trades_opened: int = 0
    open_trades: int = 0
    partially_closed_trades: int = 0
    closed_trades: int = 0
    winning_trades: int = 0
    losing_trades: int = 0
    total_net_profit_loss: float = 0.0
    total_commission_paid: float = 0.0
    win_rate: float = 0.0
    profit_factor: float = 0.0
    current_equity: float = 0.0
    current_open_pnl: float = 0.0
    current_drawdown: float = 0.0
    realized_pnl_partial_trades: float = 0.0
    avg_win_percent: float = 0.0
    avg_loss_percent: float = 0.0
    largest_win: float = 0.0
    largest_loss: float = 0.0
    avg_trade_duration_hours: float = 0.0
    avg_risk_reward_ratio: float = 0.0
    symbols_distribution: Dict[str, int] = field(default_factory=dict)
    daily_pnl: Dict[str, float] = field(default_factory=dict)
    trades_by_tag: Dict[str, int] = field(default_factory=dict)
    trades_by_strategy: Dict[str, int] = field(default_factory=dict)


# ===============================================
#      کلاس اصلی TradeManager
# ===============================================
class TradeManager:
    """مدیریت معاملات، ریسک و اجرای شبیه‌سازی شده."""

    def __init__(self, config: Dict[str, Any], db_path: Optional[str] = None):
        """مقداردهی اولیه."""
        self.config = config
        self.trade_config = config.get('trading', {})
        self.risk_config = config.get('risk_management', {})
        self.mode = self.trade_config.get('mode', 'simulation')

        # پرچم نشان‌دهنده وضعیت به‌روزرسانی قیمت
        self.price_update_running = False

        # اطلاع‌رسانی
        self.notification_callback: Optional[Callable[[str], Awaitable[None]]] = None  # تابع async

        # --- >> اضافه شدن کالبک برای گزارش نتیجه معامله << ---
        self.trade_result_callback: Optional[
            Callable[[TradeResult], Awaitable[None]]] = None  # Callback for ML Integration
        # --- << پایان اضافه شدن کالبک برای گزارش نتیجه معامله >> ---

        # مدیریت تسک پریودیک
        self._shutdown_requested = asyncio.Event()

        # دریافت پارامترهای مدیریت ریسک از پیکربندی
        self._update_config_params()

        # دیتابیس
        self.db_path = db_path or config.get('storage', {}).get('database_path', 'data/trades.db')
        self.conn: Optional[sqlite3.Connection] = None
        self.cursor: Optional[sqlite3.Cursor] = None
        self._ensure_db_directory()
        self._db_lock = threading.RLock()

        # معاملات فعال و قفل
        self.active_trades: Dict[str, Trade] = {}
        self._trades_lock = threading.RLock()  # RLock برای فراخوانی‌های تودرتو

        # کش‌های مختلف برای بهبود عملکرد
        self._price_cache = TTLCache(maxsize=200, ttl=PRICE_CACHE_TTL)  # کش قیمت‌ها با TTL 10 ثانیه
        self._correlation_cache = TTLCache(maxsize=500, ttl=600)  # کش همبستگی با TTL 10 دقیقه
        self._calculation_cache = LRUCache(maxsize=MAX_LRU_CACHE_SIZE)  # کش محاسبات با محدودیت اندازه

        # مدیریت ThreadPool
        self._thread_executor = ThreadPoolExecutor(max_workers=MAX_THREAD_WORKERS)

        # به‌روزرسانی قیمت
        self.price_fetcher_callback: Optional[Callable[[str], Awaitable[Optional[float]]]] = None  # تابع async
        self.data_fetcher_instance: Optional[Any] = None  # برای دسترسی به get_historical_data در correlation

        # مدیریت تسک پریودیک
        self._periodic_task: Optional[asyncio.Task] = None
        self._tasks = []  # لیست تسک‌های async

        # آمار و وضعیت حساب
        self.peak_equity: float = self.initial_balance  # برای محاسبه Drawdown
        self.stats: Dict[str, Any] = self._reset_stats()

        # کامپوننت‌های ماژول trade_extensions
        self.correlation_manager = None
        self.position_size_optimizer = None
        self.trade_analyzer = None
        self.dynamic_stop_manager = None
        self.risk_calculator = None

        # راه‌اندازی ماژول‌های trade_extensions اگر موجود باشند
        if EXTENSIONS_AVAILABLE:
            self._initialize_extensions()

        logger.info(f"TradeManager initialized in {self.mode} mode. Dynamic Sizing: {self.use_dynamic_sizing}, "
                    f"Correlation Check: {self.use_correlation_check}, Multi-TP: {self.use_multi_tp}, "
                    f"Trailing Stop: {self.use_trailing_stop} (ATR: {self.use_atr_trailing})")

    def _update_config_params(self):
        """به‌روزرسانی پارامترهای مدیریت ریسک بر اساس پیکربندی."""
        # دریافت پارامترهای اصلی مدیریت ریسک
        self.initial_balance = self.risk_config.get('account_balance', 10000.0)
        self.max_risk_per_trade = self.risk_config.get('max_risk_per_trade_percent', 1.0)
        self.use_dynamic_sizing = self.risk_config.get('use_dynamic_position_sizing', True)
        self.max_open_trades = self.risk_config.get('max_open_trades', 10)
        self.max_trades_per_symbol = self.risk_config.get('max_trades_per_symbol', 1)
        self.allow_opposite_trades = self.risk_config.get('allow_opposite_trades', False)
        self.use_correlation_check = self.risk_config.get('use_correlation_check', True)

        # خواندن تنظیمات multi_tp از زیربخش مربوطه
        self.use_multi_tp = self.trade_config.get('multi_tp', {}).get('enabled', True)

        # تنظیمات حد ضرر متحرک
        self.use_trailing_stop = self.risk_config.get('use_trailing_stop', True)
        self.use_atr_trailing = self.risk_config.get('use_atr_based_trailing', True)
        self.trailing_stop_activation_percent = self.risk_config.get('trailing_stop_activation_percent', 3.0)
        self.trailing_stop_distance_percent = self.risk_config.get('trailing_stop_distance_percent', 2.25)
        self.atr_trailing_multiplier = self.risk_config.get('atr_trailing_multiplier', 2.25)
        self.atr_trailing_period = self.risk_config.get('atr_trailing_period', 21)

        # خواندن نرخ کارمزد از بخش backtest
        self.commission_rate = self.config.get('backtest', {}).get('commission_rate', 0.0006)

        # تنظیمات به‌روزرسانی قیمت
        self.auto_update_prices = self.trade_config.get('auto_update_prices', True)
        self.price_update_interval = self.trade_config.get('price_update_interval', 10)

        # تنظیمات کش قیمت
        global PRICE_CACHE_TTL
        PRICE_CACHE_TTL = max(5, min(60, self.price_update_interval // 2))

    def update_config(self, new_config: Dict[str, Any]) -> None:
        """
        به‌روزرسانی تنظیمات کلاس TradeManager. این متد زمانی که تنظیمات در زمان اجرا تغییر می‌کند، فراخوانی می‌شود.

        Args:
            new_config: پیکربندی جدید
        """
        logger.info("به‌روزرسانی تنظیمات TradeManager...")

        # ذخیره پیکربندی قبلی برای مقایسه تغییرات
        old_config = copy.deepcopy(self.config)
        old_trade_config = self.trade_config.copy()
        old_risk_config = self.risk_config.copy()

        # به‌روزرسانی پیکربندی کلی
        self.config = new_config
        self.trade_config = new_config.get('trading', {})
        self.risk_config = new_config.get('risk_management', {})

        # به‌روزرسانی پارامترهای داخلی
        self._update_config_params()

        # لاگ تغییرات مهم پیکربندی
        self._log_important_config_changes(old_trade_config, old_risk_config)

        # اگر به‌روزرسانی قیمت تغییر کرده، آن را اعمال کنید
        if old_trade_config.get('auto_update_prices', True) != self.auto_update_prices:
            if self.auto_update_prices and not self.price_update_running:
                # شروع به‌روزرسانی قیمت اگر فعال شده است
                logger.info("به‌روزرسانی خودکار قیمت فعال شد - شروع تسک دوره‌ای")
                asyncio.create_task(self.start_periodic_price_update())
            elif not self.auto_update_prices and self.price_update_running:
                # توقف به‌روزرسانی قیمت اگر غیرفعال شده است
                logger.info("به‌روزرسانی خودکار قیمت غیرفعال شد - توقف تسک دوره‌ای")
                asyncio.create_task(self.stop_periodic_price_update())

        # اگر فاصله زمانی به‌روزرسانی قیمت تغییر کرده، دوباره شروع کنید
        if old_trade_config.get('price_update_interval', 10) != self.price_update_interval and self.auto_update_prices:
            logger.info(
                f"فاصله به‌روزرسانی قیمت تغییر کرد به {self.price_update_interval} ثانیه - راه‌اندازی مجدد تسک دوره‌ای")
            if self.price_update_running:
                asyncio.create_task(self.stop_periodic_price_update())
            asyncio.create_task(self.start_periodic_price_update())

        # به‌روزرسانی برخی کامپوننت‌های extension
        self._update_extensions_config()

        logger.info("به‌روزرسانی تنظیمات TradeManager انجام شد")

    def _log_important_config_changes(self, old_trade_config: Dict[str, Any], old_risk_config: Dict[str, Any]) -> None:
        """
        لاگ تغییرات مهم در پیکربندی.

        Args:
            old_trade_config: پیکربندی معاملات قبلی
            old_risk_config: پیکربندی مدیریت ریسک قبلی
        """
        # تغییرات مهم تنظیمات معاملات
        if old_trade_config.get('mode') != self.mode:
            logger.info(f"حالت معاملات تغییر کرد: {old_trade_config.get('mode')} -> {self.mode}")

        if old_trade_config.get('auto_update_prices') != self.auto_update_prices:
            logger.info(
                f"به‌روزرسانی خودکار قیمت: {old_trade_config.get('auto_update_prices')} -> {self.auto_update_prices}")

        if old_trade_config.get('price_update_interval') != self.price_update_interval:
            logger.info(
                f"فاصله به‌روزرسانی قیمت: {old_trade_config.get('price_update_interval')}s -> {self.price_update_interval}s")

        multi_tp_old = old_trade_config.get('multi_tp', {}).get('enabled', True)
        if multi_tp_old != self.use_multi_tp:
            logger.info(f"حد سود چندمرحله‌ای: {multi_tp_old} -> {self.use_multi_tp}")

        # تغییرات مهم تنظیمات مدیریت ریسک
        if old_risk_config.get('max_risk_per_trade_percent') != self.max_risk_per_trade:
            logger.info(
                f"حداکثر ریسک هر معامله: {old_risk_config.get('max_risk_per_trade_percent')}% -> {self.max_risk_per_trade}%")

        if old_risk_config.get('use_trailing_stop') != self.use_trailing_stop:
            logger.info(f"حد ضرر متحرک: {old_risk_config.get('use_trailing_stop')} -> {self.use_trailing_stop}")

        if old_risk_config.get('trailing_stop_activation_percent') != self.trailing_stop_activation_percent:
            logger.info(
                f"درصد فعال‌سازی حد ضرر متحرک: {old_risk_config.get('trailing_stop_activation_percent')}% -> {self.trailing_stop_activation_percent}%")

        if old_risk_config.get('trailing_stop_distance_percent') != self.trailing_stop_distance_percent:
            logger.info(
                f"فاصله حد ضرر متحرک: {old_risk_config.get('trailing_stop_distance_percent')}% -> {self.trailing_stop_distance_percent}%")

        if old_risk_config.get('use_atr_based_trailing') != self.use_atr_trailing:
            logger.info(
                f"حد ضرر متحرک بر اساس ATR: {old_risk_config.get('use_atr_based_trailing')} -> {self.use_atr_trailing}")

        if old_risk_config.get('atr_trailing_multiplier') != self.atr_trailing_multiplier:
            logger.info(
                f"ضریب ATR در حد ضرر متحرک: {old_risk_config.get('atr_trailing_multiplier')} -> {self.atr_trailing_multiplier}")

    def _update_extensions_config(self) -> None:
        """به‌روزرسانی کانفیگ کامپوننت‌های extensions."""
        if EXTENSIONS_AVAILABLE:
            try:
                # به‌روزرسانی تنظیمات CorrelationManager
                if self.correlation_manager and hasattr(self.correlation_manager, 'update_config'):
                    self.correlation_manager.update_config(self.config)

                # به‌روزرسانی تنظیمات PositionSizeOptimizer
                if self.position_size_optimizer and hasattr(self.position_size_optimizer, 'update_config'):
                    self.position_size_optimizer.update_config(self.config)

                # به‌روزرسانی تنظیمات DynamicStopManager
                if self.dynamic_stop_manager and hasattr(self.dynamic_stop_manager, 'update_config'):
                    self.dynamic_stop_manager.update_config(self.config)

                # به‌روزرسانی تنظیمات RiskCalculator
                if self.risk_calculator and hasattr(self.risk_calculator, 'update_config'):
                    self.risk_calculator.update_config(self.config)

                logger.debug("تنظیمات کامپوننت‌های extensions با موفقیت به‌روزرسانی شد")
            except Exception as e:
                logger.error(f"خطا در به‌روزرسانی تنظیمات extensions: {e}")

    def initialize_db(self):
        """راه‌اندازی دیتابیس و بارگذاری معاملات فعال."""
        for retry in range(MAX_DB_RETRIES):
            try:
                # استفاده از context manager برای اتصال
                with sqlite3.connect(self.db_path, check_same_thread=False, timeout=10) as conn:
                    conn.row_factory = sqlite3.Row
                    cursor = conn.cursor()
                    # ایجاد جدول trades با ستون جدید entry_reasons_json
                    cursor.execute('''
                     CREATE TABLE IF NOT EXISTS trades (
                         trade_id TEXT PRIMARY KEY,
                         symbol TEXT NOT NULL,
                         timestamp TEXT NOT NULL,
                         status TEXT NOT NULL,
                         direction TEXT,
                         entry_price REAL,
                         stop_loss REAL,
                         take_profit REAL,
                         quantity REAL,
                         remaining_quantity REAL,
                         current_price REAL,
                         exit_time TEXT,
                         exit_reason TEXT,
                         profit_loss REAL,
                         commission_paid REAL,
                         tags TEXT,
                         strategy_name TEXT,
                         timeframe TEXT,
                         market_state TEXT,
                         notes TEXT,
                         entry_reasons_json TEXT, -- <<< ستون جدید
                         data TEXT
                     )
                     ''')

                    # ایجاد ایندکس‌ها
                    cursor.execute('CREATE INDEX IF NOT EXISTS idx_trades_symbol_status ON trades(symbol, status)')
                    cursor.execute('CREATE INDEX IF NOT EXISTS idx_trades_timestamp ON trades(timestamp)')
                    cursor.execute('CREATE INDEX IF NOT EXISTS idx_trades_strategy ON trades(strategy_name)')

                    # ایجاد جدول balance_history
                    cursor.execute('''
                     CREATE TABLE IF NOT EXISTS balance_history (
                         id INTEGER PRIMARY KEY AUTOINCREMENT,
                         timestamp TEXT NOT NULL,
                         balance REAL NOT NULL,
                         equity REAL,
                         margin REAL,
                         open_pnl REAL,
                         description TEXT
                     )
                     ''')

                    # اطمینان از وجود ستون‌های اختیاری در هر دو جدول
                    if hasattr(self, '_add_missing_db_column'):
                        # ستون‌های جدول balance_history
                        self._add_missing_db_column('balance_history', 'equity', 'REAL')
                        self._add_missing_db_column('balance_history', 'margin', 'REAL')
                        self._add_missing_db_column('balance_history', 'open_pnl', 'REAL')
                        self._add_missing_db_column('balance_history', 'description', 'TEXT')

                        # ستون جدید entry_reasons_json در جدول trades
                        self._add_missing_db_column('trades', 'entry_reasons_json', 'TEXT')

                    # ایجاد جدول تاریخچه SL/TP
                    cursor.execute('''
                     CREATE TABLE IF NOT EXISTS trade_level_history (
                         id INTEGER PRIMARY KEY AUTOINCREMENT,
                         trade_id TEXT NOT NULL,
                         timestamp TEXT NOT NULL,
                         level_type TEXT NOT NULL, -- "stop_loss", "take_profit"
                         old_value REAL,
                         new_value REAL,
                         reason TEXT,
                         FOREIGN KEY (trade_id) REFERENCES trades (trade_id)
                     )
                     ''')

                    conn.commit()
                    self.conn = conn
                    self.cursor = cursor
                    logger.info(f"Database initialized at {self.db_path}")
                    self.load_active_trades()
                    self._update_stats()
                    return True

            except sqlite3.Error as e:
                logger.error(f"Database initialization error (attempt {retry + 1}/{MAX_DB_RETRIES}): {e}",
                             exc_info=True)
                if retry < MAX_DB_RETRIES - 1:
                    time.sleep(1)  # انتظار قبل از تلاش مجدد

        logger.critical(f"Failed to initialize database after {MAX_DB_RETRIES} attempts")
        self.conn = None
        self.cursor = None
        return False

    def _add_missing_db_column(self, table_name: str, column_name: str, column_type: str) -> bool:
        """اضافه کردن ستون جدید به جدول اگر وجود نداشته باشد."""
        if not self.conn or not self.cursor: return False
        try:
            with self._db_lock:
                self.cursor.execute(f"PRAGMA table_info({table_name})")
                columns = [info[1] for info in self.cursor.fetchall()]
                if column_name not in columns:
                    self.cursor.execute(f"ALTER TABLE {table_name} ADD COLUMN {column_name} {column_type}")
                    self.conn.commit()
                    logger.info(f"Added column '{column_name}' to table '{table_name}'.")
                    return True
                return True  # Already exists
        except sqlite3.Error as e:
            # Handle specific error if column already exists (might happen in race conditions)
            if "duplicate column name" in str(e).lower():
                return True
            logger.error(f"Failed to add column '{column_name}' to '{table_name}': {e}")
            return False

    def register_trade_result_callback(self, callback_function: Callable[[TradeResult], Awaitable[None]]) -> None:
        """
        ثبت تابع callback برای ارسال نتیجه معامله به ماژول‌های دیگر (مثلا ML Integration).

        Args:
            callback_function: تابع callback که یک شی TradeResult را دریافت می‌کند.
        """
        self.trade_result_callback = callback_function
        logger.info("Trade result callback registered.")

    async def process_signal(self, signal: SignalInfo) -> Optional[str]:
        """
        پردازش سیگنال و ایجاد معامله جدید با تحلیل همبستگی بیت‌کوین.

        Args:
            signal: اطلاعات سیگنال معاملاتی

        Returns:
            شناسه معامله جدید یا None در صورت رد شدن سیگنال
        """
        if not signal or not isinstance(signal, SignalInfo):
            logger.error("[TRADE_MGR] شیء سیگنال نامعتبر در process_signal دریافت شد.")
            return None

        try:
            # اطمینان از وجود قیمت‌ها و معتبر بودن آنها قبل از لاگ کردن
            entry_price_log = f"{signal.entry_price:.5f}" if signal.entry_price is not None else "N/A"
            sl_log = f"{signal.stop_loss:.5f}" if signal.stop_loss is not None else "N/A"
            tp_log = f"{signal.take_profit:.5f}" if signal.take_profit is not None else "N/A"
            rr_log = f"{signal.risk_reward_ratio:.2f}" if signal.risk_reward_ratio is not None else "N/A"
            score_log = f"{getattr(signal.score, 'final_score', 'N/A'):.2f}" if hasattr(signal,
                                                                                        'score') and signal.score else "N/A"

            # لاگ کردن اطلاعات سیگنال با جزئیات بیشتر
            signal_strategy = getattr(signal, 'strategy_name', getattr(signal, 'signal_type', 'unknown'))
            signal_timeframe = getattr(signal, 'timeframe', 'unknown')

            logger.info(
                f"[TRADE_MGR] پردازش سیگنال برای {signal.symbol}: {signal.direction}, استراتژی: {signal_strategy}, "
                f"تایم‌فریم: {signal_timeframe}, امتیاز: {score_log}, "
                f"RR: {rr_log}, ورود: {entry_price_log}, SL: {sl_log}, TP: {tp_log}"
            )

            # 1. بررسی معتبر بودن قیمت‌های سیگنال
            if not self._validate_signal_prices(signal):
                logger.warning(f"[TRADE_MGR] اعتبارسنجی قیمت‌های سیگنال برای {signal.symbol} ناموفق بود. سیگنال رد شد.")
                return None

            # 2. بررسی امکان باز کردن معامله
            if not self.can_open_new_trade(signal.symbol, signal.direction):
                # دلیل عدم امکان قبلا در خود تابع can_open_new_trade لاگ شده است
                logger.warning(f"[TRADE_MGR] امکان باز کردن معامله جدید برای {signal.symbol} وجود ندارد")
                return None

            # 3. بررسی همبستگی
            open_trades_list = self.get_open_trades()
            is_allowed, corr_level, corr_symbols, btc_compatibility_info = await self.correlation_manager.check_portfolio_correlation(
                signal.symbol, signal.direction, open_trades_list, self.data_fetcher_instance
            )

            if not is_allowed:
                logger.info(
                    f"[TRADE_MGR] سیگنال برای {signal.symbol} به دلیل همبستگی بالا ({corr_level:.2f}) با {corr_symbols} رد شد."
                )
                return None

            # بررسی همبستگی با بیت‌کوین و تنظیم امتیاز سیگنال بر اساس آن
            if btc_compatibility_info and 'correlation_score' in btc_compatibility_info:
                btc_corr_score = btc_compatibility_info.get('correlation_score', 0)
                corr_reason = btc_compatibility_info.get('reason', 'unknown')
                compatibility_score = btc_compatibility_info.get('compatibility_score', 50)

                # اعمال امتیاز همبستگی به کیفیت سیگنال
                if hasattr(signal, 'score') and hasattr(signal.score, 'final_score'):
                    original_score = signal.score.final_score

                    # تعدیل امتیاز براساس همبستگی
                    # امتیاز منفی: کاهش امتیاز سیگنال
                    # امتیاز مثبت: افزایش امتیاز سیگنال
                    impact_factor = 0.5  # ضریب تاثیر (می‌توان تنظیم کرد)
                    adjusted_score = original_score + (btc_corr_score * impact_factor)

                    # محدود کردن بین 10 و 100
                    signal.score.final_score = max(10, min(100, adjusted_score))

                    # افزودن امتیاز همبستگی به score_details
                    if hasattr(signal.score, 'score_details') and isinstance(signal.score.score_details, dict):
                        signal.score.score_details['btc_correlation_score'] = btc_corr_score
                        signal.score.score_details['btc_compatibility_score'] = compatibility_score

                    logger.info(
                        f"[TRADE_MGR] امتیاز سیگنال {signal.symbol} از {original_score} به {signal.score.final_score} "
                        f"تنظیم شد (تعدیل همبستگی بیت‌کوین: {btc_corr_score}, دلیل: {corr_reason})"
                    )

                # افزودن تگ‌های مرتبط با همبستگی
                corr_type = btc_compatibility_info.get('correlation_type', '')
                if hasattr(signal, 'add_tag'):
                    if corr_type == 'positive':
                        signal.add_tag('btc_correlated')
                    elif corr_type == 'inverse':
                        signal.add_tag('btc_inverse')
                    elif corr_type == 'zero':
                        signal.add_tag('btc_independent')

                # اگر معامله با بیت‌کوین سازگار نیست ولی به دلیل همبستگی پورتفولیو رد نشده، لاگ بزنیم
                if not btc_compatibility_info.get('is_compatible', True):
                    logger.warning(
                        f"[TRADE_MGR] معامله {signal.symbol} ({signal.direction}) با روند بیت‌کوین ناسازگار است، "
                        f"اما به دلیل نداشتن همبستگی قوی مثبت، رد نشد. امتیاز تعدیل شد: {btc_corr_score}"
                    )

            # 4. دریافت کانفیگ تطبیق‌یافته (مثلا بر اساس رژیم بازار)
            adapted_config, adapted_risk_config = self._get_adapted_config(signal)
            logger.debug(f"[TRADE_MGR] کانفیگ تطبیق‌یافته برای {signal.symbol} دریافت شد")

            # 5. محاسبه اندازه پوزیشن
            stop_distance = abs(signal.entry_price - signal.stop_loss)
            if stop_distance <= 1e-9:
                logger.warning(
                    f"[TRADE_MGR] فاصله استاپ خیلی کم یا صفر است ({stop_distance:.8f}) برای {signal.symbol}. سیگنال رد شد."
                )
                return None

            # محاسبه اندازه پوزیشن با استفاده از کانفیگ تطبیق یافته
            position_size_info = self.calculate_position_size(signal, stop_distance, adapted_risk_config)
            logger.debug(f"[TRADE_MGR] اندازه پوزیشن محاسبه شده برای {signal.symbol}: {position_size_info}")

            # بررسی نتیجه محاسبه اندازه پوزیشن (می‌تواند دیکشنری باشد)
            if isinstance(position_size_info, dict):
                quantity = position_size_info.get('position_size', 0.0)
                calculated_risk_amount = position_size_info.get('risk_amount', 0.0)
            elif isinstance(position_size_info, (float, int)):  # Fallback برای سازگاری با نسخه قبلی
                quantity = float(position_size_info)
                calculated_risk_amount = stop_distance * quantity
            else:
                logger.error(
                    f"[TRADE_MGR] اطلاعات اندازه پوزیشن نامعتبر برای {signal.symbol} برگردانده شد: {position_size_info}")
                return None

            if quantity <= 1e-9:
                logger.warning(
                    f"[TRADE_MGR] اندازه پوزیشن محاسبه شده خیلی کوچک است ({quantity:.8f}) برای {signal.symbol}. سیگنال رد شد."
                )
                return None

            # 6. ایجاد شناسه معامله
            trade_id = self._generate_trade_id(signal.symbol, signal.direction)
            logger.debug(f"[TRADE_MGR] شناسه معامله ایجاد شد: {trade_id}")

            # استخراج و سریال‌سازی دلایل ورود از SignalScore
            entry_reasons_json_str = None
            if hasattr(signal, 'score') and isinstance(signal.score, SignalScore):
                try:
                    # استفاده از to_dict برای گرفتن همه جزئیات امتیاز
                    reasons_dict = signal.score.to_dict()
                    # تبدیل به رشته JSON
                    entry_reasons_json_str = json.dumps(reasons_dict, default=str, ensure_ascii=False, indent=2)
                    logger.debug(f"[TRADE_MGR] دلایل ورود برای {trade_id} با موفقیت سریالایز شد")
                except Exception as json_err:
                    logger.error(f"[TRADE_MGR] خطا در سریال‌سازی دلایل ورود (SignalScore) برای {trade_id}: {json_err}")
                    entry_reasons_json_str = json.dumps({"error": "serialization failed", "details": str(json_err)})
            else:
                logger.warning(
                    f"[TRADE_MGR] سیگنال برای {trade_id} شیء SignalScore معتبر ندارد. امکان ذخیره دلایل ورود دقیق وجود ندارد."
                )
                entry_reasons_json_str = json.dumps({"info": "No detailed score available"})

            # استخراج صحیح فیلدهای استراتژی و تایم‌فریم
            # اولویت بندی در دریافت نام استراتژی:
            strategy_name = None
            if hasattr(signal, 'strategy_name') and signal.strategy_name:
                strategy_name = signal.strategy_name
            elif hasattr(signal, 'signal_type') and signal.signal_type:
                strategy_name = signal.signal_type
            elif hasattr(signal, 'setup_name') and signal.setup_name:
                strategy_name = signal.setup_name
            else:
                # استفاده از اطلاعات دیگر برای ساخت یک نام استراتژی معنادار
                direction_prefix = "Long" if signal.direction == "long" else "Short"
                timeframe_suffix = getattr(signal, 'timeframe', '5m')
                strategy_name = f"{direction_prefix}_{timeframe_suffix}"

            # اطمینان از معتبر بودن تایم‌فریم
            timeframe = getattr(signal, 'timeframe', None)
            if not timeframe:
                # استفاده از تایم‌فریم پیش‌فرض از کانفیگ
                default_timeframes = self.config.get('data_fetching', {}).get('timeframes', ['5m', '15m', '1h', '4h'])
                timeframe = default_timeframes[0] if default_timeframes else '5m'

            # استخراج وضعیت بازار
            market_state = getattr(signal, 'market_condition', None)
            if not market_state:
                market_state = getattr(signal, 'market_state', 'neutral')

            # ساخت تگ‌های مناسب
            tags = []
            # اضافه کردن signal_type به عنوان تگ
            if hasattr(signal, 'signal_type') and signal.signal_type:
                tags.append(signal.signal_type)

            # اضافه کردن جهت معامله به تگ‌ها
            tags.append(signal.direction)

            # اضافه کردن تایم‌فریم به تگ‌ها
            tags.append(timeframe)

            # ساخت یادداشت با اطلاعات مفید
            notes = f"ایجاد شده در {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
            if hasattr(signal, 'score') and hasattr(signal.score, 'final_score'):
                notes += f" | امتیاز سیگنال: {signal.score.final_score:.2f}"
                # اضافه کردن امتیازهای جزئی اگر وجود دارند
                if hasattr(signal.score, 'pattern_score'):
                    notes += f" | الگو: {signal.score.pattern_score:.2f}"
                if hasattr(signal.score, 'trend_score'):
                    notes += f" | روند: {signal.score.trend_score:.2f}"
                if hasattr(signal.score, 'volume_score'):
                    notes += f" | حجم: {signal.score.volume_score:.2f}"

            # ایجاد شی Trade با تمام فیلدهای لازم
            trade = Trade(
                trade_id=trade_id,
                symbol=signal.symbol,
                direction=signal.direction,
                entry_price=signal.entry_price,
                stop_loss=signal.stop_loss,
                take_profit=signal.take_profit,
                quantity=quantity,
                risk_amount=calculated_risk_amount,
                timestamp=datetime.now().astimezone(),  # زمان ایجاد معامله
                status='open',  # وضعیت اولیه
                initial_stop_loss=signal.stop_loss,  # ذخیره حد ضرر اولیه
                risk_reward_ratio=signal.risk_reward_ratio,
                # فیلدهای جدید با مقادیر صحیح و معنی‌دار
                entry_reasons_json=entry_reasons_json_str,
                strategy_name=strategy_name,  # استفاده از استراتژی استخراج شده
                timeframe=timeframe,  # استفاده از تایم‌فریم صحیح
                signal_quality=getattr(signal.score, 'final_score', None) if hasattr(signal, 'score') else None,
                market_state=market_state,  # وضعیت بازار
                tags=tags,  # تگ‌های متنوع و معنی‌دار
                notes=notes  # یادداشت‌های مفید با جزئیات سیگنال
            )

            # لاگ کردن اطلاعات معامله جدید
            logger.debug(f"[TRADE_MGR] شی معامله ایجاد شد: ID={trade.trade_id}, استراتژی={trade.strategy_name}, "
                         f"تایم‌فریم={trade.timeframe}, تگ‌ها={trade.tags}")

            # 8. تنظیم Multi-TP (اگر فعال باشد)
            # اطمینان حاصل کنید که از adapted_risk_config استفاده می‌شود
            self._setup_multi_tp_levels(trade, adapted_risk_config)
            logger.debug(f"[TRADE_MGR] تنظیم سطوح Multi-TP برای {trade_id} انجام شد")

            # 9. ذخیره در دیتابیس و فعال کردن در حافظه
            with self._trades_lock:
                self.active_trades[trade_id] = trade

            # تابع save_trade_to_db باید entry_reasons_json و سایر فیلدها را ذخیره کند
            save_successful = self.save_trade_to_db(trade)
            if not save_successful:
                logger.error(f"[TRADE_MGR] خطا در ذخیره معامله جدید {trade_id} در دیتابیس. لغو باز کردن معامله.")
                # حذف از active_trades در صورت خطا در ذخیره
                with self._trades_lock:
                    if trade_id in self.active_trades:
                        del self.active_trades[trade_id]
                return None

            # 10. به‌روزرسانی آمار کلی و تاریخچه بالانس
            self._update_stats()
            self._save_balance_history(f"معامله جدید باز شد: {trade.symbol} {trade.direction}")

            # 11. ارسال اعلان
            precision = self.get_symbol_precision(trade.symbol)
            signal_score_val = trade.signal_quality if trade.signal_quality is not None else 0.0
            notif_msg = (
                f"🚀 معامله جدید [{trade.trade_id[:8]}] 🚀\n"
                f"➡️ {trade.direction.upper()} {trade.symbol} ({trade.timeframe})\n"
                f"📈 قیمت ورود: {trade.entry_price:.{precision}f}\n"
                f"🛡️ استاپ لاس: {trade.stop_loss:.{precision}f}\n"
                f"🎯 تارگت نهایی: {trade.take_profit:.{precision}f} (RR: {trade.risk_reward_ratio:.2f})\n"
                f"💰 اندازه: {trade.quantity:.{precision}f}\n"
                f"مقدار ریسک: ${trade.risk_amount:.2f}\n"
                f"📊 امتیاز: {signal_score_val:.2f}\n"
                f"استراتژی: {trade.strategy_name}"  # همیشه نمایش استراتژی
            )

            # اضافه کردن اطلاعات همبستگی بیت‌کوین به اعلان
            if btc_compatibility_info and 'correlation_type' in btc_compatibility_info:
                corr_type = btc_compatibility_info['correlation_type']
                corr_icon = "🔄" if corr_type == 'positive' else "↔️" if corr_type == 'zero' else "⚡"
                notif_msg += f"\n{corr_icon} همبستگی با BTC: {corr_type}"

            # اضافه کردن اطلاعات سطوح TP به اعلان
            if trade.is_multitp_enabled and trade.take_profit_levels:
                tp_levels_info = "\n📊 سطوح تارگت:\n"
                for i, (price, percentage) in enumerate(trade.take_profit_levels):
                    tp_levels_info += f"  TP{i + 1}: {price:.{precision}f} ({percentage:.1f}%)\n"
                notif_msg += tp_levels_info.rstrip()  # حذف خط خالی احتمالی آخر

            # ارسال اعلان به صورت آسنکرون
            await self._send_notification(notif_msg)

            logger.info(
                f"[TRADE_MGR] معامله جدید {trade_id} با استراتژی '{trade.strategy_name}' و تایم‌فریم '{trade.timeframe}' "
                f"برای {trade.symbol} ({trade.direction}) با موفقیت باز شد. "
                f"اندازه: {trade.quantity:.{precision}f}, امتیاز: {signal_score_val:.2f}")
            return trade_id

        except Exception as e:
            logger.critical(f"[TRADE_MGR] خطای بحرانی در پردازش سیگنال برای {signal.symbol if signal else 'N/A'}: {e}",
                            exc_info=True)
            # ثبت خطا در ردیاب عملکرد
            if hasattr(self, 'performance_tracker'):
                self.performance_tracker.record_error(
                    f"خطا در پردازش سیگنال {signal.symbol if signal else 'N/A'}: {e}", 'error')

            return None

    async def update_trade_parameters(self, active_trades: Optional[List[Trade]] = None) -> int:
        """
        به‌روزرسانی پارامترهای معاملات فعال براساس تنظیمات جدید.

        Args:
            active_trades: لیست معاملات فعال برای به‌روزرسانی، اگر None باشد همه معاملات فعال به‌روز می‌شوند

        Returns:
            تعداد معاملات به‌روزرسانی شده
        """
        logger.info("شروع به‌روزرسانی پارامترهای معاملات فعال...")

        # اگر لیست معاملات فراهم نشده، همه معاملات باز را دریافت کن
        if active_trades is None:
            active_trades = self.get_open_trades()

        if not active_trades:
            logger.info("هیچ معامله فعالی برای به‌روزرسانی وجود ندارد")
            return 0

        updated_count = 0

        for trade in active_trades:
            try:
                # --- 1. به‌روزرسانی پارامترهای استاپ متحرک ---
                if hasattr(trade, 'trailing_stop_params') and isinstance(trade.trailing_stop_params, dict):
                    # استخراج تنظیمات جدید از پیکربندی
                    use_trailing = self.risk_config.get('use_trailing_stop', True)
                    activation_pct = self.risk_config.get('trailing_stop_activation_percent', 3.0)
                    distance_pct = self.risk_config.get('trailing_stop_distance_percent', 2.25)
                    use_atr = self.risk_config.get('use_atr_based_trailing', True)
                    atr_multiplier = self.risk_config.get('atr_trailing_multiplier', 2.25)

                    # ذخیره پارامترهای قبلی برای مقایسه
                    old_params = trade.trailing_stop_params.copy()

                    # به‌روزرسانی پارامترها در معامله
                    trade.trailing_stop_params.update({
                        'enabled': use_trailing,
                        'activation_percent': activation_pct,
                        'distance_percent': distance_pct,
                        'use_atr': use_atr,
                        'atr_multiplier': atr_multiplier
                    })

                    # بررسی تغییرات
                    if trade.trailing_stop_params != old_params:
                        logger.debug(f"پارامترهای استاپ متحرک برای معامله {trade.trade_id} به‌روزرسانی شد")
                        updated_count += 1
                else:
                    # ایجاد پارامترهای استاپ متحرک اگر وجود ندارند
                    trade.trailing_stop_params = {
                        'enabled': self.use_trailing_stop,
                        'activation_percent': self.trailing_stop_activation_percent,
                        'distance_percent': self.trailing_stop_distance_percent,
                        'use_atr': self.use_atr_trailing,
                        'atr_multiplier': self.atr_trailing_multiplier
                    }
                    logger.debug(f"پارامترهای استاپ متحرک برای معامله {trade.trade_id} تنظیم شد")
                    updated_count += 1

                # --- 2. به‌روزرسانی پارامترهای Multi-TP ---
                if self.use_multi_tp != trade.is_multitp_enabled:
                    # فعال/غیرفعال کردن Multi-TP بر اساس تنظیمات
                    old_setting = trade.is_multitp_enabled
                    trade.is_multitp_enabled = self.use_multi_tp

                    if old_setting != trade.is_multitp_enabled:
                        # اگر Multi-TP فعال شده، سطوح را محاسبه کن
                        if trade.is_multitp_enabled and (
                                not trade.take_profit_levels or len(trade.take_profit_levels) == 0):
                            adapted_config, adapted_risk_config = self._get_adapted_config_for_trade(trade)
                            self._setup_multi_tp_levels(trade, adapted_risk_config)
                            logger.debug(f"Multi-TP برای معامله {trade.trade_id} فعال و سطوح محاسبه شد")
                        elif not trade.is_multitp_enabled:
                            logger.debug(f"Multi-TP برای معامله {trade.trade_id} غیرفعال شد")

                        updated_count += 1

                # ذخیره تغییرات در دیتابیس اگر پارامتری به‌روز شده باشد
                if updated_count > 0:
                    self.save_trade_to_db(trade)

            except Exception as e:
                logger.error(f"خطا در به‌روزرسانی پارامترهای معامله {trade.trade_id}: {e}", exc_info=True)

        logger.info(f"پارامترهای {updated_count} معامله فعال با موفقیت به‌روزرسانی شد")
        return updated_count

    def partial_close(self, trade_id: str, exit_price: float, exit_quantity: float, exit_reason: str):
        """بستن بخشی از معامله."""
        with self._trades_lock:
            trade = self.get_trade(trade_id)
            if not trade:
                logger.warning(f"Cannot partially close: Trade {trade_id} not found")
                return False

            # فراخوانی متد partial_close کلاس Trade
            closed_portion = trade.partial_close(exit_price, exit_quantity, exit_reason, self.commission_rate)
            if closed_portion:
                # پس از خروج موفق، در دیتابیس ذخیره می‌کنیم
                self.save_trade_to_db(trade)
                self._update_stats()

                # اگر بخش خارج شده دارای کلید profit_loss است (به جای pnl)، از آن استفاده می‌کنیم
                pnl_value = closed_portion.get('profit_loss', closed_portion.get('pnl', 0))

                # اطلاع‌رسانی
                precision = self.get_symbol_precision(trade.symbol)
                asyncio.create_task(self._send_notification(
                    f"PARTIAL CLOSE [{trade.trade_id[:8]}] {trade.symbol}\n"
                    f"Quantity: {exit_quantity:.{precision}f} @ {exit_price:.{precision}f}\n"
                    f"PnL: {pnl_value:.2f}\n"
                    f"Remaining: {trade.remaining_quantity:.{precision}f}"
                ))

                return True
            return False

    def _get_adapted_config(self, signal: SignalInfo) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """
        دریافت کانفیگ تطبیق‌یافته بر اساس ویژگی‌های سیگنال (مانند وضعیت بازار، نوع سیگنال، تایم‌فریم و غیره).

        Args:
            signal: اطلاعات سیگنال

        Returns:
            Tuple شامل (کانفیگ کامل تطبیق‌یافته، بخش مدیریت ریسک کانفیگ)
        """
        # 1. ابتدا بررسی کنیم آیا سیگنال دارای فیلد adapted_config است
        if hasattr(signal, 'adapted_config') and signal.adapted_config:
            adapted_config = signal.adapted_config
            adapted_risk_config = adapted_config.get('risk_management', self.risk_config)
            logger.debug(f"استفاده از کانفیگ تطبیق‌یافته موجود در سیگنال برای {signal.symbol}")
            return adapted_config, adapted_risk_config

        # 2. بررسی وضعیت بازار برای تطبیق کانفیگ
        market_condition = getattr(signal, 'market_condition', None) or getattr(signal, 'market_state', 'neutral')

        # ایجاد کپی از کانفیگ اصلی برای تطبیق - بدون نیاز به copy.deepcopy
        # روش جایگزین copy.deepcopy با استفاده از json
        try:
            adapted_config = json.loads(json.dumps(self.config))
        except Exception:
            # در صورت خطا، استفاده از روش دستی - کپی ساده
            adapted_config = {}
            for key, value in self.config.items():
                if isinstance(value, dict):
                    adapted_config[key] = {}
                    for k, v in value.items():
                        adapted_config[key][k] = v
                else:
                    adapted_config[key] = value

        # 3. تطبیق بر اساس وضعیت بازار
        if market_condition.lower() == 'bullish':
            # در بازار صعودی، می‌توانیم ریسک بیشتری بپذیریم
            if 'risk_management' in adapted_config:
                current_risk = adapted_config['risk_management'].get('max_risk_per_trade_percent', 1.5)
                adapted_config['risk_management']['max_risk_per_trade_percent'] = min(2.0, current_risk * 1.2)
                # افزایش اندازه پوزیشن ثابت
                if 'fixed_position_size' in adapted_config['risk_management']:
                    adapted_config['risk_management']['fixed_position_size'] = adapted_config['risk_management'][
                                                                                   'fixed_position_size'] * 1.2

                logger.debug(
                    f"تطبیق کانفیگ برای بازار صعودی - افزایش ریسک به {adapted_config['risk_management']['max_risk_per_trade_percent']}%")

        elif market_condition.lower() == 'bearish':
            # در بازار نزولی، ریسک کمتری می‌پذیریم
            if 'risk_management' in adapted_config:
                current_risk = adapted_config['risk_management'].get('max_risk_per_trade_percent', 1.5)
                adapted_config['risk_management']['max_risk_per_trade_percent'] = max(0.5, current_risk * 0.8)
                # کاهش اندازه پوزیشن ثابت
                if 'fixed_position_size' in adapted_config['risk_management']:
                    adapted_config['risk_management']['fixed_position_size'] = adapted_config['risk_management'][
                                                                                   'fixed_position_size'] * 0.8

                logger.debug(
                    f"تطبیق کانفیگ برای بازار نزولی - کاهش ریسک به {adapted_config['risk_management']['max_risk_per_trade_percent']}%")

        # 4. تطبیق بر اساس کیفیت سیگنال
        signal_quality = 0.5  # مقدار پیش‌فرض متوسط
        if hasattr(signal, 'score') and hasattr(signal.score, 'final_score'):
            signal_quality = signal.score.final_score / 100.0  # نرمال‌سازی به بازه 0 تا 1

            # تطبیق براساس کیفیت سیگنال
            if signal_quality > 0.7:  # سیگنال با کیفیت بالا
                if 'risk_management' in adapted_config:
                    # افزایش اندازه پوزیشن برای سیگنال‌های با کیفیت بالا
                    risk_multiplier = 1.0 + min(0.3, (signal_quality - 0.7) * 1.0)  # حداکثر 30% افزایش
                    current_risk = adapted_config['risk_management'].get('max_risk_per_trade_percent', 1.5)
                    adapted_config['risk_management']['max_risk_per_trade_percent'] = min(2.5,
                                                                                          current_risk * risk_multiplier)
                    logger.debug(f"تطبیق کانفیگ برای سیگنال با کیفیت بالا ({signal_quality:.2f}) - "
                                 f"ضریب ریسک: {risk_multiplier:.2f}")

            elif signal_quality < 0.4:  # سیگنال با کیفیت پایین
                if 'risk_management' in adapted_config:
                    # کاهش اندازه پوزیشن برای سیگنال‌های با کیفیت پایین
                    risk_multiplier = max(0.6, 1.0 - (0.4 - signal_quality) * 1.0)  # حداکثر 40% کاهش
                    current_risk = adapted_config['risk_management'].get('max_risk_per_trade_percent', 1.5)
                    adapted_config['risk_management']['max_risk_per_trade_percent'] = max(0.5,
                                                                                          current_risk * risk_multiplier)
                    logger.debug(f"تطبیق کانفیگ برای سیگنال با کیفیت پایین ({signal_quality:.2f}) - "
                                 f"ضریب ریسک: {risk_multiplier:.2f}")

        # 5. تطبیق بر اساس تایم‌فریم
        timeframe = getattr(signal, 'timeframe', '5m')
        if timeframe:
            # برای تایم‌فریم‌های بزرگتر ریسک بیشتر و برای تایم‌فریم‌های کوچکتر ریسک کمتر
            tf_multipliers = {
                '1m': 0.7,
                '3m': 0.8,
                '5m': 0.9,
                '15m': 1.0,
                '30m': 1.1,
                '1h': 1.2,
                '2h': 1.3,
                '4h': 1.4,
                '6h': 1.5,
                '12h': 1.6,
                '1d': 1.7,
                '3d': 1.8,
                '1w': 1.9
            }

            # اعمال ضریب تایم‌فریم برای ریسک
            tf_multiplier = tf_multipliers.get(timeframe, 1.0)

            if 'risk_management' in adapted_config:
                # محدود کردن تغییرات برای جلوگیری از ریسک بیش از حد
                current_risk = adapted_config['risk_management'].get('max_risk_per_trade_percent', 1.5)
                adapted_risk = current_risk * tf_multiplier

                # محدود کردن در بازه منطقی (نیم تا دو و نیم درصد)
                adapted_config['risk_management']['max_risk_per_trade_percent'] = max(0.5, min(2.5, adapted_risk))

                logger.debug(f"تطبیق کانفیگ برای تایم‌فریم {timeframe} - ضریب: {tf_multiplier:.2f}, "
                             f"ریسک نهایی: {adapted_config['risk_management']['max_risk_per_trade_percent']:.2f}%")

        # 6. تطبیق بر اساس نوع سیگنال
        signal_type = getattr(signal, 'signal_type', None) or getattr(signal, 'strategy_name', None)
        if signal_type:
            # نگاشت نوع‌های سیگنال به ضرایب ریسک
            signal_type_multipliers = {
                'breakout': 1.2,  # ریسک بیشتر برای سیگنال‌های شکست
                'reversal': 0.8,  # ریسک کمتر برای سیگنال‌های برگشت روند
                'trend_following': 1.1,  # ریسک نسبتاً بالا برای سیگنال‌های دنبال کردن روند
                'range_bounce': 0.9,  # ریسک کمتر برای سیگنال‌های بازگشت از محدوده
                'divergence': 0.85,  # ریسک کمتر برای سیگنال‌های واگرایی
                'support_resistance': 1.0,  # ریسک متوسط برای سیگنال‌های حمایت/مقاومت
                # اضافه کردن سایر انواع سیگنال‌ها...
            }

            # پیدا کردن اولین بخش نام (قبل از اولین فاصله یا آندرلاین) برای تطبیق عمومی‌تر
            first_part = signal_type.split('_')[0].split(' ')[0].lower()
            matched_key = None

            # بررسی تطبیق دقیق
            if signal_type.lower() in signal_type_multipliers:
                matched_key = signal_type.lower()
            # بررسی تطبیق جزئی
            else:
                for key in signal_type_multipliers.keys():
                    if first_part in key.lower() or key.lower() in first_part:
                        matched_key = key
                        break

            # اعمال ضریب مناسب بر اساس نوع سیگنال
            if matched_key:
                st_multiplier = signal_type_multipliers[matched_key]

                if 'risk_management' in adapted_config:
                    current_risk = adapted_config['risk_management'].get('max_risk_per_trade_percent', 1.5)
                    adapted_risk = current_risk * st_multiplier

                    # محدود کردن در بازه منطقی
                    adapted_config['risk_management']['max_risk_per_trade_percent'] = max(0.5, min(2.5, adapted_risk))

                    logger.debug(f"تطبیق کانفیگ برای نوع سیگنال {signal_type} (تطبیق با {matched_key}) - "
                                 f"ضریب: {st_multiplier:.2f}, ریسک نهایی: {adapted_config['risk_management']['max_risk_per_trade_percent']:.2f}%")

        # دریافت بخش مدیریت ریسک از کانفیگ تطبیق‌یافته
        adapted_risk_config = adapted_config.get('risk_management', self.risk_config)

        # اطمینان از مقادیر معتبر
        if 'max_risk_per_trade_percent' in adapted_risk_config:
            adapted_risk_config['max_risk_per_trade_percent'] = max(0.1, min(5.0, adapted_risk_config[
                'max_risk_per_trade_percent']))

        return adapted_config, adapted_risk_config

    def _calculate_risk_adjustment_factors(self, signal: SignalInfo) -> Dict[str, float]:
        """Calculates risk adjustment factors."""
        # Default: return default factors
        return {
            'signal_factor': 1.0, 'drawdown_factor': 1.0, 'streak_factor': 1.0,
            'market_factor': 1.0, 'volatility_factor': 1.0, 'combined_factor': 1.0
        }

    def get_symbol_precision(self, symbol: str) -> int:
        """Gets decimal precision for a symbol's quantity."""
        # Simple precision logic - replace with actual logic if needed
        base = symbol.split('/')[0] if '/' in symbol else symbol
        if base in ["BTC", "XBT", "ETH"]: return 6
        if base in ["SOL", "BNB", "XRP", "ADA"]: return 4
        if base in ["DOGE", "SHIB"]: return 8
        return 6  # Default

    def _update_stats(self):
        """به‌روزرسانی آمار کلی معاملات و حساب."""
        with self._trades_lock:
            all_db_trades = []
            try:
                if self.cursor:
                    self.cursor.execute("SELECT data FROM trades WHERE status = 'closed'")
                    rows = self.cursor.fetchall()
                    for row in rows:
                        if row and row['data']:
                            try:
                                all_db_trades.append(Trade.from_dict(json.loads(row['data'])))
                            except Exception as parse_err:
                                logger.error(f"Failed to parse trade data from DB: {parse_err}")
                        else:
                            logger.warning("Fetched empty or invalid row from trades table.")

            except sqlite3.Error as db_err:
                logger.error(f"Error fetching closed trades for stats: {db_err}")
            except Exception as e:
                logger.error(f"Unexpected error fetching closed trades: {e}")

            # *** تغییر اصلی اینجا: ریست آمار و مقداردهی اولیه max_drawdown ***
            self.stats = self._reset_stats()  # ریست کردن آمار
            self.stats['max_drawdown'] = 0.0  # مقداردهی اولیه max_drawdown
            self.stats['max_drawdown_percent'] = 0.0  # مقداردهی اولیه درصد drawdown
            # *** پایان تغییر ***

            # لیست معاملات فعال (باز + بخشی بسته شده)
            open_trades_list = self.get_open_trades()
            partially_closed_list = [t for t in open_trades_list if t.status == 'partially_closed']

            # به‌روزرسانی آمار معاملات
            self.stats['open_trades'] = len(open_trades_list)
            self.stats['partially_closed_trades'] = len(partially_closed_list)
            self.stats['closed_trades'] = len(all_db_trades)
            self.stats['total_trades_opened'] = self.stats['open_trades'] + self.stats['closed_trades']

            # آمار معاملات بسته شده
            self._calculate_closed_trades_stats(all_db_trades)

            # آمار معاملات بخشی بسته شده
            self._calculate_partially_closed_stats(partially_closed_list)

            # محاسبه‌های پیشرفته
            self._calculate_advanced_stats(all_db_trades)  # حالا max_drawdown مقداردهی اولیه شده

            # محاسبه توزیع نمادها، استراتژی‌ها و برچسب‌ها
            self._calculate_distribution_stats(all_db_trades + open_trades_list)

            # محاسبه افت سرمایه و اکوئیتی فعلی
            self._calculate_current_drawdown()  # این تابع آمار اکوئیتی و drawdown را به‌روز می‌کند

            # گرد کردن مقادیر نهایی
            for key in ['total_net_profit_loss', 'total_commission_paid', 'current_equity', 'current_open_pnl']:
                if self.stats.get(key) is not None:
                    self.stats[key] = round(self.stats[key], 2)
            for key in ['win_rate', 'current_drawdown', 'avg_win_percent', 'avg_loss_percent']:
                if self.stats.get(key) is not None:
                    self.stats[key] = round(self.stats[key], 2)
            if self.stats.get('profit_factor') is not None:
                self.stats['profit_factor'] = round(self.stats['profit_factor'], 2)
            # اطمینان از گرد کردن drawdown
            if self.stats.get('max_drawdown') is not None:
                self.stats['max_drawdown'] = round(self.stats['max_drawdown'], 2)
            if self.stats.get('max_drawdown_percent') is not None:
                self.stats['max_drawdown_percent'] = round(self.stats['max_drawdown_percent'], 2)

            logger.debug(f"Stats Updated: Open={self.stats['open_trades']}, Closed={self.stats['closed_trades']}, "
                         f"WinRate={self.stats.get('win_rate', 'N/A')}%, PF={self.stats.get('profit_factor', 'N/A')}, "
                         f"Equity={self.stats.get('current_equity', 'N/A')}, Drawdown={self.stats.get('current_drawdown', 'N/A')}%")

    def _validate_signal_prices(self, signal: SignalInfo) -> bool:
        """
        بررسی معتبر بودن قیمت‌های ورود، حد ضرر و حد سود.

        Args:
            signal: اطلاعات سیگنال

        Returns:
            True اگر قیمت‌ها معتبر باشند، False در غیر این صورت
        """
        # تبدیل به float در صورت نیاز
        try:
            entry_price = float(signal.entry_price) if isinstance(signal.entry_price, str) else signal.entry_price
            stop_loss = float(signal.stop_loss) if isinstance(signal.stop_loss, str) else signal.stop_loss
            take_profit = float(signal.take_profit) if isinstance(signal.take_profit, str) else signal.take_profit
            logger.debug(
                f"[VALIDATE] بررسی اعتبار قیمت‌ها برای {signal.symbol}: ورود:{entry_price}, SL:{stop_loss}, TP:{take_profit}")
        except (ValueError, TypeError):
            logger.error(f"[VALIDATE] فرمت قیمت نامعتبر در سیگنال برای {signal.symbol}")
            return False

        # بررسی مقادیر صفر یا منفی
        if entry_price <= 0:
            logger.error(f"[VALIDATE] قیمت ورود نامعتبر برای {signal.symbol}: {entry_price}")
            return False

        if stop_loss <= 0:
            logger.error(f"[VALIDATE] حد ضرر نامعتبر برای {signal.symbol}: {stop_loss}")
            return False

        if take_profit <= 0:
            logger.error(f"[VALIDATE] حد سود نامعتبر برای {signal.symbol}: {take_profit}")
            return False

        # بررسی جهت و ترتیب قیمت‌ها
        if signal.direction == 'long':
            if stop_loss >= entry_price:
                logger.error(f"[VALIDATE] حد ضرر نامعتبر برای خرید {signal.symbol}: {stop_loss} >= {entry_price}")
                return False
            if take_profit <= entry_price:
                logger.error(f"[VALIDATE] حد سود نامعتبر برای خرید {signal.symbol}: {take_profit} <= {entry_price}")
                return False

            # بررسی فاصله حداقلی
            min_sl_distance = entry_price * 0.001  # حداقل 0.1% فاصله
            min_tp_distance = entry_price * 0.001

            if entry_price - stop_loss < min_sl_distance:
                logger.error(
                    f"[VALIDATE] فاصله حد ضرر تا ورود برای {signal.symbol} خیلی کم است: {entry_price - stop_loss:.8f} < {min_sl_distance:.8f}"
                )
                return False

            if take_profit - entry_price < min_tp_distance:
                logger.error(
                    f"[VALIDATE] فاصله حد سود تا ورود برای {signal.symbol} خیلی کم است: {take_profit - entry_price:.8f} < {min_tp_distance:.8f}"
                )
                return False

        else:  # short
            if stop_loss <= entry_price:
                logger.error(f"[VALIDATE] حد ضرر نامعتبر برای فروش {signal.symbol}: {stop_loss} <= {entry_price}")
                return False
            if take_profit >= entry_price:
                logger.error(f"[VALIDATE] حد سود نامعتبر برای فروش {signal.symbol}: {take_profit} >= {entry_price}")
                return False

            # بررسی فاصله حداقلی
            min_sl_distance = entry_price * 0.001  # حداقل 0.1% فاصله
            min_tp_distance = entry_price * 0.001

            if stop_loss - entry_price < min_sl_distance:
                logger.error(
                    f"[VALIDATE] فاصله حد ضرر تا ورود برای {signal.symbol} خیلی کم است: {stop_loss - entry_price:.8f} < {min_sl_distance:.8f}"
                )
                return False

            if entry_price - take_profit < min_tp_distance:
                logger.error(
                    f"[VALIDATE] فاصله حد سود تا ورود برای {signal.symbol} خیلی کم است: {entry_price - take_profit:.8f} < {min_tp_distance:.8f}"
                )
                return False

        # بررسی نسبت ریسک به ریوارد حداقلی - کاهش آستانه
        rr = signal.risk_reward_ratio
        min_rr = self.risk_config.get('min_risk_reward_ratio', 1.5)  # کاهش از 1.8 به 1.5
        if rr < min_rr:
            logger.error(f"[VALIDATE] نسبت ریسک به ریوارد برای {signal.symbol} خیلی کم است: {rr:.2f} < {min_rr}")
            return False

        logger.debug(f"[VALIDATE] قیمت‌های سیگنال برای {signal.symbol} معتبر است. RR: {rr:.2f}")
        return True

    def can_open_new_trade(self, symbol: str, direction: str) -> bool:
        """بررسی امکان باز کردن معامله جدید (بدون چک همبستگی)."""
        with self._trades_lock:
            open_trades = [t for t in self.active_trades.values() if t.status != 'closed']
            open_trades_count = len(open_trades)

            logger.debug(f"[TRADE_MGR] بررسی امکان باز کردن معامله جدید برای {symbol}. "
                         f"تعداد معاملات فعلی: {open_trades_count}/{self.max_open_trades}")

            if open_trades_count >= self.max_open_trades:
                logger.info(
                    f"[TRADE_MGR] امکان باز کردن {symbol} وجود ندارد: تعداد معاملات باز به حداکثر رسیده ({self.max_open_trades})")
                return False

            symbol_trades = [t for t in open_trades if t.symbol == symbol]
            symbol_trades_count = len(symbol_trades)

            if symbol_trades_count >= self.max_trades_per_symbol:
                logger.info(
                    f"[TRADE_MGR] امکان باز کردن {symbol} وجود ندارد: تعداد معاملات برای این نماد به حداکثر رسیده ({self.max_trades_per_symbol})")
                return False

            # بررسی معامله در جهت مخالف
            if not self.allow_opposite_trades:
                has_opposite = any(t.direction != direction for t in symbol_trades)
                if has_opposite:
                    logger.info(
                        f"[TRADE_MGR] امکان باز کردن {direction} {symbol} وجود ندارد: معامله در جهت مخالف وجود دارد و allow_opposite_trades غیرفعال است."
                    )
                    return False

            logger.debug(f"[TRADE_MGR] امکان باز کردن معامله جدید برای {symbol} ({direction}) وجود دارد")
            return True

    async def check_correlation(self, symbol: str, direction: str) -> Tuple[bool, float, List[str]]:
        """بررسی همبستگی با معاملات باز."""
        # استفاده از CorrelationManager اگر موجود باشد
        if self.correlation_manager is not None:
            try:
                open_trades = self.get_open_trades()
                # کش کردن کلید همبستگی برای بازیابی سریع
                cache_key = f"corr_{symbol}_{direction}_{','.join([t.symbol for t in open_trades])}"

                if cache_key in self._correlation_cache:
                    return self._correlation_cache[cache_key]

                # استفاده از correlation_manager برای بررسی
                is_allowed, level, symbols = await self.correlation_manager.check_portfolio_correlation(
                    symbol, direction, open_trades, self.data_fetcher_instance
                )

                # ذخیره نتیجه در کش
                self._correlation_cache[cache_key] = (is_allowed, level, symbols)
                return is_allowed, level, symbols

            except Exception as e:
                logger.error(f"خطا در استفاده از correlation manager: {e}")
                # در صورت خطا، به روش عادی ادامه می‌دهیم

        # روش عادی اگر correlation_manager موجود نباشد
        if not self.use_correlation_check or self.data_fetcher_instance is None:
            return True, 0.0, []  # اگر غیرفعال یا data_fetcher ناموجود است، مجاز است

        try:
            # کش کردن کلید همبستگی برای بازیابی سریع
            open_trades = self.get_open_trades()
            cache_key = f"corr_{symbol}_{direction}_{','.join([t.symbol for t in open_trades])}"

            if cache_key in self._correlation_cache:
                return self._correlation_cache[cache_key]

            max_corr = self.risk_config.get('max_correlation_threshold', 0.7)
            max_corr_trades = self.risk_config.get('max_correlated_trades', 2)
            corr_tf = self.risk_config.get('correlation_timeframe', '1h')
            corr_period = self.risk_config.get('correlation_period', 100)

            if not open_trades:
                return True, 0.0, []

            # دریافت داده برای نماد هدف
            # استفاده از وهله data_fetcher که ثبت شده
            df_target = await self.data_fetcher_instance.get_historical_data(symbol, corr_tf, limit=corr_period)
            if df_target is None or df_target.empty:
                logger.warning(f"Cannot check correlation for {symbol}: Failed to fetch target data.")
                return True, 0.0, []  # در صورت خطا، مجاز فرض کن

            correlated_symbols_list = []
            max_found_correlation = 0.0
            symbols_checked = set()  # برای جلوگیری از بررسی تکراری

            for trade in open_trades:
                if trade.symbol == symbol or trade.symbol in symbols_checked:
                    continue
                symbols_checked.add(trade.symbol)

                df_open = await self.data_fetcher_instance.get_historical_data(trade.symbol, corr_tf, limit=corr_period)
                if df_open is None or df_open.empty:
                    continue

                # تراز کردن داده‌ها بر اساس زمان
                common_index = df_target.index.intersection(df_open.index)
                if len(common_index) < MIN_CORRELATION_DATA_POINTS:
                    continue  # حداقل داده برای همبستگی

                # استفاده از روش کارآمدتر برای محاسبه همبستگی
                df1_aligned = df_target.loc[common_index, 'close'].pct_change().dropna()
                df2_aligned = df_open.loc[common_index, 'close'].pct_change().dropna()

                if len(df1_aligned) < MIN_CORRELATION_DATA_POINTS or len(df2_aligned) < MIN_CORRELATION_DATA_POINTS:
                    continue

                # استفاده از numpy برای محاسبه سریع‌تر
                correlation = np.corrcoef(df1_aligned.values, df2_aligned.values)[0, 1]

                # مدیریت مقادیر NaN یا Inf
                if np.isnan(correlation) or np.isinf(correlation):
                    correlation = 0.0

                abs_correlation = abs(correlation)
                max_found_correlation = max(max_found_correlation, abs_correlation)

                # در نظر گرفتن جهت معاملات
                effective_correlation = correlation
                if trade.direction != direction:
                    effective_correlation = -correlation

                if abs_correlation > max_corr:
                    correlated_symbols_list.append(trade.symbol)
                    logger.debug(
                        f"High correlation detected: {symbol} ({direction}) vs {trade.symbol} ({trade.direction}) = {correlation:.3f}")

            # بررسی تعداد نمادهای با همبستگی بالا
            result = (len(correlated_symbols_list) < max_corr_trades, max_found_correlation, correlated_symbols_list)

            # ذخیره نتیجه در کش
            self._correlation_cache[cache_key] = result

            if not result[0]:
                logger.warning(
                    f"Trade for {symbol} rejected. Correlated with {len(correlated_symbols_list)} open trades (>{max_corr_trades}): {correlated_symbols_list}")

            return result

        except Exception as e:
            logger.error(f"Error during correlation check for {symbol}: {e}")
            return True, 0.0, []  # در صورت خطا، مجاز فرض کن

    def _generate_trade_id(self, symbol: str, direction: str) -> str:
        """Generates a unique trade ID."""
        timestamp_str = datetime.now().strftime('%Y%m%d%H%M%S%f')
        safe_symbol = ''.join(c for c in symbol if c.isalnum())
        return f"{safe_symbol[:6]}_{direction[:3]}_{timestamp_str}_{uuid.uuid4().hex[:4]}"

    def _setup_multi_tp_levels(self, trade: Trade, risk_config: Dict[str, Any]):
        """تنظیم سطوح حد سود چند مرحله‌ای."""
        # بررسی فعال بودن multi-TP
        multi_tp_enabled = risk_config.get('use_multi_tp', True)
        if not multi_tp_enabled:
            trade.is_multitp_enabled = False
            logger.debug(f"Multi-TP disabled in config for {trade.symbol}.")
            return

        # محاسبه سطوح TP
        rr = trade.risk_reward_ratio
        entry = trade.entry_price
        final_tp = trade.take_profit
        direction = trade.direction

        # محاسبه فاصله ریسک
        risk_dist = abs(entry - trade.initial_stop_loss)

        # اطمینان از معنادار بودن فاصله ریسک
        if risk_dist <= entry * 0.0001:  # اگر کمتر از 0.01% قیمت ورود باشد
            logger.warning(
                f"Risk distance too small for Multi-TP calculation for {trade.symbol}. Using fallback value.")
            risk_dist = entry * 0.01  # استفاده از 1% قیمت ورود به عنوان حداقل فاصله معنادار

        # اطمینان از معنادار بودن TP نهایی نسبت به ورود
        min_tp_distance = entry * 0.005  # حداقل 0.5% فاصله
        if direction == 'long' and final_tp - entry < min_tp_distance:
            logger.warning(f"Final TP too close to entry for {trade.symbol}. Adjusting.")
            final_tp = entry * 1.02  # 2% بالاتر از ورود
        elif direction == 'short' and entry - final_tp < min_tp_distance:
            logger.warning(f"Final TP too close to entry for {trade.symbol}. Adjusting.")
            final_tp = entry * 0.98  # 2% پایین‌تر از ورود

        # دقت قیمت - استفاده از دقت بیشتر
        precision = max(8, self.get_symbol_precision(trade.symbol))
        tp_levels = []

        # جایگزین کردن استراتژی‌های فیبوناچی برای سطوح TP
        if rr >= 3.5:
            # استراتژی فیبوناچی برای RR بالا
            if direction == 'long':
                # استفاده از سطوح 0.382, 0.618, 1.0 از فاصله تا TP
                tp_distance = final_tp - entry
                tp1 = entry + (tp_distance * 0.382)
                tp2 = entry + (tp_distance * 0.618)
                tp3 = final_tp
            else:
                # معکوس برای short
                tp_distance = entry - final_tp
                tp1 = entry - (tp_distance * 0.382)
                tp2 = entry - (tp_distance * 0.618)
                tp3 = final_tp

            # اصلاح مقادیر
            tp1 = round(tp1, precision)
            tp2 = round(tp2, precision)
            tp3 = round(tp3, precision)

            # تنظیم سطوح TP با درصدهای توزیع شده
            tp_levels = [(tp1, 30), (tp2, 40), (tp3, 30)]

        elif rr >= 2.0:
            # استراتژی 2 سطح برای RR متوسط
            if direction == 'long':
                # استفاده از سطوح 0.5, 1.0 از فاصله تا TP
                tp_distance = final_tp - entry
                tp1 = entry + (tp_distance * 0.5)
                tp2 = final_tp
            else:
                # معکوس برای short
                tp_distance = entry - final_tp
                tp1 = entry - (tp_distance * 0.5)
                tp2 = final_tp

            # اصلاح مقادیر
            tp1 = round(tp1, precision)
            tp2 = round(tp2, precision)

            # تنظیم سطوح TP با درصدهای متناسب
            tp_levels = [(tp1, 50), (tp2, 50)]

        elif rr >= 1.0:
            # برای RR کم، فقط یک سطح TP
            tp_levels = [(final_tp, 100)]

        # تنظیم سطوح در معامله
        if tp_levels:
            try:
                # چاپ سطوح برای دیباگ
                level_str = ', '.join([f"({price:.{precision}f}, {pct}%)" for price, pct in tp_levels])
                logger.info(f"Setting multi-TP levels for {trade.symbol}: {level_str}")

                # اطمینان از صفر نبودن قیمت‌ها
                has_zero_price = any(abs(price) < 1e-6 for price, _ in tp_levels)
                if has_zero_price:
                    logger.error(f"Zero or near-zero TP level detected for {trade.symbol}! Using single TP.")
                    trade.is_multitp_enabled = False
                    return

                trade.set_multi_take_profit(tp_levels)
            except ValueError as e:
                logger.error(f"Error setting multi-TP for {trade.trade_id}: {e}")
                trade.is_multitp_enabled = False
        else:
            logger.debug(f"Multi-TP not applied for {trade.symbol} (RR: {rr:.2f}). Using single TP.")
            trade.is_multitp_enabled = False

    def _save_balance_history(self, description: str = ""):
        """ذخیره تاریخچه تغییر موجودی در دیتابیس."""
        if not self.conn or not self.cursor:
            return False

        try:
            # به‌روزرسانی آمار برای اطمینان از مقادیر صحیح
            self._update_stats()

            # محاسبه مقادیر
            timestamp = datetime.now().astimezone().isoformat()
            balance = self.initial_balance
            equity = self.stats.get('current_equity', self.initial_balance)
            open_pnl = self.stats.get('current_open_pnl', 0.0)

            with self._db_lock:
                # فقط با ستون‌های اصلی، بدون description
                try:
                    self.cursor.execute(
                        "INSERT INTO balance_history (timestamp, balance, equity, open_pnl) VALUES (?, ?, ?, ?)",
                        (timestamp, balance, equity, open_pnl)
                    )
                    self.conn.commit()
                    return True
                except sqlite3.Error as e:
                    # اگر هنوز مشکل وجود دارد
                    if "no column named open_pnl" in str(e):
                        logger.warning("Column 'open_pnl' missing. Trying to add it...")
                        self._add_missing_db_column('balance_history', 'open_pnl', 'REAL')
                        # بدون ستون open_pnl تلاش کنیم
                        try:
                            self.cursor.execute(
                                "INSERT INTO balance_history (timestamp, balance, equity) VALUES (?, ?, ?)",
                                (timestamp, balance, equity)
                            )
                            self.conn.commit()
                            return True
                        except:
                            logger.error("Cannot save balance history - database schema issues")
                            return False
                    else:
                        logger.error(f"Error saving balance history: {e}")
                        return False
        except Exception as e:
            logger.error(f"Unexpected error saving balance history: {e}")
            return False

    async def _send_notification(self, message: str):
        """ارسال امن اعلان."""
        if self.notification_callback:
            try:
                await self.notification_callback(message)
            except Exception as e:
                logger.error(f"Error sending notification: {e}")

    def _ensure_trade_has_required_fields(self, trade: Trade) -> None:
        """
        اطمینان از وجود تمام فیلدهای ضروری در شیء Trade و تنظیم مقادیر پیش‌فرض در صورت نیاز.

        Args:
            trade: شیء Trade مورد بررسی
        """
        try:
            # بررسی و تنظیم فیلدهای اصلی
            if not hasattr(trade, 'tags') or trade.tags is None:
                trade.tags = []

            if not hasattr(trade, 'strategy_name') or not trade.strategy_name:
                # تلاش برای استخراج استراتژی از سایر فیلدها
                if hasattr(trade, 'direction'):
                    prefix = 'Long' if trade.direction == 'long' else 'Short'
                    timeframe = getattr(trade, 'timeframe', '5m') if hasattr(trade, 'timeframe') else '5m'
                    trade.strategy_name = f"{prefix}_{timeframe}"
                else:
                    trade.strategy_name = "unknown_strategy"

            if not hasattr(trade, 'timeframe') or not trade.timeframe:
                trade.timeframe = '5m'  # تایم‌فریم پیش‌فرض

            if not hasattr(trade, 'signal_quality') or trade.signal_quality is None:
                trade.signal_quality = 50.0  # مقدار پیش‌فرض متوسط

            if not hasattr(trade, 'market_state') or not trade.market_state:
                if hasattr(trade, 'direction'):
                    # تنظیم پیش‌فرض بر اساس جهت معامله
                    trade.market_state = 'bullish' if trade.direction == 'long' else 'bearish'
                else:
                    trade.market_state = 'neutral'  # وضعیت بازار پیش‌فرض

            if not hasattr(trade, 'notes') or not trade.notes:
                trade.notes = f"بازسازی شده در {datetime.now().astimezone().isoformat()}"

            if not hasattr(trade, 'entry_reasons_json') or not trade.entry_reasons_json:
                default_reason = {
                    "info": "دلایل ورود در دسترس نیست",
                    "reconstructed": True,
                    "timestamp": datetime.now().isoformat()
                }
                trade.entry_reasons_json = json.dumps(default_reason, ensure_ascii=False)

            if not hasattr(trade, 'stop_moved_count'):
                trade.stop_moved_count = 0

            if not hasattr(trade, 'commission_paid'):
                trade.commission_paid = 0.0

            if not hasattr(trade, 'initial_stop_loss') or trade.initial_stop_loss is None:
                trade.initial_stop_loss = trade.stop_loss if hasattr(trade, 'stop_loss') else 0.0

            if not hasattr(trade, 'max_favorable_excursion'):
                trade.max_favorable_excursion = 0.0

            if not hasattr(trade, 'max_adverse_excursion'):
                trade.max_adverse_excursion = 0.0

            if not hasattr(trade,
                           'risk_reward_ratio') or trade.risk_reward_ratio is None or trade.risk_reward_ratio <= 0:
                # محاسبه نسبت ریسک به ریوارد اگر امکان‌پذیر باشد
                if (hasattr(trade, 'entry_price') and trade.entry_price and
                        hasattr(trade, 'stop_loss') and trade.stop_loss and
                        hasattr(trade, 'take_profit') and trade.take_profit):

                    risk_dist = abs(trade.entry_price - trade.stop_loss)
                    reward_dist = abs(trade.take_profit - trade.entry_price)
                    if risk_dist > 1e-9:
                        trade.risk_reward_ratio = reward_dist / risk_dist
                    else:
                        trade.risk_reward_ratio = 2.0  # مقدار پیش‌فرض
                else:
                    trade.risk_reward_ratio = 2.0  # مقدار پیش‌فرض

            # اطمینان از مقدار صحیح برای فیلدهای ضروری
            if not hasattr(trade, 'closed_portions') or trade.closed_portions is None:
                trade.closed_portions = []

            if not hasattr(trade, 'remaining_quantity') or trade.remaining_quantity is None:
                if hasattr(trade, 'status') and trade.status == 'closed':
                    trade.remaining_quantity = 0.0
                elif hasattr(trade, 'quantity'):
                    trade.remaining_quantity = trade.quantity
                else:
                    trade.remaining_quantity = 0.0

            if not hasattr(trade, 'current_price') or trade.current_price is None:
                if hasattr(trade, 'entry_price'):
                    trade.current_price = trade.entry_price
                else:
                    trade.current_price = 0.0

            # اطمینان از is_multitp_enabled و take_profit_levels
            if not hasattr(trade, 'is_multitp_enabled'):
                trade.is_multitp_enabled = False

            if not hasattr(trade, 'take_profit_levels') or trade.take_profit_levels is None:
                trade.take_profit_levels = []

            if not hasattr(trade, 'current_tp_level_index'):
                trade.current_tp_level_index = 0

            # اطمینان از وجود پارامترهای trailing stop
            if not hasattr(trade, 'trailing_stop_params') or trade.trailing_stop_params is None:
                trade.trailing_stop_params = {
                    'enabled': self.use_trailing_stop,
                    'activation_percent': self.trailing_stop_activation_percent,
                    'distance_percent': self.trailing_stop_distance_percent,
                    'use_atr': self.use_atr_trailing,
                    'atr_multiplier': self.atr_trailing_multiplier,
                    'activated': False,
                    'highest_price': trade.entry_price,
                    'lowest_price': trade.entry_price
                }

            # اطمینان از وجود فیلدهای آماری
            if not hasattr(trade, 'trade_stats') or trade.trade_stats is None:
                trade.trade_stats = {}

        except Exception as e:
            logger.error(
                f"خطا در تنظیم فیلدهای ضروری برای معامله {trade.trade_id if hasattr(trade, 'trade_id') else 'نامشخص'}: {e}")

    def load_active_trades(self):
        """بارگذاری معاملات فعال (open یا partially_closed) از دیتابیس با مدیریت بهتر خطا و اطمینان از داده‌های معتبر."""
        if not self.cursor:
            logger.error("اتصال به دیتابیس برای بارگذاری معاملات فعال ایجاد نشده است.")
            return False

        try:
            with self._trades_lock:
                self.active_trades = {}
                # بارگذاری معاملاتی که بسته نشده‌اند
                self.cursor.execute("""
                SELECT 
                    trade_id, symbol, timestamp, status, direction, 
                    entry_price, stop_loss, take_profit, quantity, 
                    remaining_quantity, current_price, exit_time, 
                    exit_reason, profit_loss, commission_paid,
                    tags, strategy_name, timeframe, market_state, notes,
                    entry_reasons_json, data
                FROM trades 
                WHERE status != 'closed'
                """)
                rows = self.cursor.fetchall()

                loaded_count = 0
                loaded_symbols = set()
                error_count = 0

                for row in rows:
                    try:
                        # اول سعی کنید از ستون data استفاده کنید
                        if row['data']:
                            try:
                                trade_data = json.loads(row['data'])
                                trade = Trade.from_dict(trade_data)

                                # بررسی کنید که فیلدهای مهم معتبر باشند
                                if not trade.symbol or not trade.direction or not trade.entry_price:
                                    logger.warning(f"معامله {row['trade_id']} داده‌های ضروری ناقص دارد. بازسازی...")

                                    # بازسازی معامله با داده‌های مستقیم از دیتابیس
                                    trade = self._recreate_trade_from_db_row(row)

                                # اطمینان از اینکه فیلدهای مهم مقادیر پیش‌فرض داشته باشند
                                self._ensure_trade_has_required_fields(trade)

                                # اضافه کردن به لیست معاملات فعال
                                self.active_trades[trade.trade_id] = trade
                                loaded_count += 1
                                loaded_symbols.add(trade.symbol)

                            except (json.JSONDecodeError, ValueError, KeyError) as json_err:
                                logger.error(f"خطا در رمزگشایی داده‌های معامله {row['trade_id']}: {json_err}")
                                # سعی کنید مستقیماً از ستون‌های دیتابیس استفاده کنید
                                trade = self._recreate_trade_from_db_row(row)

                                if trade:
                                    # اطمینان از اینکه فیلدهای مهم مقادیر پیش‌فرض داشته باشند
                                    self._ensure_trade_has_required_fields(trade)

                                    self.active_trades[trade.trade_id] = trade
                                    loaded_count += 1
                                    loaded_symbols.add(trade.symbol)
                                else:
                                    error_count += 1
                        else:
                            # اگر ستون data خالی است، مستقیماً از ستون‌های دیگر استفاده کنید
                            trade = self._recreate_trade_from_db_row(row)

                            if trade:
                                # اطمینان از اینکه فیلدهای مهم مقادیر پیش‌فرض داشته باشند
                                self._ensure_trade_has_required_fields(trade)

                                self.active_trades[trade.trade_id] = trade
                                loaded_count += 1
                                loaded_symbols.add(trade.symbol)
                            else:
                                error_count += 1

                    except Exception as e:
                        logger.error(f"خطا در بازسازی معامله از دیتابیس: {e}", exc_info=True)
                        error_count += 1

                logger.info(f"{loaded_count} معامله فعال برای {len(loaded_symbols)} نماد از دیتابیس بارگذاری شد. "
                            f"خطاها: {error_count}")
                return True

        except sqlite3.Error as e:
            logger.error(f"خطای دیتابیس هنگام بارگذاری معاملات فعال: {e}")
        except Exception as e:
            logger.error(f"خطای غیرمنتظره در بارگذاری معاملات فعال: {e}", exc_info=True)

        return False

    def _recreate_trade_from_db_row(self, row) -> Optional[Trade]:
        """
        بازسازی شیء Trade از داده‌های مستقیم سطر دیتابیس با مدیریت بهتر خطاها و مقادیر پیش‌فرض.

        Args:
            row: سطر داده‌های دیتابیس

        Returns:
            شیء Trade بازسازی شده یا None در صورت بروز خطا
        """
        try:
            # 1. استخراج فیلدهای ضروری اصلی و بررسی صحت آنها
            if not row or 'trade_id' not in row or not row['trade_id']:
                logger.error("سطر دیتابیس نامعتبر یا فاقد شناسه (trade_id) است")
                return None

            trade_id = row['trade_id']
            symbol = row.get('symbol')

            if not symbol:
                logger.error(f"معامله {trade_id} فاقد نماد (symbol) است. بازسازی امکان‌پذیر نیست.")
                return None

            # 2. تبدیل زمان‌ها به شیء datetime
            # تبدیل timestamp به datetime
            timestamp_str = row.get('timestamp')
            if timestamp_str:
                try:
                    timestamp = datetime.fromisoformat(timestamp_str)
                    # اطمینان از timezone-aware بودن
                    if timestamp.tzinfo is None:
                        timestamp = timestamp.astimezone()
                except (ValueError, TypeError):
                    logger.warning(f"فرمت timestamp نامعتبر برای {trade_id}. استفاده از زمان فعلی.")
                    timestamp = datetime.now().astimezone()
            else:
                logger.warning(f"timestamp برای {trade_id} یافت نشد. استفاده از زمان فعلی.")
                timestamp = datetime.now().astimezone()

            # تبدیل exit_time به datetime اگر وجود دارد
            exit_time = None
            exit_time_str = row.get('exit_time')
            if exit_time_str:
                try:
                    exit_time = datetime.fromisoformat(exit_time_str)
                    # اطمینان از timezone-aware بودن
                    if exit_time.tzinfo is None:
                        exit_time = exit_time.astimezone()
                except (ValueError, TypeError):
                    logger.warning(f"فرمت exit_time نامعتبر برای {trade_id}. استفاده از None.")
                    exit_time = None

            # 3. دریافت و اعتبارسنجی داده‌های مربوط به جهت و وضعیت معامله
            direction = row.get('direction')
            if not direction or direction not in ['long', 'short']:
                logger.warning(f"جهت معامله {trade_id} نامعتبر است. استفاده از 'long' به عنوان پیش‌فرض.")
                direction = 'long'

            status = row.get('status')
            if not status or status not in ['open', 'partially_closed', 'closed']:
                logger.warning(f"وضعیت معامله {trade_id} نامعتبر است. استفاده از 'open' به عنوان پیش‌فرض.")
                status = 'open'

            # 4. استخراج و اعتبارسنجی مقادیر عددی
            # پردازش قیمت ورود - ضروری برای هر معامله
            entry_price_str = row.get('entry_price')
            entry_price = 0.0
            if entry_price_str:
                try:
                    entry_price = float(entry_price_str)
                    if entry_price <= 0:
                        logger.warning(
                            f"قیمت ورود نامعتبر ({entry_price}) برای {trade_id}. استفاده از قیمت فعلی اگر موجود است.")
                        entry_price = float(row.get('current_price', 0)) or 100.0  # مقدار پیش‌فرض معنی‌دار
                except (ValueError, TypeError):
                    logger.warning(f"قیمت ورود غیرقابل تبدیل برای {trade_id}. استفاده از 100.0 به عنوان پیش‌فرض.")
                    entry_price = 100.0  # مقدار پیش‌فرض معنی‌دار
            else:
                logger.warning(f"قیمت ورود برای {trade_id} یافت نشد. استفاده از 100.0 به عنوان پیش‌فرض.")
                entry_price = 100.0  # مقدار پیش‌فرض معنی‌دار اگر هیچ داده‌ای موجود نیست

            # پردازش و اعتبارسنجی حد ضرر
            stop_loss = 0.0
            stop_loss_str = row.get('stop_loss')
            if stop_loss_str:
                try:
                    stop_loss = float(stop_loss_str)
                    # بررسی معتبر بودن حد ضرر بر اساس جهت معامله
                    if direction == 'long' and stop_loss >= entry_price:
                        logger.warning(
                            f"حد ضرر ({stop_loss}) بزرگتر از قیمت ورود برای معامله خرید {trade_id}. اصلاح...")
                        stop_loss = entry_price * 0.95  # 5% پایین‌تر
                    elif direction == 'short' and stop_loss <= entry_price:
                        logger.warning(
                            f"حد ضرر ({stop_loss}) کوچکتر از قیمت ورود برای معامله فروش {trade_id}. اصلاح...")
                        stop_loss = entry_price * 1.05  # 5% بالاتر
                except (ValueError, TypeError):
                    # تنظیم حد ضرر منطقی بر اساس جهت معامله
                    if direction == 'long':
                        stop_loss = entry_price * 0.95  # 5% پایین‌تر
                    else:  # short
                        stop_loss = entry_price * 1.05  # 5% بالاتر
                    logger.warning(f"حد ضرر غیرقابل تبدیل برای {trade_id}. استفاده از {stop_loss} به عنوان پیش‌فرض.")
            else:
                # تنظیم حد ضرر منطقی بر اساس جهت معامله
                if direction == 'long':
                    stop_loss = entry_price * 0.95  # 5% پایین‌تر
                else:  # short
                    stop_loss = entry_price * 1.05  # 5% بالاتر
                logger.warning(f"حد ضرر برای {trade_id} یافت نشد. استفاده از {stop_loss} به عنوان پیش‌فرض.")

            # پردازش و اعتبارسنجی حد سود
            take_profit = 0.0
            take_profit_str = row.get('take_profit')
            if take_profit_str:
                try:
                    take_profit = float(take_profit_str)
                    # بررسی معتبر بودن حد سود بر اساس جهت معامله
                    if direction == 'long' and take_profit <= entry_price:
                        logger.warning(
                            f"حد سود ({take_profit}) کوچکتر از قیمت ورود برای معامله خرید {trade_id}. اصلاح...")
                        take_profit = entry_price * 1.10  # 10% بالاتر
                    elif direction == 'short' and take_profit >= entry_price:
                        logger.warning(
                            f"حد سود ({take_profit}) بزرگتر از قیمت ورود برای معامله فروش {trade_id}. اصلاح...")
                        take_profit = entry_price * 0.90  # 10% پایین‌تر
                except (ValueError, TypeError):
                    # تنظیم حد سود منطقی بر اساس جهت معامله
                    if direction == 'long':
                        take_profit = entry_price * 1.10  # 10% بالاتر
                    else:  # short
                        take_profit = entry_price * 0.90  # 10% پایین‌تر
                    logger.warning(f"حد سود غیرقابل تبدیل برای {trade_id}. استفاده از {take_profit} به عنوان پیش‌فرض.")
            else:
                # تنظیم حد سود منطقی بر اساس جهت معامله
                if direction == 'long':
                    take_profit = entry_price * 1.10  # 10% بالاتر
                else:  # short
                    take_profit = entry_price * 0.90  # 10% پایین‌تر
                logger.warning(f"حد سود برای {trade_id} یافت نشد. استفاده از {take_profit} به عنوان پیش‌فرض.")

            # پردازش اندازه معامله و باقیمانده
            quantity = 0.0
            quantity_str = row.get('quantity')
            if quantity_str:
                try:
                    quantity = float(quantity_str)
                    if quantity <= 0:
                        logger.warning(
                            f"اندازه معامله نامعتبر ({quantity}) برای {trade_id}. استفاده از 1.0 به عنوان پیش‌فرض.")
                        quantity = 1.0
                except (ValueError, TypeError):
                    logger.warning(f"اندازه معامله غیرقابل تبدیل برای {trade_id}. استفاده از 1.0 به عنوان پیش‌فرض.")
                    quantity = 1.0
            else:
                logger.warning(f"اندازه معامله برای {trade_id} یافت نشد. استفاده از 1.0 به عنوان پیش‌فرض.")
                quantity = 1.0

            # مقدار باقیمانده (با توجه به وضعیت معامله)
            remaining_quantity = quantity
            remaining_quantity_str = row.get('remaining_quantity')
            if remaining_quantity_str:
                try:
                    remaining_quantity = float(remaining_quantity_str)
                    if remaining_quantity < 0 or remaining_quantity > quantity:
                        logger.warning(f"مقدار باقیمانده نامعتبر ({remaining_quantity}) برای {trade_id}. اصلاح...")
                        if status == 'closed':
                            remaining_quantity = 0.0
                        elif status == 'partially_closed':
                            remaining_quantity = quantity * 0.5  # فرض: نیمی باقی مانده
                        else:  # open
                            remaining_quantity = quantity
                except (ValueError, TypeError):
                    if status == 'closed':
                        remaining_quantity = 0.0
                    elif status == 'partially_closed':
                        remaining_quantity = quantity * 0.5  # فرض: نیمی باقی مانده
                    else:  # open
                        remaining_quantity = quantity
                    logger.warning(
                        f"مقدار باقیمانده غیرقابل تبدیل برای {trade_id}. استفاده از {remaining_quantity} به عنوان پیش‌فرض.")
            else:
                # تنظیم مقدار باقیمانده بر اساس وضعیت معامله
                if status == 'closed':
                    remaining_quantity = 0.0
                elif status == 'partially_closed':
                    remaining_quantity = quantity * 0.5  # فرض: نیمی باقی مانده
                else:  # open
                    remaining_quantity = quantity
                logger.warning(
                    f"مقدار باقیمانده برای {trade_id} یافت نشد. استفاده از {remaining_quantity} به عنوان پیش‌فرض.")

            # قیمت فعلی - مهم برای معاملات باز
            current_price = 0.0
            current_price_str = row.get('current_price')
            if current_price_str:
                try:
                    current_price = float(current_price_str)
                    if current_price <= 0:
                        logger.warning(f"قیمت فعلی نامعتبر ({current_price}) برای {trade_id}. استفاده از قیمت ورود.")
                        current_price = entry_price
                except (ValueError, TypeError):
                    current_price = entry_price
                    logger.warning(f"قیمت فعلی غیرقابل تبدیل برای {trade_id}. استفاده از قیمت ورود.")
            else:
                current_price = entry_price
                logger.warning(f"قیمت فعلی برای {trade_id} یافت نشد. استفاده از قیمت ورود.")

            # سود/زیان - می‌تواند صفر باشد
            profit_loss = 0.0
            profit_loss_str = row.get('profit_loss')
            if profit_loss_str:
                try:
                    profit_loss = float(profit_loss_str)
                except (ValueError, TypeError):
                    logger.warning(f"سود/زیان غیرقابل تبدیل برای {trade_id}. استفاده از 0.0.")

            # کارمزد پرداخت شده - می‌تواند صفر باشد
            commission_paid = 0.0
            commission_paid_str = row.get('commission_paid')
            if commission_paid_str:
                try:
                    commission_paid = float(commission_paid_str)
                    if commission_paid < 0:
                        logger.warning(f"کارمزد منفی ({commission_paid}) برای {trade_id}. استفاده از 0.0.")
                        commission_paid = 0.0
                except (ValueError, TypeError):
                    logger.warning(f"کارمزد غیرقابل تبدیل برای {trade_id}. استفاده از 0.0.")

            # 5. استخراج فیلدهای متنی با مقادیر پیش‌فرض
            # تگ‌ها (با لیست پیش‌فرض مناسب)
            tags = []
            tags_str = row.get('tags')
            if tags_str:
                # تقسیم رشته تگ‌ها و حذف فضاهای خالی اضافی
                tags = [tag.strip() for tag in tags_str.split(',') if tag.strip()]

            # اگر تگ‌ها خالی است، تگ‌های پیش‌فرض بسازیم
            if not tags:
                # اضافه کردن جهت و نماد به عنوان تگ‌های پیش‌فرض
                tags = [direction]
                # اضافه کردن نماد اصلی به تگ‌ها
                if '/' in symbol:
                    base_asset = symbol.split('/')[0]
                    tags.append(base_asset)

            # استراتژی و تایم‌فریم با مقادیر پیش‌فرض منطقی
            strategy_name = row.get('strategy_name')
            if not strategy_name:
                timeframe = row.get('timeframe') or '5m'
                # ساخت نام استراتژی بر اساس جهت و تایم‌فریم
                strategy_name = f"{direction.capitalize()}_{timeframe}"
                logger.warning(f"نام استراتژی برای {trade_id} یافت نشد. استفاده از '{strategy_name}' به عنوان پیش‌فرض.")

            # تایم‌فریم با مقدار پیش‌فرض
            timeframe = row.get('timeframe')
            if not timeframe:
                # استخراج تایم‌فریم از استراتژی اگر ممکن باشد
                if '_' in strategy_name:
                    possible_tf = strategy_name.split('_')[-1]
                    if possible_tf in ['1m', '3m', '5m', '15m', '30m', '1h', '2h', '4h', '6h', '8h', '12h', '1d', '3d',
                                       '1w']:
                        timeframe = possible_tf
                    else:
                        timeframe = '5m'  # مقدار پیش‌فرض
                else:
                    timeframe = '5m'  # مقدار پیش‌فرض
                logger.warning(f"تایم‌فریم برای {trade_id} یافت نشد. استفاده از '{timeframe}' به عنوان پیش‌فرض.")

            # وضعیت بازار با مقدار پیش‌فرض
            market_state = row.get('market_state')
            if not market_state:
                # می‌توان بر اساس جهت معامله حدس زد
                if direction == 'long':
                    market_state = 'bullish'
                else:
                    market_state = 'bearish'
                logger.warning(f"وضعیت بازار برای {trade_id} یافت نشد. استفاده از '{market_state}' به عنوان پیش‌فرض.")

            # یادداشت‌ها با زمان ایجاد پیش‌فرض
            notes = row.get('notes')
            if not notes:
                time_str = timestamp.strftime('%Y-%m-%d %H:%M:%S')
                notes = f"بازسازی شده از دیتابیس در {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}. ایجاد شده در {time_str}."

            # دلایل ورود (می‌تواند خالی باشد)
            entry_reasons_json = row.get('entry_reasons_json')
            if not entry_reasons_json:
                # ایجاد یک دلیل ورود پیش‌فرض ساده
                default_reason = {
                    "info": "دلایل ورود اصلی در دسترس نیست",
                    "reconstructed": True,
                    "timestamp": datetime.now().isoformat()
                }
                entry_reasons_json = json.dumps(default_reason, ensure_ascii=False)

            # 6. محاسبه نسبت ریسک به ریوارد
            risk_reward_ratio = 0.0
            if entry_price > 0 and stop_loss > 0 and take_profit > 0:
                if direction == 'long':
                    risk = entry_price - stop_loss
                    reward = take_profit - entry_price
                else:  # short
                    risk = stop_loss - entry_price
                    reward = entry_price - take_profit

                if risk > 0:
                    risk_reward_ratio = reward / risk
                else:
                    # مقدار پیش‌فرض منطقی
                    risk_reward_ratio = 2.0
            else:
                # مقدار پیش‌فرض منطقی
                risk_reward_ratio = 2.0

            # 7. ایجاد شیء Trade با تمام فیلدهای استخراج شده
            trade = Trade(
                trade_id=trade_id,
                symbol=symbol,
                direction=direction,
                entry_price=entry_price,
                stop_loss=stop_loss,
                take_profit=take_profit,
                quantity=quantity,
                remaining_quantity=remaining_quantity,
                timestamp=timestamp,
                status=status,
                current_price=current_price,
                exit_time=exit_time,
                exit_reason=row.get('exit_reason', ''),
                profit_loss=profit_loss,
                commission_paid=commission_paid,
                tags=tags,
                strategy_name=strategy_name,
                timeframe=timeframe,
                market_state=market_state,
                notes=notes,
                entry_reasons_json=entry_reasons_json,
                risk_reward_ratio=risk_reward_ratio
            )

            # 8. تنظیم فیلدهای اضافی مهم
            trade.initial_stop_loss = stop_loss
            trade.closed_portions = []  # آغاز با لیست خالی از بخش‌های بسته شده

            # تنظیم مقدار ریسک (برای محاسبات مدیریت ریسک)
            trade.risk_amount = abs(entry_price - stop_loss) * quantity

            # 9. تنظیم کیفیت سیگنال اگر داده خامی برای آن وجود ندارد
            if not hasattr(trade, 'signal_quality') or trade.signal_quality is None:
                # برآورد کیفیت سیگنال بر اساس RR
                if risk_reward_ratio >= 3.0:
                    trade.signal_quality = 80.0  # کیفیت بالا
                elif risk_reward_ratio >= 2.0:
                    trade.signal_quality = 65.0  # کیفیت خوب
                elif risk_reward_ratio >= 1.5:
                    trade.signal_quality = 50.0  # کیفیت متوسط
                else:
                    trade.signal_quality = 40.0  # کیفیت پایین

            logger.debug(
                f"معامله {trade_id} با استراتژی '{strategy_name}' و تایم‌فریم '{timeframe}' با موفقیت بازسازی شد.")
            return trade

        except Exception as e:
            logger.error(f"خطای غیرمنتظره در بازسازی معامله از سطر دیتابیس: {e}", exc_info=True)
            return None

    def _initialize_extensions(self):
        """راه‌اندازی کامپوننت‌های ماژول trade_extensions اگر موجود باشند."""
        try:
            # ایجاد نمونه‌های کلاس‌های extension
            self.correlation_manager = CorrelationManager(self.config)
            self.position_size_optimizer = PositionSizeOptimizer(self.config)
            self.trade_analyzer = TradeAnalyzer(self.config, self.db_path)
            self.dynamic_stop_manager = DynamicStopManager(self.config)
            self.risk_calculator = RiskCalculator(self.config)

            logger.info("Trade extensions initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize trade extensions: {e}")

    def _ensure_db_directory(self):
        """اطمینان از وجود دایرکتوری برای فایل دیتابیس."""
        try:
            db_dir = os.path.dirname(self.db_path)
            if db_dir and not os.path.exists(db_dir):
                os.makedirs(db_dir)
                logger.info(f"Created database directory: {db_dir}")
        except OSError as e:
            logger.error(f"Failed to create database directory '{db_dir}': {e}")
            raise  # ایجاد خطا برای جلوگیری از ادامه

    def _reset_stats(self) -> Dict[str, Any]:
        """ریست کردن دیکشنری آمار."""
        return {
            'total_trades_opened': 0,  # تعداد کل معاملات باز شده
            'open_trades': 0,  # تعداد معاملات در حال حاضر باز
            'partially_closed_trades': 0,  # تعداد معاملات بخشی بسته شده
            'closed_trades': 0,  # تعداد معاملات کاملا بسته شده
            'winning_trades': 0,  # تعداد معاملات بسته شده با سود
            'losing_trades': 0,  # تعداد معاملات بسته شده با ضرر (یا سر به سر)
            'total_net_profit_loss': 0.0,  # سود/ضرر خالص کل معاملات بسته شده
            'total_commission_paid': 0.0,  # کل کارمزد پرداخت شده
            'win_rate': 0.0,
            'profit_factor': 0.0,
            'current_equity': self.initial_balance,
            'current_open_pnl': 0.0,
            'current_drawdown': 0.0,
            'realized_pnl_partial_trades': 0.0,
            # آمار اضافی
            'avg_win_percent': 0.0,
            'avg_loss_percent': 0.0,
            'largest_win': 0.0,
            'largest_loss': 0.0,
            'avg_trade_duration_hours': 0.0,
            'daily_pnl': {},
            'trades_by_tag': {},
            'trades_by_strategy': {},
            # آمار پیشرفته
            'sharpe_ratio': 0.0,
            'sortino_ratio': 0.0,
            'max_drawdown_percent': 0.0,
            'recovery_factor': 0.0,
            'profit_consistency': 0.0,
            'avg_risk_reward_ratio': 0.0
        }

    def save_trade_to_db(self, trade: Trade) -> bool:
        """
        ذخیره یا به‌روزرسانی معامله در دیتابیس با اطمینان از ذخیره تمام فیلدها.
        """
        if not self.cursor or not self.conn:
            logger.error(f"امکان ذخیره معامله {trade.trade_id} وجود ندارد: اتصال به دیتابیس برقرار نیست.")
            return False

        # استفاده از RLock برای مدیریت دسترسی همزمان به دیتابیس
        with self._db_lock:
            for retry in range(MAX_DB_RETRIES):
                try:
                    # اطمینان از وجود داده‌های حیاتی قبل از سریال‌سازی
                    if not all([trade.trade_id, trade.symbol, hasattr(trade, 'timestamp'), trade.status]):
                        logger.error(f"داده‌های ضروری برای معامله {trade.trade_id} موجود نیست، امکان ذخیره وجود ندارد.")
                        return False

                    # --- محاسبه PNL برای ذخیره بر اساس وضعیت معامله ---
                    pnl_to_save = None
                    if trade.status in ['open', 'partially_closed']:
                        # برای معاملات باز یا بخشی بسته شده، سود/زیان خالص فعلی (تحقق یافته + شناور) را ذخیره کن
                        pnl_to_save = self._sanitize_float(trade.get_net_pnl())
                        logger.debug(f"محاسبه PNL برای معامله باز/بخشی بسته شده {trade.trade_id}: {pnl_to_save}")
                    elif trade.status == 'closed':
                        # برای معاملات بسته شده، از PNL نهایی محاسبه شده استفاده کن
                        pnl_to_save = self._sanitize_float(trade.profit_loss)
                        logger.debug(f"استفاده از PNL نهایی برای معامله بسته شده {trade.trade_id}: {pnl_to_save}")
                    else:
                        # وضعیت نامشخص یا fallback
                        pnl_to_save = self._sanitize_float(trade.profit_loss)
                        logger.warning(
                            f"معامله {trade.trade_id} وضعیت نامشخص '{trade.status}' دارد. استفاده از PNL ذخیره شده قبلی: {pnl_to_save}")

                    # --- ایمن‌سازی سایر مقادیر عددی و آماده‌سازی داده‌ها ---
                    # تبدیل کل شی به دیکشنری (شامل مقادیر float ایمن‌شده توسط to_dict)
                    trade_dict = trade.to_dict()

                    # دریافت سایر مقادیر مورد نیاز برای ستون‌های خاص
                    current_price_to_save = trade_dict.get('current_price')
                    entry_reasons_to_save = trade_dict.get('entry_reasons_json')
                    data_json_full = json.dumps(trade_dict, default=str, ensure_ascii=False)

                    # اطمینان از وجود تگ‌ها و تبدیل آنها به رشته
                    tags_list = trade.tags if hasattr(trade, 'tags') and trade.tags else []
                    tags_str = ",".join(tags_list)

                    # تبدیل timestamp ها به رشته ISO با استفاده از helper
                    timestamp_iso = self._dt_to_iso(trade.timestamp)
                    exit_time_iso = self._dt_to_iso(trade.exit_time)

                    # اطمینان از وجود فیلدهای مهم و مقداردهی آنها
                    strategy_name = trade.strategy_name if hasattr(trade,
                                                                   'strategy_name') and trade.strategy_name else "unknown_strategy"
                    timeframe = trade.timeframe if hasattr(trade, 'timeframe') and trade.timeframe else "5m"
                    market_state = trade.market_state if hasattr(trade,
                                                                 'market_state') and trade.market_state else "neutral"
                    notes = trade.notes if hasattr(trade, 'notes') and trade.notes else ""

                    # ایمن‌سازی سایر مقادیر float که مستقیماً در ستون‌ها ذخیره می‌شوند
                    entry_price_to_save = self._sanitize_float(trade.entry_price)
                    stop_loss_to_save = self._sanitize_float(trade.stop_loss)
                    take_profit_to_save = self._sanitize_float(trade.take_profit)
                    quantity_to_save = self._sanitize_float(trade.quantity)
                    remaining_quantity_to_save = self._sanitize_float(trade.remaining_quantity)
                    commission_paid_to_save = self._sanitize_float(trade.commission_paid)

                    # --- مقادیر نهایی برای کوئری SQL ---
                    values = (
                        trade.trade_id, trade.symbol, timestamp_iso, trade.status,
                        trade.direction, entry_price_to_save, stop_loss_to_save, take_profit_to_save,
                        quantity_to_save, remaining_quantity_to_save,
                        current_price_to_save,  # ستون current_price جدول
                        exit_time_iso,
                        trade.exit_reason,
                        pnl_to_save,  # استفاده از PNL محاسبه/ایمن‌شده برای ستون profit_loss
                        commission_paid_to_save,
                        tags_str,
                        strategy_name,
                        timeframe,
                        market_state,
                        notes,
                        entry_reasons_to_save,  # ستون دلایل ورود
                        data_json_full  # ستون داده کامل معامله
                    )

                    # کوئری SQL (بدون تغییر، همچنان 22 placeholder)
                    query = '''
                    INSERT OR REPLACE INTO trades (
                        trade_id, symbol, timestamp, status, direction, entry_price, stop_loss,
                        take_profit, quantity, remaining_quantity, current_price,
                        exit_time, exit_reason, profit_loss, commission_paid,
                        tags, strategy_name, timeframe, market_state, notes,
                        entry_reasons_json,
                        data
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    '''

                    # اجرای کوئری
                    self.cursor.execute(query, values)
                    self.conn.commit()

                    # لاگ دقیق‌تر برای معاملات باز
                    if trade.status in ['open', 'partially_closed']:
                        logger.debug(f"وضعیت به‌روز شده برای معامله باز/بخشی بسته شده {trade.trade_id} ذخیره شد. "
                                     f"قیمت فعلی: {current_price_to_save}, PNL فعلی: {pnl_to_save}, "
                                     f"استراتژی: {strategy_name}, تایم‌فریم: {timeframe}")
                    else:
                        logger.debug(
                            f"وضعیت معامله {trade.trade_id} در دیتابیس ذخیره/به‌روز شد. وضعیت: {trade.status}, "
                            f"استراتژی: {strategy_name}, تایم‌فریم: {timeframe}")

                    return True  # عملیات موفق بود

                except sqlite3.Error as e:
                    logger.error(
                        f"خطای دیتابیس در ذخیره معامله {trade.trade_id} (تلاش {retry + 1}/{MAX_DB_RETRIES}): {e}")
                    if "database is locked" in str(e).lower() and retry < MAX_DB_RETRIES - 1:
                        # اگر دیتابیس قفل بود، کمی صبر کن و دوباره تلاش کن
                        time.sleep(random.uniform(0.1, 0.5))
                    elif retry < MAX_DB_RETRIES - 1:
                        time.sleep(0.5)  # تاخیر استاندارد برای سایر خطاهای دیتابیس
                    else:
                        logger.critical(
                            f"ذخیره معامله {trade.trade_id} پس از {MAX_DB_RETRIES} تلاش به دلیل خطای دیتابیس ناموفق بود.")
                        return False  # همه تلاش‌ها ناموفق بود
                except Exception as e:
                    logger.critical(f"خطای غیرمنتظره در ذخیره معامله {trade.trade_id} در دیتابیس: {e}", exc_info=True)
                    # تلاش برای لاگ بخشی از داده‌های مشکل‌ساز
                    try:
                        partial_data_str = json.dumps(
                            {'id': trade.trade_id, 'symbol': trade.symbol, 'status': trade.status}, default=str)
                        logger.error(f"داده معامله مشکل‌دار (بخشی): {partial_data_str}")
                    except:
                        logger.error("حتی امکان سریال‌سازی بخشی از داده معامله مشکل‌دار وجود ندارد.")
                    return False  # خروج با خطا در صورت بروز خطای غیرمنتظره

            # اگر حلقه تلاش‌ها تمام شد و موفق نبود
            return False

    # --- توابع کمکی ---
    def _dt_to_iso(self, dt_obj):
        """
        تبدیل ایمن datetime به رشته ISO یا بازگرداندن مقدار اصلی.

        Args:
            dt_obj: شیء datetime یا هر مقدار دیگر

        Returns:
            رشته ISO یا مقدار اصلی در صورت نامعتبر بودن
        """
        if isinstance(dt_obj, datetime):
            try:
                # اطمینان از timezone-aware بودن قبل از isoformat
                if dt_obj.tzinfo is None:
                    # اضافه کردن timezone برای اطمینان از سازگاری
                    try:
                        import pytz
                        dt_obj = pytz.utc.localize(dt_obj)
                    except ImportError:
                        # اگر pytz نصب نیست، از timezone محلی استفاده کنیم
                        dt_obj = dt_obj.astimezone()
                return dt_obj.isoformat()
            except Exception as e:
                logger.warning(f"خطا در تبدیل datetime {dt_obj} به ISO: {e}")
                return str(dt_obj)  # بازگرداندن رشته به عنوان fallback
        return dt_obj  # بازگرداندن None یا مقادیر دیگر

    def _sanitize_float(self, value: Optional[Union[float, int, str]]) -> Optional[float]:
        """
        تبدیل مقدار به float معتبر یا None، با مدیریت NaN و Inf.

        Args:
            value: مقدار ورودی که می‌تواند float، int، string یا None باشد

        Returns:
            مقدار float معتبر یا None در صورت نامعتبر بودن
        """
        if value is None:
            return None
        try:
            f_val = float(value)
            # بررسی مقادیر نامعتبر float
            if math.isnan(f_val) or math.isinf(f_val):
                logger.debug(f"مقدار float نامعتبر ({value}) به None تبدیل شد.")
                return None
            return f_val
        except (ValueError, TypeError):
            logger.debug(f"مقدار '{value}' قابل تبدیل به float نیست، None برگردانده می‌شود.")
            return None

    def _evaluate_signal_quality(self, signal: SignalInfo) -> float:
        """
        ارزیابی کیفیت سیگنال بر اساس فاکتورهای مختلف.

        Args:
            signal: اطلاعات سیگنال

        Returns:
            امتیاز کیفیت از 0 تا 1
        """
        # 1. بررسی اینکه آیا امتیاز در سیگنال وجود دارد
        if hasattr(signal, 'score') and hasattr(signal.score, 'final_score'):
            # نرمال‌سازی به بازه 0 تا 1
            return min(1.0, max(0.0, signal.score.final_score / 100.0))

        # 2. در صورت عدم وجود امتیاز، برآورد بر اساس نسبت ریسک به ریوارد
        if hasattr(signal, 'risk_reward_ratio') and signal.risk_reward_ratio:
            # نسبت ریسک به ریوارد بالاتر = کیفیت بهتر
            # معمولاً RR بین 1 تا 5 است
            # فرمول: min(1.0, RR / 5.0)
            rr_quality = min(1.0, signal.risk_reward_ratio / 5.0)
            return rr_quality

        # 3. در صورت عدم وجود، بر اساس فاصله استاپ از ورود
        if hasattr(signal, 'entry_price') and hasattr(signal, 'stop_loss') and signal.entry_price > 0:
            # محاسبه درصد فاصله استاپ
            stop_distance_percent = abs(signal.entry_price - signal.stop_loss) / signal.entry_price * 100

            # فاصله مناسب استاپ معمولاً بین 0.5% تا 3% است
            if stop_distance_percent < 0.5:
                # فاصله خیلی کم = کیفیت پایین
                return 0.3
            elif stop_distance_percent > 3.0:
                # فاصله خیلی زیاد = کیفیت متوسط
                return 0.5
            else:
                # فاصله مناسب = کیفیت خوب
                return 0.7

        # 4. در صورت عدم وجود هیچ‌کدام از موارد بالا، مقدار متوسط برگردان
        return 0.5

    def _generate_meaningful_tags(self, signal: SignalInfo) -> List[str]:
        """
        تولید تگ‌های معنادار برای معامله بر اساس ویژگی‌های سیگنال.

        Args:
            signal: اطلاعات سیگنال

        Returns:
            لیست تگ‌های معنادار
        """
        tags = []

        # اضافه کردن جهت معامله
        if hasattr(signal, 'direction') and signal.direction:
            tags.append(signal.direction)

        # اضافه کردن تایم‌فریم
        if hasattr(signal, 'timeframe') and signal.timeframe:
            tags.append(signal.timeframe)

        # اضافه کردن نوع سیگنال
        if hasattr(signal, 'signal_type') and signal.signal_type:
            # بررسی اینکه آیا تگ قبلاً اضافه نشده است
            if signal.signal_type not in tags:
                tags.append(signal.signal_type)

        # اضافه کردن نام استراتژی اگر متفاوت از نوع سیگنال است
        if hasattr(signal, 'strategy_name') and signal.strategy_name:
            if signal.strategy_name not in tags:
                tags.append(signal.strategy_name)

        # اضافه کردن وضعیت بازار
        if hasattr(signal, 'market_condition') and signal.market_condition:
            tags.append(signal.market_condition)
        elif hasattr(signal, 'market_state') and signal.market_state:
            tags.append(signal.market_state)

        # اضافه کردن تگ کیفیت سیگنال
        if hasattr(signal, 'score') and hasattr(signal.score, 'final_score'):
            score = signal.score.final_score
            if score >= 80:
                tags.append('high_quality')
            elif score >= 60:
                tags.append('good_quality')
            elif score <= 40:
                tags.append('low_quality')

        # اضافه کردن تگ نوع الگو (اگر وجود دارد)
        if hasattr(signal, 'pattern_type') and signal.pattern_type:
            tags.append(signal.pattern_type)

        # اطمینان از اینکه حداقل دو تگ وجود دارد
        if len(tags) < 2:
            # اضافه کردن تگ‌های پیش‌فرض
            symbols_parts = signal.symbol.split('/')
            if len(symbols_parts) > 0:
                base_asset = symbols_parts[0]
                tags.append(base_asset)

        return tags

    def save_stop_loss_change(self, trade_id: str, old_value: float, new_value: float, reason: str = "trailing_stop"):
        """ثبت تغییر استاپ لاس در تاریخچه."""
        if not self.cursor or not self.conn:
            return False

        with self._db_lock:
            try:
                timestamp = datetime.now().astimezone().isoformat()

                self.cursor.execute(
                    "INSERT INTO trade_level_history (trade_id, timestamp, level_type, old_value, new_value, reason) VALUES (?, ?, ?, ?, ?, ?)",
                    (trade_id, timestamp, "stop_loss", old_value, new_value, reason)
                )

                self.conn.commit()
                return True
            except sqlite3.Error as e:
                logger.error(f"Error saving stop loss change: {e}")
                return False

    def register_price_fetcher(self, callback_function: Callable[[str], Awaitable[Optional[float]]],
                               data_fetcher_instance: Any):
        """ثبت تابع دریافت قیمت و وهله DataFetcher."""
        self.price_fetcher_callback = callback_function
        # نیاز به وهله DataFetcher برای استفاده در check_correlation
        self.data_fetcher_instance = data_fetcher_instance
        logger.info("Price fetcher callback and DataFetcher instance registered.")

    def register_notification_callback(self, callback_function: Callable[[str], Awaitable[None]]):
        """ثبت تابع اطلاع‌رسانی."""
        self.notification_callback = callback_function
        logger.info("Notification callback registered.")

    def calculate_position_size(self,
                                signal: SignalInfo,
                                stop_distance: float,
                                adapted_risk_config: Dict[str, Any]
                                ) -> Union[float, Dict[str, Any]]:
        """
        محاسبه اندازه پوزیشن با پشتیبانی صحیح از fixed_position_size به عنوان ارزش USDT.

        Args:
            signal: اطلاعات سیگنال معاملاتی
            stop_distance: فاصله بین قیمت ورود و حد ضرر
            adapted_risk_config: پیکربندی مدیریت ریسک تطبیق یافته

        Returns:
            اندازه پوزیشن یا دیکشنری شامل اطلاعات پوزیشن
        """
        # --- دریافت تنظیمات لازم از کانفیگ تطبیق‌یافته ---
        # پیش‌فرض False برای محاسبه پویا اگر مشخص نشده باشد
        use_dynamic = adapted_risk_config.get('use_dynamic_position_sizing', False)
        # خواندن مقدار USDT ثابت و اعتبارسنجی
        fixed_value_usdt = self._sanitize_float(adapted_risk_config.get('fixed_position_size', 200))
        if fixed_value_usdt is None or fixed_value_usdt <= 0:
            fixed_value_usdt = 200  # مقدار پیش‌فرض اگر نامعتبر باشد

        # خواندن حداکثر مقدار اندازه پوزیشن (USDT) و اعتبارسنجی
        max_pos_value_usdt = self._sanitize_float(adapted_risk_config.get('max_position_size'))

        # اطمینان از معتبر بودن قیمت ورود
        entry_price = self._sanitize_float(signal.entry_price)
        if not entry_price or entry_price <= 0:
            logger.error(
                f"قیمت ورود نامعتبر ({signal.entry_price}) برای {signal.symbol}. امکان محاسبه اندازه پوزیشن وجود ندارد.")
            return 0.0

        # اطمینان از معتبر بودن فاصله استاپ
        if stop_distance <= 1e-9:
            logger.warning(
                f"فاصله استاپ صفر یا نامعتبر ({stop_distance:.8f}) برای {signal.symbol}. امکان محاسبه اندازه وجود ندارد.")
            return 0.0

        # --- >> حالت استفاده از محاسبه پویا << ---
        if use_dynamic:
            try:
                # فرض: _update_stats() و _calculate_risk_adjustment_factors() وجود دارند
                self._update_stats()
                current_equity = self._sanitize_float(self.stats.get('current_equity', self.initial_balance))

                if current_equity is None or current_equity <= 0:
                    logger.error("امکان محاسبه اندازه پوزیشن پویا وجود ندارد: موجودی فعلی صفر، منفی یا نامعتبر است.")
                    return 0.0

                base_risk_percent = self._sanitize_float(adapted_risk_config.get('max_risk_per_trade_percent', 1.5))
                if base_risk_percent is None:
                    base_risk_percent = 1.5  # مقدار پیش‌فرض

                risk_factors = self._calculate_risk_adjustment_factors(signal)
                final_risk_percent = base_risk_percent * risk_factors.get('combined_factor', 1.0)

                # محدود کردن درصد ریسک
                max_risk_increase = adapted_risk_config.get('max_risk_increase', 1.5)
                max_risk_decrease = adapted_risk_config.get('max_risk_decrease', 0.5)
                final_risk_percent = max(base_risk_percent * max_risk_decrease,
                                         min(final_risk_percent, base_risk_percent * max_risk_increase))

                risk_amount = current_equity * (final_risk_percent / 100.0)
                position_size_raw = risk_amount / stop_distance

                # اعمال محدودیت حداکثر ارزش پوزیشن (اگر تعریف شده باشد)
                if max_pos_value_usdt is not None and max_pos_value_usdt > 0:
                    calculated_value = position_size_raw * entry_price
                    if calculated_value > max_pos_value_usdt:
                        logger.info(
                            f"مقدار محاسبه شده پوزیشن ({calculated_value:.2f} USDT) با max_position_size محدود شد ({max_pos_value_usdt} USDT)")
                        position_size_raw = max_pos_value_usdt / entry_price
                        risk_amount = position_size_raw * stop_distance  # محاسبه مجدد ریسک واقعی
                        final_risk_percent = (risk_amount / current_equity) * 100 if current_equity > 0 else 0.0

                # گرد کردن و بررسی حداقل حجم
                precision = self.get_symbol_precision(signal.symbol)
                final_quantity = round(position_size_raw, precision)
                min_trade_size = 1 / (10 ** precision) if precision > 0 else 1.0

                if final_quantity < min_trade_size - 1e-9:  # استفاده از خطای مجاز
                    logger.warning(
                        f"مقدار محاسبه شده پویا ({final_quantity:.{precision}f}) کمتر از حداقل ({min_trade_size:.{precision}f}) است. برگرداندن صفر.")
                    return 0.0

                logger.info(
                    f"مقدار نهایی (پویا): {final_quantity:.{precision}f} {signal.symbol.split('/')[0]} "
                    f"(درصد ریسک: {final_risk_percent:.3f}, مقدار ریسک: {risk_amount:.2f} USDT)"
                )

                # برگرداندن دیکشنری با اطلاعات اضافی
                return {
                    'position_size': final_quantity,
                    'risk_amount': risk_amount,
                    'risk_percent': final_risk_percent,
                    'calculation_type': 'dynamic',
                    'precision': precision
                }

            except Exception as e:
                logger.error(f"خطا در محاسبه پویای اندازه پوزیشن برای {signal.symbol}: {e}", exc_info=True)
                return 0.0

        # --- >> حالت استفاده از اندازه ثابت USDT << ---
        else:
            if fixed_value_usdt is not None and fixed_value_usdt > 0:
                try:
                    # محاسبه مقدار ارز بر اساس ارزش USDT و قیمت ورود
                    position_size_raw = fixed_value_usdt / entry_price

                    # اعمال محدودیت حداکثر ارزش پوزیشن
                    fixed_value_to_use = fixed_value_usdt
                    if max_pos_value_usdt is not None and fixed_value_usdt > max_pos_value_usdt:
                        logger.warning(
                            f"مقدار ثابت USDT ({fixed_value_usdt}) بیشتر از max_position_size ({max_pos_value_usdt}) است. استفاده از max_position_size.")
                        position_size_raw = max_pos_value_usdt / entry_price
                        fixed_value_to_use = max_pos_value_usdt  # ثبت مقدار واقعی استفاده شده

                    # گرد کردن و بررسی حداقل حجم
                    precision = self.get_symbol_precision(signal.symbol)
                    final_quantity = round(position_size_raw, precision)
                    min_trade_size = 1 / (10 ** precision) if precision > 0 else 1.0

                    if final_quantity < min_trade_size - 1e-9:  # استفاده از خطای مجاز
                        logger.warning(
                            f"مقدار محاسبه شده از USDT ثابت ({final_quantity:.{precision}f}) کمتر از حداقل ({min_trade_size:.{precision}f}) است. برگرداندن صفر.")
                        return 0.0

                    # محاسبه ریسک برای لاگ (اختیاری)
                    calculated_risk_amount = final_quantity * stop_distance
                    risk_percent = (
                                               calculated_risk_amount / self.initial_balance) * 100 if self.initial_balance > 0 else 0

                    logger.info(
                        f"مقدار نهایی (USDT ثابت): {final_quantity:.{precision}f} {signal.symbol.split('/')[0]} "
                        f"(براساس {fixed_value_to_use:.2f} USDT، ریسک تقریبی: {calculated_risk_amount:.2f} USDT، {risk_percent:.2f}%)"
                    )

                    # برگرداندن دیکشنری با اطلاعات اضافی
                    return {
                        'position_size': final_quantity,
                        'risk_amount': calculated_risk_amount,
                        'risk_percent': risk_percent,
                        'calculation_type': 'fixed_usdt',
                        'precision': precision
                    }

                except Exception as e:
                    logger.error(f"خطا در محاسبه اندازه پوزیشن از USDT ثابت برای {signal.symbol}: {e}", exc_info=True)
                    return 0.0
            else:
                # اگر اندازه ثابت تعریف نشده یا نامعتبر است و محاسبه پویا هم غیرفعال است
                logger.error(
                    f"خطای اندازه‌گیری پوزیشن برای {signal.symbol}: محاسبه پویا غیرفعال است و 'fixed_position_size' معتبر (USDT > 0) در پیکربندی وجود ندارد.")
                return 0.0

    def _calculate_current_drawdown(self) -> float:
        """محاسبه drawdown فعلی بر اساس اکوئیتی فعلی و پیک اکوئیتی."""
        try:
            # محاسبه اکوئیتی فعلی
            closed_pnl, open_pnl = self._calculate_total_pnl()
            current_equity = self.initial_balance + closed_pnl + open_pnl

            # ذخیره در آمار
            self.stats['current_equity'] = current_equity
            self.stats['current_open_pnl'] = open_pnl

            # به‌روزرسانی پیک اکوئیتی
            self.peak_equity = max(self.peak_equity, current_equity)

            # محاسبه drawdown
            if self.peak_equity > 0:
                drawdown = max(0, self.peak_equity - current_equity)
                drawdown_percent = (drawdown / self.peak_equity) * 100
            else:
                drawdown_percent = 0.0

            self.stats['current_drawdown'] = round(drawdown_percent, 2)
            return drawdown_percent

        except Exception as e:
            logger.error(f"Error calculating current drawdown: {e}")
            return self.stats.get('current_drawdown', 0.0)

    def _calculate_total_pnl(self) -> Tuple[float, float]:
        """محاسبه کل سود/زیان بسته شده و باز."""
        # سود/زیان بسته شده
        closed_pnl = 0.0
        for t in self.active_trades.values():
            for p in t.closed_portions:
                p_net_pnl = p.get('net_pnl', 0)
                # تبدیل به float در صورت نیاز
                if isinstance(p_net_pnl, str):
                    try:
                        p_net_pnl = float(p_net_pnl)
                    except ValueError:
                        p_net_pnl = 0.0
                closed_pnl += p_net_pnl

        # سود/زیان شناور
        open_pnl = 0.0
        for trade in self.active_trades.values():
            if trade.status != 'closed':
                open_pnl += trade.get_floating_pnl()

        return closed_pnl, open_pnl

    async def update_trade_prices(self):
        """
        به‌روزرسانی قیمت‌ها و بررسی شرایط خروج با منطق بهبود یافته و ذخیره قیمت به‌روز شده.
        """
        if not self.price_fetcher_callback:
            logger.debug("تابع callback دریافت قیمت ثبت نشده است. پرش از به‌روزرسانی قیمت.")
            return

        trades_to_update = []
        with self._trades_lock:
            # فقط معاملاتی که بسته نشده‌اند را برای آپدیت انتخاب می‌کنیم
            trades_to_update = [t for t in self.active_trades.values() if t.status != 'closed']

        if not trades_to_update:
            # اگر معامله باز فعالی وجود ندارد، خارج می‌شویم
            return

            # --- دریافت قیمت‌های فعلی ---
        symbols_needed = list(set(t.symbol for t in trades_to_update))
        prices = {}
        symbols_to_fetch = []

        # بررسی کش قیمت قبل از درخواست جدید
        for symbol in symbols_needed:
            if symbol in self._price_cache:
                prices[symbol] = self._price_cache[symbol]
                logger.debug(f"استفاده از قیمت کش‌شده برای {symbol}: {prices[symbol]}")
            else:
                symbols_to_fetch.append(symbol)

        # دریافت قیمت‌های جدید فقط برای نمادهای ضروری
        if symbols_to_fetch:
            logger.debug(f"دریافت قیمت برای {len(symbols_to_fetch)} نماد: {', '.join(symbols_to_fetch)}")
            price_tasks = [self.price_fetcher_callback(symbol) for symbol in symbols_to_fetch]
            # اجرای موازی درخواست‌ها
            price_results = await asyncio.gather(*price_tasks, return_exceptions=True)

            for i, symbol in enumerate(symbols_to_fetch):
                if isinstance(price_results[i], Exception):
                    logger.error(f"خطا در دریافت قیمت برای {symbol} در به‌روزرسانی: {price_results[i]}")
                    prices[symbol] = None  # قیمت نامعتبر است
                else:
                    price = price_results[i]
                    prices[symbol] = price
                    if price is not None:
                        # به‌روزرسانی کش قیمت
                        self._price_cache[symbol] = price
                        logger.debug(f"قیمت دریافت شده برای {symbol}: {price}")
                    else:
                        logger.warning(f"دریافت‌کننده قیمت برای {symbol} مقدار None برگرداند")
        # --- پایان دریافت قیمت‌ها ---

        actions_taken = False  # برای پیگیری اینکه آیا آماری نیاز به به‌روزرسانی دارد
        trades_to_save_price = set()  # مجموعه شناسه‌های معاملاتی که فقط قیمتشان آپدیت شده و نیاز به ذخیره دارند

        # --- بررسی هر معامله باز ---
        for trade in trades_to_update:
            current_price = prices.get(trade.symbol)

            # اگر قیمت برای نماد دریافت نشد، از این معامله رد شو
            if current_price is None:
                logger.warning(
                    f"رد کردن به‌روزرسانی برای معامله {trade.trade_id}: قیمت برای {trade.symbol} در دسترس نیست.")
                continue

            # ذخیره قیمت قبلی برای محاسبات استاپ متحرک
            previous_price = trade.current_price

            # بررسی شرایط خروج (این متد قیمت فعلی معامله را هم آپدیت می‌کند)
            should_exit, exit_reason, exit_quantity = trade.check_exit_conditions(current_price)
            price_was_updated = trade.current_price == current_price  # بررسی اینکه آیا قیمت واقعا آپدیت شد

            # پرچم‌هایی برای جلوگیری از ذخیره تکراری
            trade_status_changed = False
            trailing_stop_updated_flag = False

            # 1. اگر شرط خروج فعال شده است
            if should_exit and exit_quantity > 1e-9:
                actions_taken = True
                trade_status_changed = True  # وضعیت معامله تغییر خواهد کرد

                # خروج جزئی
                if exit_quantity < trade.remaining_quantity - 1e-9:
                    closed_portion = trade.partial_close(current_price, exit_quantity, exit_reason,
                                                         self.commission_rate)
                    self.save_trade_to_db(trade)  # ذخیره معامله با وضعیت جدید 'partially_closed'
                    # ارسال اعلان
                    precision = self.get_symbol_precision(trade.symbol)
                    await self._send_notification(
                        f"PARTIAL CLOSE [{trade.trade_id[:8]}] {trade.symbol}\n"
                        f"📉 مقدار: {exit_quantity:.{precision}f} @ {current_price:.{precision}f}\n"
                        f"دلیل: {exit_reason}\n"
                        f"باقی‌مانده: {trade.remaining_quantity:.{precision}f}"
                        f"{f'، استاپ جدید: {trade.stop_loss:.{precision}f}' if 'take_profit_level' in exit_reason else ''}"
                    )

                    # اگر بخشی از معامله هنوز باز است، استاپ متحرک را دوباره بررسی می‌کنیم
                    if trade.status != 'closed':
                        atr = await self._get_atr_for_trailing(trade.symbol) if self.use_atr_trailing else 0
                        trailing_stop_updated_flag = self._check_trailing_stop(trade, previous_price, current_price,
                                                                               atr)
                        if trailing_stop_updated_flag:
                            actions_taken = True  # آپدیت استاپ هم یک اقدام محسوب می‌شود
                            # ذخیره در DB داخل _check_trailing_stop انجام می‌شود

                # بستن کامل
                else:
                    # تابع close_trade وضعیت را 'closed' می‌کند و در دیتابیس ذخیره می‌کند و اعلان ارسال می‌کند
                    self.close_trade(trade.trade_id, current_price, exit_reason)

            # 2. اگر شرط خروج فعال نشده و معامله هنوز باز است، استاپ متحرک را بررسی کن
            elif trade.status != 'closed':
                # دریافت پارامترهای استاپ متحرک از معامله یا پیکربندی کلی
                trailing_stop_enabled = getattr(trade, 'trailing_stop_params', {}).get('enabled',
                                                                                       self.use_trailing_stop)
                use_atr = getattr(trade, 'trailing_stop_params', {}).get('use_atr', self.use_atr_trailing)

                if trailing_stop_enabled:
                    # فقط در صورت فعال بودن استاپ متحرک، ATR را محاسبه کن
                    atr = await self._get_atr_for_trailing(trade.symbol) if use_atr else 0
                    trailing_stop_updated_flag = self._check_trailing_stop(trade, previous_price, current_price, atr)
                    if trailing_stop_updated_flag:
                        actions_taken = True  # آپدیت استاپ یک اقدام است
                        # ذخیره در DB داخل _check_trailing_stop انجام می‌شود

            # --- >> بخش اصلاح شده: ثبت برای ذخیره اگر فقط قیمت آپدیت شده << ---
            # اگر قیمت در شی Trade آپدیت شده است،
            # و هیچ تغییر وضعیتی (خروج کامل/جزئی) رخ نداده است،
            # و استاپ متحرک هم آپدیت نشده است (چون آپدیت استاپ خودش ذخیره می‌کند)،
            # و معامله هنوز بسته نشده است،
            # آنگاه شناسه معامله را برای ذخیره قیمت جدید اضافه کن.
            if price_was_updated and not trade_status_changed and not trailing_stop_updated_flag and trade.status != 'closed':
                trades_to_save_price.add(trade.trade_id)
            # --- >> پایان بخش اصلاح شده << ---

        # --- >> بخش اصلاح شده: ذخیره معاملاتی که فقط قیمتشان آپدیت شده << ---
        # بعد از بررسی تمام معاملات، معاملاتی که فقط قیمتشان آپدیت شده را ذخیره کن
        if trades_to_save_price:
            saved_count = 0
            for trade_id in trades_to_save_price:
                # دریافت مجدد شی معامله برای اطمینان از آخرین وضعیت
                trade_to_save = self.get_trade(trade_id)
                # دوباره چک کن که معامله هنوز بسته نشده باشد
                if trade_to_save and trade_to_save.status != 'closed':
                    if self.save_trade_to_db(trade_to_save):
                        saved_count += 1
                        actions_taken = True  # ذخیره قیمت هم یک اقدام است که نیاز به آپدیت آمار دارد

            if saved_count > 0:
                logger.debug(f"قیمت {saved_count} معامله در دیتابیس ذخیره شد.")
        # --- >> پایان بخش اصلاح شده << ---

        # اگر هرگونه اقدامی (خروج، آپدیت استاپ، ذخیره قیمت) انجام شده، آمار کلی را به‌روز کن
        if actions_taken:
            self._update_stats()
            # ثبت تغییر موجودی در تاریخچه
            self._save_balance_history("به‌روزرسانی خودکار قیمت‌ها و بررسی معاملات انجام شد")

    async def _get_atr_for_trailing(self, symbol: str) -> float:
        """دریافت مقدار ATR برای استاپ متحرک."""
        # استفاده از کش برای کاهش تعداد درخواست‌ها
        cache_key = f"atr_{symbol}"
        if cache_key in self._calculation_cache:
            return self._calculation_cache[cache_key]

        if not self.data_fetcher_instance:
            return 0.0

        try:
            # استفاده از تایم‌فریم کوتاه‌تر مثل 15m یا 1h برای ATR
            tf = '15m'
            period = self.risk_config.get('atr_trailing_period', 14)  # دوره ATR قابل تنظیم
            df = await self.data_fetcher_instance.get_historical_data(symbol, tf, limit=period + 5)
            if df is not None and not df.empty and len(df) >= period:
                atr_values = talib.ATR(df['high'].values, df['low'].values, df['close'].values, timeperiod=period)
                last_atr = atr_values[~np.isnan(atr_values)][-1] if not np.all(np.isnan(atr_values)) else 0

                # ذخیره در کش
                self._calculation_cache[cache_key] = last_atr
                return last_atr
        except Exception as e:
            logger.error(f"خطا در دریافت ATR برای استاپ متحرک روی {symbol}: {e}")

        return 0.0

    def _check_trailing_stop(self, trade: Trade, previous_price: float, current_price: float, atr: float = 0) -> bool:
        """
        بررسی و تنظیم استاپ متحرک با منطق بهتر و سازگار با تنظیمات.

        Args:
            trade: معامله مورد بررسی
            previous_price: قیمت قبلی
            current_price: قیمت فعلی
            atr: مقدار ATR فعلی (اختیاری)

        Returns:
            True اگر استاپ آپدیت شد، False در غیر این صورت
        """
        # استفاده از DynamicStopManager اگر موجود باشد
        if self.dynamic_stop_manager is not None:
            try:
                old_stop = trade.stop_loss

                # دریافت داده‌های قیمت برای استاپ مبتنی بر ساختار قیمت
                if hasattr(self.dynamic_stop_manager,
                           'use_structure_based_stops') and self.dynamic_stop_manager.use_structure_based_stops:
                    # محاسبه استاپ جدید
                    price_data = None
                    if hasattr(self, '_get_price_data_for_stop_calc'):
                        price_data = self._get_price_data_for_stop_calc(trade.symbol)

                    new_stop = self.dynamic_stop_manager.update_trailing_stop(
                        trade=trade,
                        current_price=current_price,
                        atr=atr,
                        price_data=price_data
                    )
                else:
                    # محاسبه استاپ جدید بدون داده‌های ساختار قیمت
                    new_stop = self.dynamic_stop_manager.update_trailing_stop(
                        trade=trade,
                        current_price=current_price,
                        atr=atr
                    )

                # به‌روزرسانی فقط اگر استاپ سودمندتر باشد
                changed = False
                if trade.direction == 'long' and new_stop > old_stop:
                    trade.stop_loss = new_stop
                    trade.stop_moved_count += 1
                    changed = True
                elif trade.direction == 'short' and new_stop < old_stop:
                    trade.stop_loss = new_stop
                    trade.stop_moved_count += 1
                    changed = True

                if changed:
                    logger.info(
                        f"استاپ متحرک برای {trade.trade_id} ({trade.symbol}) از {old_stop:.5f} به {new_stop:.5f} به‌روز شد")
                    self.save_trade_to_db(trade)
                    self.save_stop_loss_change(trade.trade_id, old_stop, new_stop, "dynamic_stop_manager")
                    return True

                return False

            except Exception as e:
                logger.error(f"خطا در استفاده از مدیریت استاپ پویا: {e}")
                # در صورت خطا، به روش عادی ادامه می‌دهیم

        # استفاده از پارامترهای استاپ متحرک از معامله یا از تنظیمات کلی
        use_trailing_stop = getattr(trade, 'trailing_stop_params', {}).get('enabled', self.use_trailing_stop)

        # بررسی کنید که آیا استاپ متحرک فعال است
        if not use_trailing_stop or trade.status == 'closed':
            return False

        try:
            # دریافت پارامترهای تنظیمات از معامله یا تنظیمات کلی
            # پارامترهای خاص معامله اولویت دارند
            trailing_params = getattr(trade, 'trailing_stop_params', {})
            activation_perc = trailing_params.get('activation_percent',
                                                  self.risk_config.get('trailing_stop_activation_percent', 3.0))
            distance_perc = trailing_params.get('distance_percent',
                                                self.risk_config.get('trailing_stop_distance_percent', 2.25))
            use_atr = trailing_params.get('use_atr', self.use_atr_trailing)
            atr_multiplier = trailing_params.get('atr_multiplier',
                                                 self.risk_config.get('atr_trailing_multiplier', 2.0))

            # استاپ فعلی به عنوان مقدار پیش‌فرض
            new_stop_loss = trade.stop_loss
            changed = False

            # محاسبه درصد تغییر قیمت از ورود (برای فعال‌سازی استاپ متحرک)
            price_change_percent = 0
            if trade.entry_price > 0:
                if trade.direction == 'long':
                    price_change_percent = ((current_price - trade.entry_price) / trade.entry_price) * 100
                else:  # short
                    price_change_percent = ((trade.entry_price - current_price) / trade.entry_price) * 100

            # ردیابی قیمت‌های حداکثر/حداقل
            if not hasattr(trade, 'trailing_stop_params') or 'highest_price' not in trade.trailing_stop_params:
                trade.trailing_stop_params = {
                    'enabled': use_trailing_stop,
                    'activation_percent': activation_perc,
                    'distance_percent': distance_perc,
                    'use_atr': use_atr,
                    'atr_multiplier': atr_multiplier,
                    'activated': False,
                    'highest_price': current_price if trade.direction == 'long' else trade.entry_price,
                    'lowest_price': current_price if trade.direction == 'short' else trade.entry_price
                }

            # به‌روزرسانی قیمت‌های حداکثر/حداقل
            if trade.direction == 'long':
                trade.trailing_stop_params['highest_price'] = max(
                    trade.trailing_stop_params.get('highest_price', current_price),
                    current_price
                )
            else:  # short
                trade.trailing_stop_params['lowest_price'] = min(
                    trade.trailing_stop_params.get('lowest_price', current_price),
                    current_price
                )

            # بررسی اینکه آیا استاپ متحرک قبلاً فعال شده یا باید فعال شود
            activated = trade.trailing_stop_params.get('activated', False)
            if not activated and price_change_percent >= activation_perc:
                # فعال‌سازی استاپ متحرک
                trade.trailing_stop_params['activated'] = True
                activated = True
                logger.info(
                    f"استاپ متحرک برای {trade.trade_id} ({trade.symbol}) فعال شد - تغییر قیمت: {price_change_percent:.2f}%")

            # فقط اگر استاپ متحرک فعال شده است، اقدام به تنظیم آن کن
            if activated:
                # محاسبه مقدار استاپ جدید
                if use_atr and atr > 0:
                    # استاپ مبتنی بر ATR
                    if trade.direction == 'long':
                        high_price = trade.trailing_stop_params.get('highest_price', current_price)
                        new_stop_loss = high_price - (atr * atr_multiplier)
                    else:  # short
                        low_price = trade.trailing_stop_params.get('lowest_price', current_price)
                        new_stop_loss = low_price + (atr * atr_multiplier)
                else:
                    # استاپ مبتنی بر درصد ثابت
                    if trade.direction == 'long':
                        high_price = trade.trailing_stop_params.get('highest_price', current_price)
                        new_stop_loss = high_price * (1 - distance_perc / 100)
                    else:  # short
                        low_price = trade.trailing_stop_params.get('lowest_price', current_price)
                        new_stop_loss = low_price * (1 + distance_perc / 100)

                # به‌روزرسانی فقط اگر استاپ جدید سودآورتر باشد
                if trade.direction == 'long' and new_stop_loss > trade.stop_loss:
                    old_stop = trade.stop_loss
                    trade.stop_loss = new_stop_loss
                    trade.stop_moved_count += 1
                    changed = True
                    logger.info(
                        f"استاپ متحرک برای {trade.trade_id} ({trade.symbol}) از {old_stop:.5f} به {new_stop_loss:.5f} به‌روز شد")
                    self.save_trade_to_db(trade)
                    # ثبت تغییر استاپ لاس در تاریخچه
                    self.save_stop_loss_change(trade.trade_id, old_stop, new_stop_loss, "trailing_stop")
                elif trade.direction == 'short' and new_stop_loss < trade.stop_loss:
                    old_stop = trade.stop_loss
                    trade.stop_loss = new_stop_loss
                    trade.stop_moved_count += 1
                    changed = True
                    logger.info(
                        f"استاپ متحرک برای {trade.trade_id} ({trade.symbol}) از {old_stop:.5f} به {new_stop_loss:.5f} به‌روز شد")
                    self.save_trade_to_db(trade)
                    # ثبت تغییر استاپ لاس در تاریخچه
                    self.save_stop_loss_change(trade.trade_id, old_stop, new_stop_loss, "trailing_stop")

            return changed

        except Exception as e:
            logger.error(f"خطا در به‌روزرسانی استاپ متحرک برای {trade.trade_id}: {e}")
            return False

    def close_trade(self, trade_id: str, exit_price: float, exit_reason: str) -> bool:
        """بستن کامل مقدار باقی‌مانده معامله."""
        with self._trades_lock:
            if trade_id not in self.active_trades:
                logger.warning(f"Attempt to close non-existent or already removed trade: {trade_id}")
                return False
            trade = self.active_trades[trade_id]
            if trade.status == 'closed':
                # ممکن است در یک چرخه آپدیت دیگر بسته شده باشد
                return False

            # استفاده از متد close_trade خود شی Trade
            try:
                closed_portion = trade.close_trade(exit_price, exit_reason, self.commission_rate)
                # وضعیت و PnL نهایی در خود شی محاسبه شده است
                self.save_trade_to_db(trade)  # ذخیره وضعیت نهایی
                self._update_stats()  # به‌روزرسانی آمار کلی

                # ثبت تغییر موجودی در تاریخچه
                self._save_balance_history(f"معامله بسته شد: {trade.symbol} {trade.direction}")

                # ارسال اعلان برای بستن کامل
                precision = self.get_symbol_precision(trade.symbol)
                asyncio.create_task(self._send_notification(
                    f"🛑 معامله بسته شد [{trade.trade_id[:8]}] {trade.symbol}\n"
                    f"دلیل: {exit_reason}\n"
                    f"قیمت خروج: {exit_price:.{precision}f}\n"
                    f"سود/زیان نهایی: {trade.profit_loss:.2f} ({trade.profit_loss_percent:.2f}%)\n"
                    f"مدت: {trade.get_age():.1f} ساعت"
                ))

                # ارسال نتیجه معامله به سیستم ML اگر کالبک ثبت شده باشد
                if self.trade_result_callback:
                    trade_result = TradeResult(
                        trade_id=trade.trade_id,
                        symbol=trade.symbol,
                        direction=trade.direction,
                        entry_price=trade.entry_price,
                        exit_price=exit_price,
                        profit_loss=trade.profit_loss,
                        profit_loss_percent=trade.profit_loss_percent,
                        duration_hours=trade.get_age(),
                        exit_reason=exit_reason,
                        strategy_name=trade.strategy_name,
                        timeframe=trade.timeframe,
                        signal_quality=trade.signal_quality,
                        stop_moved_count=trade.stop_moved_count,
                        tags=trade.tags,
                        market_state=trade.market_state
                    )
                    # ارسال به صورت آسنکرون
                    asyncio.create_task(self.trade_result_callback(trade_result))

                logger.info(f"معامله {trade_id} با موفقیت بسته شد. سود/زیان نهایی: {trade.profit_loss:.2f}")
                return True
            except Exception as e:
                logger.error(f"خطا در نهایی‌سازی بستن معامله {trade_id}: {e}", exc_info=True)
                return False

        # --- توابع دریافت اطلاعات ---

    def get_trade(self, trade_id: str) -> Optional[Trade]:
        """دریافت اطلاعات یک معامله."""
        with self._trades_lock:
            return self.active_trades.get(trade_id)

    def get_open_trades(self) -> List[Trade]:
        """دریافت تمام معاملات باز یا بخشی بسته شده."""
        with self._trades_lock:
            return [t for t in self.active_trades.values() if t.status != 'closed']

    def get_trades_by_symbol(self, symbol: str, include_closed: bool = False) -> List[Trade]:
        """دریافت معاملات یک نماد خاص."""
        with self._trades_lock:
            return [t for t in self.active_trades.values()
                    if t.symbol == symbol and (include_closed or t.status != 'closed')]

    def get_trades_by_tag(self, tag: str, include_closed: bool = False) -> List[Trade]:
        """دریافت معاملات با برچسب خاص."""
        with self._trades_lock:
            return [t for t in self.active_trades.values()
                    if tag in t.tags and (include_closed or t.status != 'closed')]

    def get_trades_by_strategy(self, strategy_name: str, include_closed: bool = False) -> List[Trade]:
        """دریافت معاملات با استراتژی خاص."""
        with self._trades_lock:
            return [t for t in self.active_trades.values()
                    if t.strategy_name == strategy_name and (include_closed or t.status != 'closed')]

    def _calculate_closed_trades_stats(self, closed_trades: List[Trade]):
        """محاسبه آمار معاملات بسته شده."""
        if closed_trades:
            self.stats['winning_trades'] = sum(
                1 for t in closed_trades if t.profit_loss is not None and t.profit_loss > 0)
            self.stats['losing_trades'] = self.stats['closed_trades'] - self.stats['winning_trades']

            # محاسبه کل سود/زیان و کارمزد
            self.stats['total_net_profit_loss'] = sum(t.profit_loss or 0 for t in closed_trades)
            self.stats['total_commission_paid'] = sum(t.commission_paid or 0 for t in closed_trades)

            # نرخ برد
            if self.stats['closed_trades'] > 0:
                self.stats['win_rate'] = (self.stats['winning_trades'] / self.stats['closed_trades']) * 100

            # آمار برد/باخت
            winning_trades = [t for t in closed_trades if t.profit_loss is not None and t.profit_loss > 0]
            losing_trades = [t for t in closed_trades if t.profit_loss is not None and t.profit_loss <= 0]

            if winning_trades:
                self.stats['avg_win_percent'] = sum(t.profit_loss_percent or 0 for t in winning_trades) / len(
                    winning_trades)
                self.stats['largest_win'] = max(t.profit_loss or 0 for t in winning_trades)

            if losing_trades:
                self.stats['avg_loss_percent'] = sum(abs(t.profit_loss_percent or 0) for t in losing_trades) / len(
                    losing_trades)
                self.stats['largest_loss'] = abs(min(t.profit_loss or 0 for t in losing_trades))

            # محاسبه ضریب سود (profit factor)
            total_profit = sum(t.profit_loss for t in winning_trades)
            total_loss = abs(sum(t.profit_loss for t in losing_trades))
            if total_loss > 1e-9:
                self.stats['profit_factor'] = total_profit / total_loss
            else:
                self.stats['profit_factor'] = float('inf') if total_profit > 0 else 0.0

            # میانگین مدت معامله
            durations = [t.get_age() for t in closed_trades]
            if durations:
                self.stats['avg_trade_duration_hours'] = sum(durations) / len(durations)

            # میانگین نسبت ریسک به ریوارد
            risk_rewards = [t.risk_reward_ratio for t in closed_trades if t.risk_reward_ratio]
            if risk_rewards:
                self.stats['avg_risk_reward_ratio'] = sum(risk_rewards) / len(risk_rewards)

    def _calculate_partially_closed_stats(self, partial_trades: List[Trade]):
        """محاسبه آمار معاملات بخشی بسته شده."""
        realized_pnl = sum(t.get_realized_pnl() for t in partial_trades)
        self.stats['realized_pnl_partial_trades'] = round(realized_pnl, 2)

        # محاسبه سود/زیان شناور معاملات باز
        floating_pnl = sum(t.get_floating_pnl() for t in partial_trades)
        self.stats['partially_closed_floating_pnl'] = round(floating_pnl, 2)

    def _calculate_advanced_stats(self, closed_trades: List[Trade]):
        """محاسبه آمار پیشرفته."""
        if not closed_trades:
            return
        self.stats['max_drawdown'] = 0.0
        # محاسبه Sharpe و Sortino ratio
        daily_returns = self._calculate_daily_returns(closed_trades)
        if daily_returns:
            # Sharpe Ratio
            returns_array = np.array(list(daily_returns.values()))
            avg_return = np.mean(returns_array)
            std_return = np.std(returns_array)

            if std_return > 0:
                self.stats['sharpe_ratio'] = round((avg_return / std_return) * np.sqrt(252),
                                                   2)  # ضرب در ریشه 252 برای سالانه کردن

            # Sortino Ratio
            downside_returns = [r for r in returns_array if r < 0]
            if downside_returns:
                downside_std = np.std(downside_returns)
                if downside_std > 0:
                    self.stats['sortino_ratio'] = round((avg_return / downside_std) * np.sqrt(252), 2)

        # محاسبه Recovery Factor
        if hasattr(self, 'peak_equity') and self.stats['max_drawdown'] > 0:
            self.stats['recovery_factor'] = round(self.stats['total_net_profit_loss'] / self.stats['max_drawdown'], 2)

        # محاسبه Profit Consistency
        if daily_returns:
            positive_days = sum(1 for r in daily_returns.values() if r > 0)
            total_days = len(daily_returns)
            if total_days > 0:
                self.stats['profit_consistency'] = round((positive_days / total_days) * 100, 2)

    def _calculate_daily_returns(self, closed_trades: List[Trade]) -> Dict[str, float]:
        """محاسبه بازده روزانه از معاملات بسته شده."""
        daily_pnl = {}

        for trade in closed_trades:
            if trade.exit_time:
                exit_date = trade.exit_time.date().isoformat()
                if exit_date not in daily_pnl:
                    daily_pnl[exit_date] = 0

                daily_pnl[exit_date] += trade.profit_loss or 0

        # تبدیل به درصد از سرمایه
        daily_returns = {}
        for date, pnl in daily_pnl.items():
            # تخمین سرمایه در آن تاریخ (غیردقیق)
            estimated_equity = self.initial_balance  # در نسخه‌های پیشرفته‌تر می‌توان تاریخچه موجودی را پیگیری کرد
            if estimated_equity > 0:
                daily_returns[date] = pnl / estimated_equity

        self.stats['daily_pnl'] = daily_pnl
        return daily_returns

    def _calculate_distribution_stats(self, all_trades: List[Trade]):
        """محاسبه توزیع معاملات بر اساس نماد، استراتژی و برچسب."""
        # توزیع نمادها
        symbols_distribution = {}
        for trade in all_trades:
            symbol = trade.symbol
            if symbol not in symbols_distribution:
                symbols_distribution[symbol] = 0
            symbols_distribution[symbol] += 1

        self.stats['symbols_distribution'] = symbols_distribution

        # توزیع استراتژی‌ها
        strategies_distribution = {}
        for trade in all_trades:
            strategy = trade.strategy_name or 'unknown'
            if strategy not in strategies_distribution:
                strategies_distribution[strategy] = 0
            strategies_distribution[strategy] += 1

        self.stats['trades_by_strategy'] = strategies_distribution

        # توزیع برچسب‌ها
        tags_distribution = {}
        for trade in all_trades:
            for tag in trade.tags:
                if tag not in tags_distribution:
                    tags_distribution[tag] = 0
                tags_distribution[tag] += 1

        self.stats['trades_by_tag'] = tags_distribution

    def get_stats(self) -> Dict[str, Any]:
        """دریافت آمار معاملات (آخرین آمار محاسبه شده)."""
        # برای اطمینان، آمار را قبل از بازگرداندن به‌روز کن
        self._update_stats()
        return self.stats.copy()

    def get_portfolio_stats(self) -> PortfolioStats:
        """دریافت آمار پورتفولیو به صورت ساختاریافته."""
        # به‌روزرسانی آمار
        self._update_stats()

        # تبدیل به PortfolioStats
        stats_dict = self.stats.copy()
        portfolio_stats = PortfolioStats(
            total_trades_opened=stats_dict.get('total_trades_opened', 0),
            open_trades=stats_dict.get('open_trades', 0),
            partially_closed_trades=stats_dict.get('partially_closed_trades', 0),
            closed_trades=stats_dict.get('closed_trades', 0),
            winning_trades=stats_dict.get('winning_trades', 0),
            losing_trades=stats_dict.get('losing_trades', 0),
            total_net_profit_loss=stats_dict.get('total_net_profit_loss', 0.0),
            total_commission_paid=stats_dict.get('total_commission_paid', 0.0),
            win_rate=stats_dict.get('win_rate', 0.0),
            profit_factor=stats_dict.get('profit_factor', 0.0),
            current_equity=stats_dict.get('current_equity', 0.0),
            current_open_pnl=stats_dict.get('current_open_pnl', 0.0),
            current_drawdown=stats_dict.get('current_drawdown', 0.0),
            realized_pnl_partial_trades=stats_dict.get('realized_pnl_partial_trades', 0.0),
            avg_win_percent=stats_dict.get('avg_win_percent', 0.0),
            avg_loss_percent=stats_dict.get('avg_loss_percent', 0.0),
            largest_win=stats_dict.get('largest_win', 0.0),
            largest_loss=stats_dict.get('largest_loss', 0.0),
            avg_trade_duration_hours=stats_dict.get('avg_trade_duration_hours', 0.0),
            avg_risk_reward_ratio=stats_dict.get('avg_risk_reward_ratio', 0.0),
            symbols_distribution=stats_dict.get('symbols_distribution', {}),
            daily_pnl=stats_dict.get('daily_pnl', {}),
            trades_by_tag=stats_dict.get('trades_by_tag', {}),
            trades_by_strategy=stats_dict.get('trades_by_strategy', {})
        )

        return portfolio_stats

    def get_trade_summaries(self, include_closed: bool = False) -> List[TradeSummary]:
        """دریافت خلاصه معاملات به صورت ساختاریافته."""
        with self._trades_lock:
            trades = list(self.active_trades.values())
            if not include_closed:
                trades = [t for t in trades if t.status != 'closed']

            summaries = [TradeSummary.from_trade(t) for t in trades]

            # مرتب‌سازی بر اساس زمان ورود (جدیدترین اول)
            summaries.sort(key=lambda s: s.entry_time or datetime.min, reverse=True)

            return summaries

    def export_trade_history(self, include_open: bool = False, limit: int = 1000) -> List[Dict[str, Any]]:
        """استخراج تاریخچه معاملات از دیتابیس."""
        history = []
        query = "SELECT data FROM trades"
        if not include_open:
            query += " WHERE status = 'closed'"
        query += " ORDER BY timestamp DESC"  # مرتب‌سازی معکوس برای دریافت جدیدترین معاملات اول
        query += f" LIMIT {limit}"  # محدود کردن تعداد رکوردها

        try:
            if not self.cursor:
                return []

            with self._db_lock:
                self.cursor.execute(query)
                rows = self.cursor.fetchall()
                for row in rows:
                    if row['data']:
                        try:
                            history.append(json.loads(row['data']))
                        except:
                            pass
        except Exception as e:
            logger.error(f"Error exporting trade history: {e}")

        return history

    def export_trades_to_dataframe(self, include_open: bool = True) -> pd.DataFrame:
        """صدور معاملات به فرمت DataFrame برای تحلیل."""
        try:
            trade_history = self.export_trade_history(include_open=include_open)

            if not trade_history:
                return pd.DataFrame()

            # استخراج فیلدهای کلیدی
            trade_rows = []
            for trade_data in trade_history:
                row = {
                    'trade_id': trade_data.get('trade_id'),
                    'symbol': trade_data.get('symbol'),
                    'direction': trade_data.get('direction'),
                    'entry_price': trade_data.get('entry_price'),
                    'stop_loss': trade_data.get('stop_loss'),
                    'take_profit': trade_data.get('take_profit'),
                    'quantity': trade_data.get('quantity'),
                    'timestamp': trade_data.get('timestamp'),  # ISO string
                    'status': trade_data.get('status'),
                    'exit_price': trade_data.get('exit_price'),
                    'exit_time': trade_data.get('exit_time'),  # ISO string
                    'exit_reason': trade_data.get('exit_reason'),
                    'profit_loss': trade_data.get('profit_loss'),
                    'profit_loss_percent': trade_data.get('profit_loss_percent'),
                    'risk_reward_ratio': trade_data.get('risk_reward_ratio'),
                    'strategy_name': trade_data.get('strategy_name'),
                    'timeframe': trade_data.get('timeframe'),
                    'tags': ','.join(trade_data.get('tags', [])),
                    'commission_paid': trade_data.get('commission_paid'),
                    'max_favorable_excursion': trade_data.get('max_favorable_excursion'),
                    'max_adverse_excursion': trade_data.get('max_adverse_excursion'),
                    'stop_moved_count': trade_data.get('stop_moved_count')
                }
                trade_rows.append(row)

            # ایجاد DataFrame
            df = pd.DataFrame(trade_rows)

            # تبدیل ستون‌های زمان به datetime
            if 'timestamp' in df.columns:
                df['timestamp'] = pd.to_datetime(df['timestamp'], errors='coerce')
            if 'exit_time' in df.columns:
                df['exit_time'] = pd.to_datetime(df['exit_time'], errors='coerce')

            # محاسبه ستون مدت معامله
            if 'timestamp' in df.columns and 'exit_time' in df.columns:
                df['duration_hours'] = (df['exit_time'] - df['timestamp']).dt.total_seconds() / 3600

            # مرتب‌سازی براساس زمان
            if 'timestamp' in df.columns:
                df.sort_values('timestamp', inplace=True)

            return df

        except Exception as e:
            logger.error(f"Error exporting trades to DataFrame: {e}")
            return pd.DataFrame()

        # --- مدیریت تسک پریودیک ---

    async def periodic_price_update(self):
        """حلقه به‌روزرسانی دوره‌ای قیمت‌ها."""
        logger.info(f"شروع حلقه به‌روزرسانی دوره‌ای قیمت‌ها (فاصله زمانی: {self.price_update_interval}ثانیه)")
        self.price_update_running = True

        while not self._shutdown_requested.is_set():
            start_loop_time = time.monotonic()
            try:
                if self.auto_update_prices:
                    # استفاده از متد داینامیک به‌روزرسانی قیمت
                    await self.update_trade_prices()
            except Exception as e:
                logger.error(f"خطا در حلقه به‌روزرسانی دوره‌ای قیمت: {e}", exc_info=True)

            # محاسبه زمان انتظار برای رسیدن به اینتروال بعدی
            elapsed_time = time.monotonic() - start_loop_time
            wait_time = max(0, self.price_update_interval - elapsed_time)

            try:
                # انتظار با قابلیت لغو شدن
                await asyncio.wait_for(self._shutdown_requested.wait(), timeout=wait_time)
                # اگر wait کامل شد (یعنی shutdown درخواست شده)، از حلقه خارج شو
                if self._shutdown_requested.is_set():
                    logger.info("درخواست خاتمه دریافت شد، توقف به‌روزرسانی دوره‌ای قیمت.")
                    break
            except asyncio.TimeoutError:
                continue  # ادامه به حلقه بعدی

        logger.info("حلقه به‌روزرسانی دوره‌ای قیمت پایان یافت.")
        self.price_update_running = False

    async def start_periodic_price_update(self):
        """شروع تسک پس‌زمینه به‌روزرسانی قیمت‌ها."""
        if self._periodic_task and not self._periodic_task.done():
            logger.warning("تسک به‌روزرسانی دوره‌ای قیمت در حال اجرا است.")
            return
        if not self.auto_update_prices:
            logger.info("به‌روزرسانی خودکار قیمت غیرفعال است، عدم شروع تسک دوره‌ای.")
            return

        self._shutdown_requested.clear()
        self._periodic_task = asyncio.create_task(self.periodic_price_update())
        self._tasks.append(self._periodic_task)
        logger.info("تسک به‌روزرسانی دوره‌ای قیمت شروع شد.")

    async def stop_periodic_price_update(self):
        """توقف تسک پس‌زمینه به‌روزرسانی قیمت‌ها."""
        if not self._periodic_task or self._periodic_task.done():
            logger.debug("تسک به‌روزرسانی دوره‌ای قیمت در حال اجرا نیست یا قبلاً متوقف شده است.")
            return

        logger.info("درخواست توقف برای تسک به‌روزرسانی دوره‌ای قیمت...")
        self._shutdown_requested.set()
        try:
            # منتظر ماندن برای اتمام تسک (با تایم‌اوت)
            await asyncio.wait_for(self._periodic_task, timeout=5)
            logger.info("تسک به‌روزرسانی دوره‌ای قیمت به‌طور عادی متوقف شد.")
        except asyncio.TimeoutError:
            logger.warning("تایم‌اوت هنگام انتظار برای توقف تسک دوره‌ای، در حال لغو...")
            self._periodic_task.cancel()
        except asyncio.CancelledError:
            logger.info("تسک به‌روزرسانی دوره‌ای قیمت لغو شد.")
        except Exception as e:
            logger.error(f"خطا در توقف تسک دوره‌ای: {e}")
        finally:
            self._periodic_task = None

    def cleanup_resources(self):
        """پاکسازی منابع ThreadPoolExecutor و کش‌ها."""
        try:
            # بستن ThreadPoolExecutor
            if hasattr(self, '_thread_executor'):
                self._thread_executor.shutdown(wait=False)

            # پاکسازی کش‌ها
            if hasattr(self, '_price_cache'):
                self._price_cache.clear()
            if hasattr(self, '_correlation_cache'):
                self._correlation_cache.clear()
            if hasattr(self, '_calculation_cache'):
                self._calculation_cache.clear()

        except Exception as e:
            logger.error(f"خطا در پاکسازی منابع: {e}")

    async def shutdown(self):
        """پاکسازی و بستن منابع TradeManager."""
        logger.info("خاتمه کار TradeManager...")
        await self.stop_periodic_price_update()

        # لغو همه تسک‌های در حال اجرا
        for task in self._tasks:
            if task and not task.done():
                task.cancel()

        # پاکسازی منابع
        self.cleanup_resources()

        # بستن اتصال دیتابیس
        if self.conn:
            try:
                # اجرای بستن در ترد جداگانه
                await asyncio.to_thread(self.conn.close)
                logger.info("اتصال به دیتابیس بسته شد.")
            except Exception as e:
                logger.error(f"خطا در بستن اتصال دیتابیس: {e}")
        self.conn = None
        self.cursor = None

        logger.info("خاتمه کار TradeManager به پایان رسید.")

        # --- متدهای کمکی اضافی ---

    def _get_adapted_config_for_trade(self, trade: Trade) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """
        دریافت کانفیگ مناسب برای معامله موجود براساس استراتژی و تایم‌فریم.

        Args:
            trade: معامله مورد نظر

        Returns:
            Tuple شامل (کانفیگ کامل، بخش مدیریت ریسک)
        """
        # ایجاد یک سیگنال مجازی برای استفاده از متد get_adapted_config
        mock_signal = SignalInfo(
            symbol=trade.symbol,
            direction=trade.direction,
            entry_price=trade.entry_price,
            stop_loss=trade.stop_loss,
            take_profit=trade.take_profit,
            risk_reward_ratio=trade.risk_reward_ratio
        )

        # تنظیم سایر فیلدهای سیگنال
        if hasattr(trade, 'strategy_name'):
            mock_signal.strategy_name = trade.strategy_name

        if hasattr(trade, 'timeframe'):
            mock_signal.timeframe = trade.timeframe

        if hasattr(trade, 'market_state'):
            mock_signal.market_state = trade.market_state

        # استفاده از متد اصلی _get_adapted_config
        return self._get_adapted_config(mock_signal)

    def set_trade_tags(self, trade_id: str, tags: List[str]) -> bool:
        """تنظیم برچسب‌های یک معامله."""
        trade = self.get_trade(trade_id)
        if not trade:
            logger.warning(f"Cannot set tags: Trade {trade_id} not found")
            return False

        # تنظیم برچسب‌ها
        trade.tags = tags

        # ذخیره در دیتابیس
        return self.save_trade_to_db(trade)

    def add_trade_tag(self, trade_id: str, tag: str) -> bool:
        """اضافه کردن یک برچسب به معامله."""
        trade = self.get_trade(trade_id)
        if not trade:
            logger.warning(f"Cannot add tag: Trade {trade_id} not found")
            return False

        # اضافه کردن برچسب اگر هنوز وجود ندارد
        if tag not in trade.tags:
            trade.add_tag(tag)

            # ذخیره در دیتابیس
            return self.save_trade_to_db(trade)

        return True

    def remove_trade_tag(self, trade_id: str, tag: str) -> bool:
        """حذف یک برچسب از معامله."""
        trade = self.get_trade(trade_id)
        if not trade:
            logger.warning(f"Cannot remove tag: Trade {trade_id} not found")
            return False

        # حذف برچسب اگر وجود دارد
        if tag in trade.tags:
            trade.remove_tag(tag)

            # ذخیره در دیتابیس
            return self.save_trade_to_db(trade)

        return True

    def set_trade_note(self, trade_id: str, note: str) -> bool:
        """تنظیم یادداشت برای معامله."""
        trade = self.get_trade(trade_id)
        if not trade:
            logger.warning(f"Cannot set note: Trade {trade_id} not found")
            return False

        # تنظیم یادداشت
        trade.set_note(note)

        # ذخیره در دیتابیس
        return self.save_trade_to_db(trade)

    def set_max_duration(self, trade_id: str, days: float) -> bool:
        """تنظیم حداکثر مدت نگهداری معامله."""
        trade = self.get_trade(trade_id)
        if not trade:
            logger.warning(f"Cannot set max duration: Trade {trade_id} not found")
            return False

        # تنظیم حداکثر مدت
        trade.set_max_duration(days)

        # ذخیره در دیتابیس
        return self.save_trade_to_db(trade)

    def set_partial_tp(self, trade_id: str, percent: float, portion_size: float = 0.5) -> bool:
        """تنظیم خروج جزئی خودکار برای معامله."""
        trade = self.get_trade(trade_id)
        if not trade:
            logger.warning(f"Cannot set partial TP: Trade {trade_id} not found")
            return False

        # تنظیم خروج جزئی
        trade.set_partial_tp(percent, portion_size)

        # ذخیره در دیتابیس
        return self.save_trade_to_db(trade)

    def update_stop_loss(self, trade_id: str, new_stop_loss: float, reason: str = "manual") -> bool:
        """به‌روزرسانی دستی حد ضرر یک معامله."""
        trade = self.get_trade(trade_id)
        if not trade:
            logger.warning(f"Cannot update stop loss: Trade {trade_id} not found")
            return False

        # بررسی معتبر بودن استاپ جدید
        if new_stop_loss <= 0:
            logger.error(f"Invalid stop loss value: {new_stop_loss}")
            return False

        # بررسی جهت معامله
        if trade.direction == 'long' and new_stop_loss >= trade.entry_price:
            logger.error(f"Invalid stop loss for LONG position: {new_stop_loss} >= {trade.entry_price}")
            return False
        elif trade.direction == 'short' and new_stop_loss <= trade.entry_price:
            logger.error(f"Invalid stop loss for SHORT position: {new_stop_loss} <= {trade.entry_price}")
            return False

        # ثبت تغییر استاپ لاس
        old_stop = trade.stop_loss
        trade.stop_loss = new_stop_loss
        trade.stop_moved_count += 1

        # ذخیره در دیتابیس
        success = self.save_trade_to_db(trade)

        if success:
            # ثبت در تاریخچه تغییرات
            self.save_stop_loss_change(trade_id, old_stop, new_stop_loss, reason)
            logger.info(f"Manual stop loss update for {trade_id}: {old_stop:.5f} -> {new_stop_loss:.5f} ({reason})")

        return success

    def update_take_profit(self, trade_id: str, new_take_profit: float, reason: str = "manual") -> bool:
        """
        به‌روزرسانی دستی حد سود یک معامله موجود.

        Args:
            trade_id: شناسه معامله برای به‌روزرسانی.
            new_take_profit: قیمت جدید حد سود.
            reason: دلیل تغییر (پیش‌فرض: دستی).

        Returns:
            True اگر به‌روزرسانی موفق بود، False در غیر این صورت.
        """
        with self._trades_lock:
            trade = self.get_trade(trade_id)
            if not trade:
                logger.warning(f"Cannot update take profit: Trade {trade_id} not found")
                return False

            if trade.status == 'closed':
                logger.warning(f"Cannot update take profit for already closed trade {trade_id}")
                return False

            # بررسی معتبر بودن TP جدید
            if new_take_profit <= 0:
                logger.error(f"Invalid take profit value: {new_take_profit}")
                return False

            # بررسی جهت معامله
            if trade.direction == 'long':
                if new_take_profit <= trade.entry_price:
                    logger.error(
                        f"Invalid take profit for LONG position: {new_take_profit} <= entry {trade.entry_price}")
                    return False
                # بررسی اینکه TP جدید از SL فعلی بالاتر باشد
                if new_take_profit <= trade.stop_loss:
                    logger.error(
                        f"Invalid take profit for LONG position: {new_take_profit} <= stop_loss {trade.stop_loss}")
                    return False
            elif trade.direction == 'short':
                if new_take_profit >= trade.entry_price:
                    logger.error(
                        f"Invalid take profit for SHORT position: {new_take_profit} >= entry {trade.entry_price}")
                    return False
                # بررسی اینکه TP جدید از SL فعلی پایین‌تر باشد
                if new_take_profit >= trade.stop_loss:
                    logger.error(
                        f"Invalid take profit for SHORT position: {new_take_profit} >= stop_loss {trade.stop_loss}")
                    return False

            # ثبت تغییر حد سود
            old_tp = trade.take_profit
            trade.take_profit = new_take_profit

            # آیا معامله حد سود چند مرحله‌ای دارد؟
            if trade.is_multitp_enabled and trade.take_profit_levels:
                # اگر Multi-TP فعال است، باید منطق مدیریت سطوح TP بازبینی شود.
                # ساده‌ترین راه این است که Multi-TP را غیرفعال کنیم یا سطوح را دوباره محاسبه کنیم.
                # در اینجا، فرض می‌کنیم که تغییر TP نهایی، استراتژی Multi-TP را غیرفعال می‌کند
                # یا حداقل نیاز به بازبینی دستی دارد.
                logger.warning(f"Manually updated final TP for multi-TP trade {trade_id}. "
                               f"Multi-TP levels might need review or recalculation.")
                # می‌توان Multi-TP را غیرفعال کرد:
                # trade.is_multitp_enabled = False
                # trade.take_profit_levels = []
                # trade.current_tp_level_index = 0

                # یا فقط آخرین سطح را آپدیت کرد (اگر منطقی باشد):
                last_index = len(trade.take_profit_levels) - 1
                if last_index >= 0:
                    new_levels = trade.take_profit_levels.copy()
                    new_levels[last_index] = (new_take_profit, new_levels[last_index][1])
                    trade.take_profit_levels = new_levels
                    # محاسبه مجدد R/R اگر لازم است
                    if trade.initial_stop_loss is not None and trade.entry_price is not None:
                        risk_dist = abs(trade.entry_price - trade.initial_stop_loss)
                        if risk_dist > 1e-9:
                            reward_dist = abs(new_take_profit - trade.entry_price)
                            trade.risk_reward_ratio = reward_dist / risk_dist

            # ذخیره در دیتابیس
            success = self.save_trade_to_db(trade)

            if success:
                # ثبت تغییر در تاریخچه (اگر لازم است)
                # self.save_level_change(trade_id, "take_profit", old_tp, new_take_profit, reason)
                logger.info(
                    f"Manually updated take profit for {trade_id}: {old_tp:.5f} -> {new_take_profit:.5f} ({reason})")

            return success

    def recalculate_multi_tp(self, trade_id: str) -> bool:
        """محاسبه مجدد سطوح حد سود چندگانه با استفاده از TP نهایی فعلی."""
        trade = self.get_trade(trade_id)
        if not trade:
            logger.warning(f"Cannot recalculate TP levels: Trade {trade_id} not found")
            return False

        # محاسبه مجدد سطوح TP
        adapted_config, adapted_risk_config = self._get_adapted_config_for_trade(trade)
        self._setup_multi_tp_levels(trade, adapted_risk_config)

        # ذخیره در دیتابیس
        return self.save_trade_to_db(trade)

    def manual_close_trade(self, trade_id: str, exit_price: float = None, exit_reason: str = "manual_close",
                           exit_portion: float = 1.0) -> bool:
        """بستن دستی کامل یا بخشی از معامله."""
        trade = self.get_trade(trade_id)
        if not trade:
            logger.warning(f"Cannot close trade: Trade {trade_id} not found")
            return False

        if trade.status == 'closed':
            logger.warning(f"Trade {trade_id} is already closed")
            return False

        # اگر قیمت خروج ارائه نشده است، از قیمت فعلی استفاده کن
        if exit_price is None:
            if trade.current_price:
                exit_price = trade.current_price
            else:
                # درخواست قیمت فعلی
                if self.price_fetcher_callback:
                    try:
                        exit_price = asyncio.run(self.price_fetcher_callback(trade.symbol))
                    except Exception as e:
                        logger.error(f"Error fetching current price for {trade.symbol}: {e}")
                        return False

                if not exit_price:
                    logger.error(f"Cannot close trade {trade_id}: No exit price available")
                    return False

        try:
            # بستن بخشی یا کامل معامله
            if exit_portion < 1.0 and exit_portion > 0:
                exit_quantity = trade.remaining_quantity * exit_portion
                if exit_quantity <= 1e-9:
                    logger.error(f"Invalid exit quantity: {exit_quantity}")
                    return False

                # بستن بخشی
                closed_portion = trade.partial_close(exit_price, exit_quantity, exit_reason, self.commission_rate)
                if closed_portion:
                    self.save_trade_to_db(trade)
                    self._update_stats()

                    # اطلاع‌رسانی
                    asyncio.create_task(self._send_notification(
                        f"MANUAL PARTIAL CLOSE [{trade.trade_id[:8]}] {trade.symbol}\n"
                        f"Quantity: {exit_quantity:.6f} @ {exit_price:.6f}\n"
                        f"PnL: {closed_portion.get('net_pnl', 0):.2f}\n"
                        f"Remaining: {trade.remaining_quantity:.6f}"
                    ))

                    logger.info(f"Manually partially closed trade {trade_id} ({exit_portion:.0%})")
                    return True
            else:
                # بستن کامل
                success = self.close_trade(trade_id, exit_price, exit_reason)
                if success:
                    logger.info(f"Manually closed trade {trade_id}")
                    return True

            return False

        except Exception as e:
            logger.error(f"Error manually closing trade {trade_id}: {e}")
            return False

    def plot_equity_curve(self) -> Optional[Dict[str, Any]]:
        """ساخت داده‌های نمودار منحنی سرمایه."""
        try:
            # دریافت تاریخچه موجودی از دیتابیس
            if not self.conn or not self.cursor:
                return None

            with self._db_lock:
                self.cursor.execute(
                    "SELECT timestamp, balance, equity, open_pnl FROM balance_history ORDER BY timestamp"
                )
                rows = self.cursor.fetchall()

            if not rows:
                return None

            # تبدیل به دیکشنری برای نمودار
            dates = []
            equity_values = []
            balance_values = []
            open_pnl_values = []

            for row in rows:
                try:
                    timestamp = datetime.fromisoformat(row['timestamp']).strftime('%Y-%m-%d %H:%M:%S')
                    dates.append(timestamp)
                    balance_values.append(row['balance'])
                    equity_values.append(row['equity'])
                    open_pnl_values.append(row['open_pnl'])
                except (ValueError, TypeError, KeyError):
                    pass

            # محاسبه آمار اضافی
            if equity_values:
                peak_equity = max(equity_values)
                current_equity = equity_values[-1]
                drawdown = [(peak - eq) / peak * 100 if peak > 0 else 0 for eq, peak in
                            zip(equity_values, np.maximum.accumulate(equity_values))]
                max_drawdown = max(drawdown) if drawdown else 0

                # بازده از ابتدا
                initial_equity = equity_values[0] if equity_values[0] > 0 else current_equity
                total_return = (current_equity - initial_equity) / initial_equity * 100 if initial_equity > 0 else 0
            else:
                peak_equity = 0
                current_equity = 0
                max_drawdown = 0
                total_return = 0

            return {
                'dates': dates,
                'equity': equity_values,
                'balance': balance_values,
                'open_pnl': open_pnl_values,
                'drawdown': drawdown if 'drawdown' in locals() else [],
                'stats': {
                    'peak_equity': peak_equity,
                    'current_equity': current_equity,
                    'max_drawdown_percent': max_drawdown,
                    'total_return_percent': total_return
                }
            }

        except Exception as e:
            logger.error(f"Error plotting equity curve: {e}")
            return None

    def get_daily_profit_chart_data(self) -> Optional[Dict[str, Any]]:
        """دریافت داده‌های نمودار سود/زیان روزانه."""
        try:
            # به‌روزرسانی آمار
            self._update_stats()

            daily_pnl = self.stats.get('daily_pnl', {})
            if not daily_pnl:
                return None

            # مرتب‌سازی براساس تاریخ
            sorted_days = sorted(daily_pnl.keys())
            profits = []
            losses = []

            for day in sorted_days:
                pnl = daily_pnl[day]
                if pnl >= 0:
                    profits.append(pnl)
                    losses.append(0)
                else:
                    profits.append(0)
                    losses.append(abs(pnl))

            # محاسبه تجمعی
            cumulative = []
            running_sum = 0
            for day in sorted_days:
                running_sum += daily_pnl[day]
                cumulative.append(running_sum)

            return {
                'dates': sorted_days,
                'profits': profits,
                'losses': losses,
                'cumulative': cumulative
            }

        except Exception as e:
            logger.error(f"Error getting daily profit chart data: {e}")
            return None

    def get_trading_heatmap_data(self) -> Optional[Dict[str, Any]]:
        """دریافت داده‌های هیت‌مپ معاملاتی (روز/ساعت)."""
        try:
            if not self.conn or not self.cursor:
                return None

            # دریافت داده‌های همه معاملات
            with self._db_lock:
                self.cursor.execute(
                    "SELECT data FROM trades WHERE status = 'closed'"
                )
                rows = self.cursor.fetchall()

            if not rows:
                return None

            # پردازش داده‌ها
            day_hour_pnl = {}  # {(day, hour): [pnl_values]}

            for row in rows:
                try:
                    trade_data = json.loads(row['data'])
                    entry_time = datetime.fromisoformat(trade_data.get('timestamp')) if trade_data.get(
                        'timestamp') else None
                    pnl = trade_data.get('profit_loss')

                    if entry_time and pnl is not None:
                        day_name = entry_time.strftime('%A')  # نام روز هفته
                        hour = entry_time.hour

                        key = (day_name, hour)
                        if key not in day_hour_pnl:
                            day_hour_pnl[key] = []

                        day_hour_pnl[key].append(pnl)

                except (json.JSONDecodeError, ValueError, TypeError, KeyError):
                    pass

            # محاسبه میانگین‌ها
            days = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
            hours = list(range(24))

            avg_pnl = {}
            win_rate = {}
            trade_count = {}

            for day in days:
                for hour in hours:
                    key = (day, hour)
                    values = day_hour_pnl.get(key, [])

                    if values:
                        avg_pnl[key] = sum(values) / len(values)
                        win_count = sum(1 for v in values if v > 0)
                        win_rate[key] = (win_count / len(values)) * 100
                        trade_count[key] = len(values)
                    else:
                        avg_pnl[key] = 0
                        win_rate[key] = 0
                        trade_count[key] = 0

            return {
                'avg_pnl': avg_pnl,
                'win_rate': win_rate,
                'trade_count': trade_count,
                'days': days,
                'hours': hours
            }

        except Exception as e:
            logger.error(f"Error getting trading heatmap data: {e}")
            return None

    def get_symbol_performance_data(self) -> Optional[Dict[str, Any]]:
        """دریافت داده‌های عملکرد به تفکیک نماد."""
        try:
            # به‌روزرسانی آمار
            self._update_stats()

            symbol_dist = self.stats.get('symbols_distribution', {})
            if not symbol_dist:
                return None

            # دریافت داده‌های معاملاتی نمادها
            symbol_data = {}

            for trade in self.export_trade_history(include_open=True):
                symbol = trade.get('symbol')
                if not symbol:
                    continue

                if symbol not in symbol_data:
                    symbol_data[symbol] = {
                        'total_trades': 0,
                        'winning_trades': 0,
                        'losing_trades': 0,
                        'total_pnl': 0,
                        'win_rate': 0,
                        'avg_pnl': 0,
                        'avg_win': 0,
                        'avg_loss': 0,
                        'best_trade': 0,
                        'worst_trade': 0
                    }

                # تجمیع داده‌ها
                if trade.get('status') == 'closed':
                    pnl = trade.get('profit_loss', 0)
                    symbol_data[symbol]['total_trades'] += 1
                    symbol_data[symbol]['total_pnl'] += pnl if pnl else 0

                    if pnl and pnl > 0:
                        symbol_data[symbol]['winning_trades'] += 1
                        symbol_data[symbol]['best_trade'] = max(symbol_data[symbol]['best_trade'], pnl)
                    elif pnl and pnl <= 0:
                        symbol_data[symbol]['losing_trades'] += 1
                        symbol_data[symbol]['worst_trade'] = min(symbol_data[symbol]['worst_trade'], pnl)

            # محاسبه آمار
            for symbol, data in symbol_data.items():
                if data['total_trades'] > 0:
                    data['win_rate'] = (data['winning_trades'] / data['total_trades']) * 100
                    data['avg_pnl'] = data['total_pnl'] / data['total_trades']

                    if data['winning_trades'] > 0:
                        winning_values = [trade.get('profit_loss', 0) for trade in self.export_trade_history()
                                          if trade.get('symbol') == symbol and trade.get('profit_loss', 0) > 0]
                        data['avg_win'] = sum(winning_values) / len(winning_values) if winning_values else 0

                    if data['losing_trades'] > 0:
                        losing_values = [trade.get('profit_loss', 0) for trade in self.export_trade_history()
                                         if trade.get('symbol') == symbol and trade.get('profit_loss', 0) <= 0]
                        data['avg_loss'] = sum(losing_values) / len(losing_values) if losing_values else 0

            # مرتب‌سازی بر اساس سودآوری
            sorted_symbols = sorted(symbol_data.items(), key=lambda x: x[1]['total_pnl'], reverse=True)

            return {
                'symbols': [item[0] for item in sorted_symbols],
                'data': symbol_data
            }

        except Exception as e:
            logger.error(f"Error getting symbol performance data: {e}")
            return None