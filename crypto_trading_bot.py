"""
ماژول crypto_trading_bot.py: برنامه اصلی ربات معاملاتی ارزهای دیجیتال
این ماژول مسئول یکپارچه‌سازی تمام اجزای سیستم و اجرای ربات معاملاتی است.
بهینه‌سازی شده با قابلیت‌های یادگیری تطبیقی، مدیریت همبستگی و مکانیزم بازخورد سریع
"""

import os
import sys
import asyncio
import argparse
import logging
import json
import time
import traceback
import uuid
from datetime import datetime, timedelta
import platform
import copy
import difflib

import yaml
from typing import Dict, Optional, Any, List, Tuple, Set, Union, Callable
import signal
if sys.platform == "win32":
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

# ماژول‌های سیستم
from market_data_fetcher import MarketDataFetcher
from signal_processor import SignalProcessor
from trade_manager import TradeManager
from signal_generator import SignalGenerator, SignalInfo, TradeResult
from multi_tp_trade import Trade  # Import the Trade class
from signal_generator import TradeResult # Import the TradeResult class
# ماژول‌های هوش مصنوعی
from trading_brain_ai import TradingBrainAI
from ml_signal_integration import MLSignalIntegration
# تنظیم لاگر
logger = logging.getLogger(__name__)

# --- >> هندلر سیگنال‌های سیستم برای خروج تمیز (اصلاح شده) << ---
def signal_handler(sig, frame):
    """هندلر سیگنال برای خروج تمیز ربات."""
    global bot_instance
    print(f"\nدریافت سیگنال {sig}، در حال شروع به خاموش کردن...")
    logger.info(f"سیگنال {sig} دریافت شد، در حال خاموش کردن...")

    # درخواست توقف ربات (اگر وجود دارد و متد stop دارد)
    if bot_instance and hasattr(bot_instance, 'stop') and callable(bot_instance.stop):
        try:
            bot_instance.stop()
            logger.info("درخواست توقف به ربات ارسال شد.")
        except Exception as e:
            logger.error(f"خطا در متوقف کردن ربات: {e}")


class BotPerformanceTracker:
    """
    کلاس ردیابی و تحلیل عملکرد ربات معاملاتی
    """

    def __init__(self, config: Dict[str, Any], db_path: Optional[str] = None):
        """مقداردهی اولیه با تنظیمات"""
        self.config = config.get('performance_tracking', {})
        self.enabled = self.config.get('enabled', True)
        self.metrics_file = self.config.get('metrics_file', 'data/performance_metrics.json')
        self.max_history = self.config.get('max_history_days', 90)
        self.db_path = db_path

        # متریک‌های عملکرد
        self.daily_metrics: Dict[str, Dict[str, Any]] = {}
        self.current_day_metrics: Dict[str, Any] = {
            'date': datetime.now().strftime('%Y-%m-%d'),
            'signals_generated': 0,
            'trades_executed': 0,
            'successful_trades': 0,
            'failed_trades': 0,
            'total_profit_usdt': 0.0,
            'total_loss_usdt': 0.0,
            'total_profit_percentage': 0.0,
            'max_drawdown': 0.0,
            'win_rate': 0.0,
            'profit_factor': 0.0,
            'avg_return_per_trade': 0.0,
            'avg_trade_duration': 0.0,
            'errors': 0,
            'warnings': 0,
            'timeframe_performance': {},
            'symbol_performance': {},
            'strategy_performance': {}
        }

        # آمار سیستم
        self.system_stats: Dict[str, Any] = {
            'start_time': datetime.now().isoformat(),
            'uptime_seconds': 0,
            'restart_count': 0,
            'error_count': 0,
            'warning_count': 0,
            'last_restart': None,
            'last_error': None,
            'cpu_usage_avg': 0.0,
            'memory_usage_avg': 0.0
        }

        # بارگیری داده‌های موجود
        self._load_metrics()

        logger.info(f"سیستم ردیابی عملکرد راه‌اندازی شد: فعال={self.enabled}")

    def _load_metrics(self) -> None:
        """بارگیری متریک‌ها از فایل"""
        if not self.enabled:
            return

        try:
            if os.path.exists(self.metrics_file):
                with open(self.metrics_file, 'r') as f:
                    data = json.load(f)
                    self.daily_metrics = data.get('daily_metrics', {})
                    self.system_stats = data.get('system_stats', self.system_stats)

                    # به‌روزرسانی restart_count
                    self.system_stats['restart_count'] += 1
                    self.system_stats['last_restart'] = datetime.now().isoformat()

                    # حذف داده‌های قدیمی‌تر از max_history
                    cutoff_date = (datetime.now() - timedelta(days=self.max_history)).strftime('%Y-%m-%d')
                    self.daily_metrics = {
                        date: metrics for date, metrics in self.daily_metrics.items()
                        if date >= cutoff_date
                    }

                logger.info(f"متریک‌های عملکرد از {self.metrics_file} بارگذاری شد")
        except Exception as e:
            logger.error(f"خطا در بارگذاری متریک‌های عملکرد: {e}", exc_info=True)

    def save_metrics(self) -> None:
        """ذخیره متریک‌ها در فایل"""
        if not self.enabled:
            return

        try:
            # اطمینان از وجود دایرکتوری
            os.makedirs(os.path.dirname(os.path.abspath(self.metrics_file)), exist_ok=True)

            # به‌روزرسانی آمار سیستم
            self.system_stats['uptime_seconds'] = (datetime.now() - datetime.fromisoformat(self.system_stats['start_time'])).total_seconds()

            # به‌روزرسانی متریک‌های روز جاری
            current_date = datetime.now().strftime('%Y-%m-%d')
            self.current_day_metrics['date'] = current_date

            # محاسبه نسبت‌ها
            if self.current_day_metrics['trades_executed'] > 0:
                self.current_day_metrics['win_rate'] = self.current_day_metrics['successful_trades'] / self.current_day_metrics['trades_executed']
                self.current_day_metrics['avg_return_per_trade'] = (self.current_day_metrics['total_profit_percentage'] - self.current_day_metrics['total_loss_usdt']) / self.current_day_metrics['trades_executed']

            if self.current_day_metrics['total_loss_usdt'] > 0:
                self.current_day_metrics['profit_factor'] = self.current_day_metrics['total_profit_usdt'] / self.current_day_metrics['total_loss_usdt']

            # ذخیره متریک‌های روز جاری
            self.daily_metrics[current_date] = self.current_day_metrics

            # ساخت دیکشنری برای ذخیره
            data = {
                'daily_metrics': self.daily_metrics,
                'system_stats': self.system_stats,
                'last_updated': datetime.now().isoformat()
            }

            # ذخیره در فایل
            with open(self.metrics_file, 'w') as f:
                json.dump(data, f, indent=2)
        except Exception as e:
            logger.error(f"خطا در ذخیره متریک‌های عملکرد: {e}", exc_info=True)

    def record_signal_generated(self, signal: SignalInfo) -> None:
        """ثبت تولید سیگنال جدید"""
        if not self.enabled:
            return

        try:
            self.current_day_metrics['signals_generated'] += 1

            # به‌روزرسانی آمار تایم‌فریم
            timeframe = signal.timeframe
            if timeframe not in self.current_day_metrics['timeframe_performance']:
                self.current_day_metrics['timeframe_performance'][timeframe] = {
                    'signals': 0, 'trades': 0, 'successful_trades': 0
                }
            self.current_day_metrics['timeframe_performance'][timeframe]['signals'] += 1

            # به‌روزرسانی آمار نماد
            symbol = signal.symbol
            if symbol not in self.current_day_metrics['symbol_performance']:
                self.current_day_metrics['symbol_performance'][symbol] = {
                    'signals': 0, 'trades': 0, 'successful_trades': 0, 'profit_usdt': 0.0
                }
            self.current_day_metrics['symbol_performance'][symbol]['signals'] += 1

            # به‌روزرسانی آمار استراتژی
            strategy = signal.signal_type
            if strategy not in self.current_day_metrics['strategy_performance']:
                self.current_day_metrics['strategy_performance'][strategy] = {
                    'signals': 0, 'trades': 0, 'successful_trades': 0, 'profit_usdt': 0.0
                }
            self.current_day_metrics['strategy_performance'][strategy]['signals'] += 1

            # ذخیره خودکار هر 10 سیگنال
            if self.current_day_metrics['signals_generated'] % 10 == 0:
                self.save_metrics()
        except Exception as e:
            logger.error(f"خطا در ثبت سیگنال: {e}", exc_info=True)

    def record_trade_executed(self, trade: Trade) -> None:
        """ثبت معامله اجرا شده"""
        if not self.enabled:
            return

        try:
            self.current_day_metrics['trades_executed'] += 1

            # به‌روزرسانی آمار نماد
            symbol = trade.symbol
            if symbol not in self.current_day_metrics['symbol_performance']:
                self.current_day_metrics['symbol_performance'][symbol] = {
                    'signals': 0, 'trades': 0, 'successful_trades': 0, 'profit_usdt': 0.0
                }
            self.current_day_metrics['symbol_performance'][symbol]['trades'] += 1

            # ذخیره خودکار هر 5 معامله
            if self.current_day_metrics['trades_executed'] % 5 == 0:
                self.save_metrics()
        except Exception as e:
            logger.error(f"خطا در ثبت اجرای معامله: {e}", exc_info=True)

    def record_trade_result(self, trade: Trade, trade_result: TradeResult) -> None:
        """ثبت نتیجه معامله"""
        if not self.enabled:
            return

        try:
            profit_usdt = trade_result.profit_pct * trade.volume
            is_successful = trade_result.profit_pct > 0

            # به‌روزرسانی آمار کلی
            if is_successful:
                self.current_day_metrics['successful_trades'] += 1
                self.current_day_metrics['total_profit_usdt'] += profit_usdt
                self.current_day_metrics['total_profit_percentage'] += trade_result.profit_pct
            else:
                self.current_day_metrics['failed_trades'] += 1
                self.current_day_metrics['total_loss_usdt'] += abs(profit_usdt)

            # محاسبه میانگین مدت زمان معامله
            if trade_result.trade_duration:
                duration_seconds = trade_result.trade_duration.total_seconds()
                current_avg_duration = self.current_day_metrics['avg_trade_duration']
                trades_count = self.current_day_metrics['trades_executed']

                # به‌روزرسانی میانگین
                self.current_day_metrics['avg_trade_duration'] = (
                    (current_avg_duration * (trades_count - 1)) + duration_seconds
                ) / trades_count

            # به‌روزرسانی آمار نماد
            symbol = trade.symbol
            if symbol in self.current_day_metrics['symbol_performance']:
                symbol_stats = self.current_day_metrics['symbol_performance'][symbol]
                symbol_stats['profit_usdt'] += profit_usdt
                if is_successful:
                    symbol_stats['successful_trades'] += 1

            # به‌روزرسانی آمار استراتژی و تایم‌فریم
            if hasattr(trade, 'signal_id') and trade.signal_id:
                # فرض می‌کنیم اطلاعات استراتژی و تایم‌فریم از trade_result قابل دسترسی هستند
                strategy = trade_result.signal_type
                timeframe = trade_result.timeframe

                if strategy in self.current_day_metrics['strategy_performance']:
                    strategy_stats = self.current_day_metrics['strategy_performance'][strategy]
                    strategy_stats['trades'] += 1
                    strategy_stats['profit_usdt'] += profit_usdt
                    if is_successful:
                        strategy_stats['successful_trades'] += 1

                if timeframe in self.current_day_metrics['timeframe_performance']:
                    timeframe_stats = self.current_day_metrics['timeframe_performance'][timeframe]
                    timeframe_stats['trades'] += 1
                    if is_successful:
                        timeframe_stats['successful_trades'] += 1

            # ذخیره بعد از هر نتیجه معامله
            self.save_metrics()
        except Exception as e:
            logger.error(f"خطا در ثبت نتیجه معامله: {e}", exc_info=True)

    def record_error(self, error_message: str, severity: str = 'error') -> None:
        """ثبت خطا"""
        if not self.enabled:
            return

        try:
            if severity == 'error':
                self.system_stats['error_count'] += 1
                self.current_day_metrics['errors'] += 1
                self.system_stats['last_error'] = {
                    'message': error_message,
                    'timestamp': datetime.now().isoformat()
                }
            elif severity == 'warning':
                self.system_stats['warning_count'] += 1
                self.current_day_metrics['warnings'] += 1

            # ذخیره بعد از هر خطای جدی
            if severity == 'error':
                self.save_metrics()
        except Exception as e:
            logger.error(f"خطا در ثبت رویداد خطا: {e}", exc_info=True)

    def update_system_stats(self, cpu_usage: float, memory_usage: float) -> None:
        """به‌روزرسانی آمار سیستم"""
        if not self.enabled:
            return

        try:
            # به‌روزرسانی میانگین
            current_cpu_avg = self.system_stats['cpu_usage_avg']
            current_mem_avg = self.system_stats['memory_usage_avg']

            # محاسبه میانگین وزنی برای صاف‌سازی تغییرات
            alpha = 0.1  # ضریب وزن‌دهی نمونه جدید
            self.system_stats['cpu_usage_avg'] = (current_cpu_avg * (1 - alpha)) + (cpu_usage * alpha)
            self.system_stats['memory_usage_avg'] = (current_mem_avg * (1 - alpha)) + (memory_usage * alpha)

            # به‌روزرسانی زمان کارکرد
            self.system_stats['uptime_seconds'] = (datetime.now() - datetime.fromisoformat(self.system_stats['start_time'])).total_seconds()
        except Exception as e:
            logger.error(f"خطا در به‌روزرسانی آمار سیستم: {e}", exc_info=True)

    def get_performance_report(self) -> Dict[str, Any]:
        """تولید گزارش عملکرد"""
        if not self.enabled:
            return {'enabled': False}

        try:
            # به‌روزرسانی کلی آمار
            self.system_stats['uptime_seconds'] = (datetime.now() - datetime.fromisoformat(self.system_stats['start_time'])).total_seconds()

            # گزارش روزانه
            current_date = datetime.now().strftime('%Y-%m-%d')
            daily_report = self.current_day_metrics.copy()

            # گزارش کلی
            overall_report = {}

            # محاسبه آمار کلی از daily_metrics
            total_trades = sum(day['trades_executed'] for day in self.daily_metrics.values())
            total_successful = sum(day['successful_trades'] for day in self.daily_metrics.values())
            total_profit = sum(day['total_profit_usdt'] for day in self.daily_metrics.values())
            total_loss = sum(day['total_loss_usdt'] for day in self.daily_metrics.values())

            overall_report['total_days'] = len(self.daily_metrics)
            overall_report['total_trades'] = total_trades
            overall_report['win_rate'] = total_successful / total_trades if total_trades > 0 else 0
            overall_report['total_profit_usdt'] = total_profit
            overall_report['total_loss_usdt'] = total_loss
            overall_report['net_profit_usdt'] = total_profit - total_loss
            overall_report['profit_factor'] = total_profit / total_loss if total_loss > 0 else 0

            return {
                'enabled': self.enabled,
                'daily': daily_report,
                'overall': overall_report,
                'system': self.system_stats
            }
        except Exception as e:
            logger.error(f"خطا در تولید گزارش عملکرد: {e}", exc_info=True)
            return {'enabled': self.enabled, 'error': str(e)}

    def update_config(self, new_config: Dict[str, Any]) -> None:
        """به‌روزرسانی تنظیمات در زمان اجرا"""
        try:
            if 'performance_tracking' in new_config:
                old_config = self.config.copy()
                self.config = new_config.get('performance_tracking', {})

                # به‌روزرسانی متغیرهای کلیدی
                self.enabled = self.config.get('enabled', True)
                self.metrics_file = self.config.get('metrics_file', 'data/performance_metrics.json')
                self.max_history = self.config.get('max_history_days', 90)

                # لاگ تغییرات
                logger.info(f"تنظیمات ردیابی عملکرد به‌روزرسانی شد - فعال: {self.enabled}")

                # اگر تغییری در مسیر فایل داده شده، داده‌ها را ذخیره و بارگذاری مجدد کنیم
                if old_config.get('metrics_file') != self.metrics_file:
                    logger.info(f"مسیر فایل متریک‌ها تغییر کرده. ذخیره در مسیر جدید: {self.metrics_file}")
                    self.save_metrics()
        except Exception as e:
            logger.error(f"خطا در به‌روزرسانی تنظیمات ردیابی عملکرد: {e}", exc_info=True)


class BackupManager:
    """
    کلاس مدیریت پشتیبان‌گیری خودکار از داده‌های حساس
    """

    def __init__(self, config: Dict[str, Any]):
        """مقداردهی اولیه با تنظیمات"""
        self.config = config.get('backup', {})
        self.enabled = self.config.get('enabled', True)
        self.backup_dir = self.config.get('directory', 'data/backups')
        self.interval_hours = self.config.get('interval_hours', 24)
        self.keep_days = self.config.get('keep_days', 7)
        self.files_to_backup = self.config.get('files', [])
        self.compression = self.config.get('compression', True)

        self._last_backup = None
        self._backup_task = None

        # مسیرهای داده‌های حساس
        self.critical_files = self.config.get('critical_files', [])

        # ایجاد دایرکتوری پشتیبان
        os.makedirs(self.backup_dir, exist_ok=True)

        logger.info(f"مدیریت پشتیبان‌گیری راه‌اندازی شد: فعال={self.enabled}, بازه زمانی={self.interval_hours}h")

    async def start_automated_backup(self) -> None:
        """شروع فرآیند پشتیبان‌گیری خودکار"""
        if not self.enabled:
            logger.info("پشتیبان‌گیری خودکار غیرفعال است")
            return

        # لغو تسک قبلی در صورت وجود
        if self._backup_task and not self._backup_task.done():
            self._backup_task.cancel()

        # ایجاد تسک جدید
        self._backup_task = asyncio.create_task(self._backup_loop())
        logger.info(f"پشتیبان‌گیری خودکار هر {self.interval_hours} ساعت شروع شد")

    async def _backup_loop(self) -> None:
        """حلقه اصلی پشتیبان‌گیری خودکار"""
        try:
            while True:
                # انجام پشتیبان‌گیری
                await self.create_backup()

                # پاکسازی پشتیبان‌های قدیمی
                self._cleanup_old_backups()

                # انتظار تا پشتیبان‌گیری بعدی
                await asyncio.sleep(self.interval_hours * 3600)
        except asyncio.CancelledError:
            logger.info("حلقه پشتیبان‌گیری لغو شد")
        except Exception as e:
            logger.error(f"خطا در حلقه پشتیبان‌گیری: {e}", exc_info=True)

    async def create_backup(self) -> Optional[str]:
        """ایجاد فایل پشتیبان"""
        if not self.enabled:
            return None

        try:
            # ایجاد نام فایل با timestamp
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            backup_filename = f"backup_{timestamp}.zip" if self.compression else f"backup_{timestamp}"
            backup_path = os.path.join(self.backup_dir, backup_filename)

            # جمع‌آوری فایل‌های پشتیبان‌گیری
            files_to_backup = self._get_files_to_backup()

            if not files_to_backup:
                logger.warning("فایلی برای پشتیبان‌گیری یافت نشد")
                return None

            # ایجاد فایل پشتیبان
            if self.compression:
                import zipfile
                with zipfile.ZipFile(backup_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
                    for file_path in files_to_backup:
                        if os.path.exists(file_path):
                            # ذخیره با حفظ ساختار نسبی دایرکتوری
                            arcname = os.path.basename(file_path)
                            zipf.write(file_path, arcname)
            else:
                # ایجاد دایرکتوری پشتیبان
                os.makedirs(backup_path, exist_ok=True)

                # کپی فایل‌ها
                import shutil
                for file_path in files_to_backup:
                    if os.path.exists(file_path):
                        dst = os.path.join(backup_path, os.path.basename(file_path))
                        shutil.copy2(file_path, dst)

            self._last_backup = datetime.now()
            logger.info(f"پشتیبان در مسیر {backup_path} ایجاد شد")
            return backup_path
        except Exception as e:
            logger.error(f"خطا در ایجاد پشتیبان: {e}", exc_info=True)
            return None

    def _get_files_to_backup(self) -> List[str]:
        """جمع‌آوری لیست فایل‌های پشتیبان‌گیری"""
        files = []

        # فایل‌های مشخص شده
        files.extend(self.files_to_backup)

        # فایل‌های حساس
        files.extend(self.critical_files)

        # فایل تنظیمات
        config_paths = ['config.yaml', 'config.json', 'configs/config.yaml', 'configs/config.json']
        for path in config_paths:
            if os.path.exists(path):
                files.append(path)

        # دیتابیس‌ها
        db_paths = ['data/trades.db', 'data/signals.db', 'trades.db', 'signals.db']
        for path in db_paths:
            if os.path.exists(path):
                files.append(path)

        # فایل‌های عملکرد
        performance_files = ['data/performance_metrics.json', 'data/system_metrics.json']
        for path in performance_files:
            if os.path.exists(path):
                files.append(path)

        # حذف موارد تکراری
        return list(set(files))

    def _cleanup_old_backups(self) -> None:
        """پاکسازی پشتیبان‌های قدیمی"""
        try:
            # محاسبه تاریخ cut-off
            cutoff_date = datetime.now() - timedelta(days=self.keep_days)

            # جستجوی فایل‌های پشتیبان
            for filename in os.listdir(self.backup_dir):
                if filename.startswith('backup_'):
                    file_path = os.path.join(self.backup_dir, filename)

                    # بررسی تاریخ فایل
                    try:
                        date_part = filename.split('_')[1][:8]  # استخراج YYYYMMDD
                        file_date = datetime.strptime(date_part, '%Y%m%d')

                        if file_date < cutoff_date:
                            os.remove(file_path)
                    except (ValueError, IndexError):
                        # فرمت نام فایل نامعتبر است
                        continue
        except Exception as e:
            logger.error(f"خطا در پاکسازی پشتیبان‌های قدیمی: {e}", exc_info=True)

    def stop(self) -> None:
        """توقف فرآیند پشتیبان‌گیری خودکار"""
        if self._backup_task and not self._backup_task.done():
            self._backup_task.cancel()
            logger.info("پشتیبان‌گیری خودکار متوقف شد")

    def update_config(self, new_config: Dict[str, Any]) -> None:
        """به‌روزرسانی تنظیمات در زمان اجرا"""
        try:
            if 'backup' in new_config:
                old_config = self.config.copy()
                self.config = new_config.get('backup', {})

                # به‌روزرسانی متغیرهای کلیدی
                old_enabled = self.enabled
                self.enabled = self.config.get('enabled', True)
                old_interval = self.interval_hours
                self.interval_hours = self.config.get('interval_hours', 24)
                self.keep_days = self.config.get('keep_days', 7)
                self.files_to_backup = self.config.get('files', [])
                self.compression = self.config.get('compression', True)
                self.critical_files = self.config.get('critical_files', [])

                # بررسی تغییر مسیر پشتیبان
                new_backup_dir = self.config.get('directory', 'data/backups')
                if new_backup_dir != self.backup_dir:
                    logger.info(f"مسیر پشتیبان‌گیری تغییر کرد از '{self.backup_dir}' به '{new_backup_dir}'")
                    self.backup_dir = new_backup_dir
                    os.makedirs(self.backup_dir, exist_ok=True)

                # اگر وضعیت فعال بودن تغییر کرده، اقدام لازم را انجام دهیم
                if old_enabled != self.enabled:
                    if self.enabled:
                        logger.info("پشتیبان‌گیری فعال شد - راه‌اندازی مجدد فرآیند")
                        asyncio.create_task(self.start_automated_backup())
                    else:
                        logger.info("پشتیبان‌گیری غیرفعال شد - توقف فرآیند")
                        self.stop()

                # اگر بازه زمانی تغییر کرده و فعال است، حلقه پشتیبان‌گیری را مجدداً راه‌اندازی کنیم
                elif self.enabled and old_interval != self.interval_hours:
                    logger.info(f"بازه زمانی پشتیبان‌گیری تغییر کرد به {self.interval_hours} ساعت - راه‌اندازی مجدد فرآیند")
                    asyncio.create_task(self.start_automated_backup())

                logger.info(f"تنظیمات پشتیبان‌گیری به‌روزرسانی شد - فعال: {self.enabled}, بازه: {self.interval_hours}h")
        except Exception as e:
            logger.error(f"خطا در به‌روزرسانی تنظیمات پشتیبان‌گیری: {e}", exc_info=True)


class TradingStrategyManager:
    """
    کلاس مدیریت استراتژی‌های معاملاتی با قابلیت تغییر دینامیک
    """

    def __init__(self, config: Dict[str, Any]):
        """مقداردهی اولیه با تنظیمات"""
        self.config = config.get('strategy_management', {})
        self.enabled = self.config.get('enabled', True)
        self.strategies_dir = self.config.get('strategies_dir', 'strategies')
        self.default_strategy = self.config.get('default_strategy', 'default')
        self.auto_rotation = self.config.get('auto_rotation', False)
        self.rotation_interval_hours = self.config.get('rotation_interval_hours', 24)

        # استراتژی‌های فعال و وضعیت آنها
        self.active_strategies: Dict[str, Dict[str, Any]] = {}

        # مجموعه پارامترهای پیکربندی برای هر استراتژی
        self.strategy_configs: Dict[str, Dict[str, Any]] = {}

        # وضعیت کنونی
        self.current_strategy = self.default_strategy

        # تسک چرخش خودکار
        self._rotation_task = None

        logger.info(f"مدیریت استراتژی راه‌اندازی شد: فعال={self.enabled}, پیش‌فرض={self.default_strategy}")

    async def initialize(self) -> None:
        """مقداردهی اولیه و بارگیری استراتژی‌ها"""
        if not self.enabled:
            return

        # بررسی و ایجاد دایرکتوری استراتژی‌ها
        os.makedirs(self.strategies_dir, exist_ok=True)

        # بارگیری استراتژی‌های موجود
        await self._load_strategies()

        # شروع چرخش خودکار اگر فعال است
        if self.auto_rotation:
            await self.start_auto_rotation()

    async def _load_strategies(self) -> None:
        """بارگیری استراتژی‌های موجود از دایرکتوری"""
        try:
            # جستجوی فایل‌های استراتژی
            for filename in os.listdir(self.strategies_dir):
                if filename.endswith('.yaml') or filename.endswith('.json'):
                    strategy_name = os.path.splitext(filename)[0]
                    strategy_path = os.path.join(self.strategies_dir, filename)

                    # بارگیری فایل استراتژی
                    with open(strategy_path, 'r', encoding='utf-8') as f:
                        if filename.endswith('.yaml'):
                            strategy_config = yaml.safe_load(f)
                        else:  # .json
                            strategy_config = json.load(f)

                    # ذخیره پیکربندی استراتژی
                    self.strategy_configs[strategy_name] = strategy_config

                    # ایجاد وضعیت استراتژی
                    self.active_strategies[strategy_name] = {
                        'name': strategy_name,
                        'file': filename,
                        'path': strategy_path,
                        'last_used': None,
                        'trades_executed': 0,
                        'successful_trades': 0,
                        'win_rate': 0.0,
                        'profit_factor': 0.0,
                        'enabled': strategy_config.get('enabled', True)
                    }

            logger.info(f"{len(self.active_strategies)} استراتژی بارگذاری شد")

            # اطمینان از وجود استراتژی پیش‌فرض
            if self.default_strategy not in self.active_strategies:
                logger.warning(f"استراتژی پیش‌فرض '{self.default_strategy}' یافت نشد. ایجاد استراتژی پیش‌فرض.")
                self._create_default_strategy()

        except Exception as e:
            logger.error(f"خطا در بارگذاری استراتژی‌ها: {e}", exc_info=True)
            # ایجاد استراتژی پیش‌فرض در صورت خطا
            self._create_default_strategy()

    def _create_default_strategy(self) -> None:
        """ایجاد استراتژی پیش‌فرض"""
        try:
            # ایجاد استراتژی پیش‌فرض
            default_config = {
                'name': 'default',
                'description': 'استراتژی معاملاتی پیش‌فرض با تنظیمات محافظه‌کارانه',
                'enabled': True,
                'risk_management': {
                    'max_risk_per_trade_percent': 1.0,
                    'min_risk_reward_ratio': 2.0,
                    'preferred_risk_reward_ratio': 2.5,
                    'default_stop_loss_percent': 1.5
                },
                'signal_generation': {
                    'minimum_signal_score': 33,
                    'timeframes': ['5m', '15m', '1h', '4h'],
                    'timeframe_weights': {'5m': 0.7, '15m': 0.85, '1h': 1.0, '4h': 1.2}
                },
                'filters': {
                    'min_volume_usdt': 1000000,
                    'max_spread_percent': 0.5
                }
            }

            # ذخیره در دیکشنری
            self.strategy_configs[self.default_strategy] = default_config

            # ایجاد وضعیت استراتژی
            self.active_strategies[self.default_strategy] = {
                'name': self.default_strategy,
                'file': f"{self.default_strategy}.yaml",
                'path': os.path.join(self.strategies_dir, f"{self.default_strategy}.yaml"),
                'last_used': None,
                'trades_executed': 0,
                'successful_trades': 0,
                'win_rate': 0.0,
                'profit_factor': 0.0,
                'enabled': True
            }

            # ذخیره در فایل
            os.makedirs(self.strategies_dir, exist_ok=True)
            with open(os.path.join(self.strategies_dir, f"{self.default_strategy}.yaml"), 'w', encoding='utf-8') as f:
                yaml.dump(default_config, f, sort_keys=False, default_flow_style=False)

            logger.info(f"استراتژی پیش‌فرض ایجاد شد: {self.default_strategy}")
        except Exception as e:
            logger.error(f"خطا در ایجاد استراتژی پیش‌فرض: {e}", exc_info=True)

    async def switch_strategy(self, strategy_name: str) -> bool:
        """تغییر استراتژی فعال"""
        if not self.enabled:
            return False

        if strategy_name not in self.active_strategies:
            logger.error(f"استراتژی '{strategy_name}' یافت نشد")
            return False

        if not self.active_strategies[strategy_name]['enabled']:
            logger.error(f"استراتژی '{strategy_name}' غیرفعال است")
            return False

        logger.info(f"تغییر استراتژی از '{self.current_strategy}' به '{strategy_name}'")
        self.current_strategy = strategy_name
        self.active_strategies[strategy_name]['last_used'] = datetime.now()

        return True

    def get_current_strategy_config(self) -> Dict[str, Any]:
        """دریافت پیکربندی استراتژی فعال"""
        if not self.enabled or self.current_strategy not in self.strategy_configs:
            # استفاده از پیکربندی پیش‌فرض
            return self.config

        # ترکیب پیکربندی پایه با پیکربندی استراتژی
        base_config = self.config.copy()
        strategy_config = self.strategy_configs[self.current_strategy]

        # ادغام بخش‌های مربوطه
        if 'risk_management' in strategy_config:
            base_config['risk_management'] = {**base_config.get('risk_management', {}), **strategy_config['risk_management']}

        if 'signal_generation' in strategy_config:
            base_config['signal_generation'] = {**base_config.get('signal_generation', {}), **strategy_config['signal_generation']}

        if 'filters' in strategy_config:
            base_config['filters'] = {**base_config.get('filters', {}), **strategy_config['filters']}

        if 'exchange' in strategy_config:
            # فقط بخش‌های غیرحساس را ادغام می‌کنیم
            exchange_config = strategy_config['exchange']
            if 'symbols' in exchange_config:
                base_config['exchange']['symbols'] = exchange_config['symbols']
            if 'trading_pairs' in exchange_config:
                base_config['exchange']['trading_pairs'] = exchange_config['trading_pairs']

        return base_config

    async def start_auto_rotation(self) -> None:
        """شروع چرخش خودکار استراتژی‌ها"""
        if not self.enabled or not self.auto_rotation:
            return

        # لغو تسک قبلی در صورت وجود
        if self._rotation_task and not self._rotation_task.done():
            self._rotation_task.cancel()

        # ایجاد تسک جدید
        self._rotation_task = asyncio.create_task(self._rotation_loop())
        logger.info(f"چرخش خودکار استراتژی‌ها هر {self.rotation_interval_hours} ساعت آغاز شد")

    async def _rotation_loop(self) -> None:
        """حلقه اصلی چرخش خودکار استراتژی‌ها"""
        try:
            while True:
                # انتخاب استراتژی بعدی
                next_strategy = self._select_next_strategy()

                if next_strategy and next_strategy != self.current_strategy:
                    # تغییر استراتژی
                    await self.switch_strategy(next_strategy)

                # انتظار تا چرخش بعدی
                await asyncio.sleep(self.rotation_interval_hours * 3600)
        except asyncio.CancelledError:
            logger.info("حلقه چرخش استراتژی لغو شد")
        except Exception as e:
            logger.error(f"خطا در حلقه چرخش استراتژی: {e}", exc_info=True)

    def _select_next_strategy(self) -> Optional[str]:
        """انتخاب استراتژی بعدی برای چرخش"""
        # فیلتر استراتژی‌های فعال
        enabled_strategies = [
            name for name, info in self.active_strategies.items()
            if info['enabled']
        ]

        if not enabled_strategies:
            return None

        # استراتژی با بهترین عملکرد (قابل توسعه با الگوریتم‌های پیچیده‌تر)
        best_strategy = None
        best_score = -1

        for name in enabled_strategies:
            info = self.active_strategies[name]

            # محاسبه امتیاز ساده
            score = (info['win_rate'] * 0.5) + (info['profit_factor'] * 0.5)

            if score > best_score:
                best_score = score
                best_strategy = name

        # اگر هیچ استراتژی‌ای امتیاز معتبر نداشت، انتخاب تصادفی
        if best_score <= 0:
            import random
            return random.choice(enabled_strategies)

        return best_strategy

    def record_trade_result(self, strategy_name: str, success: bool, profit_factor: float) -> None:
        """ثبت نتیجه معامله برای یک استراتژی"""
        if not self.enabled or strategy_name not in self.active_strategies:
            return

        strategy = self.active_strategies[strategy_name]
        strategy['trades_executed'] += 1

        if success:
            strategy['successful_trades'] += 1

        # به‌روزرسانی نرخ موفقیت
        if strategy['trades_executed'] > 0:
            strategy['win_rate'] = strategy['successful_trades'] / strategy['trades_executed']

        # به‌روزرسانی فاکتور سود
        strategy['profit_factor'] = (strategy['profit_factor'] * (strategy['trades_executed'] - 1) + profit_factor) / strategy['trades_executed']

    def stop(self) -> None:
        """توقف فرآیند چرخش خودکار"""
        if self._rotation_task and not self._rotation_task.done():
            self._rotation_task.cancel()
            logger.info("چرخش خودکار استراتژی متوقف شد")

    def update_config(self, new_config: Dict[str, Any]) -> None:
        """به‌روزرسانی تنظیمات در زمان اجرا"""
        try:
            if 'strategy_management' in new_config:
                old_config = self.config.copy()
                self.config = new_config.get('strategy_management', {})

                # به‌روزرسانی متغیرهای کلیدی
                old_enabled = self.enabled
                self.enabled = self.config.get('enabled', True)
                old_auto_rotation = self.auto_rotation
                self.auto_rotation = self.config.get('auto_rotation', False)
                old_rotation_interval = self.rotation_interval_hours
                self.rotation_interval_hours = self.config.get('rotation_interval_hours', 24)
                old_default_strategy = self.default_strategy
                new_default_strategy = self.config.get('default_strategy', 'default')

                # بررسی تغییر مسیر استراتژی‌ها
                new_strategies_dir = self.config.get('strategies_dir', 'strategies')
                if new_strategies_dir != self.strategies_dir:
                    logger.info(f"مسیر استراتژی‌ها تغییر کرد از '{self.strategies_dir}' به '{new_strategies_dir}'")
                    self.strategies_dir = new_strategies_dir
                    os.makedirs(self.strategies_dir, exist_ok=True)
                    # بارگذاری مجدد استراتژی‌ها
                    asyncio.create_task(self._load_strategies())

                # اگر استراتژی پیش‌فرض تغییر کرده
                if old_default_strategy != new_default_strategy:
                    logger.info(f"استراتژی پیش‌فرض تغییر کرد از '{old_default_strategy}' به '{new_default_strategy}'")
                    self.default_strategy = new_default_strategy
                    # اطمینان از وجود استراتژی پیش‌فرض
                    if self.default_strategy not in self.active_strategies:
                        logger.warning(f"استراتژی پیش‌فرض جدید '{self.default_strategy}' یافت نشد. ایجاد استراتژی پیش‌فرض.")
                        self._create_default_strategy()

                # اگر وضعیت فعال بودن تغییر کرده
                if old_enabled != self.enabled:
                    if self.enabled:
                        logger.info("مدیریت استراتژی فعال شد")
                        asyncio.create_task(self.initialize())
                    else:
                        logger.info("مدیریت استراتژی غیرفعال شد - توقف فرآیند")
                        self.stop()

                # اگر وضعیت چرخش خودکار تغییر کرده
                if old_auto_rotation != self.auto_rotation:
                    if self.auto_rotation and self.enabled:
                        logger.info("چرخش خودکار استراتژی فعال شد - راه‌اندازی مجدد فرآیند")
                        asyncio.create_task(self.start_auto_rotation())
                    elif not self.auto_rotation:
                        logger.info("چرخش خودکار استراتژی غیرفعال شد - توقف فرآیند")
                        self.stop()

                # اگر بازه زمانی تغییر کرده و چرخش خودکار فعال است
                elif self.auto_rotation and self.enabled and old_rotation_interval != self.rotation_interval_hours:
                    logger.info(f"بازه زمانی چرخش استراتژی تغییر کرد به {self.rotation_interval_hours} ساعت - راه‌اندازی مجدد فرآیند")
                    self.stop()
                    asyncio.create_task(self.start_auto_rotation())

                logger.info(f"تنظیمات مدیریت استراتژی به‌روزرسانی شد - فعال: {self.enabled}, چرخش خودکار: {self.auto_rotation}")
        except Exception as e:
            logger.error(f"خطا در به‌روزرسانی تنظیمات مدیریت استراتژی: {e}", exc_info=True)

    def reload_strategies(self) -> None:
        """بارگذاری مجدد فایل‌های استراتژی از دیسک"""
        try:
            # ذخیره نام استراتژی فعلی
            current = self.current_strategy

            # پاکسازی داده‌های فعلی
            self.active_strategies = {}
            self.strategy_configs = {}

            # بارگذاری مجدد
            asyncio.create_task(self._load_strategies())

            # برگرداندن استراتژی فعلی اگر هنوز وجود دارد
            if current in self.active_strategies:
                self.current_strategy = current
            else:
                self.current_strategy = self.default_strategy

            logger.info(f"استراتژی‌ها مجدداً بارگذاری شدند. استراتژی فعال: {self.current_strategy}")
        except Exception as e:
            logger.error(f"خطا در بارگذاری مجدد استراتژی‌ها: {e}", exc_info=True)


class ConfigurationManager:
    """
    کلاس مدیریت تنظیمات ربات با قابلیت مشاهده تغییرات و به‌روزرسانی پویا
    """

    def __init__(self, config: Dict[str, Any], config_path: str):
        """مقداردهی اولیه با تنظیمات"""
        self.config = config
        self.config_path = config_path
        self.last_modified_time = 0
        self.listeners = []  # لیست توابعی که باید بعد از تغییر تنظیمات فراخوانی شوند
        self.modified_sections = set()  # مجموعه بخش‌هایی که تغییر کرده‌اند

        # ذخیره زمان آخرین تغییر فایل
        try:
            self.last_modified_time = os.path.getmtime(config_path)
        except Exception:
            self.last_modified_time = time.time()

        logger.info(f"مدیریت تنظیمات راه‌اندازی شد. فایل تنظیمات: {config_path}")

    def register_update_listener(self, callback: Callable[[Dict[str, Any], set], None]) -> None:
        """ثبت یک تابع برای فراخوانی در زمان به‌روزرسانی تنظیمات"""
        self.listeners.append(callback)
        logger.debug(f"یک شنونده جدید برای تغییرات تنظیمات ثبت شد. تعداد کل: {len(self.listeners)}")

    def check_for_changes(self) -> bool:
        """بررسی تغییرات در فایل تنظیمات و بارگذاری مجدد در صورت نیاز"""
        try:
            current_mtime = os.path.getmtime(self.config_path)

            if current_mtime > self.last_modified_time:
                logger.info(f"تغییر در فایل تنظیمات شناسایی شد: {self.config_path}")

                # بارگذاری تنظیمات جدید
                new_config = self._load_config(self.config_path)
                if new_config:
                    # مقایسه با تنظیمات قبلی برای تشخیص بخش‌های تغییر یافته
                    self._detect_changed_sections(new_config)

                    # به‌روزرسانی تنظیمات
                    self.config = new_config
                    self.last_modified_time = current_mtime

                    # اطلاع‌رسانی به شنوندگان
                    self._notify_listeners()

                    return True

            return False

        except Exception as e:
            logger.error(f"خطا در بررسی تغییرات فایل تنظیمات: {e}", exc_info=True)
            return False

    def _load_config(self, config_path: str) -> Optional[Dict[str, Any]]:
        """بارگذاری تنظیمات از فایل"""
        try:
            if config_path.endswith(('.yaml', '.yml')):
                with open(config_path, 'r', encoding='utf-8') as f:
                    config = yaml.safe_load(f)
            elif config_path.endswith('.json'):
                with open(config_path, 'r', encoding='utf-8') as f:
                    config = json.load(f)
            else:
                logger.error(f"فرمت فایل تنظیمات پشتیبانی نمی‌شود: {config_path}")
                return None

            return config

        except Exception as e:
            logger.error(f"خطا در بارگذاری فایل تنظیمات {config_path}: {e}", exc_info=True)
            return None

    def _detect_changed_sections(self, new_config: Dict[str, Any]) -> None:
        """تشخیص بخش‌های تغییر یافته در تنظیمات"""
        self.modified_sections = set()

        # بررسی بخش‌های مهم
        important_sections = [
            'risk_management', 'signal_generation', 'trading', 'strategy_management',
            'backup', 'performance_tracking', 'adaptive_learning', 'correlation_management',
            'circuit_breaker', 'trading_brain_ai', 'ml_signal_integration', 'pattern_scores'
        ]

        for section in important_sections:
            # اگر بخش در یکی از دو تنظیمات وجود ندارد
            if (section in self.config) != (section in new_config):
                self.modified_sections.add(section)
                continue

            # اگر هر دو تنظیمات این بخش را دارند، مقایسه محتوا
            if section in self.config and section in new_config:
                old_section = self.config[section]
                new_section = new_config[section]

                # مقایسه عمیق
                if not self._are_equal(old_section, new_section):
                    self.modified_sections.add(section)

        # لاگ کردن بخش‌های تغییر یافته
        if self.modified_sections:
            logger.info(f"بخش‌های تغییر یافته در تنظیمات: {', '.join(self.modified_sections)}")

    def _are_equal(self, a: Any, b: Any) -> bool:
        """مقایسه عمیق دو مقدار برای تشخیص برابری"""
        # مقایسه دیکشنری‌ها
        if isinstance(a, dict) and isinstance(b, dict):
            if set(a.keys()) != set(b.keys()):
                return False
            return all(self._are_equal(a[key], b[key]) for key in a)

        # مقایسه لیست‌ها
        elif isinstance(a, list) and isinstance(b, list):
            if len(a) != len(b):
                return False
            return all(self._are_equal(x, y) for x, y in zip(a, b))

        # مقایسه مقادیر ساده
        else:
            return a == b

    def _notify_listeners(self) -> None:
        """اطلاع‌رسانی به شنوندگان در مورد تغییرات تنظیمات"""
        if not self.listeners:
            logger.debug("هیچ شنونده‌ای برای تغییرات تنظیمات ثبت نشده است")
            return

        logger.debug(f"اطلاع‌رسانی به {len(self.listeners)} شنونده در مورد تغییرات تنظیمات")

        for listener in self.listeners:
            try:
                listener(self.config, self.modified_sections)
            except Exception as e:
                logger.error(f"خطا در فراخوانی شنونده تغییرات تنظیمات: {e}", exc_info=True)

    def update_section(self, section: str, section_config: Dict[str, Any]) -> bool:
        """به‌روزرسانی یک بخش خاص از تنظیمات و ذخیره آن در فایل"""
        try:
            # به‌روزرسانی تنظیمات در حافظه
            self.config[section] = section_config
            self.modified_sections = {section}

            # به‌روزرسانی فایل
            if self._save_config():
                # اطلاع‌رسانی به شنوندگان
                self._notify_listeners()
                return True
            return False

        except Exception as e:
            logger.error(f"خطا در به‌روزرسانی بخش {section} تنظیمات: {e}", exc_info=True)
            return False

    def _save_config(self) -> bool:
        """ذخیره تنظیمات در فایل"""
        try:
            # ایجاد پشتیبان از فایل فعلی
            backup_path = f"{self.config_path}.bak"
            try:
                import shutil
                shutil.copy2(self.config_path, backup_path)
                logger.debug(f"نسخه پشتیبان از تنظیمات ایجاد شد: {backup_path}")
            except Exception as e:
                logger.warning(f"خطا در ایجاد پشتیبان از فایل تنظیمات: {e}")

            # ذخیره تنظیمات جدید
            if self.config_path.endswith(('.yaml', '.yml')):
                with open(self.config_path, 'w', encoding='utf-8') as f:
                    yaml.dump(self.config, f, default_flow_style=False, allow_unicode=True)
            elif self.config_path.endswith('.json'):
                with open(self.config_path, 'w', encoding='utf-8') as f:
                    json.dump(self.config, f, indent=2, ensure_ascii=False)

            # به‌روزرسانی زمان آخرین تغییر
            self.last_modified_time = os.path.getmtime(self.config_path)

            logger.info(f"تنظیمات با موفقیت در {self.config_path} ذخیره شد")
            return True

        except Exception as e:
            logger.error(f"خطا در ذخیره تنظیمات: {e}", exc_info=True)
            return False

    def get_config(self) -> Dict[str, Any]:
        """دریافت کل تنظیمات فعلی"""
        return self.config

    def get_section(self, section: str) -> Dict[str, Any]:
        """دریافت یک بخش خاص از تنظیمات"""
        return self.config.get(section, {})


class CryptoTradingBot:
    """
    کلاس اصلی ربات معاملاتی ارزهای دیجیتال با مدیریت بهتر منابع
    - بهینه‌سازی شده با سیستم یادگیری تطبیقی
    - مدیریت همبستگی نمادها
    - مکانیزم توقف اضطراری
    - سیستم پشتیبان‌گیری خودکار
    - مدیریت استراتژی‌های قابل تعویض
    - ردیابی عملکرد پیشرفته
    """

    def __init__(self, config_path: str):
        """
        مقداردهی اولیه با فایل تنظیمات

        Args:
            config_path: مسیر فایل تنظیمات
        """
        self.config_path = config_path
        self.config = self._load_config(config_path)

        # ایجاد مدیریت تنظیمات
        self.config_manager = ConfigurationManager(self.config, config_path)

        self.trading_brain: Optional[TradingBrainAI] = None
        self.ml_integration: Optional[MLSignalIntegration] = None
        self._setup_logging()

        # وهله‌های اجزای سیستم
        self.exchange_client = None
        self.data_fetcher = None
        self.signal_generator = None
        self.signal_processor = None
        self.trade_manager = None

        # اجزای جدید
        self.performance_tracker = None
        self.backup_manager = None
        self.strategy_manager = None

        # مدیریت وضعیت اجرا
        self._shutdown_requested = asyncio.Event()
        self.db_path = self.config.get('storage', {}).get('database_path', 'data/trades.db')
        self.active_symbols = []

        # شناسه یکتا برای این اجرا
        self.instance_id = str(uuid.uuid4())

        # زمان شروع به کار
        self.start_time = time.time()

        # وضعیت اجرا
        self.running_status = {
            'state': 'initialized',
            'instance_id': self.instance_id,
            'uptime_seconds': 0,
            'start_time': datetime.now().isoformat(),
            'last_status_update': time.time(),
            'components_status': {},
            'system_info': self._get_system_info(),
            'config_changes': []  # لیست تغییرات تنظیمات که در حین اجرا رخ داده‌اند
        }

        # ثبت تابع شنونده برای تغییرات تنظیمات
        self.config_manager.register_update_listener(self._handle_config_changes)

        logger.info(f"ربات معاملاتی ارز دیجیتال با تنظیمات از {config_path} راه‌اندازی شد")
        logger.info(f"شناسه نمونه: {self.instance_id}")

        # تنظیم متغیر نمونه بات برای استفاده در هندلر سیگنال
        global bot_instance
        bot_instance = self

    def _get_system_info(self) -> Dict[str, str]:
        """جمع‌آوری اطلاعات سیستم"""
        return {
            'os': platform.system(),
            'platform': platform.platform(),
            'python_version': platform.python_version(),
            'hostname': platform.node()
        }

    def _load_config(self, config_path: str) -> Dict[str, Any]:
        """
        بارگذاری تنظیمات از فایل با پشتیبانی از فرمت‌های مختلف

        Args:
            config_path: مسیر فایل تنظیمات

        Returns:
            دیکشنری تنظیمات
        """
        try:
            # بررسی وجود فایل
            if not os.path.exists(config_path):
                raise FileNotFoundError(f"فایل تنظیمات یافت نشد: {config_path}")

            with open(config_path, 'r', encoding='utf-8') as f:
                if config_path.endswith(('.yaml', '.yml')):
                    import yaml
                    config = yaml.safe_load(f)
                elif config_path.endswith('.json'):
                    import json
                    config = json.load(f)
                else:
                    raise ValueError(f"فرمت فایل تنظیمات پشتیبانی نمی‌شود: {config_path}")

            # اعتبارسنجی اولیه تنظیمات
            required_sections = ['exchange', 'data_fetching', 'signal_processing', 'trading', 'risk_management']
            missing_sections = [section for section in required_sections if section not in config]

            if missing_sections:
                raise ValueError(f"بخش‌های تنظیمات مورد نیاز یافت نشد: {', '.join(missing_sections)}")

            # اضافه کردن بخش‌های پیش‌فرض
            self._add_default_config_sections(config)

            logger.info(f"تنظیمات با موفقیت از {config_path} بارگذاری شد")
            return config

        except FileNotFoundError:
            logger.critical(f"خطای بحرانی: فایل تنظیمات در {config_path} یافت نشد. خروج.")
            raise SystemExit(f"فایل تنظیمات یافت نشد: {config_path}")
        except (json.JSONDecodeError, yaml.YAMLError, ValueError) as e:
            logger.critical(f"خطای بحرانی: فایل تنظیمات {config_path} نامعتبر یا ناقص است: {e}. خروج.")
            raise SystemExit(f"فایل تنظیمات نامعتبر: {e}")
        except Exception as e:
            logger.critical(f"خطای بحرانی غیرمنتظره در بارگذاری تنظیمات {config_path}: {e}. خروج.", exc_info=True)
            raise SystemExit(f"خطا در بارگذاری تنظیمات: {e}")

    def reload_config(self) -> bool:
        """
        بارگذاری مجدد تنظیمات از فایل

        Returns:
            نتیجه موفقیت آمیز بودن عملیات
        """
        try:
            result = self.config_manager.check_for_changes()
            if result:
                logger.info("تنظیمات با موفقیت بارگذاری مجدد و اعمال شد")
            return result
        except Exception as e:
            logger.error(f"خطا در بارگذاری مجدد تنظیمات: {e}", exc_info=True)
            return False

    def _handle_config_changes(self, config: Dict[str, Any], modified_sections: set) -> None:
        """
        هندلر تغییرات تنظیمات

        Args:
            config: تنظیمات جدید
            modified_sections: بخش‌های تغییر یافته
        """
        try:
            logger.info(f"تغییرات تنظیمات دریافت شد. بخش‌های تغییر یافته: {', '.join(modified_sections)}")

            # ذخیره تنظیمات جدید
            self.config = config

            # ثبت تغییر در وضعیت
            timestamp = datetime.now().isoformat()
            self.running_status['config_changes'].append({
                'timestamp': timestamp,
                'sections': list(modified_sections)
            })

            # محدود کردن تعداد تغییرات ثبت شده
            if len(self.running_status['config_changes']) > 20:
                self.running_status['config_changes'] = self.running_status['config_changes'][-20:]

            # به‌روزرسانی تنظیمات اجزای سیستم
            self._update_components_config(modified_sections)

            # اقدامات خاص بر اساس بخش‌های تغییر یافته
            if 'logging' in modified_sections:
                # تنظیم مجدد لاگینگ
                self._setup_logging()

            if 'exchange' in modified_sections:
                # به‌روزرسانی اتصال به صرافی
                if self.exchange_client and hasattr(self.exchange_client, 'update_config'):
                    self.exchange_client.update_config(self.config.get('exchange', {}))

            # اگر بخش‌های مرتبط با معامله تغییر کرده‌اند، معاملات فعلی را بررسی کنیم
            trading_related_sections = {'risk_management', 'trading', 'circuit_breaker'}
            if trading_related_sections.intersection(modified_sections) and self.trade_manager:
                # به‌روزرسانی پارامترهای معاملات فعال
                asyncio.create_task(self._update_active_trades())

            # اگر بخش‌های مرتبط با سیگنال تغییر کرده‌اند، سیگنال‌دهی را به‌روز کنیم
            signal_related_sections = {'signal_generation', 'pattern_scores', 'market_regime'}
            if signal_related_sections.intersection(modified_sections) and self.signal_processor:
                # اجرای یک دور پردازش سیگنال با تنظیمات جدید
                asyncio.create_task(self._trigger_signal_processing())

        except Exception as e:
            logger.error(f"خطا در هندل کردن تغییرات تنظیمات: {e}", exc_info=True)

    async def _update_active_trades(self) -> None:
        """به‌روزرسانی پارامترهای معاملات فعال"""
        try:
            if not self.trade_manager:
                return

            # دریافت معاملات فعال
            active_trades = await self.trade_manager.get_active_positions()

            if not active_trades:
                logger.info("هیچ معامله فعالی برای به‌روزرسانی وجود ندارد")
                return

            logger.info(f"به‌روزرسانی پارامترهای {len(active_trades)} معامله فعال...")

            # فراخوانی متد به‌روزرسانی پارامترها در trade_manager
            if hasattr(self.trade_manager, 'update_trade_parameters'):
                result = await self.trade_manager.update_trade_parameters(active_trades)
                logger.info(f"{result} معامله با موفقیت به‌روزرسانی شد")
            else:
                logger.warning("متد update_trade_parameters در trade_manager وجود ندارد")

        except Exception as e:
            logger.error(f"خطا در به‌روزرسانی معاملات فعال: {e}", exc_info=True)

    async def _trigger_signal_processing(self) -> None:
        """اجرای یک دور پردازش سیگنال با تنظیمات جدید"""
        try:
            if not self.signal_processor:
                return

            logger.info("شروع یک دور پردازش سیگنال با تنظیمات جدید...")

            # فراخوانی متد پردازش سیگنال با نمادهای فعال
            if hasattr(self.signal_processor, 'process_all_symbols'):
                await self.signal_processor.process_all_symbols(force=True)
            elif hasattr(self.signal_processor, 'process_all'):
                await self.signal_processor.process_all(force=True)
            else:
                logger.warning("متد مناسب برای پردازش همه نمادها در signal_processor یافت نشد")

        except Exception as e:
            logger.error(f"خطا در اجرای پردازش سیگنال: {e}", exc_info=True)

    def _update_components_config(self, modified_sections: set = None) -> None:
        """
        به‌روزرسانی پیکربندی در تمام اجزای سیستم

        Args:
            modified_sections: بخش‌های تغییر یافته (اگر None باشد، همه به‌روزرسانی می‌شوند)
        """
        try:
            # دیکشنری برای نگه‌داری اجزایی که به‌روزرسانی کانفیگ دارند با تابع مربوطه
            components_with_update = {}

            # کانفیگ از استراتژی فعال (در صورت فعال بودن مدیریت استراتژی)
            if self.strategy_manager and self.strategy_manager.enabled:
                self.config = self.strategy_manager.get_current_strategy_config()
                components_with_update['Strategy Manager'] = self.strategy_manager

            # به‌روزرسانی پیکربندی در اجزای سیستم
            components = {
                'Signal Generator': self.signal_generator,
                'Signal Processor': self.signal_processor,
                'Trade Manager': self.trade_manager,
                'Data Fetcher': self.data_fetcher,
                'ML Integration': self.ml_integration,
                'Performance Tracker': self.performance_tracker,
                'Backup Manager': self.backup_manager,
                'Trading Brain AI': self.trading_brain
            }

            # ثبت اجزایی که روش update_config دارند
            for name, component in components.items():
                if component and hasattr(component, 'update_config'):
                    components_with_update[name] = component

            # تعداد اجزایی که به‌روزرسانی شده‌اند
            updated_count = 0

            # به‌روزرسانی فقط اجزایی که متد update_config دارند
            for name, component in components_with_update.items():
                try:
                    # بررسی بخش‌های مربوط به این کامپوننت
                    if self._should_update_component(name, modified_sections):
                        update_method = getattr(component, 'update_config')
                        update_method(self.config)
                        updated_count += 1
                        logger.debug(f"تنظیمات {name} به‌روزرسانی شد")
                except Exception as e:
                    logger.error(f"خطا در به‌روزرسانی تنظیمات {name}: {e}", exc_info=True)

            # به‌روزرسانی اجزایی که فقط فیلد config دارند (روش قدیمی)
            for name, component in components.items():
                if component not in components_with_update.values() and component and hasattr(component, 'config'):
                    try:
                        if self._should_update_component(name, modified_sections):
                            # فقط به‌روزرسانی اجزایی که update_config ندارند
                            component.config = self.config
                            updated_count += 1
                            logger.debug(f"تنظیمات {name} به صورت مستقیم به‌روزرسانی شد")
                    except Exception as e:
                        logger.error(f"خطا در به‌روزرسانی مستقیم تنظیمات {name}: {e}", exc_info=True)

            logger.info(f"تنظیمات {updated_count} جزء به‌روزرسانی شد")

        except Exception as e:
            logger.error(f"خطا در به‌روزرسانی تنظیمات اجزا: {e}", exc_info=True)

    def _should_update_component(self, component_name: str, modified_sections: set = None) -> bool:
        """
        تعیین اینکه آیا یک کامپوننت باید به‌روزرسانی شود یا خیر

        Args:
            component_name: نام کامپوننت
            modified_sections: بخش‌های تغییر یافته

        Returns:
            True اگر باید به‌روزرسانی شود
        """
        # اگر بخش‌های تغییریافته مشخص نشده، همه کامپوننت‌ها به‌روزرسانی شوند
        if modified_sections is None:
            return True

        # نگاشت کامپوننت‌ها به بخش‌های مرتبط
        component_sections = {
            'Signal Generator': {'signal_generation', 'pattern_scores', 'market_regime'},
            'Signal Processor': {'signal_processing', 'signal_generation'},
            'Trade Manager': {'trading', 'risk_management', 'circuit_breaker'},
            'Data Fetcher': {'data_fetching', 'exchange'},
            'ML Integration': {'ml_signal_integration', 'trading_brain_ai'},
            'Performance Tracker': {'performance_tracking'},
            'Backup Manager': {'backup'},
            'Strategy Manager': {'strategy_management'},
            'Trading Brain AI': {'trading_brain_ai'}
        }

        # بخش‌های مربوط به این کامپوننت
        relevant_sections = component_sections.get(component_name, set())

        # اگر هر کدام از بخش‌های مرتبط تغییر کرده‌اند، کامپوننت باید به‌روزرسانی شود
        return bool(relevant_sections.intersection(modified_sections))

    def _add_default_config_sections(self, config: Dict[str, Any]):
        """
        اضافه کردن بخش‌های پیش‌فرض به تنظیمات

        Args:
            config: دیکشنری تنظیمات
        """
        # 1. یادگیری تطبیقی
        if 'adaptive_learning' not in config:
            config['adaptive_learning'] = {
                'enabled': True,
                'data_file': 'data/adaptive_learning_data.json',
                'max_history_per_symbol': 100,
                'learning_rate': 0.1,
                'symbol_performance_weight': 0.3,
                'pattern_performance_weight': 0.3,
                'regime_performance_weight': 0.2
            }

        # 2. مدیریت همبستگی
        if 'correlation_management' not in config:
            config['correlation_management'] = {
                'enabled': True,
                'data_file': 'data/correlation_data.json',
                'correlation_threshold': 0.7,
                'max_exposure_per_group': 3,
                'update_interval': 86400
            }

        # 3. مکانیزم توقف اضطراری
        if 'circuit_breaker' not in config:
            config['circuit_breaker'] = {
                'enabled': True,
                'max_consecutive_losses': 3,
                'max_daily_losses_r': 5.0,
                'cool_down_period_minutes': 60,
                'reset_period_hours': 24
            }

        # 4. پشتیبان‌گیری خودکار
        if 'backup' not in config:
            config['backup'] = {
                'enabled': True,
                'directory': 'data/backups',
                'interval_hours': 24,
                'keep_days': 7,
                'compression': True,
                'critical_files': [
                    'data/trades.db',
                    'data/adaptive_learning_data.json'
                ]
            }

        # 5. مدیریت استراتژی
        if 'strategy_management' not in config:
            config['strategy_management'] = {
                'enabled': True,
                'strategies_dir': 'strategies',
                'default_strategy': 'default',
                'auto_rotation': False,
                'rotation_interval_hours': 24
            }

        # 6. ردیابی عملکرد
        if 'performance_tracking' not in config:
            config['performance_tracking'] = {
                'enabled': True,
                'metrics_file': 'data/performance_metrics.json',
                'max_history_days': 90
            }

        # 7. تنظیمات به‌روزرسانی خودکار کانفیگ
        if 'config_management' not in config:
            config['config_management'] = {
                'auto_reload': True,
                'check_interval_seconds': 30,
                'notify_changes': True,
                'backup_before_update': True
            }

    def _setup_logging(self):
        """راه‌اندازی سیستم لاگینگ با پشتیبانی از چرخش لاگ و فرمت‌های بهتر"""
        log_config = self.config.get('logging', {})
        log_level_str = log_config.get('level', 'INFO').upper()
        log_format = log_config.get('format', '%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        log_file = log_config.get('file')
        log_rotate = log_config.get('rotate', False)
        log_max_size = log_config.get('max_size', 10 * 1024 * 1024)  # 10MB پیش‌فرض
        log_backup_count = log_config.get('backup_count', 5)

        log_level = getattr(logging, log_level_str, logging.INFO)
        formatter = logging.Formatter(log_format)

        # تنظیم لاگر اصلی
        root_logger = logging.getLogger()
        if root_logger.hasHandlers():
            root_logger.handlers.clear()  # پاک کردن هندلرهای قبلی
        root_logger.setLevel(log_level)

        # هندلر کنسول
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(log_level)
        console_handler.setFormatter(formatter)
        root_logger.addHandler(console_handler)

        # هندلر فایل
        if log_file:
            try:
                log_dir = os.path.dirname(log_file)
                if log_dir and not os.path.exists(log_dir):
                    os.makedirs(log_dir)

                if log_rotate:
                    # هندلر فایل چرخشی
                    from logging.handlers import RotatingFileHandler
                    file_handler = RotatingFileHandler(
                        log_file,
                        maxBytes=log_max_size,
                        backupCount=log_backup_count,
                        encoding='utf-8'
                    )
                else:
                    # هندلر فایل ساده
                    file_handler = logging.FileHandler(log_file, encoding='utf-8')

                file_handler.setLevel(log_level)
                file_handler.setFormatter(formatter)
                root_logger.addHandler(file_handler)
                logger.info(f"لاگ در فایل: {log_file}{' (با چرخش)' if log_rotate else ''}")
            except Exception as e:
                logger.error(f"خطا در تنظیم هندلر فایل لاگ: {e}", exc_info=True)
        else:
            logger.info("لاگ فایل غیرفعال است.")

    async def initialize_components(self):
        """
        راه‌اندازی اجزای سیستم با مدیریت خطا و بازیابی

        Returns:
            True در صورت موفقیت
        """
        try:
            logger.info("در حال راه‌اندازی اجزای سیستم...")

            # 0. راه‌اندازی مدیریت استراتژی (قبل از بقیه کامپوننت‌ها)
            self.strategy_manager = TradingStrategyManager(self.config)
            await self.strategy_manager.initialize()
            self.running_status['components_status']['strategy_manager'] = 'initialized'
            logger.info("مدیریت استراتژی راه‌اندازی شد")

            # تعیین کانفیگ فعال بر اساس استراتژی انتخاب شده
            if self.strategy_manager.enabled:
                self.config = self.strategy_manager.get_current_strategy_config()
                logger.info(f"استفاده از پیکربندی استراتژی: {self.strategy_manager.current_strategy}")

            # --- >> اضافه شدن راه‌اندازی کامپوننت‌های هوش مصنوعی << ---
            # ایجاد نمونه TradingBrainAI
            ai_config = self.config.get('trading_brain_ai', {})
            if ai_config.get('enabled', True):
                self.trading_brain = TradingBrainAI(config=self.config)

                # اضافه کردن متد update_config به TradingBrainAI اگر وجود ندارد
                if not hasattr(self.trading_brain, 'update_config'):
                    self.trading_brain.update_config = lambda new_config: setattr(self.trading_brain, 'config',
                                                                                  new_config)

                self.running_status['components_status']['trading_brain_ai'] = 'initialized'
                logger.info("هوش مصنوعی معاملاتی راه‌اندازی شد")
            else:
                logger.info("هوش مصنوعی معاملاتی در تنظیمات غیرفعال است.")

            # 1. راه‌اندازی ExchangeClient
            from exchange_client import ExchangeClient
            self.exchange_client = ExchangeClient(self.config)
            await self.exchange_client._init_session()

            # اضافه کردن متد update_config به ExchangeClient اگر وجود ندارد
            if not hasattr(self.exchange_client, 'update_config'):
                def update_exchange_config(new_config):
                    """به‌روزرسانی تنظیمات در ExchangeClient"""
                    self.exchange_client.config = new_config
                    exchange_config = new_config.get('exchange', {})
                    # به‌روزرسانی API کلید‌ها (در صورت تغییر)
                    if 'api_key' in exchange_config:
                        self.exchange_client.api_key = exchange_config.get('api_key')
                    if 'api_secret' in exchange_config:
                        self.exchange_client.api_secret = exchange_config.get('api_secret')
                    if 'api_passphrase' in exchange_config:
                        self.exchange_client.api_passphrase = exchange_config.get('api_passphrase')
                    # به‌روزرسانی تنظیمات وب‌سوکت
                    if 'websocket' in exchange_config:
                        websocket_config = exchange_config.get('websocket', {})
                        self.exchange_client.ws_enabled = websocket_config.get('enabled', True)
                        self.exchange_client.ping_interval = websocket_config.get('ping_interval', 20)
                        self.exchange_client.auto_reconnect = websocket_config.get('auto_reconnect', True)
                    logger.info("تنظیمات ExchangeClient به‌روزرسانی شد")

                self.exchange_client.update_config = update_exchange_config

            self.running_status['components_status']['exchange_client'] = 'initialized'
            logger.info("کلاینت صرافی راه‌اندازی شد")

            # 2. راه‌اندازی MarketDataFetcher
            from market_data_fetcher import MarketDataFetcher
            self.data_fetcher = MarketDataFetcher(self.config, self.exchange_client)
            await self.data_fetcher.initialize()

            # اضافه کردن متد update_config به MarketDataFetcher اگر وجود ندارد
            if not hasattr(self.data_fetcher, 'update_config'):
                def update_data_fetcher_config(new_config):
                    """به‌روزرسانی تنظیمات در MarketDataFetcher"""
                    self.data_fetcher.config = new_config
                    data_config = new_config.get('data_fetching', {})
                    # به‌روزرسانی پارامترهای کلیدی
                    if 'max_symbols' in data_config:
                        self.data_fetcher.max_symbols = data_config.get('max_symbols', 0)
                    if 'auto_symbols' in data_config:
                        self.data_fetcher.auto_symbols = data_config.get('auto_symbols', True)
                    if 'timeframes' in data_config:
                        self.data_fetcher.timeframes = data_config.get('timeframes', ['5m', '15m', '1h', '4h'])
                    if 'max_concurrent_fetches' in data_config:
                        self.data_fetcher.max_concurrent_fetches = data_config.get('max_concurrent_fetches', 10)
                    # به‌روزرسانی تنظیمات کش
                    if 'cache' in data_config:
                        cache_config = data_config.get('cache', {})
                        self.data_fetcher.use_redis = cache_config.get('use_redis', False)
                        self.data_fetcher.use_memory_cache = cache_config.get('use_memory_cache', True)
                    logger.info("تنظیمات MarketDataFetcher به‌روزرسانی شد")

                self.data_fetcher.update_config = update_data_fetcher_config

            self.running_status['components_status']['data_fetcher'] = 'initialized'
            logger.info("دریافت‌کننده داده‌های بازار راه‌اندازی شد")

            # 3. راه‌اندازی SignalGenerator با سیستم یادگیری تطبیقی
            from signal_generator import SignalGenerator
            self.signal_generator = SignalGenerator(self.config)

            # اضافه کردن متد update_config به SignalGenerator اگر وجود ندارد
            if not hasattr(self.signal_generator, 'update_config'):
                def update_signal_generator_config(new_config):
                    """به‌روزرسانی تنظیمات در SignalGenerator"""
                    self.signal_generator.config = new_config

                    # به‌روزرسانی پارامترهای مختلف بر اساس بخش‌ها
                    if 'signal_generation' in new_config:
                        signal_config = new_config.get('signal_generation', {})
                        if hasattr(self.signal_generator, 'minimum_signal_score'):
                            self.signal_generator.minimum_signal_score = signal_config.get('minimum_signal_score', 33)
                        if hasattr(self.signal_generator, 'timeframes'):
                            self.signal_generator.timeframes = signal_config.get('timeframes',
                                                                                 ['5m', '15m', '1h', '4h'])
                        if hasattr(self.signal_generator, 'timeframe_weights'):
                            self.signal_generator.timeframe_weights = signal_config.get('timeframe_weights', {})

                    # به‌روزرسانی امتیازهای الگو
                    if 'pattern_scores' in new_config:
                        pattern_scores = new_config.get('pattern_scores', {})
                        if hasattr(self.signal_generator, 'pattern_scores'):
                            self.signal_generator.pattern_scores = pattern_scores

                    # به‌روزرسانی پارامترهای مربوط به رژیم بازار
                    if 'market_regime' in new_config and hasattr(self.signal_generator, 'market_regime_detector'):
                        regime_config = new_config.get('market_regime', {})
                        if hasattr(self.signal_generator.market_regime_detector, 'update_config'):
                            self.signal_generator.market_regime_detector.update_config(regime_config)

                    # به‌روزرسانی پارامترهای یادگیری تطبیقی
                    if 'adaptive_learning' in new_config and hasattr(self.signal_generator, 'adaptive_learning'):
                        adaptive_config = new_config.get('adaptive_learning', {})
                        if hasattr(self.signal_generator.adaptive_learning, 'update_config'):
                            self.signal_generator.adaptive_learning.update_config(adaptive_config)

                    # به‌روزرسانی مدیریت همبستگی
                    if 'correlation_management' in new_config and hasattr(self.signal_generator, 'correlation_manager'):
                        correlation_config = new_config.get('correlation_management', {})
                        if hasattr(self.signal_generator.correlation_manager, 'update_config'):
                            self.signal_generator.correlation_manager.update_config(correlation_config)

                    # به‌روزرسانی مکانیزم توقف اضطراری
                    if 'circuit_breaker' in new_config and hasattr(self.signal_generator, 'circuit_breaker'):
                        circuit_config = new_config.get('circuit_breaker', {})
                        if hasattr(self.signal_generator.circuit_breaker, 'update_config'):
                            self.signal_generator.circuit_breaker.update_config(circuit_config)

                    logger.info("تنظیمات SignalGenerator به‌روزرسانی شد")

                self.signal_generator.update_config = update_signal_generator_config

            self.running_status['components_status']['signal_generator'] = 'initialized'
            logger.info("مولد سیگنال راه‌اندازی شد")

            # --- >> اضافه شدن راه‌اندازی MLSignalIntegration << ---
            ml_integration_config = self.config.get('ml_signal_integration', {})
            if ml_integration_config.get('enabled', True) and self.trading_brain:
                self.ml_integration = MLSignalIntegration(
                    signal_generator=self.signal_generator,
                    trading_brain=self.trading_brain
                )

                # اضافه کردن متد update_config به MLSignalIntegration اگر وجود ندارد
                if not hasattr(self.ml_integration, 'update_config'):
                    def update_ml_integration_config(new_config):
                        """به‌روزرسانی تنظیمات در MLSignalIntegration"""
                        if 'ml_signal_integration' in new_config:
                            ml_config = new_config.get('ml_signal_integration', {})
                            self.ml_integration.config = ml_config
                            # به‌روزرسانی پارامترهای کلیدی
                            self.ml_integration.enhance_signals = ml_config.get('enhance_signals', True)
                            self.ml_integration.register_trade_results = ml_config.get('register_trade_results', True)
                            self.ml_integration.sync_interval_hours = ml_config.get('sync_interval_hours', 1)
                            logger.info("تنظیمات MLSignalIntegration به‌روزرسانی شد")

                    self.ml_integration.update_config = update_ml_integration_config

                self.running_status['components_status']['ml_signal_integration'] = 'initialized'
                logger.info("یکپارچه‌سازی سیگنال ML راه‌اندازی شد")
            else:
                logger.info("یکپارچه‌سازی سیگنال ML غیرفعال است یا هوش مصنوعی معاملاتی راه‌اندازی نشده است.")

            # 4. راه‌اندازی ردیابی عملکرد
            self.performance_tracker = BotPerformanceTracker(self.config, self.db_path)
            self.running_status['components_status']['performance_tracker'] = 'initialized'
            logger.info("ردیابی عملکرد راه‌اندازی شد")

            # 5. راه‌اندازی پشتیبان‌گیری خودکار
            self.backup_manager = BackupManager(self.config)
            self.running_status['components_status']['backup_manager'] = 'initialized'
            logger.info("مدیریت پشتیبان‌گیری راه‌اندازی شد")

            # 6. راه‌اندازی TradeManager
            from trade_manager import TradeManager
            self.trade_manager = TradeManager(self.config, self.db_path)
            self.trade_manager.initialize_db()

            # اضافه کردن متد update_config به TradeManager اگر وجود ندارد
            if not hasattr(self.trade_manager, 'update_config'):
                def update_trade_manager_config(new_config):
                    """به‌روزرسانی تنظیمات در TradeManager"""
                    self.trade_manager.config = new_config

                    # به‌روزرسانی پارامترهای معاملاتی
                    if 'trading' in new_config:
                        trading_config = new_config.get('trading', {})
                        self.trade_manager.mode = trading_config.get('mode', 'simulation')
                        self.trade_manager.auto_update_prices = trading_config.get('auto_update_prices', True)
                        self.trade_manager.price_update_interval = trading_config.get('price_update_interval', 10)

                        # به‌روزرسانی تنظیمات حد سود چندمرحله‌ای
                        if 'multi_tp' in trading_config:
                            multi_tp_config = trading_config.get('multi_tp', {})
                            if hasattr(self.trade_manager, 'use_multi_tp'):
                                self.trade_manager.use_multi_tp = multi_tp_config.get('enabled', True)

                    # به‌روزرسانی پارامترهای مدیریت ریسک
                    if 'risk_management' in new_config:
                        risk_config = new_config.get('risk_management', {})
                        for param, value in risk_config.items():
                            # به‌روزرسانی متغیرهای مختلف مدیریت ریسک
                            if hasattr(self.trade_manager, param):
                                setattr(self.trade_manager, param, value)
                                logger.debug(f"پارامتر '{param}' در TradeManager به {value} تنظیم شد")

                    # اعمال تغییرات به معاملات فعال
                    asyncio.create_task(self._update_active_trades())

                    logger.info("تنظیمات TradeManager به‌روزرسانی شد")

                self.trade_manager.update_config = update_trade_manager_config

                # اضافه کردن متد update_trade_parameters اگر وجود ندارد
                if not hasattr(self.trade_manager, 'update_trade_parameters'):
                    async def update_trade_parameters(self, active_trades):
                        """به‌روزرسانی پارامترهای معاملات فعال"""
                        updated_count = 0

                        for trade in active_trades:
                            try:
                                # به‌روزرسانی پارامترهای تریلینگ استاپ
                                if hasattr(trade, 'trailing_stop_params'):
                                    # استخراج تنظیمات جدید
                                    use_trailing = self.config.get('risk_management', {}).get('use_trailing_stop', True)
                                    activation_pct = self.config.get('risk_management', {}).get(
                                        'trailing_stop_activation_percent', 3.0)
                                    distance_pct = self.config.get('risk_management', {}).get(
                                        'trailing_stop_distance_percent', 2.25)
                                    use_atr = self.config.get('risk_management', {}).get('use_atr_based_trailing', True)
                                    atr_multiplier = self.config.get('risk_management', {}).get(
                                        'atr_trailing_multiplier', 2.25)

                                    # به‌روزرسانی پارامترها در معامله
                                    trade.trailing_stop_params.update({
                                        'enabled': use_trailing,
                                        'activation_percent': activation_pct,
                                        'distance_percent': distance_pct,
                                        'use_atr': use_atr,
                                        'atr_multiplier': atr_multiplier
                                    })
                                    updated_count += 1
                            except Exception as e:
                                logger.error(f"خطا در به‌روزرسانی پارامترهای معامله {trade.id}: {e}")

                        return updated_count

                    self.trade_manager.update_trade_parameters = update_trade_parameters.__get__(self.trade_manager,
                                                                                                 TradeManager)

            # ثبت کالبک‌های جدید در TradeManager
            self.trade_manager.register_price_fetcher(self._price_fetcher, self.data_fetcher)
            # --- >> ثبت کالبک نتیجه معامله در TradeManager << ---
            if self.ml_integration and ml_integration_config.get('register_trade_results', True):
                self.trade_manager.register_trade_result_callback(self.ml_integration.register_trade_result)
                logger.info("کالبک نتیجه معامله در TradeManager ثبت شد")

            self.running_status['components_status']['trade_manager'] = 'initialized'
            logger.info("مدیریت معاملات راه‌اندازی شد")

            # 7. راه‌اندازی SignalProcessor
            from signal_processor import SignalProcessor
            # --- >> ارسال MLSignalIntegration به SignalProcessor << ---
            self.signal_processor = SignalProcessor(
                config=self.config,
                market_data_fetcher=self.data_fetcher,
                signal_generator=self.signal_generator,
                ml_integration=self.ml_integration  # Pass the ML integration instance
            )

            # اضافه کردن متد update_config به SignalProcessor اگر وجود ندارد
            if not hasattr(self.signal_processor, 'update_config'):
                def update_signal_processor_config(new_config):
                    """به‌روزرسانی تنظیمات در SignalProcessor"""
                    self.signal_processor.config = new_config

                    # به‌روزرسانی پارامترهای کلیدی
                    if 'signal_processing' in new_config:
                        proc_config = new_config.get('signal_processing', {})
                        if hasattr(self.signal_processor, 'auto_forward_signals'):
                            self.signal_processor.auto_forward_signals = proc_config.get('auto_forward_signals', True)
                        if hasattr(self.signal_processor, 'signal_max_age_minutes'):
                            self.signal_processor.signal_max_age_minutes = proc_config.get('signal_max_age_minutes', 30)
                        if hasattr(self.signal_processor, 'check_incomplete_interval'):
                            self.signal_processor.check_incomplete_interval = proc_config.get(
                                'check_incomplete_interval', 60)
                        if hasattr(self.signal_processor, 'ohlcv_limit_per_tf'):
                            self.signal_processor.ohlcv_limit_per_tf = proc_config.get('ohlcv_limit_per_tf', 500)
                        if hasattr(self.signal_processor, 'use_ensemble_strategy'):
                            self.signal_processor.use_ensemble_strategy = proc_config.get('use_ensemble_strategy', True)

                    logger.info("تنظیمات SignalProcessor به‌روزرسانی شد")

                self.signal_processor.update_config = update_signal_processor_config

            await self.signal_processor.initialize()

            # تغییر مهم: ثبت صریح callback بین SignalProcessor و TradeManager
            self.signal_processor.register_trade_manager_callback(self.trade_manager.process_signal)
            logger.info(f"کالبک TradeManager در SignalProcessor ثبت شد")

            self.running_status['components_status']['signal_processor'] = 'initialized'
            logger.info("پردازش‌کننده سیگنال راه‌اندازی شد")

            # 8. دریافت و تنظیم نمادهای فعال
            await self._fetch_active_symbols()

            # اطمینان از فعال بودن auto_forward_signals
            if hasattr(self.signal_processor, 'auto_forward_signals'):
                if not self.signal_processor.auto_forward_signals:
                    self.signal_processor.auto_forward_signals = True
                    logger.info(f"مقدار auto_forward_signals در SignalProcessor به True تنظیم شد")

            self.running_status['state'] = 'ready'
            logger.info("تمام اجزا با موفقیت راه‌اندازی شدند")
            return True

        except Exception as e:
            logger.critical(f"خطای بحرانی در طول راه‌اندازی اجزا: {e}", exc_info=True)
            self.running_status['state'] = 'error'
            self.running_status['error'] = str(e)
            # ثبت در ردیابی عملکرد
            if self.performance_tracker:
                self.performance_tracker.record_error(f"خطای راه‌اندازی: {e}", 'error')
            # تلاش برای shutdown تمیز
            await self.shutdown()
            return False

    async def _handle_signal_generated(self, signal: SignalInfo) -> None:
        """
        پردازش سیگنال تولید شده برای ردیابی عملکرد

        Args:
            signal: اطلاعات سیگنال تولید شده
        """
        try:
            # ثبت در ردیابی عملکرد
            if self.performance_tracker:
                self.performance_tracker.record_signal_generated(signal)

            # به‌روزرسانی همبستگی‌ها
            if hasattr(self.signal_generator, 'correlation_manager'):
                # دریافت موقعیت‌های فعال از TradeManager
                active_positions = await self.trade_manager.get_active_positions()
                # به‌روزرسانی در مدیریت همبستگی
                active_pos_dict = {pos.symbol: {'direction': pos.side} for pos in active_positions}
                self.signal_generator.correlation_manager.update_active_positions(active_pos_dict)

        except Exception as e:
            logger.error(f"خطا در پردازش سیگنال تولید شده: {e}", exc_info=True)

    async def _handle_trade_result(self, trade: Trade, trade_result: TradeResult) -> None:
        """
        پردازش نتیجه معامله برای یادگیری تطبیقی و ردیابی عملکرد

        Args:
            trade: اطلاعات معامله
            trade_result: نتیجه معامله
        """
        try:
            # ثبت در یادگیری تطبیقی
            if hasattr(self.signal_generator, 'register_trade_result'):
                self.signal_generator.register_trade_result(trade_result)

            # ثبت در ردیابی عملکرد
            if self.performance_tracker:
                self.performance_tracker.record_trade_result(trade, trade_result)

            # ثبت در مدیریت استراتژی
            if self.strategy_manager and self.strategy_manager.enabled:
                strategy_name = self.strategy_manager.current_strategy
                self.strategy_manager.record_trade_result(
                    strategy_name,
                    trade_result.profit_pct > 0,
                    trade_result.profit_r
                )

            # بررسی وجود مکانیزم توقف اضطراری
            if hasattr(self.signal_generator, 'circuit_breaker'):
                # اضافه کردن به سیستم توقف اضطراری
                self.signal_generator.circuit_breaker.add_trade_result(trade_result)

                # بررسی فعال بودن توقف اضطراری
                is_active, reason = self.signal_generator.circuit_breaker.check_if_active()
                if is_active:
                    logger.warning(f"توقف اضطراری فعال است: {reason}")

        except Exception as e:
            logger.error(f"خطا در پردازش نتیجه معامله: {e}", exc_info=True)

    async def _price_fetcher(self, symbol: str) -> Optional[float]:
        """
        تابع wrapper برای دریافت قیمت فعلی

        Args:
            symbol: نماد ارز

        Returns:
            قیمت فعلی یا None
        """
        if not self.data_fetcher:
            logger.error("تابع دریافت قیمت قبل از راه‌اندازی MarketDataFetcher فراخوانی شده است.")
            return None

        try:
            price = await self.data_fetcher.get_current_price(symbol)
            if price is None:
                logger.warning(f"دریافت قیمت فعلی برای {symbol} از طریق MarketDataFetcher ناموفق بود.")
            return price
        except Exception as e:
            logger.error(f"خطا در دریافت قیمت فعلی برای {symbol}: {e}")
            return None

    async def _fetch_active_symbols(self):
        """دریافت لیست نمادهای فعال"""
        try:
            logger.info("در حال دریافت نمادهای فعال...")
            symbols = await self.data_fetcher.get_active_symbols()

            if not symbols:
                logger.warning("هیچ نماد فعالی از صرافی دریافت نشد.")
                # استفاده از لیست پیش‌فرض
                default_symbols = ['BTC/USDT', 'ETH/USDT']
                logger.warning(f"استفاده از نمادهای پیش‌فرض: {default_symbols}")
                self.active_symbols = default_symbols
            else:
                self.active_symbols = symbols
                logger.info(f"{len(symbols)} نماد فعال از صرافی دریافت شد.")

                # محدود کردن تعداد نمادها
                max_symbols = self.config.get('data_fetching', {}).get('max_symbols', 0)
                if max_symbols > 0 and len(self.active_symbols) > max_symbols:
                    logger.info(f"محدود کردن نمادهای فعال به {max_symbols}")
                    self.active_symbols = self.active_symbols[:max_symbols]

            # اعمال لیست نمادها به SignalProcessor
            self.signal_processor.set_active_symbols(self.active_symbols)
            logger.info(f"{len(self.active_symbols)} نماد فعال برای پردازش تنظیم شد.")

            # به‌روزرسانی همبستگی‌ها
            if hasattr(self.signal_generator, 'correlation_manager'):
                # آماده‌سازی داده برای محاسبه همبستگی
                symbols_data = {}
                for symbol in self.active_symbols:
                    timeframes = self.config.get('signal_generation', {}).get('timeframes', ['4h'])
                    if '4h' not in timeframes:
                        timeframes.append('4h')  # اطمینان از وجود تایم‌فریم مناسب

                    for tf in timeframes:
                        try:
                            ohlcv = await self.data_fetcher.get_ohlcv(symbol, tf, limit=100)
                            if ohlcv is not None and len(ohlcv) > 50:  # حداقل داده لازم
                                if symbol not in symbols_data:
                                    symbols_data[symbol] = {}
                                symbols_data[symbol][tf] = ohlcv
                        except Exception as e:
                            logger.debug(f"خطا در دریافت داده {symbol} {tf} برای همبستگی: {e}")

                # به‌روزرسانی همبستگی‌ها
                self.signal_generator.correlation_manager.update_correlations(symbols_data)

        except Exception as e:
            logger.error(f"خطا در دریافت نمادهای فعال: {e}. استفاده از لیست خالی.", exc_info=True)
            self.active_symbols = []
            self.signal_processor.set_active_symbols([])

    async def start_services(self):
        """
        شروع سرویس‌های پس‌زمینه با نظارت و مدیریت خطا

        Returns:
            True در صورت موفقیت
        """
        try:
            # تنظیم وضعیت اجرا
            self.running_status['state'] = 'starting_services'

            # شروع به‌روزرسانی دوره‌ای قیمت‌ها در TradeManager
            if self.trade_manager.auto_update_prices:
                await self.trade_manager.start_periodic_price_update()
                self.running_status['components_status']['trade_manager'] = 'running'
            else:
                logger.info("به‌روزرسانی خودکار قیمت‌ها غیرفعال است.")
                self.running_status['components_status']['trade_manager'] = 'running_no_updates'

            # شروع پردازش دوره‌ای سیگنال‌ها
            await self.signal_processor.start_periodic_processing()
            self.running_status['components_status']['signal_processor'] = 'running'

            # شروع پشتیبان‌گیری خودکار
            if self.backup_manager and self.backup_manager.enabled:
                await self.backup_manager.start_automated_backup()
                self.running_status['components_status']['backup_manager'] = 'running'

            # شروع نظارت بر تغییرات تنظیمات
            config_watch_enabled = self.config.get('config_management', {}).get('auto_reload', True)
            if config_watch_enabled:
                # تنظیم تسک نظارت بر تنظیمات
                self._config_check_task = asyncio.create_task(self._config_watch_loop())
                logger.info("نظارت بر تغییرات تنظیمات آغاز شد")
                self.running_status['components_status']['config_watcher'] = 'running'

            # به‌روزرسانی وضعیت
            self.running_status['state'] = 'running'
            logger.info("سرویس‌های پس‌زمینه شروع شدند")

            # ایجاد پشتیبان اولیه
            if self.backup_manager and self.backup_manager.enabled:
                backup_path = await self.backup_manager.create_backup()
                if backup_path:
                    logger.info(f"پشتیبان اولیه در {backup_path} ایجاد شد")

            return True

        except Exception as e:
            logger.error(f"خطا در شروع سرویس‌های پس‌زمینه: {e}", exc_info=True)
            self.running_status['state'] = 'error'
            self.running_status['error'] = str(e)
            # ثبت در ردیابی عملکرد
            if self.performance_tracker:
                self.performance_tracker.record_error(f"خطا در شروع سرویس‌ها: {e}", 'error')
            return False

    async def _config_watch_loop(self):
        """حلقه نظارت بر تغییرات تنظیمات"""
        try:
            check_interval = self.config.get('config_management', {}).get('check_interval_seconds', 30)
            logger.info(f"شروع نظارت بر تغییرات تنظیمات هر {check_interval} ثانیه")

            while not self._shutdown_requested.is_set():
                # بررسی تغییرات در فایل تنظیمات
                self.config_manager.check_for_changes()

                # انتظار تا بررسی بعدی
                await asyncio.sleep(check_interval)

        except asyncio.CancelledError:
            logger.info("حلقه نظارت بر تنظیمات لغو شد")
        except Exception as e:
            logger.error(f"خطا در حلقه نظارت بر تنظیمات: {e}", exc_info=True)

    async def shutdown(self):
        """پاکسازی و بستن اجزا به صورت امن و ترتیبی"""
        logger.info("شروع خاموش شدن تمیز...")
        self._shutdown_requested.set()
        self.running_status['state'] = 'shutting_down'

        # لیست کامپوننت‌ها برای shutdown به ترتیب معکوس
        components = [
            ('Config Watcher', getattr(self, '_config_check_task', None), None),
            ('Backup Manager', self.backup_manager, getattr(self.backup_manager, 'stop', None)),
            ('Strategy Manager', self.strategy_manager, getattr(self.strategy_manager, 'stop', None)),
            ('Signal Processor', self.signal_processor,
             getattr(self.signal_processor, 'stop_periodic_processing', None)),
            ('Signal Processor', self.signal_processor, getattr(self.signal_processor, 'shutdown', None)),
            ('Trade Manager', self.trade_manager, getattr(self.trade_manager, 'stop_periodic_price_update', None)),
            ('Trade Manager', self.trade_manager, getattr(self.trade_manager, 'shutdown', None)),
            ('Data Fetcher', self.data_fetcher, getattr(self.data_fetcher, 'shutdown', None)),
            ('Exchange Client', self.exchange_client, getattr(self.exchange_client, 'close', None)),
            ('Signal Generator', self.signal_generator, getattr(self.signal_generator, 'shutdown', None)),
            ('ML Integration', self.ml_integration, getattr(self.ml_integration, 'shutdown', None)),
            ('Trading Brain AI', self.trading_brain, getattr(self.trading_brain, 'shutdown', None))
        ]

        # ایجاد پشتیبان نهایی قبل از خروج
        try:
            if self.backup_manager and self.backup_manager.enabled:
                backup_path = await self.backup_manager.create_backup()
                if backup_path:
                    logger.info(f"پشتیبان نهایی در {backup_path} ایجاد شد")
        except Exception as e:
            logger.warning(f"ایجاد پشتیبان نهایی ناموفق بود: {e}")

        # ذخیره متریک‌های عملکرد
        try:
            if self.performance_tracker:
                self.performance_tracker.save_metrics()
                logger.info("متریک‌های عملکرد ذخیره شدند")
        except Exception as e:
            logger.warning(f"ذخیره متریک‌های عملکرد ناموفق بود: {e}")

        # لغو تسک config_check_task
        if hasattr(self, '_config_check_task') and self._config_check_task and not self._config_check_task.done():
            try:
                self._config_check_task.cancel()
                await asyncio.sleep(0.1)  # فرصت کوتاهی برای لغو
                logger.info("تسک نظارت بر تنظیمات لغو شد")
            except Exception as e:
                logger.error(f"خطا در لغو تسک نظارت بر تنظیمات: {e}")

        for name, component, shutdown_method in components:
            if component and shutdown_method:
                try:
                    logger.debug(f"در حال خاموش کردن {name}...")
                    result = shutdown_method()
                    # اگر coroutine باشد
                    if asyncio.iscoroutine(result):
                        await result
                    self.running_status['components_status'][name.lower().replace(' ', '_')] = 'shutdown'
                except Exception as e:
                    logger.error(f"خطا در خاموش کردن {name}: {e}")

        # لغو تسک‌های باقی‌مانده asyncio
        try:
            tasks = [t for t in asyncio.all_tasks() if t is not asyncio.current_task()]
            if tasks:
                logger.info(f"لغو {len(tasks)} تسک باقی‌مانده...")
                for task in tasks:
                    task.cancel()
                await asyncio.gather(*tasks, return_exceptions=True)
        except Exception as e:
            logger.error(f"خطا در لغو تسک‌ها: {e}")

        self.running_status['state'] = 'shutdown'
        logger.info("خاموش شدن کامل شد.")

    def _format_uptime(self, seconds: int) -> str:
        """
        فرمت‌بندی مدت زمان کارکرد

        Args:
            seconds: مدت زمان به ثانیه

        Returns:
            رشته فرمت‌بندی شده
        """
        days, remainder = divmod(seconds, 86400)
        hours, remainder = divmod(remainder, 3600)
        minutes, seconds = divmod(remainder, 60)

        if days > 0:
            return f"{days}روز {hours}ساعت {minutes}دقیقه {seconds}ثانیه"
        elif hours > 0:
            return f"{hours}ساعت {minutes}دقیقه {seconds}ثانیه"
        elif minutes > 0:
            return f"{minutes}دقیقه {seconds}ثانیه"
        else:
            return f"{seconds}ثانیه"

    async def _train_ai_models(self) -> bool:
        """
        انجام آموزش اولیه برای مدل‌های TradingBrainAI با جلوگیری از فراخوانی دوگانه
        """
        # افزودن محافظ برای جلوگیری از فراخوانی همزمان
        if hasattr(self, '_training_in_progress') and self._training_in_progress:
            logger.warning("آموزش هوش مصنوعی قبلاً شروع شده، درخواست نادیده گرفته می‌شود")
            return False

        self._training_in_progress = True

        try:
            if not self.trading_brain or not self.config.get('trading_brain_ai', {}).get('enabled', True):
                logger.info("هوش مصنوعی معاملاتی غیرفعال یا راه‌اندازی نشده است. عدم آموزش.")
                return False

            logger.info("شروع آموزش اولیه مدل‌های هوش مصنوعی...")

            # --- 1. دریافت داده‌های تاریخی برای آموزش ---
            symbol_for_training = "BTC/USDT"  # یا چندین نماد
            timeframe_for_training = "1h"  # یا یک تایم‌فریم پایه

            # تعیین lookback مناسب برای feature engineering
            feature_lookback = self.config.get('trading_brain_ai', {}).get('feature_engineering', {}).get(
                'lookback_window', 20)
            # دریافت داده‌های کافی برای پوشش lookback و دوره آموزش
            training_data_limit = 365 * 24  # 1 سال داده ساعتی
            fetch_limit = training_data_limit + feature_lookback + 100  # افزودن بافر

            logger.info(
                f"دریافت داده‌های تاریخی برای آموزش هوش مصنوعی: {symbol_for_training} {timeframe_for_training} (تعداد: {fetch_limit})")
            historical_df = await self.data_fetcher.get_historical_data(
                symbol_for_training,
                timeframe_for_training,
                limit=fetch_limit
            )

            if historical_df is None or historical_df.empty:
                logger.error("دریافت داده‌های تاریخی برای آموزش هوش مصنوعی ناموفق بود.")
                return False

            logger.info(f"{len(historical_df)} داده برای آموزش دریافت شد.")

            # --- 2. آماده‌سازی داده و جلوگیری از فراخوانی دوباره آن در train_models ---
            prepared_data = self.trading_brain.prepare_data(historical_df, return_dict=True)

            if prepared_data is None:
                logger.error("آماده‌سازی داده برای آموزش هوش مصنوعی ناموفق بود.")
                return False

            ml_data = prepared_data['ml_data']
            seq_data = prepared_data['seq_data']

            if ml_data is None or seq_data is None:
                logger.error("داده‌های آماده‌سازی شده برای آموزش هوش مصنوعی ناقص است.")
                return False

            # --- 3. آموزش مدل‌های پیش‌بینی با ارسال صریح data=None برای جلوگیری از فراخوانی دوباره prepare_data ---
            logger.info("در حال آموزش مدل‌های پیش‌بینی هوش مصنوعی (Ensemble)...")
            training_result = await self.trading_brain.train_models(data=None, ml_data=ml_data, seq_data=seq_data)

            if training_result.get('status') != 'success':
                logger.error(
                    f"آموزش مدل‌های پیش‌بینی هوش مصنوعی ناموفق بود: {training_result.get('message', 'خطای نامشخص')}")
                return False

            logger.info("مدل‌های پیش‌بینی هوش مصنوعی با موفقیت آموزش داده شدند.")

            # --- 4. آموزش مدل Position Sizer (در صورت فعال بودن) ---
            if self.config.get('trading_brain_ai', {}).get('position_sizer', {}).get('enabled', False):
                logger.info("در حال آموزش مدل Position Sizer هوش مصنوعی...")
                position_sizer_training_data = prepared_data['features_df']  # استفاده از داده‌های feature-engineered

                # اینجا هم باید از await استفاده کنیم اگر تابع async است
                position_sizer_result = await self.trading_brain.train_position_sizer(
                    data=None,  # تنظیم صریح data=None برای جلوگیری از فراخوانی دوباره prepare_data
                    features=position_sizer_training_data.drop(
                        columns=[col for col in position_sizer_training_data.columns if
                                 col.startswith('target_')]).values
                )

                if position_sizer_result.get('status') != 'success':
                    logger.warning(
                        f"آموزش Position Sizer هوش مصنوعی ناموفق بود: {position_sizer_result.get('message', 'خطای نامشخص')}")
                else:
                    logger.info("مدل Position Sizer هوش مصنوعی با موفقیت آموزش داده شد.")

            logger.info("آموزش اولیه مدل‌های هوش مصنوعی کامل شد.")
            return True

        except Exception as e:
            logger.critical(f"خطای بحرانی در آموزش مدل‌های هوش مصنوعی: {e}", exc_info=True)
            if self.performance_tracker:
                self.performance_tracker.record_error(f"خطای آموزش هوش مصنوعی: {e}", 'error')
            return False
        finally:
            # حتی در صورت بروز خطا، وضعیت را پاک می‌کنیم
            self._training_in_progress = False

    async def run(self):
        """
        اجرای ربات معاملاتی با مدیریت خطا و گزارش وضعیت
        """
        try:
            logger.info("شروع اجرای ربات معاملاتی ارز دیجیتال...")
            self.running_status['state'] = 'initializing'

            # راه‌اندازی کامپوننت‌ها
            if not await self.initialize_components():
                logger.critical("راه‌اندازی اجزا ناموفق بود. ربات نمی‌تواند شروع شود.")
                return False

            # --- آموزش مدل‌های هوش مصنوعی با افزودن محافظ برای یکبار اجرا ---
            if self.trading_brain and not hasattr(self, '_ai_training_completed'):
                training_success = await self._train_ai_models()
                self._ai_training_completed = True  # فقط یکبار آموزش، حتی اگر ناموفق باشد

                if not training_success:
                    logger.warning("آموزش مدل‌های هوش مصنوعی ناموفق بود، اما ربات با قابلیت‌های محدود ادامه می‌دهد")

            # شروع سرویس‌ها
            if not await self.start_services():
                logger.critical("شروع سرویس‌های پس‌زمینه ناموفق بود. در حال خاموش شدن.")
                await self.shutdown()
                return False

            logger.info("ربات در حال اجرا است. Ctrl+C برای خروج.")

            # حلقه اصلی با به‌روزرسانی وضعیت
            while not self._shutdown_requested.is_set():
                # به‌روزرسانی زمان کارکرد
                self.running_status['uptime_seconds'] = int(time.time() - self.start_time)
                self.running_status['last_status_update'] = time.time()

                # بررسی وضعیت کامپوننت‌ها
                await self._check_component_health()

                # به‌روزرسانی دوره‌ای داده‌های بازار (هر 30 دقیقه)
                if self.running_status['uptime_seconds'] % 1800 < 10:
                    try:
                        await self._fetch_active_symbols()
                    except Exception as e:
                        logger.error(f"خطا در به‌روزرسانی نمادهای فعال: {e}")

                # انتظار برای سیگنال توقف
                await asyncio.sleep(10)

            # خروج عادی
            logger.info("سیگنال خاموش شدن دریافت شد. آغاز فرآیند خاموش شدن...")
            await self.shutdown()
            logger.info("ربات با موفقیت خاموش شد.")
            return True

        except asyncio.CancelledError:
            logger.info("اجرای ربات لغو شد. در حال خاموش شدن...")
            await self.shutdown()
            return False
        except SystemExit as e:
            logger.critical(f"فراخوانی SystemExit: {e}. اجبار به خاموش شدن.")
            try:
                await self.shutdown()
            except:
                pass
            return False
        except Exception as e:
            logger.critical(f"خطای بحرانی کنترل نشده در اجرای ربات: {e}. تلاش برای خاموش شدن.", exc_info=True)
            # ثبت خطا
            if self.performance_tracker:
                self.performance_tracker.record_error(f"خطای کنترل نشده: {e}", 'error')

            try:
                await self.shutdown()
            except:
                pass
            return False

    async def _check_component_health(self) -> None:
        """بررسی سلامت کامپوننت‌ها و اقدام به بازیابی در صورت نیاز"""
        try:
            # بررسی وضعیت ExchangeClient
            if self.exchange_client:
                try:
                    # به جای پینگ مستقیم، از یک درخواست سبک مثل گرفتن زمان سرور برای تست اتصال استفاده می‌کنیم
                    server_time = await self.exchange_client.get_server_time()
                    if server_time is None:
                        # اگر زمان سرور None برگشت، ممکن است مشکلی در اتصال یا API باشد
                        raise ConnectionError("دریافت زمان سرور ناموفق بود (احتمالاً مشکل اتصال یا خطای API)")
                except ConnectionError as ce:
                    logger.error(f"خطای اتصال در بررسی وضعیت کلاینت صرافی: {ce}. تلاش برای راه‌اندازی مجدد نشست...")
                    try:
                        # تلاش برای راه‌اندازی مجدد session
                        await self.exchange_client._init_session()
                    except Exception as reinit_error:
                        logger.error(f"راه‌اندازی مجدد نشست صرافی ناموفق بود: {reinit_error}")
                except Exception as e:
                    # سایر خطاهای غیرمنتظره
                    logger.error(f"خطا در بررسی سلامت کلاینت صرافی: {e}")

            # بررسی وضعیت DataFetcher
            if self.data_fetcher and hasattr(self.data_fetcher, 'last_fetch_time'):
                # بررسی آخرین زمان دریافت داده
                last_fetch_time = getattr(self.data_fetcher, 'last_fetch_time', 0)
                elapsed = time.time() - last_fetch_time
                # اگر بیش از 10 دقیقه از آخرین دریافت داده گذشته باشد
                if elapsed > 600:
                    logger.warning(
                        f"زمان زیادی از آخرین دریافت داده گذشته است: {int(elapsed)} ثانیه. بررسی عملکرد DataFetcher.")

            # بررسی وضعیت SignalProcessor
            if self.signal_processor and hasattr(self.signal_processor, 'is_running'):
                if not self.signal_processor.is_running:
                    logger.warning("SignalProcessor متوقف شده است. تلاش برای راه‌اندازی مجدد...")
                    try:
                        await self.signal_processor.start_periodic_processing()
                    except Exception as e:
                        logger.error(f"راه‌اندازی مجدد SignalProcessor ناموفق بود: {e}")

            # بررسی وضعیت TradeManager
            if self.trade_manager and hasattr(self.trade_manager, 'price_update_running'):
                if not self.trade_manager.price_update_running and self.trade_manager.auto_update_prices:
                    logger.warning("به‌روزرسانی قیمت در TradeManager متوقف شده است. تلاش برای راه‌اندازی مجدد...")
                    try:
                        await self.trade_manager.start_periodic_price_update()
                    except Exception as e:
                        logger.error(f"راه‌اندازی مجدد به‌روزرسانی قیمت در TradeManager ناموفق بود: {e}")

            # بررسی وضعیت نظارت بر تنظیمات
            if hasattr(self, '_config_check_task') and self._config_check_task and self._config_check_task.done():
                if not self._shutdown_requested.is_set():
                    logger.warning("تسک نظارت بر تنظیمات متوقف شده است. تلاش برای راه‌اندازی مجدد...")
                    try:
                        self._config_check_task = asyncio.create_task(self._config_watch_loop())
                        logger.info("تسک نظارت بر تنظیمات مجدداً راه‌اندازی شد")
                    except Exception as e:
                        logger.error(f"راه‌اندازی مجدد تسک نظارت بر تنظیمات ناموفق بود: {e}")

        except Exception as e:
            # خطای کلی در فرآیند بررسی سلامت
            logger.error(f"خطا در طول بررسی سلامت اجزا: {e}", exc_info=True)

    def stop(self):
        """درخواست توقف ربات از خارج"""
        logger.info("درخواست توقف خارجی دریافت شد.")
        self._shutdown_requested.set()

    def get_status(self) -> Dict[str, Any]:
        """
        دریافت وضعیت فعلی ربات

        Returns:
            دیکشنری وضعیت
        """
        # افزودن آمار از کامپوننت‌ها
        component_stats = {}

        if self.trade_manager:
            component_stats['trade_manager'] = self.trade_manager.get_stats()

        if self.signal_processor:
            try:
                component_stats['signal_processor'] = self.signal_processor.get_signals_summary()
            except:
                pass

        # اضافه کردن آمار از کامپوننت‌های جدید
        if self.performance_tracker:
            try:
                component_stats['performance'] = self.performance_tracker.get_performance_report()
            except Exception as e:
                logger.error(f"خطا در دریافت متریک‌های عملکرد: {e}")

        if self.strategy_manager and self.strategy_manager.enabled:
            try:
                component_stats['strategy'] = {
                    'current_strategy': self.strategy_manager.current_strategy,
                    'active_strategies': len(self.strategy_manager.active_strategies),
                    'auto_rotation': self.strategy_manager.auto_rotation
                }
            except Exception as e:
                logger.error(f"خطا در دریافت متریک‌های استراتژی: {e}")

        # بررسی وضعیت توقف اضطراری
        if hasattr(self.signal_generator, 'circuit_breaker'):
            try:
                is_active, reason = self.signal_generator.circuit_breaker.check_if_active()
                component_stats['circuit_breaker'] = {
                    'active': is_active,
                    'reason': reason if is_active else None
                }
            except Exception as e:
                logger.error(f"خطا در بررسی وضعیت توقف اضطراری: {e}")

        # اضافه کردن آمار تغییرات تنظیمات
        config_changes = self.running_status.get('config_changes', [])
        if config_changes:
            component_stats['config_changes'] = {
                'count': len(config_changes),
                'last_change': config_changes[-1] if config_changes else None
            }

        # ترکیب با وضعیت اجرا
        status = {
            **self.running_status,
            'component_stats': component_stats,
            'active_symbols_count': len(self.active_symbols),
            'formatted_uptime': self._format_uptime(self.running_status['uptime_seconds'])
        }

        return status

    async def execute_command(self, command: str, params: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        اجرای دستور از راه دور

        Args:
            command: دستور مورد نظر
            params: پارامترهای دستور

        Returns:
            نتیجه اجرای دستور
        """
        result = {
            'success': False,
            'message': 'دستور ناشناخته',
            'data': None
        }

        if not params:
            params = {}

        try:
            if command == 'status':
                # دریافت وضعیت کامل
                result['data'] = self.get_status()
                result['success'] = True
                result['message'] = 'وضعیت با موفقیت دریافت شد'

            elif command == 'stop':
                # توقف ربات
                self.stop()
                result['success'] = True
                result['message'] = 'سیگنال توقف با موفقیت ارسال شد'

            elif command == 'restart_component':
                # راه‌اندازی مجدد یک کامپوننت
                component = params.get('component')
                if not component:
                    result['message'] = 'نام کامپوننت مورد نیاز است'
                    return result

                await self._restart_component(component)
                result['success'] = True
                result['message'] = f'راه‌اندازی مجدد کامپوننت {component} آغاز شد'

            elif command == 'switch_strategy':
                # تغییر استراتژی
                strategy = params.get('strategy')
                if not strategy:
                    result['message'] = 'نام استراتژی مورد نیاز است'
                    return result

                if not self.strategy_manager or not self.strategy_manager.enabled:
                    result['message'] = 'مدیریت استراتژی فعال نیست'
                    return result

                success = await self.strategy_manager.switch_strategy(strategy)
                if success:
                    result['success'] = True
                    result['message'] = f'تغییر به استراتژی {strategy} انجام شد'

                    # به‌روزرسانی کانفیگ
                    self.config = self.strategy_manager.get_current_strategy_config()

                    # به‌روزرسانی کامپوننت‌ها با کانفیگ جدید
                    self._update_components_config()
                else:
                    result['message'] = f'تغییر به استراتژی {strategy} ناموفق بود'

            elif command == 'create_backup':
                # ایجاد پشتیبان
                if not self.backup_manager or not self.backup_manager.enabled:
                    result['message'] = 'مدیریت پشتیبان‌گیری فعال نیست'
                    return result

                backup_path = await self.backup_manager.create_backup()
                if backup_path:
                    result['success'] = True
                    result['message'] = 'پشتیبان با موفقیت ایجاد شد'
                    result['data'] = {'backup_path': backup_path}
                else:
                    result['message'] = 'ایجاد پشتیبان ناموفق بود'

            elif command == 'active_positions':
                # دریافت لیست موقعیت‌های فعال
                if not self.trade_manager:
                    result['message'] = 'مدیریت معاملات راه‌اندازی نشده است'
                    return result

                positions = await self.trade_manager.get_active_positions()
                result['success'] = True
                result['message'] = f'{len(positions)} موقعیت فعال دریافت شد'
                result['data'] = [pos.to_dict() for pos in positions]

            elif command == 'emergency_close_all':
                # بستن تمام موقعیت‌ها در شرایط اضطراری
                if not self.trade_manager:
                    result['message'] = 'مدیریت معاملات راه‌اندازی نشده است'
                    return result

                reason = params.get('reason', 'درخواست بستن اضطراری')
                closed = await self.trade_manager.close_all_positions(reason)
                result['success'] = True
                result['message'] = f'بستن اضطراری برای {closed} موقعیت آغاز شد'

            elif command == 'update_config':
                # به‌روزرسانی بخشی از کانفیگ در زمان اجرا
                config_section = params.get('section')
                config_data = params.get('data')

                if not config_section or not config_data or not isinstance(config_data, dict):
                    result['message'] = 'بخش و داده معتبر مورد نیاز است'
                    return result

                # به‌روزرسانی کانفیگ از طریق ConfigurationManager
                if self.config_manager.update_section(config_section, config_data):
                    result['success'] = True
                    result['message'] = f'بخش کانفیگ {config_section} به‌روزرسانی شد'
                else:
                    result['message'] = f'به‌روزرسانی بخش کانفیگ {config_section} ناموفق بود'

            elif command == 'reload_config':
                # بارگذاری مجدد فایل کانفیگ
                success = self.reload_config()
                if success:
                    result['success'] = True
                    result['message'] = 'کانفیگ با موفقیت مجدداً بارگذاری شد'
                else:
                    result['message'] = 'بارگذاری مجدد کانفیگ ناموفق بود'

            elif command == 'reload_strategies':
                # بارگذاری مجدد فایل‌های استراتژی
                if not self.strategy_manager or not self.strategy_manager.enabled:
                    result['message'] = 'مدیریت استراتژی فعال نیست'
                    return result

                self.strategy_manager.reload_strategies()
                result['success'] = True
                result['message'] = 'استراتژی‌ها با موفقیت مجدداً بارگذاری شدند'

            elif command == 'update_trade_parameters':
                # به‌روزرسانی پارامترهای معاملات فعال
                if not self.trade_manager:
                    result['message'] = 'مدیریت معاملات راه‌اندازی نشده است'
                    return result

                updated_count = await self._update_active_trades()
                result['success'] = True
                result['message'] = f'پارامترهای {updated_count} معامله به‌روزرسانی شد'

            elif command == 'force_signal_process':
                # اجرای اجباری یک دور پردازش سیگنال
                if not self.signal_processor:
                    result['message'] = 'پردازش‌کننده سیگنال راه‌اندازی نشده است'
                    return result

                await self._trigger_signal_processing()
                result['success'] = True
                result['message'] = 'پردازش سیگنال اجباری آغاز شد'

            else:
                result['message'] = f'دستور ناشناخته: {command}'

        except Exception as e:
            result['message'] = f'خطا در اجرای دستور: {str(e)}'
            result['error'] = str(e)
            logger.error(f"خطا در اجرای دستور {command}: {e}", exc_info=True)

        return result

    async def _restart_component(self, component_name: str) -> bool:
        """
        راه‌اندازی مجدد یک کامپوننت خاص

        Args:
            component_name: نام کامپوننت

        Returns:
            موفقیت در راه‌اندازی مجدد
        """
        try:
            if component_name == 'signal_processor':
                if self.signal_processor:
                    # توقف و راه‌اندازی مجدد
                    await self.signal_processor.stop_periodic_processing()
                    await self.signal_processor.start_periodic_processing()
                    self.running_status['components_status']['signal_processor'] = 'running'
                    logger.info("پردازش‌کننده سیگنال مجدداً راه‌اندازی شد")
                    return True

            elif component_name == 'trade_manager':
                if self.trade_manager:
                    # توقف و راه‌اندازی مجدد به‌روزرسانی قیمت
                    await self.trade_manager.stop_periodic_price_update()
                    await self.trade_manager.start_periodic_price_update()
                    self.running_status['components_status']['trade_manager'] = 'running'
                    logger.info("به‌روزرسانی قیمت در مدیریت معاملات مجدداً راه‌اندازی شد")
                    return True

            elif component_name == 'data_fetcher':
                if self.data_fetcher:
                    # راه‌اندازی مجدد
                    await self.data_fetcher.shutdown()
                    await self.data_fetcher.initialize()
                    self.running_status['components_status']['data_fetcher'] = 'running'
                    logger.info("دریافت‌کننده داده مجدداً راه‌اندازی شد")
                    return True

            elif component_name == 'exchange_client':
                if self.exchange_client:
                    # بستن و راه‌اندازی مجدد نشست
                    await self.exchange_client.close()
                    await self.exchange_client._init_session()
                    self.running_status['components_status']['exchange_client'] = 'running'
                    logger.info("نشست کلاینت صرافی مجدداً راه‌اندازی شد")
                    return True

            elif component_name == 'config_watcher':
                if hasattr(self, '_config_check_task'):
                    # لغو و راه‌اندازی مجدد تسک نظارت بر تنظیمات
                    if self._config_check_task and not self._config_check_task.done():
                        self._config_check_task.cancel()
                    self._config_check_task = asyncio.create_task(self._config_watch_loop())
                    self.running_status['components_status']['config_watcher'] = 'running'
                    logger.info("نظارت بر تنظیمات مجدداً راه‌اندازی شد")
                    return True

            logger.warning(f"کامپوننت {component_name} یافت نشد یا قابلیت راه‌اندازی مجدد ندارد")
            return False

        except Exception as e:
            logger.error(f"خطا در راه‌اندازی مجدد کامپوننت {component_name}: {e}", exc_info=True)
            return False