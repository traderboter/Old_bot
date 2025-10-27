#!/usr/bin/env python
# -*- coding: utf-8 -*-
# main.py - نقطه ورودی اصلی برای ربات معاملاتی ارز دیجیتال
# بهینه‌سازی شده برای پشتیبانی از بارگذاری پویای تنظیمات و ذخیره‌سازی داده‌ها

import asyncio
import argparse
import logging
import signal
import sys
import os
import traceback
import time
import yaml
import json
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, Optional

# --- ایمپورت‌های اصلی سیستم ---
from crypto_trading_bot import CryptoTradingBot

# تنظیم لاگر اولیه (بعداً با تنظیمات کانفیگ به‌روزرسانی می‌شود)
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
)
logger = logging.getLogger(__name__)

# متغیرهای گلوبال برای دسترسی در هندلرهای سیگنال
bot_instance = None
config = None
config_last_modified = 0
config_check_interval = 30  # بررسی تغییرات تنظیمات هر 30 ثانیه


# --- توابع کمکی ---
def ensure_directory(path: str) -> bool:
    """اطمینان از وجود یک دایرکتوری و ایجاد آن در صورت نیاز."""
    try:
        os.makedirs(path, exist_ok=True)
        return True
    except Exception as e:
        logger.error(f"خطا در ایجاد دایرکتوری {path}: {e}")
        return False


def format_time(timestamp: Optional[float] = None) -> str:
    """فرمت‌بندی زمان با قالب مناسب برای فایل‌نام."""
    timestamp = timestamp or time.time()
    return datetime.fromtimestamp(timestamp).strftime('%Y%m%d_%H%M%S')


def load_config(config_path: str) -> Dict[str, Any]:
    """بارگذاری فایل تنظیمات با پشتیبانی از YAML و JSON."""
    config_path = Path(config_path)

    if not config_path.exists():
        raise FileNotFoundError(f"فایل تنظیمات یافت نشد: {config_path}")

    try:
        if config_path.suffix in ('.yaml', '.yml'):
            return yaml.safe_load(config_path.read_text(encoding='utf-8'))
        elif config_path.suffix == '.json':
            return json.loads(config_path.read_text(encoding='utf-8'))
        else:
            raise ValueError(f"فرمت فایل تنظیمات پشتیبانی نمی‌شود: {config_path.suffix}")
    except Exception as e:
        raise ValueError(f"خطا در بارگذاری تنظیمات: {e}")


def check_config_changes(config_path: str) -> bool:
    """بررسی تغییرات در فایل تنظیمات."""
    global config_last_modified
    try:
        current_mtime = os.path.getmtime(config_path)
        if current_mtime > config_last_modified:
            config_last_modified = current_mtime
            return True
        return False
    except Exception as e:
        logger.error(f"خطا در بررسی تغییرات فایل تنظیمات: {e}")
        return False


def setup_logging(config: Dict[str, Any], verbose: bool) -> None:
    """تنظیم سیستم لاگینگ بر اساس تنظیمات و آرگومان‌های خط فرمان."""
    log_config = config.get('logging', {})
    log_level_str = log_config.get('level', 'INFO').upper()
    log_format = log_config.get('format', '%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    log_file = log_config.get('file')
    log_rotate = log_config.get('rotate', False)
    log_max_size = log_config.get('max_size', 10 * 1024 * 1024)  # 10MB پیش‌فرض
    log_backup_count = log_config.get('backup_count', 5)

    # تعیین سطح لاگ
    if verbose:
        log_level = logging.DEBUG
    else:
        log_level = getattr(logging, log_level_str, logging.INFO)

    # تنظیم لاگر اصلی
    root_logger = logging.getLogger()
    if root_logger.hasHandlers():
        root_logger.handlers.clear()  # پاک کردن هندلرهای قبلی
    root_logger.setLevel(log_level)

    # فرمتر اصلی
    formatter = logging.Formatter(log_format)

    # هندلر کنسول
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(log_level)
    console_handler.setFormatter(formatter)
    root_logger.addHandler(console_handler)

    # هندلر فایل (در صورت نیاز)
    if log_file:
        try:
            log_dir = os.path.dirname(log_file)
            if log_dir:
                ensure_directory(log_dir)

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
            logger.info(f"لاگینگ به فایل: {log_file}{' (با چرخش)' if log_rotate else ''}")

        except Exception as e:
            logger.error(f"خطا در تنظیم هندلر فایل لاگ: {e}", exc_info=True)


# --- هندلر سیگنال سیستم ---
def signal_handler(sig, frame):
    """هندلر سیگنال برای خروج تمیز ربات."""
    global bot_instance
    print(f"\nسیگنال {sig} دریافت شد، درحال خاموش شدن...")
    logger.info(f"سیگنال {sig} دریافت شد، در حال خاموش شدن...")

    # درخواست توقف ربات
    if bot_instance and hasattr(bot_instance, 'stop') and callable(bot_instance.stop):
        try:
            bot_instance.stop()
            logger.info("سیگنال توقف به ربات ارسال شد.")
        except Exception as e:
            logger.error(f"خطا در درخواست توقف ربات: {e}", exc_info=True)


# --- تابع اصلی ---
async def main():
    """تابع اصلی برنامه."""
    global bot_instance, config, config_last_modified

    # --- پردازش آرگومان‌های خط فرمان ---
    parser = argparse.ArgumentParser(description='ربات معاملاتی ارز دیجیتال')
    parser.add_argument('-c', '--config', type=str, default='config.yaml',
                        help='مسیر فایل تنظیمات (پیش‌فرض: config.yaml)')
    parser.add_argument('-v', '--verbose', action='store_true',
                        help='فعال‌سازی لاگینگ جزئیات (DEBUG)')
    parser.add_argument('--symbols', type=str,
                        help='لیست نمادهای معاملاتی با جداکننده کاما')
    parser.add_argument('--strategy', type=str,
                        help='انتخاب استراتژی خاص برای معامله')
    parser.add_argument('--backup', action='store_true',
                        help='ایجاد نسخه پشتیبان از داده‌ها قبل از شروع')
    parser.add_argument('--no-trading', action='store_true',
                        help='اجرا در حالت شبیه‌سازی (بدون معامله واقعی)')
    parser.add_argument('--no-watch-config', action='store_true',
                        help='غیرفعال کردن بررسی خودکار تغییرات فایل تنظیمات')
    parser.add_argument('--update-config', type=str,
                        help='به‌روزرسانی یک بخش از تنظیمات با مقدار JSON (فرمت: section:json_value)')

    args = parser.parse_args()

    # --- بارگذاری تنظیمات ---
    try:
        config = load_config(args.config)
        config_last_modified = os.path.getmtime(args.config)  # ثبت زمان آخرین تغییر

        # اضافه کردن بخش config_management اگر وجود ندارد
        if 'config_management' not in config:
            config['config_management'] = {
                'auto_reload': True,
                'check_interval_seconds': 30,
                'notify_changes': True,
                'backup_before_update': True
            }
            logger.info("بخش مدیریت تنظیمات به تنظیمات اضافه شد")

        # ایجاد پوشه‌های لازم
        data_dir = config.get('storage', {}).get('data_directory', 'data')
        ensure_directory(data_dir)

    except (FileNotFoundError, ValueError) as e:
        print(f"خطای بحرانی در بارگذاری تنظیمات: {e}", file=sys.stderr)
        return 1

    # --- به‌روزرسانی بخش تنظیمات از خط فرمان ---
    if args.update_config:
        try:
            section, json_value = args.update_config.split(':', 1)
            section_value = json.loads(json_value)

            if section in config:
                if isinstance(section_value, dict) and isinstance(config[section], dict):
                    # ادغام مقادیر جدید با مقادیر موجود
                    config[section].update(section_value)
                    logger.info(f"بخش '{section}' با مقادیر جدید به‌روزرسانی شد")
                else:
                    # جایگزینی کامل بخش
                    config[section] = section_value
                    logger.info(f"بخش '{section}' با مقدار جدید جایگزین شد")
            else:
                # ایجاد بخش جدید
                config[section] = section_value
                logger.info(f"بخش جدید '{section}' به تنظیمات اضافه شد")

            # ذخیره تنظیمات
            with open(args.config, 'w', encoding='utf-8') as f:
                if args.config.endswith(('.yaml', '.yml')):
                    yaml.dump(config, f, default_flow_style=False, allow_unicode=True)
                else:
                    json.dump(config, f, indent=2, ensure_ascii=False)

            logger.info(f"تنظیمات به‌روزرسانی شده در {args.config} ذخیره شد")

        except ValueError:
            logger.error(f"فرمت نامعتبر برای --update-config: {args.update_config}. فرمت صحیح: section:json_value")
            return 1
        except json.JSONDecodeError:
            logger.error(f"مقدار JSON نامعتبر در --update-config: {args.update_config}")
            return 1
        except Exception as e:
            logger.error(f"خطا در به‌روزرسانی تنظیمات: {e}", exc_info=True)
            return 1

    # --- تنظیم سیستم لاگینگ ---
    setup_logging(config, args.verbose)

    # --- ارسال لاگ شروع ---
    logger.info(f"--- شروع اجرای ربات معاملاتی --- (نسخه: {config.get('version', 'N/A')})")

    # --- پشتیبان‌گیری اولیه (اگر درخواست شده باشد) ---
    if args.backup:
        try:
            backup_dir = config.get('backup', {}).get('directory', os.path.join(data_dir, 'backups'))
            ensure_directory(backup_dir)
            backup_time = format_time()

            # شناسایی فایل‌های مهم برای پشتیبان‌گیری
            db_file = config.get('storage', {}).get('database_path', 'data/trades.db')
            data_files = [
                args.config,  # فایل تنظیمات
                db_file,  # فایل دیتابیس
                config.get('adaptive_learning', {}).get('data_file', 'data/adaptive_learning_data.json'),
                config.get('correlation_management', {}).get('data_file', 'data/correlation_data.json'),
                config.get('performance_tracking', {}).get('metrics_file', 'data/performance_metrics.json')
            ]

            # ایجاد نسخه پشتیبان به صورت دستی
            import shutil
            import zipfile

            backup_file = os.path.join(backup_dir, f"manual_backup_{backup_time}.zip")
            with zipfile.ZipFile(backup_file, 'w', zipfile.ZIP_DEFLATED) as zipf:
                for file_path in data_files:
                    if os.path.exists(file_path):
                        zipf.write(file_path, os.path.basename(file_path))
                        logger.info(f"فایل {file_path} به پشتیبان اضافه شد.")

            logger.info(f"نسخه پشتیبان دستی در {backup_file} ایجاد شد.")
        except Exception as e:
            logger.error(f"خطا در ایجاد نسخه پشتیبان دستی: {e}", exc_info=True)

    # --- اعمال تغییرات command-line روی کانفیگ ---
    if args.no_trading:
        config['trading'] = config.get('trading', {})
        config['trading']['mode'] = 'simulation'
        logger.info("معاملات واقعی غیرفعال شد (حالت شبیه‌سازی).")

    if args.symbols:
        symbols_list = [s.strip() for s in args.symbols.split(',')]
        if symbols_list:
            config['exchange'] = config.get('exchange', {})
            config['exchange']['symbols'] = symbols_list
            logger.info(f"نمادهای معاملاتی از خط فرمان: {symbols_list}")

        # --- ایجاد نمونه ربات ---
    try:
        # ایجاد نمونه ربات اصلی
        bot_instance = CryptoTradingBot(args.config)

        # تنظیم استراتژی از خط فرمان
        if args.strategy and hasattr(bot_instance, 'strategy_manager'):
            strategy_manager = getattr(bot_instance, 'strategy_manager')
            if strategy_manager and hasattr(strategy_manager, 'switch_strategy'):
                logger.info(f"در حال تنظیم استراتژی اولیه به: {args.strategy}")
                try:
                    # ایجاد تسک موقت برای فراخوانی متد async
                    loop = asyncio.get_running_loop()
                    loop.create_task(strategy_manager.switch_strategy(args.strategy))
                except Exception as e:
                    logger.error(f"خطا در تنظیم استراتژی اولیه: {e}", exc_info=True)
    except SystemExit:
        logger.critical("فراخوانی SystemExit در طول راه‌اندازی ربات.")
        return 1
    except Exception as e:
        logger.critical(f"خطا در راه‌اندازی ربات: {e}", exc_info=True)
        return 1

        # --- تنظیم هندلرهای سیگنال سیستم ---
    loop = asyncio.get_running_loop()
    for sig in (signal.SIGINT, signal.SIGTERM):
        try:
            # استفاده از lambda برای اطمینان از ارسال صحیح سیگنال به هندلر
            loop.add_signal_handler(sig, lambda s=sig: signal_handler(s, None))
        except NotImplementedError:
            # Fallback برای سیستم‌هایی که add_signal_handler ندارند
            signal.signal(sig, signal_handler)

    # تنظیم مستقیم watch_config در ربات
    if args.no_watch_config:
        if 'config_management' in config:
            config['config_management']['auto_reload'] = False
            logger.info("نظارت بر تغییرات فایل تنظیمات در ربات غیرفعال شد (--no-watch-config)")

    # --- اجرای ربات ---
    bot_exit_code = 0
    logger.info("در حال شروع حلقه اصلی ربات...")
    try:
        # اجرای متد run ربات
        success = await bot_instance.run()
        bot_exit_code = 0 if success else 1
    except KeyboardInterrupt:
        logger.info("KeyboardInterrupt در طول اجرای ربات دریافت شد. در حال توقف...")
        # متد stop قبلاً در signal_handler فراخوانی شده، اما برای اطمینان
        if bot_instance and hasattr(bot_instance, 'stop'):
            bot_instance.stop()
        bot_exit_code = 1
    except Exception as e:
        logger.critical(f"خطای پیش‌بینی نشده در حلقه اجرای ربات: {e}", exc_info=True)
        if bot_instance and hasattr(bot_instance, 'stop'):
            bot_instance.stop()
        bot_exit_code = 1
    finally:
        logger.info("حلقه اجرای ربات به پایان رسید.")

    logger.info("تابع اصلی به پایان رسید.")
    return bot_exit_code


# --- اجرای برنامه اصلی ---
if __name__ == "__main__":
    exit_code = 1  # کد خروج پیش‌فرض در صورت خطا
    try:
        # اجرای حلقه اصلی asyncio
        exit_code = asyncio.run(main())
        logger.info(f"برنامه با کد خروج {exit_code} به پایان رسید.")
    except KeyboardInterrupt:
        # اگر Ctrl+C قبل از اجرای کامل main یا حین shutdown زده شود
        print("\nبرنامه با فشار کلید قطع شد (KeyboardInterrupt در سطح بالا).")
        try:
            logger.warning("برنامه با فشار کلید قطع شد (KeyboardInterrupt در سطح بالا).")
        except:
            pass
        exit_code = 1
    except SystemExit as e:
        # اجازه به خروج عادی با کدی که برنامه تعیین کرده
        exit_code = e.code
        try:
            logger.info(f"برنامه با SystemExit و کد {exit_code} خارج شد.")
        except:
            pass
    except Exception as e:
        # خطاهای پیش‌بینی نشده در سطح asyncio.run
        print(f"\nخطای فاجعه‌بار در طول asyncio.run: {e}", file=sys.stderr)
        traceback.print_exc()
        try:
            logger.critical(f"خطای فاجعه‌بار در طول asyncio.run: {e}", exc_info=True)
        except:
            pass
        exit_code = 1
    finally:
        # تلاش برای بستن تمیز هندلرهای لاگ (مخصوصاً فایل لاگ)
        try:
            logging.shutdown()
        except:
            pass
        sys.exit(exit_code)