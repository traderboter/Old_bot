"""
ماژول market_data_fetcher.py: دریافت داده‌های بازار از منابع مختلف
با استفاده از ExchangeClient و کش‌کردن هوشمند با پشتیبانی از Delta Updates.
"""

import logging
import asyncio
import time
import os
import redis.asyncio as redis
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Union, Tuple, Set, TypeVar
from concurrent.futures import ThreadPoolExecutor
import random

import pandas as pd
import numpy as np
from functools import lru_cache
import json
from io import StringIO

# استفاده از کلاینت جدید صرافی
from exchange_client import ExchangeClient

# تنظیم لاگر
logger = logging.getLogger(__name__)

# تعریف نوع برای اشاره به DataFrame
DataFrame = TypeVar('DataFrame', bound=pd.DataFrame)


class SmartCache:
    """سیستم کش هوشمند با مدیریت انقضا و بهینه‌سازی مصرف منابع"""

    def __init__(self, config: Dict[str, Any]):
        """
        مقداردهی اولیه سیستم کش

        Args:
            config: تنظیمات کش از فایل کانفیگ
        """
        self.config = config
        self.cache_settings = config.get('data_fetching', {}).get('cache', {})

        # تعریف لاگر
        self.logger = logging.getLogger(__name__)

        # تنظیمات کش حافظه
        self.use_memory_cache = self.cache_settings.get('use_memory_cache', True)
        self.max_memory_items = self.cache_settings.get('max_memory_items', 1000)
        self.memory_cache: Dict[
            str, Dict[str, Any]] = {}  # {key: {'data': data, 'timestamp': time, 'access_count': count}}
        self.memory_cache_hits = 0
        self.memory_cache_misses = 0

        # تنظیمات کش Redis
        self.use_redis_cache = self.cache_settings.get('use_redis', False)
        self.redis_client = None
        self._thread_executor = ThreadPoolExecutor(max_workers=4)  # برای عملیات I/O در Redis
        if self.use_redis_cache:
            self._init_redis()

        # LRU (Least Recently Used) tracking
        self.access_order: List[str] = []  # لیست کلیدها به ترتیب آخرین دسترسی

        # تنظیمات زمان انقضا
        self.cache_expiry_seconds = self.cache_settings.get('expiry_seconds', {})
        self.smart_caching = self.config.get('data_fetching', {}).get('delta_updates', {}).get('smart_caching', True)

        # آمار کش
        self.stats = {
            "memory_hits": 0,
            "memory_misses": 0,
            "redis_hits": 0,
            "redis_misses": 0,
            "total_keys": 0,
            "cleanup_operations": 0,
            "expired_keys_removed": 0
        }

        self.logger.info(
            f"سیستم کش هوشمند راه‌اندازی شد. کش حافظه: {self.use_memory_cache}، Redis: {self.use_redis_cache}، انقضای هوشمند: {self.smart_caching}")

    def update_config(self, config: Dict[str, Any]) -> None:
        """
        به‌روزرسانی تنظیمات در زمان اجرا

        Args:
            config: تنظیمات جدید
        """
        try:
            old_config = self.config
            self.config = config

            # به‌روزرسانی تنظیمات کش
            new_cache_settings = config.get('data_fetching', {}).get('cache', {})
            old_cache_settings = self.cache_settings
            self.cache_settings = new_cache_settings

            # بررسی تغییرات در تنظیمات اصلی کش
            new_use_memory_cache = new_cache_settings.get('use_memory_cache', True)
            new_use_redis = new_cache_settings.get('use_redis', False)
            new_max_memory_items = new_cache_settings.get('max_memory_items', 1000)
            new_smart_caching = config.get('data_fetching', {}).get('delta_updates', {}).get('smart_caching', True)

            # بررسی تغییر در حالت کش حافظه
            if new_use_memory_cache != self.use_memory_cache:
                self.use_memory_cache = new_use_memory_cache
                self.logger.info(f"وضعیت کش حافظه تغییر کرد: {self.use_memory_cache}")

                # اگر کش حافظه غیرفعال شد، پاکسازی کنیم
                if not self.use_memory_cache:
                    self.memory_cache.clear()
                    self.access_order.clear()
                    self.logger.info("کش حافظه پاکسازی شد")

            # بررسی تغییر در حداکثر موارد کش حافظه
            if new_max_memory_items != self.max_memory_items:
                self.max_memory_items = new_max_memory_items
                self.logger.info(f"حداکثر موارد کش حافظه تغییر کرد: {self.max_memory_items}")
                # اگر حداکثر کاهش یافته، پاکسازی انجام دهیم
                if len(self.memory_cache) > self.max_memory_items:
                    self._cleanup_memory_cache()

            # بررسی تغییر در حالت کش Redis
            if new_use_redis != self.use_redis_cache:
                self.use_redis_cache = new_use_redis
                self.logger.info(f"وضعیت کش Redis تغییر کرد: {self.use_redis_cache}")

                # اگر Redis فعال شده، اتصال را برقرار کنیم
                if self.use_redis_cache and self.redis_client is None:
                    self._init_redis()

            # بررسی تغییر در تنظیمات Redis
            if self.use_redis_cache and new_cache_settings.get('redis', {}) != old_cache_settings.get('redis', {}):
                self.logger.info("تنظیمات Redis تغییر کرد، اتصال مجدد...")
                self._init_redis()

            # بررسی تغییر در زمان‌های انقضا
            new_expiry_seconds = new_cache_settings.get('expiry_seconds', {})
            if new_expiry_seconds != self.cache_expiry_seconds:
                self.cache_expiry_seconds = new_expiry_seconds
                self.logger.info("زمان‌های انقضای کش به‌روزرسانی شدند")

            # بررسی تغییر در حالت کش هوشمند
            if new_smart_caching != self.smart_caching:
                self.smart_caching = new_smart_caching
                self.logger.info(f"وضعیت کش هوشمند تغییر کرد: {self.smart_caching}")

            self.logger.info("تنظیمات کش هوشمند با موفقیت به‌روزرسانی شدند")

        except Exception as e:
            self.logger.error(f"خطا در به‌روزرسانی تنظیمات کش هوشمند: {e}", exc_info=True)

    def _init_redis(self) -> None:
        """راه‌اندازی اتصال Redis"""
        try:
            redis_settings = self.cache_settings.get('redis', {})
            redis_host = redis_settings.get('host', 'localhost')
            redis_port = redis_settings.get('port', 6379)
            redis_db = redis_settings.get('db', 0)
            redis_password = redis_settings.get('password')
            redis_path = redis_settings.get('path')

            # اتصال قبلی را ببندیم اگر وجود دارد
            if self.redis_client:
                asyncio.create_task(self.redis_client.close())
                self.redis_client = None

            # اتصال با Unix socket یا TCP
            if redis_path and os.path.exists(redis_path):
                self.redis_client = redis.Redis(
                    unix_socket_path=redis_path,
                    db=redis_db,
                    password=redis_password,
                    decode_responses=False,
                    socket_timeout=5.0,
                    socket_connect_timeout=5.0
                )
                self.logger.info(f"کش Redis با استفاده از سوکت Unix راه‌اندازی شد: {redis_path}")
            else:
                self.redis_client = redis.Redis(
                    host=redis_host,
                    port=redis_port,
                    db=redis_db,
                    password=redis_password,
                    decode_responses=False,
                    socket_timeout=5.0,
                    socket_connect_timeout=5.0
                )
                self.logger.info(f"کش Redis با استفاده از TCP راه‌اندازی شد: {redis_host}:{redis_port}")

            # تست اتصال
            asyncio.create_task(self._test_redis_connection())

        except Exception as e:
            self.logger.error(f"خطا در راه‌اندازی Redis: {e}", exc_info=True)
            self.use_redis_cache = False
            self.redis_client = None

    async def _test_redis_connection(self) -> None:
        """تست اتصال Redis"""
        try:
            if self.redis_client:
                await self.redis_client.ping()
                self.logger.info("تست اتصال Redis موفقیت‌آمیز بود")
        except Exception as e:
            self.logger.error(f"تست اتصال Redis ناموفق بود: {e}", exc_info=True)
            self.use_redis_cache = False
            self.redis_client = None

    def _get_smart_expiry(self, timeframe: str) -> int:
        """
        محاسبه زمان انقضای هوشمند بر اساس تایم‌فریم

        Args:
            timeframe: تایم‌فریم ('1m', '5m', '1h', etc.)

        Returns:
            زمان انقضا بر حسب ثانیه
        """
        if not self.smart_caching:
            # استفاده از زمان ثابت از کانفیگ
            return self.cache_expiry_seconds.get(timeframe, 3600)

        # محاسبه زمان تا کندل بعدی
        current_time = time.time()
        seconds_in_timeframe = self._timeframe_to_seconds(timeframe)

        # زمان شروع کندل فعلی
        current_candle_start = int(current_time / seconds_in_timeframe) * seconds_in_timeframe

        # زمان شروع کندل بعدی
        next_candle_start = current_candle_start + seconds_in_timeframe

        # زمان باقی‌مانده تا کندل بعدی + 10 ثانیه اضافه
        expiry = int(next_candle_start - current_time) + 10

        # محدودیت‌های منطقی
        return max(60, min(expiry, 86400))  # بین 1 دقیقه تا 1 روز

    @staticmethod
    def _timeframe_to_seconds(timeframe: str) -> int:
        """تبدیل تایم‌فریم به ثانیه"""
        try:
            if 'm' in timeframe:
                return int(timeframe.replace('m', '')) * 60
            elif 'h' in timeframe:
                return int(timeframe.replace('h', '')) * 3600
            elif 'd' in timeframe:
                return int(timeframe.replace('d', '')) * 86400
            elif 'w' in timeframe:
                return int(timeframe.replace('w', '')) * 604800
            else:
                return 60  # پیش‌فرض 1 دقیقه
        except ValueError:
            return 3600  # پیش‌فرض 1 ساعت در صورت خطا

    def _update_access_tracking(self, key: str) -> None:
        """
        به‌روزرسانی ردیابی LRU برای کلید

        Args:
            key: کلید مورد دسترسی
        """
        # حذف کلید از لیست قبلی (اگر وجود دارد)
        if key in self.access_order:
            self.access_order.remove(key)

        # افزودن به انتهای لیست (جدیدترین دسترسی)
        self.access_order.append(key)

        # آپدیت شمارنده دسترسی
        if key in self.memory_cache:
            self.memory_cache[key]['access_count'] = self.memory_cache[key].get('access_count', 0) + 1

    def _cleanup_memory_cache(self) -> None:
        """پاکسازی کش حافظه بر اساس LRU"""
        if len(self.memory_cache) <= self.max_memory_items:
            return

        # تعداد آیتم‌های اضافی که باید حذف شوند
        items_to_remove = len(self.memory_cache) - self.max_memory_items

        # حذف قدیمی‌ترین آیتم‌ها بر اساس دسترسی
        removed = 0
        while removed < items_to_remove and self.access_order:
            oldest_key = self.access_order.pop(0)
            if oldest_key in self.memory_cache:
                del self.memory_cache[oldest_key]
                removed += 1
                self.logger.debug(f"پاکسازی LRU: کلید {oldest_key} از کش حافظه حذف شد")

        # آپدیت آمار
        self.stats["cleanup_operations"] += 1
        self.stats["expired_keys_removed"] += removed

    async def get(self, key: str, expiry_time: Optional[int] = None) -> Optional[Any]:
        """
        دریافت داده از کش

        Args:
            key: کلید کش
            expiry_time: زمان انقضا (اختیاری)

        Returns:
            داده کش شده یا None
        """
        # بررسی کش حافظه
        if self.use_memory_cache:
            memory_result = self._get_from_memory(key, expiry_time)
            if memory_result is not None:
                self.memory_cache_hits += 1
                self.stats["memory_hits"] += 1
                return memory_result
            self.memory_cache_misses += 1
            self.stats["memory_misses"] += 1

        # بررسی کش Redis
        if self.use_redis_cache and self.redis_client:
            redis_result = await self._get_from_redis(key)
            if redis_result is not None:
                # ذخیره در کش حافظه برای دسترسی‌های بعدی
                if self.use_memory_cache:
                    self._store_in_memory(key, redis_result)
                self.stats["redis_hits"] += 1
                return redis_result
            else:
                self.stats["redis_misses"] += 1

        return None

    def _get_from_memory(self, key: str, expiry_time: Optional[int] = None) -> Optional[Any]:
        """
        دریافت از کش حافظه با بررسی انقضا

        Args:
            key: کلید کش
            expiry_time: زمان انقضا (اختیاری)

        Returns:
            داده یا None در صورت انقضا یا عدم وجود
        """
        if key not in self.memory_cache:
            return None

        cached_item = self.memory_cache[key]
        current_time = time.time()
        age = current_time - cached_item['timestamp']

        # اگر expiry_time تعیین نشده، از مقدار smart expiry استفاده کن
        max_age = expiry_time if expiry_time is not None else self._get_cache_expiry_for_key(key)

        if age > max_age:
            # منقضی شده - حذف از کش
            del self.memory_cache[key]
            # حذف از لیست دسترسی
            if key in self.access_order:
                self.access_order.remove(key)
            return None

        # به‌روزرسانی ردیابی دسترسی
        self._update_access_tracking(key)

        return cached_item['data']

    async def _get_from_redis(self, key: str) -> Optional[Any]:
        """
        دریافت از کش Redis

        Args:
            key: کلید کش

        Returns:
            داده یا None
        """
        if not self.redis_client:
            return None

        try:
            # دریافت داده به صورت async در ترد جداگانه
            loop = asyncio.get_event_loop()
            cached_bytes = await loop.run_in_executor(
                self._thread_executor,
                lambda: self.redis_client.get(key)
            )

            if cached_bytes is None:
                return None

            # تبدیل بایت‌های JSON به داده
            json_string = cached_bytes.decode('utf-8')

            # تشخیص نوع داده (DataFrame یا دیگر)
            try:
                if '"index"' in json_string and '"columns"' in json_string:
                    # احتمالا DataFrame است
                    df = pd.read_json(StringIO(json_string), orient='split')

                    # تنظیم مجدد ایندکس زمان
                    if 'timestamp' in df.columns:
                        df['timestamp'] = pd.to_datetime(df['timestamp'], errors='coerce', utc=True)
                        df = df.set_index('timestamp')

                    df.sort_index(inplace=True)
                    return df
                else:
                    # داده عمومی
                    return json.loads(json_string)

            except Exception as e:
                self.logger.error(f"خطا در تجزیه داده Redis برای {key}: {e}")
                # در صورت خطا، داده را نامعتبر فرض کن
                await loop.run_in_executor(self._thread_executor, lambda: self.redis_client.delete(key))
                return None

        except Exception as e:
            self.logger.error(f"خطا در دریافت {key} از Redis: {e}")
            return None

    def _get_cache_expiry_for_key(self, key: str) -> int:
        """
        تعیین مدت اعتبار کش بر اساس کلید

        Args:
            key: کلید کش

        Returns:
            مدت انقضا (ثانیه)
        """
        # استخراج تایم‌فریم از کلید (فرض: فرمت 'ohlcv:SYMBOL:TIMEFRAME')
        parts = key.split(':')
        if len(parts) >= 3 and parts[0] == 'ohlcv':
            timeframe = parts[2]
            return self._get_smart_expiry(timeframe)

        # برای سایر انواع کلیدها، از مقدار پیش‌فرض استفاده کن
        return 300  # 5 دقیقه پیش‌فرض

    def _store_in_memory(self, key: str, data: Any) -> None:
        """
        ذخیره داده در کش حافظه

        Args:
            key: کلید کش
            data: داده برای ذخیره
        """
        if not self.use_memory_cache:
            return

        # اگر حافظه پر است، پاکسازی کن
        self._cleanup_memory_cache()

        # ذخیره‌سازی
        self.memory_cache[key] = {
            'data': data,
            'timestamp': time.time(),
            'access_count': 0
        }

        # به‌روزرسانی ردیابی دسترسی
        self._update_access_tracking(key)

        # آپدیت آمار
        self.stats["total_keys"] = len(self.memory_cache)

    async def set(self, key: str, data: Any, expiry_seconds: Optional[int] = None) -> None:
        """
        ذخیره داده در کش

        Args:
            key: کلید کش
            data: داده برای ذخیره
            expiry_seconds: مدت انقضا (اختیاری)
        """
        # محاسبه مدت انقضا
        if expiry_seconds is None:
            expiry_seconds = self._get_cache_expiry_for_key(key)

        # ذخیره در کش حافظه
        self._store_in_memory(key, data)

        # ذخیره در Redis
        if self.use_redis_cache and self.redis_client:
            await self._store_in_redis(key, data, expiry_seconds)

    async def _store_in_redis(self, key: str, data: Any, expiry_seconds: int) -> None:
        """
        ذخیره داده در کش Redis

        Args:
            key: کلید کش
            data: داده برای ذخیره
            expiry_seconds: مدت انقضا
        """
        if not self.redis_client:
            return

        try:
            # تبدیل داده به JSON
            if hasattr(data, 'to_json'):  # احتمالا DataFrame
                json_data = data.reset_index().to_json(orient='split', date_format='iso').encode('utf-8')
            else:
                json_data = json.dumps(data).encode('utf-8')

            # ذخیره در Redis
            loop = asyncio.get_event_loop()
            await loop.run_in_executor(
                self._thread_executor,
                lambda: self.redis_client.setex(key, expiry_seconds, json_data)
            )

        except Exception as e:
            self.logger.error(f"خطا در ذخیره {key} در Redis: {e}")

    async def delete(self, key: str) -> None:
        """
        حذف کلید از کش

        Args:
            key: کلید کش
        """
        # حذف از کش حافظه
        if key in self.memory_cache:
            del self.memory_cache[key]
            if key in self.access_order:
                self.access_order.remove(key)

        # حذف از Redis
        if self.use_redis_cache and self.redis_client:
            try:
                loop = asyncio.get_event_loop()
                await loop.run_in_executor(self._thread_executor, lambda: self.redis_client.delete(key))
            except Exception as e:
                self.logger.error(f"خطا در حذف {key} از Redis: {e}")

    async def clear_all(self) -> None:
        """پاکسازی تمام کش‌ها"""
        # پاکسازی کش حافظه
        self.memory_cache.clear()
        self.access_order.clear()

        # پاکسازی کش Redis (فقط کلیدهای مرتبط با این سرویس)
        if self.use_redis_cache and self.redis_client:
            try:
                loop = asyncio.get_event_loop()
                # حذف کلیدهای با پیشوند 'ohlcv:'
                pattern = 'ohlcv:*'
                keys = await loop.run_in_executor(self._thread_executor, lambda: self.redis_client.keys(pattern))
                if keys:
                    await loop.run_in_executor(self._thread_executor, lambda: self.redis_client.delete(*keys))
                    self.logger.info(f"{len(keys)} کلید از کش Redis پاکسازی شد")
            except Exception as e:
                self.logger.error(f"خطا در پاکسازی کش Redis: {e}")

        # ریست آمار
        self.stats["total_keys"] = 0
        self.stats["cleanup_operations"] += 1

    def get_stats(self) -> Dict[str, Any]:
        """
        دریافت آمار عملکرد کش

        Returns:
            دیکشنری آمار
        """
        hit_ratio = 0
        total_operations = self.memory_cache_hits + self.memory_cache_misses
        if total_operations > 0:
            hit_ratio = self.memory_cache_hits / total_operations

        memory_stats = {
            'memory_cache_enabled': self.use_memory_cache,
            'memory_cache_items': len(self.memory_cache),
            'memory_cache_hits': self.memory_cache_hits,
            'memory_cache_misses': self.memory_cache_misses,
            'memory_hit_ratio': hit_ratio,
            'redis_cache_enabled': self.use_redis_cache and self.redis_client is not None,
            'smart_caching': self.smart_caching
        }

        # ترکیب آمار داخلی و آمار ردیابی شده
        combined_stats = {**memory_stats, **self.stats}

        return combined_stats


class MarketDataFetcher:
    """
    کلاس بهینه‌سازی شده دریافت داده‌های بازار با کش هوشمند و مدیریت بهتر منابع
    """

    def __init__(self, config: Dict[str, Any], exchange_client: ExchangeClient):
        """
        مقداردهی اولیه با تنظیمات و وهله ExchangeClient خارجی

        Args:
            config: دیکشنری تنظیمات
            exchange_client: وهله ExchangeClient از پیش ساخته شده
        """
        self.config = config
        self.data_config = config.get('data_fetching', {})
        self.exchange_client = exchange_client

        # تایم‌فریم‌های مورد نیاز
        self.timeframes: List[str] = self.data_config.get('timeframes', ['5m', '15m', '1h', '4h'])

        # راه‌اندازی کش هوشمند
        self.cache = SmartCache(config)

        # مدیریت دریافت موازی
        self.max_concurrent_fetches = self.data_config.get('max_concurrent_fetches', 10)
        self.fetch_semaphore = asyncio.Semaphore(self.max_concurrent_fetches)

        # انتظار بین درخواست‌ها برای مدیریت Rate Limit
        self.delay_between_requests = 1.0 / self.data_config.get('rate_limits', {}).get('requests_per_second', 5)
        self.last_request_time = 0

        # قابلیت دریافت‌های دلتا
        self.use_delta_updates = self.data_config.get('delta_updates', {}).get('enabled', True)

        # داده‌های ساختگی برای تست
        self.use_mock_data = self.data_config.get('use_mock_data', False)

        # کش نمادهای فعال
        self._active_symbols_cache: List[str] = []
        self._active_symbols_last_update = 0
        self._active_symbols_cache_ttl = 3600  # 1 ساعت

        # مکانیزم backoff برای کنترل خطاها
        self._error_backoff = {
            'count': 0,
            'max_delay': 60,  # حداکثر تاخیر 60 ثانیه
            'base_delay': 2,  # تاخیر پایه 2 ثانیه
        }

        # ردیابی شکاف‌ها
        self._gap_tracking = {
            "detected_gaps": {},  # {symbol: {timeframe: [(start_ts, end_ts, missing_count), ...], ...}, ...}
            "filled_gaps": {},  # {symbol: {timeframe: count, ...}, ...}
            "total_detected": 0,
            "total_filled": 0,
            "last_check_time": {}  # {timeframe: timestamp, ...}
        }

        # سیستم اولویت‌بندی نمادها
        self._symbol_priorities = {}  # {symbol: priority_score, ...}
        self._high_priority_symbols = set()
        self._last_priority_calculation = 0

        # آمار کلی
        self._stats = {
            "total_requests": 0,
            "successful_requests": 0,
            "failed_requests": 0,
            "cache_hits": 0,
            "total_candles_fetched": 0,
            "errors_by_type": {},
            "last_error_time": {},
            "average_response_time_ms": 0,
            "last_fetch_time": 0  # زمان آخرین دریافت داده
        }

        self.logger = logging.getLogger(__name__)
        self.logger.info(
            f"دریافت‌کننده داده‌های بازار با {len(self.timeframes)} تایم‌فریم راه‌اندازی شد، "
            f"حداکثر درخواست همزمان: {self.max_concurrent_fetches}، "
            f"به‌روزرسانی‌های دلتا: {self.use_delta_updates}"
        )

        # شروع تسک‌های نگهداری
        self._maintenance_tasks = []
        self._start_maintenance_tasks()

    def update_config(self, new_config: Dict[str, Any]) -> None:
        """
        به‌روزرسانی تنظیمات در زمان اجرا

        Args:
            new_config: تنظیمات جدید
        """
        try:
            old_config = self.config
            old_data_config = self.data_config

            # به‌روزرسانی تنظیمات اصلی
            self.config = new_config
            self.data_config = new_config.get('data_fetching', {})

            # به‌روزرسانی کش هوشمند
            self.cache.update_config(new_config)

            # بررسی تغییرات در پارامترهای مهم

            # تایم‌فریم‌ها
            new_timeframes = self.data_config.get('timeframes', ['5m', '15m', '1h', '4h'])
            if set(new_timeframes) != set(self.timeframes):
                old_timeframes = self.timeframes.copy()
                self.timeframes = new_timeframes
                self.logger.info(f"تایم‌فریم‌ها تغییر کردند از {old_timeframes} به {new_timeframes}")

            # حداکثر درخواست همزمان
            new_max_concurrent = self.data_config.get('max_concurrent_fetches', 10)
            if new_max_concurrent != self.max_concurrent_fetches:
                old_max_concurrent = self.max_concurrent_fetches
                self.max_concurrent_fetches = new_max_concurrent

                # به‌روزرسانی سمافور
                self.fetch_semaphore = asyncio.Semaphore(self.max_concurrent_fetches)

                self.logger.info(f"حداکثر درخواست همزمان تغییر کرد از {old_max_concurrent} به {new_max_concurrent}")

            # نرخ درخواست
            new_rate_limit = self.data_config.get('rate_limits', {}).get('requests_per_second', 5)
            new_delay = 1.0 / new_rate_limit
            if new_delay != self.delay_between_requests:
                old_delay = self.delay_between_requests
                self.delay_between_requests = new_delay
                self.logger.info(f"فاصله درخواست‌ها تغییر کرد از {old_delay:.4f}s به {new_delay:.4f}s (نرخ: {new_rate_limit} در ثانیه)")

            # دریافت‌های دلتا
            new_delta_updates = self.data_config.get('delta_updates', {}).get('enabled', True)
            if new_delta_updates != self.use_delta_updates:
                self.use_delta_updates = new_delta_updates
                self.logger.info(f"به‌روزرسانی‌های دلتا {'فعال' if self.use_delta_updates else 'غیرفعال'} شد")

            # داده‌های ساختگی
            new_mock_data = self.data_config.get('use_mock_data', False)
            if new_mock_data != self.use_mock_data:
                self.use_mock_data = new_mock_data
                self.logger.info(f"داده‌های ساختگی {'فعال' if self.use_mock_data else 'غیرفعال'} شد")

            # TTL کش نمادهای فعال
            new_symbols_ttl = self.data_config.get('active_symbols_cache_ttl', 3600)
            if new_symbols_ttl != self._active_symbols_cache_ttl:
                self._active_symbols_cache_ttl = new_symbols_ttl
                self.logger.info(f"TTL کش نمادهای فعال تغییر کرد به {self._active_symbols_cache_ttl} ثانیه")

            # اگر max_symbols تغییر کرده، کش نمادهای فعال را به‌روزرسانی کنیم
            old_max_symbols = old_data_config.get('max_symbols', 0)
            new_max_symbols = self.data_config.get('max_symbols', 0)
            if new_max_symbols != old_max_symbols and self._active_symbols_cache:
                self.logger.info("تغییر در حداکثر تعداد نمادها: اعمال به‌روزرسانی در لیست نمادها")
                # برنامه‌ریزی یک به‌روزرسانی نمادهای فعال
                asyncio.create_task(self.get_active_symbols(force_refresh=True))

            # به‌روزرسانی تسک‌های نگهداری اگر تنظیمات مرتبط تغییر کرده‌اند
            refresh_tasks = False

            # بررسی تغییر در تنظیمات مرتبط با تسک‌ها
            if (
                old_data_config.get('delta_updates', {}).get('enabled') != self.data_config.get('delta_updates', {}).get('enabled') or
                old_data_config.get('cache', {}).get('use_redis') != self.data_config.get('cache', {}).get('use_redis')
            ):
                refresh_tasks = True

            if refresh_tasks:
                self.logger.info("بازسازی تسک‌های نگهداری به دلیل تغییر تنظیمات...")
                self._restart_maintenance_tasks()

            self.logger.info("تنظیمات دریافت‌کننده داده‌های بازار با موفقیت به‌روزرسانی شدند")

        except Exception as e:
            self.logger.error(f"خطا در به‌روزرسانی تنظیمات دریافت‌کننده داده‌های بازار: {e}", exc_info=True)

    def _start_maintenance_tasks(self):
        """راه‌اندازی تسک‌های نگهداری"""
        # تسک پاکسازی کش
        cache_cleanup_task = asyncio.create_task(self._cache_cleanup_loop())
        # تسک بررسی شکاف‌ها
        gap_check_task = asyncio.create_task(self._gap_check_loop())

        # نگهداری ارجاع به تسک‌ها برای لغو بعدی
        self._maintenance_tasks = [cache_cleanup_task, gap_check_task]

    def _restart_maintenance_tasks(self):
        """بازسازی تسک‌های نگهداری"""
        # لغو تسک‌های موجود
        for task in self._maintenance_tasks:
            if not task.done():
                task.cancel()

        # پاکسازی لیست
        self._maintenance_tasks = []

        # شروع مجدد تسک‌ها
        self._start_maintenance_tasks()
        self.logger.info("تسک‌های نگهداری مجدداً راه‌اندازی شدند")

    async def _cache_cleanup_loop(self):
        """پاکسازی دوره‌ای کش"""
        while True:
            try:
                await asyncio.sleep(600)  # هر 10 دقیقه
                await self.cache.clear_all()
                self.logger.info("پاکسازی دوره‌ای کش انجام شد")
            except asyncio.CancelledError:
                break
            except Exception as e:
                self.logger.error(f"خطا در حلقه پاکسازی کش: {e}")
                await asyncio.sleep(60)  # انتظار کوتاه در صورت خطا

    async def _gap_check_loop(self):
        """بررسی دوره‌ای شکاف‌ها"""
        while True:
            try:
                await asyncio.sleep(3600)  # هر 1 ساعت

                # انتخاب یک تایم‌فریم تصادفی برای بررسی شکاف
                timeframe = random.choice(self.timeframes)

                # انتخاب چند نماد برای بررسی
                symbols = await self.get_active_symbols()
                sample_size = min(20, len(symbols))
                sample_symbols = random.sample(symbols, sample_size)

                # بررسی شکاف‌ها
                gaps_found = await self._check_gaps_for_symbols(sample_symbols, timeframe)

                self.logger.info(
                    f"بررسی شکاف برای {timeframe} کامل شد: {gaps_found} شکاف در {sample_size} نماد یافت شد"
                )

                # ذخیره زمان آخرین بررسی
                self._gap_tracking["last_check_time"][timeframe] = time.time()

            except asyncio.CancelledError:
                break
            except Exception as e:
                self.logger.error(f"خطا در حلقه بررسی شکاف: {e}")
                await asyncio.sleep(300)  # انتظار کوتاه در صورت خطا

    async def _check_gaps_for_symbols(self, symbols: List[str], timeframe: str) -> int:
        """بررسی شکاف‌ها برای یک لیست از نمادها و یک تایم‌فریم"""
        total_gaps_found = 0

        for symbol in symbols:
            try:
                # دریافت داده‌های موجود
                data = await self.get_historical_data(symbol, timeframe, limit=500)

                if data is None or len(data) < 2:
                    continue

                # مرتب‌سازی بر اساس timestamp
                data = data.sort_index()

                # محاسبه مدت زمان تایم‌فریم
                tf_duration_ms = self._get_timeframe_ms(timeframe)

                # بررسی شکاف‌ها
                gaps = []
                for i in range(1, len(data.index)):
                    time_diff = (data.index[i] - data.index[i - 1]).total_seconds() * 1000

                    # اگر فاصله بین دو کندل بیشتر از انتظار است
                    if time_diff > tf_duration_ms * 1.5:
                        missing_count = int(time_diff / tf_duration_ms) - 1
                        if missing_count > 0:
                            start_ts = data.index[i - 1].timestamp() * 1000 + tf_duration_ms
                            end_ts = data.index[i].timestamp() * 1000 - tf_duration_ms
                            gaps.append((start_ts, end_ts, missing_count))

                if gaps:
                    # ذخیره شکاف‌های یافت شده
                    if symbol not in self._gap_tracking["detected_gaps"]:
                        self._gap_tracking["detected_gaps"][symbol] = {}
                    if timeframe not in self._gap_tracking["detected_gaps"][symbol]:
                        self._gap_tracking["detected_gaps"][symbol][timeframe] = []

                    self._gap_tracking["detected_gaps"][symbol][timeframe].extend(gaps)
                    self._gap_tracking["total_detected"] += len(gaps)
                    total_gaps_found += len(gaps)

                    # برنامه‌ریزی پر کردن شکاف‌ها
                    for start_ts, end_ts, missing_count in gaps:
                        # ایجاد تسک پر کردن شکاف با تاخیر تصادفی
                        delay = random.uniform(1, 10)  # تاخیر بین 1 تا 10 ثانیه
                        asyncio.create_task(
                            self._fill_gap_task(symbol, timeframe, start_ts, end_ts, missing_count, delay))

            except Exception as e:
                self.logger.error(f"خطا در بررسی شکاف‌های {symbol} [{timeframe}]: {e}")

        return total_gaps_found

    async def _fill_gap_task(self, symbol: str, timeframe: str, start_ts: int, end_ts: int, missing_count: int,
                             delay: float):
        """تسک پر کردن شکاف با تاخیر"""
        try:
            # تاخیر برای جلوگیری از فشار زیاد به API
            await asyncio.sleep(delay)

            # دریافت داده‌های شکاف
            self.logger.info(f"پر کردن شکاف برای {symbol} [{timeframe}]: {missing_count} کندل گمشده")
            start_time_s = start_ts / 1000

            # محاسبه حداکثر تعداد کندل برای دریافت (کمی بیشتر از تعداد گمشده)
            limit = min(missing_count + 2, 50)

            # دریافت داده‌ها
            async with self.fetch_semaphore:
                await self._wait_for_rate_limit()
                gap_data = await self.exchange_client.fetch_ohlcv(symbol, timeframe, since=int(start_time_s),
                                                                  limit=limit)

            if gap_data is not None and not gap_data.empty:
                self.logger.info(f"شکاف {symbol} [{timeframe}] با موفقیت پر شد: {len(gap_data)} کندل")

                # آپدیت آمار
                if symbol not in self._gap_tracking["filled_gaps"]:
                    self._gap_tracking["filled_gaps"][symbol] = {}
                if timeframe not in self._gap_tracking["filled_gaps"][symbol]:
                    self._gap_tracking["filled_gaps"][symbol][timeframe] = 0

                self._gap_tracking["filled_gaps"][symbol][timeframe] += 1
                self._gap_tracking["total_filled"] += 1

                # کش داده‌های جدید
                full_data = await self.get_historical_data(symbol, timeframe, force_refresh=True)

                # اضافه کردن به نمادهای با اولویت بالا
                self._high_priority_symbols.add(symbol)

        except Exception as e:
            self.logger.error(f"خطا در پر کردن شکاف برای {symbol} [{timeframe}]: {e}")

    async def initialize(self) -> None:
        """راه‌اندازی اولیه و اطمینان از آماده بودن اکسچنج‌کلاینت"""
        try:
            await self.exchange_client._init_session()
            await self._calculate_symbol_priorities()
            self.logger.debug("نشست ExchangeClient از طریق MarketDataFetcher راه‌اندازی شد")
        except Exception as e:
            self.logger.error(f"خطا در راه‌اندازی نشست exchange client: {e}", exc_info=True)
            raise

    async def shutdown(self) -> None:
        """پاکسازی منابع"""
        # لغو تسک‌های نگهداری
        for task in self._maintenance_tasks:
            if task and not task.done():
                task.cancel()
                try:
                    await task
                except asyncio.CancelledError:
                    pass

        # پاکسازی کش
        try:
            if hasattr(self, 'cache'):
                await self.cache.clear_all()
                self.logger.debug("کش در طول خاموش‌شدن پاکسازی شد")
        except Exception as e:
            self.logger.error(f"خطا در طول خاموش‌شدن: {e}", exc_info=True)

    async def _wait_for_rate_limit(self) -> None:
        """انتظار هوشمند برای رعایت Rate Limit"""
        current_time = time.time()
        time_since_last_request = current_time - self.last_request_time

        if time_since_last_request < self.delay_between_requests:
            delay = self.delay_between_requests - time_since_last_request
            # فقط اگر تاخیر قابل توجه است، انتظار بکش
            if delay > 0.05:
                await asyncio.sleep(delay)

        self.last_request_time = time.time()

    async def _apply_backoff_strategy(self, success: bool = True) -> None:
        """
        اعمال استراتژی backoff برای کنترل خطاها

        Args:
            success: آیا عملیات موفق بوده است
        """
        if success:
            # ریست کانتر در صورت موفقیت
            self._error_backoff['count'] = max(0, self._error_backoff['count'] - 1)
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

        self.logger.warning(
            f"اعمال تاخیر backoff {total_delay:.2f} ثانیه پس از خطا (شمارنده: {self._error_backoff['count']})")
        await asyncio.sleep(total_delay)

    async def _calculate_symbol_priorities(self):
        """محاسبه اولویت نمادها برای به‌روزرسانی‌های هوشمند"""
        if time.time() - self._last_priority_calculation < 3600:  # هر ساعت یک بار
            return

        symbols = await self.get_active_symbols()
        priorities = {}

        # نمادهای مهم همیشه اولویت بالا دارند
        important_coins = {'BTC', 'ETH', 'SOL', 'BNB', 'XRP', 'DOGE', 'ADA', 'DOT'}

        for symbol in symbols:
            # مقدار پایه
            priority = 1.0

            # افزایش اولویت برای نمادهای مهم
            for coin in important_coins:
                if coin in symbol:
                    priority += 5.0
                    break

            # افزایش اولویت برای نمادهایی که شکاف داشته‌اند
            if symbol in self._gap_tracking["detected_gaps"]:
                priority += 3.0

            # افزایش اولویت برای نمادهایی که خطا داشته‌اند
            if symbol in self._stats.get("last_error_time", {}):
                priority += 2.0

            priorities[symbol] = priority

        # مرتب‌سازی بر اساس اولویت
        sorted_symbols = sorted(priorities.items(), key=lambda x: x[1], reverse=True)

        # انتخاب 20% نمادها با بالاترین اولویت
        high_priority_count = max(10, int(len(symbols) * 0.2))
        self._high_priority_symbols = {symbol for symbol, _ in sorted_symbols[:high_priority_count]}
        self._symbol_priorities = {symbol: priority for symbol, priority in sorted_symbols}

        self._last_priority_calculation = time.time()
        self.logger.info(f"اولویت‌های نماد به‌روزرسانی شدند: {high_priority_count} نماد با اولویت بالا انتخاب شدند")

    async def get_active_symbols(self, force_refresh: bool = False) -> List[str]:
        """
        دریافت لیست همه جفت ارزهای فعال با کش هوشمند

        Args:
            force_refresh: دریافت مجدد از API بدون استفاده از کش

        Returns:
            لیست نمادهای فعال
        """
        current_time = time.time()
        cache_age = current_time - self._active_symbols_last_update

        # استفاده از کش اگر معتبر است
        if not force_refresh and self._active_symbols_cache and cache_age < self._active_symbols_cache_ttl:
            return self._active_symbols_cache

        try:
            # دریافت نمادها از API
            await self._wait_for_rate_limit()
            symbols = await self.exchange_client.get_all_usdt_symbols()
            self._stats["last_fetch_time"] = time.time()

            if not symbols:
                self.logger.warning("هیچ نماد فعالی از API صرافی دریافت نشد")
                # اگر کش قبلی داریم، از آن استفاده کن
                if self._active_symbols_cache:
                    return self._active_symbols_cache

                # در غیر این صورت، لیست پیش‌فرض را برگردان
                return ['BTC/USDT', 'ETH/USDT']

            # اعمال فیلتر تعداد حداکثر
            max_symbols = self.data_config.get('max_symbols', 0)
            if max_symbols > 0 and len(symbols) > max_symbols:
                self.logger.info(f"محدود کردن نمادهای فعال به {max_symbols}")
                symbols = symbols[:max_symbols]

            # به‌روزرسانی کش
            self._active_symbols_cache = symbols
            self._active_symbols_last_update = current_time

            # ریست کانتر خطا در صورت موفقیت
            await self._apply_backoff_strategy(success=True)

            # بروزرسانی اولویت نمادها
            await self._calculate_symbol_priorities()

            self.logger.info(f"{len(symbols)} نماد فعال از صرافی دریافت شد")
            return symbols

        except Exception as e:
            self.logger.error(f"خطا در دریافت نمادهای فعال: {e}", exc_info=True)

            # اعمال استراتژی backoff
            await self._apply_backoff_strategy(success=False)

            # اگر کش قبلی داریم، از آن استفاده کن
            if self._active_symbols_cache:
                return self._active_symbols_cache

            # در غیر این صورت، لیست خالی را برگردان
            return []

    async def get_current_price(self, symbol: str) -> Optional[float]:
        """
        دریافت قیمت فعلی با اولویت وب‌سوکت

        Args:
            symbol: نماد ارز

        Returns:
            قیمت فعلی یا None
        """
        try:
            # تلاش برای دریافت از وب‌سوکت
            if hasattr(self.exchange_client, 'get_current_price_ws'):
                price = await self.exchange_client.get_current_price_ws(symbol)
                if price is not None and price > 0:
                    return price

            # اگر وب‌سوکت نبود یا موفق نبود، از API استفاده کن
            await self._wait_for_rate_limit()

            async with self.fetch_semaphore:
                start_time = time.time()
                price = await self.exchange_client.get_current_price(symbol)
                response_time = (time.time() - start_time) * 1000
                self._stats["last_fetch_time"] = time.time()

                # آپدیت آمار
                self._stats["total_requests"] += 1
                if price is not None and price > 0:
                    self._stats["successful_requests"] += 1
                    # میانگین موزون زمان پاسخ
                    self._stats["average_response_time_ms"] = (
                            self._stats["average_response_time_ms"] * 0.9 + response_time * 0.1
                    )
                    # ریست کانتر خطا در صورت موفقیت
                    await self._apply_backoff_strategy(success=True)
                    return price
                else:
                    self._stats["failed_requests"] += 1

            self.logger.warning(f"دریافت قیمت فعلی برای {symbol} ناموفق بود")
            return None

        except Exception as e:
            self.logger.error(f"خطا در دریافت قیمت فعلی برای {symbol}: {e}")
            self._stats["failed_requests"] += 1
            self._stats["errors_by_type"][type(e).__name__] = self._stats["errors_by_type"].get(type(e).__name__, 0) + 1
            self._stats["last_error_time"][symbol] = time.time()
            # اعمال استراتژی backoff
            await self._apply_backoff_strategy(success=False)
            return None

    @staticmethod
    def _get_cache_key(symbol: str, timeframe: str) -> str:
        """ساخت کلید یکتا برای کش"""
        clean_symbol = symbol.replace('/', '_').replace(':', '_').upper()
        return f"ohlcv:{clean_symbol}:{timeframe}"

    async def fetch_new_candles(self, symbol: str, timeframe: str, last_timestamp: int) -> pd.DataFrame:
        """
        دریافت فقط کندل‌های جدید با کنترل خطا

        Args:
            symbol: نماد ارز
            timeframe: تایم‌فریم
            last_timestamp: آخرین timestamp دریافتی (میلی‌ثانیه)

        Returns:
            دیتافریم کندل‌های جدید یا DataFrame خالی
        """
        try:
            # محاسبه زمان پایان (زمان فعلی)
            end_time = int(time.time() * 1000)

            # محاسبه طول تایم‌فریم به میلی‌ثانیه
            tf_ms = self._get_timeframe_ms(timeframe)

            # بررسی فاصله زمانی (آیا زمان کافی برای کندل جدید گذشته؟)
            if end_time - last_timestamp < tf_ms:
                return pd.DataFrame()  # هنوز زمان کندل جدید نرسیده

            # لاگ دیباگ
            from_time_str = datetime.fromtimestamp(last_timestamp / 1000).strftime('%Y-%m-%d %H:%M:%S')
            self.logger.debug(
                f"دریافت کندل‌های جدید برای {symbol} {timeframe} از {from_time_str}"
            )

            # دریافت کندل‌های جدید
            await self._wait_for_rate_limit()
            start_time = time.time()
            df_new = await self.exchange_client.fetch_ohlcv(
                symbol,
                timeframe,
                start_time_s=last_timestamp / 1000,  # تبدیل به ثانیه
                end_time_s=end_time / 1000  # تبدیل به ثانیه
            )
            self._stats["last_fetch_time"] = time.time()

            response_time = (time.time() - start_time) * 1000
            self._stats["total_requests"] += 1

            if df_new is None:
                self.logger.warning(f"دریافت کندل‌های جدید برای {symbol} {timeframe} ناموفق بود")
                self._stats["failed_requests"] += 1
                self._stats["last_error_time"][symbol] = time.time()
                # اعمال استراتژی backoff
                await self._apply_backoff_strategy(success=False)
                return pd.DataFrame()

            if df_new.empty:
                self.logger.debug(f"کندل جدیدی برای {symbol} {timeframe} موجود نیست")
                self._stats["successful_requests"] += 1
                # ریست کانتر خطا حتی در صورت خالی بودن (چون عملیات موفق بوده)
                await self._apply_backoff_strategy(success=True)
                return df_new

            # اطمینان از صحت داده‌ها
            self._validate_dataframe(df_new)

            # آپدیت آمار
            self._stats["successful_requests"] += 1
            self._stats["total_candles_fetched"] += len(df_new)
            # میانگین موزون زمان پاسخ
            self._stats["average_response_time_ms"] = (
                    self._stats["average_response_time_ms"] * 0.9 + response_time * 0.1
            )

            # ریست کانتر خطا در صورت موفقیت
            await self._apply_backoff_strategy(success=True)

            # لاگ موفقیت
            self.logger.debug(f"{len(df_new)} کندل جدید برای {symbol} {timeframe} دریافت شد")
            return df_new

        except Exception as e:
            self.logger.error(f"خطا در دریافت کندل‌های جدید برای {symbol} {timeframe}: {e}", exc_info=True)
            self._stats["failed_requests"] += 1
            self._stats["errors_by_type"][type(e).__name__] = self._stats["errors_by_type"].get(type(e).__name__, 0) + 1
            self._stats["last_error_time"][symbol] = time.time()
            # اعمال استراتژی backoff
            await self._apply_backoff_strategy(success=False)
            return pd.DataFrame()

    def _validate_dataframe(self, df: pd.DataFrame) -> bool:
        """
        اعتبارسنجی ساختار داده‌ها

        Args:
            df: دیتافریم برای بررسی

        Returns:
            True اگر معتبر است
        """
        if df is None or df.empty:
            return False

        # بررسی ستون‌های مورد نیاز
        required_columns = ['open', 'high', 'low', 'close', 'volume']
        for col in required_columns:
            if col not in df.columns:
                self.logger.warning(f"ستون مورد نیاز {col} در DataFrame یافت نشد")
                return False

        # بررسی نوع داده‌ها و مقادیر نامعتبر
        for col in required_columns:
            if df[col].isnull().any():
                self.logger.warning(f"مقادیر NaN در ستون {col} یافت شد")
                # پاکسازی NaN
                df.dropna(subset=[col], inplace=True)

            # تبدیل به فلوت
            try:
                df[col] = pd.to_numeric(df[col], errors='coerce')
            except Exception as e:
                self.logger.error(f"خطا در تبدیل {col} به عدد: {e}")

        return True

    async def get_historical_data(
            self,
            symbol: str,
            timeframe: str,
            limit: int = 500,
            force_refresh: bool = False
    ) -> Optional[pd.DataFrame]:
        """
        دریافت داده‌های تاریخی با کش هوشمند و مدیریت خطا

        Args:
            symbol: نماد ارز
            timeframe: تایم‌فریم
            limit: تعداد کندل‌ها
            force_refresh: دریافت مجدد بدون استفاده از کش

        Returns:
            دیتافریم داده‌ها یا None در صورت خطا
        """
        if self.use_mock_data:
            # تولید داده‌های ساختگی برای تست
            return self._generate_mock_data(symbol, timeframe, limit)

        # ساخت کلید کش
        cache_key = self._get_cache_key(symbol, timeframe)

        # تلاش برای دریافت از کش
        if not force_refresh:
            cached_df = await self.cache.get(cache_key)
            if cached_df is not None and len(cached_df) >= limit:
                self._stats["cache_hits"] += 1
                return cached_df.iloc[-limit:] if len(cached_df) > limit else cached_df

        # استراتژی دریافت داده بر اساس وضعیت کش
        if not force_refresh and await self.cache.get(cache_key) is not None and self.use_delta_updates:
            # استراتژی به‌روزرسانی دلتا (فقط دریافت کندل‌های جدید)
            return await self._delta_update_strategy(symbol, timeframe, limit, cache_key)
        else:
            # استراتژی دریافت کامل
            return await self._full_fetch_strategy(symbol, timeframe, limit, cache_key)

    async def _full_fetch_strategy(
            self,
            symbol: str,
            timeframe: str,
            limit: int,
            cache_key: str
    ) -> Optional[pd.DataFrame]:
        """
        استراتژی دریافت کامل: دریافت تمام داده‌ها از API با پشتیبانی از تعداد زیاد کندل‌ها

        Args:
            symbol: نماد ارز
            timeframe: تایم‌فریم
            limit: تعداد کندل‌ها
            cache_key: کلید کش

        Returns:
            دیتافریم یا None
        """
        try:
            # افزایش تعداد درخواست برای نمادهای با اولویت بالا برای اطمینان از پر شدن شکاف‌ها
            adjusted_limit = limit
            if symbol in self._high_priority_symbols:
                adjusted_limit = max(limit, 200)  # حداقل 200 کندل برای نمادهای مهم

            # دریافت کامل داده‌ها از API
            await self._wait_for_rate_limit()

            async with self.fetch_semaphore:
                start_time = time.time()

                # برای تعداد زیاد کندل، از روش چند درخواست استفاده می‌کنیم
                if adjusted_limit > 200:
                    fetched_df = await self._fetch_multiple_batches(symbol, timeframe, adjusted_limit)
                else:
                    fetched_df = await self.exchange_client.fetch_ohlcv(symbol, timeframe, limit=adjusted_limit)

                self._stats["last_fetch_time"] = time.time()
                response_time = (time.time() - start_time) * 1000

                # آپدیت آمار
                self._stats["total_requests"] += 1

            if fetched_df is None:
                self.logger.error(f"دریافت داده برای {symbol} {timeframe} ناموفق بود")
                self._stats["failed_requests"] += 1
                self._stats["last_error_time"][symbol] = time.time()
                # اعمال استراتژی backoff
                await self._apply_backoff_strategy(success=False)
                return None

            if fetched_df.empty:
                self.logger.warning(f"داده خالی برای {symbol} {timeframe} دریافت شد")
                self._stats["successful_requests"] += 1
                # ریست کانتر خطا حتی در صورت خالی بودن (چون عملیات موفق بوده)
                await self._apply_backoff_strategy(success=True)
                return pd.DataFrame()

            # اعتبارسنجی داده‌ها
            self._validate_dataframe(fetched_df)

            # آپدیت آمار
            self._stats["successful_requests"] += 1
            self._stats["total_candles_fetched"] += len(fetched_df)
            # میانگین موزون زمان پاسخ
            self._stats["average_response_time_ms"] = (
                    self._stats["average_response_time_ms"] * 0.9 + response_time * 0.1
            )

            # ذخیره در کش
            await self.cache.set(cache_key, fetched_df)

            # ریست کانتر خطا در صورت موفقیت
            await self._apply_backoff_strategy(success=True)

            return fetched_df

        except Exception as e:
            self.logger.error(f"خطا در استراتژی دریافت کامل برای {symbol} {timeframe}: {e}", exc_info=True)
            self._stats["failed_requests"] += 1
            self._stats["errors_by_type"][type(e).__name__] = self._stats["errors_by_type"].get(type(e).__name__, 0) + 1
            self._stats["last_error_time"][symbol] = time.time()
            # اعمال استراتژی backoff
            await self._apply_backoff_strategy(success=False)
            return None

    async def _fetch_multiple_batches(
            self,
            symbol: str,
            timeframe: str,
            limit: int
    ) -> Optional[pd.DataFrame]:
        """
        دریافت تعداد زیادی کندل در چندین batch

        Args:
            symbol: نماد ارز
            timeframe: تایم‌فریم
            limit: تعداد کل کندل‌های مورد نیاز

        Returns:
            دیتافریم داده‌ها یا None
        """
        try:
            all_candles = []
            batch_size = 200  # حداکثر کندل در هر درخواست
            remaining = limit

            # محاسبه طول کندل به ثانیه
            candle_duration_ms = self._get_timeframe_ms(timeframe)
            candle_duration_s = candle_duration_ms / 1000 if candle_duration_ms else 3600

            # زمان پایان (فعلی)
            end_time_s = time.time()
            current_end_time = end_time_s

            # زمان شروع کلی
            start_time_s = end_time_s - (limit * candle_duration_s)

            while remaining > 0 and current_end_time > start_time_s:
                batch_limit = min(batch_size, remaining)

                # محاسبه زمان شروع برای این batch
                batch_start_time = current_end_time - (batch_limit * candle_duration_s)

                self.logger.debug(f"دریافت بسته برای {symbol} {timeframe}: {batch_limit} کندل")

                # دریافت کندل‌ها
                batch_df = await self.exchange_client.fetch_ohlcv(
                    symbol,
                    timeframe,
                    limit=batch_limit,
                    start_time_s=batch_start_time,
                    end_time_s=current_end_time
                )
                self._stats["last_fetch_time"] = time.time()

                if batch_df is None or batch_df.empty:
                    # اگر داده‌ای دریافت نشد، احتمالاً به ابتدای داده‌ها رسیده‌ایم
                    self.logger.warning(f"داده بیشتری برای {symbol} {timeframe} موجود نیست")
                    break

                # اضافه کردن کندل‌های جدید (جدیدترین‌ها اول)
                all_candles.append(batch_df)

                remaining -= len(batch_df)
                current_end_time = batch_start_time

                # تاخیر کوتاه برای رعایت rate limit
                await asyncio.sleep(0.2)

                # بررسی اینکه آیا کافی کندل دریافت کردیم
                if len(batch_df) < batch_limit:
                    # احتمالاً دیگر کندل‌های قدیمی‌تری موجود نیست
                    self.logger.info(f"به انتهای داده‌های موجود برای {symbol} {timeframe} رسیدیم")
                    break

            if not all_candles:
                self.logger.warning(f"داده‌ای برای {symbol} {timeframe} بازگردانده نشد")
                return pd.DataFrame()

            # ترکیب تمام batch‌ها
            final_df = pd.concat(all_candles)

            # حذف ردیف‌های تکراری
            final_df = final_df[~final_df.index.duplicated(keep='last')]

            # مرتب‌سازی بر اساس زمان (قدیمی‌ترین‌ها اول)
            final_df = final_df.sort_index()

            # برش‌ زدن به تعداد دقیق درخواست شده
            if len(final_df) > limit:
                final_df = final_df.iloc[-limit:]

            self.logger.info(f"{len(final_df)} کندل با موفقیت برای {symbol} {timeframe} دریافت شد")
            return final_df

        except Exception as e:
            self.logger.error(f"خطا در دریافت چندگانه برای {symbol} {timeframe}: {e}", exc_info=True)
            return None

    @staticmethod
    def _get_timeframe_ms(timeframe: str) -> int:
        """تبدیل تایم‌فریم به میلی‌ثانیه"""
        try:
            if 'm' in timeframe:
                return int(timeframe.replace('m', '')) * 60 * 1000
            elif 'h' in timeframe:
                return int(timeframe.replace('h', '')) * 60 * 60 * 1000
            elif 'd' in timeframe:
                return int(timeframe.replace('d', '')) * 24 * 60 * 60 * 1000
            elif 'w' in timeframe:
                return int(timeframe.replace('w', '')) * 7 * 24 * 60 * 60 * 1000
            else:
                return 60 * 1000  # پیش‌فرض 1 دقیقه
        except ValueError:
            return 60 * 1000  # پیش‌فرض 1 دقیقه در صورت خطا

    async def _delta_update_strategy(
            self,
            symbol: str,
            timeframe: str,
            limit: int,
            cache_key: str
    ) -> Optional[pd.DataFrame]:
        """
        استراتژی به‌روزرسانی دلتا: دریافت فقط کندل‌های جدید و ترکیب با کش

        Args:
            symbol: نماد ارز
            timeframe: تایم‌فریم
            limit: تعداد کندل‌ها
            cache_key: کلید کش

        Returns:
            دیتافریم به‌روز شده یا None
        """
        try:
            # دریافت کش فعلی
            cached_df = await self.cache.get(cache_key)
            if cached_df is None or cached_df.empty:
                # اگر کش خالی است، استراتژی دریافت کامل را اجرا کن
                return await self._full_fetch_strategy(symbol, timeframe, limit, cache_key)

            # آخرین timestamp موجود در کش
            last_timestamp = int(cached_df.index[-1].timestamp() * 1000)

            # دریافت کندل‌های جدید
            async with self.fetch_semaphore:
                new_df = await self.fetch_new_candles(symbol, timeframe, last_timestamp)

            if new_df is None:
                # در صورت خطا، کش موجود را برگردان
                self.logger.warning(f"خطا در دریافت به‌روزرسانی دلتا برای {symbol} {timeframe}، استفاده از داده‌های کش")
                return cached_df.iloc[-limit:] if len(cached_df) > limit else cached_df

            if new_df.empty:
                # اگر کندل جدیدی نیست، کش موجود را برگردان
                return cached_df.iloc[-limit:] if len(cached_df) > limit else cached_df

            # ترکیب کش با داده‌های جدید
            combined_df = pd.concat([cached_df, new_df])

            # حذف ردیف‌های تکراری
            combined_df = combined_df[~combined_df.index.duplicated(keep='last')]

            # مرتب‌سازی بر اساس زمان
            combined_df = combined_df.sort_index()

            # ذخیره در کش
            await self.cache.set(cache_key, combined_df)

            # برگرداندن با limit درخواستی
            return combined_df.iloc[-limit:] if len(combined_df) > limit else combined_df

        except Exception as e:
            self.logger.error(f"خطا در استراتژی به‌روزرسانی دلتا برای {symbol} {timeframe}: {e}", exc_info=True)

            # دریافت مجدد از کش در صورت خطا
            cached_df = await self.cache.get(cache_key)
            if cached_df is not None:
                return cached_df.iloc[-limit:] if len(cached_df) > limit else cached_df

            # در صورت عدم موفقیت، به استراتژی دریافت کامل بازگرد
            return await self._full_fetch_strategy(symbol, timeframe, limit, cache_key)


    async def get_multi_timeframe_data(
            self,
            symbol: str,
            timeframes: Optional[List[str]] = None,
            force_refresh: bool = False,
            limit_per_tf: int = 500
    ) -> Dict[str, Optional[pd.DataFrame]]:
        """
        دریافت داده‌ها در چندین تایم‌فریم به صورت موازی با مدیریت خطا

        Args:
            symbol: نماد ارز
            timeframes: لیست تایم‌فریم‌ها
            force_refresh: دریافت مجدد بدون استفاده از کش
            limit_per_tf: تعداد کندل برای هر تایم‌فریم

        Returns:
            دیکشنری از دیتافریم‌ها برای هر تایم‌فریم
        """
        target_timeframes = timeframes if timeframes else self.timeframes
        self.logger.debug(f"دریافت داده‌های چند تایم‌فریمی برای {symbol} در {target_timeframes}")

        # ایجاد تسک‌ها برای دریافت موازی
        tasks = [
            self.get_historical_data(symbol, tf, limit=limit_per_tf, force_refresh=force_refresh)
            for tf in target_timeframes
        ]

        # اجرای موازی تسک‌ها
        results = await asyncio.gather(*tasks, return_exceptions=True)

        # پردازش نتایج
        result_dict: Dict[str, Optional[pd.DataFrame]] = {}
        for i, tf in enumerate(target_timeframes):
            if isinstance(results[i], Exception):
                self.logger.error(f"خطا در دریافت داده برای {symbol} {tf}: {results[i]}")
                result_dict[tf] = None
            else:
                result_dict[tf] = results[i]
                if results[i] is not None:
                    if results[i].empty:
                        self.logger.warning(f"DataFrame خالی برای {symbol} {tf}")
                    else:
                        self.logger.debug(f"داده برای {symbol} {tf} با موفقیت دریافت شد ({len(results[i])} ردیف)")

        return result_dict

    def _generate_mock_data(self, symbol: str, timeframe: str, limit: int = 100) -> pd.DataFrame:
        """
        تولید داده‌های ساختگی برای تست

        Args:
            symbol: نماد ارز
            timeframe: تایم‌فریم
            limit: تعداد کندل‌ها

        Returns:
            دیتافریم داده‌های ساختگی
        """
        import random

        self.logger.info(f"تولید داده‌های ساختگی برای {symbol} {timeframe} (تعداد={limit})")

        # تعیین قیمت پایه بر اساس نماد
        base_price = 0
        if 'BTC' in symbol:
            base_price = 50000
        elif 'ETH' in symbol:
            base_price = 3000
        elif 'SOL' in symbol:
            base_price = 100
        else:
            base_price = 10

        # ایجاد تاریخ‌ها
        tf_seconds = self._get_timeframe_ms(timeframe) / 1000
        end_time = int(time.time())
        timestamps = [end_time - (i * tf_seconds) for i in range(limit)]
        timestamps.reverse()  # مرتب‌سازی صعودی

        # تولید قیمت‌ها
        np.random.seed(hash(symbol + timeframe) % 10000)  # تکرارپذیری برای هر نماد/تایم‌فریم

        # ساخت سری قیمت با روند نمایی و نوسان
        price_series = []
        current_price = base_price

        for i in range(limit):
            # نوسان تصادفی
            volatility = base_price * 0.02  # 2% نوسان
            change = np.random.normal(0, volatility)

            # روند ملایم
            trend = base_price * 0.001 * np.sin(i / 20)  # روند سینوسی ملایم

            # قیمت جدید
            current_price += change + trend
            current_price = max(current_price, base_price * 0.5)
            # ساخت OHLC
            open_price = current_price * (1 + np.random.normal(0, 0.001))  # کمی تغییر در open
            high_price = current_price * (1 + np.random.uniform(0, 0.01))
            low_price = current_price * (1 - np.random.uniform(0, 0.01))
            # اطمینان از اینکه open/close بین high/low باشند
            close_price = open_price + np.random.normal(0, volatility * 0.3)
            high_price = max(high_price, open_price, close_price)
            low_price = min(low_price, open_price, close_price)

            # تولید حجم
            volume = np.random.gamma(2.0, base_price * 10)

            # ساختن دیکشنری داده
            price_series.append({
                'timestamp': pd.Timestamp(timestamps[i], unit='s', tz='UTC'),
                'open': open_price,
                'high': high_price,
                'low': low_price,
                'close': close_price,
                'volume': volume
            })

            # ساخت دیتافریم
        df = pd.DataFrame(price_series)
        df.set_index('timestamp', inplace=True)

        return df

    def get_stats(self) -> Dict[str, Any]:
        """دریافت آمار عملکرد"""
        stats = {
            "fetcher_stats": self._stats,
            "cache_stats": self.cache.get_stats(),
            "gap_stats": self._gap_tracking,
            "high_priority_symbols_count": len(self._high_priority_symbols),
            "error_backoff": self._error_backoff
        }
        return stats