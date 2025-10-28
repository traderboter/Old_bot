# exchange_client.py
"""
ماژول کلاینت صرافی با استفاده از aiohttp.
پیاده‌سازی بهینه‌سازی شده با مدیریت خطا، کش‌کردن چندلایه و پشتیبانی از ویژگی‌های پیشرفته.
قابلیت به‌روزرسانی تنظیمات در زمان اجرا اضافه شده است.
"""

import aiohttp
import asyncio
import base64
import hashlib
import hmac
import json
import logging
import time
import pandas as pd
import redis.asyncio as redis
import websockets
import random
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, Any, Optional, List, Union, Callable, Awaitable, Tuple, Set, cast
from urllib.parse import urlencode
from datetime import datetime, timedelta
from collections import deque
import uuid
import os
from functools import lru_cache
from concurrent.futures import ThreadPoolExecutor

logger = logging.getLogger(__name__)

# نگاشت تایم‌فریم‌ها به Granularity (دقیقه) برای صرافی
TIMEFRAME_MAP = {
    "1m": 1, "3m": 3, "5m": 5, "15m": 15, "30m": 30,
    "1h": 60, "2h": 120, "4h": 240, "8h": 480, "12h": 720,
    "1d": 1440, "1w": 10080
}

# مپینگ عکس برای تبدیل granularity به timeframe
GRANULARITY_TO_TIMEFRAME = {v: k for k, v in TIMEFRAME_MAP.items()}


# تعریف انواع قراردادها
class ContractType(Enum):
    PERPETUAL = "perpetual"  # قراردادهای دائمی
    FUTURES = "futures"  # قراردادهای آتی با تاریخ انقضا
    ALL = "all"  # همه انواع قراردادها


@dataclass
class ExchangeConfig:
    """کلاس ساختارمند برای نگهداری تنظیمات صرافی"""
    api_key: Optional[str] = None
    api_secret: Optional[str] = None
    api_passphrase: Optional[str] = None
    api_version: str = "2"
    base_url: str = "https://api-futures.kucoin.com"
    max_concurrent_requests: int = 10  # حداکثر تعداد درخواست‌های همزمان
    request_timeout: int = 30  # زمان انتظار درخواست (ثانیه)
    connection_timeout: int = 10  # زمان انتظار اتصال (ثانیه)
    retry_count: int = 3  # تعداد تلاش مجدد درخواست
    retry_delay_base: float = 0.5  # تاخیر پایه بین تلاش‌های مجدد (ثانیه)
    use_redis_cache: bool = False  # استفاده از Redis برای کش
    memory_cache_ttl: int = 60  # زمان انقضای کش حافظه (ثانیه)
    redis_cache_ttl: int = 600  # زمان انقضای کش Redis (ثانیه)
    redis_host: str = "localhost"  # هاست Redis
    redis_port: int = 6379  # پورت Redis
    redis_db: int = 0  # شماره دیتابیس Redis
    redis_password: Optional[str] = None  # رمز عبور Redis (اختیاری)
    debug_mode: bool = False  # حالت دیباگ
    rate_limit_requests_per_second: float = 5.0  # تعداد درخواست در ثانیه
    websocket_enabled: bool = True  # فعال‌سازی وب‌سوکت
    websocket_ping_interval: int = 20  # فاصله زمانی ping (ثانیه)
    websocket_reconnect_delay: int = 5  # تاخیر اتصال مجدد (ثانیه)
    health_check_interval: int = 60  # فاصله زمانی بررسی سلامت (ثانیه)


@dataclass
class PriceData:
    """کلاس ساختارمند برای نگهداری داده‌های قیمت"""
    symbol: str
    price: float
    timestamp: float
    bid: Optional[float] = None
    ask: Optional[float] = None
    volume_24h: Optional[float] = None
    change_24h: Optional[float] = None
    change_rate_24h: Optional[float] = None
    source: str = "api"  # منبع داده: "api" یا "websocket"

    def is_fresh(self, max_age_seconds: int = 60) -> bool:
        """بررسی تازگی داده قیمت"""
        return (time.time() - self.timestamp) < max_age_seconds


@dataclass
class CacheEntry:
    """کلاس نگهداری داده کش شده با زمان انقضا"""
    data: Any
    timestamp: float = field(default_factory=time.time)
    expiry: float = 60  # زمان انقضا (ثانیه)

    def is_expired(self) -> bool:
        """بررسی انقضای داده کش شده"""
        return (time.time() - self.timestamp) > self.expiry


class ExchangeClient:
    """
    کلاینت آسنکرون برای تعامل با API صرافی با پشتیبانی از کش چندلایه و مدیریت خطای پیشرفته.
    قابلیت به‌روزرسانی تنظیمات در زمان اجرا اضافه شده است.
    """

    def __init__(self, config: Dict[str, Any]):
        """
        مقداردهی اولیه کلاینت صرافی.

        Args:
            config: دیکشنری تنظیمات شامل پارامترهای اتصال و ویژگی‌های کش.
        """
        exchange_config = config.get('exchange', {})

        # تنظیمات کلاینت
        self.config = ExchangeConfig(
            api_key=exchange_config.get('api_key'),
            api_secret=exchange_config.get('api_secret'),
            api_passphrase=exchange_config.get('api_passphrase'),
            api_version=str(exchange_config.get('api_version', '2')),
            base_url=exchange_config.get('base_url', "https://api-futures.kucoin.com").rstrip('/'),
            max_concurrent_requests=exchange_config.get('max_concurrent_requests', 10),
            request_timeout=exchange_config.get('request_timeout', 30),
            connection_timeout=exchange_config.get('connection_timeout', 10),
            retry_count=exchange_config.get('retry_count', 3),
            retry_delay_base=exchange_config.get('retry_delay_base', 0.5),
            use_redis_cache=exchange_config.get('use_redis_cache', False),
            memory_cache_ttl=exchange_config.get('memory_cache_ttl', 60),
            redis_cache_ttl=exchange_config.get('redis_cache_ttl', 600),
            redis_host=exchange_config.get('redis_host', 'localhost'),
            redis_port=exchange_config.get('redis_port', 6379),
            redis_db=exchange_config.get('redis_db', 0),
            redis_password=exchange_config.get('redis_password'),
            debug_mode=exchange_config.get('debug_mode', False),
            rate_limit_requests_per_second=exchange_config.get('rate_limit_requests_per_second', 5.0),
            websocket_enabled=exchange_config.get('websocket', {}).get('enabled', True),
            websocket_ping_interval=exchange_config.get('websocket', {}).get('ping_interval', 20),
            websocket_reconnect_delay=exchange_config.get('websocket', {}).get('reconnect_delay', 5),
            health_check_interval=exchange_config.get('health_check_interval', 60)
        )

        # بررسی اولیه برای وجود URL صحیح فیوچرز
        if "api-futures" not in self.config.base_url:
            logger.warning(f"آدرس پایه '{self.config.base_url}' ممکن است برای API فیوچرز مناسب نباشد!")

        # بررسی اولیه برای وجود Credential ها
        if not all([self.config.api_key, self.config.api_secret, self.config.api_passphrase]):
            logger.warning(
                "اطلاعات احراز هویت API (کلید، رمز، پسورد) ناقص است. متدهای نیازمند احراز هویت کار نخواهند کرد.")

        # وهله ClientSession برای ارسال درخواست‌ها
        self._session: Optional[aiohttp.ClientSession] = None

        # قفل برای ایجاد session به صورت thread-safe
        self._session_lock = asyncio.Lock()

        # یک Semaphore برای محدود کردن تعداد درخواست‌های همزمان (مدیریت Rate Limit)
        self._request_lock = asyncio.Semaphore(self.config.max_concurrent_requests)

        # قفل‌های کش
        self._cache_lock = asyncio.Lock()
        self._redis_lock = asyncio.Lock()

        # کلاینت Redis
        self._redis_client = None
        if self.config.use_redis_cache:
            self._init_redis()

        # Thread Pool برای عملیات I/O
        self._thread_executor = ThreadPoolExecutor(max_workers=4)

        # کش‌های داخلی
        self._memory_cache: Dict[str, CacheEntry] = {}
        self._price_cache: Dict[str, PriceData] = {}
        self._symbol_info_cache: Dict[str, Dict[str, Any]] = {}

        # آخرین زمان ریست کردن کش
        self._last_cache_cleanup = time.time()

        # زمان سرور و تفاوت با زمان محلی
        self._server_time_diff: Optional[float] = None
        self._last_server_time_sync = 0

        # مدیریت Rate Limit
        self._rate_limit_time_window = 1.0 / self.config.rate_limit_requests_per_second
        self._endpoint_last_request: Dict[str, float] = {}
        self._rate_limit_violations: Dict[str, int] = {}
        self._min_request_interval: Dict[str, float] = {}
        for endpoint in ['kline', 'ticker', 'orders', 'account', 'public', 'server_time']:
            self._min_request_interval[endpoint] = self._rate_limit_time_window

        # آمار API
        self._api_stats = {
            'requests_count': 0,
            'error_count': 0,
            'cache_hits': 0,
            'cache_misses': 0,
            'last_request_time': 0,
            'response_times': deque(maxlen=100),
            'status_codes': {},
            'endpoint_stats': {}
        }

        # تایمر کش
        self._cache_cleanup_task: Optional[asyncio.Task] = None
        self._health_check_task: Optional[asyncio.Task] = None

        # وب‌سوکت
        self._ws_connection = None
        self._ws_last_pong_time = 0
        self._ws_running = False
        self._ws_subscriptions: Set[str] = set()
        self._ws_message_handlers: Dict[str, Callable] = {}
        self._ws_token_expiry = 0
        self._ws_reconnect_task: Optional[asyncio.Task] = None
        self._ws_lock = asyncio.Lock()

        # WebSocket قیمت‌های دریافتی
        self._ws_prices: Dict[str, PriceData] = {}

        # مدیریت وضعیت خطا
        self._error_backoff = {
            'count': 0,
            'max_delay': 60,  # حداکثر تاخیر 60 ثانیه
            'base_delay': 2,  # تاخیر پایه 2 ثانیه
        }

        # وضعیت کلی کلاینت
        self._client_health_status = {
            'status': 'initialized',
            'last_error': None,
            'last_update_time': time.time(),
            'http_status': 'not_connected',
            'ws_status': 'disabled' if not self.config.websocket_enabled else 'not_connected'
        }

        logger.info(f"کلاینت صرافی برای {self.config.base_url} راه‌اندازی شد")

    def update_config(self, new_config: Dict[str, Any]) -> None:
        """
        به‌روزرسانی تنظیمات کلاینت صرافی در زمان اجرا.

        Args:
            new_config: تنظیمات جدید
        """
        try:
            # استخراج تنظیمات صرافی
            exchange_config = new_config.get('exchange', {})
            if not exchange_config:
                logger.warning("بخش 'exchange' در تنظیمات جدید یافت نشد.")
                return

            logger.info("شروع به‌روزرسانی تنظیمات کلاینت صرافی...")

            # به‌روزرسانی تنظیمات اصلی
            self.config.api_key = exchange_config.get('api_key', self.config.api_key)
            self.config.api_secret = exchange_config.get('api_secret', self.config.api_secret)
            self.config.api_passphrase = exchange_config.get('api_passphrase', self.config.api_passphrase)
            self.config.api_version = str(exchange_config.get('api_version', self.config.api_version))

            # به‌روزرسانی آدرس پایه API
            new_base_url = exchange_config.get('base_url', self.config.base_url).rstrip('/')
            if new_base_url != self.config.base_url:
                logger.info(f"آدرس پایه API تغییر کرد از '{self.config.base_url}' به '{new_base_url}'")
                self.config.base_url = new_base_url
                # بازنشانی نشست HTTP در چرخه بعدی بررسی سلامت
                asyncio.create_task(self._reset_http_session())

            # تنظیمات درخواست‌ها
            self.config.max_concurrent_requests = exchange_config.get('max_concurrent_requests',
                                                                      self.config.max_concurrent_requests)
            if self.config.max_concurrent_requests != self._request_lock._value:
                # بازنشانی Semaphore برای مطابقت با تنظیمات جدید
                self._request_lock = asyncio.Semaphore(self.config.max_concurrent_requests)
                logger.info(f"محدودیت درخواست‌های همزمان به‌روزرسانی شد: {self.config.max_concurrent_requests}")

            self.config.request_timeout = exchange_config.get('request_timeout', self.config.request_timeout)
            self.config.connection_timeout = exchange_config.get('connection_timeout', self.config.connection_timeout)
            self.config.retry_count = exchange_config.get('retry_count', self.config.retry_count)
            self.config.retry_delay_base = exchange_config.get('retry_delay_base', self.config.retry_delay_base)

            # تنظیمات کش
            old_redis_enabled = self.config.use_redis_cache
            self.config.use_redis_cache = exchange_config.get('use_redis_cache', self.config.use_redis_cache)
            self.config.memory_cache_ttl = exchange_config.get('memory_cache_ttl', self.config.memory_cache_ttl)
            self.config.redis_cache_ttl = exchange_config.get('redis_cache_ttl', self.config.redis_cache_ttl)

            # تنظیمات Redis
            redis_config_changed = False

            new_redis_host = exchange_config.get('redis_host', self.config.redis_host)
            if new_redis_host != self.config.redis_host:
                self.config.redis_host = new_redis_host
                redis_config_changed = True

            new_redis_port = exchange_config.get('redis_port', self.config.redis_port)
            if new_redis_port != self.config.redis_port:
                self.config.redis_port = new_redis_port
                redis_config_changed = True

            new_redis_db = exchange_config.get('redis_db', self.config.redis_db)
            if new_redis_db != self.config.redis_db:
                self.config.redis_db = new_redis_db
                redis_config_changed = True

            new_redis_password = exchange_config.get('redis_password', self.config.redis_password)
            if new_redis_password != self.config.redis_password:
                self.config.redis_password = new_redis_password
                redis_config_changed = True

            # اگر تنظیمات Redis تغییر کرده یا وضعیت فعال بودن آن تغییر کرده است
            if (
                    redis_config_changed or old_redis_enabled != self.config.use_redis_cache) and self.config.use_redis_cache:
                logger.info("تنظیمات Redis تغییر کرده است. اتصال مجدد...")
                # بازنشانی اتصال Redis
                self._init_redis()

            # تنظیمات مدیریت نرخ
            new_rate_limit = exchange_config.get('rate_limit_requests_per_second',
                                                 self.config.rate_limit_requests_per_second)
            if new_rate_limit != self.config.rate_limit_requests_per_second:
                self.config.rate_limit_requests_per_second = new_rate_limit
                self._rate_limit_time_window = 1.0 / self.config.rate_limit_requests_per_second
                for endpoint in self._min_request_interval:
                    self._min_request_interval[endpoint] = self._rate_limit_time_window
                logger.info(f"محدودیت نرخ درخواست به‌روزرسانی شد: {new_rate_limit} درخواست در ثانیه")

            # تنظیمات WebSocket
            websocket_config = exchange_config.get('websocket', {})
            if websocket_config:
                # ذخیره وضعیت فعلی WebSocket
                current_ws_enabled = self.config.websocket_enabled

                # به‌روزرسانی تنظیمات
                self.config.websocket_enabled = websocket_config.get('enabled', current_ws_enabled)
                self.config.websocket_ping_interval = websocket_config.get('ping_interval',
                                                                           self.config.websocket_ping_interval)
                self.config.websocket_reconnect_delay = websocket_config.get('reconnect_delay',
                                                                             self.config.websocket_reconnect_delay)

                # اگر وضعیت فعال‌سازی WebSocket تغییر کرده است
                if current_ws_enabled != self.config.websocket_enabled:
                    if self.config.websocket_enabled:
                        logger.info("وب‌سوکت فعال شد. شروع اتصال...")
                        asyncio.create_task(self.start_websocket())
                    else:
                        logger.info("وب‌سوکت غیرفعال شد. قطع اتصال...")
                        asyncio.create_task(self.stop_websocket())

            # به‌روزرسانی سایر تنظیمات
            self.config.debug_mode = exchange_config.get('debug_mode', self.config.debug_mode)
            self.config.health_check_interval = exchange_config.get('health_check_interval',
                                                                    self.config.health_check_interval)

            # به‌روزرسانی وضعیت کلاینت
            self._client_health_status['last_update_time'] = time.time()

            logger.info("تنظیمات کلاینت صرافی با موفقیت به‌روزرسانی شد.")

        except Exception as e:
            logger.error(f"خطا در به‌روزرسانی تنظیمات کلاینت صرافی: {e}", exc_info=True)
            self._client_health_status['last_error'] = {
                'time': time.time(),
                'message': f"خطا در به‌روزرسانی تنظیمات: {str(e)}"
            }

    def _init_redis(self) -> None:
        """راه‌اندازی اتصال Redis"""
        try:
            if not self.config.use_redis_cache:
                logger.info("کش Redis غیرفعال است. اتصال برقرار نمی‌شود.")
                return

            # بستن اتصال قبلی اگر وجود دارد
            if self._redis_client:
                asyncio.create_task(self._redis_client.close())
                self._redis_client = None

            # ایجاد کلاینت Redis
            self._redis_client = redis.Redis(
                host=self.config.redis_host,
                port=self.config.redis_port,
                db=self.config.redis_db,
                password=self.config.redis_password,
                decode_responses=False,
                socket_timeout=5.0,
                socket_connect_timeout=5.0
            )

            # تست اتصال در یک تسک جداگانه
            asyncio.create_task(self._test_redis_connection())
            logger.info(f"اتصال Redis به {self.config.redis_host}:{self.config.redis_port} پیکربندی شد")

        except Exception as e:
            logger.error(f"خطا در راه‌اندازی Redis: {e}", exc_info=True)
            self.config.use_redis_cache = False
            self._redis_client = None
            self._client_health_status['last_error'] = {
                'time': time.time(),
                'message': f"خطا در راه‌اندازی Redis: {str(e)}"
            }

    async def _test_redis_connection(self) -> None:
        """تست اتصال Redis"""
        try:
            if self._redis_client:
                await self._redis_client.ping()
                logger.info("تست اتصال Redis موفقیت‌آمیز بود")
        except Exception as e:
            logger.error(f"تست اتصال Redis ناموفق بود: {e}")
            self.config.use_redis_cache = False
            self._redis_client = None
            self._client_health_status['last_error'] = {
                'time': time.time(),
                'message': f"تست اتصال Redis ناموفق: {str(e)}"
            }

    async def _init_session(self) -> None:
        """
        ایجاد یا بازیابی session HTTP با تنظیمات بهینه
        """
        # استفاده از قفل برای جلوگیری از ایجاد همزمان چندین session
        async with self._session_lock:
            # اگر session از خارج تنظیم شده و معتبر است، از آن استفاده کن
            if self._session is not None and not self._session.closed:
                return

            # تنظیمات TCP بهینه
            connector = aiohttp.TCPConnector(
                limit=100,  # حداکثر 100 اتصال همزمان
                ttl_dns_cache=300,  # کش DNS به مدت 5 دقیقه
                keepalive_timeout=30.0,  # زمان نگهداری اتصال باز
                enable_cleanup_closed=True,  # پاکسازی اتصالات بسته شده
                limit_per_host=20,  # حداکثر 20 اتصال همزمان به هر هاست
                ssl=False  # غیرفعال کردن SSL برای کارایی بیشتر (در صورت نیاز)
            )

            # تنظیمات timeout
            timeout = aiohttp.ClientTimeout(
                total=self.config.request_timeout,
                connect=self.config.connection_timeout,
                sock_read=self.config.request_timeout / 2,
                sock_connect=self.config.connection_timeout
            )

            try:
                self._session = aiohttp.ClientSession(
                    connector=connector,
                    timeout=timeout,
                    headers={"Accept-Encoding": "gzip, deflate"},
                    json_serialize=json.dumps
                )
                logger.info("نشست HTTP جدید با تنظیمات بهینه ایجاد شد")
                self._client_health_status['http_status'] = 'connected'

                # شروع تایمر پاکسازی کش و بررسی سلامت
                self._start_maintenance_tasks()

            except Exception as e:
                logger.critical(f"خطا در ایجاد نشست HTTP: {e}", exc_info=True)
                self._session = None
                self._client_health_status['http_status'] = 'error'
                self._client_health_status['last_error'] = {
                    'time': time.time(),
                    'message': f"خطا در ایجاد نشست HTTP: {str(e)}"
                }
                raise ConnectionError("خطا در راه‌اندازی نشست HTTP") from e

    def _start_maintenance_tasks(self) -> None:
        """راه‌اندازی تسک‌های نگهداری"""
        # اگر قبلا تسک‌ها شروع شده‌اند، آنها را لغو کن
        if self._cache_cleanup_task and not self._cache_cleanup_task.done():
            self._cache_cleanup_task.cancel()
        if self._health_check_task and not self._health_check_task.done():
            self._health_check_task.cancel()

        # ایجاد تسک‌های جدید
        self._cache_cleanup_task = asyncio.create_task(self._cache_cleanup_loop())
        self._health_check_task = asyncio.create_task(self._health_check_loop())

        logger.debug("تسک‌های نگهداری شروع شدند")

    async def _cache_cleanup_loop(self) -> None:
        """پاکسازی دوره‌ای کش"""
        try:
            while True:
                await asyncio.sleep(300)  # هر 5 دقیقه

                async with self._cache_lock:
                    # حذف موارد منقضی شده از کش حافظه
                    current_time = time.time()
                    expired_keys = [k for k, v in self._memory_cache.items() if v.is_expired()]
                    for key in expired_keys:
                        del self._memory_cache[key]

                    # حذف قیمت‌های قدیمی
                    expired_price_keys = [k for k, v in self._price_cache.items() if not v.is_fresh(300)]  # 5 دقیقه
                    for key in expired_price_keys:
                        del self._price_cache[key]

                    # محدود کردن اندازه کش
                    if len(self._memory_cache) > 1000:
                        # حذف قدیمی‌ترین موارد
                        sorted_items = sorted(self._memory_cache.items(), key=lambda x: x[1].timestamp)
                        for key, _ in sorted_items[:len(sorted_items) // 4]:  # حذف 25% قدیمی‌ترین موارد
                            del self._memory_cache[key]

                    logger.debug(
                        f"پاکسازی کش: {len(expired_keys)} مورد منقضی و {len(expired_price_keys)} قیمت قدیمی حذف شدند")

        except asyncio.CancelledError:
            logger.debug("حلقه پاکسازی کش لغو شد")
        except Exception as e:
            logger.error(f"خطا در حلقه پاکسازی کش: {e}")
            # شروع مجدد لوپ در صورت خطا
            await asyncio.sleep(60)
            asyncio.create_task(self._cache_cleanup_loop())

    async def _health_check_loop(self) -> None:
        """بررسی دوره‌ای سلامت سیستم"""
        try:
            while True:
                await asyncio.sleep(self.config.health_check_interval)

                # همگام‌سازی زمان سرور
                time_diff = time.time() - self._last_server_time_sync
                if time_diff > 3600:  # هر ساعت
                    await self._sync_server_time()

                # تست اتصال HTTP
                if self._session and not self._session.closed:
                    try:
                        # یک درخواست ساده برای تست اتصال
                        server_time = await self.get_server_time()
                        if server_time:
                            logger.debug("بررسی سلامت: اتصال HTTP سالم است")
                            self._client_health_status['http_status'] = 'connected'
                        else:
                            logger.warning("بررسی سلامت: آزمایش اتصال HTTP ناموفق بود")
                            self._client_health_status['http_status'] = 'error'
                            await self._reset_http_session()
                    except Exception as e:
                        logger.warning(f"بررسی سلامت: خطا در آزمایش اتصال HTTP: {e}")
                        self._client_health_status['http_status'] = 'error'
                        await self._reset_http_session()
                else:
                    self._client_health_status['http_status'] = 'not_connected'
                    await self._init_session()

                # بررسی وضعیت وب‌سوکت
                if self.config.websocket_enabled:
                    if self._ws_running:
                        time_since_pong = time.time() - self._ws_last_pong_time
                        if time_since_pong > self.config.websocket_ping_interval * 3:
                            logger.warning("بررسی سلامت: اتصال وب‌سوکت مرده به نظر می‌رسد")
                            self._client_health_status['ws_status'] = 'error'
                            await self._restart_websocket()
                        else:
                            self._client_health_status['ws_status'] = 'connected'
                    else:
                        self._client_health_status['ws_status'] = 'not_connected'
                else:
                    self._client_health_status['ws_status'] = 'disabled'

                # به‌روزرسانی وضعیت کلی
                self._client_health_status['status'] = 'healthy'
                if self._client_health_status['http_status'] == 'error':
                    self._client_health_status['status'] = 'degraded'
                if self._client_health_status['http_status'] == 'not_connected':
                    self._client_health_status['status'] = 'critical'

                # لاگ وضعیت سلامت
                if self._client_health_status['status'] != 'healthy':
                    logger.warning(f"وضعیت سلامت کلاینت صرافی: {self._client_health_status['status']}")

        except asyncio.CancelledError:
            logger.debug("حلقه بررسی سلامت لغو شد")
        except Exception as e:
            logger.error(f"خطا در حلقه بررسی سلامت: {e}")
            # شروع مجدد لوپ در صورت خطا
            await asyncio.sleep(60)
            asyncio.create_task(self._health_check_loop())

    async def _reset_http_session(self) -> None:
        """بازنشانی نشست HTTP در صورت بروز مشکل"""
        async with self._session_lock:
            if self._session and not self._session.closed:
                try:
                    await self._session.close()
                except Exception as e:
                    logger.error(f"خطا در بستن نشست HTTP: {e}")

            self._session = None
            await self._init_session()
            logger.info("نشست HTTP بازنشانی شد")

    async def _restart_websocket(self) -> None:
        """راه‌اندازی مجدد اتصال وب‌سوکت"""
        logger.info("شروع راه‌اندازی مجدد وب‌سوکت...")
        try:
            await self.stop_websocket()
            await asyncio.sleep(1)  # تأخیر کوتاه
            await self.start_websocket()
            logger.info("راه‌اندازی مجدد وب‌سوکت با موفقیت انجام شد")
        except Exception as e:
            logger.error(f"خطا در راه‌اندازی مجدد وب‌سوکت: {e}")
            self._client_health_status['ws_status'] = 'error'
            self._client_health_status['last_error'] = {
                'time': time.time(),
                'message': f"خطا در راه‌اندازی مجدد وب‌سوکت: {str(e)}"
            }

    async def _sync_server_time(self) -> None:
        """همگام‌سازی زمان محلی با زمان سرور"""
        try:
            # استفاده از دامنه مخصوص این درخواست
            endpoint = "/api/v1/timestamp"

            # درخواست مستقیم (بدون استفاده از _make_request برای جلوگیری از حلقه)
            if not self._session or self._session.closed:
                await self._init_session()

            url = f"{self.config.base_url}{endpoint}"
            async with self._session.get(url) as response:
                if response.status == 200:
                    response_data = await response.json()
                    if response_data.get('code') == '200000':
                        server_time_ms = response_data.get('data')
                        if server_time_ms and isinstance(server_time_ms, int):
                            local_time_ms = int(time.time() * 1000)
                            self._server_time_diff = server_time_ms - local_time_ms
                            self._last_server_time_sync = time.time()
                            logger.debug(f"زمان سرور همگام‌سازی شد. اختلاف: {self._server_time_diff}ms")
                        else:
                            logger.warning("داده زمان سرور نامعتبر دریافت شد")
                    else:
                        logger.warning(f"خطای API زمان سرور: {response_data.get('msg')}")
                else:
                    logger.warning(f"API زمان سرور کد وضعیت {response.status} برگرداند")

        except Exception as e:
            logger.error(f"خطا در همگام‌سازی زمان سرور: {e}")

    def get_server_adjusted_time(self) -> int:
        """دریافت زمان فعلی با تنظیم براساس اختلاف زمان سرور"""
        local_time_ms = int(time.time() * 1000)
        if self._server_time_diff is not None:
            return local_time_ms + self._server_time_diff
        return local_time_ms

    # @lru_cache(maxsize=512) # غیرفعال کردن کش lru چون از کش داخلی استفاده می‌کنیم
    def _get_kucoin_symbol(self, symbol: str) -> str:
        """
        تبدیل نماد استاندارد (مانند 'BTC/USDT' یا 'BTCUSDT') به فرمت مورد انتظار
        API فیوچرز کوکوین (مانند 'XBTUSDTM').

        Args:
            symbol: نماد در فرمت استاندارد یا فرمت رایج دیگر.

        Returns:
            نماد در فرمت API فیوچرز کوکوین یا نماد اصلی در صورت عدم امکان تبدیل.
        """
        original_symbol = symbol  # ذخیره نماد ورودی برای کش

        # 1. بررسی کش داخلی اولویت دارد
        if original_symbol in self._symbol_info_cache and 'kucoin_symbol' in self._symbol_info_cache[original_symbol]:
            # logger.debug(f"Cache hit for {original_symbol} -> {self._symbol_info_cache[original_symbol]['kucoin_symbol']}")
            return self._symbol_info_cache[original_symbol]['kucoin_symbol']

        logger.debug(f"تبدیل نماد '{original_symbol}' به فرمت KuCoin Futures...")

        try:
            # 2. بررسی فرمت‌های رایج فیوچرز کوکوین (اگر ورودی خودش در این فرمت بود)
            if symbol.endswith("USDTM") and not symbol.startswith("XBT"):  # Handle case like BTCUSDTM -> XBTUSDTM
                if symbol.startswith("BTC"):
                    kucoin_symbol = "XBT" + symbol[3:]
                    logger.debug(f"نماد '{original_symbol}' شبیه فرمت فیوچرز است، BTC به XBT تبدیل شد: {kucoin_symbol}")
                    self._cache_symbol_mapping(original_symbol, kucoin_symbol)
                    return kucoin_symbol
                elif "/" not in symbol:  # It already seems like a valid futures symbol (non-BTC)
                    logger.debug(f"نماد '{original_symbol}' در حال حاضر در فرمت صحیح فیوچرز غیر BTC است.")
                    self._cache_symbol_mapping(original_symbol, symbol)
                    return symbol
            elif symbol.endswith("USDTM") and symbol.startswith("XBT"):  # Already correct format like XBTUSDTM
                logger.debug(f"نماد '{original_symbol}' از قبل در فرمت صحیح XBT فیوچرز است.")
                self._cache_symbol_mapping(original_symbol, symbol)
                return symbol
            elif symbol.endswith("USD") and not symbol.endswith("USDTM"):  # Handle futures like BTCUSD0628
                logger.debug(f"نماد '{original_symbol}' به نظر یک قرارداد فیوچرز تاریخ‌دار است.")
                self._cache_symbol_mapping(original_symbol, symbol)
                return symbol  # Assume dated futures are passed as is

            # 3. تلاش برای جداسازی BASE/QUOTE
            base = None
            quote = None
            symbol_upper = original_symbol.upper()

            if '/' in symbol_upper:
                parts = symbol_upper.split('/')
                if len(parts) == 2:
                    base, quote = parts
            elif symbol_upper.endswith('USDT'):
                # تلاش برای حدس زدن از فرمت BASEQUOTE
                base = symbol_upper[:-4]
                quote = 'USDT'
            # می‌توانید الگوهای رایج دیگر را هم اینجا اضافه کنید (مثل ETHBTC)

            if not base or not quote:
                logger.warning(
                    f"نمی‌توان base/quote را از نماد '{original_symbol}' استخراج کرد. بازگرداندن به همان شکل.")
                # در کش هم ذخیره می‌کنیم که دوباره تلاش نکنیم
                self._cache_symbol_mapping(original_symbol, original_symbol.upper())
                return original_symbol.upper()

            # 4. تبدیل‌های خاص (BTC به XBT)
            if base == "BTC":
                base = "XBT"
                logger.debug("تبدیل پایه 'BTC' به 'XBT'")
            # سایر تبدیل‌ها را اینجا اضافه کنید

            # 5. ساخت نماد نهایی فیوچرز (فرض USDT Perpetual)
            # TODO: مدیریت بهتر سایر Quote ها یا انواع قرارداد (مثل تاریخ‌دار)
            if quote == "USDT":
                # فرمت رایج برای قراردادهای دائمی USDT
                kucoin_symbol = f"{base}{quote}M".upper()
            else:
                # مدیریت سایر ارزهای Quote (ممکن است نیاز به بررسی بیشتر داشته باشد)
                logger.warning(
                    f"ارز quote '{quote}' برای '{original_symbol}'. فرض پسوند قرارداد دائمی 'M'. صحت آن را بررسی کنید.")
                # فعلا همان فرمت USDTM را فرض می‌گیریم، شاید نیاز به اصلاح باشد
                kucoin_symbol = f"{base}{quote}M".upper()

            # 6. ذخیره در کش و بازگرداندن نتیجه
            self._cache_symbol_mapping(original_symbol, kucoin_symbol)
            logger.info(f"تبدیل '{original_symbol}' به نماد KuCoin Futures: '{kucoin_symbol}'")
            return kucoin_symbol

        except Exception as e:
            logger.error(
                f"خطا در تبدیل نماد '{original_symbol}' به فرمت KuCoin Futures: {e}. بازگرداندن اصلی (در حالت بزرگ).",
                exc_info=True)
            # در صورت خطا، نماد اصلی را برمی‌گردانیم تا حداقل تلاشی انجام شود
            return original_symbol.upper()

    def _cache_symbol_mapping(self, original_symbol: str, kucoin_symbol: str) -> None:
        """ذخیره نگاشت نماد در کش داخلی."""
        asyncio.create_task(self._async_cache_symbol_mapping(original_symbol, kucoin_symbol))

    async def _async_cache_symbol_mapping(self, original_symbol: str, kucoin_symbol: str) -> None:
        """ورژن آسنکرون برای ذخیره نگاشت نماد در کش داخلی."""
        async with self._cache_lock:  # Ensure thread safety for cache access
            if original_symbol not in self._symbol_info_cache:
                self._symbol_info_cache[original_symbol] = {}
            self._symbol_info_cache[original_symbol]['kucoin_symbol'] = kucoin_symbol
            # Optionally log cache update if needed for debugging
            # logger.debug(f"Cached mapping: {original_symbol} -> {kucoin_symbol}")

    def _create_signature(self, timestamp: str, method: str, endpoint: str, body_or_query: str = '') -> str:
        """
        ایجاد امضای HMAC-SHA256 برای احراز هویت API.

        Args:
            timestamp: زمان به میلی‌ثانیه
            method: متد HTTP (GET, POST, ...)
            endpoint: مسیر درخواست
            body_or_query: بدنه یا query string درخواست

        Returns:
            امضای Base64 برای استفاده در هدر

        Raises:
            ValueError: اگر API Secret تنظیم نشده باشد
        """
        if not self.config.api_secret:
            raise ValueError("کلید مخفی API تنظیم نشده است. امکان ایجاد امضا وجود ندارد.")

        # رشته مورد نیاز برای امضا: timestamp + method + endpoint + body/querystring
        str_to_sign = f"{timestamp}{method.upper()}{endpoint}{body_or_query}"

        if self.config.debug_mode:
            logger.debug(f"رشته برای امضا: {str_to_sign}")  # حساس! در محیط تولید حذف شود

        # ایجاد امضا با hmac-sha256
        signed_object = hmac.new(
            self.config.api_secret.encode('utf-8'),
            str_to_sign.encode('utf-8'),
            hashlib.sha256
        ).digest()

        # انکود Base64
        signature = base64.b64encode(signed_object).decode('utf-8')

        if self.config.debug_mode:
            logger.debug(f"امضای تولید شده (base64): {signature}")  # حساس!

        return signature

    async def _apply_rate_limit(self, rate_limit_key: str) -> None:
        """
        اعمال محدودیت نرخ برای یک اندپوینت خاص

        Args:
            rate_limit_key: کلید اندپوینت برای اعمال محدودیت نرخ
        """
        # فعال‌سازی backoff برای اندپوینت‌هایی که نقض rate limit داشته‌اند
        backoff_multiplier = 1.0
        if rate_limit_key in self._rate_limit_violations and self._rate_limit_violations[rate_limit_key] > 0:
            # افزایش تاخیر با توجه به تعداد نقض‌ها
            violations = self._rate_limit_violations[rate_limit_key]
            backoff_multiplier = min(5.0, 1.0 + (violations * 0.15))

        # محاسبه حداقل فاصله زمانی بین درخواست‌ها
        if rate_limit_key not in self._min_request_interval:
            self._min_request_interval[rate_limit_key] = self._rate_limit_time_window

        min_interval = self._min_request_interval[rate_limit_key] * backoff_multiplier

        # بررسی زمان آخرین درخواست و انتظار در صورت نیاز
        current_time = time.time()
        if rate_limit_key in self._endpoint_last_request:
            elapsed_time = current_time - self._endpoint_last_request[rate_limit_key]
            if elapsed_time < min_interval:
                wait_time = min_interval - elapsed_time
                if wait_time > 0.05:  # فقط برای تاخیرهای قابل توجه
                    await asyncio.sleep(wait_time)

        # به‌روزرسانی زمان آخرین درخواست
        self._endpoint_last_request[rate_limit_key] = time.time()

    async def _handle_rate_limit_violation(self, rate_limit_key: str, retry_after: Optional[float] = None) -> None:
        """
        ثبت و مدیریت نقض محدودیت نرخ

        Args:
            rate_limit_key: کلید اندپوینت که نقض محدودیت نرخ داشته
            retry_after: زمان پیشنهادی برای انتظار قبل از درخواست بعدی (ثانیه)
        """
        # افزایش شمارنده نقض‌ها
        if rate_limit_key not in self._rate_limit_violations:
            self._rate_limit_violations[rate_limit_key] = 0
        self._rate_limit_violations[rate_limit_key] += 1

        # تنظیم حداقل فاصله زمانی برای این اندپوینت
        current_interval = self._min_request_interval.get(rate_limit_key, self._rate_limit_time_window)

        # اگر retry_after مشخص شده، از آن استفاده کن
        if retry_after:
            new_interval = max(current_interval, retry_after)
        else:
            # افزایش تاخیر به صورت نمایی
            violations = self._rate_limit_violations[rate_limit_key]
            backoff_multiplier = min(5.0, 1.0 + (violations * 0.15))
            new_interval = self._rate_limit_time_window * backoff_multiplier

        self._min_request_interval[rate_limit_key] = new_interval

        logger.warning(f"محدودیت نرخ برای {rate_limit_key} نقض شد. فاصله جدید: {new_interval:.2f}s")

    async def _recover_from_rate_limit_violation(self, rate_limit_key: str) -> None:
        """
        بهبود تدریجی از نقض محدودیت نرخ

        Args:
            rate_limit_key: کلید اندپوینت
        """
        # کاهش تدریجی شمارنده نقض‌ها
        if rate_limit_key in self._rate_limit_violations and self._rate_limit_violations[rate_limit_key] > 0:
            self._rate_limit_violations[rate_limit_key] -= 1

            # در صورت رسیدن به صفر، بازنشانی فاصله زمانی به مقدار پیش‌فرض
            if self._rate_limit_violations[rate_limit_key] == 0:
                self._min_request_interval[rate_limit_key] = self._rate_limit_time_window
                logger.info(f"بهبود کامل محدودیت نرخ برای {rate_limit_key}")
            else:
                # کاهش تدریجی فاصله زمانی
                current_interval = self._min_request_interval.get(rate_limit_key, self._rate_limit_time_window)
                new_interval = max(self._rate_limit_time_window, current_interval * 0.9)  # کاهش 10 درصدی
                self._min_request_interval[rate_limit_key] = new_interval
                logger.debug(f"پیشرفت بهبود محدودیت نرخ برای {rate_limit_key}. فاصله جدید: {new_interval:.2f}s")

    async def _fetch_with_cache(
            self,
            method: str,
            endpoint: str,
            rate_limit_key: str,
            params: Optional[Dict[str, Any]] = None,
            signed: bool = False,
            cache_ttl: Optional[int] = None
    ) -> Optional[Any]:
        """
        درخواست HTTP با کش چندلایه

        Args:
            method: متد HTTP
            endpoint: مسیر درخواست
            rate_limit_key: کلید برای مدیریت rate limit
            params: پارامترهای درخواست
            signed: آیا درخواست نیاز به امضا دارد
            cache_ttl: زمان انقضای کش (ثانیه)

        Returns:
            داده‌های دریافتی یا None در صورت خطا
        """
        # ساخت کلید کش
        clean_params = params or {}
        cache_key = f"{method}:{endpoint}:{json.dumps(clean_params, sort_keys=True)}"

        # تعیین TTL کش (حافظه و Redis)
        memory_ttl = cache_ttl or self.config.memory_cache_ttl
        redis_ttl = cache_ttl or self.config.redis_cache_ttl

        # بررسی کش حافظه
        memory_result = await self._get_from_memory_cache(cache_key, memory_ttl)
        if memory_result is not None:
            self._api_stats['cache_hits'] += 1
            return memory_result

        # بررسی کش Redis
        if self.config.use_redis_cache and self._redis_client:
            redis_result = await self._get_from_redis_cache(cache_key)
            if redis_result is not None:
                # ذخیره در کش حافظه برای دسترسی‌های بعدی
                await self._set_in_memory_cache(cache_key, redis_result, memory_ttl)
                self._api_stats['cache_hits'] += 1
                return redis_result

        # اعمال محدودیت نرخ درخواست
        await self._apply_rate_limit(rate_limit_key)

        # دریافت از API
        response = await self._make_request(method, endpoint, params, signed)

        # ذخیره در کش‌ها اگر درخواست موفق بود
        if response is not None:
            # بهبود از نقض محدودیت نرخ در صورت موفقیت
            await self._recover_from_rate_limit_violation(rate_limit_key)

            # ذخیره در کش حافظه
            await self._set_in_memory_cache(cache_key, response, memory_ttl)

            # ذخیره در کش Redis
            if self.config.use_redis_cache and self._redis_client:
                await self._set_in_redis_cache(cache_key, response, redis_ttl)

        return response

    async def _get_from_memory_cache(self, key: str, ttl: int) -> Optional[Any]:
        """
        دریافت داده از کش حافظه

        Args:
            key: کلید کش
            ttl: زمان انقضا (ثانیه)

        Returns:
            داده کش شده یا None
        """
        async with self._cache_lock:
            if key in self._memory_cache:
                cache_entry = self._memory_cache[key]
                if not cache_entry.is_expired():
                    if self.config.debug_mode:
                        logger.debug(f"یافتن کش حافظه برای {key}")
                    return cache_entry.data
                else:
                    # حذف داده منقضی شده
                    del self._memory_cache[key]
        return None

    async def _set_in_memory_cache(self, key: str, data: Any, ttl: int) -> None:
        """
        ذخیره داده در کش حافظه

        Args:
            key: کلید کش
            data: داده برای ذخیره
            ttl: زمان انقضا (ثانیه)
        """
        async with self._cache_lock:
            self._memory_cache[key] = CacheEntry(data=data, timestamp=time.time(), expiry=ttl)

            # بررسی سایز کش و پاکسازی در صورت نیاز
            if len(self._memory_cache) > 1000:  # محدودیت حداکثر 1000 آیتم
                # حذف قدیمی‌ترین آیتم‌ها
                entries = [(k, v.timestamp) for k, v in self._memory_cache.items()]
                entries.sort(key=lambda x: x[1])  # مرتب‌سازی بر اساس زمان (قدیمی‌ترین اول)

                # حذف 25% قدیمی‌ترین موارد
                for k, _ in entries[:len(entries) // 4]:
                    del self._memory_cache[k]

    async def _get_from_redis_cache(self, key: str) -> Optional[Any]:
        """
        دریافت داده از کش Redis

        Args:
            key: کلید کش

        Returns:
            داده کش شده یا None
        """
        if not self._redis_client:
            return None

        try:
            async with self._redis_lock:
                # دریافت داده از Redis
                cached_data = await self._redis_client.get(key)

                if cached_data:
                    # تبدیل داده JSON به Python
                    return json.loads(cached_data.decode('utf-8'))

        except Exception as e:
            logger.error(f"خطا در بازیابی داده از کش Redis: {e}")

        return None

    async def _set_in_redis_cache(self, key: str, data: Any, ttl: int) -> None:
        """
        ذخیره داده در کش Redis

        Args:
            key: کلید کش
            data: داده برای ذخیره
            ttl: زمان انقضا (ثانیه)
        """
        if not self._redis_client:
            return

        try:
            async with self._redis_lock:
                # تبدیل داده به JSON
                json_data = json.dumps(data).encode('utf-8')

                # ذخیره در Redis با زمان انقضا
                await self._redis_client.setex(key, ttl, json_data)

        except Exception as e:
            logger.error(f"خطا در ذخیره داده در کش Redis: {e}")

    async def _make_request(
            self,
            method: str,
            endpoint: str,
            params: Optional[Dict[str, Any]] = None,
            signed: bool = False,
            retry_count: Optional[int] = None
    ) -> Optional[Union[Dict, List]]:
        """
        انجام درخواست HTTP به API

        Args:
            method: متد HTTP (GET, POST, ...)
            endpoint: مسیر درخواست
            params: پارامترهای درخواست
            signed: آیا درخواست نیاز به امضا دارد
            retry_count: تعداد تلاش مجدد در صورت خطا

        Returns:
            پاسخ از API یا None در صورت خطا
        """
        # اطمینان از وجود aiohttp Session
        if not self._session or self._session.closed:
            await self._init_session()

        if self._session is None:
            logger.error("امکان ارسال درخواست وجود ندارد: نشست اتصال موجود نیست.")
            self._api_stats['error_count'] += 1
            return None

        # تنظیم تعداد تلاش مجدد
        retries_left = retry_count if retry_count is not None else self.config.retry_count

        while retries_left >= 0:
            start_time = time.time()
            url = f"{self.config.base_url}{endpoint}"
            headers = {'Content-Type': 'application/json', 'Accept': 'application/json'}
            body_str = ''  # بدنه درخواست برای POST/PUT (رشته JSON)
            query_string = ''  # Query String برای GET/DELETE
            request_params = None  # پارامترهای Query برای aiohttp
            data_payload = None  # بدنه درخواست برای aiohttp

            # حذف مقادیر None از پارامترها
            clean_params = {k: v for k, v in params.items() if v is not None} if params else {}

            if method.upper() in ['GET', 'DELETE']:
                if clean_params:
                    # برای GET/DELETE، پارامترها به query string تبدیل می‌شوند
                    query_string = urlencode(clean_params, safe='-')
                    request_params = clean_params  # aiohttp خودش query string را می‌سازد
            else:  # POST, PUT
                if clean_params:
                    # برای POST/PUT، پارامترها به رشته JSON تبدیل و در بدنه قرار می‌گیرند
                    body_str = json.dumps(clean_params)
                    data_payload = body_str

            # افزودن هدرهای احراز هویت در صورت نیاز
            if signed:
                if not all([self.config.api_key, self.config.api_secret, self.config.api_passphrase]):
                    logger.error(f"اطلاعات احراز هویت برای درخواست امضا شده به {endpoint} ناقص است")
                    return None

                # ایجاد timestamp به میلی‌ثانیه
                timestamp = str(self.get_server_adjusted_time())

                try:
                    # ایجاد امضا (برای GET/DELETE از query_string، برای POST/PUT از body_str)
                    sig_payload = query_string if method.upper() in ['GET', 'DELETE'] else body_str
                    signature = self._create_signature(timestamp, method, endpoint, sig_payload)
                except ValueError as e:
                    logger.error(f"خطا در ایجاد امضا: {e}")
                    return None

                # افزودن هدرهای لازم
                headers['KC-API-KEY'] = self.config.api_key
                headers['KC-API-SIGN'] = signature
                headers['KC-API-TIMESTAMP'] = timestamp
                headers['KC-API-PASSPHRASE'] = self.config.api_passphrase
                headers['KC-API-KEY-VERSION'] = self.config.api_version

            # ثبت آمار درخواست
            self._api_stats['requests_count'] += 1
            self._api_stats['last_request_time'] = start_time

            # ارسال درخواست
            try:
                async with self._request_lock:
                    if self.config.debug_mode:
                        log_params_str = (f"پارامترها: {request_params}" if request_params else
                                          f"بدنه: {body_str[:150]}..." if body_str else "هیچ")
                        logger.debug(f"ارسال {method} به API: {endpoint} | {log_params_str}")

                    async with self._session.request(
                            method, url, headers=headers, data=data_payload, params=request_params
                    ) as response:
                        status_code = response.status
                        response_time = (time.time() - start_time) * 1000  # به میلی‌ثانیه

                        # ثبت آمار
                        self._api_stats['response_times'].append(response_time)
                        self._api_stats['status_codes'][status_code] = self._api_stats['status_codes'].get(status_code,
                                                                                                           0) + 1

                        # ثبت آمار اندپوینت
                        endpoint_name = endpoint.split('/')[-1]
                        if endpoint_name not in self._api_stats['endpoint_stats']:
                            self._api_stats['endpoint_stats'][endpoint_name] = {
                                'total': 0,
                                'success': 0,
                                'failures': 0,
                                'avg_time': 0,
                                'response_times': []
                            }

                        endpoint_stats = self._api_stats['endpoint_stats'][endpoint_name]
                        endpoint_stats['total'] += 1
                        endpoint_stats['response_times'].append(response_time)
                        if len(endpoint_stats['response_times']) > 100:
                            endpoint_stats['response_times'] = endpoint_stats['response_times'][-100:]
                        endpoint_stats['avg_time'] = sum(endpoint_stats['response_times']) / len(
                            endpoint_stats['response_times'])

                        # بررسی rate limit
                        if status_code == 429:
                            retry_after = float(response.headers.get('Retry-After', 1))
                            await self._handle_rate_limit_violation(endpoint.split('/')[1], retry_after)
                            endpoint_stats['failures'] += 1

                            # تاخیر قبل از تلاش مجدد
                            wait_time = retry_after + random.uniform(0, 0.5)  # افزودن jitter
                            logger.warning(
                                f"محدودیت نرخ برای {endpoint} نقض شد. انتظار {wait_time:.2f}s قبل از تلاش مجدد.")
                            await asyncio.sleep(wait_time)
                            retries_left -= 1
                            continue

                        # بررسی خطاهای HTTP
                        if status_code >= 500:
                            # خطای سرور - تلاش مجدد
                            endpoint_stats['failures'] += 1
                            wait_time = self.config.retry_delay_base * (
                                    2 ** (self.config.retry_count - retries_left)) + random.uniform(0, 1)
                            logger.warning(f"خطای سرور {status_code} برای {endpoint}. تلاش مجدد در {wait_time:.2f}s")
                            await asyncio.sleep(wait_time)
                            retries_left -= 1
                            continue
                        elif status_code >= 400:
                            # خطای کلاینت - ممکن است تلاش مجدد سودی نداشته باشد
                            endpoint_stats['failures'] += 1
                            logger.error(f"خطای کلاینت {status_code} برای {endpoint}")

                            # برخی خطاهای کلاینت نیاز به تلاش مجدد ندارند
                            if status_code in [400, 401, 403, 404]:
                                response_text = await response.text()
                                logger.error(f"خطای API: {response_text[:200]}")
                                return None

                            # برای سایر خطاها، تلاش مجدد
                            wait_time = self.config.retry_delay_base * (2 ** (self.config.retry_count - retries_left))
                            await asyncio.sleep(wait_time)
                            retries_left -= 1
                            continue

                        # دریافت و تحلیل پاسخ
                        response_text = await response.text()

                        try:
                            response_data = json.loads(response_text)
                        except json.JSONDecodeError:
                            logger.error(f"پاسخ JSON نامعتبر از {endpoint}: {response_text[:200]}")
                            endpoint_stats['failures'] += 1
                            return None

                        # بررسی کد موفقیت API
                        if response_data.get('code') == '200000':
                            # پاسخ موفق
                            endpoint_stats['success'] += 1
                            return response_data.get('data')
                        else:
                            # خطای API
                            error_code = response_data.get('code')
                            error_msg = response_data.get('msg', 'خطای ناشناخته')
                            logger.error(f"خطای API برای {endpoint}: کد={error_code}، پیام='{error_msg}'")
                            endpoint_stats['failures'] += 1

                            # برخی کدهای خطا نیاز به تلاش مجدد دارند
                            if error_code in ['30000', '30001', '30002']:  # خطاهای موقتی
                                wait_time = self.config.retry_delay_base * (
                                        2 ** (self.config.retry_count - retries_left))
                                await asyncio.sleep(wait_time)
                                retries_left -= 1
                                continue

                            return None  # برای سایر خطاها

            except (aiohttp.ClientError, asyncio.TimeoutError) as e:
                # خطای شبکه یا timeout
                logger.warning(f"خطای درخواست برای {endpoint}: {type(e).__name__}: {e}")
                self._api_stats['error_count'] += 1

                # تلاش مجدد با backoff
                if retries_left > 0:
                    wait_time = self.config.retry_delay_base * (2 ** (self.config.retry_count - retries_left))
                    logger.info(f"تلاش مجدد در {wait_time:.2f}s ({retries_left} تلاش باقی‌مانده)")
                    await asyncio.sleep(wait_time)

                retries_left -= 1
                continue

            except Exception as e:
                # سایر خطاهای غیرمنتظره
                logger.error(f"خطای غیرمنتظره برای {endpoint}: {type(e).__name__}: {e}", exc_info=True)
                self._api_stats['error_count'] += 1
                return None

        # اگر به اینجا برسیم، همه تلاش‌ها ناموفق بوده‌اند
        logger.error(f"همه تلاش‌ها برای {endpoint} ناموفق بود")
        return None

    async def get_server_time(self) -> Optional[int]:
        """
        دریافت زمان سرور (ms).

        Returns:
            زمان سرور به میلی‌ثانیه یا None در صورت خطا
        """
        endpoint = "/api/v1/timestamp"
        rate_limit_key = "public"

        # استفاده از کش کوتاه‌مدت
        response_data = await self._fetch_with_cache(
            'GET', endpoint, rate_limit_key, signed=False, cache_ttl=5  # کش 5 ثانیه‌ای
        )

        if response_data and isinstance(response_data, int):
            # محاسبه تفاوت زمانی با سرور
            local_time_ms = int(time.time() * 1000)
            self._server_time_diff = response_data - local_time_ms
            self._last_server_time_sync = time.time()

            if abs(self._server_time_diff) > 30000:  # بیش از 30 ثانیه
                logger.warning(f"اختلاف زمانی زیاد با سرور: {self._server_time_diff}ms")

            if self.config.debug_mode:
                logger.debug(f"زمان سرور (ms): {response_data}، اختلاف: {self._server_time_diff}ms")

            return response_data
        else:
            logger.error("دریافت زمان سرور ناموفق بود.")
            return None

    def normalize_timestamp(self, timestamp: int) -> int:
        """
        نرمال‌سازی timestamp به فرمت میلی‌ثانیه

        Args:
            timestamp: مقدار timestamp برای نرمال‌سازی

        Returns:
            timestamp نرمال‌شده (میلی‌ثانیه)
        """
        # تشخیص فرمت فعلی
        if timestamp <= 9999999999:  # ثانیه (کمتر از 11 رقم)
            return timestamp * 1000
        elif timestamp <= 9999999999999:  # میلی‌ثانیه (11-13 رقم)
            return timestamp
        else:  # میکروثانیه یا نانوثانیه
            return timestamp // 1000  # تبدیل به میلی‌ثانیه

    @staticmethod
    def timeframe_to_ms(timeframe: str) -> Optional[int]:
        """
        تبدیل رشته تایم‌فریم به میلی‌ثانیه

        Args:
            timeframe: رشته تایم‌فریم (مثلاً '5m', '1h')

        Returns:
            مدت زمان به میلی‌ثانیه یا None در صورت خطا
        """
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
            logger.warning(f"خطا در تجزیه مدت زمان تایم‌فریم: {timeframe}")
            return None

    async def fetch_ohlcv(
            self,
            symbol: str,
            timeframe: str,
            limit: int = 200,
            start_time_s: Optional[int] = None,
            end_time_s: Optional[int] = None,
            since: Optional[int] = None
    ) -> Optional[pd.DataFrame]:
        """
        دریافت کندل‌های OHLCV با پشتیبانی از تعداد زیاد کندل‌ها.
        اگر limit بیشتر از 200 باشد، درخواست‌های متعدد ارسال می‌شود.

        Args:
            symbol: نماد ارز (مثلا 'BTC/USDT')
            timeframe: تایم‌فریم (مثلا '1m', '1h', '1d')
            limit: حداکثر تعداد کندل‌ها
            start_time_s: زمان شروع به ثانیه (اختیاری)
            end_time_s: زمان پایان به ثانیه (اختیاری)
            since: زمان شروع به میلی‌ثانیه (اختیاری، برای سازگاری)
        """
        kucoin_symbol = self._get_kucoin_symbol(symbol)
        granularity = TIMEFRAME_MAP.get(timeframe)

        if not granularity:
            logger.error(f"تایم‌فریم نامعتبر: {timeframe}")
            return None

        # اگر since داده شده باشد، آن را تبدیل به start_time_s می‌کنیم
        if since is not None:
            start_time_s = since / 1000

        # اگر limit بیشتر از 200 نباشد، از روش عادی استفاده می‌کنیم
        if limit <= 200:
            endpoint = "/api/v1/kline/query"
            params = {'symbol': kucoin_symbol, 'granularity': granularity}

            if start_time_s:
                params['from'] = int(start_time_s * 1000)
            if end_time_s:
                params['to'] = int(end_time_s * 1000)

            return await self._fetch_single_ohlcv(endpoint, params, symbol, timeframe)

        # برای limit بیشتر از 200، از چندین درخواست استفاده می‌کنیم
        all_candles = []
        batch_size = 200
        remaining = limit

        # محاسبه زمان پایان اگر مشخص نشده باشد
        if end_time_s is None:
            end_time_s = time.time()

        # محاسبه طول کندل به ثانیه
        candle_duration_ms = self.timeframe_to_ms(timeframe)
        candle_duration_s = candle_duration_ms / 1000 if candle_duration_ms else 3600  # پیش‌فرض 1 ساعت

        # اگر زمان شروع مشخص نشده، آن را محاسبه می‌کنیم
        if start_time_s is None:
            start_time_s = end_time_s - (limit * candle_duration_s)

        current_end_time = end_time_s

        while remaining > 0 and current_end_time > start_time_s:
            batch_limit = min(batch_size, remaining)

            # محاسبه زمان شروع برای این batch
            batch_start_time = current_end_time - (batch_limit * candle_duration_s)

            endpoint = "/api/v1/kline/query"
            params = {
                'symbol': kucoin_symbol,
                'granularity': granularity,
                'from': int(batch_start_time * 1000),
                'to': int(current_end_time * 1000)
            }

            # دریافت کندل‌ها
            batch_df = await self._fetch_single_ohlcv(endpoint, params, symbol, timeframe)

            if batch_df is None or batch_df.empty:
                # اگر داده‌ای دریافت نشد، احتمالاً به ابتدای داده‌ها رسیده‌ایم
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
                break

        if not all_candles:
            logger.warning(f"هیچ داده‌ای برای {symbol} {timeframe} بازگردانده نشد")
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

        logger.info(f"با موفقیت {len(final_df)} کندل برای {symbol} {timeframe} دریافت شد")
        return final_df

    async def _fetch_single_ohlcv(
            self,
            endpoint: str,
            params: Dict[str, Any],
            symbol: str,
            timeframe: str
    ) -> Optional[pd.DataFrame]:
        """
        دریافت یک batch از کندل‌های OHLCV

        Args:
            endpoint: مسیر API
            params: پارامترهای درخواست
            symbol: نماد ارز
            timeframe: تایم‌فریم
        """
        rate_limit_key = "kline"

        # درخواست با پشتیبانی کش
        response_data = await self._fetch_with_cache(
            'GET', endpoint, rate_limit_key, params=params, signed=False, cache_ttl=60
        )

        # پردازش پاسخ
        if response_data and isinstance(response_data, list):
            if not response_data:
                return pd.DataFrame()

            try:
                # ساخت دیتافریم
                columns = ['timestamp', 'open', 'close', 'high', 'low', 'volume', 'turnover']
                df = pd.DataFrame(response_data, columns=columns[:len(response_data[0])])

                # تبدیل ستون timestamp به datetime
                df['timestamp'] = pd.to_numeric(df['timestamp'], errors='coerce')
                df = df.dropna(subset=['timestamp'])

                # نرمال‌سازی timestamp
                df['timestamp'] = df['timestamp'].apply(self.normalize_timestamp)
                df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms', utc=True)

                # تنظیم timestamp به عنوان ایندکس
                df = df.set_index('timestamp')

                # تبدیل ستون‌های قیمت به عددی
                numeric_cols = ['open', 'high', 'low', 'close', 'volume']
                for col in numeric_cols:
                    if col in df.columns:
                        df[col] = pd.to_numeric(df[col], errors='coerce')

                # حذف ردیف‌های با مقادیر نامعتبر
                df = df.dropna(subset=numeric_cols[:4])  # حذف ردیف‌هایی که قیمت ندارند

                # مرتب‌سازی بر اساس زمان
                df = df.sort_index()

                return df
            except Exception as e:
                logger.error(f"خطا در پردازش داده‌های OHLCV برای {symbol} {timeframe}: {e}", exc_info=True)
                return None
        else:
            logger.error(f"خطا در دریافت داده‌های OHLCV برای {symbol} {timeframe}")
            return None

    async def get_all_usdt_symbols(
            self,
            contract_type: ContractType = ContractType.PERPETUAL
    ) -> List[str]:
        """
        دریافت لیست قراردادهای فعال USDT.

        Args:
            contract_type: نوع قرارداد (PERPETUAL, FUTURES, ALL)

        Returns:
            لیست نمادها در فرمت استاندارد (مثلا ['BTC/USDT', 'ETH/USDT'])
        """
        endpoint = "/api/v1/contracts/active"
        rate_limit_key = "public"

        # استفاده از کش 5 دقیقه‌ای
        response_data = await self._fetch_with_cache(
            'GET', endpoint, rate_limit_key, signed=False, cache_ttl=300
        )

        usdt_symbols: List[str] = []

        if response_data and isinstance(response_data, list):
            for contract in response_data:
                if isinstance(contract, dict):
                    symbol_futures = contract.get('symbol')
                    quote_currency = contract.get('quoteCurrency')
                    status = contract.get('status', '').lower()
                    contract_type_val = contract.get('type', '').lower()

                    # فیلتر بر اساس نوع قرارداد
                    if contract_type != ContractType.ALL:
                        is_perpetual = 'M' in symbol_futures and not any(c.isdigit() for c in symbol_futures)

                        if contract_type == ContractType.PERPETUAL and not is_perpetual:
                            continue
                        elif contract_type == ContractType.FUTURES and is_perpetual:
                            continue

                    # ذخیره اطلاعات قرارداد در کش
                    if symbol_futures:
                        contract_info = {
                            'symbol': symbol_futures,
                            'quote_currency': quote_currency,
                            'status': status,
                            'type': contract.get('type'),
                            'multiplier': contract.get('multiplier'),
                            'index_price': contract.get('indexPrice'),
                            'mark_price': contract.get('markPrice'),
                            'funding_rate': contract.get('fundingRate'),
                            'max_leverage': contract.get('maxLeverage')
                        }
                        self._symbol_info_cache[symbol_futures] = contract_info

                    # فیلتر بر اساس USDT و وضعیت Open
                    if quote_currency == 'USDT' and status == 'open':
                        # تبدیل فرمت فیوچرز به BASE/QUOTE
                        if symbol_futures.endswith('USDTM'):
                            base = symbol_futures[:-5]
                            # تبدیل XBT به BTC
                            if base == "XBT":
                                base = "BTC"

                            standard_symbol = f"{base}/USDT"
                            usdt_symbols.append(standard_symbol)

                            # ذخیره نگاشت بین فرمت استاندارد و فرمت صرافی
                            if standard_symbol not in self._symbol_info_cache:
                                self._symbol_info_cache[standard_symbol] = {}
                            self._symbol_info_cache[standard_symbol]['kucoin_symbol'] = symbol_futures

            logger.info(f"{len(usdt_symbols)} قرارداد فعال USDT {contract_type.value} دریافت شد.")
        else:
            logger.error("دریافت لیست قراردادهای فعال ناموفق بود.")

        return sorted(usdt_symbols)

    async def get_contract_details(self, symbol: str) -> Optional[Dict[str, Any]]:
        """
        دریافت جزئیات قرارداد

        Args:
            symbol: نماد ارز (مثلا 'BTC/USDT')

        Returns:
            دیکشنری با جزئیات قرارداد یا None در صورت خطا
        """
        kucoin_symbol = self._get_kucoin_symbol(symbol)
        endpoint = f"/api/v1/contracts/{kucoin_symbol}"
        rate_limit_key = "public"

        # استفاده از کش 5 دقیقه‌ای
        response_data = await self._fetch_with_cache(
            'GET', endpoint, rate_limit_key, signed=False, cache_ttl=300
        )

        if response_data and isinstance(response_data, dict):
            # ذخیره در کش
            if symbol not in self._symbol_info_cache:
                self._symbol_info_cache[symbol] = {}
            self._symbol_info_cache[symbol].update(response_data)

            return response_data
        else:
            logger.error(f"دریافت جزئیات قرارداد برای {symbol} ناموفق بود")
            return None

    async def get_current_price(self, symbol: str) -> Optional[float]:
        """
        دریافت قیمت فعلی

        Args:
            symbol: نماد ارز (مثلا 'BTC/USDT')

        Returns:
            قیمت فعلی یا None در صورت خطا
        """
        # بررسی وب‌سوکت - اولویت اول
        if symbol in self._ws_prices and self._ws_prices[symbol].is_fresh(5):  # 5 ثانیه
            return self._ws_prices[symbol].price

        # اگر قیمت در کش موجود است و تازه است، از آن استفاده کن
        if symbol in self._price_cache and self._price_cache[symbol].is_fresh(5):  # 5 ثانیه
            return self._price_cache[symbol].price

        # در غیر این صورت، از API دریافت کن
        kucoin_symbol = self._get_kucoin_symbol(symbol)
        endpoint = f"/api/v1/contracts/{kucoin_symbol}"
        rate_limit_key = "ticker"

        # استفاده از کش کوتاه‌مدت
        response_data = await self._fetch_with_cache(
            'GET', endpoint, rate_limit_key, signed=False, cache_ttl=5  # 5 ثانیه
        )

        if response_data and isinstance(response_data, dict):
            mark_price_str = response_data.get('markPrice')  # قیمت مارک
            last_price_str = response_data.get('lastTradePrice')  # قیمت آخرین معامله
            price_str = mark_price_str or last_price_str  # اولویت با مارک

            if price_str:
                try:
                    price = float(price_str)
                    # ذخیره در کش
                    self._price_cache[symbol] = PriceData(
                        symbol=symbol,
                        price=price,
                        timestamp=time.time(),
                        source="api"
                    )

                    if self.config.debug_mode:
                        logger.debug(f"قیمت {symbol}: {price}")

                    return price
                except (ValueError, TypeError):
                    logger.error(f"خطا در تبدیل قیمت برای {symbol}: '{price_str}'")

        logger.error(f"دریافت قیمت برای {symbol} ناموفق بود.")
        return None

    async def get_ticker(self, symbol: str) -> Optional[Dict[str, Any]]:
        """
        دریافت اطلاعات کامل ticker برای یک نماد

        Args:
            symbol: نماد ارز (مثلا 'BTC/USDT')

        Returns:
            دیکشنری با اطلاعات ticker یا None در صورت خطا
        """
        kucoin_symbol = self._get_kucoin_symbol(symbol)
        endpoint = f"/api/v1/ticker"
        params = {'symbol': kucoin_symbol}
        rate_limit_key = "ticker"

        # استفاده از کش کوتاه‌مدت
        response_data = await self._fetch_with_cache(
            'GET', endpoint, rate_limit_key, params=params, signed=False, cache_ttl=5  # 5 ثانیه
        )

        if response_data and isinstance(response_data, dict):
            # به‌روزرسانی کش قیمت
            try:
                price = float(response_data.get('price', 0))
                self._price_cache[symbol] = PriceData(
                    symbol=symbol,
                    price=price,
                    timestamp=time.time(),
                    bid=float(response_data.get('bestBidPrice', 0)) if 'bestBidPrice' in response_data else None,
                    ask=float(response_data.get('bestAskPrice', 0)) if 'bestAskPrice' in response_data else None,
                    volume_24h=float(response_data.get('volume24h', 0)) if 'volume24h' in response_data else None,
                    change_24h=float(
                        response_data.get('priceChangePercent', 0)) if 'priceChangePercent' in response_data else None,
                    source="api"
                )
            except (ValueError, TypeError, KeyError) as e:
                logger.error(f"خطا در پردازش داده‌های ticker برای {symbol}: {e}")

            return response_data
        else:
            logger.error(f"دریافت ticker برای {symbol} ناموفق بود")
            return None

    async def get_funding_rate(self, symbol: str) -> Optional[float]:
        """
        دریافت نرخ فاندینگ برای قرارداد

        Args:
            symbol: نماد ارز (مثلا 'BTC/USDT')

        Returns:
            نرخ فاندینگ یا None در صورت خطا
        """
        # اگر در کش موجود است، از آن استفاده کن
        cache_key = f"funding_rate:{symbol}"
        cached_rate = await self._get_from_memory_cache(cache_key, 300)  # 5 دقیقه
        if cached_rate is not None:
            return cached_rate

        # در غیر این صورت، از API دریافت کن
        contract_details = await self.get_contract_details(symbol)

        if contract_details and 'fundingRate' in contract_details:
            try:
                funding_rate = float(contract_details['fundingRate'])
                # ذخیره در کش
                await self._set_in_memory_cache(cache_key, funding_rate, 300)
                return funding_rate
            except (ValueError, TypeError):
                logger.error(f"خطا در تبدیل نرخ فاندینگ برای {symbol}")

        logger.error(f"دریافت نرخ فاندینگ برای {symbol} ناموفق بود")
        return None

    async def get_funding_history(
            self,
            symbol: str,
            start_time_s: Optional[int] = None,
            end_time_s: Optional[int] = None,
            limit: int = 50
    ) -> Optional[List[Dict[str, Any]]]:
        """
        دریافت تاریخچه فاندینگ

        Args:
            symbol: نماد ارز (مثلا 'BTC/USDT')
            start_time_s: زمان شروع به ثانیه (اختیاری)
            end_time_s: زمان پایان به ثانیه (اختیاری)
            limit: حداکثر تعداد نتایج

        Returns:
            لیست دیکشنری‌های تاریخچه فاندینگ یا None در صورت خطا
        """
        kucoin_symbol = self._get_kucoin_symbol(symbol)
        endpoint = "/api/v1/funding-history"
        rate_limit_key = "public"

        params = {'symbol': kucoin_symbol}
        if start_time_s:
            params['startAt'] = int(start_time_s * 1000)
        if end_time_s:
            params['endAt'] = int(end_time_s * 1000)
        if limit:
            params['pageSize'] = limit

        # تعیین TTL کش بر اساس بازه زمانی
        cache_ttl = None
        if start_time_s and end_time_s:
            # اگر بازه زمانی در گذشته است، از کش استفاده کن
            current_time_s = time.time()
            if end_time_s < current_time_s - 3600:  # بیش از 1 ساعت قبل
                cache_ttl = 3600  # 1 ساعت

        # استفاده از کش
        response_data = await self._fetch_with_cache(
            'GET', endpoint, rate_limit_key, params=params, signed=False, cache_ttl=cache_ttl
        )

        if response_data and isinstance(response_data, dict) and 'dataList' in response_data:
            return response_data['dataList']
        else:
            logger.error(f"دریافت تاریخچه فاندینگ برای {symbol} ناموفق بود")
            return None

    async def get_order_book(self, symbol: str, limit: int = 20) -> Optional[Dict[str, Any]]:
        """
        دریافت دفتر سفارشات

        Args:
            symbol: نماد ارز (مثلا 'BTC/USDT')
            limit: حداکثر تعداد سفارشات

        Returns:
            دیکشنری با اطلاعات دفتر سفارشات یا None در صورت خطا
        """
        kucoin_symbol = self._get_kucoin_symbol(symbol)
        endpoint = "/api/v1/level2/depth"
        params = {'symbol': kucoin_symbol}
        rate_limit_key = "public"

        if limit:
            params['limit'] = limit

        # استفاده از کش کوتاه‌مدت
        response_data = await self._fetch_with_cache(
            'GET', endpoint, rate_limit_key, params=params, signed=False, cache_ttl=3  # 3 ثانیه
        )

        if response_data and isinstance(response_data, dict):
            return response_data
        else:
            logger.error(f"دریافت دفتر سفارشات برای {symbol} ناموفق بود")
            return None

    async def get_trade_history(
            self,
            symbol: str,
            limit: int = 50
    ) -> Optional[List[Dict[str, Any]]]:
        """
        دریافت تاریخچه معاملات اخیر

        Args:
            symbol: نماد ارز (مثلا 'BTC/USDT')
            limit: حداکثر تعداد معاملات

        Returns:
            لیست دیکشنری‌های معاملات یا None در صورت خطا
        """
        kucoin_symbol = self._get_kucoin_symbol(symbol)
        endpoint = "/api/v1/trade/history"
        params = {'symbol': kucoin_symbol}
        rate_limit_key = "public"

        if limit:
            params['pageSize'] = limit

        # استفاده از کش کوتاه‌مدت
        response_data = await self._fetch_with_cache(
            'GET', endpoint, rate_limit_key, params=params, signed=False, cache_ttl=5  # 5 ثانیه
        )

        if response_data and isinstance(response_data, list):
            return response_data
        else:
            logger.error(f"دریافت تاریخچه معاملات برای {symbol} ناموفق بود")
            return None

    async def place_order(
            self,
            symbol: str,
            side: str,
            order_type: str,
            leverage: int,
            size: float,
            price: Optional[float] = None,
            stop_price: Optional[float] = None,
            client_order_id: Optional[str] = None
    ) -> Optional[Dict[str, Any]]:
        """
        ثبت سفارش جدید

        Args:
            symbol: نماد ارز (مثلا 'BTC/USDT')
            side: جهت سفارش ('buy' یا 'sell')
            order_type: نوع سفارش ('limit', 'market', 'stop_limit', 'stop_market')
            leverage: اهرم (مثلا 10)
            size: اندازه سفارش (تعداد قرارداد)
            price: قیمت سفارش (برای سفارشات limit)
            stop_price: قیمت توقف (برای سفارشات stop)
            client_order_id: شناسه سفارش کلاینت (اختیاری)

        Returns:
            دیکشنری با اطلاعات سفارش ثبت شده یا None در صورت خطا
        """
        # تبدیل نماد به فرمت صرافی
        kucoin_symbol = self._get_kucoin_symbol(symbol)
        endpoint = "/api/v1/orders"
        rate_limit_key = "orders"

        # ساخت پارامترهای درخواست
        params = {
            'symbol': kucoin_symbol,
            'side': side.lower(),
            'leverage': leverage,
            'size': size
        }

        # تعیین نوع سفارش و پارامترهای مربوطه
        if order_type == 'limit':
            params['price'] = price
            params['type'] = 'limit'
        elif order_type == 'market':
            params['type'] = 'market'
        elif order_type == 'stop_limit':
            params['price'] = price
            params['stopPrice'] = stop_price
            params['stop'] = 'down' if side.lower() == 'sell' else 'up'
            params['type'] = 'limit'
        elif order_type == 'stop_market':
            params['stopPrice'] = stop_price
            params['stop'] = 'down' if side.lower() == 'sell' else 'up'
            params['type'] = 'market'
        else:
            logger.error(f"نوع سفارش نامعتبر: {order_type}")
            return None

        # اضافه کردن شناسه سفارش کلاینت اگر داده شده
        if client_order_id:
            params['clientOid'] = client_order_id

        # ارسال درخواست
        logger.info(
            f"ثبت سفارش {order_type} {side} برای {symbol}: اندازه={size}، قیمت={price}، قیمت توقف={stop_price}")

        # سفارش‌ها نباید کش شوند
        response_data = await self._make_request('POST', endpoint, params, signed=True)

        if response_data and isinstance(response_data, dict) and 'orderId' in response_data:
            logger.info(f"سفارش با موفقیت برای {symbol} ثبت شد: {response_data['orderId']}")
            return response_data
        else:
            error_msg = "ثبت سفارش ناموفق بود"
            if isinstance(response_data, dict) and response_data.get('error'):
                error_msg += f": {response_data.get('error_msg')}"
            logger.error(f"{error_msg} برای {symbol}")
            return None

    async def cancel_order(self, order_id: str, symbol: Optional[str] = None) -> bool:
        """
        لغو سفارش

        Args:
            order_id: شناسه سفارش
            symbol: نماد ارز (اختیاری)

        Returns:
            موفقیت عملیات
        """
        endpoint = f"/api/v1/orders/{order_id}"
        rate_limit_key = "orders"

        logger.info(f"لغو سفارش: {order_id}")
        response_data = await self._make_request('DELETE', endpoint, {}, signed=True)

        if response_data:
            logger.info(f"سفارش با موفقیت لغو شد: {order_id}")
            return True
        else:
            logger.error(f"لغو سفارش ناموفق بود: {order_id}")
            return False

    async def get_order_status(self, order_id: str) -> Optional[Dict[str, Any]]:
        """
        دریافت وضعیت سفارش

        Args:
            order_id: شناسه سفارش

        Returns:
            دیکشنری با اطلاعات سفارش یا None در صورت خطا
        """
        endpoint = f"/api/v1/orders/{order_id}"
        rate_limit_key = "orders"

        logger.debug(f"دریافت وضعیت سفارش: {order_id}")
        response_data = await self._make_request('GET', endpoint, {}, signed=True)

        if response_data and isinstance(response_data, dict):
            return response_data
        else:
            logger.error(f"دریافت وضعیت سفارش ناموفق بود: {order_id}")
            return None

    async def get_open_orders(self, symbol: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        دریافت لیست سفارشات باز

        Args:
            symbol: نماد ارز (اختیاری، برای فیلتر کردن)

        Returns:
            لیست دیکشنری‌های سفارشات باز
        """
        endpoint = "/api/v1/orders"
        params = {'status': 'active'}
        rate_limit_key = "orders"

        if symbol:
            params['symbol'] = self._get_kucoin_symbol(symbol)

        logger.debug(f"دریافت سفارشات باز" + (f" برای {symbol}" if symbol else ""))
        response_data = await self._make_request('GET', endpoint, params, signed=True)

        if response_data and isinstance(response_data, dict) and 'items' in response_data:
            return response_data['items']
        else:
            logger.error(f"دریافت سفارشات باز ناموفق بود" + (f" برای {symbol}" if symbol else ""))
            return []

    async def get_account_overview(self) -> Optional[Dict[str, Any]]:
        """
        دریافت اطلاعات کلی حساب

        Returns:
            دیکشنری با اطلاعات حساب یا None در صورت خطا
        """
        endpoint = "/api/v1/account-overview"
        rate_limit_key = "account"

        logger.debug("دریافت اطلاعات کلی حساب")
        response_data = await self._make_request('GET', endpoint, {}, signed=True)

        if response_data and isinstance(response_data, dict):
            return response_data
        else:
            logger.error("دریافت اطلاعات کلی حساب ناموفق بود")
            return None

    async def get_positions(self, symbol: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        دریافت لیست پوزیشن‌های باز

        Args:
            symbol: نماد ارز (اختیاری، برای فیلتر کردن)

        Returns:
            لیست دیکشنری‌های پوزیشن‌های باز
        """
        endpoint = "/api/v1/positions"
        params = {}
        rate_limit_key = "account"

        if symbol:
            params['symbol'] = self._get_kucoin_symbol(symbol)

        logger.debug(f"دریافت پوزیشن‌ها" + (f" برای {symbol}" if symbol else ""))
        response_data = await self._make_request('GET', endpoint, params, signed=True)

        if response_data and isinstance(response_data, list):
            return response_data
        else:
            logger.error(f"دریافت پوزیشن‌ها ناموفق بود" + (f" برای {symbol}" if symbol else ""))
            return []

    async def _init_websocket(self, token: str = None) -> bool:
        """
        راه‌اندازی اتصال وب‌سوکت.

        Args:
            token: توکن وب‌سوکت (اختیاری)

        Returns:
            موفقیت اتصال
        """
        if not self.config.websocket_enabled:
            logger.warning("وب‌سوکت در تنظیمات غیرفعال است.")
            return False

        async with self._ws_lock:
            # اگر اتصال فعلی باز است، آن را ببند
            if self._ws_connection and not self._ws_connection.closed:
                await self._ws_connection.close()
                self._ws_connection = None

            try:
                # دریافت توکن اگر ارائه نشده باشد
                if not token:
                    token_info = await self._get_ws_token()
                    if not token_info:
                        logger.error("دریافت توکن وب‌سوکت ناموفق بود. امکان برقراری اتصال وب‌سوکت وجود ندارد.")
                        return False

                    token = token_info.get('token')
                    servers = token_info.get('instanceServers', [])
                    if not token or not servers:
                        logger.error("اطلاعات توکن وب‌سوکت نامعتبر است.")
                        return False

                    server = servers[0]
                    ws_url = f"{server.get('endpoint')}?token={token}"
                    self._ws_token_expiry = time.time() + (
                            server.get('pingInterval', 30000) * 6 / 1000)  # 6 ping intervals
                else:
                    # استفاده از توکن ارائه شده
                    response = await self._get_ws_token()
                    if not response:
                        logger.error("دریافت اطلاعات سرور وب‌سوکت ناموفق بود.")
                        return False

                    servers = response.get('instanceServers', [])
                    if not servers:
                        logger.error("هیچ سرور وب‌سوکتی در دسترس نیست.")
                        return False

                    server = servers[0]
                    ws_url = f"{server.get('endpoint')}?token={token}"
                    self._ws_token_expiry = time.time() + (
                            server.get('pingInterval', 30000) * 6 / 1000)  # 6 ping intervals

                # ایجاد اتصال وب‌سوکت
                self._ws_connection = await websockets.connect(
                    ws_url,
                    ping_interval=None,  # ما خودمان ping می‌فرستیم
                    ping_timeout=None,
                    close_timeout=10,
                    max_size=10 * 1024 * 1024,  # 10 MB
                    max_queue=1024
                )

                # ارسال پیام welcome
                welcome_message = await self._ws_connection.recv()
                welcome_data = json.loads(welcome_message)
                if welcome_data.get('type') != 'welcome':
                    logger.error(f"انتظار پیام welcome داشتیم، اما دریافت کردیم: {welcome_data}")
                    await self._ws_connection.close()
                    self._ws_connection = None
                    return False

                logger.info("اتصال وب‌سوکت با موفقیت برقرار شد.")
                self._ws_last_pong_time = time.time()
                self._ws_running = True
                self._client_health_status['ws_status'] = 'connected'

                # شروع تسک های ping و گوش دادن
                asyncio.create_task(self._ws_ping_loop())
                asyncio.create_task(self._ws_listen_loop())

                return True

            except Exception as e:
                logger.error(f"خطا در برقراری اتصال وب‌سوکت: {e}", exc_info=True)
                if self._ws_connection and not self._ws_connection.closed:
                    await self._ws_connection.close()
                self._ws_connection = None
                self._client_health_status['ws_status'] = 'error'
                self._client_health_status['last_error'] = {
                    'time': time.time(),
                    'message': f"خطا در برقراری اتصال وب‌سوکت: {str(e)}"
                }
                return False

    async def _get_ws_token(self) -> Optional[Dict[str, Any]]:
        """
        دریافت توکن وب‌سوکت از API.

        Returns:
            اطلاعات توکن یا None در صورت خطا
        """
        endpoint = "/api/v1/bullet-public"
        rate_limit_key = "public"

        # استفاده از کش 50 دقیقه‌ای (توکن 60 دقیقه اعتبار دارد)
        response_data = await self._fetch_with_cache(
            'POST', endpoint, rate_limit_key, signed=False, cache_ttl=3000
        )

        if response_data and isinstance(response_data, dict) and response_data.get('token'):
            return response_data
        else:
            logger.error("دریافت توکن وب‌سوکت ناموفق بود.")
            return None

    async def _ws_ping_loop(self):
        """لوپ ارسال پینگ‌های دوره‌ای به سرور وب‌سوکت."""
        logger.debug("شروع حلقه پینگ وب‌سوکت.")
        ping_counter = 0

        while self._ws_running and self._ws_connection and not self._ws_connection.closed:
            try:
                # ارسال پینگ
                ping_counter += 1
                ping_id = f"ping-{int(time.time() * 1000)}-{ping_counter}"
                ping_msg = {
                    "id": ping_id,
                    "type": "ping"
                }

                await self._ws_connection.send(json.dumps(ping_msg))
                logger.debug(f"پینگ وب‌سوکت ارسال شد: {ping_id}")

                # انتظار برای بررسی بعدی
                await asyncio.sleep(self.config.websocket_ping_interval)

                # بررسی اعتبار توکن
                if time.time() > self._ws_token_expiry - 300:  # 5 دقیقه قبل از انقضا
                    logger.info("توکن وب‌سوکت نزدیک به انقضا است. شروع نوسازی توکن.")
                    await self._refresh_ws_connection()
                    return  # خروج از لوپ فعلی

            except websockets.exceptions.ConnectionClosed:
                logger.warning("اتصال وب‌سوکت در حلقه پینگ بسته شد.")
                break
            except Exception as e:
                logger.error(f"خطا در حلقه پینگ وب‌سوکت: {e}", exc_info=True)
                await asyncio.sleep(5)  # تاخیر کوتاه قبل از تلاش دوباره

        logger.info("حلقه پینگ وب‌سوکت پایان یافت.")

        # اگر به اینجا برسیم و هنوز _ws_running فعال است، راه‌اندازی مجدد اتصال
        if self._ws_running:
            asyncio.create_task(self._reconnect_ws())

    async def _ws_listen_loop(self):
        """لوپ گوش دادن به پیام‌های وب‌سوکت."""
        logger.debug("شروع حلقه گوش دادن وب‌سوکت.")

        while self._ws_running and self._ws_connection and not self._ws_connection.closed:
            try:
                # دریافت پیام
                message = await self._ws_connection.recv()

                # پردازش پیام
                await self._process_ws_message(message)

            except websockets.exceptions.ConnectionClosed:
                logger.warning("اتصال وب‌سوکت در حلقه گوش دادن بسته شد.")
                break
            except asyncio.TimeoutError:
                logger.warning("تایم‌اوت در انتظار برای پیام وب‌سوکت.")
                continue
            except Exception as e:
                logger.error(f"خطا در حلقه گوش دادن وب‌سوکت: {e}", exc_info=True)
                await asyncio.sleep(1)  # تاخیر کوتاه قبل از تلاش دوباره

        logger.info("حلقه گوش دادن وب‌سوکت پایان یافت.")

        # اگر به اینجا برسیم و هنوز _ws_running فعال است، راه‌اندازی مجدد اتصال
        if self._ws_running:
            asyncio.create_task(self._reconnect_ws())

    async def _process_ws_message(self, message: str):
        """پردازش پیام دریافتی از وب‌سوکت."""
        try:
            data = json.loads(message)
            msg_type = data.get('type')

            # به‌روزرسانی زمان آخرین پاسخ
            self._ws_last_pong_time = time.time()

            if msg_type == 'welcome':
                logger.info(f"پیام خوش‌آمدگویی وب‌سوکت دریافت شد: {data.get('id')}")
            elif msg_type == 'pong':
                logger.debug(f"پاسخ پینگ وب‌سوکت دریافت شد: {data.get('id')}")
            elif msg_type == 'ack':
                # تایید اشتراک یا لغو اشتراک
                sub_id = data.get('id')
                if sub_id and ':' in sub_id:
                    action, topic = sub_id.split(':', 1)
                    if action == 'sub':
                        self._ws_subscriptions.add(topic)
                        logger.info(f"اشتراک برای {topic} تایید شد")
            elif msg_type == 'message':
                # پیام داده - بر اساس موضوع پردازش کن
                await self._handle_ws_data_message(data)
            elif msg_type == 'error':
                # پیام خطا
                error_code = data.get('code')
                error_msg = data.get('data')
                logger.error(f"خطای وب‌سوکت: کد={error_code}، پیام={error_msg}")

                # برخی خطاها نیاز به اتصال مجدد دارند
                if error_code in ['400', '401', '403', '429']:
                    logger.warning("اتصال مجدد وب‌سوکت به دلیل خطا...")
                    await self._reconnect_ws()
            else:
                logger.warning(f"نوع پیام وب‌سوکت ناشناخته: {msg_type}")

        except json.JSONDecodeError:
            logger.warning(f"JSON نامعتبر از وب‌سوکت دریافت شد: {message[:100]}...")
        except Exception as e:
            logger.error(f"خطا در پردازش پیام وب‌سوکت: {e}", exc_info=True)

    async def _handle_ws_data_message(self, data: Dict[str, Any]):
        """پردازش پیام‌های داده وب‌سوکت."""
        try:
            # بررسی موضوع و داده‌ها
            topic = data.get('topic')
            if not topic:
                return

            message_data = data.get('data')
            if not message_data:
                return

            # پردازش انواع مختلف پیام
            if 'ticker' in topic:
                # پیام ticker - به‌روزرسانی قیمت
                await self._process_ticker_message(topic, message_data)
            elif 'candle' in topic:
                # پیام کندل
                await self._process_candle_message(topic, message_data)
            elif 'level2' in topic:
                # پیام order book
                await self._process_orderbook_message(topic, message_data)
            elif 'match' in topic:
                # پیام معاملات
                await self._process_trade_message(topic, message_data)

            # اجرای handler اختصاصی اگر ثبت شده باشد
            if topic in self._ws_message_handlers:
                await self._ws_message_handlers[topic](message_data)

        except Exception as e:
            logger.error(f"خطا در هندل کردن پیام داده وب‌سوکت: {e}", exc_info=True)

    async def _process_ticker_message(self, topic: str, data: Dict[str, Any]):
        """پردازش پیام ticker از وب‌سوکت."""
        try:
            # استخراج نماد از موضوع
            parts = topic.split(':')
            if len(parts) < 2:
                return

            symbol_raw = parts[1]
            # تبدیل به فرمت استاندارد (مثلاً XBTUSDTM -> BTC/USDT)
            symbol = self._convert_to_standard_symbol(symbol_raw)
            if not symbol:
                return

            # استخراج قیمت
            price = float(data.get('price', 0))
            if price <= 0:
                return

            # ذخیره در کش
            self._ws_prices[symbol] = PriceData(
                symbol=symbol,
                price=price,
                timestamp=time.time(),
                bid=float(data.get('bestBidPrice', 0)) if 'bestBidPrice' in data else None,
                ask=float(data.get('bestAskPrice', 0)) if 'bestAskPrice' in data else None,
                volume_24h=float(data.get('volume24h', 0)) if 'volume24h' in data else None,
                change_24h=float(data.get('priceChangePercent', 0)) if 'priceChangePercent' in data else None,
                source="websocket"
            )

            logger.debug(f"قیمت {symbol} از طریق وب‌سوکت به‌روزرسانی شد: {price}")

        except Exception as e:
            logger.error(f"خطا در پردازش پیام ticker: {e}", exc_info=True)

    def _convert_to_standard_symbol(self, kucoin_symbol: str) -> Optional[str]:
        """تبدیل نماد در فرمت صرافی به فرمت استاندارد (BTC/USDT)."""
        try:
            # بررسی کش
            for std_symbol, info in self._symbol_info_cache.items():
                if info.get('kucoin_symbol') == kucoin_symbol:
                    return std_symbol

            # تبدیل دستی
            if kucoin_symbol.endswith('USDTM'):
                base = kucoin_symbol[:-5]
                # تبدیل XBT به BTC
                if base == "XBT":
                    base = "BTC"
                return f"{base}/USDT"

            return kucoin_symbol

        except Exception as e:
            logger.error(f"خطا در تبدیل نماد {kucoin_symbol}: {e}")
            return None

    async def _process_candle_message(self, topic: str, data: Dict[str, Any]):
        """پردازش پیام کندل از وب‌سوکت."""
        # پیاده‌سازی پردازش پیام کندل
        pass

    async def _process_orderbook_message(self, topic: str, data: Dict[str, Any]):
        """پردازش پیام order book از وب‌سوکت."""
        # پیاده‌سازی پردازش order book
        pass

    async def _process_trade_message(self, topic: str, data: Dict[str, Any]):
        """پردازش پیام معاملات از وب‌سوکت."""
        # پیاده‌سازی پردازش معاملات
        pass

    async def _reconnect_ws(self):
        """تلاش برای اتصال مجدد وب‌سوکت با استراتژی backoff."""
        if not self._ws_running:
            return

        # جلوگیری از چندین تلاش همزمان
        async with self._ws_lock:
            if self._ws_reconnect_task and not self._ws_reconnect_task.done():
                logger.debug("تسک اتصال مجدد در حال اجرا است.")
                return

            self._ws_reconnect_task = asyncio.current_task()

        try:
            # بستن اتصال قبلی اگر هنوز باز است
            if self._ws_connection and not self._ws_connection.closed:
                await self._ws_connection.close()
                self._ws_connection = None

            # تاخیر با استراتژی backoff
            attempt = 0
            max_attempts = 10

            while attempt < max_attempts and self._ws_running:
                # محاسبه تاخیر با jitter
                delay = min(60, self.config.websocket_reconnect_delay * (2 ** attempt))
                jitter = random.uniform(0, delay * 0.2)  # 0-20% تصادفی‌سازی
                total_delay = delay + jitter

                logger.info(
                    f"انتظار {total_delay:.2f} ثانیه قبل از تلاش {attempt + 1}/{max_attempts} برای اتصال مجدد وب‌سوکت")
                await asyncio.sleep(total_delay)

                # تلاش برای اتصال مجدد
                connected = await self._init_websocket()
                if connected:
                    logger.info("اتصال مجدد به وب‌سوکت با موفقیت انجام شد.")

                    # اشتراک مجدد در تاپیک‌ها
                    for topic in list(self._ws_subscriptions):
                        await self.subscribe_ws_topic(topic)

                    return

                attempt += 1

            # اگر به اینجا برسیم، همه تلاش‌ها ناموفق بوده‌اند
            if self._ws_running:
                logger.error("اتصال مجدد به وب‌سوکت پس از چندین تلاش ناموفق بود.")
                self._client_health_status['ws_status'] = 'error'

        except Exception as e:
            logger.error(f"خطا در اتصال مجدد وب‌سوکت: {e}", exc_info=True)
            self._client_health_status['ws_status'] = 'error'
        finally:
            self._ws_reconnect_task = None

    async def _refresh_ws_connection(self):
        """نوسازی اتصال وب‌سوکت (برای تمدید توکن)."""
        try:
            logger.info("نوسازی اتصال وب‌سوکت برای تمدید توکن.")

            # دریافت توکن جدید
            token_info = await self._get_ws_token()
            if not token_info:
                logger.error("دریافت توکن جدید وب‌سوکت ناموفق بود.")
                return

            # راه‌اندازی مجدد اتصال با توکن جدید
            await self._init_websocket(token_info.get('token'))

            # اشتراک مجدد در تاپیک‌ها
            for topic in list(self._ws_subscriptions):
                await self.subscribe_ws_topic(topic)

        except Exception as e:
            logger.error(f"خطا در نوسازی اتصال وب‌سوکت: {e}", exc_info=True)
            # تلاش برای اتصال مجدد در صورت خطا
            await self._reconnect_ws()

    async def subscribe_ws_topic(self, topic: str) -> bool:
        """اشتراک در یک تاپیک وب‌سوکت."""
        if not self._ws_running or not self._ws_connection or self._ws_connection.closed:
            logger.warning(f"امکان اشتراک در {topic} وجود ندارد: وب‌سوکت متصل نیست.")
            return False

        try:
            # ساخت پیام اشتراک
            subscription_id = f"sub:{topic}:{int(time.time() * 1000)}"
            subscribe_message = {
                "id": subscription_id,
                "type": "subscribe",
                "topic": topic,
                "privateChannel": False,
                "response": True
            }

            # ارسال درخواست اشتراک
            await self._ws_connection.send(json.dumps(subscribe_message))
            logger.info(f"درخواست اشتراک برای {topic} ارسال شد")

            return True

        except Exception as e:
            logger.error(f"خطا در اشتراک در {topic}: {e}")
            return False

    async def unsubscribe_ws_topic(self, topic: str) -> bool:
        """لغو اشتراک از یک تاپیک وب‌سوکت."""
        if not self._ws_running or not self._ws_connection or self._ws_connection.closed:
            return False

        try:
            # ساخت پیام لغو اشتراک
            unsubscription_id = f"unsub:{topic}:{int(time.time() * 1000)}"
            unsubscribe_message = {
                "id": unsubscription_id,
                "type": "unsubscribe",
                "topic": topic,
                "response": True
            }

            # ارسال درخواست لغو اشتراک
            await self._ws_connection.send(json.dumps(unsubscribe_message))
            logger.info(f"درخواست لغو اشتراک برای {topic} ارسال شد")

            # حذف از لیست اشتراک‌ها
            if topic in self._ws_subscriptions:
                self._ws_subscriptions.remove(topic)

            return True

        except Exception as e:
            logger.error(f"خطا در لغو اشتراک از {topic}: {e}")
            return False

    async def register_ws_handler(self, topic: str, handler: Callable[[Dict[str, Any]], Awaitable[None]]):
        """ثبت تابع handler برای یک تاپیک وب‌سوکت خاص."""
        self._ws_message_handlers[topic] = handler
        logger.debug(f"هندلر برای تاپیک وب‌سوکت ثبت شد: {topic}")

    async def start_websocket(self) -> bool:
        """راه‌اندازی اتصال وب‌سوکت و شروع گوش دادن به پیام‌ها."""
        if not self.config.websocket_enabled:
            logger.warning("وب‌سوکت در تنظیمات غیرفعال است.")
            self._client_health_status['ws_status'] = 'disabled'
            return False

        try:
            logger.info("شروع اتصال وب‌سوکت...")
            self._ws_running = True

            # راه‌اندازی اتصال
            connected = await self._init_websocket()
            if not connected:
                logger.error("راه‌اندازی اتصال وب‌سوکت ناموفق بود.")
                self._ws_running = False
                self._client_health_status['ws_status'] = 'error'
                return False

            logger.info("اتصال وب‌سوکت با موفقیت شروع شد.")
            self._client_health_status['ws_status'] = 'connected'
            return True

        except Exception as e:
            logger.error(f"خطا در شروع وب‌سوکت: {e}", exc_info=True)
            self._ws_running = False
            self._client_health_status['ws_status'] = 'error'
            return False

    async def stop_websocket(self):
        """توقف اتصال وب‌سوکت."""
        logger.info("توقف اتصال وب‌سوکت...")

        self._ws_running = False

        # بستن اتصال فعلی
        if self._ws_connection and not self._ws_connection.closed:
            try:
                await self._ws_connection.close()
            except Exception as e:
                logger.error(f"خطا در بستن اتصال وب‌سوکت: {e}")

        self._ws_connection = None
        self._ws_subscriptions.clear()
        self._client_health_status['ws_status'] = 'not_connected'
        logger.info("اتصال وب‌سوکت متوقف شد.")

    async def get_ws_ticker(self, symbol: str) -> bool:
        """اشتراک در تغییرات قیمت (ticker) برای یک نماد خاص."""
        kucoin_symbol = self._get_kucoin_symbol(symbol)
        topic = f"/contractMarket/ticker:{kucoin_symbol}"

        return await self.subscribe_ws_topic(topic)

    async def get_ws_candles(self, symbol: str, timeframe: str) -> bool:
        """اشتراک در تغییرات کندل برای یک نماد و تایم‌فریم خاص."""
        kucoin_symbol = self._get_kucoin_symbol(symbol)
        # تبدیل تایم‌فریم به فرمت مورد نیاز API
        ws_timeframe = None
        if timeframe == '1m':
            ws_timeframe = '1min'
        elif timeframe == '5m':
            ws_timeframe = '5min'
        elif timeframe == '15m':
            ws_timeframe = '15min'
        elif timeframe == '30m':
            ws_timeframe = '30min'
        elif timeframe == '1h':
            ws_timeframe = '1hour'
        elif timeframe == '4h':
            ws_timeframe = '4hour'
        elif timeframe == '1d':
            ws_timeframe = '1day'

        if not ws_timeframe:
            logger.error(f"تایم‌فریم غیرقابل پشتیبانی برای کندل‌های وب‌سوکت: {timeframe}")
            return False

        topic = f"/contractMarket/candle:{kucoin_symbol}_{ws_timeframe}"

        return await self.subscribe_ws_topic(topic)

    async def get_ws_orderbook(self, symbol: str) -> bool:
        """اشتراک در تغییرات order book برای یک نماد خاص."""
        kucoin_symbol = self._get_kucoin_symbol(symbol)
        topic = f"/contractMarket/level2:{kucoin_symbol}"

        return await self.subscribe_ws_topic(topic)

    async def get_ws_trades(self, symbol: str) -> bool:
        """اشتراک در معاملات جدید برای یک نماد خاص."""
        kucoin_symbol = self._get_kucoin_symbol(symbol)
        topic = f"/contractMarket/execution:{kucoin_symbol}"

        return await self.subscribe_ws_topic(topic)

    def get_stats(self) -> Dict[str, Any]:
        """دریافت آمار عملکرد کلاینت."""
        stats = {
            "api_stats": dict(self._api_stats),
            "ws_stats": {
                "running": self._ws_running,
                "connected": bool(self._ws_connection and not self._ws_connection.closed),
                "subscriptions": list(self._ws_subscriptions),
                "last_pong_time": self._ws_last_pong_time,
                "prices_cached": len(self._ws_prices)
            },
            "cache_stats": {
                "memory_cache_size": len(self._memory_cache),
                "price_cache_size": len(self._price_cache),
                "symbol_cache_size": len(self._symbol_info_cache)
            },
            "rate_limit_stats": {
                "violations": dict(self._rate_limit_violations),
                "min_intervals": {k: v for k, v in self._min_request_interval.items() if
                                  v > self._rate_limit_time_window}
            },
            "health_status": self._client_health_status
        }

        # محاسبه میانگین زمان پاسخ
        if self._api_stats["response_times"]:
            stats["api_stats"]["avg_response_time_ms"] = sum(self._api_stats["response_times"]) / len(
                self._api_stats["response_times"])

        # حذف داده‌های خام برای کاهش حجم
        stats["api_stats"].pop("response_times", None)

        return stats

    async def close(self) -> None:
        """پایان دادن به کارکرد ExchangeClient و آزادسازی منابع."""
        logger.info("در حال خاموش کردن ExchangeClient...")

        # توقف وب‌سوکت
        if self._ws_running:
            await self.stop_websocket()

        # لغو تسک‌های نگهداری
        if self._cache_cleanup_task and not self._cache_cleanup_task.done():
            self._cache_cleanup_task.cancel()
            try:
                await self._cache_cleanup_task
            except asyncio.CancelledError:
                pass

        if self._health_check_task and not self._health_check_task.done():
            self._health_check_task.cancel()
            try:
                await self._health_check_task
            except asyncio.CancelledError:
                pass

        # بستن HTTP session
        if self._session and not self._session.closed:
            await self._session.close()
            logger.info("نشست HTTP بسته شد.")
            self._session = None

        # بستن اتصال Redis
        if self._redis_client:
            await self._redis_client.close()
            logger.info("اتصال Redis بسته شد.")
            self._redis_client = None

        # تعطیلی thread pool
        self._thread_executor.shutdown(wait=True)

        logger.info("خاموش کردن ExchangeClient کامل شد.")