"""
ماژول trade_extensions.py: توابع و کلاس‌های کمکی برای گسترش قابلیت‌های مدیریت معاملات
این ماژول شامل کلاس‌های کمکی برای پشتیبانی از قابلیت‌های پیشرفته مدیریت معاملات است.
"""

import logging
import threading
import functools
from typing import Dict, List, Tuple, Any, Optional, Union, Callable
from datetime import datetime, timedelta
import numpy as np
import pandas as pd
import json
from multi_tp_trade import Trade
import os
import sqlite3
from signal_generator import SignalInfo
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
import warnings

# تنظیم لاگر
logger = logging.getLogger(__name__)

# تعریف دکوراتورها و توابع کمکی
def cache_result(timeout_seconds: int = 3600):
    """
    دکوراتور برای کش کردن نتایج توابع با زمان انقضا

    Args:
        timeout_seconds: مدت زمان اعتبار کش به ثانیه
    """
    def decorator(func):
        cache = {}

        @functools.wraps(func)
        async def wrapper(*args, **kwargs):
            # ساخت کلید منحصر به فرد برای کش
            key = f"{func.__name__}:{str(args)}:{str(kwargs)}"
            current_time = datetime.now()

            # بررسی کش و زمان انقضا
            if key in cache:
                result, timestamp = cache[key]
                if (current_time - timestamp).total_seconds() < timeout_seconds:
                    return result

            # محاسبه نتیجه جدید
            result = await func(*args, **kwargs)
            cache[key] = (result, current_time)

            return result
        return wrapper
    return decorator

def safe_execution(default_value: Any = None, log_error: bool = True):
    """
    دکوراتور برای اجرای ایمن توابع با مدیریت خطا

    Args:
        default_value: مقدار پیش‌فرض در صورت بروز خطا
        log_error: آیا خطا در لاگ ثبت شود
    """
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            try:
                return func(*args, **kwargs)
            except Exception as e:
                if log_error:
                    logger.error(f"خطا در {func.__name__}: {e}")
                return default_value
        return wrapper
    return decorator

@dataclass
class MarketMetrics:
    """کلاس نگهداری معیارهای بازار برای استفاده در تصمیم‌گیری"""
    volatility: float = 0.0  # نوسان‌پذیری بازار
    trend_strength: float = 0.0  # قدرت روند
    volume_profile: float = 0.0  # پروفایل حجم
    market_condition: str = "neutral"  # وضعیت بازار (صعودی، نزولی، خنثی)


async def analyze_btc_trend_strength(data_fetcher, config: Dict[str, Any] = None) -> Dict[str, Any]:
    """
    تحلیل قدرت روند بیت‌کوین با استفاده از شاخص‌های متعدد

    Args:
        data_fetcher: وهله DataFetcher برای دریافت داده‌ها
        config: تنظیمات اختیاری

    Returns:
        دیکشنری از اطلاعات روند
    """
    # تنظیمات پیش‌فرض
    if not config:
        config = {
            'btc_symbol': 'BTCUSDT',
            'timeframes': ['1h', '4h', '1d'],
            'ema_periods': [20, 50, 100],
            'rsi_period': 14,
            'volume_period': 20,
            'adx_period': 14
        }

    results = {}

    try:
        for tf in config['timeframes']:
            # دریافت داده‌های قیمت برای این تایم‌فریم
            df = await data_fetcher.get_historical_data(
                config['btc_symbol'],
                tf,
                limit=max(200, config['ema_periods'][-1] * 2)  # داده کافی برای محاسبات
            )

            if df is None or df.empty:
                results[tf] = {'error': 'No data available'}
                continue

            # 1. محاسبه EMAs
            for period in config['ema_periods']:
                df[f'ema_{period}'] = df['close'].ewm(span=period, adjust=False).mean()

            # 2. محاسبه RSI
            delta = df['close'].diff()
            gain = delta.mask(delta < 0, 0)
            loss = -delta.mask(delta > 0, 0)
            avg_gain = gain.rolling(window=config['rsi_period']).mean()
            avg_loss = loss.rolling(window=config['rsi_period']).mean()
            rs = avg_gain / avg_loss
            df['rsi'] = 100 - (100 / (1 + rs))

            # 3. محاسبه حجم نرمال‌شده
            if 'volume' in df.columns:
                df['volume_sma'] = df['volume'].rolling(window=config['volume_period']).mean()
                df['volume_ratio'] = df['volume'] / df['volume_sma']

            # 4. محاسبه ADX (شاخص قدرت روند)
            try:
                import talib
                df['adx'] = talib.ADX(df['high'].values, df['low'].values, df['close'].values,
                                      timeperiod=config['adx_period'])
            except ImportError:
                # محاسبه ساده شیب قیمت به عنوان جایگزین ADX
                df['price_slope'] = (df['close'] - df['close'].shift(5)) / df['close'].shift(5) * 100

            # 5. تحلیل روند
            # آخرین مقادیر (فعلی)
            latest = {}
            latest['close'] = float(df['close'].iloc[-1])

            # وضعیت EMA
            ema_status = []
            for period in config['ema_periods']:
                latest[f'ema_{period}'] = float(df[f'ema_{period}'].iloc[-1])
                if df['close'].iloc[-1] > df[f'ema_{period}'].iloc[-1]:
                    ema_status.append(1)  # بالای EMA
                else:
                    ema_status.append(-1)  # زیر EMA

            # قرارگیری EMAها نسبت به هم
            ema_alignment = 0
            if all(ema_status[i] >= ema_status[i + 1] for i in range(len(ema_status) - 1)):
                ema_alignment = sum(ema_status)  # همه صعودی
            elif all(ema_status[i] <= ema_status[i + 1] for i in range(len(ema_status) - 1)):
                ema_alignment = sum(ema_status)  # همه نزولی

            # RSI وضعیت
            latest['rsi'] = float(df['rsi'].iloc[-1])
            if latest['rsi'] > 70:
                rsi_status = "اشباع خرید"
            elif latest['rsi'] < 30:
                rsi_status = "اشباع فروش"
            else:
                rsi_status = "خنثی"

            # حجم وضعیت
            if 'volume_ratio' in df:
                latest['volume_ratio'] = float(df['volume_ratio'].iloc[-1])
                volume_status = "بالا" if latest['volume_ratio'] > 1.5 else "عادی"
            else:
                volume_status = "نامشخص"

            # ADX وضعیت
            if 'adx' in df:
                latest['adx'] = float(df['adx'].iloc[-1])
                if latest['adx'] > 25:
                    trend_strength = "قوی"
                elif latest['adx'] > 20:
                    trend_strength = "متوسط"
                else:
                    trend_strength = "ضعیف"
            elif 'price_slope' in df:
                latest['price_slope'] = float(df['price_slope'].iloc[-1])
                if abs(latest['price_slope']) > 5:
                    trend_strength = "قوی"
                elif abs(latest['price_slope']) > 2:
                    trend_strength = "متوسط"
                else:
                    trend_strength = "ضعیف"
            else:
                trend_strength = "نامشخص"

            # تعیین روند براساس ترکیب شاخص‌ها
            if sum(ema_status) > 0 and latest['rsi'] > 50:
                trend = "صعودی"
                if latest['rsi'] > 70:
                    trend += " (احتمال اصلاح)"
            elif sum(ema_status) < 0 and latest['rsi'] < 50:
                trend = "نزولی"
                if latest['rsi'] < 30:
                    trend += " (احتمال بازگشت)"
            else:
                trend = "خنثی"

            # محاسبه قدرت روند (0-100)
            trend_factors = []

            # فاکتور EMA
            ema_factor = abs(sum(ema_status)) / len(ema_status) * 100 / 3
            trend_factors.append(ema_factor)

            # فاکتور RSI
            rsi_factor = abs(latest['rsi'] - 50) * 2
            trend_factors.append(rsi_factor)

            # فاکتور ADX
            if 'adx' in latest:
                adx_factor = min(100, latest['adx'] * 2)
                trend_factors.append(adx_factor)
            elif 'price_slope' in latest:
                slope_factor = min(100, abs(latest['price_slope']) * 5)
                trend_factors.append(slope_factor)

            # فاکتور حجم
            if 'volume_ratio' in latest:
                volume_factor = min(100, latest['volume_ratio'] * 25)
                trend_factors.append(volume_factor)

            # قدرت نهایی روند
            trend_strength_value = sum(trend_factors) / len(trend_factors)

            # ذخیره نتایج در تایم‌فریم فعلی
            results[tf] = {
                'trend': trend,
                'trend_strength': trend_strength_value,
                'trend_strength_category': trend_strength,
                'ema_status': ema_status,
                'ema_alignment': ema_alignment,
                'rsi': latest['rsi'],
                'rsi_status': rsi_status,
                'latest_values': latest,
                'volume_status': volume_status
            }

        # خلاصه کلی (میانگین وزن‌دار تایم‌فریم‌ها)
        # وزن بیشتر به تایم‌فریم‌های بزرگتر
        weights = {
            '1m': 0.1, '5m': 0.15, '15m': 0.2, '30m': 0.25,
            '1h': 0.3, '2h': 0.35, '4h': 0.4, '6h': 0.45,
            '8h': 0.5, '12h': 0.6, '1d': 0.7, '3d': 0.8, '1w': 0.9
        }

        # محاسبه خلاصه
        trend_strengths = []
        trend_votes = {'صعودی': 0, 'نزولی': 0, 'خنثی': 0}

        for tf, data in results.items():
            if 'error' in data:
                continue

            tf_weight = weights.get(tf, 0.3)
            trend_strengths.append(data['trend_strength'] * tf_weight)

            base_trend = data['trend'].split(' ')[0]  # حذف پرانتز توضیحات
            trend_votes[base_trend] = trend_votes.get(base_trend, 0) + tf_weight

        if trend_strengths:
            avg_strength = sum(trend_strengths) / sum(
                weights.get(tf, 0.3) for tf in results if 'error' not in results[tf])
        else:
            avg_strength = 0

        # تعیین روند غالب
        dominant_trend = max(trend_votes.items(), key=lambda x: x[1])[0]

        # نتیجه نهایی
        results['summary'] = {
            'dominant_trend': dominant_trend,
            'trend_strength': avg_strength,
            'trend_votes': trend_votes,
            'timeframes_analyzed': list(results.keys())
        }

        return results

    except Exception as e:
        logger.error(f"خطا در تحلیل قدرت روند بیت‌کوین: {e}")
        return {'error': str(e)}


async def determine_market_phase(data_fetcher, config: Dict[str, Any] = None) -> Dict[str, Any]:
    """
    تشخیص فاز کلی بازار (صعودی، نزولی، رنج)

    Args:
        data_fetcher: وهله DataFetcher برای دریافت داده‌ها
        config: تنظیمات اختیاری

    Returns:
        دیکشنری از اطلاعات فاز بازار
    """
    # تنظیمات پیش‌فرض
    if not config:
        config = {
            'btc_symbol': 'BTCUSDT',
            'market_cap_symbol': 'TOTAL',  # نماد مارکت کپ کل در TradingView
            'timeframes': ['1d', '1w'],
            'ma_periods': [50, 200],
            'analysis_period': 90,  # تعداد روز برای تحلیل
            'volatility_period': 20
        }

    try:
        results = {}

        # تحلیل بیت‌کوین
        btc_phase = await _analyze_asset_phase(config['btc_symbol'], data_fetcher, config)
        results['btc_phase'] = btc_phase

        # تلاش برای تحلیل مارکت کپ کل (اگر داده در دسترس است)
        try:
            market_cap_phase = await _analyze_asset_phase(config['market_cap_symbol'], data_fetcher, config)
            results['market_cap_phase'] = market_cap_phase
        except:
            results['market_cap_phase'] = {'phase': 'unknown', 'reason': 'data_unavailable'}

        # تحلیل چند آلتکوین شاخص
        alt_symbols = ['ETHUSDT', 'BNBUSDT', 'SOLUSDT', 'ADAUSDT']
        alt_phases = {}

        for symbol in alt_symbols:
            try:
                alt_phases[symbol] = await _analyze_asset_phase(symbol, data_fetcher, config)
            except:
                alt_phases[symbol] = {'phase': 'unknown', 'reason': 'data_unavailable'}

        results['alt_phases'] = alt_phases

        # تعیین فاز غالب بازار
        phases_count = {'bull': 0, 'bear': 0, 'accumulation': 0, 'distribution': 0, 'sideways': 0}

        # وزن بیشتر برای بیت‌کوین و مارکت کپ کل
        if btc_phase['phase'] in phases_count:
            phases_count[btc_phase['phase']] += 3

        if 'market_cap_phase' in results and results['market_cap_phase']['phase'] in phases_count:
            phases_count[results['market_cap_phase']['phase']] += 2

        # وزن کمتر برای آلتکوین‌ها
        for symbol, phase_data in alt_phases.items():
            if phase_data['phase'] in phases_count:
                phases_count[phase_data['phase']] += 1

        # تعیین فاز غالب
        dominant_phase = max(phases_count.items(), key=lambda x: x[1])[0] if phases_count else 'unknown'

        # وضعیت نسبت به ATH
        btc_ath_status = btc_phase.get('ath_status', {})

        # محاسبه نسبت آلتکوین به بیت‌کوین
        altcoin_btc_ratio = await _calculate_altcoin_btc_ratio(data_fetcher)

        # محاسبه وضعیت بازار
        market_status = _determine_market_status(dominant_phase, btc_phase, altcoin_btc_ratio)

        # ساخت خلاصه نهایی
        results['summary'] = {
            'dominant_phase': dominant_phase,
            'market_status': market_status,
            'btc_ath_percent': btc_ath_status.get('percent_from_ath', 0),
            'altcoin_btc_ratio': altcoin_btc_ratio.get('current_ratio', 0),
            'altcoin_btc_ratio_trend': altcoin_btc_ratio.get('ratio_trend', 'neutral'),
            'phases_count': phases_count,
            'confidence': _calculate_phase_confidence(phases_count, dominant_phase)
        }

        return results

    except Exception as e:
        logger.error(f"خطا در تعیین فاز بازار: {e}")
        return {'error': str(e), 'phase': 'unknown'}


async def _analyze_asset_phase(symbol: str, data_fetcher, config: Dict[str, Any]) -> Dict[str, Any]:
    """تحلیل فاز بازار برای یک دارایی خاص"""
    results = {'symbol': symbol}

    # دریافت داده‌های تایم‌فریم روزانه
    df_daily = await data_fetcher.get_historical_data(
        symbol,
        '1d',
        limit=max(config['analysis_period'], 250)  # حداقل 250 روز برای محاسبه MA200
    )

    if df_daily is None or df_daily.empty:
        return {'phase': 'unknown', 'reason': 'no_data'}

    # محاسبه میانگین‌های متحرک
    for period in config['ma_periods']:
        df_daily[f'ma_{period}'] = df_daily['close'].rolling(window=period).mean()

    # محاسبه ATH و ATL در دوره تحلیل
    ath = df_daily['high'].max()
    ath_date = df_daily.loc[df_daily['high'] == ath].index[0]
    atl = df_daily['low'].min()
    atl_date = df_daily.loc[df_daily['low'] == atl].index[0]

    current_price = df_daily['close'].iloc[-1]
    percent_from_ath = ((current_price - ath) / ath) * 100
    percent_from_atl = ((current_price - atl) / atl) * 100

    # محاسبه نوسان‌پذیری
    df_daily['daily_returns'] = df_daily['close'].pct_change()
    recent_volatility = df_daily['daily_returns'].iloc[-config['volatility_period']:].std() * 100

    # دریافت داده‌های هفتگی
    df_weekly = await data_fetcher.get_historical_data(
        symbol,
        '1w',
        limit=52  # حداکثر یک سال
    )

    # تحلیل روند هفتگی
    weekly_trend = 'neutral'
    if df_weekly is not None and not df_weekly.empty and len(df_weekly) >= 8:
        df_weekly['8_week_ma'] = df_weekly['close'].rolling(window=8).mean()
        if df_weekly['close'].iloc[-1] > df_weekly['8_week_ma'].iloc[-1]:
            weekly_trend = 'up'
        else:
            weekly_trend = 'down'

    # 1. تشخیص فاز صعودی (Bull)
    is_bull = (
            df_daily['close'].iloc[-1] > df_daily['ma_50'].iloc[-1] > df_daily['ma_200'].iloc[-1] and
            percent_from_ath > -20 and
            weekly_trend == 'up'
    )

    # 2. تشخیص فاز نزولی (Bear)
    is_bear = (
            df_daily['close'].iloc[-1] < df_daily['ma_50'].iloc[-1] < df_daily['ma_200'].iloc[-1] and
            percent_from_ath < -30 and
            weekly_trend == 'down'
    )

    # 3. تشخیص فاز انباشت (Accumulation)
    is_accumulation = (
            percent_from_atl < 30 and
            percent_from_atl > 0 and
            recent_volatility < 5 and
            not is_bull and not is_bear
    )

    # 4. تشخیص فاز توزیع (Distribution)
    is_distribution = (
            percent_from_ath > -20 and
            percent_from_ath < 0 and
            recent_volatility < 5 and
            not is_bull and not is_bear
    )

    # 5. فاز رنج (Sideways)
    is_sideways = not (is_bull or is_bear or is_accumulation or is_distribution)

    # تعیین فاز نهایی
    if is_bull:
        phase = 'bull'
    elif is_bear:
        phase = 'bear'
    elif is_accumulation:
        phase = 'accumulation'
    elif is_distribution:
        phase = 'distribution'
    else:
        phase = 'sideways'

    results['phase'] = phase
    results['ma_status'] = {
        'above_ma50': df_daily['close'].iloc[-1] > df_daily['ma_50'].iloc[-1],
        'above_ma200': df_daily['close'].iloc[-1] > df_daily['ma_200'].iloc[-1],
        'ma50_above_ma200': df_daily['ma_50'].iloc[-1] > df_daily['ma_200'].iloc[-1]
    }
    results['ath_status'] = {
        'ath': ath,
        'ath_date': str(ath_date.date()),
        'percent_from_ath': percent_from_ath
    }
    results['atl_status'] = {
        'atl': atl,
        'atl_date': str(atl_date.date()),
        'percent_from_atl': percent_from_atl
    }
    results['volatility'] = {
        'recent_volatility': recent_volatility,
        'weekly_trend': weekly_trend
    }

    return results


async def _calculate_altcoin_btc_ratio(data_fetcher) -> Dict[str, Any]:
    """محاسبه نسبت آلتکوین به بیت‌کوین"""
    try:
        # محاسبه نسبت با استفاده از شاخص مارکت کپ
        total_data = await data_fetcher.get_historical_data('TOTAL', '1d', limit=60)
        btc_data = await data_fetcher.get_historical_data('BTCUSDT', '1d', limit=60)
        total_without_btc = await data_fetcher.get_historical_data('TOTAL2', '1d', limit=60)

        if total_data is None or btc_data is None:
            # استفاده از چند ارز بزرگ به عنوان جایگزین
            alt_symbols = ['ETHUSDT', 'BNBUSDT', 'SOLUSDT', 'ADAUSDT', 'XRPUSDT']
            alt_data = {}

            for symbol in alt_symbols:
                df = await data_fetcher.get_historical_data(symbol, '1d', limit=30)
                if df is not None and not df.empty:
                    alt_data[symbol] = df

            if not alt_data or 'ETHUSDT' not in alt_data:
                return {'current_ratio': 0, 'ratio_trend': 'neutral'}

            # محاسبه نسبت ساده ETH/BTC به عنوان نماینده
            eth_btc_ratio = alt_data['ETHUSDT']['close'].iloc[-1] / btc_data['close'].iloc[-1]
            eth_btc_ratio_30d_ago = alt_data['ETHUSDT']['close'].iloc[0] / btc_data['close'].iloc[0]

            ratio_change = (eth_btc_ratio - eth_btc_ratio_30d_ago) / eth_btc_ratio_30d_ago * 100

            ratio_trend = 'up' if ratio_change > 5 else 'down' if ratio_change < -5 else 'neutral'

            return {
                'current_ratio': eth_btc_ratio,
                'ratio_change_percent': ratio_change,
                'ratio_trend': ratio_trend
            }

        else:
            # محاسبه نسبت مارکت کپ کل بدون بیت‌کوین به مارکت کپ بیت‌کوین
            altcoin_mcap = total_without_btc['close'].iloc[-1]
            btc_mcap = btc_data['close'].iloc[-1] * btc_data['volume'].iloc[-1]  # تقریب ساده

            alt_btc_ratio = altcoin_mcap / btc_mcap

            # محاسبه روند 30 روزه
            altcoin_mcap_30d_ago = total_without_btc['close'].iloc[0]
            btc_mcap_30d_ago = btc_data['close'].iloc[0] * btc_data['volume'].iloc[0]

            alt_btc_ratio_30d_ago = altcoin_mcap_30d_ago / btc_mcap_30d_ago
            ratio_change = (alt_btc_ratio - alt_btc_ratio_30d_ago) / alt_btc_ratio_30d_ago * 100

            ratio_trend = 'up' if ratio_change > 5 else 'down' if ratio_change < -5 else 'neutral'

            return {
                'current_ratio': alt_btc_ratio,
                'ratio_30d_ago': alt_btc_ratio_30d_ago,
                'ratio_change_percent': ratio_change,
                'ratio_trend': ratio_trend
            }

    except Exception as e:
        logger.error(f"خطا در محاسبه نسبت آلتکوین/بیت‌کوین: {e}")
        return {'current_ratio': 0, 'ratio_trend': 'neutral'}


def _determine_market_status(phase: str, btc_phase: Dict[str, Any],
                             alt_btc_ratio: Dict[str, Any]) -> str:
    """تعیین وضعیت کلی بازار"""
    if phase == 'bull':
        if alt_btc_ratio.get('ratio_trend') == 'up':
            return 'alt_season'
        else:
            return 'btc_season'
    elif phase == 'bear':
        if btc_phase.get('ath_status', {}).get('percent_from_ath', -100) < -60:
            return 'deep_bear'
        else:
            return 'early_bear'
    elif phase == 'accumulation':
        return 'bottom_accumulation'
    elif phase == 'distribution':
        return 'top_distribution'
    else:
        return 'market_uncertainty'


def _calculate_phase_confidence(phases_count: Dict[str, int], dominant_phase: str) -> float:
    """محاسبه اطمینان از تشخیص فاز"""
    total_votes = sum(phases_count.values())
    if total_votes == 0:
        return 0

    dominant_votes = phases_count.get(dominant_phase, 0)
    return (dominant_votes / total_votes) * 100


async def analyze_correlation_lag(symbol1: str, symbol2: str, data_fetcher,
                                  timeframe: str = '1h', max_lag: int = 5) -> Dict[str, Any]:
    """
    تحلیل همبستگی با تاخیر بین دو نماد

    Args:
        symbol1: نماد اول (مثلاً یک آلتکوین)
        symbol2: نماد دوم (معمولاً بیت‌کوین)
        data_fetcher: وهله DataFetcher برای دریافت داده‌ها
        timeframe: تایم‌فریم برای تحلیل
        max_lag: حداکثر تاخیر برای بررسی

    Returns:
        دیکشنری نتایج تحلیل تاخیر
    """
    try:
        # دریافت داده‌های تاریخی
        period = 100 + max_lag  # دوره بیشتر برای بررسی تاخیر

        df1 = await data_fetcher.get_historical_data(symbol1, timeframe, limit=period)
        df2 = await data_fetcher.get_historical_data(symbol2, timeframe, limit=period)

        if df1 is None or df2 is None or df1.empty or df2.empty:
            return {'error': 'Insufficient data', 'best_lag': 0, 'lag_correlations': {}}

        # محاسبه تغییرات قیمت
        returns1 = df1['close'].pct_change().fillna(0)
        returns2 = df2['close'].pct_change().fillna(0)

        # محاسبه همبستگی با تاخیرهای مختلف
        lag_correlations = {}

        # همبستگی بدون تاخیر (lag=0)
        lag_correlations[0] = float(returns1.corr(returns2))

        # همبستگی با تاخیرهای مثبت (نماد دوم جلوتر)
        for lag in range(1, max_lag + 1):
            shifted = returns2.shift(lag)
            corr = float(returns1.corr(shifted))
            if not np.isnan(corr) and not np.isinf(corr):
                lag_correlations[lag] = corr
            else:
                lag_correlations[lag] = 0.0

        # همبستگی با تاخیرهای منفی (نماد اول جلوتر)
        for lag in range(1, max_lag + 1):
            shifted = returns1.shift(lag)
            corr = float(shifted.corr(returns2))
            if not np.isnan(corr) and not np.isinf(corr):
                lag_correlations[-lag] = corr
            else:
                lag_correlations[-lag] = 0.0

        # یافتن بهترین تاخیر (با بیشترین مقدار مطلق همبستگی)
        best_lag = 0
        best_corr = 0.0
        best_abs_corr = 0.0

        for lag, corr in lag_correlations.items():
            abs_corr = abs(corr)
            if abs_corr > best_abs_corr:
                best_abs_corr = abs_corr
                best_corr = corr
                best_lag = lag

        return {
            'best_lag': best_lag,
            'best_correlation': best_corr,
            'lag_correlations': lag_correlations,
            'interpretation': interpret_lag_correlation(best_lag, best_corr, symbol1, symbol2)
        }

    except Exception as e:
        logger.error(f"خطا در تحلیل همبستگی با تاخیر بین {symbol1} و {symbol2}: {e}")
        return {'error': str(e), 'best_lag': 0, 'lag_correlations': {}}


def interpret_lag_correlation(lag: int, correlation: float, symbol1: str, symbol2: str) -> str:
    """
    تفسیر نتایج همبستگی با تاخیر

    Args:
        lag: بهترین تاخیر
        correlation: مقدار همبستگی
        symbol1: نماد اول
        symbol2: نماد دوم

    Returns:
        متن تفسیر
    """
    abs_corr = abs(correlation)
    corr_strength = ""

    if abs_corr < 0.2:
        corr_strength = "ضعیف"
    elif abs_corr < 0.5:
        corr_strength = "متوسط"
    else:
        corr_strength = "قوی"

    corr_type = "مثبت" if correlation > 0 else "منفی"

    if lag == 0:
        return f"همبستگی {corr_type} {corr_strength} ({correlation:.2f}) بدون تاخیر بین {symbol1} و {symbol2}"
    elif lag > 0:
        return f"همبستگی {corr_type} {corr_strength} ({correlation:.2f}) با تاخیر {lag} دوره. {symbol2} جلوتر از {symbol1} حرکت می‌کند."
    else:
        return f"همبستگی {corr_type} {corr_strength} ({correlation:.2f}) با تاخیر {abs(lag)} دوره. {symbol1} جلوتر از {symbol2} حرکت می‌کند."

class CorrelationManager:
    """
    مدیریت همبستگی بین نمادها برای کنترل ریسک سبد معاملات
    """

    def __init__(self, config: Dict[str, Any]):
        """
        مقداردهی اولیه

        Args:
            config: دیکشنری تنظیمات
        """
        self.config = config
        self.risk_config = config.get('risk_management', {})

        # آستانه‌های همبستگی
        self.max_correlation = self.risk_config.get('max_correlation_threshold', 0.7)
        self.max_correlated_trades = self.risk_config.get('max_correlated_trades', 2)

        # تنظیمات محاسبه همبستگی
        self.correlation_timeframe = self.risk_config.get('correlation_timeframe', '1h')
        self.correlation_period = self.risk_config.get('correlation_period', 100)

        # کش همبستگی‌های قبلی - بهینه‌سازی با استفاده از TTL (Time To Live)
        self.correlation_cache = {}  # {symbol_pair: (correlation, timestamp)}
        self.cache_expiry_seconds = self.risk_config.get('correlation_cache_expiry_seconds', 3600)

        # استفاده از ThreadPoolExecutor برای محاسبات همزمان
        self._executor = ThreadPoolExecutor(max_workers=4)

        # قفل برای دسترسی همزمان به کش
        self._cache_lock = threading.RLock()

        # تنظیمات جدید برای همبستگی با بیت‌کوین
        self.btc_correlation_config = self.risk_config.get('btc_correlation', {})
        self.consider_btc_trend = self.btc_correlation_config.get('consider_btc_trend', True)
        self.btc_symbol = self.btc_correlation_config.get('btc_symbol', 'BTCUSDT')
        self.btc_trend_timeframe = self.btc_correlation_config.get('btc_trend_timeframe', '4h')
        self.btc_trend_period = self.btc_correlation_config.get('btc_trend_period', 50)
        self.inverse_correlation_threshold = self.btc_correlation_config.get('inverse_correlation_threshold', -0.2)
        self.zero_correlation_threshold = self.btc_correlation_config.get('zero_correlation_threshold', 0.2)
        self.btc_trend_ema_fast = self.btc_correlation_config.get('btc_trend_ema_fast', 20)
        self.btc_trend_ema_slow = self.btc_correlation_config.get('btc_trend_ema_slow', 50)
        self.btc_trend_strength_threshold = self.btc_correlation_config.get('btc_trend_strength_threshold', 0.01)

        # کش روند بیت‌کوین
        self.btc_trend_cache = {
            'trend': None,  # 'bullish', 'bearish', 'neutral'
            'strength': 0.0,
            'timestamp': None,
            'last_price': 0.0,
            'data': None
        }

        # مدت زمان اعتبار کش روند (به ثانیه)
        self.btc_trend_cache_expiry = self.btc_correlation_config.get('btc_trend_cache_expiry', 1800)  # 30 دقیقه
        # تنظیم تحلیل‌گر بیت‌کوین
        self.btc_analyzer = None  # برای ساخت تنبل (lazy initialization)
        self.consider_btc_trend = self.config.get('risk_management', {}).get('btc_correlation', {}).get(
            'consider_btc_trend', True)
        logger.info("CorrelationManager با آستانه همبستگی %.2f راه‌اندازی شد", self.max_correlation)
        if self.consider_btc_trend:
            logger.info("تحلیل روند بیت‌کوین با تایم‌فریم %s فعال شد", self.btc_trend_timeframe)

    def __del__(self):
        """پاکسازی منابع هنگام حذف شیء"""
        if hasattr(self, '_executor'):
            self._executor.shutdown(wait=False)

    def update_config(self, new_config: Dict[str, Any]):
        """
        به‌روزرسانی تنظیمات در زمان اجرا

        Args:
            new_config: دیکشنری تنظیمات جدید
        """
        old_config = self.config.copy()

        # به‌روزرسانی تنظیمات اصلی
        self.config = new_config

        # به‌روزرسانی تنظیمات ریسک
        self.risk_config = new_config.get('risk_management', {})

        # به‌روزرسانی پارامترهای همبستگی
        old_max_correlation = self.max_correlation
        old_max_correlated_trades = self.max_correlated_trades

        self.max_correlation = self.risk_config.get('max_correlation_threshold', 0.7)
        self.max_correlated_trades = self.risk_config.get('max_correlated_trades', 2)
        self.correlation_timeframe = self.risk_config.get('correlation_timeframe', '1h')
        self.correlation_period = self.risk_config.get('correlation_period', 100)
        self.cache_expiry_seconds = self.risk_config.get('correlation_cache_expiry_seconds', 3600)

        # به‌روزرسانی تنظیمات بیت‌کوین
        self.btc_correlation_config = self.risk_config.get('btc_correlation', {})
        old_consider_btc_trend = self.consider_btc_trend

        self.consider_btc_trend = self.btc_correlation_config.get('consider_btc_trend', True)
        self.btc_symbol = self.btc_correlation_config.get('btc_symbol', 'BTCUSDT')
        self.btc_trend_timeframe = self.btc_correlation_config.get('btc_trend_timeframe', '4h')
        self.btc_trend_period = self.btc_correlation_config.get('btc_trend_period', 50)
        self.inverse_correlation_threshold = self.btc_correlation_config.get('inverse_correlation_threshold', -0.2)
        self.zero_correlation_threshold = self.btc_correlation_config.get('zero_correlation_threshold', 0.2)
        self.btc_trend_ema_fast = self.btc_correlation_config.get('btc_trend_ema_fast', 20)
        self.btc_trend_ema_slow = self.btc_correlation_config.get('btc_trend_ema_slow', 50)
        self.btc_trend_strength_threshold = self.btc_correlation_config.get('btc_trend_strength_threshold', 0.01)
        self.btc_trend_cache_expiry = self.btc_correlation_config.get('btc_trend_cache_expiry', 1800)

        # پاکسازی کش در صورت تغییر پارامترهای مهم
        if (old_max_correlation != self.max_correlation or
            old_max_correlated_trades != self.max_correlated_trades or
            old_consider_btc_trend != self.consider_btc_trend):
            with self._cache_lock:
                self.correlation_cache = {}
                self.btc_trend_cache = {
                    'trend': None,
                    'strength': 0.0,
                    'timestamp': None,
                    'last_price': 0.0,
                    'data': None
                }
            logger.info("کش همبستگی به دلیل تغییر پارامترهای مهم پاکسازی شد")

        logger.info("تنظیمات CorrelationManager به‌روزرسانی شد: max_correlation=%.2f, max_correlated_trades=%d",
                   self.max_correlation, self.max_correlated_trades)
        if self.consider_btc_trend:
            logger.info("تحلیل روند بیت‌کوین فعال است با تایم‌فریم %s", self.btc_trend_timeframe)
        else:
            logger.info("تحلیل روند بیت‌کوین غیرفعال است")

    @cache_result(timeout_seconds=3600)  # استفاده از دکوراتور کش
    async def calculate_correlation(self, symbol1: str, symbol2: str,
                                    df1: pd.DataFrame, df2: pd.DataFrame) -> float:
        """
        محاسبه همبستگی بین دو نماد با کنترل‌های بهتر خطا

        Args:
            symbol1: نماد اول
            symbol2: نماد دوم
            df1: دیتافریم قیمت نماد اول
            df2: دیتافریم قیمت نماد دوم

        Returns:
            ضریب همبستگی پیرسون
        """
        if df1.empty or df2.empty:
            logger.debug(f"دیتافریم خالی برای محاسبه همبستگی: {symbol1} یا {symbol2}")
            return 0.0

        try:
            # مشترک‌سازی شاخص‌ها
            common_index = df1.index.intersection(df2.index)

            # بررسی حداقل تعداد داده‌ها - افزایش به 30 برای نتایج آماری معتبرتر
            if len(common_index) < 20:
                logger.debug(
                    f"داده‌های ناکافی برای محاسبه همبستگی: {symbol1} و {symbol2}، {len(common_index)} نقطه یافت شد")
                return 0.0

            # استفاده از loc برای استخراج داده‌های مشترک - بهینه‌تر از reindex
            df1_aligned = df1.loc[common_index]
            df2_aligned = df2.loc[common_index]

            # بررسی ستون 'close' وجود دارد
            if 'close' not in df1_aligned.columns or 'close' not in df2_aligned.columns:
                logger.warning(f"ستون 'close' برای محاسبه همبستگی یافت نشد: {symbol1} یا {symbol2}")
                return 0.0

            # محاسبه تغییرات قیمت
            returns1 = df1_aligned['close'].pct_change().fillna(0)
            returns2 = df2_aligned['close'].pct_change().fillna(0)

            # بررسی سطح تغییرات - برای اطمینان از معنادار بودن
            if returns1.std() < 1e-6 or returns2.std() < 1e-6:
                logger.debug(f"تغییرات قیمت ناچیز برای محاسبه همبستگی: {symbol1} یا {symbol2}")
                return 0.0

            # استفاده از numpy برای محاسبه سریع‌تر همبستگی
            # حذف داده‌های ناقص
            mask = ~(np.isnan(returns1.values) | np.isnan(returns2.values))
            valid_data_count = np.sum(mask)

            if valid_data_count < 5:  # حداقل 5 نقطه داده معتبر
                logger.debug(f"تعداد داده معتبر پس از فیلتر بسیار کم است: {valid_data_count}")
                return 0.0

            # اگر فقط یک داده معتبر وجود دارد، corrcoef خطا می‌دهد
            if valid_data_count == 1:
                return 0.0

            try:
                # استفاده از pandas به جای numpy برای مدیریت بهتر خطاها
                correlation = returns1[mask].corr(returns2[mask])

                # مدیریت مقادیر NaN یا Inf
                if pd.isna(correlation) or np.isinf(correlation):
                    return 0.0

                return correlation
            except Exception as inner_e:
                # خطا در محاسبه همبستگی - تلاش دوم با روش دیگر
                logger.debug(f"خطا در محاسبه همبستگی، استفاده از روش جایگزین: {inner_e}")
                try:
                    correlation_matrix = np.corrcoef(returns1.values[mask], returns2.values[mask])
                    if correlation_matrix.shape == (2, 2):
                        correlation = correlation_matrix[0, 1]
                        if np.isnan(correlation) or np.isinf(correlation):
                            return 0.0
                        return correlation
                    return 0.0
                except Exception as np_error:
                    logger.debug(f"محاسبه همبستگی با روش جایگزین هم ناموفق بود: {np_error}")
                    return 0.0

        except Exception as e:
            logger.error(f"خطا در محاسبه همبستگی بین {symbol1} و {symbol2}: {e}")
            return 0.0

    async def _get_cached_correlation(self, symbol1: str, symbol2: str) -> Optional[float]:
        """
        بازیابی همبستگی کش شده اگر معتبر باشد

        Args:
            symbol1: نماد اول
            symbol2: نماد دوم

        Returns:
            همبستگی کش شده یا None اگر معتبر نباشد
        """
        # ایجاد کلید منظم شده برای جستجوی کش
        key_pair = tuple(sorted([symbol1, symbol2]))
        cache_key = f"{key_pair[0]}_{key_pair[1]}"

        with self._cache_lock:
            if cache_key in self.correlation_cache:
                cached_corr, timestamp = self.correlation_cache[cache_key]
                age = (datetime.now() - timestamp).total_seconds()
                if age < self.cache_expiry_seconds:
                    return cached_corr

        return None

    async def analyze_btc_trend(self, data_fetcher) -> Dict[str, Any]:
        """
        تحلیل روند بیت‌کوین

        Args:
            data_fetcher: وهله MarketDataFetcher برای دریافت داده‌ها

        Returns:
            دیکشنری از اطلاعات روند شامل {'trend', 'strength', 'last_price'}
        """
        # بررسی کش روند بیت‌کوین
        current_time = datetime.now()
        if (self.btc_trend_cache['timestamp'] and
                (current_time - self.btc_trend_cache['timestamp']).total_seconds() < self.btc_trend_cache_expiry):
            return {
                'trend': self.btc_trend_cache['trend'],
                'strength': self.btc_trend_cache['strength'],
                'last_price': self.btc_trend_cache['last_price']
            }

        try:
            # دریافت داده‌های بیت‌کوین
            btc_data = await data_fetcher.get_historical_data(
                self.btc_symbol,
                self.btc_trend_timeframe,
                limit=self.btc_trend_period
            )

            if btc_data is None or btc_data.empty or len(btc_data) < self.btc_trend_ema_slow:
                logger.warning(f"داده‌های ناکافی برای تحلیل روند بیت‌کوین: {self.btc_symbol}")
                return {
                    'trend': 'neutral',
                    'strength': 0.0,
                    'last_price': 0.0
                }

            # محاسبه میانگین متحرک نمایی سریع و آهسته
            btc_data['ema_fast'] = btc_data['close'].ewm(span=self.btc_trend_ema_fast, adjust=False).mean()
            btc_data['ema_slow'] = btc_data['close'].ewm(span=self.btc_trend_ema_slow, adjust=False).mean()

            # محاسبه شکاف نسبی بین دو EMA
            btc_data['ema_gap'] = (btc_data['ema_fast'] - btc_data['ema_slow']) / btc_data['ema_slow']

            # قیمت آخر و شکاف EMA
            last_ema_gap = btc_data['ema_gap'].iloc[-1]
            last_price = btc_data['close'].iloc[-1]

            # تعیین روند و قدرت روند
            trend = 'neutral'
            if last_ema_gap > self.btc_trend_strength_threshold:
                trend = 'bullish'
            elif last_ema_gap < -self.btc_trend_strength_threshold:
                trend = 'bearish'

            # محاسبه قدرت روند (مقدار مطلق شکاف EMA)
            trend_strength = abs(last_ema_gap)

            # تحلیل تکمیلی - شیب EMA
            ema_slope = (btc_data['ema_fast'].iloc[-1] - btc_data['ema_fast'].iloc[-5]) / btc_data['ema_fast'].iloc[-5]

            # تقویت قدرت روند بر اساس شیب
            if (trend == 'bullish' and ema_slope > 0) or (trend == 'bearish' and ema_slope < 0):
                trend_strength = trend_strength * (1 + abs(ema_slope))

            # به‌روزرسانی کش
            self.btc_trend_cache = {
                'trend': trend,
                'strength': trend_strength,
                'timestamp': current_time,
                'last_price': last_price,
                'data': btc_data
            }

            logger.debug(f"روند بیت‌کوین: {trend}، قدرت: {trend_strength:.4f}، قیمت: {last_price}")

            return {
                'trend': trend,
                'strength': trend_strength,
                'last_price': last_price
            }

        except Exception as e:
            logger.error(f"خطا در تحلیل روند بیت‌کوین: {e}")
            return {
                'trend': 'neutral',
                'strength': 0.0,
                'last_price': 0.0
            }

    async def check_btc_correlation_compatibility(self, symbol: str, direction: str,
                                                  data_fetcher) -> Dict[str, Any]:
        """
        بررسی سازگاری همبستگی با بیت‌کوین برای معامله با استفاده از تحلیل‌گر پیشرفته

        Args:
            symbol: نماد معامله
            direction: جهت معامله ('long' یا 'short')
            data_fetcher: وهله MarketDataFetcher برای دریافت داده‌ها

        Returns:
            دیکشنری اطلاعات سازگاری همبستگی
        """
        if not self.consider_btc_trend or symbol == self.btc_symbol:
            return {
                'is_compatible': True,
                'reason': 'btc_trend_analysis_disabled_or_btc_itself',
                'correlation_score': 0  # امتیاز خنثی
            }

        try:
            # آیا تحلیل‌گر بیت‌کوین وجود دارد؟
            if hasattr(self, 'btc_analyzer') and self.btc_analyzer:
                analyzer = self.btc_analyzer
            else:
                # اگر تحلیل‌گر وجود ندارد، آن را ایجاد کن
                self.btc_analyzer = BTCCorrelationAnalyzer(self.config)
                analyzer = self.btc_analyzer

            # دریافت خلاصه تحلیل همبستگی
            correlation_summary = await analyzer.get_correlation_summary(symbol, direction, data_fetcher)

            # معیار سازگاری - آیا امتیاز منفی سنگین دارد؟
            correlation_score = correlation_summary.get('correlation_score', 0)
            is_compatible = correlation_score > -30  # اگر امتیاز خیلی پایین است (مثلاً کمتر از -30)، ناسازگار است

            # تعیین دلیل براساس نوع همبستگی و روند
            reason = "default_compatible"

            correlation_type = correlation_summary.get('correlation_type', 'unknown')
            btc_trend = correlation_summary.get('btc_trend', 'neutral')

            if not is_compatible:
                if correlation_type == 'positive':
                    if btc_trend == 'bullish' and direction == 'short':
                        reason = 'rejected_short_correlated_coin_in_btc_bullish_trend'
                    elif btc_trend == 'bearish' and direction == 'long':
                        reason = 'rejected_long_correlated_coin_in_btc_bearish_trend'
                elif correlation_type == 'inverse':
                    reason = 'rejected_inverse_correlation_with_unusual_pattern'
                else:
                    reason = 'rejected_due_to_negative_compatibility_score'
            else:
                # دلیل سازگاری
                if correlation_type == 'positive':
                    if (btc_trend == 'bullish' and direction == 'long') or (
                            btc_trend == 'bearish' and direction == 'short'):
                        reason = 'approved_trade_aligned_with_btc_trend'
                    else:
                        reason = 'approved_despite_opposite_btc_trend'
                elif correlation_type == 'inverse':
                    reason = 'approved_inverse_correlation'
                else:
                    reason = 'approved_zero_correlation'

            # ساخت پاسخ نهایی
            return {
                'is_compatible': is_compatible,
                'reason': reason,
                'correlation_with_btc': correlation_summary.get('weighted_correlation', 0),
                'correlation_type': correlation_type,
                'btc_trend': btc_trend,
                'btc_trend_strength': correlation_summary.get('btc_trend_strength', 0),
                'direction': direction,
                'correlation_score': correlation_score,
                'best_lag': correlation_summary.get('best_lag', 0),
                'is_high_volume': correlation_summary.get('is_high_volume', False),
                'compatibility_score': correlation_summary.get('compatibility_score', 50)
            }

        except Exception as e:
            logger.error(f"خطا در بررسی سازگاری همبستگی بیت‌کوین: {e}")
            return {
                'is_compatible': True,  # در صورت خطا فرض بر سازگاری است
                'reason': f'error_in_analysis: {str(e)}',
                'correlation_score': 0  # امتیاز خنثی
            }

    async def check_portfolio_correlation(self, new_symbol: str, direction: str,
                                          open_trades: List[Trade],
                                          data_fetcher) -> Tuple[bool, float, List[str], Dict[str, Any]]:
        """
        بررسی همبستگی بین نماد جدید و معاملات فعلی

        Args:
            new_symbol: نماد جدید
            direction: جهت معامله ('long' یا 'short')
            open_trades: لیست معاملات فعال
            data_fetcher: وهله MarketDataFetcher برای دریافت داده‌ها

        Returns:
            tuple شامل (آیا معامله مجاز است، سطح همبستگی، لیست نمادهای همبسته، اطلاعات همبستگی با بیت‌کوین)
        """
        if not open_trades:
            # بررسی همبستگی با بیت‌کوین حتی اگر معامله فعالی نباشد
            btc_compatibility = await self.check_btc_correlation_compatibility(
                new_symbol, direction, data_fetcher
            )

            return True, 0.0, [], btc_compatibility

        # دریافت داده‌های نماد جدید
        new_symbol_data = await data_fetcher.get_historical_data(
            new_symbol,
            self.correlation_timeframe,
            limit=self.correlation_period
        )

        if new_symbol_data is None or new_symbol_data.empty:
            logger.warning(f"نمی‌توان داده‌ها را برای تحلیل همبستگی دریافت کرد: {new_symbol}")
            return True, 0.0, [], {'is_compatible': True, 'reason': 'no_data', 'correlation_score': 0}

        # بررسی همبستگی با تمام معاملات فعال
        correlated_symbols = []
        max_correlation_value = 0.0

        # مدیریت صف معاملات فعال برای اولویت‌بندی - نمادهای غیرهمبسته قبلی را نادیده بگیر
        potentially_correlated_trades = []
        already_checked_symbols = set()

        for trade in open_trades:
            # نادیده گرفتن معاملات همان نماد
            if trade.symbol == new_symbol or trade.symbol in already_checked_symbols:
                continue

            # بازیابی از کش
            cached_corr = await self._get_cached_correlation(new_symbol, trade.symbol)
            if cached_corr is not None:
                # اصلاح همبستگی برای جهت معامله
                direction_modifier = -1.0 if trade.direction != direction else 1.0
                correlation = cached_corr * direction_modifier

                absolute_correlation = abs(correlation)
                if absolute_correlation > max_correlation_value:
                    max_correlation_value = absolute_correlation

                if absolute_correlation > self.max_correlation:
                    correlated_symbols.append(trade.symbol)

                already_checked_symbols.add(trade.symbol)
            else:
                potentially_correlated_trades.append(trade)

        # محاسبه همبستگی‌های باقیمانده
        for trade in potentially_correlated_trades:
            # دریافت داده‌های معامله فعال
            trade_symbol_data = await data_fetcher.get_historical_data(trade.symbol, self.correlation_timeframe,
                                                                       limit=self.correlation_period)
            if trade_symbol_data is None or trade_symbol_data.empty:
                continue

            # محاسبه همبستگی
            correlation = await self.calculate_correlation(
                new_symbol,
                trade.symbol,
                new_symbol_data,
                trade_symbol_data
            )

            # ذخیره در کش
            key_pair = tuple(sorted([new_symbol, trade.symbol]))
            cache_key = f"{key_pair[0]}_{key_pair[1]}"

            with self._cache_lock:
                self.correlation_cache[cache_key] = (correlation, datetime.now())

            # اصلاح همبستگی برای جهت معامله
            direction_modifier = -1.0 if trade.direction != direction else 1.0
            correlation = correlation * direction_modifier

            absolute_correlation = abs(correlation)
            if absolute_correlation > max_correlation_value:
                max_correlation_value = absolute_correlation

            if absolute_correlation > self.max_correlation:
                correlated_symbols.append(trade.symbol)

        # تصمیم‌گیری بر اساس تعداد نمادهای همبسته
        is_allowed = len(correlated_symbols) < self.max_correlated_trades

        if not is_allowed:
            logger.info(
                f"همبستگی بیش از حد برای {new_symbol}: {len(correlated_symbols)} > {self.max_correlated_trades}")
            logger.info(f"نمادهای همبسته: {', '.join(correlated_symbols)}")

        # بررسی همبستگی با بیت‌کوین
        btc_compatibility = await self.check_btc_correlation_compatibility(
            new_symbol, direction, data_fetcher
        )

        # ترکیب نتایج همبستگی پورتفولیو و همبستگی بیت‌کوین
        final_is_allowed = is_allowed and btc_compatibility['is_compatible']

        if not btc_compatibility['is_compatible']:
            logger.info(
                f"معامله {new_symbol} به دلیل همبستگی نامطلوب با بیت‌کوین رد شد: {btc_compatibility['reason']}")

        return final_is_allowed, max_correlation_value, correlated_symbols, btc_compatibility

    def clear_cache(self, older_than_hours: int = 24):
        """
        پاکسازی کش قدیمی برای آزادسازی حافظه

        Args:
            older_than_hours: حذف کش‌های قدیمی‌تر از این مقدار ساعت
        """
        cutoff_time = datetime.now() - timedelta(hours=older_than_hours)

        with self._cache_lock:
            keys_to_remove = []
            for key, (_, timestamp) in self.correlation_cache.items():
                if timestamp < cutoff_time:
                    keys_to_remove.append(key)

            for key in keys_to_remove:
                del self.correlation_cache[key]

            # پاکسازی کش روند بیت‌کوین
            if self.btc_trend_cache['timestamp'] and self.btc_trend_cache['timestamp'] < cutoff_time:
                self.btc_trend_cache = {
                    'trend': None,
                    'strength': 0.0,
                    'timestamp': None,
                    'last_price': 0.0,
                    'data': None
                }

        logger.debug(f"{len(keys_to_remove)} مورد کش منقضی شده پاکسازی شد")


class BTCCorrelationAnalyzer:
    """
    تحلیل‌گر پیشرفته همبستگی با بیت‌کوین با پشتیبانی از همبستگی چندتایم‌فریمی و تحلیل تاخیر
    """

    def __init__(self, config: Dict[str, Any]):
        """
        مقداردهی اولیه

        Args:
            config: دیکشنری تنظیمات
        """
        self.config = config
        self.btc_correlation_config = config.get('risk_management', {}).get('btc_correlation', {})

        # تنظیمات همبستگی
        self.consider_btc_trend = self.btc_correlation_config.get('consider_btc_trend', True)
        self.btc_symbol = self.btc_correlation_config.get('btc_symbol', 'BTCUSDT')

        # تایم‌فریم‌های تحلیل همبستگی
        self.correlation_timeframes = self.btc_correlation_config.get('correlation_timeframes',
                                                                      ['15m', '1h', '4h', '1d'])
        self.correlation_timeframe_weights = self.btc_correlation_config.get('correlation_timeframe_weights',
                                                                             [0.1, 0.2, 0.3, 0.4])

        # تنظیمات تایم‌فریم اصلی
        self.primary_correlation_timeframe = self.btc_correlation_config.get('primary_correlation_timeframe', '1h')
        self.correlation_period = self.btc_correlation_config.get('correlation_period', 100)

        # آستانه‌های همبستگی
        self.inverse_correlation_threshold = self.btc_correlation_config.get('inverse_correlation_threshold', -0.2)
        self.zero_correlation_threshold = self.btc_correlation_config.get('zero_correlation_threshold', 0.2)

        # تنظیمات تحلیل تاخیر
        self.lag_correlation_analysis_enabled = self.btc_correlation_config.get('analyze_lag_correlation', True)
        self.max_lag_periods = self.btc_correlation_config.get('max_lag_periods', 5)

        # تنظیمات تحلیل روند بیت‌کوین
        self.btc_trend_timeframe = self.btc_correlation_config.get('btc_trend_timeframe', '4h')
        self.btc_trend_period = self.btc_correlation_config.get('btc_trend_period', 50)
        self.btc_trend_ema_fast = self.btc_correlation_config.get('btc_trend_ema_fast', 20)
        self.btc_trend_ema_slow = self.btc_correlation_config.get('btc_trend_ema_slow', 50)
        self.btc_trend_strength_threshold = self.btc_correlation_config.get('btc_trend_strength_threshold', 0.01)

        # تنظیمات تحلیل حجم معاملات بیت‌کوین
        self.btc_volume_analysis_enabled = self.btc_correlation_config.get('analyze_btc_volume', False)
        self.btc_volume_timeframe = self.btc_correlation_config.get('btc_volume_timeframe', '1d')
        self.btc_volume_period = self.btc_correlation_config.get('btc_volume_period', 20)
        self.btc_high_volume_threshold = self.btc_correlation_config.get('btc_high_volume_threshold', 1.5)

        # کش نتایج
        self._correlation_cache = {}  # {symbol_pair: (correlation, timestamp)}
        self._lag_correlation_cache = {}  # {symbol_pair_lag: (correlation, timestamp)}
        self._btc_trend_cache = {
            'trend': None,
            'strength': 0.0,
            'timestamp': None,
            'last_price': 0.0,
            'data': None
        }
        self._btc_volume_cache = {
            'normalized_volume': 0.0,
            'timestamp': None,
            'is_high_volume': False
        }

        # زمان انقضای کش (به ثانیه)
        self.cache_expiry_seconds = self.btc_correlation_config.get('correlation_cache_expiry_seconds', 3600)
        self.btc_trend_cache_expiry = self.btc_correlation_config.get('btc_trend_cache_expiry', 1800)
        self.btc_volume_cache_expiry = self.btc_correlation_config.get('btc_volume_cache_expiry', 1800)

        # قفل برای دسترسی همزمان
        self._cache_lock = threading.RLock()

        logger.info(f"BTCCorrelationAnalyzer با {len(self.correlation_timeframes)} تایم فریم راه‌اندازی شد")

    def update_config(self, new_config: Dict[str, Any]):
        """
        به‌روزرسانی تنظیمات در زمان اجرا

        Args:
            new_config: دیکشنری تنظیمات جدید
        """
        old_config = self.config.copy()

        # به‌روزرسانی تنظیمات اصلی
        self.config = new_config

        # به‌روزرسانی تنظیمات همبستگی بیت‌کوین
        old_btc_correlation_config = self.btc_correlation_config.copy()
        self.btc_correlation_config = new_config.get('risk_management', {}).get('btc_correlation', {})

        # به‌روزرسانی پارامترهای کلیدی
        old_consider_btc_trend = self.consider_btc_trend
        old_timeframes = self.correlation_timeframes

        self.consider_btc_trend = self.btc_correlation_config.get('consider_btc_trend', True)
        self.btc_symbol = self.btc_correlation_config.get('btc_symbol', 'BTCUSDT')
        self.correlation_timeframes = self.btc_correlation_config.get('correlation_timeframes',
                                                                    ['15m', '1h', '4h', '1d'])
        self.correlation_timeframe_weights = self.btc_correlation_config.get('correlation_timeframe_weights',
                                                                           [0.1, 0.2, 0.3, 0.4])
        self.primary_correlation_timeframe = self.btc_correlation_config.get('primary_correlation_timeframe', '1h')
        self.correlation_period = self.btc_correlation_config.get('correlation_period', 100)

        # آستانه‌های همبستگی
        self.inverse_correlation_threshold = self.btc_correlation_config.get('inverse_correlation_threshold', -0.2)
        self.zero_correlation_threshold = self.btc_correlation_config.get('zero_correlation_threshold', 0.2)

        # تنظیمات تحلیل تاخیر
        old_lag_analysis = self.lag_correlation_analysis_enabled
        self.lag_correlation_analysis_enabled = self.btc_correlation_config.get('analyze_lag_correlation', True)
        self.max_lag_periods = self.btc_correlation_config.get('max_lag_periods', 5)

        # تنظیمات تحلیل روند
        self.btc_trend_timeframe = self.btc_correlation_config.get('btc_trend_timeframe', '4h')
        self.btc_trend_period = self.btc_correlation_config.get('btc_trend_period', 50)
        self.btc_trend_ema_fast = self.btc_correlation_config.get('btc_trend_ema_fast', 20)
        self.btc_trend_ema_slow = self.btc_correlation_config.get('btc_trend_ema_slow', 50)
        self.btc_trend_strength_threshold = self.btc_correlation_config.get('btc_trend_strength_threshold', 0.01)

        # تنظیمات کش
        self.cache_expiry_seconds = self.btc_correlation_config.get('correlation_cache_expiry_seconds', 3600)
        self.btc_trend_cache_expiry = self.btc_correlation_config.get('btc_trend_cache_expiry', 1800)
        self.btc_volume_cache_expiry = self.btc_correlation_config.get('btc_volume_cache_expiry', 1800)

        # پاکسازی کش در صورت تغییر پارامترهای مهم
        if (old_consider_btc_trend != self.consider_btc_trend or
            old_lag_analysis != self.lag_correlation_analysis_enabled or
            set(old_timeframes) != set(self.correlation_timeframes)):
            with self._cache_lock:
                self._correlation_cache = {}
                self._lag_correlation_cache = {}
                self._btc_trend_cache = {
                    'trend': None,
                    'strength': 0.0,
                    'timestamp': None,
                    'last_price': 0.0,
                    'data': None
                }
                self._btc_volume_cache = {
                    'normalized_volume': 0.0,
                    'timestamp': None,
                    'is_high_volume': False
                }
            logger.info("کش تحلیل‌گر همبستگی بیت‌کوین به دلیل تغییر پارامترهای مهم پاکسازی شد")

        logger.info("تنظیمات BTCCorrelationAnalyzer به‌روزرسانی شد")
        if self.consider_btc_trend:
            logger.info(f"تحلیل روند بیت‌کوین فعال است با تایم‌فریم {self.btc_trend_timeframe}")
        else:
            logger.info("تحلیل روند بیت‌کوین غیرفعال است")

    async def calculate_multi_timeframe_correlation(self, symbol: str, data_fetcher) -> Dict[str, float]:
        """
        محاسبه همبستگی بین ارز و بیت‌کوین در چندین تایم‌فریم

        Args:
            symbol: نماد ارز
            data_fetcher: وهله DataFetcher برای دریافت داده‌ها

        Returns:
            دیکشنری همبستگی در تایم‌فریم‌های مختلف
        """
        if symbol == self.btc_symbol:
            # اگر نماد خود بیت‌کوین است، همبستگی کامل (1.0) برگردان
            return {tf: 1.0 for tf in self.correlation_timeframes}

        # کش کردن کلید برای بررسی
        cache_key = f"multi_tf_corr_{symbol}_{self.btc_symbol}"

        with self._cache_lock:
            if cache_key in self._correlation_cache:
                cached_corr, timestamp = self._correlation_cache[cache_key]
                age = (datetime.now() - timestamp).total_seconds()
                if age < self.cache_expiry_seconds:
                    return cached_corr

        # محاسبه همبستگی در هر تایم‌فریم
        correlations = {}

        for tf in self.correlation_timeframes:
            try:
                # دریافت داده‌های تاریخی برای ارز و بیت‌کوین
                symbol_data = await data_fetcher.get_historical_data(
                    symbol,
                    tf,
                    limit=self.correlation_period
                )

                btc_data = await data_fetcher.get_historical_data(
                    self.btc_symbol,
                    tf,
                    limit=self.correlation_period
                )

                if symbol_data is None or btc_data is None or symbol_data.empty or btc_data.empty:
                    logger.warning(f"داده‌های ناکافی برای تحلیل همبستگی: {symbol} با {self.btc_symbol} در تایم‌فریم {tf}")
                    correlations[tf] = 0.0
                    continue

                # مشترک‌سازی شاخص‌ها
                common_index = symbol_data.index.intersection(btc_data.index)
                if len(common_index) < 20:  # حداقل 20 نقطه داده
                    logger.debug(
                        f"نقاط داده مشترک ناکافی برای محاسبه همبستگی: {symbol} با {self.btc_symbol} در تایم‌فریم {tf}")
                    correlations[tf] = 0.0
                    continue

                # استفاده از loc برای استخراج داده‌های مشترک
                df1_aligned = symbol_data.loc[common_index]
                df2_aligned = btc_data.loc[common_index]

                # محاسبه تغییرات قیمت
                returns1 = df1_aligned['close'].pct_change().fillna(0)
                returns2 = df2_aligned['close'].pct_change().fillna(0)

                # بررسی سطح تغییرات
                if returns1.std() < 1e-6 or returns2.std() < 1e-6:
                    logger.debug(f"تغییرات قیمت ناچیز برای محاسبه همبستگی: {symbol} یا {self.btc_symbol} در تایم‌فریم {tf}")
                    correlations[tf] = 0.0
                    continue

                # حذف داده‌های ناقص
                mask = ~(np.isnan(returns1.values) | np.isnan(returns2.values))
                valid_data_count = np.sum(mask)

                if valid_data_count < 5:  # حداقل 5 نقطه داده معتبر
                    logger.debug(f"تعداد داده معتبر پس از فیلتر بسیار کم است در تایم‌فریم {tf}: {valid_data_count}")
                    correlations[tf] = 0.0
                    continue

                # محاسبه همبستگی
                correlation = returns1[mask].corr(returns2[mask])

                # مدیریت مقادیر NaN یا Inf
                if pd.isna(correlation) or np.isinf(correlation):
                    correlations[tf] = 0.0
                else:
                    correlations[tf] = correlation

            except Exception as e:
                logger.error(f"خطا در محاسبه همبستگی برای {symbol} با {self.btc_symbol} در تایم‌فریم {tf}: {e}")
                correlations[tf] = 0.0

        # ذخیره در کش
        with self._cache_lock:
            self._correlation_cache[cache_key] = (correlations, datetime.now())

        return correlations

    def calculate_weighted_correlation(self, tf_correlations: Dict[str, float]) -> float:
        """
        محاسبه میانگین وزن‌دار همبستگی‌ها در تایم‌فریم‌های مختلف

        Args:
            tf_correlations: دیکشنری همبستگی در تایم‌فریم‌های مختلف

        Returns:
            میانگین وزن‌دار همبستگی
        """
        total_weight = 0
        weighted_sum = 0

        for i, tf in enumerate(self.correlation_timeframes):
            if tf in tf_correlations:
                # اگر وزن‌ها تعریف شده باشند، از آن‌ها استفاده کن
                weight = self.correlation_timeframe_weights[i] if i < len(self.correlation_timeframe_weights) else 1.0

                weighted_sum += tf_correlations[tf] * weight
                total_weight += weight

        # اگر هیچ همبستگی معتبری وجود نداشت یا مجموع وزن‌ها صفر است
        if total_weight == 0:
            return 0.0

        return weighted_sum / total_weight

    async def calculate_lag_correlations(self, symbol: str, data_fetcher, timeframe: str = None) -> Dict[int, float]:
        """
        محاسبه همبستگی با تاخیر بین ارز و بیت‌کوین

        Args:
            symbol: نماد ارز
            data_fetcher: وهله DataFetcher برای دریافت داده‌ها
            timeframe: تایم‌فریم مورد نظر (اختیاری)

        Returns:
            دیکشنری همبستگی در تاخیرهای مختلف
        """
        if not self.lag_correlation_analysis_enabled:
            return {0: 0.0}

        if symbol == self.btc_symbol:
            # اگر نماد خود بیت‌کوین است، همبستگی کامل (1.0) با تاخیر 0 برگردان
            return {0: 1.0}

        # استفاده از تایم‌فریم پیش‌فرض اگر تعیین نشده باشد
        tf = timeframe or self.primary_correlation_timeframe

        # کش کردن کلید برای بررسی
        cache_key = f"lag_corr_{symbol}_{self.btc_symbol}_{tf}"

        with self._cache_lock:
            if cache_key in self._lag_correlation_cache:
                cached_corr, timestamp = self._lag_correlation_cache[cache_key]
                age = (datetime.now() - timestamp).total_seconds()
                if age < self.cache_expiry_seconds:
                    return cached_corr

        try:
            # دریافت داده‌های تاریخی با تعداد بیشتر برای محاسبه تاخیر
            extended_period = self.correlation_period + self.max_lag_periods

            symbol_data = await data_fetcher.get_historical_data(
                symbol,
                tf,
                limit=extended_period
            )

            btc_data = await data_fetcher.get_historical_data(
                self.btc_symbol,
                tf,
                limit=extended_period
            )

            if symbol_data is None or btc_data is None or symbol_data.empty or btc_data.empty:
                logger.warning(f"داده‌های ناکافی برای تحلیل همبستگی با تاخیر: {symbol} و {self.btc_symbol}")
                return {0: 0.0}

            # محاسبه تغییرات قیمت
            symbol_returns = symbol_data['close'].pct_change().fillna(0)
            btc_returns = btc_data['close'].pct_change().fillna(0)

            # محاسبه همبستگی با تاخیرهای مختلف
            lag_correlations = {}

            # همبستگی بدون تاخیر (lag=0)
            lag_correlations[0] = symbol_returns.corr(btc_returns)

            # همبستگی با تاخیرهای مختلف
            for lag in range(1, self.max_lag_periods + 1):
                # همبستگی با تاخیر مثبت (بیت‌کوین جلوتر)
                lagged_btc = btc_returns.shift(lag)
                lag_correlations[lag] = symbol_returns.corr(lagged_btc)

                # همبستگی با تاخیر منفی (ارز جلوتر)
                lagged_symbol = symbol_returns.shift(lag)
                lag_correlations[-lag] = lagged_symbol.corr(btc_returns)

            # مدیریت مقادیر NaN یا Inf
            for lag in list(lag_correlations.keys()):
                if pd.isna(lag_correlations[lag]) or np.isinf(lag_correlations[lag]):
                    lag_correlations[lag] = 0.0

            # ذخیره در کش
            with self._cache_lock:
                self._lag_correlation_cache[cache_key] = (lag_correlations, datetime.now())

            return lag_correlations

        except Exception as e:
            logger.error(f"خطا در محاسبه همبستگی‌های با تاخیر برای {symbol} و {self.btc_symbol}: {e}")
            return {0: 0.0}

    def find_best_lag_correlation(self, lag_correlations: Dict[int, float]) -> Tuple[int, float]:
        """
        یافتن بهترین همبستگی با تاخیر

        Args:
            lag_correlations: دیکشنری همبستگی در تاخیرهای مختلف

        Returns:
            تاپل (بهترین تاخیر، مقدار همبستگی)
        """
        if not lag_correlations:
            return 0, 0.0

        # یافتن بیشترین همبستگی مطلق
        best_lag = 0
        best_corr = 0.0
        best_abs_corr = 0.0

        for lag, corr in lag_correlations.items():
            abs_corr = abs(corr)
            if abs_corr > best_abs_corr:
                best_abs_corr = abs_corr
                best_corr = corr
                best_lag = lag

        return best_lag, best_corr

    async def analyze_btc_trend(self, data_fetcher) -> Dict[str, Any]:
        """
        تحلیل روند بیت‌کوین

        Args:
            data_fetcher: وهله MarketDataFetcher برای دریافت داده‌ها

        Returns:
            دیکشنری از اطلاعات روند شامل {'trend', 'strength', 'last_price'}
        """
        # بررسی کش روند بیت‌کوین
        current_time = datetime.now()
        if (self._btc_trend_cache['timestamp'] and
                (current_time - self._btc_trend_cache['timestamp']).total_seconds() < self.btc_trend_cache_expiry):
            return {
                'trend': self._btc_trend_cache['trend'],
                'strength': self._btc_trend_cache['strength'],
                'last_price': self._btc_trend_cache['last_price']
            }

        try:
            # دریافت داده‌های بیت‌کوین
            btc_data = await data_fetcher.get_historical_data(
                self.btc_symbol,
                self.btc_trend_timeframe,
                limit=self.btc_trend_period
            )

            if btc_data is None or btc_data.empty or len(btc_data) < self.btc_trend_ema_slow:
                logger.warning(f"داده‌های ناکافی برای تحلیل روند بیت‌کوین: {self.btc_symbol}")
                return {
                    'trend': 'neutral',
                    'strength': 0.0,
                    'last_price': 0.0
                }

            # محاسبه میانگین متحرک نمایی سریع و آهسته
            btc_data['ema_fast'] = btc_data['close'].ewm(span=self.btc_trend_ema_fast, adjust=False).mean()
            btc_data['ema_slow'] = btc_data['close'].ewm(span=self.btc_trend_ema_slow, adjust=False).mean()

            # محاسبه شکاف نسبی بین دو EMA
            btc_data['ema_gap'] = (btc_data['ema_fast'] - btc_data['ema_slow']) / btc_data['ema_slow']

            # قیمت آخر و شکاف EMA
            last_ema_gap = btc_data['ema_gap'].iloc[-1]
            last_price = btc_data['close'].iloc[-1]

            # تعیین روند و قدرت روند
            trend = 'neutral'
            if last_ema_gap > self.btc_trend_strength_threshold:
                trend = 'bullish'
            elif last_ema_gap < -self.btc_trend_strength_threshold:
                trend = 'bearish'

            # محاسبه قدرت روند (مقدار مطلق شکاف EMA)
            trend_strength = abs(last_ema_gap)

            # تحلیل تکمیلی - شیب EMA
            ema_slope = (btc_data['ema_fast'].iloc[-1] - btc_data['ema_fast'].iloc[-5]) / btc_data['ema_fast'].iloc[-5]

            # تقویت قدرت روند بر اساس شیب
            if (trend == 'bullish' and ema_slope > 0) or (trend == 'bearish' and ema_slope < 0):
                trend_strength = trend_strength * (1 + abs(ema_slope))

            # به‌روزرسانی کش
            self._btc_trend_cache = {
                'trend': trend,
                'strength': trend_strength,
                'timestamp': current_time,
                'last_price': last_price,
                'data': btc_data
            }

            logger.debug(f"روند بیت‌کوین تحلیل شد: {trend}، قدرت: {trend_strength:.4f}، قیمت: {last_price}")

            return {
                'trend': trend,
                'strength': trend_strength,
                'last_price': last_price
            }

        except Exception as e:
            logger.error(f"خطا در تحلیل روند بیت‌کوین: {e}")
            return {
                'trend': 'neutral',
                'strength': 0.0,
                'last_price': 0.0
            }

    async def analyze_btc_volume(self, data_fetcher) -> Dict[str, Any]:
        """
        تحلیل حجم معاملات بیت‌کوین

        Args:
            data_fetcher: وهله MarketDataFetcher برای دریافت داده‌ها

        Returns:
            دیکشنری از اطلاعات حجم معاملات
        """
        if not self.btc_volume_analysis_enabled:
            return {
                'normalized_volume': 0.0,
                'is_high_volume': False,
                'volume_factor': 1.0
            }

        # بررسی کش حجم معاملات بیت‌کوین
        current_time = datetime.now()
        if (self._btc_volume_cache['timestamp'] and
                (current_time - self._btc_volume_cache['timestamp']).total_seconds() < self.btc_volume_cache_expiry):
            volume_data = self._btc_volume_cache.copy()
            return {
                'normalized_volume': volume_data['normalized_volume'],
                'is_high_volume': volume_data['is_high_volume'],
                'volume_factor': 1.2 if volume_data['is_high_volume'] else 1.0
            }

        try:
            # دریافت داده‌های حجم بیت‌کوین
            btc_volume_data = await data_fetcher.get_historical_data(
                self.btc_symbol,
                self.btc_volume_timeframe,
                limit=self.btc_volume_period
            )

            if btc_volume_data is None or btc_volume_data.empty or 'volume' not in btc_volume_data.columns:
                logger.warning(f"داده‌های حجم ناکافی برای تحلیل بیت‌کوین: {self.btc_symbol}")
                return {
                    'normalized_volume': 0.0,
                    'is_high_volume': False,
                    'volume_factor': 1.0
                }

            # محاسبه میانگین و انحراف معیار حجم معاملات
            volume_series = btc_volume_data['volume']
            avg_volume = volume_series.rolling(window=self.btc_volume_period).mean().iloc[-1]

            if pd.isna(avg_volume) or avg_volume == 0:
                return {
                    'normalized_volume': 0.0,
                    'is_high_volume': False,
                    'volume_factor': 1.0
                }

            # محاسبه حجم نرمال‌شده (نسبت حجم فعلی به میانگین)
            current_volume = volume_series.iloc[-1]
            normalized_volume = current_volume / avg_volume

            # تعیین وضعیت حجم بالا
            is_high_volume = normalized_volume > self.btc_high_volume_threshold

            # به‌روزرسانی کش
            self._btc_volume_cache = {
                'normalized_volume': normalized_volume,
                'is_high_volume': is_high_volume,
                'timestamp': current_time
            }

            logger.debug(f"حجم بیت‌کوین تحلیل شد: نرمال شده={normalized_volume:.2f}، حجم بالا={is_high_volume}")

            return {
                'normalized_volume': normalized_volume,
                'is_high_volume': is_high_volume,
                'volume_factor': 1.2 if is_high_volume else 1.0  # ضریب تاثیر حجم بالا
            }

        except Exception as e:
            logger.error(f"خطا در تحلیل حجم بیت‌کوین: {e}")
            return {
                'normalized_volume': 0.0,
                'is_high_volume': False,
                'volume_factor': 1.0
            }

    async def get_correlation_summary(self, symbol: str, direction: str, data_fetcher) -> Dict[str, Any]:
        """
        دریافت خلاصه تحلیل همبستگی برای یک ارز

        Args:
            symbol: نماد ارز
            direction: جهت معامله ('long' یا 'short')
            data_fetcher: وهله DataFetcher برای دریافت داده‌ها

        Returns:
            دیکشنری خلاصه تحلیل همبستگی
        """
        # اگر خود بیت‌کوین باشد، همبستگی کامل و خنثی برگردان
        if symbol == self.btc_symbol:
            return {
                'correlation': 1.0,
                'correlation_type': 'self',
                'weighted_correlation': 1.0,
                'best_lag': 0,
                'best_lag_correlation': 1.0,
                'btc_trend': 'neutral',
                'btc_trend_strength': 0.0,
                'is_high_volume': False,
                'correlation_score': 0,
                'compatibility_score': 100
            }

        try:
            # گام 1: تحلیل روند بیت‌کوین
            btc_trend_info = await self.analyze_btc_trend(data_fetcher)

            # گام 2: محاسبه همبستگی چندتایم‌فریمی
            tf_correlations = await self.calculate_multi_timeframe_correlation(symbol, data_fetcher)

            # گام 3: محاسبه همبستگی وزن‌دار
            weighted_correlation = self.calculate_weighted_correlation(tf_correlations)

            # گام 4: محاسبه همبستگی با تاخیر
            lag_correlations = await self.calculate_lag_correlations(symbol, data_fetcher)
            best_lag, best_lag_correlation = self.find_best_lag_correlation(lag_correlations)

            # گام 5: تحلیل حجم معاملات بیت‌کوین
            btc_volume_info = await self.analyze_btc_volume(data_fetcher)

            # تعیین نوع همبستگی
            correlation_type = 'unknown'
            if weighted_correlation <= self.inverse_correlation_threshold:
                correlation_type = 'inverse'
            elif abs(weighted_correlation) <= self.zero_correlation_threshold:
                correlation_type = 'zero'
            else:
                correlation_type = 'positive'

            # محاسبه امتیاز همبستگی
            correlation_score = self._calculate_correlation_score(
                correlation_type=correlation_type,
                weighted_correlation=weighted_correlation,
                best_lag_correlation=best_lag_correlation,
                btc_trend=btc_trend_info['trend'],
                btc_trend_strength=btc_trend_info['strength'],
                direction=direction,
                is_high_volume=btc_volume_info['is_high_volume']
            )

            # محاسبه امتیاز سازگاری
            compatibility_score = 100 + correlation_score
            compatibility_score = max(0, min(100, compatibility_score))

            return {
                'correlation': weighted_correlation,
                'correlation_type': correlation_type,
                'tf_correlations': tf_correlations,
                'weighted_correlation': weighted_correlation,
                'best_lag': best_lag,
                'best_lag_correlation': best_lag_correlation,
                'btc_trend': btc_trend_info['trend'],
                'btc_trend_strength': btc_trend_info['strength'],
                'btc_last_price': btc_trend_info['last_price'],
                'is_high_volume': btc_volume_info['is_high_volume'],
                'normalized_volume': btc_volume_info['normalized_volume'],
                'correlation_score': correlation_score,
                'compatibility_score': compatibility_score
            }

        except Exception as e:
            logger.error(f"خطا در خلاصه همبستگی برای {symbol}: {e}")
            return {
                'correlation': 0.0,
                'correlation_type': 'error',
                'weighted_correlation': 0.0,
                'best_lag': 0,
                'best_lag_correlation': 0.0,
                'btc_trend': 'neutral',
                'btc_trend_strength': 0.0,
                'is_high_volume': False,
                'correlation_score': 0,
                'compatibility_score': 50
            }

    def _calculate_correlation_score(self, correlation_type: str, weighted_correlation: float,
                                     best_lag_correlation: float, btc_trend: str, btc_trend_strength: float,
                                     direction: str, is_high_volume: bool) -> int:
        """
        محاسبه امتیاز همبستگی برای تعدیل کیفیت سیگنال

        Args:
            correlation_type: نوع همبستگی ('positive', 'zero', 'inverse')
            weighted_correlation: همبستگی وزن‌دار
            best_lag_correlation: بهترین همبستگی با تاخیر
            btc_trend: روند بیت‌کوین ('bullish', 'bearish', 'neutral')
            btc_trend_strength: قدرت روند بیت‌کوین
            direction: جهت معامله ('long', 'short')
            is_high_volume: آیا حجم معاملات بیت‌کوین بالاست

        Returns:
            امتیاز همبستگی (-50 تا 50)
        """
        score = 0

        # 1. امتیاز بر اساس نوع همبستگی
        if correlation_type == 'positive':
            base_score = 10  # همبستگی مثبت با بیت‌کوین - امتیاز پایه مثبت
        elif correlation_type == 'zero':
            base_score = 0  # بدون همبستگی - امتیاز خنثی
        else:  # 'inverse'
            base_score = 5  # همبستگی معکوس - امتیاز کمی مثبت

        score += base_score

        # 2. بررسی سازگاری روند بیت‌کوین با جهت معامله
        if correlation_type == 'positive':
            if btc_trend == 'bullish' and direction == 'long':
                # معامله خرید با روند صعودی بیت‌کوین و همبستگی مثبت
                trend_bonus = 15 * btc_trend_strength  # بونوس متناسب با قدرت روند
                score += min(25, trend_bonus)
            elif btc_trend == 'bearish' and direction == 'short':
                # معامله فروش با روند نزولی بیت‌کوین و همبستگی مثبت
                trend_bonus = 15 * btc_trend_strength
                score += min(25, trend_bonus)
            elif btc_trend == 'bullish' and direction == 'short':
                # معامله فروش بر خلاف روند صعودی بیت‌کوین با همبستگی مثبت
                trend_penalty = 20 * btc_trend_strength
                score -= min(35, trend_penalty)
            elif btc_trend == 'bearish' and direction == 'long':
                # معامله خرید بر خلاف روند نزولی بیت‌کوین با همبستگی مثبت
                trend_penalty = 20 * btc_trend_strength
                score -= min(35, trend_penalty)

        # 3. امتیاز برای همبستگی معکوس (در جهت مخالف بیت‌کوین)
        elif correlation_type == 'inverse':
            if btc_trend == 'bullish' and direction == 'short':
                # معامله فروش در روند صعودی بیت‌کوین با همبستگی معکوس
                score += 15
            elif btc_trend == 'bearish' and direction == 'long':
                # معامله خرید در روند نزولی بیت‌کوین با همبستگی معکوس
                score += 15

        # 4. تعدیل براساس قدرت همبستگی
        correlation_strength = abs(weighted_correlation)
        if correlation_type == 'positive':
            score += int(correlation_strength * 10)  # تا 10 امتیاز برای همبستگی قوی
        elif correlation_type == 'inverse':
            score += int(correlation_strength * 5)  # تا 5 امتیاز برای همبستگی معکوس قوی

        # 5. بونوس برای همبستگی با تاخیر معنادار
        if abs(best_lag_correlation) > correlation_strength + 0.1:
            # اگر همبستگی با تاخیر قوی‌تر است
            lag_bonus = int((abs(best_lag_correlation) - correlation_strength) * 15)
            score += min(10, lag_bonus)

        # 6. تعدیل برای حجم بالای بیت‌کوین
        if is_high_volume:
            # در حجم بالا، تاثیر همبستگی مثبت بیشتر است
            if correlation_type == 'positive':
                if (btc_trend == 'bullish' and direction == 'long') or (
                        btc_trend == 'bearish' and direction == 'short'):
                    score += 10  # بونوس برای معامله همسو با روند در حجم بالا
                elif (btc_trend == 'bullish' and direction == 'short') or (
                        btc_trend == 'bearish' and direction == 'long'):
                    score -= 15  # جریمه برای معامله خلاف روند در حجم بالا

        # محدود کردن امتیاز نهایی بین -50 تا 50
        return max(-50, min(50, score))


class DynamicStopManager:
    """
    مدیریت پیشرفته حد ضرر دینامیک با چندین استراتژی
    """

    def __init__(self, config: Dict[str, Any]):
        """
        مقداردهی اولیه

        Args:
            config: دیکشنری تنظیمات
        """
        self.config = config
        self.risk_config = config.get('risk_management', {})

        # تنظیمات استاپ متحرک
        self.use_trailing_stop = self.risk_config.get('use_trailing_stop', True)
        self.trailing_activation_percent = self.risk_config.get('trailing_stop_activation_percent', 3.0)
        self.trailing_distance_percent = self.risk_config.get('trailing_stop_distance_percent', 2.25)

        # تنظیمات استاپ مبتنی بر ATR
        self.use_atr_trailing = self.risk_config.get('use_atr_based_trailing', False)
        self.atr_multiplier = self.risk_config.get('atr_trailing_multiplier', 2.0)

        # تنظیمات استاپ مبتنی بر ساختار قیمت
        self.use_structure_based_stops = self.risk_config.get('use_structure_based_stops', False)
        self.swing_detection_lookback = self.risk_config.get('swing_detection_lookback', 10)

        # تنظیمات استاپ زمانی
        self.use_time_based_stops = self.risk_config.get('use_time_based_stops', False)
        self.max_trade_duration_hours = self.risk_config.get('max_trade_duration_hours', 48)

        # تنظیمات استاپ مبتنی بر عملکرد
        self.use_performance_based_stops = self.risk_config.get('use_performance_based_stops', False)
        self.profit_lock_activation = self.risk_config.get('profit_lock_activation_percent', 2.0)
        self.profit_lock_percent = self.risk_config.get('profit_lock_percent', 75)  # قفل 75% سود

        # استاپ چند مرحله‌ای
        self.use_multi_stage_trailing = self.risk_config.get('use_multi_stage_trailing', False)
        self.trailing_stages = self.risk_config.get('trailing_stages', [
            {'activation': 1.0, 'distance': 0.8},  # مرحله 1
            {'activation': 2.0, 'distance': 0.6},  # مرحله 2
            {'activation': 3.5, 'distance': 0.4}  # مرحله 3
        ])

        # استاپ مبتنی بر حجم
        self.use_volume_based_stops = self.risk_config.get('use_volume_based_stops', False)
        self.volume_spike_factor = self.risk_config.get('volume_spike_factor', 3.0)
        self.volume_lookback = self.risk_config.get('volume_lookback', 20)

        logger.info(f"DynamicStopManager با {self._get_active_strategies()} استراتژی راه‌اندازی شد")

    def update_config(self, new_config: Dict[str, Any]):
        """
        به‌روزرسانی تنظیمات در زمان اجرا

        Args:
            new_config: دیکشنری تنظیمات جدید
        """
        old_config = self.config.copy()

        # به‌روزرسانی تنظیمات اصلی
        self.config = new_config

        # به‌روزرسانی تنظیمات ریسک
        old_risk_config = self.risk_config.copy()
        self.risk_config = new_config.get('risk_management', {})

        # ذخیره تنظیمات قبلی برای مقایسه
        old_use_trailing_stop = self.use_trailing_stop
        old_trailing_activation_percent = self.trailing_activation_percent
        old_trailing_distance_percent = self.trailing_distance_percent
        old_use_atr_trailing = self.use_atr_trailing
        old_use_structure_based_stops = self.use_structure_based_stops
        old_use_time_based_stops = self.use_time_based_stops
        old_use_multi_stage_trailing = self.use_multi_stage_trailing
        old_use_volume_based_stops = self.use_volume_based_stops

        # به‌روزرسانی تنظیمات استاپ متحرک
        self.use_trailing_stop = self.risk_config.get('use_trailing_stop', True)
        self.trailing_activation_percent = self.risk_config.get('trailing_stop_activation_percent', 3.0)
        self.trailing_distance_percent = self.risk_config.get('trailing_stop_distance_percent', 2.25)

        # به‌روزرسانی تنظیمات استاپ مبتنی بر ATR
        self.use_atr_trailing = self.risk_config.get('use_atr_based_trailing', False)
        self.atr_multiplier = self.risk_config.get('atr_trailing_multiplier', 2.0)

        # به‌روزرسانی تنظیمات استاپ مبتنی بر ساختار قیمت
        self.use_structure_based_stops = self.risk_config.get('use_structure_based_stops', False)
        self.swing_detection_lookback = self.risk_config.get('swing_detection_lookback', 10)

        # به‌روزرسانی تنظیمات استاپ زمانی
        self.use_time_based_stops = self.risk_config.get('use_time_based_stops', False)
        self.max_trade_duration_hours = self.risk_config.get('max_trade_duration_hours', 48)

        # به‌روزرسانی تنظیمات استاپ مبتنی بر عملکرد
        self.use_performance_based_stops = self.risk_config.get('use_performance_based_stops', False)
        self.profit_lock_activation = self.risk_config.get('profit_lock_activation_percent', 2.0)
        self.profit_lock_percent = self.risk_config.get('profit_lock_percent', 75)

        # به‌روزرسانی استاپ چند مرحله‌ای
        self.use_multi_stage_trailing = self.risk_config.get('use_multi_stage_trailing', False)
        self.trailing_stages = self.risk_config.get('trailing_stages', [
            {'activation': 1.0, 'distance': 0.8},
            {'activation': 2.0, 'distance': 0.6},
            {'activation': 3.5, 'distance': 0.4}
        ])

        # به‌روزرسانی استاپ مبتنی بر حجم
        self.use_volume_based_stops = self.risk_config.get('use_volume_based_stops', False)
        self.volume_spike_factor = self.risk_config.get('volume_spike_factor', 3.0)
        self.volume_lookback = self.risk_config.get('volume_lookback', 20)

        # لاگ تغییرات مهم
        changes = []
        if old_use_trailing_stop != self.use_trailing_stop:
            changes.append(f"استاپ متحرک: {'فعال' if self.use_trailing_stop else 'غیرفعال'}")

        if old_trailing_activation_percent != self.trailing_activation_percent:
            changes.append(
                f"درصد فعال‌سازی استاپ متحرک: {old_trailing_activation_percent} → {self.trailing_activation_percent}")

        if old_trailing_distance_percent != self.trailing_distance_percent:
            changes.append(
                f"درصد فاصله استاپ متحرک: {old_trailing_distance_percent} → {self.trailing_distance_percent}")

        if old_use_atr_trailing != self.use_atr_trailing:
            changes.append(f"استاپ مبتنی بر ATR: {'فعال' if self.use_atr_trailing else 'غیرفعال'}")

        if old_use_structure_based_stops != self.use_structure_based_stops:
            changes.append(f"استاپ مبتنی بر ساختار: {'فعال' if self.use_structure_based_stops else 'غیرفعال'}")

        if old_use_time_based_stops != self.use_time_based_stops:
            changes.append(f"استاپ زمان‌بندی شده: {'فعال' if self.use_time_based_stops else 'غیرفعال'}")

        if old_use_multi_stage_trailing != self.use_multi_stage_trailing:
            changes.append(f"استاپ چند مرحله‌ای: {'فعال' if self.use_multi_stage_trailing else 'غیرفعال'}")

        if old_use_volume_based_stops != self.use_volume_based_stops:
            changes.append(f"استاپ مبتنی بر حجم: {'فعال' if self.use_volume_based_stops else 'غیرفعال'}")

        if changes:
            logger.info(f"تنظیمات DynamicStopManager به‌روزرسانی شد: {', '.join(changes)}")
            logger.info(f"استراتژی‌های فعال: {self._get_active_strategies()}")
        else:
            logger.debug("تنظیمات DynamicStopManager به‌روزرسانی شد (بدون تغییر در پارامترهای کلیدی)")

    def _get_active_strategies(self) -> str:
        """
        استخراج استراتژی‌های فعال

        Returns:
            رشته توصیفی استراتژی‌های فعال
        """
        strategies = []
        if self.use_trailing_stop:
            strategies.append("trailing")
        if self.use_atr_trailing:
            strategies.append("ATR-based")
        if self.use_structure_based_stops:
            strategies.append("structure-based")
        if self.use_time_based_stops:
            strategies.append("time-based")
        if self.use_performance_based_stops:
            strategies.append("performance-based")
        if self.use_multi_stage_trailing:
            strategies.append("multi-stage")
        if self.use_volume_based_stops:
            strategies.append("volume-based")

        return ", ".join(strategies) if strategies else "none"

    def update_trailing_stop(self, trade: Trade, current_price: float,
                             atr: float = 0, volume_data: Optional[pd.DataFrame] = None) -> float:
        """
        به‌روزرسانی استاپ متحرک با استراتژی‌های پیشرفته

        Args:
            trade: معامله فعال
            current_price: قیمت فعلی
            atr: مقدار ATR فعلی (اختیاری)
            volume_data: دیتافریم اطلاعات حجم (اختیاری)

        Returns:
            قیمت استاپ جدید
        """
        if not trade or not trade.stop_loss:
            return 0

        # اگر هیچ استراتژی فعال نباشد، استاپ اصلی را برگردان
        if not (self.use_trailing_stop or self.use_atr_trailing or
                self.use_structure_based_stops or self.use_multi_stage_trailing or
                self.use_performance_based_stops):
            return trade.stop_loss

        # محاسبه تغییر قیمت از ورود (به درصد)
        price_change_percent = 0
        if trade.direction == 'long':
            price_change_percent = ((current_price - trade.entry_price) / trade.entry_price) * 100
        else:  # 'short'
            price_change_percent = ((trade.entry_price - current_price) / trade.entry_price) * 100

        # استاپ اولیه
        current_stop = trade.stop_loss
        new_stop = current_stop

        # بررسی استراتژی‌های مختلف و انتخاب بهترین استاپ
        stop_candidates = []

        # 1. استاپ متحرک چند مرحله‌ای
        if self.use_multi_stage_trailing and price_change_percent > 0:
            multi_stage_stop = self._calculate_multi_stage_stop(trade, current_price, price_change_percent)
            if multi_stage_stop != current_stop:
                stop_candidates.append(('multi_stage', multi_stage_stop))

        # 2. استاپ متحرک استاندارد
        elif self.use_trailing_stop and price_change_percent >= self.trailing_activation_percent:
            trailing_stop = self._calculate_trailing_stop(trade, current_price)
            if trailing_stop != current_stop:
                stop_candidates.append(('trailing', trailing_stop))

        # 3. استاپ مبتنی بر ATR
        if self.use_atr_trailing and atr > 0 and price_change_percent > 0:
            atr_stop = self._calculate_atr_stop(trade, current_price, atr)
            if atr_stop != current_stop:
                stop_candidates.append(('atr', atr_stop))

        # 4. استاپ مبتنی بر عملکرد
        if self.use_performance_based_stops and price_change_percent >= self.profit_lock_activation:
            performance_stop = self._calculate_performance_stop(trade, current_price, price_change_percent)
            if performance_stop != current_stop:
                stop_candidates.append(('performance', performance_stop))

        # 5. استاپ مبتنی بر حجم (به صورت شرطی)
        if self.use_volume_based_stops and volume_data is not None and not volume_data.empty:
            # فقط زمانی که افزایش حجم قابل توجه است بررسی شود
            volume_stop = self._calculate_volume_based_stop(trade, current_price, volume_data)
            if volume_stop != current_stop:
                stop_candidates.append(('volume', volume_stop))

        # انتخاب بهترین استاپ با توجه به جهت معامله
        if stop_candidates:
            if trade.direction == 'long':
                # برای معاملات خرید، بزرگترین استاپ انتخاب می‌شود
                best_stop = max(stop_candidates, key=lambda x: x[1])
                if best_stop[1] > current_stop:
                    new_stop = best_stop[1]
                    logger.debug(
                        f"استاپ معامله {trade.trade_id} از {current_stop} به {new_stop} با استراتژی {best_stop[0]} به‌روزرسانی شد")
            else:
                # برای معاملات فروش، کوچکترین استاپ انتخاب می‌شود
                best_stop = min(stop_candidates, key=lambda x: x[1])
                if best_stop[1] < current_stop:
                    new_stop = best_stop[1]
                    logger.debug(
                        f"استاپ معامله {trade.trade_id} از {current_stop} به {new_stop} با استراتژی {best_stop[0]} به‌روزرسانی شد")

        return new_stop

    def _calculate_trailing_stop(self, trade: Trade, current_price: float) -> float:
        """
        محاسبه استاپ متحرک استاندارد

        Args:
            trade: معامله فعال
            current_price: قیمت فعلی

        Returns:
            قیمت استاپ جدید
        """
        if trade.direction == 'long':
            new_stop = current_price * (1 - (self.trailing_distance_percent / 100))
            return max(new_stop, trade.stop_loss)
        else:
            new_stop = current_price * (1 + (self.trailing_distance_percent / 100))
            return min(new_stop, trade.stop_loss)

    def _calculate_atr_stop(self, trade: Trade, current_price: float, atr: float) -> float:
        """
        محاسبه استاپ مبتنی بر ATR

        Args:
            trade: معامله فعال
            current_price: قیمت فعلی
            atr: مقدار ATR فعلی

        Returns:
            قیمت استاپ جدید
        """
        if trade.direction == 'long':
            new_stop = current_price - (atr * self.atr_multiplier)
            return max(new_stop, trade.stop_loss)
        else:
            new_stop = current_price + (atr * self.atr_multiplier)
            return min(new_stop, trade.stop_loss)

    def _calculate_performance_stop(self, trade: Trade, current_price: float, price_change_percent: float) -> float:
        """
        محاسبه استاپ مبتنی بر عملکرد (قفل سود)

        Args:
            trade: معامله فعال
            current_price: قیمت فعلی
            price_change_percent: تغییر قیمت به درصد

        Returns:
            قیمت استاپ جدید
        """
        # قفل بخشی از سود
        if trade.direction == 'long':
            # محاسبه مقدار سود
            profit_amount = current_price - trade.entry_price
            # استاپ در سطحی که درصد مشخصی از سود را قفل می‌کند
            lock_amount = profit_amount * (self.profit_lock_percent / 100)
            new_stop = trade.entry_price + lock_amount
            return max(new_stop, trade.stop_loss)
        else:
            # محاسبه مقدار سود
            profit_amount = trade.entry_price - current_price
            # استاپ در سطحی که درصد مشخصی از سود را قفل می‌کند
            lock_amount = profit_amount * (self.profit_lock_percent / 100)
            new_stop = trade.entry_price - lock_amount
            return min(new_stop, trade.stop_loss)

    def _calculate_multi_stage_stop(self, trade: Trade, current_price: float, price_change_percent: float) -> float:
        """
        محاسبه استاپ چند مرحله‌ای

        Args:
            trade: معامله فعال
            current_price: قیمت فعلی
            price_change_percent: تغییر قیمت به درصد

        Returns:
            قیمت استاپ جدید
        """
        # تعیین مرحله فعلی بر اساس تغییر قیمت
        current_stage = None
        for stage in sorted(self.trailing_stages, key=lambda x: x['activation'], reverse=True):
            if price_change_percent >= stage['activation']:
                current_stage = stage
                break

        if not current_stage:
            return trade.stop_loss

        # محاسبه استاپ بر اساس مرحله فعلی
        if trade.direction == 'long':
            new_stop = current_price * (1 - (current_stage['distance'] / 100))
            return max(new_stop, trade.stop_loss)
        else:
            new_stop = current_price * (1 + (current_stage['distance'] / 100))
            return min(new_stop, trade.stop_loss)

    def _calculate_volume_based_stop(self, trade: Trade, current_price: float, volume_data: pd.DataFrame) -> float:
        """
        محاسبه استاپ مبتنی بر حجم

        Args:
            trade: معامله فعال
            current_price: قیمت فعلی
            volume_data: دیتافریم اطلاعات حجم

        Returns:
            قیمت استاپ جدید
        """
        try:
            # بررسی داده‌های حجم
            if 'volume' not in volume_data.columns or len(volume_data) < self.volume_lookback:
                return trade.stop_loss

            # محاسبه میانگین حجم
            recent_volumes = volume_data['volume'].tail(self.volume_lookback)
            avg_volume = recent_volumes.mean()
            latest_volume = recent_volumes.iloc[-1]

            # بررسی افزایش حجم
            if latest_volume <= avg_volume * self.volume_spike_factor:
                return trade.stop_loss  # عدم افزایش قابل توجه حجم

            # در صورت افزایش حجم، استاپ نزدیک‌تر به قیمت فعلی
            if trade.direction == 'long':
                # استفاده از فاصله کمتر برای استاپ در افزایش حجم
                tight_distance = self.trailing_distance_percent * 0.6
                new_stop = current_price * (1 - (tight_distance / 100))
                return max(new_stop, trade.stop_loss)
            else:
                tight_distance = self.trailing_distance_percent * 0.6
                new_stop = current_price * (1 + (tight_distance / 100))
                return min(new_stop, trade.stop_loss)

        except Exception as e:
            logger.warning(f"خطا در محاسبه استاپ مبتنی بر حجم: {e}")
            return trade.stop_loss

    def check_time_based_exit(self, trade: Trade) -> bool:
        """
        بررسی شرایط خروج مبتنی بر زمان

        Args:
            trade: معامله فعال

        Returns:
            آیا باید از معامله خارج شود
        """
        if not self.use_time_based_stops or trade.status != 'open':
            return False

        # محاسبه مدت زمان معامله
        current_time = datetime.now()
        if not hasattr(trade, 'timestamp') or not trade.timestamp:
            return False

        # تبدیل timestamp به datetime اگر رشته است
        entry_time = trade.timestamp
        if isinstance(entry_time, str):
            try:
                entry_time = datetime.fromisoformat(entry_time)
            except (ValueError, TypeError):
                logger.warning(f"فرمت timestamp نامعتبر برای معامله {trade.trade_id}")
                return False

        trade_duration = current_time - entry_time
        trade_hours = trade_duration.total_seconds() / 3600

        # بررسی اتمام زمان معامله
        if trade_hours > self.max_trade_duration_hours:
            logger.info(f"خروج مبتنی بر زمان برای معامله {trade.trade_id}: "
                        f"مدت {trade_hours:.2f} ساعت > {self.max_trade_duration_hours} ساعت")
            return True

        return False

    def check_structure_based_exit(self, trade: Trade, price_data: pd.DataFrame) -> bool:
        """
        بررسی شرایط خروج مبتنی بر ساختار قیمت

        Args:
            trade: معامله فعال
            price_data: دیتافریم قیمت

        Returns:
            آیا باید از معامله خارج شود
        """
        if not self.use_structure_based_stops or trade.status != 'open' or price_data.empty:
            return False

        try:
            # محاسبه قله‌ها و دره‌ها با الگوریتم بهینه‌تر
            recent_data = price_data.tail(self.swing_detection_lookback)

            # استفاده از الگوریتم پیشرفته تشخیص نقاط چرخش
            pivots = self._detect_pivots(recent_data)

            if trade.direction == 'long':
                # معاملات خرید - بررسی شکست آخرین دره
                if len(pivots['lows']) >= 2:
                    previous_low = pivots['lows'][-2]['value']
                    current_low = recent_data['low'].iloc[-1]

                    if current_low < previous_low:
                        logger.info(f"خروج مبتنی بر ساختار برای معامله {trade.trade_id} (خرید): "
                                    f"کف فعلی {current_low} کف قبلی {previous_low} را شکست")
                        return True

            else:  # 'short'
                # معاملات فروش - بررسی شکست آخرین قله
                if len(pivots['highs']) >= 2:
                    previous_high = pivots['highs'][-2]['value']
                    current_high = recent_data['high'].iloc[-1]

                    if current_high > previous_high:
                        logger.info(f"خروج مبتنی بر ساختار برای معامله {trade.trade_id} (فروش): "
                                    f"سقف فعلی {current_high} سقف قبلی {previous_high} را شکست")
                        return True

            return False

        except Exception as e:
            logger.error(f"خطا در بررسی خروج مبتنی بر ساختار: {e}")
            return False

    def _detect_pivots(self, df: pd.DataFrame) -> Dict[str, List[Dict[str, Any]]]:
        """
        تشخیص نقاط چرخش (قله‌ها و دره‌ها) در داده‌های قیمت

        Args:
            df: دیتافریم قیمت

        Returns:
            دیکشنری شامل قله‌ها و دره‌ها
        """
        result = {
            'highs': [],
            'lows': []
        }

        if len(df) < 5:  # نیاز به حداقل 5 نقطه
            return result

        try:
            # استفاده از الگوریتم پنجره متحرک برای تشخیص نقاط چرخش
            window_size = 2  # تعداد نقاط در هر طرف

            for i in range(window_size, len(df) - window_size):
                # بررسی قله‌ها
                is_peak = True
                for j in range(1, window_size + 1):
                    if df['high'].iloc[i] <= df['high'].iloc[i - j] or df['high'].iloc[i] <= df['high'].iloc[i + j]:
                        is_peak = False
                        break

                if is_peak:
                    result['highs'].append({
                        'index': df.index[i],
                        'position': i,
                        'value': df['high'].iloc[i]
                    })

                # بررسی دره‌ها
                is_valley = True
                for j in range(1, window_size + 1):
                    if df['low'].iloc[i] >= df['low'].iloc[i - j] or df['low'].iloc[i] >= df['low'].iloc[i + j]:
                        is_valley = False
                        break

                if is_valley:
                    result['lows'].append({
                        'index': df.index[i],
                        'position': i,
                        'value': df['low'].iloc[i]
                    })

            return result

        except Exception as e:
            logger.error(f"خطا در تشخیص نقاط چرخش قیمت: {e}")
            return result

    def check_volume_based_exit(self, trade: Trade, price_data: pd.DataFrame) -> bool:
        """
        بررسی شرایط خروج مبتنی بر حجم

        Args:
            trade: معامله فعال
            price_data: دیتافریم قیمت با اطلاعات حجم

        Returns:
            آیا باید از معامله خارج شود
        """
        if not self.use_volume_based_stops or trade.status != 'open' or price_data.empty:
            return False

        if 'volume' not in price_data.columns:
            return False

        try:
            # بررسی افزایش حجم همراه با حرکت قیمت در جهت مخالف معامله
            recent_data = price_data.tail(min(self.volume_lookback, len(price_data)))

            if len(recent_data) < 3:  # نیاز به حداقل 3 کندل
                return False

            avg_volume = recent_data['volume'].iloc[:-1].mean()
            latest_volume = recent_data['volume'].iloc[-1]
            volume_spike = latest_volume > (avg_volume * self.volume_spike_factor)

            if not volume_spike:
                return False

            # بررسی حرکت قیمت در جهت مخالف معامله
            if trade.direction == 'long':
                # برای معاملات خرید، کاهش قیمت با حجم بالا
                price_drop = recent_data['close'].iloc[-1] < recent_data['open'].iloc[-1]
                if volume_spike and price_drop:
                    logger.info(f"خروج مبتنی بر حجم برای معامله {trade.trade_id} (خرید): "
                                f"افزایش حجم ({latest_volume:.0f} > {avg_volume:.0f} * {self.volume_spike_factor}) با کاهش قیمت")
                    return True
            else:
                # برای معاملات فروش، افزایش قیمت با حجم بالا
                price_rise = recent_data['close'].iloc[-1] > recent_data['open'].iloc[-1]
                if volume_spike and price_rise:
                    logger.info(f"خروج مبتنی بر حجم برای معامله {trade.trade_id} (فروش): "
                                f"افزایش حجم ({latest_volume:.0f} > {avg_volume:.0f} * {self.volume_spike_factor}) با افزایش قیمت")
                    return True

            return False

        except Exception as e:
            logger.error(f"خطا در بررسی خروج مبتنی بر حجم: {e}")
            return False

    def get_exit_strategy(self, trade: Trade, price_data: pd.DataFrame) -> Dict[str, Any]:
        """
        محاسبه و پیشنهاد استراتژی خروج برای یک معامله

        Args:
            trade: معامله فعال
            price_data: دیتافریم قیمت

        Returns:
            دیکشنری استراتژی‌های خروج
        """
        if not trade or trade.status != 'open' or price_data.empty:
            return {'status': 'invalid_input'}

        current_price = price_data['close'].iloc[-1] if not price_data.empty else 0
        if current_price <= 0:
            return {'status': 'invalid_price'}

        # محاسبه ATR
        atr = 0
        if len(price_data) >= 14:
            try:
                # محاسبه ATR ساده
                high_low = price_data['high'] - price_data['low']
                high_close = np.abs(price_data['high'] - price_data['close'].shift(1))
                low_close = np.abs(price_data['low'] - price_data['close'].shift(1))

                ranges = pd.concat([high_low, high_close, low_close], axis=1)
                true_range = ranges.max(axis=1)
                atr = true_range.rolling(14).mean().iloc[-1]
            except Exception as e:
                logger.warning(f"خطا در محاسبه ATR: {e}")

        # بررسی شرایط خروج
        time_exit = self.check_time_based_exit(trade)
        structure_exit = self.check_structure_based_exit(trade, price_data)
        volume_exit = self.check_volume_based_exit(trade, price_data)

        # محاسبه استاپ جدید
        new_stop = self.update_trailing_stop(
            trade=trade,
            current_price=current_price,
            atr=atr,
            volume_data=price_data
        )

        # محاسبه سود/زیان فعلی
        current_profit = 0
        if trade.direction == 'long':
            current_profit = ((current_price - trade.entry_price) / trade.entry_price) * 100
        else:
            current_profit = ((trade.entry_price - current_price) / trade.entry_price) * 100

        return {
            'status': 'success',
            'current_price': current_price,
            'current_profit_percent': current_profit,
            'suggested_stop': new_stop,
            'initial_stop': trade.stop_loss,
            'stop_moved': abs(new_stop - trade.stop_loss) > 0.0001,
            'exit_signals': {
                'time_exit': time_exit,
                'structure_exit': structure_exit,
                'volume_exit': volume_exit
            },
            'any_exit_signal': time_exit or structure_exit or volume_exit,
            'atr': atr
        }


class PositionSizeOptimizer:
    """
    محاسبه‌گر اندازه بهینه پوزیشن با استفاده از الگوریتم‌های پیشرفته مدیریت سرمایه
    """

    def __init__(self, config: Dict[str, Any]):
        """
        مقداردهی اولیه با تنظیمات

        Args:
            config: دیکشنری تنظیمات
        """
        self.config = config
        self.risk_config = config.get('risk_management', {})

        # تنظیمات پایه مدیریت ریسک
        self.account_balance = self.risk_config.get('account_balance', 10000.0)
        self.base_risk_percent = self.risk_config.get('max_risk_per_trade_percent', 1.5)
        self.max_position_size = self.risk_config.get('max_position_size', 200)
        self.fixed_position_size = self.risk_config.get('fixed_position_size', 200)

        # تنظیمات مدیریت سرمایه پویا
        self.use_dynamic_position_sizing = self.risk_config.get('use_dynamic_position_sizing', True)
        self.drawdown_protection = self.risk_config.get('drawdown_protection', True)
        self.winning_streak_factor = self.risk_config.get('winning_streak_factor', 1.1)
        self.losing_streak_factor = self.risk_config.get('losing_streak_factor', 0.9)
        self.max_risk_increase = self.risk_config.get('max_risk_increase', 1.5)
        self.kelly_fraction = self.risk_config.get('kelly_fraction', 0.5)

        # تنظیمات پیشرفته جدید
        self.volatility_adjustment = self.risk_config.get('volatility_adjustment', True)
        self.market_condition_adjustment = self.risk_config.get('market_condition_adjustment', True)
        self.max_risk_decrease = self.risk_config.get('max_risk_decrease', 0.5)
        self.use_equity_curve_based_sizing = self.risk_config.get('use_equity_curve_based_sizing', False)
        self.equity_curve_window = self.risk_config.get('equity_curve_window', 20)

        # آمار متغیر
        self._stats = {
            'current_drawdown': 0.0,
            'win_rate': 0.5,  # مقدار پیش‌فرض
            'avg_win_percent': 2.0,  # مقدار پیش‌فرض
            'avg_loss_percent': 1.0,  # مقدار پیش‌فرض
            'consecutive_wins': 0,
            'consecutive_losses': 0,
            'equity_curve': []  # حفظ منحنی سرمایه اخیر
        }

        # قفل برای به‌روزرسانی همزمان
        self._lock = threading.RLock()

        logger.info("PositionSizeOptimizer با استراتژی %s راه‌اندازی شد",
                    "پویا" if self.use_dynamic_position_sizing else "ثابت")

    def update_config(self, new_config: Dict[str, Any]):
        """
        به‌روزرسانی تنظیمات در زمان اجرا

        Args:
            new_config: دیکشنری تنظیمات جدید
        """
        with self._lock:
            old_config = self.config.copy()
            old_risk_config = self.risk_config.copy()

            # به‌روزرسانی تنظیمات اصلی
            self.config = new_config
            self.risk_config = new_config.get('risk_management', {})

            # ذخیره تنظیمات قبلی برای مقایسه
            old_use_dynamic = self.use_dynamic_position_sizing
            old_base_risk_percent = self.base_risk_percent
            old_max_position_size = self.max_position_size
            old_fixed_position_size = self.fixed_position_size

            # به‌روزرسانی تنظیمات پایه
            self.account_balance = self.risk_config.get('account_balance', 10000.0)
            self.base_risk_percent = self.risk_config.get('max_risk_per_trade_percent', 1.5)
            self.max_position_size = self.risk_config.get('max_position_size', 200)
            self.fixed_position_size = self.risk_config.get('fixed_position_size', 200)

            # به‌روزرسانی تنظیمات مدیریت سرمایه پویا
            self.use_dynamic_position_sizing = self.risk_config.get('use_dynamic_position_sizing', True)
            self.drawdown_protection = self.risk_config.get('drawdown_protection', True)
            self.winning_streak_factor = self.risk_config.get('winning_streak_factor', 1.1)
            self.losing_streak_factor = self.risk_config.get('losing_streak_factor', 0.9)
            self.max_risk_increase = self.risk_config.get('max_risk_increase', 1.5)
            self.kelly_fraction = self.risk_config.get('kelly_fraction', 0.5)

            # به‌روزرسانی تنظیمات پیشرفته
            self.volatility_adjustment = self.risk_config.get('volatility_adjustment', True)
            self.market_condition_adjustment = self.risk_config.get('market_condition_adjustment', True)
            self.max_risk_decrease = self.risk_config.get('max_risk_decrease', 0.5)
            self.use_equity_curve_based_sizing = self.risk_config.get('use_equity_curve_based_sizing', False)
            self.equity_curve_window = self.risk_config.get('equity_curve_window', 20)

            # لاگ تغییرات مهم
            changes = []
            if old_use_dynamic != self.use_dynamic_position_sizing:
                changes.append(f"مدیریت سرمایه پویا: {'فعال' if self.use_dynamic_position_sizing else 'غیرفعال'}")

            if old_base_risk_percent != self.base_risk_percent:
                changes.append(f"درصد ریسک پایه: {old_base_risk_percent}% → {self.base_risk_percent}%")

            if old_max_position_size != self.max_position_size:
                changes.append(f"حداکثر اندازه پوزیشن: {old_max_position_size} → {self.max_position_size}")

            if old_fixed_position_size != self.fixed_position_size:
                changes.append(f"اندازه پوزیشن ثابت: {old_fixed_position_size} → {self.fixed_position_size}")

            if changes:
                logger.info(f"تنظیمات PositionSizeOptimizer به‌روزرسانی شد: {', '.join(changes)}")
            else:
                logger.debug("تنظیمات PositionSizeOptimizer به‌روزرسانی شد (بدون تغییر در پارامترهای کلیدی)")

    def update_account_balance(self, new_balance: float):
        """
        به‌روزرسانی موجودی حساب

        Args:
            new_balance: موجودی جدید
        """
        with self._lock:
            # به‌روزرسانی منحنی سرمایه
            if self.use_equity_curve_based_sizing:
                self._stats['equity_curve'].append(new_balance)
                # حفظ فقط N مورد آخر
                if len(self._stats['equity_curve']) > self.equity_curve_window:
                    self._stats['equity_curve'] = self._stats['equity_curve'][-self.equity_curve_window:]

            # به‌روزرسانی موجودی
            self.account_balance = new_balance

    def update_statistics(self, drawdown: float, win_rate: float,
                          avg_win_percent: float, avg_loss_percent: float,
                          consecutive_wins: int, consecutive_losses: int):
        """
        به‌روزرسانی آمار برای محاسبات دقیق‌تر

        Args:
            drawdown: درصد افت سرمایه فعلی
            win_rate: نرخ برد در معاملات قبلی
            avg_win_percent: میانگین درصد سود در معاملات سودده
            avg_loss_percent: میانگین درصد ضرر در معاملات زیان‌ده
            consecutive_wins: تعداد معاملات موفق متوالی
            consecutive_losses: تعداد معاملات ناموفق متوالی
        """
        with self._lock:
            self._stats.update({
                'current_drawdown': drawdown,
                'win_rate': win_rate,
                'avg_win_percent': avg_win_percent,
                'avg_loss_percent': avg_loss_percent,
                'consecutive_wins': consecutive_wins,
                'consecutive_losses': consecutive_losses
            })

    def _calculate_equity_curve_factor(self) -> float:
        """
        محاسبه فاکتور تعدیل بر اساس روند منحنی سرمایه

        Returns:
            ضریب تعدیل منحنی سرمایه
        """
        if not self.use_equity_curve_based_sizing or len(self._stats['equity_curve']) < 5:
            return 1.0

        try:
            # محاسبه شیب منحنی سرمایه اخیر با استفاده از regression
            equity_values = np.array(self._stats['equity_curve'])
            x = np.arange(len(equity_values))

            # برآورد خط روند
            slope, _ = np.polyfit(x, equity_values, 1)

            # نرمال‌سازی شیب نسبت به میانگین سرمایه
            norm_slope = slope / np.mean(equity_values)

            # تبدیل به فاکتور - شیب مثبت بیشتر منجر به ریسک بیشتر، شیب منفی به ریسک کمتر
            if norm_slope > 0:
                factor = 1.0 + min(0.3, norm_slope * 10)  # حداکثر 30% افزایش
            else:
                factor = max(0.7, 1.0 + (norm_slope * 10))  # حداکثر 30% کاهش

            return factor

        except Exception as e:
            logger.warning(f"خطا در محاسبه فاکتور منحنی سرمایه: {e}")
            return 1.0

    def _calculate_volatility_factor(self, volatility: float) -> float:
        """
        محاسبه فاکتور تعدیل بر اساس نوسان بازار

        Args:
            volatility: شاخص نوسان‌پذیری بازار (نرمال شده بین 0 تا 1)

        Returns:
            ضریب تعدیل نوسان
        """
        if not self.volatility_adjustment or volatility <= 0:
            return 1.0

        # در نوسان بالا، ریسک کمتر
        return max(self.max_risk_decrease, 1.0 - (volatility * 0.5))

    def _calculate_market_condition_factor(self, market_metrics: Optional[MarketMetrics] = None) -> float:
        """
        محاسبه فاکتور تعدیل بر اساس شرایط بازار

        Args:
            market_metrics: معیارهای بازار

        Returns:
            ضریب تعدیل شرایط بازار
        """
        if not self.market_condition_adjustment or market_metrics is None:
            return 1.0

        # استفاده از قدرت روند و پروفایل حجم در محاسبه فاکتور
        trend_factor = 0.7 + (market_metrics.trend_strength * 0.6)  # بین 0.7 تا 1.3

        # در شرایط خاص بازار، تنظیم ضرایب بیشتر
        condition_factor = 1.0
        if market_metrics.market_condition == "bullish":
            condition_factor = 1.1
        elif market_metrics.market_condition == "bearish":
            condition_factor = 0.9

        return trend_factor * condition_factor

    def calculate_optimal_position_size(self,
                                        signal: SignalInfo,
                                        stop_distance: float,
                                        market_metrics: Optional[MarketMetrics] = None,
                                        adapted_risk_config: Optional[Dict[str, Any]] = None,
                                        _lock_acquired: bool = False) -> Dict[str, Any]:
        """
        محاسبه اندازه بهینه پوزیشن با استفاده از ترکیبی از فاکتورها

        Args:
            signal: سیگنال معاملاتی
            stop_distance: فاصله مطلق بین قیمت ورود و حد ضرر
            market_metrics: معیارهای بازار (اختیاری)
            adapted_risk_config: تنظیمات ریسک تطبیق‌یافته (اختیاری)
            _lock_acquired: آیا قفل قبلاً اخذ شده است

        Returns:
            دیکشنری حاوی اطلاعات اندازه پوزیشن و ریسک
        """
        # استفاده از تنظیمات تطبیق‌یافته اگر ارائه شده
        if not _lock_acquired:
            with self._lock:
                return self.calculate_optimal_position_size(
                    signal,
                    stop_distance,
                    market_metrics,
                    adapted_risk_config,
                    _lock_acquired=True
                )

        risk_config = adapted_risk_config if adapted_risk_config else self.risk_config

        # بررسی اندازه ثابت
        fixed_size = risk_config.get('fixed_position_size', self.fixed_position_size)
        if not self.use_dynamic_position_sizing and fixed_size is not None and fixed_size > 0:
            return {
                'position_size': fixed_size,
                'risk_amount': fixed_size * stop_distance,
                'risk_percent': (fixed_size * stop_distance / self.account_balance) * 100,
                'calculation_method': 'fixed_size'
            }

        # اگر اندازه پویا غیرفعال است، اما اندازه ثابت تعریف نشده
        if not self.use_dynamic_position_sizing:
            # استفاده از درصد پایه
            risk_percent = risk_config.get('max_risk_per_trade_percent', self.base_risk_percent)
            risk_amount = self.account_balance * (risk_percent / 100)
            position_size = risk_amount / stop_distance if stop_distance > 0 else 0

            return {
                'position_size': position_size,
                'risk_amount': risk_amount,
                'risk_percent': risk_percent,
                'calculation_method': 'basic_percentage'
            }

        # محاسبه پویا
        # استخراج پارامترهای ضروری
        base_risk_percent = risk_config.get('max_risk_per_trade_percent', self.base_risk_percent)
        max_risk_increase = risk_config.get('max_risk_increase', self.max_risk_increase)

        # 1. تعدیل بر اساس کیفیت سیگنال
        signal_quality = min(1.0, max(0.1, signal.score.final_score / 100))
        signal_factor = 0.7 + (signal_quality * 0.6)  # بین 0.7 تا 1.3

        # 2. تعدیل بر اساس drawdown
        drawdown_factor = 1.0
        if self.drawdown_protection and self._stats['current_drawdown'] > 0:
            max_drawdown = risk_config.get('max_drawdown_percent', 20.0)
            if self._stats['current_drawdown'] >= max_drawdown:
                drawdown_factor = 0.2  # کاهش شدید در drawdown بالا
            else:
                drawdown_factor = 1.0 - (self._stats['current_drawdown'] / max_drawdown) * 0.8

        # 3. تعدیل بر اساس توالی معاملات
        streak_factor = 1.0
        if self._stats['consecutive_wins'] > 0:
            # افزایش تدریجی با برد متوالی (حداکثر 3 برد)
            streak_factor = min(max_risk_increase,
                                self.winning_streak_factor ** min(self._stats['consecutive_wins'], 3))
        elif self._stats['consecutive_losses'] > 0:
            # کاهش تدریجی با باخت متوالی (حداکثر 3 باخت)
            streak_factor = self.losing_streak_factor ** min(self._stats['consecutive_losses'], 3)

        # 4. تعدیل با معیار کلی
        kelly_factor = 1.0
        if self._stats['win_rate'] > 0 and self._stats['avg_win_percent'] > 0 and self._stats['avg_loss_percent'] > 0:
            # محاسبه معیار کلی بهینه‌سازی شده
            win_rate = self._stats['win_rate'] / 100  # تبدیل به عدد بین 0 و 1
            win_loss_ratio = self._stats['avg_win_percent'] / self._stats['avg_loss_percent']

            # نسخه بهبودیافته فرمول کلی
            kelly = win_rate - ((1 - win_rate) / win_loss_ratio)

            # استفاده از کسری از کلی برای محافظه‌کاری و محدودیت به مقادیر مثبت
            kelly_adjusted = max(0, kelly * self.kelly_fraction)

            # ترکیب با فاکتور قبلی (با وزن 30%)
            kelly_factor = 0.7 + (kelly_adjusted * 0.3 * 10)  # مقیاس‌بندی مناسب

        # 5. فاکتورهای جدید
        # منحنی سرمایه
        equity_factor = self._calculate_equity_curve_factor()

        # نوسان و شرایط بازار
        volatility_factor = 1.0
        market_factor = 1.0

        if market_metrics:
            volatility_factor = self._calculate_volatility_factor(market_metrics.volatility)
            market_factor = self._calculate_market_condition_factor(market_metrics)

        # ترکیب تمام فاکتورها با وزن‌های مناسب
        combined_factor = (
                signal_factor * 0.25 +
                drawdown_factor * 0.2 +
                streak_factor * 0.15 +
                kelly_factor * 0.15 +
                equity_factor * 0.1 +
                volatility_factor * 0.1 +
                market_factor * 0.05
        )

        # محدودیت نهایی
        final_factor = max(self.max_risk_decrease, min(combined_factor, max_risk_increase))
        adjusted_risk_percent = base_risk_percent * final_factor

        # محاسبه مقدار ریسک و اندازه پوزیشن
        risk_amount = self.account_balance * (adjusted_risk_percent / 100)
        position_size = risk_amount / stop_distance if stop_distance > 0 else 0

        # محدودیت حداکثر اندازه
        max_position_size = risk_config.get('max_position_size', self.max_position_size)
        if max_position_size is not None and position_size > max_position_size:
            position_size = max_position_size
            risk_amount = position_size * stop_distance
            adjusted_risk_percent = (risk_amount / self.account_balance) * 100

        # ثبت اطلاعات تصمیم‌گیری
        calculation_details = {
            'signal_quality': signal_quality,
            'signal_factor': signal_factor,
            'drawdown': self._stats['current_drawdown'],
            'drawdown_factor': drawdown_factor,
            'streak_factor': streak_factor,
            'kelly_factor': kelly_factor,
            'equity_factor': equity_factor,
            'volatility_factor': volatility_factor,
            'market_factor': market_factor,
            'combined_factor': combined_factor,
            'final_factor': final_factor
        }

        return {
            'position_size': position_size,
            'risk_amount': risk_amount,
            'risk_percent': adjusted_risk_percent,
            'calculation_method': 'dynamic',
            'details': calculation_details
        }

    def export_configuration(self) -> Dict[str, Any]:
        """
        صدور تنظیمات فعلی به فرمت قابل ذخیره‌سازی

        Returns:
            دیکشنری تنظیمات
        """
        with self._lock:
            config = {
                'base_risk_percent': self.base_risk_percent,
                'max_position_size': self.max_position_size,
                'use_dynamic_position_sizing': self.use_dynamic_position_sizing,
                'drawdown_protection': self.drawdown_protection,
                'winning_streak_factor': self.winning_streak_factor,
                'losing_streak_factor': self.losing_streak_factor,
                'max_risk_increase': self.max_risk_increase,
                'max_risk_decrease': self.max_risk_decrease,
                'kelly_fraction': self.kelly_fraction,
                'volatility_adjustment': self.volatility_adjustment,
                'market_condition_adjustment': self.market_condition_adjustment,
                'use_equity_curve_based_sizing': self.use_equity_curve_based_sizing,
                'equity_curve_window': self.equity_curve_window,
                'current_stats': self._stats.copy()
            }
            return config


class RiskCalculator:
    """
    محاسبه‌گر ریسک پیشرفته برای تصمیم‌گیری معاملاتی
    """

    def __init__(self, config: Dict[str, Any]):
        """
        مقداردهی اولیه

        Args:
            config: دیکشنری تنظیمات
        """
        self.config = config
        self.risk_config = config.get('risk_management', {})

        # تنظیمات اصلی ریسک
        self.account_balance = self.risk_config.get('account_balance', 10000.0)
        self.max_risk_per_trade_percent = self.risk_config.get('max_risk_per_trade_percent', 1.5)
        self.max_risk_per_symbol_percent = self.risk_config.get('max_risk_per_symbol_percent', 2.5)
        self.max_daily_risk_percent = self.risk_config.get('max_daily_risk_percent', 5.0)
        self.max_open_risk_percent = self.risk_config.get('max_open_risk_percent', 10.0)

        # تنظیمات پیشرفته
        self.account_risk_multiplier = self.risk_config.get('account_risk_multiplier', 1.0)
        self.min_risk_reward_ratio = self.risk_config.get('min_risk_reward_ratio', 1.8)
        self.preferred_risk_reward_ratio = self.risk_config.get('preferred_risk_reward_ratio', 2.5)

        # تنظیمات بهینه‌سازی شده جدید
        self.correlations_in_risk_calc = self.risk_config.get('consider_correlations_in_risk', True)
        self.market_volatility_adjustment = self.risk_config.get('market_volatility_adjustment', True)
        self.dynamic_risk_adjustment = self.risk_config.get('dynamic_risk_adjustment', True)
        self.max_risk_per_market = self.risk_config.get('max_risk_per_market', 4.0)  # حداکثر ریسک در یک بازار
        self.position_scaling = self.risk_config.get('position_scaling', True)  # مقیاس‌بندی اندازه پوزیشن
        self.hedging_adjustment = self.risk_config.get('hedging_adjustment', True)  # تعدیل برای معاملات پوششی

        # مدیریت معاملات مرتبط
        self.related_markets = self.risk_config.get('related_markets', {})

        # قفل برای به‌روزرسانی همزمان
        self._lock = threading.RLock()

        logger.info("RiskCalculator با حداکثر ریسک معامله: %.2f%% راه‌اندازی شد", self.max_risk_per_trade_percent)

    def update_config(self, new_config: Dict[str, Any]):
        """
        به‌روزرسانی تنظیمات در زمان اجرا

        Args:
            new_config: دیکشنری تنظیمات جدید
        """
        with self._lock:
            old_config = self.config.copy()
            old_risk_config = self.risk_config.copy()

            # به‌روزرسانی تنظیمات اصلی
            self.config = new_config
            self.risk_config = new_config.get('risk_management', {})

            # ذخیره تنظیمات قبلی برای مقایسه
            old_max_risk_per_trade = self.max_risk_per_trade_percent
            old_max_risk_per_symbol = self.max_risk_per_symbol_percent
            old_max_daily_risk = self.max_daily_risk_percent
            old_max_open_risk = self.max_open_risk_percent

            # به‌روزرسانی تنظیمات اصلی ریسک
            self.account_balance = self.risk_config.get('account_balance', 10000.0)
            self.max_risk_per_trade_percent = self.risk_config.get('max_risk_per_trade_percent', 1.5)
            self.max_risk_per_symbol_percent = self.risk_config.get('max_risk_per_symbol_percent', 2.5)
            self.max_daily_risk_percent = self.risk_config.get('max_daily_risk_percent', 5.0)
            self.max_open_risk_percent = self.risk_config.get('max_open_risk_percent', 10.0)

            # به‌روزرسانی تنظیمات پیشرفته
            self.account_risk_multiplier = self.risk_config.get('account_risk_multiplier', 1.0)
            self.min_risk_reward_ratio = self.risk_config.get('min_risk_reward_ratio', 1.8)
            self.preferred_risk_reward_ratio = self.risk_config.get('preferred_risk_reward_ratio', 2.5)

            # به‌روزرسانی تنظیمات بهینه‌سازی شده
            self.correlations_in_risk_calc = self.risk_config.get('consider_correlations_in_risk', True)
            self.market_volatility_adjustment = self.risk_config.get('market_volatility_adjustment', True)
            self.dynamic_risk_adjustment = self.risk_config.get('dynamic_risk_adjustment', True)
            self.max_risk_per_market = self.risk_config.get('max_risk_per_market', 4.0)
            self.position_scaling = self.risk_config.get('position_scaling', True)
            self.hedging_adjustment = self.risk_config.get('hedging_adjustment', True)

            # به‌روزرسانی معاملات مرتبط
            self.related_markets = self.risk_config.get('related_markets', {})

            # لاگ تغییرات مهم
            changes = []
            if old_max_risk_per_trade != self.max_risk_per_trade_percent:
                changes.append(f"حداکثر ریسک معامله: {old_max_risk_per_trade}% → {self.max_risk_per_trade_percent}%")

            if old_max_risk_per_symbol != self.max_risk_per_symbol_percent:
                changes.append(f"حداکثر ریسک نماد: {old_max_risk_per_symbol}% → {self.max_risk_per_symbol_percent}%")

            if old_max_daily_risk != self.max_daily_risk_percent:
                changes.append(f"حداکثر ریسک روزانه: {old_max_daily_risk}% → {self.max_daily_risk_percent}%")

            if old_max_open_risk != self.max_open_risk_percent:
                changes.append(f"حداکثر ریسک باز: {old_max_open_risk}% → {self.max_open_risk_percent}%")

            if changes:
                logger.info(f"تنظیمات RiskCalculator به‌روزرسانی شد: {', '.join(changes)}")
            else:
                logger.debug("تنظیمات RiskCalculator به‌روزرسانی شد (بدون تغییر در پارامترهای کلیدی)")

    def update_account_balance(self, new_balance: float):
        """به‌روزرسانی موجودی حساب"""
        with self._lock:
            self.account_balance = new_balance

    def calculate_trade_risk(self, entry_price: float, stop_loss: float, position_size: float,
                             direction: str) -> Dict[str, float]:
        """
        محاسبه ریسک یک معامله

        Args:
            entry_price: قیمت ورود
            stop_loss: قیمت حد ضرر
            position_size: اندازه پوزیشن
            direction: جهت معامله ('long' یا 'short')

        Returns:
            دیکشنری اطلاعات ریسک
        """
        # محاسبه مقدار ریسک در واحد پول
        if direction == 'long':
            risk_amount = (entry_price - stop_loss) * position_size
        else:  # 'short'
            risk_amount = (stop_loss - entry_price) * position_size

        # اطمینان از مثبت بودن ریسک
        risk_amount = abs(risk_amount)

        # محاسبه درصد ریسک
        risk_percent = (risk_amount / self.account_balance) * 100

        # محاسبه درصد فاصله استاپ لاس
        stop_distance_percent = abs(entry_price - stop_loss) / entry_price * 100

        return {
            'risk_amount': risk_amount,
            'risk_percent': risk_percent,
            'position_size': position_size,
            'account_balance': self.account_balance,
            'stop_distance_percent': stop_distance_percent,
        }

    def check_symbol_risk(self, symbol: str, new_risk_amount: float, active_trades: List[Trade]) -> Tuple[bool, float]:
        """
        بررسی ریسک متمرکز روی یک نماد خاص

        Args:
            symbol: نماد ارز
            new_risk_amount: مقدار ریسک معامله جدید
            active_trades: لیست معاملات فعال

        Returns:
            tuple شامل (آیا ریسک قابل قبول است، مجموع ریسک فعلی نماد)
        """
        # محاسبه ریسک فعلی روی این نماد
        current_symbol_risk = sum(
            trade.risk_amount for trade in active_trades
            if hasattr(trade, 'symbol') and trade.symbol == symbol and trade.status == 'open'
        )

        # ریسک کل بعد از اضافه شدن معامله جدید
        total_symbol_risk = current_symbol_risk + new_risk_amount

        # محاسبه درصد ریسک
        total_symbol_risk_percent = (total_symbol_risk / self.account_balance) * 100

        # بررسی محدودیت ریسک برای یک نماد
        is_acceptable = total_symbol_risk_percent <= self.max_risk_per_symbol_percent

        if not is_acceptable:
            logger.info(
                f"ریسک نماد بیش از حد مجاز: {total_symbol_risk_percent:.2f}% > {self.max_risk_per_symbol_percent:.2f}%")

        return is_acceptable, total_symbol_risk_percent

    def check_market_risk(self, symbol: str, new_risk_amount: float, active_trades: List[Trade]) -> Tuple[bool, float]:
        """
        بررسی ریسک متمرکز روی یک بازار (شامل نمادهای مرتبط)

        Args:
            symbol: نماد ارز
            new_risk_amount: مقدار ریسک معامله جدید
            active_trades: لیست معاملات فعال

        Returns:
            tuple شامل (آیا ریسک قابل قبول است، مجموع ریسک فعلی بازار)
        """
        if not self.related_markets:
            # اگر بازارهای مرتبط تعریف نشده، فقط بررسی نماد را انجام بده
            return self.check_symbol_risk(symbol, new_risk_amount, active_trades)

        # پیدا کردن گروه بازار
        symbol_market = None
        for market, symbols in self.related_markets.items():
            if symbol in symbols:
                symbol_market = market
                break

        if not symbol_market:
            # اگر بازار مرتبطی پیدا نشد، فقط بررسی نماد را انجام بده
            return self.check_symbol_risk(symbol, new_risk_amount, active_trades)

        # محاسبه ریسک فعلی در این بازار
        market_symbols = self.related_markets[symbol_market]
        current_market_risk = sum(
            trade.risk_amount for trade in active_trades
            if hasattr(trade, 'symbol') and trade.symbol in market_symbols and trade.status == 'open'
        )

        # محاسبه ریسک جدید
        total_market_risk = current_market_risk + new_risk_amount
        total_market_risk_percent = (total_market_risk / self.account_balance) * 100

        # بررسی محدودیت ریسک برای یک بازار
        is_acceptable = total_market_risk_percent <= self.max_risk_per_market

        if not is_acceptable:
            logger.info(
                f"ریسک بازار {symbol_market} بیش از حد مجاز: {total_market_risk_percent:.2f}% > {self.max_risk_per_market:.2f}%")

        return is_acceptable, total_market_risk_percent

    def check_total_risk(self, new_risk_amount: float, active_trades: List[Trade]) -> Tuple[bool, float]:
        """
        بررسی ریسک کل پورتفولیو

        Args:
            new_risk_amount: مقدار ریسک معامله جدید
            active_trades: لیست معاملات فعال

        Returns:
            tuple شامل (آیا ریسک قابل قبول است، درصد ریسک کل)
        """
        # محاسبه ریسک فعلی
        current_total_risk = sum(
            trade.risk_amount for trade in active_trades
            if hasattr(trade, 'status') and trade.status == 'open'
        )

        # ریسک کل بعد از اضافه شدن معامله جدید
        total_risk = current_total_risk + new_risk_amount

        # محاسبه درصد ریسک
        total_risk_percent = (total_risk / self.account_balance) * 100

        # بررسی محدودیت ریسک کلی
        is_acceptable = total_risk_percent <= self.max_open_risk_percent

        if not is_acceptable:
            logger.info(
                f"ریسک کل پورتفولیو بیش از حد مجاز: {total_risk_percent:.2f}% > {self.max_open_risk_percent:.2f}%")

        return is_acceptable, total_risk_percent

    def check_daily_risk(self, new_risk_amount: float, active_trades: List[Trade]) -> Tuple[bool, float]:
        """
        بررسی ریسک روزانه

        Args:
            new_risk_amount: مقدار ریسک معامله جدید
            active_trades: لیست معاملات فعال

        Returns:
            tuple شامل (آیا ریسک قابل قبول است، درصد ریسک روزانه)
        """
        # محاسبه ریسک معاملات امروز
        today = datetime.now().date()

        # تبدیل timestamp به datetime اگر رشته است
        todays_trades = []
        for trade in active_trades:
            if not hasattr(trade, 'timestamp'):
                continue

            entry_time = trade.timestamp
            if isinstance(entry_time, str):
                try:
                    entry_time = datetime.fromisoformat(entry_time)
                except (ValueError, TypeError):
                    continue

            if entry_time.date() == today:
                todays_trades.append(trade)

        daily_risk = sum(trade.risk_amount for trade in todays_trades if hasattr(trade, 'risk_amount'))
        daily_risk += new_risk_amount

        # محاسبه درصد ریسک
        daily_risk_percent = (daily_risk / self.account_balance) * 100

        # بررسی محدودیت ریسک روزانه
        is_acceptable = daily_risk_percent <= self.max_daily_risk_percent

        if not is_acceptable:
            logger.info(f"ریسک روزانه بیش از حد مجاز: {daily_risk_percent:.2f}% > {self.max_daily_risk_percent:.2f}%")

        return is_acceptable, daily_risk_percent

    def check_correlation_adjusted_risk(self, symbol: str, direction: str, new_risk_amount: float,
                                        active_trades: List[Trade],
                                        correlation_data: Optional[Dict[str, float]] = None) -> Tuple[bool, float]:
        """
        بررسی ریسک با تعدیل همبستگی

        Args:
            symbol: نماد ارز
            direction: جهت معامله ('long' یا 'short')
            new_risk_amount: مقدار ریسک معامله جدید
            active_trades: لیست معاملات فعال
            correlation_data: دیکشنری همبستگی‌ها {symbol_pair: correlation}

        Returns:
            tuple شامل (آیا ریسک قابل قبول است، درصد ریسک تعدیل شده)
        """
        if not self.correlations_in_risk_calc or not correlation_data:
            # اگر تعدیل همبستگی فعال نیست، فقط بررسی کل را انجام بده
            return self.check_total_risk(new_risk_amount, active_trades)

        # محاسبه تعدیل شده با در نظر گرفتن همبستگی
        effective_risk = new_risk_amount
        current_adjusted_risk = 0

        for trade in active_trades:
            if not hasattr(trade, 'status') or trade.status != 'open' or not hasattr(trade, 'symbol'):
                continue

            # بررسی همبستگی با معامله جدید
            if trade.symbol == symbol:
                # همان نماد، همبستگی کامل
                trade_contribution = trade.risk_amount
            else:
                # پیدا کردن همبستگی
                pair_key = f"{symbol}_{trade.symbol}"
                reverse_pair_key = f"{trade.symbol}_{symbol}"

                correlation = 0.0
                if pair_key in correlation_data:
                    correlation = correlation_data[pair_key]
                elif reverse_pair_key in correlation_data:
                    correlation = correlation_data[reverse_pair_key]

                # تعدیل جهت معامله
                if hasattr(trade, 'direction') and trade.direction != direction:
                    correlation = -correlation

                # تعدیل ریسک بر اساس همبستگی
                trade_contribution = trade.risk_amount * abs(correlation)

            current_adjusted_risk += trade_contribution

        # محاسبه ریسک کل تعدیل شده
        total_adjusted_risk = current_adjusted_risk + effective_risk
        adjusted_risk_percent = (total_adjusted_risk / self.account_balance) * 100

        # افزایش آستانه مجاز برای ریسک تعدیل شده (عملاً مقدار بیشتری اجازه داده می‌شود)
        adjusted_threshold = self.max_open_risk_percent * 1.5
        is_acceptable = adjusted_risk_percent <= adjusted_threshold

        if not is_acceptable:
            logger.info(
                f"ریسک تعدیل‌شده با همبستگی بیش از حد مجاز: {adjusted_risk_percent:.2f}% > {adjusted_threshold:.2f}%")

        return is_acceptable, adjusted_risk_percent

    def calculate_optimal_risk_reward(self, signal_quality: float,
                                      market_volatility: float = 0.5) -> float:
        """
        محاسبه نسبت ریسک به ریوارد مطلوب بر اساس کیفیت سیگنال و نوسان بازار

        Args:
            signal_quality: کیفیت سیگنال (0-1)
            market_volatility: نوسان بازار (0-1)

        Returns:
            نسبت ریسک به ریوارد مطلوب
        """
        # تطبیق RR بر اساس کیفیت سیگنال
        base_rr = self.preferred_risk_reward_ratio
        min_rr = self.min_risk_reward_ratio

        # سیگنال‌های قوی‌تر می‌توانند RR کمتری داشته باشند
        quality_adjusted_rr = base_rr - ((base_rr - min_rr) * signal_quality)

        # تعدیل بر اساس نوسان بازار
        if self.market_volatility_adjustment:
            # در بازارهای با نوسان بالا، نیاز به RR بالاتر است
            volatility_factor = 1.0 + (market_volatility * 0.5)  # حداکثر 50% افزایش
            final_rr = quality_adjusted_rr * volatility_factor
        else:
            final_rr = quality_adjusted_rr

        # اطمینان از رعایت حداقل RR
        return max(final_rr, min_rr)

    def get_position_sizing_recommendation(self, entry_price: float, stop_loss: float,
                                           direction: str, signal_quality: float,
                                           market_volatility: float = 0.5) -> Dict[str, Any]:
        """
        توصیه‌های اندازه پوزیشن و مدیریت ریسک

        Args:
            entry_price: قیمت ورود
            stop_loss: قیمت حد ضرر
            direction: جهت معامله
            signal_quality: کیفیت سیگنال (0-1)
            market_volatility: نوسان بازار (0-1)

        Returns:
            دیکشنری توصیه‌ها
        """
        # محاسبه فاصله حد ضرر به درصد
        if direction == 'long':
            stop_distance_percent = ((entry_price - stop_loss) / entry_price) * 100
        else:
            stop_distance_percent = ((stop_loss - entry_price) / entry_price) * 100

        # تعدیل درصد ریسک بر اساس کیفیت سیگنال و نوسان بازار
        base_risk_percent = self.max_risk_per_trade_percent

        if self.dynamic_risk_adjustment:
            # تعدیل ریسک بر اساس کیفیت سیگنال (70% تا 130%)
            signal_factor = 0.7 + (signal_quality * 0.6)

            # تعدیل ریسک بر اساس نوسان بازار (کاهش ریسک در نوسان بالا)
            volatility_factor = 1.0 - (market_volatility * 0.3)  # حداکثر 30% کاهش

            # ترکیب تعدیل‌ها
            adjusted_risk_percent = base_risk_percent * signal_factor * volatility_factor
        else:
            adjusted_risk_percent = base_risk_percent

        # محدودیت ریسک
        adjusted_risk_percent = max(base_risk_percent * 0.5, min(adjusted_risk_percent, base_risk_percent * 1.5))

        # محاسبه مقدار ریسک و اندازه پوزیشن
        risk_amount = self.account_balance * (adjusted_risk_percent / 100)

        # محاسبه اندازه پوزیشن
        if stop_distance_percent > 0:
            position_size = risk_amount / (entry_price * (stop_distance_percent / 100))
        else:
            position_size = 0
            logger.warning("فاصله استاپ نامعتبر: استاپ بسیار نزدیک به قیمت ورود است")

        # محاسبه RR مطلوب
        optimal_rr = self.calculate_optimal_risk_reward(signal_quality, market_volatility)

        # محاسبه قیمت حد سود
        if direction == 'long':
            take_profit = entry_price * (1 + (stop_distance_percent * optimal_rr / 100))
        else:
            take_profit = entry_price * (1 - (stop_distance_percent * optimal_rr / 100))

        # محاسبه چند سطح حد سود
        take_profit_levels = self._calculate_multiple_take_profits(entry_price, stop_loss, direction, optimal_rr)

        return {
            'position_size': position_size,
            'risk_amount': risk_amount,
            'risk_percent': adjusted_risk_percent,
            'take_profit': take_profit,
            'take_profit_levels': take_profit_levels,
            'risk_reward_ratio': optimal_rr,
            'stop_distance_percent': stop_distance_percent,
            'signal_quality_factor': signal_quality,
            'volatility_factor': volatility_factor if self.dynamic_risk_adjustment else 1.0
        }

    def _calculate_multiple_take_profits(self, entry_price: float, stop_loss: float,
                                         direction: str, risk_reward_ratio: float) -> List[Dict[str, float]]:
        """
        محاسبه چندین سطح حد سود

        Args:
            entry_price: قیمت ورود
            stop_loss: قیمت حد ضرر
            direction: جهت معامله
            risk_reward_ratio: نسبت ریسک به ریوارد کلی

        Returns:
            لیست سطوح حد سود
        """
        # محاسبه فاصله حد ضرر
        stop_distance = abs(entry_price - stop_loss)

        # طراحی سطوح حد سود با توزیع مناسب
        tp_levels = []

        # سه سطح به عنوان مثال (1/3، 2/3 و کامل RR)
        fractions = [0.382, 0.618, 1.0]  # استفاده از سطوح فیبوناچی

        for i, fraction in enumerate(fractions):
            current_rr = risk_reward_ratio * fraction

            if direction == 'long':
                tp_price = entry_price + (stop_distance * current_rr)
                distance_percent = ((tp_price - entry_price) / entry_price) * 100
            else:
                tp_price = entry_price - (stop_distance * current_rr)
                distance_percent = ((entry_price - tp_price) / entry_price) * 100

            tp_levels.append({
                'level': i + 1,
                'price': tp_price,
                'risk_reward': current_rr,
                'distance_percent': distance_percent,
                'suggested_size': 1.0 / len(fractions)  # توزیع مساوی
            })

        return tp_levels

    def evaluate_trade_opportunity(self, symbol: str, direction: str, entry_price: float,
                                   stop_loss: float, position_size: float,
                                   signal_quality: float,
                                   active_trades: List[Trade],
                                   correlation_data: Optional[Dict[str, float]] = None,
                                   market_volatility: float = 0.5) -> Dict[str, Any]:
        """
        ارزیابی جامع فرصت معاملاتی

        Args:
            symbol: نماد ارز
            direction: جهت معامله
            entry_price: قیمت ورود
            stop_loss: قیمت حد ضرر
            position_size: اندازه پوزیشن
            signal_quality: کیفیت سیگنال (0-1)
            active_trades: لیست معاملات فعال
            correlation_data: داده‌های همبستگی
            market_volatility: نوسان بازار (0-1)

        Returns:
            دیکشنری ارزیابی جامع
        """
        # محاسبه ریسک معامله
        risk_info = self.calculate_trade_risk(entry_price, stop_loss, position_size, direction)
        risk_amount = risk_info['risk_amount']

        # بررسی محدودیت‌های ریسک
        symbol_allowed, symbol_risk = self.check_symbol_risk(symbol, risk_amount, active_trades)
        market_allowed, market_risk = self.check_market_risk(symbol, risk_amount, active_trades)
        total_allowed, total_risk = self.check_total_risk(risk_amount, active_trades)
        daily_allowed, daily_risk = self.check_daily_risk(risk_amount, active_trades)

        # بررسی ریسک تعدیل شده با همبستگی
        corr_allowed, corr_risk = self.check_correlation_adjusted_risk(
            symbol, direction, risk_amount, active_trades, correlation_data)

        # محاسبه نسبت ریسک به ریوارد
        rr_ratio = self.calculate_optimal_risk_reward(signal_quality, market_volatility)

        # محاسبه امتیاز کلی معامله (0-100)
        base_score = signal_quality * 100  # امتیاز پایه بر اساس کیفیت سیگنال

        # تعدیل‌های امتیاز
        risk_adjustments = []

        # تعدیل ریسک نماد
        if symbol_risk > 0:
            symbol_factor = 1.0 - (symbol_risk / self.max_risk_per_symbol_percent)
            risk_adjustments.append(max(0, symbol_factor))

        # تعدیل ریسک بازار
        if market_risk > 0:
            market_factor = 1.0 - (market_risk / self.max_risk_per_market)
            risk_adjustments.append(max(0, market_factor))

        # تعدیل ریسک کل
        if total_risk > 0:
            total_factor = 1.0 - (total_risk / self.max_open_risk_percent)
            risk_adjustments.append(max(0, total_factor))

        # تعدیل همبستگی
        if corr_risk > 0:
            adjusted_threshold = self.max_open_risk_percent * 1.5
            corr_factor = 1.0 - (corr_risk / adjusted_threshold)
            risk_adjustments.append(max(0, corr_factor))

        # میانگین تعدیل‌ها
        if risk_adjustments:
            risk_adjustment_avg = sum(risk_adjustments) / len(risk_adjustments)
            adjusted_score = base_score * risk_adjustment_avg
        else:
            adjusted_score = base_score

        # محدودیت نهایی امتیاز
        final_score = max(0, min(adjusted_score, 100))

        # تصمیم نهایی
        is_acceptable = symbol_allowed and market_allowed and total_allowed and daily_allowed and corr_allowed

        # تعیین سطح ریسک
        if final_score >= 75:
            risk_level = "کم"
        elif final_score >= 50:
            risk_level = "متوسط"
        else:
            risk_level = "بالا"

        return {
            'trade_id': f"{symbol}_{direction}_{datetime.now().strftime('%Y%m%d%H%M%S')}",
            'symbol': symbol,
            'direction': direction,
            'entry_price': entry_price,
            'stop_loss': stop_loss,
            'position_size': position_size,
            'risk': {
                'amount': risk_amount,
                'percent': risk_info['risk_percent'],
                'symbol_risk_percent': symbol_risk,
                'market_risk_percent': market_risk,
                'total_risk_percent': total_risk,
                'daily_risk_percent': daily_risk,
                'correlation_adjusted_risk_percent': corr_risk
            },
            'risk_reward_ratio': rr_ratio,
            'signal_quality': signal_quality,
            'assessment': {
                'is_acceptable': is_acceptable,
                'constraints_violated': {
                    'symbol_risk': not symbol_allowed,
                    'market_risk': not market_allowed,
                    'total_risk': not total_allowed,
                    'daily_risk': not daily_allowed,
                    'correlation_risk': not corr_allowed
                },
                'risk_level': risk_level,
                'score': final_score
            },
            'recommendations': self._generate_trade_recommendations(
                symbol, direction, entry_price, stop_loss, position_size,
                is_acceptable, final_score, risk_info, rr_ratio)
        }

    def _generate_trade_recommendations(self, symbol: str, direction: str,
                                        entry_price: float, stop_loss: float, position_size: float,
                                        is_acceptable: bool, score: float,
                                        risk_info: Dict[str, float], rr_ratio: float) -> List[str]:
        """
        تولید توصیه‌های مرتبط با معامله

        Args:
            مقادیر مختلف مربوط به معامله

        Returns:
            لیست توصیه‌ها
        """
        recommendations = []

        if not is_acceptable:
            # توصیه‌های مدیریت ریسک
            if risk_info['risk_percent'] > self.max_risk_per_trade_percent:
                reduced_size = position_size * (self.max_risk_per_trade_percent / risk_info['risk_percent'])
                recommendations.append(
                    f"کاهش اندازه پوزیشن از {position_size:.4f} به {reduced_size:.4f} برای رعایت محدودیت ریسک.")
            else:
                recommendations.append(
                    "منتظر فرصت بهتر بمانید به دلیل محدودیت‌های ریسک پورتفولیو.")
        else:
            # توصیه‌های بهبود معامله
            if score < 75:
                # توصیه‌های برای معاملات با امتیاز کمتر
                if risk_info['stop_distance_percent'] < 1.0:
                    recommendations.append(
                        "حد ضرر را دورتر قرار دهید تا احتمال خروج زودهنگام کاهش یابد.")

                if rr_ratio < 2.0:
                    recommendations.append(
                        f"نسبت ریسک به ریوارد {rr_ratio:.2f} پایین است. حد سود را دورتر قرار دهید.")

                recommendations.append(
                    "با توجه به کیفیت متوسط سیگنال، از اندازه پوزیشن کوچکتر استفاده کنید.")

            else:
                # توصیه‌های برای معاملات با امتیاز بالا
                recommendations.append(
                    "سیگنال با کیفیت بالا شناسایی شد. ورود چندمرحله‌ای را در نظر بگیرید.")

                if risk_info['risk_percent'] < self.max_risk_per_trade_percent * 0.7:
                    increased_size = position_size * (
                                self.max_risk_per_trade_percent / risk_info['risk_percent']) * 0.85
                    recommendations.append(
                        f"افزایش اندازه پوزیشن از {position_size:.4f} به {increased_size:.4f} برای استفاده بهینه از ریسک.")

        # توصیه‌های عمومی
        if direction == 'long':
            recommendations.append(
                f"تنظیم سطوح حد سود در {entry_price * (1 + (risk_info['stop_distance_percent'] * rr_ratio / 100)):.2f} "
                f"برای {rr_ratio:.1f}R، خروج تدریجی در نظر گرفته شود.")
        else:
            recommendations.append(
                f"تنظیم سطوح حد سود در {entry_price * (1 - (risk_info['stop_distance_percent'] * rr_ratio / 100)):.2f} "
                f"برای {rr_ratio:.1f}R، خروج تدریجی در نظر گرفته شود.")

        return recommendations


class TradeAnalyzer:
    """
    تحلیل معاملات گذشته و استخراج آمار و الگوهای معاملاتی
    """

    def __init__(self, config: Dict[str, Any], db_path: str = None):
        """
        مقداردهی اولیه

        Args:
            config: دیکشنری تنظیمات
            db_path: مسیر دیتابیس (اختیاری)
        """
        self.config = config
        self.db_path = db_path or config.get('storage', {}).get('database_path', 'data/trades.db')
        self.conn = None
        self.cursor = None

        # تنظیمات تحلیل
        self.analysis_config = config.get('analysis', {})
        self.min_trades_for_analysis = self.analysis_config.get('min_trades_for_analysis', 10)
        self.use_advanced_metrics = self.analysis_config.get('use_advanced_metrics', True)
        self.statistical_significance = self.analysis_config.get('statistical_significance', 0.05)
        self.backtest_dir = config.get('backtest', {}).get('results_dir', 'backtest_results')

        # آمار پایه
        self._init_stats()

        # کش نتایج تحلیل با timestamp
        self._analysis_cache = {}  # {analysis_id: (result, timestamp)}
        self._cache_expiry_seconds = 1800  # 30 دقیقه

        # اتصال به دیتابیس
        self._connect_to_db()
        logger.info("TradeAnalyzer راه‌اندازی شد")

    def update_config(self, new_config: Dict[str, Any]):
        """
        به‌روزرسانی تنظیمات در زمان اجرا

        Args:
            new_config: دیکشنری تنظیمات جدید
        """
        old_config = self.config.copy()
        old_analysis_config = self.analysis_config.copy()

        # به‌روزرسانی تنظیمات اصلی
        self.config = new_config

        # به‌روزرسانی تنظیمات تحلیل
        self.analysis_config = new_config.get('analysis', {})

        # ذخیره تنظیمات قبلی برای مقایسه
        old_min_trades = self.min_trades_for_analysis
        old_use_advanced = self.use_advanced_metrics

        # به‌روزرسانی پارامترهای تحلیل
        self.min_trades_for_analysis = self.analysis_config.get('min_trades_for_analysis', 10)
        self.use_advanced_metrics = self.analysis_config.get('use_advanced_metrics', True)
        self.statistical_significance = self.analysis_config.get('statistical_significance', 0.05)
        self.backtest_dir = new_config.get('backtest', {}).get('results_dir', 'backtest_results')

        # به‌روزرسانی مسیر دیتابیس
        new_db_path = new_config.get('storage', {}).get('database_path', 'data/trades.db')
        if self.db_path != new_db_path:
            # بستن اتصال فعلی
            self._close_db()
            self.db_path = new_db_path
            # اتصال مجدد به دیتابیس جدید
            self._connect_to_db()
            logger.info(f"مسیر دیتابیس تغییر کرد: {self.db_path}")

        # لاگ تغییرات مهم
        changes = []
        if old_min_trades != self.min_trades_for_analysis:
            changes.append(f"حداقل معاملات برای تحلیل: {old_min_trades} → {self.min_trades_for_analysis}")

        if old_use_advanced != self.use_advanced_metrics:
            changes.append(f"معیارهای پیشرفته: {'فعال' if self.use_advanced_metrics else 'غیرفعال'}")

        if changes:
            logger.info(f"تنظیمات TradeAnalyzer به‌روزرسانی شد: {', '.join(changes)}")
        else:
            logger.debug("تنظیمات TradeAnalyzer به‌روزرسانی شد (بدون تغییر در پارامترهای کلیدی)")

        # پاکسازی کش در صورت تغییر پارامترهای مهم
        if old_min_trades != self.min_trades_for_analysis or old_use_advanced != self.use_advanced_metrics:
            self._analysis_cache = {}
            logger.info("کش تحلیل‌ها به دلیل تغییر پارامترهای مهم پاکسازی شد")

    def _init_stats(self):
        """مقداردهی اولیه آمار"""
        self.stats = {
            'total_trades': 0,
            'winning_trades': 0,
            'losing_trades': 0,
            'win_rate': 0.0,
            'profit_factor': 0.0,
            'average_win': 0.0,
            'average_loss': 0.0,
            'largest_win': 0.0,
            'largest_loss': 0.0,
            'avg_win_duration': 0.0,
            'avg_loss_duration': 0.0,
            'consecutive_wins': 0,
            'consecutive_losses': 0,
            'max_consecutive_wins': 0,
            'max_consecutive_losses': 0,
            'total_profit': 0.0,
            'total_loss': 0.0,
            'net_profit': 0.0,
            'return_percent': 0.0,
            'sharpe_ratio': 0.0,
            'sortino_ratio': 0.0,
            'max_drawdown': 0.0,
            'max_drawdown_percent': 0.0,
            'best_symbol': None,
            'worst_symbol': None,
            'best_setup': None,
            'worst_setup': None,
            # معیارهای پیشرفته جدید
            'profit_consistency': 0.0,  # ثبات سود
            'recovery_factor': 0.0,  # فاکتور بازیابی
            'expectancy': 0.0,  # امید ریاضی
            'avg_daily_trades': 0.0,  # میانگین معاملات روزانه
            'time_in_market_percent': 0.0,  # درصد زمان در بازار
            'performance_by_day': {},  # عملکرد بر اساس روز هفته
            'performance_by_hour': {},  # عملکرد بر اساس ساعت روز
            'performance_by_setup': {},  # عملکرد بر اساس استراتژی
            'performance_by_symbol': {},  # عملکرد بر اساس نماد
            'performance_by_timeframe': {},  # عملکرد بر اساس تایم‌فریم
            'performance_trend': [],  # روند عملکرد
        }

    def _connect_to_db(self):
        """اتصال به دیتابیس SQLite"""
        try:
            # اتصال به دیتابیس با timeout برای مدیریت قفل‌ها
            self.conn = sqlite3.connect(self.db_path, timeout=10)
            self.conn.row_factory = sqlite3.Row
            self.cursor = self.conn.cursor()
            logger.debug(f"اتصال به دیتابیس: {self.db_path}")

            # بررسی ساختار دیتابیس
            self._ensure_db_structure()

        except Exception as e:
            logger.error(f"خطا در اتصال به دیتابیس: {e}")
            self.conn = None
            self.cursor = None

    def _ensure_db_structure(self):
        """اطمینان از وجود جداول و ساختار مورد نیاز"""
        if not self.conn or not self.cursor:
            return

        try:
            # بررسی وجود جدول آمار
            self.cursor.execute("""
            CREATE TABLE IF NOT EXISTS trade_stats (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                date TEXT NOT NULL,
                stats TEXT NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
            """)

            # بررسی وجود جدول تاریخچه موجودی
            self.cursor.execute("""
            CREATE TABLE IF NOT EXISTS balance_history (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT NOT NULL,
                balance REAL NOT NULL,
                equity REAL,
                margin REAL
            )
            """)

            self.conn.commit()

        except Exception as e:
            logger.error(f"خطا در ایجاد ساختار دیتابیس: {e}")

    def _close_db(self):
        """بستن اتصال دیتابیس"""
        if self.conn:
            self.conn.close()
            self.conn = None
            self.cursor = None

    def __del__(self):
        """پاکسازی منابع هنگام حذف شیء"""
        self._close_db()

    @safe_execution(default_value={})
    def analyze_trades(self, days_back: int = 30, use_cache: bool = True) -> Dict[str, Any]:
        """
        تحلیل معاملات در بازه زمانی مشخص

        Args:
            days_back: تعداد روزهای گذشته برای تحلیل
            use_cache: استفاده از کش برای تحلیل‌های مکرر

        Returns:
            دیکشنری آمار و تحلیل‌ها
        """
        # بررسی کش
        cache_key = f"trade_analysis_{days_back}"
        if use_cache and cache_key in self._analysis_cache:
            cached_result, timestamp = self._analysis_cache[cache_key]
            age = (datetime.now() - timestamp).total_seconds()
            if age < self._cache_expiry_seconds:
                return cached_result

        if not self.conn or not self.cursor:
            logger.error("اتصال دیتابیس در دسترس نیست")
            self._connect_to_db()
            if not self.conn:
                return self.stats

        try:
            # تعیین تاریخ شروع
            start_date = (datetime.now() - timedelta(days=days_back)).strftime('%Y-%m-%d')

            # دریافت معاملات بسته شده در بازه زمانی
            query = """
            SELECT * FROM trades 
            WHERE status = 'closed' 
            AND exit_time >= ? 
            ORDER BY exit_time ASC
            """

            self.cursor.execute(query, (start_date,))
            trades = [dict(row) for row in self.cursor.fetchall()]

            if not trades:
                logger.info(f"معامله بسته شده‌ای در {days_back} روز گذشته یافت نشد")
                return self.stats

            # ریست آمار
            self._init_stats()

            # محاسبه آمار پایه
            self._calculate_basic_metrics(trades)

            # محاسبه معیارهای پیشرفته
            if self.use_advanced_metrics:
                self._calculate_advanced_metrics(trades, days_back)

            # محاسبه حداکثر افت سرمایه (Drawdown)
            self._calculate_drawdown()

            # ذخیره نتیجه در کش
            self._analysis_cache[cache_key] = (self.stats.copy(), datetime.now())

            return self.stats

        except Exception as e:
            logger.error(f"خطا در تحلیل معاملات: {e}")
            return self.stats

    def _calculate_basic_metrics(self, trades: List[Dict[str, Any]]):
        """
        محاسبه معیارهای اصلی تحلیل

        Args:
            trades: لیست معاملات بسته شده
        """
        # محاسبه آمار پایه
        self.stats['total_trades'] = len(trades)
        winning_trades = [t for t in trades if t.get('profit_loss', 0) > 0]
        losing_trades = [t for t in trades if t.get('profit_loss', 0) <= 0]

        self.stats['winning_trades'] = len(winning_trades)
        self.stats['losing_trades'] = len(losing_trades)

        if self.stats['total_trades'] > 0:
            self.stats['win_rate'] = (self.stats['winning_trades'] / self.stats['total_trades']) * 100

        # محاسبه سود و ضرر
        self.stats['total_profit'] = sum(t.get('profit_loss', 0) for t in winning_trades)
        self.stats['total_loss'] = abs(sum(t.get('profit_loss', 0) for t in losing_trades))
        self.stats['net_profit'] = self.stats['total_profit'] - self.stats['total_loss']

        if self.stats['total_loss'] > 0:
            self.stats['profit_factor'] = self.stats['total_profit'] / self.stats['total_loss']

        # محاسبه میانگین‌ها
        if winning_trades:
            self.stats['average_win'] = self.stats['total_profit'] / len(winning_trades)
            self.stats['largest_win'] = max(t.get('profit_loss', 0) for t in winning_trades)

            # محاسبه میانگین مدت معاملات سودده
            win_durations = self._calculate_trade_durations(winning_trades)
            if win_durations:
                self.stats['avg_win_duration'] = sum(win_durations) / len(win_durations)

        if losing_trades:
            self.stats['average_loss'] = self.stats['total_loss'] / len(losing_trades)
            self.stats['largest_loss'] = min(t.get('profit_loss', 0) for t in losing_trades)

            # محاسبه میانگین مدت معاملات زیان‌ده
            loss_durations = self._calculate_trade_durations(losing_trades)
            if loss_durations:
                self.stats['avg_loss_duration'] = sum(loss_durations) / len(loss_durations)

        # محاسبه توالی برد و باخت
        self._calculate_streaks(trades)

        # تحلیل عملکرد بر اساس نماد
        self._analyze_performance_by_category(trades, 'symbol')

        # تعیین بهترین و بدترین نماد
        if self.stats['performance_by_symbol']:
            best_symbol = max(self.stats['performance_by_symbol'].items(),
                              key=lambda x: x[1].get('profit_loss', 0))
            worst_symbol = min(self.stats['performance_by_symbol'].items(),
                               key=lambda x: x[1].get('profit_loss', 0))

            self.stats['best_symbol'] = best_symbol[0]
            self.stats['worst_symbol'] = worst_symbol[0]

    def _calculate_trade_durations(self, trades: List[Dict[str, Any]]) -> List[float]:
        """
        محاسبه مدت زمان معاملات

        Args:
            trades: لیست معاملات

        Returns:
            لیست مدت زمان‌ها به دقیقه
        """
        durations = []
        for trade in trades:
            if trade.get('exit_time') and trade.get('timestamp'):
                try:
                    entry_time = datetime.fromisoformat(trade['timestamp'])
                    exit_time = datetime.fromisoformat(trade['exit_time'])
                    duration_minutes = (exit_time - entry_time).total_seconds() / 60
                    durations.append(duration_minutes)
                except (ValueError, TypeError) as e:
                    logger.warning(f"خطا در محاسبه مدت معامله: {e}")
        return durations

    def _calculate_streaks(self, trades: List[Dict[str, Any]]):
        """
        محاسبه توالی‌های برد و باخت

        Args:
            trades: لیست معاملات به ترتیب زمان
        """
        current_streak = 0
        for i, trade in enumerate(trades):
            if trade.get('profit_loss', 0) > 0:
                if current_streak >= 0:
                    current_streak += 1
                else:
                    current_streak = 1
            else:
                if current_streak <= 0:
                    current_streak -= 1
                else:
                    current_streak = -1

            if current_streak > self.stats['max_consecutive_wins']:
                self.stats['max_consecutive_wins'] = current_streak
            elif current_streak < -self.stats['max_consecutive_losses']:
                self.stats['max_consecutive_losses'] = -current_streak

        # تعیین وضعیت فعلی توالی
        if current_streak > 0:
            self.stats['consecutive_wins'] = current_streak
            self.stats['consecutive_losses'] = 0
        else:
            self.stats['consecutive_wins'] = 0
            self.stats['consecutive_losses'] = -current_streak

    def _analyze_performance_by_category(self, trades: List[Dict[str, Any]], category: str):
        """
        تحلیل عملکرد بر اساس یک دسته‌بندی

        Args:
            trades: لیست معاملات
            category: نام فیلد برای دسته‌بندی
        """
        performance = {}
        for trade in trades:
            category_value = trade.get(category)
            if not category_value:
                continue

            if category_value not in performance:
                performance[category_value] = {
                    'total': 0,
                    'wins': 0,
                    'losses': 0,
                    'profit_loss': 0,
                    'win_rate': 0,
                    'avg_profit': 0,
                    'avg_loss': 0,
                    'profit_factor': 0
                }

            performance[category_value]['total'] += 1
            performance[category_value]['profit_loss'] += trade.get('profit_loss', 0)

            if trade.get('profit_loss', 0) > 0:
                performance[category_value]['wins'] += 1
            else:
                performance[category_value]['losses'] += 1

        # محاسبه آمار اضافی برای هر دسته
        for category_value, perf in performance.items():
            if perf['total'] > 0:
                perf['win_rate'] = (perf['wins'] / perf['total']) * 100

                # محاسبه میانگین سود و ضرر
                win_trades = [t for t in trades if t.get(category) == category_value and t.get('profit_loss', 0) > 0]
                loss_trades = [t for t in trades if t.get(category) == category_value and t.get('profit_loss', 0) <= 0]

                if win_trades:
                    perf['avg_profit'] = sum(t.get('profit_loss', 0) for t in win_trades) / len(win_trades)

                if loss_trades:
                    perf['avg_loss'] = sum(abs(t.get('profit_loss', 0)) for t in loss_trades) / len(loss_trades)

                # محاسبه ضریب سود
                total_profit = sum(t.get('profit_loss', 0) for t in win_trades)
                total_loss = sum(abs(t.get('profit_loss', 0)) for t in loss_trades)

                if total_loss > 0:
                    perf['profit_factor'] = total_profit / total_loss

        # ذخیره در آمار
        self.stats[f'performance_by_{category}'] = performance

    def _calculate_advanced_metrics(self, trades: List[Dict[str, Any]], days_back: int):
        """
        محاسبه معیارهای پیشرفته تحلیل

        Args:
            trades: لیست معاملات
            days_back: تعداد روزهای تحلیل
        """
        # تحلیل بر اساس زمان
        self._analyze_performance_by_time(trades)

        # محاسبه شارپ و سورتینو
        self._calculate_risk_adjusted_metrics(trades)

        # محاسبه امید ریاضی
        if self.stats['win_rate'] > 0 and self.stats['average_win'] > 0 and self.stats['average_loss'] > 0:
            win_rate = self.stats['win_rate'] / 100
            lose_rate = 1 - win_rate
            self.stats['expectancy'] = (win_rate * self.stats['average_win']) - (lose_rate * self.stats['average_loss'])

        # محاسبه فاکتور بازیابی
        if self.stats['max_drawdown'] > 0:
            self.stats['recovery_factor'] = self.stats['net_profit'] / self.stats['max_drawdown']

        # محاسبه ثبات سود
        daily_pnl = self._calculate_daily_pnl(trades)
        if daily_pnl:
            daily_returns = [pnl / self.stats['net_profit'] if self.stats['net_profit'] != 0 else 0 for pnl in
                             daily_pnl.values()]
            if daily_returns:
                self.stats['profit_consistency'] = 1.0 - np.std(daily_returns)

        # میانگین معاملات روزانه
        if days_back > 0:
            self.stats['avg_daily_trades'] = self.stats['total_trades'] / days_back

        # تحلیل بر اساس استراتژی
        self._analyze_performance_by_category(trades, 'setup')
        if self.stats['performance_by_setup']:
            best_setup = max(self.stats['performance_by_setup'].items(),
                             key=lambda x: x[1].get('profit_loss', 0))
            worst_setup = min(self.stats['performance_by_setup'].items(),
                              key=lambda x: x[1].get('profit_loss', 0))

            self.stats['best_setup'] = best_setup[0]
            self.stats['worst_setup'] = worst_setup[0]

        # تحلیل بر اساس تایم‌فریم
        self._analyze_performance_by_category(trades, 'timeframe')

        # محاسبه روند عملکرد
        self._calculate_performance_trend(trades)

    def _analyze_performance_by_time(self, trades: List[Dict[str, Any]]):
        """
        تحلیل عملکرد بر اساس زمان (روز هفته و ساعت روز)

        Args:
            trades: لیست معاملات
        """
        # عملکرد بر اساس روز هفته
        day_performance = {
            'Monday': {'count': 0, 'profit_loss': 0},
            'Tuesday': {'count': 0, 'profit_loss': 0},
            'Wednesday': {'count': 0, 'profit_loss': 0},
            'Thursday': {'count': 0, 'profit_loss': 0},
            'Friday': {'count': 0, 'profit_loss': 0},
            'Saturday': {'count': 0, 'profit_loss': 0},
            'Sunday': {'count': 0, 'profit_loss': 0}
        }

        # عملکرد بر اساس ساعت روز
        hour_performance = {hour: {'count': 0, 'profit_loss': 0} for hour in range(24)}

        # محاسبه زمان در بازار
        total_market_time = 0
        total_elapsed_time = 0

        for trade in trades:
            try:
                # تبدیل زمان‌ها
                entry_time = datetime.fromisoformat(trade['timestamp']) if trade.get('timestamp') else None
                exit_time = datetime.fromisoformat(trade['exit_time']) if trade.get('exit_time') else None

                if entry_time and exit_time:
                    # محاسبه مدت معامله
                    trade_duration = (exit_time - entry_time).total_seconds() / 3600  # ساعت
                    total_market_time += trade_duration

                    # افزودن به آمار روز هفته
                    day_of_week = entry_time.strftime('%A')
                    day_performance[day_of_week]['count'] += 1
                    day_performance[day_of_week]['profit_loss'] += trade.get('profit_loss', 0)

                    # افزودن به آمار ساعت روز
                    hour_of_day = entry_time.hour
                    hour_performance[hour_of_day]['count'] += 1
                    hour_performance[hour_of_day]['profit_loss'] += trade.get('profit_loss', 0)
            except (ValueError, TypeError, KeyError) as e:
                logger.warning(f"خطا در پردازش زمان معامله: {e}")

        # محاسبه درصد زمان در بازار
        first_trade = trades[0] if trades else None
        last_trade = trades[-1] if trades else None

        if first_trade and last_trade and first_trade.get('timestamp') and last_trade.get('exit_time'):
            try:
                start_time = datetime.fromisoformat(first_trade['timestamp'])
                end_time = datetime.fromisoformat(last_trade['exit_time'])
                total_elapsed_time = (end_time - start_time).total_seconds() / 3600  # ساعت

                if total_elapsed_time > 0:
                    self.stats['time_in_market_percent'] = (total_market_time / total_elapsed_time) * 100
            except (ValueError, TypeError) as e:
                logger.warning(f"خطا در محاسبه زمان در بازار: {e}")

        # محاسبه میانگین برای هر دسته
        for day, data in day_performance.items():
            if data['count'] > 0:
                data['avg_pnl'] = data['profit_loss'] / data['count']

        for hour, data in hour_performance.items():
            if data['count'] > 0:
                data['avg_pnl'] = data['profit_loss'] / data['count']

        # ذخیره در آمار
        self.stats['performance_by_day'] = day_performance
        self.stats['performance_by_hour'] = hour_performance

    def _calculate_daily_pnl(self, trades: List[Dict[str, Any]]) -> Dict[str, float]:
        """
        محاسبه سود و زیان روزانه

        Args:
            trades: لیست معاملات

        Returns:
            دیکشنری تاریخ و سود/زیان
        """
        daily_pnl = {}

        for trade in trades:
            if not trade.get('exit_time'):
                continue

            try:
                exit_date = datetime.fromisoformat(trade['exit_time']).date().isoformat()

                if exit_date not in daily_pnl:
                    daily_pnl[exit_date] = 0

                daily_pnl[exit_date] += trade.get('profit_loss', 0)
            except (ValueError, TypeError) as e:
                logger.warning(f"خطا در محاسبه سود/زیان روزانه: {e}")

        return daily_pnl

    def _calculate_risk_adjusted_metrics(self, trades: List[Dict[str, Any]]):
        """
        محاسبه معیارهای تعدیل شده با ریسک (شارپ و سورتینو)

        Args:
            trades: لیست معاملات
        """
        # محاسبه بازده روزانه
        daily_pnl = self._calculate_daily_pnl(trades)
        if not daily_pnl:
            return

        daily_returns = list(daily_pnl.values())

        if len(daily_returns) < 5:  # نیاز به حداقل داده
            return

        # محاسبه شارپ
        avg_return = np.mean(daily_returns)
        std_return = np.std(daily_returns)

        if std_return > 0:
            risk_free_rate = 0  # نرخ بدون ریسک را صفر فرض می‌کنیم
            self.stats['sharpe_ratio'] = (avg_return - risk_free_rate) / std_return

        # محاسبه سورتینو
        downside_returns = [r for r in daily_returns if r < 0]
        if downside_returns:
            downside_deviation = np.std(downside_returns)
            if downside_deviation > 0:
                self.stats['sortino_ratio'] = (avg_return - risk_free_rate) / downside_deviation

    def _calculate_performance_trend(self, trades: List[Dict[str, Any]]):
        """
        محاسبه روند عملکرد در طول زمان

        Args:
            trades: لیست معاملات
        """
        # گروه‌بندی بر اساس هفته
        weekly_performance = {}

        for trade in trades:
            if not trade.get('exit_time'):
                continue

            try:
                exit_time = datetime.fromisoformat(trade['exit_time'])
                year, week, _ = exit_time.isocalendar()
                week_key = f"{year}-W{week:02d}"

                if week_key not in weekly_performance:
                    weekly_performance[week_key] = {
                        'profit_loss': 0,
                        'count': 0,
                        'wins': 0,
                        'losses': 0
                    }

                weekly_performance[week_key]['count'] += 1
                weekly_performance[week_key]['profit_loss'] += trade.get('profit_loss', 0)

                if trade.get('profit_loss', 0) > 0:
                    weekly_performance[week_key]['wins'] += 1
                else:
                    weekly_performance[week_key]['losses'] += 1

            except (ValueError, TypeError) as e:
                logger.warning(f"خطا در محاسبه عملکرد هفتگی: {e}")

        # محاسبه آمار اضافی و مرتب‌سازی بر اساس زمان
        sorted_performance = []
        for week, data in sorted(weekly_performance.items()):
            if data['count'] > 0:
                win_rate = (data['wins'] / data['count']) * 100 if data['count'] > 0 else 0

                sorted_performance.append({
                    'period': week,
                    'profit_loss': data['profit_loss'],
                    'count': data['count'],
                    'win_rate': win_rate
                })

        self.stats['performance_trend'] = sorted_performance

    def _calculate_drawdown(self):
        """محاسبه حداکثر افت سرمایه (Drawdown)"""
        try:
            if not self.conn or not self.cursor:
                return

            self.cursor.execute("SELECT * FROM balance_history ORDER BY timestamp ASC")
            balance_history = [dict(row) for row in self.cursor.fetchall()]

            if balance_history:
                max_balance = balance_history[0].get('balance', 0)
                max_drawdown = 0
                max_drawdown_percent = 0
                current_drawdown = 0
                current_drawdown_percent = 0

                for entry in balance_history:
                    balance = entry.get('balance', 0)
                    if balance > max_balance:
                        max_balance = balance
                        current_drawdown = 0
                        current_drawdown_percent = 0
                    else:
                        current_drawdown = max_balance - balance
                        current_drawdown_percent = (current_drawdown / max_balance) * 100 if max_balance > 0 else 0

                        if current_drawdown > max_drawdown:
                            max_drawdown = current_drawdown
                            max_drawdown_percent = current_drawdown_percent

                self.stats['max_drawdown'] = max_drawdown
                self.stats['max_drawdown_percent'] = max_drawdown_percent

                # ذخیره drawdown فعلی
                self.stats['current_drawdown'] = current_drawdown
                self.stats['current_drawdown_percent'] = current_drawdown_percent
        except Exception as e:
            logger.warning(f"محاسبه drawdown ناموفق بود: {e}")

    @safe_execution(default_value=pd.DataFrame())
    def get_trade_heatmap(self, days_back: int = 90) -> pd.DataFrame:
        """
        ایجاد هیت‌مپ معاملات بر اساس روز هفته و ساعت روز

        Args:
            days_back: تعداد روزهای گذشته برای تحلیل

        Returns:
            دیتافریم برای ساخت هیت‌مپ
        """
        if not self.conn or not self.cursor:
            logger.error("اتصال دیتابیس در دسترس نیست")
            self._connect_to_db()
            if not self.conn:
                return pd.DataFrame()

        try:
            # تعیین تاریخ شروع
            start_date = (datetime.now() - timedelta(days=days_back)).strftime('%Y-%m-%d')

            # دریافت معاملات بسته شده در بازه زمانی
            query = """
            SELECT * FROM trades 
            WHERE status = 'closed' 
            AND exit_time >= ? 
            """

            self.cursor.execute(query, (start_date,))
            trades = [dict(row) for row in self.cursor.fetchall()]

            if not trades:
                logger.info(f"معامله بسته شده‌ای در {days_back} روز گذشته یافت نشد")
                return pd.DataFrame()

            # ساخت دیتافریم
            trade_data = []
            for trade in trades:
                try:
                    entry_time = datetime.fromisoformat(trade['timestamp'])
                    day_of_week = entry_time.strftime('%A')
                    hour_of_day = entry_time.hour
                    profit_loss = trade.get('profit_loss', 0)

                    trade_data.append({
                        'day_of_week': day_of_week,
                        'hour_of_day': hour_of_day,
                        'profit_loss': profit_loss,
                        'is_win': profit_loss > 0
                    })
                except (ValueError, TypeError, KeyError) as e:
                    logger.warning(f"خطا در پردازش معامله برای هیت‌مپ: {e}")

            df = pd.DataFrame(trade_data)
            if df.empty:
                return pd.DataFrame()

            # ایجاد هیت‌مپ PnL میانگین
            days = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
            hours = list(range(24))

            # استفاده از pandas pivot_table برای محاسبه بهینه‌تر
            heatmap_avg_pnl = pd.pivot_table(
                df,
                values='profit_loss',
                index='day_of_week',
                columns='hour_of_day',
                aggfunc='mean',
                fill_value=0
            )

            # ایجاد هیت‌مپ نرخ برد
            heatmap_win_rate = pd.pivot_table(
                df,
                values='is_win',
                index='day_of_week',
                columns='hour_of_day',
                aggfunc=lambda x: (sum(x) / len(x)) * 100 if len(x) > 0 else 0,
                fill_value=0
            )

            # ایجاد هیت‌مپ تعداد معاملات
            heatmap_count = pd.pivot_table(
                df,
                values='profit_loss',
                index='day_of_week',
                columns='hour_of_day',
                aggfunc='count',
                fill_value=0
            )

            # مرتب‌سازی بر اساس روزهای هفته
            if not heatmap_avg_pnl.empty:
                heatmap_avg_pnl = heatmap_avg_pnl.reindex(days)

            if not heatmap_win_rate.empty:
                heatmap_win_rate = heatmap_win_rate.reindex(days)

            if not heatmap_count.empty:
                heatmap_count = heatmap_count.reindex(days)

            # ترکیب همه اطلاعات در یک دیکشنری
            result = {
                'avg_pnl': heatmap_avg_pnl,
                'win_rate': heatmap_win_rate,
                'count': heatmap_count
            }

            # بازگشت میانگین PnL به عنوان نتیجه اصلی (برای سازگاری با کد قبلی)
            return heatmap_avg_pnl

        except Exception as e:
            logger.error(f"خطا در ایجاد هیت‌مپ معاملات: {e}")
            return pd.DataFrame()

    @safe_execution(default_value={})
    def get_backtest_parameters(self) -> Dict[str, Any]:
        """
        استخراج پارامترهای بهینه از نتایج بک‌تست

        Returns:
            دیکشنری پارامترهای بهینه
        """
        try:
            backtest_dir = self.config.get('backtest', {}).get('results_dir', 'backtest_results')
            summary_file = os.path.join(backtest_dir, "grid_search_summary.json")

            if not os.path.exists(summary_file):
                logger.warning(f"فایل خلاصه بک‌تست یافت نشد: {summary_file}")
                return {}

            with open(summary_file, 'r') as f:
                summary = json.load(f)

            best_params = summary.get('best_params', {})

            # افزودن اطلاعات اضافی
            best_params['performance'] = summary.get('best_performance', {})
            best_params['date_generated'] = summary.get('date', datetime.now().isoformat())

            return best_params

        except Exception as e:
            logger.error(f"خطا در دریافت پارامترهای بک‌تست: {e}")
            return {}

    def get_optimization_recommendations(self) -> Dict[str, Any]:
        """
        تولید توصیه‌های بهینه‌سازی بر اساس آمار عملکرد

        Returns:
            دیکشنری توصیه‌های بهینه‌سازی پارامترها
        """
        if not self.stats or self.stats['total_trades'] < self.min_trades_for_analysis:
            return {
                'status': 'insufficient_data',
                'message': f'نیاز به حداقل {self.min_trades_for_analysis} معامله برای تحلیل معتبر'
            }

        recommendations = {
            'status': 'success',
            'parameters': {},
            'explanations': {}
        }

        # 1. توصیه‌های مدیریت ریسک
        if self.stats['win_rate'] < 50:
            # برای نرخ برد پایین، استفاده از RR بالاتر
            recommendations['parameters']['min_risk_reward_ratio'] = 2.5
            recommendations['explanations']['min_risk_reward_ratio'] = (
                f"افزایش نسبت ریسک به ریوارد به دلیل نرخ برد پایین ({self.stats['win_rate']:.1f}%)"
            )
        elif self.stats['win_rate'] > 65:
            # برای نرخ برد بالا، می‌توان ریسک بیشتری پذیرفت
            recommendations['parameters']['max_risk_per_trade_percent'] = 2.0
            recommendations['explanations']['max_risk_per_trade_percent'] = (
                f"افزایش درصد ریسک به دلیل نرخ برد بالا ({self.stats['win_rate']:.1f}%)"
            )

        # 2. توصیه‌های اندازه پوزیشن
        if self.stats['max_drawdown_percent'] > 20:
            # کاهش اندازه پوزیشن در صورت افت سرمایه بالا
            recommendations['parameters']['max_risk_decrease'] = 0.3
            recommendations['explanations']['max_risk_decrease'] = (
                f"کاهش حداکثر ریسک به دلیل افت سرمایه بالا ({self.stats['max_drawdown_percent']:.1f}%)"
            )

        # 3. توصیه‌های استاپ متحرک
        if self.stats['avg_win_duration'] > 0 and self.stats['avg_loss_duration'] > 0:
            win_loss_duration_ratio = self.stats['avg_win_duration'] / self.stats['avg_loss_duration']

            if win_loss_duration_ratio > 3:
                # معاملات سودده بسیار طولانی‌تر از معاملات زیان‌ده
                recommendations['parameters']['trailing_stop_activation_percent'] = 2.25
                recommendations['parameters']['trailing_stop_distance_percent'] = 1.66
                recommendations['explanations']['trailing_stop'] = (
                    "فعال‌سازی سریع‌تر استاپ متحرک به دلیل مدت زمان طولانی معاملات سودده"
                )

        # 4. توصیه‌های مبتنی بر زمان
        best_days = []
        worst_days = []

        if self.stats['performance_by_day']:
            days_performance = [(day, data['profit_loss'])
                                for day, data in self.stats['performance_by_day'].items()
                                if data['count'] >= 5]  # حداقل 5 معامله

            if days_performance:
                days_performance.sort(key=lambda x: x[1], reverse=True)
                best_days = [day for day, _ in days_performance[:2]]
                worst_days = [day for day, _ in days_performance[-2:]]

                if best_days and worst_days:
                    recommendations['parameters']['preferred_trading_days'] = best_days
                    recommendations['parameters']['avoid_trading_days'] = worst_days
                    recommendations['explanations']['trading_days'] = (
                        f"معامله در روزهای {', '.join(best_days)} و اجتناب از روزهای {', '.join(worst_days)}"
                    )

        # 5. توصیه‌های مبتنی بر نماد
        if self.stats['best_symbol'] and self.stats['worst_symbol']:
            recommendations['parameters']['preferred_symbols'] = [self.stats['best_symbol']]
            recommendations['parameters']['avoid_symbols'] = [self.stats['worst_symbol']]
            recommendations['explanations']['symbols'] = (
                f"تمرکز بیشتر بر {self.stats['best_symbol']} و کاهش معاملات {self.stats['worst_symbol']}"
            )

        return recommendations

    def generate_performance_report(self, days_back: int = 30) -> Dict[str, Any]:
        """
        تولید گزارش کامل عملکرد معاملاتی

        Args:
            days_back: تعداد روزهای گذشته برای تحلیل

        Returns:
            دیکشنری گزارش عملکرد
        """
        # تحلیل معاملات
        self.analyze_trades(days_back)

        # ساخت گزارش
        report = {
            "summary": {
                "period": f"{days_back} روز گذشته",
                "total_trades": self.stats['total_trades'],
                "win_rate": f"{self.stats['win_rate']:.2f}%",
                "profit_factor": f"{self.stats['profit_factor']:.2f}",
                "net_profit": f"{self.stats['net_profit']:.2f}",
                "max_drawdown": f"{self.stats['max_drawdown_percent']:.2f}%",
                "best_symbol": self.stats['best_symbol'],
                "worst_symbol": self.stats['worst_symbol'],
                "sharpe_ratio": f"{self.stats['sharpe_ratio']:.2f}",
                "sortino_ratio": f"{self.stats['sortino_ratio']:.2f}",
                "expectancy": f"{self.stats['expectancy']:.2f}",
                "recovery_factor": f"{self.stats['recovery_factor']:.2f}"
            },
            "trades": {
                "winning": {
                    "count": self.stats['winning_trades'],
                    "average_profit": f"{self.stats['average_win']:.2f}",
                    "largest_win": f"{self.stats['largest_win']:.2f}",
                    "average_duration": f"{self.stats['avg_win_duration']:.2f} دقیقه"
                },
                "losing": {
                    "count": self.stats['losing_trades'],
                    "average_loss": f"{self.stats['average_loss']:.2f}",
                    "largest_loss": f"{self.stats['largest_loss']:.2f}",
                    "average_duration": f"{self.stats['avg_loss_duration']:.2f} دقیقه"
                },
                "streaks": {
                    "current_wins": self.stats['consecutive_wins'],
                    "current_losses": self.stats['consecutive_losses'],
                    "max_consecutive_wins": self.stats['max_consecutive_wins'],
                    "max_consecutive_losses": self.stats['max_consecutive_losses']
                }
            },
            "performance": {
                "by_day": self._format_performance_data(self.stats['performance_by_day']),
                "by_hour": self._format_performance_data(self.stats['performance_by_hour']),
                "by_symbol": self._format_performance_data(self.stats['performance_by_symbol']),
                "by_setup": self._format_performance_data(self.stats['performance_by_setup']),
                "trend": self.stats['performance_trend']
            },
            "recommendations": self._generate_trading_recommendations()
        }

        return report

    def _format_performance_data(self, perf_data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        فرمت‌بندی داده‌های عملکرد برای گزارش

        Args:
            perf_data: دیکشنری داده‌های عملکرد

        Returns:
            لیست داده‌های فرمت‌بندی شده
        """
        formatted = []
        for key, data in perf_data.items():
            # تبدیل به فرمت مناسب برای گزارش
            if isinstance(data, dict):
                item = {'name': key}
                item.update({k: f"{v:.2f}" if isinstance(v, float) else v for k, v in data.items()})
                formatted.append(item)

        # مرتب‌سازی بر اساس سودآوری
        if formatted:
            try:
                formatted.sort(key=lambda x: float(x.get('profit_loss', 0))
                if isinstance(x.get('profit_loss'), str)
                else x.get('profit_loss', 0),
                               reverse=True)
            except (ValueError, TypeError):
                pass

        return formatted

    def _generate_trading_recommendations(self) -> List[str]:
        """
        تولید توصیه‌های معاملاتی بر اساس تحلیل‌ها

        Returns:
            لیست توصیه‌ها
        """
        recommendations = []

        # توصیه براساس نرخ برد
        if self.stats['win_rate'] < 40:
            recommendations.append(
                "نرخ برد زیر 40% است. معیارهای ورود را بهبود ببخشید و مکان‌یابی حد ضرر را بررسی کنید.")
        elif self.stats['win_rate'] > 65:
            recommendations.append(
                "نرخ برد عالی بالای 65%. می‌توانید اندازه پوزیشن یا ریسک هر معامله را اندکی افزایش دهید.")

        # توصیه براساس ضریب سود
        if self.stats['profit_factor'] < 1.0:
            recommendations.append(
                "ضریب سود کمتر از 1.0 است. بر کاهش سریع‌تر ضررها و ادامه دادن سودها تمرکز کنید.")
        elif self.stats['profit_factor'] > 2.0:
            recommendations.append(
                "ضریب سود قوی بالای 2.0. استراتژی به خوبی کار می‌کند، رویکرد فعلی را حفظ کنید.")

        # توصیه براساس متوسط سود/ضرر
        if self.stats['average_win'] < abs(self.stats['average_loss']):
            recommendations.append(
                "میانگین ضرر بیشتر از میانگین سود است. با تنظیم حد سودهای بازتر یا حد ضررهای محدودتر، نسبت ریسک به ریوارد را بهبود ببخشید.")

        # توصیه براساس توالی معاملات
        if self.stats['consecutive_losses'] >= 3:
            recommendations.append(
                f"در حال حاضر در یک توالی {self.stats['consecutive_losses']} معامله باخت هستید. اندازه پوزیشن‌ها را موقتاً کاهش دهید.")

        # توصیه براساس افت سرمایه
        if self.stats['max_drawdown_percent'] > 15:
            recommendations.append(
                f"حداکثر افت سرمایه {self.stats['max_drawdown_percent']:.2f}% قابل توجه است. شیوه‌های مدیریت ریسک را بررسی کنید.")

        # توصیه بر اساس روز هفته
        if self.stats['performance_by_day']:
            best_day = max(self.stats['performance_by_day'].items(),
                           key=lambda x: x[1].get('profit_loss', 0) if x[1].get('count', 0) > 3 else -float('inf'),
                           default=(None, None))

            worst_day = min(self.stats['performance_by_day'].items(),
                            key=lambda x: x[1].get('profit_loss', 0) if x[1].get('count', 0) > 3 else float('inf'),
                            default=(None, None))

            if best_day[0] and worst_day[0]:
                recommendations.append(
                    f"عملکرد در روز {best_day[0]} قوی‌ترین و در روز {worst_day[0]} ضعیف‌ترین است. "
                    f"برنامه معاملاتی خود را تنظیم کنید.")

        # توصیه بر اساس زمان روز
        if self.stats['performance_by_hour']:
            best_hours = [(hour, data) for hour, data in self.stats['performance_by_hour'].items()
                          if data.get('count', 0) > 3]

            if best_hours:
                best_hour = max(best_hours, key=lambda x: x[1].get('profit_loss', 0), default=(None, None))
                if best_hour[0] is not None:
                    recommendations.append(
                        f"سودآورترین ساعت معامله حدود {best_hour[0]}:00 است. "
                        f"فعالیت در این زمان را متمرکز کنید.")

        # توصیه‌های عمومی
        if len(recommendations) == 0:
            recommendations.append("عملکرد کلی متعادل است. با استراتژی فعلی ادامه دهید.")

        return recommendations

    def save_analysis_to_db(self, period_days: int = 30) -> bool:
        """
        ذخیره نتایج تحلیل در دیتابیس

        Args:
            period_days: دوره تحلیل به روز

        Returns:
            موفقیت عملیات
        """
        if not self.conn or not self.cursor:
            logger.error("اتصال دیتابیس در دسترس نیست")
            return False

        try:
            # تحلیل معاملات
            self.analyze_trades(period_days)

            # تبدیل به JSON
            stats_json = json.dumps(self.stats)
            current_date = datetime.now().strftime('%Y-%m-%d')

            # ذخیره در دیتابیس
            self.cursor.execute(
                "INSERT INTO trade_stats (date, stats) VALUES (?, ?)",
                (current_date, stats_json)
            )

            self.conn.commit()
            logger.info(f"تحلیل معاملات برای {period_days} روز در دیتابیس ذخیره شد")
            return True

        except Exception as e:
            logger.error(f"خطا در ذخیره تحلیل در دیتابیس: {e}")
            return False


class TradingStrategyOptimizer:
    """
    بهینه‌سازی استراتژی‌های معاملاتی بر اساس داده‌های تاریخی
    """

    def __init__(self, config: Dict[str, Any], trade_analyzer: TradeAnalyzer = None):
        """
        مقداردهی اولیه

        Args:
            config: دیکشنری تنظیمات
            trade_analyzer: نمونه TradeAnalyzer (اختیاری)
        """
        self.config = config
        self.strategy_config = config.get('strategy', {})
        self.analyzer = trade_analyzer

        # تنظیمات بهینه‌سازی
        self.optimization_window_days = self.strategy_config.get('optimization_window_days', 30)
        self.min_trades_for_optimization = self.strategy_config.get('min_trades_for_optimization', 20)
        self.auto_adapt_parameters = self.strategy_config.get('auto_adapt_parameters', True)

        # پارامترهای استراتژی
        self.strategy_parameters = {}
        self._load_initial_parameters()

        # کش آخرین بهینه‌سازی
        self._last_optimization = {
            'timestamp': None,
            'parameters': {},
            'performance': {}
        }

        logger.info("TradingStrategyOptimizer راه‌اندازی شد")

    def update_config(self, new_config: Dict[str, Any]):
        """
        به‌روزرسانی تنظیمات در زمان اجرا

        Args:
            new_config: دیکشنری تنظیمات جدید
        """
        old_config = self.config.copy()
        old_strategy_config = self.strategy_config.copy()

        # به‌روزرسانی تنظیمات اصلی
        self.config = new_config
        self.strategy_config = new_config.get('strategy', {})

        # ذخیره تنظیمات قبلی برای مقایسه
        old_window_days = self.optimization_window_days
        old_min_trades = self.min_trades_for_optimization
        old_auto_adapt = self.auto_adapt_parameters

        # به‌روزرسانی تنظیمات بهینه‌سازی
        self.optimization_window_days = self.strategy_config.get('optimization_window_days', 30)
        self.min_trades_for_optimization = self.strategy_config.get('min_trades_for_optimization', 20)
        self.auto_adapt_parameters = self.strategy_config.get('auto_adapt_parameters', True)

        # لاگ تغییرات مهم
        changes = []
        if old_window_days != self.optimization_window_days:
            changes.append(f"دوره بهینه‌سازی: {old_window_days} → {self.optimization_window_days} روز")

        if old_min_trades != self.min_trades_for_optimization:
            changes.append(f"حداقل معاملات برای بهینه‌سازی: {old_min_trades} → {self.min_trades_for_optimization}")

        if old_auto_adapt != self.auto_adapt_parameters:
            changes.append(f"تطبیق خودکار پارامترها: {'فعال' if self.auto_adapt_parameters else 'غیرفعال'}")

        if changes:
            logger.info(f"تنظیمات TradingStrategyOptimizer به‌روزرسانی شد: {', '.join(changes)}")

            # بارگذاری مجدد پارامترها در صورت تغییر تنظیمات مهم
            if old_auto_adapt != self.auto_adapt_parameters:
                self._load_initial_parameters()
                logger.info("پارامترهای استراتژی مجدداً بارگذاری شدند")
        else:
            logger.debug("تنظیمات TradingStrategyOptimizer به‌روزرسانی شد (بدون تغییر در پارامترهای کلیدی)")

    def _load_initial_parameters(self):
        """بارگذاری پارامترهای اولیه استراتژی"""
        # بارگذاری پارامترهای استراتژی از تنظیمات
        self.strategy_parameters = self.strategy_config.get('parameters', {})

        # بارگذاری پارامترهای بهینه از بک‌تست اگر وجود داشته باشد
        if self.analyzer:
            try:
                backtest_params = self.analyzer.get_backtest_parameters()
                if backtest_params:
                    # ادغام پارامترهای بک‌تست با پارامترهای اولیه
                    for key, value in backtest_params.items():
                        if key in self.strategy_parameters:
                            self.strategy_parameters[key] = value
                    logger.info("پارامترهای بهینه از نتایج بک‌تست بارگذاری شدند")
            except Exception as e:
                logger.warning(f"بارگذاری پارامترهای بک‌تست ناموفق بود: {e}")

    def optimize_parameters(self, recent_trades: List[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        بهینه‌سازی پارامترهای استراتژی بر اساس معاملات اخیر

        Args:
            recent_trades: لیست معاملات اخیر (اختیاری)

        Returns:
            دیکشنری پارامترهای بهینه‌سازی شده
        """
        # اگر به‌روزرسانی خودکار غیرفعال است، پارامترهای فعلی را برگردان
        if not self.auto_adapt_parameters:
            return self.strategy_parameters.copy()

        # اگر معاملات ارائه نشده و تحلیلگر دسترس است، دریافت از تحلیلگر
        if not recent_trades and self.analyzer:
            self.analyzer.analyze_trades(self.optimization_window_days)

            # دریافت معاملات از دیتابیس
            if hasattr(self.analyzer, 'conn') and hasattr(self.analyzer, 'cursor') and self.analyzer.conn:
                try:
                    # تعیین تاریخ شروع
                    start_date = (datetime.now() - timedelta(days=self.optimization_window_days)).strftime(
                        '%Y-%m-%d')

                    # دریافت معاملات بسته شده در بازه زمانی
                    query = """
                        SELECT * FROM trades 
                        WHERE status = 'closed' 
                        AND exit_time >= ? 
                        ORDER BY exit_time ASC
                        """

                    self.analyzer.cursor.execute(query, (start_date,))
                    recent_trades = [dict(row) for row in self.analyzer.cursor.fetchall()]
                except Exception as e:
                    logger.error(f"خطا در دریافت معاملات برای بهینه‌سازی: {e}")
                    recent_trades = []

        # اطمینان از داشتن تعداد کافی معامله
        if not recent_trades or len(recent_trades) < self.min_trades_for_optimization:
            logger.info(
                f"معاملات ناکافی برای بهینه‌سازی. نیاز: {self.min_trades_for_optimization}، موجود: {len(recent_trades) if recent_trades else 0}")
            return self.strategy_parameters.copy()

        # جداسازی معاملات سودده و زیان‌ده
        winning_trades = [t for t in recent_trades if t.get('profit_loss', 0) > 0]
        losing_trades = [t for t in recent_trades if t.get('profit_loss', 0) <= 0]

        # محاسبه آمار اصلی
        win_rate = len(winning_trades) / len(recent_trades) if recent_trades else 0

        # اجرای بهینه‌سازی بر اساس آمار معاملات
        optimized_params = self._adapt_parameters_based_on_statistics(
            win_rate, recent_trades, winning_trades, losing_trades)

        # ذخیره نتایج بهینه‌سازی
        self._last_optimization = {
            'timestamp': datetime.now(),
            'parameters': optimized_params,
            'performance': {
                'win_rate': win_rate,
                'trades_analyzed': len(recent_trades),
                'optimization_window_days': self.optimization_window_days
            }
        }

        # به‌روزرسانی پارامترهای استراتژی
        self.strategy_parameters = optimized_params

        logger.info(f"پارامترهای استراتژی بر اساس {len(recent_trades)} معامله اخیر بهینه‌سازی شدند")
        return optimized_params

    def _adapt_parameters_based_on_statistics(self, win_rate: float,
                                              all_trades: List[Dict[str, Any]],
                                              winning_trades: List[Dict[str, Any]],
                                              losing_trades: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        تطبیق پارامترها بر اساس آمار معاملات

        Args:
            win_rate: نرخ برد
            all_trades: همه معاملات
            winning_trades: معاملات سودده
            losing_trades: معاملات زیان‌ده

        Returns:
            پارامترهای بهینه‌سازی شده
        """
        # کپی پارامترهای فعلی به عنوان پایه
        adapted_params = self.strategy_parameters.copy()

        # تحلیل میانگین مدت معاملات سودده و زیان‌ده
        avg_win_duration = 0
        if winning_trades:
            win_durations = []
            for trade in winning_trades:
                if trade.get('exit_time') and trade.get('timestamp'):
                    try:
                        entry_time = datetime.fromisoformat(trade['timestamp'])
                        exit_time = datetime.fromisoformat(trade['exit_time'])
                        duration_minutes = (exit_time - entry_time).total_seconds() / 60
                        win_durations.append(duration_minutes)
                    except (ValueError, TypeError):
                        pass

            if win_durations:
                avg_win_duration = sum(win_durations) / len(win_durations)

        avg_loss_duration = 0
        if losing_trades:
            loss_durations = []
            for trade in losing_trades:
                if trade.get('exit_time') and trade.get('timestamp'):
                    try:
                        entry_time = datetime.fromisoformat(trade['timestamp'])
                        exit_time = datetime.fromisoformat(trade['exit_time'])
                        duration_minutes = (exit_time - entry_time).total_seconds() / 60
                        loss_durations.append(duration_minutes)
                    except (ValueError, TypeError):
                        pass

            if loss_durations:
                avg_loss_duration = sum(loss_durations) / len(loss_durations)

        # 1. تطبیق پارامترهای تریلینگ استاپ
        if 'trailing_stop_activation_percent' in adapted_params:
            if win_rate < 0.4:
                # نرخ برد پایین - فعال‌سازی زودتر تریلینگ استاپ
                adapted_params['trailing_stop_activation_percent'] = max(0.5,
                                                                         adapted_params.get(
                                                                             'trailing_stop_activation_percent',
                                                                             3.0) * 0.8)
            elif win_rate > 0.6:
                # نرخ برد بالا - تاخیر در فعال‌سازی تریلینگ
                adapted_params['trailing_stop_activation_percent'] = min(2.0,
                                                                         adapted_params.get(
                                                                             'trailing_stop_activation_percent',
                                                                             3.0) * 1.2)

        # 2. تطبیق فاصله تریلینگ استاپ
        if 'trailing_stop_distance_percent' in adapted_params:
            # اگر معاملات سودده سریع بسته می‌شوند، تریلینگ نزدیک‌تر
            if avg_win_duration < avg_loss_duration and avg_win_duration > 0:
                adapted_params['trailing_stop_distance_percent'] = max(0.3,
                                                                       adapted_params.get(
                                                                           'trailing_stop_distance_percent',
                                                                           0.5) * 0.85)
            # اگر معاملات سودده مدت طولانی باز هستند، تریلینگ دورتر
            elif avg_win_duration > avg_loss_duration * 2 and avg_win_duration > 0:
                adapted_params['trailing_stop_distance_percent'] = min(1.0,
                                                                       adapted_params.get(
                                                                           'trailing_stop_distance_percent',
                                                                           0.5) * 1.15)

        # 3. تطبیق نسبت ریسک به ریوارد
        if 'min_risk_reward_ratio' in adapted_params:
            if win_rate < 0.5:
                # نرخ برد کمتر از 50% - افزایش RR
                adapted_params['min_risk_reward_ratio'] = min(3.0,
                                                              adapted_params.get('min_risk_reward_ratio',
                                                                                 1.8) + 0.2)
            elif win_rate > 0.6:
                # نرخ برد بالا - کاهش RR
                adapted_params['min_risk_reward_ratio'] = max(1.5,
                                                              adapted_params.get('min_risk_reward_ratio',
                                                                                 1.8) - 0.1)

        # 4. تطبیق ریسک هر معامله
        if 'max_risk_per_trade_percent' in adapted_params:
            base_risk = adapted_params.get('max_risk_per_trade_percent', 1.5)
            # تعدیل بر اساس نرخ برد و تعداد معاملات
            confidence_factor = (win_rate - 0.5) * 2  # -1 تا 1

            # فقط تغییر کوچک (حداکثر 20%)
            risk_adjustment = 1.0 + (confidence_factor * 0.2)
            new_risk = base_risk * risk_adjustment

            # محدودیت‌ها
            adapted_params['max_risk_per_trade_percent'] = max(0.5, min(new_risk, 2.5))

        # 5. تطبیق بر اساس زمان معاملات
        if 'max_trade_duration_hours' in adapted_params and avg_win_duration > 0:
            # تبدیل به ساعت
            avg_win_hours = avg_win_duration / 60

            # تنظیم حداکثر مدت بر اساس میانگین معاملات سودده
            if avg_win_hours > 0:
                adapted_params['max_trade_duration_hours'] = min(72, max(24, avg_win_hours * 3))

        # 6. تطبیق حساسیت همبستگی
        if 'max_correlation_threshold' in adapted_params:
            # با نرخ برد پایین، همبستگی سختگیرانه‌تر
            if win_rate < 0.45:
                adapted_params['max_correlation_threshold'] = max(0.5,
                                                                  adapted_params.get(
                                                                      'max_correlation_threshold', 0.7) - 0.1)
            # با نرخ برد بالا، انعطاف‌پذیرتر
            elif win_rate > 0.65:
                adapted_params['max_correlation_threshold'] = min(0.85,
                                                                  adapted_params.get(
                                                                      'max_correlation_threshold', 0.7) + 0.05)

        return adapted_params

    def get_optimized_parameters(self) -> Dict[str, Any]:
        """
        دریافت پارامترهای بهینه فعلی

        Returns:
            دیکشنری پارامترهای بهینه
        """
        return self.strategy_parameters.copy()

    def get_last_optimization_info(self) -> Dict[str, Any]:
        """
        دریافت اطلاعات آخرین بهینه‌سازی

        Returns:
            دیکشنری اطلاعات آخرین بهینه‌سازی
        """
        return self._last_optimization.copy()

    def save_parameters_to_file(self, filepath: str = None) -> bool:
        """
        ذخیره پارامترهای بهینه در فایل

        Args:
            filepath: مسیر فایل (اختیاری)

        Returns:
            موفقیت عملیات
        """
        try:
            if not filepath:
                # مسیر پیش‌فرض
                filepath = self.strategy_config.get('parameters_path', 'data/strategy_parameters.json')

            # ساخت دیرکتوری اگر وجود ندارد
            os.makedirs(os.path.dirname(filepath), exist_ok=True)

            # ذخیره پارامترها
            save_data = {
                'parameters': self.strategy_parameters,
                'last_optimization': self._last_optimization,
                'timestamp': datetime.now().isoformat()
            }

            with open(filepath, 'w') as f:
                json.dump(save_data, f, indent=4)

            logger.info(f"پارامترهای استراتژی در {filepath} ذخیره شدند")
            return True

        except Exception as e:
            logger.error(f"خطا در ذخیره پارامترهای استراتژی: {e}")
            return False

    def load_parameters_from_file(self, filepath: str = None) -> bool:
        """
        بارگذاری پارامترهای بهینه از فایل

        Args:
            filepath: مسیر فایل (اختیاری)

        Returns:
            موفقیت عملیات
        """
        try:
            if not filepath:
                # مسیر پیش‌فرض
                filepath = self.strategy_config.get('parameters_path', 'data/strategy_parameters.json')

            if not os.path.exists(filepath):
                logger.warning(f"فایل پارامترها یافت نشد: {filepath}")
                return False

            # بارگذاری پارامترها
            with open(filepath, 'r') as f:
                load_data = json.load(f)

            if 'parameters' in load_data:
                self.strategy_parameters = load_data['parameters']

            if 'last_optimization' in load_data:
                self._last_optimization = load_data['last_optimization']

            logger.info(f"پارامترهای استراتژی از {filepath} بارگذاری شدند")
            return True

        except Exception as e:
            logger.error(f"خطا در بارگذاری پارامترهای استراتژی: {e}")
            return False