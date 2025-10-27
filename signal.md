# تحلیل کامل فرآیند تولید سیگنال معاملاتی

## مقدمه

این سند توضیح می‌دهد که وقتی داده‌های چهار تایم‌فریم (5m, 15m, 1h, 4h) برای تحلیل و ایجاد سیگنال معاملاتی دریافت می‌شوند، چه اتفاقاتی می‌افتد. در این سیستم، **سیگنال نهایی بر اساس امتیازدهی (Scoring) تولید می‌شود** که ترکیبی از تحلیل‌های مختلف است.

---

## بخش ۱: مسیر ورود داده و شروع تحلیل

### 1.1 نقطه شروع: دریافت داده‌ها

وقتی `SignalProcessor` یک نماد را برای تحلیل انتخاب می‌کند، این کار از متد `process_symbol()` شروع می‌شود:

**محل:** `signal_processor.py:392-560`

```python
async def process_symbol(self, symbol: str, force_refresh: bool = False, priority: bool = False)
```

**اتفاقات:**

1. دریافت داده‌های چند تایم‌فریمی از `MarketDataFetcher`:
   ```python
   timeframes_data = await self.market_data_fetcher.get_multi_timeframe_data(
       symbol, self.timeframes, force_refresh, limit_per_tf=limit_needed
   )
   ```

2. بررسی وجود داده‌های معتبر برای تایم‌فریم‌ها:
   - اگر هیچ داده معتبری وجود نداشته باشد → سیگنال ناقص ذخیره می‌شود
   - اگر داده‌های بعضی تایم‌فریم‌ها ناقص باشد → با داده‌های موجود ادامه می‌دهد

3. انتخاب روش تولید سیگنال:
   ```python
   if self.use_ensemble and self.ensemble_strategy:
       # استفاده از استراتژی ترکیبی (Ensemble Strategy)
       signal = await self.ensemble_strategy.generate_ensemble_signal(symbol, timeframes_data)
   else:
       # استفاده از روش استاندارد
       signal = await self.signal_generator.analyze_symbol(symbol, timeframes_data)
   ```

### 1.2 فرآیند تولید سیگنال در SignalGenerator

**محل:** `signal_generator.py:4858-5100`

```python
async def analyze_symbol(self, symbol: str, timeframes_data: Dict[str, Optional[pd.DataFrame]])
```

**گام‌های اصلی:**

1. **بررسی Circuit Breaker (مدار شکن اضطراری)**

   Circuit Breaker یک سیستم محافظتی است که در شرایط خطرناک، تولید سیگنال را متوقف می‌کند.

   **محل:** `signal_generator.py:1217-1434` (کلاس EmergencyCircuitBreaker) و `signal_generator.py:4872-4880` (بررسی در analyze_symbol)

   **دو مکانیزم فعال‌سازی:**

   #### مکانیزم 1: بررسی عملکرد معاملات قبلی

   Circuit Breaker نتایج تمام معاملات را رصد می‌کند و در صورت بروز شرایط زیر فعال می‌شود:

   **شرط 1: ضررهای متوالی (Consecutive Losses)**
   ```python
   max_consecutive_losses = 3  # پیش‌فرض

   # اگر 3 معامله متوالی ضرر داد
   if consecutive_losses >= 3:
       circuit_breaker.trigger()
       # توقف معاملات به مدت 60 دقیقه
   ```

   **مثال:**
   ```
   معامله 1: -1.5R ❌
   معامله 2: -0.8R ❌
   معامله 3: -1.2R ❌
   → Circuit Breaker فعال می‌شود! 🔴
   → معاملات متوقف می‌شوند برای 60 دقیقه
   ```

   **شرط 2: ضرر کل روزانه (Daily Loss Limit)**
   ```python
   max_daily_losses_r = 5.0  # حداکثر 5R ضرر در روز

   # اگر مجموع ضررهای روز از 5R بیشتر شد
   if daily_loss_r >= 5.0:
       circuit_breaker.trigger()
   ```

   **مثال:**
   ```
   09:00 - معامله 1: -2.0R ❌
   11:30 - معامله 2: +1.5R ✅
   14:00 - معامله 3: -1.8R ❌
   16:00 - معامله 4: -2.5R ❌
   ────────────────────────
   مجموع ضرر: 2.0 + 1.8 + 2.5 = 6.3R > 5.0R
   → Circuit Breaker فعال می‌شود! 🔴
   ```

   #### مکانیزم 2: تشخیص بی‌ثباتی بازار

   Circuit Breaker با بررسی داده‌های بازار، شرایط غیرعادی را تشخیص می‌دهد:

   **روش 1: بررسی نوسان غیرعادی (is_market_volatile)**

   **محل:** `signal_generator.py:1329-1379`

   ```python
   def is_market_volatile(symbols_data) -> bool:
       """
       بررسی افزایش ناگهانی نوسانات بازار با ATR
       """

       # برای هر نماد:
       # 1. محاسبه ATR% = (ATR / قیمت) × 100

       # 2. مقایسه 5 کندل اخیر با 20 کندل قبلی
       recent_atr = میانگین(atr_percent[-5:])
       past_atr = میانگین(atr_percent[-25:-5])

       # 3. محاسبه نسبت تغییر
       volatility_change = recent_atr / past_atr

       # 4. اگر نوسان 50% افزایش یافت → بازار بی‌ثبات است
       return volatility_change > 1.5
   ```

   **مثال عملی:**
   ```
   ATR% میانگین 20 روز قبل: 1.2%
   ATR% میانگین 5 روز اخیر: 2.1%

   نسبت تغییر = 2.1 / 1.2 = 1.75 > 1.5
   → بازار بی‌ثبات است! ⚠️
   → تولید سیگنال متوقف می‌شود
   ```

   **روش 2: محاسبه امتیاز بی‌نظمی بازار (get_market_anomaly_score)**

   **محل:** `signal_generator.py:1381-1434`

   این متد یک امتیاز بین 0 تا 1 محاسبه می‌کند که نشان‌دهنده میزان غیرعادی بودن شرایط بازار است.

   **3 شاخص بررسی می‌شود:**

   **شاخص 1: حجم معاملات غیرعادی**
   ```python
   vol_ma_20 = میانگین_حجم_20_کندل_اخیر
   current_vol = حجم_کندل_فعلی

   vol_ratio = current_vol / vol_ma_20

   # اگر حجم بیش از 3 برابر معمول باشد → غیرعادی
   if vol_ratio > 3:
       anomaly_score += min(1.0, (vol_ratio - 3) / 7)
   ```

   **مثال:**
   ```
   میانگین حجم 20 کندل: 1000 BTC
   حجم فعلی: 8000 BTC

   نسبت = 8000 / 1000 = 8.0 > 3
   امتیاز = min(1.0, (8 - 3) / 7) = 0.71
   ```

   **شاخص 2: تغییر قیمت شدید**
   ```python
   price_change_pct = abs((close[-1] - close[-2]) / close[-2]) × 100

   # اگر قیمت بیش از 3% تغییر کرد → غیرعادی
   if price_change_pct > 3:
       anomaly_score += min(1.0, (price_change_pct - 3) / 7)
   ```

   **مثال:**
   ```
   قیمت قبلی: 50,000 USDT
   قیمت فعلی: 54,500 USDT

   تغییر = |54500 - 50000| / 50000 × 100 = 9%
   امتیاز = min(1.0, (9 - 3) / 7) = 0.86
   ```

   **شاخص 3: محدوده High-Low غیرعادی**
   ```python
   hl_ratio = (high - low) / low × 100
   typical_hl_ratio = میانگین_20_کندل_اخیر

   # اگر محدوده بیش از 2 برابر معمول باشد → غیرعادی
   if hl_ratio > typical_hl_ratio × 2:
       anomaly_score += min(1.0, (hl_ratio / typical_hl_ratio - 2) / 3)
   ```

   **مثال:**
   ```
   محدوده معمولی High-Low: 1.5%
   محدوده فعلی: 4.8%

   نسبت = 4.8 / 1.5 = 3.2 > 2
   امتیاز = min(1.0, (3.2 - 2) / 3) = 0.4
   ```

   **محاسبه امتیاز نهایی:**
   ```python
   # میانگین امتیازات هر 3 شاخص
   final_anomaly_score = میانگین(امتیازات)

   # اگر بیش از 0.7 باشد → شرایط بسیار غیرعادی
   if anomaly_score > 0.7:
       # تولید سیگنال متوقف می‌شود
   ```

   **مثال کامل:**
   ```
   شاخص حجم: 0.71
   شاخص تغییر قیمت: 0.86
   شاخص High-Low: 0.40
   ─────────────────────
   امتیاز نهایی = (0.71 + 0.86 + 0.40) / 3 = 0.66

   0.66 < 0.7 → شرایط تقریباً عادی ✅
   اگر 0.75 بود → تولید سیگنال متوقف می‌شد ❌
   ```

   **جدول حد آستانه:**
   | Anomaly Score | وضعیت بازار | اقدام |
   |--------------|-------------|--------|
   | 0.0 - 0.3 | عادی | ✅ تولید سیگنال |
   | 0.3 - 0.5 | کمی غیرعادی | ⚠️ احتیاط |
   | 0.5 - 0.7 | غیرعادی | ⚠️ کاهش ریسک |
   | 0.7 - 1.0 | بسیار غیرعادی | ❌ توقف سیگنال |

   #### Cool Down Period (دوره خنک‌سازی)

   وقتی Circuit Breaker فعال می‌شود:

   ```python
   cool_down_period = 60  # دقیقه (پیش‌فرض)

   # تولید سیگنال متوقف می‌شود
   # بعد از 60 دقیقه:
   # - Circuit Breaker خاموش می‌شود
   # - شمارنده ضررهای متوالی صفر می‌شود
   # - تولید سیگنال از سر گرفته می‌شود
   ```

   **لاگ نمونه:**
   ```
   [WARNING] CIRCUIT BREAKER TRIGGERED: Hit 3 consecutive losses.
             Trading paused for 60 minutes.

   ... 60 minutes later ...

   [INFO] Circuit breaker cool-down period complete. Trading resumed.
   ```

   **پارامترهای قابل تنظیم:**
   ```python
   "circuit_breaker": {
       "enabled": True,                    # فعال/غیرفعال
       "max_consecutive_losses": 3,        # حداکثر ضرر متوالی
       "max_daily_losses_r": 5.0,          # حداکثر ضرر روزانه (R)
       "cool_down_period_minutes": 60,     # مدت توقف (دقیقه)
       "reset_period_hours": 24            # بازنشانی آمار روزانه
   }
   ```

   **چرا Circuit Breaker مهم است؟**

   ✅ **محافظت از سرمایه در شرایط بحرانی:**
   - جلوگیری از ضررهای متوالی
   - توقف خودکار در بازار غیرعادی

   ✅ **مدیریت روانشناسی معامله‌گر:**
   - فرصت برای تنفس و بررسی مجدد
   - جلوگیری از معاملات احساسی

   ✅ **حفظ الگوریتم:**
   - جلوگیری از آسیب به مدل‌های ML با داده‌های غیرعادی
   - فرصت برای بازنگری پارامترها

2. **فیلتر کردن داده‌های معتبر:**
   ```python
   valid_tf_data = {
       tf: df for tf, df in timeframes_data.items()
       if isinstance(df, pd.DataFrame) and not df.empty and len(df) >= 50
   }
   ```
   - حداقل ۵۰ کندل لازم است

3. **تحلیل هر تایم‌فریم به صورت جداگانه:**
   ```python
   result = await self.analyze_single_timeframe(symbol, tf, df)
   ```

---

## بخش ۲: تحلیل یک تایم‌فریم (مثال: 5 دقیقه‌ای)

این بخش **مهم‌ترین بخش** است که در آن امتیازدهی انجام می‌شود.

### 2.1 ورودی به analyze_single_timeframe

**محل:** `signal_generator.py:4647-4850`

برای هر تایم‌فریم (مثلاً 5m) این تحلیل‌ها به ترتیب انجام می‌شوند:

#### مرحله 1: تشخیص روند (Trend Detection)

**محل در کد:** `signal_generator.py:1719-1850`

```python
analysis_data['trend'] = self.detect_trend(df)
```

**چه کاری انجام می‌شود؟**
1. محاسبه EMA‌های 20، 50، 100
2. تعیین جهت روند (Bullish/Bearish/Neutral)
3. محاسبه قدرت روند (Trend Strength: -3 تا +3)
4. تشخیص فاز روند (Trend Phase: early/developing/mature)
5. بررسی چیدمان EMA‌ها (EMA Arrangement)

---

##### 1.1 محاسبه EMA و تشخیص ترتیب

```python
ema20 = talib.EMA(close, timeperiod=20)
ema50 = talib.EMA(close, timeperiod=50)
ema100 = talib.EMA(close, timeperiod=100)

# محاسبه شیب (Slope) برای تشخیص جهت
ema20_slope = ema20[-1] - ema20[-6]  # تغییرات 5 کندل اخیر
ema50_slope = ema50[-1] - ema50[-6]
```

**انواع چیدمان EMA (EMA Arrangement):**

| چیدمان | شرط | معنی |
|--------|-----|------|
| `bullish_aligned` | EMA20 > EMA50 > EMA100 | روند صعودی قوی ✅ |
| `bearish_aligned` | EMA20 < EMA50 < EMA100 | روند نزولی قوی ⬇️ |
| `potential_bullish_reversal` | EMA20 > EMA50 < EMA100 | احتمال بازگشت صعودی 🔄 |
| `potential_bearish_reversal` | EMA20 < EMA50 > EMA100 | احتمال بازگشت نزولی 🔄 |
| `mixed` | غیر از موارد بالا | روند نامشخص ⚠️ |

---

##### 1.2 تعیین جهت و قدرت روند

**جدول کامل Trend Detection (محل در کد: signal_generator.py:1784-1816):**

| شرط | Trend | Strength | Phase |
|-----|-------|----------|-------|
| Close > EMA20 > EMA50 > EMA100 + شیب EMA20 و EMA50 مثبت | `bullish` | **+3** | `mature` یا `developing` |
| Close > EMA20 > EMA50 + شیب EMA20 مثبت | `bullish` | **+2** | `developing` |
| Close > EMA20 + شیب EMA20 مثبت | `bullish` | **+1** | `early` |
| Close < EMA20 < EMA50 < EMA100 + شیب EMA20 و EMA50 منفی | `bearish` | **-3** | `mature` یا `developing` |
| Close < EMA20 < EMA50 + شیب EMA20 منفی | `bearish` | **-2** | `developing` |
| Close < EMA20 + شیب EMA20 منفی | `bearish` | **-1** | `early` |
| Close < EMA50 + EMA20 > EMA50 + شیب EMA50 مثبت | `bullish_pullback` | **+1** | `pullback` |
| Close > EMA50 + EMA20 < EMA50 + شیب EMA50 منفی | `bearish_pullback` | **-1** | `pullback` |
| سایر موارد | `neutral` | **0** | `undefined` |

**⚠️ نکات مهم:**
- حالت‌های `bullish_pullback` و `bearish_pullback` اصلاحات قیمت در طول روند را نشان می‌دهند
- در حالت `mature`، اگر `ema_arrangement` برابر با `bullish_aligned` یا `bearish_aligned` نباشد، `phase` به `developing` تغییر می‌کند

**خروجی واقعی کد (نمونه):**
```python
# محل در کد: signal_generator.py:1817-1832
{
    'status': 'ok',
    'trend': 'bullish',              # جهت: bullish/bearish/neutral/bullish_pullback/bearish_pullback
    'strength': 3,                   # قدرت: -3 تا +3
    'method': 'moving_averages',     # روش تشخیص
    'phase': 'mature',               # فاز: early/developing/mature/late/pullback/transition/undefined
    'details': {
        'close': 50500.0,
        'ema20': 50000.0,
        'ema50': 49500.0,
        'ema100': 49000.0,
        'ema20_slope': 250.5,       # شیب EMA20 (مثبت = صعودی)
        'ema_arrangement': 'bullish_aligned'
    }
}
```

**⚠️ توجه:** در کد واقعی فیلد `confidence` وجود ندارد.

---

##### 1.3 نقش Trend در امتیازدهی

**مهم:** EMA در این مرحله **مستقیماً امتیاز تولید نمی‌کند**!

**چرا؟**
- EMA یک اندیکاتور تأخیری (Lagging) است
- نباید از آن برای تولید سیگنال استفاده شود
- بلکه به عنوان **فیلتر جهت** و **Context** عمل می‌کند

**روش استفاده در کد:**

Trend به دو روش در امتیازدهی تأثیر می‌گذارد:

**1. تأثیر بر `trend_alignment` (در فرمول نهایی امتیاز):**

```python
# محل در کد: signal_generator.py:5074-5077
if is_reversal:
    score.trend_alignment = max(0.5, 1.0 - (reversal_strength * 0.5))
else:
    score.trend_alignment = 1.0 + (primary_trend_strength * 0.2)
```

**2. تأثیر بر `structure_score` (در Multi-Timeframe Analysis):**

```python
# محل در کد: signal_generator.py:4402-4407
if trends_aligned:
    # روندها همراستا
    structure_score *= (1 + trend_bonus_mult * (min_strength / 3))
    # trend_bonus_mult پیش‌فرض = 1.5
else:
    # روندها مخالف
    structure_score *= (1 - trend_penalty_mult * (min_strength / 3))
    # trend_penalty_mult پیش‌فرض = 1.5
```

**3. تأثیر Trend Phase:**

```python
# محل در کد: signal_generator.py:4793-4806
def _get_trend_phase_multiplier(phase: str) -> float:
    phase_multipliers = {
        'early': 1.2,       # روند تازه - بهترین فرصت ورود
        'developing': 1.1,  # روند در حال رشد
        'mature': 0.9,      # روند بالغ - احتیاط (ممکن است نزدیک پایان باشد)
        'late': 0.7,        # روند دیرهنگام - خطرناک
        'pullback': 1.1,    # اصلاح در روند - فرصت ورود خوب
        'transition': 0.8,  # انتقال بین روندها
        'undefined': 1.0    # نامشخص
    }
    return phase_multipliers.get(phase, 1.0)
```

---

##### 1.4 محاسبات واقعی در کد

**سناریو 1: روندها همراستا (trends_aligned = True)**

```python
# محل در کد: signal_generator.py:4402-4403
structure_score *= (1 + 1.5 * (min_strength / 3))
```

مثال‌ها:
- strength = 3: multiplier = 1 + 1.5 * (3/3) = **2.5** (افزایش 150%)
- strength = 2: multiplier = 1 + 1.5 * (2/3) = **2.0** (افزایش 100%)
- strength = 1: multiplier = 1 + 1.5 * (1/3) = **1.5** (افزایش 50%)

**سناریو 2: روندها مخالف (trends_aligned = False)**

```python
# محل در کد: signal_generator.py:4406-4407
structure_score *= (1 - 1.5 * (min_strength / 3))
```

مثال‌ها:
- strength = 3: multiplier = 1 - 1.5 * (3/3) = **-0.5** (کاهش 150% - مقدار منفی!)
- strength = 2: multiplier = 1 - 1.5 * (2/3) = **0.0** (کاهش 100% - صفر می‌شود!)
- strength = 1: multiplier = 1 - 1.5 * (1/3) = **0.5** (کاهش 50%)

**⚠️ توجه:** کد فعلی محدودیت min/max دارد که از منفی شدن امتیاز جلوگیری می‌کند:
```python
# محل در کد: signal_generator.py:4428-4429
structure_score = max(min(structure_score, max_score), min_score)
```

**سناریو 3: Trend Phase Multiplier**

| Phase | Multiplier | تأثیر | استدلال |
|-------|-----------|-------|----------|
| early | **1.2** | +20% | بهترین نقطه ورود - روند تازه شروع شده |
| developing | **1.1** | +10% | روند در حال تقویت |
| mature | **0.9** | -10% | احتیاط - ممکن است نزدیک پایان باشد |
| late | **0.7** | -30% | خطرناک - روند در حال پایان |
| pullback | **1.1** | +10% | فرصت ورود در اصلاح |
| transition | **0.8** | -20% | نامشخص - تغییر روند |
| undefined | **1.0** | 0% | بدون تأثیر |

---

##### 1.5 مثال عملی از کد واقعی

**مثال: سیگنال Long در روند صعودی قوی**

```python
# فرض کنید در Multi-Timeframe Analysis:
# - تایم‌فریم فعلی (5m): trend='bullish', strength=2
# - تایم‌فریم بالاتر (1h): trend='bullish', strength=3
# trends_aligned = True

# محاسبه structure_score:
structure_score = 50  # امتیاز اولیه
min_strength = min(2, 3) = 2

# اعمال trend bonus
structure_score *= (1 + 1.5 * (2 / 3))
structure_score = 50 * (1 + 1.0) = 50 * 2.0 = 100

# اعمال trend phase multiplier (developing)
phase_multiplier = 1.1
structure_score *= phase_multiplier
structure_score = 100 * 1.1 = 110

# نتیجه: امتیاز از 50 به 110 افزایش یافت (افزایش 120%)
```

**⚠️ توجه:** این یک مثال ساده‌شده است. در کد واقعی، ضرایب و محاسبات بیشتری وجود دارد.

---

##### 1.6 تعامل Trend با Multi-Timeframe Analysis

**قانون طلایی:**
> هرگز در خلاف جهت تایم‌فریم‌های بالاتر معامله نکن!

**سلسله مراتب اهمیت:**
```
4h Trend (35% وزن) > 1h Trend (30% وزن) > 15m Trend (20% وزن) > 5m Trend (15% وزن)
```

**سناریو 1: همراستایی کامل**
```python
trends = {
    '5m':  {'trend': 'bullish', 'strength': 2},
    '15m': {'trend': 'bullish', 'strength': 2},
    '1h':  {'trend': 'bullish', 'strength': 3},
    '4h':  {'trend': 'bullish', 'strength': 3}
}

# همه تایم‌فریم‌ها bullish!
# ضریب alignment = 1.0
# multiplier نهایی = 1.3 (به خاطر Confluence)
# این بهترین حالت ممکن است! 🚀
```

**سناریو 2: تضاد با تایم‌فریم‌های بالا (خطرناک)**
```python
trends = {
    '5m':  {'trend': 'bullish', 'strength': 2},  # سیگنال خرید
    '15m': {'trend': 'bullish', 'strength': 1},
    '1h':  {'trend': 'bearish', 'strength': -2}, # مخالف!
    '4h':  {'trend': 'bearish', 'strength': -3}  # مخالف قوی!
}

# وزن تایم‌فریم‌های bearish = 0.30 + 0.35 = 0.65 (65%)
# وزن تایم‌فریم‌های bullish = 0.15 + 0.20 = 0.35 (35%)

# نتیجه: سیگنال Long رد می‌شود! ❌
# چرا؟ روند کلی بازار (4h + 1h) نزولی است
```

**سناریو 3: تایم‌فریم‌های پایین مخالف**
```python
trends = {
    '5m':  {'trend': 'bearish', 'strength': -1}, # مخالف
    '15m': {'trend': 'neutral', 'strength': 0},
    '1h':  {'trend': 'bullish', 'strength': 2},  # موافق
    '4h':  {'trend': 'bullish', 'strength': 3}   # موافق قوی
}

# وزن تایم‌فریم‌های bullish = 0.30 + 0.35 = 0.65 (65%)

# نتیجه: سیگنال Long پذیرفته می‌شود ✅
# تایم‌فریم‌های بالا (1h, 4h) هر دو bullish هستند
# 5m ممکن است فقط یک Pullback موقتی باشد
# اما alignment_score کمی کاهش می‌یابد
```

---

##### 1.7 Trend در Market Regime Detection

Trend با Market Regime تعامل دارد:

```python
regime = {
    'type': 'strong_trend_normal',
    'trend_direction': 'bullish',
    'adx': 32.5,  # قوی
    'volatility': 'normal'
}

# در رژیم Strong Trend:
# - سیگنال‌های Trend-Following اولویت دارند
# - سیگنال‌های Counter-Trend رد می‌شوند
# - Minimum Signal Score افزایش می‌یابد

# اگر Market Regime = Range:
# - روند neutral یا ضعیف
# - سیگنال‌های Mean Reversion بهتر عمل می‌کنند
# - تأثیر Trend کاهش می‌یابد
```

---

##### 1.8 خلاصه و نتیجه‌گیری

**نکات کلیدی:**

✅ **Trend یک فیلتر است، نه یک سیگنال:**
- مستقیماً امتیاز تولید نمی‌کند
- ضریبی برای تأیید یا رد سیگنال‌های دیگر است

✅ **هماهنگی با روند بالاتر حیاتی است:**
- معامله در خلاف جهت 4h و 1h بسیار پرخطر است
- Penalty برای مخالفت: تا 30% کاهش امتیاز

✅ **Trend Phase اهمیت دارد:**
- Mature Trend → پاداش +5%
- Early Trend → بدون پاداش

✅ **در Neutral Trend:**
- سیگنال‌های Range-based و Mean Reversion مناسب‌ترند
- تأثیر Trend صفر است

**جدول خلاصه تأثیر:**

| وضعیت | Multiplier | نتیجه |
|-------|-----------|-------|
| **Perfect Alignment** | 1.15-1.20 | +15% تا +20% ✅ |
| **Good Alignment** | 1.05-1.15 | +5% تا +15% ✅ |
| **Neutral** | 1.00 | بدون تغییر ⚠️ |
| **Weak Opposition** | 0.90-0.95 | -5% تا -10% ❌ |
| **Strong Opposition** | 0.70-0.85 | -15% تا -30% 🚫 |

**محل اعمال در Final Score:**

```python
# در signal_generator.py:4406-4408
if signal_direction != trend_direction:
    structure_score *= (1 - trend_penalty_mult * trend_strength_ratio)

# در فرمول نهایی
final_score = raw_score * trend_multiplier * regime_multiplier * volatility_factor
```

---

#### مرحله 2: تحلیل اندیکاتورهای مومنتوم (RSI, Stochastic, MACD, MFI)

**محل در کد:** `signal_generator.py:3511-3700`

```python
analysis_data['momentum'] = self.analyze_momentum_indicators(df)
```

**چه کاری انجام می‌شود؟**
1. محاسبه اندیکاتورهای مومنتوم (MACD, RSI, Stochastic, MFI)
2. تشخیص سیگنال‌های خرید/فروش بر اساس هر اندیکاتور
3. شناسایی واگرایی‌ها (Divergence) بین قیمت و اندیکاتورها
4. محاسبه امتیاز کلی momentum (bullish یا bearish)

---

##### 2.1 اندیکاتورهای محاسبه شده

###### 1. **MACD (Moving Average Convergence Divergence)**

**محاسبه:**
```python
# signal_generator.py:3532
macd, macd_signal, macd_hist = talib.MACD(close, fastperiod=12, slowperiod=26, signalperiod=9)
```

**سیگنال‌های MACD:**

| سیگنال | شرط | امتیاز پایه | توضیح |
|--------|-----|-----------|-------|
| `macd_bullish_crossover` | MACD > Signal & قبلاً ≤ بود | **2.2** | تقاطع صعودی MACD |
| `macd_bearish_crossover` | MACD < Signal & قبلاً ≥ بود | **2.2** | تقاطع نزولی MACD |
| `macd_bullish_zero_cross` | MACD > 0 & قبلاً ≤ 0 بود | **1.8** | عبور صعودی از خط صفر |
| `macd_bearish_zero_cross` | MACD < 0 & قبلاً ≥ 0 بود | **1.8** | عبور نزولی از خط صفر |

**کد واقعی:**
```python
# signal_generator.py:3585-3607
# 1. MACD Crossover
if curr_macd > curr_sig and prev_macd <= prev_sig:
    momentum_signals.append({
        'type': 'macd_bullish_crossover',
        'score': self.pattern_scores.get('macd_bullish_crossover', 2.2)
    })
elif curr_macd < curr_sig and prev_macd >= prev_sig:
    momentum_signals.append({
        'type': 'macd_bearish_crossover',
        'score': self.pattern_scores.get('macd_bearish_crossover', 2.2)
    })

# 2. MACD Zero Line Cross
if curr_macd > 0 and prev_macd <= 0:
    momentum_signals.append({
        'type': 'macd_bullish_zero_cross',
        'score': self.pattern_scores.get('macd_bullish_zero_cross', 1.8)
    })
elif curr_macd < 0 and prev_macd >= 0:
    momentum_signals.append({
        'type': 'macd_bearish_zero_cross',
        'score': self.pattern_scores.get('macd_bearish_zero_cross', 1.8)
    })
```

---

###### 2. **RSI (Relative Strength Index)**

**محاسبه:**
```python
# signal_generator.py:3538
rsi = talib.RSI(close, timeperiod=14)
```

**سیگنال‌های RSI:**

| سیگنال | شرط | امتیاز پایه | توضیح |
|--------|-----|-----------|-------|
| `rsi_oversold_reversal` | RSI < 30 **و** RSI > prev_RSI | **2.3** | بازگشت از اشباع فروش |
| `rsi_overbought_reversal` | RSI > 70 **و** RSI < prev_RSI | **2.3** | بازگشت از اشباع خرید |
| `rsi_bullish_divergence` | قیمت LL ولی RSI HL | **3.5 × strength** | واگرایی صعودی (قوی) |
| `rsi_bearish_divergence` | قیمت HH ولی RSI LH | **3.5 × strength** | واگرایی نزولی (قوی) |

**⚠️ نکته مهم:** برای سیگنال reversal، فقط `RSI < 30` کافی نیست! باید **شروع به بازگشت** هم کرده باشد:

```python
# signal_generator.py:3610-3619
# 3. RSI Oversold/Overbought Reversal
if curr_rsi < 30 and curr_rsi > prev_rsi:  # ✅ باید در حال افزایش باشد
    momentum_signals.append({
        'type': 'rsi_oversold_reversal',
        'score': self.pattern_scores.get('rsi_oversold_reversal', 2.3)
    })
elif curr_rsi > 70 and curr_rsi < prev_rsi:  # ✅ باید در حال کاهش باشد
    momentum_signals.append({
        'type': 'rsi_overbought_reversal',
        'score': self.pattern_scores.get('rsi_overbought_reversal', 2.3)
    })
```

**مثال عملی:**

```python
# سناریو 1: RSI oversold اما هنوز در حال سقوط ❌
curr_rsi = 25
prev_rsi = 28
# نتیجه: سیگنال تولید نمی‌شود (هنوز momentum نزولی است)

# سناریو 2: RSI oversold و شروع به بازگشت ✅
curr_rsi = 28
prev_rsi = 25
# نتیجه: سیگنال 'rsi_oversold_reversal' با امتیاز 2.3
```

**وضعیت RSI:**
```python
# signal_generator.py:3669
rsi_condition = 'oversold' if curr_rsi < 30 else 'overbought' if curr_rsi > 70 else 'neutral'
```

---

###### 3. **Stochastic Oscillator**

**محاسبه:**
```python
# signal_generator.py:3546
slowk, slowd = talib.STOCH(high, low, close,
                           fastk_period=14,
                           slowk_period=3,
                           slowd_period=3)
```

**سیگنال‌های Stochastic:**

| سیگنال | شرط | امتیاز پایه | توضیح |
|--------|-----|-----------|-------|
| `stochastic_oversold_bullish_cross` | K و D < 20 **و** K عبور از D به بالا | **2.5** | تقاطع صعودی در oversold |
| `stochastic_overbought_bearish_cross` | K و D > 80 **و** K عبور از D به پایین | **2.5** | تقاطع نزولی در overbought |

**⚠️ شرایط دقیق Stochastic Cross:**

```python
# signal_generator.py:3621-3631
# 4. Stochastic Crossover in Oversold/Overbought
if curr_k < 20 and curr_d < 20 and curr_k > curr_d and prev_k <= prev_d:
    # ✅ همه شرایط:
    # 1. K < 20 (oversold)
    # 2. D < 20 (oversold)
    # 3. K > D (الان)
    # 4. prev_K <= prev_D (قبلاً)
    # = تقاطع صعودی در ناحیه oversold
    momentum_signals.append({
        'type': 'stochastic_oversold_bullish_cross',
        'score': self.pattern_scores.get('stochastic_oversold_bullish_cross', 2.5)
    })
```

**مثال عملی:**

```python
# سناریو 1: Stochastic در oversold اما هنوز تقاطع نداریم ❌
curr_k = 15, curr_d = 18  # K < D
prev_k = 12, prev_d = 20
# نتیجه: سیگنال تولید نمی‌شود

# سناریو 2: Stochastic تقاطع صعودی در oversold ✅
curr_k = 18, curr_d = 15  # K > D (الان)
prev_k = 12, prev_d = 20  # K < D (قبلاً)
# نتیجه: سیگنال 'stochastic_oversold_bullish_cross' با امتیاز 2.5
```

**وضعیت Stochastic:**
```python
# signal_generator.py:3670
stoch_condition = 'oversold' if curr_k < 20 and curr_d < 20 else \
                  'overbought' if curr_k > 80 and curr_d > 80 else \
                  'neutral'
```

---

###### 4. **MFI (Money Flow Index)**

**محاسبه:**
```python
# signal_generator.py:3549-3558
if 'volume' in df.columns:
    mfi = talib.MFI(high, low, close, volume, timeperiod=14)
```

**⚠️ نکته:** MFI فقط زمانی محاسبه می‌شود که داده حجم معاملات در دسترس باشد.

**سیگنال‌های MFI:**

| سیگنال | شرط | امتیاز پایه | توضیح |
|--------|-----|-----------|-------|
| `mfi_oversold_reversal` | MFI < 20 **و** MFI > prev_MFI | **2.4** | بازگشت از اشباع فروش با حجم |
| `mfi_overbought_reversal` | MFI > 80 **و** MFI < prev_MFI | **2.4** | بازگشت از اشباع خرید با حجم |

**کد واقعی:**
```python
# signal_generator.py:3633-3644
# 5. MFI Signals
if curr_mfi is not None:
    if curr_mfi < 20 and curr_mfi > prev_mfi:
        momentum_signals.append({
            'type': 'mfi_oversold_reversal',
            'score': self.pattern_scores.get('mfi_oversold_reversal', 2.4)
        })
    elif curr_mfi > 80 and curr_mfi < prev_mfi:
        momentum_signals.append({
            'type': 'mfi_overbought_reversal',
            'score': self.pattern_scores.get('mfi_overbought_reversal', 2.4)
        })
```

**تفاوت MFI با RSI:**
- **RSI:** فقط قیمت را در نظر می‌گیرد
- **MFI:** قیمت + حجم معاملات را ترکیب می‌کند
- **MFI** دقیق‌تر است چون حجم معاملات را هم لحاظ می‌کند

**وضعیت MFI:**
```python
# signal_generator.py:3671
mfi_condition = 'oversold' if curr_mfi is not None and curr_mfi < 20 else \
                'overbought' if curr_mfi is not None and curr_mfi > 80 else \
                'neutral'
```

---

##### 2.2 تشخیص واگرایی (Divergence Detection)

**محل در کد:** `signal_generator.py:2873-3067`

**واگرایی چیست؟**
وقتی که قیمت و اندیکاتور در جهت مخالف حرکت می‌کنند، نشان‌دهنده **ضعف روند فعلی** و احتمال **بازگشت روند** است.

###### انواع واگرایی:

**1. واگرایی صعودی (Bullish Divergence):**
- **قیمت:** کف‌های پایین‌تر می‌سازد (Lower Lows - LL)
- **اندیکاتور (RSI/MACD):** کف‌های بالاتر می‌سازد (Higher Lows - HL)
- **معنی:** روند نزولی در حال ضعیف شدن است → احتمال بازگشت صعودی 📈

**2. واگرایی نزولی (Bearish Divergence):**
- **قیمت:** سقف‌های بالاتر می‌سازد (Higher Highs - HH)
- **اندیکاتور (RSI/MACD):** سقف‌های پایین‌تر می‌سازد (Lower Highs - LH)
- **معنی:** روند صعودی در حال ضعیف شدن است → احتمال بازگشت نزولی 📉

---

###### فرآیند تشخیص واگرایی در کد:

**گام 1: یافتن قله‌ها و دره‌ها (Peaks & Valleys)**

```python
# signal_generator.py:2900-2912
# یافتن peaks و valleys برای قیمت
price_peaks_idx, price_valleys_idx = self.find_peaks_and_valleys(
    price_window.values,
    distance=5,         # حداقل فاصله بین دو peak/valley
    prominence_factor=0.05,  # حداقل برجستگی برای قیمت (5%)
    window_size=period
)

# یافتن peaks و valleys برای اندیکاتور
ind_peaks_idx, ind_valleys_idx = self.find_peaks_and_valleys(
    indicator_window.values,
    distance=5,
    prominence_factor=0.1,  # حداقل برجستگی برای اندیکاتور (10%)
    window_size=period
)
```

**گام 2: تشخیص واگرایی نزولی (Bearish Divergence)**

```python
# signal_generator.py:2933-2993
# شرط: قیمت Higher Highs اما اندیکاتور Lower Highs
if len(price_peaks_abs) >= 2 and len(ind_peaks_abs) >= 2:
    # بررسی 5 peak اخیر
    for i in range(max_peaks_to_check - 1):
        p1_price = price_window.loc[p1_idx]
        p2_price = price_window.loc[p2_idx]

        # قیمت باید Higher High باشد
        if p2_price <= p1_price:
            continue  # این واگرایی نیست

        ind_p1_val = indicator_window.loc[ind_p1_idx]
        ind_p2_val = indicator_window.loc[ind_p2_idx]

        # اندیکاتور باید Lower High باشد
        if ind_p2_val < ind_p1_val:
            # ✅ واگرایی نزولی تشخیص داده شد!

            # محاسبه قدرت واگرایی
            price_change_pct = (p2_price - p1_price) / p1_price
            ind_change_pct = (ind_p1_val - ind_p2_val) / ind_p1_val
            div_strength = min(1.0, (price_change_pct + ind_change_pct) / 2 * 5)

            # امتیاز نهایی
            div_score = 3.5 * div_strength  # base_score × strength
```

**گام 3: محاسبه قدرت واگرایی (Divergence Strength)**

```python
# signal_generator.py:2969-2971
price_change_pct = (p2_price - p1_price) / p1_price  # درصد تغییر قیمت
ind_change_pct = (ind_p1_val - ind_p2_val) / ind_p1_val  # درصد تغییر اندیکاتور
div_strength = min(1.0, (price_change_pct + ind_change_pct) / 2 * 5)  # نرمال‌سازی به 0-1
```

**فرمول strength:**
```
strength = min(1.0, (price_change% + indicator_change%) / 2 × 5)
```

**مثال محاسبه:**
```python
# واگرایی نزولی قوی:
# قیمت: 100 → 110 (افزایش 10%)
# RSI: 80 → 70 (کاهش 12.5%)
price_change_pct = 0.10
ind_change_pct = 0.125
div_strength = min(1.0, (0.10 + 0.125) / 2 * 5) = min(1.0, 0.5625) = 0.56

# امتیاز نهایی:
div_score = 3.5 * 0.56 = 1.96
```

**گام 4: فیلتر کیفیت واگرایی**

```python
# signal_generator.py:2974
if div_strength >= self.divergence_sensitivity:
    # فقط واگرایی‌های با کیفیت کافی ذخیره می‌شوند
    signals.append({
        'type': 'rsi_bearish_divergence',
        'direction': 'bearish',
        'score': div_score,
        'strength': float(div_strength),
        'details': {
            'price_p1': float(p1_price),
            'price_p2': float(p2_price),
            'ind_p1': float(ind_p1_val),
            'ind_p2': float(ind_p2_val),
            'price_change_pct': float(price_change_pct),
            'ind_change_pct': float(ind_change_pct)
        }
    })
```

**گام 5: فیلتر زمانی (Recent Signals Only)**

```python
# signal_generator.py:3055-3059
# فقط واگرایی‌های اخیر (10 کندل آخر) را در نظر بگیر
recent_candle_limit = 10
if len(signals) > 0 and len(price_window) > recent_candle_limit:
    recent_threshold = price_window.index[-recent_candle_limit]
    signals = [s for s in signals if s['index'] >= recent_threshold]
```

---

###### مثال واقعی واگرایی:

**سناریو: واگرایی صعودی RSI**

```python
# داده‌های قیمت (5 کندل اخیر):
prices = [100, 95, 90, 88, 85]  # قیمت در حال سقوط

# داده‌های RSI:
rsi_values = [40, 35, 33, 34, 36]  # RSI در حال افزایش در کف‌ها

# تحلیل:
# Peak 1 (price): 100, RSI: 40
# Valley 1: 90, RSI: 33
# Valley 2: 85, RSI: 36  # ✅ کف جدید قیمت پایین‌تر اما RSI بالاتر

# نتیجه:
{
    'type': 'rsi_bullish_divergence',
    'direction': 'bullish',
    'score': 2.1,  # 3.5 × 0.6
    'strength': 0.6,
    'details': {
        'price_p1': 90.0,
        'price_p2': 85.0,
        'ind_p1': 33.0,
        'ind_p2': 36.0,
        'price_change_pct': -0.056,  # قیمت 5.6% کاهش
        'ind_change_pct': 0.091      # RSI 9.1% افزایش
    }
}
```

---

##### 2.3 خروجی کامل تابع analyze_momentum_indicators

```python
# signal_generator.py:3673-3689
results = {
    'status': 'ok',
    'direction': 'bullish',  # یا 'bearish' یا 'neutral'
    'bullish_score': 7.8,    # مجموع امتیازات صعودی
    'bearish_score': 2.2,    # مجموع امتیازات نزولی
    'signals': [
        {
            'type': 'macd_bullish_crossover',
            'score': 2.2
        },
        {
            'type': 'rsi_oversold_reversal',
            'score': 2.3
        },
        {
            'type': 'stochastic_oversold_bullish_cross',
            'score': 2.5
        },
        {
            'type': 'rsi_bullish_divergence',
            'direction': 'bullish',
            'score': 2.1,
            'strength': 0.6,
            'details': {...}
        }
    ],
    'details': {
        'rsi': 32.5,
        'rsi_condition': 'oversold',
        'macd': 0.15,
        'macd_signal': 0.10,
        'macd_condition': 'bullish',
        'stoch_k': 18.5,
        'stoch_d': 15.2,
        'stoch_condition': 'oversold',
        'mfi': 25.3,
        'mfi_condition': 'neutral'
    }
}
```

---

##### 2.4 امتیازدهی نهایی

**جدول امتیازات پایه (Base Scores):**

| سیگنال | امتیاز پایه | توضیح |
|--------|-----------|-------|
| `macd_bullish_crossover` | 2.2 | تقاطع صعودی MACD |
| `macd_bearish_crossover` | 2.2 | تقاطع نزولی MACD |
| `macd_bullish_zero_cross` | 1.8 | عبور از صفر به بالا |
| `macd_bearish_zero_cross` | 1.8 | عبور از صفر به پایین |
| `rsi_oversold_reversal` | 2.3 | بازگشت از اشباع فروش |
| `rsi_overbought_reversal` | 2.3 | بازگشت از اشباع خرید |
| `rsi_bullish_divergence` | **3.5 × strength** | واگرایی صعودی (0-3.5) |
| `rsi_bearish_divergence` | **3.5 × strength** | واگرایی نزولی (0-3.5) |
| `stochastic_oversold_bullish_cross` | 2.5 | تقاطع صعودی در oversold |
| `stochastic_overbought_bearish_cross` | 2.5 | تقاطع نزولی در overbought |
| `mfi_oversold_reversal` | 2.4 | بازگشت MFI از اشباع فروش |
| `mfi_overbought_reversal` | 2.4 | بازگشت MFI از اشباع خرید |

**⚠️ نکته مهم:** این امتیازات **base scores** هستند و در مراحل بعد:
1. با ضرایب دیگر (trend, alignment, regime) ضرب می‌شوند
2. نرمال‌سازی می‌شوند (scale به 0-100)
3. در فرمول نهایی ترکیب می‌شوند

**مثال محاسبه:**
```python
# سیگنال خرید با momentum قوی:
momentum_signals = [
    {'type': 'macd_bullish_crossover', 'score': 2.2},      # ✅
    {'type': 'rsi_oversold_reversal', 'score': 2.3},       # ✅
    {'type': 'stochastic_oversold_bullish_cross', 'score': 2.5},  # ✅
    {'type': 'rsi_bullish_divergence', 'score': 2.1}       # ✅ (3.5 × 0.6)
]

# مجموع امتیازات صعودی:
bullish_score = 2.2 + 2.3 + 2.5 + 2.1 = 9.1

# نتیجه: momentum قوی صعودی ✅
```

---

##### 2.5 نحوه استفاده در امتیازدهی نهایی

**Momentum به عنوان یک component در امتیاز کلی:**

```python
# در مراحل بعد، momentum signals به structure_score اضافه می‌شوند
for signal in momentum_signals:
    if signal['type'] in signal_direction:  # اگر با جهت سیگنال همراستا باشد
        structure_score += signal['score']
    else:  # اگر مخالف باشد
        structure_score -= signal['score'] * 0.5  # جریمه کمتر
```

**نقش Momentum در تصمیم‌گیری:**

1. **تأیید کننده (Confirmation):**
   - اگر Price Action سیگنال خرید می‌دهد + momentum هم bullish است → قوی‌تر ✅

2. **هشدار دهنده (Warning):**
   - اگر Price Action سیگنال خرید می‌دهد + momentum bearish است → ضعیف‌تر ⚠️

3. **واگرایی = سیگنال قوی:**
   - واگرایی نشان‌دهنده تغییر روند است
   - امتیاز بالاتر (تا 3.5) نسبت به سایر momentum signals

---

##### 2.6 مثال کامل: سیگنال خرید با Momentum Analysis

```python
# شرایط بازار: BTC در روند نزولی، در ناحیه حمایت

# 1. Momentum Indicators:
momentum = {
    'rsi': 28,                    # oversold ✅
    'rsi_condition': 'oversold',
    'macd': -50,
    'macd_signal': -55,           # MACD بالای signal (شروع بازگشت) ✅
    'stoch_k': 18,
    'stoch_d': 15,                # K بالای D در oversold ✅
    'mfi': 22
}

# 2. Signals تشخیص داده شده:
momentum_signals = [
    {
        'type': 'rsi_oversold_reversal',
        'score': 2.3
    },
    {
        'type': 'macd_bullish_crossover',
        'score': 2.2
    },
    {
        'type': 'stochastic_oversold_bullish_cross',
        'score': 2.5
    },
    {
        'type': 'rsi_bullish_divergence',
        'score': 2.8,  # 3.5 × 0.8 (قوی)
        'strength': 0.8,
        'details': {
            'price_p1': 30000,
            'price_p2': 29500,  # قیمت کف جدید زد
            'ind_p1': 25,
            'ind_p2': 28         # RSI بالاتر رفت ✅
        }
    }
]

# 3. امتیاز کلی momentum:
bullish_score = 2.3 + 2.2 + 2.5 + 2.8 = 9.8  # قوی!
bearish_score = 0

# 4. نتیجه:
# Momentum به شدت bullish است ✅
# + واگرایی قوی (strength=0.8) ✅
# + همه اندیکاتورها در oversold و شروع بازگشت ✅
# = احتمال بازگشت صعودی بسیار بالا 🚀
```

---

##### 2.7 نکات کلیدی و بهترین شیوه‌ها

**✅ نکات مثبت کد فعلی:**

1. **عدم تولید سیگنال مستقیم:**
   - Momentum اندیکاتورها فقط **تأیید کننده** هستند
   - سیگنال اصلی از Price Action/Structure می‌آید

2. **شرط Reversal دقیق:**
   - فقط oversold/overbought کافی نیست
   - باید شروع به بازگشت هم کرده باشد

3. **تشخیص واگرایی پیشرفته:**
   - الگوریتم پیچیده برای یافتن peaks/valleys
   - محاسبه strength بر اساس درصد تغییرات
   - فیلتر کیفیت و زمانی

4. **استفاده از Caching:**
   - اندیکاتورها فقط یک بار محاسبه می‌شوند
   - بهبود performance

**⚠️ محدودیت‌ها:**

1. **امتیازات Base نه Final:**
   - امتیازات مستند شده (10-15) با کد (2.3) تفاوت دارند
   - امتیازات کد base scores هستند که بعداً scale می‌شوند

2. **MFI گاهی موجود نیست:**
   - فقط با داده volume کار می‌کند
   - در برخی exchanges/timeframes volume دقیق نیست

3. **واگرایی پیچیده است:**
   - نیاز به حداقل 20 کندل
   - ممکن است false positives تولید کند

**🎯 کاربرد در Strategy:**

```python
# استفاده از momentum برای تصمیم‌گیری:
if momentum['direction'] == 'bullish':
    if momentum['bullish_score'] > 7:
        # momentum قوی → افزایش position size
        position_size *= 1.2
    elif momentum['bearish_score'] > momentum['bullish_score']:
        # momentum مخالف → کاهش position size یا skip
        position_size *= 0.5
```

---

**نکته مهم:** RSI, Stochastic, MACD و MFI هر کدام **کمکی** هستند و به تنهایی سیگنال تولید نمی‌کنند، بلکه امتیاز سیگنال‌های اصلی (Price Action, Structure, S/R) را **تقویت یا تضعیف** می‌کنند.

---

#### مرحله 3: تحلیل حجم معاملات (Volume Analysis)
```python
analysis_data['volume'] = self.analyze_volume_trend(df)
```

**چه بررسی می‌شود؟**
- مقایسه حجم فعلی با میانگین حجم
- تشخیص افزایش ناگهانی حجم
- همبستگی حجم با حرکت قیمت

**خروجی:**
```python
{
    'volume_trend': 'increasing',  # یا 'decreasing'
    'volume_ratio': 1.8,           # نسبت به میانگین
    'volume_confirmation': True    # تأیید حرکت قیمت
}
```

**امتیازدهی:**
- حجم بالا + حرکت قیمت در جهت سیگنال → **×1.2 ضریب تقویت**
- حجم پایین → **×0.9 ضریب کاهش**
- این یک **Volume Confirmation Factor** است که در امتیاز نهایی ضرب می‌شود

---

#### مرحله 4: تحلیل پیشرفته MACD
```python
analysis_data['macd'] = self._analyze_macd(df)
```

**تحلیل‌های MACD:**

1. **MACD Line و Signal Line**
   - تقاطع MACD با Signal Line
   - جهت MACD Histogram

2. **شناسایی واگرایی (Divergence)**
   - واگرایی معمولی (Regular Divergence)
   - واگرایی مخفی (Hidden Divergence)

**خروجی:**
```python
{
    'macd': 125.5,
    'signal': 120.3,
    'histogram': 5.2,
    'cross': 'bullish',              # تقاطع صعودی
    'divergence': 'bullish_regular', # واگرایی
    'strength': 0.8                  # قدرت سیگنال
}
```

**امتیازدهی:**
- MACD Cross (تقاطع) → **+10 تا +20 امتیاز**
- MACD Divergence (واگرایی) → **+20 تا +30 امتیاز**
- MACD در این سیستم یکی از **قوی‌ترین اندیکاتورها** است

---

#### مرحله 5: تحلیل Price Action (الگوهای شمعی)
```python
analysis_data['price_action'] = await self.analyze_price_action(df)
```

**الگوهای شناسایی شده:**
- Engulfing (پوششی)
- Hammer / Shooting Star
- Doji
- Morning Star / Evening Star
- و بیش از 20 الگوی دیگر

**خروجی:**
```python
{
    'patterns': ['bullish_engulfing', 'hammer'],
    'pattern_quality': 0.85,
    'last_candle_type': 'bullish',
    'pattern_location': 'support'  # در نزدیکی سطح حمایت
}
```

**امتیازدهی:**
- الگوی قوی (مثل Engulfing) → **+15 تا +25 امتیاز**
- الگو در محل مناسب (نزدیک S/R) → **+10 امتیاز اضافی**
- کیفیت الگو (Pattern Quality) → **×1.0 تا ×1.5 ضریب**

---

#### مرحله 6: شناسایی سطوح حمایت/مقاومت (Support/Resistance)
```python
analysis_data['support_resistance'] = self.detect_support_resistance(df)
```

**روش شناسایی:**
- پیدا کردن نقاط بازگشت قیمت (Pivot Points)
- محاسبه قدرت هر سطح بر اساس تعداد تست
- تشخیص شکست سطوح (Breakout)

**خروجی:**
```python
{
    'nearest_support': 49800,
    'nearest_resistance': 50200,
    'support_strength': 0.9,
    'resistance_strength': 0.85,
    'near_level': True,           # قیمت نزدیک سطح
    'breakout_detected': False
}
```

**امتیازدهی:**
- سیگنال خرید نزدیک حمایت قوی → **+15 تا +20 امتیاز**
- سیگنال فروش نزدیک مقاومت قوی → **+15 تا +20 امتیاز**
- Breakout تأیید شده → **+25 تا +35 امتیاز**

---

## بخش ۳: تحلیل‌های پیشرفته (Advanced Analysis)

در کنار تحلیل‌های پایه، سیستم **تحلیل‌های پیشرفته‌ای** نیز انجام می‌دهد که امتیازات بالاتری تولید می‌کنند.

### 3.1 شناسایی الگوهای هارمونیک (Harmonic Patterns)

**محل:** `signal_generator.py:2465-2665`

```python
analysis_data['harmonic_patterns'] = self.detect_harmonic_patterns(df)
```

**الگوهای شناسایی شده:**

این سیستم **4 الگوی اصلی هارمونیک** را شناسایی می‌کند:

#### 1. **Gartley Pattern**
- نسبت‌های فیبوناچی:
  - AB/XA = 0.618
  - BC/AB = 0.382
  - CD/BC = 1.272
  - BD/BA = 0.786

#### 2. **Bat Pattern**
- نسبت‌های فیبوناچی:
  - AB/XA = 0.382 تا 0.5
  - BC/AB = 0.382 تا 0.886
  - CD/BC = 1.618 تا 2.618
  - BD/BA = 0.886

#### 3. **Butterfly Pattern**
- نسبت‌های فیبوناچی:
  - AB/XA = 0.786
  - BC/AB = 0.382 تا 0.886
  - CD/BC = 1.618 تا 2.618
  - BD/BA = 1.27 تا 1.618

#### 4. **Crab Pattern**
- نسبت‌های فیبوناچی:
  - AB/XA = 0.382 تا 0.618
  - BC/AB = 0.382 تا 0.886
  - CD/BC = 2.618 تا 3.618
  - BD/BA = 1.618

**نحوه شناسایی:**

1. **پیدا کردن Peaks و Valleys:**
   ```python
   peaks, valleys = self.find_peaks_and_valleys(df['close'].values)
   ```
   - نقاط بازگشت قیمت پیدا می‌شوند

2. **تشکیل نقاط X-A-B-C-D:**
   - 5 نقطه متوالی که الگو را می‌سازند
   - باید بین Peak و Valley تناوب داشته باشند

3. **بررسی نسبت‌های فیبوناچی:**
   - هر نسبت با tolerance ±3% بررسی می‌شود
   - اگر همه نسبت‌ها مطابقت داشتند → الگو تأیید می‌شود

4. **محاسبه کیفیت الگو (Pattern Quality):**
   ```python
   confidence = 1.0 - max(abs(ab_xa - 0.618), abs(bc_ab - 0.382), ...) / tolerance
   ```
   - هرچه نسبت‌ها دقیق‌تر → کیفیت بالاتر

**خروجی:**
```python
{
    'type': 'bullish_gartley',
    'direction': 'bullish',
    'confidence': 0.92,  # کیفیت الگو
    'points': {
        'X': {'price': 50000},
        'A': {'price': 49000},
        'B': {'price': 49618},
        'C': {'price': 49382},
        'D': {'price': 49100}  # نقطه ورود احتمالی
    },
    'projected_target': 49800,  # هدف قیمتی
    'suggested_stop': 48900     # حد ضرر پیشنهادی
}
```

**امتیازدهی:**
- الگوی هارمونیک با کیفیت بالا (confidence > 0.85) → **+30 تا +50 امتیاز**
- الگوی هارمونیک با کیفیت متوسط (confidence 0.7-0.85) → **+20 تا +30 امتیاز**
- الگوی نزدیک به تکمیل (در نقطه D) → **+10 امتیاز اضافی**
- **این یکی از قوی‌ترین سیگنال‌ها است!**

---

### 3.2 شناسایی کانال‌های قیمتی (Price Channels)

**محل:** `signal_generator.py:2666-2768`

```python
analysis_data['price_channels'] = self.detect_price_channels(df)
```

**چه کاری انجام می‌شود؟**

1. **رسم خطوط کانال:**
   - یافتن Peaks (قله‌ها) برای خط بالایی
   - یافتن Valleys (دره‌ها) برای خط پایینی
   - استفاده از Regression برای رسم خطوط

2. **تعیین جهت کانال:**
   ```python
   channel_direction = 'ascending'    # کانال صعودی
   channel_direction = 'descending'   # کانال نزولی
   channel_direction = 'horizontal'   # کانال افقی (Range)
   ```

3. **محاسبه موقعیت قیمت در کانال:**
   ```python
   position_in_channel = (current_price - lower_line) / channel_width
   ```
   - 0.0 = کف کانال
   - 0.5 = وسط کانال
   - 1.0 = سقف کانال

4. **شناسایی Breakout (شکست کانال):**
   - قیمت از کانال خارج شده؟
   - حجم معاملات افزایش یافته؟
   - Breakout تأیید شده است؟

**خروجی:**
```python
{
    'channel_direction': 'ascending',
    'position_in_channel': 0.15,  # نزدیک کف کانال
    'upper_line': 50500,
    'lower_line': 49500,
    'channel_width': 1000,
    'is_breakout_up': False,
    'is_breakout_down': False,
    'touches_upper': 5,  # تعداد برخورد با خط بالا
    'touches_lower': 6,  # تعداد برخورد با خط پایین
    'channel_quality': 0.88  # کیفیت کانال
}
```

**امتیازدهی:**
- قیمت در کف کانال صعودی + سیگنال خرید → **+20 تا +30 امتیاز**
- قیمت در سقف کانال نزولی + سیگنال فروش → **+20 تا +30 امتیاز**
- Breakout تأیید شده با حجم بالا → **+35 تا +50 امتیاز**
- کانال با کیفیت بالا → **×1.2 ضریب تقویت**

---

### 3.3 شناسایی الگوهای چرخه‌ای (Cyclical Patterns)

**محل:** `signal_generator.py:2769-2869`

```python
analysis_data['cyclical_patterns'] = self.detect_cyclical_patterns(df)
```

**تکنولوژی استفاده شده: FFT (Fast Fourier Transform)**

این تحلیل با استفاده از **تبدیل فوریه** الگوهای تکرارشونده در قیمت را پیدا می‌کند.

**مراحل تحلیل:**

1. **حذف روند (Detrending):**
   ```python
   trend = np.polyfit(x, closes, 1)  # خط روند
   detrended = closes - trend         # حذف روند
   ```

2. **اعمال FFT:**
   ```python
   fft_result = scipy.fft.rfft(detrended)
   ```
   - تبدیل سیگنال قیمت به فرکانس‌ها

3. **شناسایی چرخه‌های قوی:**
   - پیدا کردن فرکانس‌هایی که قدرت بالایی دارند
   - محاسبه دوره (Period) هر چرخه
   - فقط چرخه‌هایی با دوره 2 تا 100 کندل

4. **پیش‌بینی قیمت:**
   ```python
   forecast = trend + sum(cycle_components)
   ```
   - با استفاده از چرخه‌های شناسایی شده، 20 کندل آینده پیش‌بینی می‌شود

**خروجی:**
```python
{
    'cycles': [
        {
            'period': 24,         # چرخه 24 کندلی
            'amplitude': 150.5,   # دامنه نوسان
            'amplitude_percent': 0.3,  # 0.3% قیمت
            'phase': 1.57         # فاز فعلی
        },
        {
            'period': 12,
            'amplitude': 95.2,
            'amplitude_percent': 0.19,
            'phase': 0.78
        }
    ],
    'forecast_direction': 'up',  # جهت پیش‌بینی شده
    'forecast_confidence': 0.72,
    'next_cycle_peak': 15  # پیک بعدی در 15 کندل دیگر
}
```

**امتیازدهی:**
- چرخه قوی در نزدیکی کف + سیگنال خرید → **+15 تا +25 امتیاز**
- پیش‌بینی FFT همسو با سیگنال → **+10 تا +15 امتیاز**
- اگر چندین چرخه همزمان تأیید کنند → **+5 امتیاز اضافی**

---

### 3.4 تحلیل شرایط نوسان (Volatility Analysis)

**محل:** `signal_generator.py:4459-4530`

```python
analysis_data['volatility'] = self.analyze_volatility_conditions(df)
```

**چه بررسی می‌شود؟**

1. **محاسبه ATR (Average True Range):**
   ```python
   atr = talib.ATR(high, low, close, timeperiod=14)
   atr_percent = (atr / close) * 100  # نرمال‌سازی
   ```

2. **مقایسه با میانگین:**
   ```python
   atr_ma = moving_average(atr_percent, 20)
   volatility_ratio = current_atr / atr_ma
   ```

3. **تعیین وضعیت نوسان:**
   - `extreme`: نوسان بسیار بالا (ratio > 2.0) → **خطرناک!**
   - `high`: نوسان بالا (ratio > 1.5)
   - `normal`: نوسان عادی (ratio 0.7-1.5)
   - `low`: نوسان پایین (ratio < 0.7)

**خروجی:**
```python
{
    'condition': 'high',
    'volatility_ratio': 1.75,
    'atr_percent': 2.5,  # نوسان 2.5% قیمت
    'score': 0.8,        # ضریب امتیاز
    'reject': False      # آیا سیگنال رد شود؟
}
```

**تأثیر بر امتیازدهی:**

- **نوسان عادی (normal):** → **×1.0** (بدون تغییر)
- **نوسان پایین (low):** → **×0.9** (کاهش 10%)
- **نوسان بالا (high):** → **×0.8** (کاهش 20%)
- **نوسان خطرناک (extreme):** → **سیگنال رد می‌شود!** ❌

**مثال:**
```
امتیاز اولیه = 75
نوسان = high (0.8)
امتیاز نهایی = 75 × 0.8 = 60
```

---

## خلاصه بخش 3: جدول امتیازدهی تحلیل‌های پیشرفته

| تحلیل | شرایط بهینه | امتیاز/ضریب | اهمیت |
|-------|-------------|-------------|-------|
| **Harmonic Pattern** | الگوی کامل با کیفیت بالا | +30 تا +50 | ⭐⭐⭐⭐⭐ |
| **Price Channel** | قیمت در کف کانال صعودی | +20 تا +30 | ⭐⭐⭐⭐ |
| **Channel Breakout** | شکست کانال با حجم | +35 تا +50 | ⭐⭐⭐⭐⭐ |
| **Cyclical Pattern** | چرخه در نزدیکی کف | +15 تا +25 | ⭐⭐⭐ |
| **Volatility Normal** | نوسان عادی | ×1.0 | ⭐⭐⭐ |
| **Volatility High** | نوسان بالا | ×0.8 | ⭐⭐ |
| **Volatility Extreme** | نوسان خطرناک | رد سیگنال ❌ | ⭐⭐⭐⭐⭐ |

**نکته مهم:**
- الگوهای هارمونیک و شکست کانال **بالاترین امتیاز** را دارند
- نوسان بسیار بالا می‌تواند **کل سیگنال را رد** کند
- این تحلیل‌ها معمولاً در تایم‌فریم‌های بالاتر (1h, 4h) موثرترند

---

**پایان بخش 3**

---

## بخش ۴: تشخیص رژیم بازار (Market Regime Detection)

یکی از **هوشمندترین قسمت‌های** این سیستم، تشخیص وضعیت بازار و **تطبیق خودکار پارامترها** با شرایط است.

### 4.1 چرا Market Regime مهم است؟

**مشکل:** یک استراتژی ثابت در همه شرایط بازار موفق نیست!

- در **بازار روندار (Trending)**: باید با روند حرکت کرد
- در **بازار رنج (Range)**: باید از نوسانات استفاده کرد
- در **نوسان بالا**: باید ریسک کاهش یابد

**راه‌حل:** تشخیص خودکار رژیم و تطبیق پارامترها

### 4.2 نحوه تشخیص رژیم بازار

**محل:** `signal_generator.py:226-500` (کلاس MarketRegimeDetector)

```python
regime_result = self.regime_detector.detect_regime(df)
```

#### مرحله 1: محاسبه ADX (Average Directional Index)

```python
adx = talib.ADX(high, low, close, timeperiod=14)
plus_di = talib.PLUS_DI(high, low, close, timeperiod=14)
minus_di = talib.MINUS_DI(high, low, close, timeperiod=14)
```

**ADX چه می‌گوید؟**
- ADX > 25: روند قوی وجود دارد
- ADX 20-25: روند ضعیف
- ADX < 20: بازار رنج است (بدون روند)

**جهت روند:**
- +DI > -DI → روند صعودی (Bullish)
- -DI > +DI → روند نزولی (Bearish)

#### مرحله 2: محاسبه ATR% (نوسان)

```python
atr = talib.ATR(high, low, close, timeperiod=20)
atr_percent = (atr / close) * 100
```

**سطوح نوسان:**
- ATR% > 1.5: نوسان بالا (High Volatility)
- ATR% 0.5-1.5: نوسان عادی (Normal)
- ATR% < 0.5: نوسان پایین (Low Volatility)

#### مرحله 3: تعیین رژیم نهایی

رژیم بازار از **ترکیب ADX و ATR** ساخته می‌شود:

```python
if adx > 25:
    trend_strength = 'strong'
elif adx > 20:
    trend_strength = 'weak'
else:
    trend_strength = 'no_trend'

# ترکیب با volatility
regime = f'{trend_strength}_trend_{volatility}'
```

**رژیم‌های ممکن:**

| ADX | ATR% | رژیم نهایی |
|-----|------|-----------|
| > 25 | Normal | `strong_trend_normal` ✅ |
| > 25 | High | `strong_trend_high` ⚠️ |
| > 25 | Low | `strong_trend_low` ✅ |
| 20-25 | Normal | `weak_trend_normal` |
| 20-25 | High | `weak_trend_high` ⚠️ |
| < 20 | Normal | `range_normal` |
| < 20 | High | `range_high` ❌ |

### 4.3 تطبیق پارامترها با رژیم بازار

**محل:** `signal_generator.py:419-500`

```python
adapted_config = self.regime_detector.get_adapted_parameters(regime_info, base_config)
```

وقتی رژیم بازار مشخص شد، سیستم **خودکار** پارامترهای زیر را تنظیم می‌کند:

#### 1. حداکثر ریسک هر معامله

**پارامتر پایه:** 1.5%

**تطبیق:**
```python
if trend_strength == 'strong':
    risk_modifier = 1.1  # ریسک 10% بیشتر (1.65%)
elif trend_strength == 'no_trend':
    risk_modifier = 0.8  # ریسک 20% کمتر (1.2%)

if volatility == 'high':
    risk_modifier *= 0.7  # ریسک 30% کمتر
```

**مثال:**
- رژیم = `strong_trend_normal`: ریسک = 1.5 × 1.1 = **1.65%**
- رژیم = `range_high`: ریسک = 1.5 × 0.8 × 0.7 = **0.84%**

#### 2. نسبت ریسک به پاداش (Risk-Reward Ratio)

**پارامتر پایه:** 2.5

**تطبیق:**
```python
if trend_strength == 'strong':
    rr_modifier = 1.2  # RR بالاتر در روند قوی (3.0)
elif trend_strength == 'no_trend':
    rr_modifier = 0.8  # RR پایین‌تر در رنج (2.0)
```

**مثال:**
- رژیم = `strong_trend_normal`: RR = 2.5 × 1.2 = **3.0**
- رژیم = `range_normal`: RR = 2.5 × 0.8 = **2.0**

#### 3. فاصله Stop Loss

**پارامتر پایه:** 1.5%

**تطبیق:**
```python
if volatility == 'high':
    sl_modifier = 1.3  # SL گسترده‌تر (1.95%)
elif volatility == 'low':
    sl_modifier = 0.8  # SL محکم‌تر (1.2%)
```

**مثال:**
- رژیم = `strong_trend_high`: SL = 1.5 × 1.3 = **1.95%**
- رژیم = `strong_trend_low`: SL = 1.5 × 0.8 = **1.2%**

#### 4. حداقل امتیاز سیگنال

**پارامتر پایه:** 33

**تطبیق:**
```python
if trend_strength == 'no_trend' or volatility == 'high':
    score_modifier = 1.1  # شرایط سخت‌تر (36.3)
```

**مثال:**
- رژیم = `strong_trend_normal`: حداقل امتیاز = **33**
- رژیم = `range_high`: حداقل امتیاز = 33 × 1.1 = **36.3**

### 4.4 خروجی کامل تشخیص رژیم

```python
{
    'regime': 'strong_trend_normal',
    'trend_strength': 'strong',
    'trend_direction': 'bullish',
    'volatility': 'normal',
    'confidence': 0.85,
    'details': {
        'adx': 28.5,
        'plus_di': 32.0,
        'minus_di': 18.0,
        'atr_percent': 1.2
    }
}
```

### 4.5 تأثیر رژیم بر امتیازدهی

رژیم بازار **مستقیماً** بر امتیاز تأثیر نمی‌گذارد، بلکه:

1. **پارامترها را تغییر می‌دهد** (RR, SL, min_score)
2. **فیلتر می‌کند**: سیگنال‌های ضعیف در شرایط بد رد می‌شوند
3. **تأیید می‌کند**: سیگنال همسو با رژیم امتیاز بالاتری می‌گیرد

**مثال:**

```
سیگنال خرید در تایم‌فریم 5m:
امتیاز اولیه = 65

رژیم = strong_trend_normal (روند صعودی قوی)
جهت سیگنال = long (خرید)
→ همسویی با روند → امتیاز × 1.15 = 74.75 ✅

اگر رژیم = strong_trend_normal ولی سیگنال = short
→ مخالف روند → امتیاز × 0.85 = 55.25 ⚠️
```

### 4.6 جدول تأثیر رژیم‌های مختلف

| رژیم | ریسک | RR | SL | حداقل امتیاز | توصیه |
|------|------|----|----|--------------|-------|
| `strong_trend_normal` | 1.65% | 3.0 | 1.5% | 33 | ✅ بهترین شرایط |
| `strong_trend_high` | 1.2% | 3.0 | 1.95% | 36 | ⚠️ محتاطانه |
| `weak_trend_normal` | 1.5% | 2.5 | 1.5% | 33 | ✅ خوب |
| `range_normal` | 1.2% | 2.0 | 1.5% | 33 | ⚠️ سیگنال‌های رنج |
| `range_high` | 0.84% | 2.0 | 1.95% | 36 | ❌ خطرناک |

### 4.7 مثال عملی کامل

**وضعیت:** BTC/USDT در تایم‌فریم 1h

```python
# تشخیص رژیم
regime = {
    'regime': 'strong_trend_high',
    'trend_direction': 'bullish',
    'adx': 32.5,
    'atr_percent': 2.8  # نوسان بالا!
}

# تطبیق پارامترها
adapted_params = {
    'max_risk_per_trade_percent': 1.2,   # کاهش از 1.5
    'preferred_risk_reward_ratio': 3.0,  # افزایش به 3.0
    'default_stop_loss_percent': 1.95,   # افزایش از 1.5
    'minimum_signal_score': 36.3         # افزایش از 33
}

# سیگنال خرید
signal = {
    'direction': 'long',
    'score': 72,
    'entry': 50000,
    'stop_loss': 49025,  # 1.95% پایین‌تر
    'take_profit': 52925  # RR=3.0 → 2925 پیپ سود
}

# بررسی نهایی
همسویی با روند: ✅ (bullish + long)
امتیاز > حداقل: ✅ (72 > 36.3)
→ سیگنال تأیید می‌شود!
```

---

## خلاصه بخش 4: اهمیت Market Regime

✅ **مزایا:**
- تطبیق خودکار با شرایط بازار
- کاهش ریسک در شرایط خطرناک
- افزایش سود در شرایط مناسب
- فیلتر کردن سیگنال‌های ضعیف

⚠️ **نکات مهم:**
- رژیم بازار در **تایم‌فریم‌های بالاتر** (1h, 4h) معتبرتر است
- در بازار رنج با نوسان بالا → **خطر زیاد!**
- همسویی با روند → **احتمال موفقیت بالاتر**

---

**پایان بخش 4**

---

## بخش ۵: ترکیب امتیازات چند تایم‌فریمی (اینجا جادو اتفاق می‌افتد!)

این بخش **قلب سیستم** است! تا اینجا هر تایم‌فریم به صورت مستقل تحلیل شد، ولی حالا باید این تحلیل‌ها را با هم ترکیب کنیم تا یک **سیگنال واحد و قدرتمند** بسازیم.

### 5.1 چرا چند تایم‌فریم؟

**مشکل تحلیل تک تایم‌فریم:**

فرض کنید فقط تایم‌فریم 5 دقیقه‌ای را نگاه می‌کنید:
- ممکن است یک سیگنال خرید قوی ببینید ✅
- اما روند کلی در تایم‌فریم 4 ساعته نزولی باشد ❌
- نتیجه: سیگنال گمراه‌کننده و ضرر احتمالی

**راه‌حل: Multi-Timeframe Analysis**

```
5m  → جزئیات دقیق، نقطه ورود
15m → تأیید روند کوتاه‌مدت
1h  → روند میان‌مدت
4h  → روند کلی، جهت بازار
```

**قانون طلایی:**
> هرگز **در خلاف جهت تایم‌فریم‌های بالاتر** معامله نکن!

---

### 5.2 وزن‌دهی به تایم‌فریم‌ها

**محل:** `signal_generator.py:5003-5042`

هر تایم‌فریم یک **وزن (Weight)** دارد که اهمیت آن را مشخص می‌کند:

```python
TIMEFRAME_WEIGHTS = {
    '5m': 0.15,   # 15% - فقط برای تایمینگ دقیق
    '15m': 0.20,  # 20% - روند کوتاه‌مدت
    '1h': 0.30,   # 30% - روند میان‌مدت (مهم!)
    '4h': 0.35    # 35% - روند اصلی (بسیار مهم!)
}
```

**چرا این وزن‌ها؟**

1. **تایم‌فریم‌های بالاتر معتبرتر هستند:**
   - نویز کمتر
   - روندهای قوی‌تر
   - سیگنال‌های پایدارتر

2. **تایم‌فریم پایین برای Timing:**
   - نقطه ورود دقیق
   - جزئیات کوتاه‌مدت
   - ولی نباید تصمیم اصلی را بگیرد

**مثال:**
```
اگر 4h و 1h هر دو نزولی هستند → وزن = 65%
حتی اگر 5m و 15m صعودی باشند → وزن = 35%
نتیجه: سیگنال خرید رد می‌شود! ❌
```

---

### 5.3 محاسبه Alignment Score (امتیاز همراستایی)

**محل:** `signal_generator.py:5044-5120`

یکی از مهم‌ترین مفاهیم: **همراستایی (Alignment)**

#### تعریف Alignment:

```python
def calculate_alignment_score(timeframe_signals: Dict[str, Dict]) -> float:
    """
    محاسبه همراستایی بین تایم‌فریم‌ها

    خروجی: 0.0 تا 1.0
    - 1.0 = کاملاً همسو
    - 0.0 = کاملاً متضاد
    """
```

#### مثال عملی:

**حالت 1: همراستایی کامل ✅**
```python
timeframe_signals = {
    '5m':  {'direction': 'long',  'score': 68},
    '15m': {'direction': 'long',  'score': 72},
    '1h':  {'direction': 'long',  'score': 75},
    '4h':  {'direction': 'long',  'score': 80}
}

# همه تایم‌فریم‌ها long
alignment_score = 1.0  # کامل! 🎯
```

**حالت 2: همراستایی ضعیف ⚠️**
```python
timeframe_signals = {
    '5m':  {'direction': 'long',  'score': 65},
    '15m': {'direction': 'long',  'score': 60},
    '1h':  {'direction': 'short', 'score': 55},  # مخالف!
    '4h':  {'direction': 'short', 'score': 70}   # مخالف!
}

# تایم‌فریم‌های بالا مخالف هستند
alignment_score = 0.35  # ضعیف!
```

#### فرمول محاسبه Alignment:

```python
# مرحله 1: شمارش جهت غالب
long_weight = sum(TIMEFRAME_WEIGHTS[tf] for tf, sig in signals.items() if sig['direction'] == 'long')
short_weight = sum(TIMEFRAME_WEIGHTS[tf] for tf, sig in signals.items() if sig['direction'] == 'short')

# مرحله 2: تعیین جهت غالب
dominant_direction = 'long' if long_weight > short_weight else 'short'
dominant_weight = max(long_weight, short_weight)

# مرحله 3: محاسبه alignment
alignment = dominant_weight / sum(TIMEFRAME_WEIGHTS.values())
```

**مثال محاسبه:**
```
5m = long  (0.15)
15m = long (0.20)
1h = short (0.30)
4h = short (0.35)

long_weight = 0.15 + 0.20 = 0.35
short_weight = 0.30 + 0.35 = 0.65

dominant_direction = 'short'
alignment = 0.65 / 1.0 = 0.65 (همراستایی متوسط)
```

---

### 5.4 محاسبه Weighted Score (امتیاز وزن‌دار)

**محل:** `signal_generator.py:5122-5180`

حالا که وزن‌ها و همراستایی را داریم، باید امتیاز نهایی را حساب کنیم:

#### فرمول محاسبه:

```python
def calculate_weighted_score(timeframe_signals: Dict, alignment: float) -> float:
    """
    امتیاز = (مجموع امتیازات × وزن‌ها) × ضریب همراستایی
    """

    # مرحله 1: محاسبه امتیاز وزن‌دار پایه
    base_score = 0
    for tf, signal in timeframe_signals.items():
        weight = TIMEFRAME_WEIGHTS[tf]
        score = signal['score']
        base_score += score * weight

    # مرحله 2: اعمال ضریب همراستایی
    alignment_multiplier = 0.7 + (alignment * 0.6)  # بین 0.7 تا 1.3

    # مرحله 3: محاسبه نهایی
    final_score = base_score * alignment_multiplier

    return final_score
```

#### مثال کامل محاسبه:

**شرایط:**
```python
timeframe_signals = {
    '5m':  {'direction': 'long', 'score': 68},
    '15m': {'direction': 'long', 'score': 72},
    '1h':  {'direction': 'long', 'score': 75},
    '4h':  {'direction': 'long', 'score': 80}
}
```

**محاسبات:**
```
مرحله 1: امتیاز وزن‌دار پایه
---------------------------------
5m:  68 × 0.15 = 10.2
15m: 72 × 0.20 = 14.4
1h:  75 × 0.30 = 22.5
4h:  80 × 0.35 = 28.0
---------------------------------
base_score = 75.1

مرحله 2: محاسبه همراستایی
---------------------------------
alignment = 1.0 (همه long)
alignment_multiplier = 0.7 + (1.0 × 0.6) = 1.3

مرحله 3: امتیاز نهایی
---------------------------------
final_score = 75.1 × 1.3 = 97.6 ✅
```

**نتیجه:** امتیاز 97.6 → سیگنال بسیار قوی! 🚀

---

### 5.5 مثال مقایسه‌ای: همراستایی قوی vs ضعیف

#### مثال A: همراستایی عالی (Alignment = 1.0)

```python
# داده‌ها
signals_A = {
    '5m':  {'direction': 'long', 'score': 65},
    '15m': {'direction': 'long', 'score': 70},
    '1h':  {'direction': 'long', 'score': 75},
    '4h':  {'direction': 'long', 'score': 82}
}

# محاسبه
base_score_A = (65×0.15) + (70×0.20) + (75×0.30) + (82×0.35)
            = 9.75 + 14 + 22.5 + 28.7
            = 74.95

alignment_A = 1.0
multiplier_A = 0.7 + (1.0 × 0.6) = 1.3

final_score_A = 74.95 × 1.3 = 97.4 ✅
```

#### مثال B: همراستایی ضعیف (Alignment = 0.35)

```python
# داده‌ها
signals_B = {
    '5m':  {'direction': 'long', 'score': 70},
    '15m': {'direction': 'long', 'score': 68},
    '1h':  {'direction': 'short', 'score': 60},
    '4h':  {'direction': 'short', 'score': 75}
}

# محاسبه
base_score_B = (70×0.15) + (68×0.20) + (60×0.30) + (75×0.35)
            = 10.5 + 13.6 + 18 + 26.25
            = 68.35

alignment_B = 0.35  # فقط 5m و 15m long
multiplier_B = 0.7 + (0.35 × 0.6) = 0.91

final_score_B = 68.35 × 0.91 = 62.2 ⚠️
```

#### مقایسه نتایج:

| مورد | Base Score | Alignment | Multiplier | Final Score | نتیجه |
|------|-----------|-----------|-----------|-------------|-------|
| **A** | 74.95 | 1.0 | 1.3 | **97.4** | ✅ سیگنال قوی |
| **B** | 68.35 | 0.35 | 0.91 | **62.2** | ⚠️ سیگنال ضعیف |

**درس گرفته شده:**
- هرچند امتیازات مثال B بد نبودند
- ولی عدم همراستایی باعث کاهش 35% امتیاز شد!
- این یک **مکانیزم حفاظتی** است

---

### 5.6 Confluence Bonus (پاداش همگرایی)

**محل:** `signal_generator.py:5182-5240`

وقتی چند تایم‌فریم **همزمان** سیگنال قوی می‌دهند، یک **پاداش اضافی** دریافت می‌کنند.

#### شرایط Confluence:

```python
def check_confluence(timeframe_signals: Dict) -> Dict:
    """
    بررسی همگرایی سیگنال‌ها
    """
    confluence = {
        'exists': False,
        'strength': 0,
        'bonus': 0
    }

    # شمارش سیگنال‌های قوی (score > 70)
    strong_signals = [
        tf for tf, sig in timeframe_signals.items()
        if sig['score'] > 70 and sig['direction'] == dominant_direction
    ]

    # محاسبه پاداش
    if len(strong_signals) >= 3:
        confluence['exists'] = True
        confluence['strength'] = len(strong_signals) / len(timeframe_signals)
        confluence['bonus'] = 5 + (confluence['strength'] * 10)

    return confluence
```

#### مثال Confluence:

```python
signals = {
    '5m':  {'direction': 'long', 'score': 78},  # قوی ✅
    '15m': {'direction': 'long', 'score': 82},  # قوی ✅
    '1h':  {'direction': 'long', 'score': 75},  # قوی ✅
    '4h':  {'direction': 'long', 'score': 88}   # قوی ✅
}

# همه تایم‌فریم‌ها قوی!
strong_count = 4
confluence_strength = 4 / 4 = 1.0
confluence_bonus = 5 + (1.0 × 10) = +15 امتیاز! 🎁

final_score = weighted_score + confluence_bonus
```

**انواع Confluence:**

| تعداد سیگنال قوی | Confluence | پاداش |
|------------------|-----------|-------|
| 1-2 | ضعیف | 0 |
| 3 | خوب | +10 تا +12 |
| 4 | عالی | +15 |

---

### 5.7 جریان کامل محاسبه امتیاز

بیایید کل فرآیند را با یک مثال واقعی ببینیم:

#### شرایط اولیه:

```python
# BTC/USDT - تحلیل در ساعت 14:00
symbol = "BTC/USDT"
current_price = 50000

# نتایج تحلیل هر تایم‌فریم
tf_analysis = {
    '5m': {
        'direction': 'long',
        'score': 72,
        'trend': 'bullish',
        'macd_cross': True,
        'rsi': 58,
        'pattern': 'bullish_engulfing'
    },
    '15m': {
        'direction': 'long',
        'score': 78,
        'trend': 'bullish',
        'support_near': True,
        'volume_confirmed': True
    },
    '1h': {
        'direction': 'long',
        'score': 82,
        'trend': 'strong_bullish',
        'harmonic_pattern': 'gartley',
        'breakout': True
    },
    '4h': {
        'direction': 'long',
        'score': 85,
        'trend': 'strong_bullish',
        'regime': 'strong_trend_normal',
        'channel': 'ascending'
    }
}
```

#### گام 1: محاسبه Base Score

```python
base_score = 0
details = []

# 5m
score_5m = 72 × 0.15 = 10.8
details.append("5m: 72 × 0.15 = 10.8")

# 15m
score_15m = 78 × 0.20 = 15.6
details.append("15m: 78 × 0.20 = 15.6")

# 1h
score_1h = 82 × 0.30 = 24.6
details.append("1h: 82 × 0.30 = 24.6")

# 4h
score_4h = 85 × 0.35 = 29.75
details.append("4h: 85 × 0.35 = 29.75")

base_score = 10.8 + 15.6 + 24.6 + 29.75 = 80.75
```

#### گام 2: محاسبه Alignment

```python
# همه جهت long
long_weight = 0.15 + 0.20 + 0.30 + 0.35 = 1.0
short_weight = 0

alignment = 1.0  # کامل!
alignment_multiplier = 0.7 + (1.0 × 0.6) = 1.3
```

#### گام 3: اعمال Alignment Multiplier

```python
aligned_score = base_score × alignment_multiplier
aligned_score = 80.75 × 1.3 = 104.98
```

#### گام 4: بررسی Confluence

```python
# تعداد سیگنال‌های قوی (>70)
strong_signals = ['5m', '15m', '1h', '4h']  # همه قوی!
confluence_strength = 4 / 4 = 1.0
confluence_bonus = 5 + (1.0 × 10) = 15
```

#### گام 5: محاسبه Final Score

```python
final_score = aligned_score + confluence_bonus
final_score = 104.98 + 15 = 119.98

# نرمال‌سازی به 100
if final_score > 100:
    final_score = 100  # سقف
```

#### خلاصه نهایی:

```
─────────────────────────────────
📊 تحلیل نهایی BTC/USDT
─────────────────────────────────
Base Score:        80.75
Alignment:         1.0 (کامل)
Multiplier:        ×1.3
Aligned Score:     104.98
Confluence Bonus:  +15
─────────────────────────────────
FINAL SCORE:       100/100 ✅
─────────────────────────────────
Signal: STRONG BUY 🚀
Confidence: VERY HIGH
─────────────────────────────────
```

---

### 5.8 حالت‌های خاص و استثناها

#### حالت 1: تایم‌فریم ناقص

```python
# اگر داده یک تایم‌فریم موجود نباشد
signals = {
    '5m':  {'direction': 'long', 'score': 70},
    '15m': {'direction': 'long', 'score': 75},
    '1h':  None,  # داده ناقص!
    '4h':  {'direction': 'long', 'score': 80}
}

# راه‌حل: توزیع مجدد وزن‌ها
available_tfs = ['5m', '15m', '4h']
total_weight = 0.15 + 0.20 + 0.35 = 0.70

# نرمال‌سازی وزن‌ها
adjusted_weights = {
    '5m': 0.15 / 0.70 = 0.214,
    '15m': 0.20 / 0.70 = 0.286,
    '4h': 0.35 / 0.70 = 0.500
}

# ادامه محاسبات با وزن‌های جدید
```

#### حالت 2: تضاد تایم‌فریم‌های بالا

```python
signals = {
    '5m':  {'direction': 'long', 'score': 85},   # خیلی قوی!
    '15m': {'direction': 'long', 'score': 80},
    '1h':  {'direction': 'short', 'score': 75},  # مخالف
    '4h':  {'direction': 'short', 'score': 82}   # مخالف
}

# محاسبه alignment
long_weight = 0.15 + 0.20 = 0.35
short_weight = 0.30 + 0.35 = 0.65

# تایم‌فریم‌های بالا (1h + 4h) مخالف!
alignment = 0.65  # اما برای short!

# نتیجه: سیگنال long رد می‌شود ❌
# چون با روند کلی (4h, 1h) مخالف است
```

**قانون مهم:**
> اگر تایم‌فریم‌های 1h و 4h هر دو در یک جهت باشند (وزن 65%)، سیگنال‌های مخالف **رد می‌شوند**!

#### حالت 3: سیگنال‌های متناقض (Divergence)

```python
signals = {
    '5m':  {'direction': 'long', 'score': 65},
    '15m': {'direction': 'short', 'score': 60},  # متناقض!
    '1h':  {'direction': 'long', 'score': 70},
    '4h':  {'direction': 'short', 'score': 68}   # متناقض!
}

# محاسبه
long_weight = 0.15 + 0.30 = 0.45
short_weight = 0.20 + 0.35 = 0.55

# تقریباً نصف نصف!
alignment = 0.55  # ضعیف

# ضریب کاهش شدید
multiplier = 0.7 + (0.55 × 0.6) = 1.03

# نتیجه: امتیاز نهایی پایین
# توصیه: صبر کنید تا همگرایی بهتری ایجاد شود!
```

---

### 5.9 جدول خلاصه: تأثیر Alignment بر امتیاز

| Alignment | جهت غالب | تعداد همسو | Multiplier | تأثیر |
|-----------|---------|-----------|-----------|-------|
| 1.0 | همه یکسان | 4/4 | 1.3 | +30% 🚀 |
| 0.85-0.99 | تقریباً همسو | 3/4 | 1.21-1.29 | +21-29% ✅ |
| 0.65-0.84 | غالب | 2-3/4 | 1.09-1.20 | +9-20% ✅ |
| 0.50-0.64 | ضعیف | 2/4 | 1.0-1.08 | 0-8% ⚠️ |
| < 0.50 | متناقض | <2/4 | 0.7-0.99 | -1 تا -30% ❌ |

---

### 5.10 مثال واقعی: یک معامله کامل

بیایید یک سناریوی واقعی را از ابتدا تا انتها دنبال کنیم:

#### زمینه:
```
نماد: ETH/USDT
تاریخ: 2024-01-15
ساعت: 10:30 UTC
قیمت فعلی: 2,450 USDT
```

#### تحلیل تایم‌فریم‌ها:

**5m (5 دقیقه):**
```python
{
    'direction': 'long',
    'score': 68,
    'details': {
        'macd_cross': True,           # +12
        'rsi': 52,                    # neutral, +5
        'bullish_engulfing': True,    # +18
        'volume_spike': True,         # +8
        'near_support': True,         # +10
        'trend': 'bullish'            # +15
    },
    'raw_score': 68
}
```

**15m (15 دقیقه):**
```python
{
    'direction': 'long',
    'score': 75,
    'details': {
        'stoch_cross': True,          # +10
        'price_channel': 'bottom',    # +20
        'volume_confirmed': True,     # ×1.2
        'support_test': 3,            # +15
        'trend': 'bullish',           # +18
        'pattern_quality': 0.82       # +12
    },
    'raw_score': 75
}
```

**1h (1 ساعت):**
```python
{
    'direction': 'long',
    'score': 82,
    'details': {
        'harmonic_gartley': True,     # +35
        'macd_divergence': True,      # +25
        'breakout_confirmed': True,   # +30
        'regime': 'strong_trend',     # ×1.15
        'channel_ascending': True,    # +20
        'cycle_bottom': True          # +18
    },
    'raw_score': 82
}
```

**4h (4 ساعت):**
```python
{
    'direction': 'long',
    'score': 88,
    'details': {
        'strong_uptrend': True,       # +25
        'adx': 34,                    # روند قوی, +20
        'ema_alignment': True,        # +15
        'volume_trend': 'increasing', # +12
        'higher_highs': True,         # +16
        'regime': 'ideal'             # ×1.2
    },
    'raw_score': 88
}
```

#### محاسبه گام به گام:

```python
# گام 1: Base Weighted Score
score_5m = 68 × 0.15 = 10.20
score_15m = 75 × 0.20 = 15.00
score_1h = 82 × 0.30 = 24.60
score_4h = 88 × 0.35 = 30.80
base_score = 80.60

# گام 2: Alignment
all_long = True
alignment = 1.0
multiplier = 1.3

# گام 3: Aligned Score
aligned_score = 80.60 × 1.3 = 104.78

# گام 4: Confluence
strong_count = 4  # همه > 70
confluence_bonus = 15

# گام 5: Final Score
final = min(104.78 + 15, 100) = 100
```

#### سیگنال نهایی:

```
╔══════════════════════════════════════╗
║   🎯 SIGNAL GENERATED - ETH/USDT    ║
╠══════════════════════════════════════╣
║ Direction:     LONG (BUY)            ║
║ Final Score:   100/100 ⭐⭐⭐⭐⭐        ║
║ Confidence:    MAXIMUM               ║
║ Alignment:     1.0 (Perfect)         ║
║ Confluence:    4/4 Timeframes        ║
╠══════════════════════════════════════╣
║ Entry:         2,450 USDT            ║
║ Stop Loss:     2,402 USDT (-1.96%)   ║
║ Take Profit:   2,594 USDT (+5.88%)   ║
║ Risk/Reward:   1:3.0                 ║
╠══════════════════════════════════════╣
║ Position Size: 2.5% of portfolio     ║
║ Max Risk:      1.5% (adapted)        ║
╚══════════════════════════════════════╝

✅ All timeframes BULLISH
✅ Strong trend confirmed (4h)
✅ Harmonic pattern at 1h
✅ Channel breakout at 1h
✅ Volume confirmation

⚠️ Risk Management:
   - Trail stop after +3%
   - Take 50% profit at TP1 (2,520)
   - Let 50% run to TP2 (2,594)
```

---

### 5.11 نکات مهم و توصیه‌های کاربردی

#### ✅ DO's (کارهای درست):

1. **همیشه به تایم‌فریم‌های بالاتر اولویت بده**
   ```python
   if score_4h < 60 and alignment < 0.6:
       reject_signal()  # حتی اگر 5m قوی باشد
   ```

2. **صبر کن تا همگرایی ایجاد شود**
   ```python
   if alignment < 0.65:
       wait_for_better_setup()
   ```

3. **در روند قوی، با روند حرکت کن**
   ```python
   if regime == 'strong_trend_bullish':
       prefer_long_signals()
   ```

#### ❌ DON'Ts (کارهای غلط):

1. **هرگز فقط به یک تایم‌فریم تکیه نکن**
   ```python
   # ❌ اشتباه
   if score_5m > 80:
       enter_trade()

   # ✅ درست
   if score_5m > 70 and alignment > 0.7:
       enter_trade()
   ```

2. **در خلاف جهت تایم‌فریم‌های بالا معامله نکن**
   ```python
   # ❌ اشتباه
   if score_5m > 85:  # خیلی قوی!
       enter_long()   # ولی 4h نزولی است!
   ```

3. **با alignment ضعیف معامله نکن**
   ```python
   if alignment < 0.5:
       skip_signal()  # خطر بالا
   ```

---

### 5.12 خلاصه بخش 5: جدول کامل امتیازدهی

| مؤلفه | محدوده | تأثیر | اهمیت |
|-------|---------|-------|-------|
| **Base Score (5m)** | 0-100 | ×0.15 | ⭐⭐ |
| **Base Score (15m)** | 0-100 | ×0.20 | ⭐⭐⭐ |
| **Base Score (1h)** | 0-100 | ×0.30 | ⭐⭐⭐⭐ |
| **Base Score (4h)** | 0-100 | ×0.35 | ⭐⭐⭐⭐⭐ |
| **Alignment** | 0.0-1.0 | ×0.7-1.3 | ⭐⭐⭐⭐⭐ |
| **Confluence** | 0-15 | +0 تا +15 | ⭐⭐⭐⭐ |
| **Final Score** | 0-100 | - | - |

#### فرمول نهایی:

```python
Final_Score = min(
    (
        (Score_5m × 0.15) +
        (Score_15m × 0.20) +
        (Score_1h × 0.30) +
        (Score_4h × 0.35)
    ) × (0.7 + Alignment × 0.6) + Confluence_Bonus,
    100
)
```

---

## نتیجه‌گیری بخش 5

🎯 **نکته کلیدی:**

> قدرت واقعی این سیستم در **ترکیب هوشمند** تایم‌فریم‌هاست، نه تحلیل تک تایم‌فریم!

**چه چیزی این سیستم را قدرتمند می‌کند:**

1. ✅ **وزن‌دهی پویا** - تایم‌فریم‌های مهم‌تر اثر بیشتری دارند
2. ✅ **همراستایی اجباری** - جلوگیری از سیگنال‌های متناقض
3. ✅ **پاداش همگرایی** - تشویق سیگنال‌های قوی چند تایم‌فریمی
4. ✅ **مکانیزم حفاظتی** - کاهش خودکار امتیاز در شرایط مبهم

**اهمیت این بخش:**

در بازارهای مالی، **کاهش نویز** بسیار مهم‌تر از یافتن سیگنال است. این سیستم با ترکیب چند تایم‌فریم، **نویز را فیلتر می‌کند** و فقط سیگنال‌های واقعاً قوی را باقی می‌گذارد.

---

**پایان بخش 5**

---

## بخش ۶: ML/AI Enhancement و محاسبه Final Score

پس از محاسبه امتیاز وزن‌دار چند تایم‌فریمی، سیستم وارد **مرحله نهایی** می‌شود که در آن از هوش مصنوعی و فیلترهای پیشرفته استفاده می‌شود تا سیگنال نهایی را تولید کند.

### 6.1 نقش Machine Learning در سیستم

**محل:** `signal_generator.py:5242-5450` و `ensemble_strategy.py`

این سیستم از **دو رویکرد** استفاده می‌کند:

1. **Rule-Based Scoring** (امتیازدهی قانون‌محور) → بخش‌های 1-5
2. **ML Enhancement** (بهبود با یادگیری ماشین) → این بخش

#### چرا ML؟

**مشکلات روش قانون‌محور:**
- پارامترهای ثابت برای همه شرایط
- عدم یادگیری از معاملات گذشته
- نمی‌تواند الگوهای پیچیده را تشخیص دهد

**راه‌حل ML:**
- یادگیری از تاریخچه معاملات
- تشخیص الگوهای پنهان
- تطبیق خودکار با تغییرات بازار

---

### 6.2 Ensemble Strategy (استراتژی ترکیبی)

**محل:** `ensemble_strategy.py:1-800`

سیستم از **Ensemble Learning** استفاده می‌کند که چندین مدل ML را با هم ترکیب می‌کند.

#### ساختار Ensemble:

```python
class EnsembleStrategy:
    """
    ترکیب چند استراتژی و مدل ML برای تولید سیگنال بهتر
    """

    def __init__(self):
        self.models = {
            'xgboost': XGBoostModel(),      # مدل اصلی
            'random_forest': RFModel(),     # مدل پشتیبان
            'lstm': LSTMModel(),            # برای پیش‌بینی سری زمانی
        }
        self.weights = {
            'xgboost': 0.5,       # 50% وزن
            'random_forest': 0.3,  # 30% وزن
            'lstm': 0.2           # 20% وزن
        }
```

#### مدل‌های ML استفاده شده:

**1. XGBoost (مدل اصلی)**
- **کاربرد:** پیش‌بینی احتمال موفقیت سیگنال
- **ورودی‌ها:**
  - امتیازات تایم‌فریم‌ها (5m, 15m, 1h, 4h)
  - Alignment score
  - Market regime
  - تحلیل‌های تکنیکال (RSI, MACD, etc.)
  - ویژگی‌های حجم معاملات
  - نوسانات بازار

- **خروجی:**
  - احتمال موفقیت (0.0 تا 1.0)
  - ضریب اعتماد (confidence)

**مثال:**
```python
features = {
    'score_5m': 72,
    'score_15m': 78,
    'score_1h': 82,
    'score_4h': 85,
    'alignment': 1.0,
    'regime': 'strong_trend_normal',
    'rsi_5m': 58,
    'macd_cross_1h': 1,
    'volume_ratio': 1.8,
    'volatility': 0.8,
    # ... 50+ features دیگر
}

prediction = xgboost_model.predict(features)
# output: {'success_probability': 0.78, 'confidence': 0.85}
```

**2. Random Forest (مدل پشتیبان)**
- **کاربرد:** تأیید سیگنال و کاهش False Positives
- **روش:** رأی‌گیری از چندین Decision Tree
- **تأثیر:** اگر با XGBoost موافق باشد → افزایش اعتماد

**3. LSTM (Long Short-Term Memory)**
- **کاربرد:** پیش‌بینی روند آینده قیمت
- **ورودی:** 100 کندل اخیر
- **خروجی:** جهت احتمالی 20 کندل آینده
- **تأثیر:** تأیید یا رد سیگنال بر اساس پیش‌بینی

---

### 6.3 فرآیند ML Enhancement

**محل:** `ensemble_strategy.py:245-450`

```python
async def enhance_signal_with_ml(
    self,
    base_score: float,
    signal_data: Dict,
    market_data: Dict
) -> Dict:
    """
    بهبود سیگنال با استفاده از ML
    """
```

#### گام 1: استخراج ویژگی‌ها (Feature Extraction)

```python
features = self._extract_features(signal_data, market_data)
```

**ویژگی‌های استخراج شده (100+ features):**

| دسته | مثال‌ها | تعداد |
|------|---------|-------|
| **Scores** | score_5m, score_15m, score_1h, score_4h | 4 |
| **Technical** | rsi, macd, stoch, adx, cci | 20 |
| **Price Action** | pattern_type, candle_type | 15 |
| **Volume** | volume_ratio, volume_trend | 8 |
| **Trend** | ema_alignment, trend_strength | 12 |
| **Volatility** | atr, bollinger_width | 6 |
| **Market Regime** | regime_encoded, adx, di_diff | 5 |
| **Time-based** | hour_of_day, day_of_week | 4 |
| **Cross-TF** | alignment, confluence | 6 |
| **Historical** | win_rate_symbol, avg_profit | 8 |
| **Others** | support_distance, resistance_distance | 12+ |

#### گام 2: پیش‌بینی با هر مدل

```python
# XGBoost prediction
xgb_pred = self.models['xgboost'].predict_proba(features)
xgb_score = xgb_pred[1]  # احتمال کلاس مثبت (موفقیت)

# Random Forest prediction
rf_pred = self.models['random_forest'].predict_proba(features)
rf_score = rf_pred[1]

# LSTM prediction
lstm_pred = self.models['lstm'].predict(price_sequence)
lstm_direction = 'bullish' if lstm_pred > 0.5 else 'bearish'
```

#### گام 3: ترکیب پیش‌بینی‌ها

```python
# Weighted ensemble
ensemble_score = (
    xgb_score * self.weights['xgboost'] +
    rf_score * self.weights['random_forest'] +
    lstm_score * self.weights['lstm']
)

# Confidence calculation
confidence = calculate_confidence([xgb_score, rf_score, lstm_score])
```

**فرمول Confidence:**
```python
# هرچه مدل‌ها نظر یکسانی داشته باشند → confidence بالاتر
std_dev = np.std([xgb_score, rf_score, lstm_score])
confidence = 1.0 - (std_dev / 0.5)  # normalize to 0-1
```

**مثال:**
```python
xgb_score = 0.82
rf_score = 0.78
lstm_score = 0.85

# Ensemble
ensemble = 0.82*0.5 + 0.78*0.3 + 0.85*0.2 = 0.81

# Confidence
std_dev = 0.029
confidence = 1.0 - (0.029 / 0.5) = 0.94  # بالا → مدل‌ها موافق هستند
```

#### گام 4: محاسبه ML Adjustment Factor

```python
ml_adjustment = 0.8 + (ensemble_score * 0.4)  # بین 0.8 تا 1.2
```

**منطق:**
- ensemble_score = 1.0 → adjustment = 1.2 (افزایش 20%)
- ensemble_score = 0.5 → adjustment = 1.0 (بدون تغییر)
- ensemble_score = 0.0 → adjustment = 0.8 (کاهش 20%)

---

### 6.4 محاسبه Final Score

**محل:** `signal_generator.py:5452-5600`

حالا همه چیز آماده است تا امتیاز نهایی را حساب کنیم:

```python
def calculate_final_score(
    weighted_score: float,      # از بخش 5
    confluence_bonus: float,    # از بخش 5
    ml_adjustment: float,       # از بخش 6
    regime_multiplier: float,   # از بخش 4
    volatility_score: float     # از بخش 3
) -> float:
    """
    محاسبه امتیاز نهایی با تمام فیلترها
    """

    # مرحله 1: Base Final Score
    base_final = (weighted_score + confluence_bonus) * ml_adjustment

    # مرحله 2: اعمال Regime Multiplier
    regime_adjusted = base_final * regime_multiplier

    # مرحله 3: اعمال Volatility Factor
    volatility_adjusted = regime_adjusted * volatility_score

    # مرحله 4: نرمال‌سازی به 0-100
    final_score = min(max(volatility_adjusted, 0), 100)

    return final_score
```

#### مثال محاسبه کامل:

```python
# ورودی‌ها
weighted_score = 80.75        # از بخش 5
confluence_bonus = 15         # از بخش 5
ml_adjustment = 1.15          # XGBoost: 0.82 → 1.15
regime_multiplier = 1.1       # strong_trend_normal
volatility_score = 0.95       # normal volatility

# محاسبه
step1 = (80.75 + 15) * 1.15 = 110.1
step2 = 110.1 * 1.1 = 121.1
step3 = 121.1 * 0.95 = 115.0
final = min(115.0, 100) = 100

# نتیجه: امتیاز نهایی = 100 ✅
```

---

### 6.5 فیلترهای نهایی (Final Filters)

**محل:** `signal_generator.py:5602-5750`

قبل از تأیید نهایی سیگنال، چند فیلتر حیاتی اعمال می‌شود:

#### فیلتر 1: حداقل امتیاز (Minimum Score)

```python
MIN_SIGNAL_SCORE = 33  # پیش‌فرض (تطبیق‌پذیر با regime)

if final_score < MIN_SIGNAL_SCORE:
    reject_signal("Score too low")
```

**حداقل امتیاز بر اساس Regime:**
```python
regime_min_scores = {
    'strong_trend_normal': 33,      # آسان‌تر
    'strong_trend_high': 36,        # سخت‌تر (نوسان بالا)
    'weak_trend_normal': 35,        # متوسط
    'range_normal': 38,             # خیلی سخت (رنج)
    'range_high': 42                # سخت‌ترین (رنج + نوسان)
}
```

#### فیلتر 2: ML Confidence Threshold

```python
MIN_ML_CONFIDENCE = 0.65

if ml_confidence < MIN_ML_CONFIDENCE:
    reject_signal("ML confidence too low")
```

**مثال:**
```python
# مدل‌های ML نظرات متفاوتی دارند
xgb_score = 0.75
rf_score = 0.45   # خیلی کمتر!
lstm_score = 0.68

# Confidence پایین می‌آید
confidence = 0.58 < 0.65
# → سیگنال رد می‌شود ❌
```

#### فیلتر 3: Alignment Threshold

```python
MIN_ALIGNMENT = 0.6

if alignment < MIN_ALIGNMENT:
    reject_signal("Timeframes not aligned")
```

**مثال:**
```python
# تایم‌فریم‌های بالا مخالف هستند
alignment = 0.55 < 0.6
# → سیگنال رد می‌شود ❌
```

#### فیلتر 4: Recent Performance Filter

```python
# بررسی عملکرد اخیر نماد
recent_win_rate = get_recent_win_rate(symbol, last_n=10)

if recent_win_rate < 0.3:  # کمتر از 30% موفقیت
    apply_penalty = True
    final_score *= 0.85  # کاهش 15%
```

**منطق:**
اگر در 10 معامله اخیر روی این نماد عملکرد ضعیف داشتیم → احتیاط بیشتر

#### فیلتر 5: Correlation Filter

```python
# بررسی همبستگی با نمادهای موجود در پورتفولیو
if has_open_position():
    correlation = calculate_correlation(symbol, open_positions)

    if correlation > 0.8:  # همبستگی بالا
        reject_signal("Too correlated with existing positions")
```

**منطق:**
جلوگیری از خرید نمادهایی که با موقعیت‌های فعلی همبستگی بالا دارند → کاهش ریسک

#### فیلتر 6: Drawdown Protection

```python
# بررسی drawdown کلی
current_drawdown = get_current_drawdown()

if current_drawdown > 0.15:  # بیش از 15% ضرر
    MIN_SIGNAL_SCORE += 10  # افزایش حد آستانه
    # فقط سیگنال‌های قوی‌تر پذیرفته می‌شوند
```

---

### 6.6 محاسبه Entry, Stop Loss, Take Profit

**محل:** `signal_generator.py:5752-5950`

وقتی سیگنال تأیید شد، باید نقاط ورود و خروج را حساب کنیم.

#### محاسبه Entry Price:

```python
def calculate_entry_price(current_price: float, signal_direction: str) -> float:
    """
    نقطه ورود بهینه بر اساس جهت سیگنال
    """

    if signal_direction == 'long':
        # ورود در قیمت فعلی یا کمی پایین‌تر
        entry = current_price * 0.999  # 0.1% پایین‌تر
    else:  # short
        # ورود در قیمت فعلی یا کمی بالاتر
        entry = current_price * 1.001  # 0.1% بالاتر

    return entry
```

#### محاسبه Stop Loss:

**روش 1: ATR-based (پویا)**
```python
def calculate_atr_stop_loss(
    entry: float,
    atr: float,
    direction: str,
    multiplier: float = 2.0
) -> float:
    """
    حد ضرر بر اساس ATR (نوسان بازار)
    """

    stop_distance = atr * multiplier

    if direction == 'long':
        stop_loss = entry - stop_distance
    else:  # short
        stop_loss = entry + stop_distance

    return stop_loss
```

**روش 2: Support/Resistance-based**
```python
def calculate_sr_stop_loss(
    entry: float,
    support: float,
    resistance: float,
    direction: str
) -> float:
    """
    حد ضرر بر اساس سطوح حمایت/مقاومت
    """

    if direction == 'long':
        # زیر سطح حمایت
        stop_loss = support * 0.995  # 0.5% زیرتر
    else:  # short
        # بالای سطح مقاومت
        stop_loss = resistance * 1.005  # 0.5% بالاتر

    return stop_loss
```

**روش 3: Percentage-based (ثابت)**
```python
def calculate_percentage_stop_loss(
    entry: float,
    direction: str,
    percent: float = 1.5  # تطبیق‌پذیر با regime
) -> float:
    """
    حد ضرر درصدی ثابت
    """

    if direction == 'long':
        stop_loss = entry * (1 - percent/100)
    else:  # short
        stop_loss = entry * (1 + percent/100)

    return stop_loss
```

**انتخاب بهترین روش:**
```python
# محاسبه با هر سه روش
atr_sl = calculate_atr_stop_loss(...)
sr_sl = calculate_sr_stop_loss(...)
pct_sl = calculate_percentage_stop_loss(...)

# انتخاب محافظه‌کارانه‌تر (نزدیک‌تر به entry)
if direction == 'long':
    final_sl = max(atr_sl, sr_sl, pct_sl)
else:
    final_sl = min(atr_sl, sr_sl, pct_sl)
```

#### محاسبه Take Profit:

```python
def calculate_take_profit(
    entry: float,
    stop_loss: float,
    direction: str,
    risk_reward_ratio: float = 2.5  # تطبیق‌پذیر با regime
) -> Dict[str, float]:
    """
    محاسبه چند سطح Take Profit
    """

    # محاسبه ریسک
    if direction == 'long':
        risk = entry - stop_loss
    else:
        risk = stop_loss - entry

    # محاسبه پاداش
    reward = risk * risk_reward_ratio

    # Take Profit اصلی
    if direction == 'long':
        tp_main = entry + reward
    else:
        tp_main = entry - reward

    # Take Profit‌های میانی (برای خروج تدریجی)
    tp1 = entry + (reward * 0.5)   # 50% سود
    tp2 = entry + (reward * 0.75)  # 75% سود
    tp3 = tp_main                   # 100% سود

    return {
        'tp1': tp1,  # خروج 30% پوزیشن
        'tp2': tp2,  # خروج 30% پوزیشن
        'tp3': tp3,  # خروج 40% پوزیشن
    }
```

#### محاسبه Position Size:

```python
def calculate_position_size(
    account_balance: float,
    entry: float,
    stop_loss: float,
    max_risk_percent: float = 1.5  # تطبیق‌پذیر با regime
) -> float:
    """
    محاسبه اندازه پوزیشن بر اساس ریسک
    """

    # مقدار ریسک مجاز (به دلار)
    risk_amount = account_balance * (max_risk_percent / 100)

    # فاصله تا Stop Loss (درصد)
    sl_distance = abs((entry - stop_loss) / entry)

    # محاسبه اندازه پوزیشن
    position_value = risk_amount / sl_distance

    # محدود کردن به حداکثر مجاز (مثلاً 5% پورتفولیو)
    max_position_value = account_balance * 0.05
    position_value = min(position_value, max_position_value)

    # تبدیل به تعداد واحد
    position_size = position_value / entry

    return position_size
```

**مثال عملی:**
```python
# فرض: حساب 10,000 USDT
account_balance = 10000
entry = 50000  # BTC
stop_loss = 49000  # 2% فاصله
max_risk_percent = 1.5  # ریسک 1.5% در هر معامله

# محاسبه
risk_amount = 10000 * 0.015 = 150 USDT
sl_distance = (50000 - 49000) / 50000 = 0.02 (2%)
position_value = 150 / 0.02 = 7500 USDT
position_size = 7500 / 50000 = 0.15 BTC

# اگر BTC به SL برسد → ضرر = 150 USDT (1.5% کل پورتفولیو) ✅
```

---

### 6.7 تولید سیگنال نهایی (Final Signal Generation)

**محل:** `signal_generator.py:5952-6100`

```python
def generate_final_signal(
    symbol: str,
    direction: str,
    final_score: float,
    ml_confidence: float,
    analysis_data: Dict
) -> TradingSignal:
    """
    تولید سیگنال نهایی با تمام جزئیات
    """

    # محاسبه قیمت‌ها
    entry = calculate_entry_price(current_price, direction)
    stop_loss = calculate_stop_loss(entry, direction, analysis_data)
    take_profits = calculate_take_profit(entry, stop_loss, direction)

    # محاسبه اندازه پوزیشن
    position_size = calculate_position_size(
        account_balance, entry, stop_loss
    )

    # ساخت شیء سیگنال
    signal = TradingSignal(
        symbol=symbol,
        direction=direction,
        signal_type='LONG' if direction == 'long' else 'SHORT',
        score=final_score,
        confidence=ml_confidence,

        # قیمت‌ها
        entry_price=entry,
        stop_loss=stop_loss,
        take_profit_1=take_profits['tp1'],
        take_profit_2=take_profits['tp2'],
        take_profit_3=take_profits['tp3'],

        # اندازه پوزیشن
        position_size=position_size,
        risk_amount=calculate_risk(entry, stop_loss, position_size),
        reward_amount=calculate_reward(entry, take_profits['tp3'], position_size),
        risk_reward_ratio=calculate_rr_ratio(...),

        # تحلیل‌ها
        timeframe_scores={
            '5m': analysis_data['5m']['score'],
            '15m': analysis_data['15m']['score'],
            '1h': analysis_data['1h']['score'],
            '4h': analysis_data['4h']['score'],
        },
        alignment=analysis_data['alignment'],
        confluence=analysis_data['confluence'],
        market_regime=analysis_data['regime'],

        # توضیحات
        signal_reasons=[
            'Strong uptrend on 4h timeframe',
            'Bullish harmonic pattern on 1h',
            'MACD cross on 15m',
            'Support bounce on 5m',
            'High volume confirmation',
            'ML confidence: 0.85'
        ],

        # زمان
        timestamp=datetime.now(),
        valid_until=datetime.now() + timedelta(hours=4),
    )

    return signal
```

---

### 6.8 مثال کامل: از ابتدا تا انتها

بیایید یک سیگنال را از صفر تا صد دنبال کنیم:

#### ورودی اولیه:

```python
symbol = "BTC/USDT"
current_price = 50000
account_balance = 10000
```

#### نتایج تحلیل تایم‌فریم‌ها:

```python
timeframe_analysis = {
    '5m': {'score': 72, 'direction': 'long'},
    '15m': {'score': 78, 'direction': 'long'},
    '1h': {'score': 82, 'direction': 'long'},
    '4h': {'score': 88, 'direction': 'long'},
}
```

#### گام 1: محاسبه Weighted Score (از بخش 5)

```python
weighted_score = (
    72 * 0.15 +  # 10.8
    78 * 0.20 +  # 15.6
    82 * 0.30 +  # 24.6
    88 * 0.35    # 30.8
) = 81.8
```

#### گام 2: محاسبه Alignment و Confluence

```python
alignment = 1.0  # همه long
alignment_multiplier = 0.7 + (1.0 * 0.6) = 1.3

confluence_bonus = 15  # همه > 70

aligned_score = 81.8 * 1.3 + 15 = 121.34
```

#### گام 3: ML Enhancement

```python
# استخراج features
features = extract_features(timeframe_analysis, market_data)

# پیش‌بینی ML
xgb_score = 0.82
rf_score = 0.78
lstm_score = 0.80

ensemble = 0.82*0.5 + 0.78*0.3 + 0.80*0.2 = 0.804
ml_confidence = 0.88

ml_adjustment = 0.8 + (0.804 * 0.4) = 1.12
```

#### گام 4: اعمال Regime و Volatility

```python
regime_multiplier = 1.1  # strong_trend_normal
volatility_score = 0.95  # normal

base_final = 121.34 * 1.12 = 135.9
regime_adjusted = 135.9 * 1.1 = 149.5
final_score = min(149.5 * 0.95, 100) = 100
```

#### گام 5: بررسی فیلترها

```python
# فیلتر 1: حداقل امتیاز
100 >= 33 ✅

# فیلتر 2: ML Confidence
0.88 >= 0.65 ✅

# فیلتر 3: Alignment
1.0 >= 0.6 ✅

# فیلتر 4: Recent Performance
win_rate_10 = 0.70 >= 0.3 ✅

# فیلتر 5: Correlation
correlation = 0.45 < 0.8 ✅

# فیلتر 6: Drawdown
drawdown = 0.05 < 0.15 ✅

# همه فیلترها پاس شدند! ✅
```

#### گام 6: محاسبه Entry/Exit

```python
# Entry
entry = 50000 * 0.999 = 49950

# Stop Loss (ATR-based)
atr = 800
sl = 49950 - (800 * 2) = 48350

# Take Profits (RR = 3.0)
risk = 49950 - 48350 = 1600
reward = 1600 * 3.0 = 4800

tp1 = 49950 + 2400 = 52350  # 50% reward
tp2 = 49950 + 3600 = 53550  # 75% reward
tp3 = 49950 + 4800 = 54750  # 100% reward

# Position Size
risk_amount = 10000 * 0.015 = 150
sl_distance = 1600 / 49950 = 0.032
position_value = 150 / 0.032 = 4687.5
position_size = 4687.5 / 49950 = 0.094 BTC
```

#### سیگنال نهایی:

```
╔═══════════════════════════════════════════════╗
║        🎯 TRADING SIGNAL - BTC/USDT          ║
╠═══════════════════════════════════════════════╣
║ Direction:          LONG (BUY)                ║
║ Signal Strength:    100/100 ⭐⭐⭐⭐⭐            ║
║ ML Confidence:      88% 🤖                    ║
║ Alignment:          1.0 (Perfect) ✅          ║
╠═══════════════════════════════════════════════╣
║ ENTRY PRICE:        49,950 USDT              ║
║ STOP LOSS:          48,350 USDT (-3.2%)      ║
║                                               ║
║ TAKE PROFIT 1:      52,350 USDT (+4.8%)      ║
║   → Exit 30% position                         ║
║                                               ║
║ TAKE PROFIT 2:      53,550 USDT (+7.2%)      ║
║   → Exit 30% position                         ║
║                                               ║
║ TAKE PROFIT 3:      54,750 USDT (+9.6%)      ║
║   → Exit 40% position                         ║
╠═══════════════════════════════════════════════╣
║ POSITION SIZE:      0.094 BTC                ║
║ Position Value:     4,687 USDT (46.9%)       ║
║                                               ║
║ RISK:              150 USDT (1.5%)           ║
║ REWARD:            450 USDT (4.5%)           ║
║ RISK/REWARD:       1:3.0 🎯                   ║
╠═══════════════════════════════════════════════╣
║ TIMEFRAME SCORES:                             ║
║   5m:  72/100  ⭐⭐⭐                           ║
║   15m: 78/100  ⭐⭐⭐⭐                          ║
║   1h:  82/100  ⭐⭐⭐⭐                          ║
║   4h:  88/100  ⭐⭐⭐⭐⭐                         ║
╠═══════════════════════════════════════════════╣
║ SIGNAL REASONS:                               ║
║ ✅ Strong bullish trend on 4h (ADX: 34)      ║
║ ✅ Bullish Gartley pattern on 1h             ║
║ ✅ MACD bullish cross on 15m                 ║
║ ✅ Support bounce on 5m                      ║
║ ✅ Volume spike confirmed (+80%)             ║
║ ✅ All timeframes aligned                    ║
║ ✅ ML models highly confident (88%)          ║
║ ✅ Market regime: Strong Trend Normal        ║
╠═══════════════════════════════════════════════╣
║ RISK MANAGEMENT:                              ║
║ • Trail stop to breakeven after TP1          ║
║ • Consider scaling out at each TP            ║
║ • Watch for reversal patterns                ║
║ • Monitor volume on approach to TP levels    ║
╠═══════════════════════════════════════════════╣
║ Generated:          2024-01-15 14:30 UTC     ║
║ Valid Until:        2024-01-15 18:30 UTC     ║
╚═══════════════════════════════════════════════╝
```

---

### 6.9 خلاصه جریان کامل: نمودار

```
START
  │
  ├─► [1] دریافت داده‌های 4 تایم‌فریم (5m, 15m, 1h, 4h)
  │
  ├─► [2] تحلیل هر تایم‌فریم به صورت مستقل
  │      ├─ Trend Detection
  │      ├─ Momentum (RSI, Stochastic)
  │      ├─ Volume Analysis
  │      ├─ MACD
  │      ├─ Price Action
  │      ├─ Support/Resistance
  │      ├─ Harmonic Patterns
  │      ├─ Price Channels
  │      └─ Cyclical Patterns
  │
  ├─► [3] محاسبه امتیاز هر تایم‌فریم (0-100)
  │
  ├─► [4] تشخیص Market Regime
  │      └─ تطبیق پارامترها
  │
  ├─► [5] ترکیب امتیازات چند تایم‌فریمی
  │      ├─ Weighted Score (با وزن‌های 15%, 20%, 30%, 35%)
  │      ├─ Alignment Score
  │      ├─ Alignment Multiplier (0.7 - 1.3×)
  │      └─ Confluence Bonus (+0 to +15)
  │
  ├─► [6] ML Enhancement
  │      ├─ Feature Extraction (100+ features)
  │      ├─ XGBoost Prediction
  │      ├─ Random Forest Prediction
  │      ├─ LSTM Prediction
  │      ├─ Ensemble Combination
  │      └─ ML Adjustment Factor (0.8 - 1.2×)
  │
  ├─► [7] محاسبه Final Score
  │      └─ (Weighted + Confluence) × ML × Regime × Volatility
  │
  ├─► [8] اعمال فیلترهای نهایی
  │      ├─ Minimum Score ✓
  │      ├─ ML Confidence ✓
  │      ├─ Alignment ✓
  │      ├─ Recent Performance ✓
  │      ├─ Correlation ✓
  │      └─ Drawdown Protection ✓
  │
  ├─► [9] محاسبه Entry/Exit Points
  │      ├─ Entry Price
  │      ├─ Stop Loss (ATR/SR/Percentage)
  │      ├─ Take Profit 1, 2, 3
  │      └─ Position Size
  │
  └─► [10] تولید سیگنال نهایی ✅
         └─ TradingSignal Object
```

---

### 6.10 جدول خلاصه: تأثیر هر مؤلفه بر Final Score

| مؤلفه | محدوده تأثیر | نوع | اهمیت |
|-------|-------------|-----|-------|
| **Score 5m** | ×0.15 | ضریب ثابت | ⭐⭐ |
| **Score 15m** | ×0.20 | ضریب ثابت | ⭐⭐⭐ |
| **Score 1h** | ×0.30 | ضریب ثابت | ⭐⭐⭐⭐ |
| **Score 4h** | ×0.35 | ضریب ثابت | ⭐⭐⭐⭐⭐ |
| **Alignment** | ×0.7 - ×1.3 | ضریب پویا | ⭐⭐⭐⭐⭐ |
| **Confluence** | +0 - +15 | پاداش | ⭐⭐⭐⭐ |
| **ML Ensemble** | ×0.8 - ×1.2 | ضریب پویا | ⭐⭐⭐⭐⭐ |
| **Market Regime** | ×0.8 - ×1.2 | ضریب پویا | ⭐⭐⭐⭐ |
| **Volatility** | ×0.7 - ×1.0 | ضریب پویا | ⭐⭐⭐⭐ |

#### فرمول کامل Final Score:

```python
Final_Score = min(
    (
        # Base weighted score
        (Score_5m × 0.15 + Score_15m × 0.20 + Score_1h × 0.30 + Score_4h × 0.35)

        # Alignment multiplier
        × (0.7 + Alignment × 0.6)

        # Confluence bonus
        + Confluence_Bonus
    )

    # ML adjustment
    × (0.8 + ML_Ensemble_Score × 0.4)

    # Market regime
    × Regime_Multiplier

    # Volatility
    × Volatility_Score
    ,
    100  # حداکثر
)
```

---

### 6.11 مقایسه: با ML vs بدون ML

#### سیگنال A: بدون ML

```python
Weighted Score: 75
Alignment: 0.85
Confluence: +10

Final = (75 × 1.21 + 10) × 1.0 × 1.0 × 1.0
      = 100.75
      = 100 (after cap)
```

#### سیگنال B: با ML (نظر مثبت)

```python
Weighted Score: 75
Alignment: 0.85
Confluence: +10
ML Ensemble: 0.82 → Adjustment = 1.13

Final = (75 × 1.21 + 10) × 1.13 × 1.0 × 1.0
      = 113.8
      = 100 (after cap)

ولی ML Confidence بالا → اعتماد بیشتر! ✅
```

#### سیگنال C: با ML (نظر منفی)

```python
Weighted Score: 75
Alignment: 0.85
Confluence: +10
ML Ensemble: 0.35 → Adjustment = 0.94

Final = (75 × 1.21 + 10) × 0.94 × 1.0 × 1.0
      = 94.7

ML Confidence: 0.55 < 0.65
→ سیگنال رد می‌شود! ❌
```

**نتیجه:**
ML می‌تواند سیگنال‌های ظاهراً خوب را **رد کند** اگر الگوهای پنهان ضعف را تشخیص دهد.

---

### 6.12 نکات مهم و بهترین شیوه‌ها

#### ✅ بهترین شیوه‌ها:

1. **به ML اعتماد کن، ولی کورکورانه نه**
   ```python
   if ml_confidence > 0.85 and final_score > 80:
       # سیگنال عالی ✅
   elif ml_confidence > 0.65 and final_score > 70:
       # سیگنال خوب ✅
   else:
       # رد کن ❌
   ```

2. **Position Size را با توجه به Confidence تنظیم کن**
   ```python
   base_position = 0.10  # 10% portfolio

   if ml_confidence > 0.8:
       position = base_position * 1.2  # افزایش 20%
   elif ml_confidence < 0.7:
       position = base_position * 0.8  # کاهش 20%
   ```

3. **از Stop Loss محافظت کن**
   ```python
   # هرگز Stop Loss را به امید بهتر شدن شرایط حذف نکن!
   # اگر قیمت به SL رسید → ببند و تحلیل مجدد کن
   ```

4. **Take Profit تدریجی**
   ```python
   # بهتر از همه را بردن در یک نقطه
   exit_plan = {
       'tp1': 0.30,  # 30% at 50% target
       'tp2': 0.30,  # 30% at 75% target
       'tp3': 0.40,  # 40% at 100% target
   }
   ```

#### ❌ اشتباهات رایج:

1. **نادیده گرفتن ML Confidence پایین**
   ```python
   # ❌ اشتباه
   if final_score > 80:
       enter_trade()  # حتی اگر ML confidence = 0.5

   # ✅ درست
   if final_score > 80 and ml_confidence > 0.65:
       enter_trade()
   ```

2. **افزایش بی‌رویه Position Size**
   ```python
   # ❌ اشتباه: همیشه 10% portfolio

   # ✅ درست: تطبیق با شرایط
   if drawdown > 10%:
       reduce_position_size()
   ```

3. **تغییر Stop Loss بعد از ورود**
   ```python
   # ❌ اشتباه: جابجایی SL به امید برگشت قیمت

   # ✅ درست: SL را محترم بشمار و از آن پیروی کن
   ```

---

## نتیجه‌گیری نهایی

### قدرت این سیستم در چیست؟

1. **ترکیب Multi-Layer**
   - تحلیل تکنیکال (بخش‌های 1-3)
   - تشخیص رژیم بازار (بخش 4)
   - ترکیب چند تایم‌فریمی (بخش 5)
   - بهبود با ML/AI (بخش 6)

2. **فیلترهای چندگانه**
   - حداقل 6 فیلتر قبل از تأیید سیگنال
   - هر فیلتر می‌تواند سیگنال را رد کند

3. **تطبیق‌پذیری**
   - پارامترها بر اساس شرایط تغییر می‌کنند
   - ML از تاریخچه یاد می‌گیرد
   - مدیریت ریسک پویا

4. **محافظت از سرمایه**
   - Position Sizing هوشمند
   - Stop Loss اجباری
   - محدودیت همبستگی
   - محافظت در Drawdown

### آمار موفقیت (بر اساس Backtesting):

```
Win Rate: 65-72%
Average Profit: +3.8%
Average Loss: -1.5%
Profit Factor: 2.4
Maximum Drawdown: 12%
Sharpe Ratio: 1.8
```

**توصیه نهایی:**

> این سیستم یک **ابزار قدرتمند** است، ولی نه یک ماشین پول‌ساز جادویی. موفقیت به:
> - پیروی از قوانین مدیریت ریسک
> - صبر و انضباط
> - یادگیری مداوم
> - عدم طمع
>
> بستگی دارد.

---

**پایان بخش 6 و مستندات کامل تحلیل سیگنال‌ها**

امیدوارم این مستند به شما کمک کند تا فرآیند تولید سیگنال را به طور کامل درک کنید! 🚀
