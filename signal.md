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

**محل:** `signal_generator.py:4647-4790`

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

# پیدا کردن آخرین اندیس معتبر (بدون NaN)
last_valid_idx = -1
while last_valid_idx >= -len(df) and (np.isnan(ema20[last_valid_idx]) or
                                      np.isnan(ema50[last_valid_idx]) or
                                      np.isnan(ema100[last_valid_idx])):
    last_valid_idx -= 1

# محاسبه شیب (Slope) برای تشخیص جهت
# تفاوت بین آخرین مقدار معتبر و 5 کندل قبل از آن
ema20_slope = ema20[last_valid_idx] - ema20[last_valid_idx - 5] if last_valid_idx >= 5 else 0
ema50_slope = ema50[last_valid_idx] - ema50[last_valid_idx - 5] if last_valid_idx >= 5 else 0
```

**انواع چیدمان EMA (EMA Arrangement):**

| چیدمان | شرط | معنی |
|--------|-----|------|
| `bullish_aligned` | EMA20 > EMA50 > EMA100 | روند صعودی قوی ✅ |
| `bearish_aligned` | EMA20 < EMA50 < EMA100 | روند نزولی قوی ⬇️ |
| `potential_bullish_reversal` | EMA20 > EMA50 < EMA100 | احتمال بازگشت صعودی 🔄 |
| `potential_bearish_reversal` | EMA20 < EMA50 > EMA100 | احتمال بازگشت نزولی 🔄 |
| `unknown` | غیر از موارد بالا | روند نامشخص ⚠️ |

---

##### 1.2 تعیین جهت و قدرت روند

**جدول کامل Trend Detection (محل در کد: signal_generator.py:1784-1816):**

**⚠️ توجه:** شرایط به ترتیب با `if-elif` بررسی می‌شوند، یعنی اولین شرط که برقرار باشد اعمال می‌شود.

| شرط | Trend | Strength | Phase | خط کد |
|-----|-------|----------|-------|-------|
| Close > EMA20 > EMA50 > EMA100 **و** ema20_slope > 0 **و** ema50_slope > 0 | `bullish` | **+3** | `mature` اگر `ema_arrangement == 'bullish_aligned'` وگرنه `developing` | 1784-1787 |
| Close > EMA20 > EMA50 **و** ema20_slope > 0 | `bullish` | **+2** | `developing` | 1788-1791 |
| Close > EMA20 **و** ema20_slope > 0 | `bullish` | **+1** | `early` | 1792-1795 |
| Close < EMA20 < EMA50 < EMA100 **و** ema20_slope < 0 **و** ema50_slope < 0 | `bearish` | **-3** | `mature` اگر `ema_arrangement == 'bearish_aligned'` وگرنه `developing` | 1796-1799 |
| Close < EMA20 < EMA50 **و** ema20_slope < 0 | `bearish` | **-2** | `developing` | 1800-1803 |
| Close < EMA20 **و** ema20_slope < 0 | `bearish` | **-1** | `early` | 1804-1807 |
| Close < EMA50 **و** EMA20 > EMA50 **و** ema50_slope > 0 | `bullish_pullback` | **+1** | `pullback` | 1808-1811 |
| Close > EMA50 **و** EMA20 < EMA50 **و** ema50_slope < 0 | `bearish_pullback` | **-1** | `pullback` | 1812-1815 |
| هیچ‌کدام از موارد بالا | `neutral` | **0** | `undefined` | 1780-1782 |

**⚠️ نکات مهم:**
- حالت‌های `bullish_pullback` و `bearish_pullback` اصلاحات قیمت در طول روند را نشان می‌دهند
- در حالت `mature`، اگر `ema_arrangement` برابر با `bullish_aligned` یا `bearish_aligned` نباشد، `phase` به `developing` تغییر می‌کند
- **نکته کد واقعی:** در حالت strength=2 و strength=1، فقط `ema20_slope` چک می‌شود (نه `ema50_slope`)، در حالی که در strength=3 هر دو slope چک می‌شوند
  - این موضوع یک نقطه ضعف است که در `Suggested_Improvment.md` (مشکل 4) بیشتر توضیح داده شده است

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
                                     # ⚠️ توجه: ema50_slope محاسبه می‌شود اما در details قرار نمی‌گیرد
        'ema_arrangement': 'bullish_aligned'
    }
}
```

**⚠️ توجه:**
- در کد واقعی فیلد `confidence` وجود ندارد (این یکی از پیشنهادات بهبود در مشکل 5 است)
- `ema50_slope` محاسبه می‌شود و در برخی شرایط (strength=3) استفاده می‌شود، اما در `details` خروجی قرار نمی‌گیرد

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

**⚠️ نکته مهم:** محاسبه `structure_score` شامل **چهار مرحله** است:

```python
# محل در کد: signal_generator.py:4395-4429

# مرحله 1: امتیاز پایه
base_score = 1.0  # پیش‌فرض
structure_score = base_score

# مرحله 2: اضافه/کسر Bonus/Penalty ثابت
if trends_aligned:
    structure_score += 0.2  # confirm_bonus
else:
    structure_score -= 0.3  # contradict_penalty

# مرحله 3: اعمال Multiplier متغیر
if trends_aligned:
    structure_score *= (1 + 1.5 * (min_strength / 3))
else:
    structure_score *= (1 - 1.5 * (min_strength / 3))

# مرحله 4: محدودیت min/max
structure_score = max(min(structure_score, 1.5), 0.5)  # محدود به [0.5, 1.5]
```

---

**سناریو 1: روندها همراستا (trends_aligned = True)**

محاسبه کامل با strength = 3:
```python
structure_score = 1.0           # base
structure_score += 0.2          # confirm_bonus → 1.2
structure_score *= (1 + 1.5)    # multiplier → 1.2 * 2.5 = 3.0
structure_score = min(3.0, 1.5) # محدودیت max → 1.5
# نتیجه نهایی: 1.5
```

| Strength | قبل Multiplier | Multiplier | قبل محدودیت | نتیجه نهایی |
|----------|---------------|-----------|-------------|-------------|
| 3 | 1.2 | 2.5 | 3.0 | **1.5** (محدود شد) |
| 2 | 1.2 | 2.0 | 2.4 | **1.5** (محدود شد) |
| 1 | 1.2 | 1.5 | 1.8 | **1.5** (محدود شد) |

---

**سناریو 2: روندها مخالف (trends_aligned = False)**

محاسبه کامل با strength = 3:
```python
structure_score = 1.0           # base
structure_score -= 0.3          # contradict_penalty → 0.7
structure_score *= (1 - 1.5)    # multiplier → 0.7 * (-0.5) = -0.35
structure_score = max(-0.35, 0.5) # محدودیت min → 0.5
# نتیجه نهایی: 0.5
```

| Strength | قبل Multiplier | Multiplier | قبل محدودیت | نتیجه نهایی |
|----------|---------------|-----------|-------------|-------------|
| 3 | 0.7 | -0.5 | -0.35 | **0.5** (محدود شد) |
| 2 | 0.7 | 0.0 | 0.0 | **0.5** (محدود شد) |
| 1 | 0.7 | 0.5 | 0.35 | **0.5** (محدود شد - نزدیک بود!) |

**نتیجه‌گیری:**
- تمام حالات aligned به **1.5** ختم می‌شوند (حداکثر)
- تمام حالات conflicting به **0.5** ختم می‌شوند (حداقل)
- محدودیت min/max باعث می‌شود تفاوت واقعی فقط **3x** باشد (1.5 / 0.5)

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

##### 1.7 تشخیص Reversal و تأثیر آن

**محل در کد:** `signal_generator.py:3706-3730` و `signal_generator.py:5071-5077`

کد فعلی قابلیت تشخیص سیگنال‌های **Reversal** (بازگشت روند) را دارد، اما به صورت محدود.

**چه موقع سیگنال Reversal شناسایی می‌شود؟**

```python
# محل: signal_generator.py:3714-3719
is_reversal = False

# شرط 1: RSI Divergence (واگرایی)
if any('rsi_bullish_divergence' == s.get('type') for s in momentum_signals):
    is_reversal = True
    reversal_strength += 0.7

if any('rsi_bearish_divergence' == s.get('type') for s in momentum_signals):
    is_reversal = True
    reversal_strength += 0.7

# شرط 2: Oversold/Overbought در خلاف روند
# اگر RSI oversold در روند نزولی → احتمال بازگشت صعودی
# اگر RSI overbought در روند صعودی → احتمال بازگشت نزولی
```

**تأثیر Reversal بر امتیازدهی:**

```python
# محل: signal_generator.py:5071-5077
if is_reversal:
    # سیگنال در خلاف روند است اما دلیل reversal دارد
    reversal_modifier = max(0.3, 1.0 - (reversal_strength * 0.7))
    score.timeframe_weight = 1.0 + (higher_tf_ratio * 0.3 * reversal_modifier)
    score.trend_alignment = max(0.5, 1.0 - (reversal_strength * 0.5))
else:
    # سیگنال با روند همراستا است
    score.timeframe_weight = 1.0 + (higher_tf_ratio * 0.5)
    score.trend_alignment = 1.0 + (primary_trend_strength * 0.2)
```

**محاسبه Trend Alignment:**

| Scenario | Reversal Strength | trend_alignment | تفسیر |
|----------|------------------|-----------------|-------|
| **Reversal قوی** | 1.0 | max(0.5, 0.5) = **0.5** | کاهش 50% |
| **Reversal متوسط** | 0.7 | max(0.5, 0.65) = **0.65** | کاهش 35% |
| **Reversal ضعیف** | 0.3 | max(0.5, 0.85) = **0.85** | کاهش 15% |
| **With Trend** | - | 1.0 + (3 * 0.2) = **1.6** | افزایش 60% |

**⚠️ محدودیت‌های کد فعلی:**

1. **فقط RSI Divergence:**
   - الگوهای کلاسیک reversal (Head & Shoulders، Double Top/Bottom) در نظر گرفته نمی‌شوند

2. **عدم بررسی Support/Resistance:**
   - Reversal در سطوح قوی S/R معتبرتر است
   - کد فعلی این را چک نمی‌کند

3. **یکسان‌سازی Counter-Trend:**
   - هر سیگنال خلاف روند (بدون دلیل reversal) جریمه سنگین می‌شود
   - ممکن است فرصت‌های reversal معتبر را از دست بدهیم

**مثال عملی:**

```python
# سناریو: سیگنال Long در روند Bearish با RSI Bullish Divergence
trend = 'bearish'
strength = -3
is_reversal = True
reversal_strength = 0.7

# محاسبه:
trend_alignment = max(0.5, 1.0 - (0.7 * 0.5))
               = max(0.5, 0.65)
               = 0.65

# نتیجه: سیگنال 35% کاهش می‌یابد (به جای 50% در صورت نبود divergence)
```

**نکته:** این بخش در Suggested_Improvment.md دارای پیشنهادات بهبود است.

---

#### مرحله 2: تحلیل اندیکاتورهای مومنتوم (RSI, Stochastic, MACD, MFI)

**محل در کد:** `signal_generator.py:3511-3691`

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

**محل در کد:** `signal_generator.py:4911-4921` و `signal_generator.py:5246-5253`

Momentum در **دو مرحله** استفاده می‌شود:

---

**مرحله 1: انتخاب Base Signal (خط 4911-4921)**

برای هر تایم‌فریم، سیستم تصمیم می‌گیرد که **Price Action** یا **Momentum** پایه سیگنال باشد:

```python
# signal_generator.py:4911-4921
mom_res = result.get('momentum', {})
pa_res = result.get('price_action', {})

# محاسبه امتیاز خالص (bullish - bearish) برای هر کدام
pa_score = pa_res.get('bullish_score', 0) - pa_res.get('bearish_score', 0)
mom_score = mom_res.get('bullish_score', 0) - mom_res.get('bearish_score', 0)

# انتخاب قوی‌تر:
if abs(pa_score) >= abs(mom_score):
    # Price Action قوی‌تر است
    base_signal_score = pa_score
    base_direction = 'bullish' if pa_score > 0 else 'bearish'
elif abs(mom_score) > 0:
    # Momentum قوی‌تر است
    base_signal_score = mom_score
    base_direction = 'bullish' if mom_score > 0 else 'bearish'
```

**نکته:** اگر Price Action و Momentum هر دو قوی باشند، **Price Action** اولویت دارد.

---

**مرحله 2: محاسبه Multi-Timeframe Score (خط 5246-5253)**

در این مرحله، momentum به امتیاز کلی اضافه می‌شود:

```python
# signal_generator.py:5246-5253
# 1. دریافت داده‌های momentum
mom_data = result.get('momentum', {})
momentum_directions[tf] = mom_data.get('direction', 'neutral')
momentum_strength = mom_data.get('momentum_strength', 1.0)  # پیش‌فرض: 1.0

# 2. اضافه کردن امتیازات صعودی/نزولی
bullish_score += mom_data.get('bullish_score', 0) * tf_weight * momentum_strength
bearish_score += mom_data.get('bearish_score', 0) * tf_weight * momentum_strength

# 3. اضافه کردن momentum signals به لیست کل signals
mom_signals = [
    {**s,
     'timeframe': tf,
     'score': s.get('score', 0) * tf_weight * momentum_strength
    }
    for s in mom_data.get('signals', [])
]
all_signals.extend(mom_signals)
```

**فرمول نهایی امتیاز momentum:**
```
امتیاز هر signal = base_score × timeframe_weight × momentum_strength
```

**مثال محاسبه:**
```python
# فرض: تایم‌فریم 5m با وزن 0.15
tf_weight = 0.15
momentum_strength = 1.0  # پیش‌فرض

# Signal: rsi_oversold_reversal با امتیاز 2.3
signal_score = 2.3 × 0.15 × 1.0 = 0.345

# این امتیاز به bullish_score اضافه می‌شود:
bullish_score += 0.345
```

---

**نقش Momentum در تصمیم‌گیری:**

| سناریو | Price Action | Momentum | نتیجه |
|--------|-------------|----------|-------|
| **تأیید کامل** | Bullish (+8) | Bullish (+9) | قوی‌ترین ✅ (momentum base signal می‌شود) |
| **تأیید متوسط** | Bullish (+10) | Bullish (+5) | خوب ✅ (price action base signal می‌شود) |
| **تضاد** | Bullish (+8) | Bearish (-6) | ضعیف ⚠️ (price action base، اما نزولی‌ها هم جمع می‌شوند) |
| **خنثی** | Bullish (+8) | Neutral (0) | متوسط (فقط price action) |

**توضیح تضاد:**
- اگر Price Action صعودی (+8) و Momentum نزولی (-6) باشد:
- Price Action به عنوان base signal انتخاب می‌شود (چون abs(8) > abs(-6))
- اما در Multi-Timeframe Score، **هر دو** جمع می‌شوند:
  - `bullish_score += 8 (از price action)`
  - `bearish_score += 6 (از momentum)`
  - نتیجه: تضعیف سیگنال نهایی

**⚠️ نکته مهم:** Momentum Signals به صورت **جداگانه** به امتیاز کلی اضافه می‌شوند، **نه** به عنوان جریمه/پاداش مستقیم.

---

**تأثیر Divergence:**

واگرایی (Divergence) **قوی‌ترین** سیگنال momentum است:

```python
# امتیاز واگرایی:
divergence_score = 3.5 × divergence_strength  # حداکثر 3.5

# در مقایسه با سایر signals:
rsi_oversold_reversal = 2.3      # ثابت
macd_bullish_crossover = 2.2     # ثابت
divergence = 2.1 تا 3.5          # متغیر بر اساس قدرت
```

واگرایی اغلب باعث می‌شود Momentum به عنوان **Base Signal** انتخاب شود.

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

**📍 کد مرجع:** `signal_generator.py:1658-1717` - تابع `analyze_volume_trend()`

```python
analysis_data['volume'] = self.analyze_volume_trend(df, window=20)
```

حجم معاملات یکی از مهمترین عوامل تأیید کننده سیگنال‌های معاملاتی است. این بخش حجم معاملات فعلی را با میانگین متحرک حجم مقایسه کرده و الگوهای حجمی را شناسایی می‌کند.

---

##### 🔍 فرآیند تحلیل حجم

**گام 1: محاسبه میانگین متحرک حجم (Volume SMA)**

```python
# signal_generator.py:1667-1670
if use_bottleneck:
    vol_sma = bn.move_mean(vol_series.values, window=20, min_count=20)
else:
    vol_sma = vol_series.rolling(window=20, min_periods=20).mean().values
```

- از میانگین متحرک ساده 20 دوره‌ای استفاده می‌شود
- **Optimization:** اگر کتابخانه `bottleneck` نصب باشد، از `bn.move_mean()` استفاده می‌کند (سریع‌تر)
- در غیر این صورت از `pandas.rolling().mean()` استفاده می‌شود
- این میانگین به عنوان مبنای مقایسه برای حجم فعلی عمل می‌کند

**گام 2: محاسبه نسبت حجم (Volume Ratio)**

```python
vol_ratio = current_volume / vol_sma
```

**فرمول:**
```
Volume Ratio = حجم فعلی / میانگین متحرک 20 دوره‌ای حجم
```

این نسبت نشان می‌دهد که حجم فعلی چند برابر میانگین حجم اخیر است.

**گام 3: طبقه‌بندی الگوی حجمی**

بر اساس **آستانه پایه (Base Threshold)**: `volume_multiplier_threshold = 1.3` (پیش‌فرض)

| Volume Ratio | Trend | Pattern | توضیح |
|-------------|-------|---------|-------|
| `> 2.6` (2.0 × 1.3) | `strongly_increasing` | `climax_volume` | حجم بسیار بالا - احتمال اوج حرکت |
| `> 1.95` (1.5 × 1.3) | `increasing` | `spike` | افزایش ناگهانی حجم |
| `> 1.3` | `increasing` | `above_average` | حجم بالاتر از متوسط |
| `< 0.867` (1/1.3) | `decreasing` | `below_average` | حجم کمتر از متوسط |
| `< 0.51` (1/(1.3×1.5)) | `strongly_decreasing` | `dry_up` | حجم بسیار پایین - خشک شدن بازار |
| بقیه موارد | `neutral` | `normal` | حجم عادی |

**کد واقعی از implementation:**
```python
# signal_generator.py:1687-1704
if current_ratio > self.volume_multiplier_threshold * 2.0:
    results['trend'] = 'strongly_increasing'
    results['pattern'] = 'climax_volume'
elif current_ratio > self.volume_multiplier_threshold * 1.5:
    results['trend'] = 'increasing'
    results['pattern'] = 'spike'
elif current_ratio > self.volume_multiplier_threshold:
    results['trend'] = 'increasing'
    results['pattern'] = 'above_average'
elif current_ratio < 1.0 / (self.volume_multiplier_threshold * 1.5):
    results['trend'] = 'strongly_decreasing'
    results['pattern'] = 'dry_up'
elif current_ratio < 1.0 / self.volume_multiplier_threshold:
    results['trend'] = 'decreasing'
    results['pattern'] = 'below_average'
else:
    results['trend'] = 'neutral'
    results['pattern'] = 'normal'
```

**گام 4: تعیین تأیید حجمی (Volume Confirmation)**

```python
is_confirmed_by_volume = current_ratio > volume_multiplier_threshold
```

**شرط تأیید:**
- اگر `Volume Ratio > 1.3` → سیگنال توسط حجم تأیید می‌شود
- اگر `Volume Ratio ≤ 1.3` → سیگنال توسط حجم تأیید نمی‌شود

**گام 5: محاسبه روند میانگین حجم (Volume MA Trend)**

اگر حداقل 10 کندل موجود باشد:

```python
vol_sma_slope = (vol_sma[-1] - vol_sma[-10]) / vol_sma[-10]

if vol_sma_slope > 0.05:    # افزایش 5%
    volume_ma_trend = 'increasing'
elif vol_sma_slope < -0.05:  # کاهش 5%
    volume_ma_trend = 'decreasing'
else:
    volume_ma_trend = 'flat'
```

این نشان می‌دهد که روند کلی حجم در حال افزایش، کاهش یا ثابت است.

---

##### 📊 خروجی تابع analyze_volume_trend

```python
{
    'status': 'ok',                        # وضعیت محاسبات
    'current_ratio': 1.8,                  # نسبت حجم فعلی به میانگین
    'trend': 'increasing',                 # روند حجم
    'pattern': 'spike',                    # الگوی حجمی
    'is_confirmed_by_volume': True,        # آیا سیگنال توسط حجم تأیید می‌شود؟
    'volume_ma_trend': 'increasing',       # روند میانگین حجم
    'volume_ma_slope': 0.08                # شیب میانگین حجم (8% افزایش)
}
```

---

##### 🎯 تأثیر حجم بر امتیاز نهایی

**1. محاسبه ضریب تأیید حجمی در یک تایم‌فریم:**

```python
# signal_generator.py:5079
volume_confirmation = 1.0 + (volume_confirmation_factor * 0.4)
```

**فرمول:**
```
Volume Confirmation Factor = 1.0 + (عامل تأیید حجمی × 0.4)
```

**مثال:**
- اگر حجم سیگنال را تأیید کند: `volume_confirmation_factor = 1.0`
  - `volume_confirmation = 1.0 + (1.0 × 0.4) = 1.4` → **+40% افزایش امتیاز**
- اگر حجم سیگنال را تأیید نکند: `volume_confirmation_factor = 0.0`
  - `volume_confirmation = 1.0 + (0.0 × 0.4) = 1.0` → **بدون تغییر امتیاز**

**2. محاسبه ضریب تأیید حجمی چندتایم‌فریمی:**

```python
# signal_generator.py:5360-5367
weighted_volume_factor = 0.0
total_weight = 0.0

for timeframe, is_confirmed in volume_confirmations.items():
    tf_weight = timeframe_weights.get(timeframe, 1.0)
    weighted_volume_factor += (1 if is_confirmed else 0) * tf_weight
    total_weight += tf_weight

# Safety check برای division by zero
volume_confirmation_factor = weighted_volume_factor / total_weight if total_weight > 0 else 0.0
```

**مثال محاسبه چندتایم‌فریمی:**

فرض کنید وزن‌های تایم‌فریم:
- `5m`: وزن = 1.0
- `15m`: وزن = 1.5
- `1h`: وزن = 2.0
- `4h`: وزن = 2.5

و تأیید حجم در هر تایم‌فریم:
- `5m`: تأیید شده (1)
- `15m`: تأیید شده (1)
- `1h`: تأیید نشده (0)
- `4h`: تأیید شده (1)

```
weighted_volume_factor = (1 × 1.0) + (1 × 1.5) + (0 × 2.0) + (1 × 2.5)
                       = 1.0 + 1.5 + 0 + 2.5
                       = 5.0

total_weight = 1.0 + 1.5 + 2.0 + 2.5 = 7.0

volume_confirmation_factor = 5.0 / 7.0 = 0.714 (≈71%)

volume_confirmation = 1.0 + (0.714 × 0.4) = 1.286
```

**نتیجه:** در این مثال، 71% از تایم‌فریم‌ها (به صورت وزنی) سیگنال را تأیید کرده‌اند، که منجر به **+28.6% افزایش امتیاز** می‌شود.

**3. اعمال ضریب حجم بر امتیاز نهایی:**

```python
final_score = base_score × volume_confirmation × (سایر ضرایب)
```

---

##### 📝 مثال‌های کاربردی

**مثال 1: سیگنال خرید با حجم بالا**

```
Current Volume: 2,500,000
Volume SMA(20): 1,200,000
Volume Ratio = 2,500,000 / 1,200,000 = 2.08

طبقه‌بندی:
- 2.08 > 1.95 (1.5 × 1.3) → trend = 'increasing', pattern = 'spike'
- 2.08 > 1.3 → is_confirmed_by_volume = True

تأثیر بر امتیاز:
- volume_confirmation_factor = 1.0 (تأیید شده)
- volume_confirmation = 1.0 + (1.0 × 0.4) = 1.4
- امتیاز نهایی با 40% افزایش می‌یابد ✓
```

**مثال 2: سیگنال فروش با حجم پایین**

```
Current Volume: 600,000
Volume SMA(20): 1,200,000
Volume Ratio = 600,000 / 1,200,000 = 0.5

طبقه‌بندی:
- 0.5 < 0.51 → trend = 'strongly_decreasing', pattern = 'dry_up'
- 0.5 < 1.3 → is_confirmed_by_volume = False

تأثیر بر امتیاز:
- volume_confirmation_factor = 0.0 (تأیید نشده)
- volume_confirmation = 1.0 + (0.0 × 0.4) = 1.0
- امتیاز نهایی تغییر نمی‌کند ⚠️
```

**مثال 3: حجم اوج (Climax Volume)**

```
Current Volume: 5,000,000
Volume SMA(20): 1,500,000
Volume Ratio = 5,000,000 / 1,500,000 = 3.33

طبقه‌بندی:
- 3.33 > 2.6 (2.0 × 1.3) → trend = 'strongly_increasing', pattern = 'climax_volume'
- این می‌تواند نشانه اوج حرکت و احتمال برگشت باشد

هشدار:
- حجم بسیار بالا ممکن است نشان‌دهنده exhaustion (خستگی بازار) باشد
- باید با سایر اندیکاتورها (RSI بالا، واگرایی) بررسی شود
```

---

##### ⚙️ پارامترهای قابل تنظیم

**در فایل تنظیمات (`signal_config`):**

```python
{
    'volume_multiplier_threshold': 1.3,  # آستانه اصلی برای تأیید حجمی
    # مقادیر پیشنهادی: 1.2 تا 1.5

    # مثال: اگر به 1.5 تغییر دهید:
    # - above_average: ratio > 1.5
    # - spike: ratio > 2.25
    # - climax: ratio > 3.0
}
```

**تأثیر تغییر آستانه:**
- **کاهش آستانه (مثلاً 1.2):** سیگنال‌های بیشتری تأیید می‌شوند (حساسیت بالاتر)
- **افزایش آستانه (مثلاً 1.5):** فقط سیگنال‌های با حجم واقعاً بالا تأیید می‌شوند (دقت بالاتر)

---

##### 🎯 نکات کلیدی

1. **حجم بالا = تأیید قوی‌تر:**
   - حجم بالا نشان می‌دهد اعتماد و مشارکت بیشتر معامله‌گران در جهت حرکت
   - سیگنال‌های با حجم بالا معمولاً قابل اعتمادتر هستند

2. **الگوی Climax Volume:**
   - حجم بسیار بالا (ratio > 2.6) می‌تواند نشانه خستگی بازار باشد
   - ممکن است به برگشت قیمت منجر شود
   - باید با احتیاط بررسی شود

3. **الگوی Dry-Up:**
   - حجم بسیار پایین (ratio < 0.51) نشان‌دهنده عدم علاقه بازار است
   - سیگنال‌های با حجم بسیار پایین معمولاً ضعیف هستند

4. **تحلیل چندتایم‌فریمی:**
   - تأیید حجمی در تایم‌فریم‌های بالاتر وزن بیشتری دارد
   - اگر حجم در 4h تأیید کند، تأثیر بیشتری بر امتیاز نهایی دارد

5. **ترکیب با سایر عوامل:**
   - حجم تنها یکی از عوامل است
   - باید با روند، اندیکاتورهای مومنتوم، و رژیم بازار ترکیب شود

---

#### مرحله 4: تحلیل پیشرفته MACD

**📍 کد مرجع:** `signal_generator.py:4534-4645` - تابع `_analyze_macd()`

```python
analysis_data['macd'] = self._analyze_macd(df)
```

این بخش یک **تحلیل چندلایه و پیشرفته از MACD** انجام می‌دهد که فراتر از تحلیل ساده crossover است. این تابع 5 نوع تحلیل مختلف را ترکیب می‌کند تا سیگنال‌های دقیق‌تر تولید کند.

**⚠️ تفاوت با مرحله 2:**
- **مرحله 2 (Momentum Indicators):** تحلیل ساده MACD (crossover و zero-cross)
- **مرحله 4 (تحلیل پیشرفته):** تحلیل عمیق شامل market type، histogram patterns، trendline breaks، و divergence

---

##### 📊 اجزای تحلیل پیشرفته MACD

تابع `_analyze_macd` پنج تحلیل مستقل انجام می‌دهد:

```python
# signal_generator.py:4576-4593
market_type = self._detect_macd_market_type(dif, hist, ema20, ema50)           # 1
macd_crosses = self._detect_detailed_macd_crosses(dif, dea, df.index)          # 2
dif_behavior = self._detect_dif_behavior(dif, df.index)                        # 3
hist_analysis = self._analyze_macd_histogram(hist, close, df.index)            # 4
macd_divergence = self._detect_divergence_generic(close, dif, 'macd')          # 5
```

---

##### 1️⃣ تشخیص نوع بازار (Market Type Detection)

**📍 کد:** `signal_generator.py:3125-3150` - تابع `_detect_macd_market_type()`

این تحلیل **نوع بازار** را با ترکیب MACD، Histogram و EMA تشخیص می‌دهد:

**فرمول تصمیم‌گیری:**

| Market Type | شرایط | معنی | استراتژی |
|------------|-------|------|----------|
| `A_bullish_strong` | DIF > 0 **و** HIST > 0 **و** EMA20 > EMA50 | روند صعودی قوی | ✅ خرید قوی |
| `B_bullish_correction` | DIF > 0 **و** HIST < 0 **و** EMA20 > EMA50 | اصلاح در روند صعودی | ⚠️ منتظر بمانید |
| `C_bearish_strong` | DIF < 0 **و** HIST < 0 **و** EMA20 < EMA50 | روند نزولی قوی | ✅ فروش قوی |
| `D_bearish_rebound` | DIF < 0 **و** HIST > 0 **و** EMA20 < EMA50 | بازگشت موقت در روند نزولی | ⚠️ منتظر بمانید |
| `X_transition` | سایر موارد | انتقالی / بدون روند واضح | ❌ معامله نکنید |

**کد واقعی:**
```python
# signal_generator.py:3136-3145
if curr_dif > 0 and curr_hist > 0 and curr_ema20 > curr_ema50:
    return "A_bullish_strong"
elif curr_dif > 0 and curr_hist < 0 and curr_ema20 > curr_ema50:
    return "B_bullish_correction"
elif curr_dif < 0 and curr_hist < 0 and curr_ema20 < curr_ema50:
    return "C_bearish_strong"
elif curr_dif < 0 and curr_hist > 0 and curr_ema20 < curr_ema50:
    return "D_bearish_rebound"
else:
    return "X_transition"
```

**مثال:**
```
DIF = 150, HIST = 20, EMA20 = 50100, EMA50 = 49800
→ DIF > 0 ✓, HIST > 0 ✓, EMA20 > EMA50 ✓
→ Market Type = "A_bullish_strong"
→ استراتژی: جستجوی فرصت‌های خرید
```

---

##### 2️⃣ تقاطع‌های تفصیلی MACD (Detailed Crosses)

**📍 کد:** `signal_generator.py:3152-3246` - تابع `_detect_detailed_macd_crosses()`

این تحلیل **تقاطع DIF و DEA** را با جزئیات بیشتری بررسی می‌کند:

**سیگنال‌های صعودی (Golden Cross):**

| سیگنال | شرط | امتیاز پایه | معنی |
|--------|-----|-----------|------|
| `macd_gold_cross_below_zero` | DIF > DEA شد **و** DIF < 0 | **2.5** | تقاطع صعودی در ناحیه منفی (قوی‌تر) |
| `macd_gold_cross_above_zero` | DIF > DEA شد **و** DIF > 0 | **2.5** | تقاطع صعودی در ناحیه مثبت (ضعیف‌تر) |

**سیگنال‌های نزولی (Death Cross):**

| سیگنال | شرط | امتیاز پایه | معنی |
|--------|-----|-----------|------|
| `macd_death_cross_above_zero` | DIF < DEA شد **و** DIF > 0 | **2.5** | تقاطع نزولی در ناحیه مثبت (قوی‌تر) |
| `macd_death_cross_below_zero` | DIF < DEA شد **و** DIF < 0 | **2.5** | تقاطع نزولی در ناحیه منفی (ضعیف‌تر) |

**محاسبه قدرت تقاطع:**

```python
# signal_generator.py:3186-3187
cross_strength = min(1.0, abs(dif - dea) * 5)
signal_score = base_score * cross_strength
```

**فرمول:**
```
Cross Strength = min(1.0, |DIF - DEA| × 5)
Final Score = Base Score × Cross Strength
```

**مثال:**
```
DIF قبلی = -50, DEA قبلی = -40 → DIF < DEA
DIF فعلی = -35, DEA فعلی = -38 → DIF > DEA ✅ تقاطع صعودی!

محل تقاطع: DIF = -35 < 0 → macd_gold_cross_below_zero
Cross Strength = min(1.0, |-35 - (-38)| × 5) = min(1.0, 3 × 5) = min(1.0, 15) = 1.0
Final Score = 2.5 × 1.0 = 2.5
```

**نکته مهم:** تقاطع **زیر صفر** قوی‌تر از **بالای صفر** است چون نشان می‌دهد بازار از ناحیه ضعیف شروع به بهبود کرده.

---

##### 3️⃣ رفتار خط DIF (DIF Line Behavior)

**📍 کد:** `signal_generator.py:3281-3410` - تابع `_detect_dif_behavior()`

این تحلیل دو نوع رفتار خط DIF را بررسی می‌کند:

**الف) عبور از خط صفر (Zero Line Crosses)**

| سیگنال | شرط | امتیاز | معنی |
|--------|-----|-------|------|
| `dif_cross_zero_up_first` | اولین عبور صعودی DIF از صفر | **2.0** | شروع روند صعودی |
| `dif_cross_zero_up_second` | دومین عبور صعودی DIF از صفر | **2.0** | تقویت روند صعودی |
| `dif_cross_zero_down_first` | اولین عبور نزولی DIF از صفر | **2.0** | شروع روند نزولی |
| `dif_cross_zero_down_second` | دومین عبور نزولی DIF از صفر | **2.0** | تقویت روند نزولی |

**کد:**
```python
# signal_generator.py:3304-3316
crossed_up = dif[i-1] < 0 and dif[i] > 0
if crossed_up:
    cross_up_count += 1
    signal_type = f"dif_cross_zero_up_{'first' if cross_up_count == 1 else 'second'}"
```

**ب) شکست خطوط روند (Trendline Breaks)**

این بخش **خط روند DIF** را محاسبه کرده و شکست آن را شناسایی می‌کند:

| سیگنال | شرط | امتیاز | معنی |
|--------|-----|-------|------|
| `dif_trendline_break_up` | DIF از خط روند مقاومت عبور کرد | **3.0** | شکست صعودی - قوی |
| `dif_trendline_break_down` | DIF از خط روند حمایت عبور کرد | **3.0** | شکست نزولی - قوی |

**فرآیند:**
1. پیدا کردن قله‌ها و دره‌های DIF با median filter
2. رسم خط روند بین دو قله/دره اخیر
3. بررسی شکست خط روند توسط DIF فعلی

**کد:**
```python
# signal_generator.py:3328-3336
smooth_dif_vals = scipy.signal.medfilt(dif_for_trend.values, kernel_size=5)
peaks_iloc, valleys_iloc = self.find_peaks_and_valleys(smooth_dif_vals, ...)

# رسم خط روند: y = k*x + b
k = (p2_val - p1_val) / (p2_idx - p1_idx)
b = p1_val - k * p1_idx

# بررسی شکست
if current_dif > trendline_val + margin:  # شکست صعودی
    signal = 'dif_trendline_break_up'
```

---

##### 4️⃣ تحلیل هیستوگرام MACD (Histogram Analysis)

**📍 کد:** `signal_generator.py:3414-3509` - تابع `_analyze_macd_histogram()`

هیستوگرام MACD (HIST = DIF - DEA) الگوهای مهمی را نشان می‌دهد:

**الف) الگوهای تک‌نقطه‌ای:**

| سیگنال | شرط | امتیاز | معنی |
|--------|-----|-------|------|
| `macd_hist_shrink_head` | HIST مثبت به قله رسید | **1.5** | کاهش قدرت صعود - احتمال برگشت |
| `macd_hist_pull_feet` | HIST منفی به کف رسید | **1.5** | کاهش قدرت نزول - احتمال برگشت |

**کد:**
```python
# signal_generator.py:3433-3442
peaks_iloc, valleys_iloc = self.find_peaks_and_valleys(hist.values, ...)
for idx in peaks_iloc:
    if hist[idx] > 0:
        signals.append({'type': 'macd_hist_shrink_head', 'score': 1.5})
```

**ب) واگرایی هیستوگرام (Histogram Divergence):**

| سیگنال | شرط | امتیاز | معنی |
|--------|-----|-------|------|
| `macd_hist_top_divergence` | قیمت HH ولی HIST LH | **3.8** | واگرایی نزولی - قوی |
| `macd_hist_bottom_divergence` | قیمت LL ولی HIST HL | **3.8** | واگرایی صعودی - قوی |

**کد:**
```python
# signal_generator.py:3455-3466
if len(peaks) >= 2:
    p1, p2 = peaks[-2], peaks[-1]
    # قیمت بالاتر رفته ولی HIST پایین‌تر → واگرایی نزولی
    if hist[p2] < hist[p1] and close[p2] > close[p1]:
        signals.append({'type': 'macd_hist_top_divergence', 'score': 3.8})
```

**ج) الگوی Kill Long Bin:**

| سیگنال | شرط | امتیاز | معنی |
|--------|-----|-------|------|
| `macd_hist_kill_long_bin` | HIST بین دو دره همیشه منفی | **2.0** | فشار فروش مداوم |

این الگو نشان می‌دهد که HIST بین دو دره به بالای صفر نرسیده → فشار فروش قوی.

```python
# signal_generator.py:3481-3494
if len(valleys) >= 2:
    v1, v2 = valleys[-2], valleys[-1]
    hist_between = hist[v1:v2+1]
    if hist_between.max() < 0:  # همیشه منفی بوده
        signals.append({'type': 'macd_hist_kill_long_bin', 'score': 2.0})
```

---

##### 5️⃣ واگرایی MACD (MACD Divergence)

**📍 کد:** `signal_generator.py:4589-4590` - استفاده از `_detect_divergence_generic()`

واگرایی بین **قیمت** و **خط DIF** را شناسایی می‌کند (این تابع قبلاً در بخش Momentum Indicators توضیح داده شد).

**سیگنال‌های احتمالی:**
- `macd_bullish_regular_divergence`
- `macd_bearish_regular_divergence`
- `macd_bullish_hidden_divergence`
- `macd_bearish_hidden_divergence`

---

##### 📊 خروجی نهایی تابع _analyze_macd

```python
{
    'status': 'ok',
    'market_type': 'A_bullish_strong',
    'direction': 'bullish',                # یا 'bearish' یا 'neutral'
    'bullish_score': 8.3,                  # مجموع امتیازات صعودی
    'bearish_score': 2.0,                  # مجموع امتیازات نزولی
    'signals': [
        {
            'type': 'macd_gold_cross_below_zero',
            'direction': 'bullish',
            'score': 2.5,
            'strength': 1.0,
            'details': {'dif': -35, 'dea': -38, 'above_zero': False}
        },
        {
            'type': 'macd_hist_bottom_divergence',
            'direction': 'bullish',
            'score': 3.8
        },
        {
            'type': 'dif_trendline_break_up',
            'direction': 'bullish',
            'score': 3.0
        }
    ],
    'details': {
        'dif': -35.2,
        'dea': -38.1,
        'hist': 2.9,
        'dif_slope': 5.3,          # شیب DIF (مثبت = صعودی)
        'dea_slope': 3.2,          # شیب DEA
        'hist_slope': 2.1,         # شیب Histogram
        'market_type': 'A_bullish_strong'
    }
}
```

---

##### 🎯 محاسبه جهت نهایی

```python
# signal_generator.py:4596-4605
bullish_score = sum(s['score'] for s in signals if s['direction'] == 'bullish')
bearish_score = sum(s['score'] for s in signals if s['direction'] == 'bearish')

if bullish_score > bearish_score * 1.1:
    direction = 'bullish'
elif bearish_score > bullish_score * 1.1:
    direction = 'bearish'
else:
    direction = 'neutral'
```

**فرمول:**
- اگر `bullish_score > bearish_score × 1.1` → جهت صعودی
- اگر `bearish_score > bullish_score × 1.1` → جهت نزولی
- در غیر این صورت → خنثی

**مثال:**
```
bullish_score = 8.3 (cross: 2.5 + divergence: 3.8 + trendline: 3.0)
bearish_score = 2.0

8.3 > 2.0 × 1.1 = 2.2 ✓
→ direction = 'bullish'
```

---

##### 📝 مثال کامل تحلیل MACD

**وضعیت بازار:**
```
DIF = -35, DEA = -38, HIST = 3
DIF قبلی = -50, DEA قبلی = -40
EMA20 = 50100, EMA50 = 49800
قیمت فعلی = 50050
```

**تحلیل‌ها:**

1. **Market Type:**
   - DIF < 0, HIST > 0, EMA20 > EMA50 → `D_bearish_rebound` (بازگشت موقت)

2. **MACD Cross:**
   - DIF(-35) > DEA(-38) و قبلاً DIF(-50) < DEA(-40) بود
   - تقاطع صعودی زیر صفر → `macd_gold_cross_below_zero`
   - Cross strength = min(1.0, |-35-(-38)| × 5) = 1.0
   - Score = 2.5 × 1.0 = **2.5**

3. **DIF Behavior:**
   - DIF از -50 به -35 رسیده (صعودی) ولی هنوز زیر صفر
   - فرض: شکست خط روند → `dif_trendline_break_up`
   - Score = **3.0**

4. **Histogram:**
   - HIST = 3 > 0 (مثبت شده)
   - فرض: واگرایی کف → `macd_hist_bottom_divergence`
   - Score = **3.8**

5. **Divergence:**
   - فرض: واگرایی صعودی معمولی شناسایی شد
   - Score = **3.5**

**نتیجه نهایی:**
```
bullish_score = 2.5 + 3.0 + 3.8 + 3.5 = 12.8
bearish_score = 0

direction = 'bullish' (قوی)
```

---

##### 🎯 امتیازدهی در سیستم

تمام سیگنال‌های MACD در لیست `signals` قرار می‌گیرند و در محاسبه امتیاز نهایی استفاده می‌شوند:

```python
for signal in macd_result['signals']:
    if signal['direction'] == trade_direction:
        total_score += signal['score']
```

**خلاصه امتیازات:**

| سیگنال | امتیاز پایه | دسته‌بندی قدرت |
|--------|-----------|----------------|
| Golden/Death Cross | 2.5 | متوسط |
| DIF Zero Cross | 2.0 | متوسط |
| **DIF Trendline Break** | **3.0** | **قوی** |
| Histogram Peaks/Valleys | 1.5 | ضعیف |
| **Histogram Divergence** | **3.8** | **بسیار قوی** |
| Kill Long Bin | 2.0 | متوسط |
| **MACD Divergence** | **3.5** | **بسیار قوی** |

---

##### 🔑 نکات کلیدی

1. **تحلیل چندلایه:** MACD از 5 جنبه مختلف تحلیل می‌شود
2. **Market Type مهم است:** نوع بازار استراتژی معاملاتی را تعیین می‌کند
3. **محل تقاطع:** تقاطع زیر صفر قوی‌تر از بالای صفر است
4. **واگرایی = طلا:** واگرایی‌های MACD بالاترین امتیاز را دارند (3.8)
5. **Histogram = تأیید کننده:** تغییرات هیستوگرام نشان‌دهنده تغییر قدرت روند است
6. **Trendline Breaks:** شکست خطوط روند DIF سیگنال‌های قوی هستند (3.0)

---

#### مرحله 5: تحلیل Price Action (الگوهای شمعی و تحلیل‌های فنی)

**📍 کد مرجع:** `signal_generator.py:3867-4014` - تابع `analyze_price_action()`

```python
analysis_data['price_action'] = await self.analyze_price_action(df)
```

این بخش **جامع‌ترین تحلیل فنی** را انجام می‌دهد و شامل 4 دسته اصلی است:
1. الگوهای شمعی (Candlestick Patterns)
2. الگوهای چند-کندلی (Multi-Candle Patterns)
3. تحلیل Bollinger Bands
4. تحلیل ترکیبی حجم و قیمت

---

##### 1️⃣ الگوهای شمعی تک-کندلی (Single Candle Patterns)

**📍 کد:** `signal_generator.py:1839-1953` - تابع `detect_candlestick_patterns()`

سیستم با استفاده از **TA-Lib** این الگوها را شناسایی می‌کند:

| الگو | نام فارسی | جهت | امتیاز پایه | نوع سیگنال |
|------|-----------|-----|------------|-----------|
| `hammer` | چکش | Bullish | **2.0** | برگشتی صعودی |
| `inverted_hammer` | چکش وارونه | Bullish | **2.0** | برگشتی صعودی |
| `engulfing` | پوششی | Neutral* | **2.5** | قوی (جهت بستگی به value دارد) |
| `morning_star` | ستاره صبحگاهی | Bullish | **3.0** | برگشتی قوی صعودی |
| `evening_star` | ستاره عصرگاهی | Bearish | **3.0** | برگشتی قوی نزولی |
| `harami` | حامله | Neutral* | **1.8** | تردید/برگشت |
| `doji` | دوجی | Neutral | **1.5** | تردید بازار |
| `dragonfly_doji` | دوجی سنجاقک | Bullish | **2.2** | برگشتی صعودی |
| `gravestone_doji` | دوجی سنگ قبر | Bearish | **2.2** | برگشتی نزولی |
| `shooting_star` | ستاره دنباله‌دار | Bearish | **2.3** | برگشتی نزولی |
| `marubozu` | مارابوزو | Neutral* | **2.0** | قوی (بدون سایه) |
| `hanging_man` | مرد آویزان | Bearish | **2.1** | برگشتی نزولی |

*جهت Neutral به معنی است که جهت الگو توسط خود کتابخانه تعیین می‌شود (بر اساس value مثبت/منفی).

**محاسبه قدرت و امتیاز:**

```python
# signal_generator.py:1931-1936
pattern_strength = min(1.0, abs(pattern_value) / 100)
if pattern_strength < 0.1:
    pattern_strength = 0.7  # حداقل قدرت

pattern_score = base_score * pattern_strength
```

**فرمول:**
```
Pattern Strength = min(1.0, |pattern_value| / 100)
Final Score = Base Score × Pattern Strength
```

**مثال:**
```
Hammer detected: pattern_value = 85
Pattern Strength = min(1.0, 85/100) = 0.85
Base Score = 2.0
Final Score = 2.0 × 0.85 = 1.7
```

---

##### 2️⃣ الگوهای چند-کندلی (Multi-Candle Patterns)

**📍 کد:** `signal_generator.py:1955-2310`

**الف) Head and Shoulders (سر و شانه)**

**📍 کد:** `signal_generator.py:1976-2118`

یکی از قوی‌ترین الگوهای برگشتی:

**ساختار الگو:**
```
        Head (سر)
       /    \
      /      \
 L.Shoulder  R.Shoulder
    /          \
 Dip1 ------- Dip2 (Neckline خط گردن)
```

**شرایط تشخیص:**
1. پیدا کردن 3 قله (left shoulder, head, right shoulder)
2. `head_price > left_shoulder_price` و `head_price > right_shoulder_price`
3. دو شانه تقریباً هم‌سطح: `shoulder_diff < 10%`
4. فاصله زمانی متقارن: `time_gap_ratio > 0.6`
5. دو دره بین قله‌ها (dips) برای تشکیل neckline
6. neckline تقریباً افقی: `neckline_diff < 5%`

**کد واقعی:**
```python
# signal_generator.py:2000-2003
if head_price > left_shoulder_price and head_price > right_shoulder_price:
    shoulder_diff_percent = abs(right_shoulder_price - left_shoulder_price) / left_shoulder_price
    if shoulder_diff_percent < 0.1:  # شانه‌ها هم‌سطح
```

**محاسبه Price Target:**
```python
pattern_height = head_price - neckline_price
price_target = neckline_price - pattern_height  # برای bearish
```

**امتیازدهی:**
```python
# signal_generator.py:2029-2041
pattern_quality = (1.0 - shoulder_diff) × time_gap_ratio × (1.0 - neckline_diff)
score = 4.0 × pattern_quality
```

**خروجی:**
```python
{
    'type': 'head_and_shoulders',
    'direction': 'bearish',
    'breakout_confirmed': True/False,
    'neckline_price': 50000,
    'price_target': 48500,
    'pattern_quality': 0.85,
    'score': 3.4,
    'points': {...}
}
```

**Inverse Head & Shoulders (سر و شانه معکوس)**

همان منطق ولی با دره‌ها و قله‌های معکوس:
- `head_price < left_shoulder` و `head_price < right_shoulder`
- Direction: Bullish
- Price Target: `neckline + pattern_height`

---

**ب) Triangle Patterns (الگوهای مثلث)**

**📍 کد:** `signal_generator.py:2120-2219`

**3 نوع مثلث:**

| نوع | شرایط | جهت | امتیاز پایه |
|-----|-------|-----|------------|
| **Ascending Triangle** | خط بالا افقی، خط پایین صعودی | Bullish | **3.5** |
| **Descending Triangle** | خط بالا نزولی، خط پایین افقی | Bearish | **3.5** |
| **Symmetric Triangle** | خط بالا نزولی، خط پایین صعودی | بستگی به موقعیت | **3.5** |

**تشخیص:**
```python
# signal_generator.py:2154-2156
is_ascending = abs(upper_slope) < 0.001 and lower_slope > 0.001
is_descending = upper_slope < -0.001 and abs(lower_slope) < 0.001
is_symmetric = upper_slope < -0.001 and lower_slope > 0.001
```

**محاسبه نقطه همگرایی (Convergence Point):**
```python
# signal_generator.py:2158-2160
convergence_x = (lower_intercept - upper_intercept) / (upper_slope - lower_slope)
convergence_y = upper_slope * convergence_x + upper_intercept
```

**Pattern Quality:**
```python
# signal_generator.py:2173-2175
total_touches = len(peaks) + len(valleys)
pattern_quality = min(1.0, total_touches / 6) × min(1.0, 1.0 - width / (upper * 0.2))
```

**Price Target:**
```python
pattern_height = max(highs[peaks]) - min(lows[valleys])
target = current_price ± pattern_height
```

---

**ج) Flag Patterns (الگوهای پرچم)**

**📍 کد:** `signal_generator.py:2224-2310`

پرچم‌ها الگوهای ادامه‌دهنده روند هستند:

**ساختار:**
```
    /|      (Pole - میله پرچم)
   / |
  /  |
 /   |
/    ////   (Flag - پرچم)
    ////
   ////
```

**شرایط تشخیص:**

1. **Pole (میله):** حرکت قوی قیمت
   ```python
   # signal_generator.py:2243-2244
   pole_price_change_pct = (closes[pole_end] - closes[pole_start]) / closes[pole_start]

   is_bullish_pole = pole_price_change_pct > 0.03  # 3% افزایش
   is_bearish_pole = pole_price_change_pct < -0.03  # 3% کاهش
   ```

2. **Volume قوی در Pole:**
   ```python
   pole_volume > avg_volume × 1.5
   ```

3. **Flag:** اصلاح کوچک با خطوط موازی
   - Bull Flag: شیب‌های منفی (اصلاح نزولی)
   - Bear Flag: شیب‌های مثبت (اصلاح صعودی)

**کد تشخیص:**
```python
# signal_generator.py:2274-2283
if is_bullish_pole:
    is_valid_flag = (upper_slope < 0 and lower_slope < 0) or are_lines_parallel
    pattern_type = 'bull_flag'
elif is_bearish_pole:
    is_valid_flag = (upper_slope > 0 and lower_slope > 0) or are_lines_parallel
    pattern_type = 'bear_flag'
```

**Pattern Quality:**
```python
flag_quality = (1.0 if strong_volume else 0.7) × (1.0 - slopes_diff / 0.001)
```

**Price Target:**
```python
pole_height = abs(pole_price_change)
price_target = current_price + pole_height  # bull flag
price_target = current_price - pole_height  # bear flag
```

---

##### 3️⃣ تحلیل Bollinger Bands

**📍 کد:** `signal_generator.py:3893-3948`

**محاسبه:**
```python
upper, middle, lower = talib.BBANDS(close, timeperiod=20, nbdevup=2, nbdevdn=2)
```

**الف) BB Position:**
```python
bb_position = (current_price - lower_band) / (upper_band - lower_band)
```
- `bb_position = 0`: قیمت در باند پایین
- `bb_position = 0.5`: قیمت در میانه
- `bb_position = 1.0`: قیمت در باند بالا

**ب) BB Width و Squeeze:**
```python
bb_width = (upper - lower) / middle
bb_squeeze = bb_width < avg_width × 0.8
```

**سیگنال‌های Bollinger:**

| سیگنال | شرط | جهت | امتیاز |
|--------|-----|-----|--------|
| `bollinger_squeeze` | عرض باند < 80% میانگین | Neutral | **2.0** |
| `bollinger_upper_break` | قیمت > باند بالا | Bullish | **2.5** |
| `bollinger_lower_break` | قیمت < باند پایین | Bearish | **2.5** |

**کد:**
```python
# signal_generator.py:3929-3947
if bb_squeeze:
    signals.append({'type': 'bollinger_squeeze', 'score': 2.0})

if current_close > current_upper:
    signals.append({'type': 'bollinger_upper_break', 'direction': 'bullish', 'score': 2.5})
elif current_close < current_lower:
    signals.append({'type': 'bollinger_lower_break', 'direction': 'bearish', 'score': 2.5})
```

---

##### 4️⃣ تحلیل ترکیبی حجم و قیمت

**📍 کد:** `signal_generator.py:3953-3982`

**محاسبه:**
```python
avg_volume = mean(volume[-30:-1])
current_volume = volume[-1]
volume_ratio = current_volume / avg_volume
```

**سیگنال‌های حجم:**

| سیگنال | شرط | امتیاز |
|--------|-----|--------|
| `high_volume_bullish` | volume_ratio > 2.5 **و** کندل سبز | **2.8** |
| `high_volume_bearish` | volume_ratio > 2.5 **و** کندل قرمز | **2.8** |

**کد:**
```python
# signal_generator.py:3970-3982
if volume_ratio > 2.5:
    if current_close > df['open'].iloc[-1]:  # Bullish candle
        signals.append({'type': 'high_volume_bullish', 'score': 2.8})
    else:  # Bearish candle
        signals.append({'type': 'high_volume_bearish', 'score': 2.8})
```

---

##### 📊 خروجی نهایی تابع analyze_price_action

```python
{
    'status': 'ok',
    'direction': 'bullish',
    'bullish_score': 8.5,
    'bearish_score': 2.0,
    'atr': 125.5,
    'signals': [
        {
            'type': 'hammer',
            'direction': 'bullish',
            'score': 1.7,
            'strength': 0.85
        },
        {
            'type': 'bull_flag',
            'direction': 'bullish',
            'score': 2.7,
            'pattern_quality': 0.9,
            'price_target': 51200
        },
        {
            'type': 'bollinger_lower_break',
            'direction': 'bearish',
            'score': 2.5
        }
    ],
    'details': {
        'candle_patterns': [...],
        'bollinger_bands': {
            'upper': 50500,
            'middle': 50000,
            'lower': 49500,
            'position': 0.25,
            'width': 0.02,
            'squeeze': False
        },
        'volume_analysis': {
            'current_volume': 5500000,
            'avg_volume': 3200000,
            'volume_ratio': 1.72,
            'is_high_volume': True
        }
    }
}
```

---

##### 🎯 امتیازدهی نهایی

**محاسبه جهت:**
```python
# signal_generator.py:3991-4006
bullish_score = sum(s['score'] for s in signals if s['direction'] == 'bullish')
bearish_score = sum(s['score'] for s in signals if s['direction'] == 'bearish')

if bullish_score > bearish_score:
    direction = 'bullish'
elif bearish_score > bullish_score:
    direction = 'bearish'
else:
    direction = 'neutral'
```

**خلاصه امتیازات:**

| دسته الگو | بازه امتیاز | مثال |
|-----------|-------------|------|
| الگوهای شمعی تک-کندلی | 1.5 - 3.0 | Hammer: 2.0, Morning Star: 3.0 |
| Head & Shoulders | 3.0 - 4.0 | با quality بالا: 4.0 |
| Triangle Patterns | 2.5 - 3.5 | با quality بالا: 3.5 |
| Flag Patterns | 2.0 - 3.0 | با volume قوی: 3.0 |
| Bollinger Signals | 2.0 - 2.5 | Break: 2.5, Squeeze: 2.0 |
| High Volume Signals | 2.8 | با کندل قوی |

---

##### 🔑 نکات کلیدی

1. **الگوهای چند-کندلی قوی‌تر:** Head & Shoulders و Flag امتیاز بالاتر از الگوهای تک-کندلی دارند

2. **Pattern Quality مهم است:** تمام الگوهای چند-کندلی دارای `pattern_quality` هستند که در امتیاز ضرب می‌شود

3. **Price Target:** الگوهای پیچیده (H&S, Triangle, Flag) price target محاسبه می‌کنند

4. **Bollinger Squeeze = آرامش قبل از طوفان:** نشانه انفجار حرکت آینده

5. **حجم = تأیید کننده:** سیگنال‌هایی که با حجم بالا همراه باشند قوی‌تر هستند

6. **ATR = معیار نوسانات:** برای محاسبه SL/TP استفاده می‌شود

---

#### مرحله 6: شناسایی سطوح حمایت/مقاومت (Support/Resistance Detection)

**محل:** `signal_generator.py:2312-2414`

```python
analysis_data['support_resistance'] = self.detect_support_resistance(df, lookback=50)
```

این تحلیل **سطوح کلیدی حمایت و مقاومت** را شناسایی کرده و قدرت آنها را محاسبه می‌کند. سطوح S/R نقاطی هستند که قیمت بارها در آنها واکنش نشان داده است.

---

##### 🔍 الگوریتم شناسایی (4 مرحله اصلی)

**مرحله 1: پیدا کردن Peaks و Valleys (نقاط بازگشت)**

```python
# استفاده از scipy.signal.find_peaks با فیلترهای کیفی
resistance_peaks, _ = self.find_peaks_and_valleys(
    highs,
    order=3,           # حداقل 3 کندل برای تشکیل peak
    distance=5         # حداقل فاصله بین peaks
)
_, support_valleys = self.find_peaks_and_valleys(
    lows,
    order=3,
    distance=5
)
```

**کد مرجع:** `signal_generator.py:1605-1656`

**فرآیند:**
1. **Peak Detection:** قله‌های قیمت با `scipy.signal.find_peaks()` شناسایی می‌شوند
2. **Valley Detection:** دره‌های قیمت با اعمال peak detection روی `-data`
3. **Prominence Filter:** فقط peaks با برجستگی بالا (`prominence >= median * 0.5`) حفظ می‌شوند
4. **Quality Filter:** حذف peaks ضعیف بر اساس `width` و `rel_height`

**فرمول Prominence:**
```python
prominence = np.std(data) * prominence_factor  # prominence_factor = 0.1
quality_threshold = np.median(prominences) * 0.5
valid_peaks = peaks[prominences >= quality_threshold]
```

---

**مرحله 2: ادغام سطوح نزدیک (Level Consolidation)**

سطوحی که به هم نزدیک هستند در یک **cluster** ادغام می‌شوند:

```python
def consolidate_levels(levels: np.ndarray, atr: float):
    threshold = atr * 0.3  # سطوح نزدیکتر از 30% ATR ادغام می‌شوند

    # Clustering الگوریتم
    for level in sorted_levels:
        if abs(level - cluster_mean) <= threshold:
            current_cluster.append(level)  # اضافه به cluster فعلی
        else:
            save_cluster()                 # ذخیره cluster قبلی
            start_new_cluster(level)       # شروع cluster جدید
```

**کد مرجع:** `signal_generator.py:2333-2370`

**محاسبه قدرت Cluster:**
```python
cluster_mean = np.mean(current_cluster)           # میانگین قیمت‌های cluster
cluster_strength = min(1.0, len(cluster) / 3) *   # تعداد تست‌ها (max = 3)
                   (1.0 - std/mean)                # یکنواختی cluster
```

**فاکتورهای قدرت:**
- **تعداد تست‌ها:** هر چه سطح بیشتر تست شود → قوی‌تر
- **یکنواختی:** cluster متمرکزتر → قوی‌تر (std کمتر)

**مثال:**
```python
# سطوح خام: [50000, 50050, 50100, 51000, 51020]
# با ATR = 200 → threshold = 60

Cluster 1: [50000, 50050]  # فاصله < 60
→ mean = 50025, strength = 0.67 * 0.999 = 0.66

Cluster 2: [50100]
→ mean = 50100, strength = 0.33

Cluster 3: [51000, 51020]  # فاصله < 60
→ mean = 51010, strength = 0.67 * 0.998 = 0.66
```

---

**مرحله 3: تشخیص شکست سطوح (Breakout Detection)**

**کد مرجع:** `signal_generator.py:2384-2387`

```python
# شکست مقاومت (صعودی)
broken_resistance = next((level for level in resistance_levels if
    current_close > level['price'] and      # قیمت فعلی بالاتر از سطح
    prev_low < level['price']               # کندل قبلی زیر سطح بود
), None)

# شکست حمایت (نزولی)
broken_support = next((level for level in support_levels if
    current_close < level['price'] and      # قیمت فعلی پایین‌تر از سطح
    prev_high > level['price']              # کندل قبلی بالای سطح بود
), None)
```

**شرایط Breakout:**
1. قیمت فعلی **از سطح عبور کند**
2. کندل قبلی **در سمت دیگر سطح** بوده باشد
3. یعنی breakout در همین کندل اتفاق افتاده (تازه شکسته)

---

**مرحله 4: تحلیل Zone ها (ناحیه‌های چند لایه)**

**کد مرجع:** `signal_generator.py:2416-2463`

برخی نواحی **چند سطح S/R نزدیک هم** دارند که یک **Zone قوی** می‌سازند:

```python
def _analyze_sr_zones(levels, current_price, zone_type):
    # Clustering سطوح که < 1% فاصله دارند
    for i in range(1, len(sorted_levels)):
        distance_pct = abs(level[i] - level[i-1]) / level[i-1]
        if distance_pct < 0.01:  # کمتر از 1%
            current_cluster.append(level[i])
        else:
            if len(cluster) >= 2:  # حداقل 2 سطح
                zones.append(cluster)
```

**مشخصات هر Zone:**
```python
{
    'min': 49900,              # کف zone
    'max': 50100,              # سقف zone
    'center': 50000,           # مرکز zone
    'width': 200,              # عرض zone (max - min)
    'strength': 0.85,          # میانگین قدرت سطوح داخل zone
    'levels_count': 3,         # تعداد سطوح در zone
    'distance_to_price': 150   # فاصله تا قیمت فعلی
}
```

**کاربرد Zones:**
- Zone های عریض → **ناحیه مهم تردید**
- Zone با `levels_count` بالا → **بسیار قوی**
- نزدیک بودن به zone → **احتمال واکنش قیمت**

---

##### 📊 خروجی کامل

```python
{
    'status': 'ok',

    # سطوح مقاومت (بالای قیمت فعلی)
    'resistance_levels': [
        {'price': 50200, 'strength': 0.85},
        {'price': 51000, 'strength': 0.92},
        {'price': 52500, 'strength': 0.67}
    ],

    # سطوح حمایت (پایین قیمت فعلی)
    'support_levels': [
        {'price': 49800, 'strength': 0.90},
        {'price': 48500, 'strength': 0.78},
        {'price': 47200, 'strength': 0.65}
    ],

    # جزئیات
    'details': {
        'nearest_resistance': {'price': 50200, 'strength': 0.85},
        'nearest_support': {'price': 49800, 'strength': 0.90},
        'broken_resistance': None,                    # یا {'price': ..., 'strength': ...}
        'broken_support': None,
        'atr': 180.5
    },

    # Zone های مقاومت
    'resistance_zones': {
        'status': 'ok',
        'zones': [
            {
                'min': 50150, 'max': 50250, 'center': 50200,
                'width': 100, 'strength': 0.88,
                'levels_count': 3, 'distance_to_price': 200
            }
        ]
    },

    # Zone های حمایت
    'support_zones': {
        'status': 'ok',
        'zones': [
            {
                'min': 49750, 'max': 49850, 'center': 49800,
                'width': 100, 'strength': 0.91,
                'levels_count': 2, 'distance_to_price': 200
            }
        ]
    }
}
```

---

##### 💯 امتیازدهی

**کد مرجع:** `signal_generator.py:5284-5297`

**فقط شکست سطوح امتیاز می‌دهند:**

```python
# 1. شکست مقاومت (Bullish)
if broken_resistance:
    level_strength = broken_resistance['strength']  # 0.0 تا 1.0
    base_score = pattern_scores['broken_resistance']  # 3.0
    score = base_score * timeframe_weight * level_strength
    bullish_score += score

    # مثال: 3.0 * 1.0 * 0.85 = +2.55 امتیاز

# 2. شکست حمایت (Bearish)
if broken_support:
    level_strength = broken_support['strength']
    base_score = pattern_scores['broken_support']  # 3.0
    score = base_score * timeframe_weight * level_strength
    bearish_score += score

    # مثال: 3.0 * 1.0 * 0.90 = +2.70 امتیاز
```

**جدول امتیازات:**

| سیگنال | امتیاز پایه | فاکتور قدرت | محدوده نهایی | نوع |
|--------|------------|-------------|--------------|-----|
| `broken_resistance` | **3.0** | `level_strength` (0.5-1.0) | **1.5 تا 3.0** | صعودی |
| `broken_support` | **3.0** | `level_strength` (0.5-1.0) | **1.5 تا 3.0** | نزولی |

**⚠️ نکته مهم:**
- در کد فعلی **فقط breakout ها** امتیاز می‌دهند
- قرار گرفتن **نزدیک سطوح** امتیاز ندارد (یک مشکل!)
- این در بخش پیشنهادات بهبود برطرف می‌شود

---

##### 🎯 کاربردها در سیستم

**1. محاسبه Stop Loss/Take Profit:**

**کد:** `signal_generator.py:4127-4212`

```python
# Stop Loss از نزدیکترین سطح
if direction == 'long' and nearest_support:
    stop_loss = nearest_support * 0.999  # کمی زیر حمایت
elif direction == 'short' and nearest_resistance:
    stop_loss = nearest_resistance * 1.001  # کمی بالای مقاومت

# Take Profit تا نزدیکترین مانع
if direction == 'long' and nearest_resistance:
    if nearest_resistance > current_price + (risk * min_rr):
        take_profit = nearest_resistance * 0.999
elif direction == 'short' and nearest_support:
    if nearest_support < current_price - (risk * min_rr):
        take_profit = nearest_support * 1.001
```

**2. تأیید Reversal Signals:**

**کد:** `signal_generator.py:3754-3772`

```python
if current_close and broken_resistance:
    # قیمت تازه مقاومت را شکسته
    if abs(current_close - broken_resistance) / broken_resistance < 0.01:
        strength += 0.6  # تقویت سیگنال برگشتی
        is_reversal = True

if current_close and broken_support:
    # قیمت تازه حمایت را شکسته
    if abs(current_close - broken_support) / broken_support < 0.01:
        strength += 0.6
        is_reversal = True
```

**3. Higher Timeframe Zone Analysis:**

**کد:** `signal_generator.py:4363-4377`

```python
# بررسی zone های HTF (Higher Timeframe)
for zone in htf_resistance_zones:
    dist = abs(zone['center'] - current_price)
    if dist < nearest_resistance_distance:
        nearest_htf_resistance = zone

# Position relative to HTF zones → تأثیر در تصمیم‌گیری
```

---

##### 📈 مثال واقعی

**سناریو:** قیمت BTC در 49,950 USDT

```python
# سطوح شناسایی شده
resistance_levels = [
    {'price': 50200, 'strength': 0.85},  # 3 بار تست شده
    {'price': 51500, 'strength': 0.70}
]
support_levels = [
    {'price': 49800, 'strength': 0.90},  # 4 بار تست شده
    {'price': 48000, 'strength': 0.65}
]

# قیمت فعلی: 49,950
nearest_resistance = {'price': 50200, 'strength': 0.85}  # فاصله: 250 (0.5%)
nearest_support = {'price': 49800, 'strength': 0.90}     # فاصله: 150 (0.3%)

# قیمت بین دو سطح قوی گیر کرده → Range محتمل
```

**اگر قیمت به 50,220 برسد:**
```python
# Breakout تأیید می‌شود
broken_resistance = {'price': 50200, 'strength': 0.85}
score = 3.0 * 1.0 * 0.85 = +2.55 امتیاز صعودی

# + تنظیم SL/TP:
stop_loss = 50200 * 1.001 = 50,250  # بالای سطح شکسته شده (pullback)
take_profit = 51500 * 0.999 = 51,450  # نزدیک مقاومت بعدی
```

---

##### ✅ نکات کلیدی

1. **ATR-Based Clustering:** سطوح بر اساس نوسانات واقعی (ATR) ادغام می‌شوند

2. **Dynamic Quality Filter:** فقط peaks با کیفیت بالا حفظ می‌شوند

3. **Multi-Layer Zones:** سطوح نزدیک → zone قوی‌تر

4. **Breakout Confirmation:** شکست باید در همان کندل رخ دهد (نه کندل‌های قبل)

5. **Integration با SL/TP:** سطوح S/R مستقیماً در مدیریت ریسک استفاده می‌شوند

6. **⚠️ محدودیت فعلی:** قرار گرفتن نزدیک سطوح امتیاز نمی‌دهد (فقط breakout)

---

## بخش ۳: تحلیل‌های پیشرفته (Advanced Analysis)

در کنار تحلیل‌های پایه، سیستم **تحلیل‌های پیشرفته‌ای** نیز انجام می‌دهد که امتیازات بالاتری تولید می‌کنند.

### 3.1 شناسایی الگوهای هارمونیک (Harmonic Patterns)

**محل:** `signal_generator.py:2465-2665`

```python
analysis_data['harmonic_patterns'] = self.detect_harmonic_patterns(
    df, lookback=100, tolerance=0.03  # ±3%
)
```

الگوهای هارمونیک بر اساس **نسبت‌های دقیق فیبوناچی** بین 5 نقطه بازگشت (X-A-B-C-D) تشکیل می‌شوند. این الگوها بسیار قوی و نادر هستند.

---

##### الگوریتم شناسایی (4 مرحله)

**مرحله 1: شناسایی X-A-B-C-D**

```python
# 1. Peaks/Valleys
peaks, valleys = self.find_peaks_and_valleys(df['close'].values)
all_points = [(idx, 'peak'/'valley', price), ...]
all_points.sort(key=lambda x: x[0])  # مرتب‌سازی زمانی

# 2. انتخاب 5 نقطه متوالی
for i in range(len(all_points) - 4):
    X, A, B, C, D = all_points[i:i + 5]

    # شرط: تناوب peak/valley (X≠A≠B≠C≠D)
    if not all_alternating:
        continue
```

**کد:** `signal_generator.py:2475-2492`

---

**مرحله 2: محاسبه نسبت‌های فیبوناچی**

```python
xa = abs(x_price - a_price)
ab = abs(a_price - b_price)
bc = abs(b_price - c_price)
cd = abs(c_price - d_price)

ab_xa = ab / xa      # نسبت AB به XA
bc_ab = bc / ab      # نسبت BC به AB
cd_bc = cd / bc      # نسبت CD به BC
bd_ba = abs(d_price - b_price) / abs(a_price - b_price)
```

**کد:** `signal_generator.py:2500-2511`

---

**مرحله 3: تطبیق با 4 الگو**

| الگو | AB/XA | BC/AB | CD/BC | BD/BA | ویژگی |
|------|-------|-------|-------|-------|-------|
| **Gartley** | **0.618** | **0.382** | **1.272** | **0.786** | محافظه‌کارانه |
| **Bat** | **0.382** | **0.382** | **1.618** | **0.886** | بازگشت عمیق (88.6%) |
| **Butterfly** | **0.786** | **0.382** | **1.618** | **1.27** | تجاوز از X |
| **Crab** | **0.382** | **0.618** | **3.618** | **1.618** | شدیدترین (تجاوز 161.8%) |

**کد:**
- Gartley: `2515-2549`
- Bat: `2551-2585`
- Butterfly: `2587-2621`
- Crab: `2623-2657`

**تطبیق با Tolerance:**
```python
is_in_range = lambda val, target: abs(val - target) <= 0.03  # ±3%

# مثال Gartley:
is_gartley = (
    is_in_range(ab_xa, 0.618) and
    is_in_range(bc_ab, 0.382) and
    is_in_range(cd_bc, 1.272) and
    is_in_range(bd_ba, 0.786)
)
```

---

**مرحله 4: محاسبه Confidence**

```python
confidence = 1.0 - max(
    abs(ab_xa - target1),
    abs(bc_ab - target2),
    abs(cd_bc - target3),
    abs(bd_ba - target4)
) / tolerance  # 0.03
```

**مثال:**
```python
# نسبت‌های واقعی:
ab_xa = 0.625  # هدف: 0.618 → انحراف: 0.007
bc_ab = 0.380  # هدف: 0.382 → انحراف: 0.002
cd_bc = 1.280  # هدف: 1.272 → انحراف: 0.008 (max)
bd_ba = 0.790  # هدف: 0.786 → انحراف: 0.004

confidence = 1.0 - (0.008 / 0.03) = 0.733 = 73.3%
```

**فیلتر:** فقط `confidence >= 0.7` قبول می‌شوند

**کد:** `signal_generator.py:2522-2527, 2558-2563, ...`

---

##### خروجی

```python
[
    {
        'type': 'bullish_gartley',
        'direction': 'bullish',
        'confidence': 0.92,

        'points': {
            'X': {'index': 10, 'price': 50000.0},
            'A': {'index': 15, 'price': 49000.0},
            'B': {'index': 22, 'price': 49618.0},
            'C': {'index': 28, 'price': 49382.0},
            'D': {'index': 35, 'price': 49786.0}  # نقطه ورود
        },

        'ratios': {
            'AB/XA': 0.618, 'BC/AB': 0.382,
            'CD/BC': 1.275, 'BD/BA': 0.788
        },

        'index': 35,  # آخرین نقطه
        'score': 3.68  # base (4.0) × confidence (0.92)
    }
]
```

---

##### امتیازدهی

**کد:** `signal_generator.py:5300-5311`

```python
for pattern in harmonic_patterns:
    base_score = self.pattern_scores.get(pattern_type, 4.0)
    pattern_score = base_score * confidence * tf_weight

    if direction == 'bullish':
        bullish_score += pattern_score
```

**جدول امتیازات:**

| الگو | Base Score | با Conf=0.9 | با Conf=0.7 |
|------|-----------|-------------|-------------|
| Gartley/Bat | **4.0** | **3.6** | **2.8** |
| Butterfly | **4.5** | **4.05** | **3.15** |
| Crab | **5.0** | **4.5** | **3.5** |

**محدوده کل:** 2.8 تا 5.0 (با TF weight)

---

##### کاربردها

**1. محاسبه SL/TP:**

**کد:** `signal_generator.py:4049-4089`

```python
if harmonic_found:
    best_pattern = sorted(patterns, key=lambda x: x['confidence'])[0]
    D_price = best_pattern['points']['D']['price']
    A_price = best_pattern['points']['A']['price']
    X_price = best_pattern['points']['X']['price']

    # Bullish:
    entry = D_price
    take_profit = X_price  # بازگشت به X
    stop_loss = D_price * 0.99  # کمی زیر D (نقطه ورود)
```

**مثال RR:**
```
Entry: 49786 (D)
TP: 50000 (X) → Reward = 214
SL: 49708 (D × 0.99) → Risk = 78
RR = 2.7:1 ✓✓✓
```

---

**2. تقویت Reversal:**

**کد:** `signal_generator.py:3739-3743`

```python
for pattern in harmonic_patterns:
    if pattern.get('type', '').endswith('butterfly') or pattern.get('type', '').endswith('crab'):
        pattern_quality = pattern.get('confidence', 0.7)
        strength += 0.8 * pattern_quality
        is_reversal = True
```

---

**3. Pattern Multiplier:**

**کد:** `signal_generator.py:5089`

```python
harmonic_count = count_harmonic_patterns()
score.harmonic_pattern_score = 1.0 + (harmonic_count * 0.2)
# 1 الگو → ×1.2, 2 الگو → ×1.4
```

---

##### نکات کلیدی

1. **نادر اما قوی:** الگوهای هارمونیک کمیاب اما بسیار قابل اعتماد هستند
2. **Tolerance ±3%:** نسبت‌ها باید در محدوده دقیق باشند
3. **4 الگو:** Gartley < Bat < Butterfly < Crab (از ضعیف به قوی)
4. **X-A-B-C-D Alternation:** نقاط باید متناوب peak/valley باشند
5. **Confidence >= 0.7:** فیلتر کیفیت
6. **امتیازات بالا:** 2.8 تا 5.0 (قوی‌ترین سیگنال‌ها)
7. **Integration SL/TP:** مستقیماً در محاسبه ریسک استفاده می‌شود

---

### 3.2 شناسایی کانال‌های قیمتی (Price Channels)

**محل:** `signal_generator.py:2666-2768`

```python
analysis_data['price_channels'] = self.detect_price_channels(
    df,
    lookback=100,
    min_touches=3  # حداقل تعداد تماس با خطوط
)
```

کانال‌های قیمتی نواحی‌ای هستند که قیمت بین دو خط موازی (بالا و پایین) حرکت می‌کند. شناسایی کانال برای پیش‌بینی **bounce** (بازگشت از دیوار) یا **breakout** (شکست) استفاده می‌شود.

---

##### الگوریتم شناسایی (5 مرحله)

**مرحله 1: شناسایی Peaks و Valleys**

**کد:** `signal_generator.py:2680-2684`

```python
peaks, valleys = self.find_peaks_and_valleys(
    closes,
    distance=5,
    prominence_factor=0.1
)
```

**شرط:** حداقل `min_touches` (پیش‌فرض: 3) peak و valley نیاز است.

---

**مرحله 2: رسم خطوط با Linear Regression**

**کد:** `signal_generator.py:2687-2695`

```python
# خط بالایی (Upper Line) - اتصال Peaks
if len(peaks) >= 2:
    up_slope, up_intercept = np.polyfit(peak_indices, peak_values, 1)

# خط پایینی (Lower Line) - اتصال Valleys
if len(valleys) >= 2:
    down_slope, down_intercept = np.polyfit(valley_indices, valley_values, 1)
```

**Regression خطی:**
```
y = slope * x + intercept
```

**مثال:**
```python
# Peaks at: (10, 50000), (30, 50500), (50, 51000)
up_slope = 20  # شیب صعودی
up_intercept = 49800

# خط بالایی در زمان x:
upper_line(x) = 20 * x + 49800

# در x=60: upper = 20*60 + 49800 = 51000
```

---

**مرحله 3: محاسبه ویژگی‌های کانال**

**کد:** `signal_generator.py:2697-2704`

```python
# عرض کانال
last_idx = len(closes) - 1
up_line_current = up_slope * last_idx + up_intercept
down_line_current = down_slope * last_idx + down_intercept
channel_width = up_line_current - down_line_current

# جهت کانال
channel_slope = (up_slope + down_slope) / 2
if channel_slope > 0.001:
    channel_direction = 'ascending'      # صعودی ↗
elif channel_slope < -0.001:
    channel_direction = 'descending'     # نزولی ↘
else:
    channel_direction = 'horizontal'     # افقی (Range) →
```

**3 نوع کانال:**

| نوع | شیب میانگین | ویژگی | استراتژی |
|-----|-------------|--------|----------|
| **Ascending** | > 0.001 | صعودی | خرید در کف، نگهداری |
| **Descending** | < -0.001 | نزولی | فروش در سقف، شورت |
| **Horizontal** | -0.001 to 0.001 | Range | خرید کف/فروش سقف |

---

**مرحله 4: ارزیابی کیفیت کانال (Quality)**

**کد:** `signal_generator.py:2709-2730`

```python
# محاسبه انحراف معیار از خطوط
up_dev = np.std(peak_values - up_line)
down_dev = np.std(valley_values - down_line)

# شمارش تماس‌های معتبر
valid_up_touches = sum(1 for i, v in zip(peak_indices, peak_values)
                       if abs(v - (up_slope * i + up_intercept)) < up_dev)

valid_down_touches = sum(1 for i, v in zip(valley_indices, valley_values)
                         if abs(v - (down_slope * i + down_intercept)) < down_dev)

# محاسبه کیفیت
channel_quality = min(1.0, (valid_up_touches + valid_down_touches) / (min_touches * 2))
```

**فرمول کیفیت:**
```
quality = min(1.0, total_valid_touches / (min_touches × 2))
```

**مثال:**
```python
min_touches = 3  # نیاز: حداقل 6 تماس (3 بالا + 3 پایین)

valid_up_touches = 5    # 5 peak نزدیک خط بالا
valid_down_touches = 4  # 4 valley نزدیک خط پایین
total = 9

quality = min(1.0, 9 / 6) = min(1.0, 1.5) = 1.0  # کیفیت عالی ✓
```

**فیلتر کیفیت:**
```python
if quality >= self.channel_quality_threshold:  # پیش‌فرض: 0.6
    # کانال قبول شد
```

---

**مرحله 5: تشخیص موقعیت و Breakout**

**کد:** `signal_generator.py:2722-2727`

```python
# موقعیت در کانال (0.0 = کف, 1.0 = سقف)
position_in_channel = (last_close - down_line_current) / channel_width

# تشخیص Breakout
is_breakout_up = last_close > up_line_current + up_dev
is_breakout_down = last_close < down_line_current - down_dev
```

**موقعیت در کانال:**

| Position | محدوده | معنی | سیگنال |
|----------|--------|------|--------|
| **< 0.2** | کف کانال | احتمال صعود | Bullish Bounce |
| **0.2-0.8** | وسط کانال | خنثی | Wait |
| **> 0.8** | سقف کانال | احتمال نزول | Bearish Bounce |
| **> 1.0** | بالای کانال | شکست صعودی | Bullish Breakout |
| **< 0.0** | زیر کانال | شکست نزولی | Bearish Breakout |

**Breakout Condition:**
```
Breakout Up: price > upper_line + std_deviation
Breakout Down: price < lower_line - std_deviation
```

---

##### خروجی کامل

```python
{
    'status': 'ok',

    'channels': [
        {
            'type': 'ascending_channel',
            'direction': 'ascending',

            # پارامترهای خطوط
            'upper_slope': 20.5,
            'upper_intercept': 49800.0,
            'lower_slope': 18.2,
            'lower_intercept': 48900.0,

            # ویژگی‌ها
            'width': 1100.0,              # عرض کانال
            'quality': 0.88,              # کیفیت (0-1)
            'position_in_channel': 0.15,  # موقعیت فعلی (نزدیک کف)

            # تماس‌ها
            'up_touches': 5,              # تعداد تماس با خط بالا
            'down_touches': 6,            # تعداد تماس با خط پایین

            # Breakout
            'breakout': None              # 'up', 'down', یا None
        }
    ],

    # سیگنال (اگر وجود داشته باشد)
    'signal': {
        'type': 'channel_bounce',     # یا 'channel_breakout'
        'direction': 'bullish',       # یا 'bearish'
        'score': 2.64                 # 3.0 × quality (0.88)
    },

    'details': {}
}
```

---

##### امتیازدهی

**کد:** `signal_generator.py:2749-2760, 5315-5324`

**2 نوع سیگنال:**

**1. Channel Breakout (شکست کانال)**

**کد:** `2749-2754`

```python
if breakout_direction == 'up':
    results['signal'] = {
        'type': 'channel_breakout',
        'direction': 'bullish',
        'score': 4.0 * channel_quality
    }
elif breakout_direction == 'down':
    results['signal'] = {
        'type': 'channel_breakout',
        'direction': 'bearish',
        'score': 4.0 * channel_quality
    }
```

**جدول امتیازات Breakout:**

| Quality | Base Score | امتیاز نهایی | قدرت |
|---------|-----------|--------------|------|
| 1.0 | 4.0 | **4.0** | بسیار قوی |
| 0.8 | 4.0 | **3.2** | قوی |
| 0.6 | 4.0 | **2.4** | متوسط |

---

**2. Channel Bounce (بازگشت از دیوار)**

**کد:** `2755-2760`

```python
elif position_in_channel < 0.2:  # نزدیک کف
    results['signal'] = {
        'type': 'channel_bounce',
        'direction': 'bullish',
        'score': 3.0 * channel_quality
    }
elif position_in_channel > 0.8:  # نزدیک سقف
    results['signal'] = {
        'type': 'channel_bounce',
        'direction': 'bearish',
        'score': 3.0 * channel_quality
    }
```

**جدول امتیازات Bounce:**

| Position | Quality | Base Score | امتیاز نهایی | نوع |
|----------|---------|-----------|--------------|-----|
| < 0.2 | 1.0 | 3.0 | **3.0** | Bullish Bounce |
| < 0.2 | 0.8 | 3.0 | **2.4** | Bullish Bounce |
| > 0.8 | 1.0 | 3.0 | **3.0** | Bearish Bounce |
| > 0.8 | 0.8 | 3.0 | **2.4** | Bearish Bounce |

**محدوده کل:** 2.4 تا 4.0

**⚠️ نکته:** Breakout قوی‌تر از Bounce است (4.0 vs 3.0)

---

##### کاربردها

**1. تأیید Reversal Signals:**

**کد:** `signal_generator.py:3746-3751`

```python
channel_signal = channel_data.get('signal', {})

if channel_signal:
    signal_type = channel_signal['type']
    if signal_type == 'channel_bounce':
        signal_score = channel_signal['score'] / 3.0  # normalize
        strength += signal_score
        is_reversal = True
```

**استفاده:** سیگنال Channel Bounce برای **تقویت Reversal** استفاده می‌شود.

---

**2. امتیازدهی در Multi-Timeframe:**

**کد:** `signal_generator.py:5315-5324`

```python
for tf, result in analysis.items():
    channel_data = result.get('price_channels', {})
    channel_signal = channel_data.get('signal', {})

    if channel_signal:
        signal_type = channel_signal['type']
        signal_direction = channel_signal['direction']
        signal_score = channel_signal['score'] * tf_weight

        if signal_direction == 'bullish':
            bullish_score += signal_score
        elif signal_direction == 'bearish':
            bearish_score += signal_score
```

---

##### مثال واقعی

**سناریو:** BTC/USDT در کانال صعودی

```python
# داده‌های کانال
{
    'type': 'ascending_channel',
    'direction': 'ascending',

    'upper_slope': 25,          # شیب صعودی
    'upper_intercept': 49500,
    'lower_slope': 22,
    'lower_intercept': 48000,

    'width': 1500,              # عرض کانال فعلی
    'quality': 0.92,            # کیفیت بالا (92%)
    'position_in_channel': 0.12, # نزدیک کف (12%)

    'up_touches': 6,            # 6 بار سقف تست شده
    'down_touches': 7,          # 7 بار کف تست شده
    'breakout': None
}

# محاسبه خطوط در کندل فعلی (idx=100):
upper_line = 25 * 100 + 49500 = 52000
lower_line = 22 * 100 + 48000 = 50200
current_price = 50380

# موقعیت:
position = (50380 - 50200) / (52000 - 50200) = 180 / 1800 = 0.10
# یعنی 10% از کف → نزدیک کف! ✓

# سیگنال:
{
    'type': 'channel_bounce',
    'direction': 'bullish',
    'score': 3.0 * 0.92 = 2.76
}

# استراتژی:
Entry: 50380 (کف کانال)
TP: 51800 (80% کانال)
SL: 50100 (زیر کانال)

Risk: 50380 - 50100 = 280
Reward: 51800 - 50380 = 1420
RR = 5.07:1 ✓✓✓
```

**اگر Breakout اتفاق بیفتد:**
```python
# قیمت به 52200 می‌رسد (بالاتر از 52000)
current_price = 52200
is_breakout_up = 52200 > 52000 + 150 (std) = True ✓

# سیگنال:
{
    'type': 'channel_breakout',
    'direction': 'bullish',
    'score': 4.0 * 0.92 = 3.68  # قوی‌تر!
}

# استراتژی Breakout:
Entry: 52200
TP: 53700 (ارتفاع کانال اضافه شود: 52200 + 1500)
SL: 51950 (بازگشت به داخل کانال)

Risk: 250
Reward: 1500
RR = 6.0:1 ✓✓✓
```

---

##### نکات کلیدی

1. **Linear Regression:** خطوط با regression خطی رسم می‌شوند (np.polyfit)

2. **Quality-Based Filtering:** فقط کانال‌های با quality >= 0.6 قبول می‌شوند

3. **3 نوع کانال:** Ascending (صعودی), Descending (نزولی), Horizontal (Range)

4. **Position Matters:** موقعیت در کانال (0-1) برای تشخیص Bounce استفاده می‌شود

5. **Breakout > Bounce:** شکست کانال امتیاز بیشتری دارد (4.0 vs 3.0)

6. **Touch Count:** کانال‌های با تماس بیشتر → کیفیت بالاتر

7. **Std Deviation:** برای تشخیص Breakout از انحراف معیار استفاده می‌شود

8. **⚠️ محدودیت:** فقط **یک کانال اصلی** شناسایی می‌شود (بهترین)

---

### 3.3 شناسایی الگوهای چرخه‌ای (Cyclical Patterns)

**محل:** `signal_generator.py:2769-2871`

```python
analysis_data['cyclical_patterns'] = self.detect_cyclical_patterns(
    df,
    lookback=200  # حداقل 200 کندل برای FFT
)
```

این تحلیل با **FFT (Fast Fourier Transform)** الگوهای تکرارشونده (چرخه‌ای) در قیمت را شناسایی و **20 کندل آینده** را پیش‌بینی می‌کند.

---

##### الگوریتم FFT-Based (4 مرحله)

**مرحله 1: Detrending (حذف روند)**

**کد:** `signal_generator.py:2777-2784`

```python
# گرفتن آخرین 200 کندل
df_window = df.iloc[-lookback:]
closes = df_window['close'].values

# محاسبه خط روند (Linear Regression)
x = np.arange(len(closes))  # [0, 1, 2, ..., 199]
trend_coeffs = np.polyfit(x, closes, 1)  # [slope, intercept]
trend = np.polyval(trend_coeffs, x)  # خط روند

# حذف روند از قیمت
detrended = closes - trend
```

**چرا Detrending؟**
- FFT برای **نوسانات** کار می‌کند نه روند
- اگر روند حذف نشود، FFT روند را به عنوان یک فرکانس پایین می‌بیند
- Detrending به ما اجازه می‌دهد فقط **الگوهای تکرارشونده** را ببینیم

**مثال:**
```python
# قیمت اصلی: [50000, 50100, 50200, 50050, 50150, ...]  # روند صعودی + نوسان
# خط روند: [50000, 50100, 50200, 50300, 50400, ...]  # فقط روند
# Detrended: [0, 0, 0, -250, -250, ...]  # فقط نوسانات ✓
```

---

**مرحله 2: اعمال FFT و استخراج فرکانس‌ها**

**کد:** `signal_generator.py:2786-2792`

```python
from scipy import fft

# اعمال FFT (Real FFT برای داده واقعی)
close_fft = fft.rfft(detrended)  # FFT coefficients (complex numbers)
fft_freqs = fft.rfftfreq(len(detrended))  # فرکانس‌های متناظر

# محاسبه قدرت (Magnitude) هر فرکانس
close_fft_mag = np.abs(close_fft)

# یافتن فرکانس‌های قوی (بالاتر از threshold)
threshold = np.mean(close_fft_mag) + np.std(close_fft_mag)
significant_freq_indices = np.where(close_fft_mag > threshold)[0]
```

**FFT چیست؟**
- تبدیل سیگنال زمانی به فرکانسی
- **ورودی:** سری زمانی قیمت (detrended)
- **خروجی:** قدرت هر فرکانس (چرخه)

**مثال ساده:**
```python
# سیگنال ورودی: نوسان 10 روزه + نوسان 30 روزه
signal = sin(2π × t / 10) + sin(2π × t / 30)

# FFT خروجی:
# فرکانس 0.1 (period=10): magnitude = 1.0 🔴
# فرکانس 0.033 (period=30): magnitude = 1.0 🔴
# سایر فرکانس‌ها: magnitude ≈ 0
```

**Threshold:**
```python
# قدرت‌ها: [0.1, 0.2, 15.5, 0.3, 8.2, 0.1, ...]
# mean = 2.5, std = 5.0
# threshold = 2.5 + 5.0 = 7.5
# فرکانس‌های قوی: [15.5, 8.2]  # فقط اینها Significant هستند
```

---

**مرحله 3: فیلتر و استخراج چرخه‌ها**

**کد:** `signal_generator.py:2794-2813`

```python
# فیلتر: فقط چرخه‌های با دوره منطقی (2 تا lookback/2)
filtered_indices = [i for i in significant_freq_indices
                    if 2 <= 1 / fft_freqs[i] <= lookback / 2]

cycles = []
for idx in filtered_indices:
    if fft_freqs[idx] > 0:
        # تبدیل فرکانس به دوره (Period)
        period = int(1 / fft_freqs[idx])  # Period در کندل

        # دامنه (Amplitude) - قدرت نوسان
        amplitude = close_fft_mag[idx] / len(detrended)  # Normalize

        # فاز (Phase) - موقعیت فعلی در چرخه
        phase = np.angle(close_fft[idx])  # رادیان

        # قدرت نسبی (به درصد قیمت)
        cycle_power = amplitude / np.mean(closes) * 100

        cycles.append({
            'period': period,
            'amplitude': float(amplitude),
            'amplitude_percent': float(cycle_power),
            'phase': float(phase)
        })

# مرتب‌سازی بر اساس قدرت (قوی‌ترین اول)
cycles = sorted(cycles, key=lambda x: x['amplitude'], reverse=True)
top_cycles = cycles[:5]  # فقط 5 چرخه قوی‌ترین
```

**فیلتر دوره:**
- **حداقل:** 2 کندل (خیلی کوتاه‌تر = نویز)
- **حداکثر:** lookback/2 = 100 کندل (خیلی طولانی‌تر = unreliable)

**محاسبات:**

| فرکانس | دوره (Period) | دامنه | قدرت نسبی | توضیح |
|---------|--------------|-------|-----------|-------|
| 0.0417 | 1/0.0417 = **24** | 150.5 | 0.3% | چرخه 24 کندلی قوی |
| 0.0833 | 1/0.0833 = **12** | 95.2 | 0.19% | چرخه 12 کندلی متوسط |

**Phase (فاز):**
```
phase = 0 → ابتدای چرخه (کف)
phase = π/2 → صعود
phase = π → اوج چرخه
phase = 3π/2 → نزول
phase = 2π → بازگشت به کف
```

---

**مرحله 4: پیش‌بینی (Forecast) با ترکیب چرخه‌ها**

**کد:** `signal_generator.py:2815-2843`

```python
if len(top_cycles) >= self.cycle_min_cycles:  # حداقل 2 چرخه
    forecast_length = 20  # پیش‌بینی 20 کندل آینده
    forecast = np.zeros(forecast_length)

    # آخرین نقطه روند
    last_trend = trend[-1]
    trend_slope = trend_coeffs[0]

    # محاسبه پیش‌بینی برای هر کندل
    for i in range(forecast_length):
        # 1. ادامه روند
        point_forecast = last_trend + trend_slope * (i + 1)

        # 2. اضافه کردن تمام چرخه‌ها
        for cycle in top_cycles:
            period = cycle['period']
            amplitude = cycle['amplitude']
            phase = cycle['phase']

            # زمان آینده
            t = len(closes) + i

            # محاسبه مقدار چرخه در زمان t
            cycle_component = amplitude * np.cos(2 * np.pi * t / period + phase)

            # اضافه کردن به پیش‌بینی
            point_forecast += cycle_component

        forecast[i] = point_forecast
```

**فرمول پیش‌بینی:**
```
forecast(t) = trend(t) + Σ [amplitude_i × cos(2π × t / period_i + phase_i)]
```

**مثال محاسبه:**
```python
# کندل 201 (اولین کندل پیش‌بینی):
trend_201 = 50500 + 5 × 1 = 50505  # روند صعودی 5 واحد/کندل

# چرخه 1 (period=24, amp=150, phase=π):
cycle1 = 150 × cos(2π × 201 / 24 + π) = 150 × cos(52.6 + π) ≈ -120

# چرخه 2 (period=12, amp=95, phase=0):
cycle2 = 95 × cos(2π × 201 / 12 + 0) = 95 × cos(105.2) ≈ 60

# پیش‌بینی نهایی:
forecast_201 = 50505 + (-120) + 60 = 50445
```

**تعیین جهت پیش‌بینی:**
```python
forecast_direction = 'bullish' if forecast[-1] > closes[-1] else 'bearish'
forecast_strength = abs(forecast[-1] - closes[-1]) / closes[-1]
```

---

##### خروجی کامل

```python
{
    'status': 'ok',

    # چرخه‌های شناسایی شده (5 قوی‌ترین)
    'cycles': [
        {
            'period': 24,                # چرخه 24 کندلی
            'amplitude': 150.5,          # دامنه: 150.5 واحد
            'amplitude_percent': 0.3,    # 0.3% قیمت فعلی
            'phase': 3.14                # فاز: π (در اوج)
        },
        {
            'period': 12,
            'amplitude': 95.2,
            'amplitude_percent': 0.19,
            'phase': 0.78
        }
    ],

    # پیش‌بینی 20 کندل آینده
    'forecast': {
        'values': [50445, 50462, 50478, ..., 50890],  # 20 مقدار
        'direction': 'bullish',        # جهت کلی پیش‌بینی
        'strength': 0.0078            # 0.78% تغییر
    },

    # سیگنال
    'signal': {
        'type': 'cycle_bullish_forecast',
        'direction': 'bullish',
        'score': 1.95  # 2.5 × clarity (0.78) × cycles_strength (1.0)
    },

    # جزئیات
    'details': {
        'total_cycles_detected': 8,      # تعداد کل چرخه‌های یافت شده
        'significant_cycles': 5,         # 5 قوی‌ترین انتخاب شدند
        'detrend_coeffs': [5.2, 50000]   # [slope, intercept] خط روند
    }
}
```

---

##### امتیازدهی

**کد:** `signal_generator.py:2843-2857, 5328-5338`

```python
# محاسبه امتیاز
prediction_clarity = min(1.0, forecast_strength * 5)  # 0.0 تا 1.0
cycles_strength = min(1.0, sum(c['amplitude_percent'] for c in top_cycles) / 10)

# امتیاز پایه: 2.5
base_score = 2.5
signal_score = base_score * prediction_clarity * cycles_strength

if forecast_direction == 'bullish':
    results['signal'] = {
        'type': 'cycle_bullish_forecast',
        'direction': 'bullish',
        'score': signal_score
    }
```

**فرمول:**
```
score = 2.5 × prediction_clarity × cycles_strength

prediction_clarity = min(1.0, |forecast_change| × 5)
cycles_strength = min(1.0, Σ amplitude_percent / 10)
```

**جدول امتیازات:**

| Forecast Change | Clarity | Total Amp% | Cycles Strength | امتیاز نهایی |
|-----------------|---------|-----------|-----------------|--------------|
| 1% | 1.0 | 5% | 0.5 | 2.5 × 1.0 × 0.5 = **1.25** |
| 0.5% | 1.0 | 10% | 1.0 | 2.5 × 1.0 × 1.0 = **2.5** |
| 0.2% | 1.0 | 3% | 0.3 | 2.5 × 1.0 × 0.3 = **0.75** |

**محدوده کل:** 0.75 تا 2.5

---

##### مثال واقعی

**سناریو:** BTC/USDT، 200 کندل اخیر

```python
# بعد از FFT:
{
    'cycles': [
        {
            'period': 28,              # چرخه هفتگی (تقریباً)
            'amplitude': 180.0,
            'amplitude_percent': 0.36,  # 0.36% قیمت
            'phase': 4.71              # 3π/2 → در حال نزول به کف
        },
        {
            'period': 14,              # چرخه نیم‌هفتگی
            'amplitude': 120.0,
            'amplitude_percent': 0.24,
            'phase': 1.57              # π/2 → در حال صعود
        },
        {
            'period': 7,               # چرخه روزانه
            'amplitude': 85.0,
            'amplitude_percent': 0.17,
            'phase': 0.0               # 0 → در کف
        }
    ],

    'forecast': {
        'values': [50050, 50095, 50140, ..., 50680],
        'direction': 'bullish',        # پیش‌بینی صعودی
        'strength': 0.0126            # 1.26% افزایش
    }
}

# محاسبه امتیاز:
prediction_clarity = min(1.0, 0.0126 × 5) = min(1.0, 0.063) = 0.063... wait, فرمول اشتباه است!

# اصلاح:
# strength = 0.0126 = 1.26%
prediction_clarity = min(1.0, 1.26 × 5) = min(1.0, 6.3) = 1.0 ✓

cycles_strength = (0.36 + 0.24 + 0.17) / 10 = 0.77 / 10 = 0.077
# این خیلی کم است! باید به درصد باشد:
cycles_strength = (0.36 + 0.24 + 0.17 + ... تا 5 چرخه) / 10

# فرض کنیم مجموع = 1.5%
cycles_strength = 1.5 / 10 = 0.15

score = 2.5 × 1.0 × 0.15 = 0.375  # خیلی کم!

# به نظر می‌رسد این سیستم امتیازات پایینی می‌دهد
```

---

##### نکات کلیدی

1. **FFT-Based:** تنها تحلیل مبتنی بر فرکانس در سیستم

2. **Detrending ضروری:** بدون آن FFT نتیجه غلط می‌دهد

3. **حداقل 200 کندل:** FFT به داده کافی نیاز دارد

4. **Top 5 Cycles:** فقط 5 چرخه قوی‌ترین استفاده می‌شوند

5. **20-Candle Forecast:** پیش‌بینی کوتاه‌مدت (نه بلندمدت)

6. **امتیازات پایین:** معمولاً 0.75 تا 2.5 (کمتر از سایر سیگنال‌ها)

7. **Phase مهم:** فاز چرخه نشان می‌دهد الان در کجای چرخه هستیم

8. **⚠️ محدودیت:** FFT فقط برای بازارهای **چرخه‌ای** (Range) خوب کار می‌کند، نه ترندهای قوی

---

### 3.4 تحلیل شرایط نوسان (Volatility Analysis)

**محل:** `signal_generator.py:4459-4530`

```python
analysis_data['volatility'] = self.analyze_volatility_conditions(df)
```

این بخش برای **محافظت از سرمایه** در شرایط نوسان غیرعادی طراحی شده است.

---

#### الگوریتم تحلیل نوسان

تشخیص نوسان در **۵ مرحله** انجام می‌شود:

##### مرحله ۱: محاسبه ATR (Average True Range)

ATR نوسان واقعی بازار را اندازه‌گیری می‌کند با در نظر گرفتن گپ‌های قیمتی:

```python
# کد: signal_generator.py:4472
high_p = df['high'].values.astype(np.float64)
low_p = df['low'].values.astype(np.float64)
close_p = df['close'].values.astype(np.float64)

atr = talib.ATR(high_p, low_p, close_p, timeperiod=14)
```

**فرمول ATR:**
```
True Range (TR) = max(
    high - low,
    abs(high - close_prev),
    abs(low - close_prev)
)

ATR(14) = میانگین متحرک 14 دوره‌ای از TR
```

**مثال محاسبه:**
```
کندل فعلی:
  High = 50,000
  Low = 49,000
  Close قبلی = 48,500

TR = max(
    50,000 - 49,000 = 1,000,
    |50,000 - 48,500| = 1,500,  ← بیشترین
    |49,000 - 48,500| = 500
) = 1,500

ATR = میانگین 14 TR اخیر
```

---

##### مرحله ۲: نرمال‌سازی ATR (Percentage ATR)

برای مقایسه بین قیمت‌های مختلف، ATR به درصد تبدیل می‌شود:

```python
# کد: signal_generator.py:4479-4481
valid_close_p = close_p[-len(valid_atr):]
atr_pct = (valid_atr / valid_close_p) * 100
```

**فرمول:**
```
ATR% = (ATR / قیمت فعلی) × 100
```

**مثال:**
```
ATR = 1,500
قیمت = 50,000

ATR% = (1,500 / 50,000) × 100 = 3.0%

→ نوسان معمولی قیمت 3% است
```

---

##### مرحله ۳: محاسبه میانگین نوسان (ATR% Moving Average)

برای تشخیص نوسان **غیرعادی**، ATR فعلی با میانگین تاریخی مقایسه می‌شود:

```python
# کد: signal_generator.py:4484-4491
if use_bottleneck:
    atr_pct_ma = bn.move_mean(atr_pct, window=self.vol_atr_ma_period, min_count=1)
else:
    for i in range(len(atr_pct)):
        start_idx = max(0, i - self.vol_atr_ma_period + 1)
        atr_pct_ma[i] = np.mean(atr_pct[start_idx:i + 1])
```

**پارامترها:**
- **window = 30:** میانگین 30 دوره اخیر (پیش‌فرض: `vol_atr_ma_period`)

**مثال:**
```
ATR% در 20 کندل اخیر:
[2.1, 2.3, 2.0, 2.4, 2.2, ..., 3.0]

ATR%_MA = میانگین این مقادیر = 2.3%
ATR% فعلی = 3.0%
```

---

##### مرحله ۴: محاسبه نسبت نوسان (Volatility Ratio)

این نسبت نشان می‌دهد نوسان فعلی چقدر از حالت عادی فاصله دارد:

```python
# کد: signal_generator.py:4494-4498
current_atr_pct = atr_pct[-1]
current_atr_pct_ma = atr_pct_ma[-1]

volatility_ratio = current_atr_pct / current_atr_pct_ma if current_atr_pct_ma > 0 else 1.0
```

**فرمول:**
```
Volatility Ratio = ATR% فعلی / میانگین ATR%
```

**تفسیر:**
```
ratio = 1.0  →  نوسان عادی (ATR فعلی = میانگین)
ratio = 1.5  →  نوسان 50% بیشتر از حالت عادی
ratio = 2.0  →  نوسان 2 برابر حالت عادی (خطرناک!)
ratio = 0.5  →  نوسان 50% کمتر از حالت عادی
```

**مثال:**
```
ATR% فعلی = 3.0%
میانگین ATR% = 2.3%

Volatility Ratio = 3.0 / 2.3 = 1.30

→ نوسان 30% بیشتر از حالت عادی است
```

---

##### مرحله ۵: طبقه‌بندی وضعیت نوسان

بر اساس `volatility_ratio`، وضعیت نوسان تعیین می‌شود:

```python
# کد: signal_generator.py:4500-4512
vol_condition = 'normal'
vol_score = 1.0

if volatility_ratio > self.vol_extreme_thresh:      # پیش‌فرض: 1.8
    vol_condition = 'extreme'
    vol_score = 0.5                                 # از پیش‌فرض scores.extreme
elif volatility_ratio > self.vol_high_thresh:       # پیش‌فرض: 1.3
    vol_condition = 'high'
    vol_score = 0.8                                 # از پیش‌فرض scores.high
elif volatility_ratio < self.vol_low_thresh:        # پیش‌فرض: 0.7
    vol_condition = 'low'
    vol_score = 0.9                                 # از پیش‌فرض scores.low
```

**جدول طبقه‌بندی (با مقادیر پیش‌فرض):**

| شرایط | Ratio | وضعیت | ضریب امتیاز | تفسیر |
|-------|-------|-------|-------------|-------|
| ratio ≥ 1.8 | 1.8+ | **extreme** | **×0.5** | نوسان خطرناک - سیگنال رد می‌شود ❌ |
| 1.3 ≤ ratio < 1.8 | 1.3-1.8 | **high** | **×0.8** | نوسان بالا - کاهش 20% امتیاز |
| 0.7 ≤ ratio < 1.3 | 0.7-1.3 | **normal** | **×1.0** | نوسان عادی - بدون تغییر ✓ |
| ratio < 0.7 | 0.0-0.7 | **low** | **×0.9** | نوسان پایین - کاهش 10% امتیاز |

**نکته:** این آستانه‌ها قابل تنظیم در فایل کانفیگ هستند:
```python
# کد: signal_generator.py:1514-1517
self.vol_high_thresh = self.vol_config.get('high_volatility_threshold', 1.3)
self.vol_low_thresh = self.vol_config.get('low_volatility_threshold', 0.7)
self.vol_extreme_thresh = self.vol_config.get('extreme_volatility_threshold', 1.8)
self.vol_scores = self.vol_config.get('scores', {})
```

---

#### رد سیگنال در نوسان خطرناک

در صورت فعال بودن `vol_reject_extreme`، سیگنال‌ها در نوسان خطرناک رد می‌شوند:

```python
# کد: signal_generator.py:4514
reject_due_to_extreme = vol_condition == 'extreme' and self.vol_reject_extreme
```

**منطق:**
```
اگر نوسان = extreme و vol_reject_extreme = True
→ reject = True
→ سیگنال نادیده گرفته می‌شود
```

---

#### خروجی تحلیل نوسان

```python
{
    'status': 'ok',                    # 'ok', 'disabled_or_insufficient_data', 'error'
    'score': 0.8,                      # ضریب امتیاز (0.5, 0.8, 0.9, یا 1.0)
    'condition': 'high',               # 'low', 'normal', 'high', 'extreme'
    'reject': False,                   # True اگر باید سیگنال رد شود
    'volatility_ratio': 1.45,          # نسبت نوسان فعلی به میانگین
    'details': {
        'current_atr_pct': 3.2,        # ATR% فعلی
        'average_atr_pct': 2.2,        # میانگین ATR%
        'raw_atr': 1600.5              # مقدار خام ATR
    }
}
```

---

#### مثال‌های کامل محاسبه

##### مثال ۱: نوسان بالا (High Volatility)

**داده‌های ورودی:**
```
قیمت فعلی = 50,000 USDT
ATR(14) = 1,600
ATR% فعلی = (1,600 / 50,000) × 100 = 3.2%

میانگین 20 روزه ATR%:
[2.0, 2.1, 2.3, 2.2, 2.4, ..., 2.0]  →  میانگین = 2.2%
```

**محاسبات:**
```
1️⃣ Volatility Ratio = 3.2 / 2.2 = 1.45

2️⃣ طبقه‌بندی:
   1.45 > 1.3  ✓  (vol_high_thresh)
   1.45 < 1.8  ✓  (vol_extreme_thresh)
   → condition = 'high'

3️⃣ امتیازدهی:
   vol_score = 0.8

4️⃣ تصمیم:
   reject = False  (چون extreme نیست)
```

**تأثیر بر سیگنال:**
```
امتیاز اولیه سیگنال = 75
امتیاز نهایی = 75 × 0.8 = 60

→ به دلیل نوسان بالا، امتیاز 20% کاهش یافت
```

---

##### مثال ۲: نوسان خطرناک (Extreme Volatility) - رد سیگنال

**داده‌های ورودی:**
```
قیمت فعلی = 45,000 USDT  (افت شدید!)
ATR(14) = 3,600
ATR% فعلی = (3,600 / 45,000) × 100 = 8.0%

میانگین 20 روزه ATR% = 2.5%
```

**محاسبات:**
```
1️⃣ Volatility Ratio = 8.0 / 2.5 = 3.2

2️⃣ طبقه‌بندی:
   3.2 > 1.8  ✓  (vol_extreme_thresh)
   → condition = 'extreme'

3️⃣ امتیازدهی:
   vol_score = 0.5

4️⃣ تصمیم (با vol_reject_extreme = True):
   reject = True  ❌
```

**نتیجه:**
```
⚠️ سیگنال رد می‌شود!

دلیل: نوسان 3.2 برابر حالت عادی است
       → خطر از دست دادن سرمایه بسیار بالاست
       → بهتر است معامله انجام نشود
```

---

##### مثال ۳: نوسان پایین (Low Volatility)

**داده‌های ورودی:**
```
قیمت فعلی = 50,000 USDT
ATR(14) = 800
ATR% فعلی = (800 / 50,000) × 100 = 1.6%

میانگین 20 روزه ATR% = 2.5%
```

**محاسبات:**
```
1️⃣ Volatility Ratio = 1.6 / 2.5 = 0.64

2️⃣ طبقه‌بندی:
   0.64 < 0.7  ✓  (vol_low_thresh)
   → condition = 'low'

3️⃣ امتیازدهی:
   vol_score = 0.9

4️⃣ تصمیم:
   reject = False
```

**تأثیر بر سیگنال:**
```
امتیاز اولیه = 65
امتیاز نهایی = 65 × 0.9 = 58.5 ≈ 59

→ نوسان پایین باعث کاهش 10% امتیاز شد
   (چون فرصت سود کمتری وجود دارد)
```

---

#### تأثیر بر امتیاز نهایی سیگنال

نوسان به عنوان **ضریب** در محاسبه امتیاز نهایی استفاده می‌شود:

```python
final_score = base_score × volatility_score
```

**جدول تأثیر:**

| وضعیت نوسان | ضریب | تأثیر بر امتیاز 75 | تفسیر |
|-------------|------|-------------------|-------|
| **Extreme** (ratio ≥ 1.8) | ×0.5 | 75 → **37.5** (یا رد) | خطرناک - رد سیگنال |
| **High** (ratio ≥ 1.3) | ×0.8 | 75 → **60** | کاهش 20% - احتیاط |
| **Normal** (0.7-1.3) | ×1.0 | 75 → **75** | بدون تغییر - ایده‌آل |
| **Low** (ratio < 0.7) | ×0.9 | 75 → **67.5** | کاهش 10% - فرصت کم |

---

#### شرایط عدم فعال‌سازی

تحلیل نوسان در موارد زیر غیرفعال می‌شود:

```python
# کد: signal_generator.py:4462-4464
if not self.vol_enabled or df is None or len(df) < max(self.vol_atr_period, self.vol_atr_ma_period) + 10:
    results['status'] = 'disabled_or_insufficient_data'
    return results
```

**شرایط:**
1. `vol_enabled = False` → نوسان در کانفیگ غیرفعال شده
2. `df is None` → داده موجود نیست
3. `len(df) < max(14, 30) + 10 = 40` → حداقل 40 کندل لازم است

**در این صورت:**
```python
{
    'status': 'disabled_or_insufficient_data',
    'score': 1.0,  # بدون تأثیر
    'condition': 'normal'
}
```

---

#### نکات کلیدی

##### ✅ چرا نوسان پایین امتیاز کاهش می‌دهد؟

```
نوسان پایین = حرکات قیمتی کوچک
              = فرصت سود کمتر
              = نیاز به صبر بیشتر برای رسیدن به Target

→ سیگنال ضعیف‌تر است
```

##### ✅ چرا نوسان بالا خطرناک است؟

```
نوسان بالا = حرکات قیمتی ناگهانی
            = احتمال StopLoss شدن زیاد
            = ریسک بالا

→ بهتر است منتظر آرام شدن بازار بود
```

##### ✅ استراتژی بهینه

```
1. در نوسان Normal: معامله بدون نگرانی ✓
2. در نوسان High: استفاده از حجم کمتر
3. در نوسان Extreme: عدم معامله (منتظر ماندن)
4. در نوسان Low: افزایش حجم (ریسک کمتر)
```

---

#### خلاصه الگوریتم

```
┌─────────────────────────────────────┐
│ 1. محاسبه ATR(14)                  │
└──────────────┬──────────────────────┘
               ↓
┌─────────────────────────────────────┐
│ 2. نرمال‌سازی: ATR% = ATR/Price×100│
└──────────────┬──────────────────────┘
               ↓
┌─────────────────────────────────────┐
│ 3. میانگین MA(20) از ATR%          │
└──────────────┬──────────────────────┘
               ↓
┌─────────────────────────────────────┐
│ 4. Ratio = ATR% / MA(ATR%)          │
└──────────────┬──────────────────────┘
               ↓
┌─────────────────────────────────────┐
│ 5. طبقه‌بندی:                       │
│    - Ratio ≥ 1.8  → Extreme (×0.5) │
│    - Ratio ≥ 1.3  → High (×0.8)    │
│    - Ratio < 0.7  → Low (×0.9)     │
│    - دیگر موارد   → Normal (×1.0)  │
└─────────────────────────────────────┘
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

---

### 4.2 نحوه تشخیص رژیم بازار

**محل:** `market_regime_detector.py:416-590` (کلاس MarketRegimeDetector)

```python
regime_result = self.regime_detector.detect_regime(df)
```

---

#### مرحله 1: محاسبه اندیکاتورها

**محل در کد:** `market_regime_detector.py:226-390`

```python
df_with_indicators, success = self._calculate_indicators(df)
```

**اندیکاتورهای محاسبه شده:**

##### 1. ADX و DI (Average Directional Index)

```python
# market_regime_detector.py:253-259
adx = talib.ADX(high, low, close, timeperiod=14)
plus_di = talib.PLUS_DI(high, low, close, timeperiod=14)
minus_di = talib.MINUS_DI(high, low, close, timeperiod=14)
```

**ADX چه می‌گوید؟**
- ADX > 25: روند قوی وجود دارد (Strong Trend)
- ADX 20-25: روند ضعیف (Weak Trend)
- ADX < 20: بازار رنج است (No Trend / Range)

**جهت روند:**
- +DI > -DI → روند صعودی (Bullish)
- -DI > +DI → روند نزولی (Bearish)
- +DI ≈ -DI → بدون جهت مشخص (Neutral)

---

##### 2. ATR و ATR% (Average True Range)

```python
# market_regime_detector.py:261-268
atr = talib.ATR(high, low, close, timeperiod=20)
atr_percent = (atr / close) * 100
```

**سطوح نوسان:**
- ATR% > 1.5: نوسان بالا (High Volatility) ⚠️
- ATR% 0.5-1.5: نوسان عادی (Normal Volatility) ✅
- ATR% < 0.5: نوسان پایین (Low Volatility) ✅

**چرا ATR% مهم است:**
- در نوسان بالا: Stop Loss باید گسترده‌تر باشد
- در نوسان بالا: ریسک هر معامله باید کمتر باشد
- در نوسان پایین: می‌توان Stop Loss محکم‌تر گذاشت

---

##### 3. Bollinger Bands Width

```python
# market_regime_detector.py:270-279
upper, middle, lower = talib.BBANDS(
    close,
    timeperiod=20,
    nbdevup=2,
    nbdevdn=2,
    matype=0
)
bb_width = ((upper - lower) / middle) * 100
```

**کاربرد Bollinger Width:**
- BB Width بالا → نوسان در حال افزایش (احتمال شکست)
- BB Width پایین → نوسان در حال کاهش (احتمال رنج)
- BB Squeeze → فشردگی → نزدیک به حرکت بزرگ

---

##### 4. RSI (Relative Strength Index)

```python
# market_regime_detector.py:281-282
rsi = talib.RSI(close, timeperiod=14)
```

**کاربرد RSI در تشخیص رژیم:**
- RSI > 70: احتمال بازار اشباع خرید (نزدیک به peak)
- RSI < 30: احتمال بازار اشباع فروش (نزدیک به bottom)
- RSI ≈ 50: بازار در تعادل

---

##### 5. Volume Analysis (اختیاری)

```python
# market_regime_detector.py:284-306
if self.use_volume_analysis and 'volume' in df.columns:
    # محاسبه تغییرات حجم
    volume_ma = df['volume'].rolling(window=20).mean()
    volume_ratio = df['volume'] / volume_ma
    volume_change = df['volume'].pct_change()

    # همبستگی قیمت و حجم
    correlation = df['close'].pct_change().iloc[-20:].corr(
        df['volume'].pct_change().iloc[-20:]
    )
```

**کاربرد Volume:**
- Volume بالا + حرکت قیمت → تأیید روند
- Volume پایین + حرکت قیمت → روند ضعیف
- Volume Divergence → هشدار تغییر روند

---

#### مرحله 2: تشخیص حالت‌های خاص

##### 2.1 تشخیص Breakout (شکست)

**محل در کد:** `market_regime_detector.py:332-390`

```python
is_breakout, breakout_direction = self._detect_breakout(df)
```

**شرایط Breakout:**

```python
# محاسبه بالاترین و پایین‌ترین قیمت در N کندل اخیر
lookback = self.breakout_lookback  # 10 کندل
recent_high = df['high'].iloc[-(lookback+1):-1].max()
recent_low = df['low'].iloc[-(lookback+1):-1].max()

current_close = df['close'].iloc[-1]

# شرط Bullish Breakout
if current_close > recent_high:
    # بررسی حجم
    volume_confirmation = current_volume > avg_volume * 1.5

    # بررسی قدرت حرکت (تغییرات قیمت)
    price_move_percent = (current_close - recent_high) / recent_high
    strong_move = price_move_percent > (self.breakout_threshold / 100)  # 2%

    if volume_confirmation and strong_move:
        is_breakout = True
        breakout_direction = "bullish"

# مشابه برای Bearish Breakout
```

**شرایط کامل Breakout:**
1. **شکست قیمتی:** قیمت از highest/lowest اخیر عبور کند
2. **تأیید حجم:** حجم معامله > 1.5 × میانگین حجم
3. **قدرت حرکت:** حرکت قیمت > 2% (قابل تنظیم)

---

##### 2.2 تشخیص Choppy Market (بازار آشفته)

**محل در کد:** `market_regime_detector.py:392-414`

```python
is_choppy = self._is_choppy_market(df)
```

**شرایط Choppy:**

```python
# محاسبه نوسانات جهت قیمت
close_changes = df['close'].pct_change().iloc[-20:]  # 20 کندل اخیر
direction_changes = (close_changes > 0).astype(int).diff().abs()
direction_change_rate = direction_changes.sum() / len(direction_changes)

# اگر جهت خیلی زیاد تغییر کند → Choppy
if direction_change_rate > self.choppy_threshold:  # 0.3 = 30%
    # بررسی اضافی: ADX پایین و BB Width بالا
    if current_adx < 20 and current_bb_width > median_bb_width * 1.5:
        is_choppy = True
```

**علائم Choppy Market:**
1. تغییرات مکرر جهت قیمت (بیش از 30%)
2. ADX پایین (کمتر از 20)
3. Bollinger Width بالا (نوسان بالا)
4. عدم روند مشخص

---

#### مرحله 3: تعیین رژیم نهایی

**محل در کد:** `market_regime_detector.py:484-508`

رژیم بازار از **ترکیب** اندیکاتورها و حالت‌های خاص تعیین می‌شود:

```python
# گام 1: تعیین قدرت روند (بر اساس ADX)
if current_adx > 25:
    trend_strength = 'strong'
elif current_adx > 20:
    trend_strength = 'weak'
else:
    trend_strength = 'no_trend'

# گام 2: تعیین جهت روند (بر اساس DI)
if current_plus_di > current_minus_di:
    trend_direction = 'bullish'
elif current_minus_di > current_plus_di:
    trend_direction = 'bearish'
else:
    trend_direction = 'neutral'

# گام 3: تعیین نوسان (بر اساس ATR%)
if current_atr_percent > 1.5:
    volatility_level = 'high'
elif current_atr_percent < 0.5:
    volatility_level = 'low'
else:
    volatility_level = 'normal'

# گام 4: ترکیب برای تعیین رژیم نهایی
if is_breakout:
    regime = 'breakout'
elif is_choppy:
    regime = 'choppy'
elif trend_strength == 'strong':
    if volatility_level == 'high':
        regime = 'strong_trend_high_volatility'
    else:
        regime = 'strong_trend'  # یا 'strong_trend_normal'
elif trend_strength == 'weak':
    if volatility_level == 'high':
        regime = 'weak_trend_high_volatility'
    else:
        regime = 'weak_trend'
else:  # no_trend
    if volatility_level == 'high':
        regime = 'range_high_volatility'
    elif volatility_level == 'low':
        regime = 'tight_range'
    else:
        regime = 'range'
```

---

#### جدول کامل رژیم‌های ممکن

| رژیم | شرایط ADX | شرایط ATR% | توضیح | اولویت سیگنال |
|------|----------|-----------|-------|--------------|
| **breakout** | Any | Any | شکست از محدوده + حجم بالا | 🚀 Trend-Following |
| **strong_trend** | > 25 | 0.5-1.5 | روند قوی، نوسان عادی | ✅ Trend-Following |
| **strong_trend_high_volatility** | > 25 | > 1.5 | روند قوی، نوسان بالا | ⚠️ Trend-Following (محتاطانه) |
| **weak_trend** | 20-25 | 0.5-1.5 | روند ضعیف، نوسان عادی | 🔄 Trend + Reversal |
| **weak_trend_high_volatility** | 20-25 | > 1.5 | روند ضعیف، نوسان بالا | ⚠️ Trend (خیلی محتاطانه) |
| **range** | < 20 | 0.5-1.5 | بدون روند، نوسان عادی | 🔄 Reversal (Mean Reversion) |
| **range_high_volatility** | < 20 | > 1.5 | بدون روند، نوسان بالا | ❌ خطرناک! |
| **tight_range** | < 20 | < 0.5 | بدون روند، نوسان پایین | 🔄 Reversal (کم ریسک) |
| **choppy** | < 20 | High BB Width | بازار آشفته، غیرقابل پیش‌بینی | ❌ خیلی خطرناک! |

---

#### محاسبه Confidence (اطمینان از تشخیص)

**محل در کد:** `market_regime_detector.py:509-544`

```python
# گام 1: محاسبه ثبات ADX
recent_adx = df['adx'].iloc[-5:]  # 5 کندل اخیر
adx_stability = 1.0 - min(1.0, recent_adx.std() / max(0.1, recent_adx.mean()))
# هرچه std کمتر → ثبات بیشتر → confidence بالاتر

# گام 2: همبستگی حجم و قیمت (اگر volume موجود باشد)
if self.use_volume_analysis:
    correlation = df['close'].pct_change().iloc[-20:].corr(
        df['volume'].pct_change().iloc[-20:]
    )
    volume_price_correlation = abs(correlation)

# گام 3: ترکیب فاکتورها
confidence_factors = [
    adx_stability * 0.5,  # ثبات ADX (وزن: 50%)
    0.3,                  # پایه اطمینان (30%)
]

# بونوس: اگر breakout با جهت روند همراستا باشد
if is_breakout and breakout_direction == trend_direction:
    confidence_factors.append(0.2)  # +20%

# بونوس: اگر حجم بالا و همبستگی قوی باشد
if volume_ratio > 1.5:
    confidence_factors.append(0.1 * volume_price_correlation)

# confidence نهایی
confidence = min(1.0, sum(confidence_factors))
```

**فرمول Confidence:**
```
Confidence = min(1.0, ADX_Stability×0.5 + 0.3 + Breakout_Bonus + Volume_Bonus)
```

**مثال محاسبه:**

```python
# سناریو 1: روند قوی با ADX پایدار
adx_stability = 0.9  # ADX پایدار
confidence = 0.9 × 0.5 + 0.3 = 0.75  # اطمینان خوب ✅

# سناریو 2: breakout با حجم بالا
adx_stability = 0.8
is_breakout = True  # +0.2
volume_ratio = 2.0  # +0.1 × correlation
confidence = 0.8 × 0.5 + 0.3 + 0.2 + 0.1 × 0.8 = 0.98  # اطمینان عالی! 🚀

# سناریو 3: رنج با ADX ناپایدار
adx_stability = 0.4  # ADX متغیر
confidence = 0.4 × 0.5 + 0.3 = 0.5  # اطمینان متوسط ⚠️
```

---

### 4.3 خروجی کامل تشخیص رژیم

```python
# نمونه خروجی واقعی
{
    'regime': 'strong_trend',                 # نوع رژیم
    'trend_strength': 'strong',               # قدرت روند
    'trend_direction': 'bullish',             # جهت روند
    'volatility': 'normal',                   # سطح نوسان
    'confidence': 0.85,                       # اطمینان از تشخیص (0-1)
    'details': {
        'adx': 28.5,                          # مقدار ADX
        'plus_di': 32.0,                      # +DI
        'minus_di': 18.0,                     # -DI
        'atr_percent': 1.2,                   # نوسان به درصد
        'adx_stability': 0.88,                # ثبات ADX
        'bollinger_width': 2.5,               # عرض باندهای بولینگر
        'rsi': 62.3,                          # RSI فعلی
        'volume_change': 0.15,                # تغییرات حجم
        'volume_ratio': 1.4,                  # نسبت حجم به میانگین
        'volume_price_correlation': 0.72      # همبستگی حجم و قیمت
    }
}
```

---

### 4.4 تطبیق پارامترها با رژیم بازار

**محل:** `market_regime_detector.py:648-850`

```python
adapted_config = self.regime_detector.get_strategy_parameters(regime_info, base_config)
```

وقتی رژیم بازار مشخص شد، سیستم **خودکار** پارامترهای زیر را تنظیم می‌کند:

---

#### پارامترهای قابل تنظیم

**1. Stop Loss Distance (فاصله استاپ لاس)**

```python
# market_regime_detector.py:690-792
'stop_loss': multiplier  # ضریب تعدیل
```

| رژیم | ضریب SL | مقدار نهایی | دلیل |
|------|---------|------------|------|
| `strong_trend` | 0.9 | 1.5% × 0.9 = **1.35%** | روند قوی → SL نزدیک‌تر |
| `strong_trend_high_volatility` | 1.2 | 1.5% × 1.2 = **1.8%** | نوسان بالا → SL گسترده‌تر |
| `range` | 1.1 | 1.5% × 1.1 = **1.65%** | رنج → SL کمی دورتر |
| `range_high_volatility` | 1.3 | 1.5% × 1.3 = **1.95%** | رنج + نوسان → SL خیلی دورتر |
| `choppy` | 1.25 | 1.5% × 1.25 = **1.88%** | آشفته → SL دورتر |
| `breakout` | 0.85 | 1.5% × 0.85 = **1.28%** | شکست → SL محکم |

---

**2. Risk-Reward Ratio (نسبت ریسک به پاداش)**

```python
'risk_reward': multiplier
```

| رژیم | ضریب RR | مقدار نهایی | دلیل |
|------|---------|------------|------|
| `strong_trend` | 1.2 | 2.5 × 1.2 = **3.0** | روند قوی → هدف دورتر |
| `strong_trend_high_volatility` | 1.1 | 2.5 × 1.1 = **2.75** | نوسان بالا → هدف کمی نزدیک‌تر |
| `range` | 0.9 | 2.5 × 0.9 = **2.25** | رنج → هدف نزدیک‌تر |
| `range_high_volatility` | 0.8 | 2.5 × 0.8 = **2.0** | رنج + نوسان → هدف خیلی نزدیک |
| `choppy` | 0.85 | 2.5 × 0.85 = **2.13** | آشفته → هدف نزدیک |
| `breakout` | 1.3 | 2.5 × 1.3 = **3.25** | شکست → هدف دورتر |

---

**3. Max Risk per Trade (حداکثر ریسک هر معامله)**

```python
'max_risk': multiplier
```

| رژیم | ضریب Risk | مقدار نهایی | دلیل |
|------|-----------|------------|------|
| `strong_trend` | 1.0 | 1.0% × 1.0 = **1.0%** | شرایط خوب → ریسک عادی |
| `strong_trend_high_volatility` | 0.85 | 1.0% × 0.85 = **0.85%** | نوسان بالا → ریسک کمتر |
| `range` | 0.9 | 1.0% × 0.9 = **0.9%** | رنج → ریسک کمتر |
| `range_high_volatility` | 0.75 | 1.0% × 0.75 = **0.75%** | رنج + نوسان → ریسک خیلی کمتر |
| `choppy` | 0.7 | 1.0% × 0.7 = **0.7%** | آشفته → ریسک خیلی کم |
| `breakout` | 1.1 | 1.0% × 1.1 = **1.1%** | فرصت خوب → ریسک کمی بیشتر |

---

**4. Position Size (اندازه معامله)**

```python
'position_size': multiplier
```

| رژیم | ضریب Size | اثر | دلیل |
|------|-----------|-----|------|
| `strong_trend` | 1.1 | +10% | شرایط عالی → معامله بزرگتر |
| `strong_trend_high_volatility` | 0.9 | -10% | نوسان بالا → معامله کوچکتر |
| `range` | 0.95 | -5% | رنج → معامله کمی کوچکتر |
| `range_high_volatility` | 0.8 | -20% | رنج + نوسان → معامله خیلی کوچکتر |
| `choppy` | 0.7 | -30% | آشفته → معامله خیلی کوچک |
| `breakout` | 1.2 | +20% | فرصت عالی → معامله بزرگتر |

---

**5. Signal Weights (وزن سیگنال‌ها)**

```python
'trend_following_weight': multiplier  # وزن سیگنال‌های Trend-Following
'reversal_weight': multiplier         # وزن سیگنال‌های Reversal
```

| رژیم | Trend Weight | Reversal Weight | منطق |
|------|--------------|-----------------|------|
| `strong_trend` | 1.5 | 0.6 | تقویت Trend / تضعیف Reversal |
| `weak_trend` | 1.2 | 0.8 | کمی تقویت Trend |
| `range` | 0.8 | 1.3 | تضعیف Trend / تقویت Reversal |
| `tight_range` | 0.7 | 1.4 | تضعیف قوی Trend / تقویت قوی Reversal |
| `choppy` | 0.5 | 0.8 | تضعیف همه (بازار غیرقابل پیش‌بینی) |
| `breakout` | 1.6 | 0.5 | تقویت قوی Trend / تضعیف قوی Reversal |

---

**6. Trailing Stop Settings**

```python
'use_trailing_stop': True/False
'trailing_stop_activation_percent': multiplier
'trailing_stop_distance_percent': multiplier
```

| رژیم | استفاده | فعال‌سازی | فاصله | دلیل |
|------|---------|----------|-------|------|
| `strong_trend` | ✅ | 0.9 | 0.9 | سریع و نزدیک (حفظ سود) |
| `weak_trend` | ✅ | 1.0 | 1.0 | عادی |
| `range` | ✅ | 1.0 | 0.9 | عادی اما نزدیک |
| `tight_range` | ❌ | - | - | نیازی نیست |
| `choppy` | ❌ | - | - | غیرقابل پیش‌بینی |
| `breakout` | ✅ | 0.8 | 0.7 | خیلی سریع و نزدیک |

---

### 4.5 تأثیر رژیم بر امتیازدهی

رژیم بازار **مستقیماً** بر امتیاز تأثیر نمی‌گذارد، بلکه:

#### 1. تطبیق پارامترها

```python
# مثال: رژیم strong_trend
adapted_params = {
    'stop_loss': 1.35%,        # کمتر از پیش‌فرض (1.5%)
    'risk_reward': 3.0,        # بیشتر از پیش‌فرض (2.5%)
    'max_risk': 1.0%,          # عادی
    'position_size': 1.1×      # 10% بزرگتر
}
```

#### 2. تأیید/رد سیگنال‌ها

```python
# سیگنال Trend-Following در رژیم strong_trend
signal_score *= 1.5  # تقویت +50%

# سیگنال Reversal در رژیم strong_trend
signal_score *= 0.6  # تضعیف -40%

# سیگنال Reversal در رژیم range
signal_score *= 1.3  # تقویت +30%
```

#### 3. فیلتر کردن

```python
# در رژیم choppy یا range_high_volatility
if regime in ['choppy', 'range_high_volatility']:
    # رد بیشتر سیگنال‌ها
    minimum_signal_score *= 1.2  # افزایش آستانه پذیرش

    # یا حتی توقف کامل
    if confidence < 0.5:
        return None  # هیچ سیگنالی تولید نشود
```

---

### 4.6 مثال عملی کامل

**سناریو:** BTC/USDT در تایم‌فریم 1h

```python
# گام 1: تشخیص رژیم
regime = {
    'regime': 'strong_trend_high_volatility',
    'trend_strength': 'strong',
    'trend_direction': 'bullish',
    'volatility': 'high',
    'confidence': 0.82,
    'details': {
        'adx': 32.5,           # > 25 → strong
        'plus_di': 35.0,       # > minus_di → bullish
        'minus_di': 18.0,
        'atr_percent': 2.8,    # > 1.5 → high volatility
        'adx_stability': 0.85,
        'volume_ratio': 1.8    # حجم بالا
    }
}

# گام 2: تطبیق پارامترها
base_config = {
    'max_risk_per_trade_percent': 1.0,
    'preferred_risk_reward_ratio': 2.5,
    'default_stop_loss_percent': 1.5
}

adapted_params = {
    'max_risk_per_trade_percent': 1.0 × 0.85 = 0.85,    # کاهش ریسک
    'preferred_risk_reward_ratio': 2.5 × 1.1 = 2.75,    # هدف کمی دورتر
    'default_stop_loss_percent': 1.5 × 1.2 = 1.8,       # SL گسترده‌تر
    'trend_following_weight': 1.3,                      # تقویت Trend
    'reversal_weight': 0.5,                             # تضعیف Reversal
    'position_size_multiplier': 0.9                     # معامله کوچکتر
}

# گام 3: سیگنال خرید (Trend-Following)
signal = {
    'direction': 'long',
    'type': 'trend_following',
    'base_score': 70,
    'entry': 50000
}

# اعمال تطبیق‌ها
signal['adjusted_score'] = 70 × 1.3 = 91  # تقویت Trend-Following
signal['stop_loss'] = 50000 × (1 - 0.018) = 49100  # 1.8% پایین‌تر
signal['take_profit'] = 50000 + (900 × 2.75) = 52475  # RR=2.75

# محاسبه Position Size
account_balance = 10000 USDT
risk_per_trade = 10000 × 0.0085 = 85 USDT
risk_per_unit = 50000 - 49100 = 900 USDT
base_position = 85 / 900 = 0.0944 BTC
adjusted_position = 0.0944 × 0.9 = 0.085 BTC  # 10% کوچکتر

# نتیجه نهایی
final_signal = {
    'direction': 'long',
    'score': 91,                    # بعد از تقویت
    'entry': 50000,
    'stop_loss': 49100,             # 1.8% (به دلیل نوسان بالا)
    'take_profit': 52475,           # RR = 2.75
    'position_size': 0.085,         # BTC
    'max_risk': 85,                 # USDT (0.85% حساب)
    'expected_profit': 2475 × 0.085 = 210 USDT
}

# بررسی نهایی
✅ همسویی با روند: bullish + long
✅ امتیاز بالا: 91 > 60
✅ confidence بالا: 0.82 > 0.6
✅ regime مناسب: strong_trend
⚠️ نوسان بالا: ریسک و position کاهش یافت
→ سیگنال تأیید می‌شود با پارامترهای محافظه‌کارانه!
```

---

### 4.7 جدول کامل تأثیر رژیم‌ها

| رژیم | SL | RR | Risk | Size | Trend W. | Reversal W. | Trailing | توصیه |
|------|----|----|------|------|----------|-------------|----------|-------|
| `strong_trend` | 0.9× | 1.2× | 1.0× | 1.1× | 1.5× | 0.6× | ✅ (fast) | ✅✅ عالی |
| `strong_trend_high_volatility` | 1.2× | 1.1× | 0.85× | 0.9× | 1.3× | 0.5× | ✅ (normal) | ⚠️ خوب اما محتاط |
| `weak_trend` | 0.95× | 1.05× | 0.9× | 1.0× | 1.2× | 0.8× | ✅ (normal) | ✅ خوب |
| `weak_trend_high_volatility` | 1.15× | 1.0× | 0.8× | 0.85× | 1.1× | 0.7× | ✅ (slow) | ⚠️ محتاطانه |
| `range` | 1.1× | 0.9× | 0.9× | 0.95× | 0.8× | 1.3× | ✅ (normal) | 🔄 Reversal بهتر |
| `range_high_volatility` | 1.3× | 0.8× | 0.75× | 0.8× | 0.7× | 1.2× | ✅ (slow) | ❌ خطرناک |
| `tight_range` | 1.05× | 0.95× | 0.95× | 1.0× | 0.7× | 1.4× | ❌ | 🔄 Scalping |
| `choppy` | 1.25× | 0.85× | 0.7× | 0.7× | 0.5× | 0.8× | ❌ | ❌❌ خیلی خطرناک |
| `breakout` | 0.85× | 1.3× | 1.1× | 1.2× | 1.6× | 0.5× | ✅ (very fast) | ✅✅✅ فرصت عالی |

---

## خلاصه بخش 4: اهمیت Market Regime

✅ **مزایا:**
- تطبیق خودکار با شرایط بازار
- کاهش ریسک در شرایط خطرناک (choppy, range_high_volatility)
- افزایش سود در شرایط مناسب (strong_trend, breakout)
- تقویت/تضعیف سیگنال‌ها بر اساس regime
- جلوگیری از معامله در شرایط نامناسب

⚠️ **نکات مهم:**
- در رژیم `choppy` یا `range_high_volatility` → **توقف یا کاهش شدید معاملات**
- در رژیم `breakout` → **افزایش شدید فعالیت** (فرصت طلایی)
- همیشه به `confidence` توجه کن: confidence < 0.5 → **احتیاط بیشتر**
- Regime detection تأخیر دارد → نباید تنها ابزار تصمیم‌گیری باشد

🎯 **بهترین رژیم‌ها:**
1. **breakout**: فرصت طلایی 🏆
2. **strong_trend**: بهترین شرایط Trend-Following ✅
3. **tight_range**: مناسب Mean Reversion و Scalping ✅

❌ **بدترین رژیم‌ها:**
1. **choppy**: خیلی خطرناک! 🚫
2. **range_high_volatility**: بسیار خطرناک! 🚫
3. **weak_trend_high_volatility**: خطرناک ⚠️

---

**نکته طلایی:** سیستم هوشمند است و در رژیم‌های بد، **خودکار** معاملات را کاهش یا متوقف می‌کند!
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
