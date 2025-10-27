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
   - اگر بازار خیلی بی‌ثبات باشد → تولید سیگنال متوقف می‌شود
   - محل: `signal_generator.py:4872-4880`

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
```python
analysis_data['trend'] = self.detect_trend(df)
```

**چه کاری انجام می‌شود؟**
- محاسبه EMA‌های 20، 50، 100
- تعیین جهت روند (Bullish/Bearish/Neutral)
- محاسبه قدرت روند (Trend Strength)

**خروجی نمونه:**
```python
{
    'direction': 'bullish',  # یا 'bearish' یا 'neutral'
    'strength': 0.75,        # بین 0 تا 1
    'ema20': 50000,
    'ema50': 49500,
    'ema100': 49000
}
```

**امتیازدهی:**
- EMA در این مرحله **مستقیماً امتیاز تولید نمی‌کند**
- بلکه برای **تأیید جهت سیگنال** استفاده می‌شود
- اگر سیگنال خرید باشد ولی روند نزولی → امتیاز کاهش می‌یابد

---

#### مرحله 2: تحلیل اندیکاتورهای مومنتوم (RSI, Stochastic)
```python
analysis_data['momentum'] = self.analyze_momentum_indicators(df)
```

**اندیکاتورهای محاسبه شده:**

1. **RSI (Relative Strength Index)**
   - محاسبه RSI با دوره 14
   - شناسایی اشباع خرید/فروش

   **خروجی:**
   ```python
   {
       'rsi': 45.2,
       'rsi_status': 'neutral',  # oversold, overbought, neutral
       'rsi_divergence': False   # واگرایی RSI
   }
   ```

2. **Stochastic Oscillator**
   - محاسبه %K و %D
   - شناسایی تقاطع‌ها

   **خروجی:**
   ```python
   {
       'stoch_k': 55.3,
       'stoch_d': 50.1,
       'stoch_signal': 'bullish_cross'  # یا 'bearish_cross' یا 'none'
   }
   ```

**امتیازدهی:**
- RSI < 30 (oversold) + سیگنال خرید → **+10 تا +15 امتیاز**
- RSI > 70 (overbought) + سیگنال فروش → **+10 تا +15 امتیاز**
- Stochastic Cross + تأیید جهت → **+5 تا +10 امتیاز**
- واگرایی RSI → **+15 تا +20 امتیاز** (سیگنال قوی)

**نکته مهم:** RSI و Stochastic هر دو **کمکی** هستند و به تنهایی سیگنال تولید نمی‌کنند، بلکه امتیاز سیگنال را تقویت می‌کنند.

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

**پایان بخش 2 - قسمت اول**

در بخش بعدی توضیح خواهم داد:
- تحلیل‌های پیشرفته (Harmonic Patterns, Price Channels, Cyclical Patterns)
- تشخیص رژیم بازار (Market Regime)
- نحوه ترکیب امتیازات
- محاسبه امتیاز نهایی

آیا می‌خواهید ادامه دهم؟
