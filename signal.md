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

در بخش بعدی توضیح می‌دهم:
- **بخش 4:** Market Regime Detection و تطبیق پارامترها
- **بخش 5:** ترکیب امتیازات چند تایم‌فریمی
- **بخش 6:** ML/AI Enhancement و محاسبه Final Score
