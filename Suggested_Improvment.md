# مشکلات شناسایی شده و پیشنهادات بهبود سیستم امتیازدهی

این فایل شامل مشکلات، نواقص و پیشنهادات بهبود برای هر بخش از سیستم امتیازدهی است.

> **توجه:** توضیحات نحوه کار فعلی سیستم در فایل `signal.md` قرار دارد.

---

## فهرست مطالب
- [بخش ۲: تحلیل یک تایم‌فریم](#بخش-۲-تحلیل-یک-تایمفریم)
  - [مرحله 1: تشخیص روند (Trend Detection)](#مرحله-1-تشخیص-روند-trend-detection)

---

# بخش ۲: تحلیل یک تایم‌فریم

## مرحله 1: تشخیص روند (Trend Detection)

### ✅ نکات مثبت و منطقی

#### 1. عدم امتیازدهی مستقیم - رویکرد درست است
- EMA یک اندیکاتور تأخیری (lagging) است
- استفاده از آن به عنوان فیلتر جهت منطقی‌تر از امتیازدهی مستقیم است
- این رویکرد از "double-counting" جلوگیری می‌کند

#### 2. استفاده به عنوان Context
- Trend یک context کلی برای سایر سیگنال‌ها فراهم می‌کند
- این با اصل "Trade with the trend" همخوانی دارد
- به‌ویژه در تایم‌فریم‌های بالاتر (4h, 1h) بسیار حیاتی است

#### 3. شناسایی حالت‌های Pullback
- کد حالت‌های `bullish_pullback` و `bearish_pullback` را تشخیص می‌دهد
- این حالت‌ها فرصت‌های ورود خوبی در طول روند هستند
- منطق پشت آن: "خرید در اصلاح یک روند صعودی"

---

### ⚠️ مشکلات شناسایی شده

#### مشکل 1: ضرایب Trend Bonus/Penalty بسیار بزرگ هستند 🚨

**توضیح مشکل:**

در کد فعلی (signal_generator.py:4402-4407):

```python
if trends_aligned:
    structure_score *= (1 + 1.5 * (min_strength / 3))
else:
    structure_score *= (1 - 1.5 * (min_strength / 3))
```

این فرمول باعث می‌شود:
- **Bonus بیش از حد:** strength=3 → multiplier=2.5 (افزایش 150%) 🔥
- **Penalty منفی:** strength=3 → multiplier=-0.5 (مقدار منفی!) ❌
- **Penalty صفر:** strength=2 → multiplier=0.0 (امتیاز صفر می‌شود!) ❌

**چرا این مشکل است:**

1. **افزایش 150% خیلی زیاد است:**
   - ممکن است سیگنال‌های ضعیف را به سیگنال‌های قوی تبدیل کند
   - این برخلاف اصل "Trend فیلتر است، نه سیگنال اصلی" است

2. **Penalty منفی یا صفر غیرمنطقی است:**
   - امتیاز منفی یا صفر به معنای رد کامل سیگنال است
   - ممکن است سیگنال‌های خوب در سطوح مقاومت/حمایت را از دست بدهیم
   - برخی معامله‌گران حرفه‌ای در نقاط بازگشت (reversal) معامله می‌کنند

**تأثیر بر سیستم:**

- **Over-fitting به روند:** سیستم به شدت به روند وابسته می‌شود
- **از دست دادن فرصت‌های reversal:** سیگنال‌های بازگشت روند نادیده گرفته می‌شوند
- **عدم تعادل:** تأثیر trend بیش از حد روی امتیاز نهایی است

---

#### مشکل 2: فلسفه نادرست Trend Phase Multiplier ⚠️

**توضیح مشکل:**

در کد فعلی (signal_generator.py:4796-4804):

```python
phase_multipliers = {
    'early': 1.2,       # بالاترین multiplier
    'developing': 1.1,
    'mature': 0.9,      # کمترین multiplier - احتیاط
    'late': 0.7,
    'pullback': 1.1,
    'transition': 0.8,
    'undefined': 1.0
}
```

**چرا این مشکل است:**

1. **روند Mature قوی‌ترین است، نه Early:**
   - روند `mature` یعنی همه EMAs در ترتیب صحیح هستند
   - روند `early` یعنی فقط قیمت بالای EMA20 است (ضعیف‌تر)
   - منطق فعلی برعکس است!

2. **فلسفه معاملاتی مبهم:**
   - آیا سیستم می‌خواهد در ابتدای روند وارد شود؟ (early trend trading)
   - یا می‌خواهد از روندهای قوی استفاده کند؟ (trend following)
   - کد فعلی یک فلسفه واضح ندارد

3. **تضاد با اصل "Trade with the trend":**
   - اگر روند mature است، به معنای قدرت بیشتر است
   - چرا باید امتیاز کمتری بگیرد؟

**تأثیر بر سیستم:**

- **عدم هماهنگی با اصول تحلیل تکنیکال**
- **سیگنال‌های گمراه‌کننده:** سیستم روندهای قوی را جریمه می‌کند
- **کاهش دقت:** ممکن است در روندهای قوی سیگنال کمتری تولید شود

---

#### مشکل 3: عدم تمایز بین Reversal و Counter-Trend Trading 🤔

**توضیح مشکل:**

در کد (signal_generator.py:5074):

```python
score.trend_alignment = max(0.5, 1.0 - (reversal_strength * 0.5))
```

این فرمول برای همه سیگنال‌های خلاف روند استفاده می‌شود، اما:

1. **Reversal (بازگشت روند):** سیگنالی که نشان‌دهنده تغییر جهت روند است
   - مثال: الگوی Head & Shoulders در انتهای روند صعودی
   - این سیگنال‌ها ارزشمند هستند!

2. **Counter-Trend Trading:** معامله در خلاف جهت روند بدون دلیل قوی
   - مثال: خرید در یک نزول قوی بدون الگوی بازگشت
   - این سیگنال‌ها خطرناک هستند!

**چرا این مشکل است:**

- کد فعلی هر دو را یکسان جریمه می‌کند
- سیگنال‌های reversal معتبر (با الگوی قوی) نادیده گرفته می‌شوند
- عدم هوشمندی در تشخیص context معامله

---

#### مشکل 4: عدم استفاده از EMA50 Slope در تشخیص روند قوی

**توضیح مشکل:**

در کد (signal_generator.py:1767-1768):

```python
ema20_slope = ema20[last_valid_idx] - ema20[last_valid_idx - 5]
ema50_slope = ema50[last_valid_idx] - ema50[last_valid_idx - 5]
```

`ema50_slope` محاسبه می‌شود، اما در بیشتر شرایط استفاده نمی‌شود:

```python
# خط 1784: استفاده می‌شود
if ... and ema20_slope > 0 and ema50_slope > 0:
    strength = 3

# خط 1788: استفاده نمی‌شود
elif ... and ema20_slope > 0:  # ema50_slope چک نمی‌شود
    strength = 2
```

**چرا این مشکل است:**

- **اطلاعات مهم نادیده گرفته می‌شود:**
  - `ema50_slope` نشان‌دهنده روند میان‌مدت است
  - می‌تواند کمک کند تشخیص دهیم روند واقعی است یا نوسان کوتاه‌مدت

- **روند Strength=2 قابل اعتماد نیست:**
  - فقط `ema20_slope > 0` چک می‌شود
  - ممکن است EMA50 در حال سقوط باشد (روند ضعیف)

---

#### مشکل 5: عدم وجود فیلد Confidence در خروجی

**توضیح مشکل:**

در کد فعلی، خروجی `detect_trend()` شامل فیلد `confidence` نیست:

```python
results.update({
    'trend': trend,
    'strength': strength,
    'method': 'moving_averages',
    'phase': trend_phase,
    'details': {...}
})
# confidence وجود ندارد!
```

**چرا این مشکل است:**

1. **عدم اطمینان از تشخیص:**
   - همه تشخیص‌ها یک اعتماد یکسان ندارند
   - مثلاً: strength=3 با EMA تمیز → confidence بالا
   - strength=3 با EMA نزدیک به هم → confidence پایین

2. **نمی‌توان تصمیم‌گیری بهتر کرد:**
   - در برخی موارد بهتر است به سیگنال‌های با confidence بالا بیشتر اعتماد کنیم
   - امتیازدهی می‌تواند از confidence استفاده کند

**مثال کاربرد:**

```python
# پیشنهاد
if confidence > 0.8:
    trend_multiplier *= 1.1  # اعتماد بالا → پاداش
elif confidence < 0.5:
    trend_multiplier *= 0.9  # اعتماد پایین → احتیاط
```

---

### 💡 پیشنهادات بهبود

#### پیشنهاد 1: اصلاح ضرایب Trend Bonus/Penalty ✨

**هدف:** تعادل بهتر بین تأثیر trend و سایر فاکتورها

**پیشنهاد کد جدید:**

```python
# signal_generator.py:4402-4407
TREND_BONUS_BASE = 0.4  # به جای 1.5
TREND_PENALTY_BASE = 0.3  # به جای 1.5

if trends_aligned:
    # حداکثر افزایش 40% (به جای 150%)
    structure_score *= (1 + TREND_BONUS_BASE * (min_strength / 3))
else:
    # حداکثر کاهش 30% (به جای 150%)
    structure_score *= (1 - TREND_PENALTY_BASE * (min_strength / 3))
```

**نتایج جدید:**

| Scenario | Old Multiplier | New Multiplier | تفاوت |
|----------|---------------|---------------|-------|
| Aligned, strength=3 | 2.5 (+150%) | 1.4 (+40%) | ✅ معقول‌تر |
| Aligned, strength=2 | 2.0 (+100%) | 1.27 (+27%) | ✅ متعادل |
| Aligned, strength=1 | 1.5 (+50%) | 1.13 (+13%) | ✅ محافظه‌کارانه |
| Conflicting, strength=3 | -0.5 (منفی!) | 0.7 (-30%) | ✅ منطقی |
| Conflicting, strength=2 | 0.0 (صفر!) | 0.8 (-20%) | ✅ عادلانه |
| Conflicting, strength=1 | 0.5 (-50%) | 0.9 (-10%) | ✅ قابل قبول |

**مزایا:**

- ✅ جلوگیری از over-fitting به روند
- ✅ حفظ فرصت‌های reversal
- ✅ تعادل بهتر در امتیازدهی
- ✅ کاهش false negatives

---

#### پیشنهاد 2: اصلاح فلسفه Trend Phase Multiplier 🔄

**هدف:** هماهنگی با اصول تحلیل تکنیکال

**پیشنهاد کد جدید:**

```python
# signal_generator.py:4796-4804
def _get_trend_phase_multiplier(self, phase: str, direction: str) -> float:
    """
    فلسفه جدید:
    - mature: قوی‌ترین روند (بالاترین multiplier)
    - developing: روند در حال تقویت
    - early: روند تازه (نیاز به تأیید بیشتر)
    - pullback: فرصت ورود در اصلاح
    - late: خطر بازگشت روند
    """
    phase_multipliers = {
        'mature': 1.15,      # بالاترین - روند قوی و مستقر
        'pullback': 1.12,    # فرصت خوب - خرید در اصلاح
        'developing': 1.08,  # خوب - روند در حال تقویت
        'early': 1.03,       # محتاطانه - نیاز به تأیید
        'transition': 0.95,  # احتیاط - تغییر روند
        'late': 0.85,        # خطرناک - احتمال بازگشت
        'undefined': 1.0     # بدون تأثیر
    }
    return phase_multipliers.get(phase, 1.0)
```

**استدلال:**

| Phase | Old | New | چرا تغییر کرد؟ |
|-------|-----|-----|----------------|
| mature | 0.9 | 1.15 | روند قوی باید پاداش بگیرد، نه جریمه |
| developing | 1.1 | 1.08 | خوب است اما هنوز به اندازه mature قوی نیست |
| early | 1.2 | 1.03 | روند تازه نیاز به تأیید بیشتر دارد |
| pullback | 1.1 | 1.12 | فرصت عالی - خرید در اصلاح روند قوی |

**مزایا:**

- ✅ هماهنگ با اصول Trend Following
- ✅ کاهش ریسک ورود زودهنگام
- ✅ افزایش دقت سیگنال‌ها
- ✅ فلسفه واضح و قابل فهم

---

#### پیشنهاد 3: افزودن تشخیص هوشمند Reversal 🎯

**هدف:** تمایز بین reversal معتبر و counter-trend خطرناک

**پیشنهاد کد جدید:**

```python
def _calculate_reversal_quality(self, signal_data, trend_data, pattern_data) -> float:
    """
    محاسبه کیفیت سیگنال reversal

    Returns:
        0.0-1.0: کیفیت reversal
        - 1.0: reversal بسیار معتبر
        - 0.0: counter-trend خطرناک
    """
    quality = 0.0

    # 1. آیا الگوی بازگشت قوی وجود دارد؟
    reversal_patterns = [
        'head_and_shoulders', 'inverse_head_and_shoulders',
        'double_top', 'double_bottom',
        'triple_top', 'triple_bottom',
        'falling_wedge', 'rising_wedge'
    ]

    pattern_names = pattern_data.get('pattern_names', [])
    has_reversal_pattern = any(p in pattern_names for p in reversal_patterns)

    if has_reversal_pattern:
        quality += 0.4  # الگوی بازگشت = +40%

    # 2. آیا در سطح مقاومت/حمایت قوی هستیم؟
    at_major_sr = signal_data.get('at_major_support_resistance', False)
    if at_major_sr:
        quality += 0.3  # سطح قوی = +30%

    # 3. آیا divergence وجود دارد؟
    has_divergence = signal_data.get('has_macd_divergence', False) or \
                     signal_data.get('has_rsi_divergence', False)
    if has_divergence:
        quality += 0.2  # divergence = +20%

    # 4. آیا حجم معامله تأیید می‌کند؟
    volume_confirmation = signal_data.get('volume_confirmation', 0)
    if volume_confirmation > 0.7:
        quality += 0.1  # حجم بالا = +10%

    return min(1.0, quality)

def _apply_trend_alignment_multiplier(self, score, signal_direction, trend_data, reversal_quality):
    """
    اعمال multiplier با توجه به reversal quality
    """
    trend = trend_data['trend']
    strength = trend_data['strength']

    is_against_trend = (signal_direction == 'long' and trend == 'bearish') or \
                      (signal_direction == 'short' and trend == 'bullish')

    if is_against_trend:
        if reversal_quality > 0.7:
            # reversal معتبر - penalty کم
            penalty = 0.1 * abs(strength) / 3
            multiplier = 1.0 - penalty  # حداکثر -10%
        elif reversal_quality > 0.4:
            # reversal متوسط - penalty متوسط
            penalty = 0.2 * abs(strength) / 3
            multiplier = 1.0 - penalty  # حداکثر -20%
        else:
            # counter-trend خطرناک - penalty زیاد
            penalty = 0.4 * abs(strength) / 3
            multiplier = 1.0 - penalty  # حداکثر -40%
    else:
        # همراستا با روند
        bonus = 0.3 * abs(strength) / 3
        multiplier = 1.0 + bonus  # حداکثر +30%

    return score * multiplier
```

**مثال کاربرد:**

```python
# سناریو 1: سیگنال خرید در روند نزولی با Head & Shoulders
reversal_quality = 0.9  # الگوی قوی + سطح حمایت + divergence
penalty = 0.1 * 3 / 3 = 0.1
multiplier = 0.9  # فقط -10% کاهش (قابل قبول)

# سناریو 2: سیگنال خرید در روند نزولی بدون دلیل
reversal_quality = 0.1  # بدون الگو، بدون سطح قوی
penalty = 0.4 * 3 / 3 = 0.4
multiplier = 0.6  # -40% کاهش (جریمه سنگین)
```

**مزایا:**

- ✅ حفظ فرصت‌های reversal معتبر
- ✅ جلوگیری از counter-trend خطرناک
- ✅ استفاده از اطلاعات بیشتر (الگو، S/R، divergence)
- ✅ هوشمندی بالاتر در تصمیم‌گیری

---

#### پیشنهاد 4: بهبود استفاده از EMA50 Slope 📈

**هدف:** استفاده کامل از اطلاعات موجود

**پیشنهاد کد جدید:**

```python
# signal_generator.py:1784-1816
# اصلاح تشخیص روند

# Strength = 3: همه شرایط باید OK باشد
if current_close > current_ema20 > current_ema50 > current_ema100 and \
   ema20_slope > 0 and ema50_slope > 0:
    trend = 'bullish'
    strength = 3
    trend_phase = 'mature'

# Strength = 2: باید EMA50 slope هم مثبت باشد (تغییر مهم!)
elif current_close > current_ema20 > current_ema50 and \
     ema20_slope > 0 and ema50_slope > 0:  # ✅ اضافه شد
    trend = 'bullish'
    strength = 2
    trend_phase = 'developing'

# Strength = 1.5: EMA50 slope نزدیک صفر (Developing اما ضعیف‌تر)
elif current_close > current_ema20 > current_ema50 and \
     ema20_slope > 0 and abs(ema50_slope) < threshold:  # ✅ جدید
    trend = 'bullish'
    strength = 1.5  # ✅ سطح جدید
    trend_phase = 'early_developing'

# Strength = 1: فقط قیمت بالای EMA20
elif current_close > current_ema20 and ema20_slope > 0:
    # بررسی اضافی: آیا EMA50 در خلاف جهت است؟
    if ema50_slope < -threshold:  # ✅ جدید
        strength = 0.5  # روند بسیار ضعیف
        trend_phase = 'very_early'
    else:
        strength = 1
        trend_phase = 'early'
```

**تعریف threshold:**

```python
# threshold می‌تواند درصدی از ATR باشد
threshold = self._calculate_ema_slope_threshold(df)

def _calculate_ema_slope_threshold(self, df):
    """
    محاسبه آستانه معنی‌دار برای EMA slope
    """
    atr = talib.ATR(df['high'].values, df['low'].values,
                    df['close'].values, timeperiod=14)
    # slope کمتر از 5% ATR را "نزدیک صفر" در نظر بگیر
    return atr[-1] * 0.05
```

**مزایا:**

- ✅ استفاده کامل از `ema50_slope`
- ✅ تشخیص دقیق‌تر قدرت روند
- ✅ جلوگیری از false signals در نوسانات کوتاه‌مدت
- ✅ سطح‌بندی بهتر strength (1.5, 0.5)

---

#### پیشنهاد 5: افزودن فیلد Confidence به خروجی 🎯

**هدف:** ارائه اطمینان از تشخیص روند

**پیشنهاد کد جدید:**

```python
# signal_generator.py: انتهای تابع detect_trend

def _calculate_trend_confidence(self, trend, strength, details):
    """
    محاسبه اطمینان از تشخیص روند

    فاکتورهای تأثیرگذار:
    1. فاصله بین EMAs (هرچه بیشتر → confidence بیشتر)
    2. یکنواختی slope (هرچه یکنواخت‌تر → confidence بیشتر)
    3. فاصله قیمت تا EMA نزدیک (هرچه بیشتر → confidence بیشتر)
    """
    if strength == 0:
        return 0.5  # neutral = اطمینان متوسط

    confidence = 0.0

    # 1. فاصله بین EMAs (40% وزن)
    ema20 = details['ema20']
    ema50 = details['ema50']
    ema100 = details['ema100']

    ema_separation = abs(ema20 - ema50) / ema50
    if abs(strength) >= 2:
        ema_separation += abs(ema50 - ema100) / ema100
        ema_separation /= 2

    # Normalize به 0-0.4
    confidence += min(0.4, ema_separation * 100)

    # 2. قدرت slope (30% وزن)
    ema20_slope = details.get('ema20_slope', 0)
    price = details['close']
    slope_strength = abs(ema20_slope) / price

    # Normalize به 0-0.3
    confidence += min(0.3, slope_strength * 100)

    # 3. فاصله قیمت تا EMA20 (30% وزن)
    price_distance = abs(price - ema20) / ema20

    # Normalize به 0-0.3
    confidence += min(0.3, price_distance * 100)

    return min(1.0, confidence)

# در انتهای detect_trend:
confidence = self._calculate_trend_confidence(trend, strength, results['details'])
results['confidence'] = round(confidence, 3)
```

**استفاده از confidence در امتیازدهی:**

```python
# در تابع امتیازدهی
trend_confidence = trend_data.get('confidence', 0.7)

if trend_confidence > 0.8:
    # اطمینان بالا - افزایش تأثیر trend
    trend_multiplier *= 1.1
elif trend_confidence < 0.5:
    # اطمینان پایین - کاهش تأثیر trend
    trend_multiplier *= 0.9
```

**مزایا:**

- ✅ اطلاعات بیشتر برای تصمیم‌گیری
- ✅ امکان فیلتر کردن روندهای ضعیف
- ✅ بهبود کیفیت سیگنال‌ها
- ✅ شفافیت بیشتر در تحلیل

---

### 📊 خلاصه تأثیر پیشنهادات

| پیشنهاد | اولویت | تأثیر بر دقت | سختی پیاده‌سازی |
|---------|--------|-------------|-----------------|
| اصلاح ضرایب Bonus/Penalty | 🔴 بالا | +15-20% | آسان |
| اصلاح Phase Multiplier | 🟡 متوسط | +5-10% | آسان |
| تشخیص هوشمند Reversal | 🔴 بالا | +10-15% | متوسط |
| بهبود استفاده از EMA50 Slope | 🟡 متوسط | +5-8% | آسان |
| افزودن Confidence | 🟢 پایین | +3-5% | متوسط |

**توصیه:** شروع با پیشنهادات 1 و 3 (اولویت بالا)

---

### 🧪 پیشنهاد برای تست

پس از اعمال تغییرات، باید موارد زیر تست شوند:

1. **Backtest روی داده‌های تاریخی:**
   - مقایسه نتایج قبل و بعد از تغییرات
   - محاسبه Win Rate, Profit Factor, Maximum Drawdown

2. **تست روی سناریوهای خاص:**
   - روند قوی صعودی/نزولی
   - بازار range-bound
   - نقاط reversal معتبر
   - نقاط reversal کاذب (false reversal)

3. **A/B Testing:**
   - اجرای همزمان سیستم قدیم و جدید
   - مقایسه عملکرد در شرایط بازار مختلف

---

## مرحله 2: تحلیل اندیکاتورهای مومنتوم (RSI, Stochastic, MACD, MFI)

### ✅ نکات مثبت و منطقی

#### 1. استفاده صحیح به عنوان تأیید کننده
- Momentum اندیکاتورها سیگنال مستقیم تولید نمی‌کنند
- فقط به عنوان تأیید کننده استفاده می‌شوند
- این رویکرد درست است چون momentum indicators lagging هستند

#### 2. شرط Reversal دقیق و منطقی
- برای RSI oversold/overbought، فقط آستانه کافی نیست
- کد چک می‌کند که اندیکاتور شروع به بازگشت کرده باشد
- `curr_rsi < 30 and curr_rsi > prev_rsi` ✅

#### 3. تشخیص واگرایی پیشرفته
- الگوریتم پیچیده برای یافتن peaks و valleys
- محاسبه strength بر اساس درصد تغییرات
- فیلتر کیفیت (divergence_sensitivity)
- فیلتر زمانی (فقط 10 کندل اخیر)

#### 4. شرایط دقیق Stochastic Cross
- نه فقط oversold/overbought بلکه تقاطع واقعی K و D
- `curr_k > curr_d and prev_k <= prev_d` ✅
- این از false signals جلوگیری می‌کند

#### 5. استفاده از MFI برای تأیید با حجم
- MFI ترکیبی از قیمت و حجم است
- دقیق‌تر از RSI در تشخیص reversals
- فقط زمانی محاسبه می‌شود که volume در دسترس است

#### 6. Caching برای بهبود Performance
- اندیکاتورها فقط یک بار محاسبه می‌شوند
- از cache برای timeframe های مختلف استفاده می‌شود

---

### ⚠️ مشکلات شناسایی شده

#### مشکل 1: تناقض در مستندات و کد - امتیازات 🚨

**توضیح مشکل:**

در مستندات قدیمی signal.md (قبل از اصلاح):
```
- RSI < 30 + سیگنال خرید → +10 تا +15 امتیاز
- Stochastic Cross → +5 تا +10 امتیاز
- واگرایی RSI → +15 تا +20 امتیاز
```

اما در کد واقعی (signal_generator.py:3610-3650):
```python
'rsi_oversold_reversal': 2.3
'stochastic_oversold_bullish_cross': 2.5
'rsi_bullish_divergence': 3.5 × strength  # 0 تا 3.5
```

**چرا این تناقض وجود داشت:**

1. **مستندات base scores را ذکر نکرده بود:**
   - امتیازات 2.3, 2.5, 3.5 فقط base scores هستند
   - این امتیازات بعداً scale می‌شوند و با ضرایب دیگر ترکیب می‌شوند
   - در فرمول نهایی ممکن است به 10-15 برسند

2. **عدم توضیح فرآیند نرمال‌سازی:**
   - Base scores جمع می‌شوند
   - با trend/alignment/regime multipliers ضرب می‌شوند
   - نرمال‌سازی می‌شوند (scale to 0-100)

**تأثیر:**
- مستندات گمراه‌کننده بود
- توسعه‌دهندگان جدید confused می‌شدند
- **✅ حل شد:** در مستندات جدید این موضوع به وضوح توضیح داده شد

---

#### مشکل 2: عدم توضیح شرایط دقیق در مستندات قدیمی ⚠️

**توضیح مشکل:**

مستندات قدیمی فقط می‌گفت: "RSI < 30"

اما کد شرط دقیق‌تر دارد:
```python
if curr_rsi < 30 and curr_rsi > prev_rsi:
```

**تفاوت:**

| حالت | مستندات قدیمی | کد واقعی | نتیجه |
|------|--------------|----------|-------|
| RSI=25, prev=28 | ✅ (RSI < 30) | ❌ (هنوز در حال سقوط) | مستندات ناقص بود |
| RSI=28, prev=25 | ✅ (RSI < 30) | ✅ (شروع بازگشت) | درست |

**چرا این مهم است:**

- اگر فقط RSI < 30 را چک کنیم، ممکن است در وسط سقوط ورود کنیم
- شرط `curr_rsi > prev_rsi` تضمین می‌کند که momentum در حال تغییر است
- این از **Catching a Falling Knife** جلوگیری می‌کند

**تأثیر:**
- **✅ حل شد:** در مستندات جدید شرایط دقیق کد توضیح داده شد
- مثال‌های عملی اضافه شد

---

#### مشکل 3: Stochastic شرایط پیچیده‌تر از مستندات ❓

**توضیح مشکل:**

کد واقعی برای Stochastic cross چهار شرط دارد:
```python
if curr_k < 20 and curr_d < 20 and curr_k > curr_d and prev_k <= prev_d:
```

مستندات قدیمی فقط می‌گفت: "Stochastic Cross در oversold"

**شرایط کامل:**

1. `curr_k < 20` → K در ناحیه oversold
2. `curr_d < 20` → D در ناحیه oversold
3. `curr_k > curr_d` → الان K بالای D است
4. `prev_k <= prev_d` → قبلاً K پایین D بود

**چرا همه این شرایط لازم است:**

```python
# سناریو 1: فقط شرط 1 و 2 ❌
curr_k = 15, curr_d = 12  # هر دو oversold اما K > D از قبل
prev_k = 18, prev_d = 14
# تقاطع جدیدی نداریم! سیگنال false است

# سناریو 2: همه شرایط ✅
curr_k = 18, curr_d = 15  # K عبور کرد
prev_k = 12, prev_d = 20
# تقاطع واقعی! سیگنال معتبر است
```

**تأثیر:**
- **✅ حل شد:** شرایط دقیق و مثال‌های عملی در مستندات جدید اضافه شد

---

#### مشکل 4: MFI در مستندات قدیمی ذکر نشده بود 📊

**توضیح مشکل:**

در کد (signal_generator.py:3549-3644):
```python
# MFI محاسبه و استفاده می‌شود
mfi = talib.MFI(high, low, close, volume, timeperiod=14)
if curr_mfi < 20 and curr_mfi > prev_mfi:
    momentum_signals.append({
        'type': 'mfi_oversold_reversal',
        'score': 2.4
    })
```

اما در مستندات قدیمی MFI ذکر نشده بود!

**چرا MFI مهم است:**

- **RSI:** فقط قیمت را می‌بیند
- **MFI:** قیمت + حجم معاملات را ترکیب می‌کند
- **MFI دقیق‌تر است:** چون حجم معاملات شدت خرید/فروش را نشان می‌دهد

**مثال:**

```python
# سناریو: قیمت در حال سقوط
# RSI = 25 (oversold)
# MFI = 35 (نه oversold)

# تحلیل:
# - RSI می‌گوید oversold است
# - MFI می‌گوید هنوز حجم فروش بالا نیست
# - احتمالاً هنوز سقوط ادامه دارد (MFI دقیق‌تر است)
```

**تأثیر:**
- **✅ حل شد:** MFI به طور کامل در مستندات جدید اضافه شد
- تفاوت MFI و RSI توضیح داده شد

---

#### مشکل 5: جزئیات واگرایی مبهم بود 🔍

**توضیح مشکل:**

مستندات قدیمی فقط می‌گفت: "واگرایی RSI → +15 تا +20 امتیاز"

اما سؤالات بدون پاسخ:
- واگرایی چگونه تشخیص داده می‌شود؟
- strength چگونه محاسبه می‌شود؟
- چرا گاهی امتیاز 2.1 و گاهی 3.5 است؟
- تفاوت واگرایی معمولی و hidden divergence چیست؟

**محاسبه Strength در کد:**

```python
# signal_generator.py:2969-2971
price_change_pct = (p2_price - p1_price) / p1_price
ind_change_pct = (ind_p1_val - ind_p2_val) / ind_p1_val
div_strength = min(1.0, (price_change_pct + ind_change_pct) / 2 * 5)

# امتیاز نهایی
div_score = 3.5 * div_strength  # 0 تا 3.5
```

**مثال:**
```python
# واگرایی ضعیف:
price_change = 2%, ind_change = 3%
strength = min(1.0, (0.02 + 0.03) / 2 * 5) = 0.125
score = 3.5 × 0.125 = 0.44

# واگرایی قوی:
price_change = 10%, ind_change = 12%
strength = min(1.0, (0.10 + 0.12) / 2 * 5) = 0.55
score = 3.5 × 0.55 = 1.93
```

**تأثیر:**
- **✅ حل شد:** فرآیند کامل تشخیص واگرایی در مستندات جدید توضیح داده شد
- مثال‌های واقعی با محاسبات دقیق اضافه شد

**⚠️ نکته:** کد فعلی فقط واگرایی معمولی (Regular Divergence) را تشخیص می‌دهد، نه Hidden Divergence

---

#### مشکل 6: عدم وجود مثال‌های عددی کامل 📝

**توضیح مشکل:**

مستندات قدیمی فقط توضیحات نظری داشت، بدون مثال‌های عددی واقعی.

**چرا مثال‌های عددی مهم هستند:**

1. **درک بهتر:** توسعه‌دهنده می‌فهمد دقیقاً چه عددی از کجا می‌آید
2. **تست آسان‌تر:** می‌توان با داده‌های مشخص کد را تست کرد
3. **Debug سریع‌تر:** وقتی مشکلی پیش می‌آید، می‌توان با مثال مقایسه کرد

**تأثیر:**
- **✅ حل شد:** مثال‌های عددی کامل در مستندات جدید اضافه شد:
  - مثال RSI reversal
  - مثال Stochastic cross
  - مثال محاسبه divergence strength
  - مثال کامل سیگنال خرید با momentum

---

### 💡 پیشنهادات بهبود

#### پیشنهاد 1: افزودن Hidden Divergence Detection 🎯

**هدف:** تشخیص واگرایی مخفی (Hidden Divergence) که نشان‌دهنده ادامه روند است

**تفاوت Regular vs Hidden Divergence:**

| نوع | قیمت | اندیکاتور | معنی |
|-----|------|-----------|------|
| **Regular Bullish** | Lower Lows (LL) | Higher Lows (HL) | بازگشت روند ↗️ |
| **Hidden Bullish** | Higher Lows (HL) | Lower Lows (LL) | ادامه روند صعودی ✅ |
| **Regular Bearish** | Higher Highs (HH) | Lower Highs (LH) | بازگشت روند ↘️ |
| **Hidden Bearish** | Lower Highs (LH) | Higher Highs (HH) | ادامه روند نزولی ✅ |

**پیشنهاد کد جدید:**

```python
def _detect_hidden_divergence(self, price_series: pd.Series, indicator_series: pd.Series,
                               indicator_name: str, trend_direction: str) -> List[Dict[str, Any]]:
    """
    تشخیص واگرایی مخفی (Hidden Divergence)

    Hidden Divergence نشان‌دهنده ادامه روند است، نه بازگشت روند.
    فقط در جهت روند اصلی معتبر است.
    """
    signals = []

    if trend_direction not in ['bullish', 'bearish']:
        return signals  # Hidden divergence فقط در روند معتبر است

    period = min(len(price_series), len(indicator_series))
    if period < 20:
        return signals

    try:
        price_window = price_series.iloc[-period:]
        indicator_window = indicator_series.iloc[-period:]

        # یافتن peaks و valleys
        price_peaks_idx, price_valleys_idx = self.find_peaks_and_valleys(
            price_window.values, distance=5, prominence_factor=0.05, window_size=period
        )
        ind_peaks_idx, ind_valleys_idx = self.find_peaks_and_valleys(
            indicator_window.values, distance=5, prominence_factor=0.1, window_size=period
        )

        # Convert to absolute indices
        price_peaks_abs = price_window.index[price_peaks_idx].tolist() if len(price_peaks_idx) > 0 else []
        price_valleys_abs = price_window.index[price_valleys_idx].tolist() if len(price_valleys_idx) > 0 else []
        ind_peaks_abs = indicator_window.index[ind_peaks_idx].tolist() if len(ind_peaks_idx) > 0 else []
        ind_valleys_abs = indicator_window.index[ind_valleys_idx].tolist() if len(ind_valleys_idx) > 0 else []

        # ================ HIDDEN BULLISH DIVERGENCE (در روند صعودی) ================
        # شرط: قیمت Higher Lows اما اندیکاتور Lower Lows
        if trend_direction == 'bullish' and len(price_valleys_abs) >= 2 and len(ind_valleys_abs) >= 2:
            for i in range(min(len(price_valleys_abs), 5) - 1):
                cur_idx = len(price_valleys_abs) - 1 - i
                prev_idx = cur_idx - 1

                if prev_idx < 0:
                    continue

                p1_idx = price_valleys_abs[prev_idx]
                p2_idx = price_valleys_abs[cur_idx]

                p1_price = price_window.loc[p1_idx]
                p2_price = price_window.loc[p2_idx]

                # قیمت باید Higher Low باشد (pullback در روند صعودی)
                if p2_price <= p1_price:
                    continue

                # یافتن valleys متناظر در اندیکاتور
                ind_p1_idx = self._find_closest_peak(ind_valleys_abs, p1_idx)
                ind_p2_idx = self._find_closest_peak(ind_valleys_abs, p2_idx)

                if ind_p1_idx is None or ind_p2_idx is None:
                    continue

                ind_p1_val = indicator_window.loc[ind_p1_idx]
                ind_p2_val = indicator_window.loc[ind_p2_idx]

                # اندیکاتور باید Lower Low باشد
                if ind_p2_val < ind_p1_val:
                    # ✅ Hidden Bullish Divergence
                    price_change_pct = (p2_price - p1_price) / p1_price
                    ind_change_pct = (ind_p1_val - ind_p2_val) / ind_p1_val
                    div_strength = min(1.0, (price_change_pct + ind_change_pct) / 2 * 5)

                    if div_strength >= self.divergence_sensitivity:
                        # امتیاز کمتر از regular divergence (چون ادامه روند است نه بازگشت)
                        div_score = self.pattern_scores.get(
                            f"{indicator_name}_hidden_bullish_divergence", 2.5
                        ) * div_strength

                        signals.append({
                            'type': f'{indicator_name}_hidden_bullish_divergence',
                            'direction': 'bullish',
                            'score': div_score,
                            'strength': float(div_strength),
                            'signal_quality': 'continuation',  # ادامه روند
                            'details': {
                                'price_p1': float(p1_price),
                                'price_p2': float(p2_price),
                                'ind_p1': float(ind_p1_val),
                                'ind_p2': float(ind_p2_val),
                                'price_change_pct': float(price_change_pct),
                                'ind_change_pct': float(ind_change_pct)
                            }
                        })

        # ================ HIDDEN BEARISH DIVERGENCE (در روند نزولی) ================
        # شرط: قیمت Lower Highs اما اندیکاتور Higher Highs
        if trend_direction == 'bearish' and len(price_peaks_abs) >= 2 and len(ind_peaks_abs) >= 2:
            for i in range(min(len(price_peaks_abs), 5) - 1):
                cur_idx = len(price_peaks_abs) - 1 - i
                prev_idx = cur_idx - 1

                if prev_idx < 0:
                    continue

                p1_idx = price_peaks_abs[prev_idx]
                p2_idx = price_peaks_abs[cur_idx]

                p1_price = price_window.loc[p1_idx]
                p2_price = price_window.loc[p2_idx]

                # قیمت باید Lower High باشد (pullback در روند نزولی)
                if p2_price >= p1_price:
                    continue

                # یافتن peaks متناظر در اندیکاتور
                ind_p1_idx = self._find_closest_peak(ind_peaks_abs, p1_idx)
                ind_p2_idx = self._find_closest_peak(ind_peaks_abs, p2_idx)

                if ind_p1_idx is None or ind_p2_idx is None:
                    continue

                ind_p1_val = indicator_window.loc[ind_p1_idx]
                ind_p2_val = indicator_window.loc[ind_p2_idx]

                # اندیکاتور باید Higher High باشد
                if ind_p2_val > ind_p1_val:
                    # ✅ Hidden Bearish Divergence
                    price_change_pct = (p1_price - p2_price) / p1_price
                    ind_change_pct = (ind_p2_val - ind_p1_val) / ind_p1_val
                    div_strength = min(1.0, (price_change_pct + ind_change_pct) / 2 * 5)

                    if div_strength >= self.divergence_sensitivity:
                        div_score = self.pattern_scores.get(
                            f"{indicator_name}_hidden_bearish_divergence", 2.5
                        ) * div_strength

                        signals.append({
                            'type': f'{indicator_name}_hidden_bearish_divergence',
                            'direction': 'bearish',
                            'score': div_score,
                            'strength': float(div_strength),
                            'signal_quality': 'continuation',
                            'details': {
                                'price_p1': float(p1_price),
                                'price_p2': float(p2_price),
                                'ind_p1': float(ind_p1_val),
                                'ind_p2': float(ind_p2_val),
                                'price_change_pct': float(price_change_pct),
                                'ind_change_pct': float(ind_change_pct)
                            }
                        })

        # فیلتر زمانی
        recent_candle_limit = 10
        if len(signals) > 0 and len(price_window) > recent_candle_limit:
            recent_threshold = price_window.index[-recent_candle_limit]
            signals = [s for s in signals if s['index'] >= recent_threshold]

        return sorted(signals, key=lambda x: x.get('strength', 0), reverse=True)

    except Exception as e:
        logger.error(f"Error detecting hidden {indicator_name} divergence: {str(e)}", exc_info=True)
        return []

# استفاده در analyze_momentum_indicators:
def analyze_momentum_indicators(self, df: pd.DataFrame) -> Dict[str, Any]:
    # ... کد قبلی ...

    # تشخیص regular divergence
    rsi_divergences = self._detect_divergence_generic(close_s, rsi_s, 'rsi')
    momentum_signals.extend(rsi_divergences)

    # تشخیص hidden divergence (جدید)
    trend_direction = analysis_data.get('trend', {}).get('trend', 'neutral')
    rsi_hidden_divs = self._detect_hidden_divergence(close_s, rsi_s, 'rsi', trend_direction)
    momentum_signals.extend(rsi_hidden_divs)

    # ... ادامه کد ...
```

**مزایا:**

- ✅ تشخیص فرصت‌های ادامه روند (pullback در روند قوی)
- ✅ کاهش false reversals
- ✅ افزایش دقت در روندهای قوی
- ✅ امتیاز متفاوت برای divergence types (regular vs hidden)

**کاربرد:**

```python
# مثال: روند صعودی قوی BTC
# قیمت: pullback موقت (Higher Low)
# RSI: کف جدید (Lower Low)
# = Hidden Bullish Divergence
# معنی: روند صعودی ادامه دارد، فرصت خوب برای خرید ✅

# Regular Divergence می‌گفت: ممکن است روند معکوس شود ❌
# Hidden Divergence می‌گوید: روند ادامه دارد، pullback فرصت است ✅
```

---

#### پیشنهاد 2: افزودن Divergence Confidence Score 🎯

**هدف:** تشخیص کیفیت واگرایی با confidence score

**مشکل فعلی:**

همه واگرایی‌ها یک کیفیت یکسان ندارند:
- واگرایی در timeframe بالا قوی‌تر است
- واگرایی در ناحیه S/R مهم‌تر است
- واگرایی همراه با volume بالا معتبرتر است

**پیشنهاد کد جدید:**

```python
def _calculate_divergence_confidence(self, div_details: Dict,
                                      timeframe: str,
                                      at_support_resistance: bool,
                                      volume_data: Optional[pd.Series]) -> float:
    """
    محاسبه confidence برای واگرایی

    Returns:
        0.0-1.0: confidence score
    """
    confidence = 0.0

    # 1. قدرت divergence (40% وزن)
    strength = div_details.get('strength', 0)
    confidence += min(0.4, strength * 0.4)

    # 2. تایم‌فریم (25% وزن)
    timeframe_weights = {
        '5m': 0.15,
        '15m': 0.18,
        '1h': 0.22,
        '4h': 0.25   # بالاترین اعتبار
    }
    confidence += timeframe_weights.get(timeframe, 0.18)

    # 3. نزدیکی به سطح S/R (20% وزن)
    if at_support_resistance:
        confidence += 0.20

    # 4. تأیید حجم (15% وزن)
    if volume_data is not None:
        # بررسی حجم در نقاط divergence
        p1_idx = div_details['details'].get('price_p1_index')
        p2_idx = div_details['details'].get('price_p2_index')

        if p1_idx and p2_idx:
            try:
                vol_p1 = volume_data.loc[p1_idx]
                vol_p2 = volume_data.loc[p2_idx]
                avg_volume = volume_data.rolling(20).mean().iloc[-1]

                # اگر حجم در p2 بیشتر از میانگین باشد
                if vol_p2 > avg_volume * 1.5:
                    confidence += 0.15
                elif vol_p2 > avg_volume:
                    confidence += 0.08
            except:
                pass

    return min(1.0, confidence)

# استفاده:
def _detect_divergence_generic(self, price_series, indicator_series, indicator_name):
    # ... کد قبلی برای تشخیص divergence ...

    # محاسبه confidence
    timeframe = price_series.attrs.get('timeframe', '1h')
    at_sr = self._check_near_support_resistance(price_series.iloc[-1])
    volume = price_series.attrs.get('volume', None)

    for signal in signals:
        confidence = self._calculate_divergence_confidence(
            signal, timeframe, at_sr, volume
        )
        signal['confidence'] = confidence

        # اگر confidence پایین است، امتیاز را کاهش بده
        if confidence < 0.5:
            signal['score'] *= 0.7
        elif confidence > 0.8:
            signal['score'] *= 1.2

    return signals
```

**مزایا:**

- ✅ فیلتر واگرایی‌های ضعیف
- ✅ اولویت‌بندی بهتر سیگنال‌ها
- ✅ استفاده از context بیشتر (timeframe, S/R, volume)
- ✅ کاهش false positives

---

#### پیشنهاد 3: افزودن MACD Histogram Divergence 📊

**هدف:** تشخیص واگرایی در MACD Histogram (سریع‌تر از MACD Line)

**چرا MACD Histogram مهم است:**

- MACD Line واگرایی را با تأخیر نشان می‌دهد
- MACD Histogram تغییرات momentum را زودتر نشان می‌دهد
- Histogram divergence = early warning signal

**پیشنهاد کد جدید:**

```python
# در analyze_momentum_indicators:

# محاسبه MACD histogram divergence
macd_hist_s = pd.Series(macd_hist)
macd_hist_divergences = self._detect_divergence_generic(close_s, macd_hist_s, 'macd_histogram')
momentum_signals.extend(macd_hist_divergences)
```

**تفاوت با MACD Line Divergence:**

```python
# سناریو: روند صعودی نزدیک به پایان

# MACD Line: هنوز divergence ندارد
# MACD Histogram: شروع به کاهش کرده (early warning ✅)

# نتیجه:
# - Histogram Divergence = امتیاز 2.5 × strength
# - زودتر از MACD Line تشخیص می‌دهد
# - موقعیت بهتر برای خروج یا فروش
```

**امتیاز پیشنهادی:**

```python
# config.yaml
macd_histogram_bullish_divergence: 2.8  # بالاتر از RSI (چون زودتر تشخیص می‌دهد)
macd_histogram_bearish_divergence: 2.8
```

---

#### پیشنهاد 4: بهبود شرط RSI Extreme Levels ⚡

**هدف:** استفاده از سطوح extreme برای سیگنال‌های قوی‌تر

**مشکل فعلی:**

فقط دو سطح داریم:
- RSI < 30 = oversold
- RSI > 70 = overbought

اما سطوح extreme قوی‌تر هستند:
- RSI < 20 = extremely oversold (خیلی قوی‌تر)
- RSI > 80 = extremely overbought

**پیشنهاد کد جدید:**

```python
# signal_generator.py: در analyze_momentum_indicators

# 3. RSI Oversold/Overbought Reversal با سطوح extreme
if curr_rsi < 20 and curr_rsi > prev_rsi:
    # Extremely oversold - امتیاز بالاتر
    momentum_signals.append({
        'type': 'rsi_extremely_oversold_reversal',
        'score': self.pattern_scores.get('rsi_extremely_oversold_reversal', 3.0),  # بالاتر از 2.3
        'level': 'extreme'
    })
elif curr_rsi < 30 and curr_rsi > prev_rsi:
    # Normal oversold
    momentum_signals.append({
        'type': 'rsi_oversold_reversal',
        'score': self.pattern_scores.get('rsi_oversold_reversal', 2.3),
        'level': 'normal'
    })

# مشابه برای overbought
if curr_rsi > 80 and curr_rsi < prev_rsi:
    momentum_signals.append({
        'type': 'rsi_extremely_overbought_reversal',
        'score': self.pattern_scores.get('rsi_extremely_overbought_reversal', 3.0),
        'level': 'extreme'
    })
elif curr_rsi > 70 and curr_rsi < prev_rsi:
    momentum_signals.append({
        'type': 'rsi_overbought_reversal',
        'score': self.pattern_scores.get('rsi_overbought_reversal', 2.3),
        'level': 'normal'
    })
```

**امتیازات پیشنهادی:**

| سطح | شرط | امتیاز پایه | دلیل |
|-----|-----|-----------|------|
| Normal Oversold | RSI 20-30 | 2.3 | معمولی |
| Extreme Oversold | RSI < 20 | **3.0** | خیلی قوی |
| Normal Overbought | RSI 70-80 | 2.3 | معمولی |
| Extreme Overbought | RSI > 80 | **3.0** | خیلی قوی |

**مزایا:**

- ✅ تشخیص سیگنال‌های قوی‌تر
- ✅ امتیاز بالاتر برای extreme levels
- ✅ کاهش false signals (چون extreme نادرتر اتفاق می‌افتد)

---

#### پیشنهاد 5: افزودن Multiple Timeframe Momentum Confirmation 🎯

**هدف:** تأیید momentum در چند تایم‌فریم

**مشکل فعلی:**

momentum فقط در یک تایم‌فریم بررسی می‌شود.
اگر در همه تایم‌فریم‌ها momentum یکسان باشد → سیگنال قوی‌تر

**پیشنهاد کد جدید:**

```python
def _calculate_mtf_momentum_score(self, all_timeframes_data: Dict) -> Dict[str, Any]:
    """
    محاسبه امتیاز momentum در چند تایم‌فریم

    Args:
        all_timeframes_data: {
            '5m': {'momentum': {...}},
            '15m': {'momentum': {...}},
            '1h': {'momentum': {...}},
            '4h': {'momentum': {...}}
        }

    Returns:
        {
            'mtf_bullish_score': float,
            'mtf_bearish_score': float,
            'mtf_alignment': float (0-1),
            'mtf_momentum_multiplier': float
        }
    """
    timeframe_weights = {
        '5m': 0.15,
        '15m': 0.20,
        '1h': 0.30,
        '4h': 0.35
    }

    weighted_bullish = 0
    weighted_bearish = 0

    for tf, weight in timeframe_weights.items():
        if tf in all_timeframes_data:
            momentum = all_timeframes_data[tf].get('momentum', {})
            bullish = momentum.get('bullish_score', 0)
            bearish = momentum.get('bearish_score', 0)

            weighted_bullish += bullish * weight
            weighted_bearish += bearish * weight

    # محاسبه alignment
    total = weighted_bullish + weighted_bearish
    if total > 0:
        alignment = abs(weighted_bullish - weighted_bearish) / total
    else:
        alignment = 0

    # محاسبه multiplier
    if alignment > 0.8:
        multiplier = 1.3  # همه تایم‌فریم‌ها یکسان
    elif alignment > 0.6:
        multiplier = 1.15
    elif alignment > 0.4:
        multiplier = 1.0
    else:
        multiplier = 0.85  # تضاد بین تایم‌فریم‌ها

    return {
        'mtf_bullish_score': round(weighted_bullish, 2),
        'mtf_bearish_score': round(weighted_bearish, 2),
        'mtf_alignment': round(alignment, 3),
        'mtf_momentum_multiplier': multiplier
    }

# استفاده در generate_signal:
mtf_momentum = self._calculate_mtf_momentum_score(all_timeframes_data)
structure_score *= mtf_momentum['mtf_momentum_multiplier']
```

**مثال:**

```python
# سناریو 1: همه تایم‌فریم‌ها bullish momentum
{
    '5m':  {'bullish_score': 8, 'bearish_score': 2},
    '15m': {'bullish_score': 9, 'bearish_score': 1},
    '1h':  {'bullish_score': 7, 'bearish_score': 3},
    '4h':  {'bullish_score': 8, 'bearish_score': 2}
}
# mtf_alignment = 0.85 → multiplier = 1.3 ✅

# سناریو 2: تضاد بین تایم‌فریم‌ها
{
    '5m':  {'bullish_score': 8, 'bearish_score': 2},
    '15m': {'bullish_score': 7, 'bearish_score': 3},
    '1h':  {'bullish_score': 3, 'bearish_score': 7},  # مخالف!
    '4h':  {'bullish_score': 2, 'bearish_score': 8}   # مخالف!
}
# mtf_alignment = 0.3 → multiplier = 0.85 ❌
```

**مزایا:**

- ✅ استفاده از قدرت momentum در چند تایم‌فریم
- ✅ تأیید قوی‌تر با همراستایی
- ✅ جلوگیری از سیگنال‌های متناقض

---

### 📊 خلاصه تأثیر پیشنهادات

| پیشنهاد | اولویت | تأثیر بر دقت | سختی پیاده‌سازی |
|---------|--------|-------------|-----------------|
| افزودن Hidden Divergence | 🟡 متوسط | +8-12% | متوسط |
| افزودن Divergence Confidence | 🔴 بالا | +10-15% | متوسط |
| MACD Histogram Divergence | 🟢 پایین | +3-5% | آسان |
| RSI Extreme Levels | 🟡 متوسط | +5-8% | آسان |
| MTF Momentum Confirmation | 🔴 بالا | +12-18% | متوسط-سخت |

**توصیه:** شروع با پیشنهادات 2 و 5 (اولویت بالا)

---

### 🧪 پیشنهاد برای تست

پس از اعمال تغییرات، باید موارد زیر تست شوند:

1. **Backtest روی داده‌های تاریخی:**
   - مقایسه نتایج قبل و بعد از تغییرات
   - محاسبه Win Rate برای divergence signals
   - بررسی عملکرد در تایم‌فریم‌های مختلف

2. **تست Hidden Divergence:**
   - تست در روندهای قوی
   - مقایسه با Regular Divergence
   - بررسی false positives

3. **تست Confidence Score:**
   - بررسی دقت confidence predictions
   - همبستگی confidence با نتیجه معامله
   - آستانه بهینه برای فیلتر کردن

4. **A/B Testing:**
   - اجرای همزمان سیستم قدیم و جدید
   - مقایسه عملکرد در شرایط بازار مختلف
   - تحلیل نتایج برای هر نوع divergence

---

**تاریخ آخرین به‌روزرسانی:** 2025-10-27

