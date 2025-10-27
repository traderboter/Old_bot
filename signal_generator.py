"""
Advanced Multi-Timeframe Trading Signal System

This system combines the best features of both codebases to create a professional
trading signal generation system with:

1. Adaptive learning system that improves parameters based on past trade results
2. Correlation management to avoid overexposure to similar instruments
3. Emergency circuit breaker to prevent losses in abnormal market conditions
4. Advanced technical analysis including harmonic patterns, price channels, and cycle analysis
5. Multi-timeframe confirmation with structure analysis
6. Advanced volatility filtering and regime-based parameter adaptation
7. Comprehensive MACD analysis with divergence detection

The system is designed for production use with optimized performance through caching,
parallel execution, and robust error handling.
"""

import logging
import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional, Tuple, Union, Set, TypeVar, cast, Callable, DefaultDict
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone, timedelta
import copy
import asyncio
import random
import time
import json
import os
from functools import lru_cache, partial
import warnings
from collections import defaultdict, deque
from pathlib import Path

# Core technical analysis libraries
import talib
from scipy import signal as sig_processing
from scipy import stats
import scipy

# Optional library for optimized rolling calculations
try:
    import bottleneck as bn
    use_bottleneck = True
except ImportError:
    use_bottleneck = False

# Concurrent processing
from concurrent.futures import ThreadPoolExecutor

# Type definitions
DataFrame = TypeVar('DataFrame', bound=pd.DataFrame)
TimeSeriesData = Union[np.ndarray, pd.Series]
T = TypeVar('T')

# Configure logger
logger = logging.getLogger(__name__)
if use_bottleneck:
    logger.info("Bottleneck library found, using optimized rolling calculations.")
else:
    logger.info("Bottleneck library not found, using standard pandas rolling calculations.")


# ===============================================
#      Dataclass Models for Signal Information
# ===============================================
@dataclass
class SignalScore:
    """Detailed signal score components for signal quality evaluation"""
    base_score: float = 0.0  # Raw base score
    timeframe_weight: float = 1.0  # Higher timeframe confirmation factor
    trend_alignment: float = 1.0  # Alignment with trend factor
    volume_confirmation: float = 1.0  # Volume confirmation factor
    pattern_quality: float = 1.0  # Pattern quality factor
    confluence_score: float = 0.0  # Confluence score (includes RR)
    final_score: float = 0.0  # Final calculated score
    symbol_performance_factor: float = 1.0  # Symbol historical performance
    correlation_safety_factor: float = 1.0  # Correlation safety factor
    macd_analysis_score: float = 1.0  # MACD analysis score
    structure_score: float = 1.0  # Higher timeframe structure score
    volatility_score: float = 1.0  # Volatility condition score
    harmonic_pattern_score: float = 1.0  # Harmonic pattern score
    price_channel_score: float = 1.0  # Price channel score
    cyclical_pattern_score: float = 1.0  # Cyclical pattern score

    def to_dict(self) -> Dict[str, float]:
        """Convert to dictionary for serialization"""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, float]) -> 'SignalScore':
        """Create from dictionary"""
        return cls(**data)


@dataclass
class SignalInfo:
    """Complete trading signal information class"""
    symbol: str
    timeframe: str  # Primary timeframe (usually shortest)
    signal_type: str  # 'multi_timeframe', 'reversal', 'breakout', 'harmonic', etc.
    direction: str  # 'long' or 'short'
    entry_price: float
    stop_loss: float
    take_profit: float
    risk_reward_ratio: float
    timestamp: datetime  # Signal generation time
    pattern_names: List[str] = field(default_factory=list)  # Patterns involved in signal
    score: SignalScore = field(default_factory=SignalScore)  # Detailed score
    confirmation_timeframes: List[str] = field(default_factory=list)  # Analyzed timeframes
    rejected_reason: Optional[str] = None  # Reason if rejected
    # Additional information (optional)
    regime: Optional[str] = None  # Detected market regime
    is_reversal: bool = False  # Whether signal is a reversal
    adapted_config: Optional[Dict[str, Any]] = None  # For storing adapted config
    # Advanced analysis details
    macd_details: Optional[Dict[str, Any]] = field(default_factory=dict)
    volatility_details: Optional[Dict[str, Any]] = field(default_factory=dict)
    htf_details: Optional[Dict[str, Any]] = field(default_factory=dict)
    harmonic_details: Optional[Dict[str, Any]] = field(default_factory=dict)
    channel_details: Optional[Dict[str, Any]] = field(default_factory=dict)
    cyclical_details: Optional[Dict[str, Any]] = field(default_factory=dict)
    # New fields
    correlated_symbols: List[Tuple[str, float]] = field(default_factory=list)  # Correlated symbols and their correlation value
    signal_id: str = ""  # Unique ID for signal tracking
    market_context: Dict[str, Any] = field(default_factory=dict)  # Market context information
    trade_result: Optional[Dict[str, Any]] = None  # Trade result for learning

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization"""
        result = asdict(self)
        # Convert datetime to ISO string
        if self.timestamp:
            result['timestamp'] = self.timestamp.isoformat()
        # Convert SignalScore to dictionary
        if self.score:
            result['score'] = self.score.to_dict()
        return result

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'SignalInfo':
        """Create from dictionary"""
        data_copy = data.copy()
        # Convert ISO string to datetime
        if 'timestamp' in data_copy and isinstance(data_copy['timestamp'], str):
            data_copy['timestamp'] = datetime.fromisoformat(data_copy['timestamp'])
        # Convert score dictionary to SignalScore
        if 'score' in data_copy and isinstance(data_copy['score'], dict):
            data_copy['score'] = SignalScore.from_dict(data_copy['score'])
        return cls(**data_copy)

    def ensure_aware_timestamp(self) -> None:
        """Ensure timestamp is timezone-aware"""
        if self.timestamp and self.timestamp.tzinfo is None:
            # Convert naive to aware with UTC
            self.timestamp = self.timestamp.replace(tzinfo=timezone.utc)

    def generate_signal_id(self) -> None:
        """Generate unique ID for the signal"""
        if not self.signal_id:
            time_part = int(time.time())
            random_part = random.randint(1000, 9999)
            symbol_part = ''.join(c for c in self.symbol if c.isalnum())[:5]
            direction_part = 'L' if self.direction == 'long' else 'S'
            self.signal_id = f"{symbol_part}_{direction_part}_{time_part}_{random_part}"


@dataclass
class TradeResult:
    """Trade result class for adaptive learning system"""
    signal_id: str  # Related signal ID
    symbol: str  # Traded symbol
    direction: str  # Trade direction ('long' or 'short')
    entry_price: float  # Entry price
    exit_price: float  # Exit price
    stop_loss: float  # Initial stop loss
    take_profit: float  # Initial take profit
    entry_time: datetime  # Entry time
    exit_time: datetime  # Exit time
    exit_reason: str  # Exit reason ('tp', 'sl', 'manual', 'trailing')
    profit_pct: float  # Profit/loss as percentage
    profit_r: float  # Profit/loss in R
    market_regime: Optional[str] = None  # Market regime during trade
    pattern_names: List[str] = field(default_factory=list)  # Patterns involved
    timeframe: str = ""  # Primary timeframe
    signal_score: float = 0.0  # Initial signal score
    trade_duration: Optional[timedelta] = None  # Trade duration
    signal_type: str = ""  # Signal type

    def __post_init__(self):
        """Calculate trade duration after initialization"""
        if self.entry_time and self.exit_time and not self.trade_duration:
            self.trade_duration = self.exit_time - self.entry_time

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization"""
        result = asdict(self)
        # Convert datetime to ISO string
        if self.entry_time:
            result['entry_time'] = self.entry_time.isoformat()
        if self.exit_time:
            result['exit_time'] = self.exit_time.isoformat()
        if self.trade_duration:
            result['trade_duration'] = str(self.trade_duration)
        return result

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'TradeResult':
        """Create from dictionary"""
        data_copy = data.copy()
        # Convert ISO string to datetime
        if 'entry_time' in data_copy and isinstance(data_copy['entry_time'], str):
            data_copy['entry_time'] = datetime.fromisoformat(data_copy['entry_time'])
        if 'exit_time' in data_copy and isinstance(data_copy['exit_time'], str):
            data_copy['exit_time'] = datetime.fromisoformat(data_copy['exit_time'])
        # Remove trade_duration to recalculate in __post_init__
        if 'trade_duration' in data_copy and isinstance(data_copy['trade_duration'], str):
            del data_copy['trade_duration']
        return cls(**data_copy)


# ===============================================
#      Market Regime Detector
# ===============================================
class MarketRegimeDetector:
    """Detects market regime (trend, volatility) and adapts parameters."""

    def __init__(self, config: Dict[str, Any]):
        """Initialize with configuration."""
        self.config = config.get('market_regime', {})
        self.enabled = self.config.get('enabled', True)

        # Indicator parameters
        self.adx_period = self.config.get('adx_period', 14)
        self.volatility_period = self.config.get('volatility_period', 20)  # ATR period

        # Detection thresholds
        self.strong_trend_threshold = self.config.get('strong_trend_threshold', 25)
        self.weak_trend_threshold = self.config.get('weak_trend_threshold', 20)
        self.high_volatility_threshold = self.config.get('high_volatility_threshold', 1.5)  # ATR %
        self.low_volatility_threshold = self.config.get('low_volatility_threshold', 0.5)  # ATR %

        # Cache results
        self._regime_cache: Dict[str, Tuple[Dict[str, Any], float]] = {}
        self._cache_ttl_seconds = 300  # 5 minutes

        # Required samples
        self._required_samples = max(self.adx_period, self.volatility_period) + 10  # Extra buffer

        # Added: sequence for analyzing regime change trends
        self._regime_history = deque(maxlen=10)  # Keep last 10 regimes
        self._regime_transition_probabilities: DefaultDict[str, DefaultDict[str, int]] = defaultdict(
            lambda: defaultdict(int))

        logger.info(f"MarketRegimeDetector initialized. Enabled: {self.enabled}")

    def detect_regime(self, df: DataFrame) -> Dict[str, Any]:
        """Detect market regime based on ADX and ATR."""
        if not self.enabled:
            return {'regime': 'disabled', 'confidence': 1.0, 'details': {}}  # Return disabled state

        # For better performance, check cache
        cache_key = self._generate_cache_key(df)
        if cache_key in self._regime_cache:
            cached_result, timestamp = self._regime_cache[cache_key]
            if time.time() - timestamp < self._cache_ttl_seconds:
                return cached_result

        # Minimum data needed
        if df is None or len(df) < self._required_samples:
            logger.debug(
                f"Not enough data ({len(df) if df is not None else 0} rows) to detect market regime (requires {self._required_samples}).")
            return {'regime': 'unknown_data', 'confidence': 0.0, 'details': {}}

        try:
            # Ensure copy to avoid unintended changes
            df_copy = df.copy()
            high_prices = df_copy['high'].values.astype(np.float64)
            low_prices = df_copy['low'].values.astype(np.float64)
            close_prices = df_copy['close'].values.astype(np.float64)

            # Calculate ADX, +DI, -DI with converted arrays
            adx = talib.ADX(high_prices, low_prices, close_prices, timeperiod=self.adx_period)
            plus_di = talib.PLUS_DI(high_prices, low_prices, close_prices, timeperiod=self.adx_period)
            minus_di = talib.MINUS_DI(high_prices, low_prices, close_prices, timeperiod=self.adx_period)

            # Calculate ATR% with converted arrays
            atr = talib.ATR(high_prices, low_prices, close_prices, timeperiod=self.volatility_period)
            atr_percent = np.where(close_prices > 0, (atr / close_prices) * 100, 0)

            # Get last valid values (ignoring initial NaNs)
            last_valid_idx = self._find_last_valid_index([adx, atr_percent])
            if last_valid_idx is None:
                logger.warning("Could not find valid ADX/ATR values for regime detection.")
                return {'regime': 'unknown_calc', 'confidence': 0.0, 'details': {}}

            current_adx = adx[last_valid_idx]
            current_plus_di = plus_di[last_valid_idx]
            current_minus_di = minus_di[last_valid_idx]
            current_atr_percent = atr_percent[last_valid_idx]

            # Determine trend strength and direction
            if current_adx > self.strong_trend_threshold:
                trend_strength = 'strong'
            elif current_adx > self.weak_trend_threshold:
                trend_strength = 'weak'
            else:
                trend_strength = 'no_trend'

            trend_direction = 'bullish' if current_plus_di > current_minus_di else 'bearish'

            # Determine volatility level
            if current_atr_percent > self.high_volatility_threshold:
                volatility = 'high'
            elif current_atr_percent < self.low_volatility_threshold:
                volatility = 'low'
            else:
                volatility = 'normal'

            # Determine final regime
            if trend_strength == 'strong':
                regime = f'strong_trend_{volatility}'
            elif trend_strength == 'weak':
                regime = f'weak_trend_{volatility}'
            else:
                regime = f'range_{volatility}'  # No trend = Range

            # Calculate confidence (e.g., based on ADX distance from thresholds)
            confidence = min(1.0, abs(current_adx - self.weak_trend_threshold) / (
                    self.strong_trend_threshold - self.weak_trend_threshold + 1e-6))
            confidence = max(0.1, confidence)  # Minimum confidence 0.1

            details = {
                'adx': round(current_adx, 2),
                'plus_di': round(current_plus_di, 2),
                'minus_di': round(current_minus_di, 2),
                'atr_percent': round(current_atr_percent, 3)
            }

            # Added: regime transition probability information
            if self._regime_history:
                prev_regime = self._regime_history[-1]
                self._regime_transition_probabilities[prev_regime][regime] += 1

                # Calculate next regime change probability
                next_regime_probs = self._calculate_next_regime_probabilities(regime)
                details['next_regime_probabilities'] = next_regime_probs

            # Add to regime history
            self._regime_history.append(regime)

            logger.debug(f"Regime Detected: {regime}, Strength: {trend_strength} ({details['adx']}), "
                         f"Direction: {trend_direction}, Volatility: {volatility} ({details['atr_percent']}), Confidence: {confidence:.2f}")

            result = {
                'regime': regime,
                'trend_strength': trend_strength,
                'trend_direction': trend_direction,
                'volatility': volatility,
                'confidence': confidence,
                'details': details
            }

            # Save to cache
            self._regime_cache[cache_key] = (result, time.time())

            return result

        except Exception as e:
            logger.error(f"Error detecting market regime: {str(e)}", exc_info=True)
            return {'regime': 'error', 'confidence': 0.0, 'details': {'error': str(e)}}

    def _calculate_next_regime_probabilities(self, current_regime: str) -> Dict[str, float]:
        """Calculate transition probabilities to next regimes"""
        transitions = self._regime_transition_probabilities[current_regime]
        total_transitions = sum(transitions.values())

        if total_transitions == 0:
            return {}

        return {next_regime: count / total_transitions for next_regime, count in transitions.items()}

    def _find_last_valid_index(self, arrays: List[np.ndarray]) -> Optional[int]:
        """Find the last valid index across multiple arrays"""
        if not arrays:
            return None

        # Last index of arrays
        max_len = min(len(arr) for arr in arrays)
        if max_len == 0:
            return None

        # Search from end to beginning
        for i in range(-1, -max_len - 1, -1):
            # Check if all arrays are valid at this index
            if all(not np.isnan(arr[i]) for arr in arrays):
                return i

        return None

    def _generate_cache_key(self, df: DataFrame) -> str:
        """Generate cache key from dataframe"""
        if df is None or len(df) == 0:
            return "empty_dataframe"

        # For simplicity, use last_valid_index and last candle value
        try:
            last_idx = df.index[-1]
            last_close = df['close'].iloc[-1]
            # Create key from timestamp and price
            timestamp_str = str(last_idx) if isinstance(last_idx, (int, float)) else str(
                last_idx.timestamp()) if hasattr(last_idx, 'timestamp') else str(last_idx)
            return f"{timestamp_str}_{last_close:.6f}"
        except (IndexError, KeyError):
            # In case of error
            return f"dataframe_len_{len(df)}"

    def get_adapted_parameters(self, regime_info: Dict[str, Any], base_config: Dict[str, Any]) -> Dict[str, Any]:
        """Adjust strategy parameters based on market regime."""
        if not self.enabled or regime_info.get('regime', 'disabled') in ['disabled', 'unknown_data', 'unknown_calc',
                                                                         'error']:
            return base_config  # If disabled or unknown, return base config

        # Create deep copy to avoid changing original config
        adapted_config = copy.deepcopy(base_config)
        regime = regime_info.get('regime')
        trend_strength = regime_info.get('trend_strength')
        volatility = regime_info.get('volatility')
        confidence = regime_info.get('confidence', 0.5)  # Confidence in regime detection

        # Get relevant config sections
        risk_params = adapted_config.setdefault('risk_management', {})
        signal_params = adapted_config.setdefault('signal_generation', {})

        # Base values for recovery
        base_risk = copy.deepcopy(self.config.get('risk_management', {}))
        base_signal = copy.deepcopy(self.config.get('signal_generation', {}))

        # --- Adjust risk parameters ---
        # 1. Base risk percentage
        base_risk_percent = base_risk.get('max_risk_per_trade_percent', 1.5)
        risk_modifier = 1.0
        if trend_strength == 'strong':
            risk_modifier = 1.1  # Slightly more risk in strong trend
        elif trend_strength == 'no_trend':
            risk_modifier = 0.8  # Less risk in range

        if volatility == 'high':
            risk_modifier *= 0.7  # Less risk in high volatility
        elif volatility == 'low':
            risk_modifier *= 0.9  # Slightly less risk in low volatility

        # Apply with confidence consideration
        risk_params['max_risk_per_trade_percent'] = base_risk_percent * (1.0 + (risk_modifier - 1.0) * confidence)

        # 2. Risk-reward ratio
        base_rr = base_risk.get('preferred_risk_reward_ratio', 2.5)
        rr_modifier = 1.0
        if trend_strength == 'strong':
            rr_modifier = 1.2  # Higher RR in trend
        elif trend_strength == 'no_trend':
            rr_modifier = 0.8  # Lower RR in range

        risk_params['preferred_risk_reward_ratio'] = base_rr * (1.0 + (rr_modifier - 1.0) * confidence)
        # Ensure minimum RR
        risk_params['preferred_risk_reward_ratio'] = max(base_risk.get('min_risk_reward_ratio', 1.5),
                                                         risk_params['preferred_risk_reward_ratio'])

        # 3. Default stop loss percentage (for when ATR/SR is not available)
        base_sl_percent = base_risk.get('default_stop_loss_percent', 1.5)
        sl_modifier = 1.0
        if volatility == 'high':
            sl_modifier = 1.3  # Wider stop loss
        elif volatility == 'low':
            sl_modifier = 0.8  # Tighter stop loss

        risk_params['default_stop_loss_percent'] = base_sl_percent * (1.0 + (sl_modifier - 1.0) * confidence)

        # --- Adjust signal parameters ---
        # 1. Minimum signal score
        base_min_score = base_signal.get('minimum_signal_score', 33)
        score_modifier = 1.0
        if trend_strength == 'no_trend' or volatility == 'high':
            score_modifier = 1.1  # More strict in range or high volatility

        signal_params['minimum_signal_score'] = base_min_score * (1.0 + (score_modifier - 1.0) * confidence)

        # Round values for better readability
        for params in [risk_params, signal_params]:
            for key, value in params.items():
                if isinstance(value, float):
                    params[key] = round(value, 2)

        logger.debug(f"Regime '{regime}' (Conf: {confidence:.2f}) -> Adapted Params: "
                     f"Risk%: {risk_params['max_risk_per_trade_percent']:.2f}, "
                     f"RR: {risk_params['preferred_risk_reward_ratio']:.2f}, "
                     f"MinScore: {signal_params['minimum_signal_score']:.2f}")

        return adapted_config


# ===============================================
#      Adaptive Learning System
# ===============================================
class AdaptiveLearningSystem:
    """Adaptive learning system to improve signal parameters based on past results"""

    def __init__(self, config: Dict[str, Any]):
        """Initialize with configuration"""
        self.config = config.get('adaptive_learning', {})
        self.enabled = self.config.get('enabled', True)
        self.data_file = self.config.get('data_file', 'adaptive_learning_data.json')
        self.max_history_per_symbol = self.config.get('max_history_per_symbol', 100)
        self.learning_rate = self.config.get('learning_rate', 0.1)  # Learning rate (adaptation speed)
        self.symbol_performance_weight = self.config.get('symbol_performance_weight', 0.3)
        self.pattern_performance_weight = self.config.get('pattern_performance_weight', 0.3)
        self.regime_performance_weight = self.config.get('regime_performance_weight', 0.2)
        self.default_pattern_score = self.config.get('default_pattern_score', 1.0)

        # Learning data
        self.trade_history: List[TradeResult] = []
        self.symbol_performance: Dict[str, Dict[str, float]] = {}  # {symbol: {'avg_profit_r': x, 'win_rate': y, ...}}
        self.pattern_performance: Dict[str, Dict[str, float]] = {}  # {pattern: {'avg_profit_r': x, 'win_rate': y, ...}}
        self.regime_performance: Dict[str, Dict[str, float]] = {}  # {regime: {'avg_profit_r': x, 'win_rate': y, ...}}
        self.timeframe_performance: Dict[
            str, Dict[str, float]] = {}  # {timeframe: {'avg_profit_r': x, 'win_rate': y, ...}}

        # Calculation cache
        self._performance_cache: Dict[str, Dict[str, Any]] = {}
        self._cache_ttl_seconds = 3600  # 1 hour

        # Load existing data
        self._load_data()

        logger.info(f"AdaptiveLearningSystem initialized. Enabled: {self.enabled}, Data file: {self.data_file}")

    def _load_data(self) -> None:
        """Load learning data from file"""
        try:
            if not os.path.exists(self.data_file):
                logger.info(f"Adaptive learning data file not found: {self.data_file}, starting with empty data.")
                return

            with open(self.data_file, 'r') as f:
                data = json.load(f)

            # Restore trade history
            if 'trade_history' in data:
                self.trade_history = [TradeResult.from_dict(trade) for trade in data['trade_history']]

            # Restore performance data
            self.symbol_performance = data.get('symbol_performance', {})
            self.pattern_performance = data.get('pattern_performance', {})
            self.regime_performance = data.get('regime_performance', {})
            self.timeframe_performance = data.get('timeframe_performance', {})

            logger.info(f"Loaded adaptive learning data: {len(self.trade_history)} trades, "
                        f"{len(self.symbol_performance)} symbols, {len(self.pattern_performance)} patterns.")
        except Exception as e:
            logger.error(f"Error loading adaptive learning data: {e}", exc_info=True)
            # Continue with empty data if loading fails

    def save_data(self) -> None:
        """Save learning data to file"""
        if not self.enabled:
            return

        try:
            # Create directory if needed
            os.makedirs(os.path.dirname(os.path.abspath(self.data_file)), exist_ok=True)

            # Prepare data for saving
            data = {
                'trade_history': [trade.to_dict() for trade in self.trade_history],
                'symbol_performance': self.symbol_performance,
                'pattern_performance': self.pattern_performance,
                'regime_performance': self.regime_performance,
                'timeframe_performance': self.timeframe_performance,
                'last_updated': datetime.now().isoformat()
            }

            # Save to file
            with open(self.data_file, 'w') as f:
                json.dump(data, f, indent=2)

            logger.info(f"Saved adaptive learning data to {self.data_file}")
        except Exception as e:
            logger.error(f"Error saving adaptive learning data: {e}", exc_info=True)

    def add_trade_result(self, trade_result: TradeResult) -> None:
        """Add a trade result to learning data and update performance metrics"""
        if not self.enabled:
            return

        try:
            # Add to history
            self.trade_history.append(trade_result)

            # Limit history size
            if len(self.trade_history) > self.max_history_per_symbol * 10:
                # Remove oldest trades
                self.trade_history = self.trade_history[-self.max_history_per_symbol * 10:]

            # Update performance metrics
            self._update_symbol_performance(trade_result)
            self._update_pattern_performance(trade_result)
            self._update_regime_performance(trade_result)
            self._update_timeframe_performance(trade_result)

            # Clear cache
            self._performance_cache.clear()

            # Auto-save every 10 trades
            if len(self.trade_history) % 10 == 0:
                self.save_data()

            logger.debug(f"Added trade result for {trade_result.symbol}: "
                         f"Profit R: {trade_result.profit_r:.2f}, Exit: {trade_result.exit_reason}")
        except Exception as e:
            logger.error(f"Error adding trade result: {e}", exc_info=True)

    def _update_symbol_performance(self, trade_result: TradeResult) -> None:
        """Update performance metrics for a symbol"""
        symbol = trade_result.symbol
        direction = trade_result.direction

        # Create structure if it doesn't exist
        if symbol not in self.symbol_performance:
            self.symbol_performance[symbol] = {
                'long': {'count': 0, 'win_count': 0, 'avg_profit_r': 0.0, 'win_rate': 0.0},
                'short': {'count': 0, 'win_count': 0, 'avg_profit_r': 0.0, 'win_rate': 0.0},
                'total': {'count': 0, 'win_count': 0, 'avg_profit_r': 0.0, 'win_rate': 0.0}
            }

        # Update stats for specific direction
        perf = self.symbol_performance[symbol][direction]
        perf['count'] += 1
        is_win = trade_result.profit_r > 0
        if is_win:
            perf['win_count'] += 1

        # Update moving average profit
        perf['avg_profit_r'] = ((perf['avg_profit_r'] * (perf['count'] - 1)) + trade_result.profit_r) / perf[
            'count']
        perf['win_rate'] = perf['win_count'] / perf['count']

        # Update total stats
        total = self.symbol_performance[symbol]['total']
        total['count'] += 1
        if is_win:
            total['win_count'] += 1
        total['avg_profit_r'] = ((total['avg_profit_r'] * (total['count'] - 1)) + trade_result.profit_r) / total[
            'count']
        total['win_rate'] = total['win_count'] / total['count']

    def _update_pattern_performance(self, trade_result: TradeResult) -> None:
        """Update performance metrics for patterns"""
        for pattern in trade_result.pattern_names:
            # Create structure if it doesn't exist
            if pattern not in self.pattern_performance:
                self.pattern_performance[pattern] = {
                    'count': 0, 'win_count': 0, 'avg_profit_r': 0.0, 'win_rate': 0.0
                }

            # Update stats
            perf = self.pattern_performance[pattern]
            perf['count'] += 1
            is_win = trade_result.profit_r > 0
            if is_win:
                perf['win_count'] += 1

            # Update moving average profit
            perf['avg_profit_r'] = ((perf['avg_profit_r'] * (perf['count'] - 1)) + trade_result.profit_r) / perf[
                'count']
            perf['win_rate'] = perf['win_count'] / perf['count']

    def _update_regime_performance(self, trade_result: TradeResult) -> None:
        """Update performance metrics for market regimes"""
        if not trade_result.market_regime:
            return

        regime = trade_result.market_regime
        direction = trade_result.direction

        # Create structure if it doesn't exist
        if regime not in self.regime_performance:
            self.regime_performance[regime] = {
                'long': {'count': 0, 'win_count': 0, 'avg_profit_r': 0.0, 'win_rate': 0.0},
                'short': {'count': 0, 'win_count': 0, 'avg_profit_r': 0.0, 'win_rate': 0.0},
                'total': {'count': 0, 'win_count': 0, 'avg_profit_r': 0.0, 'win_rate': 0.0}
            }

        # Update stats for specific direction
        perf = self.regime_performance[regime][direction]
        perf['count'] += 1
        is_win = trade_result.profit_r > 0
        if is_win:
            perf['win_count'] += 1

        # Update moving average profit
        perf['avg_profit_r'] = ((perf['avg_profit_r'] * (perf['count'] - 1)) + trade_result.profit_r) / perf[
            'count']
        perf['win_rate'] = perf['win_count'] / perf['count']

        # Update total stats
        total = self.regime_performance[regime]['total']
        total['count'] += 1
        if is_win:
            total['win_count'] += 1
        total['avg_profit_r'] = ((total['avg_profit_r'] * (total['count'] - 1)) + trade_result.profit_r) / total[
            'count']
        total['win_rate'] = total['win_count'] / total['count']

    def _update_timeframe_performance(self, trade_result: TradeResult) -> None:
        """Update performance metrics for timeframes"""
        if not trade_result.timeframe:
            return

        timeframe = trade_result.timeframe
        direction = trade_result.direction

        # Create structure if it doesn't exist
        if timeframe not in self.timeframe_performance:
            self.timeframe_performance[timeframe] = {
                'long': {'count': 0, 'win_count': 0, 'avg_profit_r': 0.0, 'win_rate': 0.0},
                'short': {'count': 0, 'win_count': 0, 'avg_profit_r': 0.0, 'win_rate': 0.0},
                'total': {'count': 0, 'win_count': 0, 'avg_profit_r': 0.0, 'win_rate': 0.0}
            }

        # Update stats for specific direction
        perf = self.timeframe_performance[timeframe][direction]
        perf['count'] += 1
        is_win = trade_result.profit_r > 0
        if is_win:
            perf['win_count'] += 1

        # Update moving average profit
        perf['avg_profit_r'] = ((perf['avg_profit_r'] * (perf['count'] - 1)) + trade_result.profit_r) / perf[
            'count']
        perf['win_rate'] = perf['win_count'] / perf['count']

        # Update total stats
        total = self.timeframe_performance[timeframe]['total']
        total['count'] += 1
        if is_win:
            total['win_count'] += 1
        total['avg_profit_r'] = ((total['avg_profit_r'] * (total['count'] - 1)) + trade_result.profit_r) / total[
            'count']
        total['win_rate'] = total['win_count'] / total['count']

    def get_symbol_performance_factor(self, symbol: str, direction: str) -> float:
        """Calculate performance factor for a symbol in a specific direction"""
        if not self.enabled or symbol not in self.symbol_performance:
            return 1.0

        # Use cache for performance
        cache_key = f"symbol_{symbol}_{direction}_perf"
        if cache_key in self._performance_cache:
            cached_result, timestamp = self._performance_cache[cache_key]
            if time.time() - timestamp < self._cache_ttl_seconds:
                return cached_result

        try:
            perf = self.symbol_performance[symbol][direction]
            # Ensure we have minimum trades
            if perf['count'] < 3:
                result = 1.0
            else:
                # Combine win_rate and avg_profit_r
                win_rate_factor = perf['win_rate'] / 0.5  # Normalize relative to 50% win rate
                avg_profit_factor = (perf['avg_profit_r'] + 1.0) / 1.0  # Normalize relative to avg_profit_r = 0

                # Calculate final factor with limits
                result = min(1.5, max(0.5, (win_rate_factor * 0.6 + avg_profit_factor * 0.4)))

            # Save to cache
            self._performance_cache[cache_key] = (result, time.time())
            return result

        except Exception as e:
            logger.error(f"Error calculating symbol performance factor: {e}", exc_info=True)
            return 1.0

    def get_pattern_performance_factors(self, patterns: List[str]) -> Dict[str, float]:
        """Calculate performance factors for a set of patterns"""
        if not self.enabled or not patterns:
            return {pattern: 1.0 for pattern in patterns}

        result = {}
        for pattern in patterns:
            cache_key = f"pattern_{pattern}_perf"
            if cache_key in self._performance_cache:
                cached_result, timestamp = self._performance_cache[cache_key]
                if time.time() - timestamp < self._cache_ttl_seconds:
                    result[pattern] = cached_result
                    continue

            try:
                if pattern not in self.pattern_performance or self.pattern_performance[pattern]['count'] < 5:
                    factor = 1.0
                else:
                    perf = self.pattern_performance[pattern]
                    win_rate_factor = perf['win_rate'] / 0.5  # Normalize relative to 50% win rate
                    avg_profit_factor = (perf['avg_profit_r'] + 1.0) / 1.0  # Normalize relative to avg_profit_r = 0

                    # Calculate final factor with limits
                    factor = min(1.5, max(0.5, (win_rate_factor * 0.7 + avg_profit_factor * 0.3)))

                # Save to cache
                self._performance_cache[cache_key] = (factor, time.time())
                result[pattern] = factor

            except Exception as e:
                logger.error(f"Error calculating pattern performance factor for {pattern}: {e}", exc_info=True)
                result[pattern] = 1.0

        return result

    def get_regime_performance_factor(self, regime: str, direction: str) -> float:
        """Calculate performance factor for a market regime in a specific direction"""
        if not self.enabled or not regime or regime not in self.regime_performance:
            return 1.0

        # Use cache for performance
        cache_key = f"regime_{regime}_{direction}_perf"
        if cache_key in self._performance_cache:
            cached_result, timestamp = self._performance_cache[cache_key]
            if time.time() - timestamp < self._cache_ttl_seconds:
                return cached_result

        try:
            perf = self.regime_performance[regime][direction]
            # Ensure we have minimum trades
            if perf['count'] < 5:
                result = 1.0
            else:
                # Combine win_rate and avg_profit_r
                win_rate_factor = perf['win_rate'] / 0.5  # Normalize relative to 50% win rate
                avg_profit_factor = (perf['avg_profit_r'] + 1.0) / 1.0  # Normalize relative to avg_profit_r = 0

                # Calculate final factor with limits
                result = min(1.3, max(0.7, (win_rate_factor * 0.5 + avg_profit_factor * 0.5)))

            # Save to cache
            self._performance_cache[cache_key] = (result, time.time())
            return result

        except Exception as e:
            logger.error(f"Error calculating regime performance factor: {e}", exc_info=True)
            return 1.0

    def get_timeframe_performance_factor(self, timeframe: str, direction: str) -> float:
        """Calculate performance factor for a timeframe in a specific direction"""
        if not self.enabled or not timeframe or timeframe not in self.timeframe_performance:
            return 1.0

        # Use cache for performance
        cache_key = f"timeframe_{timeframe}_{direction}_perf"
        if cache_key in self._performance_cache:
            cached_result, timestamp = self._performance_cache[cache_key]
            if time.time() - timestamp < self._cache_ttl_seconds:
                return cached_result

        try:
            perf = self.timeframe_performance[timeframe][direction]
            # Ensure we have minimum trades
            if perf['count'] < 5:
                result = 1.0
            else:
                # Combine win_rate and avg_profit_r
                win_rate_factor = perf['win_rate'] / 0.5  # Normalize relative to 50% win rate
                avg_profit_factor = (perf['avg_profit_r'] + 1.0) / 1.0  # Normalize relative to avg_profit_r = 0

                # Calculate final factor with limits
                result = min(1.2, max(0.8, (win_rate_factor * 0.6 + avg_profit_factor * 0.4)))

            # Save to cache
            self._performance_cache[cache_key] = (result, time.time())
            return result

        except Exception as e:
            logger.error(f"Error calculating timeframe performance factor: {e}", exc_info=True)
            return 1.0

    def get_adaptive_pattern_scores(self, pattern_scores: Dict[str, float]) -> Dict[str, float]:
        """Calculate adapted scores for patterns based on past performance"""
        if not self.enabled:
            return pattern_scores

        # Use cache for performance
        cache_key = "adaptive_pattern_scores"
        if cache_key in self._performance_cache:
            cached_result, timestamp = self._performance_cache[cache_key]
            if time.time() - timestamp < self._cache_ttl_seconds:
                return cached_result

        try:
            adjusted_scores = copy.deepcopy(pattern_scores)

            # Adjust scores based on performance
            for pattern, score in pattern_scores.items():
                if pattern in self.pattern_performance and self.pattern_performance[pattern]['count'] >= 5:
                    perf = self.pattern_performance[pattern]

                    # Calculate adjustment factor
                    performance_factor = self.get_pattern_performance_factors([pattern])[pattern]

                    # Apply gradual adjustment with learning rate
                    adjusted_score = score * (1.0 + (performance_factor - 1.0) * self.learning_rate)
                    adjusted_scores[pattern] = adjusted_score

            # Save to cache
            self._performance_cache[cache_key] = (adjusted_scores, time.time())
            return adjusted_scores

        except Exception as e:
            logger.error(f"Error calculating adaptive pattern scores: {e}", exc_info=True)
            return pattern_scores

    def get_adaptive_sl_percent(self, base_sl_percent: float, symbol: str, timeframe: str, direction: str) -> float:
        """Calculate adapted stop loss percentage based on past performance"""
        if not self.enabled:
            return base_sl_percent

        # Use cache for performance
        cache_key = f"adaptive_sl_{symbol}_{timeframe}_{direction}"
        if cache_key in self._performance_cache:
            cached_result, timestamp = self._performance_cache[cache_key]
            if time.time() - timestamp < self._cache_ttl_seconds:
                return cached_result

        try:
            # Filter relevant trades
            relevant_trades = [t for t in self.trade_history
                               if t.symbol == symbol and t.timeframe == timeframe and t.direction == direction
                               and t.exit_reason == 'sl']  # Only trades with SL exit

            if len(relevant_trades) < 3:
                # Not enough data
                return base_sl_percent

            # Analyze previous stop losses
            sl_distances = []
            for trade in relevant_trades:
                # Calculate percentage distance of SL from entry price
                if trade.direction == 'long':
                    sl_distance_percent = (trade.entry_price - trade.stop_loss) / trade.entry_price * 100
                else:
                    sl_distance_percent = (trade.stop_loss - trade.entry_price) / trade.entry_price * 100
                sl_distances.append(sl_distance_percent)

            # Calculate average SL distances
            avg_sl_distance = sum(sl_distances) / len(sl_distances)

            # Adjust SL with calculated average
            adjusted_sl_percent = base_sl_percent * (1.0 +
                                                     (avg_sl_distance / base_sl_percent - 1.0) * self.learning_rate)

            # Limit changes
            adjusted_sl_percent = max(base_sl_percent * 0.7, min(base_sl_percent * 1.3, adjusted_sl_percent))

            # Save to cache
            self._performance_cache[cache_key] = (adjusted_sl_percent, time.time())
            return adjusted_sl_percent

        except Exception as e:
            logger.error(f"Error calculating adaptive SL percent: {e}", exc_info=True)
            return base_sl_percent

# ===============================================
#      Correlation Manager
# ===============================================
class CorrelationManager:
    """Manages correlations between symbols for portfolio diversification and risk reduction"""

    def __init__(self, config: Dict[str, Any]):
        """Initialize with configuration"""
        self.config = config.get('correlation_management', {})
        self.enabled = self.config.get('enabled', True)
        self.data_file = self.config.get('data_file', 'correlation_data.json')
        self.correlation_threshold = self.config.get('correlation_threshold', 0.7)
        self.max_exposure_per_group = self.config.get('max_exposure_per_group', 3)
        self.update_interval = self.config.get('update_interval', 86400)  # 24 hours
        self.lookback_periods = self.config.get('lookback_periods', 100)
        self.correlation_groups: Dict[str, List[str]] = {}
        self.correlation_matrix: Dict[str, Dict[str, float]] = {}
        self.last_update_time = 0
        self.active_positions: Dict[str, Dict[str, Any]] = {}

        # Load existing data
        self._load_data()

        logger.info(f"CorrelationManager initialized. Enabled: {self.enabled}, "
                    f"Correlation threshold: {self.correlation_threshold}")

    def _load_data(self) -> None:
        """Load correlation data from file"""
        try:
            if not os.path.exists(self.data_file):
                logger.info(f"Correlation data file not found: {self.data_file}, starting with empty data.")
                return

            with open(self.data_file, 'r') as f:
                data = json.load(f)

            self.correlation_matrix = data.get('correlation_matrix', {})
            self.correlation_groups = data.get('correlation_groups', {})
            self.last_update_time = data.get('last_update_time', 0)

            logger.info(f"Loaded correlation data: {len(self.correlation_matrix)} symbols, "
                        f"{len(self.correlation_groups)} groups.")
        except Exception as e:
            logger.error(f"Error loading correlation data: {e}", exc_info=True)
            # Continue with empty data if loading fails

    def save_data(self) -> None:
        """Save correlation data to file"""
        if not self.enabled:
            return

        try:
            # Create directory if needed
            os.makedirs(os.path.dirname(os.path.abspath(self.data_file)), exist_ok=True)

            # Prepare data for saving
            data = {
                'correlation_matrix': self.correlation_matrix,
                'correlation_groups': self.correlation_groups,
                'last_update_time': self.last_update_time,
                'update_timestamp': datetime.now().isoformat()
            }

            # Save to file
            with open(self.data_file, 'w') as f:
                json.dump(data, f, indent=2)

            logger.info(f"Saved correlation data to {self.data_file}")
        except Exception as e:
            logger.error(f"Error saving correlation data: {e}", exc_info=True)

    def update_correlations(self, symbols_data: Dict[str, pd.DataFrame]) -> None:
        """Update correlation matrix between symbols"""
        if not self.enabled or len(symbols_data) < 2:
            return

        # Check if update is needed based on time
        current_time = time.time()
        if current_time - self.last_update_time < self.update_interval:
            logger.debug("Skipping correlation update, not enough time passed since last update.")
            return

        try:
            logger.info(f"Updating correlations for {len(symbols_data)} symbols...")

            # Extract closing prices
            symbol_prices = {}
            for symbol, df in symbols_data.items():
                if df is not None and len(df) >= self.lookback_periods:
                    symbol_prices[symbol] = df['close'].iloc[-self.lookback_periods:].values

            # Calculate correlation between all symbol pairs
            new_correlation_matrix = {}
            symbols = list(symbol_prices.keys())

            for i, symbol1 in enumerate(symbols):
                if symbol1 not in new_correlation_matrix:
                    new_correlation_matrix[symbol1] = {}

                prices1 = symbol_prices[symbol1]

                for j, symbol2 in enumerate(symbols[i:], i):
                    if symbol1 == symbol2:
                        new_correlation_matrix[symbol1][symbol2] = 1.0
                        continue

                    if symbol2 not in new_correlation_matrix:
                        new_correlation_matrix[symbol2] = {}

                    prices2 = symbol_prices[symbol2]

                    # Calculate correlation coefficient
                    try:
                        corr = np.corrcoef(prices1, prices2)[0, 1]
                        # Check for NaN
                        if np.isnan(corr):
                            corr = 0.0
                    except Exception:
                        corr = 0.0

                    # Store in matrix
                    new_correlation_matrix[symbol1][symbol2] = corr
                    new_correlation_matrix[symbol2][symbol1] = corr

            # Update main matrix
            self.correlation_matrix = new_correlation_matrix

            # Update correlation groups
            self._update_correlation_groups()

            # Update time
            self.last_update_time = current_time

            # Save data
            self.save_data()

            logger.info(
                f"Updated correlations for {len(new_correlation_matrix)} symbols with {len(self.correlation_groups)} groups.")
        except Exception as e:
            logger.error(f"Error updating correlations: {e}", exc_info=True)

    def _update_correlation_groups(self) -> None:
        """Update correlation groups based on correlation matrix"""
        try:
            # Reset groups
            self.correlation_groups = {}

            # Create data structure for simple clustering algorithm
            symbols = list(self.correlation_matrix.keys())
            if not symbols:
                return

            # Group symbols with high correlation
            group_id = 0
            ungrouped_symbols = set(symbols)

            while ungrouped_symbols:
                # Select a base symbol
                base_symbol = next(iter(ungrouped_symbols))
                current_group = [base_symbol]
                ungrouped_symbols.remove(base_symbol)

                # Find all symbols correlated with base symbol
                for symbol in list(ungrouped_symbols):
                    if base_symbol in self.correlation_matrix and symbol in self.correlation_matrix[base_symbol]:
                        corr = abs(self.correlation_matrix[base_symbol][symbol])
                        if corr >= self.correlation_threshold:
                            current_group.append(symbol)
                            ungrouped_symbols.remove(symbol)

                # Save group if it has more than one symbol
                if len(current_group) > 1:
                    self.correlation_groups[f"group_{group_id}"] = current_group
                    group_id += 1
        except Exception as e:
            logger.error(f"Error updating correlation groups: {e}", exc_info=True)

    def get_correlated_symbols(self, symbol: str, threshold: Optional[float] = None) -> List[Tuple[str, float]]:
        """Get list of symbols correlated with a specific symbol"""
        if not self.enabled or symbol not in self.correlation_matrix:
            return []

        try:
            corr_threshold = threshold if threshold is not None else self.correlation_threshold
            correlated = []

            for other_symbol, corr in self.correlation_matrix[symbol].items():
                if other_symbol != symbol and abs(corr) >= corr_threshold:
                    correlated.append((other_symbol, corr))

            # Sort by correlation magnitude (descending)
            return sorted(correlated, key=lambda x: abs(x[1]), reverse=True)
        except Exception as e:
            logger.error(f"Error getting correlated symbols for {symbol}: {e}", exc_info=True)
            return []

    def update_active_positions(self, positions: Dict[str, Dict[str, Any]]) -> None:
        """Update list of active positions"""
        if not self.enabled:
            return

        self.active_positions = positions

    def get_correlation_safety_factor(self, symbol: str, direction: str) -> float:
        """Calculate correlation safety factor for a symbol given active positions"""
        if not self.enabled or not self.active_positions:
            return 1.0

        try:
            # Find correlation group of symbol
            symbol_group = None
            for group_id, group_symbols in self.correlation_groups.items():
                if symbol in group_symbols:
                    symbol_group = group_id
                    break

            if not symbol_group:
                return 1.0  # Symbol is not in any correlation group

            # Check number of active positions in this group
            group_positions = 0
            for pos_symbol, pos_info in self.active_positions.items():
                # Check if position symbol is in correlation group
                if pos_symbol in self.correlation_groups.get(symbol_group, []):
                    # Check position direction
                    pos_direction = pos_info.get('direction', '')
                    # Positions with opposite direction are not dangerous from correlation perspective
                    if direction == pos_direction:
                        group_positions += 1

            # Calculate safety factor based on number of active positions in group
            if group_positions >= self.max_exposure_per_group:
                return 0.5  # Substantial score reduction to prevent concentration risk
            elif group_positions > 0:
                # Gradual reduction based on position count
                return 1.0 - (0.5 * group_positions / self.max_exposure_per_group)

            return 1.0  # No other active positions in this group

        except Exception as e:
            logger.error(f"Error calculating correlation safety factor for {symbol}: {e}", exc_info=True)
            return 1.0

# ===============================================
#      Emergency Circuit Breaker
# ===============================================
class EmergencyCircuitBreaker:
    """Emergency stop mechanism to prevent consecutive losses in abnormal market conditions"""

    def __init__(self, config: Dict[str, Any]):
        """Initialize with configuration"""
        self.config = config.get('circuit_breaker', {})
        self.enabled = self.config.get('enabled', True)
        self.max_consecutive_losses = self.config.get('max_consecutive_losses', 3)
        self.max_daily_losses_r = self.config.get('max_daily_losses_r', 5.0)
        self.cool_down_period_minutes = self.config.get('cool_down_period_minutes', 60)
        self.reset_period_hours = self.config.get('reset_period_hours', 24)

        # Internal variables
        self.consecutive_losses = 0
        self.daily_loss_r = 0.0
        self.triggered = False
        self.trigger_time = None
        self.last_reset_time = datetime.now(timezone.utc)
        self.trade_log: List[Dict[str, Any]] = []

        logger.info(f"EmergencyCircuitBreaker initialized. Enabled: {self.enabled}, "
                    f"Max consecutive losses: {self.max_consecutive_losses}, "
                    f"Max daily loss R: {self.max_daily_losses_r}")

    def add_trade_result(self, trade_result: TradeResult) -> None:
        """Register a trade result and check for emergency stop conditions"""
        if not self.enabled:
            return

        try:
            # Reset daily stats if needed
            current_time = datetime.now(timezone.utc)
            hours_since_reset = (current_time - self.last_reset_time).total_seconds() / 3600
            if hours_since_reset >= self.reset_period_hours:
                self._reset_daily_stats()

            # Register new trade
            trade_info = {
                'time': current_time,
                'symbol': trade_result.symbol,
                'direction': trade_result.direction,
                'profit_r': trade_result.profit_r,
                'exit_reason': trade_result.exit_reason
            }
            self.trade_log.append(trade_info)

            # Update stats
            if trade_result.profit_r < 0:
                self.consecutive_losses += 1
                self.daily_loss_r -= trade_result.profit_r  # Negative * negative = positive
            else:
                self.consecutive_losses = 0  # Reset consecutive loss counter

            # Check stop conditions
            if self.consecutive_losses >= self.max_consecutive_losses:
                self._trigger_circuit_breaker(f"Hit {self.consecutive_losses} consecutive losses")
            elif self.daily_loss_r >= self.max_daily_losses_r:
                self._trigger_circuit_breaker(
                    f"Daily loss of {self.daily_loss_r:.2f}R exceeded limit of {self.max_daily_losses_r}R")

            # Log status
            logger.debug(f"Circuit breaker status: consecutive_losses={self.consecutive_losses}, "
                         f"daily_loss_r={self.daily_loss_r:.2f}, triggered={self.triggered}")
        except Exception as e:
            logger.error(f"Error processing trade result in circuit breaker: {e}", exc_info=True)

    def _trigger_circuit_breaker(self, reason: str) -> None:
        """Activate emergency stop"""
        if self.triggered:
            return  # Already triggered

        self.triggered = True
        self.trigger_time = datetime.now(timezone.utc)

        logger.warning(f"CIRCUIT BREAKER TRIGGERED: {reason}. "
                       f"Trading paused for {self.cool_down_period_minutes} minutes.")

    def _reset_daily_stats(self) -> None:
        """Reset daily statistics"""
        self.daily_loss_r = 0.0
        self.last_reset_time = datetime.now(timezone.utc)
        # Clean up old trades
        self.trade_log = [t for t in self.trade_log if
                          (datetime.now(timezone.utc) - t['time']).total_seconds() < self.reset_period_hours * 3600]

    def check_if_active(self) -> Tuple[bool, Optional[str]]:
        """Check if circuit breaker is active and remaining time"""
        if not self.enabled:
            return False, None

        if not self.triggered:
            return False, None

        # Check cool down period end
        current_time = datetime.now(timezone.utc)
        if self.trigger_time:
            minutes_since_trigger = (current_time - self.trigger_time).total_seconds() / 60
            if minutes_since_trigger >= self.cool_down_period_minutes:
                # Reset circuit breaker
                self.triggered = False
                self.trigger_time = None
                self.consecutive_losses = 0  # Reset consecutive loss counter

                logger.info("Circuit breaker cool-down period complete. Trading resumed.")
                return False, None
            else:
                # Still active
                minutes_remaining = self.cool_down_period_minutes - minutes_since_trigger
                return True, f"Cooling down, {int(minutes_remaining)} minutes remaining"

        return self.triggered, None

    def is_market_volatile(self, symbols_data: Dict[str, DataFrame]) -> bool:
        """Detect abnormal market volatility based on ATR"""
        if not self.enabled or not symbols_data:
            return False

        try:
            volatility_scores = []

            for symbol, df in symbols_data.items():
                if df is None or len(df) < 30:
                    continue

                # Calculate ATR
                atr = talib.ATR(
                    df['high'].values.astype(np.float64),
                    df['low'].values.astype(np.float64),
                    df['close'].values.astype(np.float64),
                    timeperiod=14
                )

                # Calculate ATR% relative to price
                close_prices = df['close'].values[-len(atr):]
                atr_percent = np.where(~np.isnan(atr) & (close_prices > 0),
                                       (atr / close_prices) * 100,
                                       np.nan)

                # Calculate average and standard deviation of recent valid ATR%s
                valid_atr_percent = atr_percent[~np.isnan(atr_percent)]
                if len(valid_atr_percent) < 5:
                    continue

                # Compare last 5 values to previous 20 values
                recent_atr_percent = valid_atr_percent[-5:].mean()
                past_atr_percent = valid_atr_percent[-25:-5].mean() if len(
                    valid_atr_percent) >= 25 else valid_atr_percent[:-5].mean()

                # Volatility change ratio
                if past_atr_percent > 0:
                    volatility_change = recent_atr_percent / past_atr_percent
                    volatility_scores.append(volatility_change)

            # Average volatility change ratio across different symbols
            if volatility_scores:
                avg_volatility_change = sum(volatility_scores) / len(volatility_scores)
                # Threshold for significant volatility increase
                return avg_volatility_change > 1.5

            return False
        except Exception as e:
            logger.error(f"Error checking market volatility: {e}", exc_info=True)
            return False

    def get_market_anomaly_score(self, symbols_data: Dict[str, DataFrame]) -> float:
        """Calculate market anomaly score based on multiple indicators"""
        if not self.enabled or not symbols_data:
            return 0.0

        try:
            anomaly_factors = []

            for symbol, df in symbols_data.items():
                if df is None or len(df) < 50:
                    continue

                # Volume analysis
                if 'volume' in df.columns:
                    # 20-period moving average volume
                    vol_ma = df['volume'].rolling(window=20).mean()
                    if not vol_ma.isna().all():
                        last_valid_idx = vol_ma.last_valid_index()
                        if last_valid_idx is not None:
                            last_vol = df.loc[last_valid_idx, 'volume']
                            last_vol_ma = vol_ma[last_valid_idx]
                            if last_vol_ma > 0:
                                vol_ratio = last_vol / last_vol_ma
                                if vol_ratio > 3:  # Abnormal volume
                                    anomaly_factors.append(min(1.0, (vol_ratio - 3) / 7))

                # Price change analysis
                if len(df) >= 2:
                    last_close = df['close'].iloc[-1]
                    prev_close = df['close'].iloc[-2]
                    if prev_close > 0:
                        price_change_pct = abs((last_close - prev_close) / prev_close) * 100
                        if price_change_pct > 3:  # Abnormal price change
                            anomaly_factors.append(min(1.0, (price_change_pct - 3) / 7))

                # High-Low range analysis
                if len(df) >= 1:
                    last_high = df['high'].iloc[-1]
                    last_low = df['low'].iloc[-1]
                    if last_low > 0:
                        hl_ratio = (last_high - last_low) / last_low * 100
                        typical_hl = df['high'].sub(df['low']).div(df['low']).mul(100).rolling(window=20).mean()
                        last_typical_hl = typical_hl.iloc[-1] if not typical_hl.isna().all() else 1.0
                        if last_typical_hl > 0 and hl_ratio > last_typical_hl * 2:
                            anomaly_factors.append(min(1.0, (hl_ratio / last_typical_hl - 2) / 3))

            # Calculate final score
            if anomaly_factors:
                return sum(anomaly_factors) / len(anomaly_factors)

            return 0.0
        except Exception as e:
            logger.error(f"Error calculating market anomaly score: {e}", exc_info=True)
            return 0.0

# ===============================================
#      Main Signal Generator Class
# ===============================================
class SignalGenerator:
    """Multi-timeframe trading signal generator with adaptive learning, correlation management,
    and emergency circuit breaker. Optimized for production use."""

    def __init__(self, config: Dict[str, Any]):
        """Initialize with configuration."""
        self.config = config  # Store main config
        self.signal_config = config.get('signal_generation', {})
        self.signal_processing = config.get('signal_processing', {})
        self.risk_config = config.get('risk_management', {})
        self.core_config = config.get('core', {})

        # Notification settings
        notification = config.get('notification', {})
        self.events = notification.get('events', {})
        self.signal_generated = self.events.get('signal_generated')

        # Timeframes and weights
        self.timeframes = self.signal_config.get('timeframes', ['5m', '15m', '1h', '4h'])
        self.timeframe_weights = self.signal_config.get('timeframe_weights', {
            '5m': 0.7, '15m': 0.85, '1h': 1.0, '4h': 1.2
        })

        # Base thresholds and RR
        self.base_minimum_signal_score = self.signal_config.get('minimum_signal_score', 180.0)
        self.base_min_risk_reward_ratio = self.risk_config.get('min_risk_reward_ratio', 1.8)
        self.base_preferred_risk_reward_ratio = self.risk_config.get('preferred_risk_reward_ratio', 2.5)
        self.base_default_sl_percent = self.risk_config.get('default_stop_loss_percent', 1.5)
        self.notification_config = self.signal_processing.get('notification', {})
        self.min_score_to_notify = self.notification_config.get('min_score_to_notify')

        # Analysis parameters
        self.pattern_scores = self.signal_config.get('pattern_scores', {})
        self.volume_multiplier_threshold = self.signal_config.get('volume_multiplier_threshold', 1.3)
        self.divergence_sensitivity = self.signal_config.get('divergence_sensitivity', 0.75)
        self.peak_detection_settings = {
            'order': self.signal_config.get('peak_detection_order', 3),
            'distance': self.signal_config.get('peak_detection_distance', 5),
            'prominence_factor': self.signal_config.get('peak_detection_prominence_factor', 0.1)
        }

        # MACD analysis settings
        self.macd_peak_detection_settings = {
            'smooth_kernel': 5,
            'distance': 5,
            'prominence_factor': 0.1
        }
        self.macd_trendline_period = 80
        self.macd_cross_period = 20
        self.macd_hist_period = 60

        # HTF structure settings
        self.htf_config = self.signal_config.get('structure_confirmation', {})
        self.htf_enabled = self.htf_config.get('enabled', True)
        self.htf_timeframe_method = self.htf_config.get('timeframe_method', 'next_higher')
        self.htf_fixed_tf1 = self.htf_config.get('fixed_first_timeframe', '1h')
        self.htf_fixed_tf2 = self.htf_config.get('fixed_second_timeframe', '4h')
        self.htf_sr_lookback = self.htf_config.get('support_resistance_lookback', 50)
        self.htf_sr_atr_multiplier = self.htf_config.get('support_resistance_atr_multiplier', 1.0)
        self.htf_trend_indicator = self.htf_config.get('trend_indicator', 'ema')
        self.htf_score_config = {
            'base': self.htf_config.get('base_score', 1.0),
            'confirm_bonus': self.htf_config.get('confirmation_bonus', 0.2),
            'trend_bonus_mult': self.htf_config.get('trend_bonus_multiplier', 1.5),
            'contradict_penalty': self.htf_config.get('contradiction_penalty', 0.3),
            'trend_penalty_mult': self.htf_config.get('trend_penalty_multiplier', 1.5),
            'min_score': self.htf_config.get('min_score', 0.5),
            'max_score': self.htf_config.get('max_score', 1.5),
        }

        # Volatility filter settings
        self.vol_config = self.signal_config.get('volatility_filter', {})
        self.vol_enabled = self.vol_config.get('enabled', True)
        self.vol_atr_period = self.vol_config.get('atr_period', 14)
        self.vol_atr_ma_period = self.vol_config.get('atr_ma_period', 30)
        self.vol_high_thresh = self.vol_config.get('high_volatility_threshold', 1.3)
        self.vol_low_thresh = self.vol_config.get('low_volatility_threshold', 0.7)
        self.vol_extreme_thresh = self.vol_config.get('extreme_volatility_threshold', 1.8)
        self.vol_scores = self.vol_config.get('scores', {})
        self.vol_reject_extreme = self.vol_config.get('reject_on_extreme_volatility', True)

        # Harmonic pattern settings
        self.harmonic_config = self.signal_config.get('harmonic_patterns', {})
        self.harmonic_enabled = self.harmonic_config.get('enabled', True)
        self.harmonic_lookback = self.harmonic_config.get('lookback', 100)
        self.harmonic_tolerance = self.harmonic_config.get('tolerance', 0.03)
        self.harmonic_min_quality = self.harmonic_config.get('min_quality', 0.7)

        # Price channel settings
        self.channel_config = self.signal_config.get('price_channels', {})
        self.channel_enabled = self.channel_config.get('enabled', True)
        self.channel_lookback = self.channel_config.get('lookback', 100)
        self.channel_min_touches = self.channel_config.get('min_touches', 3)
        self.channel_quality_threshold = self.channel_config.get('quality_threshold', 0.7)

        # Cyclical pattern settings
        self.cycle_config = self.signal_config.get('cyclical_patterns', {})
        self.cycle_enabled = self.cycle_config.get('enabled', True)
        self.cycle_lookback = self.cycle_config.get('lookback', 200)
        self.cycle_min_cycles = self.cycle_config.get('min_cycles', 2)
        self.cycle_fourier_periods = self.cycle_config.get('fourier_periods', [5, 10, 20, 40, 60])

        # Initialize advanced systems
        self.regime_detector = MarketRegimeDetector(config)
        self.adaptive_learning = AdaptiveLearningSystem(config)
        self.correlation_manager = CorrelationManager(config)
        self.circuit_breaker = EmergencyCircuitBreaker(config)

        # Parallel execution management
        max_workers = self.core_config.get('max_workers', 8)
        # If max_workers is zero or negative, use ThreadPoolExecutor default
        self.executor = ThreadPoolExecutor(max_workers=max_workers if max_workers > 0 else None)

        # Cache for repetitive calculations
        self._analysis_cache: Dict[str, Dict[str, Any]] = {}
        self._cache_ttl_seconds = 60  # 1 minute
        self._indicator_cache = {}

        # Automatically adapt pattern scores through adaptive learning
        if self.adaptive_learning.enabled:
            self.pattern_scores = self.adaptive_learning.get_adaptive_pattern_scores(self.pattern_scores)

        logger.info(
            f"SignalGenerator initialized. Base Min Score: {self.base_minimum_signal_score}, Base Min RR: {self.base_min_risk_reward_ratio}")
        logger.info(f"Using ThreadPoolExecutor with max_workers={max_workers if max_workers > 0 else 'default'}")
        logger.info(
            f"Extended features enabled: HTF: {self.htf_enabled}, Volatility: {self.vol_enabled}, Harmonic: {self.harmonic_enabled}, Channels: {self.channel_enabled}, Cycles: {self.cycle_enabled}")

    def shutdown(self):
        """Close Executor and save data"""
        logger.info("Shutting down SignalGenerator...")
        try:
            # Save adaptive learning data
            if hasattr(self, 'adaptive_learning'):
                self.adaptive_learning.save_data()

            # Save correlation data
            if hasattr(self, 'correlation_manager'):
                self.correlation_manager.save_data()

            # Close executor
            import sys
            if sys.version_info >= (3, 9):
                # In Python 3.9+ we can use cancel_futures
                self.executor.shutdown(wait=False, cancel_futures=True)
            else:
                # In older versions
                self.executor.shutdown(wait=False)
        except Exception as e:
            logger.error(f"Error shutting down: {e}")
        logger.info("SignalGenerator shut down.")

    # --- Basic Analysis Functions with Optimizations ---

    def _cache_key(self, symbol: str, timeframe: str, indicator: str, params: tuple) -> str:
        """Generate cache key for indicators"""
        return f"{symbol}_{timeframe}_{indicator}_{params}"

    def _get_cached_indicator(self, key: str) -> Optional[np.ndarray]:
        """Get indicator from cache"""
        return self._indicator_cache.get(key)

    def _cache_indicator(self, key: str, data: np.ndarray) -> None:
        """Cache indicator data"""
        self._indicator_cache[key] = data

    def find_peaks_and_valleys(self, data: np.ndarray, order: int = 3, distance: int = 5,
                               prominence_factor: float = 0.1, window_size: Optional[int] = None) -> Tuple[
        np.ndarray, np.ndarray]:
        """Identify peaks and valleys in data."""
        if data is None or len(data) < (max(order, distance) * 2 + 1):
            return np.array([], dtype=int), np.array([], dtype=int)
        try:
            if window_size and window_size < len(data):
                data = data[-window_size:]

            valid_mask = np.isfinite(data)
            if not np.any(valid_mask):
                return np.array([], dtype=int), np.array([], dtype=int)

            indices = np.arange(len(data))
            valid_indices = indices[valid_mask]
            valid_data = data[valid_mask]

            if len(valid_data) < (max(order, distance) * 2 + 1):
                return np.array([], dtype=int), np.array([], dtype=int)

            prominence = np.std(valid_data) * prominence_factor if np.std(valid_data) > 1e-6 else None
            if prominence is not None and prominence < 1e-6:
                prominence = None

            peaks_rel, peaks_props = sig_processing.find_peaks(valid_data, distance=distance, prominence=prominence,
                                                               width=order, rel_height=0.5, wlen=distance * 3)
            valleys_rel, valleys_props = sig_processing.find_peaks(-valid_data, distance=distance,
                                                                   prominence=prominence, width=order,
                                                                   rel_height=0.5, wlen=distance * 3)

            if len(peaks_rel) > 0 and 'prominences' in peaks_props:
                quality_threshold = np.median(peaks_props['prominences']) * 0.5 if len(
                    peaks_props['prominences']) > 0 else 0
                if quality_threshold > 0:
                    quality_peaks = peaks_rel[peaks_props['prominences'] >= quality_threshold]
                    peaks_rel = quality_peaks

            if len(valleys_rel) > 0 and 'prominences' in valleys_props:
                quality_threshold = np.median(valleys_props['prominences']) * 0.5 if len(
                    valleys_props['prominences']) > 0 else 0
                if quality_threshold > 0:
                    quality_valleys = valleys_rel[valleys_props['prominences'] >= quality_threshold]
                    valleys_rel = quality_valleys

            peaks = valid_indices[peaks_rel]
            valleys = valid_indices[valleys_rel]

            return peaks, valleys
        except Exception as e:
            logger.error(f"Error finding peaks/valleys: {e}", exc_info=False)
            return np.array([], dtype=int), np.array([], dtype=int)

    def analyze_volume_trend(self, df: pd.DataFrame, window: int = 20) -> Dict[str, Any]:
        """Analyze volume trend"""
        results = {'status': 'ok', 'current_ratio': 1.0, 'trend': 'neutral', 'pattern': 'normal',
                   'is_confirmed_by_volume': False}
        if 'volume' not in df.columns or len(df) < window + 1:
            results['status'] = 'insufficient_data'
            return results
        try:
            vol_series = df['volume'].replace([np.inf, -np.inf], np.nan).fillna(0)
            if use_bottleneck:
                vol_sma = bn.move_mean(vol_series.values, window=window, min_count=window)
            else:
                vol_sma = vol_series.rolling(window=window, min_periods=window).mean().values

            valid_indices = ~np.isnan(vol_sma) & (vol_sma > 1e-9)
            vol_ratio = np.full_like(vol_series.values, np.nan, dtype=float)
            vol_ratio[valid_indices] = vol_series.values[valid_indices] / vol_sma[valid_indices]

            last_valid_idx = -1
            while last_valid_idx >= -len(vol_ratio) and np.isnan(vol_ratio[last_valid_idx]):
                last_valid_idx -= 1
            if abs(last_valid_idx) > len(vol_ratio):
                results['status'] = 'calculation_error'
                return results

            current_ratio = vol_ratio[last_valid_idx]
            results['current_ratio'] = round(current_ratio, 3)
            results['is_confirmed_by_volume'] = current_ratio > self.volume_multiplier_threshold

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

            if len(vol_sma) >= 10:
                vol_sma_slope = (vol_sma[-1] - vol_sma[-10]) / vol_sma[-10] if vol_sma[-10] > 0 else 0
                results['volume_ma_trend'] = ('increasing' if vol_sma_slope > 0.05 else
                                              'decreasing' if vol_sma_slope < -0.05 else 'flat')
                results['volume_ma_slope'] = round(vol_sma_slope, 3)

            return results
        except Exception as e:
            logger.error(f"Error analyzing volume trend: {e}", exc_info=False)
            results['status'] = 'error'
            results['message'] = str(e)
            return results

    def detect_trend(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Detect trend using moving averages."""
        results = {'status': 'error', 'trend': 'neutral', 'strength': 0, 'method': 'unknown', 'details': {}}
        required_len = 100 + 5
        if df is None or len(df) < required_len:
            results['status'] = 'insufficient_data'
            results['method'] = 'insufficient_data'
            return results
        try:
            # Get cached EMAs or calculate them
            symbol = df.attrs.get('symbol', 'unknown')
            timeframe = df.attrs.get('timeframe', 'unknown')
            close_prices = df['close'].values.astype(np.float64)

            ema20_key = self._cache_key(symbol, timeframe, 'EMA', (20,))
            ema50_key = self._cache_key(symbol, timeframe, 'EMA', (50,))
            ema100_key = self._cache_key(symbol, timeframe, 'EMA', (100,))

            ema20 = self._get_cached_indicator(ema20_key)
            if ema20 is None:
                ema20 = talib.EMA(close_prices, timeperiod=20)
                self._cache_indicator(ema20_key, ema20)

            ema50 = self._get_cached_indicator(ema50_key)
            if ema50 is None:
                ema50 = talib.EMA(close_prices, timeperiod=50)
                self._cache_indicator(ema50_key, ema50)

            ema100 = self._get_cached_indicator(ema100_key)
            if ema100 is None:
                ema100 = talib.EMA(close_prices, timeperiod=100)
                self._cache_indicator(ema100_key, ema100)

            last_valid_idx = -1
            while last_valid_idx >= -len(df) and (
                    np.isnan(ema20[last_valid_idx]) or np.isnan(ema50[last_valid_idx]) or np.isnan(
                ema100[last_valid_idx])):
                last_valid_idx -= 1

            if abs(last_valid_idx) > len(df):
                results['status'] = 'calculation_error'
                results['method'] = 'calculation_error'
                return results

            current_close = close_prices[last_valid_idx]
            current_ema20 = ema20[last_valid_idx]
            current_ema50 = ema50[last_valid_idx]
            current_ema100 = ema100[last_valid_idx]
            ema20_slope = ema20[last_valid_idx] - ema20[last_valid_idx - 5] if last_valid_idx >= 5 else 0
            ema50_slope = ema50[last_valid_idx] - ema50[last_valid_idx - 5] if last_valid_idx >= 5 else 0

            ema_arrangement = 'unknown'
            if current_ema20 > current_ema50 > current_ema100:
                ema_arrangement = 'bullish_aligned'
            elif current_ema20 < current_ema50 < current_ema100:
                ema_arrangement = 'bearish_aligned'
            elif current_ema20 > current_ema50 and current_ema50 < current_ema100:
                ema_arrangement = 'potential_bullish_reversal'
            elif current_ema20 < current_ema50 and current_ema50 > current_ema100:
                ema_arrangement = 'potential_bearish_reversal'

            trend = 'neutral'
            strength = 0
            trend_phase = 'undefined'

            if current_close > current_ema20 > current_ema50 > current_ema100 and ema20_slope > 0 and ema50_slope > 0:
                trend = 'bullish'
                strength = 3
                trend_phase = 'mature' if ema_arrangement == 'bullish_aligned' else 'developing'
            elif current_close > current_ema20 > current_ema50 and ema20_slope > 0:
                trend = 'bullish'
                strength = 2
                trend_phase = 'developing'
            elif current_close > current_ema20 and ema20_slope > 0:
                trend = 'bullish'
                strength = 1
                trend_phase = 'early'
            elif current_close < current_ema20 < current_ema50 < current_ema100 and ema20_slope < 0 and ema50_slope < 0:
                trend = 'bearish'
                strength = -3
                trend_phase = 'mature' if ema_arrangement == 'bearish_aligned' else 'developing'
            elif current_close < current_ema20 < current_ema50 and ema20_slope < 0:
                trend = 'bearish'
                strength = -2
                trend_phase = 'developing'
            elif current_close < current_ema20 and ema20_slope < 0:
                trend = 'bearish'
                strength = -1
                trend_phase = 'early'
            elif current_close < current_ema50 and current_ema20 > current_ema50 and ema50_slope > 0:
                trend = 'bullish_pullback'
                strength = 1
                trend_phase = 'pullback'
            elif current_close > current_ema50 and current_ema20 < current_ema50 and ema50_slope < 0:
                trend = 'bearish_pullback'
                strength = -1
                trend_phase = 'pullback'

            results.update({
                'trend': trend,
                'strength': strength,
                'method': 'moving_averages',
                'phase': trend_phase,
                'details': {
                    'close': round(current_close, 5),
                    'ema20': round(current_ema20, 5),
                    'ema50': round(current_ema50, 5),
                    'ema100': round(current_ema100, 5),
                    'ema20_slope': round(ema20_slope, 5),
                    'ema_arrangement': ema_arrangement
                }
            })
            results['status'] = 'ok'
            return results
        except Exception as e:
            logger.error(f"Error detecting trend: {e}", exc_info=False)
            results['status'] = 'error'
            results['method'] = 'error'
            return results

    async def detect_candlestick_patterns(self, df: pd.DataFrame) -> List[Dict[str, Any]]:
        """Detect candlestick patterns using talib."""
        patterns_found = []
        if df is None or len(df) < 10:
            return patterns_found

        try:
            open_p = df['open'].values.astype(np.float64)
            high_p = df['high'].values.astype(np.float64)
            low_p = df['low'].values.astype(np.float64)
            close_p = df['close'].values.astype(np.float64)

            talib_patterns_to_check = [
                (talib.CDLHAMMER, 'hammer', 'bullish'),
                (talib.CDLINVERTEDHAMMER, 'inverted_hammer', 'bullish'),
                (talib.CDLENGULFING, 'engulfing', 'neutral'),
                (talib.CDLMORNINGSTAR, 'morning_star', 'bullish'),
                (talib.CDLEVENINGSTAR, 'evening_star', 'bearish'),
                (talib.CDLHARAMI, 'harami', 'neutral'),
                (talib.CDLDOJI, 'doji', 'neutral'),
                (talib.CDLSHOOTINGSTAR, 'shooting_star', 'bearish'),
                (talib.CDLMARUBOZU, 'marubozu', 'neutral'),
                (talib.CDLHANGINGMAN, 'hanging_man', 'bearish'),
                (talib.CDLDRAGONFLYDOJI, 'dragonfly_doji', 'bullish'),
                (talib.CDLGRAVESTONEDOJI, 'gravestone_doji', 'bearish')
            ]

            loop = asyncio.get_running_loop()
            futures = []

            # Submit function and arguments to thread pool
            for func_tuple in talib_patterns_to_check:
                # بررسی صحت فرمت تاپل
                if len(func_tuple) != 3:
                    logger.warning(f"Invalid pattern tuple format: {func_tuple}")
                    continue

                func, name, direction = func_tuple

                # Submit function and arguments to thread pool
                future = loop.run_in_executor(
                    self.executor,
                    func,
                    open_p, high_p, low_p, close_p
                )
                # ذخیره future همراه با متادیتا به عنوان یک تاپل با ۳ عنصر
                futures.append((future, name, direction))

            # Collect results
            results = await asyncio.gather(*(f[0] for f in futures), return_exceptions=True)

            # === تکمیل کد پردازش نتایج ===
            last_idx = len(df) - 1
            for i, result in enumerate(results):
                # اطمینان از اینکه i در محدوده مجاز است
                if i >= len(futures):
                    logger.error(f"Index out of range: {i} >= {len(futures)}")
                    continue

                # استخراج ایمن متادیتا
                try:
                    # بررسی اینکه futures[i] حداقل ۳ عنصر دارد
                    if len(futures[i]) < 3:
                        logger.error(f"Future tuple at index {i} has insufficient elements: {len(futures[i])}")
                        continue

                    future_obj, pattern_name, default_direction = futures[i]
                except Exception as e:
                    logger.error(f"Error unpacking future at index {i}: {e}")
                    continue

                # بررسی خطا در نتیجه
                if isinstance(result, Exception):
                    logger.error(f"Error processing pattern {pattern_name}: {result}")
                    continue

                # بررسی اینکه نتیجه معتبر است
                if len(result) != len(df):
                    logger.warning(f"Pattern {pattern_name} returned invalid result length: {len(result)} != {len(df)}")
                    continue

                # بررسی وجود الگو در کندل آخر
                pattern_value = result[last_idx]
                if pattern_value != 0:
                    # تعیین جهت الگو
                    pattern_direction = default_direction
                    if pattern_value > 0 and default_direction == 'neutral':
                        pattern_direction = 'bullish'
                    elif pattern_value < 0 and default_direction == 'neutral':
                        pattern_direction = 'bearish'

                    # محاسبه قدرت الگو
                    pattern_strength = min(1.0, abs(pattern_value) / 100)
                    if pattern_strength < 0.1:  # حداقل قدرت
                        pattern_strength = 0.7

                    # محاسبه امتیاز الگو
                    pattern_score = self.pattern_scores.get(pattern_name, 2.0) * pattern_strength

                    # اضافه کردن به لیست الگوهای یافت شده
                    patterns_found.append({
                        'type': pattern_name,
                        'direction': pattern_direction,
                        'index': last_idx,
                        'score': pattern_score,
                        'strength': pattern_strength
                    })

            # Add additional multi-candle pattern detection
            await self._detect_multi_candle_patterns(df, patterns_found)
            return patterns_found

        except Exception as e:
            logger.error(f"Error detecting candlestick patterns: {e}", exc_info=True)
            return []

    async def _detect_multi_candle_patterns(self, df: pd.DataFrame, patterns: List[Dict[str, Any]]) -> None:
        """Detect advanced multi-candle patterns."""
        if df is None or len(df) < 10:
            return

        try:
            head_and_shoulders = await self._detect_head_and_shoulders(df)
            if head_and_shoulders:
                patterns.extend(head_and_shoulders)

            triangles = await self._detect_triangle_patterns(df)
            if triangles:
                patterns.extend(triangles)

            flags = await self._detect_flag_patterns(df)
            if flags:
                patterns.extend(flags)

        except Exception as e:
            logger.error(f"Error detecting multi-candle patterns: {e}", exc_info=False)

    async def _detect_head_and_shoulders(self, df: pd.DataFrame) -> List[Dict[str, Any]]:
        """Detect head and shoulders pattern."""
        patterns = []
        if df is None or len(df) < 30:
            return patterns

        try:
            closes = df['close'].values
            highs = df['high'].values
            lows = df['low'].values

            peaks, _ = self.find_peaks_and_valleys(highs, distance=5, prominence_factor=0.05)
            _, valleys = self.find_peaks_and_valleys(lows, distance=5, prominence_factor=0.05)

            if len(peaks) >= 3:
                for i in range(len(peaks) - 2):
                    left_shoulder_idx = peaks[i]
                    head_idx = peaks[i + 1]
                    right_shoulder_idx = peaks[i + 2]

                    left_shoulder_price = highs[left_shoulder_idx]
                    head_price = highs[head_idx]
                    right_shoulder_price = highs[right_shoulder_idx]

                    if head_price > left_shoulder_price and head_price > right_shoulder_price:
                        shoulder_diff_percent = abs(
                            right_shoulder_price - left_shoulder_price) / left_shoulder_price
                        if shoulder_diff_percent < 0.1:
                            left_time_gap = head_idx - left_shoulder_idx
                            right_time_gap = right_shoulder_idx - head_idx
                            time_gap_ratio = min(left_time_gap, right_time_gap) / max(left_time_gap, right_time_gap)

                            if time_gap_ratio > 0.6:
                                dips_between_left_head = [v for v in valleys if
                                                          v > left_shoulder_idx and v < head_idx]
                                dips_between_head_right = [v for v in valleys if
                                                           v > head_idx and v < right_shoulder_idx]

                                if dips_between_left_head and dips_between_head_right:
                                    left_dip_idx = dips_between_left_head[-1]
                                    right_dip_idx = dips_between_head_right[0]

                                    left_dip_price = lows[left_dip_idx]
                                    right_dip_price = lows[right_dip_idx]

                                    neckline_diff_percent = abs(right_dip_price - left_dip_price) / left_dip_price
                                    if neckline_diff_percent < 0.05:
                                        neckline_price = (left_dip_price + right_dip_price) / 2
                                        last_price = closes[-1]
                                        last_idx = len(df) - 1
                                        breakout_confirmed = last_price < neckline_price and last_idx > right_shoulder_idx
                                        pattern_height = head_price - neckline_price
                                        price_target = neckline_price - pattern_height
                                        pattern_quality = (1.0 - shoulder_diff_percent) * time_gap_ratio * (
                                                1.0 - neckline_diff_percent)

                                        pattern = {
                                            'type': 'head_and_shoulders',
                                            'direction': 'bearish',
                                            'index': right_shoulder_idx,
                                            'breakout_confirmed': breakout_confirmed,
                                            'neckline_price': neckline_price,
                                            'price_target': price_target,
                                            'pattern_quality': round(pattern_quality, 2),
                                            'score': self.pattern_scores.get('head_and_shoulders',
                                                                             4.0) * pattern_quality,
                                            'points': {
                                                'left_shoulder': left_shoulder_idx,
                                                'head': head_idx,
                                                'right_shoulder': right_shoulder_idx,
                                                'left_dip': left_dip_idx,
                                                'right_dip': right_dip_idx
                                            }
                                        }
                                        patterns.append(pattern)

            # Inverse head & shoulders (similar logic with inversed price references)
            if len(valleys) >= 3:
                for i in range(len(valleys) - 2):
                    left_shoulder_idx = valleys[i]
                    head_idx = valleys[i + 1]
                    right_shoulder_idx = valleys[i + 2]

                    left_shoulder_price = lows[left_shoulder_idx]
                    head_price = lows[head_idx]
                    right_shoulder_price = lows[right_shoulder_idx]

                    if head_price < left_shoulder_price and head_price < right_shoulder_price:
                        shoulder_diff_percent = abs(
                            right_shoulder_price - left_shoulder_price) / left_shoulder_price
                        if shoulder_diff_percent < 0.1:
                            left_time_gap = head_idx - left_shoulder_idx
                            right_time_gap = right_shoulder_idx - head_idx
                            time_gap_ratio = min(left_time_gap, right_time_gap) / max(left_time_gap, right_time_gap)

                            if time_gap_ratio > 0.6:
                                peaks_between_left_head = [p for p in peaks if
                                                           p > left_shoulder_idx and p < head_idx]
                                peaks_between_head_right = [p for p in peaks if
                                                            p > head_idx and p < right_shoulder_idx]

                                if peaks_between_left_head and peaks_between_head_right:
                                    left_peak_idx = peaks_between_left_head[-1]
                                    right_peak_idx = peaks_between_head_right[0]

                                    left_peak_price = highs[left_peak_idx]
                                    right_peak_price = highs[right_peak_idx]

                                    neckline_diff_percent = abs(right_peak_price - left_peak_price) / left_peak_price
                                    if neckline_diff_percent < 0.05:
                                        neckline_price = (left_peak_price + right_peak_price) / 2
                                        last_price = closes[-1]
                                        last_idx = len(df) - 1
                                        breakout_confirmed = last_price > neckline_price and last_idx > right_shoulder_idx
                                        pattern_height = neckline_price - head_price
                                        price_target = neckline_price + pattern_height
                                        pattern_quality = (1.0 - shoulder_diff_percent) * time_gap_ratio * (
                                                1.0 - neckline_diff_percent)

                                        pattern = {
                                            'type': 'inverse_head_and_shoulders',
                                            'direction': 'bullish',
                                            'index': right_shoulder_idx,
                                            'breakout_confirmed': breakout_confirmed,
                                            'neckline_price': neckline_price,
                                            'price_target': price_target,
                                            'pattern_quality': round(pattern_quality, 2),
                                            'score': self.pattern_scores.get('inverse_head_and_shoulders',
                                                                             4.0) * pattern_quality,
                                            'points': {
                                                'left_shoulder': left_shoulder_idx,
                                                'head': head_idx,
                                                'right_shoulder': right_shoulder_idx,
                                                'left_peak': left_peak_idx,
                                                'right_peak': right_peak_idx
                                            }
                                        }
                                        patterns.append(pattern)

            return patterns
        except Exception as e:
            logger.error(f"Error detecting head and shoulders patterns: {e}", exc_info=False)
            return []

    async def _detect_triangle_patterns(self, df: pd.DataFrame) -> List[Dict[str, Any]]:
        """Detect triangle patterns (ascending, descending, symmetric)."""
        patterns = []
        if df is None or len(df) < 30:
            return patterns

        try:
            highs = df['high'].values
            lows = df['low'].values
            closes = df['close'].values

            peaks, valleys = self.find_peaks_and_valleys(
                closes,
                distance=self.peak_detection_settings['distance'],
                prominence_factor=self.peak_detection_settings['prominence_factor']
            )

            if len(peaks) < 2 or len(valleys) < 2:
                return patterns

            last_peaks = peaks[-3:] if len(peaks) >= 3 else peaks
            last_valleys = valleys[-3:] if len(valleys) >= 3 else valleys

            if len(last_peaks) < 2 or len(last_valleys) < 2:
                return patterns

            peak_x = last_peaks
            peak_y = highs[last_peaks]
            valley_x = last_valleys
            valley_y = lows[last_valleys]

            upper_slope, upper_intercept = np.polyfit(peak_x, peak_y, 1)
            lower_slope, lower_intercept = np.polyfit(valley_x, valley_y, 1)

            is_ascending = abs(upper_slope) < 0.001 and lower_slope > 0.001
            is_descending = upper_slope < -0.001 and abs(lower_slope) < 0.001
            is_symmetric = upper_slope < -0.001 and lower_slope > 0.001

            if abs(upper_slope - lower_slope) > 1e-6:
                convergence_x = (lower_intercept - upper_intercept) / (upper_slope - lower_slope)
                convergence_y = upper_slope * convergence_x + upper_intercept
            else:
                convergence_x = 0
                convergence_y = 0

            last_idx = len(df) - 1
            is_valid_convergence = convergence_x > last_idx and convergence_x < last_idx + 30

            if is_valid_convergence:
                current_upper = upper_slope * last_idx + upper_intercept
                current_lower = lower_slope * last_idx + lower_intercept
                pattern_width = current_upper - current_lower

                total_touches = len(last_peaks) + len(last_valleys)
                pattern_quality = min(1.0, total_touches / 6) * min(1.0,
                                                                    1.0 - pattern_width / (current_upper * 0.2))

                last_close = closes[-1]
                position_in_pattern = (last_close - current_lower) / pattern_width if pattern_width > 0 else 0.5

                if is_ascending:
                    triangle_type = 'ascending_triangle'
                    direction = 'bullish'
                elif is_descending:
                    triangle_type = 'descending_triangle'
                    direction = 'bearish'
                elif is_symmetric:
                    triangle_type = 'symmetric_triangle'
                    direction = 'bullish' if position_in_pattern > 0.5 else 'bearish'
                else:
                    return patterns

                pattern_height = max(highs[last_peaks]) - min(lows[last_valleys])
                price_target = last_close + pattern_height if direction == 'bullish' else last_close - pattern_height

                pattern = {
                    'type': triangle_type,
                    'direction': direction,
                    'index': last_idx,
                    'score': self.pattern_scores.get(triangle_type, 3.5) * pattern_quality,
                    'pattern_quality': round(pattern_quality, 2),
                    'convergence_point': {
                        'x': int(convergence_x),
                        'y': float(convergence_y)
                    },
                    'current_width': float(pattern_width),
                    'position_in_pattern': float(position_in_pattern),
                    'price_target': float(price_target),
                    'upper_line': {
                        'slope': float(upper_slope),
                        'intercept': float(upper_intercept)
                    },
                    'lower_line': {
                        'slope': float(lower_slope),
                        'intercept': float(lower_intercept)
                    }
                }
                patterns.append(pattern)

            return patterns
        except Exception as e:
            logger.error(f"Error detecting triangle patterns: {e}", exc_info=False)
            return []

    async def _detect_flag_patterns(self, df: pd.DataFrame) -> List[Dict[str, Any]]:
        """Detect flag and pennant patterns."""
        patterns = []
        if df is None or len(df) < 30:
            return patterns

        try:
            closes = df['close'].values
            highs = df['high'].values
            lows = df['low'].values
            volumes = df['volume'].values if 'volume' in df.columns else None

            trend_window = 20
            if len(closes) < trend_window + 5:
                return patterns

            pole_start = len(closes) - trend_window - 5
            pole_end = pole_start + 5

            pole_price_change = closes[pole_end] - closes[pole_start]
            pole_price_change_pct = pole_price_change / closes[pole_start] if closes[pole_start] > 0 else 0

            strong_volume = False
            if volumes is not None:
                avg_volume = np.mean(volumes[max(0, pole_start - 5):pole_start])
                pole_volume = np.mean(volumes[pole_start:pole_end])
                strong_volume = pole_volume > avg_volume * 1.5

            is_bullish_pole = pole_price_change_pct > 0.03
            is_bearish_pole = pole_price_change_pct < -0.03

            if not (is_bullish_pole or is_bearish_pole):
                return patterns

            flag_start = pole_end
            flag_end = len(closes) - 1

            if flag_end - flag_start < 5:
                return patterns

            flag_highs = highs[flag_start:flag_end + 1]
            flag_lows = lows[flag_start:flag_end + 1]

            x_indices = np.arange(flag_start, flag_end + 1)
            upper_slope, upper_intercept = np.polyfit(x_indices, flag_highs, 1)
            lower_slope, lower_intercept = np.polyfit(x_indices, flag_lows, 1)

            slopes_difference = abs(upper_slope - lower_slope)
            are_lines_parallel = slopes_difference < 0.0005

            if is_bullish_pole:
                is_valid_flag = (upper_slope < 0 and lower_slope < 0) or are_lines_parallel
                pattern_type = 'bull_flag'
                direction = 'bullish'
            elif is_bearish_pole:
                is_valid_flag = (upper_slope > 0 and lower_slope > 0) or are_lines_parallel
                pattern_type = 'bear_flag'
                direction = 'bearish'
            else:
                is_valid_flag = False

            if is_valid_flag:
                flag_quality = (1.0 if strong_volume else 0.7) * (1.0 - (slopes_difference / 0.001))
                pole_height = abs(pole_price_change)
                price_target = closes[-1] + pole_height if direction == 'bullish' else closes[-1] - pole_height

                pattern = {
                    'type': pattern_type,
                    'direction': direction,
                    'index': flag_end,
                    'score': self.pattern_scores.get(pattern_type, 3.0) * flag_quality,
                    'pattern_quality': round(flag_quality, 2),
                    'price_target': float(price_target),
                    'pole_start': pole_start,
                    'pole_end': pole_end,
                    'flag_start': flag_start,
                    'flag_end': flag_end,
                    'pole_height': float(pole_height),
                    'slope_difference': float(slopes_difference),
                    'strong_volume': strong_volume
                }
                patterns.append(pattern)

            return patterns
        except Exception as e:
            logger.error(f"Error detecting flag patterns: {e}", exc_info=False)
            return []

    def detect_support_resistance(self, df: pd.DataFrame, lookback: int = 50) -> Dict[str, Any]:
        """Detect key support and resistance levels."""
        results = {'status': 'ok', 'resistance_levels': [], 'support_levels': [], 'details': {}}
        if df is None or len(df) < lookback:
            results['status'] = 'insufficient_data'
            return results

        try:
            df_window = df.iloc[-lookback:]
            highs = df_window['high'].values.astype(np.float64)
            lows = df_window['low'].values.astype(np.float64)
            closes = df_window['close'].values.astype(np.float64)

            resistance_peaks, _ = self.find_peaks_and_valleys(highs, order=self.peak_detection_settings['order'],
                                                              distance=self.peak_detection_settings['distance'])
            _, support_valleys = self.find_peaks_and_valleys(lows, order=self.peak_detection_settings['order'],
                                                             distance=self.peak_detection_settings['distance'])

            resistance_levels_raw = highs[resistance_peaks] if len(resistance_peaks) > 0 else []
            support_levels_raw = lows[support_valleys] if len(support_valleys) > 0 else []

            def consolidate_levels(levels: np.ndarray, atr: float) -> List[Dict[str, Any]]:
                if len(levels) == 0 or atr <= 1e-9:
                    return [{'price': float(level), 'strength': 1.0} for level in np.unique(levels)]

                threshold = atr * 0.3
                if threshold <= 1e-9:
                    threshold = np.mean(levels) * 0.001 if np.mean(levels) > 0 else 1e-5

                sorted_levels = np.sort(levels)
                clusters = []
                cluster_indices = []

                if len(sorted_levels) > 0:
                    current_cluster = [sorted_levels[0]]
                    current_indices = [0]

                    for i in range(1, len(sorted_levels)):
                        if abs(sorted_levels[i] - np.mean(current_cluster)) <= threshold:
                            current_cluster.append(sorted_levels[i])
                            current_indices.append(i)
                        else:
                            cluster_mean = np.mean(current_cluster)
                            cluster_strength = min(1.0, len(current_cluster) / 3) * (
                                    1.0 - (np.std(current_cluster) / cluster_mean if cluster_mean > 0 else 0))
                            clusters.append({'price': float(cluster_mean), 'strength': float(cluster_strength)})
                            cluster_indices.append(current_indices)

                            current_cluster = [sorted_levels[i]]
                            current_indices = [i]

                    if current_cluster:
                        cluster_mean = np.mean(current_cluster)
                        cluster_strength = min(1.0, len(current_cluster) / 3) * (
                                1.0 - (np.std(current_cluster) / cluster_mean if cluster_mean > 0 else 0))
                        clusters.append({'price': float(cluster_mean), 'strength': float(cluster_strength)})
                        cluster_indices.append(current_indices)

                return sorted(clusters, key=lambda x: x['price'])

            atr_values = talib.ATR(highs, lows, closes, timeperiod=14)
            last_atr = atr_values[~np.isnan(atr_values)][-1] if not np.all(np.isnan(atr_values)) else (
                    highs[-1] - lows[-1])

            results['resistance_levels'] = consolidate_levels(resistance_levels_raw, last_atr)
            results['support_levels'] = consolidate_levels(support_levels_raw, last_atr)

            current_close = closes[-1]
            prev_close = closes[-2] if len(closes) > 1 else current_close
            prev_low = lows[-2] if len(lows) > 1 else current_close
            prev_high = highs[-2] if len(highs) > 1 else current_close

            broken_resistance = next((level for level in results['resistance_levels'] if
                                      current_close > level['price'] and prev_low < level['price']), None)
            broken_support = next((level for level in results['support_levels'] if
                                   current_close < level['price'] and prev_high > level['price']), None)

            resistance_candidates = [r for r in results['resistance_levels'] if r['price'] > current_close]
            support_candidates = [s for s in results['support_levels'] if s['price'] < current_close]

            nearest_resistance = min(resistance_candidates,
                                     key=lambda x: x['price'] - current_close) if resistance_candidates else None
            nearest_support = max(support_candidates,
                                  key=lambda x: current_close - x['price']) if support_candidates else None

            results['details'] = {
                'nearest_resistance': nearest_resistance,
                'nearest_support': nearest_support,
                'broken_resistance': broken_resistance,
                'broken_support': broken_support,
                'atr': round(last_atr, 8) if last_atr else None
            }

            results['resistance_zones'] = self._analyze_sr_zones(results['resistance_levels'], current_close,
                                                                 'resistance')
            results['support_zones'] = self._analyze_sr_zones(results['support_levels'], current_close, 'support')

            return results
        except Exception as e:
            logger.error(f"Error detecting support/resistance: {e}", exc_info=False)
            results['status'] = 'error'
            results['message'] = str(e)
            return results

    def _analyze_sr_zones(self, levels: List[Dict[str, Any]], current_price: float, zone_type: str) -> Dict[
        str, Any]:
        """Analyze support/resistance zones for clustering."""
        if not levels:
            return {'status': 'no_levels', 'zones': []}

        try:
            sorted_levels = sorted(levels, key=lambda x: abs(x['price'] - current_price))
            clusters = []
            current_cluster = [sorted_levels[0]]

            for i in range(1, len(sorted_levels)):
                if abs(sorted_levels[i]['price'] - sorted_levels[i - 1]['price']) / sorted_levels[i - 1][
                    'price'] < 0.01:
                    current_cluster.append(sorted_levels[i])
                else:
                    if len(current_cluster) >= 2:
                        clusters.append(current_cluster)
                    current_cluster = [sorted_levels[i]]

            if len(current_cluster) >= 2:
                clusters.append(current_cluster)

            zones = []
            for cluster in clusters:
                zone_min = min(level['price'] for level in cluster)
                zone_max = max(level['price'] for level in cluster)
                zone_strength = sum(level['strength'] for level in cluster) / len(cluster)
                zone_center = (zone_min + zone_max) / 2
                zone_width = zone_max - zone_min

                zone = {
                    'min': float(zone_min),
                    'max': float(zone_max),
                    'center': float(zone_center),
                    'width': float(zone_width),
                    'strength': float(zone_strength),
                    'levels_count': len(cluster),
                    'distance_to_price': float(abs(zone_center - current_price))
                }
                zones.append(zone)

            zones = sorted(zones, key=lambda x: x['distance_to_price'])
            return {'status': 'ok', 'zones': zones}

        except Exception as e:
            logger.error(f"Error analyzing {zone_type} zones: {e}", exc_info=False)
            return {'status': 'error', 'message': str(e), 'zones': []}

    def detect_harmonic_patterns(self, df: pd.DataFrame, lookback: int = 100, tolerance: float = 0.03) -> List[
        Dict[str, Any]]:
        """Detect harmonic price patterns (Gartley, Bat, Butterfly, Crab)."""
        patterns = []
        if not self.harmonic_enabled or df is None or len(df) < lookback:
            return patterns

        try:
            df_window = df.iloc[-lookback:].copy()

            peaks, valleys = self.find_peaks_and_valleys(
                df_window['close'].values,
                distance=self.peak_detection_settings['distance'],
                prominence_factor=self.peak_detection_settings['prominence_factor']
            )

            if len(peaks) + len(valleys) < 5:
                return patterns

            all_points = [(idx, 'peak', df_window['high'].iloc[idx]) for idx in peaks]
            all_points.extend([(idx, 'valley', df_window['low'].iloc[idx]) for idx in valleys])
            all_points.sort(key=lambda x: x[0])

            for i in range(len(all_points) - 4):
                X, A, B, C, D = all_points[i:i + 5]

                if not ((X[1] != A[1]) and (A[1] != B[1]) and (B[1] != C[1]) and (C[1] != D[1])):
                    continue

                x_price = X[2]
                a_price = A[2]
                b_price = B[2]
                c_price = C[2]
                d_price = D[2]

                xa = abs(x_price - a_price)
                ab = abs(a_price - b_price)
                bc = abs(b_price - c_price)
                cd = abs(c_price - d_price)

                if xa == 0 or ab == 0 or bc == 0:
                    continue

                ab_xa = ab / xa
                bc_ab = bc / ab
                cd_bc = cd / bc
                bd_ba = abs(d_price - b_price) / abs(a_price - b_price) if abs(a_price - b_price) > 0 else 0

                is_in_range = lambda val, target: abs(val - target) <= tolerance

                # Gartley Pattern
                if (is_in_range(ab_xa, 0.618) and
                        is_in_range(bc_ab, 0.382) and
                        is_in_range(cd_bc, 1.272) and
                        is_in_range(bd_ba, 0.786)):

                    pattern_type = "bullish_gartley" if A[1] == 'valley' else "bearish_gartley"
                    confidence = 1.0 - max(
                        abs(ab_xa - 0.618),
                        abs(bc_ab - 0.382),
                        abs(cd_bc - 1.272),
                        abs(bd_ba - 0.786)
                    ) / tolerance

                    if confidence >= self.harmonic_min_quality:
                        patterns.append({
                            'type': pattern_type,
                            'direction': 'bullish' if pattern_type.startswith('bullish') else 'bearish',
                            'points': {
                                'X': {'index': X[0], 'price': float(x_price)},
                                'A': {'index': A[0], 'price': float(a_price)},
                                'B': {'index': B[0], 'price': float(b_price)},
                                'C': {'index': C[0], 'price': float(c_price)},
                                'D': {'index': D[0], 'price': float(d_price)},
                            },
                            'ratios': {
                                'AB/XA': float(ab_xa),
                                'BC/AB': float(bc_ab),
                                'CD/BC': float(cd_bc),
                                'BD/BA': float(bd_ba)
                            },
                            'confidence': float(confidence),
                            'index': D[0],
                            'score': self.pattern_scores.get(pattern_type, 4.0) * confidence
                        })

                # Bat Pattern
                if (is_in_range(ab_xa, 0.382) and
                        is_in_range(bc_ab, 0.382) and
                        is_in_range(cd_bc, 1.618) and
                        is_in_range(bd_ba, 0.886)):

                    pattern_type = "bullish_bat" if A[1] == 'valley' else "bearish_bat"
                    confidence = 1.0 - max(
                        abs(ab_xa - 0.382),
                        abs(bc_ab - 0.382),
                        abs(cd_bc - 1.618),
                        abs(bd_ba - 0.886)
                    ) / tolerance

                    if confidence >= self.harmonic_min_quality:
                        patterns.append({
                            'type': pattern_type,
                            'direction': 'bullish' if pattern_type.startswith('bullish') else 'bearish',
                            'points': {
                                'X': {'index': X[0], 'price': float(x_price)},
                                'A': {'index': A[0], 'price': float(a_price)},
                                'B': {'index': B[0], 'price': float(b_price)},
                                'C': {'index': C[0], 'price': float(c_price)},
                                'D': {'index': D[0], 'price': float(d_price)},
                            },
                            'ratios': {
                                'AB/XA': float(ab_xa),
                                'BC/AB': float(bc_ab),
                                'CD/BC': float(cd_bc),
                                'BD/BA': float(bd_ba)
                            },
                            'confidence': float(confidence),
                            'index': D[0],
                            'score': self.pattern_scores.get(pattern_type, 4.0) * confidence
                        })

                # Butterfly Pattern
                if (is_in_range(ab_xa, 0.786) and
                        is_in_range(bc_ab, 0.382) and
                        is_in_range(cd_bc, 1.618) and
                        is_in_range(bd_ba, 1.27)):

                    pattern_type = "bullish_butterfly" if A[1] == 'valley' else "bearish_butterfly"
                    confidence = 1.0 - max(
                        abs(ab_xa - 0.786),
                        abs(bc_ab - 0.382),
                        abs(cd_bc - 1.618),
                        abs(bd_ba - 1.27)
                    ) / tolerance

                    if confidence >= self.harmonic_min_quality:
                        patterns.append({
                            'type': pattern_type,
                            'direction': 'bullish' if pattern_type.startswith('bullish') else 'bearish',
                            'points': {
                                'X': {'index': X[0], 'price': float(x_price)},
                                'A': {'index': A[0], 'price': float(a_price)},
                                'B': {'index': B[0], 'price': float(b_price)},
                                'C': {'index': C[0], 'price': float(c_price)},
                                'D': {'index': D[0], 'price': float(d_price)},
                            },
                            'ratios': {
                                'AB/XA': float(ab_xa),
                                'BC/AB': float(bc_ab),
                                'CD/BC': float(cd_bc),
                                'BD/BA': float(bd_ba)
                            },
                            'confidence': float(confidence),
                            'index': D[0],
                            'score': self.pattern_scores.get(pattern_type, 4.0) * confidence
                        })

                # Crab Pattern
                if (is_in_range(ab_xa, 0.382) and
                        is_in_range(bc_ab, 0.618) and
                        is_in_range(cd_bc, 3.618) and
                        is_in_range(bd_ba, 1.618)):

                    pattern_type = "bullish_crab" if A[1] == 'valley' else "bearish_crab"
                    confidence = 1.0 - max(
                        abs(ab_xa - 0.382),
                        abs(bc_ab - 0.618),
                        abs(cd_bc - 3.618),
                        abs(bd_ba - 1.618)
                    ) / tolerance

                    if confidence >= self.harmonic_min_quality:
                        patterns.append({
                            'type': pattern_type,
                            'direction': 'bullish' if pattern_type.startswith('bullish') else 'bearish',
                            'points': {
                                'X': {'index': X[0], 'price': float(x_price)},
                                'A': {'index': A[0], 'price': float(a_price)},
                                'B': {'index': B[0], 'price': float(b_price)},
                                'C': {'index': C[0], 'price': float(c_price)},
                                'D': {'index': D[0], 'price': float(d_price)},
                            },
                            'ratios': {
                                'AB/XA': float(ab_xa),
                                'BC/AB': float(bc_ab),
                                'CD/BC': float(cd_bc),
                                'BD/BA': float(bd_ba)
                            },
                            'confidence': float(confidence),
                            'index': D[0],
                            'score': self.pattern_scores.get(pattern_type, 4.0) * confidence
                        })

            patterns = sorted(patterns, key=lambda x: x['confidence'], reverse=True)
            return [p for p in patterns if p['confidence'] >= self.harmonic_min_quality]

        except Exception as e:
            logger.error(f"Error detecting harmonic patterns: {e}", exc_info=False)
            return []

    def detect_price_channels(self, df: pd.DataFrame, lookback: int = 100, min_touches: int = 3) -> Dict[
        str, Any]:
        """Detect price channels and potential breakouts or bounces."""
        results = {'status': 'ok', 'channels': [], 'details': {}}
        if not self.channel_enabled or df is None or len(df) < lookback:
            results['status'] = 'insufficient_data'
            return results

        try:
            df_window = df.iloc[-lookback:]
            highs = df_window['high'].values
            lows = df_window['low'].values
            closes = df_window['close'].values

            peaks, valleys = self.find_peaks_and_valleys(
                closes,
                distance=self.peak_detection_settings['distance'],
                prominence_factor=self.peak_detection_settings['prominence_factor']
            )

            if len(peaks) >= min_touches and len(valleys) >= min_touches:
                peak_indices = np.array(peaks)
                peak_values = closes[peak_indices]
                if len(peak_indices) >= 2:
                    up_slope, up_intercept = np.polyfit(peak_indices, peak_values, 1)

                    valley_indices = np.array(valleys)
                    valley_values = closes[valley_indices]
                    if len(valley_indices) >= 2:
                        down_slope, down_intercept = np.polyfit(valley_indices, valley_values, 1)

                        last_idx = len(closes) - 1
                        up_line_current = up_slope * last_idx + up_intercept
                        down_line_current = down_slope * last_idx + down_intercept
                        channel_width = up_line_current - down_line_current

                        if channel_width > 0:
                            channel_slope = (up_slope + down_slope) / 2
                            channel_direction = 'ascending' if channel_slope > 0.001 else 'descending' if channel_slope < -0.001 else 'horizontal'

                            up_line = up_slope * peak_indices + up_intercept
                            down_line = down_slope * valley_indices + down_intercept

                            up_dev = np.std(peak_values - up_line) if len(peak_values) > 0 else 0
                            down_dev = np.std(valley_values - down_line) if len(valley_values) > 0 else 0

                            valid_up_touches = sum(1 for i, v in zip(peak_indices, peak_values) if
                                                   abs(v - (
                                                           up_slope * i + up_intercept)) < up_dev) if up_dev > 0 else len(
                                peak_indices)
                            valid_down_touches = sum(1 for i, v in zip(valley_indices, valley_values) if
                                                     abs(v - (
                                                             down_slope * i + down_intercept)) < down_dev) if down_dev > 0 else len(
                                valley_indices)

                            last_close = closes[last_idx]
                            position_in_channel = (
                                                          last_close - down_line_current) / channel_width if channel_width > 0 else 0.5

                            is_breakout_up = last_close > up_line_current + up_dev if up_dev > 0 else last_close > up_line_current * 1.01
                            is_breakout_down = last_close < down_line_current - down_dev if down_dev > 0 else last_close < down_line_current * 0.99
                            breakout_direction = 'up' if is_breakout_up else 'down' if is_breakout_down else None

                            channel_quality = min(1.0,
                                                  (valid_up_touches + valid_down_touches) / (min_touches * 2))

                            if valid_up_touches >= min_touches - 1 and valid_down_touches >= min_touches - 1 and channel_quality >= self.channel_quality_threshold:
                                channel_info = {
                                    'type': f'{channel_direction}_channel',
                                    'direction': channel_direction,
                                    'upper_slope': float(up_slope),
                                    'upper_intercept': float(up_intercept),
                                    'lower_slope': float(down_slope),
                                    'lower_intercept': float(down_intercept),
                                    'width': float(channel_width),
                                    'quality': float(channel_quality),
                                    'position_in_channel': float(position_in_channel),
                                    'breakout': breakout_direction,
                                    'up_touches': int(valid_up_touches),
                                    'down_touches': int(valid_down_touches)
                                }
                                results['channels'].append(channel_info)

                                if breakout_direction == 'up':
                                    results['signal'] = {'type': 'channel_breakout', 'direction': 'bullish',
                                                         'score': 4.0 * channel_quality}
                                elif breakout_direction == 'down':
                                    results['signal'] = {'type': 'channel_breakout', 'direction': 'bearish',
                                                         'score': 4.0 * channel_quality}
                                elif position_in_channel < 0.2:
                                    results['signal'] = {'type': 'channel_bounce', 'direction': 'bullish',
                                                         'score': 3.0 * channel_quality}
                                elif position_in_channel > 0.8:
                                    results['signal'] = {'type': 'channel_bounce', 'direction': 'bearish',
                                                         'score': 3.0 * channel_quality}

            return results
        except Exception as e:
            logger.error(f"Error detecting price channels: {e}", exc_info=False)
            results['status'] = 'error'
            results['message'] = str(e)
            return results

    def detect_cyclical_patterns(self, df: pd.DataFrame, lookback: int = 200) -> Dict[str, Any]:
        """Detect cyclical patterns and make forecasts using FFT."""
        results = {'status': 'ok', 'cycles': [], 'forecast': None, 'details': {}}
        if not self.cycle_enabled or df is None or len(df) < lookback:
            results['status'] = 'disabled_or_insufficient_data'
            return results

        try:
            df_window = df.iloc[-lookback:].copy()
            closes = df_window['close'].values

            # Remove trend to focus on oscillations
            x = np.arange(len(closes))
            trend_coeffs = np.polyfit(x, closes, 1)
            trend = np.polyval(trend_coeffs, x)
            detrended = closes - trend

            from scipy import fft
            close_fft = fft.rfft(detrended)
            fft_freqs = fft.rfftfreq(len(detrended))

            close_fft_mag = np.abs(close_fft)
            threshold = np.mean(close_fft_mag) + np.std(close_fft_mag)
            significant_freq_indices = np.where(close_fft_mag > threshold)[0]

            filtered_indices = [i for i in significant_freq_indices if 2 <= 1 / fft_freqs[i] <= lookback / 2]

            cycles = []
            for idx in filtered_indices:
                if fft_freqs[idx] > 0:
                    period = int(1 / fft_freqs[idx])
                    amplitude = close_fft_mag[idx] / len(detrended)
                    phase = np.angle(close_fft[idx])

                    cycle_power = amplitude / np.mean(closes) * 100

                    cycles.append({
                        'period': period,
                        'amplitude': float(amplitude),
                        'amplitude_percent': float(cycle_power),
                        'phase': float(phase)
                    })

            cycles = sorted(cycles, key=lambda x: x['amplitude'], reverse=True)
            top_cycles = cycles[:min(5, len(cycles))]

            if len(top_cycles) >= self.cycle_min_cycles:
                forecast_length = 20
                forecast = np.zeros(forecast_length)
                last_trend = trend[-1]
                trend_slope = trend_coeffs[0]

                for i in range(forecast_length):
                    point_forecast = last_trend + trend_slope * (i + 1)

                    for cycle in top_cycles:
                        period = cycle['period']
                        amplitude = cycle['amplitude']
                        phase = cycle['phase']
                        t = len(closes) + i
                        cycle_component = amplitude * np.cos(2 * np.pi * t / period + phase)
                        point_forecast += cycle_component

                    forecast[i] = point_forecast

                forecast_direction = 'bullish' if forecast[-1] > closes[-1] else 'bearish'
                forecast_strength = abs(forecast[-1] - closes[-1]) / closes[-1]

                results['forecast'] = {
                    'values': [float(f) for f in forecast],
                    'direction': forecast_direction,
                    'strength': float(forecast_strength)
                }

                prediction_clarity = min(1.0, forecast_strength * 5)
                cycles_strength = min(1.0, sum(c['amplitude_percent'] for c in top_cycles) / 10)

                if forecast_direction == 'bullish':
                    results['signal'] = {
                        'type': 'cycle_bullish_forecast',
                        'direction': 'bullish',
                        'score': 2.5 * prediction_clarity * cycles_strength
                    }
                else:
                    results['signal'] = {
                        'type': 'cycle_bearish_forecast',
                        'direction': 'bearish',
                        'score': 2.5 * prediction_clarity * cycles_strength
                    }

            results['cycles'] = top_cycles
            results['details'] = {
                'total_cycles_detected': len(cycles),
                'significant_cycles': len(top_cycles),
                'detrend_coeffs': [float(c) for c in trend_coeffs]
            }

            return results
        except Exception as e:
            logger.error(f"Error detecting cyclical patterns: {e}", exc_info=False)
            results['status'] = 'error'
            results['message'] = str(e)
            return results

    def _detect_divergence_generic(self, price_series: pd.Series, indicator_series: pd.Series,
                                   indicator_name: str) -> List[Dict[str, Any]]:
        """Detect divergence between price series and an indicator."""
        signals = []
        # Check for valid input
        if price_series is None or indicator_series is None:
            logger.debug(f"Invalid input for {indicator_name} divergence detection: series is None")
            return signals

        # Ensure both series have similar lengths and enough data
        period = min(len(price_series), len(indicator_series))
        if period < 20:  # Need at least 20 candles for meaningful divergence
            logger.debug(f"Insufficient data for {indicator_name} divergence detection: period={period}")
            return signals

        try:
            # Create windows with matching indices
            price_window = price_series.iloc[-period:]
            indicator_window = indicator_series.iloc[-period:]

            # Safety check for valid values
            if price_window.isna().all() or indicator_window.isna().all():
                logger.debug(f"All NaN values in {indicator_name} divergence detection series")
                return signals

            # Find peaks and valleys
            try:
                price_peaks_idx, price_valleys_idx = self.find_peaks_and_valleys(
                    price_window.values,
                    distance=5,
                    prominence_factor=0.05,
                    window_size=period
                )

                ind_peaks_idx, ind_valleys_idx = self.find_peaks_and_valleys(
                    indicator_window.values,
                    distance=5,
                    prominence_factor=0.1,
                    window_size=period
                )
            except Exception as e:
                logger.warning(f"Error finding peaks/valleys in {indicator_name} divergence: {e}")
                return signals

            # Safety check for indices
            if (len(price_peaks_idx) == 0 and len(price_valleys_idx) == 0) or \
                    (len(ind_peaks_idx) == 0 and len(ind_valleys_idx) == 0):
                logger.debug(f"No peaks/valleys found for {indicator_name} divergence detection")
                return signals

            # Convert relative indices to pandas indices
            try:
                price_peaks_abs = price_window.index[price_peaks_idx].tolist() if len(price_peaks_idx) > 0 else []
                price_valleys_abs = price_window.index[price_valleys_idx].tolist() if len(price_valleys_idx) > 0 else []
                ind_peaks_abs = indicator_window.index[ind_peaks_idx].tolist() if len(ind_peaks_idx) > 0 else []
                ind_valleys_abs = indicator_window.index[ind_valleys_idx].tolist() if len(ind_valleys_idx) > 0 else []
            except Exception as e:
                logger.warning(f"Error converting indices in {indicator_name} divergence: {e}")
                return signals

            # ================ BEARISH DIVERGENCE DETECTION ================
            # Bearish divergence: Price makes higher highs, but indicator makes lower highs
            if len(price_peaks_abs) >= 2 and len(ind_peaks_abs) >= 2:
                # Check the last few peaks for recent divergence
                max_peaks_to_check = min(len(price_peaks_abs), 5)  # Check at most the last 5 peaks

                for i in range(max_peaks_to_check - 1):
                    cur_idx = len(price_peaks_abs) - 1 - i
                    prev_idx = cur_idx - 1

                    if prev_idx < 0 or cur_idx >= len(price_peaks_abs):
                        continue

                    p1_idx = price_peaks_abs[prev_idx]
                    p2_idx = price_peaks_abs[cur_idx]

                    p1_price = price_window.loc[p1_idx]
                    p2_price = price_window.loc[p2_idx]

                    # Price should be making higher highs
                    if p2_price <= p1_price:
                        continue

                    # Find corresponding indicator peaks
                    ind_p1_idx = self._find_closest_peak(ind_peaks_abs, p1_idx)
                    ind_p2_idx = self._find_closest_peak(ind_peaks_abs, p2_idx)

                    if ind_p1_idx is None or ind_p2_idx is None:
                        continue

                    ind_p1_val = indicator_window.loc[ind_p1_idx]
                    ind_p2_val = indicator_window.loc[ind_p2_idx]

                    # Check for bearish divergence - indicator makes lower highs
                    if ind_p2_val < ind_p1_val:
                        # Calculate divergence strength
                        price_change_pct = (p2_price - p1_price) / p1_price if p1_price != 0 else 0
                        ind_change_pct = (ind_p1_val - ind_p2_val) / ind_p1_val if ind_p1_val != 0 else 0
                        div_strength = min(1.0, (price_change_pct + ind_change_pct) / 2 * 5)

                        # Ensure minimum quality threshold
                        if div_strength >= self.divergence_sensitivity:
                            div_score = self.pattern_scores.get(f"{indicator_name}_bearish_divergence",
                                                                3.5) * div_strength

                            signals.append({
                                'type': f'{indicator_name}_bearish_divergence',
                                'direction': 'bearish',
                                'index': p2_idx,
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

            # ================ BULLISH DIVERGENCE DETECTION ================
            # Bullish divergence: Price makes lower lows, but indicator makes higher lows
            if len(price_valleys_abs) >= 2 and len(ind_valleys_abs) >= 2:
                # Check the last few valleys for recent divergence
                max_valleys_to_check = min(len(price_valleys_abs), 5)  # Check at most the last 5 valleys

                for i in range(max_valleys_to_check - 1):
                    cur_idx = len(price_valleys_abs) - 1 - i
                    prev_idx = cur_idx - 1

                    if prev_idx < 0 or cur_idx >= len(price_valleys_abs):
                        continue

                    p1_idx = price_valleys_abs[prev_idx]
                    p2_idx = price_valleys_abs[cur_idx]

                    p1_price = price_window.loc[p1_idx]
                    p2_price = price_window.loc[p2_idx]

                    # Price should be making lower lows
                    if p2_price >= p1_price:
                        continue

                    # Find corresponding indicator valleys
                    ind_p1_idx = self._find_closest_peak(ind_valleys_abs, p1_idx)
                    ind_p2_idx = self._find_closest_peak(ind_valleys_abs, p2_idx)

                    if ind_p1_idx is None or ind_p2_idx is None:
                        continue

                    ind_p1_val = indicator_window.loc[ind_p1_idx]
                    ind_p2_val = indicator_window.loc[ind_p2_idx]

                    # Check for bullish divergence - indicator makes higher lows
                    if ind_p2_val > ind_p1_val:
                        # Calculate divergence strength
                        price_change_pct = (p1_price - p2_price) / p1_price if p1_price != 0 else 0
                        ind_change_pct = (ind_p2_val - ind_p1_val) / ind_p1_val if ind_p1_val != 0 else 0
                        div_strength = min(1.0, (price_change_pct + ind_change_pct) / 2 * 5)

                        # Ensure minimum quality threshold
                        if div_strength >= self.divergence_sensitivity:
                            div_score = self.pattern_scores.get(f"{indicator_name}_bullish_divergence",
                                                                3.5) * div_strength

                            signals.append({
                                'type': f'{indicator_name}_bullish_divergence',
                                'direction': 'bullish',
                                'index': p2_idx,
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

            # Filter for recent signals only - only consider last N candles
            recent_candle_limit = 10
            if len(signals) > 0 and len(price_window) > recent_candle_limit:
                recent_threshold = price_window.index[-recent_candle_limit]
                signals = [s for s in signals if s['index'] >= recent_threshold]

            # Sort by strength
            return sorted(signals, key=lambda x: x.get('strength', 0), reverse=True)

        except Exception as e:
            logger.error(f"Error detecting {indicator_name} divergence: {str(e)}", exc_info=True)
            return []

    def _find_closest_peak(self, peaks_list, target_idx):
        """یافتن نزدیک‌ترین قله/دره به یک ایندکس هدف، با پشتیبانی از انواع مختلف داده"""
        if not peaks_list or target_idx is None:
            return None

        try:
            # تشخیص نوع داده‌ها و مدیریت مناسب آنها
            if all(isinstance(idx, pd.Timestamp) for idx in peaks_list) and isinstance(target_idx, pd.Timestamp):
                # هر دو تایم‌استمپ هستند - میتوان از فاصله زمانی استفاده کرد
                distances = [(abs((idx - target_idx).total_seconds()), idx) for idx in peaks_list]

            elif all(isinstance(idx, pd.Timestamp) for idx in peaks_list) and isinstance(target_idx, int):
                # ایندکس‌ها تایم‌استمپ هستند، اما هدف عدد صحیح است
                # تبدیل براساس موقعیت نسبی
                distances = [(abs(i - target_idx), idx) for i, idx in enumerate(peaks_list)]

            elif all(isinstance(idx, int) for idx in peaks_list) and isinstance(target_idx, pd.Timestamp):
                # ایندکس‌ها عدد صحیح هستند، اما هدف تایم‌استمپ است
                # از موقعیت نسبی استفاده می‌کنیم بدون تفریق مستقیم
                peak_positions = list(range(len(peaks_list)))
                # موقعیت نسبی هر پیک را با موقعیت هدف مقایسه می‌کنیم
                distances = [(abs(pos), peaks_list[i]) for i, pos in enumerate(peak_positions)]

            else:
                # فرض می‌کنیم هر دو از نوع یکسان هستند یا قابل مقایسه
                try:
                    distances = [(abs(idx - target_idx), idx) for idx in peaks_list]
                except TypeError:
                    # اگر مقایسه مستقیم ممکن نیست، از موقعیت نسبی استفاده می‌کنیم
                    distances = [(float(i), idx) for i, idx in enumerate(peaks_list)]

            closest_peak = min(distances, key=lambda x: x[0])[1]
            return closest_peak

        except Exception as e:
            # ثبت خطا برای اشکال‌زدایی
            peak_type = type(peaks_list[0]) if peaks_list else None
            target_type = type(target_idx)
            logger.error(f"Error in _find_closest_peak: {e}, peaks_type: {peak_type}, target_type: {target_type}")
            return peaks_list[0] if peaks_list else None

    def detect_divergence(self, price_data: np.ndarray, indicator_data: np.ndarray,
                          threshold: Optional[float] = None) -> List[Dict[str, Any]]:
        """Detect divergence between price and indicator data."""
        sensitivity = threshold if threshold is not None else self.divergence_sensitivity
        if price_data is None or indicator_data is None or len(price_data) != len(indicator_data) or len(
                price_data) < 20:
            return []

        try:
            price_series = pd.Series(price_data)
            indicator_series = pd.Series(indicator_data)
            return self._detect_divergence_generic(price_series, indicator_series, 'rsi')
        except Exception as e:
            logger.error(f"Error in detect_divergence: {e}", exc_info=False)
            return []

    def _detect_macd_market_type(self, dif: pd.Series, hist: pd.Series, ema20: pd.Series,
                                 ema50: pd.Series) -> str:
        """Detect MACD market type for trading strategies."""
        if dif.empty or hist.empty or ema20.empty or ema50.empty or len(dif) < 1:
            return "unknown_data"
        try:
            curr_dif = dif.iloc[-1]
            curr_hist = hist.iloc[-1]
            curr_ema20 = ema20.iloc[-1]
            curr_ema50 = ema50.iloc[-1]

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
        except IndexError:
            return "unknown_index"
        except Exception as e:
            logger.error(f"MACD market type error: {e}")
            return "error"

    def _detect_detailed_macd_crosses(self, dif: pd.Series, dea: pd.Series, dates_index: pd.Index) -> List[
        Dict[str, Any]]:
        """Analyze MACD crosses with detailed context information."""
        signals = []
        min_len = max(2, self.macd_cross_period)
        if len(dif) < min_len or len(dea) < min_len or len(dates_index) != len(dif) or len(dates_index) != len(dea):
            # logger.warning(
            #     f"_detect_detailed_macd_crosses: Mismatch lengths or insufficient data. "
            #     f"dif:{len(dif)}, dea:{len(dea)}, index:{len(dates_index)}, min_len:{min_len}")
            return signals

        try:
            # بررسی داده‌های اخیر در محدوده macd_cross_period
            cross_window = min(len(dif), self.macd_cross_period)
            dif_window = dif.iloc[-cross_window:]
            dea_window = dea.iloc[-cross_window:]
            dates_window = dates_index[-cross_window:]

            for i in range(1, len(dif_window)):
                current_abs_index = dates_window[i]
                dif_val_curr = dif_window.iloc[i]
                dif_val_prev = dif_window.iloc[i - 1]
                dea_val_curr = dea_window.iloc[i]
                dea_val_prev = dea_window.iloc[i - 1]

                # تقاطع صعودی MACD (Golden Cross)
                if dif_val_prev < dea_val_prev and dif_val_curr > dea_val_curr:
                    # تعیین نوع تقاطع براساس محل تقاطع نسبت به خط صفر
                    if dif_val_curr < 0:
                        cross_type = "macd_gold_cross_below_zero"
                    else:
                        cross_type = "macd_gold_cross_above_zero"

                    # محاسبه قدرت تقاطع
                    cross_strength = min(1.0, abs(dif_val_curr - dea_val_curr) * 5)
                    signal_score = self.pattern_scores.get(cross_type, 2.5) * cross_strength

                    signals.append({
                        'type': cross_type,
                        'direction': 'bullish',
                        'index': current_abs_index,
                        'date': current_abs_index,
                        'score': signal_score,
                        'strength': cross_strength,
                        'details': {
                            'dif': float(dif_val_curr),
                            'dea': float(dea_val_curr),
                            'above_zero': dif_val_curr > 0
                        }
                    })

                # تقاطع نزولی MACD (Death Cross)
                elif dif_val_prev > dea_val_prev and dif_val_curr < dea_val_curr:
                    # تعیین نوع تقاطع براساس محل تقاطع نسبت به خط صفر
                    if dif_val_curr > 0:
                        cross_type = "macd_death_cross_above_zero"
                    else:
                        cross_type = "macd_death_cross_below_zero"

                    # محاسبه قدرت تقاطع
                    cross_strength = min(1.0, abs(dif_val_curr - dea_val_curr) * 5)
                    signal_score = self.pattern_scores.get(cross_type, 2.5) * cross_strength

                    signals.append({
                        'type': cross_type,
                        'direction': 'bearish',
                        'index': current_abs_index,
                        'date': current_abs_index,
                        'score': signal_score,
                        'strength': cross_strength,
                        'details': {
                            'dif': float(dif_val_curr),
                            'dea': float(dea_val_curr),
                            'above_zero': dif_val_curr > 0
                        }
                    })

            # فیلتر کردن فقط سیگنال‌های کندل آخر
            last_date = dates_index[-1]
            recent_signals = [s for s in signals if s['date'] == last_date]
            return recent_signals

        except IndexError as idx_err:
            err_index = 'unknown'
            try:
                err_index = i
            except NameError:
                pass
            logger.error(
                f"IndexError in detailed MACD crosses (relative index i={err_index}): {idx_err}. "
                f"Lengths - dif: {len(dif)}, dea: {len(dea)}, index: {len(dates_index)}")
            return []
        except Exception as e:
            logger.error(f"Error detecting detailed MACD crosses: {e}", exc_info=True)
            return []

    def _find_closest_index(self, index_list: List, target_idx) -> Optional:
        """یافتن نزدیک‌ترین شاخص در لیست براساس فاصله زمانی یا عددی"""
        if not index_list or len(index_list) == 0:
            return None

        try:
            # محاسبه فاصله‌ها براساس نوع داده
            if isinstance(target_idx, pd.Timestamp) and isinstance(index_list[0], pd.Timestamp):
                # فاصله زمانی برای شاخص‌های تاریخی
                distances = [abs((idx - target_idx).total_seconds()) for idx in index_list]
            else:
                # فاصله عددی برای شاخص‌های عددی
                distances = [abs(idx - target_idx) for idx in index_list]

            # یافتن شاخص با کمترین فاصله
            min_distance_index = distances.index(min(distances))

            # تعیین حداکثر فاصله قابل قبول
            max_allowed_distance = 0

            if isinstance(target_idx, pd.Timestamp):
                max_allowed_distance = 86400  # یک روز به ثانیه
            else:
                max_allowed_distance = 5  # فاصله عددی مناسب

            if distances[min_distance_index] > max_allowed_distance:
                return None

            return index_list[min_distance_index]
        except Exception as e:
            logger.debug(f"Error finding closest index: {e}")
            return None

    def _detect_dif_behavior(self, dif: pd.Series, dates_index: pd.Index) -> List[Dict[str, Any]]:
        """Analyze DIF line behavior (zero crosses, trendline breaks)."""
        signals = []
        if not isinstance(dif, pd.Series) or not isinstance(dates_index, pd.Index):
            logger.warning("_detect_dif_behavior: Invalid input types.")
            return signals

        min_len_zero = max(2, self.macd_cross_period)
        min_len_trend = max(10, self.macd_trendline_period)
        required_len = max(min_len_zero, min_len_trend)

        if len(dif) < required_len or len(dates_index) != len(dif):
            # logger.warning(
            #     f"_detect_dif_behavior: Mismatch lengths or insufficient data. dif:{len(dif)}, index:{len(dates_index)}, required:{required_len}")
            return signals

        try:
            dif_vals = dif.values

            # --- Zero line crosses ---
            cross_up_count = 0
            cross_down_count = 0
            for i in range(1, len(dif)):
                crossed_up = dif_vals[i - 1] < 0 and dif_vals[i] > 0
                crossed_down = dif_vals[i - 1] > 0 and dif_vals[i] < 0

                if crossed_up or crossed_down:
                    current_abs_index = dates_index[i]
                    current_datetime = dates_index[i]

                    if crossed_up:
                        cross_up_count += 1
                        signal_type = f"dif_cross_zero_up_{'first' if cross_up_count == 1 else 'second'}"
                        signals.append({'type': signal_type, 'direction': 'bullish', 'index': current_abs_index,
                                        'date': current_datetime,
                                        'score': self.pattern_scores.get(signal_type, 2.0)})
                    elif crossed_down:
                        cross_down_count += 1
                        signal_type = f"dif_cross_zero_down_{'first' if cross_down_count == 1 else 'second'}"
                        signals.append({'type': signal_type, 'direction': 'bearish', 'index': current_abs_index,
                                        'date': current_datetime,
                                        'score': self.pattern_scores.get(signal_type, 2.0)})

            # --- Trendline breaks ---
            dif_for_trend = dif.iloc[-self.macd_trendline_period:]
            if len(dif_for_trend) > self.macd_peak_detection_settings['smooth_kernel'] * 2:
                if pd.api.types.is_numeric_dtype(dif_for_trend.dtype):
                    smooth_dif_vals = scipy.signal.medfilt(dif_for_trend.values,
                                                           kernel_size=self.macd_peak_detection_settings[
                                                               'smooth_kernel'])

                    peaks_iloc_rel, valleys_iloc_rel = self.find_peaks_and_valleys(
                        smooth_dif_vals,
                        distance=self.macd_peak_detection_settings['distance'],
                        prominence_factor=self.macd_peak_detection_settings['prominence_factor']
                    )

                    def check_trendline_break(points_iloc_relative, is_resistance):
                        if len(points_iloc_relative) < 2:
                            return None

                        p1_rel_idx, p2_rel_idx = points_iloc_relative[-2], points_iloc_relative[-1]
                        p1_val, p2_val = smooth_dif_vals[p1_rel_idx], smooth_dif_vals[p2_rel_idx]

                        if p2_rel_idx == p1_rel_idx:  # Prevent division by zero
                            return None
                        k = (p2_val - p1_val) / (p2_rel_idx - p1_rel_idx)
                        b = p1_val - k * p1_rel_idx  # Intercept relative to relative index

                        p2_abs_index = dif_for_trend.index[p2_rel_idx]
                        try:
                            p2_loc_in_original_dif = dif.index.get_loc(p2_abs_index)
                            start_check_original_iloc = p2_loc_in_original_dif + 1
                        except KeyError:
                            return None

                        for i_original_iloc in range(start_check_original_iloc, len(dif)):
                            current_abs_index = dates_index[i_original_iloc]

                            i_rel_trend = len(smooth_dif_vals) - (len(dif) - i_original_iloc)

                            if i_rel_trend < 0: continue

                            trendline_val = k * i_rel_trend + b
                            current_dif = dif_vals[i_original_iloc]
                            margin = abs(current_dif * 0.01)
                            current_datetime = current_abs_index

                            if is_resistance and current_dif > trendline_val + margin:
                                return {
                                    'type': 'dif_trendline_break_up',
                                    'direction': 'bullish',
                                    'index': current_abs_index,
                                    'date': current_datetime,
                                    'score': self.pattern_scores.get('dif_trendline_break_up', 3.0)
                                }
                            elif not is_resistance and current_dif < trendline_val - margin:
                                return {
                                    'type': 'dif_trendline_break_down',
                                    'direction': 'bearish',
                                    'index': current_abs_index,
                                    'date': current_datetime,
                                    'score': self.pattern_scores.get('dif_trendline_break_down', 3.0)
                                }
                        return None

                    break_up_signal = check_trendline_break(peaks_iloc_rel, is_resistance=True)
                    if break_up_signal:
                        signals.append(break_up_signal)

                    break_down_signal = check_trendline_break(valleys_iloc_rel, is_resistance=False)
                    if break_down_signal:
                        signals.append(break_down_signal)
                else:
                    logger.debug(f"_detect_dif_behavior: Not enough numeric data in dif_for_trend after smoothing.")

            if not signals:
                return []

            if not dates_index.empty:
                last_date = dates_index[-1]
                recent_signals = [s for s in signals if s['date'] == last_date]
                return recent_signals
            else:
                return []

        except IndexError as idx_err:
            logger.error(f"IndexError in DIF behavior: {idx_err}. Lengths - dif: {len(dif)}, index: {len(dates_index)}")
            return []
        except Exception as e:
            logger.error(f"Error detecting DIF behavior: {e}", exc_info=True)
            return []

    def _analyze_macd_histogram(self, hist: pd.Series, close: pd.Series, dates_index: pd.Index) -> List[Dict[str, Any]]:
        """Analyze MACD histogram features and patterns."""
        signals = []
        min_len = max(10, self.macd_hist_period)
        if len(hist) < min_len or len(close) != len(hist) or len(dates_index) != len(hist):
            # logger.warning(
            #     f"_analyze_macd_histogram: Mismatch lengths or insufficient data. hist:{len(hist)}, close:{len(close)}, index:{len(dates_index)}, min_len:{min_len}")
            return signals

        try:
            hist_window = hist
            close_window = close

            peaks_iloc, valleys_iloc = self.find_peaks_and_valleys(
                hist_window.values,
                distance=self.macd_peak_detection_settings['distance'],
                prominence_factor=self.macd_peak_detection_settings['prominence_factor']
            )

            for idx_rel in peaks_iloc:
                if hist_window.iloc[idx_rel] > 0:
                    abs_idx = dates_index[idx_rel]
                    signals.append({
                        'type': 'macd_hist_shrink_head',
                        'direction': 'bearish',
                        'index': abs_idx,
                        'date': abs_idx,
                        'score': self.pattern_scores.get('macd_hist_shrink_head', 1.5)
                    })
            for idx_rel in valleys_iloc:
                if hist_window.iloc[idx_rel] < 0:
                    abs_idx = dates_index[idx_rel]
                    signals.append({
                        'type': 'macd_hist_pull_feet',
                        'direction': 'bullish',
                        'index': abs_idx,
                        'date': abs_idx,
                        'score': self.pattern_scores.get('macd_hist_pull_feet', 1.5)
                    })

            # Histogram divergence with price
            if len(peaks_iloc) >= 2:
                p1_rel, p2_rel = peaks_iloc[-2], peaks_iloc[-1]
                p1_abs, p2_abs = dates_index[p1_rel], dates_index[p2_rel]
                if hist_window.iloc[p2_rel] < hist_window.iloc[p1_rel] and close_window.loc[p2_abs] > close_window.loc[
                    p1_abs]:
                    signals.append({
                        'type': 'macd_hist_top_divergence',
                        'direction': 'bearish',
                        'index': p2_abs,
                        'date': p2_abs,
                        'score': self.pattern_scores.get('macd_hist_top_divergence', 3.8)
                    })
            if len(valleys_iloc) >= 2:
                v1_rel, v2_rel = valleys_iloc[-2], valleys_iloc[-1]
                v1_abs, v2_abs = dates_index[v1_rel], dates_index[v2_rel]
                if hist_window.iloc[v2_rel] > hist_window.iloc[v1_rel] and close_window.loc[v2_abs] < close_window.loc[
                    v1_abs]:
                    signals.append({
                        'type': 'macd_hist_bottom_divergence',
                        'direction': 'bullish',
                        'index': v2_abs,
                        'date': v2_abs,
                        'score': self.pattern_scores.get('macd_hist_bottom_divergence', 3.8)
                    })

            # Kill Long / Force Short bins
            if len(valleys_iloc) >= 2:
                for i in range(len(valleys_iloc) - 1):
                    v1_rel, v2_rel = valleys_iloc[i], valleys_iloc[i + 1]
                    v1_abs, v2_abs = dates_index[v1_rel], dates_index[v2_rel]
                    if hist_window.iloc[v1_rel] < 0 and hist_window.iloc[v2_rel] < 0:
                        hist_between = hist_window.iloc[v1_rel: v2_rel + 1]
                        if not hist_between.empty and hist_between.max() < 0:
                            signals.append({
                                'type': 'macd_hist_kill_long_bin',
                                'direction': 'bearish',
                                'index': v2_abs,
                                'date': v2_abs,
                                'score': self.pattern_scores.get('macd_hist_kill_long_bin', 2.0)
                            })

            if not dates_index.empty:
                last_date = dates_index[-1]
                recent_signals = [s for s in signals if s['date'] == last_date]
                return recent_signals
            else:
                return []

        except IndexError as idx_err:
            logger.error(
                f"IndexError in MACD histogram: {idx_err}. Lengths - hist: {len(hist)}, close: {len(close)}, index: {len(dates_index)}")
            return []
        except Exception as e:
            logger.error(f"Error analyzing MACD histogram: {e}", exc_info=True)
            return []

    def analyze_momentum_indicators(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Analyze momentum indicators (MACD, RSI, Stochastic, MFI)."""
        results = {'status': 'ok', 'direction': 'neutral', 'signals': [], 'details': {}}
        required_len = 35
        if df is None or len(df) < required_len:
            results['status'] = 'insufficient_data'
            return results
        try:
            symbol = df.attrs.get('symbol', 'unknown')
            timeframe = df.attrs.get('timeframe', 'unknown')

            close_p = df['close'].values.astype(np.float64)
            high_p = df['high'].values.astype(np.float64)
            low_p = df['low'].values.astype(np.float64)

            # Get cached indicators or calculate them
            macd_key = self._cache_key(symbol, timeframe, 'MACD', (12, 26, 9))
            cached_macd = self._get_cached_indicator(macd_key)
            if cached_macd is not None:
                macd, macd_signal, macd_hist = cached_macd
            else:
                macd, macd_signal, macd_hist = talib.MACD(close_p, fastperiod=12, slowperiod=26, signalperiod=9)
                self._cache_indicator(macd_key, (macd, macd_signal, macd_hist))

            rsi_key = self._cache_key(symbol, timeframe, 'RSI', (14,))
            rsi = self._get_cached_indicator(rsi_key)
            if rsi is None:
                rsi = talib.RSI(close_p, timeperiod=14)
                self._cache_indicator(rsi_key, rsi)

            stoch_key = self._cache_key(symbol, timeframe, 'STOCH', (14, 3, 3))
            cached_stoch = self._get_cached_indicator(stoch_key)
            if cached_stoch is not None:
                slowk, slowd = cached_stoch
            else:
                slowk, slowd = talib.STOCH(high_p, low_p, close_p, fastk_period=14, slowk_period=3, slowd_period=3)
                self._cache_indicator(stoch_key, (slowk, slowd))

            # Calculate MFI if volume exists
            mfi = None
            if 'volume' in df.columns:
                try:
                    volume_p = df['volume'].values.astype(np.float64)
                    mfi_key = self._cache_key(symbol, timeframe, 'MFI', (14,))
                    mfi = self._get_cached_indicator(mfi_key)
                    if mfi is None:
                        mfi = talib.MFI(high_p, low_p, close_p, volume_p, timeperiod=14)
                        self._cache_indicator(mfi_key, mfi)
                except Exception as mfi_err:
                    logger.debug(f"Error calculating MFI: {mfi_err}")

            # Find last valid index
            last_valid_idx = -1
            while last_valid_idx >= -len(df) and (
                    np.isnan(macd[last_valid_idx]) or np.isnan(rsi[last_valid_idx]) or np.isnan(
                slowk[last_valid_idx])):
                last_valid_idx -= 1

            if abs(last_valid_idx) > len(df):
                results['status'] = 'calculation_error'
                return results

            # Get current and previous values
            curr_macd, prev_macd = macd[last_valid_idx], macd[last_valid_idx - 1]
            curr_sig, prev_sig = macd_signal[last_valid_idx], macd_signal[last_valid_idx - 1]
            curr_rsi, prev_rsi = rsi[last_valid_idx], rsi[last_valid_idx - 1]
            curr_k, prev_k = slowk[last_valid_idx], slowk[last_valid_idx - 1]
            curr_d, prev_d = slowd[last_valid_idx], slowd[last_valid_idx - 1]

            curr_mfi, prev_mfi = (mfi[last_valid_idx], mfi[last_valid_idx - 1]) if mfi is not None and not np.isnan(
                mfi[last_valid_idx]) else (None, None)

            momentum_signals = []

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

            # 3. RSI Oversold/Overbought Reversal
            if curr_rsi < 30 and curr_rsi > prev_rsi:
                momentum_signals.append({
                    'type': 'rsi_oversold_reversal',
                    'score': self.pattern_scores.get('rsi_oversold_reversal', 2.3)
                })
            elif curr_rsi > 70 and curr_rsi < prev_rsi:
                momentum_signals.append({
                    'type': 'rsi_overbought_reversal',
                    'score': self.pattern_scores.get('rsi_overbought_reversal', 2.3)
                })

            # 4. Stochastic Crossover in Oversold/Overbought
            if curr_k < 20 and curr_d < 20 and curr_k > curr_d and prev_k <= prev_d:
                momentum_signals.append({
                    'type': 'stochastic_oversold_bullish_cross',
                    'score': self.pattern_scores.get('stochastic_oversold_bullish_cross', 2.5)
                })
            elif curr_k > 80 and curr_d > 80 and curr_k < curr_d and prev_k >= prev_d:
                momentum_signals.append({
                    'type': 'stochastic_overbought_bearish_cross',
                    'score': self.pattern_scores.get('stochastic_overbought_bearish_cross', 2.5)
                })

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

            # 6. RSI Divergence
            close_s = pd.Series(close_p)
            rsi_s = pd.Series(rsi)
            rsi_divergences = self._detect_divergence_generic(close_s, rsi_s, 'rsi')
            momentum_signals.extend(rsi_divergences)

            results['signals'] = momentum_signals

            # Calculate overall momentum direction
            bullish_score = sum(
                s['score'] for s in momentum_signals if 'bullish' in s.get('direction', s.get('type', '')))
            bearish_score = sum(
                s['score'] for s in momentum_signals if 'bearish' in s.get('direction', s.get('type', '')))

            if bullish_score > bearish_score:
                results['direction'] = 'bullish'
            elif bearish_score > bullish_score:
                results['direction'] = 'bearish'

            results['bullish_score'] = round(bullish_score, 2)
            results['bearish_score'] = round(bearish_score, 2)

            # Indicator states
            rsi_condition = 'oversold' if curr_rsi < 30 else 'overbought' if curr_rsi > 70 else 'neutral'
            stoch_condition = 'oversold' if curr_k < 20 and curr_d < 20 else 'overbought' if curr_k > 80 and curr_d > 80 else 'neutral'
            mfi_condition = 'oversold' if curr_mfi is not None and curr_mfi < 20 else 'overbought' if curr_mfi is not None and curr_mfi > 80 else 'neutral'

            results['details'] = {
                'rsi': round(curr_rsi, 2),
                'macd': round(curr_macd, 5),
                'macd_signal': round(curr_sig, 5),
                'stoch_k': round(curr_k, 2),
                'stoch_d': round(curr_d, 2),
                'mfi': round(curr_mfi, 2) if curr_mfi is not None else None,
                'rsi_condition': rsi_condition,
                'stoch_condition': stoch_condition,
                'mfi_condition': mfi_condition
            }

            return results

        except Exception as e:
            logger.error(f"Error analyzing momentum indicators: {e}", exc_info=False)
            results['status'] = 'error'
            results['message'] = str(e)
            return results

    def detect_reversal_conditions(self, analysis_results: Dict[str, Dict[str, Any]], timeframe: str) -> Tuple[
        bool, float]:
        """Detect reversal conditions in a specific timeframe."""
        try:
            result = analysis_results.get(timeframe, {})
            if result.get('status') != 'ok':
                return False, 0.0

            momentum_data = result.get('momentum', {})
            pa_data = result.get('price_action', {})
            sr_data = result.get('support_resistance', {}).get('details', {})
            trend_data = result.get('trend', {})
            harmonic_patterns = result.get('harmonic_patterns', [])
            channel_data = result.get('price_channels', {})
            channel_signal = channel_data.get('signal', {})

            is_reversal = False
            strength = 0.0

            # 1. RSI Divergence
            div_signals = momentum_data.get('signals', [])
            if any('rsi_bullish_divergence' == s.get('type') for s in div_signals):
                strength += 0.7
                is_reversal = True
            if any('rsi_bearish_divergence' == s.get('type') for s in div_signals):
                strength += 0.7
                is_reversal = True

            # 2. Oversold/Overbought against trend
            rsi_cond = momentum_data.get('details', {}).get('rsi_condition', 'neutral')
            trend = trend_data.get('trend', 'neutral')
            if (rsi_cond == 'oversold' and 'bearish' in trend) or (rsi_cond == 'overbought' and 'bullish' in trend):
                strength += 0.5
                is_reversal = True

            # 3. Reversal candlestick patterns
            reversal_patterns = ['hammer', 'inverted_hammer', 'morning_star', 'evening_star', 'bullish_engulfing',
                                 'bearish_engulfing', 'dragonfly_doji', 'gravestone_doji']
            pa_signals = pa_data.get('signals', [])
            pattern_strength = sum(s.get('score', 0) / 3.0 for s in pa_signals if
                                   any(p in s.get('type', '') for p in reversal_patterns))
            if pattern_strength > 0:
                strength += pattern_strength
                is_reversal = True

            # 4. Harmonic patterns
            for pattern in harmonic_patterns:
                if pattern.get('type', '').endswith('butterfly') or pattern.get('type', '').endswith('crab'):
                    pattern_quality = pattern.get('confidence', 0.7)
                    strength += 0.8 * pattern_quality
                    is_reversal = True

            # 5. Price channel signals
            if channel_signal:
                signal_type = channel_signal.get('type', '')
                if signal_type == 'channel_bounce':
                    signal_score = channel_signal.get('score', 0) / 3.0
                    strength += signal_score
                    is_reversal = True

            # 6. Support/Resistance fakeouts
            nearest_resist = sr_data.get('nearest_resistance', {}).get('price') if isinstance(
                sr_data.get('nearest_resistance'), dict) else sr_data.get('nearest_resistance')
            nearest_support = sr_data.get('nearest_support', {}).get('price') if isinstance(
                sr_data.get('nearest_support'), dict) else sr_data.get('nearest_support')
            broken_resist = sr_data.get('broken_resistance', {}).get('price') if isinstance(
                sr_data.get('broken_resistance'), dict) else sr_data.get('broken_resistance')
            broken_support = sr_data.get('broken_support', {}).get('price') if isinstance(
                sr_data.get('broken_support'), dict) else sr_data.get('broken_support')

            current_close = result.get('price_action', {}).get('details', {}).get('close')
            if current_close and nearest_resist and broken_resist:
                if abs(current_close - broken_resist) / broken_resist < 0.01:
                    strength += 0.6
                    is_reversal = True
            if current_close and nearest_support and broken_support:
                if abs(current_close - broken_support) / broken_support < 0.01:
                    strength += 0.6
                    is_reversal = True

            return is_reversal, min(1.0, strength)

        except Exception as e:
            logger.error(f"Error detecting reversal conditions for {timeframe}: {e}", exc_info=False)
            return False, 0.0

    def _analyze_basic_momentum(self, rsi: pd.Series, slowk: np.ndarray, slowd: np.ndarray, dif: pd.Series,
                                dea: pd.Series) -> Dict[str, Any]:
        """Basic momentum analysis from core indicators."""
        results = {'status': 'ok', 'direction': 'neutral', 'signals': [], 'details': {}}
        required_len = 2
        if rsi.empty or len(slowk) < required_len or dif.empty or dea.empty:
            results['status'] = 'insufficient_data'
            return results

        try:
            curr_rsi = rsi.iloc[-1]
            prev_rsi = rsi.iloc[-2]
            curr_k = slowk[-1]
            prev_k = slowk[-2]
            curr_d = slowd[-1]
            prev_d = slowd[-2]
            curr_dif = dif.iloc[-1]
            prev_dif = dif.iloc[-2]
            curr_dea = dea.iloc[-1]
            prev_dea = dea.iloc[-2]

            momentum_signals = []

            # 1. MACD Crossover
            if curr_dif > curr_dea and prev_dif <= prev_dea:
                momentum_signals.append(
                    {'type': 'macd_bullish_crossover', 'score': self.pattern_scores.get('macd_bullish_crossover', 2.2)})
            elif curr_dif < curr_dea and prev_dif >= prev_dea:
                momentum_signals.append(
                    {'type': 'macd_bearish_crossover', 'score': self.pattern_scores.get('macd_bearish_crossover', 2.2)})

            # 2. MACD Zero Line Cross
            if curr_dif > 0 and prev_dif <= 0:
                momentum_signals.append({'type': 'macd_bullish_zero_cross',
                                         'score': self.pattern_scores.get('macd_bullish_zero_cross', 1.8)})
            elif curr_dif < 0 and prev_dif >= 0:
                momentum_signals.append({'type': 'macd_bearish_zero_cross',
                                         'score': self.pattern_scores.get('macd_bearish_zero_cross', 1.8)})

            # 3. RSI Oversold/Overbought Reversal
            if curr_rsi < 30 and curr_rsi > prev_rsi:
                momentum_signals.append(
                    {'type': 'rsi_oversold_reversal', 'score': self.pattern_scores.get('rsi_oversold_reversal', 2.3)})
            elif curr_rsi > 70 and curr_rsi < prev_rsi:
                momentum_signals.append({'type': 'rsi_overbought_reversal',
                                         'score': self.pattern_scores.get('rsi_overbought_reversal', 2.3)})

            # 4. Stochastic Crossover in Oversold/Overbought
            if curr_k < 20 and curr_d < 20 and curr_k > curr_d and prev_k <= prev_d:
                momentum_signals.append({
                    'type': 'stochastic_oversold_bullish_cross',
                    'score': self.pattern_scores.get('stochastic_oversold_bullish_cross', 2.5)
                })
            elif curr_k > 80 and curr_d > 80 and curr_k < curr_d and prev_k >= prev_d:
                momentum_signals.append({
                    'type': 'stochastic_overbought_bearish_cross',
                    'score': self.pattern_scores.get('stochastic_overbought_bearish_cross', 2.5)
                })

            results['signals'] = momentum_signals

            # Overall momentum direction
            bullish_score = sum(s['score'] for s in momentum_signals if 'bullish' in s['type'])
            bearish_score = sum(s['score'] for s in momentum_signals if 'bearish' in s['type'])

            if bullish_score > bearish_score * 1.1:
                results['direction'] = 'bullish'
            elif bearish_score > bullish_score * 1.1:
                results['direction'] = 'bearish'

            results['bullish_score'] = round(bullish_score, 2)
            results['bearish_score'] = round(bearish_score, 2)
            results['details'] = {
                'rsi': round(curr_rsi, 2),
                'macd': round(curr_dif, 5),
                'macd_signal': round(curr_dea, 5),
                'stoch_k': round(curr_k, 2),
                'stoch_d': round(curr_d, 2)
            }

            return results

        except Exception as e:
            logger.error(f"Error analyzing basic momentum: {e}", exc_info=False)
            results['status'] = 'error'
            results['message'] = str(e)
            return results

    async def analyze_price_action(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Analyze price action including candlestick patterns and Bollinger Bands."""
        results = {'status': 'ok', 'direction': 'neutral', 'signals': [], 'details': {}, 'atr': None}
        if df is None or len(df) < 20:
            results['status'] = 'insufficient_data'
            return results

        try:
            symbol = df.attrs.get('symbol', 'unknown')
            timeframe = df.attrs.get('timeframe', 'unknown')

            # Calculate ATR
            high_p = df['high'].values.astype(np.float64)
            low_p = df['low'].values.astype(np.float64)
            close_p = df['close'].values.astype(np.float64)

            # Check cached ATR or calculate
            atr_key = self._cache_key(symbol, timeframe, 'ATR', (14,))
            atr = self._get_cached_indicator(atr_key)
            if atr is None:
                atr = talib.ATR(high_p, low_p, close_p, timeperiod=14)
                self._cache_indicator(atr_key, atr)

            last_atr = atr[~np.isnan(atr)][-1] if not np.all(np.isnan(atr)) else 0
            results['atr'] = round(last_atr, 8) if last_atr > 0 else None

            # Calculate Bollinger Bands
            bb_key = self._cache_key(symbol, timeframe, 'BBANDS', (20, 2))
            cached_bb = self._get_cached_indicator(bb_key)
            if cached_bb is not None:
                upper, middle, lower = cached_bb
            else:
                upper, middle, lower = talib.BBANDS(close_p, timeperiod=20, nbdevup=2, nbdevdn=2, matype=0)
                self._cache_indicator(bb_key, (upper, middle, lower))

            # Analyze price position relative to Bollinger Bands
            current_close = df['close'].iloc[-1]
            current_upper = upper[-1]
            current_lower = lower[-1]
            current_middle = middle[-1]

            if not np.isnan(current_upper) and not np.isnan(current_lower) and not np.isnan(current_middle):
                bb_position = (current_close - current_lower) / (
                        current_upper - current_lower) if current_upper > current_lower else 0.5
                bb_width = (current_upper - current_lower) / current_middle if current_middle > 0 else 0
                bb_squeeze = False

                if len(df) >= 40:
                    prev_widths = [(upper[i] - lower[i]) / middle[i] if middle[i] > 0 else 0 for i in range(-20, -1)]
                    avg_width = np.mean([w for w in prev_widths if w > 0])
                    bb_squeeze = bb_width < avg_width * 0.8

                results['details']['bollinger_bands'] = {
                    'upper': float(current_upper),
                    'middle': float(current_middle),
                    'lower': float(current_lower),
                    'position': float(bb_position),
                    'width': float(bb_width),
                    'squeeze': bb_squeeze
                }

                # Add Bollinger Bands signals
                if bb_squeeze:
                    results['signals'].append({
                        'type': 'bollinger_squeeze',
                        'direction': 'neutral',
                        'score': self.pattern_scores.get('bollinger_squeeze', 2.0)
                    })

                if current_close > current_upper:
                    results['signals'].append({
                        'type': 'bollinger_upper_break',
                        'direction': 'bullish',
                        'score': self.pattern_scores.get('bollinger_upper_break', 2.5)
                    })
                elif current_close < current_lower:
                    results['signals'].append({
                        'type': 'bollinger_lower_break',
                        'direction': 'bearish',
                        'score': self.pattern_scores.get('bollinger_lower_break', 2.5)
                    })

            # Detect candlestick patterns
            candle_patterns = await self.detect_candlestick_patterns(df)

            # Advanced volume analysis
            volume_analysis = {}
            if 'volume' in df.columns and len(df) >= 30:
                volume = df['volume'].values
                avg_volume = np.mean(volume[-30:-1])
                current_volume = volume[-1]
                volume_ratio = current_volume / avg_volume if avg_volume > 0 else 1.0

                volume_analysis = {
                    'current_volume': float(current_volume),
                    'avg_volume': float(avg_volume),
                    'volume_ratio': float(volume_ratio),
                    'is_high_volume': volume_ratio > 1.5,
                    'is_very_high_volume': volume_ratio > 2.5,
                    'is_low_volume': volume_ratio < 0.5
                }

                # Add volume signals
                if volume_ratio > 2.5:
                    if current_close > df['open'].iloc[-1]:  # Bullish candle
                        results['signals'].append({
                            'type': 'high_volume_bullish',
                            'direction': 'bullish',
                            'score': self.pattern_scores.get('high_volume_bullish', 2.8)
                        })
                    else:  # Bearish candle
                        results['signals'].append({
                            'type': 'high_volume_bearish',
                            'direction': 'bearish',
                            'score': self.pattern_scores.get('high_volume_bearish', 2.8)
                        })

            results['details']['candle_patterns'] = candle_patterns
            results['details']['volume_analysis'] = volume_analysis

            # Combine all signals
            price_action_signals = candle_patterns + results['signals']

            # Calculate scores
            bullish_score = 0
            bearish_score = 0
            for pattern in price_action_signals:
                if pattern.get('direction') == 'bullish':
                    bullish_score += pattern.get('score', 0)
                elif pattern.get('direction') == 'bearish':
                    bearish_score += pattern.get('score', 0)

            results['signals'] = price_action_signals
            if bullish_score > bearish_score:
                results['direction'] = 'bullish'
            elif bearish_score > bullish_score:
                results['direction'] = 'bearish'

            results['bullish_score'] = round(bullish_score, 2)
            results['bearish_score'] = round(bearish_score, 2)

            return results

        except Exception as e:
            logger.error(f"Error analyzing price action: {e}", exc_info=False)
            results['status'] = 'error'
            results['message'] = str(e)
            return results

    def calculate_risk_reward(self, symbol: str, direction: str, current_price: float,
                              analysis_results: Dict[str, Dict[str, Any]],
                              adapted_risk_config: Dict[str, Any]) -> Dict[str, float]:
        """Calculate optimal stop loss and take profit levels with adaptive parameters."""
        # Get adaptive SL percent if adaptive learning is enabled
        if self.adaptive_learning.enabled:
            base_sl_percent = adapted_risk_config.get('default_stop_loss_percent', self.base_default_sl_percent)
            # Determine primary timeframe for SL adaptation
            tf = self._determine_primary_timeframe(analysis_results)
            if tf:
                adapted_sl_percent = self.adaptive_learning.get_adaptive_sl_percent(
                    base_sl_percent, symbol, tf, direction
                )
                adapted_risk_config['default_stop_loss_percent'] = adapted_sl_percent

        default_sl_percent = adapted_risk_config.get('default_stop_loss_percent', self.base_default_sl_percent)
        preferred_rr = adapted_risk_config.get('preferred_risk_reward_ratio', self.base_preferred_risk_reward_ratio)
        min_rr = adapted_risk_config.get('min_risk_reward_ratio', self.base_min_risk_reward_ratio)

        try:
            # Find valid timeframes and select highest
            valid_tfs = [tf for tf, res in analysis_results.items() if res.get('status') == 'ok']
            if not valid_tfs:
                raise ValueError("No valid analysis results for RR calculation")

            # Select highest valid timeframe (with highest weight)
            highest_tf = max(valid_tfs, key=lambda x: self.timeframe_weights.get(x, 0))

            # Get price action and support/resistance results
            pa_result = analysis_results[highest_tf].get('price_action', {})
            sr_result = analysis_results[highest_tf].get('support_resistance', {}).get('details', {})
            channel_result = analysis_results[highest_tf].get('price_channels', {})
            channel_found = channel_result.get('status') == 'ok' and len(channel_result.get('channels', [])) > 0
            harmonic_patterns = analysis_results[highest_tf].get('harmonic_patterns', [])
            harmonic_found = len(harmonic_patterns) > 0

            stop_loss = None
            take_profit = None
            calculation_method = "None"

            # Get ATR for dynamic adjustments
            atr = pa_result.get('atr')
            atr = atr if atr and atr > 0 else current_price * 0.005  # Default meaningful ATR value

            # 1. First priority: use harmonic patterns for SL/TP
            if harmonic_found and direction in ['long', 'short']:
                best_pattern = sorted(harmonic_patterns, key=lambda x: x.get('confidence', 0), reverse=True)[0]
                pattern_type = best_pattern.get('type', '')
                pattern_direction = best_pattern.get('direction', '')

                if (direction == 'long' and pattern_direction == 'bullish') or (
                        direction == 'short' and pattern_direction == 'bearish'):
                    pattern_points = best_pattern.get('points', {})
                    if 'D' in pattern_points and 'X' in pattern_points:
                        d_point = pattern_points['D']
                        x_point = pattern_points['X']

                        if direction == 'long':
                            stop_loss = d_point.get('price', 0) * 0.99  # Slightly below D point (entry)
                            # Target based on pattern type
                            if 'butterfly' in pattern_type or 'crab' in pattern_type:
                                take_profit = current_price + (current_price - stop_loss) * 1.618  # Higher target
                            else:
                                take_profit = x_point.get('price', 0)  # Target to X point

                            calculation_method = f"Harmonic_{pattern_type}"

                        elif direction == 'short':
                            stop_loss = d_point.get('price', 0) * 1.01  # Slightly above D point (entry)
                            if 'butterfly' in pattern_type or 'crab' in pattern_type:
                                take_profit = current_price - (stop_loss - current_price) * 1.618  # Higher target
                            else:
                                take_profit = x_point.get('price', 0)  # Target to X point

                            calculation_method = f"Harmonic_{pattern_type}"

            # 2. Second priority: use price channels if found
            if stop_loss is None and channel_found:
                channel = channel_result.get('channels', [])[0]

                if direction == 'long':
                    if channel.get('direction') in ['ascending', 'horizontal']:
                        lower_line_current = channel.get('lower_slope', 0) * (
                                len(pa_result.get('details', {}).get('candle_patterns', [])) - 1) + channel.get(
                            'lower_intercept', 0)
                        stop_loss = lower_line_current * 0.99  # Slightly below lower channel line

                        # Target to upper channel line
                        upper_line_current = channel.get('upper_slope', 0) * (
                                len(pa_result.get('details', {}).get('candle_patterns', [])) - 1) + channel.get(
                            'upper_intercept', 0)
                        take_profit = upper_line_current * 0.99

                        calculation_method = f"Price_Channel_{channel.get('direction')}"

                elif direction == 'short':
                    if channel.get('direction') in ['descending', 'horizontal']:
                        upper_line_current = channel.get('upper_slope', 0) * (
                                len(pa_result.get('details', {}).get('candle_patterns', [])) - 1) + channel.get(
                            'upper_intercept', 0)
                        stop_loss = upper_line_current * 1.01  # Slightly above upper channel line

                        # Target to lower channel line
                        lower_line_current = channel.get('lower_slope', 0) * (
                                len(pa_result.get('details', {}).get('candle_patterns', [])) - 1) + channel.get(
                            'lower_intercept', 0)
                        take_profit = lower_line_current * 1.01

                        calculation_method = f"Price_Channel_{channel.get('direction')}"

            # 3. Third priority: use support/resistance levels
            nearest_resist = sr_result.get('nearest_resistance', {}).get('price') if isinstance(
                sr_result.get('nearest_resistance'), dict) else sr_result.get('nearest_resistance')
            nearest_support = sr_result.get('nearest_support', {}).get('price') if isinstance(
                sr_result.get('nearest_support'), dict) else sr_result.get('nearest_support')

            if stop_loss is None:
                if direction == 'long' and nearest_support and nearest_support < current_price:
                    stop_loss = nearest_support * 0.999
                    calculation_method = "Support Level"
                elif direction == 'short' and nearest_resist and nearest_resist > current_price:
                    stop_loss = nearest_resist * 1.001
                    calculation_method = "Resistance Level"

            # Check if S/R is too far and use ATR instead
            is_sl_too_far = False
            if stop_loss is not None and atr > 0:
                sl_dist_atr_ratio = abs(current_price - stop_loss) / atr
                if sl_dist_atr_ratio > 3.0:
                    is_sl_too_far = True
                    stop_loss = None

            # Use ATR if no S/R levels or they're too far
            if stop_loss is None and atr > 0:
                sl_multiplier = adapted_risk_config.get('atr_trailing_multiplier', 2.0)
                if direction == 'long':
                    stop_loss = current_price - (atr * sl_multiplier)
                else:
                    stop_loss = current_price + (atr * sl_multiplier)
                calculation_method = f"ATR x{sl_multiplier}"

            # Default percentage as last resort
            if stop_loss is None:
                if direction == 'long':
                    stop_loss = current_price * (1 - default_sl_percent / 100)
                else:
                    stop_loss = current_price * (1 + default_sl_percent / 100)
                calculation_method = f"Percentage {default_sl_percent}%"

            # Safety checks for SL
            min_sl_distance = atr * 0.5 if atr > 0 else current_price * 0.001
            if direction == 'long' and (current_price - stop_loss) < min_sl_distance:
                original_sl = stop_loss
                stop_loss = current_price - min_sl_distance
                calculation_method = f"Minimum Distance (was {original_sl:.6f})"
            elif direction == 'short' and (stop_loss - current_price) < min_sl_distance:
                original_sl = stop_loss
                stop_loss = current_price + min_sl_distance
                calculation_method = f"Minimum Distance (was {original_sl:.6f})"

            # Calculate final risk distance
            risk_distance = abs(current_price - stop_loss)
            if risk_distance <= 1e-6:
                if self.signal_generated:
                    logger.warning(f"Risk distance too small ({risk_distance}) for {symbol}. Using default percentage.")
                risk_distance = current_price * (default_sl_percent / 100)
                if direction == 'long':
                    stop_loss = current_price - risk_distance
                else:
                    stop_loss = current_price + risk_distance

            # Calculate TP if not set yet
            if take_profit is None:
                reward_distance = risk_distance * preferred_rr
                reward_distance = max(reward_distance, current_price * 0.001)  # Ensure non-zero reward distance

                if direction == 'long':
                    take_profit = current_price + reward_distance
                else:
                    take_profit = current_price - reward_distance

                # Adjust TP based on nearby S/R
                if direction == 'long' and nearest_resist and nearest_resist < take_profit:
                    if nearest_resist > current_price + (risk_distance * min_rr):
                        take_profit = nearest_resist * 0.999
                    else:
                        if self.signal_generated:
                            logger.warning(
                                f"Nearest resistance {nearest_resist} would make TP too close, keeping calculated TP.")
                elif direction == 'short' and nearest_support and nearest_support > take_profit:
                    if nearest_support < current_price - (risk_distance * min_rr):
                        take_profit = nearest_support * 1.001
                    else:
                        if self.signal_generated:
                            logger.warning(
                                f"Nearest support {nearest_support} would make TP too close, keeping calculated TP.")

            # Safety checks for TP
            if direction == 'long' and take_profit <= current_price + (risk_distance * min_rr * 0.9):
                if self.signal_generated:
                    logger.warning(
                        f"Calculated TP {take_profit} for LONG {symbol} does not meet min RR ({min_rr}). Adjusting TP.")
                take_profit = current_price + (risk_distance * min_rr)
            elif direction == 'short' and take_profit >= current_price - (risk_distance * min_rr * 0.9):
                if self.signal_generated:
                    logger.warning(
                        f"Calculated TP {take_profit} for SHORT {symbol} does not meet min RR ({min_rr}). Adjusting TP.")
                take_profit = current_price - (risk_distance * min_rr)

            # Calculate final RR
            final_reward_distance = abs(take_profit - current_price)
            final_rr = final_reward_distance / risk_distance

            # Ensure TP and SL are not zero
            if abs(take_profit) < 1e-6:
                logger.error(f"Calculated TP for {symbol} is near zero! Using minimum viable TP.")
                take_profit = current_price * (1.05 if direction == 'long' else 0.95)

            if abs(stop_loss) < 1e-6:
                logger.error(f"Calculated SL for {symbol} is near zero! Using minimum viable SL.")
                stop_loss = current_price * (0.95 if direction == 'long' else 1.05)

            precision = 8  # Higher precision to prevent rounding to zero
            return {
                'stop_loss': round(stop_loss, precision),
                'take_profit': round(take_profit, precision),
                'risk_reward_ratio': round(final_rr, 2),
                'risk_amount_per_unit': round(risk_distance, precision),
                'sl_method': calculation_method
            }

        except Exception as e:
            logger.error(f"Error calculating risk/reward for {symbol}: {e}", exc_info=True)
            # Default values in case of error
            if direction == 'long':
                stop_loss = current_price * (1 - default_sl_percent / 100)
                take_profit = current_price * (1 + (default_sl_percent * min_rr) / 100)
            else:
                stop_loss = current_price * (1 + default_sl_percent / 100)
                take_profit = current_price * (1 - (default_sl_percent * min_rr) / 100)

            precision = 8
            return {
                'stop_loss': round(stop_loss, precision),
                'take_profit': round(take_profit, precision),
                'risk_reward_ratio': min_rr,
                'risk_amount_per_unit': round(current_price * (default_sl_percent / 100), precision),
                'sl_method': 'Error Fallback'
            }

    def _determine_primary_timeframe(self, analysis_results: Dict[str, Dict[str, Any]]) -> Optional[str]:
        """Determine primary timeframe from valid analysis results"""
        valid_tfs = [tf for tf, res in analysis_results.items() if res.get('status') == 'ok']
        if not valid_tfs:
            return None

        # Sort by minutes (smallest timeframe)
        sorted_tfs = sorted(valid_tfs, key=self._get_tf_minutes)
        return sorted_tfs[0] if sorted_tfs else None

    def _get_tf_minutes(self, timeframe: str) -> int:
        """Convert timeframe to minutes for sorting."""
        try:
            if 'm' in timeframe:
                return int(timeframe.replace('m', ''))
            elif 'h' in timeframe:
                return int(timeframe.replace('h', '')) * 60
            elif 'd' in timeframe:
                return int(timeframe.replace('d', '')) * 1440
            elif 'w' in timeframe:
                return int(timeframe.replace('w', '')) * 10080
            else:
                return 99999  # Large value for unknown
        except ValueError:
            return 99999

    def analyze_higher_timeframe_structure(self, df: pd.DataFrame, tf: str,
                                           analysis_results: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
        """Analyze higher timeframe structure for confirmation."""
        results = {'status': 'ok', 'score': 1.0, 'details': {}}
        if not self.htf_enabled or df is None or tf is None:
            results['status'] = 'disabled_or_missing_data'
            return results

        try:
            # Determine higher timeframe to check
            htf = None

            if self.htf_timeframe_method == 'next_higher':
                current_minutes = self._get_tf_minutes(tf)
                valid_tfs = [t for t in self.timeframes if self._get_tf_minutes(t) > current_minutes]
                if valid_tfs:
                    htf = min(valid_tfs, key=self._get_tf_minutes)
            elif self.htf_timeframe_method == 'fixed':
                if tf != self.htf_fixed_tf1:
                    htf = self.htf_fixed_tf1
                elif tf != self.htf_fixed_tf2:
                    htf = self.htf_fixed_tf2

            if not htf or htf not in analysis_results:
                htf = max(self.timeframes, key=self._get_tf_minutes)
                if htf == tf or htf not in analysis_results:
                    results['status'] = 'no_higher_timeframe'
                    return results

            htf_result = analysis_results[htf]
            if htf_result.get('status') != 'ok':
                results['status'] = 'missing_htf_analysis'
                return results

            # Get current trend and price action data
            current_trend = analysis_results[tf].get('trend', {})
            htf_trend = htf_result.get('trend', {})
            current_momentum = analysis_results[tf].get('momentum', {})
            htf_momentum = htf_result.get('momentum', {})
            htf_sr = htf_result.get('support_resistance', {})

            # Check trend alignment across timeframes
            current_trend_dir = current_trend.get('trend', 'neutral')
            htf_trend_dir = htf_trend.get('trend', 'neutral')
            current_momentum_dir = current_momentum.get('direction', 'neutral')
            htf_momentum_dir = htf_momentum.get('direction', 'neutral')

            # Identify market phase
            current_trend_phase = current_trend.get('phase', 'unknown')
            htf_trend_phase = htf_trend.get('phase', 'unknown')

            # Check if trends align and calculate score modifier
            trends_aligned = False
            if ('bullish' in current_trend_dir and 'bullish' in htf_trend_dir) or (
                    'bearish' in current_trend_dir and 'bearish' in htf_trend_dir):
                trends_aligned = True

            # Check if momentum aligns
            momentum_aligned = current_momentum_dir == htf_momentum_dir

            # Get HTF support/resistance zones
            htf_resistance_zones = htf_sr.get('resistance_zones', {}).get('zones', [])
            htf_support_zones = htf_sr.get('support_zones', {}).get('zones', [])

            # Get current price
            current_price = df['close'].iloc[-1]

            # Check distance to nearest HTF zones
            nearest_htf_resistance = None
            nearest_htf_support = None
            nearest_resist_distance = float('inf')
            nearest_support_distance = float('inf')

            for zone in htf_resistance_zones:
                dist = abs(zone.get('center', 0) - current_price)
                if dist < nearest_resist_distance:
                    nearest_resist_distance = dist
                    nearest_htf_resistance = zone

            for zone in htf_support_zones:
                dist = abs(zone.get('center', 0) - current_price)
                if dist < nearest_support_distance:
                    nearest_support_distance = dist
                    nearest_htf_support = zone

            # Check position relative to HTF zones
            price_above_support = False
            price_below_resistance = False
            at_support_zone = False
            at_resistance_zone = False

            if nearest_htf_support:
                support_center = nearest_htf_support.get('center', 0)
                support_width = nearest_htf_support.get('width', 0)
                price_above_support = current_price > support_center
                at_support_zone = abs(current_price - support_center) <= support_width / 2

            if nearest_htf_resistance:
                resistance_center = nearest_htf_resistance.get('center', 0)
                resistance_width = nearest_htf_resistance.get('width', 0)
                price_below_resistance = current_price < resistance_center
                at_resistance_zone = abs(current_price - resistance_center) <= resistance_width / 2

            # Calculate structure score
            base_score = self.htf_score_config['base']
            structure_score = base_score

            # Adjust score based on trend alignment
            if trends_aligned:
                structure_score += self.htf_score_config['confirm_bonus']
                structure_score *= (1 + self.htf_score_config['trend_bonus_mult'] * (min(
                    abs(current_trend.get('strength', 0)), abs(htf_trend.get('strength', 0))) / 3))
            else:
                structure_score -= self.htf_score_config['contradict_penalty']
                structure_score *= (1 - self.htf_score_config['trend_penalty_mult'] * (min(
                    abs(current_trend.get('strength', 0)), abs(htf_trend.get('strength', 0))) / 3))

            # Adjust score based on momentum alignment
            if momentum_aligned:
                structure_score *= 1.05
            else:
                structure_score *= 0.95

            # Adjust score based on position relative to HTF S/R
            if 'bullish' in current_trend_dir and price_above_support and price_below_resistance:
                structure_score *= 1.1
            elif 'bearish' in current_trend_dir and price_below_resistance and price_above_support:
                structure_score *= 1.1

            # Bonus for being at HTF S/R zones
            if 'bullish' in current_trend_dir and at_support_zone:
                structure_score *= 1.2
            elif 'bearish' in current_trend_dir and at_resistance_zone:
                structure_score *= 1.2

            # Limit the structure score
            structure_score = max(min(structure_score, self.htf_score_config['max_score']),
                                  self.htf_score_config['min_score'])

            # Create detailed results
            results['score'] = round(structure_score, 2)
            results['htf'] = htf
            results['aligned'] = trends_aligned
            results['details'] = {
                'trends_aligned': trends_aligned,
                'momentum_aligned': momentum_aligned,
                'tf_trend': current_trend_dir,
                'htf_trend': htf_trend_dir,
                'tf_momentum': current_momentum_dir,
                'htf_momentum': htf_momentum_dir,
                'tf_phase': current_trend_phase,
                'htf_phase': htf_trend_phase,
                'at_htf_support': at_support_zone,
                'at_htf_resistance': at_resistance_zone
            }
            if nearest_htf_support:
                results['details']['nearest_htf_support'] = nearest_htf_support
            if nearest_htf_resistance:
                results['details']['nearest_htf_resistance'] = nearest_htf_resistance
            return results

        except Exception as e:
            logger.error(f"Error analyzing higher timeframe structure: {e}", exc_info=True)
            results['status'] = 'error'
            results['message'] = str(e)
            return results

    def analyze_volatility_conditions(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Analyze volatility conditions for filtering signals."""
        results = {'status': 'ok', 'score': 1.0, 'condition': 'normal', 'details': {}}
        if not self.vol_enabled or df is None or len(df) < max(self.vol_atr_period, self.vol_atr_ma_period) + 10:
            results['status'] = 'disabled_or_insufficient_data'
            return results

        try:
            # Get ATR and normalized ATR
            high_p = df['high'].values.astype(np.float64)
            low_p = df['low'].values.astype(np.float64)
            close_p = df['close'].values.astype(np.float64)

            atr = talib.ATR(high_p, low_p, close_p, timeperiod=self.vol_atr_period)
            # Filter out invalid values
            valid_atr = atr[~np.isnan(atr)]
            if len(valid_atr) < self.vol_atr_ma_period:
                results['status'] = 'insufficient_valid_data'
                return results

            # Calculate normalized ATR (ATR/Price)
            valid_close_p = close_p[-len(valid_atr):]
            atr_pct = (valid_atr / valid_close_p) * 100

            # Calculate MA of normalized ATR
            atr_pct_ma = np.zeros_like(atr_pct)
            if use_bottleneck:
                atr_pct_ma = bn.move_mean(atr_pct, window=self.vol_atr_ma_period, min_count=1)
            else:
                # Manual calculation as fallback
                for i in range(len(atr_pct)):
                    start_idx = max(0, i - self.vol_atr_ma_period + 1)
                    atr_pct_ma[i] = np.mean(atr_pct[start_idx:i + 1])

            # Get current normalized ATR and MA
            current_atr_pct = atr_pct[-1]
            current_atr_pct_ma = atr_pct_ma[-1]

            # Calculate volatility ratio
            volatility_ratio = current_atr_pct / current_atr_pct_ma if current_atr_pct_ma > 0 else 1.0

            # Determine volatility condition
            vol_condition = 'normal'
            vol_score = 1.0

            if volatility_ratio > self.vol_extreme_thresh:
                vol_condition = 'extreme'
                vol_score = self.vol_scores.get('extreme', 0.5)
            elif volatility_ratio > self.vol_high_thresh:
                vol_condition = 'high'
                vol_score = self.vol_scores.get('high', 0.8)
            elif volatility_ratio < self.vol_low_thresh:
                vol_condition = 'low'
                vol_score = self.vol_scores.get('low', 0.9)

            reject_due_to_extreme = vol_condition == 'extreme' and self.vol_reject_extreme
            results.update({
                'score': vol_score,
                'condition': vol_condition,
                'reject': reject_due_to_extreme,
                'volatility_ratio': round(volatility_ratio, 2),
                'details': {
                    'current_atr_pct': round(current_atr_pct, 3),
                    'average_atr_pct': round(current_atr_pct_ma, 3),
                    'raw_atr': round(valid_atr[-1], 5)
                }
            })

            return results
        except Exception as e:
            logger.error(f"Error analyzing volatility conditions: {e}", exc_info=True)
            results['status'] = 'error'
            results['message'] = str(e)
            return results

    def _analyze_macd(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Comprehensive MACD analysis with divergence detection, trendlines, and histogram patterns."""
        results = {'status': 'ok', 'signals': [], 'direction': 'neutral', 'market_type': 'unknown', 'details': {}}
        if df is None or len(df) < 50:
            results['status'] = 'insufficient_data'
            return results

        try:
            symbol = df.attrs.get('symbol', 'unknown')
            timeframe = df.attrs.get('timeframe', 'unknown')
            close_p = df['close'].values.astype(np.float64)

            # Get EMA20 and EMA50 for market type detection
            ema20_key = self._cache_key(symbol, timeframe, 'EMA', (20,))
            ema50_key = self._cache_key(symbol, timeframe, 'EMA', (50,))

            ema20 = self._get_cached_indicator(ema20_key)
            if ema20 is None:
                ema20 = talib.EMA(close_p, timeperiod=20)
                self._cache_indicator(ema20_key, ema20)

            ema50 = self._get_cached_indicator(ema50_key)
            if ema50 is None:
                ema50 = talib.EMA(close_p, timeperiod=50)
                self._cache_indicator(ema50_key, ema50)

            # Get MACD, Signal, and Histogram
            macd_key = self._cache_key(symbol, timeframe, 'MACD', (12, 26, 9))
            cached_macd = self._get_cached_indicator(macd_key)
            if cached_macd is not None:
                dif, dea, hist = cached_macd
            else:
                dif, dea, hist = talib.MACD(close_p, fastperiod=12, slowperiod=26, signalperiod=9)
                self._cache_indicator(macd_key, (dif, dea, hist))

            # Convert to pandas series for easier indexing
            dif_s = pd.Series(dif)
            dea_s = pd.Series(dea)
            hist_s = pd.Series(hist)
            ema20_s = pd.Series(ema20)
            ema50_s = pd.Series(ema50)

            # Detect market type based on MACD and EMAs
            market_type = self._detect_macd_market_type(dif_s, hist_s, ema20_s, ema50_s)

            # Detect MACD crosses (golden/death)
            macd_crosses = self._detect_detailed_macd_crosses(dif_s, dea_s, df.index)

            # Detect DIF line behavior (zero crosses, trendline breaks)
            dif_behavior = self._detect_dif_behavior(dif_s, df.index)

            # Analyze MACD histogram features
            hist_analysis = self._analyze_macd_histogram(hist_s, pd.Series(close_p, index=df.index), df.index)

            # Detect MACD divergence
            macd_divergence = self._detect_divergence_generic(
                pd.Series(close_p, index=df.index), dif_s, 'macd')

            # Combine all MACD signals
            all_signals = macd_crosses + dif_behavior + hist_analysis + macd_divergence

            # Calculate direction and scores
            bullish_score = sum(
                s.get('score', 0) for s in all_signals if s.get('direction', '') == 'bullish')
            bearish_score = sum(
                s.get('score', 0) for s in all_signals if s.get('direction', '') == 'bearish')

            direction = 'neutral'
            if bullish_score > bearish_score * 1.1:
                direction = 'bullish'
            elif bearish_score > bullish_score * 1.1:
                direction = 'bearish'

            # Prepare current values for details
            last_valid_idx = -1
            while last_valid_idx >= -len(df) and (
                    np.isnan(dif[last_valid_idx]) or np.isnan(dea[last_valid_idx]) or np.isnan(hist[last_valid_idx])):
                last_valid_idx -= 1

            if abs(last_valid_idx) <= len(df):
                curr_dif = dif[last_valid_idx]
                curr_dea = dea[last_valid_idx]
                curr_hist = hist[last_valid_idx]

                hist_slope = hist[last_valid_idx] - hist[max(0, last_valid_idx - 3)]
                dif_slope = dif[last_valid_idx] - dif[max(0, last_valid_idx - 3)]
                dea_slope = dea[last_valid_idx] - dea[max(0, last_valid_idx - 3)]

                results['details'] = {
                    'dif': round(float(curr_dif), 6),
                    'dea': round(float(curr_dea), 6),
                    'hist': round(float(curr_hist), 6),
                    'dif_slope': round(float(dif_slope), 6) if dif_slope is not None else 0,
                    'dea_slope': round(float(dea_slope), 6) if dea_slope is not None else 0,
                    'hist_slope': round(float(hist_slope), 6) if hist_slope is not None else 0,
                    'market_type': market_type
                }

            # Update results
            results['signals'] = all_signals
            results['direction'] = direction
            results['market_type'] = market_type
            results['bullish_score'] = round(bullish_score, 2)
            results['bearish_score'] = round(bearish_score, 2)

            return results

        except Exception as e:
            logger.error(f"Error in MACD analysis: {e}", exc_info=True)
            results['status'] = 'error'
            results['message'] = str(e)
            return results

    async def analyze_single_timeframe(self, symbol: str, timeframe: str, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Analyze a single timeframe with error handling and caching.

        Args:
            symbol: Currency symbol
            timeframe: Timeframe
            df: Price dataframe

        Returns:
            Analysis results for this timeframe
        """
        # Check cache
        cache_key = f"{symbol}_{timeframe}_{id(df)}"
        if cache_key in self._analysis_cache:
            cached_result, timestamp = self._analysis_cache[cache_key]
            if time.time() - timestamp < self._cache_ttl_seconds:
                return cached_result

        # Set attributes for caching indicators
        df.attrs['symbol'] = symbol
        df.attrs['timeframe'] = timeframe

        # Prepare initial result
        results = {
            'symbol': symbol,
            'timeframe': timeframe,
            'status': 'error',
            'timestamp': pd.Timestamp.now(tz='UTC')
        }

        # Check for sufficient data
        if df is None or df.empty or len(df) < 50:  # Minimum data for most analyses
            results['status'] = 'insufficient_data'
            return results

        try:
            # Start analyses
            analysis_data = {}

            # 1. Detect trend
            logger.debug(f"Detecting trend for {symbol} {timeframe}...")
            analysis_data['trend'] = self.detect_trend(df)

            # 2. Analyze momentum
            logger.debug(f"Analyzing momentum for {symbol} {timeframe}...")
            analysis_data['momentum'] = self.analyze_momentum_indicators(df)

            # 3. Analyze volume
            logger.debug(f"Analyzing volume for {symbol} {timeframe}...")
            analysis_data['volume'] = self.analyze_volume_trend(df)

            # 4. Advanced MACD analysis
            logger.debug(f"Analyzing MACD for {symbol} {timeframe}...")
            analysis_data['macd'] = self._analyze_macd(df)

            # 5. Analyze price action
            logger.debug(f"Analyzing price action for {symbol} {timeframe}...")
            # Note: analyze_price_action is an async function and must be awaited
            analysis_data['price_action'] = await self.analyze_price_action(df)

            # 6. Detect support/resistance
            logger.debug(f"Detecting support/resistance for {symbol} {timeframe}...")
            analysis_data['support_resistance'] = self.detect_support_resistance(df)

            # 7. Detect harmonic patterns (if enabled)
            if self.harmonic_enabled:
                logger.debug(f"Detecting harmonic patterns for {symbol} {timeframe}...")
                analysis_data['harmonic_patterns'] = self.detect_harmonic_patterns(df)

            # 8. Detect price channels (if enabled)
            if self.channel_enabled:
                logger.debug(f"Detecting price channels for {symbol} {timeframe}...")
                analysis_data['price_channels'] = self.detect_price_channels(df)

            # 9. Detect cyclical patterns (if enabled)
            if self.cycle_enabled:
                logger.debug(f"Detecting cyclical patterns for {symbol} {timeframe}...")
                analysis_data['cyclical_patterns'] = self.detect_cyclical_patterns(df)

            # 10. Analyze volatility conditions
            if self.vol_enabled:
                logger.debug(f"Analyzing volatility for {symbol} {timeframe}...")
                analysis_data['volatility'] = self.analyze_volatility_conditions(df)

            # 11. Detect market regime
            if self.regime_detector.enabled:
                try:
                    logger.debug(f"Detecting market regime for {symbol} {timeframe}...")
                    regime_result = self.regime_detector.detect_regime(df)

                    # Check for valid output
                    if regime_result is None:
                        logger.warning(f"Market regime detection returned None for {symbol} {timeframe}")
                        # Default valid value
                        analysis_data['market_regime'] = {
                            'regime': 'unknown',
                            'trend_strength': 'unknown',
                            'trend_direction': 'neutral',
                            'volatility': 'normal',
                            'confidence': 0.0,
                            'details': {'error': 'Regime detection returned None'}
                        }
                    else:
                        analysis_data['market_regime'] = regime_result
                except Exception as e:
                    logger.error(f"Error in market regime detection for {symbol} {timeframe}: {e}", exc_info=True)
                    # Default value in case of error
                    analysis_data['market_regime'] = {
                        'regime': 'error',
                        'trend_strength': 'unknown',
                        'trend_direction': 'neutral',
                        'volatility': 'normal',
                        'confidence': 0.0,
                        'details': {'error': str(e)}
                    }

            # Check overall status of results
            all_statuses = [res.get('status') for res in analysis_data.values() if
                            isinstance(res, dict) and 'status' in res]
            if all(s == 'ok' or s == 'insufficient_data' for s in all_statuses):
                results['status'] = 'ok'
            else:
                # Log which analyses failed
                failed_analyses = {
                    k: v.get('status')
                    for k, v in analysis_data.items()
                    if isinstance(v, dict) and 'status' in v and v.get('status') not in ['ok', 'insufficient_data']
                }
                # logger.warning(f"Analyses failed for {symbol} {timeframe}: {failed_analyses}")

            # Add analysis results to final result
            results.update(analysis_data)

            # Store in cache
            self._analysis_cache[cache_key] = (results, time.time())

            return results

        except Exception as e:
            # Log error details
            logger.error(f"Critical error during analysis of {symbol} on {timeframe}: {e}", exc_info=True)
            results['error'] = str(e)
            return results


    def _get_trend_phase_multiplier(self, phase: str, direction: str) -> float:
        """Calculate score adjustment factor based on trend phase"""
        # Different phases have different multipliers
        phase_multipliers = {
            'early': 1.2,  # Early trend - higher score
            'developing': 1.1,  # Developing trend - good score
            'mature': 0.9,  # Mature trend - somewhat cautious
            'late': 0.7,  # Late trend - cautious
            'pullback': 1.1,  # Pullback in trend - good entry opportunity
            'transition': 0.8,  # Transition between trends - ambiguous
            'undefined': 1.0  # Undefined - no change
        }

        return phase_multipliers.get(phase, 1.0)

    def _calculate_timeframe_alignment(self, trend_directions: Dict[str, str],
                                       momentum_directions: Dict[str, str],
                                       macd_directions: Dict[str, str],
                                       final_direction: str) -> float:
        """Calculate the alignment level across timeframes and indicators"""
        if not trend_directions or final_direction == 'neutral':
            return 1.0

        # Count timeframes aligned with final direction
        aligned_trend_count = 0
        total_trend_count = 0
        aligned_momentum_count = 0
        total_momentum_count = 0
        aligned_macd_count = 0
        total_macd_count = 0

        for tf, direction in trend_directions.items():
            total_trend_count += 1
            if (final_direction == 'bullish' and 'bullish' in direction) or (
                    final_direction == 'bearish' and 'bearish' in direction):
                aligned_trend_count += 1

        for tf, direction in momentum_directions.items():
            total_momentum_count += 1
            if direction == final_direction:
                aligned_momentum_count += 1

        for tf, direction in macd_directions.items():
            total_macd_count += 1
            if direction == final_direction:
                aligned_macd_count += 1

        # Calculate overall alignment factor (0.7 to 1.3)
        total_count = total_trend_count + total_momentum_count + total_macd_count
        aligned_count = aligned_trend_count + aligned_momentum_count + aligned_macd_count

        if total_count == 0:
            return 1.0

        # Trend alignment is weighted more than momentum or MACD
        weighted_alignment = (
                (aligned_trend_count / total_trend_count) * 0.5 +
                (aligned_momentum_count / total_momentum_count) * 0.3 +
                (aligned_macd_count / total_macd_count) * 0.2
        ) if total_trend_count > 0 and total_momentum_count > 0 and total_macd_count > 0 else (
                aligned_count / total_count)

        # Convert to factor between 0.7 and 1.3
        return 0.7 + (weighted_alignment * 0.6)

    async def analyze_symbol(self, symbol: str, timeframes_data: Dict[str, Optional[pd.DataFrame]]) -> Optional[
        SignalInfo]:
        """
        Complete symbol analysis across all timeframes with market regime adaptation and SL/TP calculation.

        Args:
            symbol: Currency symbol
            timeframes_data: Dictionary of timeframe data {timeframe: df}

        Returns:
            Complete SignalInfo with final SL/TP and adapted config, or None
        """
        try:
            # Check if emergency circuit breaker is active
            if self.circuit_breaker.enabled:
                is_active, reason = self.circuit_breaker.check_if_active()
                if is_active:
                    logger.warning(f"Circuit breaker active: {reason}. Skipping signal generation for {symbol}.")
                    return None
                # Optional: Market volatility check can stay here if needed
                # risk_reduction_factor = 1.0 if not self.circuit_breaker.is_market_volatile(timeframes_data) else 0.7
            else:
                risk_reduction_factor = 1.0

            # Update correlations if enabled (Consider timing and necessity)
            if self.correlation_manager.enabled:
                # self.correlation_manager.update_correlations(timeframes_data) # Might be slow, update less frequently?
                pass  # Assuming correlation updated elsewhere or less frequently

            # 0. Filter valid data (DataFrames, not analysis results yet)
            valid_tf_data = {
                tf: df for tf, df in timeframes_data.items()
                if isinstance(df, pd.DataFrame) and not df.empty and len(df) >= 50
            }

            if not valid_tf_data:
                logger.debug(f"No valid/sufficient DataFrame provided for {symbol} in any timeframe.")
                return None

            # 1. Analyze each valid timeframe to gather analysis results and base signals
            analysis_results: Dict[str, Dict[str, Any]] = {}
            base_signals: Dict[str, Dict[str, Any]] = {}
            all_tf_tasks = []

            async def run_analysis(tf, df):
                """Helper async function to run analysis for one timeframe."""
                result = await self.analyze_single_timeframe(symbol, tf, df)
                analysis_results[tf] = result
                if isinstance(result, dict) and result.get('status') == 'ok':
                    # Extract base signal (adjust extraction logic as needed)
                    base_signal_score = 0
                    base_direction = 'neutral'
                    pa_res = result.get('price_action', {})
                    mom_res = result.get('momentum', {})
                    pa_score = pa_res.get('bullish_score', 0) - pa_res.get('bearish_score', 0)
                    mom_score = mom_res.get('bullish_score', 0) - mom_res.get('bearish_score', 0)

                    # Prioritize stronger signal (price action or momentum)
                    if abs(pa_score) >= abs(mom_score):
                        base_signal_score = pa_score
                        base_direction = 'bullish' if pa_score > 0 else ('bearish' if pa_score < 0 else 'neutral')
                    elif abs(mom_score) > 0:
                        base_signal_score = mom_score
                        base_direction = 'bullish' if mom_score > 0 else ('bearish' if mom_score < 0 else 'neutral')

                    if base_direction != 'neutral':
                        base_signals[tf] = {
                            'final_score': abs(base_signal_score),
                            'direction': base_direction
                        }

            for tf, df in valid_tf_data.items():
                all_tf_tasks.append(run_analysis(tf, df))

            await asyncio.gather(*all_tf_tasks, return_exceptions=True)

            # Filter successful analysis results (dictionaries with status 'ok')
            successful_analysis_results = {
                tf: res for tf, res in analysis_results.items()
                if isinstance(res, dict) and res.get('status') == 'ok'
            }

            if not successful_analysis_results:
                # logger.warning(f"No successful analysis results for {symbol} after running all timeframes.")
                return None

            # 2. Detect market regime
            regime_info = {'regime': 'unknown', 'confidence': 0.0, 'details': {}}
            if self.regime_detector.enabled:
                key_func = lambda x: self.timeframe_weights.get(x, 0)
                tf_for_regime = max(successful_analysis_results.keys(), key=key_func, default=None)
                if tf_for_regime and tf_for_regime in valid_tf_data:
                    # Pass the DataFrame to detect_regime
                    regime_info = self.regime_detector.detect_regime(valid_tf_data[tf_for_regime])

            # 3. Get adapted parameters
            adapted_config = self.regime_detector.get_adapted_parameters(regime_info, self.config)
            adapted_risk_config = adapted_config.get('risk_management', self.risk_config)
            adapted_signal_config = adapted_config.get('signal_generation', self.signal_config)

            # 4. Calculate multi-timeframe score, passing ANALYSIS RESULTS, BASE SIGNALS, and TIMEFRAMES DATA (DataFrames)
            score_result = self.calculate_multi_timeframe_score(
                symbol,
                successful_analysis_results,
                base_signals,
                valid_tf_data
            )

            final_direction = score_result.get('final_direction', 'neutral')
            bullish_score = score_result.get('final_bullish_score', 0)
            bearish_score = score_result.get('final_bearish_score', 0)

            if score_result.get('volatility_rejection', False):
                logger.info(f"Rejected signal for {symbol} due to extreme volatility.")
                return None

            if final_direction == 'neutral' or final_direction == 'error':
                logger.debug(
                    f"No clear direction for {symbol}: Bull={bullish_score:.2f}, Bear={bearish_score:.2f}, Dir={final_direction}")
                return None

            valid_tfs_sorted = sorted(successful_analysis_results.keys(), key=self._get_tf_minutes)
            if not valid_tfs_sorted:
                return None

            primary_tf = valid_tfs_sorted[0]
            primary_df = valid_tf_data[primary_tf]
            current_price = primary_df['close'].iloc[-1]
            last_idx = primary_df.index[-1]
            signal_timestamp = last_idx.to_pydatetime() if hasattr(last_idx, 'to_pydatetime') else datetime.now(
                timezone.utc)
            direction = 'long' if final_direction == 'bullish' else 'short'

            # 5. بررسی همبستگی با بیت‌کوین
            btc_compatibility = {}
            if self.correlation_manager.enabled and hasattr(self.correlation_manager,
                                                            'check_btc_correlation_compatibility'):
                try:
                    # برای استفاده از این متد، نیاز به data_fetcher داریم
                    # فرض می‌کنیم این قابلیت در اختیار ماست یا از خود داده‌های موجود استفاده می‌کنیم
                    data_fetcher = self.data_fetcher if hasattr(self, 'data_fetcher') else None

                    if data_fetcher:
                        btc_compatibility = await self.correlation_manager.check_btc_correlation_compatibility(
                            symbol, direction, data_fetcher
                        )

                        # بررسی سازگاری با بیت‌کوین - اگر ناسازگار بود، سیگنال رد می‌شود
                        if not btc_compatibility.get('is_compatible', True):
                            logger.info(
                                f"Rejected signal for {symbol}: Incompatible with Bitcoin trend. "
                                f"BTC Trend: {btc_compatibility.get('btc_trend', 'unknown')}, "
                                f"Correlation: {btc_compatibility.get('correlation_with_btc', 0):.2f}, "
                                f"Direction: {direction}, "
                                f"Reason: {btc_compatibility.get('reason', 'unknown')}"
                            )
                            return None
                except Exception as e:
                    logger.error(f"Error checking Bitcoin correlation for {symbol}: {e}")
                    # در صورت خطا، ادامه می‌دهیم بدون اینکه سیگنال را رد کنیم
                    btc_compatibility = {'is_compatible': True, 'reason': f'error: {str(e)}'}

            # 6. بررسی همبستگی با سایر ارزهای معامله‌شده
            correlation_safety = 1.0
            correlated_symbols = []
            if self.correlation_manager.enabled:
                correlation_safety = self.correlation_manager.get_correlation_safety_factor(symbol, direction)
                if direction == 'long':
                    bullish_score *= correlation_safety
                else:
                    bearish_score *= correlation_safety
                correlated_symbols = self.correlation_manager.get_correlated_symbols(symbol)

            # 7. Calculate risk/reward
            final_risk_reward_info = self.calculate_risk_reward(
                symbol, direction, current_price, successful_analysis_results, adapted_risk_config
            )
            final_sl = final_risk_reward_info['stop_loss']
            final_tp = final_risk_reward_info['take_profit']
            final_rr = final_risk_reward_info['risk_reward_ratio']
            sl_method = final_risk_reward_info.get('sl_method', 'Unknown')

            min_rr = adapted_risk_config.get('min_risk_reward_ratio', self.base_min_risk_reward_ratio)
            if final_rr < min_rr:
                # min_notify_score = self.notification_config.get('min_score_to_notify', 0)
                # current_final_score = (bullish_score if direction == 'long' else bearish_score)
                # if min_notify_score is None or current_final_score >= min_notify_score:
                #     logger.info(
                #         f"Rejected signal for {symbol}: Final RR {final_rr:.2f} < {min_rr:.2f} [Regime: {regime_info.get('regime')}]"
                #     )
                return None

            # 8. Calculate final score object
            score = SignalScore()
            is_reversal, reversal_strength = self.detect_reversal_conditions(successful_analysis_results, primary_tf)
            score.base_score = bullish_score if final_direction == 'bullish' else bearish_score

            higher_tf_confirmations = 0
            total_higher_tfs = 0
            primary_tf_weight = self.timeframe_weights.get(primary_tf, 1.0)
            for tf, res in successful_analysis_results.items():
                tf_w = self.timeframe_weights.get(tf, 1.0)
                if tf_w > primary_tf_weight:
                    total_higher_tfs += 1
                    trend_dir = res.get('trend', {}).get('trend', 'neutral')
                    if (final_direction == 'bullish' and 'bullish' in trend_dir) or \
                            (final_direction == 'bearish' and 'bearish' in trend_dir):
                        higher_tf_confirmations += 1
            higher_tf_ratio = higher_tf_confirmations / total_higher_tfs if total_higher_tfs > 0 else 0
            primary_trend_strength = abs(successful_analysis_results.get(primary_tf, {})
                                         .get('trend', {})
                                         .get('strength', 0))

            if is_reversal:
                reversal_modifier = max(0.3, 1.0 - (reversal_strength * 0.7))
                score.timeframe_weight = 1.0 + (higher_tf_ratio * 0.3 * reversal_modifier)
                score.trend_alignment = max(0.5, 1.0 - (reversal_strength * 0.5))
            else:
                score.timeframe_weight = 1.0 + (higher_tf_ratio * 0.5)
                score.trend_alignment = 1.0 + (primary_trend_strength * 0.2)

            score.volume_confirmation = 1.0 + (score_result.get('volume_confirmation_factor', 0) * 0.4)
            pattern_names = score_result.get('pattern_names', [])
            score.pattern_quality = 1.0 + min(0.5, len(pattern_names) * 0.1)
            score.confluence_score = min(0.5, max(0, (final_rr - min_rr) * 0.25))
            score.correlation_safety_factor = correlation_safety
            score.macd_analysis_score = 1.0 + ((score_result.get('timeframe_alignment_factor', 1.0) - 1.0) * 0.5)
            score.structure_score = score_result.get('htf_structure_factor', 1.0)
            score.volatility_score = score_result.get('volatility_factor', 1.0)
            harmonic_count = sum(1 for p in pattern_names if
                                 'harmonic' in p or 'butterfly' in p or 'crab' in p or 'gartley' in p or 'bat' in p)
            score.harmonic_pattern_score = 1.0 + (harmonic_count * 0.2)
            channel_count = sum(1 for p in pattern_names if 'channel' in p)
            score.price_channel_score = 1.0 + (channel_count * 0.1)
            cycle_count = sum(1 for p in pattern_names if 'cycle' in p)
            score.cyclical_pattern_score = 1.0 + (cycle_count * 0.05)
            if self.adaptive_learning.enabled:
                score.symbol_performance_factor = self.adaptive_learning.get_symbol_performance_factor(symbol,
                                                                                                       direction)

            # 9. Recalculate final score
            score.final_score = (score.base_score *
                                 score.timeframe_weight *
                                 score.trend_alignment *
                                 score.volume_confirmation *
                                 score.pattern_quality *
                                 (1.0 + score.confluence_score) *
                                 score.symbol_performance_factor *
                                 score.correlation_safety_factor *
                                 score.macd_analysis_score *
                                 score.structure_score *
                                 score.volatility_score *
                                 score.harmonic_pattern_score *
                                 score.price_channel_score *
                                 score.cyclical_pattern_score)

            # 10. Check final score against adapted threshold again
            min_score = adapted_signal_config.get('minimum_signal_score', self.base_minimum_signal_score)
            if score.final_score < min_score:
                # min_notify_score = self.notification_config.get('min_score_to_notify', 0)
                # if min_notify_score is None or score.final_score >= min_notify_score:
                #     logger.info(
                #         f"Rejected signal for {symbol}: Final Score {score.final_score:.2f} < {min_score:.2f} [Regime: {regime_info.get('regime')}]"
                #     )
                return None

            # 11. Gather market context
            market_context = {
                'regime': regime_info.get('regime', 'unknown'),
                'volatility': regime_info.get('volatility', 'unknown'),
                'trend_direction': regime_info.get('trend_direction', 'unknown'),
                'trend_strength': regime_info.get('trend_strength', 'unknown'),
                'timeframe_alignment': score_result.get('timeframe_alignment_factor', 1.0),
                'htf_structure': score_result.get('htf_structure_factor', 1.0),
                'volatility_factor': score_result.get('volatility_factor', 1.0),
                'anomaly_score': self.circuit_breaker.get_market_anomaly_score(
                    timeframes_data) if self.circuit_breaker.enabled else 0
            }

            # اضافه کردن اطلاعات همبستگی با بیت‌کوین به market_context
            if btc_compatibility:
                market_context['btc_compatibility'] = {
                    'btc_trend': btc_compatibility.get('btc_trend', 'unknown'),
                    'correlation_with_btc': btc_compatibility.get('correlation_with_btc', 0),
                    'correlation_type': btc_compatibility.get('correlation_type', 'unknown'),
                    'is_compatible': btc_compatibility.get('is_compatible', True),
                    'reason': btc_compatibility.get('reason', 'unknown')
                }

            # 12. Create final SignalInfo object
            signal_info = SignalInfo(
                symbol=symbol,
                timeframe=primary_tf,
                signal_type="reversal" if is_reversal else "multi_timeframe",
                direction=direction,
                entry_price=current_price,
                stop_loss=final_sl,
                take_profit=final_tp,
                risk_reward_ratio=final_rr,
                timestamp=signal_timestamp,
                pattern_names=pattern_names,
                score=score,  # Use the calculated score object
                confirmation_timeframes=list(successful_analysis_results.keys()),
                regime=regime_info.get('regime'),
                is_reversal=is_reversal,
                adapted_config=adapted_config,
                correlated_symbols=correlated_symbols,
                market_context=market_context,
                # Add details from the primary timeframe's analysis results
                macd_details=successful_analysis_results.get(primary_tf, {}).get('macd', {}).get('details'),
                volatility_details=successful_analysis_results.get(primary_tf, {}).get('volatility', {}).get('details'),
                harmonic_details=successful_analysis_results.get(primary_tf, {}).get('harmonic_patterns'),
                channel_details=successful_analysis_results.get(primary_tf, {}).get('price_channels'),
                cyclical_details=successful_analysis_results.get(primary_tf, {}).get('cyclical_patterns')
            )

            signal_info.generate_signal_id()
            signal_info.ensure_aware_timestamp()

            # اضافه کردن اطلاعات همبستگی با بیت‌کوین به لاگ سیگنال
            btc_info = ""
            if btc_compatibility:
                btc_corr = btc_compatibility.get('correlation_with_btc', 0)
                btc_corr_type = btc_compatibility.get('correlation_type', 'unknown')
                btc_trend = btc_compatibility.get('btc_trend', 'unknown')
                btc_info = f", BTC Trend: {btc_trend}, BTC Corr: {btc_corr:.2f} ({btc_corr_type})"

            # logger.info(
            #     f"Generated {direction.upper()} signal for {symbol} "
            #     f"[Score: {score.final_score:.2f}, RR: {final_rr:.2f}, SL: {sl_method}, "
            #     f"Regime: {signal_info.regime}{btc_info}]{' (REVERSAL)' if is_reversal else ''}"
            # )

            return signal_info

        except Exception as e:
            logger.error(f"Critical error generating signal for {symbol}: {e}", exc_info=True)
            return None

    def calculate_multi_timeframe_score(self, symbol: str,
                                        analysis_results: Dict[str, Dict[str, Any]],
                                        base_signals: Dict[str, Dict[str, Any]],
                                        timeframes_data: Dict[str, pd.DataFrame]) -> Dict[str, Any]:
        """
        Calculate multi-timeframe score with weighted volume confirmation.
        Now accepts both analysis results and the original timeframes_data (DataFrames).
        """
        try:
            bullish_score = 0.0
            bearish_score = 0.0
            all_signals = []
            trend_directions = {}
            momentum_directions = {}
            macd_directions = {}
            volume_confirmations = {}
            htf_scores = {}
            volatility_scores = {}
            vol_reject_signal = False  # Initialize rejection flag

            # Review each timeframe for which we have SUCCESSFUL analysis results
            for tf, result in analysis_results.items():
                # Skip if analysis wasn't successful for this timeframe
                if not isinstance(result, dict) or result.get('status') != 'ok':
                    continue

                # Use timeframes_data (dict of DataFrames) to check for emptiness
                # Check if the DataFrame for this timeframe exists and is not empty
                if tf not in timeframes_data or not isinstance(timeframes_data[tf], pd.DataFrame) or timeframes_data[
                    tf].empty:
                    logger.warning(f"Skipping score calculation for {symbol} {tf}: Missing or empty DataFrame.")
                    continue

                tf_weight = self.timeframe_weights.get(tf, 1.0)

                # Extract trend scores
                trend_data = result.get('trend', {})
                trend_strength = trend_data.get('strength', 0)
                trend_directions[tf] = trend_data.get('trend', 'neutral')
                trend_phase = trend_data.get('phase', 'undefined')

                if trend_strength > 0:
                    phase_multiplier = self._get_trend_phase_multiplier(trend_phase, 'bullish')
                    bullish_score += trend_strength * tf_weight * phase_multiplier
                elif trend_strength < 0:
                    phase_multiplier = self._get_trend_phase_multiplier(trend_phase, 'bearish')
                    bearish_score += abs(trend_strength) * tf_weight * phase_multiplier

                # Extract momentum scores
                mom_data = result.get('momentum', {})
                momentum_directions[tf] = mom_data.get('direction', 'neutral')
                momentum_strength = mom_data.get('momentum_strength', 1.0)
                bullish_score += mom_data.get('bullish_score', 0) * tf_weight * momentum_strength
                bearish_score += mom_data.get('bearish_score', 0) * tf_weight * momentum_strength
                mom_signals = [{**s, 'timeframe': tf, 'score': s.get('score', 0) * tf_weight * momentum_strength} for s
                               in mom_data.get('signals', [])]
                all_signals.extend(mom_signals)

                # Extract MACD scores
                macd_data = result.get('macd', {})
                macd_directions[tf] = macd_data.get('direction', 'neutral')
                macd_market_type = macd_data.get('market_type', 'unknown')
                macd_type_strength = 1.0
                if macd_market_type.startswith('A_'):
                    macd_type_strength = 1.2
                elif macd_market_type.startswith('C_'):
                    macd_type_strength = 1.2
                elif macd_market_type.startswith(('B_', 'D_')):
                    macd_type_strength = 1.0
                else:
                    macd_type_strength = 0.8
                bullish_score += macd_data.get('bullish_score', 0) * tf_weight * macd_type_strength
                bearish_score += macd_data.get('bearish_score', 0) * tf_weight * macd_type_strength
                macd_signals = [{**s, 'timeframe': tf, 'score': s.get('score', 0) * tf_weight * macd_type_strength} for
                                s in macd_data.get('signals', [])]
                all_signals.extend(macd_signals)

                # Extract price action scores
                pa_data = result.get('price_action', {})
                bullish_score += pa_data.get('bullish_score', 0) * tf_weight
                bearish_score += pa_data.get('bearish_score', 0) * tf_weight
                pa_signals = [{**s, 'timeframe': tf, 'score': s.get('score', 0) * tf_weight} for s in
                              pa_data.get('signals', [])]
                all_signals.extend(pa_signals)

                # Extract S/R breakout scores
                sr_data = result.get('support_resistance', {}).get('details', {})
                if sr_data.get('broken_resistance'):
                    resistance_level = sr_data['broken_resistance']
                    level_str = resistance_level.get('strength', 1.0) if isinstance(resistance_level, dict) else 1.0
                    score = self.pattern_scores.get('broken_resistance', 3.0) * tf_weight * level_str
                    bullish_score += score
                    all_signals.append({'type': 'broken_resistance', 'timeframe': tf, 'score': score,
                                        'direction': 'bullish'})  # Added direction
                if sr_data.get('broken_support'):
                    support_level = sr_data['broken_support']
                    level_str = support_level.get('strength', 1.0) if isinstance(support_level, dict) else 1.0
                    score = self.pattern_scores.get('broken_support', 3.0) * tf_weight * level_str
                    bearish_score += score
                    all_signals.append({'type': 'broken_support', 'timeframe': tf, 'score': score,
                                        'direction': 'bearish'})  # Added direction

                # Extract harmonic patterns
                harmonic_patterns = result.get('harmonic_patterns', [])
                for pattern in harmonic_patterns:
                    pattern_type = pattern.get('type', '')
                    pattern_direction = pattern.get('direction', 'neutral')
                    pattern_confidence = pattern.get('confidence', 0.7)
                    pattern_score = self.pattern_scores.get(pattern_type, 4.0) * pattern_confidence * tf_weight
                    if pattern_direction == 'bullish':
                        bullish_score += pattern_score
                    elif pattern_direction == 'bearish':
                        bearish_score += pattern_score
                    all_signals.append(
                        {'type': pattern_type, 'timeframe': tf, 'direction': pattern_direction, 'score': pattern_score})

                # Extract price channels
                channel_data = result.get('price_channels', {})
                channel_signal = channel_data.get('signal', {})
                if channel_signal:
                    signal_type = channel_signal.get('type', '')
                    signal_direction = channel_signal.get('direction', 'neutral')
                    signal_score = channel_signal.get('score', 0) * tf_weight
                    if signal_direction == 'bullish':
                        bullish_score += signal_score
                    elif signal_direction == 'bearish':
                        bearish_score += signal_score
                    all_signals.append(
                        {'type': signal_type, 'timeframe': tf, 'direction': signal_direction, 'score': signal_score})

                # Extract cyclical patterns
                cycle_data = result.get('cyclical_patterns', {})
                cycle_signal = cycle_data.get('signal', {})
                if cycle_signal:
                    signal_type = cycle_signal.get('type', '')
                    signal_direction = cycle_signal.get('direction', 'neutral')
                    signal_score = cycle_signal.get('score', 0) * tf_weight
                    if signal_direction == 'bullish':
                        bullish_score += signal_score
                    elif signal_direction == 'bearish':
                        bearish_score += signal_score
                    all_signals.append(
                        {'type': signal_type, 'timeframe': tf, 'direction': signal_direction, 'score': signal_score})

                # Get HTF structure scores
                # Note: analyze_higher_timeframe_structure needs the DataFrame, passing the result dict here
                # We need the actual DataFrame for the HTF analysis
                htf_df = timeframes_data.get(tf)  # Get the DataFrame for the current timeframe
                if htf_df is not None:
                    htf_structure = self.analyze_higher_timeframe_structure(htf_df, tf, analysis_results)
                    htf_scores[tf] = htf_structure.get('score', 1.0)
                else:
                    htf_scores[tf] = 1.0  # Default score if DF is missing

                # Get volatility scores
                volatility_data = result.get('volatility', {})
                volatility_scores[tf] = volatility_data  # Store the whole dict
                if volatility_data.get('reject', False):
                    vol_reject_signal = True  # Set rejection flag if any TF rejects

                # Store volume confirmation
                volume_confirmations[tf] = result.get('volume', {}).get('is_confirmed_by_volume', False)

            # Calculate weighted volume confirmation
            weighted_volume_factor = 0.0
            total_weight_vol = 0.0
            for tf, is_confirmed in volume_confirmations.items():
                tf_weight = self.timeframe_weights.get(tf, 1.0)
                weighted_volume_factor += (1 if is_confirmed else 0) * tf_weight
                total_weight_vol += tf_weight
            volume_confirmation_factor = weighted_volume_factor / total_weight_vol if total_weight_vol > 0 else 0.0

            # Calculate HTF structure factor
            weighted_htf_factor = 0.0
            total_weight_htf = 0.0
            for tf, score in htf_scores.items():
                tf_weight = self.timeframe_weights.get(tf, 1.0)
                weighted_htf_factor += score * tf_weight
                total_weight_htf += tf_weight
            htf_structure_factor = weighted_htf_factor / total_weight_htf if total_weight_htf > 0 else 1.0

            # Calculate volatility factor and check rejection
            weighted_vol_factor = 0.0
            total_weight_vol_score = 0.0
            # Check for rejection based on the specific flag set earlier
            # vol_reject_signal is already set if any timeframe indicated rejection

            for tf, vol_data in volatility_scores.items():
                tf_weight = self.timeframe_weights.get(tf, 1.0)
                score = vol_data.get('score', 1.0) if isinstance(vol_data, dict) else 1.0
                weighted_vol_factor += score * tf_weight
                total_weight_vol_score += tf_weight
            volatility_factor = weighted_vol_factor / total_weight_vol_score if total_weight_vol_score > 0 else 1.0

            # Determine final direction
            final_direction = 'neutral'
            margin = 1.1  # 10% margin
            if bullish_score > bearish_score * margin:
                final_direction = 'bullish'
            elif bearish_score > bullish_score * margin:
                final_direction = 'bearish'

            # Extract pattern names
            pattern_names = list(set(s.get('type', '') for s in all_signals))

            # Prepare result
            result_output = {
                'symbol': symbol,
                'final_bullish_score': round(bullish_score, 2),
                'final_bearish_score': round(bearish_score, 2),
                'final_direction': final_direction,
                'signals': sorted(all_signals, key=lambda s: s.get('score', 0), reverse=True),
                'pattern_names': pattern_names,
                'trend_directions': trend_directions,
                'momentum_directions': momentum_directions,
                'macd_directions': macd_directions,
                'volume_confirmation_factor': round(volume_confirmation_factor, 3),
                'htf_structure_factor': round(htf_structure_factor, 3),
                'volatility_factor': round(volatility_factor, 3),
                'volatility_rejection': vol_reject_signal
            }

            # Recalculate timeframe alignment factor based on final direction
            result_output['timeframe_alignment_factor'] = round(self._calculate_timeframe_alignment(
                trend_directions, momentum_directions, macd_directions, final_direction
            ), 3)

            return result_output

        except Exception as e:
            logger.error(f"Error calculating multi-timeframe score for {symbol}: {e}", exc_info=True)
            return {
                'symbol': symbol,
                'final_bullish_score': 0,
                'final_bearish_score': 0,
                'final_direction': 'error',
                'error': str(e)
            }

    def register_trade_result(self, trade_result: TradeResult) -> None:
        """Register a trade result for system learning and feedback"""
        # Add to adaptive learning system
        if self.adaptive_learning.enabled:
            self.adaptive_learning.add_trade_result(trade_result)

        # Add to emergency circuit breaker
        if self.circuit_breaker.enabled:
            self.circuit_breaker.add_trade_result(trade_result)

        logger.info(f"Registered trade result for {trade_result.symbol}: "
                    f"Signal ID: {trade_result.signal_id}, "
                    f"Direction: {trade_result.direction}, "
                    f"Profit: {trade_result.profit_r:.2f}R, "
                    f"Exit reason: {trade_result.exit_reason}")

    def update_active_positions(self, positions: Dict[str, Dict[str, Any]]) -> None:
        """Update active positions for correlation management system"""
        if self.correlation_manager.enabled:
            self.correlation_manager.update_active_positions(positions)

    async def process_multiple_symbols(self, symbols_data: Dict[str, Dict[str, pd.DataFrame]]) -> List[SignalInfo]:
        """
        Process multiple symbols concurrently to find signals

        Args:
            symbols_data: Dictionary of data {symbol: {timeframe: dataframe}}

        Returns:
            List of valid signals
        """
        if not symbols_data:
            return []

        # Check market conditions before signal generation
        if self.circuit_breaker.enabled:
            # Check if circuit breaker is active
            is_active, reason = self.circuit_breaker.check_if_active()
            if is_active:
                logger.warning(f"Circuit breaker active: {reason}. Skipping all signal generation.")
                return []

            # Check for abnormal market volatility
            market_anomaly_score = self.circuit_breaker.get_market_anomaly_score({
                symbol: tf_data.get(max(tf_data.keys(), key=self._get_tf_minutes))
                for symbol, tf_data in symbols_data.items()
                if tf_data
            })
            if market_anomaly_score > 0.7:  # High threshold for very abnormal conditions
                logger.warning(
                    f"Extreme market anomaly detected (score: {market_anomaly_score:.2f}). No signals will be generated.")
                return []
            elif market_anomaly_score > 0.5:
                logger.warning(
                    f"High market anomaly detected (score: {market_anomaly_score:.2f}). Applying strict filtering.")

        # Update correlations before signal generation
        if self.correlation_manager.enabled:
            # Use a specific timeframe (e.g., 4h) for correlation calculation
            selected_tf = '4h'
            correlation_data = {}
            for symbol, tf_data in symbols_data.items():
                if selected_tf in tf_data and tf_data[selected_tf] is not None:
                    correlation_data[symbol] = tf_data[selected_tf]

            self.correlation_manager.update_correlations(correlation_data)

            # آنالیز روند بیت‌کوین اگر قابلیت جدید همبستگی با بیت‌کوین وجود دارد
            if hasattr(self.correlation_manager, 'analyze_btc_trend'):
                try:
                    # نیاز به data_fetcher برای گرفتن داده‌های بیت‌کوین
                    data_fetcher = self.data_fetcher if hasattr(self, 'data_fetcher') else None

                    if data_fetcher:
                        btc_trend_info = await self.correlation_manager.analyze_btc_trend(data_fetcher)
                        logger.info(
                            f"Bitcoin trend analysis: {btc_trend_info['trend']}, "
                            f"Strength: {btc_trend_info['strength']:.2f}, "
                            f"Price: {btc_trend_info['last_price']:.2f}"
                        )
                except Exception as e:
                    logger.error(f"Error analyzing Bitcoin trend: {e}")

        # Create tasks for each symbol
        tasks = []
        for symbol, timeframes_data in symbols_data.items():
            task = asyncio.create_task(
                self.analyze_symbol(symbol, timeframes_data),
                name=f"analyze_{symbol}"
            )
            tasks.append(task)

        # Execute all tasks concurrently
        results = await asyncio.gather(*tasks, return_exceptions=True)

        # Filter successful results
        valid_signals = []
        for result in results:
            if isinstance(result, Exception):
                logger.error(f"Error in symbol processing: {result}")
            elif result is not None:
                valid_signals.append(result)

        # Sort by score
        valid_signals.sort(key=lambda x: x.score.final_score, reverse=True)

        # Post-process for portfolio diversification
        if self.correlation_manager.enabled and valid_signals:
            diversified_signals = self._diversify_signals(valid_signals)

            if len(diversified_signals) < len(valid_signals):
                logger.info(
                    f"Filtered {len(valid_signals) - len(diversified_signals)} correlated signals to diversify portfolio.")

            return diversified_signals

        return valid_signals

    def _diversify_signals(self, signals: List[SignalInfo]) -> List[SignalInfo]:
        """
        Filter highly correlated signals to diversify trading portfolio and consider Bitcoin correlation

        Args:
            signals: List of signal candidates

        Returns:
            Filtered list of signals to avoid overexposed correlations
        """
        if not signals or not self.correlation_manager.enabled:
            return signals

        try:
            max_exposure = self.correlation_manager.max_exposure_per_group
            corr_threshold = self.correlation_manager.correlation_threshold

            # Signals are already sorted by score - highest scores first
            accepted_signals = []
            accepted_symbols = set()

            # گروه‌بندی سیگنال‌ها بر اساس همبستگی با بیت‌کوین
            btc_correlated_signals = []
            btc_inverse_signals = []
            btc_neutral_signals = []
            other_signals = []

            # بررسی آیا قابلیت جدید همبستگی با بیت‌کوین فعال است
            has_btc_compatibility = hasattr(self.correlation_manager, 'check_btc_correlation_compatibility')

            for signal in signals:
                symbol = signal.symbol

                # بررسی اطلاعات همبستگی با بیت‌کوین اگر موجود است
                btc_compatibility = signal.market_context.get('btc_compatibility', {})

                if btc_compatibility:
                    correlation_with_btc = btc_compatibility.get('correlation_with_btc', 0)
                    correlation_type = btc_compatibility.get('correlation_type', 'unknown')

                    if correlation_type == 'positive':
                        btc_correlated_signals.append(signal)
                    elif correlation_type == 'inverse':
                        btc_inverse_signals.append(signal)
                    elif correlation_type == 'zero':
                        btc_neutral_signals.append(signal)
                    else:
                        other_signals.append(signal)
                else:
                    other_signals.append(signal)

            # مرحله اول: ترتیب‌دهی سیگنال‌ها بر اساس همبستگی با بیت‌کوین و امتیاز
            # در شرایط نزولی بیت‌کوین، اولویت با سیگنال‌های همبستگی معکوس یا صفر برای long است
            # در شرایط صعودی بیت‌کوین، اولویت با سیگنال‌های همبستگی مثبت برای long است

            # گرفتن روند بیت‌کوین از اولین سیگنال (اگر موجود باشد)
            btc_trend = 'neutral'
            if signals and 'btc_compatibility' in signals[0].market_context:
                btc_trend = signals[0].market_context['btc_compatibility'].get('btc_trend', 'neutral')

            # مرتب‌سازی سیگنال‌ها بر اساس اولویت‌های همبستگی
            prioritized_signals = []

            if btc_trend == 'bearish':
                # در روند نزولی، اولویت برای long با همبستگی معکوس یا صفر است
                # ابتدا سیگنال‌های long با همبستگی معکوس یا صفر
                long_inverse = [s for s in btc_inverse_signals if s.direction == 'long']
                long_neutral = [s for s in btc_neutral_signals if s.direction == 'long']
                # سپس سیگنال‌های short با همبستگی مثبت
                short_correlated = [s for s in btc_correlated_signals if s.direction == 'short']
                # بعد سایر سیگنال‌ها
                remaining = (
                        [s for s in btc_correlated_signals if s.direction == 'long'] +
                        [s for s in btc_inverse_signals if s.direction == 'short'] +
                        [s for s in btc_neutral_signals if s.direction == 'short'] +
                        other_signals
                )

                prioritized_signals = long_inverse + long_neutral + short_correlated + remaining

            elif btc_trend == 'bullish':
                # در روند صعودی، اولویت برای long با همبستگی مثبت است
                # ابتدا سیگنال‌های long با همبستگی مثبت
                long_correlated = [s for s in btc_correlated_signals if s.direction == 'long']
                # سپس سیگنال‌های short با همبستگی معکوس یا صفر
                short_inverse = [s for s in btc_inverse_signals if s.direction == 'short']
                short_neutral = [s for s in btc_neutral_signals if s.direction == 'short']
                # بعد سایر سیگنال‌ها
                remaining = (
                        [s for s in btc_inverse_signals if s.direction == 'long'] +
                        [s for s in btc_neutral_signals if s.direction == 'long'] +
                        [s for s in btc_correlated_signals if s.direction == 'short'] +
                        other_signals
                )

                prioritized_signals = long_correlated + short_inverse + short_neutral + remaining

            else:
                # در حالت خنثی، از همان ترتیب اصلی استفاده می‌کنیم
                prioritized_signals = signals

            # مرحله دوم: فیلتر کردن سیگنال‌های بسیار همبسته
            for signal in prioritized_signals:
                symbol = signal.symbol

                # اگر نماد قبلاً قبول نشده است
                if symbol not in accepted_symbols:
                    # یافتن نمادهای همبسته با این نماد
                    correlated = [s[0] for s in
                                  self.correlation_manager.get_correlated_symbols(symbol, corr_threshold)]

                    # شمارش نمادهای همبسته که قبلاً قبول شده‌اند
                    accepted_correlated = len([s for s in correlated if s in accepted_symbols])

                    # اگر تعداد نمادهای همبسته کمتر از آستانه است، این سیگنال را قبول کن
                    if accepted_correlated < max_exposure:
                        accepted_signals.append(signal)
                        accepted_symbols.add(symbol)
                    else:
                        logger.debug(
                            f"Filtered out {symbol} signal due to correlation limits (already have {accepted_correlated} correlated symbols)")
                else:
                    logger.debug(f"Skipped duplicate signal for {symbol}")

            # لاگ اطلاعات خلاصه
            if len(accepted_signals) < len(signals):
                logger.info(
                    f"Filtered {len(signals) - len(accepted_signals)} signals for portfolio diversification. "
                    f"BTC trend: {btc_trend}. "
                    f"Accepted {len([s for s in accepted_signals if s.direction == 'long'])} long and "
                    f"{len([s for s in accepted_signals if s.direction == 'short'])} short signals."
                )

            return accepted_signals

        except Exception as e:
            logger.error(f"Error in signal diversification: {e}", exc_info=True)
            return signals  # Return original signals if error

# Example usage
if __name__ == "__main__":
    # Setup logging
    logging.basicConfig(level=logging.INFO,
                        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

    # Load configuration
    import json
    try:
        with open('signal_config.json', 'r') as f:
            config = json.load(f)
    except FileNotFoundError:
        # Default configuration if file not found
        config = {
            "signal_generation": {
                "timeframes": ["5m", "15m", "1h", "4h"],
                "timeframe_weights": {
                    "5m": 0.7,
                    "15m": 0.85,
                    "1h": 1.0,
                    "4h": 1.2
                },
                "minimum_signal_score": 180.0,
                "pattern_scores": {
                    "macd_bullish_crossover": 2.2,
                    "macd_bearish_crossover": 2.2,
                    "rsi_oversold_reversal": 2.3,
                    "rsi_overbought_reversal": 2.3,
                    "hammer": 1.8,
                    "engulfing": 2.2,
                    "inverse_head_and_shoulders": 4.0,
                    "head_and_shoulders": 4.0
                }
            },
            "risk_management": {
                "min_risk_reward_ratio": 1.8,
                "preferred_risk_reward_ratio": 2.5,
                "default_stop_loss_percent": 1.5,
                "max_risk_per_trade_percent": 1.5
            },
            "market_regime": {
                "enabled": True
            },
            "adaptive_learning": {
                "enabled": True
            },
            "correlation_management": {
                "enabled": True,
                "correlation_threshold": 0.7,
                "max_exposure_per_group": 2
            },
            "circuit_breaker": {
                "enabled": True,
                "max_consecutive_losses": 3,
                "max_daily_losses_r": 5.0
            }
        }

    # Create signal generator
    signal_gen = SignalGenerator(config)

    # Example async run
    async def example_run():
        # Sample data for testing
        import pandas as pd
        import numpy as np
        from datetime import datetime, timedelta

        def create_sample_data(symbol, length=200):
            dates = [datetime.now() - timedelta(minutes=i) for i in range(length, 0, -1)]
            close = np.cumsum(np.random.normal(0, 1, length)) + 100
            open_prices = close - np.random.normal(0, 0.5, length)
            high = np.maximum(close, open_prices) + np.random.normal(0, 0.5, length)
            low = np.minimum(close, open_prices) - np.random.normal(0, 0.5, length)
            volume = np.random.normal(1000, 200, length)

            df = pd.DataFrame({
                'open': open_prices,
                'high': high,
                'low': low,
                'close': close,
                'volume': volume
            }, index=dates)
            return df

        # Create sample data for multiple symbols and timeframes
        symbols_data = {}
        for symbol in ['BTCUSDT', 'ETHUSDT', 'ADAUSDT']:
            symbols_data[symbol] = {
                '5m': create_sample_data(symbol, 300),
                '15m': create_sample_data(symbol, 250),
                '1h': create_sample_data(symbol, 200),
                '4h': create_sample_data(symbol, 150)
            }

        # Process signals
        signals = await signal_gen.process_multiple_symbols(symbols_data)

        # Print signals
        for signal in signals:
            print(f"Symbol: {signal.symbol}, Direction: {signal.direction}, Score: {signal.score.final_score:.2f}")
            print(
                f"  Entry: {signal.entry_price:.4f}, SL: {signal.stop_loss:.4f}, TP: {signal.take_profit:.4f}, RR: {signal.risk_reward_ratio:.2f}")
            print(f"  Signal Type: {signal.signal_type}, Regime: {signal.regime}")
            print(
                f"  Patterns: {', '.join(signal.pattern_names[:5])}{' ...' if len(signal.pattern_names) > 5 else ''}")
            print()

        # Shutdown properly
        signal_gen.shutdown()

    # Run example
    import asyncio
    asyncio.run(example_run())