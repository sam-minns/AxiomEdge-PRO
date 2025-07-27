# =============================================================================
# FEATURE ENGINEERING MODULE
# =============================================================================

import os
import gc
import logging
import multiprocessing
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any
from collections import defaultdict
from scipy.signal import hilbert
from scipy.stats import entropy, skew, kurtosis
import warnings

# Optional imports with fallbacks
try:
    import pywt
    PYWT_AVAILABLE = True
except ImportError:
    PYWT_AVAILABLE = False

try:
    from arch import arch_model
    ARCH_AVAILABLE = True
except ImportError:
    ARCH_AVAILABLE = False

try:
    from statsmodels.tsa.stattools import pacf
    STATSMODELS_AVAILABLE = True
except ImportError:
    STATSMODELS_AVAILABLE = False

try:
    from sklearn.ensemble import IsolationForest
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False

from .config import ConfigModel
from .utils import get_optimal_system_settings

logger = logging.getLogger(__name__)

def _parallel_process_symbol_wrapper(symbol_tuple, feature_engineer_instance):
    """
    Wrapper to call the instance method for a single symbol.
    """
    symbol, symbol_data_by_tf = symbol_tuple
    logger.info(f"  - Starting parallel processing for symbol: {symbol}...")
    # The original processing logic is called here
    return feature_engineer_instance._process_single_symbol_stack(symbol_data_by_tf)

class FeatureEngineer:
    """
    Advanced feature engineering system for financial time series data.

    Generates 200+ technical, statistical, and behavioral features from OHLCV data
    across multiple timeframes. Includes sophisticated anomaly detection, pattern
    recognition, and market microstructure analysis.

    Features include:
    - Technical indicators (RSI, MACD, Bollinger Bands, etc.)
    - Statistical measures (volatility, skewness, entropy)
    - Behavioral patterns (candlestick analysis, volume patterns)
    - Market microstructure (bid-ask dynamics, order flow)
    - Multi-timeframe fusion and cross-timeframe relationships
    - Anomaly detection and outlier identification
    - AI-discovered behavioral patterns
    """

    TIMEFRAME_MAP = {
        'M1': 1, 'M5': 5, 'M15': 15, 'M30': 30, 'H1': 60, 'H4': 240,
        'D1': 1440, 'DAILY': 1440, 'W1': 10080, 'MN1': 43200
    }

    ANOMALY_FEATURES = [
        'ATR', 'bollinger_bandwidth', 'RSI', 'RealVolume', 'candle_body_size',
        'pct_change', 'candle_body_size_vs_atr', 'atr_vs_daily_atr', 'MACD_hist',
        'wick_to_body_ratio', 'overnight_gap_pct', 'RSI_zscore', 'volume_ma_ratio', 'volatility_hawkes'
    ]

    # List of target columns to exclude from feature list
    NON_FEATURE_COLS = [
        'Open', 'High', 'Low', 'Close', 'RealVolume', 'Symbol', 'Timestamp',
        'signal_pressure', 'target_signal_pressure_class', 'target_timing_score',
        'target_bullish_engulfing', 'target_bearish_engulfing', 'target_volatility_spike'
    ]

    def __init__(self, config: ConfigModel, timeframe_roles: Dict[str, str], playbook: Dict):
        """
        Initialize the FeatureEngineer with configuration and timeframe roles.

        Args:
            config: Configuration object with feature engineering parameters
            timeframe_roles: Dictionary mapping role names to timeframe identifiers
            playbook: Strategy playbook containing historical patterns and insights
        """
        self.config = config
        self.roles = timeframe_roles
        self.playbook = playbook
        self.hurst_warning_symbols = set()

        # Initialize feature categories for organization
        self.feature_categories = {
            "Time-Based": ['hour', 'day_', 'session', 'month', 'week_'],
            "Statistical & Fractal": ['skew', 'kurtosis', 'entropy', 'hurst', 'pacf', 'garch', 'zscore'],
            "Volatility": ['volatility', 'ATR', 'bollinger_bandwidth', 'bb_width', 'vix'],
            "Structural & Event-Based": ['gap', 'displacement', 'breakout', 'structural_break', 'orb_'],
            "Sentiment & Emotion": ['sentiment', 'fear_greed'],
            "Behavioral Insights": ['drawdown_percent', 'anchor_price', 'insidebar'],
            "Fundamental Price/Volume": ['Close', 'Open', 'High', 'Low', 'RealVolume', 'pct_change', 'prev_session_high', 'orb_high']
        }

    def label_data_multi_task(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Orchestrates the creation of primary and auxiliary target labels.
        """
        logger.info("-> Generating labels for multi-task learning...")
        df_labeled = df.copy()
        lookahead = self.config.LOOKAHEAD_CANDLES

        logger.info("  - Generating primary signal pressure labels...")
        pressure_series = self._calculate_signal_pressure_series(df_labeled, lookahead)
        df_labeled['signal_pressure'] = pressure_series
        df_labeled = self._label_primary_target(df_labeled)

        logger.info("  - Generating auxiliary confirmation labels (timing, patterns, volatility)...")
        df_labeled = self._calculate_timing_score(df_labeled, lookahead)
        df_labeled = self._calculate_future_engulfing(df_labeled, lookahead)
        df_labeled = self._calculate_future_volatility_spike(df_labeled, lookahead)

        return df_labeled.drop(columns=['signal_pressure'], errors='ignore')

    def _calculate_future_engulfing(self, df: pd.DataFrame, lookahead: int) -> pd.DataFrame:
        """Calculates binary flags for engulfing patterns in the near future."""
        df_copy = df.copy()
        df_copy['target_bullish_engulfing'] = 0
        df_copy['target_bearish_engulfing'] = 0

        close_prev = df_copy['Close'].shift(1)
        open_prev = df_copy['Open'].shift(1)

        is_bull_engulf = (df_copy['Open'] < close_prev) & \
                         (df_copy['Close'] > open_prev) & \
                         (df_copy['Close'] > close_prev) & \
                         (df_copy['Open'] < open_prev)

        is_bear_engulf = (df_copy['Open'] > close_prev) & \
                         (df_copy['Close'] < open_prev) & \
                         (df_copy['Close'] < close_prev) & \
                         (df_copy['Open'] > open_prev)

        df_copy['target_bullish_engulfing'] = is_bull_engulf.rolling(window=lookahead, min_periods=1).max().shift(-lookahead).fillna(0).astype(int)
        df_copy['target_bearish_engulfing'] = is_bear_engulf.rolling(window=lookahead, min_periods=1).max().shift(-lookahead).fillna(0).astype(int)
        return df_copy

    def _calculate_future_volatility_spike(self, df: pd.DataFrame, lookahead: int, threshold_multiplier: float = 2.0) -> pd.DataFrame:
        """Calculates a binary flag for a volatility spike in the near future."""
        df_copy = df.copy()
        df_copy['target_volatility_spike'] = 0
        if 'ATR' not in df_copy.columns:
            return df_copy

        historical_avg_atr = df_copy['ATR'].rolling(window=20, min_periods=10).mean()
        threshold = historical_avg_atr * threshold_multiplier
        
        future_atr_max = df_copy['ATR'].rolling(window=lookahead, min_periods=1).max().shift(-lookahead)
        df_copy['target_volatility_spike'] = (future_atr_max > threshold).fillna(0).astype(int)
        return df_copy

    def _calculate_timing_score(self, df: pd.DataFrame, lookahead: int) -> pd.DataFrame:
        """
        Calculates a score based on whether TP or SL is hit first.
        Returns a score from -1 (SL hit first) to 1 (TP hit first).
        """
        df['target_timing_score'] = 0.0
        if not all(col in df.columns for col in ['Close', 'ATR', 'High', 'Low']):
            return df

        tp_mult = getattr(self.config, 'TP_ATR_MULTIPLIER', 2.0)
        sl_mult = getattr(self.config, 'SL_ATR_MULTIPLIER', 1.0)

        for i in range(len(df) - lookahead):
            entry_price = df['Close'].iloc[i]
            atr = df['ATR'].iloc[i] if 'ATR' in df.columns else df['Close'].iloc[i] * 0.02
            if pd.isna(atr) or atr <= 0: 
                continue

            future_slice = df.iloc[i+1 : i+1+lookahead]
            
            tp_long = entry_price + (atr * tp_mult)
            sl_long = entry_price - (atr * sl_mult)
            tp_short = entry_price - (atr * tp_mult)
            sl_short = entry_price + (atr * sl_mult)

            # Check long scenario
            tp_hit_long = (future_slice['High'] >= tp_long).any()
            sl_hit_long = (future_slice['Low'] <= sl_long).any()
            
            # Check short scenario  
            tp_hit_short = (future_slice['Low'] <= tp_short).any()
            sl_hit_short = (future_slice['High'] >= sl_short).any()

            if tp_hit_long and not sl_hit_long:
                df.iloc[i, df.columns.get_loc('target_timing_score')] = 1.0
            elif sl_hit_long and not tp_hit_long:
                df.iloc[i, df.columns.get_loc('target_timing_score')] = -1.0
            elif tp_hit_short and not sl_hit_short:
                df.iloc[i, df.columns.get_loc('target_timing_score')] = -1.0
            elif sl_hit_short and not tp_hit_short:
                df.iloc[i, df.columns.get_loc('target_timing_score')] = 1.0

        return df

    def _calculate_signal_pressure_series(self, df: pd.DataFrame, lookahead: int) -> pd.Series:
        """
        Calculates signal pressure based on future price movements.
        """
        if 'Close' not in df.columns:
            return pd.Series(0, index=df.index)
            
        future_returns = df['Close'].shift(-lookahead) / df['Close'] - 1
        
        # Normalize returns to pressure scale
        pressure = np.tanh(future_returns * 10)  # Scale and bound between -1 and 1
        
        return pressure.fillna(0)

    def _label_primary_target(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Creates primary classification target from signal pressure.
        """
        if 'signal_pressure' not in df.columns:
            df['target_signal_pressure_class'] = 0
            return df
            
        # Convert continuous pressure to discrete classes
        pressure = df['signal_pressure']
        
        # Define thresholds
        strong_buy_threshold = 0.3
        buy_threshold = 0.1
        sell_threshold = -0.1
        strong_sell_threshold = -0.3
        
        conditions = [
            pressure >= strong_buy_threshold,
            (pressure >= buy_threshold) & (pressure < strong_buy_threshold),
            (pressure > sell_threshold) & (pressure < buy_threshold),
            (pressure > strong_sell_threshold) & (pressure <= sell_threshold),
            pressure <= strong_sell_threshold
        ]
        
        choices = [4, 3, 2, 1, 0]  # Strong Buy, Buy, Hold, Sell, Strong Sell
        
        df['target_signal_pressure_class'] = np.select(conditions, choices, default=2)

        return df

    def apply_discovered_features(self, df: pd.DataFrame, discovered_patterns: Dict[str, Dict[str, str]]) -> pd.DataFrame:
        """
        Applies AI-discovered patterns as new features to the DataFrame.

        This method iterates through a dictionary of named patterns, each containing a
        Python lambda function as a string. It includes a heuristic to fix simple
        syntactical errors like mismatched parentheses before safely evaluating and
        applying these lambdas to create new feature columns.

        Args:
            df: The input DataFrame to which features will be added.
            discovered_patterns: A dictionary where keys are feature names and
                values are dictionaries containing a 'lambda' string and a 'description'.

        Returns:
            A new DataFrame with the added feature columns.
        """
        if not discovered_patterns:
            return df
        logger.info(f"-> Injecting {len(discovered_patterns)} AI-discovered alpha features...")
        df_copy = df.copy()
        safe_globals_for_lambda_definition = {
            "__builtins__": {
                "abs": abs, "min": min, "max": max, "round": round, "sum": sum, "len": len,
                "all": all, "any": any, "float": float, "int": int, "str": str, "bool": bool,
                "True": True, "False": False, "None": None,
                "isinstance": isinstance, "getattr": getattr, "hasattr": hasattr,
            }, "np": np, "pd": pd,
        }

        for pattern_name, pattern_info in discovered_patterns.items():
            lambda_str = pattern_info.get("lambda")
            description = pattern_info.get("description", "N/A")

            # Heuristic fix for mismatched parentheses from AI generation.
            if isinstance(lambda_str, str):
                open_paren_count = lambda_str.count('(')
                close_paren_count = lambda_str.count(')')

                if open_paren_count != close_paren_count:
                    logger.warning(f"  - Malformed lambda for '{pattern_name}'; attempting heuristic fix for mismatched parentheses.")
                    logger.debug(f"    Original lambda: {lambda_str}")

                    if close_paren_count > open_paren_count:
                        # More closing parens; trim from the end.
                        diff = close_paren_count - open_paren_count
                        num_trimmed = 0
                        while num_trimmed < diff and lambda_str.endswith(')'):
                            lambda_str = lambda_str[:-1]
                            num_trimmed += 1

                    elif open_paren_count > close_paren_count:
                        # More opening parens; append to the end.
                        diff = open_paren_count - close_paren_count
                        lambda_str += ')' * diff

                    logger.debug(f"    Corrected lambda: {lambda_str}")

            try:
                if not lambda_str or not isinstance(lambda_str, str) or "lambda row:" not in lambda_str.strip():
                    logger.warning(f"  - Skipping invalid or missing lambda string for pattern: {pattern_name}.")
                    df_copy[pattern_name] = 0.0
                    continue

                logger.info(f"  - Applying pattern: '{pattern_name}' (Desc: {description})")

                compiled_lambda = eval(lambda_str, safe_globals_for_lambda_definition, {})

                def apply_lambda_safely(row_data):
                    """
                    Safely applies the compiled lambda to a single row of data.
                    """
                    if pd.isna(row_data).all():
                        return np.nan
                    try:
                        return compiled_lambda(row_data)
                    except Exception:
                        raise

                df_copy[pattern_name] = df_copy.apply(apply_lambda_safely, axis=1).astype(float)

            except SyntaxError as e_syntax:
                logger.error(f"  - Failed to parse lambda for pattern '{pattern_name}' after potential fix: {e_syntax}")
                logger.debug(f"    Final (problematic) lambda: {lambda_str}")
                df_copy[pattern_name] = 0.0

            except Exception as e_runtime:
                logger.error(f"  - Failed to apply pattern '{pattern_name}' due to a runtime error: {e_runtime}", exc_info=True)
                logger.debug(f"    Problematic lambda: {lambda_str}")
                df_copy[pattern_name] = 0.0

        return df_copy

    def _calculate_simple_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate basic price and volume features"""
        df['pct_change'] = df.groupby('Symbol')['Close'].pct_change() if 'Symbol' in df.columns else df['Close'].pct_change()
        df['overnight_gap_pct'] = df.groupby('Symbol')['Open'].transform(
            lambda x: (x / x.shift(1).replace(0, np.nan)) - 1
        ) if 'Symbol' in df.columns else (df['Open'] / df['Open'].shift(1).replace(0, np.nan)) - 1

        # Price ratios
        df['high_low_ratio'] = df['High'] / df['Low'].replace(0, np.nan)
        df['close_open_ratio'] = df['Close'] / df['Open'].replace(0, np.nan)

        # Volume features
        if 'RealVolume' in df.columns:
            df['volume_ma_ratio'] = df['RealVolume'] / df['RealVolume'].rolling(20).mean()
            df['volume_std'] = df['RealVolume'].rolling(20).std()

        return df

    def _calculate_price_derivatives(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate price derivatives and momentum features"""
        close = df['Close']

        # First and second derivatives
        df['price_velocity'] = close.diff()
        df['price_acceleration'] = df['price_velocity'].diff()

        # Momentum indicators
        for period in [5, 10, 20]:
            df[f'momentum_{period}'] = close / close.shift(period) - 1
            df[f'roc_{period}'] = close.pct_change(period)

        # Price position in recent range
        for window in [10, 20, 50]:
            rolling_min = close.rolling(window).min()
            rolling_max = close.rolling(window).max()
            df[f'price_position_{window}'] = (close - rolling_min) / (rolling_max - rolling_min).replace(0, np.nan)

        return df

    def _calculate_volume_derivatives(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate volume-based features"""
        if 'RealVolume' not in df.columns:
            return df

        volume = df['RealVolume']

        # Volume momentum
        df['volume_momentum'] = volume.pct_change()
        df['volume_acceleration'] = df['volume_momentum'].diff()

        # Volume moving averages
        for period in [5, 10, 20]:
            df[f'volume_ma_{period}'] = volume.rolling(period).mean()
            df[f'volume_ratio_{period}'] = volume / df[f'volume_ma_{period}']

        # Price-volume relationship
        df['price_volume_corr'] = df['Close'].rolling(20).corr(volume)

        return df

    def _calculate_statistical_moments(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate statistical moments of returns"""
        returns = df['Close'].pct_change()

        for window in [10, 20, 50]:
            df[f'returns_mean_{window}'] = returns.rolling(window).mean()
            df[f'returns_std_{window}'] = returns.rolling(window).std()
            df[f'returns_skew_{window}'] = returns.rolling(window).skew()
            df[f'returns_kurt_{window}'] = returns.rolling(window).kurt()

        return df

    def _calculate_ohlc_ratios(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate OHLC-based ratios and patterns"""
        # Candle body and wick analysis
        df['candle_body_size'] = abs(df['Close'] - df['Open'])
        df['upper_wick'] = df['High'] - np.maximum(df['Open'], df['Close'])
        df['lower_wick'] = np.minimum(df['Open'], df['Close']) - df['Low']
        df['total_range'] = df['High'] - df['Low']

        # Ratios
        df['body_to_range_ratio'] = df['candle_body_size'] / df['total_range'].replace(0, np.nan)
        df['upper_wick_ratio'] = df['upper_wick'] / df['total_range'].replace(0, np.nan)
        df['lower_wick_ratio'] = df['lower_wick'] / df['total_range'].replace(0, np.nan)
        df['wick_to_body_ratio'] = (df['upper_wick'] + df['lower_wick']) / df['candle_body_size'].replace(0, np.nan)

        # Doji patterns
        df['is_doji'] = (df['candle_body_size'] / df['total_range'] < 0.1).astype(int)

        return df

    def _calculate_technical_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate standard technical indicators"""
        close = df['Close']
        high = df['High']
        low = df['Low']

        # RSI
        for period in getattr(self.config, 'RSI_STANDARD_PERIODS', [14, 21]):
            delta = close.diff()
            gain = (delta.where(delta > 0, 0)).rolling(period).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(period).mean()
            rs = gain / loss.replace(0, np.nan)
            df[f'RSI_{period}'] = 100 - (100 / (1 + rs))

        # ATR (Average True Range)
        tr1 = high - low
        tr2 = abs(high - close.shift(1))
        tr3 = abs(low - close.shift(1))
        true_range = np.maximum(tr1, np.maximum(tr2, tr3))
        df['ATR'] = true_range.rolling(14).mean()

        # Bollinger Bands
        bb_period = getattr(self.config, 'BOLLINGER_PERIOD', 20)
        bb_std = 2
        sma = close.rolling(bb_period).mean()
        std = close.rolling(bb_period).std()
        df[f'BB_upper_{bb_period}'] = sma + (std * bb_std)
        df[f'BB_lower_{bb_period}'] = sma - (std * bb_std)
        df[f'BB_middle_{bb_period}'] = sma
        df[f'BB_width_{bb_period}'] = (df[f'BB_upper_{bb_period}'] - df[f'BB_lower_{bb_period}']) / sma
        df[f'BB_position_{bb_period}'] = (close - df[f'BB_lower_{bb_period}']) / (df[f'BB_upper_{bb_period}'] - df[f'BB_lower_{bb_period}'])

        # MACD
        ema_fast = close.ewm(span=12).mean()
        ema_slow = close.ewm(span=26).mean()
        df['MACD'] = ema_fast - ema_slow
        df['MACD_signal'] = df['MACD'].ewm(span=9).mean()
        df['MACD_hist'] = df['MACD'] - df['MACD_signal']

        # Stochastic
        stoch_period = getattr(self.config, 'STOCHASTIC_PERIOD', 14)
        lowest_low = low.rolling(stoch_period).min()
        highest_high = high.rolling(stoch_period).max()
        df[f'Stoch_K_{stoch_period}'] = 100 * (close - lowest_low) / (highest_high - lowest_low)
        df[f'Stoch_D_{stoch_period}'] = df[f'Stoch_K_{stoch_period}'].rolling(3).mean()

        return df

    def _calculate_rsi(self, df: pd.DataFrame, period: int = 14) -> pd.DataFrame:
        """Calculate RSI (Relative Strength Index) indicator"""
        delta = df['Close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period, min_periods=1).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period, min_periods=1).mean()
        rs = gain / (loss + 1e-9)
        df['RSI'] = 100 - (100 / (1 + rs))
        return df

    def _calculate_atr(self, df: pd.DataFrame, period: int = 14) -> pd.DataFrame:
        """Calculate Average True Range (ATR) indicator"""
        high_low = df['High'] - df['Low']
        high_close = np.abs(df['High'] - df['Close'].shift())
        low_close = np.abs(df['Low'] - df['Close'].shift())
        true_range = np.maximum(high_low, np.maximum(high_close, low_close))
        df['ATR'] = true_range.rolling(window=period, min_periods=1).mean()
        return df

    def _calculate_behavioral_metrics(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate behavioral and psychological trading metrics.

        These features capture trader psychology, market sentiment, and behavioral patterns
        that often drive price movements beyond pure technical analysis.
        """
        # Anchor price bias - tendency to anchor to recent significant levels
        df['anchor_price_high_20'] = df['High'].rolling(20, min_periods=1).max()
        df['anchor_price_low_20'] = df['Low'].rolling(20, min_periods=1).min()
        df['distance_from_anchor_high'] = (df['Close'] - df['anchor_price_high_20']) / df['anchor_price_high_20']
        df['distance_from_anchor_low'] = (df['Close'] - df['anchor_price_low_20']) / df['anchor_price_low_20']

        # Drawdown psychology - how current price relates to recent peaks
        rolling_max = df['Close'].rolling(50, min_periods=1).max()
        df['drawdown_percent'] = (df['Close'] - rolling_max) / rolling_max
        df['days_since_high'] = (df.index.to_series().diff().dt.days.fillna(1).cumsum() -
                                df.groupby((df['Close'] == rolling_max).cumsum()).cumcount())

        # Fear and greed indicators
        df['fear_greed_rsi'] = np.where(df['RSI'] < 30, 1, np.where(df['RSI'] > 70, -1, 0))

        # Volume-based sentiment
        avg_volume = df['RealVolume'].rolling(20, min_periods=1).mean()
        df['volume_sentiment'] = np.where(
            (df['RealVolume'] > avg_volume) & (df['Close'] > df['Open']), 1,
            np.where((df['RealVolume'] > avg_volume) & (df['Close'] < df['Open']), -1, 0)
        )

        # Momentum persistence (trend following behavior)
        df['momentum_persistence'] = df['Close'].pct_change().rolling(5, min_periods=1).apply(
            lambda x: (x > 0).sum() - (x < 0).sum()
        )

        return df

    def _calculate_session_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate session-based features for different trading sessions.

        Captures the unique characteristics of Asian, London, and New York sessions
        which often exhibit different volatility and trend patterns.
        """
        if not isinstance(df.index, pd.DatetimeIndex):
            # Add placeholder features if no datetime index
            df['asian_session_volatility'] = 0.0
            df['london_session_volatility'] = 0.0
            df['ny_session_volatility'] = 0.0
            df['session_transition'] = 0
            return df

        # Define session hours (UTC)
        df['hour_utc'] = df.index.hour

        # Session definitions
        asian_session = (df['hour_utc'] >= 0) & (df['hour_utc'] < 8)
        london_session = (df['hour_utc'] >= 8) & (df['hour_utc'] < 16)
        ny_session = (df['hour_utc'] >= 13) & (df['hour_utc'] < 21)

        # Session volatility characteristics
        returns = df['Close'].pct_change()
        df['asian_session_volatility'] = returns.where(asian_session).rolling(20, min_periods=1).std()
        df['london_session_volatility'] = returns.where(london_session).rolling(20, min_periods=1).std()
        df['ny_session_volatility'] = returns.where(ny_session).rolling(20, min_periods=1).std()

        # Session transitions (often high volatility periods)
        df['session_transition'] = (
            ((df['hour_utc'] == 8) | (df['hour_utc'] == 13) | (df['hour_utc'] == 21)).astype(int)
        )

        # Fill NaN values
        for col in ['asian_session_volatility', 'london_session_volatility', 'ny_session_volatility']:
            df[col] = df[col].fillna(method='ffill').fillna(0)

        return df

    def _calculate_pattern_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate candlestick pattern and price action features.

        Identifies common candlestick patterns and price action setups that
        traders use for entry and exit decisions.
        """
        # Basic candlestick components
        df['candle_body'] = abs(df['Close'] - df['Open'])
        df['upper_wick'] = df['High'] - np.maximum(df['Open'], df['Close'])
        df['lower_wick'] = np.minimum(df['Open'], df['Close']) - df['Low']
        df['candle_range'] = df['High'] - df['Low']

        # Relative sizes
        df['body_to_range_ratio'] = df['candle_body'] / (df['candle_range'] + 1e-9)
        df['upper_wick_ratio'] = df['upper_wick'] / (df['candle_range'] + 1e-9)
        df['lower_wick_ratio'] = df['lower_wick'] / (df['candle_range'] + 1e-9)

        # Doji patterns (small body relative to range)
        df['is_doji'] = (df['body_to_range_ratio'] < 0.1).astype(int)

        # Hammer and hanging man patterns
        df['is_hammer'] = (
            (df['lower_wick_ratio'] > 0.6) &
            (df['upper_wick_ratio'] < 0.1) &
            (df['body_to_range_ratio'] < 0.3)
        ).astype(int)

        # Inside bars (consolidation pattern)
        df['is_inside_bar'] = (
            (df['High'] < df['High'].shift(1)) &
            (df['Low'] > df['Low'].shift(1))
        ).astype(int)

        # Outside bars (breakout pattern)
        df['is_outside_bar'] = (
            (df['High'] > df['High'].shift(1)) &
            (df['Low'] < df['Low'].shift(1))
        ).astype(int)

        return df

    def _calculate_time_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate time-based features"""
        if not isinstance(df.index, pd.DatetimeIndex):
            logger.warning("DataFrame index is not DatetimeIndex. Skipping time features.")
            cols = ['hour', 'day_of_week', 'is_asian_session', 'is_london_session', 'is_ny_session', 'month', 'week_of_year']
            for col in cols:
                df[col] = np.nan if col in ['hour', 'day_of_week', 'month', 'week_of_year'] else 0
            return df

        df['hour'] = df.index.hour.astype(float)
        df['day_of_week'] = df.index.dayofweek.astype(float)
        df['month'] = df.index.month.astype(float)
        df['week_of_year'] = df.index.isocalendar().week.astype(float)

        # Trading sessions (assuming UTC time)
        df['is_asian_session'] = ((df.index.hour >= 0) & (df.index.hour < 8)).astype(int)
        df['is_london_session'] = ((df.index.hour >= 8) & (df.index.hour < 16)).astype(int)
        df['is_ny_session'] = ((df.index.hour >= 13) & (df.index.hour < 21)).astype(int)

        # Cyclical encoding
        df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24)
        df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24)
        df['day_sin'] = np.sin(2 * np.pi * df['day_of_week'] / 7)
        df['day_cos'] = np.cos(2 * np.pi * df['day_of_week'] / 7)
        df['month_sin'] = np.sin(2 * np.pi * df['month'] / 12)
        df['month_cos'] = np.cos(2 * np.pi * df['month'] / 12)

        return df

    def _calculate_entropy_features(self, df: pd.DataFrame, window: int = 30) -> pd.DataFrame:
        """Calculate entropy-based features"""
        def roll_entropy_raw(series_arr: np.ndarray):
            series_no_nan = series_arr[~np.isnan(series_arr)]
            if len(series_no_nan) < 2:
                return np.nan
            try:
                hist, _ = np.histogram(series_no_nan, bins=10, density=False)
                counts = hist / len(series_no_nan)
                return entropy(counts[counts > 0], base=2)
            except ValueError:
                return np.nan

        min_p = max(1, window//2)
        log_returns = np.log(df['Close'].replace(0, np.nan) / df['Close'].shift(1).replace(0, np.nan))
        df['shannon_entropy_returns'] = log_returns.rolling(window, min_periods=min_p).apply(roll_entropy_raw, raw=True)

        return df

    def _calculate_parkinson_volatility(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate Parkinson volatility estimator using high-low range.
        More efficient than close-to-close volatility for intraday data.
        """
        window = getattr(self.config, 'PARKINSON_VOLATILITY_WINDOW', 20)

        def parkinson_estimator_raw(high_low_log_sq_window_arr: np.ndarray):
            if np.isnan(high_low_log_sq_window_arr).all() or len(high_low_log_sq_window_arr) == 0:
                return np.nan
            valid_terms = high_low_log_sq_window_arr[~np.isnan(high_low_log_sq_window_arr)]
            if len(valid_terms) == 0:
                return np.nan
            return np.sqrt(np.sum(valid_terms) / (4 * len(valid_terms) * np.log(2)))

        high_low_ratio_log_sq = (np.log(df['High'].replace(0, np.nan) / df['Low'].replace(0, np.nan)) ** 2).replace([np.inf, -np.inf], np.nan)
        df['volatility_parkinson'] = high_low_ratio_log_sq.rolling(window=window, min_periods=max(1, window//2)).apply(parkinson_estimator_raw, raw=True)

        return df

    def _calculate_yang_zhang_volatility(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate Yang-Zhang volatility estimator which accounts for overnight gaps.
        More accurate than other estimators for markets with gaps.
        """
        window = getattr(self.config, 'YANG_ZHANG_VOLATILITY_WINDOW', 20)

        def yang_zhang_estimator_windowed(window_series: pd.Series) -> float:
            try:
                sub_df_ohlc_window = df.loc[window_series.index]

                if sub_df_ohlc_window.isnull().values.any() or len(sub_df_ohlc_window) < max(5, window // 2):
                    return np.nan

                o = sub_df_ohlc_window['Open'].replace(0, np.nan)
                h = sub_df_ohlc_window['High'].replace(0, np.nan)
                l = sub_df_ohlc_window['Low'].replace(0, np.nan)
                c = sub_df_ohlc_window['Close'].replace(0, np.nan)
                c_prev = c.shift(1)

                if o.isnull().any() or h.isnull().any() or l.isnull().any() or c.isnull().any():
                    return np.nan

                log_ho, log_lo, log_co = np.log(h / o), np.log(l / o), np.log(c / o)
                log_oc_prev = np.log(o / c_prev)

                if np.isinf(log_oc_prev.iloc[1:]).any() or log_oc_prev.iloc[1:].isnull().all():
                    return np.nan
                if np.isinf(log_co).any() or log_co.isnull().all():
                    return np.nan

                n = len(sub_df_ohlc_window)
                sigma_o_sq = np.nanvar(log_oc_prev.iloc[1:], ddof=0)
                sigma_c_sq = np.nanvar(log_co, ddof=0)

                sigma_rs_sq_terms = log_ho * (log_ho - log_co) + log_lo * (log_lo - log_co)
                sigma_rs_sq = np.nanmean(sigma_rs_sq_terms)

                if pd.isna(sigma_o_sq) or pd.isna(sigma_c_sq) or pd.isna(sigma_rs_sq):
                    return np.nan

                k = 0.34 / (1.34 + (n + 1) / (n - 1)) if n > 1 else 0.34 / 1.34
                vol_sq = sigma_o_sq + k * sigma_c_sq + (1 - k) * sigma_rs_sq

                return np.sqrt(vol_sq) if vol_sq >= 0 else np.nan
            except Exception as e:
                symbol_for_log = df['Symbol'].iloc[0] if 'Symbol' in df.columns and not df.empty else "Unknown"
                logger.debug(f"  - Yang-Zhang calculation error for symbol {symbol_for_log} in a window: {e}")
                return np.nan

        required_cols = ['Open', 'High', 'Low', 'Close']
        if not all(col in df.columns for col in required_cols):
            df['volatility_yang_zhang'] = np.nan
            return df

        df['volatility_yang_zhang'] = df['Close'].rolling(
            window=window, min_periods=max(5, window // 2)
        ).apply(yang_zhang_estimator_windowed, raw=False)

        # Multi-layered fallback for NaN values
        if 'volatility_parkinson' in df.columns:
            df['volatility_yang_zhang'].fillna(df['volatility_parkinson'], inplace=True)

        if 'ATR' in df.columns:
            # As a final fallback, use ATR normalised by price
            df['volatility_yang_zhang'].fillna(df['ATR'] / df['Close'].replace(0, np.nan), inplace=True)

        return df

    def _calculate_garch_volatility(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate GARCH(1,1) volatility forecast using rolling windows.
        Provides conditional volatility estimates based on past volatility clustering.
        """
        df['garch_volatility'] = np.nan
        if not ARCH_AVAILABLE:
            return df

        log_returns_scaled = np.log(df['Close'].replace(0, np.nan) / df['Close'].shift(1).replace(0, np.nan)).dropna() * 1000
        if len(log_returns_scaled) < 20:
            return df

        try:
            garch_window = 100
            min_periods_garch = max(20, garch_window // 2)

            def rolling_garch_fit(series_window_values: np.ndarray):
                series_no_nan = series_window_values[~np.isnan(series_window_values)]
                if len(series_no_nan) < 20:
                    return np.nan
                try:
                    garch_model = arch_model(series_no_nan, vol='Garch', p=1, q=1, rescale=False, dist='normal')
                    res = garch_model.fit(update_freq=0, disp='off', show_warning=False, options={'maxiter': 50})
                    if res.convergence_flag == 0:
                        forecast = res.forecast(horizon=1, reindex=False, align='origin')
                        pred_vol_scaled_garch = np.sqrt(forecast.variance.iloc[-1, 0])
                        return pred_vol_scaled_garch / 1000.0
                    return np.nan
                except Exception:
                    return np.nan

            garch_vol_series = log_returns_scaled.rolling(window=garch_window, min_periods=min_periods_garch).apply(
                rolling_garch_fit, raw=True
            )
            df['garch_volatility'] = garch_vol_series.reindex(df.index).ffill()
        except Exception as e:
            symbol_for_log = df['Symbol'].iloc[0] if 'Symbol' in df.columns and not df.empty else "UnknownSymbol"
            logger.debug(f"  - GARCH calculation error for symbol {symbol_for_log}: {e}")

        return df

    def _calculate_cycle_features(self, df: pd.DataFrame, window: int = 40) -> pd.DataFrame:
        """Calculate cycle-based features using Hilbert transform"""
        df['dominant_cycle_phase'], df['dominant_cycle_period'] = np.nan, np.nan
        close_series = df['Close'].dropna()

        if len(close_series) < window + 1:
            return df

        try:
            # Detrend the close series
            detrended_close = close_series - close_series.rolling(window=window, center=False, min_periods=1).mean().fillna(method='bfill').fillna(method='ffill')

            # Apply Hilbert transform
            analytic_signal = hilbert(detrended_close.values)
            instantaneous_phase = np.unwrap(np.angle(analytic_signal))

            df.loc[close_series.index, 'dominant_cycle_phase'] = instantaneous_phase

            # Estimate period from phase derivative
            phase_diff = np.diff(instantaneous_phase)
            period_estimate = 2 * np.pi / np.mean(phase_diff[phase_diff > 0]) if len(phase_diff[phase_diff > 0]) > 0 else np.nan
            df['dominant_cycle_period'] = period_estimate

        except Exception as e:
            logger.warning(f"Cycle feature calculation failed: {e}")

        return df

    def _calculate_autocorrelation_features(self, df: pd.DataFrame, lags: int = 5) -> pd.DataFrame:
        """
        Calculate Partial Autocorrelation Function (PACF) features for return series.
        Helps identify mean-reverting vs trending behavior in price movements.
        """
        if not STATSMODELS_AVAILABLE:
            for i in range(1, lags + 1):
                df[f'pacf_lag_{i}'] = np.nan
            return df

        # Compute log returns
        log_returns = np.log(df['Close'].replace(0, np.nan) / df['Close'].shift(1).replace(0, np.nan)).dropna()

        # Define PACF calculation parameters
        pacf_window = getattr(self.config, 'AUTOCORR_LAG', 5) * 3
        num_lags = getattr(self.config, 'AUTOCORR_LAG', 5)
        lag_cols = [f'pacf_lag_{i}' for i in range(1, num_lags + 1)]

        # If there aren't enough data points, return NaN columns
        if len(log_returns) < pacf_window:
            for col in lag_cols:
                df[col] = np.nan
            return df

        # Helper function to calculate all PACF lags for a given window (Series)
        def calculate_pacf_for_window(window_series: pd.Series) -> pd.Series:
            series_no_nan = window_series.dropna()
            nan_series = pd.Series(data=[np.nan] * num_lags, index=lag_cols)

            if len(series_no_nan) < num_lags + 5:
                return nan_series

            try:
                # Calculate PACF for all lags up to the configured limit
                pacf_vals = pacf(series_no_nan, nlags=num_lags, method='yw')
                # Return a pandas Series, skipping lag 0 which is always 1
                return pd.Series(data=pacf_vals[1:], index=lag_cols)
            except Exception:
                return nan_series

        # Manually iterate through rolling windows
        results_list = []

        # The loop starts after the first full window is available
        for i in range(pacf_window, len(log_returns) + 1):
            window = log_returns.iloc[i - pacf_window : i]
            results_list.append(calculate_pacf_for_window(window))

        # If results were generated, create a DataFrame and align its index
        if results_list:
            pacf_results_df = pd.DataFrame(results_list)
            # The index of the results should correspond to the END of each window
            pacf_results_df.index = log_returns.index[pacf_window - 1:]
            # Join the results back to the original dataframe
            df = df.join(pacf_results_df)
        else:  # Failsafe if no results were generated
            for col in lag_cols:
                df[col] = np.nan

        return df

    def _calculate_fourier_transform_features(self, df: pd.DataFrame, window: int = 50) -> pd.DataFrame:
        """
        Calculates the dominant frequency and amplitude of the price series using a Fourier Transform.
        Helps identify cyclical patterns and dominant periodicities in price movements.
        """
        # This helper function performs the core FFT calculation on a NumPy array
        def get_dominant_freq_amp_raw(series_arr: np.ndarray) -> tuple:
            # Remove any potential NaN values from the window slice
            series_no_nan = series_arr[~np.isnan(series_arr)]
            n = len(series_no_nan)

            # Check for invalid conditions (e.g., not enough data, no price movement)
            if n < 20 or (len(series_no_nan) > 1 and np.all(np.diff(series_no_nan) == 0)):
                return (np.nan, np.nan)

            try:
                # Perform the Fast Fourier Transform
                fft_vals = np.fft.fft(series_no_nan)
                fft_freq = np.fft.fftfreq(n)

                # Find the dominant frequency, ignoring the zero-frequency DC component
                positive_freq_indices = np.where(fft_freq > 1e-9)[0]
                if len(positive_freq_indices) == 0:
                    return (np.nan, np.nan)

                idx_max_amplitude = positive_freq_indices[np.argmax(np.abs(fft_vals[positive_freq_indices]))]

                dominant_frequency = np.abs(fft_freq[idx_max_amplitude])
                dominant_amplitude = np.abs(fft_vals[idx_max_amplitude]) / n

                return (dominant_frequency, dominant_amplitude)
            except Exception:
                return (np.nan, np.nan)

        min_p = max(20, window // 2)
        close_values = df['Close'].values
        results_list = []

        # Manually iterate through the data to create windows
        for i in range(len(close_values)):
            if i < min_p - 1:
                # For the initial periods where the window is not full, append NaNs
                results_list.append((np.nan, np.nan))
                continue

            # Define the start of the window, ensuring it doesn't go below index 0
            start_index = max(0, i - window + 1)
            window_slice = close_values[start_index : i + 1]

            # Calculate the features for the current window and append the result
            results_list.append(get_dominant_freq_amp_raw(window_slice))

        # Convert the list of result tuples into the final DataFrame columns
        if results_list:
            fft_results_df = pd.DataFrame(results_list, index=df.index, columns=['fft_dom_freq', 'fft_dom_amp'])
            df['fft_dom_freq'] = fft_results_df['fft_dom_freq']
            df['fft_dom_amp'] = fft_results_df['fft_dom_amp']
        else:
            df['fft_dom_freq'], df['fft_dom_amp'] = np.nan, np.nan

        return df

    def _calculate_wavelet_features(self, df: pd.DataFrame, wavelet_name='db4', level=4) -> pd.DataFrame:
        """
        Calculate wavelet decomposition features for multi-scale analysis.
        Captures price movements at different frequency scales.
        """
        if not PYWT_AVAILABLE:
            for i in range(level + 1):
                df[f'wavelet_coeff_energy_L{i}'] = np.nan
            return df

        min_len_for_wavelet = 30 + (level * 5)
        close_series_dropna = df['Close'].dropna()
        if len(close_series_dropna) < min_len_for_wavelet:
            for i in range(level + 1):
                df[f'wavelet_coeff_energy_L{i}'] = np.nan
            return df

        try:
            coeffs = pywt.wavedec(close_series_dropna.values, wavelet_name, level=level)
            energies = {}
            for i, c_arr in enumerate(coeffs):
                energies[f'wavelet_coeff_energy_L{i}'] = np.sum(np.square(c_arr)) / len(c_arr) if len(c_arr) > 0 else np.nan
            for col_name, energy_val in energies.items():
                df[col_name] = energy_val
        except Exception as e:
            symbol_for_log = df['Symbol'].iloc[0] if 'Symbol' in df.columns and not df.empty else "UnknownSymbol"
            logger.debug(f"  - Wavelet calculation error for symbol {symbol_for_log}: {e}")
            for i in range(level + 1):
                df[f'wavelet_coeff_energy_L{i}'] = np.nan

        return df

    def _calculate_quantile_features(self, df: pd.DataFrame, window: int = 50) -> pd.DataFrame:
        """
        Calculate quantile-based features for return distribution analysis.
        Provides insights into the distribution characteristics of returns.
        """
        log_returns = np.log(df['Close'].replace(0, np.nan) / df['Close'].shift(1).replace(0, np.nan))
        min_p = max(1, window//2)
        df['return_q25'] = log_returns.rolling(window, min_periods=min_p).quantile(0.25)
        df['return_q75'] = log_returns.rolling(window, min_periods=min_p).quantile(0.75)
        df['return_iqr'] = df['return_q75'] - df['return_q25']
        return df

    def _calculate_regression_features(self, df: pd.DataFrame, window: int = 20) -> pd.DataFrame:
        """
        Calculate rolling regression slope (beta) for trend analysis.
        Provides a measure of the strength and direction of price trends.
        """
        def get_slope(series_arr: np.ndarray):
            valid_values = series_arr[~np.isnan(series_arr)]
            if len(valid_values) < 2:
                return np.nan
            y = valid_values
            x = np.arange(len(y))
            try:
                return np.polyfit(x, y, 1)[0]
            except (np.linalg.LinAlgError, ValueError):
                return np.nan
        min_p = max(2, window//4)
        df['rolling_beta'] = df['Close'].rolling(window, min_periods=min_p).apply(get_slope, raw=True)
        return df

    def _calculate_relative_performance(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate relative performance vs market average.
        Helps identify outperforming and underperforming assets.
        """
        if 'pct_change' not in df.columns and 'Close' in df.columns and 'Symbol' in df.columns:
             df['pct_change'] = df.groupby('Symbol')['Close'].pct_change()
        elif 'pct_change' not in df.columns:
            df['pct_change'] = df['Close'].pct_change()

        df_for_market_ret = df.copy()
        if not isinstance(df_for_market_ret.index, pd.DatetimeIndex):
            if 'Timestamp' in df_for_market_ret.columns:
                df_for_market_ret = df_for_market_ret.set_index('Timestamp')
            else:
                logger.warning("Timestamp missing for market return calculation. Skipping relative performance.")
                df['relative_performance'] = 0.0
                return df

        if 'pct_change' not in df_for_market_ret.columns or df_for_market_ret['pct_change'].isnull().all():
            logger.warning("pct_change column missing or all NaN. Skipping relative performance.")
            df['relative_performance'] = 0.0
            return df

        mean_market_returns = df_for_market_ret.groupby(df_for_market_ret.index)['pct_change'].mean()
        df['market_return_temp'] = df.index.map(mean_market_returns)
        df['relative_performance'] = df['pct_change'] - df['market_return_temp']
        df.drop(columns=['market_return_temp'], inplace=True, errors='ignore')
        df['relative_performance'].fillna(0.0, inplace=True)
        return df

    def _detect_anomalies(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Detect anomalies in market data using Isolation Forest.
        Helps identify unusual market conditions that may affect model performance.
        """
        if not SKLEARN_AVAILABLE:
            df['anomaly_score'] = 1
            return df

        anomaly_features_present = [f for f in self.ANOMALY_FEATURES if f in df.columns and df[f].isnull().sum() < len(df) and df[f].nunique(dropna=False) > 1]
        if not anomaly_features_present:
            df['anomaly_score'] = 1
            return df

        df_anomaly_subset = df[anomaly_features_present].copy()
        for col in df_anomaly_subset.columns:
            if df_anomaly_subset[col].isnull().any():
                df_anomaly_subset[col].fillna(df_anomaly_subset[col].median(), inplace=True)

        if df_anomaly_subset.empty or df_anomaly_subset.nunique(dropna=False).max() < 2:
            df['anomaly_score'] = 1
            return df

        model = IsolationForest(contamination=getattr(self.config, 'anomaly_contamination_factor', 0.1), random_state=42)
        try:
            if not all(np.issubdtype(dtype, np.number) for dtype in df_anomaly_subset.dtypes):
                logger.warning(f"Non-numeric data found in anomaly features for symbol {df['Symbol'].iloc[0] if not df.empty else 'Unknown'}. Skipping anomaly detection.")
                df['anomaly_score'] = 1
                return df

            predictions = model.fit_predict(df_anomaly_subset)
            df['anomaly_score'] = pd.Series(predictions, index=df_anomaly_subset.index).reindex(df.index)
        except ValueError as e:
            symbol_for_log = df['Symbol'].iloc[0] if 'Symbol' in df.columns and not df.empty else "UnknownSymbol"
            logger.warning(f"Could not fit IsolationForest for symbol {symbol_for_log}: {e}. Defaulting anomaly_score to 1.")
            df['anomaly_score'] = 1

        df['anomaly_score'].ffill(inplace=True)
        df['anomaly_score'].bfill(inplace=True)
        df['anomaly_score'].fillna(1, inplace=True)
        return df

    def _calculate_kama_manual(self, series: pd.Series, n: int = 10, pow1: int = 2, pow2: int = 30) -> pd.Series:
        """
        Calculate Kaufman Adaptive Moving Average (KAMA) manually.
        Adapts to market volatility - faster in trending markets, slower in ranging markets.
        """
        change = abs(series - series.shift(n))
        volatility = (series - series.shift()).abs().rolling(n, min_periods=1).sum()
        er = (change / volatility.replace(0, np.nan)).fillna(0).clip(0, 1)
        sc_fast, sc_slow = 2 / (pow1 + 1), 2 / (pow2 + 1)
        sc = (er * (sc_fast - sc_slow) + sc_slow) ** 2
        kama_values = pd.Series(index=series.index, dtype=float)
        first_valid_idx = series.first_valid_index()
        if first_valid_idx is None:
            return kama_values
        kama_values.loc[first_valid_idx] = series.loc[first_valid_idx]
        for i in range(series.index.get_loc(first_valid_idx) + 1, len(series)):
            current_idx, prev_idx = series.index[i], series.index[i-1]
            if pd.isna(series.loc[current_idx]):
                kama_values.loc[current_idx] = kama_values.loc[prev_idx]
                continue
            if pd.isna(kama_values.loc[prev_idx]):
                kama_values.loc[current_idx] = series.loc[current_idx]
            else:
                kama_values.loc[current_idx] = kama_values.loc[prev_idx] + sc.loc[current_idx] * (series.loc[current_idx] - kama_values.loc[prev_idx])
        return kama_values

    def _calculate_kama_regime(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate KAMA-based regime identification.
        Uses fast and slow KAMA to determine market regime (trending vs ranging).
        """
        fast_kama = self._calculate_kama_manual(df["Close"], n=getattr(self.config, 'KAMA_REGIME_FAST', 10), pow1=2, pow2=getattr(self.config, 'KAMA_REGIME_FAST', 10))
        slow_kama = self._calculate_kama_manual(df["Close"], n=getattr(self.config, 'KAMA_REGIME_SLOW', 30), pow1=2, pow2=getattr(self.config, 'KAMA_REGIME_SLOW', 30))
        df["kama_trend"] = np.sign(fast_kama - slow_kama).fillna(0).astype(int)
        return df

    def _calculate_displacement(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate displacement signals based on candle range vs historical volatility.
        Identifies significant price movements that break normal volatility patterns.
        """
        df_copy = df.copy()
        df_copy["candle_range"] = np.abs(df_copy["High"] - df_copy["Low"])
        displacement_period = getattr(self.config, 'DISPLACEMENT_PERIOD', 20)
        displacement_strength = getattr(self.config, 'DISPLACEMENT_STRENGTH', 2.0)

        mstd = df_copy["candle_range"].rolling(displacement_period, min_periods=max(1, displacement_period//2)).std()
        threshold = mstd * displacement_strength
        df_copy["displacement_signal_active"] = (df_copy["candle_range"] > threshold).astype(int)
        variation = df_copy["Close"] - df_copy["Open"]
        df["green_displacement"] = ((df_copy["displacement_signal_active"] == 1) & (variation > 0)).astype(int).shift(1).fillna(0)
        df["red_displacement"] = ((df_copy["displacement_signal_active"] == 1) & (variation < 0)).astype(int).shift(1).fillna(0)
        return df

    def _calculate_gaps(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate gap detection features.
        Identifies price gaps that may indicate strong momentum or news events.
        """
        lookback = getattr(self.config, 'GAP_DETECTION_LOOKBACK', 1)
        df["is_bullish_gap"] = (df["High"].shift(lookback) < df["Low"]).astype(int).fillna(0)
        df["is_bearish_gap"] = (df["High"] < df["Low"].shift(lookback)).astype(int).fillna(0)
        return df

    def _calculate_candle_info(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate detailed candlestick information and patterns.
        Provides insights into price action and market sentiment.
        """
        df["candle_way"] = np.sign(df["Close"] - df["Open"]).fillna(0).astype(int)
        ohlc_range = (df["High"] - df["Low"]).replace(0, np.nan)
        df["filling_ratio"] = (np.abs(df["Close"] - df["Open"]) / ohlc_range).fillna(0)
        return df

    def _calculate_hawkes_volatility(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate Hawkes process-based volatility clustering.
        Models volatility as a self-exciting process where high volatility increases probability of future high volatility.
        """
        if 'ATR' not in df.columns or df['ATR'].isnull().all():
            df['volatility_hawkes'] = np.nan
            return df
        atr_shocks = df['ATR'].diff().clip(lower=0).fillna(0)
        hawkes_kappa = getattr(self.config, 'HAWKES_KAPPA', 0.1)
        hawkes_intensity = atr_shocks.ewm(alpha=1 - hawkes_kappa, adjust=False, min_periods=1).mean()
        df['volatility_hawkes'] = hawkes_intensity
        return df

    def _calculate_accumulation_distribution(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate Accumulation/Distribution line and its slope.
        Measures the cumulative flow of money into and out of a security.
        """
        if not all(col in df.columns for col in ['High', 'Low', 'Close', 'RealVolume']) or df['RealVolume'].isnull().all():
             df['AD_line'], df['AD_line_slope'] = np.nan, np.nan
             return df
        hl_range = (df['High'] - df['Low'])
        clv = np.where(hl_range == 0, 0, ((df['Close'] - df['Low']) - (df['High'] - df['Close'])) / hl_range.replace(0, np.nan))
        clv_series = pd.Series(clv, index=df.index).fillna(0)
        ad = (clv_series * df['RealVolume']).cumsum()
        df['AD_line'] = ad
        df['AD_line_slope'] = df['AD_line'].diff(5)  # 5-period slope
        return df

    def _calculate_mad(self, df: pd.DataFrame, window: int = 20) -> pd.DataFrame:
        """
        Calculate Mean Absolute Deviation (MAD).
        Provides a robust measure of price dispersion less sensitive to outliers than standard deviation.
        """
        min_p = max(1, window//2)
        df['mad'] = df['Close'].rolling(window, min_periods=min_p).apply(lambda x: np.mean(np.abs(x - np.mean(x))), raw=True)
        return df

    def _calculate_price_volume_correlation(self, df: pd.DataFrame, window: int = 20) -> pd.DataFrame:
        """
        Calculate rolling correlation between price returns and volume.
        Helps identify the relationship between price movements and trading activity.
        """
        if 'RealVolume' not in df.columns or df['RealVolume'].isnull().all() or 'Close' not in df.columns:
            df['price_vol_corr'] = np.nan
            return df
        min_p = max(5, window//2)  # Need more periods for stable correlation
        # Correlation between returns and volume
        log_returns = np.log(df['Close'].replace(0, np.nan) / df['Close'].shift(1).replace(0, np.nan))
        df['price_vol_corr'] = log_returns.rolling(window, min_periods=min_p).corr(df['RealVolume'])
        return df

    def _calculate_bollinger_bands(self, df: pd.DataFrame, period: int = 20, std_dev: float = 2.0) -> pd.DataFrame:
        """Calculate Bollinger Bands with specified period and standard deviation."""
        ma = df['Close'].rolling(window=period, min_periods=max(1, period//2)).mean()
        std = df['Close'].rolling(window=period, min_periods=max(1, period//2)).std()
        df[f'bollinger_upper_{period}_{std_dev}'] = ma + (std * std_dev)
        df[f'bollinger_lower_{period}_{std_dev}'] = ma - (std * std_dev)

        # Add epsilon to prevent division by zero
        df[f'bollinger_bandwidth_{period}_{std_dev}'] = (
            df[f'bollinger_upper_{period}_{std_dev}'] - df[f'bollinger_lower_{period}_{std_dev}']
        ) / (ma + 1e-9)

        # Set default bollinger bands if this matches the default configuration
        if hasattr(self.config, 'DYNAMIC_INDICATOR_PARAMS') and 'Default' in self.config.DYNAMIC_INDICATOR_PARAMS:
            default_params = self.config.DYNAMIC_INDICATOR_PARAMS['Default']
            if (period == default_params.get('bollinger_period', 20) and
                std_dev == default_params.get('bollinger_std_dev', 2.0) and
                'bollinger_upper' not in df.columns):
                df['bollinger_upper'] = df[f'bollinger_upper_{period}_{std_dev}']
                df['bollinger_lower'] = df[f'bollinger_lower_{period}_{std_dev}']
                df['bollinger_bandwidth'] = df[f'bollinger_bandwidth_{period}_{std_dev}']

        return df

    def _calculate_hurst_exponent(self, df: pd.DataFrame, window: int = 100) -> pd.DataFrame:
        """
        Calculate the Hurst exponent with state-based neutral imputation for the warm-up period.
        """
        df['hurst_exponent'], df['hurst_intercept'] = np.nan, np.nan

        try:
            from hurst import compute_Hc
            hurst_available = True
        except ImportError:
            logger.warning("hurst package not available. Skipping Hurst exponent calculation.")
            return df

        if not hurst_available:
            return df

        symbol_for_log = df.get('Symbol', pd.Series(['UnknownSymbol'])).iloc[0] if not df.empty else 'UnknownSymbol'

        def apply_hurst_raw(series_arr: np.ndarray, component_index: int):
            series_no_nan = series_arr[~np.isnan(series_arr)]

            if len(series_no_nan) < 100 or np.all(np.diff(series_no_nan) == 0):
                return np.nan

            try:
                result_tuple = compute_Hc(series_no_nan, kind='price', simplified=True)
                return result_tuple[component_index]
            except Exception as e_hurst:
                if not hasattr(self, 'hurst_warning_symbols'):
                    self.hurst_warning_symbols = set()
                if symbol_for_log not in self.hurst_warning_symbols:
                    logger.debug(f"Hurst calculation error for {symbol_for_log} (window {window}): {e_hurst}")
                    self.hurst_warning_symbols.add(symbol_for_log)
                return np.nan

        min_p_hurst = max(50, window // 2)

        rolling_close_hurst = df['Close'].rolling(window=window, min_periods=min_p_hurst)

        df['hurst_exponent'] = rolling_close_hurst.apply(apply_hurst_raw, raw=True, args=(0,))
        df['hurst_intercept'] = rolling_close_hurst.apply(apply_hurst_raw, raw=True, args=(1,))

        # Use state-based neutral imputation instead of backfilling
        # A Hurst exponent of 0.5 indicates a random walk (no memory)
        df['hurst_exponent'].fillna(0.5, inplace=True)
        df['hurst_intercept'].fillna(0.0, inplace=True)

        return df

    def _calculate_trend_pullback_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate trend pullback features based on ADX, EMA, and RSI."""
        if not all(x in df.columns for x in ['ADX', 'EMA_20', 'EMA_50', 'RSI', 'Close']):
            df['is_bullish_pullback'], df['is_bearish_pullback'] = 0, 0
            return df

        adx_threshold = getattr(self.config, 'ADX_THRESHOLD_TREND', 25)
        rsi_overbought = getattr(self.config, 'RSI_OVERBOUGHT', 70)
        rsi_oversold = getattr(self.config, 'RSI_OVERSOLD', 30)

        is_uptrend = (df['ADX'] > adx_threshold) & (df['EMA_20'] > df['EMA_50'])
        is_bullish_pullback_signal = (df['Close'] < df['EMA_20']) & (df['RSI'] < (rsi_overbought - 10))
        df['is_bullish_pullback'] = (is_uptrend & is_bullish_pullback_signal).astype(int)

        is_downtrend = (df['ADX'] > adx_threshold) & (df['EMA_20'] < df['EMA_50'])
        is_bearish_pullback_signal = (df['Close'] > df['EMA_20']) & (df['RSI'] > (rsi_oversold + 10))
        df['is_bearish_pullback'] = (is_downtrend & is_bearish_pullback_signal).astype(int)

        return df

    def _calculate_divergence_features(self, df: pd.DataFrame, lookback: int = 14) -> pd.DataFrame:
        """Calculate bullish and bearish divergence features."""
        if not all(x in df.columns for x in ['Close', 'RSI']):
            df['is_bullish_divergence'], df['is_bearish_divergence'] = 0, 0
            return df

        rolling_close_min = df['Close'].rolling(window=lookback, min_periods=max(1, lookback//2)).min()
        rolling_close_max = df['Close'].rolling(window=lookback, min_periods=max(1, lookback//2)).max()
        rolling_rsi_min = df['RSI'].rolling(window=lookback, min_periods=max(1, lookback//2)).min()
        rolling_rsi_max = df['RSI'].rolling(window=lookback, min_periods=max(1, lookback//2)).max()

        price_higher_high = (df['Close'] >= rolling_close_max) & (df['Close'].shift(1) < rolling_close_max.shift(1))
        rsi_lower_high = (df['RSI'] < rolling_rsi_max)
        df['is_bearish_divergence'] = (price_higher_high & rsi_lower_high).astype(int)

        price_lower_low = (df['Close'] <= rolling_close_min) & (df['Close'].shift(1) > rolling_close_min.shift(1))
        rsi_higher_low = (df['RSI'] > rolling_rsi_min)
        df['is_bullish_divergence'] = (price_lower_low & rsi_higher_low).astype(int)

        return df

    def _apply_kalman_filter(self, series: pd.Series) -> pd.Series:
        """Apply Kalman filter to smooth a time series."""
        try:
            from pykalman import KalmanFilter
            pykalman_available = True
        except ImportError:
            return series  # Skip if library not available

        if not pykalman_available:
            return series

        if series.isnull().all() or len(series.dropna()) < 2:
            return series

        series_filled = series.copy().ffill().bfill()  # Fill NaNs for KF
        if series_filled.isnull().all() or series_filled.nunique() < 2:
            return series_filled

        try:
            kf = KalmanFilter(
                initial_state_mean=series_filled.iloc[0] if not series_filled.empty else 0,
                n_dim_obs=1
            )
            kf = kf.em(
                series_filled.values,
                n_iter=5,
                em_vars=['transition_covariance', 'observation_covariance', 'initial_state_covariance']
            )
            (smoothed_state_means, _) = kf.smooth(series_filled.values)
            return pd.Series(smoothed_state_means.flatten(), index=series.index)
        except (ValueError, np.linalg.LinAlgError) as e:
            logger.debug(f"Kalman filter failed on series (len {len(series_filled)}, unique {series_filled.nunique()}): {e}. Returning original.")
            return series

    def _calculate_meta_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate meta features by combining existing features."""
        if 'RSI' in df.columns and 'bollinger_bandwidth' in df.columns:
            df['rsi_x_bolli'] = df['RSI'] * df['bollinger_bandwidth']

        if 'ADX' in df.columns and 'market_volatility_index' in df.columns:
            df['adx_x_vol_rank'] = df['ADX'] * df['market_volatility_index']

        if 'hurst_exponent' in df.columns and 'ADX' in df.columns:
            df['hurst_x_adx'] = df['hurst_exponent'] * df['ADX']

        if 'ATR' in df.columns and 'DAILY_ctx_ATR' in df.columns:
            df['atr_ratio_short_long'] = df['ATR'] / df['DAILY_ctx_ATR'].replace(0, np.nan)

        if 'hurst_intercept' in df.columns and 'ADX' in df.columns:
            df['hurst_intercept_x_adx'] = df['hurst_intercept'] * df['ADX']

        if 'hurst_intercept' in df.columns and 'ATR' in df.columns:
            df['hurst_intercept_x_atr'] = df['hurst_intercept'] * df['ATR']

        if 'volatility_parkinson' in df.columns and 'volatility_yang_zhang' in df.columns:
            df['vol_parkinson_yz_ratio'] = df['volatility_parkinson'] / df['volatility_yang_zhang'].replace(0, np.nan)

        return df

    def _calculate_liquidity_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate liquidity features including estimated spread and illiquidity ratio."""
        df_safe = df[(df['High'] >= df['Low']) & (df['Low'] > 0)].copy()
        if df_safe.empty or len(df_safe) < 2:
            df['estimated_spread'], df['illiquidity_ratio'] = np.nan, np.nan
            return df

        beta_sum_sq_log_hl = (np.log(df_safe['High'] / df_safe['Low'].replace(0, np.nan))**2).rolling(window=2, min_periods=2).sum()
        gamma_high = df_safe['High'].rolling(window=2, min_periods=2).max()
        gamma_low = df_safe['Low'].rolling(window=2, min_periods=2).min()
        gamma = (np.log(gamma_high / gamma_low.replace(0, np.nan))**2).fillna(0)
        alpha_denom = (3 - 2 * np.sqrt(2))
        if alpha_denom == 0:
            alpha_denom = 1e-9  # Avoid division by zero

        # Ensure terms for sqrt are non-negative and denominators are non-zero
        term1_sqrt = np.sqrt(beta_sum_sq_log_hl.clip(lower=0) / 2)
        term2_sqrt = np.sqrt(gamma.clip(lower=0) / alpha_denom)
        alpha = (np.sqrt(beta_sum_sq_log_hl.clip(lower=0)) - term1_sqrt) / alpha_denom - term2_sqrt
        alpha = alpha.replace([np.inf, -np.inf], np.nan)
        spread = 2 * (np.exp(alpha) - 1) / (np.exp(alpha).replace(-1, np.nan) + 1)
        df['estimated_spread'] = spread.reindex(df.index)

        if 'ATR' in df.columns:
            df['illiquidity_ratio'] = df['estimated_spread'] / df['ATR'].replace(0, np.nan)
        else:
            df['illiquidity_ratio'] = np.nan

        return df

    def _calculate_order_flow_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate order flow features including tick rule direction and volume imbalance."""
        price_changes = df['Close'].diff()
        df['tick_rule_direction'] = np.sign(price_changes).replace(0, np.nan).ffill().fillna(0).astype(int)

        if 'RealVolume' in df.columns and not df['RealVolume'].isnull().all():
            signed_volume = df['tick_rule_direction'] * df['RealVolume']
            df['volume_imbalance_5'] = signed_volume.rolling(5, min_periods=1).sum()
            df['volume_imbalance_20'] = signed_volume.rolling(20, min_periods=1).sum()
        else:
            df['volume_imbalance_5'], df['volume_imbalance_20'] = np.nan, np.nan

        return df

    def _calculate_depth_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate market depth proxy features."""
        ohlc_range = (df['High'] - df['Low']).replace(0, np.nan)
        df['depth_proxy_filling_ratio'] = (df['Close'] - df['Open']).abs() / ohlc_range

        upper_shadow = df['High'] - df[['Close', 'Open']].max(axis=1)
        lower_shadow = df[['Close', 'Open']].min(axis=1) - df['Low']
        df['upper_shadow_pressure'] = upper_shadow / ohlc_range
        df['lower_shadow_pressure'] = lower_shadow / ohlc_range

        return df

    def _calculate_structural_breaks(self, df: pd.DataFrame, window: int = 100) -> pd.DataFrame:
        """Calculate structural break detection using CUSUM statistics."""
        log_returns = np.log(df['Close'].replace(0, np.nan) / df['Close'].shift(1).replace(0, np.nan)).dropna()
        df['structural_break_cusum'] = 0

        if len(log_returns) < window:
            return df

        rolling_mean_ret = log_returns.rolling(window=window, min_periods=max(1, window//2)).mean()
        rolling_std_ret = log_returns.rolling(window=window, min_periods=max(1, window//2)).std().replace(0, np.nan)
        standardized_returns_arr = np.where(
            rolling_std_ret.notna() & (rolling_std_ret != 0),
            (log_returns - rolling_mean_ret) / rolling_std_ret,
            0
        )
        standardized_returns_series = pd.Series(standardized_returns_arr, index=log_returns.index)

        def cusum_calc_raw(x_std_ret_window_arr: np.ndarray):
            x_no_nan = x_std_ret_window_arr[~np.isnan(x_std_ret_window_arr)]
            if len(x_no_nan) < 2:
                return 0
            cumsum_vals = x_no_nan.cumsum()
            return cumsum_vals.max() - cumsum_vals.min() if len(cumsum_vals) > 0 else 0

        min_p_cusum = max(10, window // 4)
        cusum_stat = standardized_returns_series.rolling(window=window, min_periods=min_p_cusum).apply(cusum_calc_raw, raw=True)
        break_threshold = 5.0  # Example threshold
        df.loc[cusum_stat.index, 'structural_break_cusum'] = (cusum_stat > break_threshold).astype(int)
        df['structural_break_cusum'] = df['structural_break_cusum'].ffill().fillna(0)

        return df

    def _apply_pca_standard(self, df: pd.DataFrame, pca_features: list) -> pd.DataFrame:
        """Apply standard PCA to specified features."""
        if not pca_features or not all(f in df.columns for f in pca_features):
            logger.warning("PCA features missing from DataFrame. Skipping standard PCA.")
            for i in range(self.config.PCA_N_COMPONENTS):
                df[f'RSI_PCA_{i+1}'] = np.nan
            return df

        df_pca_subset = df[pca_features].copy().astype(np.float32)
        df_pca_subset.fillna(df_pca_subset.median(), inplace=True)
        df_pca_subset = df_pca_subset.loc[:, df_pca_subset.var(ddof=0) > 1e-6]

        # Set n_components to a float to target explained variance (e.g., 95%)
        n_components_target = 0.95

        if df_pca_subset.shape[1] < 2 or df_pca_subset.shape[0] < df_pca_subset.shape[1]:
            logger.warning(f"Not enough features/samples for standard PCA ({df_pca_subset.shape[1]} features available). Skipping.")
            for i in range(self.config.PCA_N_COMPONENTS):
                df[f'RSI_PCA_{i+1}'] = np.nan
            return df

        try:
            from sklearn.pipeline import Pipeline
            from sklearn.preprocessing import StandardScaler
            from sklearn.decomposition import PCA

            pipeline = Pipeline([
                ('scaler', StandardScaler()),
                ('pca', PCA(n_components=n_components_target, random_state=42))
            ])
            principal_components = pipeline.fit_transform(df_pca_subset)

            # The actual number of components selected by the algorithm
            actual_n_components = principal_components.shape[1]
            pca_cols = [f'RSI_PCA_{i+1}' for i in range(actual_n_components)]

            pca_df = pd.DataFrame(principal_components, columns=pca_cols, index=df_pca_subset.index)
            df_out = df.join(pca_df, how='left')
            logger.info(f"Standard PCA complete. Explained variance for {actual_n_components} components: {pipeline.named_steps['pca'].explained_variance_ratio_.sum():.2%}")
            return df_out
        except Exception as e:
            logger.error(f"Standard PCA failed: {e}. Skipping.")
            for i in range(self.config.PCA_N_COMPONENTS):
                df[f'RSI_PCA_{i+1}'] = np.nan
            return df

    def _apply_pca_incremental(self, df: pd.DataFrame, pca_features: list) -> pd.DataFrame:
        """Apply incremental PCA to specified features for large datasets."""
        if not pca_features or not all(f in df.columns for f in pca_features):
            logger.warning("PCA features missing from DataFrame. Skipping incremental PCA.")
            for i in range(self.config.PCA_N_COMPONENTS):
                df[f'RSI_PCA_{i+1}'] = np.nan
            return df

        df_pca_subset = df[pca_features].copy().astype(np.float32)
        df_pca_subset.fillna(df_pca_subset.median(), inplace=True)
        df_pca_subset = df_pca_subset.loc[:, df_pca_subset.var(ddof=0) > 1e-6]

        n_features = df_pca_subset.shape[1]
        n_components = min(self.config.PCA_N_COMPONENTS, n_features)

        # Ensure dimensionality reduction by using at least one fewer component than features
        if n_components >= n_features and n_features > 1:
            n_components = n_features - 1
            logger.warning(f"PCA components adjusted to {n_components} to ensure dimensionality reduction.")

        if n_features < 2 or df_pca_subset.shape[0] < n_components:
            logger.warning(f"Not enough features/samples for Incremental PCA ({n_features} features available). Skipping.")
            for i in range(self.config.PCA_N_COMPONENTS):
                df[f'RSI_PCA_{i+1}'] = np.nan
            return df

        try:
            from sklearn.preprocessing import StandardScaler
            from sklearn.decomposition import IncrementalPCA
            import sys

            scaler = StandardScaler()
            ipca = IncrementalPCA(n_components=n_components)
            batch_size = min(max(1000, df_pca_subset.shape[0] // 100), len(df_pca_subset))

            if batch_size == 0 and len(df_pca_subset) > 0:
                batch_size = len(df_pca_subset)

            logger.info(f"Fitting IncrementalPCA in batches of {batch_size}...")

            sample_for_scaler_fit_size = min(len(df_pca_subset), 50000)
            logger.info(f"Fitting StandardScaler on a sample of {sample_for_scaler_fit_size} rows...")
            if sample_for_scaler_fit_size > 0:
                scaler.fit(df_pca_subset.sample(n=sample_for_scaler_fit_size, random_state=42))
            else:
                logger.warning("Scaler is being fitted on a very small or empty DataFrame.")
                scaler.fit(df_pca_subset)

            total_batches = (df_pca_subset.shape[0] + batch_size - 1) // batch_size if batch_size > 0 else 0
            for i in range(0, df_pca_subset.shape[0], batch_size):
                batch = df_pca_subset.iloc[i:i + batch_size]

                if batch.shape[0] < n_components:
                    logger.warning(f"Skipping final batch of size {batch.shape[0]} as it's smaller than n_components ({n_components}).")
                    continue

                current_batch_num = (i // batch_size) + 1
                sys.stdout.write(f"\r  - Fitting IPCA on batch {current_batch_num}/{total_batches}...")
                sys.stdout.flush()

                ipca.partial_fit(scaler.transform(batch))

            sys.stdout.write('\n')

            logger.info("Transforming full dataset in batches with fitted IncrementalPCA...")
            transformed_batches = []
            for i in range(0, df_pca_subset.shape[0], batch_size):
                batch_to_transform = df_pca_subset.iloc[i:i + batch_size]
                if batch_to_transform.empty:
                    continue
                transformed_batches.append(ipca.transform(scaler.transform(batch_to_transform)))

            if not transformed_batches:
                logger.warning("No batches transformed by IncrementalPCA. Skipping PCA features.")
                for i in range(self.config.PCA_N_COMPONENTS):
                    df[f'RSI_PCA_{i+1}'] = np.nan
                return df

            principal_components = np.vstack(transformed_batches)
            pca_cols = [f'RSI_PCA_{i+1}' for i in range(n_components)]

            logger.info("Assigning new PCA features directly to the DataFrame...")

            for i in range(self.config.PCA_N_COMPONENTS):
                df[f'RSI_PCA_{i+1}'] = np.nan

            pca_df = pd.DataFrame(principal_components, columns=pca_cols, index=df_pca_subset.index)
            df.update(pca_df)

            logger.info(f"IncrementalPCA reduction complete. Explained variance for {n_components} components: {ipca.explained_variance_ratio_.sum():.2%}")
            return df
        except Exception as e:
            logger.error(f"Incremental PCA failed: {e}. Skipping.")
            for i in range(self.config.PCA_N_COMPONENTS):
                df[f'RSI_PCA_{i+1}'] = np.nan
            return df

    def _calculate_rsi_series(self, series: pd.Series, period: int) -> pd.Series:
        """Calculate RSI for a given series and period."""
        delta = series.diff()
        gain = delta.clip(lower=0).ewm(alpha=1/period, adjust=False, min_periods=1).mean()
        loss = (-delta.clip(upper=0)).ewm(alpha=1/period, adjust=False, min_periods=1).mean()
        rs = gain / loss.replace(0, np.nan)
        rsi = 100 - (100 / (1 + rs.fillna(np.inf)))
        return rsi.replace([np.inf, -np.inf], 50).fillna(50)

    def _calculate_rsi_mse(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate the Mean Squared Error of RSI against its moving average
        to quantify the stability of momentum, adding it as 'rsi_mse'.
        """
        # Get parameters from the configuration
        period = getattr(self.config, 'RSI_MSE_PERIOD', 14)
        sma_period = getattr(self.config, 'RSI_MSE_SMA_PERIOD', 10)
        mse_window = getattr(self.config, 'RSI_MSE_WINDOW', 20)

        # Use the existing robust internal RSI calculation
        rsi = self._calculate_rsi_series(df['Close'], period=period)

        # Calculate the moving average of the RSI
        rsi_ma = rsi.rolling(window=sma_period, min_periods=max(1, sma_period // 2)).mean()

        # Calculate the squared error and then the rolling mean (MSE)
        squared_error = (rsi - rsi_ma) ** 2
        df['rsi_mse'] = squared_error.rolling(window=mse_window, min_periods=max(1, mse_window // 2)).mean()

        return df

    def _calculate_dynamic_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate dynamic indicators based on market regime."""
        if 'market_volatility_index' not in df.columns or 'hurst_exponent' not in df.columns:
            symbol_for_log = df['Symbol'].iloc[0] if 'Symbol' in df.columns and not df.empty else "UnknownSymbol"
            logger.warning(f"Required columns for dynamic indicators missing for symbol {symbol_for_log}. Using default parameters.")
            default_params = self.config.DYNAMIC_INDICATOR_PARAMS['Default']
            df = self._calculate_bollinger_bands(df, period=default_params['bollinger_period'], std_dev=default_params['bollinger_std_dev'])
            df = self._calculate_rsi(df, period=default_params['rsi_period'])
            df['market_regime_str'] = "Default_Default"
            known_regimes_list = list(self.config.DYNAMIC_INDICATOR_PARAMS.keys())
            df['market_regime'] = known_regimes_list.index('Default') if 'Default' in known_regimes_list else 0
            return df

        df['volatility_regime_label'] = pd.cut(
            df['market_volatility_index'],
            bins=[0, 0.3, 0.7, 1.01],
            labels=['LowVolatility', 'Default', 'HighVolatility'],
            right=False
        ).astype(str).fillna('Default')

        df['trend_regime_label'] = pd.cut(
            df['hurst_exponent'],
            bins=[0, 0.4, 0.6, 1.01],
            labels=['Ranging', 'Default', 'Trending'],
            right=False
        ).astype(str).fillna('Default')

        df['market_regime_str'] = df['volatility_regime_label'] + "_" + df['trend_regime_label']

        def get_fallback_regime(row):
            vol_reg, trend_reg = row['volatility_regime_label'], row['trend_regime_label']
            if f"{vol_reg}_{trend_reg}" in self.config.DYNAMIC_INDICATOR_PARAMS:
                return f"{vol_reg}_{trend_reg}"
            if f"{vol_reg}_Default" in self.config.DYNAMIC_INDICATOR_PARAMS:
                return f"{vol_reg}_Default"
            if f"Default_{trend_reg}" in self.config.DYNAMIC_INDICATOR_PARAMS:
                return f"Default_{trend_reg}"
            return "Default"

        df['market_regime_str'] = df.apply(get_fallback_regime, axis=1)
        known_regimes_list = list(self.config.DYNAMIC_INDICATOR_PARAMS.keys())
        df['market_regime'] = pd.Categorical(
            df['market_regime_str'],
            categories=known_regimes_list,
            ordered=True
        ).codes
        df['market_regime'] = df['market_regime'].replace(
            -1,
            known_regimes_list.index('Default') if 'Default' in known_regimes_list else 0
        )

        # Calculate regime-specific indicators
        df['bollinger_upper'], df['bollinger_lower'], df['bollinger_bandwidth'], df['RSI'] = np.nan, np.nan, np.nan, np.nan
        for regime_code_val, group_indices in df.groupby('market_regime').groups.items():
            if group_indices.empty:
                continue
            regime_name_str = known_regimes_list[regime_code_val]
            params_for_regime = self.config.DYNAMIC_INDICATOR_PARAMS.get(
                regime_name_str,
                self.config.DYNAMIC_INDICATOR_PARAMS['Default']
            )
            group_df_slice = df.loc[group_indices]
            if group_df_slice.empty:
                continue

            # Calculate Bollinger Bands
            ma = group_df_slice['Close'].rolling(
                window=params_for_regime['bollinger_period'],
                min_periods=max(1, params_for_regime['bollinger_period']//2)
            ).mean()
            std = group_df_slice['Close'].rolling(
                window=params_for_regime['bollinger_period'],
                min_periods=max(1, params_for_regime['bollinger_period']//2)
            ).std()
            df.loc[group_indices, 'bollinger_upper'] = ma + (std * params_for_regime['bollinger_std_dev'])
            df.loc[group_indices, 'bollinger_lower'] = ma - (std * params_for_regime['bollinger_std_dev'])
            df.loc[group_indices, 'bollinger_bandwidth'] = (
                df.loc[group_indices, 'bollinger_upper'] - df.loc[group_indices, 'bollinger_lower']
            ) / ma.replace(0, np.nan)

            # Calculate RSI
            delta = group_df_slice['Close'].diff()
            gain = delta.clip(lower=0).ewm(alpha=1/params_for_regime['rsi_period'], adjust=False, min_periods=1).mean()
            loss = (-delta.clip(upper=0)).ewm(alpha=1/params_for_regime['rsi_period'], adjust=False, min_periods=1).mean()
            rs = gain / loss.replace(0, np.nan)
            rsi_calc_group = 100 - (100 / (1 + rs.fillna(np.inf)))
            df.loc[group_indices, 'RSI'] = rsi_calc_group.replace([np.inf, -np.inf], 50).fillna(50)

        # Apply proxy-based imputation for the warm-up period
        if 'bollinger_upper' in df.columns and df['bollinger_upper'].isnull().any():
            middle_band = (df['bollinger_upper'] + df['bollinger_lower']) / 2
            middle_band.fillna(df['Close'], inplace=True)
            volatility_proxy = df['ATR'] * 1.5
            df['bollinger_upper'].fillna(middle_band + volatility_proxy, inplace=True)
            df['bollinger_lower'].fillna(middle_band - volatility_proxy, inplace=True)
            safe_middle_band = middle_band.replace(0, np.nan)
            df['bollinger_bandwidth'].fillna(
                (df['bollinger_upper'] - df['bollinger_lower']) / safe_middle_band,
                inplace=True
            )

        # Drop temporary helper columns
        df.drop(columns=['volatility_regime_label', 'trend_regime_label', 'market_regime_str'], inplace=True, errors='ignore')
        return df

    def _add_higher_tf_state_features(self, base_df: pd.DataFrame, htf_df: pd.DataFrame, tf_name: str) -> pd.DataFrame:
        """
        Calculate state-based features on a higher timeframe (HTF) and merge them
        onto the base timeframe dataframe.
        """
        if htf_df.empty:
            return base_df

        # Calculate base indicators on the HTF data first
        htf_df['EMA_20'] = htf_df['Close'].ewm(span=20, adjust=False).mean()
        htf_df['EMA_50'] = htf_df['Close'].ewm(span=50, adjust=False).mean()
        htf_df['RSI'] = self._calculate_rsi_series(htf_df['Close'], period=14)
        exp1 = htf_df['Close'].ewm(span=12, adjust=False).mean()
        exp2 = htf_df['Close'].ewm(span=26, adjust=False).mean()
        htf_df['MACD'] = exp1 - exp2
        htf_df['MACD_hist'] = htf_df['MACD'].ewm(span=9, adjust=False).mean()

        # Create the new state-based features
        features_to_add = pd.DataFrame(index=htf_df.index)

        # EMA State
        features_to_add[f'ema_cross_{tf_name}_bullish'] = (htf_df['EMA_20'] > htf_df['EMA_50']).astype(int)
        features_to_add[f'ema_slope_{tf_name}_bullish'] = (htf_df['EMA_50'].diff() > 0).astype(int)

        # RSI State
        rsi_overbought = getattr(self.config, 'RSI_OVERBOUGHT', 70)
        rsi_oversold = getattr(self.config, 'RSI_OVERSOLD', 30)
        features_to_add[f'rsi_{tf_name}_overbought'] = (htf_df['RSI'] > rsi_overbought).astype(int)
        features_to_add[f'rsi_{tf_name}_oversold'] = (htf_df['RSI'] < rsi_oversold).astype(int)

        # MACD State
        features_to_add[f'macd_{tf_name}_bullish'] = ((htf_df['MACD'] > 0) & (htf_df['MACD_hist'] > 0)).astype(int)

        # Breakout State
        range_high = htf_df['High'].rolling(20).max()
        range_low = htf_df['Low'].rolling(20).min()
        features_to_add[f'breakout_up_{tf_name}'] = (htf_df['Close'] > range_high.shift(1)).astype(int)
        features_to_add[f'breakout_down_{tf_name}'] = (htf_df['Close'] < range_low.shift(1)).astype(int)

        # Volume State
        if 'RealVolume' in htf_df.columns:
            features_to_add[f'volume_surge_{tf_name}'] = (
                htf_df['RealVolume'] > htf_df['RealVolume'].rolling(20).mean() * 1.5
            ).astype(int)

        # Create a composite confirmation score
        features_to_add[f'confirm_score_{tf_name}'] = (
            features_to_add[f'ema_cross_{tf_name}_bullish'] +
            (features_to_add[f'rsi_{tf_name}_oversold'] == 0).astype(int) +  # Not oversold is bullish
            features_to_add[f'macd_{tf_name}_bullish'] +
            features_to_add[f'breakout_up_{tf_name}']
        )

        # Merge the new features onto the base dataframe
        if not base_df.index.is_monotonic_increasing:
            base_df = base_df.sort_index()

        merged_df = pd.merge_asof(
            left=base_df,
            right=features_to_add,
            left_index=True,
            right_index=True,
            direction='backward'
        )
        return merged_df

    def _calculate_session_features(self, df: pd.DataFrame, orb_minutes: int = 30) -> pd.DataFrame:
        """Calculate session-based features like previous high/low and opening range breakout."""
        if not isinstance(df.index, pd.DatetimeIndex):
            return df

        df['date'] = df.index.date

        # Calculate previous session's high and low, grouped by date
        daily_aggregates = df.groupby('date').agg(
            High=('High', 'max'),
            Low=('Low', 'min'),
            Open=('Open', 'first')
        )
        daily_aggregates['prev_session_high'] = daily_aggregates['High'].shift(1)
        daily_aggregates['prev_session_low'] = daily_aggregates['Low'].shift(1)
        df = df.join(daily_aggregates[['prev_session_high', 'prev_session_low']], on='date')

        # Fill the NaN values for the very first day using the Open price as a proxy
        df['prev_session_high'].fillna(df['Open'], inplace=True)
        df['prev_session_low'].fillna(df['Open'], inplace=True)

        # Calculate Opening Range Breakout (ORB)
        df['is_orb_breakout'] = 0
        df['orb_high'] = np.nan
        df['orb_low'] = np.nan

        for date, group in df.groupby('date'):
            start_time = group.index.min()
            orb_end_time = start_time + pd.Timedelta(minutes=orb_minutes)
            orb_mask = (group.index >= start_time) & (group.index <= orb_end_time)

            if not group[orb_mask].empty:
                orb_high = group[orb_mask]['High'].max()
                orb_low = group[orb_mask]['Low'].min()

                df.loc[group.index, 'orb_high'] = orb_high
                df.loc[group.index, 'orb_low'] = orb_low

                post_orb_mask = group.index > orb_end_time
                breakouts = (group['Close'] > orb_high) | (group['Close'] < orb_low)
                df.loc[group[post_orb_mask & breakouts].index, 'is_orb_breakout'] = 1

        # Calculate anchor price (session open)
        df['anchor_price'] = df.groupby('date')['Open'].transform('first')

        df.drop(columns=['date'], inplace=True)
        return df

    def _calculate_pattern_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate features based on candle patterns like inside bars."""
        prev_high = df['High'].shift(1)
        prev_low = df['Low'].shift(1)
        is_inside_bar = (df['High'] < prev_high) & (df['Low'] > prev_low)

        ib_groups = (is_inside_bar != is_inside_bar.shift()).cumsum()
        df['ii_insidebar_count'] = is_inside_bar.groupby(ib_groups).cumsum()

        is_pattern_start = (df['ii_insidebar_count'] == 1) & (is_inside_bar)
        mother_bar_high = df['High'].shift(1).where(is_pattern_start)
        df['high_ii_bar'] = mother_bar_high.ffill()

        return df

    def _calculate_behavioral_metrics(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate behavioral metrics like rolling drawdown and volatility spikes."""
        rolling_max = df['Close'].rolling(window=100, min_periods=30).max()
        df['drawdown_percent'] = (df['Close'] - rolling_max) / rolling_max.replace(0, np.nan)

        atr_ma = df['ATR'].rolling(window=20, min_periods=10).mean()
        df['volatility_spike'] = (df['ATR'] > atr_ma * 2.0).astype(int)

        return df

    def _add_external_data_features(self, df: pd.DataFrame, macro_data: Optional[pd.DataFrame]) -> pd.DataFrame:
        """Merge and add features from external data sources like VIX, with a fallback mechanism."""
        df_merged = df.copy()  # Start with a copy

        if macro_data is not None and not macro_data.empty:
            if not isinstance(macro_data.index, pd.DatetimeIndex):
                if 'Timestamp' in macro_data.columns:
                    macro_data = macro_data.set_index('Timestamp')
                else:
                    macro_data.index = pd.to_datetime(macro_data.index)

            df_merged = pd.merge_asof(
                left=df.sort_index(),
                right=macro_data.sort_index(),
                left_index=True,
                right_index=True,
                direction='backward'
            )

        # Process VIX data if present, otherwise prepare for fallback
        if 'VIX' in df_merged.columns and not df_merged['VIX'].isnull().all():
            df_merged.rename(columns={'VIX': 'vix'}, inplace=True)
            df_merged['vix_5d_avg'] = df_merged['vix'].rolling(window=5, min_periods=1).mean()
        else:
            # VIX data was not available or was all NaN
            df_merged['vix'] = np.nan
            df_merged['vix_5d_avg'] = np.nan

        # Apply fallback for VIX if the column is missing or all NaN
        if 'vix' not in df_merged.columns or df_merged['vix'].isnull().all():
            logger.warning("  - VIX data not found. Applying fallback using the asset's realized volatility.")
            if 'realized_volatility' in df_merged.columns and not df_merged['realized_volatility'].isnull().all():
                # Use the asset's own volatility as a proxy
                df_merged['vix'] = df_merged['realized_volatility']
                df_merged['vix_5d_avg'] = df_merged['vix'].rolling(window=5, min_periods=1).mean()
            else:
                # Final fallback: use a constant value
                logger.warning("  - No volatility data available. Using constant fallback for VIX.")
                df_merged['vix'] = 20.0  # Typical VIX level
                df_merged['vix_5d_avg'] = 20.0

        return df_merged

    def _add_placeholder_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add columns for features requiring data feeds not currently integrated."""
        placeholders = {
            'overreaction_flag': 0,
            'sentiment_score': 0.0,
            'news_sentiment_score': 0.0,
            'tweet_sentiment_count': 0,
            'tweet_sentiment_score': 0.0,
            'days_since_event': 999
        }
        for col, val in placeholders.items():
            if col not in df.columns:
                df[col] = val
        return df

    def _process_single_symbol_stack(self, symbol_data_by_tf: Dict[str, pd.DataFrame], macro_data: Optional[pd.DataFrame] = None) -> Optional[pd.DataFrame]:
        """
        Processes a stack of multi-timeframe data for a single symbol to engineer a rich set of features.
        """
        base_df_orig = symbol_data_by_tf.get(self.roles.get('base', 'D1'))
        if base_df_orig is None or base_df_orig.empty:
            logger.warning(f"No base timeframe data for symbol in _process_single_symbol_stack. Skipping.")
            return None

        df = base_df_orig.copy()

        # Ensure we have required columns
        required_cols = ['Open', 'High', 'Low', 'Close']
        if not all(col in df.columns for col in required_cols):
            logger.error(f"Missing required OHLC columns. Available: {df.columns.tolist()}")
            return None

        logger.info("    - Calculating foundational indicators (ATR, RSI, volatility)...")
        df = self._calculate_atr(df, period=14)
        df = self._calculate_rsi(df, period=14)

        # Calculate realized volatility and market volatility index
        if 'ATR' in df.columns:
            df['realized_volatility'] = df['Close'].pct_change().rolling(14, min_periods=7).std() * np.sqrt(252)
            rolling_window = 252
            min_p = max(1, rolling_window // 2)
            rolling_min = df['realized_volatility'].rolling(window=rolling_window, min_periods=min_p).min()
            rolling_max = df['realized_volatility'].rolling(window=rolling_window, min_periods=min_p).max()
            df['market_volatility_index'] = (df['realized_volatility'] - rolling_min) / (rolling_max - rolling_min).replace(0, np.nan)
            df['market_volatility_index'].fillna(0.5, inplace=True)
            df['market_volatility_index'].clip(0, 1, inplace=True)

        logger.info("    - Calculating session, pattern, and behavioral features...")
        df = self._calculate_session_features(df)
        df = self._calculate_pattern_features(df)
        df = self._calculate_behavioral_metrics(df)

        logger.info("    - Calculating technical indicators...")
        df = self._calculate_technical_indicators(df)

        logger.info("    - Calculating basic features...")
        df = self._calculate_simple_features(df)
        df = self._calculate_price_derivatives(df)
        df = self._calculate_volume_derivatives(df)
        df = self._calculate_statistical_moments(df)
        df = self._calculate_ohlc_ratios(df)

        logger.info("    - Calculating advanced features...")
        df = self._calculate_time_features(df)
        df = self._calculate_entropy_features(df)
        df = self._calculate_cycle_features(df)

        logger.info("    - Calculating advanced volatility features...")
        df = self._calculate_parkinson_volatility(df)
        df = self._calculate_yang_zhang_volatility(df)
        df = self._calculate_garch_volatility(df)
        df = self._calculate_hawkes_volatility(df)

        logger.info("    - Calculating advanced statistical features...")
        df = self._calculate_autocorrelation_features(df)
        df = self._calculate_fourier_transform_features(df)
        df = self._calculate_wavelet_features(df)

        logger.info("    - Calculating market microstructure features...")
        df = self._calculate_displacement(df)
        df = self._calculate_gaps(df)
        df = self._calculate_candle_info(df)
        df = self._calculate_accumulation_distribution(df)
        df = self._calculate_mad(df)
        df = self._calculate_price_volume_correlation(df)
        df = self._calculate_kama_regime(df)
        df = self._calculate_quantile_features(df)
        df = self._calculate_regression_features(df)
        df = self._calculate_relative_performance(df)

        logger.info("    - Detecting anomalies...")
        df = self._detect_anomalies(df)

        # Add EMA features from config
        logger.info("    - Calculating EMAs...")
        for period in getattr(self.config, 'EMA_PERIODS', [20, 50, 100, 200]):
            df[f'EMA_{period}'] = df['Close'].ewm(span=period, adjust=False, min_periods=1).mean()
            df[f'EMA_{period}_slope'] = df[f'EMA_{period}'].diff()
            df[f'EMA_{period}_slope_norm'] = df[f'EMA_{period}'].diff() / (df[f'EMA_{period}'] + 1e-9)

        # Add higher timeframe features if available
        for tf_name, htf_df in symbol_data_by_tf.items():
            if tf_name != self.roles.get('base', 'D1'):
                df = self._add_higher_tf_features(df, htf_df, tf_name)

        logger.info(f"    - Feature engineering complete. Shape: {df.shape}")
        return df

    def _add_higher_tf_features(self, base_df: pd.DataFrame, htf_df: pd.DataFrame, tf_name: str) -> pd.DataFrame:
        """Add features from higher timeframe data"""
        if htf_df.empty:
            return base_df

        # Calculate basic indicators on higher timeframe
        htf_features = htf_df.copy()

        # RSI
        delta = htf_features['Close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
        rs = gain / loss.replace(0, np.nan)
        htf_features[f'RSI_14_{tf_name}'] = 100 - (100 / (1 + rs))

        # EMA
        htf_features[f'EMA_20_{tf_name}'] = htf_features['Close'].ewm(span=20).mean()
        htf_features[f'EMA_50_{tf_name}'] = htf_features['Close'].ewm(span=50).mean()

        # Trend direction
        htf_features[f'trend_{tf_name}'] = (htf_features[f'EMA_20_{tf_name}'] > htf_features[f'EMA_50_{tf_name}']).astype(int)

        # Merge with base timeframe (forward fill)
        feature_cols = [col for col in htf_features.columns if tf_name in col]

        if isinstance(base_df.index, pd.DatetimeIndex) and isinstance(htf_features.index, pd.DatetimeIndex):
            for col in feature_cols:
                base_df[col] = base_df.index.to_series().map(
                    htf_features[col].reindex(base_df.index, method='ffill')
                )

        return base_df

    def _add_higher_tf_context(self, base_df: pd.DataFrame, higher_tf_df: Optional[pd.DataFrame], tf_name: str) -> pd.DataFrame:
        """
        Merges higher timeframe data onto the base dataframe using a robust, time-series-aware method.
        """
        # Define the contextual features to be carried over from the higher timeframe
        ctx_cols_map = {
            'Close': f"{tf_name}_ctx_Close", 'ATR': f"{tf_name}_ctx_ATR",
            'RSI': f"{tf_name}_ctx_RSI", 'ADX': f"{tf_name}_ctx_ADX",
            'RealVolume': f"{tf_name}_ctx_RealVolume"
        }

        # Guard clause: If there's no higher timeframe data, create placeholder NaN columns and exit.
        if higher_tf_df is None or higher_tf_df.empty:
            for col_name in ctx_cols_map.values():
                base_df[col_name] = np.nan
            base_df[f"{tf_name}_ctx_Trend"] = np.nan
            return base_df

        # Ensure both dataframes have a sorted DatetimeIndex, which is required for merge_asof
        if not isinstance(base_df.index, pd.DatetimeIndex) or not base_df.index.is_monotonic_increasing:
            base_df = base_df.sort_index()
        if not isinstance(higher_tf_df.index, pd.DatetimeIndex) or not higher_tf_df.index.is_monotonic_increasing:
            higher_tf_df = higher_tf_df.sort_index()

        # Select and rename columns from the higher timeframe DF to prevent merge conflicts
        cols_to_use = [col for col in ctx_cols_map.keys() if col in higher_tf_df.columns]
        if not cols_to_use:
            logger.warning(f"None of the desired context features found in higher timeframe '{tf_name}'.")
            return base_df

        higher_tf_subset = higher_tf_df[cols_to_use].rename(columns=ctx_cols_map)

        # Use merge_asof for a robust, time-series-aware join
        # This joins each row in `base_df` with the LAST available row from `higher_tf_subset`
        merged_df = pd.merge_asof(
            left=base_df,
            right=higher_tf_subset,
            left_index=True,
            right_index=True,
            direction='backward'  # Use the last known value from the higher timeframe
        )

        # Now, calculate the trend on the newly merged context column
        ctx_close_col = f"{tf_name}_ctx_Close"
        if ctx_close_col in merged_df.columns:
            merged_df[f"{tf_name}_ctx_Trend"] = np.sign(merged_df[ctx_close_col].diff(2)).fillna(0).astype(int)
        else:
            merged_df[f"{tf_name}_ctx_Trend"] = np.nan

        return merged_df

    def engineer_daily_benchmark_features(self, daily_df: pd.DataFrame, benchmark_df: pd.DataFrame) -> pd.DataFrame:
        """
        Engineers a set of daily benchmark-relative features, including a 200-period
        EMA trend filter for the SPY benchmark.
        """
        benchmark_ticker_clean = getattr(self.config, 'BENCHMARK_TICKER', '^SPY').lower().replace('^', '')

        if benchmark_df.empty or daily_df.empty:
            logger.warning(f"Benchmark ({getattr(self.config, 'BENCHMARK_TICKER', '^SPY')}) or daily asset data is empty; cannot engineer daily benchmark features.")
            return daily_df

        logger.info(f"-> Engineering strategic D1-to-D1 benchmark features against {getattr(self.config, 'BENCHMARK_TICKER', '^SPY')}...")

        try:
            df_asset = daily_df.copy()
            if isinstance(df_asset.index, pd.MultiIndex):
                df_asset.reset_index(inplace=True)
            if 'Timestamp' in df_asset.columns:
                df_asset = df_asset.set_index('Timestamp')
            if not isinstance(df_asset.index, pd.DatetimeIndex):
                df_asset.index = pd.to_datetime(df_asset.index)

            df_benchmark = benchmark_df.copy()
            if isinstance(df_benchmark.index, pd.MultiIndex):
                df_benchmark.reset_index(inplace=True)
            if 'Date' in df_benchmark.columns:
                df_benchmark = df_benchmark.set_index('Date')
            if not isinstance(df_benchmark.index, pd.DatetimeIndex):
                df_benchmark.index = pd.to_datetime(df_benchmark.index)
            if isinstance(df_benchmark.columns, pd.MultiIndex):
                df_benchmark.columns = df_benchmark.columns.get_level_values(0)

            df_benchmark[f'{benchmark_ticker_clean}_sma_200'] = df_benchmark['close'].rolling(window=200, min_periods=100).mean()
            df_benchmark[f'is_{benchmark_ticker_clean}_bullish'] = (df_benchmark['close'] > df_benchmark[f'{benchmark_ticker_clean}_sma_200']).fillna(False).astype(int)
            df_benchmark[f'{benchmark_ticker_clean}_returns'] = df_benchmark['close'].pct_change()

            benchmark_features_to_add = df_benchmark[[f'is_{benchmark_ticker_clean}_bullish', f'{benchmark_ticker_clean}_returns']]
            df_merged = pd.merge(df_asset, benchmark_features_to_add, left_index=True, right_index=True, how='left')
            df_merged[[f'is_{benchmark_ticker_clean}_bullish', f'{benchmark_ticker_clean}_returns']] = df_merged[[f'is_{benchmark_ticker_clean}_bullish', f'{benchmark_ticker_clean}_returns']].ffill()

            all_enriched_groups = []
            if 'Symbol' in df_merged.columns:
                for symbol, group_df in df_merged.groupby('Symbol'):
                    group_copy = group_df.copy()
                    group_copy['asset_returns'] = group_copy['Close'].pct_change()
                    group_copy[f'relative_strength_vs_{benchmark_ticker_clean}'] = (group_copy['asset_returns'] - group_copy[f'{benchmark_ticker_clean}_returns'])
                    group_copy[f'correlation_with_{benchmark_ticker_clean}'] = group_copy['asset_returns'].rolling(window=60, min_periods=30).corr(group_copy[f'{benchmark_ticker_clean}_returns'])
                    all_enriched_groups.append(group_copy)

                if not all_enriched_groups:
                    logger.warning("No symbol groups were processed for benchmark features.")
                    return daily_df

                final_df = pd.concat(all_enriched_groups).sort_index()
                final_df.drop(columns=[f'{benchmark_ticker_clean}_returns', 'asset_returns'], inplace=True, errors='ignore')
                return final_df
            else:
                logger.warning("No 'Symbol' column found in daily data to group by. Returning partially processed data.")
                return df_merged

        except Exception as e:
            logger.error("An unexpected error occurred during benchmark feature engineering.", exc_info=True)
            raise e

    def engineer_features(self, base_df: pd.DataFrame, data_by_tf: Dict[str, pd.DataFrame],
                         macro_data: Optional[pd.DataFrame] = None) -> pd.DataFrame:
        """
        Main feature engineering method that orchestrates the entire process.
        """
        logger.info("-> Stage 2: Engineering Full Feature Stack...")

        system_settings = get_optimal_system_settings()
        num_workers = system_settings.get('num_workers', 1)
        use_parallel = num_workers > 1
        logger.info(f"-> Processing mode selected: {'Parallel' if use_parallel else 'Serial'}")

        base_tf_name = self.roles.get('base', 'D1')
        if base_tf_name not in data_by_tf or data_by_tf[base_tf_name].empty:
            logger.critical(f"Base timeframe '{base_tf_name}' data is missing or empty. Cannot proceed.")
            return pd.DataFrame()

        # Group data by symbol
        symbols_data = defaultdict(dict)
        for tf_name, tf_df in data_by_tf.items():
            if 'Symbol' in tf_df.columns:
                for symbol in tf_df['Symbol'].unique():
                    symbol_data = tf_df[tf_df['Symbol'] == symbol].copy()
                    if not symbol_data.empty:
                        symbols_data[symbol][tf_name] = symbol_data
            else:
                # Single symbol case
                symbols_data['SINGLE_SYMBOL'][tf_name] = tf_df

        # Process each symbol
        all_processed_dfs = []

        if use_parallel and len(symbols_data) > 1:
            # Parallel processing
            try:
                with multiprocessing.Pool(processes=num_workers) as pool:
                    symbol_tuples = [(symbol, symbol_data_by_tf) for symbol, symbol_data_by_tf in symbols_data.items()]
                    results = pool.starmap(_parallel_process_symbol_wrapper,
                                         [(symbol_tuple, self) for symbol_tuple in symbol_tuples])

                    for result in results:
                        if result is not None:
                            all_processed_dfs.append(result)

            except Exception as e:
                logger.warning(f"Parallel processing failed: {e}. Falling back to serial processing.")
                use_parallel = False

        if not use_parallel:
            # Serial processing
            for symbol, symbol_data_by_tf in symbols_data.items():
                logger.info(f"  - Processing symbol: {symbol}")
                processed_df = self._process_single_symbol_stack(symbol_data_by_tf, macro_data)
                if processed_df is not None:
                    all_processed_dfs.append(processed_df)

        if not all_processed_dfs:
            logger.error("No data was successfully processed.")
            return pd.DataFrame()

        # Combine all processed data
        final_df = pd.concat(all_processed_dfs, ignore_index=True)

        logger.info("Applying post-processing steps...")
        final_df = self._impute_missing_values(final_df)

        # Add noise features for model robustness
        final_df['noise_1'] = np.random.normal(0, 1, len(final_df))
        final_df['noise_2'] = np.random.uniform(-1, 1, len(final_df))

        logger.info("Applying final data shift and cleaning...")
        feature_cols_to_shift = [c for c in final_df.columns if c not in self.NON_FEATURE_COLS]

        if 'Symbol' in final_df.columns:
            final_df[feature_cols_to_shift] = final_df.groupby('Symbol', group_keys=False)[feature_cols_to_shift].shift(1)
        else:
            final_df[feature_cols_to_shift] = final_df[feature_cols_to_shift].shift(1)

        final_df.dropna(subset=feature_cols_to_shift, how='all', inplace=True)

        logger.info(f"[SUCCESS] Feature engineering complete. Final dataset shape: {final_df.shape}")
        return final_df

    def _impute_missing_values(self, df: pd.DataFrame) -> pd.DataFrame:
        """Impute missing values in features"""
        feature_cols = [col for col in df.columns if col not in self.NON_FEATURE_COLS]

        for col in feature_cols:
            if df[col].isnull().any():
                # Forward fill first, then backward fill, then fill with median
                df[col] = df[col].fillna(method='ffill').fillna(method='bfill').fillna(df[col].median())

        return df


class TechnicalIndicators:
    """
    Collection of technical indicator calculations.
    Provides static methods for common technical analysis indicators.
    """

    @staticmethod
    def sma(data: pd.Series, period: int) -> pd.Series:
        """Simple Moving Average."""
        return data.rolling(window=period).mean()

    @staticmethod
    def ema(data: pd.Series, period: int) -> pd.Series:
        """Exponential Moving Average."""
        return data.ewm(span=period).mean()

    @staticmethod
    def rsi(data: pd.Series, period: int = 14) -> pd.Series:
        """Relative Strength Index."""
        delta = data.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        return 100 - (100 / (1 + rs))

    @staticmethod
    def bollinger_bands(data: pd.Series, period: int = 20, std_dev: float = 2) -> Dict[str, pd.Series]:
        """Bollinger Bands."""
        sma = data.rolling(window=period).mean()
        std = data.rolling(window=period).std()
        return {
            'upper': sma + (std * std_dev),
            'middle': sma,
            'lower': sma - (std * std_dev)
        }

    @staticmethod
    def macd(data: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9) -> Dict[str, pd.Series]:
        """MACD indicator."""
        ema_fast = data.ewm(span=fast).mean()
        ema_slow = data.ewm(span=slow).mean()
        macd_line = ema_fast - ema_slow
        signal_line = macd_line.ewm(span=signal).mean()
        histogram = macd_line - signal_line
        return {
            'macd': macd_line,
            'signal': signal_line,
            'histogram': histogram
        }


class StatisticalFeatures:
    """
    Collection of statistical feature calculations.
    Provides methods for statistical analysis of price data.
    """

    @staticmethod
    def rolling_statistics(data: pd.Series, window: int) -> Dict[str, pd.Series]:
        """Calculate rolling statistical features."""
        return {
            'mean': data.rolling(window=window).mean(),
            'std': data.rolling(window=window).std(),
            'skew': data.rolling(window=window).skew(),
            'kurt': data.rolling(window=window).kurt(),
            'min': data.rolling(window=window).min(),
            'max': data.rolling(window=window).max()
        }

    @staticmethod
    def price_momentum(data: pd.Series, periods: List[int]) -> Dict[str, pd.Series]:
        """Calculate price momentum over multiple periods."""
        momentum = {}
        for period in periods:
            momentum[f'momentum_{period}'] = data.pct_change(periods=period)
        return momentum

    @staticmethod
    def volatility_measures(data: pd.Series, window: int = 20) -> Dict[str, pd.Series]:
        """Calculate various volatility measures."""
        returns = data.pct_change()
        return {
            'realized_vol': returns.rolling(window=window).std() * np.sqrt(252),
            'parkinson_vol': np.sqrt(252 / window * np.log(data.rolling(window=window).max() / data.rolling(window=window).min()).rolling(window=window).sum()),
            'garman_klass_vol': returns.rolling(window=window).apply(lambda x: np.sqrt(np.mean(x**2)) * np.sqrt(252))
        }


class BehavioralPatterns:
    """
    Detection and analysis of behavioral patterns in price data.
    Identifies market microstructure and behavioral finance patterns.
    """

    @staticmethod
    def gap_analysis(data: pd.DataFrame) -> Dict[str, pd.Series]:
        """Analyze price gaps."""
        gaps = data['Open'] - data['Close'].shift(1)
        gap_pct = gaps / data['Close'].shift(1) * 100

        return {
            'gap_size': gaps,
            'gap_pct': gap_pct,
            'gap_up': (gaps > 0).astype(int),
            'gap_down': (gaps < 0).astype(int),
            'significant_gap': (np.abs(gap_pct) > 1).astype(int)
        }

    @staticmethod
    def candle_patterns(data: pd.DataFrame) -> Dict[str, pd.Series]:
        """Identify basic candlestick patterns."""
        body_size = np.abs(data['Close'] - data['Open'])
        upper_wick = data['High'] - np.maximum(data['Open'], data['Close'])
        lower_wick = np.minimum(data['Open'], data['Close']) - data['Low']

        return {
            'body_size': body_size,
            'upper_wick': upper_wick,
            'lower_wick': lower_wick,
            'doji': (body_size < (data['High'] - data['Low']) * 0.1).astype(int),
            'hammer': ((lower_wick > body_size * 2) & (upper_wick < body_size * 0.5)).astype(int),
            'shooting_star': ((upper_wick > body_size * 2) & (lower_wick < body_size * 0.5)).astype(int)
        }

    @staticmethod
    def volume_patterns(data: pd.DataFrame) -> Dict[str, pd.Series]:
        """Analyze volume patterns."""
        if 'Volume' not in data.columns:
            return {}

        volume_ma = data['Volume'].rolling(window=20).mean()

        return {
            'volume_ratio': data['Volume'] / volume_ma,
            'volume_spike': (data['Volume'] > volume_ma * 2).astype(int),
            'volume_dry_up': (data['Volume'] < volume_ma * 0.5).astype(int),
            'price_volume_trend': ((data['Close'] > data['Close'].shift(1)) & (data['Volume'] > volume_ma)).astype(int)
        }

    def _calculate_relative_strength(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate relative strength against benchmark (SPY)."""
        if 'Close' in df.columns and 'SPY_Close' in df.columns:
            df['relative_strength_vs_spy'] = (df['Close'] / df['SPY_Close'])
        else:
            df['relative_strength_vs_spy'] = np.nan
        return df

    def _calculate_benchmark_correlation(self, df: pd.DataFrame, window: int = 50) -> pd.DataFrame:
        """Calculate rolling correlation with benchmark."""
        if 'Close' in df.columns and 'SPY_Close' in df.columns:
            asset_returns = df['Close'].pct_change()
            spy_returns = df['SPY_Close'].pct_change()
            df['correlation_with_spy'] = asset_returns.rolling(window=window, min_periods=window//2).corr(spy_returns)
        else:
            df['correlation_with_spy'] = np.nan
        return df

    def _calculate_benchmark_trend_filter(self, df: pd.DataFrame, window: int = 200) -> pd.DataFrame:
        """Calculate benchmark trend filter."""
        if 'SPY_Close' in df.columns:
            spy_sma = df['SPY_Close'].rolling(window=window, min_periods=window//2).mean()
            df['is_spy_bullish'] = (df['SPY_Close'] > spy_sma).astype(int)
        else:
            df['is_spy_bullish'] = 0
        return df

    def _apply_pca_reduction(self, df: pd.DataFrame, fitted_pca_pipeline: Optional[Pipeline] = None) -> pd.DataFrame:
        """
        Apply PCA reduction to high-dimensional feature space.

        Args:
            df: DataFrame with features
            fitted_pca_pipeline: Pre-fitted PCA pipeline, if None will fit new one

        Returns:
            DataFrame with PCA-reduced features
        """
        try:
            from sklearn.decomposition import PCA
            from sklearn.preprocessing import StandardScaler
            from sklearn.pipeline import Pipeline

            # Get numeric features for PCA
            numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()

            # Exclude target columns and non-feature columns
            exclude_cols = ['Symbol', 'target_signal_pressure_class_h30', 'signal_pressure',
                          'target_timing_score', 'target_bullish_engulfing', 'target_bearish_engulfing',
                          'target_volatility_spike']

            pca_features = [col for col in numeric_cols if col not in exclude_cols and not col.startswith('target_')]

            if len(pca_features) < 10:  # Need minimum features for PCA
                logger.warning(f"Insufficient features for PCA: {len(pca_features)}")
                return df

            # Prepare data
            X = df[pca_features].fillna(0)

            if fitted_pca_pipeline is None:
                # Fit new PCA pipeline
                n_components = min(50, len(pca_features) // 2)  # Reduce dimensionality
                pca_pipeline = Pipeline([
                    ('scaler', StandardScaler()),
                    ('pca', PCA(n_components=n_components, random_state=42))
                ])

                X_pca = pca_pipeline.fit_transform(X)
            else:
                # Use pre-fitted pipeline
                X_pca = fitted_pca_pipeline.transform(X)

            # Create PCA feature names
            n_components = X_pca.shape[1]
            pca_feature_names = [f'pca_component_{i+1}' for i in range(n_components)]

            # Add PCA features to dataframe
            for i, feature_name in enumerate(pca_feature_names):
                df[feature_name] = X_pca[:, i]

            logger.info(f"Applied PCA reduction: {len(pca_features)} -> {n_components} components")
            return df

        except Exception as e:
            logger.error(f"Error in PCA reduction: {e}")
            return df

    def _log_feature_nan_stats(self, df: pd.DataFrame, stage: str):
        """Logs the percentage of NaN values for each feature at a given stage."""
        nan_stats = {}
        for col in df.columns:
            if col not in ['Symbol', 'Timestamp']:
                nan_pct = (df[col].isnull().sum() / len(df)) * 100
                if nan_pct > 0:
                    nan_stats[col] = nan_pct

        if nan_stats:
            logger.info(f"NaN statistics at {stage}:")
            for col, pct in sorted(nan_stats.items(), key=lambda x: x[1], reverse=True)[:10]:
                logger.info(f"  {col}: {pct:.1f}% NaN")
        else:
            logger.info(f"No NaN values found at {stage}")


class AnomalyDetector:
    """
    Detects anomalies and outliers in market data.
    Uses statistical and machine learning methods for anomaly detection.
    """

    def __init__(self, contamination: float = 0.1):
        """
        Initialize anomaly detector.

        Args:
            contamination: Expected proportion of outliers
        """
        self.contamination = contamination

    def detect_price_anomalies(self, data: pd.DataFrame) -> Dict[str, pd.Series]:
        """Detect price anomalies using statistical methods."""
        anomalies = {}

        for col in ['Open', 'High', 'Low', 'Close']:
            if col in data.columns:
                # Z-score based detection
                z_scores = np.abs((data[col] - data[col].mean()) / data[col].std())
                anomalies[f'{col}_zscore_anomaly'] = (z_scores > 3).astype(int)

                # IQR based detection
                Q1 = data[col].quantile(0.25)
                Q3 = data[col].quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - 1.5 * IQR
                upper_bound = Q3 + 1.5 * IQR
                anomalies[f'{col}_iqr_anomaly'] = ((data[col] < lower_bound) | (data[col] > upper_bound)).astype(int)

        return anomalies

    def detect_volume_anomalies(self, data: pd.DataFrame) -> Dict[str, pd.Series]:
        """Detect volume anomalies."""
        if 'Volume' not in data.columns:
            return {}

        volume_ma = data['Volume'].rolling(window=20).mean()
        volume_std = data['Volume'].rolling(window=20).std()

        return {
            'volume_spike_anomaly': (data['Volume'] > volume_ma + 3 * volume_std).astype(int),
            'volume_drought_anomaly': (data['Volume'] < volume_ma - 2 * volume_std).astype(int)
        }

    def detect_return_anomalies(self, data: pd.DataFrame) -> Dict[str, pd.Series]:
        """Detect return anomalies."""
        returns = data['Close'].pct_change()

        # Rolling statistics
        rolling_mean = returns.rolling(window=20).mean()
        rolling_std = returns.rolling(window=20).std()

        return {
            'return_anomaly': (np.abs(returns - rolling_mean) > 3 * rolling_std).astype(int),
            'extreme_positive_return': (returns > rolling_mean + 3 * rolling_std).astype(int),
            'extreme_negative_return': (returns < rolling_mean - 3 * rolling_std).astype(int)
        }
