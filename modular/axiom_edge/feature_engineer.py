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
    Comprehensive feature engineering class that creates 200+ technical, statistical,
    and behavioral features from raw OHLCV data.
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
        self.config = config
        self.roles = timeframe_roles
        self.playbook = playbook
        self.hurst_warning_symbols = set()

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
