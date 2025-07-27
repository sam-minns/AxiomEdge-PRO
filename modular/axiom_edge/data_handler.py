# =============================================================================
# DATA HANDLER & COLLECTION MODULE
# =============================================================================

import os
import pickle
import requests
import pandas as pd
import numpy as np
import pathlib
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
from collections import defaultdict
import logging
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)

class DataHandler:
    """
    Market data handler with intelligent two-tier caching system.

    Provides efficient data retrieval with automatic caching to minimize API calls
    and improve performance. Uses both in-memory and persistent disk caching.

    Cache hierarchy:
        1. In-memory cache for session-speed access
        2. On-disk cache for persistence between sessions
        3. API fetch as last resort

    Attributes:
        cache: In-memory cache dictionary
        cache_path: Path to disk cache directory
    """

    def __init__(self, cache_dir: str = "./data_cache", api_key: Optional[str] = None):
        """
        Initialize the data handler with caching directories.

        Args:
            cache_dir: Directory path for persistent cache storage
            api_key: Optional API key for data providers
        """
        self.cache = {}
        self.cache_path = pathlib.Path(cache_dir)
        self.cache_path.mkdir(parents=True, exist_ok=True)
        self.api_key = api_key or os.getenv("FINANCIAL_API_KEY")
        print(f"Data cache initialized at: {self.cache_path.resolve()}")
        
    def get_data(self, symbol: str, start_date: str, end_date: str,
                 timeframe: str = "1D", source: str = "yahoo") -> pd.DataFrame:
        """
        Retrieve market data using intelligent caching hierarchy.

        Checks in-memory cache first, then disk cache, and finally fetches from API
        if data is not found in either cache.

        Args:
            symbol: Trading symbol to retrieve
            start_date: Start date in YYYY-MM-DD format
            end_date: End date in YYYY-MM-DD format
            timeframe: Data timeframe (1D, 1H, etc.)
            source: Data source (yahoo, alpha_vantage, polygon)

        Returns:
            DataFrame with OHLCV data
        """
        # Sanitize symbol for safe filename
        safe_symbol = symbol.replace('/', '_').replace('=', '')
        cache_key = f"{safe_symbol}_{timeframe}_{start_date}_{end_date}"
        pickle_file_path = self.cache_path / f"{cache_key}.pkl"

        # Check in-memory cache first
        if cache_key in self.cache:
            print(f"Loading {symbol} from in-memory cache.")
            return self.cache[cache_key].copy()

        # Check on-disk cache
        if pickle_file_path.exists():
            print(f"Loading {symbol} from on-disk cache: {pickle_file_path}")
            try:
                with open(pickle_file_path, 'rb') as f:
                    df = pickle.load(f)
                self.cache[cache_key] = df
                return df.copy()
            except Exception as e:
                logger.warning(f"Failed to load from disk cache: {e}")

        # Fetch from API as last resort
        print(f"Fetching {symbol} from {source} API...")
        df = self._fetch_from_api(symbol, start_date, end_date, timeframe, source)

        if not df.empty:
            print(f"Saving {symbol} data to both caches.")
            self.cache[cache_key] = df
            try:
                with open(pickle_file_path, 'wb') as f:
                    pickle.dump(df, f)
            except Exception as e:
                logger.warning(f"Failed to cache to disk: {e}")

        return df.copy()

    def _fetch_from_api(self, symbol: str, start_date: str, end_date: str,
                       timeframe: str = "1D", source: str = "yahoo") -> pd.DataFrame:
        """
        Fetch market data from external API.

        Uses multiple data providers with automatic column standardization.

        Args:
            symbol: Trading symbol to fetch
            start_date: Start date in YYYY-MM-DD format
            end_date: End date in YYYY-MM-DD format
            timeframe: Data timeframe
            source: Data source provider

        Returns:
            DataFrame with OHLCV data or empty DataFrame if fetch fails
        """
        try:
            if source == "yahoo":
                return self._fetch_yahoo_finance(symbol, start_date, end_date)
            elif source == "alpha_vantage":
                return self._fetch_alpha_vantage(symbol, timeframe)
            elif source == "polygon":
                return self._fetch_polygon(symbol, start_date, end_date, timeframe)
            else:
                logger.error(f"Unsupported data source: {source}")
                return pd.DataFrame()
        except Exception as e:
            print(f"Error fetching data for {symbol}: {e}")
            return pd.DataFrame()

    def _fetch_alpha_vantage(self, symbol: str, timeframe: str) -> pd.DataFrame:
        """Fetch data from Alpha Vantage API"""
        if not self.api_key:
            logger.error("Alpha Vantage API key not provided")
            return pd.DataFrame()
            
        function_map = {
            "1D": "TIME_SERIES_DAILY",
            "1H": "TIME_SERIES_INTRADAY",
            "5M": "TIME_SERIES_INTRADAY"
        }
        
        function = function_map.get(timeframe, "TIME_SERIES_DAILY")
        url = f"https://www.alphavantage.co/query"
        
        params = {
            "function": function,
            "symbol": symbol,
            "apikey": self.api_key,
            "outputsize": "full"
        }
        
        if timeframe in ["1H", "5M"]:
            params["interval"] = timeframe.lower()
            
        response = requests.get(url, params=params)
        data = response.json()
        
        # Parse the response based on the function type
        if "Time Series" in str(data):
            time_series_key = [k for k in data.keys() if "Time Series" in k][0]
            df = pd.DataFrame.from_dict(data[time_series_key], orient='index')
            df.index = pd.to_datetime(df.index)
            df = df.astype(float)
            
            # Standardize column names
            df.columns = ['Open', 'High', 'Low', 'Close', 'Volume']
            df = df.sort_index()
            
            return df
        else:
            logger.error(f"API Error: {data}")
            return pd.DataFrame()

    def _fetch_yahoo_finance(self, symbol: str, start_date: str, end_date: str) -> pd.DataFrame:
        """
        Fetch data from Yahoo Finance with improved error handling and standardization.

        Args:
            symbol: Trading symbol to fetch
            start_date: Start date in YYYY-MM-DD format
            end_date: End date in YYYY-MM-DD format

        Returns:
            DataFrame with standardized OHLCV data
        """
        try:
            import yfinance as yf
            data = yf.download(symbol, start=start_date, end=end_date, progress=False, auto_adjust=True)
            if data.empty:
                print(f"Warning: No data returned for {symbol} for the given date range.")
                return pd.DataFrame()

            # Standardize column names to lowercase
            data.rename(columns={
                'Open': 'open', 'High': 'high', 'Low': 'low',
                'Close': 'close', 'Volume': 'volume'
            }, inplace=True)
            return data[['open', 'high', 'low', 'close', 'volume']]
        except ImportError:
            logger.error("yfinance package not installed. Install with: pip install yfinance")
            return pd.DataFrame()
        except Exception as e:
            print(f"Yahoo Finance error for {symbol}: {e}")
            return pd.DataFrame()

    def _fetch_polygon(self, symbol: str, start_date: str, end_date: str, timeframe: str) -> pd.DataFrame:
        """Fetch data from Polygon.io API"""
        if not self.api_key:
            logger.error("Polygon API key not provided")
            return pd.DataFrame()
            
        # Convert timeframe to Polygon format
        timespan_map = {
            "1D": "day",
            "1H": "hour", 
            "5M": "minute"
        }
        
        timespan = timespan_map.get(timeframe, "day")
        multiplier = 5 if timeframe == "5M" else 1
        
        url = f"https://api.polygon.io/v2/aggs/ticker/{symbol}/range/{multiplier}/{timespan}/{start_date}/{end_date}"
        
        params = {"apikey": self.api_key}
        response = requests.get(url, params=params)
        data = response.json()
        
        if "results" in data:
            df = pd.DataFrame(data["results"])
            df['timestamp'] = pd.to_datetime(df['t'], unit='ms')
            df.set_index('timestamp', inplace=True)
            
            # Rename columns to standard format
            df = df.rename(columns={
                'o': 'Open',
                'h': 'High', 
                'l': 'Low',
                'c': 'Close',
                'v': 'Volume'
            })
            
            return df[['Open', 'High', 'Low', 'Close', 'Volume']]
        else:
            logger.error(f"Polygon API Error: {data}")
            return pd.DataFrame()

    def get_broker_spreads(self, symbols: List[str], broker: str = "oanda") -> Dict[str, float]:
        """
        Collect current spread information for given symbols from broker APIs.
        This can be used independently for broker information tasks.
        """
        spreads = {}
        
        for symbol in symbols:
            try:
                if broker.lower() == "oanda":
                    spread = self._get_oanda_spread(symbol)
                elif broker.lower() == "interactive_brokers":
                    spread = self._get_ib_spread(symbol)
                else:
                    logger.warning(f"Unsupported broker: {broker}")
                    spread = None
                    
                if spread is not None:
                    spreads[symbol] = spread
                    logger.info(f"{symbol} spread: {spread:.5f}")
                    
            except Exception as e:
                logger.error(f"Error getting spread for {symbol}: {e}")
                
        return spreads

    def _get_oanda_spread(self, symbol: str) -> Optional[float]:
        """Get current spread from OANDA API"""
        # This would require OANDA API credentials and implementation
        # Placeholder implementation
        logger.info(f"Getting OANDA spread for {symbol}")
        return None

    def _get_ib_spread(self, symbol: str) -> Optional[float]:
        """Get current spread from Interactive Brokers API"""
        # This would require IB API implementation
        # Placeholder implementation  
        logger.info(f"Getting IB spread for {symbol}")
        return None

    def clear_cache(self):
        """Clear both in-memory and disk cache"""
        self.cache.clear()
        for file in self.cache_path.glob("*.pkl"):
            file.unlink()
        logger.info("Cache cleared")

    def get_cache_info(self) -> Dict[str, Any]:
        """Get information about current cache state"""
        disk_files = list(self.cache_path.glob("*.pkl"))
        return {
            "in_memory_items": len(self.cache),
            "disk_cache_files": len(disk_files),
            "cache_directory": str(self.cache_path),
            "total_cache_size_mb": sum(f.stat().st_size for f in disk_files) / (1024 * 1024)
        }

    def preload_data(self, symbols: List[str], start_date: str, end_date: str,
                     timeframe: str = "1D", source: str = "yahoo") -> Dict[str, pd.DataFrame]:
        """
        Preload data for multiple symbols to warm up the cache.

        Args:
            symbols: List of trading symbols to preload
            start_date: Start date in YYYY-MM-DD format
            end_date: End date in YYYY-MM-DD format
            timeframe: Data timeframe
            source: Data source provider

        Returns:
            Dictionary mapping symbols to their data
        """
        data_dict = {}
        print(f"Preloading data for {len(symbols)} symbols...")

        for i, symbol in enumerate(symbols, 1):
            print(f"  [{i}/{len(symbols)}] Loading {symbol}...")
            try:
                data = self.get_data(symbol, start_date, end_date, timeframe, source)
                if not data.empty:
                    data_dict[symbol] = data
                    print(f"    ✓ Loaded {len(data)} records")
                else:
                    print(f"    ✗ No data available")
            except Exception as e:
                print(f"    ✗ Error: {e}")

        print(f"Preloading complete. Successfully loaded {len(data_dict)}/{len(symbols)} symbols.")
        return data_dict

    def get_available_symbols(self) -> List[str]:
        """
        Get list of symbols available in cache.

        Returns:
            List of symbols that have cached data
        """
        symbols = set()
        for cache_key in self.cache.keys():
            symbol = cache_key.split('_')[0]
            symbols.add(symbol)

        # Also check disk cache
        for file_path in self.cache_path.glob("*.pkl"):
            symbol = file_path.stem.split('_')[0]
            symbols.add(symbol)

        return sorted(list(symbols))


class DataLoader:
    """
    Handles loading and parsing of local data files.
    Supports multiple timeframes and file formats.
    """
    
    def __init__(self, config):
        self.config = config

    def load_and_parse_data(self, filenames: List[str]) -> Tuple[Optional[Dict[str, pd.DataFrame]], List[str]]:
        """Load and parse multiple data files organized by timeframe"""
        logger.info("-> Stage 1: Loading and Preparing Multi-Timeframe Data...")
        data_by_tf = defaultdict(list)
        
        logger.info(f"  - Found {len(filenames)} data files to process.")
        for i, filename in enumerate(filenames):
            logger.info(f"    - [{i+1}/{len(filenames)}] Parsing '{filename}'...")
            file_path = os.path.join(self.config.BASE_PATH, filename)
            if not os.path.exists(file_path):
                logger.warning(f"      - File not found, skipping: {file_path}")
                continue
            df = self._parse_single_file(file_path, filename)
            if df is not None:
                tf = filename.split('_')[1]
                data_by_tf[tf].append(df)
                logger.info(f"      - Successfully parsed. Shape: {df.shape}")

        if not data_by_tf:
            logger.critical("No valid data files were loaded. Cannot proceed.")
            return None, []

        # Process data for each timeframe (matching main file logic)
        processed_dfs: Dict[str, pd.DataFrame] = {}
        for tf, dfs in data_by_tf.items():
            if dfs:
                combined = pd.concat(dfs).sort_index()

                # --- Remove duplicate timestamps on a per-symbol basis ---
                if 'Symbol' in combined.columns:
                    original_rows = len(combined)
                    # Use reset_index to bring Timestamp into a column for de-duplication
                    combined.reset_index(inplace=True)
                    combined.drop_duplicates(subset=['Timestamp', 'Symbol'], keep='first', inplace=True)
                    # Restore the Timestamp index
                    combined.set_index('Timestamp', inplace=True)
                    rows_removed = original_rows - len(combined)
                    if rows_removed > 0:
                        logger.warning(f"  - Removed {rows_removed} duplicate timestamp entries for timeframe {tf}.")

                # Ensure data is sorted by timestamp before returning
                final_combined = combined.sort_index()
                processed_dfs[tf] = final_combined
                logger.info(f"  - Combined data for {tf}: {len(final_combined):,} rows for {len(final_combined['Symbol'].unique())} symbols.")

        detected_timeframes = list(processed_dfs.keys())
        if not processed_dfs:
            logger.critical("  - Data loading failed for all files.")
            return None, []

        logger.info(f"[SUCCESS] Data loading complete. Detected timeframes: {detected_timeframes}")
        return processed_dfs, detected_timeframes

    def _parse_single_file(self, file_path: str, filename: str) -> Optional[pd.DataFrame]:
        """Parse a single data file matching main file implementation"""
        try:
            parts = filename.split('_'); symbol, tf = parts[0], parts[1]
            df = pd.read_csv(file_path, delimiter='\t' if '\t' in open(file_path, encoding='utf-8').readline() else ',')
            df.columns = [c.upper().replace('<', '').replace('>', '') for c in df.columns]

            # 1. Find the separate DATE and TIME columns that were likely split on read
            date_col = next((c for c in df.columns if 'DATE' in c), None)
            time_col = next((c for c in df.columns if 'TIME' in c), None)

            # 2. Recombine them into a single, robust Timestamp column
            if date_col and time_col:
                df['Timestamp'] = pd.to_datetime(df[date_col] + ' ' + df[time_col], errors='coerce')
                # 3. FIX: Drop the now-redundant intermediate columns to prevent NaN issues later
                df.drop(columns=[date_col, time_col], inplace=True)
            elif date_col:
                df['Timestamp'] = pd.to_datetime(df[date_col], errors='coerce')
                df.drop(columns=[date_col], inplace=True)
            else:
                logger.error(f"  - No date/time columns found in {filename}."); return None

            # 4. Set the robust Timestamp as the official index for all future operations
            df.dropna(subset=['Timestamp'], inplace=True)
            df.set_index('Timestamp', inplace=True)

            # Rename standard columns for consistency
            col_map = {c: c.capitalize() for c in df.columns if c.lower() in ['open', 'high', 'low', 'close', 'tickvol', 'volume', 'spread']}
            df.rename(columns=col_map, inplace=True)
            vol_col = 'Volume' if 'Volume' in df.columns else 'Tickvol'
            df.rename(columns={vol_col: 'RealVolume'}, inplace=True, errors='ignore')

            df['Symbol'] = symbol

            # Ensure numeric types are correct
            for col in df.columns:
                if df[col].dtype == 'object' and col != 'Symbol':
                    df[col] = pd.to_numeric(df[col], errors='coerce')

            if 'RealVolume' not in df.columns: df['RealVolume'] = 0
            df['RealVolume'] = pd.to_numeric(df['RealVolume'], errors='coerce').fillna(0).astype('int32')
            for col in ['Open', 'High', 'Low', 'Close']:
                if col in df.columns:
                    df[col] = pd.to_numeric(df[col], errors='coerce').astype('float32')

            return df
        except Exception as e:
            logger.error(f"  - Failed to load {filename}: {e}", exc_info=True)
            return None


class CacheManager:
    """
    Manages caching operations for the data handler.
    Provides both in-memory and persistent disk caching.
    """

    def __init__(self, cache_path: str = None):
        """
        Initialize cache manager.

        Args:
            cache_path: Path for persistent cache storage
        """
        self.cache_path = cache_path
        self.memory_cache = {}
        self.cache_metadata = {}

    def get_cache_key(self, symbol: str, timeframe: str, start_date: str = None, end_date: str = None) -> str:
        """Generate cache key for data request."""
        key_parts = [symbol, timeframe]
        if start_date:
            key_parts.append(start_date)
        if end_date:
            key_parts.append(end_date)
        return "_".join(key_parts)

    def is_cached(self, cache_key: str) -> bool:
        """Check if data is available in cache."""
        return cache_key in self.memory_cache or (
            self.cache_path and os.path.exists(f"{self.cache_path}_{cache_key}.parquet")
        )

    def get_from_cache(self, cache_key: str) -> Optional[pd.DataFrame]:
        """Retrieve data from cache."""
        try:
            # Try memory cache first
            if cache_key in self.memory_cache:
                return self.memory_cache[cache_key].copy()

            # Try disk cache
            if self.cache_path:
                cache_file = f"{self.cache_path}_{cache_key}.parquet"
                if os.path.exists(cache_file):
                    data = pd.read_parquet(cache_file)
                    # Store in memory cache for faster access
                    self.memory_cache[cache_key] = data.copy()
                    return data

            return None

        except Exception as e:
            logger.error(f"Error retrieving from cache: {e}")
            return None

    def store_in_cache(self, cache_key: str, data: pd.DataFrame) -> bool:
        """Store data in cache."""
        try:
            # Store in memory cache
            self.memory_cache[cache_key] = data.copy()

            # Store in disk cache if path is provided
            if self.cache_path:
                cache_file = f"{self.cache_path}_{cache_key}.parquet"
                data.to_parquet(cache_file)

            # Update metadata
            self.cache_metadata[cache_key] = {
                'timestamp': datetime.now().isoformat(),
                'rows': len(data),
                'columns': len(data.columns)
            }

            return True

        except Exception as e:
            logger.error(f"Error storing in cache: {e}")
            return False

    def clear_cache(self, pattern: str = None) -> int:
        """Clear cache entries matching pattern."""
        cleared_count = 0

        try:
            if pattern:
                # Clear specific pattern
                keys_to_remove = [key for key in self.memory_cache.keys() if pattern in key]
            else:
                # Clear all
                keys_to_remove = list(self.memory_cache.keys())

            for key in keys_to_remove:
                if key in self.memory_cache:
                    del self.memory_cache[key]
                    cleared_count += 1

                # Remove disk cache file
                if self.cache_path:
                    cache_file = f"{self.cache_path}_{key}.parquet"
                    if os.path.exists(cache_file):
                        os.remove(cache_file)

            return cleared_count

        except Exception as e:
            logger.error(f"Error clearing cache: {e}")
            return 0


class DataValidator:
    """
    Validates market data quality and consistency.
    Provides comprehensive data quality checks.
    """

    @staticmethod
    def validate_ohlc_data(data: pd.DataFrame) -> Dict[str, Any]:
        """
        Validate OHLC data for consistency and quality.

        Args:
            data: DataFrame with OHLC data

        Returns:
            Validation results dictionary
        """
        validation_results = {
            'is_valid': True,
            'warnings': [],
            'errors': [],
            'statistics': {}
        }

        try:
            required_columns = ['Open', 'High', 'Low', 'Close']
            missing_columns = [col for col in required_columns if col not in data.columns]

            if missing_columns:
                validation_results['is_valid'] = False
                validation_results['errors'].append(f"Missing required columns: {missing_columns}")
                return validation_results

            # Check for negative prices
            negative_prices = (data[required_columns] < 0).any().any()
            if negative_prices:
                validation_results['warnings'].append("Negative prices detected")

            # Check OHLC relationships
            invalid_high = (data['High'] < data[['Open', 'Low', 'Close']].max(axis=1)).sum()
            invalid_low = (data['Low'] > data[['Open', 'High', 'Close']].min(axis=1)).sum()

            if invalid_high > 0:
                validation_results['warnings'].append(f"{invalid_high} rows with invalid High prices")

            if invalid_low > 0:
                validation_results['warnings'].append(f"{invalid_low} rows with invalid Low prices")

            # Check for missing data
            missing_data_pct = (data[required_columns].isnull().sum() / len(data) * 100).max()
            if missing_data_pct > 5:
                validation_results['warnings'].append(f"High missing data: {missing_data_pct:.1f}%")

            # Calculate statistics
            validation_results['statistics'] = {
                'total_rows': len(data),
                'missing_data_pct': missing_data_pct,
                'date_range': {
                    'start': data.index.min().isoformat() if not data.empty else None,
                    'end': data.index.max().isoformat() if not data.empty else None
                },
                'price_range': {
                    'min': data[required_columns].min().min(),
                    'max': data[required_columns].max().max()
                }
            }

            return validation_results

        except Exception as e:
            validation_results['is_valid'] = False
            validation_results['errors'].append(f"Validation error: {e}")
            return validation_results


def determine_timeframe_roles(detected_tfs: List[str]) -> Dict[str, Optional[str]]:
    """
    Determines the role of each timeframe based on available data.
    """
    if not detected_tfs:
        raise ValueError("No timeframes were detected from data files.")

    # Sort timeframes by their minute values for logical assignment
    tf_map = {'M1': 1, 'M5': 5, 'M15': 15, 'M30': 30, 'H1': 60, 'H4': 240, 'D1': 1440, 'DAILY': 1440, 'W1': 10080, 'MN1': 43200}
    sorted_tfs = sorted(detected_tfs, key=lambda x: tf_map.get(x, 999999))

    roles = {}

    # Assign roles based on available timeframes
    if len(sorted_tfs) == 1:
        roles[sorted_tfs[0]] = 'primary'
    elif len(sorted_tfs) == 2:
        roles[sorted_tfs[0]] = 'primary'
        roles[sorted_tfs[1]] = 'secondary'
    else:
        roles[sorted_tfs[0]] = 'primary'
        roles[sorted_tfs[1]] = 'secondary'
        roles[sorted_tfs[2]] = 'tertiary'
        # Additional timeframes get None role
        for tf in sorted_tfs[3:]:
            roles[tf] = None

    logger.info(f"Timeframe roles assigned: {roles}")
    return roles


def get_macro_context_data(
    tickers: Dict[str, str],
    period: str = "10y",
    results_dir: str = "Results"
) -> pd.DataFrame:
    """
    Fetches and caches macroeconomic time series data.

    Downloads and processes time series data for given macroeconomic tickers,
    such as VIX or Treasury yields. It includes robust caching and fallback
    mechanisms to ensure data availability and prevent datetime column errors.

    Args:
        tickers: A dictionary mapping a desired name to its yfinance ticker.
        period: The historical data period to fetch (e.g., "10y").
        results_dir: The directory to store cached data.

    Returns:
        A pandas DataFrame containing the requested macroeconomic data with a
        standardised 'Timestamp' column.
    """
    import yfinance as yf
    import json

    logger.info(f"-> Fetching macro time series: {list(tickers.keys())}...")
    cache_dir = os.path.join(results_dir)
    os.makedirs(cache_dir, exist_ok=True)
    parquet_path = os.path.join(cache_dir, "macro_data.parquet")
    meta_path = os.path.join(cache_dir, "macro_cache_metadata.json")

    logger.info("  - Downloading bulk tickers...")
    all_data = yf.download(list(tickers.values()), period=period,
                           progress=False, auto_adjust=True)
    close_prices = pd.DataFrame()

    if not all_data.empty:
        closes = all_data.get('Close', pd.DataFrame()).copy()
        if isinstance(closes, pd.Series):
            closes = closes.to_frame(name=list(tickers.values())[0])
        closes.dropna(axis=1, how='all', inplace=True)
        close_prices = closes

    vix_symbol = "^VIX"
    if vix_symbol in tickers.values() and vix_symbol not in close_prices.columns:
        logger.warning(f"  - VIX missing in bulk fetch. Trying dedicated fetch for {vix_symbol}...")
        try:
            vix_data = yf.download(vix_symbol, period=period, progress=False, auto_adjust=True)
            if not vix_data.empty and not vix_data['Close'].isnull().all():
                vs = vix_data['Close'].rename(vix_symbol)
                close_prices = close_prices.join(vs, how='outer') if not close_prices.empty else vs.to_frame()
                logger.info("  - Successfully fetched VIX separately.")
            else:
                logger.error("  - VIX fetch returned no valid data.")
        except Exception as e:
            logger.error(f"  - Exception during VIX fetch: {e}")

    if close_prices.empty:
        logger.error("  - No macro data available after attempts.")
        return pd.DataFrame()

    close_prices.rename(columns={v: k for k, v in tickers.items() if v in close_prices}, inplace=True)
    close_prices.ffill(inplace=True)
    close_prices.to_parquet(parquet_path)
    meta = {"tickers": list(tickers.keys()), "last_date": close_prices.index.max().strftime("%Y-%m-%d")}
    with open(meta_path, 'w') as f:
        json.dump(meta, f, indent=4)
    logger.info("  - Macro data cached.")

    # Standardise the index name before resetting it into a column.
    close_prices.index.name = 'Timestamp'
    df = close_prices.reset_index()
    return df


def _get_available_features_from_df(df: pd.DataFrame) -> List[str]:
    """
    Identifies and returns a list of feature columns from a dataframe,
    excluding non-feature columns like OHLC, target, or identifiers.
    """
    if df is None or df.empty:
        return []

    # Define columns that are not features
    non_feature_cols = {
        'Open', 'High', 'Low', 'Close', 'Volume', 'Adj Close',
        'Date', 'Timestamp', 'DateTime', 'Time',
        'target', 'Target', 'label', 'Label',
        'symbol', 'Symbol', 'ticker', 'Ticker',
        'regime', 'Regime', 'cycle', 'Cycle'
    }

    # Get all columns that are not in the non-feature set
    feature_cols = [col for col in df.columns if col not in non_feature_cols]

    # Additional filtering for numeric columns only
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    feature_cols = [col for col in feature_cols if col in numeric_cols]

    return feature_cols


def get_walk_forward_periods(start_date: pd.Timestamp, end_date: pd.Timestamp,
                           train_window: str, retrain_freq: str, gap: str) -> Tuple[List, List, List, List]:
    """
    Generates lists of start/end dates for training and testing periods for walk-forward validation.

    Args:
        start_date: Start date for the walk-forward analysis
        end_date: End date for the walk-forward analysis
        train_window: Training window size (e.g., '180D')
        retrain_freq: Retraining frequency (e.g., '30D')
        gap: Gap between training and testing periods (e.g., '1D')

    Returns:
        Tuple of (train_start_dates, train_end_dates, test_start_dates, test_end_dates)
    """
    train_start_dates, train_end_dates, test_start_dates, test_end_dates = [], [], [], []
    current_date = start_date
    train_offset = pd.tseries.frequencies.to_offset(train_window)
    retrain_offset = pd.tseries.frequencies.to_offset(retrain_freq)
    gap_offset = pd.tseries.frequencies.to_offset(gap)

    while True:
        train_start = current_date
        train_end = train_start + train_offset
        test_start = train_end + gap_offset
        test_end = test_start + retrain_offset

        if test_end > end_date:
            # Adjust the last test period to not go beyond the data end date
            test_end = end_date
            if test_start >= test_end:  # Break if the last test period is invalid
                 break

        train_start_dates.append(train_start)
        train_end_dates.append(train_end)
        test_start_dates.append(test_start)
        test_end_dates.append(test_end)

        current_date += retrain_offset
        if current_date + train_offset > end_date:
            break

    return train_start_dates, train_end_dates, test_start_dates, test_end_dates


def train_and_diagnose_regime(df: pd.DataFrame, results_dir: str, n_regimes: int = 5) -> Dict:
    """
    Trains a K-Means clustering model to identify market regimes or loads a pre-existing one.
    Diagnoses the current regime and returns a summary of all regime characteristics.

    Args:
        df: DataFrame with market data and features
        results_dir: Directory to save/load regime models
        n_regimes: Number of market regimes to identify

    Returns:
        Dictionary containing regime analysis results
    """
    try:
        from sklearn.cluster import KMeans
        from sklearn.preprocessing import StandardScaler
        import joblib
    except ImportError:
        logger.error("scikit-learn not available for regime analysis")
        return {"current_diagnosed_regime": "Unknown", "regime_characteristics": {}}

    logger.info("-> Performing data-driven market regime analysis...")
    regime_model_path = os.path.join(results_dir, 'regime_model.pkl')
    regime_scaler_path = os.path.join(results_dir, 'regime_scaler.pkl')

    # Features that define a market's "personality"
    regime_features = ['ATR', 'ADX', 'hurst_exponent', 'realized_volatility', 'bollinger_bandwidth']
    regime_features = [f for f in regime_features if f in df.columns]  # Ensure all features exist

    if not regime_features:
        logger.warning("No regime features available in dataframe")
        return {"current_diagnosed_regime": "Unknown", "regime_characteristics": {}}

    df_regime = df[regime_features].dropna()

    if os.path.exists(regime_model_path) and os.path.exists(regime_scaler_path):
        logger.info("  - Loading existing regime model and scaler.")
        model = joblib.load(regime_model_path)
        scaler = joblib.load(regime_scaler_path)
    else:
        logger.warning("  - No regime model found. Training and saving a new one.")
        scaler = StandardScaler()
        df_regime_scaled = scaler.fit_transform(df_regime)

        model = KMeans(n_clusters=n_regimes, random_state=42, n_init=10)
        model.fit(df_regime_scaled)

        joblib.dump(model, regime_model_path)
        joblib.dump(scaler, regime_scaler_path)
        logger.info(f"  - New regime model saved to {regime_model_path}")

    # Diagnose the most recent data point
    last_valid_data = df[regime_features].dropna().iloc[-1:]
    if last_valid_data.empty:
        logger.warning("No valid data for regime diagnosis")
        return {"current_diagnosed_regime": "Unknown", "regime_characteristics": {}}

    last_data_scaled = scaler.transform(last_valid_data)
    current_regime_id = model.predict(last_data_scaled)[0]

    # Create a summary for the AI
    centers_unscaled = scaler.inverse_transform(model.cluster_centers_)
    regime_summary = {
        "current_diagnosed_regime": f"Regime_{current_regime_id}",
        "regime_characteristics": {
            f"Regime_{i}": {feat: round(val, 4) for feat, val in zip(regime_features, center)}
            for i, center in enumerate(centers_unscaled)
        }
    }
    logger.info(f"  - Current market condition diagnosed as: Regime_{current_regime_id}")
    return regime_summary


def apply_genetic_rules_to_df(full_df: pd.DataFrame, rules: Tuple[str, str], config) -> pd.DataFrame:
    """
    Applies the evolved genetic rules to the entire dataframe to generate
    a 'primary_model_signal' column for the meta-labeler.

    Args:
        full_df: DataFrame with market data and features
        rules: Tuple of (long_rule, short_rule) strings
        config: Configuration object

    Returns:
        DataFrame with added 'primary_model_signal' column
    """
    logger.info("-> Applying evolved genetic rules to the full dataset...")
    df_with_signals = full_df.copy()
    long_rule, short_rule = rules

    try:
        # Import here to avoid circular imports
        from .genetic_programmer import GeneticProgrammer

        # We pass an empty dict for the gene pool as it's not needed for parsing.
        gp_parser = GeneticProgrammer({}, config)

        all_signals = []
        # Process symbol by symbol to ensure data integrity
        for symbol, group in df_with_signals.groupby('Symbol'):
            logger.info(f"  - Applying rules for symbol: {symbol}")
            symbol_group = group.copy()

            long_signals = gp_parser._parse_and_eval_rule(long_rule, symbol_group)
            short_signals = gp_parser._parse_and_eval_rule(short_rule, symbol_group)

            signals = pd.Series(0, index=symbol_group.index)
            signals[long_signals] = 1
            signals[short_signals] = -1

            symbol_group['primary_model_signal'] = signals
            all_signals.append(symbol_group)

        final_df = pd.concat(all_signals).sort_index()
        logger.info("[SUCCESS] Evolved rules applied. 'primary_model_signal' column created.")
        return final_df

    except ImportError:
        logger.error("GeneticProgrammer not available for rule application")
        # Return dataframe with zero signals as fallback
        df_with_signals['primary_model_signal'] = 0
        return df_with_signals


def json_serializer_default(o):
    """
    Custom JSON serializer for complex Python types.

    Handles serialization of Enums, Paths, and datetime objects that are not
    natively JSON serializable.

    Args:
        o: Object to serialize

    Returns:
        JSON-serializable representation of the object

    Raises:
        TypeError: If object type is not supported
    """
    from enum import Enum
    import pathlib
    from datetime import datetime, date

    if isinstance(o, Enum):
        return o.value
    if isinstance(o, (pathlib.Path, datetime, date)):
        return str(o)

    raise TypeError(f"Object of type {o.__class__.__name__} is not JSON serializable")


def _sanitize_keys_for_json(obj: Any) -> Any:
    """
    Recursively sanitise dictionary keys for JSON serialisation.

    Converts non-string keys (like Enums) to strings to ensure the entire
    structure is JSON-serialisable.

    Args:
        obj: Object to sanitise (dict, list, or primitive)

    Returns:
        Sanitised object with string keys
    """
    from enum import Enum

    if isinstance(obj, dict):
        return {str(k.value if isinstance(k, Enum) else k): _sanitize_keys_for_json(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [_sanitize_keys_for_json(elem) for elem in obj]
    else:
        return obj


def get_optimal_system_settings() -> Dict[str, int]:
    """
    Analyze system resources to determine optimal processing settings.

    Implements conservative logic to prevent out-of-memory errors by considering
    both CPU cores and available RAM when determining parallel processing limits.

    Returns:
        Dictionary containing optimal settings for parallel processing
    """
    import os
    import psutil

    settings = {}
    try:
        cpu_count = os.cpu_count()
        virtual_mem = psutil.virtual_memory()
        available_gb = virtual_mem.available / (1024**3)

        logger.info("System Resource Analysis")
        logger.info(f"  Total CPU Cores: {cpu_count}")
        logger.info(f"  Available RAM: {available_gb:.2f} GB")

        # Determine optimal worker count based on system resources
        if cpu_count is None:
            num_workers = 1
            logger.warning("Could not determine CPU count. Defaulting to 1 worker.")
        elif cpu_count <= 4:
            # Conservative approach for systems with few cores
            num_workers = max(1, cpu_count - 1)
            logger.info(f"  Low CPU count detected. Setting to {num_workers} worker(s).")
        elif available_gb < 8:
            # Limit workers on memory-constrained systems
            num_workers = max(1, cpu_count // 2)
            logger.warning(f"  Low memory detected (<8GB). Limiting to {num_workers} worker(s) to prevent OOM.")
        else:
            # Balanced approach: reserve cores for OS and main process
            num_workers = max(1, cpu_count - 2)
            logger.info(f"  System capable. Setting to {num_workers} worker(s).")

        settings['num_workers'] = num_workers

    except Exception as e:
        logger.error(f"Could not determine optimal system settings: {e}. Defaulting to 1 worker.")
        settings['num_workers'] = 1

    return settings


def _parallel_process_symbol_wrapper(symbol_tuple, feature_engineer_instance):
    """
    Wrapper function for parallel processing of individual symbols.

    Enables multiprocessing by providing a top-level function that can be pickled
    and distributed across worker processes.

    Args:
        symbol_tuple: Tuple containing (symbol, symbol_data_by_tf)
        feature_engineer_instance: FeatureEngineer instance to use for processing

    Returns:
        Processed feature data for the symbol
    """
    symbol, symbol_data_by_tf = symbol_tuple
    logger.info(f"  Starting parallel processing for symbol: {symbol}...")
    return feature_engineer_instance._process_single_symbol_stack(symbol_data_by_tf)


def get_and_cache_asset_types(symbols: List[str], config_dict: Dict, gemini_analyzer) -> Dict[str, str]:
    """Classifies assets using the AI and caches the results to avoid repeated API calls."""
    cache_path = os.path.join(config_dict.get("BASE_PATH", "."), "Results", "asset_type_cache.json")

    if os.path.exists(cache_path):
        try:
            with open(cache_path, 'r') as f:
                cached_data = json.load(f)
            # Check if the cached symbols match the current symbols
            if set(cached_data.keys()) == set(symbols):
                logger.info("Asset types loaded from cache.")
                return cached_data
        except (IOError, json.JSONDecodeError) as e:
            logger.warning(f"Could not read asset type cache, will re-fetch. Error: {e}")

    logger.info("Fetching asset classifications from AI...")
    classified_assets = gemini_analyzer.classify_asset_symbols(symbols)

    if classified_assets:
        try:
            # Ensure Results directory exists
            os.makedirs(os.path.dirname(cache_path), exist_ok=True)
            with open(cache_path, 'w') as f:
                json.dump(classified_assets, f, indent=4)
            logger.info(f"Asset classifications saved to cache at {cache_path}")
        except IOError as e:
            logger.error(f"Could not save asset type cache: {e}")
        return classified_assets

    logger.error("Failed to classify assets using AI, returning empty dictionary.")
    return {}


def get_and_cache_contract_sizes(symbols: List[str], config, gemini_analyzer, api_timer) -> Dict[str, float]:
    """
    Gets contract sizes for symbols using AI analysis with caching.
    """
    cache_file = os.path.join(config.BASE_PATH, 'contract_sizes_cache.json')

    # Load existing cache
    cached_sizes = {}
    if os.path.exists(cache_file):
        try:
            with open(cache_file, 'r') as f:
                cached_sizes = json.load(f)
        except Exception as e:
            logger.warning(f"Could not load contract sizes cache: {e}")

    # Determine which symbols need analysis
    uncached_symbols = [s for s in symbols if s not in cached_sizes]

    if uncached_symbols:
        logger.info(f"Determining contract sizes for {len(uncached_symbols)} symbols using AI...")
        api_timer.wait_if_needed()
        new_sizes = gemini_analyzer.get_contract_sizes_for_assets(uncached_symbols)

        # Update cache with new sizes
        cached_sizes.update(new_sizes)

        # Save updated cache
        try:
            with open(cache_file, 'w') as f:
                json.dump(cached_sizes, f, indent=2)
        except Exception as e:
            logger.warning(f"Could not save contract sizes cache: {e}")

    # Return only the requested symbols
    return {symbol: cached_sizes.get(symbol, 1.0) for symbol in symbols}
