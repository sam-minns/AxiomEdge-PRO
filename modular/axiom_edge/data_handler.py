# =============================================================================
# DATA HANDLER & COLLECTION MODULE
# =============================================================================

import os
import pickle
import requests
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
from collections import defaultdict
import logging
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)

class DataHandler:
    """
    Handles data collection, caching, and retrieval from various sources.
    Can be used independently for data collection tasks.
    """
    
    def __init__(self, cache_dir: str = "data_cache", api_key: Optional[str] = None):
        self.cache_path = Path(cache_dir)
        self.cache_path.mkdir(exist_ok=True)
        self.cache: Dict[str, pd.DataFrame] = {}
        self.api_key = api_key or os.getenv("FINANCIAL_API_KEY")
        
    def get_data(self, symbol: str, start_date: str, end_date: str, 
                 timeframe: str = "1D", source: str = "alpha_vantage") -> pd.DataFrame:
        """
        Retrieves asset data, checking caches before fetching from the API.
        The cache hierarchy is: In-Memory -> On-Disk -> API.
        """
        # Sanitize filename
        safe_symbol = symbol.replace('/', '_').replace('=', '')
        cache_key = f"{safe_symbol}_{timeframe}_{start_date}_{end_date}"
        pickle_file_path = self.cache_path / f"{cache_key}.pkl"

        # 1. Check in-memory cache first
        if cache_key in self.cache:
            logger.info(f"Loading {symbol} from in-memory cache.")
            return self.cache[cache_key].copy()

        # 2. Check on-disk cache
        if pickle_file_path.exists():
            try:
                logger.info(f"Loading {symbol} from disk cache.")
                df = pd.read_pickle(pickle_file_path)
                self.cache[cache_key] = df
                return df.copy()
            except Exception as e:
                logger.warning(f"Failed to load from disk cache: {e}")

        # 3. Fetch from API
        logger.info(f"Fetching {symbol} from {source} API...")
        df = self._fetch_from_api(symbol, start_date, end_date, timeframe, source)
        
        if not df.empty:
            # Cache the data
            self.cache[cache_key] = df
            try:
                df.to_pickle(pickle_file_path)
                logger.info(f"Cached {symbol} data to disk.")
            except Exception as e:
                logger.warning(f"Failed to cache to disk: {e}")
        
        return df

    def _fetch_from_api(self, symbol: str, start_date: str, end_date: str, 
                       timeframe: str, source: str) -> pd.DataFrame:
        """Fetch data from the specified API source"""
        try:
            if source == "alpha_vantage":
                return self._fetch_alpha_vantage(symbol, timeframe)
            elif source == "yahoo":
                return self._fetch_yahoo_finance(symbol, start_date, end_date)
            elif source == "polygon":
                return self._fetch_polygon(symbol, start_date, end_date, timeframe)
            else:
                logger.error(f"Unsupported data source: {source}")
                return pd.DataFrame()
        except Exception as e:
            logger.error(f"Error fetching data for {symbol}: {e}")
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
        """Fetch data from Yahoo Finance (requires yfinance package)"""
        try:
            import yfinance as yf
            ticker = yf.Ticker(symbol)
            df = ticker.history(start=start_date, end=end_date)
            return df
        except ImportError:
            logger.error("yfinance package not installed. Install with: pip install yfinance")
            return pd.DataFrame()
        except Exception as e:
            logger.error(f"Yahoo Finance error: {e}")
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

        # Concatenate data for each timeframe
        consolidated_data = {}
        symbols = []
        for tf, dfs in data_by_tf.items():
            if dfs:
                consolidated_df = pd.concat(dfs, ignore_index=True)
                consolidated_df = consolidated_df.sort_values('timestamp').reset_index(drop=True)
                consolidated_data[tf] = consolidated_df
                
                # Extract unique symbols
                if 'symbol' in consolidated_df.columns:
                    symbols.extend(consolidated_df['symbol'].unique().tolist())

        symbols = list(set(symbols))  # Remove duplicates
        logger.info(f"  - Consolidated data for {len(consolidated_data)} timeframes")
        logger.info(f"  - Found {len(symbols)} unique symbols: {symbols}")
        
        return consolidated_data, symbols

    def _parse_single_file(self, file_path: str, filename: str) -> Optional[pd.DataFrame]:
        """Parse a single data file"""
        try:
            # Determine file format and parse accordingly
            if filename.endswith('.csv'):
                df = pd.read_csv(file_path)
            elif filename.endswith('.parquet'):
                df = pd.read_parquet(file_path)
            elif filename.endswith('.json'):
                df = pd.read_json(file_path)
            else:
                logger.warning(f"Unsupported file format: {filename}")
                return None
                
            # Standardize column names and data types
            df = self._standardize_dataframe(df, filename)
            return df
            
        except Exception as e:
            logger.error(f"Error parsing {filename}: {e}")
            return None

    def _standardize_dataframe(self, df: pd.DataFrame, filename: str) -> pd.DataFrame:
        """Standardize dataframe format and column names"""
        # Extract symbol from filename if not present
        if 'symbol' not in df.columns:
            symbol = filename.split('_')[0]
            df['symbol'] = symbol

        # Ensure timestamp column exists and is properly formatted
        if 'timestamp' not in df.columns:
            if 'date' in df.columns:
                df['timestamp'] = df['date']
            elif 'time' in df.columns:
                df['timestamp'] = df['time']
            elif df.index.name in ['date', 'time', 'timestamp']:
                df = df.reset_index()
                df['timestamp'] = df[df.index.name]

        # Convert timestamp to datetime
        if 'timestamp' in df.columns:
            df['timestamp'] = pd.to_datetime(df['timestamp'])

        # Standardize OHLCV column names
        column_mapping = {
            'open': 'Open', 'high': 'High', 'low': 'Low', 'close': 'Close', 'volume': 'Volume',
            'o': 'Open', 'h': 'High', 'l': 'Low', 'c': 'Close', 'v': 'Volume'
        }
        
        df = df.rename(columns=column_mapping)
        
        # Ensure required columns exist
        required_cols = ['Open', 'High', 'Low', 'Close']
        for col in required_cols:
            if col not in df.columns:
                logger.warning(f"Missing required column '{col}' in {filename}")

        return df
