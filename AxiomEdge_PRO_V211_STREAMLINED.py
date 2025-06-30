#  STREAMLINED
# --- SCRIPT VERSION ---
VERSION = "211"
# ---------------------

import os
import re
import json
from json import JSONDecoder, JSONDecodeError
import time
import warnings
import logging
import sys
import random
from datetime import datetime, date, timedelta
from logging.handlers import RotatingFileHandler
from typing import List, Dict, Any, Optional, Tuple, Union, Callable, ClassVar
from collections import defaultdict
import pathlib
from enum import Enum
import hashlib
import psutil
import inspect
import tlars

# --- LOAD ENVIRONMENT VARIABLES ---
from dotenv import load_dotenv
load_dotenv()
# --- END ---

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import shap
import xgboost as xgb
import optuna
import requests
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.metrics import f1_score, classification_report, mean_squared_error
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import RobustScaler, MinMaxScaler, StandardScaler
from sklearn.utils.class_weight import compute_class_weight
from pydantic import BaseModel, DirectoryPath, confloat, conint, Field, ValidationError
from sklearn.ensemble import IsolationForest
from sklearn.decomposition import PCA, IncrementalPCA
from sklearn.feature_selection import mutual_info_classif
import yfinance as yf
from hurst import compute_Hc
from trexselector import trex
from pykalman import KalmanFilter
import networkx as nx

# --- PHASE 1 IMPORTS ---
from sklearn.cluster import KMeans
import joblib
# -----------------------

import scipy
from scipy.stats import entropy, skew, kurtosis
from scipy.signal import hilbert
from statsmodels.tsa.stattools import pacf
try:
    import pywt
    PYWT_AVAILABLE = True
except ImportError:
    PYWT_AVAILABLE = False
    print("WARNING: PyWavelets is not installed. Wavelet features will be skipped. Install with: pip install PyWavelets")

try:
    from arch import arch_model
    ARCH_AVAILABLE = True
except ImportError:
    ARCH_AVAILABLE = False
    print("WARNING: arch is not installed. GARCH features will be skipped. Install with: pip install arch")

try:
    from hurst import compute_Hc
    HURST_AVAILABLE = True
except ImportError:
    HURST_AVAILABLE = False
    print("WARNING: hurst is not installed. Hurst Exponent feature will be skipped. Install with: pip install hurst")

try:
    from sktime.transformations.panel.rocket import MiniRocket
    MINIROCKET_AVAILABLE = True
except ImportError:
    MINIROCKET_AVAILABLE = False
    print("WARNING: sktime is not installed. MiniRocket strategies will be unavailable. Install with: pip install sktime")
    
try:
    from pykalman import KalmanFilter
    PYKALMAN_AVAILABLE = True
except ImportError:
    PYKALMAN_AVAILABLE = False
    print("WARNING: pykalman is not installed. Kalman Filter features will be skipped. Install with: pip install pykalman")

try:
    import tlars
    TLARS_AVAILABLE = True
except ImportError:
    TLARS_AVAILABLE = False
    print("WARNING: tlars is not installed. Certain features will be unavailable.")


# --- DIAGNOSTICS & LOGGING SETUP ---
logger = logging.getLogger("ML_Trading_Framework")

# --- GNN Specific Imports (requires PyTorch, PyG) ---
try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from torch_geometric.data import Data
    from torch_geometric.nn import GCNConv
    from torch.optim import Adam
    GNN_AVAILABLE = True
except ImportError:
    GNN_AVAILABLE = False
    class _dummy_module_container:
        Module = object
        def __init__(self):
            self.Module = object
    torch = _dummy_module_container()
    torch.nn = _dummy_module_container()
    nn = _dummy_module_container()
    F = None
    Data = None
    GCNConv = None
    Adam = None

# --- LOGGING SWITCHES ---
LOG_ANOMALY_SKIPS = False
LOG_PARTIAL_PROFITS = True
# -----------------------------

def flush_loggers():
    """Flushes all handlers for all active loggers to disk."""
    for handler in logging.getLogger().handlers:
        handler.flush()
    for handler in logging.getLogger("ML_Trading_Framework").handlers:
        handler.flush()

def _setup_logging(log_file_path: str, report_label: str):
    """Configures the global logger for the framework."""
    global logger

    # Clear any existing handlers to prevent duplicate logs if this is called again
    if logger.hasHandlers():
        logger.handlers.clear()

    logger.setLevel(logging.DEBUG) # Capture all levels of logs

    # Create a handler for console output
    ch = logging.StreamHandler(sys.stdout)
    ch.setLevel(logging.INFO) # Console shows INFO and above

    # Create a handler for file output
    # Ensure the directory for the log file exists
    os.makedirs(os.path.dirname(log_file_path), exist_ok=True)
    fh = RotatingFileHandler(log_file_path, maxBytes=5*1024*1024, backupCount=3, encoding='utf-8')
    fh.setLevel(logging.DEBUG) # File logs everything (DEBUG and above)

    # Define the log format
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(filename)s:%(lineno)d - %(message)s')
    ch.setFormatter(formatter)
    fh.setFormatter(formatter)

    # Add handlers to the logger
    logger.addHandler(ch)
    logger.addHandler(fh)

    logger.info(f"--- Logging initialized for: {report_label} ---")
    logger.info(f"--- Log file: {log_file_path} ---")
    
    # Also configure Optuna's logger to be less verbose
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    
    # Filter out specific warnings if needed
    warnings.filterwarnings('ignore', category=FutureWarning)
    warnings.filterwarnings('ignore', category=UserWarning, module='sklearn.metrics._classification')
    warnings.filterwarnings('ignore', category=pd.errors.PerformanceWarning)

class DataHandler:
    """
    Handles fetching and caching of market data with a two-tier system:
    1. In-memory cache (self.cache) for session-speed access.
    2. On-disk cache (self.cache_path) for persistence between sessions.
    """
    def __init__(self, cache_dir: str = "./data_cache"):
        self.cache = {}
        self.cache_path = pathlib.Path(cache_dir)
        self.cache_path.mkdir(parents=True, exist_ok=True)
        print(f"Data cache initialized at: {self.cache_path.resolve()}")

    def _fetch_from_api(self, symbol: str, start_date: str, end_date: str) -> pd.DataFrame:
        """Private method to fetch data using the yfinance API."""
        print(f"Fetching {symbol} data from {start_date} to {end_date} via API...")
        try:
            # Using yfinance as a concrete example for fetching data
            data = yf.download(symbol, start=start_date, end=end_date, progress=False, auto_adjust=True)
            if data.empty:
                print(f"Warning: No data returned for {symbol} for the given date range.")
                return pd.DataFrame()
            # Standardize column names to lowercase
            data.rename(columns={'Open': 'open', 'High': 'high', 'Low': 'low', 'Close': 'close', 'Volume': 'volume'}, inplace=True)
            return data[['open', 'high', 'low', 'close', 'volume']]
        except Exception as e:
            print(f"Error fetching data for {symbol}: {e}")
            return pd.DataFrame()

    def get_data(self, symbol: str, start_date: str, end_date: str) -> pd.DataFrame:
        """
        Retrieves asset data, checking caches before fetching from the API.
        The cache hierarchy is: In-Memory -> On-Disk -> API.
        """
        # Sanitize filename
        safe_symbol = symbol.replace('/', '_').replace('=', '')
        cache_key = f"{safe_symbol}_{start_date}_{end_date}"
        pickle_file_path = self.cache_path / f"{cache_key}.pkl"

        # 1. Check in-memory cache first
        if cache_key in self.cache:
            print(f"Loading {symbol} from in-memory cache.")
            return self.cache[cache_key].copy()

        # 2. Check on-disk cache
        if pickle_file_path.exists():
            print(f"Loading {symbol} from on-disk cache: {pickle_file_path}")
            # Use a dummy import for pickle as it's not explicitly used but implied by .pkl files
            import pickle
            with open(pickle_file_path, 'rb') as f:
                df = pickle.load(f)
            self.cache[cache_key] = df  # Add to in-memory cache for this session
            return df.copy()

        # 3. Fetch from API as a last resort
        df = self._fetch_from_api(symbol, start_date, end_date)
        if not df.empty:
            print(f"Saving {symbol} data to both caches.")
            self.cache[cache_key] = df
            import pickle
            with open(pickle_file_path, 'wb') as f:
                pickle.dump(df, f)
        return df.copy()

def calculate_features_and_stops(df: pd.DataFrame, atr_period: int = 14, atr_multiplier: float = 2.0) -> pd.DataFrame:
    
    if df.empty:
        return df

    high_low = df['High'] - df['Low']
    high_close = np.abs(df['High'] - df['Close'].shift())
    low_close = np.abs(df['Low'] - df['Close'].shift())

    # Calculate True Range (TR)
    tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1, skipna=False)
    
    # Calculate Average True Range (ATR) using an Exponential Moving Average
    # Note: Using pandas ewm is a standard and robust way to calculate ATR
    df['ATR'] = tr.ewm(alpha=1/atr_period, adjust=False, min_periods=1).mean()

    # Calculate and add the stop-loss levels
    df['stop_loss_long'] = df['Close'] - (df['ATR'] * atr_multiplier)
    df['stop_loss_short'] = df['Close'] + (df['ATR'] * atr_multiplier)
    
    print(f"Successfully calculated ATR({atr_period}) and stop-loss levels.")
    return df

def json_serializer_default(o):
    """
    A custom serializer for json.dump to handle complex types like Paths, datetimes, and Enums.
    """
    if isinstance(o, Enum):
        return o.value  # Convert Enum members to their string value
    if isinstance(o, (pathlib.Path, datetime, date)):
        return str(o)   # Convert Path and datetime objects to a simple string
        
    # Let the base class default method raise the TypeError for other types
    raise TypeError(f"Object of type {o.__class__.__name__} is not JSON serializable")
    
def _sanitize_keys_for_json(obj: Any) -> Any:
    """
    Recursively traverses a dict or list to convert any non-string keys
    (like Enums) into strings, making the structure JSON-serializable.
    """
    if isinstance(obj, dict):
        return {str(k.value if isinstance(k, Enum) else k): _sanitize_keys_for_json(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [_sanitize_keys_for_json(elem) for elem in obj]
    else:
        return obj    

# =============================================================================
# 3. CONFIGURATION & VALIDATION
# =============================================================================

# --- FRAMEWORK STATE DEFINITION ---
class OperatingState(Enum):
    """Defines the operational states of the trading framework."""
    CONSERVATIVE_BASELINE = "Conservative Baseline"
    AGGRESSIVE_EXPANSION = "Aggressive Expansion"
    DRAWDOWN_CONTROL = "Drawdown Control"
    OPPORTUNISTIC_SURGE = "Opportunistic Surge"
    MAINTENANCE_DORMANCY = "Maintenance Dormancy"
# ------------------------------------

class EarlyInterventionConfig(BaseModel):
    """Configuration for the adaptive early intervention system."""
    enabled: bool = True
    attempt_threshold: conint(ge=2) = 2
    min_profitability_for_f1_override: confloat(ge=0) = 3.0
    max_f1_override_value: confloat(ge=0.4, le=0.6) = 0.50

class ConfigModel(BaseModel):

    # --- Core Run, Capital & State Parameters ---
    BASE_PATH: DirectoryPath
    REPORT_LABEL: str
    INITIAL_CAPITAL: confloat(gt=0)
    operating_state: OperatingState = OperatingState.CONSERVATIVE_BASELINE
    
    # UPDATED: Added 'pca' as a feature selection option
    FEATURE_SELECTION_METHOD: str = 'pca' # Options: 'trex', 'mutual_info', 'pca'

    # --- AI & Optimization Parameters ---
    OPTUNA_TRIALS: conint(gt=0) = 75
    MAX_TRAINING_RETRIES_PER_CYCLE: conint(ge=0) = 3
    CALCULATE_SHAP_VALUES: bool = True
    
    # --- Confidence Gate Control ---
    USE_STATIC_CONFIDENCE_GATE: bool = False
    STATIC_CONFIDENCE_GATE: confloat(ge=0.5, le=0.95) = 0.70

    # --- Dynamic Labeling & Trade Definition ---
    TP_ATR_MULTIPLIER: confloat(gt=0.5, le=10.0) = 2.0
    SL_ATR_MULTIPLIER: confloat(gt=0.5, le=10.0) = 1.5
    LOOKAHEAD_CANDLES: conint(gt=0) = 150
    LABELING_METHOD: str = 'signal_pressure'
    MIN_F1_SCORE_GATE: confloat(ge=0.3, le=0.7) = 0.45
    LABEL_LONG_QUANTILE: confloat(ge=0.5, le=1.0) = 0.95
    LABEL_SHORT_QUANTILE: confloat(ge=0.0, le=0.5) = 0.05

    # --- Walk-Forward & Data Parameters ---
    TRAINING_WINDOW: str = '365D'
    RETRAINING_FREQUENCY: str = '90D'
    FORWARD_TEST_GAP: str = '1D'
    
    # --- Risk & Portfolio Management ---
    MAX_DD_PER_CYCLE: confloat(ge=0.05, lt=1.0) = 0.25
    RISK_CAP_PER_TRADE_USD: confloat(gt=0) = 1000.0
    BASE_RISK_PER_TRADE_PCT: confloat(gt=0, lt=1) = 0.01
    MAX_CONCURRENT_TRADES: conint(ge=1, le=20) = 3
    USE_TIERED_RISK: bool = False
    RISK_PROFILE: str = 'Medium'
    TIERED_RISK_CONFIG: Dict[int, Dict[str, Dict[str, Union[float, int]]]] = Field(default_factory=lambda: {
            2000:  {'Low': {'risk_pct': 0.01,  'pairs': 1}, 'Medium': {'risk_pct': 0.01,  'pairs': 1}, 'High': {'risk_pct': 0.01,  'pairs': 1}},
            5000:  {'Low': {'risk_pct': 0.008, 'pairs': 1}, 'Medium': {'risk_pct': 0.012, 'pairs': 1}, 'High': {'risk_pct': 0.012, 'pairs': 2}},
            10000: {'Low': {'risk_pct': 0.006, 'pairs': 2}, 'Medium': {'risk_pct': 0.008, 'pairs': 2}, 'High': {'risk_pct': 0.01,  'pairs': 2}},
            15000: {'Low': {'risk_pct': 0.007, 'pairs': 2}, 'Medium': {'risk_pct': 0.009, 'pairs': 2}, 'High': {'risk_pct': 0.012, 'pairs': 2}},
            25000: {'Low': {'risk_pct': 0.008, 'pairs': 2}, 'Medium': {'risk_pct': 0.012, 'pairs': 2}, 'High': {'risk_pct': 0.016, 'pairs': 2}},
            50000: {'Low': {'risk_pct': 0.008, 'pairs': 3}, 'Medium': {'risk_pct': 0.012, 'pairs': 3}, 'High': {'risk_pct': 0.016, 'pairs': 3}},
            100000:{'Low': {'risk_pct': 0.007, 'pairs': 4}, 'Medium': {'risk_pct': 0.01,  'pairs': 4}, 'High': {'risk_pct': 0.014, 'pairs': 4}},
            9000000000: {'Low': {'risk_pct': 0.005, 'pairs': 6}, 'Medium': {'risk_pct': 0.0075,'pairs': 6}, 'High': {'risk_pct': 0.01,  'pairs': 6}}
        })

    STATE_BASED_CONFIG: Dict[OperatingState, Dict[str, Any]] = Field(default_factory=lambda: {
        OperatingState.CONSERVATIVE_BASELINE: {
            "max_dd_per_cycle": 0.15, "base_risk_pct": 0.0075, "max_concurrent_trades": 2,
            "optimization_objective": ["maximize_f1", "maximize_log_trades"],
            "min_f1_gate": 0.40
        },
        OperatingState.AGGRESSIVE_EXPANSION: {
            "max_dd_per_cycle": 0.30, "base_risk_pct": 0.015, "max_concurrent_trades": 5,
            "optimization_objective": ["maximize_pnl", "maximize_log_trades"],
            "min_f1_gate": 0.42
        },
        OperatingState.DRAWDOWN_CONTROL: {
            "max_dd_per_cycle": 0.10, "base_risk_pct": 0.005, "max_concurrent_trades": 1,
            "optimization_objective": ["maximize_sortino", "minimize_trades"],
            "min_f1_gate": 0.38
        },
        OperatingState.OPPORTUNISTIC_SURGE: {
            "max_dd_per_cycle": 0.20, "base_risk_pct": 0.0125, "max_concurrent_trades": 3,
            "optimization_objective": ["maximize_pnl", "minimize_trades"],
            "min_f1_gate": 0.40
        },
        OperatingState.MAINTENANCE_DORMANCY: {
            "max_dd_per_cycle": 0.05, "base_risk_pct": 0.0, "max_concurrent_trades": 0,
            "optimization_objective": ["maximize_f1", "minimize_trades"],
            "min_f1_gate": 0.99
        }
    })
    CONFIDENCE_TIERS: Dict[str, Dict[str, Any]] = Field(default_factory=lambda: {
            'ultra_high': {'min': 0.80, 'risk_mult': 1.2, 'rr': 2.5},
            'high':       {'min': 0.70, 'risk_mult': 1.0, 'rr': 2.0},
            'standard':   {'min': 0.60, 'risk_mult': 0.8, 'rr': 1.5}
        })
    USE_TP_LADDER: bool = True
    TP_LADDER_LEVELS_PCT: List[confloat(gt=0, lt=1)] = Field(default_factory=lambda: [0.25, 0.25, 0.25, 0.25])
    TP_LADDER_RISK_MULTIPLIERS: List[confloat(gt=0)] = Field(default_factory=lambda: [1.0, 2.0, 3.0, 4.0])

    # --- Broker & Execution Simulation ---
    COMMISSION_PER_LOT: confloat(ge=0.0) = 3.5
    USE_REALISTIC_EXECUTION: bool = True
    SIMULATE_LATENCY: bool = True
    EXECUTION_LATENCY_MS: conint(ge=50, le=500) = 150
    USE_VARIABLE_SLIPPAGE: bool = True
    SLIPPAGE_VOLATILITY_FACTOR: confloat(ge=0.0, le=5.0) = 1.5
    SPREAD_CONFIG: Dict[str, Dict[str, float]] = Field(default_factory=lambda: {
        'default': {'normal_pips': 1.8, 'volatile_pips': 5.5},
        'EURUSD':  {'normal_pips': 1.2, 'volatile_pips': 4.0},
        'GBPUSD':  {'normal_pips': 1.6, 'volatile_pips': 5.0},
        'AUDUSD':  {'normal_pips': 1.4, 'volatile_pips': 4.8},
        'USDCAD':  {'normal_pips': 1.7, 'volatile_pips': 5.5},
        'USDJPY':  {'normal_pips': 1.3, 'volatile_pips': 4.5},
        'AUDCAD':  {'normal_pips': 1.9, 'volatile_pips': 6.0},
        'AUDNZD':  {'normal_pips': 2.2, 'volatile_pips': 7.0},
        'NZDJPY':  {'normal_pips': 2.0, 'volatile_pips': 6.5},
        'XAUUSD_M15':    {'normal_pips': 25.0, 'volatile_pips': 80.0},
        'XAUUSD_H1':     {'normal_pips': 20.0, 'volatile_pips': 70.0},
        'XAUUSD_Daily':  {'normal_pips': 18.0, 'volatile_pips': 60.0},
        'US30_M15':      {'normal_pips': 50.0, 'volatile_pips': 150.0},
        'US30_H1':       {'normal_pips': 45.0, 'volatile_pips': 140.0},
        'US30_Daily':    {'normal_pips': 40.0, 'volatile_pips': 130.0},
        'NDX100_M15':    {'normal_pips': 20.0, 'volatile_pips': 60.0},
        'NDX100_H1':     {'normal_pips': 18.0, 'volatile_pips': 55.0},
        'NDX100_Daily':  {'normal_pips': 16.0, 'volatile_pips': 50.0},
    })
    CONTRACT_SIZE: confloat(gt=0) = 100000.0
    LEVERAGE: conint(gt=0) = 30
    MIN_LOT_SIZE: confloat(gt=0) = 0.01
    LOT_STEP: confloat(gt=0) = 0.01
    
    DYNAMIC_INDICATOR_PARAMS: Dict[str, Dict[str, Any]] = Field(default_factory=lambda: {
        "HighVolatility_Trending":  { "bollinger_period": 15, "bollinger_std_dev": 2.5, "rsi_period": 10 },
        "HighVolatility_Ranging":   { "bollinger_period": 20, "bollinger_std_dev": 2.8, "rsi_period": 12 },
        "HighVolatility_Default":   { "bollinger_period": 18, "bollinger_std_dev": 2.5, "rsi_period": 11 },
        "LowVolatility_Trending":   { "bollinger_period": 30, "bollinger_std_dev": 1.8, "rsi_period": 20 },
        "LowVolatility_Ranging":    { "bollinger_period": 35, "bollinger_std_dev": 1.5, "rsi_period": 25 },
        "LowVolatility_Default":    { "bollinger_period": 30, "bollinger_std_dev": 1.8, "rsi_period": 22 },
        "Default_Trending":         { "bollinger_period": 20, "bollinger_std_dev": 2.0, "rsi_period": 14 },
        "Default_Ranging":          { "bollinger_period": 25, "bollinger_std_dev": 2.2, "rsi_period": 18 },
        "Default":                  { "bollinger_period": 20, "bollinger_std_dev": 2.0, "rsi_period": 14 }
    })
    
    # --- Feature Engineering Parameters ---
    TREND_FILTER_THRESHOLD: confloat(gt=0) = 25.0
    BOLLINGER_PERIOD: conint(gt=0) = 20
    STOCHASTIC_PERIOD: conint(gt=0) = 14
    MIN_VOLATILITY_RANK: confloat(ge=0.0, le=1.0) = 0.1
    MAX_VOLATILITY_RANK: confloat(ge=0.0, le=1.0) = 0.9
    HAWKES_KAPPA: confloat(gt=0) = 0.5
    anomaly_contamination_factor: confloat(ge=0.001, le=0.1) = 0.01
    USE_PCA_REDUCTION: bool = True
    PCA_N_COMPONENTS: conint(gt=1, le=20) = 15 # Changed from 10 to 20
    RSI_PERIODS_FOR_PCA: List[conint(gt=1)] = Field(default_factory=lambda: [5, 10, 15, 20, 25])
    ADX_THRESHOLD_TREND: int = 20
    RSI_OVERSOLD: int = 30
    RSI_OVERBOUGHT: int = 70
    VOLUME_BREAKOUT_RATIO: float = 1.5
    BOLLINGER_SQUEEZE_LOOKBACK: int = 50
    DISPLACEMENT_STRENGTH: int = 3
    DISPLACEMENT_PERIOD: conint(gt=1) = 50
    GAP_DETECTION_LOOKBACK: conint(gt=1) = 2
    PARKINSON_VOLATILITY_WINDOW: conint(gt=1) = 30
    YANG_ZHANG_VOLATILITY_WINDOW: conint(gt=1) = 30
    KAMA_REGIME_FAST: conint(gt=1) = 10
    KAMA_REGIME_SLOW: conint(gt=1) = 66
    AUTOCORR_LAG: conint(gt=0) = 10
    HURST_EXPONENT_WINDOW: conint(ge=100) = 100
    
    # --- GNN Specific Parameters ---
    GNN_EMBEDDING_DIM: conint(gt=0) = 8
    GNN_EPOCHS: conint(gt=0) = 50
    
    # --- Caching & Performance ---
    USE_FEATURE_CACHING: bool = True
    
    # --- State & Info Parameters (populated at runtime) ---
    selected_features: List[str] = Field(default_factory=list)
    run_timestamp: str
    strategy_name: str
    nickname: str = ""
    analysis_notes: str = ""

    # --- File Path Management (Internal, populated by __init__) ---
    result_folder_path: str = Field(default="", repr=False) 
    MODEL_SAVE_PATH: str = Field(default="", repr=False)
    PLOT_SAVE_PATH: str = Field(default="", repr=False)
    REPORT_SAVE_PATH: str = Field(default="", repr=False)
    SHAP_PLOT_PATH: str = Field(default="", repr=False)
    LOG_FILE_PATH: str = Field(default="", repr=False)
    CHAMPION_FILE_PATH: str = Field(default="", repr=False)
    HISTORY_FILE_PATH: str = Field(default="", repr=False)
    PLAYBOOK_FILE_PATH: str = Field(default="", repr=False)
    DIRECTIVES_FILE_PATH: str = Field(default="", repr=False)
    NICKNAME_LEDGER_PATH: str = Field(default="", repr=False)
    REGIME_CHAMPIONS_FILE_PATH: str = Field(default="", repr=False)
    CACHE_PATH: str = Field(default="", repr=False)
    CACHE_METADATA_PATH: str = Field(default="", repr=False)
    DISCOVERED_PATTERNS_PATH: str = Field(default="", repr=False) # For AI-discovered patterns

    # --- AI Guardrail Parameters (to prevent meta-overfitting) ---
    PARAMS_LOG_FILE: ClassVar[str] = 'strategy_params_log.json'
    MAX_PARAM_DRIFT_TOLERANCE: ClassVar[float] = 40.0 # Max 40% average change in params
    MIN_CYCLES_FOR_ADAPTATION: ClassVar[int] = 5    # Must have at least 5 prior runs to adapt
    MIN_HOLDOUT_SHARPE: ClassVar[float] = 0.35        # Strategy must be at least marginally profitable on unseen data
    HOLDOUT_SET_PERCENTAGE: confloat(ge=0.0, le=0.3) = 0.15 # 15% of data for guardrail validation

    def __init__(self, **data: Any):
        super().__init__(**data)
        results_dir = os.path.join(self.BASE_PATH, "Results")
        os.makedirs(results_dir, exist_ok=True)
        
        # A dummy version in case regex fails
        VERSION = "1.0" 

        version_match = re.search(r'V(\d+\.?\d*)', self.REPORT_LABEL)
        version_str = f"_V{version_match.group(1)}" if version_match else f"_V{VERSION}"

        safe_nickname = "".join(c if c.isalnum() or c in (' ', '_', '-') else '_' for c in self.nickname)
        safe_strategy_name = "".join(c if c.isalnum() or c in (' ', '_', '-') else '_' for c in self.strategy_name)

        folder_name_base = safe_nickname if safe_nickname and safe_nickname != "init_setup" else self.REPORT_LABEL.replace(version_str, "")
        folder_name = f"{folder_name_base}{version_str}"
        
        run_id_prefix = f"{folder_name}_{safe_strategy_name}" if safe_strategy_name and safe_strategy_name != "InitialSetupPhase" else folder_name
        run_id = f"{run_id_prefix}_{self.run_timestamp}"
        
        self.result_folder_path = os.path.join(results_dir, folder_name)

        if self.nickname and self.nickname != "init_setup" and self.strategy_name and self.strategy_name != "InitialSetupPhase":
            os.makedirs(self.result_folder_path, exist_ok=True)
            os.makedirs(os.path.join(self.result_folder_path, "models"), exist_ok=True)
            os.makedirs(os.path.join(self.result_folder_path, "shap_plots"), exist_ok=True)
            
            self.MODEL_SAVE_PATH = os.path.join(self.result_folder_path, "models")
            self.PLOT_SAVE_PATH = os.path.join(self.result_folder_path, f"{run_id}_equity_curve.png")
            self.REPORT_SAVE_PATH = os.path.join(self.result_folder_path, f"{run_id}_report.txt")
            self.SHAP_PLOT_PATH = os.path.join(self.result_folder_path, "shap_plots") # This is a directory
            self.LOG_FILE_PATH = os.path.join(self.result_folder_path, f"{run_id}_run.log")
        else:
            self.MODEL_SAVE_PATH = os.path.join(results_dir, "init_models")
            self.PLOT_SAVE_PATH = os.path.join(results_dir, "init_equity.png")
            self.REPORT_SAVE_PATH = os.path.join(results_dir, "init_report.txt")
            self.SHAP_PLOT_PATH = os.path.join(results_dir, "init_shap_plots")
            self.LOG_FILE_PATH = os.path.join(results_dir, f"init_run_{self.run_timestamp}.log") # Unique init log

        # Global/shared file paths
        self.CHAMPION_FILE_PATH = os.path.join(results_dir, "champion.json")
        self.HISTORY_FILE_PATH = os.path.join(results_dir, "historical_runs.jsonl")
        self.PLAYBOOK_FILE_PATH = os.path.join(results_dir, "strategy_playbook.json")
        self.DIRECTIVES_FILE_PATH = os.path.join(results_dir, "framework_directives.json")
        self.NICKNAME_LEDGER_PATH = os.path.join(results_dir, "nickname_ledger.json")
        self.REGIME_CHAMPIONS_FILE_PATH = os.path.join(results_dir, "regime_champions.json")
        self.DISCOVERED_PATTERNS_PATH = os.path.join(results_dir, "discovered_patterns.json")
        
        cache_dir = os.path.join(self.BASE_PATH, "Cache")
        os.makedirs(cache_dir, exist_ok=True) 
        self.CACHE_PATH = os.path.join(cache_dir, "feature_cache.parquet")
        self.CACHE_METADATA_PATH = os.path.join(cache_dir, "feature_cache_metadata.json")

# =============================================================================
# 3. GEMINI AI ANALYZER & API TIMER
# =============================================================================

class APITimer:
    def __init__(self, interval_seconds: int = 61):
        self.interval = timedelta(seconds=interval_seconds)
        self.last_call_time: Optional[datetime] = None
        if self.interval.total_seconds() > 0: logger.info(f"API Timer initialized with a {self.interval.total_seconds():.0f}-second interval.")
        else: logger.info("API Timer initialized with a 0-second interval (timer is effectively disabled).")
    def _wait_if_needed(self):
        if self.interval.total_seconds() <= 0: return
        if self.last_call_time is None: return
        elapsed = datetime.now() - self.last_call_time
        wait_time_delta = self.interval - elapsed
        wait_seconds = wait_time_delta.total_seconds()
        if wait_seconds > 0:
            logger.info(f"  - Time since last API call: {elapsed.total_seconds():.1f} seconds.")
            logger.info(f"  - Waiting for {wait_seconds:.1f} seconds to respect the {self.interval.total_seconds():.0f}s interval...")
            if hasattr(logging, 'flush') and callable(logging.flush): logging.flush() # More robust flush
            elif sys.stdout and hasattr(sys.stdout, 'flush') and callable(sys.stdout.flush): sys.stdout.flush()
            time.sleep(wait_seconds)
        else: logger.info(f"  - Time since last API call ({elapsed.total_seconds():.1f}s) exceeds interval. No wait needed.")
    def call(self, api_function: Callable, *args, **kwargs) -> Any:
        self._wait_if_needed()
        self.last_call_time = datetime.now()
        logger.info(f"  - Making API call to '{api_function.__name__}' at {self.last_call_time.strftime('%H:%M:%S')}...")
        result = api_function(*args, **kwargs)
        logger.info(f"  - API call to '{api_function.__name__}' complete.")
        return result

class GeminiAnalyzer:
    def __init__(self):
        self.api_key = os.getenv("GEMINI_API_KEY")
        self.api_key_valid = True # Assume valid initially
        if not self.api_key or "YOUR" in self.api_key or "PASTE" in self.api_key:
            logger.warning("!CRITICAL! GEMINI_API_KEY not found in environment or is a placeholder.")
            try:
                self.api_key = input(">>> Please paste your Gemini API Key and press Enter, or press Enter to skip: ").strip()
                if not self.api_key:
                    logger.warning("No API Key provided. AI analysis will be skipped.")
                    self.api_key_valid = False
                else:
                    logger.info("Using API Key provided via manual input.")
            except Exception:
                logger.warning("Could not read input (non-interactive environment?). AI analysis will be skipped.")
                self.api_key_valid = False
                self.api_key = None
        else:
            logger.info("Successfully loaded GEMINI_API_KEY from environment.")

        self.headers = {"Content-Type": "application/json"}
        self.primary_model = "gemini-2.0-flash"
        self.backup_model = "gemini-1.5-flash"
        self.tools = [{"function_declarations": [{"name": "search_web", "description": "Searches web.", "parameters": {"type": "object", "properties": {"query": {"type": "string"}}, "required": ["query"]}}]}]
        self.tool_config = {"function_calling_config": {"mode": "AUTO"}}

    def _sanitize_dict(self, data: Any) -> Any:
        if isinstance(data, dict):
            # --- LOGIC TO HANDLE NON-STRING KEYS ---
            new_dict = {}
            for k, v in data.items():
                # Sanitize the key first
                if isinstance(k, (datetime, date, pd.Timestamp)):
                    new_key = k.isoformat()
                # Ensure other non-standard key types are converted to string
                elif not isinstance(k, (str, int, float, bool, type(None))):
                    new_key = str(k)
                else:
                    new_key = k
                
                # Recursively sanitize the value
                new_dict[new_key] = self._sanitize_dict(v)
            return new_dict

        elif isinstance(data, list):
            return [self._sanitize_dict(elem) for elem in data]
        elif isinstance(data, (datetime, date)):
            return data.isoformat()
        elif isinstance(data, Enum):
            return data.value
        elif isinstance(data, (int, float, str, bool, type(None))):
            return data
        else:
            # Fallback for any other type
            return str(data)

    def classify_asset_symbols(self, symbols: List[str]) -> Dict[str, str]:
        """
        Uses the Gemini API to classify a list of financial symbols.
        """
        logger.info(f"-> Engaging AI to classify {len(symbols)} asset symbols...")

        prompt = (
            "You are a financial data expert. Your task is to classify a list of trading symbols into their most specific asset class.\n\n"
            f"**SYMBOLS TO CLASSIFY:**\n{json.dumps(symbols, indent=2)}\n\n"
            "**INSTRUCTIONS:**\n"
            "1.  For each symbol, determine its asset class from the following options: 'Forex', 'Indices', 'Commodities', 'Stocks', 'Crypto'.\n"
            "2.  'XAUUSD' is 'Commodities', 'US30' is 'Indices', 'EURUSD' is 'Forex', etc.\n"
            "3.  Respond ONLY with a single, valid JSON object that maps each symbol string to its classification string.\n\n"
            "**EXAMPLE JSON RESPONSE:**\n"
            "```json\n"
            "{\n"
            '  "EURUSD": "Forex",\n'
            '  "XAUUSD": "Commodities",\n'
            '  "US30": "Indices",\n'
            '  "AAPL": "Stocks",\n'
            '  "BTCUSD": "Crypto"\n'
            "}\n"
            "```"
        )

        response_text = self._call_gemini(prompt)
        classified_assets = self._extract_json_from_response(response_text)

        if isinstance(classified_assets, dict) and all(isinstance(k, str) and isinstance(v, str) for k, v in classified_assets.items()):
            logger.info("  - AI successfully classified asset symbols.")
            for symbol in symbols:
                if symbol not in classified_assets:
                    classified_assets[symbol] = "Unknown"
                    logger.warning(f"  - AI did not classify '{symbol}'. Marked as 'Unknown'.")
            return classified_assets

        logger.error("  - AI failed to return a valid symbol classification dictionary. Using fallback detection.")
        return {}

    def _call_gemini(self, prompt: str) -> str:
        if not self.api_key_valid:
            logger.warning("Gemini API key is not valid. Skipping API call.")
            return "{}"

        # Reflected to  fit Gemini 2.0 Flash's large context window ---
        # Set a very generous limit (in characters), well within the 1M token limit.
        max_prompt_length = 950000 
        
        if len(prompt) > max_prompt_length:
            # We only warn, we do NOT truncate the prompt anymore.
            logger.warning(f"Prompt length ({len(prompt)}) is very large, approaching the model's context window limit of ~1M tokens.")        

        payload = {
            "contents": [{"parts": [{"text": prompt}]}],
            "tools": self.tools,
            "tool_config": self.tool_config
        }
        # For non-tool use, ensure tools and tool_config are not in payload or are None
        # This logic appears to always include tools, which may be intentional for some prompts.
        # A more robust solution might pass a flag, but respecting "no refactor" rule.
        if "search_web" not in prompt: # Simple heuristic to disable tools for non-search prompts
            payload.pop("tools", None)
            payload.pop("tool_config", None)

        sanitized_payload = self._sanitize_dict(payload)

        models_to_try = [self.primary_model, self.backup_model]
        retry_delays = [5, 15, 30] # Seconds

        for model_name in models_to_try:
            logger.info(f"Attempting to call Gemini API with model: {model_name}")
            api_url = f"https://generativelanguage.googleapis.com/v1beta/models/{model_name}:generateContent?key={self.api_key}"

            for attempt, delay in enumerate([0] + retry_delays):
                if delay > 0:
                    logger.warning(f"Retrying API call to {model_name} in {delay} seconds... (Attempt {attempt + 1}/{len(retry_delays) + 1})")
                    flush_loggers()
                    time.sleep(delay)
                try:
                    response = requests.post(api_url, headers=self.headers, data=json.dumps(sanitized_payload), timeout=180) # Increased timeout
                    response.raise_for_status()

                    result = response.json()

                    # Standard text response parsing
                    if "candidates" in result and result["candidates"]:
                        candidate = result["candidates"][0]
                        if "content" in candidate and "parts" in candidate["content"] and candidate["content"]["parts"]:
                            text_part = candidate["content"]["parts"][0].get("text")
                            if text_part:
                                logger.info(f"Successfully received and extracted text response from model: {model_name}")
                                return text_part

                    logger.error(f"Invalid Gemini response structure from {model_name}: No 'text' part found. Response: {result}")
                    # Continue to next attempt or model if structure is not as expected

                except requests.exceptions.HTTPError as e:
                    logger.error(f"!! HTTP Error for model '{model_name}': {e.response.status_code} {e.response.reason}")
                    if e.response and e.response.text:
                        logger.error(f"   - API Error Details: {e.response.text}")
                    if e.response is not None and e.response.status_code in [400, 401, 403, 404, 429]: # Non-retryable or specific errors
                        break # Break from retries for this model, try next model
                except requests.exceptions.RequestException as e:
                    logger.error(f"Gemini API request failed for model {model_name} on attempt {attempt + 1} (Network Error): {e}")
                except json.JSONDecodeError as e:
                    logger.error(f"Failed to decode Gemini response JSON from {model_name}: {e} - Response Text: {response.text if 'response' in locals() else 'N/A'}")

            logger.warning(f"Failed to get a valid text response from model {model_name} after all retries.")

        logger.critical("API connection failed for all primary and backup models. Could not get a final text response.")
        return "{}"

    def _extract_json_from_response(self, response_text: str) -> dict:
        logger.debug(f"RAW AI RESPONSE TO BE PARSED:\n--- START ---\n{response_text}\n--- END ---")
        match_backticks = re.search(r"```json\s*([\s\S]*?)\s*```", response_text, re.DOTALL)
        if match_backticks:
            json_str = match_backticks.group(1).strip()
            try:
                suggestions = json.loads(json_str)
                if isinstance(suggestions, dict):
                    logger.info("Successfully extracted JSON object from within backticks.")
                    if isinstance(suggestions.get("current_params"), dict):
                        nested_params = suggestions.pop("current_params")
                        suggestions.update(nested_params)
                    return suggestions
                else:
                    logger.warning(f"Parsed JSON from backticks was type {type(suggestions)}, not a dictionary. Continuing search.")
            except JSONDecodeError as e:
                logger.warning(f"JSON decoding from backticks failed: {e}. Trying broader search.")
        decoder = JSONDecoder()
        pos = 0
        while pos < len(response_text):
            brace_pos = response_text.find('{', pos)
            if brace_pos == -1:
                break
            try:
                suggestions, end_pos = decoder.raw_decode(response_text, brace_pos)
                if isinstance(suggestions, dict):
                    logger.info("Successfully extracted JSON object using JSONDecoder.raw_decode.")
                    if isinstance(suggestions.get("current_params"), dict):
                        nested_params = suggestions.pop("current_params")
                        suggestions.update(nested_params)
                    return suggestions
                else:
                    logger.warning(f"JSONDecoder.raw_decode found a JSON structure of type {type(suggestions)}, not a dict. Continuing search.")
                    pos = end_pos
            except JSONDecodeError:
                pos = brace_pos + 1
        logger.error("!! CRITICAL JSON PARSE FAILURE !! No valid JSON dictionary could be decoded from the AI response.")
        return {}

    def establish_strategic_directive(self, historical_results: List[Dict], current_state: OperatingState) -> str:
        logger.info("-> Establishing strategic directive for the upcoming cycle...")

        if current_state == OperatingState.DRAWDOWN_CONTROL:
            directive = (
                "**STRATEGIC DIRECTIVE: PHASE 3 (DRAWDOWN CONTROL)**\n"
                "The system is in a drawdown. Your absolute priority is capital preservation. "
                "Your suggestions must aggressively reduce risk. Prioritize stability over performance. "
                "Your goal is to stop the losses and find any stable, working model to re-establish a baseline."
            )
            logger.info(f"  - Directive set to: DRAWDOWN CONTROL")
            return directive

        can_optimize = False
        if len(historical_results) >= 2:
            last_two_cycles = historical_results[-2:]
            completed_successfully = all(c.get("Status") == "Completed" for c in last_two_cycles)
            executed_trades = any(c.get("NumTrades", 0) > 0 for c in last_two_cycles)

            if completed_successfully and executed_trades:
                can_optimize = True
                logger.info("  - Logic Check: Last two cycles completed and at least one traded. Permitting optimization.")
            elif completed_successfully and not executed_trades:
                logger.warning("  - Logic Check: Last two cycles completed but NEITHER executed trades. Forcing baseline establishment.")
            else:
                logger.info("  - Logic Check: One or more of the last two cycles did not complete successfully. Baseline establishment is required.")

        if can_optimize:
            directive = (
                "**STRATEGIC DIRECTIVE: PHASE 2 (PERFORMANCE OPTIMIZATION)**\n"
                "A stable baseline model is trading successfully. Your primary goal is now to improve profitability and "
                "risk-adjusted returns. Focus on refining features based on SHAP, tuning risk parameters, and "
                "maximizing financial metrics like Sortino or MAR ratio."
            )
            logger.info(f"  - Directive set to: PERFORMANCE OPTIMIZATION")
        else:
            directive = (
                "**STRATEGIC DIRECTIVE: PHASE 1 (BASELINE ESTABLISHMENT)**\n"
                "The system has not yet found a stable, consistently trading model. Your primary goal is to find a configuration "
                "that can pass the F1-score quality gate AND **execute trades**. Prioritize suggestions that make the model *less* conservative "
                "and more likely to participate in the market (e.g., simpler features, easier labeling, lower confidence gates). "
                "A model that trades is better than a perfect model that never trades."
            )
            logger.info(f"  - Directive set to: BASELINE ESTABLISHMENT")

        return directive

    def get_comprehensive_initial_setup(self, script_version: str, ledger: Dict, memory: Dict, playbook: Dict, health_report: Dict, directives: List[Dict], data_summary: Dict, diagnosed_regime: str, regime_champions: Dict, correlation_summary_for_ai: str, master_macro_list: Dict, prime_directive_str: str) -> Dict:
        """
        Performs a single, comprehensive AI call guided by a clear,
        context-aware primary objective to prevent analysis paralysis.
        """
        if not self.api_key_valid:
            logger.warning("No API key. Skipping AI-driven comprehensive setup.")
            return {}

        logger.info("-> Performing Comprehensive Initial AI Setup...")
        logger.info(f"  - PRIME DIRECTIVE FOR THIS RUN: {prime_directive_str}")
        asset_list_str = ", ".join(data_summary.get('assets_detected', []))

        # The prompt now starts with the single most important instruction.
        prompt = (
            "You are a Master Trading Strategist configuring a trading framework. Your goal is to produce a single JSON object containing the complete setup for the first run.\n\n"
            f"**PRIME DIRECTIVE:** {prime_directive_str}\n\n"
            "All of your choices below for strategy, features, and parameters MUST be optimized to achieve this single prime directive above all else.\n\n"
            "**TASK 1: DEFINE BROKER SIMULATION COSTS**\n"
            "   - Based on the asset list, populate `COMMISSION_PER_LOT` and a reasonable `SPREAD_CONFIG` dictionary.\n\n"
            "**TASK 2: SELECT STRATEGY & PARAMETERS TO ACHIEVE THE PRIME DIRECTIVE**\n"
            "   - Analyze all context to select the best strategy and parameters from the playbook.\n"
            "   - If the directive is to ESTABLISH A TRADING BASELINE, you MUST choose parameters that encourage trading (e.g., wider labeling quantiles like 0.85/0.15, simpler features, lower-complexity strategies). A model that trades is a success; a model that does not is a failure, regardless of theoretical quality.\n"
            "   - Provide a unique `nickname` and `analysis_notes` justifying how your choices directly serve the Prime Directive.\n\n"
            "**TASK 3: SELECT MACROECONOMIC TICKERS**\n"
            "   - Select a dictionary of relevant tickers from the `MASTER_MACRO_TICKER_LIST`.\n\n"

            "--- REQUIRED JSON OUTPUT STRUCTURE ---\n"
            "Respond ONLY with a single, valid JSON object.\n"
            "{\n"
            '  "strategy_name": "string",\n'
            '  "selected_features": ["list", "of", "strings"],\n'
            '  "nickname": "string",\n'
            '  "analysis_notes": "string",\n'
            '  "TP_ATR_MULTIPLIER": "float",\n'
            '  "SL_ATR_MULTIPLIER": "float",\n'
            '  "LOOKAHEAD_CANDLES": "integer",\n'
            '  "LABEL_LONG_QUANTILE": "float",\n'
            '  "LABEL_SHORT_QUANTILE": "float",\n'
            '  "COMMISSION_PER_LOT": "float",\n'
            '  "SPREAD_CONFIG": {},\n'
            '  "selected_macro_tickers": {} \n'
            "}\n\n"
            "--- CONTEXT FOR YOUR DECISION ---\n"
            f"**1. MASTER MACRO TICKER LIST (for Task 3):**\n{json.dumps(master_macro_list, indent=2)}\n\n"
            f"**2. MARKET DATA SUMMARY & CORRELATION (for Task 2):**\n`diagnosed_regime`: '{diagnosed_regime}'\n{correlation_summary_for_ai}\n{json.dumps(self._sanitize_dict(data_summary), indent=2)}\n\n"
            f"**3. STRATEGY PLAYBOOK (for Task 2):**\n{json.dumps(self._sanitize_dict(playbook), indent=2)}\n\n"
            f"**4. FRAMEWORK MEMORY (for Task 2):**\n{json.dumps(self._sanitize_dict(memory), indent=2)}\n"
        )

        response_text = self._call_gemini(prompt)
        suggestions = self._extract_json_from_response(response_text)

        if suggestions and "strategy_name" in suggestions and "selected_macro_tickers" in suggestions:
            logger.info("  - Comprehensive Initial AI Setup complete.")
            return suggestions
        else:
            logger.error("  - AI-driven comprehensive setup failed validation. The JSON was malformed or missing required keys.")
            return {}
    
    def determine_optimal_label_quantiles(self, signal_pressure_summary: Dict, prime_directive: str) -> Dict:
        """
        Analyzes the statistical properties of a signal to recommend optimal labeling quantiles
        based on a high-level strategic directive.
        """
        if not self.api_key_valid:
            return {}

        logger.info("-> Engaging AI to determine optimal labeling quantiles...")

        prompt = (
            "You are an expert data scientist optimizing a trading model's data labeling process.\n"
            "You have been given a statistical summary of a 'signal_pressure' metric, where high values suggest future upward movement and low values suggest downward movement. Your task is to choose the upper and lower quantiles for labeling this data.\n\n"
            f"**PRIME DIRECTIVE:** {prime_directive}\n\n"
            "**INSTRUCTIONS:**\n"
            "1.  **Align with the Prime Directive.** Your choice of quantiles MUST reflect the directive's goal.\n"
            "    - If the directive is to **ESTABLISH TRADING BASELINE**, you must choose **wider, less-extreme quantiles** (e.g., `long_quantile` between 0.80-0.90, `short_quantile` between 0.10-0.20). This creates a more balanced dataset to ensure the model learns to trade.\n"
            "    - If the directive is to **OPTIMIZE PERFORMANCE**, you can choose **stricter, more extreme quantiles** (e.g., `long_quantile` > 0.90, `short_quantile` < 0.10). This prioritizes signal precision over trade frequency.\n"
            "2.  **Analyze the Statistical Summary.** Use the `skew` and `kurtosis` values to inform your decision. For highly skewed data, you might consider asymmetric quantiles. For data with high kurtosis (fat tails), more extreme quantiles might be appropriate to capture only the strongest signals.\n"
            "3.  **Provide Your Recommendation.** Respond ONLY with a single, valid JSON object containing your two chosen quantile values.\n\n"
            "--- DATA FOR YOUR ANALYSIS ---\n\n"
            f"**Statistical Summary of Signal Pressure:**\n{json.dumps(signal_pressure_summary, indent=2)}\n\n"
            "--- REQUIRED JSON OUTPUT ---\n"
            "```json\n"
            "{\n"
            '  "LABEL_LONG_QUANTILE": "float, // e.g., 0.85",\n'
            '  "LABEL_SHORT_QUANTILE": "float, // e.g., 0.15"\n'
            "}\n"
            "```"
        )

        response_text = self._call_gemini(prompt)
        suggestions = self._extract_json_from_response(response_text)

        # Basic validation of the AI's response
        if suggestions and isinstance(suggestions.get('LABEL_LONG_QUANTILE'), float) and isinstance(suggestions.get('LABEL_SHORT_QUANTILE'), float):
            logger.info(f"  - AI has determined optimal quantiles: Long={suggestions['LABEL_LONG_QUANTILE']}, Short={suggestions['LABEL_SHORT_QUANTILE']}")
            return suggestions
        
        logger.warning("  - AI failed to return valid quantiles. Will use framework defaults.")
        return {}

    def analyze_cycle_and_suggest_changes(
        self,
        historical_results: List[Dict],
        strategy_details: Dict, # This will be the current config dump
        cycle_status: str,
        shap_history: Dict[str, Any], # Can be pd.DataFrame or Dict of DFs
        available_features: List[str],
        strategic_directive: str
    ) -> Dict:
        if not self.api_key_valid: return {}

        base_prompt_intro = "You are an expert trading model analyst and portfolio manager. Your goal is to make intelligent, data-driven changes to the trading model configuration to align with the current strategic directive."

        task_guidance = (
            "**YOUR TASK:**\n"
            "1.  **Review the `STRATEGIC DIRECTIVE`.** This is your most important instruction.\n"
            "2.  **Analyze the `CYCLE STATUS` and `HISTORICAL RESULTS`.**\n"
            "   - **SPECIAL RULE:** If `NumTrades` for the last cycle was `0` and the `STRATEGIC DIRECTIVE` is `BASELINE ESTABLISHMENT`, your primary suggestion MUST be to make the model *less* conservative (e.g., lower `STATIC_CONFIDENCE_GATE` or adjust `LABEL_LONG_QUANTILE` / `LABEL_SHORT_QUANTILE` to be less extreme).\n"
            "3.  **Propose Standard Changes:** If the directive is to optimize, suggest changes to the current configuration that are aligned with the goal. Consider `TP_ATR_MULTIPLIER`, `SL_ATR_MULTIPLIER`, `LOOKAHEAD_CANDLES`, risk parameters, or feature refinement based on SHAP if available.\n"
            "4.  **Propose Exploration (Optional):** If the `STRATEGIC DIRECTIVE` is `PERFORMANCE OPTIMIZATION` and you have a strong, research-backed hypothesis for a novel strategy that could outperform the current one, you can propose to invent a new one by returning `{\"action\": \"EXPLORE_NEW_STRATEGY\"}`. Use this option judiciously."
        )

        json_schema_definition = (
            "### REQUIRED JSON RESPONSE STRUCTURE ###\n"
            "// To propose exploring a new strategy, return: {\"action\": \"EXPLORE_NEW_STRATEGY\", \"analysis_notes\": \"Your reasoning...\"}\n"
            "// To propose standard parameter changes, return a JSON object with the new parameters.\n"
            "// If no changes are needed, return an empty JSON object: {}\n"
            "{\n"
            '  "analysis_notes": "str,            // Your detailed reasoning for the suggested changes, referencing the STRATEGIC DIRECTIVE.",\n'
            '  "model_confidence_score": "int,    // Your 1-10 confidence in this configuration decision.",\n'
            '  // ... and any other parameter from the ConfigModel you wish to change (e.g., "STATIC_CONFIDENCE_GATE", "TP_ATR_MULTIPLIER") OR the "action" key.\n'
            "}\n"
            "Respond ONLY with the JSON object.\n"
        )
        # SHAP history for multi-task model might be a dict of dataframes.
        # Present it to the AI in a digestible format.
        shap_summary_for_ai = "SHAP data not available or not applicable for this model."
        if isinstance(shap_history, pd.DataFrame) and not shap_history.empty: # Single model SHAP
             shap_summary_for_ai = shap_history.head(10).to_string()
        elif isinstance(shap_history, dict) and shap_history: # Multi-task SHAP
            shap_reports = []
            for model_name, shap_df in shap_history.items():
                if isinstance(shap_df, pd.DataFrame) and not shap_df.empty:
                    shap_reports.append(f"  SHAP for '{model_name}':\n{shap_df.head(5).to_string()}")
            if shap_reports:
                shap_summary_for_ai = "\n".join(shap_reports)


        data_context = (
            f"--- DATA FOR YOUR ANALYSIS ---\n\n"
            f"**A. CURRENT CYCLE STATUS:** `{cycle_status}`\n\n"
            f"**B. CURRENT RUN - CYCLE-BY-CYCLE HISTORY:**\n{json.dumps(self._sanitize_dict(historical_results), indent=2)}\n\n"
            f"**C. FEATURE IMPORTANCE HISTORY (SHAP values over time or for current models):**\n{shap_summary_for_ai}\n\n"
            f"**D. CURRENT STRATEGY & AVAILABLE FEATURES:**\n`strategy_name`: {strategy_details.get('strategy_name')}\n`available_features`: {available_features}\n"
            f"**E. CURRENT CONFIGURATION (Relevant Thresholds):**\n"
            f"   - `STATIC_CONFIDENCE_GATE`: {strategy_details.get('STATIC_CONFIDENCE_GATE')}\n"
            f"   - `LABEL_LONG_QUANTILE`: {strategy_details.get('LABEL_LONG_QUANTILE')}\n"
            f"   - `LABEL_SHORT_QUANTILE`: {strategy_details.get('LABEL_SHORT_QUANTILE')}\n"
        )

        prompt = (
            f"{base_prompt_intro}\n\n"
            f"{strategic_directive}\n\n"
            f"{task_guidance}\n\n"
            f"{json_schema_definition}\n\n{data_context}"
        )

        response_text = self._call_gemini(prompt)
        suggestions = self._extract_json_from_response(response_text)
        return suggestions

    def select_best_tradeoff(self, best_trials: List[optuna.trial.FrozenTrial], risk_profile: str, strategic_directive: str) -> int:
        if not self.api_key_valid:
            logger.warning("No API key. Skipping AI-driven trade-off selection. Selecting trial with highest primary objective.")
            return max(best_trials, key=lambda t: t.values[0] if t.values else -float('inf')).number

        if not best_trials:
            raise ValueError("Cannot select from an empty list of trials.")

        trial_summaries = []
        for trial in best_trials:
            obj1_val = trial.values[0] if trial.values and len(trial.values) > 0 else float('nan')
            obj2_val = trial.values[1] if trial.values and len(trial.values) > 1 else float('nan')
            trial_summaries.append(
                f" - Trial {trial.number}: Objective 1 Score = {obj1_val:.4f}, Objective 2 Score = {obj2_val:.4f}"
            )
        trials_text = "\n".join(trial_summaries)

        prompt = (
            "You are a portfolio manager performing model selection. You have a Pareto front of models from multi-objective optimization.\n\n"
            f"{strategic_directive}\n\n"
            "**YOUR TASK:**\n"
            "Review the trials and the strategic directive. Select the single best trial that aligns with the current phase of the plan.\n\n"
            "**CRITICAL RULE FOR BASELINE ESTABLISHMENT:**\n"
            "If the `STRATEGIC DIRECTIVE` is `PHASE 1 (BASELINE ESTABLISHMENT)` and the framework has failed to execute trades in the previous cycle, you **MUST** prioritize the trial that maximizes **Objective 2 (related to trade frequency, e.g., 'maximize_log_trades')**, as long as its Objective 1 (e.g., F1 score) is still reasonable. Your goal is to break the deadlock and get a trading model onto the books.\n\n"
            "**PARETO FRONT OF MODELS:**\n"
            f"{trials_text}\n\n"
            "**JSON OUTPUT FORMAT:**\n"
            "```json\n"
            "{\n"
            '  "selected_trial_number": "int, // The number of the trial you have chosen.",\n'
            '  "analysis_notes": "str // Your reasoning, explicitly referencing the strategic directive and the critical rule."\n'
            "}\n"
            "```"
        )

        response_text = self._call_gemini(prompt)
        suggestions = self._extract_json_from_response(response_text)
        selected_trial_number_str = suggestions.get('selected_trial_number')
        selected_trial_number = None
        if isinstance(selected_trial_number_str, str):
            match = re.search(r'\d+', selected_trial_number_str)
            if match:
                selected_trial_number = int(match.group(0))
        elif isinstance(selected_trial_number_str, int):
            selected_trial_number = selected_trial_number_str

        if selected_trial_number is not None:
            valid_numbers = {t.number for t in best_trials}
            if selected_trial_number in valid_numbers:
                logger.info(f"  - AI has selected Trial #{selected_trial_number} based on the strategic directive.")
                logger.info(f"  - AI Rationale: {suggestions.get('analysis_notes', 'N/A')}")
                return selected_trial_number
            else:
                logger.error(f"  - AI selected an invalid trial number ({selected_trial_number}). Falling back to best primary objective.")
                return max(best_trials, key=lambda t: t.values[0] if t.values else -float('inf')).number
        else:
            logger.error("  - AI failed to return a valid trial number. Falling back to best primary objective.")
            return max(best_trials, key=lambda t: t.values[0] if t.values else -float('inf')).number
            
    def propose_mid_cycle_intervention(self, failure_history: List[Dict], pre_analysis_summary: str, current_config: Dict, playbook: Dict, quarantine_list: List[str]) -> Dict:
        """
        [ENHANCED PHILOSOPHY] Performs root-cause analysis on training or backtesting failures,
        guiding the AI with strategic goals rather than prescriptive numbers.
        """
        if not self.api_key_valid: return {}
        logger.warning("! AI DOCTOR !: Engaging advanced diagnostics for course-correction.")

        available_playbook = {k: v for k, v in playbook.items() if k not in quarantine_list and not v.get("retired")}

        # Determine the failure type to select the correct prompt
        if "0 trades" in pre_analysis_summary:
            logger.info("  - Diagnosing a 'Zero Trades' failure mode. Applying goal-oriented guidance.")
            task_prompt = (
                "**PRIME DIRECTIVE: AI INTERVENTION - INDUCE MARKET PARTICIPATION**\n"
                "The current strategy is training successfully but has failed to execute trades for multiple cycles. This indicates the model is too conservative for the 'BASELINE ESTABLISHMENT' phase.\n\n"
                "**STRATEGIC GOAL:** Your task is to propose parameter changes that will make the model **less selective and more likely to trade**. You must analyze the current configuration and suggest a logical adjustment to achieve this goal.\n\n"
                "**PRESCRIPTION - CHOOSE THE MOST LOGICAL ACTION FROM YOUR TOOLKIT:**\n"
                "1.  **`OVERRIDE_CONFIDENCE_GATE`**: This is a direct tool to lower the entry barrier. Analyze the model's typical prediction probabilities (if available) and select a new, lower `STATIC_CONFIDENCE_GATE` that is more likely to be met. You MUST also set `USE_STATIC_CONFIDENCE_GATE` to `true`.\n\n"
                "2.  **`WIDEN_LABELING_QUANTILES`**: This is a data-centric tool. The current quantiles are creating a highly imbalanced dataset (likely ~90% 'Hold' signals). Propose new, less extreme `LABEL_LONG_QUANTILE` and `LABEL_SHORT_QUANTILE` values to create a more balanced training set (e.g., aiming for a 70-80% 'Hold' distribution).\n\n"
                "3.  **`CHANGE_LABELING_METHOD`**: An alternative data-centric tool. If the 'signal_pressure' method seems fundamentally too strict, propose switching to a different, potentially simpler, labeling method."
            )
            json_schema_definition = (
                "### REQUIRED JSON RESPONSE STRUCTURE ###\n"
                "{\n"
                '  "action": "str, // MUST be one of: OVERRIDE_CONFIDENCE_GATE, WIDEN_LABELING_QUANTILES, CHANGE_LABELING_METHOD",\n'
                '  "parameters": "Dict, // REQUIRED. You must determine and provide the optimal parameters for your chosen action based on the strategic goal.",\n'
                '  "analysis_notes": "str // Justify your choice of action and parameters, explaining how they will achieve the goal of increasing trade frequency."\n'
                "}\n"
            )
        else: # Original prompt for training failures
            logger.info("  - Diagnosing a 'Training Failure' mode.")
            task_prompt = (
                # This part remains the same as it deals with a different problem
                "**PRIME DIRECTIVE: AI DOCTOR - DIAGNOSE AND PRESCRIBE**\n"
                "The current model is failing to train. Your task is to act as an expert data scientist, diagnose the **root cause** of the failure using the provided diagnostics, and prescribe a single, logical intervention.\n\n"
                # ... (rest of the training failure prompt)
            )
            json_schema_definition = (
                 "### REQUIRED JSON RESPONSE STRUCTURE ###\n"
                "// You MUST choose exactly ONE action and provide a detailed diagnosis in `analysis_notes`.\n"
                "// The `parameters` key is REQUIRED for all actions and MUST contain the new values.\n"
                "{\n"
                '  "action": "str, // MUST be one of: ADJUST_LABELING_DIFFICULTY, REFINE_FEATURE_SET, CHANGE_LABELING_METHOD",\n'
                '  "parameters": "Dict, // REQUIRED. Example for ADJUST_LABELING_DIFFICULTY: {\\"LOOKAHEAD_CANDLES\\": 50}. Example for CHANGE_LABELING_METHOD: {\\"LABELING_METHOD\\": \\"new_method_name\\"}",\n'
                '  "analysis_notes": "str // Your detailed diagnosis and reasoning for the chosen action."\n'
                "}\n"
            )

        prompt = (
            "You are the AI Doctor, a lead quantitative strategist performing a real-time intervention on a failing model.\n\n"
            f"{task_prompt}\n\n"
            f"{json_schema_definition}\n"
            "Respond ONLY with the JSON object.\n\n"
            "--- DIAGNOSTIC REPORT & CONTEXT ---\n\n"
            f"**1. PRE-ANALYSIS SUMMARY (YOUR PRIMARY EVIDENCE):**\n{pre_analysis_summary}\n\n"
            f"**2. RAW FAILURE DATA (Attempt-by-Attempt):**\n{json.dumps(self._sanitize_dict(failure_history), indent=2)}\n\n"
            f"**3. CURRENT CONFIGURATION:**\n{json.dumps(self._sanitize_dict(current_config), indent=2)}\n\n"
        )
        
        response_text = self._call_gemini(prompt)
        return self._extract_json_from_response(response_text)        

    def propose_strategic_intervention(self, failure_history: List[Dict], playbook: Dict, last_failed_strategy: str, quarantine_list: List[str], dynamic_best_config: Optional[Dict] = None) -> Dict:
        if not self.api_key_valid: return {}
        logger.warning("! STRATEGIC INTERVENTION !: Current strategy has failed repeatedly. Engaging AI for a new path.")

        is_quarantined = last_failed_strategy in quarantine_list
        generative_option_prompt = ""
        if is_quarantined:
             generative_option_prompt = (
                f"\n**OPTION C: INVENT A NEW STRATEGY (Generative)**\n"
                f"   - The current strategy `{last_failed_strategy}` is quarantined. This is a chance to be creative.\n"
                f"   - Propose a new hybrid strategy by combining elements of successful strategies with the concept of the failed one.\n"
                f"   - To select this, respond with: `{{\"action\": \"invent_new_strategy\"}}`. The framework will then prompt you separately to define the new strategy."
            )

        available_playbook = { k: v for k, v in playbook.items() if k not in quarantine_list and not v.get("retired") and (GNN_AVAILABLE or not v.get("requires_gnn"))}
        feature_selection_guidance = (
            "**You MUST provide a `selected_features` list.** Start with a **small, targeted set of 4-6 features** from the playbook for the new strategy you choose. "
            "The list MUST include at least TWO multi-timeframe context features (e.g., `DAILY_ctx_Trend`, `H1_ctx_SMA`)."
        )
        base_prompt = (
            f"You are a master strategist executing an emergency intervention. The current strategy, "
            f"**`{last_failed_strategy}`**, has failed multiple consecutive cycles and is now in quarantine: {is_quarantined}.\n\n"
            f"**RECENT FAILED HISTORY (for context):**\n{json.dumps(self._sanitize_dict(failure_history), indent=2)}\n\n"
            f"**AVAILABLE STRATEGIES (PLAYBOOK - excluding quarantined {quarantine_list}):**\n{json.dumps(self._sanitize_dict(available_playbook), indent=2)}\n\n"
        )
        if dynamic_best_config:
            best_strat_name = dynamic_best_config.get('final_params', {}).get('strategy_name', 'N/A')
            best_strat_mar = dynamic_best_config.get('final_metrics', {}).get('mar_ratio', 0)
            anchor_option_prompt = (
                f"**OPTION A: REVERT TO PERSONAL BEST (The Anchor)**\n"
                f"   - Revert to the most successful configuration from this run: **`{best_strat_name}`** (achieved a MAR Ratio of: {best_strat_mar:.2f}).\n"
                f"   - This is a safe, data-driven reset to a proven state. This option weighs the safety of a proven configuration against the risk of exploration.\n"
                f"   - To select this, respond with: `{{\"action\": \"revert\"}}`\n\n"
                f"**OPTION B: EXPLORE A NEW STRATEGY**\n"
                f"   - Propose a brand new strategy from the available playbook. **Prioritize Simplicity:** Strongly prefer a `complexity` of 'low' or 'medium' to return to a stable baseline.\n"
                f"   - To select this, respond with the full JSON configuration for the new strategy (including `strategy_name`, `selected_features`, etc.). "
                f"   - {feature_selection_guidance}\n"
            )
            prompt = (
                f"{base_prompt}"
                "**YOUR TASK: CHOOSE YOUR NEXT MOVE**\n\n"
                f"{anchor_option_prompt}"
                f"{generative_option_prompt}"
            )
        else:
            explore_only_prompt = (
                 "**CRITICAL INSTRUCTIONS:**\n"
                f"1.  **CRITICAL CONSTRAINT:** The following strategies are in 'quarantine' due to recent, repeated failures. **YOU MUST NOT SELECT ANY STRATEGY FROM THIS LIST: {quarantine_list}**\n"
                "2.  **Select a NEW, SIMPLER STRATEGY:** You **MUST** choose a *different* strategy from the available playbook that is NOT in the quarantine list. Prioritize strategies with a `complexity` of 'low' or 'medium'.\n"
                f"3.  **Propose a Safe Starting Configuration:** Provide a reasonable and SAFE starting configuration for this new strategy. {feature_selection_guidance} Start with conservative values for thresholds (e.g., STATIC_CONFIDENCE_GATE: 0.75), `RETRAINING_FREQUENCY`: '90D', `MAX_DD_PER_CYCLE`: 0.15 (float), `RISK_PROFILE`: 'Medium', `OPTUNA_TRIALS`: 50.\n"
            )
            prompt = (
                f"{base_prompt}"
                f"{explore_only_prompt}\n"
                "Respond ONLY with a valid JSON object for the new configuration, including `strategy_name` and `selected_features` and other relevant parameters like `STATIC_CONFIDENCE_GATE`."
            )
        response_text = self._call_gemini(prompt)
        suggestions = self._extract_json_from_response(response_text)
        return suggestions

    def propose_playbook_amendment(self, quarantined_strategy_name: str, framework_history: Dict, playbook: Dict) -> Dict:
        if not self.api_key_valid: return {}
        logger.warning(f"! PLAYBOOK REVIEW !: Strategy '{quarantined_strategy_name}' is under review for permanent amendment due to chronic failure.")
        prompt = (
            "You are a Head Strategist reviewing a chronically failing trading strategy for a permanent amendment to the core `strategy_playbook.json`.\n\n"
            f"**STRATEGY UNDER REVIEW:** `{quarantined_strategy_name}`\n\n"
            "**YOUR TASK:**\n"
            "Analyze this strategy's performance across the entire `FRAMEWORK HISTORY`. Based on its consistent failures, you must propose a permanent change to its definition in the playbook. You have three options:\n\n"
            "1.  **RETIRE:** If the strategy is fundamentally flawed and unsalvageable, mark it for retirement. "
            "Respond with `{\"action\": \"retire\", \"analysis_notes\": \"Your reasoning...\"}`.\n\n"
            "2.  **REWORK:** If the strategy's concept is sound but its implementation is poor, propose a new, more robust default configuration. This means changing its default `selected_features` and/or other parameters like `dd_range` to be more conservative. "
            "Respond with `{\"action\": \"rework\", \"new_config\": { ... new parameters ... }, \"analysis_notes\": \"Your reasoning...\"}`.\n\n"
            "3.  **NO CHANGE:** If you believe the recent failures were anomalous and the strategy does not warrant a permanent change, you can choose to do nothing. "
            "Respond with `{\"action\": \"no_change\", \"analysis_notes\": \"Your reasoning...\"}`.\n\n"
            "**CRITICAL:** You MUST provide a brief justification for your decision in an `analysis_notes` key.\n"
            "Your response must be a single JSON object with an `action` key and other keys depending on the action.\n\n"
            "--- CONTEXT ---\n"
            f"**1. CURRENT PLAYBOOK DEFINITION for `{quarantined_strategy_name}`:**\n{json.dumps(self._sanitize_dict(playbook.get(quarantined_strategy_name, {})), indent=2)}\n\n"
            f"**2. FULL FRAMEWORK HISTORY (LAST 10 RUNS):**\n{json.dumps(self._sanitize_dict(framework_history.get('historical_runs', [])[-10:]), indent=2)}\n"
        )
        response_text = self._call_gemini(prompt)
        suggestions = self._extract_json_from_response(response_text)
        return suggestions

    def propose_regime_based_strategy_switch(self, regime_data: Dict, playbook: Dict, current_strategy_name: str, quarantine_list: List[str]) -> Dict:
        if not self.api_key_valid: return {}
        logger.info("  - Performing Pre-Cycle Regime Analysis...")
        available_playbook = {k: v for k, v in playbook.items() if k not in quarantine_list and not v.get("retired")}
        prompt = (
            "You are a market regime analyst. The framework is about to start a new walk-forward cycle.\n\n"
            "**YOUR TASK:**\n"
            f"The framework is currently configured to use the **`{current_strategy_name}`** strategy. Based on the `RECENT MARKET DATA SUMMARY` provided below, decide if this is still the optimal choice.\n\n"
            "1.  **Analyze the Data**: Review the `average_adx`, `volatility_rank`, and `trending_percentage` to diagnose the current market regime (e.g., strong trend, weak trend, ranging, volatile, quiet).\n"
            "2.  **Review the Playbook**: Compare your diagnosis with the intended purpose of the strategies in the `STRATEGY PLAYBOOK`.\n"
            "3.  **Make a Decision**:\n"
            "    - If you believe a **different strategy is better suited** to the current market regime, respond with the JSON configuration for that new strategy (just the strategy name and a **small, targeted feature set of 4-6 features** from its playbook defaults). **Complexity Preference**: Unless there is a strong reason, prefer switching to a strategy of 'low' or 'medium' complexity to maintain stability.\n"
            "    - If you believe the **current strategy remains the best fit**, respond with `null`.\n\n"
            "**RESPONSE FORMAT**: Respond ONLY with the JSON for the new strategy OR the word `null`.\n\n"
            "--- CONTEXT FOR YOUR DECISION ---\n"
            f"**1. RECENT MARKET DATA SUMMARY (Last ~30 Days):**\n{json.dumps(self._sanitize_dict(regime_data), indent=2)}\n\n"
            f"**2. STRATEGY PLAYBOOK (Your options):**\n{json.dumps(self._sanitize_dict(available_playbook), indent=2)}\n"
        )
        response_text = self._call_gemini(prompt)
        if response_text.strip().lower() == 'null':
            logger.info("  - AI analysis confirms current strategy is optimal for the upcoming regime. No changes made.")
            return {}
        suggestions = self._extract_json_from_response(response_text)
        return suggestions

    def propose_new_playbook_strategy(self, failed_strategy_name: str, playbook: Dict, framework_history: Dict) -> Dict:
        if not self.api_key_valid: return {}

        logger.info(f"-> Engaging AI to invent a new strategy to replace the failed '{failed_strategy_name}'...")

        successful_strategies = [
            run for run in framework_history.get("historical_runs", [])
            if run.get("final_metrics", {}).get("mar_ratio", 0) > 0.5
        ]
        positive_example = random.choice(successful_strategies) if successful_strategies else None

        positive_example_prompt = ""
        if positive_example:
            positive_example_prompt = (
                f"**INSPIRATION:** As a good example, the strategy `{positive_example.get('strategy_name')}` was successful in a previous run. "
                f"It achieved a MAR ratio of {positive_example.get('final_metrics', {}).get('mar_ratio', 0):.2f}. "
                f"Its description is: \"{playbook.get(positive_example.get('strategy_name'), {}).get('description')}\". "
                "Consider blending concepts from this successful strategy with the failed one."
            )
        else:
            positive_example_prompt = "There are no highly successful strategies in recent history. You must be creative and propose a robust, simple strategy from scratch."

        prompt = (
            "You are a Senior Quantitative Strategist tasked with inventing a cutting-edge trading strategy for our playbook. The existing strategy has been quarantined due to repeated failures.\n\n"
            f"**FAILED STRATEGY:** `{failed_strategy_name}`\n"
            f"**FAILED STRATEGY DETAILS:** {json.dumps(playbook.get(failed_strategy_name, {}), indent=2)}\n\n"
            f"{positive_example_prompt}\n\n"
            "**YOUR TASK:**\n"
            "1. **Perform Grounded Research:** Before inventing a strategy, use your web search tool to find novel ideas relevant to the current market. Look for:\n"
            "    - Recently published (last 1-2 years) academic papers on quantitative finance or signal processing.\n"
            "    - Unconventional combinations of technical indicators discussed on reputable trading forums or blogs.\n"
            "    - Concepts from other fields (e.g., machine learning, physics) that have been applied to financial markets.\n"
            "    - *Example Search Query: 'novel alpha signals from order book imbalance' or 'combining Hilbert-Huang transform with RSI for trading'.*\n\n"
            "2. **Synthesize and Invent:** Combine the most promising insights from your external research with the internal context (what worked and what failed historically). Create a new, hybrid strategy. Give it a creative, descriptive name (e.g., 'WaveletMomentumFilter', 'HawkesProcessBreakout').\n\n"
            "3. **Write a Clear Description:** Explain the logic of your new strategy. What is its core concept? What market regime is it designed for?\n\n"
            "4. **Define Key Parameters:** Set `complexity`, `ideal_regime`, `dd_range`, and `lookahead_range`. The new strategy definition MUST NOT contain a `selected_features` key. Default features should be implied by the description if necessary for future AI use.\n\n"
            "**OUTPUT FORMAT:** Respond ONLY with a single JSON object for the new strategy entry. The key for the object should be its new name.\n"
            "**EXAMPLE STRUCTURE:**\n"
            "```json\n"
            "{\n"
            '  "NewStrategyName": {\n'
            '    "description": "A clear, concise description of the new strategy logic. Example features: `ATR`, `ADX`.",\n'
            '    "complexity": "medium",\n'
            '    "ideal_regime": ["Some Regime"],\n'
            '    "asset_class_suitability": ["Any"],\n'
            '    "ideal_macro_env": ["Neutral"],\n'
            '    "lookahead_range": [50, 100],\n'
            '    "dd_range": [0.15, 0.30]\n'
            '  }\n'
            "}\n"
            "```"
        )

        response_text = self._call_gemini(prompt)
        new_strategy_definition = self._extract_json_from_response(response_text)

        if new_strategy_definition and isinstance(new_strategy_definition, dict) and len(new_strategy_definition) == 1:
            strategy_name = next(iter(new_strategy_definition))
            strategy_body = new_strategy_definition[strategy_name]
            if isinstance(strategy_body, dict) and 'description' in strategy_body and 'complexity' in strategy_body:
                logger.info(f"  - AI has successfully proposed a new strategy named '{strategy_name}'.")
                return new_strategy_definition

        logger.error("  - AI failed to generate a valid new strategy definition.")
        return {}

    def define_gene_pool(self, strategy_goal: str, available_features: List[str]) -> Dict:
        if not self.api_key_valid: return {}

        logger.info(f"-> Engaging AI to define a gene pool for a '{strategy_goal}' strategy...")

        prompt = (
            "You are a specialist in financial feature engineering. Your task is to provide the building blocks ('genes') for a genetic programming algorithm that will evolve a trading strategy.\n\n"
            f"**STRATEGY GOAL:** The algorithm needs to create a **'{strategy_goal}'** strategy.\n\n"
            "**YOUR TASK:**\n"
            "Based on the strategy goal, select the most relevant components from the provided lists. Your choices will directly influence the search space of the evolutionary algorithm.\n"
            "1.  **Indicators (`indicators`):** From `all_available_features`, select 8-12 indicators that are most relevant to the strategy goal. This is the most important choice.\n"
            "2.  **Operators (`operators`):** Choose a set of comparison operators. Include standards like `>` and `<`. You can also include cross-over style operators like `crosses_above`.\n"
            "3.  **Constants (`constants`):** Provide a list of meaningful numerical constants for comparison (e.g., RSI levels like 30, 70; ADX levels like 25).\n\n"
            "**OUTPUT FORMAT:** Respond ONLY with a single JSON object containing three keys: `indicators`, `operators`, and `constants`.\n\n"
            "--- AVAILABLE FEATURES FOR SELECTION ---\n"
            f"{json.dumps(available_features, indent=2)}\n\n"
            "--- AVAILABLE OPERATORS FOR SELECTION ---\n"
            f"{json.dumps(['>', '<', '>=', '<=', '==', 'crosses_above', 'crosses_below'], indent=2)}"
        )

        response_text = self._call_gemini(prompt)
        gene_pool = self._extract_json_from_response(response_text)

        if gene_pool and all(k in gene_pool for k in ['indicators', 'operators', 'constants']):
            logger.info("  - AI successfully defined the gene pool.")
            # Ensure the AI didn't hallucinate features
            gene_pool['indicators'] = [f for f in gene_pool.get('indicators', []) if f in available_features]
            return gene_pool
        else:
            logger.error("  - AI failed to return a valid gene pool. Using fallback.")
            return {
                "indicators": random.sample(available_features, min(10, len(available_features))),
                "operators": ['>', '<'],
                "constants": [0, 20, 25, 30, 50, 70, 75, 80, 100]
            }

# =============================================================================
# PHASE 3: GENETIC PROGRAMMER
# =============================================================================

class GeneticProgrammer:
    """
    Evolves trading rules using a genetic algorithm.
    The AI defines the 'gene pool' (indicators, operators), and this class
    handles the evolutionary process of creating, evaluating, and evolving
    a population of rule-based strategies.
    """
    def __init__(self, gene_pool: Dict, config: ConfigModel, population_size: int = 50, generations: int = 25, mutation_rate: float = 0.1, crossover_rate: float = 0.7):
        self.config = config
        self.indicators = gene_pool.get('indicators', [])
        self.operators = gene_pool.get('operators', ['>', '<'])
        self.constants = gene_pool.get('constants', [0, 25, 50, 75, 100])
        self.population_size = population_size
        self.generations = generations
        self.mutation_rate = mutation_rate
        self.crossover_rate = crossover_rate
        self.population: List[Tuple[str, str]] = []

        if not self.indicators:
            # Create a dummy config for the error message if the main one isn't fully formed yet
            dummy_params = {'BASE_PATH': '.', 'REPORT_LABEL': 'err', 'INITIAL_CAPITAL': 1, 'LOOKAHEAD_CANDLES': 1, 'TRAINING_WINDOW': '1D', 'RETRAINING_FREQUENCY': '1D', 'FORWARD_TEST_GAP': '1D', 'RISK_CAP_PER_TRADE_USD': 1, 'BASE_RISK_PER_TRADE_PCT': 0.01, 'CONFIDENCE_TIERS': {}, 'selected_features': [], 'run_timestamp': 'err', 'strategy_name': 'err'}
            if not isinstance(config, ConfigModel):
                config = ConfigModel(**dummy_params)
            raise ValueError("GeneticProgrammer cannot be initialized with an empty pool of indicators.")

    def _create_individual_rule(self) -> str:
        """Creates a single logical rule string, e.g., '(RSI > 30)'."""
        indicator1 = random.choice(self.indicators)
        operator = random.choice(self.operators)
        
        # 50/50 chance of comparing to a constant or another indicator
        if random.random() < 0.5 or len(self.indicators) < 2:
            value = random.choice(self.constants)
        else:
            value = random.choice([i for i in self.indicators if i != indicator1])
        
        return f"({indicator1} {operator} {value})"

    def _create_individual_chromosome(self, depth: int = 2) -> str:
        """Creates a full rule string, potentially with multiple conditions."""
        rule = self._create_individual_rule()
        for _ in range(depth - 1):
            logic_op = random.choice(['AND', 'OR'])
            next_rule = self._create_individual_rule()
            rule = f"({rule} {logic_op} {next_rule})"
        return rule

    def create_initial_population(self):
        """Generates the starting population of trading rules."""
        self.population = []
        for _ in range(self.population_size):
            long_rule = self._create_individual_chromosome(depth=random.randint(1, 3))
            short_rule = self._create_individual_chromosome(depth=random.randint(1, 3))
            self.population.append((long_rule, short_rule))
        logger.info(f"  - GP: Initial population of {self.population_size} created.")

    def _parse_and_eval_rule(self, rule_str: str, df: pd.DataFrame) -> pd.Series:
        """Safely evaluates a rule string against a dataframe."""
        try:
            features_in_rule = set(re.findall(r'[a-zA-Z_][a-zA-Z0-9_]*', rule_str))
            
            safe_locals = {}
            for feature in features_in_rule:
                if feature in df.columns:
                    safe_locals[feature] = df[feature]
                elif feature not in ['AND', 'OR']:
                    return pd.Series(False, index=df.index) 

            rule_str = rule_str.replace(' AND ', ' & ').replace(' OR ', ' | ')
            result = eval(rule_str, {"__builtins__": {}}, safe_locals)
            
            if isinstance(result, pd.Series) and result.dtype == bool:
                return result.fillna(False)
            else:
                return pd.Series(False, index=df.index)
        except Exception:
            return pd.Series(False, index=df.index)

    def evaluate_fitness(self, chromosome: Tuple[str, str], df_eval: pd.DataFrame) -> float:
        """Calculates the fitness (Calmar Ratio) of a chromosome on a data slice."""
        long_rule, short_rule = chromosome
        df = df_eval.copy()

        long_signals = self._parse_and_eval_rule(long_rule, df)
        short_signals = self._parse_and_eval_rule(short_rule, df)

        signals = pd.Series(0, index=df.index)
        signals[long_signals] = 1
        signals[short_signals] = -1

        if signals.abs().sum() < 5: return -10.0

        lookahead = self.config.LOOKAHEAD_CANDLES
        tp_mult = self.config.TP_ATR_MULTIPLIER
        sl_mult = self.config.SL_ATR_MULTIPLIER
        
        pnl = np.zeros(len(df))
        
        for i in range(len(df) - lookahead):
            if signals.iloc[i] != 0:
                direction = signals.iloc[i]
                entry_price = df['Close'].iloc[i]
                atr = df['ATR'].iloc[i]

                if pd.isna(atr) or atr <= 0: continue

                tp_level = entry_price + (atr * tp_mult * direction)
                sl_level = entry_price - (atr * sl_mult * direction)
                
                future_highs = df['High'].iloc[i+1 : i+1+lookahead]
                future_lows = df['Low'].iloc[i+1 : i+1+lookahead]
                
                hit_tp = np.any(future_highs >= tp_level) if direction == 1 else np.any(future_lows <= tp_level)
                hit_sl = np.any(future_lows <= sl_level) if direction == 1 else np.any(future_highs >= sl_level)
                
                if hit_tp and not hit_sl: pnl[i] = (atr * tp_mult)
                elif hit_sl: pnl[i] = -(atr * sl_mult)

        pnl_series = pd.Series(pnl)
        if pnl_series.abs().sum() == 0: return -5.0
            
        equity_curve = pnl_series.cumsum()
        running_max = equity_curve.cummax()
        drawdown = running_max - equity_curve
        max_dd = drawdown.max()
        
        total_pnl = equity_curve.iloc[-1]
        calmar = total_pnl / max_dd if max_dd > 0 else total_pnl if total_pnl > 0 else 0.0
        
        complexity_penalty = (len(long_rule.split()) + len(short_rule.split())) / 1000.0
        return calmar - complexity_penalty

    def _selection(self, fitness_scores: List[float]) -> List[Tuple[str, str]]:
        """Selects parents for the next generation using tournament selection."""
        selected = []
        for _ in range(self.population_size):
            tournament_size = 5
            aspirants_indices = random.sample(range(self.population_size), tournament_size)
            winner_index = max(aspirants_indices, key=lambda i: fitness_scores[i])
            selected.append(self.population[winner_index])
        return selected

    def _crossover_rules(self, rule1: str, rule2: str) -> Tuple[str, str]:
        """Performs crossover on two rule strings."""
        if ' AND ' not in rule1 or ' AND ' not in rule2:
            return rule1, rule2

        point1 = rule1.find(' AND ')
        point2 = rule2.find(' AND ')

        new_rule1 = rule1[:point1] + rule2[point2:]
        new_rule2 = rule2[:point2] + rule1[point1:]
        
        return new_rule1, new_rule2
        
    def _crossover(self, parents: List[Tuple[str, str]]) -> List[Tuple[str, str]]:
        """Creates the next generation through crossover."""
        offspring = []
        for i in range(0, self.population_size, 2):
            parent1, parent2 = parents[i], parents[i+1]

            if random.random() < self.crossover_rate:
                long_child1, long_child2 = self._crossover_rules(parent1[0], parent2[0])
                short_child1, short_child2 = self._crossover_rules(parent1[1], parent2[1])
                offspring.append((long_child1, short_child1))
                offspring.append((long_child2, short_child2))
            else:
                offspring.extend([parent1, parent2])
        return offspring

    def _mutate_rule(self, rule: str) -> str:
        """Applies mutation to a single rule string."""
        parts = re.split(r'(\sAND\s|\sOR\s|\s|\(|\))', rule)
        parts = [p for p in parts if p and p.strip()]
        if not parts: return rule

        mutation_point = random.randint(0, len(parts) - 1)
        part_to_mutate = parts[mutation_point]

        if part_to_mutate in self.indicators: parts[mutation_point] = random.choice(self.indicators)
        elif part_to_mutate in self.operators: parts[mutation_point] = random.choice(self.operators)
        elif part_to_mutate in ['AND', 'OR']: parts[mutation_point] = 'OR' if part_to_mutate == 'AND' else 'AND'
        elif part_to_mutate.replace('.','',1).isdigit(): parts[mutation_point] = str(random.choice(self.constants))
            
        return "".join(parts)

    def _mutation(self, offspring: List[Tuple[str, str]]) -> List[Tuple[str, str]]:
        """Applies mutation to the offspring."""
        for i in range(len(offspring)):
            if random.random() < self.mutation_rate:
                long_rule, short_rule = offspring[i]
                mutated_long = self._mutate_rule(long_rule)
                mutated_short = self._mutate_rule(short_rule)
                offspring[i] = (mutated_long, mutated_short)
        return offspring

    def run_evolution(self, df_eval: pd.DataFrame) -> Tuple[Tuple[str, str], float]:
        """Executes the full genetic algorithm to find the best trading rule."""
        logger.info("-> Starting Genetic Programming evolution...")
        self.create_initial_population()

        best_chromosome_overall = self.population[0]
        best_fitness_overall = -np.inf

        for gen in range(self.generations):
            fitness_scores = [self.evaluate_fitness(chromo, df_eval) for chromo in self.population]

            best_fitness_gen = max(fitness_scores)
            best_chromosome_gen = self.population[fitness_scores.index(best_fitness_gen)]

            if best_fitness_gen > best_fitness_overall:
                best_fitness_overall = best_fitness_gen
                best_chromosome_overall = best_chromosome_gen

            logger.info(f"  - GP Generation {gen+1}/{self.generations} | Best Fitness: {best_fitness_gen:.4f} | Overall Best: {best_fitness_overall:.4f}")

            parents = self._selection(fitness_scores)
            offspring = self._crossover(parents)
            self.population = self._mutation(offspring)
            self.population[0] = best_chromosome_gen

        logger.info("-> Genetic Programming evolution finished.")
        logger.info(f"  - Best Evolved Long Rule: {best_chromosome_overall[0]}")
        logger.info(f"  - Best Evolved Short Rule: {best_chromosome_overall[1]}")
        logger.info(f"  - Best Fitness (Calmar): {best_fitness_overall:.4f}")

        return best_chromosome_overall, best_fitness_overall

# =============================================================================
# 5. DATA LOADER & FEATURE ENGINEERING
# =============================================================================

class DataLoader:
    def __init__(self, config: ConfigModel): self.config = config
    def _parse_single_file(self, file_path: str, filename: str) -> Optional[pd.DataFrame]:
        try:
            parts = filename.split('_'); symbol, tf = parts[0], parts[1]
            df = pd.read_csv(file_path, delimiter='\t' if '\t' in open(file_path, encoding='utf-8').readline() else ',')
            df.columns = [c.upper().replace('<', '').replace('>', '') for c in df.columns]
            date_col = next((c for c in df.columns if 'DATE' in c), None)
            time_col = next((c for c in df.columns if 'TIME' in c), None)
            if date_col and time_col: df['Timestamp'] = pd.to_datetime(df[date_col] + ' ' + df[time_col], errors='coerce')
            elif date_col: df['Timestamp'] = pd.to_datetime(df[date_col], errors='coerce')
            else: logger.error(f"  - No date/time columns found in {filename}."); return None
            df.dropna(subset=['Timestamp'], inplace=True); df.set_index('Timestamp', inplace=True)
            col_map = {c: c.capitalize() for c in df.columns if c.lower() in ['open', 'high', 'low', 'close', 'tickvol', 'volume', 'spread']}
            df.rename(columns=col_map, inplace=True)
            vol_col = 'Volume' if 'Volume' in df.columns else 'Tickvol'
            df.rename(columns={vol_col: 'RealVolume'}, inplace=True, errors='ignore')

            df['Symbol'] = symbol

            for col in df.columns:
                if df[col].dtype == 'object' and col != 'Symbol':
                    df[col] = pd.to_numeric(df[col], errors='coerce')

            if 'RealVolume' not in df.columns: df['RealVolume'] = 0
            df['RealVolume'] = pd.to_numeric(df['RealVolume'], errors='coerce').fillna(0).astype('int32')
            for col in ['Open', 'High', 'Low', 'Close']:
                if col in df.columns:
                    df[col] = pd.to_numeric(df[col], errors='coerce').astype('float32')

            return df
        except Exception as e: logger.error(f"  - Failed to load {filename}: {e}", exc_info=True); return None

    def load_and_parse_data(self, filenames: List[str]) -> Tuple[Optional[Dict[str, pd.DataFrame]], List[str]]:
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

        processed_dfs: Dict[str, pd.DataFrame] = {}
        for tf, dfs in data_by_tf.items():
            if dfs:
                combined = pd.concat(dfs)
                # Ensure data is sorted by timestamp before returning
                final_combined = combined.sort_index()
                processed_dfs[tf] = final_combined
                logger.info(f"  - Combined data for {tf}: {len(final_combined):,} rows for {len(final_combined['Symbol'].unique())} symbols.")
        detected_timeframes = list(processed_dfs.keys())
        if not processed_dfs: logger.critical("  - Data loading failed for all files."); return None, []
        logger.info(f"[SUCCESS] Data loading complete. Detected timeframes: {detected_timeframes}")
        return processed_dfs, detected_timeframes
        
class FeatureEngineer:
    TIMEFRAME_MAP = {'M1': 1, 'M5': 5, 'M15': 15, 'M30': 30, 'H1': 60, 'H4': 240, 'D1': 1440, 'DAILY': 1440, 'W1': 10080, 'MN1': 43200}
    ANOMALY_FEATURES = [
        'ATR', 'bollinger_bandwidth', 'RSI', 'RealVolume', 'candle_body_size',
        'pct_change', 'candle_body_size_vs_atr', 'atr_vs_daily_atr', 'MACD_hist',
        'wick_to_body_ratio', 'overnight_gap_pct', 'RSI_zscore', 'volume_ma_ratio', 'volatility_hawkes'
    ]

    def __init__(self, config: 'ConfigModel', timeframe_roles: Dict[str, str], playbook: Dict):
        self.config = config
        self.roles = timeframe_roles
        self.playbook = playbook
        self.hurst_warning_symbols = set()

    # --- Feature Calculation Helper Methods ---
    def label_signal_pressure(self, df: pd.DataFrame, lookahead: int, long_quantile: float = 0.95, short_quantile: float = 0.05) -> Tuple[pd.DataFrame, pd.Series]:
        """
        Generates a multi-class target based on signal quality and returns both
        the labeled DataFrame AND the raw signal_pressure Series for external analysis.
        """
        logger.info(f"-> Stage 3: Generating Flexible Labels via Signal Pressure (Quantiles: {long_quantile}/{short_quantile})...")

        all_labeled_groups = []
        all_pressure_series = []

        for symbol, group in df.groupby('Symbol'):
            group_copy = group.copy()
            if len(group_copy) < lookahead + 5:
                group_copy['target'] = 1 # Default to hold if not enough data
                group_copy['signal_pressure'] = 0.0
                all_labeled_groups.append(group_copy)
                all_pressure_series.append(group_copy['signal_pressure'])
                continue

            def _calculate_forward_sharpe(series):
                future_returns = np.log(series.shift(-lookahead) / series.shift(-1))
                rolling_std = future_returns.rolling(window=lookahead, min_periods=max(2, lookahead // 4)).std()
                return (future_returns / rolling_std.replace(0, np.nan)).fillna(0)
            
            group_copy['signal_pressure'] = _calculate_forward_sharpe(group_copy['Close'])
            all_pressure_series.append(group_copy['signal_pressure'])

            pressure_values = group_copy['signal_pressure'][group_copy['signal_pressure'] != 0].dropna()
            
            if len(pressure_values) < 20:
                logger.warning(f"  - Not enough valid signal pressure values for '{symbol}' to set labels. Defaulting to 'Hold'.")
                group_copy['target'] = 1
                all_labeled_groups.append(group_copy.drop(columns=['signal_pressure']))
                continue
                
            long_threshold = pressure_values.quantile(long_quantile)
            short_threshold = pressure_values.quantile(short_quantile)

            group_copy['target'] = 1 # Default to Hold
            group_copy.loc[group_copy['signal_pressure'] >= long_threshold, 'target'] = 2 # Long
            group_copy.loc[group_copy['signal_pressure'] <= short_threshold, 'target'] = 0 # Short
            
            dist = group_copy['target'].value_counts(normalize=True) * 100
            logger.info(f"    - Label distribution for '{symbol}': Hold={dist.get(1, 0):.2f}%, Long={dist.get(2, 0):.2f}%, Short={dist.get(0, 0):.2f}%")

            all_labeled_groups.append(group_copy.drop(columns=['signal_pressure']))

        final_df = pd.concat(all_labeled_groups) if all_labeled_groups else pd.DataFrame()
        final_pressure_series = pd.concat(all_pressure_series) if all_pressure_series else pd.Series(dtype=float)

        return final_df, final_pressure_series

    def apply_discovered_features(self, df: pd.DataFrame, discovered_patterns: List[Dict]) -> pd.DataFrame:
        if not discovered_patterns: return df
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
        for i, pattern_info in enumerate(discovered_patterns):
            pattern_name = pattern_info.get("pattern_name", f"alpha_pattern_{i+1}")
            lambda_str = pattern_info.get("lambda_function_str")
            description = pattern_info.get("pattern_description", "N/A")
            try:
                if not lambda_str or not isinstance(lambda_str, str) or "lambda row:" not in lambda_str.strip():
                    logger.warning(f"  - Skipping invalid lambda for pattern: {pattern_name}. Lambda: '{lambda_str}'")
                    df_copy[pattern_name] = 0.0
                    continue
                logger.info(f"  - Applying pattern: '{pattern_name}' (Desc: {description})")
                compiled_lambda = eval(lambda_str, safe_globals_for_lambda_definition, {})
                def apply_lambda_safely(row_data):
                    if pd.isna(row_data).all(): return np.nan
                    try: return compiled_lambda(row_data)
                    except Exception: raise
                df_copy[pattern_name] = df_copy.apply(apply_lambda_safely, axis=1).astype(float)
            except Exception as e_outer:
                logger.error(f"  - Failed to define/apply pattern '{pattern_name}': {e_outer}", exc_info=True)
                logger.debug(f"    Problematic lambda for '{pattern_name}': {lambda_str}")
                df_copy[pattern_name] = 0.0
        return df_copy

    def _add_higher_tf_context(self, base_df: pd.DataFrame, higher_tf_df: Optional[pd.DataFrame], tf_name: str) -> pd.DataFrame:
        if higher_tf_df is None or higher_tf_df.empty:
            placeholder_cols = [f"{tf_name}_ctx_Close", f"{tf_name}_ctx_Trend", f"{tf_name}_ctx_ATR", f"{tf_name}_ctx_RSI", f"{tf_name}_ctx_ADX", f"{tf_name}_ctx_RealVolume"]
            for col in placeholder_cols:
                if col not in base_df.columns: base_df[col] = np.nan
            return base_df

        ctx_features_agg = {'Close': 'last', 'High': 'max', 'Low': 'min', 'Open': 'first',
                            'ATR': 'mean', 'RSI': 'mean', 'ADX': 'mean', 'RealVolume': 'sum'}

        base_tf_str = self.roles.get('base', 'M15').upper()
        minutes_base = self.TIMEFRAME_MAP.get(base_tf_str)
        if minutes_base is None:
            logger.warning(f"Base timeframe '{base_tf_str}' not in TIMEFRAME_MAP. Using 15 min default for resampling context.")
            minutes_base = 15
        pandas_freq_base = f"{minutes_base}T"

        if not isinstance(higher_tf_df.index, pd.DatetimeIndex): higher_tf_df.index = pd.to_datetime(higher_tf_df.index)
        if not isinstance(base_df.index, pd.DatetimeIndex): base_df.index = pd.to_datetime(base_df.index)

        resampled_features_dict = {}
        for col, method in ctx_features_agg.items():
            if col in higher_tf_df.columns:
                resampled_series = higher_tf_df[col].resample(pandas_freq_base).agg(method).ffill()
                resampled_features_dict[f"{tf_name}_ctx_{col}"] = resampled_series

        if not resampled_features_dict: return base_df
        resampled_df = pd.DataFrame(resampled_features_dict)
        merged_df = base_df.join(resampled_df, how='left') # Join on index (Timestamp)

        ctx_close_col = f"{tf_name}_ctx_Close"
        if ctx_close_col in merged_df.columns:
            merged_df[f"{tf_name}_ctx_Trend"] = np.sign(merged_df[ctx_close_col].diff(2)).fillna(0).astype(int)
        return merged_df

    def _detect_anomalies(self, df: pd.DataFrame) -> pd.DataFrame: # From V211
        anomaly_features_present = [f for f in self.ANOMALY_FEATURES if f in df.columns and df[f].isnull().sum() < len(df) and df[f].nunique(dropna=False) > 1] # Check nunique > 1
        if not anomaly_features_present:
            df['anomaly_score'] = 1 # Normal (no anomaly)
            return df

        df_anomaly_subset = df[anomaly_features_present].copy()
        for col in df_anomaly_subset.columns: # Impute medians for IF
            if df_anomaly_subset[col].isnull().any():
                df_anomaly_subset[col].fillna(df_anomaly_subset[col].median(), inplace=True)

        if df_anomaly_subset.empty or df_anomaly_subset.nunique(dropna=False).max() < 2:
            df['anomaly_score'] = 1
            return df

        model = IsolationForest(contamination=self.config.anomaly_contamination_factor, random_state=42)
        try:
            # Ensure all data is numeric for IsolationForest
            if not all(np.issubdtype(dtype, np.number) for dtype in df_anomaly_subset.dtypes):
                logger.warning(f"Non-numeric data found in anomaly features for symbol {df['Symbol'].iloc[0] if not df.empty else 'Unknown'}. Skipping anomaly detection.")
                df['anomaly_score'] = 1
                return df

            predictions = model.fit_predict(df_anomaly_subset) # -1 for anomalies, 1 for inliers
            df['anomaly_score'] = pd.Series(predictions, index=df_anomaly_subset.index).reindex(df.index)
        except ValueError as e:
            symbol_for_log = df['Symbol'].iloc[0] if 'Symbol' in df.columns and not df.empty else "UnknownSymbol"
            logger.warning(f"Could not fit IsolationForest for symbol {symbol_for_log}: {e}. Defaulting anomaly_score to 1.")
            df['anomaly_score'] = 1
        df['anomaly_score'].ffill(inplace=True); df['anomaly_score'].bfill(inplace=True); df['anomaly_score'].fillna(1, inplace=True)
        return df

    def _calculate_relative_performance(self, df: pd.DataFrame) -> pd.DataFrame: # From V211
        if 'pct_change' not in df.columns and 'Close' in df.columns and 'Symbol' in df.columns:
             df['pct_change'] = df.groupby('Symbol')['Close'].pct_change()
        elif 'pct_change' not in df.columns: # Single symbol or pct_change pre-calculated
            df['pct_change'] = df['Close'].pct_change()


        df_for_market_ret = df.copy()
        if not isinstance(df_for_market_ret.index, pd.DatetimeIndex):
            if 'Timestamp' in df_for_market_ret.columns: df_for_market_ret = df_for_market_ret.set_index('Timestamp')
            else:
                logger.warning("Timestamp missing for market return calculation. Skipping relative performance.")
                df['relative_performance'] = 0.0
                return df

        if 'pct_change' not in df_for_market_ret.columns or df_for_market_ret['pct_change'].isnull().all():
            logger.warning("pct_change column missing or all NaN. Skipping relative performance.")
            df['relative_performance'] = 0.0
            return df

        # Calculate mean returns across symbols for each timestamp
        mean_market_returns = df_for_market_ret.groupby(df_for_market_ret.index)['pct_change'].mean()
        df['market_return_temp'] = df.index.map(mean_market_returns)
        df['relative_performance'] = df['pct_change'] - df['market_return_temp']
        df.drop(columns=['market_return_temp'], inplace=True, errors='ignore')
        df['relative_performance'].fillna(0.0, inplace=True) # Fill NaNs in final result
        return df

    # --- Individual Feature Calculation Methods ---
    def _calculate_atr(self, df: pd.DataFrame, period: int = 14) -> pd.DataFrame:
        if not all(col in df.columns for col in ['High', 'Low', 'Close']):
            df['ATR'] = np.nan
            logger.warning(f"Cannot calculate ATR due to missing OHLC columns for symbol {df['Symbol'].iloc[0] if 'Symbol' in df.columns and not df.empty else 'Unknown'}.")
            return df

        high_low = df['High'] - df['Low']
        high_close = np.abs(df['High'] - df['Close'].shift())
        low_close = np.abs(df['Low'] - df['Close'].shift())

        tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1, skipna=False)
        # Use Exponential Moving Average for ATR
        df['ATR'] = tr.ewm(alpha=1/period, adjust=False, min_periods=period).mean()
        return df

    def _calculate_price_derivatives(self, g: pd.DataFrame, price_col: str = 'Close') -> pd.DataFrame:
        g[f'{price_col}_velocity'] = g[price_col].diff()
        g[f'{price_col}_acceleration'] = g[f'{price_col}_velocity'].diff()
        g[f'{price_col}_jerk'] = g[f'{price_col}_acceleration'].diff()
        return g

    def _calculate_volume_derivatives(self, g: pd.DataFrame) -> pd.DataFrame:
        if 'RealVolume' in g.columns:
            g['volume_velocity'] = g['RealVolume'].diff()
            g['volume_acceleration'] = g['volume_velocity'].diff()
        else:
            g['volume_velocity'], g['volume_acceleration'] = np.nan, np.nan
        return g

    def _calculate_statistical_moments(self, g: pd.DataFrame, window: int = 20) -> pd.DataFrame:
        log_returns = np.log(g['Close'].replace(0, np.nan) / g['Close'].shift(1).replace(0, np.nan))
        min_p = max(1, window // 2)
        g['returns_skew'] = log_returns.rolling(window, min_periods=min_p).skew()
        g['returns_kurtosis'] = log_returns.rolling(window, min_periods=min_p).kurt()
        return g

    def _calculate_simple_features(self, df: pd.DataFrame) -> pd.DataFrame:
        df['pct_change'] = df.groupby('Symbol')['Close'].pct_change() if 'Symbol' in df.columns else df['Close'].pct_change()
        df['overnight_gap_pct'] = df.groupby('Symbol')['Open'].transform(
            lambda x: (x / x.shift(1).replace(0,np.nan)) - 1
        ) if 'Symbol' in df.columns else (df['Open'] / df['Open'].shift(1).replace(0,np.nan)) -1

        df['candle_body_size'] = (df['Close'] - df['Open']).abs()
        upper_wick = df['High'] - df[['Close', 'Open']].max(axis=1)
        lower_wick = df[['Close', 'Open']].min(axis=1) - df['Low']
        df['wick_to_body_ratio'] = (upper_wick + lower_wick) / df['candle_body_size'].replace(0, np.nan)

        if 'RSI' in df.columns:
            rolling_rsi_mean = df['RSI'].rolling(20, min_periods=10).mean()
            rolling_rsi_std = df['RSI'].rolling(20, min_periods=10).std().replace(0, np.nan)
            df['RSI_zscore'] = (df['RSI'] - rolling_rsi_mean) / rolling_rsi_std

        if 'RealVolume' in df.columns and not df['RealVolume'].empty:
            vol_ma = df['RealVolume'].rolling(20, min_periods=10).mean()
            df['volume_ma_ratio'] = df['RealVolume'] / vol_ma.replace(0, np.nan)

        if 'ATR' in df.columns:
             df['candle_body_size_vs_atr'] = df['candle_body_size'] / df['ATR'].replace(0, np.nan)

        if 'DAILY_ctx_ATR' in df.columns and 'ATR' in df.columns: # Assuming DAILY_ctx_ATR comes from higher TF context
            df['atr_vs_daily_atr'] = df['ATR'] / df['DAILY_ctx_ATR'].replace(0, np.nan)
        return df

    def _calculate_hawkes_volatility(self, df: pd.DataFrame) -> pd.DataFrame:
        if 'ATR' not in df.columns or df['ATR'].isnull().all():
            df['volatility_hawkes'] = np.nan
            return df
        atr_shocks = df['ATR'].diff().clip(lower=0).fillna(0)
        hawkes_intensity = atr_shocks.ewm(alpha=1 - self.config.HAWKES_KAPPA, adjust=False, min_periods=1).mean()
        df['volatility_hawkes'] = hawkes_intensity
        return df

    def _calculate_ohlc_ratios(self, df: pd.DataFrame) -> pd.DataFrame:
        df['ohlc_range'] = df['High'] - df['Low']
        # Avoid division by zero if ohlc_range is 0
        safe_ohlc_range = df['ohlc_range'].replace(0, np.nan)
        df['close_to_high'] = (df['High'] - df['Close']) / safe_ohlc_range
        df['close_to_low'] = (df['Close'] - df['Low']) / safe_ohlc_range
        return df

    def _calculate_accumulation_distribution(self, df: pd.DataFrame) -> pd.DataFrame:
        if not all(col in df.columns for col in ['High', 'Low', 'Close', 'RealVolume']) or df['RealVolume'].isnull().all():
             df['AD_line'], df['AD_line_slope'] = np.nan, np.nan
             return df
        hl_range = (df['High'] - df['Low'])
        clv = np.where(hl_range == 0, 0, ((df['Close'] - df['Low']) - (df['High'] - df['Close'])) / hl_range.replace(0, np.nan))
        clv_series = pd.Series(clv, index=df.index).fillna(0)
        ad = (clv_series * df['RealVolume']).cumsum()
        df['AD_line'] = ad
        df['AD_line_slope'] = df['AD_line'].diff(5) # 5-period slope
        return df

    def _calculate_mad(self, df: pd.DataFrame, window: int = 20) -> pd.DataFrame: # Mean Absolute Deviation
        min_p = max(1,window//2)
        df['mad'] = df['Close'].rolling(window, min_periods=min_p).apply(lambda x: np.mean(np.abs(x - np.mean(x))), raw=True)
        return df

    def _calculate_price_volume_correlation(self, df: pd.DataFrame, window: int = 20) -> pd.DataFrame:
        if 'RealVolume' not in df.columns or df['RealVolume'].isnull().all() or 'Close' not in df.columns:
            df['price_vol_corr'] = np.nan
            return df
        min_p = max(5,window//2) # Need more periods for stable correlation
        # Correlation between returns and volume
        log_returns = np.log(df['Close'].replace(0,np.nan) / df['Close'].shift(1).replace(0,np.nan))
        df['price_vol_corr'] = log_returns.rolling(window, min_periods=min_p).corr(df['RealVolume'])
        return df

    def _calculate_quantile_features(self, df: pd.DataFrame, window: int = 50) -> pd.DataFrame:
        log_returns = np.log(df['Close'].replace(0,np.nan) / df['Close'].shift(1).replace(0,np.nan))
        min_p = max(1,window//2)
        df['return_q25'] = log_returns.rolling(window, min_periods=min_p).quantile(0.25)
        df['return_q75'] = log_returns.rolling(window, min_periods=min_p).quantile(0.75)
        df['return_iqr'] = df['return_q75'] - df['return_q25']
        return df

    def _calculate_regression_features(self, df: pd.DataFrame, window: int = 20) -> pd.DataFrame: # Rolling Slope
        def get_slope(series_arr: np.ndarray):
            valid_values = series_arr[~np.isnan(series_arr)]
            if len(valid_values) < 2 : return np.nan
            y = valid_values; x = np.arange(len(y))
            try: return np.polyfit(x, y, 1)[0]
            except (np.linalg.LinAlgError, ValueError): return np.nan
        min_p = max(2,window//4)
        df['rolling_beta'] = df['Close'].rolling(window, min_periods=min_p).apply(get_slope, raw=True)
        return df

    def _calculate_displacement(self, df: pd.DataFrame) -> pd.DataFrame: # From V211
        df_copy = df.copy()
        df_copy["candle_range"] = np.abs(df_copy["High"] - df_copy["Low"])
        mstd = df_copy["candle_range"].rolling(self.config.DISPLACEMENT_PERIOD, min_periods=max(1,self.config.DISPLACEMENT_PERIOD//2)).std()
        threshold = mstd * self.config.DISPLACEMENT_STRENGTH
        df_copy["displacement_signal_active"] = (df_copy["candle_range"] > threshold).astype(int)
        variation = df_copy["Close"] - df_copy["Open"]
        df["green_displacement"] = ((df_copy["displacement_signal_active"] == 1) & (variation > 0)).astype(int).shift(1).fillna(0)
        df["red_displacement"] = ((df_copy["displacement_signal_active"] == 1) & (variation < 0)).astype(int).shift(1).fillna(0)
        return df

    def _calculate_gaps(self, df: pd.DataFrame) -> pd.DataFrame: # From V211
        lookback = self.config.GAP_DETECTION_LOOKBACK
        df["is_bullish_gap"] = (df["High"].shift(lookback) < df["Low"]).astype(int).fillna(0)
        df["is_bearish_gap"] = (df["High"] < df["Low"].shift(lookback)).astype(int).fillna(0)
        return df

    def _calculate_candle_info(self, df: pd.DataFrame) -> pd.DataFrame: # From V211
        df["candle_way"] = np.sign(df["Close"] - df["Open"]).fillna(0).astype(int)
        ohlc_range = (df["High"] - df["Low"]).replace(0, np.nan)
        df["filling_ratio"] = (np.abs(df["Close"] - df["Open"]) / ohlc_range).fillna(0)
        return df

    def _calculate_parkinson_volatility(self, df: pd.DataFrame) -> pd.DataFrame: # From V211
        window = self.config.PARKINSON_VOLATILITY_WINDOW
        def parkinson_estimator_raw(high_low_log_sq_window_arr: np.ndarray):
            if np.isnan(high_low_log_sq_window_arr).all() or len(high_low_log_sq_window_arr) == 0: return np.nan
            valid_terms = high_low_log_sq_window_arr[~np.isnan(high_low_log_sq_window_arr)]
            if len(valid_terms) == 0: return np.nan
            return np.sqrt(np.sum(valid_terms) / (4 * len(valid_terms) * np.log(2)))

        high_low_ratio_log_sq = (np.log(df['High'].replace(0,np.nan) / df['Low'].replace(0,np.nan)) ** 2).replace([np.inf, -np.inf], np.nan)
        df['volatility_parkinson'] = high_low_ratio_log_sq.rolling(window=window, min_periods=max(1,window//2)).apply(parkinson_estimator_raw, raw=True)
        return df

    def _calculate_yang_zhang_volatility(self, df: pd.DataFrame) -> pd.DataFrame: # From V211
        window = self.config.YANG_ZHANG_VOLATILITY_WINDOW
        # This nested function will now receive a window (as a Series)
        # and use its index to slice the full DataFrame 'df'.
        def yang_zhang_estimator_windowed(window_series: pd.Series) -> float:
            # Slice the original DataFrame 'df' using the index from the passed window
            sub_df_ohlc_window = df.loc[window_series.index]

            if sub_df_ohlc_window.isnull().values.any() or len(sub_df_ohlc_window) < max(2, window // 4): return np.nan
            o, h, l, c = sub_df_ohlc_window['Open'], sub_df_ohlc_window['High'], sub_df_ohlc_window['Low'], sub_df_ohlc_window['Close']
            log_ho, log_lo, log_co = np.log(h / o), np.log(l / o), np.log(c / o)
            log_oc_prev = np.log(o / c.shift(1))
            n = len(sub_df_ohlc_window)
            if log_oc_prev.iloc[1:].isnull().all() or log_co.isnull().all(): return np.nan
            sigma_o_sq = np.nanvar(log_oc_prev.iloc[1:], ddof=0)
            sigma_c_sq = np.nanvar(log_co, ddof=0)
            sigma_rs_sq_terms = log_ho * (log_ho - log_co) + log_lo * (log_lo - log_co)
            sigma_rs_sq = np.nanmean(sigma_rs_sq_terms)
            k = 0.34 / (1.34 + (n + 1) / (n - 1)) if n > 1 else 0.34 / 1.34
            vol_sq = sigma_o_sq + k * sigma_c_sq + (1 - k) * sigma_rs_sq
            return np.sqrt(vol_sq) if vol_sq >= 0 else np.nan

        required_cols = ['Open', 'High', 'Low', 'Close']
        if not all(col in df.columns for col in required_cols):
            df['volatility_yang_zhang'] = np.nan
            return df

        # Roll on a single column (e.g., 'Close') to create the windows.
        # The .apply() will now correctly pass a window of data whose index can be used
        # to slice the full OHLC data from the parent 'df'.
        df['volatility_yang_zhang'] = df['Close'].rolling(
            window=window, min_periods=max(2, window // 2)
        ).apply(yang_zhang_estimator_windowed, raw=False)
        return df

    def _calculate_kama_manual(self, series: pd.Series, n: int = 10, pow1: int = 2, pow2: int = 30) -> pd.Series: # From V211
        change = abs(series - series.shift(n))
        volatility = (series - series.shift()).abs().rolling(n, min_periods=1).sum()
        er = (change / volatility.replace(0, np.nan)).fillna(0).clip(0,1)
        sc_fast, sc_slow = 2 / (pow1 + 1), 2 / (pow2 + 1)
        sc = (er * (sc_fast - sc_slow) + sc_slow) ** 2
        kama_values = pd.Series(index=series.index, dtype=float)
        first_valid_idx = series.first_valid_index()
        if first_valid_idx is None: return kama_values
        kama_values.loc[first_valid_idx] = series.loc[first_valid_idx]
        for i in range(series.index.get_loc(first_valid_idx) + 1, len(series)):
            current_idx, prev_idx = series.index[i], series.index[i-1]
            if pd.isna(series.loc[current_idx]): kama_values.loc[current_idx] = kama_values.loc[prev_idx]; continue
            if pd.isna(kama_values.loc[prev_idx]): kama_values.loc[current_idx] = series.loc[current_idx]
            else: kama_values.loc[current_idx] = kama_values.loc[prev_idx] + sc.loc[current_idx] * (series.loc[current_idx] - kama_values.loc[prev_idx])
        return kama_values

    def _calculate_kama_regime(self, df: pd.DataFrame) -> pd.DataFrame: # From V211
        fast_kama = self._calculate_kama_manual(df["Close"], n=self.config.KAMA_REGIME_FAST, pow1=2, pow2=self.config.KAMA_REGIME_FAST) # pow2 can be same as n for faster KAMA
        slow_kama = self._calculate_kama_manual(df["Close"], n=self.config.KAMA_REGIME_SLOW, pow1=2, pow2=self.config.KAMA_REGIME_SLOW)
        df["kama_trend"] = np.sign(fast_kama - slow_kama).fillna(0).astype(int)
        return df

    def _calculate_cycle_features(self, df: pd.DataFrame, window: int = 40) -> pd.DataFrame: # From V211
        df['dominant_cycle_phase'], df['dominant_cycle_period'] = np.nan, np.nan
        symbol = df['Symbol'].iloc[0] if 'Symbol' in df.columns and not df.empty else 'UNKNOWN_SYMBOL'
        close_series = df['Close'].dropna()
        if len(close_series) < window + 1: return df
        try:
            # Detrending can help Hilbert transform
            detrended_close = close_series - close_series.rolling(window=window, center=True, min_periods=1).mean().fillna(method='bfill').fillna(method='ffill')
            analytic_signal = hilbert(detrended_close.values)
            instantaneous_phase = np.unwrap(np.angle(analytic_signal))
            df.loc[close_series.index, 'dominant_cycle_phase'] = instantaneous_phase
            if len(instantaneous_phase) > 1:
                instantaneous_frequency = np.diff(instantaneous_phase) / (2.0 * np.pi)
                inst_freq_series = pd.Series(instantaneous_frequency, index=close_series.index[1:])
                safe_inst_freq_np = np.where(np.abs(inst_freq_series.values) < 1e-9, np.nan, inst_freq_series.values)
                safe_inst_freq_series = pd.Series(safe_inst_freq_np, index=inst_freq_series.index)
                if not safe_inst_freq_series.isnull().all():
                    dominant_cycle_period_series = 1.0 / np.abs(safe_inst_freq_series)
                    rolling_period = dominant_cycle_period_series.rolling(window=window, min_periods=max(1, window // 2)).mean()
                    df.loc[rolling_period.index, 'dominant_cycle_period'] = rolling_period
        except Exception as e: logger.debug(f"  - Cycle Features Error for symbol {symbol} (window {window}): {e}")
        return df

    def _calculate_autocorrelation_features(self, df: pd.DataFrame, lags: int = 5) -> pd.DataFrame: 
        log_returns = np.log(df['Close'].replace(0,np.nan) / df['Close'].shift(1).replace(0,np.nan)).dropna()
        pacf_window = self.config.AUTOCORR_LAG * 3
        if len(log_returns) < lags + pacf_window + 5:
            for i in range(1, lags + 1): df[f'pacf_lag_{i}'] = np.nan
            return df
        def rolling_pacf_calc(series_window_values: np.ndarray, lag_num_arg: int):
            series_no_nan = series_window_values[~np.isnan(series_window_values)]
            if len(series_no_nan) < lags + 5 : return np.nan
            try:
                pacf_vals = pacf(series_no_nan, nlags=lags, method='yw')
                return pacf_vals[lag_num_arg] if lag_num_arg < len(pacf_vals) else np.nan
            except Exception: return np.nan
        for i in range(1, lags + 1):
            df[f'pacf_lag_{i}'] = log_returns.rolling(window=pacf_window, min_periods=lags + 5).apply(
                rolling_pacf_calc, raw=True, args=(i,)
            )
        return df

    def _calculate_entropy_features(self, df: pd.DataFrame, window: int = 30) -> pd.DataFrame: 
        def roll_entropy_raw(series_arr: np.ndarray):
            series_no_nan = series_arr[~np.isnan(series_arr)]
            if len(series_no_nan) < 2: return np.nan
            try:
                hist, _ = np.histogram(series_no_nan, bins=10, density=False)
                counts = hist / len(series_no_nan)
                return entropy(counts[counts > 0], base=2)
            except ValueError: return np.nan
        min_p = max(1,window//2)
        log_returns = np.log(df['Close'].replace(0,np.nan) / df['Close'].shift(1).replace(0,np.nan))
        df['shannon_entropy_returns'] = log_returns.rolling(window, min_periods=min_p).apply(roll_entropy_raw, raw=True)
        return df

    def _calculate_fourier_transform_features(self, df: pd.DataFrame, window: int = 50) -> pd.DataFrame: 
        def get_dominant_freq_amp_raw(series_arr: np.ndarray) -> tuple[float, float]:
            series_no_nan = series_arr[~np.isnan(series_arr)]
            n = len(series_no_nan)
            if n < max(10, window // 4) or (np.diff(series_no_nan) == 0).all(): return np.nan, np.nan
            try:
                fft_vals = scipy.fft.fft(series_no_nan) # Use scipy.fft
                fft_freq = scipy.fft.fftfreq(n)
                positive_freq_indices = np.where(fft_freq > 1e-9)[0]
                if not positive_freq_indices.any(): return np.nan, np.nan
                idx_max_amplitude = positive_freq_indices[np.argmax(np.abs(fft_vals[positive_freq_indices]))]
                dominant_frequency = np.abs(fft_freq[idx_max_amplitude])
                dominant_amplitude = np.abs(fft_vals[idx_max_amplitude]) / n
                return dominant_frequency, dominant_amplitude
            except Exception: return np.nan, np.nan
        min_p = max(10, window//2)
        results_list_tuples = [get_dominant_freq_amp_raw(w) for w in df['Close'].rolling(window, min_periods=min_p)]
        if results_list_tuples:
            # Create DataFrame from list of tuples
            fft_df_results = pd.DataFrame(results_list_tuples, index=df.index[-len(results_list_tuples):], columns=['fft_dom_freq', 'fft_dom_amp'])
            df['fft_dom_freq'] = fft_df_results['fft_dom_freq'].reindex(df.index)
            df['fft_dom_amp'] = fft_df_results['fft_dom_amp'].reindex(df.index)
        else:
            df['fft_dom_freq'], df['fft_dom_amp'] = np.nan, np.nan
        return df

    def _calculate_wavelet_features(self, df: pd.DataFrame, wavelet_name='db4', level=4) -> pd.DataFrame: # From V211
        if not PYWT_AVAILABLE:
            for i in range(level + 1): df[f'wavelet_coeff_energy_L{i}'] = np.nan
            return df
        min_len_for_wavelet = 30 + (level * 5)
        close_series_dropna = df['Close'].dropna()
        if len(close_series_dropna) < min_len_for_wavelet :
            for i in range(level + 1): df[f'wavelet_coeff_energy_L{i}'] = np.nan
            return df
        try:
            coeffs = pywt.wavedec(close_series_dropna.values, wavelet_name, level=level)
            energies = {}
            for i, c_arr in enumerate(coeffs):
                energies[f'wavelet_coeff_energy_L{i}'] = np.sum(np.square(c_arr)) / len(c_arr) if len(c_arr) > 0 else np.nan
            for col_name, energy_val in energies.items(): df[col_name] = energy_val
        except Exception as e:
            symbol_for_log = df['Symbol'].iloc[0] if 'Symbol' in df.columns and not df.empty else "UnknownSymbol"
            logger.debug(f"  - Wavelet calculation error for symbol {symbol_for_log}: {e}")
            for i in range(level + 1): df[f'wavelet_coeff_energy_L{i}'] = np.nan
        return df

    def _calculate_garch_volatility(self, df: pd.DataFrame) -> pd.DataFrame: # From V211
        df['garch_volatility'] = np.nan
        if not ARCH_AVAILABLE: return df
        log_returns_scaled = np.log(df['Close'].replace(0,np.nan) / df['Close'].shift(1).replace(0,np.nan)).dropna() * 1000
        if len(log_returns_scaled) < 20: return df
        try:
            garch_window = 100; min_periods_garch = max(20, garch_window // 2)
            def rolling_garch_fit(series_window_values: np.ndarray):
                series_no_nan = series_window_values[~np.isnan(series_window_values)]
                if len(series_no_nan) < 20: return np.nan
                try:
                    garch_model = arch_model(series_no_nan, vol='Garch', p=1, q=1, rescale=False, dist='normal')
                    res = garch_model.fit(update_freq=0, disp='off', show_warning=False, options={'maxiter': 50})
                    if res.convergence_flag == 0:
                        forecast = res.forecast(horizon=1, reindex=False, align='origin')
                        pred_vol_scaled_garch = np.sqrt(forecast.variance.iloc[-1,0])
                        return pred_vol_scaled_garch / 1000.0
                    return np.nan
                except Exception: return np.nan
            garch_vol_series = log_returns_scaled.rolling(window=garch_window, min_periods=min_periods_garch).apply(
                rolling_garch_fit, raw=True
            )
            df['garch_volatility'] = garch_vol_series.reindex(df.index).ffill()
        except Exception as e:
            symbol_for_log = df['Symbol'].iloc[0] if 'Symbol' in df.columns and not df.empty else "UnknownSymbol"
            logger.debug(f"  - GARCH Error for symbol {symbol_for_log}: {e}")
        return df

    def _calculate_dynamic_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
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

        df['volatility_regime_label'] = pd.cut(df['market_volatility_index'], bins=[0, 0.3, 0.7, 1.01], labels=['LowVolatility', 'Default', 'HighVolatility'], right=False).astype(str).fillna('Default')
        df['trend_regime_label'] = pd.cut(df['hurst_exponent'], bins=[0, 0.4, 0.6, 1.01], labels=['Ranging', 'Default', 'Trending'], right=False).astype(str).fillna('Default')
        df['market_regime_str'] = df['volatility_regime_label'] + "_" + df['trend_regime_label']
        def get_fallback_regime(row):
            vol_reg, trend_reg = row['volatility_regime_label'], row['trend_regime_label']
            if f"{vol_reg}_{trend_reg}" in self.config.DYNAMIC_INDICATOR_PARAMS: return f"{vol_reg}_{trend_reg}"
            if f"{vol_reg}_Default" in self.config.DYNAMIC_INDICATOR_PARAMS: return f"{vol_reg}_Default"
            if f"Default_{trend_reg}" in self.config.DYNAMIC_INDICATOR_PARAMS: return f"Default_{trend_reg}"
            return "Default"
        df['market_regime_str'] = df.apply(get_fallback_regime, axis=1)
        known_regimes_list = list(self.config.DYNAMIC_INDICATOR_PARAMS.keys())
        df['market_regime'] = pd.Categorical(df['market_regime_str'], categories=known_regimes_list, ordered=True).codes
        df['market_regime'] = df['market_regime'].replace(-1, known_regimes_list.index('Default') if 'Default' in known_regimes_list else 0)

        df['bollinger_upper'], df['bollinger_lower'], df['bollinger_bandwidth'], df['RSI'] = np.nan, np.nan, np.nan, np.nan
        for regime_code_val, group_indices in df.groupby('market_regime').groups.items():
            if group_indices.empty: continue # Skip if group is empty (FIXED)
            regime_name_str = known_regimes_list[regime_code_val]
            params_for_regime = self.config.DYNAMIC_INDICATOR_PARAMS.get(regime_name_str, self.config.DYNAMIC_INDICATOR_PARAMS['Default'])
            group_df_slice = df.loc[group_indices]
            if group_df_slice.empty: continue
            ma = group_df_slice['Close'].rolling(window=params_for_regime['bollinger_period'], min_periods=max(1,params_for_regime['bollinger_period']//2)).mean()
            std = group_df_slice['Close'].rolling(window=params_for_regime['bollinger_period'], min_periods=max(1,params_for_regime['bollinger_period']//2)).std()
            df.loc[group_indices, 'bollinger_upper'] = ma + (std * params_for_regime['bollinger_std_dev'])
            df.loc[group_indices, 'bollinger_lower'] = ma - (std * params_for_regime['bollinger_std_dev'])
            df.loc[group_indices, 'bollinger_bandwidth'] = (df.loc[group_indices, 'bollinger_upper'] - df.loc[group_indices, 'bollinger_lower']) / ma.replace(0, np.nan)
            delta = group_df_slice['Close'].diff()
            gain = delta.clip(lower=0).ewm(alpha=1/params_for_regime['rsi_period'], adjust=False, min_periods=1).mean()
            loss = (-delta.clip(upper=0)).ewm(alpha=1/params_for_regime['rsi_period'], adjust=False, min_periods=1).mean()
            rs = gain / loss.replace(0, np.nan)
            rsi_calc_group = 100 - (100 / (1 + rs.fillna(np.inf)))
            df.loc[group_indices, 'RSI'] = rsi_calc_group.replace([np.inf, -np.inf], 50).fillna(50)
        df.drop(columns=['volatility_regime_label', 'trend_regime_label'], inplace=True, errors='ignore')
        return df

    def _calculate_rsi(self, df: pd.DataFrame, period: int = 14) -> pd.DataFrame: # standalone RSI
        delta = df['Close'].diff()
        gain = delta.clip(lower=0).ewm(alpha=1/period, adjust=False, min_periods=1).mean()
        loss = (-delta.clip(upper=0)).ewm(alpha=1/period, adjust=False, min_periods=1).mean()
        rs = gain / loss.replace(0, np.nan)
        rsi_series = 100 - (100 / (1 + rs.fillna(np.inf)))
        df[f'RSI_{period}'] = rsi_series.replace([np.inf, -np.inf], 50).fillna(50)
        if 'Default' in self.config.DYNAMIC_INDICATOR_PARAMS and \
           period == self.config.DYNAMIC_INDICATOR_PARAMS['Default']['rsi_period'] and \
           'RSI' not in df.columns: # Fallback if dynamic RSI not set
            df['RSI'] = df[f'RSI_{period}']
        return df

    def _calculate_bollinger_bands(self, df: pd.DataFrame, period: int = 20, std_dev: float = 2.0) -> pd.DataFrame: # From V211 (standalone BB)
        ma = df['Close'].rolling(window=period, min_periods=max(1,period//2)).mean()
        std = df['Close'].rolling(window=period, min_periods=max(1,period//2)).std()
        df[f'bollinger_upper_{period}_{std_dev}'] = ma + (std * std_dev)
        df[f'bollinger_lower_{period}_{std_dev}'] = ma - (std * std_dev)
        df[f'bollinger_bandwidth_{period}_{std_dev}'] = (df[f'bollinger_upper_{period}_{std_dev}'] - df[f'bollinger_lower_{period}_{std_dev}']) / ma.replace(0, np.nan)
        if 'Default' in self.config.DYNAMIC_INDICATOR_PARAMS and \
           period == self.config.DYNAMIC_INDICATOR_PARAMS['Default']['bollinger_period'] and \
           std_dev == self.config.DYNAMIC_INDICATOR_PARAMS['Default']['bollinger_std_dev'] and \
           'bollinger_upper' not in df.columns : # Fallback if dynamic BB not set
            df['bollinger_upper'] = df[f'bollinger_upper_{period}_{std_dev}']
            df['bollinger_lower'] = df[f'bollinger_lower_{period}_{std_dev}']
            df['bollinger_bandwidth'] = df[f'bollinger_bandwidth_{period}_{std_dev}']
        return df

    def _calculate_hurst_exponent(self, df: pd.DataFrame, window: int = 100) -> pd.DataFrame: 
        df['hurst_exponent'], df['hurst_intercept'] = np.nan, np.nan
        if not HURST_AVAILABLE: return df
        symbol_for_log = df.get('Symbol', pd.Series(['UnknownSymbol'])).iloc[0] # Safer get
        def apply_hurst_raw(series_arr: np.ndarray, component_index: int):
            series_no_nan = series_arr[~np.isnan(series_arr)]
            if len(series_no_nan) < 20 or np.all(np.diff(series_no_nan) == 0): return np.nan
            try:
                result_tuple = compute_Hc(series_no_nan, kind='price', simplified=True)
                return result_tuple[component_index]
            except Exception as e_hurst:
                if symbol_for_log not in self.hurst_warning_symbols:
                    logger.debug(f"Hurst calculation error for {symbol_for_log} (window {window}): {e_hurst}")
                    self.hurst_warning_symbols.add(symbol_for_log)
                return np.nan
        min_p_hurst = max(20, window // 2)
        rolling_close_hurst = df['Close'].rolling(window=window, min_periods=min_p_hurst)
        df['hurst_exponent'] = rolling_close_hurst.apply(apply_hurst_raw, raw=True, args=(0,))
        df['hurst_intercept'] = rolling_close_hurst.apply(apply_hurst_raw, raw=True, args=(1,))
        return df

    def _calculate_trend_pullback_features(self, df: pd.DataFrame) -> pd.DataFrame: # From V211
        if not all(x in df.columns for x in ['ADX', 'EMA_20', 'EMA_50', 'RSI', 'Close']):
            df['is_bullish_pullback'], df['is_bearish_pullback'] = 0, 0
            return df
        is_uptrend = (df['ADX'] > self.config.ADX_THRESHOLD_TREND) & (df['EMA_20'] > df['EMA_50'])
        is_bullish_pullback_signal = (df['Close'] < df['EMA_20']) & (df['RSI'] < (self.config.RSI_OVERBOUGHT - 10))
        df['is_bullish_pullback'] = (is_uptrend & is_bullish_pullback_signal).astype(int)
        is_downtrend = (df['ADX'] > self.config.ADX_THRESHOLD_TREND) & (df['EMA_20'] < df['EMA_50'])
        is_bearish_pullback_signal = (df['Close'] > df['EMA_20']) & (df['RSI'] > (self.config.RSI_OVERSOLD + 10))
        df['is_bearish_pullback'] = (is_downtrend & is_bearish_pullback_signal).astype(int)
        return df

    def _calculate_divergence_features(self, df: pd.DataFrame, lookback: int = 14) -> pd.DataFrame: 
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
        if not PYKALMAN_AVAILABLE: return series # Skip if library not available
        if series.isnull().all() or len(series.dropna()) < 2: return series
        series_filled = series.copy().ffill().bfill() # Fill NaNs for KF
        if series_filled.isnull().all() or series_filled.nunique() < 2: return series_filled
        try:
            kf = KalmanFilter(initial_state_mean=series_filled.iloc[0] if not series_filled.empty else 0, n_dim_obs=1)
            kf = kf.em(series_filled.values, n_iter=5, em_vars=['transition_covariance', 'observation_covariance', 'initial_state_covariance'])
            (smoothed_state_means, _) = kf.smooth(series_filled.values)
            return pd.Series(smoothed_state_means.flatten(), index=series.index)
        except (ValueError, np.linalg.LinAlgError) as e:
            logger.debug(f"Kalman filter failed on series (len {len(series_filled)}, unique {series_filled.nunique()}): {e}. Returning original.")
            return series

    def _calculate_meta_features(self, df: pd.DataFrame) -> pd.DataFrame: # From V211
        if 'RSI' in df.columns and 'bollinger_bandwidth' in df.columns: df['rsi_x_bolli'] = df['RSI'] * df['bollinger_bandwidth']
        if 'ADX' in df.columns and 'market_volatility_index' in df.columns: df['adx_x_vol_rank'] = df['ADX'] * df['market_volatility_index']
        if 'hurst_exponent' in df.columns and 'ADX' in df.columns: df['hurst_x_adx'] = df['hurst_exponent'] * df['ADX']
        if 'ATR' in df.columns and 'DAILY_ctx_ATR' in df.columns: df['atr_ratio_short_long'] = df['ATR'] / df['DAILY_ctx_ATR'].replace(0, np.nan)
        if 'hurst_intercept' in df.columns and 'ADX' in df.columns: df['hurst_intercept_x_adx'] = df['hurst_intercept'] * df['ADX']
        if 'hurst_intercept' in df.columns and 'ATR' in df.columns: df['hurst_intercept_x_atr'] = df['hurst_intercept'] * df['ATR']
        if 'volatility_parkinson' in df.columns and 'volatility_yang_zhang' in df.columns:
            df['vol_parkinson_yz_ratio'] = df['volatility_parkinson'] / df['volatility_yang_zhang'].replace(0, np.nan)
        return df

    def _calculate_liquidity_features(self, df: pd.DataFrame) -> pd.DataFrame: # From V211
        df_safe = df[(df['High'] >= df['Low']) & (df['Low'] > 0)].copy()
        if df_safe.empty or len(df_safe) < 2:
            df['estimated_spread'], df['illiquidity_ratio'] = np.nan, np.nan
            return df
        beta_sum_sq_log_hl = (np.log(df_safe['High'] / df_safe['Low'].replace(0,np.nan))**2).rolling(window=2, min_periods=2).sum()
        gamma_high = df_safe['High'].rolling(window=2, min_periods=2).max()
        gamma_low = df_safe['Low'].rolling(window=2, min_periods=2).min()
        gamma = (np.log(gamma_high / gamma_low.replace(0,np.nan))**2).fillna(0)
        alpha_denom = (3 - 2 * np.sqrt(2))
        if alpha_denom == 0: alpha_denom = 1e-9 # Avoid division by zero
        # Ensure terms for sqrt are non-negative and denominators are non-zero
        term1_sqrt = np.sqrt(beta_sum_sq_log_hl.clip(lower=0) / 2)
        term2_sqrt = np.sqrt(gamma.clip(lower=0) / alpha_denom) # Denom already handled
        alpha = (np.sqrt(beta_sum_sq_log_hl.clip(lower=0)) - term1_sqrt ) / alpha_denom - term2_sqrt
        alpha = alpha.replace([np.inf, -np.inf], np.nan)
        spread = 2 * (np.exp(alpha) - 1) / (np.exp(alpha).replace(-1, np.nan) + 1) # Avoid exp(alpha) == -1
        df['estimated_spread'] = spread.reindex(df.index)
        if 'ATR' in df.columns: df['illiquidity_ratio'] = df['estimated_spread'] / df['ATR'].replace(0, np.nan)
        else: df['illiquidity_ratio'] = np.nan
        return df

    def _calculate_order_flow_features(self, df: pd.DataFrame) -> pd.DataFrame:
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
        ohlc_range = (df['High'] - df['Low']).replace(0, np.nan)
        df['depth_proxy_filling_ratio'] = (df['Close'] - df['Open']).abs() / ohlc_range
        upper_shadow = df['High'] - df[['Close', 'Open']].max(axis=1)
        lower_shadow = df[['Close', 'Open']].min(axis=1) - df['Low']
        df['upper_shadow_pressure'] = upper_shadow / ohlc_range
        df['lower_shadow_pressure'] = lower_shadow / ohlc_range
        return df

    def _calculate_time_features(self, df: pd.DataFrame) -> pd.DataFrame:
        if not isinstance(df.index, pd.DatetimeIndex):
            logger.warning("DataFrame index is not DatetimeIndex. Skipping time features.")
            cols = ['hour', 'day_of_week', 'is_asian_session', 'is_london_session', 'is_ny_session', 'month', 'week_of_year']
            for col in cols: df[col] = np.nan if col in ['hour', 'day_of_week', 'month', 'week_of_year'] else 0
            return df
        df['hour'] = df.index.hour.astype(float)
        df['day_of_week'] = df.index.dayofweek.astype(float)
        df['month'] = df.index.month.astype(float)
        df['week_of_year'] = df.index.isocalendar().week.astype(float)
        df['is_asian_session'] = ((df.index.hour >= 0) & (df.index.hour < 8)).astype(int)
        df['is_london_session'] = ((df.index.hour >= 7) & (df.index.hour < 16)).astype(int)
        df['is_ny_session'] = ((df.index.hour >= 12) & (df.index.hour < 21)).astype(int)
        return df

    def _calculate_structural_breaks(self, df: pd.DataFrame, window=100) -> pd.DataFrame: # From V211
        log_returns = np.log(df['Close'].replace(0,np.nan) / df['Close'].shift(1).replace(0,np.nan)).dropna()
        df['structural_break_cusum'] = 0
        if len(log_returns) < window: return df
        rolling_mean_ret = log_returns.rolling(window=window, min_periods=max(1,window//2)).mean()
        rolling_std_ret = log_returns.rolling(window=window, min_periods=max(1,window//2)).std().replace(0, np.nan)
        standardized_returns_arr = np.where(rolling_std_ret.notna() & (rolling_std_ret != 0),
                                            (log_returns - rolling_mean_ret) / rolling_std_ret, 0)
        standardized_returns_series = pd.Series(standardized_returns_arr, index=log_returns.index)
        def cusum_calc_raw(x_std_ret_window_arr: np.ndarray):
            x_no_nan = x_std_ret_window_arr[~np.isnan(x_std_ret_window_arr)]
            if len(x_no_nan) < 2: return 0
            cumsum_vals = x_no_nan.cumsum()
            return cumsum_vals.max() - cumsum_vals.min() if len(cumsum_vals) > 0 else 0
        min_p_cusum = max(10, window // 4)
        cusum_stat = standardized_returns_series.rolling(window=window, min_periods=min_p_cusum).apply(cusum_calc_raw, raw=True)
        break_threshold = 5.0 # Example threshold
        df.loc[cusum_stat.index, 'structural_break_cusum'] = (cusum_stat > break_threshold).astype(int)
        df['structural_break_cusum'] = df['structural_break_cusum'].ffill().fillna(0)
        return df

    def _apply_pca_standard(self, df: pd.DataFrame, pca_features: list) -> pd.DataFrame:
        if not pca_features or not all(f in df.columns for f in pca_features):
            logger.warning("PCA features missing from DataFrame. Skipping standard PCA.")
            for i in range(self.config.PCA_N_COMPONENTS): df[f'RSI_PCA_{i+1}'] = np.nan
            return df
        df_pca_subset = df[pca_features].copy().astype(np.float32)
        df_pca_subset.fillna(df_pca_subset.median(), inplace=True)
        df_pca_subset = df_pca_subset.loc[:, df_pca_subset.var(ddof=0) > 1e-6]
        if df_pca_subset.shape[1] < self.config.PCA_N_COMPONENTS or df_pca_subset.shape[0] < self.config.PCA_N_COMPONENTS:
            logger.warning(f"Not enough features/samples for standard PCA. Skipping.")
            for i in range(self.config.PCA_N_COMPONENTS): df[f'RSI_PCA_{i+1}'] = np.nan
            return df
        pipeline = Pipeline([('scaler', StandardScaler()), ('pca', PCA(n_components=self.config.PCA_N_COMPONENTS, random_state=42))])
        try:
            principal_components = pipeline.fit_transform(df_pca_subset)
            pca_cols = [f'RSI_PCA_{i+1}' for i in range(self.config.PCA_N_COMPONENTS)]
            pca_df = pd.DataFrame(principal_components, columns=pca_cols, index=df_pca_subset.index)
            df_out = df.join(pca_df, how='left')
            logger.info(f"Standard PCA complete. Explained variance: {pipeline.named_steps['pca'].explained_variance_ratio_.sum():.2%}")
            return df_out
        except Exception as e:
            logger.error(f"Standard PCA failed: {e}. Skipping.", exc_info=True)
            for i in range(self.config.PCA_N_COMPONENTS): df[f'RSI_PCA_{i+1}'] = np.nan
            return df

    def _apply_pca_incremental(self, df: pd.DataFrame, pca_features: list) -> pd.DataFrame:
        if not pca_features or not all(f in df.columns for f in pca_features):
            logger.warning("PCA features missing from DataFrame. Skipping incremental PCA.")
            for i in range(self.config.PCA_N_COMPONENTS): df[f'RSI_PCA_{i+1}'] = np.nan
            return df

        df_pca_subset = df[pca_features].copy().astype(np.float32)
        df_pca_subset.fillna(df_pca_subset.median(), inplace=True)
        df_pca_subset = df_pca_subset.loc[:, df_pca_subset.var(ddof=0) > 1e-6]

        if df_pca_subset.shape[1] < self.config.PCA_N_COMPONENTS or df_pca_subset.shape[0] < self.config.PCA_N_COMPONENTS:
            logger.warning(f"Not enough features/samples for Incremental PCA. Skipping.")
            for i in range(self.config.PCA_N_COMPONENTS): df[f'RSI_PCA_{i+1}'] = np.nan
            return df

        scaler = StandardScaler()
        ipca = IncrementalPCA(n_components=self.config.PCA_N_COMPONENTS)
        batch_size = min(max(1000, df_pca_subset.shape[0] // 100), len(df_pca_subset))

        if batch_size == 0 and len(df_pca_subset) > 0: batch_size = len(df_pca_subset)

        logger.info(f"Fitting IncrementalPCA in batches of {batch_size}...")
        try:
            sample_for_scaler_fit_size = min(len(df_pca_subset), 50000)
            logger.info(f"Fitting StandardScaler on a sample of {sample_for_scaler_fit_size} rows...")
            if sample_for_scaler_fit_size > 0:
                scaler.fit(df_pca_subset.sample(n=sample_for_scaler_fit_size, random_state=42))
            else:
                logger.warning("Scaler is being fitted on a very small or empty DataFrame.")
                scaler.fit(df_pca_subset)

            total_batches = (df_pca_subset.shape[0] + batch_size - 1) // batch_size
            for i in range(0, df_pca_subset.shape[0], batch_size):
                batch = df_pca_subset.iloc[i:i + batch_size]
                if batch.empty: continue
                
                current_batch_num = (i // batch_size) + 1
                sys.stdout.write(f"\r  - Fitting IPCA on batch {current_batch_num}/{total_batches}...")
                sys.stdout.flush()

                ipca.partial_fit(scaler.transform(batch))
            
            sys.stdout.write('\n')

            logger.info("Transforming full dataset in batches with fitted IncrementalPCA...")
            transformed_batches = []
            for i in range(0, df_pca_subset.shape[0], batch_size):
                batch_to_transform = df_pca_subset.iloc[i:i + batch_size]
                if batch_to_transform.empty: continue
                transformed_batches.append(ipca.transform(scaler.transform(batch_to_transform)))

            if not transformed_batches:
                 logger.warning("No batches transformed by IncrementalPCA. Skipping PCA features.")
                 for i in range(self.config.PCA_N_COMPONENTS): df[f'RSI_PCA_{i+1}'] = np.nan
                 return df

            principal_components = np.vstack(transformed_batches)
            pca_cols = [f'RSI_PCA_{i+1}' for i in range(self.config.PCA_N_COMPONENTS)]

            logger.info("Assigning new PCA features directly to the DataFrame...")
            for i, col_name in enumerate(pca_cols):
                df[col_name] = principal_components[:, i]

            logger.info(f"IncrementalPCA reduction complete. Explained variance: {ipca.explained_variance_ratio_.sum():.2%}")
            return df
        except Exception as e:
            sys.stdout.write('\n')
            logger.error(f"Incremental PCA failed: {e}. Skipping.", exc_info=True)
            for i in range(self.config.PCA_N_COMPONENTS): df[f'RSI_PCA_{i+1}'] = np.nan
            return df

    def _calculate_rsi_series(self, series: pd.Series, period: int) -> pd.Series:
        delta = series.diff()
        gain = delta.clip(lower=0).ewm(alpha=1/period, adjust=False, min_periods=1).mean()
        loss = (-delta.clip(upper=0)).ewm(alpha=1/period, adjust=False, min_periods=1).mean()
        rs = gain / loss.replace(0, np.nan)
        rsi = 100 - (100 / (1 + rs.fillna(np.inf)))
        return rsi.replace([np.inf, -np.inf], 50).fillna(50)


    def _apply_pca_reduction(self, df: pd.DataFrame) -> pd.DataFrame:
        if not self.config.USE_PCA_REDUCTION: return df
        logger.info("Applying PCA reduction to RSI features.")
        df_for_pca_calc = pd.DataFrame(index=df.index)
        for period in self.config.RSI_PERIODS_FOR_PCA:
            if 'Symbol' in df.columns:
                df_for_pca_calc[f'RSI_{period}'] = df.groupby('Symbol')['Close'].transform(
                    lambda s_close: self._calculate_rsi_series(s_close, period)
                )
            else:
                df_for_pca_calc[f'RSI_{period}'] = self._calculate_rsi_series(df['Close'], period)

        rsi_features_for_pca = [f'RSI_{period}' for period in self.config.RSI_PERIODS_FOR_PCA if f'RSI_{period}' in df_for_pca_calc.columns]
        if not rsi_features_for_pca:
            logger.error("No RSI features for PCA were created. Skipping PCA.")
            return df

        df_with_rsi_for_pca = df.join(df_for_pca_calc[rsi_features_for_pca], how='left')
        if df_with_rsi_for_pca[rsi_features_for_pca].dropna().empty:
            logger.error("Not enough data for PCA on RSI after NaN drop. Skipping.")
            return df

        num_samples_for_pca = len(df_with_rsi_for_pca[rsi_features_for_pca].dropna())

        if num_samples_for_pca < 500_000:
            if num_samples_for_pca > self.config.PCA_N_COMPONENTS:
                logger.info(f"Dataset size ({num_samples_for_pca} rows) is suitable for standard PCA.")
                return self._apply_pca_standard(df_with_rsi_for_pca, rsi_features_for_pca)
            else:
                logger.warning(f"Not enough samples for standard PCA ({num_samples_for_pca}). Skipping PCA.")
                return df
        else:
            if num_samples_for_pca >= self.config.PCA_N_COMPONENTS:
                logger.info(f"Dataset size ({num_samples_for_pca} rows) is large. Using Incremental PCA.")
                return self._apply_pca_incremental(df_with_rsi_for_pca, rsi_features_for_pca)
            else:
                logger.warning(f"Not enough samples for Incremental PCA ({num_samples_for_pca}). Skipping PCA.")
                return df

    # --- Main Orchestration Methods ---

    def _process_single_symbol_stack(self, symbol_data_by_tf: Dict[str, pd.DataFrame]) -> Optional[pd.DataFrame]:
        """
        Processes a stack of multi-timeframe data for a single symbol to engineer features.
        This is the core feature generation logic per symbol.
        """
        base_df_orig = symbol_data_by_tf.get(self.roles['base'])
        if base_df_orig is None or base_df_orig.empty:
            logger.warning(f"No base timeframe data for symbol in _process_single_symbol_stack. Skipping.")
            return None

        df = base_df_orig.copy()
        if not isinstance(df.index, pd.DatetimeIndex):
            if 'Timestamp' in df.columns: df = df.set_index('Timestamp')
            else:
                logger.error("Base DataFrame in _process_single_symbol_stack has no DatetimeIndex or Timestamp column.")
                return None
        df = df.sort_index()

        logger.info("    - Calculating foundational indicators (ATR, Hurst, Volatility)...")

        # 1. Foundational (Calculated first as many others depend on them)
        df = self._calculate_atr(df, period=14)

        if 'ATR' in df.columns:
            df['realized_volatility'] = df['Close'].pct_change().rolling(14, min_periods=7).std() * np.sqrt(252)
            df['market_volatility_index'] = df['realized_volatility'].rank(pct=True)
        else:
            df['realized_volatility'] = np.nan
            df['market_volatility_index'] = np.nan

        df = self._calculate_hurst_exponent(df, window=self.config.HURST_EXPONENT_WINDOW)

        logger.info("    - Calculating standard technical indicators (EMAs, ADX, MACD, Stoch)...")

        # 2. Standard Technical Indicators
        df['EMA_20'] = df['Close'].ewm(span=20, adjust=False, min_periods=1).mean()
        df['EMA_50'] = df['Close'].ewm(span=50, adjust=False, min_periods=1).mean()

        if 'High' in df.columns and 'Low' in df.columns and 'ATR' in df.columns:
            plus_dm = df['High'].diff(); plus_dm[plus_dm < 0] = 0
            minus_dm = df['Low'].diff(); minus_dm = abs(minus_dm[minus_dm < 0]); minus_dm.fillna(0, inplace=True)
            plus_di_num = plus_dm.ewm(alpha=1/14, adjust=False, min_periods=1).mean()
            minus_di_num = minus_dm.ewm(alpha=1/14, adjust=False, min_periods=1).mean()
            safe_atr = df['ATR'].replace(0,np.nan)
            plus_di = 100 * (plus_di_num / safe_atr)
            minus_di = 100 * (minus_di_num / safe_atr)
            dx_num = np.abs(plus_di - minus_di)
            dx_den = (plus_di + minus_di).replace(0, np.nan)
            dx = 100 * (dx_num / dx_den)
            df['ADX'] = dx.ewm(alpha=1/14, adjust=False, min_periods=1).mean()
        else:
            df['ADX'] = np.nan

        exp1 = df['Close'].ewm(span=12, adjust=False, min_periods=1).mean()
        exp2 = df['Close'].ewm(span=26, adjust=False, min_periods=1).mean()
        df['MACD'] = exp1 - exp2
        df['MACD_signal'] = df['MACD'].ewm(span=9, adjust=False, min_periods=1).mean()
        df['MACD_hist'] = df['MACD'] - df['MACD_signal']

        if 'Low' in df.columns and 'High' in df.columns and 'Close' in df.columns:
            low_k = df['Low'].rolling(window=self.config.STOCHASTIC_PERIOD, min_periods=max(1,self.config.STOCHASTIC_PERIOD//2)).min()
            high_k = df['High'].rolling(window=self.config.STOCHASTIC_PERIOD, min_periods=max(1,self.config.STOCHASTIC_PERIOD//2)).max()
            df['stoch_k'] = 100 * ((df['Close'] - low_k) / (high_k - low_k).replace(0, np.nan))
        else:
            df['stoch_k'] = np.nan

        df['momentum_20'] = df['Close'].pct_change(20)

        # 3. Dynamic & Regime-Aware Indicators
        df = self._calculate_dynamic_indicators(df)

        logger.info("    - Applying signal enhancement layer (Kalman Filters, Confirmation)...")

        # 4. Signal Enhancement
        if 'RSI' in df.columns: df['RSI_kalman'] = self._apply_kalman_filter(df['RSI'])
        if 'ADX' in df.columns: df['ADX_kalman'] = self._apply_kalman_filter(df['ADX'])
        if 'stoch_k' in df.columns: df['stoch_k_kalman'] = self._apply_kalman_filter(df['stoch_k'])
        df = self._calculate_trend_pullback_features(df)
        df = self._calculate_divergence_features(df)

        logger.info("    - Calculating microstructure and advanced volatility features...")

        # 5. Microstructure & Advanced Volatility
        df = self._calculate_displacement(df)
        df = self._calculate_gaps(df)
        df = self._calculate_candle_info(df)
        df = self._calculate_kama_regime(df)
        df = self._calculate_parkinson_volatility(df)
        df = self._calculate_yang_zhang_volatility(df)

        logger.info("    - Calculating contextual and scientific features...")

        # 6. Contextual (Higher TF) Features
        medium_tf_name = self.roles.get('medium')
        high_tf_name = self.roles.get('high')
        if medium_tf_name and medium_tf_name in symbol_data_by_tf:
            df = self._add_higher_tf_context(df, symbol_data_by_tf.get(medium_tf_name), medium_tf_name)
        if high_tf_name and high_tf_name in symbol_data_by_tf:
            df = self._add_higher_tf_context(df, symbol_data_by_tf.get(high_tf_name), high_tf_name)

        # 7. Scientific & Statistical Features
        df = self._calculate_simple_features(df)
        df = self._calculate_price_derivatives(df)
        df = self._calculate_volume_derivatives(df)
        df = self._calculate_statistical_moments(df)
        df = self._calculate_ohlc_ratios(df)
        df = self._calculate_accumulation_distribution(df)
        df = self._calculate_mad(df)
        df = self._calculate_price_volume_correlation(df)
        df = self._calculate_quantile_features(df)
        df = self._calculate_regression_features(df)
        df = self._calculate_cycle_features(df)
        df = self._calculate_autocorrelation_features(df, lags=self.config.AUTOCORR_LAG)
        df = self._calculate_entropy_features(df)
        df = self._calculate_fourier_transform_features(df)
        df = self._calculate_wavelet_features(df)
        df = self._calculate_garch_volatility(df)
        df = self._calculate_hawkes_volatility(df)
        df = self._calculate_time_features(df)
        df = self._calculate_liquidity_features(df)
        df = self._calculate_order_flow_features(df)
        df = self._calculate_depth_features(df)
        df = self._calculate_structural_breaks(df)

        # 8. Meta Features (interactions between other features)
        df = self._calculate_meta_features(df)

        # 9. Anomaly Detection (last, on engineered features)
        df = self._detect_anomalies(df)

        if 'Symbol' not in df.columns and not df.empty and 'Symbol' in base_df_orig.columns:
            df['Symbol'] = base_df_orig['Symbol'].iloc[0]

        return df

    def engineer_features(self, base_df: pd.DataFrame, data_by_tf: Dict[str, pd.DataFrame]) -> pd.DataFrame:
        """
        Orchestrates the creation of a full feature stack for all symbols.
        This is the main entry point for feature engineering.
        """
        # The base_df argument is unused as the method correctly sources the base data
        # from the data_by_tf dictionary. We keep it for compatibility with the caller.
        logger.info("-> Stage 2: Engineering Full Feature Stack...")
        base_tf_name = self.roles.get('base')
        if base_tf_name not in data_by_tf or data_by_tf[base_tf_name].empty:
            logger.critical(f"Base timeframe '{base_tf_name}' data is missing or empty. Cannot proceed with feature engineering.")
            return pd.DataFrame()

        all_symbols_processed_dfs = []
        base_df_global = data_by_tf[base_tf_name] # This contains all symbols for the base TF

        if 'Symbol' not in base_df_global.columns:
             logger.critical(f"Base timeframe '{base_tf_name}' data is missing 'Symbol' column. Cannot group by symbol.")
             return pd.DataFrame()

        if not isinstance(base_df_global.index, pd.DatetimeIndex) and 'Timestamp' in base_df_global.columns:
             base_df_global = base_df_global.set_index('Timestamp')
        elif not isinstance(base_df_global.index, pd.DatetimeIndex):
            logger.critical(f"Base timeframe '{base_tf_name}' data is missing 'Timestamp' index or column.")
            return pd.DataFrame()

        unique_symbols = base_df_global['Symbol'].unique()
        total_symbols = len(unique_symbols)

        for i, symbol in enumerate(unique_symbols):
            logger.info(f"  - [{i+1}/{total_symbols}] Engineering features for symbol: {symbol}...")
            symbol_specific_data_all_tfs = {}
            for tf_loop_name, df_full_tf_loop in data_by_tf.items():
                if 'Symbol' in df_full_tf_loop.columns and symbol in df_full_tf_loop['Symbol'].unique():
                    symbol_df_for_tf_current = df_full_tf_loop[df_full_tf_loop['Symbol'] == symbol].copy()
                    if not isinstance(symbol_df_for_tf_current.index, pd.DatetimeIndex):
                        if 'Timestamp' in symbol_df_for_tf_current.columns:
                            symbol_df_for_tf_current = symbol_df_for_tf_current.set_index('Timestamp')
                        else:
                            logger.warning(f"Timestamp missing for {symbol} on {tf_loop_name}. Skipping this TF context for this symbol.")
                            continue
                    symbol_specific_data_all_tfs[tf_loop_name] = symbol_df_for_tf_current.sort_index() # Ensure sorted
            
            if symbol_specific_data_all_tfs.get(base_tf_name, pd.DataFrame()).empty:
                logger.warning(f"    - No base timeframe data found for symbol {symbol} after filtering. Skipping this symbol.")
                continue

            processed_symbol_df = self._process_single_symbol_stack(symbol_specific_data_all_tfs)
            if processed_symbol_df is not None and not processed_symbol_df.empty:
                all_symbols_processed_dfs.append(processed_symbol_df)
            else:
                logger.warning(f"    - Feature processing returned no data for symbol {symbol}.")

        if not all_symbols_processed_dfs:
            logger.critical("Feature engineering resulted in no processable data across all symbols.")
            return pd.DataFrame()

        logger.info("  - Concatenating feature data for all symbols...")
        final_df = pd.concat(all_symbols_processed_dfs, sort=False) # Sort=False is fine, will sort by index later
        
        # Ensure Timestamp index for final_df and sort
        if not isinstance(final_df.index, pd.DatetimeIndex) and 'Timestamp' in final_df.columns:
             final_df = final_df.set_index('Timestamp')
        elif not isinstance(final_df.index, pd.DatetimeIndex):
            logger.error("Final concatenated DataFrame has no DatetimeIndex or 'Timestamp' column after symbol processing.")
            return pd.DataFrame()
        final_df = final_df.sort_index() # CRITICAL: Sort by time after concatenation

        logger.info("  - Calculating cross-symbol features (e.g., relative performance)...")
        final_df = self._calculate_relative_performance(final_df)

        if self.config.USE_PCA_REDUCTION:
            logger.info("  - Applying PCA reduction to RSI features (if configured)...")
            final_df = self._apply_pca_reduction(final_df)

        logger.info("  - Adding noise-contrastive features for diagnostics...")
        final_df['noise_1'] = np.random.normal(0, 1, len(final_df))
        final_df['noise_2'] = np.random.uniform(-1, 1, len(final_df))

        logger.info("  - Applying final data shift (features for t-1 predict t) and cleaning...")
        # Define identity columns that should NOT be shifted (prices, volume, symbol, timestamp)
        # Also, 'target_*' columns created by labeling should NOT be shifted here.
        # This shift is for features used as input to the model.
        identity_cols = ['Open','High','Low','Close','RealVolume','Symbol','Timestamp'] # Timestamp is index
        
        # Features to shift are all columns EXCEPT identity and target columns
        feature_cols_to_shift = [c for c in final_df.columns if c not in identity_cols and not c.startswith('target')]
        
        if 'Symbol' in final_df.columns: # Group by symbol before shifting
            final_df[feature_cols_to_shift] = final_df.groupby('Symbol', group_keys=False)[feature_cols_to_shift].shift(1)
        else: # If somehow it's a single symbol df (should not happen if called with multi-symbol data)
            final_df[feature_cols_to_shift] = final_df[feature_cols_to_shift].shift(1)

        final_df.replace([np.inf,-np.inf],np.nan,inplace=True)

        # Drop rows where critical features are NaN AFTER the shift.
        # These features are essential for basic model operation or further calculations.
        critical_features_for_dropna = ['ATR', 'RSI', 'ADX', 'bollinger_bandwidth'] # Example critical features
        subset_for_dropna_check = [f for f in critical_features_for_dropna if f in final_df.columns]

        if subset_for_dropna_check:
            # This dropna is after the shift, so it removes rows where lagged critical features are NaN.
            final_df.dropna(subset=subset_for_dropna_check, how='any', inplace=True)
        else:
            # Fallback: if no critical features defined or present, drop rows where ALL shifted features are NaN
            final_df.dropna(how='all', subset=feature_cols_to_shift, inplace=True)

        logger.info(f"[SUCCESS] Feature engineering (Stage 2) complete. Final dataset shape before labeling: {final_df.shape}")
        return final_df

def _get_available_features_from_df(df: pd.DataFrame) -> List[str]: # Helper
    """Gets all column names that are not OHLC, Symbol, Timestamp, or target."""
    if df.empty: return []
    excluded_cols = ['Open', 'High', 'Low', 'Close', 'Volume', 'Tickvol', 'RealVolume', 'Symbol', 'Timestamp']
    features = [col for col in df.columns if col not in excluded_cols and not col.startswith('target')]
    return features

# =============================================================================
# 6. MODELS & TRAINER
# =============================================================================

def check_label_quality(df_train_labeled: pd.DataFrame, min_label_pct: float = 0.02) -> bool:
    """Checks if the generated labels are of sufficient quality for training."""
    if 'target' not in df_train_labeled.columns or df_train_labeled['target'].abs().sum() == 0:
        logger.warning("  - LABEL SANITY CHECK FAILED: No non-zero labels were generated.")
        return False

    label_counts = df_train_labeled['target'].value_counts(normalize=True)

    long_pct = label_counts.get(2, 0) # Long is class 2
    short_pct = label_counts.get(0, 0) # Short is class 0

    if (long_pct + short_pct) < min_label_pct:
        logger.warning(f"  - LABEL SANITY CHECK FAILED: Total trade labels ({long_pct+short_pct:.2%}) is below threshold ({min_label_pct:.2%}).")
        return False

    logger.info(f"  - Label Sanity Check Passed. Distribution: Longs={long_pct:.2%}, Shorts={short_pct:.2%}")
    return True

class GNNModel(torch.nn.Module if GNN_AVAILABLE else object):
    def __init__(self, in_channels, hidden_channels, out_channels):
        super(GNNModel, self).__init__()
        self.conv1 = GCNConv(in_channels, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, out_channels)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.conv2(x, edge_index)
        return x

class TimeSeriesTransformer(nn.Module if GNN_AVAILABLE else object):
    def __init__(
        self,
        feature_size=9,
        num_layers=2,
        d_model=64,
        nhead=8,
        dim_feedforward=256,
        dropout=0.1,
        seq_length=30,
        prediction_length=1
    ):
        super(TimeSeriesTransformer, self).__init__()
        self.input_fc = nn.Linear(feature_size, d_model)
        self.pos_embedding = nn.Parameter(torch.zeros(1, seq_length, d_model))
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=dim_feedforward,
            dropout=dropout, activation="relu"
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.fc_out = nn.Linear(d_model, prediction_length)

    def forward(self, src):
        batch_size, seq_len, _ = src.shape
        src = self.input_fc(src)
        src = src + self.pos_embedding[:, :seq_len, :]
        src = src.permute(1, 0, 2)
        encoded = self.transformer_encoder(src)
        last_step = encoded[-1, :, :]
        out = self.fc_out(last_step)
        return out

class ModelTrainer:
    def __init__(self, config: ConfigModel, gemini_analyzer: 'GeminiAnalyzer'):
        """
        Initializes the ModelTrainer with all necessary components and logic.
        """
        self.config = config
        self.gemini_analyzer = gemini_analyzer
        self.shap_summary: Optional[pd.DataFrame] = None
        self.class_weights: Optional[Dict] = None
        self.classification_report_str: str = "N/A"
        self.is_minirocket_model = 'MiniRocket' in self.config.strategy_name
        self.objective = 'multi:softprob'
        self.eval_metric = 'mlogloss'

    def train(self, df_train_labeled: pd.DataFrame, feature_list: List[str]) -> Optional[Tuple[Pipeline, float, List[str]]]:
        """
        Main method for training the model.
        Returns the final pipeline, threshold, AND the specific list of feature names the pipeline expects.
        """
        logger.info("-> Starting model training orchestration...")

        sample_size = min(len(df_train_labeled), 75000)
        logger.info(f"  - Using a random sample of {sample_size} rows for the entire training process.")
        df_sample = df_train_labeled.sample(n=sample_size, random_state=42)
        df_sample = df_sample.dropna(subset=['target'])
        
        if df_sample.empty:
            logger.error("Training aborted: Data sample is empty after dropping NaNs from target.")
            return None

        y_sample = df_sample['target'].copy().astype(int)
        X_sample = df_sample[feature_list].copy()

        logger.info(f"  - Stage 1: Running Feature Selection method: '{self.config.FEATURE_SELECTION_METHOD}'...")
        
        X_for_tuning, elite_feature_names, features_for_pipeline_input = None, [], []
        pre_fitted_transformer = None

        if self.config.FEATURE_SELECTION_METHOD == 'pca':
            pruned_list = self._remove_redundant_features(X_sample)
            X_for_tuning, elite_feature_names, pre_fitted_transformer = self._select_features_with_pca(X_sample[pruned_list])
            features_for_pipeline_input = pruned_list # The PCA pipeline needs the original pre-PCA features as input
        else:
            pruned_list = self._remove_redundant_features(X_sample)
            if self.config.FEATURE_SELECTION_METHOD == 'trex':
                elite_features = self._select_features_with_trex(X_sample[pruned_list], y_sample)
            else:
                elite_features = self._select_elite_features_mi(X_sample[pruned_list], y_sample)
            
            if not elite_features:
                logger.error("Feature selection resulted in an empty list. Aborting training.")
                return None
            
            X_for_tuning = X_sample[elite_features]
            elite_feature_names = elite_features
            features_for_pipeline_input = elite_features

        if X_for_tuning is None or X_for_tuning.empty or not features_for_pipeline_input:
            logger.error("Feature selection resulted in an empty feature set. Aborting training.")
            return None
        
        logger.info(f"  - Stage 2: Optimizing hyperparameters with {X_for_tuning.shape[1]} final features...")
        num_classes = y_sample.nunique()
        self.class_weights = dict(zip(np.unique(y_sample), compute_class_weight('balanced', classes=np.unique(y_sample), y=y_sample)))
        
        study = self._optimize_hyperparameters(X_for_tuning, y_sample, num_classes)
        if not study or not study.best_trials: return None
        
        logger.info("  - Stage 3: Selecting best model and training...")
        best_trial = self._select_best_trial_from_study(study, "primary_model") 
        best_threshold, f1_val = self._find_best_threshold(best_trial.params, X_for_tuning, y_sample, num_classes)
        
        final_pipeline = self._train_final_model(
            best_params=best_trial.params,
            X_input=X_sample[features_for_pipeline_input],
            y_input=y_sample,
            final_feature_names=elite_feature_names,
            num_classes=num_classes,
            pre_fitted_transformer=pre_fitted_transformer
        )

        if final_pipeline is None: return None
            
        logger.info(f"[SUCCESS] Model training complete. Best Threshold: {best_threshold:.2f}, Val F1: {f1_val:.3f}")
        return final_pipeline, best_threshold, features_for_pipeline_input

    def _train_minirocket_pipeline(self, df_sample: pd.DataFrame) -> Optional[Tuple[Pipeline, float]]:
        logger.info("  - MiniRocket path selected. Preparing 3D panel data from sample...")
        minirocket_feature_list = ['Close', 'RealVolume', 'ATR']
        lookback_window = 100 
        X_3d, y_3d, _ = self._prepare_3d_data(df_sample, minirocket_feature_list, lookback_window)
        if X_3d.size == 0:
            logger.error("  - Failed to create 3D data for MiniRocket.")
            return None, None
        X_3d_transposed = np.transpose(X_3d, (0, 2, 1))
        logger.info(f"  - Fitting MiniRocket transformer on data with shape: {X_3d_transposed.shape}")
        minirocket_transformer = MiniRocket(random_state=42)
        X_transformed = minirocket_transformer.fit_transform(X_3d_transposed)
        y_mapped = pd.Series(y_3d).astype(int)
        logger.info("  - Training XGBoost classifier on MiniRocket features...")
        xgb_classifier = xgb.XGBClassifier(objective='multi:softprob', num_class=3, eval_metric=self.eval_metric, use_label_encoder=False, seed=42)
        xgb_classifier.fit(X_transformed, y_mapped)
        full_pipeline = Pipeline([('minirocket', minirocket_transformer), ('classifier', xgb_classifier)])
        logger.info("[SUCCESS] MiniRocket model training complete.")
        return full_pipeline, 0.5 

    def _prepare_3d_data(self, df: pd.DataFrame, feature_list: List[str], lookback: int) -> Tuple[np.ndarray, np.ndarray, pd.Index]:
        df_features = df[feature_list].fillna(0)
        X_values = df_features.values
        y_values = df['target'].values
        windows, labels, label_indices = [], [], []
        if len(df_features) < lookback: return np.array([]), np.array([]), pd.Index([])
        for i in range(len(df_features) - lookback + 1):
            windows.append(X_values[i : i + lookback])
            label_idx = i + lookback - 1
            labels.append(y_values[label_idx])
            label_indices.append(df.index[label_idx])
        return np.stack(windows), np.array(labels), pd.Index(label_indices)

    def _select_features_with_pca(self, X: pd.DataFrame) -> Tuple[pd.DataFrame, list, Pipeline]:
        logger.info(f"    - Reducing dimensionality using PCA to {self.config.PCA_N_COMPONENTS} components...")
        X_numeric = X.select_dtypes(include=np.number).fillna(0)
        
        pca_pipeline = Pipeline([
            ('scaler', StandardScaler()),
            ('pca', PCA(n_components=self.config.PCA_N_COMPONENTS, random_state=42))
        ])
        
        X_transformed = pca_pipeline.fit_transform(X_numeric)
        
        pca_feature_names = [f'PC_{i+1}' for i in range(X_transformed.shape[1])]
        X_final = pd.DataFrame(X_transformed, columns=pca_feature_names, index=X.index)
        
        logger.info(f"    - PCA complete. Explained variance: {pca_pipeline.named_steps['pca'].explained_variance_ratio_.sum():.2%}")
        return X_final, pca_feature_names, pca_pipeline

    def _remove_redundant_features(self, df: pd.DataFrame, threshold: float = 0.95) -> List[str]:
        """
        Prunes highly correlated features.
        Returns a list that excludes non-numeric columns that were never processed.
        """
        logger.info(f"    - Sub-step: Pruning features with Spearman correlation > {threshold}...")
        all_original_cols = df.columns.tolist()
        df_numeric = df.select_dtypes(include=np.number)
        
        if df_numeric.empty:
            logger.warning("    - No numeric features found for correlation pruning.")
            return [col for col in all_original_cols if col not in df.select_dtypes(include=np.number).columns]

        corr_matrix = df_numeric.corr(method='spearman').abs()
        upper_tri = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
        to_drop = {column for column in upper_tri.columns if any(upper_tri[column] > threshold)}
        
        if to_drop:
            logger.warning(f"    - Correlation Pruning: Removing {len(to_drop)} numeric feature(s).")
            
        # Returns the list of numeric columns that survived, plus all non-numeric columns
        numeric_kept = [col for col in df_numeric.columns if col not in to_drop]
        return numeric_kept

    def _select_features_with_trex(self, X: pd.DataFrame, y: pd.Series) -> List[str]:
        logger.info("    - Selecting features using TRexSelector...")
        X_clean = X.select_dtypes(include=np.number).copy().fillna(X.median(numeric_only=True))
        y_binary = y.apply(lambda x: 0 if x == 1 else 1)
        variances = X_clean.var()
        features_to_keep = variances[variances > 1e-6].index.tolist()
        X_variant = X_clean[features_to_keep]
        if X_variant.shape[1] < X_clean.shape[1]: logger.warning(f"    - TRex: Removed {X_clean.shape[1] - X_variant.shape[1]} constant features.")
        if X_variant.shape[1] < 2: return X.columns.tolist()[:15]
        try:
            res = trex(X=X_variant.values, y=y_binary.values, tFDR=0.2, verbose=False)
            selected_features = X_variant.columns[list(res.get("selected_var", []))].tolist()
            logger.info(f"    - TRexSelector selected {len(selected_features)} features.")
            return selected_features if selected_features else X_variant.columns.tolist()[:15]
        except Exception as e:
            logger.error(f"    - TRexSelector failed: {e}. Returning pre-variance check features as fallback.")
            return features_to_keep

    def _select_elite_features_mi(self, X: pd.DataFrame, y: pd.Series, top_n: int = 30) -> List[str]:
        logger.info(f"    - Selecting top {top_n} features using Mutual Information...")
        X_clean = X.select_dtypes(include=np.number).copy().fillna(X.median(numeric_only=True))
        variances = X_clean.var()
        non_constant_cols = variances[variances > 1e-6].index.tolist()
        X_final_numeric = X_clean[non_constant_cols]
        if X_final_numeric.empty:
            logger.error("    - No non-constant numeric features for MI selection.")
            return X.select_dtypes(include=np.number).columns.tolist()[:top_n]
        mi_scores = mutual_info_classif(X_final_numeric, y, random_state=42)
        mi_series = pd.Series(mi_scores, index=X_final_numeric.columns).sort_values(ascending=False)
        return mi_series.head(top_n).index.tolist()

    def _optimize_hyperparameters(self, X: pd.DataFrame, y: pd.Series, num_classes: int) -> Optional[optuna.study.Study]:
        state_rules = self.config.STATE_BASED_CONFIG[self.config.operating_state]
        optimization_objective_names = state_rules.get("optimization_objective", ["maximize_f1", "maximize_log_trades"])
        logger.info(f"    - Optuna Study Objectives: {optimization_objective_names}")
        obj1_label = optimization_objective_names[0].replace('_', ' ').title()
        obj2_label = optimization_objective_names[1].replace('_', ' ').title()

        def dynamic_progress_callback(study: optuna.study.Study, trial: optuna.trial.FrozenTrial):
            progress_str = f"> Optuna Optimization: Trial {trial.number + 1}/{self.config.OPTUNA_TRIALS}"
            if study.best_trials:
                best_values = study.best_trials[0].values
                if best_values and len(best_values) >= 2:
                    progress_str += f" | Best Trial -> {obj1_label}: {best_values[0]:.4f}, {obj2_label}: {best_values[1]:.4f}"
            sys.stdout.write(f"\r{progress_str}\x1b[K"); sys.stdout.flush()
        
        def custom_objective(trial: optuna.Trial) -> Tuple[float, float]:
            params = {
                'objective': self.objective, 'eval_metric': self.eval_metric, 'seed': 42,
                'n_estimators': trial.suggest_int('n_estimators', 100, 800, step=50),
                'max_depth': trial.suggest_int('max_depth', 3, 7),
                'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.2, log=True),
                'subsample': trial.suggest_float('subsample', 0.6, 1.0),
                'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
                'gamma': trial.suggest_float('gamma', 0, 5)
            }
            if num_classes > 2: params['num_class'] = num_classes
            
            model = xgb.XGBClassifier(**params)
            skf = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
            fold_scores = defaultdict(list)
            
            for train_idx, val_idx in skf.split(X, y):
                X_train, X_val, y_train, y_val = X.iloc[train_idx], X.iloc[val_idx], y.iloc[train_idx], y.iloc[val_idx]
                
                # Applies the calculated class weights during fitting
                fit_params = {
                    'sample_weight': y_train.map(self.class_weights)
                }
                model.fit(X_train, y_train, **fit_params, verbose=False)
                
                preds = model.predict(X_val)
                f1 = f1_score(y_val, preds, average='weighted', labels=[l for l in [0, 2] if l in y.unique()], zero_division=0)
                activity = np.log1p(np.sum(preds != 1))
                fold_scores["f1"].append(f1)
                fold_scores["activity"].append(activity)

            obj1, obj2 = 0.0, 0.0
            if 'f1' in optimization_objective_names[0]:
                obj1 = np.mean(fold_scores["f1"])
            
            if 'log_trades' in optimization_objective_names[1]:
                obj2 = np.mean(fold_scores["activity"])
                if 'minimize' in optimization_objective_names[1]:
                    obj2 = -obj2
            
            return obj1, obj2

        try:
            study = optuna.create_study(directions=['maximize', 'maximize'])
            study.optimize(custom_objective, n_trials=self.config.OPTUNA_TRIALS, timeout=3600, n_jobs=-1, callbacks=[dynamic_progress_callback])
            sys.stdout.write("\n")
            return study
        except Exception as e:
            sys.stdout.write("\n")
            logger.error(f"    - Optuna study failed: {e}", exc_info=True)
            return None

    def _select_best_trial_from_study(self, study: optuna.study.Study, task_name: str) -> optuna.trial.FrozenTrial:
        if not study.best_trials: raise ValueError("Cannot select from an empty study.")
        logger.info(f"    - Selecting best trial for {task_name} from {len(study.best_trials)} non-dominated trial(s).")
        if len(study.best_trials) == 1: return study.best_trials[0]
        best_trial = None
        try:
            selected_trial_number = self.gemini_analyzer.select_best_tradeoff(
                study.best_trials, self.config.RISK_PROFILE, self.config.analysis_notes
            )
            best_trial = next((t for t in study.best_trials if t.number == selected_trial_number), None)
        except Exception as e:
            logger.error(f"AI-based trial selection failed: {e}. Falling back.")
        if not best_trial:
            best_trial = max(study.best_trials, key=lambda t: t.values[0])
            logger.info(f"    - Falling back to trial with best primary objective: Trial #{best_trial.number}")
        return best_trial

    def _find_best_threshold(self, best_params, X, y, num_classes) -> Tuple[float, float]:
        logger.info("    - Tuning classification threshold for F1 score...")
        X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.25, stratify=y if y.nunique() > 1 else None, random_state=42)
        temp_params = {'objective': self.objective, **best_params}
        if num_classes > 2: temp_params['num_class'] = num_classes
        temp_pipeline = Pipeline([('scaler', RobustScaler()), ('model', xgb.XGBClassifier(**temp_params))])
        fit_params = {'model__sample_weight': y_train.map(self.class_weights)}
        temp_pipeline.fit(X_train, y_train, **fit_params)
        probs = temp_pipeline.predict_proba(X_val)
        best_f1, best_thresh, best_preds = -1.0, 0.5, None
        for confidence_gate in np.arange(0.33, 0.96, 0.05):
            preds = np.argmax(probs, axis=1)
            preds[np.max(probs, axis=1) < confidence_gate] = 1 
            f1 = f1_score(y_val, preds, average='weighted', zero_division=0)
            if f1 > best_f1: best_f1, best_thresh, best_preds = f1, confidence_gate, preds
        if best_preds is not None:
            self.classification_report_str = classification_report(y_val, best_preds, zero_division=0)
        return best_thresh, best_f1

    def _train_final_model(self, best_params: Dict, X_input: pd.DataFrame, y_input: pd.Series, final_feature_names: List[str], num_classes: int, pre_fitted_transformer: Pipeline = None) -> Optional[Pipeline]:
        logger.info(f"    - Training final model on all data using {len(final_feature_names)} features/components...")
        try:
            final_params = {'objective': self.objective, **best_params, 'random_state': 42}
            if num_classes > 2: final_params['num_class'] = num_classes
            
            model = xgb.XGBClassifier(**final_params)
            
            steps = []
            # Ensure the input to the pipeline is numeric
            X_for_fitting = X_input.select_dtypes(include=np.number).fillna(0)

            if pre_fitted_transformer:
                steps.append(('transformer', pre_fitted_transformer))
            else:
                steps.append(('scaler', RobustScaler()))
            
            steps.append(('model', model))
            final_pipeline = Pipeline(steps)
            
            fit_params = {'model__sample_weight': y_input.map(self.class_weights)}
            final_pipeline.fit(X_for_fitting, y_input, **fit_params)

            if self.config.CALCULATE_SHAP_VALUES:
                if 'transformer' in final_pipeline.named_steps:
                    X_for_shap = final_pipeline.named_steps['transformer'].transform(X_for_fitting)
                else:
                    X_for_shap = final_pipeline.named_steps['scaler'].transform(X_for_fitting)
                self._generate_shap_summary(final_pipeline.named_steps['model'], X_for_shap, final_feature_names)

            return final_pipeline
        except Exception as e:
            logger.error(f"    - Error during final model training: {e}", exc_info=True)
            return None

    def _generate_shap_summary(self, model, X_scaled, feature_names):
        logger.info("    - Generating SHAP feature importance summary...")
        try:
            X_sample = shap.utils.sample(X_scaled, 2000) if len(X_scaled) > 2000 else X_scaled
            explainer = shap.Explainer(model, X_sample)
            shap_values = explainer(X_sample)
            mean_abs_shap = np.abs(shap_values.values).mean(axis=0)
            if mean_abs_shap.ndim > 1: mean_abs_shap = mean_abs_shap.mean(axis=1)
            if len(mean_abs_shap) != len(feature_names):
                logger.error(f"SHAP shape mismatch: {len(mean_abs_shap)} values vs {len(feature_names)} features.")
                return
            self.shap_summary = pd.DataFrame({'SHAP_Importance': mean_abs_shap}, index=feature_names).sort_values('SHAP_Importance', ascending=False)
            logger.info("    - SHAP summary generated.")
        except Exception as e:
            logger.error(f"    - Failed to generate SHAP summary: {e}", exc_info=True)
            self.shap_summary = None

# =============================================================================
# 7. BACKTESTER & 8. PERFORMANCE ANALYZER
# =============================================================================

class Backtester:

    def __init__(self, config: 'ConfigModel'):
        self.config = config
        self.use_tp_ladder = self.config.USE_TP_LADDER
        if self.use_tp_ladder:
            if len(self.config.TP_LADDER_LEVELS_PCT) != len(self.config.TP_LADDER_RISK_MULTIPLIERS):
                logger.error("TP Ladder config error: Lengths of PCT and Multipliers do not match. Disabling.")
                self.use_tp_ladder = False
            elif not np.isclose(sum(self.config.TP_LADDER_LEVELS_PCT), 1.0):
                logger.error(f"TP Ladder config error: 'TP_LADDER_LEVELS_PCT' sum ({sum(self.config.TP_LADDER_LEVELS_PCT)}) is not 1.0. Disabling.")
                self.use_tp_ladder = False
            else:
                logger.info("Take-Profit Ladder is ENABLED.")
        else:
            logger.info("Take-Profit Ladder is DISABLED (or was disabled due to config error).")


    def _get_tiered_risk_params(self, equity: float) -> Tuple[float, int]:
        """Looks up risk percentage and max trades from the tiered config."""
        sorted_tiers = sorted(self.config.TIERED_RISK_CONFIG.keys())
        for tier_cap in sorted_tiers:
            if equity <= tier_cap:
                profile_tier_config = self.config.TIERED_RISK_CONFIG[tier_cap]
                profile_settings = profile_tier_config.get(self.config.RISK_PROFILE, profile_tier_config.get('Medium'))
                if profile_settings is None: # Should not happen if 'Medium' is always defined
                    logger.error(f"Tiered risk profile '{self.config.RISK_PROFILE}' or 'Medium' not found for tier {tier_cap}. Using global defaults.")
                    return self.config.BASE_RISK_PER_TRADE_PCT, self.config.MAX_CONCURRENT_TRADES
                return profile_settings['risk_pct'], profile_settings['pairs']

        # If equity is above the highest tier cap, use the highest tier's settings
        highest_tier_key = sorted_tiers[-1]
        highest_tier_config = self.config.TIERED_RISK_CONFIG[highest_tier_key]
        profile_settings = highest_tier_config.get(self.config.RISK_PROFILE, highest_tier_config.get('Medium'))
        if profile_settings is None:
            logger.error(f"Tiered risk profile '{self.config.RISK_PROFILE}' or 'Medium' not found for highest tier {highest_tier_key}. Using global defaults.")
            return self.config.BASE_RISK_PER_TRADE_PCT, self.config.MAX_CONCURRENT_TRADES
        return profile_settings['risk_pct'], profile_settings['pairs']

    def _calculate_realistic_costs(self, candle: Dict, on_exit: bool = False) -> Tuple[float, float]:
        """Calculates dynamic spread and variable slippage (returns cost in price units)."""
        symbol = candle.get('Symbol', 'Unknown')
        instrument_price = candle.get('Close', 1.0)

        # Point size determination
        is_jpy_pair = 'JPY' in symbol.upper()
        is_low_price_instrument = instrument_price < 10.0 and not is_jpy_pair

        if is_jpy_pair: point_size = 0.01
        elif is_low_price_instrument: point_size = 0.0001
        else:
            if "XAU" in symbol.upper() or "SIL" in symbol.upper(): point_size = 0.01
            elif any(idx_str in symbol.upper() for idx_str in ["US30", "NDX100", "SPX500", "GER30", "UK100"]): point_size = 1.0
            else: point_size = 0.0001

        spread_cost_currency = 0.0
        if not on_exit:
            spread_info = self.config.SPREAD_CONFIG.get(symbol)
            if spread_info is None:
                spread_info = self.config.SPREAD_CONFIG.get('default', {'normal_pips': 2.0, 'volatile_pips': 6.0})
            vol_rank = candle.get('market_volatility_index', 0.5)
            spread_pips = spread_info['volatile_pips'] if vol_rank > 0.8 else spread_info['normal_pips']
            spread_cost_currency = spread_pips * point_size

        slippage_cost_currency = 0.0
        if self.config.USE_VARIABLE_SLIPPAGE:
            atr = candle.get('ATR', 0.0)
            if pd.isna(atr) or atr <= 1e-9:
                atr = instrument_price * 0.0005
            vol_rank = candle.get('market_volatility_index', 0.5)
            random_factor = random.uniform(0.1, 1.2 if on_exit else 1.0)
            slippage_cost_currency = atr * vol_rank * random_factor * self.config.SLIPPAGE_VOLATILITY_FACTOR

        return spread_cost_currency, slippage_cost_currency

    def _calculate_latency_cost(self, signal_candle: Dict, exec_candle: Dict) -> float:
        """Calculates a randomized, volatility-based cost (in price units) to simulate execution latency."""
        if not self.config.SIMULATE_LATENCY: return 0.0

        atr = signal_candle.get('ATR')
        if pd.isna(atr) or atr is None or atr <= 1e-9: return 0.0

        sig_ts = pd.to_datetime(signal_candle['Timestamp'])
        exec_ts = pd.to_datetime(exec_candle['Timestamp'])

        bar_duration_sec = (exec_ts - sig_ts).total_seconds()
        if bar_duration_sec <= 0: return 0.0

        min_delay_ms = 50
        max_delay_ms = self.config.EXECUTION_LATENCY_MS
        simulated_delay_sec = random.uniform(min_delay_ms, max_delay_ms) / 1000.0

        delay_ratio = min(simulated_delay_sec / bar_duration_sec, 1.0)
        latency_cost_price_units = atr * delay_ratio
        return latency_cost_price_units

    @staticmethod
    def get_last_run_params(log_file: str) -> dict | None:
        if not pathlib.Path(log_file).exists(): return None
        with open(log_file, 'r') as f:
            try: return json.load(f)[-1]
            except (json.JSONDecodeError, IndexError): return None

    @staticmethod
    def log_run_params(params: dict, log_file: str):
        all_params = []
        if pathlib.Path(log_file).exists():
            with open(log_file, 'r') as f:
                try: all_params = json.load(f)
                except json.JSONDecodeError: pass

        sanitized_params = _sanitize_keys_for_json(params)
        all_params.append(sanitized_params)

        with open(log_file, 'w') as f:
            json.dump(all_params, f, indent=4, default=json_serializer_default)

    @staticmethod
    def calculate_parameter_drift(suggested: dict, previous: dict) -> float:
        numeric_params = [k for k, v in suggested.items() if isinstance(v, (int, float))]
        if not numeric_params: return 0.0
        total_drift = 0.0
        for key in numeric_params:
            if key in previous and isinstance(previous.get(key), (int, float)) and previous[key] != 0:
                total_drift += abs((suggested[key] - previous[key]) / previous[key]) * 100
        return total_drift / len(numeric_params)

    @staticmethod
    def run_validation_on_holdout_set(params: dict, holdout_data: pd.DataFrame, config: ConfigModel, model_trainer: 'ModelTrainer', symbol: str, playbook: dict) -> dict:
        print(f"\nGUARDRAIL: Running validation on secret holdout set ({len(holdout_data)} bars)...")
        if holdout_data.empty:
            print("GUARDRAIL: Holdout data is empty. Cannot validate.")
            return {'sharpe_ratio': -999}

        split_point = int(len(holdout_data) * 0.7)
        guardrail_train_data = holdout_data.iloc[:split_point]
        guardrail_val_data = holdout_data.iloc[split_point:]

        if guardrail_train_data.empty or guardrail_val_data.empty:
            logger.warning("  - Not enough data in holdout set for guardrail validation split. Skipping.")
            return {"sharpe_ratio": 0.0, "total_pnl": 0.0, "max_drawdown": 0.0, "num_trades": 0}

        fe_validator = FeatureEngineer(config, {}, playbook)
        labeled_guardrail_train_data = fe_validator.label_signal_pressure(
            guardrail_train_data, config.LOOKAHEAD_CANDLES, config.LABEL_LONG_QUANTILE, config.LABEL_SHORT_QUANTILE
        )

        if 'target' not in labeled_guardrail_train_data.columns:
            logger.error("GUARDRAIL: Label generation failed for the validation set. Cannot proceed.")
            return {'sharpe_ratio': -999}

        original_config_params = config.model_dump()
        for key, value in params.items():
            if hasattr(config, key):
                setattr(config, key, value)

        feature_list = _get_available_features_from_df(labeled_guardrail_train_data)
        train_result = model_trainer.train(labeled_guardrail_train_data, feature_list)

        for key, value in original_config_params.items():
            setattr(config, key, value)

        if not train_result:
            logger.error("GUARDRAIL: Model training failed for the validation set. Cannot backtest.")
            return {'sharpe_ratio': -999}

        pipeline, threshold = train_result
        backtester = Backtester(config)
        trades_df, equity_curve, _, _, _ = backtester.run_backtest_chunk(
            guardrail_val_data, pipeline, config.INITIAL_CAPITAL, feature_list, confidence_threshold=threshold
        )
        analyzer = PerformanceAnalyzer(config)
        backtest_results = analyzer._calculate_metrics(trades_df, equity_curve)

        pnl = backtest_results.get('total_net_profit', 0)
        sharpe = backtest_results.get('sharpe_ratio', 0)
        logger.info(f"  - Holdout validation complete. PnL: {pnl:.2f}, Sharpe: {sharpe:.2f}")
        return backtest_results

    @staticmethod
    def validate_ai_suggestion(suggested_params: dict, historical_performance_log: list, holdout_data: pd.DataFrame, config: ConfigModel, model_trainer: 'ModelTrainer', symbol: str, playbook: dict) -> bool:
        print("\n" + "="*50)
        print("AI SUGGESTION VALIDATION PROTOCOL INITIATED")
        print("="*50)
        last_run_params = Backtester.get_last_run_params(config.PARAMS_LOG_FILE)
        if last_run_params:
            param_drift = Backtester.calculate_parameter_drift(suggested_params, last_run_params)
            print(f"GUARDRAIL CHECK 1: Parameter drift is {param_drift:.2f}%. (Tolerance: {config.MAX_PARAM_DRIFT_TOLERANCE}%)")
            if param_drift > config.MAX_PARAM_DRIFT_TOLERANCE:
                print(">>> GUARDRAIL FAILED: AI suggested overly drastic parameter change. REJECTING.")
                return False
        print(f"GUARDRAIL CHECK 2: Performance history contains {len(historical_performance_log)} cycles. (Minimum required: {config.MIN_CYCLES_FOR_ADAPTATION})")
        if len(historical_performance_log) > 0 and len(historical_performance_log) < config.MIN_CYCLES_FOR_ADAPTATION:
            print(f">>> GUARDRAIL FAILED: Not enough performance history to justify strategic change. REJECTING.")
            return False
        print("GUARDRAIL CHECK 3: Testing suggestion on unseen holdout data.")
        holdout_performance = Backtester.run_validation_on_holdout_set(suggested_params, holdout_data, config, model_trainer, symbol, playbook)
        holdout_sharpe = holdout_performance.get('sharpe_ratio', -999)
        print(f"Holdout set Sharpe Ratio is {holdout_sharpe:.3f}. (Minimum required: {config.MIN_HOLDOUT_SHARPE})")
        if holdout_sharpe < config.MIN_HOLDOUT_SHARPE:
            print(">>> GUARDRAIL FAILED: Suggested strategy performed poorly on secret holdout validation set. REJECTING.")
            return False
        print("\n" + "="*50)
        print(">>> GUARDRAIL PASSED: AI suggestion is validated and will be implemented.")
        print("="*50)
        Backtester.log_run_params(config.model_dump(), config.PARAMS_LOG_FILE)
        return True

    def run_backtest_chunk(self, df_chunk_in: pd.DataFrame, pipelines: Union[Pipeline, Dict[str, Any]], initial_equity: float, feature_list: List[str], confidence_threshold: float, trade_lockout_until: Optional[pd.Timestamp] = None, is_minirocket_model: bool = False) -> Tuple[pd.DataFrame, pd.Series, bool, Optional[Dict], Dict]:
        if df_chunk_in.empty or not pipelines or not feature_list:
            logger.warning("Backtest chunk, pipelines, or feature_list is empty. Returning initial state.")
            return pd.DataFrame(), pd.Series([initial_equity] if initial_equity is not None else [], dtype=float), False, None, {}

        df_chunk = df_chunk_in.copy()
        trades_log: List[Dict] = []
        equity: float = initial_equity
        equity_curve_values: List[float] = [initial_equity]
        open_positions: Dict[str, Dict] = {}
        chunk_peak_equity: float = initial_equity
        circuit_breaker_tripped: bool = False
        breaker_details: Optional[Dict] = None
        last_trade_pnl_val: float = 0.0
        daily_metrics_report: Dict[str, Dict] = {}
        current_trading_day: Optional[date] = None
        day_start_equity_val: float = initial_equity
        day_peak_equity_val: float = initial_equity

        lookback_window = 100

        def finalize_daily_metrics_helper(day_obj: Optional[date], equity_at_day_close: float):
            nonlocal day_start_equity_val, day_peak_equity_val
            if day_obj is None: return
            daily_pnl = equity_at_day_close - day_start_equity_val
            daily_dd_val = (day_peak_equity_val - equity_at_day_close) / day_peak_equity_val if day_peak_equity_val > 0 else 0.0
            daily_metrics_report[day_obj.isoformat()] = {'pnl': round(daily_pnl, 2), 'drawdown_pct': round(daily_dd_val * 100, 2)}

        def close_trade_helper(pos_details: Dict, exit_price_val: float, exit_reason_str: str, candle_at_exit: Dict, is_ambiguous_exit: bool = False):
            nonlocal equity, last_trade_pnl_val
            pnl_gross = (exit_price_val - pos_details['entry_price']) * pos_details['direction'] * pos_details['lot_size'] * self.config.CONTRACT_SIZE
            commission_cost = self.config.COMMISSION_PER_LOT * pos_details['lot_size'] * 2
            _, exit_slippage_price_units = self._calculate_realistic_costs(candle_at_exit, on_exit=True)
            slippage_cost_currency = exit_slippage_price_units * pos_details['lot_size'] * self.config.CONTRACT_SIZE
            pnl_net = pnl_gross - commission_cost - slippage_cost_currency
            equity += pnl_net
            last_trade_pnl_val = pnl_net
            mae_price_units = abs(pos_details['mae_price'] - pos_details['entry_price'])
            mfe_price_units = abs(pos_details['mfe_price'] - pos_details['entry_price'])
            trades_log.append({
                'ExecTime': candle_at_exit['Timestamp'], 'Symbol': pos_details['symbol'], 'PNL': pnl_net,
                'Equity': equity, 'Confidence': pos_details.get('confidence', 0.0), 'Direction': pos_details['direction'],
                'ExitReason': exit_reason_str, 'MAE_Points': round(mae_price_units, 5),
                'MFE_Points': round(mfe_price_units, 5), 'AmbiguousExit': is_ambiguous_exit
            })
            equity_curve_values.append(equity)
            return pnl_net

        candles_as_dicts = df_chunk.reset_index().to_dict('records')
        start_index = lookback_window if is_minirocket_model else 1

        for i in range(start_index, len(candles_as_dicts)):
            current_candle, prev_candle = candles_as_dicts[i], candles_as_dicts[i-1]
            current_ts = pd.to_datetime(current_candle['Timestamp'])

            if current_ts.date() != current_trading_day:
                if current_trading_day is not None: finalize_daily_metrics_helper(current_trading_day, equity_curve_values[-1])
                current_trading_day = current_ts.date()
                day_start_equity_val = equity_curve_values[-1] if equity_curve_values else initial_equity
                day_peak_equity_val = day_start_equity_val

            if equity <= 0:
                logger.critical(f"Equity is zero or negative at {current_ts}. Stopping backtest chunk.")
                break

            if not circuit_breaker_tripped:
                day_peak_equity_val = max(day_peak_equity_val, equity)
                chunk_peak_equity = max(chunk_peak_equity, equity)
                if (chunk_peak_equity - equity) / chunk_peak_equity > self.config.MAX_DD_PER_CYCLE:
                    circuit_breaker_tripped = True
                    breaker_details = {"num_trades_before_trip": len(trades_log), "equity_at_trip": equity}
                    logger.warning(f"CIRCUIT BREAKER TRIPPED at {current_ts}!")
                    for sym_key in list(open_positions.keys()):
                        close_trade_helper(open_positions[sym_key], current_candle['Open'], "Circuit Breaker", current_candle, is_ambiguous_exit=False)
                        del open_positions[sym_key]
                    continue

            for sym_key, pos in list(open_positions.items()):
                high, low, sl, tp = current_candle['High'], current_candle['Low'], pos['sl'], pos['tp']
                sl_hit = (low <= sl) if pos['direction'] == 1 else (high >= sl)
                tp_hit = (high >= tp) if pos['direction'] == 1 else (low <= tp)
                if sl_hit or tp_hit:
                    exit_price, reason, ambiguous = (sl, "Stop Loss", sl_hit and tp_hit) if sl_hit else (tp, "Take Profit", False)
                    close_trade_helper(pos, exit_price, reason, current_candle, is_ambiguous_exit=ambiguous)
                    del open_positions[sym_key]

            symbol = prev_candle['Symbol']
            risk_pct, max_trades = self._get_tiered_risk_params(equity) if self.config.USE_TIERED_RISK else (self.config.BASE_RISK_PER_TRADE_PCT, self.config.MAX_CONCURRENT_TRADES)
            is_locked_out_by_time = trade_lockout_until is not None and current_ts < trade_lockout_until

            if not circuit_breaker_tripped and not is_locked_out_by_time and symbol not in open_positions and len(open_positions) < max_trades:
                if prev_candle.get('anomaly_score', 1) == -1: continue
                vol_rank = prev_candle.get('market_volatility_index', 0.5)
                if not (self.config.MIN_VOLATILITY_RANK <= vol_rank <= self.config.MAX_VOLATILITY_RANK): continue

                if is_minirocket_model:
                    window_df = df_chunk_in.iloc[i - lookback_window : i]
                    pred_input = np.transpose(window_df[feature_list].values, (1, 0))[np.newaxis, :, :]
                    predicted_probabilities = pipelines.predict_proba(pred_input)[0]
                else:
                    features_df = pd.DataFrame([{f: prev_candle.get(f) for f in feature_list}]).fillna(0)
                    predicted_probabilities = pipelines.predict_proba(features_df)[0]

                confidence = np.max(predicted_probabilities)
                if confidence >= confidence_threshold:
                    prediction = np.argmax(predicted_probabilities)
                    direction = {0: -1, 2: 1}.get(prediction, 0)
                    if direction != 0:
                        atr = prev_candle.get('ATR', 0.0)
                        if pd.isna(atr) or atr <= 1e-9: continue
                        sl_dist = atr * self.config.SL_ATR_MULTIPLIER
                        risk_usd = equity * risk_pct
                        spread, slippage = self._calculate_realistic_costs(prev_candle)
                        latency_cost = self._calculate_latency_cost(prev_candle, current_candle)
                        entry_price = current_candle['Open'] + (spread + slippage + latency_cost) * direction
                        sl_price = entry_price - sl_dist * direction
                        tp_price = entry_price + sl_dist * self.config.TP_ATR_MULTIPLIER * direction

                        is_jpy = 'JPY' in symbol.upper()
                        point_size = 0.01 if is_jpy else 0.0001
                        value_per_point_lot = self.config.CONTRACT_SIZE * point_size
                        risk_per_lot = sl_dist * value_per_point_lot
                        if risk_per_lot <= 1e-9: continue

                        lot_size = round((risk_usd / risk_per_lot) / self.config.LOT_STEP) * self.config.LOT_STEP
                        lot_size = max(lot_size, self.config.MIN_LOT_SIZE)

                        margin_needed = (lot_size * self.config.CONTRACT_SIZE * entry_price) / self.config.LEVERAGE
                        if equity - sum(p.get('margin_used', 0) for p in open_positions.values()) < margin_needed: continue

                        open_positions[symbol] = {'symbol': symbol, 'direction': direction, 'entry_price': entry_price, 'sl': sl_price, 'tp': tp_price,
                                                  'confidence': confidence, 'lot_size': lot_size, 'margin_used': margin_needed,
                                                  'mfe_price': entry_price, 'mae_price': entry_price, 'entry_candle_timestamp': current_ts}
                        logger.info(f"Opened {'Long' if direction == 1 else 'Short'} for {symbol} at {entry_price:.5f}")

        if current_trading_day is not None:
            finalize_daily_metrics_helper(current_trading_day, equity_curve_values[-1] if equity_curve_values else initial_equity)

        return pd.DataFrame(trades_log), pd.Series(equity_curve_values, dtype=float), circuit_breaker_tripped, breaker_details, daily_metrics_report
        
class PerformanceAnalyzer:
    def __init__(self,config:ConfigModel):
        self.config=config

    def generate_full_report(self,trades_df:Optional[pd.DataFrame],equity_curve:Optional[pd.Series],cycle_metrics:List[Dict],aggregated_shap:Optional[pd.DataFrame]=None, framework_memory:Optional[Dict]=None, aggregated_daily_dd:Optional[List[Dict]]=None, last_classification_report:str="N/A") -> Dict[str, Any]: # MODIFIED
        logger.info("-> Stage 4: Generating Final Performance Report...")
        if equity_curve is not None and len(equity_curve) > 1: self.plot_equity_curve(equity_curve)
        
        # Aggregated_shap is just the single DataFrame from the trainer
        if aggregated_shap is not None and not aggregated_shap.empty:
             self.plot_shap_summary(aggregated_shap)

        metrics = self._calculate_metrics(trades_df, equity_curve) if trades_df is not None and not trades_df.empty else {}
        self.generate_text_report(metrics, cycle_metrics, aggregated_shap, framework_memory, aggregated_daily_dd, last_classification_report) # MODIFIED

        logger.info(f"[SUCCESS] Final report generated and saved to: {self.config.REPORT_SAVE_PATH}")
        return metrics

    def plot_equity_curve(self,equity_curve:pd.Series):
        plt.style.use('seaborn-v0_8-darkgrid')
        plt.figure(figsize=(16,8))
        plt.plot(equity_curve.values,color='dodgerblue',linewidth=2)
        plt.title(f"{self.config.nickname or self.config.REPORT_LABEL} - Walk-Forward Equity Curve",fontsize=16,weight='bold')
        plt.xlabel("Trade Event Number (including partial closes)",fontsize=12)
        plt.ylabel("Equity ($)",fontsize=12)
        plt.grid(True,which='both',linestyle=':')
        try:
            plt.savefig(self.config.PLOT_SAVE_PATH)
            plt.close()
            logger.info(f"  - Equity curve plot saved to: {self.config.PLOT_SAVE_PATH}")
        except Exception as e:
            logger.error(f"  - Failed to save equity curve plot: {e}")

    def plot_shap_summary(self,shap_summary:pd.DataFrame):
        plt.style.use('seaborn-v0_8-darkgrid')
        plt.figure(figsize=(12,10))
        shap_summary.head(20).sort_values(by='SHAP_Importance').plot(kind='barh',legend=False,color='mediumseagreen')
        title_str = f"{self.config.nickname or self.config.REPORT_LABEL} ({self.config.strategy_name}) - Aggregated Feature Importance"
        plt.title(title_str,fontsize=16,weight='bold')
        plt.xlabel("Mean Absolute SHAP Value",fontsize=12)
        plt.ylabel("Feature",fontsize=12)
        plt.tight_layout()
        try:
            # Construct a full path for the final aggregated SHAP summary
            final_shap_path = os.path.join(self.config.SHAP_PLOT_PATH, f"{self.config.run_timestamp}_aggregated_shap_summary.png")
            plt.savefig(final_shap_path)
            plt.close()
            logger.info(f"  - SHAP summary plot saved to: {final_shap_path}")
        except Exception as e:
            logger.error(f"  - Failed to save SHAP plot: {e}")

    def _calculate_metrics(self,trades_df:pd.DataFrame,equity_curve:pd.Series)->Dict[str,Any]:
        m={}
        m['initial_capital']=self.config.INITIAL_CAPITAL
        m['ending_capital']=equity_curve.iloc[-1]
        m['total_net_profit']=m['ending_capital']-m['initial_capital']
        m['net_profit_pct']=(m['total_net_profit']/m['initial_capital']) if m['initial_capital']>0 else 0

        returns=trades_df['PNL']/m['initial_capital']
        wins=trades_df[trades_df['PNL']>0]
        losses=trades_df[trades_df['PNL']<0]
        m['gross_profit']=wins['PNL'].sum()
        m['gross_loss']=abs(losses['PNL'].sum())
        m['profit_factor']=m['gross_profit']/m['gross_loss'] if m['gross_loss']>0 else np.inf

        m['total_trade_events']=len(trades_df)
        final_exits_df = trades_df[trades_df['ExitReason'].str.contains("Stop Loss|Take Profit", na=False)]
        m['total_trades'] = len(final_exits_df)

        m['winning_trades']=len(final_exits_df[final_exits_df['PNL'] > 0])
        m['losing_trades']=len(final_exits_df[final_exits_df['PNL'] < 0])
        m['win_rate']=m['winning_trades']/m['total_trades'] if m['total_trades']>0 else 0

        m['avg_win_amount']=wins['PNL'].mean() if len(wins)>0 else 0
        m['avg_loss_amount']=abs(losses['PNL'].mean()) if len(losses)>0 else 0

        avg_full_win = final_exits_df[final_exits_df['PNL'] > 0]['PNL'].mean() if len(final_exits_df[final_exits_df['PNL'] > 0]) > 0 else 0
        avg_full_loss = abs(final_exits_df[final_exits_df['PNL'] < 0]['PNL'].mean()) if len(final_exits_df[final_exits_df['PNL'] < 0]) > 0 else 0
        m['payoff_ratio']=avg_full_win/avg_full_loss if avg_full_loss > 0 else np.inf
        m['expected_payoff']=(m['win_rate']*avg_full_win)-((1-m['win_rate'])*avg_full_loss) if m['total_trades']>0 else 0

        running_max=equity_curve.cummax()
        drawdown_abs=running_max-equity_curve
        m['max_drawdown_abs']=drawdown_abs.max() if not drawdown_abs.empty else 0
        m['max_drawdown_pct']=((drawdown_abs/running_max).replace([np.inf,-np.inf],0).max())*100

        exec_times=pd.to_datetime(trades_df['ExecTime']).dt.tz_localize(None)
        years=((exec_times.max()-exec_times.min()).days/365.25) if not trades_df.empty else 1
        years = max(years, 1/365.25)
        m['cagr']=(((m['ending_capital']/m['initial_capital'])**(1/years))-1) if years>0 and m['initial_capital']>0 else 0

        pnl_std=returns.std()
        m['sharpe_ratio']=(returns.mean()/pnl_std)*np.sqrt(252*24*4) if pnl_std>0 else 0
        downside_returns=returns[returns<0]
        downside_std=downside_returns.std()
        m['sortino_ratio']=(returns.mean()/downside_std)*np.sqrt(252*24*4) if downside_std>0 else np.inf
        m['calmar_ratio']=m['cagr']/(m['max_drawdown_pct']/100) if m['max_drawdown_pct']>0 else np.inf
        m['mar_ratio']=m['calmar_ratio']
        m['recovery_factor']=m['total_net_profit']/m['max_drawdown_abs'] if m['max_drawdown_abs']>0 else np.inf

        pnl_series = final_exits_df['PNL']
        win_streaks = (pnl_series > 0).astype(int).groupby((pnl_series <= 0).cumsum()).cumsum()
        loss_streaks = (pnl_series < 0).astype(int).groupby((pnl_series >= 0).cumsum()).cumsum()
        m['longest_win_streak'] = win_streaks.max() if not win_streaks.empty else 0
        m['longest_loss_streak'] = loss_streaks.max() if not loss_streaks.empty else 0
        return m

    def _get_comparison_block(self, metrics: Dict, memory: Dict, ledger: Dict, width: int) -> str:
        champion = memory.get('champion_config')
        historical_runs = memory.get('historical_runs', [])
        previous_run = historical_runs[-1] if historical_runs else None

        def get_data(source: Optional[Dict], key: str, is_percent: bool = False) -> str:
            if not source: return "N/A"
            val = source.get(key) if isinstance(source, dict) and key in source else source.get("final_metrics", {}).get(key) if isinstance(source, dict) else None
            if val is None or not isinstance(val, (int, float)): return "N/A"
            return f"{val:.2f}%" if is_percent else f"{val:.2f}"

        def get_info(source: Optional[Union[Dict, ConfigModel]], key: str) -> str:
            if not source: return "N/A"
            if hasattr(source, key):
                return str(getattr(source, key, 'N/A'))
            elif isinstance(source, dict):
                return str(source.get(key, 'N/A'))
            return "N/A"

        def get_nickname(source: Optional[Union[Dict, ConfigModel]]) -> str:
            if not source: return "N/A"
            version_key = 'REPORT_LABEL' if hasattr(source, 'REPORT_LABEL') else 'script_version'
            version = get_info(source, version_key)
            return ledger.get(version, "N/A")

        c_nick, p_nick, champ_nick = get_nickname(self.config), get_nickname(previous_run), get_nickname(champion)
        c_strat, p_strat, champ_strat = get_info(self.config, 'strategy_name'), get_info(previous_run, 'strategy_name'), get_info(champion, 'strategy_name')
        c_mar, p_mar, champ_mar = get_data(metrics, 'mar_ratio'), get_data(previous_run, 'mar_ratio'), get_data(champion, 'mar_ratio')
        c_mdd, p_mdd, champ_mdd = get_data(metrics, 'max_drawdown_pct', True), get_data(previous_run, 'max_drawdown_pct', True), get_data(champion, 'max_drawdown_pct', True)
        c_pf, p_pf, champ_pf = get_data(metrics, 'profit_factor'), get_data(previous_run, 'profit_factor'), get_data(champion, 'profit_factor')

        col_w = (width - 5) // 4
        header = f"| {'Metric'.ljust(col_w-1)}|{'Current Run'.center(col_w)}|{'Previous Run'.center(col_w)}|{'All-Time Champion'.center(col_w)}|"
        sep = f"+{'-'*(col_w)}+{'-'*(col_w)}+{'-'*(col_w)}+{'-'*(col_w)}+"
        rows = [
            f"| {'Run Nickname'.ljust(col_w-1)}|{c_nick.center(col_w)}|{p_nick.center(col_w)}|{champ_nick.center(col_w)}|",
            f"| {'Strategy'.ljust(col_w-1)}|{c_strat.center(col_w)}|{p_strat.center(col_w)}|{champ_strat.center(col_w)}|",
            f"| {'MAR Ratio'.ljust(col_w-1)}|{c_mar.center(col_w)}|{p_mar.center(col_w)}|{champ_mar.center(col_w)}|",
            f"| {'Max Drawdown'.ljust(col_w-1)}|{c_mdd.center(col_w)}|{p_mdd.center(col_w)}|{champ_mdd.center(col_w)}|",
            f"| {'Profit Factor'.ljust(col_w-1)}|{c_pf.center(col_w)}|{p_pf.center(col_w)}|{champ_pf.center(col_w)}|"
        ]
        return "\n".join([header, sep] + rows)

    def generate_text_report(self, m: Dict[str, Any], cycle_metrics: List[Dict], aggregated_shap: Optional[pd.DataFrame] = None, framework_memory: Optional[Dict] = None, aggregated_daily_dd: Optional[List[Dict]] = None, last_classification_report:str="N/A"): # MODIFIED
        WIDTH = 90
        def _box_top(w): return f"+{'-' * (w-2)}+"
        def _box_mid(w): return f"+{'-' * (w-2)}+"
        def _box_bot(w): return f"+{'-' * (w-2)}+"
        def _box_line(text, w):
            padding = w - 4 - len(text)
            return f"| {text}{' ' * padding} |" if padding >= 0 else f"| {text[:w-5]}... |"
        def _box_title(title, w): return f"| {title.center(w-4)} |"
        def _box_text_kv(key, val, w):
            val_str = str(val)
            key_len = len(key)
            padding = w - 4 - key_len - len(val_str)
            return f"| {key}{' ' * padding}{val_str} |"

        ledger = {};
        if self.config.NICKNAME_LEDGER_PATH and os.path.exists(self.config.NICKNAME_LEDGER_PATH):
            try:
                with open(self.config.NICKNAME_LEDGER_PATH, 'r') as f: ledger = json.load(f)
            except (json.JSONDecodeError, IOError): logger.warning("Could not load nickname ledger for reporting.")

        report = [_box_top(WIDTH)]
        report.append(_box_title('ADAPTIVE WALK-FORWARD PERFORMANCE REPORT', WIDTH))
        report.append(_box_mid(WIDTH))
        report.append(_box_line(f"Nickname: {self.config.nickname or 'N/A'} ({self.config.strategy_name})", WIDTH))
        report.append(_box_line(f"Version: {self.config.REPORT_LABEL}", WIDTH))
        report.append(_box_line(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", WIDTH))

        if self.config.analysis_notes:
            report.append(_box_line(f"AI Notes: {self.config.analysis_notes}", WIDTH))

        if framework_memory:
            report.append(_box_mid(WIDTH))
            report.append(_box_title('I. PERFORMANCE vs. HISTORY', WIDTH))
            report.append(_box_mid(WIDTH))
            report.append(self._get_comparison_block(m, framework_memory, ledger, WIDTH))

        sections = {
            "II. EXECUTIVE SUMMARY": [
                (f"Initial Capital:", f"${m.get('initial_capital', 0):>15,.2f}"),
                (f"Ending Capital:", f"${m.get('ending_capital', 0):>15,.2f}"),
                (f"Total Net Profit:", f"${m.get('total_net_profit', 0):>15,.2f} ({m.get('net_profit_pct', 0):.2%})"),
                (f"Profit Factor:", f"{m.get('profit_factor', 0):>15.2f}"),
                (f"Win Rate (Full Trades):", f"{m.get('win_rate', 0):>15.2%}"),
                (f"Expected Payoff:", f"${m.get('expected_payoff', 0):>15.2f}")
            ],
            "III. CORE PERFORMANCE METRICS": [
                (f"Annual Return (CAGR):", f"{m.get('cagr', 0):>15.2%}"),
                (f"Sharpe Ratio (annual):", f"${m.get('sharpe_ratio', 0):>15.2f}"),
                (f"Sortino Ratio (annual):", f"${m.get('sortino_ratio', 0):>15.2f}"),
                (f"Calmar Ratio / MAR:", f"${m.get('mar_ratio', 0):>15.2f}")
            ],
            "IV. RISK & DRAWDOWN ANALYSIS": [
                (f"Max Drawdown (Cycle):", f"{m.get('max_drawdown_pct', 0):>15.2f}% (${m.get('max_drawdown_abs', 0):,.2f})"),
                (f"Recovery Factor:", f"${m.get('recovery_factor', 0):>15.2f}"),
                (f"Longest Losing Streak:", f"{m.get('longest_loss_streak', 0):>15} trades")
            ],
            "V. TRADE-LEVEL STATISTICS": [
                (f"Total Unique Trades:", f"{m.get('total_trades', 0):>15}"),
                (f"Total Trade Events (incl. partials):", f"{m.get('total_trade_events', 0):>15}"),
                (f"Average Win Event:", f"${m.get('avg_win_amount', 0):>15,.2f}"),
                (f"Average Loss Event:", f"${m.get('avg_loss_amount', 0):>15,.2f}"),
                (f"Payoff Ratio (Full Trades):", f"${m.get('payoff_ratio', 0):>15.2f}")
            ]
        }
        for title, data in sections.items():
            if not m: continue
            report.append(_box_mid(WIDTH))
            report.append(_box_title(title, WIDTH))
            report.append(_box_mid(WIDTH))
            for key, val in data: report.append(_box_text_kv(key, val, WIDTH))

        report.append(_box_mid(WIDTH))
        report.append(_box_title('VI. WALK-FORWARD CYCLE BREAKDOWN', WIDTH))
        report.append(_box_mid(WIDTH))

        cycle_df = pd.DataFrame(cycle_metrics)
        if not cycle_df.empty:
            if 'BreakerContext' in cycle_df.columns:
                cycle_df['BreakerContext'] = cycle_df['BreakerContext'].apply(
                    lambda x: f"Trades: {x.get('num_trades_before_trip', 'N/A')}, PNL: {x.get('pnl_before_trip', 'N/A'):.2f}" if isinstance(x, dict) else ""
                ).fillna('')
            if 'trade_summary' in cycle_df.columns:
                cycle_df['MAE/MFE (Losses)'] = cycle_df['trade_summary'].apply(
                    lambda s: f"${s.get('avg_mae_loss',0):.2f}/${s.get('avg_mfe_loss',0):.2f}" if isinstance(s, dict) else "N/A"
                )
                cycle_df.drop(columns=['trade_summary'], inplace=True)

            cycle_df_str = cycle_df.to_string(index=False)
        else:
            cycle_df_str = "No trades were executed."

        for line in cycle_df_str.split('\n'): report.append(_box_line(line, WIDTH))

        report.append(_box_mid(WIDTH))
        report.append(_box_title('VII. MODEL FEATURE IMPORTANCE (TOP 15)', WIDTH))
        report.append(_box_mid(WIDTH))
        shap_str = aggregated_shap.head(15).to_string() if aggregated_shap is not None else "SHAP summary was not generated."
        for line in shap_str.split('\n'): report.append(_box_line(line, WIDTH))

        if aggregated_daily_dd:
            report.append(_box_mid(WIDTH))
            report.append(_box_title('VIII. HIGH DAILY DRAWDOWN EVENTS (>15%)', WIDTH))
            report.append(_box_mid(WIDTH))
            high_dd_events = []
            for cycle_idx, cycle_dd_report in enumerate(aggregated_daily_dd):
                for day, data in cycle_dd_report.items():
                    if data['drawdown_pct'] > 15.0:
                        high_dd_events.append(f"Cycle {cycle_idx+1} | {day} | DD: {data['drawdown_pct']:.2f}% | PNL: ${data['pnl']:,.2f}")

            if high_dd_events:
                for event in high_dd_events:
                    report.append(_box_line(event, WIDTH))
            else:
                report.append(_box_line("No days with drawdown greater than 15% were recorded.", WIDTH))

        report.append(_box_mid(WIDTH))
        report.append(_box_title('IX. PER-CLASS PERFORMANCE (LAST CYCLE VALIDATION)', WIDTH))
        report.append(_box_mid(WIDTH))
        if last_classification_report and last_classification_report != "N/A":
            for line in last_classification_report.split('\n'):
                report.append(_box_line(line, WIDTH))
        else:
            report.append(_box_line("No classification report was generated for the last cycle.", WIDTH))

        report.append(_box_bot(WIDTH))
        final_report = "\n".join(report)
        logger.info("\n" + final_report)
        try:
            with open(self.config.REPORT_SAVE_PATH,'w',encoding='utf-8') as f: f.write(final_report)
        except IOError as e: logger.error(f"  - Failed to save text report: {e}",exc_info=True)

def get_macro_context_data(
    tickers: Dict[str, str],
    period: str = "10y",
    results_dir: str = "Results"
) -> pd.DataFrame:
    """
    Fetches and intelligently caches a time series of data for key macroeconomic indicators.
    This version gracefully handles and reports failed downloads for individual tickers.
    """
    logger.info(f"-> Fetching/updating external macroeconomic time series for: {list(tickers.keys())}...")
    
    cache_dir = os.path.join(results_dir)
    os.makedirs(cache_dir, exist_ok=True)
    data_cache_path = os.path.join(cache_dir, "macro_data.parquet")
    metadata_cache_path = os.path.join(cache_dir, "macro_cache_metadata.json")

    # --- Full Download Logic (with improved error handling) ---
    logger.info("  - No valid cache found or cache is stale. Performing full download...")
    
    # yfinance automatically handles multiple tickers and reports errors
    all_data = yf.download(list(tickers.values()), period=period, progress=False, auto_adjust=True)
    
    if all_data.empty:
        logger.error("  - Failed to download any macro data whatsoever.")
        return pd.DataFrame()

    close_prices = all_data.get('Close', pd.DataFrame()) # Use .get() for safety
    if isinstance(close_prices, pd.Series):
        # If only one ticker was successful, it returns a Series
        close_prices = close_prices.to_frame(name=list(tickers.values())[0])

    # --- Identify and remove failed downloads ---
    failed_tickers = []
    for ticker_symbol, name in tickers.items():
        if name not in close_prices.columns or close_prices[name].isnull().all():
            failed_tickers.append(f"{ticker_symbol} ({name})")
    
    if failed_tickers:
        logger.warning(f"  - Could not download price data for the following tickers: {', '.join(failed_tickers)}. They will be excluded.")
        # Drop the failed columns before proceeding
        columns_to_drop = [tickers[t.split(' ')[0]] for t in failed_tickers]
        close_prices.drop(columns=columns_to_drop, inplace=True, errors='ignore')

    if close_prices.empty:
        logger.error("  - All macro tickers failed to download. No macro features will be available.")
        return pd.DataFrame()

    ticker_to_name_map = {v: k for k, v in tickers.items()}
    close_prices.rename(columns=ticker_to_name_map, inplace=True)
    close_prices.ffill(inplace=True)
    
    close_prices.to_parquet(data_cache_path)
    metadata = {"tickers": list(tickers.keys()), "last_date": close_prices.index.max().strftime('%Y-%m-%d')}
    with open(metadata_cache_path, 'w') as f:
        json.dump(metadata, f, indent=4)
    logger.info("  - Macro data downloaded and saved to new cache.")

    df_to_return = close_prices.reset_index()
    return df_to_return.rename(columns={df_to_return.columns[0]: 'Timestamp'})
    
# =============================================================================
# 9. FRAMEWORK ORCHESTRATION & MEMORY
# =============================================================================

def run_monte_carlo_simulation(price_series: pd.Series, n_simulations: int = 5000, n_days: int = 90) -> np.ndarray:
    """Generates Monte Carlo price path simulations using Geometric Brownian Motion."""
    log_returns = np.log(1 + price_series.pct_change())

    u = log_returns.mean()
    var = log_returns.var()
    drift = u - (0.5 * var)
    stdev = log_returns.std()

    daily_returns = np.exp(drift + stdev * np.random.normal(0, 1, (n_days, n_simulations)))

    price_paths = np.zeros_like(daily_returns)
    price_paths[0] = price_series.iloc[-1]
    for t in range(1, n_days):
        price_paths[t] = price_paths[t - 1] * daily_returns[t]

    return price_paths

def _sanitize_ai_suggestions(suggestions: Dict[str, Any]) -> Dict[str, Any]:
    """Validates and sanitizes critical numeric parameters from the AI."""
    sanitized = suggestions.copy()
    bounds = {
        'MAX_DD_PER_CYCLE': (0.05, 0.99), 'MAX_CONCURRENT_TRADES': (1, 20),
        'PARTIAL_PROFIT_TRIGGER_R': (0.1, 10.0), 'PARTIAL_PROFIT_TAKE_PCT': (0.1, 0.9),
        'OPTUNA_TRIALS': (25, 200),
        'LOOKAHEAD_CANDLES': (10, 500),
        'anomaly_contamination_factor': (0.001, 0.1)
    }
    integer_keys = ['MAX_CONCURRENT_TRADES', 'OPTUNA_TRIALS', 'LOOKAHEAD_CANDLES']

    for key, (lower, upper) in bounds.items():
        if key in sanitized and isinstance(sanitized.get(key), (int, float)):
            original_value = sanitized[key]
            clamped_value = max(lower, min(original_value, upper))
            if key in integer_keys: clamped_value = int(round(clamped_value))
            if original_value != clamped_value:
                logger.warning(f"  - Sanitizing AI suggestion for '{key}': Clamped value from {original_value} to {clamped_value} to meet model constraints.")
                sanitized[key] = clamped_value
    return sanitized

def _sanitize_frequency_string(freq_str: Any, default: str = '90D') -> str:
    """More robustly sanitizes a string to be a valid pandas frequency."""
    if isinstance(freq_str, int):
        sanitized_freq = f"{freq_str}D"
        logger.warning(f"AI provided a unit-less number for frequency. Interpreting '{freq_str}' as '{sanitized_freq}'.")
        return sanitized_freq

    if not isinstance(freq_str, str): freq_str = str(freq_str)
    if freq_str.isdigit():
        sanitized_freq = f"{freq_str}D"
        logger.warning(f"AI provided a unit-less string for frequency. Interpreting '{freq_str}' as '{sanitized_freq}'.")
        return sanitized_freq

    try:
        pd.tseries.frequencies.to_offset(freq_str)
        logger.info(f"Using valid frequency alias from AI: '{freq_str}'")
        return freq_str
    except ValueError:
        match = re.search(r'(\d+)\s*([A-Za-z]+)', freq_str)
        if match:
            num, unit_text = match.groups()
            unit_map = {'day': 'D', 'days': 'D', 'week': 'W', 'weeks': 'W', 'month': 'M', 'months': 'M'}
            unit = unit_map.get(unit_text.lower())
            if unit:
                sanitized_freq = f"{num}{unit}"
                logger.warning(f"Sanitizing AI-provided frequency '{freq_str}' to '{sanitized_freq}'.")
                return sanitized_freq

    logger.error(f"Could not parse a valid frequency from '{freq_str}'. Falling back to default '{default}'.")
    return default

def load_memory(champion_path: str, history_path: str) -> Dict:
    champion_config = None
    if os.path.exists(champion_path):
        try:
            with open(champion_path, 'r') as f: champion_config = json.load(f)
        except (json.JSONDecodeError, IOError) as e: logger.error(f"Could not read or parse champion file at {champion_path}: {e}")
    historical_runs = []
    if os.path.exists(history_path):
        try:
            with open(history_path, 'r') as f:
                for i, line in enumerate(f):
                    if not line.strip(): continue
                    try: historical_runs.append(json.loads(line))
                    except json.JSONDecodeError: logger.warning(f"Skipping malformed line {i+1} in history file: {history_path}")
        except IOError as e: logger.error(f"Could not read history file at {history_path}: {e}")
    return {"champion_config": champion_config, "historical_runs": historical_runs}

def _recursive_sanitize(data: Any) -> Any:
    """Recursively traverses a dict/list to convert non-JSON-serializable types."""
    if isinstance(data, dict):
        return {key: _recursive_sanitize(value) for key, value in data.items()}
    if isinstance(data, list):
        return [_recursive_sanitize(item) for item in data]
    if isinstance(data, Enum): return data.value
    if isinstance(data, (np.int64, np.int32)): return int(data)
    if isinstance(data, (np.float64, np.float32)):
        if np.isnan(data) or np.isinf(data): return None
        return float(data)
    if isinstance(data, (pd.Timestamp, datetime, date)): return data.isoformat()
    if isinstance(data, pathlib.Path): return str(data)
    return data

def save_run_to_memory(config: ConfigModel, new_run_summary: Dict, current_memory: Dict, diagnosed_regime: str) -> Optional[Dict]:
    """
    [Phase 1 Implemented] Saves the completed run summary to the history file and updates
    both the overall champion and the specific champion for the diagnosed market regime.
    """
    try:
        sanitized_summary = _recursive_sanitize(new_run_summary)
        with open(config.HISTORY_FILE_PATH, 'a') as f: f.write(json.dumps(sanitized_summary) + '\n')
        logger.info(f"-> Run summary appended to history file: {config.HISTORY_FILE_PATH}")
    except IOError as e: logger.error(f"Could not write to history file: {e}")

    MIN_TRADES_FOR_CHAMPION = 10
    new_metrics = new_run_summary.get("final_metrics", {})
    new_mar = new_metrics.get("mar_ratio", -np.inf)
    new_trade_count = new_metrics.get("total_trades", 0)

    # --- Overall Champion Logic ---
    current_champion = current_memory.get("champion_config")
    is_new_overall_champion = False
    if new_trade_count >= MIN_TRADES_FOR_CHAMPION and new_mar >= 0:
        if current_champion is None:
            is_new_overall_champion = True
            logger.info("Setting first-ever champion.")
        else:
            current_mar = current_champion.get("final_metrics", {}).get("mar_ratio", -np.inf)
            if new_mar is not None and new_mar > current_mar:
                is_new_overall_champion = True
    else:
        logger.info(f"Current run did not qualify for Overall Champion consideration. Trades: {new_trade_count}/{MIN_TRADES_FOR_CHAMPION}, MAR: {new_mar:.2f} (must be >= 0).")

    champion_to_save = new_run_summary if is_new_overall_champion else current_champion
    if is_new_overall_champion:
        prev_champ_mar = current_champion.get("final_metrics", {}).get("mar_ratio", -np.inf) if current_champion else -np.inf
        logger.info(f"NEW OVERALL CHAMPION! Current run's MAR Ratio ({new_mar:.2f}) beats previous champion's ({prev_champ_mar:.2f}).")

    try:
        if champion_to_save:
            with open(config.CHAMPION_FILE_PATH, 'w') as f: json.dump(_recursive_sanitize(champion_to_save), f, indent=4)
            logger.info(f"-> Overall Champion file updated: {config.CHAMPION_FILE_PATH}")
    except (IOError, TypeError) as e: logger.error(f"Could not write to overall champion file: {e}")

    # --- Regime-Specific Champion Logic ---
    regime_champions = {}
    if os.path.exists(config.REGIME_CHAMPIONS_FILE_PATH):
        try:
            with open(config.REGIME_CHAMPIONS_FILE_PATH, 'r') as f: regime_champions = json.load(f)
        except (json.JSONDecodeError, IOError) as e: logger.warning(f"Could not load regime champions file for updating: {e}")

    current_regime_champion = regime_champions.get(diagnosed_regime)
    is_new_regime_champion = False
    if new_trade_count >= MIN_TRADES_FOR_CHAMPION and new_mar >= 0:
         if current_regime_champion is None or new_mar > current_regime_champion.get("final_metrics", {}).get("mar_ratio", -np.inf):
             is_new_regime_champion = True

    if is_new_regime_champion:
        regime_champions[diagnosed_regime] = new_run_summary
        prev_mar = current_regime_champion.get("final_metrics", {}).get("mar_ratio", -np.inf) if current_regime_champion else -np.inf
        logger.info(f"NEW REGIME CHAMPION for '{diagnosed_regime}'! MAR Ratio ({new_mar:.2f}) beats previous ({prev_mar:.2f}).")
        try:
            with open(config.REGIME_CHAMPIONS_FILE_PATH, 'w') as f: json.dump(_recursive_sanitize(regime_champions), f, indent=4)
            logger.info(f"-> Regime Champions file updated: {config.REGIME_CHAMPIONS_FILE_PATH}")
        except (IOError, TypeError) as e: logger.error(f"Could not write to regime champions file: {e}")

    return champion_to_save

def initialize_playbook(playbook_path: str) -> Dict:
    """
    Initializes the strategy playbook.
    Every strategy now includes a default 'selected_features' list
    to serve as a robust fallback if the AI fails to provide a list.
    """
    DEFAULT_PLAYBOOK = {
        "EmaCrossoverRsiFilter": {
            "description": "[DIAGNOSTIC/MOMENTUM] A simple baseline strategy. Enters on an EMA cross, filtered by a basic RSI condition.",
            "style": "momentum",
            "selected_features": ['EMA_20', 'EMA_50', 'RSI', 'ADX', 'ATR'],
            "complexity": "low", "ideal_regime": ["Trending"], "asset_class_suitability": ["Any"], "ideal_macro_env": ["Any"]
        },
        "MeanReversionBollinger": {
            "description": "[DIAGNOSTIC/REVERSION] A simple baseline strategy. Enters when price touches Bollinger Bands in a low-ADX environment.",
            "style": "mean_reversion",
            "selected_features": ['bollinger_bandwidth', 'RSI', 'ADX', 'market_regime', 'stoch_k'],
            "complexity": "low", "ideal_regime": ["Ranging"], "asset_class_suitability": ["Any"], "ideal_macro_env": ["Neutral"]
        },
        "BreakoutVolumeSpike": {
             "description": "[DIAGNOSTIC/VOLATILITY] A simple baseline strategy that looks for price breakouts accompanied by a significant increase in volume.",
             "style": "volatility_breakout",
             "selected_features": ['ATR', 'volume_ma_ratio', 'bollinger_bandwidth', 'ADX', 'RealVolume'],
             "complexity": "low", "ideal_regime": ["Low Volatility"], "asset_class_suitability": ["Any"], "ideal_macro_env": ["Any"]
        },
        "ADXMomentum": {
            "description": "[MOMENTUM] A classic momentum strategy that enters when ADX confirms a strong trend and MACD indicates accelerating momentum.",
            "style": "momentum",
            "selected_features": ['ADX', 'MACD_hist', 'momentum_20', 'market_regime', 'EMA_50', 'DAILY_ctx_Trend'],
            "complexity": "medium", "ideal_regime": ["Strong Trending"], "asset_class_suitability": ["Any"], "ideal_macro_env": ["Any"]
        },
        "ClassicBollingerRSI": {
            "description": "[RANGING] A traditional mean-reversion strategy entering at the outer bands, filtered by low trend strength.",
            "style": "mean_reversion",
            "selected_features": ['bollinger_bandwidth', 'RSI', 'ADX', 'market_regime', 'stoch_k', 'cci'],
            "complexity": "low", "ideal_regime": ["Ranging"], "asset_class_suitability": ["Any"], "ideal_macro_env": ["Neutral"]
        },
        "VolatilityExpansionBreakout": {
            "description": "[BREAKOUT] Enters on strong breakouts that occur after a period of low-volatility consolidation (Bollinger Squeeze).",
            "style": "volatility_breakout",
            "selected_features": ['bollinger_bandwidth', 'ATR', 'market_volatility_index', 'DAILY_ctx_Trend', 'volume_ma_ratio'],
            "complexity": "medium", "ideal_regime": ["Low Volatility"], "asset_class_suitability": ["Any"], "ideal_macro_env": ["Event-Driven"]
        },
        "GNN_Market_Structure": {
            "description": "[SPECIALIZED] Uses a GNN to model inter-asset correlations for predictive features.",
            "style": "graph_based",
            "selected_features": ['ATR', 'RSI', 'ADX', 'bollinger_bandwidth', 'stoch_k', 'momentum_20'], # Base features for nodes
            "requires_gnn": True, "complexity": "specialized", "ideal_regime": ["Any"], "asset_class_suitability": ["Any"]
        },
        "Meta_Labeling_Filter": {
            "description": "[SPECIALIZED] Uses a secondary ML filter to improve a simple primary model's signal quality.",
            "style": "filter",
            "selected_features": ['ADX', 'ATR', 'bollinger_bandwidth', 'H1_ctx_Trend', 'DAILY_ctx_Trend', 'momentum_20', 'relative_performance'],
            "requires_meta_labeling": True, "complexity": "specialized", "ideal_regime": ["Any"], "asset_class_suitability": ["Any"]
        },
    }
    
    if not os.path.exists(playbook_path):
        logger.warning(f"'strategy_playbook.json' not found. Seeding a new one at: {playbook_path}")
        try:
            with open(playbook_path, 'w') as f:
                json.dump(DEFAULT_PLAYBOOK, f, indent=4)
            return DEFAULT_PLAYBOOK
        except IOError as e:
            logger.error(f"Failed to create playbook file: {e}. Using in-memory default.")
            return DEFAULT_PLAYBOOK
            
    try:
        with open(playbook_path, 'r') as f:
            playbook = json.load(f)

        updated = False
        # Loop through default playbook to add missing strategies or missing feature lists
        for strategy_name, default_config in DEFAULT_PLAYBOOK.items():
            if strategy_name not in playbook:
                playbook[strategy_name] = default_config
                logger.info(f"  - Adding new strategy to playbook: '{strategy_name}'")
                updated = True
            elif 'selected_features' not in playbook[strategy_name]:
                 playbook[strategy_name]['selected_features'] = default_config.get('selected_features', [])
                 logger.info(f"  - Adding missing 'selected_features' list to strategy: '{strategy_name}'")
                 updated = True

        if updated:
            logger.info("Playbook was updated. Saving changes...")
            with open(playbook_path, 'w') as f:
                json.dump(playbook, f, indent=4)

        logger.info(f"Successfully loaded and verified dynamic playbook from {playbook_path}")
        return playbook
    except (json.JSONDecodeError, IOError) as e:
        logger.error(f"Failed to load or parse playbook file: {e}. Using in-memory default.")
        return DEFAULT_PLAYBOOK

def load_nickname_ledger(ledger_path: str) -> Dict:
    logger.info("-> Loading Nickname Ledger...")
    if os.path.exists(ledger_path):
        try:
            with open(ledger_path, 'r') as f: return json.load(f)
        except (json.JSONDecodeError, IOError) as e:
            logger.error(f"  - Could not read or parse nickname ledger. Creating a new one. Error: {e}")
    return {}

def perform_strategic_review(history: Dict, directives_path: str) -> Tuple[Dict, List[Dict]]:
    logger.info("--- STRATEGIC REVIEW: Analyzing long-term strategy health...")
    health_report, directives, historical_runs = {}, [], history.get("historical_runs", [])
    if len(historical_runs) < 3:
        logger.info("--- STRATEGIC REVIEW: Insufficient history for a full review.")
        return health_report, directives

    for name in set(run.get('strategy_name') for run in historical_runs if run.get('strategy_name')):
        strategy_runs = [run for run in historical_runs if run.get('strategy_name') == name]
        if len(strategy_runs) < 3: continue
        failures = sum(1 for run in strategy_runs if run.get("final_metrics", {}).get("mar_ratio", 0) < 0.1)
        total_cycles = sum(len(run.get("cycle_details", [])) for run in strategy_runs)
        breaker_trips = sum(sum(1 for c in run.get("cycle_details",[]) if c.get("Status")=="Circuit Breaker") for run in strategy_runs)
        health_report[name] = {"ChronicFailureRate": f"{failures/len(strategy_runs):.0%}", "CircuitBreakerFrequency": f"{breaker_trips/total_cycles if total_cycles>0 else 0:.0%}", "RunsAnalyzed": len(strategy_runs)}

    recent_runs = historical_runs[-3:]
    if len(recent_runs) >= 3 and len(set(r.get('strategy_name') for r in recent_runs)) == 1:
        stagnant_strat_name = recent_runs[0].get('strategy_name')
        calmar_values = [r.get("final_metrics", {}).get("mar_ratio", 0) for r in recent_runs]
        if calmar_values[2] <= calmar_values[1] <= calmar_values[0]:
            if stagnant_strat_name in health_report: health_report[stagnant_strat_name]["StagnationWarning"] = True
            directives.append({"action": "FORCE_EXPLORATION", "strategy": stagnant_strat_name, "reason": f"Stagnation: No improvement over last 3 runs (MAR Ratios: {[round(c, 2) for c in calmar_values]})."})
            logger.warning(f"--- STRATEGIC REVIEW: Stagnation detected for '{stagnant_strat_name}'. Creating directive.")

    try:
        with open(directives_path, 'w') as f: json.dump(directives, f, indent=4)
        logger.info(f"--- STRATEGIC REVIEW: Directives saved to {directives_path}" if directives else "--- STRATEGIC REVIEW: No new directives generated.")
    except IOError as e: logger.error(f"--- STRATEGIC REVIEW: Failed to write to directives file: {e}")

    if health_report: logger.info(f"--- STRATEGIC REVIEW: Health report generated.\n{json.dumps(health_report, indent=2)}")
    return health_report, directives

def determine_timeframe_roles(detected_tfs: List[str]) -> Dict[str, Optional[str]]:
    if not detected_tfs: raise ValueError("No timeframes were detected from data files.")
    tf_with_values = sorted([(tf, FeatureEngineer.TIMEFRAME_MAP.get(tf.upper(), 99999)) for tf in detected_tfs], key=lambda x: x[1])
    sorted_tfs = [tf[0] for tf in tf_with_values]
    roles = {'base': sorted_tfs[0], 'medium': None, 'high': None}
    if len(sorted_tfs) == 2: roles['high'] = sorted_tfs[1]
    elif len(sorted_tfs) >= 3:
        roles['medium'], roles['high'] = sorted_tfs[1], sorted_tfs[2]
    logger.info(f"Dynamically determined timeframe roles: {roles}")
    return roles

def train_and_diagnose_regime(df: pd.DataFrame, results_dir: str, n_regimes: int = 5) -> Tuple[int, Dict]:
    """
    Trains a K-Means clustering model to identify market regimes or loads a pre-existing one.
    Diagnoses the current regime and returns a summary of all regime characteristics.
    """
    logger.info("-> Performing data-driven market regime analysis...")
    regime_model_path = os.path.join(results_dir, 'regime_model.pkl')
    regime_scaler_path = os.path.join(results_dir, 'regime_scaler.pkl')

    # Features that define a market's "personality"
    regime_features = ['ATR', 'ADX', 'hurst_exponent', 'realized_volatility', 'bollinger_bandwidth']
    regime_features = [f for f in regime_features if f in df.columns] # Ensure all features exist
    
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

def apply_genetic_rules_to_df(full_df: pd.DataFrame, rules: Tuple[str, str], config: ConfigModel) -> pd.DataFrame:
    """
    Applies the evolved genetic rules to the entire dataframe to generate
    a 'primary_model_signal' column for the meta-labeler.
    """
    logger.info("-> Applying evolved genetic rules to the full dataset...")
    df_with_signals = full_df.copy()
    long_rule, short_rule = rules
    
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

def _generate_cache_metadata(config: ConfigModel, files: List[str], tf_roles: Dict, feature_engineer_class: type) -> Dict:
    """
    Generates a dictionary of metadata to validate the feature cache.
    This includes parameters that affect the output of feature engineering.
    """
    file_metadata = {}
    for filename in sorted(files):
        file_path = os.path.join(config.BASE_PATH, filename)
        if os.path.exists(file_path):
            stat = os.stat(file_path)
            file_metadata[filename] = {"mtime": stat.st_mtime, "size": stat.st_size}

    script_hash = ""
    try:
        fe_source_code = inspect.getsource(feature_engineer_class)
        script_hash = hashlib.sha256(fe_source_code.encode('utf-8')).hexdigest()
    except Exception as e:
        logger.warning(f"Could not generate FeatureEngineer class hash: {e}")

    # Create a dictionary of ONLY the parameters that affect feature creation
    feature_params = {
        'script_sha256_hash': script_hash,
        'DYNAMIC_INDICATOR_PARAMS': config.DYNAMIC_INDICATOR_PARAMS.copy(),
        'USE_PCA_REDUCTION': config.USE_PCA_REDUCTION,
        'PCA_N_COMPONENTS': config.PCA_N_COMPONENTS,
        'RSI_PERIODS_FOR_PCA': config.RSI_PERIODS_FOR_PCA,
        'ADX_THRESHOLD_TREND': config.ADX_THRESHOLD_TREND,
        'DISPLACEMENT_PERIOD': config.DISPLACEMENT_PERIOD,
        'GAP_DETECTION_LOOKBACK': config.GAP_DETECTION_LOOKBACK,
        'PARKINSON_VOLATILITY_WINDOW': config.PARKINSON_VOLATILITY_WINDOW,
        'YANG_ZHANG_VOLATILITY_WINDOW': config.YANG_ZHANG_VOLATILITY_WINDOW,
        'KAMA_REGIME_FAST': config.KAMA_REGIME_FAST,
        'KAMA_REGIME_SLOW': config.KAMA_REGIME_SLOW,
        'AUTOCORR_LAG': config.AUTOCORR_LAG,
        'HURST_EXPONENT_WINDOW': config.HURST_EXPONENT_WINDOW,
        'HAWKES_KAPPA': config.HAWKES_KAPPA,
        'anomaly_contamination_factor': config.anomaly_contamination_factor
    }
    return {"files": file_metadata, "params": feature_params}

def _apply_operating_state_rules(config: ConfigModel) -> ConfigModel:
    """
    Applies the risk and behavior rules based on the current operating state.
    This function modifies the config object for the upcoming cycle.
    """
    state = config.operating_state
    if state not in config.STATE_BASED_CONFIG:
        logger.warning(f"Operating State '{state.value}' not found in STATE_BASED_CONFIG. Using defaults.")
        return config

    state_rules = config.STATE_BASED_CONFIG[state]
    logger.info(f"-> Applying rules for Operating State: '{state.value}'")

    # Override config parameters with the rules for the current state
    config.MAX_DD_PER_CYCLE = state_rules["max_dd_per_cycle"]
    config.BASE_RISK_PER_TRADE_PCT = state_rules["base_risk_pct"]
    config.MAX_CONCURRENT_TRADES = state_rules["max_concurrent_trades"]
    
    logger.info(f"  - Set MAX_DD_PER_CYCLE to {config.MAX_DD_PER_CYCLE:.0%}")
    logger.info(f"  - Set BASE_RISK_PER_TRADE_PCT to {config.BASE_RISK_PER_TRADE_PCT:.3%}")
    logger.info(f"  - Set MAX_CONCURRENT_TRADES to {config.MAX_CONCURRENT_TRADES}")

    return config

def _validate_and_fix_spread_config(ai_suggestions: Dict[str, Any], fallback_config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Checks the SPREAD_CONFIG from the AI. If the format is invalid, it replaces it
    with the default from the fallback_config to prevent a crash.
    """
    if 'SPREAD_CONFIG' not in ai_suggestions:
        return ai_suggestions # No spread config provided, nothing to do.

    spread_config = ai_suggestions['SPREAD_CONFIG']
    is_valid = True

    if not isinstance(spread_config, dict):
        is_valid = False
    else:
        # Check each value in the dictionary
        for symbol, value in spread_config.items():
            if not isinstance(value, dict) or 'normal_pips' not in value or 'volatile_pips' not in value:
                is_valid = False
                logger.warning(f"  - Invalid SPREAD_CONFIG entry found for '{symbol}'. Value was: {value}")
                break # Found an invalid entry, no need to check further

    if not is_valid:
        logger.warning("AI returned an invalid format for SPREAD_CONFIG. Discarding AI suggestion for spreads and using the framework's default values.")
        # Replace the invalid AI suggestion with the original default from the fallback config
        ai_suggestions['SPREAD_CONFIG'] = fallback_config.get('SPREAD_CONFIG', {})
    else:
        logger.info("  - AI-provided SPREAD_CONFIG format is valid.")

    return ai_suggestions    

def deep_merge_dicts(original: dict, updates: dict) -> dict:
    """
    Recursively merges two dictionaries. 'updates' values will overwrite
    'original' values, except for nested dicts which are merged.
    """
    merged = original.copy()
    for key, value in updates.items():
        if key in merged and isinstance(merged.get(key), dict) and isinstance(value, dict):
            merged[key] = deep_merge_dicts(merged[key], value)
        else:
            merged[key] = value
    return merged

def _is_maintenance_period() -> Tuple[bool, str]:
    """
    Checks for periods where trading should be paused for operational integrity.
    Returns a tuple of (is_maintenance, reason).
    """
    now = datetime.now()       
    # Pause for year-end illiquidity period
    if (now.month == 12 and now.day >= 23) or (now.month == 1 and now.day <= 2):
        return True, "Year-end holiday period (low liquidity)"
        
    return False, ""

def _detect_surge_opportunity(df_slice: pd.DataFrame, lookback_days: int = 5, threshold: float = 2.5) -> bool:
    """
    Analyzes a recent slice of data to detect a sudden volatility spike.
    This acts as a trigger for the OPPORTUNISTIC_SURGE state.
    """
    if df_slice.empty or 'ATR' not in df_slice.columns:
        return False
        
    recent_data = df_slice.last(f'{lookback_days}D')
    if len(recent_data) < 20: # Ensure enough data for a meaningful average
        return False
        
    # Calculate the average ATR over the lookback period, excluding the most recent candle
    historical_avg_atr = recent_data['ATR'].iloc[:-1].mean()
    latest_atr = recent_data['ATR'].iloc[-1]
    
    if pd.isna(historical_avg_atr) or pd.isna(latest_atr) or historical_avg_atr == 0:
        return False
        
    # If the latest ATR is significantly higher than the recent average, flag it as a surge opportunity
    if latest_atr > (historical_avg_atr * threshold):
        logger.info(f"! VOLATILITY SURGE DETECTED ! Latest ATR ({latest_atr:.4f}) is > {threshold}x the recent average ({historical_avg_atr:.4f}).")
        return True
        
    return False

def _run_feature_learnability_test(df_train_labeled: pd.DataFrame, feature_list: list, target_col: str) -> str:
    """
    Checks the information content of features against the label using Mutual Information.
    Returns a string summary for the AI.
    """
    if target_col not in df_train_labeled.columns:
        return f"Feature Learnability: Target column '{target_col}' not found."

    valid_features = [f for f in feature_list if f in df_train_labeled.columns]
    if not valid_features:
        return "Feature Learnability: No valid features found to test."

    X = df_train_labeled[valid_features].copy()
    y = df_train_labeled[target_col].copy()

    # Align and drop NaNs
    aligned_df = pd.concat([X, y], axis=1).dropna()
    if aligned_df.empty:
        return "Feature Learnability: No non-NaN data available for features and target."

    X_aligned = aligned_df[valid_features]
    y_aligned = aligned_df[target_col]

    # Impute remaining NaNs just in case (e.g., if a column was all NaN before drop)
    X_aligned.fillna(X_aligned.median(), inplace=True)

    try:
        # For regression target, convert to discrete bins for mutual_info_classif
        if pd.api.types.is_numeric_dtype(y_aligned) and y_aligned.nunique() > 10:
            y_binned = pd.qcut(y_aligned, q=5, labels=False, duplicates='drop')
            if y_binned.nunique() < 2:
                 return "Feature Learnability: Target could not be binned for MI score."
            scores = mutual_info_classif(X_aligned, y_binned, random_state=42)
        else:
            scores = mutual_info_classif(X_aligned, y_aligned, random_state=42)

        mi_scores = pd.Series(scores, index=X_aligned.columns).sort_values(ascending=False)
        
        top_5 = mi_scores.head(5)
        summary = ", ".join([f"{idx}: {score:.4f}" for idx, score in top_5.items()])
        return f"Feature Learnability vs '{target_col}' (Top 5 MI Scores): {summary}"
    except Exception as e:
        logger.error(f"  - Could not run feature learnability test: {e}")
        return f"Feature Learnability: Error during calculation - {e}"

def _label_distribution_report(df: pd.DataFrame, label_col: str) -> str:
    """
    Generates a report on the class balance of the labels.
    Returns a string summary for the AI.
    """
    if label_col not in df.columns:
        return f"Label Distribution: Target column '{label_col}' not found."
        
    counts = df[label_col].value_counts(normalize=True)
    report_dict = {str(k): f"{v:.2%}" for k, v in counts.to_dict().items()}
    return f"Label Distribution for '{label_col}': {report_dict}"

def _generate_pre_analysis_summary(df_train: pd.DataFrame, features: list, target: str) -> str:
    """Generates a text summary of label distribution and feature learnability."""
    label_report = _label_distribution_report(df_train, target)
    learnability_report = _run_feature_learnability_test(df_train, features, target)
    return f"{label_report}\n{learnability_report}"

def _generate_raw_data_summary_for_ai(data_by_tf: Dict[str, pd.DataFrame], tf_roles: Dict) -> Dict:
    """Generates a data summary for AI based on raw/lightly processed data."""
    logger.info("Generating RAW data summary for AI...")
    summary = {
        "timeframes_detected": list(data_by_tf.keys()),
        "timeframe_roles": tf_roles,
        "base_timeframe": tf_roles.get('base'),
    }
    base_tf_data = data_by_tf.get(tf_roles.get('base'))
    if base_tf_data is not None and not base_tf_data.empty:
        summary["assets_detected"] = list(base_tf_data['Symbol'].unique())
        summary["date_range_min"] = str(base_tf_data.index.min())
        summary["date_range_max"] = str(base_tf_data.index.max())
        
        # Calculate raw ATR for the base timeframe if possible
        raw_atr_summary = {}
        for symbol, group_df in base_tf_data.groupby('Symbol'):
            if len(group_df) > 14: # Min periods for ATR
                high_low = group_df['High'] - group_df['Low']
                high_close = np.abs(group_df['High'] - group_df['Close'].shift())
                low_close = np.abs(group_df['Low'] - group_df['Close'].shift())
                tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1, skipna=False)
                atr = tr.ewm(alpha=1/14, adjust=False, min_periods=1).mean()
                raw_atr_summary[symbol] = round(atr.iloc[-1], 5) if not atr.empty else "N/A"
            else:
                raw_atr_summary[symbol] = "N/A (Insufficient data)"
        summary["avg_base_tf_raw_atr_approx"] = raw_atr_summary
        summary["feature_engineered_df_shape"] = "To be determined after AI config"
        summary["engineered_feature_count"] = "To be determined after AI config"
    else:
        summary["assets_detected"] = []
        summary["date_range_min"] = "N/A"
        summary["date_range_max"] = "N/A"
        summary["avg_base_tf_raw_atr_approx"] = {}

    return summary

def _generate_inter_asset_correlation_summary_for_ai(data_by_tf: Dict[str, pd.DataFrame], tf_roles: Dict, top_n: int = 5) -> str:
    """Generates a text summary of inter-asset price correlations from raw data."""
    logger.info("Generating inter-asset price correlation summary for AI...")
    base_tf_name = tf_roles.get('base')
    if not base_tf_name or base_tf_name not in data_by_tf:
        return "Inter-asset correlation: Base timeframe data not available."

    base_df = data_by_tf[base_tf_name]
    if base_df.empty or 'Symbol' not in base_df.columns or 'Close' not in base_df.columns:
        return "Inter-asset correlation: Data insufficient for analysis."

    # Pivot to get symbols as columns, Close prices as values
    try:
        price_pivot = base_df.pivot_table(index=base_df.index, columns='Symbol', values='Close')
        price_pivot = price_pivot.ffill().bfill() # Handle missing values
        if price_pivot.shape[1] < 2:
            return "Inter-asset correlation: Fewer than 2 assets with price data."

        corr_matrix = price_pivot.corr(method='spearman').abs()
        upper_tri = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
        top_corr_pairs = upper_tri.stack().sort_values(ascending=False).head(top_n)

        summary_str = "Top Inter-Asset Price Correlations (Spearman's Rho):\n"
        if top_corr_pairs.empty:
            summary_str += "No significant inter-asset correlations found."
        else:
            for (asset1, asset2), corr_val in top_corr_pairs.items():
                summary_str += f"- {asset1} & {asset2}: {corr_val:.3f}\n"
        return summary_str
    except Exception as e:
        logger.error(f"Error generating inter-asset correlation: {e}")
        return "Inter-asset correlation: Error during calculation."

def _diagnose_raw_market_regime(data_by_tf: Dict[str, pd.DataFrame], tf_roles: Dict) -> str:
    """Diagnoses a very basic market regime from raw data for initial AI input."""
    logger.info("Diagnosing RAW market regime (heuristic)...")
    base_tf_name = tf_roles.get('base')
    if not base_tf_name or base_tf_name not in data_by_tf:
        return "Unknown (Base timeframe data not available)"

    base_df = data_by_tf[base_tf_name].copy()
    if base_df.empty or len(base_df) < 50: # Need some data
        return "Unknown (Insufficient raw data for regime diagnosis)"

    # Use a simple measure like recent price volatility (e.g., std of log returns)
    # Consider the average across all symbols for a general market feel
    regime_str = "Regime assessment pending full feature engineering."
    try:
        log_returns = np.log(base_df.groupby('Symbol')['Close'].pct_change().add(1))
        # Look at the volatility of the last ~30 periods (e.g. days if base_tf is daily)
        recent_vol = log_returns.groupby(base_df['Symbol']).rolling(window=30, min_periods=15).std().groupby(level=0).last()

        if not recent_vol.empty:
            avg_recent_vol = recent_vol.mean()
            # Heuristic thresholds (these are arbitrary and would need tuning)
            if avg_recent_vol > 0.02: # Example: >2% daily stdev is high vol
                regime_str = "Potentially High Volatility (Raw Data Heuristic)"
            elif avg_recent_vol < 0.005: # Example: <0.5% daily stdev is low vol
                regime_str = "Potentially Low Volatility (Raw Data Heuristic)"
            else:
                regime_str = "Potentially Medium Volatility (Raw Data Heuristic)"
    except Exception as e:
        logger.warning(f"Could not perform raw regime diagnosis: {e}")
        regime_str = "Unknown (Error in raw regime diagnosis)"
    
    return regime_str

def _generate_raw_data_health_report(data_by_tf: Dict[str, pd.DataFrame], tf_roles: Dict) -> Dict:
    """Generates a basic health report from raw data."""
    logger.info("Generating RAW data health report...")
    report = {"status": "OK", "issues": []}
    base_tf_name = tf_roles.get('base')
    if not base_tf_name or base_tf_name not in data_by_tf:
        report["status"] = "Error"
        report["issues"].append("Base timeframe data not available.")
        return report

    base_df = data_by_tf[base_tf_name]
    if base_df.empty:
        report["status"] = "Error"
        report["issues"].append("Base timeframe DataFrame is empty.")
        return report

    report["raw_shape_base_tf"] = base_df.shape
    
    # Check for NaNs in critical OHLC columns
    for col in ['Open', 'High', 'Low', 'Close']:
        if col in base_df.columns:
            nan_pct = base_df[col].isnull().sum() * 100 / len(base_df)
            if nan_pct > 5: # Allow some NaNs, but flag if excessive
                report["issues"].append(f"High NaN percentage in raw '{col}': {nan_pct:.2f}%")
        else:
            report["issues"].append(f"Critical column '{col}' missing in raw base TF data.")

    if not report["issues"]:
        report["status"] = "Raw data seems reasonable for initial assessment."
    else:
        report["status"] = "Potential issues found in raw data."
        
    return report

def _log_config_and_environment(config: ConfigModel):
    """Logs key configuration parameters and system environment details."""
    logger.info("--- CONFIGURATION & ENVIRONMENT ---")
    
    # Log key config parameters
    config_summary = {
        "Strategy": config.strategy_name,
        "Nickname": config.nickname,
        "Initial Capital": f"${config.INITIAL_CAPITAL:,.2f}",
        "Operating State": config.operating_state.value,
        "Training Window": config.TRAINING_WINDOW,
        "Retraining Freq": config.RETRAINING_FREQUENCY,
        "Optuna Trials": config.OPTUNA_TRIALS,
        "Feature Selection": config.FEATURE_SELECTION_METHOD,
        "Labeling Method": config.LABELING_METHOD,
        "Max DD per Cycle": f"{config.MAX_DD_PER_CYCLE:.2%}",
        "Base Risk per Trade": f"{config.BASE_RISK_PER_TRADE_PCT:.3%}"
    }
    for key, value in config_summary.items():
        logger.info(f"  - {key}: {value}")

    # Log system environment
    try:
        process = psutil.Process(os.getpid())
        mem_info = process.memory_info()
        logger.info("--- System Info ---")
        logger.info(f"  - Python Version: {sys.version.split(' ')[0]}")
        logger.info(f"  - CPU Cores: {psutil.cpu_count(logical=True)}")
        logger.info(f"  - Memory Usage: {mem_info.rss / 1024**2:.2f} MB")
    except Exception as e:
        logger.warning(f"Could not retrieve system info: {e}")
    logger.info("---------------------------------")

def get_walk_forward_periods(start_date: pd.Timestamp, end_date: pd.Timestamp, train_window: str, retrain_freq: str, gap: str) -> Tuple[List, List, List, List]:
    """
    Generates lists of start/end dates for training and testing periods for walk-forward validation.
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
            if test_start >= test_end: # Break if the last test period is invalid
                 break

        train_start_dates.append(train_start)
        train_end_dates.append(train_end)
        test_start_dates.append(test_start)
        test_end_dates.append(test_end)

        current_date += retrain_offset
        if current_date + train_offset > end_date:
            break
            
    return train_start_dates, train_end_dates, test_start_dates, test_end_dates

def _generate_nickname(strategy: str, ai_suggestion: Optional[str], ledger: Dict, ledger_path: str, version: str) -> str:
    """Generates a unique, memorable nickname for the run."""
    if ai_suggestion and ai_suggestion not in ledger.values():
        logger.info(f"Using unique nickname from AI: '{ai_suggestion}'")
        ledger[version] = ai_suggestion
    else:
        if version in ledger:
            logger.info(f"Using existing nickname from ledger for version {version}: '{ledger[version]}'")
            return ledger[version]
        
        # Generate a new nickname if AI didn't provide a unique one
        adjectives = ["Quantum", "Cyber", "Nova", "Helios", "Orion", "Apex", "Zenith", "Vector", "Pulse", "Omega"]
        nouns = ["Vortex", "Matrix", "Catalyst", "Protocol", "Horizon", "Synchron", "Dynamo", "Sentinel", "Echo", "Pioneer"]
        
        while True:
            new_nick = f"{random.choice(adjectives)}{random.choice(nouns)}{random.randint(10, 99)}"
            if new_nick not in ledger.values():
                logger.info(f"Generated new unique nickname: '{new_nick}'")
                ledger[version] = new_nick
                break
    
    try:
        with open(ledger_path, 'w') as f:
            json.dump(ledger, f, indent=4)
    except IOError as e:
        logger.error(f"Could not save updated nickname ledger: {e}")
        
    return ledger[version]

def get_and_cache_asset_types(symbols: List[str], config_dict: Dict, gemini_analyzer: GeminiAnalyzer) -> Dict[str, str]:
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
            with open(cache_path, 'w') as f:
                json.dump(classified_assets, f, indent=4)
            logger.info(f"Asset classifications saved to cache at {cache_path}")
        except IOError as e:
            logger.error(f"Could not save asset type cache: {e}")
        return classified_assets
    
    logger.error("Failed to classify assets using AI, returning empty dictionary.")
    return {}

def generate_dynamic_config(primary_asset_class: str, base_config: Dict) -> Dict:
    """Adjusts the base configuration based on the primary asset class."""
    logger.info(f"Dynamically generating configuration for primary asset class: {primary_asset_class}")
    dynamic_config = base_config.copy()

    if primary_asset_class == 'Commodities' or primary_asset_class == 'Indices':
        logger.info("Adjusting config for Commodities/Indices: Higher risk, wider TP/SL.")
        dynamic_config.update({
            "TP_ATR_MULTIPLIER": 2.5,
            "SL_ATR_MULTIPLIER": 2.0,
            "BASE_RISK_PER_TRADE_PCT": 0.012,
            "MAX_DD_PER_CYCLE": 0.30,
        })
    elif primary_asset_class == 'Crypto':
        logger.info("Adjusting config for Crypto: Highest risk, widest TP/SL.")
        dynamic_config.update({
            "TP_ATR_MULTIPLIER": 3.0,
            "SL_ATR_MULTIPLIER": 2.5,
            "BASE_RISK_PER_TRADE_PCT": 0.015,
            "MAX_DD_PER_CYCLE": 0.35,
        })
    else: # Default for Forex
        logger.info("Using standard configuration for Forex.")
        dynamic_config.update({
            "TP_ATR_MULTIPLIER": 2.0,
            "SL_ATR_MULTIPLIER": 1.5,
            "BASE_RISK_PER_TRADE_PCT": 0.01,
            "MAX_DD_PER_CYCLE": 0.25,
        })
    return dynamic_config

def _update_operating_state(config: ConfigModel, cycle_history: List[Dict], current_state: OperatingState) -> OperatingState:
    """
    Analyzes recent performance to determine the next operating state.
    """
    if not cycle_history:
        return current_state # No history, no change

    last_cycle = cycle_history[-1]
    metrics = last_cycle.get("metrics", {})
    status = last_cycle.get("status")
    
    # Priority 1: If circuit breaker was tripped, immediately move to Drawdown Control
    if status == "Breaker Tripped":
        if current_state != OperatingState.DRAWDOWN_CONTROL:
            logger.warning("! STATE CHANGE ! Circuit breaker tripped. Moving to DRAWDOWN_CONTROL.")
            return OperatingState.DRAWDOWN_CONTROL
        else:
            logger.info("Remains in DRAWDOWN_CONTROL after another breaker trip.")
            return OperatingState.DRAWDOWN_CONTROL

    # Analyze performance to decide on other state changes
    mar_ratio = metrics.get('mar_ratio', 0)
    pnl = metrics.get('TotalPNL', 0)

    # Logic for moving out of Drawdown Control
    if current_state == OperatingState.DRAWDOWN_CONTROL:
        if pnl > 0 and mar_ratio > 0.3:
            logger.info("! STATE CHANGE ! Positive performance observed. Moving from Drawdown Control to CONSERVATIVE_BASELINE.")
            return OperatingState.CONSERVATIVE_BASELINE
        else:
            logger.info("Performance still weak. Remaining in DRAWDOWN_CONTROL.")
            return OperatingState.DRAWDOWN_CONTROL
    
    # Logic for moving from Baseline to Aggressive
    if current_state == OperatingState.CONSERVATIVE_BASELINE:
        if len(cycle_history) >= 2:
            # Check if last two cycles were profitable and stable
            last_two_cycles_ok = all(c.get("metrics", {}).get("TotalPNL", -1) >= 0 and not c.get("metrics", {}).get("BreakerTripped") for c in cycle_history[-2:])
            if last_two_cycles_ok:
                logger.info("! STATE CHANGE ! Consistent positive performance. Moving from Baseline to AGGRESSIVE_EXPANSION.")
                return OperatingState.AGGRESSIVE_EXPANSION
    
    # Logic for moving from Aggressive back to Baseline (a sign of trouble)
    if current_state == OperatingState.AGGRESSIVE_EXPANSION:
        if pnl < 0 or mar_ratio < 0.2:
            logger.warning("! STATE CHANGE ! Performance dip detected. Moving from Aggressive back to CONSERVATIVE_BASELINE.")
            return OperatingState.CONSERVATIVE_BASELINE
            
    # If in Opportunistic Surge, revert to the previous state after one cycle
    # (This assumes the state before surge is tracked, or it defaults to baseline)
    if current_state == OperatingState.OPPORTUNISTIC_SURGE:
         logger.info("! STATE CHANGE ! Opportunistic Surge cycle complete. Reverting to CONSERVATIVE_BASELINE.")
         return OperatingState.CONSERVATIVE_BASELINE

    # Default: no change
    return current_state

def _run_feature_learnability_test(df_train_labeled: pd.DataFrame, feature_list: list, target_col: str) -> str:
    """
    Checks the information content of features against the label using Mutual Information.
    Returns a string summary for the AI.
    """
    if target_col not in df_train_labeled.columns:
        return f"Feature Learnability: Target column '{target_col}' not found."

    valid_features = [f for f in feature_list if f in df_train_labeled.columns]
    if not valid_features:
        return "Feature Learnability: No valid features found to test."

    X = df_train_labeled[valid_features].copy()
    y = df_train_labeled[target_col].copy()

    aligned_df = pd.concat([X, y], axis=1).dropna()
    if aligned_df.empty:
        return "Feature Learnability: No non-NaN data available for features and target."

    X_aligned = aligned_df[valid_features]
    y_aligned = aligned_df[target_col]

    X_aligned.fillna(X_aligned.median(), inplace=True)

    try:
        if pd.api.types.is_numeric_dtype(y_aligned) and y_aligned.nunique() > 10:
            y_binned = pd.qcut(y_aligned, q=5, labels=False, duplicates='drop')
            if y_binned.nunique() < 2:
                 return "Feature Learnability: Target could not be binned for MI score."
            scores = mutual_info_classif(X_aligned, y_binned, random_state=42)
        else:
            scores = mutual_info_classif(X_aligned, y_aligned, random_state=42)

        mi_scores = pd.Series(scores, index=X_aligned.columns).sort_values(ascending=False)
        
        top_5 = mi_scores.head(5)
        summary = ", ".join([f"{idx}: {score:.4f}" for idx, score in top_5.items()])
        return f"Feature Learnability vs '{target_col}' (Top 5 MI Scores): {summary}"
    except Exception as e:
        logger.error(f"  - Could not run feature learnability test: {e}")
        return f"Feature Learnability: Error during calculation - {e}"

def _label_distribution_report(df: pd.DataFrame, label_col: str) -> str:
    """
    Generates a report on the class balance of the labels.
    Returns a string summary for the AI.
    """
    if label_col not in df.columns:
        return f"Label Distribution: Target column '{label_col}' not found."
        
    counts = df[label_col].value_counts(normalize=True)
    report_dict = {str(k): f"{v:.2%}" for k, v in counts.to_dict().items()}
    return f"Label Distribution for '{label_col}': {report_dict}"

def _generate_pre_analysis_summary(df_train: pd.DataFrame, features: list, target: str) -> str:
    """Generates a text summary of label distribution and feature learnability."""
    label_report = _label_distribution_report(df_train, target)
    learnability_report = _run_feature_learnability_test(df_train, features, target)
    return f"{label_report}\n{learnability_report}"

def run_single_instance(fallback_config_dict: Dict, framework_history_loaded: Dict, playbook_loaded: Dict, nickname_ledger_loaded: Dict, directives_loaded: List[Dict], api_interval_seconds: int):

    run_timestamp_str = datetime.now().strftime("%Y%m%d-%H%M%S")
    gemini_analyzer = GeminiAnalyzer()
    api_timer = APITimer(interval_seconds=api_interval_seconds)

    # --- Phase 1: Initial Setup (Minimal Config for Paths, Basic Logging) ---
    temp_minimal_config_dict = fallback_config_dict.copy()
    temp_minimal_config_dict.update({
        "REPORT_LABEL": f"AxiomEdge_V{VERSION}_Setup",
        "run_timestamp": run_timestamp_str,
        "strategy_name": "InitialSetupPhase",
        "nickname": "init_setup",
        "selected_features": []
    })
    
    try:
        temp_config_for_paths_obj = ConfigModel(**temp_minimal_config_dict)
    except ValidationError as e:
        print(f"CRITICAL: Pydantic validation error for temp_minimal_config_dict: {e}")
        return {"status": "error", "message": "Initial temporary config validation failed"}

    prelim_log_dir = os.path.join(temp_config_for_paths_obj.BASE_PATH, "Results", "PrelimLogs")
    os.makedirs(prelim_log_dir, exist_ok=True)
    prelim_log_path = os.path.join(prelim_log_dir, f"pre_run_{run_timestamp_str}.log")
    _setup_logging(prelim_log_path, temp_config_for_paths_obj.REPORT_LABEL)
    logger = logging.getLogger("ML_Trading_Framework")

    # --- Phase 2: Data Loading & Caching/Feature Engineering ---
    data_loader = DataLoader(temp_config_for_paths_obj)
    all_files = [f for f in os.listdir(temp_config_for_paths_obj.BASE_PATH) if f.endswith(('.csv', '.txt')) and re.match(r'^[A-Z0-9]+_[A-Z0-9]+', f)]
    data_by_tf, detected_timeframes = data_loader.load_and_parse_data(all_files)
    if not data_by_tf:
        logger.critical("Data loading failed. Exiting.")
        return {"status": "error", "message": "Data loading failed"}
    
    tf_roles = determine_timeframe_roles(detected_timeframes)
    fe_initial = FeatureEngineer(temp_config_for_paths_obj, tf_roles, playbook_loaded)
    base_tf_data = data_by_tf[tf_roles['base']]

    full_df = None
    cache_is_valid = False
    current_metadata = _generate_cache_metadata(temp_config_for_paths_obj, all_files, tf_roles, FeatureEngineer)

    if temp_config_for_paths_obj.USE_FEATURE_CACHING:
        if os.path.exists(temp_config_for_paths_obj.CACHE_PATH) and os.path.exists(temp_config_for_paths_obj.CACHE_METADATA_PATH):
            logger.info("-> Found existing feature cache. Validating...")
            try:
                with open(temp_config_for_paths_obj.CACHE_METADATA_PATH, 'r') as f:
                    cached_metadata = json.load(f)
                if cached_metadata == current_metadata:
                    logger.info("  - Cache is VALID. Loading features from disk...")
                    full_df = pd.read_parquet(temp_config_for_paths_obj.CACHE_PATH)
                    logger.info(f"  - Successfully loaded features from cache. Shape: {full_df.shape}")
                    cache_is_valid = True
                else:
                    logger.info("  - Cache is INVALID (data or parameters changed). Recalculating features.")
            except (IOError, json.JSONDecodeError) as e:
                logger.warning(f"  - Could not read or parse cache metadata. Recalculating features. Error: {e}")
        else:
            logger.info("-> No feature cache found. Features will be calculated and cached.")
    else:
        logger.info("-> Feature caching is disabled in config. Recalculating features.")
        
    if not cache_is_valid:
        logger.info("-> Starting feature engineering calculation...")
        full_df = fe_initial.engineer_features(base_tf_data, data_by_tf)

        if temp_config_for_paths_obj.USE_FEATURE_CACHING and not full_df.empty:
            try:
                os.makedirs(os.path.dirname(temp_config_for_paths_obj.CACHE_PATH), exist_ok=True)
                full_df.to_parquet(temp_config_for_paths_obj.CACHE_PATH)
                with open(temp_config_for_paths_obj.CACHE_METADATA_PATH, 'w') as f:
                    json.dump(current_metadata, f, indent=4)
                logger.info(f"-> Saved new feature cache to: {temp_config_for_paths_obj.CACHE_PATH}")
            except IOError as e:
                logger.error(f"  - Failed to save feature cache: {e}")

    if full_df is None or full_df.empty:
        logger.critical("Feature engineering resulted in an empty dataframe. Exiting.")
        return {"status": "error", "message": "Feature engineering failed or cache was invalid and recalculation failed."}

    # --- Phase 3: AI-Driven Setup ---
    raw_data_summary = _generate_raw_data_summary_for_ai(data_by_tf, tf_roles)
    correlation_summary = _generate_inter_asset_correlation_summary_for_ai(data_by_tf, tf_roles)
    raw_regime = _diagnose_raw_market_regime(data_by_tf, tf_roles)
    raw_health_report = _generate_raw_data_health_report(data_by_tf, tf_roles)
    
    MASTER_MACRO_TICKERS = {
        "VIX": "^VIX", "DXY": "DX-Y.NYB", "US10Y_YIELD": "^TNX",
        "EURUSD": "EURUSD=X", "GBPUSD": "GBPUSD=X", "AUDUSD": "AUDUSD=X",
        "USDCAD": "USDCAD=X", "USDJPY": "USDJPY=X",
        "GOLD": "GC=F", "OIL_WTI": "CL=F", "GERMAN10Y_YIELD": "DE10Y-DE"
    }

    last_run = framework_history_loaded.get('historical_runs', [])[-1] if framework_history_loaded.get('historical_runs') else None
    if not last_run or last_run.get("final_metrics", {}).get("total_trades", 0) == 0:
        prime_directive = "ESTABLISH TRADING BASELINE. The last run failed to produce any trades. Your absolute priority is to configure a model that actively participates in the market. Profitability is a secondary concern to execution."
    else:
        prime_directive = "OPTIMIZE PERFORMANCE. A successful trading baseline exists. Your priority is now to improve risk-adjusted returns (e.g., MAR Ratio), profitability, and model stability."

    ai_comprehensive_setup = api_timer.call(
        gemini_analyzer.get_comprehensive_initial_setup,
        VERSION, nickname_ledger_loaded, framework_history_loaded, playbook_loaded,
        raw_health_report, directives_loaded, raw_data_summary, raw_regime,
        load_memory(temp_config_for_paths_obj.CHAMPION_FILE_PATH, temp_config_for_paths_obj.HISTORY_FILE_PATH).get('regime_champions', {}),
        correlation_summary, MASTER_MACRO_TICKERS,
        prime_directive_str=prime_directive
    )
    
    # --- Phase 4: Final Configuration & Logging ---
    ai_initial_suggestions = _validate_and_fix_spread_config(ai_comprehensive_setup, fallback_config_dict)
    final_config_dict = fallback_config_dict.copy()
    final_config_dict.update(_sanitize_ai_suggestions(ai_initial_suggestions))

    for freq_key in ['TRAINING_WINDOW', 'RETRAINING_FREQUENCY', 'FORWARD_TEST_GAP']:
        if freq_key in final_config_dict:
            final_config_dict[freq_key] = _sanitize_frequency_string(final_config_dict[freq_key])

    version_label = f"AxiomEdge_V{VERSION}"
    config_nickname = _generate_nickname(final_config_dict.get('strategy_name', 'DefaultStrategy'),
                                          final_config_dict.get('nickname'),
                                          nickname_ledger_loaded,
                                          temp_config_for_paths_obj.NICKNAME_LEDGER_PATH,
                                          version_label)
    final_config_dict.update({
        "REPORT_LABEL": version_label,
        "nickname": config_nickname,
        "run_timestamp": run_timestamp_str
    })
    
    try:
        config = ConfigModel(**final_config_dict)
    except ValidationError as e:
        logger.critical(f"FATAL: Final configuration failed Pydantic validation: {e}")
        return {"status": "error", "message": f"Final config validation failed: {e}"}
        
    _setup_logging(config.LOG_FILE_PATH, config.REPORT_LABEL)
    _log_config_and_environment(config)
    fe = FeatureEngineer(config, tf_roles, playbook_loaded)
    
    # --- Phase 5: Walk-Forward Validation Loop ---
    all_available_engineered_features = _get_available_features_from_df(full_df)
    train_start_dates, train_end_dates, test_start_dates, test_end_dates = get_walk_forward_periods(
        full_df.index.min(), full_df.index.max(), config.TRAINING_WINDOW, config.RETRAINING_FREQUENCY, config.FORWARD_TEST_GAP
    )
    
    if not train_start_dates:
        logger.critical("Cannot proceed: No valid walk-forward periods could be generated.")
        return {"status": "error", "message": "Walk-forward period generation failed."}

    historical_cycle_results = []
    all_trades_dfs = []
    all_equity_curves_dict_by_cycle = {}
    aggregated_daily_metrics_for_report = []
    current_run_operating_state = OperatingState.CONSERVATIVE_BASELINE
    
    model_trainer = ModelTrainer(config, gemini_analyzer)
    backtester = Backtester(config)
    report_generator = PerformanceAnalyzer(config)
    
    # --- INITIALIZE THE COUNTER HERE ---
    baseline_failure_cycles = 0
    
    for cycle_num, (train_start, train_end, test_start, test_end) in enumerate(zip(train_start_dates, train_end_dates, test_start_dates, test_end_dates)):
        cycle_label = f"Cycle {cycle_num + 1}/{len(train_start_dates)}"
        logger.info(f"\n--- Starting {cycle_label} | Train: {train_start.date()}->{train_end.date()} | Test: {test_start.date()}->{test_end.date()} ---")
        cycle_start_time = time.time()
        
        cycle_specific_config_obj = config.model_copy(deep=True)
        cycle_specific_config_obj.operating_state = current_run_operating_state
        cycle_specific_config_obj = _apply_operating_state_rules(cycle_specific_config_obj)
        model_trainer.config = cycle_specific_config_obj
        backtester.config = cycle_specific_config_obj

        df_train_cycle_raw = full_df.loc[train_start:train_end].copy()
        df_test_cycle = full_df.loc[test_start:test_end].copy()
        
        if df_train_cycle_raw.empty or df_test_cycle.empty:
            logger.warning(f"{cycle_label}: Not enough data. Skipping cycle.")
            continue

        # 1. Temporarily label with default quantiles to get the raw signal pressure series
        _, raw_signal_pressure = fe.label_signal_pressure(
            df_train_cycle_raw, config.LOOKAHEAD_CANDLES, config.LABEL_LONG_QUANTILE, config.LABEL_SHORT_QUANTILE
        )
        
        # 2. Analyze the raw signal and ask the AI for optimal quantiles
        long_q, short_q = config.LABEL_LONG_QUANTILE, config.LABEL_SHORT_QUANTILE # Start with defaults
        if not raw_signal_pressure.empty:
            pressure_summary = raw_signal_pressure.describe().to_dict()
            
            # Determine the prime directive for this specific cycle
            if not historical_cycle_results or historical_cycle_results[-1].get("metrics", {}).get("NumTrades", 0) == 0:
                cycle_prime_directive = "ESTABLISH TRADING BASELINE. The last run failed to produce any trades. Your absolute priority is to configure a model that actively participates in the market. Profitability is a secondary concern to execution."
            else:
                cycle_prime_directive = "OPTIMIZE PERFORMANCE. A successful trading baseline exists. Your priority is now to improve risk-adjusted returns (e.g., MAR Ratio), profitability, and model stability."

            ai_quantiles = api_timer.call(gemini_analyzer.determine_optimal_label_quantiles, pressure_summary, cycle_prime_directive)

            if ai_quantiles:
                long_q = ai_quantiles.get('LABEL_LONG_QUANTILE', long_q)
                short_q = ai_quantiles.get('LABEL_SHORT_QUANTILE', short_q)
                logger.info(f"Applying AI-determined quantiles for this cycle: Long={long_q}, Short={short_q}")
            else:
                logger.warning("Could not get quantiles from AI. Using hardcoded defaults for this cycle.")
        
        # 3. Apply the final labels using the determined (or fallback) quantiles
        df_labeled_train_chunk, _ = fe.label_signal_pressure(
            df_train_cycle_raw, config.LOOKAHEAD_CANDLES, long_q, short_q
        )

        if df_labeled_train_chunk.empty:
            logger.warning(f"{cycle_label}: Labeling resulted in empty training data. Skipping cycle.")
            continue

        trained_model_info = None
        training_attempts = 0
        current_feature_list_for_training = all_available_engineered_features

        while training_attempts < config.MAX_TRAINING_RETRIES_PER_CYCLE and trained_model_info is None:
            training_attempts += 1
            logger.info(f"{cycle_label}: Training attempt {training_attempts}/{config.MAX_TRAINING_RETRIES_PER_CYCLE} with {len(current_feature_list_for_training)} candidate features...")
            
            trained_model_info = model_trainer.train(df_labeled_train_chunk, current_feature_list_for_training)
            
            if trained_model_info is None:
                logger.warning(f"{cycle_label}: Training attempt {training_attempts} failed.")
                if training_attempts < config.MAX_TRAINING_RETRIES_PER_CYCLE:
                    logger.info(f"{cycle_label}: Engaging AI Doctor for mid-cycle intervention...")
                    failure_hist_for_ai = [{"attempt": i + 1, "status": "failed"} for i in range(training_attempts)]
                    pre_analysis_summary = _generate_pre_analysis_summary(df_labeled_train_chunk, current_feature_list_for_training, 'target')
                    
                    ai_intervention = api_timer.call(gemini_analyzer.propose_mid_cycle_intervention,
                                                     failure_hist_for_ai,
                                                     pre_analysis_summary,
                                                     cycle_specific_config_obj.model_dump(),
                                                     playbook_loaded,
                                                     [])
                    
                    if ai_intervention and 'action' in ai_intervention:
                        logger.info(f"AI Doctor prescribed action: {ai_intervention['action']}. Notes: {ai_intervention.get('analysis_notes')}")
                        params = ai_intervention.get('parameters', {})
                        if ai_intervention['action'] == 'ADJUST_LABELING_DIFFICULTY' and params and 'LOOKAHEAD_CANDLES' in params:
                            config.LOOKAHEAD_CANDLES = params['LOOKAHEAD_CANDLES']
                            fe.config.LOOKAHEAD_CANDLES = params['LOOKAHEAD_CANDLES']
                            logger.info(f"Re-labeling data with new LOOKAHEAD_CANDLES: {config.LOOKAHEAD_CANDLES}")
                            df_labeled_train_chunk = fe.label_signal_pressure(df_train_cycle_raw, config.LOOKAHEAD_CANDLES, config.LABEL_LONG_QUANTILE, config.LABEL_SHORT_QUANTILE)
                        elif ai_intervention['action'] == 'REFINE_FEATURE_SET' and params and 'selected_features' in params:
                            current_feature_list_for_training = params['selected_features']
                            logger.info(f"Retrying with AI-refined feature set of {len(current_feature_list_for_training)} features.")
                        else:
                            logger.info("Retrying with same parameters as AI intervention was not specific enough.")
                    else:
                        logger.warning("AI Doctor failed to provide a valid intervention. Aborting cycle.")
                        break
                else:
                         logger.error(f"{cycle_label}: All {config.MAX_TRAINING_RETRIES_PER_CYCLE} training attempts failed. Skipping cycle.")

        if trained_model_info is None:
            historical_cycle_results.append({"cycle": cycle_label, "status": "Training Failed", "metrics": {}})
            continue

        pipeline, confidence_threshold, features_for_backtest = trained_model_info
        
        is_minirocket_pipeline = isinstance(pipeline, Pipeline) and 'minirocket' in pipeline.named_steps
        if is_minirocket_pipeline:
            features_for_backtest = ['Close', 'RealVolume', 'ATR']
        
        logger.info(f"{cycle_label}: Running backtest on test data with {len(features_for_backtest)} features...")
        current_equity = historical_cycle_results[-1]['metrics'].get('EndEquity', config.INITIAL_CAPITAL) if historical_cycle_results else config.INITIAL_CAPITAL
        
        final_threshold_for_backtest = config.STATIC_CONFIDENCE_GATE if config.USE_STATIC_CONFIDENCE_GATE else confidence_threshold
        logger.info(f"Using confidence threshold for backtest: {final_threshold_for_backtest:.2f} ({'Static' if config.USE_STATIC_CONFIDENCE_GATE else 'Dynamic'})")

        trades_df, equity_series, breaker_tripped, breaker_ctx, daily_metrics = backtester.run_backtest_chunk(
            df_test_cycle, 
            pipeline,
            current_equity, 
            features_for_backtest, 
            final_threshold_for_backtest,
            is_minirocket_model=is_minirocket_pipeline
        )
        
        all_trades_dfs.append(trades_df)
        all_equity_curves_dict_by_cycle[cycle_label] = equity_series
        aggregated_daily_metrics_for_report.append(daily_metrics)

        cycle_performance = report_generator._calculate_metrics(trades_df, equity_series) if not trades_df.empty else {}
        cycle_performance.update({
            'BreakerTripped': breaker_tripped,
            'BreakerContext': breaker_ctx,
            'StartEquity': current_equity,
            'EndEquity': equity_series.iloc[-1] if not equity_series.empty else current_equity,
            'NumTrades': len(trades_df)
        })

        historical_cycle_results.append({
            "cycle": cycle_label,
            "status": "Completed" if not breaker_tripped else "Breaker Tripped",
            "metrics": cycle_performance,
            "config_at_cycle_start": cycle_specific_config_obj.model_dump(),
            "selected_features_for_cycle": features_for_backtest 
        })
        logger.info(f"{cycle_label}: Completed. PNL: {cycle_performance.get('total_net_profit', 0):.2f}, Trades: {cycle_performance.get('total_trades', 0)}")
        
        current_run_operating_state = _update_operating_state(config, historical_cycle_results, current_run_operating_state)
        config.operating_state = current_run_operating_state
        
        cycle_duration = time.time() - cycle_start_time
        logger.info(f"{cycle_label} Duration: {cycle_duration:.2f} seconds.")
        
        # Check for consecutive "no-trade" cycles to trigger a strategic intervention
        if not trades_df.empty:
            baseline_failure_cycles = 0  # Reset on any trading activity
        else:
            baseline_failure_cycles += 1  # Increment the failure counter

        # If we have 2 or more consecutive failures, call the AI Doctor for a root cause analysis
        if baseline_failure_cycles >= 2:
            logger.warning(f"!! STRATEGIC INTERVENTION TRIGGERED !! {baseline_failure_cycles} consecutive cycles with no trades.")
            
            # 1. Create a clear, focused summary of the problem for the AI
            failure_hist_for_ai = [res for res in historical_cycle_results if res.get("metrics", {}).get("NumTrades", -1) == 0]
            pre_analysis_summary = (
                f"Root Cause Analysis: {len(failure_hist_for_ai)} consecutive cycles with 0 trades.\n"
                f"Diagnosis: The current strategy ('{config.strategy_name}') combined with the '{config.LABELING_METHOD}' "
                f"labeling and '{config.FEATURE_SELECTION_METHOD}' feature selection is too selective. "
                f"A fundamental change is required."
            )

            # 2. Call the AI Doctor with this specific context
            logger.info("Engaging AI Doctor for a fundamental strategy change...")
            ai_intervention = api_timer.call(gemini_analyzer.propose_mid_cycle_intervention,
                                             failure_hist_for_ai,
                                             pre_analysis_summary,
                                             config.model_dump(),
                                             playbook_loaded,
                                             []) # quarantine_list can be passed if needed

            # 3. Implement the AI's new, more focused prescription
            if ai_intervention and 'action' in ai_intervention:
                logger.info(f"AI Doctor prescribed action: {ai_intervention['action']}. Notes: {ai_intervention.get('analysis_notes')}")
                
                # CORRECTED LINE: Ensure params is a dictionary, even if the AI returns null for the key.
                params = ai_intervention.get('parameters') or {}
                
                # Use if/elif to handle different AI actions
                action_taken = False
                if ai_intervention['action'] == 'ADJUST_LABELING_DIFFICULTY' and 'LOOKAHEAD_CANDLES' in params:
                    config.LOOKAHEAD_CANDLES = params['LOOKAHEAD_CANDLES']
                    fe.config.LOOKAHEAD_CANDLES = params['LOOKAHEAD_CANDLES'] # Update feature engineer too
                    logger.warning(f"AI FIX APPLIED: LABELING LOOKAHEAD changed to: {config.LOOKAHEAD_CANDLES}")
                    action_taken = True
                
                elif ai_intervention['action'] == 'CHANGE_LABELING_METHOD' and 'LABELING_METHOD' in params:
                    # NOTE: This requires your FeatureEngineer class to have multiple labeling methods.
                    # This action may not be fully supported yet, but we handle the config change.
                    config.LABELING_METHOD = params['LABELING_METHOD']
                    logger.warning(f"AI FIX APPLIED: LABELING METHOD changed to: {config.LABELING_METHOD}")
                    action_taken = True

                elif ai_intervention['action'] == 'REFINE_FEATURE_SET' and 'selected_features' in params:
                    config.selected_features = params['selected_features']
                    logger.warning(f"AI FIX APPLIED: FEATURE SET changed. New count: {len(config.selected_features)}")
                    action_taken = True
                
                # A more general way to handle any parameter the AI suggests changing
                elif 'parameters' in ai_intervention and isinstance(params, dict):
                    for key, value in params.items():
                        if hasattr(config, key):
                            setattr(config, key, value)
                            logger.warning(f"AI FIX APPLIED: Parameter '{key}' changed to: {value}")
                            action_taken = True

                if action_taken:
                    baseline_failure_cycles = 0  # Reset the counter only if a valid action was taken
                else:
                    logger.error("AI intervention action was recognized but parameters were missing or invalid.")

            else:
                logger.error("AI Doctor intervention failed to return a valid action. The run will continue with the same strategy.")

    # --- Phase 9: Post-Walk-Forward Loop ---
    logger.info("Walk-Forward Validation Complete.")
    final_trades_df = pd.concat(all_trades_dfs, ignore_index=True) if all_trades_dfs else pd.DataFrame()
    
    final_equity_curve_list = [config.INITIAL_CAPITAL]
    if historical_cycle_results:
        for cycle_res in historical_cycle_results:
            cycle_equity = all_equity_curves_dict_by_cycle.get(cycle_res["cycle"])
            if cycle_equity is not None and not cycle_equity.empty:
                final_equity_curve_list.extend(cycle_equity.iloc[1:].tolist())

    final_equity_curve = pd.Series(final_equity_curve_list, dtype=float).drop_duplicates().reset_index(drop=True)
    if final_equity_curve.empty:
        final_equity_curve = pd.Series([config.INITIAL_CAPITAL], dtype=float)

    final_metrics = report_generator.generate_full_report(
        final_trades_df,
        final_equity_curve,
        historical_cycle_results,
        model_trainer.shap_summary,
        framework_history_loaded,
        aggregated_daily_metrics_for_report,
        model_trainer.classification_report_str
    )
    
    save_run_to_memory(config, 
                       {"final_metrics": final_metrics, 
                        "cycle_details": historical_cycle_results,
                        "config_summary": config.model_dump(exclude_defaults=True)},
                       framework_history_loaded,
                       raw_regime
                      )

    logger.info(f"Run {config.REPORT_LABEL} - {config.nickname} completed. Report: {config.REPORT_SAVE_PATH}")
    return {"status": "success", "metrics": final_metrics, "report_path": config.REPORT_SAVE_PATH}
    
def main():
    """Main entry point for the trading framework."""
    initial_log_dir = os.path.join(os.getcwd(), "Results", "PrelimLogs")
    os.makedirs(initial_log_dir, exist_ok=True)
    initial_log_path = os.path.join(initial_log_dir, f"framework_boot_{datetime.now().strftime('%Y%m%d-%H%M%S')}.log")
    _setup_logging(initial_log_path, f"AxiomEdge_V{VERSION}_Boot")
    global logger
    logger = logging.getLogger("ML_Trading_Framework")
    
    logger.info(f"--- ML Trading Framework V{VERSION} Initializing ---")

    base_config = {
        "BASE_PATH": os.getcwd(),
        "INITIAL_CAPITAL": 10000.0,
        "OPTUNA_TRIALS": 75,
        "TRAINING_WINDOW": '365D',
        "RETRAINING_FREQUENCY": '90D',
        "FORWARD_TEST_GAP": "1D",
        "LOOKAHEAD_CANDLES": 150,
        "CALCULATE_SHAP_VALUES": True,
        "USE_FEATURE_CACHING": True,
        "MAX_TRAINING_RETRIES_PER_CYCLE": 3,
        "run_timestamp": datetime.now().strftime("%Y%m%d-%H%M%S"),
        "nickname": "bootstrap",
        "REPORT_LABEL": f"AxiomEdge_V{VERSION}_Fallback",
        "strategy_name": "DefaultStrategy"       
    }

    all_files = [f for f in os.listdir(base_config['BASE_PATH']) if f.endswith(('.csv', '.txt')) and re.match(r'^[A-Z0-9]+_[A-Z0-9]+', f)]
    if not all_files:
        print("ERROR: No data files (*.csv, *.txt) found in the current directory.")
        print("Please place your data files here and ensure they are named like: 'EURUSD_H1.csv'")
        input("Press Enter to exit.")
        return
        
    symbols = sorted(list(set([f.split('_')[0] for f in all_files])))
    
    gemini_analyzer_for_setup = GeminiAnalyzer()
    api_timer_for_setup = APITimer(interval_seconds=61)
    
    asset_types = api_timer_for_setup.call(get_and_cache_asset_types, symbols, base_config, gemini_analyzer_for_setup)
    
    if not asset_types:
        logger.error("Could not determine asset types via AI.")
        print("Could not automatically classify assets. Please select the primary asset class:")
        print("1. Forex\n2. Commodities\n3. Indices\n4. Crypto")
        choice = input("Enter number: ")
        class_map = {'1': 'Forex', '2': 'Commodities', '3': 'Indices', '4': 'Crypto'}
        primary_class = class_map.get(choice, 'Forex')
        logger.info(f"Using manually selected primary class: {primary_class}")
        
        print("Please provide a minimum of three timeframes for the asset (e.g., M15, H1, D1):")
        timeframes_input = input("Enter timeframes separated by commas: ")
        timeframes = [tf.strip().upper() for tf in timeframes_input.split(',') if tf.strip()]
        
        if len(timeframes) < 3:
            print("ERROR: A minimum of three timeframes is required. Exiting.")
            input("Press Enter to exit.")
            return
        
        filtered_files = []
        for f in all_files:
            parts = f.split('_')
            if len(parts) > 1:
                tf_from_filename = parts[1].split('.')[0].upper()
                if tf_from_filename in timeframes:
                    filtered_files.append(f)
        all_files = filtered_files
        if not all_files:
            print("ERROR: No data files found matching the specified timeframes. Exiting.")
            input("Press Enter to exit.")
            return
        
        asset_types = {s: primary_class for s in symbols}
        
    else:
        from collections import Counter
        class_counts = Counter(asset_types.values())
        primary_class = class_counts.most_common(1)[0][0]

    fallback_config = generate_dynamic_config(primary_class, base_config)
    
    CONTINUOUS_RUN_HOURS = 0
    MAX_RUNS = 1
    api_interval_seconds = 61
    run_count = 0
    script_start_time = datetime.now()
    is_continuous = CONTINUOUS_RUN_HOURS > 0 or MAX_RUNS > 1
    
    temp_config_dict = fallback_config.copy()
    temp_config_dict['REPORT_LABEL'] = 'init'
    temp_config_dict['strategy_name'] = 'init'
    bootstrap_config = ConfigModel(**temp_config_dict)
    
    results_dir = os.path.join(bootstrap_config.BASE_PATH, "Results")
    os.makedirs(results_dir, exist_ok=True)
    playbook_file_path = os.path.join(results_dir, "strategy_playbook.json")
    playbook = initialize_playbook(playbook_file_path)

    while True:
        run_count += 1
        if is_continuous: logger.info(f"\n{'='*30} STARTING DAEMON RUN {run_count} {'='*30}\n")
        else: logger.info(f"\n{'='*30} STARTING SINGLE RUN {'='*30}\n")
        flush_loggers()

        nickname_ledger = load_nickname_ledger(bootstrap_config.NICKNAME_LEDGER_PATH)
        framework_history = load_memory(bootstrap_config.CHAMPION_FILE_PATH, bootstrap_config.HISTORY_FILE_PATH)
        directives = []
        if os.path.exists(bootstrap_config.DIRECTIVES_FILE_PATH):
            try:
                with open(bootstrap_config.DIRECTIVES_FILE_PATH, 'r') as f: directives = json.load(f)
                if directives: logger.info(f"Loaded {len(directives)} directive(s) for this run.")
            except (json.JSONDecodeError, IOError) as e: logger.error(f"Could not load directives file: {e}")
        
        flush_loggers()

        try:
            run_single_instance(fallback_config, framework_history, playbook, nickname_ledger, directives, api_interval_seconds)
        except Exception as e:
            logger.critical(f"A critical, unhandled error occurred during run {run_count}: {e}", exc_info=True)
            if not is_continuous: break
            logger.info("Attempting to continue after a 60-second cooldown..."); time.sleep(60)

        if not is_continuous:
            logger.info("Single run complete. Exiting.")
            break
        if MAX_RUNS > 0 and run_count >= MAX_RUNS:
            logger.info(f"Reached max run limit of {MAX_RUNS}. Exiting daemon mode.")
            break
        if CONTINUOUS_RUN_HOURS > 0 and (datetime.now() - script_start_time).total_seconds() / 3600 >= CONTINUOUS_RUN_HOURS:
            logger.info(f"Reached max runtime of {CONTINUOUS_RUN_HOURS} hours. Exiting daemon mode.")
            break

        try:
            sys.stdout.write("\n")
            for i in range(10, 0, -1):
                sys.stdout.write(f"\r>>> Run {run_count} complete. Press Ctrl+C to stop. Continuing in {i:2d} seconds..."); sys.stdout.flush(); time.sleep(1)
            sys.stdout.write("\n\n")
        except KeyboardInterrupt:
            logger.info("\n\nDaemon stopped by user. Exiting gracefully.")
            break
            
if __name__ == '__main__':
    main()        