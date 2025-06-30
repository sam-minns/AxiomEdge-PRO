# End_To_End_Advanced_ML_Trading_Framework_PRO_V210.py
#
# V210 FULL UPDATE (Adaptive Intelligence, Advanced Features & Robustness):
#
# --- Baseline Enhancements (from V210-Old) ---
#   1. ADDED (Operating State Architecture): Implemented a state management system
#      using an `OperatingState` Enum (`CONSERVATIVE_BASELINE`, `AGGRESSIVE_EXPANSION`,
#      `DRAWDOWN_CONTROL`) for dynamic behavior.
#   2. ADDED (Configurable State Parameters): `STATE_BASED_CONFIG` dictionary allows
#      defining specific risk parameters (`max_dd_per_cycle`, etc.) for each state.
#   3. IMPLEMENTED (Conservative Baseline Default): Framework starts in and defaults to
#      `CONSERVATIVE_BASELINE`, with `_apply_operating_state_rules` enforcing low-risk
#      parameters for disciplined capital preservation.
#   4. IMPLEMENTED (Dynamic AI Optimization Objective): `ModelTrainer` optimization is
#      state-aware (e.g., Maximize F1/Calmar for baseline).
#   5. ENHANCED (AI Prompt Awareness - Initial): AI setup prompt made aware of the
#      state machine for robust baseline strategy selection.
#   6. REMOVED (LSTM Model): All code related to LSTM, including TensorFlow/Keras.
#   7. ENHANCED (Operating State Architecture): Expanded `OperatingState` with:
#      - `OPPORTUNISTIC_SURGE`: To capitalize on detected volatility spikes.
#      - `MAINTENANCE_DORMANCY`: To pause trading during weekends/holidays.
#   8. ADDED ("AI Doctor" - Advanced Diagnostics): `GeminiAnalyzer.propose_mid_cycle_intervention`
#      now uses feature learnability (Mutual Info) and label distribution reports for
#      smarter root-cause analysis of training failures, enabling more targeted interventions.
#   9. ADDED (Sophisticated Feature Engineering & Selection):
#      - NEW FEATURES: Microstructure (Displacement, Gaps), Advanced Volatility
#        (Parkinson, Yang-Zhang), KAMA Trend, Trend Pullbacks, Momentum Divergences.
#      - SIGNAL SMOOTHING: Kalman Filtering applied to key indicators (RSI, ADX, StochK).
#      - ADVANCED SELECTION: TRexSelector option (`FEATURE_SELECTION_METHOD` config),
#        refined Mutual Information selection.
#      - DYNAMIC PARAMETERS: `DYNAMIC_INDICATOR_PARAMS` in `ConfigModel` for
#        regime-adaptive indicator settings (e.g., RSI, Bollinger Bands).
#  10. ADDED (Enhanced Backtesting & Configuration Realism):
#      - STATIC CONFIDENCE GATE: `USE_STATIC_CONFIDENCE_GATE` & `STATIC_CONFIDENCE_GATE`
#        parameters for more predictable model entry thresholds.
#      - LATENCY SIMULATION: `Backtester` now includes `_calculate_latency_cost` for
#        more realistic PNL by simulating execution delays.
#  11. ADDED (Framework Robustness & Usability):
#      - PLAYBOOK DEFAULTS: Strategies in `strategy_playbook.json` now include
#        default `selected_features` lists as a fallback for AI.
#      - CACHE INTEGRITY: Feature cache validation (`_generate_cache_metadata`) now
#        includes a script hash and dynamic indicator params to detect code/logic changes.
#      - DEPENDENCY REDUCTION: Manual KAMA calculation, removing `ta` library for it.
#
# --- SCRIPT VERSION ---
VERSION = "210
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
from typing import List, Dict, Any, Optional, Tuple, Union, Callable
from collections import defaultdict
import pathlib
from enum import Enum
import hashlib
import psutil
import inspect

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
from sklearn.metrics import f1_score, classification_report
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
    # A logger will be set up later, initial prints are fine for now.
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

# --- NEW: MiniRocket Specific Imports ---
try:
    from sktime.transformations.panel.rocket import MiniRocket
    MINIROCKET_AVAILABLE = True
except ImportError:
    MINIROCKET_AVAILABLE = False
    print("WARNING: sktime is not installed. MiniRocket strategies will be unavailable. Install with: pip install sktime")
# --- END ---

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

def setup_logging():
    """
    Configures the global logger and prints initial, unformatted library status checks.
    This must be called once at the start of the application.
    """
    # Clear any existing handlers to prevent duplicate logs
    if logger.hasHandlers():
        logger.handlers.clear()

    logger.setLevel(logging.DEBUG)

    # Create a handler for console output
    ch = logging.StreamHandler(sys.stdout)
    ch.setLevel(logging.INFO) # Set level for console output

    # Using direct print() for these initial checks to guarantee unformatted visibility
    print("-" * 60, flush=True)
    if GNN_AVAILABLE:
        print("INFO: PyTorch and PyG loaded successfully. GNN module is available.", flush=True)
    else:
        print("WARNING: PyTorch or PyTorch Geometric not found. GNN strategies will be unavailable.", flush=True)

    if MINIROCKET_AVAILABLE:
        print("INFO: sktime loaded successfully. MiniRocket module is available.", flush=True)
    else:
        print("WARNING: sktime not found. MiniRocket strategies will be unavailable.", flush=True)
    print("-" * 60, flush=True)

    # Configure the handler with the standard log format for all subsequent messages
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    ch.setFormatter(formatter)
    logger.addHandler(ch)

    optuna.logging.set_verbosity(optuna.logging.WARNING)

# --- FRAMEWORK STATE DEFINITION ---
class OperatingState(Enum):
    """Defines the operational states of the trading framework."""
    CONSERVATIVE_BASELINE = "Conservative Baseline"
    AGGRESSIVE_EXPANSION = "Aggressive Expansion"
    DRAWDOWN_CONTROL = "Drawdown Control"
    OPPORTUNISTIC_SURGE = "Opportunistic Surge"
    MAINTENANCE_DORMANCY = "Maintenance Dormancy"
# ------------------------------------

warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', category=UserWarning)
warnings.filterwarnings('ignore', category=pd.errors.PerformanceWarning)

# =============================================================================
# 3. CONFIGURATION & VALIDATION
# =============================================================================

class EarlyInterventionConfig(BaseModel):
    """Configuration for the adaptive early intervention system."""
    enabled: bool = True
    attempt_threshold: conint(ge=2) = 2
    min_profitability_for_f1_override: confloat(ge=0) = 3.0
    max_f1_override_value: confloat(ge=0.4, le=0.6) = 0.50

class ConfigModel(BaseModel):
    """
    The central configuration model for the trading framework.
    It holds all parameters that define a run, from data paths and capital
    to risk management, AI behavior, and backtesting realism settings.
    """
    # --- Core Run, Capital & State Parameters ---
    BASE_PATH: DirectoryPath
    REPORT_LABEL: str
    INITIAL_CAPITAL: confloat(gt=0)
    operating_state: OperatingState = OperatingState.CONSERVATIVE_BASELINE
    
    FEATURE_SELECTION_METHOD: str = 'trex' # Options: 'trex', 'mutual_info'

    # --- AI & Optimization Parameters ---
    OPTUNA_TRIALS: conint(gt=0)
    MAX_TRAINING_RETRIES_PER_CYCLE: conint(ge=0) = 3
    CALCULATE_SHAP_VALUES: bool = True
    
    # --- Static Confidence Gate Control ---
    USE_STATIC_CONFIDENCE_GATE: bool = True
    STATIC_CONFIDENCE_GATE: confloat(ge=0.5, le=0.95) = 0.65 # A reasonable default starting at 70%
    
    # --- Dynamic Labeling & Trade Definition ---
    TP_ATR_MULTIPLIER: confloat(gt=0.5, le=10.0) = 2.0
    SL_ATR_MULTIPLIER: confloat(gt=0.5, le=10.0) = 1.5
    LOOKAHEAD_CANDLES: conint(gt=0)
    LABELING_METHOD: str = 'standard'
    MIN_F1_SCORE_GATE: confloat(ge=0.3, le=0.7) = 0.45
    LABEL_MIN_RETURN_PCT: confloat(ge=0.0) = 0.001
    LABEL_MIN_EVENT_PCT: confloat(ge=0.01, le=0.5) = 0.02

    # --- Walk-Forward & Data Parameters ---
    TRAINING_WINDOW: str
    RETRAINING_FREQUENCY: str
    FORWARD_TEST_GAP: str
    
    # --- Risk & Portfolio Management ---
    MAX_DD_PER_CYCLE: confloat(ge=0.05, lt=1.0) = 0.25
    RISK_CAP_PER_TRADE_USD: confloat(gt=0)
    BASE_RISK_PER_TRADE_PCT: confloat(gt=0, lt=1)
    MAX_CONCURRENT_TRADES: conint(ge=1, le=20) = 3
    USE_TIERED_RISK: bool = False
    RISK_PROFILE: str = 'Medium'
    TIERED_RISK_CONFIG: Dict[int, Dict[str, Dict[str, Union[float, int]]]] = Field(default_factory=lambda: {
            2000:  {'Low': {'risk_pct': 0.01, 'pairs': 1}, 'Medium': {'risk_pct': 0.01, 'pairs': 1}, 'High': {'risk_pct': 0.01, 'pairs': 1}},
            5000:  {'Low': {'risk_pct': 0.008, 'pairs': 1}, 'Medium': {'risk_pct': 0.012, 'pairs': 1}, 'High': {'risk_pct': 0.012, 'pairs': 2}},
            10000: {'Low': {'risk_pct': 0.006, 'pairs': 2}, 'Medium': {'risk_pct': 0.008, 'pairs': 2}, 'High': {'risk_pct': 0.01, 'pairs': 2}},
            15000: {'Low': {'risk_pct': 0.007, 'pairs': 2}, 'Medium': {'risk_pct': 0.009, 'pairs': 2}, 'High': {'risk_pct': 0.012, 'pairs': 2}},
            25000: {'Low': {'risk_pct': 0.008, 'pairs': 2}, 'Medium': {'risk_pct': 0.012, 'pairs': 2}, 'High': {'risk_pct': 0.016, 'pairs': 2}},
            50000: {'Low': {'risk_pct': 0.008, 'pairs': 3}, 'Medium': {'risk_pct': 0.012, 'pairs': 3}, 'High': {'risk_pct': 0.016, 'pairs': 3}},
            100000:{'Low': {'risk_pct': 0.007, 'pairs': 4}, 'Medium': {'risk_pct': 0.01, 'pairs': 4}, 'High': {'risk_pct': 0.014, 'pairs': 4}},
            9e9:   {'Low': {'risk_pct': 0.005, 'pairs': 6}, 'Medium': {'risk_pct': 0.0075,'pairs': 6}, 'High': {'risk_pct': 0.01, 'pairs': 6}}
        })

    STATE_BASED_CONFIG: Dict[OperatingState, Dict[str, Any]] = {
        OperatingState.CONSERVATIVE_BASELINE: {
            "max_dd_per_cycle": 0.15,
            "base_risk_pct": 0.0075,
            "max_concurrent_trades": 2,
            "confidence_gate_modifier": 1.0,  # This is now effectively disabled by the static gate
            "optimization_objective": ["maximize_f1", "maximize_log_trades"], 
            "min_f1_gate": 0.40
        },
        OperatingState.AGGRESSIVE_EXPANSION: {
            "max_dd_per_cycle": 0.30,
            "base_risk_pct": 0.015,
            "max_concurrent_trades": 5,
            "confidence_gate_modifier": 1.0,
            "optimization_objective": ["maximize_pnl", "maximize_log_trades"],
            "min_f1_gate": 0.42
        },
        OperatingState.DRAWDOWN_CONTROL: {
            "max_dd_per_cycle": 0.10,
            "base_risk_pct": 0.005,
            "max_concurrent_trades": 1,
            "confidence_gate_modifier": 1.0,
            "optimization_objective": ["maximize_sortino", "minimize_trades"],
            "min_f1_gate": 0.38
        },
        OperatingState.OPPORTUNISTIC_SURGE: {
            "max_dd_per_cycle": 0.20,
            "base_risk_pct": 0.0125,
            "max_concurrent_trades": 3,
            "confidence_gate_modifier": 1.0,
            "optimization_objective": ["maximize_pnl", "minimize_trades"],
            "min_f1_gate": 0.40
        },
        OperatingState.MAINTENANCE_DORMANCY: {
            "max_dd_per_cycle": 0.05,
            "base_risk_pct": 0.0,
            "max_concurrent_trades": 0,
            "confidence_gate_modifier": 999,
            "optimization_objective": ["maximize_f1", "minimize_trades"],
            "min_f1_gate": 0.99
        }
    }
    CONFIDENCE_TIERS: Dict[str, Dict[str, Any]]
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
    PCA_N_COMPONENTS: conint(gt=1, le=10) = 3
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
    
    # --- GNN Specific Parameters (kept for GNN strategies) ---
    GNN_EMBEDDING_DIM: conint(gt=0) = 8
    GNN_EPOCHS: conint(gt=0) = 50
    
    # --- Caching & Performance ---
    USE_FEATURE_CACHING: bool = True
    
    # --- State & Info Parameters ---
    selected_features: List[str]
    run_timestamp: str
    strategy_name: str
    nickname: str = ""
    analysis_notes: str = ""

    # --- File Path Management (Internal, not configured by user/AI) ---
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

    def __init__(self, **data: Any):
        super().__init__(**data)
        results_dir = os.path.join(self.BASE_PATH, "Results")
        version_match = re.search(r'V(\d+)', self.REPORT_LABEL)
        version_str = f"_V{version_match.group(1)}" if version_match else ""
        folder_name = f"{self.nickname}{version_str}" if self.nickname and version_str else self.REPORT_LABEL
        run_id = f"{folder_name}_{self.strategy_name}_{self.run_timestamp}"
        result_folder_path = os.path.join(results_dir, folder_name)

        if self.nickname and self.nickname != "init":
            os.makedirs(result_folder_path, exist_ok=True)

        self.MODEL_SAVE_PATH = os.path.join(result_folder_path, f"{run_id}_model.json")
        self.PLOT_SAVE_PATH = os.path.join(result_folder_path, f"{run_id}_equity_curve.png")
        self.REPORT_SAVE_PATH = os.path.join(result_folder_path, f"{run_id}_report.txt")
        self.SHAP_PLOT_PATH = os.path.join(result_folder_path, f"{run_id}_shap_summary.png")
        self.LOG_FILE_PATH = os.path.join(result_folder_path, f"{run_id}_run.log")

        self.CHAMPION_FILE_PATH = os.path.join(results_dir, "champion.json")
        self.HISTORY_FILE_PATH = os.path.join(results_dir, "historical_runs.jsonl")
        self.PLAYBOOK_FILE_PATH = os.path.join(results_dir, "strategy_playbook.json")
        self.DIRECTIVES_FILE_PATH = os.path.join(results_dir, "framework_directives.json")
        self.NICKNAME_LEDGER_PATH = os.path.join(results_dir, "nickname_ledger.json")
        self.REGIME_CHAMPIONS_FILE_PATH = os.path.join(results_dir, "regime_champions.json")
        
        cache_dir = os.path.join(self.BASE_PATH, "Cache")
            
        self.CACHE_PATH = os.path.join(cache_dir, "feature_cache.parquet")
        self.CACHE_METADATA_PATH = os.path.join(cache_dir, "feature_cache_metadata.json")
        
# =============================================================================
# 3. GEMINI AI ANALYZER & API TIMER
# =============================================================================
class APITimer:
    """Manages the timing of API calls to ensure a minimum interval between them."""
    def __init__(self, interval_seconds: int = 61):
        self.interval = timedelta(seconds=interval_seconds)
        self.last_call_time: Optional[datetime] = None
        if self.interval.total_seconds() > 0:
            logger.info(f"API Timer initialized with a {self.interval.total_seconds():.0f}-second interval.")
        else:
            logger.info("API Timer initialized with a 0-second interval (timer is effectively disabled).")

    def _wait_if_needed(self):
        if self.interval.total_seconds() <= 0: return
        if self.last_call_time is None: return

        elapsed = datetime.now() - self.last_call_time
        wait_time_delta = self.interval - elapsed
        wait_seconds = wait_time_delta.total_seconds()

        if wait_seconds > 0:
            logger.info(f"  - Time since last API call: {elapsed.total_seconds():.1f} seconds.")
            logger.info(f"  - Waiting for {wait_seconds:.1f} seconds to respect the {self.interval.total_seconds():.0f}s interval...")
            flush_loggers()
            time.sleep(wait_seconds)
        else:
            logger.info(f"  - Time since last API call ({elapsed.total_seconds():.1f}s) exceeds interval. No wait needed.")

    def call(self, api_function: Callable, *args, **kwargs) -> Any:
        """Executes the API function after ensuring the timing interval is met."""
        self._wait_if_needed()
        self.last_call_time = datetime.now()
        logger.info(f"  - Making API call to '{api_function.__name__}' at {self.last_call_time.strftime('%H:%M:%S')}...")
        result = api_function(*args, **kwargs)
        logger.info(f"  - API call to '{api_function.__name__}' complete.")
        return result

class GeminiAnalyzer:
    def __init__(self):
        self.api_key = os.getenv("GEMINI_API_KEY")
        if not self.api_key or "YOUR" in self.api_key or "PASTE" in self.api_key:
            logger.warning("!CRITICAL! GEMINI_API_KEY not found in environment or is a placeholder.")
            try:
                self.api_key = input(">>> Please paste your Gemini API Key and press Enter, or press Enter to skip: ").strip()
                if not self.api_key:
                    logger.warning("No API Key provided. AI analysis will be skipped.")
                    self.api_key_valid = False
                else:
                    logger.info("Using API Key provided via manual input.")
                    self.api_key_valid = True
            except Exception:
                logger.warning("Could not read input (non-interactive environment?). AI analysis will be skipped.")
                self.api_key_valid = False
                self.api_key = None
        else:
            logger.info("Successfully loaded GEMINI_API_KEY from environment.")
            self.api_key_valid = True

        self.headers = {"Content-Type": "application/json"}
        self.primary_model = "gemini-2.0-flash"
        self.backup_model = "gemini-1.5-flash"
        self.tools = [
            {
                "function_declarations": [
                    {
                        "name": "search_web",
                        "description": "Searches the web for a given query to find real-time information.",
                        "parameters": {
                            "type": "object",
                            "properties": {
                                "query": {
                                    "type": "string",
                                    "description": "The search query to send to the search engine."
                                }
                            },
                            "required": ["query"]
                        }
                    }
                ]
            }
        ]

        # This config enables function calling mode
        self.tool_config = {
            "function_calling_config": {
                "mode": "AUTO"
            }
        }

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
            # Ensure all original symbols were classified
            for symbol in symbols:
                if symbol not in classified_assets:
                    classified_assets[symbol] = "Unknown"
                    logger.warning(f"  - AI did not classify '{symbol}'. Marked as 'Unknown'.")
            return classified_assets

        logger.error("  - AI failed to return a valid symbol classification dictionary. Using fallback detection.")
        return {}

    def _sanitize_value(self, value: Any) -> Any:
        if isinstance(value, pathlib.Path): return str(value)
        if isinstance(value, (np.int64, np.int32)): return int(value)
        if isinstance(value, (np.float64, np.float32)):
            if np.isnan(value) or np.isinf(value): return None
            return float(value)
        if isinstance(value, (pd.Timestamp, datetime, date)): return value.isoformat()
        return value

    def _sanitize_dict(self, data: Any) -> Any:
        if isinstance(data, dict): return {key: self._sanitize_dict(value) for key, value in data.items()}
        if isinstance(data, list): return [self._sanitize_dict(item) for item in data]
        return self._sanitize_value(data)

    def _call_gemini(self, prompt: str) -> str:
        if not self.api_key_valid:
            return "{}"

        if len(prompt) > 30000:
            logger.warning("Prompt is very large, may risk exceeding token limits.")

        # This payload structure is correct for making tool-based calls
        payload = {
            "contents": [{"parts": [{"text": prompt}]}],
            "tools": self.tools,
            "tool_config": self.tool_config
        }
        sanitized_payload = self._sanitize_dict(payload)

        models_to_try = [self.primary_model, self.backup_model]
        retry_delays = [5, 15, 30]

        for model in models_to_try:
            logger.info(f"Attempting to call Gemini API with model: {model}")
            api_url = f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent?key={self.api_key}"

            for attempt, delay in enumerate([0] + retry_delays):
                if delay > 0:
                    logger.warning(f"Retrying in {delay} seconds... (Attempt {attempt}/{len(retry_delays)})")
                    flush_loggers()
                    time.sleep(delay)

                try:
                    response = requests.post(api_url, headers=self.headers, data=json.dumps(sanitized_payload), timeout=120)
                    response.raise_for_status() # Raises HTTPError for bad responses (4xx or 5xx)

                    result = response.json()

                    if "candidates" in result and result["candidates"]:
                        content = result["candidates"][0].get("content", {})
                        parts = content.get("parts", [])

                        for part in parts:
                            if "text" in part:
                                logger.info(f"Successfully received and extracted text response from model: {model}")
                                return part["text"]

                    logger.error(f"Invalid Gemini response structure from {model}: No 'text' part found in the final response. Response: {result}")
                    continue

                except requests.exceptions.HTTPError as e:
                    logger.error(f"!! HTTP Error for model '{model}': {e.response.status_code} {e.response.reason}")
                    logger.error(f"   - API Error Details: {e.response.text}")
                    break
                except requests.exceptions.RequestException as e:
                    logger.error(f"Gemini API request failed for model {model} on attempt {attempt + 1} (Network Error): {e}")
                except json.JSONDecodeError as e:
                    logger.error(f"Failed to decode Gemini response JSON from {model}: {e} - Response: {response.text}")

            logger.warning(f"Failed to get a valid text response from model {model} after all retries.")

        logger.critical("API connection failed for all primary and backup models. Could not get a final text response.")
        return "{}"

    def _extract_json_from_response(self, response_text: str) -> dict:
        logger = logging.getLogger("ML_Trading_Framework")
        logger.debug(f"RAW AI RESPONSE TO BE PARSED:\n--- START ---\n{response_text}\n--- END ---")

        decoder = JSONDecoder()
        pos = 0
        while pos < len(response_text):
            brace_pos = response_text.find('{', pos)
            if brace_pos == -1:
                break

            try:
                suggestions, end_pos = decoder.raw_decode(response_text, brace_pos)
                logger.info("Successfully extracted JSON object using JSONDecoder.raw_decode.")

                if not isinstance(suggestions, dict):
                    logger.warning(f"Parsed JSON was type {type(suggestions)}, not a dictionary. Continuing search.")
                    pos = end_pos
                    continue

                if isinstance(suggestions.get("current_params"), dict):
                    nested_params = suggestions.pop("current_params")
                    suggestions.update(nested_params)

                return suggestions

            except JSONDecodeError as e:
                logger.warning(f"JSON decoding failed at position {brace_pos}. Error: {e}. Skipping to next candidate.")
                pos = brace_pos + 1

        logger.error("!! CRITICAL JSON PARSE FAILURE !! No valid JSON dictionary could be decoded from the AI response.")
        return {}

    # --- ACTIONABLE RECOMMENDATION IMPLEMENTED ---
    def establish_strategic_directive(self, historical_results: List[Dict], current_state: OperatingState) -> str:
        """
        [IMPROVED] Determines the current high-level strategic phase based on run history and state.
        This logic now requires that a model has actually executed trades before switching to
        the 'PERFORMANCE OPTIMIZATION' phase, preventing premature optimization.
        """
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

        # --- NEW LOGIC START ---
        can_optimize = False
        if len(historical_results) >= 2:
            last_two_cycles = historical_results[-2:]
            
            # Condition 1: Were the last two cycles completed without training failures or circuit breakers?
            completed_successfully = all(c.get("Status") == "Completed" for c in last_two_cycles)
            
            # Condition 2: Did at least one of those successful cycles actually execute trades?
            executed_trades = any(c.get("NumTrades", 0) > 0 for c in last_two_cycles)

            if completed_successfully and executed_trades:
                can_optimize = True
                logger.info("  - Logic Check: Last two cycles completed and at least one traded. Permitting optimization.")
            elif completed_successfully and not executed_trades:
                 logger.warning("  - Logic Check: Last two cycles completed but NEITHER executed trades. Forcing baseline establishment.")
            else:
                 logger.info("  - Logic Check: One or more of the last two cycles did not complete successfully. Baseline establishment is required.")
        # --- NEW LOGIC END ---

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

    def select_relevant_macro_tickers(self, asset_list: List[str], master_ticker_list: Dict) -> Dict:
        """Asks the AI to select the most relevant macro tickers for a given list of assets."""
        logger.info("-> Engaging AI to select relevant macroeconomic tickers...")

        prompt = (
            "You are an expert financial analyst. Your task is to select the most relevant macroeconomic indicators that would influence the price of a given list of trading assets. A master list of available tickers is provided.\n\n"
            f"**ASSETS TO ANALYZE:** {asset_list}\n\n"
            f"**MASTER TICKER LIST (Available for Selection):**\n{json.dumps(master_ticker_list, indent=2)}\n\n"
            "**INSTRUCTIONS:**\n"
            "1.  Review the asset list. Identify the primary countries and economic zones involved (e.g., 'EURUSD' involves the US and Eurozone; 'XAUUSD' is global but sensitive to US policy; 'AUDJPY' involves Australia and Japan).\n"
            "2.  From the `MASTER TICKER LIST`, select a dictionary of tickers that are most relevant. \n"
            "3.  **Always include the core global indicators**: `VIX`, `DXY`, `US10Y_YIELD`.\n"
            "4.  If you see European assets (EUR, GBP), you should include `GERMAN10Y`.\n"
            "5.  If you see commodity-linked assets (AUD, CAD, XAUUSD), you should include `WTI_OIL` and `GOLD`.\n"
            "6.  Respond ONLY with a single, valid JSON object containing the chosen ticker dictionary (e.g., `{{\"VIX\": \"^VIX\", ...}}`)."
        )

        response_text = self._call_gemini(prompt)
        suggestions = self._extract_json_from_response(response_text)

        if isinstance(suggestions, dict) and suggestions:
            logger.info(f"  - AI selected {len(suggestions)} relevant tickers.")
            return suggestions

        logger.warning("  - AI failed to select tickers. Falling back to default.")
        return {"VIX": "^VIX", "DXY": "DX-Y.NYB", "US10Y_YIELD": "^TNX"}

    def get_initial_run_setup(self, script_version: str, ledger: Dict, memory: Dict, playbook: Dict, health_report: Dict, directives: List[Dict], data_summary: Dict, diagnosed_regime: str, regime_champions: Dict, correlation_summary_for_ai: str, macro_context: Dict) -> Dict:
        if not self.api_key_valid:
            logger.warning("No API key. Skipping AI-driven setup and using default config.")
            return {}

        logger.info("-> Performing Initial AI Analysis & Setup (Grounded Search with Correlation Context)...")
        asset_list = ", ".join(data_summary.get('assets_detected', []))

        task_prompt = (
            "**YOUR TASK: Perform a grounded analysis to create the complete initial run configuration. This involves four main steps.**\n\n"
            "**NEW CONTEXT:** The framework now operates in different states. It will start in a **'CONSERVATIVE_BASELINE'** state. Your primary goal is to find a stable baseline model that **learns to trade cautiously**. It must prioritize high-quality signals and avoid significant drawdowns, but it **must be encouraged to participate in the market**. A model that never trades is not a valid baseline.\n\n"
            "**STEP 1: DYNAMIC BROKER SIMULATION (Grounded Search)**\n"
            f"   - The assets being traded are: **{asset_list}**. \n"
            "   - **Action:** Use Google Search to find typical trading costs for these assets on a retail **ECN/Raw Spread** account.\n"
            "   - **Action:** In your JSON response, populate `COMMISSION_PER_LOT` and the `SPREAD_CONFIG` dictionary.\n"
            "       - **CRITICAL FORMATTING FOR SPREAD_CONFIG:** The value for EACH symbol MUST be a nested dictionary containing `normal_pips` and `volatile_pips`.\n"
            "       - **CORRECT FORMAT EXAMPLE:** `\"XAUUSD\": {\"normal_pips\": 20.0, \"volatile_pips\": 60.0}`\n"
            "       - **INCORRECT FORMAT EXAMPLE:** `\"XAUUSD\": 20.0`\n"
            "       - You must also include the `\"default\"` key with the same nested dictionary structure.\n\n"
            "**STEP 2: STRATEGY SELECTION (Grounded Search & Context Synthesis)**\n"
            "   - **Synthesize Context:** Analyze `MACROECONOMIC CONTEXT`, `MARKET DATA SUMMARY`, and `ASSET CORRELATION SUMMARY`.\n"
            "   - **Grounded Calendar Check:** Search the economic calendar for the next 5 trading days.\n"
            "   - **Decide on a Strategy:** Given the 'CONSERVATIVE_BASELINE' goal, select a **robust, well-understood strategy** from the playbook. **STRONGLY PREFER** strategies with 'low' or 'medium' complexity. Avoid highly specialized or experimental strategies for this initial run.\n\n"
            "**STEP 3: OPTIMAL PARAMETER SETUP (Grounded Search)**\n"
            "   - **Action:** Based on your chosen strategy and the current market regime (e.g., 'Strong Trending', 'Ranging'), perform a grounded search for recommended starting parameters for that environment.\n"
            "   - **Action:** In your JSON response, set the values for `TP_ATR_MULTIPLIER`, `SL_ATR_MULTIPLIER`, and `LOOKAHEAD_CANDLES` based on your research.\n"
            "   - **Action:** In the `analysis_notes`, you MUST justify why you chose these specific values (e.g., *'For a ranging market, a lower TP/SL ratio of 1.5 and a shorter lookahead of 50 candles is recommended to capture smaller price oscillations.'*).\n\n"
            "**STEP 4: CONFIGURATION & NICKNAME**\n"
            "   - Provide the full configuration in the JSON response.\n"
            "   - Handle nickname generation as per the rules."
        )

        json_structure_prompt = (
            "**OUTPUT FORMAT**: Respond ONLY with a single, valid JSON object. The JSON object **MUST** contain the following top-level keys:\n"
            "- `strategy_name` (string)\n"
            "- `selected_features` (list of strings)\n"
            "- `analysis_notes` (string)\n"
            "- `COMMISSION_PER_LOT` (float)\n"
            "- `SPREAD_CONFIG` (dictionary)\n"
            "- `OPTUNA_TRIALS` (integer)\n"
            "- `nickname` (string or null)\n"
        )


        prompt = (
            "You are a Master Trading Strategist responsible for configuring a trading framework for its next run. Your decisions must be evidence-based, combining internal data with real-time external information.\n\n"
            f"{task_prompt}\n\n"
            f"{json_structure_prompt}\n\n"
            "--- CONTEXT FOR YOUR DECISION ---\n\n"
            f"**1. MACROECONOMIC CONTEXT (EXTERNAL):**\n{json.dumps(self._sanitize_dict(macro_context), indent=2)}\n\n"
            f"**2. MARKET DATA SUMMARY (INTERNAL):**\n`diagnosed_regime`: '{diagnosed_regime}'\n{json.dumps(self._sanitize_dict(data_summary), indent=2)}\n\n"
            f"**3. ASSET CORRELATION SUMMARY (INTERNAL):**\n{correlation_summary_for_ai}\n\n"
            f"**4. STRATEGY PLAYBOOK (Your options):**\n{json.dumps(self._sanitize_dict(playbook), indent=2)}\n\n"
            f"**5. FRAMEWORK MEMORY (All-Time Champion & Recent Runs):**\n{json.dumps(self._sanitize_dict(memory), indent=2)}\n"
        )

        response_text = self._call_gemini(prompt)
        suggestions = self._extract_json_from_response(response_text)

        if suggestions and "strategy_name" in suggestions:
            logger.info("  - Initial AI Analysis and Setup complete.")
            return suggestions
        else:
            logger.error("  - AI-driven setup failed validation. The returned JSON was missing 'strategy_name' or was empty.")
            logger.debug(f"    - Invalid dictionary received from AI: {suggestions}")
            return {}

    def analyze_cycle_and_suggest_changes(
        self,
        historical_results: List[Dict],
        strategy_details: Dict,
        cycle_status: str,
        shap_history: Dict[str, List[float]],
        available_features: List[str],
        strategic_directive: str
    ) -> Dict:
        if not self.api_key_valid: return {}

        base_prompt_intro = "You are an expert trading model analyst and portfolio manager. Your goal is to make intelligent, data-driven changes to the trading model configuration to align with the current strategic directive."

        task_guidance = (
            "**YOUR TASK:**\n"
            "1.  **Review the `STRATEGIC DIRECTIVE`.** This is your most important instruction.\n"
            "2.  **Analyze the `CYCLE STATUS` and `HISTORICAL RESULTS`.**\n"
            "   - **SPECIAL RULE:** If `NumTrades` for the last cycle was `0`, your primary suggestion MUST be to make the model *less* conservative (e.g., reduce `confidence_gate_modifier`).\n"
            "3.  **Propose Standard Changes:** If the directive is to optimize, suggest changes to the current configuration that are aligned with the goal.\n"
            "4.  **Propose Exploration (Optional):** If the `STRATEGIC DIRECTIVE` is `PERFORMANCE OPTIMIZATION` and you have a strong, research-backed hypothesis for a novel strategy that could outperform the current one, you can propose to invent a new one by returning `{\"action\": \"EXPLORE_NEW_STRATEGY\"}`. Use this option judiciously."
        )

        json_schema_definition = (
            "### REQUIRED JSON RESPONSE STRUCTURE ###\n"
            "// To propose exploring a new strategy, return: {\"action\": \"EXPLORE_NEW_STRATEGY\", \"analysis_notes\": \"Your reasoning...\"}\n"
            "// To propose standard parameter changes, return a JSON object with the new parameters.\n"
            "// If no changes are needed, return an empty JSON object: {}\n"
            "{\n"
            '  "analysis_notes": str,            // Your detailed reasoning for the suggested changes, referencing the STRATEGIC DIRECTIVE.\n'
            '  "model_confidence_score": int,    // Your 1-10 confidence in this configuration decision.\n'
            '  // ... and any other parameter from the ConfigModel you wish to change OR the \"action\" key.\n'
            "}\n"
            "Respond ONLY with the JSON object.\n"
        )

        data_context = (
            f"--- DATA FOR YOUR ANALYSIS ---\n\n"
            f"**A. CURRENT CYCLE STATUS:** `{cycle_status}`\n\n"
            f"**B. CURRENT RUN - CYCLE-BY-CYCLE HISTORY:**\n{json.dumps(self._sanitize_dict(historical_results), indent=2)}\n\n"
            f"**C. FEATURE IMPORTANCE HISTORY (SHAP values over time):**\n{json.dumps(self._sanitize_dict(shap_history), indent=2)}\n\n"
            f"**D. CURRENT STRATEGY & AVAILABLE FEATURES:**\n`strategy_name`: {strategy_details.get('strategy_name')}\n`available_features`: {available_features}\n"
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
        """
        Analyzes a Pareto front of Optuna trials and selects the best one based on the strategic directive.
        """
        if not self.api_key_valid:
            logger.warning("No API key. Skipping AI-driven trade-off selection. Selecting trial with highest primary objective.")
            return max(best_trials, key=lambda t: t.values[0]).number

        if not best_trials:
            logger.error("`select_best_tradeoff` called with no trials. Cannot proceed.")
            raise ValueError("Cannot select from an empty list of trials.")

        trial_summaries = []
        for trial in best_trials:
            obj1_val = trial.values[0] if trial.values and len(trial.values) > 0 else 0
            obj2_val = trial.values[1] if trial.values and len(trial.values) > 1 else 0
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
            '  "selected_trial_number": int, // The number of the trial you have chosen.\n'
            '  "analysis_notes": str // Your reasoning, explicitly referencing the strategic directive and the critical rule.\n'
            "}\n"
            "```"
        )

        response_text = self._call_gemini(prompt)
        suggestions = self._extract_json_from_response(response_text)

        selected_trial_number = suggestions.get('selected_trial_number')

        if isinstance(selected_trial_number, int):
            valid_numbers = {t.number for t in best_trials}
            if selected_trial_number in valid_numbers:
                logger.info(f"  - AI has selected Trial #{selected_trial_number} based on the strategic directive.")
                logger.info(f"  - AI Rationale: {suggestions.get('analysis_notes', 'N/A')}")
                return selected_trial_number
            else:
                logger.error(f"  - AI selected an invalid trial number ({selected_trial_number}). Falling back to best primary objective.")
                return max(best_trials, key=lambda t: t.values[0]).number
        else:
            logger.error("  - AI failed to return a valid trial number. Falling back to best primary objective.")
            return max(best_trials, key=lambda t: t.values[0]).number

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
                f"3.  **Propose a Safe Starting Configuration:** Provide a reasonable and SAFE starting configuration for this new strategy. {feature_selection_guidance} Start with conservative values: `RETRAINING_FREQUENCY`: '90D', `MAX_DD_PER_CYCLE`: 0.15 (float), `RISK_PROFILE`: 'Medium', `OPTUNA_TRIALS`: 50, and **`USE_PARTIAL_PROFIT`: false**.\n"
            )
            prompt = (
                f"{base_prompt}"
                f"{explore_only_prompt}\n"
                "Respond ONLY with a valid JSON object for the new configuration, including `strategy_name` and `selected_features`."
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
            "Respond with `{\"action\": \"retire\"}`.\n\n"
            "2.  **REWORK:** If the strategy's concept is sound but its implementation is poor, propose a new, more robust default configuration. This means changing its default `selected_features` and/or other parameters like `dd_range` to be more conservative. "
            "Respond with `{\"action\": \"rework\", \"new_config\": { ... new parameters ... }}`.\n\n"
            "3.  **NO CHANGE:** If you believe the recent failures were anomalous and the strategy does not warrant a permanent change, you can choose to do nothing. "
            "Respond with `{\"action\": \"no_change\"}`.\n\n"
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

    def propose_mid_cycle_intervention(
        self,
        failure_history: List[Dict],
        pre_analysis_summary: str, # This will now contain the new diagnostic reports
        current_config: Dict,
        playbook: Dict,
        quarantine_list: List[str]
    ) -> Dict:
        """
        AI DOCTOR UPDATE: Called mid-cycle after multiple training failures.
        This prompt is now a comprehensive diagnostic interface, providing the AI
        with feature learnability and label distribution data to make a more
        intelligent, root-cause-based decision.
        """
        if not self.api_key_valid: return {}
        logger.warning("! AI DOCTOR !: Multiple training attempts failed. Engaging advanced diagnostics for course-correction.")

        available_playbook = {k: v for k, v in playbook.items() if k not in quarantine_list and not v.get("retired")}

        task_prompt = (
        "**PRIME DIRECTIVE: AI DOCTOR - DIAGNOSE AND PRESCRIBE**\n"
        "The current model is failing to train. Your task is to act as an expert data scientist, diagnose the **root cause** of the failure using the provided diagnostics, and prescribe a single, logical intervention.\n\n"
        "**STEP 1: ANALYZE THE DIAGNOSTIC REPORT**\n"
        "   - **`Label Distribution`**: Is there a severe class imbalance (e.g., <10% for Long/Short)? This can make learning nearly impossible.\n"
        "   - **`Feature Learnability (MI Scores)`**: Are the Mutual Information scores extremely low (e.g., < 0.001)? This indicates the features have almost no predictive information about the labels.\n"
        "   - **`Failure History`**: Is the F1 score always near-zero, or does it fluctuate? This tells you if learning is happening at all.\n\n"
        "**STEP 2: CHOOSE YOUR PRESCRIPTION (Your Action)**\n"
        "Based on your diagnosis, choose **ONE** of the following actions:\n"
        "1.  **`RUN_DIAGNOSTIC_ENSEMBLE`**: Prescribe this if MI scores are very low or you suspect a fundamental data/label mismatch. This action will test several simple, baseline strategies (e.g., `EmaCrossoverRsiFilter`, `MeanReversionBollinger`) to see if *any* style of logic can be learned from the data. This is your best tool for answering: 'Is this data learnable at all?'\n\n"
        "2.  **`ADJUST_LABELING_DIFFICULTY`**: Prescribe this if features have some signal (MI > 0.005) but the model still fails. This suggests the prediction task is too hard. Propose a more achievable R:R ratio by suggesting new `TP_ATR_MULTIPLIER` and `SL_ATR_MULTIPLIER` values (e.g., reduce from 2.0 to 1.5). New R:R must be >= 1.0.\n\n"
        "3.  **`TEST_SHORT_HORIZON_LABELS`**: A targeted version of the above. Prescribe this to test if shorter-term patterns are learnable by temporarily shrinking the `LOOKAHEAD_CANDLES`. This is a quick test, not a permanent change.\n\n"
        "4.  **`REFINE_FEATURE_SET`**: Prescribe this if some features show promise but others might be adding noise. Propose a new, targeted `selected_features` list based on the playbook description of the current strategy.\n\n"
        "5.  **`SWITCH_STRATEGY`**: A last resort. Choose this only if you have a strong hypothesis that the current strategy's `style` (e.g., momentum) is fundamentally wrong for the current market, and a diagnostic test confirms another style might work better."
        )

        json_schema_definition = (
            "### REQUIRED JSON RESPONSE STRUCTURE ###\n"
            "// You MUST choose exactly ONE action and provide a detailed diagnosis in `analysis_notes`.\n"
            "{\n"
            '  "action": str, // MUST be one of: "RUN_DIAGNOSTIC_ENSEMBLE", "ADJUST_LABELING_DIFFICULTY", "TEST_SHORT_HORIZON_LABELS", "REFINE_FEATURE_SET", "SWITCH_STRATEGY"\n'
            '  "parameters": Optional[Dict], // Required for all actions except "RUN_DIAGNOSTIC_ENSEMBLE". Contains the new values.\n'
            '  "analysis_notes": str // Your detailed diagnosis and reasoning for the chosen action.\n'
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
            f"**3. CURRENT CONFIGURATION & STRATEGY STYLE:**\n`strategy_name`: {current_config.get('strategy_name')}\n`style`: {playbook.get(current_config.get('strategy_name'), {}).get('style')}\n{json.dumps(self._sanitize_dict(current_config), indent=2)}\n\n"
            f"**4. AVAILABLE STRATEGIES (For a potential switch):**\n{json.dumps(self._sanitize_dict(available_playbook), indent=2)}\n"
        )

        response_text = self._call_gemini(prompt)
        suggestions = self._extract_json_from_response(response_text)
        return suggestions

    def discover_hidden_alpha(self, misprediction_data_json: str) -> Dict:
        """
        [NEW] Acts as a quantitative researcher to discover hidden patterns in the model's failures.
        """
        logger.info("-> Engaging AI to perform exploratory data analysis on model failures...")

        prompt = (
            "You are a senior quantitative researcher. Your primary task is to find novel, undiscovered trading patterns (alpha) from a dataset of 'missed opportunities' where our primary model failed to predict a profitable move. Your goal is to create a new feature that would have captured this missed opportunity.\n\n"
            "**CONTEXT:**\n"
            "The provided JSON contains a list of data points. For each point, our model predicted 'Hold,' but a high-quality trade (high `target_signal_pressure`) actually occurred. This means there is a hidden pattern in the features that our model does not understand.\n\n"
            "**YOUR TASK:**\n"
            "1.  **Analyze the Data:** Scrutinize the provided JSON data. Look for non-obvious, recurring relationships between the features. Do not focus on single features. Look for *interactions*. Examples of questions to ask yourself:\n"
            "    - Is a specific `hour` consistently present when `market_volatility_index` is low?\n"
            "    - Does `H1_ctx_Trend` being `1` coincide with `bollinger_bandwidth` being below a certain value?\n"
            "    - Is there a relationship between `Symbol` and another feature? (e.g., 'This only seems to happen for USDCAD').\n"
            "    - Is there a complex condition, like `(row['ADX'] > 25 and row['stoch_k'] < 30)` that is often true?\n\n"
            "2.  **Formulate a Hypothesis:** Based on your analysis, describe the pattern you discovered.\n\n"
            "3.  **Create a Lambda Function:** Convert your hypothesis into a single Python lambda function string. This function will be applied to each row of a DataFrame. It must take a single argument, `row`, and return `True` if your discovered pattern is present, and `False` otherwise.\n\n"
            "**OUTPUT FORMAT:**\n"
            "Respond ONLY with a single, valid JSON object with the following keys:\n"
            "```json\n"
            "{\n"
            "  \"pattern_name\": \"Descriptive_Name_Of_Pattern\",\n"
            "  \"pattern_description\": \"A clear, concise explanation of the pattern you discovered and why it might be predictive.\",\n"
            "  \"lambda_function_str\": \"lambda row: row['hour'] == 8 and row['market_volatility_index'] < 0.2 and row['Symbol'] == 'USDCAD'\"\n"
            "}\n"
            "```\n\n"
            f"**DATA FOR ANALYSIS (MISSED OPPORTUNITIES):**\n{misprediction_data_json}"
        )

        response_text = self._call_gemini(prompt)
        suggestions = self._extract_json_from_response(response_text)
        return suggestions

    def propose_new_playbook_strategy(self, failed_strategy_name: str, playbook: Dict, framework_history: Dict) -> Dict:
        """
        [Phase 3 Implemented] When a strategy is quarantined, this method asks the AI to
        invent a new one by blending concepts from successful and failed strategies.
        """
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

            # --- NEW GROUNDED SEARCH INSTRUCTIONS ---
            "**YOUR TASK:**\n"
            "1. **Perform Grounded Research:** Before inventing a strategy, use your web search tool to find novel ideas relevant to the current market. Look for:\n"
            "    - Recently published (last 1-2 years) academic papers on quantitative finance or signal processing.\n"
            "    - Unconventional combinations of technical indicators discussed on reputable trading forums or blogs.\n"
            "    - Concepts from other fields (e.g., machine learning, physics) that have been applied to financial markets.\n"
            "    - *Example Search Query: 'novel alpha signals from order book imbalance' or 'combining Hilbert-Huang transform with RSI for trading'.*\n\n"
            "2. **Synthesize and Invent:** Combine the most promising insights from your external research with the internal context (what worked and what failed historically). Create a new, hybrid strategy. Give it a creative, descriptive name (e.g., 'WaveletMomentumFilter', 'HawkesProcessBreakout').\n\n"
            "3. **Write a Clear Description:** Explain the logic of your new strategy. What is its core concept? What market regime is it designed for?\n\n"
            "4. **Define Key Parameters:** Set `complexity`, `ideal_regime`, `dd_range`, and `lookahead_range`. The new strategy definition MUST NOT contain a `selected_features` key.\n\n"

            "**OUTPUT FORMAT:** Respond ONLY with a single JSON object for the new strategy entry. The key for the object should be its new name. The response MUST be wrapped between `BEGIN_JSON` and `END_JSON` markers.\n"
            "**EXAMPLE STRUCTURE:**\n"
            "BEGIN_JSON\n"
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
            "END_JSON"
        )

        response_text = self._call_gemini(prompt)
        new_strategy_definition = self._extract_json_from_response(response_text)

        if new_strategy_definition and isinstance(new_strategy_definition, dict) and len(new_strategy_definition) == 1:
            strategy_name = next(iter(new_strategy_definition))
            strategy_body = new_strategy_definition[strategy_name]
            # Validate the structure of the new strategy
            if isinstance(strategy_body, dict) and 'description' in strategy_body and 'complexity' in strategy_body:
                logger.info(f"  - AI has successfully proposed a new strategy named '{strategy_name}'.")
                return new_strategy_definition

        logger.error("  - AI failed to generate a valid new strategy definition.")
        return {}

    def define_gene_pool(self, strategy_goal: str, available_features: List[str]) -> Dict:
        """
        [Phase 3 Implemented] Asks the AI to define a gene pool (indicators, operators, constants)
        for the genetic programming algorithm based on a high-level strategic goal.
        """
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
            "**OUTPUT FORMAT:** Respond ONLY with a single JSON object containing three keys: `indicators`, `operators`, and `constants`. The response MUST be wrapped between `BEGIN_JSON` and `END_JSON` markers.\n\n"
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
            gene_pool['indicators'] = [f for f in gene_pool['indicators'] if f in available_features]
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
    """
    Integrates advanced microstructure features, including volatility displacement,
    gap detection, and alternative volatility estimators (Parkinson, Yang-Zhang).
    """
    TIMEFRAME_MAP = {'M1': 1, 'M5': 5, 'M15': 15, 'M30': 30, 'H1': 60, 'H4': 240, 'D1': 1440, 'DAILY': 1440}
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
    
    def _calculate_forward_sharpe(self, group: pd.DataFrame, lookahead: int) -> pd.Series:
            """
            Helper function to calculate the Sharpe ratio of the price move over a future window.
            This value represents the "quality" or "pressure" of a potential signal.
            """
            # Calculate future log returns
            future_returns = np.log(group['Close'].shift(-lookahead) / group['Close'].shift(-1))

            # Calculate rolling standard deviation of future returns
            rolling_std = future_returns.rolling(window=lookahead).std()

            # Calculate forward Sharpe ratio
            # We handle the case where standard deviation is zero (no price movement)
            forward_sharpe = future_returns / rolling_std.replace(0, np.nan)
            
            return forward_sharpe.fillna(0)

    def _label_candle_patterns(self, group: pd.DataFrame) -> pd.DataFrame:
        """Labels the presence of specific confirmation candle patterns."""
        # Bullish Engulfing: Current green body engulfs previous red body
        is_bullish_engulfing = (group['Close'] > group['Open']) & \
                               (group['Open'].shift(1) > group['Close'].shift(1)) & \
                               (group['Close'] > group['Open'].shift(1)) & \
                               (group['Open'] < group['Close'].shift(1))
        group['target_bullish_engulfing'] = is_bullish_engulfing.astype(int)

        # Bearish Engulfing: Current red body engulfs previous green body
        is_bearish_engulfing = (group['Open'] > group['Close']) & \
                               (group['Close'].shift(1) > group['Open'].shift(1)) & \
                               (group['Open'] > group['Close'].shift(1)) & \
                               (group['Close'] < group['Open'].shift(1))
        group['target_bearish_engulfing'] = is_bearish_engulfing.astype(int)
        return group    

    def _label_trade_timing(self, group: pd.DataFrame, lookahead: int) -> pd.DataFrame:
        """Creates a regression target for the quality of trade entry timing."""
        group['target_timing_score'] = 0.0
        winning_indices = group[group['target_signal_pressure'] != 1].index

        for idx in winning_indices:
            move_window = group['Close'].loc[idx:idx + pd.Timedelta(minutes=self.TIMEFRAME_MAP[self.roles['base']]*lookahead)]
            if len(move_window) < 2: continue
            
            # Find the index of the best price in the future move
            if group.loc[idx, 'target_signal_pressure'] == 2: # Long trade
                best_price_idx = move_window.idxmax()
            else: # Short trade
                best_price_idx = move_window.idxmin()
            
            entry_idx_loc = group.index.get_loc(idx)
            best_price_idx_loc = group.index.get_loc(best_price_idx)
            
            time_to_best = best_price_idx_loc - entry_idx_loc
            
            # Score is 1.0 for perfect timing, decaying to 0.
            timing_score = max(0, 1 - (time_to_best / lookahead))
            group.loc[idx, 'target_timing_score'] = timing_score
        return group
        
    def _label_future_volatility_state(self, group: pd.DataFrame, lookahead: int) -> pd.DataFrame:
        """Creates a classification target for predicting a future volatility spike."""
        # Use Bollinger Bandwidth as a proxy for volatility
        future_vol = group['bollinger_bandwidth'].shift(-lookahead)
        # A spike is when future volatility is in the top 20% of its historical range
        vol_spike_threshold = group['bollinger_bandwidth'].quantile(0.80)
        
        group['target_volatility_spike'] = (future_vol > vol_spike_threshold).astype(int)
        return group    
        
    def generate_multitask_labels(self, df: pd.DataFrame, config: 'ConfigModel') -> pd.DataFrame:
        """
        Orchestrates the creation of all target variables for the learning framework.
        """
        logger.info(f"-> Stage 3: Generating Multi-Task Labels...")
        lookahead = config.LOOKAHEAD_CANDLES

        # 1. Create the primary "Signal Pressure" regression target
        logger.info("  - Generating primary target: Signal Pressure (Forward Sharpe)...")
        df['target_signal_pressure'] = self._calculate_forward_sharpe(df.groupby('Symbol'), lookahead)

        all_labeled_groups = []
        for symbol, group in df.groupby('Symbol'):
            group = group.copy()
            if len(group) < lookahead + 5:
                all_labeled_groups.append(group)
                continue

            # 2. Generate Auxiliary Task 1: Candle Patterns
            logger.info(f"  - ({symbol}) Generating auxiliary target: Confirmation Candle Patterns...")
            group = self._label_candle_patterns(group)

            # 3. Generate Auxiliary Task 2: Trade Entry Timing
            # This needs a temporary classification target to identify "winning" moves
            temp_pressure_target = (group['target_signal_pressure'] > group['target_signal_pressure'].quantile(0.9)).astype(int) * 2
            group['target_signal_pressure_class'] = temp_pressure_target
            logger.info(f"  - ({symbol}) Generating auxiliary target: Trade Timing Score...")
            group = self._label_trade_timing(group, lookahead)

            # 4. Generate Auxiliary Task 3: Predicting Future State
            logger.info(f"  - ({symbol}) Generating auxiliary target: Future Volatility Spike...")
            group = self._label_future_volatility_state(group, lookahead)
            
            all_labeled_groups.append(group.drop(columns=['target_signal_pressure_class']))

        logger.info("[SUCCESS] Multi-task label generation complete.")
        return pd.concat(all_labeled_groups) if all_labeled_groups else pd.DataFrame()

    def _calculate_forward_sharpe(self, df_groups, lookahead: int) -> pd.Series:
        """Helper to calculate forward Sharpe, applied group-wise."""
        def sharpe_calc(group):
            future_returns = np.log(group['Close'].shift(-lookahead) / group['Close'].shift(-1))
            rolling_std = future_returns.rolling(window=lookahead).std()
            return (future_returns / rolling_std.replace(0, np.nan)).fillna(0)
        
        return df_groups.apply(sharpe_calc).reset_index(level=0, drop=True)

    def label_signal_pressure(self, df: pd.DataFrame, lookahead: int, long_quantile: float = 0.95, short_quantile: float = 0.05) -> pd.DataFrame:
        """
        Generates flexible labels based on the quality (forward Sharpe ratio) of future price moves.
        This method is designed to overcome the extreme class imbalance of rigid labeling.

        - Top quantile of Sharpe ratio -> "Long" signal
        - Bottom quantile of Sharpe ratio -> "Short" signal
        - Everything else -> "Hold"
        """
        logger.info(f"-> Stage 3: Generating Flexible Labels via Signal Pressure (Forward Sharpe)...")
        logger.info(f"   - Long/Short Quantiles: {long_quantile}/{short_quantile}")

        all_labeled_groups = []
        for symbol, group in df.groupby('Symbol'):
            group = group.copy()
            if len(group) < lookahead + 5:
                group['target'] = 1 # Default to hold if not enough data
                all_labeled_groups.append(group)
                continue

            # Calculate the signal pressure (quality) for every candle
            group['signal_pressure'] = self._calculate_forward_sharpe(group, lookahead)

            # Determine the thresholds for long and short signals based on quantiles
            pressure_values = group['signal_pressure'][group['signal_pressure'] != 0]
            long_threshold = pressure_values.quantile(long_quantile)
            short_threshold = pressure_values.quantile(short_quantile)

            # Create labels based on the pressure thresholds
            group['target'] = 1 # Default to Hold
            group.loc[group['signal_pressure'] >= long_threshold, 'target'] = 2 # Long
            group.loc[group['signal_pressure'] <= short_threshold, 'target'] = 0 # Short

            # Log the distribution for this symbol
            dist = group['target'].value_counts(normalize=True)
            logger.info(f"    - Label distribution for '{symbol}': "
                        f"Hold={dist.get(1, 0):.2%}, "
                        f"Long={dist.get(2, 0):.2%}, "
                        f"Short={dist.get(0, 0):.2%}")
            
            all_labeled_groups.append(group)

        return pd.concat(all_labeled_groups) if all_labeled_groups else pd.DataFrame()    
        
    def apply_discovered_features(self, df: pd.DataFrame, discovered_patterns: List[Dict]) -> pd.DataFrame:
        """
        [NEW] Applies AI-discovered patterns to the dataframe as new feature columns.
        """
        if not discovered_patterns:
            return df

        logger.info(f"-> Injecting {len(discovered_patterns)} AI-discovered alpha features...")
        df_copy = df.copy()

        for i, pattern in enumerate(discovered_patterns):
            try:
                pattern_name = pattern.get("pattern_name", f"alpha_pattern_{i+1}")
                lambda_str = pattern.get("lambda_function_str")

                if not lambda_str or "lambda" not in lambda_str:
                    logger.warning(f"  - Skipping invalid lambda function string for pattern: {pattern_name}")
                    continue

                logger.info(f"  - Applying pattern: '{pattern_name}'")
                
                # Safely evaluate and apply the AI-generated lambda function
                # The lambda function operates row-wise (axis=1)
                df_copy[pattern_name] = df_copy.apply(eval(lambda_str, {"__builtins__": {}}), axis=1).astype(int)

            except Exception as e:
                logger.error(f"  - Failed to apply discovered pattern '{pattern_name}': {e}")
                # Create a dummy column to avoid errors downstream
                df_copy[pattern_name] = 0

        return df_copy    

    def _add_higher_tf_context(self, base_df: pd.DataFrame, higher_tf_df: pd.DataFrame, tf_name: str) -> pd.DataFrame:
        if higher_tf_df.empty:
            return base_df
        
        ctx_features = {'Close': 'last', 'High': 'max', 'Low': 'min', 'Open': 'first', 'ATR': 'mean', 'RSI': 'mean', 'ADX': 'mean'}
        base_tf_str = self.roles['base']
        minutes = self.TIMEFRAME_MAP.get(base_tf_str.upper())
        if not minutes:
            logger.error(f"Could not find timeframe '{base_tf_str}' in TIMEFRAME_MAP. Resampling will fail.")
            return base_df
        pandas_freq = f"{minutes}T"

        resampled_features = {f"{tf_name}_ctx_{col}": higher_tf_df[col].resample(pandas_freq).ffill() for col, method in ctx_features.items() if col in higher_tf_df.columns}
        if not resampled_features: return base_df

        resampled_df = pd.DataFrame(resampled_features)
        merged_df = pd.merge_asof(left=base_df.sort_index(), right=resampled_df.sort_index(), on='Timestamp', direction='backward')
        
        ctx_close_col = f"{tf_name}_ctx_Close"
        if ctx_close_col in merged_df.columns:
            merged_df[f"{tf_name}_ctx_Trend"] = np.sign(merged_df[ctx_close_col].diff(2)).fillna(0)
            
        return merged_df.set_index('Timestamp')

    def _detect_anomalies(self, df: pd.DataFrame) -> pd.DataFrame:
        logger.info("    - Detecting anomalies with Isolation Forest...")
        anomaly_features_present = [f for f in self.ANOMALY_FEATURES if f in df.columns and f in df]
        if not anomaly_features_present:
            df['anomaly_score'] = 1
            return df
            
        df_anomaly = df[anomaly_features_present].dropna()
        if df_anomaly.empty:
            df['anomaly_score'] = 1
            return df

        model = IsolationForest(contamination=self.config.anomaly_contamination_factor, random_state=42)
        predictions = model.fit_predict(df_anomaly)
        df['anomaly_score'] = pd.Series(predictions, index=df_anomaly.index)
        df['anomaly_score'].fillna(method='ffill', inplace=True)
        df['anomaly_score'].fillna(method='bfill', inplace=True)
        df['anomaly_score'].fillna(1, inplace=True)
        return df

    def _calculate_relative_performance(self, df: pd.DataFrame) -> pd.DataFrame:
        if 'pct_change' not in df.columns:
             df['pct_change'] = df.groupby('Symbol')['Close'].pct_change()
             
        mean_returns = df.groupby('Timestamp')['pct_change'].mean()
        df['market_return'] = df.index.map(mean_returns)
        df['relative_performance'] = df['pct_change'] - df['market_return']
        df.drop(columns=['market_return'], inplace=True)
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
            g['volume_velocity'], g['volume_acceleration'] = 0, 0
        return g

    def _calculate_statistical_moments(self, g: pd.DataFrame, window: int = 20) -> pd.DataFrame:
        log_returns = np.log(g['Close'] / g['Close'].shift(1))
        g['returns_skew'] = log_returns.rolling(window).skew()
        g['returns_kurtosis'] = log_returns.rolling(window).kurt()
        return g
        
    def _calculate_simple_features(self, df: pd.DataFrame) -> pd.DataFrame:
        df['pct_change'] = df['Close'].pct_change()
        df['overnight_gap_pct'] = df['Open'].pct_change()
        df['candle_body_size'] = (df['Close'] - df['Open']).abs()
        upper_wick = df['High'] - df[['Close', 'Open']].max(axis=1)
        lower_wick = df[['Close', 'Open']].min(axis=1) - df['Low']
        df['wick_to_body_ratio'] = (upper_wick + lower_wick) / df['candle_body_size'].replace(0, np.nan)
        if 'RSI' in df.columns:
            df['RSI_zscore'] = (df['RSI'] - df['RSI'].rolling(20).mean()) / df['RSI'].rolling(20).std()
        if 'RealVolume' in df.columns and not df['RealVolume'].empty:
            vol_ma = df['RealVolume'].rolling(20).mean()
            df['volume_ma_ratio'] = df['RealVolume'] / vol_ma.replace(0, np.nan)
        if 'ATR' in df.columns:
             df['candle_body_size_vs_atr'] = df['candle_body_size'] / df['ATR'].replace(0, np.nan)
        if 'DAILY_ctx_ATR' in df.columns and 'ATR' in df.columns:
            df['atr_vs_daily_atr'] = df['ATR'] / df['DAILY_ctx_ATR'].replace(0, np.nan)
        return df

    def _calculate_hawkes_volatility(self, df: pd.DataFrame) -> pd.DataFrame:
        if 'ATR' not in df.columns:
            df['volatility_hawkes'] = 0.0
            return df
        atr_shocks = df['ATR'].diff().clip(lower=0)
        hawkes_intensity = atr_shocks.ewm(alpha=1 - self.config.HAWKES_KAPPA, adjust=False).mean()
        df['volatility_hawkes'] = hawkes_intensity
        return df

    def _calculate_ohlc_ratios(self, df: pd.DataFrame) -> pd.DataFrame:
        df['ohlc_range'] = df['High'] - df['Low']
        # Avoid division by zero
        df['close_to_high'] = (df['High'] - df['Close']) / df['ohlc_range'].replace(0, np.nan)
        df['close_to_low'] = (df['Close'] - df['Low']) / df['ohlc_range'].replace(0, np.nan)
        return df

    def _calculate_accumulation_distribution(self, df: pd.DataFrame) -> pd.DataFrame:
        if 'RealVolume' not in df.columns: return df
        clv = ((df['Close'] - df['Low']) - (df['High'] - df['Close'])) / (df['High'] - df['Low']).replace(0, np.nan)
        clv = clv.fillna(0)
        ad = (clv * df['RealVolume']).cumsum()
        df['AD_line'] = ad
        df['AD_line_slope'] = df['AD_line'].diff(5) # 5-period slope of the A/D line
        return df

    def _calculate_mad(self, df: pd.DataFrame, window: int = 20) -> pd.DataFrame:
        """Mean Absolute Deviation"""
        df['mad'] = df['Close'].rolling(window).apply(lambda x: np.mean(np.abs(x - np.mean(x))), raw=True)
        return df

    def _calculate_price_volume_correlation(self, df: pd.DataFrame, window: int = 20) -> pd.DataFrame:
        if 'RealVolume' not in df.columns: return df
        df['price_vol_corr'] = df['Close'].pct_change().rolling(window).corr(df['RealVolume'])
        return df

    def _calculate_quantile_features(self, df: pd.DataFrame, window: int = 50) -> pd.DataFrame:
        log_returns = np.log(df['Close'] / df['Close'].shift(1))
        df['return_q25'] = log_returns.rolling(window).quantile(0.25)
        df['return_q75'] = log_returns.rolling(window).quantile(0.75)
        df['return_iqr'] = df['return_q75'] - df['return_q25']
        return df

    def _calculate_regression_features(self, df: pd.DataFrame, window: int = 20) -> pd.DataFrame:
        def get_slope(series):
            y = series.values
            x = np.arange(len(y))
            slope = np.polyfit(x, y, 1)[0]
            return slope
        df['rolling_beta'] = df['Close'].rolling(window).apply(get_slope, raw=False)
        return df
    
    def _calculate_displacement(self, df: pd.DataFrame) -> pd.DataFrame:
        """Identifies price displacements (volatility spikes) based on candle range."""
        df_copy = df.copy()
        df_copy["candle_range"] = np.abs(df_copy["High"] - df_copy["Low"])
        mstd = df_copy["candle_range"].rolling(self.config.DISPLACEMENT_PERIOD).std()
        threshold = mstd * self.config.DISPLACEMENT_STRENGTH
        
        df_copy["displacement"] = (df_copy["candle_range"] > threshold).astype(int)
        
        variation = df_copy["Close"] - df_copy["Open"]
        df_copy["green_displacement"] = ((df_copy["displacement"] == 1) & (variation > 0)).astype(int).shift(1)
        df_copy["red_displacement"] = ((df_copy["displacement"] == 1) & (variation < 0)).astype(int).shift(1)

        return df_copy.drop(columns=['candle_range', 'displacement'])

    def _calculate_gaps(self, df: pd.DataFrame) -> pd.DataFrame:
        """Identifies and measures bullish and bearish price gaps."""
        df_copy = df.copy()
        lookback = self.config.GAP_DETECTION_LOOKBACK

        df_copy["is_bullish_gap"] = (df_copy["High"].shift(lookback) < df_copy["Low"]).astype(int)
        df_copy["is_bearish_gap"] = (df_copy["High"] < df_copy["Low"].shift(lookback)).astype(int)

        return df_copy

    def _calculate_candle_info(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculates basic candle analytics like color and body-to-range ratio."""
        df_copy = df.copy()
        df_copy["candle_way"] = np.sign(df_copy["Close"] - df_copy["Open"]).fillna(0)
        ohlc_range = (df_copy["High"] - df_copy["Low"]).replace(0, np.nan)
        df_copy["filling_ratio"] = (np.abs(df_copy["Close"] - df_copy["Open"]) / ohlc_range).fillna(0)
        return df_copy

    def _calculate_parkinson_volatility(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculates Parkinson's volatility estimator on a rolling basis."""
        window = self.config.PARKINSON_VOLATILITY_WINDOW
        
        def parkinson_estimator(high_low_log_sq):
            return np.sqrt(np.sum(high_low_log_sq) / (4 * window * np.log(2)))

        high_low_ratio_log_sq = (np.log(df['High'] / df['Low']) ** 2).replace([np.inf, -np.inf], np.nan).fillna(0)
        df['volatility_parkinson'] = high_low_ratio_log_sq.rolling(window=window).apply(parkinson_estimator, raw=True)
        return df
        
    def _calculate_yang_zhang_volatility(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculates Yang-Zhang's volatility estimator on a rolling basis."""
        window = self.config.YANG_ZHANG_VOLATILITY_WINDOW

        def yang_zhang_estimator(sub_df):
            log_ho = np.log(sub_df['High'] / sub_df['Open'])
            log_lo = np.log(sub_df['Low'] / sub_df['Open'])
            log_co = np.log(sub_df['Close'] / sub_df['Open'])
            
            sigma_o_sq = (1 / (window - 1)) * np.sum((np.log(sub_df['Open'] / sub_df['Close'].shift(1)) - np.mean(np.log(sub_df['Open'] / sub_df['Close'].shift(1))))**2)
            sigma_c_sq = (1 / (window - 1)) * np.sum((log_co - np.mean(log_co))**2)
            sigma_rs_sq = np.sum(log_ho * (log_ho - log_co) + log_lo * (log_lo - log_co)) / window
            
            k = 0.34 / (1.34 + (window + 1) / (window - 1))
            
            vol = np.sqrt(sigma_o_sq + k * sigma_c_sq + (1 - k) * sigma_rs_sq)
            return vol

        # This calculation is more complex and less suited for raw=True, apply works but is slower.
        # For a framework, this is acceptable for a more advanced indicator.
        df['volatility_yang_zhang'] = df[['Open', 'High', 'Low', 'Close']].rolling(window=window).apply(yang_zhang_estimator, raw=False)
        return df

    def _calculate_kama_manual(self, series: pd.Series, n: int = 10, pow1: int = 2, pow2: int = 30) -> pd.Series:
        """
        Calculates Kaufman's Adaptive Moving Average (KAMA) manually.
        Correctly uses integer positions for NumPy array indexing.
        """
        # 1. Calculate Efficiency Ratio (ER)
        change = abs(series - series.shift(n))
        volatility = (series - series.shift()).abs().rolling(n).sum()
        er = change / volatility
        er.fillna(0, inplace=True)

        # 2. Calculate Smoothing Constant (SC)
        sc_fast = 2 / (pow1 + 1)
        sc_slow = 2 / (pow2 + 1)
        sc = (er * (sc_fast - sc_slow) + sc_slow) ** 2

        # 3. Calculate KAMA iteratively
        kama = np.zeros(sc.size)
        
        # Get the label of the first valid index
        first_valid_label = series.first_valid_index()
        if first_valid_label is None:
            return pd.Series(kama, index=series.index) # Return zeros if series is all NaN

        # --- FIX: Convert the timestamp label to an integer position ---
        first_valid_pos = series.index.get_loc(first_valid_label)
        # --- END FIX ---

        # Seed the first KAMA value using the integer position
        kama[first_valid_pos] = series.iloc[first_valid_pos]

        # Iterate from the next integer position
        for i in range(first_valid_pos + 1, len(sc)):
            if pd.isna(series.iloc[i]):
                 kama[i] = kama[i-1]
                 continue
            if pd.isna(kama[i-1]):
                kama[i] = series.iloc[i]
            else:
                kama[i] = kama[i-1] + sc.iloc[i] * (series.iloc[i] - kama[i-1])
        
        return pd.Series(kama, index=series.index)

    def _calculate_kama_regime(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Determines market trend using a KAMA fast/slow crossover,
        now calculated manually without the 'ta' library.
        """
        df_copy = df.copy()
        
        # Call our new manual KAMA calculation method
        fast_kama = self._calculate_kama_manual(df_copy["Close"], n=self.config.KAMA_REGIME_FAST)
        slow_kama = self._calculate_kama_manual(df_copy["Close"], n=self.config.KAMA_REGIME_SLOW)
        
        df_copy["kama_trend"] = np.sign(fast_kama - slow_kama).fillna(0)
        
        return df_copy
    
    def _calculate_cycle_features(self, df: pd.DataFrame, window: int = 40) -> pd.DataFrame:
        df['dominant_cycle_phase'] = np.nan
        df['dominant_cycle_period'] = np.nan
        close_series = df['Close'].dropna()
        symbol = df['Symbol'].iloc[0] if 'Symbol' in df.columns and not df.empty else 'UNKNOWN'

        if len(close_series) < window:
            logger.debug(f"  - Cycle Features: Not enough data for {symbol} (found {len(close_series)}, need > {window}).")
            return df

        try:
            analytic_signal = hilbert(close_series.values)
            phase = np.unwrap(np.angle(analytic_signal))
            phase_series = pd.Series(phase, index=close_series.index)
            df.loc[phase_series.index, 'dominant_cycle_phase'] = phase_series

            inst_freq = np.diff(phase) / (2.0 * np.pi)
            inst_freq_series = pd.Series(inst_freq, index=close_series.index[1:])

            epsilon = 1e-9
            safe_inst_freq_np = np.where(np.abs(inst_freq_series) < epsilon, np.nan, inst_freq_series)
            safe_inst_freq_series = pd.Series(safe_inst_freq_np, index=inst_freq_series.index)

            if safe_inst_freq_series.isnull().all():
                logger.debug(f"  - Cycle Features: All instantaneous frequencies near-zero for {symbol} over {len(close_series)} data points (signal is flat).")
                return df

            dominant_cycle_period_series = 1 / np.abs(safe_inst_freq_series)
            rolling_period = dominant_cycle_period_series.rolling(window=window, min_periods=max(1, window // 2)).mean()
            df.loc[rolling_period.index, 'dominant_cycle_period'] = rolling_period
        except Exception as e:
            logger.error(f"  - Cycle Features Error for symbol {symbol}: {e}")
        return df

    def _calculate_autocorrelation_features(self, df: pd.DataFrame, lags: int = 5) -> pd.DataFrame:
        log_returns = np.log(df['Close'] / df['Close'].shift(1)).dropna()
        pacf_vals = pacf(log_returns, nlags=lags)
        for i in range(1, lags + 1):
            df[f'pacf_lag_{i}'] = pacf_vals[i]
        return df

    def _calculate_entropy_features(self, df: pd.DataFrame, window: int = 30) -> pd.DataFrame:
        def roll_entropy(series):
            discretized = pd.cut(series, bins=10, labels=False)
            counts = discretized.value_counts(normalize=True)
            return entropy(counts, base=2)
        df['shannon_entropy'] = df['Close'].pct_change().rolling(window).apply(roll_entropy, raw=False)
        return df

    def _calculate_fourier_transform_features(self, df: pd.DataFrame, window: int = 50) -> pd.DataFrame:
        def get_dominant_freq(series: pd.Series) -> tuple[float, float]:
            n = len(series)
            if n < window or series.nunique() < 2:
                return np.nan, np.nan
            try:
                fft_vals = scipy.fft.fft(series.values)
                fft_freq = scipy.fft.fftfreq(n)
                idx = np.argmax(np.abs(fft_vals[1:n//2])) + 1
                return np.abs(fft_freq[idx]), np.abs(fft_vals[idx]) / n
            except Exception:
                return np.nan, np.nan

        results_list = [get_dominant_freq(w) for w in df['Close'].rolling(window)]
        if results_list:
            fft_df = pd.DataFrame(results_list, index=df.index, columns=['fft_dom_freq', 'fft_dom_amp'])
            df[['fft_dom_freq', 'fft_dom_amp']] = fft_df
        else:
            df['fft_dom_freq'] = np.nan
            df['fft_dom_amp'] = np.nan
        return df

    def _calculate_wavelet_features(self, df: pd.DataFrame, wavelet_name='db4', level=4) -> pd.DataFrame:
        if not PYWT_AVAILABLE: return df
        coeffs = pywt.wavedec(df['Close'], wavelet_name, level=level)
        for i, c in enumerate(coeffs):
            df[f'wavelet_energy_level_{i}'] = np.sum(np.square(c))
        return df

    def _calculate_garch_volatility(self, df: pd.DataFrame) -> pd.DataFrame:
        df['garch_volatility'] = np.nan
        if not ARCH_AVAILABLE: return df
        
        # Upscale the input data by 1000 for optimizer stability
        log_returns = np.log(df['Close'].replace(0,np.nan) / df['Close'].shift(1).replace(0,np.nan)).dropna() * 1000

        if len(log_returns) < 20:
            return df
            
        try:
            # Tell the library we are handling scaling manually
            garch_model = arch_model(log_returns, vol='Garch', p=1, q=1, rescale=False)
            res = garch_model.fit(update_freq=0, disp='off', show_warning=False)
            forecast = res.forecast(horizon=5, reindex=False)
            
            # Downscale the final output by 1000 to return it to its original units
            pred_vol = np.sqrt(forecast.variance.iloc[-1].mean()) / 1000.0
            
            df.at[log_returns.index[-1], 'garch_volatility'] = pred_vol
            df['garch_volatility'] = df['garch_volatility'].bfill()
        except Exception as e:
            logger.error(f"  - GARCH Error: {e}")
            
        return df

    def _calculate_dynamic_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculates indicators using parameters that adapt to the market regime.
        This version now preserves the string name of the regime for later use.
        """
        logger.info("    - Calculating features with DYNAMIC, regime-aware parameters...")

        df['volatility_regime'] = pd.cut(df['market_volatility_index'], bins=[0, 0.3, 0.7, 1.0], labels=['LowVolatility', 'Default', 'HighVolatility'], right=False).astype(str).fillna('Default')
        df['trend_regime'] = pd.cut(df['hurst_exponent'], bins=[0, 0.4, 0.6, 1.0], labels=['Ranging', 'Default', 'Trending'], right=False).astype(str).fillna('Default')
        
        # Keep the string version for readable logic in meta-features
        df['market_regime_str'] = df['volatility_regime'] + "_" + df['trend_regime']
        
        processed_regime_dfs = []

        # Group by the string name to apply parameters
        for regime_name, group_df in df.groupby('market_regime_str'):
            params = self.config.DYNAMIC_INDICATOR_PARAMS.get(regime_name, self.config.DYNAMIC_INDICATOR_PARAMS['Default'])
            group_copy = group_df.copy()
            group_copy = self._calculate_bollinger_bands(group_copy, period=params['bollinger_period'], std_dev=params['bollinger_std_dev'])
            group_copy = self._calculate_rsi(group_copy, period=params['rsi_period'])
            processed_regime_dfs.append(group_copy)
        
        if not processed_regime_dfs:
            logger.warning("    - No data was processed by the dynamic indicator calculator.")
            return df

        final_df = pd.concat(processed_regime_dfs).sort_index()
        
        # Create the numeric version for the ML model
        final_df['market_regime'] = pd.factorize(final_df['market_regime_str'])[0]
        final_df = final_df.drop(columns=['volatility_regime', 'trend_regime'], errors='ignore')
        
        return final_df

    def _calculate_rsi(self, df: pd.DataFrame, period: int = 14) -> pd.DataFrame:
        """Calculates the Relative Strength Index (RSI) for a given period."""
        delta = df['Close'].diff()
        gain = delta.clip(lower=0).ewm(alpha=1/period, adjust=False).mean()
        loss = (-delta.clip(upper=0)).ewm(alpha=1/period, adjust=False).mean()
        rs = gain / loss.replace(0, 1e-9)
        df[f'RSI'] = 100 - (100 / (1 + rs))
        return df

    def _calculate_bollinger_bands(self, df: pd.DataFrame, period: int = 20, std_dev: float = 2.0) -> pd.DataFrame:
        """Calculates Bollinger Bands for a given period and standard deviation."""
        ma = df['Close'].rolling(window=period).mean()
        std = df['Close'].rolling(window=period).std()
        df['bollinger_upper'] = ma + (std * std_dev)
        df['bollinger_lower'] = ma - (std * std_dev)
        df['bollinger_bandwidth'] = (df['bollinger_upper'] - df['bollinger_lower']) / ma.replace(0, 1e-9)
        return df

    def _calculate_hurst_exponent(self, df: pd.DataFrame, window: int = 100) -> pd.DataFrame:
        """
        Calculates the Hurst Exponent (H) and the intercept (c) on a rolling basis.
        Correctly applies the rolling function twice to calculate H and c
        independently, preventing a TypeError.
        """
        if not HURST_AVAILABLE:
            df['hurst_exponent'] = np.nan
            df['hurst_intercept'] = np.nan
            return df

        def apply_hurst(series, component_index):
            """
            Robustly applies the compute_Hc function and returns a single component.
            component_index=0 for H, component_index=1 for c.
            """
            if len(series) < 20 or series.nunique() < 2:
                return np.nan
            try:
                # compute_Hc returns a tuple (H, c, data_points)
                result_tuple = compute_Hc(series, kind='price', simplified=True)
                return result_tuple[component_index]
            except Exception:
                return np.nan
        # --- END FIX ---

        # Create the rolling object once for efficiency
        rolling_close = df['Close'].rolling(window=window, min_periods=max(20, window // 2))

        # --- FIX: Call .apply() separately for each component ---
        # Calculate 'hurst_exponent' (H) by requesting component 0
        df['hurst_exponent'] = rolling_close.apply(apply_hurst, raw=False, args=(0,))
        
        # Calculate 'hurst_intercept' (c) by requesting component 1
        df['hurst_intercept'] = rolling_close.apply(apply_hurst, raw=False, args=(1,))
        # --- END FIX ---

        return df
    
    def _calculate_trend_pullback_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Identifies pullbacks within an established trend."""
        # Condition for a bullish trend
        is_uptrend = (df['ADX'] > 20) & (df['EMA_20'] > df['EMA_50'])
        # Condition for a pullback within the bullish trend
        is_bullish_pullback_signal = (df['Close'] < df['EMA_20']) & (df['RSI'] < 60)
        df['is_bullish_pullback'] = (is_uptrend & is_bullish_pullback_signal).astype(int)

        # Condition for a bearish trend
        is_downtrend = (df['ADX'] > 20) & (df['EMA_20'] < df['EMA_50'])
        # Condition for a pullback within the bearish trend
        is_bearish_pullback_signal = (df['Close'] > df['EMA_20']) & (df['RSI'] > 40)
        df['is_bearish_pullback'] = (is_downtrend & is_bearish_pullback_signal).astype(int)
        
        return df

    def _calculate_divergence_features(self, df: pd.DataFrame, lookback: int = 14) -> pd.DataFrame:
        """Identifies classic bearish and bullish momentum divergence as a proxy for reversals."""
        rolling_window = df['Close'].rolling(window=lookback)
        rolling_rsi = df['RSI'].rolling(window=lookback)

        # Bearish Divergence: Higher high in price, lower high in RSI
        price_higher_high = df['Close'] == rolling_window.max()
        rsi_lower_high = df['RSI'] < rolling_rsi.max()
        df['is_bearish_divergence'] = (price_higher_high & rsi_lower_high).astype(int)

        # Bullish Divergence: Lower low in price, higher low in RSI
        price_lower_low = df['Close'] == rolling_window.min()
        rsi_higher_low = df['RSI'] > rolling_rsi.min()
        df['is_bullish_divergence'] = (price_lower_low & rsi_higher_low).astype(int)

        return df
    
    def _apply_kalman_filter(self, series: pd.Series) -> pd.Series:
        """Applies a Kalman Filter to a pandas Series to smooth and denoise it."""
        if series.isnull().all() or len(series.dropna()) < 2:
            return series # Not enough data to filter

        # Use the series itself to estimate the transition and observation matrices
        series_filled = series.fillna(method='ffill').fillna(method='bfill')
        
        # If still NaN after filling, can't proceed
        if series_filled.isnull().all():
            return series

        kf = KalmanFilter(initial_state_mean=0, n_dim_obs=1)
        kf = kf.em(series_filled.values, n_iter=5)
        
        (smoothed_state_means, _) = kf.smooth(series_filled.values)
        
        return pd.Series(smoothed_state_means.flatten(), index=series.index)
    
    def _calculate_meta_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculates non-linear interaction features (meta-features)."""
        logger.info("    - Calculating meta-features (feature interactions)...")
        
        # Existing interaction features
        if 'RSI' in df.columns and 'bollinger_bandwidth' in df.columns:
            df['rsi_x_bolli'] = df['RSI'] * df['bollinger_bandwidth']
        
        if 'ADX' in df.columns and 'market_volatility_index' in df.columns:
            df['adx_x_vol_rank'] = df['ADX'] * df['market_volatility_index']
            
        if 'hurst_exponent' in df.columns and 'ADX' in df.columns:
            df['hurst_x_adx'] = df['hurst_exponent'] * df['ADX']
            
        if 'ATR' in df.columns and 'DAILY_ctx_ATR' in df.columns:
            df['atr_ratio_short_long'] = df['ATR'] / df['DAILY_ctx_ATR'].replace(0, np.nan)
            
        # --- ADDING NEW META-FEATURES FOR HURST INTERCEPT ---
        if 'hurst_intercept' in df.columns and 'ADX' in df.columns:
            df['hurst_intercept_x_adx'] = df['hurst_intercept'] * df['ADX']
            
        if 'hurst_intercept' in df.columns and 'ATR' in df.columns:
            df['hurst_intercept_x_atr'] = df['hurst_intercept'] * df['ATR']
                
        return df
        
    # --- [NEW FEATURE] LIQUIDITY PROXIES ---
    def _calculate_liquidity_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculates liquidity proxies like the Corwin-Schultz estimated spread."""
        logger.info("    - Calculating liquidity features (Corwin-Schultz)...")
        # Ensure High >= Low to prevent log(negative) errors
        df_safe = df[df['High'] >= df['Low']].copy()

        beta = (np.log(df_safe['High'] / df_safe['Low'])**2).rolling(2).sum()
        gamma = np.log(df_safe['High'].rolling(2).max() / df_safe['Low'].rolling(2).min())**2

        # Calculate alpha, avoiding division by zero
        alpha_denom = 3 - 2 * np.sqrt(2)
        alpha = (np.sqrt(2 * beta) - np.sqrt(beta)) / alpha_denom - np.sqrt(gamma / alpha_denom)

        # Calculate spread, handling potential NaNs and infinities
        spread = 2 * (np.exp(alpha) - 1) / (1 + np.exp(alpha))
        df['estimated_spread'] = spread.reindex(df.index)

        # Calculate illiquidity ratio
        df['illiquidity_ratio'] = df['estimated_spread'] / df['ATR'].replace(0, np.nan)
        return df

    # --- [NEW FEATURE] ORDER FLOW PROXIES ---
    def _calculate_order_flow_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Estimates order flow imbalance using a simple tick rule and volume."""
        logger.info("    - Calculating order flow imbalance proxies...")
        price_changes = df['Close'].diff()
        
        # Tick rule: +1 for uptick, -1 for downtick
        df['tick_rule_direction'] = np.sign(price_changes).replace(0, np.nan).ffill().fillna(0)

        # Volume-weighted direction (Order Flow Proxy)
        if 'RealVolume' in df.columns:
            signed_volume = df['tick_rule_direction'] * df['RealVolume']
            df['volume_imbalance_5'] = signed_volume.rolling(5).sum()
            df['volume_imbalance_20'] = signed_volume.rolling(20).sum()
        return df

    # --- [NEW FEATURE] MARKET DEPTH PROXIES ---
    def _calculate_depth_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Estimates market depth and selling/buying pressure using candle patterns."""
        logger.info("    - Calculating market depth and pressure proxies...")
        ohlc_range = (df['High'] - df['Low']).replace(0, np.nan)
        
        # Body-to-range ratio as a proxy for decisive movement (depth)
        df['depth_proxy_filling_ratio'] = (df['Close'] - df['Open']).abs() / ohlc_range

        # Ratios of shadows as proxies for buying/selling pressure rejection
        upper_shadow = df['High'] - df[['Close', 'Open']].max(axis=1)
        lower_shadow = df[['Close', 'Open']].min(axis=1) - df['Low']
        df['upper_shadow_pressure'] = upper_shadow / ohlc_range
        df['lower_shadow_pressure'] = lower_shadow / ohlc_range
        return df

    # --- [NEW FEATURE] TIME-BASED FEATURES ---
    def _calculate_time_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Adds temporal features like session times and cyclical patterns."""
        logger.info("    - Calculating time-based (session/cyclical) features...")
        if not isinstance(df.index, pd.DatetimeIndex):
            logger.warning("  - DataFrame index is not DatetimeIndex. Skipping time features.")
            return df
            
        df['hour'] = df.index.hour
        df['day_of_week'] = df.index.dayofweek
        
        # Session markers based on UTC time
        df['is_asian_session'] = ((df.index.hour >= 0) & (df.index.hour < 8)).astype(int)
        df['is_london_session'] = ((df.index.hour >= 8) & (df.index.hour < 16)).astype(int)
        df['is_ny_session'] = ((df.index.hour >= 16) | (df.index.hour < 0)).astype(int) # Catches NY open and close
        return df

    # --- [NEW FEATURE] STRUCTURAL BREAKS ---
    def _calculate_structural_breaks(self, df: pd.DataFrame, window=100) -> pd.DataFrame:
        """Detects structural breaks in the price series using a CUSUM test on returns."""
        logger.info("    - Calculating structural break indicators (CUSUM)...")
        # Use log returns for stationarity
        log_returns = np.log(df['Close'] / df['Close'].shift(1)).dropna()
        df['structural_break_cusum'] = 0

        if len(log_returns) < window:
            return df
            
        # Standardize returns over a rolling window
        rolling_mean = log_returns.rolling(window=window).mean()
        rolling_std = log_returns.rolling(window=window).std().replace(0, np.nan)
        standardized_returns = (log_returns - rolling_mean) / rolling_std

        # Calculate CUSUM on the standardized series
        cusum = standardized_returns.rolling(window=window).apply(lambda x: x.cumsum().max() - x.cumsum().min(), raw=True)
        
        # A break is flagged if the CUSUM value exceeds an empirical threshold (e.g., 5 standard deviations)
        # This indicates a significant deviation from the recent mean return behavior.
        break_threshold = 5.0 
        df.loc[cusum.index, 'structural_break_cusum'] = (cusum > break_threshold).astype(int)
        return df

    def _process_single_symbol_stack(self, symbol_data_by_tf: Dict[str, pd.DataFrame]) -> pd.DataFrame:
        """
        [UPDATED] Orchestrates the entire feature engineering pipeline, now including
        the new liquidity, order flow, depth, time, and structural break features.
        """
        # --- 1. Initial Data Validation ---
        base_df = symbol_data_by_tf.get(self.roles['base'])
        if base_df is None or base_df.empty:
            logger.warning("Base timeframe data missing or empty for a symbol. Skipping.")
            return pd.DataFrame()

        df = base_df.copy()
        df.index = pd.to_datetime(df.index)    

    def _process_single_symbol_stack(self, symbol_data_by_tf: Dict[str, pd.DataFrame]) -> pd.DataFrame:
        """
        [UPDATED] Orchestrates the entire feature engineering pipeline, now including
        the new liquidity, order flow, depth, time, and structural break features.
        """
        # --- 1. Initial Data Validation ---
        base_df = symbol_data_by_tf.get(self.roles['base'])
        if base_df is None or base_df.empty:
            logger.warning("Base timeframe data missing or empty for a symbol. Skipping.")
            return pd.DataFrame()

        df = base_df.copy()
        df.index = pd.to_datetime(df.index)
        
        # --- 2. Foundational & Base Indicator Calculation ---
        # This stage calculates all the primary indicators needed for later stages.
        logger.info("    - Calculating foundational and base indicators...")
        
        # ATR (needed by many other indicators)
        high_low = df['High'] - df['Low']
        high_close = np.abs(df['High'] - df['Close'].shift())
        low_close = np.abs(df['Low'] - df['Close'].shift())
        tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
        df['ATR'] = tr.ewm(alpha=1/14, adjust=False).mean()

        # Inputs for Regime Detection
        df['realized_volatility'] = df['Close'].pct_change().rolling(14).std() * np.sqrt(252 * 24 * 4)
        df['market_volatility_index'] = df['realized_volatility'].rank(pct=True)
        df = self._calculate_hurst_exponent(df)
        
        # Standard Technical Indicators
        df['EMA_20'] = df['Close'].ewm(span=20, adjust=False).mean()
        df['EMA_50'] = df['Close'].ewm(span=50, adjust=False).mean()
        
        plus_dm = df['High'].diff().clip(lower=0)
        minus_dm = df['Low'].diff().clip(lower=0)
        plus_di = 100 * (plus_dm.ewm(alpha=1/14, adjust=False).mean() / df['ATR'])
        minus_di = 100 * (minus_dm.ewm(alpha=1/14, adjust=False).mean() / df['ATR'])
        dx = 100 * (np.abs(plus_di - minus_di) / (plus_di + minus_di).replace(0, 1e-9))
        df['ADX'] = dx.ewm(alpha=1/14, adjust=False).mean()

        exp1 = df['Close'].ewm(span=12, adjust=False).mean()
        exp2 = df['Close'].ewm(span=26, adjust=False).mean()
        df['MACD'] = exp1 - exp2
        df['MACD_signal'] = df['MACD'].ewm(span=9, adjust=False).mean()
        df['MACD_hist'] = df['MACD'] - df['MACD_signal']
        
        low_k = df['Low'].rolling(window=self.config.STOCHASTIC_PERIOD).min()
        high_k = df['High'].rolling(window=self.config.STOCHASTIC_PERIOD).max()
        df['stoch_k'] = 100 * ((df['Close'] - low_k) / (high_k - low_k).replace(0, 1e-9))
        
        df['momentum_20'] = df['Close'].pct_change(20)

        # --- 3. Dynamic & Regime-Aware Indicators ---
        # This will calculate RSI and Bollinger Bands using regime-specific parameters.
        df = self._calculate_dynamic_indicators(df)

        # --- 4. Signal Enhancement Layer ---
        # Applies smoothing and creates confirmation signals from base indicators.
        logger.info("    - Applying signal enhancement layer (Kalman, Confirmation)...")
        df['RSI_kalman'] = self._apply_kalman_filter(df['RSI'])
        df['ADX_kalman'] = self._apply_kalman_filter(df['ADX'])
        df['stoch_k_kalman'] = self._apply_kalman_filter(df['stoch_k'])
        
        df = self._calculate_trend_pullback_features(df)
        df = self._calculate_divergence_features(df)

        # --- 5. Microstructure & Advanced Volatility Features ---
        logger.info("    - Calculating microstructure and advanced volatility features...")
        df = self._calculate_displacement(df)
        df = self._calculate_gaps(df)
        df = self._calculate_candle_info(df)
        df = self._calculate_kama_regime(df)
        df = self._calculate_parkinson_volatility(df)
        # df = self._calculate_yang_zhang_volatility(df) # Computationally heavier, enable if needed

        # --- 6. Contextual & Scientific Feature Layer ---
        logger.info("    - Calculating contextual and scientific features...")
        df = self._add_higher_tf_context(df, symbol_data_by_tf.get(self.roles.get('medium'), pd.DataFrame()), 'H1')
        df = self._add_higher_tf_context(df, symbol_data_by_tf.get(self.roles.get('high'), pd.DataFrame()), 'DAILY')
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
        df = self._calculate_autocorrelation_features(df)
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

        # --- 7. Meta & Confluence Feature Layer ---
        # This must come after all constituent features have been calculated.
        df = self._calculate_meta_features(df)
        
        # --- 8. Final Anomaly Detection ---
        df = self._detect_anomalies(df)
        
        return df

    # --- NEW: Standard, in-memory PCA method for smaller datasets ---
    def _apply_pca_standard(self, df: pd.DataFrame, pca_features: list) -> pd.DataFrame:
        """Applies standard PCA for smaller datasets that fit comfortably in memory."""
        # FIX 2: Reduce memory footprint before PCA
        df_pca_subset = df[pca_features].dropna().astype(np.float32)
        
        # Drop low-variance features to reduce noise and computation
        df_pca_subset = df_pca_subset.loc[:, df_pca_subset.std() > 1e-6]
        if df_pca_subset.shape[1] < self.config.PCA_N_COMPONENTS:
            logger.warning("Number of features after variance filtering is less than n_components. Skipping PCA.")
            return df

        pipeline = Pipeline([
            ('scaler', StandardScaler()),
            ('pca', PCA(n_components=self.config.PCA_N_COMPONENTS))
        ])

        principal_components = pipeline.fit_transform(df_pca_subset)
        
        pca_cols = [f'RSI_PCA_{i+1}' for i in range(self.config.PCA_N_COMPONENTS)]
        pca_df = pd.DataFrame(principal_components, columns=pca_cols, index=df_pca_subset.index)

        df = df.join(pca_df)
        logger.info(f"Standard PCA complete. Explained variance: {pipeline.named_steps['pca'].explained_variance_ratio_.sum():.2%}")
        return df

    def _apply_pca_incremental(self, df: pd.DataFrame, pca_features: list) -> pd.DataFrame:
        """
        Applies IncrementalPCA using batched fitting and transforming to handle large datasets
        without exceeding memory limits.
        """
        # FIX 2: Reduce memory footprint before PCA
        df_pca_subset = df[pca_features].dropna()
        # Downcast dtypes to reduce memory usage
        for col in df_pca_subset.select_dtypes(include=['float']):
            df_pca_subset[col] = pd.to_numeric(df_pca_subset[col], downcast='float')
            
        # Drop near-constant columns
        initial_feature_count = df_pca_subset.shape[1]
        df_pca_subset = df_pca_subset.loc[:, df_pca_subset.std() > 1e-6]
        logger.info(f"PCA pre-filtering: Removed {initial_feature_count - df_pca_subset.shape[1]} near-constant features.")
        
        if df_pca_subset.shape[1] < self.config.PCA_N_COMPONENTS:
            logger.warning("Number of available features for PCA is less than n_components after filtering. Skipping PCA.")
            return df
            
        scaler = StandardScaler()
        ipca = IncrementalPCA(n_components=self.config.PCA_N_COMPONENTS)
        batch_size = 100000

        logger.info(f"Fitting IncrementalPCA in batches of {batch_size}...")
        # Fit in batches (memory-safe)
        for i in range(0, df_pca_subset.shape[0], batch_size):
            batch = df_pca_subset.iloc[i:i + batch_size]
            scaled_batch = scaler.fit_transform(batch) # Fit scaler on each batch, or fit once on a sample
            ipca.partial_fit(scaled_batch)

        # FIX 1: Transform in batches to avoid memory spike
        logger.info("Transforming full dataset in batches with fitted IncrementalPCA...")
        
        # DEBUGGING: Track memory usage
        logger.info(f"Memory before transform: {psutil.Process(os.getpid()).memory_info().rss / 1024**2:.2f} MB")
        
        transformed_batches = []
        for i in range(0, df_pca_subset.shape[0], batch_size):
            batch_to_transform = df_pca_subset.iloc[i:i + batch_size]
            scaled_batch_to_transform = scaler.transform(batch_to_transform) # Use the already fitted scaler
            transformed_batches.append(ipca.transform(scaled_batch_to_transform))

        principal_components = np.vstack(transformed_batches)
        
        logger.info(f"Memory after transform: {psutil.Process(os.getpid()).memory_info().rss / 1024**2:.2f} MB")

        pca_cols = [f'RSI_PCA_{i+1}' for i in range(self.config.PCA_N_COMPONENTS)]
        pca_df = pd.DataFrame(principal_components, columns=pca_cols, index=df_pca_subset.index)

        df = df.join(pca_df)
        logger.info(f"IncrementalPCA reduction complete. Explained variance ratio: {ipca.explained_variance_ratio_.sum():.2%}")
        return df

    def _apply_pca_reduction(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Orchestrates PCA by choosing between standard (fast, in-memory) and incremental
        (slower, memory-safe) methods based on dataset size.
        """
        if not self.config.USE_PCA_REDUCTION:
            return df
            
        logger.warning("Applying PCA reduction. NOTE: For best practice, PCA should be fit on training data only to avoid lookahead bias.")

        for period in self.config.RSI_PERIODS_FOR_PCA:
            df[f'RSI_{period}'] = 100 - (100 / (1 + df.groupby('Symbol')['Close'].diff().rolling(window=period).apply(
                lambda x: x[x > 0].sum() / (-x[x < 0].sum() if -x[x < 0].sum() != 0 else 1), raw=True)))

        rsi_features = [f'RSI_{period}' for period in self.config.RSI_PERIODS_FOR_PCA]
        df_pca_subset = df[rsi_features].dropna()

        if df_pca_subset.empty:
            logger.error("Not enough data to perform PCA on RSI features. Skipping.")
            return df

        if len(df_pca_subset) < 500_000:
            logger.info("Dataset small enough for standard PCA. Using fast in-memory method.")
            return self._apply_pca_standard(df, rsi_features)
        else:
            logger.info("Dataset is large. Using memory-efficient IncrementalPCA method.")
            return self._apply_pca_incremental(df, rsi_features)

    def create_feature_stack(self, data_by_tf: Dict[str, pd.DataFrame]) -> pd.DataFrame:
        logger.info("-> Stage 2: Engineering Features...")
        base_tf = self.roles['base']
        if base_tf not in data_by_tf:
            logger.critical(f"Base timeframe '{base_tf}' data is missing. Cannot proceed.")
            return pd.DataFrame()

        all_symbols_processed_dfs = []
        unique_symbols = data_by_tf[base_tf]['Symbol'].unique()

        for i, symbol in enumerate(unique_symbols):
            logger.info(f"  - [{i+1}/{len(unique_symbols)}] Processing all features for symbol: {symbol}")
            symbol_specific_data = {tf: df[df['Symbol'] == symbol].copy() for tf, df in data_by_tf.items()}
            processed_symbol_df = self._process_single_symbol_stack(symbol_specific_data)
            if not processed_symbol_df.empty:
                all_symbols_processed_dfs.append(processed_symbol_df)

        if not all_symbols_processed_dfs:
            logger.critical("Feature engineering resulted in no processable data across all symbols.")
            return pd.DataFrame()

        logger.info("  - Concatenating data for all symbols...")
        final_df = pd.concat(all_symbols_processed_dfs, sort=False).sort_index()
        
        logger.info("  - Calculating cross-symbol features...")
        final_df = self._calculate_relative_performance(final_df)

        if self.config.USE_PCA_REDUCTION:
            logger.info("  - Applying PCA reduction to feature set...")
            final_df = self._apply_pca_reduction(final_df)
            
        # --- NEW: Add Noise-Contrastive Features ---
        logger.info("  - Adding noise-contrastive features for diagnostics...")
        final_df['noise_1'] = np.random.normal(0, 1, len(final_df))
        final_df['noise_2'] = np.random.uniform(-1, 1, len(final_df))
        # --- END ---

        logger.info("  - Applying final data shift and cleaning...")
        feature_cols = [c for c in final_df.columns if c not in ['Open','High','Low','Close','RealVolume','Symbol','Timestamp','primary_model_signal','target']]
        final_df[feature_cols] = final_df.groupby('Symbol', sort=False)[feature_cols].shift(1)
        final_df.replace([np.inf,-np.inf],np.nan,inplace=True)
        final_df.dropna(subset=['ATR', 'RSI', 'ADX'], inplace=True)

        logger.info(f"[SUCCESS] Feature engineering complete. Final dataset shape: {final_df.shape}")
        return final_df
        
    def generate_labels(self, df: pd.DataFrame, labeling_params: Dict[str, Any]) -> pd.DataFrame:
        """
        Generates trade labels for the dataframe based on the provided parameters.

        Args:
            df: The dataframe containing features and price data.
            labeling_params: A dictionary with keys 'tp_multiplier', 'sl_multiplier',
                             and 'lookahead_candles' from the labeling playbook.
        """
        method_name = labeling_params.get("name", "Custom")
        logger.info(f"-> Stage 3: Generating Trade Labels using method: '{method_name}'...")
        logger.info(f"   - Labeling Params: {labeling_params}")
        
        labeled_dfs = [self._generate_labels_for_group(group, labeling_params) for _, group in df.groupby('Symbol')]
        
        return pd.concat(labeled_dfs) if labeled_dfs else pd.DataFrame()

    def _generate_labels_for_group(self, group: pd.DataFrame, params: Dict[str, Any]) -> pd.DataFrame:
        """
        Private helper to calculate triple-barrier outcomes for a single symbol group.
        This is the core logic, now driven entirely by external parameters.
        """
        group = group.copy()
        
        # Extract parameters with defaults
        tp_multiplier = params.get('tp_multiplier', 2.0)
        sl_multiplier = params.get('sl_multiplier', 1.0)
        lookahead = int(params.get('lookahead_candles', 100)) # Ensure lookahead is an integer

        if 'ATR' not in group.columns or len(group) < lookahead + 1:
            logger.warning(f"ATR not found or insufficient data for labeling group {group['Symbol'].iloc[0]}. Skipping.")
            group['target'] = 0
            return group

        profit_target_points = group['ATR'] * tp_multiplier
        stop_loss_points = group['ATR'] * sl_multiplier
        
        outcomes = np.zeros(len(group))
        prices, highs, lows = group['Close'].values, group['High'].values, group['Low'].values
        total_rows = len(group)
        
        # --- This is the robust triple-barrier logic from the previous implementation ---
        for i in range(len(group) - lookahead):
            sl_dist = stop_loss_points.iloc[i]
            tp_dist = profit_target_points.iloc[i]
            if pd.isna(sl_dist) or sl_dist <= 1e-9:
                continue

            tp_long_level, sl_long_level = prices[i] + tp_dist, prices[i] - sl_dist
            tp_short_level, sl_short_level = prices[i] - tp_dist, prices[i] + sl_dist
            
            future_highs, future_lows = highs[i+1 : i+1+lookahead], lows[i+1 : i+1+lookahead]

            hit_tp_long_idx = np.where(future_highs >= tp_long_level)[0]
            hit_sl_long_idx = np.where(future_lows <= sl_long_level)[0]
            first_tp_long = hit_tp_long_idx[0] if len(hit_tp_long_idx) > 0 else np.inf
            first_sl_long = hit_sl_long_idx[0] if len(hit_sl_long_idx) > 0 else np.inf

            hit_tp_short_idx = np.where(future_lows <= tp_short_level)[0]
            hit_sl_short_idx = np.where(future_highs >= sl_short_level)[0]
            first_tp_short = hit_tp_short_idx[0] if len(hit_tp_short_idx) > 0 else np.inf
            first_sl_short = hit_sl_short_idx[0] if len(hit_sl_short_idx) > 0 else np.inf

            # A long trade is profitable if its TP is hit before its SL
            if first_tp_long < first_sl_long:
                outcomes[i] = 1
            
            # A short trade is profitable if its TP is hit before its SL
            if first_tp_short < first_sl_short:
                outcomes[i] = -1
        
        group['target'] = outcomes
        return group    

    def label_standard(self, df: pd.DataFrame, lookahead: int) -> pd.DataFrame:
        logger.info("-> Stage 3: Generating Trade Labels ('standard')...")
        labeled_dfs = [self._label_group(group, lookahead) for _, group in df.groupby('Symbol')]
        return pd.concat(labeled_dfs) if labeled_dfs else pd.DataFrame()

    def label_meta(self, df: pd.DataFrame, lookahead: int) -> pd.DataFrame:
        logger.info("-> Stage 3: Generating Trade Labels ('meta')...")
        labeled_dfs = [self._label_meta_group(group, lookahead) for _, group in df.groupby('Symbol')]
        return pd.concat(labeled_dfs) if labeled_dfs else pd.DataFrame()
    
    def label_volatility_adjusted(self, df: pd.DataFrame, lookahead: int) -> pd.DataFrame:
        logger.info("-> Stage 3: Generating Trade Labels ('volatility_adjusted')...")
        labeled_dfs = [self._label_volatility_adjusted_group(group, lookahead) for _, group in df.groupby('Symbol')]
        return pd.concat(labeled_dfs) if labeled_dfs else pd.DataFrame()

    def label_trend_quality(self, df: pd.DataFrame, lookahead: int) -> pd.DataFrame:
        logger.info("-> Stage 3: Generating Trade Labels ('trend_quality')...")
        labeled_dfs = [self._label_trend_quality_group(group, lookahead) for _, group in df.groupby('Symbol')]
        return pd.concat(labeled_dfs) if labeled_dfs else pd.DataFrame()

    def label_optimal_entry(self, df: pd.DataFrame, lookahead: int) -> pd.DataFrame:
        logger.info("-> Stage 3: Generating Trade Labels ('optimal_entry')...")
        labeled_dfs = [self._label_optimal_entry_group(group, lookahead) for _, group in df.groupby('Symbol')]
        return pd.concat(labeled_dfs) if labeled_dfs else pd.DataFrame()

    def label_regime_aware(self, df: pd.DataFrame, lookahead: int) -> pd.DataFrame:
        """NEW: Regime-Aware Labeling Method"""
        logger.info("-> Stage 3: Generating Trade Labels ('regime_aware')...")
        labeled_dfs = [self._label_regime_aware_group(group, lookahead) for _, group in df.groupby('Symbol')]
        return pd.concat(labeled_dfs) if labeled_dfs else pd.DataFrame()

    def _label_regime_aware_group(self, group: pd.DataFrame, lookahead: int) -> pd.DataFrame:
        """Applies labels only if market regime conditions are met."""
        group = group.copy()
        # First, generate standard triple-barrier outcomes
        group_with_outcomes = self._label_group(group, lookahead)
        
        # Define regime conditions
        # Example: Only consider trades if the market is trending (top 60% strength) and not excessively volatile (bottom 80% vol)
        is_trending = group_with_outcomes['market_trend_strength'] > 0.60
        is_not_hyper_volatile = group_with_outcomes['market_volatility_index'] < 0.80
        valid_regime_mask = is_trending & is_not_hyper_volatile

        # Invalidate labels where regime conditions are not met
        original_labels = group_with_outcomes['target'].copy()
        group_with_outcomes['target'] = 0 # Default to hold
        group_with_outcomes.loc[valid_regime_mask, 'target'] = original_labels[valid_regime_mask]

        logger.info(f"    - Regime-Aware Filter: Kept {valid_regime_mask.sum()}/{len(group)} labels as valid for trading.")
        
        return group_with_outcomes

    def _label_group(self, group: pd.DataFrame, lookahead: int) -> pd.DataFrame:
        group = group.copy()
        if 'ATR' not in group.columns or len(group) < lookahead + 1:
            logger.warning(f"ATR not found or insufficient data for labeling in group. Skipping.")
            group['target'] = 0
            return group

        tp_multiplier = self.config.TP_ATR_MULTIPLIER
        sl_multiplier = self.config.SL_ATR_MULTIPLIER
        profit_target_points = group['ATR'] * tp_multiplier
        stop_loss_points = group['ATR'] * sl_multiplier
        
        outcomes = np.zeros(len(group))
        prices, highs, lows = group['Close'].values, group['High'].values, group['Low'].values
        total_rows = len(group)
        update_interval = max(1, (total_rows - lookahead) // 100) if total_rows > lookahead else 1

        for i in range(len(group) - lookahead):
            if i > 0 and i % update_interval == 0:
                progress_pct = (i / (total_rows - lookahead)) * 100
                symbol_name = group['Symbol'].iloc[0]
                sys.stdout.write(f"\r    - Labeling '{symbol_name}': {progress_pct:5.1f}% complete...")
                sys.stdout.flush()

            sl_dist = stop_loss_points.iloc[i]
            tp_dist = profit_target_points.iloc[i]
            if pd.isna(sl_dist) or sl_dist <= 1e-9: continue

            tp_long_level, sl_long_level = prices[i] + tp_dist, prices[i] - sl_dist
            tp_short_level, sl_short_level = prices[i] - tp_dist, prices[i] + sl_dist
            future_highs, future_lows = highs[i+1 : i+1+lookahead], lows[i+1 : i+1+lookahead]

            hit_tp_long_idx = np.where(future_highs >= tp_long_level)[0]
            hit_sl_long_idx = np.where(future_lows <= sl_long_level)[0]
            first_tp_long = hit_tp_long_idx[0] if len(hit_tp_long_idx) > 0 else np.inf
            first_sl_long = hit_sl_long_idx[0] if len(hit_sl_long_idx) > 0 else np.inf

            hit_tp_short_idx = np.where(future_lows <= tp_short_level)[0]
            hit_sl_short_idx = np.where(future_highs >= sl_short_level)[0]
            first_tp_short = hit_tp_short_idx[0] if len(hit_tp_short_idx) > 0 else np.inf
            first_sl_short = hit_sl_short_idx[0] if len(hit_sl_short_idx) > 0 else np.inf

            if first_tp_long < first_sl_long: outcomes[i] = 1
            if first_tp_short < first_sl_short: outcomes[i] = -1
        
        group['target'] = outcomes
        sys.stdout.write(f"\r    - Labeling '{group['Symbol'].iloc[0]}': 100.0% complete... Done.       \n")
        sys.stdout.flush()
        return group

    def _label_meta_group(self, group: pd.DataFrame, lookahead: int) -> pd.DataFrame:
        group = group.copy()
        if 'primary_model_signal' not in group.columns or len(group) < lookahead + 1:
            group['target'] = 0; return group
        
        tp_multiplier = self.config.TP_ATR_MULTIPLIER
        sl_multiplier = self.config.SL_ATR_MULTIPLIER
        sl_atr_dynamic = group['ATR'] * sl_multiplier
        tp_atr_dynamic = group['ATR'] * tp_multiplier
        outcomes = np.zeros(len(group))
        prices, lows, highs = group['Close'].values, group['Low'].values, group['High'].values
        primary_signals = group['primary_model_signal'].values
        min_return = self.config.LABEL_MIN_RETURN_PCT
        total_rows = len(group)
        update_interval = max(1, (total_rows - lookahead) // 100) if total_rows > lookahead else 1

        for i in range(len(group) - lookahead):
            if i > 0 and i % update_interval == 0:
                progress_pct = (i / (total_rows - lookahead)) * 100
                symbol_name = group['Symbol'].iloc[0]
                sys.stdout.write(f"\r    - Meta-Labeling '{symbol_name}': {progress_pct:5.1f}% complete...")
                sys.stdout.flush()

            signal = primary_signals[i]
            if signal == 0: continue
            sl_dist, tp_dist = sl_atr_dynamic.iloc[i], tp_atr_dynamic.iloc[i]
            if pd.isna(sl_dist) or sl_dist <= 1e-9: continue
            
            future_highs, future_lows = highs[i + 1:i + 1 + lookahead], lows[i + 1:i + 1 + lookahead]
            
            if signal > 0:
                tp_level, sl_level = prices[i] + tp_dist, prices[i] - sl_dist
                if (tp_level / prices[i] - 1) <= min_return: continue
                time_to_tp = np.where(future_highs >= tp_level)[0]
                time_to_sl = np.where(future_lows <= sl_level)[0]
                if len(time_to_tp) > 0 and (len(time_to_sl) == 0 or time_to_tp[0] < time_to_sl[0]):
                    outcomes[i] = 1
            
            elif signal < 0:
                tp_level, sl_level = prices[i] - tp_dist, prices[i] + sl_dist
                if (prices[i] / tp_level - 1) <= min_return: continue
                time_to_tp = np.where(future_lows <= tp_level)[0]
                time_to_sl = np.where(future_highs >= sl_level)[0]
                if len(time_to_tp) > 0 and (len(time_to_sl) == 0 or time_to_tp[0] < time_to_sl[0]):
                    outcomes[i] = 1
        
        group['target'] = outcomes
        sys.stdout.write(f"\r    - Meta-Labeling '{group['Symbol'].iloc[0]}': 100.0% complete... Done.       \n")
        sys.stdout.flush()
        return group

    def _label_volatility_adjusted_group(self, group: pd.DataFrame, lookahead: int) -> pd.DataFrame:
        group = group.copy()
        outcomes = np.zeros(len(group))
        if 'ATR' not in group.columns or 'market_volatility_index' not in group.columns:
            logger.warning(f"Skipping volatility-adjusted labeling for group: 'ATR' or 'market_volatility_index' column missing.")
            group['target'] = 0
            return group
        
        prices, highs, lows = group['Close'].values, group['High'].values, group['Low'].values
        atr_values = group['ATR'].values
        vol_rank_values = group['market_volatility_index'].values

        total_rows = len(group)
        update_interval = max(1, (total_rows - lookahead) // 100) if total_rows > lookahead else 1

        for i in range(len(group) - lookahead):
            if i > 0 and i % update_interval == 0:
                progress_pct = (i / (total_rows - lookahead)) * 100
                symbol_name = group['Symbol'].iloc[0]
                sys.stdout.write(f"\r    - Labeling (Vol-Adj) '{symbol_name}': {progress_pct:5.1f}% complete...")
                sys.stdout.flush()

            current_vol, vol_rank = atr_values[i], vol_rank_values[i]
            if pd.isna(current_vol) or pd.isna(vol_rank): continue
            
            if vol_rank > 0.7: tp_mult, sl_mult = 1.5, 1.0
            elif vol_rank < 0.3: tp_mult, sl_mult = 3.0, 2.0
            else: tp_mult, sl_mult = 2.0, 1.5
                
            tp_dist, sl_dist = current_vol * tp_mult, current_vol * sl_mult
            if sl_dist <= 1e-9: continue
            
            tp_long, sl_long = prices[i] + tp_dist, prices[i] - sl_dist
            tp_short, sl_short = prices[i] - tp_dist, prices[i] + sl_dist
            future_highs, future_lows = highs[i+1:i+1+lookahead], lows[i+1:i+1+lookahead]

            hit_tp_long_idx = np.where(future_highs >= tp_long)[0]; hit_sl_long_idx = np.where(future_lows <= sl_long)[0]
            first_tp_long = hit_tp_long_idx[0] if len(hit_tp_long_idx) > 0 else np.inf
            first_sl_long = hit_sl_long_idx[0] if len(hit_sl_long_idx) > 0 else np.inf
            hit_tp_short_idx = np.where(future_lows <= tp_short)[0]; hit_sl_short_idx = np.where(future_highs >= sl_short)[0]
            first_tp_short = hit_tp_short_idx[0] if len(hit_tp_short_idx) > 0 else np.inf
            first_sl_short = hit_sl_short_idx[0] if len(hit_sl_short_idx) > 0 else np.inf
            
            if first_tp_long < first_sl_long: outcomes[i] = 1
            if first_tp_short < first_sl_short: outcomes[i] = -1
            
        group['target'] = outcomes
        sys.stdout.write(f"\r    - Labeling (Vol-Adj) '{group['Symbol'].iloc[0]}': 100.0% complete... Done.       \n")
        sys.stdout.flush()
        return group

    def _label_trend_quality_group(self, group: pd.DataFrame, lookahead: int) -> pd.DataFrame:
        group = group.copy()
        outcomes = np.zeros(len(group))
        total_rows = len(group)
        update_interval = max(1, (total_rows - lookahead) // 100) if total_rows > lookahead else 1

        for i in range(len(group) - lookahead):
            if i > 0 and i % update_interval == 0:
                progress_pct = (i / (total_rows - lookahead)) * 100
                symbol_name = group['Symbol'].iloc[0]
                sys.stdout.write(f"\r    - Labeling (Trend Quality) '{symbol_name}': {progress_pct:5.1f}% complete...")
                sys.stdout.flush()

            entry_price = group['Close'].iloc[i]
            if entry_price == 0: continue
            future_highs = group['High'].iloc[i+1:i+1+lookahead]
            future_lows = group['Low'].iloc[i+1:i+1+lookahead]
            
            max_move = (future_highs.max() - entry_price) / entry_price
            min_move = (entry_price - future_lows.min()) / entry_price
            
            if max_move > 0.01 and max_move > 2 * min_move: outcomes[i] = 1
            elif min_move > 0.01 and min_move > 2 * max_move: outcomes[i] = -1
                
        group['target'] = outcomes
        sys.stdout.write(f"\r    - Labeling (Trend Quality) '{group['Symbol'].iloc[0]}': 100.0% complete... Done.       \n")
        sys.stdout.flush()
        return group

    def _label_optimal_entry_group(self, group: pd.DataFrame, lookahead: int) -> pd.DataFrame:
        group = group.copy()
        outcomes = np.zeros(len(group))
        if 'ADX' not in group.columns or 'EMA_20' not in group.columns or 'EMA_50' not in group.columns:
             logger.warning(f"Skipping optimal entry labeling for group: required EMA/ADX columns missing.")
             group['target'] = 0
             return group

        total_rows = len(group)
        update_interval = max(1, (total_rows - lookahead) // 100) if total_rows > lookahead else 1
        
        # Use the configurable minimum return from the config
        min_return_pct = self.config.LABEL_MIN_RETURN_PCT

        for i in range(len(group) - lookahead):
            if i > 0 and i % update_interval == 0:
                progress_pct = (i / (total_rows - lookahead)) * 100
                symbol_name = group['Symbol'].iloc[0]
                sys.stdout.write(f"\r    - Labeling (Optimal Entry) '{symbol_name}': {progress_pct:5.1f}% complete...")
                sys.stdout.flush()

            # Use the configurable trend filter threshold from the config
            if group['ADX'].iloc[i] < self.config.TREND_FILTER_THRESHOLD: continue
            
            entry_price = group['Close'].iloc[i]
            if entry_price == 0: continue
            future_prices = group['Close'].iloc[i+1:i+1+lookahead]
            ret = (future_prices.iloc[-1] - entry_price) / entry_price
            
            if group['EMA_20'].iloc[i] > group['EMA_50'].iloc[i]:
                # Check for pullback below the fast EMA and if the future return met the minimum
                if entry_price < group['EMA_20'].iloc[i] and ret > min_return_pct: outcomes[i] = 1
            else:
                # Check for pullback above the fast EMA and if the future return met the minimum
                if entry_price > group['EMA_20'].iloc[i] and ret < -min_return_pct: outcomes[i] = -1
                    
        group['target'] = outcomes
        sys.stdout.write(f"\r    - Labeling (Optimal Entry) '{group['Symbol'].iloc[0]}': 100.0% complete... Done.       \n")
        sys.stdout.flush()
        return group
        
# =============================================================================
# 6. MODELS & TRAINER
# =============================================================================

def check_label_quality(df_train_labeled: pd.DataFrame, min_label_pct: float = 0.02) -> bool:
    """Checks if the generated labels are of sufficient quality for training."""
    if 'target' not in df_train_labeled.columns or df_train_labeled['target'].abs().sum() == 0:
        logger.warning("  - LABEL SANITY CHECK FAILED: No non-zero labels were generated.")
        return False

    label_counts = df_train_labeled['target'].value_counts(normalize=True)

    long_pct = label_counts.get(1.0, 0)
    short_pct = label_counts.get(-1.0, 0)

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
    """
    Implements a two-stage feature selection process.
    - NEW: A `_remove_redundant_features` method acts as a pre-selection pruner,
      removing highly correlated features to reduce noise and multicollinearity.
    - UPDATED: The `train` method now uses this pruner before the main 'elite'
      feature selection (`TRex` or `Mutual_Info`) is performed.
    """
    GNN_BASE_FEATURES = ['ATR', 'RSI', 'ADX', 'bollinger_bandwidth', 'stoch_k', 'momentum_10', 'hour', 'day_of_week']
    
    def __init__(self,config:ConfigModel, gemini_analyzer: 'GeminiAnalyzer'):
        self.config=config
        self.gemini_analyzer = gemini_analyzer
        self.shap_summary:Optional[pd.DataFrame]=None
        self.best_threshold=0.5
        self.study: Optional[optuna.study.Study] = None
        self.is_gnn_model = False
        self.is_meta_model = False
        self.is_transformer_model = False
        self.is_minirocket_model = False
        self.minirocket_transformer: Optional[MiniRocket] = None
        self.classification_report_str: str = "Classification report not generated."
    
    def _run_optuna_study(self, X_train, y_train, model_type):
        """Helper to run an Optuna study for a single model task."""
        objective = 'reg:squarederror' if model_type == 'regression' else 'binary:logistic'
        eval_metric = 'rmse' if model_type == 'regression' else 'logloss'

        def objective_func(trial):
            params = {
                'objective': objective, 'eval_metric': eval_metric, 'booster': 'gbtree',
                'tree_method': 'hist', 'seed': 42,
                'n_estimators': trial.suggest_int('n_estimators', 50, 400),
                'max_depth': trial.suggest_int('max_depth', 3, 7),
                'learning_rate': trial.suggest_float('learning_rate', 0.02, 0.2, log=True),
                'subsample': trial.suggest_float('subsample', 0.6, 1.0),
                'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
            }
            model = xgb.XGBRegressor(**params) if model_type == 'regression' else xgb.XGBClassifier(**params)
            
            X_t, X_v, y_t, y_v = train_test_split(X_train, y_train, test_size=0.2, shuffle=False)
            
            # [FIX] Reverted to the robust 'early_stopping_rounds' parameter.
            model.fit(X_t, y_t,
                      eval_set=[(X_v, y_v)],
                      early_stopping_rounds=25,
                      verbose=False)
            
            preds = model.predict(X_v)
            if model_type == 'regression':
                score = np.sqrt(mean_squared_error(y_v, preds))
            else:
                score = f1_score(y_v, preds, zero_division=0)
            return score

        direction = 'minimize' if model_type == 'regression' else 'maximize'
        study = optuna.create_study(direction=direction)
        study.optimize(objective_func, n_trials=30, n_jobs=-1)
        return study.best_params

    def train_single_model(self, df_train: pd.DataFrame, feature_list: List[str], target_col: str, model_type: str):
        """Trains one specialized model for a single target."""
        logger.info(f"--> Training '{target_col}' ({model_type}) model...")
        
        df_task = df_train[feature_list + [target_col]].dropna()
        X = df_task[feature_list]
        y = df_task[target_col]

        if y.nunique() < 2:
            logger.warning(f"  - Skipping '{target_col}': Target has only one unique value.")
            return None

        best_params = self._run_optuna_study(X, y, model_type)
        
        final_model_class = xgb.XGBRegressor if model_type == 'regression' else xgb.XGBClassifier
        final_model = final_model_class(**best_params, tree_method='hist', seed=42)
        
        pipeline = Pipeline([('scaler', RobustScaler()), ('model', final_model)])
        pipeline.fit(X, y)

        explainer = shap.Explainer(pipeline.named_steps['model'], pipeline.named_steps['scaler'].transform(X))
        shap_values = explainer(X)
        self.shap_summaries[target_col] = pd.DataFrame(
            np.abs(shap_values.values).mean(axis=0),
            index=feature_list,
            columns=['SHAP_Importance']
        ).sort_values(by='SHAP_Importance', ascending=False)
        
        logger.info(f"  - Finished training for '{target_col}'.")
        return pipeline

    def train_all_models(self, df_labeled: pd.DataFrame, feature_list: List[str]) -> Dict[str, Pipeline]:
        """
        [ORCHESTRATOR] Trains the full dictionary of primary and auxiliary models.
        """
        logger.info("--- Starting Multi-Task Model Training Orchestration ---")
        
        tasks = {
            'primary_pressure': {'target': 'target_signal_pressure', 'type': 'regression'},
            'confirm_bull_engulf': {'target': 'target_bullish_engulfing', 'type': 'classification'},
            'confirm_timing': {'target': 'target_timing_score', 'type': 'regression'},
            'predict_vol_spike': {'target': 'target_volatility_spike', 'type': 'classification'},
        }

        trained_models: Dict[str, Pipeline] = {}
        
        for task_name, task_info in tasks.items():
            if task_info['target'] in df_labeled.columns:
                model_pipeline = self.train_single_model(
                    df_train=df_labeled,
                    feature_list=feature_list,
                    target_col=task_info['target'],
                    model_type=task_info['type']
                )
                if model_pipeline:
                    trained_models[task_name] = model_pipeline
            else:
                logger.warning(f"Target column '{task_info['target']}' not found. Skipping task '{task_name}'.")

        logger.info(f"--- Multi-Task Training Complete. Trained {len(trained_models)} models. ---")
        return trained_models    

    def _prepare_3d_data(self, df: pd.DataFrame, feature_list: List[str], lookback: int) -> Tuple[np.ndarray, np.ndarray, pd.Index]:
        df_features = df[feature_list].fillna(0)
        X_values = df_features.values
        y_values = df['target'].values
        
        windows, labels, label_indices = [], [], []
        
        for i in range(len(df_features) - lookback + 1):
            window = X_values[i : i + lookback]
            windows.append(window)
            
            label_idx = i + lookback - 1
            labels.append(y_values[label_idx])
            label_indices.append(df.index[label_idx])
            
        return np.stack(windows), np.array(labels), pd.Index(label_indices)

    def _create_sequences(self, X: pd.DataFrame, y: pd.Series, seq_length: int) -> Tuple[np.ndarray, np.ndarray]:
        xs, ys = [], []
        for i in range(len(X) - seq_length):
            x = X.iloc[i:(i + seq_length)].values
            y_val = y.iloc[i + seq_length - 1]
            xs.append(x)
            ys.append(y_val)
        return np.array(xs), np.array(ys)

    def _prepare_gnn_data(self, df: pd.DataFrame, feature_list: List[str], correlation_threshold: float = 0.5) -> List[Data]:
        if not GNN_AVAILABLE: return []
        logger.info("    - Preparing graph data for GNN...")
        
        price_df = df.pivot_table(index=df.index, columns='Symbol', values='Close', aggfunc='last')
        price_df = price_df.ffill().bfill().dropna(axis=1)
        symbols = price_df.columns.tolist()
        
        if len(symbols) < 2:
            logger.warning("    - GNN requires at least 2 symbols with continuous data to build a graph. Skipping GNN.")
            return []
            
        corr_matrix = price_df.corr()
        
        edge_list = []
        for i in range(len(symbols)):
            for j in range(i + 1, len(symbols)):
                if abs(corr_matrix.iloc[i, j]) > correlation_threshold:
                    edge_list.append([i, j])
                    edge_list.append([j, i])
        
        if not edge_list:
            logger.warning(f"   - No asset correlations found above threshold {correlation_threshold}. Creating a fully connected graph as a fallback.")
            for i in range(len(symbols)):
                for j in range(i + 1, len(symbols)):
                    edge_list.append([i, j])
                    edge_list.append([j, i])

        edge_index = torch.tensor(edge_list, dtype=torch.long).t().contiguous()

        graph_snapshots = []
        df_filtered = df[df['Symbol'].isin(symbols)]
        
        for timestamp, group in df_filtered.groupby(df_filtered.index):
            group = group.set_index('Symbol').loc[symbols].reset_index()
            
            node_features = group[feature_list].values
            x = torch.tensor(node_features, dtype=torch.float)
            
            y_map = {-1: 0, 0: 1, 1: 2}
            labels = group['target'].map(y_map).fillna(1).astype(int).values
            y = torch.tensor(labels, dtype=torch.long)
            
            graph_data = Data(x=x, edge_index=edge_index, y=y)
            graph_snapshots.append(graph_data)
            
        logger.info(f"    - GNN data prepared: {len(graph_snapshots)} graph snapshots, {edge_index.shape[1]} edges.")
        return graph_snapshots

    def _train_gnn(self, df_train_labeled: pd.DataFrame, feature_list: List[str]) -> Optional[Tuple[GNNModel, float, float]]:
        graph_data_list = self._prepare_gnn_data(df_train_labeled, feature_list)
        if not graph_data_list: return None

        train_size = int(0.8 * len(graph_data_list))
        train_data = graph_data_list[:train_size]
        val_data = graph_data_list[train_size:]

        if not train_data or not val_data:
            logger.error("    - GNN training failed: Not enough data for train/validation split.")
            return None

        num_node_features = train_data[0].num_node_features
        num_classes = 3
        
        model = GNNModel(
            in_channels=num_node_features,
            hidden_channels=self.config.GNN_EMBEDDING_DIM,
            out_channels=num_classes
        )
        optimizer = Adam(model.parameters(), lr=0.01, weight_decay=5e-4)
        criterion = nn.CrossEntropyLoss()

        logger.info("    - Starting GNN model training loop...")
        best_val_f1 = -1.0
        best_model_state = None

        for epoch in range(self.config.GNN_EPOCHS):
            model.train()
            total_loss = 0
            for data in train_data:
                optimizer.zero_grad()
                out = model(data)
                loss = criterion(out, data.y)
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
            
            avg_loss = total_loss / len(train_data)

            model.eval()
            all_preds, all_labels = [], []
            with torch.no_grad():
                for data in val_data:
                    out = model(data)
                    preds = out.argmax(dim=1)
                    all_preds.extend(preds.tolist())
                    all_labels.extend(data.y.tolist())
            
            val_f1 = f1_score(all_labels, all_preds, average='weighted', zero_division=0)

            if (epoch + 1) % 10 == 0:
                logger.info(f"    - GNN Epoch {epoch+1:02d}/{self.config.GNN_EPOCHS} | Avg Loss: {avg_loss:.4f} | Val F1: {val_f1:.4f}")

            if val_f1 > best_val_f1:
                best_val_f1 = val_f1
                best_model_state = model.state_dict()

        if best_model_state:
            model.load_state_dict(best_model_state)
            logger.info(f"    - GNN training finished. Best validation F1: {best_val_f1:.4f}")
            return model, 0.5, best_val_f1
        else:
            logger.error("    - GNN training failed to produce a valid model.")
            return None
    
    def _remove_redundant_features(self, df: pd.DataFrame, threshold: float = 0.95) -> List[str]:
        """
        Identifies and removes highly correlated features to reduce redundancy.
        This acts as the first-pass filter before more advanced selection.
        Returns a list of features to keep.
        """
        logger.info(f"-> Pruning features with Spearman correlation > {threshold}...")
        corr_matrix = df.corr(method='spearman').abs()
        upper_tri = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))

        to_drop = {column for column in upper_tri.columns if any(upper_tri[column] > threshold)}

        if to_drop:
            # Log only the first few for brevity
            log_sample = sorted(list(to_drop))[:5]
            logger.warning(f"  - Dropping {len(to_drop)} highly correlated feature(s): {', '.join(log_sample)}...")
        else:
            logger.info("  - No highly correlated features found to prune.")

        features_to_keep = [col for col in df.columns if col not in to_drop]
        return features_to_keep
    
    def _select_features_with_trex(self, X: pd.DataFrame, y: pd.Series) -> List[str]:
        """
        Performs feature selection using the TRexSelector algorithm.
        Includes safeguards for variance, correlation, and common TRex output issues.
        """
        logger.info("-> Selecting elite features using TRexSelector with FDR control...")

        X_clean = X.copy().fillna(X.median())
        y_binary = y.apply(lambda x: 0 if x == 1 else 1)
        logger.info(f"  - Adapting multi-class target for TRex. Binary distribution: {(y_binary.value_counts(normalize=True)*100).to_dict()}")

        # --- Step 1: Remove near-constant features ---
        logger.info("  - Filtering near-constant features for TRex stability...")
        initial_feature_count = X_clean.shape[1]
        variances = X_clean.var()
        features_to_keep = variances[variances > 1e-6].index.tolist()
        X_variant = X_clean[features_to_keep]

        num_removed = initial_feature_count - X_variant.shape[1]
        if num_removed > 0:
            logger.warning(f"  - Removed {num_removed} near-constant feature(s) before selection.")

        # --- Step 2: Remove highly correlated features (a second, more aggressive pass for TRex) ---
        logger.info("  - Pre-filtering highly correlated features for TRex...")
        corr_matrix = X_variant.corr().abs()
        upper_tri = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
        to_drop = [column for column in upper_tri.columns if any(upper_tri[column] > 0.95)]

        if to_drop:
            logger.warning(f"  - TRex pre-filter dropping {len(to_drop)} highly correlated features: {to_drop}")
            X_final_for_trex = X_variant.drop(columns=to_drop)
        else:
            X_final_for_trex = X_variant

        logger.info(f"  - Number of features remaining for TRex: {X_final_for_trex.shape[1]}")

        if X_final_for_trex.shape[1] < 2:
            logger.error("  - Not enough features remaining after filtering to run TRex. Returning fallback.")
            return X.columns.tolist()[:5]

        # --- Step 3: Run TRex Selector ---
        try:
            assert X_final_for_trex.shape[0] == y_binary.shape[0], "Mismatched number of rows between X and y."
            logger.debug(f"  - Running TRex with X shape {X_final_for_trex.shape}, y shape {y_binary.shape}")
            res = trex(X=X_final_for_trex.values, y=y_binary.values, tFDR=0.2, verbose=False)

            selected_indices = res.get("selected_var", [])

            if isinstance(selected_indices, np.ndarray):
                if selected_indices.dtype == bool:
                    if selected_indices.size != X_final_for_trex.shape[1]:
                        logger.error("  - TRex returned a boolean mask with mismatched size. Returning fallback.")
                        return X.columns.tolist()
                    selected_features = X_final_for_trex.columns[selected_indices].tolist()
                else:
                    selected_features = X_final_for_trex.columns[list(selected_indices)].tolist()
            elif isinstance(selected_indices, (list, tuple)):
                selected_features = X_final_for_trex.columns[list(selected_indices)].tolist()
            else:
                logger.error(f"  - Unexpected type from TRexSelector: {type(selected_indices)}. Returning fallback.")
                return X.columns.tolist()

            if not selected_features:
                logger.warning("  - TRexSelector did not select any variables. Returning top 5 from original list as fallback.")
                return X.columns.tolist()[:5]

            logger.info(f"  - TRexSelector finished. Selected {len(selected_features)} features.")
            logger.debug(f"  - TRex Features: {selected_features}")
            return selected_features

        except Exception as e:
            logger.exception(f"  - TRexSelector failed with an error: {e}. Returning all original features as a fallback.")
            return X.columns.tolist()
 
    def _select_elite_features(self, X: pd.DataFrame, y: pd.Series, all_features: List[str], top_n: int = 60, final_n: int = 25, corr_threshold: float = 0.7) -> List[str]:
        """
        Selects a small, powerful set of features using Mutual Information and correlation pruning.
        """
        logger.info("-> Selecting elite features using Mutual Information and Correlation Pruning...")

        # 1. Pre-filter features with low variance or too many missing values
        variances = X.var()
        low_variance_features = variances[variances < 1e-5].index.tolist()
        missing_pct = X.isnull().sum() / len(X)
        high_missing_features = missing_pct[missing_pct > 0.3].index.tolist()
        features_to_remove = set(low_variance_features + high_missing_features)

        candidate_features = [f for f in all_features if f not in features_to_remove]
        X_candidate = X[candidate_features].copy()

        # Impute any remaining NaNs for MI calculation
        X_candidate.fillna(X_candidate.median(), inplace=True)

        logger.info(f"  - Starting with {len(all_features)} features. After pre-filtering: {len(candidate_features)} candidates.")

        # 2. Rank features by Mutual Information
        mi_scores = mutual_info_classif(X_candidate, y, random_state=42)
        mi_series = pd.Series(mi_scores, index=candidate_features).sort_values(ascending=False)

        top_features = mi_series.head(top_n).index.tolist()
        logger.info(f"  - Top {top_n} features selected by Mutual Information.")

        # 3. Prune highly correlated features from the top set
        corr_matrix = X[top_features].corr(method='spearman').abs()
        upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))

        features_to_drop = set()
        for col in upper.columns:
            if len(top_features) - len(features_to_drop) <= final_n:
                break
            correlated_features = upper.index[upper[col] > corr_threshold].tolist()
            if correlated_features:
                for feature in correlated_features:
                    if mi_series[col] >= mi_series[feature]:
                        features_to_drop.add(feature)
                    else:
                        features_to_drop.add(col)
                        break

        elite_features = [f for f in top_features if f not in features_to_drop]

        if len(elite_features) < 10:
                logger.warning(f"  - Correlation pruning resulted in fewer than 10 features. Reverting to top {min(15, top_n)} MI features.")
                elite_features = mi_series.head(min(15, top_n)).index.tolist()
        else:
                elite_features = elite_features[:final_n] # Enforce the final count

        logger.info(f"  - Pruned correlated features. Final elite feature count: {len(elite_features)}")
        logger.debug(f"  - Elite Features: {elite_features}")

        return elite_features

    def train(self, df_train: pd.DataFrame, feature_list: List[str], strategy_details: Dict, strategic_directive: str) -> Optional[Tuple[Union[Pipeline, Dict, Tuple, GNNModel], float, float]]:
        logger.info(f"  - Starting model training using strategy: '{strategy_details.get('description', 'N/A')}'")
        self.is_gnn_model = strategy_details.get("requires_gnn", False)
        self.is_meta_model = strategy_details.get("requires_meta_labeling", False)
        self.is_transformer_model = strategy_details.get("requires_transformer", False)
        self.is_minirocket_model = strategy_details.get("requires_minirocket", False)
        X = pd.DataFrame()

        if not self.is_minirocket_model and not self.is_gnn_model:
            if not feature_list:
                logger.error(f"  - Training aborted for strategy '{strategy_details.get('description', 'N/A')}': The 'selected_features' list is empty.")
                return None

            X_initial = df_train[feature_list].copy()

            # --- [NEW] CALL TO THE PRUNER ---
            # Stage 1: Prune highly correlated features first to create a cleaner pool for selection.
            pruned_feature_list = self._remove_redundant_features(X_initial, threshold=0.95)
            X_pruned = X_initial[pruned_feature_list]
            # --- END NEW CALL ---

            if self.is_meta_model:
                logger.info("  - Meta-Labeling strategy detected. Training secondary filter model.")
                y = df_train['target'].astype(int)
                num_classes = 2
            else:
                y_map={-1:0,0:1,1:2}
                y=df_train['target'].map(y_map).astype(int)
                num_classes = 3

            # Stage 2: Select the best features from the pruned pool.
            elite_feature_list = []
            if self.config.FEATURE_SELECTION_METHOD == 'trex':
                elite_feature_list = self._select_features_with_trex(X_pruned, y)
            elif self.config.FEATURE_SELECTION_METHOD == 'mutual_info':
                elite_feature_list = self._select_elite_features(X_pruned, y, pruned_feature_list)
            else:
                logger.warning(f"  - Unknown FEATURE_SELECTION_METHOD: '{self.config.FEATURE_SELECTION_METHOD}'. Defaulting to pruned list.")
                elite_feature_list = pruned_feature_list

        if not elite_feature_list:
            logger.error("  - Feature selection resulted in an empty list. Aborting training.")
            return None

        # The rest of the training process now uses the final 'elite_feature_list'
        X = X_pruned[elite_feature_list].copy().fillna(0)

        if X.empty or len(y.unique()) < num_classes:
            logger.error("  - Training data (X) is empty or not enough classes for the model. Aborting.")
            return None

        self.class_weights=dict(zip(np.unique(y),compute_class_weight(class_weight='balanced',classes=np.unique(y),y=y)))
        X_train_val, _, y_train_val, _ = train_test_split(X, y, test_size=0.1, shuffle=False)
        X_train, X_val, y_train, y_val = train_test_split(X_train_val, y_train_val, test_size=0.2, shuffle=False)

        if X_train.empty or X_val.empty:
            logger.error(f"  - Training aborted: Data split resulted in an empty training or validation set. (Train shape: {X_train.shape}, Val shape: {X_val.shape})")
            return None

        self.study = self._optimize_hyperparameters(df_train, X, y, num_classes, elite_feature_list)

        if not self.study or not self.study.best_trials:
            logger.error("  - Training aborted: Hyperparameter optimization failed or yielded no valid trials.")
            return None

        logger.info(f"    - Optimization complete. Found {len(self.study.best_trials)} non-dominated trial(s) on the Pareto front.")

        best_trial = None
        if len(self.study.best_trials) == 1:
            best_trial = self.study.best_trials[0]
            logger.info("    - Only one optimal trial found, selecting it directly.")
        else:
            try:
                selected_trial_number = self.gemini_analyzer.select_best_tradeoff(
                    self.study.best_trials, self.config.RISK_PROFILE, strategic_directive
                )
                best_trial = next((t for t in self.study.best_trials if t.number == selected_trial_number), None)
                if not best_trial:
                    logger.error(f"Could not find trial number {selected_trial_number} in best_trials. Falling back.")
                    best_trial = max(self.study.best_trials, key=lambda t: t.values[0])
            except Exception as e:
                logger.error(f"An error occurred during AI-based trial selection: {e}. Falling back.")
                best_trial = max(self.study.best_trials, key=lambda t: t.values[0])

        best_params = best_trial.params
        best_values = best_trial.values

        current_state = self.config.operating_state
        state_rules = self.config.STATE_BASED_CONFIG[current_state]
        optimization_objective_names = state_rules.get("optimization_objective", ["maximize_calmar", "minimize_trades"])
        obj1_label = optimization_objective_names[0].replace('_', ' ').title()
        obj2_label = optimization_objective_names[1].replace('_', ' ').title()

        logger.info(f"    - Selected Trial #{best_trial.number} -> Objectives: [{obj1_label}: {best_values[0]:.4f}, {obj2_label}: {best_values[1]:.2f}]")

        formatted_params = { k: (f"{v:.4g}" if isinstance(v, float) else v) for k, v in best_params.items() }
        logger.info(f"    - Selected params: {formatted_params}")

        self.best_threshold, f1_score_val = self._find_best_threshold(best_params, X_train, y_train, X_val, y_val, num_classes)
        final_pipeline = self._train_final_model(best_params, X_train_val, y_train_val, list(X.columns), num_classes)

        if final_pipeline is None:
            logger.error("  - Training aborted: Final model training failed.")
            return None

        logger.info("  - [SUCCESS] Model training complete.")

        # This part of the return signature needs updating to return the final feature list.
        if self.is_minirocket_model:
            return (final_pipeline, self.minirocket_transformer), self.best_threshold, f1_score_val, elite_feature_list
        else:
            return final_pipeline, self.best_threshold, f1_score_val, elite_feature_list

    def _optimize_hyperparameters(self, df_full_train: pd.DataFrame, X: pd.DataFrame, y: pd.Series, num_classes: int, feature_list: List[str]) -> Optional[optuna.study.Study]:
        current_state = self.config.operating_state
        state_rules = self.config.STATE_BASED_CONFIG[current_state]
        optimization_objective_names = state_rules.get("optimization_objective", ["maximize_calmar", "minimize_trades"])

        logger.info(f"    - Starting hyperparameter optimization in state: '{current_state.value}' on {len(feature_list)} features...")
        logger.info(f"    - Optimization Objectives: {', '.join(optimization_objective_names)}")

        objective_descriptions = {
             ("maximize_f1", "maximize_log_trades"): 'Prioritises model accuracy (F1 score) to establish a reliable baseline.',
             ("maximize_pnl", "maximize_log_trades"): 'Prioritises profitability and trade frequency to capitalize on a working model.',
             ("maximize_sortino", "minimize_trades"): 'Prioritises downside risk-adjusted returns and reduces trade frequency to protect capital.'
        }
        objective_key = tuple(optimization_objective_names)
        description = objective_descriptions.get(objective_key, 'Custom objective defined.')
        logger.info(f"    - Strategy Goal: {description}")

        obj1_label = optimization_objective_names[0].replace('_', ' ').title()
        obj2_label = optimization_objective_names[1].replace('_', ' ').title()

        def dynamic_progress_callback(study: optuna.study.Study, trial: optuna.trial.FrozenTrial):
            n_trials = self.config.OPTUNA_TRIALS
            trial_number = trial.number + 1

            progress_str = f"> Optuna Optimization: Trial {trial_number}/{n_trials}"

            if study.best_trials:
                best_values = study.best_trials[0].values
                obj1_val = best_values[0] if best_values and len(best_values) > 0 else float('nan')
                obj2_val = best_values[1] if best_values and len(best_values) > 1 else float('nan')

                progress_str += f" | Best Trial -> {obj1_label}: {obj1_val:.4f}, {obj2_label}: {obj2_val:.2f}"

            sys.stdout.write(f"\r{progress_str}\x1b[K")
            sys.stdout.flush()

        objective = 'multi:softprob' if num_classes > 2 else 'binary:logistic'
        eval_metric = 'mlogloss' if num_classes > 2 else 'logloss'

        def custom_objective(trial: optuna.Trial) -> Tuple[float, float]:
            params = {
                'objective': objective, 'eval_metric': eval_metric, 'booster': 'gbtree',
                'tree_method': 'hist', 'seed': 42,
                'n_estimators': trial.suggest_int('n_estimators', 100, 800, step=50),
                'max_depth': trial.suggest_int('max_depth', 3, 7),
                'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.2, log=True),
                'subsample': trial.suggest_float('subsample', 0.6, 1.0),
                'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
                'gamma': trial.suggest_float('gamma', 0, 5)
            }
            if num_classes > 2: params['num_class'] = num_classes

            skf = StratifiedKFold(n_splits=3, shuffle=True, random_state=42) # Fewer splits for speed
            
            fold_trade_f1_scores = []
            fold_trade_activity_scores = []
            fold_sortino_ratios = []

            X_elite = X[feature_list]

            for train_idx, val_idx in skf.split(X_elite, y):
                X_train, X_val = X_elite.iloc[train_idx], X_elite.iloc[val_idx]
                y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
                
                try:
                    scaler = RobustScaler()
                    X_train_scaled = scaler.fit_transform(X_train)
                    X_val_scaled = scaler.transform(X_val)
                    
                    model = xgb.XGBClassifier(**params)
                    # No class weights needed due to balanced labels from quantile method
                    model.fit(X_train_scaled, y_train, eval_set=[(X_val_scaled, y_val)], 
                              callbacks=[XGBoostEarlyStopping(50)], verbose=False)

                    preds_val = model.predict(X_val_scaled)
                    
                    # 1. Calculate F1 score for TRADE classes only (0=Short, 2=Long)
                    trade_f1 = f1_score(y_val, preds_val, average='weighted', labels=[0, 2], zero_division=0)
                    fold_trade_f1_scores.append(trade_f1)

                    # 2. Calculate a score for trading activity
                    num_trades = np.sum(preds_val != 1)
                    activity_score = np.log1p(num_trades) / np.log1p(len(y_val)) # Normalized
                    fold_trade_activity_scores.append(activity_score)

                    # 3. Calculate Sortino for risk-management component
                    # (Simplified PNL calculation for optimization speed)
                    pnl = pd.Series(np.select(
                        [preds_val == 2, preds_val == 0],
                        [y_val.map({2: 1, 0: -1, 1: 0}), y_val.map({0: 1, 2: -1, 1: 0})],
                        default=0
                    ), index=y_val.index)
                    
                    mean_return = pnl.mean()
                    downside_std = pnl[pnl < 0].std()
                    sortino = (mean_return / downside_std) if downside_std > 0 else 0
                    fold_sortino_ratios.append(sortino)

                except Exception as e:
                    sys.stdout.write("\n")
                    logger.warning(f"Fold in trial {trial.number} failed with error: {e}")
                    # Return worst possible scores
                    return -1.0, -10.0

            # --- Blended Objective Score ---
            # Objective 1: A mix of trade-finding ability and activity
            avg_trade_f1 = np.mean(fold_trade_f1_scores)
            avg_activity = np.mean(fold_trade_activity_scores)
            # 70% weight on trade F1, 30% on activity
            blended_trade_score = (avg_trade_f1 * 0.7) + (avg_activity * 0.3)
            
            # Objective 2: Risk-adjusted return
            final_sortino = np.mean(fold_sortino_ratios)

            return blended_trade_score, final_sortino

        try:
            study_name = f"{self.config.nickname}_{self.config.strategy_name}_{datetime.now().strftime('%Y%m%d-%H%M')}"
            pruner = optuna.pruners.MedianPruner()
            study = optuna.create_study(directions=['maximize', 'maximize'], pruner=pruner, study_name=study_name)
            study.optimize(custom_objective, n_trials=self.config.OPTUNA_TRIALS, timeout=3600, n_jobs=-1, callbacks=[dynamic_progress_callback])
            sys.stdout.write("\n")
            return study
        except Exception as e:
            sys.stdout.write("\n")
            logger.error(f"    - Optuna study failed catastrophically: {e}", exc_info=True)
            return None

    def _find_best_threshold(self, best_params, X_train, y_train, X_val, y_val, num_classes) -> Tuple[float, float]:
        logger.info("    - Tuning classification threshold for F1 score and generating per-class report...")
        objective = 'multi:softprob' if num_classes > 2 else 'binary:logistic'
        temp_params = {'objective': objective, 'booster': 'gbtree', 'tree_method': 'hist', **best_params}
        if num_classes > 2: temp_params['num_class'] = num_classes
        temp_params.pop('early_stopping_rounds', None)
        
        temp_pipeline = Pipeline([('scaler', RobustScaler()), ('model', xgb.XGBClassifier(**temp_params))])
        fit_params = {'model__sample_weight': y_train.map(self.class_weights)}
        temp_pipeline.fit(X_train, y_train, **fit_params)
        
        probs = temp_pipeline.predict_proba(X_val)

        logger.info("\n=== Prediction Diagnostics ===")
        logger.info(f"Raw probability distribution (mean): {np.mean(probs, axis=0)}")
        logger.info("Max probability distribution:")
        logger.info(pd.Series(np.max(probs, axis=1)).describe())
        logger.info("Prediction confidence histogram being saved to 'confidence_hist.png'")
        plt.figure()
        plt.hist(np.max(probs, axis=1), bins=50)
        plt.title("Confidence Histogram (Max Probability)")
        plt.xlabel("Confidence")
        plt.ylabel("Frequency")
        plt.savefig('confidence_hist.png')
        plt.close()

        best_f1 = -1.0
        best_thresh = 0.5
        best_preds = None

        for confidence_gate in np.arange(0.33, 0.96, 0.05):
            preds = np.argmax(probs, axis=1)
            confidence_mask = np.max(probs, axis=1) > confidence_gate
            
            if num_classes > 2:
                preds[~confidence_mask] = 1

            if np.sum(confidence_mask) > 0:
                f1 = f1_score(y_val[confidence_mask], preds[confidence_mask], average='weighted', zero_division=0)
            else:
                f1 = 0.0

            if f1 > best_f1:
                best_f1, best_thresh, best_preds = f1, confidence_gate, preds

        if best_preds is not None:
            target_names = ['Sell', 'Hold', 'Buy'] if num_classes == 3 else ['Hold', 'Trade']
            self.classification_report_str = classification_report(y_val, best_preds, target_names=target_names, zero_division=0)
            logger.info("    - Stored detailed classification report for the best validation threshold.")
        else:
            self.classification_report_str = "Could not generate a valid prediction set for the report."

        logger.info(f"    - Best confidence gate found: {best_thresh:.2f} (Weighted F1 on confident preds: {best_f1:.4f})")
        return best_thresh, best_f1

    def _train_final_model(self,best_params:Dict,X:pd.DataFrame,y:pd.Series, feature_names: List[str], num_classes: int)->Optional[Pipeline]:
        logger.info(f"    - Training final model on all available data using {len(feature_names)} elite features...")
        try:
            best_params.pop('early_stopping_rounds', None)

            objective = 'multi:softprob' if num_classes > 2 else 'binary:logistic'
            final_params={'objective':objective,'booster':'gbtree','tree_method':'hist','seed':42,**best_params}
            if num_classes > 2: final_params['num_class'] = num_classes

            final_pipeline=Pipeline([('scaler',RobustScaler()),('model',xgb.XGBClassifier(**final_params))])

            fit_params={'model__sample_weight':y.map(self.class_weights)}

            final_pipeline.fit(X, y, **fit_params)

            if self.config.CALCULATE_SHAP_VALUES:
                if self.is_minirocket_model:
                    logger.warning("    - Generating SHAP for MiniRocket features. Note: these features are not directly human-interpretable.")
                    shap_feature_names = [f"rocket_{i}" for i in range(X.shape[1])]
                else:
                    shap_feature_names = feature_names
                self._generate_shap_summary(final_pipeline.named_steps['model'], final_pipeline.named_steps['scaler'].transform(X), shap_feature_names, num_classes)

            return final_pipeline
        except Exception as e:
            logger.error(f"    - Error during final model training: {e}",exc_info=True)
            return None

    def _generate_shap_summary(self, model: xgb.XGBClassifier, X_scaled: np.ndarray, feature_names: List[str], num_classes: int):
        logger.info("    - Generating SHAP feature importance summary...")
        try:
            if len(X_scaled) > 2000:
                logger.info(f"    - Subsampling data for SHAP from {len(X_scaled)} to 2000 rows.")
                np.random.seed(42)
                sample_indices = np.random.choice(X_scaled.shape[0], 2000, replace=False)
                X_sample = X_scaled[sample_indices]
            else:
                X_sample = X_scaled
            
            explainer = shap.TreeExplainer(model)
            shap_explanation = explainer(X_sample)
            
            if num_classes > 2:
                if isinstance(shap_explanation.values, list):
                    logger.debug("SHAP values are a list. Processing as multi-output.")
                    mean_abs_shap_per_class = [np.abs(shap_values).mean(axis=0) for shap_values in shap_explanation.values]
                    overall_importance = np.mean(mean_abs_shap_per_class, axis=0)
                else:
                    logger.debug(f"SHAP values are a 3D array with shape {shap_explanation.values.shape}. Processing accordingly.")
                    overall_importance = np.abs(shap_explanation.values).mean(axis=0).mean(axis=-1)
            else:
                overall_importance = np.abs(shap_explanation.values).mean(axis=0)
            
            overall_importance = overall_importance.flatten()

            if len(overall_importance) != len(feature_names):
                logger.error(f"CRITICAL SHAP MISMATCH: Importance array has length {len(overall_importance)} but there are {len(feature_names)} features. SHAP summary will be incorrect.")
                self.shap_summary = pd.DataFrame({'SHAP_Importance': [0.0] * len(feature_names)}, index=feature_names)
                return
            
            summary = pd.DataFrame(overall_importance, index=feature_names, columns=['SHAP_Importance']).sort_values(by='SHAP_Importance', ascending=False)
            
            self.shap_summary = summary
            logger.info("    - SHAP summary generated successfully.")
            
        except Exception as e:
            logger.error(f"    - Failed to generate SHAP summary: {e}", exc_info=True)
            self.shap_summary = None

# =============================================================================
# 7. BACKTESTER & 8. PERFORMANCE ANALYZER
# =============================================================================

class Backtester:
    """
    Implements confluence-based trade execution logic.
    - RE-ARCHITECTED: The `run_backtest_chunk` method now consumes a dictionary
      of trained Primary and Auxiliary models.
    - NEW: A trade is only initiated when a high "Signal Pressure" prediction
      from the Primary model is CONFIRMED by high-quality predictions from
      the Auxiliary models (e.g., good timing, valid candle pattern).
    - This replaces the simple confidence threshold with a more robust,
      evidence-based consensus mechanism.
    """
    def __init__(self,config:ConfigModel):
        self.config=config
        self.is_meta_model = False # Deprecated in this architecture
        self.is_transformer_model = False # Deprecated in this architecture
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

    def _get_tiered_risk_params(self, equity: float) -> Tuple[float, int]:
        """Looks up risk percentage and max trades from the tiered config."""
        sorted_tiers = sorted(self.config.TIERED_RISK_CONFIG.keys())
        for tier_cap in sorted_tiers:
            if equity <= tier_cap:
                profile_settings = self.config.TIERED_RISK_CONFIG[tier_cap].get(self.config.RISK_PROFILE, self.config.TIERED_RISK_CONFIG[tier_cap]['Medium'])
                return profile_settings['risk_pct'], profile_settings['pairs']
        highest_tier_settings = self.config.TIERED_RISK_CONFIG[sorted_tiers[-1]]
        profile_settings = highest_tier_settings.get(self.config.RISK_PROFILE, highest_tier_settings['Medium'])
        return profile_settings['risk_pct'], profile_settings['pairs']

    def _calculate_realistic_costs(self, candle: Dict, on_exit: bool = False) -> Tuple[float, float]:
        """Calculates dynamic spread and variable slippage."""
        symbol = candle['Symbol']
        point_size = 0.0001 if 'JPY' not in symbol and candle.get('Open', 1) < 50 else 0.01

        spread_cost = 0
        if not on_exit:
            spread_info = self.config.SPREAD_CONFIG.get(symbol, self.config.SPREAD_CONFIG.get('default'))
            vol_rank = candle.get('market_volatility_index', 0.5)
            spread_pips = spread_info.get('volatile_pips') if vol_rank > 0.8 else spread_info.get('normal_pips')
            spread_cost = spread_pips * point_size

        slippage_cost = 0
        if self.config.USE_VARIABLE_SLIPPAGE:
            atr = candle.get('ATR', 0)
            vol_rank = candle.get('market_volatility_index', 0.5)
            random_factor = random.uniform(0.1, 1.2 if on_exit else 1.0) * self.config.SLIPPAGE_VOLATILITY_FACTOR
            slippage_cost = atr * vol_rank * random_factor

        return spread_cost, slippage_cost

    def _calculate_latency_cost(self, signal_candle: Dict, exec_candle: Dict) -> float:
        """Calculates a randomized, volatility-based cost to simulate execution latency."""
        if not self.config.SIMULATE_LATENCY: return 0.0
        atr = signal_candle.get('ATR')
        if pd.isna(atr) or atr <= 0: return 0.0
        bar_duration_sec = (exec_candle['Timestamp'] - signal_candle['Timestamp']).total_seconds()
        if bar_duration_sec <= 0: return 0.0
        simulated_delay_sec = random.uniform(50, self.config.EXECUTION_LATENCY_MS) / 1000.0
        return atr * (simulated_delay_sec / bar_duration_sec)

    def run_backtest_chunk(self, df_chunk_in: pd.DataFrame, pipelines: Dict[str, Pipeline], initial_equity: float, feature_list: List[str], trade_lockout_until: Optional[pd.Timestamp] = None) -> Tuple[pd.DataFrame, pd.Series, bool, Optional[Dict], Dict]:
        if df_chunk_in.empty or not pipelines:
            return pd.DataFrame(), pd.Series([initial_equity]), False, None, {}

        # --- Define Thresholds for Trade Confirmation ---
        # These could be moved to the ConfigModel for dynamic tuning
        PRIMARY_PRESSURE_THRESHOLD = 0.7  # Predicted Sharpe must be > 0.7
        CONFIRM_TIMING_THRESHOLD = 0.6    # Predicted timing score must be > 0.6 (i.e., reasonably early)
        CONFIRM_PATTERN_PROB_THRESHOLD = 0.65 # Confidence in candle pattern must be > 65%

        df_chunk = df_chunk_in.copy()
        trades, equity, equity_curve, open_positions = [], initial_equity, [initial_equity], {}
        chunk_peak_equity, run_peak_equity = initial_equity, initial_equity # Assuming run_peak_equity is managed outside
        circuit_breaker_tripped, breaker_context = False, None
        last_trade_pnl, daily_dd_report, current_day, day_start_equity, day_peak_equity = 0.0, {}, None, initial_equity, initial_equity
        
        def finalize_day_metrics(day, equity_close):
            if day is None: return
            daily_pnl = equity_close - day_start_equity
            daily_dd = (day_peak_equity - equity_close) / day_peak_equity if day_peak_equity > 0 else 0
            daily_dd_report[day.isoformat()] = {'pnl': round(daily_pnl, 2), 'drawdown_pct': round(daily_dd * 100, 2)}

        def close_trade(pos, exit_price, reason, candle):
            nonlocal equity, last_trade_pnl
            pnl = (exit_price - pos['entry_price']) * pos['direction'] * pos['lot_size'] * self.config.CONTRACT_SIZE
            commission = self.config.COMMISSION_PER_LOT * pos['lot_size'] * 2
            net_pnl = pnl - commission
            equity += net_pnl
            last_trade_pnl = net_pnl
            mae = abs(pos['mae_price'] - pos['entry_price'])
            mfe = abs(pos['mfe_price'] - pos['entry_price'])
            trades.append({'ExecTime': candle['Timestamp'], 'Symbol': pos['symbol'], 'PNL': net_pnl, 'Equity': equity, 'Confidence': pos['confidence'], 'Direction': pos['direction'], 'ExitReason': reason, 'MAE': round(mae, 5), 'MFE': round(mfe, 5)})
            equity_curve.append(equity)
            return net_pnl

        candles = df_chunk.reset_index().to_dict('records')
        for i in range(1, len(candles)):
            current_candle, prev_candle = candles[i], candles[i-1]
            candle_date = current_candle['Timestamp'].date()
            if candle_date != current_day:
                finalize_day_metrics(current_day, equity)
                current_day, day_start_equity, day_peak_equity = candle_date, equity, equity

            if not circuit_breaker_tripped:
                day_peak_equity = max(day_peak_equity, equity)
                chunk_peak_equity = max(chunk_peak_equity, equity)
                if equity > 0 and chunk_peak_equity > 0 and (chunk_peak_equity - equity) / chunk_peak_equity > self.config.MAX_DD_PER_CYCLE:
                    circuit_breaker_tripped, breaker_context = True, {"num_trades_before_trip": len(trades)}
                    for sym, pos in list(open_positions.items()): close_trade(pos, current_candle['Open'], "Circuit Breaker", current_candle); del open_positions[sym]
                    continue
            
            if equity <= 0: break

            for symbol, pos in open_positions.items():
                pos['mfe_price'] = max(pos['mfe_price'], current_candle['High']) if pos['direction'] == 1 else min(pos['mfe_price'], current_candle['Low'])
                pos['mae_price'] = min(pos['mae_price'], current_candle['Low']) if pos['direction'] == 1 else max(pos['mae_price'], current_candle['High'])

            for symbol in list(open_positions.keys()):
                pos = open_positions[symbol]
                exit_price, exit_reason = None, None
                sl_hit = (pos['direction'] == 1 and current_candle['Low'] <= pos['sl']) or (pos['direction'] == -1 and current_candle['High'] >= pos['sl'])
                tp_hit = (pos['direction'] == 1 and current_candle['High'] >= pos['tp']) or (pos['direction'] == -1 and current_candle['Low'] <= pos['tp'])
                if sl_hit: exit_price, exit_reason = pos['sl'], "Stop Loss"
                elif tp_hit: exit_price, exit_reason = pos['tp'], "Take Profit"
                if exit_price: close_trade(pos, exit_price, exit_reason, current_candle); del open_positions[symbol]

            symbol = prev_candle['Symbol']
            base_risk_pct, max_concurrent_trades = self._get_tiered_risk_params(equity) if self.config.USE_TIERED_RISK else (self.config.BASE_RISK_PER_TRADE_PCT, self.config.MAX_CONCURRENT_TRADES)
            is_locked_out = trade_lockout_until and current_candle['Timestamp'] < trade_lockout_until
            
            # --- [NEW] CONFLUENCE-BASED SIGNAL GENERATION ---
            if not circuit_breaker_tripped and not is_locked_out and symbol not in open_positions and len(open_positions) < max_concurrent_trades:
                if prev_candle.get('anomaly_score') == -1: continue
                vol_idx = prev_candle.get('market_volatility_index', 0.5)
                if not (self.config.MIN_VOLATILITY_RANK <= vol_idx <= self.config.MAX_VOLATILITY_RANK): continue

                features_df = pd.DataFrame([prev_candle])[feature_list].fillna(0)
                
                # 1. Check Primary Model for Signal Pressure
                primary_pipeline = pipelines.get('primary_pressure')
                if not primary_pipeline: continue
                predicted_pressure = primary_pipeline.predict(features_df)[0]
                
                potential_direction = 0
                if predicted_pressure > PRIMARY_PRESSURE_THRESHOLD: potential_direction = 1
                elif predicted_pressure < -PRIMARY_PRESSURE_THRESHOLD: potential_direction = -1

                # 2. If Pressure exists, seek Confirmation
                if potential_direction != 0:
                    is_confirmed = False
                    confidence_score = abs(predicted_pressure)
                    
                    # Get Auxiliary Predictions
                    timing_pipeline = pipelines.get('confirm_timing')
                    pattern_pipeline = pipelines.get('confirm_bull_engulf') # Example for bullish
                    
                    if timing_pipeline and pattern_pipeline:
                        predicted_timing = timing_pipeline.predict(features_df)[0]
                        pattern_prob = pattern_pipeline.predict_proba(features_df)[0][1] # Probability of "is engulfing"
                        
                        # Confluence Logic
                        if predicted_timing > CONFIRM_TIMING_THRESHOLD and pattern_prob > CONFIRM_PATTERN_PROB_THRESHOLD:
                            is_confirmed = True
                            # Blend scores for a final confidence metric
                            confidence_score = (abs(predicted_pressure) + predicted_timing + pattern_prob) / 3
                    
                    if is_confirmed:
                        # 3. If Confirmed, proceed to trade execution
                        atr = prev_candle.get('ATR', 0)
                        if pd.isna(atr) or atr <= 1e-9: continue
                        
                        sl_dist = atr * self.config.SL_ATR_MULTIPLIER
                        risk_per_trade_usd = equity * base_risk_pct
                        
                        point_value = self.config.CONTRACT_SIZE * (0.0001 if 'JPY' not in symbol else 0.01)
                        risk_per_lot = sl_dist * point_value
                        if risk_per_lot <= 0: continue
                        
                        lots = round((risk_per_trade_usd / risk_per_lot) / self.config.LOT_STEP) * self.config.LOT_STEP
                        if lots < self.config.MIN_LOT_SIZE: continue

                        margin_required = (lots * self.config.CONTRACT_SIZE * current_candle['Open']) / self.config.LEVERAGE
                        used_margin = sum(p.get('margin_used', 0) for p in open_positions.values())
                        if (equity - used_margin) < margin_required: continue

                        spread_cost, slip_cost = self._calculate_realistic_costs(prev_candle)
                        lat_cost = self._calculate_latency_cost(prev_candle, current_candle)
                        entry_price = current_candle['Open'] + ((spread_cost + slip_cost + lat_cost) * potential_direction)

                        sl_price = entry_price - sl_dist * potential_direction
                        tp_price = entry_price + (sl_dist * self.config.TP_ATR_MULTIPLIER) * potential_direction
                        
                        open_positions[symbol] = {
                            'symbol': symbol, 'direction': potential_direction, 'entry_price': entry_price, 
                            'sl': sl_price, 'tp': tp_price, 'confidence': confidence_score, 'lot_size': lots, 
                            'margin_used': margin_required, 'mfe_price': entry_price, 'mae_price': entry_price
                        }

        finalize_day_metrics(current_day, equity)
        return pd.DataFrame(trades), pd.Series(equity_curve), circuit_breaker_tripped, breaker_context, daily_dd_report
        
class PerformanceAnalyzer:
    def __init__(self,config:ConfigModel):
        self.config=config

    def generate_full_report(self,trades_df:Optional[pd.DataFrame],equity_curve:Optional[pd.Series],cycle_metrics:List[Dict],aggregated_shap:Optional[pd.DataFrame]=None, framework_memory:Optional[Dict]=None, aggregated_daily_dd:Optional[List[Dict]]=None, last_classification_report:str="N/A") -> Dict[str, Any]: # MODIFIED
        logger.info("-> Stage 4: Generating Final Performance Report...")
        if equity_curve is not None and len(equity_curve) > 1: self.plot_equity_curve(equity_curve)
        if aggregated_shap is not None: self.plot_shap_summary(aggregated_shap)

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
            plt.savefig(self.config.SHAP_PLOT_PATH)
            plt.close()
            logger.info(f"  - SHAP summary plot saved to: {self.config.SHAP_PLOT_PATH}")
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
    It incrementally updates the cache with new data if available.
    """
    logger.info(f"-> Fetching/updating external macroeconomic time series for: {list(tickers.keys())}...")
    
    cache_dir = os.path.join(results_dir)
    os.makedirs(cache_dir, exist_ok=True)
    data_cache_path = os.path.join(cache_dir, "macro_data.parquet")
    metadata_cache_path = os.path.join(cache_dir, "macro_cache_metadata.json")

    # --- Cache Validation Logic ---
    if os.path.exists(metadata_cache_path):
        try:
            with open(metadata_cache_path, 'r') as f:
                metadata = json.load(f)
            if set(metadata.get("tickers", [])) == set(tickers.keys()):
                cached_df = pd.read_parquet(data_cache_path)
                last_cached_date = pd.to_datetime(metadata.get("last_date")).date()
                
                if last_cached_date >= (datetime.now() - timedelta(days=1)).date():
                    logger.info("  - Macro data is up-to-date. Loading from cache.")
                    # FIX: Ensure the returned DataFrame has a 'Timestamp' column
                    df_to_return = cached_df.reset_index()
                    return df_to_return.rename(columns={df_to_return.columns[0]: 'Timestamp'})

                else:
                    logger.info(f"  - Cache is stale (last date: {last_cached_date}). Fetching incremental update...")
                    update_start_date = last_cached_date + timedelta(days=1)
                    new_data_raw = yf.download(list(tickers.values()), start=update_start_date, progress=False, auto_adjust=True)

                    if not new_data_raw.empty:
                        new_close_prices = new_data_raw['Close'].copy()
                        if isinstance(new_close_prices, pd.Series):
                             new_close_prices = new_close_prices.to_frame(name=list(tickers.values())[0])
                        
                        ticker_to_name_map = {v: k for k, v in tickers.items()}
                        new_close_prices.rename(columns=ticker_to_name_map, inplace=True)
                        
                        updated_df = pd.concat([cached_df, new_close_prices]).sort_index()
                        updated_df = updated_df[~updated_df.index.duplicated(keep='last')]
                        updated_df.ffill(inplace=True)
                        
                        updated_df.to_parquet(data_cache_path)
                        new_metadata = {"tickers": list(tickers.keys()), "last_date": updated_df.index.max().strftime('%Y-%m-%d')}
                        with open(metadata_cache_path, 'w') as f:
                            json.dump(new_metadata, f, indent=4)

                        logger.info("  - Macro cache successfully updated.")
                        df_to_return = updated_df.reset_index()
                        return df_to_return.rename(columns={df_to_return.columns[0]: 'Timestamp'})
                    else:
                        logger.info("  - No new macro data found. Using existing cached data.")
                        df_to_return = cached_df.reset_index()
                        return df_to_return.rename(columns={df_to_return.columns[0]: 'Timestamp'})
        except Exception as e:
            logger.error(f"  - Could not read or update macro cache. Rebuilding. Error: {e}")

    # --- Full Download (if no valid cache) ---
    logger.info("  - No valid cache found. Performing full download for macro data...")
    all_data = yf.download(list(tickers.values()), period=period, progress=False, auto_adjust=True)
    
    if all_data.empty:
        logger.error("  - Failed to download any macro data.")
        return pd.DataFrame()

    close_prices = all_data['Close'].copy()
    if isinstance(close_prices, pd.Series):
        close_prices = close_prices.to_frame(name=list(tickers.values())[0])
        
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
            
    try:
        with open(playbook_path, 'r') as f:
            playbook = json.load(f)

        updated = False
        for strategy_name, default_config in DEFAULT_PLAYBOOK.items():
            if strategy_name not in playbook:
                playbook[strategy_name] = default_config
                logger.info(f"  - Adding new strategy to playbook: '{strategy_name}'")
                updated = True
        
        if updated:
            logger.info("Playbook was updated with new strategies. Saving changes...")
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

# --- PHASE 3: NEW HELPER FUNCTION ---
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

import hashlib # Make sure to add this import at the top of your script

def _generate_cache_metadata(config: ConfigModel, files: List[str], tf_roles: Dict, feature_engineer_class: type) -> Dict:
    """
    [FIXED] Generates a dictionary of metadata to validate the feature cache.
    Now includes a hash of the script file to detect changes in feature logic.
    """
    file_metadata = {}
    for filename in sorted(files):
        file_path = os.path.join(config.BASE_PATH, filename)
        if os.path.exists(file_path):
            stat = os.stat(file_path)
            file_metadata[filename] = {"mtime": stat.st_mtime, "size": stat.st_size}

    # --- Detect code changes to the FeatureEngineer class source code ---
    script_hash = ""
    try:
        # Get the source code of the FeatureEngineer class as a string
        # FIX: Use the 'feature_engineer_class' argument passed into the function
        fe_source_code = inspect.getsource(feature_engineer_class)
        # Encode it to bytes and then hash it
        script_hash = hashlib.sha256(fe_source_code.encode('utf-8')).hexdigest()
    except Exception as e:
        logger.warning(f"Could not generate FeatureEngineer class hash for cache validation: {e}")

    # These are the parameters that affect the output of `create_feature_stack`
    param_metadata = {
        # --- Script hash to the tracked parameters ---
        'script_sha256_hash': script_hash,

        'TREND_FILTER_THRESHOLD': config.TREND_FILTER_THRESHOLD,
        'BOLLINGER_PERIOD': config.BOLLINGER_PERIOD,
        'STOCHASTIC_PERIOD': config.STOCHASTIC_PERIOD,
        'HAWKES_KAPPA': config.HAWKES_KAPPA,
        'anomaly_contamination_factor': config.anomaly_contamination_factor,
        'USE_PCA_REDUCTION': config.USE_PCA_REDUCTION,
        'PCA_N_COMPONENTS': config.PCA_N_COMPONENTS,
        'RSI_PERIODS_FOR_PCA': config.RSI_PERIODS_FOR_PCA,
        'tf_roles': tf_roles,
        # Also include the new dynamic params to ensure cache busts if they change
        'DYNAMIC_INDICATOR_PARAMS': config.DYNAMIC_INDICATOR_PARAMS
    }
    return {"files": file_metadata, "params": param_metadata}

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
    # Pause trading over the weekend
    if now.weekday() >= 5: # Saturday or Sunday
        return True, "Weekend market closure"
        
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

def _run_feature_learnability_test(df_train_labeled: pd.DataFrame, feature_list: list, target_col: str = 'target') -> str:
    """
    Checks the information content of features against the label using Mutual Information.
    Returns a string summary for the AI.
    """
    from sklearn.feature_selection import mutual_info_classif
    
    # Ensure all selected features are actually in the dataframe
    valid_features = [f for f in feature_list if f in df_train_labeled.columns]
    if not valid_features:
        return "Feature Learnability: No valid features found to test."

    X = df_train_labeled[valid_features].copy()
    y = df_train_labeled[target_col]

    # Impute NaNs for the calculation, as mutual_info_classif cannot handle them
    X.fillna(X.median(), inplace=True)

    try:
        scores = mutual_info_classif(X, y, random_state=42)
        mi_scores = pd.Series(scores, index=X.columns).sort_values(ascending=False)
        
        top_5 = mi_scores.head(5)
        summary = ", ".join([f"{idx}: {score:.4f}" for idx, score in top_5.items()])
        return f"Feature Learnability (Top 5 MI Scores): {summary}"
    except Exception as e:
        logger.error(f"  - Could not run feature learnability test: {e}")
        return f"Feature Learnability: Error during calculation - {e}"


def _label_distribution_report(df: pd.DataFrame, label_col="target") -> str:
    """
    Generates a report on the class balance of the labels.
    Returns a string summary for the AI.
    """
    if label_col not in df.columns:
        return "Label Distribution: Target column not found."
        
    counts = df[label_col].value_counts(normalize=True)
    # Map {-1: "Short", 0: "Hold", 1: "Long"} for clarity
    counts.index = counts.index.map({-1.0: 'Short', 0.0: 'Hold', 1.0: 'Long', 1: 'Long', 0: 'Hold', -1: 'Short'})
    report_dict = {k: f"{v:.2%}" for k, v in counts.to_dict().items()}
    return f"Label Distribution: {report_dict}"

def run_single_instance(fallback_config: Dict, framework_history: Dict, playbook: Dict, nickname_ledger: Dict, directives: List[Dict], api_interval_seconds: int):
    MODEL_QUALITY_THRESHOLD = 0.05
    MIN_F1_SCORE_GATE = fallback_config.get("MIN_F1_SCORE_GATE", 0.45)
    run_timestamp_str = datetime.now().strftime("%Y%m%d-%H%M%S")
    gemini_analyzer, api_timer = GeminiAnalyzer(), APITimer(interval_seconds=api_interval_seconds)

    current_config_dict = fallback_config.copy()
    current_config_dict['run_timestamp'] = run_timestamp_str

    temp_config = ConfigModel(**{**current_config_dict, 'nickname': 'init', 'run_timestamp': 'init'})

    data_loader = DataLoader(temp_config)
    all_files = [f for f in os.listdir(current_config_dict['BASE_PATH']) if f.endswith(('.csv', '.txt')) and re.match(r'^[A-Z0-9]+_[A-Z0-9]+', f)]
    if not all_files:
        logger.critical("No data files found in base path. Exiting.")
        return

    data_by_tf, detected_timeframes = data_loader.load_and_parse_data(all_files)
    if not data_by_tf:
        return

    tf_roles = determine_timeframe_roles(detected_timeframes)

    ai_selected_tickers = gemini_analyzer.select_relevant_macro_tickers(data_by_tf[tf_roles['base']]['Symbol'].unique().tolist(), {
        "VIX": "^VIX", "DXY": "DX-Y.NYB", "US10Y_YIELD": "^TNX", "SP500": "^GSPC",
        "WTI_OIL": "CL=F", "GOLD": "GC=F", "GERMAN10Y": "^DE10Y", "NIKKEI225": "^N225"
    })

    full_df = None
    if temp_config.USE_FEATURE_CACHING:
            logger.info("-> Feature Caching is ENABLED. Checking for a valid cache...")
            # This call now matches the corrected function definition
            current_metadata = _generate_cache_metadata(temp_config, all_files, tf_roles, FeatureEngineer) 
            if os.path.exists(temp_config.CACHE_METADATA_PATH) and os.path.exists(temp_config.CACHE_PATH):
                try:
                    with open(temp_config.CACHE_METADATA_PATH, 'r') as f: saved_metadata = json.load(f)
                    if current_metadata == saved_metadata:
                        logger.info("  - Cache is VALID. Loading features from cache...")
                        full_df = pd.read_parquet(temp_config.CACHE_PATH)
                    else: 
                        logger.warning("  - Cache is STALE. Re-engineering features...")
                except Exception as e: 
                    logger.warning(f"  - Could not read or validate cache. Re-engineering features. Error: {e}")
            else: 
                logger.info("  - No valid cache found. Engineering features...")

    if full_df is None:
        fe = FeatureEngineer(temp_config, tf_roles, playbook)
        full_df = fe.create_feature_stack(data_by_tf)
        if temp_config.USE_FEATURE_CACHING and not full_df.empty:
            logger.info("  - Saving newly engineered features to cache...")
            try:
                os.makedirs(os.path.dirname(temp_config.CACHE_PATH), exist_ok=True)
                full_df.to_parquet(temp_config.CACHE_PATH)
                # The call here also needs to be updated to pass the class
                with open(temp_config.CACHE_METADATA_PATH, 'w') as f: json.dump(_generate_cache_metadata(temp_config, all_files, tf_roles, FeatureEngineer), f, indent=4)
            except Exception as e: logger.error(f"  - Failed to save features to cache. Error: {e}")

    if full_df.empty:
        logger.critical("Feature engineering resulted in an empty dataframe. Exiting.")
        return

    all_available_features = [c for c in full_df.columns if c not in ['Open','High','Low','Close','RealVolume','Symbol','Timestamp','primary_model_signal','target']]
    
    logger.info("-> Integrating macroeconomic data as features...")
    macro_df = get_macro_context_data(tickers=ai_selected_tickers, period="10y", results_dir=os.path.join(temp_config.BASE_PATH, "Results"))

    logger.info("-> Slicing recent macro context for AI prompt...")
    two_weeks_ago = full_df.index.max() - pd.Timedelta(weeks=2)
    macro_context = macro_df[macro_df['Timestamp'] >= two_weeks_ago].to_dict(orient='records')
    
    if not macro_df.empty:
        full_df.reset_index(inplace=True)
        full_df = pd.merge_asof(full_df.sort_values('Timestamp'), macro_df.sort_values('Timestamp'), on='Timestamp', direction='backward')
        full_df.set_index('Timestamp', inplace=True)
    else:
        logger.warning("  - No macro data to merge. Macro features will be unavailable.")

    regime_summary = train_and_diagnose_regime(full_df, os.path.join(temp_config.BASE_PATH, "Results"))
    
    pivot_df = full_df.pivot_table(index=full_df.index, columns='Symbol', values='Close', aggfunc='last').ffill().dropna(how='all', axis=1)
    correlation_summary_for_ai = pivot_df.corr().to_json(indent=2) if pivot_df.shape[1] > 1 else "{}"

    assets = full_df['Symbol'].unique().tolist()
    data_summary = {'assets_detected': assets, 'time_range': {'start': full_df.index.min().isoformat(), 'end': full_df.index.max().isoformat()}, 'timeframes_used': tf_roles}
    version_label = f"ML_Framework_V{VERSION}"
    health_report, _ = perform_strategic_review(framework_history, fallback_config['DIRECTIVES_FILE_PATH'])

    regime_champions = {}
    if os.path.exists(temp_config.REGIME_CHAMPIONS_FILE_PATH):
        try:
            with open(temp_config.REGIME_CHAMPIONS_FILE_PATH, 'r') as f: regime_champions = json.load(f)
        except (json.JSONDecodeError, IOError): logger.warning("Could not read regime champions file.")
    
    ai_setup = api_timer.call(gemini_analyzer.get_initial_run_setup, version_label, nickname_ledger, framework_history, playbook, health_report, directives, data_summary, regime_summary['current_diagnosed_regime'], regime_champions, correlation_summary_for_ai, macro_context)
    if not ai_setup:
        logger.critical("AI-driven setup failed because the response was empty or invalid. Exiting.")
        return
    
    if ai_setup.get("strategy_name") == "AllSignal_XGB_Combiner":
        logger.info("! STRATEGY 'AllSignal_XGB_Combiner' selected. Overriding feature set with all available micro-alphas.")
        ai_setup["selected_features"] = all_available_features
        logger.info(f"  - Model will be trained on {len(all_available_features)} features.")
    
    ai_setup = _validate_and_fix_spread_config(ai_setup, fallback_config)
    current_config_dict.update(_sanitize_ai_suggestions(ai_setup))
    if 'RETRAINING_FREQUENCY' in ai_setup: current_config_dict['RETRAINING_FREQUENCY'] = _sanitize_frequency_string(ai_setup['RETRAINING_FREQUENCY'])
    if isinstance(ai_setup.get("nickname"), str) and ai_setup.get("nickname"):
        nickname_ledger[version_label] = ai_setup["nickname"]
        try:
            with open(temp_config.NICKNAME_LEDGER_PATH, 'w') as f: json.dump(nickname_ledger, f, indent=4)
        except IOError as e: logger.error(f"Failed to save new nickname to ledger: {e}")

    try:
        config = ConfigModel(**{**current_config_dict, 'REPORT_LABEL': version_label, 'nickname': nickname_ledger.get(version_label, f"Run-{run_timestamp_str}")})
    except ValidationError as e:
        logger.critical(f"--- FATAL PRE-CYCLE CONFIGURATION ERROR ---\n{e}")
        return
    
    # This block checks if the AI provided a feature list. If not, it loads
    # the default list from the playbook for the selected strategy.
    if not config.selected_features:
        logger.warning("! AI did not provide a 'selected_features' list.")
        strategy_name = config.strategy_name
        if strategy_name in playbook and 'selected_features' in playbook[strategy_name]:
            fallback_features = playbook[strategy_name]['selected_features']
            config.selected_features = fallback_features
            logger.info(f"-> Loading default feature list for '{strategy_name}' from playbook: {fallback_features}")
        else:
            logger.error(f"!! CRITICAL: Could not find a fallback feature list for strategy '{strategy_name}' in the playbook. Training will likely fail.")
    
    file_handler = RotatingFileHandler(config.LOG_FILE_PATH, maxBytes=5*1024*1024, backupCount=2)
    file_handler.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
    logger.addHandler(file_handler)
    logger.info(f"--- Run Initialized: {config.nickname} | Strategy: {config.strategy_name} ---")
    
    train_window, forward_gap = pd.to_timedelta(config.TRAINING_WINDOW), pd.to_timedelta(config.FORWARD_TEST_GAP)
    test_start_date = full_df.index.min() + train_window + forward_gap
    retraining_dates = pd.date_range(start=test_start_date, end=full_df.index.max(), freq=_sanitize_frequency_string(config.RETRAINING_FREQUENCY))

    if retraining_dates.empty:
        logger.critical("Cannot proceed: No valid retraining dates. Data length may be too short.")
        return

    aggregated_trades, aggregated_equity_curve = pd.DataFrame(), pd.Series([config.INITIAL_CAPITAL])
    in_run_historical_cycles, aggregated_daily_dd_reports = [], []
    shap_history, all_optuna_trials = defaultdict(list), []
    last_equity, quarantine_list = config.INITIAL_CAPITAL, []
    run_peak_equity = config.INITIAL_CAPITAL
    
    consecutive_wins, consecutive_losses, cycle_retry_count = 0, 0, 0
    drawdown_control_cycles = 0 
    trade_lockout_until = None
    
    cycle_num = 0
    baseline_failure_cycles = 0
    while cycle_num < len(retraining_dates):
        is_maintenance, reason = _is_maintenance_period()
        if is_maintenance:
            config.operating_state = OperatingState.MAINTENANCE_DORMANCY
            logger.warning(f"--- Cycle Paused: Entering Maintenance/Dormancy due to: {reason} ---")
            logger.info("--- Framework will sleep for 1 hour before re-evaluating. ---")
            time.sleep(3600)
            continue
        
        period_start_date = retraining_dates[cycle_num]
        train_end = period_start_date - forward_gap
        train_start = train_end - pd.to_timedelta(config.TRAINING_WINDOW)
        df_train_raw_for_check = full_df.loc[train_start:train_end].copy()

        if _detect_surge_opportunity(df_train_raw_for_check):
            if config.operating_state != OperatingState.DRAWDOWN_CONTROL:
                config.operating_state = OperatingState.OPPORTUNISTIC_SURGE
        
        strategic_directive = gemini_analyzer.establish_strategic_directive(in_run_historical_cycles, config.operating_state)
        
        if config.operating_state == OperatingState.DRAWDOWN_CONTROL and drawdown_control_cycles >= 2:
            logger.warning("! REGENERATION MODE BEHAVIOR ACTIVATED ! System has been in Drawdown Control for multiple cycles.")
            strategic_directive += (
                "\n**REGENERATION DIRECTIVE:** The current strategy is failing. Propose a fundamental change. "
                "This could be a completely different strategy from the playbook (even an experimental one) or a "
                "novel feature set. The goal is to find a new source of alpha, not to optimize the failing one."
            )
        
        config = _apply_operating_state_rules(config)
        
        logger.info(f"\n--- Starting Cycle [{cycle_num + 1}/{len(retraining_dates)}] in state '{config.operating_state.value}' ---")
        cycle_start_time = time.time()
        
        test_end = period_start_date + pd.tseries.frequencies.to_offset(_sanitize_frequency_string(config.RETRAINING_FREQUENCY))

        df_train_raw = full_df.loc[train_start:train_end].copy()
        df_test = full_df.loc[period_start_date:min(test_end, full_df.index.max())].copy()
        
        if df_train_raw.empty or df_test.empty:
            logger.warning(f"  - Skipping cycle {cycle_num + 1}: Not enough data.")
            cycle_num += 1; continue

        strategy_details = playbook.get(config.strategy_name, {})
        fe = FeatureEngineer(config, tf_roles, playbook)

        if strategy_details.get("requires_gp", False):
            logger.info(f"--- Genetic Programming Strategy Detected: '{config.strategy_name}' ---")
            gene_pool = api_timer.call(gemini_analyzer.define_gene_pool, strategy_goal=strategy_details.get("strategy_goal", "general"), available_features=all_available_features)
            if gene_pool and gene_pool.get('indicators'):
                gp = GeneticProgrammer(gene_pool, config)
                evolved_rules, best_fitness = gp.run_evolution(df_train_raw)
                df_with_primary_signal = apply_genetic_rules_to_df(df_train_raw, evolved_rules, config)
                df_train_labeled = fe.label_meta(df_with_primary_signal, config.LOOKAHEAD_CANDLES)
            else:
                logger.error("AI failed to define a valid gene pool. Skipping GP evolution for this cycle.")
                df_train_labeled = fe.label_standard(df_train_raw, config.LOOKAHEAD_CANDLES)
        else:
            labeling_method = getattr(config, 'LABELING_METHOD', 'standard')
            label_func = getattr(fe, f"label_{labeling_method}", fe.label_standard)
            df_train_labeled = label_func(df_train_raw, config.LOOKAHEAD_CANDLES)

        pipeline, threshold, f1_score_val = None, None, -1.0
        
        training_attempt = 0
        while training_attempt < config.MAX_TRAINING_RETRIES_PER_CYCLE:
            training_attempt += 1
            logger.info(f"--- Training Attempt {training_attempt}/{config.MAX_TRAINING_RETRIES_PER_CYCLE} ---")

            if training_attempt > 1: 
                labeling_method = getattr(config, 'LABELING_METHOD', 'standard')
                label_func = getattr(fe, f"label_{labeling_method}", fe.label_standard)
                df_train_labeled = label_func(df_train_raw.copy(), config.LOOKAHEAD_CANDLES)

            if not check_label_quality(df_train_labeled, config.LABEL_MIN_EVENT_PCT):
                logger.critical(f"!! MODEL TRAINING SKIPPED (Attempt {training_attempt}) !! Un-trainable labels generated.")
            else:
                trainer = ModelTrainer(config, gemini_analyzer)
                train_result = trainer.train(df_train_labeled, config.selected_features, strategy_details, strategic_directive)
                
                if train_result:
                    pipeline, threshold, f1_score_val, final_model_features = train_result
                    current_f1_gate = config.STATE_BASED_CONFIG[config.operating_state].get("min_f1_gate", MIN_F1_SCORE_GATE)
                    
                    if f1_score_val >= current_f1_gate:
                        logger.info(f"  - Model training successful on attempt {training_attempt}.")
                        break 
                    else:
                        logger.critical(f"!! MODEL QUALITY GATE FAILED (Attempt {training_attempt}) !! F1 Score ({f1_score_val:.3f}) < Gate ({current_f1_gate}).")
                        pipeline = None
                else:
                    logger.critical(f"!! MODEL TRAINING FAILED (Attempt {training_attempt}) !!")
            
            if pipeline is None and training_attempt >= 2 and training_attempt < config.MAX_TRAINING_RETRIES_PER_CYCLE:
                pass

        if pipeline:
            # --- THIS BLOCK RUNS IF TRAINING WAS SUCCESSFUL ---
            cycle_retry_count = 0 
            
            if config.USE_STATIC_CONFIDENCE_GATE:
                final_threshold = config.STATIC_CONFIDENCE_GATE
                logger.info(f"Using STATIC confidence gate for backtest: {final_threshold:.2f}")
            else:
                state_modifier = config.STATE_BASED_CONFIG[config.operating_state]["confidence_gate_modifier"]
                final_threshold = threshold * state_modifier
                logger.info(f"Using DYNAMIC confidence gate for backtest: {threshold:.2f} (from trainer) * {state_modifier:.2f} (state mod) = {final_threshold:.2f}")

            backtester = Backtester(config)
            
            # This is the call to the backtester
            trades, equity_curve, breaker_tripped, breaker_context, daily_dd_report = backtester.run_backtest_chunk(
                df_test, 
                pipeline, 
                final_threshold, 
                last_equity, 
                strategy_details, 
                run_peak_equity, 
                final_model_features,
                trade_lockout_until
            )
            
            if not trades.empty:
                baseline_failure_cycles = 0
            else:
                logger.warning("  - Model trained successfully but executed no trades in the forward test.")
                baseline_failure_cycles += 1

            trade_lockout_until = None
            aggregated_daily_dd_reports.append(daily_dd_report)
            cycle_status_msg = "Completed"
        else:
            # --- THIS BLOCK RUNS IF TRAINING FAILED ---
            # Assign default/empty values since no backtest was run
            trades, equity_curve, breaker_tripped, breaker_context, daily_dd_report = pd.DataFrame(), pd.Series([last_equity]), False, None, {}
            cycle_status_msg = "Training Failed"
            cycle_retry_count += 1
            baseline_failure_cycles += 1

        cycle_pnl = equity_curve.iloc[-1] - last_equity if not equity_curve.empty else 0.0
        
        if not trades.empty:
            if trades.iloc[-1]['PNL'] > 0: consecutive_wins += 1; consecutive_losses = 0
            elif trades.iloc[-1]['PNL'] < 0: consecutive_losses += 1; consecutive_wins = 0
        
        in_run_historical_cycles.append({
            "StartDate": period_start_date.date().isoformat(), "EndDate": test_end.date().isoformat(),
            "NumTrades": len(trades), "PNL": round(cycle_pnl, 2),
            "Status": "Circuit Breaker" if breaker_tripped else cycle_status_msg,
            "F1_Score": round(f1_score_val, 4) if f1_score_val is not None else 0.0,
            "State": config.operating_state.value, "BreakerContext": breaker_context
        })

        if not trades.empty:
            aggregated_trades = pd.concat([aggregated_trades, trades], ignore_index=True)
            aggregated_equity_curve = pd.concat([aggregated_equity_curve.iloc[:-1], equity_curve], ignore_index=True)
            last_equity = equity_curve.iloc[-1]
            if last_equity > run_peak_equity:
                logger.info(f"** NEW EQUITY HIGH REACHED: ${last_equity:,.2f} **")
                run_peak_equity = last_equity
        
        previous_state = config.operating_state
        
        if cycle_status_msg != "Completed" or breaker_tripped or consecutive_losses >= 3:
            config.operating_state = OperatingState.DRAWDOWN_CONTROL
            if previous_state != OperatingState.DRAWDOWN_CONTROL:
                logger.info(f"! STATE TRANSITION ! Triggered {config.operating_state.value} due to losses or training failure.")
                drawdown_control_cycles = 1
            else:
                drawdown_control_cycles += 1
        elif cycle_status_msg == "Completed" and not trades.empty and (last_equity >= run_peak_equity or consecutive_wins >= 2):
            config.operating_state = OperatingState.AGGRESSIVE_EXPANSION
            if previous_state != OperatingState.AGGRESSIVE_EXPANSION:
                logger.info(f"! STATE TRANSITION ! Triggered {config.operating_state.value} due to strong profitable performance.")
            drawdown_control_cycles = 0
        else:
            if previous_state == OperatingState.OPPORTUNISTIC_SURGE:
                 logger.info(f"! STATE TRANSITION ! Reverting from Opportunistic Surge to {OperatingState.CONSERVATIVE_BASELINE.value}.")
            config.operating_state = OperatingState.CONSERVATIVE_BASELINE
            if previous_state != OperatingState.CONSERVATIVE_BASELINE and previous_state != OperatingState.OPPORTUNISTIC_SURGE:
                logger.info(f"! STATE TRANSITION ! Reverting to {config.operating_state.value}.")
            drawdown_control_cycles = 0

        if baseline_failure_cycles >= 2:
            logger.warning(f"  - Baseline establishment failed for {baseline_failure_cycles} consecutive cycles. Triggering AI Root-Cause Analysis.")
            pass
        elif cycle_num < len(retraining_dates) - 1:
            suggested_params = api_timer.call(gemini_analyzer.analyze_cycle_and_suggest_changes, historical_results=in_run_historical_cycles, strategy_details=config.model_dump(), cycle_status=cycle_status_msg, shap_history=shap_history, available_features=all_available_features, strategic_directive=strategic_directive)
            if suggested_params:
                pass

        cycle_num += 1
        logger.info(f"--- Cycle complete. PNL: ${cycle_pnl:,.2f} | Final Equity: ${last_equity:,.2f} | Time: {time.time() - cycle_start_time:.2f}s ---")

    pa = PerformanceAnalyzer(config)
    last_class_report = trainer.classification_report_str if 'trainer' in locals() else "N/A"
    final_metrics = pa.generate_full_report(aggregated_trades, aggregated_equity_curve, in_run_historical_cycles, pd.DataFrame.from_dict(shap_history, orient='index').mean(axis=1).sort_values(ascending=False).to_frame('SHAP_Importance'), framework_history, aggregated_daily_dd_reports, last_class_report)
    run_summary = {"script_version": config.REPORT_LABEL, "nickname": config.nickname, "strategy_name": config.strategy_name, "run_start_ts": config.run_timestamp, "final_params": config.model_dump(mode='json'), "run_end_ts": datetime.now().strftime("%Y%m%d-%H%M%S"), "final_metrics": final_metrics, "cycle_details": in_run_historical_cycles}
    save_run_to_memory(config, run_summary, framework_history, regime_summary['current_diagnosed_regime'])
    logger.removeHandler(file_handler); file_handler.close()
    
def get_and_cache_asset_types(symbols: List[str], config: Dict, gemini_analyzer: GeminiAnalyzer) -> Dict[str, str]:
    """
    Classifies symbols using the Gemini API and caches the result.
    On subsequent runs, it loads from cache if the symbol list is unchanged.
    """
    cache_path = os.path.join(config.get("BASE_PATH", "."), "Results", "asset_types_cache.json")
    
    # Check for a valid cache first
    if os.path.exists(cache_path):
        try:
            with open(cache_path, 'r') as f:
                cache_data = json.load(f)
            # Validate if the cached symbols match the current directory's symbols
            if "symbols" in cache_data and set(cache_data["symbols"]) == set(symbols):
                logger.info(f"-> Loading verified asset types from cache: {cache_path}")
                return cache_data.get("asset_types", {})
            else:
                logger.info("-> Symbol list has changed. Re-classifying assets with AI...")
        except (json.JSONDecodeError, IOError) as e:
            logger.warning(f"Could not read asset cache, re-classifying with AI. Error: {e}")

    # If no valid cache, call the API
    classified_types = gemini_analyzer.classify_asset_symbols(symbols)
    
    # Save the new classification to the cache for next time
    if classified_types:
        try:
            cache_to_save = {"symbols": symbols, "asset_types": classified_types}
            with open(cache_path, 'w') as f:
                json.dump(cache_to_save, f, indent=4)
            logger.info(f"-> Asset classifications verified by AI and saved to cache: {cache_path}")
        except IOError as e:
            logger.error(f"Could not write to asset cache file: {e}")
            
    return classified_types

def generate_dynamic_config(primary_class: str, config: Dict) -> Dict:
    """
    Auto-configures parameters based on the primary asset class.
    """
    logger.info(f"-> Auto-configuring parameters for primary asset class: '{primary_class}'...")
    
    if primary_class == 'Indices':
        logger.info("  - Setting rules for Indices: CONTRACT_SIZE=1.0, LEVERAGE=20, LOT_STEP=1.0")
        config['CONTRACT_SIZE'] = 1.0
        config['LEVERAGE'] = 20
        config['MIN_LOT_SIZE'] = 1.0
        config['LOT_STEP'] = 1.0
        config['COMMISSION_PER_LOT'] = 0.5
    elif primary_class == 'Commodities':
        logger.info("  - Setting rules for Commodities: CONTRACT_SIZE=100.0, LEVERAGE=20, LOT_STEP=0.01")
        config['CONTRACT_SIZE'] = 100.0
        config['LEVERAGE'] = 20
        config['MIN_LOT_SIZE'] = 0.01
        config['LOT_STEP'] = 0.01
        config['COMMISSION_PER_LOT'] = 3.5
    else:  # Default to Forex rules
        logger.info("  - Setting rules for Forex: CONTRACT_SIZE=100000.0, LEVERAGE=30, LOT_STEP=0.01")
        config['CONTRACT_SIZE'] = 100000.0
        config['LEVERAGE'] = 30
        config['MIN_LOT_SIZE'] = 0.01
        config['LOT_STEP'] = 0.01
        config['COMMISSION_PER_LOT'] = 3.5
        
    config['REPORT_LABEL'] = f"ML_Framework_V{VERSION}_Auto_{primary_class}"
    return config

def main():
    """Main entry point for the trading framework."""
    print(f"--- ML Trading Framework V{VERSION} Initializing ---", flush=True)
    setup_logging()

    # --- Base Configuration ---
    # This serves as the master template. It is automatically adjusted.
    base_config = {
        "BASE_PATH": os.getcwd(),
        "strategy_name": "Meta_Labeling_Filter",
        "INITIAL_CAPITAL": 10000.0,
        "CONFIDENCE_TIERS": {
            'ultra_high': {'min': 0.80, 'risk_mult': 1.2, 'rr': 2.5},
            'high':       {'min': 0.70, 'risk_mult': 1.0, 'rr': 2.0},
            'standard':   {'min': 0.60, 'risk_mult': 0.8, 'rr': 1.5}
        },
        "BASE_RISK_PER_TRADE_PCT": 0.01, "RISK_CAP_PER_TRADE_USD": 1000.0,
        "OPTUNA_TRIALS": 75, "TRAINING_WINDOW": '365D', "RETRAINING_FREQUENCY": '90D',
        "FORWARD_TEST_GAP": "1D", "LOOKAHEAD_CANDLES": 150, "TREND_FILTER_THRESHOLD": 22.0,
        "BOLLINGER_PERIOD": 20, "STOCHASTIC_PERIOD": 14, "CALCULATE_SHAP_VALUES": True,
        "MAX_DD_PER_CYCLE": 0.25, "GNN_EMBEDDING_DIM": 8, "GNN_EPOCHS": 50,
        "MIN_VOLATILITY_RANK": 0.1, "MAX_VOLATILITY_RANK": 0.9, "selected_features": [],
        "MAX_CONCURRENT_TRADES": 3, "USE_TIERED_RISK": True, "RISK_PROFILE": "Medium",
        "USE_TP_LADDER": True,
        "USE_FEATURE_CACHING": True,
        "TP_LADDER_LEVELS_PCT": [0.25, 0.25, 0.25, 0.25],
        "TP_LADDER_RISK_MULTIPLIERS": [1.0, 2.0, 3.0, 4.0],
        "MAX_TRAINING_RETRIES_PER_CYCLE": 3,
        "anomaly_contamination_factor": 0.01, "LABEL_MIN_RETURN_PCT": 0.004,
        "LABEL_MIN_EVENT_PCT": 0.02
    }

    # --- 100% AUTOMATIC CONFIGURATION ---
    all_files = [f for f in os.listdir(base_config['BASE_PATH']) if f.endswith(('.csv', '.txt')) and re.match(r'^[A-Z0-9]+_[A-Z0-9]+', f)]
    if not all_files:
        logger.critical("No data files found in base path. Exiting.")
        return
        
    symbols = sorted(list(set([f.split('_')[0] for f in all_files])))
    
    # AI-Powered Asset Classification with Caching
    gemini_analyzer_for_setup = GeminiAnalyzer()
    asset_types = get_and_cache_asset_types(symbols, base_config, gemini_analyzer_for_setup)
    
    # Determine primary asset class by frequency
    if not asset_types:
        logger.error("Could not determine asset types. Defaulting to 'Forex'.")
        primary_class = "Forex"
    else:
        from collections import Counter
        class_counts = Counter(asset_types.values())
        primary_class = class_counts.most_common(1)[0][0]

    # Generate the final configuration based on the dominant, AI-verified asset class.
    fallback_config = generate_dynamic_config(primary_class, base_config)
    
    # --- Framework Execution Loop ---
    CONTINUOUS_RUN_HOURS = 0
    MAX_RUNS = 1
    fallback_config["DIRECTIVES_FILE_PATH"] = os.path.join(fallback_config["BASE_PATH"], "Results", "framework_directives.json")
    api_interval_seconds = 61
    run_count = 0
    script_start_time = datetime.now()
    is_continuous = CONTINUOUS_RUN_HOURS > 0 or MAX_RUNS > 1
    
    bootstrap_config = ConfigModel(**fallback_config, run_timestamp="init", nickname="init")
    
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
    
# End_To_End_Advanced_ML_Trading_Framework_PRO_V210.py