# End_To_End_Advanced_ML_Trading_Framework_PRO_V210_Linux.py
#
# V210 UPDATE (Dynamic Operating States & Conservative Baseline):
#   1. ADDED (Operating State Architecture): Implemented a new state management system
#      using an `OperatingState` Enum (`CONSERVATIVE_BASELINE`, `AGGRESSIVE_EXPANSION`,
#      `DRAWDOWN_CONTROL`) to allow the framework to dynamically change its behavior.
#   2. ADDED (Configurable State Parameters): A new `STATE_BASED_CONFIG` dictionary has
#      been added to the main configuration. This allows for defining specific risk
#      parameters (`max_dd_per_cycle`, `base_risk_pct`, etc.) for each operating state.
#   3. IMPLEMENTED (Group 1 - Conservative Baseline): The framework now starts in and
#      defaults to the `CONSERVATIVE_BASELINE` state. Before each cycle, a new function
#      `_apply_operating_state_rules` enforces the low-risk parameters defined in the
#      config, ensuring a disciplined, capital-preservation-first approach.
#   4. IMPLEMENTED (Dynamic AI Optimization Objective): The `ModelTrainer`'s optimization
#      process is now state-aware. In the `CONSERVATIVE_BASELINE` state, it uses a
#      conservative objective function (Maximize Sharpe/Calmar, penalize complexity)
#      to find the most stable and reliable model.
#   5. ENHANCED (AI Prompt Awareness): The initial AI setup prompt has been updated to
#      make it aware of the new state machine, instructing it to select a robust
#      strategy suitable for establishing a stable baseline.
#   6. Removed all code related to the LSTM model, including the TensorFlow/Keras imports. 
#
# --- SCRIPT VERSION ---
VERSION = "210"
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

# --- LOAD ENVIRONMENT VARIABLES ---
from dotenv import load_dotenv
load_dotenv()
# --- END ---

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import shap
import xgboost as xgb
from xgboost.callback import EarlyStopping as XGBoostEarlyStopping
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
import yfinance as yf
from hurst import compute_Hc

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

# This try-except block for Pruning can be removed entirely,
# but is left here as a harmless placeholder in case you reintroduce it later.
try:
    from optuna.integration import XGBoostPruningCallback
    PRUNING_AVAILABLE = True
except ModuleNotFoundError:
    PRUNING_AVAILABLE = False
    class XGBoostPruningCallback:
        def __init__(self, trial, observation_key): pass
        def __call__(self, env): pass

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

    # V208/V209 FIX: Set Optuna's verbosity to WARNING to prevent it from
    # interrupting the custom progress bar with INFO-level trial logs.
    optuna.logging.set_verbosity(optuna.logging.WARNING)

# --- FRAMEWORK STATE DEFINITION ---
class OperatingState(Enum):
    """Defines the operational states of the trading framework."""
    CONSERVATIVE_BASELINE = "Conservative Baseline"
    AGGRESSIVE_EXPANSION = "Aggressive Expansion"
    DRAWDOWN_CONTROL = "Drawdown Control"
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

    # --- AI & Optimization Parameters ---
    OPTUNA_TRIALS: conint(gt=0)
    MAX_TRAINING_RETRIES_PER_CYCLE: conint(ge=0) = 3
    CALCULATE_SHAP_VALUES: bool = True
    
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
            "confidence_gate_modifier": 1.0,
            "optimization_objective": ["maximize_calmar", "minimize_trades"]
        },
        OperatingState.AGGRESSIVE_EXPANSION: {
            "max_dd_per_cycle": 0.30,
            "base_risk_pct": 0.015,
            "max_concurrent_trades": 5,
            "confidence_gate_modifier": 0.95,
            "optimization_objective": ["maximize_pnl", "maximize_trades"]
        },
        OperatingState.DRAWDOWN_CONTROL: {
            "max_dd_per_cycle": 0.10,
            "base_risk_pct": 0.005,
            "max_concurrent_trades": 1,
            "confidence_gate_modifier": 1.05,
            "optimization_objective": ["maximize_calmar", "minimize_trades"]
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
                    
                    # --- NEW ROBUST PARSING LOGIC ---
                    if "candidates" in result and result["candidates"]:
                        content = result["candidates"][0].get("content", {})
                        parts = content.get("parts", [])
                        
                        # Find the first text part in the response, as the final answer will be text.
                        for part in parts:
                            if "text" in part:
                                logger.info(f"Successfully received and extracted text response from model: {model}")
                                return part["text"]
                    
                    # If the loop completes and no text part was found, the response is invalid for our purposes.
                    logger.error(f"Invalid Gemini response structure from {model}: No 'text' part found in the final response. Response: {result}")
                    # This will trigger the retry mechanism.
                    continue

                except requests.exceptions.HTTPError as e:
                    logger.error(f"!! HTTP Error for model '{model}': {e.response.status_code} {e.response.reason}")
                    logger.error(f"   - API Error Details: {e.response.text}")
                    # Break from the retry loop for this model; a server error is unlikely to be fixed by retrying.
                    break 
                except requests.exceptions.RequestException as e:
                    logger.error(f"Gemini API request failed for model {model} on attempt {attempt + 1} (Network Error): {e}")
                except json.JSONDecodeError as e:
                    logger.error(f"Failed to decode Gemini response JSON from {model}: {e} - Response: {response.text}")

            logger.warning(f"Failed to get a valid text response from model {model} after all retries.")

        logger.critical("API connection failed for all primary and backup models. Could not get a final text response.")
        return "{}"

    def _extract_json_from_response(self, response_text: str) -> dict:
        """
        Extracts the first valid JSON object from the AI's response text using
        the robust json.JSONDecoder.raw_decode method. Includes comprehensive
        logging for debugging AI responses.
        """
        logger = logging.getLogger("ML_Trading_Framework")

        # Always log the full raw AI response to the debug file for traceability.
        logger.debug(f"RAW AI RESPONSE TO BE PARSED:\n--- START ---\n{response_text}\n--- END ---")

        decoder = JSONDecoder()
        pos = 0
        while pos < len(response_text):
            # Find the starting position of a potential JSON object.
            brace_pos = response_text.find('{', pos)
            if brace_pos == -1:
                break  # No more '{' characters found, exit the loop.

            try:
                # Attempt to decode a JSON object from the current position.
                suggestions, end_pos = decoder.raw_decode(response_text, brace_pos)
                logger.info("Successfully extracted JSON object using JSONDecoder.raw_decode.")

                # Ensure the extracted object is the expected dictionary type.
                if not isinstance(suggestions, dict):
                    logger.warning(
                        f"Parsed JSON was type {type(suggestions)}, not a dictionary. Continuing search."
                    )
                    # A valid JSON object was found, but not the type we need.
                    # Continue searching from the end of this object.
                    pos = end_pos
                    continue

                # Un-nest parameters if the AI nested them under 'current_params'.
                if isinstance(suggestions.get("current_params"), dict):
                    nested_params = suggestions.pop("current_params")
                    suggestions.update(nested_params)

                return suggestions

            except JSONDecodeError as e:
                # A decoding attempt failed. This is not critical yet, as there may be
                # other valid JSON objects further in the string.
                logger.warning(
                    f"JSON decoding failed at position {brace_pos}. Error: {e}. Skipping to next candidate."
                )
                pos = brace_pos + 1

        # If the loop completes without finding a valid dictionary, log a critical error.
        # The raw response has already been logged at the DEBUG level.
        logger.error("!! CRITICAL JSON PARSE FAILURE !! No valid JSON dictionary could be decoded from the AI response.")
        return {}

    def get_initial_run_setup(self, script_version: str, ledger: Dict, memory: Dict, playbook: Dict, health_report: Dict, directives: List[Dict], data_summary: Dict, diagnosed_regime: str, regime_champions: Dict, correlation_summary_for_ai: str, macro_context: Dict) -> Dict:
        if not self.api_key_valid:
            logger.warning("No API key. Skipping AI-driven setup and using default config.")
            return {}

        logger.info("-> Performing Initial AI Analysis & Setup (Grounded Search with Correlation Context)...")
        asset_list = ", ".join(data_summary.get('assets_detected', []))

        # CORRECTED PROMPT V2: The instructions for the JSON output are now extremely specific.
        task_prompt = (
            "**YOUR TASK: Perform a grounded analysis to create the complete initial run configuration. This involves three main steps.**\n\n"
            "**NEW CONTEXT:** The framework now operates in different states. It will start in a **'CONSERVATIVE_BASELINE'** state. Your primary goal is to choose a strategy and parameters that are **robust, stable, and suitable for capital preservation** to establish a solid baseline.\n\n"
            "**STEP 1: DYNAMIC BROKER SIMULATION (Grounded Search)**\n"
            f"   - The assets being traded are: **{asset_list}**. \n"
            "   - **Action:** Use Google Search to find typical trading costs for these assets on a retail **ECN/Raw Spread** account.\n"
            "   - **Action:** In your JSON response, you must:\n"
            "       - Populate `COMMISSION_PER_LOT` with a single `float` value representing the **average** commission across the assets.\n"
            "       - Update the `SPREAD_CONFIG` dictionary with per-asset values in the format: `\"symbol\": {\"normal_pips\": <value>, \"volatile_pips\": <value>}`.\n"
            "       - **Crucially**, `SPREAD_CONFIG` must also include the `\"default\"` key.\n\n"
            "**STEP 2: STRATEGY SELECTION (Grounded Search & Context Synthesis)**\n"
            "   - **Synthesize Context:** Analyze `MACROECONOMIC CONTEXT`, `MARKET DATA SUMMARY`, and `ASSET CORRELATION SUMMARY`.\n"
            "   - **Grounded Calendar Check:** Search the economic calendar for the next 5 trading days.\n"
            "   - **Decide on a Strategy:** Given the 'CONSERVATIVE_BASELINE' goal, select a **robust, well-understood strategy** from the playbook. **STRONGLY PREFER** strategies with 'low' or 'medium' complexity. Avoid highly specialized or experimental strategies for this initial run.\n\n"
            "**STEP 3: CONFIGURATION & NICKNAME**\n"
            "   - Provide the full configuration in the JSON response.\n"
            "   - Handle nickname generation as per the rules."
        )
        
        # NEW: Explicitly define the required JSON structure for the AI.
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

        # The validation logic remains the same, but the stricter prompt makes success more likely.
        if suggestions and "strategy_name" in suggestions:
            logger.info("  - Initial AI Analysis and Setup complete.")
            return suggestions
        else:
            # Enhanced logging for failure case
            logger.error("  - AI-driven setup failed validation. The returned JSON was missing 'strategy_name' or was empty.")
            logger.debug(f"    - Invalid dictionary received from AI: {suggestions}")
            return {}

        # Validate that the AI provided the new required fields
        if suggestions and "strategy_name" in suggestions and "LABELING_METHOD" in suggestions and "MIN_F1_SCORE_GATE" in suggestions:
            logger.info("  - Initial AI Analysis and Setup complete.")
            logger.info(f"  - AI selected Labeling Method: '{suggestions.get('LABELING_METHOD')}'")
            logger.info(f"  - AI set dynamic F1 Score Gate: {suggestions.get('MIN_F1_SCORE_GATE')}")
            return suggestions
        else:
            logger.error("  - AI-driven setup failed to return a valid configuration with the required new fields (LABELING_METHOD, MIN_F1_SCORE_GATE). Reverting to fallback.")
            # Construct a safe fallback if the AI fails
            return {
                "strategy_name": "ClassicBollingerRSI",
                "selected_features": ["RSI", "ADX", "bollinger_bandwidth", "ATR"],
                "analysis_notes": "CRITICAL FALLBACK: AI setup failed to provide dynamic gates/labels; reverted to a safe default.",
                "LABELING_METHOD": "volatility_adjusted",
                "MIN_F1_SCORE_GATE": 0.40,
                "OPTUNA_TRIALS": 30
            }

    def analyze_cycle_and_suggest_changes(
        self,
        historical_results: List[Dict],
        framework_history: Dict,
        available_features: List[str],
        strategy_details: Dict,
        cycle_status: str,
        shap_history: Dict[str, List[float]],
        all_optuna_trials: List[Dict],
        cycle_start_date: str,
        cycle_end_date: str,
        correlation_summary_for_ai: str,
        macro_context: Dict,
        account_health_state: str,
        overall_drawdown_pct: float,
        strategic_forecast: Optional[Dict] = None
    ) -> Dict:
        if not self.api_key_valid: return {}

        base_prompt_intro = "You are an expert trading model analyst and portfolio manager. Your primary goal is to create a STABLE and PROFITABLE strategy by making intelligent, data-driven changes. You must balance aggressive profit-seeking with disciplined risk management based on the overall health of the account."

        json_schema_definition = (
            "### REQUIRED JSON RESPONSE STRUCTURE ###\n"
            "// If no changes are needed, return an empty JSON object: {}\n"
            "{\n"
            '  "analysis_notes": str,            // Your detailed reasoning for the suggested changes, or why no changes are needed.\n'
            '  "model_confidence_score": int,    // Your 1-10 confidence in this configuration decision.\n'
            '  "MAX_DD_PER_CYCLE": Optional[float],\n'
            '  "MAX_CONCURRENT_TRADES": Optional[int],\n'
            '  "selected_features": Optional[List[str]]\n'
            '  // ... and any other parameter from the ConfigModel you wish to change.\n'
            "}\n"
            "Respond ONLY with the JSON object wrapped between `BEGIN_JSON` and `END_JSON` markers.\n"
        )

        health_based_instructions = ""
        if account_health_state == 'Critical':
            health_based_instructions = (
                "**CRITICAL DIRECTIVE: The account is in a severe drawdown. Your absolute top priority is CAPITAL PRESERVATION. "
                "You MUST suggest changes that drastically reduce risk. This includes, but is not limited to: "
                "1. Proposing a much lower `MAX_DD_PER_CYCLE`. "
                "2. Reducing `MAX_CONCURRENT_TRADES` to 1. "
                "3. Suggesting a less aggressive `RISK_PROFILE` ('Low'). "
                "Do not propose aggressive changes until the drawdown is significantly recovered.**"
            )
        elif account_health_state == 'Caution':
            health_based_instructions = (
                "**CAUTIONARY DIRECTIVE: The account is in a moderate drawdown. Your primary goal is to stabilize the equity curve. "
                "Propose conservative changes. Consider reducing `MAX_DD_PER_CYCLE` or suggesting a more defensive feature set.**"
            )

        task_guidance = ""
        if any(cycle.get("Status") == "Circuit Breaker" for cycle in historical_results):
            task_guidance = (
                "**CRITICAL: CIRCUIT BREAKER TRIPPED!**\n"
                "The last cycle failed immediately due to excessive drawdown. This indicates a severe model generalization problem or a wrong strategy for the current regime. Your top priority is to stabilize the next cycle.\n"
                "**DO NOT just lower the risk.** Propose a more fundamental change:\n"
                "1. **Re-evaluate the Strategy:** Was the strategy type (e.g., breakout) wrong for the market? If so, suggest a different one (e.g., mean-reversion).\n"
                "2. **Simplify the Features:** Propose a smaller, more robust set of `selected_features` based on the most stable SHAP values.\n"
                "3. **Adjust Labeling:** Consider suggesting a wider `SL_ATR_MULTIPLIER` to give trades more room to breathe as a stability measure."
            )
        elif cycle_status == "TRAINING_FAILURE":
            task_guidance = (
                "**CRITICAL: MODEL TRAINING FAILURE!**\n"
                "The model failed the quality gate. Your top priority is to propose a change that **increases model stability and signal quality**. Your first instinct should be to **drastically simplify the feature set** based on the most historically stable features from the SHAP history. Avoid failed hyperparameters from the Optuna history."
            )
        else: # Standard cycle or Probation
            task_guidance = (
                "**STANDARD CYCLE REVIEW**\n"
                "Your task is to synthesize all data points into a coherent set of changes. Propose a new configuration that improves robustness and profitability. Your suggestions MUST be consistent with the current `PORTFOLIO HEALTH STATUS`."
            )

        optuna_summary = {}
        if all_optuna_trials:
            sorted_trials = sorted(all_optuna_trials, key=lambda x: x.get('value', -99), reverse=True)
            optuna_summary = {"best_5_trials": sorted_trials[:5], "worst_5_trials": sorted_trials[-5:]}

        data_context = (
            f"--- DATA FOR YOUR ANALYSIS ---\n\n"
            f"**A. PORTFOLIO HEALTH STATUS (Most Important Context):**\n"
            f"  - `account_health_state`: '{account_health_state}'\n"
            f"  - `overall_drawdown_pct`: {overall_drawdown_pct:.2%}\n\n"
            f"**B. CURRENT RUN - CYCLE-BY-CYCLE HISTORY:**\n{json.dumps(self._sanitize_dict(historical_results), indent=2)}\n\n"
            f"**C. MACROECONOMIC CONTEXT:**\n{json.dumps(self._sanitize_dict(macro_context), indent=2)}\n\n"
            f"**D. ASSET CORRELATION SUMMARY (INTERNAL):**\n{correlation_summary_for_ai}\n\n"
            f"**E. FEATURE IMPORTANCE HISTORY (SHAP values over time):**\n{json.dumps(self._sanitize_dict(shap_history), indent=2)}\n\n"
            f"**F. HYPERPARAMETER HISTORY (Sample from Optuna Trials):**\n{json.dumps(self._sanitize_dict(optuna_summary), indent=2)}\n\n"
            f"**G. CURRENT STRATEGY & AVAILABLE FEATURES:**\n`strategy_name`: {strategy_details.get('strategy_name')}\n`available_features`: {available_features}\n"
        )
        prompt = (
            f"{base_prompt_intro}\n\n"
            f"**YOUR TASK:**\n{health_based_instructions}\n{task_guidance}\n\n"
            f"{json_schema_definition}\n\n{data_context}"
        )

        response_text = self._call_gemini(prompt)
        suggestions = self._extract_json_from_response(response_text)
        return suggestions

    def select_best_tradeoff(self, best_trials: List[optuna.trial.FrozenTrial], risk_profile: str) -> int:
        """
        Analyzes a Pareto front of Optuna trials and selects the best one based on a risk profile.
        """
        if not self.api_key_valid:
            logger.warning("No API key. Skipping AI-driven trade-off selection. Selecting trial with highest Calmar.")
            # Fallback: find the trial with the best first objective (Calmar)
            return max(best_trials, key=lambda t: t.values[0]).number

        if not best_trials:
            logger.error("`select_best_tradeoff` called with no trials. Cannot proceed.")
            # This case should ideally be handled before calling, but as a safeguard:
            raise ValueError("Cannot select from an empty list of trials.")

        # Convert trials to a simplified, readable format for the AI
        trial_summaries = []
        for trial in best_trials:
            # The objectives are [calmar, -num_trades]
            calmar = trial.values[0] if trial.values and len(trial.values) > 0 else 0
            num_trades = -trial.values[1] if trial.values and len(trial.values) > 1 else 0
            trial_summaries.append(
                f" - Trial {trial.number}: Calmar Ratio = {calmar:.3f}, Avg. Trades per Cycle = {num_trades:.1f}"
            )
        
        trials_text = "\n".join(trial_summaries)

        prompt = (
            "You are a portfolio manager performing model selection. You have run a multi-objective optimization, "
            "resulting in a Pareto front of non-dominated models. Your task is to select the single best model "
            "that aligns with the specified `RISK_PROFILE`.\n\n"
            "**ANALYSIS GUIDELINES:**\n"
            f" - The current `RISK_PROFILE` is: **'{risk_profile}'**\n"
            "   - 'Low' risk profile: Strongly prefer models with lower trade frequency, even at the cost of some Calmar. Stability and cost-efficiency are paramount.\n"
            "   - 'Medium' risk profile: Seek a balance. The best Calmar is desired, but not if it comes with an excessive number of trades.\n"
            "   - 'High' risk profile: Prioritize the highest Calmar Ratio. A higher number of trades is acceptable if it generates superior risk-adjusted returns.\n"
            " - The objectives are to MAXIMIZE Calmar Ratio and MINIMIZE the number of trades.\n\n"
            "**PARETO FRONT OF MODELS:**\n"
            f"{trials_text}\n\n"
            "**YOUR TASK:**\n"
            "Review the trials and the risk profile. Respond ONLY with a single JSON object containing the number of the trial you have selected. "
            "Provide a brief justification for your choice in the `analysis_notes` key.\n\n"
            "**JSON OUTPUT FORMAT:**\n"
            "```json\n"
            "{\n"
            '  "selected_trial_number": int, // The number of the trial you have chosen.\n'
            '  "analysis_notes": str // Your reasoning.\n'
            "}\n"
            "```"
        )

        response_text = self._call_gemini(prompt)
        suggestions = self._extract_json_from_response(response_text)

        selected_trial_number = suggestions.get('selected_trial_number')

        if isinstance(selected_trial_number, int):
            # Verify the AI chose a valid trial number
            valid_numbers = {t.number for t in best_trials}
            if selected_trial_number in valid_numbers:
                logger.info(f"  - AI has selected Trial #{selected_trial_number} based on '{risk_profile}' profile.")
                logger.info(f"  - AI Rationale: {suggestions.get('analysis_notes', 'N/A')}")
                return selected_trial_number
            else:
                logger.error(f"  - AI selected an invalid trial number ({selected_trial_number}). Valid choices were: {valid_numbers}. Falling back to best Calmar.")
                return max(best_trials, key=lambda t: t.values[0]).number
        else:
            logger.error("  - AI failed to return a valid trial number in the expected format. Falling back to best Calmar.")
            return max(best_trials, key=lambda t: t.values[0]).number

    def propose_strategic_intervention(self, failure_history: List[Dict], playbook: Dict, last_failed_strategy: str, quarantine_list: List[str], dynamic_best_config: Optional[Dict] = None) -> Dict:
        if not self.api_key_valid: return {}
        logger.warning("! STRATEGIC INTERVENTION !: Current strategy has failed repeatedly. Engaging AI for a new path.")
        
        # --- PHASE 3 CHANGE: Add a new option for the AI ---
        # If the strategy is quarantined, give the AI the option to invent a new one.
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
                f"{generative_option_prompt}" # Add the new option here
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
        pre_analysis_summary: str,
        current_config: Dict,
        playbook: Dict,
        quarantine_list: List[str]
    ) -> Dict:
        """
        [Phase 2 Implemented] Called mid-cycle after multiple training failures to propose
        a major strategic pivot. The prompt is now structured to guide the AI toward
        a hierarchical set of solutions based on a heuristic pre-analysis from the framework.
        """
        if not self.api_key_valid: return {}
        logger.warning("! STRATEGIC INTERVENTION !: Multiple training attempts failed. Engaging AI for a major course-correction.")

        available_playbook = {k: v for k, v in playbook.items() if k not in quarantine_list and not v.get("retired")}

        json_schema_definition = (
            "### REQUIRED JSON RESPONSE STRUCTURE ###\n"
            "// You MUST choose exactly ONE action from the list below based on the pre-analysis.\n"
            "{\n"
            '  "action": str, // MUST be one of: "ADJUST_METRICS", "REDEFINE_LABELS", "SWITCH_STRATEGY", "CONTINUE_STANDARD_RETRY"\n'
            '  "parameters": Optional[Dict], // Required for all actions except "CONTINUE". Contains the new values.\n'
            '  "analysis_notes": str // Your detailed reasoning for the chosen action, referencing the pre-analysis.\n'
            "}\n"
            '// Example for ADJUST_METRICS: {"action": "ADJUST_METRICS", "parameters": {"MIN_F1_SCORE_GATE": 0.50}}\n'
            '// Example for REDEFINE_LABELS: {"action": "REDEFINE_LABELS", "parameters": {"LABELING_METHOD": "volatility_adjusted", "TP_ATR_MULTIPLIER": 2.5}}\n'
            '// Example for SWITCH_STRATEGY: {"action": "SWITCH_STRATEGY", "parameters": {"strategy_name": "...", "selected_features": [...]}}\n'
        )

        task_prompt = (
            "**PRIME DIRECTIVE: STRATEGIC INTERVENTION**\n"
            "The current model has failed its first two training attempts. Our internal heuristics have performed a pre-analysis of these failures. Your task is to review this analysis and the raw data, then decide on the single best course of action to fix the problem.\n\n"
            "**YOUR OPTIONS (Choose ONE):**\n"
            "1.  **`ADJUST_METRICS`:** Choose this if the pre-analysis indicates a **'High Profitability / Low Accuracy'** problem. This means the model is profitable in backtests but isn't a good classifier. Lowering the `MIN_F1_SCORE_GATE` is the correct response.\n\n"
            "2.  **`REDEFINE_LABELS`:** Choose this if the pre-analysis suggests a **'Fundamental Model/Label Issue'**. The model can't learn the current trade definition. You can propose changing the `LABELING_METHOD` itself (options: 'standard', 'volatility_adjusted', 'trend_quality', 'optimal_entry') or adjust the `TP_ATR_MULTIPLIER` / `SL_ATR_MULTIPLIER` values.\n\n"
            "3.  **`SWITCH_STRATEGY`:** Choose this if the pre-analysis points to a **'Strategy-Regime Mismatch'** or a fundamental failure. You must select a completely different strategy from the playbook that is better suited to the environment and provide a new `selected_features` list for it.\n\n"
            "4.  **`CONTINUE_STANDARD_RETRY`:** A fallback option if you believe the pre-analysis is wrong and a standard retry is sufficient. Use this sparingly."
        )

        prompt = (
            "You are a lead quantitative strategist performing a real-time intervention on a failing model.\n\n"
            f"{task_prompt}\n\n"
            f"{json_schema_definition}\n"
            "Respond ONLY with the JSON object wrapped between `BEGIN_JSON` and `END_JSON` markers.\n\n"
            "--- EVIDENCE & CONTEXT ---\n\n"
            f"**1. HEURISTIC PRE-ANALYSIS (Your Primary Guide):**\n{pre_analysis_summary}\n\n"
            f"**2. RAW FAILURE DATA (Attempt-by-Attempt):**\n{json.dumps(self._sanitize_dict(failure_history), indent=2)}\n\n"
            f"**3. CURRENT CONFIGURATION:**\n{json.dumps(self._sanitize_dict(current_config), indent=2)}\n\n"
            f"**4. AVAILABLE STRATEGIES (For a potential switch):**\n{json.dumps(self._sanitize_dict(available_playbook), indent=2)}\n"
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
    Enhanced feature engineering with additional simple features.
    This class includes standard technical indicators, advanced market structure analysis, volatility regime detection,
    and feature interaction/normalization to improve model performance.
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

    def _calculate_hurst_exponent(self, df: pd.DataFrame, window: int = 100) -> pd.DataFrame:
        if not HURST_AVAILABLE: return df
        df['hurst_exponent'] = df['Close'].rolling(window).apply(lambda x: compute_Hc(x)[0], raw=False)
        return df
        
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

    def _process_single_symbol_stack(self, symbol_data_by_tf: Dict[str, pd.DataFrame]) -> pd.DataFrame:
        """
        Orchestrates the entire feature engineering pipeline for a single symbol.
        It calls all individual feature calculation methods in a logical order.
        """
        # --- 1. Initial Data Validation ---
        base_df = symbol_data_by_tf.get(self.roles['base'])
        if base_df is None or base_df.empty:
            logger.warning("Base timeframe data missing or empty for a symbol. Skipping.")
            return pd.DataFrame()

        df = base_df.copy()
        # Ensure index is a DatetimeIndex for all time-series operations
        df.index = pd.to_datetime(df.index)

        # --- 2. Foundational Indicators ---
        # These are the basic building blocks for many other features.
        delta = df['Close'].diff()
        gain = delta.clip(lower=0).ewm(alpha=1/14, adjust=False).mean()
        loss = (-delta.clip(upper=0)).ewm(alpha=1/14, adjust=False).mean()
        rs = gain / loss.replace(0, 1e-9)
        df['RSI'] = 100 - (100 / (1 + rs))
        
        exp1 = df['Close'].ewm(span=12, adjust=False).mean()
        exp2 = df['Close'].ewm(span=26, adjust=False).mean()
        df['MACD'] = exp1 - exp2
        df['MACD_signal'] = df['MACD'].ewm(span=9, adjust=False).mean()
        df['MACD_hist'] = df['MACD'] - df['MACD_signal']

        high_low = df['High'] - df['Low']
        high_close = np.abs(df['High'] - df['Close'].shift())
        low_close = np.abs(df['Low'] - df['Close'].shift())
        tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
        df['ATR'] = tr.ewm(alpha=1/14, adjust=False).mean()

        ma = df['Close'].rolling(window=self.config.BOLLINGER_PERIOD).mean()
        std = df['Close'].rolling(window=self.config.BOLLINGER_PERIOD).std()
        df['bollinger_upper'] = ma + (std * 2)
        df['bollinger_lower'] = ma - (std * 2)
        df['bollinger_bandwidth'] = (df['bollinger_upper'] - df['bollinger_lower']) / ma.replace(0, 1e-9)

        low_k = df['Low'].rolling(window=self.config.STOCHASTIC_PERIOD).min()
        high_k = df['High'].rolling(window=self.config.STOCHASTIC_PERIOD).max()
        df['stoch_k'] = 100 * ((df['Close'] - low_k) / (high_k - low_k).replace(0, 1e-9))
        
        plus_dm = df['High'].diff().clip(lower=0)
        minus_dm = df['Low'].diff().clip(lower=0) # Note: this is typically -df['Low'].diff().clip(lower=0) but using absolute diffs
        plus_di = 100 * (plus_dm.ewm(alpha=1/14, adjust=False).mean() / df['ATR'])
        minus_di = 100 * (minus_dm.ewm(alpha=1/14, adjust=False).mean() / df['ATR'])
        dx = 100 * (np.abs(plus_di - minus_di) / (plus_di + minus_di).replace(0, 1e-9))
        df['ADX'] = dx.ewm(alpha=1/14, adjust=False).mean()

        df['momentum_20'] = df['Close'].pct_change(20)
        df['EMA_20'] = df['Close'].ewm(span=20, adjust=False).mean()
        df['EMA_50'] = df['Close'].ewm(span=50, adjust=False).mean()

        # --- 3. Contextual, Advanced, and Scientific Features ---
        # These calls leverage the robust helper methods we refined.
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
        df = self._calculate_hurst_exponent(df) # This now creates 'hurst_exponent' and 'hurst_intercept'
        df = self._calculate_hawkes_volatility(df)

        # --- 4. Regime and Meta-Feature Layer ---
        # This layer must come after the advanced features are calculated.
        if 'Close' in df.columns and not df['Close'].isnull().all():
            df['realized_volatility'] = df['Close'].pct_change().rolling(14).std() * np.sqrt(252 * 24 * 4) # Annualized example
            df['market_volatility_index'] = df['realized_volatility'].rank(pct=True)
        if 'ADX' in df.columns and not df['ADX'].isnull().all():
            df['market_trend_strength'] = df['ADX'].rank(pct=True)
        if 'market_volatility_index' in df.columns and 'market_trend_strength' in df.columns:
            combined_metric = df['market_volatility_index'].fillna(0.5) + df['market_trend_strength'].fillna(0.5)
            if combined_metric.nunique() > 1:
                df['market_regime'] = pd.qcut(combined_metric, 4, labels=False, duplicates='drop')
        
        # This call now creates interactions for both hurst_exponent and hurst_intercept
        df = self._calculate_meta_features(df)

        # --- 5. Final Anomaly Detection ---
        # This runs last to analyze the fully-featured dataset.
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

    def train(self, df_train: pd.DataFrame, feature_list: List[str], strategy_details: Dict) -> Optional[Tuple[Union[Pipeline, Dict, Tuple, GNNModel], float, float]]:
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
            X = df_train[feature_list].copy().fillna(0)
            if self.is_meta_model:
                logger.info("  - Meta-Labeling strategy detected. Training secondary filter model.")
                y = df_train['target'].astype(int); num_classes = 2
            else:
                y_map={-1:0,0:1,1:2}; y=df_train['target'].map(y_map).astype(int); num_classes = 3
        
        if X.empty or len(y.unique()) < num_classes:
            logger.error("  - Training data (X) is empty or not enough classes for the model. Aborting.")
            return None

        self.class_weights=dict(zip(np.unique(y),compute_class_weight(class_weight='balanced',classes=np.unique(y),y=y)))
        X_train_val, _, y_train_val, _ = train_test_split(X, y, test_size=0.1, shuffle=False)
        X_train, X_val, y_train, y_val = train_test_split(X_train_val, y_train_val, test_size=0.2, shuffle=False)

        if X_train.empty or X_val.empty:
            logger.error(f"  - Training aborted: Data split resulted in an empty training or validation set. (Train shape: {X_train.shape}, Val shape: {X_val.shape})")
            return None
        
        self.study=self._optimize_hyperparameters(df_train, X, y, num_classes)
        
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
                    self.study.best_trials, self.config.RISK_PROFILE
                )
                best_trial = next((t for t in self.study.best_trials if t.number == selected_trial_number), None)
                if not best_trial:
                    logger.error(f"Could not find trial number {selected_trial_number} in best_trials. Falling back to best primary objective.")
                    best_trial = max(self.study.best_trials, key=lambda t: t.values[0])
            except Exception as e:
                logger.error(f"An error occurred during AI-based trial selection: {e}. Falling back to best primary objective.")
                best_trial = max(self.study.best_trials, key=lambda t: t.values[0])

        best_params = best_trial.params
        best_values = best_trial.values
        logger.info(f"    - Selected Trial #{best_trial.number} -> Objectives: [Obj1: {best_values[0]:.4f}, Obj2: {best_values[1]:.2f}]")
        formatted_params = { k: (f"{v:.4g}" if isinstance(v, float) else v) for k, v in best_params.items() }
        logger.info(f"    - Selected params: {formatted_params}")

        self.best_threshold, f1_score_val = self._find_best_threshold(best_params, X_train, y_train, X_val, y_val, num_classes)
        final_pipeline = self._train_final_model(best_params, X_train_val, y_train_val, list(X.columns), num_classes)
        
        if final_pipeline is None:
            logger.error("  - Training aborted: Final model training failed.")
            return None

        logger.info("  - [SUCCESS] Model training complete.")
        
        if self.is_minirocket_model:
            return (final_pipeline, self.minirocket_transformer), self.best_threshold, f1_score_val
        else:
            return final_pipeline, self.best_threshold, f1_score_val

    def _optimize_hyperparameters(self, df_full_train: pd.DataFrame, X: pd.DataFrame, y: pd.Series, num_classes: int) -> Optional[optuna.study.Study]:
        # V210: Get the optimization objective based on the current operating state
        current_state = self.config.operating_state
        state_rules = self.config.STATE_BASED_CONFIG[current_state]
        optimization_objective_names = state_rules.get("optimization_objective", ["maximize_calmar", "minimize_trades"])

        logger.info(f"    - Starting hyperparameter optimization in state: '{current_state.value}'...")
        logger.info(f"    - Optimization Objective: {optimization_objective_names}")

        def dynamic_progress_callback(study: optuna.study.Study, trial: optuna.trial.FrozenTrial):
            n_trials = self.config.OPTUNA_TRIALS
            trial_number = trial.number + 1
            best_value = study.best_trials[0].values[0] if study.best_trials else float('nan')
            
            objective_display_name = optimization_objective_names[0].split('_')[1].capitalize()
            
            progress_str = f"> Optuna Optimization: Trial {trial_number}/{n_trials} | Best Score ({objective_display_name}): {best_value:.4f}"
            sys.stdout.write(f"\r{progress_str.ljust(80)}")
            sys.stdout.flush()

        objective = 'multi:softprob' if num_classes > 2 else 'binary:logistic'
        eval_metric = 'mlogloss' if num_classes > 2 else 'logloss'

        def custom_objective(trial: optuna.Trial) -> Tuple[float, float]:
            # Parameter passing to ensure compatibility.
            params = {
                'objective': objective, 'eval_metric': eval_metric, 'booster': 'gbtree',
                'tree_method': 'hist', 'seed': 42,
                'n_estimators': trial.suggest_int('n_estimators', 100, 800, step=50),
                'max_depth': trial.suggest_int('max_depth', 3, 7),
                'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.2, log=True),
                'subsample': trial.suggest_float('subsample', 0.6, 1.0),
                'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
                'gamma': trial.suggest_float('gamma', 0, 5),
                'reg_lambda': trial.suggest_float('reg_lambda', 1e-8, 5.0, log=True),
                'alpha': trial.suggest_float('alpha', 1e-8, 5.0, log=True),
                'early_stopping_rounds': 50
            }
            if num_classes > 2: params['num_class'] = num_classes

            complexity_penalty = 1.0 + (params['max_depth'] / 10.0) * 0.5 + (params['n_estimators'] / 1000.0) * 0.5
            skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
            fold_pnls = []
            fold_trade_counts = []

            for train_idx, val_idx in skf.split(X, y):
                X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
                y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
                df_val = df_full_train.iloc[val_idx]

                try:
                    scaler = RobustScaler()
                    X_train_scaled = scaler.fit_transform(X_train)
                    X_val_scaled = scaler.transform(X_val)
                    
                    # Pass params to constructor
                    model = xgb.XGBClassifier(**params)
                    
                    # Pass fit-specific params directly
                    model.fit(
                        X_train_scaled,
                        y_train,
                        eval_set=[(X_val_scaled, y_val)],
                        verbose=False,
                        sample_weight=y_train.map(self.class_weights)
                    )

                    preds_val = model.predict(X_val_scaled)
                    pnl_results = []
                    lookahead, tp_multiplier, sl_multiplier = self.config.LOOKAHEAD_CANDLES, self.config.TP_ATR_MULTIPLIER, self.config.SL_ATR_MULTIPLIER
                    for i in range(len(preds_val)):
                        signal = preds_val[i]
                        direction = 1 if signal == 2 else -1 if signal == 0 else 0
                        if direction == 0 or (i + lookahead) >= len(df_val): pnl_results.append(0); continue
                        entry_candle = df_val.iloc[i]
                        entry_price, atr = entry_candle['Close'], entry_candle['ATR']
                        if pd.isna(atr) or atr <= 0: pnl_results.append(0); continue
                        tp_dist, sl_dist = atr * tp_multiplier, atr * sl_multiplier
                        tp_level, sl_level = entry_price + (tp_dist * direction), entry_price - (sl_dist * direction)
                        future_candles = df_val.iloc[i+1 : i+1+lookahead]
                        future_highs, future_lows = future_candles['High'].values, future_candles['Low'].values
                        hit_tp_idx = np.where(future_highs >= tp_level if direction == 1 else future_lows <= tp_level)[0]
                        hit_sl_idx = np.where(future_lows <= sl_level if direction == 1 else future_highs >= sl_level)[0]
                        first_tp, first_sl = (hit_tp_idx[0] if len(hit_tp_idx) > 0 else np.inf), (hit_sl_idx[0] if len(hit_sl_idx) > 0 else np.inf)
                        if first_tp < first_sl: pnl_results.append(tp_dist * direction)
                        elif first_sl < first_tp: pnl_results.append(-sl_dist * direction)
                        else: pnl_results.append(0)
                    fold_pnls.append(pd.Series(pnl_results))
                    fold_trade_counts.append((pd.Series(pnl_results) != 0).sum())
                except Exception as e:
                    sys.stdout.write("\n")
                    logger.warning(f"Fold in trial {trial.number} failed with error: {e}")
                    return -10.0, -10.0 if "minimize" in optimization_objective_names[1] else 0.0

            full_pnl, avg_trades = pd.concat(fold_pnls), np.mean(fold_trade_counts)
            total_pnl, calmar = 0, -5.0
            if full_pnl.abs().sum() > 0:
                equity_curve = full_pnl.cumsum()
                running_max, total_pnl = equity_curve.cummax(), equity_curve.iloc[-1]
                drawdown = running_max - equity_curve
                max_drawdown = drawdown.max()
                calmar = total_pnl / max_drawdown if max_drawdown > 0 else total_pnl if total_pnl > 0 else -1.0
            
            final_calmar = calmar / complexity_penalty
            final_pnl = total_pnl / complexity_penalty
            
            obj1 = final_calmar if optimization_objective_names[0] == "maximize_calmar" else final_pnl
            obj2 = -avg_trades if optimization_objective_names[1] == "minimize_trades" else avg_trades
            
            return obj1, obj2

        try:
            study_name = f"{self.config.nickname}_{self.config.strategy_name}_{datetime.now().strftime('%Y%m%d-%H%M')}"
            
            pruner = optuna.pruners.MedianPruner()
            study = optuna.create_study(
                directions=['maximize', 'maximize'], 
                pruner=pruner,
                study_name=study_name
            )
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
            self.classification_report_str = classification_report(
                y_val, 
                best_preds, 
                target_names=target_names, 
                zero_division=0
            )
            logger.info("    - Stored detailed classification report for the best validation threshold.")
        else:
            self.classification_report_str = "Could not generate a valid prediction set for the report."

        logger.info(f"    - Best confidence gate found: {best_thresh:.2f} (Weighted F1 on confident preds: {best_f1:.4f})")
        return best_thresh, best_f1

    def _train_final_model(self,best_params:Dict,X:pd.DataFrame,y:pd.Series, feature_names: List[str], num_classes: int)->Optional[Pipeline]:
        logger.info("    - Training final model on all available data...")
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
    def __init__(self,config:ConfigModel):
        self.config=config
        self.is_meta_model = False
        self.is_transformer_model = False
        self.use_tp_ladder = self.config.USE_TP_LADDER

        if self.use_tp_ladder:
            if len(self.config.TP_LADDER_LEVELS_PCT) != len(self.config.TP_LADDER_RISK_MULTIPLIERS):
                logger.error("TP Ladder config error: 'TP_LADDER_LEVELS_PCT' and 'TP_LADDER_RISK_MULTIPLIERS' must have the same length. Disabling ladder.")
                self.use_tp_ladder = False
            elif not np.isclose(sum(self.config.TP_LADDER_LEVELS_PCT), 1.0):
                logger.error(f"TP Ladder config error: 'TP_LADDER_LEVELS_PCT' sum ({sum(self.config.TP_LADDER_LEVELS_PCT)}) is not 1.0. Disabling ladder.")
                self.use_tp_ladder = False
            else:
                 logger.info("Take-Profit Ladder is ENABLED. Standard partial profit logic will be skipped.")

    def _get_tiered_risk_params(self, equity: float) -> Tuple[float, int]:
        """Looks up risk percentage and max trades from the tiered config."""
        sorted_tiers = sorted(self.config.TIERED_RISK_CONFIG.keys())

        for tier_cap in sorted_tiers:
            if equity <= tier_cap:
                tier_settings = self.config.TIERED_RISK_CONFIG[tier_cap]
                profile_settings = tier_settings.get(self.config.RISK_PROFILE, tier_settings['Medium'])
                return profile_settings['risk_pct'], profile_settings['pairs']

        highest_tier_cap = sorted_tiers[-1]
        tier_settings = self.config.TIERED_RISK_CONFIG[highest_tier_cap]
        profile_settings = tier_settings.get(self.config.RISK_PROFILE, tier_settings['Medium'])
        return profile_settings['risk_pct'], profile_settings['pairs']
        
    def _calculate_realistic_costs(self, candle: Dict, on_exit: bool = False) -> Tuple[float, float]:
        """Calculates dynamic spread and variable slippage."""
        symbol = candle['Symbol']
        point_size = 0.0001 if 'JPY' not in symbol and candle.get('Open', 1) < 50 else 0.01

        spread_cost = 0
        if not on_exit:
            if symbol in self.config.SPREAD_CONFIG:
                spread_info = self.config.SPREAD_CONFIG[symbol]
            else:
                spread_info = self.config.SPREAD_CONFIG.get('default', {'normal_pips': 1.8, 'volatile_pips': 5.5})
            
            vol_rank = candle.get('market_volatility_index', 0.5)
            spread_pips = spread_info.get('volatile_pips', 5.5) if vol_rank > 0.8 else spread_info.get('normal_pips', 1.8)
            spread_cost = spread_pips * point_size

        slippage_cost = 0
        if self.config.USE_VARIABLE_SLIPPAGE:
            atr = candle.get('ATR', 0)
            vol_rank = candle.get('market_volatility_index', 0.5)
            random_factor = random.uniform(0.1, 1.2 if on_exit else 1.0) * self.config.SLIPPAGE_VOLATILITY_FACTOR
            slippage_cost = atr * vol_rank * random_factor

        return spread_cost, slippage_cost

    def run_backtest_chunk(self, df_chunk_in: pd.DataFrame, pipeline: Union[Pipeline, Dict, Tuple, GNNModel], confidence_threshold: float, initial_equity: float, strategy_details: Dict, run_peak_equity: float, feature_list: List[str], trade_lockout_until: Optional[pd.Timestamp] = None) -> Tuple[pd.DataFrame, pd.Series, bool, Optional[Dict], Dict]:
        if df_chunk_in.empty:
            return pd.DataFrame(), pd.Series([initial_equity]), False, None, {}

        df_chunk = df_chunk_in.copy()
        self.is_meta_model = strategy_details.get("requires_meta_labeling", False)
        self.is_transformer_model = strategy_details.get("requires_transformer", False)
        is_minirocket_model = strategy_details.get("requires_minirocket", False)
        is_gnn_model = strategy_details.get("requires_gnn", False)

        xgb_pipeline = None
        minirocket_transformer = None
        gnn_model = None
        
        if is_minirocket_model:
            xgb_pipeline, minirocket_transformer = pipeline
        elif is_gnn_model:
            gnn_model = pipeline
        elif not self.is_transformer_model:
            xgb_pipeline = pipeline

        trades, equity, equity_curve, open_positions = [], initial_equity, [initial_equity], {}
        chunk_peak_equity = initial_equity
        circuit_breaker_tripped = False
        breaker_context = None
        
        last_trade_pnl = 0.0

        daily_dd_report = {}
        current_day = None
        day_start_equity = initial_equity
        day_peak_equity = initial_equity

        # --- GNN Pre-computation (before loop) ---
        gnn_feature_df = None
        gnn_edge_index = None
        gnn_symbols = []
        if is_gnn_model and GNN_AVAILABLE:
            logger.info("  - Backtesting with GNN model. Pre-computing graph structure.")
            gnn_model.eval() # Set model to evaluation mode
            
            price_df = df_chunk.pivot_table(index=df_chunk.index, columns='Symbol', values='Close', aggfunc='last').ffill().bfill().dropna(axis=1)
            gnn_symbols = price_df.columns.tolist()
            
            if len(gnn_symbols) >= 2:
                corr_matrix = price_df.corr()
                edge_list = []
                for i in range(len(gnn_symbols)):
                    for j in range(i + 1, len(gnn_symbols)):
                        if abs(corr_matrix.iloc[i, j]) > 0.5: # Use same threshold as training
                            edge_list.append([i, j])
                            edge_list.append([j, i])
                gnn_edge_index = torch.tensor(edge_list, dtype=torch.long).t().contiguous()

                # Pivot all necessary features for fast lookup
                gnn_feature_list = self.config.selected_features
                gnn_feature_df = df_chunk.pivot(index=df_chunk.index, columns='Symbol', values=gnn_feature_list)
                # Recreate MultiIndex-like column names for lookup
                gnn_feature_df.columns = ['_'.join(map(str, col)).strip() for col in gnn_feature_df.columns.values]
                gnn_feature_df = gnn_feature_df.ffill().bfill()
            else:
                logger.warning("  - Not enough symbols to run GNN backtest. GNN will not generate signals.")
                is_gnn_model = False # Disable GNN for this chunk
        # --- End GNN Pre-computation ---

        def finalize_day_metrics(day_to_finalize, equity_at_close):
            if day_to_finalize is None: return
            daily_pnl = equity_at_close - day_start_equity
            daily_dd_pct = ((day_peak_equity - equity_at_close) / day_peak_equity) * 100 if day_peak_equity > 0 else 0
            daily_dd_report[day_to_finalize.isoformat()] = {'pnl': round(daily_pnl, 2), 'drawdown_pct': round(daily_dd_pct, 2)}
            
        def close_trade(pos_to_close, exit_price, exit_reason, candle_info):
            nonlocal equity, last_trade_pnl
            pnl = (exit_price - pos_to_close['entry_price']) * pos_to_close['direction'] * pos_to_close['lot_size'] * self.config.CONTRACT_SIZE
            commission_cost = self.config.COMMISSION_PER_LOT * pos_to_close['lot_size'] * 2
            net_pnl = pnl - commission_cost
            
            equity += net_pnl
            last_trade_pnl = net_pnl
            
            mae = abs(pos_to_close['mae_price'] - pos_to_close['entry_price'])
            mfe = abs(pos_to_close['mfe_price'] - pos_to_close['entry_price'])
            
            trade_record = {
                'ExecTime': candle_info['Timestamp'], 'Symbol': pos_to_close['symbol'], 'PNL': net_pnl, 
                'Equity': equity, 'Confidence': pos_to_close['confidence'], 
                'Direction': pos_to_close['direction'], 'ExitReason': exit_reason, 
                'MAE': round(mae, 5), 'MFE': round(mfe, 5)
            }
            trades.append(trade_record)
            equity_curve.append(equity)
            return net_pnl

        candles = df_chunk.reset_index().to_dict('records')

        for i in range(1, len(candles)):
            current_candle = candles[i]
            prev_candle = candles[i-1]

            account_health_state = 'Normal'
            if run_peak_equity > 0:
                overall_drawdown_pct = (run_peak_equity - equity) / run_peak_equity
                if overall_drawdown_pct > 0.30:
                    account_health_state = 'Critical'
                elif overall_drawdown_pct > 0.15:
                    account_health_state = 'Caution'
            
            candle_date = current_candle['Timestamp'].date()
            if candle_date != current_day:
                finalize_day_metrics(current_day, equity)
                current_day, day_start_equity, day_peak_equity = candle_date, equity, equity
            
            if not circuit_breaker_tripped:
                day_peak_equity = max(day_peak_equity, equity)
                chunk_peak_equity = max(chunk_peak_equity, equity)
                if equity > 0 and chunk_peak_equity > 0 and (chunk_peak_equity - equity) / chunk_peak_equity > self.config.MAX_DD_PER_CYCLE:
                    logger.warning(f"  - CYCLE CIRCUIT BREAKER TRIPPED! Drawdown exceeded {self.config.MAX_DD_PER_CYCLE:.0%} for this cycle. Closing all positions.")
                    circuit_breaker_tripped = True
                    trade_df = pd.DataFrame(trades)
                    breaker_context = {"num_trades_before_trip": len(trade_df), "pnl_before_trip": round(trade_df['PNL'].sum(), 2), "last_5_trades_pnl": [round(p, 2) for p in trade_df['PNL'].tail(5).tolist()]} if not trade_df.empty else {}
                    
                    for sym, pos in list(open_positions.items()):
                        close_trade(pos, current_candle['Open'], "Circuit Breaker", current_candle)
                        del open_positions[sym]
                    
                    continue
            
            if equity <= 0:
                logger.critical("  - ACCOUNT BLOWN!")
                break

            for symbol, pos in open_positions.items():
                if pos['direction'] == 1:
                    pos['mfe_price'] = max(pos['mfe_price'], current_candle['High'])
                    pos['mae_price'] = min(pos['mae_price'], current_candle['Low'])
                else:
                    pos['mfe_price'] = min(pos['mfe_price'], current_candle['Low'])
                    pos['mae_price'] = max(pos['mae_price'], current_candle['High'])
            
            symbols_to_close = []
            for symbol, pos in open_positions.items():
                exit_price, exit_reason = None, None
                candle_low, candle_high = current_candle['Low'], current_candle['High']
                
                sl_hit = (pos['direction'] == 1 and candle_low <= pos['sl']) or \
                         (pos['direction'] == -1 and candle_high >= pos['sl'])
                tp_hit = (pos['direction'] == 1 and candle_high >= pos['tp']) or \
                         (pos['direction'] == -1 and candle_low <= pos['tp'])

                if sl_hit:
                    exit_reason = "Stop Loss"
                    _, sl_slippage = self._calculate_realistic_costs(current_candle, on_exit=True)
                    exit_price = pos['sl'] - (sl_slippage * pos['direction'])
                elif tp_hit:
                    exit_reason = "Take Profit"
                    exit_price = pos['tp']

                if exit_price is not None:
                    close_trade(pos, exit_price, exit_reason, current_candle)
                    symbols_to_close.append(symbol)
                    if equity <= 0: continue
            
            for symbol in set(symbols_to_close):
                if symbol in open_positions: del open_positions[symbol]

            symbol = prev_candle['Symbol'] 
            if self.config.USE_TIERED_RISK:
                base_risk_pct, max_concurrent_trades = self._get_tiered_risk_params(equity)
            else:
                base_risk_pct, max_concurrent_trades = self.config.BASE_RISK_PER_TRADE_PCT, self.config.MAX_CONCURRENT_TRADES
            
            effective_max_concurrent = max_concurrent_trades
            min_confidence_modifier = 0.0
            if account_health_state == 'Critical':
                effective_max_concurrent = 1
                min_confidence_modifier = 0.1
            
            # Check for trade lockout before attempting to open a new position
            is_locked_out = trade_lockout_until is not None and current_candle['Timestamp'] < trade_lockout_until
            
            if not circuit_breaker_tripped and not is_locked_out and symbol not in open_positions and len(open_positions) < effective_max_concurrent:
                if prev_candle.get('anomaly_score') == -1: continue
                vol_idx = prev_candle.get('market_volatility_index', 0.5)
                if not (self.config.MIN_VOLATILITY_RANK <= vol_idx <= self.config.MAX_VOLATILITY_RANK): continue

                direction, confidence = 0, 0
                adjusted_confidence_threshold = confidence_threshold + min_confidence_modifier

                if is_minirocket_model:
                    lookback = self.config.MINIROCKET_LOOKBACK
                    if i >= lookback:
                        start_idx = i - lookback
                        
                        # Create the 3D sequence from original features specified in the config
                        feature_list = self.config.selected_features
                        sequence_candles = candles[start_idx:i]
                        sequence_data = [[c.get(feat, 0) for feat in feature_list] for c in sequence_candles]
                        sequence_3d = np.expand_dims(np.array(sequence_data), axis=0)

                        # Transform the sequence using the fitted MiniRocket transformer
                        seq_transformed = minirocket_transformer.transform(sequence_3d)

                        # Predict using the full XGBoost pipeline (which includes the scaler)
                        probs = xgb_pipeline.predict_proba(seq_transformed)[0]
                        
                        max_confidence = np.max(probs)
                        if max_confidence >= adjusted_confidence_threshold:
                            pred_class = np.argmax(probs)
                            direction = 1 if pred_class == 2 else -1 if pred_class == 0 else 0
                            confidence = max_confidence
                        prev_candle['prob_short'], prev_candle['prob_hold'], prev_candle['prob_long'] = probs[0], probs[1], probs[2]

                elif is_gnn_model and gnn_edge_index is not None and prev_candle['Timestamp'] in gnn_feature_df.index:
                    gnn_feature_list = self.config.selected_features
                    current_ts = prev_candle['Timestamp']
                    
                    # Construct the feature matrix `x` for all nodes at this timestamp
                    node_features_list = []
                    for s in gnn_symbols:
                        # Extract features for symbol `s` at this timestamp from the pre-pivoted df
                        symbol_features = [gnn_feature_df.at[current_ts, f'{feat}_{s}'] for feat in gnn_feature_list]
                        node_features_list.append(symbol_features)
                    
                    x = torch.tensor(node_features_list, dtype=torch.float)
                    
                    graph_data = Data(x=x, edge_index=gnn_edge_index)

                    with torch.no_grad():
                        out = gnn_model(graph_data)
                        probs_all_nodes = F.softmax(out, dim=1)
                        preds_all_nodes = out.argmax(dim=1)

                    # Extract the prediction for the specific symbol of this candle
                    symbol_index = gnn_symbols.index(symbol)
                    probs = probs_all_nodes[symbol_index].numpy()
                    pred_class = preds_all_nodes[symbol_index].item()

                    max_confidence = np.max(probs)
                    if max_confidence >= adjusted_confidence_threshold:
                        direction = 1 if pred_class == 2 else -1 if pred_class == 0 else 0
                        confidence = max_confidence
                    
                    prev_candle['prob_short'], prev_candle['prob_hold'], prev_candle['prob_long'] = probs[0], probs[1], probs[2]

                elif not self.is_transformer_model: # Fallback to standard XGBoost
                    prev_candle_df = pd.DataFrame([prev_candle])[feature_list].fillna(0)
                    if not prev_candle_df.empty:
                        probs = xgb_pipeline.predict_proba(prev_candle_df)[0]
                        max_confidence = np.max(probs)
                        if max_confidence >= adjusted_confidence_threshold:
                            pred_class = np.argmax(probs)
                            direction = 1 if pred_class == 2 else -1 if pred_class == 0 else 0
                            confidence = max_confidence
                        prev_candle['prob_short'], prev_candle['prob_hold'], prev_candle['prob_long'] = probs[0], probs[1], probs[2]

                if direction != 0:
                    atr = prev_candle.get('ATR',0)
                    if pd.isna(atr) or atr<=1e-9: continue

                    tier_name = 'standard'
                    if confidence >= self.config.CONFIDENCE_TIERS['ultra_high']['min']: tier_name = 'ultra_high'
                    elif confidence >= self.config.CONFIDENCE_TIERS['high']['min']: tier_name = 'high'
                    tier = self.config.CONFIDENCE_TIERS[tier_name]
                    
                    sl_dist = atr * 1.5
                    if sl_dist <= 0: continue

                    risk_modifier = 1.0
                    if last_trade_pnl < 0: risk_modifier *= 0.75
                    if account_health_state == 'Caution': risk_modifier *= 0.5
                    elif account_health_state == 'Critical': risk_modifier *= 0.25
                    
                    risk_per_trade_usd = equity * base_risk_pct * tier['risk_mult'] * risk_modifier
                    risk_per_trade_usd = min(risk_per_trade_usd, self.config.RISK_CAP_PER_TRADE_USD)
                    
                    point_value = self.config.CONTRACT_SIZE * (0.0001 if 'JPY' not in symbol else 0.01)
                    risk_per_lot = sl_dist * point_value
                    if risk_per_lot <= 0: continue
                    
                    lots = risk_per_trade_usd / risk_per_lot
                    
                    lots = round(lots / self.config.LOT_STEP) * self.config.LOT_STEP
                    
                    if lots < self.config.MIN_LOT_SIZE:
                        continue

                    margin_required = (lots * self.config.CONTRACT_SIZE * current_candle['Open']) / self.config.LEVERAGE
                    used_margin = sum(p.get('margin_used', 0) for p in open_positions.values())
                    if (equity - used_margin) < margin_required: continue

                    entry_price_base = current_candle['Open'] 
                    spread_cost, slippage_cost = self._calculate_realistic_costs(prev_candle)
                    entry_price = entry_price_base + ((spread_cost + slippage_cost) * direction)
                    sl_price = entry_price - sl_dist * direction
                    tp_price = entry_price + (sl_dist * tier['rr']) * direction
                    
                    open_positions[symbol] = {
                        'symbol': symbol, 'direction': direction, 'entry_price': entry_price, 
                        'sl': sl_price, 'tp': tp_price, 'confidence': confidence, 'lot_size': lots, 
                        'margin_used': margin_required, 'mfe_price': entry_price, 'mae_price': entry_price
                    }

            day_peak_equity = max(day_peak_equity, equity)

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

def get_macro_context_data(additional_tickers: Optional[List[str]] = None) -> Dict[str, Any]:
    """
    [Phase 1 Implemented] Fetches the latest data for key macroeconomic indicators.
    Now dynamically includes additional tickers suggested by the AI during the initial setup phase.
    """
    logger.info("-> Fetching external macroeconomic context data...")
    macro_context = {}
    # Baseline tickers that are always fetched
    tickers = {
        "VIX": "^VIX",
        "DXY": "DX-Y.NYB",
        "US10Y_YIELD": "^TNX"
    }

    if additional_tickers:
        logger.info(f"  - Including AI-suggested macro tickers: {additional_tickers}")
        for ticker in additional_tickers:
            # Avoid duplicating a ticker if AI suggests one we already have
            if ticker not in tickers.values():
                # Use the ticker itself as the name if it's not a default one
                # A simple sanitization for the key name
                safe_name = ticker.replace('^', '').replace('=X', '').replace('.NYB', '')
                tickers[safe_name] = ticker
    
    for name, ticker in tickers.items():
        try:
            # Download a slightly longer period to ensure we get 5 valid trading days
            data = yf.download(ticker, period="10d", progress=False, auto_adjust=True)

            # Check if data is valid and sufficient
            if data is not None and not data.empty and len(data) >= 6:
                close = data['Close']
                # Handle potential multi-level columns from yfinance
                if isinstance(close, pd.DataFrame):
                    close = close.iloc[:, 0]

                latest_level = close.iloc[-1]
                one_week_ago_level = close.iloc[-6] # 5 trading days before the last one
                
                # Ensure values are plain floats
                if hasattr(one_week_ago_level, "item"): one_week_ago_level = one_week_ago_level.item()
                if hasattr(latest_level, "item"): latest_level = latest_level.item()

                if one_week_ago_level != 0 and pd.notna(one_week_ago_level):
                    week_change_pct = ((latest_level - one_week_ago_level) / one_week_ago_level) * 100
                else:
                    week_change_pct = 0.0
                    
                macro_context[name] = {"level": round(latest_level, 2), "1_week_change_pct": round(week_change_pct, 2)}
            else:
                logger.warning(f"  - Not enough data returned for {name} ({ticker}) to calculate 1-week change.")
                macro_context[name] = {"error": "Insufficient data"}

        except Exception as e:
            logger.error(f"  - Failed to download or process macro data for {name} ({ticker}): {e}")
            macro_context[name] = {"error": str(e)}

    logger.info(f"  - Macro context generated: {macro_context}")
    return macro_context

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
    [Phase 1 Implemented] Initializes the strategy playbook.
    The playbook now only contains descriptions and metadata. The static `selected_features`
    list has been removed, as feature selection is now a dynamic, AI-driven task.
    This function also includes logic to migrate old playbook formats.
    """
    DEFAULT_PLAYBOOK = {
        "ADXMomentum": {
            "description": "[MOMENTUM] A classic momentum strategy that enters when ADX confirms a strong trend and MACD indicates accelerating momentum. Ideal for trending environments. Example features: `ADX`, `MACD_hist`, `momentum_20`, `market_regime`.",
            "lookahead_range": [60, 180], "dd_range": [0.20, 0.35], "complexity": "medium",
            "ideal_regime": ["Strong Trending"], "asset_class_suitability": ["Any"], "ideal_macro_env": ["Any"]
        },
        "ClassicBollingerRSI": {
            "description": "[RANGING] A traditional mean-reversion strategy entering at the outer bands, filtered by low trend strength. Ideal for ranging markets. Example features: `bollinger_bandwidth`, `RSI`, `ADX`, `market_regime`.",
            "lookahead_range": [20, 70], "dd_range": [0.1, 0.2], "complexity": "low",
            "ideal_regime": ["Ranging", "Low Volatility"], "asset_class_suitability": ["Any"], "ideal_macro_env": ["Neutral"]
        },
        "VolatilityExpansionBreakout": {
            "description": "[BREAKOUT] Enters on strong breakouts that occur after a period of low-volatility consolidation (Bollinger Squeeze). Example features: `bollinger_bandwidth`, `ATR`, `market_volatility_index`, `DAILY_ctx_Trend`.",
            "lookahead_range": [70, 140], "dd_range": [0.2, 0.4], "complexity": "medium",
            "ideal_regime": ["Low Volatility", "High Volatility"], "asset_class_suitability": ["Any"], "ideal_macro_env": ["Event-Driven", "Neutral"]
        },
        "ICTMarketStructure": {
            "description": "[PRICE ACTION/INSTITUTIONAL] A methodology focused on identifying liquidity zones and Fair Value Gaps (FVG). Example Features: `fvg_bullish_exists`, `choch_up_signal`, `liquidity_grab_up`, `DAILY_ctx_Trend`.",
            "lookahead_range": [40, 120], "dd_range": [0.2, 0.35], "complexity": "high",
            "ideal_regime": ["Strong Trending", "Weak Trending"], "asset_class_suitability": ["Forex Majors", "Indices"], "ideal_macro_env": ["Neutral", "Risk-On", "Risk-Off"]
        },
        "MeanReversionZScore": {
            "description": "[MEAN REVERSION] Exploits statistical deviations from the mean, entering when RSI reaches an extreme Z-score in a non-trending market. Example Features: `RSI_zscore`, `bollinger_bandwidth`, `stoch_k`, `market_regime`.",
            "lookahead_range": [20, 70], "dd_range": [0.10, 0.25], "complexity": "medium",
            "ideal_regime": ["Ranging", "Low Volatility"], "asset_class_suitability": ["Forex Majors", "Indices"], "ideal_macro_env": ["Any"]
        },
        "GNN_Market_Structure": {
            "description": "[SPECIALIZED] Uses a GNN to model inter-asset correlations for predictive features. Since this is a specialized model, the AI should rely on its internal feature generation.",
            "lookahead_range": [80, 150], "dd_range": [0.15, 0.3], "complexity": "specialized",
            "requires_gnn": True, "ideal_regime": ["Any"], "ideal_macro_env": ["Any"], "asset_class_suitability": ["Forex Majors", "Indices"]
        },
        "Meta_Labeling_Filter": {
            "description": "[SPECIALIZED] Uses a secondary ML filter to improve a simple primary model's signal quality. Example Features: `ADX`, `ATR`, `bollinger_bandwidth`, `H1_ctx_Trend`, `DAILY_ctx_Trend`, `momentum_20`, `relative_performance`.",
            "lookahead_range": [50, 100], "dd_range": [0.1, 0.25], "complexity": "specialized",
            "requires_meta_labeling": True, "ideal_regime": ["Any"], "asset_class_suitability": ["Any"], "ideal_macro_env": ["Any"]
        },
        "GeneticTrendFollower": {
            "description": "[GENERATIVE] Evolves a new trend-following strategy from scratch using genetic programming. The AI will define the building blocks (genes). The evolved rule is then used as a signal for a meta-filter model.",
            "strategy_goal": "trend-following",
            "lookahead_range": [60, 180], "dd_range": [0.20, 0.40], "complexity": "generative",
            "requires_gp": True,
            "ideal_regime": ["Any"], "asset_class_suitability": ["Any"], "ideal_macro_env": ["Any"]
        },
        "GeneticMeanReversion": {
            "description": "[GENERATIVE] Evolves a new mean-reversion strategy from scratch using genetic programming. The AI will define the building blocks (genes). The evolved rule is then used as a signal for a meta-filter model.",
            "strategy_goal": "mean-reversion",
            "lookahead_range": [30, 90], "dd_range": [0.10, 0.25], "complexity": "generative",
            "requires_gp": True,
            "ideal_regime": ["Any"], "asset_class_suitability": ["Any"], "ideal_macro_env": ["Any"]
        }
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
        # Add any new strategies from the default playbook that are missing
        for strategy_name, default_config in DEFAULT_PLAYBOOK.items():
            if strategy_name not in playbook:
                playbook[strategy_name] = default_config
                logger.info(f"  - Adding new strategy to playbook: '{strategy_name}'")
                updated = True
        
        # This loop migrates any older playbook files by removing the now-obsolete static feature lists
        for strategy_name in list(playbook.keys()):
            if 'selected_features' in playbook[strategy_name]:
                logger.info(f"  - Migrating legacy playbook: removing 'selected_features' key from '{strategy_name}'.")
                del playbook[strategy_name]['selected_features']
                updated = True
            if 'features' in playbook[strategy_name]: # For even older versions
                logger.info(f"  - Migrating legacy playbook: removing 'features' key from '{strategy_name}'.")
                del playbook[strategy_name]['features']
                updated = True

        if updated:
            logger.info("Playbook was updated (new strategies added or legacy keys migrated). Saving changes...")
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
                updated = True
        
        # This loop ensures any older playbook file is updated to the new format
        for strategy_name in list(playbook.keys()):
            if 'features' in playbook[strategy_name]:
                logger.info(f"  - Removing legacy 'features' key from '{strategy_name}' in playbook.")
                del playbook[strategy_name]['features']
                updated = True

        if updated:
            logger.info("Playbook was updated (new strategies added or legacy keys removed). Saving changes...")
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

def _generate_cache_metadata(config: ConfigModel, files: List[str], tf_roles: Dict) -> Dict:
    """Generates a dictionary of metadata to validate the feature cache."""
    file_metadata = {}
    for filename in sorted(files):
        file_path = os.path.join(config.BASE_PATH, filename)
        if os.path.exists(file_path):
            stat = os.stat(file_path)
            file_metadata[filename] = {"mtime": stat.st_mtime, "size": stat.st_size}

    # These are the parameters that affect the output of `create_feature_stack`
    param_metadata = {
        'TREND_FILTER_THRESHOLD': config.TREND_FILTER_THRESHOLD,
        'BOLLINGER_PERIOD': config.BOLLINGER_PERIOD,
        'STOCHASTIC_PERIOD': config.STOCHASTIC_PERIOD,
        'HAWKES_KAPPA': config.HAWKES_KAPPA,
        'anomaly_contamination_factor': config.anomaly_contamination_factor,
        'USE_PCA_REDUCTION': config.USE_PCA_REDUCTION,
        'PCA_N_COMPONENTS': config.PCA_N_COMPONENTS,
        'RSI_PERIODS_FOR_PCA': config.RSI_PERIODS_FOR_PCA,
        'tf_roles': tf_roles
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
    
    full_df = None
    if temp_config.USE_FEATURE_CACHING:
        logger.info("-> Feature Caching is ENABLED. Checking for a valid cache...")
        current_metadata = _generate_cache_metadata(temp_config, all_files, tf_roles)
        
        if os.path.exists(temp_config.CACHE_METADATA_PATH) and os.path.exists(temp_config.CACHE_PATH):
            logger.info(f"  - Found existing cache files.")
            try:
                with open(temp_config.CACHE_METADATA_PATH, 'r') as f:
                    saved_metadata = json.load(f)
                
                if current_metadata == saved_metadata:
                    logger.info("  - Cache is VALID. Loading features from cache...")
                    start_time = time.time()
                    full_df = pd.read_parquet(temp_config.CACHE_PATH)
                    end_time = time.time()
                    logger.info(f"[SUCCESS] Loaded features from cache in {end_time - start_time:.2f} seconds. Shape: {full_df.shape}")
                else:
                    logger.warning("  - Cache is STALE (input files or parameters changed). Re-engineering features...")
            except (json.JSONDecodeError, IOError, Exception) as e:
                logger.warning(f"  - Could not read or validate cache. Re-engineering features. Error: {e}")
        else:
            logger.info("  - No valid cache found. Engineering features...")

    if full_df is None:
        fe = FeatureEngineer(temp_config, tf_roles, playbook)
        full_df = fe.create_feature_stack(data_by_tf)
        if full_df.empty:
            logger.critical("Feature engineering resulted in an empty dataframe. Exiting.")
            return

        if temp_config.USE_FEATURE_CACHING:
            logger.info("  - Saving newly engineered features to cache...")
            try:
                os.makedirs(os.path.dirname(temp_config.CACHE_PATH), exist_ok=True)
                full_df.to_parquet(temp_config.CACHE_PATH)
                current_metadata = _generate_cache_metadata(temp_config, all_files, tf_roles)
                with open(temp_config.CACHE_METADATA_PATH, 'w') as f:
                    json.dump(current_metadata, f, indent=4)
                logger.info(f"  - Features and metadata saved to cache: {temp_config.CACHE_PATH}")
            except Exception as e:
                logger.error(f"  - Failed to save features to cache. Error: {e}")

    macro_context = get_macro_context_data()
    regime_summary = train_and_diagnose_regime(full_df, os.path.join(temp_config.BASE_PATH, "Results"))
    
    logger.info("  - Calculating asset correlation matrix for AI context...")
    pivot_df = full_df.pivot_table(index=full_df.index, columns='Symbol', values='Close', aggfunc='last').ffill().dropna(how='all', axis=1)
    correlation_summary_for_ai = pivot_df.corr().to_json(indent=2) if pivot_df.shape[1] > 1 else "{}"

    summary_df = full_df.reset_index()
    assets = summary_df['Symbol'].unique().tolist()
    data_summary = {
        'assets_detected': assets,
        'time_range': {'start': summary_df['Timestamp'].min().isoformat(), 'end': summary_df['Timestamp'].max().isoformat()},
        'timeframes_used': tf_roles,
        'asset_statistics': {asset: {'avg_atr': round(full_df[full_df['Symbol'] == asset]['ATR'].mean(), 5), 'avg_adx': round(full_df[full_df['Symbol'] == asset]['ADX'].mean(), 2)} for asset in assets}
    }

    script_name = os.path.basename(__file__) if '__file__' in locals() else fallback_config["REPORT_LABEL"]
    version_label = script_name.replace(".py", "")
    health_report, _ = perform_strategic_review(framework_history, fallback_config['DIRECTIVES_FILE_PATH'])

    regime_champions = {}
    if os.path.exists(temp_config.REGIME_CHAMPIONS_FILE_PATH):
        try:
            with open(temp_config.REGIME_CHAMPIONS_FILE_PATH, 'r') as f: regime_champions = json.load(f)
        except (json.JSONDecodeError, IOError): logger.warning("Could not read regime champions file.")
    
    ai_setup = api_timer.call(gemini_analyzer.get_initial_run_setup, version_label, nickname_ledger, framework_history, playbook, health_report, directives, data_summary, regime_summary['current_diagnosed_regime'], regime_champions, correlation_summary_for_ai, macro_context)
    if not ai_setup:
        logger.critical("AI-driven setup failed because the response was empty or invalid. Check logs for details. Exiting.")
        return

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
    
    if config.USE_PCA_REDUCTION:
        # Define the names of the features that would have been created
        pca_features_to_add = [f'RSI_PCA_{i+1}' for i in range(config.PCA_N_COMPONENTS)]
        
        added_count = 0
        for feat in pca_features_to_add:
            # Check if the feature exists in the dataframe and is not already in the selection
            if feat in full_df.columns and feat not in config.selected_features:
                config.selected_features.append(feat)
                added_count += 1
        
        if added_count > 0:
            logger.info(f"FIX APPLIED: Dynamically added {added_count} PCA features to the model's feature list.")
            logger.debug(f"Updated feature list includes: {config.selected_features}")
    
    file_handler = RotatingFileHandler(config.LOG_FILE_PATH, maxBytes=5*1024*1024, backupCount=2)
    file_handler.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
    logger.addHandler(file_handler)
    logger.info(f"--- Run Initialized: {config.nickname} | Strategy: {config.strategy_name} ---")

    all_available_features = [c for c in full_df.columns if c not in ['Open','High','Low','Close','RealVolume','Symbol','Timestamp','primary_model_signal','target']]
    
    if not hasattr(config, 'selected_features') or not config.selected_features:
        logger.info("-> No features pre-selected. Using example features from playbook description as fallback.")
        strategy_details_desc = playbook.get(config.strategy_name, {}).get("description", "")
        found_features = re.findall(r'`([a-zA-Z0-9_]+)`', strategy_details_desc)
        if found_features:
            config.selected_features = [f for f in found_features if f in all_available_features]
            logger.info(f"  - Using features: {config.selected_features}")
        else:
            logger.critical(f"Could not determine features for strategy '{config.strategy_name}'. Cannot proceed.")
            return

    train_window, forward_gap = pd.to_timedelta(config.TRAINING_WINDOW), pd.to_timedelta(config.FORWARD_TEST_GAP)
    test_start_date = full_df.index.min() + train_window + forward_gap
    retraining_dates = pd.date_range(start=test_start_date, end=full_df.index.max(), freq=_sanitize_frequency_string(config.RETRAINING_FREQUENCY))

    if retraining_dates.empty:
        logger.critical("Cannot proceed: No valid retraining dates could be determined. The total data length may be too short for the specified training window.")
        return

    aggregated_trades, aggregated_equity_curve = pd.DataFrame(), pd.Series([config.INITIAL_CAPITAL])
    in_run_historical_cycles, aggregated_daily_dd_reports = [], []
    shap_history, all_optuna_trials = defaultdict(list), []
    last_equity, quarantine_list = config.INITIAL_CAPITAL, []
    run_peak_equity = config.INITIAL_CAPITAL
    
    # State tracking variables
    consecutive_wins, consecutive_losses = 0, 0
    drawdown_control_cycles = 0
    trade_lockout_until = None
    
    cycle_num, cycle_retry_count = 0, 0
    while cycle_num < len(retraining_dates):
        period_start_date = retraining_dates[cycle_num]
        
        # --- Strategic Pivot Logic (Group 4) ---
        if drawdown_control_cycles >= 2:
            logger.warning(f"! STRATEGIC PIVOT TRIGGERED ! Strategy '{config.strategy_name}' failed for two consecutive cycles.")
            last_failed_strategy = config.strategy_name
            quarantine_list.append(last_failed_strategy)
            logger.info(f"  - Current quarantine list: {quarantine_list}")
            
            intervention_suggestions = api_timer.call(gemini_analyzer.propose_strategic_intervention,
                failure_history=in_run_historical_cycles[-2:],
                playbook=playbook,
                last_failed_strategy=last_failed_strategy,
                quarantine_list=quarantine_list
            )
            
            # **CORRECTED LOGIC**: Handle the AI's choice to invent a new strategy
            if intervention_suggestions and intervention_suggestions.get('action') == "invent_new_strategy":
                logger.info("  - AI requested to invent a new strategy. Engaging generative playbook evolution...")
                new_strategy_json = api_timer.call(gemini_analyzer.propose_new_playbook_strategy,
                                                   failed_strategy_name=last_failed_strategy,
                                                   playbook=playbook,
                                                   framework_history=framework_history)
                if new_strategy_json:
                    playbook.update(new_strategy_json)
                    try:
                        with open(config.PLAYBOOK_FILE_PATH, 'w') as f:
                            json.dump(playbook, f, indent=4)
                        logger.info("  - Playbook successfully updated with new AI-generated strategy.")
                        # Set the newly created strategy as the one to use
                        new_strategy_name = next(iter(new_strategy_json))
                        intervention_suggestions['strategy_name'] = new_strategy_name
                        intervention_suggestions['selected_features'] = [] # Force AI to select features for the new strategy
                    except IOError as e:
                        logger.error(f"  - Failed to save updated playbook: {e}")
                else:
                    logger.error("  - AI failed to generate a valid new strategy. Pivot will use a random strategy instead.")
                    intervention_suggestions = {} # Clear suggestions to trigger fallback

            if intervention_suggestions and intervention_suggestions.get('strategy_name'):
                logger.info("  - AI proposed a new strategy. Applying changes for pivot.")
                config = ConfigModel(**{**config.model_dump(mode='json'), **_sanitize_ai_suggestions(intervention_suggestions)})
                logger.info(f"  - New strategy selected: '{config.strategy_name}'. Features: {config.selected_features}")
            else:
                logger.critical("  - Strategic Pivot failed: AI did not return a valid new strategy. Aborting run.")
                break
                
            config.operating_state = OperatingState.CONSERVATIVE_BASELINE
            drawdown_control_cycles = 0
            logger.info(f"! PIVOT COMPLETE ! Resetting to {config.operating_state.value} with new strategy.")

        config = _apply_operating_state_rules(config)
        
        logger.info(f"\n--- Starting Cycle [{cycle_num + 1}/{len(retraining_dates)}] in state '{config.operating_state.value}' ---")
        cycle_start_time = time.time()
        
        train_end = period_start_date - forward_gap
        train_start = train_end - pd.to_timedelta(config.TRAINING_WINDOW)
        test_end = period_start_date + pd.tseries.frequencies.to_offset(_sanitize_frequency_string(config.RETRAINING_FREQUENCY))

        df_train_raw = full_df.loc[train_start:train_end].copy()
        df_test = full_df.loc[period_start_date:min(test_end, full_df.index.max())].copy()
        
        if df_train_raw.empty or df_test.empty:
            logger.warning(f"  - Skipping cycle {cycle_num + 1}: Not enough data in training or testing period.")
            cycle_num += 1
            continue

        strategy_details = playbook.get(config.strategy_name, {})
        fe = FeatureEngineer(config, tf_roles, playbook)
        
        labeling_method = getattr(config, 'LABELING_METHOD', 'standard')
        label_func = getattr(fe, f"label_{labeling_method}", fe.label_standard)
        df_train_labeled = label_func(df_train_raw, config.LOOKAHEAD_CANDLES)

        pipeline, threshold, f1_score_val = None, None, -1.0
        
        if not check_label_quality(df_train_labeled, config.LABEL_MIN_EVENT_PCT):
            logger.critical(f"!! MODEL TRAINING SKIPPED !! Un-trainable labels generated.")
        else:
            trainer = ModelTrainer(config, gemini_analyzer)
            train_result = trainer.train(df_train_labeled, config.selected_features, strategy_details)
            
            if train_result:
                pipeline, threshold, f1_score_val = train_result
                if f1_score_val < getattr(config, 'MIN_F1_SCORE_GATE', MIN_F1_SCORE_GATE):
                    logger.critical(f"!! MODEL QUALITY GATE FAILED !! F1 Score ({f1_score_val:.3f}) < Gate ({getattr(config, 'MIN_F1_SCORE_GATE', MIN_F1_SCORE_GATE)}).")
                    pipeline = None
            else:
                logger.critical(f"!! MODEL TRAINING FAILED !!")

        if pipeline:
            cycle_retry_count = 0 
            state_modifier = config.STATE_BASED_CONFIG[config.operating_state]["confidence_gate_modifier"]
            final_threshold = threshold * state_modifier
            logger.info(f"  - Original Threshold: {threshold:.3f}, State Modifier: {state_modifier:.2f} -> Final Threshold: {final_threshold:.3f}")

            backtester = Backtester(config)
            trades, equity_curve, breaker_tripped, breaker_context, daily_dd_report = backtester.run_backtest_chunk(df_test, pipeline, final_threshold, last_equity, strategy_details, run_peak_equity, config.selected_features, trade_lockout_until)
            trade_lockout_until = None
            aggregated_daily_dd_reports.append(daily_dd_report)
            cycle_status_msg = "Completed"
        else:
            trades, equity_curve, breaker_tripped, breaker_context = pd.DataFrame(), pd.Series([last_equity]), False, None
            cycle_status_msg = "Training Failed"
            cycle_retry_count += 1
        
        cycle_pnl = equity_curve.iloc[-1] - last_equity if not equity_curve.empty else 0.0
        
        if not trades.empty:
            last_trade_pnl = trades.iloc[-1]['PNL']
            if last_trade_pnl > 0:
                consecutive_wins += 1
                consecutive_losses = 0
            elif last_trade_pnl < 0:
                consecutive_losses += 1
                consecutive_wins = 0
        
        trade_summary = {}
        if not trades.empty:
            losing_trades = trades[trades['PNL'] < 0]
            if not losing_trades.empty:
                trade_summary['avg_mae_loss'] = losing_trades['MAE'].mean()
                trade_summary['avg_mfe_loss'] = losing_trades['MFE'].mean()
        
        cycle_result = {
            "StartDate": period_start_date.date().isoformat(), "EndDate": test_end.date().isoformat(),
            "NumTrades": len(trades), "PNL": round(cycle_pnl, 2),
            "Status": "Circuit Breaker" if breaker_tripped else cycle_status_msg,
            "F1_Score": round(f1_score_val, 4), "trade_summary": trade_summary,
            "State": config.operating_state.value
        }
        if breaker_tripped:
            cycle_result["BreakerContext"] = breaker_context

        in_run_historical_cycles.append(cycle_result)

        if not trades.empty:
            aggregated_trades = pd.concat([aggregated_trades, trades], ignore_index=True)
            new_equity_curve = pd.concat([aggregated_equity_curve.iloc[:-1], equity_curve], ignore_index=True)
            aggregated_equity_curve = new_equity_curve
            last_equity = equity_curve.iloc[-1]
            if last_equity > run_peak_equity:
                logger.info(f"** NEW EQUITY HIGH REACHED: ${last_equity:,.2f} **")
                run_peak_equity = last_equity
        
        previous_state = config.operating_state
        
        if breaker_tripped or consecutive_losses >= 3:
            config.operating_state = OperatingState.DRAWDOWN_CONTROL
            if previous_state != OperatingState.DRAWDOWN_CONTROL:
                logger.info(f"! STATE TRANSITION ! Triggered {config.operating_state.value} due to losses. Consecutive Losses: {consecutive_losses}, Breaker Tripped: {breaker_tripped}")
                drawdown_control_cycles = 1
                if not trades.empty:
                    last_trade_time = pd.to_datetime(trades.iloc[-1]['ExecTime'])
                    trade_lockout_until = last_trade_time.normalize() + pd.Timedelta(days=1)
                    logger.info(f"  - Engaging 1-day trade lockout until: {trade_lockout_until.date()}")
            else:
                drawdown_control_cycles += 1
        
        elif last_equity >= run_peak_equity or consecutive_wins >= 4 or (cycle_pnl > 0 and (cycle_pnl / last_equity) > 0.05):
            config.operating_state = OperatingState.AGGRESSIVE_EXPANSION
            if previous_state != OperatingState.AGGRESSIVE_EXPANSION:
                logger.info(f"! STATE TRANSITION ! Triggered {config.operating_state.value} due to strong performance.")
            drawdown_control_cycles = 0

        else:
            config.operating_state = OperatingState.CONSERVATIVE_BASELINE
            if previous_state != OperatingState.CONSERVATIVE_BASELINE:
                logger.info(f"! STATE TRANSITION ! Reverting to {config.operating_state.value}.")
            drawdown_control_cycles = 0

        if cycle_num < len(retraining_dates) - 1:
            macro_context = get_macro_context_data()
            suggested_params = api_timer.call(
                gemini_analyzer.analyze_cycle_and_suggest_changes,
                historical_results=in_run_historical_cycles, framework_history=framework_history,
                available_features=all_available_features, strategy_details=config.model_dump(),
                cycle_status=cycle_status_msg, shap_history=shap_history, all_optuna_trials=all_optuna_trials,
                cycle_start_date=period_start_date.isoformat(), cycle_end_date=min(test_end, full_df.index.max()).isoformat(),
                correlation_summary_for_ai=correlation_summary_for_ai, macro_context=macro_context,
                account_health_state="Normal", overall_drawdown_pct=0.0
            )
            if suggested_params:
                config = ConfigModel(**{**config.model_dump(mode='json'), **_sanitize_ai_suggestions(suggested_params)})
                logger.info(f"  - AI suggestions applied for next cycle. Notes: {suggested_params.get('analysis_notes', 'N/A')}")

        cycle_num += 1
        logger.info(f"--- Cycle complete. PNL: ${cycle_pnl:,.2f} | Final Equity: ${last_equity:,.2f} | Time: {time.time() - cycle_start_time:.2f}s ---")

    pa = PerformanceAnalyzer(config)
    last_class_report = trainer.classification_report_str if 'trainer' in locals() and hasattr(trainer, 'classification_report_str') else "N/A"
    final_metrics = pa.generate_full_report(aggregated_trades, aggregated_equity_curve, in_run_historical_cycles, pd.DataFrame.from_dict(shap_history, orient='index').mean(axis=1).sort_values(ascending=False).to_frame('SHAP_Importance'), framework_history, aggregated_daily_dd_reports, last_class_report)
    run_summary = {"script_version": config.REPORT_LABEL, "nickname": config.nickname, "strategy_name": config.strategy_name, "run_start_ts": config.run_timestamp, "final_params": config.model_dump(mode='json'), "run_end_ts": datetime.now().strftime("%Y%m%d-%H%M%S"), "final_metrics": final_metrics, "cycle_details": in_run_historical_cycles}
    save_run_to_memory(config, run_summary, framework_history, regime_summary['current_diagnosed_regime'])
    logger.removeHandler(file_handler); file_handler.close()

def main():
    """Main entry point for the trading framework."""
    # This initial print is for immediate feedback, with flush=True to guarantee it appears first.
    print(f"--- ML Trading Framework V{VERSION} Initializing ---", flush=True)

    # CRITICAL FIX: Configure the logger at the very beginning of execution.
    setup_logging()
    
    logger.info("Framework entry point reached. Starting main execution loop.")

    CONTINUOUS_RUN_HOURS = 0; MAX_RUNS = 1
    fallback_config={
        "BASE_PATH": os.getcwd(), "REPORT_LABEL": f"ML_Framework_V{VERSION}",
        "strategy_name": "Meta_Labeling_Filter", "INITIAL_CAPITAL": 1000.0,
        "COMMISSION_PER_LOT": 3.5,
        "CONFIDENCE_TIERS": {
            'ultra_high': {'min': 0.80, 'risk_mult': 1.2, 'rr': 2.5},
            'high':       {'min': 0.70, 'risk_mult': 1.0, 'rr': 2.0},
            'standard':   {'min': 0.60, 'risk_mult': 0.8, 'rr': 1.5}
        },
        "BASE_RISK_PER_TRADE_PCT": 0.01,"RISK_CAP_PER_TRADE_USD": 500.0,
        "OPTUNA_TRIALS": 30,
        "TRAINING_WINDOW": '365D', "RETRAINING_FREQUENCY": '90D',
        "FORWARD_TEST_GAP": "1D", "LOOKAHEAD_CANDLES": 150, 
        
        # Lower the threshold to make the 'optimal_entry' labeler less strict
        "TREND_FILTER_THRESHOLD": 22.0,

        "BOLLINGER_PERIOD": 20, "STOCHASTIC_PERIOD": 14, "CALCULATE_SHAP_VALUES": True,
        "MAX_DD_PER_CYCLE": 0.25,"GNN_EMBEDDING_DIM": 8, "GNN_EPOCHS": 50,
        "MIN_VOLATILITY_RANK": 0.1, "MAX_VOLATILITY_RANK": 0.9,
        "selected_features": [],
        "MAX_CONCURRENT_TRADES": 3,
        "USE_PARTIAL_PROFIT": False,
        "PARTIAL_PROFIT_TRIGGER_R": 1.5,
        "PARTIAL_PROFIT_TAKE_PCT": 0.5,
        "MAX_TRAINING_RETRIES_PER_CYCLE": 3,
        "anomaly_contamination_factor": 0.01,
        "LABEL_MIN_RETURN_PCT": 0.004,
        "LABEL_MIN_EVENT_PCT": 0.02,
        "USE_TIERED_RISK": True,
        "RISK_PROFILE": "Medium",
        "TIERED_RISK_CONFIG": {
            2000:  {'Low': {'risk_pct': 0.01, 'pairs': 1}, 'Medium': {'risk_pct': 0.01, 'pairs': 1}, 'High': {'risk_pct': 0.01, 'pairs': 1}},
            5000:  {'Low': {'risk_pct': 0.008, 'pairs': 1}, 'Medium': {'risk_pct': 0.012, 'pairs': 1}, 'High': {'risk_pct': 0.012, 'pairs': 2}},
            10000: {'Low': {'risk_pct': 0.006, 'pairs': 2}, 'Medium': {'risk_pct': 0.008, 'pairs': 2}, 'High': {'risk_pct': 0.01, 'pairs': 2}},
            15000: {'Low': {'risk_pct': 0.007, 'pairs': 2}, 'Medium': {'risk_pct': 0.009, 'pairs': 2}, 'High': {'risk_pct': 0.012, 'pairs': 2}},
            25000: {'Low': {'risk_pct': 0.008, 'pairs': 2}, 'Medium': {'risk_pct': 0.012, 'pairs': 2}, 'High': {'risk_pct': 0.016, 'pairs': 2}},
            50000: {'Low': {'risk_pct': 0.008, 'pairs': 3}, 'Medium': {'risk_pct': 0.012, 'pairs': 3}, 'High': {'risk_pct': 0.016, 'pairs': 3}},
            100000:{'Low': {'risk_pct': 0.007, 'pairs': 4}, 'Medium': {'risk_pct': 0.01, 'pairs': 4}, 'High': {'risk_pct': 0.014, 'pairs': 4}},
            9e9:   {'Low': {'risk_pct': 0.005, 'pairs': 6}, 'Medium': {'risk_pct': 0.0075,'pairs': 6}, 'High': {'risk_pct': 0.01, 'pairs': 6}}
        },
        "CONTRACT_SIZE": 100000.0,
        "LEVERAGE": 30,
        "MIN_LOT_SIZE": 0.01,
        "LOT_STEP": 0.01
    }

    fallback_config["DIRECTIVES_FILE_PATH"] = os.path.join(fallback_config["BASE_PATH"], "Results", "framework_directives.json")
    api_interval_seconds = 61
    run_count = 0; script_start_time = datetime.now(); is_continuous = CONTINUOUS_RUN_HOURS > 0 or MAX_RUNS > 1
    bootstrap_config = ConfigModel(**fallback_config, run_timestamp="init", nickname="init")
    
    results_dir = os.path.join(bootstrap_config.BASE_PATH, "Results")
    os.makedirs(results_dir, exist_ok=True)
    playbook_file_path = os.path.join(results_dir, "strategy_playbook.json")
    playbook = initialize_playbook(playbook_file_path)

    while True:
        run_count += 1
        if is_continuous: logger.info(f"\n{'='*30} STARTING DAEMON RUN {run_count} {'='*30}\n")
        else: logger.info(f"\n{'='*30} STARTING SINGLE RUN {'='*30}\n")
        flush_loggers() # Ensure header is printed

        nickname_ledger = load_nickname_ledger(bootstrap_config.NICKNAME_LEDGER_PATH)
        framework_history = load_memory(bootstrap_config.CHAMPION_FILE_PATH, bootstrap_config.HISTORY_FILE_PATH)
        directives = []
        if os.path.exists(bootstrap_config.DIRECTIVES_FILE_PATH):
            try:
                with open(bootstrap_config.DIRECTIVES_FILE_PATH, 'r') as f: directives = json.load(f)
                if directives: logger.info(f"Loaded {len(directives)} directive(s) for this run.")
            except (json.JSONDecodeError, IOError) as e: logger.error(f"Could not load directives file: {e}")
        
        flush_loggers() # Flush after initial setup logs

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

# End_To_End_Advanced_ML_Trading_Framework_PRO_V210_Linux.py