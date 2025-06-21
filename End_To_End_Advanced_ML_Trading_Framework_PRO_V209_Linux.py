# End_To_End_Advanced_ML_Trading_Framework_PRO_V209_Linux.py
#
# V209 UPDATE (Performance & Efficiency):
#   1. ADDED (Feature Engineering Cache): The framework now saves the results of the
#      computationally expensive feature engineering stage (Stage 2) to a `.parquet` file.
#   2. ADDED (Cache Validation): On subsequent runs, the script creates a metadata "fingerprint"
#      of the input data files (modification times, sizes) and the core feature generation
#      parameters from the configuration.
#   3. ADDED (Automatic Cache Loading): If a cache file exists and its metadata fingerprint
#      matches the current run's state, the framework will skip the entire feature engineering
#      process and load the processed data directly from the cache, dramatically reducing
#      startup time for repeated runs on the same data.
#   4. ADDED (Configuration Switch): A `USE_FEATURE_CACHING` boolean has been added to the
#      main configuration to easily enable or disable this new behavior.
#   5. ADDED (Early Intervention Controller): Implemented a sophisticated three-phase intervention system
#      that triggers after two consecutive training failures within a single cycle.
#   6. ADDED (Heuristic Pre-Analysis): The system now performs a hard-coded analysis on the failure data
#      to diagnose the likely root cause (e.g., 'High Profitability / Low Accuracy').
#   7. ADDED (AI Adjudication): The results of the pre-analysis are passed to a new, specialized Gemini
#      prompt, which acts as the final adjudicator to decide on the best course of action
#      (adjust metrics, redefine labels, or switch strategy).
#   8. ADDED (Flexible Configuration): The entire early intervention feature is controlled by a new
#      `EarlyInterventionConfig` block within the main `ConfigModel`, allowing it to be
#      tuned or disabled easily.
#
# V208 UPDATE (Dynamic Realism & Contextual Awareness):
#   1. ADDED (Dynamic Broker Simulation): The framework now uses a grounded search via the Gemini API during
#      the initial run setup. It searches for typical, realistic ECN spreads and commissions based on the
#      specific assets detected in the user's data files. This replaces the previous hardcoded defaults,
#      making the simulation environment more accurate and adaptable out-of-the-box.
#   2. IMPROVED (Cycle-Specific Macro Context): The external macro-economic context (VIX, DXY, etc.) is now
#      re-fetched before the start of each walk-forward cycle analysis. This ensures the AI's cycle-to-cycle
#      strategic adjustments are based on fresh, relevant market conditions rather than stale data from the
#      beginning of the run, significantly improving the relevance of its adaptations.
#
# --- SCRIPT VERSION ---
VERSION = "209"
# ---------------------

import os
import re
import json
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
from sklearn.metrics import f1_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import RobustScaler, MinMaxScaler, StandardScaler
from sklearn.utils.class_weight import compute_class_weight
from pydantic import BaseModel, DirectoryPath, confloat, conint, Field, ValidationError
from sklearn.ensemble import IsolationForest
from sklearn.decomposition import PCA
import yfinance as yf
from hurst import compute_Hc

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

def setup_logging() -> logging.Logger:
    if logger.hasHandlers():
        logger.handlers.clear()
    logger.setLevel(logging.DEBUG)
    ch = logging.StreamHandler(sys.stdout)
    ch.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    ch.setFormatter(formatter)
    logger.addHandler(ch)
    if GNN_AVAILABLE:
        logger.info("PyTorch and PyG loaded successfully. GNN module is available.")
    else:
        logger.warning("PyTorch or PyTorch Geometric not found. GNN-based strategies will be unavailable.")
    return logger

logger = setup_logging()
optuna.logging.set_verbosity(optuna.logging.WARNING)
# --- END DIAGNOSTICS & LOGGING ---


warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', category=UserWarning)
warnings.filterwarnings('ignore', category=pd.errors.PerformanceWarning)

# =============================================================================
# 3. CONFIGURATION & VALIDATION
# =============================================================================

class ConfigModel(BaseModel):
    """
    The central configuration model for the trading framework.
    It holds all parameters that define a run, from data paths and capital
    to risk management, AI behavior, and backtesting realism settings.
    """
    # --- Core Run & Capital Parameters ---
    BASE_PATH: DirectoryPath
    REPORT_LABEL: str
    INITIAL_CAPITAL: confloat(gt=0)
    
    # --- AI & Optimization Parameters ---
    OPTUNA_TRIALS: conint(gt=0)
    MAX_TRAINING_RETRIES_PER_CYCLE: conint(ge=0) = 3
    CALCULATE_SHAP_VALUES: bool = True
    
    # --- Dynamic Labeling & Trade Definition (NEW) ---
    # These parameters are now controlled by the AI to redefine trades.
    TP_ATR_MULTIPLIER: confloat(gt=0.5, le=10.0) = 2.0  # Default Take-Profit in ATR multiples
    SL_ATR_MULTIPLIER: confloat(gt=0.5, le=10.0) = 1.5  # Default Stop-Loss in ATR multiples
    LOOKAHEAD_CANDLES: conint(gt=0)
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
    CONFIDENCE_TIERS: Dict[str, Dict[str, Any]]
    USE_TP_LADDER: bool = True
    TP_LADDER_LEVELS_PCT: List[confloat(gt=0, lt=1)] = Field(default_factory=lambda: [0.25, 0.25, 0.25, 0.25])
    TP_LADDER_RISK_MULTIPLIERS: List[confloat(gt=0)] = Field(default_factory=lambda: [1.0, 2.0, 3.0, 4.0])
    USE_TIERED_RISK: bool = False # Note: Can conflict with CONFIDENCE_TIERS if both active
    RISK_PROFILE: str = 'Medium' # Options: 'Low', 'Medium', 'High'
    TIERED_RISK_CONFIG: Dict[int, Dict[str, Dict[str, Union[float, int]]]] = {}

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
        'NZDJPY':  {'normal_pips': 2.0, 'volatile_pips': 6.5}
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
        
        # V209: Add cache paths
        cache_dir = os.path.join(self.BASE_PATH, "Cache")
        # The cache directory is created when needed, not necessarily here,
        # to avoid creating it for runs where caching is disabled or not used.
            
        self.CACHE_PATH = os.path.join(cache_dir, "feature_cache.parquet")
        self.CACHE_METADATA_PATH = os.path.join(cache_dir, "feature_cache_metadata.json")
        
# =============================================================================
# 3. GEMINI AI ANALYZER & API TIMER
# =============================================================================
class APITimer:
    """Manages the timing of API calls to ensure a minimum interval between them."""
    def __init__(self, interval_seconds: int = 300):
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

# End_To_End_Advanced_ML_Trading_Framework_PRO_V208_Linux.py

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

        payload = {"contents": [{"parts": [{"text": prompt}]}]}
        sanitized_payload = self._sanitize_dict(payload)

        models_to_try = [self.primary_model, self.backup_model]
        retry_delays = [5, 15, 30]

        for model in models_to_try:
            logger.info(f"Attempting to call Gemini API with model: {model}")
            api_url = f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent?key={self.api_key}"

            for attempt, delay in enumerate([0] + retry_delays):
                if delay > 0:
                    logger.warning(f"API connection failed. Retrying in {delay} seconds... (Attempt {attempt}/{len(retry_delays)})")
                    flush_loggers()
                    time.sleep(delay)

                try:
                    response = requests.post(api_url, headers=self.headers, data=json.dumps(sanitized_payload), timeout=120)
                    response.raise_for_status()

                    result = response.json()
                    if "candidates" in result and result["candidates"] and "content" in result["candidates"][0] and "parts" in result["candidates"][0]["content"]:
                        logger.info(f"Successfully received response from model: {model}")
                        return result["candidates"][0]["content"]["parts"][0]["text"]
                    else:
                        logger.error(f"Invalid Gemini response structure from {model}: {result}")
                        continue

                except requests.exceptions.RequestException as e:
                    logger.error(f"Gemini API request failed for model {model} on attempt {attempt + 1}: {e}")
                    if attempt == len(retry_delays):
                         logger.critical(f"All retries for model {model} failed.")
                except json.JSONDecodeError as e:
                    logger.error(f"Failed to decode Gemini response JSON from {model}: {e} - Response: {response.text}")
                    continue
                except (KeyError, IndexError) as e:
                    logger.error(f"Failed to extract text from Gemini response from {model}: {e} - Response: {response.text}")
                    continue

            logger.warning(f"Failed to get a response from model {model} after all retries.")

        logger.critical("API connection failed for all primary and backup models after all retries. Stopping.")
        return "{}"

    def _extract_json_from_response(self, response_text: str) -> Dict:
        if response_text.strip().lower() == 'null':
            return {}

        try:
            match = re.search(r"```json\s*(.*?)\s*```", response_text, re.DOTALL)
            json_text = match.group(1) if match else response_text

            if json_text.strip().lower() == 'null':
                return {}

            suggestions = json.loads(json_text.strip())

            if not isinstance(suggestions, dict):
                 logger.error(f"Parsed JSON is not a dictionary. Response text: {response_text}")
                 return {}

            if 'current_params' in suggestions and isinstance(suggestions.get('current_params'), dict):
                nested_params = suggestions.pop('current_params')
                suggestions.update(nested_params)
            return suggestions
        except (json.JSONDecodeError, AttributeError) as e:
            logger.error(f"Could not parse JSON from response: {e}\nResponse text: {response_text}")
            return {}

    def get_initial_run_setup(self, script_version: str, ledger: Dict, memory: Dict, playbook: Dict, health_report: Dict, directives: List[Dict], data_summary: Dict, diagnosed_regime: str, regime_champions: Dict, correlation_summary_for_ai: str, macro_context: Dict) -> Dict:
        if not self.api_key_valid:
            logger.warning("No API key. Skipping AI-driven setup and using default config.")
            return {}

        logger.info("-> Performing Initial AI Analysis & Setup (Grounded Search with Correlation Context)...")
        asset_list = ", ".join(data_summary.get('assets_detected', []))

        # --- FIX: Made the SPREAD_CONFIG format instruction extremely explicit ---
        task_prompt = (
            "**YOUR TASK: Perform a grounded analysis to create the complete initial run configuration. This involves three main steps.**\n\n"
            "**STEP 1: DYNAMIC BROKER SIMULATION (Grounded Search)**\n"
            f"   - The assets being traded are: **{asset_list}**. \n"
            "   - **Action:** Use Google Search to find typical trading costs for these assets on a retail **ECN/Raw Spread** account.\n"
            "   - **Action:** In your JSON response, populate `COMMISSION_PER_LOT` and update the `SPREAD_CONFIG` dictionary.\n"
            "       - For `SPREAD_CONFIG`, you **MUST** return a nested dictionary for each asset.\n"
            "       - The required format is: `\"symbol\": {\"normal_pips\": <value>, \"volatile_pips\": <value>}`.\n"
            "       - `volatile_pips` should be roughly 2.5x to 4x the `normal_pips`.\n"
            "       - **Example for EURUSD**: `\"EURUSD\": {\"normal_pips\": 0.7, \"volatile_pips\": 2.1}`.\n"
            "       - **Crucially**, you must also include the `\"default\"` key with its own nested dictionary.\n\n"
            "**STEP 2: STRATEGY SELECTION (Grounded Search & Context Synthesis)**\n"
            "   - **Synthesize Internal & External Context:** Analyze `MACROECONOMIC CONTEXT`, `MARKET DATA SUMMARY`, and `ASSET CORRELATION SUMMARY`.\n"
            "   - **Grounded Calendar Check:** Use Google Search for the economic calendar for the next 5 trading days, focusing on high-impact events (FOMC, CPI, NFP) for all currencies in the correlated cluster.\n"
            "   - **Decide on a Strategy:**\n"
            "       - **High-Risk Environment:** If macro context is high-risk (high VIX) or a major event is imminent, select a volatility or risk-management strategy (e.g., `VolatilityExpansionBreakout`, `PanicFade`).\n"
            "       - **Calm/Clear Environment:** Otherwise, use the `diagnosed_regime` to choose the best strategy from the playbook.\n\n"
            "**STEP 3: CONFIGURATION & NICKNAME**\n"
            "   - Provide the full configuration (`strategy_name`, `selected_features`, etc.) in the JSON response.\n"
            "   - Handle nickname generation as per the rules."
        )
        # --- END FIX ---
        
        complexity_prompt = (
            "**STRATEGY COMPLEXITY PREFERENCE:**\n"
            "- The playbook now contains a `complexity` field ('low', 'medium', 'high', 'specialized').\n"
            "- Your primary goal is to establish a stable, profitable baseline. Therefore, you should **strongly prefer strategies with 'low' or 'medium' complexity**.\n"
            "- Only choose a 'high' or 'specialized' strategy if the market conditions or historical data provide a clear and compelling reason (e.g., a 'PanicFade' strategy during extreme VIX levels)."
        )

        nickname_prompt_part = ""
        if script_version not in ledger:
            theme = random.choice(["Astronomical Objects", "Mythological Figures", "Gemstones", "Constellations", "Legendary Swords"])
            nickname_prompt_part = (
                "**NICKNAME GENERATION**: This is a new script version. Generate a unique, cool-sounding, one-word codename for this run. "
                f"Theme: **{theme}**. Avoid these past names: {list(ledger.values())}. "
                "Place it in the `nickname` key.\n"
            )
        else:
            nickname_prompt_part = "**NICKNAME GENERATION**: A nickname already exists. Set `nickname` to `null`.\n"


        prompt = (
            "You are a Master Trading Strategist responsible for configuring a trading framework for its next run. Your decisions must be evidence-based, combining internal data with real-time external information.\n\n"
            f"{task_prompt}\n\n"
            f"{complexity_prompt}\n\n"
            f"{nickname_prompt_part}\n"
            "**PARAMETER RULES & GUIDANCE:**\n"
            "- `selected_features`: **Required.** Select 4-6 relevant features. Must include two context features (e.g., `DAILY_ctx_Trend`) unless GNN-based.\n"
            "- `RETRAINING_FREQUENCY`: Must be a string (e.g., '90D').\n"
            "- `OPTUNA_TRIALS`: An integer between 30-150.\n\n"
            "**OUTPUT FORMAT**: Respond ONLY with a single, valid JSON object.\n\n"
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
            logger.error("  - AI-driven setup failed to return a valid configuration.")
            return {}

    def get_strategic_forecast(self, config: ConfigModel, data_summary: Dict, playbook: Dict) -> Dict:
        if not self.api_key_valid: return {}
        logger.info("-> Performing AI Strategic Forecast & Contingency Planning...")
        prompt = (
            "You are a Master Trading Strategist creating a game plan for an upcoming trading period. Your primary strategy has been selected. "
            "Your task is to perform a 'pre-mortem' analysis and define your contingency plans in advance, like a chess player planning several moves ahead.\n\n"
            "**Primary Path (Path A):**\n"
            f"- Strategy: `{config.strategy_name}`\n"
            f"- Key Parameters: `selected_features`: {config.selected_features}, `RETRAINING_FREQUENCY`: {config.RETRAINING_FREQUENCY}\n"
            f"- Market Context: {json.dumps(self._sanitize_dict(data_summary), indent=2)}\n\n"
            "**YOUR THREE TASKS:**\n\n"
            "1.  **Pre-Mortem Analysis:** Imagine this run has failed. Based on the chosen strategy and the current market context, what are the **two most likely reasons for failure?** "
            "(e.g., 'A sudden volatility spike will render the mean-reversion logic obsolete,' or 'The trend is too weak, leading to many false breakout signals.')\n\n"
            "2.  **Define Contingency Path B (New Strategy):** Based on your pre-mortem, what is the single best **alternative strategy** from the playbook to pivot to if Path A fails completely? "
            "This should be a strategy that is resilient to the failure modes you identified. Provide the `strategy_name` and a safe, small starting `selected_features` list for it.\n\n"
            "3.  **Define Contingency Path C (Parameter Tweak):** If Path A fails but you believe the core strategy is still viable, what is the single most important **parameter change** you would make? "
            "(e.g., Drastically reduce `OPTUNA_TRIALS` and `max_depth`, or change `RETRAINING_FREQUENCY` to '30D').\n\n"
            f"**AVAILABLE STRATEGIES (PLAYBOOK):**\n{json.dumps(self._sanitize_dict(playbook), indent=2)}\n\n"
            "Respond ONLY with a single, valid JSON object with the keys `pre_mortem_analysis` (string), `contingency_path_b` (object), and `contingency_path_c` (object)."
        )
        response_text = self._call_gemini(prompt)
        forecast = self._extract_json_from_response(response_text)
        if forecast and all(k in forecast for k in ['pre_mortem_analysis', 'contingency_path_b', 'contingency_path_c']):
            logger.info("  - AI Strategic Forecast complete.")
            return forecast
        else:
            logger.error("  - AI failed to return a valid strategic forecast.")
            return {}

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
        strategic_forecast: Optional[Dict] = None
    ) -> Dict:
        if not self.api_key_valid: return {}

        base_prompt_intro = "You are an expert trading model analyst. Your primary goal is to create a STABLE and PROFITABLE strategy by making intelligent, data-driven changes, using the comprehensive historical and performance context provided."
        json_response_format = "Respond ONLY with a valid JSON object containing your suggested keys and an `analysis_notes` key explaining your reasoning."

        analysis_points = [
            "--- COMPREHENSIVE META-ANALYSIS FRAMEWORK ---",
            "**1. Model Quality vs. PNL:** Analyze the `BestObjectiveScore` (model quality) against the final `PNL` for each cycle. Is there a correlation? A low score followed by a high PNL is a sign of LUCK, not skill. Prioritize stability.",
            "**2. Feature Drift (SHAP History):** Analyze the `shap_importance_history`. Are core features losing predictive power over time? A sudden collapse in a feature's importance signals a regime change.",
            "**3. Granular Exit Analysis (MAE/MFE):** Look at the `trade_summary`. High MAE (Maximum Adverse Excursion) on losing trades suggests poor entries. High MFE (Maximum Favorable Excursion) on losing trades suggests take-profit levels are too far away.",
            "**4. Hyperparameter Learning:** Review the `optuna_trial_history`. Are there parameter values (e.g., `max_depth` > 8) that consistently lead to failure? Learn from this history to avoid repeating mistakes.",
            "**5. Optimization Path Analysis:** Review the `optimization_path` in the most recent cycle's history. Did the model's score improve steadily and efficiently, or was it an erratic, difficult search? A difficult search is a strong warning sign about the current feature set's quality.",
            "**6. External Context Post-Mortem (GROUNDED SEARCH):** The previous cycle ran from **{cycle_start_date}** to **{cycle_end_date}**. Use the provided `MACRO_CONTEXT`, `ASSET CORRELATION SUMMARY`, and Google Search to analyze the cycle's performance.",
            "   - **Identify Primary and Secondary Events:** Find major news events that affected both the directly traded assets AND any strongly correlated assets (e.g., FOMC, CPI, NFP for USD, EUR, GBP, JPY, etc.).",
            "   - **Correlate Performance to Indirect Causes:** In your `analysis_notes`, explain if PNL swings were caused by events affecting correlated currencies. For example: *'The strategy's failure in trading EURUSD was not due to EUR or USD news. It was caused by the unexpected rate hike from the SNB, which caused a shockwave across all European currencies, as the Swiss Franc (CHF) is strongly correlated with the Euro (EUR). The model was not prepared for this second-order impact.'* or *'The large win on AUDUSD was amplified by stronger-than-expected economic data from New Zealand, as the NZD and AUD are tightly correlated commodity currencies.'*"
        ]

        if cycle_status == "TRAINING_FAILURE":
            task_guidance = (
                "**CRITICAL: MODEL TRAINING FAILURE!**\n"
                "The model failed the quality gate. Your top priority is to propose a change that **increases model stability and signal quality**. Your first instinct should be to **drastically simplify the feature set** based on the most historically stable features from the SHAP history. Avoid failed hyperparameters from the Optuna history."
            )
        elif cycle_status == "PROBATION":
             task_guidance = (
                "**STRATEGY ON PROBATION**\n"
                "The previous cycle hit its drawdown limit. Your **only goal** is to **REDUCE RISK**. You MUST suggest changes that lower risk, like reducing `MAX_DD_PER_CYCLE` or `MAX_CONCURRENT_TRADES`. Justify your choice using the full analysis framework."
             )
        else: # Standard cycle
            task_guidance = (
                "**STANDARD CYCLE REVIEW**\n"
                "Your task is to synthesize all six analysis points into a coherent set of changes. Propose a new configuration that improves robustness and profitability. For example, if the optimization path was erratic and SHAP shows a feature is losing importance, a feature set change is strongly warranted."
            )

        optuna_summary = {}
        if all_optuna_trials:
            sorted_trials = sorted(all_optuna_trials, key=lambda x: x.get('value', -99), reverse=True)
            optuna_summary = {"best_5_trials": sorted_trials[:5], "worst_5_trials": sorted_trials[-5:]}

        data_context = (
            f"--- DATA FOR YOUR ANALYSIS ---\n\n"
            f"**A. CURRENT RUN - CYCLE-BY-CYCLE HISTORY:**\n{json.dumps(self._sanitize_dict(historical_results), indent=2)}\n\n"
            f"**B. MACROECONOMIC CONTEXT:**\n{json.dumps(self._sanitize_dict(macro_context), indent=2)}\n\n"
            f"**C. ASSET CORRELATION SUMMARY (INTERNAL):**\n{correlation_summary_for_ai}\n\n"
            f"**D. FEATURE IMPORTANCE HISTORY (SHAP values over time):**\n{json.dumps(self._sanitize_dict(shap_history), indent=2)}\n\n"
            f"**E. HYPERPARAMETER HISTORY (Sample from Optuna Trials):**\n{json.dumps(self._sanitize_dict(optuna_summary), indent=2)}\n\n"
            f"**F. CURRENT STRATEGY & AVAILABLE FEATURES:**\n`strategy_name`: {strategy_details.get('strategy_name')}\n`available_features`: {available_features}\n"
        )
        prompt = (
            f"{base_prompt_intro}\n\n**YOUR TASK:**\n{task_guidance}\n\n"
            "**ANALYTICAL FRAMEWORK (Address these points in your reasoning):**\n"
            + "\n".join(analysis_points).format(cycle_start_date=cycle_start_date, cycle_end_date=cycle_end_date)
            + f"\n\n{json_response_format}\n\n{data_context}"
        )

        response_text = self._call_gemini(prompt)
        suggestions = self._extract_json_from_response(response_text)
        return suggestions

    def propose_strategic_intervention(self, failure_history: List[Dict], playbook: Dict, last_failed_strategy: str, quarantine_list: List[str], dynamic_best_config: Optional[Dict] = None) -> Dict:
        if not self.api_key_valid: return {}
        logger.warning("! STRATEGIC INTERVENTION !: Current strategy has failed repeatedly. Engaging AI for a new path.")
        available_playbook = { k: v for k, v in playbook.items() if k not in quarantine_list and not v.get("retired") and (GNN_AVAILABLE or not v.get("requires_gnn"))}
        feature_selection_guidance = (
            "**You MUST provide a `selected_features` list.** Start with a **small, targeted set of 4-6 features** from the playbook for the new strategy you choose. "
            "The list MUST include at least TWO multi-timeframe context features (e.g., `DAILY_ctx_Trend`, `H1_ctx_SMA`)."
        )
        base_prompt = (
            f"You are a master strategist executing an emergency intervention. The current strategy, "
            f"**`{last_failed_strategy}`**, has failed multiple consecutive cycles and is now quarantined.\n\n"
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
            if 'RealVolume' not in df.columns: df['RealVolume'] = 0
            # Downcast to reduce memory usage
            df['RealVolume'] = pd.to_numeric(df['RealVolume'], errors='coerce').fillna(0).astype('int32')
            for col in ['Open', 'High', 'Low', 'Close']:
                if col in df.columns:
                    df[col] = pd.to_numeric(df[col], errors='coerce').astype('float32')

            df['Symbol'] = symbol; return df
        except Exception as e: logger.error(f"  - Failed to load {filename}: {e}", exc_info=True); return None

    def load_and_parse_data(self, filenames: List[str]) -> Tuple[Optional[Dict[str, pd.DataFrame]], List[str]]:
        logger.info("-> Stage 1: Loading and Preparing Multi-Timeframe Data...")
        data_by_tf = defaultdict(list)
        for filename in filenames:
            file_path = os.path.join(self.config.BASE_PATH, filename)
            if not os.path.exists(file_path): logger.warning(f"  - File not found, skipping: {file_path}"); continue
            df = self._parse_single_file(file_path, filename)
            if df is not None: tf = filename.split('_')[1]; data_by_tf[tf].append(df)
        processed_dfs: Dict[str, pd.DataFrame] = {}
        for tf, dfs in data_by_tf.items():
            if dfs:
                combined = pd.concat(dfs)
                # Ensure data is sorted by timestamp before returning
                final_combined = combined.sort_index()
                processed_dfs[tf] = final_combined
                logger.info(f"  - Processed {tf}: {len(final_combined):,} rows for {len(final_combined['Symbol'].unique())} symbols.")
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
        'pct_change', 'candle_body_size_vs_atr', 'atr_vs_daily_atr', 'MACD_hist'
    ]

    def __init__(self, config: 'ConfigModel', timeframe_roles: Dict[str, str]):
        self.config = config
        self.roles = timeframe_roles

    def _get_weights_ffd(self, d: float, thres: float) -> np.ndarray:
        w, k = [1.], 1
        while True:
            w_ = -w[-1] / k * (d - k + 1)
            if abs(w_) < thres: break
            w.append(w_)
            k += 1
        return np.array(w[::-1]).reshape(-1, 1)

    def _fractional_differentiation(self, series: pd.Series, d: float, thres: float = 1e-5) -> pd.Series:
        weights = self._get_weights_ffd(d, thres)
        width = len(weights)
        if width > len(series): return pd.Series(index=series.index)
        diff_series = series.rolling(width).apply(lambda x: np.dot(weights.T, x)[0], raw=True)
        diff_series.name = f"{series.name}_fracdiff_{d}"
        return diff_series

    def _get_anomaly_scores(self, df: pd.DataFrame, contamination: float) -> pd.Series:
        features_to_check = [f for f in self.ANOMALY_FEATURES if f in df.columns]
        df_clean = df[features_to_check].dropna()
        if df_clean.empty:
            return pd.Series(1, index=df.index, name='anomaly_score')
        model = IsolationForest(contamination=contamination, random_state=42, n_estimators=100)
        model.fit(df_clean)
        scores = pd.Series(model.predict(df[features_to_check].fillna(0)), index=df.index)
        scores.name = 'anomaly_score'
        return scores

    def hawkes_process(self, data: pd.Series, kappa: float) -> pd.Series:
        if not isinstance(data, pd.Series) or data.isnull().all():
            logger.warning("Hawkes process received invalid data; returning zeros.")
            return pd.Series(np.zeros(len(data)), index=data.index)
        assert kappa > 0.0
        alpha = np.exp(-kappa)
        arr = data.to_numpy()
        output = np.zeros(len(data))
        output[:] = np.nan
        for i in range(1, len(data)):
            if np.isnan(output[i - 1]): output[i] = arr[i]
            else: output[i] = output[i - 1] * alpha + arr[i]
        return pd.Series(output, index=data.index) * kappa

    def _apply_pca_to_features(self, df: pd.DataFrame, feature_prefix: str, n_components: int) -> pd.DataFrame:
        pca_features = df.filter(regex=f'^{feature_prefix}').copy()
        if pca_features.shape[1] < n_components:
            logger.warning(f"    - Not enough features ({pca_features.shape[1]}) for PCA with n_components={n_components}. Skipping.")
            return pd.DataFrame(index=df.index)
        pca_features.dropna(inplace=True)
        if pca_features.empty or pca_features.shape[1] < n_components:
            logger.warning("    - Feature set for PCA is empty or has too few columns after dropping NaNs. Skipping PCA.")
            return pd.DataFrame(index=df.index)
        scaler = StandardScaler()
        scaled_features = scaler.fit_transform(pca_features)
        pca = PCA(n_components=n_components)
        principal_components = pca.fit_transform(scaled_features)
        pc_df = pd.DataFrame(data=principal_components, columns=[f'PCA_{feature_prefix}_{i}' for i in range(n_components)], index=pca_features.index)
        return pc_df

    def _calculate_rsi_divergence(self, g: pd.DataFrame, lookback: int = 14) -> pd.DataFrame:
        low_prices = g['Low'].rolling(window=lookback, center=False).min()
        rsi_at_low = g['RSI'][g['Low'] == low_prices]
        high_prices = g['High'].rolling(window=lookback, center=False).max()
        rsi_at_high = g['RSI'][g['High'] == high_prices]
        price_makes_lower_low = (low_prices < low_prices.shift(1)).astype(int)
        rsi_makes_higher_low = (rsi_at_low > rsi_at_low.shift(1)).reindex(g.index).fillna(0).astype(int)
        g['rsi_bullish_divergence'] = (price_makes_lower_low & rsi_makes_higher_low)
        price_makes_higher_high = (high_prices > high_prices.shift(1)).astype(int)
        rsi_makes_lower_high = (rsi_at_high < rsi_at_high.shift(1)).reindex(g.index).fillna(0).astype(int)
        return g

    def _calculate_hoffman_features(self, g: pd.DataFrame) -> pd.DataFrame:
        ema20 = g['Close'].ewm(span=20, adjust=False).mean()
        g['EMA_20_slope'] = ema20.diff()
        candle_range = g['High'] - g['Low']
        candle_range = candle_range.replace(0, np.nan)
        is_strong_uptrend = g['EMA_20_slope'] > g['EMA_20_slope'].rolling(10).mean()
        is_strong_downtrend = g['EMA_20_slope'] < g['EMA_20_slope'].rolling(10).mean()
        g['is_hoffman_irb_bullish'] = (is_strong_uptrend & (((g['Close'] - g['Low']) / candle_range.replace(0,1)) < 0.45) & (((g['Open'] - g['Low']) / candle_range.replace(0,1)) < 0.45)).astype(int)
        g['is_hoffman_irb_bearish'] = (is_strong_downtrend & (((g['High'] - g['Close']) / candle_range.replace(0,1)) < 0.45) & (((g['High'] - g['Open']) / candle_range.replace(0,1)) < 0.45)).astype(int)
        return g

    def _calculate_ict_features(self, g: pd.DataFrame, swing_lookback: int = 10) -> pd.DataFrame:
        bullish_fvg_condition = g['High'].shift(2) < g['Low']
        bearish_fvg_condition = g['Low'].shift(2) > g['High']
        g['fvg_bullish_exists'] = bullish_fvg_condition.astype(int)
        g['fvg_bearish_exists'] = bearish_fvg_condition.astype(int)
        swing_highs = g['High'].rolling(swing_lookback*2+1, center=True).max()
        swing_lows = g['Low'].rolling(swing_lookback*2+1, center=True).min()
        g['liquidity_grab_up'] = ((g['High'] > swing_highs.shift(1)) & (g['Close'] < swing_highs.shift(1))).astype(int)
        g['liquidity_grab_down'] = ((g['Low'] < swing_lows.shift(1)) & (g['Close'] > swing_lows.shift(1))).astype(int)
        g['choch_up_signal'] = (g['Close'] > swing_highs.shift(1)).astype(int)
        g['choch_down_signal'] = (g['Close'] < swing_lows.shift(1)).astype(int)
        return g

    def _calculate_market_structure(self, g: pd.DataFrame, swing_lookback: int = 10) -> pd.DataFrame:
        window = swing_lookback * 2 + 1
        local_highs = g['High'].rolling(window, center=True, min_periods=window).max()
        local_lows = g['Low'].rolling(window, center=True, min_periods=window).min()
        swing_high_points = g['High'][g['High'] == local_highs]
        swing_low_points = g['Low'][g['Low'] == local_lows]
        g['swing_high'] = swing_high_points.ffill()
        g['swing_low'] = swing_low_points.ffill()
        g['bos_up_signal'] = (g['Close'] > g['swing_high'].shift(1)).astype(int)
        g['bos_down_signal'] = (g['Close'] < g['swing_low'].shift(1)).astype(int)
        g['bos_up_since'] = g.groupby((g['bos_up_signal'] == 1).cumsum()).cumcount()
        g['bos_down_since'] = g.groupby((g['bos_down_signal'] == 1).cumsum()).cumcount()
        g.drop(columns=['swing_high', 'swing_low'], inplace=True, errors='ignore')
        return g

    def _calculate_volatility_regime(self, g:pd.DataFrame, hurst_window:int=100) -> pd.DataFrame:
        g['hurst_exponent'] = g['Close'].rolling(window=hurst_window).apply(lambda x: compute_Hc(x, kind='price', simplified=True)[0] if len(x)==hurst_window else np.nan, raw=False)
        g['market_mode'] = pd.cut(g['hurst_exponent'], bins=[0, 0.4, 0.6, 1], labels=[-1, 0, 1], right=False)
        bb_width_rank = g['bollinger_bandwidth'].rolling(hurst_window).rank(pct=True)
        g['bollinger_squeeze'] = (bb_width_rank < 0.1).astype(int)
        return g

    def _calculate_zscores_and_interactions(self, g:pd.DataFrame, z_window:int=50) -> pd.DataFrame:
        for col in ['RSI', 'momentum_20', 'MACD_hist']:
             if col in g.columns:
                mean = g[col].rolling(window=z_window).mean()
                std = g[col].rolling(window=z_window).std().replace(0, np.nan)
                g[f'{col}_zscore'] = (g[col] - mean) / std
        g['momentum_20_norm_atr'] = g['momentum_20'] / g['ATR'].replace(0, np.nan)
        g['adx_x_rsi'] = (g['ADX'] / 50.0) * (g['RSI'] / 100.0)
        if 'hurst_exponent' in g.columns:
            g['hurst_x_adx'] = g['hurst_exponent'] * g['ADX']
        return g

    def _calculate_support_resistance(self, g: pd.DataFrame, period: int = 20) -> pd.DataFrame:
        g[f'support_level_{period}'] = g['Low'].rolling(window=period).min()
        g[f'resistance_level_{period}'] = g['High'].rolling(window=period).max()
        return g

    def _enhance_volume_features(self, g: pd.DataFrame, spike_multiplier: float = 2.0, spike_window: int = 50) -> pd.DataFrame:
        if 'volume' in g.columns:
            g['volume_ma'] = g['volume'].rolling(window=spike_window).mean()
            g['volume_spike'] = (g['volume'] > g['volume_ma'] * spike_multiplier).astype(int)
            g.drop(columns=['volume_ma'], inplace=True, errors='ignore')
        return g

    def _calculate_adx(self, g:pd.DataFrame, period:int) -> pd.DataFrame:
        df=g.copy();alpha=1/period;df['tr']=pd.concat([df['High']-df['Low'],abs(df['High']-df['Close'].shift()),abs(df['Low']-df['Close'].shift())],axis=1).max(axis=1)
        df['dm_plus']=((df['High']-df['High'].shift())>(df['Low'].shift()-df['Low'])).astype(int)*(df['High']-df['High'].shift()).clip(lower=0)
        df['dm_minus']=((df['Low'].shift()-df['Low'])>(df['High']-df['High'].shift())).astype(int)*(df['Low'].shift()-df['Low']).clip(lower=0)
        atr_adx=df['tr'].ewm(alpha=alpha,adjust=False).mean();di_plus=100*(df['dm_plus'].ewm(alpha=alpha,adjust=False).mean()/atr_adx.replace(0,1e-9))
        di_minus=100*(df['dm_minus'].ewm(alpha=alpha,adjust=False).mean()/atr_adx.replace(0,1e-9));dx=100*(abs(di_plus-di_minus)/(di_plus+di_minus).replace(0,1e-9))
        g['ADX']=dx.ewm(alpha=alpha,adjust=False).mean();return g

    def _calculate_bollinger_bands(self, g:pd.DataFrame, period:int) -> pd.DataFrame:
        rolling_close=g['Close'].rolling(window=period);middle_band=rolling_close.mean();std_dev=rolling_close.std()
        g['bollinger_upper'] = middle_band + (std_dev * 2); g['bollinger_lower'] = middle_band - (std_dev * 2)
        g['bollinger_middle'] = middle_band
        g['bollinger_bandwidth'] = (g['bollinger_upper'] - g['bollinger_lower']) / middle_band.replace(0,np.nan); return g

    def _calculate_stochastic(self, g:pd.DataFrame, period:int) -> pd.DataFrame:
        low_min=g['Low'].rolling(window=period).min();high_max=g['High'].rolling(window=period).max()
        g['stoch_k']=100*(g['Close']-low_min)/(high_max-low_min).replace(0,np.nan);g['stoch_d']=g['stoch_k'].rolling(window=3).mean();return g

    def _calculate_momentum(self, g:pd.DataFrame) -> pd.DataFrame:
        g['momentum_10'] = g['Close'].diff(10)
        g['momentum_20'] = g['Close'].diff(20)
        g['pct_change'] = g['Close'].pct_change()
        g['log_returns'] = np.log(g['Close'] / g['Close'].shift(1))
        return g

    def _calculate_seasonality(self, g: pd.DataFrame) -> pd.DataFrame:
        g['month'] = g.index.month
        g['week_of_year'] = g.index.isocalendar().week.astype(int)
        g['day_of_month'] = g.index.day
        return g

    def _calculate_candle_microstructure(self, g: pd.DataFrame) -> pd.DataFrame:
        g['candle_body_size'] = abs(g['Close'] - g['Open'])
        g['upper_wick'] = g['High'] - g[['Open', 'Close']].max(axis=1)
        g['lower_wick'] = g[['Open', 'Close']].min(axis=1) - g['Low']
        candle_range = (g['High'] - g['Low']).replace(0, np.nan)
        g['wick_to_body_ratio'] = (g['upper_wick'] + g['lower_wick']) / g['candle_body_size'].replace(0, 1e-9)
        g['is_doji'] = (g['candle_body_size'] / g['ATR'].replace(0,1)).lt(0.1).astype(int)
        g['is_engulfing'] = ((g['candle_body_size'] > abs(g['Close'].shift() - g['Open'].shift())) & (np.sign(g['Close']-g['Open']) != np.sign(g['Close'].shift()-g['Open'].shift()))).astype(int)
        g['candle_body_size_vs_atr'] = g['candle_body_size'] / g['ATR'].replace(0, 1)
        g['candle_body_to_range_ratio'] = g['candle_body_size'] / candle_range
        return g

    def _calculate_indicator_dynamics(self, g: pd.DataFrame, period: int = 5) -> pd.DataFrame:
        def get_slope(series):
            if len(series) < 2 or series.isnull().all(): return np.nan
            series_float = series.fillna(method='ffill').fillna(method='bfill').astype(float)
            if series_float.isnull().all(): return np.nan
            return np.polyfit(np.arange(len(series_float)), series_float, 1)[0]
        g['RSI_slope'] = g['RSI'].rolling(window=period).apply(get_slope, raw=False)
        g['momentum_10_slope'] = g['momentum_10'].rolling(window=period).apply(get_slope, raw=False)
        if 'MACD_hist' in g.columns:
            g['MACD_hist_slope'] = g['MACD_hist'].rolling(window=period).apply(get_slope, raw=False)
        g['RSI_slope_acceleration'] = g['RSI_slope'].diff()
        g['momentum_10_slope_acceleration'] = g['momentum_10_slope'].diff()
        return g

    def _calculate_markov_features(self, g: pd.DataFrame) -> pd.DataFrame:
        candle_color = np.sign(g['Close'] - g['Open']).fillna(0)
        blocks = (candle_color != candle_color.shift()).cumsum()
        streaks = candle_color.groupby(blocks).cumsum()
        g['markov_streak'] = streaks
        return g

    def _calculate_htf_features(self,df:pd.DataFrame,p:str,s:int,a:int)->pd.DataFrame:
        tf_id = p.upper()
        results=[]
        def get_rolling_slope(series, window):
            if series.notna().sum() < 2: return np.nan
            series_clean = series.dropna()
            if len(series_clean) < 2: return np.nan
            return np.polyfit(series_clean.index.astype(np.int64) // 10**9, series_clean.values, 1)[0]
        for symbol,group in df.groupby('Symbol'):
            g=group.copy()
            sma=g['Close'].rolling(s,min_periods=s).mean()
            atr=(g['High']-g['Low']).rolling(a,min_periods=a).mean()
            trend=np.sign(g['Close']-sma)
            lin_reg_slope = g['Close'].rolling(window=s).apply(get_rolling_slope, raw=False, args=(s,))
            temp_df=pd.DataFrame(index=g.index)
            temp_df[f'{tf_id}_ctx_SMA']=sma
            temp_df[f'{tf_id}_ctx_ATR']=atr
            temp_df[f'{tf_id}_ctx_Trend']=trend
            temp_df[f'{tf_id}_ctx_LinRegSlope'] = lin_reg_slope
            shifted_df=temp_df.shift(1);shifted_df['Symbol']=symbol;results.append(shifted_df)
        if not results: return pd.DataFrame()
        return pd.concat(results).reset_index()

    def _calculate_base_tf_native(self, g:pd.DataFrame)->pd.DataFrame:
        g_out = g.copy()
        lookback=14

        # --- BLOCK 1: CORE INDICATORS (Dependencies for other features) ---
        g_out['ATR']=(g_out['High']-g_out['Low']).rolling(lookback).mean()
        delta=g_out['Close'].diff()
        gain=delta.where(delta > 0,0).ewm(com=lookback-1,adjust=False).mean()
        loss=-delta.where(delta < 0,0).ewm(com=lookback-1,adjust=False).mean()
        rs = gain / loss.replace(0, 1e-9)
        g_out['RSI']=100-(100/(1+rs))
        
        # --- [FIX] Generate multiple RSI periods for PCA ---
        if getattr(self.config, 'USE_PCA_REDUCTION', False) and hasattr(self.config, 'RSI_PERIODS_FOR_PCA'):
            for period in self.config.RSI_PERIODS_FOR_PCA:
                delta_pca = g_out['Close'].diff()
                gain_pca = delta_pca.where(delta_pca > 0, 0).ewm(com=period - 1, adjust=False).mean()
                loss_pca = -delta_pca.where(delta_pca < 0, 0).ewm(com=period - 1, adjust=False).mean()
                rs_pca = gain_pca / loss_pca.replace(0, 1e-9)
                g_out[f'rsi_{period}'] = 100 - (100 / (1 + rs_pca))
        # --- [END FIX] ---

        g_out=self._calculate_adx(g_out,lookback)
        g_out=self._calculate_bollinger_bands(g_out,self.config.BOLLINGER_PERIOD)
        g_out=self._calculate_stochastic(g_out,self.config.STOCHASTIC_PERIOD)
        g_out = self._calculate_momentum(g_out)

        g_out['EMA_20'] = g_out['Close'].ewm(span=20, adjust=False).mean()
        g_out['EMA_50'] = g_out['Close'].ewm(span=50, adjust=False).mean()
        g_out['EMA_100'] = g_out['Close'].ewm(span=100, adjust=False).mean()
        g_out['EMA_200'] = g_out['Close'].ewm(span=200, adjust=False).mean()
        
        ema_12 = g_out['Close'].ewm(span=12, adjust=False).mean()
        ema_26 = g_out['Close'].ewm(span=26, adjust=False).mean()
        g_out['MACD_line'] = ema_12 - ema_26
        g_out['MACD_signal'] = g_out['MACD_line'].ewm(span=9, adjust=False).mean()
        g_out['MACD_hist'] = g_out['MACD_line'] - g_out['MACD_signal']
        
        # --- BLOCK 2: CANDLE & VOLUME FEATURES (Depend on Block 1) ---
        g_out = self._calculate_candle_microstructure(g_out)
        
        if 'RealVolume' in g_out.columns:
            g_out['volume'] = g_out['RealVolume']
        else:
            g_out['volume'] = 0

        # --- BLOCK 3: NEW SIMPLE FEATURES (Depend on Block 1 & 2) ---
        g_out['overnight_gap_pct'] = (g_out['Open'] - g_out['Close'].shift(1)) / g_out['Close'].shift(1)
        g_out['intraday_range_pct'] = (g_out['High'] - g_out['Low']) / g_out['Open'].replace(0, np.nan)
        g_out['close_vs_open_pct'] = (g_out['Close'] - g_out['Open']) / g_out['Open'].replace(0, np.nan)
        
        g_out['ema_20_vs_50'] = (g_out['EMA_20'] - g_out['EMA_50']) / g_out['EMA_50'].replace(0, np.nan)
        g_out['ema_50_vs_200'] = (g_out['EMA_50'] - g_out['EMA_200']) / g_out['EMA_200'].replace(0, np.nan)
        
        g_out['is_bullish_hammer'] = ((g_out['lower_wick'] > 2 * g_out['candle_body_size']) & (g_out['upper_wick'] < g_out['candle_body_size']) & (g_out['Close'] > g_out['Open'])).astype(int)
        g_out['is_bearish_shooting_star'] = ((g_out['upper_wick'] > 2 * g_out['candle_body_size']) & (g_out['lower_wick'] < g_out['candle_body_size']) & (g_out['Close'] < g_out['Open'])).astype(int)
        
        if g_out['volume'].sum() > 0:
            g_out['volume_ma_ratio'] = g_out['volume'] / g_out['volume'].rolling(20).mean().replace(0, np.nan)
            g_out['volume_trend'] = g_out['volume'].rolling(5).apply(lambda x: np.polyfit(np.arange(len(x)), x, 1)[0] if x.notna().all() else np.nan, raw=False)
        else:
            g_out['volume_ma_ratio'] = 0
            g_out['volume_trend'] = 0

        g_out['momentum_5'] = g_out['Close'].pct_change(5)
        g_out['momentum_10_vs_20'] = g_out['momentum_10'] - g_out['momentum_20']
        
        g_out['atr_ratio'] = g_out['ATR'] / g_out['ATR'].rolling(20).mean().replace(0, np.nan)
        g_out['volatility_change'] = g_out['ATR'].pct_change()

        # --- BLOCK 4: DERIVED & ADVANCED FEATURES (Depend on previous blocks) ---
        g_out = self._calculate_indicator_dynamics(g_out)
        g_out = self._calculate_markov_features(g_out)
        g_out = self._calculate_seasonality(g_out)
        
        g_out['market_regime']=np.where(g_out['ADX']>self.config.TREND_FILTER_THRESHOLD,1,0)
        sma_fast = g_out['Close'].rolling(window=20).mean()
        sma_slow = g_out['Close'].rolling(window=50).mean()
        signal_series = pd.Series(np.where(sma_fast > sma_slow, 1.0, -1.0), index=g_out.index)
        g_out['primary_model_signal'] = signal_series.diff().fillna(0)
        
        g_out['market_volatility_index'] = g_out['ATR'].rolling(100).rank(pct=True)
        g_out['close_fracdiff'] = self._fractional_differentiation(g_out['Close'], d=0.5)
        g_out['abs_log_returns'] = g_out['log_returns'].abs().fillna(0)
        g_out['volatility_hawkes'] = self.hawkes_process(g_out['abs_log_returns'], kappa=self.config.HAWKES_KAPPA)
        g_out['returns_autocorr_10'] = g_out['log_returns'].rolling(10).corr(g_out['log_returns'].shift(1))

        g_out['donchian_upper'] = g_out['High'].rolling(20).max()
        g_out['donchian_lower'] = g_out['Low'].rolling(20).min()
        g_out['donchian_channel'] = g_out['donchian_upper'] - g_out['donchian_lower']
        g_out['linear_regression'] = g_out['Close'].rolling(window=14).apply(lambda x: np.polyfit(np.arange(len(x)), x, 1)[0], raw=False)
        g_out['SMA_30_weekly'] = g_out['Close'].rolling(window=30*5).mean()

        ha_close = (g_out['Open'] + g_out['High'] + g_out['Low'] + g_out['Close']) / 4
        ha_open = ((g_out['Open'].shift(1) + g_out['Close'].shift(1)) / 2).bfill()
        g_out['ha_body_size'] = abs(ha_close - ha_open)
        g_out['ha_color'] = np.sign(ha_close - ha_open)
        ha_blocks = (g_out['ha_color'] != g_out['ha_color'].shift()).cumsum()
        g_out['ha_streak'] = g_out.groupby(ha_blocks)['ha_color'].cumsum().abs()

        g_out['fractal_up'] = ((g_out['High'] > g_out['High'].shift(1)) & (g_out['High'] > g_out['High'].shift(2)) & (g_out['High'] > g_out['High'].shift(-1)) & (g_out['High'] > g_out['High'].shift(-2))).astype(int)
        g_out['fractal_down'] = ((g_out['Low'] < g_out['Low'].shift(1)) & (g_out['Low'] < g_out['Low'].shift(2)) & (g_out['Low'] < g_out['Low'].shift(-1)) & (g_out['Low'] < g_out['Low'].shift(-2))).astype(int)

        if g_out['volume'].sum() > 0:
            g_out = self._enhance_volume_features(g_out)
            if g_out.index.nlevels > 1:
                 g_out['relative_strength'] = (g_out['pct_change'] - g_out.groupby(level=0)['pct_change'].transform('mean'))
            else: g_out['relative_strength'] = 0
        else:
            g_out['relative_strength'] = 0; g_out['volume_spike'] = 0

        g_out = self._calculate_rsi_divergence(g_out, lookback=lookback)
        g_out = self._calculate_hoffman_features(g_out)
        g_out = self._calculate_ict_features(g_out, swing_lookback=10)
        g_out = self._calculate_market_structure(g_out, swing_lookback=10)
        g_out = self._calculate_volatility_regime(g_out, hurst_window=100)
        g_out = self._calculate_support_resistance(g_out, period=20)
        g_out = self._calculate_zscores_and_interactions(g_out, z_window=50)

        if g_out['volume'].sum() > 0:
            tpv = ((g_out['High'] + g_out['Low'] + g_out['Close']) / 3) * g_out['volume']
            cum_volume = g_out.groupby(g_out.index.date)['volume'].transform('cumsum')
            cum_tpv = g_out.groupby(g_out.index.date).apply(lambda x: tpv.loc[x.index].cumsum()).reset_index(level=0, drop=True)
            g_out['VWAP'] = cum_tpv / cum_volume.replace(0, np.nan)
            g_out['VWAP'] = g_out['VWAP'].ffill()
            g_out['price_to_vwap'] = (g_out['Close'] - g_out['VWAP']) / g_out['ATR'].replace(0, np.nan)
            g_out['price_vs_vwap_sign'] = np.sign(g_out['Close'] - g_out['VWAP'])
            g_out['vwap_slope'] = g_out['VWAP'].diff()
        else:
            g_out['VWAP']=np.nan; g_out['price_to_vwap']=np.nan; g_out['price_vs_vwap_sign']=np.nan; g_out['vwap_slope']=np.nan

        return g_out

    def _calculate_relative_performance(self, df: pd.DataFrame) -> pd.DataFrame:
        if 'pct_change' not in df.columns:
            logger.warning("  - 'pct_change' not found, cannot calculate relative performance.")
            return df
        if 'Symbol' in df.columns and df['Symbol'].nunique() > 1:
            df['avg_market_pct_change'] = df.groupby(level=0)['pct_change'].transform('mean')
            df['relative_performance'] = df['pct_change'] - df['avg_market_pct_change']
        else:
            df['relative_performance'] = 0
        return df

    def _process_single_symbol_stack(self, data_by_tf_single_symbol: Dict[str, pd.DataFrame]) -> pd.DataFrame:
        base_tf, medium_tf, high_tf = self.roles['base'], self.roles['medium'], self.roles['high']
        df_base = data_by_tf_single_symbol[base_tf]

        df_base_featured = self._calculate_base_tf_native(df_base)
        df_merged = df_base_featured.reset_index()

        if medium_tf and medium_tf in data_by_tf_single_symbol and not data_by_tf_single_symbol[medium_tf].empty:
            df_medium_ctx = self._calculate_htf_features(data_by_tf_single_symbol[medium_tf], medium_tf, 50, 14)
            if not df_medium_ctx.empty:
                df_merged = pd.merge_asof(df_merged.sort_values('Timestamp'), df_medium_ctx.sort_values('Timestamp'), on='Timestamp', by='Symbol', direction='backward')

        if high_tf and high_tf in data_by_tf_single_symbol and not data_by_tf_single_symbol[high_tf].empty:
            df_high_ctx = self._calculate_htf_features(data_by_tf_single_symbol[high_tf], high_tf, 20, 14)
            if not df_high_ctx.empty:
                df_merged = pd.merge_asof(df_merged.sort_values('Timestamp'), df_high_ctx.sort_values('Timestamp'), on='Timestamp', by='Symbol', direction='backward')

        df_final = df_merged.set_index('Timestamp').copy()
        del df_merged, df_base_featured

        if medium_tf:
            tf_id = medium_tf.upper()
            df_final[f'adx_x_{tf_id}_trend'] = df_final['ADX'] * df_final.get(f'{tf_id}_ctx_Trend', 0)
        if high_tf:
            tf_id = high_tf.upper()
            df_final[f'atr_x_{tf_id}_trend'] = df_final['ATR'] * df_final.get(f'{tf_id}_ctx_Trend', 0)
            df_final['atr_vs_daily_atr'] = df_final['ATR'] / df_final.get(f'{tf_id}_ctx_ATR', 1).replace(0, 1)

        if getattr(self.config, 'USE_PCA_REDUCTION', False):
            logger.info(f"    - Applying PCA to RSI features for symbol.")
            rsi_pc_df = self._apply_pca_to_features(df_final, 'rsi_', self.config.PCA_N_COMPONENTS)
            if not rsi_pc_df.empty:
                df_final = df_final.join(rsi_pc_df)
                cols_to_drop = [c for c in df_final.columns if c.startswith('rsi_')]
                df_final.drop(columns=cols_to_drop, inplace=True)
        df_final['anomaly_score'] = self._get_anomaly_scores(df_final, self.config.anomaly_contamination_factor)
        return df_final

    def create_feature_stack(self, data_by_tf: Dict[str, pd.DataFrame]) -> pd.DataFrame:
        logger.info("-> Stage 2: Engineering Features (Processing Symbol-by-Symbol to conserve memory)...")
        base_tf = self.roles['base']
        if base_tf not in data_by_tf:
            logger.critical(f"Base timeframe '{base_tf}' data is missing. Cannot proceed.")
            return pd.DataFrame()

        all_symbols_processed_dfs = []
        unique_symbols = data_by_tf[base_tf]['Symbol'].unique()

        for i, symbol in enumerate(unique_symbols):
            logger.info(f"  - ({i+1}/{len(unique_symbols)}) Processing features for symbol: {symbol}")
            symbol_specific_data = {tf: df[df['Symbol'] == symbol].copy() for tf, df in data_by_tf.items()}
            processed_symbol_df = self._process_single_symbol_stack(symbol_specific_data)
            del symbol_specific_data
            if not processed_symbol_df.empty:
                all_symbols_processed_dfs.append(processed_symbol_df)

        if not all_symbols_processed_dfs:
            logger.critical("Feature engineering resulted in no processable data across all symbols.")
            return pd.DataFrame()

        logger.info("  - Concatenating data for all symbols...")
        final_df = pd.concat(all_symbols_processed_dfs, sort=False).sort_index()
        del all_symbols_processed_dfs

        logger.info("  - Calculating cross-symbol features (relative performance)...")
        final_df = self._calculate_relative_performance(final_df)

        logger.info("  - Applying final data shift and cleaning...")
        feature_cols = [c for c in final_df.columns if c not in ['Open','High','Low','Close','RealVolume','Symbol']]
        final_df[feature_cols] = final_df.groupby('Symbol', sort=False)[feature_cols].shift(1)
        final_df.replace([np.inf,-np.inf],np.nan,inplace=True)
        # Drop rows with NaN in essential core features to ensure model stability
        core_features = ['ATR', 'RSI', 'ADX']
        final_df.dropna(subset=core_features, inplace=True)

        logger.info(f"  - Merged data and created features. Final dataset shape: {final_df.shape}")
        logger.info("[SUCCESS] Feature engineering complete.")
        return final_df

    def label_outcomes(self,df:pd.DataFrame,lookahead:int)->pd.DataFrame:
        logger.info("  - Generating trade labels with VOLATILITY-ADJUSTED DYNAMIC BARRIERS...");
        labeled_dfs=[self._label_group(group,lookahead) for _,group in df.groupby('Symbol')];
        return pd.concat(labeled_dfs)

    def _label_group(self, group: pd.DataFrame, lookahead: int) -> pd.DataFrame:
        """
        Calculates trade outcomes (1 for long win, -1 for short win, 0 for no outcome)
        based on dynamic, volatility-adjusted take-profit and stop-loss levels.
        This method is now controlled by parameters in the ConfigModel for AI-driven optimization.
        """
        group = group.copy()
        if 'ATR' not in group.columns or len(group) < lookahead + 1:
            logger.warning(f"ATR not found or insufficient data for labeling in group. Skipping.")
            group['target'] = 0
            return group

        # Use dynamic parameters from the main configuration, enabling AI control
        tp_multiplier = self.config.TP_ATR_MULTIPLIER
        sl_multiplier = self.config.SL_ATR_MULTIPLIER

        profit_target_points = group['ATR'] * tp_multiplier
        stop_loss_points = group['ATR'] * sl_multiplier
        
        outcomes = np.zeros(len(group))
        prices = group['Close'].values
        highs = group['High'].values
        lows = group['Low'].values

        for i in range(len(group) - lookahead):
            sl_dist = stop_loss_points.iloc[i]
            tp_dist = profit_target_points.iloc[i]

            if pd.isna(sl_dist) or sl_dist <= 1e-9:
                continue

            # Define levels for both long and short scenarios
            tp_long_level = prices[i] + tp_dist
            sl_long_level = prices[i] - sl_dist
            tp_short_level = prices[i] - tp_dist
            sl_short_level = prices[i] + sl_dist

            # Slice future price action
            future_highs = highs[i+1 : i+1+lookahead]
            future_lows = lows[i+1 : i+1+lookahead]

            # Find first time hitting TP/SL for a long trade
            hit_tp_long_idx = np.where(future_highs >= tp_long_level)[0]
            hit_sl_long_idx = np.where(future_lows <= sl_long_level)[0]
            first_tp_long = hit_tp_long_idx[0] if len(hit_tp_long_idx) > 0 else np.inf
            first_sl_long = hit_sl_long_idx[0] if len(hit_sl_long_idx) > 0 else np.inf

            # Find first time hitting TP/SL for a short trade
            hit_tp_short_idx = np.where(future_lows <= tp_short_level)[0]
            hit_sl_short_idx = np.where(future_highs >= sl_short_level)[0]
            first_tp_short = hit_tp_short_idx[0] if len(hit_tp_short_idx) > 0 else np.inf
            first_sl_short = hit_sl_short_idx[0] if len(hit_sl_short_idx) > 0 else np.inf

            # Pessimistic assignment: only assign a label if one barrier is hit before the other
            if first_tp_long < first_sl_long:
                outcomes[i] = 1  # Long trade won
            if first_tp_short < first_sl_short:
                outcomes[i] = -1 # Short trade won
        
        group['target'] = outcomes
        return group

    def _label_meta_group(self, group: pd.DataFrame, lookahead: int) -> pd.DataFrame:
        group = group.copy()
        if 'primary_model_signal' not in group.columns or len(group) < lookahead + 1:
            group['target'] = 0; return group
        
        # Use dynamic parameters from the main configuration for meta-labeling as well
        tp_multiplier = self.config.TP_ATR_MULTIPLIER
        sl_multiplier = self.config.SL_ATR_MULTIPLIER
        
        sl_atr_dynamic = group['ATR'] * sl_multiplier
        tp_atr_dynamic = group['ATR'] * tp_multiplier
        
        outcomes = np.zeros(len(group))
        prices, lows, highs = group['Close'].values, group['Low'].values, group['High'].values
        primary_signals = group['primary_model_signal'].values
        min_return = self.config.LABEL_MIN_RETURN_PCT

        for i in range(len(group) - lookahead):
            signal = primary_signals[i]
            if signal == 0: continue

            sl_dist, tp_dist = sl_atr_dynamic[i], tp_atr_dynamic[i]
            if pd.isna(sl_dist) or sl_dist <= 1e-9: continue
            
            future_highs, future_lows = highs[i + 1:i + 1 + lookahead], lows[i + 1:i + 1 + lookahead]
            
            if signal > 0: # Primary model signals a long
                tp_level, sl_level = prices[i] + tp_dist, prices[i] - sl_dist
                if (tp_level / prices[i] - 1) <= min_return: continue
                time_to_tp = np.where(future_highs >= tp_level)[0]
                time_to_sl = np.where(future_lows <= sl_level)[0]
                if len(time_to_tp) > 0 and (len(time_to_sl) == 0 or time_to_tp[0] < time_to_sl[0]):
                    outcomes[i] = 1 # Primary signal was correct
            
            elif signal < 0: # Primary model signals a short
                tp_level, sl_level = prices[i] - tp_dist, prices[i] + sl_dist
                if (prices[i] / tp_level - 1) <= min_return: continue
                time_to_tp = np.where(future_lows <= tp_level)[0]
                time_to_sl = np.where(future_highs >= sl_level)[0]
                if len(time_to_tp) > 0 and (len(time_to_sl) == 0 or time_to_tp[0] < time_to_sl[0]):
                    outcomes[i] = 1 # Primary signal was correct
        
        group['target'] = outcomes
        return group

    def label_meta_outcomes(self, df: pd.DataFrame, lookahead: int) -> pd.DataFrame:
        logger.info("  - Generating BINARY meta-labels (1=correct, 0=incorrect)...")
        labeled_dfs = [self._label_meta_group(group, lookahead) for _, group in df.groupby('Symbol')]
        if not labeled_dfs: return pd.DataFrame()
        return pd.concat(labeled_dfs)

# =============================================================================
# 6. MODELS & TRAINER (V206 UPDATE)
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
    def __init__(self,config:ConfigModel):
        self.config=config
        self.shap_summary:Optional[pd.DataFrame]=None
        self.class_weights:Optional[Dict[int,float]]=None
        self.best_threshold=0.5
        self.study: Optional[optuna.study.Study] = None
        self.is_gnn_model = False
        self.is_meta_model = False
        self.is_transformer_model = False
        self.gnn_model: Optional[GNNModel] = None
        self.gnn_scaler = MinMaxScaler()
        self.asset_map: Dict[str, int] = {}

    def train(self, df_train: pd.DataFrame, feature_list: List[str], strategy_details: Dict) -> Optional[Tuple[Pipeline, float]]:
        logger.info(f"  - Starting model training using strategy: '{strategy_details.get('description', 'N/A')}'")
        self.is_gnn_model = strategy_details.get("requires_gnn", False)
        self.is_meta_model = strategy_details.get("requires_meta_labeling", False)
        self.is_transformer_model = strategy_details.get("requires_transformer", False)
        X = pd.DataFrame()

        if self.is_transformer_model:
            if not GNN_AVAILABLE:
                logger.error("  - Skipping Transformer model training: PyTorch libraries not found.")
                return None
            logger.info("  - Transformer strategy detected. Training regression model.")
            df_train = df_train.copy()
            df_train['target_price'] = df_train['Close'].shift(-1)
            df_train.dropna(subset=['target_price'], inplace=True)
            X = df_train[feature_list].copy().fillna(0)
            y = df_train['target_price']
            X_seq, y_seq = [], []
            seq_len = 30
            for i in range(len(X) - seq_len):
                X_seq.append(X.iloc[i:i+seq_len].values)
                y_seq.append(y.iloc[i+seq_len-1])
            X_seq, y_seq = torch.tensor(np.array(X_seq), dtype=torch.float32), torch.tensor(np.array(y_seq), dtype=torch.float32).unsqueeze(1)
            dataset = torch.utils.data.TensorDataset(X_seq, y_seq)
            train_loader = torch.utils.data.DataLoader(dataset, batch_size=32, shuffle=True)
            model = TimeSeriesTransformer(feature_size=len(feature_list), seq_length=seq_len, prediction_length=1)
            criterion = nn.MSELoss()
            optimizer = Adam(model.parameters(), lr=0.001)
            for epoch in range(20):
                for x_batch, y_batch in train_loader:
                    optimizer.zero_grad()
                    output = model(x_batch)
                    loss = criterion(output, y_batch)
                    loss.backward()
                    optimizer.step()
            logger.info("  - [SUCCESS] Transformer training complete.")
            return model, 0.0

        if self.is_gnn_model:
            if not GNN_AVAILABLE:
                logger.error("  - Skipping GNN model training: PyTorch/PyG libraries not found.")
                return None
            logger.info("  - GNN strategy detected. Generating graph embeddings as features...")
            gnn_embeddings = self._train_gnn(df_train)
            if gnn_embeddings.empty:
                logger.error("  - GNN embedding generation failed. Aborting cycle.")
                return None
            X = gnn_embeddings
            feature_list = list(X.columns)
            logger.info(f"  - Feature set replaced by {len(feature_list)} GNN embeddings.")
            y_map={-1:0,0:1,1:2}; y=df_train['target'].map(y_map).astype(int); num_classes = 3
        else:
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
        
        # Simple split for final threshold tuning and training, CV is done in optimization
        X_train_val, _, y_train_val, _ = train_test_split(X, y, test_size=0.1, shuffle=False)
        X_train, X_val, y_train, y_val = train_test_split(X_train_val, y_train_val, test_size=0.2, shuffle=False)

        if X_train.empty or X_val.empty:
            logger.error(f"  - Training aborted: Data split resulted in an empty training or validation set. (Train shape: {X_train.shape}, Val shape: {X_val.shape})")
            return None
        
        # Pass the full training set (X, y) to the optimization function for K-Fold CV
        self.study=self._optimize_hyperparameters(X, y, num_classes)
        if not self.study or not self.study.best_trials:
            logger.error("  - Training aborted: Hyperparameter optimization failed.")
            return None

        logger.info(f"    - Optimization complete. Best Objective Score: {self.study.best_value:.4f}")
        logger.info(f"    - Best params: {self.study.best_params}")
        
        # Use the smaller val set for faster threshold finding
        self.best_threshold = self._find_best_threshold(self.study.best_params, X_train, y_train, X_val, y_val, num_classes)
        final_pipeline=self._train_final_model(self.study.best_params,X_train_val,y_train_val, feature_list, num_classes)

        if final_pipeline is None:
            logger.error("  - Training aborted: Final model training failed.")
            return None

        logger.info("  - [SUCCESS] Model training complete.")
        return final_pipeline, self.best_threshold

    def _create_graph_data(self, df: pd.DataFrame) -> Tuple[Optional[Data], Dict[str, int]]:
        logger.info("    - Creating graph structure from asset correlations...")
        pivot_df = df.pivot(columns='Symbol', values='Close').ffill().dropna(how='all', axis=1)
        if pivot_df.shape[1] < 2:
            logger.warning("    - Not enough assets to build a correlation graph. Skipping GNN.")
            return None, {}
        corr_matrix = pivot_df.corr()
        assets = corr_matrix.index.tolist()
        asset_map = {asset: i for i, asset in enumerate(assets)}
        edge_list = []
        for i in range(len(assets)):
            for j in range(i + 1, len(assets)):
                if abs(corr_matrix.iloc[i, j]) > 0.3:
                    edge_list.extend([[asset_map[assets[i]], asset_map[assets[j]]], [asset_map[assets[j]], asset_map[assets[i]]]])
        if not edge_list:
            logger.warning("    - No strong correlations found. Creating a fully connected graph as fallback.")
            edge_list = [[i, j] for i in range(len(assets)) for j in range(len(assets)) if i != j]
        edge_index = torch.tensor(edge_list, dtype=torch.long).t().contiguous()
        feature_cols = [f for f in self.GNN_BASE_FEATURES if f in df.columns]
        node_features = df.groupby('Symbol')[feature_cols].mean().reindex(assets).fillna(0)
        node_features_scaled = pd.DataFrame(self.gnn_scaler.fit_transform(node_features), index=node_features.index)
        x = torch.tensor(node_features_scaled.values, dtype=torch.float)
        return Data(x=x, edge_index=edge_index), asset_map

    def _train_gnn(self, df: pd.DataFrame) -> pd.DataFrame:
        graph_data, self.asset_map = self._create_graph_data(df)
        if graph_data is None: return pd.DataFrame()
        self.gnn_model = GNNModel(in_channels=graph_data.num_node_features, hidden_channels=self.config.GNN_EMBEDDING_DIM * 2, out_channels=self.config.GNN_EMBEDDING_DIM)
        optimizer = Adam(self.gnn_model.parameters(), lr=0.01, weight_decay=5e-4)
        self.gnn_model.train()
        for epoch in range(self.config.GNN_EPOCHS):
            optimizer.zero_grad()
            out = self.gnn_model(graph_data)
            loss = out.mean()
            loss.backward()
            optimizer.step()
        self.gnn_model.eval()
        with torch.no_grad():
            embeddings = self.gnn_model(graph_data).numpy()
        embedding_df = pd.DataFrame(embeddings, index=self.asset_map.keys(), columns=[f"gnn_{i}" for i in range(self.config.GNN_EMBEDDING_DIM)])
        full_embeddings = df['Symbol'].map(embedding_df.to_dict('index')).apply(pd.Series)
        full_embeddings.index = df.index
        return full_embeddings

    def _get_gnn_embeddings_for_test(self, df_test: pd.DataFrame) -> pd.DataFrame:
        if not self.is_gnn_model or self.gnn_model is None or not self.asset_map: return pd.DataFrame()
        feature_cols = [f for f in self.GNN_BASE_FEATURES if f in df_test.columns]
        test_node_features = df_test.groupby('Symbol')[feature_cols].mean()
        aligned_features = test_node_features.reindex(self.asset_map.keys()).fillna(0)
        test_node_features_scaled = pd.DataFrame(self.gnn_scaler.transform(aligned_features), index=aligned_features.index)
        x = torch.tensor(test_node_features_scaled.values, dtype=torch.float)
        graph_data, _ = self._create_graph_data(df_test)
        if graph_data is None: return pd.DataFrame()
        graph_data.x = x
        self.gnn_model.eval()
        with torch.no_grad():
            embeddings = self.gnn_model(graph_data).numpy()
        embedding_df = pd.DataFrame(embeddings, index=self.asset_map.keys(), columns=[f"gnn_{i}" for i in range(self.config.GNN_EMBEDDING_DIM)])
        full_embeddings = df_test['Symbol'].map(embedding_df.to_dict('index')).apply(pd.Series)
        full_embeddings.index = df_test.index
        return full_embeddings

    def _find_best_threshold(self, best_params, X_train, y_train, X_val, y_val, num_classes) -> float:
        logger.info("    - Tuning classification threshold for F1 score...")
        objective = 'multi:softprob' if num_classes > 2 else 'binary:logistic'
        temp_params = {'objective':objective,'booster':'gbtree','tree_method':'hist',**best_params}
        if num_classes > 2: temp_params['num_class'] = num_classes
        temp_params.pop('early_stopping_rounds', None)
        temp_pipeline = Pipeline([('scaler', RobustScaler()), ('model', xgb.XGBClassifier(**temp_params))])
        fit_params={'model__sample_weight':y_train.map(self.class_weights)}
        temp_pipeline.fit(X_train, y_train, **fit_params)
        probs = temp_pipeline.predict_proba(X_val)
        best_f1, best_thresh = -1, 0.5
        for threshold in np.arange(0.3, 0.7, 0.01):
            if num_classes > 2:
                max_probs = np.max(probs, axis=1)
                preds = np.argmax(probs, axis=1)
                preds = np.where(max_probs > threshold, preds, 1)
            else:
                preds = (probs[:, 1] > threshold).astype(int)
            f1 = f1_score(y_val, preds, average='macro', zero_division=0)
            if f1 > best_f1:
                best_f1, best_thresh = f1, threshold
        logger.info(f"    - Best threshold found: {best_thresh:.2f} (F1: {best_f1:.4f})")
        return best_thresh

    def _optimize_hyperparameters(self, X:pd.DataFrame, y:pd.Series, num_classes: int) -> Optional[optuna.study.Study]:
        logger.info(f"    - Starting hyperparameter optimization with 5-Fold CV and Complexity Penalty ({self.config.OPTUNA_TRIALS} trials)...")

        def dynamic_progress_callback(study: optuna.study.Study, trial: optuna.trial.FrozenTrial):
            n_trials = self.config.OPTUNA_TRIALS
            trial_number = trial.number + 1
            best_value = study.best_value if study.best_trial else float('nan')
            progress_str = f"> Optuna Optimization: Trial {trial_number}/{n_trials} | Best Score: {best_value:.4f}"
            sys.stdout.write(f"\r{progress_str.ljust(80)}")
            sys.stdout.flush()

        objective = 'multi:softprob' if num_classes > 2 else 'binary:logistic'
        eval_metric = 'mlogloss' if num_classes > 2 else 'logloss'
        
        def custom_objective(trial: optuna.Trial) -> float:
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

            # V207 UPDATE: Added explicit complexity penalty
            complexity_penalty = 1.0 + (params['max_depth'] / 10.0) * 0.5 + (params['n_estimators'] / 1000.0) * 0.5
            
            # V207 UPDATE: Using StratifiedKFold for robust evaluation
            skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
            fold_scores = []

            for train_idx, val_idx in skf.split(X, y):
                X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
                y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]

                try:
                    scaler = RobustScaler()
                    X_train_scaled = scaler.fit_transform(X_train)
                    X_val_scaled = scaler.transform(X_val)
                    
                    model = xgb.XGBClassifier(**params)
                    fit_params = {'sample_weight': y_train.map(self.class_weights)}
                    model.fit(X_train_scaled, y_train, eval_set=[(X_val_scaled, y_val)], verbose=False, **fit_params)

                    preds_val = model.predict(X_val_scaled)
                    pnl_map = {0: -1, 1: 0, 2: 1} if num_classes > 2 else {0: -1, 1: 1}
                    pnl_val = pd.Series(preds_val).map(pnl_map)
                    
                    mean_return = pnl_val.mean()
                    std_return = pnl_val.std()
                    
                    sharpe_proxy = mean_return / std_return if std_return > 1e-9 else 0.0
                    fold_scores.append(sharpe_proxy)

                except Exception as e:
                    sys.stdout.write("\n")
                    logger.warning(f"Fold in trial {trial.number} failed with error: {e}")
                    fold_scores.append(-2.0) # Penalize failure heavily

            avg_sharpe_proxy = np.mean(fold_scores)
            final_score = avg_sharpe_proxy / complexity_penalty
            return final_score

        try:
            study = optuna.create_study(direction='maximize')
            study.optimize(custom_objective, n_trials=self.config.OPTUNA_TRIALS, timeout=3600, n_jobs=-1, callbacks=[dynamic_progress_callback])
            sys.stdout.write("\n")
            return study
        except Exception as e:
            sys.stdout.write("\n")
            logger.error(f"    - Optuna study failed catastrophically: {e}", exc_info=True)
            return None

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
                self._generate_shap_summary(final_pipeline.named_steps['model'], final_pipeline.named_steps['scaler'].transform(X), feature_names, num_classes)

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
                mean_abs_shap_per_class = shap_explanation.abs.mean(0).values
                overall_importance = mean_abs_shap_per_class.mean(axis=1) if mean_abs_shap_per_class.ndim == 2 else mean_abs_shap_per_class
            else:
                overall_importance = np.abs(shap_explanation.values).mean(axis=0)
            summary = pd.DataFrame(overall_importance, index=feature_names, columns=['SHAP_Importance']).sort_values(by='SHAP_Importance', ascending=False)
            self.shap_summary = summary
            logger.info("    - SHAP summary generated successfully.")
        except Exception as e:
            logger.error(f"    - Failed to generate SHAP summary: {e}", exc_info=True)
            self.shap_summary = None

# =============================================================================
# 7. BACKTESTER & 8. PERFORMANCE ANALYZER (V206 UPDATE)
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

        # Spread cost is only applied on entry
        spread_cost = 0
        if not on_exit:
            # --- FIX: Robustly get spread info to prevent KeyError ---
            if symbol in self.config.SPREAD_CONFIG:
                spread_info = self.config.SPREAD_CONFIG[symbol]
            else:
                # Fallback to 'default' key, with a hardcoded ultimate fallback
                spread_info = self.config.SPREAD_CONFIG.get('default', {'normal_pips': 1.8, 'volatile_pips': 5.5})
            # --- END FIX ---
            
            vol_rank = candle.get('market_volatility_index', 0.5)
            spread_pips = spread_info.get('volatile_pips', 5.5) if vol_rank > 0.8 else spread_info.get('normal_pips', 1.8)
            spread_cost = spread_pips * point_size

        slippage_cost = 0
        if self.config.USE_VARIABLE_SLIPPAGE:
            atr = candle.get('ATR', 0)
            vol_rank = candle.get('market_volatility_index', 0.5)
            # Slippage can be higher on panicked stop-loss exits
            random_factor = random.uniform(0.1, 1.2 if on_exit else 1.0) * self.config.SLIPPAGE_VOLATILITY_FACTOR
            slippage_cost = atr * vol_rank * random_factor

        return spread_cost, slippage_cost

    def run_backtest_chunk(self, df_chunk_in: pd.DataFrame, confidence_threshold: float, initial_equity: float, strategy_details: Dict) -> Tuple[pd.DataFrame, pd.Series, bool, Optional[Dict], Dict]:
        if df_chunk_in.empty:
            return pd.DataFrame(), pd.Series([initial_equity]), False, None, {}

        df_chunk = df_chunk_in.copy()
        self.is_meta_model = strategy_details.get("requires_meta_labeling", False)
        self.is_transformer_model = strategy_details.get("requires_transformer", False)
        trades, equity, equity_curve, open_positions = [], initial_equity, [initial_equity], {}
        chunk_peak_equity = initial_equity
        circuit_breaker_tripped = False
        breaker_context = None
        candles = df_chunk.reset_index().to_dict('records')

        daily_dd_report = {}
        current_day = None
        day_start_equity = initial_equity
        day_peak_equity = initial_equity

        def finalize_day_metrics(day_to_finalize, equity_at_close):
            if day_to_finalize is None: return
            daily_pnl = equity_at_close - day_start_equity
            daily_dd_pct = ((day_peak_equity - equity_at_close) / day_peak_equity) * 100 if day_peak_equity > 0 else 0
            daily_dd_report[day_to_finalize.isoformat()] = {'pnl': round(daily_pnl, 2), 'drawdown_pct': round(daily_dd_pct, 2)}
            
        def close_trade(pos_to_close, exit_price, exit_reason, candle_info):
            nonlocal equity
            pnl = (exit_price - pos_to_close['entry_price']) * pos_to_close['direction'] * pos_to_close['lot_size'] * self.config.CONTRACT_SIZE
            commission_cost = self.config.COMMISSION_PER_LOT * pos_to_close['lot_size'] * 2 # Entry and Exit
            net_pnl = pnl - commission_cost
            
            equity += net_pnl
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

        for i in range(1, len(candles)):
            current_candle = candles[i]
            prev_candle = candles[i-1]

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
                    
                    # Close all open positions at current candle's open
                    for sym, pos in list(open_positions.items()):
                        close_trade(pos, current_candle['Open'], "Circuit Breaker", current_candle)
                        del open_positions[sym]
                    
                    continue # Skip new trade checks for this candle

            if equity <= 0:
                logger.critical("  - ACCOUNT BLOWN!")
                break

            # Update MFE/MAE for open positions
            for symbol, pos in open_positions.items():
                if pos['direction'] == 1:
                    pos['mfe_price'] = max(pos['mfe_price'], current_candle['High'])
                    pos['mae_price'] = min(pos['mae_price'], current_candle['Low'])
                else: # Short
                    pos['mfe_price'] = min(pos['mfe_price'], current_candle['Low'])
                    pos['mae_price'] = max(pos['mae_price'], current_candle['High'])
            
            symbols_to_close = []
            for symbol, pos in open_positions.items():
                exit_price, exit_reason = None, None
                candle_low, candle_high = current_candle['Low'], current_candle['High']
                
                # Pessimistic exit check: If both SL and TP are hit in the same bar, assume SL is hit first
                sl_hit = (pos['direction'] == 1 and candle_low <= pos['sl']) or \
                         (pos['direction'] == -1 and candle_high >= pos['sl'])
                tp_hit = (pos['direction'] == 1 and candle_high >= pos['tp']) or \
                         (pos['direction'] == -1 and candle_low <= pos['tp'])

                if sl_hit and tp_hit:
                    exit_reason = "Stop Loss (Pessimistic)"
                    _, sl_slippage = self._calculate_realistic_costs(current_candle, on_exit=True)
                    exit_price = pos['sl'] - (sl_slippage * pos['direction'])
                elif sl_hit:
                    exit_reason = "Stop Loss"
                    _, sl_slippage = self._calculate_realistic_costs(current_candle, on_exit=True)
                    exit_price = pos['sl'] - (sl_slippage * pos['direction'])
                elif tp_hit:
                    exit_reason = "Take Profit"
                    exit_price = pos['tp'] # Assume no slippage on limit orders

                # Standard TP/SL exit logic
                if exit_price is not None:
                    close_trade(pos, exit_price, exit_reason, current_candle)
                    symbols_to_close.append(symbol)
                    if equity <= 0: continue
            
            for symbol in set(symbols_to_close):
                if symbol in open_positions:
                    del open_positions[symbol]
            
            # --- New Trade Entry Logic ---
            symbol = prev_candle['Symbol'] 
            if self.config.USE_TIERED_RISK:
                base_risk_pct, max_concurrent_trades = self._get_tiered_risk_params(equity)
            else:
                base_risk_pct, max_concurrent_trades = self.config.BASE_RISK_PER_TRADE_PCT, self.config.MAX_CONCURRENT_TRADES

            if not circuit_breaker_tripped and symbol not in open_positions and len(open_positions) < max_concurrent_trades:
                if prev_candle.get('anomaly_score') == -1: continue
                vol_idx = prev_candle.get('market_volatility_index', 0.5)
                if not (self.config.MIN_VOLATILITY_RANK <= vol_idx <= self.config.MAX_VOLATILITY_RANK): continue

                direction, confidence = 0, 0
                # Meta-model or Standard Model Signal
                if not self.is_transformer_model:
                    if self.is_meta_model:
                        prob_take_trade = prev_candle.get('prob_1', 0)
                        primary_signal = prev_candle.get('primary_model_signal', 0)
                        if prob_take_trade > confidence_threshold and primary_signal != 0:
                            direction, confidence = int(np.sign(primary_signal)), prob_take_trade
                    else: # Standard model
                        if 'prob_short' in prev_candle:
                            probs=np.array([prev_candle['prob_short'],prev_candle['prob_hold'],prev_candle['prob_long']])
                            max_confidence=np.max(probs)
                            if max_confidence >= confidence_threshold:
                                pred_class=np.argmax(probs)
                                direction=1 if pred_class==2 else -1 if pred_class==0 else 0
                                confidence = max_confidence

                if direction != 0:
                    atr = prev_candle.get('ATR',0)
                    if pd.isna(atr) or atr<=1e-9: continue

                    # Determine risk tier
                    tier_name = 'standard'
                    if confidence >= self.config.CONFIDENCE_TIERS['ultra_high']['min']: tier_name = 'ultra_high'
                    elif confidence >= self.config.CONFIDENCE_TIERS['high']['min']: tier_name = 'high'
                    tier = self.config.CONFIDENCE_TIERS[tier_name]
                    
                    # Calculate position size
                    sl_dist = atr * 1.5
                    if sl_dist <= 0: continue
                    
                    risk_per_trade_usd = equity * base_risk_pct * tier['risk_mult']
                    risk_per_trade_usd = min(risk_per_trade_usd, self.config.RISK_CAP_PER_TRADE_USD)
                    
                    # Calculate lot size based on monetary risk
                    point_value = self.config.CONTRACT_SIZE * (0.0001 if 'JPY' not in symbol else 0.01)
                    risk_per_lot = sl_dist * point_value
                    if risk_per_lot <= 0: continue
                    
                    lots = risk_per_trade_usd / risk_per_lot
                    lots = max(self.config.MIN_LOT_SIZE, round(lots / self.config.LOT_STEP) * self.config.LOT_STEP)
                    if lots < self.config.MIN_LOT_SIZE: continue

                    margin_required = (lots * self.config.CONTRACT_SIZE * current_candle['Open']) / self.config.LEVERAGE
                    used_margin = sum(p.get('margin_used', 0) for p in open_positions.values())
                    if (equity - used_margin) < margin_required: continue

                    # Calculate entry and exit prices
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

    def generate_full_report(self,trades_df:Optional[pd.DataFrame],equity_curve:Optional[pd.Series],cycle_metrics:List[Dict],aggregated_shap:Optional[pd.DataFrame]=None, framework_memory:Optional[Dict]=None, aggregated_daily_dd:Optional[List[Dict]]=None) -> Dict[str, Any]:
        logger.info("-> Stage 4: Generating Final Performance Report...")
        if equity_curve is not None and len(equity_curve) > 1: self.plot_equity_curve(equity_curve)
        if aggregated_shap is not None: self.plot_shap_summary(aggregated_shap)

        metrics = self._calculate_metrics(trades_df, equity_curve) if trades_df is not None and not trades_df.empty else {}
        self.generate_text_report(metrics, cycle_metrics, aggregated_shap, framework_memory, aggregated_daily_dd)

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

    def generate_text_report(self, m: Dict[str, Any], cycle_metrics: List[Dict], aggregated_shap: Optional[pd.DataFrame] = None, framework_memory: Optional[Dict] = None, aggregated_daily_dd: Optional[List[Dict]] = None):
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

        report.append(_box_bot(WIDTH))
        final_report = "\n".join(report)
        logger.info("\n" + final_report)
        try:
            with open(self.config.REPORT_SAVE_PATH,'w',encoding='utf-8') as f: f.write(final_report)
        except IOError as e: logger.error(f"  - Failed to save text report: {e}",exc_info=True)

def get_macro_context_data() -> Dict[str, Any]:
    """
    Fetches the latest data for key macroeconomic indicators (VIX, DXY, US10Y),
    with robust error handling for data structure and content.
    """
    logger.info("-> Fetching external macroeconomic context data (VIX, DXY, US10Y)...")
    macro_context = {}
    tickers = {
        "VIX": "^VIX",
        "DXY": "DX-Y.NYB",
        "US10Y_YIELD": "^TNX"
    }

    for name, ticker in tickers.items():
        try:
            data = yf.download(ticker, period="2wk", progress=False)

            if not data.empty and len(data) > 5:
                close = data['Close']
                if isinstance(close, pd.DataFrame):
                    close = close.iloc[:, 0]
                latest_level = close.iloc[-1]
                one_week_ago_level = close.iloc[-6]
                if hasattr(one_week_ago_level, "item"): one_week_ago_level = one_week_ago_level.item()
                if hasattr(latest_level, "item"): latest_level = latest_level.item()
                if one_week_ago_level != 0:
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
    try:
        sanitized_summary = _recursive_sanitize(new_run_summary)
        with open(config.HISTORY_FILE_PATH, 'a') as f: f.write(json.dumps(sanitized_summary) + '\n')
        logger.info(f"-> Run summary appended to history file: {config.HISTORY_FILE_PATH}")
    except IOError as e: logger.error(f"Could not write to history file: {e}")

    MIN_TRADES_FOR_CHAMPION = 10
    current_champion = current_memory.get("champion_config")
    new_metrics = new_run_summary.get("final_metrics", {})
    new_mar = new_metrics.get("mar_ratio", -np.inf)
    new_trade_count = new_metrics.get("total_trades", 0)

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

    if is_new_overall_champion:
        champion_to_save = new_run_summary
        champion_mar = current_champion.get("final_metrics", {}).get("mar_ratio", -np.inf) if current_champion else -np.inf
        logger.info(f"NEW OVERALL CHAMPION! Current run's MAR Ratio ({new_mar:.2f}) beats previous champion's ({champion_mar:.2f}).")
    else:
        champion_to_save = current_champion

    try:
        if champion_to_save:
            with open(config.CHAMPION_FILE_PATH, 'w') as f: json.dump(_recursive_sanitize(champion_to_save), f, indent=4)
            logger.info(f"-> Overall Champion file updated: {config.CHAMPION_FILE_PATH}")
    except (IOError, TypeError) as e: logger.error(f"Could not write to overall champion file: {e}")

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
    
    Loads the playbook from a JSON file if it exists, otherwise creates a new one 
    with default strategies. It also updates the existing playbook with any new 
    default strategies that are missing.
    """
    DEFAULT_PLAYBOOK = {
        "ADXMomentum": {
            "description": "[MOMENTUM] A classic momentum strategy that enters when ADX confirms a strong trend and MACD indicates accelerating momentum.",
            "features": ["ADX", "MACD_hist", "MACD_hist_slope", "momentum_20", "market_regime"],
            "lookahead_range": [60, 180], "dd_range": [0.20, 0.35], "complexity": "medium",
            "ideal_regime": ["Strong Trending"], "asset_class_suitability": ["Any"], "ideal_macro_env": ["Any"]
        },
        "ADXTrendFilterMA": {
            "description": "[HYBRID/TREND] Uses a dual moving average system for entry signals but employs the ADX as a trend strength filter. Trades are only permitted when the ADX is above 25, avoiding choppy, non-trending markets.",
            "features": ["EMA_50", "EMA_200", "ADX", "market_regime"],
            "lookahead_range": [80, 200], "dd_range": [0.20, 0.35], "complexity": "medium",
            "ideal_regime": ["Strong Trending"], "ideal_macro_env": ["Risk-On", "Risk-Off"], "asset_class_suitability": ["Indices", "Equities", "Forex Majors"]
        },
        "BOS_Momentum_Confirmation": {
            "description": "[HYBRID/BOS] This strategy identifies a Break of Structure (BOS) and confirms it with a strong momentum reading from the RSI. A bullish BOS is only valid if the RSI is also firmly in bullish territory (e.g., > 60), filtering out weak breakouts.",
            "features": ["bos_up_signal", "bos_down_signal", "RSI", "ADX", "volume"],
            "lookahead_range": [60, 160], "dd_range": [0.20, 0.35], "complexity": "high",
            "ideal_regime": ["Strong Trending"], "ideal_macro_env": ["Risk-On", "Risk-Off"], "asset_class_suitability": ["Any"]
        },
        "BasicRSIBounce": {
            "description": "[RANGING] Simple RSI bounce strategy that buys when RSI crosses above 30 and sells when crosses below 70.",
            "features": ["RSI", "ADX", "bollinger_bandwidth"],
            "lookahead_range": [20, 60], "dd_range": [0.1, 0.25], "complexity": "low",
            "ideal_regime": ["Ranging"], "asset_class_suitability": ["Any"], "ideal_macro_env": ["Neutral"]
        },
        "BreakoutVolumeConfirmation": {
            "description": "[HYBRID/BREAKOUT] Identifies a breakout of a recent fractal high/low and confirms the validity of the move with a significant spike in volume, filtering out low-conviction false breakouts.",
            "features": ["fractal_up", "fractal_down", "volume", "ATR", "ADX"],
            "lookahead_range": [60, 150], "dd_range": [0.25, 0.40], "complexity": "medium",
            "ideal_regime": ["Strong Trending", "High Volatility"], "ideal_macro_env": ["Risk-On", "Risk-Off"], "asset_class_suitability": ["Equities", "Indices"]
        },
        "CHoCH_Orderblock_Entry": {
            "description": "[HYBRID/CHOCH] This strategy identifies a Change of Character (CHoCH) and then waits for price to pull back to the Fair Value Gap that initiated the CHoCH move.",
            "features": ["choch_up_signal", "choch_down_signal", "fvg_bullish_exists", "fvg_bearish_exists", "volume_spike", "DAILY_ctx_Trend"],
            "lookahead_range": [50, 130], "dd_range": [0.15, 0.25], "complexity": "high",
            "ideal_regime": ["Any"], "asset_class_suitability": ["Forex Majors", "Indices", "Crypto"], "ideal_macro_env": ["Any"]
        },
        "ClassicBollingerRSI": {
            "description": "[RANGING] A traditional mean-reversion strategy entering at the outer bands, filtered by low trend strength.",
            "features": ["bollinger_lower", "bollinger_upper", "RSI", "ADX", "market_mode"],
            "lookahead_range": [20, 70], "dd_range": [0.1, 0.2], "complexity": "low",
            "ideal_regime": ["Ranging", "Low Volatility"], "asset_class_suitability": ["Any"], "ideal_macro_env": ["Neutral"]
        },
        "DivergenceConfluenceReversal": {
            "description": "[HYBRID/REVERSAL] A high-confluence strategy that only takes a trade when MACD divergence aligns with a reading from another oscillator (Stochastic), indicating a strong area of potential reversal.",
            "features": ["MACD_hist", "rsi_bullish_divergence", "rsi_bearish_divergence", "stoch_k"],
            "lookahead_range": [60, 160], "dd_range": [0.25, 0.40], "complexity": "high",
            "ideal_regime": ["Any"], "ideal_macro_env": ["Neutral", "Risk-On", "Risk-Off"], "asset_class_suitability": ["Forex Majors"]
        },
        "DivergenceRSIConfirmation": {
            "description": "[HYBRID/REVERSAL] Identifies a completed MACD divergence and requires the RSI to be in an overbought (>70) or oversold (<30) state before entering, adding a layer of momentum confirmation.",
            "features": ["MACD_hist", "rsi_bullish_divergence", "rsi_bearish_divergence", "RSI", "ADX"],
            "lookahead_range": [50, 150], "dd_range": [0.20, 0.35], "complexity": "medium",
            "ideal_regime": ["Any"], "ideal_macro_env": ["Any"], "asset_class_suitability": ["Forex Majors", "Indices"]
        },
        "DojiBollingerReversal": {
            "description": "[HYBRID/CANDLESTICK] A mean-reversion strategy that enters after a Doji candle forms on the upper or lower Bollinger Band, signaling trend exhaustion and indecision at a statistical extreme, anticipating a reversal back to the mean.",
            "features": ["is_doji", "bollinger_upper", "bollinger_lower", "ADX", "wick_to_body_ratio"],
            "lookahead_range": [25, 80], "dd_range": [0.10, 0.20], "complexity": "medium",
            "ideal_regime": ["Ranging", "Low Volatility"], "ideal_macro_env": ["Neutral"], "asset_class_suitability": ["Forex Crosses", "Forex Majors"]
        },
        "DonchianBreakout": {
            "description": "[BREAKOUT] A classic breakout strategy that buys/sells when the price breaks the Donchian Channel.",
            "features": ["donchian_channel", "ATR", "ADX", "volume_spike"],
            "lookahead_range": [50, 150], "dd_range": [0.2, 0.4], "complexity": "low",
            "ideal_regime": ["Strong Trending", "High Volatility"], "asset_class_suitability": ["Commodities", "Indices"], "ideal_macro_env": ["Event-Driven", "Risk-On", "Risk-Off"]
        },
        "DynamicBreakoutMomentum": {
            "description": "[HYBRID/BREAKOUT] Trades breakouts from a Donchian Channel but filters signals with a momentum indicator (e.g., Stochastic). A breakout above the upper channel is only valid if the Stochastic %K is also crossing above 80, indicating strong buying pressure.",
            "features": ["donchian_upper", "donchian_lower", "stoch_k", "ADX", "ATR", "volume"],
            "lookahead_range": [50, 130], "dd_range": [0.20, 0.40], "complexity": "medium",
            "ideal_regime": ["High Volatility", "Strong Trending"], "ideal_macro_env": ["Event-Driven"], "asset_class_suitability": ["Indices", "Commodities", "Forex Majors"]
        },
        "DynamicRangeTrader": {
            "description": "[RANGING] Buys at dynamic support and sells at dynamic resistance, using Hurst to confirm a range-bound market.",
            "features": ["support_level_20", "resistance_level_20", "RSI", "stoch_k", "hurst_exponent", "ADX"],
            "lookahead_range": [30, 80], "dd_range": [0.10, 0.20], "complexity": "medium",
            "ideal_regime": ["Ranging"], "asset_class_suitability": ["Forex Crosses", "Forex Majors"], "ideal_macro_env": ["Any"]
        },
        "DynamicSR_BreakoutRSI": {
            "description": "[HYBRID/BREAKOUT] This strategy confirms a breakout of a dynamic resistance/support level (Donchian Channel) with the RSI. The breakout is only considered valid if the RSI has also crossed a key level (e.g., 60 for an upside breakout), indicating momentum is backing the move.",
            "features": ["donchian_upper", "donchian_lower", "RSI", "volume", "ATR"],
            "lookahead_range": [50, 140], "dd_range": [0.20, 0.35], "complexity": "high",
            "ideal_regime": ["Strong Trending", "High Volatility"], "ideal_macro_env": ["Risk-On", "Risk-Off"], "asset_class_suitability": ["Equities", "Indices", "Commodities"]
        },
        "DynamicSR_CandlestickReversal": {
            "description": "[HYBRID/REVERSAL] Looks for a reversal candlestick pattern (e.g., an Engulfing or Doji) forming directly at a dynamic support or resistance level (e.g., Bollinger Band), providing a strong signal for a potential reversal.",
            "features": ["bollinger_upper", "bollinger_lower", "is_engulfing", "is_doji", "ADX"],
            "lookahead_range": [40, 100], "dd_range": [0.15, 0.30], "complexity": "medium",
            "ideal_regime": ["Ranging", "Weak Trending"], "ideal_macro_env": ["Neutral"], "asset_class_suitability": ["Any"]
        },
        "DynamicSR_ChannelBreakout": {
            "description": "[HYBRID/BREAKOUT] Defines a dynamic channel using Bollinger Bands and enters when the price breaks out of the bands after a period of contraction (a 'squeeze'). The breakout must be accompanied by an increase in the ADX, confirming expanding volatility.",
            "features": ["bollinger_bandwidth", "ADX", "ADX_slope", "volume", "momentum_20"],
            "lookahead_range": [50, 130], "dd_range": [0.20, 0.35], "complexity": "medium",
            "ideal_regime": ["Low Volatility", "High Volatility"], "ideal_macro_env": ["Event-Driven", "Neutral"], "asset_class_suitability": ["Any"]
        },
        "DynamicSR_MACD": {
            "description": "[HYBRID/TREND] Uses a moving average (e.g., 50 EMA) as a dynamic support/resistance level. It enters on a bounce off the EMA, but only if the MACD histogram is also positive (for longs) or negative (for shorts), confirming trend momentum.",
            "features": ["EMA_50", "MACD_hist", "ADX", "DAILY_ctx_Trend"],
            "lookahead_range": [40, 120], "dd_range": [0.15, 0.25], "complexity": "medium",
            "ideal_regime": ["Weak Trending", "Strong Trending"], "ideal_macro_env": ["Any"], "asset_class_suitability": ["Any"]
        },
        "EMAScalpingFractal": {
            "description": "[SCALPING] A high-frequency scalping strategy using EMA alignment and Williams Fractal for entry signals.",
            "features": ["EMA_20", "EMA_50", "fractal_up", "fractal_down", "ATR"],
            "lookahead_range": [15, 45], "dd_range": [0.1, 0.25], "complexity": "low",
            "ideal_regime": ["Weak Trending", "Strong Trending"], "asset_class_suitability": ["Forex Majors", "Indices"], "ideal_macro_env": ["Neutral"]
        },
        "ElliotWaveFibonacci": {
            "description": "[HYBRID/TREND] A confluence strategy that waits for a strong trend (high ADX), then enters on a pullback to a dynamic support level (50 EMA), anticipating the next wave of momentum.",
            "features": ["ADX", "EMA_50", "RSI", "volume"],
            "lookahead_range": [100, 250], "dd_range": [0.25, 0.40], "complexity": "high",
            "ideal_regime": ["Strong Trending"], "ideal_macro_env": ["Any"], "asset_class_suitability": ["Indices", "Equities"]
        },
        "ElliotWaveMomentum": {
            "description": "[HYBRID/TREND] Identifies a corrective pullback (proxied by a short-term counter-trend streak) within a strong primary trend and combines it with an oversold Stochastic oscillator to signal a high-probability entry for the next impulsive move.",
            "features": ["markov_streak", "stoch_k", "stoch_d", "ADX", "DAILY_ctx_Trend"],
            "lookahead_range": [80, 200], "dd_range": [0.20, 0.35], "complexity": "high",
            "ideal_regime": ["Strong Trending"], "ideal_macro_env": ["Risk-On", "Risk-Off"], "asset_class_suitability": ["Indices", "Forex Majors"]
        },
        "EmaCrossoverRsiFilter": {
            "description": "[TRENDING] Classic 50/200 EMA crossover signal, filtered by RSI for momentum confirmation.",
            "features": ["EMA_50", "EMA_200", "RSI", "ADX"],
            "lookahead_range": [60, 180], "dd_range": [0.2, 0.4], "complexity": "low",
            "ideal_regime": ["Strong Trending"], "ideal_macro_env": ["Risk-On", "Risk-Off"], "asset_class_suitability": ["Indices", "Equities", "Forex Majors"]
        },
        "EngulfingRsiScalp": {
            "description": "[SCALPING] An alternative scalping strategy using the 200 EMA for trend, entering on engulfing patterns confirmed by RSI.",
            "features": ["EMA_200", "is_engulfing", "RSI"],
            "lookahead_range": [15, 45], "dd_range": [0.1, 0.25], "complexity": "low",
            "ideal_regime": ["Weak Trending", "Strong Trending"], "ideal_macro_env": ["Neutral"], "asset_class_suitability": ["Forex Majors", "Indices"]
        },
        "EngulfingVolumeConfirmation": {
            "description": "[HYBRID/CANDLESTICK] A classic strategy that identifies a strong bullish or bearish engulfing candle and requires the candle's volume to be significantly higher than average, confirming institutional participation.",
            "features": ["is_engulfing", "volume", "ATR", "DAILY_ctx_Trend"],
            "lookahead_range": [20, 70], "dd_range": [0.10, 0.25], "complexity": "medium",
            "ideal_regime": ["Any"], "ideal_macro_env": ["Any"], "asset_class_suitability": ["Equities", "Forex Majors", "Indices"]
        },
        "FVG_MA_Confluence": {
            "description": "[HYBRID/PRICE ACTION] A confluence strategy that identifies fresh Fair Value Gaps and waits for the price to enter that zone. An entry is only triggered if this zone aligns with a key dynamic S&R level like the 50 EMA.",
            "features": ["fvg_bullish_exists", "fvg_bearish_exists", "EMA_50", "RSI"],
            "lookahead_range": [40, 120], "dd_range": [0.15, 0.25], "complexity": "high",
            "ideal_regime": ["Any"], "ideal_macro_env": ["Any"], "asset_class_suitability": ["Forex Majors", "Indices"]
        },
        "FibonacciDivergenceReversal": {
            "description": "[HYBRID/REVERSAL] A counter-trend strategy that looks for bullish/bearish RSI divergence occurring at the outer Bollinger Bands, signaling potential trend exhaustion and a high-probability reversal point.",
            "features": ["bollinger_upper", "bollinger_lower", "rsi_bullish_divergence", "rsi_bearish_divergence", "ADX", "market_volatility_index"],
            "lookahead_range": [30, 90], "dd_range": [0.20, 0.35], "complexity": "high",
            "ideal_regime": ["Ranging", "Weak Trending"], "ideal_macro_env": ["Neutral"], "asset_class_suitability": ["Any"]
        },
        "FibonacciMomentum": {
            "description": "[HYBRID/TREND] Combines dynamic support levels with momentum confirmation. Enters on a bounce from the middle Bollinger Band but only if the RSI confirms the primary trend's momentum is still intact.",
            "features": ["bollinger_middle", "RSI", "DAILY_ctx_Trend", "ADX", "momentum_10"],
            "lookahead_range": [40, 120], "dd_range": [0.15, 0.30], "complexity": "high",
            "ideal_regime": ["Strong Trending", "Weak Trending"], "ideal_macro_env": ["Risk-On", "Risk-Off"], "asset_class_suitability": ["Forex Majors", "Indices", "Commodities"]
        },
        "FilteredBreakout": {
            "description": "[BREAKOUT] A hybrid that trades high-volatility breakouts but only in the direction of the long-term daily trend.",
            "features": ["ATR", "bollinger_bandwidth", "DAILY_ctx_Trend", "ADX", "hour", "anomaly_score", "RSI_slope"],
            "lookahead_range": [60, 120], "dd_range": [0.2, 0.35], "complexity": "medium",
            "ideal_regime": ["High Volatility", "Strong Trending"], "ideal_macro_env": ["Event-Driven", "Risk-On", "Risk-Off"], "asset_class_suitability": ["Indices", "Commodities", "Forex Majors"]
        },
        "FvgLiquidityGrab": {
            "description": "[HYBRID/PRICE ACTION] This strategy waits for a liquidity grab (a sweep of a recent high or low) and then looks for an immediate shift in market structure (CHoCH) that leaves behind a Fair Value Gap (FVG). It enters on the retest of the FVG, anticipating a reversal.",
            "features": ["liquidity_grab_up", "liquidity_grab_down", "fvg_bullish_exists", "fvg_bearish_exists", "choch_up_signal", "choch_down_signal"],
            "lookahead_range": [50, 130], "dd_range": [0.20, 0.35], "complexity": "high",
            "ideal_regime": ["Any"], "ideal_macro_env": ["Any"], "asset_class_suitability": ["Forex Majors", "Indices", "Crypto"]
        },
        "FvgOrderblockEntry": {
            "description": "[HYBRID/PRICE ACTION] Identifies a Fair Value Gap (FVG) and waits for the price to retrace into it. The entry is confirmed by a prior liquidity grab, which acts as a proxy for an institutional orderblock.",
            "features": ["fvg_bullish_exists", "fvg_bearish_exists", "liquidity_grab_up", "liquidity_grab_down", "DAILY_ctx_Trend"],
            "lookahead_range": [40, 110], "dd_range": [0.15, 0.25], "complexity": "high",
            "ideal_regime": ["Weak Trending", "Strong Trending"], "ideal_macro_env": ["Neutral", "Risk-On", "Risk-Off"], "asset_class_suitability": ["Forex Majors", "Indices"]
        },
        "GNN_Market_Structure": {
            "description": "[SPECIALIZED] Uses a GNN to model inter-asset correlations for predictive features.",
            "features": [], "lookahead_range": [80, 150], "dd_range": [0.15, 0.3], "complexity": "specialized",
            "requires_gnn": True, "ideal_regime": ["Any"], "ideal_macro_env": ["Any"], "asset_class_suitability": ["Forex Majors", "Indices"]
        },
        "HeikinAshiDynamicSR": {
            "description": "[HYBRID/TREND] A visual trend-following strategy that confirms a bounce off a dynamic support/resistance level (50 EMA) with a change in Heikin-Ashi candle color.",
            "features": ["EMA_50", "ha_color", "ha_streak", "DAILY_ctx_Trend"],
            "lookahead_range": [70, 180], "dd_range": [0.20, 0.30], "complexity": "medium",
            "ideal_regime": ["Strong Trending"], "ideal_macro_env": ["Any"], "asset_class_suitability": ["Any"]
        },
        "HeikinAshiMACD": {
            "description": "[HYBRID/TREND] Combines the smoothed trend visualization of Heikin-Ashi candles with the MACD indicator. A long entry is triggered when Heikin-Ashi candles turn green while the MACD histogram is also positive, signaling a potential new uptrend.",
            "features": ["ha_color", "ha_streak", "MACD_hist", "ADX"],
            "lookahead_range": [60, 160], "dd_range": [0.15, 0.25], "complexity": "medium",
            "ideal_regime": ["Strong Trending", "Weak Trending"], "ideal_macro_env": ["Risk-On", "Risk-Off"], "asset_class_suitability": ["Indices", "Commodities", "Forex Majors"]
        },
        "HeikinAshiTrend": {
            "description": "[TRENDING/PRICE ACTION] A robust trend-following strategy using clean Heikin-Ashi candle signals to ride trends.",
            "features": ["ha_color", "ha_body_size", "ha_streak", "DAILY_ctx_LinRegSlope", "market_volatility_index", "ADX"],
            "lookahead_range": [60, 160], "dd_range": [0.15, 0.25], "complexity": "medium",
            "ideal_regime": ["Strong Trending", "Weak Trending"], "asset_class_suitability": ["Indices", "Commodities", "Forex Majors"], "ideal_macro_env": ["Risk-On", "Risk-Off"]
        },
        "HiddenDivergenceTrend": {
            "description": "[HYBRID/DIVERGENCE] A trend-following strategy that looks for hidden divergence. In an uptrend, price makes a higher low, but the RSI makes a lower low. This signals a likely continuation of the primary trend.",
            "features": ["rsi_bullish_divergence", "rsi_bearish_divergence", "DAILY_ctx_Trend", "ADX", "EMA_50"],
            "lookahead_range": [60, 160], "dd_range": [0.15, 0.25], "complexity": "high",
            "ideal_regime": ["Weak Trending", "Strong Trending"], "ideal_macro_env": ["Any"], "asset_class_suitability": ["Any"]
        },
        "HoffmanTrendRetracement": {
            "description": "[TRENDING] Enters on the resumption of a strong trend after a pause indicated by an Inventory Retracement Bar (IRB).",
            "features": ["EMA_20_slope", "is_hoffman_irb_bullish", "is_hoffman_irb_bearish", "ADX", "ATR"],
            "lookahead_range": [60, 160], "dd_range": [0.2, 0.4], "complexity": "high",
            "ideal_regime": ["Strong Trending"], "ideal_macro_env": ["Risk-On", "Risk-Off"], "asset_class_suitability": ["Equities", "Indices", "Forex Majors"]
        },
        "ICTMarketStructure": {
            "description": "[PRICE ACTION/INSTITUTIONAL] A methodology focused on identifying liquidity zones and Fair Value Gaps (FVG).",
            "features": ["fvg_bullish_exists", "fvg_bearish_exists", "choch_up_signal", "choch_down_signal", "liquidity_grab_up", "liquidity_grab_down", "DAILY_ctx_Trend", "bos_up_signal", "bos_down_signal"],
            "lookahead_range": [40, 120], "dd_range": [0.2, 0.35], "complexity": "high",
            "ideal_regime": ["Strong Trending", "Weak Trending"], "asset_class_suitability": ["Forex Majors", "Indices"], "ideal_macro_env": ["Neutral", "Risk-On", "Risk-Off"]
        },
        "LinRegAngleConfluence": {
            "description": "[HYBRID/TREND] A confluence strategy that looks for entry signals where a Linear Regression line is respected and the price is also at a dynamic S&R level, like a Bollinger Band.",
            "features": ["linear_regression", "bollinger_middle", "RSI", "ADX"],
            "lookahead_range": [100, 250], "dd_range": [0.30, 0.45], "complexity": "medium",
            "ideal_regime": ["Strong Trending"], "ideal_macro_env": ["Any"], "asset_class_suitability": ["Any"]
        },
        "LinRegAngleMomentum": {
            "description": "[HYBRID/TREND] Uses a Linear Regression line as a proxy for trend angle. An entry is triggered on a bounce from the line, confirmed by the MACD histogram ticking higher (for longs) or lower (for shorts).",
            "features": ["linear_regression", "MACD_hist", "ADX", "RSI"],
            "lookahead_range": [80, 220], "dd_range": [0.25, 0.40], "complexity": "medium",
            "ideal_regime": ["Strong Trending"], "ideal_macro_env": ["Any"], "asset_class_suitability": ["Commodities", "Indices"]
        },
        "LinRegBounceStochastic": {
            "description": "[HYBRID/TREND] Enters in the direction of the primary trend when price pulls back and bounces off a dynamic trendline (Linear Regression). The entry is confirmed by the Stochastic oscillator moving out of an oversold/overbought condition.",
            "features": ["linear_regression", "stoch_k", "stoch_d", "ADX", "DAILY_ctx_Trend"],
            "lookahead_range": [60, 150], "dd_range": [0.20, 0.30], "complexity": "medium",
            "ideal_regime": ["Weak Trending", "Strong Trending"], "ideal_macro_env": ["Neutral", "Risk-On", "Risk-Off"], "asset_class_suitability": ["Forex Majors", "Indices", "Equities"]
        },
        "LinRegBreakRSI": {
            "description": "[HYBRID/BREAKOUT] A classic technical analysis strategy that enters on a confirmed break of a dynamic trendline (Linear Regression). The signal is filtered by RSI, requiring it to show strong momentum.",
            "features": ["linear_regression", "RSI", "volume", "ATR"],
            "lookahead_range": [70, 180], "dd_range": [0.25, 0.40], "complexity": "medium",
            "ideal_regime": ["Strong Trending"], "ideal_macro_env": ["Any"], "asset_class_suitability": ["Any"]
        },
        "MACD_RSI_Divergence": {
            "description": "[HYBRID/DIVERGENCE] A powerful reversal strategy that requires a divergence signal on BOTH the MACD and the RSI simultaneously. The price making a new high while both indicators fail to do so is a very strong signal of an impending reversal.",
            "features": ["MACD_hist", "rsi_bullish_divergence", "rsi_bearish_divergence", "volume"],
            "lookahead_range": [50, 150], "dd_range": [0.20, 0.35], "complexity": "high",
            "ideal_regime": ["Any"], "ideal_macro_env": ["Any"], "asset_class_suitability": ["Any"]
        },
        "MACDFractalBreakout": {
            "description": "[BREAKOUT] A breakout strategy that uses Williams Fractal signals for entry, qualified by MACD trend direction.",
            "features": ["MACD_hist", "fractal_up", "fractal_down", "ATR", "ADX"],
            "lookahead_range": [40, 120], "dd_range": [0.2, 0.35], "complexity": "medium",
            "ideal_regime": ["Weak Trending", "Strong Trending", "High Volatility"], "ideal_macro_env": ["Neutral", "Risk-On", "Risk-Off"], "asset_class_suitability": ["Any"]
        },
        "MACDTrendFollowing": {
            "description": "[TRENDING] A classic trend-following strategy using MACD crossovers filtered by a long-term 200-period EMA.",
            "features": ["EMA_200", "MACD_line", "MACD_signal", "MACD_hist", "ADX"],
            "lookahead_range": [50, 150], "dd_range": [0.15, 0.3], "complexity": "low",
            "ideal_regime": ["Strong Trending", "Weak Trending"], "asset_class_suitability": ["Forex Majors", "Indices", "Equities"], "ideal_macro_env": ["Risk-On", "Risk-Off", "Neutral"]
        },
        "MAVolumeHybrid": {
            "description": "[HYBRID] Combines moving average crossover with volume confirmation for higher probability entries.",
            "features": ["EMA_20", "EMA_50", "volume", "volume_spike"],
            "lookahead_range": [40, 100], "dd_range": [0.15, 0.3], "complexity": "low",
            "ideal_regime": ["Weak Trending", "Strong Trending"], "asset_class_suitability": ["Any"], "ideal_macro_env": ["Any"]
        },
        "MeanReversionOscillator": {
            "description": "[RANGING] A pure mean-reversion strategy using oscillators for entry in low-volatility environments.",
            "features": ["RSI", "stoch_k", "ADX", "market_volatility_index", "close_fracdiff", "hour", "day_of_week", "wick_to_body_ratio", "hurst_exponent"],
            "lookahead_range": [20, 60], "dd_range": [0.15, 0.25], "complexity": "medium",
            "ideal_regime": ["Ranging", "Low Volatility"], "ideal_macro_env": ["Neutral"], "asset_class_suitability": ["Forex Crosses", "Forex Majors"]
        },
        "MeanReversionZScore": {
            "description": "[MEAN REVERSION] Exploits statistical deviations from the mean, entering when RSI reaches an extreme Z-score in a non-trending market.",
            "features": ["RSI_zscore", "bollinger_bandwidth", "stoch_k", "stoch_d", "market_mode"],
            "lookahead_range": [20, 70], "dd_range": [0.10, 0.25], "complexity": "medium",
            "ideal_regime": ["Ranging", "Low Volatility"], "asset_class_suitability": ["Forex Majors", "Indices"], "ideal_macro_env": ["Any"]
        },
        "Meta_Labeling_Filter": {
            "description": "[SPECIALIZED] Uses a secondary ML filter to improve a simple primary model's signal quality.",
            "features": ["ADX", "RSI_slope", "ATR", "bollinger_bandwidth", "H1_ctx_Trend", "DAILY_ctx_Trend", "momentum_20", "relative_performance"],
            "lookahead_range": [50, 100], "dd_range": [0.1, 0.25], "complexity": "specialized",
            "requires_meta_labeling": True, "ideal_regime": ["Any"], "asset_class_suitability": ["Any"], "ideal_macro_env": ["Any"]
        },
        "MomentumCrossoverMA": {
            "description": "[HYBRID/TREND] A simple yet effective momentum strategy that uses a fast/slow moving average crossover (e.g., 9 EMA crossing 21 EMA) as a signal, but only takes the trade if a momentum indicator is also positive.",
            "features": ["EMA_20", "EMA_50", "momentum_10", "ADX"],
            "lookahead_range": [40, 100], "dd_range": [0.15, 0.25], "complexity": "low",
            "ideal_regime": ["Weak Trending", "Strong Trending"], "ideal_macro_env": ["Any"], "asset_class_suitability": ["Any"]
        },
        "MultiStructureBreakout": {
            "description": "[HYBRID/MARKET STRUCTURE] A high-conviction breakout strategy that requires a Break of Structure (BOS) on the base timeframe, confirmed by a trend context feature from a higher timeframe.",
            "features": ["bos_up_signal", "bos_down_signal", "volume_spike", "ATR", "DAILY_ctx_Trend"],
            "lookahead_range": [60, 150], "dd_range": [0.20, 0.35], "complexity": "high",
            "ideal_regime": ["Strong Trending"], "asset_class_suitability": ["Forex Majors", "Indices"], "ideal_macro_env": ["Risk-On", "Risk-Off"]
        },
        "NakedPriceAction": {
            "description": "[PRICE ACTION] A pure price action strategy that ignores most indicators, trading on engulfing/doji candles.",
            "features": ["is_engulfing", "is_doji", "wick_to_body_ratio", "DAILY_ctx_LinRegSlope", "hour", "day_of_week"],
            "lookahead_range": [40, 100], "dd_range": [0.15, 0.3], "complexity": "low",
            "ideal_regime": ["Any"], "ideal_macro_env": ["Any"], "asset_class_suitability": ["Any"]
        },
        "OscillatorMomentum": {
            "description": "[HYBRID/OSCILLATOR] An unconventional strategy that enters long when the RSI breaks out above a previous high (e.g., 65), suggesting that momentum is accelerating powerfully in the trend's direction.",
            "features": ["RSI", "RSI_slope_acceleration", "momentum_20", "ADX"],
            "lookahead_range": [50, 120], "dd_range": [0.20, 0.35], "complexity": "medium",
            "ideal_regime": ["Strong Trending", "High Volatility"], "ideal_macro_env": ["Risk-On", "Risk-Off"], "asset_class_suitability": ["Any"]
        },
        "OvernightGapFade": {
            "description": "[HYBRID/INTRADAY] Fades significant overnight gaps with volume confirmation.",
            "features": ["overnight_gap_pct", "volume", "RSI", "ATR"],
            "lookahead_range": [20, 60], "dd_range": [0.2, 0.4], "complexity": "medium",
            "ideal_regime": ["High Volatility"], "asset_class_suitability": ["Equities", "Indices"], "ideal_macro_env": ["Any"]
        },
        "PanicFade": {
            "description": "[CRISIS/EVENT-DRIVEN] A counter-trend strategy designed to fade extreme, news-driven price spikes or drops.",
            "features": ["anomaly_score", "ATR", "wick_to_body_ratio", "candle_body_size_vs_atr", "RSI", "market_volatility_index"],
            "lookahead_range": [20, 50], "dd_range": [0.25, 0.45], "complexity": "specialized",
            "ideal_regime": ["High Volatility"], "asset_class_suitability": ["Any"], "ideal_macro_env": ["Event-Driven", "Risk-Off"]
        },
        "PriceActionSwing": {
            "description": "[PRICE ACTION] Simple swing strategy based on higher highs/lower lows with confirmation from candle close.",
            "features": ["Close", "High", "Low", "ATR"],
            "lookahead_range": [50, 150], "dd_range": [0.2, 0.4], "complexity": "low",
            "ideal_regime": ["Weak Trending", "Strong Trending"], "asset_class_suitability": ["Any"], "ideal_macro_env": ["Any"]
        },
        "RSIDivergenceReversal": {
            "description": "[REVERSAL] A counter-trend strategy that enters when price action diverges from the RSI, signaling trend exhaustion.",
            "features": ["rsi_bullish_divergence", "rsi_bearish_divergence", "stoch_k", "ADX", "market_mode"],
            "lookahead_range": [30, 90], "dd_range": [0.15, 0.3], "complexity": "medium",
            "ideal_regime": ["Ranging", "Weak Trending"], "asset_class_suitability": ["Any"], "ideal_macro_env": ["Neutral"]
        },
        "RSIMACDConfluence": {
            "description": "[HYBRID/OSCILLATOR] Requires both RSI and MACD to confirm signals in the same direction.",
            "features": ["RSI", "MACD_hist", "ADX"],
            "lookahead_range": [50, 120], "dd_range": [0.2, 0.35], "complexity": "low",
            "ideal_regime": ["Any"], "asset_class_suitability": ["Any"], "ideal_macro_env": ["Any"]
        },
        "RSIPullback": {
            "description": "[TRENDING] Enters on pullbacks in an uptrend, using RSI to identify oversold conditions.",
            "features": ["RSI", "SMA_50", "SMA_200"],
            "lookahead_range": [40, 120], "dd_range": [0.15, 0.3], "complexity": "low",
            "ideal_regime": ["Strong Trending", "Weak Trending"], "ideal_macro_env": ["Risk-On", "Risk-Off"], "asset_class_suitability": ["Indices", "Equities", "Forex Majors"]
        },
        "RSI_Stochastic_Confluence": {
            "description": "[HYBRID/OSCILLATOR] A mean-reversion strategy that requires both the RSI and the Stochastic oscillator to be in oversold/overbought territory simultaneously before triggering a trade.",
            "features": ["RSI", "stoch_k", "ADX", "bollinger_bandwidth"],
            "lookahead_range": [20, 60], "dd_range": [0.10, 0.20], "complexity": "low",
            "ideal_regime": ["Ranging", "Low Volatility"], "ideal_macro_env": ["Neutral"], "asset_class_suitability": ["Forex Crosses", "Indices"]
        },
        "RangeBound": {
            "description": "[RANGING] Trades reversals in a sideways channel, filtered by low ADX.",
            "features": ["ADX", "RSI", "stoch_k", "stoch_d", "bollinger_bandwidth", "hour", "wick_to_body_ratio"],
            "lookahead_range": [20, 60], "dd_range": [0.1, 0.2], "complexity": "low",
            "ideal_regime": ["Ranging", "Low Volatility"], "ideal_macro_env": ["Neutral"], "asset_class_suitability": ["Forex Crosses", "Forex Majors"]
        },
        "ReversalCandlestickMA": {
            "description": "[HYBRID/REVERSAL] Detects a reversal candlestick pattern (Doji with long wick) at a significant moving average (like the 200 EMA), providing a powerful confluence for a potential change in trend direction.",
            "features": ["is_doji", "wick_to_body_ratio", "EMA_200", "ADX"],
            "lookahead_range": [30, 80], "dd_range": [0.15, 0.30], "complexity": "medium",
            "ideal_regime": ["Ranging", "Weak Trending"], "ideal_macro_env": ["Neutral"], "asset_class_suitability": ["Any"]
        },
        "ReversalPatternDivergence": {
            "description": "[HYBRID/REVERSAL] Identifies a swing failure/liquidity grab pattern and requires confirmation from MACD divergence, where the indicator fails to make a new high/low along with the price, signaling underlying weakness in the trend.",
            "features": ["liquidity_grab_up", "liquidity_grab_down", "MACD_hist", "volume", "ADX"],
            "lookahead_range": [70, 180], "dd_range": [0.20, 0.35], "complexity": "high",
            "ideal_regime": ["Strong Trending", "Weak Trending"], "ideal_macro_env": ["Any"], "asset_class_suitability": ["Forex Majors", "Indices", "Equities"]
        },
        "SeasonalCycleRSI": {
            "description": "[HYBRID/CYCLICAL] A speculative strategy that combines seasonal tendencies (month of year) with the RSI oscillator. It looks for buying opportunities during historically bullish months only if RSI is also in oversold territory (<30).",
            "features": ["month", "RSI", "market_volatility_index", "DAILY_ctx_Trend"],
            "lookahead_range": [28, 84], "dd_range": [0.20, 0.40], "complexity": "specialized",
            "ideal_regime": ["Any"], "ideal_macro_env": ["Any"], "asset_class_suitability": ["Indices", "Commodities"]
        },
        "SeasonalCycleVolatility": {
            "description": "[HYBRID/CYCLICAL] This strategy hypothesizes that certain times of day (e.g., London open) correlate with higher volatility. It enters a volatility breakout trade (break of Donchian channel) only during specific hours.",
            "features": ["hour", "donchian_channel", "ATR", "bollinger_bandwidth"],
            "lookahead_range": [20, 60], "dd_range": [0.25, 0.45], "complexity": "specialized",
            "ideal_regime": ["High Volatility", "Ranging"], "ideal_macro_env": ["Event-Driven"], "asset_class_suitability": ["Any"]
        },
        "SimpleMAChannel": {
            "description": "[TRENDING] Uses a channel created by two moving averages (20 and 50) with entries on pullbacks to the faster MA.",
            "features": ["EMA_20", "EMA_50", "RSI", "ATR"],
            "lookahead_range": [40, 120], "dd_range": [0.15, 0.3], "complexity": "low",
            "ideal_regime": ["Weak Trending", "Strong Trending"], "asset_class_suitability": ["Any"], "ideal_macro_env": ["Any"]
        },
        "SmoothedDynamicSR": {
            "description": "[HYBRID/PRICE ACTION] Combines the clarity of Heikin-Ashi charts with dynamic support and resistance (Bollinger Bands). A reversal of HA candle color at the bands triggers an entry.",
            "features": ["ha_color", "bollinger_upper", "bollinger_lower", "ATR"],
            "lookahead_range": [60, 150], "dd_range": [0.10, 0.20], "complexity": "medium",
            "ideal_regime": ["Ranging", "Weak Trending"], "ideal_macro_env": ["Neutral"], "asset_class_suitability": ["Forex Majors", "Indices"]
        },
        "SmoothedMACDCrossover": {
            "description": "[HYBRID/TREND] Uses Heikin-Ashi charts to filter out market noise and enters on a clear MACD crossover. The signal is only taken after a sequence of at least two same-colored HA bricks, confirming the trend direction.",
            "features": ["ha_color", "ha_streak", "MACD_line", "MACD_signal", "ADX"],
            "lookahead_range": [80, 200], "dd_range": [0.15, 0.25], "complexity": "medium",
            "ideal_regime": ["Strong Trending"], "ideal_macro_env": ["Any"], "asset_class_suitability": ["Indices", "Equities"]
        },
        "StanWeinsteinBreakout": {
            "description": "[BREAKOUT] A long-term trend-following strategy that enters on breakouts from consolidation, confirmed by volume.",
            "features": ["SMA_30_weekly", "volume", "relative_strength"],
            "lookahead_range": [100, 200], "dd_range": [0.2, 0.4], "complexity": "high",
            "ideal_regime": ["Strong Trending"], "ideal_macro_env": ["Risk-On"], "asset_class_suitability": ["Equities", "Indices"]
        },
        "StochasticScalp": {
            "description": "[SCALPING] A high-frequency strategy using the Stochastic for overbought/oversold entry signals confirmed by short-term EMA trend.",
            "features": ["stoch_k", "stoch_d", "EMA_20", "MACD_hist_slope", "candle_body_to_range_ratio"],
            "lookahead_range": [10, 40], "dd_range": [0.05, 0.20], "complexity": "low",
            "ideal_regime": ["Weak Trending", "Strong Trending"], "asset_class_suitability": ["Forex Majors", "Indices"], "ideal_macro_env": ["Any"]
        },
        "StructureSwingFailure": {
            "description": "[HYBRID/MARKET STRUCTURE] A reversal strategy that looks for a failure to create a new higher high or lower low (a liquidity grab). It enters on the subsequent break of the *previous* market structure point (CHoCH), confirmed by a spike in volume.",
            "features": ["liquidity_grab_up", "liquidity_grab_down", "choch_up_signal", "choch_down_signal", "volume", "MACD_hist"],
            "lookahead_range": [50, 140], "dd_range": [0.20, 0.30], "complexity": "high",
            "ideal_regime": ["Any"], "ideal_macro_env": ["Any"], "asset_class_suitability": ["Any"]
        },
        "SupportResistanceRSI": {
            "description": "[HYBRID/SR] Combines support/resistance levels with RSI confirmation for entries.",
            "features": ["support_level_20", "resistance_level_20", "RSI", "ADX"],
            "lookahead_range": [40, 120], "dd_range": [0.15, 0.3], "complexity": "low",
            "ideal_regime": ["Ranging", "Weak Trending"], "asset_class_suitability": ["Any"], "ideal_macro_env": ["Neutral"]
        },
        "TrendFilterScalper": {
            "description": "[HYBRID/SCALPING] Uses higher timeframe trend as filter for intraday scalping signals.",
            "features": ["EMA_20", "DAILY_ctx_Trend", "stoch_k", "ATR"],
            "lookahead_range": [10, 30], "dd_range": [0.05, 0.15], "complexity": "low",
            "ideal_regime": ["Weak Trending", "Strong Trending"], "asset_class_suitability": ["Forex Majors", "Indices"], "ideal_macro_env": ["Any"]
        },
        "TrendPullback": {
            "description": "[TRENDING] Enters on pullbacks during a confirmed trend, using market structure and statistical momentum.",
            "features": ["bos_up_signal", "bos_down_signal", "market_mode", "RSI_zscore", "DAILY_ctx_Trend"],
            "lookahead_range": [50, 150], "dd_range": [0.15, 0.3], "complexity": "medium",
            "ideal_regime": ["Strong Trending", "Weak Trending"], "asset_class_suitability": ["Forex Majors", "Indices", "Commodities"], "ideal_macro_env": ["Risk-On", "Risk-Off"]
        },
        "VWAPMomentum": {
            "description": "[TRENDING/INTRADAY] A trend-following strategy that uses VWAP as a dynamic filter. Enters on pullbacks to a rising VWAP.",
            "features": ["vwap_slope", "price_vs_vwap_sign", "DAILY_ctx_LinRegSlope", "ADX", "momentum_10_slope_acceleration", "volume"],
            "lookahead_range": [40, 120], "dd_range": [0.15, 0.3], "complexity": "medium",
            "ideal_regime": ["Weak Trending", "Strong Trending"], "ideal_macro_env": ["Neutral", "Risk-On"], "asset_class_suitability": ["Indices", "Equities"]
        },
        "VWAPReversion": {
            "description": "[RANGING/INTRADAY] A classic mean-reversion strategy that enters when the price deviates significantly from the daily VWAP.",
            "features": ["price_to_vwap", "ADX", "RSI", "bollinger_bandwidth", "hour", "DAILY_ctx_Trend"],
            "lookahead_range": [20, 60], "dd_range": [0.1, 0.2], "complexity": "medium",
            "ideal_regime": ["Ranging"], "ideal_macro_env": ["Neutral"], "asset_class_suitability": ["Indices", "Equities"]
        },
        "VWAP_Crossover": {
            "description": "[HYBRID/VOLUME] A volume-based trend strategy that uses a price crossover of the daily VWAP. A crossover is confirmed by an increase in volume, validating the signal.",
            "features": ["price_vs_vwap_sign", "volume", "ADX", "vwap_slope"],
            "lookahead_range": [50, 130], "dd_range": [0.15, 0.30], "complexity": "medium",
            "ideal_regime": ["Weak Trending", "Strong Trending"], "ideal_macro_env": ["Any"], "asset_class_suitability": ["Equities", "Indices"]
        },
        "VolatilityAdjustedMA": {
            "description": "[HYBRID/VOLATILITY] Adjusts moving average entries based on current volatility regime.",
            "features": ["EMA_50", "ATR", "bollinger_bandwidth", "market_volatility_index"],
            "lookahead_range": [50, 150], "dd_range": [0.15, 0.3], "complexity": "medium",
            "ideal_regime": ["Any"], "asset_class_suitability": ["Any"], "ideal_macro_env": ["Any"]
        },
        "VolatilityExpansionBreakout": {
            "description": "[BREAKOUT] Enters on strong breakouts that occur after a period of low-volatility consolidation (Bollinger Squeeze).",
            "features": ["bollinger_bandwidth", "bollinger_squeeze", "ATR", "market_volatility_index", "DAILY_ctx_Trend"],
            "lookahead_range": [70, 140], "dd_range": [0.2, 0.4], "complexity": "medium",
            "ideal_regime": ["Low Volatility", "High Volatility"], "asset_class_suitability": ["Any"], "ideal_macro_env": ["Event-Driven", "Neutral"]
        },
        "VolumeBreakout": {
            "description": "[BREAKOUT] Capitalizes on price breaking through established support or resistance, confirmed by a significant volume spike.",
            "features": ["support_level_20", "resistance_level_20", "volume_spike", "ATR", "ADX"],
            "lookahead_range": [40, 120], "dd_range": [0.20, 0.40], "complexity": "medium",
            "ideal_regime": ["High Volatility", "Strong Trending"], "asset_class_suitability": ["Equities", "Indices", "Crypto"], "ideal_macro_env": ["Any"]
        }
    }

    if not os.path.exists(playbook_path):
        logger.warning(f"'strategy_playbook.json' not found. Seeding a new one with default strategies at: {playbook_path}")
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

        # Update playbook with missing strategies or missing complexity field
        missing_keys = [k for k in DEFAULT_PLAYBOOK if k not in playbook]
        updated_count = 0
        for strategy_name, strategy_data in playbook.items():
            if 'complexity' not in strategy_data and strategy_name in DEFAULT_PLAYBOOK:
                playbook[strategy_name]['complexity'] = DEFAULT_PLAYBOOK[strategy_name].get('complexity', 'medium')
                updated_count += 1
        
        if missing_keys:
            logger.info(f"Updating playbook with {len(missing_keys)} new default strategies...")
            for k in missing_keys:
                playbook[k] = DEFAULT_PLAYBOOK[k]
        
        if missing_keys or updated_count > 0:
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

def run_single_instance(fallback_config: Dict, framework_history: Dict, playbook: Dict, nickname_ledger: Dict, directives: List[Dict], api_interval_seconds: int):
    MODEL_QUALITY_THRESHOLD = 0.05
    run_timestamp_str = datetime.now().strftime("%Y%m%d-%H%M%S")
    gemini_analyzer, api_timer = GeminiAnalyzer(), APITimer(interval_seconds=api_interval_seconds)

    current_config_dict = fallback_config.copy()
    current_config_dict['run_timestamp'] = run_timestamp_str
    # This 'temp_config' is used for pre-AI-analysis stages like data loading and feature engineering.
    # It contains the base parameters that will determine the cache validity.
    temp_config = ConfigModel(**{**current_config_dict, 'nickname': 'init', 'run_timestamp': 'init'})
    
    data_loader = DataLoader(temp_config)
    all_files = [f for f in os.listdir(current_config_dict['BASE_PATH']) if f.endswith(('.csv', '.txt')) and re.match(r'^[A-Z0-9]+_[A-Z0-9]+', f)]
    if not all_files: logger.critical("No data files found in base path. Exiting."); return
    data_by_tf, detected_timeframes = data_loader.load_and_parse_data(all_files)
    if not data_by_tf: return

    tf_roles = determine_timeframe_roles(detected_timeframes)
    
    # --- V209: Feature Caching Logic ---
    full_df = None

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
        fe = FeatureEngineer(temp_config, tf_roles)
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
    # --- End V209 Caching Logic ---
    
    macro_context = get_macro_context_data()

    logger.info("  - Calculating asset correlation matrix for AI context...")
    pivot_df = full_df.pivot(columns='Symbol', values='Close').ffill().dropna(how='all', axis=1)
    if pivot_df.shape[1] > 1:
        corr_matrix = pivot_df.corr()
        strong_correlations = corr_matrix[abs(corr_matrix) > 0.6]
        correlation_summary_for_ai = strong_correlations.to_json(indent=2)
        logger.info("  - Correlation summary generated.")
    else:
        correlation_summary_for_ai = "{}"
        logger.warning("  - Not enough assets to generate a meaningful correlation matrix for the AI.")

    summary_df = full_df.reset_index()
    assets = summary_df['Symbol'].unique().tolist()
    data_summary = {
        'assets_detected': assets,
        'time_range': {'start': summary_df['Timestamp'].min().isoformat(), 'end': summary_df['Timestamp'].max().isoformat()},
        'timeframes_used': tf_roles,
        'asset_statistics': {asset: {'avg_atr': round(full_df[full_df['Symbol'] == asset]['ATR'].mean(), 5), 'avg_adx': round(full_df[full_df['Symbol'] == asset]['ADX'].mean(), 2), 'trending_pct': f"{round(full_df[full_df['Symbol'] == asset]['market_regime'].mean() * 100, 1)}%"} for asset in assets}
    }

    script_name = os.path.basename(__file__) if '__file__' in locals() else fallback_config["REPORT_LABEL"]
    version_label = script_name.replace(".py", "")
    health_report, _ = perform_strategic_review(framework_history, fallback_config['DIRECTIVES_FILE_PATH'])

    try:
        avg_adx = np.mean([s['avg_adx'] for s in data_summary['asset_statistics'].values()])
        avg_trending_pct = np.mean([float(s['trending_pct'].strip('%')) for s in data_summary['asset_statistics'].values()])
        if avg_adx > 25 and avg_trending_pct > 60: diagnosed_regime = "Strong Trending"
        elif avg_adx > 20 and avg_trending_pct > 40: diagnosed_regime = "Weak Trending"
        else: diagnosed_regime = "Ranging"
    except (ValueError, TypeError): diagnosed_regime = "Ranging"

    regime_champions = {}
    if os.path.exists(temp_config.REGIME_CHAMPIONS_FILE_PATH):
        try:
            with open(temp_config.REGIME_CHAMPIONS_FILE_PATH, 'r') as f: regime_champions = json.load(f)
        except (json.JSONDecodeError, IOError): logger.warning("Could not read regime champions file.")
    
    ai_setup = api_timer.call(gemini_analyzer.get_initial_run_setup, version_label, nickname_ledger, framework_history, playbook, health_report, directives, data_summary, diagnosed_regime, regime_champions, correlation_summary_for_ai, macro_context)
    if not ai_setup: logger.critical("AI-driven setup failed. Exiting."); return

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
        logger.critical("--- FATAL PRE-CYCLE CONFIGURATION ERROR ---")
        logger.critical("The parameters suggested by the AI are invalid and failed Pydantic validation.")
        logger.critical(f"Validation Error details:\n{e}")
        logger.critical(f"Final dictionary that failed validation:\n{json.dumps(current_config_dict, indent=2, default=str)}")
        logger.critical("The run will be terminated. Please check the AI's logic or the fallback configuration.")
        return
    
    if not config.selected_features and config.strategy_name in playbook:
        config.selected_features = playbook[config.strategy_name].get("features", [])
        if not config.selected_features: logger.critical(f"FATAL: No features for '{config.strategy_name}'."); return

    file_handler = RotatingFileHandler(config.LOG_FILE_PATH, maxBytes=5*1024*1024, backupCount=2)
    file_handler.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
    logger.addHandler(file_handler)
    logger.info(f"--- Run Initialized: {config.nickname} | Strategy: {config.strategy_name} ---")

    all_available_features = [c for c in full_df.columns if c not in ['Open','High','Low','Close','RealVolume','Symbol','Timestamp','primary_model_signal','target']]
    train_window, forward_gap = pd.to_timedelta(config.TRAINING_WINDOW), pd.to_timedelta(config.FORWARD_TEST_GAP)
    test_start_date = full_df.index.min() + train_window + forward_gap
    retraining_dates = pd.date_range(start=test_start_date, end=full_df.index.max(), freq=_sanitize_frequency_string(config.RETRAINING_FREQUENCY))

    if retraining_dates.empty: logger.critical("Cannot proceed: No valid retraining dates could be determined. The total data length may be too short for the specified training window."); return

    aggregated_trades, aggregated_equity_curve = pd.DataFrame(), pd.Series([config.INITIAL_CAPITAL])
    in_run_historical_cycles, aggregated_daily_dd_reports = [], []
    shap_history, all_optuna_trials = defaultdict(list), []
    last_equity, quarantine_list = config.INITIAL_CAPITAL, []
    probationary_strategy, consecutive_failures_on_probation = None, 0
    run_configs_and_metrics = []
    last_successful_pipeline, last_successful_threshold, last_successful_features, last_successful_is_gnn = None, None, None, False
    
    cycle_num, cycle_retry_count = 0, 0
    while cycle_num < len(retraining_dates):
        period_start_date = retraining_dates[cycle_num]
        logger.info(f"\n--- Starting Cycle [{cycle_num + 1}/{len(retraining_dates)}] ---")
        cycle_start_time = time.time()

        train_end = period_start_date - forward_gap
        train_start = train_end - pd.to_timedelta(config.TRAINING_WINDOW)
        test_end = period_start_date + pd.tseries.frequencies.to_offset(_sanitize_frequency_string(config.RETRAINING_FREQUENCY))

        df_train_raw = full_df.loc[train_start:train_end].copy()
        df_test = full_df.loc[period_start_date:min(test_end, full_df.index.max())].copy()
        
        if df_train_raw.empty or df_test.empty: cycle_num += 1; continue

        strategy_details = playbook.get(config.strategy_name, {})
        fe = FeatureEngineer(config, tf_roles) # Use final 'config' here
        
        if strategy_details.get("requires_transformer"):
            df_train_labeled = df_train_raw 
        elif strategy_details.get("requires_meta_labeling"):
            df_train_labeled = fe.label_meta_outcomes(df_train_raw, config.LOOKAHEAD_CANDLES)
        else:
            df_train_labeled = fe.label_outcomes(df_train_raw, config.LOOKAHEAD_CANDLES)
        
        best_objective_score = -1.0
        optimization_path = [] 
        
        if not strategy_details.get("requires_transformer") and not check_label_quality(df_train_labeled, config.LABEL_MIN_EVENT_PCT):
            logger.critical(f"!! MODEL TRAINING SKIPPED !! Un-trainable labels. Retry {cycle_retry_count+1}/{config.MAX_TRAINING_RETRIES_PER_CYCLE}.")
            cycle_retry_count += 1
        else:
            config.selected_features = [f for f in config.selected_features if f in all_available_features]
            trainer = ModelTrainer(config)
            train_result = trainer.train(df_train_labeled, config.selected_features, strategy_details)
            
            if trainer.study:
                all_optuna_trials.extend([{'params': t.params, 'value': t.value} for t in trainer.study.trials if t.value is not None])
                optimization_path = [t.value for t in trainer.study.trials if t.value is not None] 
            best_objective_score = trainer.study.best_value if trainer.study and trainer.study.best_value is not None else -1

            if not train_result or (not trainer.is_transformer_model and best_objective_score < MODEL_QUALITY_THRESHOLD):
                logger.critical(f"!! MODEL QUALITY GATE FAILED !! Score ({best_objective_score:.3f}) < Threshold ({MODEL_QUALITY_THRESHOLD}). Retry {cycle_retry_count+1}/{config.MAX_TRAINING_RETRIES_PER_CYCLE}.")
                cycle_retry_count += 1
            else:
                pipeline, threshold = train_result
                cycle_retry_count = 0 
                last_successful_pipeline, last_successful_threshold, last_successful_features, last_successful_is_gnn = pipeline, threshold, config.selected_features, trainer.is_gnn_model
                
                if trainer.shap_summary is not None:
                    for feature, row in trainer.shap_summary.iterrows():
                        shap_history[feature].append(round(row['SHAP_Importance'], 4))
                
                if trainer.is_transformer_model:
                    X_test = df_test[config.selected_features].copy().fillna(0)
                    X_test_seq = []
                    seq_len = 30 
                    for i in range(len(X_test) - seq_len):
                        X_test_seq.append(X_test.iloc[i:i+seq_len].values)
                    
                    if X_test_seq:
                        X_test_seq = torch.tensor(np.array(X_test_seq), dtype=torch.float32)
                        with torch.no_grad():
                            predictions = pipeline(X_test_seq).numpy().flatten()
                        df_test.loc[:, 'predicted_price'] = np.nan
                        df_test.iloc[seq_len:, df_test.columns.get_loc('predicted_price')] = predictions
                elif trainer.is_gnn_model:
                    X_test = trainer._get_gnn_embeddings_for_test(df_test)
                    if not X_test.empty:
                       pass 
                else: 
                    X_test = df_test[config.selected_features].copy().fillna(0)
                    if not X_test.empty:
                        probs = pipeline.predict_proba(X_test)
                        if strategy_details.get("requires_meta_labeling"):
                            df_test.loc[:, ['prob_0', 'prob_1']] = probs
                        else:
                            df_test.loc[:, ['prob_short', 'prob_hold', 'prob_long']] = probs

                backtester = Backtester(config)
                trades, equity_curve, breaker_tripped, breaker_context, daily_dd_report = backtester.run_backtest_chunk(df_test, threshold, last_equity, strategy_details)
                aggregated_daily_dd_reports.append(daily_dd_report)
                cycle_status_msg = "Completed"

        if cycle_retry_count > config.MAX_TRAINING_RETRIES_PER_CYCLE:
            logger.error(f"!! STRATEGY FAILURE !! Exceeded max training retries for '{config.strategy_name}'.")
            
            if config.strategy_name not in quarantine_list:
                logger.critical(f"!! QUARANTINING STRATEGY: '{config.strategy_name}' due to repeated training failures.")
                quarantine_list.append(config.strategy_name)

            logger.info("  - Engaging AI for strategic intervention...")
            personal_best_config = max(run_configs_and_metrics, key=lambda x: x.get('final_metrics', {}).get('mar_ratio', -np.inf)) if run_configs_and_metrics else None
            
            intervention_suggestion = api_timer.call(gemini_analyzer.propose_strategic_intervention, in_run_historical_cycles[-2:], playbook, config.strategy_name, quarantine_list, personal_best_config)
            
            if intervention_suggestion:
                sanitized_suggestions = _sanitize_ai_suggestions(intervention_suggestion)
                config = ConfigModel(**{**config.model_dump(mode='json'), **sanitized_suggestions})
                logger.warning(f"  - Intervention successful. Switching to strategy: {config.strategy_name}")
                cycle_retry_count = 0 
                continue 
            else:
                logger.critical("  - Strategic intervention FAILED. Halting run."); break

        elif cycle_retry_count > 0:
            logger.info("  - Engaging AI for retry parameters...")
            ai_suggestions = api_timer.call(
                gemini_analyzer.analyze_cycle_and_suggest_changes,
                historical_results=in_run_historical_cycles,
                framework_history=framework_history,
                available_features=all_available_features,
                strategy_details=config.model_dump(),
                cycle_status="TRAINING_FAILURE",
                shap_history=shap_history,
                all_optuna_trials=all_optuna_trials,
                cycle_start_date=train_start.isoformat(),
                cycle_end_date=train_end.isoformat(),
                correlation_summary_for_ai=correlation_summary_for_ai,
                macro_context=macro_context
            )
            if ai_suggestions:
                config = ConfigModel(**{**config.model_dump(mode='json'), **_sanitize_ai_suggestions(ai_suggestions)})
            continue
        
        cycle_pnl = equity_curve.iloc[-1] - last_equity if not equity_curve.empty else 0
        
        trade_summary = {}
        if not trades.empty:
            losing_trades = trades[trades['PNL'] < 0]
            if not losing_trades.empty:
                trade_summary['avg_mae_loss'] = losing_trades['MAE'].mean()
                trade_summary['avg_mfe_loss'] = losing_trades['MFE'].mean()
        
        cycle_result = {"StartDate": period_start_date.date().isoformat(), "EndDate": test_end.date().isoformat(), "NumTrades": len(trades), "PNL": round(cycle_pnl, 2), "Status": "Circuit Breaker" if breaker_tripped else cycle_status_msg, "BestObjectiveScore": round(best_objective_score, 4), "trade_summary": trade_summary, "optimization_path": optimization_path}
        if breaker_tripped: cycle_result["BreakerContext"] = breaker_context
        in_run_historical_cycles.append(cycle_result)

        if not trades.empty:
            aggregated_trades = pd.concat([aggregated_trades, trades], ignore_index=True)
            aggregated_equity_curve = pd.concat([aggregated_equity_curve, equity_curve.iloc[1:]], ignore_index=True)
            last_equity = equity_curve.iloc[-1]
            run_configs_and_metrics.append({"final_params": config.model_dump(mode='json'), "final_metrics": PerformanceAnalyzer(config)._calculate_metrics(aggregated_trades, aggregated_equity_curve)})

        if breaker_tripped:
            if probationary_strategy == config.strategy_name:
                consecutive_failures_on_probation += 1
                if consecutive_failures_on_probation >= 2:
                    logger.critical(f"!! QUARANTINING STRATEGY: '{config.strategy_name}' due to repeated failures on probation.")
                    if config.strategy_name not in quarantine_list: quarantine_list.append(config.strategy_name)
                    personal_best_config = max(run_configs_and_metrics, key=lambda x: x.get('final_metrics', {}).get('mar_ratio', -np.inf)) if run_configs_and_metrics else None
                    intervention_suggestion = api_timer.call(gemini_analyzer.propose_strategic_intervention, in_run_historical_cycles[-2:], playbook, config.strategy_name, quarantine_list, personal_best_config)
                    if intervention_suggestion:
                        config = ConfigModel(**{**config.model_dump(mode='json'), **_sanitize_ai_suggestions(intervention_suggestion)})
                    else:
                        logger.error("  - Strategic intervention FAILED. Halting run."); break
                    probationary_strategy, consecutive_failures_on_probation = None, 0
            else:
                probationary_strategy, consecutive_failures_on_probation = config.strategy_name, 1
        
        cycle_status_for_ai = "PROBATION" if probationary_strategy else "COMPLETED"
        
        logger.info("-> Re-fetching external macroeconomic context for cycle analysis...")
        macro_context = get_macro_context_data()
        
        suggested_params = api_timer.call(
            gemini_analyzer.analyze_cycle_and_suggest_changes,
            historical_results=in_run_historical_cycles,
            framework_history=framework_history,
            available_features=all_available_features,
            strategy_details=config.model_dump(),
            cycle_status=cycle_status_for_ai,
            shap_history=shap_history,
            all_optuna_trials=all_optuna_trials,
            cycle_start_date=period_start_date.isoformat(),
            cycle_end_date=min(test_end, full_df.index.max()).isoformat(),
            correlation_summary_for_ai=correlation_summary_for_ai,
            macro_context=macro_context
        )
        if suggested_params:
            config = ConfigModel(**{**config.model_dump(mode='json'), **_sanitize_ai_suggestions(suggested_params)})

        cycle_num += 1
        logger.info(f"--- Cycle complete. PNL: ${cycle_pnl:,.2f} | Final Equity: ${last_equity:,.2f} | Time: {time.time() - cycle_start_time:.2f}s ---")

    pa = PerformanceAnalyzer(config)
    final_metrics = pa.generate_full_report(aggregated_trades, aggregated_equity_curve, in_run_historical_cycles, pd.DataFrame.from_dict(shap_history, orient='index', columns=[f'C{i+1}' for i in range(len(next(iter(shap_history.values()),[])))]).mean(axis=1).sort_values(ascending=False).to_frame('SHAP_Importance'), framework_history, aggregated_daily_dd_reports)
    run_summary = {"script_version": config.REPORT_LABEL, "nickname": config.nickname, "strategy_name": config.strategy_name, "run_start_ts": config.run_timestamp, "final_params": config.model_dump(mode='json'), "run_end_ts": datetime.now().strftime("%Y%m%d-%H%M%S"), "final_metrics": final_metrics, "cycle_details": in_run_historical_cycles}
    save_run_to_memory(config, run_summary, framework_history, diagnosed_regime)
    logger.removeHandler(file_handler); file_handler.close()
def main():
    CONTINUOUS_RUN_HOURS = 0; MAX_RUNS = 1
    fallback_config={
        "BASE_PATH": os.getcwd(), "REPORT_LABEL": f"ML_Framework_V{VERSION}",
        "strategy_name": "Meta_Labeling_Filter", "INITIAL_CAPITAL": 10000.0,
        "COMMISSION_PER_LOT": 3.5,
        "CONFIDENCE_TIERS": {
            'ultra_high': {'min': 0.8, 'risk_mult': 1.2, 'rr': 2.5},
            'high':       {'min': 0.7, 'risk_mult': 1.0, 'rr': 2.0},
            'standard':   {'min': 0.6, 'risk_mult': 0.8, 'rr': 1.5}
        },
        "BASE_RISK_PER_TRADE_PCT": 0.01,"RISK_CAP_PER_TRADE_USD": 500.0,
        "OPTUNA_TRIALS": 30,
        "TRAINING_WINDOW": '365D', "RETRAINING_FREQUENCY": '90D',
        "FORWARD_TEST_GAP": "1D", "LOOKAHEAD_CANDLES": 150, "TREND_FILTER_THRESHOLD": 25.0,
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
        "USE_FEATURE_CACHING": True, # V209: Added feature caching switch
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
    api_interval_seconds = 300
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

        nickname_ledger = load_nickname_ledger(bootstrap_config.NICKNAME_LEDGER_PATH)
        framework_history = load_memory(bootstrap_config.CHAMPION_FILE_PATH, bootstrap_config.HISTORY_FILE_PATH)
        directives = []
        if os.path.exists(bootstrap_config.DIRECTIVES_FILE_PATH):
            try:
                with open(bootstrap_config.DIRECTIVES_FILE_PATH, 'r') as f: directives = json.load(f)
                if directives: logger.info(f"Loaded {len(directives)} directive(s) for this run.")
            except (json.JSONDecodeError, IOError) as e: logger.error(f"Could not load directives file: {e}")

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

# End_To_End_Advanced_ML_Trading_Framework_PRO_V209_Linux.py