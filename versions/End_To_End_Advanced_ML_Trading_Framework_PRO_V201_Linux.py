# End_To_End_Advanced_ML_Trading_Framework_PRO_V201_Linux.py
#
# V201 UPDATE (Advanced Feature Engineering & Trade Management):
#   1. ADDED (Hawkes Process Volatility Feature): Integrated a Hawkes process to model price activity clustering. This is added as a new feature `volatility_hawkes`, providing the model with a sophisticated measure of market intensity.
#   2. ADDED (PCA Dimensionality Reduction): Applied Principal Component Analysis (PCA) to groups of correlated features (e.g., multiple RSI periods) to reduce noise and create more robust feature sets.
#   3. ADDED (Advanced Take-Profit Ladder): The Backtester now includes an optional, multi-level take-profit ladder, allowing for granular trade management by exiting positions in partial increments at progressively distant profit targets.
#
# V200 UPDATE (Meta-Learning & Optimization Analysis):
#   1. ADDED (Optimization Path Analysis): The framework now records the entire sequence of objective scores from each cycle's Optuna study. This "optimization path" reveals the difficulty and efficiency of the hyperparameter search.
#   2. ENHANCED (AI Meta-Analysis Prompt): The AI's core analysis prompt is now instructed to analyze the optimization path. It can distinguish between models found through a smooth, efficient search (indicating a good strategy-to-market fit) and those found after an erratic, difficult search (a warning sign of a noisy feature space or poor fit, even if the final score passed the quality gate).

# --- SCRIPT VERSION ---
VERSION = "201"
# --------------------

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
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import RobustScaler, MinMaxScaler, StandardScaler
from sklearn.utils.class_weight import compute_class_weight
from pydantic import BaseModel, DirectoryPath, confloat, conint, Field
from sklearn.ensemble import IsolationForest
from sklearn.decomposition import PCA


# --- DIAGNOSTICS & LOGGING SETUP ---
logger = logging.getLogger("ML_Trading_Framework")

# --- GNN Specific Imports (requires PyTorch, PyG) ---
try:
    import torch
    import torch.nn.functional as F
    from torch_geometric.data import Data
    from torch_geometric.nn import GCNConv
    from torch.optim import Adam
    GNN_AVAILABLE = True
except ImportError:
    GNN_AVAILABLE = False
    class GCNConv: pass
    class Adam: pass
    class Data: pass
    def F(): pass
    torch = None

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
# 2. CONFIGURATION & VALIDATION
# =============================================================================
class ConfigModel(BaseModel):
    BASE_PATH: DirectoryPath; REPORT_LABEL: str; INITIAL_CAPITAL: confloat(gt=0)
    CONFIDENCE_TIERS: Dict[str, Dict[str, Any]]; BASE_RISK_PER_TRADE_PCT: confloat(gt=0, lt=1)
    SPREAD_PCTG_OF_ATR: confloat(ge=0); SLIPPAGE_PCTG_OF_ATR: confloat(ge=0)
    MAX_DD_PER_CYCLE: confloat(ge=0.05, lt=1.0) = 0.25; RISK_CAP_PER_TRADE_USD: confloat(gt=0)
    OPTUNA_TRIALS: conint(gt=0); TRAINING_WINDOW: str; RETRAINING_FREQUENCY: str
    FORWARD_TEST_GAP: str; LOOKAHEAD_CANDLES: conint(gt=0); CALCULATE_SHAP_VALUES: bool = True
    TREND_FILTER_THRESHOLD: confloat(gt=0) = 25.0; BOLLINGER_PERIOD: conint(gt=0) = 20
    STOCHASTIC_PERIOD: conint(gt=0) = 14; GNN_EMBEDDING_DIM: conint(gt=0) = 8
    GNN_EPOCHS: conint(gt=0) = 50; MIN_VOLATILITY_RANK: confloat(ge=0.0, le=1.0) = 0.1
    MAX_VOLATILITY_RANK: confloat(ge=0.0, le=1.0) = 0.9; selected_features: List[str]
    run_timestamp: str; strategy_name: str; nickname: str = ""
    analysis_notes: str = ""
    # Portfolio & Risk Parameters
    MAX_CONCURRENT_TRADES: conint(ge=1, le=20) = 3
    USE_PARTIAL_PROFIT: bool = False
    PARTIAL_PROFIT_TRIGGER_R: confloat(gt=0) = 1.5
    PARTIAL_PROFIT_TAKE_PCT: confloat(ge=0.1, le=0.9) = 0.5

    # --- NEW (V201): Advanced Feature & Trade Management ---
    HAWKES_KAPPA: confloat(gt=0) = 0.5
    USE_PCA_REDUCTION: bool = True
    PCA_N_COMPONENTS: conint(gt=1, le=10) = 3
    RSI_PERIODS_FOR_PCA: List[conint(gt=1)] = Field(default_factory=lambda: [5, 10, 15, 20, 25])
    USE_TP_LADDER: bool = True
    TP_LADDER_LEVELS_PCT: List[confloat(gt=0, lt=1)] = Field(default_factory=lambda: [0.25, 0.25, 0.25, 0.25])
    TP_LADDER_RISK_MULTIPLIERS: List[confloat(gt=0)] = Field(default_factory=lambda: [1.0, 2.0, 3.0, 4.0])

    # New parameter for V184
    MAX_TRAINING_RETRIES_PER_CYCLE: conint(ge=0) = 3
    # New parameters for V186
    anomaly_contamination_factor: confloat(ge=0.001, le=0.1) = 0.01
    # New parameters for V187
    USE_TIERED_RISK: bool = False
    RISK_PROFILE: str = 'Medium' # Options: 'Low', 'Medium', 'High'
    TIERED_RISK_CONFIG: Dict[int, Dict[str, Dict[str, Union[float, int]]]] = {}
    # New parameters for V191
    LABEL_MIN_RETURN_PCT: confloat(ge=0.0) = 0.001
    LABEL_MIN_EVENT_PCT: confloat(ge=0.01, le=0.5) = 0.02
    # --- NEW (V196): Broker and Margin Simulation ---
    CONTRACT_SIZE: confloat(gt=0) = 100000.0
    LEVERAGE: conint(gt=0) = 30
    MIN_LOT_SIZE: confloat(gt=0) = 0.01
    LOT_STEP: confloat(gt=0) = 0.01


    MODEL_SAVE_PATH: str = Field(default="", repr=False); PLOT_SAVE_PATH: str = Field(default="", repr=False)
    REPORT_SAVE_PATH: str = Field(default="", repr=False); SHAP_PLOT_PATH: str = Field(default="", repr=False)
    LOG_FILE_PATH: str = Field(default="", repr=False); CHAMPION_FILE_PATH: str = Field(default="", repr=False)
    HISTORY_FILE_PATH: str = Field(default="", repr=False); PLAYBOOK_FILE_PATH: str = Field(default="", repr=False)
    DIRECTIVES_FILE_PATH: str = Field(default="", repr=False); NICKNAME_LEDGER_PATH: str = Field(default="", repr=False)
    # --- FIX: V195 requires this new file path for the Regime Champions feature.
    REGIME_CHAMPIONS_FILE_PATH: str = Field(default="", repr=False)


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
        # --- FIX: V195 requires this new file path for the Regime Champions feature.
        self.REGIME_CHAMPIONS_FILE_PATH = os.path.join(results_dir, "regime_champions.json")
        
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

    def get_initial_run_setup(self, script_version: str, ledger: Dict, memory: Dict, playbook: Dict, health_report: Dict, directives: List[Dict], data_summary: Dict, diagnosed_regime: str, regime_champions: Dict) -> Dict:
        if not self.api_key_valid:
            logger.warning("No API key. Skipping AI-driven setup and using default config.")
            return {}

        logger.info("-> Performing Initial AI Analysis & Setup (Regime-Aware)...")
        logger.info(f"  - Diagnosed Market Regime: {diagnosed_regime}")

        available_playbook = {
            k: v for k, v in playbook.items()
            if not v.get("retired") and (GNN_AVAILABLE or not v.get("requires_gnn"))
        }
        if not GNN_AVAILABLE:
            logger.warning("GNN strategies filtered from playbook due to missing libraries.")

        regime_champion = regime_champions.get(diagnosed_regime)
        task_prompt = ""

        if regime_champion:
            champ_strat = regime_champion.get("strategy_name", "N/A")
            champ_mar = regime_champion.get("final_metrics", {}).get("mar_ratio", 0)
            logger.info(f"  - Found existing champion for '{diagnosed_regime}': '{champ_strat}' (MAR: {champ_mar:.2f})")
            task_prompt = (
                "**YOUR TASK: Choose your starting strategy.**\n\n"
                f"The current market regime has been diagnosed as **'{diagnosed_regime}'**. We have a proven champion for this exact regime.\n\n"
                f"**OPTION A: DEPLOY THE REGIME CHAMPION**\n"
                f"- **Strategy**: `{champ_strat}` (MAR Ratio: {champ_mar:.2f})\n"
                f"- This is the safe, data-driven choice. To select it, respond with: `{{\"action\": \"deploy_champion\"}}`\n\n"
                f"**OPTION B: EXPLORE A NEW STRATEGY**\n"
                f"- If you believe the subtle market context warrants a different approach, select a new strategy from the `STRATEGY PLAYBOOK`.\n"
                f"- Justify your choice in `analysis_notes`.\n"
                f"- If you choose this path, you must provide the full configuration (`strategy_name`, `selected_features`, `OPTUNA_TRIALS`, etc.) in the JSON response.\n"
            )
        else:
            logger.info(f"  - No existing champion found for '{diagnosed_regime}'. AI will select from playbook.")
            task_prompt = (
                "**YOUR TASK: Select and configure the optimal starting strategy.**\n\n"
                f"The current market regime has been diagnosed as **'{diagnosed_regime}'**. There is no proven champion for this regime yet.\n"
                "Your task is to analyze the market data and the playbook to propose the best initial configuration.\n"
                "You must provide the full configuration (`strategy_name`, `selected_features`, `OPTUNA_TRIALS`, etc.) in the JSON response.\n"
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
            "You are a Master Trading Strategist responsible for configuring a trading framework for its next run.\n\n"
            f"{task_prompt}\n"
            f"{nickname_prompt_part}\n"
            "**PARAMETER RULES & GUIDANCE (if exploring):**\n"
            "- `selected_features`: **Required.** Select 4-6 relevant features. Must include two context features (e.g., `DAILY_ctx_Trend`) unless GNN-based.\n"
            "- `RETRAINING_FREQUENCY`: Must be a string (e.g., '90D').\n"
            "- `OPTUNA_TRIALS`: An integer between 30-150.\n\n"
            "**OUTPUT FORMAT**: Respond ONLY with a single, valid JSON object.\n\n"
            "--- CONTEXT FOR YOUR DECISION ---\n\n"
            f"**1. MARKET DATA SUMMARY:**\n{json.dumps(self._sanitize_dict(data_summary), indent=2)}\n\n"
            f"**2. STRATEGY PLAYBOOK (Your options):**\n{json.dumps(self._sanitize_dict(available_playbook), indent=2)}\n\n"
            f"**3. FRAMEWORK MEMORY (All-Time Champion & Recent Runs):**\n{json.dumps(self._sanitize_dict(memory), indent=2)}\n"
        )

        response_text = self._call_gemini(prompt)
        suggestions = self._extract_json_from_response(response_text)

        if suggestions.get("action") == "deploy_champion" and regime_champion:
            logger.info("  - AI chose to deploy the proven regime champion.")
            champion_params = regime_champion.get("final_params", {})
            # Keep the nickname from the AI response, but use the champion's other params
            champion_params['nickname'] = suggestions.get('nickname')
            champion_params['analysis_notes'] = f"Deploying proven champion for '{diagnosed_regime}' regime."
            return champion_params

        if suggestions and "strategy_name" in suggestions:
            logger.info("  - Initial AI Analysis and Setup complete (AI chose to explore).")
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
        strategic_forecast: Optional[Dict] = None
    ) -> Dict:
        if not self.api_key_valid: return {}

        base_prompt_intro = "You are an expert trading model analyst. Your primary goal is to create a STABLE and PROFITABLE strategy by making intelligent, data-driven changes, using the comprehensive historical and performance context provided."
        json_response_format = "Respond ONLY with a valid JSON object containing your suggested keys and an `analysis_notes` key explaining your reasoning."

        # V200: Comprehensive Multi-faceted Analysis Prompt with Meta-Analysis
        analysis_points = [
            "--- COMPREHENSIVE META-ANALYSIS FRAMEWORK ---",
            "**1. Model Quality vs. PNL:** Analyze the `BestObjectiveScore` (model quality) against the final `PNL` for each cycle. Is there a correlation? A low score followed by a high PNL is a sign of LUCK, not skill. Prioritize stability.",
            "**2. Feature Drift (SHAP History):** Analyze the `shap_importance_history`. Are core features losing predictive power over time? A sudden collapse in a feature's importance signals a regime change and may require a new feature set.",
            "**3. Granular Exit Analysis (MAE/MFE):** Look at the `trade_summary`. High MAE (Maximum Adverse Excursion) on losing trades suggests poor entries. High MFE (Maximum Favorable Excursion) on losing trades suggests take-profit levels are too far away.",
            "**4. Hyperparameter Learning:** Review the `optuna_trial_history`. Are there parameter values (e.g., `max_depth` > 8) that consistently lead to failure? Learn from this history to avoid repeating mistakes.",
            "**5. Optimization Path Analysis:** Review the `optimization_path` in the most recent cycle's history. Did the model's score improve steadily and efficiently, or was it an erratic, difficult search? A difficult search is a strong warning sign about the current feature set's quality or its fit for the market, even if the final score was acceptable."
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
                "Your task is to synthesize all five analysis points into a coherent set of changes. Propose a new configuration that improves robustness and profitability. For example, if the optimization path was erratic and SHAP shows a feature is losing importance, a feature set change is strongly warranted."
            )

        # Sanitize Optuna trials to keep prompt size manageable
        optuna_summary = {}
        if all_optuna_trials:
            sorted_trials = sorted(all_optuna_trials, key=lambda x: x.get('value', -99), reverse=True)
            optuna_summary = {"best_5_trials": sorted_trials[:5], "worst_5_trials": sorted_trials[-5:]}

        data_context = (
            f"--- DATA FOR YOUR ANALYSIS ---\n\n"
            f"**A. CURRENT RUN - CYCLE-BY-CYCLE HISTORY:**\n{json.dumps(self._sanitize_dict(historical_results), indent=2)}\n\n"
            f"**B. FEATURE IMPORTANCE HISTORY (SHAP values over time):**\n{json.dumps(self._sanitize_dict(shap_history), indent=2)}\n\n"
            f"**C. HYPERPARAMETER HISTORY (Sample from Optuna Trials):**\n{json.dumps(self._sanitize_dict(optuna_summary), indent=2)}\n\n"
            f"**D. CURRENT STRATEGY & AVAILABLE FEATURES:**\n`strategy_name`: {strategy_details.get('strategy_name')}\n`available_features`: {available_features}\n"
        )
        prompt = f"{base_prompt_intro}\n\n**YOUR TASK:**\n{task_guidance}\n\n**ANALYTICAL FRAMEWORK (Address these points in your reasoning):**\n" + "\n".join(analysis_points) + f"\n\n{json_response_format}\n\n{data_context}"
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
                f"   - Propose a brand new strategy from the available playbook.\n"
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
                "2.  **Select a NEW STRATEGY:** You **MUST** choose a *different* strategy from the available playbook that is NOT in the quarantine list.\n"
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
            "    - If you believe a **different strategy is better suited** to the current market regime, respond with the JSON configuration for that new strategy (just the strategy name and a **small, targeted feature set of 4-6 features** from its playbook defaults).\n"
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
# 4. DATA LOADER & 5. FEATURE ENGINEERING
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
    TIMEFRAME_MAP = {'M1': 1,'M5': 5,'M15': 15,'M30': 30,'H1': 60,'H4': 240,'D1': 1440, 'DAILY': 1440}
    ANOMALY_FEATURES = [
        'ATR', 'bollinger_bandwidth', 'RSI', 'RealVolume', 'candle_body_size',
        'pct_change', 'candle_body_size_vs_atr', 'atr_vs_daily_atr'
    ]

    def __init__(self, config: ConfigModel, timeframe_roles: Dict[str, str]):
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
        """
        Calculates the intensity of events using a Hawkes process with an exponential kernel.
        """
        if not isinstance(data, pd.Series) or data.isnull().all():
            logger.warning("Hawkes process received invalid data; returning zeros.")
            return pd.Series(np.zeros(len(data)), index=data.index)

        assert kappa > 0.0
        alpha = np.exp(-kappa)
        arr = data.to_numpy()
        output = np.zeros(len(data))
        output[:] = np.nan
        for i in range(1, len(data)):
            if np.isnan(output[i - 1]):
                output[i] = arr[i]
            else:
                output[i] = output[i - 1] * alpha + arr[i]

        return pd.Series(output, index=data.index) * kappa

    def _apply_pca_to_features(self, df: pd.DataFrame, feature_prefix: str, n_components: int) -> pd.DataFrame:
        """
        Applies PCA to a subset of features and returns the principal components.
        """
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

        pc_df = pd.DataFrame(data=principal_components,
                             columns=[f'PCA_{feature_prefix}_{i}' for i in range(n_components)],
                             index=pca_features.index)

        return pc_df

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
        g['bollinger_bandwidth'] = (g['bollinger_upper'] - g['bollinger_lower']) / middle_band.replace(0,np.nan); return g

    def _calculate_stochastic(self, g:pd.DataFrame, period:int) -> pd.DataFrame:
        low_min=g['Low'].rolling(window=period).min();high_max=g['High'].rolling(window=period).max()
        g['stoch_k']=100*(g['Close']-low_min)/(high_max-low_min).replace(0,np.nan);g['stoch_d']=g['stoch_k'].rolling(window=3).mean();return g

    def _calculate_momentum(self, g:pd.DataFrame) -> pd.DataFrame:
        g['momentum_10'] = g['Close'].diff(10)
        g['momentum_20'] = g['Close'].diff(20)
        g['pct_change'] = g['Close'].pct_change()
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
        g['wick_to_body_ratio'] = (g['upper_wick'] + g['lower_wick']) / g['candle_body_size'].replace(0, 1e-9)
        g['is_doji'] = (g['candle_body_size'] / g['ATR'].replace(0,1)).lt(0.1).astype(int)
        g['is_engulfing'] = ((g['candle_body_size'] > abs(g['Close'].shift() - g['Open'].shift())) & (np.sign(g['Close']-g['Open']) != np.sign(g['Close'].shift()-g['Open'].shift()))).astype(int)
        g['candle_body_size_vs_atr'] = g['candle_body_size'] / g['ATR'].replace(0, 1)
        return g

    def _calculate_indicator_dynamics(self, g: pd.DataFrame, period: int = 5) -> pd.DataFrame:
        def get_slope(series):
            if len(series) < 2 or series.isnull().all():
                return np.nan
            # Ensure series is float for polyfit
            series_float = series.fillna(method='ffill').fillna(method='bfill').astype(float)
            if series_float.isnull().all():
                 return np.nan
            return np.polyfit(np.arange(len(series_float)), series_float, 1)[0]

        g['RSI_slope'] = g['RSI'].rolling(window=period, min_periods=period).apply(get_slope, raw=False)
        g['momentum_10_slope'] = g['momentum_10'].rolling(window=period, min_periods=period).apply(get_slope, raw=False)
        return g

    def _calculate_htf_features(self,df:pd.DataFrame,p:str,s:int,a:int)->pd.DataFrame:
        tf_id = p.upper()
        results=[]
        # This function iterates internally, so it's safe for single or multi-symbol DFs
        for symbol,group in df.groupby('Symbol'):
            g=group.copy();sma=g['Close'].rolling(s,min_periods=s).mean();atr=(g['High']-g['Low']).rolling(a,min_periods=a).mean();trend=np.sign(g['Close']-sma)
            temp_df=pd.DataFrame(index=g.index)
            temp_df[f'{tf_id}_ctx_SMA']=sma
            temp_df[f'{tf_id}_ctx_ATR']=atr
            temp_df[f'{tf_id}_ctx_Trend']=trend
            shifted_df=temp_df.shift(1);shifted_df['Symbol']=symbol;results.append(shifted_df)
        if not results: return pd.DataFrame()
        return pd.concat(results).reset_index()

    def _calculate_base_tf_native(self, g:pd.DataFrame)->pd.DataFrame:
        g_out=g.copy();lookback=14
        g_out['ATR']=(g['High']-g['Low']).rolling(lookback).mean();delta=g['Close'].diff();gain=delta.where(delta > 0,0).ewm(com=lookback-1,adjust=False).mean()
        loss=-delta.where(delta < 0,0).ewm(com=lookback-1,adjust=False).mean()
        g_out['RSI']=100-(100/(1+(gain/loss.replace(0,1e-9))))

        if self.config.USE_PCA_REDUCTION:
            for p in self.config.RSI_PERIODS_FOR_PCA:
                gain_p = delta.where(delta > 0, 0).ewm(com=p - 1, adjust=False).mean()
                loss_p = -delta.where(delta < 0, 0).ewm(com=p - 1, adjust=False).mean()
                g_out[f'rsi_{p}'] = 100 - (100 / (1 + (gain_p / loss_p.replace(0, 1e-9))))

        g_out=self._calculate_adx(g_out,lookback)
        g_out=self._calculate_bollinger_bands(g_out,self.config.BOLLINGER_PERIOD)
        g_out=self._calculate_stochastic(g_out,self.config.STOCHASTIC_PERIOD)
        g_out = self._calculate_momentum(g_out)
        g_out = self._calculate_seasonality(g_out)
        g_out = self._calculate_candle_microstructure(g_out)
        g_out = self._calculate_indicator_dynamics(g_out)
        g_out['hour'] = g_out.index.hour;g_out['day_of_week'] = g_out.index.dayofweek
        g_out['market_regime']=np.where(g_out['ADX']>self.config.TREND_FILTER_THRESHOLD,1,0)
        sma_fast = g_out['Close'].rolling(window=20).mean(); sma_slow = g_out['Close'].rolling(window=50).mean()
        signal_series = pd.Series(np.where(sma_fast > sma_slow, 1.0, -1.0), index=g_out.index)
        g_out['primary_model_signal'] = signal_series.diff().fillna(0)
        g_out['market_volatility_index'] = g_out['ATR'].rolling(100).rank(pct=True)
        g_out['close_fracdiff'] = self._fractional_differentiation(g_out['Close'], d=0.5)

        g_out['log_returns'] = np.log(g_out['Close'] / g_out['Close'].shift(1))
        g_out['abs_log_returns'] = g_out['log_returns'].abs().fillna(0)
        g_out['volatility_hawkes'] = self.hawkes_process(g_out['abs_log_returns'], kappa=self.config.HAWKES_KAPPA)

        return g_out

    def _calculate_relative_performance(self, df: pd.DataFrame) -> pd.DataFrame:
        if 'pct_change' not in df.columns:
            logger.warning("  - 'pct_change' not found, cannot calculate relative performance.")
            return df
        # This calculation requires data from all symbols at a given timestamp
        df['avg_market_pct_change'] = df.groupby(level=0)['pct_change'].transform('mean')
        df['relative_performance'] = df['pct_change'] - df['avg_market_pct_change']
        return df

    def _process_single_symbol_stack(self, data_by_tf_single_symbol: Dict[str, pd.DataFrame]) -> pd.DataFrame:
        """Helper function to process all features for a single symbol."""
        base_tf, medium_tf, high_tf = self.roles['base'], self.roles['medium'], self.roles['high']
        df_base = data_by_tf_single_symbol[base_tf]

        # 1. Calculate base features for the single symbol's base timeframe data
        df_base_featured = self._calculate_base_tf_native(df_base)
        df_merged = df_base_featured.reset_index()

        # 2. Merge with Higher Timeframe (HTF) context data
        if medium_tf and medium_tf in data_by_tf_single_symbol and not data_by_tf_single_symbol[medium_tf].empty:
            df_medium_ctx = self._calculate_htf_features(data_by_tf_single_symbol[medium_tf], medium_tf, 50, 14)
            if not df_medium_ctx.empty:
                df_merged = pd.merge_asof(df_merged.sort_values('Timestamp'), df_medium_ctx.sort_values('Timestamp'), on='Timestamp', by='Symbol', direction='backward')

        if high_tf and high_tf in data_by_tf_single_symbol and not data_by_tf_single_symbol[high_tf].empty:
            df_high_ctx = self._calculate_htf_features(data_by_tf_single_symbol[high_tf], high_tf, 20, 14)
            if not df_high_ctx.empty:
                df_merged = pd.merge_asof(df_merged.sort_values('Timestamp'), df_high_ctx.sort_values('Timestamp'), on='Timestamp', by='Symbol', direction='backward')

        df_final = df_merged.set_index('Timestamp').copy()
        del df_merged, df_base_featured # Free up memory

        # 3. Add context-dependent features
        if medium_tf:
            tf_id = medium_tf.upper()
            df_final[f'adx_x_{tf_id}_trend'] = df_final['ADX'] * df_final.get(f'{tf_id}_ctx_Trend', 0)
        if high_tf:
            tf_id = high_tf.upper()
            df_final[f'atr_x_{tf_id}_trend'] = df_final['ATR'] * df_final.get(f'{tf_id}_ctx_Trend', 0)
            df_final['atr_vs_daily_atr'] = df_final['ATR'] / df_final.get(f'{tf_id}_ctx_ATR', 1).replace(0, 1)

        # 4. Apply PCA Dimensionality Reduction
        if self.config.USE_PCA_REDUCTION:
            logger.info(f"    - Applying PCA to RSI features for symbol.")
            rsi_pc_df = self._apply_pca_to_features(df_final, 'rsi_', self.config.PCA_N_COMPONENTS)
            if not rsi_pc_df.empty:
                df_final = df_final.join(rsi_pc_df)
                cols_to_drop = [c for c in df_final.columns if c.startswith('rsi_')]
                df_final.drop(columns=cols_to_drop, inplace=True)
                logger.info(f"    - Added {len(rsi_pc_df.columns)} PCA features and dropped {len(cols_to_drop)} intermediate RSI features.")

        # 5. Generate anomaly scores (can be done per-symbol)
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
            # Create a dictionary of dataframes containing only the current symbol
            symbol_specific_data = {tf: df[df['Symbol'] == symbol].copy() for tf, df in data_by_tf.items()}
            
            processed_symbol_df = self._process_single_symbol_stack(symbol_specific_data)
            del symbol_specific_data # Free memory
            
            if not processed_symbol_df.empty:
                all_symbols_processed_dfs.append(processed_symbol_df)

        if not all_symbols_processed_dfs:
            logger.critical("Feature engineering resulted in no processable data across all symbols.")
            return pd.DataFrame()
        
        logger.info("  - Concatenating data for all symbols...")
        final_df = pd.concat(all_symbols_processed_dfs, sort=False).sort_index()
        del all_symbols_processed_dfs # Free memory

        # Calculations that require the full multi-symbol dataframe
        logger.info("  - Calculating cross-symbol features (relative performance)...")
        final_df = self._calculate_relative_performance(final_df)

        logger.info("  - Applying final data shift and cleaning...")
        feature_cols = [c for c in final_df.columns if c not in ['Open','High','Low','Close','RealVolume','Symbol']]
        # Use groupby with shift to prevent data leakage between symbols at concatenation points
        final_df[feature_cols] = final_df.groupby('Symbol', sort=False)[feature_cols].shift(1)
        
        final_df.replace([np.inf,-np.inf],np.nan,inplace=True)
        final_df.dropna(inplace=True)

        logger.info(f"  - Merged data and created features. Final dataset shape: {final_df.shape}")
        logger.info("[SUCCESS] Feature engineering complete.")
        return final_df


    def label_outcomes(self,df:pd.DataFrame,lookahead:int)->pd.DataFrame:
        logger.info("  - Generating trade labels with Regime-Adjusted Barriers...");
        # The groupby operation here handles the multi-symbol dataframe correctly
        labeled_dfs=[self._label_group(group,lookahead) for _,group in df.groupby('Symbol')];return pd.concat(labeled_dfs)

    def _label_group(self,group:pd.DataFrame,lookahead:int)->pd.DataFrame:
        if len(group)<lookahead+1:return group
        is_trending=group['market_regime'] == 1
        sl_multiplier=np.where(is_trending,2.0,1.5);tp_multiplier=np.where(is_trending,4.0,2.5)
        sl_atr_dynamic=group['ATR']*sl_multiplier;tp_atr_dynamic=group['ATR']*tp_multiplier
        outcomes=np.zeros(len(group));prices,lows,highs=group['Close'].values,group['Low'].values,group['High'].values
        min_return = self.config.LABEL_MIN_RETURN_PCT

        for i in range(len(group)-lookahead):
            sl_dist,tp_dist=sl_atr_dynamic[i],tp_atr_dynamic[i]
            if pd.isna(sl_dist) or sl_dist<=1e-9:continue

            tp_long,sl_long=prices[i]+tp_dist,prices[i]-sl_dist
            future_highs,future_lows=highs[i+1:i+1+lookahead],lows[i+1:i+1+lookahead]
            time_to_tp_long=np.where(future_highs>=tp_long)[0]; time_to_sl_long=np.where(future_lows<=sl_long)[0]
            first_tp_long=time_to_tp_long[0] if len(time_to_tp_long)>0 else np.inf
            first_sl_long=time_to_sl_long[0] if len(time_to_sl_long)>0 else np.inf

            tp_short,sl_short=prices[i]-tp_dist,prices[i]+sl_dist
            time_to_tp_short=np.where(future_lows<=tp_short)[0]; time_to_sl_short=np.where(future_highs>=sl_short)[0]
            first_tp_short=time_to_tp_short[0] if len(time_to_tp_short)>0 else np.inf
            first_sl_short=time_to_sl_short[0] if len(time_to_sl_short)>0 else np.inf

            if first_tp_long < first_sl_long:
                if (tp_long / prices[i] - 1) > min_return:
                    outcomes[i]=1
            if first_tp_short < first_sl_short:
                if (prices[i] / tp_short - 1) > min_return:
                    outcomes[i]=-1

        group['target']=outcomes;return group

    def _label_meta_group(self, group: pd.DataFrame, lookahead: int) -> pd.DataFrame:
        if 'primary_model_signal' not in group.columns or len(group) < lookahead + 1:
            group['target'] = 0
            return group

        is_trending = group['market_regime'] == 1
        sl_multiplier = np.where(is_trending, 2.0, 1.5)
        tp_multiplier = np.where(is_trending, 4.0, 2.5)
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

            if signal > 0:
                tp_long, sl_long = prices[i] + tp_dist, prices[i] - sl_dist
                if (tp_long / prices[i] - 1) <= min_return: continue
                time_to_tp = np.where(future_highs >= tp_long)[0]
                time_to_sl = np.where(future_lows <= sl_long)[0]
                if len(time_to_tp) > 0 and (len(time_to_sl) == 0 or time_to_tp[0] < time_to_sl[0]):
                    outcomes[i] = 1

            elif signal < 0:
                tp_short, sl_short = prices[i] - tp_dist, prices[i] + sl_dist
                if (prices[i] / tp_short - 1) <= min_return: continue
                time_to_tp = np.where(future_lows <= tp_short)[0]
                time_to_sl = np.where(future_highs >= sl_short)[0]
                if len(time_to_tp) > 0 and (len(time_to_sl) == 0 or time_to_tp[0] < time_to_sl[0]):
                    outcomes[i] = 1

        group['target'] = outcomes
        return group

    def label_meta_outcomes(self, df: pd.DataFrame, lookahead: int) -> pd.DataFrame:
        logger.info("  - Generating BINARY meta-labels (1=correct, 0=incorrect)...")
        labeled_dfs = [self._label_meta_group(group, lookahead) for _, group in df.groupby('Symbol')]
        if not labeled_dfs: return pd.DataFrame()
        return pd.concat(labeled_dfs)

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
        self.gnn_model: Optional[GNNModel] = None
        self.gnn_scaler = MinMaxScaler()
        self.asset_map: Dict[str, int] = {}

    def train(self, df_train: pd.DataFrame, feature_list: List[str], strategy_details: Dict) -> Optional[Tuple[Pipeline, float]]:
        logger.info(f"  - Starting model training using strategy: '{strategy_details.get('description', 'N/A')}'")
        self.is_gnn_model = strategy_details.get("requires_gnn", False)
        self.is_meta_model = strategy_details.get("requires_meta_labeling", False)

        X = pd.DataFrame()

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

        if X.empty:
            logger.error("  - Training data (X) is empty after feature selection. Aborting cycle.")
            return None
        if len(y.unique()) < num_classes:
            logger.warning(f"  - Skipping cycle: Not enough classes ({len(y.unique())}) for {num_classes}-class model.")
            return None

        self.class_weights=dict(zip(np.unique(y),compute_class_weight(class_weight='balanced',classes=np.unique(y),y=y)))
        X_train_val, X_stability, y_train_val, y_stability = train_test_split(X, y, test_size=0.1, shuffle=False)
        X_train, X_val, y_train, y_val = train_test_split(X_train_val, y_train_val, test_size=0.2, shuffle=False)

        if X_train.empty or X_val.empty:
            logger.error(f"  - Training aborted: Data split resulted in an empty training or validation set. (Train shape: {X_train.shape}, Val shape: {X_val.shape})")
            return None

        self.study=self._optimize_hyperparameters(X_train,y_train,X_val,y_val, X_stability, y_stability, num_classes)
        if not self.study or not self.study.best_trials:
            logger.error("  - Training aborted: Hyperparameter optimization failed.")
            return None

        logger.info(f"    - Optimization complete. Best Objective Score: {self.study.best_value:.4f}")
        logger.info(f"    - Best params: {self.study.best_params}")

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
                preds = np.where(max_probs > threshold, preds, 1) # Default to 'hold' class
            else:
                preds = (probs[:, 1] > threshold).astype(int)

            f1 = f1_score(y_val, preds, average='macro', zero_division=0)
            if f1 > best_f1:
                best_f1, best_thresh = f1, threshold

        logger.info(f"    - Best threshold found: {best_thresh:.2f} (F1: {best_f1:.4f})")
        return best_thresh

    def _optimize_hyperparameters(self,X_train,y_train,X_val,y_val, X_stability, y_stability, num_classes)->Optional[optuna.study.Study]:
        logger.info(f"    - Starting hyperparameter optimization with risk-adjusted objective ({self.config.OPTUNA_TRIALS} trials)...")

        def dynamic_progress_callback(study: optuna.study.Study, trial: optuna.trial.FrozenTrial):
            n_trials = self.config.OPTUNA_TRIALS
            trial_number = trial.number + 1
            best_value = study.best_value if study.best_trial else float('nan')
            progress_str = f"> Optuna Optimization: Trial {trial_number}/{n_trials} | Best Score: {best_value:.4f}"
            sys.stdout.write(f"\r{progress_str.ljust(80)}")
            sys.stdout.flush()

        objective = 'multi:softprob' if num_classes > 2 else 'binary:logistic'
        eval_metric = 'mlogloss' if num_classes > 2 else 'logloss'

        # --- V197 ENHANCEMENT: New objective function targeting a Sharpe Ratio proxy ---
        def custom_objective(trial:optuna.Trial):
            param={'objective':objective,'eval_metric':eval_metric,'booster':'gbtree','tree_method':'hist','seed':42,
                   'n_estimators':trial.suggest_int('n_estimators',200, self.config.model_dump().get('n_estimators_max', 1000) ,step=50),
                   'max_depth':trial.suggest_int('max_depth',3, self.config.model_dump().get('max_depth_max', 8)),
                   'learning_rate':trial.suggest_float('learning_rate',0.01,0.2,log=True),
                   'subsample':trial.suggest_float('subsample',0.6,1.0),
                   'colsample_bytree':trial.suggest_float('colsample_bytree',0.6,1.0),
                   'gamma':trial.suggest_float('gamma',0,5),
                   'reg_lambda':trial.suggest_float('reg_lambda',1e-8,5.0,log=True),
                   'alpha':trial.suggest_float('alpha',1e-8,5.0,log=True),
                   'early_stopping_rounds':50}

            if num_classes > 2: param['num_class'] = num_classes

            try:
                scaler=RobustScaler()
                X_train_scaled=scaler.fit_transform(X_train)
                X_val_scaled=scaler.transform(X_val)
                model=xgb.XGBClassifier(**param)
                fit_params={'sample_weight':y_train.map(self.class_weights)}
                model.fit(X_train_scaled,y_train,eval_set=[(X_val_scaled,y_val)],verbose=False,**fit_params)
                
                # Predict on both validation and stability sets to get a fuller picture of performance
                preds_val = model.predict(X_val_scaled)
                preds_stability = model.predict(scaler.transform(X_stability))

                # Map predictions to PNL values
                pnl_map = {0: -1, 1: 0, 2: 1} if num_classes > 2 else {0: -1, 1: 1}
                pnl_val = pd.Series(preds_val).map(pnl_map)
                pnl_stability = pd.Series(preds_stability).map(pnl_map)

                # Combine PNL from both sets
                combined_pnl = pd.concat([pnl_val, pnl_stability])
                
                # Calculate a simple Sharpe Ratio for the trial
                mean_return = combined_pnl.mean()
                std_return = combined_pnl.std()
                
                # Avoid division by zero and ensure positive score for positive returns
                if std_return > 1e-9:
                    sharpe_proxy = mean_return / std_return
                else:
                    sharpe_proxy = mean_return * 10 # Reward if there's profit with zero volatility
                    
                # The objective is now the Sharpe proxy itself. It can be negative, guiding the model away from losses.
                objective_score = sharpe_proxy
                return objective_score

            except Exception as e:
                sys.stdout.write("\n")
                logger.warning(f"Trial {trial.number} failed with error: {e}")
                return -2.0 # Return a harsh penalty for errors

        try:
            study=optuna.create_study(direction='maximize')
            study.optimize(custom_objective, n_trials=self.config.OPTUNA_TRIALS, timeout=3600, n_jobs=-1, callbacks=[dynamic_progress_callback])
            sys.stdout.write("\n")
            return study
        except Exception as e:
            sys.stdout.write("\n")
            logger.error(f"    - Optuna study failed catastrophically: {e}",exc_info=True)
            return None

    def _train_final_model(self,best_params:Dict,X:pd.DataFrame,y:pd.Series, feature_names: List[str], num_classes: int)->Optional[Pipeline]:
        logger.info("    - Training final model...")
        try:
            best_params.pop('early_stopping_rounds', None)
            objective = 'multi:softprob' if num_classes > 2 else 'binary:logistic'
            final_params={'objective':objective,'booster':'gbtree','tree_method':'hist','seed':42,**best_params}
            if num_classes > 2: final_params['num_class'] = num_classes

            final_pipeline=Pipeline([('scaler',RobustScaler()),('model',xgb.XGBClassifier(**final_params))])
            fit_params={'model__sample_weight':y.map(self.class_weights)}
            final_pipeline.fit(X,y,**fit_params)

            if self.config.CALCULATE_SHAP_VALUES:
                self._generate_shap_summary(final_pipeline.named_steps['model'],final_pipeline.named_steps['scaler'].transform(X), feature_names, num_classes)

            return final_pipeline
        except Exception as e:
            logger.error(f"    - Error during final model training: {e}",exc_info=True)
            return None

    def _generate_shap_summary(self, model: xgb.XGBClassifier, X_scaled: np.ndarray, feature_names: List[str], num_classes: int):
        logger.info("    - Generating SHAP feature importance summary...")
        try:
            # V197 NOTE: A dominant feature in SHAP (e.g., ATR being 10x others) might indicate over-reliance.
            # Consider feature transformations (log, normalization) or running cycles without it to improve model robustness.
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
            else: # Binary classification
                overall_importance = np.abs(shap_explanation.values).mean(axis=0)

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
        self.use_tp_ladder = self.config.USE_TP_LADDER

        # V201: TP Ladder Config Validation
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
                profile_settings = tier_settings.get(self.config.RISK_PROFILE, tier_settings['Medium']) # Default to medium
                return profile_settings['risk_pct'], profile_settings['pairs']

        highest_tier_cap = sorted_tiers[-1]
        tier_settings = self.config.TIERED_RISK_CONFIG[highest_tier_cap]
        profile_settings = tier_settings.get(self.config.RISK_PROFILE, tier_settings['Medium'])
        return profile_settings['risk_pct'], profile_settings['pairs']

    def run_backtest_chunk(self, df_chunk_in: pd.DataFrame, confidence_threshold: float, initial_equity: float, is_meta_model: bool) -> Tuple[pd.DataFrame, pd.Series, bool, Optional[Dict], Dict]:
        if df_chunk_in.empty:
            return pd.DataFrame(), pd.Series([initial_equity]), False, None, {}

        df_chunk = df_chunk_in.copy()
        self.is_meta_model = is_meta_model
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
                    open_positions.clear()

            if equity <= 0:
                logger.critical("  - ACCOUNT BLOWN!")
                break

            # V199: Update MAE/MFE for all open positions
            for symbol, pos in open_positions.items():
                if pos['direction'] == 1: # Long
                    pos['mfe_price'] = max(pos['mfe_price'], current_candle['High'])
                    pos['mae_price'] = min(pos['mae_price'], current_candle['Low'])
                else: # Short
                    pos['mfe_price'] = min(pos['mfe_price'], current_candle['Low'])
                    pos['mae_price'] = max(pos['mae_price'], current_candle['High'])
            
            symbols_to_close = []
            for symbol, pos in open_positions.items():
                pnl, exit_price, exit_reason = 0, None, ""

                # --- V201: Take-Profit Ladder Logic ---
                if self.use_tp_ladder:
                    # Check for TP ladder hits
                    for level_idx, tp_level_price in enumerate(pos['tp_prices']):
                        if not pos['tp_levels_hit'][level_idx]:
                            if (pos['direction'] == 1 and current_candle['High'] >= tp_level_price) or \
                               (pos['direction'] == -1 and current_candle['Low'] <= tp_level_price):
                                
                                size_to_exit_pct = self.config.TP_LADDER_LEVELS_PCT[level_idx]
                                level_pnl = pos['initial_risk_amt'] * self.config.TP_LADDER_RISK_MULTIPLIERS[level_idx] * size_to_exit_pct
                                equity += level_pnl
                                day_peak_equity = max(day_peak_equity, equity)
                                equity_curve.append(equity)

                                trades.append({
                                    'ExecTime': current_candle['Timestamp'], 'Symbol': symbol, 'PNL': level_pnl,
                                    'Equity': equity, 'Confidence': pos['confidence'], 'Direction': pos['direction'],
                                    'ExitReason': f"TP Ladder {level_idx+1}"
                                })
                                pos['tp_levels_hit'][level_idx] = True
                                pos['risk_amt_remaining'] -= (pos['initial_risk_amt'] * size_to_exit_pct)
                    
                    # Check for Stop Loss on the remaining position
                    if pos['risk_amt_remaining'] > 1e-6:
                        if (pos['direction'] == 1 and current_candle['Low'] <= pos['sl']) or \
                           (pos['direction'] == -1 and current_candle['High'] >= pos['sl']):
                            
                            sl_pnl = -pos['risk_amt_remaining']
                            equity += sl_pnl
                            day_peak_equity = max(day_peak_equity, equity)
                            equity_curve.append(equity)
                            
                            mae = abs(pos['mae_price'] - pos['entry_price'])
                            mfe = abs(pos['mfe_price'] - pos['entry_price'])
                            
                            trades.append({
                                'ExecTime': current_candle['Timestamp'], 'Symbol': symbol, 'PNL': sl_pnl, 'Equity': equity, 
                                'Confidence': pos['confidence'], 'Direction': pos['direction'], 'ExitReason': "Stop Loss (Ladder)",
                                'MAE': round(mae, 5), 'MFE': round(mfe, 5)
                            })
                            symbols_to_close.append(symbol)
                            if equity <= 0: continue

                    # If all TP levels are hit, the position is fully closed
                    if all(pos['tp_levels_hit']):
                        symbols_to_close.append(symbol)

                # --- Original Single TP/SL & Partial Profit Logic ---
                else:
                    if self.config.USE_PARTIAL_PROFIT and not pos['partial_profit_taken']:
                        partial_tp_price = pos['entry_price'] + (pos['sl_dist'] * self.config.PARTIAL_PROFIT_TRIGGER_R * pos['direction'])
                        if (pos['direction'] == 1 and current_candle['High'] >= partial_tp_price) or \
                           (pos['direction'] == -1 and current_candle['Low'] <= partial_tp_price):
                            partial_pnl = pos['risk_amt'] * self.config.PARTIAL_PROFIT_TRIGGER_R * self.config.PARTIAL_PROFIT_TAKE_PCT
                            equity += partial_pnl
                            day_peak_equity = max(day_peak_equity, equity)
                            equity_curve.append(equity)
                            trades.append({
                                'ExecTime': current_candle['Timestamp'], 'Symbol': symbol, 'PNL': partial_pnl,
                                'Equity': equity, 'Confidence': pos['confidence'], 'Direction': pos['direction'],
                                'ExitReason': f"Partial TP ({self.config.PARTIAL_PROFIT_TAKE_PCT:.0%})"
                            })
                            pos['risk_amt'] *= (1 - self.config.PARTIAL_PROFIT_TAKE_PCT)
                            pos['sl'] = pos['entry_price']
                            pos['partial_profit_taken'] = True

                    if pos['direction'] == 1:
                        if current_candle['Low'] <= pos['sl']: pnl, exit_price, exit_reason = -pos['risk_amt'], pos['sl'], "Stop Loss"
                        elif current_candle['High'] >= pos['tp']: pnl, exit_price, exit_reason = pos['risk_amt'] * pos['rr'], pos['tp'], "Take Profit"
                    elif pos['direction'] == -1:
                        if current_candle['High'] >= pos['sl']: pnl, exit_price, exit_reason = -pos['risk_amt'], pos['sl'], "Stop Loss"
                        elif current_candle['Low'] <= pos['tp']: pnl, exit_price, exit_reason = pos['risk_amt'] * pos['rr'], pos['tp'], "Take Profit"

                    if exit_price:
                        equity += pnl
                        day_peak_equity = max(day_peak_equity, equity)
                        equity_curve.append(equity)

                        mae = abs(pos['mae_price'] - pos['entry_price'])
                        mfe = abs(pos['mfe_price'] - pos['entry_price'])

                        trades.append({
                            'ExecTime': current_candle['Timestamp'], 'Symbol': symbol, 'PNL': pnl,
                            'Equity': equity, 'Confidence': pos['confidence'], 'Direction': pos['direction'],
                            'ExitReason': exit_reason,
                            'MAE': round(mae, 5), 'MFE': round(mfe, 5)
                        })
                        symbols_to_close.append(symbol)
                        if equity <= 0: continue
            
            for symbol in set(symbols_to_close):
                if symbol in open_positions:
                    del open_positions[symbol]
            
            symbol = current_candle['Symbol'] # Re-fetch symbol for entry logic
            if self.config.USE_TIERED_RISK:
                base_risk_pct, max_concurrent_trades = self._get_tiered_risk_params(equity)
            else:
                base_risk_pct, max_concurrent_trades = self.config.BASE_RISK_PER_TRADE_PCT, self.config.MAX_CONCURRENT_TRADES

            if not circuit_breaker_tripped and symbol not in open_positions and len(open_positions) < max_concurrent_trades:
                if prev_candle.get('anomaly_score') == -1: continue
                vol_idx = prev_candle.get('market_volatility_index', 0.5)
                if not (self.config.MIN_VOLATILITY_RANK <= vol_idx <= self.config.MAX_VOLATILITY_RANK): continue

                direction, confidence = 0, 0
                if self.is_meta_model:
                    prob_take_trade = current_candle.get('prob_1', 0)
                    primary_signal = current_candle.get('primary_model_signal', 0)
                    if prob_take_trade > confidence_threshold and primary_signal != 0:
                        direction, confidence = int(np.sign(primary_signal)), prob_take_trade
                else:
                    if 'prob_short' in current_candle:
                        probs=np.array([current_candle['prob_short'],current_candle['prob_hold'],current_candle['prob_long']])
                        max_confidence=np.max(probs)
                        if max_confidence >= confidence_threshold:
                            pred_class=np.argmax(probs)
                            direction=1 if pred_class==2 else -1 if pred_class==0 else 0
                            confidence = max_confidence

                if direction != 0:
                    atr = prev_candle.get('ATR',0)
                    if pd.isna(atr) or atr<=1e-9: continue

                    if confidence>=self.config.CONFIDENCE_TIERS['ultra_high']['min']: tier_name='ultra_high'
                    elif confidence>=self.config.CONFIDENCE_TIERS['high']['min']: tier_name='high'
                    else: tier_name='standard'

                    tier=self.config.CONFIDENCE_TIERS[tier_name]
                    base_risk_amt = equity * base_risk_pct * tier['risk_mult']
                    risk_amt = min(base_risk_amt, self.config.RISK_CAP_PER_TRADE_USD)
                    sl_dist = (atr * 1.5) + (atr * self.config.SLIPPAGE_PCTG_OF_ATR)
                    tp_dist = (sl_dist * tier['rr'])
                    if tp_dist<=0 or sl_dist<=0: continue

                    lots = max(self.config.MIN_LOT_SIZE, round((risk_amt / (sl_dist * self.config.CONTRACT_SIZE)) / self.config.LOT_STEP) * self.config.LOT_STEP)
                    if lots < self.config.MIN_LOT_SIZE: continue

                    margin_required = (lots * self.config.CONTRACT_SIZE) / self.config.LEVERAGE
                    used_margin = sum(p.get('margin_used', 0) for p in open_positions.values())
                    if (equity - used_margin) < margin_required: continue

                    entry_price = current_candle['Open'] + ((atr * self.config.SPREAD_PCTG_OF_ATR) * direction)
                    sl_price, tp_price = entry_price - sl_dist * direction, entry_price + tp_dist * direction

                    open_positions[symbol]={
                        'direction':direction, 'entry_price': entry_price, 'sl':sl_price,'tp':tp_price,
                        'risk_amt':risk_amt, 'rr':tier['rr'], 'confidence':confidence,
                        'sl_dist': sl_dist, 'partial_profit_taken': False,
                        'lot_size': lots, 'margin_used': margin_required,
                        'mfe_price': entry_price, 'mae_price': entry_price
                    }

                    # V201: Setup TP Ladder if enabled
                    if self.use_tp_ladder:
                        pos = open_positions[symbol]
                        pos['tp_prices'] = [entry_price + (sl_dist * mult * direction) for mult in self.config.TP_LADDER_RISK_MULTIPLIERS]
                        pos['tp_levels_hit'] = [False] * len(pos['tp_prices'])
                        pos['initial_risk_amt'] = risk_amt
                        pos['risk_amt_remaining'] = risk_amt

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
            # V199: Summarize MAE/MFE for the report if present in cycle data
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


def initialize_playbook(base_path: str) -> Dict:
    results_dir = os.path.join(base_path, "Results"); os.makedirs(results_dir, exist_ok=True)
    playbook_path = os.path.join(results_dir, "strategy_playbook.json")
    DEFAULT_PLAYBOOK = { "FilteredBreakout": { "description": "[BREAKOUT] A hybrid that trades high-volatility breakouts (ATR, Bollinger) but only in the direction of the long-term daily trend.", "features": [ "ATR", "bollinger_bandwidth", "DAILY_ctx_Trend", "ADX", "hour", "anomaly_score", "RSI_slope" ], "lookahead_range": [ 60, 120 ], "dd_range": [ 0.2, 0.35 ] }, "RangeBound": { "description": "[RANGING] Trades reversals in a sideways channel, filtered by low ADX.", "features": [ "ADX", "RSI", "stoch_k", "stoch_d", "bollinger_bandwidth", "hour", "wick_to_body_ratio" ], "lookahead_range": [ 20, 60 ], "dd_range": [ 0.1, 0.2 ] }, "TrendPullback": { "description": "[TRENDING] Enters on pullbacks (RSI) in the direction of an established long-term trend.", "features": [ "DAILY_ctx_Trend", "ADX", "RSI", "H1_ctx_Trend", "momentum_10_slope", "hour", "close_fracdiff", "relative_performance" ], "lookahead_range": [ 50, 150 ], "dd_range": [ 0.15, 0.3 ] }, "ConfluenceTrend": { "description": "[TRENDING] A more conservative trend strategy that requires confluence from multiple indicators.", "features": [ "DAILY_ctx_Trend", "H1_ctx_Trend", "RSI_slope", "stoch_k", "ADX", "momentum_10" ], "lookahead_range": [ 60, 160 ], "dd_range": [ 0.15, 0.3 ] }, "MeanReversionOscillator": { "description": "[RANGING] A pure mean-reversion strategy using oscillators for entry signals in low-volatility environments.", "features": [ "RSI", "stoch_k", "ADX", "market_volatility_index", "close_fracdiff", "hour", "day_of_week", "wick_to_body_ratio" ], "lookahead_range": [ 20, 60 ], "dd_range": [ 0.15, 0.25 ] }, "GNN_Market_Structure": { "description": "[SPECIALIZED] Uses a GNN to model inter-asset correlations for predictive features.", "features": [], "lookahead_range": [ 80, 150 ], "dd_range": [ 0.15, 0.3 ], "requires_gnn": True }, "Meta_Labeling_Filter": { "description": "[SPECIALIZED] Uses a secondary ML filter to improve a simple primary model's signal quality.", "features": [ "ADX", "RSI_slope", "ATR", "bollinger_bandwidth", "H1_ctx_Trend", "DAILY_ctx_Trend", "momentum_20", "relative_performance" ], "lookahead_range": [ 50, 100 ], "dd_range": [ 0.1, 0.25 ], "requires_meta_labeling": True }, "BollingerMeanReversion": { "description": "[RANGING] Trades mean-reversion in low-volatility channels defined by Bollinger Bands, filtered by low ADX.", "features": [ "ADX", "RSI", "stoch_k", "bollinger_bandwidth", "hour", "day_of_week", "anomaly_score", "wick_to_body_ratio" ], "lookahead_range": [ 20, 60 ], "dd_range": [ 0.1, 0.2 ] }, "VolatilityExpansionBreakout": { "description": "[BREAKOUT] Enters on strong breakouts that occur after a period of low-volatility consolidation (Bollinger Squeeze).", "features": [ "bollinger_bandwidth", "ATR", "market_volatility_index", "momentum_20", "DAILY_ctx_Trend", "H1_ctx_Trend", "RSI_slope" ], "lookahead_range": [ 70, 140 ], "dd_range": [ 0.2, 0.4 ] }, "RegimeClarityFilter": { "description": "[SPECIALIZED] A non-trading filter model that predicts the market regime (Trend, Range, Unclear) to gate other strategies.", "features": [ "ADX", "market_volatility_index", "bollinger_bandwidth", "close_fracdiff", "ATR", "DAILY_ctx_ATR", "relative_performance" ], "lookahead_range": [ 30, 90 ], "dd_range": [ 0.05, 0.1 ] } }
    if not os.path.exists(playbook_path):
        logger.warning(f"'strategy_playbook.json' not found. Seeding a new one with default strategies at: {playbook_path}")
        try:
            with open(playbook_path, 'w') as f: json.dump(DEFAULT_PLAYBOOK, f, indent=4)
            return DEFAULT_PLAYBOOK
        except IOError as e: logger.error(f"Failed to create playbook file: {e}. Using in-memory default."); return DEFAULT_PLAYBOOK
    try:
        with open(playbook_path, 'r') as f: playbook = json.load(f)
        if any(key not in playbook for key in DEFAULT_PLAYBOOK):
            logger.info("Updating playbook with new default strategies...")
            playbook.update({k: v for k, v in DEFAULT_PLAYBOOK.items() if k not in playbook})
            with open(playbook_path, 'w') as f: json.dump(playbook, f, indent=4)
        logger.info(f"Successfully loaded dynamic playbook from {playbook_path}"); return playbook
    except (json.JSONDecodeError, IOError) as e: logger.error(f"Failed to load or parse playbook file: {e}. Using in-memory default."); return DEFAULT_PLAYBOOK

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
    temp_config = ConfigModel(**{**current_config_dict, 'nickname': 'init', 'run_timestamp': 'init'})
    
    data_loader = DataLoader(temp_config)
    all_files = [f for f in os.listdir(current_config_dict['BASE_PATH']) if f.endswith(('.csv', '.txt')) and re.match(r'^[A-Z0-9]+_[A-Z0-9]+', f)]
    if not all_files: logger.critical("No data files found in base path. Exiting."); return
    data_by_tf, detected_timeframes = data_loader.load_and_parse_data(all_files)
    if not data_by_tf: return

    tf_roles = determine_timeframe_roles(detected_timeframes)
    fe = FeatureEngineer(temp_config, tf_roles)
    full_df = fe.create_feature_stack(data_by_tf)
    if full_df.empty: logger.critical("Feature engineering resulted in an empty dataframe. Exiting."); return

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
    
    ai_setup = api_timer.call(gemini_analyzer.get_initial_run_setup, version_label, nickname_ledger, framework_history, playbook, health_report, directives, data_summary, diagnosed_regime, regime_champions)
    if not ai_setup: logger.critical("AI-driven setup failed. Exiting."); return

    current_config_dict.update(_sanitize_ai_suggestions(ai_setup))
    if 'RETRAINING_FREQUENCY' in ai_setup: current_config_dict['RETRAINING_FREQUENCY'] = _sanitize_frequency_string(ai_setup['RETRAINING_FREQUENCY'])

    if isinstance(ai_setup.get("nickname"), str) and ai_setup.get("nickname"):
        nickname_ledger[version_label] = ai_setup["nickname"]
        try:
            with open(temp_config.NICKNAME_LEDGER_PATH, 'w') as f: json.dump(nickname_ledger, f, indent=4)
        except IOError as e: logger.error(f"Failed to save new nickname to ledger: {e}")

    config = ConfigModel(**{**current_config_dict, 'REPORT_LABEL': version_label, 'nickname': nickname_ledger.get(version_label, f"Run-{run_timestamp_str}")})
    
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

    if retraining_dates.empty: logger.critical("Cannot proceed: No valid retraining dates could be determined."); return

    # V199: Initialize histories for the AI's long-term memory
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

        df_train_raw, df_test = full_df.loc[train_start:train_end], full_df.loc[period_start_date:min(test_end, full_df.index.max())]
        if df_train_raw.empty or df_test.empty: cycle_num += 1; continue

        strategy_details = playbook.get(config.strategy_name, {})
        fe.config = config
        df_train_labeled = fe.label_meta_outcomes(df_train_raw, config.LOOKAHEAD_CANDLES) if strategy_details.get("requires_meta_labeling") else fe.label_outcomes(df_train_raw, config.LOOKAHEAD_CANDLES)
        
        best_objective_score = -1.0
        optimization_path = [] #V200
        
        if not check_label_quality(df_train_labeled, config.LABEL_MIN_EVENT_PCT):
            logger.critical("!! MODEL TRAINING SKIPPED !! Un-trainable labels. Retry {cycle_retry_count+1}/{config.MAX_TRAINING_RETRIES_PER_CYCLE}.")
            cycle_retry_count += 1
        else:
            config.selected_features = [f for f in config.selected_features if f in all_available_features]
            trainer = ModelTrainer(config)
            train_result = trainer.train(df_train_labeled, config.selected_features, strategy_details)
            
            if trainer.study:
                all_optuna_trials.extend([{'params': t.params, 'value': t.value} for t in trainer.study.trials if t.value is not None])
                optimization_path = [t.value for t in trainer.study.trials if t.value is not None] #V200
            best_objective_score = trainer.study.best_value if trainer.study and trainer.study.best_value is not None else -1

            if not train_result or best_objective_score < MODEL_QUALITY_THRESHOLD:
                logger.critical(f"!! MODEL QUALITY GATE FAILED !! Score ({best_objective_score:.3f}) < Threshold ({MODEL_QUALITY_THRESHOLD}). Retry {cycle_retry_count+1}/{config.MAX_TRAINING_RETRIES_PER_CYCLE}.")
                cycle_retry_count += 1
            else:
                pipeline, threshold = train_result
                cycle_retry_count = 0 
                last_successful_pipeline, last_successful_threshold, last_successful_features, last_successful_is_gnn = pipeline, threshold, config.selected_features, trainer.is_gnn_model
                
                if trainer.shap_summary is not None:
                    for feature, row in trainer.shap_summary.iterrows():
                        shap_history[feature].append(round(row['SHAP_Importance'], 4))

                X_test = trainer._get_gnn_embeddings_for_test(df_test) if trainer.is_gnn_model else df_test[config.selected_features].copy().fillna(0)
                if not X_test.empty:
                    probs = pipeline.predict_proba(X_test)
                    if strategy_details.get("requires_meta_labeling"): df_test[['prob_0', 'prob_1']] = probs
                    else: df_test[['prob_short', 'prob_hold', 'prob_long']] = probs

                backtester = Backtester(config)
                trades, equity_curve, breaker_tripped, breaker_context, daily_dd_report = backtester.run_backtest_chunk(df_test, threshold, last_equity, strategy_details.get("requires_meta_labeling", False))
                aggregated_daily_dd_reports.append(daily_dd_report)
                cycle_status_msg = "Completed"

        if cycle_retry_count > config.MAX_TRAINING_RETRIES_PER_CYCLE:
            logger.error(f"!! STRATEGY FAILURE !! Exceeded max training retries. Triggering strategic intervention.")
            personal_best_config = max(run_configs_and_metrics, key=lambda x: x.get('final_metrics', {}).get('mar_ratio', -np.inf)) if run_configs_and_metrics else None
            intervention_suggestion = api_timer.call(gemini_analyzer.propose_strategic_intervention, in_run_historical_cycles[-2:], playbook, config.strategy_name, quarantine_list, personal_best_config)
            if intervention_suggestion:
                config = ConfigModel(**{**config.model_dump(mode='json'), **intervention_suggestion})
                logger.warning(f"  - Intervention successful. Switching to strategy: {config.strategy_name}")
                cycle_retry_count = 0
                continue 
            else:
                logger.critical("  - Strategic intervention FAILED. Halting run."); break

        elif cycle_retry_count > 0:
            logger.info("  - Engaging AI for retry parameters...")
            ai_suggestions = api_timer.call(gemini_analyzer.analyze_cycle_and_suggest_changes, in_run_historical_cycles, framework_history, all_available_features, config.model_dump(), "TRAINING_FAILURE", shap_history, all_optuna_trials)
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
        
        cycle_result = {"StartDate": period_start_date.date().isoformat(), "EndDate": test_end.date().isoformat(), "NumTrades": len(trades), "PNL": round(cycle_pnl, 2), "Status": "Circuit Breaker" if breaker_tripped else cycle_status_msg, "BestObjectiveScore": round(best_objective_score, 4), "trade_summary": trade_summary, "optimization_path": optimization_path} #V200
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
        suggested_params = api_timer.call(gemini_analyzer.analyze_cycle_and_suggest_changes, in_run_historical_cycles, framework_history, all_available_features, config.model_dump(), cycle_status_for_ai, shap_history, all_optuna_trials)
        if suggested_params:
            config = ConfigModel(**{**config.model_dump(mode='json'), **_sanitize_ai_suggestions(suggested_params)})

        cycle_num += 1
        logger.info(f"--- Cycle complete. PNL: ${cycle_pnl:,.2f} | Final Equity: ${last_equity:,.2f} | Time: {time.time() - cycle_start_time:.2f}s ---")

    # Final report generation
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
        # --- V197 CHANGE: Refined confidence tiers for more achievable profit targets ---
        "CONFIDENCE_TIERS": {
            'ultra_high': {'min': 0.8, 'risk_mult': 1.2, 'rr': 2.5},
            'high':       {'min': 0.7, 'risk_mult': 1.0, 'rr': 2.0},
            'standard':   {'min': 0.6, 'risk_mult': 0.8, 'rr': 1.5}
        },
        "BASE_RISK_PER_TRADE_PCT": 0.01,"RISK_CAP_PER_TRADE_USD": 500.0,
        "SPREAD_PCTG_OF_ATR": 0.05, "SLIPPAGE_PCTG_OF_ATR": 0.02,
        "OPTUNA_TRIALS": 50, "TRAINING_WINDOW": '365D', "RETRAINING_FREQUENCY": '90D',
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
        # --- V197 CHANGE: Increased minimum return to filter for more significant signals ---
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
        # --- NEW (V196): Default broker constraints ---
        "CONTRACT_SIZE": 100000.0,
        "LEVERAGE": 30,
        "MIN_LOT_SIZE": 0.01,
        "LOT_STEP": 0.01
    }

    fallback_config["DIRECTIVES_FILE_PATH"] = os.path.join(fallback_config["BASE_PATH"], "Results", "framework_directives.json")

    api_interval_seconds = 300

    run_count = 0; script_start_time = datetime.now(); is_continuous = CONTINUOUS_RUN_HOURS > 0 or MAX_RUNS > 1

    bootstrap_config = ConfigModel(**fallback_config, run_timestamp="init", nickname="init")
    playbook = initialize_playbook(bootstrap_config.BASE_PATH)

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
    
# End_To_End_Advanced_ML_Trading_Framework_PRO_V201_Linux.py
