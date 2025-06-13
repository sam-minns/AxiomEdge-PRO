# End_To_End_Advanced_ML_Trading_Framework_PRO_V181_Sanitization.py
#
# End_To_End_Advanced_ML_Trading_Framework_PRO_V181_Sanitization.py
#
# V181 UPDATE (AI Response Sanitization):
# 1. FIXED: A critical ValidationError caused by the AI model suggesting parameter
#    values (e.g., a negative MAX_DD_PER_CYCLE) that violate the framework's
#    constraints.
# 2. ADDED: A sanitization layer that intercepts all numerical parameter suggestions
#    from the AI. It validates them against predefined, safe boundaries and
#    clamps any out-of-range values before they are applied, preventing crashes
#    and ensuring stable operation.
# 3. IMPROVED: Overall framework robustness against unpredictable or "hallucinated"
#    AI outputs.
#
# V180 UPDATE (Critical Fixes & Cleanup):
# 1. FIXED: A critical NameError that occurred during the final report generation
#    (framework_memory was not defined), preventing the script from completing.
# 2. FIXED: An issue where multiple empty or temporary result folders were created
#    during a single run. The logic now ensures only one, correctly named
#    folder is created per run after the AI assigns a nickname.

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
from sklearn.preprocessing import RobustScaler, MinMaxScaler
from sklearn.utils.class_weight import compute_class_weight
from pydantic import BaseModel, DirectoryPath, confloat, conint, Field
from sklearn.ensemble import IsolationForest


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

# --- V178 LOGGING SWITCHES ---
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
# 2. GEMINI AI ANALYZER & API TIMER
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
        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key or "YOUR" in api_key or "PASTE" in api_key:
            logger.warning("!CRITICAL! GEMINI_API_KEY not found in environment or is a placeholder.")
            try:
                api_key = input(">>> Please paste your Gemini API Key and press Enter, or press Enter to skip: ").strip()
                if not api_key:
                    logger.warning("No API Key provided. AI analysis will be skipped.")
                    self.api_key_valid = False
                else:
                    logger.info("Using API Key provided via manual input.")
                    self.api_key_valid = True
            except Exception:
                logger.warning("Could not read input. AI analysis will be skipped.")
                self.api_key_valid = False
                api_key = None
        else:
            logger.info("Successfully loaded GEMINI_API_KEY from environment.")
            self.api_key_valid = True

        if self.api_key_valid:
            self.api_url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash:generateContent?key={api_key}"
            self.headers = {"Content-Type": "application/json"}
        else:
            self.api_url = ""
            self.headers = {}

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
        if not self.api_key_valid: return "{}"
        if len(prompt) > 30000: logger.warning("Prompt is very large, may risk exceeding token limits.")
        payload = {"contents": [{"parts": [{"text": prompt}]}]}
        sanitized_payload = self._sanitize_dict(payload)
        
        retry_delays = [5, 15, 30]
        
        for attempt, delay in enumerate([0] + retry_delays):
            if delay > 0:
                logger.warning(f"API connection failed. Retrying in {delay} seconds... (Attempt {attempt}/{len(retry_delays)})")
                flush_loggers()
                time.sleep(delay)
                
            try:
                response = requests.post(self.api_url, headers=self.headers, data=json.dumps(sanitized_payload), timeout=120)
                response.raise_for_status()
                
                result = response.json()
                if "candidates" in result and result["candidates"] and "content" in result["candidates"][0] and "parts" in result["candidates"][0]["content"]:
                    return result["candidates"][0]["content"]["parts"][0]["text"]
                else:
                    logger.error(f"Invalid Gemini response structure: {result}")
                    return "{}"

            except requests.exceptions.RequestException as e:
                logger.error(f"Gemini API request failed on attempt {attempt + 1}: {e}")
                if attempt == len(retry_delays):
                    logger.critical("API connection failed after all retries. Stopping.")
                    return "{}"
            except json.JSONDecodeError as e:
                logger.error(f"Failed to decode Gemini response JSON: {e} - Response: {response.text}")
                return "{}"
            except (KeyError, IndexError) as e:
                logger.error(f"Failed to extract text from Gemini response: {e} - Response: {response.text}")
                return "{}"

        return "{}"

    def _extract_json_from_response(self, response_text: str) -> Dict:
        try:
            match = re.search(r"```json\s*(\{.*?\})\s*```", response_text, re.DOTALL)
            if not match: match = re.search(r"(\{.*\})", response_text, re.DOTALL)
            if not match:
                logger.error(f"Could not find JSON block in response.\nResponse text: {response_text}")
                return {}

            suggestions = json.loads(match.group(1).strip())
            if 'current_params' in suggestions and isinstance(suggestions.get('current_params'), dict):
                nested_params = suggestions.pop('current_params')
                suggestions.update(nested_params)
            return suggestions
        except (json.JSONDecodeError, AttributeError) as e:
            logger.error(f"Could not parse JSON from response: {e}\nResponse text: {response_text}"); return {}

    def get_initial_run_setup(self, script_version: str, ledger: Dict, memory: Dict, playbook: Dict, health_report: Dict, directives: List[Dict], data_summary: Dict) -> Dict:
        if not self.api_key_valid:
            logger.warning("No API key. Skipping AI-driven setup and using default config.")
            return {}

        logger.info("-> Performing Initial AI Analysis & Setup (Single API Call)...")

        nickname_prompt_part = ""
        if script_version not in ledger:
            theme = random.choice(["Astronomical Objects", "Mythological Figures", "Gemstones", "Constellations", "Legendary Swords"])
            nickname_prompt_part = (
                "1.  **Generate Nickname**: The script version is new. Generate a unique, cool-sounding, one-word codename for this run. "
                f"The theme is **{theme}**. Do not use any from this list of past names: {list(ledger.values())}. "
                "Place the new name in the `nickname` key of your JSON response.\n"
            )
        else:
            nickname_prompt_part = "1.  **Generate Nickname**: A nickname for this script version already exists. You do not need to generate a new one. Set the `nickname` key in your response to `null`.\n"

        directive_str = "No specific directives for this run."
        if directives:
            directive_str = "**CRITICAL DIRECTIVES FOR THIS RUN:**\n" + "\n".join(f"- {d.get('reason', 'Unnamed directive')}" for d in directives)

        health_report_str = "No long-term health report available."
        if health_report:
            health_report_str = f"**STRATEGIC HEALTH ANALYSIS (Lower scores are better):**\n{json.dumps(health_report, indent=2)}\n\n"

        if not GNN_AVAILABLE:
            playbook = {k: v for k, v in playbook.items() if not v.get("requires_gnn")}
            logger.warning("GNN strategies filtered from playbook due to missing libraries.")

        prompt = (
            "You are a Master Trading Strategist responsible for configuring a trading framework for its next run.\n\n"
            "**YOUR THREE TASKS:**\n"
            f"{nickname_prompt_part}"
            "2.  **Analyze Market Data**: Review the `MARKET DATA SUMMARY` and comment on conditions in the `analysis_notes` key.\n"
            "3.  **Select Strategy & Configure**: Based on your analysis, select the optimal `strategy_name` from the playbook. Then, provide a complete and robust starting configuration. **This is your most important task.**\n\n"
            "**PARAMETER RULES & GUIDANCE:**\n"
            "- `strategy_name`: MUST be one of the keys from the playbook.\n"
            "- `selected_features`: **CRITICAL: You MUST provide a NON-EMPTY list.** Use the `features` list from the chosen playbook strategy as your starting point.\n"
            "- `RETRAINING_FREQUENCY`: **MUST be a string** with a number and unit (e.g., '90D', '8W', '2M').\n"
            "- `USE_PARTIAL_PROFIT`: **MUST be a boolean (true/false).** Start with `false` unless you have a strong reason to enable it.\n"
            "- If `USE_PARTIAL_PROFIT` is true, provide `PARTIAL_PROFIT_TRIGGER_R` (float) and `PARTIAL_PROFIT_TAKE_PCT` (float).\n"
            "- `MAX_CONCURRENT_TRADES`: An integer from 1 to 10.\n"
            "- `MAX_DD_PER_CYCLE`: A float between 0.05 and 0.5.\n\n"
            "**OUTPUT FORMAT**: Respond ONLY with a single, valid JSON object containing all required keys.\n\n"
            "--- CONTEXT FOR YOUR DECISION ---\n\n"
            f"**1. MARKET DATA SUMMARY:**\n{json.dumps(self._sanitize_dict(data_summary), indent=2)}\n\n"
            f"**2. CRITICAL DIRECTIVES:**\n{directive_str}\n\n"
            f"**3. STRATEGIC HEALTH & MEMORY:**\n{health_report_str}"
            f"Framework Memory (Champion & History):\n{json.dumps(self._sanitize_dict(memory), indent=2)}\n\n"
            f"**4. STRATEGY PLAYBOOK (Your options):**\n{json.dumps(playbook, indent=2)}\n"
        )


        response_text = self._call_gemini(prompt)
        suggestions = self._extract_json_from_response(response_text)

        if suggestions and "strategy_name" in suggestions:
            logger.info("  - Initial AI Analysis and Setup complete.")
            return suggestions
        else:
            logger.error("  - AI-driven setup failed to return a valid configuration.")
            return {}

    def analyze_cycle_and_suggest_changes(self, historical_results: List[Dict], available_features: List[str], strategy_details: Dict, daily_dd_report: Dict, mc_results: Dict) -> Dict:
        if not self.api_key_valid: return {}
        
        base_prompt_intro = "You are an expert trading model analyst. Your primary goal is to create a STABLE and PROFITABLE strategy by tuning its parameters within a walk-forward analysis."
        json_response_format = "Respond ONLY with a valid JSON object containing your suggested keys."

        partial_tp_guidance = (
            "**PARTIAL PROFIT-TAKING LOGIC:**\n"
            "- `USE_PARTIAL_PROFIT`: This is a boolean master switch.\n"
            "- **Set to `true` ONLY if you see evidence that winning trades are consistently failing to reach their full targets and are reversing.** The goal is to capture some profit before a reversal.\n"
            "- **Keep it `false` if trades are running to their full potential** or if the strategy is mostly losing. Enabling it on a losing strategy just locks in smaller wins and doesn't fix the core problem.\n"
            "- If you set it to `true`, you MUST also provide `PARTIAL_PROFIT_TRIGGER_R` (a float > 0.1) and `PARTIAL_PROFIT_TAKE_PCT` (a float between 0.1 and 0.9)."
        )

        base_suggestions = (
            "- `MAX_DD_PER_CYCLE` (float between 0.05 and 0.5), `RETRAINING_FREQUENCY` (string with unit), `MAX_CONCURRENT_TRADES` (int 1-10).\n"
            "- `selected_features`: **MUST be a NON-EMPTY list**.\n"
            f"{partial_tp_guidance}"
        )

        prompt_details = (
            "You are managing a model. Your goal is to tune its features and parameters.\n\n"
            "**Suggest a NEW configuration for:**\n"
            f"{base_suggestions}"
            "- `LOOKAHEAD_CANDLES`, `OPTUNA_TRIALS` (integers).\n"
        )
        
        if historical_results and historical_results[-1].get("Status") == "Circuit Breaker":
            prompt_details = (
                "You are an expert trading model analyst acting in a **SAFETY FIRST** capacity. The previous trading cycle hit its maximum drawdown ('Circuit Breaker'). Your primary goal is to **REDUCE RISK**.\n\n"
                "**CRITICAL INSTRUCTIONS:**\n"
                "1.  **You MUST suggest changes that lower risk.** This is not optional.\n"
                "2.  **Lower risk by:** reducing `MAX_DD_PER_CYCLE` (must be between 0.05-0.25), reducing `MAX_CONCURRENT_TRADES`, or considering enabling `USE_PARTIAL_PROFIT` to lock in smaller profits earlier.\n"
                "3.  For `RETRAINING_FREQUENCY`, **you MUST respond with a string with a unit**. Consider a shorter frequency for faster adaptation.\n"
                "4.  For `selected_features`, **you MUST provide a NON-EMPTY list**. Consider simplifying the model by reducing the number of features.\n"
            )

        data_context = (
            f"**SUMMARIZED HISTORICAL CYCLE RESULTS:**\n{json.dumps(self._sanitize_dict(historical_results), indent=2)}\n\n"
            f"**AVAILABLE FEATURES FOR THIS STRATEGY:**\n{available_features}"
        )

        prompt = f"{base_prompt_intro}\n\n{prompt_details}\n\n{json_response_format}\n\n{data_context}"

        response_text = self._call_gemini(prompt)
        suggestions = self._extract_json_from_response(response_text)
        return suggestions

    def propose_strategic_intervention(self, failure_history: List[Dict], playbook: Dict, last_failed_strategy: str, quarantine_list: List[str]) -> Dict:
        if not self.api_key_valid: return {}
        logger.warning("! STRATEGIC INTERVENTION !: Current strategy has failed repeatedly. Engaging AI to select a new strategy.")

        available_playbook = { k: v for k, v in playbook.items() if k not in quarantine_list and (GNN_AVAILABLE or not v.get("requires_gnn"))}

        prompt = (
            "You are a master strategist executing an emergency intervention. The current strategy, "
            f"**`{last_failed_strategy}`**, has failed for {len(failure_history)} consecutive cycles by hitting its Circuit Breaker. "
            "Continuing to tweak its parameters is illogical and has failed.\n\n"
            "**CRITICAL INSTRUCTIONS:**\n"
            f"1.  **CRITICAL CONSTRAINT:** The following strategies are in 'quarantine' due to recent, repeated failures. **YOU MUST NOT SELECT ANY STRATEGY FROM THIS LIST: {quarantine_list}**\n"
            "2.  **Select a NEW STRATEGY:** You **MUST** choose a *different* strategy from the available playbook that is NOT in the quarantine list. You are forced to explore a novel approach.\n"
            "3.  **Propose a Safe Starting Configuration:** Provide a reasonable and SAFE starting configuration for this new strategy. Start with conservative values: `RETRAINING_FREQUENCY`: '90D', `MAX_DD_PER_CYCLE`: 0.15 (float), `MAX_CONCURRENT_TRADES`: 2, and **`USE_PARTIAL_PROFIT`: false**. **Ensure `selected_features` is a NON-EMPTY list.**\n\n"
            f"**RECENT FAILED HISTORY:**\n{json.dumps(self._sanitize_dict(failure_history), indent=2)}\n\n"
            f"**AVAILABLE STRATEGIES (PLAYBOOK):**\n{json.dumps(available_playbook, indent=2)}\n\n"
            "Respond ONLY with a valid JSON object for the new configuration, including `strategy_name`."
        )

        response_text = self._call_gemini(prompt)
        suggestions = self._extract_json_from_response(response_text)
        return suggestions

# =============================================================================
# 3. CONFIGURATION & VALIDATION
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
    
    MODEL_SAVE_PATH: str = Field(default="", repr=False); PLOT_SAVE_PATH: str = Field(default="", repr=False)
    REPORT_SAVE_PATH: str = Field(default="", repr=False); SHAP_PLOT_PATH: str = Field(default="", repr=False)
    LOG_FILE_PATH: str = Field(default="", repr=False); CHAMPION_FILE_PATH: str = Field(default="", repr=False)
    HISTORY_FILE_PATH: str = Field(default="", repr=False); PLAYBOOK_FILE_PATH: str = Field(default="", repr=False)
    DIRECTIVES_FILE_PATH: str = Field(default="", repr=False); NICKNAME_LEDGER_PATH: str = Field(default="", repr=False)

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
            df['RealVolume'] = pd.to_numeric(df['RealVolume'], errors='coerce').fillna(0)
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
                all_symbols_df = [df[~df.index.duplicated(keep='first')].sort_index() for _, df in combined.groupby('Symbol')]
                final_combined = pd.concat(all_symbols_df).sort_index()
                processed_dfs[tf] = final_combined
                logger.info(f"  - Processed {tf}: {len(final_combined):,} rows for {len(final_combined['Symbol'].unique())} symbols.")
        detected_timeframes = list(processed_dfs.keys())
        if not processed_dfs: logger.critical("  - Data loading failed for all files."); return None, []
        logger.info(f"[SUCCESS] Data loading complete. Detected timeframes: {detected_timeframes}")
        return processed_dfs, detected_timeframes

class FeatureEngineer:
    TIMEFRAME_MAP = {'M1': 1,'M5': 5,'M15': 15,'M30': 30,'H1': 60,'H4': 240,'D1': 1440, 'DAILY': 1440}
    ANOMALY_FEATURES = ['ATR', 'bollinger_bandwidth']

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

    def _get_anomaly_scores(self, df: pd.DataFrame, features_to_check: list, contamination: float = 0.01) -> pd.Series:
        df_clean = df[features_to_check].dropna()
        if df_clean.empty:
            return pd.Series(1, index=df.index, name='anomaly_score')
        
        model = IsolationForest(contamination=contamination, random_state=42, n_estimators=100)
        model.fit(df_clean)

        scores = pd.Series(model.predict(df[features_to_check].fillna(0)), index=df.index)
        scores.name = 'anomaly_score'
        return scores

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
        return g

    def _calculate_seasonality(self, g: pd.DataFrame) -> pd.DataFrame:
        g['month'] = g.index.month
        g['week_of_year'] = g.index.isocalendar().week.astype(int)
        g['day_of_month'] = g.index.day
        return g

    def _calculate_candlestick_patterns(self, g: pd.DataFrame) -> pd.DataFrame:
        g['candle_body_size'] = abs(g['Close'] - g['Open'])
        g['is_doji'] = (g['candle_body_size'] / g['ATR'].replace(0,1)).lt(0.1).astype(int)
        g['is_engulfing'] = ((g['candle_body_size'] > abs(g['Close'].shift() - g['Open'].shift())) & (np.sign(g['Close']-g['Open']) != np.sign(g['Close'].shift()-g['Open'].shift()))).astype(int)
        return g

    def _calculate_htf_features(self,df:pd.DataFrame,p:str,s:int,a:int)->pd.DataFrame:
        tf_id = p.upper()
        logger.info(f"    - Calculating HTF features for {tf_id}...");results=[]
        for symbol,group in df.groupby('Symbol'):
            g=group.copy();sma=g['Close'].rolling(s,min_periods=s).mean();atr=(g['High']-g['Low']).rolling(a,min_periods=a).mean();trend=np.sign(g['Close']-sma)
            temp_df=pd.DataFrame(index=g.index)
            temp_df[f'{tf_id}_ctx_SMA']=sma
            temp_df[f'{tf_id}_ctx_ATR']=atr
            temp_df[f'{tf_id}_ctx_Trend']=trend
            shifted_df=temp_df.shift(1);shifted_df['Symbol']=symbol;results.append(shifted_df)
        return pd.concat(results).reset_index()

    def _calculate_base_tf_native(self, g:pd.DataFrame)->pd.DataFrame:
        g_out=g.copy();lookback=14
        g_out['ATR']=(g['High']-g['Low']).rolling(lookback).mean();delta=g['Close'].diff();gain=delta.where(delta > 0,0).ewm(com=lookback-1,adjust=False).mean()
        loss=-delta.where(delta < 0,0).ewm(com=lookback-1,adjust=False).mean();g_out['RSI']=100-(100/(1+(gain/loss.replace(0,1e-9))))
        g_out=self._calculate_adx(g_out,lookback)
        g_out=self._calculate_bollinger_bands(g_out,self.config.BOLLINGER_PERIOD)
        g_out=self._calculate_stochastic(g_out,self.config.STOCHASTIC_PERIOD)
        g_out = self._calculate_momentum(g_out)
        g_out = self._calculate_seasonality(g_out)
        g_out = self._calculate_candlestick_patterns(g_out)
        g_out['hour'] = g_out.index.hour;g_out['day_of_week'] = g_out.index.dayofweek
        g_out['market_regime']=np.where(g_out['ADX']>self.config.TREND_FILTER_THRESHOLD,1,0)
        sma_fast = g_out['Close'].rolling(window=20).mean(); sma_slow = g_out['Close'].rolling(window=50).mean()
        signal_series = pd.Series(np.where(sma_fast > sma_slow, 1.0, -1.0), index=g_out.index)
        g_out['primary_model_signal'] = signal_series.diff().fillna(0)
        g_out['market_volatility_index'] = g_out['ATR'].rolling(100).rank(pct=True)
        g_out['close_fracdiff'] = self._fractional_differentiation(g_out['Close'], d=0.5)
        return g_out

    def create_feature_stack(self, data_by_tf: Dict[str, pd.DataFrame]) -> pd.DataFrame:
        logger.info("-> Stage 2: Engineering Features from Multi-Timeframe Data...")

        base_tf, medium_tf, high_tf = self.roles['base'], self.roles['medium'], self.roles['high']
        if base_tf not in data_by_tf:
            logger.critical(f"Base timeframe '{base_tf}' data is missing. Cannot proceed."); return pd.DataFrame()

        df_base_list = [self._calculate_base_tf_native(group) for _, group in data_by_tf[base_tf].groupby('Symbol')]
        df_base = pd.concat(df_base_list).reset_index()
        df_merged = df_base

        if medium_tf and medium_tf in data_by_tf:
            df_medium_ctx = self._calculate_htf_features(data_by_tf[medium_tf], medium_tf, 50, 14)
            df_merged = pd.merge_asof(df_merged.sort_values('Timestamp'), df_medium_ctx.sort_values('Timestamp'), on='Timestamp', by='Symbol', direction='backward')

        if high_tf and high_tf in data_by_tf:
            df_high_ctx = self._calculate_htf_features(data_by_tf[high_tf], high_tf, 20, 14)
            df_merged = pd.merge_asof(df_merged.sort_values('Timestamp'), df_high_ctx.sort_values('Timestamp'), on='Timestamp', by='Symbol', direction='backward')

        df_final = df_merged.set_index('Timestamp').copy()

        if medium_tf:
            tf_id = medium_tf.upper()
            df_final[f'adx_x_{tf_id}_trend'] = df_final['ADX'] * df_final.get(f'{tf_id}_ctx_Trend', 0)
        if high_tf:
            tf_id = high_tf.upper()
            df_final[f'atr_x_{tf_id}_trend'] = df_final['ATR'] * df_final.get(f'{tf_id}_ctx_Trend', 0)

        logger.info("  - Generating anomaly detection scores...")
        df_final['anomaly_score'] = self._get_anomaly_scores(df_final, self.ANOMALY_FEATURES)

        feature_cols = [c for c in df_final.columns if c not in ['Open','High','Low','Close','RealVolume','Symbol']]
        df_final[feature_cols] = df_final.groupby('Symbol')[feature_cols].shift(1)

        df_final.replace([np.inf,-np.inf],np.nan,inplace=True)
        df_final.dropna(inplace=True)

        logger.info(f"  - Merged data and created features. Final dataset shape: {df_final.shape}")
        logger.info("[SUCCESS] Feature engineering complete.");return df_final

    def label_outcomes(self,df:pd.DataFrame,lookahead:int)->pd.DataFrame:
        logger.info("  - Generating trade labels with Regime-Adjusted Barriers...");
        labeled_dfs=[self._label_group(group,lookahead) for _,group in df.groupby('Symbol')];return pd.concat(labeled_dfs)

    def _label_group(self,group:pd.DataFrame,lookahead:int)->pd.DataFrame:
        if len(group)<lookahead+1:return group
        is_trending=group['market_regime'] == 1
        sl_multiplier=np.where(is_trending,2.0,1.5);tp_multiplier=np.where(is_trending,4.0,2.5)
        sl_atr_dynamic=group['ATR']*sl_multiplier;tp_atr_dynamic=group['ATR']*tp_multiplier
        outcomes=np.zeros(len(group));prices,lows,highs=group['Close'].values,group['Low'].values,group['High'].values

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

            if first_tp_long < first_sl_long: outcomes[i]=1
            if first_tp_short < first_sl_short: outcomes[i]=-1

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

        for i in range(len(group) - lookahead):
            signal = primary_signals[i]
            if signal == 0: continue

            sl_dist, tp_dist = sl_atr_dynamic[i], tp_atr_dynamic[i]
            if pd.isna(sl_dist) or sl_dist <= 1e-9: continue

            future_highs, future_lows = highs[i + 1:i + 1 + lookahead], lows[i + 1:i + 1 + lookahead]

            if signal > 0:
                tp_long, sl_long = prices[i] + tp_dist, prices[i] - sl_dist
                time_to_tp = np.where(future_highs >= tp_long)[0]
                time_to_sl = np.where(future_lows <= sl_long)[0]
                if len(time_to_tp) > 0 and (len(time_to_sl) == 0 or time_to_tp[0] < time_to_sl[0]):
                    outcomes[i] = 1

            elif signal < 0:
                tp_short, sl_short = prices[i] - tp_dist, prices[i] + sl_dist
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
                preds = np.where(max_probs > threshold, preds, 1)
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

        def custom_objective(trial:optuna.Trial):
            param={'objective':objective,'eval_metric':eval_metric,'booster':'gbtree','tree_method':'hist','seed':42,
                   'n_estimators':trial.suggest_int('n_estimators',200,1000,step=50),
                   'max_depth':trial.suggest_int('max_depth',3,8),
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
                preds=model.predict(X_val_scaled)
                f1 = f1_score(y_val,preds,average='macro', zero_division=0)
                pnl_map = {0: -1, 1: 0, 2: 1} if num_classes > 2 else {0: -1, 1: 1}
                pnl = pd.Series(preds).map(pnl_map)
                downside_returns = pnl[pnl < 0]
                downside_std = downside_returns.std()
                sortino = (pnl.mean() / downside_std) if downside_std > 0 else 0
                objective_score = (0.4 * f1) + (0.6 * sortino)

                X_stability_scaled = scaler.transform(X_stability)
                stability_preds = model.predict(X_stability_scaled)
                stability_pnl = pd.Series(stability_preds).map(pnl_map)
                if stability_pnl.sum() < 0: objective_score -= 0.5

                return objective_score
            except Exception as e:
                sys.stdout.write("\n")
                logger.warning(f"Trial {trial.number} failed with error: {e}")
                return -2.0

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
# 7. BACKTESTER & 8. PERFORMANCE ANALYZER
# =============================================================================
class Backtester:
    def __init__(self,config:ConfigModel):
        self.config=config
        self.is_meta_model = False

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

        for candle in candles:
            candle_date = candle['Timestamp'].date()
            if candle_date != current_day:
                finalize_day_metrics(current_day, equity)
                current_day, day_start_equity, day_peak_equity = candle_date, equity, equity

            # Update peaks and check for circuit breaker before any trade actions for this candle
            if not circuit_breaker_tripped:
                day_peak_equity = max(day_peak_equity, equity)
                chunk_peak_equity = max(chunk_peak_equity, equity)
                if equity > 0 and chunk_peak_equity > 0 and (chunk_peak_equity - equity) / chunk_peak_equity > self.config.MAX_DD_PER_CYCLE:
                    logger.warning(f"  - CYCLE CIRCUIT BREAKER TRIPPED! Drawdown exceeded {self.config.MAX_DD_PER_CYCLE:.0%} for this cycle. Closing all positions.")
                    circuit_breaker_tripped = True
                    trade_df = pd.DataFrame(trades)
                    breaker_context = {"num_trades_before_trip": len(trade_df), "pnl_before_trip": round(trade_df['PNL'].sum(), 2), "last_5_trades_pnl": [round(p, 2) for p in trade_df['PNL'].tail(5).tolist()]} if not trade_df.empty else {}
                    open_positions.clear() # Force close all positions immediately
            
            if equity <= 0:
                logger.critical("  - ACCOUNT BLOWN!")
                break

            symbol = candle['Symbol'] # The key fix: get the symbol for the current candle

            # --- 1. CHECK AND MANAGE EXISTING POSITIONS for this symbol ---
            if symbol in open_positions:
                pos = open_positions[symbol]
                pnl, exit_price, exit_reason = 0, None, ""

                # 1a. Check for partial profit-taking
                if self.config.USE_PARTIAL_PROFIT and not pos['partial_profit_taken']:
                    partial_tp_price = pos['entry_price'] + (pos['sl_dist'] * self.config.PARTIAL_PROFIT_TRIGGER_R * pos['direction'])
                    if (pos['direction'] == 1 and candle['High'] >= partial_tp_price) or \
                       (pos['direction'] == -1 and candle['Low'] <= partial_tp_price):
                        
                        partial_pnl = pos['risk_amt'] * self.config.PARTIAL_PROFIT_TRIGGER_R * self.config.PARTIAL_PROFIT_TAKE_PCT
                        equity += partial_pnl
                        day_peak_equity = max(day_peak_equity, equity)
                        equity_curve.append(equity)
                        
                        trades.append({
                            'ExecTime': candle['Timestamp'], 'Symbol': symbol, 'PNL': partial_pnl,
                            'Equity': equity, 'Confidence': pos['confidence'], 'Direction': pos['direction'],
                            'ExitReason': f"Partial TP ({self.config.PARTIAL_PROFIT_TAKE_PCT:.0%})"
                        })
                        
                        # Update position state after partial TP
                        pos['risk_amt'] *= (1 - self.config.PARTIAL_PROFIT_TAKE_PCT)
                        pos['sl'] = pos['entry_price'] # Move SL to Break-Even
                        pos['partial_profit_taken'] = True
                        if LOG_PARTIAL_PROFITS:
                            logger.info(f"  - Partial profit taken for {symbol}. Moved SL to breakeven.")

                # 1b. Check for final Stop Loss or Take Profit
                if pos['direction'] == 1:
                    if candle['Low'] <= pos['sl']: pnl, exit_price, exit_reason = -pos['risk_amt'], pos['sl'], "Stop Loss"
                    elif candle['High'] >= pos['tp']: pnl, exit_price, exit_reason = pos['risk_amt'] * pos['rr'], pos['tp'], "Take Profit"
                elif pos['direction'] == -1:
                    if candle['High'] >= pos['sl']: pnl, exit_price, exit_reason = -pos['risk_amt'], pos['sl'], "Stop Loss"
                    elif candle['Low'] <= pos['tp']: pnl, exit_price, exit_reason = pos['risk_amt'] * pos['rr'], pos['tp'], "Take Profit"
                
                if exit_price:
                    equity += pnl
                    day_peak_equity = max(day_peak_equity, equity)
                    equity_curve.append(equity)
                    
                    trades.append({
                        'ExecTime': candle['Timestamp'], 'Symbol': symbol, 'PNL': pnl,
                        'Equity': equity, 'Confidence': pos['confidence'], 'Direction': pos['direction'],
                        'ExitReason': exit_reason
                    })
                    del open_positions[symbol]
                    if equity <= 0: continue # Skip to next candle if account is blown

            # --- 2. CHECK FOR NEW TRADE ENTRY for this symbol ---
            if not circuit_breaker_tripped and symbol not in open_positions and len(open_positions) < self.config.MAX_CONCURRENT_TRADES:
                
                # Anomaly & Volatility Filters
                if candle.get('anomaly_score') == -1:
                    if LOG_ANOMALY_SKIPS and random.random() < 0.1:
                         logger.info(f"  - Skipping trade check for {symbol} due to anomaly detection at {candle['Timestamp']}")
                    continue
                
                vol_idx = candle.get('market_volatility_index', 0.5)
                if not (self.config.MIN_VOLATILITY_RANK <= vol_idx <= self.config.MAX_VOLATILITY_RANK):
                    continue

                # Signal Generation
                direction, confidence = 0, 0
                if self.is_meta_model:
                    prob_take_trade = candle.get('prob_1', 0)
                    primary_signal = candle.get('primary_model_signal', 0)
                    if prob_take_trade > confidence_threshold and primary_signal != 0:
                        direction = int(np.sign(primary_signal))
                        confidence = prob_take_trade
                else: # Standard 3-class model
                    if 'prob_short' in candle:
                        probs=np.array([candle['prob_short'],candle['prob_hold'],candle['prob_long']])
                        max_confidence=np.max(probs)
                        if max_confidence >= confidence_threshold:
                            pred_class=np.argmax(probs)
                            direction=1 if pred_class==2 else -1 if pred_class==0 else 0
                            confidence = max_confidence
                
                # Open new position if signal is valid
                if direction != 0:
                    atr=candle.get('ATR',0)
                    if pd.isna(atr) or atr<=1e-9: continue

                    if confidence>=self.config.CONFIDENCE_TIERS['ultra_high']['min']: tier_name='ultra_high'
                    elif confidence>=self.config.CONFIDENCE_TIERS['high']['min']: tier_name='high'
                    else: tier_name='standard'

                    tier=self.config.CONFIDENCE_TIERS[tier_name]
                    base_risk_amt = equity * self.config.BASE_RISK_PER_TRADE_PCT * tier['risk_mult']
                    risk_amt = min(base_risk_amt, self.config.RISK_CAP_PER_TRADE_USD)
                    sl_dist=(atr*1.5)+(atr*self.config.SPREAD_PCTG_OF_ATR)+(atr*self.config.SLIPPAGE_PCTG_OF_ATR)
                    tp_dist=(atr*1.5*tier['rr'])
                    if tp_dist<=0 or sl_dist<=0: continue

                    entry_price=candle['Close']
                    sl_price,tp_price=entry_price-sl_dist*direction,entry_price+tp_dist*direction
                    
                    open_positions[symbol]={
                        'direction':direction, 'entry_price': entry_price, 'sl':sl_price,'tp':tp_price, 
                        'risk_amt':risk_amt, 'rr':tier['rr'], 'confidence':confidence,
                        'sl_dist': sl_dist, 'partial_profit_taken': False
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
        final_exits_df = trades_df[trades_df['ExitReason'].isin(["Stop Loss", "Take Profit"])]
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
                (f"Annual Return (CAGR):", f"${m.get('cagr', 0):>15.2%}"),
                (f"Sharpe Ratio (annual):", f"${m.get('sharpe_ratio', 0):>15.2f}"),
                (f"Sortino Ratio (annual):", f"${m.get('sortino_ratio', 0):>15.2f}"),
                (f"Calmar Ratio / MAR:", f"{m.get('mar_ratio', 0):>15.2f}")
            ],
            "IV. RISK & DRAWDOWN ANALYSIS": [
                (f"Max Drawdown (Cycle):", f"{m.get('max_drawdown_pct', 0):>15.2f}% (${m.get('max_drawdown_abs', 0):,.2f})"),
                (f"Recovery Factor:", f"{m.get('recovery_factor', 0):>15.2f}"),
                (f"Longest Losing Streak:", f"{m.get('longest_loss_streak', 0):>15} trades")
            ],
            "V. TRADE-LEVEL STATISTICS": [
                (f"Total Unique Trades:", f"{m.get('total_trades', 0):>15}"),
                (f"Total Trade Events (incl. partials):", f"{m.get('total_trade_events', 0):>15}"),
                (f"Average Win Event:", f"${m.get('avg_win_amount', 0):>15,.2f}"),
                (f"Average Loss Event:", f"${m.get('avg_loss_amount', 0):>15,.2f}"),
                (f"Payoff Ratio (Full Trades):", f"{m.get('payoff_ratio', 0):>15.2f}")
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
        cycle_df_str = pd.DataFrame(cycle_metrics).to_string(index=False) if not pd.DataFrame(cycle_metrics).empty else "No trades were executed."
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
    """
    Validates and sanitizes critical numeric parameters from the AI to prevent model validation errors.
    """
    sanitized = suggestions.copy()
    
    # Define bounds based on ConfigModel constraints
    bounds = {
        'MAX_DD_PER_CYCLE': (0.05, 0.99),
        'MAX_CONCURRENT_TRADES': (1, 20),
        'PARTIAL_PROFIT_TRIGGER_R': (0.1, 10.0),
        'PARTIAL_PROFIT_TAKE_PCT': (0.1, 0.9),
        'OPTUNA_TRIALS': (10, 200),
        'LOOKAHEAD_CANDLES': (10, 500)
    }

    for key, (lower, upper) in bounds.items():
        if key in sanitized and isinstance(sanitized.get(key), (int, float)):
            original_value = sanitized[key]
            
            # Clamp the value
            clamped_value = max(lower, min(original_value, upper))
            
            # Ensure integer fields remain integers
            if isinstance(original_value, int):
                clamped_value = int(round(clamped_value))

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

    if not isinstance(freq_str, str):
        freq_str = str(freq_str)
    
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

def save_run_to_memory(config: ConfigModel, new_run_summary: Dict, current_memory: Dict) -> Optional[Dict]:
    try:
        with open(config.HISTORY_FILE_PATH, 'a') as f: f.write(json.dumps(new_run_summary) + '\n')
        logger.info(f"-> Run summary appended to history file: {config.HISTORY_FILE_PATH}")
    except IOError as e: logger.error(f"Could not write to history file: {e}")
    current_champion = current_memory.get("champion_config")
    new_calmar = new_run_summary.get("final_metrics", {}).get("mar_ratio", 0)
    is_new_champion = (current_champion is None or (new_calmar is not None and new_calmar > current_champion.get("final_metrics", {}).get("mar_ratio", -np.inf)))
    if is_new_champion:
        champion_to_save = new_run_summary
        champion_calmar = current_champion.get("final_metrics", {}).get("mar_ratio", -np.inf) if current_champion else -np.inf
        logger.info(f"NEW CHAMPION! Current run's MAR Ratio ({new_calmar:.2f}) beats previous champion's ({champion_calmar:.2f}).")
    else:
        champion_to_save = current_champion
        champ_calmar_val = current_champion.get("final_metrics", {}).get("mar_ratio", 0)
        logger.info(f"Current run's MAR Ratio ({new_calmar:.2f}) did not beat champion's ({champ_calmar_val:.2f}).")
    try:
        with open(config.CHAMPION_FILE_PATH, 'w') as f: json.dump(champion_to_save, f, indent=4)
        logger.info(f"-> Champion file updated: {config.CHAMPION_FILE_PATH}")
    except (IOError, TypeError) as e: logger.error(f"Could not write to champion file: {e}")
    return champion_to_save

def initialize_playbook(base_path: str) -> Dict:
    results_dir = os.path.join(base_path, "Results"); os.makedirs(results_dir, exist_ok=True)
    playbook_path = os.path.join(results_dir, "strategy_playbook.json")
    DEFAULT_PLAYBOOK = {
        "VolatilityBreakout": {"description": "Enters on explosive breakouts from low-volatility consolidations.", "features": ["ATR", "bollinger_bandwidth", "ADX", "hour", "day_of_week", "anomaly_score", "close_fracdiff"], "lookahead_range": [60, 120], "dd_range": [0.20, 0.35]},
        "RangeBound": {"description": "Trades reversals in a sideways channel, filtered by low ADX.", "features": ["ADX", "RSI", "stoch_k", "stoch_d", "bollinger_bandwidth", "hour", "is_doji"], "lookahead_range": [20, 60], "dd_range": [0.10, 0.20]},
        "TrendPullback": {"description": "Enters on pullbacks (RSI) in the direction of an established long-term trend.", "features": ["DAILY_ctx_Trend", "ADX", "RSI", "H1_ctx_Trend", "momentum_10", "hour", "close_fracdiff"], "lookahead_range": [50, 150], "dd_range": [0.15, 0.30]},
        "GNN_Market_Structure": {"description": "Uses a GNN to model inter-asset correlations for predictive features.", "features": [], "lookahead_range": [80, 150], "dd_range": [0.15, 0.30], "requires_gnn": True},
        "Meta_Labeling_Filter": {"description": "Uses a secondary ML filter to improve a simple primary model's signal quality.", "features": ["ADX", "RSI", "ATR", "bollinger_bandwidth", "H1_ctx_Trend", "DAILY_ctx_Trend", "momentum_20"], "lookahead_range": [50, 100], "dd_range": [0.10, 0.25], "requires_meta_labeling": True},
        "FilteredBreakout": {"description": "A hybrid that trades high-volatility breakouts (ATR, Bollinger) but only in the direction of the long-term daily trend.", "features": ["ATR", "bollinger_bandwidth", "DAILY_ctx_Trend", "ADX", "hour", "anomaly_score"], "lookahead_range": [60, 120], "dd_range": [0.20, 0.35] },
    }
    if not os.path.exists(playbook_path):
        logger.warning(f"'strategy_playbook.json' not found. Seeding a new one with default strategies at: {playbook_path}")
        try:
            with open(playbook_path, 'w') as f: json.dump(DEFAULT_PLAYBOOK, f, indent=4)
            return DEFAULT_PLAYBOOK
        except IOError as e: logger.error(f"Failed to create playbook file: {e}. Using in-memory default."); return DEFAULT_PLAYBOOK
    try:
        with open(playbook_path, 'r') as f: playbook = json.load(f)
        updated = any(key not in playbook for key in DEFAULT_PLAYBOOK)
        if updated:
            logger.info("Updating playbook with new default strategies...")
            playbook.update({k: v for k, v in DEFAULT_PLAYBOOK.items() if k not in playbook})
            with open(playbook_path, 'w') as f: json.dump(playbook, f, indent=4)
        logger.info(f"Successfully loaded dynamic playbook from {playbook_path}"); return playbook
    except (json.JSONDecodeError, IOError) as e: logger.error(f"Failed to load or parse playbook file: {e}. Using in-memory default."); return DEFAULT_PLAYBOOK

def load_nickname_ledger(ledger_path: str) -> Dict:
    logger.info("-> Loading Nickname Ledger...")
    if os.path.exists(ledger_path):
        try:
            with open(ledger_path, 'r') as f:
                ledger = json.load(f)
                logger.info(f"  - Loaded existing nickname ledger from: {ledger_path}")
                return ledger
        except (json.JSONDecodeError, IOError) as e:
            logger.error(f"  - Could not read or parse nickname ledger. Creating a new one. Error: {e}")
    return {}

def perform_strategic_review(history: Dict, directives_path: str) -> Tuple[Dict, List[Dict]]:
    logger.info("--- STRATEGIC REVIEW: Analyzing long-term strategy health...")
    health_report = {}
    directives = []
    historical_runs = history.get("historical_runs", [])
    if len(historical_runs) < 3:
        logger.info("--- STRATEGIC REVIEW: Insufficient history for a full review.")
        return health_report, directives

    strategy_names = set(run.get('strategy_name') for run in historical_runs if run.get('strategy_name'))
    for name in strategy_names:
        strategy_runs = [run for run in historical_runs if run.get('strategy_name') == name]
        if len(strategy_runs) < 3: continue
        failures = sum(1 for run in strategy_runs if run.get("final_metrics", {}).get("mar_ratio", 0) < 0.1)
        chronic_failure_rate = failures / len(strategy_runs)
        total_cycles = sum(len(run.get("cycle_details", [])) for run in strategy_runs)
        breaker_trips = sum(sum(1 for cycle in run.get("cycle_details", []) if cycle.get("Status") == "Circuit Breaker") for run in strategy_runs)
        circuit_breaker_freq = (breaker_trips / total_cycles) if total_cycles > 0 else 0
        health_report[name] = {"ChronicFailureRate": f"{chronic_failure_rate:.0%}", "CircuitBreakerFrequency": f"{circuit_breaker_freq:.0%}", "RunsAnalyzed": len(strategy_runs)}

    recent_runs = historical_runs[-3:]
    if len(recent_runs) >= 3 and len(set(r.get('strategy_name') for r in recent_runs)) == 1:
        stagnant_strat_name = recent_runs[0].get('strategy_name')
        calmar_values = [r.get("final_metrics", {}).get("mar_ratio", 0) for r in recent_runs]
        if calmar_values[2] <= calmar_values[1] <= calmar_values[0]:
            if stagnant_strat_name in health_report: health_report[stagnant_strat_name]["StagnationWarning"] = True
            stagnation_directive = {"action": "FORCE_EXPLORATION", "strategy": stagnant_strat_name, "reason": f"Stagnation: No improvement over last 3 runs (MAR Ratios: {[round(c, 2) for c in calmar_values]})."}
            directives.append(stagnation_directive)
            logger.warning(f"--- STRATEGIC REVIEW: Stagnation detected for '{stagnant_strat_name}'. Creating directive.")

    try:
        with open(directives_path, 'w') as f: json.dump(directives, f, indent=4)
        logger.info(f"--- STRATEGIC REVIEW: Directives saved to {directives_path}" if directives else "--- STRATEGIC REVIEW: No new directives generated. Cleared old directives file.")
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
        roles['medium'] = sorted_tfs[1]
        roles['high'] = sorted_tfs[2]
        if len(sorted_tfs) > 3: logger.warning(f"Detected {len(sorted_tfs)} timeframes. Using {roles['base']}, {roles['medium']}, and {roles['high']}.")
    logger.info(f"Dynamically determined timeframe roles: {roles}")
    return roles

def run_single_instance(fallback_config: Dict, framework_history: Dict, playbook: Dict, nickname_ledger: Dict, directives: List[Dict], api_interval_seconds: int):
    run_timestamp_str = datetime.now().strftime("%Y%m%d-%H%M%S")
    gemini_analyzer = GeminiAnalyzer()
    api_timer = APITimer(interval_seconds=api_interval_seconds)

    current_config_dict = fallback_config.copy()
    current_config_dict['run_timestamp'] = run_timestamp_str

    # Use a temporary config object that doesn't create directories for initial data processing
    temp_config = ConfigModel(**{**current_config_dict, 'nickname': '', 'run_timestamp': 'temp'})
    data_loader = DataLoader(temp_config)

    all_files = [f for f in os.listdir(current_config_dict['BASE_PATH']) if f.endswith(('.csv', '.txt')) and re.match(r'^[A-Z0-9]+_[A-Z0-9]+', f)]
    if not all_files:
        logger.critical("No data files found in the base path matching pattern. Exiting.")
        return

    data_by_tf, detected_timeframes = data_loader.load_and_parse_data(all_files)
    if not data_by_tf: return

    tf_roles = determine_timeframe_roles(detected_timeframes)
    fe = FeatureEngineer(temp_config, tf_roles)
    full_df = fe.create_feature_stack(data_by_tf)
    if full_df.empty:
        logger.critical("Feature engineering resulted in an empty dataframe. Exiting.")
        return

    data_summary = {}
    summary_df = full_df.reset_index()
    assets = summary_df['Symbol'].unique().tolist()
    data_summary['assets_detected'] = assets
    data_summary['time_range'] = {'start': summary_df['Timestamp'].min().isoformat(), 'end': summary_df['Timestamp'].max().isoformat()}
    data_summary['timeframes_used'] = tf_roles
    asset_stats = {asset: {'avg_atr': round(full_df[full_df['Symbol'] == asset]['ATR'].mean(), 5), 'avg_adx': round(full_df[full_df['Symbol'] == asset]['ADX'].mean(), 2), 'trending_pct': f"{round(full_df[full_df['Symbol'] == asset]['market_regime'].mean() * 100, 1)}%"} for asset in assets}
    data_summary['asset_statistics'] = asset_stats
    if len(assets) > 1:
        pivot_df = full_df.pivot(columns='Symbol', values='Close').ffill().bfill()
        data_summary['asset_correlation_matrix'] = pivot_df.corr().round(3).to_dict()

    script_name = os.path.basename(__file__) if '__file__' in locals() else fallback_config["REPORT_LABEL"]
    version_label = script_name.replace(".py", "")
    health_report, _ = perform_strategic_review(framework_history, fallback_config['DIRECTIVES_FILE_PATH'])

    ai_setup = api_timer.call(gemini_analyzer.get_initial_run_setup, version_label, nickname_ledger, framework_history, playbook, health_report, directives, data_summary)

    if not ai_setup:
        logger.critical("AI-driven setup failed. Using fallback configuration and exiting.")
        return
    
    # V181 FIX: Sanitize initial setup from AI
    ai_setup = _sanitize_ai_suggestions(ai_setup)
    
    if 'RETRAINING_FREQUENCY' in ai_setup:
        ai_setup['RETRAINING_FREQUENCY'] = _sanitize_frequency_string(ai_setup['RETRAINING_FREQUENCY'])

    current_config_dict.update(ai_setup)

    if isinstance(ai_setup.get("nickname"), str) and ai_setup.get("nickname"):
        new_nickname = ai_setup["nickname"]
        nickname_ledger[version_label] = new_nickname
        bootstrap_config_for_path = ConfigModel(**fallback_config, run_timestamp="init", nickname="init")
        try:
            with open(bootstrap_config_for_path.NICKNAME_LEDGER_PATH, 'w') as f:
                json.dump(nickname_ledger, f, indent=4)
            logger.info(f"  - Saved new nickname '{new_nickname}' to ledger.")
        except IOError as e:
            logger.error(f"  - Failed to save the new nickname to the ledger: {e}")

    current_config_dict['REPORT_LABEL'] = version_label
    current_config_dict['nickname'] = nickname_ledger.get(version_label, f"Run-{run_timestamp_str}")

    config = ConfigModel(**current_config_dict)

    if not config.selected_features and config.strategy_name in playbook:
        default_features = playbook[config.strategy_name].get("features")
        if default_features:
            logger.warning(f"AI response for initial setup was missing 'selected_features'.")
            logger.warning(f"Using default feature list from playbook for strategy '{config.strategy_name}'.")
            config.selected_features = default_features
        else:
            logger.critical(f"FATAL: AI did not provide features and no default features exist in playbook for '{config.strategy_name}'.")
            return
    elif not config.selected_features and not playbook.get(config.strategy_name, {}).get('requires_gnn'):
        logger.critical(f"FATAL: AI did not provide features and strategy '{config.strategy_name}' is not in playbook or has no defaults.")
        return


    file_handler = RotatingFileHandler(config.LOG_FILE_PATH, maxBytes=5*1024*1024, backupCount=2)
    file_handler.setLevel(logging.DEBUG)
    file_formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(file_formatter)
    logger.addHandler(file_handler)

    logger.info(f"--- Run Initialized: {config.nickname} | Strategy: {config.strategy_name} ---")
    if config.analysis_notes:
        logger.info(f"--- AI Analysis Notes: {config.analysis_notes} ---")

    all_available_features = [c for c in full_df.columns if c not in ['Open','High','Low','Close','RealVolume','Symbol','Timestamp','primary_model_signal','target']]
    start_date, end_date = full_df.index.min(), full_df.index.max()
    train_window, forward_gap = pd.to_timedelta(config.TRAINING_WINDOW), pd.to_timedelta(config.FORWARD_TEST_GAP)
    
    retrain_freq_str = _sanitize_frequency_string(config.RETRAINING_FREQUENCY)
    retrain_offset = pd.tseries.frequencies.to_offset(retrain_freq_str)
    test_start_date = start_date + train_window + forward_gap
    retraining_dates = pd.date_range(start=test_start_date, end=end_date, freq=retrain_freq_str)
    
    total_cycles = len(retraining_dates)
    logger.info(f"Walk-forward analysis will run for {total_cycles} cycles with a retraining frequency of {retrain_freq_str}.")

    aggregated_trades, aggregated_equity_curve = pd.DataFrame(), pd.Series([config.INITIAL_CAPITAL])
    in_run_historical_cycles, aggregated_shap = [], pd.DataFrame()
    last_equity, quarantine_list, failed_cycles_in_a_row = config.INITIAL_CAPITAL, [], 0
    run_summary = {"script_version": config.REPORT_LABEL, "nickname": config.nickname, "strategy_name": config.strategy_name, "run_start_ts": config.run_timestamp, "initial_params": config.model_dump(mode='json')}
    aggregated_daily_dd_reports = []
    
    original_retraining_dates = list(retraining_dates)
    cycle_num = 0
    short_cycle_streak = 0

    while cycle_num < len(original_retraining_dates):
        period_start_date = original_retraining_dates[cycle_num]
        logger.info(f"\n--- Starting Cycle [{cycle_num + 1}/{len(original_retraining_dates)}] ---")
        cycle_start_time = time.time()
        
        retrain_freq_str = _sanitize_frequency_string(config.RETRAINING_FREQUENCY)
        is_short_cycle = 'M' not in retrain_freq_str.upper() and 'Y' not in retrain_freq_str.upper() and pd.to_timedelta(retrain_freq_str) < pd.Timedelta(days=30)

        if short_cycle_streak >= 4:
            logger.warning("Forced Exploration Triggered: AI has suggested short cycles for 4 consecutive periods. Overriding to test a longer cycle.")
            config.RETRAINING_FREQUENCY = '90D'
            short_cycle_streak = 0
        
        if is_short_cycle: short_cycle_streak += 1
        else: short_cycle_streak = 0
        
        retrain_offset = pd.tseries.frequencies.to_offset(retrain_freq_str)
        train_end = period_start_date - forward_gap
        train_start = train_end - train_window
        test_end = period_start_date + retrain_offset
        if test_end > end_date: test_end = end_date

        df_train_raw, df_test = full_df.loc[train_start:train_end].copy(), full_df.loc[period_start_date:test_end].copy()
        if df_train_raw.empty or df_test.empty:
            logger.warning(f"  - No data for cycle [{cycle_num + 1}/{len(original_retraining_dates)}]. Skipping.")
            cycle_num += 1
            continue
        logger.info(f"  - Dates | Train: {train_start.date()}-{train_end.date()} | Test: {period_start_date.date()}-{test_end.date()}")

        strategy_details = playbook.get(config.strategy_name, {})
        is_meta_model = strategy_details.get("requires_meta_labeling", False)

        fe.config = config
        df_train_labeled = fe.label_meta_outcomes(df_train_raw, config.LOOKAHEAD_CANDLES) if is_meta_model else fe.label_outcomes(df_train_raw, config.LOOKAHEAD_CANDLES)
        if df_train_labeled.empty or ('target' in df_train_labeled and df_train_labeled['target'].abs().sum() == 0):
            logger.warning("  - No valid labels generated for this training cycle. Skipping.")
            cycle_num += 1
            continue

        if 'selected_features' in config.model_dump():
            config.selected_features = [f for f in config.selected_features if f in all_available_features]

        trainer = ModelTrainer(config)
        feature_list_to_use = config.selected_features if not strategy_details.get("requires_gnn") else trainer.GNN_BASE_FEATURES
        train_result = trainer.train(df_train_labeled, feature_list_to_use, strategy_details)
        if not train_result:
            logger.error("  - Model training failed for this cycle. Skipping.")
            cycle_num += 1
            continue

        pipeline, threshold = train_result
        if trainer.shap_summary is not None:
            aggregated_shap = trainer.shap_summary if aggregated_shap.empty else aggregated_shap.add(trainer.shap_summary, fill_value=0)

        X_test = trainer._get_gnn_embeddings_for_test(df_test) if trainer.is_gnn_model else df_test[feature_list_to_use].copy().fillna(0)
        if not X_test.empty:
            probs = pipeline.predict_proba(X_test)
            if is_meta_model and probs.shape[1] == 2: df_test[['prob_0', 'prob_1']] = probs
            elif not is_meta_model and probs.shape[1] == 3: df_test[['prob_short', 'prob_hold', 'prob_long']] = probs

        backtester = Backtester(config)
        trades, equity_curve, breaker_tripped, breaker_context, daily_dd_report = backtester.run_backtest_chunk(df_test, threshold, last_equity, is_meta_model)
        aggregated_daily_dd_reports.append(daily_dd_report)
        
        cycle_pnl = equity_curve.iloc[-1] - last_equity if not equity_curve.empty else 0
        cycle_result = {"StartDate": period_start_date.date().isoformat(), "EndDate": test_end.date().isoformat(), "NumTrades": len(trades), "PNL": round(cycle_pnl, 2), "Status": "Circuit Breaker" if breaker_tripped else "Completed"}
        if breaker_tripped: cycle_result["BreakerContext"] = breaker_context
        in_run_historical_cycles.append(cycle_result)

        if not trades.empty:
            aggregated_trades = pd.concat([aggregated_trades, trades], ignore_index=True)
            aggregated_equity_curve = pd.concat([aggregated_equity_curve, equity_curve.iloc[1:]], ignore_index=True)
            last_equity = equity_curve.iloc[-1]
            
            mc_days = retrain_offset.n * 30 if hasattr(retrain_offset, 'months') else retrain_offset.n
            price_paths = run_monte_carlo_simulation(df_test['Close'], n_days=mc_days)
            var_5pct = np.percentile(price_paths[-1], 5)
            mc_results = {'var_5pct': var_5pct, 'current_equity': last_equity}
        else:
            mc_results = {'var_5pct': last_equity, 'current_equity': last_equity}

        if breaker_tripped:
            failed_cycles_in_a_row += 1
            if failed_cycles_in_a_row >= 2:
                if config.strategy_name not in quarantine_list: quarantine_list.append(config.strategy_name)
                intervention_suggestion = api_timer.call(gemini_analyzer.propose_strategic_intervention, in_run_historical_cycles[-2:], playbook, config.strategy_name, quarantine_list)
                
                if intervention_suggestion and intervention_suggestion.get("strategy_name") in playbook:
                    # V181 FIX: Sanitize before applying
                    intervention_suggestion = _sanitize_ai_suggestions(intervention_suggestion)
                    if 'RETRAINING_FREQUENCY' in intervention_suggestion:
                        intervention_suggestion['RETRAINING_FREQUENCY'] = _sanitize_frequency_string(intervention_suggestion['RETRAINING_FREQUENCY'])
                    if 'selected_features' in intervention_suggestion:
                        intervention_suggestion['selected_features'] = [f for f in intervention_suggestion['selected_features'] if f in all_available_features]

                    logger.info(f"  - Intervention successful. Switching to strategy: {intervention_suggestion['strategy_name']}")
                    config = ConfigModel(**{**config.model_dump(mode='json'), **intervention_suggestion})
                    failed_cycles_in_a_row = 0
                else:
                    logger.error("  - Strategic intervention FAILED. Reducing risk and continuing.")
                    config.MAX_DD_PER_CYCLE = max(0.05, config.MAX_DD_PER_CYCLE * 0.75)
        else:
            failed_cycles_in_a_row = 0

        suggested_params = api_timer.call(gemini_analyzer.analyze_cycle_and_suggest_changes, in_run_historical_cycles, all_available_features, strategy_details, daily_dd_report, mc_results)
        if suggested_params:
            # V181 FIX: Sanitize before applying
            suggested_params = _sanitize_ai_suggestions(suggested_params)
            logger.info(f"  - AI suggests updating params: {suggested_params}")

            if 'selected_features' in suggested_params:
                suggested_params['selected_features'] = [f for f in suggested_params['selected_features'] if f in all_available_features]
            if 'RETRAINING_FREQUENCY' in suggested_params:
                suggested_params['RETRAINING_FREQUENCY'] = _sanitize_frequency_string(suggested_params['RETRAINING_FREQUENCY'])

            config = ConfigModel(**{**config.model_dump(mode='json'), **suggested_params})
        
        cycle_num += 1
        logger.info(f"--- Cycle complete. PNL: ${cycle_pnl:,.2f} | Final Equity: ${last_equity:,.2f} | Time: {time.time() - cycle_start_time:.2f}s ---")

    final_metrics = {}
    if not aggregated_trades.empty:
        pa = PerformanceAnalyzer(config)
        final_metrics = pa.generate_full_report(aggregated_trades, aggregated_equity_curve, in_run_historical_cycles, aggregated_shap, framework_history, aggregated_daily_dd_reports)

    run_summary.update({"run_end_ts": datetime.now().strftime("%Y%m%d-%H%M%S"), "final_metrics": final_metrics, "cycle_details": in_run_historical_cycles, "top_5_features": aggregated_shap.head(5).index.tolist() if not aggregated_shap.empty else [], "final_params": config.model_dump(mode='json')})
    save_run_to_memory(config, run_summary, framework_history)

    logger.removeHandler(file_handler)
    file_handler.close()

def main():
    CONTINUOUS_RUN_HOURS = 0; MAX_RUNS = 1
    fallback_config={
        "BASE_PATH": os.getcwd(), "REPORT_LABEL": "ML_Framework_V181_Sanitization",
        "strategy_name": "Meta_Labeling_Filter", "INITIAL_CAPITAL": 100000.0,
        "CONFIDENCE_TIERS": {'ultra_high':{'min':0.8,'risk_mult':1.2,'rr':3.0},'high':{'min':0.7,'risk_mult':1.0,'rr':2.5},'standard':{'min':0.6,'risk_mult':0.8,'rr':2.0}},
        "BASE_RISK_PER_TRADE_PCT": 0.01,"RISK_CAP_PER_TRADE_USD": 5000.0,
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
        "PARTIAL_PROFIT_TAKE_PCT": 0.5
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

        if MAX_RUNS > 1 and run_count >= MAX_RUNS:
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
    if os.name == 'nt': os.system("chcp 65001 > nul")
    main()

# End_To_End_Advanced_ML_Trading_Framework_PRO_V181_Sanitization.py