# End_To_End_Advanced_ML_Trading_Framework_PRO_V140_GNN_Hybrid_Integration.py
#
# V140 UPDATE (GNN Hybrid Integration):
# 1. GNN MODULE INTEGRATED: Transplanted the Graph Neural Network logic from the
#    hybrid script directly into the ModelTrainer. The trainer can now operate in
#    a 'GNN Mode' to generate features based on inter-asset correlations.
# 2. DYNAMIC GNN ACTIVATION: Added a 'GNN_Market_Structure' strategy to the default
#    playbook with a `requires_gnn: true` flag. The ModelTrainer uses this flag
#    to dynamically trigger the GNN embedding pipeline, replacing standard features.
# 3. DECOUPLED PREDICTION: Prediction logic has been moved out of the Backtester
#    and into the main walk-forward loop. This handles the separate prediction
#    needs of GNN vs. standard models before backtesting, simplifying the process.
# 4. CONFIG & ENVIRONMENT: The ConfigModel and default parameters now include GNN
#    hyperparameters (GNN_EMBEDDING_DIM, GNN_EPOCHS). The script also checks for
#    PyTorch/PyG libraries on startup.
#
# V139 UPDATE (AI Consistency FIX):
# 1. CRITICAL FIX - Misleading AI Prompt: Removed unsupported trading concepts from the
#    "Circuit Breaker" AI prompt to align with backtester capabilities.
# 2. BUG FIX - Stale Configuration State: Ensured the main config object is updated
#    when a feature list fallback occurs.

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
from typing import List, Dict, Any, Optional, Tuple
from collections import defaultdict
import copy

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
from pydantic import BaseModel, DirectoryPath, confloat, conint

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
    # Define dummy classes if import fails to prevent crashes on script load
    class GCNConv: pass
    class Adam: pass
    class Data: pass
    def F(): pass
    torch = None

def setup_logging() -> logging.Logger:
    if logger.hasHandlers():
        logger.handlers.clear()
    logger.setLevel(logging.DEBUG)
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
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
# 2. GEMINI AI ANALYZER
# =============================================================================
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

    def _sanitize_value(self, value):
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
        if len(prompt) > 28000: logger.warning("Prompt is very large, may risk exceeding token limits.")
        payload = {"contents": [{"parts": [{"text": prompt}]}]}
        sanitized_payload = self._sanitize_dict(payload)
        try:
            response = requests.post(self.api_url, headers=self.headers, data=json.dumps(sanitized_payload), timeout=120)
            response.raise_for_status()
            result = response.json()
            if "candidates" in result and result["candidates"] and "content" in result["candidates"][0] and "parts" in result["candidates"][0]["content"]:
                return result["candidates"][0]["content"]["parts"][0]["text"]
            else:
                logger.error(f"Invalid Gemini response structure: {result}"); return "{}"
        except requests.exceptions.RequestException as e:
            logger.error(f"Gemini API request failed: {e}"); return "{}"
        except (KeyError, IndexError) as e:
            logger.error(f"Failed to extract text from Gemini response: {e} - Response: {response.text}"); return "{}"

    def _extract_json_from_response(self, response_text: str) -> Dict:
        try:
            match = re.search(r"```json\s*(\{.*?\})\s*```", response_text, re.DOTALL)
            if not match: match = re.search(r"(\{.*\})", response_text, re.DOTALL)
            return json.loads(match.group(1).strip()) if match else {}
        except (json.JSONDecodeError, AttributeError) as e:
            logger.error(f"Could not parse JSON from response: {e}\nResponse text: {response_text}"); return {}

    def get_pre_flight_config(self, memory: Dict, playbook: Dict, health_report: Dict, directives: List[Dict], fallback_config: Dict, exploration_rate: float) -> Dict:
        if not self.api_key_valid:
            logger.warning("No API key. Skipping Pre-Flight Check and using default config.")
            return fallback_config
        if not memory.get("historical_runs"):
            logger.info("Framework memory is empty. Using default config for the first run.")
            return fallback_config

        logger.info("-> Stage 0: Pre-Flight Analysis of Framework Memory...")
        champion_strategy = memory.get("champion_config", {}).get("strategy_name")
        is_exploration = random.random() < exploration_rate and champion_strategy is not None
        
        directive_str = "No specific directives for this run."
        if directives:
            directive_str = "CRITICAL DIRECTIVES FOR THIS RUN:\n"
            for d in directives:
                if d.get('action') == 'QUARANTINE':
                    directive_str += f"- The following strategies are underperforming and are QUARANTINED. DO NOT SELECT them: {d['strategies']}\n"
                if d.get('action') == 'FORCE_EXPLORATION':
                    directive_str += f"- The champion '{d['strategy']}' is stagnating. You MUST SELECT a DIFFERENT strategy to force exploration.\n"
        
        health_report_str = "No long-term health report available."
        if health_report:
            health_report_str = f"STRATEGIC HEALTH ANALYSIS (Lower scores are better):\n{json.dumps(health_report, indent=2)}\n\n"

        # Filter out GNN strategies if PyG is not available
        if not GNN_AVAILABLE:
            original_playbook_size = len(playbook)
            playbook = {k: v for k, v in playbook.items() if not v.get("requires_gnn")}
            if len(playbook) < original_playbook_size:
                logger.warning("GNN strategies filtered from playbook due to missing libraries.")

        if is_exploration:
            logger.info(f"--- ENTERING EXPLORATION MODE (Chance: {exploration_rate:.0%}) ---")
            quarantined_strats = [d.get('strategies', []) for d in directives if d.get('action') == 'QUARANTINE']
            quarantined_strats = [item for sublist in quarantined_strats for item in sublist]
            
            available_strategies = [s for s in playbook if s != champion_strategy and s not in quarantined_strats]
            if not available_strategies:
                available_strategies = [s for s in playbook if s not in quarantined_strats]
            if not available_strategies:
                available_strategies = list(playbook.keys())
            
            chosen_strategy = random.choice(available_strategies) if available_strategies else "TrendFollower"
            prompt = (
                "You are a trading strategist in **EXPLORATION MODE**. Your goal is to test a non-champion strategy to gather new performance data.\n"
                f"Your randomly assigned strategy to test is **'{chosen_strategy}'**. "
                "Based on the playbook definition for this strategy, propose a reasonable starting configuration.\n\n"
                "Respond ONLY with a valid JSON object containing: `strategy_name`, `selected_features` (if applicable), `LOOKAHEAD_CANDLES`, `OPTUNA_TRIALS`, and `MAX_DD_PER_CYCLE`. "
                "CRITICAL: `MAX_DD_PER_CYCLE` must be a float between 0.05 and 0.9 (e.g., 0.2 for 20% drawdown).\n\n"
                f"STRATEGY PLAYBOOK:\n{json.dumps(playbook, indent=2)}\n\n"
            )
        else:
            logger.info("--- ENTERING EXPLOITATION MODE (Optimizing Champion) ---")
            prompt = (
                "You are a master trading strategist. Your task is to select the optimal **master strategy** for the next run. Your ultimate goal is to produce a high and STABLE risk-adjusted return (MAR Ratio).\n\n"
                "**INSTRUCTIONS:**\n"
                "1. **Follow Directives**: You MUST follow any instructions in the `CRITICAL DIRECTIVES` section. This overrides all other considerations.\n"
                "2. **Review Strategic Health**: Review the `STRATEGIC HEALTH ANALYSIS`. Heavily penalize strategies with high `ChronicFailureRate` or `CircuitBreakerFrequency`.\n"
                "3. **Check for Stagnation**: If a strategy has a `StagnationWarning`, strongly consider choosing a **different** strategy to force exploration.\n"
                "4. **Select Strategy & Features**: Choose the `strategy_name`. If the strategy is based on technical indicators, select a small, relevant list of `selected_features`. If it's a GNN strategy, the features are learned automatically.\n"
                "5. **Define Initial Parameters**: Suggest initial values for `LOOKAHEAD_CANDLES`, `OPTUNA_TRIALS`, and `MAX_DD_PER_CYCLE`. CRITICAL: `MAX_DD_PER_CYCLE` must be a float between 0.05 and 0.9. BE CONSERVATIVE.\n\n"
                "Respond ONLY with a valid JSON object containing `strategy_name`, `selected_features`, `LOOKAHEAD_CANDLES`, `OPTUNA_TRIALS`, and `MAX_DD_PER_CYCLE` at the top level.\n\n"
                f"**{directive_str}**\n\n"
                f"{health_report_str}"
                f"STRATEGY PLAYBOOK (Your options):\n{json.dumps(playbook, indent=2)}\n\n"
                f"FRAMEWORK MEMORY (Champion & History):\n{json.dumps(self._sanitize_dict(memory), indent=2)}"
            )

        response_text = self._call_gemini(prompt)
        logger.info(f"  - Pre-Flight Analysis (Raw): {response_text}")
        suggestions = self._extract_json_from_response(response_text)

        if suggestions and "strategy_name" in suggestions:
            final_config = fallback_config.copy()
            final_config.update(suggestions)
            logger.info(f"  - Pre-Flight Check complete. AI chose strategy '{final_config['strategy_name']}' with params: {suggestions}")
            return final_config
        else:
            logger.warning("  - Pre-Flight Check failed to select a strategy. Using fallback.")
            return fallback_config
            
    def analyze_cycle_and_suggest_changes(self, historical_results: List[Dict], available_features: List[str]) -> Dict:
        if not self.api_key_valid: return {}
        logger.info("  - AI Strategist: Tuning selected strategy based on recent performance...")
        recent_history = historical_results[-5:]
        last_cycle = recent_history[-1] if recent_history else {}

        if last_cycle.get("status") == "Circuit Breaker":
            logger.warning("  - AI Strategist: CIRCUIT BREAKER DETECTED. Engaging safety-oriented prompt.")
            prompt = (
                "You are an expert trading model analyst acting in a **SAFETY FIRST** capacity. The previous trading cycle hit its maximum drawdown ('Circuit Breaker'). Your primary goal is to **REDUCE RISK** to prevent this from happening again.\n\n"
                "**CRITICAL INSTRUCTIONS:**\n"
                "1.  **You MUST suggest changes that lower risk.** This is not optional.\n"
                "2.  **DO NOT increase `MAX_DD_PER_CYCLE`.** You can only keep it the same or decrease it.\n"
                "3.  Consider other risk-reducing changes: **reduce `LOOKAHEAD_CANDLES`** for shorter-term trades, **reduce `OPTUNA_TRIALS`** to prevent overfitting, or **simplify the model by reducing `selected_features`** (if not a GNN model).\n"
                "4.  Your ultimate objective is to create a STABLE strategy that generates a good risk-adjusted return (MAR Ratio), not to chase maximum profit.\n\n"
                "Respond ONLY with a valid JSON object containing your new, safer parameters: `MAX_DD_PER_CYCLE`, `LOOKAHEAD_CANDLES`, `OPTUNA_TRIALS`, and `selected_features`.\n\n"
                f"FAILED CYCLE CONTEXT:\n{json.dumps(self._sanitize_dict(last_cycle), indent=2)}\n\n"
                f"AVAILABLE FEATURES FOR THIS STRATEGY:\n{available_features}"
            )
        else:
            prompt = (
                "You are an expert trading model analyst. Your goal is to tune the parameters of a **pre-selected master strategy** to improve its risk-adjusted return (MAR Ratio) based on recent performance.\n\n"
                "Analyze the recent cycle history and suggest parameter adjustments for the next cycle. You can adjust `MAX_DD_PER_CYCLE`, `LOOKAHEAD_CANDLES`, `OPTUNA_TRIALS`, and `selected_features`.\n\n"
                "Respond ONLY with a valid JSON object containing the keys: `MAX_DD_PER_CYCLE`, `LOOKAHEAD_CANDLES`, `OPTUNA_TRIALS`, and `selected_features`. "
                "CRITICAL: `MAX_DD_PER_CYCLE` must be a float between 0.05 and 0.5.\n\n"
                f"SUMMARIZED HISTORICAL CYCLE RESULTS:\n{json.dumps(self._sanitize_dict(recent_history), indent=2)}\n\n"
                f"AVAILABLE FEATURES FOR THIS STRATEGY:\n{available_features}"
            )

        response_text = self._call_gemini(prompt)
        logger.info(f"    - AI Strategist Raw Response: {response_text}")
        suggestions = self._extract_json_from_response(response_text)
        logger.info(f"    - Parsed Suggestions: {suggestions}")
        return suggestions

    def create_hybrid_strategy(self, historical_runs: List[Dict], current_playbook: Dict) -> Optional[Dict]:
        """Analyzes history and prompts the AI to synthesize a new hybrid strategy."""
        if not self.api_key_valid: return None
        logger.info("--- HYBRID SYNTHESIS: Analyzing historical champions...")

        champions = {}
        for run in historical_runs:
            name = run.get("strategy_name")
            if not name: continue
            calmar = run.get("final_metrics", {}).get("calmar_ratio", 0)
            if calmar > champions.get(name, {}).get("calmar", -1):
                champions[name] = {"calmar": calmar, "run_summary": run}

        if len(champions) < 2:
            logger.info("--- HYBRID SYNTHESIS: Need at least two different, successful strategies in history to create a hybrid. Skipping.")
            return None

        sorted_champs = sorted(champions.values(), key=lambda x: x["calmar"], reverse=True)
        champ1_summary = sorted_champs[0]["run_summary"]
        champ2_summary = sorted_champs[1]["run_summary"]

        def format_summary(summary: Dict):
            return {"strategy_name": summary.get("strategy_name"),"calmar_ratio": summary.get("final_metrics", {}).get("calmar_ratio"),"profit_factor": summary.get("final_metrics", {}).get("profit_factor"),"win_rate": summary.get("final_metrics", {}).get("win_rate"),"top_5_features": summary.get("top_5_features")}

        prompt = (
            "You are a master quantitative strategist. Your task is to synthesize a new HYBRID trading strategy by combining the best elements of two successful, but different, historical strategies.\n\n"
            "Analyze the provided performance data for the following two champion archetypes:\n"
            f"1.  **Strategy A:**\n{json.dumps(format_summary(champ1_summary), indent=2)}\n\n"
            f"2.  **Strategy B:**\n{json.dumps(format_summary(champ2_summary), indent=2)}\n\n"
            "Based on this analysis, define a new hybrid strategy. You must:\n"
            "1.  **Name:** Create a unique name for the new strategy (e.g., 'Hybrid_TrendMomentum_V1'). It CANNOT be one of the existing strategy names.\n"
            "2.  **Description:** Write a brief (1-2 sentence) description of its goal.\n"
            "3.  **Features:** Create a new feature list which MUST be an array of strings.\n"
            "4.  **Parameter Ranges:** Suggest new `lookahead_range` and `dd_range`. CRITICAL: These MUST be two-element arrays of numbers (e.g., `[100, 200]`).\n\n"
            f"Existing strategy names to avoid are: {list(current_playbook.keys())}\n\n"
            "Respond ONLY with a valid JSON object for the new strategy, where the key is the new strategy name."
        )
        logger.info("--- HYBRID SYNTHESIS: Prompting AI to create new strategy...")
        response_text = self._call_gemini(prompt)
        logger.info(f"--- HYBRID SYNTHESIS (Raw AI Response): {response_text}")
        new_hybrid = self._extract_json_from_response(response_text)

        if new_hybrid and isinstance(new_hybrid, dict) and len(new_hybrid) == 1:
            hybrid_name = list(new_hybrid.keys())[0]
            hybrid_body = new_hybrid[hybrid_name]
            
            if hybrid_name in current_playbook:
                logger.warning(f"--- HYBRID SYNTHESIS: AI created a hybrid with a name that already exists ('{hybrid_name}'). Discarding.")
                return None
            if not isinstance(hybrid_body.get('features'), list) or not all(isinstance(f, str) for f in hybrid_body['features']):
                logger.error(f"--- HYBRID SYNTHESIS: AI created a hybrid with an invalid 'features' list. Discarding. Content: {hybrid_body.get('features')}")
                return None
            for key in ['lookahead_range', 'dd_range']:
                val = hybrid_body.get(key)
                if not isinstance(val, list) or len(val) != 2 or not all(isinstance(n, (int, float)) for n in val):
                    logger.error(f"--- HYBRID SYNTHESIS: AI created a hybrid with an invalid '{key}'. It must be a two-element list of numbers. Discarding. Content: {val}")
                    return None

            logger.info(f"--- HYBRID SYNTHESIS: Successfully synthesized and validated new strategy: '{hybrid_name}'")
            return new_hybrid
        else:
            logger.error(f"--- HYBRID SYNTHESIS: Failed to parse a valid hybrid strategy from AI response.")
            return None
            
    def generate_nickname(self, used_names: List[str]) -> str:
        """Prompts the AI to generate a new, unique, one-word codename."""
        if not self.api_key_valid:
            return f"Run_{int(time.time())}"
            
        theme = random.choice(["Astronomical Objects", "Mythological Figures", "Gemstones", "Constellations", "Legendary Swords"])
        prompt = (
            "You are a creative writer. Your task is to generate a single, unique, cool-sounding, one-word codename for a trading strategy program.\n"
            f"The theme for the codename is: **{theme}**.\n"
            "The codename must not be in the following list of already used names:\n"
            f"{used_names}\n\n"
            "Respond ONLY with the single codename."
        )
        
        for _ in range(3):
            response = self._call_gemini(prompt).strip().capitalize()
            response = re.sub(r'[`"*]', '', response)
            if response and response not in used_names:
                logger.info(f"Generated new unique nickname: {response}")
                return response
        
        logger.warning("Failed to generate a unique AI nickname after 3 attempts. Using fallback.")
        return f"Run_{int(time.time())}"

# =============================================================================
# 3. CONFIGURATION & VALIDATION
# =============================================================================
class ConfigModel(BaseModel):
    BASE_PATH: DirectoryPath; REPORT_LABEL: str; INITIAL_CAPITAL: confloat(gt=0)
    CONFIDENCE_TIERS: Dict[str, Dict[str, Any]]; BASE_RISK_PER_TRADE_PCT: confloat(gt=0, lt=1)
    SPREAD_PCTG_OF_ATR: confloat(ge=0); SLIPPAGE_PCTG_OF_ATR: confloat(ge=0); OPTUNA_TRIALS: conint(gt=0)
    TRAINING_WINDOW: str; RETRAINING_FREQUENCY: str; FORWARD_TEST_GAP: str; LOOKAHEAD_CANDLES: conint(gt=0)
    MODEL_SAVE_PATH: str = ""; PLOT_SAVE_PATH: str = ""; REPORT_SAVE_PATH: str = ""; SHAP_PLOT_PATH: str = ""
    LOG_FILE_PATH: str = ""; CHAMPION_FILE_PATH: str = ""; HISTORY_FILE_PATH: str = ""; PLAYBOOK_FILE_PATH: str = ""; DIRECTIVES_FILE_PATH: str = ""; NICKNAME_LEDGER_PATH: str = ""
    TREND_FILTER_THRESHOLD: confloat(gt=0) = 25.0
    BOLLINGER_PERIOD: conint(gt=0) = 20; STOCHASTIC_PERIOD: conint(gt=0) = 14; CALCULATE_SHAP_VALUES: bool = True
    MAX_DD_PER_CYCLE: confloat(ge=0.05, lt=1.0) = 0.25
    RISK_CAP_PER_TRADE_USD: confloat(gt=0)
    
    # --- New GNN Parameters ---
    GNN_EMBEDDING_DIM: conint(gt=0) = 8
    GNN_EPOCHS: conint(gt=0) = 50

    selected_features: List[str]
    run_timestamp: str
    strategy_name: str
    nickname: str = ""

    def __init__(self, **data: Any):
        super().__init__(**data)
        results_dir = os.path.join(self.BASE_PATH, "Results")
        
        version_match = re.search(r'V(\d+)', self.REPORT_LABEL)
        version_str = f"_V{version_match.group(1)}" if version_match else ""
        
        if self.nickname and version_str:
            folder_name = f"{self.nickname}{version_str}"
        else:
            folder_name = self.REPORT_LABEL
        
        run_id = f"{folder_name}_{self.strategy_name}_{self.run_timestamp}"
        result_folder_path=os.path.join(results_dir, folder_name);os.makedirs(result_folder_path,exist_ok=True)

        self.MODEL_SAVE_PATH=os.path.join(result_folder_path,f"{run_id}_model.json")
        self.PLOT_SAVE_PATH=os.path.join(result_folder_path,f"{run_id}_equity_curve.png")
        self.REPORT_SAVE_PATH=os.path.join(result_folder_path,f"{run_id}_report.txt")
        self.SHAP_PLOT_PATH=os.path.join(result_folder_path,f"{run_id}_shap_summary.png")
        self.LOG_FILE_PATH=os.path.join(result_folder_path,f"{run_id}_run.log")
        self.CHAMPION_FILE_PATH=os.path.join(results_dir,"champion.json")
        self.HISTORY_FILE_PATH=os.path.join(results_dir,"historical_runs.jsonl")
        self.PLAYBOOK_FILE_PATH=os.path.join(results_dir,"strategy_playbook.json")
        self.DIRECTIVES_FILE_PATH=os.path.join(results_dir,"framework_directives.json")
        self.NICKNAME_LEDGER_PATH=os.path.join(results_dir,"nickname_ledger.json")

# =============================================================================
# 4. DATA LOADER & 5. FEATURE ENGINEERING
# =============================================================================
class DataLoader:
    def __init__(self, config: ConfigModel): self.config = config
    def load_and_parse_data(self, filenames: List[str]) -> Tuple[Optional[Dict[str, pd.DataFrame]], List[str]]:
        logger.info("-> Stage 1: Loading and Preparing Multi-Timeframe Data...")
        data_by_tf = defaultdict(list)
        for filename in filenames:
            file_path = os.path.join(self.config.BASE_PATH, filename)
            if not os.path.exists(file_path): 
                logger.warning(f"  - File not found, skipping: {file_path}")
                continue
            try:
                parts = filename.split('_')
                if len(parts) < 2: continue
                symbol, tf = parts[0], parts[1]

                df=pd.read_csv(file_path,delimiter='\t' if '\t' in open(file_path).readline() else ',');df.columns=[c.upper().replace('<','').replace('>','') for c in df.columns]
                date_col=next((c for c in df.columns if 'DATE' in c),None);time_col=next((c for c in df.columns if 'TIME' in c),None)
                if date_col and time_col: df['Timestamp'] = pd.to_datetime(df[date_col] + ' ' + df[time_col], errors='coerce')
                elif date_col: df['Timestamp'] = pd.to_datetime(df[date_col], errors='coerce')
                else: raise ValueError("No date/time columns found.")
                df.dropna(subset=['Timestamp'],inplace=True); df.set_index('Timestamp', inplace=True)
                col_map={c:c.capitalize() for c in df.columns if c.lower() in ['open','high','low','close','tickvol','volume','spread']}
                df.rename(columns=col_map,inplace=True);vol_col='Volume' if 'Volume' in df.columns else 'Tickvol'
                df.rename(columns={vol_col:'RealVolume'},inplace=True,errors='ignore')
                if 'RealVolume' not in df.columns:df['RealVolume']=0
                df['Symbol']=symbol
                data_by_tf[tf].append(df)
            except Exception as e: 
                logger.error(f"  - Failed to load {filename}: {e}", exc_info=True)

        processed_dfs:Dict[str,pd.DataFrame]={}
        for tf,dfs in data_by_tf.items():
            if dfs:
                combined=pd.concat(dfs);all_symbols_df=[df[~df.index.duplicated(keep='first')].sort_index() for _,df in combined.groupby('Symbol')]
                final_combined=pd.concat(all_symbols_df).sort_index();final_combined['RealVolume']=pd.to_numeric(final_combined['RealVolume'],errors='coerce').fillna(0)
                processed_dfs[tf]=final_combined;logger.info(f"  - Processed {tf}: {len(final_combined):,} rows for {len(final_combined['Symbol'].unique())} symbols.")
        
        detected_timeframes = list(processed_dfs.keys())
        if not processed_dfs:
            logger.critical("  - Data loading failed for all files.");
            return None, []
            
        logger.info(f"[SUCCESS] Data loading complete. Detected timeframes: {detected_timeframes}")
        return processed_dfs, detected_timeframes

class FeatureEngineer:
    TIMEFRAME_MAP = {'M1': 1,'M5': 5,'M15': 15,'M30': 30,'H1': 60,'H4': 240,'D1': 1440, 'DAILY': 1440}

    def __init__(self, config: ConfigModel, timeframe_roles: Dict[str, str]):
        self.config = config
        self.roles = timeframe_roles

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
        logger.info(f"    - Calculating HTF features for {p}...");results=[]
        for symbol,group in df.groupby('Symbol'):
            g=group.copy();sma=g['Close'].rolling(s,min_periods=s).mean();atr=(g['High']-g['Low']).rolling(a,min_periods=a).mean();trend=np.sign(g['Close']-sma)
            temp_df=pd.DataFrame(index=g.index);temp_df[f'{p}_ctx_SMA']=sma;temp_df[f'{p}_ctx_ATR']=atr;temp_df[f'{p}_ctx_Trend']=trend
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
        g_out = self._calculate_candlestick_patterns(g_out) # Depends on ATR
        g_out['hour'] = g_out.index.hour;g_out['day_of_week'] = g_out.index.dayofweek
        g_out['market_regime']=np.where(g_out['ADX']>self.config.TREND_FILTER_THRESHOLD,1,0)
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
            df_final[f'adx_x_{medium_tf}_trend'] = df_final['ADX'] * df_final.get(f'{medium_tf}_ctx_Trend', 0)
        if high_tf:
            df_final[f'atr_x_{high_tf}_trend'] = df_final['ATR'] * df_final.get(f'{high_tf}_ctx_Trend', 0)

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

            tp_short,sl_short=prices[i]-tp_dist,prices[i]+tp_dist
            time_to_tp_short=np.where(future_lows<=tp_short)[0]; time_to_sl_short=np.where(future_highs>=sl_short)[0]
            first_tp_short=time_to_tp_short[0] if len(time_to_tp_short)>0 else np.inf
            first_sl_short=time_to_sl_short[0] if len(time_to_sl_short)>0 else np.inf

            if first_tp_long < first_sl_long: outcomes[i]=1
            if first_tp_short < first_sl_short: outcomes[i]=-1

        group['target']=outcomes;return group

# --- GNN Model Definition ---
class GNNModel(torch.nn.Module if GNN_AVAILABLE else object):
    def __init__(self, in_channels, hidden_channels, out_channels):
        super(GNNModel, self).__init__()
        self.conv1 = GCNConv(in_channels, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, out_channels)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, training=self.training)
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
        # --- GNN Specific Attributes ---
        self.is_gnn_model = False
        self.gnn_model: Optional[GNNModel] = None
        self.gnn_scaler = MinMaxScaler()
        self.asset_map: Dict[str, int] = {}


    def train(self, df_train: pd.DataFrame, feature_list: List[str], strategy_details: Dict) -> Optional[Tuple[Pipeline, float]]:
        logger.info(f"  - Starting model training using strategy: '{strategy_details.get('description', 'N/A')}'")
        self.is_gnn_model = strategy_details.get("requires_gnn", False)

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
            feature_list = list(X.columns) # The features are now the embeddings
            logger.info(f"  - Feature set replaced by {len(feature_list)} GNN embeddings.")
        else:
            X = df_train[feature_list].copy().fillna(0)
        
        y_map={-1:0,0:1,1:2};y=df_train['target'].map(y_map).astype(int)

        if len(y.unique()) < 3:
            logger.warning(f"  - Skipping cycle: Not enough classes ({len(y.unique())}) for 3-class model.")
            return None

        self.class_weights=dict(zip(np.unique(y),compute_class_weight(class_weight='balanced',classes=np.unique(y),y=y)))

        X_train_val, X_stability, y_train_val, y_stability = train_test_split(X, y, test_size=0.1, shuffle=False)
        X_train, X_val, y_train, y_val = train_test_split(X_train_val, y_train_val, test_size=0.2, shuffle=False)

        self.study=self._optimize_hyperparameters(X_train,y_train,X_val,y_val, X_stability, y_stability)
        if not self.study or not self.study.best_trials:logger.error("  - Training aborted: Hyperparameter optimization failed.");return None
        logger.info(f"    - Optimization complete. Best Objective Score: {self.study.best_value:.4f}");logger.info(f"    - Best params: {self.study.best_params}")

        self.best_threshold = self._find_best_threshold(self.study.best_params, X_train, y_train, X_val, y_val)
        final_pipeline=self._train_final_model(self.study.best_params,X_train_val,y_train_val, feature_list)
        if final_pipeline is None:logger.error("  - Training aborted: Final model training failed.");return None

        logger.info("  - [SUCCESS] Model training complete.");return final_pipeline, self.best_threshold

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
        
        if not edge_list: # Create a fully connected graph if no strong correlations exist
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

        self.gnn_model = GNNModel(in_channels=graph_data.num_node_features, 
                                hidden_channels=self.config.GNN_EMBEDDING_DIM * 2, 
                                out_channels=self.config.GNN_EMBEDDING_DIM)
        optimizer = Adam(self.gnn_model.parameters(), lr=0.01, weight_decay=5e-4)
        self.gnn_model.train()
        
        for epoch in range(self.config.GNN_EPOCHS):
            optimizer.zero_grad()
            out = self.gnn_model(graph_data)
            loss = out.mean() # Simple unsupervised loss
            loss.backward()
            optimizer.step()
        
        self.gnn_model.eval()
        with torch.no_grad():
            embeddings = self.gnn_model(graph_data).numpy()

        embedding_df = pd.DataFrame(embeddings, index=self.asset_map.keys())
        full_embeddings = df['Symbol'].map(lambda s: embedding_df.loc[s]).apply(pd.Series)
        full_embeddings.columns = [f"gnn_{i}" for i in range(self.config.GNN_EMBEDDING_DIM)]
        full_embeddings.index = df.index
        return full_embeddings

    def _get_gnn_embeddings_for_test(self, df_test: pd.DataFrame) -> pd.DataFrame:
        if not self.is_gnn_model: return pd.DataFrame()
        
        feature_cols = [f for f in self.GNN_BASE_FEATURES if f in df_test.columns]
        test_node_features = df_test.groupby('Symbol')[feature_cols].mean()

        # Ensure all assets from training are present in test node features
        for asset in self.asset_map.keys():
            if asset not in test_node_features.index:
                test_node_features.loc[asset] = 0
        test_node_features = test_node_features.reindex(list(self.asset_map.keys()))

        test_node_features_scaled = pd.DataFrame(self.gnn_scaler.transform(test_node_features), index=test_node_features.index)
        x = torch.tensor(test_node_features_scaled.values, dtype=torch.float)
        
        # Re-use the graph structure from training for consistency
        graph_data, _ = self._create_graph_data(df_test) 
        graph_data.x = x

        self.gnn_model.eval()
        with torch.no_grad():
            embeddings = self.gnn_model(graph_data).numpy()

        embedding_df = pd.DataFrame(embeddings, index=self.asset_map.keys())
        full_embeddings = df_test['Symbol'].map(lambda s: embedding_df.loc[s]).apply(pd.Series)
        full_embeddings.columns = [f"gnn_{i}" for i in range(self.config.GNN_EMBEDDING_DIM)]
        full_embeddings.index = df_test.index
        return full_embeddings

    def _find_best_threshold(self, best_params, X_train, y_train, X_val, y_val) -> float:
        logger.info("    - Tuning classification threshold for F1 score...")
        temp_params = {'objective':'multi:softprob','num_class':3,'booster':'gbtree','tree_method':'hist','use_label_encoder':False,**best_params}
        temp_params.pop('early_stopping_rounds', None)
        temp_pipeline = Pipeline([('scaler', RobustScaler()), ('model', xgb.XGBClassifier(**temp_params))])
        fit_params={'model__sample_weight':y_train.map(self.class_weights)}
        temp_pipeline.fit(X_train, y_train, **fit_params)
        probs = temp_pipeline.predict_proba(X_val)
        best_f1, best_thresh = -1, 0.5
        for threshold in np.arange(0.3, 0.7, 0.01):
            max_probs = np.max(probs, axis=1);preds = np.argmax(probs, axis=1)
            preds = np.where(max_probs > threshold, preds, 1) # Default to 'hold'
            f1 = f1_score(y_val, preds, average='macro', zero_division=0)
            if f1 > best_f1: best_f1, best_thresh = f1, threshold
        logger.info(f"    - Best threshold found: {best_thresh:.2f} (F1: {best_f1:.4f})"); return best_thresh

    def _optimize_hyperparameters(self,X_train,y_train,X_val,y_val, X_stability, y_stability)->Optional[optuna.study.Study]:
        logger.info(f"    - Starting hyperparameter optimization with risk-adjusted objective ({self.config.OPTUNA_TRIALS} trials)...")

        def custom_objective(trial:optuna.Trial):
            param={'objective':'multi:softprob','num_class':3,'eval_metric':'mlogloss','booster':'gbtree','tree_method':'hist','use_label_encoder':False,'seed':42,
                    'n_estimators':trial.suggest_int('n_estimators',200,1000,step=50),
                    'max_depth':trial.suggest_int('max_depth',3,8),
                    'learning_rate':trial.suggest_float('learning_rate',0.01,0.2,log=True),
                    'subsample':trial.suggest_float('subsample',0.6,1.0),
                    'colsample_bytree':trial.suggest_float('colsample_bytree',0.6,1.0),
                    'gamma':trial.suggest_float('gamma',0,5),
                    'reg_lambda':trial.suggest_float('reg_lambda',1e-8,5.0,log=True),
                    'alpha':trial.suggest_float('alpha',1e-8,5.0,log=True),
                    'early_stopping_rounds':50
            }
            try:
                scaler=RobustScaler();X_train_scaled=scaler.fit_transform(X_train);X_val_scaled=scaler.transform(X_val)
                model=xgb.XGBClassifier(**param); fit_params={'sample_weight':y_train.map(self.class_weights)}
                model.fit(X_train_scaled,y_train,eval_set=[(X_val_scaled,y_val)],verbose=False,**fit_params)

                preds=model.predict(X_val_scaled); f1 = f1_score(y_val,preds,average='macro', zero_division=0)
                pnl_map = {0: -1, 1: 0, 2: 1}; pnl = pd.Series(preds).map(pnl_map)
                downside_returns = pnl[pnl < 0]; downside_std = downside_returns.std()
                sortino = (pnl.mean() / downside_std) if downside_std > 0 else 0
                objective_score = (0.4 * f1) + (0.6 * sortino)

                X_stability_scaled = scaler.transform(X_stability); stability_preds = model.predict(X_stability_scaled)
                stability_pnl = pd.Series(stability_preds).map(pnl_map)
                if stability_pnl.sum() < 0: objective_score -= 0.5

                return objective_score
            except Exception as e:logger.warning(f"Trial {trial.number} failed with error: {e}");return -2.0

        try:study=optuna.create_study(direction='maximize');study.optimize(custom_objective,n_trials=self.config.OPTUNA_TRIALS,timeout=3600);return study
        except Exception as e:logger.error(f"    - Optuna study failed catastrophically: {e}",exc_info=True);return None

    def _train_final_model(self,best_params:Dict,X:pd.DataFrame,y:pd.Series, feature_names: List[str])->Optional[Pipeline]:
        logger.info("    - Training final model...");
        try:
            best_params.pop('early_stopping_rounds', None)
            final_params={'objective':'multi:softprob','num_class':3,'eval_metric':'mlogloss','booster':'gbtree','tree_method':'hist','use_label_encoder':False,'seed':42,**best_params}
            final_pipeline=Pipeline([('scaler',RobustScaler()),('model',xgb.XGBClassifier(**final_params))])
            fit_params={'model__sample_weight':y.map(self.class_weights)}
            final_pipeline.fit(X,y,**fit_params)
            if self.config.CALCULATE_SHAP_VALUES:self._generate_shap_summary(final_pipeline.named_steps['model'],final_pipeline.named_steps['scaler'].transform(X), feature_names)
            return final_pipeline
        except Exception as e:logger.error(f"    - Error during final model training: {e}",exc_info=True);return None

    def _generate_shap_summary(self, model: xgb.XGBClassifier, X_scaled: np.ndarray, feature_names: List[str]):
        logger.info("    - Generating SHAP feature importance summary...")
        try:
            if len(X_scaled) > 2000:
                logger.info(f"    - Subsampling data for SHAP from {len(X_scaled)} to 2000 rows for performance.")
                np.random.seed(42); sample_indices = np.random.choice(X_scaled.shape[0], 2000, replace=False)
                X_sample = X_scaled[sample_indices]
            else: X_sample = X_scaled

            explainer = shap.TreeExplainer(model)
            shap_explanation = explainer(X_sample)

            mean_abs_shap_per_class = shap_explanation.abs.mean(0).values
            overall_importance = mean_abs_shap_per_class.mean(axis=1) if mean_abs_shap_per_class.ndim == 2 else mean_abs_shap_per_class

            summary = pd.DataFrame(overall_importance, index=feature_names, columns=['SHAP_Importance']).sort_values(by='SHAP_Importance', ascending=False)
            self.shap_summary = summary
            logger.info("    - SHAP summary generated successfully.")
        except Exception as e:
            logger.error(f"    - Failed to generate SHAP summary: {e}", exc_info=True); self.shap_summary = None

class Backtester:
    def __init__(self,config:ConfigModel):self.config=config
    def run_backtest_chunk(self,df_chunk_in:pd.DataFrame, confidence_threshold:float, initial_equity:float) -> Tuple[pd.DataFrame,pd.Series,bool,Optional[Dict]]:
        if df_chunk_in.empty:return pd.DataFrame(),pd.Series([initial_equity]), False, None
        df_chunk=df_chunk_in.copy()
        trades,equity,equity_curve,open_positions=[],initial_equity,[initial_equity],{}
        chunk_peak_equity = initial_equity
        circuit_breaker_tripped = False
        breaker_context = None

        candles=df_chunk.reset_index().to_dict('records')
        for candle in candles:
            if not circuit_breaker_tripped:
                if equity > chunk_peak_equity: chunk_peak_equity = equity
                if equity > 0 and chunk_peak_equity > 0 and (chunk_peak_equity - equity) / chunk_peak_equity > self.config.MAX_DD_PER_CYCLE:
                    logger.warning(f"  - CIRCUIT BREAKER TRIPPED! Drawdown exceeded {self.config.MAX_DD_PER_CYCLE:.0%} for this cycle. Closing all positions.")
                    circuit_breaker_tripped = True
                    trade_df = pd.DataFrame(trades)
                    if not trade_df.empty:
                        breaker_context = {
                            "num_trades_before_trip": len(trade_df),
                            "pnl_before_trip": round(trade_df['PNL'].sum(), 2),
                            "last_5_trades_pnl": [round(p, 2) for p in trade_df['PNL'].tail(5).tolist()]
                        }
                    open_positions = {}

            symbol=candle['Symbol']
            if symbol in open_positions:
                pos=open_positions[symbol];pnl,exit_price=0,None
                if pos['direction']==1:
                    if candle['Low']<=pos['sl']:pnl,exit_price=-pos['risk_amt'],pos['sl']
                    elif candle['High']>=pos['tp']:pnl,exit_price=pos['risk_amt']*pos['rr'],pos['tp']
                elif pos['direction']==-1:
                    if candle['High']>=pos['sl']:pnl,exit_price=-pos['risk_amt'],pos['sl']
                    elif candle['Low']<=pos['tp']:pnl,exit_price=pos['risk_amt']*pos['rr'],pos['tp']
                if exit_price:
                    equity+=pnl; equity_curve.append(equity)
                    if equity<=0:logger.critical("  - ACCOUNT BLOWN!");break
                    trades.append({'ExecTime':candle['Timestamp'],'Symbol':symbol,'PNL':pnl,'Equity':equity,'Confidence':pos['confidence'],'Direction':pos['direction']})
                    del open_positions[symbol]

            if circuit_breaker_tripped: continue

            if symbol not in open_positions:
                probs=np.array([candle['prob_short'],candle['prob_hold'],candle['prob_long']])
                confidence=np.max(probs)
                if confidence>=confidence_threshold:
                    pred_class=np.argmax(probs); direction=1 if pred_class==2 else-1 if pred_class==0 else 0
                    if direction!=0:
                        atr=candle.get('ATR',0);
                        if pd.isna(atr) or atr<=1e-9:continue
                        tier_name='standard'
                        if confidence>=self.config.CONFIDENCE_TIERS['ultra_high']['min']:tier_name='ultra_high'
                        elif confidence>=self.config.CONFIDENCE_TIERS['high']['min']:tier_name='high'
                        
                        tier=self.config.CONFIDENCE_TIERS[tier_name]
                        base_risk_amt = equity * self.config.BASE_RISK_PER_TRADE_PCT * tier['risk_mult']
                        risk_amt = min(base_risk_amt, self.config.RISK_CAP_PER_TRADE_USD)
                        if base_risk_amt > risk_amt:
                            logger.debug(f"Risk capped: Equity-based risk was ${base_risk_amt:,.2f}, capped at ${risk_amt:,.2f}.")

                        sl_dist=(atr*1.5)+(atr*self.config.SPREAD_PCTG_OF_ATR)+(atr*self.config.SLIPPAGE_PCTG_OF_ATR);tp_dist=(atr*1.5*tier['rr'])
                        if tp_dist<=0:continue
                        entry_price=candle['Close'];sl_price,tp_price=entry_price-sl_dist*direction,entry_price+tp_dist*direction
                        open_positions[symbol]={'direction':direction,'sl':sl_price,'tp':tp_price,'risk_amt':risk_amt,'rr':tier['rr'],'confidence':confidence}
        return pd.DataFrame(trades),pd.Series(equity_curve), circuit_breaker_tripped, breaker_context

# --- Structural Improvement: Helper functions moved closer to their usage ---
def _ljust(text, width): return str(text).ljust(width)
def _rjust(text, width): return str(text).rjust(width)
def _center(text, width): return str(text).center(width)

class PerformanceAnalyzer:
    def __init__(self,config:ConfigModel):self.config=config
    def generate_full_report(self,trades_df:Optional[pd.DataFrame],equity_curve:Optional[pd.Series],cycle_metrics:List[Dict],aggregated_shap:Optional[pd.DataFrame]=None, framework_memory:Optional[Dict]=None) -> Dict[str, Any]:
        logger.info("-> Stage 4: Generating Final Performance Report...")
        if equity_curve is not None and len(equity_curve) > 1: self.plot_equity_curve(equity_curve)
        if aggregated_shap is not None: self.plot_shap_summary(aggregated_shap)
        metrics = self._calculate_metrics(trades_df, equity_curve) if trades_df is not None and not trades_df.empty else {}
        self.generate_text_report(metrics, cycle_metrics, aggregated_shap, framework_memory)
        logger.info("[SUCCESS] Final report generated and saved.")
        return metrics

    def plot_equity_curve(self,equity_curve:pd.Series):
        plt.style.use('seaborn-v0_8-darkgrid');plt.figure(figsize=(16,8));plt.plot(equity_curve.values,color='dodgerblue',linewidth=2)
        plt.title(f"{self.config.nickname or self.config.REPORT_LABEL} - Walk-Forward Equity Curve",fontsize=16,weight='bold')
        plt.xlabel("Trade Number",fontsize=12);plt.ylabel("Equity ($)",fontsize=12);plt.grid(True,which='both',linestyle=':');plt.savefig(self.config.PLOT_SAVE_PATH);plt.close()
        logger.info(f"  - Equity curve plot saved to: {self.config.PLOT_SAVE_PATH}")

    def plot_shap_summary(self,shap_summary:pd.DataFrame):
        plt.style.use('seaborn-v0_8-darkgrid');plt.figure(figsize=(12,10))
        shap_summary.head(20).sort_values(by='SHAP_Importance').plot(kind='barh',legend=False,color='mediumseagreen')
        title_str = f"{self.config.nickname or self.config.REPORT_LABEL} ({self.config.strategy_name}) - Aggregated Feature Importance"
        plt.title(title_str,fontsize=16,weight='bold')
        plt.xlabel("Mean Absolute SHAP Value",fontsize=12);plt.ylabel("Feature",fontsize=12);plt.tight_layout();plt.savefig(self.config.SHAP_PLOT_PATH);plt.close()
        logger.info(f"  - SHAP summary plot saved to: {self.config.SHAP_PLOT_PATH}")

    def _calculate_metrics(self,trades_df:pd.DataFrame,equity_curve:pd.Series)->Dict[str,Any]:
        m={};m['initial_capital']=self.config.INITIAL_CAPITAL;m['ending_capital']=equity_curve.iloc[-1]
        m['total_net_profit']=m['ending_capital']-m['initial_capital'];m['net_profit_pct']=m['total_net_profit']/m['initial_capital'] if m['initial_capital']>0 else 0
        returns=trades_df['PNL']/m['initial_capital'];wins=trades_df[trades_df['PNL']>0];losses=trades_df[trades_df['PNL']<0]
        m['gross_profit']=wins['PNL'].sum();m['gross_loss']=abs(losses['PNL'].sum());m['profit_factor']=m['gross_profit']/m['gross_loss'] if m['gross_loss']>0 else np.inf
        m['total_trades']=len(trades_df);m['winning_trades']=len(wins);m['losing_trades']=len(losses)
        m['win_rate']=m['winning_trades']/m['total_trades'] if m['total_trades']>0 else 0
        m['avg_win_amount']=wins['PNL'].mean() if len(wins)>0 else 0;m['avg_loss_amount']=abs(losses['PNL'].mean()) if len(losses)>0 else 0
        m['payoff_ratio']=m['avg_win_amount']/m['avg_loss_amount'] if m['avg_loss_amount']>0 else np.inf
        m['expected_payoff']=(m['win_rate']*m['avg_win_amount'])-((1-m['win_rate'])*m['avg_loss_amount']) if m['total_trades']>0 else 0
        running_max=equity_curve.cummax();drawdown_abs=running_max-equity_curve;m['max_drawdown_abs']=drawdown_abs.max() if not drawdown_abs.empty else 0
        m['max_drawdown_pct']=(drawdown_abs/running_max).replace([np.inf,-np.inf],0).max()*100
        exec_times=pd.to_datetime(trades_df['ExecTime']).dt.tz_localize(None);years=(exec_times.max()-exec_times.min()).days/365.25 if not trades_df.empty else 1
        m['cagr']=((m['ending_capital']/m['initial_capital'])**(1/years)-1) if years>0 and m['initial_capital']>0 else 0
        pnl_std=returns.std();m['sharpe_ratio']=(returns.mean()/pnl_std)*np.sqrt(252*24*4) if pnl_std>0 else 0
        downside_returns=returns[returns<0];downside_std=downside_returns.std();m['sortino_ratio']=(returns.mean()/downside_std)*np.sqrt(252*24*4) if downside_std>0 else np.inf
        m['calmar_ratio']=m['cagr']/(m['max_drawdown_pct']/100) if m['max_drawdown_pct']>0 else np.inf;m['mar_ratio']=m['calmar_ratio']
        m['recovery_factor']=m['total_net_profit']/m['max_drawdown_abs'] if m['max_drawdown_abs']>0 else np.inf
        win_streak=0;loss_streak=0;longest_win_streak=0;longest_loss_streak=0
        for pnl in trades_df['PNL']:
            if pnl>0:win_streak+=1;loss_streak=0
            else:loss_streak+=1;win_streak=0
            if win_streak>longest_win_streak:longest_win_streak=win_streak
            if loss_streak>longest_loss_streak:longest_loss_streak=loss_streak
        m['longest_win_streak']=longest_win_streak;m['longest_loss_streak']=longest_loss_streak;return m

    def _get_comparison_block(self, metrics: Dict, memory: Dict, ledger: Dict) -> str:
        champion = memory.get('champion_config') if memory else None
        historical_runs = memory.get('historical_runs', [])
        previous_run = historical_runs[-1] if historical_runs else None

        def get_data(source: Dict, key: str, is_percent: bool = False):
            if not source: return "N/A"
            val = source.get("final_metrics", {}).get(key)
            if val is None: return "N/A"
            return f"{val:.2f}%" if is_percent else f"{val:.2f}"

        def get_info(source: Dict, key: str, is_nickname: bool = False):
            if not source: return "N/A"
            val = source.get(key, 'N/A')
            if is_nickname:
                return ledger.get(val, val)
            return val
        
        versions = {self.config.REPORT_LABEL}
        if previous_run: versions.add(get_info(previous_run, 'script_version'))
        if champion: versions.add(get_info(champion, 'script_version'))
        
        mapping_str = "Run & Version Mapping:\n"
        for v in sorted(list(versions)):
            nickname = ledger.get(v, "N/A")
            mapping_str += f"- {nickname}: {v}\n"

        c_nick = get_info({"script_version": self.config.REPORT_LABEL}, 'script_version', is_nickname=True)
        c_strat = self.config.strategy_name
        c_mar = get_data(metrics, 'mar_ratio')
        c_mdd = get_data(metrics, 'max_drawdown_pct', is_percent=True)
        c_pf = get_data(metrics, 'profit_factor')

        p_nick = get_info(previous_run, 'script_version', is_nickname=True)
        p_strat = get_info(previous_run, 'strategy_name')
        p_mar = get_data(previous_run, 'mar_ratio')
        p_mdd = get_data(previous_run, 'max_drawdown_pct', is_percent=True)
        p_pf = get_data(previous_run, 'profit_factor')
        
        champ_nick = get_info(champion, 'script_version', is_nickname=True)
        champ_strat = get_info(champion, 'strategy_name')
        champ_mar = get_data(champion, 'mar_ratio')
        champ_mdd = get_data(champion, 'max_drawdown_pct', is_percent=True)
        champ_pf = get_data(champion, 'profit_factor')
        
        block = f"""{mapping_str}
--------------------------------------------------------------------------------
I. PERFORMANCE vs. HISTORY
--------------------------------------------------------------------------------
                         {_center('Current Run', 20)}|{_center('Previous Run', 32)}|{_center('All-Time Champion', 32)}
-------------------- -------------------- -------------------------------- --------------------------------
Run Label            {_ljust(c_nick, 20)}|{_ljust(p_nick, 32)}|{_ljust(champ_nick, 32)}
Strategy             {_ljust(c_strat, 20)}|{_ljust(p_strat, 32)}|{_ljust(champ_strat, 32)}
MAR Ratio:           {_rjust(c_mar, 20)}|{_rjust(p_mar, 32)}|{_rjust(champ_mar, 32)}
Max Drawdown:        {_rjust(c_mdd, 20)}|{_rjust(p_mdd, 32)}|{_rjust(champ_mdd, 32)}
Profit Factor:       {_rjust(c_pf, 20)}|{_rjust(p_pf, 32)}|{_rjust(champ_pf, 32)}
--------------------------------------------------------------------------------
"""
        return block

    def generate_text_report(self,m:Dict[str,Any],cycle_metrics:List[Dict],aggregated_shap:Optional[pd.DataFrame]=None, framework_memory:Optional[Dict]=None):
        ledger = {}
        if self.config.NICKNAME_LEDGER_PATH and os.path.exists(self.config.NICKNAME_LEDGER_PATH):
            with open(self.config.NICKNAME_LEDGER_PATH, 'r') as f:
                ledger = json.load(f)

        comparison_block = self._get_comparison_block({"final_metrics": m}, framework_memory, ledger) if framework_memory else ""
        cycle_df=pd.DataFrame(cycle_metrics);cycle_report="Per-Cycle Performance:\n"+cycle_df.to_string(index=False) if not cycle_df.empty else "No trades were executed."
        shap_report="Aggregated Feature Importance (SHAP):\n"+aggregated_shap.head(15).to_string() if aggregated_shap is not None else "SHAP summary was not generated."
        report=f"""
\n================================================================================
                                   ADAPTIVE WALK-FORWARD PERFORMANCE REPORT
================================================================================
Report Nickname: {self.config.nickname or 'N/A'} ({self.config.strategy_name})
Full Version: {self.config.REPORT_LABEL}
Generated: {self.config.run_timestamp}

{comparison_block}
II. EXECUTIVE SUMMARY
--------------------------------------------------------------------------------
Initial Capital:         ${m.get('initial_capital', self.config.INITIAL_CAPITAL):>15,.2f}
Ending Capital:          ${m.get('ending_capital', self.config.INITIAL_CAPITAL):>15,.2f}
Total Net Profit:        ${m.get('total_net_profit', 0):>15,.2f} ({m.get('net_profit_pct', 0):.2%})
Profit Factor:           {m.get('profit_factor', 0):>15.2f}
Win Rate:                {m.get('win_rate', 0):>15.2%}
Expected Payoff:         ${m.get('expected_payoff', 0):>15.2f}
--------------------------------------------------------------------------------
III. CORE PERFORMANCE METRICS
--------------------------------------------------------------------------------
Annual Return (CAGR):    {m.get('cagr', 0):>15.2%}
Sharpe Ratio (annual):   {m.get('sharpe_ratio', 0):>15.2f}
Sortino Ratio (annual):  {m.get('sortino_ratio', 0):>15.2f}
Calmar Ratio / MAR:      {m.get('mar_ratio', 0):>15.2f}
--------------------------------------------------------------------------------
IV. RISK & DRAWDOWN ANALYSIS
--------------------------------------------------------------------------------
Max Drawdown:            {m.get('max_drawdown_pct', 0):>15.2f}% (${m.get('max_drawdown_abs', 0):,.2f})
Recovery Factor:         {m.get('recovery_factor', 0):>15.2f}
Longest Losing Streak:   {m.get('longest_loss_streak', 0):>15} trades
--------------------------------------------------------------------------------
V. TRADE-LEVEL STATISTICS
--------------------------------------------------------------------------------
Total Trades:            {m.get('total_trades', 0):>15}
Winning Trades:          {m.get('winning_trades', 0):>15}
Losing Trades:           {m.get('losing_trades', 0):>15}
Average Win:             ${m.get('avg_win_amount', 0):>15,.2f}
Average Loss:            ${m.get('avg_loss_amount', 0):>15,.2f}
Payoff Ratio:            {m.get('payoff_ratio', 0):>15.2f}
Longest Winning Streak:  {m.get('longest_win_streak', 0):>15} trades
--------------------------------------------------------------------------------
VI. WALK-FORWARD CYCLE BREAKDOWN
--------------------------------------------------------------------------------
{cycle_report}
--------------------------------------------------------------------------------
VII. MODEL FEATURE IMPORTANCE
--------------------------------------------------------------------------------
{shap_report}
================================================================================
"""
        logger.info(report)
        try:
            with open(self.config.REPORT_SAVE_PATH,'w',encoding='utf-8') as f:f.write(report)
            logger.info(f"  - Quantitative report saved to {self.config.REPORT_SAVE_PATH}")
        except IOError as e:logger.error(f"  - Failed to save text report: {e}",exc_info=True)

# =============================================================================
# 9. FRAMEWORK ORCHESTRATION & MEMORY
# =============================================================================
def load_memory(champion_path: str, history_path: str) -> Dict:
    """Loads champion and historical runs from the new robust file format."""
    champion_config = None
    if os.path.exists(champion_path):
        try:
            with open(champion_path, 'r') as f:
                champion_config = json.load(f)
        except (json.JSONDecodeError, IOError) as e:
            logger.error(f"Could not read or parse champion file at {champion_path}: {e}")

    historical_runs = []
    if os.path.exists(history_path):
        with open(history_path, 'r') as f:
            for i, line in enumerate(f):
                if not line.strip(): continue
                try:
                    historical_runs.append(json.loads(line))
                except json.JSONDecodeError:
                    logger.warning(f"Skipping malformed line {i+1} in history file: {history_path}")

    return {"champion_config": champion_config, "historical_runs": historical_runs}

def save_run_to_memory(config: ConfigModel, new_run_summary: Dict, current_memory: Dict) -> Optional[Dict]:
    """Saves the champion to a JSON file and appends the run summary to the JSONL history file."""
    try:
        with open(config.HISTORY_FILE_PATH, 'a') as f:
            f.write(json.dumps(new_run_summary) + '\n')
        logger.info(f"-> Run summary appended to history file: {config.HISTORY_FILE_PATH}")
    except IOError as e:
        logger.error(f"Could not write to history file: {e}")

    current_champion = current_memory.get("champion_config")
    new_calmar = new_run_summary.get("final_metrics", {}).get("calmar_ratio", 0)
    
    new_champion_obj = None
    if current_champion is None or (new_calmar is not None and new_calmar > current_champion.get("final_metrics", {}).get("calmar_ratio", 0)):
        champion_to_save = new_run_summary
        new_champion_obj = new_run_summary
        champion_calmar = current_champion.get("final_metrics", {}).get("calmar_ratio", 0) if current_champion else -1
        logger.info(f"NEW CHAMPION! Current run's Calmar Ratio ({new_calmar:.2f}) beats the previous champion's ({champion_calmar:.2f}).")
    else:
        champion_to_save = current_champion
        new_champion_obj = current_champion
        champ_calmar_val = current_champion.get("final_metrics", {}).get("calmar_ratio", 0)
        logger.info(f"Current run's Calmar Ratio ({new_calmar:.2f}) did not beat the champion's ({champ_calmar_val:.2f}).")

    try:
        with open(config.CHAMPION_FILE_PATH, 'w') as f:
            json.dump(champion_to_save, f, indent=4)
        logger.info(f"-> Champion file updated: {config.CHAMPION_FILE_PATH}")
    except IOError as e:
        logger.error(f"Could not write to champion file: {e}")
        
    return new_champion_obj

def initialize_playbook(base_path: str) -> Dict:
    """Loads the strategy playbook from JSON, creating it from a default if it doesn't exist."""
    results_dir = os.path.join(base_path, "Results")
    os.makedirs(results_dir, exist_ok=True)
    playbook_path = os.path.join(results_dir, "strategy_playbook.json")
    
    TREND_FEATURES = ['ADX', 'H1_ctx_Trend', 'D1_ctx_Trend', 'H1_ctx_SMA', 'D1_ctx_SMA', 'adx_x_H1_trend', 'atr_x_D1_trend']
    REVERSAL_FEATURES = ['RSI', 'stoch_k', 'stoch_d', 'bollinger_bandwidth']
    VOLATILITY_FEATURES = ['ATR', 'bollinger_bandwidth']
    MOMENTUM_FEATURES = ['momentum_10', 'momentum_20', 'RSI']
    RANGE_FEATURES = ['RSI', 'stoch_k', 'ADX', 'bollinger_bandwidth']
    PRICE_ACTION_FEATURES = ['is_doji', 'is_engulfing']
    SEASONALITY_FEATURES = ['month', 'week_of_year', 'day_of_month']
    SESSION_FEATURES = ['hour', 'day_of_week']

    DEFAULT_PLAYBOOK = {
        "TrendFollower": {"description": "Aims to catch long trends using HTF context and trend strength.", "features": list(set(TREND_FEATURES + SESSION_FEATURES)), "lookahead_range": [150, 250], "dd_range": [0.25, 0.40]},
        "MeanReversion": {"description": "Aims for short-term reversals using oscillators.", "features": list(set(REVERSAL_FEATURES + SESSION_FEATURES)),"lookahead_range": [40, 80], "dd_range": [0.15, 0.25]},
        "VolatilityBreakout": {"description": "Trades breakouts during high volatility sessions.", "features": list(set(VOLATILITY_FEATURES + SESSION_FEATURES)), "lookahead_range": [60, 120], "dd_range": [0.20, 0.35]},
        "Momentum": {"description": "Capitalizes on short-term price momentum.", "features": list(set(MOMENTUM_FEATURES + SESSION_FEATURES)), "lookahead_range": [30, 90], "dd_range": [0.18, 0.30]},
        "RangeBound": {"description": "Trades within established ranges, using oscillators and trend-absence (low ADX).", "features": list(set(RANGE_FEATURES + SESSION_FEATURES)), "lookahead_range": [20, 60], "dd_range": [0.10, 0.20]},
        "GNN_Market_Structure": {
            "description": "Uses a GNN to model inter-asset correlations and trades based on market structure.",
            "features": [], # Features are learned, not selected
            "lookahead_range": [80, 150],
            "dd_range": [0.15, 0.30],
            "requires_gnn": True
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
        # Add GNN strategy if it's missing from an old playbook
        if "GNN_Market_Structure" not in playbook:
            logger.info("Adding 'GNN_Market_Structure' to existing playbook.")
            playbook["GNN_Market_Structure"] = DEFAULT_PLAYBOOK["GNN_Market_Structure"]
        logger.info(f"Successfully loaded dynamic playbook from {playbook_path}")
        return playbook
    except (json.JSONDecodeError, IOError) as e:
        logger.error(f"Failed to load or parse playbook file: {e}. Using in-memory default.")
        return DEFAULT_PLAYBOOK

def initialize_nickname_ledger(ledger_path: str, analyzer: GeminiAnalyzer, script_version: str) -> Dict:
    """Loads the nickname ledger, assigning a new nickname if the current version is unseen."""
    logger.info("-> Initializing Nickname Ledger...")
    ledger = {}
    if os.path.exists(ledger_path):
        try:
            with open(ledger_path, 'r') as f:
                ledger = json.load(f)
            logger.info(f"  - Loaded existing nickname ledger from: {ledger_path}")
        except (json.JSONDecodeError, IOError) as e:
            logger.error(f"  - Could not read or parse nickname ledger file. Creating a new one. Error: {e}")
            ledger = {}
    else:
        logger.info("  - No nickname ledger found. A new one will be created.")

    if script_version not in ledger:
        logger.info(f"  - Script version '{script_version}' is new. Generating a codename...")
        used_nicknames = list(ledger.values())
        new_nickname = analyzer.generate_nickname(used_nicknames)
        ledger[script_version] = new_nickname
        logger.info(f"  - Assigned new nickname: '{new_nickname}'")
        
        try:
            with open(ledger_path, 'w') as f:
                json.dump(ledger, f, indent=4)
            logger.info(f"  - Saved updated ledger to: {ledger_path}")
        except IOError as e:
            logger.error(f"  - Failed to save the new nickname to the ledger: {e}")
    else:
        logger.info(f"  - Found existing nickname for '{script_version}': '{ledger[script_version]}'")
        
    return ledger

def check_and_create_hybrid(history: Dict, playbook: Dict, analyzer: GeminiAnalyzer, playbook_path: str) -> Optional[Dict]:
    """Checks if conditions are met to synthesize a new hybrid strategy and does so."""
    historical_runs = history.get("historical_runs", [])
    if len(historical_runs) < 5:
        return None

    successful_strategies = set(
        run.get("strategy_name") for run in historical_runs 
        if run.get("strategy_name") and run.get("final_metrics", {}).get("calmar_ratio", 0) > 0.5
    )
    if len(successful_strategies) < 2:
        return None

    logger.info("--- TRIGGER MET: Conditions are suitable for hybrid strategy synthesis.")
    new_hybrid = analyzer.create_hybrid_strategy(historical_runs, playbook)

    if new_hybrid:
        playbook.update(new_hybrid)
        try:
            with open(playbook_path, 'w') as f:
                json.dump(playbook, f, indent=4)
            logger.info(f"Successfully updated and saved playbook with new hybrid strategy.")
            return playbook
        except IOError as e:
            logger.error(f"Failed to save updated playbook: {e}")
    return None

def perform_strategic_review(history: Dict) -> Dict:
    """Analyzes long-term history to generate a health report for all strategies."""
    logger.info("--- STRATEGIC REVIEW: Analyzing long-term strategy health...")
    health_report = {}
    historical_runs = history.get("historical_runs", [])
    if len(historical_runs) < 3:
        logger.info("--- STRATEGIC REVIEW: Insufficient history for a full review.")
        return health_report

    strategy_names = set(run.get('strategy_name') for run in historical_runs if run.get('strategy_name'))

    for name in strategy_names:
        strategy_runs = [run for run in historical_runs if run.get('strategy_name') == name][-5:]
        if not strategy_runs:
            continue

        failures = sum(1 for run in strategy_runs if run.get("final_metrics", {}).get("calmar_ratio", 0) < 0.1)
        chronic_failure_rate = failures / len(strategy_runs)

        total_cycles = 0
        breaker_trips = 0
        for run in strategy_runs:
            cycles = run.get("cycle_details", [])
            total_cycles += len(cycles)
            breaker_trips += sum(1 for cycle in cycles if cycle.get("Status") == "Circuit Breaker")
        
        circuit_breaker_freq = (breaker_trips / total_cycles) if total_cycles > 0 else 0

        health_report[name] = {
            "ChronicFailureRate": f"{chronic_failure_rate:.0%}",
            "CircuitBreakerFrequency": f"{circuit_breaker_freq:.0%}",
            "RunsAnalyzed": len(strategy_runs)
        }

    # Stagnation Check
    recent_runs = historical_runs[-5:]
    if len(recent_runs) >= 3:
        last_three_strats = [r.get('strategy_name') for r in recent_runs[-3:]]
        if len(set(last_three_strats)) == 1 and last_three_strats[0] is not None:
            stagnant_strat_name = last_three_strats[0]
            
            calmar1 = recent_runs[-3].get("final_metrics", {}).get("calmar_ratio", 0)
            calmar2 = recent_runs[-2].get("final_metrics", {}).get("calmar_ratio", 0)
            calmar3 = recent_runs[-1].get("final_metrics", {}).get("calmar_ratio", 0)
            
            is_not_improving = calmar3 <= calmar2 and calmar2 <= calmar1
            
            if stagnant_strat_name in health_report:
                high_breaker_freq = float(health_report[stagnant_strat_name].get("CircuitBreakerFrequency", "0%").strip('%')) > 33.0

                if is_not_improving or high_breaker_freq:
                    health_report[stagnant_strat_name]["StagnationWarning"] = True
                    logger.warning(f"--- STRATEGIC REVIEW: Stagnation detected for strategy '{stagnant_strat_name}'.")

    if health_report:
        logger.info(f"--- STRATEGIC REVIEW: Health report generated.\n{json.dumps(health_report, indent=2)}")
    else:
        logger.info("--- STRATEGIC REVIEW: No strategies with sufficient history to review.")
        
    return health_report

def determine_timeframe_roles(detected_tfs: List[str]) -> Dict[str, Optional[str]]:
    """Sorts detected timeframes and assigns base, medium, and high roles."""
    tf_with_values = sorted(
        [(tf, FeatureEngineer.TIMEFRAME_MAP.get(tf.upper(), 99999)) for tf in detected_tfs],
        key=lambda x: x[1]
    )
    sorted_tfs = [tf[0] for tf in tf_with_values]
    
    roles = {'base': None, 'medium': None, 'high': None}
    if not sorted_tfs:
        raise ValueError("No timeframes were detected from the provided data files.")
        
    roles['base'] = sorted_tfs[0]
    if len(sorted_tfs) == 2:
        roles['high'] = sorted_tfs[1]
    elif len(sorted_tfs) >= 3:
        roles['medium'] = sorted_tfs[1]
        roles['high'] = sorted_tfs[2]
        if len(sorted_tfs) > 3:
            logger.warning(f"Detected {len(sorted_tfs)} timeframes. Using {roles['base']} as base, {roles['medium']} as medium, and {roles['high']} as high.")
            
    logger.info(f"Dynamically determined timeframe roles: {roles}")
    return roles


def run_single_instance(fallback_config: Dict, framework_history: Dict, is_continuous: bool, playbook: Dict, nickname_ledger: Dict):
    """Encapsulates the logic for a single, complete run of the framework."""
    run_timestamp_str = datetime.now().strftime("%Y%m%d-%H%M%S")
    gemini_analyzer = GeminiAnalyzer()

    current_config = fallback_config.copy()
    current_config['run_timestamp'] = run_timestamp_str
    
    # Instantiate config early to get paths for logging
    temp_config_for_paths = ConfigModel(**current_config, nickname=nickname_ledger.get(current_config['REPORT_LABEL'], ""))
    for handler in logger.handlers[:]:
        if isinstance(handler, logging.FileHandler): logger.removeHandler(handler)
    fh = RotatingFileHandler(temp_config_for_paths.LOG_FILE_PATH, maxBytes=10*1024*1024, backupCount=5)
    fh.setLevel(logging.DEBUG)
    formatter = logging.Formatter('%(asctime)s - [%(levelname)s] - %(message)s')
    fh.setFormatter(formatter)
    logger.addHandler(fh)

    # --- Data Loading and Dynamic Setup ---
    all_files = [f for f in os.listdir(temp_config_for_paths.BASE_PATH) if f.endswith('.csv')]
    data_by_tf, detected_tfs = DataLoader(temp_config_for_paths).load_and_parse_data(filenames=all_files)
    if not data_by_tf: return

    timeframe_roles = determine_timeframe_roles(detected_tfs)
    fe = FeatureEngineer(temp_config_for_paths, timeframe_roles)
    df_featured = fe.create_feature_stack(data_by_tf)
    if df_featured.empty: return

    # --- AI Pre-flight Check ---
    health_report = {}
    if is_continuous:
        health_report = perform_strategic_review(framework_history)
    
    ai_config_suggestion = gemini_analyzer.get_pre_flight_config(
        framework_history, playbook, health_report, [], current_config, exploration_rate=0.25
    )
    
    if 'parameters' in ai_config_suggestion and isinstance(ai_config_suggestion.get('parameters'), dict):
        nested_params = ai_config_suggestion.pop('parameters')
        ai_config_suggestion.update(nested_params)

    # --- AI Parameter Sanitization ---
    def sanitize_param(value, p_min, p_max, p_default, is_float=False):
        try:
            num_val = float(value) if is_float else int(value)
            if p_min <= num_val <= p_max: return num_val
            logger.warning(f"AI suggested parameter value {num_val} is outside of range [{p_min}, {p_max}]. Clamping value.")
            return max(p_min, min(num_val, p_max))
        except (ValueError, TypeError): return p_default

    if 'MAX_DD_PER_CYCLE' in ai_config_suggestion:
        dd = ai_config_suggestion['MAX_DD_PER_CYCLE']
        if isinstance(dd, (int, float)) and dd >= 1.0:
            logger.warning(f"AI suggested MAX_DD_PER_CYCLE >= 1 ({dd}). Interpreting as percentage and scaling to {dd/100.0}.")
            dd = dd / 100.0
        ai_config_suggestion['MAX_DD_PER_CYCLE'] = sanitize_param(dd, 0.05, 0.9, 0.25, is_float=True)
    if 'LOOKAHEAD_CANDLES' in ai_config_suggestion:
        ai_config_suggestion['LOOKAHEAD_CANDLES'] = sanitize_param(ai_config_suggestion['LOOKAHEAD_CANDLES'], 10, 500, 150)
    if 'OPTUNA_TRIALS' in ai_config_suggestion:
        ai_config_suggestion['OPTUNA_TRIALS'] = sanitize_param(ai_config_suggestion['OPTUNA_TRIALS'], 10, 150, 30)

    current_config.update(ai_config_suggestion)
    
    initial_config = ConfigModel(**current_config, nickname=nickname_ledger.get(current_config['REPORT_LABEL'], ""))
    
    logger.info("==========================================================")
    logger.info("  STARTING END-TO-END MULTI-ASSET ML TRADING FRAMEWORK");
    logger.info("==========================================================")
    logger.info(f"-> Starting with configuration: {initial_config.REPORT_LABEL} (Nickname: {initial_config.nickname})")
    logger.info(f"-> MASTER STRATEGY SELECTED: {initial_config.strategy_name}")

    # --- Dynamic Date Calculation ---
    min_data_date = df_featured.index.min()
    initial_training_period = pd.Timedelta(initial_config.TRAINING_WINDOW)
    test_start_date = min_data_date + initial_training_period
    max_date = df_featured.index.max()
    retraining_dates = pd.date_range(start=test_start_date, end=max_date, freq=initial_config.RETRAINING_FREQUENCY)
    logger.info(f"Dynamic walk-forward start date calculated: {test_start_date.date()}. Found {len(retraining_dates)} cycles.")

    all_trades,full_equity_curve,cycle_metrics,all_shap=[],[initial_config.INITIAL_CAPITAL],[],[]
    in_run_historical_cycles = []

    logger.info("-> Stage 3: Starting Adaptive Walk-Forward Analysis with AI Tuning...")
    for i,period_start_date in enumerate(retraining_dates):
        config = ConfigModel(**current_config) # Re-validate config each cycle
        logger.info(f"\n{'='*25} CYCLE {i+1}/{len(retraining_dates)}: {period_start_date.date()} {'='*25}")
        
        strategy_details = playbook.get(config.strategy_name, {})
        is_gnn_strategy = strategy_details.get("requires_gnn", False)

        logger.info(f"  - Using Config: LOOKAHEAD={config.LOOKAHEAD_CANDLES}, OPTUNA_TRIALS={config.OPTUNA_TRIALS}, MAX_DD_PER_CYCLE={config.MAX_DD_PER_CYCLE:.2%}")
        
        all_available_features = [c for c in df_featured.columns if c not in ['Open','High','Low','Close','RealVolume','Symbol','target']]
        valid_features = config.selected_features
        if not is_gnn_strategy:
            valid_features = [feat for feat in config.selected_features if feat in all_available_features]
            if not valid_features:
                chosen_strategy_features = strategy_details.get('features', [])
                default_feature_count = min(len(chosen_strategy_features), 7)
                valid_features = [f for f in chosen_strategy_features[:default_feature_count] if f in all_available_features]
                logger.warning(f"Feature list for cycle was invalid or empty. Falling back to: {valid_features}")
                config.selected_features = valid_features

        train_end=period_start_date-pd.Timedelta(config.FORWARD_TEST_GAP)
        train_start=train_end-pd.Timedelta(config.TRAINING_WINDOW)
        test_end=(period_start_date+pd.Timedelta(config.RETRAINING_FREQUENCY))-pd.Timedelta(days=1)
        if test_end>max_date:test_end=max_date

        df_train_raw=df_featured.loc[train_start:train_end];df_test_chunk=df_featured.loc[period_start_date:test_end]
        if df_train_raw.empty or df_test_chunk.empty: cycle_metrics.append({'Cycle':i+1,'StartDate':period_start_date.date().isoformat(), 'Strategy': config.strategy_name, 'NumTrades':0,'WinRate':"N/A",'CyclePnL':"$0.00",'Status':"Skipped (No Data)"});continue

        df_train_labeled=fe.label_outcomes(df_train_raw,lookahead=config.LOOKAHEAD_CANDLES)
        if df_train_labeled.empty or 'target' not in df_train_labeled.columns:
            cycle_metrics.append({'Cycle':i+1,'StartDate':period_start_date.date().isoformat(), 'Strategy': config.strategy_name, 'NumTrades':0,'WinRate':"N/A",'CyclePnL':"$0.00",'Status':"Skipped (Label Error)"});continue

        trainer=ModelTrainer(config)
        training_result=trainer.train(df_train_labeled, valid_features, strategy_details)
        if training_result is None:
            cycle_metrics.append({'Cycle':i+1,'StartDate':period_start_date.date().isoformat(), 'Strategy': config.strategy_name, 'NumTrades':0,'WinRate':"N/A",'CyclePnL':"$0.00",'Status':"Failed (Training Error)"});continue

        model,best_threshold=training_result
        if trainer.shap_summary is not None:all_shap.append(trainer.shap_summary)

        # --- DECOUPLED PREDICTION LOGIC ---
        logger.info(f"  - Generating predictions for test chunk...")
        if trainer.is_gnn_model:
            X_test = trainer._get_gnn_embeddings_for_test(df_test_chunk)
        else:
            X_test = df_test_chunk[valid_features].copy().fillna(0)
        
        if not X_test.empty:
            class_probs = model.predict_proba(X_test)
            df_test_chunk.loc[:, ['prob_short', 'prob_hold', 'prob_long']] = class_probs
        else:
            logger.warning("  - Test set was empty after feature generation. No predictions made.")
            df_test_chunk.loc[:, ['prob_short', 'prob_hold', 'prob_long']] = 0.33
        # --- END PREDICTION LOGIC ---

        logger.info(f"  - Backtesting on out-of-sample data from {period_start_date.date()} to {test_end.date()}...")
        backtester=Backtester(config)
        chunk_trades_df,chunk_equity,breaker_tripped,breaker_context = backtester.run_backtest_chunk(df_test_chunk,best_threshold,initial_equity=full_equity_curve[-1])

        cycle_pnl=0;cycle_win_rate="N/A"; status = "No Trades"
        if breaker_tripped: status = "Circuit Breaker"
        if not chunk_trades_df.empty:
            all_trades.append(chunk_trades_df);full_equity_curve.extend(chunk_equity.iloc[1:].tolist())
            pnl,wr=chunk_trades_df['PNL'].sum(),(chunk_trades_df['PNL']>0).mean()
            cycle_pnl=pnl;cycle_win_rate=f"{wr:.2%}"
            status = "Success" if not breaker_tripped else "Circuit Breaker"

        cycle_metrics.append({'Cycle':i+1,'StartDate':period_start_date.date().isoformat(),'Strategy': config.strategy_name,'NumTrades':len(chunk_trades_df),'WinRate':cycle_win_rate,'CyclePnL':f"${cycle_pnl:,.2f}",'Status':status})

        cycle_results_for_ai={"cycle":i+1, "strategy_name": config.strategy_name, "objective_score":trainer.study.best_value if trainer.study else 0, "best_threshold":best_threshold, "cycle_pnl":cycle_pnl, "win_rate":cycle_win_rate, "num_trades":len(chunk_trades_df), "status": status, "current_params":{k:v for k,v in config.model_dump().items() if k in ['LOOKAHEAD_CANDLES','OPTUNA_TRIALS','selected_features','MAX_DD_PER_CYCLE']}, "shap_summary": trainer.shap_summary.head(10).to_dict() if trainer.shap_summary is not None else {}, "breaker_context": breaker_context}
        in_run_historical_cycles.append(cycle_results_for_ai)

        suggested_params = gemini_analyzer.analyze_cycle_and_suggest_changes(in_run_historical_cycles, all_available_features)

        if suggested_params:
            logger.info(f"  - AI suggests new parameters for next cycle: {suggested_params}")
            current_config.update(suggested_params)
            if is_continuous:
                logger.info("  - Daemon Mode: Respecting API rate limits (5-minute delay between cycles)..."); time.sleep(300)
            else:
                logger.info("  - Single Run Mode: 5-second delay between cycles..."); time.sleep(5)

    logger.info("\n==========================================================")
    logger.info("                                  WALK-FORWARD ANALYSIS COMPLETE");logger.info("==========================================================")

    run_summary = {"script_version": initial_config.REPORT_LABEL, "nickname": initial_config.nickname, "strategy_name": initial_config.strategy_name, "timestamp": initial_config.run_timestamp, "final_metrics": {}, "initial_config": {k: v for k, v in initial_config.model_dump().items() if k in ['LOOKAHEAD_CANDLES', 'MAX_DD_PER_CYCLE', 'OPTUNA_TRIALS', 'selected_features']}, "top_5_features": {}, "cycle_details": cycle_metrics}
    
    updated_champion = save_run_to_memory(initial_config, run_summary, framework_history)
    if updated_champion:
        framework_history['champion_config'] = updated_champion

    reporter=PerformanceAnalyzer(initial_config)
    if all_shap: aggregated_shap=pd.concat(all_shap).groupby(level=0)['SHAP_Importance'].mean().sort_values(ascending=False).to_frame()
    else: aggregated_shap=None

    final_trades_df=pd.concat(all_trades,ignore_index=True) if all_trades else pd.DataFrame()
    final_metrics = reporter.generate_full_report(final_trades_df,pd.Series(full_equity_curve),cycle_metrics,aggregated_shap, framework_history)
    
    run_summary["final_metrics"] = gemini_analyzer._sanitize_dict(final_metrics)
    if aggregated_shap is not None:
        run_summary["top_5_features"] = aggregated_shap.head(5).to_dict().get('SHAP_Importance', {})
    
    try:
        with open(initial_config.HISTORY_FILE_PATH, 'r') as f: lines = f.readlines()
        if lines:
            lines[-1] = json.dumps(run_summary) + '\n'
            with open(initial_config.HISTORY_FILE_PATH, 'w') as f: f.writelines(lines)
    except Exception as e:
        logger.error(f"Could not finalize run summary in history file: {e}")

def main():
    CONTINUOUS_RUN_HOURS = 0
    MAX_RUNS = 1 

    fallback_config={
        "BASE_PATH": os.getcwd(), "REPORT_LABEL": "ML_Framework_V140_GNN_Hybrid_Integration",
        "strategy_name": "TrendFollower", "INITIAL_CAPITAL": 100000.0,
        "CONFIDENCE_TIERS": {'ultra_high':{'min':0.8,'risk_mult':1.2,'rr':3.0},'high':{'min':0.7,'risk_mult':1.0,'rr':2.5},'standard':{'min':0.6,'risk_mult':0.8,'rr':2.0}},
        "BASE_RISK_PER_TRADE_PCT": 0.01, 
        "RISK_CAP_PER_TRADE_USD": 5000.0,
        "SPREAD_PCTG_OF_ATR": 0.05, "SLIPPAGE_PCTG_OF_ATR": 0.02,
        "OPTUNA_TRIALS": 30, "TRAINING_WINDOW": '365D', "RETRAINING_FREQUENCY": '90D',
        "FORWARD_TEST_GAP": "1D", "LOOKAHEAD_CANDLES": 150, "TREND_FILTER_THRESHOLD": 25.0,
        "BOLLINGER_PERIOD": 20, "STOCHASTIC_PERIOD": 14, "CALCULATE_SHAP_VALUES": True, "MAX_DD_PER_CYCLE": 0.25,
        "GNN_EMBEDDING_DIM": 8, "GNN_EPOCHS": 50,
        "selected_features": []
    }

    run_count = 0
    script_start_time = datetime.now()
    is_continuous = CONTINUOUS_RUN_HOURS > 0 or MAX_RUNS > 1

    bootstrap_config = ConfigModel(**fallback_config, run_timestamp="init")
    playbook = initialize_playbook(bootstrap_config.BASE_PATH)
    analyzer = GeminiAnalyzer()
    nickname_ledger = initialize_nickname_ledger(bootstrap_config.NICKNAME_LEDGER_PATH, analyzer, bootstrap_config.REPORT_LABEL)
    
    while True:
        run_count += 1
        if is_continuous:
            logger.info(f"\n{'='*30} STARTING DAEMON RUN {run_count} {'='*30}\n")
        else:
            logger.info(f"\n{'='*30} STARTING SINGLE RUN {'='*30}\n")

        framework_history = load_memory(bootstrap_config.CHAMPION_FILE_PATH, bootstrap_config.HISTORY_FILE_PATH)

        if is_continuous:
            updated_playbook = check_and_create_hybrid(framework_history, playbook, analyzer, bootstrap_config.PLAYBOOK_FILE_PATH)
            if updated_playbook: playbook = updated_playbook

        try:
            run_single_instance(fallback_config, framework_history, is_continuous, playbook, nickname_ledger)
        except Exception as e:
            logger.critical(f"A critical, unhandled error occurred during run {run_count}: {e}", exc_info=True)
            if is_continuous:
                logger.info("Attempting to continue to the next run after a 1-minute cooldown...")
                time.sleep(60)
            else:
                break

        if not is_continuous:
            logger.info("Single run complete. Exiting.")
            break

        if MAX_RUNS > 1 and run_count >= MAX_RUNS:
            logger.info(f"Reached max run limit of {MAX_RUNS}. Exiting daemon mode.")
            break
        if CONTINUOUS_RUN_HOURS > 0:
            elapsed_hours = (datetime.now() - script_start_time).total_seconds() / 3600
            if elapsed_hours >= CONTINUOUS_RUN_HOURS:
                logger.info(f"Reached max runtime of {CONTINUOUS_RUN_HOURS} hours. Exiting daemon mode.")
                break

        try:
            print("\n")
            logger.info(f"Run {run_count} complete. Next run will start automatically.")
            for i in range(10, 0, -1):
                sys.stdout.write(f"\r>>> Press Ctrl+C to stop the daemon. Continuing in {i:2d} seconds...")
                sys.stdout.flush()
                time.sleep(1)
            sys.stdout.write("\n\n")
            logger.info("Countdown complete. Continuing daemon mode...")
        except KeyboardInterrupt:
            logger.info("\n\nDaemon stopped by user. Exiting gracefully.")
            break

if __name__ == '__main__':
    if os.name == 'nt':
        os.system("chcp 65001 > nul")
    main()

# End_To_End_Advanced_ML_Trading_Framework_PRO_V140_GNN_Hybrid_Integration.py