# End_To_End_Advanced_ML_Trading_Framework_PRO_V119_Hybrid_Synthesis
#
# V119 UPDATE (Hybrid Strategy Synthesis Engine):
# 1. NEW - DYNAMIC PLAYBOOK: The strategy playbook is now externalized to a
#    'strategy_playbook.json' file. The script seeds this file on the first run.
# 2. NEW - HYBRID STRATEGY CREATION: In daemon mode, the AI can now analyze its
#    own performance history from 'framework_memory.json'.
# 3. NEW - AI STRATEGIST ROLE: When triggered, the AI will synthesize a new,
#    novel hybrid strategy by combining the features and logic of past successful
#    runs, save it to the playbook, and make it available for future selection.
#
# V118 UPDATE (Single Run vs. Daemon Mode):
# 1. NEW - DUAL RUN MODES: Added distinct "Single Run" (default) and "Daemon" modes
#    with different inter-cycle timings for flexibility and API rate management.

import os
import re
import json
import time
import warnings
import logging
import sys
from datetime import datetime, timedelta
from logging.handlers import RotatingFileHandler
from typing import List, Dict, Any, Optional, Tuple
import random
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
from sklearn.preprocessing import RobustScaler
from sklearn.utils.class_weight import compute_class_weight
from pydantic import BaseModel, DirectoryPath, confloat, conint

# --- DIAGNOSTICS ---
import xgboost

print("="*60)
print(f"Python Executable: {sys.executable}")
print(f"XGBoost Version Detected: {xgboost.__version__}")
print("="*60)
# --- END DIAGNOSTICS ---

warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', category=UserWarning)
warnings.filterwarnings('ignore', category=pd.errors.PerformanceWarning)

# =============================================================================
# 1. LOGGING SETUP
# =============================================================================
def setup_logging() -> logging.Logger:
    logger = logging.getLogger("ML_Trading_Framework")
    if logger.hasHandlers():
        logger.handlers.clear()
    logger.setLevel(logging.DEBUG)
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    logger.addHandler(ch)
    return logger

logger = setup_logging()
optuna.logging.set_verbosity(optuna.logging.WARNING)

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
        if isinstance(value, (pd.Timestamp, datetime)): return value.isoformat()
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

    def get_pre_flight_config(self, memory: Dict, playbook: Dict, fallback_config: Dict, exploration_rate: float) -> Dict:
        if not self.api_key_valid:
            logger.warning("No API key. Skipping Pre-Flight Check and using default config.")
            return fallback_config
        if not memory.get("historical_runs"):
            logger.info("Framework memory is empty. Using default config for the first run.")
            return fallback_config

        logger.info("-> Stage 0: Pre-Flight Analysis of Framework Memory...")
        champion_strategy = memory.get("champion_config", {}).get("strategy_name")
        is_exploration = random.random() < exploration_rate and champion_strategy is not None

        if is_exploration:
            logger.info(f"--- ENTERING EXPLORATION MODE (Chance: {exploration_rate:.0%}) ---")
            non_champion_strategies = [s for s in playbook if s != champion_strategy]
            if not non_champion_strategies: non_champion_strategies = list(playbook.keys())
            chosen_strategy = random.choice(non_champion_strategies)
            prompt = (
                "You are a trading strategist in **EXPLORATION MODE**. Your goal is to test a non-champion strategy to gather new performance data. "
                f"The current champion strategy is '{champion_strategy}', so you must not choose it. "
                f"Your randomly assigned strategy to test is **'{chosen_strategy}'**. "
                "Based on the playbook definition for this strategy, propose a reasonable starting configuration. "
                "Respond ONLY with a valid JSON object containing: `strategy_name`, `LOOKAHEAD_CANDLES`, `MAX_DD_PER_CYCLE`, `OPTUNA_TRIALS`, and `selected_features`.\n\n"
                f"STRATEGY PLAYBOOK:\n{json.dumps(playbook, indent=2)}\n\n"
                f"FRAMEWORK MEMORY (for context):\n{json.dumps(self._sanitize_dict(memory), indent=2)}"
            )
        else:
            logger.info("--- ENTERING EXPLOITATION MODE (Optimizing Champion) ---")
            prompt = (
                "You are a master trading strategist. Your task is to select the optimal **master strategy** for an entire new run from the provided `strategy_playbook`. "
                "Analyze the `framework_memory`, which contains the current `champion_config` and all `historical_runs`. Your decision should be based on which strategy archetype has shown the best **risk-adjusted performance (high Calmar Ratio, high Profit Factor)** in the past.\n\n"
                "1.  **Review the Champion**: Note which strategy the current champion used. This is your baseline to beat.\n"
                "2.  **Select a Strategy**: Choose the `strategy_name` from the playbook that you believe has the highest potential for this new run. You can choose the champion again if you think it can be improved.\n\n"
                "After selecting the strategy, define its initial parameters for the run:\n"
                "- `LOOKAHEAD_CANDLES`: Choose a value within the strategy's recommended `lookahead_range`.\n"
                "- `MAX_DD_PER_CYCLE`: Choose a value within the strategy's recommended `dd_range`.\n"
                "- `selected_features`: Select a promising SUBSET of features from the strategy's `features` list.\n"
                "- `OPTUNA_TRIALS`: Set between 20 and 40.\n\n"
                "Respond ONLY with a valid JSON object containing: `strategy_name`, `LOOKAHEAD_CANDLES`, `MAX_DD_PER_CYCLE`, `OPTUNA_TRIALS`, and `selected_features`.\n\n"
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
        prompt = (
            "You are an expert trading model analyst. Your primary goal is to tune the parameters of a **pre-selected master strategy** to adapt to changing market conditions within a walk-forward run.\n\n"
            "Analyze the recent cycle history. If a 'Circuit Breaker' was tripped, you MUST analyze the `breaker_context` to understand the failure mode (e.g., 'death by a thousand cuts' vs. catastrophic loss) and make targeted suggestions.\n\n"
            "Based on your analysis, suggest NEW parameters for the next cycle, **adhering to the available features for the chosen strategy**.\n\n"
            "1.  **Risk Management**: Suggest a new float value for `MAX_DD_PER_CYCLE` (between 0.15 and 0.40).\n"
            "2.  **Model Parameters**: Suggest new integer values for `LOOKAHEAD_CANDLES` (50-200) and `OPTUNA_TRIALS` (15-30).\n"
            "3.  **Feature Selection**: Select a SUBSET of features from the `available_features` list ONLY.\n\n"
            "Respond ONLY with a valid JSON object containing the keys: `MAX_DD_PER_CYCLE`, `LOOKAHEAD_CANDLES`, `OPTUNA_TRIALS`, and `selected_features`.\n\n"
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

        # Find the two best-performing, different strategies
        champions = {}
        for run in historical_runs:
            name = run.get("strategy_name")
            calmar = run.get("final_metrics", {}).get("calmar_ratio", 0)
            if calmar > champions.get(name, {}).get("calmar", -1):
                champions[name] = {"calmar": calmar, "run_summary": run}

        if len(champions) < 2:
            logger.info("--- HYBRID SYNTHESIS: Need at least two different, successful strategies in history to create a hybrid. Skipping.")
            return None

        # Sort by calmar and pick top 2
        sorted_champs = sorted(champions.values(), key=lambda x: x["calmar"], reverse=True)
        champ1_summary = sorted_champs[0]["run_summary"]
        champ2_summary = sorted_champs[1]["run_summary"]

        def format_summary(summary: Dict):
            return {
                "strategy_name": summary.get("strategy_name"),
                "calmar_ratio": summary.get("final_metrics", {}).get("calmar_ratio"),
                "profit_factor": summary.get("final_metrics", {}).get("profit_factor"),
                "win_rate": summary.get("final_metrics", {}).get("win_rate"),
                "top_5_features": summary.get("top_5_features")
            }

        prompt = (
            "You are a master quantitative strategist. Your task is to synthesize a new HYBRID trading strategy by combining the best elements of two successful, but different, historical strategies.\n\n"
            "Analyze the provided performance data for the following two champion archetypes:\n"
            f"1.  **Strategy A:**\n{json.dumps(format_summary(champ1_summary), indent=2)}\n\n"
            f"2.  **Strategy B:**\n{json.dumps(format_summary(champ2_summary), indent=2)}\n\n"
            "Based on this analysis, define a new hybrid strategy. You must:\n"
            "1.  **Name:** Create a unique name for the new strategy that reflects its parents (e.g., 'Hybrid_TrendMomentum_V1'). It CANNOT be one of the existing strategy names.\n"
            "2.  **Description:** Write a brief (1-2 sentence) description of its goal (e.g., 'Uses HTF trend context to identify direction, but enters on short-term momentum signals.').\n"
            "3.  **Features:** Create a new feature list (5-7 features) by selecting the most powerful and complementary features from BOTH strategies. Do not just merge the lists; select the best combination.\n"
            "4.  **Parameters:** Suggest a new `lookahead_range` and `dd_range` that are a logical blend of the parent strategies.\n\n"
            f"Existing strategy names to avoid are: {list(current_playbook.keys())}\n\n"
            "Respond ONLY with a valid JSON object for the new strategy, where the key is the new strategy name. Example format:\n"
            "```json\n"
            "{\n"
            "  \"Hybrid_NewStrategy_V1\": {\n"
            "    \"description\": \"Your new description here.\",\n"
            "    \"features\": [\"feature1\", \"feature2\", \"feature3\", \"feature4\", \"feature5\"],\n"
            "    \"lookahead_range\": [100, 200],\n"
            "    \"dd_range\": [0.20, 0.35]\n"
            "  }\n"
            "}\n"
            "```"
        )
        logger.info("--- HYBRID SYNTHESIS: Prompting AI to create new strategy...")
        response_text = self._call_gemini(prompt)
        logger.info(f"--- HYBRID SYNTHESIS (Raw AI Response): {response_text}")
        new_hybrid = self._extract_json_from_response(response_text)

        if new_hybrid and isinstance(new_hybrid, dict) and len(new_hybrid) == 1:
            hybrid_name = list(new_hybrid.keys())[0]
            if hybrid_name in current_playbook:
                logger.warning(f"--- HYBRID SYNTHESIS: AI created a hybrid with a name that already exists ('{hybrid_name}'). Discarding.")
                return None
            logger.info(f"--- HYBRID SYNTHESIS: Successfully synthesized new strategy: '{hybrid_name}'")
            return new_hybrid
        else:
            logger.error(f"--- HYBRID SYNTHESIS: Failed to parse a valid hybrid strategy from AI response.")
            return None


# =============================================================================
# 3. CONFIGURATION & VALIDATION
# =============================================================================
class ConfigModel(BaseModel):
    BASE_PATH: DirectoryPath; REPORT_LABEL: str; FORWARD_TEST_START_DATE: str; INITIAL_CAPITAL: confloat(gt=0)
    CONFIDENCE_TIERS: Dict[str, Dict[str, float]]; BASE_RISK_PER_TRADE_PCT: confloat(gt=0, lt=1)
    SPREAD_PCTG_OF_ATR: confloat(ge=0); SLIPPAGE_PCTG_OF_ATR: confloat(ge=0); OPTUNA_TRIALS: conint(gt=0)
    TRAINING_WINDOW: str; RETRAINING_FREQUENCY: str; FORWARD_TEST_GAP: str; LOOKAHEAD_CANDLES: conint(gt=0)
    MODEL_SAVE_PATH: str = ""; PLOT_SAVE_PATH: str = ""; REPORT_SAVE_PATH: str = ""; SHAP_PLOT_PATH: str = ""
    LOG_FILE_PATH: str = ""; MEMORY_FILE_PATH: str = ""; PLAYBOOK_FILE_PATH: str = ""
    TREND_FILTER_THRESHOLD: confloat(gt=0) = 25.0
    BOLLINGER_PERIOD: conint(gt=0) = 20; STOCHASTIC_PERIOD: conint(gt=0) = 14; CALCULATE_SHAP_VALUES: bool = True
    MAX_DD_PER_CYCLE: confloat(gt=0.05, lt=1.0) = 0.25
    selected_features: List[str]
    run_timestamp: str
    strategy_name: str

    def __init__(self, **data: Any):
        super().__init__(**data)
        results_dir = os.path.join(self.BASE_PATH, "Results")
        run_id = f"{self.REPORT_LABEL}_{self.strategy_name}_{self.run_timestamp}"
        result_folder_path=os.path.join(results_dir, self.REPORT_LABEL);os.makedirs(result_folder_path,exist_ok=True)

        self.MODEL_SAVE_PATH=os.path.join(result_folder_path,f"{run_id}_model.json")
        self.PLOT_SAVE_PATH=os.path.join(result_folder_path,f"{run_id}_equity_curve.png")
        self.REPORT_SAVE_PATH=os.path.join(result_folder_path,f"{run_id}_report.txt")
        self.SHAP_PLOT_PATH=os.path.join(result_folder_path,f"{run_id}_shap_summary.png")
        self.LOG_FILE_PATH=os.path.join(result_folder_path,f"{run_id}_run.log")
        self.MEMORY_FILE_PATH=os.path.join(results_dir,"framework_memory.json")
        self.PLAYBOOK_FILE_PATH=os.path.join(results_dir,"strategy_playbook.json")

# =============================================================================
# 4. DATA LOADER & 5. FEATURE ENGINEERING
# =============================================================================
class DataLoader:
    def __init__(self, config: ConfigModel): self.config = config
    def load_and_parse_data(self, filenames: List[str]) -> Optional[Dict[str, pd.DataFrame]]:
        logger.info("-> Stage 1: Loading and Preparing Multi-Timeframe Data...")
        data_by_tf: Dict[str, List[pd.DataFrame]] = {'D1': [], 'H1': [], 'M15': []}
        for filename in filenames:
            file_path = os.path.join(self.config.BASE_PATH, filename)
            if not os.path.exists(file_path): logger.warning(f"  - File not found, skipping: {file_path}"); continue
            try:
                parts=filename.split('_');symbol,tf_str=parts[0],parts[1];tf='D1' if 'Daily' in tf_str else tf_str
                if tf not in data_by_tf: continue
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
                df['Symbol']=symbol;data_by_tf[tf].append(df)
            except Exception as e: logger.error(f"  - Failed to load {filename}: {e}", exc_info=True)

        processed_dfs:Dict[str,pd.DataFrame]={}
        for tf,dfs in data_by_tf.items():
            if dfs:
                combined=pd.concat(dfs);all_symbols_df=[df[~df.index.duplicated(keep='first')].sort_index() for _,df in combined.groupby('Symbol')]
                final_combined=pd.concat(all_symbols_df).sort_index();final_combined['RealVolume']=pd.to_numeric(final_combined['RealVolume'],errors='coerce').fillna(0)
                processed_dfs[tf]=final_combined;logger.info(f"  - Processed {tf}: {len(final_combined):,} rows for {len(final_combined['Symbol'].unique())} symbols.")
            else:logger.warning(f"  - No data found for {tf} timeframe.");processed_dfs[tf]=pd.DataFrame()
        if not processed_dfs or any(df.empty for df in processed_dfs.values()):logger.critical("  - Data loading failed.");return None
        logger.info("[SUCCESS] Data loading and preparation complete.");return processed_dfs

class FeatureEngineer:
    def __init__(self, config: ConfigModel): self.config = config
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

    def _calculate_m15_native(self, g:pd.DataFrame)->pd.DataFrame:
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
        g_out['market_regime']=np.where(g_out['ADX']>self.config.TREND_FILTER_THRESHOLD,1,0) # 1 for Trend, 0 for Range
        return g_out

    def create_feature_stack(self, data_by_tf: Dict[str, pd.DataFrame]) -> pd.DataFrame:
        logger.info("-> Stage 2: Engineering Features from Multi-Timeframe Data...")
        if any(df.empty for df in data_by_tf.values()): logger.critical("  - One or more timeframes have no data."); return pd.DataFrame()

        df_d1=self._calculate_htf_features(data_by_tf['D1'],'D1',20,14);df_h1=self._calculate_htf_features(data_by_tf['H1'],'H1',50,14)
        df_m15_base_list = [self._calculate_m15_native(group) for _, group in data_by_tf['M15'].groupby('Symbol')]
        df_m15_base = pd.concat(df_m15_base_list).reset_index()

        df_h1_sorted=df_h1.sort_values('Timestamp');df_d1_sorted=df_d1.sort_values('Timestamp')
        df_merged=pd.merge_asof(df_m15_base.sort_values('Timestamp'),df_h1_sorted,on='Timestamp',by='Symbol',direction='backward')
        df_merged=pd.merge_asof(df_merged.sort_values('Timestamp'),df_d1_sorted,on='Timestamp',by='Symbol',direction='backward')
        df_final=df_merged.set_index('Timestamp').copy()

        df_final['adx_x_h1_trend']=df_final['ADX']*df_final['H1_ctx_Trend']
        df_final['atr_x_d1_trend']=df_final['ATR']*df_final['D1_ctx_Trend']

        all_features = ModelTrainer.BASE_FEATURES
        feature_cols = list(set([f for f in all_features if f in df_final.columns]))
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

class ModelTrainer:
    # Expanded Feature Sets for Playbook
    TREND_FEATURES = ['ADX', 'adx_x_h1_trend', 'atr_x_d1_trend', 'H1_ctx_Trend', 'D1_ctx_Trend', 'H1_ctx_SMA', 'D1_ctx_SMA']
    REVERSAL_FEATURES = ['RSI', 'stoch_k', 'stoch_d', 'bollinger_bandwidth']
    VOLATILITY_FEATURES = ['ATR', 'bollinger_bandwidth']
    MOMENTUM_FEATURES = ['momentum_10', 'momentum_20', 'RSI']
    RANGE_FEATURES = ['RSI', 'stoch_k', 'ADX', 'bollinger_bandwidth']
    PRICE_ACTION_FEATURES = ['is_doji', 'is_engulfing']
    SEASONALITY_FEATURES = ['month', 'week_of_year', 'day_of_month']
    SESSION_FEATURES = ['hour', 'day_of_week']
    BASE_FEATURES = list(set(TREND_FEATURES + REVERSAL_FEATURES + VOLATILITY_FEATURES + MOMENTUM_FEATURES + RANGE_FEATURES + PRICE_ACTION_FEATURES + SEASONALITY_FEATURES + SESSION_FEATURES))

    def __init__(self,config:ConfigModel):
        self.config=config
        self.shap_summary:Optional[pd.DataFrame]=None
        self.class_weights:Optional[Dict[int,float]]=None
        self.best_threshold=0.5
        self.study: Optional[optuna.study.Study] = None

    def train(self,df_train:pd.DataFrame, feature_list: List[str])->Optional[Tuple[Pipeline,float]]:
        logger.info(f"  - Starting model training using {len(feature_list)} features...");
        y_map={-1:0,0:1,1:2};y=df_train['target'].map(y_map).astype(int)
        X=df_train[feature_list].copy().fillna(0)

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
        final_pipeline=self._train_final_model(self.study.best_params,X_train_val,y_train_val)
        if final_pipeline is None:logger.error("  - Training aborted: Final model training failed.");return None

        logger.info("  - [SUCCESS] Model training complete.");return final_pipeline, self.best_threshold

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

    def _train_final_model(self,best_params:Dict,X:pd.DataFrame,y:pd.Series)->Optional[Pipeline]:
        logger.info("    - Training final model...");
        try:
            best_params.pop('early_stopping_rounds', None)
            final_params={'objective':'multi:softprob','num_class':3,'eval_metric':'mlogloss','booster':'gbtree','tree_method':'hist','use_label_encoder':False,'seed':42,**best_params}
            final_pipeline=Pipeline([('scaler',RobustScaler()),('model',xgb.XGBClassifier(**final_params))])
            fit_params={'model__sample_weight':y.map(self.class_weights)}
            final_pipeline.fit(X,y,**fit_params)
            if self.config.CALCULATE_SHAP_VALUES:self._generate_shap_summary(final_pipeline.named_steps['model'],final_pipeline.named_steps['scaler'].transform(X),X.columns)
            return final_pipeline
        except Exception as e:logger.error(f"    - Error during final model training: {e}",exc_info=True);return None

    def _generate_shap_summary(self, model: xgb.XGBClassifier, X_scaled: np.ndarray, feature_names: pd.Index):
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
    def run_backtest_chunk(self,df_chunk_in:pd.DataFrame,model:Pipeline,feature_list:List[str],confidence_threshold:float,initial_equity:float)->Tuple[pd.DataFrame,pd.Series,bool,Optional[Dict]]:
        if df_chunk_in.empty:return pd.DataFrame(),pd.Series([initial_equity]), False, None
        df_chunk=df_chunk_in.copy()
        X_test=df_chunk[feature_list].copy().fillna(0)
        class_probs=model.predict_proba(X_test)
        df_chunk.loc[:,['prob_short','prob_hold','prob_long']]=class_probs
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
                        tier=self.config.CONFIDENCE_TIERS[tier_name];risk_amt=equity*self.config.BASE_RISK_PER_TRADE_PCT*tier['risk_mult']
                        sl_dist=(atr*1.5)+(atr*self.config.SPREAD_PCTG_OF_ATR)+(atr*self.config.SLIPPAGE_PCTG_OF_ATR);tp_dist=(atr*1.5*tier['rr'])
                        if tp_dist<=0:continue
                        entry_price=candle['Close'];sl_price,tp_price=entry_price-sl_dist*direction,entry_price+tp_dist*direction
                        open_positions[symbol]={'direction':direction,'sl':sl_price,'tp':tp_price,'risk_amt':risk_amt,'rr':tier['rr'],'confidence':confidence}
        return pd.DataFrame(trades),pd.Series(equity_curve), circuit_breaker_tripped, breaker_context

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
        plt.title(f"{self.config.REPORT_LABEL} - Walk-Forward Equity Curve",fontsize=16,weight='bold')
        plt.xlabel("Trade Number",fontsize=12);plt.ylabel("Equity ($)",fontsize=12);plt.grid(True,which='both',linestyle=':');plt.savefig(self.config.PLOT_SAVE_PATH);plt.close()
        logger.info(f"  - Equity curve plot saved to: {self.config.PLOT_SAVE_PATH}")

    def plot_shap_summary(self,shap_summary:pd.DataFrame):
        plt.style.use('seaborn-v0_8-darkgrid');plt.figure(figsize=(12,10))
        shap_summary.head(20).sort_values(by='SHAP_Importance').plot(kind='barh',legend=False,color='mediumseagreen')
        title_str = f"{self.config.REPORT_LABEL} ({self.config.strategy_name}) - Aggregated Feature Importance"
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

    def _get_comparison_block(self, metrics: Dict, memory: Dict) -> str:
        champion = memory.get('champion_config') if memory else None
        historical_runs = memory.get('historical_runs', [])
        previous_run = historical_runs[-1] if historical_runs else None

        def get_data(source: Dict, key: str, is_percent: bool = False):
            if not source: return "N/A"
            val = source.get("final_metrics", {}).get(key)
            if val is None: return "N/A"
            return f"{val:.2f}%" if is_percent else f"{val:.2f}"

        def get_info(source: Dict, key: str):
            if not source: return "N/A"
            return source.get(key, 'N/A')

        c_ver = self.config.REPORT_LABEL
        c_strat = self.config.strategy_name
        c_mar = get_data(metrics, 'mar_ratio')
        c_mdd = get_data(metrics, 'max_drawdown_pct', is_percent=True)
        c_pf = get_data(metrics, 'profit_factor')

        p_ver = get_info(previous_run, 'script_version')
        p_strat = get_info(previous_run, 'strategy_name')
        p_mar = get_data(previous_run, 'mar_ratio')
        p_mdd = get_data(previous_run, 'max_drawdown_pct', is_percent=True)
        p_pf = get_data(previous_run, 'profit_factor')

        champ_ver = get_info(champion, 'script_version')
        champ_strat = get_info(champion, 'strategy_name')
        champ_mar = get_data(champion, 'mar_ratio')
        champ_mdd = get_data(champion, 'max_drawdown_pct', is_percent=True)
        champ_pf = get_data(champion, 'profit_factor')

        block = f"""
--------------------------------------------------------------------------------
I. PERFORMANCE vs. HISTORY
--------------------------------------------------------------------------------
                         {_center('Current Run', 20)}|{_center('Previous Run', 32)}|{_center('All-Time Champion', 32)}
-------------------- -------------------- -------------------------------- --------------------------------
Run Label            {_ljust(c_ver, 20)}|{_ljust(p_ver, 32)}|{_ljust(champ_ver, 32)}
Strategy             {_ljust(c_strat, 20)}|{_ljust(p_strat, 32)}|{_ljust(champ_strat, 32)}
MAR Ratio:           {_rjust(c_mar, 20)}|{_rjust(p_mar, 32)}|{_rjust(champ_mar, 32)}
Max Drawdown:        {_rjust(c_mdd, 20)}|{_rjust(p_mdd, 32)}|{_rjust(champ_mdd, 32)}
Profit Factor:       {_rjust(c_pf, 20)}|{_rjust(p_pf, 32)}|{_rjust(champ_pf, 32)}
--------------------------------------------------------------------------------
"""
        return block

    def generate_text_report(self,m:Dict[str,Any],cycle_metrics:List[Dict],aggregated_shap:Optional[pd.DataFrame]=None, framework_memory:Optional[Dict]=None):
        comparison_block = self._get_comparison_block({"final_metrics": m}, framework_memory) if framework_memory else ""
        cycle_df=pd.DataFrame(cycle_metrics);cycle_report="Per-Cycle Performance:\n"+cycle_df.to_string(index=False) if not cycle_df.empty else "No trades were executed."
        shap_report="Aggregated Feature Importance (SHAP):\n"+aggregated_shap.head(15).to_string() if aggregated_shap is not None else "SHAP summary was not generated."
        report=f"""
\n================================================================================
                                ADAPTIVE WALK-FORWARD PERFORMANCE REPORT
================================================================================
Report Label: {self.config.REPORT_LABEL} ({self.config.strategy_name})
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

def _ljust(text, width): return str(text).ljust(width)
def _rjust(text, width): return str(text).rjust(width)
def _center(text, width): return str(text).center(width)

# =============================================================================
# 9. FRAMEWORK ORCHESTRATION & MEMORY
# =============================================================================
def migrate_and_load_memory(memory_path: str) -> Dict:
    default_memory = {"champion_config": None, "historical_runs": []}
    if not os.path.exists(memory_path): return default_memory
    try:
        with open(memory_path, 'r') as f: memory = json.load(f)
        if "champion_config" not in memory or "historical_runs" not in memory:
            logger.warning("Memory file is malformed. Starting fresh."); return default_memory
        return memory
    except (json.JSONDecodeError, IOError) as e:
        logger.error(f"Could not read or parse memory file at {memory_path}: {e}"); return default_memory

def save_run_to_memory(memory_path: str, new_run_summary: Dict, current_memory: Dict):
    try:
        current_memory["historical_runs"].append(new_run_summary)
        current_champion = current_memory.get("champion_config")
        new_calmar = new_run_summary.get("final_metrics", {}).get("calmar_ratio", 0)

        if current_champion is None or (new_calmar is not None and new_calmar > current_champion.get("final_metrics", {}).get("calmar_ratio", 0)):
            champion_calmar = current_champion.get("final_metrics", {}).get("calmar_ratio", 0) if current_champion else -1
            logger.info(f"NEW CHAMPION! Current run's Calmar Ratio ({new_calmar:.2f}) beats the previous champion's ({champion_calmar:.2f}).")
            current_memory["champion_config"] = new_run_summary
        else:
            champ_calmar_val = current_champion.get("final_metrics", {}).get("calmar_ratio", 0)
            logger.info(f"Current run's Calmar Ratio ({new_calmar:.2f}) did not beat the champion's ({champ_calmar_val:.2f}).")

        with open(memory_path, 'w') as f: json.dump(current_memory, f, indent=4)
        logger.info(f"-> Run summary and champion status updated in framework memory: {memory_path}")
    except IOError as e:
        logger.error(f"Could not write to memory file at {memory_path}: {e}")

def initialize_playbook(base_path: str) -> Dict:
    """Loads the strategy playbook from JSON, creating it from a default if it doesn't exist."""
    results_dir = os.path.join(base_path, "Results")
    os.makedirs(results_dir, exist_ok=True)
    playbook_path = os.path.join(results_dir, "strategy_playbook.json")

    DEFAULT_PLAYBOOK = {
        "TrendFollower": {"description": "Aims to catch long trends using HTF context and trend strength.", "features": list(set(ModelTrainer.TREND_FEATURES + ModelTrainer.SESSION_FEATURES)), "lookahead_range": (150, 250), "dd_range": (0.25, 0.40)},
        "MeanReversion": {"description": "Aims for short-term reversals using oscillators.", "features": list(set(ModelTrainer.REVERSAL_FEATURES + ModelTrainer.SESSION_FEATURES)),"lookahead_range": (40, 80), "dd_range": (0.15, 0.25)},
        "VolatilityBreakout": {"description": "Trades breakouts during high volatility sessions.", "features": list(set(ModelTrainer.VOLATILITY_FEATURES + ModelTrainer.SESSION_FEATURES)), "lookahead_range": (60, 120), "dd_range": (0.20, 0.35)},
        "Momentum": {"description": "Capitalizes on short-term price momentum.", "features": list(set(ModelTrainer.MOMENTUM_FEATURES + ModelTrainer.SESSION_FEATURES)), "lookahead_range": (30, 90), "dd_range": (0.18, 0.30)},
        "RangeBound": {"description": "Trades within established ranges, using oscillators and trend-absence (low ADX).", "features": list(set(ModelTrainer.RANGE_FEATURES + ModelTrainer.SESSION_FEATURES)), "lookahead_range": (20, 60), "dd_range": (0.10, 0.20)},
        "Seasonality": {"description": "Leverages recurring seasonal patterns or calendar effects.", "features": list(set(ModelTrainer.SEASONALITY_FEATURES + ModelTrainer.SESSION_FEATURES)), "lookahead_range": (50, 120), "dd_range": (0.15, 0.28)},
        "PriceAction": {"description": "Trades based on the statistical outcomes of historical candlestick formations.", "features": list(set(ModelTrainer.PRICE_ACTION_FEATURES + ModelTrainer.SESSION_FEATURES)), "lookahead_range": (20, 80), "dd_range": (0.10, 0.25)}
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
        logger.info(f"Successfully loaded dynamic playbook from {playbook_path}")
        return playbook
    except (json.JSONDecodeError, IOError) as e:
        logger.error(f"Failed to load or parse playbook file: {e}. Using in-memory default.")
        return DEFAULT_PLAYBOOK

def check_and_create_hybrid(history: Dict, playbook: Dict, analyzer: GeminiAnalyzer, playbook_path: str) -> Optional[Dict]:
    """Checks if conditions are met to synthesize a new hybrid strategy and does so."""
    historical_runs = history.get("historical_runs", [])
    if len(historical_runs) < 5: # Require at least 5 runs to have meaningful history
        return None

    # Find successful, different strategies (Calmar > 0.5 as a simple threshold for success)
    successful_strategies = set(
        run["strategy_name"] for run in historical_runs if run.get("final_metrics", {}).get("calmar_ratio", 0) > 0.5
    )

    if len(successful_strategies) < 2:
        return None # Need at least two different successful archetypes

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
    return None

def run_single_instance(fallback_config: Dict, framework_history: Dict, is_continuous: bool, playbook: Dict):
    """Encapsulates the logic for a single, complete run of the framework."""
    run_timestamp_str = datetime.now().strftime("%Y%m%d-%H%M%S")
    gemini_analyzer = GeminiAnalyzer() # A local instance for this specific run
    
    current_config = fallback_config.copy()
    current_config['run_timestamp'] = run_timestamp_str

    ai_config_suggestion = gemini_analyzer.get_pre_flight_config(framework_history, playbook, current_config, exploration_rate=0.25)
    current_config.update(ai_config_suggestion)

    chosen_strategy_features = playbook.get(current_config['strategy_name'], {}).get('features', [])
    if not chosen_strategy_features: # Fallback if AI hallucinates a strategy name
        logger.warning(f"Strategy '{current_config['strategy_name']}' not in playbook. Falling back to TrendFollower.")
        current_config['strategy_name'] = 'TrendFollower'
        chosen_strategy_features = playbook.get('TrendFollower', {}).get('features', [])

    current_config['selected_features'] = [feat for feat in current_config['selected_features'] if feat in chosen_strategy_features]
    if not current_config['selected_features']:
        current_config['selected_features'] = chosen_strategy_features[:5]
        logger.warning("AI-suggested features were not valid for the chosen strategy. Falling back to a default subset.")

    initial_config = ConfigModel(**current_config)

    for handler in logger.handlers[:]:
        if isinstance(handler, logging.FileHandler): logger.removeHandler(handler)
    fh = RotatingFileHandler(initial_config.LOG_FILE_PATH, maxBytes=10*1024*1024, backupCount=5)
    fh.setLevel(logging.DEBUG)
    formatter = logging.Formatter('%(asctime)s - [%(levelname)s] - %(message)s')
    fh.setFormatter(formatter)
    logger.addHandler(fh)

    logger.info("==========================================================")
    logger.info("  STARTING END-TO-END MULTI-ASSET ML TRADING FRAMEWORK");
    logger.info("==========================================================")
    logger.info(f"-> Starting with configuration: {initial_config.REPORT_LABEL}")
    logger.info(f"-> MASTER STRATEGY SELECTED: {initial_config.strategy_name}")

    files_to_process=[
        "AUDUSD_Daily_202001060000_202506020000.csv","AUDUSD_H1_202001060000_202506021800.csv","AUDUSD_M15_202105170000_202506021830.csv",
        "EURUSD_Daily_202001060000_202506020000.csv","EURUSD_H1_202001060000_202506021800.csv","EURUSD_M15_202106020100_202506021830.csv",
        "GBPUSD_Daily_202001060000_202506020000.csv","GBPUSD_H1_202001060000_202506021800.csv","GBPUSD_M15_202106020015_202506021830.csv",
        "USDCAD_Daily_202001060000_202506020000.csv","USDCAD_H1_202001060000_202506021800.csv","USDCAD_M15_202105170000_202506021830.csv"
    ]
    
    try:
        data_by_tf=DataLoader(initial_config).load_and_parse_data(filenames=files_to_process)
        if not data_by_tf:return
        fe=FeatureEngineer(initial_config);df_featured=fe.create_feature_stack(data_by_tf)
        if df_featured.empty:return
    except Exception as e:logger.critical(f"[FATAL] Initial setup failed: {e}",exc_info=True);return

    test_start_date=pd.to_datetime(initial_config.FORWARD_TEST_START_DATE);max_date=df_featured.index.max()
    retraining_dates=pd.date_range(start=test_start_date,end=max_date,freq=initial_config.RETRAINING_FREQUENCY)
    all_trades,full_equity_curve,cycle_metrics,all_shap=[],[initial_config.INITIAL_CAPITAL],[],[]
    in_run_historical_cycles = []

    logger.info("-> Stage 3: Starting Adaptive Walk-Forward Analysis with AI Tuning...")
    for i,period_start_date in enumerate(retraining_dates):
        config=ConfigModel(**current_config)
        logger.info(f"\n{'='*25} CYCLE {i+1}/{len(retraining_dates)}: {period_start_date.date()} {'='*25}")
        logger.info(f"  - Using Config: LOOKAHEAD={config.LOOKAHEAD_CANDLES}, OPTUNA_TRIALS={config.OPTUNA_TRIALS}, MAX_DD_CYCLE={config.MAX_DD_PER_CYCLE}")
        logger.info(f"  - Features for this cycle: {config.selected_features}")

        train_end=period_start_date-pd.Timedelta(config.FORWARD_TEST_GAP)
        train_start=train_end-pd.Timedelta(config.TRAINING_WINDOW)
        test_end=(period_start_date+pd.Timedelta(config.RETRAINING_FREQUENCY))-pd.Timedelta(days=1)
        if test_end>max_date:test_end=max_date

        df_train_raw=df_featured.loc[train_start:train_end];df_test_chunk=df_featured.loc[period_start_date:test_end]
        if df_train_raw.empty or df_test_chunk.empty: cycle_metrics.append({'Cycle':i+1,'StartDate':period_start_date.date(), 'Strategy': config.strategy_name, 'NumTrades':0,'WinRate':"N/A",'CyclePnL':"$0.00",'Status':"Skipped (No Data)"});continue

        df_train_labeled=fe.label_outcomes(df_train_raw,lookahead=config.LOOKAHEAD_CANDLES)
        if df_train_labeled.empty or 'target' not in df_train_labeled.columns:
            cycle_metrics.append({'Cycle':i+1,'StartDate':period_start_date.date(), 'Strategy': config.strategy_name, 'NumTrades':0,'WinRate':"N/A",'CyclePnL':"$0.00",'Status':"Skipped (Label Error)"});continue

        trainer=ModelTrainer(config);training_result=trainer.train(df_train_labeled, feature_list=config.selected_features)
        if training_result is None:
            cycle_metrics.append({'Cycle':i+1,'StartDate':period_start_date.date(), 'Strategy': config.strategy_name, 'NumTrades':0,'WinRate':"N/A",'CyclePnL':"$0.00",'Status':"Failed (Training Error)"});continue

        model,best_threshold=training_result
        if trainer.shap_summary is not None:all_shap.append(trainer.shap_summary)

        logger.info(f"  - Backtesting on out-of-sample data from {period_start_date.date()} to {test_end.date()}...")
        backtester=Backtester(config)
        chunk_trades_df,chunk_equity,breaker_tripped,breaker_context = backtester.run_backtest_chunk(df_test_chunk,model,config.selected_features,best_threshold,initial_equity=full_equity_curve[-1])

        cycle_pnl=0;cycle_win_rate="N/A"; status = "No Trades"
        if breaker_tripped: status = "Circuit Breaker"
        if not chunk_trades_df.empty:
            all_trades.append(chunk_trades_df);full_equity_curve.extend(chunk_equity.iloc[1:].tolist())
            pnl,wr=chunk_trades_df['PNL'].sum(),(chunk_trades_df['PNL']>0).mean()
            cycle_pnl=pnl;cycle_win_rate=f"{wr:.2%}"
            status = "Success" if not breaker_tripped else "Circuit Breaker"

        cycle_metrics.append({'Cycle':i+1,'Strategy': config.strategy_name,'StartDate':period_start_date.date(),'NumTrades':len(chunk_trades_df),'WinRate':cycle_win_rate,'CyclePnL':f"${cycle_pnl:,.2f}",'Status':status})

        cycle_results_for_ai={"cycle":i+1, "strategy_name": config.strategy_name, "objective_score":trainer.study.best_value if trainer.study else 0, "best_threshold":best_threshold, "cycle_pnl":cycle_pnl, "win_rate":cycle_win_rate, "num_trades":len(chunk_trades_df), "status": status, "current_params":{k:v for k,v in config.model_dump().items() if k in ['LOOKAHEAD_CANDLES','OPTUNA_TRIALS','selected_features','MAX_DD_PER_CYCLE']}, "shap_summary": trainer.shap_summary.head(10).to_dict() if trainer.shap_summary is not None else {}, "breaker_context": breaker_context}
        in_run_historical_cycles.append(cycle_results_for_ai)

        suggested_params = gemini_analyzer.analyze_cycle_and_suggest_changes(in_run_historical_cycles, chosen_strategy_features)

        if suggested_params:
            logger.info(f"  - AI suggests new parameters: {suggested_params}")
            current_config.update(suggested_params)
            if is_continuous:
                logger.info("  - Daemon Mode: Respecting API rate limits (5-minute delay between cycles)..."); time.sleep(300)
            else:
                logger.info("  - Single Run Mode: 5-second delay between cycles..."); time.sleep(5)

    logger.info("\n==========================================================")
    logger.info("                                WALK-FORWARD ANALYSIS COMPLETE");logger.info("==========================================================")

    reporter=PerformanceAnalyzer(initial_config)
    if all_shap: aggregated_shap=pd.concat(all_shap).groupby(level=0)['SHAP_Importance'].mean().sort_values(ascending=False).to_frame()
    else: aggregated_shap=None

    final_trades_df=pd.concat(all_trades,ignore_index=True) if all_trades else pd.DataFrame()
    final_metrics = reporter.generate_full_report(final_trades_df,pd.Series(full_equity_curve),cycle_metrics,aggregated_shap, framework_history)

    run_summary = {"script_version": initial_config.REPORT_LABEL, "strategy_name": initial_config.strategy_name, "timestamp": initial_config.run_timestamp, "final_metrics": gemini_analyzer._sanitize_dict(final_metrics), "initial_config": {k: v for k, v in initial_config.model_dump().items() if k in ['LOOKAHEAD_CANDLES', 'MAX_DD_PER_CYCLE', 'OPTUNA_TRIALS', 'selected_features']}, "top_5_features": aggregated_shap.head(5).to_dict().get('SHAP_Importance', {}) if aggregated_shap is not None else {}}
    save_run_to_memory(initial_config.MEMORY_FILE_PATH, run_summary, framework_history)

def main():
    # --- Daemon Mode Controls ---
    # DEFAULT BEHAVIOR: Run once with a 5-second delay between cycles.
    # To enable DAEMON mode, set either CONTINUOUS_RUN_HOURS > 0 or MAX_RUNS > 1.
    # In DAEMON mode:
    # - The delay between cycles is increased to 5 minutes to respect API rate limits.
    # - A 10-second interruptible countdown appears between each full run.
    # - The AI may attempt to synthesize new HYBRID strategies between runs.
    CONTINUOUS_RUN_HOURS = 0
    MAX_RUNS = 1 # Set > 1 to enable Daemon mode
    # --- End Controls ---

    fallback_config={
        "BASE_PATH": os.getcwd(), "REPORT_LABEL": "ML_Framework_V119_Hybrid_Synthesis",
        "strategy_name": "TrendFollower",
        "FORWARD_TEST_START_DATE": "2024-01-01", "INITIAL_CAPITAL": 100000.0,
        "CONFIDENCE_TIERS": {'ultra_high':{'min':0.8,'risk_mult':1.2,'rr':3.0},'high':{'min':0.7,'risk_mult':1.0,'rr':2.5},'standard':{'min':0.6,'risk_mult':0.8,'rr':2.0}},
        "BASE_RISK_PER_TRADE_PCT": 0.01, "SPREAD_PCTG_OF_ATR": 0.05, "SLIPPAGE_PCTG_OF_ATR": 0.02,
        "OPTUNA_TRIALS": 30, "TRAINING_WINDOW": '365D', "RETRAINING_FREQUENCY": '90D',
        "FORWARD_TEST_GAP": "1D", "LOOKAHEAD_CANDLES": 150, "TREND_FILTER_THRESHOLD": 25.0,
        "BOLLINGER_PERIOD": 20, "STOCHASTIC_PERIOD": 14, "CALCULATE_SHAP_VALUES": True, "MAX_DD_PER_CYCLE": 0.3,
        "selected_features": []
    }

    run_count = 0
    script_start_time = datetime.now()
    is_continuous = CONTINUOUS_RUN_HOURS > 0 or MAX_RUNS > 1

    # Initialize components that persist across the daemon loop
    playbook = initialize_playbook(fallback_config["BASE_PATH"])
    analyzer = GeminiAnalyzer() # A single analyzer instance for the whole session
    playbook_path = os.path.join(fallback_config["BASE_PATH"], "Results", "strategy_playbook.json")

    while True:
        run_count += 1
        if is_continuous:
            logger.info(f"\n{'='*30} STARTING DAEMON RUN {run_count} {'='*30}\n")
        else:
            logger.info(f"\n{'='*30} STARTING SINGLE RUN {'='*30}\n")

        framework_history = migrate_and_load_memory(os.path.join(fallback_config['BASE_PATH'], "Results", "framework_memory.json"))

        # --- HYBRID SYNTHESIS STEP ---
        if is_continuous: # Only try to create hybrids in daemon mode
            updated_playbook = check_and_create_hybrid(framework_history, playbook, analyzer, playbook_path)
            if updated_playbook:
                playbook = updated_playbook # Use the new playbook for the next run

        run_single_instance(fallback_config, framework_history, is_continuous, playbook)

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

# End_To_End_Advanced_ML_Trading_Framework_PRO_V119_Hybrid_Synthesis.py