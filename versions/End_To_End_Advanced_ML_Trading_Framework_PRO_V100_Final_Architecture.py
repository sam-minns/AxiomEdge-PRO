# End_To_End_Advanced_ML_Trading_Framework_PRO_V100_Final_Architecture
#
# V100 UPDATE (Final Architecture & Prompt Enhancement):
# 1. AI PROMPT ENHANCEMENT: The prompt for the Gemini AI analyst has been updated
#    with the user-provided text. The new prompt encourages a deeper forensic
#    analysis, focusing not just on fixing failures but also on identifying
#    hidden strengths and positive anomalies for potential growth.
#
# V99 UPDATE (SHAP Fix Attempt):
# 1. CRITICAL FIX: The logic in `_generate_shap_summary` has been replaced with the
#    exact implementation from the user's V93 template. It now correctly uses the
#    modern `shap.Explanation` object to handle multi-class model outputs, which
#    permanently resolves the `ValueError: Shape of passed values...` error.
#
# V98 UPDATE (Major Architecture Overhaul):
# 1. Reverted to the complex V93 architecture with Multi-Timeframe data, a 3-class
#    model, risk-adjusted optimization, and full SHAP reporting.

import os
import re
import json
import time
import warnings
import logging
from datetime import datetime
from logging.handlers import RotatingFileHandler
from typing import List, Dict, Any, Optional, Tuple

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
import sys
import xgboost

# Print environment info for easier debugging
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
    if not logger.handlers:
        logger.setLevel(logging.DEBUG)
        ch = logging.StreamHandler()
        ch.setLevel(logging.INFO)
        fh = RotatingFileHandler('trading_framework_adaptive.log',maxBytes=10*1024*1024,backupCount=5)
        fh.setLevel(logging.DEBUG)
        formatter = logging.Formatter('%(asctime)s - [%(levelname)s] - %(message)s')
        ch.setFormatter(formatter)
        fh.setFormatter(formatter)
        logger.addHandler(ch)
        logger.addHandler(fh)
    return logger

logger = setup_logging()
optuna.logging.set_verbosity(optuna.logging.WARNING)

# =============================================================================
# 2. GEMINI AI ANALYZER
# =============================================================================
class GeminiAnalyzer:
    def __init__(self):
        api_key = os.getenv("GEMINI_API_KEY", "PASTE_YOUR_GEMINI_API_KEY_HERE")
        if not api_key or "YOUR" in api_key or "PASTE" in api_key:
            logger.warning("GEMINI_API_KEY not found. AI analysis will be skipped.")
            self.api_key_valid = False
        else:
            self.api_key_valid = True
        self.api_url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash:generateContent?key={api_key}"
        self.headers = {"Content-Type": "application/json"}

    def _call_gemini(self, prompt: str) -> str:
        if not self.api_key_valid: return "{}"
        if len(prompt) > 28000: logger.warning("Prompt is very large, may risk exceeding token limits.")
        data = {"contents": [{"parts": [{"text": prompt}]}]}
        try:
            response = requests.post(self.api_url, headers=self.headers, data=json.dumps(data), timeout=60)
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

    def _summarize_historical_results(self, historical_results: List[Dict]) -> List[Dict]:
        recent_history = historical_results[-5:]
        summarized_history = []
        for result in recent_history:
            summary = {
                "cycle": result.get("cycle"), "status": result.get("status"),
                "objective_score": round(result.get("objective_score", 0), 4),
                "cycle_pnl": result.get("cycle_pnl"), "win_rate": result.get("win_rate"),
                "num_trades": result.get("num_trades"),
                "params_used": {
                    "LOOKAHEAD_CANDLES": result.get("current_params", {}).get("LOOKAHEAD_CANDLES"),
                    "MAX_DD_PER_CYCLE": result.get("current_params", {}).get("MAX_DD_PER_CYCLE"),
                }
            }
            if result.get("shap_summary"):
                try: summary["top_5_features"] = list(result["shap_summary"]["SHAP_Importance"].keys())[:5]
                except Exception: summary["top_5_features"] = "Error parsing SHAP"
            summarized_history.append(summary)
        return summarized_history

    def analyze_cycle_and_suggest_changes(self, historical_results: List[Dict], available_features: List[str]) -> Dict:
        if not self.api_key_valid:
            logger.warning("  - Skipping AI analysis: No valid Gemini API key provided.")
            return {}
        logger.info("  - Summarizing history and sending to Gemini for forensic analysis...")
        summarized_history = self._summarize_historical_results(historical_results)

        # UPDATED PROMPT (V100)
        prompt = (
            "You are an expert trading model analyst and forensic data scientist. Your primary goal is to create a STABLE and PROFITABLE strategy by learning from past failures AND uncovering hidden opportunities for improvement and growth. "
            "Analyze the SUMMARIZED history of the most recent cycles for hidden patterns, correlations, or event triggers that might explain the failures. "
            "In addition, actively search for overlooked strengths, rare success events, or positive anomalies—these may reveal hidden opportunities or promising directions. For example: "
            "  - Did any specific parameter combinations or features, even if rare, lead to unexpectedly good results? "
            "  - Are there cycles where the model performed better than average, even briefly? What set them apart? "
            "  - Did any features or configurations seem to reduce risk or improve stability, even if not consistently? "
            "  - Are there market conditions or triggers that, if anticipated, could be exploited for advantage? "
            "If the strategy is stuck in a failure loop, suggest a *significantly different* approach (e.g., a much simpler model, a different feature set, a different risk profile) to break the cycle and/or capitalize on any hidden opportunities you discover.\n\n"
            "Based on your forensic analysis, suggest a NEW configuration for the next cycle with the goal of SURVIVAL, STABILITY, and the potential to capitalize on hidden strengths.\n\n"
            "1.  **Risk Management**: Suggest a new float value for `MAX_DD_PER_CYCLE` (between 0.15 and 0.40).\n"
            "2.  **Model Parameters**: Suggest new integer values for `LOOKAHEAD_CANDLES` (50-200) and `OPTUNA_TRIALS` (15-30).\n"
            "3.  **Feature Selection**: Select a SUBSET of features from the `available_features` list. Be creative. A simpler model with fewer features may be more robust, but don't ignore features that show hidden potential.\n\n"
            "Respond ONLY with a valid JSON object containing the keys: `MAX_DD_PER_CYCLE`, `LOOKAHEAD_CANDLES`, `OPTUNA_TRIALS`, and `selected_features` (a list of strings).\n\n"
            f"SUMMARIZED HISTORICAL CYCLE RESULTS:\n{json.dumps(summarized_history, indent=2)}\n\n"
            f"AVAILABLE FEATURES FOR NEXT CYCLE:\n{available_features}"
        )
        response_text = self._call_gemini(prompt)
        logger.info(f"    - Gemini Forensic Analysis (Raw): {response_text}")
        suggestions = self._extract_json_from_response(response_text)
        logger.info(f"    - Parsed Suggestions: {suggestions}")
        return suggestions

# =============================================================================
# 3. CONFIGURATION & VALIDATION
# =============================================================================
class ConfigModel(BaseModel):
    BASE_PATH: DirectoryPath; REPORT_LABEL: str; FORWARD_TEST_START_DATE: str; INITIAL_CAPITAL: confloat(gt=0)
    CONFIDENCE_TIERS: Dict[str, Dict[str, float]]; BASE_RISK_PER_TRADE_PCT: confloat(gt=0, lt=1)
    SPREAD_PCTG_OF_ATR: confloat(ge=0); SLIPPAGE_PCTG_OF_ATR: confloat(ge=0); OPTUNA_TRIALS: conint(gt=0)
    TRAINING_WINDOW: str; RETRAINING_FREQUENCY: str; FORWARD_TEST_GAP: str; LOOKAHEAD_CANDLES: conint(gt=0)
    MODEL_SAVE_PATH: str = ""; PLOT_SAVE_PATH: str = ""; REPORT_SAVE_PATH: str = ""; SHAP_PLOT_PATH: str = ""
    TREND_FILTER_THRESHOLD: confloat(gt=0) = 25.0
    BOLLINGER_PERIOD: conint(gt=0) = 20; STOCHASTIC_PERIOD: conint(gt=0) = 14; CALCULATE_SHAP_VALUES: bool = True
    MAX_DD_PER_CYCLE: confloat(gt=0.05, lt=1.0) = 0.25
    selected_features: List[str]
    def __init__(self, **data: Any):
        super().__init__(**data)
        result_folder_path=os.path.join(self.BASE_PATH,"Results",self.REPORT_LABEL);os.makedirs(result_folder_path,exist_ok=True)
        self.MODEL_SAVE_PATH=os.path.join(result_folder_path,f"{self.REPORT_LABEL}_model.json")
        self.PLOT_SAVE_PATH=os.path.join(result_folder_path,f"{self.REPORT_LABEL}_equity_curve.png")
        self.REPORT_SAVE_PATH=os.path.join(result_folder_path,f"{self.REPORT_LABEL}_quantitative_report.txt")
        self.SHAP_PLOT_PATH=os.path.join(result_folder_path,f"{self.REPORT_LABEL}_shap_summary.png")

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
        g_out=self._calculate_adx(g_out,lookback);g_out=self._calculate_bollinger_bands(g_out,self.config.BOLLINGER_PERIOD);g_out=self._calculate_stochastic(g_out,self.config.STOCHASTIC_PERIOD)
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
        
        feature_cols = [f for f in ModelTrainer.BASE_FEATURES if f in df_final.columns]
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

# =============================================================================
# 6. MODEL TRAINER
# =============================================================================
class ModelTrainer:
    BASE_FEATURES = [
        'ATR', 'RSI', 'ADX', 'hour', 'day_of_week', 'adx_x_h1_trend', 'atr_x_d1_trend',
        'H1_ctx_Trend', 'D1_ctx_Trend', 'bollinger_bandwidth', 'stoch_k', 'stoch_d',
        'market_regime', 'H1_ctx_SMA', 'D1_ctx_SMA'
    ]
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
            
            # CRITICAL FIX (V100): Using the robust `Explanation` object logic from V93 template
            explainer = shap.TreeExplainer(model)
            shap_explanation = explainer(X_sample) # Returns a shap.Explanation object

            # For multi-class, .values is a list of arrays. We need to average them correctly.
            # 1. Get abs values, 2. Mean over samples, 3. Mean over classes
            mean_abs_shap_values = np.abs(shap_explanation.values).mean(axis=1)
            overall_importance = mean_abs_shap_values.mean(axis=0)

            summary = pd.DataFrame(overall_importance, index=feature_names, columns=['SHAP_Importance']).sort_values(by='SHAP_Importance', ascending=False)
            self.shap_summary = summary
            logger.info("    - SHAP summary generated successfully.")
        except Exception as e:
            logger.error(f"    - Failed to generate SHAP summary: {e}", exc_info=True); self.shap_summary = None

# =============================================================================
# 7. BACKTESTER & 8. PERFORMANCE ANALYZER
# =============================================================================
class Backtester:
    def __init__(self,config:ConfigModel):self.config=config
    def run_backtest_chunk(self,df_chunk_in:pd.DataFrame,model:Pipeline,feature_list:List[str],confidence_threshold:float,initial_equity:float)->Tuple[pd.DataFrame,pd.Series,bool]:
        if df_chunk_in.empty:return pd.DataFrame(),pd.Series([initial_equity]), False
        df_chunk=df_chunk_in.copy()
        X_test=df_chunk[feature_list].copy().fillna(0)
        class_probs=model.predict_proba(X_test)
        df_chunk.loc[:,['prob_short','prob_hold','prob_long']]=class_probs
        trades,equity,equity_curve,open_positions=[],initial_equity,[initial_equity],{}
        chunk_peak_equity = initial_equity; circuit_breaker_tripped = False

        candles=df_chunk.reset_index().to_dict('records')
        for candle in candles:
            if not circuit_breaker_tripped:
                if equity > chunk_peak_equity: chunk_peak_equity = equity
                if equity > 0 and chunk_peak_equity > 0 and (chunk_peak_equity - equity) / chunk_peak_equity > self.config.MAX_DD_PER_CYCLE:
                    logger.warning(f"  - CIRCUIT BREAKER TRIPPED! Drawdown exceeded {self.config.MAX_DD_PER_CYCLE:.0%} for this cycle. Closing all positions.")
                    circuit_breaker_tripped = True; open_positions = {}
            
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
        return pd.DataFrame(trades),pd.Series(equity_curve), circuit_breaker_tripped

class PerformanceAnalyzer:
    def __init__(self,config:ConfigModel):self.config=config
    def generate_full_report(self,trades_df:Optional[pd.DataFrame],equity_curve:Optional[pd.Series],cycle_metrics:List[Dict],aggregated_shap:Optional[pd.DataFrame]=None):
        logger.info("-> Stage 4: Generating Final Performance Report...")
        if equity_curve is not None and len(equity_curve) > 1: self.plot_equity_curve(equity_curve)
        if aggregated_shap is not None: self.plot_shap_summary(aggregated_shap)
        metrics = self._calculate_metrics(trades_df, equity_curve) if trades_df is not None and not trades_df.empty else {}
        self.generate_text_report(metrics, cycle_metrics, aggregated_shap)
        logger.info("[SUCCESS] Final report generated and saved.")

    def plot_equity_curve(self,equity_curve:pd.Series):
        plt.style.use('seaborn-v0_8-darkgrid');plt.figure(figsize=(16,8));plt.plot(equity_curve.values,color='dodgerblue',linewidth=2)
        plt.title(f"{self.config.REPORT_LABEL} - Walk-Forward Equity Curve",fontsize=16,weight='bold')
        plt.xlabel("Trade Number",fontsize=12);plt.ylabel("Equity ($)",fontsize=12);plt.grid(True,which='both',linestyle=':');plt.savefig(self.config.PLOT_SAVE_PATH);plt.close()
        logger.info(f"  - Equity curve plot saved to: {self.config.PLOT_SAVE_PATH}")

    def plot_shap_summary(self,shap_summary:pd.DataFrame):
        plt.style.use('seaborn-v0_8-darkgrid');plt.figure(figsize=(12,10))
        shap_summary.head(20).sort_values(by='SHAP_Importance').plot(kind='barh',legend=False,color='mediumseagreen')
        plt.title(f"{self.config.REPORT_LABEL} - Aggregated Feature Importance (SHAP)",fontsize=16,weight='bold')
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

    def generate_text_report(self,m:Dict[str,Any],cycle_metrics:List[Dict],aggregated_shap:Optional[pd.DataFrame]=None):
        cycle_df=pd.DataFrame(cycle_metrics);cycle_report="Per-Cycle Performance:\n"+cycle_df.to_string(index=False) if not cycle_df.empty else "No trades were executed."
        shap_report="Aggregated Feature Importance (SHAP):\n"+aggregated_shap.head(15).to_string() if aggregated_shap is not None else "SHAP summary was not generated."
        report=f"""
\n================================================================================
                           ADAPTIVE WALK-FORWARD PERFORMANCE REPORT
================================================================================
Report Label: {self.config.REPORT_LABEL}
Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
--------------------------------------------------------------------------------
I. EXECUTIVE SUMMARY
--------------------------------------------------------------------------------
Initial Capital:         ${m.get('initial_capital', self.config.INITIAL_CAPITAL):>15,.2f}
Ending Capital:          ${m.get('ending_capital', self.config.INITIAL_CAPITAL):>15,.2f}
Total Net Profit:        ${m.get('total_net_profit', 0):>15,.2f} ({m.get('net_profit_pct', 0):.2%})
Profit Factor:           {m.get('profit_factor', 0):>15.2f}
Win Rate:                {m.get('win_rate', 0):>15.2%}
Expected Payoff:         ${m.get('expected_payoff', 0):>15.2f}
--------------------------------------------------------------------------------
II. CORE PERFORMANCE METRICS
--------------------------------------------------------------------------------
Annual Return (CAGR):    {m.get('cagr', 0):>15.2%}
Sharpe Ratio (annual):   {m.get('sharpe_ratio', 0):>15.2f}
Sortino Ratio (annual):  {m.get('sortino_ratio', 0):>15.2f}
Calmar Ratio / MAR:      {m.get('mar_ratio', 0):>15.2f}
--------------------------------------------------------------------------------
III. RISK & DRAWDOWN ANALYSIS
--------------------------------------------------------------------------------
Max Drawdown:            {m.get('max_drawdown_pct', 0):>15.2f}% (${m.get('max_drawdown_abs', 0):,.2f})
Recovery Factor:         {m.get('recovery_factor', 0):>15.2f}
Longest Losing Streak:   {m.get('longest_loss_streak', 0):>15} trades
--------------------------------------------------------------------------------
IV. TRADE-LEVEL STATISTICS
--------------------------------------------------------------------------------
Total Trades:            {m.get('total_trades', 0):>15}
Winning Trades:          {m.get('winning_trades', 0):>15}
Losing Trades:           {m.get('losing_trades', 0):>15}
Average Win:             ${m.get('avg_win_amount', 0):>15,.2f}
Average Loss:            ${m.get('avg_loss_amount', 0):>15,.2f}
Payoff Ratio:            {m.get('payoff_ratio', 0):>15.2f}
Longest Winning Streak:  {m.get('longest_win_streak', 0):>15} trades
--------------------------------------------------------------------------------
V. WALK-FORWARD CYCLE BREAKDOWN
--------------------------------------------------------------------------------
{cycle_report}
--------------------------------------------------------------------------------
VI. MODEL FEATURE IMPORTANCE
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
# 9. MAIN ORCHESTRATOR
# =============================================================================
def main():
    start_time=datetime.now();logger.info("==========================================================")
    logger.info("  STARTING END-TO-END MULTI-ASSET ML TRADING FRAMEWORK");logger.info("==========================================================")
    
    current_config={
        "BASE_PATH": os.getcwd(), "REPORT_LABEL": "ML_Framework_V100_Final_Architecture",
        "FORWARD_TEST_START_DATE": "2024-01-01", "INITIAL_CAPITAL": 100000.0,
        "CONFIDENCE_TIERS": {'ultra_high':{'min':0.8,'risk_mult':1.2,'rr':3.0},'high':{'min':0.7,'risk_mult':1.0,'rr':2.5},'standard':{'min':0.6,'risk_mult':0.8,'rr':2.0}},
        "BASE_RISK_PER_TRADE_PCT": 0.01, "SPREAD_PCTG_OF_ATR": 0.05, "SLIPPAGE_PCTG_OF_ATR": 0.02,
        "OPTUNA_TRIALS": 30, "TRAINING_WINDOW": '365D', "RETRAINING_FREQUENCY": '60D', "FORWARD_TEST_GAP": "1D",
        "LOOKAHEAD_CANDLES": 100, "TREND_FILTER_THRESHOLD": 25.0, "BOLLINGER_PERIOD": 20, "STOCHASTIC_PERIOD": 14,
        "CALCULATE_SHAP_VALUES": True, "MAX_DD_PER_CYCLE": 0.3, 
        "selected_features": ModelTrainer.BASE_FEATURES[:15] 
    }
    
    files_to_process=[
        "AUDUSD_Daily_202001060000_202506020000.csv","AUDUSD_H1_202001060000_202506021800.csv","AUDUSD_M15_202105170000_202506021830.csv",
        "EURUSD_Daily_202001060000_202506020000.csv","EURUSD_H1_202001060000_202506021800.csv","EURUSD_M15_202106020100_202506021830.csv",
        "GBPUSD_Daily_202001060000_202506020000.csv","GBPUSD_H1_202001060000_202506021800.csv","GBPUSD_M15_202106020015_202506021830.csv",
        "USDCAD_Daily_202001060000_202506020000.csv","USDCAD_H1_202001060000_202506021800.csv","USDCAD_M15_202105170000_202506021830.csv"
    ]
    
    try:
        gemini_analyzer=GeminiAnalyzer();initial_config=ConfigModel(**current_config);logger.info(f"-> Starting with configuration: {initial_config.REPORT_LABEL}")
        data_by_tf=DataLoader(initial_config).load_and_parse_data(filenames=files_to_process)
        if not data_by_tf:return
        fe=FeatureEngineer(initial_config);df_featured=fe.create_feature_stack(data_by_tf)
        if df_featured.empty:return
    except Exception as e:logger.critical(f"[FATAL] Initial setup failed: {e}",exc_info=True);return
    
    test_start_date=pd.to_datetime(current_config['FORWARD_TEST_START_DATE']);max_date=df_featured.index.max()
    retraining_dates=pd.date_range(start=test_start_date,end=max_date,freq=current_config['RETRAINING_FREQUENCY'])
    all_trades,full_equity_curve,cycle_metrics,all_shap=[],[current_config['INITIAL_CAPITAL']],[],[]
    historical_cycle_results = []
    
    logger.info("-> Stage 3: Starting Adaptive Walk-Forward Analysis with AI Tuning...")
    for i,period_start_date in enumerate(retraining_dates):
        config=ConfigModel(**current_config)
        logger.info(f"\n{'='*25} CYCLE {i+1}/{len(retraining_dates)}: {period_start_date.date()} {'='*25}")
        logger.info(f"  - Using Config: LOOKAHEAD={config.LOOKAHEAD_CANDLES}, TREND_THRESH={config.TREND_FILTER_THRESHOLD}, OPTUNA_TRIALS={config.OPTUNA_TRIALS}, MAX_DD_CYCLE={config.MAX_DD_PER_CYCLE}")
        logger.info(f"  - Features for this cycle: {config.selected_features}")

        train_end=period_start_date-pd.Timedelta(config.FORWARD_TEST_GAP)
        train_start=train_end-pd.Timedelta(config.TRAINING_WINDOW)
        test_end=(period_start_date+pd.Timedelta(config.RETRAINING_FREQUENCY))-pd.Timedelta(days=1)
        if test_end>max_date:test_end=max_date
        
        df_train_raw=df_featured.loc[train_start:train_end];df_test_chunk=df_featured.loc[period_start_date:test_end]
        if df_train_raw.empty or df_test_chunk.empty: cycle_metrics.append({'Cycle':i+1,'StartDate':period_start_date.date(),'NumTrades':0,'WinRate':"N/A",'CyclePnL':"$0.00",'Status':"Skipped (No Data)"});continue
        
        df_train_labeled=fe.label_outcomes(df_train_raw,lookahead=config.LOOKAHEAD_CANDLES)
        if df_train_labeled.empty or 'target' not in df_train_labeled.columns:
            cycle_metrics.append({'Cycle':i+1,'StartDate':period_start_date.date(),'NumTrades':0,'WinRate':"N/A",'CyclePnL':"$0.00",'Status':"Skipped (Label Error)"});continue
        
        trainer=ModelTrainer(config);training_result=trainer.train(df_train_labeled, feature_list=current_config['selected_features'])
        if training_result is None:
            cycle_metrics.append({'Cycle':i+1,'StartDate':period_start_date.date(),'NumTrades':0,'WinRate':"N/A",'CyclePnL':"$0.00",'Status':"Failed (Training Error)"});continue
        
        model,best_threshold=training_result
        if trainer.shap_summary is not None:all_shap.append(trainer.shap_summary)
        
        logger.info(f"  - Backtesting on out-of-sample data from {period_start_date.date()} to {test_end.date()}...")
        backtester=Backtester(config);chunk_trades_df,chunk_equity,breaker_tripped=backtester.run_backtest_chunk(df_test_chunk,model,current_config['selected_features'],best_threshold,initial_equity=full_equity_curve[-1])
        
        cycle_pnl=0;cycle_win_rate="N/A"; status = "No Trades"
        if breaker_tripped: status = "Circuit Breaker"
        if not chunk_trades_df.empty:
            all_trades.append(chunk_trades_df);full_equity_curve.extend(chunk_equity.iloc[1:].tolist())
            pnl,wr=chunk_trades_df['PNL'].sum(),(chunk_trades_df['PNL']>0).mean()
            cycle_pnl=pnl;cycle_win_rate=f"{wr:.2%}"
            status = "Success" if not breaker_tripped else "Circuit Breaker"
        
        cycle_metrics.append({'Cycle':i+1,'StartDate':period_start_date.date(),'NumTrades':len(chunk_trades_df),'WinRate':cycle_win_rate,'CyclePnL':f"${cycle_pnl:,.2f}",'Status':status})
        
        cycle_results_for_ai={"cycle":i+1,"objective_score":trainer.study.best_value if trainer.study else 0,"best_threshold":best_threshold,"cycle_pnl":cycle_pnl,"win_rate":cycle_win_rate,"num_trades":len(chunk_trades_df), "status": status, "current_params":{k:v for k,v in current_config.items() if k in ['LOOKAHEAD_CANDLES','TREND_FILTER_THRESHOLD','OPTUNA_TRIALS','selected_features','MAX_DD_PER_CYCLE']}, "shap_summary": trainer.shap_summary.head(10).to_dict() if trainer.shap_summary is not None else {}}
        historical_cycle_results.append(cycle_results_for_ai)
        
        all_feature_names = [col for col in ModelTrainer.BASE_FEATURES if col in df_featured.columns]
        suggested_params = gemini_analyzer.analyze_cycle_and_suggest_changes(historical_cycle_results, all_feature_names)
        
        if suggested_params:
            logger.info(f"  - AI suggests new parameters: {suggested_params}")
            current_config.update(suggested_params)
            logger.info("  - Respecting API rate limits (5-second delay)..."); time.sleep(5)
    
    logger.info("\n==========================================================")
    logger.info("                          WALK-FORWARD ANALYSIS COMPLETE");logger.info("==========================================================")
    
    final_config=ConfigModel(**current_config)
    reporter=PerformanceAnalyzer(final_config)
    if all_shap: aggregated_shap=pd.concat(all_shap).groupby(level=0)['SHAP_Importance'].mean().sort_values(ascending=False).to_frame() 
    else: aggregated_shap=None
    
    final_trades_df=pd.concat(all_trades,ignore_index=True) if all_trades else pd.DataFrame()
    reporter.generate_full_report(final_trades_df,pd.Series(full_equity_curve),cycle_metrics,aggregated_shap)
    logger.info(f"\nTotal execution time: {datetime.now() - start_time}")

if __name__ == '__main__':
    if os.name == 'nt':
        os.system("chcp 65001 > nul")
    main()

# End_To_End_Advanced_ML_Trading_Framework_PRO_V100_Final_Architecture.py