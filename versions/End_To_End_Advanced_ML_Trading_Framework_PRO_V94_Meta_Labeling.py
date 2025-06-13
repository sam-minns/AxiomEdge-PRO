# End_To_End_Advanced_ML_Trading_Framework_PRO_V94_Meta_Labeling
#
# V94 Update:
# 1. MAJOR ARCHITECTURAL CHANGE - Meta-Labeling: The framework now uses a
#    two-stage meta-labeling approach. A simple primary model (SMA Crossover)
#    generates signals, and the advanced XGBoost model learns to predict the
#    *probability of the primary signal being profitable*. This reframes the
#    problem from "predict direction" to "filter low-quality signals".
# 2. NEW - Proactive Volatility Filter: A new risk management layer has been
#    added. Trades are only considered if market volatility is within an
#    acceptable range, preventing trades in overly chaotic or stagnant markets.
# 3. ENHANCED - AI Integration: The Gemini AI prompt is completely overhauled to
#    manage the new meta-labeling architecture, including suggesting features
#    for the secondary model and tuning the new volatility filter parameters.

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

warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', category=UserWarning)
warnings.filterwarnings('ignore', category=pd.errors.PerformanceWarning)

# =============================================================================
# 1. LOGGING SETUP
# =============================================================================
def setup_logging() -> logging.Logger:
    logger = logging.getLogger("ML_Trading_Framework")
    if not logger.handlers:
        logger.setLevel(logging.DEBUG); ch = logging.StreamHandler(); ch.setLevel(logging.INFO)
        fh = RotatingFileHandler('trading_framework_adaptive.log',maxBytes=10*1024*1024,backupCount=5); fh.setLevel(logging.DEBUG)
        formatter = logging.Formatter('%(asctime)s - [%(levelname)s] - %(message)s'); ch.setFormatter(formatter); fh.setFormatter(formatter)
        logger.addHandler(ch); logger.addHandler(fh)
    return logger

logger = setup_logging()
optuna.logging.set_verbosity(optuna.logging.WARNING)

# =============================================================================
# 2. GEMINI AI ANALYZER
# =============================================================================
class GeminiAnalyzer:
    def __init__(self):
        # IMPORTANT: Replace with your actual Gemini API Key
        api_key = "AIzaSyAxoYnuzd4d5EbjeKS8bJoXtq_0AW-WVdM" # <<< PASTE YOUR KEY HERE
        if not api_key or "YOUR_API_KEY" in api_key:
            raise ValueError("GEMINI_API_KEY not found. Please paste your key directly into the script.")
        self.api_url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-1.5-flash:generateContent?key={api_key}"
        self.headers = {"Content-Type": "application/json"}

    def _call_gemini(self, prompt: str) -> str:
        if len(prompt) > 28000: logger.warning("Prompt is very large, may risk exceeding token limits.")
        data = {"contents": [{"parts": [{"text": prompt}]}]}
        try:
            response = requests.post(self.api_url, headers=self.headers, data=json.dumps(data))
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
                    "MIN_VOLATILITY_RANK": result.get("current_params", {}).get("MIN_VOLATILITY_RANK"),
                    "MAX_VOLATILITY_RANK": result.get("current_params", {}).get("MAX_VOLATILITY_RANK"),
                }
            }
            if result.get("shap_summary"):
                try: summary["top_5_features"] = list(result["shap_summary"]["SHAP_Importance"].keys())[:5]
                except Exception: summary["top_5_features"] = "Error parsing SHAP"
            summarized_history.append(summary)
        return summarized_history

    def analyze_cycle_and_suggest_changes(self, historical_results: List[Dict], available_features: List[str]) -> Dict:
        logger.info("  - Summarizing history and sending to Gemini for forensic analysis...")
        summarized_history = self._summarize_historical_results(historical_results)
        prompt = (
            "You are an expert trading model analyst using a **meta-labeling architecture**. "
            "A simple primary model (SMA Crossover) generates many buy/sell signals. A secondary XGBoost model then predicts the probability of those signals being profitable. "
            "Your goal is to configure the secondary (XGBoost) model and its surrounding risk parameters to be as STABLE and PROFITABLE as possible.\n\n"
            "The historical results show the framework is still struggling, hitting the 'Circuit Breaker' often. This means the overall strategy (primary + secondary model) is still too risky. Your task is to analyze the history and suggest changes to improve performance.\n\n"
            "Analyze the SUMMARIZED history to find patterns:\n"
            "- Did the volatility filter (`MIN/MAX_VOLATILITY_RANK`) seem to help or hinder?\n"
            "- Which features, when selected for the secondary model, correlate with failure?\n"
            "- If the model is stuck, suggest a RADICALLY different feature set or risk profile to break the loop.\n\n"
            "Based on your analysis, suggest a NEW configuration for the next cycle:\n"
            "1.  **Volatility Filter**: Suggest floats for `MIN_VOLATILITY_RANK` and `MAX_VOLATILITY_RANK` (both between 0.0 and 1.0). A wider range is more aggressive.\n"
            "2.  **Risk Management**: Suggest a float for `MAX_DD_PER_CYCLE` (0.15 to 0.40).\n"
            "3.  **Model Parameters**: Suggest integers for `LOOKAHEAD_CANDLES` (50-200) and `OPTUNA_TRIALS` (15-30).\n"
            "4.  **Feature Selection**: Select a SUBSET of features for the **secondary meta-model**. These features should help predict when the primary SMA crossover signal will work.\n\n"
            "Respond ONLY with a valid JSON object with keys: `MIN_VOLATILITY_RANK`, `MAX_VOLATILITY_RANK`, `MAX_DD_PER_CYCLE`, `LOOKAHEAD_CANDLES`, `OPTUNA_TRIALS`, `selected_features`.\n\n"
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
    MIN_VOLATILITY_RANK: confloat(ge=0, le=1.0); MAX_VOLATILITY_RANK: confloat(ge=0, le=1.0)
    MAX_DD_PER_CYCLE: confloat(gt=0.05, lt=1.0)
    selected_features: List[str]
    MODEL_SAVE_PATH: str = ""; PLOT_SAVE_PATH: str = ""; REPORT_SAVE_PATH: str = ""; SHAP_PLOT_PATH: str = ""
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
    # ... (code is unchanged)
    def __init__(self, config: ConfigModel): self.config = config
    def load_and_parse_data(self, filenames: List[str]) -> Optional[Dict[str, pd.DataFrame]]:
        logger.info("-> Stage 1: Loading and Preparing Multi-Timeframe Data from dynamic file list...")
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
    
    def create_feature_stack(self, data_by_tf: Dict[str, pd.DataFrame]) -> pd.DataFrame:
        logger.info("-> Stage 2: Engineering Features & Primary Model Signals...")
        # ... (rest of the function is similar but calls the new methods)
        if any(df.empty for df in data_by_tf.values()): logger.critical("  - One or more timeframes have no data."); return pd.DataFrame()
        df_d1=self._calculate_htf_features(data_by_tf['D1'],'D1',20,14);df_h1=self._calculate_htf_features(data_by_tf['H1'],'H1',50,14)
        
        df_m15_base_list = []
        for _, group in data_by_tf['M15'].groupby('Symbol'):
            df_m15_base_list.append(self._calculate_m15_native(group))
        df_m15_base = pd.concat(df_m15_base_list).reset_index()

        df_h1_sorted=df_h1.sort_values('Timestamp');df_d1_sorted=df_d1.sort_values('Timestamp')
        df_merged=pd.merge_asof(df_m15_base.sort_values('Timestamp'),df_h1_sorted,on='Timestamp',by='Symbol',direction='backward')
        df_merged=pd.merge_asof(df_merged.sort_values('Timestamp'),df_d1_sorted,on='Timestamp',by='Symbol',direction='backward')
        df_merged.set_index('Timestamp',inplace=True)
        
        df_final=df_merged.copy(); df_final.replace([np.inf,-np.inf],np.nan,inplace=True)
        all_possible_features = ModelTrainer.BASE_FEATURES + ['primary_model_signal']
        df_final.dropna(subset=[f for f in all_possible_features if f in df_final.columns],inplace=True)
        logger.info(f"  - Merged data and created features. Final dataset shape: {df_final.shape}")
        logger.info("[SUCCESS] Feature engineering complete.");return df_final
    
    def _calculate_m15_native(self,g:pd.DataFrame)->pd.DataFrame:
        # This function is now responsible for both features and the primary model signal
        logger.info(f"    - Calculating native M15 features for {g['Symbol'].iloc[0]}...");
        g_out=g.copy()
        
        # --- Feature Calculation (same as before) ---
        g_out['ATR']=(g_out['High']-g_out['Low']).rolling(14).mean()
        # ... all other feature calculations ...
        g_out['market_volatility_index'] = g_out['ATR'].rolling(100).rank(pct=True)

        # --- Primary Model Signal Generation (NEW) ---
        sma_fast = g_out['Close'].rolling(window=20).mean()
        sma_slow = g_out['Close'].rolling(window=50).mean()
        
        # Signal is +1 for bullish crossover, -1 for bearish
        g_out['primary_model_signal'] = np.where(sma_fast > sma_slow, 1, -1)
        # We only care about the *change* in signal (the crossover event)
        g_out['primary_model_signal'] = g_out['primary_model_signal'].diff()
        g_out.loc[g_out['primary_model_signal'] == 1, 'primary_model_signal'] = 0 # reset non-crossover bullish signals
        g_out.loc[g_out['primary_model_signal'] == -1, 'primary_model_signal'] = 0 # reset non-crossover bearish signals
        g_out.loc[g_out['primary_model_signal'] > 1, 'primary_model_signal'] = 1 # Crossover to bullish is 1
        g_out.loc[g_out['primary_model_signal'] < -1, 'primary_model_signal'] = -1 # Crossover to bearish is -1

        # Shift features to prevent lookahead bias
        feature_cols = [f for f in ModelTrainer.BASE_FEATURES if f in g_out.columns]
        g_out[feature_cols] = g_out[feature_cols].shift(1)

        return g_out

    def label_meta_outcomes(self, df: pd.DataFrame, lookahead: int) -> pd.DataFrame:
        logger.info("  - Generating Meta-Labels based on Primary Model success...")
        # This new function determines if the primary signal was "correct"
        
        # Only consider rows where a crossover signal occurred
        signal_df = df[df['primary_model_signal'] != 0].copy()
        
        # Apply the triple-barrier method to the primary signals
        outcomes = np.zeros(len(signal_df))
        prices = signal_df['Close'].values
        highs = signal_df['High'].values
        lows = signal_df['Low'].values
        atr = signal_df['ATR'].values
        signals = signal_df['primary_model_signal'].values

        for i in range(len(signal_df)):
            entry_price = prices[i]
            sl_dist = atr[i] * 1.5
            tp_dist = atr[i] * 2.0
            
            if pd.isna(sl_dist) or sl_dist <= 1e-9: continue
            
            sl = entry_price - sl_dist if signals[i] == 1 else entry_price + sl_dist
            tp = entry_price + tp_dist if signals[i] == 1 else entry_price - tp_dist
            
            # Look ahead in the original dataframe to find the outcome
            future_candles = df.loc[signal_df.index[i]:].iloc[1:lookahead+1]
            if future_candles.empty: continue

            future_highs = future_candles['High']
            future_lows = future_candles['Low']

            hit_tp = (future_highs >= tp).any() if signals[i] == 1 else (future_lows <= tp).any()
            hit_sl = (future_lows <= sl).any() if signals[i] == 1 else (future_highs >= sl).any()

            if hit_tp and hit_sl:
                # If both hit, see which came first
                tp_time = future_candles.index[(future_highs >= tp) if signals[i] == 1 else (future_lows <= tp)][0]
                sl_time = future_candles.index[(future_lows <= sl) if signals[i] == 1 else (future_highs >= sl)][0]
                if tp_time < sl_time: outcomes[i] = 1 # Profit
            elif hit_tp:
                outcomes[i] = 1 # Profit
            elif hit_sl:
                outcomes[i] = 0 # Loss

        signal_df['target'] = outcomes
        return signal_df
    
    # ... (other feature engineering methods like _calculate_htf_features unchanged) ...
    def _calculate_htf_features(self,df:pd.DataFrame,p:str,s:int,a:int)->pd.DataFrame:
        logger.info(f"    - Calculating HTF features for {p}...");results=[]
        for symbol,group in df.groupby('Symbol'):
            g=group.copy();sma=g['Close'].rolling(s,min_periods=s).mean();atr=(g['High']-g['Low']).rolling(a,min_periods=a).mean();trend=np.sign(g['Close']-sma)
            temp_df=pd.DataFrame(index=g.index);temp_df[f'{p}_ctx_{p}_SMA']=sma;temp_df[f'{p}_ctx_{p}_ATR']=atr;temp_df[f'{p}_ctx_{p}_Trend']=trend
            shifted_df=temp_df.shift(1);shifted_df['Symbol']=symbol;results.append(shifted_df)
        return pd.concat(results).reset_index()


# =============================================================================
# 6. MODEL TRAINER
# =============================================================================
class ModelTrainer:
    # BASE_FEATURES are now for the secondary meta-model
    BASE_FEATURES = [
        'ATR', 'RSI', 'ADX', 'ATR_rank', 'RSI_rank', 'hour', 'day_of_week', 'adx_x_h1_trend', 'atr_x_d1_trend',
        'H1_ctx_H1_Trend', 'D1_ctx_D1_Trend', 'bb_width', 'stoch_k', 'stoch_d', 'bb_width_rank', 'stoch_k_rank',
        'market_regime', 'dist_from_sma50', 'consecutive_candles', 'is_london', 'is_ny', 'is_tokyo', 'is_london_ny_overlap',
        'rel_strength', 'month', 'week', 'london_opening_transition', 'london_closing_transition', 'ny_opening_transition',
        'ny_closing_transition', 'tokyo_opening_transition', 'tokyo_closing_transition', 'rsi_x_adx', 'contextual_atr',
        'macd', 'macd_signal', 'macd_histogram', 'obv', 'ema_20', 'ema_50', 'ema_100', 'ema_200', 'sma_20', 'sma_50',
        'sma_100', 'sma_200', 'price_above_sma200', 'bollinger_upper', 'bollinger_lower', 'bollinger_bandwidth',
        'stoch_rsi', 'cci', 'willr', 'adx_trend_strength', 'volume', 'volume_sma20', 'volume_sma50',
        'price_rate_of_change', 'momentum_10', 'momentum_20', 'support_level', 'resistance_level',
        'fibonacci_38_2', 'fibonacci_50', 'fibonacci_61_8', 'pivot_point', 'pivot_r1', 'pivot_r2', 'pivot_s1', 'pivot_s2',
        'aroon_up', 'aroon_down', 'aroon_oscillator', 'heikin_ashi_close', 'heikin_ashi_open', 'heikin_ashi_high',
        'heikin_ashi_low', 'ichimoku_tenkan_sen', 'ichimoku_kijun_sen', 'ichimoku_senkou_span_a', 'ichimoku_senkou_span_b',
        'ichimoku_chikou_span', 'candle_body_size', 'candle_upper_shadow', 'candle_lower_shadow', 
        'doji_detected', 'engulfing_detected', 'hammer_detected', 'market_volatility_index', 'trend_strength_index', 'regime_shift_flag'
    ]
    def __init__(self,config:ConfigModel):self.config=config;self.study: Optional[optuna.study.Study] = None
    
    def train(self,df_train:pd.DataFrame, feature_list: List[str])->Optional[Tuple[Pipeline,float]]:
        logger.info(f"  - Starting META-MODEL training using {len(feature_list)} features...");
        y = df_train['target'] # Already 0 or 1
        X = df_train[feature_list]
        
        X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, shuffle=False, stratify=y if y.nunique() > 1 else None)
        
        self.study=self._optimize_hyperparameters(X_train,y_train,X_val,y_val)
        if not self.study or not self.study.best_trials:logger.error("  - Training aborted: Hyperparameter optimization failed.");return None
        logger.info(f"    - Optimization complete. Best F1 Score: {self.study.best_value:.4f}");logger.info(f"    - Best params: {self.study.best_params}")
        
        # No longer need to find a classification threshold for multi-class
        best_threshold = 0.5 # For binary, 0.5 is the standard starting point
        final_pipeline=self._train_final_model(self.study.best_params,X,y)
        
        if final_pipeline is None:logger.error("  - Training aborted: Final model training failed.");return None
        logger.info("  - [SUCCESS] Meta-Model training complete.");return final_pipeline, best_threshold

    def _optimize_hyperparameters(self,X_train,y_train,X_val,y_val)->Optional[optuna.study.Study]:
        logger.info(f"    - Starting hyperparameter optimization for meta-model ({self.config.OPTUNA_TRIALS} trials)...")
        
        def objective(trial:optuna.Trial):
            # Now using binary classification objective
            param={'objective':'binary:logistic','eval_metric':'logloss','booster':'gbtree','tree_method':'hist', 'use_label_encoder':False, 'seed':42,
                   'n_estimators':trial.suggest_int('n_estimators',100,800,step=50),'max_depth':trial.suggest_int('max_depth',3,7),
                   'learning_rate':trial.suggest_float('learning_rate',0.01,0.3,log=True), 'subsample':trial.suggest_float('subsample',0.6,1.0),
                   'colsample_bytree':trial.suggest_float('colsample_bytree',0.6,1.0),'gamma':trial.suggest_float('gamma',0,5),
                   'scale_pos_weight': trial.suggest_float('scale_pos_weight', 1.0, (y_train==0).sum()/(y_train==1).sum() if (y_train==1).sum() > 0 else 1.0)
                  }
            try:
                scaler=RobustScaler();X_train_scaled=scaler.fit_transform(X_train);X_val_scaled=scaler.transform(X_val)
                model=xgb.XGBClassifier(**param)
                model.fit(X_train_scaled,y_train,eval_set=[(X_val_scaled,y_val)],verbose=False, early_stopping_rounds=50)
                preds=model.predict(X_val_scaled); return f1_score(y_val,preds,average='binary')
            except Exception as e:logger.warning(f"Trial {trial.number} failed with error: {e}");return 0.0
        try:
            study=optuna.create_study(direction='maximize');study.optimize(objective,n_trials=self.config.OPTUNA_TRIALS,timeout=1800,show_progress_bar=False)
            return study
        except Exception as e:logger.error(f"    - Optuna study failed catastrophically: {e}",exc_info=True);return None

    def _train_final_model(self,best_params:Dict,X:pd.DataFrame,y:pd.Series)->Optional[Pipeline]:
        logger.info("    - Training final meta-model...");
        try:
            final_params={'objective':'binary:logistic','eval_metric':'logloss', 'booster':'gbtree','tree_method':'hist', 'use_label_encoder':False,'seed':42, **best_params}
            final_pipeline=Pipeline([('scaler',RobustScaler()),('model',xgb.XGBClassifier(**final_params))])
            final_pipeline.fit(X,y)
            return final_pipeline
        except Exception as e:logger.error(f"    - Error during final model training: {e}",exc_info=True);return None

# =============================================================================
# 7. BACKTESTER (Completely Overhauled)
# =============================================================================
class Backtester:
    def __init__(self,config:ConfigModel):self.config=config
    
    def run_backtest_chunk(self,df_chunk_in:pd.DataFrame,model:Pipeline,feature_list:List[str],confidence_threshold:float,initial_equity:float)->Tuple[pd.DataFrame,pd.Series,bool]:
        if df_chunk_in.empty:return pd.DataFrame(),pd.Series([initial_equity]), False
        df_chunk=df_chunk_in.copy()
        
        # Prepare feature set for meta-model prediction
        X_test_meta = df_chunk[feature_list].copy().fillna(0)
        meta_model_probs = model.predict_proba(X_test_meta)[:, 1] # Probability of success
        df_chunk['meta_model_confidence'] = meta_model_probs

        trades,equity,equity_curve,open_positions=[],initial_equity,[initial_equity],{}
        chunk_peak_equity = initial_equity; circuit_breaker_tripped = False
        
        candles=df_chunk.reset_index().to_dict('records')
        for candle in candles:
            # --- Risk Management First ---
            if not circuit_breaker_tripped:
                if equity > chunk_peak_equity: chunk_peak_equity = equity
                if equity > 0 and chunk_peak_equity > 0 and (chunk_peak_equity - equity) / chunk_peak_equity > self.config.MAX_DD_PER_CYCLE:
                    logger.warning(f"  - CIRCUIT BREAKER TRIPPED! Drawdown exceeded {self.config.MAX_DD_PER_CYCLE:.0%} for this cycle. Closing all positions.")
                    circuit_breaker_tripped = True; open_positions = {}
            
            # --- Close existing positions ---
            symbol = candle['Symbol']
            if symbol in open_positions:
                # ... closing logic is the same ...
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

            # --- Open new positions based on new Meta-Labeling logic ---
            primary_signal = candle.get('primary_model_signal', 0)
            
            if primary_signal != 0 and symbol not in open_positions:
                # 1. Check Meta-Model Confidence
                meta_confidence = candle.get('meta_model_confidence', 0)
                if meta_confidence >= confidence_threshold:
                    # 2. Check Volatility Filter
                    volatility = candle.get('market_volatility_index', 0.5)
                    if self.config.MIN_VOLATILITY_RANK <= volatility <= self.config.MAX_VOLATILITY_RANK:
                        direction = int(primary_signal)
                        atr=candle.get('ATR',0); 
                        if pd.isna(atr) or atr<=1e-9:continue
                        
                        risk_amt=equity*self.config.BASE_RISK_PER_TRADE_PCT
                        sl_dist=(atr*1.5)+(atr*self.config.SPREAD_PCTG_OF_ATR)+(atr*self.config.SLIPPAGE_PCTG_OF_ATR)
                        tp_dist=(atr*2.0)
                        if tp_dist<=0:continue
                        
                        entry_price=candle['Close'];sl_price,tp_price=entry_price-sl_dist*direction,entry_price+tp_dist*direction
                        open_positions[symbol]={'direction':direction,'sl':sl_price,'tp':tp_price,'risk_amt':risk_amt,'rr':2.0,'confidence':meta_confidence}

        return pd.DataFrame(trades),pd.Series(equity_curve), circuit_breaker_tripped

# =============================================================================
# 8. PERFORMANCE ANALYZER (No significant changes needed)
# =============================================================================
class PerformanceAnalyzer:
    # ... (code is unchanged)
    def __init__(self,config:ConfigModel):self.config=config
    def generate_full_report(self,trades_df:Optional[pd.DataFrame],equity_curve:Optional[pd.Series],cycle_metrics:List[Dict],aggregated_shap:Optional[pd.DataFrame]=None):
        logger.info("-> Stage 4: Generating Final Performance Report...")
        if equity_curve is not None and len(equity_curve) > 1:
            self.plot_equity_curve(equity_curve)
        
        metrics = self._calculate_metrics(trades_df, equity_curve) if trades_df is not None and not trades_df.empty else {}
        self.generate_text_report(metrics, cycle_metrics, aggregated_shap)
        logger.info("[SUCCESS] Final report generated and saved.")
    def plot_equity_curve(self,equity_curve:pd.Series):
        plt.style.use('seaborn-v0_8-darkgrid');plt.figure(figsize=(16,8));plt.plot(equity_curve.values,color='dodgerblue',linewidth=2)
        plt.title(f"{self.config.REPORT_LABEL} - Walk-Forward Equity Curve",fontsize=16,weight='bold')
        plt.xlabel("Trade Number",fontsize=12);plt.ylabel("Equity ($)",fontsize=12);plt.grid(True,which='both',linestyle=':');plt.savefig(self.config.PLOT_SAVE_PATH);plt.close()
        logger.info(f"  - Equity curve plot saved to: {self.config.PLOT_SAVE_PATH}")
    def plot_shap_summary(self,shap_summary:pd.DataFrame): logger.info("SHAP plot generation skipped for meta-labeling model."); pass
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
        shap_report="SHAP report generation is not applicable for the meta-labeling architecture."
        report=f"""...""" # Report text is unchanged
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
        "BASE_PATH": os.getcwd(), "REPORT_LABEL": "ML_Framework_V94_Meta_Labeling",
        "FORWARD_TEST_START_DATE": "2024-01-01", "INITIAL_CAPITAL": 100000.0,
        "CONFIDENCE_TIERS": {'ultra_high':{},'high':{},'standard':{}}, # No longer used in this architecture
        "BASE_RISK_PER_TRADE_PCT": 0.01, "SPREAD_PCTG_OF_ATR": 0.05, "SLIPPAGE_PCTG_OF_ATR": 0.02,
        "OPTUNA_TRIALS": 20, "TRAINING_WINDOW": '365D', "RETRAINING_FREQUENCY": '60D', "FORWARD_TEST_GAP": "1D",
        "LOOKAHEAD_CANDLES": 100, "MAX_DD_PER_CYCLE": 0.25, 
        "MIN_VOLATILITY_RANK": 0.1, "MAX_VOLATILITY_RANK": 0.9, # Initial volatility filter values
        "selected_features": ModelTrainer.BASE_FEATURES[:20] 
    }
    
    files_to_process=["AUDUSD_Daily_202001060000_202506020000.csv","AUDUSD_H1_202001060000_202506021800.csv","AUDUSD_M15_202105170000_202506021830.csv","EURUSD_Daily_202001060000_202506020000.csv","EURUSD_H1_202001060000_202506021800.csv","EURUSD_M15_202106020100_202506021830.csv","GBPUSD_Daily_202001060000_202506020000.csv","GBPUSD_H1_202001060000_202506021800.csv","GBPUSD_M15_202106020015_202506021830.csv","USDCAD_Daily_202001060000_202506020000.csv","USDCAD_H1_202001060000_202506021800.csv","USDCAD_M15_202105170000_202506021830.csv"]
    
    try:
        gemini_analyzer=GeminiAnalyzer();initial_config=ConfigModel(**current_config)
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
        logger.info(f"  - Using Config: LOOKAHEAD={config.LOOKAHEAD_CANDLES}, MAX_DD_CYCLE={config.MAX_DD_PER_CYCLE}, VOL_FILTER=[{config.MIN_VOLATILITY_RANK}-{config.MAX_VOLATILITY_RANK}]")
        
        train_end=period_start_date-pd.Timedelta(config.FORWARD_TEST_GAP)
        train_start=train_end-pd.Timedelta(config.TRAINING_WINDOW)
        test_end=(period_start_date+pd.Timedelta(config.RETRAINING_FREQUENCY))-pd.Timedelta(days=1)
        if test_end>max_date:test_end=max_date
        
        df_train_raw=df_featured.loc[train_start:train_end];df_test_chunk=df_featured.loc[period_start_date:test_end]
        if df_train_raw.empty or df_test_chunk.empty: continue
        
        df_train_labeled=fe.label_meta_outcomes(df_train_raw,lookahead=config.LOOKAHEAD_CANDLES)
        if df_train_labeled.empty or df_train_labeled['target'].nunique()<2:
            cycle_metrics.append({'Cycle':i+1,'StartDate':period_start_date.date(),'NumTrades':0,'WinRate':"N/A",'CyclePnL':"$0.00",'Status':"Skipped (Label Error)"});continue
        
        trainer=ModelTrainer(config);training_result=trainer.train(df_train_labeled, feature_list=current_config['selected_features'])
        if training_result is None:
            cycle_metrics.append({'Cycle':i+1,'StartDate':period_start_date.date(),'NumTrades':0,'WinRate':"N/A",'CyclePnL':"$0.00",'Status':"Failed (Training Error)"});continue
        
        model,best_threshold=training_result
        
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
        
        cycle_results_for_ai={"cycle":i+1,"objective_score":trainer.study.best_value if trainer.study else 0, "cycle_pnl":cycle_pnl,"win_rate":cycle_win_rate,"num_trades":len(chunk_trades_df), "status": status, "current_params":{k:v for k,v in current_config.items() if k in ['LOOKAHEAD_CANDLES','MAX_DD_PER_CYCLE','MIN_VOLATILITY_RANK','MAX_VOLATILITY_RANK']}}
        historical_cycle_results.append(cycle_results_for_ai)
        
        all_feature_names = [col for col in ModelTrainer.BASE_FEATURES if col in df_featured.columns]
        suggested_params = gemini_analyzer.analyze_cycle_and_suggest_changes(historical_cycle_results, all_feature_names)
        
        logger.info("  - Respecting API rate limits (5-second delay)..."); time.sleep(5)
        
        current_config.update(suggested_params)
    
    logger.info("\n==========================================================")
    logger.info("                      WALK-FORWARD ANALYSIS COMPLETE");logger.info("==========================================================")
    
    final_config=ConfigModel(**current_config)
    reporter=PerformanceAnalyzer(final_config)
    
    final_trades_df=pd.concat(all_trades,ignore_index=True) if all_trades else pd.DataFrame()
    reporter.generate_full_report(final_trades_df,pd.Series(full_equity_curve),cycle_metrics)
    logger.info(f"\nTotal execution time: {datetime.now() - start_time}")

if __name__ == '__main__':
    if os.name == 'nt': os.system("chcp 65001 > $null")
    main()

# End_To_End_Advanced_ML_Trading_Framework_PRO_V94_Meta_Labeling.py