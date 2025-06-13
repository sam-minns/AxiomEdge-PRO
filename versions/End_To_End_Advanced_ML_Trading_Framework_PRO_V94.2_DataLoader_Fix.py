# End_To_End_Advanced_ML_Trading_Framework_PRO_V94.2_DataLoader_Fix
#
# V94.2 Update:
# 1. BUG FIX: Resolved a "Data loading failed" error by removing all dependencies
#    on unused D1 and H1 data. The data pipeline is now streamlined to only
#    process the M15 data required by the meta-labeling architecture.
# 2. CLEANUP: Removed the now-redundant _calculate_htf_features method and
#    simplified the FeatureEngineer class for better readability and efficiency.
#
# V94.1 Update:
# 1. BUG FIX: Restored the quantitative performance report generation.
#
# V94 Update:
# 1. MAJOR ARCHITECTURAL CHANGE - Meta-Labeling & Volatility Filtering.

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
                "f1_score": round(result.get("f1_score", 0), 4),
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
    BASE_RISK_PER_TRADE_PCT: confloat(gt=0, lt=1)
    SPREAD_PCTG_OF_ATR: confloat(ge=0); SLIPPAGE_PCTG_OF_ATR: confloat(ge=0); OPTUNA_TRIALS: conint(gt=0)
    TRAINING_WINDOW: str; RETRAINING_FREQUENCY: str; FORWARD_TEST_GAP: str; LOOKAHEAD_CANDLES: conint(gt=0)
    MIN_VOLATILITY_RANK: confloat(ge=0, le=1.0); MAX_VOLATILITY_RANK: confloat(ge=0, le=1.0)
    MAX_DD_PER_CYCLE: confloat(gt=0.05, lt=1.0)
    selected_features: List[str]
    MODEL_SAVE_PATH: str = ""; PLOT_SAVE_PATH: str = ""; REPORT_SAVE_PATH: str = "";
    def __init__(self, **data: Any):
        super().__init__(**data)
        result_folder_path=os.path.join(self.BASE_PATH,"Results",self.REPORT_LABEL);os.makedirs(result_folder_path,exist_ok=True)
        self.MODEL_SAVE_PATH=os.path.join(result_folder_path,f"{self.REPORT_LABEL}_model.json")
        self.PLOT_SAVE_PATH=os.path.join(result_folder_path,f"{self.REPORT_LABEL}_equity_curve.png")
        self.REPORT_SAVE_PATH=os.path.join(result_folder_path,f"{self.REPORT_LABEL}_quantitative_report.txt")

# =============================================================================
# 4. DATA LOADER & 5. FEATURE ENGINEERING
# =============================================================================
class DataLoader:
    def __init__(self, config: ConfigModel): self.config = config
    def load_and_parse_data(self, filenames: List[str]) -> Optional[pd.DataFrame]:
        logger.info("-> Stage 1: Loading and Preparing M15 Data...")
        # MODIFIED: Only process M15 data, as it's all the new architecture needs.
        m15_dfs = []
        for filename in filenames:
            if "M15" not in filename: continue
            file_path = os.path.join(self.config.BASE_PATH, filename)
            if not os.path.exists(file_path): logger.warning(f"  - File not found, skipping: {file_path}"); continue
            try:
                symbol = filename.split('_')[0]
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
                df['Symbol']=symbol;m15_dfs.append(df)
            except Exception as e: logger.error(f"  - Failed to load {filename}: {e}", exc_info=True)
        
        if not m15_dfs:
            logger.critical("  - Data loading failed: No M15 files were found or processed.")
            return None
            
        combined=pd.concat(m15_dfs);all_symbols_df=[df[~df.index.duplicated(keep='first')].sort_index() for _,df in combined.groupby('Symbol')]
        final_combined=pd.concat(all_symbols_df).sort_index();final_combined['RealVolume']=pd.to_numeric(final_combined['RealVolume'],errors='coerce').fillna(0)
        logger.info(f"  - Processed M15: {len(final_combined):,} rows for {len(final_combined['Symbol'].unique())} symbols.")
        
        logger.info("[SUCCESS] Data loading and preparation complete.");return final_combined

class FeatureEngineer:
    def __init__(self, config: ConfigModel): self.config = config
    
    def create_feature_stack(self, df_m15: pd.DataFrame) -> pd.DataFrame:
        logger.info("-> Stage 2: Engineering Features & Primary Model Signals...")
        df_m15_base_list = []
        for symbol, group in df_m15.groupby('Symbol'):
            logger.info(f"    - Calculating features and primary signals for {symbol}...")
            df_m15_base_list.append(self._calculate_m15_native(group))
        
        df_final = pd.concat(df_m15_base_list)
        df_final.replace([np.inf,-np.inf],np.nan,inplace=True)
        all_possible_features = ModelTrainer.BASE_FEATURES + ['primary_model_signal']
        df_final.dropna(subset=[f for f in all_possible_features if f in df_final.columns],inplace=True)
        logger.info(f"  - Final dataset shape: {df_final.shape}")
        logger.info("[SUCCESS] Feature engineering complete.");return df_final
    
    def _calculate_m15_native(self,g:pd.DataFrame)->pd.DataFrame:
        g_out=g.copy()
        # --- Feature Calculation ---
        g_out['ATR'] = (g_out['High'] - g_out['Low']).rolling(14).mean()
        delta = g_out['Close'].diff()
        gain = delta.where(delta > 0,0).ewm(com=13, adjust=False).mean()
        loss = -delta.where(delta < 0,0).ewm(com=13, adjust=False).mean()
        g_out['RSI'] = 100 - (100 / (1 + (gain / loss.replace(0, 1e-9))))
        g_out['market_volatility_index'] = g_out['ATR'].rolling(100).rank(pct=True)
        sma_200 = g_out['Close'].rolling(window=200).mean()
        g_out['price_above_sma200'] = (g_out['Close'] > sma_200).astype(int)
        ema_12 = g_out['Close'].ewm(span=12, adjust=False).mean()
        ema_26 = g_out['Close'].ewm(span=26, adjust=False).mean()
        g_out['macd'] = ema_12 - ema_26
        g_out['macd_histogram'] = g_out['macd'] - g_out['macd'].ewm(span=9, adjust=False).mean()
        g_out['hour'] = g_out.index.hour
        g_out['day_of_week'] = g_out.index.dayofweek

        # --- Primary Model Signal Generation (SMA Crossover) ---
        sma_fast = g_out['Close'].rolling(window=20).mean()
        sma_slow = g_out['Close'].rolling(window=50).mean()
        
        signal = np.where(sma_fast > sma_slow, 1.0, -1.0)
        g_out['primary_model_signal'] = signal.diff().fillna(0) # 2.0 for bull cross, -2.0 for bear cross
        
        # Shift features to prevent lookahead bias
        feature_cols = [f for f in ModelTrainer.BASE_FEATURES if f in g_out.columns]
        g_out[feature_cols] = g_out[feature_cols].shift(1)

        return g_out

    def label_meta_outcomes(self, df: pd.DataFrame, lookahead: int) -> pd.DataFrame:
        logger.info("  - Generating Meta-Labels based on Primary Model success...")
        signal_df = df[df['primary_model_signal'] != 0].copy()
        outcomes = np.zeros(len(signal_df), dtype=int)
        
        for i in range(len(signal_df)):
            entry_idx = df.index.get_loc(signal_df.index[i])
            future_candles = df.iloc[entry_idx + 1 : entry_idx + 1 + lookahead]
            if future_candles.empty: continue

            entry_price = df.iloc[entry_idx]['Close']
            signal = np.sign(signal_df.iloc[i]['primary_model_signal'])
            sl_dist = df.iloc[entry_idx]['ATR'] * 1.5
            tp_dist = df.iloc[entry_idx]['ATR'] * 2.0
            
            if pd.isna(sl_dist) or sl_dist <= 1e-9: continue
            
            sl = entry_price - sl_dist if signal == 1 else entry_price + sl_dist
            tp = entry_price + tp_dist if signal == 1 else entry_price - tp_dist
            
            if signal == 1:
                tp_hits = future_candles.index[future_candles['High'] >= tp]
                sl_hits = future_candles.index[future_candles['Low'] <= sl]
            else:
                tp_hits = future_candles.index[future_candles['Low'] <= tp]
                sl_hits = future_candles.index[future_candles['High'] >= sl]

            first_tp_hit = tp_hits[0] if not tp_hits.empty else None
            first_sl_hit = sl_hits[0] if not sl_hits.empty else None

            if first_tp_hit and (not first_sl_hit or first_tp_hit < first_sl_hit):
                outcomes[i] = 1 # Profit
        
        signal_df['target'] = outcomes
        return signal_df

# =============================================================================
# 6. MODEL TRAINER
# =============================================================================
class ModelTrainer:
    BASE_FEATURES = ['ATR', 'RSI', 'market_volatility_index', 'price_above_sma200', 'macd', 'macd_histogram', 'hour', 'day_of_week']
    def __init__(self,config:ConfigModel):self.config=config;self.study: Optional[optuna.study.Study] = None
    
    def train(self,df_train:pd.DataFrame, feature_list: List[str])->Optional[Tuple[Pipeline,float]]:
        logger.info(f"  - Starting META-MODEL training using {len(feature_list)} features...");
        y = df_train['target']
        X = df_train[feature_list]
        
        if y.nunique() < 2:
            logger.error("  - Training aborted: Not enough classes in target labels for this cycle.")
            return None
        
        try:
            X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, shuffle=False, stratify=y)
        except ValueError: # Not enough samples for stratified split
             X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, shuffle=False)

        self.study=self._optimize_hyperparameters(X_train,y_train,X_val,y_val)
        if not self.study or not self.study.best_trials:logger.error("  - Training aborted: Hyperparameter optimization failed.");return None
        logger.info(f"    - Optimization complete. Best F1 Score: {self.study.best_value:.4f}");logger.info(f"    - Best params: {self.study.best_params}")
        
        best_threshold = 0.6 # Use a higher confidence threshold for the meta-model
        final_pipeline=self._train_final_model(self.study.best_params,X,y)
        
        if final_pipeline is None:logger.error("  - Training aborted: Final model training failed.");return None
        logger.info("  - [SUCCESS] Meta-Model training complete.");return final_pipeline, best_threshold

    def _optimize_hyperparameters(self,X_train,y_train,X_val,y_val)->Optional[optuna.study.Study]:
        logger.info(f"    - Starting hyperparameter optimization for meta-model ({self.config.OPTUNA_TRIALS} trials)...")
        
        def objective(trial:optuna.Trial):
            scale_pos_weight = (y_train==0).sum()/(y_train==1).sum() if (y_train==1).sum() > 0 else 1.0
            param={'objective':'binary:logistic','eval_metric':'logloss','booster':'gbtree','tree_method':'hist', 'use_label_encoder':False, 'seed':42,
                   'n_estimators':trial.suggest_int('n_estimators',100,800,step=50),'max_depth':trial.suggest_int('max_depth',3,7),
                   'learning_rate':trial.suggest_float('learning_rate',0.01,0.3,log=True), 'subsample':trial.suggest_float('subsample',0.6,1.0),
                   'colsample_bytree':trial.suggest_float('colsample_bytree',0.6,1.0),'gamma':trial.suggest_float('gamma',0,5),
                   'scale_pos_weight': trial.suggest_float('scale_pos_weight', 1.0, scale_pos_weight, log=True)
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
            best_params.pop('early_stopping_rounds', None)
            final_params={'objective':'binary:logistic','eval_metric':'logloss', 'booster':'gbtree','tree_method':'hist', 'use_label_encoder':False,'seed':42, **best_params}
            final_pipeline=Pipeline([('scaler',RobustScaler()),('model',xgb.XGBClassifier(**final_params))])
            final_pipeline.fit(X,y)
            return final_pipeline
        except Exception as e:logger.error(f"    - Error during final model training: {e}",exc_info=True);return None

# =============================================================================
# 7. BACKTESTER & 8. PERFORMANCE ANALYZER
# =============================================================================
class Backtester:
    def __init__(self,config:ConfigModel):self.config=config
    def run_backtest_chunk(self,df_chunk_in:pd.DataFrame,model:Pipeline,feature_list:List[str],confidence_threshold:float,initial_equity:float)->Tuple[pd.DataFrame,pd.Series,bool]:
        if df_chunk_in.empty:return pd.DataFrame(),pd.Series([initial_equity]), False
        df_chunk=df_chunk_in.copy()
        
        X_test_meta = df_chunk[feature_list].copy().fillna(0)
        meta_model_probs = model.predict_proba(X_test_meta)[:, 1]
        df_chunk['meta_model_confidence'] = meta_model_probs

        trades,equity,equity_curve,open_positions=[],initial_equity,[initial_equity],{}
        chunk_peak_equity = initial_equity; circuit_breaker_tripped = False
        
        candles=df_chunk.reset_index().to_dict('records')
        for candle in candles:
            if not circuit_breaker_tripped:
                if equity > chunk_peak_equity: chunk_peak_equity = equity
                if equity > 0 and chunk_peak_equity > 0 and (chunk_peak_equity - equity) / chunk_peak_equity > self.config.MAX_DD_PER_CYCLE:
                    logger.warning(f"  - CIRCUIT BREAKER TRIPPED! Drawdown exceeded {self.config.MAX_DD_PER_CYCLE:.0%} for this cycle. Closing all positions.")
                    circuit_breaker_tripped = True; open_positions = {}
            
            symbol = candle['Symbol']
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

            primary_signal = candle.get('primary_model_signal', 0)
            if primary_signal != 0 and symbol not in open_positions:
                meta_confidence = candle.get('meta_model_confidence', 0)
                if meta_confidence >= confidence_threshold:
                    volatility = candle.get('market_volatility_index', 0.5)
                    if self.config.MIN_VOLATILITY_RANK <= volatility <= self.config.MAX_VOLATILITY_RANK:
                        direction = int(np.sign(primary_signal))
                        atr=candle.get('ATR',0); 
                        if pd.isna(atr) or atr<=1e-9:continue
                        
                        risk_amt=equity*self.config.BASE_RISK_PER_TRADE_PCT
                        sl_dist=(atr*1.5)+(atr*self.config.SPREAD_PCTG_OF_ATR)+(atr*self.config.SLIPPAGE_PCTG_OF_ATR)
                        tp_dist=(atr*2.0)
                        if tp_dist<=0:continue
                        
                        entry_price=candle['Close'];sl_price,tp_price=entry_price-sl_dist*direction,entry_price+tp_dist*direction
                        open_positions[symbol]={'direction':direction,'sl':sl_price,'tp':tp_price,'risk_amt':risk_amt,'rr':2.0,'confidence':meta_confidence}

        return pd.DataFrame(trades),pd.Series(equity_curve), circuit_breaker_tripped

class PerformanceAnalyzer:
    def __init__(self,config:ConfigModel):self.config=config
    def generate_full_report(self,trades_df:Optional[pd.DataFrame],equity_curve:Optional[pd.Series],cycle_metrics:List[Dict]):
        logger.info("-> Stage 4: Generating Final Performance Report...")
        if equity_curve is not None and len(equity_curve) > 1:
            self.plot_equity_curve(equity_curve)
        
        metrics = self._calculate_metrics(trades_df, equity_curve) if trades_df is not None and not trades_df.empty else {}
        self.generate_text_report(metrics, cycle_metrics)
        logger.info("[SUCCESS] Final report generated and saved.")
    
    def plot_equity_curve(self,equity_curve:pd.Series):
        plt.style.use('seaborn-v0_8-darkgrid');plt.figure(figsize=(16,8));plt.plot(equity_curve.values,color='dodgerblue',linewidth=2)
        plt.title(f"{self.config.REPORT_LABEL} - Walk-Forward Equity Curve",fontsize=16,weight='bold')
        plt.xlabel("Trade Number",fontsize=12);plt.ylabel("Equity ($)",fontsize=12);plt.grid(True,which='both',linestyle=':');plt.savefig(self.config.PLOT_SAVE_PATH);plt.close()
        logger.info(f"  - Equity curve plot saved to: {self.config.PLOT_SAVE_PATH}")

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

    def generate_text_report(self,m:Dict[str,Any],cycle_metrics:List[Dict]):
        cycle_df=pd.DataFrame(cycle_metrics)
        cycle_report="Per-Cycle Performance:\n"+cycle_df.to_string(index=False) if not cycle_df.empty else "No trades were executed."
        shap_report="SHAP report generation is not applicable for the meta-labeling architecture."
        
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
        "BASE_PATH": os.getcwd(), "REPORT_LABEL": "ML_Framework_V94_Meta_Labeling",
        "FORWARD_TEST_START_DATE": "2024-01-01", "INITIAL_CAPITAL": 100000.0,
        "BASE_RISK_PER_TRADE_PCT": 0.01, "SPREAD_PCTG_OF_ATR": 0.05, "SLIPPAGE_PCTG_OF_ATR": 0.02,
        "OPTUNA_TRIALS": 20, "TRAINING_WINDOW": '365D', "RETRAINING_FREQUENCY": '60D', "FORWARD_TEST_GAP": "1D",
        "LOOKAHEAD_CANDLES": 100, "MAX_DD_PER_CYCLE": 0.25, 
        "MIN_VOLATILITY_RANK": 0.1, "MAX_VOLATILITY_RANK": 0.9,
        "selected_features": ModelTrainer.BASE_FEATURES
    }
    
    files_to_process=["AUDUSD_M15_202105170000_202506021830.csv","EURUSD_M15_202106020100_202506021830.csv","GBPUSD_M15_202106020015_202506021830.csv","USDCAD_M15_202105170000_202506021830.csv"]
    
    try:
        gemini_analyzer=GeminiAnalyzer();initial_config=ConfigModel(**current_config)
        df_m15=DataLoader(initial_config).load_and_parse_data(filenames=files_to_process)
        if df_m15 is None: return
        fe=FeatureEngineer(initial_config);df_featured=fe.create_feature_stack(df_m15)
        if df_featured.empty:return
    except Exception as e:logger.critical(f"[FATAL] Initial setup failed: {e}",exc_info=True);return
    
    test_start_date=pd.to_datetime(current_config['FORWARD_TEST_START_DATE']);max_date=df_featured.index.max()
    retraining_dates=pd.date_range(start=test_start_date,end=max_date,freq=current_config['RETRAINING_FREQUENCY'])
    all_trades,full_equity_curve,cycle_metrics=[],[current_config['INITIAL_CAPITAL']],[]
    historical_cycle_results = []
    
    logger.info("-> Stage 3: Starting Adaptive Walk-Forward Analysis with AI Tuning...")
    for i,period_start_date in enumerate(retraining_dates):
        config=ConfigModel(**current_config)
        logger.info(f"\n{'='*25} CYCLE {i+1}/{len(retraining_dates)}: {period_start_date.date()} {'='*25}")
        logger.info(f"  - Using Config: LOOKAHEAD={config.LOOKAHEAD_CANDLES}, MAX_DD_CYCLE={config.MAX_DD_PER_CYCLE}, VOL_FILTER=[{config.MIN_VOLATILITY_RANK:.2f}-{config.MAX_VOLATILITY_RANK:.2f}]")
        
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
        
        cycle_results_for_ai={"cycle":i+1,"f1_score":trainer.study.best_value if trainer.study else 0, "cycle_pnl":cycle_pnl,"win_rate":cycle_win_rate,"num_trades":len(chunk_trades_df), "status": status, "current_params":{k:v for k,v in current_config.items() if k in ['LOOKAHEAD_CANDLES','MAX_DD_PER_CYCLE','MIN_VOLATILITY_RANK','MAX_VOLATILITY_RANK']}}
        historical_cycle_results.append(cycle_results_for_ai)
        
        all_feature_names = [col for col in ModelTrainer.BASE_FEATURES if col in df_featured.columns]
        suggested_params = gemini_analyzer.analyze_cycle_and_suggest_changes(historical_cycle_results, all_feature_names)
        
        logger.info("  - Respecting API rate limits (5-second delay)..."); time.sleep(5)
        
        if suggested_params: current_config.update(suggested_params)
    
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

# End_To_End_Advanced_ML_Trading_Framework_PRO_V94.2_DataLoader_Fix.py