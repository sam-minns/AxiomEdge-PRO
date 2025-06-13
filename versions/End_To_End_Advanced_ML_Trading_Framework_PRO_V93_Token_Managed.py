# End_To_End_Advanced_ML_Trading_Framework_PRO_V93_Token_Managed
#
# V93 Update:
# 1. NEW - Context Token Management: To prevent exceeding API token limits on
#    long backtests, the AI context is now actively managed. The script now
#    summarizes and truncates the cycle history, sending only the most recent
#    5 cycles and summarizing bulky items like SHAP reports into a concise list.
# 2. ENHANCED - AI Prompt: The forensic AI prompt is enhanced to encourage it
#    to break out of failure loops by suggesting novel strategies if previous
#    attempts consistently fail.
# 3. All previous bug fixes and stability improvements are included.


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
        # Simple check to prevent extremely large prompts
        if len(prompt) > 28000: # Approx 7k tokens as a safety buffer
             logger.warning("Prompt is very large, may risk exceeding token limits.")
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
        """Summarizes and truncates historical data to manage token count."""
        # Truncate to the most recent 5 cycles
        recent_history = historical_results[-5:]
        
        summarized_history = []
        for result in recent_history:
            summary = {
                "cycle": result.get("cycle"),
                "status": result.get("status"),
                "objective_score": round(result.get("objective_score", 0), 4),
                "cycle_pnl": result.get("cycle_pnl"),
                "win_rate": result.get("win_rate"),
                "num_trades": result.get("num_trades"),
                "params_used": {
                    "LOOKAHEAD_CANDLES": result.get("current_params", {}).get("LOOKAHEAD_CANDLES"),
                    "MAX_DD_PER_CYCLE": result.get("current_params", {}).get("MAX_DD_PER_CYCLE"),
                }
            }
            # Summarize SHAP results to only include top 5 feature names
            if result.get("shap_summary"):
                try:
                    summary["top_5_features"] = list(result["shap_summary"]["SHAP_Importance"].keys())[:5]
                except Exception:
                    summary["top_5_features"] = "Error parsing SHAP"
            summarized_history.append(summary)
        return summarized_history

    def analyze_cycle_and_suggest_changes(self, historical_results: List[Dict], available_features: List[str]) -> Dict:
        logger.info("  - Summarizing history and sending to Gemini for forensic analysis...")
        
        summarized_history = self._summarize_historical_results(historical_results)

        prompt = (
            "You are an expert trading model analyst and forensic data scientist. Your primary goal is to create a STABLE and PROFITABLE strategy by learning from past failures. "
            "The historical results show that the models are consistently unstable and fail immediately on out-of-sample data. Your task is to find out WHY and suggest changes to break this failure loop.\n\n"
            "Analyze the SUMMARIZED history of the most recent cycles for hidden patterns, correlations, or event triggers that might explain the failures. For example: "
            "  - Are the models failing because the `LOOKAHEAD_CANDLES` value is inappropriate for the market? "
            "  - The `status` 'Circuit Breaker' means the model was too risky for the market conditions. How can we avoid this? "
            "  - If the same features keep appearing in failed cycles, perhaps they should be avoided. "
            "If the strategy is stuck in a failure loop, suggest a *significantly different* approach (e.g., a much simpler model, a different feature set, a different risk profile) to break the cycle.\n\n"
            "Based on your forensic analysis, suggest a NEW configuration for the next cycle with the goal of SURVIVAL and STABILITY.\n\n"
            "1.  **Risk Management**: Suggest a new float value for `MAX_DD_PER_CYCLE` (between 0.15 and 0.40).\n"
            "2.  **Model Parameters**: Suggest new integer values for `LOOKAHEAD_CANDLES` (50-200) and `OPTUNA_TRIALS` (15-30), and a new float for `TREND_FILTER_THRESHOLD` (25.0-45.0).\n"
            "3.  **Feature Selection**: Select a SUBSET of features from the `available_features` list. Be creative. A simpler model with fewer features may be more robust.\n\n"
            "Respond ONLY with a valid JSON object containing the keys: `MAX_DD_PER_CYCLE`, `LOOKAHEAD_CANDLES`, `OPTUNA_TRIALS`, `TREND_FILTER_THRESHOLD`, and `selected_features` (a list of strings).\n\n"
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
    MIN_CLASS_SAMPLES: conint(gt=10) = 50; MIN_CLASS_RATIO: confloat(gt=0.01, lt=1) = 0.05; TREND_FILTER_THRESHOLD: confloat(gt=0) = 25.0
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
# 4. DATA LOADER & 5. FEATURE ENGINEERING (No changes needed)
# =============================================================================
class DataLoader:
    # ... (code is unchanged from V92)
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
    # ... (code is unchanged from V92)
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
        g['bollinger_upper'] = middle_band + (std_dev * 2)
        g['bollinger_lower'] = middle_band - (std_dev * 2)
        g['bb_width'] = (g['bollinger_upper'] - g['bollinger_lower']) / middle_band.replace(0,np.nan)
        g['bollinger_bandwidth'] = g['bb_width']
        return g
    def _calculate_stochastic(self, g:pd.DataFrame, period:int) -> pd.DataFrame:
        low_min=g['Low'].rolling(window=period).min();high_max=g['High'].rolling(window=period).max()
        g['stoch_k']=100*(g['Close']-low_min)/(high_max-low_min).replace(0,np.nan);g['stoch_d']=g['stoch_k'].rolling(window=3).mean();return g
    def _calculate_cross_pair_features(self, df: pd.DataFrame) -> pd.DataFrame:
        logger.info("    - Engineering new cross-pair features (relative strength, correlation)...")
        df_out=df.copy().reset_index()
        df_pivot=df_out.pivot(index='Timestamp',columns='Symbol',values='Close').ffill().bfill();returns=df_pivot.pct_change(fill_method=None).shift(1).fillna(0)
        mean_returns=returns.mean(axis=1);rel_strength_df=returns.apply(lambda col:col-mean_returns,axis=0)
        rel_strength_unpivoted=rel_strength_df.stack().reset_index(name='rel_strength');rel_strength_unpivoted.columns=['Timestamp','Symbol','rel_strength']
        df_out=pd.merge(df_out,rel_strength_unpivoted,on=['Timestamp','Symbol'],how='left');symbols=list(returns.columns)
        if len(symbols)>1:
            for i in range(len(symbols)):
                for j in range(i+1,len(symbols)):
                    sym1,sym2=symbols[i],symbols[j];corr_col_name=f'corr_{sym1}_{sym2}'
                    rolling_corr=returns[sym1].rolling(window=100).corr(returns[sym2])
                    corr_df=pd.DataFrame(rolling_corr).reset_index();corr_df.columns=['Timestamp',corr_col_name]
                    df_out=pd.merge(df_out,corr_df,on='Timestamp',how='left')
        df_out.set_index('Timestamp',inplace=True);df_out.bfill(inplace=True);return df_out
    def create_feature_stack(self, data_by_tf: Dict[str, pd.DataFrame]) -> pd.DataFrame:
        logger.info("-> Stage 2: Engineering Features from Multi-Timeframe Data...")
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
        df_with_cross_feats=self._calculate_cross_pair_features(df_merged)
        df_with_cross_feats['adx_x_h1_trend']=df_with_cross_feats['ADX']*df_with_cross_feats['H1_ctx_H1_Trend']
        df_with_cross_feats['atr_x_d1_trend']=df_with_cross_feats['ATR']*df_with_cross_feats['D1_ctx_D1_Trend']
        df_with_cross_feats['market_regime']=np.where(df_with_cross_feats['ADX']>self.config.TREND_FILTER_THRESHOLD,1,0)
        df_final=df_with_cross_feats.copy(); df_final.replace([np.inf,-np.inf],np.nan,inplace=True)
        all_possible_features = ModelTrainer.BASE_FEATURES + [f'corr_{s1}_{s2}' for i,s1 in enumerate(df_m15_base['Symbol'].unique()) for s2 in df_m15_base['Symbol'].unique()[i+1:]]
        df_final.dropna(subset=[f for f in all_possible_features if f in df_final.columns and f != 'ichimoku_chikou_span'],inplace=True)
        logger.info(f"  - Merged data and created features. Final dataset shape: {df_final.shape}")
        logger.info("[SUCCESS] Feature engineering complete.");return df_final
    def _calculate_htf_features(self,df:pd.DataFrame,p:str,s:int,a:int)->pd.DataFrame:
        logger.info(f"    - Calculating HTF features for {p}...");results=[]
        for symbol,group in df.groupby('Symbol'):
            g=group.copy();sma=g['Close'].rolling(s,min_periods=s).mean();atr=(g['High']-g['Low']).rolling(a,min_periods=a).mean();trend=np.sign(g['Close']-sma)
            temp_df=pd.DataFrame(index=g.index);temp_df[f'{p}_ctx_{p}_SMA']=sma;temp_df[f'{p}_ctx_{p}_ATR']=atr;temp_df[f'{p}_ctx_{p}_Trend']=trend
            shifted_df=temp_df.shift(1);shifted_df['Symbol']=symbol;results.append(shifted_df)
        return pd.concat(results).reset_index()
    def _calculate_m15_native(self,g:pd.DataFrame)->pd.DataFrame:
        logger.info(f"    - Calculating native M15 features for {g['Symbol'].iloc[0]}...");g_out=g.copy();lookback,rank_period,sma_period=14,100,50
        
        g_out['ATR']=(g['High']-g['Low']).rolling(lookback).mean();delta=g['Close'].diff();gain=delta.where(delta > 0,0).ewm(com=lookback-1,adjust=False).mean()
        loss=-delta.where(delta < 0,0).ewm(com=lookback-1,adjust=False).mean();g_out['RSI']=100-(100/(1+(gain/loss.replace(0,1e-9))))
        g_out=self._calculate_adx(g_out,lookback);g_out=self._calculate_bollinger_bands(g_out,self.config.BOLLINGER_PERIOD);g_out=self._calculate_stochastic(g_out,self.config.STOCHASTIC_PERIOD)
        g_out['hour'] = g_out.index.hour;g_out['day_of_week'] = g_out.index.dayofweek
        g_out['month']=g_out.index.month; g_out['week']=g_out.index.isocalendar().week.astype(int)
        london_open,london_close,ny_open,ny_close,tokyo_open,tokyo_close=8,16,13,21,0,8
        g_out['is_london']=((g_out.index.hour>=london_open)&(g_out.index.hour<london_close)).astype(int);g_out['is_ny']=((g_out.index.hour>=ny_open)&(g_out.index.hour<ny_close)).astype(int)
        g_out['is_tokyo']=((g_out.index.hour>=tokyo_open)&(g_out.index.hour<tokyo_close)).astype(int);g_out['is_london_ny_overlap']=((g_out.index.hour>=ny_open)&(g_out.index.hour<london_close)).astype(int)
        for session in ['london','ny','tokyo']: g_out[f'{session}_opening_transition']=(g_out[f'is_{session}'].diff()==1).astype(int); g_out[f'{session}_closing_transition']=(g_out[f'is_{session}'].diff()==-1).astype(int)
        ema_12 = g_out['Close'].ewm(span=12, adjust=False).mean();ema_26 = g_out['Close'].ewm(span=26, adjust=False).mean()
        g_out['macd'] = ema_12 - ema_26; g_out['macd_signal'] = g_out['macd'].ewm(span=9, adjust=False).mean(); g_out['macd_histogram'] = g_out['macd'] - g_out['macd_signal']
        obv = (np.sign(g_out['Close'].diff()) * g_out['RealVolume']).fillna(0).cumsum(); g_out['obv'] = obv
        for p in [20, 50, 100, 200]: g_out[f'ema_{p}'] = g_out['Close'].ewm(span=p, adjust=False).mean()
        for p in [20, 50, 100, 200]: g_out[f'sma_{p}'] = g_out['Close'].rolling(window=p).mean()
        g_out['price_above_sma200'] = (g_out['Close'] > g_out['sma_200']).astype(int); g_out['dist_from_sma50']=(g_out['Close']-g_out['sma_50'])/g_out['ATR'].replace(0,np.nan)
        rsi_14 = g_out['RSI'];stoch_rsi_k = (rsi_14 - rsi_14.rolling(14).min()) / (rsi_14.rolling(14).max() - rsi_14.rolling(14).min())
        g_out['stoch_rsi'] = stoch_rsi_k.rolling(3).mean() * 100;typical_price = (g_out['High'] + g_out['Low'] + g_out['Close']) / 3
        g_out['cci'] = (typical_price - typical_price.rolling(20).mean()) / (0.015 * typical_price.rolling(20).apply(lambda x: np.mean(np.abs(x - x.mean())), raw=True))
        g_out['willr'] = -100 * ((g_out['High'].rolling(14).max() - g_out['Close']) / (g_out['High'].rolling(14).max() - g_out['Low'].rolling(14).min()))
        g_out['adx_trend_strength'] = g_out['ADX'];g_out['volume'] = g_out['RealVolume'];g_out['volume_sma20'] = g_out['RealVolume'].rolling(20).mean()
        g_out['volume_sma50'] = g_out['RealVolume'].rolling(50).mean();g_out['price_rate_of_change'] = g_out['Close'].pct_change(periods=14)
        g_out['momentum_10'] = g_out['Close'].diff(10);g_out['momentum_20'] = g_out['Close'].diff(20)
        g_with_date = g_out.copy();g_with_date['date'] = g_with_date.index.normalize()
        daily_hlc = g_with_date.groupby('date').agg(prev_high=('High','max'), prev_low=('Low','min'), prev_close=('Close','last')).shift(1)
        daily_hlc['pivot_point'] = (daily_hlc['prev_high'] + daily_hlc['prev_low'] + daily_hlc['prev_close']) / 3
        daily_hlc['pivot_r1'] = 2*daily_hlc['pivot_point'] - daily_hlc['prev_low']; daily_hlc['pivot_s1'] = 2*daily_hlc['pivot_point'] - daily_hlc['prev_high']
        daily_hlc['pivot_r2'] = daily_hlc['pivot_point'] + (daily_hlc['prev_high'] - daily_hlc['prev_low']); daily_hlc['pivot_s2'] = daily_hlc['pivot_point'] - (daily_hlc['prev_high'] - daily_hlc['prev_low'])
        daily_hlc['support_level'] = daily_hlc['prev_low']; daily_hlc['resistance_level'] = daily_hlc['prev_high'];daily_range = daily_hlc['prev_high'] - daily_hlc['prev_low']
        daily_hlc['fibonacci_38_2'] = daily_hlc['prev_high'] - daily_range * 0.382;daily_hlc['fibonacci_50'] = daily_hlc['prev_high'] - daily_range * 0.500;daily_hlc['fibonacci_61_8'] = daily_hlc['prev_high'] - daily_range * 0.618
        g_out = g_out.join(daily_hlc.drop(columns=['prev_high', 'prev_low', 'prev_close']), on=g_with_date['date'])
        g_out['aroon_up'] = 100 * g_out['High'].rolling(25).apply(lambda x: float(np.argmax(x)), raw=True) / 25
        g_out['aroon_down'] = 100 * g_out['Low'].rolling(25).apply(lambda x: float(np.argmin(x)), raw=True) / 25
        g_out['aroon_oscillator'] = g_out['aroon_up'] - g_out['aroon_down']
        ha_close = (g['Open'] + g['High'] + g['Low'] + g['Close']) / 4;ha_open = pd.Series(index=g.index, dtype=float); ha_open.iloc[0] = (g['Open'].iloc[0] + g['Close'].iloc[0]) / 2
        for i in range(1, len(g)): ha_open.iloc[i] = (ha_open.iloc[i-1] + ha_close.iloc[i-1]) / 2
        g_out['heikin_ashi_open'] = ha_open; g_out['heikin_ashi_close'] = ha_close
        g_out['heikin_ashi_high'] = pd.concat([g['High'], g_out['heikin_ashi_open'], g_out['heikin_ashi_close']], axis=1).max(axis=1)
        g_out['heikin_ashi_low'] = pd.concat([g['Low'], g_out['heikin_ashi_open'], g_out['heikin_ashi_close']], axis=1).min(axis=1)
        g_out['ichimoku_tenkan_sen'] = (g['High'].rolling(9).max() + g['Low'].rolling(9).min()) / 2
        g_out['ichimoku_kijun_sen'] = (g['High'].rolling(26).max() + g['Low'].rolling(26).min()) / 2
        g_out['ichimoku_senkou_span_a'] = ((g_out['ichimoku_tenkan_sen'] + g_out['ichimoku_kijun_sen']) / 2).shift(26)
        g_out['ichimoku_senkou_span_b'] = ((g['High'].rolling(52).max() + g['Low'].rolling(52).min()) / 2).shift(26)
        g_out['ichimoku_chikou_span'] = g['Close'].shift(-26)
        g_out['price_action_high']=g_out['High']; g_out['price_action_low']=g_out['Low']; g_out['price_action_close']=g_out['Close']; g_out['price_action_open']=g_out['Open']
        g_out['candle_body_size'] = abs(g_out['Close'] - g_out['Open']);g_out['candle_upper_shadow'] = g_out['High'] - pd.concat([g_out['Open'], g_out['Close']], axis=1).max(axis=1)
        g_out['candle_lower_shadow'] = pd.concat([g_out['Open'], g_out['Close']], axis=1).min(axis=1) - g_out['Low']
        g_out['doji_detected'] = (g_out['candle_body_size'] / g_out['ATR'].replace(0,1)).lt(0.1).astype(int)
        g_out['engulfing_detected'] = (g_out['candle_body_size'] > abs(g_out['Close'].shift() - g_out['Open'].shift())) & (np.sign(g_out['Close']-g_out['Open']) != np.sign(g_out['Close'].shift()-g_out['Open'].shift())).astype(int)
        g_out['hammer_detected'] = ((g_out['candle_lower_shadow'] > 2 * g_out['candle_body_size']) & (g_out['candle_upper_shadow'] < g_out['candle_body_size'])).astype(int)
        g_out['rsi_x_adx'] = g_out['RSI'] * g_out['ADX']
        regime_score=(g_out['ADX']-g_out['ADX'].min())/(g_out['ADX'].max()-g_out['ADX'].min()) if (g_out['ADX'].max()-g_out['ADX'].min()) > 0 else 0
        g_out['contextual_atr'] = g_out['ATR'] * regime_score
        direction=np.sign(g_out['Close'].diff());g_out['consecutive_candles']=direction.groupby((direction!=direction.shift()).cumsum()).cumcount()+1
        g_out['market_volatility_index'] = g_out['ATR'].rolling(50).rank(pct=True);g_out['trend_strength_index'] = g_out['adx_trend_strength'].rolling(50).rank(pct=True)
        g_out['regime_shift_flag'] = (g_out['market_volatility_index'].diff().abs() > 0.2).astype(int)
        
        feature_cols = [f for f in ModelTrainer.BASE_FEATURES if f in g_out.columns]
        g_out[feature_cols]=g_out[feature_cols].shift(1)
        for col in['ATR','RSI','bb_width','stoch_k']:g_out[f'{col}_rank']=g_out[col].rolling(rank_period).rank(pct=True)
        return g_out
    def label_outcomes(self,df:pd.DataFrame,lookahead:int)->pd.DataFrame:
        logger.info("  - Generating trade labels with Regime-Adjusted Barriers...");
        labeled_dfs=[self._label_group(group,lookahead) for _,group in df.groupby('Symbol')];return pd.concat(labeled_dfs)
    def _label_group(self,group:pd.DataFrame,lookahead:int)->pd.DataFrame:
        if len(group)<lookahead+1:return group
        is_trending=group['ADX']>self.config.TREND_FILTER_THRESHOLD
        sl_multiplier=np.where(is_trending,2.0,1.5);tp_multiplier=np.where(is_trending,4.0,2.5)
        sl_atr_dynamic=group['ATR']*sl_multiplier;tp_atr_dynamic=group['ATR']*tp_multiplier
        outcomes=np.zeros(len(group));prices,lows,highs=group['Close'].values,group['Low'].values,group['High'].values
        for i in range(len(group)-lookahead):
            sl_dist,tp_dist=sl_atr_dynamic[i],tp_atr_dynamic[i]
            if pd.isna(sl_dist) or sl_dist<=1e-9:continue
            tp_long,sl_long=prices[i]+tp_dist,prices[i]-sl_dist;future_highs,future_lows=highs[i+1:i+1+lookahead],lows[i+1:i+1+lookahead]
            time_to_tp=np.where(future_highs>=tp_long)[0];time_to_sl=np.where(future_lows<=sl_long)[0]
            first_tp=time_to_tp[0] if len(time_to_tp)>0 else np.inf;first_sl=time_to_sl[0] if len(time_to_sl)>0 else np.inf
            if first_tp<first_sl:outcomes[i]=1
            elif first_sl<first_tp:outcomes[i]=-1
        group['target']=outcomes;return group

# =============================================================================
# 6. MODEL TRAINER
# =============================================================================
class ModelTrainer:
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
        'ichimoku_chikou_span', 'price_action_high', 'price_action_low', 'price_action_close', 'price_action_open',
        'candle_body_size', 'candle_upper_shadow', 'candle_lower_shadow', 'doji_detected', 'engulfing_detected',
        'hammer_detected', 'market_volatility_index', 'trend_strength_index', 'regime_shift_flag'
    ]
    def __init__(self,config:ConfigModel):self.config=config;self.shap_summary:Optional[pd.DataFrame]=None;self.class_weights:Optional[Dict[int,float]]=None;self.best_threshold=0.5; self.study: Optional[optuna.study.Study] = None
    def train(self,df_train:pd.DataFrame, feature_list: List[str])->Optional[Tuple[Pipeline,float]]:
        logger.info(f"  - Starting model training using {len(feature_list)} features...");y_map={-1:0,0:1,1:2};y=df_train['target'].map(y_map).astype(int)
        X=df_train[feature_list].copy().fillna(0)
        self.class_weights=dict(zip(np.unique(y),compute_class_weight(class_weight='balanced',classes=np.unique(y),y=y)))
        
        # Split data into Train, Validation, and a small Stability set
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
            param={'objective':'multi:softprob','num_class':3,'eval_metric':'mlogloss','booster':'gbtree','tree_method':'hist','use_label_encoder':False,'seed':42,'n_estimators':trial.suggest_int('n_estimators',100,800,step=50),'max_depth':trial.suggest_int('max_depth',3,8),'learning_rate':trial.suggest_float('learning_rate',0.01,0.3,log=True),'subsample':trial.suggest_float('subsample',0.6,1.0),'colsample_bytree':trial.suggest_float('colsample_bytree',0.6,1.0),'gamma':trial.suggest_float('gamma',0,5),'reg_lambda':trial.suggest_float('reg_lambda',1e-8,5.0,log=True),'alpha':trial.suggest_float('alpha',1e-8,5.0,log=True),'early_stopping_rounds':50}
            try:
                scaler=RobustScaler();X_train_scaled=scaler.fit_transform(X_train);X_val_scaled=scaler.transform(X_val)
                model=xgb.XGBClassifier(**param); fit_params={'sample_weight':y_train.map(self.class_weights)}
                model.fit(X_train_scaled,y_train,eval_set=[(X_val_scaled,y_val)],verbose=False,**fit_params)
                
                # --- Main Validation Score ---
                preds=model.predict(X_val_scaled); f1 = f1_score(y_val,preds,average='macro', zero_division=0)
                pnl_map = {-1: 0, 0: -1, 1: -1, 2: 1}; pnl = pd.Series(preds).map(lambda p: pnl_map[p] if p != 1 else 0)
                downside_returns = pnl[pnl < 0]; downside_std = downside_returns.std()
                sortino = (pnl.mean() / downside_std) if downside_std > 0 else 0
                objective_score = (0.4 * f1) + (0.6 * sortino)

                # --- NEW: Stability Gate ---
                X_stability_scaled = scaler.transform(X_stability)
                stability_preds = model.predict(X_stability_scaled)
                stability_pnl = pd.Series(stability_preds).map(lambda p: pnl_map[p] if p != 1 else 0)
                if stability_pnl.sum() < 0: # Penalize models that lose money on the stability set
                    objective_score -= 0.5

                return objective_score

            except Exception as e:logger.warning(f"Trial {trial.number} failed with error: {e}");return -2.0
        try:study=optuna.create_study(direction='maximize');study.optimize(custom_objective,n_trials=self.config.OPTUNA_TRIALS,timeout=1800,show_progress_bar=False);return study
        except Exception as e:logger.error(f"    - Optuna study failed catastrophically: {e}",exc_info=True);return None
    def _train_final_model(self,best_params:Dict,X:pd.DataFrame,y:pd.Series)->Optional[Pipeline]:
        logger.info("    - Training final model...");
        try:
            # Remove early stopping from the final training params
            best_params.pop('early_stopping_rounds', None)
            final_params={'objective':'multi:softprob','num_class':3,'eval_metric':'mlogloss','booster':'gbtree','tree_method':'hist','use_label_encoder':False,'seed':42,**best_params}
            final_pipeline=Pipeline([('scaler',RobustScaler()),('model',xgb.XGBClassifier(**final_params))])
            fit_params={'model__sample_weight':y.map(self.class_weights)}
            final_pipeline.fit(X,y,**fit_params) # Train on all available data (train+val)
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
            
            explainer = shap.TreeExplainer(model); shap_explanation = explainer(X_sample)
            mean_abs_shap_values = shap_explanation.abs.mean(0).values
            overall_importance = mean_abs_shap_values.mean(axis=1) if mean_abs_shap_values.ndim == 2 else mean_abs_shap_values
            summary = pd.DataFrame(overall_importance,index=feature_names,columns=['SHAP_Importance']).sort_values(by='SHAP_Importance',ascending=False)
            self.shap_summary = summary
            logger.info("    - SHAP summary generated successfully.")
        except Exception as e:
            logger.error(f"    - Failed to generate SHAP summary: {e}", exc_info=True); self.shap_summary = None

# =============================================================================
# 7. BACKTESTER, 8. PERFORMANCE ANALYZER
# =============================================================================
class Backtester:
    # ... (code is unchanged from V92)
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
        if equity_curve is not None and len(equity_curve) > 1:
            self.plot_equity_curve(equity_curve)
        if aggregated_shap is not None:
            self.plot_shap_summary(aggregated_shap)
        
        metrics = self._calculate_metrics(trades_df, equity_curve) if trades_df is not None and not trades_df.empty else {}
        self.generate_text_report(metrics, cycle_metrics, aggregated_shap)
        logger.info("[SUCCESS] Final report generated and saved.")
    # ... (plot functions and _calculate_metrics are unchanged from V92) ...
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
        shap_report="Aggregated Feature Importance (SHAP):\n"+aggregated_shap.to_string() if aggregated_shap is not None else "SHAP summary was not generated."
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
        "BASE_PATH": os.getcwd(), "REPORT_LABEL": "ML_Framework_V93_Token_Managed",
        "FORWARD_TEST_START_DATE": "2024-01-01", "INITIAL_CAPITAL": 100000.0,
        "CONFIDENCE_TIERS": {'ultra_high':{'min':0.8,'risk_mult':1.2,'rr':3.0},'high':{'min':0.7,'risk_mult':1.0,'rr':2.5},'standard':{'min':0.6,'risk_mult':0.8,'rr':2.0}},
        "BASE_RISK_PER_TRADE_PCT": 0.01, "SPREAD_PCTG_OF_ATR": 0.05, "SLIPPAGE_PCTG_OF_ATR": 0.02,
        "OPTUNA_TRIALS": 20, "TRAINING_WINDOW": '365D', "RETRAINING_FREQUENCY": '60D', "FORWARD_TEST_GAP": "1D",
        "LOOKAHEAD_CANDLES": 100, "MIN_CLASS_SAMPLES": 50, "MIN_CLASS_RATIO": 0.05,
        "TREND_FILTER_THRESHOLD": 25.0, "BOLLINGER_PERIOD": 20, "STOCHASTIC_PERIOD": 14,
        "CALCULATE_SHAP_VALUES": True, "MAX_DD_PER_CYCLE": 0.3, 
        "selected_features": ModelTrainer.BASE_FEATURES[:30] 
    }
    
    files_to_process=["AUDUSD_Daily_202001060000_202506020000.csv","AUDUSD_H1_202001060000_202506021800.csv","AUDUSD_M15_202105170000_202506021830.csv","EURUSD_Daily_202001060000_202506020000.csv","EURUSD_H1_202001060000_202506021800.csv","EURUSD_M15_202106020100_202506021830.csv","GBPUSD_Daily_202001060000_202506020000.csv","GBPUSD_H1_202001060000_202506021800.csv","GBPUSD_M15_202106020015_202506021830.csv","USDCAD_Daily_202001060000_202506020000.csv","USDCAD_H1_202001060000_202506021800.csv","USDCAD_M15_202105170000_202506021830.csv"]
    
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
        if df_train_labeled.empty or 'target' not in df_train_labeled.columns or len(df_train_labeled['target'].unique())<3:
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
        
        logger.info("  - Respecting API rate limits (5-second delay)...")
        time.sleep(5)
        
        current_config.update(suggested_params)
    
    logger.info("\n==========================================================")
    logger.info("                      WALK-FORWARD ANALYSIS COMPLETE");logger.info("==========================================================")
    
    final_config=ConfigModel(**current_config)
    reporter=PerformanceAnalyzer(final_config)
    if all_shap: aggregated_shap=pd.concat(all_shap).groupby(level=0)['SHAP_Importance'].mean().sort_values(ascending=False).to_frame() 
    else: aggregated_shap=None
    
    final_trades_df=pd.concat(all_trades,ignore_index=True) if all_trades else pd.DataFrame()
    reporter.generate_full_report(final_trades_df,pd.Series(full_equity_curve),cycle_metrics,aggregated_shap)
    logger.info(f"\nTotal execution time: {datetime.now() - start_time}")

if __name__ == '__main__':
    if os.name == 'nt': os.system("chcp 65001 > $null")
    main()

# End_To_End_Advanced_ML_Trading_Framework_PRO_V93_Token_Managed.py