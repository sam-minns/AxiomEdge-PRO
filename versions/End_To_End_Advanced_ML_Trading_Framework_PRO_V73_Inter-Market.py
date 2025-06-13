# End_To_End_Advanced_ML_Trading_Framework_PRO_V73_Inter-Market
#
# V73 Update:
# 1. NEW - Inter-Market Analysis: The framework now analyzes all pairs collectively
#    to generate powerful new cross-sectional features.
#    - "Relative Strength" features calculate how a pair's return compares to the average of its peers.
#    - "Rolling Correlation" features track the changing correlation between different pairs over time.
# 2. ADDED - Session-Based Features: As recommended, the model is now aware of the
#    London, New York, and Tokyo trading sessions and the high-volume overlap period.
# 3. ENHANCED - Backtester Logic: The backtester is now fully multi-asset aware, tracking
#    open positions on a per-symbol basis, allowing for concurrent trades across different pairs.
# 4. CONFIG - The file list in `raw_config` is updated to include all four currency pairs.

# --- FRAMEWORK DISCLAIMER ---
# This script is a high-level, bar-based simulation for ML strategy development.
# ALWAYS perform final validation of any promising model in MQL5.

# Standard Library
import os
import json
import warnings
import logging
from datetime import datetime
from logging.handlers import RotatingFileHandler
from typing import List, Dict, Any, Optional, Tuple

# Third-Party Libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import shap
import xgboost as xgb
import optuna

# Scikit-learn Modules
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import RobustScaler
from sklearn.utils.class_weight import compute_class_weight

# Imbalanced-learn
from imblearn.over_sampling import RandomOverSampler

# Pydantic for Config Validation
from pydantic import BaseModel, DirectoryPath, confloat, conint

# --- SETUP & INITIALIZATION ---
warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', category=UserWarning)
warnings.filterwarnings('ignore', category=pd.errors.PerformanceWarning)

# =============================================================================
# 1. LOGGING SETUP
# =============================================================================
def setup_logging() -> logging.Logger:
    """Sets up a robust logger for the application."""
    logger = logging.getLogger("ML_Trading_Framework")
    if not logger.handlers:
        logger.setLevel(logging.DEBUG)
        ch = logging.StreamHandler()
        ch.setLevel(logging.INFO)
        # Per your request, set up PowerShell console output encoding
        os.system("chcp 65001 > $null")
        fh = RotatingFileHandler(
            'trading_framework_adaptive.log',
            maxBytes=10*1024*1024,
            backupCount=5
        )
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
# 2. CONFIGURATION & VALIDATION
# =============================================================================
class ConfigModel(BaseModel):
    """Configuration model validated by Pydantic."""
    BASE_PATH: DirectoryPath
    DATA_FILENAMES: List[str]
    REPORT_LABEL: str
    FORWARD_TEST_START_DATE: str
    INITIAL_CAPITAL: confloat(gt=0)
    CONFIDENCE_TIERS: Dict[str, Dict[str, float]]
    BASE_RISK_PER_TRADE_PCT: confloat(gt=0, lt=1)
    SPREAD_PCTG_OF_ATR: confloat(ge=0)
    SLIPPAGE_PCTG_OF_ATR: confloat(ge=0)
    OPTUNA_TRIALS: conint(gt=0)
    TRAINING_WINDOW: str
    RETRAINING_FREQUENCY: str
    LOOKAHEAD_CANDLES: conint(gt=0)
    MODEL_SAVE_PATH: str = ""
    PLOT_SAVE_PATH: str = ""
    REPORT_SAVE_PATH: str = ""
    SHAP_PLOT_PATH: str = ""
    MIN_CLASS_SAMPLES: conint(gt=10) = 50
    MIN_CLASS_RATIO: confloat(gt=0.01, lt=1) = 0.1
    DYNAMIC_BARRIER_ADJUSTMENT: bool = True
    MAX_BARRIER_ADJUSTMENT_ITERATIONS: conint(gt=0) = 5
    TREND_FILTER_THRESHOLD: confloat(gt=0) = 25.0
    BOLLINGER_PERIOD: conint(gt=0) = 20
    STOCHASTIC_PERIOD: conint(gt=0) = 14

    def __init__(self, **data: Any):
        super().__init__(**data)
        self.MODEL_SAVE_PATH = f"{self.REPORT_LABEL}_model.json"
        self.PLOT_SAVE_PATH = f"{self.REPORT_LABEL}_equity_curve.png"
        self.REPORT_SAVE_PATH = f"{self.REPORT_LABEL}_quantitative_report.txt"
        self.SHAP_PLOT_PATH = f"{self.REPORT_LABEL}_shap_summary.png"

raw_config = {
    "BASE_PATH": os.getcwd(),
    "DATA_FILENAMES": [
        "AUDUSD_Daily_202001060000_202506020000.csv", "AUDUSD_H1_202001060000_202506021800.csv", "AUDUSD_M15_202105170000_202506021830.csv",
        "EURUSD_Daily_202001060000_202506020000.csv", "EURUSD_H1_202001060000_202506021800.csv", "EURUSD_M15_202106020100_202506021830.csv",
        "GBPUSD_Daily_202001060000_202506020000.csv", "GBPUSD_H1_202001060000_202506021800.csv", "GBPUSD_M15_202106020015_202506021830.csv",
        "USDCAD_Daily_202001060000_202506020000.csv", "USDCAD_H1_202001060000_202506021800.csv", "USDCAD_M15_202105170000_202506021830.csv"
    ],
    "REPORT_LABEL": "ML_Framework_V73_Inter-Market",
    "FORWARD_TEST_START_DATE": "2024-01-01",
    "INITIAL_CAPITAL": 100000.0,
    "CONFIDENCE_TIERS": {
        'ultra_high': {'min': 0.80, 'risk_mult': 1.2, 'size_mult': 1.5, 'rr': 3.0},
        'high': {'min': 0.70, 'risk_mult': 1.0, 'size_mult': 1.0, 'rr': 2.5},
        'standard': {'min': 0.60, 'risk_mult': 0.8, 'size_mult': 0.5, 'rr': 2.0}
    },
    "BASE_RISK_PER_TRADE_PCT": 0.01,
    "SPREAD_PCTG_OF_ATR": 0.05,
    "SLIPPAGE_PCTG_OF_ATR": 0.02,
    "OPTUNA_TRIALS": 50,
    "TRAINING_WINDOW": '365D',
    "RETRAINING_FREQUENCY": '60D',
    "LOOKAHEAD_CANDLES": 100,
    "MIN_CLASS_SAMPLES": 50,
    "MIN_CLASS_RATIO": 0.1,
    "DYNAMIC_BARRIER_ADJUSTMENT": True,
    "MAX_BARRIER_ADJUSTMENT_ITERATIONS": 5,
    "TREND_FILTER_THRESHOLD": 25.0,
    "BOLLINGER_PERIOD": 20,
    "STOCHASTIC_PERIOD": 14
}

# =============================================================================
# 3. DATA LOADER
# =============================================================================
class DataLoader:
    def __init__(self, config: ConfigModel):
        self.config = config

    def load_and_parse_data(self) -> Optional[Dict[str, pd.DataFrame]]:
        logger.info("-> Stage 1: Loading and Preparing Multi-Timeframe Data...")
        data_by_tf: Dict[str, List[pd.DataFrame]] = {'D1': [], 'H1': [], 'M15': []}
        symbols_found = set()

        for filename in self.config.DATA_FILENAMES:
            file_path = os.path.join(self.config.BASE_PATH, filename)
            if not os.path.exists(file_path):
                logger.warning(f"  - File not found, skipping: {file_path}")
                continue
            try:
                parts = filename.split('_')
                symbol, tf_str = parts[0], parts[1]
                symbols_found.add(symbol)
                tf = 'D1' if 'Daily' in tf_str else tf_str
                if tf not in data_by_tf: continue

                df = pd.read_csv(file_path, delimiter='\t' if '\t' in open(file_path).readline() else ',')
                df.columns = [c.upper().replace('<', '').replace('>', '') for c in df.columns]
                date_col = next((c for c in df.columns if 'DATE' in c), None)
                time_col = next((c for c in df.columns if 'TIME' in c), None)
                if date_col and time_col: df['Timestamp'] = pd.to_datetime(df[date_col] + ' ' + df[time_col], errors='coerce')
                elif date_col: df['Timestamp'] = pd.to_datetime(df[date_col], errors='coerce')
                else: raise ValueError("No date/time columns found.")
                df.dropna(subset=['Timestamp'], inplace=True)
                df.set_index('Timestamp', inplace=True)
                col_map = {c: c.capitalize() for c in df.columns if c.lower() in ['open', 'high', 'low', 'close', 'tickvol', 'volume', 'spread']}
                df.rename(columns=col_map, inplace=True)
                vol_col = 'Volume' if 'Volume' in df.columns else 'Tickvol'
                df.rename(columns={vol_col: 'RealVolume'}, inplace=True, errors='ignore')
                if 'RealVolume' not in df.columns: df['RealVolume'] = 0
                df['Symbol'] = symbol
                data_by_tf[tf].append(df)
            except Exception as e:
                logger.error(f"  - Failed to load {filename}: {e}", exc_info=True)

        processed_dfs: Dict[str, pd.DataFrame] = {}
        for tf, dfs in data_by_tf.items():
            if dfs:
                combined = pd.concat(dfs)
                all_symbols_df = [
                    df[~df.index.duplicated(keep='first')].sort_index()
                    for symbol, df in combined.groupby('Symbol')
                ]
                final_combined = pd.concat(all_symbols_df).sort_index()
                final_combined['RealVolume'] = pd.to_numeric(final_combined['RealVolume'], errors='coerce').fillna(0)
                processed_dfs[tf] = final_combined
                logger.info(f"  - Processed {tf}: {len(final_combined):,} rows for {len(final_combined['Symbol'].unique())} symbols.")
            else:
                logger.warning(f"  - No data found for {tf} timeframe.")
                processed_dfs[tf] = pd.DataFrame()

        if not processed_dfs or any(df.empty for df in processed_dfs.values()):
            logger.critical("  - Data loading failed. No data to process.")
            return None
        logger.info("[SUCCESS] Data loading and preparation complete.")
        return processed_dfs

# =============================================================================
# 4. FEATURE ENGINEERING
# =============================================================================
class FeatureEngineer:
    def __init__(self, config: ConfigModel):
        self.config = config

    def _calculate_adx(self, g: pd.DataFrame, period: int) -> pd.DataFrame:
        df = g.copy()
        alpha = 1 / period
        df['tr'] = pd.concat([df['High'] - df['Low'], abs(df['High'] - df['Close'].shift()), abs(df['Low'] - df['Close'].shift())], axis=1).max(axis=1)
        df['dm_plus'] = ((df['High'] - df['High'].shift()) > (df['Low'].shift() - df['Low'])).astype(int) * (df['High'] - df['High'].shift()).clip(lower=0)
        df['dm_minus'] = ((df['Low'].shift() - df['Low']) > (df['High'] - df['High'].shift())).astype(int) * (df['Low'].shift() - df['Low']).clip(lower=0)
        atr_adx = df['tr'].ewm(alpha=alpha, adjust=False).mean()
        di_plus = 100 * (df['dm_plus'].ewm(alpha=alpha, adjust=False).mean() / atr_adx.replace(0, 1e-9))
        di_minus = 100 * (df['dm_minus'].ewm(alpha=alpha, adjust=False).mean() / atr_adx.replace(0, 1e-9))
        dx = 100 * (abs(di_plus - di_minus) / (di_plus + di_minus).replace(0, 1e-9))
        g['ADX'] = dx.ewm(alpha=alpha, adjust=False).mean()
        return g

    def _calculate_bollinger_bands(self, g: pd.DataFrame, period: int) -> pd.DataFrame:
        rolling_close = g['Close'].rolling(window=period)
        middle_band = rolling_close.mean()
        std_dev = rolling_close.std()
        g['bb_width'] = ((middle_band + (std_dev * 2)) - (middle_band - (std_dev * 2))) / middle_band.replace(0, np.nan)
        return g

    def _calculate_stochastic(self, g: pd.DataFrame, period: int) -> pd.DataFrame:
        low_min = g['Low'].rolling(window=period).min()
        high_max = g['High'].rolling(window=period).max()
        g['stoch_k'] = 100 * (g['Close'] - low_min) / (high_max - low_min).replace(0, np.nan)
        g['stoch_d'] = g['stoch_k'].rolling(window=3).mean()
        return g
    
    def _calculate_cross_pair_features(self, df: pd.DataFrame) -> pd.DataFrame:
        logger.info("    - Engineering new cross-pair features (relative strength, correlation)...")
        df_pivot = df.pivot(columns='Symbol', values='Close').ffill().bfill()
        returns = df_pivot.pct_change().fillna(0)
        
        # Relative Strength
        mean_returns = returns.mean(axis=1)
        for symbol in returns.columns:
            df.loc[df['Symbol'] == symbol, 'rel_strength'] = (returns[symbol] - mean_returns)

        # Rolling Correlation
        symbols = list(returns.columns)
        if len(symbols) > 1:
            corr_target = symbols[0] # e.g., EURUSD
            for symbol in symbols[1:]:
                rolling_corr = returns[corr_target].rolling(window=100).corr(returns[symbol])
                # Merge back carefully
                corr_df = pd.DataFrame({'Timestamp': df_pivot.index, f'corr_{corr_target}_{symbol}': rolling_corr}).dropna()
                df = pd.merge(df, corr_df, on='Timestamp', how='left')
        
        df.bfill(inplace=True)
        return df

    def create_feature_stack(self, data_by_tf: Dict[str, pd.DataFrame]) -> pd.DataFrame:
        logger.info("-> Stage 2: Engineering Features from Multi-Timeframe Data...")
        if any(df.empty for df in data_by_tf.values()):
            logger.critical("  - One or more timeframes have no data. Cannot create feature stack.")
            return pd.DataFrame()
            
        df_d1 = self._calculate_htf_features(data_by_tf['D1'], 'D1', 20, 14)
        df_h1 = self._calculate_htf_features(data_by_tf['H1'], 'H1', 50, 14)

        df_m15_base = pd.concat([
            self._calculate_m15_native(group) for _, group in data_by_tf['M15'].groupby('Symbol')
        ])

        df_merged = pd.merge_asof(df_m15_base.sort_values('Timestamp'), df_h1, on='Timestamp', by='Symbol', direction='backward')
        df_merged = pd.merge_asof(df_merged.sort_values('Timestamp'), df_d1, on='Timestamp', by='Symbol', direction='backward')
        
        df_merged.reset_index(inplace=True)
        df_merged.set_index('Timestamp', inplace=True)
        
        # --- Cross-Pair Feature Engineering ---
        df_with_cross_feats = self._calculate_cross_pair_features(df_merged)

        df_with_cross_feats['hour'] = df_with_cross_feats.index.hour
        df_with_cross_feats['day_of_week'] = df_with_cross_feats.index.dayofweek
        df_with_cross_feats['adx_x_h1_trend'] = df_with_cross_feats['ADX'] * df_with_cross_feats['H1_ctx_H1_Trend']
        df_with_cross_feats['atr_x_d1_trend'] = df_with_cross_feats['ATR'] * df_with_cross_feats['D1_ctx_D1_Trend']
        df_with_cross_feats['market_regime'] = np.where(df_with_cross_feats['ADX'] > self.config.TREND_FILTER_THRESHOLD, 1, 0)
        
        df_final = df_with_cross_feats.dropna()
        logger.info(f"  - Merged data and created features. Final dataset shape: {df_final.shape}")
        logger.info("[SUCCESS] Feature engineering complete.")
        return df_final

    def _apply_lagged_features(self, g: pd.DataFrame, p: str, s: int, a: int) -> pd.DataFrame:
        g_out = pd.DataFrame(index=g.index)
        g_out[f'{p}_SMA'] = g['Close'].rolling(s).mean()
        g_out[f'{p}_ATR'] = (g['High'] - g['Low']).rolling(a).mean()
        g_out[f'{p}_Trend'] = np.sign(g['Close'] - g_out[f'{p}_SMA'])
        return g_out.shift(1)

    def _calculate_htf_features(self, df: pd.DataFrame, p: str, s: int, a: int) -> pd.DataFrame:
        # Group by symbol and apply feature calculation
        df_out = pd.concat([self._apply_lagged_features(group, p, s, a) for _, group in df.groupby('Symbol')]).reset_index()
        f_cols = [f'{p}_SMA', f'{p}_ATR', f'{p}_Trend']
        return pd.concat([df_out[['Timestamp', 'Symbol']], df_out[f_cols].add_prefix(f'{p}_ctx_')], axis=1)

    def _calculate_m15_native(self, g: pd.DataFrame) -> pd.DataFrame:
        logger.info(f"    - Calculating native M15 features for {g['Symbol'].iloc[0]}...")
        g_out = g.copy()
        lookback, rank_period, sma_period = 14, 100, 50
        
        g_out['ATR'] = (g['High'] - g['Low']).rolling(lookback).mean()
        delta = g['Close'].diff()
        gain = delta.where(delta > 0, 0).ewm(com=lookback - 1, adjust=False).mean()
        loss = -delta.where(delta < 0, 0).ewm(com=lookback - 1, adjust=False).mean()
        g_out['RSI'] = 100 - (100 / (1 + (gain / loss.replace(0, 1e-9))))
        g_out = self._calculate_adx(g_out, lookback)
        g_out = self._calculate_bollinger_bands(g_out, self.config.BOLLINGER_PERIOD)
        g_out = self._calculate_stochastic(g_out, self.config.STOCHASTIC_PERIOD)
        
        sma_50 = g_out['Close'].rolling(window=sma_period).mean()
        g_out['dist_from_sma50'] = (g_out['Close'] - sma_50) / g_out['ATR'].replace(0, np.nan)
        direction = np.sign(g_out['Close'].diff())
        g_out['consecutive_candles'] = direction.groupby((direction != direction.shift()).cumsum()).cumcount() + 1
        
        # --- ADDED: Session-Based Features ---
        london_open, london_close, ny_open, ny_close, tokyo_open, tokyo_close = 8, 16, 13, 21, 0, 8
        g_out['is_london'] = ((g_out.index.hour >= london_open) & (g_out.index.hour < london_close)).astype(int)
        g_out['is_ny'] = ((g_out.index.hour >= ny_open) & (g_out.index.hour < ny_close)).astype(int)
        g_out['is_tokyo'] = ((g_out.index.hour >= tokyo_open) & (g_out.index.hour < tokyo_close)).astype(int)
        g_out['is_london_ny_overlap'] = ((g_out.index.hour >= ny_open) & (g_out.index.hour < london_close)).astype(int)
        for session, open_h in [('london', london_open), ('ny', ny_open), ('tokyo', tokyo_open)]:
            g_out[f'candles_into_{session}'] = ((g_out.index.hour - open_h) * g_out[f'is_{session}']).clip(lower=0)
        
        # Shift all calculated features to prevent lookahead bias
        feature_cols = ['ATR', 'RSI', 'ADX', 'bb_width', 'stoch_k', 'stoch_d', 'dist_from_sma50', 'consecutive_candles',
                        'is_london', 'is_ny', 'is_tokyo', 'is_london_ny_overlap', 'candles_into_london', 'candles_into_ny', 'candles_into_tokyo']
        g_out[feature_cols] = g_out[feature_cols].shift(1)
        
        # Rank features after shifting
        for col in ['ATR', 'RSI', 'bb_width', 'stoch_k']: g_out[f'{col}_rank'] = g_out[col].rolling(rank_period).rank(pct=True)
        
        return g_out

    def label_outcomes(self, df: pd.DataFrame, sl_atr: float, tp_atr: float, lookahead: int) -> pd.DataFrame:
        logger.info("  - Generating trade labels using Triple-Barrier Method...")
        # Labeled data should be per symbol
        labeled_dfs = [
            self._label_with_dynamic_barriers(group, sl_atr, tp_atr, lookahead, self.config.MIN_CLASS_RATIO, self.config.MAX_BARRIER_ADJUSTMENT_ITERATIONS)
            for _, group in df.groupby('Symbol')
        ]
        return pd.concat(labeled_dfs)

    def _label_with_dynamic_barriers(self, df: pd.DataFrame, initial_sl_atr: float, initial_tp_atr: float, lookahead: int, min_class_ratio: float, max_iterations: int) -> pd.DataFrame:
        best_df, best_balance, best_params = df.copy(), 0, (initial_sl_atr, initial_tp_atr)
        logger.info(f"    - Adjusting barriers for {df['Symbol'].iloc[0]}...")
        for i in range(max_iterations):
            current_sl, current_tp = initial_sl_atr * (0.8 + 0.1 * i), initial_tp_atr * (0.8 + 0.1 * i)
            labeled_df = self._label_group(df.copy(), sl_atr=current_sl, tp_atr=current_tp, lookahead=lookahead)
            if 'target' not in labeled_df.columns: continue
            class_counts = labeled_df['target'].value_counts()
            if len(class_counts) < 3: continue
            current_ratio = class_counts.min() / class_counts.max() if class_counts.max() > 0 else 0
            if current_ratio > best_balance:
                best_balance, best_df, best_params = labeled_df, current_ratio, (current_sl, current_tp)
        logger.info(f"    - Best barriers for {df['Symbol'].iloc[0]}: SL={best_params[0]:.2f}, TP={best_params[1]:.2f} -> Ratio={best_balance:.2f}")
        return best_df

    def _label_group(self, group: pd.DataFrame, sl_atr: float, tp_atr: float, lookahead: int) -> pd.DataFrame:
        if len(group) < lookahead + 1: return group
        outcomes = np.zeros(len(group))
        prices, lows, highs, atr_vals = group['Close'].values, group['Low'].values, group['High'].values, group['ATR'].values
        for i in range(len(group) - lookahead):
            atr = atr_vals[i]
            if pd.isna(atr) or atr <= 1e-9: continue
            tp_long, sl_long = prices[i] + atr * tp_atr, prices[i] - atr * sl_atr
            future_highs, future_lows = highs[i+1 : i+1+lookahead], lows[i+1 : i+1+lookahead]
            time_to_tp = np.where(future_highs >= tp_long)[0]
            time_to_sl = np.where(future_lows <= sl_long)[0]
            first_tp = time_to_tp[0] if len(time_to_tp) > 0 else np.inf
            first_sl = time_to_sl[0] if len(time_to_sl) > 0 else np.inf
            if first_tp < first_sl: outcomes[i] = 1
            elif first_sl < first_tp: outcomes[i] = -1
        group['target'] = outcomes
        return group

# =============================================================================
# 5. MODEL TRAINER
# =============================================================================
class ModelTrainer:
    FEATURES = [
        'ATR', 'RSI', 'ADX', 'ATR_rank', 'RSI_rank', 'hour', 'day_of_week', 
        'adx_x_h1_trend', 'atr_x_d1_trend', 'H1_ctx_H1_Trend', 'D1_ctx_D1_Trend', 
        'bb_width', 'stoch_k', 'stoch_d', 'bb_width_rank', 'stoch_k_rank', 
        'market_regime', 'dist_from_sma50', 'consecutive_candles',
        'is_london', 'is_ny', 'is_tokyo', 'is_london_ny_overlap',
        'candles_into_london', 'candles_into_ny', 'candles_into_tokyo',
        'rel_strength', 'corr_EURUSD_GBPUSD', 'corr_EURUSD_USDCAD' # Example correlations
    ]

    def __init__(self, config: ConfigModel):
        self.config = config
        self.shap_summary: Optional[pd.DataFrame] = None
        self.class_weights: Optional[Dict[int, float]] = None

    def train(self, df_train: pd.DataFrame) -> Optional[Pipeline]:
        logger.info(f"  - Starting model training for period {df_train.index.min().date()} to {df_train.index.max().date()}...")
        y_map = {-1: 0, 0: 1, 1: 2}
        y = df_train['target'].map(y_map).astype(int)
        
        # Ensure all feature columns exist, fill missing with 0
        feature_cols = [f for f in self.FEATURES if f in df_train.columns]
        X = df_train[feature_cols].copy().fillna(0)

        if min_ratio < self.config.MIN_CLASS_RATIO or class_counts.min() < self.config.MIN_CLASS_SAMPLES:
            logger.info(f"    - Classes are imbalanced. Applying RandomOverSampler.")
            X, y = RandomOverSampler(random_state=42).fit_resample(X, y)

        self.class_weights = dict(zip(np.unique(y), compute_class_weight(class_weight='balanced', classes=np.unique(y), y=y)))
        X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, shuffle=False)
        study = self._optimize_hyperparameters(X_train, y_train, X_val, y_val)
        if not study or not study.best_trials:
            logger.error("  - Training aborted: Hyperparameter optimization failed.")
            return None
        logger.info(f"    - Optimization complete. Best F1 Score: {study.best_value:.4f}")
        final_pipeline = self._train_final_model(study.best_params, X, y)
        logger.info("  - [SUCCESS] Model training complete.")
        return final_pipeline

    def _optimize_hyperparameters(self, X_train, y_train, X_val, y_val) -> Optional[optuna.study.Study]:
        # (Implementation is the same as V72, omitted for brevity)
        pass

    def _train_final_model(self, best_params: Dict, X: pd.DataFrame, y: pd.Series) -> Optional[Pipeline]:
        # (Implementation is the same as V72, omitted for brevity)
        pass

    def _generate_shap_summary(self, model: xgb.XGBClassifier, X_scaled: np.ndarray, feature_names: pd.Index):
        # (Implementation is the same as V72, omitted for brevity)
        pass

# =============================================================================
# 6. BACKTESTER
# =============================================================================
class Backtester:
    def __init__(self, config: ConfigModel):
        self.config = config

    def run_backtest_chunk(self, df_chunk_in: pd.DataFrame, model: Pipeline, initial_equity: float) -> Tuple[pd.DataFrame, pd.Series]:
        if df_chunk_in.empty: return pd.DataFrame(), pd.Series([initial_equity])
        df_chunk = df_chunk_in.copy()
        X_test = df_chunk[ModelTrainer.FEATURES].copy().fillna(0)
        
        class_probs = model.predict_proba(X_test)
        df_chunk['prob_short'], df_chunk['prob_hold'], df_chunk['prob_long'] = class_probs[:, 0], class_probs[:, 1], class_probs[:, 2]

        trades, equity, equity_curve, open_positions = [], initial_equity, [initial_equity], {}

        candles = df_chunk.reset_index().to_dict('records')

        for candle in candles:
            symbol = candle['Symbol']
            
            if symbol in open_positions:
                pos = open_positions[symbol]
                pnl, exit_price = 0, None
                if pos['direction'] == 1:
                    if candle['Low'] <= pos['sl']: pnl, exit_price = -pos['risk_amt'], pos['sl']
                    elif candle['High'] >= pos['tp']: pnl, exit_price = pos['risk_amt'] * pos['rr'], pos['tp']
                elif pos['direction'] == -1:
                    if candle['High'] >= pos['sl']: pnl, exit_price = -pos['risk_amt'], pos['sl']
                    elif candle['Low'] <= pos['tp']: pnl, exit_price = pos['risk_amt'] * pos['rr'], pos['tp']
                
                if exit_price:
                    equity += pnl
                    if equity <= 0: logger.critical("  - ACCOUNT BLOWN! Equity is zero or negative."); break
                    equity_curve.append(equity)
                    trades.append({'ExecTime': candle['Timestamp'],'Symbol': symbol,'PNL': pnl,'Equity': equity,'Confidence': pos['confidence'],'Direction': pos['direction']})
                    del open_positions[symbol]

            if symbol not in open_positions:
                pred_class = np.argmax([candle['prob_short'], candle['prob_hold'], candle['prob_long']])
                direction = 1 if pred_class == 2 else -1 if pred_class == 0 else 0
                confidence = candle[f"prob_{'long' if direction == 1 else 'short'}"] if direction != 0 else 0
                
                min_confidence = 0.45 if candle.get('market_regime', 0) == 1 else 0.60
                
                if direction != 0 and confidence >= min_confidence:
                    atr = candle['ATR']
                    if pd.isna(atr) or atr <= 1e-9: continue
                    
                    tier_name = 'standard'
                    if confidence >= self.config.CONFIDENCE_TIERS['ultra_high']['min']: tier_name = 'ultra_high'
                    elif confidence >= self.config.CONFIDENCE_TIERS['high']['min']: tier_name = 'high'
                        
                    tier = self.config.CONFIDENCE_TIERS[tier_name]
                    risk_amt = equity * self.config.BASE_RISK_PER_TRADE_PCT * tier['risk_mult']
                    
                    spread_cost = atr * self.config.SPREAD_PCTG_OF_ATR
                    slippage_cost = atr * self.config.SLIPPAGE_PCTG_OF_ATR
                    total_costs = spread_cost + slippage_cost
                    sl_dist = (atr * 1.5) + total_costs
                    tp_dist = (atr * 1.5 * tier['rr']) - total_costs
                    
                    if tp_dist <= 0: continue

                    entry_price = candle['Close']
                    sl_price, tp_price = entry_price - sl_dist * direction, entry_price + tp_dist * direction
                    
                    open_positions[symbol] = {'direction': direction, 'sl': sl_price, 'tp': tp_price, 'risk_amt': risk_amt, 'rr': tier['rr'], 'confidence': confidence}
                    
        return pd.DataFrame(trades), pd.Series(equity_curve)

# =============================================================================
# 7. PERFORMANCE ANALYZER
# =============================================================================
class PerformanceAnalyzer:
    def __init__(self, config: ConfigModel):
        self.config = config

    def generate_full_report(self, trades_df: Optional[pd.DataFrame], equity_curve: Optional[pd.Series], cycle_metrics: List[Dict], aggregated_shap: Optional[pd.DataFrame] = None):
        if trades_df is None or trades_df.empty or equity_curve is None or len(equity_curve) < 2:
            logger.warning("-> No trades were generated. Skipping final report.")
            self.generate_text_report({}, cycle_metrics, aggregated_shap)
            return

        logger.info("-> Stage 4: Generating Final Performance Report...")
        self.plot_equity_curve(equity_curve)
        if aggregated_shap is not None: self.plot_shap_summary(aggregated_shap)
            
        metrics = self._calculate_metrics(trades_df, equity_curve)
        self.generate_text_report(metrics, cycle_metrics, aggregated_shap)
        logger.info("[SUCCESS] Final report generated and saved.")

    def plot_equity_curve(self, equity_curve: pd.Series):
        plt.style.use('seaborn-v0_8-darkgrid')
        plt.figure(figsize=(16, 8))
        plt.plot(equity_curve.values, color='dodgerblue', linewidth=2)
        plt.title(f"{self.config.REPORT_LABEL} - Walk-Forward Equity Curve", fontsize=16, weight='bold')
        plt.xlabel("Trade Number", fontsize=12)
        plt.ylabel("Equity ($)", fontsize=12)
        plt.grid(True, which='both', linestyle=':')
        plt.savefig(self.config.PLOT_SAVE_PATH)
        plt.close()
        logger.info(f"  - Equity curve plot saved to: {self.config.PLOT_SAVE_PATH}")

    def plot_shap_summary(self, shap_summary: pd.DataFrame):
        plt.style.use('seaborn-v0_8-darkgrid')
        plt.figure(figsize=(12, 10))
        shap_summary.head(20).sort_values(by='SHAP_Importance').plot(kind='barh', legend=False, color='mediumseagreen')
        plt.title(f"{self.config.REPORT_LABEL} - Aggregated Feature Importance (SHAP)", fontsize=16, weight='bold')
        plt.xlabel("Mean Absolute SHAP Value", fontsize=12)
        plt.ylabel("Feature", fontsize=12)
        plt.tight_layout()
        plt.savefig(self.config.SHAP_PLOT_PATH)
        plt.close()
        logger.info(f"  - SHAP summary plot saved to: {self.config.SHAP_PLOT_PATH}")
        
    def _calculate_metrics(self, trades_df: pd.DataFrame, equity_curve: pd.Series) -> Dict[str, Any]:
        m = {'initial_capital': self.config.INITIAL_CAPITAL, 'ending_capital': equity_curve.iloc[-1]}
        m['net_profit'] = m['ending_capital'] - m['initial_capital']
        m['net_profit_pct'] = m['net_profit'] / m['initial_capital'] if m['initial_capital'] > 0 else 0
        wins, losses = trades_df[trades_df['PNL'] > 0], trades_df[trades_df['PNL'] <= 0]
        m['gross_profit'], m['gross_loss'] = wins['PNL'].sum(), abs(losses['PNL'].sum())
        m['profit_factor'] = m['gross_profit'] / m['gross_loss'] if m['gross_loss'] > 0 else np.inf
        m['total_trades'], m['winning_trades'], m['losing_trades'] = len(trades_df), len(wins), len(losses)
        m['win_rate'] = m['winning_trades'] / m['total_trades'] if m['total_trades'] > 0 else 0
        m['avg_win'] = wins['PNL'].mean() if len(wins) > 0 else 0
        m['avg_loss'] = losses['PNL'].mean() if len(losses) > 0 else 0
        m['expectancy'] = (m['win_rate'] * m['avg_win']) - ((1 - m['win_rate']) * abs(m['avg_loss'])) if m['total_trades'] > 0 else 0
        running_max = equity_curve.cummax()
        drawdown_abs = running_max - equity_curve
        m['max_drawdown_abs'] = drawdown_abs.max() if not drawdown_abs.empty else 0
        m['max_drawdown_pct'] = (drawdown_abs / running_max).replace([np.inf, -np.inf], 0).max() * 100
        exec_times = pd.to_datetime(trades_df['ExecTime']).dt.tz_localize(None)
        years = (exec_times.max() - exec_times.min()).days / 365.25 if not trades_df.empty else 1
        m['cagr'] = ((m['ending_capital'] / m['initial_capital']) ** (1 / years) - 1) if years > 0 and m['initial_capital'] > 0 else 0
        pnl_std = trades_df['PNL'].std()
        m['sharpe_ratio'] = (trades_df['PNL'].mean() / pnl_std) * np.sqrt(252 * (24 * 4)) if pnl_std > 0 else 0
        m['mar_ratio'] = m['cagr'] / (m['max_drawdown_pct'] / 100) if m['max_drawdown_pct'] > 0 else np.inf
        return m

    def generate_text_report(self, m: Dict[str, Any], cycle_metrics: List[Dict], aggregated_shap: Optional[pd.DataFrame] = None):
        cycle_df = pd.DataFrame(cycle_metrics)
        cycle_report = "Per-Cycle Performance:\n" + cycle_df.to_string(index=False) if not cycle_df.empty else "No trades were executed in any cycle."
        shap_report = "Aggregated Feature Importance (SHAP):\n" + aggregated_shap.to_string() if aggregated_shap is not None else "SHAP summary was not generated."
        report = f"""
\n======================================================================
           ADAPTIVE WALK-FORWARD PERFORMANCE REPORT
======================================================================
Report Label: {self.config.REPORT_LABEL}
Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
Walk-Forward: Train Window={self.config.TRAINING_WINDOW}, Retrain Freq={self.config.RETRAINING_FREQUENCY}
----------------------------------------------------------------------
I. OVERALL PERFORMANCE
----------------------------------------------------------------------
Initial Capital:        ${m.get('initial_capital', self.config.INITIAL_CAPITAL):>15,.2f}
Ending Capital:         ${m.get('ending_capital', self.config.INITIAL_CAPITAL):>15,.2f}
Net Profit:             ${m.get('net_profit', 0):>15,.2f} ({m.get('net_profit_pct', 0):.2%})
----------------------------------------------------------------------
II. TRADE STATISTICS
----------------------------------------------------------------------
Total Trades:           {m.get('total_trades', 0):>15}
Win Rate:               {m.get('win_rate', 0):>15.2%}
Profit Factor:          {m.get('profit_factor', 0):>15.2f}
Expectancy (per trade): ${m.get('expectancy', 0):>15.2f}
----------------------------------------------------------------------
III. RISK AND RETURN
----------------------------------------------------------------------
Max Drawdown:           {m.get('max_drawdown_pct', 0):>15.2f}% (${m.get('max_drawdown_abs', 0):,.2f})
Annual Return (CAGR):   {m.get('cagr', 0):>15.2%}
Sharpe Ratio (annual):  {m.get('sharpe_ratio', 0):>15.2f}
MAR Ratio (CAGR/MDD):   {m.get('mar_ratio', 0):>15.2f}
----------------------------------------------------------------------
IV. WALK-FORWARD CYCLE BREAKDOWN
----------------------------------------------------------------------
{cycle_report}
----------------------------------------------------------------------
V. MODEL FEATURE IMPORTANCE
----------------------------------------------------------------------
{shap_report}
======================================================================
"""
        logger.info(report)
        try:
            with open(self.config.REPORT_SAVE_PATH, 'w', encoding='utf-8') as f:
                f.write(report)
            logger.info(f"  - Quantitative report saved to {self.config.REPORT_SAVE_PATH}")
        except IOError as e:
            logger.error(f"  - Failed to save text report: {e}", exc_info=True)

# =============================================================================
# 8. MAIN ORCHESTRATOR
# =============================================================================
def main():
    """Main orchestrator for the trading framework."""
    start_time = datetime.now()
    logger.info("==========================================================")
    logger.info("  STARTING END-TO-END MULTI-ASSET ML TRADING FRAMEWORK")
    logger.info("==========================================================")

    try:
        config = ConfigModel(**raw_config)
        logger.info(f"-> Configuration loaded for report: {config.REPORT_LABEL}")
    except Exception as e:
        logger.critical(f"[FATAL] Config error: {e}", exc_info=True)
        return

    data_by_tf = DataLoader(config).load_and_parse_data()
    if not data_by_tf: return

    fe = FeatureEngineer(config)
    df_featured = fe.create_feature_stack(data_by_tf)
    if df_featured.empty: return

    test_start_date = pd.to_datetime(config.FORWARD_TEST_START_DATE)
    max_date = df_featured.index.max()
    retraining_dates = pd.date_range(start=test_start_date, end=max_date, freq=config.RETRAINING_FREQUENCY)

    all_trades, full_equity_curve, cycle_metrics, all_shap = [], [config.INITIAL_CAPITAL], [], []
    
    logger.info("-> Stage 3: Starting Walk-Forward Analysis...")

    for i, period_start_date in enumerate(retraining_dates):
        logger.info(f"\n{'='*25} CYCLE {i + 1}/{len(retraining_dates)}: {period_start_date.date()} {'='*25}")
        train_end = period_start_date - pd.Timedelta(days=1)
        train_start = train_end - pd.Timedelta(config.TRAINING_WINDOW)
        test_end = (period_start_date + pd.Timedelta(config.RETRAINING_FREQUENCY)) - pd.Timedelta(days=1)
        if test_end > max_date: test_end = max_date

        df_train_raw = df_featured.loc[train_start:train_end]
        df_test_chunk = df_featured.loc[period_start_date:test_end]

        if df_train_raw.empty or df_test_chunk.empty:
            logger.warning("  - Skipping cycle due to insufficient data for training or testing period.")
            cycle_metrics.append({'Cycle': i + 1, 'StartDate': period_start_date.date(), 'NumTrades': 0, 'WinRate': "N/A", 'CyclePnL': "$0.00", 'Status': "Skipped (No Data)"})
            continue

        df_train_labeled = fe.label_outcomes(df_train_raw, sl_atr=1.5, tp_atr=3.0, lookahead=config.LOOKAHEAD_CANDLES)
        if df_train_labeled.empty or 'target' not in df_train_labeled.columns or df_train_labeled['target'].nunique() < 3:
            logger.warning("  - Skipping cycle: Could not generate valid labels with enough class diversity.")
            cycle_metrics.append({'Cycle': i + 1, 'StartDate': period_start_date.date(), 'NumTrades': 0, 'WinRate': "N/A", 'CyclePnL': "$0.00", 'Status': "Skipped (Label Error)"})
            continue

        trainer = ModelTrainer(config)
        model = trainer.train(df_train_labeled)
        
        if model is None:
            logger.error(f"  - Skipping cycle: Model training failed.")
            cycle_metrics.append({'Cycle': i + 1, 'StartDate': period_start_date.date(), 'NumTrades': 0, 'WinRate': "N/A", 'CyclePnL': "$0.00", 'Status': "Failed (Training Error)"})
            continue
        
        if trainer.shap_summary is not None: all_shap.append(trainer.shap_summary)

        logger.info(f"  - Backtesting on out-of-sample data from {period_start_date.date()} to {test_end.date()}...")
        backtester = Backtester(config)
        chunk_trades_df, chunk_equity = backtester.run_backtest_chunk(df_test_chunk, model, initial_equity=full_equity_curve[-1])

        if not chunk_trades_df.empty:
            all_trades.append(chunk_trades_df)
            full_equity_curve.extend(chunk_equity.iloc[1:].tolist())
            pnl, wr = chunk_trades_df['PNL'].sum(), (chunk_trades_df['PNL'] > 0).mean()
            logger.info(f"  - [SUCCESS] Cycle complete. Trades: {len(chunk_trades_df)}, PnL: ${pnl:,.2f}, Win Rate: {wr:.2%}")
            cycle_metrics.append({'Cycle': i + 1, 'StartDate': period_start_date.date(), 'NumTrades': len(chunk_trades_df), 'WinRate': f"{wr:.2%}", 'CyclePnL': f"${pnl:,.2f}", 'Status': "Success"})
        else:
            logger.info("  - Cycle complete. No trades were executed.")
            cycle_metrics.append({'Cycle': i + 1, 'StartDate': period_start_date.date(), 'NumTrades': 0, 'WinRate': "N/A", 'CyclePnL': "$0.00", 'Status': "No Trades"})

    logger.info("\n==========================================================")
    logger.info("               WALK-FORWARD ANALYSIS COMPLETE")
    logger.info("==========================================================")
    
    reporter = PerformanceAnalyzer(config)
    aggregated_shap = pd.concat(all_shap).groupby(level=0)['SHAP_Importance'].mean().sort_values(ascending=False).to_frame() if all_shap else None
    final_trades_df = pd.concat(all_trades, ignore_index=True) if all_trades else pd.DataFrame()
    reporter.generate_full_report(final_trades_df, pd.Series(full_equity_curve), cycle_metrics, aggregated_shap)
    
    logger.info(f"\nTotal execution time: {datetime.now() - start_time}")

if __name__ == '__main__':
    # Add PowerShell console encoding helper
    print("[Console]::OutputEncoding = [System.Text.Encoding]::UTF8")
    main()

# End_To_End_Advanced_ML_Trading_Framework_PRO_V73_Inter-Market.py