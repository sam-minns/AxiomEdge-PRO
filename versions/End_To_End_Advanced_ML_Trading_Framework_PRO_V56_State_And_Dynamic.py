# End_To_End_Advanced_ML_Trading_Framework_PRO_V56_State_And_Dynamic
#
# V56 Update:
# 1. This version combines the two most successful experimental strategies.
# 2. Implemented State-Based Features (is_oversold, is_trending, etc.) from V53 to improve model adaptability.
# 3. Implemented the Dynamic Confidence Threshold from V55 to intelligently manage risk.
# 4. Removed raw RSI/Stochastic features to force the model to learn from the more robust state information.

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
from scipy.stats import ks_2samp
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
# 2. CONFIGURATION & VALIDATION (Pydantic)
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
        "AUDUSD_Daily_202001060000_202506020000.csv",
        "AUDUSD_H1_202001060000_202506021800.csv",
        "AUDUSD_M15_202105170000_202506021830.csv",
        "EURUSD_Daily_202001060000_202506020000.csv",
        "EURUSD_H1_202001060000_202506021800.csv",
        "EURUSD_M15_202106020100_202506021830.csv",
        "GBPUSD_Daily_202001060000_202506020000.csv",
        "GBPUSD_H1_202001060000_202506021800.csv",
        "GBPUSD_M15_202106020015_202506021830.csv",
        "USDCAD_Daily_202001060000_202506020000.csv",
        "USDCAD_H1_202001060000_202506021800.csv",
        "USDCAD_M15_202105170000_202506021830.csv"
    ],
    "REPORT_LABEL": "ML_Framework_V56_State_And_Dynamic",
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
        logger.info("-> Stage 1: Loading and Preparing Data...")
        data_by_tf: Dict[str, List[pd.DataFrame]] = {'D1': [], 'H1': [], 'M15': []}

        for filename in self.config.DATA_FILENAMES:
            file_path = os.path.join(self.config.BASE_PATH, filename)
            if not os.path.exists(file_path):
                logger.warning(f"  - File not found, skipping: {file_path}")
                continue
            try:
                parts = filename.split('_')
                symbol, tf_str = parts[0], parts[1]
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
                if vol_col in df.columns: df.rename(columns={vol_col: 'RealVolume'}, inplace=True)
                else: df['RealVolume'] = 0

                df['Symbol'] = symbol
                data_by_tf[tf].append(df)

            except Exception as e:
                logger.error(f"  - Failed to load {filename}: {e}", exc_info=True)

        processed_dfs: Dict[str, pd.DataFrame] = {}
        for tf, dfs in data_by_tf.items():
            if dfs:
                combined = pd.concat(dfs).drop_duplicates()
                combined = combined[~combined.index.duplicated(keep='first')].sort_index()
                combined['RealVolume'] = pd.to_numeric(combined['RealVolume'], errors='coerce').fillna(0)
                processed_dfs[tf] = combined
                logger.info(f"  - Processed {tf}: {len(combined):,} rows for {len(combined['Symbol'].unique())} symbols.")
            else:
                logger.warning(f"  - No data found for {tf} timeframe.")
                processed_dfs[tf] = pd.DataFrame()

        if not processed_dfs or all(df.empty for df in processed_dfs.values()):
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
        df['tr0'] = abs(df['High'] - df['Low'])
        df['tr1'] = abs(df['High'] - df['Close'].shift(1))
        df['tr2'] = abs(df['Low'] - df['Close'].shift(1))
        df['TR'] = df[['tr0', 'tr1', 'tr2']].max(axis=1)
        df['dm_plus'] = ((df['High'] - df['High'].shift(1)) > (df['Low'].shift(1) - df['Low'])).astype(int) * (df['High'] - df['High'].shift(1))
        df['dm_minus'] = ((df['Low'].shift(1) - df['Low']) > (df['High'] - df['High'].shift(1))).astype(int) * (df['Low'].shift(1) - df['Low'])
        df['dm_plus'] = df['dm_plus'].clip(lower=0)
        df['dm_minus'] = df['dm_minus'].clip(lower=0)
        df['ATR_adx'] = df['TR'].ewm(alpha=alpha, adjust=False).mean()
        df['dm_plus_smooth'] = df['dm_plus'].ewm(alpha=alpha, adjust=False).mean()
        df['dm_minus_smooth'] = df['dm_minus'].ewm(alpha=alpha, adjust=False).mean()
        df['di_plus'] = (df['dm_plus_smooth'] / df['ATR_adx']) * 100
        df['di_minus'] = (df['dm_minus_smooth'] / df['ATR_adx']) * 100
        df['DX'] = (abs(df['di_plus'] - df['di_minus']) / abs(df['di_plus'] + df['di_minus'])).fillna(0) * 100
        df['ADX'] = df['DX'].ewm(alpha=alpha, adjust=False).mean()
        g['ADX'] = df['ADX']
        return g

    def _calculate_bollinger_bands(self, g: pd.DataFrame, period: int) -> pd.DataFrame:
        rolling_close = g['Close'].rolling(window=period)
        middle_band = rolling_close.mean()
        std_dev = rolling_close.std()
        g['bb_upper'] = (middle_band + (std_dev * 2))
        g['bb_middle'] = middle_band
        g['bb_lower'] = (middle_band - (std_dev * 2))
        g['bb_width'] = (g['bb_upper'] - g['bb_lower']) / g['bb_middle'].replace(0, np.nan)
        return g

    def _calculate_stochastic(self, g: pd.DataFrame, period: int) -> pd.DataFrame:
        low_min = g['Low'].rolling(window=period).min()
        high_max = g['High'].rolling(window=period).max()
        range_hh_ll = (high_max - low_min).replace(0, np.nan)
        fast_k = 100 * (g['Close'] - low_min) / range_hh_ll
        g['stoch_k'] = fast_k.rolling(window=3).mean()
        return g

    def create_feature_stack(self, data_by_tf: Dict[str, pd.DataFrame]) -> pd.DataFrame:
        logger.info("-> Stage 2: Engineering Features from Multi-Timeframe Data...")
        if data_by_tf['D1'].empty or data_by_tf['H1'].empty or data_by_tf['M15'].empty:
            logger.critical("  - One or more timeframes have no data. Cannot create feature stack.")
            return pd.DataFrame()
            
        df_d1 = self._calculate_htf_features(data_by_tf['D1'], 'D1', 20, 14)
        df_h1 = self._calculate_htf_features(data_by_tf['H1'], 'H1', 50, 14)

        df_m15_base = data_by_tf['M15'].copy().groupby('Symbol').apply(self._calculate_m15_native, include_groups=False).reset_index()

        df_merged = pd.merge_asof(df_m15_base.sort_values('Timestamp'), df_h1.sort_values('Timestamp'), on='Timestamp', by='Symbol', direction='backward')
        df_merged = pd.merge_asof(df_merged, df_d1.sort_values('Timestamp'), on='Timestamp', by='Symbol', direction='backward')
        df_merged.set_index('Timestamp', inplace=True)
        df_merged = df_merged.sort_index()

        # Shift all features by 1 to prevent lookahead bias
        feature_cols = [col for col in df_merged.columns if col not in ['Open', 'High', 'Low', 'Close', 'Symbol', 'RealVolume']]
        df_merged[feature_cols] = df_merged.groupby('Symbol')[feature_cols].shift(1)

        df_merged['hour'] = df_merged.index.hour
        df_merged['day_of_week'] = df_merged.index.dayofweek
        df_merged['adx_x_h1_trend'] = df_merged['ADX'] * df_merged['H1_ctx_H1_Trend']
        df_merged['atr_x_d1_trend'] = df_merged['ATR'] * df_merged['D1_ctx_D1_Trend']
        df_merged['market_regime'] = np.where(df_merged['ADX'] > self.config.TREND_FILTER_THRESHOLD, 1, 0)
        
        df_final = df_merged.dropna()
        logger.info(f"  - Merged data and created features. Final dataset shape: {df_final.shape}")
        logger.info("[SUCCESS] Feature engineering complete.")
        return df_final

    def _apply_lagged_features(self, g: pd.DataFrame, p: str, s: int, a: int) -> pd.DataFrame:
        g[f'{p}_SMA'] = g['Close'].rolling(s).mean()
        g[f'{p}_ATR'] = g['High'].sub(g['Low']).rolling(a).mean()
        g[f'{p}_Trend'] = np.sign(g['Close'] - g[f'{p}_SMA'])
        return g

    def _calculate_htf_features(self, df: pd.DataFrame, p: str, s: int, a: int) -> pd.DataFrame:
        df_out = df.groupby('Symbol').apply(self._apply_lagged_features, p, s, a, include_groups=False).reset_index()
        f_cols = [f'{p}_SMA', f'{p}_ATR', f'{p}_Trend']
        return pd.concat([df_out[['Timestamp', 'Symbol']], df_out[f_cols].add_prefix(f'{p}_ctx_')], axis=1)

    def _calculate_m15_native(self, g: pd.DataFrame) -> pd.DataFrame:
        lookback, sma_period = 14, 50
        
        # --- Original continuous features (retained for context) ---
        g['ATR'] = g['High'].sub(g['Low']).rolling(lookback).mean()
        g = self._calculate_adx(g, lookback)
        bb = self._calculate_bollinger_bands(g, self.config.BOLLINGER_PERIOD)
        g['bb_width'] = bb['bb_width']

        # --- NEW: STATE-BASED FEATURES ---
        # RSI states
        rsi_val = (100 - (100 / (1 + (g['Close'].diff().where(lambda x: x > 0, 0).ewm(com=lookback-1, adjust=False).mean() / 
                                     -g['Close'].diff().where(lambda x: x < 0, 0).ewm(com=lookback-1, adjust=False).mean().replace(0, 1e-9)))))
        g['rsi_os'] = (rsi_val < 30).astype(int)
        g['rsi_ob'] = (rsi_val > 70).astype(int)
        
        # Stochastic states
        stoch = self._calculate_stochastic(g, self.config.STOCHASTIC_PERIOD)
        g['stoch_os'] = (stoch['stoch_k'] < 20).astype(int)
        g['stoch_ob'] = (stoch['stoch_k'] > 80).astype(int)

        # ADX trend state
        g['adx_trending'] = (g['ADX'] > self.config.TREND_FILTER_THRESHOLD).astype(int)

        # Price vs Bollinger Bands state
        g['price_vs_bb'] = np.select([g['Close'] > bb['bb_upper'], g['Close'] < bb['bb_lower']], [1, -1], default=0)

        # --- CONTEXTUAL & MOMENTUM FEATURES ---
        # Distance from SMA50
        sma_50 = g['Close'].rolling(window=sma_period).mean()
        g['dist_from_sma50'] = (g['Close'] - sma_50) / g['ATR']

        # Consecutive Up/Down Candles
        direction = np.sign(g['Close'].diff())
        g['consecutive_candles'] = (direction.groupby((direction != direction.shift()).cumsum()).cumcount() + 1) * direction
        
        return g

    def label_outcomes(self, df: pd.DataFrame, sl_atr: float, tp_atr: float, lookahead: int) -> pd.DataFrame:
        logger.info("  - Generating trade labels using Triple-Barrier Method...")
        if not self.config.DYNAMIC_BARRIER_ADJUSTMENT:
            return self._label_with_fixed_barriers(df, sl_atr, tp_atr, lookahead)

        labeled_df = self._label_with_dynamic_barriers(df, sl_atr, tp_atr, lookahead, self.config.MIN_CLASS_RATIO, self.config.MAX_BARRIER_ADJUSTMENT_ITERATIONS)

        if labeled_df.empty:
            logger.warning("  - Dynamic barrier labeling failed. Falling back to fixed barriers.")
            return self._label_with_fixed_barriers(df, sl_atr, tp_atr, lookahead)
        return labeled_df

    def _label_with_dynamic_barriers(self, df: pd.DataFrame, initial_sl_atr: float, initial_tp_atr: float, lookahead: int, min_class_ratio: float, max_iterations: int) -> pd.DataFrame:
        best_df, best_balance, best_params, best_counts = pd.DataFrame(), 0, (initial_sl_atr, initial_tp_atr), None
        logger.info("    - Adjusting barriers dynamically to find balanced classes...")
        for i in range(max_iterations):
            current_sl, current_tp = initial_sl_atr * (0.8 + 0.1 * i), initial_tp_atr * (0.8 + 0.1 * i)
            labeled_df = df.groupby('Symbol', group_keys=False).apply(self._label_group, sl_atr=current_sl, tp_atr=current_tp, lookahead=lookahead, include_groups=False)
            if labeled_df.empty or 'target' not in labeled_df.columns: continue
            class_counts = labeled_df['target'].value_counts()
            if len(class_counts) < 3: continue
            current_ratio = class_counts.min() / class_counts.max() if class_counts.max() > 0 else 0
            if current_ratio > best_balance:
                best_balance, best_df, best_params, best_counts = current_ratio, labeled_df, (current_sl, current_tp), class_counts

        if not best_df.empty:
            logger.info(f"    - Best barriers found: SL={best_params[0]:.2f}*ATR, TP={best_params[1]:.2f}*ATR -> Ratio={best_balance:.2f}")
            logger.info(f"    - Final Class distribution (Buys: 1, Sells: -1, Holds: 0):\n{best_counts.to_string()}")
            if best_balance < min_class_ratio: logger.warning(f"    - Final class ratio is below target of {min_class_ratio}.")
            return best_df
        logger.warning("    - Could not find any valid labels with dynamic barriers.")
        return pd.DataFrame()

    def _label_with_fixed_barriers(self, df: pd.DataFrame, sl_atr: float, tp_atr: float, lookahead: int) -> pd.DataFrame:
        logger.info(f"    - Using fixed barriers: SL={sl_atr:.2f}*ATR, TP={tp_atr:.2f}*ATR")
        return df.groupby('Symbol', group_keys=False).apply(self._label_group, sl_atr=sl_atr, tp_atr=tp_atr, lookahead=lookahead, include_groups=False)
    
    def _label_group(self, group: pd.DataFrame, sl_atr: float, tp_atr: float, lookahead: int) -> pd.DataFrame:
        if len(group) < lookahead + 1: return pd.DataFrame()
        outcomes, prices, lows, highs, atr, regime = np.zeros(len(group)), group['Close'].values, group['Low'].values, group['High'].values, group['ATR'].values, group['market_regime'].values
        for i in range(len(group) - lookahead):
            entry, current_atr = prices[i], atr[i]
            if pd.isna(current_atr) or current_atr <= 1e-9: continue
            
            adj_sl, adj_tp = sl_atr * (1.2 if regime[i] == 1 else 0.8), tp_atr * (1.5 if regime[i] == 1 else 0.8)
            tp_long, sl_long = entry + current_atr * adj_tp, entry - current_atr * adj_sl
            tp_short, sl_short = entry - current_atr * adj_tp, entry + current_atr * adj_sl
            
            future_lows, future_highs = lows[i+1:i+1+lookahead], highs[i+1:i+1+lookahead]
            time_tp_long = np.where(future_highs >= tp_long)[0][0] if np.any(future_highs >= tp_long) else np.inf
            time_sl_long = np.where(future_lows <= sl_long)[0][0] if np.any(future_lows <= sl_long) else np.inf
            time_tp_short = np.where(future_lows <= tp_short)[0][0] if np.any(future_lows <= tp_short) else np.inf
            time_sl_short = np.where(future_highs >= sl_short)[0][0] if np.any(future_highs >= sl_short) else np.inf
            
            long_wins, short_wins = time_tp_long < time_sl_long, time_tp_short < time_sl_short
            if long_wins and not short_wins: outcomes[i] = 1
            elif short_wins and not long_wins: outcomes[i] = -1
            elif long_wins and short_wins: outcomes[i] = 1 if time_tp_long <= time_tp_short else -1
        group['target'] = outcomes
        return group
# =============================================================================
# 5. MODEL TRAINER
# =============================================================================
class ModelTrainer:
    FEATURES = [
        'ATR', 'ADX', 'hour', 'day_of_week', 'bb_width', 
        'dist_from_sma50', 'consecutive_candles',
        'adx_x_h1_trend', 'atr_x_d1_trend', 'H1_ctx_H1_Trend', 'D1_ctx_D1_Trend',
        'rsi_os', 'rsi_ob', 'stoch_os', 'stoch_ob', 'price_vs_bb', 'adx_trending'
    ]

    def __init__(self, config: ConfigModel):
        self.config = config
        self.shap_summary: Optional[pd.DataFrame] = None
        self.class_weights: Optional[Dict[int, float]] = None

    def train(self, df_train: pd.DataFrame) -> Optional[Pipeline]:
        if df_train.empty or 'target' not in df_train.columns:
            logger.error("  - Training aborted: Empty training data or missing 'target' column.")
            return None

        logger.info(f"  - Starting model training for period {df_train.index.min().date()} to {df_train.index.max().date()}...")
        
        y_map = {-1: 0, 0: 1, 1: 2}
        y = df_train['target'].map(y_map).astype(int)
        X = df_train[self.FEATURES].copy().fillna(0)

        class_counts = y.value_counts()
        min_ratio = class_counts.min() / class_counts.max() if class_counts.max() > 0 else 0
        if min_ratio < self.config.MIN_CLASS_RATIO or class_counts.min() < self.config.MIN_CLASS_SAMPLES:
            logger.info(f"    - Classes are imbalanced (ratio={min_ratio:.2f}). Applying RandomOverSampler.")
            try:
                X, y = RandomOverSampler(random_state=42).fit_resample(X, y)
                logger.info(f"    - Classes balanced. New distribution:\n{y.value_counts().to_string()}")
            except Exception as e:
                logger.error(f"    - Failed to balance classes: {e}")
                return None
        else:
            logger.info("    - Class balance is acceptable, no resampling needed.")

        self.class_weights = dict(zip(np.unique(y), compute_class_weight(class_weight='balanced', classes=np.unique(y), y=y)))
        
        X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, shuffle=False)

        study = self._optimize_hyperparameters(X_train, y_train, X_val, y_val)
        if study is None or not study.best_trials:
            logger.error("  - Training aborted: Hyperparameter optimization failed.")
            return None

        logger.info(f"    - Optimization complete. Best F1 Score: {study.best_value:.4f}")
        logger.info(f"    - Best params: {study.best_params}")

        final_pipeline = self._train_final_model(study.best_params, X, y)
        if final_pipeline is None:
            logger.error("  - Training aborted: Final model training failed.")
            return None

        logger.info("  - [SUCCESS] Model training complete.")
        return final_pipeline

    def _optimize_hyperparameters(self, X_train, y_train, X_val, y_val) -> Optional[optuna.study.Study]:
        logger.info(f"    - Starting hyperparameter optimization ({self.config.OPTUNA_TRIALS} trials)...")
        
        def objective(trial: optuna.Trial):
            param = {
                'objective': 'multi:softprob', 'num_class': 3, 'eval_metric': 'mlogloss',
                'booster': 'gbtree', 'use_label_encoder': False, 'seed': 42,
                'n_estimators': trial.suggest_int('n_estimators', 100, 1000, step=50),
                'max_depth': trial.suggest_int('max_depth', 3, 9),
                'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
                'subsample': trial.suggest_float('subsample', 0.6, 1.0),
                'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
                'gamma': trial.suggest_float('gamma', 0, 5),
                'reg_lambda': trial.suggest_float('reg_lambda', 1e-8, 5.0, log=True),
                'alpha': trial.suggest_float('alpha', 1e-8, 5.0, log=True),
                'early_stopping_rounds': 50
            }
            try:
                scaler = RobustScaler()
                X_train_scaled = scaler.fit_transform(X_train)
                X_val_scaled = scaler.transform(X_val)

                model = xgb.XGBClassifier(**param)
                model.fit(X_train_scaled, y_train, eval_set=[(X_val_scaled, y_val)], verbose=False)
                
                preds = model.predict(X_val_scaled)
                return f1_score(y_val, preds, average='macro')
            except Exception as e:
                logger.warning(f"Trial {trial.number} failed with error: {e}")
                return 0.0
        try:
            study = optuna.create_study(direction='maximize')
            study.optimize(objective, n_trials=self.config.OPTUNA_TRIALS, timeout=600, show_progress_bar=True)
            return study
        except Exception as e:
            logger.error(f"    - Optuna study failed catastrophically: {e}", exc_info=True)
            return None

    def _train_final_model(self, best_params: Dict, X: pd.DataFrame, y: pd.Series) -> Optional[Pipeline]:
        logger.info("    - Training final model...")
        try:
            final_params = {
                'objective': 'multi:softprob', 'num_class': 3, 'eval_metric': 'mlogloss',
                'use_label_encoder': False, 'seed': 42, 
                'early_stopping_rounds': 50,
                **best_params
            }
            
            final_pipeline = Pipeline([
                ('scaler', RobustScaler()),
                ('model', xgb.XGBClassifier(**final_params))
            ])
            
            X_train_final, X_test_final, y_train_final, y_test_final = train_test_split(X, y, test_size=0.1, shuffle=False)
            
            scaler = final_pipeline.named_steps['scaler']
            X_test_scaled = scaler.fit(X_train_final).transform(X_test_final)

            fit_params = {
                'model__eval_set': [(X_test_scaled, y_test_final)],
                'model__sample_weight': y_train_final.map(self.class_weights),
                'model__verbose': False
            }
            
            final_pipeline.fit(X_train_final, y_train_final, **fit_params)

            self._generate_shap_summary(final_pipeline.named_steps['model'], 
                                        final_pipeline.named_steps['scaler'].transform(X),
                                        X.columns)
            
            return final_pipeline

        except Exception as e:
            logger.error(f"    - Error during final model training: {e}", exc_info=True)
            return None

    def _generate_shap_summary(self, model: xgb.XGBClassifier, X_scaled: np.ndarray, feature_names: pd.Index):
        logger.info("    - Generating SHAP feature importance summary...")
        try:
            explainer = shap.TreeExplainer(model)
            shap_explanation = explainer(X_scaled)

            per_class_importance = shap_explanation.abs.mean(0).values
            overall_importance = per_class_importance.mean(axis=1)

            summary = pd.DataFrame(
                overall_importance,
                index=feature_names,
                columns=['SHAP_Importance']
            ).sort_values(by='SHAP_Importance', ascending=False)
            
            self.shap_summary = summary
            logger.info("    - SHAP summary generated successfully.")
        except Exception as e:
            logger.error(f"    - Failed to generate SHAP summary: {e}", exc_info=True)
            self.shap_summary = None

# =============================================================================
# 6. BACKTESTER & 7. PERFORMANCE ANALYZER
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

        trades, equity, equity_curve, open_position = [], initial_equity, [initial_equity], None
        candles = df_chunk.reset_index().to_dict('records')

        for candle in candles:
            if open_position:
                pnl, exit_price = 0, None
                if open_position['direction'] == 1:
                    if candle['Low'] <= open_position['sl']: pnl, exit_price = -open_position['risk_amt'], open_position['sl']
                    elif candle['High'] >= open_position['tp']: pnl, exit_price = open_position['risk_amt'] * open_position['rr'], open_position['tp']
                elif open_position['direction'] == -1:
                    if candle['High'] >= open_position['sl']: pnl, exit_price = -open_position['risk_amt'], open_position['sl']
                    elif candle['Low'] <= open_position['tp']: pnl, exit_price = open_position['risk_amt'] * open_position['rr'], open_position['tp']
                
                if exit_price:
                    equity += pnl
                    if equity <= 0: logger.critical("  - ACCOUNT BLOWN! Equity is zero or negative."); break
                    equity_curve.append(equity)
                    trades.append({'ExecTime': candle['Timestamp'],'Symbol': open_position['symbol'],'PNL': pnl,'Equity': equity,'Confidence': open_position['confidence'],'Direction': open_position['direction']})
                    open_position = None
            if not open_position:
                pred_class = np.argmax([candle['prob_short'], candle['prob_hold'], candle['prob_long']])
                direction = 1 if pred_class == 2 else -1 if pred_class == 0 else 0
                confidence = candle[f"prob_{'long' if direction == 1 else 'short'}"] if direction != 0 else 0
                
                # Dynamic Confidence Threshold
                min_confidence = 0.45 if candle['market_regime'] == 1 else 0.60
                
                if direction != 0 and confidence >= min_confidence:
                    atr = candle['ATR']
                    if pd.isna(atr) or atr <= 1e-9: continue
                    
                    tier_name = 'ultra_high' if confidence >= self.config.CONFIDENCE_TIERS['ultra_high']['min'] else 'high' if confidence >= self.config.CONFIDENCE_TIERS['high']['min'] else 'standard'
                    tier = self.config.CONFIDENCE_TIERS[tier_name]
                    risk_amt = equity * self.config.BASE_RISK_PER_TRADE_PCT * tier['risk_mult']
                    entry_price, sl_dist, tp_dist = candle['Close'], atr * 1.5, (atr * 1.5) * tier['rr']
                    
                    open_position = {'symbol': candle['Symbol'], 'direction': direction, 'sl': entry_price - sl_dist * direction, 'tp': entry_price + tp_dist * direction, 'risk_amt': risk_amt, 'rr': tier['rr'], 'confidence': confidence}
        return pd.DataFrame(trades), pd.Series(equity_curve)
class PerformanceAnalyzer:
    def __init__(self, config: ConfigModel):
        self.config = config

    def generate_full_report(self, trades_df: Optional[pd.DataFrame], equity_curve: Optional[pd.Series], cycle_metrics: List[Dict], aggregated_shap: Optional[pd.DataFrame] = None):
        if trades_df is None or trades_df.empty or equity_curve is None or len(equity_curve) < 2:
            logger.warning("-> No trades were generated. Skipping final report.")
            if trades_df is not None and equity_curve is not None: self.generate_text_report({}, cycle_metrics, aggregated_shap)
            return
        logger.info("-> Stage 4: Generating Final Performance Report...")
        self.plot_equity_curve(equity_curve)
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

    def _calculate_metrics(self, trades_df: pd.DataFrame, equity_curve: pd.Series) -> Dict[str, Any]:
        m = {'initial_capital': self.config.INITIAL_CAPITAL, 'ending_capital': equity_curve.iloc[-1]}
        m['net_profit'] = m['ending_capital'] - m['initial_capital']
        m['net_profit_pct'] = m['net_profit'] / m['initial_capital'] if m['initial_capital'] > 0 else 0
        wins, losses = trades_df[trades_df['PNL'] > 0], trades_df[trades_df['PNL'] <= 0]
        m['gross_profit'], m['gross_loss'] = wins['PNL'].sum(), abs(losses['PNL'].sum())
        m['profit_factor'] = m['gross_profit'] / m['gross_loss'] if m['gross_loss'] > 0 else np.inf
        m['total_trades'], m['winning_trades'], m['losing_trades'] = len(trades_df), len(wins), len(losses)
        m['win_rate'] = m['winning_trades'] / m['total_trades'] if m['total_trades'] > 0 else 0
        m['avg_win'], m['avg_loss'] = wins['PNL'].mean() if len(wins) > 0 else 0, losses['PNL'].mean() if len(losses) > 0 else 0
        m['expectancy'] = (m['win_rate'] * m['avg_win']) - ((1 - m['win_rate']) * abs(m['avg_loss'])) if m['total_trades'] > 0 else 0
        running_max = equity_curve.cummax()
        drawdown_abs = running_max - equity_curve
        m['max_drawdown_abs'] = drawdown_abs.max() if not drawdown_abs.empty else 0
        m['max_drawdown_pct'] = (drawdown_abs / running_max).replace([np.inf, -np.inf], 0).max() * 100
        years = (pd.to_datetime(trades_df['ExecTime']).max() - pd.to_datetime(trades_df['ExecTime']).min()).days / 365.25 if not trades_df.empty else 1
        m['cagr'] = ((m['ending_capital'] / m['initial_capital']) ** (1 / years) - 1) if years > 0 and m['initial_capital'] > 0 else 0
        pnl_std = trades_df['PNL'].std()
        m['sharpe_ratio'] = (trades_df['PNL'].mean() / pnl_std) * np.sqrt(252 * 24 * 4) if pnl_std > 0 else 0
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
Initial Capital:         ${m.get('initial_capital', self.config.INITIAL_CAPITAL):>15,.2f}
Ending Capital:          ${m.get('ending_capital', self.config.INITIAL_CAPITAL):>15,.2f}
Net Profit:              ${m.get('net_profit', 0):>15,.2f} ({m.get('net_profit_pct', 0):.2%})
----------------------------------------------------------------------
II. TRADE STATISTICS
----------------------------------------------------------------------
Total Trades:            {m.get('total_trades', 0):>15}
Win Rate:                {m.get('win_rate', 0):>15.2%}
Profit Factor:           {m.get('profit_factor', 0):>15.2f}
Expectancy (per trade):  ${m.get('expectancy', 0):>15.2f}
----------------------------------------------------------------------
III. RISK AND RETURN
----------------------------------------------------------------------
Max Drawdown:            {m.get('max_drawdown_pct', 0):>15.2f}% (${m.get('max_drawdown_abs', 0):,.2f})
Annual Return (CAGR):    {m.get('cagr', 0):>15.2%}
Sharpe Ratio (annual):   {m.get('sharpe_ratio', 0):>15.2f}
MAR Ratio (CAGR/MDD):    {m.get('mar_ratio', 0):>15.2f}
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
    retraining_dates = pd.date_range(start=test_start_date, end=df_featured.index.max(), freq=config.RETRAINING_FREQUENCY)

    all_trades, full_equity_curve, cycle_metrics, all_shap = [], [config.INITIAL_CAPITAL], [], []
    
    logger.info("-> Stage 3: Starting Walk-Forward Analysis...")

    for i, period_start_date in enumerate(retraining_dates):
        logger.info(f"\n{'='*25} CYCLE {i + 1}/{len(retraining_dates)}: {period_start_date.date()} {'='*25}")
        train_end = period_start_date - pd.Timedelta(days=1)
        train_start = train_end - pd.Timedelta(config.TRAINING_WINDOW)
        test_end = retraining_dates[i + 1] if i + 1 < len(retraining_dates) else df_featured.index.max()

        df_train_raw = df_featured.loc[train_start:train_end]
        df_test_chunk = df_featured.loc[period_start_date:test_end]

        if df_train_raw.empty or df_test_chunk.empty:
            logger.warning("  - Skipping cycle due to insufficient data.")
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
    logger.info("              WALK-FORWARD ANALYSIS COMPLETE")
    logger.info("==========================================================")
    
    reporter = PerformanceAnalyzer(config)
    aggregated_shap = pd.concat(all_shap).groupby(level=0)['SHAP_Importance'].mean().sort_values(ascending=False).to_frame() if all_shap else None
    reporter.generate_full_report(pd.concat(all_trades, ignore_index=True) if all_trades else pd.DataFrame(), pd.Series(full_equity_curve), cycle_metrics, aggregated_shap)
    
    logger.info(f"\nTotal execution time: {datetime.now() - start_time}")

if __name__ == '__main__':
    main()

# End_To_End_Advanced_ML_Trading_Framework_PRO_V55_Dynamic_Confidence.py