# End_To_End_Advanced_ML_Trading_Framework_PRO_V61_AutoBNN_Fix
#
# V61 Update:
# 1. Fixed TypeError: unhashable type in AutoBNN library.
# 2. Modified the AutoBNNTrainer to pass the `likelihood_model` as a direct
#    keyword argument instead of within a dictionary, which works around
#    the internal hashing error in the library.

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
from sklearn.preprocessing import RobustScaler, StandardScaler
from sklearn.utils.class_weight import compute_class_weight

# Imbalanced-learn
from imblearn.over_sampling import RandomOverSampler

# Pydantic for Config Validation
from pydantic import BaseModel, DirectoryPath, confloat, conint

# --- NEW IMPORTS FOR GNN AND AUTOBNN ---
try:
    import torch
    import torch.nn.functional as F
    from torch_geometric.data import Data
    from torch_geometric.nn import GCNConv
    from torch.optim import Adam
    PYTORCH_AVAILABLE = True
except ImportError:
    PYTORCH_AVAILABLE = False
    warnings.warn("PyTorch or PyTorch Geometric not found. GNN components will be disabled.")

try:
    import jax
    from jax import random
    import jax.numpy as jnp
    import autobnn as ab
    from autobnn import likelihoods
    AUTOBNN_AVAILABLE = True
except ImportError:
    AUTOBNN_AVAILABLE = False
    warnings.warn("AutoBNN, JAX, or Flax not found. AutoBNN components will be disabled.")

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
            'trading_framework_hybrid.log',
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
    HYBRID_CONFIG: Dict[str, Any] 

    def __init__(self, **data: Any):
        super().__init__(**data)
        self.MODEL_SAVE_PATH = f"{self.REPORT_LABEL}_model.json"
        self.PLOT_SAVE_PATH = f"{self.REPORT_LABEL}_equity_curve.png"
        self.REPORT_SAVE_PATH = f"{self.REPORT_LABEL}_quantitative_report.txt"
        self.SHAP_PLOT_PATH = f"{self.REPORT_LABEL}_shap_summary.png"

raw_config = {
    "BASE_PATH": os.getcwd(),
    "DATA_FILENAMES": [
        "AUDUSD_M15_202105170000_202506021830.csv",
        "EURUSD_M15_202106020100_202506021830.csv",
        "GBPUSD_M15_202106020015_202506021830.csv",
    ],
    "REPORT_LABEL": "GNN_XGBoost_Hybrid_with_AutoBNN",
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
    "OPTUNA_TRIALS": 30, 
    "TRAINING_WINDOW": '365D',
    "RETRAINING_FREQUENCY": '90D',
    "LOOKAHEAD_CANDLES": 100,
    "HYBRID_CONFIG": {
        "gnn_enabled": True,
        "gnn_embedding_size": 8,
        "gnn_epochs": 10,
        "autobnn_enabled": True,
        "autobnn_num_particles": 10,
        "autobnn_num_steps": 200,
        "ensemble_weights": {"gnn_xgb": 0.6, "autobnn": 0.4}
    }
}

# =============================================================================
# 3. GNN and AutoBNN Model Classes
# =============================================================================
class GNNModel(torch.nn.Module if PYTORCH_AVAILABLE else object):
    def __init__(self, num_node_features, embedding_size=16):
        super(GNNModel, self).__init__()
        self.conv1 = GCNConv(num_node_features, 32)
        self.conv2 = GCNConv(32, embedding_size)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        embeddings = self.conv2(x, edge_index)
        return embeddings

class AutoBNNTrainer:
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        self.models = {}
        self.x_scaler = None
        if not AUTOBNN_AVAILABLE:
            logger.warning("AutoBNN is not available. AutoBNN training will be skipped.")
            return
        jax.config.update('jax_enable_x64', True)
        self.likelihood = likelihoods.NormalLikelihoodLogisticNoise(log_noise_scale=0.1)

    def fit(self, X_train: pd.DataFrame, y_train: pd.Series):
        if not AUTOBNN_AVAILABLE: return
        logger.info("    - Training AutoBNN model...")
        X_jax, self.x_scaler = self._prepare_inputs(X_train)
        
        for class_label in sorted(y_train.unique()):
            logger.info(f"      - Training AutoBNN for class {class_label}...")
            y_binary = (y_train == class_label).astype(int)
            y_jax_binary = jnp.array(y_binary.values)
            try:
                model = self._create_model('sum_of_stumps', X_jax.shape[1])
                model.fit(X_jax, y_jax_binary)
                self.models[class_label] = model
            except Exception as e:
                logger.error(f"      - AutoBNN training failed for class {class_label}: {e}", exc_info=True)
                self.models[class_label] = None

    def _prepare_inputs(self, X: pd.DataFrame) -> Tuple[jnp.ndarray, Any]:
        scaler = StandardScaler()
        X_norm = scaler.fit_transform(X)
        return jnp.array(X_norm), scaler

    def _create_model(self, model_type: str, input_dim: int):
        # FIX: Pass likelihood_model directly as a keyword argument
        common_config = {
            'input_dim': input_dim,
            'num_particles': self.config.get('num_particles', 10),
            'num_steps': self.config.get('num_steps', 200),
            'verbose': self.config.get('verbose', False)
        }
        key = random.PRNGKey(0) # Create a JAX random key

        return ab.estimators.AutoBnnMapEstimator(
            model_type,
            likelihood_model=self.likelihood,
            seed=key,
            **common_config
        )

    def predict_proba(self, X_test: pd.DataFrame) -> Optional[np.ndarray]:
        if not self.models or not AUTOBNN_AVAILABLE: return None
        X_jax = self._transform_inputs(X_test)
        all_probs = []
        
        for class_label in sorted(self.models.keys()):
            model = self.models[class_label]
            if model is None:
                all_probs.append(np.zeros(len(X_test)))
                continue
            preds = model.predict(X_jax)
            probs = np.array(preds.mean())
            all_probs.append(np.clip(probs, 0, 1))

        probs_array = np.vstack(all_probs).T
        exp_probs = np.exp(probs_array - np.max(probs_array, axis=1, keepdims=True))
        return exp_probs / np.sum(exp_probs, axis=1, keepdims=True)

    def _transform_inputs(self, X: pd.DataFrame) -> jnp.ndarray:
        if self.x_scaler is not None:
            return jnp.array(self.x_scaler.transform(X))
        return jnp.array(X.values)

# =============================================================================
# 4. DATA & FEATURE ENGINEERING
# =============================================================================
class DataLoader:
    def __init__(self, config: ConfigModel):
        self.config = config

    def load_and_parse_data(self) -> Optional[Dict[str, pd.DataFrame]]:
        logger.info("-> Stage 1: Loading and Preparing Data...")
        data_by_tf: Dict[str, List[pd.DataFrame]] = {'M15': []}
        
        for filename in self.config.DATA_FILENAMES:
            if 'M15' not in filename: continue
            
            file_path = os.path.join(self.config.BASE_PATH, filename)
            if not os.path.exists(file_path):
                logger.warning(f"  - File not found, skipping: {file_path}")
                continue
            try:
                parts = filename.split('_')
                symbol = parts[0]
                
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
                data_by_tf['M15'].append(df)

            except Exception as e:
                logger.error(f"  - Failed to load {filename}: {e}", exc_info=True)

        if not data_by_tf['M15']:
            logger.critical("No M15 data files were loaded.")
            return None
            
        combined = pd.concat(data_by_tf['M15']).drop_duplicates()
        combined = combined[~combined.index.duplicated(keep='first')].sort_index()
        logger.info(f"  - Processed M15: {len(combined):,} rows for {len(combined['Symbol'].unique())} symbols.")
            
        logger.info("[SUCCESS] Data loading and preparation complete.")
        return {'M15': combined}

class FeatureEngineer:
    def __init__(self, config: ConfigModel):
        self.config = config

    def create_feature_stack(self, data_by_tf: Dict[str, pd.DataFrame]) -> pd.DataFrame:
        logger.info("-> Stage 2: Engineering Features...")
        if data_by_tf['M15'].empty:
            logger.critical("  - M15 data is required but not available.")
            return pd.DataFrame()
            
        df_m15 = data_by_tf['M15'].copy()
        
        df_featured = df_m15.groupby('Symbol').apply(self._calculate_features, include_groups=False).reset_index()
        
        feature_cols = [col for col in df_featured.columns if col not in ['Timestamp', 'Open', 'High', 'Low', 'Close', 'Symbol', 'RealVolume']]
        df_featured[feature_cols] = df_featured.groupby('Symbol')[feature_cols].shift(1)
        
        df_featured.set_index('Timestamp', inplace=True)
        df_final = df_featured.dropna()
        
        logger.info(f"  - Created features. Final dataset shape: {df_final.shape}")
        return df_final

    def _calculate_features(self, g: pd.DataFrame) -> pd.DataFrame:
        g['hour'] = g.index.hour
        g['day_of_week'] = g.index.dayofweek
        g['ATR'] = (g['High'] - g['Low']).rolling(14).mean()
        delta = g['Close'].diff()
        gain = delta.where(delta > 0, 0).ewm(com=13, adjust=False).mean()
        loss = -delta.where(delta < 0, 0).ewm(com=13, adjust=False).mean()
        g['RSI'] = 100 - (100 / (1 + (gain / loss.replace(0, 1e-9))))
        return g

    def label_outcomes(self, df: pd.DataFrame, lookahead: int) -> pd.DataFrame:
        logger.info("  - Generating trade labels...")
        labeled_df = df.groupby('Symbol').apply(self._label_group, lookahead=lookahead, include_groups=False)
        return labeled_df.reset_index()

    def _label_group(self, group: pd.DataFrame, lookahead: int) -> pd.DataFrame:
        future_price = group['Close'].shift(-lookahead)
        group['target_raw'] = (future_price - group['Close']) / group['Close']
        threshold = group['target_raw'].std() * 0.5 
        group['target'] = np.select(
            [group['target_raw'] > threshold, group['target_raw'] < -threshold],
            [1, -1],
            default=0
        )
        return group.drop(columns=['target_raw'])

class GraphFeatureEngineer(FeatureEngineer):
    def create_graph_data(self, df: pd.DataFrame, feature_cols: List[str]) -> Optional[Data]:
        if not PYTORCH_AVAILABLE: return None
        logger.info("    - Creating graph structure from asset correlations...")
        symbols = df['Symbol'].unique()
        num_nodes = len(symbols)
        if num_nodes <= 1:
            logger.warning("    - Only one symbol found. GNN requires multiple symbols to build graph relationships.")
            return None
        
        pivot_df = df.reset_index().pivot(index='Timestamp', columns='Symbol', values='Close').ffill().bfill()
        corr_matrix = pivot_df.corr()
        
        edge_index = []
        for i in range(num_nodes):
            for j in range(i + 1, num_nodes):
                if abs(corr_matrix.iloc[i, j]) > 0.7:
                    edge_index.append([i, j])
                    edge_index.append([j, i])

        if not edge_index:
            logger.warning("    - No significant correlations found to build graph edges.")
            edge_index = [[i, i] for i in range(num_nodes)]

        node_features = df.groupby('Symbol')[feature_cols].mean().values

        return Data(
            x=torch.tensor(np.array(node_features), dtype=torch.float),
            edge_index=torch.tensor(edge_index, dtype=torch.long).t().contiguous()
        )

# =============================================================================
# 5. HYBRID MODEL TRAINER
# =============================================================================
class ModelTrainer: # The original trainer, now used specifically for XGBoost
    FEATURES = [] 
    def __init__(self, config: ConfigModel):
        self.config = config
        self.shap_summary: Optional[pd.DataFrame] = None
        self.class_weights: Optional[Dict[int, float]] = None

    def train(self, X: pd.DataFrame, y: pd.Series) -> Optional[Pipeline]:
        self.class_weights = dict(zip(np.unique(y), compute_class_weight(class_weight='balanced', classes=np.unique(y), y=y)))
        
        X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, shuffle=False)
        study = self._optimize_hyperparameters(X_train, y_train, X_val, y_val)
        if not study or not study.best_trials: return None
        
        final_pipeline = self._train_final_model(study.best_params, X, y)
        return final_pipeline

    def _optimize_hyperparameters(self, X_train, y_train, X_val, y_val) -> Optional[optuna.study.Study]:
        def objective(trial: optuna.Trial):
            param = {
                'objective': 'multi:softprob', 'num_class': 3, 'eval_metric': 'mlogloss',
                'booster': 'gbtree', 'use_label_encoder': False, 'seed': 42,
                'n_estimators': trial.suggest_int('n_estimators', 100, 1000, step=50),
                'max_depth': trial.suggest_int('max_depth', 3, 9),
                'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
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
            except Exception: return 0.0
        
        study = optuna.create_study(direction='maximize')
        study.optimize(objective, n_trials=self.config.OPTUNA_TRIALS, timeout=1200, show_progress_bar=True)
        return study

    def _train_final_model(self, best_params: Dict, X: pd.DataFrame, y: pd.Series) -> Optional[Pipeline]:
        final_params = {'objective': 'multi:softprob', 'num_class': 3, 'eval_metric': 'mlogloss', 
                        'use_label_encoder': False, 'seed': 42, 'early_stopping_rounds': 50, **best_params}
        final_pipeline = Pipeline([('scaler', RobustScaler()), ('model', xgb.XGBClassifier(**final_params))])
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, shuffle=False)
        scaler = final_pipeline.named_steps['scaler']
        X_test_scaled = scaler.fit(X_train).transform(X_test)
        fit_params = {'model__eval_set': [(X_test_scaled, y_test)], 'model__sample_weight': y_train.map(self.class_weights), 'model__verbose': False}
        final_pipeline.fit(X_train, y_train, **fit_params)
        self._generate_shap_summary(final_pipeline.named_steps['model'], final_pipeline.named_steps['scaler'].transform(X), X.columns)
        return final_pipeline

    def _generate_shap_summary(self, model, X_scaled, feature_names):
        try:
            explainer = shap.TreeExplainer(model)
            shap_explanation = explainer(X_scaled)
            per_class_importance = shap_explanation.abs.mean(0).values
            overall_importance = per_class_importance.mean(axis=1)
            self.shap_summary = pd.DataFrame(overall_importance, index=feature_names, columns=['SHAP_Importance']).sort_values(by='SHAP_Importance', ascending=False)
        except Exception as e:
            logger.error(f"Error generating SHAP summary: {e}")
            self.shap_summary = None

class HybridModelTrainer:
    BASE_FEATURES = ['hour', 'day_of_week', 'ATR', 'RSI']

    def __init__(self, config: ConfigModel):
        self.config = config
        self.hybrid_config = config.HYBRID_CONFIG
        self.gnn_model, self.xgb_pipeline, self.autobnn_trainer = None, None, None
        self.shap_summary, self.gnn_embeddings_map = None, {}
        self.feature_names_xgb = self.BASE_FEATURES.copy()

    def train(self, df_train: pd.DataFrame):
        logger.info("  - Starting HYBRID model training...")
        y_map = {-1: 0, 0: 1, 1: 2}
        y = df_train['target'].map(y_map)
        X = df_train[self.BASE_FEATURES].copy().fillna(0)
        
        if self.hybrid_config.get("gnn_enabled"): self._train_gnn(df_train)
        X_augmented = self._augment_features(X, df_train['Symbol'])

        if self.hybrid_config.get("autobnn_enabled"):
            self.autobnn_trainer = AutoBNNTrainer(config=self.hybrid_config)
            self.autobnn_trainer.fit(X, y)
        
        self.xgb_pipeline = self._train_xgboost(X_augmented, y)
        return self

    def _train_gnn(self, df: pd.DataFrame):
        if not PYTORCH_AVAILABLE: return
        graph_fe = GraphFeatureEngineer(self.config)
        graph_data = graph_fe.create_graph_data(df, self.BASE_FEATURES)
        
        if graph_data is None: 
            self.gnn_model = None
            return

        self.gnn_model = GNNModel(
            num_node_features=graph_data.num_node_features, 
            embedding_size=self.hybrid_config.get("gnn_embedding_size", 8)
        )
        optimizer = Adam(self.gnn_model.parameters(), lr=0.01)
        
        logger.info("    - Training GNN model...")
        self.gnn_model.train()
        for epoch in range(self.hybrid_config.get("gnn_epochs", 10)):
            optimizer.zero_grad()
            _ = self.gnn_model(graph_data)
        
        self.gnn_model.eval()
        with torch.no_grad():
            final_embeddings = self.gnn_model(graph_data)
        
        symbols = df['Symbol'].unique()
        self.gnn_embeddings_map = {sym: emb for sym, emb in zip(symbols, final_embeddings.detach().numpy())}
        logger.info("    - GNN training complete.")
        
    def _augment_features(self, X: pd.DataFrame, symbols: pd.Series) -> pd.DataFrame:
        if not self.gnn_model or not self.gnn_embeddings_map:
            return X
            
        logger.info("    - Augmenting feature set with GNN embeddings...")
        
        embedding_df = symbols.map(self.gnn_embeddings_map).apply(pd.Series)
        embedding_df.columns = [f'gnn_emb_{i}' for i in range(embedding_df.shape[1])]
        embedding_df.index = X.index

        X_aug = pd.concat([X, embedding_df], axis=1)
        self.feature_names_xgb = X_aug.columns.tolist()
        return X_aug

    def _train_xgboost(self, X_aug: pd.DataFrame, y: pd.Series) -> Optional[Pipeline]:
        xgb_trainer = ModelTrainer(self.config)
        xgb_trainer.FEATURES = self.feature_names_xgb
        pipeline = xgb_trainer.train(X_aug, y)
        self.shap_summary = xgb_trainer.shap_summary
        return pipeline

    def predict_proba(self, X_test: pd.DataFrame) -> Optional[np.ndarray]:
        if not self.xgb_pipeline: return None
        
        X_test_base = X_test[[c for c in self.BASE_FEATURES if c in X_test.columns]]
        X_test_aug = self._augment_features(X_test_base, X_test['Symbol'])
        
        xgb_feature_names = self.xgb_pipeline.named_steps['model'].get_booster().feature_names
        missing_cols = set(xgb_feature_names) - set(X_test_aug.columns)
        for c in missing_cols: X_test_aug[c] = 0
        X_test_aug = X_test_aug[xgb_feature_names]

        xgb_probs = self.xgb_pipeline.predict_proba(X_test_aug)
        
        if self.autobnn_trainer and self.hybrid_config.get("autobnn_enabled"):
            autobnn_probs = self.autobnn_trainer.predict_proba(X_test_base)
            if autobnn_probs is not None and autobnn_probs.shape == xgb_probs.shape:
                w_xgb = self.hybrid_config['ensemble_weights']['gnn_xgb']
                w_abnn = self.hybrid_config['ensemble_weights']['autobnn']
                return (w_xgb * xgb_probs) + (w_abnn * autobnn_probs)
                
        return xgb_probs

# =============================================================================
# 6. BACKTESTER
# =============================================================================
class Backtester:
    def __init__(self, config: ConfigModel):
        self.config = config

    def run_backtest_chunk(self, df_chunk_in: pd.DataFrame, model: HybridModelTrainer, initial_equity: float) -> Tuple[pd.DataFrame, pd.Series]:
        if df_chunk_in.empty: return pd.DataFrame(), pd.Series([initial_equity])
        
        df_chunk = df_chunk_in.copy()
        
        class_probs = model.predict_proba(df_chunk)
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
                
                min_confidence = 0.60
                
                if direction != 0 and confidence >= min_confidence:
                    atr = candle['ATR']
                    if pd.isna(atr) or atr <= 1e-9: continue
                    
                    tier_name = 'ultra_high' if confidence >= self.config.CONFIDENCE_TIERS['ultra_high']['min'] else 'high' if confidence >= self.config.CONFIDENCE_TIERS['high']['min'] else 'standard'
                    tier = self.config.CONFIDENCE_TIERS[tier_name]
                    risk_amt = equity * self.config.BASE_RISK_PER_TRADE_PCT * tier['risk_mult']
                    entry_price, sl_dist, tp_dist = candle['Close'], atr * 1.5, (atr * 1.5) * tier['rr']
                    
                    open_position = {'symbol': candle['Symbol'], 'direction': direction, 'sl': entry_price - sl_dist * direction, 'tp': entry_price + tp_dist * direction, 'risk_amt': risk_amt, 'rr': tier['rr'], 'confidence': confidence}
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
    logger.info("  STARTING HYBRID GNN-XGBOOST ML TRADING FRAMEWORK")
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
            cycle_metrics.append({'Cycle': i + 1, 'StartDate': period_start_date.date(), 'NumTrades': 0, 'WinRate': "N/A", 'CyclePnL': "$0.00", 'Status': "Skipped (No Data)"})
            continue

        df_train_labeled = fe.label_outcomes(df_train_raw, lookahead=config.LOOKAHEAD_CANDLES)
        if df_train_labeled.empty or 'target' not in df_train_labeled.columns or df_train_labeled['target'].nunique() < 2:
            cycle_metrics.append({'Cycle': i + 1, 'StartDate': period_start_date.date(), 'NumTrades': 0, 'WinRate': "N/A", 'CyclePnL': "$0.00", 'Status': "Skipped (Label Error)"})
            continue

        trainer = HybridModelTrainer(config)
        hybrid_model = trainer.train(df_train_labeled)
        
        if hybrid_model is None or hybrid_model.xgb_pipeline is None:
            cycle_metrics.append({'Cycle': i + 1, 'StartDate': period_start_date.date(), 'NumTrades': 0, 'WinRate': "N/A", 'CyclePnL': "$0.00", 'Status': "Failed (Training Error)"})
            continue
        
        if trainer.shap_summary is not None: all_shap.append(trainer.shap_summary)

        logger.info(f"  - Backtesting on out-of-sample data from {period_start_date.date()} to {test_end.date()}...")
        backtester = Backtester(config)
        chunk_trades_df, chunk_equity = backtester.run_backtest_chunk(df_test_chunk, hybrid_model, initial_equity=full_equity_curve[-1])

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

# End_To_End_Advanced_ML_Trading_Framework_PRO_V60_Hybrid_FIX.py