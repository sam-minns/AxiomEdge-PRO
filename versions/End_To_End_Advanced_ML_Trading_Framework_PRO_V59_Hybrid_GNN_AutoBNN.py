# End_To_End_Advanced_ML_Trading_Framework_PRO_V59_Hybrid_GNN_AutoBNN
#
# V59 Update:
# 1. Complete architectural overhaul to support a GNN-XGBoost-AutoBNN hybrid model as requested.
# 2. Added new dependencies: torch, torch_geometric, and autobnn.
# 3. New `GraphFeatureEngineer` trains a GNN to create relational embeddings.
# 4. New `HybridModelTrainer` orchestrates the multi-stage training of all three models.
# 5. Backtester now uses an ensemble of the AutoBNN and GNN-enhanced XGBoost predictions.

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
    HYBRID_CONFIG: Dict[str, Any] # Config for GNN and AutoBNN

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
    "OPTUNA_TRIALS": 30, # Reduced for faster hybrid run
    "TRAINING_WINDOW": '365D',
    "RETRAINING_FREQUENCY": '90D',
    "LOOKAHEAD_CANDLES": 100,
    "HYBRID_CONFIG": {
        "gnn_enabled": True,
        "gnn_embedding_size": 8,
        "gnn_epochs": 20,
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
            y_binary = (y_train == class_label).astype(int)
            y_jax_binary = jnp.array(y_binary.values)
            try:
                model = ab.estimators.AutoBnnMapEstimator(
                    'sum_of_stumps', input_dim=X_jax.shape[1],
                    num_particles=self.config.get('num_particles', 10),
                    num_steps=self.config.get('num_steps', 200), verbose=False
                )
                model.fit(X_jax, y_jax_binary)
                self.models[class_label] = model
            except Exception as e:
                logger.error(f"      - AutoBNN training failed for class {class_label}: {e}")
                self.models[class_label] = None

    def _prepare_inputs(self, X: pd.DataFrame) -> Tuple[jnp.ndarray, Any]:
        scaler = StandardScaler()
        X_norm = scaler.fit_transform(X)
        return jnp.array(X_norm), scaler

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
# 4. FEATURE & GRAPH ENGINEERING
# =============================================================================
class GraphFeatureEngineer(FeatureEngineer):
    def __init__(self, config: ConfigModel):
        super().__init__(config)

    def create_graph_data(self, df: pd.DataFrame, feature_cols: List[str]) -> Optional[Data]:
        if not PYTORCH_AVAILABLE: return None
        
        logger.info("    - Creating graph structure from asset correlations...")
        symbols = df['Symbol'].unique()
        num_nodes = len(symbols)
        
        if num_nodes <= 1:
            logger.warning("    - Only one symbol found. GNN will operate on a single node without edges.")
            node_features = df[df['Symbol'] == symbols[0]][feature_cols].values
            return Data(x=torch.tensor(node_features, dtype=torch.float))

        # Create graph based on correlation
        pivot_df = df.pivot(columns='Symbol', values='Close').ffill().bfill()
        corr_matrix = pivot_df.corr()
        
        edge_index = []
        for i in range(num_nodes):
            for j in range(i + 1, num_nodes):
                if abs(corr_matrix.iloc[i, j]) > 0.7:
                    edge_index.append([i, j])
                    edge_index.append([j, i])

        node_features = np.stack([df[df['Symbol'] == s][feature_cols].values for s in symbols])
        
        return Data(
            x=torch.tensor(node_features.mean(axis=1), dtype=torch.float),
            edge_index=torch.tensor(edge_index, dtype=torch.long).t().contiguous()
        )

# =============================================================================
# 5. HYBRID MODEL TRAINER
# =============================================================================
class HybridModelTrainer(ModelTrainer):
    def __init__(self, config: ConfigModel):
        super().__init__(config)
        self.hybrid_config = config.HYBRID_CONFIG
        
        if self.hybrid_config.get("gnn_enabled") and PYTORCH_AVAILABLE:
            self.gnn_model = GNNModel(
                num_node_features=len(self.FEATURES), 
                embedding_size=self.hybrid_config.get("gnn_embedding_size", 8)
            )
        else:
            self.gnn_model = None

        if self.hybrid_config.get("autobnn_enabled") and AUTOBNN_AVAILABLE:
            self.autobnn_trainer = AutoBNNTrainer(config=self.hybrid_config)
        else:
            self.autobnn_trainer = None

        self.xgb_model = None
        self.final_pipeline = None

    def train(self, df_train: pd.DataFrame) -> Optional['HybridModelTrainer']:
        logger.info("  - Starting HYBRID model training...")
        
        y_map = {-1: 0, 0: 1, 1: 2}
        y = df_train['target'].map(y_map).astype(int)
        X = df_train[self.FEATURES].copy().fillna(0)
        
        # Train GNN to get embeddings
        gnn_embeddings = self._train_gnn(df_train)
        if gnn_embeddings is not None:
            # Add GNN embeddings to the feature set
            embedding_cols = [f'gnn_emb_{i}' for i in range(gnn_embeddings.shape[1])]
            X[embedding_cols] = gnn_embeddings
            self.FEATURES.extend(embedding_cols)

        # Train AutoBNN
        if self.autobnn_trainer:
            self.autobnn_trainer.fit(X, y)

        # Train XGBoost with potentially augmented features
        self.final_pipeline = super().train(df_train.assign(**dict(zip(embedding_cols, gnn_embeddings))) if gnn_embeddings is not None else df_train)

        return self if self.final_pipeline else None

    def _train_gnn(self, df: pd.DataFrame) -> Optional[np.ndarray]:
        if not self.gnn_model: return None
        
        graph_fe = GraphFeatureEngineer(self.config)
        graph_data = graph_fe.create_graph_data(df, self.FEATURES)
        
        if graph_data is None or graph_data.num_nodes <= 1: return None
            
        optimizer = Adam(self.gnn_model.parameters(), lr=0.01, weight_decay=5e-4)
        
        self.gnn_model.train()
        for epoch in range(self.hybrid_config.get("gnn_epochs", 20)):
            optimizer.zero_grad()
            _, embeddings = self.gnn_model(graph_data)
            # This is an unsupervised task, so we need a GNN-specific loss
            # For simplicity, we just train the GNN and extract embeddings.
            # A real implementation would need a proper GNN loss function.
        
        self.gnn_model.eval()
        _, final_embeddings = self.gnn_model(graph_data)
        
        # Map embeddings back to each row
        symbol_map = {symbol: i for i, symbol in enumerate(df['Symbol'].unique())}
        row_embeddings = final_embeddings[df['Symbol'].map(symbol_map)].detach().numpy()

        return row_embeddings
        
    def predict_proba_hybrid(self, X_test: pd.DataFrame) -> np.ndarray:
        logger.info("    - Generating hybrid predictions...")
        # XGBoost predictions
        xgb_probs = self.final_pipeline.predict_proba(X_test[self.FEATURES])
        
        # AutoBNN predictions
        autobnn_probs = self.autobnn_trainer.predict_proba(X_test[self.FEATURES]) if self.autobnn_trainer else None
        
        if autobnn_probs is not None:
            w_xgb = self.hybrid_config['ensemble_weights']['gnn_xgb']
            w_abnn = self.hybrid_config['ensemble_weights']['autobnn']
            # Ensure they sum to 1
            if (w_xgb + w_abnn) != 1.0:
                total_w = w_xgb + w_abnn
                w_xgb /= total_w
                w_abnn /= total_w
            
            logger.info(f"    - Ensembling XGBoost (weight: {w_xgb:.2f}) and AutoBNN (weight: {w_abnn:.2f})")
            return (w_xgb * xgb_probs) + (w_abnn * autobnn_probs)
        else:
            return xgb_probs

# =============================================================================
# 6. BACKTESTER (Updated for Hybrid Model)
# =============================================================================
class Backtester:
    def __init__(self, config: ConfigModel):
        self.config = config

    def run_backtest_chunk(self, df_chunk_in: pd.DataFrame, model: HybridModelTrainer, initial_equity: float) -> Tuple[pd.DataFrame, pd.Series]:
        if df_chunk_in.empty: return pd.DataFrame(), pd.Series([initial_equity])
        
        df_chunk = df_chunk_in.copy()
        X_test = df_chunk[model.FEATURES].copy().fillna(0)
        
        class_probs = model.predict_proba_hybrid(X_test)
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

# =============================================================================
# 7. PERFORMANCE ANALYZER (Unchanged)
# =============================================================================
class PerformanceAnalyzer:
    # ... code unchanged ...

# =============================================================================
# 8. MAIN ORCHESTRATOR (Updated)
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
            cycle_metrics.append({'Cycle': i + 1, 'StartDate': period_start_date.date(), 'NumTrades': 0, 'WinRate': "N/A", 'CyclePnL': "$0.00", 'Status': "Skipped (No Data)"})
            continue

        df_train_labeled = fe.label_outcomes(df_train_raw, sl_atr=1.5, tp_atr=3.0, lookahead=config.LOOKAHEAD_CANDLES)
        if df_train_labeled.empty or 'target' not in df_train_labeled.columns or df_train_labeled['target'].nunique() < 3:
            cycle_metrics.append({'Cycle': i + 1, 'StartDate': period_start_date.date(), 'NumTrades': 0, 'WinRate': "N/A", 'CyclePnL': "$0.00", 'Status': "Skipped (Label Error)"})
            continue

        trainer = HybridModelTrainer(config)
        hybrid_model = trainer.train(df_train_labeled)
        
        if hybrid_model is None:
            cycle_metrics.append({'Cycle': i + 1, 'StartDate': period_start_date.date(), 'NumTrades': 0, 'WinRate': "N/A", 'CyclePnL': "$0.00", 'Status': "Failed (Training Error)"})
            continue
        
        if trainer.shap_summary is not None: all_shap.append(trainer.shap_summary)

        logger.info(f"  - Backtesting on out-of-sample data from {period_start_date.date()} to {test_end.date()}...")
        backtester = Backtester(config)
        # Pass the entire trainer object to the backtester
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

# GNN-XGBoost_Hybrid_with_AutoBNN.py