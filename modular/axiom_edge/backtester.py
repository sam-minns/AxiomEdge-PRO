# =============================================================================
# BACKTESTER MODULE
# =============================================================================

import logging
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any
from datetime import datetime, timedelta

from .config import ConfigModel

logger = logging.getLogger(__name__)

class PerformanceAnalyzer:
    """Performance analysis and metrics calculation for backtesting results."""
    
    def __init__(self, config: ConfigModel):
        self.config = config
    
    def _calculate_metrics(self, trades_df: Optional[pd.DataFrame], 
                          equity_curve: Optional[pd.Series]) -> Dict[str, Any]:
        """Calculate comprehensive performance metrics."""
        metrics = {}
        
        try:
            initial_capital = getattr(self.config, 'INITIAL_CAPITAL', 10000)
            metrics['initial_capital'] = initial_capital
            
            if equity_curve is not None and len(equity_curve) > 0:
                ending_capital = equity_curve.iloc[-1]
                metrics['ending_capital'] = ending_capital
                metrics['total_net_profit'] = ending_capital - initial_capital
                metrics['net_profit_pct'] = (ending_capital / initial_capital) - 1
                
                # Calculate returns
                returns = equity_curve.pct_change().dropna()
                if len(returns) > 0:
                    # Annualized metrics
                    trading_days = 252
                    total_days = len(returns)
                    years = total_days / trading_days
                    
                    # CAGR
                    if years > 0:
                        metrics['cagr'] = (ending_capital / initial_capital) ** (1/years) - 1
                    else:
                        metrics['cagr'] = 0
                    
                    # Sharpe Ratio
                    if returns.std() > 0:
                        metrics['sharpe_ratio'] = (returns.mean() * trading_days) / (returns.std() * np.sqrt(trading_days))
                    else:
                        metrics['sharpe_ratio'] = 0
                    
                    # Maximum Drawdown
                    peak = equity_curve.expanding().max()
                    drawdown = (equity_curve - peak) / peak
                    metrics['max_drawdown'] = drawdown.min()
                    metrics['max_drawdown_pct'] = abs(drawdown.min())
            
            # Trade-based metrics
            if trades_df is not None and not trades_df.empty:
                metrics['total_trades'] = len(trades_df)
                
                if 'pnl' in trades_df.columns:
                    pnl = trades_df['pnl']
                    winning_trades = pnl[pnl > 0]
                    losing_trades = pnl[pnl < 0]
                    
                    metrics['winning_trades'] = len(winning_trades)
                    metrics['losing_trades'] = len(losing_trades)
                    metrics['win_rate'] = len(winning_trades) / len(trades_df) if len(trades_df) > 0 else 0
                    
                    metrics['avg_win'] = winning_trades.mean() if len(winning_trades) > 0 else 0
                    metrics['avg_loss'] = losing_trades.mean() if len(losing_trades) > 0 else 0
                    
                    # Profit Factor
                    gross_profit = winning_trades.sum() if len(winning_trades) > 0 else 0
                    gross_loss = abs(losing_trades.sum()) if len(losing_trades) > 0 else 0
                    metrics['profit_factor'] = gross_profit / gross_loss if gross_loss > 0 else float('inf')
                    
                    metrics['expected_payoff'] = pnl.mean()
                    metrics['largest_win'] = pnl.max()
                    metrics['largest_loss'] = pnl.min()
        
        except Exception as e:
            logger.error(f"Error calculating metrics: {e}")
        
        return metrics

class Backtester:
    """
    Advanced backtesting engine with dynamic ensemble voting, 
    adaptive confidence thresholds, and comprehensive trade simulation.
    """
    
    def __init__(self, config: ConfigModel):
        self.config = config
        self.use_tp_ladder = getattr(config, 'USE_TP_LADDER', False)
        
        # Validate TP ladder configuration
        if self.use_tp_ladder:
            tp_levels = getattr(config, 'TP_LADDER_LEVELS_PCT', [])
            tp_multipliers = getattr(config, 'TP_LADDER_RISK_MULTIPLIERS', [])
            
            if not tp_levels or not tp_multipliers or len(tp_levels) != len(tp_multipliers):
                logger.error("TP Ladder config error: Invalid or mismatched lengths. Disabling.")
                self.use_tp_ladder = False
            elif not np.isclose(sum(tp_levels), 1.0):
                logger.error(f"TP Ladder config error: Levels sum ({sum(tp_levels)}) != 1.0. Disabling.")
                self.use_tp_ladder = False
            else:
                logger.info("Take-Profit Ladder is ENABLED.")
        
        logger.info("Backtester initialized")

    def run_backtest_chunk(self, df_chunk: pd.DataFrame, training_results: Dict, 
                          initial_equity: float, feature_list: List[str],
                          confidence_threshold: float = 0.5) -> Tuple[pd.DataFrame, pd.Series, bool, Optional[Dict], Dict]:
        """
        Run backtest on a data chunk using trained models.
        
        Args:
            df_chunk: Data chunk for backtesting
            training_results: Results from model training
            initial_equity: Starting capital
            feature_list: List of features to use
            confidence_threshold: Minimum confidence for trades
            
        Returns:
            Tuple of (trades_df, equity_curve, success_flag, additional_metrics, summary)
        """
        try:
            logger.info(f"Running backtest on {len(df_chunk)} samples")
            
            # Extract model from training results
            model = training_results.get('model')
            if model is None:
                logger.error("No model found in training results")
                return pd.DataFrame(), pd.Series(), False, None, {}
            
            # Prepare features
            X = df_chunk[feature_list].fillna(0)
            
            # Generate predictions
            predictions = model.predict(X)
            probabilities = model.predict_proba(X) if hasattr(model, 'predict_proba') else None
            
            # Create signals DataFrame
            signals_df = pd.DataFrame({
                'timestamp': df_chunk.index,
                'prediction': predictions,
                'confidence': probabilities[:, 1] if probabilities is not None else 0.5,
                'Close': df_chunk['Close'],
                'High': df_chunk['High'],
                'Low': df_chunk['Low'],
                'Open': df_chunk['Open']
            })
            
            # Filter by confidence threshold
            signals_df = signals_df[signals_df['confidence'] >= confidence_threshold]
            
            if signals_df.empty:
                logger.warning("No signals above confidence threshold")
                return pd.DataFrame(), pd.Series([initial_equity]), True, None, {'total_trades': 0}
            
            # Simulate trades
            trades_df, equity_curve = self._simulate_trades(signals_df, initial_equity)
            
            # Calculate summary metrics
            analyzer = PerformanceAnalyzer(self.config)
            summary = analyzer._calculate_metrics(trades_df, equity_curve)
            
            logger.info(f"Backtest completed: {len(trades_df)} trades, "
                       f"Final equity: ${equity_curve.iloc[-1]:,.2f}")
            
            return trades_df, equity_curve, True, None, summary
            
        except Exception as e:
            logger.error(f"Backtest failed: {e}", exc_info=True)
            return pd.DataFrame(), pd.Series([initial_equity]), False, None, {}

    def _simulate_trades(self, signals_df: pd.DataFrame, initial_equity: float) -> Tuple[pd.DataFrame, pd.Series]:
        """Simulate trade execution based on signals."""
        
        trades = []
        equity_values = [initial_equity]
        current_equity = initial_equity
        position = None  # Current position: None, 'long', or 'short'
        entry_price = None
        entry_time = None
        trade_id = 0
        
        # Risk management parameters
        risk_per_trade = getattr(self.config, 'BASE_RISK_PER_TRADE_PCT', 0.02)
        max_risk_usd = getattr(self.config, 'RISK_CAP_PER_TRADE_USD', 500)
        
        for idx, row in signals_df.iterrows():
            current_price = row['Close']
            signal = row['prediction']
            confidence = row['confidence']
            timestamp = row['timestamp']
            
            # Close existing position if signal changes
            if position is not None and (
                (position == 'long' and signal <= 0) or 
                (position == 'short' and signal >= 0)
            ):
                # Close position
                if position == 'long':
                    pnl = (current_price - entry_price) * position_size
                else:  # short
                    pnl = (entry_price - current_price) * position_size
                
                current_equity += pnl
                
                trades.append({
                    'trade_id': trade_id,
                    'entry_time': entry_time,
                    'exit_time': timestamp,
                    'side': position,
                    'entry_price': entry_price,
                    'exit_price': current_price,
                    'position_size': position_size,
                    'pnl': pnl,
                    'confidence': confidence
                })
                
                trade_id += 1
                position = None
                entry_price = None
                entry_time = None
            
            # Open new position
            if position is None and signal != 0:
                # Calculate position size based on risk
                risk_amount = min(current_equity * risk_per_trade, max_risk_usd)
                
                # Simple position sizing (can be enhanced)
                if signal > 0:  # Long
                    position = 'long'
                    position_size = risk_amount / (current_price * 0.02)  # Assume 2% stop loss
                else:  # Short
                    position = 'short'
                    position_size = risk_amount / (current_price * 0.02)
                
                entry_price = current_price
                entry_time = timestamp
            
            equity_values.append(current_equity)
        
        # Close any remaining position at the end
        if position is not None:
            final_price = signals_df.iloc[-1]['Close']
            if position == 'long':
                pnl = (final_price - entry_price) * position_size
            else:
                pnl = (entry_price - final_price) * position_size
            
            current_equity += pnl
            
            trades.append({
                'trade_id': trade_id,
                'entry_time': entry_time,
                'exit_time': signals_df.index[-1],
                'side': position,
                'entry_price': entry_price,
                'exit_price': final_price,
                'position_size': position_size,
                'pnl': pnl,
                'confidence': signals_df.iloc[-1]['confidence']
            })
            
            equity_values.append(current_equity)
        
        # Create DataFrames
        trades_df = pd.DataFrame(trades)
        equity_curve = pd.Series(equity_values, index=range(len(equity_values)))
        
        return trades_df, equity_curve

    def calculate_dynamic_weights(self, shap_summaries: Dict, performance_history: List[Dict]) -> Dict[str, float]:
        """Calculate dynamic ensemble weights based on SHAP importance and performance."""
        if not shap_summaries or not performance_history:
            # Equal weights if no data
            return {model: 1.0 for model in shap_summaries.keys()}
        
        weights = {}
        total_weight = 0
        
        for model_name in shap_summaries.keys():
            # Base weight from SHAP importance
            shap_df = shap_summaries[model_name]
            if shap_df is not None and not shap_df.empty:
                # Use mean SHAP importance as base weight
                importance_col = 'SHAP_Importance' if 'SHAP_Importance' in shap_df.columns else shap_df.columns[1]
                base_weight = shap_df[importance_col].mean()
            else:
                base_weight = 0.5
            
            # Adjust based on recent performance
            recent_performance = [p for p in performance_history if p.get('model') == model_name]
            if recent_performance:
                avg_performance = np.mean([p.get('f1_score', 0.5) for p in recent_performance[-5:]])  # Last 5 cycles
                performance_multiplier = avg_performance / 0.5  # Normalize around 0.5
            else:
                performance_multiplier = 1.0
            
            final_weight = base_weight * performance_multiplier
            weights[model_name] = max(0.1, final_weight)  # Minimum weight of 0.1
            total_weight += weights[model_name]
        
        # Normalize weights
        if total_weight > 0:
            weights = {model: weight / total_weight for model, weight in weights.items()}
        
        return weights

    @staticmethod
    def validate_holdout_performance(guardrail_val_data: pd.DataFrame, train_result: Dict, 
                                   config: ConfigModel, feature_list: List[str]) -> Dict[str, Any]:
        """Validate model performance on holdout data."""
        try:
            model = train_result.get('model')
            if model is None:
                return {'error': 'No model available for validation'}
            
            # Prepare validation data
            X_val = guardrail_val_data[feature_list].fillna(0)
            y_val = guardrail_val_data.get('target_signal_pressure_class', pd.Series())
            
            if y_val.empty:
                return {'error': 'No target variable for validation'}
            
            # Make predictions
            predictions = model.predict(X_val)
            
            # Calculate metrics
            from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
            
            metrics = {
                'accuracy': accuracy_score(y_val, predictions),
                'f1_score': f1_score(y_val, predictions, average='weighted', zero_division=0),
                'precision': precision_score(y_val, predictions, average='weighted', zero_division=0),
                'recall': recall_score(y_val, predictions, average='weighted', zero_division=0)
            }
            
            # Run simple backtest for financial metrics
            backtester = Backtester(config)
            trades_df, equity_curve, success, _, backtest_metrics = backtester.run_backtest_chunk(
                guardrail_val_data, train_result, config.INITIAL_CAPITAL, feature_list
            )
            
            # Combine metrics
            validation_results = {**metrics, **backtest_metrics}
            
            logger.info(f"Holdout validation - F1: {metrics['f1_score']:.3f}, "
                       f"PnL: {backtest_metrics.get('total_net_profit', 0):.2f}")
            
            return validation_results
            
        except Exception as e:
            logger.error(f"Holdout validation failed: {e}")
            return {'error': str(e)}

    @staticmethod
    def get_last_run_params(params_log_file: str) -> Optional[Dict]:
        """Get parameters from last run for drift analysis."""
        try:
            import json
            if not os.path.exists(params_log_file):
                return None
            
            with open(params_log_file, 'r') as f:
                lines = f.readlines()
                if lines:
                    return json.loads(lines[-1])
            return None
        except Exception as e:
            logger.warning(f"Could not read last run params: {e}")
            return None

    @staticmethod
    def calculate_parameter_drift(current_params: Dict, last_params: Dict) -> float:
        """Calculate parameter drift percentage between runs."""
        try:
            total_drift = 0
            param_count = 0
            
            for key in current_params:
                if key in last_params:
                    current_val = current_params[key]
                    last_val = last_params[key]
                    
                    if isinstance(current_val, (int, float)) and isinstance(last_val, (int, float)):
                        if last_val != 0:
                            drift = abs((current_val - last_val) / last_val) * 100
                            total_drift += drift
                            param_count += 1
            
            return total_drift / param_count if param_count > 0 else 0
            
        except Exception as e:
            logger.warning(f"Could not calculate parameter drift: {e}")
            return 0

    @staticmethod
    def validate_ai_suggestion(suggested_params: Dict, historical_performance_log: List, 
                             holdout_data: pd.DataFrame, config: ConfigModel, 
                             model_trainer, symbol: str, playbook: Dict) -> bool:
        """Validate AI suggestions against guardrails."""
        logger.info("AI SUGGESTION VALIDATION PROTOCOL INITIATED")
        
        try:
            # Check parameter drift
            last_run_params = Backtester.get_last_run_params(getattr(config, 'PARAMS_LOG_FILE', 'params.log'))
            if last_run_params:
                param_drift = Backtester.calculate_parameter_drift(suggested_params, last_run_params)
                max_drift = getattr(config, 'MAX_PARAM_DRIFT_TOLERANCE', 50)
                
                logger.info(f"Parameter drift: {param_drift:.2f}% (Tolerance: {max_drift}%)")
                if param_drift > max_drift:
                    logger.warning("GUARDRAIL FAILED: Excessive parameter drift. REJECTING.")
                    return False
            
            # Additional validation logic can be added here
            logger.info("AI suggestion validation PASSED")
            return True
            
        except Exception as e:
            logger.error(f"AI suggestion validation failed: {e}")
            return False
