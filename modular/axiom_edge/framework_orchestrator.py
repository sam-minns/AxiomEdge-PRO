# =============================================================================
# FRAMEWORK ORCHESTRATOR MODULE
# =============================================================================

import os
import logging
import time
import json
import pickle
import warnings
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple
import pandas as pd
import numpy as np

from .config import ConfigModel
from .data_handler import DataHandler
from .feature_engineer import FeatureEngineer
from .model_trainer import ModelTrainer
from .genetic_programmer import GeneticProgrammer
from .backtester import Backtester
from .report_generator import ReportGenerator
from .ai_analyzer import GeminiAnalyzer, APITimer
from .utils import get_optimal_system_settings

# Optional imports with fallbacks
try:
    from .telemetry import TelemetryCollector, TelemetryAnalyzer
    TELEMETRY_AVAILABLE = True
except ImportError:
    TELEMETRY_AVAILABLE = False
    warnings.warn("Telemetry modules not available. Advanced monitoring will be limited.")

logger = logging.getLogger(__name__)

class FrameworkOrchestrator:
    """
    Advanced master orchestrator for the complete AxiomEdge trading framework.

    Coordinates all framework components including data collection, feature engineering,
    model training, backtesting, genetic programming, and reporting. Manages walk-forward
    analysis, framework memory, adaptive parameter optimization, and comprehensive telemetry.

    Features:
    - Complete workflow automation and coordination
    - Advanced walk-forward analysis with multiple cycles and regime detection
    - AI-guided hyperparameter optimization with learning capabilities
    - Framework memory and adaptive learning from historical performance
    - Dynamic parameter adjustment based on market conditions and performance
    - Comprehensive telemetry and monitoring with AI doctor capabilities
    - Multi-symbol and multi-timeframe support with intelligent data management
    - Professional reporting and visualization with interactive dashboards
    - Regime-adaptive confidence gates and risk management
    - Real-time performance monitoring and intervention capabilities
    - Advanced ensemble model coordination and voting
    - Persistent framework state management and recovery
    - Automated model retraining and strategy evolution
    - Comprehensive error handling and recovery mechanisms
    """
    
    def __init__(self, config: ConfigModel):
        """Initialize the Advanced Framework Orchestrator with comprehensive configuration."""
        self.config = config

        # Initialize core components
        self.data_handler = DataHandler()
        self.gemini_analyzer = GeminiAnalyzer()
        self.api_timer = APITimer()

        # Initialize framework components
        timeframe_roles = getattr(config, 'TIMEFRAME_ROLES', {'base': 'D1'})
        playbook = getattr(config, 'STRATEGY_PLAYBOOK', {})

        self.feature_engineer = FeatureEngineer(config, timeframe_roles, playbook)
        self.model_trainer = ModelTrainer(config, self.gemini_analyzer)
        self.backtester = Backtester(config)
        self.report_generator = ReportGenerator(config)

        # Initialize genetic programmer if enabled
        if getattr(config, 'ENABLE_GENETIC_PROGRAMMING', True):
            continuous_features = getattr(config, 'CONTINUOUS_FEATURES', [])
            state_features = getattr(config, 'STATE_FEATURES', [])
            self.genetic_programmer = GeneticProgrammer(
                config, continuous_features, state_features
            )
        else:
            self.genetic_programmer = None

        # Initialize telemetry if available
        if TELEMETRY_AVAILABLE and getattr(config, 'ENABLE_TELEMETRY', True):
            self.telemetry_collector = TelemetryCollector(config)
            self.telemetry_analyzer = TelemetryAnalyzer(config)
        else:
            self.telemetry_collector = None
            self.telemetry_analyzer = None

        # Enhanced framework state management
        self.framework_memory = {
            'historical_runs': [],
            'best_performance': {},
            'intervention_history': {},
            'cycle_telemetry': [],
            'regime_history': [],
            'model_performance_tracking': {},
            'adaptive_parameters': {},
            'error_recovery_log': []
        }

        # Performance tracking and state
        self.cycle_metrics = []
        self.aggregated_shap = None
        self.trades_df = None
        self.equity_curve = None
        self.current_regime = "Unknown"
        self.framework_state = "INITIALIZED"

        # Advanced configuration
        self.enable_regime_detection = getattr(config, 'ENABLE_REGIME_DETECTION', True)
        self.enable_adaptive_parameters = getattr(config, 'ENABLE_ADAPTIVE_PARAMETERS', True)
        self.enable_intervention_system = getattr(config, 'ENABLE_INTERVENTION_SYSTEM', True)
        self.framework_memory_path = getattr(config, 'FRAMEWORK_MEMORY_PATH', 'framework_memory.pkl')

        # Load persistent framework memory if available
        self._load_framework_memory()

        # Initialize system optimization
        optimal_settings = get_optimal_system_settings()
        logger.info(f"System optimization: {optimal_settings}")

        logger.info("Advanced FrameworkOrchestrator initialized with enhanced capabilities")

    def _load_framework_memory(self):
        """Load persistent framework memory from disk."""
        try:
            if os.path.exists(self.framework_memory_path):
                with open(self.framework_memory_path, 'rb') as f:
                    saved_memory = pickle.load(f)

                # Merge with current memory structure
                for key, value in saved_memory.items():
                    if key in self.framework_memory:
                        if isinstance(value, list):
                            self.framework_memory[key].extend(value)
                        elif isinstance(value, dict):
                            self.framework_memory[key].update(value)
                        else:
                            self.framework_memory[key] = value

                logger.info(f"Loaded framework memory from {self.framework_memory_path}")
                logger.info(f"Historical runs: {len(self.framework_memory['historical_runs'])}")

        except Exception as e:
            logger.warning(f"Could not load framework memory: {e}")

    def _save_framework_memory(self):
        """Save framework memory to disk for persistence."""
        try:
            with open(self.framework_memory_path, 'wb') as f:
                pickle.dump(self.framework_memory, f)
            logger.info(f"Saved framework memory to {self.framework_memory_path}")
        except Exception as e:
            logger.error(f"Could not save framework memory: {e}")

    def _detect_market_regime(self, data: pd.DataFrame) -> str:
        """
        Detect current market regime using multiple indicators.

        Args:
            data: Market data for regime analysis

        Returns:
            Detected regime string
        """
        if not self.enable_regime_detection or data.empty:
            return "Unknown"

        try:
            # Use recent data for regime detection
            recent_data = data.tail(252)  # Last year of data

            if 'Close' not in recent_data.columns:
                return "Unknown"

            # Calculate regime indicators
            returns = recent_data['Close'].pct_change().dropna()

            # Volatility regime
            volatility = returns.std() * np.sqrt(252)

            # Trend regime
            sma_20 = recent_data['Close'].rolling(20).mean()
            sma_50 = recent_data['Close'].rolling(50).mean()
            trend_strength = (sma_20.iloc[-1] - sma_50.iloc[-1]) / sma_50.iloc[-1]

            # Determine regime
            if volatility > 0.25:
                if abs(trend_strength) > 0.05:
                    regime = "High_Volatility_Trending"
                else:
                    regime = "High_Volatility_Ranging"
            elif volatility > 0.15:
                if abs(trend_strength) > 0.03:
                    regime = "Medium_Volatility_Trending"
                else:
                    regime = "Medium_Volatility_Ranging"
            else:
                if abs(trend_strength) > 0.02:
                    regime = "Low_Volatility_Trending"
                else:
                    regime = "Low_Volatility_Ranging"

            # Store regime history
            self.framework_memory['regime_history'].append({
                'timestamp': datetime.now(),
                'regime': regime,
                'volatility': volatility,
                'trend_strength': trend_strength
            })

            logger.info(f"Detected market regime: {regime} (Vol: {volatility:.3f}, Trend: {trend_strength:.3f})")
            return regime

        except Exception as e:
            logger.error(f"Regime detection failed: {e}")
            return "Unknown"

    def _apply_adaptive_parameters(self, regime: str) -> Dict[str, Any]:
        """
        Apply adaptive parameters based on current market regime and historical performance.

        Args:
            regime: Current market regime

        Returns:
            Dictionary of adaptive parameters
        """
        if not self.enable_adaptive_parameters:
            return {}

        try:
            # Base parameters
            adaptive_params = {
                'confidence_threshold': 0.6,
                'position_size_multiplier': 1.0,
                'stop_loss_multiplier': 1.0,
                'take_profit_multiplier': 1.0
            }

            # Regime-specific adjustments
            if "High_Volatility" in regime:
                adaptive_params['confidence_threshold'] = 0.7
                adaptive_params['position_size_multiplier'] = 0.8
                adaptive_params['stop_loss_multiplier'] = 1.2
            elif "Low_Volatility" in regime:
                adaptive_params['confidence_threshold'] = 0.5
                adaptive_params['position_size_multiplier'] = 1.2
                adaptive_params['take_profit_multiplier'] = 0.8

            if "Trending" in regime:
                adaptive_params['take_profit_multiplier'] *= 1.3
            elif "Ranging" in regime:
                adaptive_params['stop_loss_multiplier'] *= 0.8

            # Historical performance adjustments
            if self.framework_memory['historical_runs']:
                recent_performance = self.framework_memory['historical_runs'][-5:]  # Last 5 runs
                avg_sharpe = np.mean([run.get('sharpe_ratio', 0) for run in recent_performance])

                if avg_sharpe < 0.5:
                    adaptive_params['confidence_threshold'] += 0.1
                    adaptive_params['position_size_multiplier'] *= 0.9
                elif avg_sharpe > 1.5:
                    adaptive_params['confidence_threshold'] -= 0.05
                    adaptive_params['position_size_multiplier'] *= 1.1

            # Store adaptive parameters
            self.framework_memory['adaptive_parameters'][regime] = adaptive_params

            logger.info(f"Applied adaptive parameters for {regime}: {adaptive_params}")
            return adaptive_params

        except Exception as e:
            logger.error(f"Adaptive parameter application failed: {e}")
            return {}

    def run_complete_framework(self, data_files: List[str],
                              symbols: Optional[List[str]] = None,
                              start_date: Optional[str] = None,
                              end_date: Optional[str] = None) -> Dict[str, Any]:
        """
        Run the complete AxiomEdge framework with walk-forward analysis.
        
        Args:
            data_files: List of data files to process
            symbols: Optional list of symbols to analyze
            start_date: Optional start date for analysis
            end_date: Optional end date for analysis
            
        Returns:
            Complete framework results
        """
        logger.info("🚀 Starting Complete AxiomEdge Framework")
        logger.info("=" * 80)
        
        start_time = time.time()
        
        try:
            # Update framework state
            self.framework_state = "RUNNING"

            # Initialize telemetry collection if available
            if self.telemetry_collector:
                self.telemetry_collector.start_collection()

            # Stage 1: Data Collection and Preparation
            logger.info("📊 Stage 1: Enhanced Data Collection and Preparation")
            data_dict = self._collect_and_prepare_data(data_files, symbols, start_date, end_date)

            if not data_dict:
                self.framework_state = "FAILED"
                return {"error": "No data collected", "status": "failed"}

            # Stage 2: Market Regime Detection
            logger.info("🎯 Stage 2: Market Regime Detection")
            combined_data = pd.concat(data_dict.values(), ignore_index=True) if data_dict else pd.DataFrame()
            self.current_regime = self._detect_market_regime(combined_data)

            # Stage 3: Adaptive Parameter Configuration
            logger.info("⚙️ Stage 3: Adaptive Parameter Configuration")
            adaptive_params = self._apply_adaptive_parameters(self.current_regime)

            # Stage 4: Feature Engineering
            logger.info("🔧 Stage 4: Advanced Feature Engineering")
            engineered_data = self._engineer_features(data_dict)

            if engineered_data.empty:
                self.framework_state = "FAILED"
                return {"error": "Feature engineering failed", "status": "failed"}

            # Stage 5: Enhanced Walk-Forward Analysis
            logger.info("📈 Stage 5: Enhanced Walk-Forward Analysis with Regime Adaptation")
            walk_forward_results = self._run_enhanced_walk_forward_analysis(
                engineered_data, self.current_regime, adaptive_params
            )

            # Stage 6: Strategy Evolution (if enabled)
            evolution_results = {}
            if getattr(self.config, 'ENABLE_GENETIC_PROGRAMMING', False) and self.genetic_programmer:
                logger.info("🧬 Stage 6: Advanced Strategy Evolution")
                evolution_results = self._run_strategy_evolution(engineered_data)
                walk_forward_results['evolution'] = evolution_results

            # Stage 7: Performance Analysis and Intervention Check
            logger.info("🔍 Stage 7: Performance Analysis and Intervention Check")
            intervention_results = self._check_performance_intervention(walk_forward_results)

            # Stage 8: Comprehensive Final Reporting
            logger.info("📋 Stage 8: Comprehensive Final Reporting")
            final_metrics = self._generate_enhanced_final_report(walk_forward_results, evolution_results)

            # Stage 9: Framework Memory Update
            logger.info("💾 Stage 9: Framework Memory Update")
            self._update_framework_memory(walk_forward_results, final_metrics)

            # Calculate execution time
            execution_time = time.time() - start_time
            self.framework_state = "COMPLETED"

            # Stop telemetry collection
            if self.telemetry_collector:
                telemetry_summary = self.telemetry_collector.stop_collection()
                final_metrics['telemetry'] = telemetry_summary

            # Compile enhanced final results
            results = {
                'status': 'completed',
                'execution_time_seconds': execution_time,
                'framework_version': getattr(self.config, 'FRAMEWORK_VERSION', '2.1.1'),
                'regime_detected': self.current_regime,
                'adaptive_parameters_used': adaptive_params,
                'data_files_processed': len(data_files),
                'symbols_analyzed': list(data_dict.keys()) if data_dict else [],
                'features_engineered': len(engineered_data.columns) if not engineered_data.empty else 0,
                'walk_forward_cycles': len(self.cycle_metrics),
                'successful_cycles': len([c for c in self.cycle_metrics if c.get('status') == 'completed']),
                'final_metrics': final_metrics,
                'cycle_breakdown': self.cycle_metrics,
                'evolution_results': evolution_results,
                'intervention_results': intervention_results,
                'framework_memory_summary': {
                    'total_historical_runs': len(self.framework_memory['historical_runs']),
                    'regime_transitions': len(self.framework_memory['regime_history']),
                    'interventions_applied': len(self.framework_memory['intervention_history'])
                }
            }
            
            logger.info("✅ Complete AxiomEdge Framework Execution Completed")
            logger.info(f"   Total execution time: {execution_time:.2f} seconds")
            logger.info(f"   Cycles completed: {len(self.cycle_metrics)}")
            
            return results
            
        except Exception as e:
            logger.error(f"❌ Framework execution failed: {e}", exc_info=True)
            return {
                'status': 'failed',
                'error': str(e),
                'execution_time_seconds': time.time() - start_time
            }

    def _collect_and_prepare_data(self, data_files: List[str], 
                                 symbols: Optional[List[str]] = None,
                                 start_date: Optional[str] = None,
                                 end_date: Optional[str] = None) -> Dict[str, pd.DataFrame]:
        """Collect and prepare data for analysis."""
        data_dict = {}
        
        try:
            if data_files:
                # Load data from files
                for file_path in data_files:
                    if os.path.exists(file_path):
                        logger.info(f"   Loading data from: {file_path}")
                        df = pd.read_csv(file_path)
                        
                        # Determine symbol name from file or data
                        if 'Symbol' in df.columns:
                            symbol = df['Symbol'].iloc[0]
                        else:
                            symbol = os.path.basename(file_path).split('.')[0]
                        
                        # Prepare data format
                        if 'timestamp' in df.columns:
                            df['timestamp'] = pd.to_datetime(df['timestamp'])
                            df.set_index('timestamp', inplace=True)
                        elif 'Date' in df.columns:
                            df['Date'] = pd.to_datetime(df['Date'])
                            df.set_index('Date', inplace=True)
                        
                        # Ensure required columns
                        required_cols = ['Open', 'High', 'Low', 'Close']
                        if all(col in df.columns for col in required_cols):
                            data_dict[symbol] = df
                            logger.info(f"   ✅ Loaded {len(df)} records for {symbol}")
                        else:
                            logger.warning(f"   ⚠️  Missing required columns in {file_path}")
                    else:
                        logger.warning(f"   ⚠️  File not found: {file_path}")
            
            elif symbols:
                # Collect data using data handler
                for symbol in symbols:
                    logger.info(f"   Collecting data for: {symbol}")
                    try:
                        data = self.data_handler.get_data(
                            symbol=symbol,
                            start_date=start_date or "2023-01-01",
                            end_date=end_date or datetime.now().strftime("%Y-%m-%d"),
                            source="yahoo"
                        )
                        if data is not None and not data.empty:
                            data_dict[symbol] = data
                            logger.info(f"   ✅ Collected {len(data)} records for {symbol}")
                        else:
                            logger.warning(f"   ⚠️  No data collected for {symbol}")
                    except Exception as e:
                        logger.error(f"   ❌ Error collecting data for {symbol}: {e}")
            
            logger.info(f"📊 Data collection completed: {len(data_dict)} symbols")
            return data_dict
            
        except Exception as e:
            logger.error(f"Data collection failed: {e}", exc_info=True)
            return {}

    def _engineer_features(self, data_dict: Dict[str, pd.DataFrame]) -> pd.DataFrame:
        """Engineer features from collected data."""
        try:
            if not data_dict:
                return pd.DataFrame()
            
            # For simplicity, use the first symbol's data for single-symbol analysis
            # In a multi-symbol framework, this would be more sophisticated
            symbol = list(data_dict.keys())[0]
            base_data = data_dict[symbol]
            
            logger.info(f"   Engineering features for {symbol} ({len(base_data)} records)")
            
            # Prepare data for feature engineering
            data_by_tf = {'D1': base_data}  # Single timeframe for now
            
            # Engineer features
            engineered_data = self.feature_engineer.engineer_features(
                base_df=base_data,
                data_by_tf=data_by_tf,
                macro_data=None
            )
            
            # Generate labels for training
            labeled_data = self.feature_engineer.label_data_multi_task(engineered_data)
            
            logger.info(f"   ✅ Feature engineering completed: {labeled_data.shape}")
            return labeled_data
            
        except Exception as e:
            logger.error(f"Feature engineering failed: {e}", exc_info=True)
            return pd.DataFrame()

    def _run_enhanced_walk_forward_analysis(self, data: pd.DataFrame, regime: str,
                                          adaptive_params: Dict[str, Any]) -> Dict[str, Any]:
        """
        Run enhanced walk-forward analysis with regime adaptation and advanced features.

        Args:
            data: Engineered feature data
            regime: Current market regime
            adaptive_params: Adaptive parameters for current regime

        Returns:
            Enhanced walk-forward analysis results
        """
        try:
            # Enhanced configuration for walk-forward analysis
            training_window = getattr(self.config, 'TRAINING_WINDOW_DAYS', 252)
            test_window = getattr(self.config, 'TEST_WINDOW_DAYS', 63)
            min_training_samples = getattr(self.config, 'MIN_TRAINING_SAMPLES', 1000)

            # Regime-specific adjustments
            if "High_Volatility" in regime:
                training_window = int(training_window * 1.2)  # Longer training in volatile markets
            elif "Low_Volatility" in regime:
                training_window = int(training_window * 0.8)  # Shorter training in stable markets

            logger.info(f"Enhanced walk-forward analysis with regime: {regime}")
            logger.info(f"Training window: {training_window} days, Test window: {test_window} days")

            if len(data) < min_training_samples:
                logger.warning(f"Insufficient data for walk-forward analysis: {len(data)} < {min_training_samples}")
                return {'status': 'insufficient_data', 'data_length': len(data)}

            # Calculate number of cycles
            available_data = len(data) - training_window
            num_cycles = max(1, available_data // test_window)

            logger.info(f"Planning {num_cycles} walk-forward cycles")

            # Initialize tracking variables
            all_predictions = []
            all_actuals = []
            cycle_results = []

            # Run enhanced cycles
            for cycle in range(num_cycles):
                start_idx = cycle * test_window
                end_idx = start_idx + training_window
                test_start_idx = end_idx
                test_end_idx = min(test_start_idx + test_window, len(data))

                if test_end_idx <= test_start_idx:
                    logger.warning(f"Insufficient test data for cycle {cycle + 1}")
                    break

                # Extract training and test data
                train_data = data.iloc[start_idx:end_idx].copy()
                test_data = data.iloc[test_start_idx:test_end_idx].copy()

                logger.info(f"   🔄 Cycle {cycle + 1}/{num_cycles}: "
                           f"Train: {len(train_data)} samples, Test: {len(test_data)} samples")

                # Run enhanced single cycle
                cycle_result = self._run_enhanced_single_cycle(
                    train_data, cycle + 1, test_data, regime, adaptive_params
                )

                cycle_results.append(cycle_result)

                # Collect predictions and actuals
                if cycle_result.get('predictions') is not None:
                    all_predictions.extend(cycle_result['predictions'])
                if cycle_result.get('actuals') is not None:
                    all_actuals.extend(cycle_result['actuals'])

            # Calculate enhanced overall metrics
            overall_metrics = self._calculate_enhanced_overall_metrics(
                all_predictions, all_actuals, cycle_results, regime
            )

            # Compile enhanced results
            results = {
                'status': 'completed',
                'regime': regime,
                'adaptive_parameters': adaptive_params,
                'num_cycles': len(cycle_results),
                'successful_cycles': len([c for c in cycle_results if c.get('status') == 'completed']),
                'overall_metrics': overall_metrics,
                'cycle_results': cycle_results,
                'training_window_used': training_window,
                'test_window_used': test_window
            }

            logger.info(f"Enhanced walk-forward analysis completed: {len(cycle_results)} cycles")
            return results

        except Exception as e:
            logger.error(f"Enhanced walk-forward analysis failed: {e}", exc_info=True)
            return {'status': 'failed', 'error': str(e)}

    def _run_walk_forward_analysis(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Run walk-forward analysis with multiple cycles."""
        try:
            # Configuration for walk-forward analysis
            training_window = getattr(self.config, 'TRAINING_WINDOW_DAYS', 252)  # 1 year
            test_window = getattr(self.config, 'TEST_WINDOW_DAYS', 63)  # 3 months
            step_size = getattr(self.config, 'WALK_FORWARD_STEP_DAYS', 21)  # 1 month
            
            logger.info(f"   Training window: {training_window} days")
            logger.info(f"   Test window: {test_window} days")
            logger.info(f"   Step size: {step_size} days")
            
            # Calculate number of cycles
            total_data_points = len(data)
            min_data_needed = training_window + test_window
            
            if total_data_points < min_data_needed:
                logger.warning(f"   Insufficient data for walk-forward analysis: {total_data_points} < {min_data_needed}")
                # Run single cycle with available data
                return self._run_single_cycle(data, 1)
            
            # Calculate cycles
            max_cycles = (total_data_points - min_data_needed) // step_size + 1
            max_cycles = min(max_cycles, getattr(self.config, 'MAX_WALK_FORWARD_CYCLES', 5))
            
            logger.info(f"   Running {max_cycles} walk-forward cycles")
            
            # Run cycles
            all_predictions = []
            all_actuals = []
            
            for cycle in range(max_cycles):
                logger.info(f"\n   🔄 Cycle {cycle + 1}/{max_cycles}")
                
                # Define data windows
                start_idx = cycle * step_size
                train_end_idx = start_idx + training_window
                test_end_idx = min(train_end_idx + test_window, total_data_points)
                
                if test_end_idx <= train_end_idx:
                    logger.warning(f"   Insufficient data for cycle {cycle + 1}")
                    break
                
                # Split data
                train_data = data.iloc[start_idx:train_end_idx].copy()
                test_data = data.iloc[train_end_idx:test_end_idx].copy()
                
                logger.info(f"   Train: {len(train_data)} samples, Test: {len(test_data)} samples")
                
                # Run cycle
                cycle_results = self._run_single_cycle(train_data, cycle + 1, test_data)
                
                if cycle_results.get('status') == 'completed':
                    # Collect predictions for ensemble analysis
                    if 'predictions' in cycle_results:
                        all_predictions.extend(cycle_results['predictions'])
                    if 'actuals' in cycle_results:
                        all_actuals.extend(cycle_results['actuals'])
            
            # Aggregate results
            overall_metrics = self._calculate_overall_metrics(all_predictions, all_actuals)
            
            return {
                'status': 'completed',
                'cycles_completed': len(self.cycle_metrics),
                'overall_metrics': overall_metrics,
                'cycle_details': self.cycle_metrics
            }
            
        except Exception as e:
            logger.error(f"Walk-forward analysis failed: {e}", exc_info=True)
            return {'status': 'failed', 'error': str(e)}

    def _run_single_cycle(self, train_data: pd.DataFrame, cycle_num: int, 
                         test_data: Optional[pd.DataFrame] = None) -> Dict[str, Any]:
        """Run a single training and testing cycle."""
        try:
            logger.info(f"     🎯 Processing Cycle {cycle_num}")
            
            # Identify target columns
            target_columns = [col for col in train_data.columns if col.startswith('target_')]
            if not target_columns:
                logger.error("     No target columns found in training data")
                return {'status': 'failed', 'error': 'No target columns'}
            
            # Use primary target
            primary_target = 'target_signal_pressure_class'
            if primary_target not in target_columns:
                primary_target = target_columns[0]
            
            logger.info(f"     Using target: {primary_target}")
            
            # Train model
            logger.info("     🤖 Training model...")
            model_results = self.model_trainer.train_and_validate_model(train_data)
            
            if 'error' in model_results:
                logger.error(f"     Model training failed: {model_results['error']}")
                cycle_metrics = {
                    'cycle': cycle_num,
                    'status': 'failed',
                    'error': model_results['error'],
                    'metrics': {}
                }
                self.cycle_metrics.append(cycle_metrics)
                return cycle_metrics
            
            # Extract model and features
            model = model_results.get('model')
            selected_features = model_results.get('selected_features', [])
            
            # Test model if test data is provided
            predictions = []
            actuals = []
            test_metrics = {}
            
            if test_data is not None and model is not None:
                logger.info("     📊 Testing model on out-of-sample data...")
                
                # Prepare test features
                X_test = test_data[selected_features].fillna(0)
                y_test = test_data[primary_target]
                
                # Make predictions
                test_predictions = model.predict(X_test)
                test_probabilities = model.predict_proba(X_test) if hasattr(model, 'predict_proba') else None
                
                # Calculate test metrics
                from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
                
                test_metrics = {
                    'accuracy': accuracy_score(y_test, test_predictions),
                    'f1_score': f1_score(y_test, test_predictions, average='weighted', zero_division=0),
                    'precision': precision_score(y_test, test_predictions, average='weighted', zero_division=0),
                    'recall': recall_score(y_test, test_predictions, average='weighted', zero_division=0)
                }
                
                predictions = test_predictions.tolist()
                actuals = y_test.tolist()
                
                logger.info(f"     Test F1 Score: {test_metrics['f1_score']:.3f}")
            
            # Aggregate SHAP data
            if model_results.get('shap_summary') is not None:
                if self.aggregated_shap is None:
                    self.aggregated_shap = model_results['shap_summary'].copy()
                else:
                    # Simple aggregation - in practice, this would be more sophisticated
                    self.aggregated_shap = pd.concat([self.aggregated_shap, model_results['shap_summary']], ignore_index=True)
            
            # Create cycle metrics
            cycle_metrics = {
                'cycle': cycle_num,
                'status': 'completed',
                'metrics': {
                    **model_results.get('metrics', {}),
                    **test_metrics
                },
                'features_used': len(selected_features),
                'training_samples': len(train_data),
                'test_samples': len(test_data) if test_data is not None else 0,
                'predictions': predictions,
                'actuals': actuals
            }
            
            self.cycle_metrics.append(cycle_metrics)
            
            logger.info(f"     ✅ Cycle {cycle_num} completed successfully")
            return cycle_metrics
            
        except Exception as e:
            logger.error(f"Cycle {cycle_num} failed: {e}", exc_info=True)
            cycle_metrics = {
                'cycle': cycle_num,
                'status': 'failed',
                'error': str(e),
                'metrics': {}
            }
            self.cycle_metrics.append(cycle_metrics)
            return cycle_metrics

    def _run_enhanced_single_cycle(self, train_data: pd.DataFrame, cycle_num: int,
                                  test_data: Optional[pd.DataFrame] = None,
                                  regime: str = "Unknown",
                                  adaptive_params: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        Run an enhanced single training and testing cycle with regime adaptation.

        Args:
            train_data: Training data
            cycle_num: Cycle number
            test_data: Optional test data
            regime: Current market regime
            adaptive_params: Adaptive parameters for current regime

        Returns:
            Enhanced cycle results
        """
        try:
            logger.info(f"     🎯 Processing Enhanced Cycle {cycle_num} (Regime: {regime})")

            cycle_start_time = time.time()

            # Apply regime-specific feature selection
            feature_list = self._get_regime_specific_features(train_data, regime)

            # Enhanced model training with regime context
            logger.info(f"       🤖 Training ensemble models with {len(feature_list)} features...")
            training_results, training_error = self.model_trainer.train_all_models(
                df_train_labeled=train_data,
                feature_list=feature_list,
                framework_history=self.framework_memory,
                regime=regime
            )

            if training_results is None:
                logger.error(f"       ❌ Model training failed: {training_error}")
                return {
                    'cycle': cycle_num,
                    'status': 'training_failed',
                    'error': training_error,
                    'regime': regime
                }

            # Enhanced backtesting with regime-adaptive parameters
            if test_data is not None and not test_data.empty:
                logger.info(f"       📊 Running enhanced backtesting...")

                # Apply adaptive confidence threshold
                confidence_threshold = adaptive_params.get('confidence_threshold', 0.6) if adaptive_params else 0.6

                trades_df, equity_curve, backtest_success, additional_metrics, summary = self.backtester.run_backtest_chunk(
                    df_chunk=test_data,
                    training_results=training_results,
                    initial_equity=getattr(self.config, 'INITIAL_EQUITY', 100000),
                    feature_list=feature_list,
                    confidence_threshold=confidence_threshold,
                    regime=regime
                )

                if not backtest_success:
                    logger.warning(f"       ⚠️ Backtesting had issues for cycle {cycle_num}")

                # Store results for aggregation
                self.trades_df = trades_df
                self.equity_curve = equity_curve

                # Enhanced cycle metrics
                cycle_metrics = {
                    'cycle': cycle_num,
                    'status': 'completed',
                    'regime': regime,
                    'adaptive_params_used': adaptive_params,
                    'training_time_seconds': time.time() - cycle_start_time,
                    'features_used': len(feature_list),
                    'models_trained': len(training_results.get('trained_pipelines', {})),
                    'trades_generated': len(trades_df) if trades_df is not None else 0,
                    'backtest_summary': summary,
                    'confidence_threshold_used': confidence_threshold,
                    'training_results': training_results,
                    'predictions': None,  # Placeholder for compatibility
                    'actuals': None       # Placeholder for compatibility
                }

                # Add SHAP summaries if available
                if 'shap_summaries' in training_results:
                    cycle_metrics['shap_summaries'] = training_results['shap_summaries']

            else:
                # Training-only cycle
                cycle_metrics = {
                    'cycle': cycle_num,
                    'status': 'training_only',
                    'regime': regime,
                    'training_time_seconds': time.time() - cycle_start_time,
                    'features_used': len(feature_list),
                    'models_trained': len(training_results.get('trained_pipelines', {})),
                    'training_results': training_results
                }

            # Store cycle metrics
            self.cycle_metrics.append(cycle_metrics)

            logger.info(f"       ✅ Enhanced cycle {cycle_num} completed successfully")
            return cycle_metrics

        except Exception as e:
            logger.error(f"Enhanced cycle {cycle_num} failed: {e}", exc_info=True)
            cycle_metrics = {
                'cycle': cycle_num,
                'status': 'failed',
                'error': str(e),
                'regime': regime,
                'metrics': {}
            }
            self.cycle_metrics.append(cycle_metrics)
            return cycle_metrics

    def _get_regime_specific_features(self, data: pd.DataFrame, regime: str) -> List[str]:
        """
        Get regime-specific feature list based on market conditions.

        Args:
            data: Training data
            regime: Current market regime

        Returns:
            List of features optimized for current regime
        """
        try:
            # Get all available features
            all_features = [col for col in data.columns if col not in ['target_signal_pressure_class_h30', 'target_signal_pressure_class_h60', 'target_signal_pressure_class_h90']]

            # Regime-specific feature preferences
            if "High_Volatility" in regime:
                # Prefer volatility and momentum features
                preferred_patterns = ['volatility', 'atr', 'rsi', 'momentum', 'macd']
            elif "Low_Volatility" in regime:
                # Prefer mean reversion and trend features
                preferred_patterns = ['sma', 'ema', 'bollinger', 'stoch', 'williams']
            elif "Trending" in regime:
                # Prefer trend-following features
                preferred_patterns = ['ema', 'macd', 'adx', 'momentum', 'slope']
            elif "Ranging" in regime:
                # Prefer oscillator and mean reversion features
                preferred_patterns = ['rsi', 'stoch', 'williams', 'bollinger', 'cci']
            else:
                # Default: use all features
                return all_features

            # Filter features based on regime preferences
            regime_features = []
            for feature in all_features:
                feature_lower = feature.lower()
                if any(pattern in feature_lower for pattern in preferred_patterns):
                    regime_features.append(feature)

            # Ensure we have enough features
            if len(regime_features) < 20:
                regime_features = all_features

            logger.info(f"Selected {len(regime_features)} regime-specific features for {regime}")
            return regime_features

        except Exception as e:
            logger.error(f"Regime-specific feature selection failed: {e}")
            return [col for col in data.columns if col not in ['target_signal_pressure_class_h30', 'target_signal_pressure_class_h60', 'target_signal_pressure_class_h90']]

    def _check_performance_intervention(self, walk_forward_results: Dict[str, Any]) -> Dict[str, Any]:
        """
        Check if performance intervention is needed and apply corrective measures.

        Args:
            walk_forward_results: Results from walk-forward analysis

        Returns:
            Intervention results and actions taken
        """
        if not self.enable_intervention_system:
            return {'intervention_enabled': False}

        try:
            intervention_results = {
                'intervention_enabled': True,
                'checks_performed': [],
                'actions_taken': [],
                'recommendations': []
            }

            # Check overall performance
            overall_metrics = walk_forward_results.get('overall_metrics', {})
            sharpe_ratio = overall_metrics.get('sharpe_ratio', 0)
            max_drawdown = overall_metrics.get('max_drawdown_pct', 0)

            intervention_results['checks_performed'].append('overall_performance')

            # Intervention triggers
            if sharpe_ratio < 0.3:
                intervention_results['actions_taken'].append('low_sharpe_intervention')
                intervention_results['recommendations'].append('Consider increasing confidence thresholds')

                # Update adaptive parameters
                if self.current_regime in self.framework_memory['adaptive_parameters']:
                    self.framework_memory['adaptive_parameters'][self.current_regime]['confidence_threshold'] += 0.1

            if abs(max_drawdown) > 0.2:
                intervention_results['actions_taken'].append('high_drawdown_intervention')
                intervention_results['recommendations'].append('Implement stricter risk management')

                # Update position sizing
                if self.current_regime in self.framework_memory['adaptive_parameters']:
                    self.framework_memory['adaptive_parameters'][self.current_regime]['position_size_multiplier'] *= 0.8

            # Check cycle consistency
            successful_cycles = walk_forward_results.get('successful_cycles', 0)
            total_cycles = walk_forward_results.get('num_cycles', 1)
            success_rate = successful_cycles / total_cycles if total_cycles > 0 else 0

            intervention_results['checks_performed'].append('cycle_consistency')

            if success_rate < 0.7:
                intervention_results['actions_taken'].append('low_success_rate_intervention')
                intervention_results['recommendations'].append('Review data quality and feature engineering')

            # Log intervention
            if intervention_results['actions_taken']:
                self.framework_memory['intervention_history'].append({
                    'timestamp': datetime.now(),
                    'regime': self.current_regime,
                    'triggers': intervention_results['actions_taken'],
                    'metrics': overall_metrics
                })

                logger.info(f"Performance intervention applied: {intervention_results['actions_taken']}")

            return intervention_results

        except Exception as e:
            logger.error(f"Performance intervention check failed: {e}")
            return {'intervention_enabled': True, 'error': str(e)}

    def _run_strategy_evolution(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Run genetic programming for strategy evolution."""
        try:
            logger.info("   🧬 Initializing genetic programming...")
            
            # Create gene pool from available features
            feature_cols = [col for col in data.columns if not col.startswith('target_')]
            
            # Categorize features for genetic programming
            continuous_features = []
            state_features = []
            
            for col in feature_cols:
                if data[col].nunique() <= 10 and data[col].dtype in ['int64', 'bool']:
                    state_features.append(col)
                else:
                    continuous_features.append(col)
            
            gene_pool = {
                'continuous_features': continuous_features[:30],  # Limit for performance
                'state_features': state_features[:10],
                'comparison_operators': ['>', '<', '>=', '<='],
                'state_operators': ['==', '!='],
                'logical_operators': ['AND', 'OR'],
                'constants': [10, 20, 30, 40, 50, 60, 70, 80, 90]
            }
            
            logger.info(f"   Gene pool: {len(gene_pool['continuous_features'])} continuous, {len(gene_pool['state_features'])} state features")
            
            # Initialize genetic programmer
            gp = GeneticProgrammer(
                gene_pool=gene_pool,
                config=self.config,
                population_size=getattr(self.config, 'GP_POPULATION_SIZE', 30),
                generations=getattr(self.config, 'GP_GENERATIONS', 15),
                mutation_rate=getattr(self.config, 'GP_MUTATION_RATE', 0.1),
                crossover_rate=getattr(self.config, 'GP_CROSSOVER_RATE', 0.7)
            )
            
            # Run evolution
            best_chromosome, best_fitness = gp.run_evolution(data, self.gemini_analyzer, self.api_timer)
            
            evolution_results = {
                'status': 'completed',
                'best_fitness': best_fitness,
                'best_rules': {
                    'long_rule': best_chromosome[0] if best_chromosome[0] else "No rule evolved",
                    'short_rule': best_chromosome[1] if best_chromosome[1] else "No rule evolved"
                },
                'population_stats': gp.get_population_stats()
            }
            
            logger.info(f"   ✅ Strategy evolution completed. Best fitness: {best_fitness:.4f}")
            return evolution_results
            
        except Exception as e:
            logger.error(f"Strategy evolution failed: {e}", exc_info=True)
            return {'status': 'failed', 'error': str(e)}

    def _calculate_overall_metrics(self, all_predictions: List, all_actuals: List) -> Dict[str, Any]:
        """Calculate overall performance metrics across all cycles."""
        if not all_predictions or not all_actuals:
            return {}
        
        try:
            from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
            
            overall_metrics = {
                'total_predictions': len(all_predictions),
                'accuracy': accuracy_score(all_actuals, all_predictions),
                'f1_score': f1_score(all_actuals, all_predictions, average='weighted', zero_division=0),
                'precision': precision_score(all_actuals, all_predictions, average='weighted', zero_division=0),
                'recall': recall_score(all_actuals, all_predictions, average='weighted', zero_division=0)
            }
            
            # Calculate per-cycle average metrics
            if self.cycle_metrics:
                cycle_f1_scores = [cycle.get('metrics', {}).get('f1_score', 0) for cycle in self.cycle_metrics if cycle.get('status') == 'completed']
                if cycle_f1_scores:
                    overall_metrics['avg_cycle_f1'] = np.mean(cycle_f1_scores)
                    overall_metrics['std_cycle_f1'] = np.std(cycle_f1_scores)
            
            return overall_metrics
            
        except Exception as e:
            logger.error(f"Error calculating overall metrics: {e}")
            return {}

    def _calculate_enhanced_overall_metrics(self, all_predictions: List, all_actuals: List,
                                          cycle_results: List[Dict], regime: str) -> Dict[str, Any]:
        """
        Calculate enhanced overall performance metrics with regime context.

        Args:
            all_predictions: All predictions across cycles
            all_actuals: All actual values across cycles
            cycle_results: Results from all cycles
            regime: Current market regime

        Returns:
            Enhanced overall metrics
        """
        try:
            enhanced_metrics = {
                'regime': regime,
                'total_cycles': len(cycle_results),
                'successful_cycles': len([c for c in cycle_results if c.get('status') == 'completed']),
                'cycle_success_rate': 0.0
            }

            if enhanced_metrics['total_cycles'] > 0:
                enhanced_metrics['cycle_success_rate'] = enhanced_metrics['successful_cycles'] / enhanced_metrics['total_cycles']

            # Aggregate backtest summaries
            all_summaries = [c.get('backtest_summary', {}) for c in cycle_results if c.get('backtest_summary')]

            if all_summaries:
                # Calculate weighted averages
                total_trades = sum(s.get('total_trades', 0) for s in all_summaries)

                if total_trades > 0:
                    # Weighted metrics
                    enhanced_metrics['total_trades'] = total_trades
                    enhanced_metrics['win_rate'] = np.average(
                        [s.get('win_rate', 0) for s in all_summaries],
                        weights=[s.get('total_trades', 1) for s in all_summaries]
                    )
                    enhanced_metrics['profit_factor'] = np.average(
                        [s.get('profit_factor', 0) for s in all_summaries],
                        weights=[s.get('total_trades', 1) for s in all_summaries]
                    )
                    enhanced_metrics['sharpe_ratio'] = np.average(
                        [s.get('sharpe_ratio', 0) for s in all_summaries],
                        weights=[s.get('total_trades', 1) for s in all_summaries]
                    )
                    enhanced_metrics['max_drawdown_pct'] = max(
                        s.get('max_drawdown_pct', 0) for s in all_summaries
                    )

                    # Regime-specific performance analysis
                    enhanced_metrics['regime_performance'] = {
                        'regime_type': regime,
                        'performance_vs_baseline': self._calculate_regime_performance_comparison(enhanced_metrics),
                        'regime_specific_insights': self._generate_regime_insights(enhanced_metrics, regime)
                    }

            # Model performance tracking
            model_performances = []
            for cycle in cycle_results:
                training_results = cycle.get('training_results', {})
                if 'horizon_performance_metrics' in training_results:
                    model_performances.append(training_results['horizon_performance_metrics'])

            if model_performances:
                enhanced_metrics['model_performance_summary'] = self._aggregate_model_performance(model_performances)

            return enhanced_metrics

        except Exception as e:
            logger.error(f"Enhanced metrics calculation failed: {e}")
            return {'error': str(e), 'regime': regime}

    def _calculate_regime_performance_comparison(self, metrics: Dict[str, Any]) -> Dict[str, float]:
        """Calculate performance comparison against historical regime performance."""
        try:
            # Get historical performance for this regime
            historical_runs = self.framework_memory.get('historical_runs', [])
            regime_runs = [run for run in historical_runs if run.get('regime') == metrics.get('regime')]

            if not regime_runs:
                return {'comparison_available': False}

            # Calculate averages
            historical_sharpe = np.mean([run.get('sharpe_ratio', 0) for run in regime_runs])
            historical_win_rate = np.mean([run.get('win_rate', 0) for run in regime_runs])

            current_sharpe = metrics.get('sharpe_ratio', 0)
            current_win_rate = metrics.get('win_rate', 0)

            return {
                'comparison_available': True,
                'sharpe_improvement': current_sharpe - historical_sharpe,
                'win_rate_improvement': current_win_rate - historical_win_rate,
                'historical_runs_count': len(regime_runs)
            }

        except Exception as e:
            logger.error(f"Regime performance comparison failed: {e}")
            return {'comparison_available': False, 'error': str(e)}

    def _generate_regime_insights(self, metrics: Dict[str, Any], regime: str) -> List[str]:
        """Generate regime-specific performance insights."""
        insights = []

        try:
            sharpe_ratio = metrics.get('sharpe_ratio', 0)
            win_rate = metrics.get('win_rate', 0)

            if "High_Volatility" in regime:
                if sharpe_ratio > 1.0:
                    insights.append("Strong performance in high volatility environment")
                else:
                    insights.append("Consider more conservative position sizing in volatile markets")

            if "Trending" in regime:
                if win_rate > 0.6:
                    insights.append("Trend-following strategies performing well")
                else:
                    insights.append("May need to improve trend detection mechanisms")

            if "Ranging" in regime:
                if sharpe_ratio > 0.8:
                    insights.append("Mean reversion strategies effective in ranging market")
                else:
                    insights.append("Consider reducing trade frequency in ranging markets")

            return insights

        except Exception as e:
            logger.error(f"Regime insights generation failed: {e}")
            return ["Unable to generate regime insights"]

    def _aggregate_model_performance(self, model_performances: List[Dict]) -> Dict[str, Any]:
        """Aggregate model performance across cycles."""
        try:
            aggregated = {
                'models_evaluated': set(),
                'average_f1_scores': {},
                'model_consistency': {}
            }

            # Collect all model names and F1 scores
            all_f1_scores = {}
            for perf_dict in model_performances:
                for model_name, metrics in perf_dict.items():
                    aggregated['models_evaluated'].add(model_name)
                    if model_name not in all_f1_scores:
                        all_f1_scores[model_name] = []
                    all_f1_scores[model_name].append(metrics.get('f1_score', 0))

            # Calculate averages and consistency
            for model_name, f1_scores in all_f1_scores.items():
                aggregated['average_f1_scores'][model_name] = np.mean(f1_scores)
                aggregated['model_consistency'][model_name] = 1.0 - np.std(f1_scores)  # Higher is more consistent

            aggregated['models_evaluated'] = list(aggregated['models_evaluated'])
            return aggregated

        except Exception as e:
            logger.error(f"Model performance aggregation failed: {e}")
            return {}

    def _update_framework_memory(self, walk_forward_results: Dict[str, Any], final_metrics: Dict[str, Any]):
        """Update framework memory with current run results."""
        try:
            # Create run summary
            run_summary = {
                'timestamp': datetime.now(),
                'regime': self.current_regime,
                'framework_version': getattr(self.config, 'FRAMEWORK_VERSION', '2.1.1'),
                'cycles_completed': walk_forward_results.get('num_cycles', 0),
                'successful_cycles': walk_forward_results.get('successful_cycles', 0),
                'overall_metrics': walk_forward_results.get('overall_metrics', {}),
                'final_metrics': final_metrics,
                'execution_time': final_metrics.get('execution_time_seconds', 0)
            }

            # Add to historical runs
            self.framework_memory['historical_runs'].append(run_summary)

            # Update best performance tracking
            current_sharpe = walk_forward_results.get('overall_metrics', {}).get('sharpe_ratio', 0)
            best_sharpe = self.framework_memory['best_performance'].get('sharpe_ratio', 0)

            if current_sharpe > best_sharpe:
                self.framework_memory['best_performance'] = {
                    'sharpe_ratio': current_sharpe,
                    'regime': self.current_regime,
                    'timestamp': datetime.now(),
                    'run_summary': run_summary
                }
                logger.info(f"New best performance achieved: Sharpe {current_sharpe:.3f} in {self.current_regime}")

            # Limit memory size to prevent unbounded growth
            max_historical_runs = getattr(self.config, 'MAX_HISTORICAL_RUNS', 100)
            if len(self.framework_memory['historical_runs']) > max_historical_runs:
                self.framework_memory['historical_runs'] = self.framework_memory['historical_runs'][-max_historical_runs:]

            # Save to disk
            self._save_framework_memory()

            logger.info("Framework memory updated successfully")

        except Exception as e:
            logger.error(f"Framework memory update failed: {e}")

    def _generate_final_report(self) -> Dict[str, Any]:
        """Generate final comprehensive report."""
        try:
            logger.info("   📋 Generating comprehensive final report...")
            
            # Create sample equity curve for demonstration
            if self.cycle_metrics:
                # Simple equity curve based on cycle performance
                initial_capital = getattr(self.config, 'INITIAL_CAPITAL', 10000)
                equity_values = [initial_capital]
                
                for cycle in self.cycle_metrics:
                    if cycle.get('status') == 'completed':
                        # Simulate return based on F1 score
                        f1_score = cycle.get('metrics', {}).get('f1_score', 0)
                        simulated_return = (f1_score - 0.5) * 0.1  # Convert F1 to return
                        equity_values.append(equity_values[-1] * (1 + simulated_return))
                
                # Create equity curve series
                dates = pd.date_range(start=datetime.now() - timedelta(days=len(equity_values)), 
                                    periods=len(equity_values), freq='D')
                self.equity_curve = pd.Series(equity_values, index=dates)
            
            # Generate comprehensive report
            final_metrics = self.report_generator.generate_full_report(
                trades_df=self.trades_df,
                equity_curve=self.equity_curve,
                cycle_metrics=self.cycle_metrics,
                aggregated_shap=self.aggregated_shap,
                framework_memory=self.framework_memory
            )
            
            # Update framework memory
            self.framework_memory['historical_runs'].append({
                'timestamp': datetime.now().isoformat(),
                'final_metrics': final_metrics,
                'cycles_completed': len(self.cycle_metrics),
                'config_snapshot': self.config.model_dump()
            })
            
            logger.info("   ✅ Final report generated successfully")
            return final_metrics
            
        except Exception as e:
            logger.error(f"Final report generation failed: {e}", exc_info=True)
            return {}

    def _generate_enhanced_final_report(self, walk_forward_results: Dict[str, Any],
                                       evolution_results: Dict[str, Any]) -> Dict[str, Any]:
        """
        Generate enhanced comprehensive final report with advanced analytics.

        Args:
            walk_forward_results: Results from walk-forward analysis
            evolution_results: Results from strategy evolution

        Returns:
            Enhanced final report
        """
        try:
            logger.info("   📋 Generating enhanced comprehensive final report...")

            # Initialize enhanced report structure
            enhanced_report = {
                'report_metadata': {
                    'generation_timestamp': datetime.now().isoformat(),
                    'framework_version': getattr(self.config, 'FRAMEWORK_VERSION', '2.1.1'),
                    'regime_context': self.current_regime,
                    'report_type': 'enhanced_comprehensive'
                },
                'executive_summary': {},
                'performance_analytics': {},
                'regime_analysis': {},
                'model_insights': {},
                'risk_assessment': {},
                'recommendations': {},
                'technical_details': {}
            }

            # Executive Summary
            overall_metrics = walk_forward_results.get('overall_metrics', {})
            enhanced_report['executive_summary'] = {
                'total_cycles': walk_forward_results.get('num_cycles', 0),
                'successful_cycles': walk_forward_results.get('successful_cycles', 0),
                'success_rate': walk_forward_results.get('successful_cycles', 0) / max(1, walk_forward_results.get('num_cycles', 1)),
                'key_metrics': {
                    'sharpe_ratio': overall_metrics.get('sharpe_ratio', 0),
                    'win_rate': overall_metrics.get('win_rate', 0),
                    'profit_factor': overall_metrics.get('profit_factor', 0),
                    'max_drawdown': overall_metrics.get('max_drawdown_pct', 0)
                },
                'regime_performance': overall_metrics.get('regime_performance', {})
            }

            # Performance Analytics
            enhanced_report['performance_analytics'] = {
                'overall_metrics': overall_metrics,
                'cycle_breakdown': walk_forward_results.get('cycle_results', []),
                'model_performance': overall_metrics.get('model_performance_summary', {}),
                'historical_comparison': self._generate_historical_comparison()
            }

            # Regime Analysis
            enhanced_report['regime_analysis'] = {
                'current_regime': self.current_regime,
                'regime_history': self.framework_memory.get('regime_history', [])[-10:],  # Last 10 regime changes
                'regime_specific_performance': overall_metrics.get('regime_performance', {}),
                'adaptive_parameters_used': walk_forward_results.get('adaptive_parameters', {})
            }

            # Model Insights
            if self.aggregated_shap is not None:
                enhanced_report['model_insights'] = {
                    'feature_importance': self.aggregated_shap.head(20).to_dict('records'),
                    'shap_analysis_available': True
                }
            else:
                enhanced_report['model_insights'] = {
                    'shap_analysis_available': False,
                    'note': 'SHAP analysis not available for this run'
                }

            # Risk Assessment
            enhanced_report['risk_assessment'] = self._generate_risk_assessment(overall_metrics)

            # Recommendations
            enhanced_report['recommendations'] = self._generate_strategic_recommendations(
                overall_metrics, self.current_regime
            )

            # Technical Details
            enhanced_report['technical_details'] = {
                'framework_memory_summary': {
                    'total_historical_runs': len(self.framework_memory['historical_runs']),
                    'interventions_applied': len(self.framework_memory['intervention_history']),
                    'regime_transitions': len(self.framework_memory['regime_history'])
                },
                'evolution_results': evolution_results,
                'telemetry_available': self.telemetry_collector is not None
            }

            # Generate visualizations if possible
            try:
                if self.equity_curve is not None and not self.equity_curve.empty:
                    # Generate enhanced equity curve plot
                    self.report_generator.plot_enhanced_equity_curve(self.equity_curve, self.trades_df)
                    enhanced_report['visualizations'] = {
                        'equity_curve_generated': True,
                        'trade_analysis_available': self.trades_df is not None and not self.trades_df.empty
                    }

                    # Generate advanced trade analysis if trades available
                    if self.trades_df is not None and not self.trades_df.empty:
                        self.report_generator.plot_advanced_trade_analysis(self.trades_df)
                        enhanced_report['visualizations']['advanced_trade_analysis_generated'] = True

            except Exception as viz_error:
                logger.warning(f"Visualization generation failed: {viz_error}")
                enhanced_report['visualizations'] = {'error': str(viz_error)}

            # Generate text reports
            try:
                # Executive summary
                executive_summary_text = self.report_generator.generate_executive_summary(
                    overall_metrics, self.trades_df
                )
                enhanced_report['reports'] = {
                    'executive_summary_text': executive_summary_text,
                    'text_reports_generated': True
                }

                # Interactive dashboard if enabled
                if getattr(self.config, 'GENERATE_INTERACTIVE_DASHBOARD', True):
                    dashboard_html = self.report_generator.create_interactive_dashboard(
                        overall_metrics, self.trades_df, self.equity_curve, self.aggregated_shap
                    )
                    enhanced_report['reports']['interactive_dashboard'] = dashboard_html

            except Exception as report_error:
                logger.warning(f"Report generation failed: {report_error}")
                enhanced_report['reports'] = {'error': str(report_error)}

            logger.info("Enhanced comprehensive final report generated successfully")
            return enhanced_report

        except Exception as e:
            logger.error(f"Enhanced final report generation failed: {e}", exc_info=True)
            return {'error': str(e), 'status': 'report_generation_failed'}

    def _generate_historical_comparison(self) -> Dict[str, Any]:
        """Generate comparison with historical performance."""
        try:
            historical_runs = self.framework_memory.get('historical_runs', [])
            if len(historical_runs) < 2:
                return {'comparison_available': False, 'reason': 'insufficient_historical_data'}

            # Calculate historical averages
            historical_sharpe = np.mean([run.get('overall_metrics', {}).get('sharpe_ratio', 0) for run in historical_runs[:-1]])
            historical_win_rate = np.mean([run.get('overall_metrics', {}).get('win_rate', 0) for run in historical_runs[:-1]])

            # Current performance
            current_run = historical_runs[-1]
            current_sharpe = current_run.get('overall_metrics', {}).get('sharpe_ratio', 0)
            current_win_rate = current_run.get('overall_metrics', {}).get('win_rate', 0)

            return {
                'comparison_available': True,
                'historical_average_sharpe': historical_sharpe,
                'current_sharpe': current_sharpe,
                'sharpe_improvement': current_sharpe - historical_sharpe,
                'historical_average_win_rate': historical_win_rate,
                'current_win_rate': current_win_rate,
                'win_rate_improvement': current_win_rate - historical_win_rate,
                'historical_runs_analyzed': len(historical_runs) - 1
            }

        except Exception as e:
            logger.error(f"Historical comparison generation failed: {e}")
            return {'comparison_available': False, 'error': str(e)}

    def _generate_risk_assessment(self, metrics: Dict[str, Any]) -> Dict[str, Any]:
        """Generate comprehensive risk assessment."""
        try:
            risk_factors = []
            risk_score = 0  # 0-10 scale

            # Sharpe ratio assessment
            sharpe = metrics.get('sharpe_ratio', 0)
            if sharpe < 0.5:
                risk_factors.append("Low risk-adjusted returns")
                risk_score += 3
            elif sharpe < 1.0:
                risk_score += 1

            # Drawdown assessment
            max_dd = abs(metrics.get('max_drawdown_pct', 0))
            if max_dd > 0.2:
                risk_factors.append("High maximum drawdown")
                risk_score += 3
            elif max_dd > 0.1:
                risk_score += 1

            # Win rate assessment
            win_rate = metrics.get('win_rate', 0)
            if win_rate < 0.4:
                risk_factors.append("Low win rate")
                risk_score += 2

            # Determine risk level
            if risk_score <= 2:
                risk_level = "LOW"
            elif risk_score <= 5:
                risk_level = "MODERATE"
            elif risk_score <= 8:
                risk_level = "HIGH"
            else:
                risk_level = "VERY HIGH"

            return {
                'risk_level': risk_level,
                'risk_score': min(10, risk_score),
                'risk_factors': risk_factors if risk_factors else ["No significant risk factors identified"],
                'regime_specific_risks': self._assess_regime_specific_risks()
            }

        except Exception as e:
            logger.error(f"Risk assessment generation failed: {e}")
            return {'error': str(e)}

    def _assess_regime_specific_risks(self) -> List[str]:
        """Assess risks specific to current market regime."""
        risks = []

        if "High_Volatility" in self.current_regime:
            risks.append("Increased position sizing risk in volatile markets")
            risks.append("Higher probability of gap risk")

        if "Low_Volatility" in self.current_regime:
            risks.append("Potential for sudden volatility spikes")
            risks.append("Reduced profit opportunities")

        if "Trending" in self.current_regime:
            risks.append("Risk of trend reversal")

        if "Ranging" in self.current_regime:
            risks.append("Whipsaw risk in sideways markets")

        return risks if risks else ["No regime-specific risks identified"]

    def _generate_strategic_recommendations(self, metrics: Dict[str, Any], regime: str) -> List[str]:
        """Generate strategic recommendations based on performance and regime."""
        recommendations = []

        try:
            sharpe = metrics.get('sharpe_ratio', 0)
            win_rate = metrics.get('win_rate', 0)
            max_dd = abs(metrics.get('max_drawdown_pct', 0))

            # Performance-based recommendations
            if sharpe < 1.0:
                recommendations.append("Consider improving risk-adjusted returns through better position sizing")

            if win_rate < 0.45:
                recommendations.append("Focus on improving trade selection criteria")

            if max_dd > 0.15:
                recommendations.append("Implement stricter risk management controls")

            # Regime-specific recommendations
            if "High_Volatility" in regime:
                recommendations.append("Consider reducing position sizes during high volatility periods")

            if "Trending" in regime:
                recommendations.append("Optimize trend-following strategies for current market conditions")

            if "Ranging" in regime:
                recommendations.append("Focus on mean reversion strategies in ranging markets")

            # Framework-specific recommendations
            if len(self.framework_memory['historical_runs']) > 5:
                recommendations.append("Leverage historical performance data for parameter optimization")

            return recommendations if recommendations else ["Current performance is satisfactory - maintain strategy"]

        except Exception as e:
            logger.error(f"Strategic recommendations generation failed: {e}")
            return ["Unable to generate recommendations due to error"]

    def run_single_cycle_analysis(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Run a single cycle analysis for testing purposes."""
        logger.info("🔬 Running Single Cycle Analysis")
        
        try:
            # Engineer features
            engineered_data = self._engineer_features({'TEST': data})
            
            if engineered_data.empty:
                return {'status': 'failed', 'error': 'Feature engineering failed'}
            
            # Run single cycle
            cycle_results = self._run_single_cycle(engineered_data, 1)
            
            # Generate report
            final_metrics = self._generate_final_report()
            
            return {
                'status': 'completed',
                'cycle_results': cycle_results,
                'final_metrics': final_metrics,
                'features_engineered': len(engineered_data.columns)
            }
            
        except Exception as e:
            logger.error(f"Single cycle analysis failed: {e}", exc_info=True)
            return {'status': 'failed', 'error': str(e)}

    def get_framework_status(self) -> Dict[str, Any]:
        """Get current framework status and statistics."""
        return {
            'cycles_completed': len(self.cycle_metrics),
            'successful_cycles': len([c for c in self.cycle_metrics if c.get('status') == 'completed']),
            'failed_cycles': len([c for c in self.cycle_metrics if c.get('status') == 'failed']),
            'features_available': len(self.aggregated_shap) if self.aggregated_shap is not None else 0,
            'framework_memory_size': len(self.framework_memory.get('historical_runs', [])),
            'last_cycle_performance': self.cycle_metrics[-1].get('metrics', {}) if self.cycle_metrics else {}
        }


def run_single_instance(fallback_config_dict: Dict, framework_history_loaded: Dict,
                       playbook_loaded: Dict, nickname_ledger_loaded: Dict,
                       directives_loaded: List[Dict], api_interval_seconds: int):
    """
    Executes a complete, end-to-end walk-forward backtest and analysis cycle.

    This is the main entry point for running a single instance of the AxiomEdge framework.
    It orchestrates all phases from data loading to final reporting.

    Args:
        fallback_config_dict: Base configuration dictionary
        framework_history_loaded: Historical framework performance data
        playbook_loaded: Strategy playbook with available strategies
        nickname_ledger_loaded: Mapping of version labels to human-readable names
        directives_loaded: List of strategic directives
        api_interval_seconds: API rate limiting interval
    """
    import os
    import re
    import gc
    import json
    import time
    import random
    import psutil
    from datetime import datetime
    from collections import defaultdict
    from .ai_analyzer import GeminiAnalyzer, APITimer
    from .config import ConfigModel, OperatingState, generate_dynamic_config, deep_merge_dicts, _generate_nickname, _log_config_and_environment, _adapt_drawdown_parameters, _update_operating_state, _apply_operating_state_rules
    from .data_handler import DataLoader, determine_timeframe_roles, get_and_cache_asset_types, get_macro_context_data, get_walk_forward_periods, apply_genetic_rules_to_df, train_and_diagnose_regime, _get_available_features_from_df
    from .feature_engineer import FeatureEngineer
    from .model_trainer import ModelTrainer
    from .backtester import Backtester
    from .report_generator import PerformanceAnalyzer
    from .telemetry import TelemetryCollector, InterventionManager
    from .genetic_programmer import GeneticProgrammer
    from .utils import _generate_cache_metadata, _generate_raw_data_summary_for_ai, _generate_inter_asset_correlation_summary_for_ai, _diagnose_raw_market_regime, _generate_raw_data_health_report, _validate_and_fix_spread_config, _generate_pre_analysis_summary, _create_label_distribution_report, _recursive_sanitize, _sanitize_ai_suggestions, save_run_to_memory, _setup_logging

    run_timestamp_str = datetime.now().strftime("%Y%m%d-%H%M%S")
    start_time = time.time()  # Track execution time
    gemini_analyzer = GeminiAnalyzer()
    api_timer = APITimer(interval_seconds=api_interval_seconds)

    # --- Phase 1: Initial Setup ---
    temp_minimal_config_dict = fallback_config_dict.copy()
    temp_minimal_config_dict.update({
        "REPORT_LABEL": f"AxiomEdge_V{fallback_config_dict.get('VERSION', '2.11')}_Setup",
        "run_timestamp": run_timestamp_str,
        "strategy_name": "InitialSetupPhase",
        "nickname": "init_setup",
        "selected_features": []
    })

    try:
        temp_config_for_paths_obj = ConfigModel(**temp_minimal_config_dict)
    except Exception as e:
        logger.critical(f"CRITICAL: Pydantic validation error for temp_minimal_config_dict: {e}")
        return {"status": "error", "message": "Initial temporary config validation failed"}

    prelim_log_dir = os.path.join(temp_config_for_paths_obj.BASE_PATH, "Results", "PrelimLogs")
    os.makedirs(prelim_log_dir, exist_ok=True)
    prelim_log_path = os.path.join(prelim_log_dir, f"pre_run_{run_timestamp_str}.log")
    _setup_logging(prelim_log_path, temp_config_for_paths_obj.REPORT_LABEL)

    # --- Phase 2: Load Raw Data ---
    data_loader = DataLoader(temp_config_for_paths_obj)
    all_files = [f for f in os.listdir(temp_config_for_paths_obj.BASE_PATH) if f.endswith(('.csv', '.txt')) and re.match(r'^[A-Z0-9]+_[A-Z0-9]+', f)]
    data_by_tf, detected_timeframes = data_loader.load_and_parse_data(all_files)
    if not data_by_tf:
        logger.critical("Data loading failed. Exiting.")
        return {"status": "error", "message": "Data loading failed"}

    tf_roles = determine_timeframe_roles(detected_timeframes)

    # --- Phase 3: AI-Driven Initial Configuration ---
    symbols = sorted(list(set([f.split('_')[0] for f in all_files])))
    asset_types = get_and_cache_asset_types(symbols, fallback_config_dict, gemini_analyzer)
    primary_class = 'Forex'
    if asset_types:
        from collections import Counter
        class_counts = Counter(asset_types.values())
        primary_class = class_counts.most_common(1)[0][0]

    config_with_asset_class = generate_dynamic_config(primary_class, fallback_config_dict)

    # --- Phase 4: Get Macro Context Data ---
    macro_data = get_macro_context_data(
        tickers=config_with_asset_class.get("MACRO_TICKERS", {}),
        config_dict=config_with_asset_class,
        gemini_analyzer=gemini_analyzer,
        api_timer=api_timer
    )

    # --- Phase 5: Generate Raw Data Summaries for AI ---
    data_summary = _generate_raw_data_summary_for_ai(data_by_tf, tf_roles)
    correlation_summary_for_ai = _generate_inter_asset_correlation_summary_for_ai(data_by_tf, tf_roles)
    diagnosed_regime = _diagnose_raw_market_regime(data_by_tf, tf_roles)
    health_report = _generate_raw_data_health_report(data_by_tf, tf_roles)

    # --- Phase 6: AI-Driven Configuration Optimization ---
    api_timer.wait_if_needed()
    initial_config = gemini_analyzer.get_initial_run_configuration(
        script_version=config_with_asset_class.get("VERSION", "2.11"),
        ledger=nickname_ledger_loaded,
        memory=framework_history_loaded,
        playbook=playbook_loaded,
        health_report=health_report,
        directives=directives_loaded,
        data_summary=data_summary,
        diagnosed_regime=diagnosed_regime,
        regime_champions={},
        correlation_summary_for_ai=correlation_summary_for_ai,
        master_macro_list=config_with_asset_class.get("MACRO_TICKERS", {}),
        prime_directive_str=config_with_asset_class.get("PRIME_DIRECTIVE", ""),
        num_features=len(_get_available_features_from_df(list(data_by_tf.values())[0])),
        num_samples=sum(len(df) for df in data_by_tf.values())
    )

    # --- Phase 7: Finalize Configuration ---
    final_config_dict = deep_merge_dicts(config_with_asset_class, {
        "strategy_name": initial_config.get("strategy_name", "default_strategy"),
        "selected_features": initial_config.get("selected_features", []),
        "nickname": _generate_nickname(
            initial_config.get("strategy_name", "default_strategy"),
            initial_config.get("reasoning", ""),
            nickname_ledger_loaded,
            os.path.join(config_with_asset_class.get("BASE_PATH", "."), "Results", "nickname_ledger.json"),
            config_with_asset_class.get("VERSION", "2.11")
        ),
        "run_timestamp": run_timestamp_str,
        "REPORT_LABEL": f"AxiomEdge_V{config_with_asset_class.get('VERSION', '2.11')}_{initial_config.get('strategy_name', 'default')}"
    })

    try:
        config = ConfigModel(**final_config_dict)
    except Exception as e:
        logger.critical(f"CRITICAL: Final config validation failed: {e}")
        return {"status": "error", "message": "Final config validation failed"}

    # --- Phase 8: Setup Logging and Environment ---
    _log_config_and_environment(config)

    # --- Phase 9: Feature Engineering ---
    logger.info("-> Stage 2: Global Feature Engineering...")
    feature_engineer = FeatureEngineer(config, tf_roles, playbook_loaded)

    # Determine base timeframe for feature engineering
    base_tf = tf_roles.get('primary', list(data_by_tf.keys())[0])
    base_df = data_by_tf[base_tf]

    df_full_engineered = feature_engineer.engineer_features(base_df, data_by_tf, macro_data)

    if df_full_engineered is None:
        logger.critical("Global feature engineering failed. Halting.")
        return {"status": "error", "message": "Global feature engineering failed."}

    # --- Phase 10: Walk-Forward Period Setup ---
    start_date, end_date = df_full_engineered.index.min(), df_full_engineered.index.max()

    train_start_dates, train_end_dates, test_start_dates, test_end_dates = get_walk_forward_periods(
        start_date, end_date, config.TRAINING_WINDOW, config.RETRAINING_FREQUENCY, config.FORWARD_TEST_GAP
    )

    if not train_start_dates:
        logger.critical("Cannot proceed: No valid walk-forward periods could be generated.")
        return {"status": "error", "message": "Walk-forward period generation failed."}

    # --- Phase 11: Initialize Telemetry and Intervention Systems ---
    from .telemetry import TelemetryCollector, InterventionManager

    telemetry_log_path = os.path.join(config.result_folder_path, "telemetry_log.jsonl")
    telemetry_collector = TelemetryCollector(telemetry_log_path)
    intervention_manager = InterventionManager(os.path.join(config.result_folder_path, "intervention_ledger.json"))

    consolidated_trades_path = os.path.join(config.result_folder_path, "temp_trades.csv")
    consolidated_equity_path = os.path.join(config.result_folder_path, "temp_equity.csv")
    for path in [consolidated_trades_path, consolidated_equity_path]:
        if os.path.exists(path):
            os.remove(path)

    # --- Phase 12: Initialize Walk-Forward Variables ---
    aggregated_daily_metrics_for_report, last_intervention_id = [], None
    cycle_directives, last_equity = {}, config.INITIAL_CAPITAL
    playbook_file_path = os.path.join(config.BASE_PATH, "Results", "strategy_playbook.json")
    strategy_failure_tracker = defaultdict(int)
    STRATEGY_QUARANTINE_THRESHOLD = 3
    quarantined_strategies = []

    # Initialize model trainer and backtester
    model_trainer = ModelTrainer(config)
    backtester = Backtester(config)

    # --- Phase 13: Main Walk-Forward Loop ---
    logger.info(f"-> Stage 3: Walk-Forward Analysis ({len(train_start_dates)} cycles)")

    for cycle_num, (train_start, train_end, test_start, test_end) in enumerate(zip(train_start_dates, train_end_dates, test_start_dates, test_end_dates)):
        cycle_label = f"Cycle {cycle_num + 1}/{len(train_start_dates)}"
        logger.info(f"\n--- Starting {cycle_label} | Train: {train_start.date()}->{train_end.date()} | Test: {test_start.date()}->{test_end.date()} ---")

        # Memory monitoring
        process = psutil.Process(os.getpid())
        mem_info = process.memory_info()
        logger.info(f"[{cycle_label}] Start of cycle memory usage: {mem_info.rss / 1024**2:.2f} MB")

        # Extract cycle data
        df_train_cycle_engineered = df_full_engineered.loc[train_start:train_end].copy()
        df_test_cycle_engineered = df_full_engineered.loc[test_start:test_end].copy()

        if df_train_cycle_engineered.empty or df_test_cycle_engineered.empty:
            logger.warning(f"[{cycle_label}] Skipping cycle due to empty data slices.")
            continue

        # Apply genetic rules if available
        genetic_rules = getattr(config, 'genetic_rules', None)
        if genetic_rules:
            df_train_cycle_engineered = apply_genetic_rules_to_df(df_train_cycle_engineered, genetic_rules, config)
            df_test_cycle_engineered = apply_genetic_rules_to_df(df_test_cycle_engineered, genetic_rules, config)

        # Model training phase
        logger.info(f"[{cycle_label}] Training models...")
        training_results = model_trainer.train_ensemble_models(
            df_train_cycle_engineered,
            config.selected_features,
            diagnosed_regime
        )

        if not training_results or not training_results.get('trained_pipelines'):
            logger.error(f"[{cycle_label}] Model training failed. Skipping cycle.")
            strategy_failure_tracker[config.strategy_name] += 1

            # Check for strategy quarantine
            if strategy_failure_tracker[config.strategy_name] >= STRATEGY_QUARANTINE_THRESHOLD:
                if config.strategy_name not in quarantined_strategies:
                    quarantined_strategies.append(config.strategy_name)
                    logger.warning(f"Strategy '{config.strategy_name}' quarantined after {STRATEGY_QUARANTINE_THRESHOLD} failures.")
            continue

        # Backtesting phase
        logger.info(f"[{cycle_label}] Running backtest...")
        cycle_history = telemetry_collector.get_last_n_cycles(5)

        trades_df, equity_curve, breaker_tripped, breaker_details, daily_metrics, rejected_signals = backtester.run_backtest_chunk(
            df_test_cycle_engineered,
            training_results,
            last_equity,
            cycle_history,
            diagnosed_regime,
            trade_lockout_until=None,
            cycle_directives=cycle_directives
        )

        # Update equity for next cycle
        if not equity_curve.empty:
            last_equity = equity_curve.iloc[-1]

        # Collect cycle metrics
        cycle_metrics = {
            'cycle_num': cycle_num + 1,
            'train_period': f"{train_start.date()} to {train_end.date()}",
            'test_period': f"{test_start.date()} to {test_end.date()}",
            'total_trades': len(trades_df) if not trades_df.empty else 0,
            'final_equity': last_equity,
            'breaker_tripped': breaker_tripped,
            'strategy_used': config.strategy_name,
            'regime': diagnosed_regime,
            'status': 'Completed'
        }

        # Add to telemetry
        telemetry_collector.log_cycle_completion(cycle_metrics)

        # Save consolidated results
        if not trades_df.empty:
            trades_df.to_csv(consolidated_trades_path, mode='a', header=not os.path.exists(consolidated_trades_path), index=False)

        if not equity_curve.empty:
            equity_df = pd.DataFrame({'equity': equity_curve})
            equity_df.to_csv(consolidated_equity_path, mode='a', header=not os.path.exists(consolidated_equity_path))

        # Memory cleanup
        del df_train_cycle_engineered, df_test_cycle_engineered
        gc.collect()

        logger.info(f"[{cycle_label}] Completed. Final equity: ${last_equity:,.2f}")

    # End of walk-forward loop

    # --- Phase 14: Final Analysis and Reporting ---
    logger.info("-> Stage 4: Final Analysis and Reporting...")

    # Load consolidated results
    final_trades_df = pd.DataFrame()
    final_equity_curve = pd.Series(dtype=float)

    if os.path.exists(consolidated_trades_path):
        final_trades_df = pd.read_csv(consolidated_trades_path)

    if os.path.exists(consolidated_equity_path):
        equity_data = pd.read_csv(consolidated_equity_path, index_col=0)
        final_equity_curve = equity_data['equity'] if 'equity' in equity_data.columns else pd.Series(dtype=float)

    # Generate comprehensive performance report
    from .backtester import PerformanceAnalyzer
    analyzer = PerformanceAnalyzer(config)

    cycle_metrics_list = telemetry_collector.get_historical_telemetry()
    aggregated_shap = getattr(model_trainer, 'aggregated_shap_summary', None)

    final_metrics = analyzer.generate_full_report(
        final_trades_df,
        final_equity_curve,
        cycle_metrics_list,
        aggregated_shap,
        framework_history_loaded,
        aggregated_daily_metrics_for_report,
        getattr(model_trainer, 'last_classification_report', "N/A")
    )

    # --- Phase 15: Framework Memory Update ---
    final_run_summary = {
        "run_id": config.run_timestamp,
        "nickname": config.nickname,
        "strategy_name": config.strategy_name,
        "final_metrics": final_metrics,
        "total_cycles": len(train_start_dates),
        "successful_cycles": len([c for c in cycle_metrics_list if c.get('status') == 'Completed']),
        "quarantined_strategies": quarantined_strategies,
        "regime_diagnosed": diagnosed_regime,
        "execution_time": time.time() - start_time,
        "cycle_details": telemetry_collector.get_historical_telemetry(),
        "config_summary": config.model_dump()
    }

    # Update regime diagnosis and save to memory
    final_diagnosed_regime = train_and_diagnose_regime(data_by_tf[base_tf], config.result_folder_path).get("current_diagnosed_regime", "N/A")
    primary_shap_summary = getattr(model_trainer, 'shap_summaries', {}).get(f'primary_model_h{config.LABEL_HORIZONS[0]}')
    save_run_to_memory(config, final_run_summary, framework_history_loaded, final_diagnosed_regime, primary_shap_summary)

    # --- Phase 16: Cleanup ---
    for path in [consolidated_trades_path, consolidated_equity_path]:
        if os.path.exists(path):
            os.remove(path)

    # --- Phase 17: Final Logging ---
    logger.info("="*80)
    logger.info("AXIOM EDGE FRAMEWORK RUN COMPLETED SUCCESSFULLY")
    logger.info("="*80)
    logger.info(f"Run {config.REPORT_LABEL} - {config.nickname} completed. Report: {config.REPORT_SAVE_PATH}")

    return {
        "status": "success",
        "metrics": final_metrics,
        "report_path": config.REPORT_SAVE_PATH,
        "final_equity": last_equity,
        "total_cycles": len(train_start_dates),
        "quarantined_strategies": quarantined_strategies,
        "regime": final_diagnosed_regime
    }
