# =============================================================================
# FRAMEWORK ORCHESTRATOR MODULE
# =============================================================================

import os
import logging
import time
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple
import pandas as pd
import numpy as np

from .config import ConfigModel
from .data_handler import DataHandler
from .feature_engineer import FeatureEngineer
from .model_trainer import ModelTrainer
from .genetic_programmer import GeneticProgrammer
from .report_generator import ReportGenerator
from .ai_analyzer import GeminiAnalyzer, APITimer
from .utils import get_optimal_system_settings

logger = logging.getLogger(__name__)

class FrameworkOrchestrator:
    """
    Orchestrates the complete AxiomEdge framework workflow including
    walk-forward analysis, model training, backtesting, and reporting.
    """
    
    def __init__(self, config: ConfigModel):
        """Initialize the Framework Orchestrator with configuration."""
        self.config = config
        
        # Initialize core components
        self.data_handler = DataHandler()
        self.gemini_analyzer = GeminiAnalyzer()
        self.api_timer = APITimer()
        
        # Initialize framework components
        timeframe_roles = {'base': 'D1'}  # Default timeframe role
        playbook = {}  # Default empty playbook
        
        self.feature_engineer = FeatureEngineer(config, timeframe_roles, playbook)
        self.model_trainer = ModelTrainer(config, self.gemini_analyzer)
        self.report_generator = ReportGenerator(config)
        
        # Framework state
        self.framework_memory = {
            'historical_runs': [],
            'best_performance': {},
            'intervention_history': {},
            'cycle_telemetry': []
        }
        
        # Performance tracking
        self.cycle_metrics = []
        self.aggregated_shap = None
        self.trades_df = None
        self.equity_curve = None
        
        logger.info("FrameworkOrchestrator initialized")

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
            # Stage 1: Data Collection and Preparation
            logger.info("📊 Stage 1: Data Collection and Preparation")
            data_dict = self._collect_and_prepare_data(data_files, symbols, start_date, end_date)
            
            if not data_dict:
                return {"error": "No data collected", "status": "failed"}
            
            # Stage 2: Feature Engineering
            logger.info("🔧 Stage 2: Feature Engineering")
            engineered_data = self._engineer_features(data_dict)
            
            if engineered_data.empty:
                return {"error": "Feature engineering failed", "status": "failed"}
            
            # Stage 3: Walk-Forward Analysis
            logger.info("⚙️  Stage 3: Walk-Forward Analysis")
            walk_forward_results = self._run_walk_forward_analysis(engineered_data)
            
            # Stage 4: Strategy Evolution (if enabled)
            if getattr(self.config, 'ENABLE_GENETIC_PROGRAMMING', False):
                logger.info("🧬 Stage 4: Strategy Evolution")
                evolution_results = self._run_strategy_evolution(engineered_data)
                walk_forward_results['evolution'] = evolution_results
            
            # Stage 5: Final Reporting
            logger.info("📋 Stage 5: Final Reporting")
            final_metrics = self._generate_final_report()
            
            # Calculate execution time
            execution_time = time.time() - start_time
            
            # Compile final results
            results = {
                'status': 'completed',
                'execution_time_seconds': execution_time,
                'data_files_processed': len(data_files),
                'symbols_analyzed': list(data_dict.keys()) if data_dict else [],
                'features_engineered': len(engineered_data.columns) if not engineered_data.empty else 0,
                'walk_forward_cycles': len(self.cycle_metrics),
                'final_metrics': final_metrics,
                'cycle_breakdown': self.cycle_metrics,
                'framework_memory': self.framework_memory
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
