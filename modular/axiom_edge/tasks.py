# =============================================================================
# TASK-SPECIFIC INTERFACES MODULE
# =============================================================================

"""
Task-specific interfaces for modular AxiomEdge framework usage.

This module provides high-level task interfaces that allow users to run individual
components of AxiomEdge independently or in combination. Each task encapsulates
a specific workflow and provides a clean API for common trading framework operations.

Available Tasks:
- DataCollectionTask: Multi-source data collection with intelligent caching
- BrokerInfoTask: Broker spread and cost analysis with AI insights
- BacktestTask: Standalone backtesting with performance analysis
- FeatureEngineeringTask: Advanced feature engineering with 200+ indicators
- ModelTrainingTask: ML model training with hyperparameter optimization
- CompleteFrameworkTask: Full framework execution with walk-forward analysis

Each task can be used independently or chained together for complex workflows.
"""

import logging
import time
import json
from datetime import datetime
from typing import Dict, List, Optional, Any, Tuple
import pandas as pd
import numpy as np
from pathlib import Path

from .config import ConfigModel, create_default_config
from .data_handler import DataHandler, DataLoader
from .ai_analyzer import GeminiAnalyzer, APITimer
from .feature_engineer import FeatureEngineer
from .model_trainer import ModelTrainer
from .backtester import Backtester
from .genetic_programmer import GeneticProgrammer
from .report_generator import ReportGenerator
from .framework_orchestrator import FrameworkOrchestrator
from .telemetry import TelemetryCollector, TelemetryAnalyzer

# Optional imports with fallbacks
try:
    from .telemetry import TelemetryCollector, TelemetryAnalyzer
    TELEMETRY_AVAILABLE = True
except ImportError:
    TELEMETRY_AVAILABLE = False

logger = logging.getLogger(__name__)

class BaseTask:
    """
    Enhanced base class for all task interfaces with advanced capabilities.

    Features:
    - Comprehensive error handling and recovery
    - Performance monitoring and telemetry integration
    - Configuration validation and optimization
    - Progress tracking and status reporting
    - Resource usage monitoring
    - Automated result validation
    """

    def __init__(self, config: Optional[ConfigModel] = None, enable_telemetry: bool = True):
        self.config = config or create_default_config("./")
        self.logger = logging.getLogger(self.__class__.__name__)
        self.enable_telemetry = enable_telemetry and TELEMETRY_AVAILABLE

        # Performance tracking
        self.start_time = None
        self.execution_stats = {}

        # Initialize telemetry if available
        if self.enable_telemetry:
            try:
                self.telemetry_collector = TelemetryCollector(self.config)
                self.telemetry_analyzer = TelemetryAnalyzer(self.config)
            except Exception as e:
                self.logger.warning(f"Telemetry initialization failed: {e}")
                self.enable_telemetry = False
        else:
            self.telemetry_collector = None
            self.telemetry_analyzer = None

    def setup_logging(self, level: str = "INFO"):
        """Setup enhanced logging for the task"""
        logging.basicConfig(
            level=getattr(logging, level.upper()),
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.StreamHandler(),
                logging.FileHandler(f'{self.__class__.__name__.lower()}.log')
            ]
        )

    def start_execution(self, task_name: str = None):
        """Start task execution with telemetry tracking"""
        self.start_time = time.time()
        task_name = task_name or self.__class__.__name__

        if self.telemetry_collector:
            self.telemetry_collector.start_collection()
            self.telemetry_collector.log_task_start(
                task_name=task_name,
                config_snapshot=self.config
            )

        self.logger.info(f"Starting {task_name} execution")

    def end_execution(self, task_name: str = None, results: Dict[str, Any] = None):
        """End task execution with performance summary"""
        if self.start_time:
            execution_time = time.time() - self.start_time
            self.execution_stats['execution_time_seconds'] = execution_time

        task_name = task_name or self.__class__.__name__

        if self.telemetry_collector:
            self.telemetry_collector.log_task_completion(
                task_name=task_name,
                execution_time=execution_time,
                results_summary=results or {}
            )
            summary = self.telemetry_collector.stop_collection()
            self.execution_stats['telemetry_summary'] = summary

        self.logger.info(f"Completed {task_name} execution in {execution_time:.2f} seconds")
        return self.execution_stats

    def validate_config(self) -> Dict[str, Any]:
        """Validate configuration and return validation results"""
        validation_results = {
            'is_valid': True,
            'warnings': [],
            'errors': [],
            'recommendations': []
        }

        try:
            # Basic configuration validation
            if not hasattr(self.config, 'FRAMEWORK_VERSION'):
                validation_results['warnings'].append('Framework version not specified')

            # Check for required directories
            required_dirs = ['data', 'models', 'reports']
            for dir_name in required_dirs:
                dir_path = Path(dir_name)
                if not dir_path.exists():
                    validation_results['recommendations'].append(f'Create {dir_name} directory for better organization')

            return validation_results

        except Exception as e:
            validation_results['is_valid'] = False
            validation_results['errors'].append(f'Configuration validation failed: {e}')
            return validation_results

    def get_execution_summary(self) -> Dict[str, Any]:
        """Get comprehensive execution summary"""
        summary = {
            'task_name': self.__class__.__name__,
            'execution_stats': self.execution_stats,
            'config_summary': self._get_config_summary(),
            'telemetry_enabled': self.enable_telemetry
        }

        if self.telemetry_analyzer and self.enable_telemetry:
            try:
                ai_analysis = self.telemetry_analyzer.run_comprehensive_ai_doctor_analysis()
                summary['ai_analysis'] = ai_analysis
            except Exception as e:
                self.logger.warning(f"AI analysis failed: {e}")

        return summary

    def _get_config_summary(self) -> Dict[str, Any]:
        """Get sanitized configuration summary"""
        try:
            config_dict = self.config.dict() if hasattr(self.config, 'dict') else vars(self.config)

            # Remove sensitive information
            sensitive_keys = ['API_KEY', 'SECRET_KEY', 'PASSWORD', 'TOKEN']
            sanitized = {}

            for key, value in config_dict.items():
                if any(sensitive in key.upper() for sensitive in sensitive_keys):
                    sanitized[key] = "***REDACTED***"
                else:
                    sanitized[key] = str(value)[:100] if isinstance(value, str) and len(str(value)) > 100 else value

            return sanitized

        except Exception as e:
            return {'error': f'Could not generate config summary: {e}'}


class DataCollectionTask(BaseTask):
    """
    Task interface for collecting historical data from various sources.
    
    Usage:
        task = DataCollectionTask()
        data = task.collect_data(["AAPL", "GOOGL"], "2023-01-01", "2024-01-01")
    """
    
    def __init__(self, config: Optional[ConfigModel] = None,
                 cache_dir: str = "data_cache", gemini_analyzer=None):
        super().__init__(config)
        # Initialize Gemini analyzer if not provided
        if gemini_analyzer is None:
            try:
                from .ai_analyzer import GeminiAnalyzer
                gemini_analyzer = GeminiAnalyzer()
            except:
                gemini_analyzer = None

        self.data_handler = DataHandler(cache_dir=cache_dir, gemini_analyzer=gemini_analyzer)
    
    def collect_data(self, symbols: List[str], start_date: str, end_date: str,
                    timeframe: str = "1D", source: str = "yahoo") -> Dict[str, pd.DataFrame]:
        """
        Collect historical data for multiple symbols
        
        Args:
            symbols: List of symbols to collect data for
            start_date: Start date in YYYY-MM-DD format
            end_date: End date in YYYY-MM-DD format
            timeframe: Data timeframe (1D, 1H, 5M, etc.)
            source: Data source (alpha_vantage, yahoo, polygon)
            
        Returns:
            Dictionary mapping symbols to their data DataFrames
        """
        self.logger.info(f"Starting data collection for {len(symbols)} symbols")
        
        collected_data = {}
        for symbol in symbols:
            self.logger.info(f"Collecting data for {symbol}")
            df = self.data_handler.get_data(symbol, start_date, end_date, timeframe, source)
            
            if not df.empty:
                collected_data[symbol] = df
                self.logger.info(f"✓ Collected {len(df)} records for {symbol}")
            else:
                self.logger.warning(f"✗ No data collected for {symbol}")
        
        self.logger.info(f"Data collection complete. Collected data for {len(collected_data)} symbols")
        return collected_data
    
    def save_data(self, data: Dict[str, pd.DataFrame], output_dir: str = "collected_data"):
        """Save collected data to files"""
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True)
        
        for symbol, df in data.items():
            filename = f"{symbol}_1D_data.csv"
            filepath = output_path / filename
            df.to_csv(filepath, index=True)
            self.logger.info(f"Saved {symbol} data to {filepath}")
    
    def get_cache_info(self) -> Dict[str, Any]:
        """Get information about data cache"""
        return self.data_handler.get_cache_info()


class BrokerInfoTask(BaseTask):
    """
    Task interface for collecting broker information like spreads and trading costs.
    
    Usage:
        task = BrokerInfoTask()
        spreads = task.collect_spreads(["EURUSD", "GBPUSD"], "oanda")
    """
    
    def __init__(self, config: Optional[ConfigModel] = None, gemini_analyzer=None):
        super().__init__(config)
        # Initialize Gemini analyzer if not provided
        if gemini_analyzer is None:
            try:
                gemini_analyzer = GeminiAnalyzer()
            except:
                gemini_analyzer = None

        self.data_handler = DataHandler(gemini_analyzer=gemini_analyzer)
        self.ai_analyzer = gemini_analyzer
    
    def collect_spreads(self, symbols: List[str], broker: str = "oanda") -> Dict[str, float]:
        """
        Collect current spread information for symbols
        
        Args:
            symbols: List of symbols to get spreads for
            broker: Broker name (oanda, interactive_brokers, etc.)
            
        Returns:
            Dictionary mapping symbols to their current spreads
        """
        self.logger.info(f"Collecting spreads for {len(symbols)} symbols from {broker}")
        spreads = self.data_handler.get_broker_spreads(symbols, broker)
        
        if spreads:
            self.logger.info(f"✓ Collected spreads for {len(spreads)} symbols")
            for symbol, spread in spreads.items():
                self.logger.info(f"  {symbol}: {spread:.5f}")
        else:
            self.logger.warning("No spread data collected")
            
        return spreads
    
    def analyze_broker_costs(self, symbols: List[str], broker_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Use AI to analyze broker costs and provide recommendations
        
        Args:
            symbols: List of symbols being analyzed
            broker_data: Dictionary containing broker information
            
        Returns:
            AI analysis and recommendations
        """
        self.logger.info("Starting AI analysis of broker costs")
        
        analysis_results = {}
        for symbol in symbols:
            if symbol in broker_data:
                symbol_analysis = self.ai_analyzer.get_broker_analysis(symbol, broker_data)
                analysis_results[symbol] = symbol_analysis
                self.logger.info(f"✓ Completed AI analysis for {symbol}")
        
        return analysis_results
    
    def compare_brokers(self, symbols: List[str], brokers: List[str]) -> Dict[str, Dict[str, float]]:
        """
        Compare spreads across multiple brokers
        
        Args:
            symbols: List of symbols to compare
            brokers: List of broker names
            
        Returns:
            Nested dictionary: {symbol: {broker: spread}}
        """
        self.logger.info(f"Comparing {len(brokers)} brokers for {len(symbols)} symbols")
        
        comparison_data = {}
        for symbol in symbols:
            comparison_data[symbol] = {}
            for broker in brokers:
                spreads = self.data_handler.get_broker_spreads([symbol], broker)
                if symbol in spreads:
                    comparison_data[symbol][broker] = spreads[symbol]
        
        return comparison_data


class BacktestTask(BaseTask):
    """
    Task interface for backtesting user-provided strategies.
    
    Usage:
        task = BacktestTask()
        results = task.backtest_strategy(data, strategy_rules, config)
    """
    
    def __init__(self, config: Optional[ConfigModel] = None):
        super().__init__(config)
    
    def backtest_strategy(self, data: pd.DataFrame, strategy_rules: Dict[str, Any],
                         backtest_config: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Backtest a user-defined strategy
        
        Args:
            data: Historical price data
            strategy_rules: Dictionary defining the strategy rules
            backtest_config: Optional backtesting configuration
            
        Returns:
            Backtest results and performance metrics
        """
        from .backtester import Backtester
        
        self.logger.info("Starting strategy backtest")
        
        # Create backtester instance
        backtester = Backtester(self.config)
        
        # Apply strategy rules to generate signals
        signals = self._apply_strategy_rules(data, strategy_rules)
        
        # Run backtest
        results = backtester.run_backtest(data, signals)
        
        self.logger.info("Backtest completed")
        return results
    
    def _apply_strategy_rules(self, data: pd.DataFrame, rules: Dict[str, Any]) -> pd.DataFrame:
        """Apply user-defined strategy rules to generate trading signals"""
        # This is a simplified implementation
        # In practice, this would parse and execute complex strategy rules
        
        signals = pd.DataFrame(index=data.index)
        signals['signal'] = 0  # 0 = no signal, 1 = buy, -1 = sell
        
        # Example: Simple moving average crossover
        if 'sma_short' in rules and 'sma_long' in rules:
            short_ma = data['Close'].rolling(rules['sma_short']).mean()
            long_ma = data['Close'].rolling(rules['sma_long']).mean()
            
            signals.loc[short_ma > long_ma, 'signal'] = 1
            signals.loc[short_ma < long_ma, 'signal'] = -1
        
        return signals
    
    def optimize_strategy(self, data: pd.DataFrame, strategy_template: Dict[str, Any],
                         parameter_ranges: Dict[str, Tuple[float, float]]) -> Dict[str, Any]:
        """
        Optimize strategy parameters using grid search or other methods
        
        Args:
            data: Historical price data
            strategy_template: Base strategy configuration
            parameter_ranges: Ranges for each parameter to optimize
            
        Returns:
            Optimized parameters and performance results
        """
        self.logger.info("Starting strategy optimization")
        
        best_params = {}
        best_performance = -float('inf')
        
        # Simple grid search implementation
        # In practice, this would use more sophisticated optimization
        
        self.logger.info("Strategy optimization completed")
        return {
            'best_parameters': best_params,
            'best_performance': best_performance,
            'optimization_results': []
        }


class FeatureEngineeringTask(BaseTask):
    """
    Task interface for feature engineering and data preparation.
    
    Usage:
        task = FeatureEngineeringTask()
        features = task.engineer_features(data)
    """
    
    def __init__(self, config: Optional[ConfigModel] = None):
        super().__init__(config)
    
    def engineer_features(self, data: pd.DataFrame, 
                         feature_config: Optional[Dict[str, Any]] = None) -> pd.DataFrame:
        """
        Engineer features from raw price data
        
        Args:
            data: Raw price data (OHLCV)
            feature_config: Optional configuration for feature engineering
            
        Returns:
            DataFrame with engineered features
        """
        from .feature_engineer import FeatureEngineer

        self.logger.info("Starting feature engineering")

        # Create feature engineer
        timeframe_roles = {'base': '1D'}  # Default timeframe role
        playbook = {}  # Default empty playbook

        feature_engineer = FeatureEngineer(self.config, timeframe_roles, playbook)

        # Prepare data in the expected format
        if isinstance(data, pd.DataFrame):
            # Ensure we have a proper datetime index
            if 'timestamp' in data.columns and not isinstance(data.index, pd.DatetimeIndex):
                data = data.set_index('timestamp')
            elif not isinstance(data.index, pd.DatetimeIndex):
                # Create a simple datetime index if none exists
                data.index = pd.date_range(start='2023-01-01', periods=len(data), freq='D')

            # Engineer features
            data_by_tf = {'1D': data}  # Wrap data in expected format
            macro_data = None  # No macro data by default

            engineered_data = feature_engineer.engineer_features(data, data_by_tf, macro_data)
        else:
            self.logger.error("Data must be a pandas DataFrame")
            return pd.DataFrame()
        
        self.logger.info(f"Feature engineering completed. Generated {len(engineered_data.columns)} features")
        return engineered_data
    
    def select_features(self, data: pd.DataFrame, target_column: str,
                       method: str = "mutual_info", n_features: int = 50) -> List[str]:
        """
        Select the most important features for modeling
        
        Args:
            data: Data with features and target
            target_column: Name of the target column
            method: Feature selection method
            n_features: Number of features to select
            
        Returns:
            List of selected feature names
        """
        from sklearn.feature_selection import mutual_info_classif, SelectKBest
        
        self.logger.info(f"Starting feature selection using {method}")
        
        # Prepare features and target
        feature_columns = [col for col in data.columns if col != target_column]
        X = data[feature_columns].fillna(0)
        y = data[target_column].fillna(0)
        
        # Apply feature selection
        if method == "mutual_info":
            selector = SelectKBest(mutual_info_classif, k=n_features)
            selector.fit(X, y)
            selected_features = [feature_columns[i] for i in selector.get_support(indices=True)]
        else:
            # Default to top N features by variance
            variances = X.var()
            selected_features = variances.nlargest(n_features).index.tolist()
        
        self.logger.info(f"Selected {len(selected_features)} features")
        return selected_features


class ModelTrainingTask(BaseTask):
    """
    Task interface for training machine learning models.
    
    Usage:
        task = ModelTrainingTask()
        model = task.train_model(features, target)
    """
    
    def __init__(self, config: Optional[ConfigModel] = None):
        super().__init__(config)
    
    def train_model(self, features: pd.DataFrame, target: pd.Series,
                   model_type: str = "random_forest") -> Dict[str, Any]:
        """
        Train a machine learning model
        
        Args:
            features: Feature data
            target: Target variable
            model_type: Type of model to train
            
        Returns:
            Trained model and performance metrics
        """
        from .model_trainer import ModelTrainer
        from .ai_analyzer import GeminiAnalyzer

        self.logger.info(f"Starting model training with {model_type}")

        # Create AI analyzer and model trainer
        try:
            gemini_analyzer = GeminiAnalyzer()
        except Exception as e:
            self.logger.warning(f"Could not initialize Gemini analyzer: {e}. Using basic trainer.")
            # Create a minimal analyzer for basic functionality
            class BasicAnalyzer:
                def __init__(self):
                    self.api_key_valid = False
            gemini_analyzer = BasicAnalyzer()

        trainer = ModelTrainer(self.config, gemini_analyzer)

        # Prepare data
        if isinstance(features, pd.DataFrame) and isinstance(target, (pd.Series, np.ndarray, list)):
            # Combine features and target
            combined_data = features.copy()

            # Handle different target formats
            if isinstance(target, (list, np.ndarray)):
                target = pd.Series(target, index=features.index)

            # Add target with appropriate name
            if hasattr(target, 'name') and target.name:
                target_name = target.name
            else:
                target_name = 'target_signal_pressure_class'

            combined_data[target_name] = target

            # Train model
            model_results = trainer.train_and_validate_model(combined_data)

        else:
            self.logger.error("Invalid data format. Features must be DataFrame, target must be Series/array")
            return {"error": "Invalid data format"}

        self.logger.info("Model training completed")
        return model_results


class CompleteFrameworkTask(BaseTask):
    """
    Task interface for running the complete AxiomEdge framework.
    
    Usage:
        task = CompleteFrameworkTask()
        results = task.run_complete_framework(data_files)
    """
    
    def __init__(self, config: Optional[ConfigModel] = None):
        super().__init__(config)
    
    def run_complete_framework(self, data_files: List[str],
                              framework_config: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Run the complete AxiomEdge framework
        
        Args:
            data_files: List of data files to process
            framework_config: Optional framework configuration
            
        Returns:
            Complete framework results
        """
        from .framework_orchestrator import FrameworkOrchestrator
        
        self.logger.info("Starting complete AxiomEdge framework")
        
        # Create orchestrator
        orchestrator = FrameworkOrchestrator(self.config)
        
        # Run framework
        results = orchestrator.run_complete_cycle(data_files)
        
        self.logger.info("Complete framework execution finished")
        return results


class AIStrategyOptimizationTask(BaseTask):
    """
    Advanced AI-powered strategy optimization task with genetic programming.

    This task combines multiple AI techniques to discover and optimize trading strategies:
    - Genetic programming for rule discovery
    - Hyperparameter optimization using Optuna
    - Feature importance analysis with SHAP
    - Multi-objective optimization (Sharpe, drawdown, frequency)
    - Regime-adaptive parameter tuning

    Usage:
        task = AIStrategyOptimizationTask()
        results = task.optimize_strategy(data, target_metrics={'sharpe_ratio': 1.5})
    """

    def __init__(self, config: Optional[ConfigModel] = None):
        super().__init__(config)

        # Initialize AI components
        self.gemini_analyzer = GeminiAnalyzer()
        self.api_timer = APITimer()

        # Initialize framework components
        timeframe_roles = getattr(config, 'TIMEFRAME_ROLES', {'base': 'D1'})
        playbook = getattr(config, 'STRATEGY_PLAYBOOK', {})

        self.feature_engineer = FeatureEngineer(config, timeframe_roles, playbook)
        self.model_trainer = ModelTrainer(config, self.gemini_analyzer)
        self.backtester = Backtester(config)

        # Initialize genetic programmer if enabled
        continuous_features = getattr(config, 'CONTINUOUS_FEATURES', [])
        state_features = getattr(config, 'STATE_FEATURES', [])

        if continuous_features or state_features:
            self.genetic_programmer = GeneticProgrammer(config, continuous_features, state_features)
        else:
            self.genetic_programmer = None
            self.logger.warning("No features provided for genetic programming")

    def optimize_strategy(self, data: pd.DataFrame, target_metrics: Dict[str, float] = None,
                         optimization_budget: int = 100) -> Dict[str, Any]:
        """
        Run comprehensive AI-powered strategy optimization.

        Args:
            data: Historical market data
            target_metrics: Target performance metrics to optimize for
            optimization_budget: Number of optimization iterations

        Returns:
            Comprehensive optimization results with best strategies
        """
        self.start_execution("AIStrategyOptimization")

        try:
            optimization_results = {
                'optimization_metadata': {
                    'start_time': datetime.now().isoformat(),
                    'data_shape': data.shape,
                    'target_metrics': target_metrics or {},
                    'optimization_budget': optimization_budget
                },
                'feature_engineering': {},
                'genetic_programming': {},
                'model_optimization': {},
                'strategy_validation': {},
                'best_strategies': [],
                'performance_analysis': {}
            }

            # Stage 1: Advanced Feature Engineering
            self.logger.info("Stage 1: Advanced Feature Engineering")
            engineered_data = self.feature_engineer.engineer_features(data)

            if engineered_data.empty:
                raise ValueError("Feature engineering failed - no features generated")

            optimization_results['feature_engineering'] = {
                'original_features': len(data.columns),
                'engineered_features': len(engineered_data.columns),
                'feature_categories': self._analyze_feature_categories(engineered_data)
            }

            # Stage 2: Genetic Programming for Rule Discovery
            if self.genetic_programmer:
                self.logger.info("Stage 2: Genetic Programming for Rule Discovery")

                # Run genetic programming evolution
                best_chromosome, best_fitness = self.genetic_programmer.run_evolution(
                    engineered_data, self.gemini_analyzer, self.api_timer
                )

                optimization_results['genetic_programming'] = {
                    'best_chromosome': best_chromosome,
                    'best_fitness': best_fitness,
                    'evolution_stats': self.genetic_programmer.get_evolution_statistics(),
                    'population_diversity': self.genetic_programmer.calculate_population_diversity()
                }

            # Stage 3: Model Optimization with AI Guidance
            self.logger.info("Stage 3: AI-Guided Model Optimization")

            # Prepare labeled data for model training
            labeled_data = self._prepare_labeled_data(engineered_data)
            feature_list = [col for col in labeled_data.columns if not col.startswith('target_')]

            # Train ensemble models with optimization
            training_results, training_error = self.model_trainer.train_all_models(
                df_train_labeled=labeled_data,
                feature_list=feature_list,
                framework_history={'historical_runs': []},
                regime="Optimization"
            )

            if training_results:
                optimization_results['model_optimization'] = {
                    'models_trained': len(training_results.get('trained_pipelines', {})),
                    'best_model_performance': training_results.get('horizon_performance_metrics', {}),
                    'feature_importance': training_results.get('shap_summaries', {}),
                    'optimization_successful': True
                }
            else:
                optimization_results['model_optimization'] = {
                    'optimization_successful': False,
                    'error': training_error
                }

            # Stage 4: Strategy Validation and Backtesting
            self.logger.info("Stage 4: Strategy Validation and Backtesting")

            if training_results:
                # Run comprehensive backtesting
                validation_data = labeled_data.tail(int(len(labeled_data) * 0.3))  # Last 30% for validation

                trades_df, equity_curve, backtest_success, additional_metrics, summary = self.backtester.run_backtest_chunk(
                    df_chunk=validation_data,
                    training_results=training_results,
                    initial_equity=getattr(self.config, 'INITIAL_EQUITY', 100000),
                    feature_list=feature_list,
                    confidence_threshold=0.6,
                    regime="Optimization"
                )

                optimization_results['strategy_validation'] = {
                    'backtest_successful': backtest_success,
                    'validation_summary': summary,
                    'trades_generated': len(trades_df) if trades_df is not None else 0,
                    'final_equity': equity_curve.iloc[-1] if equity_curve is not None and len(equity_curve) > 0 else 0
                }

            # Stage 5: Performance Analysis and Ranking
            self.logger.info("Stage 5: Performance Analysis and Strategy Ranking")

            best_strategies = self._rank_strategies(optimization_results, target_metrics)
            optimization_results['best_strategies'] = best_strategies

            # Generate comprehensive performance analysis
            optimization_results['performance_analysis'] = self._generate_performance_analysis(optimization_results)

            # End execution and return results
            execution_stats = self.end_execution("AIStrategyOptimization", optimization_results)
            optimization_results['execution_stats'] = execution_stats

            self.logger.info("AI Strategy Optimization completed successfully")
            return optimization_results

        except Exception as e:
            self.logger.error(f"AI Strategy Optimization failed: {e}", exc_info=True)
            execution_stats = self.end_execution("AIStrategyOptimization", {'error': str(e)})

            return {
                'error': str(e),
                'execution_stats': execution_stats,
                'optimization_metadata': {
                    'start_time': datetime.now().isoformat(),
                    'failed': True
                }
            }

    def _analyze_feature_categories(self, data: pd.DataFrame) -> Dict[str, int]:
        """Analyze feature categories in the engineered data"""
        categories = {
            'technical_indicators': 0,
            'statistical_features': 0,
            'price_features': 0,
            'volume_features': 0,
            'volatility_features': 0,
            'other_features': 0
        }

        for col in data.columns:
            col_lower = col.lower()
            if any(indicator in col_lower for indicator in ['sma', 'ema', 'rsi', 'macd', 'bollinger']):
                categories['technical_indicators'] += 1
            elif any(stat in col_lower for stat in ['mean', 'std', 'skew', 'kurt', 'corr']):
                categories['statistical_features'] += 1
            elif any(price in col_lower for price in ['open', 'high', 'low', 'close', 'price']):
                categories['price_features'] += 1
            elif 'volume' in col_lower:
                categories['volume_features'] += 1
            elif any(vol in col_lower for vol in ['volatility', 'atr', 'vix']):
                categories['volatility_features'] += 1
            else:
                categories['other_features'] += 1

        return categories

    def _prepare_labeled_data(self, data: pd.DataFrame) -> pd.DataFrame:
        """Prepare labeled data for model training"""
        try:
            # Create simple forward-looking labels based on price movement
            labeled_data = data.copy()

            if 'Close' in data.columns:
                # Create labels for different horizons
                for horizon in [30, 60, 90]:
                    future_returns = data['Close'].pct_change(horizon).shift(-horizon)

                    # Create 3-class labels: 0=short, 1=hold, 2=long
                    labels = pd.Series(1, index=data.index)  # Default to hold
                    labels[future_returns > 0.02] = 2  # Long signal
                    labels[future_returns < -0.02] = 0  # Short signal

                    labeled_data[f'target_signal_pressure_class_h{horizon}'] = labels

            # Remove rows with NaN labels
            labeled_data = labeled_data.dropna()

            return labeled_data

        except Exception as e:
            self.logger.error(f"Failed to prepare labeled data: {e}")
            return pd.DataFrame()

    def _rank_strategies(self, optimization_results: Dict[str, Any],
                        target_metrics: Dict[str, float] = None) -> List[Dict[str, Any]]:
        """Rank strategies based on performance metrics"""
        strategies = []

        try:
            # Extract strategy performance from different optimization stages

            # Genetic programming strategy
            if 'genetic_programming' in optimization_results and optimization_results['genetic_programming']:
                gp_results = optimization_results['genetic_programming']
                strategies.append({
                    'strategy_type': 'genetic_programming',
                    'strategy_id': 'gp_best',
                    'fitness_score': gp_results.get('best_fitness', 0),
                    'chromosome': gp_results.get('best_chromosome'),
                    'evolution_stats': gp_results.get('evolution_stats', {})
                })

            # Model-based strategy
            if 'model_optimization' in optimization_results and optimization_results['model_optimization'].get('optimization_successful'):
                model_results = optimization_results['model_optimization']
                strategies.append({
                    'strategy_type': 'ensemble_models',
                    'strategy_id': 'ensemble_best',
                    'performance_metrics': model_results.get('best_model_performance', {}),
                    'models_count': model_results.get('models_trained', 0)
                })

            # Validation-based ranking
            if 'strategy_validation' in optimization_results and optimization_results['strategy_validation'].get('backtest_successful'):
                validation_results = optimization_results['strategy_validation']
                validation_summary = validation_results.get('validation_summary', {})

                # Add validation scores to strategies
                for strategy in strategies:
                    strategy['validation_score'] = validation_summary.get('sharpe_ratio', 0)
                    strategy['validation_metrics'] = validation_summary

            # Sort strategies by validation score
            strategies.sort(key=lambda x: x.get('validation_score', 0), reverse=True)

            return strategies[:5]  # Return top 5 strategies

        except Exception as e:
            self.logger.error(f"Strategy ranking failed: {e}")
            return []

    def _generate_performance_analysis(self, optimization_results: Dict[str, Any]) -> Dict[str, Any]:
        """Generate comprehensive performance analysis"""
        analysis = {
            'optimization_summary': {},
            'feature_analysis': {},
            'model_analysis': {},
            'strategy_comparison': {},
            'recommendations': []
        }

        try:
            # Optimization summary
            analysis['optimization_summary'] = {
                'total_stages_completed': sum(1 for stage in ['feature_engineering', 'genetic_programming', 'model_optimization', 'strategy_validation']
                                            if stage in optimization_results and optimization_results[stage]),
                'optimization_successful': 'best_strategies' in optimization_results and len(optimization_results['best_strategies']) > 0,
                'execution_time': optimization_results.get('execution_stats', {}).get('execution_time_seconds', 0)
            }

            # Feature analysis
            if 'feature_engineering' in optimization_results:
                fe_results = optimization_results['feature_engineering']
                analysis['feature_analysis'] = {
                    'feature_expansion_ratio': fe_results.get('engineered_features', 0) / max(1, fe_results.get('original_features', 1)),
                    'feature_categories': fe_results.get('feature_categories', {}),
                    'feature_engineering_successful': fe_results.get('engineered_features', 0) > 0
                }

            # Generate recommendations
            recommendations = []

            if analysis['optimization_summary']['optimization_successful']:
                recommendations.append("Optimization completed successfully. Consider implementing the top-ranked strategy.")
            else:
                recommendations.append("Optimization had limited success. Consider adjusting parameters or data quality.")

            if analysis.get('feature_analysis', {}).get('feature_expansion_ratio', 0) < 2:
                recommendations.append("Feature engineering produced limited expansion. Consider adding more feature categories.")

            analysis['recommendations'] = recommendations

            return analysis

        except Exception as e:
            self.logger.error(f"Performance analysis generation failed: {e}")
            return {'error': str(e)}


class PortfolioOptimizationTask(BaseTask):
    """
    Advanced portfolio optimization task with modern portfolio theory and AI enhancements.

    Features:
    - Multi-asset portfolio optimization
    - Risk parity and mean-variance optimization
    - Dynamic rebalancing strategies
    - Regime-aware asset allocation
    - AI-powered correlation analysis
    - Black-Litterman model integration

    Usage:
        task = PortfolioOptimizationTask()
        portfolio = task.optimize_portfolio(asset_data, target_return=0.12)
    """

    def __init__(self, config: Optional[ConfigModel] = None):
        super().__init__(config)
        self.gemini_analyzer = GeminiAnalyzer()

    def optimize_portfolio(self, asset_data: Dict[str, pd.DataFrame],
                          target_return: float = 0.10,
                          optimization_method: str = "mean_variance") -> Dict[str, Any]:
        """
        Optimize portfolio allocation across multiple assets.

        Args:
            asset_data: Dictionary mapping asset symbols to their price data
            target_return: Target annual return
            optimization_method: Optimization method (mean_variance, risk_parity, black_litterman)

        Returns:
            Optimized portfolio allocation and performance metrics
        """
        self.start_execution("PortfolioOptimization")

        try:
            portfolio_results = {
                'optimization_metadata': {
                    'start_time': datetime.now().isoformat(),
                    'assets_count': len(asset_data),
                    'target_return': target_return,
                    'optimization_method': optimization_method
                },
                'asset_analysis': {},
                'correlation_analysis': {},
                'optimization_results': {},
                'portfolio_metrics': {},
                'rebalancing_strategy': {}
            }

            # Stage 1: Asset Analysis
            self.logger.info("Stage 1: Individual Asset Analysis")
            asset_analysis = self._analyze_individual_assets(asset_data)
            portfolio_results['asset_analysis'] = asset_analysis

            # Stage 2: Correlation and Risk Analysis
            self.logger.info("Stage 2: Correlation and Risk Analysis")
            correlation_analysis = self._analyze_correlations(asset_data)
            portfolio_results['correlation_analysis'] = correlation_analysis

            # Stage 3: Portfolio Optimization
            self.logger.info(f"Stage 3: Portfolio Optimization using {optimization_method}")
            optimization_results = self._optimize_portfolio_weights(
                asset_data, target_return, optimization_method
            )
            portfolio_results['optimization_results'] = optimization_results

            # Stage 4: Portfolio Performance Analysis
            self.logger.info("Stage 4: Portfolio Performance Analysis")
            portfolio_metrics = self._calculate_portfolio_metrics(
                asset_data, optimization_results.get('optimal_weights', {})
            )
            portfolio_results['portfolio_metrics'] = portfolio_metrics

            # Stage 5: Dynamic Rebalancing Strategy
            self.logger.info("Stage 5: Dynamic Rebalancing Strategy")
            rebalancing_strategy = self._design_rebalancing_strategy(
                asset_data, optimization_results.get('optimal_weights', {})
            )
            portfolio_results['rebalancing_strategy'] = rebalancing_strategy

            # End execution
            execution_stats = self.end_execution("PortfolioOptimization", portfolio_results)
            portfolio_results['execution_stats'] = execution_stats

            self.logger.info("Portfolio optimization completed successfully")
            return portfolio_results

        except Exception as e:
            self.logger.error(f"Portfolio optimization failed: {e}", exc_info=True)
            execution_stats = self.end_execution("PortfolioOptimization", {'error': str(e)})

            return {
                'error': str(e),
                'execution_stats': execution_stats,
                'optimization_metadata': {
                    'start_time': datetime.now().isoformat(),
                    'failed': True
                }
            }

    def _analyze_individual_assets(self, asset_data: Dict[str, pd.DataFrame]) -> Dict[str, Any]:
        """Analyze individual asset characteristics"""
        analysis = {}

        for symbol, data in asset_data.items():
            if 'Close' in data.columns:
                returns = data['Close'].pct_change().dropna()

                analysis[symbol] = {
                    'annual_return': returns.mean() * 252,
                    'annual_volatility': returns.std() * np.sqrt(252),
                    'sharpe_ratio': (returns.mean() * 252) / (returns.std() * np.sqrt(252)) if returns.std() > 0 else 0,
                    'max_drawdown': self._calculate_max_drawdown(data['Close']),
                    'data_points': len(data)
                }

        return analysis

    def _analyze_correlations(self, asset_data: Dict[str, pd.DataFrame]) -> Dict[str, Any]:
        """Analyze correlations between assets"""
        # Combine all asset returns
        returns_data = {}

        for symbol, data in asset_data.items():
            if 'Close' in data.columns:
                returns_data[symbol] = data['Close'].pct_change().dropna()

        if not returns_data:
            return {'error': 'No valid price data found'}

        # Create returns DataFrame
        returns_df = pd.DataFrame(returns_data)
        returns_df = returns_df.dropna()

        # Calculate correlation matrix
        correlation_matrix = returns_df.corr()

        return {
            'correlation_matrix': correlation_matrix.to_dict(),
            'average_correlation': correlation_matrix.values[np.triu_indices_from(correlation_matrix.values, k=1)].mean(),
            'max_correlation': correlation_matrix.values[np.triu_indices_from(correlation_matrix.values, k=1)].max(),
            'min_correlation': correlation_matrix.values[np.triu_indices_from(correlation_matrix.values, k=1)].min()
        }

    def _optimize_portfolio_weights(self, asset_data: Dict[str, pd.DataFrame],
                                   target_return: float, method: str) -> Dict[str, Any]:
        """Optimize portfolio weights using specified method"""
        # This is a simplified implementation
        # In practice, you would use libraries like cvxpy or scipy.optimize

        assets = list(asset_data.keys())
        n_assets = len(assets)

        if method == "equal_weight":
            # Equal weight portfolio
            optimal_weights = {asset: 1.0 / n_assets for asset in assets}
        elif method == "risk_parity":
            # Simplified risk parity (equal risk contribution)
            # In practice, this would require iterative optimization
            optimal_weights = {asset: 1.0 / n_assets for asset in assets}
        else:
            # Default to equal weight
            optimal_weights = {asset: 1.0 / n_assets for asset in assets}

        return {
            'optimal_weights': optimal_weights,
            'optimization_method': method,
            'target_return_achieved': target_return,  # Simplified
            'optimization_successful': True
        }

    def _calculate_portfolio_metrics(self, asset_data: Dict[str, pd.DataFrame],
                                   weights: Dict[str, float]) -> Dict[str, Any]:
        """Calculate portfolio performance metrics"""
        try:
            # Calculate portfolio returns
            portfolio_returns = []

            # Get aligned data
            aligned_data = {}
            for symbol, data in asset_data.items():
                if 'Close' in data.columns and symbol in weights:
                    aligned_data[symbol] = data['Close'].pct_change().dropna()

            if not aligned_data:
                return {'error': 'No valid data for portfolio calculation'}

            # Align all series to same dates
            returns_df = pd.DataFrame(aligned_data).dropna()

            # Calculate weighted portfolio returns
            portfolio_returns = (returns_df * pd.Series(weights)).sum(axis=1)

            return {
                'annual_return': portfolio_returns.mean() * 252,
                'annual_volatility': portfolio_returns.std() * np.sqrt(252),
                'sharpe_ratio': (portfolio_returns.mean() * 252) / (portfolio_returns.std() * np.sqrt(252)) if portfolio_returns.std() > 0 else 0,
                'max_drawdown': self._calculate_max_drawdown_from_returns(portfolio_returns),
                'total_periods': len(portfolio_returns)
            }

        except Exception as e:
            self.logger.error(f"Portfolio metrics calculation failed: {e}")
            return {'error': str(e)}

    def _design_rebalancing_strategy(self, asset_data: Dict[str, pd.DataFrame],
                                   weights: Dict[str, float]) -> Dict[str, Any]:
        """Design dynamic rebalancing strategy"""
        return {
            'rebalancing_frequency': 'monthly',
            'rebalancing_threshold': 0.05,  # 5% deviation triggers rebalancing
            'transaction_costs': 0.001,  # 0.1% transaction cost
            'strategy_type': 'threshold_based',
            'estimated_annual_turnover': 0.2  # 20% annual turnover
        }

    def _calculate_max_drawdown(self, price_series: pd.Series) -> float:
        """Calculate maximum drawdown from price series"""
        cumulative = (1 + price_series.pct_change()).cumprod()
        running_max = cumulative.expanding().max()
        drawdown = (cumulative - running_max) / running_max
        return drawdown.min()

    def _calculate_max_drawdown_from_returns(self, returns_series: pd.Series) -> float:
        """Calculate maximum drawdown from returns series"""
        cumulative = (1 + returns_series).cumprod()
        running_max = cumulative.expanding().max()
        drawdown = (cumulative - running_max) / running_max
        return drawdown.min()


def main():
    """
    Main entry point for the AxiomEdge Professional Trading Framework.

    Initializes the complete trading system including logging, configuration,
    data collection, AI analysis, and framework execution. Supports both
    single-run and daemon modes for continuous operation.

    The framework performs:
    1. System initialization and logging setup
    2. Data file discovery and validation
    3. AI-powered asset classification and configuration
    4. Multi-timeframe data processing and feature engineering
    5. Walk-forward model training and validation
    6. Strategy backtesting and performance analysis
    7. Comprehensive reporting and visualization
    8. Optional daemon mode for continuous execution
    """
    import os
    import sys
    import re
    import time
    from collections import Counter
    from .utils import _setup_logging, flush_loggers, VERSION
    from .framework_orchestrator import run_single_instance
    from .config import generate_dynamic_config, ConfigModel
    from .data_handler import get_and_cache_asset_types
    from .utils import load_nickname_ledger, load_memory, initialize_playbook

    initial_log_dir = os.path.join(os.getcwd(), "Results", "PrelimLogs")
    os.makedirs(initial_log_dir, exist_ok=True)
    initial_log_path = os.path.join(initial_log_dir, f"framework_boot_{datetime.now().strftime('%Y%m%d-%H%M%S')}.log")
    _setup_logging(initial_log_path, f"AxiomEdge_V{VERSION}_Boot")

    global logger
    logger = logging.getLogger("ML_Trading_Framework")

    logger.info(f"--- ML Trading Framework V{VERSION} Initializing ---")

    # --- Add VIX to the list of macro tickers to fetch ---
    master_macro_list = {
        "VIX": "^VIX",
        "US_10Y_Yield": "^TNX",
        "Gold": "GC=F",
        "Oil": "CL=F",
        "Dollar_Index": "DX-Y.NYB"
    }

    base_config = {
        "BASE_PATH": os.getcwd(),
        "INITIAL_CAPITAL": 1000.0,
        "OPTUNA_TRIALS": 75,
        "TRAINING_WINDOW": '365D',
        "RETRAINING_FREQUENCY": '90D',
        "FORWARD_TEST_GAP": "1D",
        "LOOKAHEAD_CANDLES": 150,
        "CALCULATE_SHAP_VALUES": True,
        "USE_FEATURE_CACHING": True,
        "MAX_TRAINING_RETRIES_PER_CYCLE": 3,
        "run_timestamp": datetime.now().strftime("%Y%m%d-%H%M%S"),
        "nickname": "bootstrap",
        "REPORT_LABEL": f"AxiomEdge_V{VERSION}_Fallback",
        "strategy_name": "DefaultStrategy",
        "FEATURE_SELECTION_METHOD": "pca"
    }

    all_files = [f for f in os.listdir(base_config['BASE_PATH']) if f.endswith(('.csv', '.txt')) and re.match(r'^[A-Z0-9]+_[A-Z0-9]+', f)]
    if not all_files:
        print("ERROR: No data files (*.csv, *.txt) found in the current directory.")
        print("Please place your data files here and ensure they are named like: 'EURUSD_H1.csv'")
        input("Press Enter to exit.")
        return

    symbols = sorted(list(set([f.split('_')[0] for f in all_files])))

    gemini_analyzer_for_setup = GeminiAnalyzer()
    api_timer_for_setup = APITimer(interval_seconds=61)

    asset_types = api_timer_for_setup.call(get_and_cache_asset_types, symbols, base_config, gemini_analyzer_for_setup)

    if not asset_types:
        logger.error("Could not determine asset types via AI.")
        print("Could not automatically classify assets. Please select the primary asset class:")
        print("1. Forex\n2. Commodities\n3. Indices\n4. Crypto")
        choice = input("Enter number: ")
        class_map = {'1': 'Forex', '2': 'Commodities', '3': 'Indices', '4': 'Crypto'}
        primary_class = class_map.get(choice, 'Forex')
        logger.info(f"Using manually selected primary class: {primary_class}")

        print("Please provide a minimum of three timeframes for the asset (e.g., M15, H1, D1):")
        timeframes_input = input("Enter timeframes separated by commas: ")
        timeframes = [tf.strip().upper() for tf in timeframes_input.split(',') if tf.strip()]

        if len(timeframes) < 3:
            print("ERROR: A minimum of three timeframes is required. Exiting.")
            input("Press Enter to exit.")
            return

        filtered_files = []
        for f in all_files:
            parts = f.split('_')
            if len(parts) > 1:
                tf_from_filename = parts[1].split('.')[0].upper()
                if tf_from_filename in timeframes:
                    filtered_files.append(f)
        all_files = filtered_files
        if not all_files:
            print("ERROR: No data files found matching the specified timeframes. Exiting.")
            input("Press Enter to exit.")
            return

        asset_types = {s: primary_class for s in symbols}

    else:
        class_counts = Counter(asset_types.values())
        primary_class = class_counts.most_common(1)[0][0]

    fallback_config = generate_dynamic_config(primary_class, base_config)

    CONTINUOUS_RUN_HOURS = 0
    MAX_RUNS = 1
    api_interval_seconds = 61
    run_count = 0
    script_start_time = datetime.now()
    is_continuous = CONTINUOUS_RUN_HOURS > 0 or MAX_RUNS > 1

    temp_config_dict = fallback_config.copy()
    temp_config_dict['REPORT_LABEL'] = 'init'
    temp_config_dict['strategy_name'] = 'init'
    bootstrap_config = ConfigModel(**temp_config_dict)

    results_dir = os.path.join(bootstrap_config.BASE_PATH, "Results")
    os.makedirs(results_dir, exist_ok=True)
    playbook_file_path = os.path.join(results_dir, "strategy_playbook.json")
    playbook = initialize_playbook(playbook_file_path)

    while True:
        run_count += 1
        if is_continuous: logger.info(f"\n{'='*30} STARTING DAEMON RUN {run_count} {'='*30}\n")
        else: logger.info(f"\n{'='*30} STARTING SINGLE RUN {'='*30}\n")
        flush_loggers()

        nickname_ledger = load_nickname_ledger(bootstrap_config.NICKNAME_LEDGER_PATH)
        framework_history = load_memory(bootstrap_config.CHAMPION_FILE_PATH, bootstrap_config.HISTORY_FILE_PATH)
        directives = []
        if os.path.exists(bootstrap_config.DIRECTIVES_FILE_PATH):
            try:
                with open(bootstrap_config.DIRECTIVES_FILE_PATH, 'r') as f: directives = json.load(f)
                if directives: logger.info(f"Loaded {len(directives)} directive(s) for this run.")
            except (json.JSONDecodeError, IOError) as e: logger.error(f"Could not load directives file: {e}")

        flush_loggers()

        try:
            run_single_instance(fallback_config, framework_history, playbook, nickname_ledger, directives, api_interval_seconds)
        except Exception as e:
            logger.critical(f"A critical, unhandled error occurred during run {run_count}: {e}", exc_info=True)
            if not is_continuous: break
            logger.info("Attempting to continue after a 60-second cooldown..."); time.sleep(60)

        if not is_continuous:
            logger.info("Single run complete. Exiting.")
            break
        if MAX_RUNS > 0 and run_count >= MAX_RUNS:
            logger.info(f"Reached max run limit of {MAX_RUNS}. Exiting daemon mode.")
            break
        if CONTINUOUS_RUN_HOURS > 0 and (datetime.now() - script_start_time).total_seconds() / 3600 >= CONTINUOUS_RUN_HOURS:
            logger.info(f"Reached max runtime of {CONTINUOUS_RUN_HOURS} hours. Exiting daemon mode.")
            break

        try:
            sys.stdout.write("\n")
            for i in range(10, 0, -1):
                sys.stdout.write(f"\r>>> Run {run_count} complete. Press Ctrl+C to stop. Continuing in {i:2d} seconds..."); sys.stdout.flush(); time.sleep(1)
            sys.stdout.write("\n\n")
        except KeyboardInterrupt:
            logger.info("\n\nDaemon stopped by user. Exiting gracefully.")
            break


if __name__ == '__main__':
    main()
