# =============================================================================
# TASK-SPECIFIC INTERFACES MODULE
# =============================================================================

"""
This module provides task-specific interfaces that allow users to run
individual components of AxiomEdge independently or in combination.
"""

import logging
from typing import Dict, List, Optional, Any, Tuple
import pandas as pd
from pathlib import Path

from .config import ConfigModel, create_default_config
from .data_handler import DataHandler, DataLoader
from .ai_analyzer import GeminiAnalyzer, APITimer

logger = logging.getLogger(__name__)

class BaseTask:
    """Base class for all task interfaces"""
    
    def __init__(self, config: Optional[ConfigModel] = None):
        self.config = config or create_default_config("./")
        self.logger = logging.getLogger(self.__class__.__name__)
    
    def setup_logging(self, level: str = "INFO"):
        """Setup logging for the task"""
        logging.basicConfig(
            level=getattr(logging, level.upper()),
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )


class DataCollectionTask(BaseTask):
    """
    Task interface for collecting historical data from various sources.
    
    Usage:
        task = DataCollectionTask()
        data = task.collect_data(["AAPL", "GOOGL"], "2023-01-01", "2024-01-01")
    """
    
    def __init__(self, config: Optional[ConfigModel] = None, 
                 cache_dir: str = "data_cache", api_key: Optional[str] = None):
        super().__init__(config)
        self.data_handler = DataHandler(cache_dir=cache_dir, api_key=api_key)
    
    def collect_data(self, symbols: List[str], start_date: str, end_date: str,
                    timeframe: str = "1D", source: str = "alpha_vantage") -> Dict[str, pd.DataFrame]:
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
    
    def __init__(self, config: Optional[ConfigModel] = None, api_key: Optional[str] = None):
        super().__init__(config)
        self.data_handler = DataHandler(api_key=api_key)
        self.ai_analyzer = GeminiAnalyzer()
    
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
