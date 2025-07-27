# AxiomEdge Trading Framework
# Advanced Modular Architecture for Professional Trading Systems

__version__ = "2.1.1"
__author__ = "AxiomEdge Team"
__description__ = "Professional-grade trading framework with AI-powered strategy optimization"
__license__ = "Proprietary"

# Framework metadata
FRAMEWORK_INFO = {
    'version': __version__,
    'build_date': '2024-01-01',
    'components': [
        'DataHandler', 'FeatureEngineer', 'ModelTrainer', 'Backtester',
        'GeneticProgrammer', 'ReportGenerator', 'FrameworkOrchestrator',
        'TelemetryCollector', 'TelemetryAnalyzer', 'AIAnalyzer'
    ],
    'capabilities': [
        'Multi-source data collection', '200+ technical indicators',
        'Advanced ML model training', 'Genetic programming optimization',
        'Comprehensive backtesting', 'AI-powered analysis',
        'Real-time telemetry', 'Interactive reporting'
    ]
}

# Core configuration and state management
from .config import (
    ConfigModel,
    OperatingState,
    EarlyInterventionConfig,
    create_default_config,
    validate_config,
    get_config_summary,
    validate_framework_configuration,
    generate_dynamic_config
)

# Data handling and collection
from .data_handler import (
    DataHandler,
    DataLoader,
    CacheManager,
    DataValidator,
    get_and_cache_asset_types,
    get_and_cache_contract_sizes,
    determine_timeframe_roles
)

# AI analysis and optimization
from .ai_analyzer import (
    GeminiAnalyzer,
    APITimer,
    ModelPriorityManager,
    AIResponseValidator
)

# Advanced feature engineering (200+ indicators)
from .feature_engineer import (
    FeatureEngineer,
    TechnicalIndicators,
    StatisticalFeatures,
    BehavioralPatterns,
    AnomalyDetector
)

# Machine learning and model training
from .model_trainer import (
    ModelTrainer,
    EnsembleManager,
    HyperparameterOptimizer,
    SHAPAnalyzer,
    ModelValidator
)

# Comprehensive backtesting and performance analysis
from .backtester import (
    Backtester,
    PerformanceAnalyzer,
    TradeManager,
    RiskManager,
    ExecutionSimulator
)

# Advanced genetic programming and strategy discovery
from .genetic_programmer import (
    GeneticProgrammer,
    ChromosomeManager,
    FitnessEvaluator,
    EvolutionTracker,
    PopulationManager
)

# Comprehensive reporting and visualization
from .report_generator import (
    ReportGenerator,
    VisualizationEngine,
    DashboardGenerator,
    MetricsCalculator,
    ChartGenerator
)

# Framework orchestration and coordination
from .framework_orchestrator import (
    FrameworkOrchestrator,
    WorkflowManager,
    StateManager,
    ComponentCoordinator
)

# Advanced telemetry and AI doctor capabilities
from .telemetry import (
    TelemetryCollector,
    TelemetryAnalyzer,
    InterventionManager,
    SystemMonitor,
    PerformanceTracker,
    AIDoctorAnalyzer
)

# Comprehensive utility functions
from .utils import (
    get_optimal_system_settings,
    json_serializer_default,
    flush_loggers,
    setup_logging,
    validate_data_quality,
    calculate_memory_usage,
    optimize_dataframe,
    safe_divide,
    normalize_features,
    handle_missing_data,
    detect_outliers,
    calculate_correlation_matrix,
    generate_feature_importance_report,
    create_performance_summary,
    export_results,
    load_configuration,
    save_configuration,
    validate_file_path,
    ensure_directory_exists,
    get_system_info,
    monitor_resource_usage,
    create_backup,
    cleanup_temporary_files,
    _setup_logging,
    deep_merge_dicts,
    get_walk_forward_periods,
    _generate_nickname,
    _log_config_and_environment
)

# Advanced task-specific interfaces
from .tasks import (
    BaseTask,
    DataCollectionTask,
    BrokerInfoTask,
    BacktestTask,
    FeatureEngineeringTask,
    ModelTrainingTask,
    CompleteFrameworkTask,
    AIStrategyOptimizationTask,
    PortfolioOptimizationTask,
    main as framework_main
)

# Optional imports with graceful fallbacks
try:
    from .telemetry import TelemetryCollector, TelemetryAnalyzer
    TELEMETRY_AVAILABLE = True
except ImportError:
    TELEMETRY_AVAILABLE = False
    TelemetryCollector = None
    TelemetryAnalyzer = None

try:
    from .genetic_programmer import GeneticProgrammer
    GENETIC_PROGRAMMING_AVAILABLE = True
except ImportError:
    GENETIC_PROGRAMMING_AVAILABLE = False
    GeneticProgrammer = None

# Version compatibility checks
import sys
PYTHON_VERSION = sys.version_info
MIN_PYTHON_VERSION = (3, 8)

if PYTHON_VERSION < MIN_PYTHON_VERSION:
    raise RuntimeError(f"AxiomEdge requires Python {MIN_PYTHON_VERSION[0]}.{MIN_PYTHON_VERSION[1]}+ "
                      f"but you are using Python {PYTHON_VERSION[0]}.{PYTHON_VERSION[1]}")

# Framework capabilities detection
CAPABILITIES = {
    'telemetry': TELEMETRY_AVAILABLE,
    'genetic_programming': GENETIC_PROGRAMMING_AVAILABLE,
    'ai_analysis': True,  # Always available
    'advanced_features': True,  # Always available
    'backtesting': True,  # Always available
    'reporting': True,  # Always available
}

__all__ = [
    # Framework metadata
    '__version__', '__author__', '__description__', '__license__',
    'FRAMEWORK_INFO', 'CAPABILITIES',

    # Core Configuration and State Management
    'ConfigModel', 'OperatingState', 'EarlyInterventionConfig',
    'create_default_config', 'validate_config', 'get_config_summary', 'generate_dynamic_config',

    # Data Handling and Collection
    'DataHandler', 'DataLoader', 'CacheManager', 'DataValidator',
    'get_and_cache_asset_types', 'get_and_cache_contract_sizes', 'determine_timeframe_roles',

    # AI Analysis and Optimization
    'GeminiAnalyzer', 'APITimer', 'ModelPriorityManager', 'AIResponseValidator',

    # Advanced Feature Engineering (200+ indicators)
    'FeatureEngineer', 'TechnicalIndicators', 'StatisticalFeatures',
    'BehavioralPatterns', 'AnomalyDetector',

    # Machine Learning and Model Training
    'ModelTrainer', 'EnsembleManager', 'HyperparameterOptimizer',
    'SHAPAnalyzer', 'ModelValidator',

    # Comprehensive Backtesting and Performance Analysis
    'Backtester', 'PerformanceAnalyzer', 'TradeManager',
    'RiskManager', 'ExecutionSimulator',

    # Advanced Genetic Programming and Strategy Discovery
    'GeneticProgrammer', 'ChromosomeManager', 'FitnessEvaluator',
    'EvolutionTracker', 'PopulationManager',

    # Comprehensive Reporting and Visualization
    'ReportGenerator', 'VisualizationEngine', 'DashboardGenerator',
    'MetricsCalculator', 'ChartGenerator',

    # Framework Orchestration and Coordination
    'FrameworkOrchestrator', 'WorkflowManager', 'StateManager', 'ComponentCoordinator',

    # Advanced Telemetry and AI Doctor Capabilities
    'TelemetryCollector', 'TelemetryAnalyzer', 'InterventionManager',
    'SystemMonitor', 'PerformanceTracker', 'AIDoctorAnalyzer',

    # Comprehensive Utility Functions
    'get_optimal_system_settings', 'json_serializer_default', 'flush_loggers',
    'setup_logging', 'validate_data_quality', 'calculate_memory_usage',
    'optimize_dataframe', 'safe_divide', 'normalize_features',
    'handle_missing_data', 'detect_outliers', 'calculate_correlation_matrix',
    'generate_feature_importance_report', 'create_performance_summary',
    'export_results', 'load_configuration', 'save_configuration',
    'validate_file_path', 'ensure_directory_exists', 'get_system_info',
    'monitor_resource_usage', 'create_backup', 'cleanup_temporary_files',
    '_setup_logging', 'deep_merge_dicts', 'get_walk_forward_periods',
    '_generate_nickname', '_log_config_and_environment',

    # Advanced Task-Specific Interfaces
    'BaseTask', 'DataCollectionTask', 'BrokerInfoTask', 'BacktestTask',
    'FeatureEngineeringTask', 'ModelTrainingTask', 'CompleteFrameworkTask',
    'AIStrategyOptimizationTask', 'PortfolioOptimizationTask', 'framework_main',

    # Capability flags
    'TELEMETRY_AVAILABLE', 'GENETIC_PROGRAMMING_AVAILABLE',

    # Convenience functions
    'create_framework', 'get_framework_info', 'validate_installation',
    'quick_backtest', 'quick_feature_engineering', 'quick_model_training'
]


# Convenience functions for quick framework usage
def create_framework(config_path: str = None, enable_telemetry: bool = True) -> FrameworkOrchestrator:
    """
    Create a fully configured AxiomEdge framework instance.

    Args:
        config_path: Optional path to configuration file
        enable_telemetry: Whether to enable telemetry collection

    Returns:
        Configured FrameworkOrchestrator instance
    """
    if config_path:
        config = load_configuration(config_path)
    else:
        config = create_default_config("./")

    # Enable telemetry if available and requested
    if enable_telemetry and TELEMETRY_AVAILABLE:
        config.ENABLE_TELEMETRY = True

    return FrameworkOrchestrator(config)


def get_framework_info() -> dict:
    """
    Get comprehensive framework information and capabilities.

    Returns:
        Dictionary containing framework metadata and capabilities
    """
    return {
        **FRAMEWORK_INFO,
        'capabilities_available': CAPABILITIES,
        'python_version': f"{PYTHON_VERSION[0]}.{PYTHON_VERSION[1]}.{PYTHON_VERSION[2]}",
        'telemetry_available': TELEMETRY_AVAILABLE,
        'genetic_programming_available': GENETIC_PROGRAMMING_AVAILABLE,
        'total_exports': len(__all__)
    }


def validate_installation() -> dict:
    """
    Validate AxiomEdge installation and dependencies.

    Returns:
        Dictionary containing validation results
    """
    validation_results = {
        'installation_valid': True,
        'python_version_ok': PYTHON_VERSION >= MIN_PYTHON_VERSION,
        'core_modules_available': True,
        'optional_modules': {},
        'warnings': [],
        'errors': []
    }

    # Check optional modules
    validation_results['optional_modules']['telemetry'] = TELEMETRY_AVAILABLE
    validation_results['optional_modules']['genetic_programming'] = GENETIC_PROGRAMMING_AVAILABLE

    # Check for common issues
    if not TELEMETRY_AVAILABLE:
        validation_results['warnings'].append('Telemetry module not available - advanced monitoring disabled')

    if not GENETIC_PROGRAMMING_AVAILABLE:
        validation_results['warnings'].append('Genetic programming module not available - strategy evolution disabled')

    # Try importing core modules
    try:
        from . import config, data_handler, feature_engineer, model_trainer, backtester
        validation_results['core_modules_available'] = True
    except ImportError as e:
        validation_results['installation_valid'] = False
        validation_results['core_modules_available'] = False
        validation_results['errors'].append(f'Core module import failed: {e}')

    return validation_results


def quick_backtest(data, strategy_config: dict = None) -> dict:
    """
    Quick backtesting interface for simple strategy testing.

    Args:
        data: Price data (DataFrame or file path)
        strategy_config: Optional strategy configuration

    Returns:
        Backtesting results
    """
    config = create_default_config("./")
    if strategy_config:
        for key, value in strategy_config.items():
            setattr(config, key, value)

    task = BacktestTask(config)
    return task.run_backtest(data)


def quick_feature_engineering(data, feature_config: dict = None) -> 'pd.DataFrame':
    """
    Quick feature engineering interface for data preparation.

    Args:
        data: Price data (DataFrame or file path)
        feature_config: Optional feature configuration

    Returns:
        Engineered features DataFrame
    """
    config = create_default_config("./")
    if feature_config:
        for key, value in feature_config.items():
            setattr(config, key, value)

    task = FeatureEngineeringTask(config)
    return task.engineer_features(data)


def quick_model_training(data, target_column: str = None, model_config: dict = None) -> dict:
    """
    Quick model training interface for ML model development.

    Args:
        data: Labeled training data (DataFrame or file path)
        target_column: Target column name
        model_config: Optional model configuration

    Returns:
        Model training results
    """
    config = create_default_config("./")
    if model_config:
        for key, value in model_config.items():
            setattr(config, key, value)

    task = ModelTrainingTask(config)
    return task.train_models(data, target_column)


# Framework initialization message
def _print_welcome_message():
    """Print welcome message when framework is imported."""
    if not hasattr(_print_welcome_message, '_called'):
        print(f"🚀 AxiomEdge Trading Framework v{__version__} loaded successfully!")
        print(f"📊 {len(FRAMEWORK_INFO['components'])} core components available")
        print(f"⚡ {len(FRAMEWORK_INFO['capabilities'])} advanced capabilities enabled")

        if not TELEMETRY_AVAILABLE:
            print("⚠️  Telemetry module not available - install optional dependencies for full functionality")

        if not GENETIC_PROGRAMMING_AVAILABLE:
            print("⚠️  Genetic programming module not available - install optional dependencies for strategy evolution")

        print("📖 Use get_framework_info() for detailed information")
        print("🔧 Use validate_installation() to check your setup")
        _print_welcome_message._called = True


# Initialize framework on import
_print_welcome_message()
