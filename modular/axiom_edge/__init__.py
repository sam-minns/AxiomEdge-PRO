# AxiomEdge Trading Framework
# Modular Architecture for Flexible Usage

__version__ = "2.1.1"
__author__ = "AxiomEdge Team"

# Core modules for independent usage
from .config import ConfigModel, OperatingState, EarlyInterventionConfig
from .data_handler import DataHandler, DataLoader
from .ai_analyzer import GeminiAnalyzer, APITimer

# Note: These modules will be created as stubs that import from the main file
# until the full modularization is complete
try:
    from .feature_engineer import FeatureEngineer
except ImportError:
    from .stubs import FeatureEngineerStub as FeatureEngineer

try:
    from .model_trainer import ModelTrainer
except ImportError:
    from .stubs import ModelTrainerStub as ModelTrainer

try:
    from .backtester import Backtester
except ImportError:
    from .stubs import BacktesterStub as Backtester

try:
    from .report_generator import ReportGenerator
except ImportError:
    from .stubs import ReportGeneratorStub as ReportGenerator

try:
    from .genetic_programmer import GeneticProgrammer
except ImportError:
    from .stubs import GeneticProgrammerStub as GeneticProgrammer

try:
    from .framework_orchestrator import FrameworkOrchestrator
except ImportError:
    from .stubs import FrameworkOrchestratorStub as FrameworkOrchestrator

# Utility modules
from .utils import (
    get_optimal_system_settings,
    json_serializer_default,
    flush_loggers
)

# Task-specific interfaces
from .tasks import (
    DataCollectionTask,
    BrokerInfoTask,
    BacktestTask,
    FeatureEngineeringTask,
    ModelTrainingTask,
    CompleteFrameworkTask
)

__all__ = [
    # Core Components
    'ConfigModel', 'OperatingState', 'EarlyInterventionConfig',
    'DataHandler', 'DataLoader',
    'GeminiAnalyzer', 'APITimer',
    'FeatureEngineer',
    'ModelTrainer',
    'Backtester',
    'ReportGenerator',
    'GeneticProgrammer',
    
    # Orchestration
    'FrameworkOrchestrator',
    
    # Utilities
    'get_optimal_system_settings',
    'json_serializer_default',
    'flush_loggers',
    
    # Task Interfaces
    'DataCollectionTask',
    'BrokerInfoTask', 
    'BacktestTask',
    'FeatureEngineeringTask',
    'ModelTrainingTask',
    'CompleteFrameworkTask'
]
