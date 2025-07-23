# AxiomEdge Trading Framework
# Modular Architecture for Flexible Usage

__version__ = "2.1.1"
__author__ = "AxiomEdge Team"

# Core modules for independent usage
from .config import ConfigModel, OperatingState, EarlyInterventionConfig
from .data_handler import DataHandler, DataLoader
from .ai_analyzer import GeminiAnalyzer, APITimer

# Feature engineering is now fully modularized
from .feature_engineer import FeatureEngineer

# Model training is now fully modularized
from .model_trainer import ModelTrainer

# Backtesting is now fully modularized
from .backtester import Backtester, PerformanceAnalyzer

# Report generation is now fully modularized
from .report_generator import ReportGenerator

# Genetic programming is now fully modularized
from .genetic_programmer import GeneticProgrammer

# Framework orchestration is now fully modularized
from .framework_orchestrator import FrameworkOrchestrator

# Telemetry and monitoring system
from .telemetry import TelemetryCollector, TelemetryAnalyzer

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
    # Core Components (Fully Modularized)
    'ConfigModel', 'OperatingState', 'EarlyInterventionConfig',
    'DataHandler', 'DataLoader',
    'GeminiAnalyzer', 'APITimer',
    'FeatureEngineer',       # ✅ Fully implemented
    'ModelTrainer',          # ✅ Fully implemented
    'GeneticProgrammer',     # ✅ Fully implemented
    'ReportGenerator',       # ✅ Fully implemented
    'FrameworkOrchestrator', # ✅ Fully implemented
    'Backtester',            # ✅ Fully implemented
    'PerformanceAnalyzer',   # ✅ Fully implemented

    # Telemetry and Monitoring
    'TelemetryCollector',    # ✅ Fully implemented
    'TelemetryAnalyzer',     # ✅ Fully implemented

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
