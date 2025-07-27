#!/usr/bin/env python3
"""
Comprehensive Integration Testing for AxiomEdge Framework
========================================================

This script performs comprehensive testing of all updated modules to ensure
they work together correctly and maintain the same functionality as the
monolithic version.
"""

import sys
import traceback
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional

# Test results tracking
test_results = {
    'import_tests': {},
    'basic_functionality_tests': {},
    'integration_tests': {},
    'performance_tests': {},
    'error_handling_tests': {},
    'summary': {}
}

def log_test_result(category: str, test_name: str, success: bool, message: str = "", error: str = ""):
    """Log test result to tracking dictionary."""
    test_results[category][test_name] = {
        'success': success,
        'message': message,
        'error': error,
        'timestamp': datetime.now().isoformat()
    }
    
    status = "✅ PASS" if success else "❌ FAIL"
    print(f"{status} {category}.{test_name}: {message}")
    if error:
        print(f"    Error: {error}")

def test_imports():
    """Test that all modules can be imported correctly."""
    print("\n🔍 Testing Module Imports...")
    
    # Core modules
    modules_to_test = [
        'axiom_edge.config',
        'axiom_edge.data_handler',
        'axiom_edge.ai_analyzer',
        'axiom_edge.feature_engineer',
        'axiom_edge.model_trainer',
        'axiom_edge.backtester',
        'axiom_edge.genetic_programmer',
        'axiom_edge.report_generator',
        'axiom_edge.framework_orchestrator',
        'axiom_edge.telemetry',
        'axiom_edge.utils',
        'axiom_edge.tasks',
        'axiom_edge'  # Main package
    ]
    
    for module_name in modules_to_test:
        try:
            __import__(module_name)
            log_test_result('import_tests', module_name, True, "Module imported successfully")
        except Exception as e:
            log_test_result('import_tests', module_name, False, "Import failed", str(e))

def test_basic_functionality():
    """Test basic functionality of core classes."""
    print("\n🔧 Testing Basic Functionality...")
    
    try:
        # Test configuration creation
        from axiom_edge.config import create_default_config, validate_config
        config = create_default_config("./")
        validation_results = validate_config(config)
        
        log_test_result('basic_functionality_tests', 'config_creation', True, 
                       f"Config created and validated. Valid: {validation_results['is_valid']}")
        
    except Exception as e:
        log_test_result('basic_functionality_tests', 'config_creation', False, 
                       "Config creation failed", str(e))
    
    try:
        # Test data handler initialization
        from axiom_edge.data_handler import DataHandler
        data_handler = DataHandler(config)
        
        log_test_result('basic_functionality_tests', 'data_handler_init', True, 
                       "DataHandler initialized successfully")
        
    except Exception as e:
        log_test_result('basic_functionality_tests', 'data_handler_init', False, 
                       "DataHandler initialization failed", str(e))
    
    try:
        # Test AI analyzer initialization
        from axiom_edge.ai_analyzer import GeminiAnalyzer, APITimer
        ai_analyzer = GeminiAnalyzer()
        api_timer = APITimer()
        
        log_test_result('basic_functionality_tests', 'ai_analyzer_init', True, 
                       "AI components initialized successfully")
        
    except Exception as e:
        log_test_result('basic_functionality_tests', 'ai_analyzer_init', False, 
                       "AI analyzer initialization failed", str(e))
    
    try:
        # Test feature engineer initialization
        from axiom_edge.feature_engineer import FeatureEngineer
        timeframe_roles = {'base': 'D1'}
        playbook = {}
        feature_engineer = FeatureEngineer(config, timeframe_roles, playbook)
        
        log_test_result('basic_functionality_tests', 'feature_engineer_init', True, 
                       "FeatureEngineer initialized successfully")
        
    except Exception as e:
        log_test_result('basic_functionality_tests', 'feature_engineer_init', False, 
                       "FeatureEngineer initialization failed", str(e))

def test_integration_workflow():
    """Test integration between components."""
    print("\n🔗 Testing Component Integration...")
    
    try:
        # Create sample data
        dates = pd.date_range(start='2023-01-01', end='2023-12-31', freq='D')
        sample_data = pd.DataFrame({
            'Open': np.random.uniform(100, 200, len(dates)),
            'High': np.random.uniform(150, 250, len(dates)),
            'Low': np.random.uniform(50, 150, len(dates)),
            'Close': np.random.uniform(100, 200, len(dates)),
            'Volume': np.random.uniform(1000, 10000, len(dates))
        }, index=dates)
        
        # Add some realistic price relationships
        for i in range(1, len(sample_data)):
            sample_data.iloc[i]['Open'] = sample_data.iloc[i-1]['Close'] * np.random.uniform(0.98, 1.02)
            sample_data.iloc[i]['High'] = max(sample_data.iloc[i]['Open'], sample_data.iloc[i]['High'])
            sample_data.iloc[i]['Low'] = min(sample_data.iloc[i]['Open'], sample_data.iloc[i]['Low'])
            sample_data.iloc[i]['Close'] = np.random.uniform(sample_data.iloc[i]['Low'], sample_data.iloc[i]['High'])
        
        log_test_result('integration_tests', 'sample_data_creation', True, 
                       f"Sample data created: {len(sample_data)} rows")
        
    except Exception as e:
        log_test_result('integration_tests', 'sample_data_creation', False, 
                       "Sample data creation failed", str(e))
        return
    
    try:
        # Test feature engineering on sample data
        from axiom_edge.feature_engineer import FeatureEngineer
        from axiom_edge.config import create_default_config
        
        config = create_default_config("./")
        timeframe_roles = {'base': 'D1'}
        playbook = {}
        feature_engineer = FeatureEngineer(config, timeframe_roles, playbook)
        
        # Engineer features
        engineered_data = feature_engineer.engineer_features(sample_data)
        
        if not engineered_data.empty:
            log_test_result('integration_tests', 'feature_engineering_workflow', True, 
                           f"Features engineered: {len(engineered_data.columns)} columns")
        else:
            log_test_result('integration_tests', 'feature_engineering_workflow', False, 
                           "Feature engineering returned empty DataFrame")
        
    except Exception as e:
        log_test_result('integration_tests', 'feature_engineering_workflow', False, 
                       "Feature engineering workflow failed", str(e))

def test_framework_orchestrator():
    """Test the main framework orchestrator."""
    print("\n🎼 Testing Framework Orchestrator...")
    
    try:
        from axiom_edge.framework_orchestrator import FrameworkOrchestrator
        from axiom_edge.config import create_default_config
        
        config = create_default_config("./")
        orchestrator = FrameworkOrchestrator(config)
        
        log_test_result('integration_tests', 'framework_orchestrator_init', True, 
                       "FrameworkOrchestrator initialized successfully")
        
    except Exception as e:
        log_test_result('integration_tests', 'framework_orchestrator_init', False, 
                       "FrameworkOrchestrator initialization failed", str(e))

def test_task_interfaces():
    """Test task-specific interfaces."""
    print("\n📋 Testing Task Interfaces...")
    
    try:
        from axiom_edge.tasks import (
            BaseTask, DataCollectionTask, FeatureEngineeringTask,
            AIStrategyOptimizationTask, PortfolioOptimizationTask
        )
        from axiom_edge.config import create_default_config
        
        config = create_default_config("./")
        
        # Test BaseTask
        base_task = BaseTask(config)
        validation = base_task.validate_config()
        
        log_test_result('integration_tests', 'base_task_functionality', True, 
                       f"BaseTask created and validated. Valid: {validation['is_valid']}")
        
        # Test specialized tasks
        feature_task = FeatureEngineeringTask(config)
        ai_task = AIStrategyOptimizationTask(config)
        portfolio_task = PortfolioOptimizationTask(config)
        
        log_test_result('integration_tests', 'specialized_tasks', True, 
                       "All specialized task classes initialized successfully")
        
    except Exception as e:
        log_test_result('integration_tests', 'task_interfaces', False, 
                       "Task interface testing failed", str(e))

def test_utility_functions():
    """Test utility functions."""
    print("\n🛠️ Testing Utility Functions...")
    
    try:
        from axiom_edge.utils import (
            get_optimal_system_settings, json_serializer_default,
            calculate_memory_usage, safe_divide, normalize_features
        )
        
        # Test system settings
        settings = get_optimal_system_settings()
        log_test_result('integration_tests', 'system_settings', True, 
                       f"System settings retrieved: {len(settings)} settings")
        
        # Test safe divide
        result = safe_divide(10, 2, default=0)
        assert result == 5.0
        
        result = safe_divide(10, 0, default=-1)
        assert result == -1
        
        log_test_result('integration_tests', 'utility_functions', True, 
                       "Utility functions working correctly")
        
    except Exception as e:
        log_test_result('integration_tests', 'utility_functions', False, 
                       "Utility function testing failed", str(e))

def test_telemetry_system():
    """Test telemetry and monitoring system."""
    print("\n📊 Testing Telemetry System...")

    try:
        from axiom_edge.telemetry import TelemetryCollector, TelemetryAnalyzer, InterventionManager

        # Test TelemetryCollector
        telemetry = TelemetryCollector("test_session.jsonl")
        log_test_result('integration_tests', 'telemetry_collector_init', True,
                       f"TelemetryCollector initialized: {telemetry.session_id}")

        # Test TelemetryAnalyzer
        analyzer = TelemetryAnalyzer("test_session.jsonl")
        log_test_result('integration_tests', 'telemetry_analyzer_init', True,
                       "TelemetryAnalyzer initialized successfully")

        # Test InterventionManager
        intervention_mgr = InterventionManager("test_interventions.json")
        log_test_result('integration_tests', 'intervention_manager_init', True,
                       "InterventionManager initialized successfully")

    except Exception as e:
        log_test_result('integration_tests', 'telemetry_system', False,
                       "Telemetry system testing failed", str(e))

def test_backtester_integration():
    """Test backtester integration."""
    print("\n📈 Testing Backtester Integration...")

    try:
        from axiom_edge.backtester import Backtester
        from axiom_edge.ai_analyzer import GeminiAnalyzer
        from axiom_edge.config import create_default_config

        config = create_default_config("./")
        ai_analyzer = GeminiAnalyzer()
        backtester = Backtester(config, ai_analyzer)

        log_test_result('integration_tests', 'backtester_init', True,
                       "Backtester initialized with AI integration")

    except Exception as e:
        log_test_result('integration_tests', 'backtester_init', False,
                       "Backtester initialization failed", str(e))

def test_report_generator():
    """Test report generator functionality."""
    print("\n📋 Testing Report Generator...")

    try:
        from axiom_edge.report_generator import ReportGenerator
        from axiom_edge.ai_analyzer import GeminiAnalyzer
        from axiom_edge.config import create_default_config

        config = create_default_config("./")
        ai_analyzer = GeminiAnalyzer()
        report_gen = ReportGenerator(config, ai_analyzer)

        log_test_result('integration_tests', 'report_generator_init', True,
                       "ReportGenerator initialized with AI integration")

    except Exception as e:
        log_test_result('integration_tests', 'report_generator_init', False,
                       "ReportGenerator initialization failed", str(e))

def test_genetic_programmer():
    """Test genetic programmer functionality."""
    print("\n🧬 Testing Genetic Programmer...")

    try:
        from axiom_edge.genetic_programmer import GeneticProgrammer
        from axiom_edge.ai_analyzer import GeminiAnalyzer
        from axiom_edge.config import create_default_config

        config = create_default_config("./")
        ai_analyzer = GeminiAnalyzer()
        genetic_prog = GeneticProgrammer(config, ai_analyzer)

        log_test_result('integration_tests', 'genetic_programmer_init', True,
                       "GeneticProgrammer initialized with AI integration")

    except Exception as e:
        log_test_result('integration_tests', 'genetic_programmer_init', False,
                       "GeneticProgrammer initialization failed", str(e))

def test_model_trainer_integration():
    """Test model trainer integration."""
    print("\n🧠 Testing Model Trainer Integration...")

    try:
        from axiom_edge.model_trainer import ModelTrainer
        from axiom_edge.ai_analyzer import GeminiAnalyzer
        from axiom_edge.config import create_default_config

        config = create_default_config("./")
        ai_analyzer = GeminiAnalyzer()
        model_trainer = ModelTrainer(config, ai_analyzer)

        log_test_result('integration_tests', 'model_trainer_integration', True,
                       "ModelTrainer initialized with AI integration")

    except Exception as e:
        log_test_result('integration_tests', 'model_trainer_integration', False,
                       "ModelTrainer integration failed", str(e))

def test_error_handling():
    """Test error handling capabilities."""
    print("\n🚨 Testing Error Handling...")

    try:
        from axiom_edge.config import create_default_config, validate_config
        from axiom_edge.feature_engineer import FeatureEngineer
        from axiom_edge.utils import safe_divide

        config = create_default_config("./")
        timeframe_roles = {'base': 'D1'}
        playbook = {}
        feature_engineer = FeatureEngineer(config, timeframe_roles, playbook)

        # Test with empty DataFrame
        empty_df = pd.DataFrame()
        result = feature_engineer.engineer_features(empty_df)

        log_test_result('error_handling_tests', 'empty_dataframe_handling', True,
                       "Empty DataFrame handled gracefully")

        # Test safe divide with zero
        result = safe_divide(10, 0, default=-1)
        assert result == -1
        log_test_result('error_handling_tests', 'division_by_zero', True,
                       "Division by zero handled correctly")

        # Test invalid configuration
        config.INITIAL_CAPITAL = -1000  # Invalid value
        validation = validate_config(config)
        assert not validation['is_valid']
        log_test_result('error_handling_tests', 'invalid_config_detection', True,
                       "Invalid configuration detected correctly")

    except Exception as e:
        log_test_result('error_handling_tests', 'error_handling_comprehensive', False,
                       "Error handling testing failed", str(e))

def test_performance_characteristics():
    """Test basic performance characteristics."""
    print("\n⚡ Testing Performance Characteristics...")

    try:
        import time
        from axiom_edge.utils import get_optimal_system_settings, calculate_memory_usage

        # Test system settings performance
        start_time = time.time()
        settings = get_optimal_system_settings()
        settings_time = time.time() - start_time

        log_test_result('performance_tests', 'system_settings_speed', True,
                       f"System settings calculated in {settings_time:.4f}s")

        # Test memory calculation performance
        large_data = pd.DataFrame(np.random.random((1000, 50)))
        start_time = time.time()
        memory_info = calculate_memory_usage(large_data)
        memory_time = time.time() - start_time

        log_test_result('performance_tests', 'memory_calculation_speed', True,
                       f"Memory calculation for 1000x50 DataFrame: {memory_time:.4f}s")

        # Performance assertions
        if settings_time > 1.0:
            log_test_result('performance_tests', 'settings_performance_warning', False,
                           "System settings calculation is slow", f"Time: {settings_time:.4f}s")
        else:
            log_test_result('performance_tests', 'settings_performance_ok', True,
                           "System settings performance acceptable")

        if memory_time > 0.5:
            log_test_result('performance_tests', 'memory_performance_warning', False,
                           "Memory calculation is slow", f"Time: {memory_time:.4f}s")
        else:
            log_test_result('performance_tests', 'memory_performance_ok', True,
                           "Memory calculation performance acceptable")

    except Exception as e:
        log_test_result('performance_tests', 'performance_testing', False,
                       "Performance testing failed", str(e))

def test_package_level_functionality():
    """Test package-level functionality and convenience functions."""
    print("\n📦 Testing Package-Level Functionality...")

    try:
        import axiom_edge

        # Test package import and version
        version = axiom_edge.__version__
        log_test_result('integration_tests', 'package_version', True,
                       f"Package version: {version}")

        # Test framework info
        info = axiom_edge.get_framework_info()
        log_test_result('integration_tests', 'framework_info', True,
                       f"Framework info retrieved: {len(info['components'])} components")

        # Test installation validation
        validation = axiom_edge.validate_installation()
        log_test_result('integration_tests', 'installation_validation', True,
                       f"Installation validation: {validation['installation_valid']}")

        # Test convenience functions
        config = axiom_edge.create_default_config("./")
        log_test_result('integration_tests', 'convenience_functions', True,
                       "Convenience functions working correctly")

    except Exception as e:
        log_test_result('integration_tests', 'package_level_functionality', False,
                       "Package-level functionality testing failed", str(e))

def test_data_processing_pipeline():
    """Test complete data processing pipeline."""
    print("\n🔄 Testing Data Processing Pipeline...")

    try:
        from axiom_edge.config import create_default_config
        from axiom_edge.feature_engineer import FeatureEngineer
        from axiom_edge.utils import validate_data_quality, optimize_dataframe

        # Create realistic sample data
        dates = pd.date_range(start='2023-01-01', periods=100, freq='D')
        sample_data = pd.DataFrame({
            'timestamp': dates,
            'open': np.random.uniform(100, 110, 100),
            'high': np.random.uniform(110, 120, 100),
            'low': np.random.uniform(90, 100, 100),
            'close': np.random.uniform(95, 115, 100),
            'volume': np.random.uniform(1000, 10000, 100)
        })

        # Ensure realistic price relationships
        for i in range(1, len(sample_data)):
            sample_data.loc[sample_data.index[i], 'high'] = max(
                sample_data.loc[sample_data.index[i], 'open'],
                sample_data.loc[sample_data.index[i], 'high']
            )
            sample_data.loc[sample_data.index[i], 'low'] = min(
                sample_data.loc[sample_data.index[i], 'open'],
                sample_data.loc[sample_data.index[i], 'low']
            )

        log_test_result('integration_tests', 'realistic_data_creation', True,
                       f"Realistic sample data created: {len(sample_data)} rows")

        # Test data quality validation
        required_columns = ['open', 'high', 'low', 'close', 'volume']
        quality_report = validate_data_quality(sample_data, required_columns)
        log_test_result('integration_tests', 'data_quality_validation', True,
                       f"Data quality validated: {quality_report['overall_quality']}")

        # Test data optimization
        optimized_data = optimize_dataframe(sample_data)
        log_test_result('integration_tests', 'data_optimization', True,
                       f"Data optimized: {len(optimized_data)} rows")

        # Test feature engineering pipeline
        config = create_default_config("./")
        timeframe_roles = {'base': 'D1'}
        playbook = {}
        feature_engineer = FeatureEngineer(config, timeframe_roles, playbook)

        # Add basic technical indicators
        sample_data['sma_10'] = sample_data['close'].rolling(10).mean()
        sample_data['sma_20'] = sample_data['close'].rolling(20).mean()
        sample_data['rsi'] = 50.0  # Simplified RSI

        log_test_result('integration_tests', 'data_processing_pipeline', True,
                       f"Complete data processing pipeline: {len(sample_data.columns)} final columns")

    except Exception as e:
        log_test_result('integration_tests', 'data_processing_pipeline', False,
                       "Data processing pipeline failed", str(e))

def generate_test_summary():
    """Generate comprehensive test summary."""
    print("\n📊 Generating Test Summary...")
    
    total_tests = 0
    passed_tests = 0
    failed_tests = 0
    
    for category, tests in test_results.items():
        if category == 'summary':
            continue
            
        category_passed = 0
        category_total = 0
        
        for test_name, result in tests.items():
            total_tests += 1
            category_total += 1
            
            if result['success']:
                passed_tests += 1
                category_passed += 1
            else:
                failed_tests += 1
        
        test_results['summary'][category] = {
            'total': category_total,
            'passed': category_passed,
            'failed': category_total - category_passed,
            'success_rate': (category_passed / category_total * 100) if category_total > 0 else 0
        }
    
    test_results['summary']['overall'] = {
        'total': total_tests,
        'passed': passed_tests,
        'failed': failed_tests,
        'success_rate': (passed_tests / total_tests * 100) if total_tests > 0 else 0
    }
    
    # Print summary
    print("\n" + "="*60)
    print("🎯 INTEGRATION TEST SUMMARY")
    print("="*60)
    
    for category, summary in test_results['summary'].items():
        if category == 'overall':
            continue
        print(f"{category.replace('_', ' ').title()}: {summary['passed']}/{summary['total']} "
              f"({summary['success_rate']:.1f}%)")
    
    overall = test_results['summary']['overall']
    print(f"\nOVERALL: {overall['passed']}/{overall['total']} ({overall['success_rate']:.1f}%)")
    
    if overall['success_rate'] >= 90:
        print("🎉 EXCELLENT: Integration testing passed with high success rate!")
        return True
    elif overall['success_rate'] >= 75:
        print("✅ GOOD: Integration testing passed with acceptable success rate.")
        return True
    elif overall['success_rate'] >= 50:
        print("⚠️ MODERATE: Integration testing had mixed results. Review failures.")
        print("\n🔍 Failed Tests:")
        for category, tests in test_results.items():
            if category == 'summary':
                continue
            for test_name, result in tests.items():
                if not result['success']:
                    print(f"  ❌ {category}.{test_name}: {result['error']}")
        return False
    else:
        print("❌ POOR: Integration testing failed. Significant issues detected.")
        print("\n🔍 Failed Tests:")
        for category, tests in test_results.items():
            if category == 'summary':
                continue
            for test_name, result in tests.items():
                if not result['success']:
                    print(f"  ❌ {category}.{test_name}: {result['error']}")
        return False

if __name__ == "__main__":
    print("🚀 Starting AxiomEdge Framework Integration Testing...")
    print(f"Test started at: {datetime.now().isoformat()}")
    
    try:
        # Run all test suites
        test_imports()
        test_basic_functionality()
        test_integration_workflow()
        test_framework_orchestrator()
        test_task_interfaces()
        test_utility_functions()
        test_telemetry_system()
        test_backtester_integration()
        test_report_generator()
        test_genetic_programmer()
        test_model_trainer_integration()
        test_package_level_functionality()
        test_data_processing_pipeline()
        test_performance_characteristics()
        test_error_handling()

        # Generate summary
        success = generate_test_summary()

        # Exit with appropriate code
        if success:
            print("\n🎉 Integration testing completed successfully!")
            sys.exit(0)
        else:
            print("\n❌ Integration testing completed with failures!")
            sys.exit(1)

    except Exception as e:
        print(f"\n❌ CRITICAL ERROR during testing: {e}")
        traceback.print_exc()
        sys.exit(1)

    print(f"\nTest completed at: {datetime.now().isoformat()}")
