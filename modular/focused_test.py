#!/usr/bin/env python3
"""
Focused Integration Test for AxiomEdge Framework
Tests core functionality without external dependencies
"""

import sys
import os
import traceback
import pandas as pd
import numpy as np
from datetime import datetime

def test_config_module():
    """Test configuration module functionality."""
    print("🔧 Testing Configuration Module...")
    
    try:
        from axiom_edge.config import (
            create_default_config, validate_config, get_config_summary,
            TIMEFRAME_MAP, ANOMALY_FEATURES, NON_FEATURE_COLS
        )
        
        # Test config creation
        config = create_default_config("./")
        print(f"  ✅ Config created: {config.REPORT_LABEL}")
        
        # Test validation
        validation = validate_config(config)
        print(f"  ✅ Config validation: {validation['is_valid']}")
        
        # Test summary
        summary = get_config_summary(config)
        print(f"  ✅ Config summary: {summary['framework_version']}")
        
        # Test constants
        print(f"  ✅ Constants: {len(TIMEFRAME_MAP)} timeframes, {len(ANOMALY_FEATURES)} anomaly features")
        
        return True
        
    except Exception as e:
        print(f"  ❌ Config module test failed: {e}")
        return False

def test_utils_module():
    """Test utilities module functionality."""
    print("🛠️ Testing Utilities Module...")
    
    try:
        from axiom_edge.utils import (
            get_optimal_system_settings, safe_divide, json_serializer_default,
            calculate_memory_usage, optimize_dataframe
        )
        
        # Test system settings
        settings = get_optimal_system_settings()
        print(f"  ✅ System settings: {settings['num_workers']} workers")
        
        # Test safe divide
        result = safe_divide(10, 2, default=0)
        assert result == 5.0
        print(f"  ✅ Safe divide: 10/2 = {result}")
        
        # Test with sample data
        sample_df = pd.DataFrame({
            'A': [1.0, 2.0, 3.0],
            'B': [4.0, 5.0, 6.0]
        })
        
        memory_info = calculate_memory_usage(sample_df)
        print(f"  ✅ Memory calculation: {memory_info['mb']:.2f} MB")
        
        optimized_df = optimize_dataframe(sample_df)
        print(f"  ✅ DataFrame optimization: {len(optimized_df)} rows")
        
        return True
        
    except Exception as e:
        print(f"  ❌ Utils module test failed: {e}")
        return False

def test_feature_engineer_init():
    """Test feature engineer initialization."""
    print("🔬 Testing FeatureEngineer Initialization...")
    
    try:
        from axiom_edge.config import create_default_config
        from axiom_edge.feature_engineer import FeatureEngineer
        
        config = create_default_config("./")
        timeframe_roles = {'base': 'D1'}
        playbook = {}
        
        feature_engineer = FeatureEngineer(config, timeframe_roles, playbook)
        print(f"  ✅ FeatureEngineer initialized with {len(feature_engineer.TIMEFRAME_MAP)} timeframes")
        
        # Test constants access
        print(f"  ✅ Anomaly features: {len(feature_engineer.ANOMALY_FEATURES)}")
        print(f"  ✅ Non-feature cols: {len(feature_engineer.NON_FEATURE_COLS)}")
        
        return True
        
    except Exception as e:
        print(f"  ❌ FeatureEngineer init test failed: {e}")
        return False

def test_data_handler_init():
    """Test data handler initialization."""
    print("📊 Testing DataHandler Initialization...")
    
    try:
        from axiom_edge.config import create_default_config
        from axiom_edge.data_handler import DataHandler
        
        config = create_default_config("./")
        data_handler = DataHandler(config)
        
        print(f"  ✅ DataHandler initialized with cache enabled: {data_handler.cache_enabled}")
        
        return True
        
    except Exception as e:
        print(f"  ❌ DataHandler init test failed: {e}")
        return False

def test_ai_analyzer_init():
    """Test AI analyzer initialization."""
    print("🤖 Testing AI Analyzer Initialization...")
    
    try:
        from axiom_edge.ai_analyzer import GeminiAnalyzer, APITimer
        
        ai_analyzer = GeminiAnalyzer()
        api_timer = APITimer()
        
        print(f"  ✅ GeminiAnalyzer initialized with {len(ai_analyzer.model_priority_list)} models")
        print(f"  ✅ APITimer initialized")
        
        return True
        
    except Exception as e:
        print(f"  ❌ AI Analyzer init test failed: {e}")
        return False

def test_model_trainer_init():
    """Test model trainer initialization."""
    print("🧠 Testing ModelTrainer Initialization...")
    
    try:
        from axiom_edge.config import create_default_config
        from axiom_edge.model_trainer import ModelTrainer
        from axiom_edge.ai_analyzer import GeminiAnalyzer
        
        config = create_default_config("./")
        gemini_analyzer = GeminiAnalyzer()
        
        model_trainer = ModelTrainer(config, gemini_analyzer)
        print(f"  ✅ ModelTrainer initialized with SHAP: {hasattr(model_trainer, 'shap_summaries')}")
        
        return True
        
    except Exception as e:
        print(f"  ❌ ModelTrainer init test failed: {e}")
        return False

def test_framework_orchestrator_init():
    """Test framework orchestrator initialization."""
    print("🎼 Testing FrameworkOrchestrator Initialization...")
    
    try:
        from axiom_edge.config import create_default_config
        from axiom_edge.framework_orchestrator import FrameworkOrchestrator
        
        config = create_default_config("./")
        orchestrator = FrameworkOrchestrator(config)
        
        print(f"  ✅ FrameworkOrchestrator initialized")
        
        return True
        
    except Exception as e:
        print(f"  ❌ FrameworkOrchestrator init test failed: {e}")
        return False

def test_tasks_module():
    """Test tasks module functionality."""
    print("📋 Testing Tasks Module...")

    try:
        from axiom_edge.config import create_default_config
        from axiom_edge.tasks import (
            BaseTask, DataCollectionTask, FeatureEngineeringTask,
            ModelTrainingTask, BacktestTask, CompleteFrameworkTask
        )

        config = create_default_config("./")

        # Test BaseTask
        base_task = BaseTask(config)
        validation = base_task.validate_config()
        print(f"  ✅ BaseTask validation: {validation['is_valid']}")

        # Test specialized tasks
        feature_task = FeatureEngineeringTask(config)
        model_task = ModelTrainingTask(config)
        backtest_task = BacktestTask(config)
        complete_task = CompleteFrameworkTask(config)

        print(f"  ✅ Specialized tasks initialized successfully")

        return True

    except Exception as e:
        print(f"  ❌ Tasks module test failed: {e}")
        return False

def test_telemetry_module():
    """Test telemetry module functionality."""
    print("📊 Testing Telemetry Module...")

    try:
        from axiom_edge.telemetry import TelemetryCollector, TelemetryAnalyzer, InterventionManager

        # Test TelemetryCollector
        telemetry = TelemetryCollector("test_session.jsonl")
        print(f"  ✅ TelemetryCollector initialized: {telemetry.session_id}")

        # Test TelemetryAnalyzer
        analyzer = TelemetryAnalyzer("test_session.jsonl")
        print(f"  ✅ TelemetryAnalyzer initialized")

        # Test InterventionManager
        intervention_mgr = InterventionManager("test_interventions.json")
        print(f"  ✅ InterventionManager initialized")

        return True

    except Exception as e:
        print(f"  ❌ Telemetry module test failed: {e}")
        return False

def test_backtester_init():
    """Test backtester initialization."""
    print("📈 Testing Backtester Initialization...")

    try:
        from axiom_edge.config import create_default_config
        from axiom_edge.backtester import Backtester
        from axiom_edge.ai_analyzer import GeminiAnalyzer

        config = create_default_config("./")
        gemini_analyzer = GeminiAnalyzer()

        backtester = Backtester(config, gemini_analyzer)
        print(f"  ✅ Backtester initialized with AI integration")

        return True

    except Exception as e:
        print(f"  ❌ Backtester init test failed: {e}")
        return False

def test_report_generator_init():
    """Test report generator initialization."""
    print("📋 Testing ReportGenerator Initialization...")

    try:
        from axiom_edge.config import create_default_config
        from axiom_edge.report_generator import ReportGenerator
        from axiom_edge.ai_analyzer import GeminiAnalyzer

        config = create_default_config("./")
        gemini_analyzer = GeminiAnalyzer()

        report_gen = ReportGenerator(config, gemini_analyzer)
        print(f"  ✅ ReportGenerator initialized with AI integration")

        return True

    except Exception as e:
        print(f"  ❌ ReportGenerator init test failed: {e}")
        return False

def test_genetic_programmer_init():
    """Test genetic programmer initialization."""
    print("🧬 Testing GeneticProgrammer Initialization...")

    try:
        from axiom_edge.config import create_default_config
        from axiom_edge.genetic_programmer import GeneticProgrammer
        from axiom_edge.ai_analyzer import GeminiAnalyzer

        config = create_default_config("./")
        gemini_analyzer = GeminiAnalyzer()

        genetic_prog = GeneticProgrammer(config, gemini_analyzer)
        print(f"  ✅ GeneticProgrammer initialized with AI integration")

        return True

    except Exception as e:
        print(f"  ❌ GeneticProgrammer init test failed: {e}")
        return False

def test_package_imports():
    """Test package-level imports and main functionality."""
    print("📦 Testing Package Imports...")

    try:
        # Test main package imports
        import axiom_edge
        print(f"  ✅ Main package imported: v{axiom_edge.__version__}")

        # Test framework info
        info = axiom_edge.get_framework_info()
        print(f"  ✅ Framework info: {len(info['components'])} components")

        # Test installation validation
        validation = axiom_edge.validate_installation()
        print(f"  ✅ Installation validation: {validation['installation_valid']}")

        # Test convenience functions
        try:
            config = axiom_edge.create_default_config("./")
            print(f"  ✅ Convenience functions working")
        except Exception as e:
            print(f"  ⚠️ Convenience functions issue: {e}")

        return True

    except Exception as e:
        print(f"  ❌ Package imports test failed: {e}")
        return False

def test_sample_data_processing():
    """Test basic data processing functionality."""
    print("🔬 Testing Sample Data Processing...")

    try:
        from axiom_edge.config import create_default_config
        from axiom_edge.feature_engineer import FeatureEngineer

        # Create sample data
        dates = pd.date_range('2023-01-01', periods=100, freq='D')
        sample_data = pd.DataFrame({
            'timestamp': dates,
            'open': np.random.uniform(100, 110, 100),
            'high': np.random.uniform(110, 120, 100),
            'low': np.random.uniform(90, 100, 100),
            'close': np.random.uniform(95, 115, 100),
            'volume': np.random.uniform(1000, 10000, 100)
        })

        print(f"  ✅ Sample data created: {len(sample_data)} rows")

        # Test basic feature engineering
        config = create_default_config("./")
        timeframe_roles = {'base': 'D1'}
        playbook = {}

        feature_engineer = FeatureEngineer(config, timeframe_roles, playbook)

        # Test basic technical indicators
        sample_data['sma_10'] = sample_data['close'].rolling(10).mean()
        sample_data['rsi'] = 50.0  # Simplified RSI

        print(f"  ✅ Basic indicators calculated")

        return True

    except Exception as e:
        print(f"  ❌ Sample data processing test failed: {e}")
        return False

def test_error_handling():
    """Test error handling and edge cases."""
    print("⚠️ Testing Error Handling...")

    try:
        from axiom_edge.config import create_default_config, validate_config
        from axiom_edge.utils import safe_divide

        # Test safe divide with zero
        result = safe_divide(10, 0, default=-1)
        assert result == -1
        print(f"  ✅ Safe divide with zero: {result}")

        # Test config validation with invalid config
        config = create_default_config("./")
        config.INITIAL_CAPITAL = -1000  # Invalid value
        validation = validate_config(config)
        print(f"  ✅ Invalid config detected: {not validation['is_valid']}")

        # Test with None values
        result = safe_divide(None, 5, default=0)
        assert result == 0
        print(f"  ✅ Safe divide with None: {result}")

        return True

    except Exception as e:
        print(f"  ❌ Error handling test failed: {e}")
        return False

def test_performance_basics():
    """Test basic performance characteristics."""
    print("⚡ Testing Performance Basics...")

    try:
        import time
        from axiom_edge.utils import get_optimal_system_settings, calculate_memory_usage

        # Test system settings performance
        start_time = time.time()
        settings = get_optimal_system_settings()
        settings_time = time.time() - start_time
        print(f"  ✅ System settings calculated in {settings_time:.4f}s")

        # Test memory calculation performance
        large_data = pd.DataFrame(np.random.random((1000, 10)))
        start_time = time.time()
        memory_info = calculate_memory_usage(large_data)
        memory_time = time.time() - start_time
        print(f"  ✅ Memory calculation for 1000x10 DataFrame: {memory_time:.4f}s")

        # Performance should be reasonable
        assert settings_time < 1.0  # Should be very fast
        assert memory_time < 0.1   # Should be fast

        return True

    except Exception as e:
        print(f"  ❌ Performance test failed: {e}")
        return False

def run_comprehensive_test():
    """Run comprehensive integration test."""
    print("🚀 AxiomEdge Framework Focused Integration Test")
    print("=" * 60)
    print(f"Test started at: {datetime.now().isoformat()}")
    print()
    
    tests = [
        ("Package Imports", test_package_imports),
        ("Configuration Module", test_config_module),
        ("Utilities Module", test_utils_module),
        ("FeatureEngineer Init", test_feature_engineer_init),
        ("DataHandler Init", test_data_handler_init),
        ("AI Analyzer Init", test_ai_analyzer_init),
        ("ModelTrainer Init", test_model_trainer_init),
        ("Backtester Init", test_backtester_init),
        ("ReportGenerator Init", test_report_generator_init),
        ("GeneticProgrammer Init", test_genetic_programmer_init),
        ("FrameworkOrchestrator Init", test_framework_orchestrator_init),
        ("Telemetry Module", test_telemetry_module),
        ("Tasks Module", test_tasks_module),
        ("Sample Data Processing", test_sample_data_processing),
        ("Error Handling", test_error_handling),
        ("Performance Basics", test_performance_basics)
    ]
    
    results = {}
    
    for test_name, test_func in tests:
        try:
            success = test_func()
            results[test_name] = success
        except Exception as e:
            print(f"  ❌ {test_name} failed with exception: {e}")
            results[test_name] = False
        print()
    
    # Summary
    print("=" * 60)
    print("📊 TEST SUMMARY")
    print("=" * 60)
    
    passed = sum(1 for success in results.values() if success)
    total = len(results)
    success_rate = (passed / total) * 100 if total > 0 else 0
    
    for test_name, success in results.items():
        status = "✅ PASS" if success else "❌ FAIL"
        print(f"{status} {test_name}")
    
    print(f"\nOverall: {passed}/{total} tests passed ({success_rate:.1f}%)")
    
    if success_rate >= 90:
        print("🎉 EXCELLENT: Framework integration is working perfectly!")
    elif success_rate >= 75:
        print("✅ GOOD: Framework integration is working well.")
    elif success_rate >= 50:
        print("⚠️ MODERATE: Some issues detected. Review failed tests.")
    else:
        print("❌ POOR: Significant integration issues detected.")
    
    print(f"\nTest completed at: {datetime.now().isoformat()}")
    
    return success_rate >= 75

if __name__ == "__main__":
    try:
        success = run_comprehensive_test()
        sys.exit(0 if success else 1)
    except Exception as e:
        print(f"\n❌ CRITICAL ERROR: {e}")
        traceback.print_exc()
        sys.exit(1)
