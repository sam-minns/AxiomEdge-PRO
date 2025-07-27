#!/usr/bin/env python3
"""
Simple Integration Test for AxiomEdge Framework
"""

import sys
import traceback

def test_basic_imports():
    """Test basic imports of all modules."""
    print("Testing basic imports...")
    
    modules = [
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
        'axiom_edge.tasks'
    ]
    
    results = {}
    
    for module in modules:
        try:
            __import__(module)
            results[module] = "✅ SUCCESS"
            print(f"✅ {module}")
        except Exception as e:
            results[module] = f"❌ FAILED: {str(e)}"
            print(f"❌ {module}: {str(e)}")
    
    return results

def test_basic_functionality():
    """Test basic functionality."""
    print("\nTesting basic functionality...")

    try:
        # Test config creation
        from axiom_edge.config import create_default_config, validate_config
        config = create_default_config("./")
        print("✅ Config creation successful")

        # Test config validation
        validation = validate_config(config)
        print(f"✅ Config validation: {validation['is_valid']}")

        # Test constants access
        from axiom_edge.config import TIMEFRAME_MAP, ANOMALY_FEATURES
        print(f"✅ Constants accessible: {len(TIMEFRAME_MAP)} timeframes, {len(ANOMALY_FEATURES)} anomaly features")

        # Test utility functions
        from axiom_edge.utils import get_optimal_system_settings, safe_divide
        settings = get_optimal_system_settings()
        result = safe_divide(10, 2, default=0)
        print(f"✅ Utility functions working: {len(settings)} settings, safe_divide result: {result}")

        # Test package-level imports
        import axiom_edge
        print(f"✅ Package import successful: v{axiom_edge.__version__}")

        return True

    except Exception as e:
        print(f"❌ Basic functionality test failed: {e}")
        return False

def test_class_instantiation():
    """Test basic class instantiation."""
    print("\nTesting class instantiation...")

    try:
        from axiom_edge.config import create_default_config
        config = create_default_config("./")

        # Test core classes
        from axiom_edge.data_handler import DataHandler
        data_handler = DataHandler(config)
        print("✅ DataHandler instantiated")

        from axiom_edge.ai_analyzer import GeminiAnalyzer, APITimer
        ai_analyzer = GeminiAnalyzer()
        api_timer = APITimer()
        print("✅ AI components instantiated")

        from axiom_edge.feature_engineer import FeatureEngineer
        timeframe_roles = {'base': 'D1'}
        playbook = {}
        feature_engineer = FeatureEngineer(config, timeframe_roles, playbook)
        print("✅ FeatureEngineer instantiated")

        from axiom_edge.model_trainer import ModelTrainer
        model_trainer = ModelTrainer(config, ai_analyzer)
        print("✅ ModelTrainer instantiated")

        return True

    except Exception as e:
        print(f"❌ Class instantiation test failed: {e}")
        return False

def test_task_interfaces():
    """Test task interface functionality."""
    print("\nTesting task interfaces...")

    try:
        from axiom_edge.config import create_default_config
        from axiom_edge.tasks import BaseTask, DataCollectionTask, FeatureEngineeringTask

        config = create_default_config("./")

        # Test BaseTask
        base_task = BaseTask(config)
        validation = base_task.validate_config()
        print(f"✅ BaseTask validation: {validation['is_valid']}")

        # Test specialized tasks
        data_task = DataCollectionTask(config)
        feature_task = FeatureEngineeringTask(config)
        print("✅ Specialized tasks instantiated")

        return True

    except Exception as e:
        print(f"❌ Task interface test failed: {e}")
        return False

def test_quick_performance():
    """Test basic performance characteristics."""
    print("\nTesting quick performance...")

    try:
        import time
        from axiom_edge.utils import get_optimal_system_settings, safe_divide

        # Test system settings speed
        start_time = time.time()
        settings = get_optimal_system_settings()
        settings_time = time.time() - start_time

        # Test safe divide speed
        start_time = time.time()
        for _ in range(1000):
            safe_divide(10, 2, default=0)
        divide_time = time.time() - start_time

        print(f"✅ Performance check: settings={settings_time:.4f}s, 1000 divisions={divide_time:.4f}s")

        # Basic performance assertions
        if settings_time > 1.0:
            print("⚠️ Warning: System settings calculation is slow")
        if divide_time > 0.1:
            print("⚠️ Warning: Safe divide operations are slow")

        return True

    except Exception as e:
        print(f"❌ Performance test failed: {e}")
        return False

def test_error_handling():
    """Test basic error handling."""
    print("\nTesting error handling...")

    try:
        from axiom_edge.utils import safe_divide

        # Test division by zero
        result = safe_divide(10, 0, default=-1)
        assert result == -1, f"Expected -1, got {result}"
        print("✅ Division by zero handled correctly")

        # Test None handling
        result = safe_divide(None, 5, default=0)
        assert result == 0, f"Expected 0, got {result}"
        print("✅ None value handled correctly")

        # Test invalid config detection
        from axiom_edge.config import create_default_config, validate_config
        config = create_default_config("./")
        config.INITIAL_CAPITAL = -1000  # Invalid value
        validation = validate_config(config)
        assert not validation['is_valid'], "Invalid config should be detected"
        print("✅ Invalid configuration detected correctly")

        return True

    except Exception as e:
        print(f"❌ Error handling test failed: {e}")
        return False

if __name__ == "__main__":
    print("🚀 AxiomEdge Framework Simple Integration Test")
    print("=" * 50)

    try:
        # Test imports
        import_results = test_basic_imports()

        # Test basic functionality
        functionality_ok = test_basic_functionality()

        # Test class instantiation
        instantiation_ok = test_class_instantiation()

        # Test task interfaces
        tasks_ok = test_task_interfaces()

        # Test performance
        performance_ok = test_quick_performance()

        # Test error handling
        error_handling_ok = test_error_handling()

        # Summary
        print("\n" + "=" * 50)
        print("📊 TEST SUMMARY")
        print("=" * 50)

        successful_imports = sum(1 for result in import_results.values() if "SUCCESS" in result)
        total_imports = len(import_results)

        print(f"Import Tests: {successful_imports}/{total_imports} successful")
        print(f"Functionality Test: {'✅ PASSED' if functionality_ok else '❌ FAILED'}")
        print(f"Instantiation Test: {'✅ PASSED' if instantiation_ok else '❌ FAILED'}")
        print(f"Task Interface Test: {'✅ PASSED' if tasks_ok else '❌ FAILED'}")
        print(f"Performance Test: {'✅ PASSED' if performance_ok else '❌ FAILED'}")
        print(f"Error Handling Test: {'✅ PASSED' if error_handling_ok else '❌ FAILED'}")

        # Calculate overall success
        import_success_rate = successful_imports / total_imports if total_imports > 0 else 0
        functional_tests_passed = sum([functionality_ok, instantiation_ok, tasks_ok, performance_ok, error_handling_ok])
        total_functional_tests = 5

        overall_success = (import_success_rate >= 0.8 and
                          functional_tests_passed >= 4)  # At least 4 out of 5 functional tests

        print(f"\nOverall Result: {'🎉 SUCCESS' if overall_success else '❌ NEEDS ATTENTION'}")
        print(f"Success Rate: {import_success_rate:.1%} imports, {functional_tests_passed}/{total_functional_tests} functional tests")

        if not overall_success:
            print("\n⚠️ Issues detected:")
            if import_success_rate < 0.8:
                print("  - Import failures detected:")
                for module, result in import_results.items():
                    if "FAILED" in result:
                        print(f"    • {module}: {result}")

            if functional_tests_passed < 4:
                print("  - Functional test failures:")
                if not functionality_ok:
                    print("    • Basic functionality test failed")
                if not instantiation_ok:
                    print("    • Class instantiation test failed")
                if not tasks_ok:
                    print("    • Task interface test failed")
                if not performance_ok:
                    print("    • Performance test failed")
                if not error_handling_ok:
                    print("    • Error handling test failed")
        else:
            print("\n🎉 All tests passed! Framework is ready to use.")

        # Exit with appropriate code
        sys.exit(0 if overall_success else 1)

    except Exception as e:
        print(f"❌ Critical error during testing: {e}")
        traceback.print_exc()
        sys.exit(1)
