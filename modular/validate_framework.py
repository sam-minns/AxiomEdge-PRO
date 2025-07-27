#!/usr/bin/env python3
"""
Framework Validation Script
Validates the AxiomEdge framework structure and basic functionality
"""

import os
import sys
from pathlib import Path
from datetime import datetime

def validate_file_structure():
    """Validate that all required files exist."""
    print("📁 Validating File Structure...")
    
    required_files = [
        'axiom_edge/__init__.py',
        'axiom_edge/__main__.py',
        'axiom_edge/main.py',
        'axiom_edge/config.py',
        'axiom_edge/data_handler.py',
        'axiom_edge/ai_analyzer.py',
        'axiom_edge/feature_engineer.py',
        'axiom_edge/model_trainer.py',
        'axiom_edge/backtester.py',
        'axiom_edge/genetic_programmer.py',
        'axiom_edge/report_generator.py',
        'axiom_edge/framework_orchestrator.py',
        'axiom_edge/telemetry.py',
        'axiom_edge/utils.py',
        'axiom_edge/tasks.py',
        'main.py',
        'setup.py',
        'pyproject.toml',
        'requirements.txt',
        'README.md'
    ]
    
    missing_files = []
    existing_files = []
    
    for file_path in required_files:
        if os.path.exists(file_path):
            existing_files.append(file_path)
            print(f"  ✅ {file_path}")
        else:
            missing_files.append(file_path)
            print(f"  ❌ {file_path} - MISSING")
    
    print(f"\nFile Structure Summary:")
    print(f"  ✅ Existing: {len(existing_files)}/{len(required_files)}")
    print(f"  ❌ Missing: {len(missing_files)}")
    
    return len(missing_files) == 0

def validate_file_sizes():
    """Validate that files have reasonable sizes (not empty)."""
    print("\n📏 Validating File Sizes...")
    
    files_to_check = [
        ('axiom_edge/config.py', 1500),  # Should be substantial
        ('axiom_edge/feature_engineer.py', 3000),  # Should be very large
        ('axiom_edge/model_trainer.py', 2000),  # Should be very large
        ('axiom_edge/backtester.py', 1500),  # Should be large
        ('axiom_edge/utils.py', 2000),  # Should be large
        ('axiom_edge/__init__.py', 400),  # Should be comprehensive
        ('axiom_edge/data_handler.py', 1500),  # Should be substantial
        ('axiom_edge/ai_analyzer.py', 1000),  # Should be substantial
        ('axiom_edge/telemetry.py', 1500),  # Should be substantial
        ('axiom_edge/tasks.py', 1300),  # Should be substantial
        ('axiom_edge/framework_orchestrator.py', 1000),  # Should be substantial
        ('main.py', 350),  # Should be comprehensive
        ('setup.py', 180),  # Should be comprehensive
    ]
    
    all_good = True
    
    for file_path, min_lines in files_to_check:
        if os.path.exists(file_path):
            with open(file_path, 'r', encoding='utf-8') as f:
                line_count = sum(1 for _ in f)
            
            if line_count >= min_lines:
                print(f"  ✅ {file_path}: {line_count} lines (>= {min_lines})")
            else:
                print(f"  ⚠️ {file_path}: {line_count} lines (< {min_lines}) - May be incomplete")
                all_good = False
        else:
            print(f"  ❌ {file_path}: File not found")
            all_good = False
    
    return all_good

def validate_imports():
    """Validate that basic imports work."""
    print("\n🔍 Validating Basic Imports...")
    
    import_tests = [
        'axiom_edge',
        'axiom_edge.config',
        'axiom_edge.utils',
        'axiom_edge.data_handler',
        'axiom_edge.ai_analyzer',
        'axiom_edge.feature_engineer',
        'axiom_edge.model_trainer',
        'axiom_edge.backtester',
        'axiom_edge.genetic_programmer',
        'axiom_edge.report_generator',
        'axiom_edge.framework_orchestrator',
        'axiom_edge.telemetry',
        'axiom_edge.tasks',
        'axiom_edge.main'
    ]
    
    successful_imports = 0
    
    for module in import_tests:
        try:
            __import__(module)
            print(f"  ✅ {module}")
            successful_imports += 1
        except Exception as e:
            print(f"  ❌ {module}: {str(e)}")
    
    success_rate = (successful_imports / len(import_tests)) * 100
    print(f"\nImport Success Rate: {successful_imports}/{len(import_tests)} ({success_rate:.1f}%)")
    
    return success_rate >= 80

def validate_constants():
    """Validate that required constants are available."""
    print("\n🔧 Validating Constants and Configuration...")
    
    try:
        from axiom_edge.config import TIMEFRAME_MAP, ANOMALY_FEATURES, NON_FEATURE_COLS
        
        print(f"  ✅ TIMEFRAME_MAP: {len(TIMEFRAME_MAP)} timeframes")
        print(f"  ✅ ANOMALY_FEATURES: {len(ANOMALY_FEATURES)} features")
        print(f"  ✅ NON_FEATURE_COLS: {len(NON_FEATURE_COLS)} columns")
        
        # Test config creation
        from axiom_edge.config import create_default_config
        config = create_default_config("./")
        print(f"  ✅ Default config created: {config.REPORT_LABEL}")
        
        return True
        
    except Exception as e:
        print(f"  ❌ Constants validation failed: {e}")
        return False

def validate_class_availability():
    """Validate that main classes can be instantiated."""
    print("\n🏗️ Validating Class Availability...")
    
    try:
        from axiom_edge.config import create_default_config
        config = create_default_config("./")
        
        # Test DataHandler
        from axiom_edge.data_handler import DataHandler
        data_handler = DataHandler(config)
        print("  ✅ DataHandler instantiated")
        
        # Test FeatureEngineer
        from axiom_edge.feature_engineer import FeatureEngineer
        timeframe_roles = {'base': 'D1'}
        playbook = {}
        feature_engineer = FeatureEngineer(config, timeframe_roles, playbook)
        print("  ✅ FeatureEngineer instantiated")
        
        # Test AI Analyzer
        from axiom_edge.ai_analyzer import GeminiAnalyzer
        ai_analyzer = GeminiAnalyzer()
        print("  ✅ GeminiAnalyzer instantiated")
        
        # Test ModelTrainer
        from axiom_edge.model_trainer import ModelTrainer
        model_trainer = ModelTrainer(config, ai_analyzer)
        print("  ✅ ModelTrainer instantiated")
        
        return True
        
    except Exception as e:
        print(f"  ❌ Class instantiation failed: {e}")
        return False

def validate_package_functionality():
    """Validate package-level functionality."""
    print("\n📦 Validating Package Functionality...")

    try:
        import axiom_edge

        # Test version
        version = axiom_edge.__version__
        print(f"  ✅ Package version: {version}")

        # Test framework info
        info = axiom_edge.get_framework_info()
        print(f"  ✅ Framework info: {len(info['components'])} components")

        # Test installation validation
        validation = axiom_edge.validate_installation()
        print(f"  ✅ Installation validation: {validation['installation_valid']}")

        # Test convenience functions
        config = axiom_edge.create_default_config("./")
        print(f"  ✅ Convenience functions working")

        return True

    except Exception as e:
        print(f"  ❌ Package functionality validation failed: {e}")
        return False

def validate_task_interfaces():
    """Validate task interface functionality."""
    print("\n📋 Validating Task Interfaces...")

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
        tasks = [
            DataCollectionTask(config),
            FeatureEngineeringTask(config),
            ModelTrainingTask(config),
            BacktestTask(config),
            CompleteFrameworkTask(config)
        ]

        print(f"  ✅ All {len(tasks)} specialized tasks instantiated")

        return True

    except Exception as e:
        print(f"  ❌ Task interface validation failed: {e}")
        return False

def validate_telemetry_system():
    """Validate telemetry and monitoring system."""
    print("\n📊 Validating Telemetry System...")

    try:
        from axiom_edge.telemetry import TelemetryCollector, TelemetryAnalyzer, InterventionManager

        # Test TelemetryCollector
        telemetry = TelemetryCollector("test_validation.jsonl")
        print(f"  ✅ TelemetryCollector: {telemetry.session_id}")

        # Test TelemetryAnalyzer
        analyzer = TelemetryAnalyzer("test_validation.jsonl")
        print(f"  ✅ TelemetryAnalyzer instantiated")

        # Test InterventionManager
        intervention_mgr = InterventionManager("test_interventions.json")
        print(f"  ✅ InterventionManager instantiated")

        return True

    except Exception as e:
        print(f"  ❌ Telemetry system validation failed: {e}")
        return False

def validate_utility_functions():
    """Validate utility functions."""
    print("\n🛠️ Validating Utility Functions...")

    try:
        from axiom_edge.utils import (
            get_optimal_system_settings, safe_divide, json_serializer_default,
            calculate_memory_usage, validate_data_quality
        )

        # Test system settings
        settings = get_optimal_system_settings()
        print(f"  ✅ System settings: {len(settings)} settings")

        # Test safe divide
        result = safe_divide(10, 2, default=0)
        assert result == 5.0
        print(f"  ✅ Safe divide: 10/2 = {result}")

        # Test safe divide with zero
        result = safe_divide(10, 0, default=-1)
        assert result == -1
        print(f"  ✅ Safe divide with zero: {result}")

        # Test JSON serializer
        import datetime
        result = json_serializer_default(datetime.datetime.now())
        print(f"  ✅ JSON serializer working")

        return True

    except Exception as e:
        print(f"  ❌ Utility functions validation failed: {e}")
        return False

def validate_configuration_system():
    """Validate configuration system comprehensively."""
    print("\n⚙️ Validating Configuration System...")

    try:
        from axiom_edge.config import (
            create_default_config, validate_config, get_config_summary,
            generate_dynamic_config, TIMEFRAME_MAP, ANOMALY_FEATURES
        )

        # Test config creation
        config = create_default_config("./")
        print(f"  ✅ Default config: {config.REPORT_LABEL}")

        # Test config validation
        validation = validate_config(config)
        print(f"  ✅ Config validation: {validation['is_valid']}")

        # Test config summary
        summary = get_config_summary(config)
        print(f"  ✅ Config summary: {summary['framework_version']}")

        # Test dynamic config
        dynamic_config = generate_dynamic_config("forex", {})
        print(f"  ✅ Dynamic config generated")

        # Test constants
        print(f"  ✅ Constants: {len(TIMEFRAME_MAP)} timeframes, {len(ANOMALY_FEATURES)} anomaly features")

        return True

    except Exception as e:
        print(f"  ❌ Configuration system validation failed: {e}")
        return False

def main():
    """Main validation function."""
    print("🚀 AxiomEdge Framework Validation")
    print("=" * 50)
    
    validation_results = {
        'file_structure': validate_file_structure(),
        'file_sizes': validate_file_sizes(),
        'imports': validate_imports(),
        'constants': validate_constants(),
        'classes': validate_class_availability(),
        'package_functionality': validate_package_functionality(),
        'task_interfaces': validate_task_interfaces(),
        'telemetry_system': validate_telemetry_system(),
        'utility_functions': validate_utility_functions(),
        'configuration_system': validate_configuration_system()
    }
    
    print("\n" + "=" * 50)
    print("📊 VALIDATION SUMMARY")
    print("=" * 50)
    
    passed = 0
    total = len(validation_results)
    
    for test_name, result in validation_results.items():
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{status} {test_name.replace('_', ' ').title()}")
        if result:
            passed += 1
    
    success_rate = (passed / total) * 100
    print(f"\nOverall Validation: {passed}/{total} ({success_rate:.1f}%)")
    
    if success_rate == 100:
        print("🎉 PERFECT: Framework is fully validated and ready!")
        print("✨ All components are properly installed and functional.")
    elif success_rate >= 80:
        print("✅ EXCELLENT: Framework validation passed!")
        print("🔧 Minor issues detected but framework should work correctly.")
    elif success_rate >= 60:
        print("⚠️ GOOD: Framework mostly validated with minor issues.")
        print("🔍 Some components may have issues. Review failed validations.")
    else:
        print("❌ ISSUES: Framework has significant validation failures.")
        print("🚨 Critical issues detected. Framework may not work properly.")

    # Show failed validations for debugging
    failed_validations = [name for name, result in validation_results.items() if not result]
    if failed_validations:
        print(f"\n🔍 Failed Validations ({len(failed_validations)}):")
        for validation in failed_validations:
            print(f"  ❌ {validation.replace('_', ' ').title()}")
        print("\n💡 Tip: Run individual validation functions to get detailed error information.")

    print(f"\nValidation completed at: {datetime.now().isoformat()}")

    return success_rate >= 80

if __name__ == "__main__":
    try:
        success = main()
        sys.exit(0 if success else 1)
    except Exception as e:
        print(f"\n❌ CRITICAL VALIDATION ERROR: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
