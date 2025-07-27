#!/usr/bin/env python3
# =============================================================================
# AXIOM EDGE - BASIC USAGE EXAMPLES
# =============================================================================

"""
This script demonstrates basic usage of the modular AxiomEdge framework.
Each example shows how to use individual components independently.
"""

import sys
import os
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from axiom_edge import (
    DataCollectionTask,
    BacktestTask,
    FeatureEngineeringTask,
    ModelTrainingTask,
    CompleteFrameworkTask,
    create_default_config
)

def example_1_data_collection():
    """Example 1: Collect historical data"""
    print("=" * 60)
    print("EXAMPLE 1: DATA COLLECTION")
    print("=" * 60)
    
    # Create a data collection task
    task = DataCollectionTask()
    
    # Collect data for multiple symbols
    symbols = ["AAPL", "GOOGL"]
    start_date = "2023-01-01"
    end_date = "2023-12-31"
    
    print(f"Collecting data for {symbols} from {start_date} to {end_date}")
    
    # Note: This will use fallback data since we don't have real API keys
    data = task.collect_data(symbols, start_date, end_date, source="yahoo")
    
    if data:
        for symbol, df in data.items():
            print(f"✓ {symbol}: {len(df)} records")
    else:
        print("No data collected (API keys may be missing)")
    
    # Show cache information
    cache_info = task.get_cache_info()
    print(f"Cache info: {cache_info}")

def example_2_data_validation():
    """Example 2: Data validation and quality checks"""
    print("\n" + "=" * 60)
    print("EXAMPLE 2: DATA VALIDATION")
    print("=" * 60)

    # Create sample data with some quality issues
    dates = pd.date_range(start="2023-01-01", end="2023-03-31", freq="D")
    np.random.seed(42)

    # Generate data with intentional issues
    price = 100
    prices = []
    for _ in range(len(dates)):
        price *= (1 + np.random.normal(0, 0.02))
        prices.append(price)

    sample_data = pd.DataFrame({
        'timestamp': dates,
        'Open': prices,
        'High': [p * (1 + abs(np.random.normal(0, 0.01))) for p in prices],
        'Low': [p * (1 - abs(np.random.normal(0, 0.01))) for p in prices],
        'Close': prices,
        'Volume': np.random.randint(1000000, 10000000, len(dates))
    })

    # Introduce some data quality issues
    sample_data.loc[10:15, 'Close'] = np.nan  # Missing data
    sample_data.loc[20, 'High'] = sample_data.loc[20, 'Low'] - 5  # Invalid high < low

    print(f"Sample data with quality issues: {len(sample_data)} records")

    # Validate data quality
    from axiom_edge.utils import validate_data_quality
    required_columns = ['Open', 'High', 'Low', 'Close', 'Volume']

    try:
        quality_report = validate_data_quality(sample_data, required_columns)
        print(f"Data quality: {quality_report['overall_quality']}")
        print(f"Missing data: {quality_report['missing_data_pct']:.1%}")
        print(f"Issues found: {len(quality_report['issues'])}")
    except Exception as e:
        print(f"Data validation error: {e}")

def example_3_feature_engineering():
    """Example 3: Engineer features from price data"""
    print("\n" + "=" * 60)
    print("EXAMPLE 3: FEATURE ENGINEERING")
    print("=" * 60)
    
    # Create sample price data
    dates = pd.date_range(start="2023-01-01", end="2023-12-31", freq="D")
    np.random.seed(42)
    
    # Generate realistic price data
    price = 100
    prices = []
    for _ in range(len(dates)):
        price *= (1 + np.random.normal(0, 0.02))  # 2% daily volatility
        prices.append(price)
    
    sample_data = pd.DataFrame({
        'timestamp': dates,
        'Open': prices,
        'High': [p * (1 + abs(np.random.normal(0, 0.01))) for p in prices],
        'Low': [p * (1 - abs(np.random.normal(0, 0.01))) for p in prices],
        'Close': prices,
        'Volume': np.random.randint(1000000, 10000000, len(dates))
    })
    
    print(f"Sample data shape: {sample_data.shape}")
    
    # Create feature engineering task
    task = FeatureEngineeringTask()
    
    # Engineer features
    print("Engineering features...")
    features = task.engineer_features(sample_data)
    
    print(f"Engineered features shape: {features.shape}")
    print(f"Feature columns: {list(features.columns)}")
    
    # Select top features
    if 'Close' in features.columns:
        # Create a simple target for feature selection
        features['target'] = (features['Close'].shift(-1) / features['Close'] - 1 > 0).astype(int)
        
        selected_features = task.select_features(features, 'target', n_features=10)
        print(f"Selected top 10 features: {selected_features}")

def example_4_backtesting():
    """Example 4: Backtest a simple strategy"""
    print("\n" + "=" * 60)
    print("EXAMPLE 4: STRATEGY BACKTESTING")
    print("=" * 60)
    
    # Create sample data (same as example 3)
    dates = pd.date_range(start="2023-01-01", end="2023-12-31", freq="D")
    np.random.seed(42)
    
    price = 100
    prices = []
    for _ in range(len(dates)):
        price *= (1 + np.random.normal(0, 0.02))
        prices.append(price)
    
    sample_data = pd.DataFrame({
        'timestamp': dates,
        'Open': prices,
        'High': [p * (1 + abs(np.random.normal(0, 0.01))) for p in prices],
        'Low': [p * (1 - abs(np.random.normal(0, 0.01))) for p in prices],
        'Close': prices,
        'Volume': np.random.randint(1000000, 10000000, len(dates))
    })
    sample_data.set_index('timestamp', inplace=True)
    
    print(f"Backtesting data shape: {sample_data.shape}")
    
    # Create backtest task
    task = BacktestTask()
    
    # Define a simple moving average crossover strategy
    strategy_rules = {
        "sma_short": 10,
        "sma_long": 30
    }
    
    print("Running backtest with SMA crossover strategy...")
    results = task.backtest_strategy(sample_data, strategy_rules)
    
    if results and "error" not in results:
        print("Backtest Results:")
        print(f"  Total Return: {results.get('total_return', 0):.2%}")
        print(f"  Sharpe Ratio: {results.get('sharpe_ratio', 0):.2f}")
        print(f"  Max Drawdown: {results.get('max_drawdown', 0):.2%}")
        print(f"  Number of Trades: {results.get('num_trades', 0)}")
    else:
        print(f"Backtest failed: {results}")

def example_5_model_training():
    """Example 5: Train a machine learning model"""
    print("\n" + "=" * 60)
    print("EXAMPLE 5: MODEL TRAINING")
    print("=" * 60)
    
    # Create sample feature data
    np.random.seed(42)
    n_samples = 1000
    n_features = 20
    
    # Generate random features
    features = pd.DataFrame(
        np.random.randn(n_samples, n_features),
        columns=[f'feature_{i}' for i in range(n_features)]
    )
    
    # Generate a target based on some features (to make it learnable)
    target = (
        features['feature_0'] * 0.5 + 
        features['feature_1'] * 0.3 + 
        features['feature_2'] * 0.2 + 
        np.random.randn(n_samples) * 0.1 > 0
    ).astype(int)
    
    print(f"Training data shape: {features.shape}")
    print(f"Target distribution: {target.value_counts().to_dict()}")
    
    # Create model training task
    task = ModelTrainingTask()
    
    # Train model
    print("Training model...")
    results = task.train_model(features, target)
    
    if results and "error" not in results:
        print("Training Results:")
        print(f"  Accuracy: {results.get('accuracy', 0):.3f}")
        print(f"  Number of features: {len(results.get('selected_features', []))}")
        
        # Show top feature importances
        feature_importance = results.get('feature_importance', {})
        if feature_importance:
            top_features = sorted(feature_importance.items(), key=lambda x: x[1], reverse=True)[:5]
            print("  Top 5 features:")
            for feature, importance in top_features:
                print(f"    {feature}: {importance:.3f}")
    else:
        print(f"Training failed: {results}")

def example_6_complete_framework():
    """Example 6: Run the complete framework"""
    print("\n" + "=" * 60)
    print("EXAMPLE 6: COMPLETE FRAMEWORK")
    print("=" * 60)
    
    # Create configuration
    config = create_default_config("./")
    
    # Create complete framework task
    task = CompleteFrameworkTask(config)
    
    # Simulate data files
    data_files = ["sample_data_1.csv", "sample_data_2.csv"]
    
    print(f"Running complete framework with {len(data_files)} data files...")
    
    # Run complete framework
    results = task.run_complete_framework(data_files)
    
    if results:
        print("Framework Results:")
        print(f"  Status: {results.get('status', 'unknown')}")
        print(f"  Message: {results.get('message', 'No message')}")
        print(f"  Files processed: {results.get('data_files_processed', 0)}")
        print(f"  Components run: {results.get('components_run', [])}")
    else:
        print("Framework execution failed")

def main():
    """Run all examples"""
    print("🚀 AxiomEdge Modular Framework - Basic Usage Examples")
    print("=" * 80)
    
    try:
        example_1_data_collection()
        example_2_data_validation()
        example_3_feature_engineering()
        example_4_backtesting()
        example_5_model_training()
        example_6_complete_framework()
        
        print("\n" + "=" * 80)
        print("✅ All examples completed successfully!")
        print("\nNext steps:")
        print("1. Configure API keys for real data collection")
        print("2. Customize strategies and parameters")
        print("3. Run on your own data")
        print("4. Explore advanced features")
        
    except Exception as e:
        print(f"\n❌ Error running examples: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
