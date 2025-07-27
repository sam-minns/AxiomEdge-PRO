#!/usr/bin/env python3
# =============================================================================
# AXIOM EDGE - MAIN ENTRY POINT
# =============================================================================

"""
AxiomEdge Trading Framework - Main Entry Point

This script demonstrates how to use AxiomEdge in different modes:
1. Individual task execution (data collection, backtesting, etc.)
2. Complete framework execution
3. Custom workflow combinations

Usage Examples:
    # Collect historical data only
    python main.py --task data_collection --symbols AAPL,GOOGL --start 2023-01-01 --end 2024-01-01
    
    # Get broker information
    python main.py --task broker_info --symbols EURUSD,GBPUSD --broker oanda
    
    # Backtest a strategy
    python main.py --task backtest --data_file my_data.csv --strategy_config strategy.json
    
    # Run complete framework
    python main.py --task complete --data_files "data/*.csv" --config config.json
    
    # Feature engineering only
    python main.py --task features --data_file data.csv --output features.csv
"""

import argparse
import sys
import json
import glob
from pathlib import Path
from typing import List, Dict, Any

# Import AxiomEdge modules
from axiom_edge import (
    DataCollectionTask,
    BrokerInfoTask,
    BacktestTask,
    FeatureEngineeringTask,
    ModelTrainingTask,
    CompleteFrameworkTask,
    ConfigModel,
    create_default_config,
    setup_logging
)
from axiom_edge.config import load_config_from_file
from axiom_edge.utils import save_results

def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(
        description="AxiomEdge Trading Framework",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    
    parser.add_argument(
        '--task', 
        choices=['data_collection', 'broker_info', 'backtest', 'features', 'model_training', 'complete'],
        required=True,
        help='Task to execute'
    )
    
    parser.add_argument('--config', help='Configuration file path (JSON/YAML)')
    parser.add_argument('--log-level', default='INFO', choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'])
    parser.add_argument('--log-file', help='Log file path')
    parser.add_argument('--output-dir', default='results', help='Output directory')
    
    # Data collection arguments
    parser.add_argument('--symbols', help='Comma-separated list of symbols')
    parser.add_argument('--start', help='Start date (YYYY-MM-DD)')
    parser.add_argument('--end', help='End date (YYYY-MM-DD)')
    parser.add_argument('--timeframe', default='1D', help='Data timeframe')
    parser.add_argument('--source', default='alpha_vantage', help='Data source')
    
    # Broker info arguments
    parser.add_argument('--broker', default='oanda', help='Broker name')
    
    # Backtest arguments
    parser.add_argument('--data-file', help='Data file for backtesting')
    parser.add_argument('--strategy-config', help='Strategy configuration file')
    
    # Complete framework arguments
    parser.add_argument('--data-files', help='Data files pattern (e.g., "data/*.csv")')
    
    return parser.parse_args()

def load_config(config_path: str = None) -> ConfigModel:
    """Load configuration from file or create default"""
    if config_path and Path(config_path).exists():
        return load_config_from_file(config_path)
    else:
        return create_default_config("./")

def run_data_collection_task(args, config: ConfigModel):
    """Run data collection task"""
    print("🔄 Starting Data Collection Task")
    
    if not args.symbols:
        print("❌ Error: --symbols required for data collection")
        return
    
    symbols = [s.strip() for s in args.symbols.split(',')]
    
    task = DataCollectionTask(config)
    
    # Collect data
    data = task.collect_data(
        symbols=symbols,
        start_date=args.start or "2023-01-01",
        end_date=args.end or "2024-01-01",
        timeframe=args.timeframe,
        source=args.source
    )
    
    # Save collected data
    if data:
        task.save_data(data, args.output_dir)
        print(f"✅ Data collection completed. Saved to {args.output_dir}")
        
        # Print cache info
        cache_info = task.get_cache_info()
        print(f"📊 Cache Info: {cache_info}")
    else:
        print("❌ No data collected")

def run_broker_info_task(args, config: ConfigModel):
    """Run broker information task"""
    print("🔄 Starting Broker Info Task")
    
    if not args.symbols:
        print("❌ Error: --symbols required for broker info")
        return
    
    symbols = [s.strip() for s in args.symbols.split(',')]
    
    task = BrokerInfoTask(config)
    
    # Collect spreads
    spreads = task.collect_spreads(symbols, args.broker)
    
    if spreads:
        print(f"✅ Collected spreads for {len(spreads)} symbols:")
        for symbol, spread in spreads.items():
            print(f"  📈 {symbol}: {spread:.5f}")
        
        # Analyze with AI
        analysis = task.analyze_broker_costs(symbols, {'spreads': spreads})
        if analysis:
            print("🤖 AI Analysis:")
            for symbol, result in analysis.items():
                print(f"  {symbol}: {result.get('analysis', 'No analysis available')}")
    else:
        print("❌ No spread data collected")

def run_backtest_task(args, config: ConfigModel):
    """Run backtesting task"""
    print("🔄 Starting Backtest Task")
    
    if not args.data_file:
        print("❌ Error: --data-file required for backtesting")
        return
    
    # Load data
    import pandas as pd
    try:
        data = pd.read_csv(args.data_file)
        print(f"📊 Loaded data: {len(data)} rows, {len(data.columns)} columns")
    except Exception as e:
        print(f"❌ Error loading data: {e}")
        return
    
    # Load strategy configuration
    strategy_rules = {}
    if args.strategy_config and Path(args.strategy_config).exists():
        with open(args.strategy_config, 'r') as f:
            strategy_rules = json.load(f)
    else:
        # Default simple strategy
        strategy_rules = {
            'sma_short': 10,
            'sma_long': 30
        }
        print("📝 Using default SMA crossover strategy")
    
    task = BacktestTask(config)
    
    # Run backtest
    results = task.backtest_strategy(data, strategy_rules)
    
    if results:
        print("✅ Backtest completed")
        print(f"📊 Results: {results}")
        
        # Save results
        output_file = Path(args.output_dir) / "backtest_results.json"
        save_results(results, str(output_file))
        print(f"💾 Results saved to {output_file}")
    else:
        print("❌ Backtest failed")

def run_feature_engineering_task(args, config: ConfigModel):
    """Run feature engineering task"""
    print("🔄 Starting Feature Engineering Task")
    
    if not args.data_file:
        print("❌ Error: --data-file required for feature engineering")
        return
    
    # Load data
    import pandas as pd
    try:
        data = pd.read_csv(args.data_file)
        print(f"📊 Loaded data: {len(data)} rows, {len(data.columns)} columns")
    except Exception as e:
        print(f"❌ Error loading data: {e}")
        return
    
    task = FeatureEngineeringTask(config)
    
    # Engineer features
    features = task.engineer_features(data)
    
    if not features.empty:
        print(f"✅ Feature engineering completed: {len(features.columns)} features")
        
        # Save features
        output_file = Path(args.output_dir) / "engineered_features.csv"
        features.to_csv(output_file, index=False)
        print(f"💾 Features saved to {output_file}")
    else:
        print("❌ Feature engineering failed")

def run_model_training_task(args, config: ConfigModel):
    """Run model training task"""
    print("🔄 Starting Model Training Task")
    
    if not args.data_file:
        print("❌ Error: --data-file required for model training")
        return
    
    # Load data
    import pandas as pd
    try:
        data = pd.read_csv(args.data_file)
        print(f"📊 Loaded data: {len(data)} rows, {len(data.columns)} columns")
    except Exception as e:
        print(f"❌ Error loading data: {e}")
        return
    
    # Assume last column is target
    target_column = data.columns[-1]
    feature_columns = data.columns[:-1]
    
    task = ModelTrainingTask(config)
    
    # Train model
    results = task.train_model(
        features=data[feature_columns],
        target=data[target_column]
    )
    
    if results:
        print("✅ Model training completed")
        print(f"📊 Results: {results}")
        
        # Save results
        output_file = Path(args.output_dir) / "model_results.json"
        save_results(results, str(output_file))
        print(f"💾 Results saved to {output_file}")
    else:
        print("❌ Model training failed")

def run_complete_framework_task(args, config: ConfigModel):
    """Run complete framework"""
    print("🔄 Starting Complete Framework Task")
    
    if not args.data_files:
        print("❌ Error: --data-files required for complete framework")
        return
    
    # Get data files
    data_files = glob.glob(args.data_files)
    if not data_files:
        print(f"❌ No files found matching pattern: {args.data_files}")
        return
    
    print(f"📁 Found {len(data_files)} data files")
    
    task = CompleteFrameworkTask(config)
    
    # Run complete framework
    results = task.run_complete_framework(data_files)
    
    if results:
        print("✅ Complete framework execution finished")
        print(f"📊 Results summary: {len(results)} components completed")
        
        # Save results
        output_file = Path(args.output_dir) / "complete_framework_results.json"
        save_results(results, str(output_file))
        print(f"💾 Results saved to {output_file}")
    else:
        print("❌ Complete framework execution failed")

def main():
    """Main entry point"""
    args = parse_arguments()
    
    # Setup logging
    setup_logging(args.log_level, args.log_file)
    
    # Create output directory
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    
    # Load configuration
    config = load_config(args.config)
    
    print("🚀 AxiomEdge Trading Framework")
    print(f"📋 Task: {args.task}")
    print(f"📁 Output: {args.output_dir}")
    print("-" * 50)
    
    # Route to appropriate task
    try:
        if args.task == 'data_collection':
            run_data_collection_task(args, config)
        elif args.task == 'broker_info':
            run_broker_info_task(args, config)
        elif args.task == 'backtest':
            run_backtest_task(args, config)
        elif args.task == 'features':
            run_feature_engineering_task(args, config)
        elif args.task == 'model_training':
            run_model_training_task(args, config)
        elif args.task == 'complete':
            run_complete_framework_task(args, config)
        else:
            print(f"❌ Unknown task: {args.task}")
            
    except KeyboardInterrupt:
        print("\n⏹️  Task interrupted by user")
    except Exception as e:
        print(f"❌ Task failed with error: {e}")
        if args.log_level == 'DEBUG':
            import traceback
            traceback.print_exc()
    
    print("-" * 50)
    print("✅ AxiomEdge execution completed")

if __name__ == "__main__":
    main()
