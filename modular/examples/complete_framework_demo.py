#!/usr/bin/env python3
# =============================================================================
# AXIOM EDGE - COMPLETE FRAMEWORK DEMONSTRATION
# =============================================================================

"""
This script demonstrates the complete AxiomEdge framework orchestration,
including walk-forward analysis, model training, strategy evolution,
and comprehensive reporting.
"""

import sys
import os
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from axiom_edge import (
    FrameworkOrchestrator,
    CompleteFrameworkTask,
    create_default_config
)

def create_comprehensive_market_data(n_days: int = 500, symbol: str = "DEMO") -> pd.DataFrame:
    """Create comprehensive market data for framework demonstration"""
    np.random.seed(42)
    
    # Generate realistic market data with trends and patterns
    dates = pd.date_range(start="2023-01-01", periods=n_days, freq="D")
    
    # Initialize price series
    price = 100.0
    prices = []
    volumes = []
    
    for i in range(n_days):
        # Add multiple trend components
        long_trend = 0.0002 * np.sin(2 * np.pi * i / 252)  # Annual cycle
        medium_trend = 0.0005 * np.sin(2 * np.pi * i / 63)  # Quarterly cycle
        short_trend = 0.001 * np.sin(2 * np.pi * i / 21)   # Monthly cycle
        
        # Add volatility clustering and momentum
        base_vol = 0.02
        if i > 0:
            vol_cluster = base_vol * (1 + 0.3 * abs(prices[-1] / price - 1) * 10)
            momentum = (prices[-1] / price - 1) * 0.1 if prices else 0
        else:
            vol_cluster = base_vol
            momentum = 0
        
        # Generate daily return
        daily_return = long_trend + medium_trend + short_trend + momentum + np.random.normal(0, vol_cluster)
        price *= (1 + daily_return)
        prices.append(price)
        
        # Generate volume with correlation to volatility
        base_volume = 1000000
        volume_factor = 1 + abs(daily_return) * 5  # Higher volume on big moves
        volume = int(base_volume * volume_factor * np.random.uniform(0.7, 1.3))
        volumes.append(volume)
    
    # Create OHLCV data
    data = []
    for i, (date, close_price, volume) in enumerate(zip(dates, prices, volumes)):
        # Generate realistic OHLC from close price
        daily_range = close_price * np.random.uniform(0.008, 0.025)  # 0.8% to 2.5% daily range
        
        open_price = close_price * np.random.uniform(0.997, 1.003)
        high_price = max(open_price, close_price) + daily_range * np.random.uniform(0.2, 0.8)
        low_price = min(open_price, close_price) - daily_range * np.random.uniform(0.2, 0.8)
        
        data.append({
            'timestamp': date,
            'Open': open_price,
            'High': high_price,
            'Low': low_price,
            'Close': close_price,
            'RealVolume': volume,
            'Symbol': symbol
        })
    
    df = pd.DataFrame(data)
    df.set_index('timestamp', inplace=True)
    
    return df

def save_sample_data(data: pd.DataFrame, filename: str) -> str:
    """Save sample data to CSV file"""
    os.makedirs("sample_data", exist_ok=True)
    filepath = os.path.join("sample_data", filename)
    data.to_csv(filepath)
    return filepath

def demonstrate_complete_framework():
    """Demonstrate complete framework orchestration"""
    print("=" * 80)
    print("COMPLETE FRAMEWORK ORCHESTRATION DEMONSTRATION")
    print("=" * 80)
    
    # Create sample data
    print("📊 Creating comprehensive market data...")
    market_data = create_comprehensive_market_data(400, "DEMO_STOCK")
    data_file = save_sample_data(market_data, "demo_stock.csv")
    
    print(f"   Generated {len(market_data)} days of market data")
    print(f"   Price range: ${market_data['Close'].min():.2f} - ${market_data['Close'].max():.2f}")
    print(f"   Saved to: {data_file}")
    
    # Create configuration
    print("\n⚙️  Configuring framework...")
    config = create_default_config("./")
    config.nickname = "Complete Framework Demo"
    config.REPORT_LABEL = "V2.1_COMPLETE_DEMO"
    
    # Framework-specific settings
    config.TRAINING_WINDOW_DAYS = 150  # 5 months training
    config.TEST_WINDOW_DAYS = 30       # 1 month testing
    config.WALK_FORWARD_STEP_DAYS = 15 # 2 weeks step
    config.MAX_WALK_FORWARD_CYCLES = 3 # Limit for demo
    config.ENABLE_GENETIC_PROGRAMMING = True
    config.GP_POPULATION_SIZE = 20
    config.GP_GENERATIONS = 10
    
    print(f"   Training window: {config.TRAINING_WINDOW_DAYS} days")
    print(f"   Test window: {config.TEST_WINDOW_DAYS} days")
    print(f"   Walk-forward cycles: {config.MAX_WALK_FORWARD_CYCLES}")
    print(f"   Genetic programming: {config.ENABLE_GENETIC_PROGRAMMING}")
    
    # Initialize framework orchestrator
    print("\n🚀 Initializing Framework Orchestrator...")
    orchestrator = FrameworkOrchestrator(config)
    
    # Run complete framework
    print("⚙️  Running complete framework analysis...")
    results = orchestrator.run_complete_framework(
        data_files=[data_file],
        symbols=None,
        start_date=None,
        end_date=None
    )
    
    # Display results
    if results.get('status') == 'completed':
        print(f"\n✅ Complete framework execution completed!")
        print(f"   Execution time: {results.get('execution_time_seconds', 0):.2f} seconds")
        print(f"   Data files processed: {results.get('data_files_processed', 0)}")
        print(f"   Symbols analyzed: {results.get('symbols_analyzed', [])}")
        print(f"   Features engineered: {results.get('features_engineered', 0)}")
        print(f"   Walk-forward cycles: {results.get('walk_forward_cycles', 0)}")
        
        # Display cycle breakdown
        cycle_breakdown = results.get('cycle_breakdown', [])
        if cycle_breakdown:
            print(f"\n📊 Cycle Performance Breakdown:")
            print(f"{'Cycle':<6} {'Status':<10} {'F1 Score':<10} {'Accuracy':<10} {'Features':<10}")
            print("-" * 60)
            
            for cycle in cycle_breakdown:
                cycle_num = cycle.get('cycle', 'N/A')
                status = cycle.get('status', 'N/A')
                metrics = cycle.get('metrics', {})
                f1_score = metrics.get('f1_score', 0)
                accuracy = metrics.get('accuracy', 0)
                features = cycle.get('features_used', 0)
                
                print(f"{cycle_num:<6} {status:<10} {f1_score:<10.3f} {accuracy:<10.3f} {features:<10}")
        
        # Display final metrics
        final_metrics = results.get('final_metrics', {})
        if final_metrics:
            print(f"\n📈 Final Performance Metrics:")
            print(f"   Total Return: {final_metrics.get('net_profit_pct', 0):.2%}")
            print(f"   Sharpe Ratio: {final_metrics.get('sharpe_ratio', 0):.2f}")
            print(f"   Max Drawdown: {final_metrics.get('max_drawdown_pct', 0):.2%}")
            print(f"   Win Rate: {final_metrics.get('win_rate', 0):.2%}")
        
        # Display evolution results
        evolution = results.get('evolution')
        if evolution and evolution.get('status') == 'completed':
            print(f"\n🧬 Strategy Evolution Results:")
            print(f"   Best fitness: {evolution.get('best_fitness', 0):.4f}")
            best_rules = evolution.get('best_rules', {})
            print(f"   Long rule: {best_rules.get('long_rule', 'N/A')}")
            print(f"   Short rule: {best_rules.get('short_rule', 'N/A')}")
    
    else:
        print(f"❌ Framework execution failed: {results.get('error', 'Unknown error')}")
    
    return results

def demonstrate_task_interface():
    """Demonstrate framework using task interface"""
    print("\n" + "=" * 80)
    print("TASK INTERFACE DEMONSTRATION")
    print("=" * 80)
    
    # Create sample data
    print("📊 Creating sample data for task interface...")
    market_data = create_comprehensive_market_data(300, "TASK_DEMO")
    data_file = save_sample_data(market_data, "task_demo.csv")
    
    print(f"   Generated {len(market_data)} days of data")
    print(f"   Saved to: {data_file}")
    
    # Create configuration
    config = create_default_config("./")
    config.nickname = "Task Interface Demo"
    config.MAX_WALK_FORWARD_CYCLES = 2  # Quick demo
    
    # Use task interface
    print("\n🎯 Using CompleteFrameworkTask interface...")
    task = CompleteFrameworkTask(config)
    
    # Run framework
    print("⚙️  Running framework via task interface...")
    results = task.run_complete_framework([data_file])
    
    # Display results
    if results.get('status') == 'completed':
        print(f"\n✅ Task interface execution completed!")
        print(f"   Execution time: {results.get('execution_time_seconds', 0):.2f} seconds")
        print(f"   Cycles completed: {results.get('walk_forward_cycles', 0)}")
        
        # Show framework status
        if hasattr(task, 'orchestrator'):
            status = task.orchestrator.get_framework_status()
            print(f"\n📊 Framework Status:")
            print(f"   Successful cycles: {status.get('successful_cycles', 0)}")
            print(f"   Failed cycles: {status.get('failed_cycles', 0)}")
            print(f"   Features available: {status.get('features_available', 0)}")
    else:
        print(f"❌ Task interface execution failed: {results.get('error', 'Unknown error')}")
    
    return results

def demonstrate_single_cycle_analysis():
    """Demonstrate single cycle analysis for testing"""
    print("\n" + "=" * 80)
    print("SINGLE CYCLE ANALYSIS DEMONSTRATION")
    print("=" * 80)
    
    # Create focused dataset
    print("📊 Creating focused dataset for single cycle analysis...")
    market_data = create_comprehensive_market_data(200, "SINGLE_CYCLE")
    
    print(f"   Generated {len(market_data)} days of focused data")
    
    # Create configuration
    config = create_default_config("./")
    config.nickname = "Single Cycle Demo"
    
    # Initialize orchestrator
    print("\n🔬 Running single cycle analysis...")
    orchestrator = FrameworkOrchestrator(config)
    
    # Run single cycle
    results = orchestrator.run_single_cycle_analysis(market_data)
    
    # Display results
    if results.get('status') == 'completed':
        print(f"\n✅ Single cycle analysis completed!")
        print(f"   Features engineered: {results.get('features_engineered', 0)}")
        
        cycle_results = results.get('cycle_results', {})
        if cycle_results.get('status') == 'completed':
            metrics = cycle_results.get('metrics', {})
            print(f"   F1 Score: {metrics.get('f1_score', 0):.3f}")
            print(f"   Accuracy: {metrics.get('accuracy', 0):.3f}")
            print(f"   Features used: {cycle_results.get('features_used', 0)}")
            print(f"   Training samples: {cycle_results.get('training_samples', 0)}")
        
        final_metrics = results.get('final_metrics', {})
        if final_metrics:
            print(f"\n📈 Performance Summary:")
            print(f"   Total Return: {final_metrics.get('net_profit_pct', 0):.2%}")
            print(f"   Sharpe Ratio: {final_metrics.get('sharpe_ratio', 0):.2f}")
    else:
        print(f"❌ Single cycle analysis failed: {results.get('error', 'Unknown error')}")
    
    return results

def demonstrate_multi_symbol_analysis():
    """Demonstrate multi-symbol framework analysis"""
    print("\n" + "=" * 80)
    print("MULTI-SYMBOL ANALYSIS DEMONSTRATION")
    print("=" * 80)
    
    # Create multiple symbol datasets
    symbols = ["STOCK_A", "STOCK_B", "FOREX_PAIR"]
    data_files = []
    
    print("📊 Creating multi-symbol datasets...")
    for symbol in symbols:
        # Create different market characteristics for each symbol
        if "STOCK" in symbol:
            data = create_comprehensive_market_data(250, symbol)
        else:  # FOREX
            # Create more volatile forex-like data
            np.random.seed(hash(symbol) % 2**32)
            data = create_comprehensive_market_data(250, symbol)
            # Add forex-specific volatility
            data['Close'] *= (1 + np.random.normal(0, 0.01, len(data)))
        
        filename = f"{symbol.lower()}.csv"
        filepath = save_sample_data(data, filename)
        data_files.append(filepath)
        
        print(f"   {symbol}: {len(data)} days, range ${data['Close'].min():.2f}-${data['Close'].max():.2f}")
    
    # Create configuration for multi-symbol analysis
    config = create_default_config("./")
    config.nickname = "Multi-Symbol Demo"
    config.MAX_WALK_FORWARD_CYCLES = 2  # Quick demo
    config.ENABLE_GENETIC_PROGRAMMING = False  # Disable for speed
    
    # Run framework on multiple symbols
    print(f"\n🌐 Running framework on {len(symbols)} symbols...")
    orchestrator = FrameworkOrchestrator(config)
    
    # Note: Current implementation processes first symbol
    # In production, this would handle multiple symbols simultaneously
    results = orchestrator.run_complete_framework(data_files)
    
    # Display results
    if results.get('status') == 'completed':
        print(f"\n✅ Multi-symbol analysis completed!")
        print(f"   Data files processed: {results.get('data_files_processed', 0)}")
        print(f"   Symbols analyzed: {results.get('symbols_analyzed', [])}")
        print(f"   Total execution time: {results.get('execution_time_seconds', 0):.2f} seconds")
        
        # Show performance across cycles
        cycle_breakdown = results.get('cycle_breakdown', [])
        if cycle_breakdown:
            avg_f1 = np.mean([c.get('metrics', {}).get('f1_score', 0) for c in cycle_breakdown if c.get('status') == 'completed'])
            print(f"   Average F1 Score: {avg_f1:.3f}")
    else:
        print(f"❌ Multi-symbol analysis failed: {results.get('error', 'Unknown error')}")
    
    return results

def main():
    """Run all complete framework demonstrations"""
    print("🚀 AxiomEdge Complete Framework - Comprehensive Demonstration")
    print("=" * 80)
    
    try:
        # Complete framework demonstration
        complete_results = demonstrate_complete_framework()
        
        # Task interface demonstration
        task_results = demonstrate_task_interface()
        
        # Single cycle analysis
        single_results = demonstrate_single_cycle_analysis()
        
        # Multi-symbol analysis
        multi_results = demonstrate_multi_symbol_analysis()
        
        print("\n" + "=" * 80)
        print("✅ ALL COMPLETE FRAMEWORK DEMONSTRATIONS COMPLETED!")
        print("=" * 80)
        
        print("\n🎯 Summary:")
        if complete_results.get('status') == 'completed':
            print(f"   Complete Framework: {complete_results.get('walk_forward_cycles', 0)} cycles")
        if task_results.get('status') == 'completed':
            print(f"   Task Interface: {task_results.get('walk_forward_cycles', 0)} cycles")
        if single_results.get('status') == 'completed':
            print(f"   Single Cycle: {single_results.get('features_engineered', 0)} features")
        if multi_results.get('status') == 'completed':
            print(f"   Multi-Symbol: {multi_results.get('data_files_processed', 0)} files processed")
        
        print("\n📚 Key Capabilities Demonstrated:")
        print("   ✅ Complete framework orchestration")
        print("   ✅ Walk-forward analysis with multiple cycles")
        print("   ✅ Automated feature engineering pipeline")
        print("   ✅ Model training and validation")
        print("   ✅ Strategy evolution with genetic programming")
        print("   ✅ Comprehensive performance reporting")
        print("   ✅ Task-based interface for easy usage")
        print("   ✅ Single cycle analysis for testing")
        print("   ✅ Multi-symbol data processing")
        print("   ✅ Framework memory and state management")
        
        print("\n📁 Generated Outputs:")
        print("   📊 Results/equity_curve.png - Portfolio performance visualization")
        print("   🧠 Results/shap_summary.png - Feature importance analysis")
        print("   📈 Results/trade_analysis.png - Trade distribution analysis")
        print("   📋 Results/performance_report.txt - Comprehensive text report")
        print("   🌐 Results/performance_dashboard.html - Interactive dashboard")
        print("   📄 sample_data/*.csv - Generated sample datasets")
        
        print("\n🔧 Next Steps:")
        print("   1. Integrate with live data feeds")
        print("   2. Add real-time trading execution")
        print("   3. Implement portfolio optimization")
        print("   4. Add risk management overlays")
        print("   5. Create automated scheduling system")
        print("   6. Implement multi-asset correlation analysis")
        print("   7. Add regime detection and adaptation")
        
    except Exception as e:
        print(f"\n❌ Error during demonstration: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
