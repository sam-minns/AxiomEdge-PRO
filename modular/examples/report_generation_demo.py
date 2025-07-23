#!/usr/bin/env python3
# =============================================================================
# AXIOM EDGE - REPORT GENERATION DEMONSTRATION
# =============================================================================

"""
This script demonstrates the comprehensive report generation capabilities
of the AxiomEdge framework, including text reports, visualizations, and
performance analysis.
"""

import sys
import os
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from axiom_edge import (
    ReportGenerator,
    FeatureEngineeringTask,
    ModelTrainingTask,
    create_default_config
)

def create_sample_trading_data(n_days: int = 252) -> tuple:
    """Create sample trading data for report generation"""
    np.random.seed(42)
    
    # Generate dates
    dates = pd.date_range(start="2023-01-01", periods=n_days, freq="D")
    
    # Generate realistic equity curve
    initial_capital = 10000
    daily_returns = np.random.normal(0.0008, 0.02, n_days)  # ~20% annual return, 20% volatility
    
    # Add some trend and volatility clustering
    for i in range(1, len(daily_returns)):
        # Momentum effect
        daily_returns[i] += daily_returns[i-1] * 0.1
        # Volatility clustering
        if abs(daily_returns[i-1]) > 0.03:
            daily_returns[i] *= 1.5
    
    # Create equity curve
    equity_values = [initial_capital]
    for ret in daily_returns:
        equity_values.append(equity_values[-1] * (1 + ret))
    
    equity_curve = pd.Series(equity_values[1:], index=dates)
    
    # Generate sample trades
    n_trades = 45
    trade_dates = np.random.choice(dates, n_trades, replace=False)
    trade_dates.sort()
    
    trades_data = []
    for i, trade_date in enumerate(trade_dates):
        # Generate realistic trade data
        entry_price = 100 + np.random.normal(0, 10)
        position_size = np.random.choice([100, 200, 300, 500])
        
        # Generate PnL with some winning bias
        if np.random.random() < 0.6:  # 60% win rate
            pnl = np.random.uniform(50, 500)  # Winning trade
        else:
            pnl = np.random.uniform(-400, -50)  # Losing trade
        
        exit_price = entry_price + (pnl / position_size)
        duration = np.random.randint(1, 10)  # Days
        
        trades_data.append({
            'trade_id': i + 1,
            'entry_time': trade_date,
            'exit_time': trade_date + timedelta(days=duration),
            'symbol': np.random.choice(['AAPL', 'GOOGL', 'MSFT', 'TSLA']),
            'side': np.random.choice(['long', 'short']),
            'entry_price': entry_price,
            'exit_price': exit_price,
            'position_size': position_size,
            'pnl': pnl,
            'duration': duration
        })
    
    trades_df = pd.DataFrame(trades_data)
    
    return equity_curve, trades_df

def create_sample_shap_data() -> pd.DataFrame:
    """Create sample SHAP feature importance data"""
    features = [
        'RSI_14', 'EMA_20', 'EMA_50', 'MACD', 'ATR', 'volume_ma_ratio',
        'BB_position_20', 'momentum_10', 'price_velocity', 'returns_std_20',
        'is_ny_session', 'day_of_week', 'hour_sin', 'volatility_hawkes',
        'entropy_returns', 'cycle_phase', 'trend_strength', 'support_resistance',
        'pattern_doji', 'volume_spike'
    ]
    
    # Generate realistic SHAP importance values
    np.random.seed(42)
    importance_values = np.random.exponential(0.1, len(features))
    importance_values = np.sort(importance_values)[::-1]  # Sort descending
    
    shap_df = pd.DataFrame({
        'feature': features,
        'SHAP_Importance': importance_values
    })
    
    return shap_df

def create_sample_cycle_metrics() -> list:
    """Create sample walk-forward cycle metrics"""
    cycles = []
    
    for i in range(5):
        cycle_metrics = {
            'cycle': i + 1,
            'status': 'completed',
            'metrics': {
                'total_trades': np.random.randint(8, 15),
                'total_net_profit': np.random.uniform(-500, 1500),
                'win_rate': np.random.uniform(0.4, 0.8),
                'max_drawdown_pct': np.random.uniform(0.05, 0.25),
                'sharpe_ratio': np.random.uniform(0.5, 2.0),
                'mar_ratio': np.random.uniform(0.8, 3.0)
            }
        }
        cycles.append(cycle_metrics)
    
    return cycles

def demonstrate_basic_report_generation():
    """Demonstrate basic report generation"""
    print("=" * 80)
    print("BASIC REPORT GENERATION DEMONSTRATION")
    print("=" * 80)
    
    # Create sample data
    print("📊 Creating sample trading data...")
    equity_curve, trades_df = create_sample_trading_data(252)
    shap_data = create_sample_shap_data()
    cycle_metrics = create_sample_cycle_metrics()
    
    print(f"   Generated equity curve: {len(equity_curve)} days")
    print(f"   Generated trades: {len(trades_df)} trades")
    print(f"   Generated SHAP data: {len(shap_data)} features")
    print(f"   Generated cycle metrics: {len(cycle_metrics)} cycles")
    
    # Create configuration
    config = create_default_config("./")
    config.nickname = "Demo Strategy"
    config.REPORT_LABEL = "V2.1_DEMO"
    
    # Create report generator
    print("\n📋 Initializing Report Generator...")
    report_gen = ReportGenerator(config)
    
    # Generate comprehensive report
    print("⚙️  Generating comprehensive report...")
    metrics = report_gen.generate_full_report(
        trades_df=trades_df,
        equity_curve=equity_curve,
        cycle_metrics=cycle_metrics,
        aggregated_shap=shap_data,
        last_classification_report="precision    recall  f1-score   support\n\n           0       0.85      0.82      0.83        50\n           1       0.78      0.81      0.79        45\n\n    accuracy                           0.82        95"
    )
    
    print(f"\n✅ Report generation completed!")
    
    # Display key metrics
    if metrics:
        print(f"\n📈 Key Performance Metrics:")
        print(f"   Total Return: {metrics.get('net_profit_pct', 0):.2%}")
        print(f"   CAGR: {metrics.get('cagr', 0):.2%}")
        print(f"   Sharpe Ratio: {metrics.get('sharpe_ratio', 0):.2f}")
        print(f"   Max Drawdown: {metrics.get('max_drawdown_pct', 0):.2%}")
        print(f"   Win Rate: {metrics.get('win_rate', 0):.2%}")
        print(f"   Total Trades: {metrics.get('total_trades', 0)}")
        print(f"   Profit Factor: {metrics.get('profit_factor', 0):.2f}")
    
    return metrics

def demonstrate_visualization_generation():
    """Demonstrate visualization generation"""
    print("\n" + "=" * 80)
    print("VISUALIZATION GENERATION DEMONSTRATION")
    print("=" * 80)
    
    # Create sample data with more volatility for interesting charts
    print("📊 Creating sample data for visualizations...")
    equity_curve, trades_df = create_sample_trading_data(365)
    
    # Add some dramatic events to make charts more interesting
    crash_day = len(equity_curve) // 2
    equity_curve.iloc[crash_day:crash_day+5] *= 0.9  # 10% drawdown event
    
    recovery_day = crash_day + 20
    equity_curve.iloc[recovery_day:] *= 1.15  # Recovery
    
    print(f"   Enhanced equity curve with {len(equity_curve)} days")
    print(f"   Trade data with {len(trades_df)} trades")
    
    # Create configuration
    config = create_default_config("./")
    config.nickname = "Visualization Demo"
    
    # Create report generator
    print("\n📈 Generating individual visualizations...")
    report_gen = ReportGenerator(config)
    
    # Generate equity curve plot
    print("   📊 Creating equity curve plot...")
    report_gen.plot_equity_curve(equity_curve)
    
    # Generate SHAP plot
    print("   🧠 Creating SHAP importance plot...")
    shap_data = create_sample_shap_data()
    report_gen.plot_shap_summary(shap_data)
    
    # Generate trade analysis plots
    print("   📈 Creating trade analysis plots...")
    report_gen.plot_trade_analysis(trades_df)
    
    # Generate HTML dashboard
    print("   🌐 Creating interactive HTML dashboard...")
    metrics = report_gen._calculate_metrics(trades_df, equity_curve)
    report_gen.generate_html_report(metrics, trades_df, equity_curve, shap_data)
    
    print(f"\n✅ All visualizations generated!")
    print(f"   Check the 'Results' folder for generated plots and dashboard")
    
    return metrics

def demonstrate_custom_report_sections():
    """Demonstrate custom report sections and formatting"""
    print("\n" + "=" * 80)
    print("CUSTOM REPORT SECTIONS DEMONSTRATION")
    print("=" * 80)
    
    # Create sample data
    print("📊 Creating custom sample data...")
    equity_curve, trades_df = create_sample_trading_data(180)
    
    # Create configuration with custom settings
    config = create_default_config("./")
    config.nickname = "Custom Report Demo"
    config.REPORT_LABEL = "V3.0_CUSTOM"
    config.analysis_notes = "This is a demonstration of custom report generation with enhanced analytics and detailed performance breakdown."
    
    # Create report generator
    report_gen = ReportGenerator(config)
    
    # Calculate metrics
    print("\n📊 Calculating detailed metrics...")
    metrics = report_gen._calculate_metrics(trades_df, equity_curve)
    
    # Generate summary stats
    print("📋 Generating summary statistics...")
    summary = report_gen.generate_summary_stats(metrics)
    print("\n" + summary)
    
    # Create enhanced cycle metrics with more detail
    print("\n⚙️  Creating enhanced cycle breakdown...")
    enhanced_cycles = []
    for i in range(3):
        cycle = {
            'cycle': i + 1,
            'status': 'completed',
            'metrics': {
                'total_trades': np.random.randint(12, 20),
                'total_net_profit': np.random.uniform(200, 800),
                'win_rate': np.random.uniform(0.55, 0.75),
                'max_drawdown_pct': np.random.uniform(0.08, 0.18),
                'sharpe_ratio': np.random.uniform(1.2, 2.5),
                'mar_ratio': np.random.uniform(1.5, 4.0),
                'profit_factor': np.random.uniform(1.3, 2.8),
                'avg_win': np.random.uniform(80, 150),
                'avg_loss': np.random.uniform(-60, -30)
            }
        }
        enhanced_cycles.append(cycle)
    
    # Create detailed SHAP data
    detailed_shap = create_sample_shap_data()
    
    # Add framework memory for comparison
    framework_memory = {
        'historical_performance': {
            'best_sharpe': 2.1,
            'best_return': 0.35,
            'avg_drawdown': 0.12,
            'total_cycles': 15
        }
    }
    
    # Generate comprehensive custom report
    print("📋 Generating comprehensive custom report...")
    report_gen.generate_text_report(
        metrics=metrics,
        cycle_metrics=enhanced_cycles,
        aggregated_shap=detailed_shap,
        framework_memory=framework_memory,
        last_classification_report="Custom Classification Report:\n\nClass 0 (Strong Sell): precision=0.88, recall=0.85, f1=0.86\nClass 1 (Sell): precision=0.82, recall=0.79, f1=0.80\nClass 2 (Hold): precision=0.75, recall=0.78, f1=0.76\nClass 3 (Buy): precision=0.81, recall=0.83, f1=0.82\nClass 4 (Strong Buy): precision=0.89, recall=0.87, f1=0.88\n\nOverall Accuracy: 0.83"
    )
    
    print(f"\n✅ Custom report generation completed!")
    
    return metrics

def demonstrate_performance_analysis():
    """Demonstrate detailed performance analysis"""
    print("\n" + "=" * 80)
    print("PERFORMANCE ANALYSIS DEMONSTRATION")
    print("=" * 80)
    
    # Create multiple scenarios for comparison
    scenarios = {
        'Conservative': {'return_mean': 0.0005, 'return_std': 0.015, 'win_rate': 0.65},
        'Balanced': {'return_mean': 0.0008, 'return_std': 0.020, 'win_rate': 0.60},
        'Aggressive': {'return_mean': 0.0012, 'return_std': 0.030, 'win_rate': 0.55}
    }
    
    results = {}
    
    for scenario_name, params in scenarios.items():
        print(f"\n📊 Analyzing {scenario_name} Strategy...")
        
        # Generate scenario-specific data
        np.random.seed(hash(scenario_name) % 2**32)
        n_days = 252
        daily_returns = np.random.normal(params['return_mean'], params['return_std'], n_days)
        
        # Create equity curve
        initial_capital = 10000
        equity_values = [initial_capital]
        for ret in daily_returns:
            equity_values.append(equity_values[-1] * (1 + ret))
        
        dates = pd.date_range(start="2023-01-01", periods=n_days, freq="D")
        equity_curve = pd.Series(equity_values[1:], index=dates)
        
        # Generate trades based on win rate
        n_trades = 40
        trades_data = []
        for i in range(n_trades):
            if np.random.random() < params['win_rate']:
                pnl = np.random.uniform(50, 300)  # Win
            else:
                pnl = np.random.uniform(-250, -50)  # Loss
            
            trades_data.append({
                'trade_id': i + 1,
                'pnl': pnl,
                'entry_time': dates[i * (n_days // n_trades)]
            })
        
        trades_df = pd.DataFrame(trades_data)
        
        # Create report generator
        config = create_default_config("./")
        config.nickname = f"{scenario_name} Strategy"
        report_gen = ReportGenerator(config)
        
        # Calculate metrics
        metrics = report_gen._calculate_metrics(trades_df, equity_curve)
        results[scenario_name] = metrics
        
        # Display scenario results
        print(f"   📈 {scenario_name} Results:")
        print(f"      Total Return: {metrics.get('net_profit_pct', 0):.2%}")
        print(f"      Sharpe Ratio: {metrics.get('sharpe_ratio', 0):.2f}")
        print(f"      Max Drawdown: {metrics.get('max_drawdown_pct', 0):.2%}")
        print(f"      Win Rate: {metrics.get('win_rate', 0):.2%}")
    
    # Compare scenarios
    print(f"\n📊 SCENARIO COMPARISON:")
    print(f"{'Strategy':<12} {'Return':<8} {'Sharpe':<8} {'Drawdown':<10} {'Win Rate':<10}")
    print("-" * 60)
    
    for scenario_name, metrics in results.items():
        print(f"{scenario_name:<12} {metrics.get('net_profit_pct', 0):<8.1%} "
              f"{metrics.get('sharpe_ratio', 0):<8.2f} {metrics.get('max_drawdown_pct', 0):<10.1%} "
              f"{metrics.get('win_rate', 0):<10.1%}")
    
    # Find best strategy
    best_strategy = max(results.items(), key=lambda x: x[1].get('sharpe_ratio', 0))
    print(f"\n🏆 Best Strategy: {best_strategy[0]} (Sharpe: {best_strategy[1].get('sharpe_ratio', 0):.2f})")
    
    return results

def main():
    """Run all report generation demonstrations"""
    print("🚀 AxiomEdge Report Generation - Comprehensive Demonstration")
    print("=" * 80)
    
    try:
        # Basic demonstration
        basic_metrics = demonstrate_basic_report_generation()
        
        # Visualization demonstration
        viz_metrics = demonstrate_visualization_generation()
        
        # Custom report sections
        custom_metrics = demonstrate_custom_report_sections()
        
        # Performance analysis
        analysis_results = demonstrate_performance_analysis()
        
        print("\n" + "=" * 80)
        print("✅ ALL REPORT GENERATION DEMONSTRATIONS COMPLETED!")
        print("=" * 80)
        
        print("\n🎯 Summary:")
        if basic_metrics:
            print(f"   Basic Report Return: {basic_metrics.get('net_profit_pct', 0):.2%}")
        if viz_metrics:
            print(f"   Visualization Demo Return: {viz_metrics.get('net_profit_pct', 0):.2%}")
        if custom_metrics:
            print(f"   Custom Report Return: {custom_metrics.get('net_profit_pct', 0):.2%}")
        if analysis_results:
            best_analysis = max(analysis_results.items(), key=lambda x: x[1].get('sharpe_ratio', 0))
            print(f"   Best Analysis Strategy: {best_analysis[0]} ({best_analysis[1].get('sharpe_ratio', 0):.2f} Sharpe)")
        
        print("\n📚 Key Capabilities Demonstrated:")
        print("   ✅ Comprehensive performance metrics calculation")
        print("   ✅ Professional text report generation")
        print("   ✅ Matplotlib-based static visualizations")
        print("   ✅ Plotly-based interactive HTML dashboards")
        print("   ✅ SHAP feature importance visualization")
        print("   ✅ Trade analysis and breakdown")
        print("   ✅ Walk-forward cycle reporting")
        print("   ✅ Custom report sections and formatting")
        print("   ✅ Multi-scenario performance comparison")
        print("   ✅ Risk-adjusted performance metrics")
        
        print("\n📁 Generated Files:")
        print("   📊 Results/equity_curve.png - Equity curve and drawdown plots")
        print("   🧠 Results/shap_summary.png - Feature importance visualization")
        print("   📈 Results/trade_analysis.png - Trade distribution and analysis")
        print("   📋 Results/performance_report.txt - Comprehensive text report")
        print("   🌐 Results/performance_dashboard.html - Interactive dashboard")
        
        print("\n🔧 Next Steps:")
        print("   1. Customize report templates and styling")
        print("   2. Add real-time performance monitoring")
        print("   3. Implement automated report scheduling")
        print("   4. Create executive summary dashboards")
        print("   5. Add risk attribution analysis")
        
    except Exception as e:
        print(f"\n❌ Error during demonstration: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
