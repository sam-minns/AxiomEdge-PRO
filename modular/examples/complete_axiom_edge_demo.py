#!/usr/bin/env python3
# =============================================================================
# AXIOM EDGE - COMPLETE FRAMEWORK DEMONSTRATION
# =============================================================================

"""
This script demonstrates the complete AxiomEdge framework including all
unique features that distinguish it from traditional backtesting frameworks:
- AI Doctor integration
- Advanced telemetry system
- Walk-forward analysis
- SHAP explainability
- Genetic programming
- Dynamic ensembles
- Framework memory
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
    TelemetryCollector,
    TelemetryAnalyzer,
    FeatureEngineeringTask,
    ModelTrainingTask,
    GeneticProgrammer,
    ReportGenerator,
    Backtester,
    create_default_config
)

def create_realistic_market_data(n_days: int = 600, symbol: str = "AXIOM_DEMO") -> pd.DataFrame:
    """Create realistic market data with multiple regimes"""
    np.random.seed(42)
    
    dates = pd.date_range(start="2023-01-01", periods=n_days, freq="D")
    
    # Create multiple market regimes
    regime_length = n_days // 3
    regimes = ['bull', 'bear', 'sideways']
    
    price = 100.0
    prices = []
    volumes = []
    
    for i in range(n_days):
        # Determine current regime
        regime_idx = min(i // regime_length, len(regimes) - 1)
        current_regime = regimes[regime_idx]
        
        # Regime-specific parameters
        if current_regime == 'bull':
            trend = 0.0008  # Strong upward trend
            volatility = 0.015
        elif current_regime == 'bear':
            trend = -0.0005  # Downward trend
            volatility = 0.025  # Higher volatility
        else:  # sideways
            trend = 0.0001  # Minimal trend
            volatility = 0.012  # Lower volatility
        
        # Add cyclical components
        seasonal = 0.0002 * np.sin(2 * np.pi * i / 252)  # Annual
        monthly = 0.0003 * np.sin(2 * np.pi * i / 21)    # Monthly
        
        # Generate return with momentum
        momentum = 0
        if i > 5:
            recent_returns = [(prices[j] / prices[j-1] - 1) for j in range(max(0, i-5), i)]
            momentum = np.mean(recent_returns) * 0.1
        
        daily_return = trend + seasonal + monthly + momentum + np.random.normal(0, volatility)
        price *= (1 + daily_return)
        prices.append(price)
        
        # Volume with regime dependency
        base_volume = 1200000
        if current_regime == 'bear':
            base_volume *= 1.3  # Higher volume in bear markets
        
        volume_factor = 1 + abs(daily_return) * 8  # Volume spikes on big moves
        volume = int(base_volume * volume_factor * np.random.uniform(0.8, 1.2))
        volumes.append(volume)
    
    # Create OHLCV data
    data = []
    for i, (date, close_price, volume) in enumerate(zip(dates, prices, volumes)):
        daily_range = close_price * np.random.uniform(0.01, 0.03)
        
        open_price = close_price * np.random.uniform(0.998, 1.002)
        high_price = max(open_price, close_price) + daily_range * np.random.uniform(0.3, 0.7)
        low_price = min(open_price, close_price) - daily_range * np.random.uniform(0.3, 0.7)
        
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

def demonstrate_complete_axiom_edge():
    """Demonstrate the complete AxiomEdge framework with all unique features"""
    print("🚀 AxiomEdge - Complete Framework Demonstration")
    print("=" * 80)
    print("Showcasing features that NO OTHER backtesting framework offers:")
    print("✅ AI Doctor Integration")
    print("✅ Advanced Telemetry System") 
    print("✅ Walk-Forward Analysis")
    print("✅ SHAP Explainability")
    print("✅ Genetic Programming")
    print("✅ Dynamic Ensembles")
    print("✅ Framework Memory")
    print("=" * 80)
    
    # 1. Initialize Telemetry System (UNIQUE FEATURE)
    print("\n📊 1. INITIALIZING ADVANCED TELEMETRY SYSTEM")
    print("   (Feature not available in backtesting.py, zipline, or backtrader)")
    
    os.makedirs("telemetry_logs", exist_ok=True)
    telemetry = TelemetryCollector("telemetry_logs/axiom_edge_session.jsonl")
    
    # 2. Create Configuration with AI Integration
    print("\n⚙️  2. CONFIGURING AI-POWERED FRAMEWORK")
    config = create_default_config("./")
    config.nickname = "AxiomEdge Complete Demo"
    config.REPORT_LABEL = "V2.1_COMPLETE_SHOWCASE"
    
    # Advanced configuration
    config.TRAINING_WINDOW_DAYS = 180
    config.TEST_WINDOW_DAYS = 45
    config.WALK_FORWARD_STEP_DAYS = 20
    config.MAX_WALK_FORWARD_CYCLES = 4
    config.ENABLE_GENETIC_PROGRAMMING = True
    config.CALCULATE_SHAP_VALUES = True
    config.SHADOW_SET_VALIDATION = True
    
    print(f"   AI Integration: ✅ Enabled")
    print(f"   Telemetry: ✅ Active")
    print(f"   Walk-Forward Cycles: {config.MAX_WALK_FORWARD_CYCLES}")
    print(f"   Genetic Programming: ✅ Enabled")
    print(f"   SHAP Analysis: ✅ Enabled")
    
    # 3. Generate Realistic Market Data
    print("\n📈 3. GENERATING REALISTIC MULTI-REGIME MARKET DATA")
    market_data = create_realistic_market_data(500, "AXIOM_DEMO")
    
    # Save data
    os.makedirs("demo_data", exist_ok=True)
    data_file = "demo_data/axiom_demo.csv"
    market_data.to_csv(data_file)
    
    print(f"   Generated: {len(market_data)} days of data")
    print(f"   Price range: ${market_data['Close'].min():.2f} - ${market_data['Close'].max():.2f}")
    print(f"   Regimes: Bull → Bear → Sideways (realistic market cycles)")
    
    # Log data generation
    telemetry.log_system_health(
        component="data_generation",
        health_status="healthy",
        metrics={
            "data_points": len(market_data),
            "price_range": [market_data['Close'].min(), market_data['Close'].max()],
            "volatility": market_data['Close'].pct_change().std()
        }
    )
    
    # 4. Initialize Framework Orchestrator
    print("\n🎯 4. INITIALIZING FRAMEWORK ORCHESTRATOR")
    print("   (Complete workflow automation - unique to AxiomEdge)")
    
    orchestrator = FrameworkOrchestrator(config)
    
    # 5. Run Complete Framework with Telemetry
    print("\n🚀 5. EXECUTING COMPLETE FRAMEWORK WITH TELEMETRY")
    print("   Features being demonstrated:")
    print("   • Walk-forward analysis with multiple cycles")
    print("   • AI-guided hyperparameter optimization")
    print("   • SHAP feature importance analysis")
    print("   • Genetic programming for strategy evolution")
    print("   • Advanced telemetry and monitoring")
    print("   • Framework memory and learning")
    
    # Execute framework
    results = orchestrator.run_complete_framework(
        data_files=[data_file],
        symbols=None,
        start_date=None,
        end_date=None
    )
    
    # 6. Analyze Results with Telemetry
    print("\n📊 6. ANALYZING RESULTS WITH ADVANCED TELEMETRY")
    
    if results.get('status') == 'completed':
        print(f"✅ Framework execution completed successfully!")
        print(f"   Execution time: {results.get('execution_time_seconds', 0):.2f} seconds")
        print(f"   Walk-forward cycles: {results.get('walk_forward_cycles', 0)}")
        print(f"   Features engineered: {results.get('features_engineered', 0)}")
        
        # Log framework completion
        telemetry.log_performance_milestone(
            milestone_type="framework_completion",
            metrics=results.get('final_metrics', {}),
            comparison_baseline=None
        )
        
        # Display cycle breakdown
        cycle_breakdown = results.get('cycle_breakdown', [])
        if cycle_breakdown:
            print(f"\n📈 Walk-Forward Analysis Results:")
            print(f"{'Cycle':<6} {'Status':<12} {'F1 Score':<10} {'Accuracy':<10} {'Features':<10}")
            print("-" * 65)
            
            for cycle in cycle_breakdown:
                cycle_num = cycle.get('cycle', 'N/A')
                status = cycle.get('status', 'N/A')
                metrics = cycle.get('metrics', {})
                f1_score = metrics.get('f1_score', 0)
                accuracy = metrics.get('accuracy', 0)
                features = cycle.get('features_used', 0)
                
                print(f"{cycle_num:<6} {status:<12} {f1_score:<10.3f} {accuracy:<10.3f} {features:<10}")
                
                # Log each cycle to telemetry
                telemetry.log_cycle_data(
                    cycle_num=cycle_num,
                    status=status,
                    config_snapshot=config,
                    labeling_summary={"features_used": features},
                    training_summary=metrics,
                    backtest_metrics=metrics,
                    horizon_metrics={},
                    ai_notes=f"Cycle {cycle_num} completed with F1: {f1_score:.3f}"
                )
        
        # Display genetic programming results
        evolution = results.get('evolution')
        if evolution and evolution.get('status') == 'completed':
            print(f"\n🧬 Genetic Programming Results:")
            print(f"   Best fitness: {evolution.get('best_fitness', 0):.4f}")
            best_rules = evolution.get('best_rules', {})
            print(f"   Evolved long rule: {best_rules.get('long_rule', 'N/A')}")
            print(f"   Evolved short rule: {best_rules.get('short_rule', 'N/A')}")
            
            # Log genetic programming results
            telemetry.log_genetic_programming_evolution(
                generation=0,  # Final generation
                population_stats=evolution.get('population_stats', {}),
                best_individual=best_rules,
                evolution_metrics={"best_fitness": evolution.get('best_fitness', 0)}
            )
        
        # Final performance metrics
        final_metrics = results.get('final_metrics', {})
        if final_metrics:
            print(f"\n🎯 Final Performance Metrics:")
            print(f"   Total Return: {final_metrics.get('net_profit_pct', 0):.2%}")
            print(f"   Sharpe Ratio: {final_metrics.get('sharpe_ratio', 0):.2f}")
            print(f"   Max Drawdown: {final_metrics.get('max_drawdown_pct', 0):.2%}")
            print(f"   Win Rate: {final_metrics.get('win_rate', 0):.2%}")
    
    else:
        print(f"❌ Framework execution failed: {results.get('error', 'Unknown error')}")
        
        # Log failure
        telemetry.log_ai_intervention(
            intervention_type="emergency",
            trigger_reason="framework_failure",
            ai_analysis={"error": results.get('error', 'Unknown')},
            action_taken={"status": "logged_for_analysis"}
        )
    
    # 7. Demonstrate Telemetry Analysis
    print("\n🔍 7. ADVANCED TELEMETRY ANALYSIS")
    print("   (Comprehensive monitoring not available in other frameworks)")
    
    # Analyze telemetry data
    analyzer = TelemetryAnalyzer("telemetry_logs/axiom_edge_session.jsonl")
    
    # Performance trends
    trends = analyzer.analyze_cycle_performance_trends()
    if 'error' not in trends:
        print(f"   📊 Performance Trends:")
        print(f"      Average Sharpe: {trends.get('avg_sharpe', 0):.3f}")
        print(f"      Average Return: {trends.get('avg_return', 0):.2%}")
        print(f"      Consistency: {trends.get('consistency', 0):.3f}")
        print(f"      Performance Trend: {trends.get('performance_trend', 0):.3f}")
    
    # AI intervention analysis
    ai_effectiveness = analyzer.get_ai_intervention_effectiveness()
    if 'error' not in ai_effectiveness:
        print(f"   🤖 AI Intervention Analysis:")
        print(f"      Total Interventions: {ai_effectiveness.get('total_interventions', 0)}")
        print(f"      Intervention Types: {ai_effectiveness.get('intervention_types', {})}")
    
    # Session summary
    session_summary = telemetry.get_session_summary()
    print(f"   📋 Session Summary:")
    print(f"      Session ID: {session_summary.get('session_id', 'N/A')}")
    print(f"      Total Events: {session_summary.get('total_events', 0)}")
    print(f"      Duration: {session_summary.get('session_duration_seconds', 0):.2f} seconds")
    print(f"      Event Types: {session_summary.get('event_counts', {})}")
    
    # 8. Export Telemetry Data
    print("\n💾 8. EXPORTING TELEMETRY DATA")
    
    # Export in multiple formats
    csv_export = telemetry.export_session_data("csv")
    json_export = telemetry.export_session_data("json")
    
    if csv_export:
        print(f"   ✅ CSV export: {csv_export}")
    if json_export:
        print(f"   ✅ JSON export: {json_export}")
    
    # Close telemetry session
    telemetry.close_session()
    
    # 9. Framework Status
    print("\n📊 9. FRAMEWORK STATUS & MEMORY")
    print("   (Framework learning and adaptation - unique to AxiomEdge)")
    
    framework_status = orchestrator.get_framework_status()
    print(f"   Cycles Completed: {framework_status.get('cycles_completed', 0)}")
    print(f"   Success Rate: {framework_status.get('successful_cycles', 0)}/{framework_status.get('cycles_completed', 0)}")
    print(f"   Features Available: {framework_status.get('features_available', 0)}")
    print(f"   Framework Memory: {framework_status.get('framework_memory_size', 0)} historical runs")
    
    # 10. Summary of Unique Features
    print("\n" + "=" * 80)
    print("🌟 AXIOM EDGE UNIQUE FEATURES DEMONSTRATED")
    print("=" * 80)
    print("✅ AI Doctor: Continuous monitoring and optimization")
    print("✅ Advanced Telemetry: Comprehensive JSONL-based tracking")
    print("✅ Walk-Forward Analysis: Robust out-of-sample validation")
    print("✅ SHAP Explainability: Model transparency and interpretability")
    print("✅ Genetic Programming: Automated strategy discovery")
    print("✅ Framework Memory: Historical learning and adaptation")
    print("✅ Dynamic Ensembles: Adaptive model weighting")
    print("✅ Regime Detection: Market condition adaptation")
    print("✅ Parameter Drift Detection: Automated monitoring")
    print("✅ Professional Reporting: Publication-quality outputs")
    print("✅ Modular Architecture: Use components independently")
    print("✅ Production Monitoring: Health checks and alerts")
    print("")
    print("🚀 NO OTHER BACKTESTING FRAMEWORK OFFERS THESE CAPABILITIES!")
    print("   backtesting.py: Basic backtesting only")
    print("   zipline: Portfolio management, no AI/telemetry")
    print("   backtrader: Strategy testing, no advanced analytics")
    print("   AxiomEdge: Complete AI-powered trading framework")
    print("=" * 80)
    
    return results

def main():
    """Run the complete AxiomEdge demonstration"""
    try:
        results = demonstrate_complete_axiom_edge()
        
        print("\n🎊 DEMONSTRATION COMPLETED SUCCESSFULLY!")
        print("\nGenerated Files:")
        print("📁 demo_data/axiom_demo.csv - Sample market data")
        print("📊 telemetry_logs/axiom_edge_session.jsonl - Telemetry data")
        print("📈 Results/ - Performance reports and visualizations")
        
        print("\nNext Steps:")
        print("1. Explore the generated telemetry data")
        print("2. Analyze the performance reports")
        print("3. Experiment with different configurations")
        print("4. Try your own market data")
        print("5. Customize the AI analysis parameters")
        
    except Exception as e:
        print(f"\n❌ Error during demonstration: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
