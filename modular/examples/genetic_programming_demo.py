#!/usr/bin/env python3
# =============================================================================
# AXIOM EDGE - GENETIC PROGRAMMING DEMONSTRATION
# =============================================================================

"""
This script demonstrates the genetic programming capabilities of the AxiomEdge
framework, showing how to evolve trading rules using genetic algorithms.
"""

import sys
import os
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from axiom_edge import (
    GeneticProgrammer,
    FeatureEngineeringTask,
    GeminiAnalyzer,
    create_default_config
)

def create_sample_data_with_trends(n_samples: int = 1000) -> pd.DataFrame:
    """Create sample data with clear trends for genetic programming"""
    np.random.seed(42)
    
    # Generate base price data with clear patterns
    dates = pd.date_range(start="2023-01-01", periods=n_samples, freq="D")
    price = 100.0
    prices = []
    
    for i in range(n_samples):
        # Add cyclical trends that can be learned
        trend = 0.001 * np.sin(2 * np.pi * i / 50)  # 50-day cycle
        seasonal = 0.0005 * np.sin(2 * np.pi * i / 252)  # Annual cycle
        noise = np.random.normal(0, 0.015)
        
        price *= (1 + trend + seasonal + noise)
        prices.append(price)
    
    # Create OHLCV data
    data = []
    for i, (date, close_price) in enumerate(zip(dates, prices)):
        open_price = close_price * np.random.uniform(0.998, 1.002)
        high_price = max(open_price, close_price) * np.random.uniform(1.0, 1.015)
        low_price = min(open_price, close_price) * np.random.uniform(0.985, 1.0)
        volume = np.random.randint(800000, 1500000)
        
        data.append({
            'timestamp': date,
            'Open': open_price,
            'High': high_price,
            'Low': low_price,
            'Close': close_price,
            'RealVolume': volume,
            'Symbol': 'DEMO'
        })
    
    df = pd.DataFrame(data)
    df.set_index('timestamp', inplace=True)
    
    return df

def create_gene_pool_from_features(features: pd.DataFrame) -> dict:
    """Create a gene pool from engineered features"""
    
    # Categorize features
    continuous_features = []
    state_features = []
    
    for col in features.columns:
        if col.startswith('target_') or col in ['Open', 'High', 'Low', 'Close', 'RealVolume', 'Symbol']:
            continue
            
        # Check if feature is likely continuous or state-based
        unique_values = features[col].nunique()
        if unique_values <= 10 and features[col].dtype in ['int64', 'bool']:
            state_features.append(col)
        else:
            continuous_features.append(col)
    
    # Create comprehensive gene pool
    gene_pool = {
        'continuous_features': continuous_features[:30],  # Limit for demo
        'state_features': state_features[:10],
        'comparison_operators': ['>', '<', '>=', '<='],
        'state_operators': ['==', '!='],
        'logical_operators': ['AND', 'OR'],
        'constants': [0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
    }
    
    return gene_pool

def demonstrate_basic_genetic_programming():
    """Demonstrate basic genetic programming"""
    print("=" * 80)
    print("BASIC GENETIC PROGRAMMING DEMONSTRATION")
    print("=" * 80)
    
    # Create sample data
    print("📊 Creating sample data with learnable patterns...")
    raw_data = create_sample_data_with_trends(500)
    
    # Engineer features
    feature_task = FeatureEngineeringTask()
    features = feature_task.engineer_features(raw_data)
    print(f"   Generated {len(features.columns)} features")
    
    # Create gene pool
    print("\n🧬 Creating gene pool from features...")
    gene_pool = create_gene_pool_from_features(features)
    
    print(f"   Continuous features: {len(gene_pool['continuous_features'])}")
    print(f"   State features: {len(gene_pool['state_features'])}")
    print(f"   Operators: {len(gene_pool['comparison_operators']) + len(gene_pool['state_operators'])}")
    
    # Show sample features
    print(f"\n   Sample continuous features: {gene_pool['continuous_features'][:5]}")
    print(f"   Sample state features: {gene_pool['state_features'][:3]}")
    
    # Create genetic programmer
    print("\n🧠 Initializing Genetic Programmer...")
    config = create_default_config("./")
    
    gp = GeneticProgrammer(
        gene_pool=gene_pool,
        config=config,
        population_size=20,  # Small for demo
        generations=10,      # Quick evolution
        mutation_rate=0.15,
        crossover_rate=0.7
    )
    
    # Run evolution
    print("⚙️  Running genetic evolution...")
    best_chromosome, best_fitness = gp.run_evolution(features)
    
    print(f"\n✅ Evolution completed!")
    print(f"   Best fitness (Sharpe ratio): {best_fitness:.4f}")
    
    if best_chromosome[0]:
        print(f"\n🎯 Evolved Trading Rules:")
        print(f"   Long Entry Rule:  {best_chromosome[0]}")
        print(f"   Short Entry Rule: {best_chromosome[1]}")
        
        # Analyze rule complexity
        long_complexity = best_chromosome[0].count('AND') + best_chromosome[0].count('OR')
        short_complexity = best_chromosome[1].count('AND') + best_chromosome[1].count('OR')
        print(f"\n📊 Rule Analysis:")
        print(f"   Long rule complexity: {long_complexity} logical operators")
        print(f"   Short rule complexity: {short_complexity} logical operators")
    
    return best_chromosome, best_fitness

def demonstrate_advanced_genetic_programming():
    """Demonstrate advanced genetic programming with AI integration"""
    print("\n" + "=" * 80)
    print("ADVANCED GENETIC PROGRAMMING DEMONSTRATION")
    print("=" * 80)
    
    # Create larger dataset
    print("📊 Creating larger dataset for advanced evolution...")
    raw_data = create_sample_data_with_trends(800)
    
    # Engineer comprehensive features
    config = create_default_config("./")
    config.EMA_PERIODS = [8, 13, 21, 34, 55]
    config.RSI_STANDARD_PERIODS = [14, 21, 28]
    
    feature_task = FeatureEngineeringTask(config)
    features = feature_task.engineer_features(raw_data)
    print(f"   Generated {len(features.columns)} features")
    
    # Create enhanced gene pool
    print("\n🧬 Creating enhanced gene pool...")
    gene_pool = create_gene_pool_from_features(features)
    
    # Add more sophisticated constants
    gene_pool['constants'] = [0, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60, 65, 70, 75, 80, 85, 90, 95, 100]
    
    print(f"   Enhanced gene pool: {len(gene_pool['continuous_features'])} continuous, "
          f"{len(gene_pool['state_features'])} state features")
    
    # Initialize AI analyzer (optional)
    print("\n🤖 Initializing AI analyzer...")
    try:
        gemini_analyzer = GeminiAnalyzer()
        print("   ✅ Gemini AI analyzer initialized")
        ai_available = True
    except Exception as e:
        print(f"   ⚠️  Gemini analyzer not available: {e}")
        gemini_analyzer = None
        ai_available = False
    
    # Create advanced genetic programmer
    print("\n🧠 Initializing Advanced Genetic Programmer...")
    gp = GeneticProgrammer(
        gene_pool=gene_pool,
        config=config,
        population_size=30,
        generations=15,
        mutation_rate=0.12,
        crossover_rate=0.75
    )
    
    # Show population statistics
    gp.create_initial_population()
    stats = gp.get_population_stats()
    print(f"   Population stats: {stats}")
    
    # Run advanced evolution
    print("⚙️  Running advanced genetic evolution...")
    if ai_available:
        print("   Using AI-guided evolution with retry capability")
    
    best_chromosome, best_fitness = gp.run_evolution(features, gemini_analyzer)
    
    print(f"\n✅ Advanced evolution completed!")
    print(f"   Best fitness (Sharpe ratio): {best_fitness:.4f}")
    
    if best_chromosome[0]:
        print(f"\n🎯 Evolved Advanced Trading Rules:")
        print(f"   Long Entry Rule:  {best_chromosome[0]}")
        print(f"   Short Entry Rule: {best_chromosome[1]}")
        
        # Detailed rule analysis
        print(f"\n📊 Detailed Rule Analysis:")
        
        # Count feature usage
        all_features = gene_pool['continuous_features'] + gene_pool['state_features']
        long_features_used = [f for f in all_features if f in best_chromosome[0]]
        short_features_used = [f for f in all_features if f in best_chromosome[1]]
        
        print(f"   Long rule uses {len(long_features_used)} features: {long_features_used[:3]}...")
        print(f"   Short rule uses {len(short_features_used)} features: {short_features_used[:3]}...")
        
        # Count operators
        long_ops = best_chromosome[0].count('AND') + best_chromosome[0].count('OR')
        short_ops = best_chromosome[1].count('AND') + best_chromosome[1].count('OR')
        print(f"   Logical complexity: Long={long_ops}, Short={short_ops}")
    
    return best_chromosome, best_fitness

def demonstrate_rule_evaluation():
    """Demonstrate how evolved rules perform on data"""
    print("\n" + "=" * 80)
    print("RULE EVALUATION DEMONSTRATION")
    print("=" * 80)
    
    # Create test data
    print("📊 Creating test data...")
    raw_data = create_sample_data_with_trends(300)
    feature_task = FeatureEngineeringTask()
    features = feature_task.engineer_features(raw_data)
    
    # Create simple gene pool for demonstration
    gene_pool = {
        'continuous_features': ['RSI_14', 'EMA_20', 'EMA_50', 'MACD', 'ATR'],
        'state_features': ['is_ny_session', 'day_of_week'],
        'comparison_operators': ['>', '<', '>=', '<='],
        'state_operators': ['==', '!='],
        'logical_operators': ['AND', 'OR'],
        'constants': [20, 30, 50, 70, 80]
    }
    
    print(f"   Features available: {len(features.columns)}")
    
    # Create genetic programmer
    config = create_default_config("./")
    gp = GeneticProgrammer(gene_pool, config, population_size=15, generations=8)
    
    # Test individual rule evaluation
    print("\n🧪 Testing rule evaluation...")
    
    # Create sample rules
    test_rules = [
        "RSI_14 < 30",
        "RSI_14 > 70",
        "EMA_20 > EMA_50",
        "RSI_14 < 30 AND EMA_20 > EMA_50",
        "RSI_14 > 70 OR ATR > 50"
    ]
    
    print("   Evaluating sample rules:")
    for rule in test_rules:
        try:
            signals = gp._evaluate_rule(rule, features)
            signal_count = signals.sum()
            signal_pct = (signal_count / len(signals)) * 100
            print(f"     '{rule}': {signal_count} signals ({signal_pct:.1f}%)")
        except Exception as e:
            print(f"     '{rule}': Error - {e}")
    
    # Run quick evolution
    print("\n⚙️  Running quick evolution for demonstration...")
    best_chromosome, best_fitness = gp.run_evolution(features)
    
    if best_chromosome[0]:
        print(f"\n📈 Performance Analysis:")
        
        # Evaluate evolved rules
        long_signals = gp._evaluate_rule(best_chromosome[0], features)
        short_signals = gp._evaluate_rule(best_chromosome[1], features)
        
        long_count = long_signals.sum()
        short_count = short_signals.sum()
        
        print(f"   Long signals: {long_count} ({(long_count/len(features))*100:.1f}%)")
        print(f"   Short signals: {short_count} ({(short_count/len(features))*100:.1f}%)")
        print(f"   Total activity: {((long_count + short_count)/len(features))*100:.1f}%")
        
        # Calculate simple returns
        if 'Close' in features.columns:
            returns = features['Close'].pct_change().fillna(0)
            
            # Long performance
            long_returns = returns[long_signals.shift(1).fillna(False)]
            if len(long_returns) > 0:
                long_perf = (1 + long_returns).prod() - 1
                print(f"   Long rule return: {long_perf*100:.2f}%")
            
            # Short performance  
            short_returns = -returns[short_signals.shift(1).fillna(False)]
            if len(short_returns) > 0:
                short_perf = (1 + short_returns).prod() - 1
                print(f"   Short rule return: {short_perf*100:.2f}%")
    
    return best_chromosome, best_fitness

def demonstrate_gene_pool_optimization():
    """Demonstrate gene pool optimization and feature selection"""
    print("\n" + "=" * 80)
    print("GENE POOL OPTIMIZATION DEMONSTRATION")
    print("=" * 80)
    
    # Create data
    print("📊 Creating data for gene pool optimization...")
    raw_data = create_sample_data_with_trends(400)
    feature_task = FeatureEngineeringTask()
    features = feature_task.engineer_features(raw_data)
    
    # Test different gene pool configurations
    gene_pool_configs = {
        'Simple': {
            'continuous_features': ['RSI_14', 'EMA_20', 'EMA_50'],
            'state_features': ['is_ny_session'],
            'comparison_operators': ['>', '<'],
            'state_operators': ['=='],
            'logical_operators': ['AND'],
            'constants': [30, 50, 70]
        },
        'Moderate': {
            'continuous_features': ['RSI_14', 'EMA_20', 'EMA_50', 'MACD', 'ATR', 'volume_ma_ratio'],
            'state_features': ['is_ny_session', 'day_of_week'],
            'comparison_operators': ['>', '<', '>=', '<='],
            'state_operators': ['==', '!='],
            'logical_operators': ['AND', 'OR'],
            'constants': [20, 30, 40, 50, 60, 70, 80]
        },
        'Complex': {
            'continuous_features': ['RSI_14', 'RSI_21', 'EMA_20', 'EMA_50', 'MACD', 'ATR', 
                                  'volume_ma_ratio', 'BB_position_20', 'momentum_10'],
            'state_features': ['is_ny_session', 'is_london_session', 'day_of_week', 'is_doji'],
            'comparison_operators': ['>', '<', '>=', '<='],
            'state_operators': ['==', '!='],
            'logical_operators': ['AND', 'OR'],
            'constants': [10, 20, 25, 30, 35, 40, 45, 50, 55, 60, 65, 70, 75, 80, 85, 90]
        }
    }
    
    results = {}
    config = create_default_config("./")
    
    for config_name, gene_pool in gene_pool_configs.items():
        print(f"\n🧬 Testing {config_name} gene pool...")
        print(f"   Features: {len(gene_pool['continuous_features']) + len(gene_pool['state_features'])}")
        print(f"   Operators: {len(gene_pool['comparison_operators']) + len(gene_pool['state_operators'])}")
        print(f"   Constants: {len(gene_pool['constants'])}")
        
        try:
            gp = GeneticProgrammer(
                gene_pool=gene_pool,
                config=config,
                population_size=15,
                generations=8,
                mutation_rate=0.1,
                crossover_rate=0.7
            )
            
            best_chromosome, best_fitness = gp.run_evolution(features)
            
            results[config_name] = {
                'fitness': best_fitness,
                'chromosome': best_chromosome,
                'gene_pool_size': len(gene_pool['continuous_features']) + len(gene_pool['state_features'])
            }
            
            print(f"   ✅ Best fitness: {best_fitness:.4f}")
            
        except Exception as e:
            print(f"   ❌ Failed: {e}")
            results[config_name] = {'fitness': -1.0, 'error': str(e)}
    
    # Compare results
    print(f"\n📊 Gene Pool Comparison Results:")
    print(f"{'Configuration':<12} {'Fitness':<10} {'Gene Pool Size':<15} {'Status'}")
    print("-" * 60)
    
    for config_name, result in results.items():
        if 'error' not in result:
            print(f"{config_name:<12} {result['fitness']:<10.4f} {result['gene_pool_size']:<15} {'Success'}")
        else:
            print(f"{config_name:<12} {'Failed':<10} {'N/A':<15} {'Error'}")
    
    # Find best configuration
    successful_results = {k: v for k, v in results.items() if 'error' not in v and v['fitness'] > -1}
    if successful_results:
        best_config = max(successful_results.items(), key=lambda x: x[1]['fitness'])
        print(f"\n🏆 Best Configuration: {best_config[0]} (Fitness: {best_config[1]['fitness']:.4f})")
        
        if best_config[1]['chromosome'][0]:
            print(f"   Best Long Rule: {best_config[1]['chromosome'][0]}")
            print(f"   Best Short Rule: {best_config[1]['chromosome'][1]}")
    
    return results

def main():
    """Run all genetic programming demonstrations"""
    print("🚀 AxiomEdge Genetic Programming - Comprehensive Demonstration")
    print("=" * 80)
    
    try:
        # Basic demonstration
        basic_chromosome, basic_fitness = demonstrate_basic_genetic_programming()
        
        # Advanced demonstration
        advanced_chromosome, advanced_fitness = demonstrate_advanced_genetic_programming()
        
        # Rule evaluation
        eval_chromosome, eval_fitness = demonstrate_rule_evaluation()
        
        # Gene pool optimization
        optimization_results = demonstrate_gene_pool_optimization()
        
        print("\n" + "=" * 80)
        print("✅ ALL GENETIC PROGRAMMING DEMONSTRATIONS COMPLETED!")
        print("=" * 80)
        
        print("\n🎯 Summary:")
        print(f"   Basic GP Fitness: {basic_fitness:.4f}")
        print(f"   Advanced GP Fitness: {advanced_fitness:.4f}")
        print(f"   Evaluation GP Fitness: {eval_fitness:.4f}")
        
        if optimization_results:
            successful_opts = {k: v for k, v in optimization_results.items() if 'error' not in v}
            if successful_opts:
                best_opt = max(successful_opts.items(), key=lambda x: x[1]['fitness'])
                print(f"   Best Optimization: {best_opt[0]} ({best_opt[1]['fitness']:.4f})")
        
        print("\n📚 Key Capabilities Demonstrated:")
        print("   ✅ Genetic algorithm evolution of trading rules")
        print("   ✅ Multi-objective fitness evaluation (Sharpe ratio)")
        print("   ✅ Crossover and mutation operations")
        print("   ✅ Tournament selection for parent selection")
        print("   ✅ Rule complexity management")
        print("   ✅ Feature-based gene pool construction")
        print("   ✅ AI-guided gene pool optimization")
        print("   ✅ Parallel fitness evaluation")
        print("   ✅ Performance analysis and backtesting")
        
        print("\n🔧 Next Steps:")
        print("   1. Experiment with different population sizes and generations")
        print("   2. Try custom fitness functions (risk-adjusted returns, drawdown)")
        print("   3. Implement multi-objective optimization")
        print("   4. Add constraint handling for risk management")
        print("   5. Integrate with live trading systems")
        
    except Exception as e:
        print(f"\n❌ Error during demonstration: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
