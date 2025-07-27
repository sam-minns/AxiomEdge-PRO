#!/usr/bin/env python3
# =============================================================================
# AXIOM EDGE - QUICK START GUIDE
# =============================================================================

"""
Quick start guide for the AxiomEdge framework.
This script shows the fastest way to get started with AxiomEdge.
"""

import sys
import os
import pandas as pd
import numpy as np
from datetime import datetime

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import AxiomEdge
import axiom_edge

def create_sample_data():
    """Create simple sample data for quick start"""
    print("📊 Creating sample market data...")
    
    # Generate 100 days of sample OHLCV data
    np.random.seed(42)
    dates = pd.date_range(start="2023-01-01", periods=100, freq="D")
    
    price = 100.0
    data = []
    
    for date in dates:
        # Simple random walk with slight upward bias
        price *= (1 + np.random.normal(0.001, 0.02))
        
        open_price = price * np.random.uniform(0.99, 1.01)
        high_price = max(open_price, price) * np.random.uniform(1.0, 1.02)
        low_price = min(open_price, price) * np.random.uniform(0.98, 1.0)
        volume = np.random.randint(100000, 500000)
        
        data.append({
            'timestamp': date,
            'Open': open_price,
            'High': high_price,
            'Low': low_price,
            'Close': price,
            'RealVolume': volume,
            'Symbol': 'DEMO'
        })
    
    df = pd.DataFrame(data)
    df.set_index('timestamp', inplace=True)
    
    print(f"   ✅ Created {len(df)} days of sample data")
    return df

def quick_start_demo():
    """Quick start demonstration"""
    print("🚀 AxiomEdge Quick Start Demo")
    print("=" * 50)
    
    # Step 1: Create sample data
    data = create_sample_data()
    
    # Step 2: Create default configuration
    print("\n⚙️  Creating default configuration...")
    config = axiom_edge.create_default_config("./")
    config.nickname = "Quick Start Demo"
    print(f"   ✅ Configuration created: {config.nickname}")
    
    # Step 3: Engineer features
    print("\n🔧 Engineering features...")
    feature_task = axiom_edge.FeatureEngineeringTask(config)
    features = feature_task.engineer_features(data)
    print(f"   ✅ Generated {len(features.columns)} features")
    
    # Step 4: Train a simple model
    print("\n🤖 Training model...")
    
    # Create a simple target (price goes up in next 3 days)
    future_return = features['Close'].shift(-3) / features['Close'] - 1
    target = (future_return > 0.01).astype(int)  # 1% gain target
    target.name = 'target_signal_pressure_class'
    
    # Clean data
    valid_mask = ~(target.isna() | features.isna().any(axis=1))
    clean_features = features[valid_mask]
    clean_target = target[valid_mask]
    
    if len(clean_features) > 20:  # Need minimum samples
        model_task = axiom_edge.ModelTrainingTask(config)
        results = model_task.train_model(clean_features, clean_target)
        
        if "error" not in results:
            print(f"   ✅ Model trained successfully!")
            print(f"   📊 F1 Score: {results['metrics']['f1_score']:.3f}")
            print(f"   📊 Accuracy: {results['metrics']['accuracy']:.3f}")
            print(f"   🎯 Features used: {len(results['selected_features'])}")
        else:
            print(f"   ⚠️  Model training had issues: {results['error']}")
    else:
        print(f"   ⚠️  Not enough clean data for training ({len(clean_features)} samples)")
    
    # Step 5: Generate a simple report
    print("\n📋 Generating report...")
    
    # Create simple trades for demonstration
    trades_data = []
    for i in range(5):
        trades_data.append({
            'trade_id': i + 1,
            'entry_time': data.index[i * 10],
            'exit_time': data.index[i * 10 + 5],
            'symbol': 'DEMO',
            'side': 'long',
            'entry_price': data['Close'].iloc[i * 10],
            'exit_price': data['Close'].iloc[i * 10 + 5],
            'position_size': 100,
            'pnl': (data['Close'].iloc[i * 10 + 5] - data['Close'].iloc[i * 10]) * 100,
            'duration': 5
        })
    
    trades_df = pd.DataFrame(trades_data)
    equity_curve = data['Close'] * 100  # Simple equity curve
    
    report_gen = axiom_edge.ReportGenerator(config)
    metrics = report_gen._calculate_metrics(trades_df, equity_curve)
    
    print(f"   ✅ Report generated!")
    print(f"   📈 Total Return: {metrics.get('net_profit_pct', 0):.2%}")
    print(f"   📊 Win Rate: {metrics.get('win_rate', 0):.2%}")
    
    print("\n🎉 Quick start demo completed!")
    print("\n📚 What you just did:")
    print("   ✅ Created sample market data")
    print("   ✅ Configured the framework")
    print("   ✅ Engineered 50+ technical features")
    print("   ✅ Trained a machine learning model")
    print("   ✅ Generated performance reports")
    
    print("\n🔧 Next steps:")
    print("   1. Try the complete_framework_demo.py for full capabilities")
    print("   2. Explore feature_engineering_demo.py for 200+ features")
    print("   3. Check model_training_demo.py for advanced ML")
    print("   4. Run genetic_programming_demo.py for strategy evolution")
    print("   5. Use your own data files with the framework")

def framework_info_demo():
    """Show framework information"""
    print("\n" + "=" * 50)
    print("FRAMEWORK INFORMATION")
    print("=" * 50)
    
    # Get framework info
    info = axiom_edge.get_framework_info()
    
    print(f"🚀 AxiomEdge Framework v{axiom_edge.__version__}")
    print(f"\n📦 Components ({len(info['components'])}):")
    for component in info['components']:
        print(f"   • {component}")
    
    print(f"\n⚡ Capabilities ({len(info['capabilities'])}):")
    for capability in info['capabilities']:
        print(f"   • {capability}")
    
    print(f"\n🔧 Available Features:")
    for feature, available in info['capabilities_available'].items():
        status = "✅" if available else "❌"
        print(f"   {status} {feature.replace('_', ' ').title()}")

def installation_check():
    """Check installation"""
    print("\n" + "=" * 50)
    print("INSTALLATION CHECK")
    print("=" * 50)
    
    validation = axiom_edge.validate_installation()
    
    if validation['installation_valid']:
        print("✅ Installation: VALID")
    else:
        print("❌ Installation: INVALID")
    
    print(f"\n📦 Core Modules:")
    for module, available in validation['core_modules'].items():
        status = "✅" if available else "❌"
        print(f"   {status} {module}")
    
    if validation['warnings']:
        print(f"\n⚠️  Warnings:")
        for warning in validation['warnings']:
            print(f"   • {warning}")
    
    if validation['errors']:
        print(f"\n❌ Errors:")
        for error in validation['errors']:
            print(f"   • {error}")

def main():
    """Main quick start function"""
    print("🚀 Welcome to AxiomEdge!")
    print("This is the quickest way to get started with the framework.")
    print()
    
    try:
        # Check installation first
        installation_check()
        
        # Show framework info
        framework_info_demo()
        
        # Run quick demo
        quick_start_demo()
        
        print("\n" + "=" * 50)
        print("🎉 QUICK START COMPLETED SUCCESSFULLY!")
        print("=" * 50)
        print("\nYou're now ready to explore the full AxiomEdge framework!")
        print("Check out the other example files for more advanced features.")
        
    except Exception as e:
        print(f"\n❌ Error during quick start: {e}")
        print("\nTroubleshooting:")
        print("1. Make sure all dependencies are installed")
        print("2. Check that you're in the correct directory")
        print("3. Try running: pip install -r requirements.txt")

if __name__ == "__main__":
    main()
