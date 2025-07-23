#!/usr/bin/env python3
# =============================================================================
# AXIOM EDGE - FEATURE ENGINEERING DEMONSTRATION
# =============================================================================

"""
This script demonstrates the comprehensive feature engineering capabilities
of the AxiomEdge framework, showcasing 200+ technical, statistical, and
behavioral features.
"""

import sys
import os
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from axiom_edge import (
    FeatureEngineeringTask,
    FeatureEngineer,
    ConfigModel,
    create_default_config
)

def create_realistic_market_data(n_days: int = 365) -> pd.DataFrame:
    """Create realistic market data for demonstration"""
    np.random.seed(42)
    
    # Generate dates
    dates = pd.date_range(start="2023-01-01", periods=n_days, freq="D")
    
    # Generate realistic price series with trends and volatility clustering
    price = 100.0
    prices = []
    volumes = []
    
    for i in range(n_days):
        # Add trend component
        trend = 0.0001 * np.sin(2 * np.pi * i / 252)  # Annual cycle
        
        # Add volatility clustering
        vol_base = 0.02
        vol_cluster = vol_base * (1 + 0.5 * np.sin(2 * np.pi * i / 50))
        
        # Generate price movement
        daily_return = trend + np.random.normal(0, vol_cluster)
        price *= (1 + daily_return)
        prices.append(price)
        
        # Generate volume with some correlation to volatility
        base_volume = 1000000
        volume_factor = 1 + abs(daily_return) * 10  # Higher volume on big moves
        volume = int(base_volume * volume_factor * np.random.uniform(0.5, 1.5))
        volumes.append(volume)
    
    # Create OHLC data
    data = []
    for i, (date, close_price, volume) in enumerate(zip(dates, prices, volumes)):
        # Generate realistic OHLC from close price
        daily_range = close_price * np.random.uniform(0.005, 0.03)  # 0.5% to 3% daily range
        
        open_price = close_price * np.random.uniform(0.995, 1.005)
        high_price = max(open_price, close_price) + daily_range * np.random.uniform(0, 0.5)
        low_price = min(open_price, close_price) - daily_range * np.random.uniform(0, 0.5)
        
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

def demonstrate_basic_feature_engineering():
    """Demonstrate basic feature engineering using the task interface"""
    print("=" * 80)
    print("BASIC FEATURE ENGINEERING DEMONSTRATION")
    print("=" * 80)
    
    # Create sample data
    print("📊 Creating realistic market data...")
    data = create_realistic_market_data(365)
    print(f"   Generated {len(data)} days of OHLCV data")
    print(f"   Price range: ${data['Close'].min():.2f} - ${data['Close'].max():.2f}")
    print(f"   Average volume: {data['RealVolume'].mean():,.0f}")
    
    # Create feature engineering task
    print("\n🔧 Initializing feature engineering task...")
    task = FeatureEngineeringTask()
    
    # Engineer features
    print("⚙️  Engineering features...")
    features = task.engineer_features(data)
    
    print(f"\n✅ Feature engineering completed!")
    print(f"   Original columns: {len(data.columns)}")
    print(f"   Engineered features: {len(features.columns)}")
    print(f"   Total features added: {len(features.columns) - len(data.columns)}")
    
    # Analyze feature categories
    feature_categories = analyze_feature_categories(features.columns.tolist())
    print(f"\n📈 Feature Categories:")
    for category, count in feature_categories.items():
        print(f"   {category}: {count} features")
    
    return features

def demonstrate_advanced_feature_engineering():
    """Demonstrate advanced feature engineering with custom configuration"""
    print("\n" + "=" * 80)
    print("ADVANCED FEATURE ENGINEERING DEMONSTRATION")
    print("=" * 80)
    
    # Create custom configuration
    print("⚙️  Creating custom configuration...")
    config = create_default_config("./")
    
    # Customize feature engineering parameters
    config.EMA_PERIODS = [8, 13, 21, 34, 55, 89, 144, 233]  # Fibonacci sequence
    config.RSI_STANDARD_PERIODS = [14, 21, 28]
    config.BOLLINGER_PERIOD = 20
    config.STOCHASTIC_PERIOD = 14
    
    print(f"   EMA Periods: {config.EMA_PERIODS}")
    print(f"   RSI Periods: {config.RSI_STANDARD_PERIODS}")
    
    # Create multi-timeframe data
    print("\n📊 Creating multi-timeframe data...")
    daily_data = create_realistic_market_data(365)
    
    # Simulate weekly data (every 7th day)
    weekly_data = daily_data.iloc[::7].copy()
    weekly_data['Symbol'] = 'DEMO'
    
    print(f"   Daily data: {len(daily_data)} records")
    print(f"   Weekly data: {len(weekly_data)} records")
    
    # Create feature engineer directly
    print("\n🔧 Initializing advanced feature engineer...")
    timeframe_roles = {'base': 'D1', 'higher': 'W1'}
    playbook = {}
    
    feature_engineer = FeatureEngineer(config, timeframe_roles, playbook)
    
    # Prepare multi-timeframe data
    data_by_tf = {
        'D1': daily_data,
        'W1': weekly_data
    }
    
    # Engineer features
    print("⚙️  Engineering advanced features with multi-timeframe analysis...")
    features = feature_engineer.engineer_features(daily_data, data_by_tf, macro_data=None)
    
    print(f"\n✅ Advanced feature engineering completed!")
    print(f"   Final dataset shape: {features.shape}")
    
    # Show sample of advanced features
    advanced_features = [col for col in features.columns if any(x in col for x in ['EMA_', 'RSI_', 'BB_', 'MACD', 'Stoch_', 'W1'])]
    print(f"\n🎯 Sample of advanced features ({len(advanced_features)} total):")
    for i, feature in enumerate(advanced_features[:15]):
        print(f"   {i+1:2d}. {feature}")
    if len(advanced_features) > 15:
        print(f"   ... and {len(advanced_features) - 15} more")
    
    return features

def demonstrate_feature_labeling():
    """Demonstrate multi-task labeling capabilities"""
    print("\n" + "=" * 80)
    print("MULTI-TASK LABELING DEMONSTRATION")
    print("=" * 80)
    
    # Create data with more volatility for better label examples
    print("📊 Creating volatile market data for labeling...")
    np.random.seed(123)  # Different seed for more volatility
    data = create_realistic_market_data(200)
    
    # Add some artificial volatility spikes
    spike_indices = np.random.choice(len(data), size=10, replace=False)
    for idx in spike_indices:
        if idx < len(data) - 1:
            data.iloc[idx, data.columns.get_loc('High')] *= 1.05
            data.iloc[idx, data.columns.get_loc('Low')] *= 0.95
    
    print(f"   Generated {len(data)} days with artificial volatility spikes")
    
    # Create feature engineer
    config = create_default_config("./")
    config.LOOKAHEAD_CANDLES = 5  # Look 5 days ahead for labels
    
    timeframe_roles = {'base': 'D1'}
    feature_engineer = FeatureEngineer(config, timeframe_roles, {})
    
    # First engineer basic features (needed for some labeling)
    print("\n⚙️  Engineering basic features for labeling...")
    data_with_features = feature_engineer._calculate_technical_indicators(data)
    
    # Generate labels
    print("🏷️  Generating multi-task labels...")
    labeled_data = feature_engineer.label_data_multi_task(data_with_features)
    
    # Analyze labels
    print(f"\n✅ Labeling completed!")
    
    label_columns = [col for col in labeled_data.columns if col.startswith('target_')]
    print(f"   Generated {len(label_columns)} label types:")
    
    for label_col in label_columns:
        if labeled_data[label_col].dtype in ['int64', 'int32']:
            value_counts = labeled_data[label_col].value_counts().sort_index()
            print(f"   📊 {label_col}:")
            for value, count in value_counts.items():
                print(f"      {value}: {count} samples ({count/len(labeled_data)*100:.1f}%)")
        else:
            print(f"   📊 {label_col}: mean={labeled_data[label_col].mean():.3f}, std={labeled_data[label_col].std():.3f}")
    
    return labeled_data

def analyze_feature_categories(feature_names: list) -> dict:
    """Analyze and categorize features"""
    categories = {
        "Price & Returns": 0,
        "Technical Indicators": 0,
        "Volume Analysis": 0,
        "Statistical Moments": 0,
        "Time-based": 0,
        "Volatility": 0,
        "Pattern Recognition": 0,
        "Advanced Analytics": 0,
        "Multi-timeframe": 0,
        "Other": 0
    }
    
    for feature in feature_names:
        feature_lower = feature.lower()
        
        if any(x in feature_lower for x in ['price', 'return', 'pct_change', 'momentum', 'roc']):
            categories["Price & Returns"] += 1
        elif any(x in feature_lower for x in ['rsi', 'macd', 'ema', 'sma', 'bb_', 'stoch', 'atr']):
            categories["Technical Indicators"] += 1
        elif any(x in feature_lower for x in ['volume', 'ad_line', 'pv_corr']):
            categories["Volume Analysis"] += 1
        elif any(x in feature_lower for x in ['mean', 'std', 'skew', 'kurt', 'mad']):
            categories["Statistical Moments"] += 1
        elif any(x in feature_lower for x in ['hour', 'day', 'month', 'week', 'session', 'sin', 'cos']):
            categories["Time-based"] += 1
        elif any(x in feature_lower for x in ['volatility', 'atr', 'parkinson', 'yang_zhang']):
            categories["Volatility"] += 1
        elif any(x in feature_lower for x in ['doji', 'engulf', 'wick', 'body', 'candle']):
            categories["Pattern Recognition"] += 1
        elif any(x in feature_lower for x in ['entropy', 'cycle', 'fourier', 'wavelet', 'hurst']):
            categories["Advanced Analytics"] += 1
        elif any(x in feature_lower for x in ['w1', 'h4', 'h1', 'm30', 'm15', 'm5']):
            categories["Multi-timeframe"] += 1
        else:
            categories["Other"] += 1
    
    return categories

def show_feature_quality_analysis(features: pd.DataFrame):
    """Analyze the quality of engineered features"""
    print("\n" + "=" * 80)
    print("FEATURE QUALITY ANALYSIS")
    print("=" * 80)
    
    # Basic statistics
    print("📊 Basic Statistics:")
    print(f"   Total features: {len(features.columns)}")
    print(f"   Total samples: {len(features)}")
    
    # Missing values analysis
    missing_stats = features.isnull().sum()
    features_with_missing = missing_stats[missing_stats > 0]
    
    print(f"\n🔍 Missing Values Analysis:")
    print(f"   Features with missing values: {len(features_with_missing)}")
    print(f"   Total missing values: {missing_stats.sum()}")
    
    if len(features_with_missing) > 0:
        print("   Top 10 features with most missing values:")
        for i, (feature, missing_count) in enumerate(features_with_missing.head(10).items()):
            pct_missing = (missing_count / len(features)) * 100
            print(f"   {i+1:2d}. {feature}: {missing_count} ({pct_missing:.1f}%)")
    
    # Feature variance analysis
    numeric_features = features.select_dtypes(include=[np.number])
    feature_variance = numeric_features.var().sort_values(ascending=False)
    
    print(f"\n📈 Feature Variance Analysis:")
    print("   Top 10 most variable features:")
    for i, (feature, variance) in enumerate(feature_variance.head(10).items()):
        print(f"   {i+1:2d}. {feature}: {variance:.6f}")
    
    print("\n   Top 10 least variable features:")
    for i, (feature, variance) in enumerate(feature_variance.tail(10).items()):
        print(f"   {i+1:2d}. {feature}: {variance:.6f}")
    
    # Correlation analysis
    print(f"\n🔗 Feature Correlation Analysis:")
    correlation_matrix = numeric_features.corr()
    
    # Find highly correlated feature pairs
    high_corr_pairs = []
    for i in range(len(correlation_matrix.columns)):
        for j in range(i+1, len(correlation_matrix.columns)):
            corr_val = correlation_matrix.iloc[i, j]
            if abs(corr_val) > 0.9:  # High correlation threshold
                high_corr_pairs.append((correlation_matrix.columns[i], correlation_matrix.columns[j], corr_val))
    
    print(f"   Highly correlated feature pairs (|r| > 0.9): {len(high_corr_pairs)}")
    if high_corr_pairs:
        print("   Top 10 highest correlations:")
        high_corr_pairs.sort(key=lambda x: abs(x[2]), reverse=True)
        for i, (feat1, feat2, corr) in enumerate(high_corr_pairs[:10]):
            print(f"   {i+1:2d}. {feat1} <-> {feat2}: {corr:.3f}")

def main():
    """Run all feature engineering demonstrations"""
    print("🚀 AxiomEdge Feature Engineering - Comprehensive Demonstration")
    print("=" * 80)
    
    try:
        # Basic demonstration
        basic_features = demonstrate_basic_feature_engineering()
        
        # Advanced demonstration
        advanced_features = demonstrate_advanced_feature_engineering()
        
        # Labeling demonstration
        labeled_data = demonstrate_feature_labeling()
        
        # Quality analysis
        show_feature_quality_analysis(advanced_features)
        
        print("\n" + "=" * 80)
        print("✅ ALL DEMONSTRATIONS COMPLETED SUCCESSFULLY!")
        print("=" * 80)
        
        print("\n🎯 Summary:")
        print(f"   Basic features generated: {len(basic_features.columns)} columns")
        print(f"   Advanced features generated: {len(advanced_features.columns)} columns")
        print(f"   Labeled data samples: {len(labeled_data)} rows")
        
        print("\n📚 Key Capabilities Demonstrated:")
        print("   ✅ 200+ technical and statistical features")
        print("   ✅ Multi-timeframe feature fusion")
        print("   ✅ Advanced pattern recognition")
        print("   ✅ Multi-task labeling system")
        print("   ✅ Feature quality analysis")
        print("   ✅ Configurable parameters")
        
        print("\n🔧 Next Steps:")
        print("   1. Experiment with different configuration parameters")
        print("   2. Add your own custom features")
        print("   3. Use features for model training")
        print("   4. Implement feature selection strategies")
        
    except Exception as e:
        print(f"\n❌ Error during demonstration: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
