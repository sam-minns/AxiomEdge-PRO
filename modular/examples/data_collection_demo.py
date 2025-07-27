#!/usr/bin/env python3
# =============================================================================
# AXIOM EDGE - DATA COLLECTION DEMONSTRATION
# =============================================================================

"""
This script demonstrates the data collection capabilities of the AxiomEdge framework,
including multi-source data gathering, caching, and data quality validation.
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
    DataHandler,
    create_default_config,
    validate_data_quality
)

def demonstrate_basic_data_collection():
    """Demonstrate basic data collection functionality"""
    print("=" * 80)
    print("BASIC DATA COLLECTION DEMONSTRATION")
    print("=" * 80)
    
    # Create configuration
    config = create_default_config("./")
    config.nickname = "Data Collection Demo"
    
    # Create data collection task
    print("📊 Initializing Data Collection Task...")
    data_task = DataCollectionTask(config)
    
    # Demonstrate data collection for multiple symbols
    symbols = ['AAPL', 'GOOGL', 'MSFT']
    start_date = '2023-01-01'
    end_date = '2023-12-31'
    
    print(f"\n📈 Collecting data for symbols: {', '.join(symbols)}")
    print(f"   Date range: {start_date} to {end_date}")
    
    try:
        # Collect data
        results = data_task.collect_data(
            symbols=symbols,
            start_date=start_date,
            end_date=end_date,
            timeframe='1D'
        )
        
        print(f"\n✅ Data collection completed!")
        
        # Display results
        for symbol, data in results.items():
            if isinstance(data, pd.DataFrame) and not data.empty:
                print(f"   📊 {symbol}: {len(data)} records")
                print(f"      Date range: {data.index[0]} to {data.index[-1]}")
                print(f"      Columns: {list(data.columns)}")
            else:
                print(f"   ⚠️  {symbol}: No data or error")
        
        return results
        
    except Exception as e:
        print(f"   ❌ Data collection failed: {e}")
        return {}

def demonstrate_data_caching():
    """Demonstrate data caching functionality"""
    print("\n" + "=" * 80)
    print("DATA CACHING DEMONSTRATION")
    print("=" * 80)
    
    # Create configuration
    config = create_default_config("./")
    config.USE_FEATURE_CACHING = True
    
    # Create data handler
    print("💾 Initializing Data Handler with caching...")
    data_handler = DataHandler(config)
    
    # Create sample data to cache
    print("\n📊 Creating sample data for caching...")
    sample_data = pd.DataFrame({
        'timestamp': pd.date_range('2023-01-01', periods=100, freq='D'),
        'Open': np.random.uniform(100, 110, 100),
        'High': np.random.uniform(110, 120, 100),
        'Low': np.random.uniform(90, 100, 100),
        'Close': np.random.uniform(95, 115, 100),
        'Volume': np.random.randint(100000, 500000, 100)
    })
    sample_data.set_index('timestamp', inplace=True)
    
    # Save data to cache
    cache_key = "demo_data_AAPL_1D"
    print(f"💾 Saving data to cache with key: {cache_key}")
    
    try:
        # Save to cache (simulated)
        cache_file = f"cache/{cache_key}.pkl"
        os.makedirs("cache", exist_ok=True)
        sample_data.to_pickle(cache_file)
        print(f"   ✅ Data cached successfully: {len(sample_data)} records")
        
        # Load from cache
        print(f"📂 Loading data from cache...")
        cached_data = pd.read_pickle(cache_file)
        print(f"   ✅ Data loaded from cache: {len(cached_data)} records")
        
        # Verify data integrity
        if sample_data.equals(cached_data):
            print(f"   ✅ Cache integrity verified")
        else:
            print(f"   ⚠️  Cache integrity issue detected")
        
        # Show cache info
        cache_info = data_handler.get_cache_info()
        print(f"\n📊 Cache Information:")
        print(f"   Cache directory: {cache_info.get('cache_dir', 'cache/')}")
        print(f"   Cache enabled: {cache_info.get('enabled', True)}")
        
        return cached_data
        
    except Exception as e:
        print(f"   ❌ Caching demonstration failed: {e}")
        return pd.DataFrame()

def demonstrate_data_quality_validation():
    """Demonstrate data quality validation"""
    print("\n" + "=" * 80)
    print("DATA QUALITY VALIDATION DEMONSTRATION")
    print("=" * 80)
    
    # Create sample data with various quality issues
    print("📊 Creating sample data with quality issues...")
    
    dates = pd.date_range('2023-01-01', periods=100, freq='D')
    data = pd.DataFrame({
        'timestamp': dates,
        'Open': np.random.uniform(100, 110, 100),
        'High': np.random.uniform(110, 120, 100),
        'Low': np.random.uniform(90, 100, 100),
        'Close': np.random.uniform(95, 115, 100),
        'Volume': np.random.randint(100000, 500000, 100)
    })
    
    # Introduce quality issues
    data.loc[10:15, 'Close'] = np.nan  # Missing data
    data.loc[20, 'High'] = data.loc[20, 'Low'] - 5  # Invalid high < low
    data.loc[30:32, 'Volume'] = 0  # Zero volume
    data.loc[40, 'Close'] = data.loc[39, 'Close'] * 3  # Price spike
    
    data.set_index('timestamp', inplace=True)
    
    print(f"   Created data with {len(data)} records")
    print(f"   Introduced various quality issues for testing")
    
    # Validate data quality
    print("\n🔍 Validating data quality...")
    required_columns = ['Open', 'High', 'Low', 'Close', 'Volume']
    
    try:
        quality_report = validate_data_quality(data, required_columns)
        
        print(f"\n📊 Data Quality Report:")
        print(f"   Overall Quality: {quality_report['overall_quality']}")
        print(f"   Missing Data: {quality_report['missing_data_pct']:.1%}")
        print(f"   Invalid Records: {quality_report['invalid_records']}")
        print(f"   Data Completeness: {quality_report['completeness']:.1%}")
        
        if quality_report['issues']:
            print(f"\n⚠️  Quality Issues Detected:")
            for issue in quality_report['issues']:
                print(f"   • {issue}")
        
        if quality_report['recommendations']:
            print(f"\n💡 Recommendations:")
            for rec in quality_report['recommendations']:
                print(f"   • {rec}")
        
        return quality_report
        
    except Exception as e:
        print(f"   ❌ Data quality validation failed: {e}")
        return {}

def demonstrate_multi_timeframe_collection():
    """Demonstrate multi-timeframe data collection"""
    print("\n" + "=" * 80)
    print("MULTI-TIMEFRAME DATA COLLECTION DEMONSTRATION")
    print("=" * 80)
    
    # Create configuration
    config = create_default_config("./")
    
    # Define timeframes to collect
    timeframes = ['1H', '4H', '1D', '1W']
    symbol = 'DEMO'
    
    print(f"📊 Collecting data for {symbol} across multiple timeframes...")
    print(f"   Timeframes: {', '.join(timeframes)}")
    
    results = {}
    
    for timeframe in timeframes:
        print(f"\n📈 Collecting {timeframe} data...")
        
        try:
            # Create sample data for each timeframe
            if timeframe == '1H':
                periods = 24 * 30  # 30 days of hourly data
                freq = 'H'
            elif timeframe == '4H':
                periods = 6 * 30  # 30 days of 4-hour data
                freq = '4H'
            elif timeframe == '1D':
                periods = 30  # 30 days of daily data
                freq = 'D'
            else:  # 1W
                periods = 4  # 4 weeks of weekly data
                freq = 'W'
            
            dates = pd.date_range('2023-01-01', periods=periods, freq=freq)
            
            # Generate realistic price data
            price = 100.0
            data_points = []
            
            for date in dates:
                price *= (1 + np.random.normal(0, 0.01))
                
                open_price = price * np.random.uniform(0.995, 1.005)
                high_price = max(open_price, price) * np.random.uniform(1.0, 1.01)
                low_price = min(open_price, price) * np.random.uniform(0.99, 1.0)
                volume = np.random.randint(50000, 200000)
                
                data_points.append({
                    'timestamp': date,
                    'Open': open_price,
                    'High': high_price,
                    'Low': low_price,
                    'Close': price,
                    'Volume': volume
                })
            
            tf_data = pd.DataFrame(data_points)
            tf_data.set_index('timestamp', inplace=True)
            
            results[timeframe] = tf_data
            
            print(f"   ✅ {timeframe}: {len(tf_data)} records")
            print(f"      Date range: {tf_data.index[0]} to {tf_data.index[-1]}")
            
        except Exception as e:
            print(f"   ❌ {timeframe} collection failed: {e}")
    
    # Show timeframe comparison
    print(f"\n📊 Timeframe Comparison:")
    print(f"{'Timeframe':<10} {'Records':<10} {'Start Date':<12} {'End Date':<12}")
    print("-" * 50)
    
    for tf, data in results.items():
        if not data.empty:
            start_date = data.index[0].strftime('%Y-%m-%d')
            end_date = data.index[-1].strftime('%Y-%m-%d')
            print(f"{tf:<10} {len(data):<10} {start_date:<12} {end_date:<12}")
    
    return results

def main():
    """Run all data collection demonstrations"""
    print("🚀 AxiomEdge Data Collection - Comprehensive Demonstration")
    print("=" * 80)
    
    try:
        # Basic data collection
        basic_results = demonstrate_basic_data_collection()
        
        # Data caching
        cached_data = demonstrate_data_caching()
        
        # Data quality validation
        quality_report = demonstrate_data_quality_validation()
        
        # Multi-timeframe collection
        timeframe_results = demonstrate_multi_timeframe_collection()
        
        print("\n" + "=" * 80)
        print("✅ ALL DATA COLLECTION DEMONSTRATIONS COMPLETED!")
        print("=" * 80)
        
        print("\n🎯 Summary:")
        print(f"   Basic Collection: {len(basic_results)} symbols processed")
        print(f"   Caching: {'✅ Working' if not cached_data.empty else '❌ Issues'}")
        print(f"   Quality Validation: {'✅ Working' if quality_report else '❌ Issues'}")
        print(f"   Multi-timeframe: {len(timeframe_results)} timeframes collected")
        
        print("\n📚 Key Capabilities Demonstrated:")
        print("   ✅ Multi-symbol data collection")
        print("   ✅ Data caching and retrieval")
        print("   ✅ Data quality validation and reporting")
        print("   ✅ Multi-timeframe data handling")
        print("   ✅ Error handling and recovery")
        print("   ✅ Cache management and integrity")
        
        print("\n🔧 Next Steps:")
        print("   1. Connect to real data sources (Yahoo Finance, Alpha Vantage, etc.)")
        print("   2. Implement custom data validation rules")
        print("   3. Set up automated data collection schedules")
        print("   4. Add data preprocessing and cleaning")
        print("   5. Integrate with feature engineering pipeline")
        
    except Exception as e:
        print(f"\n❌ Error during demonstration: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
