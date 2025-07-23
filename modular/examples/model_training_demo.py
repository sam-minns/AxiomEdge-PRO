#!/usr/bin/env python3
# =============================================================================
# AXIOM EDGE - MODEL TRAINING DEMONSTRATION
# =============================================================================

"""
This script demonstrates the comprehensive model training capabilities
of the AxiomEdge framework, including hyperparameter optimization,
feature selection, and advanced validation techniques.
"""

import sys
import os
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from axiom_edge import (
    ModelTrainingTask,
    FeatureEngineeringTask,
    ModelTrainer,
    GeminiAnalyzer,
    create_default_config
)

def create_sample_data_with_labels(n_samples: int = 1000) -> pd.DataFrame:
    """Create sample data with engineered features and labels"""
    np.random.seed(42)
    
    # Generate base price data
    dates = pd.date_range(start="2023-01-01", periods=n_samples, freq="D")
    price = 100.0
    prices = []
    
    for i in range(n_samples):
        # Add trend and noise
        trend = 0.0001 * np.sin(2 * np.pi * i / 252)
        noise = np.random.normal(0, 0.02)
        price *= (1 + trend + noise)
        prices.append(price)
    
    # Create OHLCV data
    data = []
    for i, (date, close_price) in enumerate(zip(dates, prices)):
        open_price = close_price * np.random.uniform(0.995, 1.005)
        high_price = max(open_price, close_price) * np.random.uniform(1.0, 1.02)
        low_price = min(open_price, close_price) * np.random.uniform(0.98, 1.0)
        volume = np.random.randint(500000, 2000000)
        
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

def demonstrate_basic_model_training():
    """Demonstrate basic model training using the task interface"""
    print("=" * 80)
    print("BASIC MODEL TRAINING DEMONSTRATION")
    print("=" * 80)
    
    # Create sample data
    print("📊 Creating sample data with features...")
    raw_data = create_sample_data_with_labels(500)
    
    # Engineer features first
    feature_task = FeatureEngineeringTask()
    features = feature_task.engineer_features(raw_data)
    print(f"   Generated {len(features.columns)} features")
    
    # Create target variable (simplified for demo)
    # In practice, this would come from the feature engineering labeling
    future_returns = features['Close'].shift(-5) / features['Close'] - 1
    target = (future_returns > 0.02).astype(int)  # Binary classification: >2% gain in 5 days
    target.name = 'target_signal_pressure_class'
    
    # Remove NaN values
    valid_mask = ~(target.isna() | features.isna().any(axis=1))
    features_clean = features[valid_mask]
    target_clean = target[valid_mask]
    
    print(f"   Clean data: {len(features_clean)} samples")
    print(f"   Target distribution: {target_clean.value_counts().to_dict()}")
    
    # Create model training task
    print("\n🤖 Initializing model training task...")
    task = ModelTrainingTask()
    
    # Train model
    print("⚙️  Training model...")
    results = task.train_model(features_clean, target_clean)
    
    if "error" not in results:
        print(f"\n✅ Model training completed!")
        print(f"   F1 Score: {results['metrics']['f1_score']:.3f}")
        print(f"   Accuracy: {results['metrics']['accuracy']:.3f}")
        print(f"   Precision: {results['metrics']['precision']:.3f}")
        print(f"   Recall: {results['metrics']['recall']:.3f}")
        print(f"   Features used: {len(results['selected_features'])}")
        
        # Show top features
        if results['feature_importance']:
            print(f"\n🎯 Top 10 Most Important Features:")
            for i, (feature, importance) in enumerate(list(results['feature_importance'].items())[:10]):
                print(f"   {i+1:2d}. {feature}: {importance:.4f}")
    else:
        print(f"❌ Model training failed: {results['error']}")
    
    return results

def demonstrate_advanced_model_training():
    """Demonstrate advanced model training with custom configuration"""
    print("\n" + "=" * 80)
    print("ADVANCED MODEL TRAINING DEMONSTRATION")
    print("=" * 80)
    
    # Create larger dataset
    print("📊 Creating larger dataset for advanced training...")
    raw_data = create_sample_data_with_labels(1000)
    
    # Engineer features with custom configuration
    config = create_default_config("./")
    config.EMA_PERIODS = [8, 13, 21, 34, 55]
    config.RSI_STANDARD_PERIODS = [14, 21, 28]
    config.OPTUNA_TRIALS = 30  # Reduced for demo speed
    config.FEATURE_SELECTION_METHOD = "mutual_info"
    config.SHADOW_SET_VALIDATION = True
    config.CALCULATE_SHAP_VALUES = True
    
    print(f"   Configuration: {config.OPTUNA_TRIALS} Optuna trials, {config.FEATURE_SELECTION_METHOD} feature selection")
    
    # Engineer features
    feature_task = FeatureEngineeringTask(config)
    features = feature_task.engineer_features(raw_data)
    
    # Create more sophisticated labels using the feature engineer
    from axiom_edge import FeatureEngineer
    timeframe_roles = {'base': 'D1'}
    feature_engineer = FeatureEngineer(config, timeframe_roles, {})
    
    # Add technical indicators needed for labeling
    features_with_indicators = feature_engineer._calculate_technical_indicators(features)
    
    # Generate multi-task labels
    labeled_data = feature_engineer.label_data_multi_task(features_with_indicators)
    
    print(f"   Labeled data shape: {labeled_data.shape}")
    
    # Analyze label distribution
    target_cols = [col for col in labeled_data.columns if col.startswith('target_')]
    print(f"   Generated {len(target_cols)} target variables:")
    for target_col in target_cols:
        if labeled_data[target_col].dtype in ['int64', 'int32']:
            dist = labeled_data[target_col].value_counts().sort_index()
            print(f"     {target_col}: {dict(dist)}")
    
    # Create advanced model trainer
    print("\n🧠 Initializing advanced model trainer...")
    try:
        gemini_analyzer = GeminiAnalyzer()
        print("   ✅ Gemini AI analyzer initialized")
    except Exception as e:
        print(f"   ⚠️  Gemini analyzer not available: {e}")
        # Create mock analyzer
        class MockAnalyzer:
            def __init__(self):
                self.api_key_valid = False
        gemini_analyzer = MockAnalyzer()
    
    trainer = ModelTrainer(config, gemini_analyzer)
    
    # Train model with advanced features
    print("⚙️  Training advanced model...")
    results = trainer.train_and_validate_model(labeled_data)
    
    if "error" not in results:
        print(f"\n✅ Advanced model training completed!")
        print(f"   Target: {results['target_column']}")
        print(f"   F1 Score: {results['metrics']['f1_score']:.3f}")
        print(f"   Accuracy: {results['metrics']['accuracy']:.3f}")
        print(f"   Features selected: {len(results['selected_features'])}")
        
        # Show feature importance
        if results['feature_importance']:
            print(f"\n📊 Feature Importance Analysis:")
            top_features = list(results['feature_importance'].items())[:15]
            for i, (feature, importance) in enumerate(top_features):
                print(f"   {i+1:2d}. {feature}: {importance:.4f}")
        
        # Show SHAP analysis if available
        if results['shap_summary'] is not None:
            print(f"\n🔍 SHAP Analysis:")
            print(f"   SHAP values calculated for {len(results['shap_summary'])} features")
            top_shap = results['shap_summary'].head(10)
            for i, row in top_shap.iterrows():
                print(f"   {i+1:2d}. {row['feature']}: {row['mean_abs_shap']:.4f}")
        
        # Classification report
        print(f"\n📋 Classification Report:")
        print(results['classification_report'])
        
    else:
        print(f"❌ Advanced model training failed: {results['error']}")
    
    return results

def demonstrate_hyperparameter_optimization():
    """Demonstrate hyperparameter optimization capabilities"""
    print("\n" + "=" * 80)
    print("HYPERPARAMETER OPTIMIZATION DEMONSTRATION")
    print("=" * 80)
    
    # Create focused dataset for optimization demo
    print("📊 Creating focused dataset for optimization...")
    raw_data = create_sample_data_with_labels(300)  # Smaller for faster optimization
    
    # Quick feature engineering
    feature_task = FeatureEngineeringTask()
    features = feature_task.engineer_features(raw_data)
    
    # Create clear signal for optimization
    # Use multiple indicators to create a more learnable target
    rsi_signal = features['RSI_14'] < 30  # Oversold
    volume_signal = features['volume_ma_ratio'] > 1.2  # Volume spike
    trend_signal = features['EMA_20'] > features['EMA_50']  # Uptrend
    
    # Combine signals for target
    combined_signal = rsi_signal & volume_signal & trend_signal
    target = combined_signal.astype(int)
    target.name = 'target_signal_pressure_class'
    
    # Clean data
    valid_mask = ~(target.isna() | features.isna().any(axis=1))
    features_clean = features[valid_mask]
    target_clean = target[valid_mask]
    
    print(f"   Dataset: {len(features_clean)} samples")
    print(f"   Target distribution: {target_clean.value_counts().to_dict()}")
    
    # Configure for optimization
    config = create_default_config("./")
    config.OPTUNA_TRIALS = 20  # Quick optimization for demo
    config.OPTUNA_N_JOBS = 1
    config.FEATURE_SELECTION_METHOD = "mutual_info"
    
    # Create trainer
    print("\n🎯 Running hyperparameter optimization...")
    try:
        gemini_analyzer = GeminiAnalyzer()
    except:
        class MockAnalyzer:
            def __init__(self):
                self.api_key_valid = False
        gemini_analyzer = MockAnalyzer()
    
    trainer = ModelTrainer(config, gemini_analyzer)
    
    # Train with optimization focus
    pipeline, f1_score, selected_features, failure_reason = trainer.train_single_model(
        df_train=pd.concat([features_clean, target_clean], axis=1),
        feature_list=features_clean.columns.tolist(),
        target_col='target_signal_pressure_class',
        model_type='classification',
        task_name='optimization_demo'
    )
    
    if pipeline is not None:
        print(f"\n✅ Hyperparameter optimization completed!")
        print(f"   Best F1 Score: {f1_score:.3f}")
        print(f"   Features selected: {len(selected_features)}")
        print(f"   Optimization trials: {config.OPTUNA_TRIALS}")
        
        # Show model parameters
        model = pipeline.named_steps['model']
        print(f"\n⚙️  Optimized Parameters:")
        params = model.get_params()
        key_params = ['n_estimators', 'max_depth', 'learning_rate', 'subsample', 'colsample_bytree']
        for param in key_params:
            if param in params:
                print(f"   {param}: {params[param]}")
        
        # Test predictions
        predictions = pipeline.predict(features_clean[selected_features])
        print(f"\n🎯 Prediction Analysis:")
        print(f"   Positive predictions: {np.sum(predictions)} ({np.mean(predictions)*100:.1f}%)")
        print(f"   Actual positives: {np.sum(target_clean)} ({np.mean(target_clean)*100:.1f}%)")
        
    else:
        print(f"❌ Hyperparameter optimization failed: {failure_reason}")
    
    return pipeline, f1_score

def demonstrate_model_comparison():
    """Demonstrate comparison of different model configurations"""
    print("\n" + "=" * 80)
    print("MODEL COMPARISON DEMONSTRATION")
    print("=" * 80)
    
    # Create consistent dataset
    print("📊 Creating dataset for model comparison...")
    raw_data = create_sample_data_with_labels(400)
    feature_task = FeatureEngineeringTask()
    features = feature_task.engineer_features(raw_data)
    
    # Create target
    future_returns = features['Close'].shift(-3) / features['Close'] - 1
    target = (future_returns > 0.01).astype(int)
    target.name = 'target_signal_pressure_class'
    
    # Clean data
    valid_mask = ~(target.isna() | features.isna().any(axis=1))
    features_clean = features[valid_mask]
    target_clean = target[valid_mask]
    
    print(f"   Dataset: {len(features_clean)} samples")
    
    # Test different configurations
    configs = {
        'Conservative': {
            'OPTUNA_TRIALS': 10,
            'FEATURE_SELECTION_METHOD': 'f_classif',
            'MIN_F1_SCORE_GATE': 0.4
        },
        'Balanced': {
            'OPTUNA_TRIALS': 15,
            'FEATURE_SELECTION_METHOD': 'mutual_info',
            'MIN_F1_SCORE_GATE': 0.3
        },
        'Aggressive': {
            'OPTUNA_TRIALS': 20,
            'FEATURE_SELECTION_METHOD': 'mutual_info',
            'MIN_F1_SCORE_GATE': 0.2
        }
    }
    
    results_comparison = {}
    
    for config_name, config_params in configs.items():
        print(f"\n🔧 Testing {config_name} Configuration...")
        
        # Create config
        config = create_default_config("./")
        for param, value in config_params.items():
            setattr(config, param, value)
        
        # Train model
        task = ModelTrainingTask(config)
        results = task.train_model(features_clean, target_clean)
        
        if "error" not in results:
            results_comparison[config_name] = {
                'f1_score': results['metrics']['f1_score'],
                'accuracy': results['metrics']['accuracy'],
                'precision': results['metrics']['precision'],
                'recall': results['metrics']['recall'],
                'n_features': len(results['selected_features'])
            }
            print(f"   ✅ {config_name}: F1={results['metrics']['f1_score']:.3f}, Features={len(results['selected_features'])}")
        else:
            print(f"   ❌ {config_name}: Failed - {results['error']}")
    
    # Compare results
    if results_comparison:
        print(f"\n📊 Model Comparison Results:")
        print(f"{'Configuration':<12} {'F1 Score':<10} {'Accuracy':<10} {'Precision':<11} {'Recall':<8} {'Features':<8}")
        print("-" * 70)
        
        for config_name, metrics in results_comparison.items():
            print(f"{config_name:<12} {metrics['f1_score']:<10.3f} {metrics['accuracy']:<10.3f} "
                  f"{metrics['precision']:<11.3f} {metrics['recall']:<8.3f} {metrics['n_features']:<8}")
        
        # Find best configuration
        best_config = max(results_comparison.items(), key=lambda x: x[1]['f1_score'])
        print(f"\n🏆 Best Configuration: {best_config[0]} (F1 Score: {best_config[1]['f1_score']:.3f})")
    
    return results_comparison

def main():
    """Run all model training demonstrations"""
    print("🚀 AxiomEdge Model Training - Comprehensive Demonstration")
    print("=" * 80)
    
    try:
        # Basic demonstration
        basic_results = demonstrate_basic_model_training()
        
        # Advanced demonstration
        advanced_results = demonstrate_advanced_model_training()
        
        # Hyperparameter optimization
        opt_pipeline, opt_f1 = demonstrate_hyperparameter_optimization()
        
        # Model comparison
        comparison_results = demonstrate_model_comparison()
        
        print("\n" + "=" * 80)
        print("✅ ALL MODEL TRAINING DEMONSTRATIONS COMPLETED!")
        print("=" * 80)
        
        print("\n🎯 Summary:")
        if "error" not in basic_results:
            print(f"   Basic Model F1 Score: {basic_results['metrics']['f1_score']:.3f}")
        if "error" not in advanced_results:
            print(f"   Advanced Model F1 Score: {advanced_results['metrics']['f1_score']:.3f}")
        if opt_f1:
            print(f"   Optimized Model F1 Score: {opt_f1:.3f}")
        if comparison_results:
            best_comparison = max(comparison_results.items(), key=lambda x: x[1]['f1_score'])
            print(f"   Best Comparison Model: {best_comparison[0]} (F1: {best_comparison[1]['f1_score']:.3f})")
        
        print("\n📚 Key Capabilities Demonstrated:")
        print("   ✅ Automated hyperparameter optimization with Optuna")
        print("   ✅ Multiple feature selection methods")
        print("   ✅ Advanced validation techniques")
        print("   ✅ SHAP feature importance analysis")
        print("   ✅ Multi-task learning support")
        print("   ✅ Model comparison and benchmarking")
        print("   ✅ Robust error handling and validation")
        
        print("\n🔧 Next Steps:")
        print("   1. Experiment with different hyperparameter ranges")
        print("   2. Try ensemble methods and model stacking")
        print("   3. Implement custom feature selection strategies")
        print("   4. Add cross-validation and time series splits")
        print("   5. Integrate with backtesting for strategy validation")
        
    except Exception as e:
        print(f"\n❌ Error during demonstration: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
