# AxiomEdge Trading Framework - Modular Architecture

AxiomEdge is a comprehensive AI-powered trading framework that can be used as individual components or as a complete integrated system. The modular design allows you to use only the parts you need for specific tasks.

## ⚠️ **IMPORTANT DISCLAIMER**

**This is an experimental framework for research purposes only. It is not financial advice. Trading financial markets involves substantial risk, and you can lose all of your invested capital. Past performance is not indicative of future results. Do not run this framework with real money without fully understanding the code and the risks involved.**

## 🌟 **What Makes AxiomEdge Unique**

Unlike traditional backtesting frameworks (like `backtesting.py`, `zipline`, or `backtrader`), AxiomEdge offers revolutionary features that set it apart:

### 🧠 **AI-Powered Intelligence**
- **AI Doctor**: Gemini AI continuously monitors performance and suggests optimizations
- **Intelligent Parameter Tuning**: AI-guided hyperparameter optimization with drift detection
- **Regime-Aware Adaptation**: Automatic strategy adjustment based on market conditions
- **Natural Language Strategy Analysis**: AI provides human-readable insights and recommendations

### 📊 **Advanced Telemetry & Monitoring**
- **Real-Time Telemetry**: Comprehensive JSONL-based telemetry system tracking every aspect of performance
- **Session Management**: Complete session tracking with historical performance comparison
- **Health Monitoring**: System component health tracking with automated alerts
- **Performance Attribution**: Detailed cycle-by-cycle performance breakdown and analysis

### 🔬 **Scientific Rigor**
- **Walk-Forward Analysis**: Robust out-of-sample validation with multiple cycles
- **SHAP Explainability**: Feature importance analysis with SHAP values for model transparency
- **Multi-Task Learning**: Simultaneous prediction of multiple market outcomes
- **Statistical Validation**: Comprehensive statistical testing and validation protocols

### 🧬 **Evolutionary Strategy Development**
- **Genetic Programming**: Automatic discovery of trading rules through evolutionary algorithms
- **Dynamic Ensembles**: Adaptive model weighting based on performance and feature importance
- **Strategy Evolution**: Continuous improvement of trading strategies through AI-guided evolution
- **Rule Complexity Management**: Intelligent balance between strategy complexity and performance

### ⚡ **Production-Ready Architecture**
- **Modular Design**: Each component can be used independently or as part of the complete framework
- **Scalable Processing**: Multi-core parallel processing for large datasets
- **Memory Efficiency**: Optimized data structures and streaming telemetry to prevent memory issues
- **Error Recovery**: Robust error handling with graceful degradation and automatic recovery

### 📈 **Comprehensive Feature Engineering**
- **200+ Features**: Extensive technical, statistical, and behavioral feature library
- **Multi-Timeframe Analysis**: Automatic integration of features across multiple timeframes
- **Pattern Recognition**: Advanced candlestick and price pattern detection
- **Entropy & Cycle Analysis**: Sophisticated market microstructure analysis

### 🎯 **Intelligent Automation**
- **Framework Memory**: Historical performance tracking with continuous learning
- **Adaptive Thresholds**: Dynamic confidence thresholds based on market conditions
- **Intervention Protocols**: Automated intervention when performance degrades
- **Parameter Drift Detection**: Monitoring and alerting for strategy parameter drift

### 📋 **Professional Reporting**
- **Multi-Format Reports**: Text, HTML, and interactive dashboard generation
- **Publication-Quality Visualizations**: High-resolution charts and plots
- **Executive Summaries**: Concise performance summaries for stakeholders
- **Regulatory Compliance**: Detailed audit trails and performance attribution

## 📄 **License & Attribution**

This project is released under a **GPL 3.0 permissive license**. While you are free to use, modify, and distribute this software, it would be nice to be acknowledged for the original code in any further developments or public-facing projects that build upon it. A simple credit or link back to the original repository is greatly appreciated.

## 🏗️ Architecture Overview

```
axiom_edge/
├── __init__.py                # Main package exports
├── config.py                  # ✅ Configuration and validation
├── data_handler.py            # ✅ Data collection and caching
├── ai_analyzer.py             # ✅ AI analysis with Gemini
├── feature_engineer.py        # ✅ 200+ Feature engineering (COMPLETE)
├── model_trainer.py           # ✅ ML model training (COMPLETE)
├── backtester.py              # ✅ Advanced backtesting engine (COMPLETE)
├── genetic_programmer.py      # ✅ Genetic algorithm optimization (COMPLETE)
├── report_generator.py        # ✅ Report generation (COMPLETE)
├── framework_orchestrator.py  # ✅ Complete framework orchestration (COMPLETE)
├── telemetry.py               # ✅ Advanced telemetry & monitoring (COMPLETE)
├── tasks.py                   # ✅ Task-specific interfaces
└── utils.py                   # ✅ Utility functions
```

### 🎯 **Implementation Status**
- **✅ ALL COMPONENTS FULLY IMPLEMENTED**: Complete production-ready framework
- **🚀 Advanced Features**: Telemetry, AI Doctor, Dynamic Ensembles, Walk-Forward Analysis
- **📊 Comprehensive Analytics**: 200+ features, SHAP analysis, Performance attribution

## 🚀 Quick Start

### Installation

```bash
# Clone the repository
git clone <repository-url>
cd AxiomEdge

# Create virtual environment (recommended)
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Or install in development mode
pip install -e .

# Set up environment variables
export GEMINI_API_KEY="your-gemini-api-key"
export FINANCIAL_API_KEY="your-financial-data-api-key"
```

### Basic Usage

```python
# Import specific components
from axiom_edge import DataCollectionTask, FeatureEngineeringTask, ConfigModel

# Create configuration
config = ConfigModel(
    BASE_PATH="./data",
    REPORT_LABEL="My_Strategy",
    INITIAL_CAPITAL=10000.0,
    FEATURE_SELECTION_METHOD="mutual_info",
    TRAINING_WINDOW="30D",
    RETRAINING_FREQUENCY="7D",
    FORWARD_TEST_GAP="1D",
    LOOKAHEAD_CANDLES=5,
    RISK_CAP_PER_TRADE_USD=500.0,
    BASE_RISK_PER_TRADE_PCT=0.02,
    CONFIDENCE_TIERS={"high": 1.5, "medium": 1.0, "low": 0.5}
)

# Use individual components
data_task = DataCollectionTask(config)
data = data_task.collect_data(["AAPL", "GOOGL"], "2023-01-01", "2024-01-01")

# Engineer 200+ features
feature_task = FeatureEngineeringTask(config)
features = feature_task.engineer_features(data["AAPL"])
print(f"Generated {len(features.columns)} features!")
```

## 📋 Available Tasks

### 1. Data Collection Task
Collect historical data from various sources (Alpha Vantage, Yahoo Finance, Polygon).

```bash
# Command line usage
python main.py --task data_collection --symbols AAPL,GOOGL --start 2023-01-01 --end 2024-01-01

# Python usage
from axiom_edge import DataCollectionTask

task = DataCollectionTask()
data = task.collect_data(["AAPL", "GOOGL"], "2023-01-01", "2024-01-01")
task.save_data(data, "collected_data/")
```

**Features:**
- Multi-source data collection
- Intelligent caching (in-memory + disk)
- Multiple timeframes support
- Data validation and cleaning

### 2. Broker Information Task
Collect broker spreads and trading costs with AI-powered analysis.

```bash
# Command line usage
python main.py --task broker_info --symbols EURUSD,GBPUSD --broker oanda

# Python usage
from axiom_edge import BrokerInfoTask

task = BrokerInfoTask()
spreads = task.collect_spreads(["EURUSD", "GBPUSD"], "oanda")
analysis = task.analyze_broker_costs(["EURUSD"], {"spreads": spreads})
```

**Features:**
- Real-time spread collection
- Multi-broker comparison
- AI-powered cost analysis
- Trading cost optimization recommendations

### 3. Backtesting Task
Backtest your own trading strategies with comprehensive performance analysis.

```bash
# Command line usage
python main.py --task backtest --data-file my_data.csv --strategy-config strategy.json

# Python usage
from axiom_edge import BacktestTask

task = BacktestTask()
strategy_rules = {"sma_short": 10, "sma_long": 30}
results = task.backtest_strategy(data, strategy_rules)
```

**Features:**
- Custom strategy rule engine
- Comprehensive performance metrics
- Risk management integration
- Parameter optimization

### 4. Feature Engineering Task ⭐ **FULLY IMPLEMENTED**
Engineer 200+ features from raw price data for machine learning models.

```bash
# Command line usage
python main.py --task features --data-file data.csv --output features.csv

# Python usage
from axiom_edge import FeatureEngineeringTask

task = FeatureEngineeringTask()
features = task.engineer_features(raw_data)
selected_features = task.select_features(features, "target_column")

# Advanced usage with custom configuration
from axiom_edge import FeatureEngineer, create_default_config

config = create_default_config("./")
config.EMA_PERIODS = [8, 13, 21, 34, 55, 89, 144]  # Fibonacci EMAs
config.RSI_STANDARD_PERIODS = [14, 21, 28]

feature_engineer = FeatureEngineer(config, {'base': 'D1'}, {})
features = feature_engineer.engineer_features(data, {'D1': data})

# Multi-task labeling
labeled_data = feature_engineer.label_data_multi_task(features)
```

**🎯 Feature Categories (200+ Total):**
- **📈 Technical Indicators (40+)**: RSI, MACD, Bollinger Bands, Stochastic, ATR, EMAs
- **📊 Price & Returns (30+)**: Momentum, ROC, price derivatives, gap analysis
- **📉 Volume Analysis (20+)**: Volume ratios, correlations, accumulation/distribution
- **📋 Statistical Moments (25+)**: Rolling statistics, skewness, kurtosis, quantiles
- **🕐 Time-based (15+)**: Session indicators, cyclical encoding, hour/day patterns
- **🔍 Pattern Recognition (20+)**: Candle patterns, OHLC ratios, doji detection
- **🧠 Advanced Analytics (15+)**: Entropy, cycle analysis, Hilbert transforms
- **⏰ Multi-timeframe (Variable)**: Higher timeframe trend, momentum, volatility

**🏷️ Multi-task Labeling System:**
- **Signal Pressure Classification**: 5-class system (Strong Buy to Strong Sell)
- **Timing Score**: TP/SL hit prediction (-1 to 1 scale)
- **Pattern Labels**: Bullish/Bearish engulfing pattern detection
- **Volatility Spike Detection**: Future volatility prediction

**⚡ Advanced Capabilities:**
- **Parallel Processing**: Multi-core feature engineering for large datasets
- **Multi-timeframe Fusion**: Integrates features from multiple timeframes
- **Robust Data Handling**: Missing value imputation, outlier detection
- **Configurable Parameters**: Customizable periods, thresholds, and methods

### 5. Model Training Task ⭐ **FULLY IMPLEMENTED**
Train machine learning models with automated hyperparameter optimization and advanced validation.

```bash
# Command line usage
python main.py --task model_training --data-file features.csv

# Python usage
from axiom_edge import ModelTrainingTask

task = ModelTrainingTask()
model_results = task.train_model(features, target)

# Advanced usage with custom configuration
from axiom_edge import ModelTrainer, GeminiAnalyzer, create_default_config

config = create_default_config("./")
config.OPTUNA_TRIALS = 100
config.FEATURE_SELECTION_METHOD = "mutual_info"
config.SHADOW_SET_VALIDATION = True
config.CALCULATE_SHAP_VALUES = True

gemini_analyzer = GeminiAnalyzer()
trainer = ModelTrainer(config, gemini_analyzer)

# Train with labeled data (from feature engineering)
results = trainer.train_and_validate_model(labeled_data)

# Single model training with custom parameters
pipeline, f1_score, features, error = trainer.train_single_model(
    df_train=data,
    feature_list=feature_list,
    target_col='target_signal_pressure_class',
    model_type='classification',
    task_name='custom_model'
)
```

**🎯 Core Capabilities:**
- **🤖 XGBoost Integration**: Optimized gradient boosting models
- **🔧 Hyperparameter Optimization**: Optuna-based automated tuning
- **📊 Feature Selection**: Multiple methods (mutual_info, f_classif, custom)
- **✅ Advanced Validation**: Shadow set validation, cross-validation
- **🧠 SHAP Analysis**: Feature importance and explainability
- **⚖️ Class Balancing**: Automatic class weight calculation
- **🎯 Multi-task Learning**: Support for multiple target variables

**🔬 Advanced Features:**
- **Optuna Integration**: Multi-objective hyperparameter optimization
- **Shadow Set Validation**: Prevents overfitting with holdout validation
- **Feature Importance Analysis**: Both model-based and SHAP-based
- **Threshold Optimization**: Automatic optimal threshold finding
- **Robust Error Handling**: Comprehensive failure detection and reporting
- **AI-Guided Optimization**: Gemini AI integration for intelligent parameter tuning

**📈 Performance Metrics:**
- **Classification**: F1-score, Accuracy, Precision, Recall, AUC
- **Regression**: RMSE, MAE, R², Explained Variance
- **Feature Analysis**: SHAP values, permutation importance
- **Validation**: Cross-validation scores, holdout performance

### 6. Genetic Programming Task ⭐ **FULLY IMPLEMENTED**
Evolve trading rules using genetic algorithms with AI-guided optimization.

```bash
# Command line usage
python main.py --task genetic_programming --data-file features.csv

# Python usage
from axiom_edge import GeneticProgrammer, FeatureEngineeringTask

# Create gene pool from features
feature_task = FeatureEngineeringTask()
features = feature_task.engineer_features(raw_data)

gene_pool = {
    'continuous_features': ['RSI_14', 'EMA_20', 'EMA_50', 'MACD', 'ATR'],
    'state_features': ['is_ny_session', 'day_of_week'],
    'comparison_operators': ['>', '<', '>=', '<='],
    'state_operators': ['==', '!='],
    'logical_operators': ['AND', 'OR'],
    'constants': [20, 30, 50, 70, 80]
}

# Initialize genetic programmer
gp = GeneticProgrammer(
    gene_pool=gene_pool,
    config=config,
    population_size=50,
    generations=25,
    mutation_rate=0.1,
    crossover_rate=0.7
)

# Evolve trading rules
best_chromosome, best_fitness = gp.run_evolution(features)
print(f"Best Long Rule: {best_chromosome[0]}")
print(f"Best Short Rule: {best_chromosome[1]}")
print(f"Fitness (Sharpe): {best_fitness:.4f}")
```

**🧬 Core Capabilities:**
- **🔄 Genetic Algorithm**: Tournament selection, crossover, mutation
- **📊 Fitness Evaluation**: Sharpe ratio-based performance measurement
- **🧠 AI Integration**: Gemini AI-guided gene pool optimization
- **⚡ Parallel Processing**: Multi-core fitness evaluation
- **🎯 Rule Evolution**: Automatic trading rule discovery
- **🔧 Configurable Parameters**: Population size, generations, mutation rates

**🎲 Evolution Operations:**
- **Selection**: Tournament selection for parent choosing
- **Crossover**: Logical operator-based rule recombination
- **Mutation**: Feature, operator, and constant mutations
- **Fitness**: Risk-adjusted return evaluation with trade frequency penalties
- **Elitism**: Best individuals preserved across generations

**🧪 Advanced Features:**
- **Gene Pool Optimization**: AI-driven feature and operator selection
- **Rule Complexity Control**: Depth-limited rule generation
- **Semantic Correctness**: Type-aware feature and operator matching
- **Performance Analytics**: Detailed evolution statistics and tracking
- **Retry Mechanisms**: AI-guided gene pool fixes for failed evolutions

### 7. Report Generation Task ⭐ **FULLY IMPLEMENTED**
Generate comprehensive performance reports with visualizations and detailed analytics.

```bash
# Command line usage
python main.py --task report_generation --trades-file trades.csv --equity-file equity.csv

# Python usage
from axiom_edge import ReportGenerator, create_default_config

config = create_default_config("./")
config.nickname = "My Strategy"
config.REPORT_LABEL = "V2.1_PRODUCTION"

report_gen = ReportGenerator(config)

# Generate comprehensive report
metrics = report_gen.generate_full_report(
    trades_df=trades_df,
    equity_curve=equity_curve,
    cycle_metrics=cycle_metrics,
    aggregated_shap=shap_data,
    last_classification_report=classification_report
)

# Generate individual components
report_gen.plot_equity_curve(equity_curve)
report_gen.plot_shap_summary(shap_data)
report_gen.plot_trade_analysis(trades_df)
report_gen.generate_html_report(metrics, trades_df, equity_curve, shap_data)

# Quick summary
summary = report_gen.generate_summary_stats(metrics)
print(summary)
```

**📊 Core Capabilities:**
- **📋 Text Reports**: Professional formatted performance reports
- **📈 Static Visualizations**: Matplotlib-based charts and plots
- **🌐 Interactive Dashboards**: Plotly-based HTML dashboards
- **📊 Performance Metrics**: Comprehensive risk-adjusted analytics
- **🧠 Feature Analysis**: SHAP importance visualization
- **📉 Trade Analysis**: Detailed trade breakdown and statistics

**📈 Visualization Types:**
- **Equity Curve**: Portfolio value over time with drawdown analysis
- **SHAP Summary**: Feature importance horizontal bar charts
- **Trade Analysis**: PnL distribution, cumulative returns, monthly breakdown
- **Interactive Dashboard**: Multi-panel Plotly dashboard with zoom/pan
- **Risk Metrics**: Drawdown periods, volatility analysis
- **Performance Attribution**: Cycle-by-cycle breakdown

**📋 Report Sections:**
- **Executive Summary**: Key performance metrics and returns
- **Performance Metrics**: CAGR, Sharpe, Sortino, Calmar ratios
- **Trade Analysis**: Win rate, profit factor, average win/loss
- **Risk Analysis**: Maximum drawdown, volatility measures
- **Walk-Forward Cycles**: Cycle-by-cycle performance breakdown
- **Feature Importance**: Top SHAP features with importance scores
- **Model Performance**: Classification reports and validation metrics

**🎨 Output Formats:**
- **Text Reports**: Professional ASCII-formatted reports (90-char width)
- **PNG Images**: High-resolution charts (300 DPI)
- **HTML Dashboards**: Interactive Plotly visualizations
- **Summary Stats**: Concise performance summaries
- **CSV Exports**: Detailed metrics for further analysis

### 9. Telemetry & Monitoring ⭐ **FULLY IMPLEMENTED**
Advanced telemetry system for comprehensive framework monitoring and analysis.

```bash
# Python usage
from axiom_edge import TelemetryCollector, TelemetryAnalyzer

# Initialize telemetry collection
telemetry = TelemetryCollector("logs/telemetry.jsonl")

# Log cycle completion
telemetry.log_cycle_data(
    cycle_num=1,
    status="completed",
    config_snapshot=config,
    labeling_summary=labeling_data,
    training_summary=training_results,
    backtest_metrics=performance_metrics,
    horizon_metrics=forward_metrics,
    ai_notes="Strong momentum signals detected"
)

# Log AI interventions
telemetry.log_ai_intervention(
    intervention_type="strategic",
    trigger_reason="performance_degradation",
    ai_analysis={"recommendation": "reduce_position_size"},
    action_taken={"position_size_multiplier": 0.8}
)

# Log performance milestones
telemetry.log_performance_milestone(
    milestone_type="new_equity_high",
    metrics=current_metrics,
    comparison_baseline=previous_best
)

# Analyze telemetry data
analyzer = TelemetryAnalyzer("logs/telemetry.jsonl")
trends = analyzer.analyze_cycle_performance_trends()
ai_effectiveness = analyzer.get_ai_intervention_effectiveness()

# Export session data
telemetry.export_session_data("csv")
telemetry.close_session()
```

**📊 Telemetry Capabilities:**
- **📈 Cycle Tracking**: Complete cycle-by-cycle performance monitoring
- **🤖 AI Intervention Logging**: Track AI decisions and their effectiveness
- **🎯 Performance Milestones**: Automatic detection and logging of achievements
- **🔍 System Health**: Component health monitoring with alerts
- **🧬 Evolution Tracking**: Genetic programming progress monitoring
- **📊 Regime Detection**: Market regime changes and strategy adaptations

**🔬 Advanced Analytics:**
- **Performance Trends**: Statistical analysis of performance evolution
- **AI Effectiveness**: Measure impact of AI interventions
- **Feature Evolution**: Track feature importance changes over time
- **Session Management**: Complete session lifecycle tracking
- **Export Capabilities**: Multiple format exports (JSON, CSV, Parquet)

### 10. Complete Framework Task ⭐ **FULLY IMPLEMENTED**
Orchestrate the complete AxiomEdge framework with walk-forward analysis and integrated workflow.

```bash
# Command line usage
python main.py --task complete --data-files "data/*.csv" --config config.json

# Python usage
from axiom_edge import FrameworkOrchestrator, CompleteFrameworkTask, create_default_config

# Using the orchestrator directly
config = create_default_config("./")
config.nickname = "Production Strategy"
config.TRAINING_WINDOW_DAYS = 252  # 1 year training
config.TEST_WINDOW_DAYS = 63       # 3 months testing
config.WALK_FORWARD_STEP_DAYS = 21 # 1 month step
config.MAX_WALK_FORWARD_CYCLES = 10
config.ENABLE_GENETIC_PROGRAMMING = True

orchestrator = FrameworkOrchestrator(config)

# Run complete framework
results = orchestrator.run_complete_framework(
    data_files=["data/AAPL.csv", "data/GOOGL.csv"],
    symbols=None,  # Or specify symbols for live data
    start_date="2023-01-01",
    end_date="2024-01-01"
)

# Using the task interface
task = CompleteFrameworkTask(config)
results = task.run_complete_framework(["data/AAPL.csv"])

# Single cycle analysis for testing
single_results = orchestrator.run_single_cycle_analysis(data)
```

**🎯 Core Orchestration:**
- **📊 Data Pipeline**: Automated data collection and preparation
- **🔧 Feature Engineering**: 200+ features with multi-timeframe analysis
- **🤖 Model Training**: Hyperparameter optimization and validation
- **🧬 Strategy Evolution**: Genetic programming for rule discovery
- **📋 Comprehensive Reporting**: Performance analysis and visualization
- **⚙️ Walk-Forward Analysis**: Robust out-of-sample validation

**🔄 Walk-Forward Analysis:**
- **Training Windows**: Configurable training period (default: 252 days)
- **Test Windows**: Out-of-sample validation period (default: 63 days)
- **Step Size**: Rolling window advancement (default: 21 days)
- **Cycle Management**: Automated cycle execution and tracking
- **Performance Aggregation**: Cross-cycle performance analysis
- **Early Intervention**: Automatic failure detection and recovery

**🧠 Intelligent Workflow:**
- **Framework Memory**: Historical performance tracking and learning
- **AI Integration**: Gemini AI-guided optimization and adaptation
- **Error Recovery**: Robust error handling with graceful degradation
- **State Management**: Comprehensive framework state tracking
- **Performance Monitoring**: Real-time cycle performance analysis

**📊 Comprehensive Analytics:**
- **Cycle Breakdown**: Per-cycle performance metrics and analysis
- **Feature Importance**: Aggregated SHAP analysis across cycles
- **Model Performance**: Classification reports and validation metrics
- **Strategy Evolution**: Genetic programming results and rule discovery
- **Risk Analysis**: Drawdown analysis and risk-adjusted metrics

**🎨 Output Generation:**
- **Text Reports**: Professional performance reports with cycle breakdown
- **Visualizations**: Equity curves, feature importance, trade analysis
- **Interactive Dashboards**: HTML dashboards with drill-down capabilities
- **Data Exports**: CSV exports for further analysis
- **Framework State**: Serialized framework memory and configuration

```bash
# Command line usage
python main.py --task complete --data-files "data/*.csv" --config config.json

# Python usage
from axiom_edge import CompleteFrameworkTask

task = CompleteFrameworkTask()
results = task.run_complete_framework(data_files)
```

**Features:**
- End-to-end automation
- Walk-forward analysis
- AI-driven optimization
- Comprehensive reporting

## 🔧 Configuration

### Configuration File Example

```json
{
    "BASE_PATH": "./data",
    "REPORT_LABEL": "My_Strategy_v1",
    "INITIAL_CAPITAL": 10000.0,
    "operating_state": "conservative_baseline",
    "FEATURE_SELECTION_METHOD": "mutual_info",
    "TRAINING_WINDOW": "30D",
    "RETRAINING_FREQUENCY": "7D",
    "FORWARD_TEST_GAP": "1D",
    "LOOKAHEAD_CANDLES": 5,
    "RISK_CAP_PER_TRADE_USD": 500.0,
    "BASE_RISK_PER_TRADE_PCT": 0.02,
    "CONFIDENCE_TIERS": {
        "high": 1.5,
        "medium": 1.0,
        "low": 0.5
    }
}
```

### Environment Variables

```bash
# Required for AI analysis
export GEMINI_API_KEY="your-gemini-api-key"

# Required for data collection
export FINANCIAL_API_KEY="your-alpha-vantage-key"
# or
export POLYGON_API_KEY="your-polygon-key"

# Optional: Broker API keys
export OANDA_API_KEY="your-oanda-key"
export IB_API_KEY="your-interactive-brokers-key"
```

## 🎯 Use Cases

### 1. Data Scientist - Feature Engineering Only
```python
from axiom_edge import FeatureEngineeringTask, DataCollectionTask

# Collect data
data_task = DataCollectionTask()
raw_data = data_task.collect_data(["AAPL"], "2023-01-01", "2024-01-01")

# Engineer features
feature_task = FeatureEngineeringTask()
features = feature_task.engineer_features(raw_data["AAPL"])
```

### 2. Quantitative Analyst - Strategy Backtesting
```python
from axiom_edge import BacktestTask

# Define your strategy
strategy = {
    "entry_condition": "RSI < 30 and SMA_10 > SMA_20",
    "exit_condition": "RSI > 70 or stop_loss_hit",
    "stop_loss_pct": 0.05,
    "take_profit_pct": 0.10
}

# Backtest
task = BacktestTask()
results = task.backtest_strategy(data, strategy)
```

### 3. Portfolio Manager - Broker Analysis
```python
from axiom_edge import BrokerInfoTask

# Compare brokers
task = BrokerInfoTask()
comparison = task.compare_brokers(
    symbols=["EURUSD", "GBPUSD", "USDJPY"],
    brokers=["oanda", "interactive_brokers", "forex_com"]
)
```

### 4. Algorithmic Trader - Complete Framework
```python
from axiom_edge import CompleteFrameworkTask

# Run full framework
task = CompleteFrameworkTask(config)
results = task.run_complete_framework(["data/EURUSD_1D.csv"])
```

## 📊 Output Examples

### Feature Engineering Output
```python
# Basic feature engineering
features = task.engineer_features(raw_data)
print(f"Generated {len(features.columns)} features")

# Output: Generated 247 features
# Feature categories:
# - Technical Indicators: 42 features
# - Price & Returns: 31 features
# - Volume Analysis: 18 features
# - Statistical Moments: 28 features
# - Time-based: 16 features
# - Pattern Recognition: 22 features
# - Advanced Analytics: 15 features
# - Multi-timeframe: 35 features
# - Other: 40 features
```

### Multi-task Labels Output
```python
labeled_data = feature_engineer.label_data_multi_task(features)

# Signal Pressure Classes (5-class system)
print(labeled_data['target_signal_pressure_class'].value_counts())
# Output:
# 2 (Hold): 156 samples (42.9%)
# 1 (Sell): 89 samples (24.5%)
# 3 (Buy): 78 samples (21.4%)
# 0 (Strong Sell): 23 samples (6.3%)
# 4 (Strong Buy): 18 samples (4.9%)

# Timing Scores (continuous -1 to 1)
print(labeled_data['target_timing_score'].describe())
# Output:
# mean: 0.023, std: 0.445, min: -1.0, max: 1.0

# Pattern Detection
print(f"Bullish Engulfing: {labeled_data['target_bullish_engulfing'].sum()} occurrences")
print(f"Bearish Engulfing: {labeled_data['target_bearish_engulfing'].sum()} occurrences")
print(f"Volatility Spikes: {labeled_data['target_volatility_spike'].sum()} occurrences")
```

### Data Collection Output
```
📊 Cache Info: {
    'in_memory_items': 2,
    'disk_cache_files': 5,
    'cache_directory': 'data_cache',
    'total_cache_size_mb': 15.7
}
```

### Backtest Results
```json
{
    "total_return": 0.157,
    "sharpe_ratio": 1.23,
    "max_drawdown": -0.08,
    "win_rate": 0.62,
    "total_trades": 45,
    "profit_factor": 1.8
}
```

### AI Analysis Output
```json
{
    "regime": "TRENDING_BULL",
    "confidence": 0.85,
    "analysis": "Strong upward momentum with low volatility",
    "recommended_strategy": "Trend following with momentum confirmation",
    "key_indicators": ["RSI_oversold", "volume_expansion", "trend_strength"],
    "parameter_adjustments": {
        "risk_per_trade": 0.025,
        "stop_loss_pct": 0.04
    }
}
```

### Feature Quality Analysis
```
📊 Feature Quality Report:
   Total features: 247
   Features with missing values: 12 (4.9%)
   High variance features: 89
   Low variance features: 23
   Highly correlated pairs (|r| > 0.9): 15

🔗 Top Feature Correlations:
   1. EMA_20 <-> EMA_21: 0.998
   2. RSI_14 <-> RSI_21: 0.892
   3. BB_upper_20 <-> BB_middle_20: 0.945
```

## 🔍 Advanced Features

### 🧬 Comprehensive Feature Engineering
```python
# Generate 200+ features with advanced analytics
from axiom_edge import FeatureEngineer, create_default_config

config = create_default_config("./")
config.EMA_PERIODS = [8, 13, 21, 34, 55, 89, 144, 233]  # Fibonacci sequence
config.RSI_STANDARD_PERIODS = [14, 21, 28]
config.LOOKAHEAD_CANDLES = 5

# Multi-timeframe feature engineering
timeframe_roles = {'base': 'D1', 'higher': 'W1'}
feature_engineer = FeatureEngineer(config, timeframe_roles, {})

# Prepare multi-timeframe data
data_by_tf = {
    'D1': daily_data,    # Base timeframe
    'W1': weekly_data,   # Higher timeframe for trend context
    'H4': hourly_data    # Lower timeframe for entry timing
}

# Engineer comprehensive feature set
features = feature_engineer.engineer_features(daily_data, data_by_tf)
print(f"Generated {len(features.columns)} features across multiple timeframes")

# Generate multi-task labels
labeled_data = feature_engineer.label_data_multi_task(features)
print("Labels generated:")
print(f"- Signal Pressure Classes: {labeled_data['target_signal_pressure_class'].value_counts()}")
print(f"- Timing Scores: {labeled_data['target_timing_score'].describe()}")
```

### 🎯 Advanced Feature Analysis
```python
# Analyze feature quality and relationships
def analyze_features(features):
    # Feature categories
    categories = {
        "Technical Indicators": [col for col in features.columns if any(x in col.lower() for x in ['rsi', 'macd', 'ema', 'bb_', 'atr'])],
        "Volume Analysis": [col for col in features.columns if 'volume' in col.lower()],
        "Statistical Moments": [col for col in features.columns if any(x in col.lower() for x in ['mean', 'std', 'skew', 'kurt'])],
        "Pattern Recognition": [col for col in features.columns if any(x in col.lower() for x in ['doji', 'wick', 'body', 'candle'])],
        "Time-based": [col for col in features.columns if any(x in col.lower() for x in ['hour', 'day', 'session'])],
        "Advanced Analytics": [col for col in features.columns if any(x in col.lower() for x in ['entropy', 'cycle', 'fourier'])]
    }

    for category, features_list in categories.items():
        print(f"{category}: {len(features_list)} features")

    return categories

feature_categories = analyze_features(features)
```

### 🔬 Custom Strategy Development with Rich Features
```python
# Define advanced strategy using engineered features
def advanced_strategy_rules(features):
    signals = pd.DataFrame(index=features.index)

    # Multi-indicator confluence
    rsi_oversold = features['RSI_14'] < 30
    bb_squeeze = features['BB_width_20'] < features['BB_width_20'].rolling(20).quantile(0.2)
    volume_spike = features['volume_ma_ratio'] > 1.5
    trend_up = features['EMA_20'] > features['EMA_50']

    # Advanced pattern recognition
    bullish_divergence = (features['Close'] < features['Close'].shift(5)) & (features['RSI_14'] > features['RSI_14'].shift(5))
    volatility_expansion = features['ATR'] > features['ATR'].rolling(20).mean() * 1.2

    # Generate signals with confluence
    signals['signal'] = 0
    buy_conditions = rsi_oversold & trend_up & volume_spike & bullish_divergence
    sell_conditions = ~trend_up & volatility_expansion

    signals.loc[buy_conditions, 'signal'] = 1
    signals.loc[sell_conditions, 'signal'] = -1

    # Add confidence scoring
    signals['confidence'] = 0.0
    signals.loc[buy_conditions, 'confidence'] = (
        features.loc[buy_conditions, 'RSI_14'].apply(lambda x: (30-x)/30) * 0.3 +
        features.loc[buy_conditions, 'volume_ma_ratio'].apply(lambda x: min(x/2, 1)) * 0.3 +
        0.4  # Base confidence
    )

    return signals

# Use with backtester
task = BacktestTask()
results = task.backtest_strategy(features, advanced_strategy_rules)
```

### 🌐 Multi-Asset Portfolio with Feature Engineering
```python
# Collect and engineer features for multiple assets
symbols = ["AAPL", "GOOGL", "MSFT", "TSLA", "SPY"]
data_task = DataCollectionTask()
feature_task = FeatureEngineeringTask()

portfolio_features = {}
for symbol in symbols:
    print(f"Processing {symbol}...")

    # Collect data
    raw_data = data_task.collect_data([symbol], "2023-01-01", "2024-01-01")

    # Engineer features
    features = feature_task.engineer_features(raw_data[symbol])
    portfolio_features[symbol] = features

    print(f"  Generated {len(features.columns)} features for {symbol}")

# Cross-asset feature analysis
def calculate_cross_asset_features(portfolio_features):
    # Calculate correlation features between assets
    symbols = list(portfolio_features.keys())

    for i, symbol1 in enumerate(symbols):
        for symbol2 in symbols[i+1:]:
            # Price correlation
            corr_20 = portfolio_features[symbol1]['Close'].rolling(20).corr(
                portfolio_features[symbol2]['Close']
            )
            portfolio_features[symbol1][f'corr_{symbol2}_20'] = corr_20

            # Volatility correlation
            vol_corr = portfolio_features[symbol1]['ATR'].rolling(20).corr(
                portfolio_features[symbol2]['ATR']
            )
            portfolio_features[symbol1][f'vol_corr_{symbol2}_20'] = vol_corr

calculate_cross_asset_features(portfolio_features)
```

## 🛠️ Development & Customization

### Adding Custom Features
```python
# Extend the FeatureEngineer class
class CustomFeatureEngineer(FeatureEngineer):
    def _calculate_custom_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add your custom technical indicators"""
        # Custom momentum indicator
        df['custom_momentum'] = (df['Close'] / df['Close'].shift(10) - 1) * 100

        # Custom volatility measure
        df['custom_volatility'] = df['Close'].rolling(20).std() / df['Close'].rolling(20).mean()

        # Custom pattern detection
        df['custom_pattern'] = (
            (df['Close'] > df['Open']) &
            (df['Close'].shift(1) < df['Open'].shift(1)) &
            (df['Volume'] > df['Volume'].rolling(20).mean() * 1.5)
        ).astype(int)

        return df

    def _process_single_symbol_stack(self, symbol_data_by_tf, macro_data=None):
        # Call parent method first
        df = super()._process_single_symbol_stack(symbol_data_by_tf, macro_data)

        # Add custom features
        if df is not None:
            df = self._calculate_custom_indicators(df)

        return df

# Use custom feature engineer
config = create_default_config("./")
custom_engineer = CustomFeatureEngineer(config, {'base': 'D1'}, {})
features = custom_engineer.engineer_features(data, {'D1': data})
```

### Creating Custom Tasks
```python
# Create custom task for specialized analysis
class TechnicalAnalysisTask(BaseTask):
    def __init__(self, config):
        super().__init__(config)
        self.feature_engineer = FeatureEngineer(config, {'base': 'D1'}, {})

    def analyze_support_resistance(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Custom support/resistance analysis"""
        # Engineer features first
        features = self.feature_engineer.engineer_features(data, {'D1': data})

        # Find support/resistance levels
        highs = features['High'].rolling(20).max()
        lows = features['Low'].rolling(20).min()

        current_price = features['Close'].iloc[-1]
        resistance = highs.iloc[-1]
        support = lows.iloc[-1]

        return {
            'current_price': current_price,
            'resistance': resistance,
            'support': support,
            'distance_to_resistance': (resistance - current_price) / current_price,
            'distance_to_support': (current_price - support) / current_price,
            'rsi_current': features['RSI_14'].iloc[-1],
            'trend_strength': features['EMA_20'].iloc[-1] - features['EMA_50'].iloc[-1]
        }

    def generate_trading_signals(self, data: pd.DataFrame) -> pd.DataFrame:
        """Generate custom trading signals"""
        features = self.feature_engineer.engineer_features(data, {'D1': data})

        signals = pd.DataFrame(index=features.index)
        signals['signal'] = 0
        signals['confidence'] = 0.0

        # Multi-factor signal generation
        rsi_oversold = features['RSI_14'] < 30
        rsi_overbought = features['RSI_14'] > 70
        trend_up = features['EMA_20'] > features['EMA_50']
        volume_spike = features['volume_ma_ratio'] > 1.3

        # Buy signals
        buy_conditions = rsi_oversold & trend_up & volume_spike
        signals.loc[buy_conditions, 'signal'] = 1
        signals.loc[buy_conditions, 'confidence'] = 0.8

        # Sell signals
        sell_conditions = rsi_overbought & ~trend_up
        signals.loc[sell_conditions, 'signal'] = -1
        signals.loc[sell_conditions, 'confidence'] = 0.7

        return signals

# Use custom task
task = TechnicalAnalysisTask(config)
analysis = task.analyze_support_resistance(data)
signals = task.generate_trading_signals(data)
```

### Extending Data Sources
```python
# Add custom data source
class CustomDataHandler(DataHandler):
    def _fetch_custom_crypto_source(self, symbol, start_date, end_date):
        """Custom cryptocurrency data source"""
        # Implement your custom API integration
        import requests

        url = f"https://api.custom-crypto.com/v1/ohlcv"
        params = {
            'symbol': symbol,
            'start': start_date,
            'end': end_date,
            'interval': '1d'
        }

        response = requests.get(url, params=params)
        data = response.json()

        # Convert to DataFrame
        df = pd.DataFrame(data['ohlcv'])
        df.columns = ['timestamp', 'Open', 'High', 'Low', 'Close', 'Volume']
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df.set_index('timestamp', inplace=True)

        return df

    def get_crypto_data(self, symbol, start_date, end_date):
        """Get cryptocurrency data"""
        return self._fetch_custom_crypto_source(symbol, start_date, end_date)

# Use custom data handler
handler = CustomDataHandler()
btc_data = handler.get_crypto_data("BTC-USD", "2023-01-01", "2024-01-01")
```

## 📊 **AxiomEdge vs. Traditional Backtesting Frameworks**

| Feature | AxiomEdge | backtesting.py | zipline | backtrader |
|---------|-----------|----------------|---------|------------|
| **AI Integration** | ✅ Gemini AI Doctor | ❌ | ❌ | ❌ |
| **Telemetry System** | ✅ Advanced JSONL | ❌ | ❌ | ❌ |
| **Walk-Forward Analysis** | ✅ Multi-cycle | ❌ | ❌ | ❌ |
| **Feature Engineering** | ✅ 200+ Features | ❌ | ❌ | ❌ |
| **Genetic Programming** | ✅ Strategy Evolution | ❌ | ❌ | ❌ |
| **SHAP Explainability** | ✅ Model Transparency | ❌ | ❌ | ❌ |
| **Multi-Task Learning** | ✅ Multiple Targets | ❌ | ❌ | ❌ |
| **Dynamic Ensembles** | ✅ Adaptive Weighting | ❌ | ❌ | ❌ |
| **Regime Detection** | ✅ Market Adaptation | ❌ | ❌ | ❌ |
| **Parameter Drift Detection** | ✅ Automated Monitoring | ❌ | ❌ | ❌ |
| **Interactive Dashboards** | ✅ Plotly + HTML | ❌ | ❌ | ❌ |
| **Framework Memory** | ✅ Historical Learning | ❌ | ❌ | ❌ |
| **Production Monitoring** | ✅ Health Checks | ❌ | ❌ | ❌ |
| **Basic Backtesting** | ✅ | ✅ | ✅ | ✅ |
| **Portfolio Management** | ✅ | ❌ | ✅ | ✅ |

## 🧪 Testing & Examples

### Run Comprehensive Examples
```bash
# Complete framework demonstration
python examples/complete_framework_demo.py

# Feature engineering demonstration
python examples/feature_engineering_demo.py

# Model training demonstration
python examples/model_training_demo.py

# Genetic programming demonstration
python examples/genetic_programming_demo.py

# Report generation demonstration
python examples/report_generation_demo.py

# Command line interface
python main.py --task complete --data-files "data/*.csv"
```

### Performance Testing
```python
# Test feature engineering performance
import time
from axiom_edge import FeatureEngineeringTask

# Create large dataset
large_data = create_realistic_market_data(5000)  # 5000 days

# Time feature engineering
start_time = time.time()
task = FeatureEngineeringTask()
features = task.engineer_features(large_data)
end_time = time.time()

print(f"Processed {len(large_data)} samples in {end_time - start_time:.2f} seconds")
print(f"Generated {len(features.columns)} features")
print(f"Processing rate: {len(large_data) / (end_time - start_time):.0f} samples/second")
```

## 📚 Documentation & Resources

### Core Documentation
- [Configuration Reference](docs/configuration.md) - Complete configuration options
- [API Documentation](docs/api.md) - Detailed API reference
- [Feature Engineering Guide](docs/feature_engineering.md) - ⭐ **NEW**: Complete feature guide
- [Strategy Development Guide](docs/strategies.md) - Strategy development best practices
- [Performance Optimization](docs/optimization.md) - Performance tuning guide
- [Troubleshooting](docs/troubleshooting.md) - Common issues and solutions

### Examples & Tutorials
- [Basic Usage Examples](examples/basic_usage.py) - Getting started
- [Feature Engineering Demo](examples/feature_engineering_demo.py) - ⭐ **NEW**: 200+ features demo
- [Advanced Strategies](examples/advanced_strategies.py) - Complex strategy examples
- [Multi-Asset Portfolio](examples/portfolio_optimization.py) - Portfolio-level analysis

### Research Papers & References
- Technical Analysis Indicators: Comprehensive implementation
- Multi-timeframe Analysis: Best practices and methodologies
- Feature Engineering for Finance: Academic research integration
- Machine Learning in Trading: Model development guidelines

## 🤝 Contributing

We welcome contributions! Here's how to get started:

### Development Setup
```bash
# Fork and clone the repository
git clone https://github.com/your-username/AxiomEdge.git
cd AxiomEdge

# Create development environment
python -m venv venv
source venv/bin/activate

# Install in development mode
pip install -e ".[dev]"

# Run tests
pytest tests/

# Run code quality checks
black axiom_edge/
flake8 axiom_edge/
mypy axiom_edge/
```

### Contribution Guidelines
1. **Fork the repository** and create a feature branch
2. **Add comprehensive tests** for new features
3. **Follow code style** (Black, PEP 8)
4. **Update documentation** for new features
5. **Submit a pull request** with detailed description

### Areas for Contribution
- **🌐 Live Trading Integration**: Real-time execution and monitoring
- **📊 Advanced Portfolio Optimization**: Multi-asset correlation analysis
- **🔄 Additional Data Sources**: Integration with more data providers
- **📱 Mobile Dashboard**: Mobile-friendly monitoring interface
- **🔐 Security Enhancements**: Advanced authentication and encryption
- **📚 Documentation**: Additional tutorials, examples, and guides
- **🧪 Testing**: Extended unit tests, integration tests, performance tests
- **🌍 Internationalization**: Multi-language support and global markets

## 📄 License & Legal

### License
This project is released under the **GNU General Public License v3.0 (GPL-3.0)**.

### Attribution
While you are free to use, modify, and distribute this software under the GPL-3.0 license, **it would be nice to be acknowledged for the original code** in any further developments or public-facing projects that build upon it. A simple credit or link back to the original repository is greatly appreciated.

### Disclaimer
**⚠️ IMPORTANT: This is an experimental framework for research purposes only. It is not financial advice. Trading financial markets involves substantial risk, and you can lose all of your invested capital. Past performance is not indicative of future results. Do not run this framework with real money without fully understanding the code and the risks involved.**

### Contact & Support
- **Issues**: [GitHub Issues](https://github.com/axiom-edge/axiom-edge/issues)
- **Discussions**: [GitHub Discussions](https://github.com/axiom-edge/axiom-edge/discussions)
- **Documentation**: [Read the Docs](https://axiom-edge.readthedocs.io/)

---

## 🎊 **Complete Implementation Achievement**

**AxiomEdge is now 100% FULLY IMPLEMENTED** - All components have been successfully modularized from the original monolithic codebase:

### ✅ **Core Components (All Complete)**
- **🔧 FeatureEngineer**: 200+ features with multi-timeframe analysis
- **🤖 ModelTrainer**: Advanced ML training with hyperparameter optimization
- **🧬 GeneticProgrammer**: Strategy evolution with genetic algorithms
- **📋 ReportGenerator**: Comprehensive performance reporting
- **🎯 Backtester**: Advanced backtesting with dynamic ensembles
- **🚀 FrameworkOrchestrator**: Complete workflow orchestration
- **📊 TelemetryCollector**: Advanced monitoring and analytics

### 🌟 **Unique Differentiators**
- **AI Doctor**: Continuous AI monitoring and optimization
- **Telemetry System**: Comprehensive performance tracking
- **Walk-Forward Analysis**: Robust out-of-sample validation
- **SHAP Explainability**: Model transparency and interpretability
- **Genetic Programming**: Automated strategy discovery
- **Dynamic Ensembles**: Adaptive model weighting
- **Framework Memory**: Historical learning and adaptation

### 🚀 **Production Ready**
- **Modular Architecture**: Use components independently or together
- **Scalable Processing**: Multi-core parallel processing
- **Robust Error Handling**: Graceful degradation and recovery
- **Comprehensive Logging**: Detailed audit trails and monitoring
- **Professional Reporting**: Publication-quality visualizations
- **Extensible Design**: Easy to customize and extend

**AxiomEdge represents the next generation of quantitative trading frameworks, combining traditional backtesting with cutting-edge AI, comprehensive telemetry, and scientific rigor.**

---

**Built with ❤️ for the quantitative finance and machine learning community**
