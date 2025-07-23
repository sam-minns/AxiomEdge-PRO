# AxiomEdge Trading Framework - Modular Architecture

AxiomEdge is a comprehensive AI-powered trading framework that can be used as individual components or as a complete integrated system. The modular design allows you to use only the parts you need for specific tasks.

## ⚠️ **IMPORTANT DISCLAIMER**

**This is an experimental framework for research purposes only. It is not financial advice. Trading financial markets involves substantial risk, and you can lose all of your invested capital. Past performance is not indicative of future results. Do not run this framework with real money without fully understanding the code and the risks involved.**

## 📄 **License & Attribution**

This project is released under a **GPL 3.0 permissive license**. While you are free to use, modify, and distribute this software, it would be nice to be acknowledged for the original code in any further developments or public-facing projects that build upon it. A simple credit or link back to the original repository is greatly appreciated.

## 🏗️ Architecture Overview

```
axiom_edge/
├── __init__.py              # Main package exports
├── config.py                # ✅ Configuration and validation
├── data_handler.py          # ✅ Data collection and caching
├── ai_analyzer.py           # ✅ AI analysis with Gemini
├── feature_engineer.py      # ✅ 200+ Feature engineering (COMPLETE)
├── model_trainer.py         # 🚧 ML model training (stub)
├── backtester.py           # 🚧 Strategy backtesting (stub)
├── genetic_programmer.py   # 🚧 Genetic algorithm optimization (stub)
├── report_generator.py     # 🚧 Report generation (stub)
├── framework_orchestrator.py # 🚧 Complete framework orchestration (stub)
├── tasks.py                # ✅ Task-specific interfaces
├── stubs.py                # 🔄 Temporary implementations
└── utils.py                # ✅ Utility functions
```

### 🎯 **Implementation Status**
- **✅ Fully Implemented**: Ready for production use
- **🚧 Stub Implementation**: Basic functionality, full implementation pending
- **🔄 Temporary**: Bridge implementations during modularization

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

### 5. Model Training Task
Train machine learning models for trading signal generation.

```bash
# Command line usage
python main.py --task model_training --data-file features.csv

# Python usage
from axiom_edge import ModelTrainingTask

task = ModelTrainingTask()
model_results = task.train_model(features, target)
```

**Features:**
- Multiple ML algorithms
- Hyperparameter optimization
- Cross-validation
- SHAP feature importance

### 6. Complete Framework Task
Run the full AxiomEdge framework with all components integrated.

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

## 🧪 Testing & Examples

### Run Comprehensive Examples
```bash
# Basic usage examples
python examples/basic_usage.py

# Feature engineering demonstration
python examples/feature_engineering_demo.py

# Command line interface
python main.py --task features --data-file examples/sample_data.csv
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
- **🧠 Model Trainer**: Complete ML pipeline implementation
- **📊 Backtester**: Advanced backtesting engine
- **🧬 Genetic Programmer**: Strategy evolution algorithms
- **📈 Report Generator**: Rich visualization and reporting
- **🔧 Framework Orchestrator**: Complete workflow automation
- **📚 Documentation**: Tutorials, examples, guides
- **🧪 Testing**: Unit tests, integration tests, performance tests

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

**Built with ❤️ for the quantitative finance and machine learning community**
