# AxiomEdge Trading Framework - Modular Architecture

AxiomEdge is a comprehensive AI-powered trading framework that can be used as individual components or as a complete integrated system. The modular design allows you to use only the parts you need for specific tasks.

## 🏗️ Architecture Overview

```
axiom_edge/
├── __init__.py              # Main package exports
├── config.py                # Configuration and validation
├── data_handler.py          # Data collection and caching
├── ai_analyzer.py           # AI analysis with Gemini
├── feature_engineer.py      # Feature engineering
├── model_trainer.py         # ML model training
├── backtester.py           # Strategy backtesting
├── genetic_programmer.py   # Genetic algorithm optimization
├── report_generator.py     # Report generation
├── framework_orchestrator.py # Complete framework orchestration
├── tasks.py                # Task-specific interfaces
└── utils.py                # Utility functions
```

## 🚀 Quick Start

### Installation

```bash
# Clone the repository
git clone <repository-url>
cd AxiomEdge

# Install dependencies
pip install -r requirements.txt

# Set up environment variables
export GEMINI_API_KEY="your-gemini-api-key"
export FINANCIAL_API_KEY="your-financial-data-api-key"
```

### Basic Usage

```python
# Import specific components
from axiom_edge import DataCollectionTask, BacktestTask, ConfigModel

# Create configuration
config = ConfigModel(
    BASE_PATH="./data",
    REPORT_LABEL="My_Strategy",
    INITIAL_CAPITAL=10000.0,
    # ... other parameters
)

# Use individual components
data_task = DataCollectionTask(config)
data = data_task.collect_data(["AAPL", "GOOGL"], "2023-01-01", "2024-01-01")
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

### 4. Feature Engineering Task
Engineer features from raw price data for machine learning models.

```bash
# Command line usage
python main.py --task features --data-file data.csv --output features.csv

# Python usage
from axiom_edge import FeatureEngineeringTask

task = FeatureEngineeringTask()
features = task.engineer_features(raw_data)
selected_features = task.select_features(features, "target_column")
```

**Features:**
- 200+ technical indicators
- Multi-timeframe feature fusion
- Automated feature selection
- Custom feature engineering

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
    "recommended_strategy": "Trend following with momentum confirmation"
}
```

## 🔍 Advanced Features

### Custom Strategy Development
```python
# Define custom strategy rules
def my_strategy_rules(data):
    signals = pd.DataFrame(index=data.index)
    
    # Your custom logic here
    rsi = calculate_rsi(data['Close'], 14)
    macd = calculate_macd(data['Close'])
    
    # Generate signals
    signals['signal'] = 0
    signals.loc[(rsi < 30) & (macd > 0), 'signal'] = 1
    signals.loc[(rsi > 70) & (macd < 0), 'signal'] = -1
    
    return signals

# Use with backtester
task = BacktestTask()
results = task.backtest_custom_strategy(data, my_strategy_rules)
```

### Multi-Asset Portfolio
```python
# Collect data for multiple assets
symbols = ["AAPL", "GOOGL", "MSFT", "TSLA"]
data_task = DataCollectionTask()

portfolio_data = {}
for symbol in symbols:
    portfolio_data[symbol] = data_task.collect_data([symbol], "2023-01-01", "2024-01-01")

# Run portfolio-level analysis
# ... portfolio optimization logic
```

## 🛠️ Development

### Adding Custom Components
```python
# Create custom task
class MyCustomTask(BaseTask):
    def __init__(self, config):
        super().__init__(config)
    
    def my_custom_method(self, data):
        # Your custom logic
        return results

# Use custom task
from axiom_edge.tasks import BaseTask
task = MyCustomTask(config)
results = task.my_custom_method(data)
```

### Extending Existing Components
```python
# Extend data handler
class MyDataHandler(DataHandler):
    def _fetch_custom_source(self, symbol, start_date, end_date):
        # Custom data source implementation
        return data

# Use extended handler
handler = MyDataHandler()
data = handler.get_data("AAPL", "2023-01-01", "2024-01-01", source="custom")
```

## 📚 Documentation

- [Configuration Reference](docs/configuration.md)
- [API Documentation](docs/api.md)
- [Strategy Development Guide](docs/strategies.md)
- [Performance Optimization](docs/optimization.md)
- [Troubleshooting](docs/troubleshooting.md)

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests
5. Submit a pull request

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.
