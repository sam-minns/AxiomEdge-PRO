# AxiomEdge Examples

This directory contains comprehensive examples demonstrating all capabilities of the AxiomEdge framework. Each example is self-contained and showcases different aspects of the framework.

## 🚀 Quick Start

**New to AxiomEdge?** Start here:

```bash
python examples/quick_start.py
```

This will give you a 5-minute overview of the framework's core capabilities.

## 📚 Example Files

### 1. **quick_start.py** - *Beginner Friendly*
**Perfect for first-time users**
- Framework overview and installation check
- Basic data creation and feature engineering
- Simple model training
- Quick performance analysis

```bash
python examples/quick_start.py
```

### 2. **basic_usage.py** - *Core Components*
**Learn individual framework components**
- Data collection and validation
- Feature engineering basics
- Model training fundamentals
- Backtesting introduction
- Report generation

```bash
python examples/basic_usage.py
```

### 3. **complete_axiom_edge_demo.py** - *Full Framework Showcase*
**See what makes AxiomEdge unique**
- Complete framework orchestration
- AI Doctor integration
- Advanced telemetry system
- Walk-forward analysis
- SHAP explainability
- Genetic programming
- Framework memory

```bash
python examples/complete_axiom_edge_demo.py
```

### 4. **complete_framework_demo.py** - *Production Workflow*
**Real-world framework usage**
- Multi-symbol analysis
- Task-based interfaces
- Single cycle analysis
- Framework orchestration
- Performance monitoring

```bash
python examples/complete_framework_demo.py
```

### 5. **data_collection_demo.py** - *Data Management*
**Master data handling**
- Multi-source data collection
- Data caching and retrieval
- Quality validation
- Multi-timeframe handling
- Error recovery

```bash
python examples/data_collection_demo.py
```

### 6. **feature_engineering_demo.py** - *200+ Features*
**Comprehensive feature engineering**
- Technical indicators (RSI, MACD, Bollinger Bands, etc.)
- Statistical features (moments, entropy, cycles)
- Pattern recognition (candlestick patterns)
- Multi-timeframe fusion
- Feature quality analysis

```bash
python examples/feature_engineering_demo.py
```

### 7. **model_training_demo.py** - *Advanced ML*
**Machine learning mastery**
- Hyperparameter optimization with Optuna
- Feature selection strategies
- SHAP explainability
- Multi-task learning
- Model comparison
- Advanced validation

```bash
python examples/model_training_demo.py
```

### 8. **genetic_programming_demo.py** - *Strategy Evolution*
**Automated strategy discovery**
- Genetic algorithm evolution
- Rule-based strategy generation
- Fitness evaluation
- Population management
- Gene pool optimization
- AI-guided evolution

```bash
python examples/genetic_programming_demo.py
```

### 9. **backtesting_demo.py** - *Strategy Testing*
**Comprehensive backtesting**
- Strategy performance analysis
- Risk metrics calculation
- Walk-forward validation
- AI-enhanced insights
- Multi-scenario testing

```bash
python examples/backtesting_demo.py
```

### 10. **report_generation_demo.py** - *Professional Reports*
**Publication-quality outputs**
- Performance metrics calculation
- Static visualizations (matplotlib)
- Interactive dashboards (plotly)
- SHAP feature importance plots
- Custom report sections
- Multi-scenario comparison

```bash
python examples/report_generation_demo.py
```

## 🎯 Learning Path

### **Beginner** (New to AxiomEdge)
1. `quick_start.py` - Get familiar with the framework
2. `basic_usage.py` - Learn core components
3. `data_collection_demo.py` - Understand data handling

### **Intermediate** (Some experience)
1. `feature_engineering_demo.py` - Master feature creation
2. `model_training_demo.py` - Advanced machine learning
3. `backtesting_demo.py` - Strategy testing

### **Advanced** (Ready for production)
1. `complete_framework_demo.py` - Full workflow
2. `genetic_programming_demo.py` - Strategy evolution
3. `complete_axiom_edge_demo.py` - Unique capabilities

### **Expert** (Framework mastery)
1. `report_generation_demo.py` - Professional outputs
2. Combine multiple examples for custom workflows
3. Extend framework with custom components

## 🔧 Prerequisites

### Required Dependencies
```bash
pip install -r requirements.txt
```

### Optional (for full functionality)
- **Gemini API Key**: For AI-enhanced analysis
- **Data API Keys**: For real market data (Yahoo Finance, Alpha Vantage, etc.)
- **GPU Support**: For faster model training

## 📊 Expected Outputs

Each example generates various outputs:

### **Data Files**
- `sample_data/` - Generated sample datasets
- `cache/` - Cached data for faster reruns

### **Results**
- `Results/` - Performance reports and visualizations
- `telemetry_logs/` - Framework monitoring data

### **Visualizations**
- `equity_curve.png` - Portfolio performance
- `shap_summary.png` - Feature importance
- `trade_analysis.png` - Trade distribution
- `performance_dashboard.html` - Interactive dashboard

## 🚨 Troubleshooting

### Common Issues

**Import Errors**
```bash
# Make sure you're in the correct directory
cd /path/to/AxiomEdge
python examples/quick_start.py
```

**Missing Dependencies**
```bash
pip install -r requirements.txt
```

**API Key Issues**
- Examples work without API keys (use sample data)
- For real data, configure API keys in config

**Memory Issues**
- Reduce dataset sizes in examples
- Use smaller population sizes for genetic programming

### Getting Help

1. **Check the logs** - Examples provide detailed output
2. **Start simple** - Begin with `quick_start.py`
3. **Read the code** - Examples are well-commented
4. **Modify parameters** - Experiment with different settings

## 🎨 Customization

### Modify Examples
All examples are designed to be easily customized:

```python
# Change data parameters
market_data = create_sample_data(n_days=500)  # More data

# Adjust model settings
config.OPTUNA_TRIALS = 50  # More optimization

# Modify strategy parameters
strategy_config["parameters"]["fast_period"] = 5  # Faster signals
```

### Create Your Own
Use examples as templates for your own analysis:

```python
# Copy and modify any example
cp examples/basic_usage.py my_custom_analysis.py
# Edit my_custom_analysis.py with your data and parameters
```

## 📈 Performance Notes

### Execution Times (approximate)
- `quick_start.py`: 30 seconds
- `basic_usage.py`: 2-3 minutes
- `feature_engineering_demo.py`: 1-2 minutes
- `model_training_demo.py`: 3-5 minutes
- `genetic_programming_demo.py`: 5-10 minutes
- `complete_framework_demo.py`: 10-15 minutes

### Optimization Tips
- Use smaller datasets for faster testing
- Reduce optimization trials for quicker results
- Enable caching for repeated runs
- Use GPU for model training if available

## 🌟 What Makes AxiomEdge Unique

These examples showcase capabilities **not available in other frameworks**:

### **vs. backtesting.py**
- ✅ AI Doctor integration
- ✅ Advanced telemetry
- ✅ 200+ engineered features
- ✅ Genetic programming

### **vs. zipline**
- ✅ Walk-forward analysis
- ✅ SHAP explainability
- ✅ Framework memory
- ✅ Real-time monitoring

### **vs. backtrader**
- ✅ ML model integration
- ✅ Multi-task learning
- ✅ Automated optimization
- ✅ Professional reporting

## 🔮 Next Steps

After running the examples:

1. **Use your own data** - Replace sample data with real market data
2. **Customize strategies** - Modify parameters and rules
3. **Integrate APIs** - Connect to live data feeds
4. **Deploy models** - Use trained models for live trading
5. **Scale up** - Run on larger datasets and longer timeframes

---

**Happy Trading with AxiomEdge! 🚀**
