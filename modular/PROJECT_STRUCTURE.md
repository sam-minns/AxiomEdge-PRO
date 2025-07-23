# AxiomEdge Project Structure

## 📁 Directory Layout

```
AxiomEdge/
├── axiom_edge/                    # Main package directory
│   ├── __init__.py                # Package initialization and exports
│   ├── config.py                  # Configuration models and validation
│   ├── data_handler.py            # Data collection and caching
│   ├── ai_analyzer.py             # AI analysis with Gemini API
│   ├── feature_engineer.py        # Feature engineering
│   ├── model_trainer.py           # ML model training
│   ├── backtester.py              # Strategy backtesting
│   ├── genetic_programmer.py      # Genetic algorithm optimization
│   ├── report_generator.py        # Report generation
│   ├── framework_orchestrator.py  # Complete framework orchestration
│   ├── telemetry.py               # Advanced telemetry and monitoring
│   ├── tasks.py                   # Task-specific interfaces
│   └── utils.py                   # Utility functions
│
├── examples/                     # Usage examples
│   ├── basic_usage.py            # Basic usage examples
│   ├── advanced_strategies.py    # Advanced strategy examples
│   ├── custom_features.py        # Custom feature engineering
│   └── portfolio_optimization.py # Portfolio-level examples
│
├── configs/                      # Configuration files
│   ├── default_config.json       # Default configuration
│   ├── conservative_config.json  # Conservative trading config
│   ├── aggressive_config.json    # Aggressive trading config
│   └── research_config.json      # Research/experimental config
│
├── docs/                        # Documentation
│   ├── api/                     # API documentation
│   ├── tutorials/               # Step-by-step tutorials
│   ├── strategies/              # Strategy development guides
│   └── troubleshooting/         # Common issues and solutions
│
├── tests/                       # Test suite
│   ├── unit/                    # Unit tests
│   ├── integration/             # Integration tests
│   └── performance/             # Performance tests
│
├── data/                        # Data directory (created at runtime)
│   ├── raw/                     # Raw data files
│   ├── processed/               # Processed data
│   └── cache/                   # Cached data
│
├── results/                     # Results directory (created at runtime)
│   ├── backtests/               # Backtest results
│   ├── models/                  # Trained models
│   ├── features/                # Feature engineering outputs
│   └── reports/                 # Generated reports
│
├── logs/                        # Log files (created at runtime)
│   ├── application.log          # Main application logs
│   ├── errors.log               # Error logs
│   └── performance.log          # Performance logs
│
├── main.py                      # Main entry point script
├── setup.py                     # Package setup configuration
├── requirements.txt             # Python dependencies
├── README.md                    # Main documentation
├── PROJECT_STRUCTURE.md         # This file
├── LICENSE                      # License file
├── .gitignore                   # Git ignore rules
└── AxiomEdge_PRO_V211_NEW.py   # Original monolithic file (for reference)
```

## 🏗️ Modular Architecture

### Core Components

#### 1. **Configuration System** (`config.py`)
- **Purpose**: Centralized configuration management
- **Features**: 
  - Pydantic-based validation
  - Multiple operating states
  - Environment-specific configs
- **Usage**: `ConfigModel`, `create_default_config()`

#### 2. **Data Handler** (`data_handler.py`)
- **Purpose**: Data collection and management
- **Features**:
  - Multi-source data collection (Alpha Vantage, Yahoo, Polygon)
  - Intelligent caching system
  - Data validation and cleaning
- **Usage**: `DataHandler`, `DataLoader`

#### 3. **AI Analyzer** (`ai_analyzer.py`)
- **Purpose**: AI-powered analysis and insights
- **Features**:
  - Gemini API integration
  - Market regime analysis
  - Strategy optimization recommendations
- **Usage**: `GeminiAnalyzer`, `APITimer`

#### 4. **Task Interfaces** (`tasks.py`)
- **Purpose**: Individual component usage
- **Features**:
  - Independent task execution
  - Flexible parameter configuration
  - Standardized interfaces
- **Usage**: `DataCollectionTask`, `BacktestTask`, etc.

### Fully Implemented Components ✅

#### 5. **Feature Engineer** (`feature_engineer.py`)
- **Purpose**: Feature engineering and data preparation
- **Status**: ✅ **COMPLETE** - Full 200+ feature engineering pipeline
- **Features**: Technical indicators, statistical measures, pattern recognition, multi-timeframe analysis

#### 6. **Model Trainer** (`model_trainer.py`)
- **Purpose**: ML model training and validation
- **Status**: ✅ **COMPLETE** - Advanced ML pipeline with hyperparameter optimization
- **Features**: Optuna optimization, SHAP analysis, ensemble learning, cross-validation

#### 7. **Backtester** (`backtester.py`)
- **Purpose**: Strategy backtesting and performance analysis
- **Status**: ✅ **COMPLETE** - Advanced backtesting with comprehensive risk management
- **Features**: Dynamic ensembles, realistic execution, take-profit ladders, performance analytics

#### 8. **Genetic Programmer** (`genetic_programmer.py`)
- **Purpose**: Genetic algorithm optimization
- **Status**: ✅ **COMPLETE** - Full genetic programming for strategy evolution
- **Features**: Rule evolution, fitness evaluation, crossover/mutation, AI-guided optimization

#### 9. **Report Generator** (`report_generator.py`)
- **Purpose**: Performance reporting and visualization
- **Status**: ✅ **COMPLETE** - Rich HTML/PDF reports with interactive charts
- **Features**: Professional reports, Plotly dashboards, SHAP visualizations, performance metrics

#### 10. **Framework Orchestrator** (`framework_orchestrator.py`)
- **Purpose**: Complete framework coordination
- **Status**: ✅ **COMPLETE** - Full walk-forward analysis pipeline
- **Features**: Multi-cycle validation, component integration, error recovery, performance tracking

#### 11. **Telemetry Collector** (`telemetry.py`)
- **Purpose**: Advanced monitoring and analytics
- **Status**: ✅ **COMPLETE** - Comprehensive telemetry system
- **Features**: Real-time logging, session management, performance analysis, data export

## 🎯 Usage Patterns

### 1. **Individual Component Usage**
```python
# Data collection only
from axiom_edge import DataCollectionTask
task = DataCollectionTask()
data = task.collect_data(["AAPL"], "2023-01-01", "2024-01-01")
```

### 2. **Task Combination**
```python
# Data + Features + Model
from axiom_edge import DataCollectionTask, FeatureEngineeringTask, ModelTrainingTask

data_task = DataCollectionTask()
feature_task = FeatureEngineeringTask()
model_task = ModelTrainingTask()

# Chain tasks together
data = data_task.collect_data(["AAPL"], "2023-01-01", "2024-01-01")
features = feature_task.engineer_features(data["AAPL"])
model = model_task.train_model(features[:-1], features["target"])
```

### 3. **Complete Framework**
```python
# Full framework execution
from axiom_edge import CompleteFrameworkTask
task = CompleteFrameworkTask()
results = task.run_complete_framework(["data/*.csv"])
```

### 4. **Command Line Usage**
```bash
# Individual tasks
python main.py --task data_collection --symbols AAPL,GOOGL
python main.py --task backtest --data-file data.csv
python main.py --task complete --data-files "data/*.csv"
```

## 🔧 Development Workflow

### Phase 1: Modular Foundation ✅
- [x] Modular package structure
- [x] Configuration system
- [x] Data handler
- [x] AI analyzer
- [x] Task interfaces
- [x] Basic examples

### Phase 2: Core Component Extraction ✅
- [x] Extract FeatureEngineer from monolithic file
- [x] Extract ModelTrainer from monolithic file
- [x] Extract Backtester from monolithic file
- [x] Extract GeneticProgrammer from monolithic file
- [x] Extract ReportGenerator from monolithic file
- [x] Extract TelemetryCollector from monolithic file

### Phase 3: Framework Integration ✅
- [x] Create FrameworkOrchestrator
- [x] Implement walk-forward analysis
- [x] Add comprehensive examples
- [x] Professional documentation

### Phase 4: Advanced Features 🚀
- [ ] Web interface
- [ ] Real-time trading integration
- [ ] Advanced visualization
- [ ] Cloud deployment support

## 🎊 **MODULARIZATION**

**All core components fully implemented:**

### ✅ **Completed Components:**
- **FeatureEngineer**: 200+ features with multi-timeframe analysis
- **ModelTrainer**: Advanced ML training with hyperparameter optimization
- **Backtester**: Sophisticated backtesting with dynamic ensembles
- **GeneticProgrammer**: Strategy evolution with genetic algorithms
- **ReportGenerator**: Professional reporting with interactive visualizations
- **FrameworkOrchestrator**: Complete workflow orchestration
- **TelemetryCollector**: Advanced monitoring and analytics

### 🌟 **Key Achievements:**
- **100% Modular**: All components can be used independently
- **Production Ready**: Comprehensive error handling and validation
- **AI-Powered**: Gemini AI integration throughout the framework
- **Scientifically Rigorous**: Walk-forward analysis and SHAP explainability
- **Professional Quality**: Publication-ready reports and visualizations

### 🚀 **Unique Features Not Found in Other Frameworks:**
- **AI Doctor**: Continuous AI monitoring and optimization
- **Advanced Telemetry**: Comprehensive JSONL-based tracking
- **Genetic Programming**: Automated strategy discovery
- **Dynamic Ensembles**: Adaptive model weighting
- **Framework Memory**: Historical learning and adaptation

## 📦 Installation & Setup

### Development Installation
```bash
# Clone repository
git clone https://github.com/sam-minns/AxiomEdge-PRO.git
cd AxiomEdge-PRO/modular

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install in development mode
pip install -e .

# Install development dependencies
pip install -e ".[dev]"
```

### Production Installation
```bash
pip install axiom-edge
```

### Configuration
```bash
# Set environment variables
export GEMINI_API_KEY="your-gemini-api-key"
export FINANCIAL_API_KEY="your-financial-data-api-key"

# Create data directories
python -c "from axiom_edge.utils import create_directory_structure; create_directory_structure('./')"
```

## 🧪 Testing

### Run Tests
```bash
# All tests
pytest

# Unit tests only
pytest tests/unit/

# Integration tests
pytest tests/integration/

# With coverage
pytest --cov=axiom_edge
```

### Example Usage
```bash
# Run basic examples
python examples/basic_usage.py

# Run specific example
python main.py --task data_collection --symbols AAPL --start 2023-01-01 --end 2024-01-01
```

## 📚 Documentation

### API Documentation
- Located in `docs/api/`
- Generated with Sphinx
- Available online at [documentation-url]

### Tutorials
- Step-by-step guides in `docs/tutorials/`
- Strategy development in `docs/strategies/`
- Troubleshooting in `docs/troubleshooting/`

## 🤝 Contributing

### Development Guidelines
1. Follow PEP 8 style guidelines
2. Add type hints to all functions
3. Write comprehensive docstrings
4. Include unit tests for new features
5. Update documentation

### Pull Request Process
1. Fork the [repository](https://github.com/sam-minns/AxiomEdge-PRO)
2. Create a feature branch
3. Make your changes
4. Add tests
5. Update documentation
6. Submit pull request to the main repository

### Getting Help
- **Issues**: [Report bugs or request features](https://github.com/sam-minns/AxiomEdge-PRO/issues)
- **Discussions**: [Community discussions](https://github.com/sam-minns/AxiomEdge-PRO/discussions)
- **Wiki**: [Project documentation](https://github.com/sam-minns/AxiomEdge-PRO/wiki)

## 📄 License

This project is licensed under the **GNU General Public License v3.0** - see the [LICENSE](https://github.com/sam-minns/AxiomEdge-PRO/blob/main/LICENSE) file for details.

### License Summary
- **License**: GNU GPL 3.0
- **Permissions**: Commercial use, modification, distribution, patent use, private use
- **Conditions**: License and copyright notice, state changes, disclose source, same license
- **Limitations**: Liability, warranty

The GPL 3.0 license ensures that AxiomEdge remains open source and that any derivatives or modifications are also made available under the same terms, fostering community collaboration and transparency.
