# AxiomEdge Project Structure

## 📁 Directory Layout

```
AxiomEdge/
├── axiom_edge/                    # Main package directory
│   ├── __init__.py               # Package initialization and exports
│   ├── config.py                 # Configuration models and validation
│   ├── data_handler.py           # Data collection and caching
│   ├── ai_analyzer.py            # AI analysis with Gemini API
│   ├── feature_engineer.py       # Feature engineering (stub)
│   ├── model_trainer.py          # ML model training (stub)
│   ├── backtester.py            # Strategy backtesting (stub)
│   ├── genetic_programmer.py    # Genetic algorithm optimization (stub)
│   ├── report_generator.py      # Report generation (stub)
│   ├── framework_orchestrator.py # Complete framework orchestration (stub)
│   ├── tasks.py                 # Task-specific interfaces
│   ├── stubs.py                 # Temporary implementations
│   └── utils.py                 # Utility functions
│
├── examples/                     # Usage examples
│   ├── basic_usage.py           # Basic usage examples
│   ├── advanced_strategies.py   # Advanced strategy examples
│   ├── custom_features.py       # Custom feature engineering
│   └── portfolio_optimization.py # Portfolio-level examples
│
├── configs/                      # Configuration files
│   ├── default_config.json      # Default configuration
│   ├── conservative_config.json # Conservative trading config
│   ├── aggressive_config.json   # Aggressive trading config
│   └── research_config.json     # Research/experimental config
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

### Stub Components (To be fully modularized)

#### 5. **Feature Engineer** (`stubs.py` → `feature_engineer.py`)
- **Purpose**: Feature engineering and data preparation
- **Current**: Stub implementation with basic features
- **Future**: Full 200+ feature engineering pipeline

#### 6. **Model Trainer** (`stubs.py` → `model_trainer.py`)
- **Purpose**: ML model training and validation
- **Current**: Basic RandomForest implementation
- **Future**: Advanced ML pipeline with hyperparameter optimization

#### 7. **Backtester** (`stubs.py` → `backtester.py`)
- **Purpose**: Strategy backtesting and performance analysis
- **Current**: Simple return-based backtesting
- **Future**: Advanced backtesting with risk management

#### 8. **Genetic Programmer** (`stubs.py` → `genetic_programmer.py`)
- **Purpose**: Genetic algorithm optimization
- **Current**: Placeholder implementation
- **Future**: Full genetic programming for strategy evolution

#### 9. **Report Generator** (`stubs.py` → `report_generator.py`)
- **Purpose**: Performance reporting and visualization
- **Current**: Basic text reports
- **Future**: Rich HTML/PDF reports with charts

#### 10. **Framework Orchestrator** (`stubs.py` → `framework_orchestrator.py`)
- **Purpose**: Complete framework coordination
- **Current**: Basic workflow coordination
- **Future**: Full walk-forward analysis pipeline

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

### Phase 1: Current State ✅
- [x] Modular package structure
- [x] Configuration system
- [x] Data handler
- [x] AI analyzer
- [x] Task interfaces
- [x] Stub implementations
- [x] Basic examples

### Phase 2: Core Component Extraction 🚧
- [ ] Extract FeatureEngineer from monolithic file
- [ ] Extract ModelTrainer from monolithic file
- [ ] Extract Backtester from monolithic file
- [ ] Extract GeneticProgrammer from monolithic file
- [ ] Extract ReportGenerator from monolithic file

### Phase 3: Framework Integration 📋
- [ ] Create FrameworkOrchestrator
- [ ] Implement walk-forward analysis
- [ ] Add comprehensive testing
- [ ] Performance optimization

### Phase 4: Advanced Features 🚀
- [ ] Web interface
- [ ] Real-time trading integration
- [ ] Advanced visualization
- [ ] Cloud deployment support

## 📦 Installation & Setup

### Development Installation
```bash
# Clone repository
git clone <repository-url>
cd AxiomEdge

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
1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests
5. Update documentation
6. Submit pull request

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.
