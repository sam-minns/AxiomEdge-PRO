You're right. The previous README focused on the high-level capabilities. To make it more comprehensive, I'll add a new section detailing the core classes and their key methods, explaining how predictions are generated, explained, and acted upon.

Here is the updated, more comprehensive README with the new **"Core Components & Methods"** section included.

-----

# AxiomEdge Professional Trading Framework v2.1.1

[](https://github.com/sam-minns/AxiomEdge-PRO)
[](https://github.com/sam-minns/AxiomEdge-PRO/blob/main/LICENSE)
[](https://python.org)
[](https://github.com/sam-minns/AxiomEdge-PRO/issues)
[](https://github.com/sam-minns/AxiomEdge-PRO/discussions)

AxiomEdge is a comprehensive, AI-powered algorithmic trading framework designed for the end-to-end development, validation, and analysis of sophisticated quantitative strategies. Its modular capabilities are integrated into a powerful, single-script orchestrator.

## 🔗 **Quick Links**

  - 📂 **[Repository](https://github.com/sam-minns/AxiomEdge-PRO)** - Main GitHub repository
  - 📋 **[Issues](https://github.com/sam-minns/AxiomEdge-PRO/issues)** - Bug reports and feature requests
  - 💬 **[Discussions](https://github.com/sam-minns/AxiomEdge-PRO/discussions)** - Community support and questions
  - 📚 **[Wiki](https://github.com/sam-minns/AxiomEdge-PRO/wiki)** - Detailed documentation and guides
  - 🚀 **[Releases](https://github.com/sam-minns/AxiomEdge-PRO/releases)** - Latest versions and changelogs

-----

## ⚠️ **IMPORTANT DISCLAIMER**

**This is an experimental framework for research purposes only. It is not financial advice. Trading financial markets involves substantial risk, and you can lose all of your invested capital. Past performance is not indicative of future results. Do not run this framework with real money without fully understanding the code and the risks involved.**

-----

## 🌟 **What Makes AxiomEdge Unique**

Unlike traditional backtesting libraries, AxiomEdge is architected around an **AI System Architect (Gemini)** that actively manages, adapts, and learns throughout the entire strategy development lifecycle.

### 🧠 **AI-Powered Intelligence**

  - **AI System Architect**: Gemini AI performs initial setup, strategy selection, dynamic parameter tuning, and proposes holistic, cycle-over-cycle improvements.
  - **AI Doctor**: When training fails, the AI performs root-cause analysis, proposes multiple corrective scenarios, and makes a final, justified decision to rescue the cycle.
  - **Regime-Aware Adaptation**: The framework operates in defined **Operating States** (e.g., `Drawdown Control`, `Aggressive Expansion`), with the AI adjusting risk parameters based on market conditions and recent performance.
  - **Alpha Feature Invention**: The AI can invent novel, complex "alpha features" by combining base indicators to capture behavioural and statistical phenomena.

### 📊 **Advanced Telemetry & Monitoring**

  - **Telemetry-Driven Feedback Loop**: A comprehensive JSONL-based telemetry system logs every AI decision and its subsequent performance impact.
  - **Framework Memory**: This "report card" of past AI interventions is fed back into future prompts, allowing the AI to learn from its own successes and failures.
  - **Champion Tracking**: Maintains an "Overall Champion" and "Regime-Specific Champion" models, providing a robust performance baseline for the AI to compete against.
  - **Performance Attribution**: Detailed cycle-by-cycle performance breakdown and analysis.

### 🔬 **Scientific Rigor**

  - **Walk-Forward Engine**: Employs a rigorous walk-forward methodology with sliding windows to ensure robust out-of-sample validation.
  - **SHAP Explainability**: Integrates SHAP analysis for deep model transparency and feature importance analysis.
  - **Multi-Horizon Ensemble Modelling**: Trains an ensemble of models, each optimised for a different prediction horizon, and uses a dynamic weighted vote for decisions.
  - **Overfitting Control**: A **Population Stability Index (PSI) Gate** prevents training on unstable data, while **Shadow Set Validation** checks for overfitting before backtesting.

### 🧬 **Evolutionary Strategy Development**

  - **Genetic Programming**: Automatically discovers novel trading rules from scratch using an evolutionary algorithm. The AI defines the "gene pool" of indicators and operators.
  - **Dynamic Ensembles**: Model weights in the ensemble are adapted based on recent performance and feature importance.
  - **AI Playbook Curation**: The AI can amend the central strategy playbook by retiring chronically failing strategies and inventing new ones to replace them.

### ⚡ **High-Performance Architecture**

  - **Intelligent Caching**: Dramatically accelerates consecutive runs by caching the results of heavy feature engineering tasks.
  - **Scalable Processing**: Built-in multi-core parallel processing for large datasets and complex calculations.
  - **Memory Efficiency**: Optimised data handling and streaming telemetry to prevent memory issues during long backtests.

### 📈 **Comprehensive Feature Engineering**

  - **200+ Features**: An extensive library of technical, statistical, behavioural, and market microstructure features.
  - **Advanced Indicators**: Includes GARCH, Wavelets, Kalman Filters, Hawkes Volatility, Hurst Exponent, and Fourier Analysis.
  - **Graph-Based Features**: Can employ a Graph Neural Network (GNN) to model inter-asset relationships.
  - **Multi-Timeframe Fusion**: Automatically integrates and synchronises features across multiple timeframes.

-----

## 🏗️ **Core Components & Methods**

This section details the primary classes and methods that drive the framework's intelligence and explain how predictions are generated and interpreted.

### \#\#\# 🧠 `GeminiAnalyzer` (The AI Architect)

This class serves as the central intelligence hub, interfacing with Google's Gemini models to perform analysis, make strategic decisions, and generate new ideas.

  - **Method:** `get_initial_run_configuration(...)`

      - **Description:** At the start of a run, this method takes in raw data summaries, historical memory, and the strategy playbook. The AI analyses this context to propose a complete, tailored configuration, including the best strategy to use, features to select, broker costs, and walk-forward timing.

  - **Method:** `generate_training_failure_scenarios(...)` & `make_final_decision_from_scenarios(...)`

      - **Description:** This is the two-step **"AI Doctor"** loop. When model training fails, the first method analyses the failure context (e.g., poor label distribution, low feature learnability) and proposes three distinct, actionable scenarios for a fix. The second method acts as a "Head Strategist," reviewing the scenarios and making a final, decisive choice for the retry attempt.

  - **Method:** `propose_holistic_cycle_update(...)`

      - **Description:** After a backtest cycle completes, this method feeds the full performance report, SHAP summary, and market regime analysis to the AI. It then proposes a set of interdependent parameter changes for the next cycle to improve performance and adapt to changing conditions.

  - **Method:** `discover_behavioral_patterns(...)` & `propose_alpha_features(...)`

      - **Description:** These methods are used for feature invention. The AI is prompted to create novel "meta-features" by combining existing base indicators in non-linear ways, complete with a testable hypothesis and the Python lambda code to implement it.

  - **Method:** `select_best_tradeoff(...)`

      - **Description:** After hyperparameter optimisation, Optuna often presents a Pareto front of "best" models. This method presents these choices to the AI, which then selects the single best model that aligns with the current strategic directive (e.g., prioritising stability in `Drawdown Control` state).

### \#\#\# 🤖 `ModelTrainer` (The ML Engine)

This class manages the entire machine learning workflow, from feature selection and hyperparameter tuning to final model training and validation.

  - **Method:** `train_all_models(...)`

      - **Description:** Orchestrates the training of the entire multi-horizon model ensemble. It includes a **PSI pre-training gate** to abort if it detects significant feature drift between the training and shadow validation sets.

  - **Method:** `train_single_model(...)`

      - **Description:** The core training function for one model in the ensemble. It runs the selected feature selection method (`trex`, `mutual_info`, `hybrid`), applies AI-driven feature weighting, and initiates hyperparameter optimisation.

  - **Method:** `_optimize_hyperparameters(...)`

      - **Description:** Uses the **Optuna** library to perform a multi-objective hyperparameter search. The objectives are to maximise a stability score (balancing profit and consistency) and the expected payoff of trades.

  - **Method:** `_generate_shap_summary(...)`

      - **Description:** This method explains the model's predictions. After the final model is trained, it uses the **SHAP (SHapley Additive exPlanations)** library to calculate the importance of each feature. The output shows which features had the most impact on the model's decisions, providing crucial transparency.

### \#\#\# 📈 `Backtester` (The Simulation Engine)

This class runs the high-fidelity simulation, executing trades based on the trained model ensemble's predictions.

  - **Method:** `run_backtest_chunk(...)`
      - **Description:** This is the main backtesting loop. For each timestep, it feeds the latest market data to all models in the trained ensemble.
      - **Prediction & Voting:** Each model outputs a prediction (Long, Short, or Hold) and a confidence score. These predictions are aggregated in a **dynamic, weighted vote**. Weights are determined by model confidence, recent performance, and SHAP importance.
      - **Execution:** If the final vote surpasses the confidence threshold, a trade is initiated, applying realistic costs for slippage, commission, and latency.

### \#\#\# 🧬 `GeneticProgrammer` (The Strategy Discoverer)

This class uses an evolutionary algorithm to automatically discover new trading rules.

  - **Method:** `run_evolution(...)`

      - **Description:** Manages the entire evolutionary process. It creates an initial population of random trading rules (based on the AI-defined gene pool), evaluates their fitness, and then iteratively applies selection, crossover, and mutation to evolve better-performing rules over dozens of generations.

  - **Method:** `evaluate_fitness(...)`

      - **Description:** This method determines how "good" an evolved rule is. It runs a fast, simplified backtest of the rule on a slice of training data and calculates a fitness score based on a combination of Sharpe Ratio, MAR Ratio, and the total number of trades generated.

-----

## 🚀 Quick Start

### Installation

```bash
# Clone the repository
git clone https://github.com/sam-minns/AxiomEdge-PRO.git
cd AxiomEdge-PRO

# Create virtual environment (recommended)
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies from the requirements list
pip install numpy pandas matplotlib scipy scikit-learn xgboost optuna pydantic python-dotenv requests joblib statsmodels yfinance shap PyWavelets arch hurst sktime torch-geometric pyarrow psutil trexselector pykalman networkx tlars

# Install PyTorch separately for your system (e.g., CPU)
pip install torch

# Set up environment variables
# Create a .env file and add your key:
# GEMINI_API_KEY="your-gemini-api-key"
```

### Basic Usage

1.  **Place Data**: Put your `SYMBOL_TIMEFRAME.csv` files in the root directory.
2.  **Set API Key**: Ensure your `GEMINI_API_KEY` is in the `.env` file.
3.  **Run**: Execute the framework from your terminal.

<!-- end list -->

```bash
python AxiomEdge_PRO_V2_1_1.py
```

The script will initiate the full, automated workflow. The AI will analyse your data, select a strategy, and begin the walk-forward backtest.

-----

## 📋 Framework Capabilities (Tasks)

While integrated into a single orchestrator, the framework's capabilities can be understood as distinct, powerful tasks.

### 1\. 📈 Feature Engineering Task

Engineer over 200 features from raw price data for machine learning models.

**Feature Categories (200+ Total):**

  - **Technical Indicators (40+)**: RSI, MACD, Bollinger Bands, Stochastic, ATR, EMAs.
  - **Statistical Moments (25+)**: Rolling statistics, skewness, kurtosis, quantiles.
  - **Advanced Analytics (30+)**: GARCH, Entropy, Cycle Analysis (Hilbert/Fourier), Wavelets, Kalman Filters.
  - **Behavioural & Microstructure (25+)**: Order flow imbalance, liquidity proxies, session features (ORB), candle patterns.
  - **Multi-timeframe (Variable)**: Higher timeframe trend, momentum, and volatility context.

### 2\. 🤖 Model Training Task

Train machine learning models with automated hyperparameter optimisation and advanced validation.

**Core Capabilities:**

  - **XGBoost Integration**: Optimised gradient boosting models.
  - **Multi-Horizon Ensembles**: Trains separate models for short, medium, and long-term prediction.
  - **Hyperparameter Optimisation**: AI-defined Optuna search spaces for each model in the ensemble.
  - **Advanced Validation**: **PSI Gate** for data stability and **Shadow Set Validation** for overfitting.
  - **SHAP Analysis**: Deep feature importance and model explainability.

### 3\. 🧬 Genetic Programming Task

Evolve trading rules using genetic algorithms with AI-guided optimisation.

**Core Capabilities:**

  - **AI-Defined Gene Pool**: The AI selects the optimal features, operators, and constants for the evolutionary search.
  - **Fitness Evaluation**: Uses a composite score based on Sharpe Ratio, MAR Ratio, and trade frequency.
  - **AI-Assisted Retry**: If evolution fails to find a profitable rule, the AI analyses the failure and proposes a new gene pool for a retry.

### 4\. 🎯 Backtesting Task

Run a high-fidelity backtest with a dynamic ensemble and realistic execution simulation.

**Core Capabilities:**

  - **Dynamic Weighted Voting**: The ensemble of models votes on trade signals, with weights adjusted based on confidence, recent performance, and feature importance.
  - **Realistic Execution**: Simulates slippage, commissions, latency, and dynamic spreads.
  - **Adaptive Risk Management**: Utilises **Operating States** to adjust risk parameters based on performance.
  - **TP Ladder**: Supports partial profit-taking at multiple risk-reward levels.

### 5\. 📊 Report Generation Task

Generate comprehensive performance reports with visualizations and detailed analytics.

**Report Sections:**

  - **Executive Summary**: Comparison of the current run vs. the previous run and the all-time champion.
  - **Performance Metrics**: CAGR, Sharpe, Sortino, Calmar ratios.
  - **Risk Analysis**: Maximum drawdown, recovery factor, and daily drawdown event logs.
  - **Walk-Forward Cycles**: Detailed cycle-by-cycle performance breakdown.
  - **Feature Importance**: Aggregated SHAP analysis across all cycles.
  - **Model Performance**: Classification reports from the final validation set.

### 6\. 🛰️ Telemetry & Monitoring Task

An advanced telemetry system for comprehensive framework monitoring and analysis.

**Core Capabilities:**

  - **AI Intervention Logging**: Tracks every AI decision (e.g., "AI Doctor" fixes, strategy switches) with a unique ID.
  - **Performance Evaluation**: After a cycle, the impact of the AI's intervention is measured (e.g., did Sharpe Ratio improve?).
  - **Historical Feedback Loop**: The AI is provided with a summary of its own past performance when making new decisions, allowing it to learn and improve its strategic suggestions over time.

-----

## 🧪 AxiomEdge vs. Traditional Backtesting Frameworks

| Feature | AxiomEdge v2.1.1 | backtesting.py | zipline | backtrader |
|---------|-----------|----------------|---------|------------|
| **AI System Architect** | ✅ Gemini AI | ❌ | ❌ | ❌ |
| **Advanced Telemetry** | ✅ Mavis-Style | ❌ | ❌ | ❌ |
| **Walk-Forward Engine** | ✅ Integrated | ❌ | ❌ | ❌ |
| **Feature Engineering** | ✅ 200+ Features | ❌ | ❌ | ❌ |
| **Genetic Programming** | ✅ Strategy Evolution | ❌ | ❌ | ❌ |
| **SHAP Explainability**| ✅ Integrated | ❌ | ❌ | ❌ |
| **Dynamic Ensembles** | ✅ Adaptive Weighting | ❌ | ❌ | ❌ |
| **Regime Adaptation** | ✅ Operating States | ❌ | ❌ | ❌ |
| **Overfitting Control**| ✅ PSI & Shadow Sets | ❌ | ❌ | ❌ |
| **Framework Memory** | ✅ Historical Learning | ❌ | ❌ | ❌ |
| **Basic Backtesting** | ✅ | ✅ | ✅ | ✅ |
| **Portfolio Analysis**| ✅ | ❌ | ✅ | ✅ |

-----

## 🤝 Contributing & Support

We welcome contributions\! Please visit the GitHub repository for guidelines on reporting issues, requesting features, and submitting pull requests.

  - **[Issues & Bugs](https://github.com/sam-minns/AxiomEdge-PRO/issues)**
  - **[Feature Requests](https://github.com/sam-minns/AxiomEdge-PRO/issues/new?template=feature_request.md)**
  - **[Community Discussions](https://github.com/sam-minns/AxiomEdge-PRO/discussions)**
