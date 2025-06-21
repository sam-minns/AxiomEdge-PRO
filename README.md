# End-to-End AI-Powered Algorithmic Trading Framework

This project is a sophisticated, fully autonomous Python framework for developing, backtesting, and deploying machine learning-based trading strategies. Its core feature is the integration of a Large LanguageModel (LLM) as a dynamic "AI Strategist" that actively manages the entire trading process, from initial setup to cycle-over-cycle adaptation and failure recovery.

The framework is built around a robust walk-forward engine, ensuring that strategies are tested out-of-sample in a manner that closely simulates live trading conditions. It is designed for resilience, adaptability, and continuous learning.

## 🤖 Key Features

* **LLM-Powered Strategy**: Utilizes Google's Gemini models to make high-level strategic decisions, replacing hard-coded logic with dynamic, context-aware reasoning.
* **♟️ "Chess Player" Forecasting**: The AI performs a "pre-mortem" analysis at the start of each run, anticipating likely failure modes and pre-defining contingency plans for more intelligent failure recovery.
* **📈 Walk-Forward Engine**: Employs a rigorous walk-forward methodology, training the model on a sliding window of historical data and testing it on the subsequent unseen period, providing a more realistic performance assessment.
* **🧠 Persistent Memory**: The framework learns from its own history. It maintains a "long-term memory" of all past runs and a "champion" model, allowing the AI to compare new results against historical successes and failures.
* **📚 Dynamic Strategy Playbook**: Strategies are not hard-coded. They are defined in an external `strategy_playbook.json` file, allowing users to add, remove, or "retire" strategies and their feature sets without touching the core codebase.
* **⚙️ Advanced Feature Engineering**: Automatically creates a rich feature stack from multi-timeframe data, including technical indicators (RSI, ADX, Bollinger Bands), candle microstructure, indicator dynamics (slope), and fractional differentiation.
* **🛡️ Robust Risk Management**: Implements multiple layers of safety, including a per-cycle drawdown "circuit breaker," tiered risk configuration based on equity, anomaly detection using Isolation Forests, and confidence-based trade sizing.

## How It Works: The Core Loop

The framework operates in a continuous loop, simulating a real-world adaptive trading system.

1.  **Initial Setup**: The AI Strategist analyzes the overall market data, long-term framework memory, and its own past notes to select an optimal strategy, a targeted feature set, and key parameters for the entire run.
2.  **Strategic Forecast**: Immediately after setup, the AI creates a "game plan." It identifies the most likely reasons its chosen strategy might fail and defines Contingency Paths B (an alternative strategy) and C (a parameter tweak) in advance.
3.  **Walk-Forward Cycle Begins**: The framework divides the data into sequential training and testing periods.
    * **Training**: An XGBoost model is trained on the current historical data window. Hyperparameters are optimized using Optuna.
    * **Label Quality Check**: Ensures that the data from the training period is suitable for creating a predictive model, preventing wasted cycles on "un-trainable" market conditions.
    * **Testing**: The newly trained model is used to "trade" the next out-of-sample data period.
4.  **Performance Analysis & Adaptation**: At the end of the testing cycle, the AI analyzes the performance.
    * **If Successful**: It suggests incremental improvements for the next cycle.
    * **If Failed (Circuit Breaker / Poor Training)**: It is reminded of its pre-run forecast and must choose whether to execute one of its planned contingencies or devise a new path based on the new data.
5.  **Memory Update**: The results of the run are saved to the `historical_runs.jsonl` log. If the run's performance (measured by the MAR ratio) is better than any previous run, it's saved as the new `champion.json`.
6.  **Repeat**: The process repeats until all data has been backtested.

## Getting Started

### 1. Prerequisites
* Python 3.8+
* A Google Gemini API Key.
* Required libraries. You will need to install the following:

    ```
   pandas==2.1.4
   matplotlib==3.8.4
   scipy==1.12.0
   scikit-learn==1.4.2
   xgboost==3.0.2
   optuna==4.3.0
   pydantic==2.11.5
   python-dotenv==1.1.0
   requests==2.32.4
   joblib==1.4.2
   statsmodels==0.14.4
   yfinance==0.2.63
   shap==0.47.2
   PyWavelets==1.8.0
   arch==7.2.0
   hurst==0.0.5
   sktime==0.27.0
   pyarrow==20.0.0
   torch_geometric==2.5.3
   tensorflow-cpu==2.19.0
   torch==2.3.0+cpu
    ```
    You can typically install these via pip: `pip install numpy pandas xgboost ...`

### 2. Environment Setup
Create a `.env` file in the root directory of the project and add your Gemini API key:
```
GEMINI_API_KEY="YOUR_API_KEY_HERE"
```

### 3. Data Preparation
Place your historical data files (`.csv` or `.txt`) in the root directory. The framework relies on a specific naming convention to identify the asset and timeframe:

* **Format**: `SYMBOL_TIMEFRAME.csv` (e.g., `EURUSD_H1.csv`, `BTCUSD_M15.txt`)
* **Columns**: The file should contain standard OHLC data. The script will automatically detect columns named `Open`, `High`, `Low`, `Close`, `Date`, `Time`, and `Volume`/`Tickvol`.

### 4. Running the Framework
Execute the script from your terminal:
```bash
python End_To_End_Advanced_ML_Trading_Framework_PRO_V192_Linux.py
```
The script can be configured to run once or in a continuous "daemon" mode for ongoing research by adjusting the `CONTINUOUS_RUN_HOURS` and `MAX_RUNS` variables in the `main()` function.

## Framework Output

The script generates several important files inside a `Results/` directory:

* **Run-Specific Folder**: A unique folder is created for each run (e.g., `Andromeda_V192/`) containing:
    * `_report.txt`: A detailed text-based performance report.
    * `_equity_curve.png`: A plot of the walk-forward equity curve.
    * `_shap_summary.png`: A feature importance plot from the aggregated SHAP values.
    * `_model.json`: The final trained XGBoost model from the last cycle.
    * `_run.log`: A detailed log of the entire run.
* **`champion.json`**: A JSON file containing the configuration and final metrics of the best-performing run to date (highest MAR ratio).
* **`historical_runs.jsonl`**: A log file where each line is a JSON object summarizing a completed run. This serves as the framework's long-term memory.
* **`strategy_playbook.json`**: A dynamic JSON file defining the available strategies, their descriptions, and default feature sets.
* **`nickname_ledger.json`**: Stores the AI-generated codenames for each script version.


### Data File Requirements

To load custom data into the framework, users must adhere to specific requirements regarding file location, naming, and content format.

#### **File Location**
* The data files, either `.csv` or `.txt`, must be placed in the root directory of the project.

#### **File Naming Convention**
The framework relies on a strict naming convention to automatically identify the financial instrument (symbol) and the chart timeframe.
* **Format**: The filename must be in the format `SYMBOL_TIMEFRAME.csv` or `SYMBOL_TIMEFRAME.txt`.
* **Examples**:
    * `EURUSD_H1.csv` for Euro vs US Dollar on the 1-hour chart.
    * `BTCUSD_M15.txt` for Bitcoin vs US Dollar on the 15-minute chart.

#### **File Content and Columns**
The file should contain standard OHLC (Open, High, Low, Close) data. The script is designed to be flexible and automatically detects the necessary columns.

* **Delimiter**: The script can automatically handle comma-separated (`.csv`) and tab-separated (`.txt`) files.
* **Column Names**: Column headers are case-insensitive. For example, `open`, `Open`, and `OPEN` are all treated the same.
* **Required Columns**:
    * **Price Data**: Columns for `Open`, `High`, `Low`, and `Close` must be present.
    * **Timestamp**: The script needs date and time information, which can be provided in one of two ways:
        1.  A single column containing both date and time.
        2.  Two separate columns, one for `Date` and one for `Time`.
* **Optional Columns**:
    * **Volume**: A column for trade volume can be included under the name `Volume` or `Tickvol`. If no volume column is found, it is treated as zero.

### Summary Table

| Requirement | Details | Example(s) |
| --- | --- | --- |
| **Location** | Root directory of the framework. | `/home/user/ml_framework/` |
| **Naming** | `SYMBOL_TIMEFRAME.extension` | `EURUSD_H1.csv`, `AAPL_D1.txt` |
| **Extension** | `.csv` or `.txt` | `data.csv`, `data.txt` |
| **Columns** | Must contain OHLC and Timestamp data. Volume is optional. | `Date, Time, Open, High, Low, Close, Volume` |

## ⚠️ Disclaimer

This is an experimental framework for research purposes only. It is not financial advice. Trading financial markets involves substantial risk, and you can lose all of your invested capital. Past performance is not indicative of future results. Do not run this framework with real money without fully understanding the code and the risks involved.
