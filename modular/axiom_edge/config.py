# =============================================================================
# CONFIGURATION & VALIDATION MODULE
# =============================================================================

import os
import re
from enum import Enum
from typing import Dict, List, Optional, Any, Union, ClassVar
from pathlib import Path
from pydantic import BaseModel, Field, validator, confloat, conint, DirectoryPath
import logging

logger = logging.getLogger(__name__)

class OperatingState(Enum):
    """
    Operational states defining the framework's trading behavior and risk profile.

    Each state represents a different approach to risk management and strategy
    execution based on recent performance, market conditions, and external factors.
    """
    # Core operational states
    CONSERVATIVE_BASELINE = "Conservative Baseline"    # Find stable trading model with reasonable quality
    AGGRESSIVE_EXPANSION = "Aggressive Expansion"      # Maximize risk-adjusted returns from proven baseline
    DRAWDOWN_CONTROL = "Drawdown Control"              # Preserve capital after consecutive failures
    PERFORMANCE_REVIEW = "Performance Review"          # Analyze and adjust after single failure

    # Market condition states
    OPPORTUNISTIC_SURGE = "Opportunistic Surge"        # Increase risk to capture volatility spikes
    VOLATILITY_SPIKE = "Volatility Spike"              # React to sharp increases in volatility
    LIQUIDITY_CRUNCH = "Liquidity Crunch"              # Reduce size due to poor market conditions
    MARKET_NEUTRAL = "Market Neutral"                  # Seek opportunities with low directional bias

    # Strategy management states
    STRATEGY_ROTATION = "Strategy Rotation"            # Explore new strategies if current stagnates
    ALGO_CALIBRATION = "Algo Calibration"              # Pause trading for offline model recalibration

    # Risk management states
    PROFIT_PROTECTION = "Profit Protection"            # Reduce risk after large windfall to lock gains
    CAPITAL_REALLOCATION = "Capital Reallocation"      # Pause trading during capital adjustments

    # External factor states
    NEWS_SENTIMENT_ALERT = "News Sentiment Alert"      # Limit exposure during high-impact news events
    MAINTENANCE_DORMANCY = "Maintenance Dormancy"      # Pause trading for system maintenance
    RESEARCH_MODE = "Research Mode"                     # Pause live execution for R&D
    COMPLIANCE_LOCKDOWN = "Compliance Lockdown"        # Halt activity due to compliance issues

class EarlyInterventionConfig(BaseModel):
    """
    Configuration for the adaptive early intervention system.

    Controls when and how the framework intervenes during poor performance
    to prevent significant losses and adapt strategy parameters.

    Attributes:
        enabled: Whether early intervention is active
        attempt_threshold: Minimum failed attempts before intervention
        min_profitability_for_f1_override: Minimum profitability to override F1 gate
        max_f1_override_value: Maximum F1 override value allowed
    """
    enabled: bool = True
    attempt_threshold: conint(ge=2) = 2
    min_profitability_for_f1_override: confloat(ge=0) = 3.0
    max_f1_override_value: confloat(ge=0.4, le=0.6) = 0.50

class ConfigModel(BaseModel):
    """
    Comprehensive configuration model for the AxiomEdge trading framework.

    Defines all parameters for trading strategy execution, risk management,
    feature engineering, model training, and system behavior. Uses Pydantic
    for validation and type checking.

    The configuration is organized into logical sections:
    - Core parameters (paths, capital, state)
    - AI and optimization settings
    - Risk management and position sizing
    - Feature engineering parameters
    - Model training and validation
    - Execution and broker simulation
    """

    # Core run, capital and state parameters
    BASE_PATH: DirectoryPath
    REPORT_LABEL: str
    INITIAL_CAPITAL: confloat(gt=0)
    operating_state: OperatingState = OperatingState.CONSERVATIVE_BASELINE
    TARGET_DD_PCT: confloat(gt=0) = 0.25
    ASSET_CLASS_BASE_DD: confloat(gt=0) = 0.25

    FEATURE_SELECTION_METHOD: str
    SHADOW_SET_VALIDATION: bool = True

    # AI and optimization parameters
    OPTUNA_TRIALS: conint(gt=0) = 75
    OPTUNA_N_JOBS: conint(ge=-1) = 1
    MAX_TRAINING_RETRIES_PER_CYCLE: conint(ge=0) = 3
    CALCULATE_SHAP_VALUES: bool = True

    # Dynamic weighted voting ensemble parameters
    MIN_SHAP_TO_VOTE: confloat(ge=0.0) = 0.001  # Minimum SHAP importance for model inclusion
    REGIME_ENSEMBLE_WEIGHTS: Dict[int, Dict[int, float]] = Field(default_factory=lambda: {
        # Default ensemble weights by regime and horizon
        0: {30: 0.6, 60: 0.3, 120: 0.1},    # Ranging market regime
        1: {30: 0.2, 60: 0.5, 120: 0.3},    # Trending market regime
        -1: {30: 0.33, 60: 0.34, 120: 0.33} # Default fallback regime
    })

    # Confidence gate control
    USE_STATIC_CONFIDENCE_GATE: bool = False
    STATIC_CONFIDENCE_GATE: confloat(ge=0.5, le=0.95) = 0.70
    REGIME_CONFIDENCE_GATES: Dict[str, confloat(ge=0.5, le=0.95)] = Field(default_factory=lambda: {
        "Trending": 0.65,
        "Ranging": 0.75,
        "Highvolatility": 0.70,
        "Default": 0.70
    })
    USE_MULTI_MODEL_CONFIRMATION: bool = False

    # Dynamic labeling and trade definition
    TP_ATR_MULTIPLIER: confloat(gt=0.5, le=10.0) = 2.0
    SL_ATR_MULTIPLIER: confloat(ge=0.5, le=10.0) = 1.5
    LOOKAHEAD_CANDLES: conint(gt=0) = 150
    LABELING_METHOD: str = 'signal_pressure'
    MIN_F1_SCORE_GATE: confloat(ge=0.3, le=0.7) = 0.45
    LABEL_LONG_QUANTILE: confloat(ge=0.5, le=1.0) = 0.95
    LABEL_SHORT_QUANTILE: confloat(ge=0.0, le=0.5) = 0.05
    LABEL_HORIZONS: List[conint(gt=0)] = Field(default_factory=lambda: [30, 60, 90])

    # Walk-forward and data parameters
    TRAINING_WINDOW: str = '365D'
    RETRAINING_FREQUENCY: str = '90D'
    FORWARD_TEST_GAP: str = '1D'

    # Risk and portfolio management
    MAX_DD_PER_CYCLE: confloat(ge=0.05, lt=1.0) = 0.25
    RISK_CAP_PER_TRADE_USD: confloat(gt=0) = 1000.0
    BASE_RISK_PER_TRADE_PCT: confloat(gt=0, lt=1) = 0.0025
    MAX_CONCURRENT_TRADES: conint(ge=1, le=20) = 1
    USE_TIERED_RISK: bool = False
    RISK_PROFILE: str = 'Medium'
    LEVERAGE: conint(gt=0) = 30
    MIN_LOT_SIZE: confloat(gt=0) = 0.01
    LOT_STEP: confloat(gt=0) = 0.01
    # Tiered risk configuration by account size and risk profile
    TIERED_RISK_CONFIG: Dict[int, Dict[str, Dict[str, Union[float, int]]]] = Field(default_factory=lambda: {
        2000:  {'Low': {'risk_pct': 0.01,  'pairs': 1}, 'Medium': {'risk_pct': 0.01,  'pairs': 1}, 'High': {'risk_pct': 0.01,  'pairs': 1}},
        5000:  {'Low': {'risk_pct': 0.008, 'pairs': 1}, 'Medium': {'risk_pct': 0.012, 'pairs': 1}, 'High': {'risk_pct': 0.012, 'pairs': 2}},
        10000: {'Low': {'risk_pct': 0.006, 'pairs': 2}, 'Medium': {'risk_pct': 0.008, 'pairs': 2}, 'High': {'risk_pct': 0.01,  'pairs': 2}},
        15000: {'Low': {'risk_pct': 0.007, 'pairs': 2}, 'Medium': {'risk_pct': 0.009, 'pairs': 2}, 'High': {'risk_pct': 0.012, 'pairs': 2}},
        25000: {'Low': {'risk_pct': 0.008, 'pairs': 2}, 'Medium': {'risk_pct': 0.012, 'pairs': 2}, 'High': {'risk_pct': 0.016, 'pairs': 2}},
        50000: {'Low': {'risk_pct': 0.008, 'pairs': 3}, 'Medium': {'risk_pct': 0.012, 'pairs': 3}, 'High': {'risk_pct': 0.016, 'pairs': 3}},
        100000:{'Low': {'risk_pct': 0.007, 'pairs': 4}, 'Medium': {'risk_pct': 0.01,  'pairs': 4}, 'High': {'risk_pct': 0.014, 'pairs': 4}},
        9000000000: {'Low': {'risk_pct': 0.005, 'pairs': 6}, 'Medium': {'risk_pct': 0.0075,'pairs': 6}, 'High': {'risk_pct': 0.01,  'pairs': 6}}
    })

    # State-based configuration for adaptive behavior
    STATE_BASED_CONFIG: Dict[OperatingState, Dict[str, Any]] = Field(default_factory=lambda: {
        OperatingState.CONSERVATIVE_BASELINE: {
            "max_dd_per_cycle_mult": 1.0, "base_risk_pct": 0.005, "max_concurrent_trades": 1,
            "optimization_weights": {"calmar": 0.8, "num_trades": 0.2}, "min_f1_gate": 0.45
        },
        OperatingState.PERFORMANCE_REVIEW: {
            "max_dd_per_cycle_mult": 1.0, "base_risk_pct": 0.005, "max_concurrent_trades": 1,
            "optimization_weights": {"calmar": 0.9, "num_trades": 0.1}, "min_f1_gate": 0.45
        },
        OperatingState.DRAWDOWN_CONTROL: {
            "max_dd_per_cycle_mult": 0.6, "base_risk_pct": 0.0025, "max_concurrent_trades": 1,
            "optimization_weights": {"calmar": 1.0, "max_dd": -0.5}, "min_f1_gate": 0.40
        },
        OperatingState.AGGRESSIVE_EXPANSION: {
            "max_dd_per_cycle_mult": 1.2, "base_risk_pct": 0.01, "max_concurrent_trades": 3,
            "optimization_weights": {"sharpe": 0.6, "total_pnl": 0.4}, "min_f1_gate": 0.42
        },
        OperatingState.OPPORTUNISTIC_SURGE: {
            "max_dd_per_cycle_mult": 1.25, "base_risk_pct": 0.02, "max_concurrent_trades": 4,
            "optimization_weights": {"sharpe": 0.7, "total_pnl": 0.3}, "min_f1_gate": 0.55
        },
        OperatingState.PROFIT_PROTECTION: {
            "max_dd_per_cycle_mult": 0.4, "base_risk_pct": 0.003, "max_concurrent_trades": 1,
            "optimization_weights": {"calmar": 0.9, "max_dd": -0.1}, "min_f1_gate": 0.52
        },
        OperatingState.STRATEGY_ROTATION: {
            "max_dd_per_cycle_mult": 0.4, "base_risk_pct": 0.0025, "max_concurrent_trades": 1,
            "optimization_weights": {"f1": 1.0}, "min_f1_gate": 0.95
        },
        OperatingState.VOLATILITY_SPIKE: {
            "max_dd_per_cycle_mult": 0.8, "base_risk_pct": 0.0125, "max_concurrent_trades": 3,
            "optimization_weights": {"sharpe": 0.8, "num_trades": 0.2}, "min_f1_gate": 0.40
        },
        OperatingState.LIQUIDITY_CRUNCH: {
            "max_dd_per_cycle_mult": 0.3, "base_risk_pct": 0.0025, "max_concurrent_trades": 1,
            "optimization_weights": {"f1": 1.0}, "min_f1_gate": 0.50
        },
        OperatingState.NEWS_SENTIMENT_ALERT: {
            "max_dd_per_cycle_mult": 0.4, "base_risk_pct": 0.004, "max_concurrent_trades": 1,
            "optimization_weights": {"f1": 1.0}, "min_f1_gate": 0.45
        },
        OperatingState.MAINTENANCE_DORMANCY: {
            "max_dd_per_cycle_mult": 0.2, "base_risk_pct": 0.0, "max_concurrent_trades": 0,
            "optimization_weights": {"f1": 1.0}, "min_f1_gate": 0.99
        },
        OperatingState.ALGO_CALIBRATION: {
            "max_dd_per_cycle_mult": 0.2, "base_risk_pct": 0.0, "max_concurrent_trades": 0,
            "optimization_weights": {"f1": 1.0}, "min_f1_gate": 0.99
        },
        OperatingState.CAPITAL_REALLOCATION: {
            "max_dd_per_cycle_mult": 0.2, "base_risk_pct": 0.0, "max_concurrent_trades": 0,
            "optimization_weights": {"f1": 1.0}, "min_f1_gate": 0.99
        },
        OperatingState.RESEARCH_MODE: {
            "max_dd_per_cycle_mult": 0.2, "base_risk_pct": 0.0, "max_concurrent_trades": 0,
            "optimization_weights": {"f1": 1.0}, "min_f1_gate": 0.99
        },
        OperatingState.COMPLIANCE_LOCKDOWN: {
            "max_dd_per_cycle_mult": 0.2, "base_risk_pct": 0.0, "max_concurrent_trades": 0,
            "optimization_weights": {"f1": 1.0}, "min_f1_gate": 0.99
        },
        OperatingState.MARKET_NEUTRAL: {
            "max_dd_per_cycle_mult": 0.4, "base_risk_pct": 0.006, "max_concurrent_trades": 2,
            "optimization_weights": {"sharpe": 1.0}, "min_f1_gate": 0.50
        }
    })

    # Confidence-based risk tiers
    CONFIDENCE_TIERS: Dict[str, Dict[str, Any]] = Field(default_factory=lambda: {
        'ultra_high': {'min': 0.80, 'risk_mult': 1.2, 'rr': 2.5},
        'high':       {'min': 0.70, 'risk_mult': 1.0, 'rr': 2.0},
        'standard':   {'min': 0.60, 'risk_mult': 0.8, 'rr': 1.5}
    })

    # Take-profit ladder configuration
    USE_TP_LADDER: bool = True
    TP_LADDER_LEVELS_PCT: List[confloat(gt=0, lt=1)] = Field(default_factory=lambda: [0.25, 0.25, 0.25, 0.25])
    TP_LADDER_RISK_MULTIPLIERS: List[confloat(gt=0)] = Field(default_factory=lambda: [1.0, 2.0, 3.0, 4.0])

    # Broker and execution simulation
    COMMISSION_PER_LOT: confloat(ge=0.0) = 3.5
    USE_REALISTIC_EXECUTION: bool = True
    SIMULATE_LATENCY: bool = True
    EXECUTION_LATENCY_MS: conint(ge=50, le=500) = 150
    USE_VARIABLE_SLIPPAGE: bool = True
    SLIPPAGE_VOLATILITY_FACTOR: confloat(ge=0.0, le=5.0) = 1.5
    SPREAD_CONFIG: Dict[str, Dict[str, float]] = Field(default_factory=lambda: {
        'default': {'normal_pips': 1.8, 'volatile_pips': 5.5},
        'EURUSD':  {'normal_pips': 1.2, 'volatile_pips': 4.0},
        'GBPUSD':  {'normal_pips': 1.6, 'volatile_pips': 5.0},
        'AUDUSD':  {'normal_pips': 1.4, 'volatile_pips': 4.8},
        'USDCAD':  {'normal_pips': 1.7, 'volatile_pips': 5.5},
        'USDJPY':  {'normal_pips': 1.3, 'volatile_pips': 4.5},
        'AUDCAD':  {'normal_pips': 1.9, 'volatile_pips': 6.0},
        'AUDNZD':  {'normal_pips': 2.2, 'volatile_pips': 7.0},
        'NZDJPY':  {'normal_pips': 2.0, 'volatile_pips': 6.5},
        'XAUUSD_M15':    {'normal_pips': 25.0, 'volatile_pips': 80.0},
        'XAUUSD_H1':     {'normal_pips': 20.0, 'volatile_pips': 70.0},
        'XAUUSD_Daily':  {'normal_pips': 18.0, 'volatile_pips': 60.0},
        'US30_M15':      {'normal_pips': 50.0, 'volatile_pips': 150.0},
        'US30_H1':       {'normal_pips': 45.0, 'volatile_pips': 140.0},
        'US30_Daily':    {'normal_pips': 40.0, 'volatile_pips': 130.0},
        'NDX100_M15':    {'normal_pips': 20.0, 'volatile_pips': 60.0},
        'NDX100_H1':       {'normal_pips': 18.0, 'volatile_pips': 55.0},
        'NDX100_Daily':  {'normal_pips': 16.0, 'volatile_pips': 50.0},
    })

    ASSET_CONTRACT_SIZES: Dict[str, float] = Field(default_factory=lambda: {
        'default': 100000.0,
        'XAUUSD': 100.0,
        'XAGUSD': 5000.0,
        'US30': 1.0,
        'NDX100': 1.0,
        'SPX500': 1.0,
        'GER40': 1.0,
        'UK100': 1.0,
        'BTCUSD': 1.0,
        'ETHUSD': 1.0,
        'WTI': 1000.0,
        'NGAS': 10000.0,
        'ZN': 100000.0,
        'options_default': 100.0,
        'SPX': 100.0,
        'NDX': 100.0,
        'RUT': 100.0,
        'mini_options': 10.0,
    })

    DYNAMIC_INDICATOR_PARAMS: Dict[str, Dict[str, Any]] = Field(default_factory=lambda: {
        "HighVolatility_Trending":  { "bollinger_period": 15, "bollinger_std_dev": 2.5, "rsi_period": 10 },
        "HighVolatility_Ranging":   { "bollinger_period": 20, "bollinger_std_dev": 2.8, "rsi_period": 12 },
        "HighVolatility_Default":   { "bollinger_period": 18, "bollinger_std_dev": 2.5, "rsi_period": 11 },
        "LowVolatility_Trending":   { "bollinger_period": 30, "bollinger_std_dev": 1.8, "rsi_period": 20 },
        "LowVolatility_Ranging":    { "bollinger_period": 35, "bollinger_std_dev": 1.5, "rsi_period": 25 },
        "LowVolatility_Default":    { "bollinger_period": 30, "bollinger_std_dev": 1.8, "rsi_period": 22 },
        "Default_Trending":         { "bollinger_period": 20, "bollinger_std_dev": 2.0, "rsi_period": 14 },
        "Default_Ranging":          { "bollinger_period": 25, "bollinger_std_dev": 2.2, "rsi_period": 18 },
        "Default":                  { "bollinger_period": 20, "bollinger_std_dev": 2.0, "rsi_period": 14 }
    })

    # Feature Engineering Parameters
    BENCHMARK_TICKER: str = 'SPY'
    EMA_PERIODS: List[int] = Field(default_factory=lambda: [20, 50, 100, 200])
    RSI_STANDARD_PERIODS: List[int] = Field(default_factory=lambda: [14, 28, 50])
    MOMENTUM_PERIODS: List[int] = Field(default_factory=lambda: [20])
    TREND_FILTER_THRESHOLD: confloat(gt=0) = 25.0
    BOLLINGER_PERIOD: conint(gt=0) = 20
    STOCHASTIC_PERIOD: conint(gt=0) = 14
    MIN_VOLATILITY_RANK: confloat(ge=0.0, le=1.0) = 0.1
    MAX_VOLATILITY_RANK: confloat(ge=0.0, le=1.0) = 0.9
    HAWKES_KAPPA: confloat(gt=0) = 0.5
    anomaly_contamination_factor: confloat(ge=0.001, le=0.1) = 0.01
    USE_PCA_REDUCTION: bool = True
    PCA_N_COMPONENTS: conint(gt=1, le=20) = 30
    RSI_PERIODS_FOR_PCA: List[conint(gt=1)] = Field(default_factory=lambda: [5, 10, 15, 20, 25])
    ADX_THRESHOLD_TREND: int = 20
    RSI_OVERSOLD: int = 30
    RSI_OVERBOUGHT: int = 70
    VOLUME_BREAKOUT_RATIO: float = 1.5
    BOLLINGER_SQUEEZE_LOOKBACK: int = 50
    DISPLACEMENT_STRENGTH: int = 3
    DISPLACEMENT_PERIOD: conint(gt=1) = 50
    GAP_DETECTION_LOOKBACK: conint(gt=1) = 2
    PARKINSON_VOLATILITY_WINDOW: conint(gt=1) = 30
    YANG_ZHANG_VOLATILITY_WINDOW: conint(gt=1) = 30
    KAMA_REGIME_FAST: conint(gt=1) = 10
    KAMA_REGIME_SLOW: conint(gt=1) = 66

    # Increased default lag for Autocorrelation
    AUTOCORR_LAG: conint(gt=0) = 20

    HURST_EXPONENT_WINDOW: conint(ge=100) = 100

    # Broadened RSI_MSE parameters
    RSI_MSE_PERIOD: conint(gt=1) = 28
    RSI_MSE_SMA_PERIOD: conint(gt=1) = 10
    RSI_MSE_WINDOW: conint(gt=1) = 28

    # GNN Specific Parameters
    GNN_EMBEDDING_DIM: conint(gt=0) = 8
    GNN_EPOCHS: conint(gt=0) = 50

    # Caching & Performance
    USE_FEATURE_CACHING: bool = True

    # State & Info Parameters (populated at runtime)
    selected_features: List[str] = Field(default_factory=list)
    run_timestamp: str
    strategy_name: str
    nickname: str = ""
    analysis_notes: str = ""

    # File Path Management (Internal, populated by __init__)
    result_folder_path: str = Field(default="", repr=False)
    MODEL_SAVE_PATH: str = Field(default="", repr=False)
    PLOT_SAVE_PATH: str = Field(default="", repr=False)
    REPORT_SAVE_PATH: str = Field(default="", repr=False)
    SHAP_PLOT_PATH: str = Field(default="", repr=False)
    LOG_FILE_PATH: str = Field(default="", repr=False)
    CHAMPION_FILE_PATH: str = Field(default="", repr=False)
    HISTORY_FILE_PATH: str = Field(default="", repr=False)
    PLAYBOOK_FILE_PATH: str = Field(default="", repr=False)
    DIRECTIVES_FILE_PATH: str = Field(default="", repr=False)
    NICKNAME_LEDGER_PATH: str = Field(default="", repr=False)
    REGIME_CHAMPIONS_FILE_PATH: str = Field(default="", repr=False)
    CACHE_PATH: str = Field(default="", repr=False)
    CACHE_METADATA_PATH: str = Field(default="", repr=False)
    DISCOVERED_PATTERNS_PATH: str = Field(default="", repr=False)
    AI_SCRATCHPAD_PATH: str = Field(default="", repr=False)
    DISQUALIFIED_TRIALS_PATH: str = Field(default="", repr=False)

    # AI Guardrail Parameters (to prevent meta-overfitting)
    PARAMS_LOG_FILE: ClassVar[str] = 'strategy_params_log.json'
    MAX_PARAM_DRIFT_TOLERANCE: ClassVar[float] = 40.0
    MIN_CYCLES_FOR_ADAPTATION: ClassVar[int] = 5
    MIN_HOLDOUT_SHARPE: ClassVar[float] = 0.35
    HOLDOUT_SET_PERCENTAGE: confloat(ge=0.0, le=0.3) = 0.15
    MIN_STABILITY_THRESHOLD: confloat(ge=0.0) = 0.05

    def __init__(self, **data: Any):
        super().__init__(**data)
        results_dir = os.path.join(self.BASE_PATH, "Results")
        os.makedirs(results_dir, exist_ok=True)

        # A dummy version in case regex fails
        VERSION = "1.0"

        version_match = re.search(r'V(\d+\.?\d*)', self.REPORT_LABEL)
        version_str = f"_V{version_match.group(1)}" if version_match else f"_V{VERSION}"

        safe_nickname = "".join(c if c.isalnum() or c in (' ', '_', '-') else '_' for c in self.nickname)
        safe_strategy_name = "".join(c if c.isalnum() or c in (' ', '_', '-') else '_' for c in self.strategy_name)

        folder_name_base = safe_nickname if safe_nickname and safe_nickname != "init_setup" else self.REPORT_LABEL.replace(version_str, "")
        folder_name = f"{folder_name_base}{version_str}"

        run_id_prefix = f"{folder_name}_{safe_strategy_name}" if safe_strategy_name and safe_strategy_name != "InitialSetupPhase" else folder_name
        run_id = f"{run_id_prefix}_{self.run_timestamp}"

        self.result_folder_path = os.path.join(results_dir, folder_name)

        if self.nickname and self.nickname != "init_setup" and self.strategy_name and self.strategy_name != "InitialSetupPhase":
            os.makedirs(self.result_folder_path, exist_ok=True)
            os.makedirs(os.path.join(self.result_folder_path, "models"), exist_ok=True)
            os.makedirs(os.path.join(self.result_folder_path, "shap_plots"), exist_ok=True)

            self.MODEL_SAVE_PATH = os.path.join(self.result_folder_path, "models")
            self.PLOT_SAVE_PATH = os.path.join(self.result_folder_path, f"{run_id}_equity_curve.png")
            self.REPORT_SAVE_PATH = os.path.join(self.result_folder_path, f"{run_id}_report.txt")
            self.SHAP_PLOT_PATH = os.path.join(self.result_folder_path, "shap_plots") # This is a directory
            self.LOG_FILE_PATH = os.path.join(self.result_folder_path, f"{run_id}_run.log")
        else:
            self.MODEL_SAVE_PATH = os.path.join(results_dir, "init_models")
            self.PLOT_SAVE_PATH = os.path.join(results_dir, "init_equity.png")
            self.REPORT_SAVE_PATH = os.path.join(results_dir, "init_report.txt")
            self.SHAP_PLOT_PATH = os.path.join(results_dir, "init_shap_plots")
            self.LOG_FILE_PATH = os.path.join(results_dir, f"init_run_{self.run_timestamp}.log") # Unique init log

        # Global/shared file paths
        self.CHAMPION_FILE_PATH = os.path.join(results_dir, "champion.json")
        self.HISTORY_FILE_PATH = os.path.join(results_dir, "historical_runs.jsonl")
        self.PLAYBOOK_FILE_PATH = os.path.join(results_dir, "strategy_playbook.json")
        self.DIRECTIVES_FILE_PATH = os.path.join(results_dir, "framework_directives.json")
        self.NICKNAME_LEDGER_PATH = os.path.join(results_dir, "nickname_ledger.json")
        self.REGIME_CHAMPIONS_FILE_PATH = os.path.join(results_dir, "regime_champions.json")
        self.DISCOVERED_PATTERNS_PATH = os.path.join(results_dir, "discovered_patterns.json")
        self.AI_SCRATCHPAD_PATH = os.path.join(results_dir, "ai_scratchpad.json")
        self.DISQUALIFIED_TRIALS_PATH = os.path.join(results_dir, "disqualified_trials.jsonl")


        cache_dir = os.path.join(self.BASE_PATH, "Cache")
        os.makedirs(cache_dir, exist_ok=True)
        self.CACHE_PATH = os.path.join(cache_dir, "feature_cache.parquet")
        self.CACHE_METADATA_PATH = os.path.join(cache_dir, "feature_cache_metadata.json")

    @validator('CONFIDENCE_TIERS')
    def validate_confidence_tiers(cls, v):
        if not isinstance(v, dict):
            raise ValueError("CONFIDENCE_TIERS must be a dictionary")
        for tier, config in v.items():
            if not isinstance(config, dict):
                raise ValueError(f"Confidence tier '{tier}' must have a dictionary configuration")
            required_keys = {'min', 'risk_mult', 'rr'}
            if not all(key in config for key in required_keys):
                raise ValueError(f"Confidence tier '{tier}' must have keys: {required_keys}")
        return v

    @validator('TP_LADDER_LEVELS_PCT')
    def validate_tp_ladder_levels(cls, v, values):
        if values.get('USE_TP_LADDER', False) and v is not None:
            if not isinstance(v, list) or not all(isinstance(x, (int, float)) for x in v):
                raise ValueError("TP_LADDER_LEVELS_PCT must be a list of numbers")
            if abs(sum(v) - 1.0) > 1e-6:
                raise ValueError("TP_LADDER_LEVELS_PCT must sum to 1.0")
        return v

    @validator('BASE_PATH')
    def validate_base_path(cls, v):
        path = Path(v)
        if not path.exists():
            logger.warning(f"BASE_PATH does not exist: {v}")
        return v

    class Config:
        use_enum_values = True
        validate_assignment = True
        extra = "forbid"

def load_config_from_file(config_path: str) -> ConfigModel:
    """Load configuration from a JSON or YAML file"""
    import json
    
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Configuration file not found: {config_path}")
    
    with open(config_path, 'r') as f:
        if config_path.endswith('.json'):
            config_data = json.load(f)
        elif config_path.endswith(('.yml', '.yaml')):
            import yaml
            config_data = yaml.safe_load(f)
        else:
            raise ValueError("Configuration file must be JSON or YAML format")
    
    return ConfigModel(**config_data)

# Additional Framework Constants
FRAMEWORK_VERSION = "2.1.1"

# Timeframe mapping for multi-timeframe analysis
TIMEFRAME_MAP = {
    'M1': 1, 'M5': 5, 'M15': 15, 'M30': 30, 'H1': 60, 'H4': 240,
    'D1': 1440, 'DAILY': 1440, 'W1': 10080, 'MN1': 43200
}

# Features used for anomaly detection
ANOMALY_FEATURES = [
    'ATR', 'bollinger_bandwidth', 'RSI', 'RealVolume', 'candle_body_size',
    'pct_change', 'candle_body_size_vs_atr', 'atr_vs_daily_atr', 'MACD_hist',
    'wick_to_body_ratio', 'overnight_gap_pct', 'RSI_zscore', 'volume_ma_ratio',
    'volatility_hawkes', 'price_acceleration', 'volume_spike_indicator'
]

# Columns to exclude from feature engineering
NON_FEATURE_COLS = [
    'Open', 'High', 'Low', 'Close', 'RealVolume', 'Volume', 'Symbol', 'Timestamp',
    'signal_pressure', 'target_signal_pressure_class', 'target_timing_score',
    'target_bullish_engulfing', 'target_bearish_engulfing', 'target_volatility_spike',
    'target_signal_pressure_class_h30', 'target_signal_pressure_class_h60',
    'target_signal_pressure_class_h90', 'target_signal_pressure_class_h120'
]

# Model performance thresholds
MODEL_PERFORMANCE_THRESHOLDS = {
    'min_f1_score': 0.45,
    'min_precision': 0.40,
    'min_recall': 0.40,
    'min_accuracy': 0.45,
    'max_overfitting_ratio': 2.0,
    'min_validation_samples': 100
}

# Feature engineering categories
FEATURE_CATEGORIES = {
    'price_features': ['Open', 'High', 'Low', 'Close', 'price_change', 'price_momentum'],
    'volume_features': ['Volume', 'RealVolume', 'volume_ma', 'volume_ratio', 'volume_spike'],
    'technical_indicators': ['RSI', 'MACD', 'bollinger', 'stochastic', 'williams'],
    'volatility_features': ['ATR', 'volatility', 'parkinson_vol', 'yang_zhang_vol'],
    'momentum_features': ['momentum', 'roc', 'acceleration', 'velocity'],
    'statistical_features': ['mean', 'std', 'skew', 'kurtosis', 'autocorr'],
    'behavioral_features': ['gap_analysis', 'pattern_recognition', 'regime_detection'],
    'anomaly_features': ANOMALY_FEATURES
}

# Risk management constants
RISK_MANAGEMENT_CONSTANTS = {
    'max_position_size_pct': 0.10,
    'max_correlation_threshold': 0.70,
    'max_sector_exposure_pct': 0.30,
    'emergency_stop_loss_pct': 0.05,
    'daily_loss_limit_pct': 0.02,
    'max_consecutive_losses': 5
}

# AI analysis constants
AI_ANALYSIS_CONSTANTS = {
    'max_api_retries': 3,
    'api_timeout_seconds': 30,
    'max_context_length': 8000,
    'min_confidence_threshold': 0.60,
    'analysis_cooldown_minutes': 5
}

# Genetic programming constants
GP_CONSTANTS = {
    'population_size': 100,
    'max_generations': 50,
    'mutation_rate': 0.15,
    'crossover_rate': 0.80,
    'tournament_size': 5,
    'max_tree_depth': 8,
    'min_fitness_threshold': 0.60
}

# Telemetry and monitoring constants
TELEMETRY_CONSTANTS = {
    'max_log_file_size_mb': 100,
    'log_retention_days': 30,
    'metrics_collection_interval_seconds': 60,
    'alert_cooldown_minutes': 15,
    'max_memory_usage_pct': 85,
    'max_cpu_usage_pct': 80
}

# Validation constants
VALIDATION_CONSTANTS = {
    'min_training_samples': 1000,
    'min_validation_samples': 200,
    'max_missing_data_pct': 0.20,
    'min_feature_variance': 1e-6,
    'max_feature_correlation': 0.95,
    'min_target_class_samples': 50
}


def validate_config(config: ConfigModel) -> Dict[str, Any]:
    """
    Validate configuration against framework constants and requirements.

    Args:
        config: Configuration to validate

    Returns:
        Validation results dictionary
    """
    validation_results = {
        'is_valid': True,
        'warnings': [],
        'errors': [],
        'recommendations': []
    }

    try:
        # Validate risk management settings
        if config.BASE_RISK_PER_TRADE_PCT > RISK_MANAGEMENT_CONSTANTS['max_position_size_pct']:
            validation_results['warnings'].append(
                f"Risk per trade ({config.BASE_RISK_PER_TRADE_PCT:.3f}) exceeds recommended maximum "
                f"({RISK_MANAGEMENT_CONSTANTS['max_position_size_pct']:.3f})"
            )

        # Validate feature engineering settings
        if config.PCA_N_COMPONENTS > 50:
            validation_results['recommendations'].append(
                "Consider reducing PCA components for better interpretability"
            )

        # Validate model training settings
        if config.OPTUNA_TRIALS < 50:
            validation_results['warnings'].append(
                "Low number of Optuna trials may result in suboptimal hyperparameters"
            )

        # Validate AI settings
        if not hasattr(config, 'CALCULATE_SHAP_VALUES') or not config.CALCULATE_SHAP_VALUES:
            validation_results['recommendations'].append(
                "Enable SHAP values calculation for better model interpretability"
            )

        return validation_results

    except Exception as e:
        validation_results['is_valid'] = False
        validation_results['errors'].append(f"Configuration validation failed: {e}")
        return validation_results


def get_config_summary(config: ConfigModel) -> Dict[str, Any]:
    """
    Get comprehensive configuration summary.

    Args:
        config: Configuration to summarize

    Returns:
        Configuration summary dictionary
    """
    try:
        summary = {
            'framework_version': FRAMEWORK_VERSION,
            'configuration_type': 'comprehensive',
            'core_settings': {
                'initial_capital': config.INITIAL_CAPITAL,
                'base_risk_per_trade': config.BASE_RISK_PER_TRADE_PCT,
                'operating_state': config.operating_state.value if hasattr(config.operating_state, 'value') else str(config.operating_state),
                'feature_selection_method': config.FEATURE_SELECTION_METHOD
            },
            'ai_settings': {
                'optuna_trials': config.OPTUNA_TRIALS,
                'calculate_shap': config.CALCULATE_SHAP_VALUES,
                'use_static_confidence': config.USE_STATIC_CONFIDENCE_GATE
            },
            'risk_management': {
                'max_dd_per_cycle': config.MAX_DD_PER_CYCLE,
                'max_concurrent_trades': config.MAX_CONCURRENT_TRADES,
                'use_tiered_risk': config.USE_TIERED_RISK
            },
            'feature_engineering': {
                'use_pca_reduction': config.USE_PCA_REDUCTION,
                'pca_components': config.PCA_N_COMPONENTS,
                'use_feature_caching': config.USE_FEATURE_CACHING
            },
            'execution_settings': {
                'use_realistic_execution': config.USE_REALISTIC_EXECUTION,
                'simulate_latency': config.SIMULATE_LATENCY,
                'use_variable_slippage': config.USE_VARIABLE_SLIPPAGE
            }
        }

        return summary

    except Exception as e:
        return {'error': f'Failed to generate config summary: {e}'}


def create_default_config(base_path: str) -> ConfigModel:
    """Create a default configuration for testing or initial setup"""
    return ConfigModel(
        BASE_PATH=base_path,
        REPORT_LABEL="Default_Config_V2.1.1",
        INITIAL_CAPITAL=10000.0,
        FEATURE_SELECTION_METHOD="mutual_info",
        strategy_name="Default_Strategy",
        run_timestamp="2024-01-01_00-00-00"
    )


def set_operating_state(config: ConfigModel, new_state: OperatingState,
                       apply_rules: bool = True, baseline_established: bool = True) -> ConfigModel:
    """
    Set the operating state and optionally apply state-based rules.

    Args:
        config: Configuration model instance
        new_state: New operating state to set
        apply_rules: Whether to apply state-based configuration rules
        baseline_established: Whether a baseline has been established

    Returns:
        Updated configuration model
    """
    try:
        old_state = config.operating_state
        config.operating_state = new_state

        logger.info(f"Operating state changed: {old_state.value} -> {new_state.value}")

        if apply_rules:
            config = _apply_operating_state_rules(config, baseline_established)

        return config

    except Exception as e:
        logger.error(f"Error setting operating state: {e}")
        return config


def validate_state_transition(current_state: OperatingState, new_state: OperatingState,
                            performance_data: Dict = None, market_conditions: Dict = None) -> tuple:
    """
    Validate if a state transition is allowed based on performance and market conditions.

    Args:
        current_state: Current operating state
        new_state: Proposed new operating state
        performance_data: Optional performance metrics
        market_conditions: Optional market condition data

    Returns:
        Tuple of (is_valid, reason)
    """
    try:
        # Define valid state transitions
        valid_transitions = {
            OperatingState.CONSERVATIVE_BASELINE: [
                OperatingState.AGGRESSIVE_EXPANSION,
                OperatingState.STRATEGY_ROTATION,
                OperatingState.DRAWDOWN_CONTROL
            ],
            OperatingState.AGGRESSIVE_EXPANSION: [
                OperatingState.CONSERVATIVE_BASELINE,
                OperatingState.STRATEGY_ROTATION,
                OperatingState.DRAWDOWN_CONTROL,
                OperatingState.PROFIT_PROTECTION
            ],
            OperatingState.STRATEGY_ROTATION: [
                OperatingState.CONSERVATIVE_BASELINE,
                OperatingState.AGGRESSIVE_EXPANSION,
                OperatingState.DRAWDOWN_CONTROL
            ],
            OperatingState.DRAWDOWN_CONTROL: [
                OperatingState.CONSERVATIVE_BASELINE,
                OperatingState.STRATEGY_ROTATION
            ],
            OperatingState.PROFIT_PROTECTION: [
                OperatingState.CONSERVATIVE_BASELINE,
                OperatingState.AGGRESSIVE_EXPANSION
            ]
        }

        # Check if transition is structurally valid
        if new_state not in valid_transitions.get(current_state, []):
            return False, f"Invalid transition from {current_state.value} to {new_state.value}"

        # Performance-based validation
        if performance_data:
            mar_ratio = performance_data.get('mar_ratio', 0)
            max_drawdown = performance_data.get('max_drawdown_pct', 0)

            # Prevent aggressive expansion with poor performance
            if new_state == OperatingState.AGGRESSIVE_EXPANSION and mar_ratio < 0.5:
                return False, "Cannot transition to Aggressive Expansion with poor MAR ratio"

            # Force drawdown control for severe drawdowns
            if max_drawdown > 25 and new_state != OperatingState.DRAWDOWN_CONTROL:
                return False, "Must transition to Drawdown Control due to severe drawdown"

        # Market condition validation
        if market_conditions:
            volatility = market_conditions.get('volatility', 0)

            # High volatility restrictions
            if volatility > 0.4 and new_state == OperatingState.AGGRESSIVE_EXPANSION:
                return False, "Cannot use Aggressive Expansion in high volatility environment"

        return True, "Valid state transition"

    except Exception as e:
        logger.error(f"Error validating state transition: {e}")
        return False, f"Validation error: {e}"


def _apply_operating_state_rules(config: ConfigModel, baseline_established: bool = True) -> ConfigModel:
    """
    Apply risk and behavior rules based on the current operating state.

    Args:
        config: Configuration model to update
        baseline_established: Whether a baseline has been established

    Returns:
        Updated configuration model
    """
    try:
        state = config.operating_state

        if state not in config.STATE_BASED_CONFIG:
            logger.warning(f"Operating State '{state.value}' not found in STATE_BASED_CONFIG. Using defaults.")
            return config

        state_rules = config.STATE_BASED_CONFIG[state]

        # Apply drawdown multiplier only if baseline is established
        if baseline_established:
            dd_multiplier = state_rules.get("max_dd_per_cycle_mult", 1.0)
            config.MAX_DD_PER_CYCLE = config.ASSET_CLASS_BASE_DD * dd_multiplier
            logger.info(f"Applied {state.value} DD multiplier ({dd_multiplier:.2f}): "
                       f"MAX_DD_PER_CYCLE = {config.MAX_DD_PER_CYCLE:.2%}")

        # Apply other state-specific configurations
        if "max_concurrent_trades" in state_rules:
            config.MAX_CONCURRENT_TRADES = state_rules["max_concurrent_trades"]

        if "min_f1_gate" in state_rules:
            config.MIN_F1_SCORE_GATE = state_rules["min_f1_gate"]

        logger.info(f"Applied operating state rules for: {state.value}")
        return config

    except Exception as e:
        logger.error(f"Error applying operating state rules: {e}")
        return config


def validate_config_integrity(config: ConfigModel) -> Dict[str, Any]:
    """
    Perform comprehensive configuration integrity validation.

    Args:
        config: Configuration model to validate

    Returns:
        Dictionary containing validation results
    """
    try:
        validation_results = {
            'is_valid': True,
            'critical_errors': [],
            'warnings': [],
            'recommendations': [],
            'dependency_issues': [],
            'performance_concerns': []
        }

        # Critical validation checks
        if config.INITIAL_CAPITAL <= 0:
            validation_results['critical_errors'].append("Initial capital must be positive")
            validation_results['is_valid'] = False

        if config.BASE_RISK_PER_TRADE_PCT <= 0 or config.BASE_RISK_PER_TRADE_PCT >= 1:
            validation_results['critical_errors'].append("Base risk per trade must be between 0 and 1")
            validation_results['is_valid'] = False

        if config.MAX_DD_PER_CYCLE <= 0 or config.MAX_DD_PER_CYCLE >= 1:
            validation_results['critical_errors'].append("Max drawdown per cycle must be between 0 and 1")
            validation_results['is_valid'] = False

        # Path validation
        if not os.path.exists(config.BASE_PATH):
            validation_results['critical_errors'].append(f"Base path does not exist: {config.BASE_PATH}")
            validation_results['is_valid'] = False

        # Risk management validation
        if config.BASE_RISK_PER_TRADE_PCT > 0.05:  # 5%
            validation_results['warnings'].append("Risk per trade exceeds 5% - consider reducing for safety")

        if config.MAX_CONCURRENT_TRADES > 10:
            validation_results['warnings'].append("High number of concurrent trades may increase correlation risk")

        # Feature engineering validation
        if config.PCA_N_COMPONENTS > 100:
            validation_results['warnings'].append("Very high PCA components may lead to overfitting")

        if len(config.selected_features) > 200:
            validation_results['performance_concerns'].append("Large number of features may slow training")

        # AI/ML validation
        if config.OPTUNA_TRIALS < 20:
            validation_results['warnings'].append("Low Optuna trials may result in suboptimal hyperparameters")

        if config.OPTUNA_TRIALS > 500:
            validation_results['performance_concerns'].append("Very high Optuna trials will significantly increase training time")

        # State-based configuration validation
        if config.operating_state not in config.STATE_BASED_CONFIG:
            validation_results['critical_errors'].append(f"Operating state {config.operating_state} not found in STATE_BASED_CONFIG")
            validation_results['is_valid'] = False

        # Confidence gate validation
        for regime, gate in config.REGIME_CONFIDENCE_GATES.items():
            if gate < 0.5 or gate > 0.95:
                validation_results['warnings'].append(f"Confidence gate for {regime} ({gate}) outside recommended range [0.5, 0.95]")

        # Labeling validation
        if config.LABEL_LONG_QUANTILE <= config.LABEL_SHORT_QUANTILE:
            validation_results['critical_errors'].append("Long quantile must be greater than short quantile")
            validation_results['is_valid'] = False

        # Time window validation
        try:
            import pandas as pd
            pd.tseries.frequencies.to_offset(config.TRAINING_WINDOW)
            pd.tseries.frequencies.to_offset(config.RETRAINING_FREQUENCY)
            pd.tseries.frequencies.to_offset(config.FORWARD_TEST_GAP)
        except Exception as e:
            validation_results['critical_errors'].append(f"Invalid time window format: {e}")
            validation_results['is_valid'] = False

        # Performance recommendations
        if config.USE_FEATURE_CACHING and not os.path.exists(os.path.dirname(config.CACHE_PATH)):
            validation_results['recommendations'].append("Create cache directory for better performance")

        if not config.CALCULATE_SHAP_VALUES:
            validation_results['recommendations'].append("Enable SHAP values for better model interpretability")

        # Memory and performance checks
        estimated_memory_gb = (len(config.selected_features) * config.OPTUNA_TRIALS * 0.001)  # Rough estimate
        if estimated_memory_gb > 8:
            validation_results['performance_concerns'].append(f"Estimated memory usage: {estimated_memory_gb:.1f}GB - consider reducing features or trials")

        logger.info(f"Configuration validation completed. Valid: {validation_results['is_valid']}")
        return validation_results

    except Exception as e:
        logger.error(f"Error validating configuration integrity: {e}")
        return {
            'is_valid': False,
            'critical_errors': [f"Validation failed: {e}"],
            'warnings': [],
            'recommendations': [],
            'dependency_issues': [],
            'performance_concerns': []
        }


def check_dependencies(config: ConfigModel = None) -> Dict[str, Any]:
    """
    Check system dependencies and package requirements.

    Args:
        config: Optional configuration model for specific checks

    Returns:
        Dictionary containing dependency check results
    """
    try:
        dependency_results = {
            'all_dependencies_met': True,
            'missing_packages': [],
            'version_issues': [],
            'optional_missing': [],
            'system_requirements': {},
            'recommendations': []
        }

        # Core package checks
        required_packages = {
            'pandas': '1.3.0',
            'numpy': '1.20.0',
            'scikit-learn': '1.0.0',
            'optuna': '2.10.0',
            'pydantic': '1.8.0'
        }

        for package, min_version in required_packages.items():
            try:
                import importlib
                module = importlib.import_module(package)

                if hasattr(module, '__version__'):
                    current_version = module.__version__
                    # Simple version comparison (not perfect but functional)
                    if current_version < min_version:
                        dependency_results['version_issues'].append(
                            f"{package}: current {current_version}, required >= {min_version}"
                        )
                        dependency_results['all_dependencies_met'] = False

            except ImportError:
                dependency_results['missing_packages'].append(package)
                dependency_results['all_dependencies_met'] = False

        # Optional package checks
        optional_packages = {
            'torch': 'PyTorch for neural networks',
            'torch_geometric': 'Graph Neural Networks',
            'shap': 'Model interpretability',
            'plotly': 'Interactive visualizations',
            'yfinance': 'Financial data fetching',
            'ta': 'Technical analysis indicators'
        }

        for package, description in optional_packages.items():
            try:
                importlib.import_module(package)
            except ImportError:
                dependency_results['optional_missing'].append(f"{package} ({description})")

        # System requirements check
        import psutil
        import platform

        dependency_results['system_requirements'] = {
            'python_version': platform.python_version(),
            'platform': platform.system(),
            'cpu_count': psutil.cpu_count(),
            'memory_gb': round(psutil.virtual_memory().total / (1024**3), 1),
            'disk_space_gb': round(psutil.disk_usage('/').free / (1024**3), 1) if platform.system() != 'Windows' else 'N/A'
        }

        # Memory recommendations based on config
        if config:
            estimated_memory_need = len(config.selected_features) * config.OPTUNA_TRIALS * 0.001
            available_memory = dependency_results['system_requirements']['memory_gb']

            if estimated_memory_need > available_memory * 0.8:
                dependency_results['recommendations'].append(
                    f"Consider reducing features or Optuna trials. Estimated need: {estimated_memory_need:.1f}GB, Available: {available_memory:.1f}GB"
                )

        # CPU recommendations
        cpu_count = dependency_results['system_requirements']['cpu_count']
        if config and config.OPTUNA_N_JOBS == -1 and cpu_count > 8:
            dependency_results['recommendations'].append(
                f"Consider limiting OPTUNA_N_JOBS to {cpu_count//2} to avoid system overload"
            )

        logger.info(f"Dependency check completed. All met: {dependency_results['all_dependencies_met']}")
        return dependency_results

    except Exception as e:
        logger.error(f"Error checking dependencies: {e}")
        return {
            'all_dependencies_met': False,
            'missing_packages': [],
            'version_issues': [f"Dependency check failed: {e}"],
            'optional_missing': [],
            'system_requirements': {},
            'recommendations': []
        }


def update_runtime_config(config: ConfigModel, updates: Dict[str, Any],
                         validate_changes: bool = True) -> tuple:
    """
    Update configuration parameters at runtime with validation.

    Args:
        config: Configuration model to update
        updates: Dictionary of parameter updates
        validate_changes: Whether to validate changes before applying

    Returns:
        Tuple of (success, updated_config, validation_results)
    """
    try:
        # Create a copy to avoid modifying original if validation fails
        import copy
        updated_config = copy.deepcopy(config)

        validation_results = {
            'applied_changes': [],
            'rejected_changes': [],
            'warnings': []
        }

        for param_name, new_value in updates.items():
            try:
                # Check if parameter exists
                if not hasattr(updated_config, param_name):
                    validation_results['rejected_changes'].append(
                        f"{param_name}: Parameter does not exist"
                    )
                    continue

                old_value = getattr(updated_config, param_name)

                # Validate specific parameter types
                if validate_changes:
                    is_valid, reason = _validate_parameter_change(param_name, old_value, new_value)
                    if not is_valid:
                        validation_results['rejected_changes'].append(
                            f"{param_name}: {reason}"
                        )
                        continue

                # Apply the change
                setattr(updated_config, param_name, new_value)
                validation_results['applied_changes'].append(
                    f"{param_name}: {old_value} -> {new_value}"
                )

                # Check for side effects and warnings
                warnings = _check_parameter_side_effects(param_name, new_value, updated_config)
                validation_results['warnings'].extend(warnings)

            except Exception as e:
                validation_results['rejected_changes'].append(
                    f"{param_name}: Error applying change - {e}"
                )

        # Validate overall configuration integrity after changes
        if validate_changes and validation_results['applied_changes']:
            integrity_check = validate_config_integrity(updated_config)
            if not integrity_check['is_valid']:
                return False, config, {
                    **validation_results,
                    'integrity_errors': integrity_check['critical_errors']
                }

        success = len(validation_results['applied_changes']) > 0
        logger.info(f"Runtime config update: {len(validation_results['applied_changes'])} applied, "
                   f"{len(validation_results['rejected_changes'])} rejected")

        return success, updated_config, validation_results

    except Exception as e:
        logger.error(f"Error updating runtime configuration: {e}")
        return False, config, {'error': str(e)}


def apply_config_changes(config: ConfigModel, change_set: Dict[str, Any],
                        change_reason: str = "Manual update") -> Dict[str, Any]:
    """
    Apply a set of configuration changes with logging and rollback capability.

    Args:
        config: Configuration model to update
        change_set: Dictionary of changes to apply
        change_reason: Reason for the changes (for logging)

    Returns:
        Dictionary containing change results and rollback information
    """
    try:
        import copy
        import datetime

        # Store original state for rollback
        original_state = {}
        for param_name in change_set.keys():
            if hasattr(config, param_name):
                original_state[param_name] = getattr(config, param_name)

        change_results = {
            'success': False,
            'timestamp': datetime.datetime.now().isoformat(),
            'reason': change_reason,
            'original_state': original_state,
            'applied_changes': {},
            'failed_changes': {},
            'rollback_available': True
        }

        # Apply changes one by one
        for param_name, new_value in change_set.items():
            try:
                if hasattr(config, param_name):
                    old_value = getattr(config, param_name)
                    setattr(config, param_name, new_value)
                    change_results['applied_changes'][param_name] = {
                        'old_value': old_value,
                        'new_value': new_value
                    }
                    logger.info(f"Applied config change: {param_name} = {new_value} (was {old_value})")
                else:
                    change_results['failed_changes'][param_name] = f"Parameter does not exist"

            except Exception as e:
                change_results['failed_changes'][param_name] = str(e)
                logger.error(f"Failed to apply config change {param_name}: {e}")

        # Check if any changes were applied
        change_results['success'] = len(change_results['applied_changes']) > 0

        # Apply state-based rules if operating state changed
        if 'operating_state' in change_results['applied_changes']:
            try:
                config = _apply_operating_state_rules(config)
                change_results['state_rules_applied'] = True
            except Exception as e:
                change_results['state_rules_error'] = str(e)

        return change_results

    except Exception as e:
        logger.error(f"Error applying configuration changes: {e}")
        return {
            'success': False,
            'error': str(e),
            'rollback_available': False
        }


def _validate_parameter_change(param_name: str, old_value: Any, new_value: Any) -> tuple:
    """
    Validate a specific parameter change.

    Args:
        param_name: Name of the parameter
        old_value: Current value
        new_value: Proposed new value

    Returns:
        Tuple of (is_valid, reason)
    """
    try:
        # Type validation
        if type(old_value) != type(new_value) and old_value is not None:
            return False, f"Type mismatch: expected {type(old_value)}, got {type(new_value)}"

        # Specific parameter validations
        if param_name == 'INITIAL_CAPITAL':
            if new_value <= 0:
                return False, "Initial capital must be positive"

        elif param_name == 'BASE_RISK_PER_TRADE_PCT':
            if new_value <= 0 or new_value >= 1:
                return False, "Risk per trade must be between 0 and 1"

        elif param_name == 'MAX_DD_PER_CYCLE':
            if new_value <= 0 or new_value >= 1:
                return False, "Max drawdown must be between 0 and 1"

        elif param_name == 'OPTUNA_TRIALS':
            if new_value < 1:
                return False, "Optuna trials must be at least 1"
            if new_value > 1000:
                return False, "Optuna trials should not exceed 1000 for performance reasons"

        elif param_name == 'MAX_CONCURRENT_TRADES':
            if new_value < 1 or new_value > 20:
                return False, "Concurrent trades must be between 1 and 20"

        elif param_name in ['LABEL_LONG_QUANTILE', 'LABEL_SHORT_QUANTILE']:
            if new_value < 0 or new_value > 1:
                return False, "Quantiles must be between 0 and 1"

        elif param_name.endswith('_PATH'):
            if isinstance(new_value, str) and not os.path.exists(os.path.dirname(new_value)):
                return False, f"Directory does not exist: {os.path.dirname(new_value)}"

        return True, "Valid change"

    except Exception as e:
        return False, f"Validation error: {e}"


def _check_parameter_side_effects(param_name: str, new_value: Any, config: ConfigModel) -> List[str]:
    """
    Check for potential side effects of parameter changes.

    Args:
        param_name: Name of the changed parameter
        new_value: New value
        config: Updated configuration

    Returns:
        List of warning messages
    """
    warnings = []

    try:
        if param_name == 'OPTUNA_TRIALS':
            if new_value > 200:
                warnings.append("High Optuna trials will significantly increase training time")

        elif param_name == 'PCA_N_COMPONENTS':
            if new_value > len(config.selected_features) * 0.8:
                warnings.append("PCA components close to feature count may not provide dimensionality reduction")

        elif param_name == 'BASE_RISK_PER_TRADE_PCT':
            if new_value > 0.02:  # 2%
                warnings.append("High risk per trade may lead to significant losses")

        elif param_name == 'MAX_CONCURRENT_TRADES':
            if new_value > 5:
                warnings.append("Many concurrent trades may increase correlation risk")

        elif param_name == 'operating_state':
            warnings.append("Operating state change will trigger rule updates")

        elif param_name in ['TRAINING_WINDOW', 'RETRAINING_FREQUENCY']:
            warnings.append("Time window changes may require data reprocessing")

        elif param_name.startswith('REGIME_CONFIDENCE_GATES'):
            warnings.append("Confidence gate changes will affect trade frequency")

    except Exception as e:
        warnings.append(f"Error checking side effects: {e}")

    return warnings


def rollback_config_changes(config: ConfigModel, change_results: Dict[str, Any]) -> bool:
    """
    Rollback configuration changes using stored original state.

    Args:
        config: Configuration model to rollback
        change_results: Results from apply_config_changes containing original state

    Returns:
        True if rollback successful, False otherwise
    """
    try:
        if not change_results.get('rollback_available', False):
            logger.error("Rollback not available for this change set")
            return False

        original_state = change_results.get('original_state', {})
        if not original_state:
            logger.error("No original state found for rollback")
            return False

        # Restore original values
        for param_name, original_value in original_state.items():
            try:
                setattr(config, param_name, original_value)
                logger.info(f"Rolled back {param_name} to {original_value}")
            except Exception as e:
                logger.error(f"Failed to rollback {param_name}: {e}")
                return False

        # Reapply state rules if operating state was rolled back
        if 'operating_state' in original_state:
            config = _apply_operating_state_rules(config)

        logger.info("Configuration rollback completed successfully")
        return True

    except Exception as e:
        logger.error(f"Error during configuration rollback: {e}")
        return False


def _log_config_and_environment(config: ConfigModel):
    """Logs key configuration parameters and system environment details."""
    logger.info("--- CONFIGURATION & ENVIRONMENT ---")

    # Log key config parameters
    config_summary = {
        "Strategy": config.strategy_name,
        "Nickname": config.nickname,
        "Initial Capital": f"${config.INITIAL_CAPITAL:,.2f}",
        "Operating State": config.operating_state.value,
        "Training Window": config.TRAINING_WINDOW,
        "Retraining Frequency": config.RETRAINING_FREQUENCY,
        "Max DD per Cycle": f"{config.MAX_DD_PER_CYCLE:.2%}",
        "Base Risk per Trade": f"{config.BASE_RISK_PER_TRADE_PCT:.2%}",
        "Max Concurrent Trades": config.MAX_CONCURRENT_TRADES,
        "Optuna Trials": config.OPTUNA_TRIALS,
        "Feature Selection": config.FEATURE_SELECTION_METHOD,
        "Selected Features Count": len(config.selected_features)
    }

    for key, value in config_summary.items():
        logger.info(f"  {key}: {value}")

    # Log system environment
    try:
        import platform
        import psutil

        system_info = {
            "Python Version": platform.python_version(),
            "Platform": platform.platform(),
            "CPU Count": psutil.cpu_count(),
            "Memory": f"{psutil.virtual_memory().total / (1024**3):.1f} GB",
            "Available Memory": f"{psutil.virtual_memory().available / (1024**3):.1f} GB"
        }

        logger.info("--- SYSTEM ENVIRONMENT ---")
        for key, value in system_info.items():
            logger.info(f"  {key}: {value}")

    except ImportError:
        logger.warning("psutil not available for system info logging")
    except Exception as e:
        logger.error(f"Error logging system environment: {e}")


def _generate_nickname(strategy: str, ai_suggestion: Optional[str], ledger: Dict, ledger_path: str, version: str) -> str:
    """Generates a unique, memorable nickname for the run."""
    if ai_suggestion and ai_suggestion not in ledger.values():
        logger.info(f"Using unique nickname from AI: '{ai_suggestion}'")
        ledger[version] = ai_suggestion
    else:
        if version in ledger:
            logger.info(f"Using existing nickname from ledger for version {version}: '{ledger[version]}'")
            return ledger[version]

        # Generate a new nickname if AI didn't provide a unique one
        import random

        adjectives = [
            "Swift", "Bold", "Clever", "Steady", "Sharp", "Agile", "Fierce", "Wise", "Calm", "Bright",
            "Quick", "Strong", "Smart", "Fast", "Keen", "Alert", "Brave", "Cool", "Deep", "Elite"
        ]

        nouns = [
            "Eagle", "Tiger", "Wolf", "Bear", "Lion", "Hawk", "Fox", "Shark", "Falcon", "Panther",
            "Phoenix", "Dragon", "Viper", "Cobra", "Raven", "Owl", "Lynx", "Jaguar", "Cheetah", "Leopard"
        ]

        # Generate nickname with strategy prefix
        strategy_prefix = strategy[:4].upper() if len(strategy) >= 4 else strategy.upper()
        adjective = random.choice(adjectives)
        noun = random.choice(nouns)
        nickname = f"{strategy_prefix}_{adjective}{noun}"

        # Ensure uniqueness
        counter = 1
        original_nickname = nickname
        while nickname in ledger.values():
            nickname = f"{original_nickname}_{counter}"
            counter += 1

        ledger[version] = nickname
        logger.info(f"Generated new nickname: '{nickname}'")

    # Save updated ledger
    try:
        import json
        with open(ledger_path, 'w') as f:
            json.dump(ledger, f, indent=4)
    except Exception as e:
        logger.error(f"Could not save nickname ledger: {e}")

    return ledger[version]


def _adapt_drawdown_parameters(config: ConfigModel, observed_dd_pct: float, breaker_tripped: bool, baseline_established: bool):
    """
    Adapts the target and max drawdown parameters based on the previous cycle's performance
    and the user's baseline establishment rules. This modifies the config object directly.
    """
    if not baseline_established and breaker_tripped:
        asset_class_base_dd = config.ASSET_CLASS_BASE_DD
        logger.warning("! ADAPTIVE DD (BASELINE PHASE) ! Acknowledging baseline failure. Resetting risk parameters.")

        # Reset to the standard target DD for the asset class. Let the Operating State handle the risk reduction.
        config.TARGET_DD_PCT = round(asset_class_base_dd, 4)
        config.MAX_DD_PER_CYCLE = round(asset_class_base_dd, 4)
        logger.info(f"  - Reset TARGET_DD_PCT and MAX_DD_PER_CYCLE to asset class baseline: {asset_class_base_dd:.2%}")

    elif baseline_established:
        # Adaptive logic for established baselines
        if observed_dd_pct > config.TARGET_DD_PCT * 1.2:  # Exceeded target by 20%
            # Tighten the target slightly
            new_target = max(config.TARGET_DD_PCT * 0.9, config.ASSET_CLASS_BASE_DD * 0.5)
            logger.warning(f"! ADAPTIVE DD ! Observed DD ({observed_dd_pct:.2%}) exceeded target. Tightening TARGET_DD_PCT: {config.TARGET_DD_PCT:.2%} -> {new_target:.2%}")
            config.TARGET_DD_PCT = round(new_target, 4)

        elif observed_dd_pct < config.TARGET_DD_PCT * 0.5 and not breaker_tripped:  # Well under target
            # Relax the target slightly
            new_target = min(config.TARGET_DD_PCT * 1.1, config.ASSET_CLASS_BASE_DD)
            logger.info(f"! ADAPTIVE DD ! Observed DD ({observed_dd_pct:.2%}) well below target. Relaxing TARGET_DD_PCT: {config.TARGET_DD_PCT:.2%} -> {new_target:.2%}")
            config.TARGET_DD_PCT = round(new_target, 4)

    logger.info(f"  - Current DD parameters: TARGET={config.TARGET_DD_PCT:.2%}, MAX={config.MAX_DD_PER_CYCLE:.2%}")


def deep_merge_dicts(original: dict, updates: dict) -> dict:
    """
    Recursively merges two dictionaries. 'updates' values will overwrite
    'original' values, except for nested dicts which are merged.
    """
    result = original.copy()
    for key, value in updates.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = deep_merge_dicts(result[key], value)
        else:
            result[key] = value
    return result


def generate_dynamic_config(primary_asset_class: str, base_config: Dict) -> Dict:
    """Adjusts the base configuration based on the primary asset class."""
    logger.info(f"Dynamically generating configuration for primary asset class: {primary_asset_class}")
    dynamic_config = base_config.copy()

    if primary_asset_class == 'Commodities' or primary_asset_class == 'Indices':
        logger.info("Adjusting config for Commodities/Indices: Higher risk, wider TP/SL.")
        dynamic_config.update({
            "TP_ATR_MULTIPLIER": 2.5,
            "SL_ATR_MULTIPLIER": 2.0,
            "BASE_RISK_PER_TRADE_PCT": 0.012,
            "TARGET_DD_PCT": 0.30,
            "ASSET_CLASS_BASE_DD": 0.30, # Set the pristine default
        })
    elif primary_asset_class == 'Crypto':
        logger.info("Adjusting config for Crypto: Highest risk, widest TP/SL.")
        dynamic_config.update({
            "TP_ATR_MULTIPLIER": 3.0,
            "SL_ATR_MULTIPLIER": 2.5,
            "BASE_RISK_PER_TRADE_PCT": 0.015,
            "TARGET_DD_PCT": 0.35,
            "ASSET_CLASS_BASE_DD": 0.35, # Set the pristine default
        })
    else: # Default for Forex
        logger.info("Using standard configuration for Forex.")
        dynamic_config.update({
            "TP_ATR_MULTIPLIER": 2.0,
            "SL_ATR_MULTIPLIER": 1.2, # MODIFICATION: Tightened from 1.5
            "BASE_RISK_PER_TRADE_PCT": 0.0025, # MODIFICATION: Reduced from 0.01 (1%) to 0.25%
            "TARGET_DD_PCT": 0.25,
            "ASSET_CLASS_BASE_DD": 0.25, # Set the pristine default
        })
    return dynamic_config


def _update_operating_state(config: ConfigModel, cycle_history: List[Dict], current_state: OperatingState, df_slice: pd.DataFrame, all_time_peak_equity: float, current_run_equity: float) -> OperatingState:
    """
    Analyzes recent performance and market data to determine the next operating state.
    Implements a two-strike rule for circuit breaker events based on cycle history.
    """
    from .utils import _is_maintenance_period

    if df_slice.empty:
        return current_state

    # --- Priority 1: Check for external or critical overrides ---
    is_maintenance, reason = _is_maintenance_period()
    if is_maintenance and current_state != OperatingState.MAINTENANCE_DORMANCY:
        logger.warning(f"! STATE CHANGE ! Entering MAINTENANCE_DORMANCY due to: {reason}")
        return OperatingState.MAINTENANCE_DORMANCY

    if not cycle_history:
        return current_state

    # --- Check for a stable baseline MUST include a check for actual trades. ---
    baseline_established = False
    if len(cycle_history) >= 2:
        # --- A baseline is only established if the last two cycles completed AND executed trades.
        valid_cycles = [
            c for c in cycle_history[-2:]
            if c.get("status") == "Completed" and c.get("metrics", {}).get("NumTrades", 0) > 0
        ]
        if len(valid_cycles) == 2:
            baseline_established = True
            logger.info("  - State Check: Confirmed that a stable trading baseline (2+ completed cycles with trades) has been established.")
        else:
            logger.info("  - State Check: A stable trading baseline has NOT yet been established.")

    # --- Priority 2: Handle Circuit Breaker with a proper two-strike rule ---
    last_cycle = cycle_history[-1]
    previous_cycle = cycle_history[-2] if len(cycle_history) > 1 else None

    last_cycle_tripped = "Breaker Tripped" in last_cycle.get("status", "")
    previous_cycle_tripped = "Breaker Tripped" in previous_cycle.get("status", "") if previous_cycle else False

    if last_cycle_tripped:
        if previous_cycle_tripped:
            if baseline_established:
                logger.critical("! STATE ESCALATION ! Second consecutive circuit breaker trip detected AFTER establishing a baseline. Escalating to DRAWDOWN_CONTROL.")
                return OperatingState.DRAWDOWN_CONTROL
            else:
                logger.warning("! STATE PERSISTENCE ! Second consecutive circuit breaker trip while STILL establishing a baseline. Remaining in PERFORMANCE_REVIEW to maintain risk and find a working model.")
                return OperatingState.PERFORMANCE_REVIEW
        else:
            logger.warning("! STATE CHANGE ! Circuit breaker tripped. Entering PERFORMANCE_REVIEW for AI-led adjustment.")
            return OperatingState.PERFORMANCE_REVIEW

    # --- Priority 3: Handle recovery from probationary/drawdown states ---
    pnl = last_cycle.get("metrics", {}).get('total_net_profit', 0)
    mar_ratio = last_cycle.get("metrics", {}).get('mar_ratio', 0)

    if current_state == OperatingState.PERFORMANCE_REVIEW:
        if pnl > 0 and mar_ratio > 0.1:
            logger.info("! STATE CHANGE ! AI intervention successful. Recovered from performance dip. Returning to CONSERVATIVE_BASELINE.")
            return OperatingState.CONSERVATIVE_BASELINE
        else:
            logger.warning(f"! STATE STABILITY ! AI intervention did not lead to strong recovery (MAR: {mar_ratio:.2f}). Remaining in PERFORMANCE_REVIEW.")
            return OperatingState.PERFORMANCE_REVIEW

    if current_state == OperatingState.DRAWDOWN_CONTROL:
        if pnl > 0 and mar_ratio > 0.3:
            logger.info("! STATE CHANGE ! Positive performance observed. Moving from Drawdown Control to CONSERVATIVE_BASELINE.")
            return OperatingState.CONSERVATIVE_BASELINE
        else:
            logger.info("Performance still weak. Remaining in DRAWDOWN_CONTROL.")
            return OperatingState.DRAWDOWN_CONTROL

    # --- Priority 4: Standard performance-based transitions ---
    if current_state == OperatingState.CONSERVATIVE_BASELINE:
        # --- We now use the corrected 'baseline_established' flag
        if baseline_established:
            logger.info("! STATE CHANGE ! Consistent positive performance. Moving from Baseline to AGGRESSIVE_EXPANSION.")
            return OperatingState.AGGRESSIVE_EXPANSION

    if current_state == OperatingState.AGGRESSIVE_EXPANSION and (pnl < 0 or mar_ratio < 0.2):
        logger.warning("! STATE CHANGE ! Performance dip detected. Moving from Aggressive back to CONSERVATIVE_BASELINE.")
        return OperatingState.CONSERVATIVE_BASELINE

    # --- Priority 5: Revert from temporary, market-driven states ---
    temporary_states = [
        OperatingState.VOLATILITY_SPIKE, OperatingState.LIQUIDITY_CRUNCH,
        OperatingState.NEWS_SENTIMENT_ALERT, OperatingState.OPPORTUNISTIC_SURGE,
        OperatingState.PROFIT_PROTECTION
    ]
    if current_state in temporary_states:
        logger.info(f"! STATE CHANGE ! Reverting from temporary state '{current_state.value}' to CONSERVATIVE_BASELINE.")
        return OperatingState.CONSERVATIVE_BASELINE

    return current_state


def validate_framework_configuration() -> Dict[str, Any]:
    """
    Validate framework configuration and data source availability.

    Returns:
        Dict containing validation results and recommendations
    """
    import os

    results = {
        "framework_self_sufficient": True,
        "gemini_configured": False,
        "issues": [],
        "recommendations": [],
        "available_sources": [
            "yahoo",           # Yahoo Finance (free, no API key)
            "sample_data",     # Generated sample data
            "csv_files",       # Local CSV files
            "web_scraping"     # Web scraping capabilities
        ]
    }

    # Check Gemini API key (only optional API key)
    gemini_key = os.getenv("GEMINI_API_KEY")
    if gemini_key and "YOUR" not in gemini_key and "PASTE" not in gemini_key:
        results["gemini_configured"] = True
        results["available_sources"].append("gemini_search")
        results["recommendations"].append("Gemini AI is configured for enhanced data collection")
    else:
        results["issues"].append("GEMINI_API_KEY not configured (optional)")
        results["recommendations"].append("Set GEMINI_API_KEY for AI-powered data search and analysis")
        results["recommendations"].append("Get free API key at: https://makersuite.google.com/app/apikey")

    # Framework capabilities
    results["capabilities"] = [
        "Self-sufficient financial data collection",
        "Yahoo Finance integration (no API key required)",
        "Sample data generation for testing and development",
        "CSV file import for custom datasets",
        "Web scraping with AI guidance (when Gemini is configured)"
    ]

    # Overall status
    results["ready_to_use"] = True  # Framework is always ready
    results["enhanced_features"] = results["gemini_configured"]

    return results
