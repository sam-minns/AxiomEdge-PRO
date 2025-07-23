# =============================================================================
# CONFIGURATION & VALIDATION MODULE
# =============================================================================

import os
from enum import Enum
from typing import Dict, List, Optional, Any
from pathlib import Path
from pydantic import BaseModel, Field, validator, confloat, conint, DirectoryPath
import logging

logger = logging.getLogger(__name__)

class OperatingState(Enum):
    CONSERVATIVE_BASELINE = "conservative_baseline"
    AGGRESSIVE_GROWTH = "aggressive_growth"
    DEFENSIVE_PRESERVATION = "defensive_preservation"
    EXPERIMENTAL_RESEARCH = "experimental_research"
    CRISIS_MANAGEMENT = "crisis_management"

class EarlyInterventionConfig(BaseModel):
    enabled: bool = True
    max_consecutive_losses: int = 5
    max_drawdown_threshold: float = 0.15
    min_win_rate_threshold: float = 0.30
    lookback_periods: int = 10

class ConfigModel(BaseModel):
    # --- Core Run, Capital & State Parameters ---
    BASE_PATH: DirectoryPath
    REPORT_LABEL: str
    INITIAL_CAPITAL: confloat(gt=0)
    operating_state: OperatingState = OperatingState.CONSERVATIVE_BASELINE
    TARGET_DD_PCT: confloat(gt=0) = 0.25 
    ASSET_CLASS_BASE_DD: confloat(gt=0) = 0.25

    FEATURE_SELECTION_METHOD: str 
    SHADOW_SET_VALIDATION: bool = True

    # --- AI & Optimization Parameters ---
    OPTUNA_TRIALS: conint(gt=0) = 75
    OPTUNA_N_JOBS: conint(ge=-1) = 1
    MAX_TRAINING_RETRIES_PER_CYCLE: conint(ge=0) = 3
    CALCULATE_SHAP_VALUES: bool = True

    # --- Walk-Forward Parameters ---
    TRAINING_WINDOW: str
    RETRAINING_FREQUENCY: str
    FORWARD_TEST_GAP: str
    LOOKAHEAD_CANDLES: conint(gt=0)

    # --- Risk Management ---
    RISK_CAP_PER_TRADE_USD: confloat(gt=0)
    BASE_RISK_PER_TRADE_PCT: confloat(gt=0, le=1)
    CONFIDENCE_TIERS: Dict[str, float]
    
    # --- Take Profit Ladder ---
    USE_TP_LADDER: bool = False
    TP_LADDER_LEVELS_PCT: Optional[List[float]] = None
    TP_LADDER_RISK_MULTIPLIERS: Optional[List[float]] = None

    # --- Technical Indicators ---
    EMA_PERIODS: List[int] = [8, 13, 21, 34, 55, 89, 144]
    RSI_PERIODS: List[int] = [14, 21]
    BOLLINGER_PERIODS: List[int] = [20]
    MACD_PARAMS: Dict[str, int] = {"fast": 12, "slow": 26, "signal": 9}

    # --- Feature Engineering ---
    FEATURE_LOOKBACK_PERIODS: List[int] = [5, 10, 20, 50]
    VOLATILITY_WINDOWS: List[int] = [10, 20, 50]
    MOMENTUM_WINDOWS: List[int] = [5, 10, 20]

    # --- Model Parameters ---
    MODEL_VALIDATION_SPLIT: confloat(gt=0, lt=1) = 0.2
    EARLY_STOPPING_PATIENCE: conint(gt=0) = 10
    MAX_EPOCHS: conint(gt=0) = 100
    BATCH_SIZE: conint(gt=0) = 32

    # --- Genetic Programming ---
    GP_POPULATION_SIZE: conint(gt=0) = 100
    GP_GENERATIONS: conint(gt=0) = 50
    GP_MUTATION_RATE: confloat(gt=0, lt=1) = 0.1
    GP_CROSSOVER_RATE: confloat(gt=0, lt=1) = 0.8

    # --- File Paths ---
    NICKNAME_LEDGER_PATH: Optional[str] = None
    REPORT_SAVE_PATH: str = "reports/performance_report.txt"
    PARAMS_LOG_FILE: str = "logs/params_log.json"
    TELEMETRY_LOG_FILE: str = "logs/telemetry_log.json"

    # --- Validation & Safety ---
    MAX_PARAM_DRIFT_TOLERANCE: confloat(gt=0) = 50.0
    ENABLE_PARAMETER_VALIDATION: bool = True
    ENABLE_EARLY_INTERVENTION: bool = True
    early_intervention: EarlyInterventionConfig = Field(default_factory=EarlyInterventionConfig)

    # --- Runtime State ---
    selected_features: List[str] = Field(default_factory=list)
    run_timestamp: str = ""
    strategy_name: str = ""
    nickname: Optional[str] = None
    analysis_notes: Optional[str] = None

    @validator('CONFIDENCE_TIERS')
    def validate_confidence_tiers(cls, v):
        if not isinstance(v, dict):
            raise ValueError("CONFIDENCE_TIERS must be a dictionary")
        for tier, multiplier in v.items():
            if not isinstance(multiplier, (int, float)) or multiplier <= 0:
                raise ValueError(f"Confidence tier '{tier}' must have a positive numeric multiplier")
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

def create_default_config(base_path: str) -> ConfigModel:
    """Create a default configuration for testing or initial setup"""
    return ConfigModel(
        BASE_PATH=base_path,
        REPORT_LABEL="Default_Config",
        INITIAL_CAPITAL=10000.0,
        FEATURE_SELECTION_METHOD="mutual_info",
        TRAINING_WINDOW="30D",
        RETRAINING_FREQUENCY="7D", 
        FORWARD_TEST_GAP="1D",
        LOOKAHEAD_CANDLES=5,
        RISK_CAP_PER_TRADE_USD=500.0,
        BASE_RISK_PER_TRADE_PCT=0.02,
        CONFIDENCE_TIERS={"high": 1.5, "medium": 1.0, "low": 0.5},
        strategy_name="Default_Strategy",
        run_timestamp="2024-01-01_00-00-00"
    )
