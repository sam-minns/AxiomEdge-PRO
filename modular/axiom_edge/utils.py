# =============================================================================
# UTILITIES MODULE
# =============================================================================

import os
import json
import logging
import multiprocessing
import psutil
from typing import Dict, Any, List, Optional
from datetime import datetime
import pandas as pd
import numpy as np

logger = logging.getLogger(__name__)

def get_optimal_system_settings() -> Dict[str, Any]:
    """
    Determine optimal system settings based on available resources.
    
    Returns:
        Dictionary containing optimal settings for the current system
    """
    try:
        # Get system information
        cpu_count = multiprocessing.cpu_count()
        memory_gb = psutil.virtual_memory().total / (1024**3)
        
        # Determine optimal number of workers
        if cpu_count >= 8 and memory_gb >= 16:
            num_workers = min(cpu_count - 2, 8)  # Leave 2 cores free, max 8 workers
            use_parallel = True
        elif cpu_count >= 4 and memory_gb >= 8:
            num_workers = min(cpu_count - 1, 4)  # Leave 1 core free, max 4 workers
            use_parallel = True
        else:
            num_workers = 1
            use_parallel = False
        
        # Determine optimal batch sizes
        if memory_gb >= 32:
            batch_size = 1024
            chunk_size = 10000
        elif memory_gb >= 16:
            batch_size = 512
            chunk_size = 5000
        elif memory_gb >= 8:
            batch_size = 256
            chunk_size = 2500
        else:
            batch_size = 128
            chunk_size = 1000
        
        settings = {
            'num_workers': num_workers,
            'use_parallel': use_parallel,
            'batch_size': batch_size,
            'chunk_size': chunk_size,
            'cpu_count': cpu_count,
            'memory_gb': round(memory_gb, 2),
            'recommended_max_features': min(1000, int(memory_gb * 50))
        }
        
        logger.info(f"System settings: {cpu_count} CPUs, {memory_gb:.1f}GB RAM")
        logger.info(f"Optimal settings: {num_workers} workers, batch_size={batch_size}")
        
        return settings
        
    except Exception as e:
        logger.warning(f"Could not determine optimal system settings: {e}")
        return {
            'num_workers': 1,
            'use_parallel': False,
            'batch_size': 128,
            'chunk_size': 1000,
            'cpu_count': 1,
            'memory_gb': 4.0,
            'recommended_max_features': 200
        }

def json_serializer_default(obj):
    """
    Default JSON serializer for objects not serializable by default json code.
    
    Args:
        obj: Object to serialize
        
    Returns:
        JSON serializable representation of the object
    """
    if isinstance(obj, (datetime, pd.Timestamp)):
        return obj.isoformat()
    elif isinstance(obj, pd.Series):
        return obj.to_dict()
    elif isinstance(obj, pd.DataFrame):
        return obj.to_dict('records')
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, (np.int64, np.int32, np.int16, np.int8)):
        return int(obj)
    elif isinstance(obj, (np.float64, np.float32, np.float16)):
        return float(obj)
    elif isinstance(obj, np.bool_):
        return bool(obj)
    elif hasattr(obj, '__dict__'):
        return str(obj)
    else:
        return str(obj)

def flush_loggers():
    """Flush all active loggers to ensure all messages are written"""
    for handler in logging.root.handlers:
        handler.flush()
    
    # Also flush specific loggers
    for name in logging.Logger.manager.loggerDict:
        logger_obj = logging.getLogger(name)
        for handler in logger_obj.handlers:
            handler.flush()

def setup_logging(log_level: str = "INFO", log_file: Optional[str] = None) -> None:
    """
    Setup logging configuration for the entire framework.
    
    Args:
        log_level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        log_file: Optional log file path
    """
    # Create logs directory if it doesn't exist
    if log_file:
        os.makedirs(os.path.dirname(log_file), exist_ok=True)
    
    # Configure logging format
    log_format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    
    # Setup handlers
    handlers = [logging.StreamHandler()]
    if log_file:
        handlers.append(logging.FileHandler(log_file))
    
    # Configure logging
    logging.basicConfig(
        level=getattr(logging, log_level.upper()),
        format=log_format,
        handlers=handlers,
        force=True
    )
    
    # Reduce noise from external libraries
    logging.getLogger('urllib3').setLevel(logging.WARNING)
    logging.getLogger('requests').setLevel(logging.WARNING)
    logging.getLogger('matplotlib').setLevel(logging.WARNING)

def validate_data_quality(data: pd.DataFrame, required_columns: List[str]) -> Dict[str, Any]:
    """
    Validate the quality of input data.
    
    Args:
        data: DataFrame to validate
        required_columns: List of required column names
        
    Returns:
        Dictionary containing validation results
    """
    validation_results = {
        'is_valid': True,
        'errors': [],
        'warnings': [],
        'statistics': {}
    }
    
    # Check if data is empty
    if data.empty:
        validation_results['is_valid'] = False
        validation_results['errors'].append("Data is empty")
        return validation_results
    
    # Check required columns
    missing_columns = [col for col in required_columns if col not in data.columns]
    if missing_columns:
        validation_results['is_valid'] = False
        validation_results['errors'].append(f"Missing required columns: {missing_columns}")
    
    # Check for excessive missing values
    missing_pct = (data.isnull().sum() / len(data)) * 100
    high_missing_cols = missing_pct[missing_pct > 50].index.tolist()
    if high_missing_cols:
        validation_results['warnings'].append(f"Columns with >50% missing values: {high_missing_cols}")
    
    # Check for duplicate rows
    duplicate_count = data.duplicated().sum()
    if duplicate_count > 0:
        validation_results['warnings'].append(f"Found {duplicate_count} duplicate rows")
    
    # Check data types
    numeric_columns = data.select_dtypes(include=[np.number]).columns
    if len(numeric_columns) == 0:
        validation_results['warnings'].append("No numeric columns found")
    
    # Calculate statistics
    validation_results['statistics'] = {
        'total_rows': len(data),
        'total_columns': len(data.columns),
        'missing_values_pct': missing_pct.mean(),
        'duplicate_rows': duplicate_count,
        'numeric_columns': len(numeric_columns),
        'memory_usage_mb': data.memory_usage(deep=True).sum() / (1024**2)
    }
    
    return validation_results

def create_directory_structure(base_path: str) -> None:
    """
    Create the standard directory structure for AxiomEdge.
    
    Args:
        base_path: Base directory path
    """
    directories = [
        'data',
        'data/raw',
        'data/processed',
        'data/cache',
        'models',
        'reports',
        'logs',
        'configs',
        'results',
        'results/backtests',
        'results/features',
        'results/models'
    ]
    
    for directory in directories:
        dir_path = os.path.join(base_path, directory)
        os.makedirs(dir_path, exist_ok=True)
        logger.info(f"Created directory: {dir_path}")

def save_results(results: Dict[str, Any], filepath: str, format: str = "json") -> None:
    """
    Save results to file in specified format.
    
    Args:
        results: Results dictionary to save
        filepath: Output file path
        format: Output format (json, csv, pickle)
    """
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    
    try:
        if format.lower() == "json":
            with open(filepath, 'w') as f:
                json.dump(results, f, indent=2, default=json_serializer_default)
        elif format.lower() == "pickle":
            import pickle
            with open(filepath, 'wb') as f:
                pickle.dump(results, f)
        elif format.lower() == "csv" and isinstance(results, dict):
            # Convert dict to DataFrame if possible
            df = pd.DataFrame([results])
            df.to_csv(filepath, index=False)
        else:
            raise ValueError(f"Unsupported format: {format}")
            
        logger.info(f"Results saved to {filepath}")
        
    except Exception as e:
        logger.error(f"Failed to save results to {filepath}: {e}")

def load_results(filepath: str) -> Any:
    """
    Load results from file.
    
    Args:
        filepath: Path to results file
        
    Returns:
        Loaded results
    """
    try:
        if filepath.endswith('.json'):
            with open(filepath, 'r') as f:
                return json.load(f)
        elif filepath.endswith('.pkl') or filepath.endswith('.pickle'):
            import pickle
            with open(filepath, 'rb') as f:
                return pickle.load(f)
        elif filepath.endswith('.csv'):
            return pd.read_csv(filepath)
        else:
            raise ValueError(f"Unsupported file format: {filepath}")
            
    except Exception as e:
        logger.error(f"Failed to load results from {filepath}: {e}")
        return None

def calculate_performance_metrics(returns: pd.Series) -> Dict[str, float]:
    """
    Calculate standard performance metrics from returns series.
    
    Args:
        returns: Series of returns
        
    Returns:
        Dictionary of performance metrics
    """
    if returns.empty:
        return {}
    
    # Remove any infinite or NaN values
    clean_returns = returns.replace([np.inf, -np.inf], np.nan).dropna()
    
    if clean_returns.empty:
        return {}
    
    # Calculate metrics
    total_return = (1 + clean_returns).prod() - 1
    annualized_return = (1 + clean_returns.mean()) ** 252 - 1
    volatility = clean_returns.std() * np.sqrt(252)
    sharpe_ratio = annualized_return / volatility if volatility > 0 else 0
    
    # Drawdown calculation
    cumulative = (1 + clean_returns).cumprod()
    running_max = cumulative.expanding().max()
    drawdown = (cumulative - running_max) / running_max
    max_drawdown = drawdown.min()
    
    # Win rate
    win_rate = (clean_returns > 0).mean()
    
    return {
        'total_return': total_return,
        'annualized_return': annualized_return,
        'volatility': volatility,
        'sharpe_ratio': sharpe_ratio,
        'max_drawdown': max_drawdown,
        'win_rate': win_rate,
        'total_trades': len(clean_returns),
        'avg_return': clean_returns.mean(),
        'std_return': clean_returns.std()
    }
