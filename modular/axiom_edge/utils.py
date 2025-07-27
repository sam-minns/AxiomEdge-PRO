# =============================================================================
# UTILITIES MODULE
# =============================================================================

import os
import json
import logging
import multiprocessing
import psutil
import pathlib
from typing import Dict, Any, List, Optional
from datetime import datetime, date
from enum import Enum
import pandas as pd
import numpy as np

logger = logging.getLogger(__name__)

def get_optimal_system_settings() -> Dict[str, int]:
    """
    Analyze system resources to determine optimal processing settings.

    Implements conservative logic to prevent out-of-memory errors by considering
    both CPU cores and available RAM when determining parallel processing limits.

    Returns:
        Dictionary containing optimal settings for parallel processing
    """
    settings = {}
    try:
        cpu_count = os.cpu_count()
        virtual_mem = psutil.virtual_memory()
        available_gb = virtual_mem.available / (1024**3)

        logger.info("System Resource Analysis")
        logger.info(f"  Total CPU Cores: {cpu_count}")
        logger.info(f"  Available RAM: {available_gb:.2f} GB")

        # Determine optimal worker count based on system resources
        if cpu_count is None:
            num_workers = 1
            logger.warning("Could not determine CPU count. Defaulting to 1 worker.")
        elif cpu_count <= 4:
            # Conservative approach for systems with few cores
            num_workers = max(1, cpu_count - 1)
            logger.info(f"  Low CPU count detected. Setting to {num_workers} worker(s).")
        elif available_gb < 8:
            # Limit workers on memory-constrained systems
            num_workers = max(1, cpu_count // 2)
            logger.warning(f"  Low memory detected (<8GB). Limiting to {num_workers} worker(s) to prevent OOM.")
        else:
            # Balanced approach: reserve cores for OS and main process
            num_workers = max(1, cpu_count - 2)
            logger.info(f"  System capable. Setting to {num_workers} worker(s).")

        settings['num_workers'] = num_workers

    except Exception as e:
        logger.error(f"Could not determine optimal system settings: {e}. Defaulting to 1 worker.")
        settings['num_workers'] = 1

    return settings

def json_serializer_default(o):
    """
    Custom JSON serializer for complex Python types.

    Handles serialization of Enums, Paths, and datetime objects that are not
    natively JSON serializable.

    Args:
        o: Object to serialize

    Returns:
        JSON-serializable representation of the object

    Raises:
        TypeError: If object type is not supported
    """
    if isinstance(o, Enum):
        return o.value
    if isinstance(o, (pathlib.Path, datetime, date)):
        return str(o)
    if isinstance(o, pd.Timestamp):
        return o.isoformat()
    elif isinstance(o, pd.Series):
        return o.to_dict()
    elif isinstance(o, pd.DataFrame):
        return o.to_dict('records')
    elif isinstance(o, np.ndarray):
        return o.tolist()
    elif isinstance(o, (np.int64, np.int32, np.int16, np.int8)):
        return int(o)
    elif isinstance(o, (np.float64, np.float32, np.float16)):
        return float(o)
    elif isinstance(o, np.bool_):
        return bool(o)

    raise TypeError(f"Object of type {o.__class__.__name__} is not JSON serializable")

def _sanitize_keys_for_json(obj: Any) -> Any:
    """
    Recursively sanitise dictionary keys for JSON serialisation.

    Converts non-string keys (like Enums) to strings to ensure the entire
    structure is JSON-serialisable.

    Args:
        obj: Object to sanitise (dict, list, or primitive)

    Returns:
        Sanitised object with string keys
    """
    if isinstance(obj, dict):
        return {str(k.value if isinstance(k, Enum) else k): _sanitize_keys_for_json(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [_sanitize_keys_for_json(elem) for elem in obj]
    else:
        return obj

def _parallel_process_symbol_wrapper(symbol_tuple, feature_engineer_instance):
    """
    Wrapper function for parallel processing of individual symbols.

    Enables multiprocessing by providing a top-level function that can be pickled
    and distributed across worker processes.

    Args:
        symbol_tuple: Tuple containing (symbol, symbol_data_by_tf)
        feature_engineer_instance: FeatureEngineer instance to use for processing

    Returns:
        Processed feature data for the symbol
    """
    symbol, symbol_data_by_tf = symbol_tuple
    logger.info(f"  Starting parallel processing for symbol: {symbol}...")
    return feature_engineer_instance._process_single_symbol_stack(symbol_data_by_tf)

def flush_loggers():
    """
    Flush all handlers for all active loggers to disk.

    Ensures all pending log messages are written to their respective outputs
    before the application continues or terminates.
    """
    for handler in logging.getLogger().handlers:
        handler.flush()
    for handler in logging.getLogger("ML_Trading_Framework").handlers:
        handler.flush()

def _setup_logging(log_file_path: str, report_label: str):
    """
    Sets up comprehensive logging for the AxiomEdge framework.

    Args:
        log_file_path: Path to the log file
        report_label: Label for the logging session
    """
    # Clear any existing handlers
    for handler in logging.root.handlers[:]:
        logging.root.removeHandler(handler)

    # Create formatter
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )

    # File handler
    file_handler = logging.FileHandler(log_file_path, encoding='utf-8')
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(formatter)

    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(formatter)

    # Configure root logger
    logging.root.setLevel(logging.DEBUG)
    logging.root.addHandler(file_handler)
    logging.root.addHandler(console_handler)

    # Log initial message
    logger = logging.getLogger("ML_Trading_Framework")
    logger.info(f"Logging initialized for {report_label}")
    logger.info(f"Log file: {log_file_path}")

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


def _get_available_features_from_df(df: pd.DataFrame) -> List[str]:
    """
    Extract available features from DataFrame, excluding target and metadata columns.

    Args:
        df: DataFrame to analyze

    Returns:
        List of feature column names
    """
    if df is None or df.empty:
        return []

    # Columns to exclude from features
    exclude_patterns = [
        'target_', 'label_', 'timestamp', 'datetime', 'date', 'time',
        'symbol', 'ticker', 'instrument', 'index', 'id', 'uid'
    ]

    available_features = []
    for col in df.columns:
        col_lower = col.lower()
        if not any(pattern in col_lower for pattern in exclude_patterns):
            available_features.append(col)

    logger.info(f"Found {len(available_features)} available features from {len(df.columns)} total columns")
    return available_features


def _create_label_distribution_report(df: pd.DataFrame, target_columns: List[str]) -> Dict[str, Any]:
    """
    Create comprehensive label distribution report for target columns.

    Args:
        df: DataFrame containing target columns
        target_columns: List of target column names

    Returns:
        Dictionary containing distribution analysis
    """
    report = {
        'total_samples': len(df),
        'target_columns': target_columns,
        'distributions': {},
        'balance_analysis': {},
        'recommendations': []
    }

    try:
        for target_col in target_columns:
            if target_col not in df.columns:
                continue

            target_data = df[target_col].dropna()
            value_counts = target_data.value_counts()

            # Calculate distribution metrics
            distribution_info = {
                'unique_values': len(value_counts),
                'value_counts': value_counts.to_dict(),
                'percentages': (value_counts / len(target_data) * 100).to_dict(),
                'most_common': value_counts.index[0] if len(value_counts) > 0 else None,
                'least_common': value_counts.index[-1] if len(value_counts) > 0 else None,
                'imbalance_ratio': value_counts.max() / value_counts.min() if len(value_counts) > 1 and value_counts.min() > 0 else 1.0
            }

            report['distributions'][target_col] = distribution_info

            # Balance analysis
            if len(value_counts) > 1:
                min_class_pct = (value_counts.min() / len(target_data)) * 100
                max_class_pct = (value_counts.max() / len(target_data)) * 100

                balance_status = 'balanced'
                if distribution_info['imbalance_ratio'] > 10:
                    balance_status = 'severely_imbalanced'
                elif distribution_info['imbalance_ratio'] > 3:
                    balance_status = 'moderately_imbalanced'
                elif distribution_info['imbalance_ratio'] > 1.5:
                    balance_status = 'slightly_imbalanced'

                report['balance_analysis'][target_col] = {
                    'status': balance_status,
                    'imbalance_ratio': distribution_info['imbalance_ratio'],
                    'min_class_percentage': min_class_pct,
                    'max_class_percentage': max_class_pct
                }

                # Generate recommendations
                if balance_status != 'balanced':
                    report['recommendations'].append(
                        f"{target_col}: Consider class balancing techniques (imbalance ratio: {distribution_info['imbalance_ratio']:.2f})"
                    )

        logger.info(f"Label distribution report created for {len(target_columns)} target columns")
        return report

    except Exception as e:
        logger.error(f"Error creating label distribution report: {e}")
        return report


def calculate_memory_usage(obj: Any) -> Dict[str, float]:
    """
    Calculate memory usage of an object in different units.

    Args:
        obj: Object to analyze

    Returns:
        Dictionary with memory usage in different units
    """
    try:
        import sys

        if isinstance(obj, pd.DataFrame):
            memory_bytes = obj.memory_usage(deep=True).sum()
        elif isinstance(obj, pd.Series):
            memory_bytes = obj.memory_usage(deep=True)
        elif isinstance(obj, np.ndarray):
            memory_bytes = obj.nbytes
        else:
            memory_bytes = sys.getsizeof(obj)

        return {
            'bytes': memory_bytes,
            'kb': memory_bytes / 1024,
            'mb': memory_bytes / (1024**2),
            'gb': memory_bytes / (1024**3)
        }

    except Exception as e:
        logger.error(f"Error calculating memory usage: {e}")
        return {'bytes': 0, 'kb': 0, 'mb': 0, 'gb': 0}


def optimize_dataframe(df: pd.DataFrame, aggressive: bool = False) -> pd.DataFrame:
    """
    Optimize DataFrame memory usage by downcasting numeric types.

    Args:
        df: DataFrame to optimize
        aggressive: Whether to use aggressive optimization

    Returns:
        Optimized DataFrame
    """
    try:
        original_memory = df.memory_usage(deep=True).sum()
        optimized_df = df.copy()

        # Optimize numeric columns
        for col in optimized_df.columns:
            col_type = optimized_df[col].dtype

            if col_type != 'object':
                # Downcast integers
                if 'int' in str(col_type):
                    optimized_df[col] = pd.to_numeric(optimized_df[col], downcast='integer')

                # Downcast floats
                elif 'float' in str(col_type):
                    optimized_df[col] = pd.to_numeric(optimized_df[col], downcast='float')

            # Aggressive optimization for object columns
            elif aggressive and col_type == 'object':
                # Try to convert to category if many repeated values
                unique_ratio = len(optimized_df[col].unique()) / len(optimized_df[col])
                if unique_ratio < 0.5:  # Less than 50% unique values
                    optimized_df[col] = optimized_df[col].astype('category')

        optimized_memory = optimized_df.memory_usage(deep=True).sum()
        reduction_pct = (1 - optimized_memory / original_memory) * 100

        logger.info(f"DataFrame optimized: {reduction_pct:.1f}% memory reduction "
                   f"({original_memory / (1024**2):.1f}MB -> {optimized_memory / (1024**2):.1f}MB)")

        return optimized_df

    except Exception as e:
        logger.error(f"Error optimizing DataFrame: {e}")
        return df


def safe_divide(numerator: float, denominator: float, default: float = 0.0) -> float:
    """
    Safely divide two numbers, handling division by zero.

    Args:
        numerator: Numerator value
        denominator: Denominator value
        default: Default value to return if division by zero

    Returns:
        Division result or default value
    """
    try:
        if denominator == 0 or np.isnan(denominator) or np.isinf(denominator):
            return default

        result = numerator / denominator

        if np.isnan(result) or np.isinf(result):
            return default

        return result

    except (ZeroDivisionError, TypeError, ValueError):
        return default


def normalize_features(df: pd.DataFrame, method: str = 'standard',
                      exclude_columns: List[str] = None) -> tuple:
    """
    Normalize features in DataFrame using specified method.

    Args:
        df: DataFrame to normalize
        method: Normalization method ('standard', 'minmax', 'robust')
        exclude_columns: Columns to exclude from normalization

    Returns:
        Tuple of (normalized DataFrame, fitted scaler)
    """
    try:
        from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler

        exclude_columns = exclude_columns or []
        numeric_columns = df.select_dtypes(include=[np.number]).columns
        columns_to_normalize = [col for col in numeric_columns if col not in exclude_columns]

        if not columns_to_normalize:
            logger.warning("No numeric columns found for normalization")
            return df.copy(), None

        # Select scaler
        if method == 'standard':
            scaler = StandardScaler()
        elif method == 'minmax':
            scaler = MinMaxScaler()
        elif method == 'robust':
            scaler = RobustScaler()
        else:
            raise ValueError(f"Unknown normalization method: {method}")

        # Normalize data
        normalized_df = df.copy()
        normalized_df[columns_to_normalize] = scaler.fit_transform(df[columns_to_normalize])

        logger.info(f"Normalized {len(columns_to_normalize)} features using {method} scaling")
        return normalized_df, scaler

    except Exception as e:
        logger.error(f"Error normalizing features: {e}")
        return df.copy(), None


def handle_missing_data(df: pd.DataFrame, strategy: str = 'forward_fill',
                       threshold: float = 0.5) -> pd.DataFrame:
    """
    Handle missing data in DataFrame using specified strategy.

    Args:
        df: DataFrame with missing data
        strategy: Strategy to handle missing data ('forward_fill', 'backward_fill', 'interpolate', 'drop', 'mean')
        threshold: Threshold for dropping columns (fraction of missing values)

    Returns:
        DataFrame with missing data handled
    """
    try:
        cleaned_df = df.copy()

        # Drop columns with too many missing values
        missing_pct = cleaned_df.isnull().sum() / len(cleaned_df)
        cols_to_drop = missing_pct[missing_pct > threshold].index

        if len(cols_to_drop) > 0:
            cleaned_df = cleaned_df.drop(columns=cols_to_drop)
            logger.info(f"Dropped {len(cols_to_drop)} columns with >{threshold*100}% missing values")

        # Handle remaining missing values
        if strategy == 'forward_fill':
            cleaned_df = cleaned_df.fillna(method='ffill')
        elif strategy == 'backward_fill':
            cleaned_df = cleaned_df.fillna(method='bfill')
        elif strategy == 'interpolate':
            cleaned_df = cleaned_df.interpolate()
        elif strategy == 'drop':
            cleaned_df = cleaned_df.dropna()
        elif strategy == 'mean':
            numeric_columns = cleaned_df.select_dtypes(include=[np.number]).columns
            cleaned_df[numeric_columns] = cleaned_df[numeric_columns].fillna(cleaned_df[numeric_columns].mean())

        # Final forward fill for any remaining NaN values
        cleaned_df = cleaned_df.fillna(method='ffill').fillna(method='bfill')

        missing_after = cleaned_df.isnull().sum().sum()
        logger.info(f"Missing data handled using {strategy} strategy. Remaining missing values: {missing_after}")

        return cleaned_df

    except Exception as e:
        logger.error(f"Error handling missing data: {e}")
        return df


def detect_outliers(df: pd.DataFrame, method: str = 'iqr', threshold: float = 3.0) -> Dict[str, Any]:
    """
    Detect outliers in DataFrame using specified method.

    Args:
        df: DataFrame to analyze
        method: Outlier detection method ('iqr', 'zscore', 'isolation_forest')
        threshold: Threshold for outlier detection

    Returns:
        Dictionary containing outlier analysis results
    """
    try:
        numeric_columns = df.select_dtypes(include=[np.number]).columns
        outlier_results = {
            'method': method,
            'threshold': threshold,
            'outliers_by_column': {},
            'total_outliers': 0,
            'outlier_percentage': 0.0
        }

        total_outliers = 0

        for col in numeric_columns:
            col_data = df[col].dropna()

            if method == 'iqr':
                Q1 = col_data.quantile(0.25)
                Q3 = col_data.quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - 1.5 * IQR
                upper_bound = Q3 + 1.5 * IQR
                outliers = col_data[(col_data < lower_bound) | (col_data > upper_bound)]

            elif method == 'zscore':
                z_scores = np.abs((col_data - col_data.mean()) / col_data.std())
                outliers = col_data[z_scores > threshold]

            elif method == 'isolation_forest':
                try:
                    from sklearn.ensemble import IsolationForest
                    iso_forest = IsolationForest(contamination=0.1, random_state=42)
                    outlier_labels = iso_forest.fit_predict(col_data.values.reshape(-1, 1))
                    outliers = col_data[outlier_labels == -1]
                except ImportError:
                    logger.warning("Scikit-learn not available for isolation forest. Using IQR method.")
                    Q1 = col_data.quantile(0.25)
                    Q3 = col_data.quantile(0.75)
                    IQR = Q3 - Q1
                    lower_bound = Q1 - 1.5 * IQR
                    upper_bound = Q3 + 1.5 * IQR
                    outliers = col_data[(col_data < lower_bound) | (col_data > upper_bound)]

            outlier_count = len(outliers)
            outlier_percentage = (outlier_count / len(col_data)) * 100 if len(col_data) > 0 else 0

            outlier_results['outliers_by_column'][col] = {
                'count': outlier_count,
                'percentage': outlier_percentage,
                'indices': outliers.index.tolist()
            }

            total_outliers += outlier_count

        outlier_results['total_outliers'] = total_outliers
        outlier_results['outlier_percentage'] = (total_outliers / len(df)) * 100 if len(df) > 0 else 0

        logger.info(f"Outlier detection completed using {method} method. "
                   f"Found {total_outliers} outliers ({outlier_results['outlier_percentage']:.2f}%)")

        return outlier_results

    except Exception as e:
        logger.error(f"Error detecting outliers: {e}")
        return {'error': str(e)}


def calculate_correlation_matrix(df: pd.DataFrame, method: str = 'pearson',
                                min_periods: int = 30) -> pd.DataFrame:
    """
    Calculate correlation matrix for DataFrame features.

    Args:
        df: DataFrame to analyze
        method: Correlation method ('pearson', 'spearman', 'kendall')
        min_periods: Minimum number of observations for correlation calculation

    Returns:
        Correlation matrix DataFrame
    """
    try:
        numeric_df = df.select_dtypes(include=[np.number])

        if numeric_df.empty:
            logger.warning("No numeric columns found for correlation analysis")
            return pd.DataFrame()

        correlation_matrix = numeric_df.corr(method=method, min_periods=min_periods)

        logger.info(f"Correlation matrix calculated using {method} method for {len(numeric_df.columns)} features")
        return correlation_matrix

    except Exception as e:
        logger.error(f"Error calculating correlation matrix: {e}")
        return pd.DataFrame()


def generate_feature_importance_report(importance_data: Dict[str, float],
                                     top_n: int = 20) -> Dict[str, Any]:
    """
    Generate comprehensive feature importance report.

    Args:
        importance_data: Dictionary mapping feature names to importance scores
        top_n: Number of top features to include in report

    Returns:
        Feature importance report dictionary
    """
    try:
        if not importance_data:
            return {'error': 'No importance data provided'}

        # Sort features by importance
        sorted_features = sorted(importance_data.items(), key=lambda x: abs(x[1]), reverse=True)

        report = {
            'total_features': len(importance_data),
            'top_features': sorted_features[:top_n],
            'bottom_features': sorted_features[-min(top_n, len(sorted_features)):],
            'importance_statistics': {
                'max_importance': max(importance_data.values()),
                'min_importance': min(importance_data.values()),
                'mean_importance': np.mean(list(importance_data.values())),
                'std_importance': np.std(list(importance_data.values()))
            },
            'feature_categories': {}
        }

        # Categorize features
        categories = {
            'technical_indicators': ['sma', 'ema', 'rsi', 'macd', 'bollinger'],
            'price_features': ['open', 'high', 'low', 'close', 'price'],
            'volume_features': ['volume', 'vol'],
            'volatility_features': ['volatility', 'atr', 'std'],
            'momentum_features': ['momentum', 'roc', 'stoch'],
            'statistical_features': ['mean', 'std', 'skew', 'kurt']
        }

        for category, keywords in categories.items():
            category_features = []
            for feature, importance in sorted_features:
                if any(keyword in feature.lower() for keyword in keywords):
                    category_features.append((feature, importance))

            if category_features:
                report['feature_categories'][category] = {
                    'count': len(category_features),
                    'top_features': category_features[:5],
                    'avg_importance': np.mean([imp for _, imp in category_features])
                }

        logger.info(f"Feature importance report generated for {len(importance_data)} features")
        return report

    except Exception as e:
        logger.error(f"Error generating feature importance report: {e}")
        return {'error': str(e)}


def create_performance_summary(metrics: Dict[str, Any]) -> str:
    """
    Create human-readable performance summary from metrics.

    Args:
        metrics: Dictionary containing performance metrics

    Returns:
        Formatted performance summary string
    """
    try:
        summary_lines = ["=== PERFORMANCE SUMMARY ==="]

        # Key metrics
        if 'total_return' in metrics:
            summary_lines.append(f"Total Return: {metrics['total_return']:.2%}")

        if 'annualized_return' in metrics:
            summary_lines.append(f"Annualized Return: {metrics['annualized_return']:.2%}")

        if 'sharpe_ratio' in metrics:
            summary_lines.append(f"Sharpe Ratio: {metrics['sharpe_ratio']:.2f}")

        if 'max_drawdown' in metrics:
            summary_lines.append(f"Maximum Drawdown: {metrics['max_drawdown']:.2%}")

        if 'volatility' in metrics:
            summary_lines.append(f"Volatility: {metrics['volatility']:.2%}")

        if 'win_rate' in metrics:
            summary_lines.append(f"Win Rate: {metrics['win_rate']:.2%}")

        if 'total_trades' in metrics:
            summary_lines.append(f"Total Trades: {metrics['total_trades']}")

        # Risk assessment
        summary_lines.append("\n=== RISK ASSESSMENT ===")

        sharpe = metrics.get('sharpe_ratio', 0)
        max_dd = abs(metrics.get('max_drawdown', 0))

        if sharpe > 1.5:
            risk_level = "LOW"
        elif sharpe > 1.0:
            risk_level = "MODERATE"
        elif sharpe > 0.5:
            risk_level = "HIGH"
        else:
            risk_level = "VERY HIGH"

        summary_lines.append(f"Risk Level: {risk_level}")

        if max_dd > 0.2:
            summary_lines.append("⚠️  High drawdown detected - consider risk management improvements")

        if sharpe < 1.0:
            summary_lines.append("⚠️  Low risk-adjusted returns - consider strategy optimization")

        return "\n".join(summary_lines)

    except Exception as e:
        logger.error(f"Error creating performance summary: {e}")
        return "Error generating performance summary"


def export_results(results: Dict[str, Any], output_dir: str,
                  formats: List[str] = None) -> Dict[str, str]:
    """
    Export results to multiple formats.

    Args:
        results: Results dictionary to export
        output_dir: Output directory path
        formats: List of formats to export ('json', 'csv', 'excel', 'pickle')

    Returns:
        Dictionary mapping formats to file paths
    """
    try:
        import os
        from pathlib import Path

        formats = formats or ['json']
        Path(output_dir).mkdir(parents=True, exist_ok=True)

        exported_files = {}
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        for format_type in formats:
            if format_type == 'json':
                file_path = os.path.join(output_dir, f"results_{timestamp}.json")
                with open(file_path, 'w') as f:
                    json.dump(results, f, indent=2, default=json_serializer_default)
                exported_files['json'] = file_path

            elif format_type == 'pickle':
                file_path = os.path.join(output_dir, f"results_{timestamp}.pkl")
                import pickle
                with open(file_path, 'wb') as f:
                    pickle.dump(results, f)
                exported_files['pickle'] = file_path

            elif format_type == 'csv' and isinstance(results, dict):
                # Export flattened results to CSV
                file_path = os.path.join(output_dir, f"results_{timestamp}.csv")
                flattened = _flatten_dict(results)
                df = pd.DataFrame([flattened])
                df.to_csv(file_path, index=False)
                exported_files['csv'] = file_path

            elif format_type == 'excel' and isinstance(results, dict):
                file_path = os.path.join(output_dir, f"results_{timestamp}.xlsx")
                flattened = _flatten_dict(results)
                df = pd.DataFrame([flattened])
                df.to_excel(file_path, index=False)
                exported_files['excel'] = file_path

        logger.info(f"Results exported to {len(exported_files)} formats in {output_dir}")
        return exported_files

    except Exception as e:
        logger.error(f"Error exporting results: {e}")
        return {}


def _flatten_dict(d: Dict[str, Any], parent_key: str = '', sep: str = '_') -> Dict[str, Any]:
    """
    Flatten nested dictionary for CSV/Excel export.

    Args:
        d: Dictionary to flatten
        parent_key: Parent key for nested items
        sep: Separator for nested keys

    Returns:
        Flattened dictionary
    """
    items = []
    for k, v in d.items():
        new_key = f"{parent_key}{sep}{k}" if parent_key else k

        if isinstance(v, dict):
            items.extend(_flatten_dict(v, new_key, sep=sep).items())
        elif isinstance(v, list) and len(v) > 0 and isinstance(v[0], dict):
            # Handle list of dictionaries
            for i, item in enumerate(v):
                if isinstance(item, dict):
                    items.extend(_flatten_dict(item, f"{new_key}_{i}", sep=sep).items())
                else:
                    items.append((f"{new_key}_{i}", item))
        else:
            items.append((new_key, v))

    return dict(items)


def load_configuration(config_path: str) -> Dict[str, Any]:
    """
    Load configuration from file.

    Args:
        config_path: Path to configuration file

    Returns:
        Configuration dictionary
    """
    try:
        with open(config_path, 'r') as f:
            if config_path.endswith('.json'):
                config = json.load(f)
            elif config_path.endswith('.yaml') or config_path.endswith('.yml'):
                import yaml
                config = yaml.safe_load(f)
            else:
                raise ValueError(f"Unsupported configuration file format: {config_path}")

        logger.info(f"Configuration loaded from {config_path}")
        return config

    except Exception as e:
        logger.error(f"Error loading configuration from {config_path}: {e}")
        return {}


def save_configuration(config: Dict[str, Any], config_path: str) -> bool:
    """
    Save configuration to file.

    Args:
        config: Configuration dictionary
        config_path: Path to save configuration

    Returns:
        True if successful, False otherwise
    """
    try:
        os.makedirs(os.path.dirname(config_path), exist_ok=True)

        with open(config_path, 'w') as f:
            if config_path.endswith('.json'):
                json.dump(config, f, indent=2, default=json_serializer_default)
            elif config_path.endswith('.yaml') or config_path.endswith('.yml'):
                import yaml
                yaml.dump(config, f, default_flow_style=False)
            else:
                raise ValueError(f"Unsupported configuration file format: {config_path}")

        logger.info(f"Configuration saved to {config_path}")
        return True

    except Exception as e:
        logger.error(f"Error saving configuration to {config_path}: {e}")
        return False


def validate_file_path(file_path: str, must_exist: bool = True) -> bool:
    """
    Validate file path.

    Args:
        file_path: Path to validate
        must_exist: Whether file must exist

    Returns:
        True if valid, False otherwise
    """
    try:
        if not file_path or not isinstance(file_path, str):
            return False

        path_obj = pathlib.Path(file_path)

        if must_exist:
            return path_obj.exists() and path_obj.is_file()
        else:
            # Check if parent directory exists or can be created
            parent_dir = path_obj.parent
            return parent_dir.exists() or parent_dir.parent.exists()

    except Exception as e:
        logger.error(f"Error validating file path {file_path}: {e}")
        return False


def ensure_directory_exists(directory_path: str) -> bool:
    """
    Ensure directory exists, create if necessary.

    Args:
        directory_path: Directory path to ensure

    Returns:
        True if directory exists or was created, False otherwise
    """
    try:
        pathlib.Path(directory_path).mkdir(parents=True, exist_ok=True)
        return True

    except Exception as e:
        logger.error(f"Error ensuring directory exists {directory_path}: {e}")
        return False


def get_system_info() -> Dict[str, Any]:
    """
    Get comprehensive system information.

    Returns:
        Dictionary containing system information
    """
    try:
        import platform

        system_info = {
            'platform': platform.platform(),
            'system': platform.system(),
            'processor': platform.processor(),
            'architecture': platform.architecture(),
            'python_version': platform.python_version(),
            'cpu_count': os.cpu_count(),
            'memory': {
                'total': psutil.virtual_memory().total,
                'available': psutil.virtual_memory().available,
                'percent': psutil.virtual_memory().percent
            },
            'disk': {
                'total': psutil.disk_usage('/').total,
                'free': psutil.disk_usage('/').free,
                'percent': psutil.disk_usage('/').percent
            }
        }

        return system_info

    except Exception as e:
        logger.error(f"Error getting system info: {e}")
        return {'error': str(e)}


def monitor_resource_usage() -> Dict[str, float]:
    """
    Monitor current resource usage.

    Returns:
        Dictionary containing current resource usage
    """
    try:
        return {
            'cpu_percent': psutil.cpu_percent(interval=1),
            'memory_percent': psutil.virtual_memory().percent,
            'disk_percent': psutil.disk_usage('/').percent,
            'memory_available_gb': psutil.virtual_memory().available / (1024**3)
        }

    except Exception as e:
        logger.error(f"Error monitoring resource usage: {e}")
        return {}


def create_backup(file_path: str, backup_dir: str = None) -> Optional[str]:
    """
    Create backup of file.

    Args:
        file_path: Path to file to backup
        backup_dir: Directory to store backup (default: same directory)

    Returns:
        Path to backup file or None if failed
    """
    try:
        import shutil

        if not os.path.exists(file_path):
            logger.warning(f"File does not exist for backup: {file_path}")
            return None

        file_path_obj = pathlib.Path(file_path)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        if backup_dir:
            ensure_directory_exists(backup_dir)
            backup_path = os.path.join(backup_dir, f"{file_path_obj.stem}_{timestamp}{file_path_obj.suffix}")
        else:
            backup_path = f"{file_path_obj.stem}_{timestamp}{file_path_obj.suffix}"
            backup_path = file_path_obj.parent / backup_path

        shutil.copy2(file_path, backup_path)
        logger.info(f"Backup created: {backup_path}")

        return str(backup_path)

    except Exception as e:
        logger.error(f"Error creating backup of {file_path}: {e}")
        return None


def cleanup_temporary_files(temp_dir: str, max_age_hours: int = 24) -> int:
    """
    Clean up temporary files older than specified age.

    Args:
        temp_dir: Directory containing temporary files
        max_age_hours: Maximum age in hours before deletion

    Returns:
        Number of files deleted
    """
    try:
        if not os.path.exists(temp_dir):
            return 0

        import time

        current_time = time.time()
        max_age_seconds = max_age_hours * 3600
        files_deleted = 0

        for root, dirs, files in os.walk(temp_dir):
            for file in files:
                file_path = os.path.join(root, file)

                try:
                    file_age = current_time - os.path.getmtime(file_path)

                    if file_age > max_age_seconds:
                        os.remove(file_path)
                        files_deleted += 1

                except OSError:
                    continue  # Skip files that can't be accessed

        logger.info(f"Cleaned up {files_deleted} temporary files from {temp_dir}")
        return files_deleted

    except Exception as e:
        logger.error(f"Error cleaning up temporary files: {e}")
        return 0


def run_monte_carlo_simulation(price_series: pd.Series, n_simulations: int = 5000, n_days: int = 90) -> np.ndarray:
    """
    Generates Monte Carlo price path simulations using Geometric Brownian Motion.

    Args:
        price_series: Historical price series for parameter estimation
        n_simulations: Number of simulation paths to generate
        n_days: Number of days to simulate forward

    Returns:
        Array of simulated price paths with shape (n_days, n_simulations)
    """
    log_returns = np.log(1 + price_series.pct_change())

    u = log_returns.mean()
    var = log_returns.var()
    drift = u - (0.5 * var)
    stdev = log_returns.std()

    daily_returns = np.exp(drift + stdev * np.random.normal(0, 1, (n_days, n_simulations)))

    price_paths = np.zeros_like(daily_returns)
    price_paths[0] = price_series.iloc[-1]
    for t in range(1, n_days):
        price_paths[t] = price_paths[t - 1] * daily_returns[t]

    return price_paths


def _sanitize_ai_suggestions(suggestions: Dict[str, Any]) -> Dict[str, Any]:
    """
    Validates and sanitizes critical numeric parameters from the AI.

    Args:
        suggestions: Dictionary of AI-suggested parameter values

    Returns:
        Sanitized dictionary with values clamped to valid ranges
    """
    sanitized = suggestions.copy()
    bounds = {
        'MAX_DD_PER_CYCLE': (0.05, 0.99),
        'MAX_CONCURRENT_TRADES': (1, 20),
        'PARTIAL_PROFIT_TRIGGER_R': (0.1, 10.0),
        'PARTIAL_PROFIT_TAKE_PCT': (0.1, 0.9),
        'OPTUNA_TRIALS': (25, 200),
        'LOOKAHEAD_CANDLES': (10, 500),
        'anomaly_contamination_factor': (0.001, 0.1),
        'LABEL_LONG_QUANTILE': (0.5, 1.0),
        'LABEL_SHORT_QUANTILE': (0.0, 0.5)
    }
    integer_keys = ['MAX_CONCURRENT_TRADES', 'OPTUNA_TRIALS', 'LOOKAHEAD_CANDLES']

    for key, (lower, upper) in bounds.items():
        if key in sanitized and isinstance(sanitized.get(key), (int, float)):
            original_value = sanitized[key]
            clamped_value = max(lower, min(original_value, upper))
            if key in integer_keys:
                clamped_value = int(round(clamped_value))
            if original_value != clamped_value:
                logger.warning(f"  - Sanitizing AI suggestion for '{key}': Clamped value from {original_value} to {clamped_value} to meet model constraints.")
                sanitized[key] = clamped_value
    return sanitized


def _sanitize_frequency_string(freq_str: Any, default: str = '90D') -> str:
    """
    More robustly sanitizes a string to be a valid pandas frequency.

    Args:
        freq_str: Frequency string or value to sanitize
        default: Default frequency to use if sanitization fails

    Returns:
        Valid pandas frequency string
    """
    if isinstance(freq_str, int):
        sanitized_freq = f"{freq_str}D"
        logger.warning(f"AI provided a unit-less number for frequency. Interpreting '{freq_str}' as '{sanitized_freq}'.")
        return sanitized_freq

    if not isinstance(freq_str, str):
        freq_str = str(freq_str)
    if freq_str.isdigit():
        sanitized_freq = f"{freq_str}D"
        logger.warning(f"AI provided a unit-less number for frequency. Interpreting '{freq_str}' as '{sanitized_freq}'.")
        return sanitized_freq

    # Check if it's a valid pandas frequency
    try:
        pd.Timedelta(freq_str)
        return freq_str
    except ValueError:
        logger.warning(f"Invalid frequency string '{freq_str}'. Using default '{default}'.")
        return default


def _recursive_sanitize(obj: Any, max_depth: int = 10, current_depth: int = 0) -> Any:
    """
    Recursively sanitizes objects for JSON serialization.

    Args:
        obj: Object to sanitize
        max_depth: Maximum recursion depth
        current_depth: Current recursion depth

    Returns:
        JSON-serializable object
    """
    if current_depth > max_depth:
        return str(obj)

    if obj is None or isinstance(obj, (bool, int, float, str)):
        return obj
    elif isinstance(obj, (list, tuple)):
        return [_recursive_sanitize(item, max_depth, current_depth + 1) for item in obj]
    elif isinstance(obj, dict):
        return {str(k): _recursive_sanitize(v, max_depth, current_depth + 1) for k, v in obj.items()}
    elif hasattr(obj, '__dict__'):
        return _recursive_sanitize(obj.__dict__, max_depth, current_depth + 1)
    else:
        return str(obj)


def load_memory(champion_path: str, history_path: str) -> Dict:
    """
    Load framework memory from champion and history files.

    Args:
        champion_path: Path to champion configuration file
        history_path: Path to historical runs file

    Returns:
        Dictionary containing champion config and historical runs
    """
    import json

    champion_config = None
    if os.path.exists(champion_path):
        try:
            with open(champion_path, 'r') as f:
                champion_config = json.load(f)
        except (json.JSONDecodeError, IOError) as e:
            logger.error(f"Could not read or parse champion file at {champion_path}: {e}")

    historical_runs = []
    if os.path.exists(history_path):
        try:
            with open(history_path, 'r') as f:
                for i, line in enumerate(f):
                    if not line.strip():
                        continue
                    try:
                        historical_runs.append(json.loads(line))
                    except json.JSONDecodeError:
                        logger.warning(f"Skipping malformed line {i+1} in history file: {history_path}")
        except IOError as e:
            logger.error(f"Could not read history file at {history_path}: {e}")

    return {"champion_config": champion_config, "historical_runs": historical_runs}


def save_run_to_memory(config, new_run_summary: Dict, current_memory: Dict,
                      diagnosed_regime: str, shap_summary: Optional[pd.DataFrame] = None) -> Optional[Dict]:
    """
    Saves the completed run summary, including SHAP feature importance,
    to the history file and updates champions.

    Args:
        config: Configuration object with file paths
        new_run_summary: Summary of the completed run
        current_memory: Current framework memory
        diagnosed_regime: Diagnosed market regime
        shap_summary: Optional SHAP feature importance summary

    Returns:
        Champion configuration or None
    """
    import json
    from enum import Enum
    from datetime import datetime, date
    import pathlib

    try:
        # Add SHAP summary to the run data if available
        if shap_summary is not None and not shap_summary.empty:
            # Convert DataFrame to dict for JSON serialization
            new_run_summary['shap_summary'] = shap_summary.to_dict()['SHAP_Importance']

        sanitized_summary = _recursive_sanitize(new_run_summary)
        with open(config.HISTORY_FILE_PATH, 'a') as f:
            f.write(json.dumps(sanitized_summary) + '\n')
        logger.info(f"-> Run summary appended to history file: {config.HISTORY_FILE_PATH}")
    except IOError as e:
        logger.error(f"Could not write to history file: {e}")

    MIN_TRADES_FOR_CHAMPION = 10
    new_metrics = new_run_summary.get("final_metrics", {})
    new_mar = new_metrics.get("mar_ratio", -np.inf)
    new_trade_count = new_metrics.get("total_trades", 0)

    # Overall Champion Logic
    current_champion = current_memory.get("champion_config")
    is_new_overall_champion = False
    if new_trade_count >= MIN_TRADES_FOR_CHAMPION and new_mar >= 0:
        if current_champion is None:
            is_new_overall_champion = True
            logger.info("Setting first-ever champion.")
        else:
            current_mar = current_champion.get("final_metrics", {}).get("mar_ratio", -np.inf)
            if new_mar is not None and new_mar > current_mar:
                is_new_overall_champion = True
    else:
        logger.info(f"Current run did not qualify for Overall Champion consideration. Trades: {new_trade_count}/{MIN_TRADES_FOR_CHAMPION}, MAR: {new_mar:.2f} (must be >= 0).")

    champion_to_save = new_run_summary if is_new_overall_champion else current_champion
    if is_new_overall_champion:
        prev_champ_mar = current_champion.get("final_metrics", {}).get("mar_ratio", -np.inf) if current_champion else -np.inf
        logger.info(f"NEW OVERALL CHAMPION! Current run's MAR Ratio ({new_mar:.2f}) beats previous champion's ({prev_champ_mar:.2f}).")

    try:
        if champion_to_save:
            with open(config.CHAMPION_FILE_PATH, 'w') as f:
                json.dump(_recursive_sanitize(champion_to_save), f, indent=4)
            logger.info(f"-> Overall Champion file updated: {config.CHAMPION_FILE_PATH}")
    except (IOError, TypeError) as e:
        logger.error(f"Could not write to overall champion file: {e}")

    # Regime-Specific Champion Logic
    regime_champions = {}
    if os.path.exists(config.REGIME_CHAMPIONS_FILE_PATH):
        try:
            with open(config.REGIME_CHAMPIONS_FILE_PATH, 'r') as f:
                regime_champions = json.load(f)
        except (json.JSONDecodeError, IOError) as e:
            logger.warning(f"Could not load regime champions file for updating: {e}")

    current_regime_champion = regime_champions.get(diagnosed_regime)
    is_new_regime_champion = False
    if new_trade_count >= MIN_TRADES_FOR_CHAMPION and new_mar >= 0:
         if current_regime_champion is None or new_mar > current_regime_champion.get("final_metrics", {}).get("mar_ratio", -np.inf):
             is_new_regime_champion = True

    if is_new_regime_champion:
        regime_champions[diagnosed_regime] = new_run_summary
        prev_mar = current_regime_champion.get("final_metrics", {}).get("mar_ratio", -np.inf) if current_regime_champion else -np.inf
        logger.info(f"NEW REGIME CHAMPION for '{diagnosed_regime}'! MAR Ratio ({new_mar:.2f}) beats previous ({prev_mar:.2f}).")
        try:
            with open(config.REGIME_CHAMPIONS_FILE_PATH, 'w') as f:
                json.dump(_recursive_sanitize(regime_champions), f, indent=4)
            logger.info(f"-> Regime Champions file updated: {config.REGIME_CHAMPIONS_FILE_PATH}")
        except (IOError, TypeError) as e:
            logger.error(f"Could not write to regime champions file: {e}")

    return champion_to_save


def initialize_playbook(playbook_path: str) -> Dict:
    """
    Initializes the strategy playbook.
    Every strategy now includes a default 'selected_features' list
    to serve as a robust fallback if the AI fails to provide a list.
    """
    import json

    DEFAULT_PLAYBOOK = {
        "EvolvedRuleStrategy": {
            "description": "[EVOLUTIONARY] A meta-strategy that uses a Genetic Programmer to discover novel trading rules from scratch. The AI defines a gene pool of indicators and operators, and the GP evolves the optimal combination. This is the ultimate tool for breaking deadlocks or finding a new edge.",
            "style": "evolutionary_feature_generation",
            "complexity": "high",
            "ideal_regime": ["Any"],
        },
        "EmaCrossoverRsiFilter": {
            "description": "[DIAGNOSTIC/MOMENTUM] A simple baseline strategy. Enters on an EMA cross, filtered by a basic RSI condition.",
            "style": "momentum",
            "selected_features": ['EMA_20', 'EMA_50', 'RSI', 'ADX', 'ATR'],
            "complexity": "low",
            "ideal_regime": ["Trending"],
            "asset_class_suitability": ["Any"],
            "ideal_macro_env": ["Any"]
        },
        "MeanReversionBollinger": {
            "description": "[MEAN REVERSION] Trades mean reversion using Bollinger Bands and RSI divergence.",
            "style": "mean_reversion",
            "selected_features": ['BB_upper', 'BB_lower', 'BB_middle', 'RSI', 'MACD', 'Volume_SMA'],
            "complexity": "medium",
            "ideal_regime": ["Sideways", "Low Volatility"],
            "asset_class_suitability": ["Stocks", "Forex"],
            "ideal_macro_env": ["Low Volatility"]
        },
        "BreakoutMomentum": {
            "description": "[BREAKOUT] Captures momentum breakouts with volume confirmation.",
            "style": "breakout",
            "selected_features": ['ATR', 'Volume_Ratio', 'Price_Range', 'RSI', 'ADX'],
            "complexity": "medium",
            "ideal_regime": ["Trending", "High Volatility"],
            "asset_class_suitability": ["Stocks", "Crypto"],
            "ideal_macro_env": ["High Volatility"]
        }
    }

    if os.path.exists(playbook_path):
        try:
            with open(playbook_path, 'r') as f:
                playbook = json.load(f)
            logger.info(f"Loaded strategy playbook from {playbook_path}")
            return playbook
        except (json.JSONDecodeError, IOError) as e:
            logger.error(f"Failed to load or parse playbook file: {e}. Using in-memory default.")
            return DEFAULT_PLAYBOOK
    else:
        logger.info("No existing playbook found. Using default strategies.")
        return DEFAULT_PLAYBOOK


def load_nickname_ledger(ledger_path: str) -> Dict:
    """
    Load nickname ledger for mapping version labels to human-readable names.

    Args:
        ledger_path: Path to the nickname ledger JSON file

    Returns:
        Dictionary mapping version labels to nicknames
    """
    import json

    logger.info("-> Loading Nickname Ledger...")
    if os.path.exists(ledger_path):
        try:
            with open(ledger_path, 'r') as f:
                return json.load(f)
        except (json.JSONDecodeError, IOError) as e:
            logger.error(f"  - Could not read or parse nickname ledger. Creating a new one. Error: {e}")
    return {}


def perform_strategic_review(history: Dict, directives_path: str) -> Tuple[Dict, List[Dict]]:
    """
    Perform strategic review of historical performance and generate directives.

    Args:
        history: Historical performance data
        directives_path: Path to save strategic directives

    Returns:
        Tuple of (health_report, directives)
    """
    import json

    logger.info("--- STRATEGIC REVIEW: Analyzing long-term strategy health...")
    health_report, directives, historical_runs = {}, [], history.get("historical_runs", [])

    if len(historical_runs) < 3:
        logger.info("--- STRATEGIC REVIEW: Insufficient history for a full review.")
        return health_report, directives

    # Analyze each strategy's performance
    for name in set(run.get('strategy_name') for run in historical_runs if run.get('strategy_name')):
        strategy_runs = [run for run in historical_runs if run.get('strategy_name') == name]
        if len(strategy_runs) < 3:
            continue

        failures = sum(1 for run in strategy_runs if run.get("final_metrics", {}).get("mar_ratio", 0) < 0.1)
        total_cycles = sum(len(run.get("cycle_details", [])) for run in strategy_runs)
        breaker_trips = sum(sum(1 for c in run.get("cycle_details",[]) if c.get("Status")=="Circuit Breaker") for run in strategy_runs)

        health_report[name] = {
            "ChronicFailureRate": f"{failures/len(strategy_runs):.0%}",
            "CircuitBreakerFrequency": f"{breaker_trips/total_cycles if total_cycles>0 else 0:.0%}",
            "RunsAnalyzed": len(strategy_runs)
        }

    # Check for stagnation in recent runs
    recent_runs = historical_runs[-3:]
    if len(recent_runs) >= 3 and len(set(r.get('strategy_name') for r in recent_runs)) == 1:
        stagnant_strat_name = recent_runs[0].get('strategy_name')
        calmar_values = [r.get("final_metrics", {}).get("mar_ratio", 0) for r in recent_runs]
        if calmar_values[2] <= calmar_values[1] <= calmar_values[0]:
            if stagnant_strat_name in health_report:
                health_report[stagnant_strat_name]["StagnationWarning"] = True
            directives.append({
                "action": "FORCE_EXPLORATION",
                "strategy": stagnant_strat_name,
                "reason": f"Stagnation: No improvement over last 3 runs (MAR Ratios: {[round(c, 2) for c in calmar_values]})."
            })
            logger.warning(f"--- STRATEGIC REVIEW: Stagnation detected for '{stagnant_strat_name}'. Creating directive.")

    # Save directives
    try:
        with open(directives_path, 'w') as f:
            json.dump(directives, f, indent=4)
        logger.info(f"--- STRATEGIC REVIEW: Directives saved to {directives_path}" if directives else "--- STRATEGIC REVIEW: No new directives generated.")
    except IOError as e:
        logger.error(f"--- STRATEGIC REVIEW: Failed to write to directives file: {e}")

    if health_report:
        logger.info(f"--- STRATEGIC REVIEW: Health report generated.\n{json.dumps(health_report, indent=2)}")

    return health_report, directives


def _is_maintenance_period() -> tuple:
    """
    Checks for periods where trading should be paused for operational integrity.
    Returns a tuple of (is_maintenance, reason).
    """
    from datetime import datetime

    now = datetime.now()
    # Pause for year-end illiquidity period
    if (now.month == 12 and now.day >= 23) or (now.month == 1 and now.day <= 2):
        return True, "Year-end holiday period (low liquidity)"

    return False, ""


def _detect_surge_opportunity(df_slice: pd.DataFrame, lookback_days: int = 5, threshold: float = 2.5) -> bool:
    """
    Analyzes a recent slice of data to detect a sudden volatility spike.
    This acts as a trigger for the OPPORTUNISTIC_SURGE state.
    """
    if df_slice.empty or 'ATR' not in df_slice.columns:
        return False

    recent_data = df_slice.last(f'{lookback_days}D')
    if len(recent_data) < 20: # Ensure enough data for a meaningful average
        return False

    # Calculate the average ATR over the lookback period, excluding the most recent candle
    historical_avg_atr = recent_data['ATR'].iloc[:-1].mean()
    latest_atr = recent_data['ATR'].iloc[-1]

    # Check if the latest ATR is significantly higher than the historical average
    if latest_atr > historical_avg_atr * threshold:
        logger.info(f"Surge opportunity detected: Latest ATR ({latest_atr:.4f}) > {threshold}x historical avg ({historical_avg_atr:.4f})")
        return True

    return False


def _run_feature_learnability_test(df_train_labeled: pd.DataFrame, feature_list: list, target_col: str) -> str:
    """
    Checks the information content of features against the label using Mutual Information.
    Returns a string summary for the AI.
    """
    if target_col not in df_train_labeled.columns:
        return f"Feature Learnability: Target column '{target_col}' not found."

    valid_features = [f for f in feature_list if f in df_train_labeled.columns]
    if not valid_features:
        return "Feature Learnability: No valid features found to test."

    X = df_train_labeled[valid_features].copy()
    y = df_train_labeled[target_col].copy()

    valid_target_mask = y.notna()
    X_clean = X[valid_target_mask].fillna(0)
    y_clean = y[valid_target_mask]

    if len(y_clean) < 10:
        return "Feature Learnability: Insufficient labeled data for analysis."

    try:
        from sklearn.feature_selection import mutual_info_classif

        # Calculate mutual information scores
        mi_scores = mutual_info_classif(X_clean, y_clean, random_state=42)

        # Get top features by MI score
        feature_mi_pairs = list(zip(valid_features, mi_scores))
        feature_mi_pairs.sort(key=lambda x: x[1], reverse=True)

        top_features = feature_mi_pairs[:5]
        avg_mi_score = np.mean(mi_scores)

        # Assess learnability
        if avg_mi_score > 0.1:
            assessment = "High"
        elif avg_mi_score > 0.05:
            assessment = "Moderate"
        else:
            assessment = "Low"

        top_features_str = ", ".join([f"{feat}({score:.3f})" for feat, score in top_features])

        return f"Feature Learnability: {assessment} (Avg MI: {avg_mi_score:.3f}). Top features: {top_features_str}"

    except ImportError:
        return "Feature Learnability: scikit-learn not available for mutual information analysis."
    except Exception as e:
        return f"Feature Learnability: Error during analysis - {e}"


def _generate_pre_analysis_summary(df_train: pd.DataFrame, features: list, target: str) -> str:
    """Generates a text summary of label distribution and feature learnability."""
    label_report = _label_distribution_report(df_train, target)
    learnability_report = _run_feature_learnability_test(df_train, features, target)
    return f"{label_report}\n{learnability_report}"


def _label_distribution_report(df: pd.DataFrame, label_col: str) -> str:
    """
    Generates a report on the class balance of the labels.
    Returns a string summary for the AI.
    """
    if label_col not in df.columns:
        return f"Label Distribution: Target column '{label_col}' not found."

    counts = df[label_col].value_counts(normalize=True)
    report_dict = {str(k): f"{v:.2%}" for k, v in counts.to_dict().items()}
    return f"Label Distribution for '{label_col}': {report_dict}"


def _generate_raw_data_summary_for_ai(data_by_tf: Dict[str, pd.DataFrame], tf_roles: Dict) -> Dict:
    """Generates a data summary for AI based on raw/lightly processed data."""
    logger.info("Generating RAW data summary for AI...")
    summary = {
        "timeframes_detected": list(data_by_tf.keys()),
        "timeframe_roles": tf_roles,
        "base_timeframe": tf_roles.get('base'),
    }

    base_tf_data = data_by_tf.get(tf_roles.get('base'))
    if base_tf_data is not None and not base_tf_data.empty:
        summary["assets_detected"] = list(base_tf_data['Symbol'].unique())
        summary["date_range"] = {
            "start": str(base_tf_data.index.min()),
            "end": str(base_tf_data.index.max()),
            "total_candles": len(base_tf_data)
        }

        # Basic price statistics
        if 'Close' in base_tf_data.columns:
            summary["price_stats"] = {
                "avg_close": base_tf_data['Close'].mean(),
                "price_volatility": base_tf_data['Close'].pct_change().std(),
                "price_range": {
                    "min": base_tf_data['Close'].min(),
                    "max": base_tf_data['Close'].max()
                }
            }
    else:
        summary["assets_detected"] = []
        summary["date_range"] = {"error": "No base timeframe data available"}

    return summary


def _generate_inter_asset_correlation_summary_for_ai(data_by_tf: Dict[str, pd.DataFrame], tf_roles: Dict, top_n: int = 5) -> str:
    """Generates a text summary of inter-asset price correlations from raw data."""
    logger.info("Generating inter-asset price correlation summary for AI...")
    base_tf_name = tf_roles.get('base')
    if not base_tf_name or base_tf_name not in data_by_tf:
        return "Inter-asset correlation: Base timeframe data not available."

    base_df = data_by_tf[base_tf_name]
    if base_df.empty or 'Symbol' not in base_df.columns or 'Close' not in base_df.columns:
        return "Inter-asset correlation: Data insufficient for analysis."

    try:
        # Pivot to get price series for each symbol
        price_matrix = base_df.pivot_table(index=base_df.index, columns='Symbol', values='Close')

        if price_matrix.shape[1] < 2:
            return "Inter-asset correlation: Need at least 2 assets for correlation analysis."

        # Calculate correlation matrix
        correlation_matrix = price_matrix.corr()

        # Get top correlations (excluding self-correlations)
        correlations = []
        for i in range(len(correlation_matrix.columns)):
            for j in range(i+1, len(correlation_matrix.columns)):
                asset1 = correlation_matrix.columns[i]
                asset2 = correlation_matrix.columns[j]
                corr_value = correlation_matrix.iloc[i, j]
                if not np.isnan(corr_value):
                    correlations.append((asset1, asset2, corr_value))

        # Sort by absolute correlation value
        correlations.sort(key=lambda x: abs(x[2]), reverse=True)

        # Format top correlations
        top_correlations = correlations[:top_n]
        correlation_summary = []
        for asset1, asset2, corr in top_correlations:
            correlation_summary.append(f"{asset1}-{asset2}: {corr:.3f}")

        return f"Inter-asset correlations (top {top_n}): {', '.join(correlation_summary)}"

    except Exception as e:
        return f"Inter-asset correlation: Error during analysis - {e}"


def _diagnose_raw_market_regime(data_by_tf: Dict[str, pd.DataFrame], tf_roles: Dict) -> str:
    """Diagnoses a very basic market regime from raw data for initial AI input."""
    logger.info("Diagnosing RAW market regime (heuristic)...")
    base_tf_name = tf_roles.get('base')
    if not base_tf_name or base_tf_name not in data_by_tf:
        return "Unknown (Base timeframe data not available)"

    base_df = data_by_tf[base_tf_name].copy()
    if base_df.empty or len(base_df) < 50: # Need some data
        return "Unknown (Insufficient raw data for regime diagnosis)"

    try:
        # Calculate basic indicators for regime diagnosis
        if 'Close' not in base_df.columns:
            return "Unknown (No price data available)"

        # Calculate volatility (rolling std of returns)
        returns = base_df['Close'].pct_change().dropna()
        volatility = returns.rolling(window=20).std().iloc[-1]

        # Calculate trend strength (price vs moving average)
        ma_50 = base_df['Close'].rolling(window=50).mean()
        current_price = base_df['Close'].iloc[-1]
        trend_strength = (current_price - ma_50.iloc[-1]) / ma_50.iloc[-1] if not np.isnan(ma_50.iloc[-1]) else 0

        # Simple regime classification
        if volatility > 0.03:  # High volatility threshold
            if abs(trend_strength) > 0.1:
                regime = "High Volatility Trending"
            else:
                regime = "High Volatility Ranging"
        else:  # Low volatility
            if abs(trend_strength) > 0.05:
                regime = "Low Volatility Trending"
            else:
                regime = "Low Volatility Ranging"

        logger.info(f"Raw regime diagnosis: {regime} (Vol: {volatility:.4f}, Trend: {trend_strength:.4f})")
        return regime

    except Exception as e:
        logger.error(f"Error in raw regime diagnosis: {e}")
        return "Unknown (Error in analysis)"


def _generate_raw_data_health_report(data_by_tf: Dict[str, pd.DataFrame], tf_roles: Dict) -> Dict:
    """Generates a basic health report from raw data."""
    logger.info("Generating RAW data health report...")
    report = {"status": "OK", "issues": []}

    base_tf_name = tf_roles.get('base')
    if not base_tf_name or base_tf_name not in data_by_tf:
        report["status"] = "Error"
        report["issues"].append("Base timeframe data not available.")
        return report

    base_df = data_by_tf[base_tf_name]

    # Check data availability
    if base_df.empty:
        report["status"] = "Error"
        report["issues"].append("Base timeframe data is empty.")
        return report

    # Check required columns
    required_columns = ['Open', 'High', 'Low', 'Close', 'Volume']
    missing_columns = [col for col in required_columns if col not in base_df.columns]
    if missing_columns:
        report["issues"].append(f"Missing required columns: {missing_columns}")

    # Check for data quality issues
    if 'Close' in base_df.columns:
        null_prices = base_df['Close'].isnull().sum()
        if null_prices > 0:
            report["issues"].append(f"Found {null_prices} null price values")

        zero_prices = (base_df['Close'] <= 0).sum()
        if zero_prices > 0:
            report["issues"].append(f"Found {zero_prices} zero or negative price values")

    # Check date range
    if hasattr(base_df.index, 'to_pydatetime'):
        date_range_days = (base_df.index.max() - base_df.index.min()).days
        if date_range_days < 30:
            report["issues"].append(f"Short date range: only {date_range_days} days")

    # Check for gaps in data
    if len(base_df) < 100:
        report["issues"].append(f"Insufficient data points: only {len(base_df)} candles")

    # Set overall status
    if report["issues"]:
        report["status"] = "Warning" if len(report["issues"]) < 3 else "Error"

    return report


def _validate_and_fix_spread_config(ai_suggestions: Dict[str, Any], fallback_config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Checks the SPREAD_CONFIG from the AI. If the format is invalid or the 'default'
    key is missing, it merges the AI's valid suggestions with the framework's
    default to ensure robustness.
    """
    if 'SPREAD_CONFIG' not in ai_suggestions or not isinstance(ai_suggestions['SPREAD_CONFIG'], dict):
        # AI provided no spread config or it's the wrong type, use the fallback entirely.
        logger.warning("AI did not provide a valid SPREAD_CONFIG. Using framework defaults.")
        ai_suggestions['SPREAD_CONFIG'] = fallback_config.get('SPREAD_CONFIG', {})
        return ai_suggestions

    valid_ai_spreads = {}
    is_partially_invalid = False

    # First, filter only the valid entries from the AI's suggestion
    for symbol, spread_value in ai_suggestions['SPREAD_CONFIG'].items():
        if isinstance(spread_value, (int, float)) and 0 < spread_value <= 0.01:  # Reasonable spread range
            valid_ai_spreads[symbol] = spread_value
        else:
            logger.warning(f"AI provided invalid spread for '{symbol}': {spread_value}. Skipping.")
            is_partially_invalid = True

    # Check if 'default' key exists in the AI's suggestion
    if 'default' not in valid_ai_spreads:
        logger.warning("AI's SPREAD_CONFIG missing 'default' key. Adding framework default.")
        fallback_default = fallback_config.get('SPREAD_CONFIG', {}).get('default', 0.0001)
        valid_ai_spreads['default'] = fallback_default
        is_partially_invalid = True

    # If there were any issues, merge with fallback to ensure completeness
    if is_partially_invalid:
        fallback_spreads = fallback_config.get('SPREAD_CONFIG', {})
        for symbol, spread in fallback_spreads.items():
            if symbol not in valid_ai_spreads:
                valid_ai_spreads[symbol] = spread
        logger.info("Merged AI's valid spread suggestions with framework defaults for robustness.")

    ai_suggestions['SPREAD_CONFIG'] = valid_ai_spreads
    return ai_suggestions


def _global_parallel_processor(symbol_tuple, feature_engineer_instance, temp_dir_path, macro_data):
    """
    A global, top-level function for multiprocessing to prevent pickling errors.
    It processes a single symbol's data and saves the result to a file.
    """
    import logging
    import pathlib

    # Get the logger instance for this worker process
    worker_logger = logging.getLogger("ML_Trading_Framework")

    symbol, symbol_data_by_tf = symbol_tuple
    worker_logger.info(f"  - Starting parallel processing for symbol: {symbol}...")

    # Call the instance method on the passed instance
    processed_df = feature_engineer_instance._process_single_symbol_stack(symbol_data_by_tf, macro_data)

    if processed_df is not None and not processed_df.empty:
        # Construct the output path using pathlib for robustness
        output_path = pathlib.Path(temp_dir_path) / f"{symbol}_processed.parquet"

        try:
            # Save to parquet for efficiency
            processed_df.to_parquet(output_path, index=True)
            worker_logger.info(f"  - Saved processed data for {symbol} to {output_path}")
            return str(output_path)
        except Exception as e:
            worker_logger.error(f"  - Failed to save processed data for {symbol}: {e}")
            return None
    else:
        worker_logger.warning(f"  - No processed data for symbol: {symbol}")
        return None


def _generate_cache_metadata(config, files: List[str], tf_roles: Dict, feature_engineer_class: type) -> Dict:
    """
    Generates a dictionary of metadata to validate the feature cache.
    This includes parameters that affect the output of feature engineering.
    """
    import hashlib
    import inspect

    file_metadata = {}
    for filename in sorted(files):
        file_path = os.path.join(config.BASE_PATH, filename)
        if os.path.exists(file_path):
            stat = os.stat(file_path)
            file_metadata[filename] = {"mtime": stat.st_mtime, "size": stat.st_size}

    script_hash = ""
    try:
        fe_source_code = inspect.getsource(feature_engineer_class)
        script_hash = hashlib.sha256(fe_source_code.encode('utf-8')).hexdigest()
    except Exception as e:
        logger.warning(f"Could not generate FeatureEngineer class hash: {e}")

    # Create a dictionary of ONLY the parameters that affect feature creation
    feature_params = {
        'script_sha256_hash': script_hash,
        'DYNAMIC_INDICATOR_PARAMS': getattr(config, 'DYNAMIC_INDICATOR_PARAMS', {}).copy(),
        'USE_PCA_REDUCTION': getattr(config, 'USE_PCA_REDUCTION', False),
        'PCA_N_COMPONENTS': getattr(config, 'PCA_N_COMPONENTS', 50),
        'RSI_PERIODS_FOR_PCA': getattr(config, 'RSI_PERIODS_FOR_PCA', [14, 21, 28]),
        'ADX_THRESHOLD_TREND': getattr(config, 'ADX_THRESHOLD_TREND', 25),
        'DISPLACEMENT_PERIOD': getattr(config, 'DISPLACEMENT_PERIOD', 1),
        'GAP_DETECTION_LOOKBACK': getattr(config, 'GAP_DETECTION_LOOKBACK', 5),
        'PARKINSON_VOLATILITY_WINDOW': getattr(config, 'PARKINSON_VOLATILITY_WINDOW', 20),
        'YANG_ZHANG_VOLATILITY_WINDOW': getattr(config, 'YANG_ZHANG_VOLATILITY_WINDOW', 20),
        'KAMA_REGIME_FAST': getattr(config, 'KAMA_REGIME_FAST', 10),
        'KAMA_REGIME_SLOW': getattr(config, 'KAMA_REGIME_SLOW', 30),
        'AUTOCORR_LAG': getattr(config, 'AUTOCORR_LAG', 5),
        'HURST_EXPONENT_WINDOW': getattr(config, 'HURST_EXPONENT_WINDOW', 100),
        'HAWKES_KAPPA': getattr(config, 'HAWKES_KAPPA', 0.1),
        'anomaly_contamination_factor': getattr(config, 'anomaly_contamination_factor', 0.05)
    }
    return {"files": file_metadata, "params": feature_params}


def deep_merge_dicts(original: dict, updates: dict) -> dict:
    """
    Recursively merges two dictionaries, with updates taking precedence.

    Args:
        original: The original dictionary
        updates: Dictionary with updates to merge

    Returns:
        Merged dictionary
    """
    result = original.copy()

    for key, value in updates.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = deep_merge_dicts(result[key], value)
        else:
            result[key] = value

    return result


def get_walk_forward_periods(start_date: pd.Timestamp, end_date: pd.Timestamp,
                           train_window: str, retrain_freq: str, gap: str) -> Tuple[List, List, List, List]:
    """
    Calculate walk-forward analysis periods.

    Args:
        start_date: Start date for analysis
        end_date: End date for analysis
        train_window: Training window size (e.g., '365D')
        retrain_freq: Retraining frequency (e.g., '90D')
        gap: Gap between train and test (e.g., '1D')

    Returns:
        Tuple of (train_start_dates, train_end_dates, test_start_dates, test_end_dates)
    """
    train_start_dates, train_end_dates = [], []
    test_start_dates, test_end_dates = [], []

    current_start = start_date
    train_window_td = pd.Timedelta(train_window)
    retrain_freq_td = pd.Timedelta(retrain_freq)
    gap_td = pd.Timedelta(gap)

    while current_start + train_window_td + gap_td < end_date:
        # Training period
        train_end = current_start + train_window_td
        train_start_dates.append(current_start)
        train_end_dates.append(train_end)

        # Test period
        test_start = train_end + gap_td
        test_end = min(test_start + retrain_freq_td, end_date)
        test_start_dates.append(test_start)
        test_end_dates.append(test_end)

        # Move to next period
        current_start += retrain_freq_td

    return train_start_dates, train_end_dates, test_start_dates, test_end_dates


def _generate_nickname(strategy: str, ai_suggestion: Optional[str], ledger: Dict,
                      ledger_path: str, version: str) -> str:
    """
    Generates a unique, memorable nickname for the run.

    Args:
        strategy: Strategy name
        ai_suggestion: AI suggestion text
        ledger: Nickname ledger
        ledger_path: Path to ledger file
        version: Framework version

    Returns:
        Unique nickname
    """
    import random

    # Base adjectives and nouns for nickname generation
    adjectives = ["Swift", "Bold", "Sharp", "Keen", "Wise", "Quick", "Smart", "Agile", "Bright", "Fast"]
    nouns = ["Eagle", "Falcon", "Tiger", "Wolf", "Lion", "Hawk", "Fox", "Bear", "Shark", "Panther"]

    # Generate base nickname
    base_nickname = f"{random.choice(adjectives)}{random.choice(nouns)}"

    # Ensure uniqueness
    counter = 1
    nickname = base_nickname
    while nickname in ledger.get("used_nicknames", []):
        nickname = f"{base_nickname}{counter}"
        counter += 1

    # Update ledger
    if "used_nicknames" not in ledger:
        ledger["used_nicknames"] = []
    ledger["used_nicknames"].append(nickname)

    # Save updated ledger
    try:
        with open(ledger_path, 'w') as f:
            json.dump(ledger, f, indent=2)
    except Exception as e:
        logger.warning(f"Could not save nickname ledger: {e}")

    return nickname


def _log_config_and_environment(config) -> None:
    """
    Logs key configuration parameters and system environment details.

    Args:
        config: Configuration object to log
    """
    logger = logging.getLogger("ML_Trading_Framework")

    logger.info("=== CONFIGURATION AND ENVIRONMENT ===")

    # Log key configuration parameters
    key_params = [
        'INITIAL_CAPITAL', 'TRAINING_WINDOW', 'RETRAINING_FREQUENCY',
        'FORWARD_TEST_GAP', 'OPTUNA_TRIALS', 'FEATURE_SELECTION_METHOD',
        'CALCULATE_SHAP_VALUES', 'USE_FEATURE_CACHING'
    ]

    for param in key_params:
        if hasattr(config, param):
            value = getattr(config, param)
            logger.info(f"  {param}: {value}")

    # Log system information
    import platform
    import psutil

    logger.info(f"  Python Version: {platform.python_version()}")
    logger.info(f"  Platform: {platform.platform()}")
    logger.info(f"  CPU Count: {psutil.cpu_count()}")
    logger.info(f"  Available Memory: {psutil.virtual_memory().total / (1024**3):.1f} GB")

    logger.info("=" * 40)
