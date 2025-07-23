# =============================================================================
# MODEL TRAINER MODULE
# =============================================================================

import os
import sys
import logging
import warnings
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any, Union
import optuna
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.feature_selection import SelectKBest, f_classif, mutual_info_classif
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.metrics import (
    f1_score, accuracy_score, precision_score, recall_score, 
    classification_report, confusion_matrix, mean_squared_error, r2_score
)
from sklearn.utils.class_weight import compute_class_weight
import xgboost as xgb

# Optional imports with fallbacks
try:
    import shap
    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False
    warnings.warn("SHAP not available. Feature importance analysis will be limited.")

try:
    from sktime.transformations.panel.rocket import MiniRocket
    MINIROCKET_AVAILABLE = True
except ImportError:
    MINIROCKET_AVAILABLE = False
    warnings.warn("MiniRocket not available. Time series features will be limited.")

from .config import ConfigModel
from .ai_analyzer import GeminiAnalyzer

logger = logging.getLogger(__name__)

# Suppress optuna logging
optuna.logging.set_verbosity(optuna.logging.WARNING)

class ModelTrainer:
    """
    Comprehensive model training class with hyperparameter optimization,
    ensemble learning, and advanced validation techniques.
    """
    
    def __init__(self, config: ConfigModel, gemini_analyzer: GeminiAnalyzer):
        """Initialize the ModelTrainer with configuration and AI analyzer."""
        self.config = config
        self.gemini_analyzer = gemini_analyzer
        self.shap_summaries: Dict[str, Optional[pd.DataFrame]] = {}
        self.class_weights: Optional[Dict] = None
        self.classification_report_str: str = "N/A"
        self.is_minirocket_model = 'MiniRocket' in getattr(config, 'strategy_name', '')
        self.study: Optional[optuna.study.Study] = None

    def train_and_validate_model(self, labeled_data: pd.DataFrame, 
                                feature_list: Optional[List[str]] = None) -> Dict[str, Any]:
        """
        Main entry point for training and validating a model.
        
        Args:
            labeled_data: DataFrame with features and target columns
            feature_list: Optional list of features to use
            
        Returns:
            Dictionary containing trained model and performance metrics
        """
        logger.info("Starting model training and validation...")
        
        # Identify target columns
        target_columns = [col for col in labeled_data.columns if col.startswith('target_')]
        if not target_columns:
            logger.error("No target columns found in data")
            return {"error": "No target columns found"}
        
        # Use primary target for single model training
        primary_target = 'target_signal_pressure_class'
        if primary_target not in target_columns:
            primary_target = target_columns[0]
            
        logger.info(f"Using target column: {primary_target}")
        
        # Prepare feature list
        if feature_list is None:
            feature_list = self._get_available_features_from_df(labeled_data)
        
        logger.info(f"Training with {len(feature_list)} features")
        
        # Train single model
        pipeline, best_f1, selected_features, failure_reason = self.train_single_model(
            df_train=labeled_data,
            feature_list=feature_list,
            target_col=primary_target,
            model_type='classification',
            task_name='primary_model'
        )
        
        if pipeline is None:
            return {"error": f"Model training failed: {failure_reason}"}
        
        # Calculate additional metrics
        X_test = labeled_data[selected_features].fillna(0)
        y_test = labeled_data[primary_target]
        
        predictions = pipeline.predict(X_test)
        probabilities = pipeline.predict_proba(X_test) if hasattr(pipeline, 'predict_proba') else None
        
        # Performance metrics
        metrics = {
            'f1_score': f1_score(y_test, predictions, average='weighted', zero_division=0),
            'accuracy': accuracy_score(y_test, predictions),
            'precision': precision_score(y_test, predictions, average='weighted', zero_division=0),
            'recall': recall_score(y_test, predictions, average='weighted', zero_division=0)
        }
        
        # Generate classification report
        class_report = classification_report(y_test, predictions, zero_division=0)
        
        # Feature importance analysis
        feature_importance = self._calculate_feature_importance(pipeline, selected_features)
        
        # SHAP analysis if available
        shap_summary = None
        if SHAP_AVAILABLE and len(X_test) > 0:
            try:
                shap_summary = self._calculate_shap_values(pipeline, X_test, selected_features)
            except Exception as e:
                logger.warning(f"SHAP analysis failed: {e}")
        
        results = {
            'model': pipeline,
            'metrics': metrics,
            'selected_features': selected_features,
            'feature_importance': feature_importance,
            'classification_report': class_report,
            'shap_summary': shap_summary,
            'predictions': predictions,
            'probabilities': probabilities.tolist() if probabilities is not None else None,
            'target_column': primary_target
        }
        
        logger.info(f"Model training completed. F1 Score: {metrics['f1_score']:.3f}")
        return results

    def train_single_model(self, df_train: pd.DataFrame, feature_list: List[str], 
                          target_col: str, model_type: str, task_name: str,
                          framework_history: Optional[Dict] = None,
                          search_space: Optional[Dict] = None,
                          config_override: Optional[ConfigModel] = None,
                          regime: str = "Unknown") -> Tuple[Optional[Pipeline], Optional[float], Optional[List[str]], Optional[str]]:
        """
        Train a single XGBoost model with hyperparameter optimization.
        
        Returns:
            Tuple of (pipeline, best_f1_score, selected_features, failure_reason)
        """
        original_config = self.config
        if config_override:
            self.config = config_override
            
        try:
            logger.info(f"Training model for task: {task_name}")
            
            # Prepare data
            X = df_train[feature_list].copy()
            y = df_train[target_col].copy()
            
            # Handle missing values
            X.fillna(0, inplace=True)
            y.fillna(0, inplace=True)
            
            # Remove rows with invalid targets
            valid_mask = ~(y.isna() | np.isinf(y))
            X = X[valid_mask]
            y = y[valid_mask]
            
            if len(X) == 0:
                return None, None, None, "NO_VALID_DATA"
            
            # Check class distribution
            if model_type == 'classification':
                unique_classes = y.unique()
                if len(unique_classes) < 2:
                    return None, None, None, "INSUFFICIENT_CLASSES"
                
                # Calculate class weights
                self.class_weights = self._calculate_class_weights(y)
            
            # Feature selection
            logger.info(f"  - Stage 1: Feature selection from {len(feature_list)} features...")
            selected_features = self._select_features(X, y, model_type)
            
            if len(selected_features) == 0:
                return None, None, None, "NO_FEATURES_SELECTED"
            
            logger.info(f"  - Selected {len(selected_features)} features")
            
            # Prepare data for optimization
            X_selected = X[selected_features]
            
            # Sample data if too large for optimization
            if len(X_selected) > 10000:
                X_sample, _, y_sample, _ = train_test_split(
                    X_selected, y, train_size=10000, stratify=y if model_type == 'classification' else None,
                    random_state=42
                )
            else:
                X_sample, y_sample = X_selected, y
            
            # Hyperparameter optimization
            logger.info(f"  - Stage 2: Hyperparameter optimization...")
            study = self._optimize_hyperparameters(X_sample, y_sample, model_type, task_name, search_space)
            
            if not study or not study.best_trials:
                return None, None, None, "OPTUNA_FAILED"
            
            # Get best parameters
            best_trial = study.best_trials[0] if study.best_trials else None
            if best_trial is None:
                return None, None, None, "NO_BEST_TRIAL"
            
            best_params = best_trial.params
            
            # Find best threshold for classification
            if model_type == 'classification':
                logger.info(f"  - Stage 3: Finding optimal threshold...")
                best_threshold, f1_at_best_thresh = self._find_best_threshold(best_params, X_sample, y_sample)
                
                # Check if F1 score meets minimum requirements
                min_f1_gate = getattr(self.config, 'MIN_F1_SCORE_GATE', 0.3)
                if f1_at_best_thresh < min_f1_gate:
                    logger.warning(f"  - F1 score {f1_at_best_thresh:.3f} below threshold {min_f1_gate:.3f}")
                    return None, None, None, "F1_SCORE_TOO_LOW"
            else:
                best_threshold = 0.5
                f1_at_best_thresh = 0.0
            
            # Train final model
            logger.info(f"  - Stage 4: Training final model...")
            final_pipeline = self._train_final_model(
                best_params=best_params,
                X_input=X_selected,
                y_input=y,
                model_type=model_type,
                task_name=task_name
            )
            
            if final_pipeline is None:
                return None, None, None, "FINAL_TRAINING_FAILED"
            
            # Validate on shadow set if enabled
            if hasattr(self.config, 'SHADOW_SET_VALIDATION') and self.config.SHADOW_SET_VALIDATION:
                # Create shadow set (last 20% of data)
                split_idx = int(len(df_train) * 0.8)
                df_shadow = df_train.iloc[split_idx:].copy()
                
                if not self._validate_on_shadow_set(final_pipeline, df_shadow, selected_features, model_type, target_col):
                    logger.warning("Shadow set validation failed")
            
            # Calculate SHAP values if enabled
            if getattr(self.config, 'CALCULATE_SHAP_VALUES', True) and SHAP_AVAILABLE:
                try:
                    shap_summary = self._calculate_shap_values(final_pipeline, X_sample, selected_features)
                    self.shap_summaries[task_name] = shap_summary
                except Exception as e:
                    logger.warning(f"SHAP calculation failed: {e}")
            
            logger.info(f"  - Model training completed successfully for {task_name}")
            return final_pipeline, f1_at_best_thresh, selected_features, None
            
        except Exception as e:
            logger.error(f"Error training model for {task_name}: {e}", exc_info=True)
            return None, None, None, f"TRAINING_ERROR: {str(e)}"
        finally:
            self.config = original_config

    def _get_available_features_from_df(self, df: pd.DataFrame) -> List[str]:
        """Get list of available features from DataFrame."""
        excluded_cols = {
            'Open', 'High', 'Low', 'Close', 'RealVolume', 'Volume', 'Symbol', 'Timestamp',
            'timestamp', 'symbol'
        }
        
        # Exclude target columns
        target_cols = {col for col in df.columns if col.startswith('target_')}
        excluded_cols.update(target_cols)
        
        # Get numeric columns that aren't excluded
        feature_cols = []
        for col in df.columns:
            if col not in excluded_cols and pd.api.types.is_numeric_dtype(df[col]):
                feature_cols.append(col)
        
        return feature_cols

    def _select_features(self, X: pd.DataFrame, y: pd.Series, model_type: str) -> List[str]:
        """Select best features using the configured method."""
        method = getattr(self.config, 'FEATURE_SELECTION_METHOD', 'mutual_info')
        n_features = min(50, len(X.columns))  # Select top 50 features or all if less
        
        try:
            if method == 'mutual_info':
                if model_type == 'classification':
                    selector = SelectKBest(mutual_info_classif, k=n_features)
                else:
                    from sklearn.feature_selection import mutual_info_regression
                    selector = SelectKBest(mutual_info_regression, k=n_features)
            else:
                # Default to f_classif
                selector = SelectKBest(f_classif, k=n_features)
            
            selector.fit(X, y)
            selected_features = X.columns[selector.get_support()].tolist()
            
            logger.info(f"Feature selection: {len(selected_features)} features selected using {method}")
            return selected_features
            
        except Exception as e:
            logger.warning(f"Feature selection failed: {e}. Using all features.")
            return X.columns.tolist()

    def _calculate_class_weights(self, y: pd.Series) -> Dict:
        """Calculate class weights for imbalanced datasets."""
        try:
            classes = np.unique(y)
            weights = compute_class_weight('balanced', classes=classes, y=y)
            class_weights = dict(zip(classes, weights))
            logger.info(f"Calculated class weights: {class_weights}")
            return class_weights
        except Exception as e:
            logger.warning(f"Could not calculate class weights: {e}")
            return {}

    def _optimize_hyperparameters(self, X: pd.DataFrame, y: pd.Series, model_type: str, 
                                 task_name: str, search_space: Optional[Dict] = None) -> Optional[optuna.study.Study]:
        """Optimize hyperparameters using Optuna."""
        
        def objective(trial):
            # Define hyperparameter search space
            if search_space:
                # Use AI-defined search space if provided
                params = {}
                for param, config in search_space.items():
                    if param in ['n_estimators', 'max_depth', 'min_child_weight']:
                        params[param] = trial.suggest_int(param, config.get('min', 50), config.get('max', 300))
                    elif param in ['learning_rate', 'subsample', 'colsample_bytree', 'reg_alpha', 'reg_lambda', 'gamma']:
                        params[param] = trial.suggest_float(param, config.get('min', 0.01), config.get('max', 1.0))
            else:
                # Default search space
                params = {
                    'n_estimators': trial.suggest_int('n_estimators', 50, 300),
                    'max_depth': trial.suggest_int('max_depth', 3, 10),
                    'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3),
                    'subsample': trial.suggest_float('subsample', 0.6, 1.0),
                    'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
                    'reg_alpha': trial.suggest_float('reg_alpha', 0, 10),
                    'reg_lambda': trial.suggest_float('reg_lambda', 0, 10),
                    'gamma': trial.suggest_float('gamma', 0, 5),
                    'min_child_weight': trial.suggest_int('min_child_weight', 1, 10)
                }
            
            # Add class weights for classification
            if model_type == 'classification' and self.class_weights:
                params['scale_pos_weight'] = list(self.class_weights.values())[1] / list(self.class_weights.values())[0] if len(self.class_weights) == 2 else 1
            
            params['random_state'] = 42
            params['n_jobs'] = 1
            
            # Create and evaluate model
            if model_type == 'classification':
                model = xgb.XGBClassifier(**params)
                cv_scores = cross_val_score(model, X, y, cv=3, scoring='f1_weighted', n_jobs=1)
            else:
                model = xgb.XGBRegressor(**params)
                cv_scores = cross_val_score(model, X, y, cv=3, scoring='neg_mean_squared_error', n_jobs=1)
            
            return cv_scores.mean()
        
        try:
            study = optuna.create_study(
                direction='maximize',
                pruner=optuna.pruners.HyperbandPruner(),
                study_name=task_name
            )
            
            study.optimize(
                objective,
                n_trials=getattr(self.config, 'OPTUNA_TRIALS', 50),
                timeout=1800,  # 30 minutes max
                n_jobs=getattr(self.config, 'OPTUNA_N_JOBS', 1),
                callbacks=[self._log_optuna_trial]
            )
            
            logger.info(f"Optuna optimization completed. Best score: {study.best_value:.4f}")
            return study
            
        except Exception as e:
            logger.error(f"Optuna optimization failed: {e}", exc_info=True)
            return None

    def _log_optuna_trial(self, study: optuna.study.Study, trial: optuna.trial.FrozenTrial):
        """Callback to log Optuna trial progress."""
        total_trials = getattr(self.config, 'OPTUNA_TRIALS', 50)
        message = f"\r  - Optimizing [{study.study_name}]: Trial {trial.number + 1}/{total_trials}"
        
        try:
            if study.best_trials:
                best_value = study.best_trials[0].value
                message += f" | Best Score: {best_value:.4f}"
        except Exception:
            message += " | Running..."
        
        sys.stdout.write(message.ljust(100))
        sys.stdout.flush()

    def _find_best_threshold(self, best_params: Dict, X: pd.DataFrame, y: pd.Series) -> Tuple[float, float]:
        """Find optimal classification threshold."""
        try:
            # Train model with best parameters
            model_params = {k: v for k, v in best_params.items() if k not in ['sl_multiplier', 'tp_multiplier']}
            model_params['random_state'] = 42
            
            if self.class_weights:
                model_params['scale_pos_weight'] = list(self.class_weights.values())[1] / list(self.class_weights.values())[0] if len(self.class_weights) == 2 else 1
            
            model = xgb.XGBClassifier(**model_params)
            
            # Use cross-validation to find best threshold
            skf = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
            best_threshold = 0.5
            best_f1 = 0.0
            
            for train_idx, val_idx in skf.split(X, y):
                X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
                y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
                
                model.fit(X_train, y_train)
                y_proba = model.predict_proba(X_val)
                
                # Try different thresholds
                for threshold in np.arange(0.3, 0.8, 0.05):
                    if y_proba.shape[1] > 1:
                        y_pred = (y_proba[:, 1] > threshold).astype(int)
                    else:
                        y_pred = (y_proba > threshold).astype(int)
                    
                    f1 = f1_score(y_val, y_pred, average='weighted', zero_division=0)
                    if f1 > best_f1:
                        best_f1 = f1
                        best_threshold = threshold
            
            return best_threshold, best_f1
            
        except Exception as e:
            logger.warning(f"Threshold optimization failed: {e}")
            return 0.5, 0.0

    def _train_final_model(self, best_params: Dict, X_input: pd.DataFrame, y_input: pd.Series,
                          model_type: str, task_name: str, pre_fitted_transformer: Optional[Pipeline] = None) -> Optional[Pipeline]:
        """Train the final model with best parameters."""
        try:
            # Prepare parameters
            model_params = {k: v for k, v in best_params.items() if k not in ['sl_multiplier', 'tp_multiplier']}
            model_params['random_state'] = 42
            model_params['n_jobs'] = 1
            
            if model_type == 'classification' and self.class_weights:
                model_params['scale_pos_weight'] = list(self.class_weights.values())[1] / list(self.class_weights.values())[0] if len(self.class_weights) == 2 else 1
            
            # Create preprocessing pipeline
            if pre_fitted_transformer:
                preprocessor = pre_fitted_transformer
            else:
                preprocessor = Pipeline([
                    ('scaler', RobustScaler()),
                ])
            
            # Create model
            if model_type == 'classification':
                model = xgb.XGBClassifier(**model_params)
            else:
                model = xgb.XGBRegressor(**model_params)
            
            # Create full pipeline
            if pre_fitted_transformer:
                pipeline = Pipeline([
                    ('preprocessor', preprocessor),
                    ('model', model)
                ])
            else:
                pipeline = Pipeline([
                    ('scaler', RobustScaler()),
                    ('model', model)
                ])
            
            # Train pipeline
            pipeline.fit(X_input, y_input)
            
            logger.info(f"Final model training completed for {task_name}")
            return pipeline
            
        except Exception as e:
            logger.error(f"Final model training failed: {e}", exc_info=True)
            return None

    def _validate_on_shadow_set(self, pipeline: Pipeline, df_shadow: pd.DataFrame, 
                               feature_list: List[str], model_type: str, target_col: str) -> bool:
        """Validate model on shadow holdout set."""
        if not getattr(self.config, 'SHADOW_SET_VALIDATION', True):
            return True
        
        if df_shadow.empty:
            logger.warning("Shadow set is empty. Skipping validation.")
            return True
        
        logger.info(f"Validating on shadow set ({len(df_shadow)} rows)...")
        
        try:
            X_shadow = df_shadow[feature_list].copy()
            y_shadow = df_shadow[target_col].copy()
            X_shadow.fillna(0, inplace=True)
            
            if model_type == 'classification':
                preds = pipeline.predict(X_shadow)
                score = f1_score(y_shadow, preds, average='weighted', zero_division=0)
                threshold = getattr(self.config, 'MIN_F1_SCORE_GATE', 0.3) * 0.8
                
                if score >= threshold:
                    logger.info(f"Shadow validation PASSED. F1: {score:.3f} (>= {threshold:.3f})")
                    return True
                else:
                    logger.warning(f"Shadow validation FAILED. F1: {score:.3f} (< {threshold:.3f})")
                    return False
            else:
                preds = pipeline.predict(X_shadow)
                score = mean_squared_error(y_shadow, preds, squared=False)
                threshold = y_shadow.std() * 2.0
                
                if score <= threshold:
                    logger.info(f"Shadow validation PASSED. RMSE: {score:.3f} (<= {threshold:.3f})")
                    return True
                else:
                    logger.warning(f"Shadow validation FAILED. RMSE: {score:.3f} (> {threshold:.3f})")
                    return False
                    
        except Exception as e:
            logger.error(f"Shadow validation error: {e}", exc_info=True)
            return False

    def _calculate_feature_importance(self, pipeline: Pipeline, feature_names: List[str]) -> Dict[str, float]:
        """Calculate feature importance from trained model."""
        try:
            # Get the model from pipeline
            model = None
            for step_name, step in pipeline.steps:
                if hasattr(step, 'feature_importances_'):
                    model = step
                    break
            
            if model is None:
                return {}
            
            importance_scores = model.feature_importances_
            feature_importance = dict(zip(feature_names, importance_scores))
            
            # Sort by importance
            sorted_importance = dict(sorted(feature_importance.items(), key=lambda x: x[1], reverse=True))
            
            return sorted_importance
            
        except Exception as e:
            logger.warning(f"Feature importance calculation failed: {e}")
            return {}

    def _calculate_shap_values(self, pipeline: Pipeline, X: pd.DataFrame, feature_names: List[str]) -> Optional[pd.DataFrame]:
        """Calculate SHAP values for feature importance analysis."""
        if not SHAP_AVAILABLE:
            return None
        
        try:
            # Get the model from pipeline
            model = None
            for step_name, step in pipeline.steps:
                if hasattr(step, 'predict'):
                    model = step
                    break
            
            if model is None:
                return None
            
            # Transform data through preprocessing steps
            X_transformed = X.copy()
            for step_name, step in pipeline.steps[:-1]:  # All steps except the model
                X_transformed = step.transform(X_transformed)
            
            # Calculate SHAP values
            explainer = shap.Explainer(model, X_transformed)
            shap_values = explainer(X_transformed)
            
            # Create summary DataFrame
            if hasattr(shap_values, 'values'):
                shap_df = pd.DataFrame(
                    shap_values.values,
                    columns=feature_names
                )
            else:
                shap_df = pd.DataFrame(
                    shap_values,
                    columns=feature_names
                )
            
            # Calculate mean absolute SHAP values
            mean_shap = shap_df.abs().mean().sort_values(ascending=False)
            
            shap_summary = pd.DataFrame({
                'feature': mean_shap.index,
                'mean_abs_shap': mean_shap.values
            })
            
            logger.info(f"SHAP analysis completed for {len(feature_names)} features")
            return shap_summary
            
        except Exception as e:
            logger.warning(f"SHAP calculation failed: {e}")
            return None
