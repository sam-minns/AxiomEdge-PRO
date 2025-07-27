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

try:
    from sklearn.decomposition import PCA
    from sklearn.feature_selection import SelectFromModel
    from sklearn.ensemble import RandomForestClassifier
    ADVANCED_SELECTION_AVAILABLE = True
except ImportError:
    ADVANCED_SELECTION_AVAILABLE = False

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from torch_geometric.nn import GCNConv, GATConv, GraphConv
    from torch_geometric.data import Data, DataLoader as GeometricDataLoader
    from torch_geometric.utils import to_networkx
    import networkx as nx
    TORCH_GEOMETRIC_AVAILABLE = True
except ImportError:
    TORCH_GEOMETRIC_AVAILABLE = False
    warnings.warn("PyTorch Geometric not available. Graph Neural Network models will be disabled.")

import json
from datetime import datetime

from .config import ConfigModel
from .ai_analyzer import GeminiAnalyzer

logger = logging.getLogger(__name__)

# Suppress optuna logging
optuna.logging.set_verbosity(optuna.logging.WARNING)


class GNNModel:
    """
    Graph Neural Network model for financial time series prediction.

    This class implements Graph Neural Networks (GNNs) for capturing complex
    relationships between financial features and market dynamics. Uses PyTorch
    Geometric for graph-based learning with proper fallback handling.

    Features:
    - Graph Convolutional Networks (GCN) for feature relationships
    - Graph Attention Networks (GAT) for adaptive feature weighting
    - Dynamic graph construction from correlation matrices
    - Temporal graph evolution for time series modeling
    - Multi-layer graph architectures with residual connections
    - Proper handling of missing torch_geometric dependency

    Note: Requires torch_geometric to be installed. If not available,
    this class will provide graceful fallback behavior.
    """

    def __init__(self, config: ConfigModel):
        """
        Initialize GNN model with configuration.

        Args:
            config: Configuration object with GNN parameters
        """
        self.config = config
        self.model = None
        self.graph_data = None
        self.is_available = TORCH_GEOMETRIC_AVAILABLE

        # GNN configuration parameters
        self.hidden_dim = getattr(config, 'GNN_HIDDEN_DIM', 64)
        self.num_layers = getattr(config, 'GNN_NUM_LAYERS', 3)
        self.dropout_rate = getattr(config, 'GNN_DROPOUT_RATE', 0.2)
        self.learning_rate = getattr(config, 'GNN_LEARNING_RATE', 0.001)
        self.num_epochs = getattr(config, 'GNN_NUM_EPOCHS', 100)
        self.correlation_threshold = getattr(config, 'GNN_CORRELATION_THRESHOLD', 0.3)
        self.model_type = getattr(config, 'GNN_MODEL_TYPE', 'GCN')  # GCN, GAT, or GraphConv

        if not self.is_available:
            logger.warning("PyTorch Geometric not available. GNN functionality disabled.")

    def is_gnn_available(self) -> bool:
        """Check if GNN functionality is available."""
        return self.is_available

    def create_graph_from_features(self, features_df: pd.DataFrame) -> Optional[Any]:
        """
        Create graph structure from feature correlation matrix.

        Args:
            features_df: DataFrame containing features for graph construction

        Returns:
            PyTorch Geometric Data object or None if not available
        """
        if not self.is_available:
            logger.warning("Cannot create graph: PyTorch Geometric not available")
            return None

        try:
            # Calculate correlation matrix
            correlation_matrix = features_df.corr().abs()

            # Create adjacency matrix based on correlation threshold
            adjacency_matrix = (correlation_matrix > self.correlation_threshold).astype(int)

            # Remove self-loops
            np.fill_diagonal(adjacency_matrix.values, 0)

            # Convert to edge indices
            edge_indices = []
            edge_weights = []

            for i in range(len(adjacency_matrix)):
                for j in range(len(adjacency_matrix)):
                    if adjacency_matrix.iloc[i, j] == 1:
                        edge_indices.append([i, j])
                        edge_weights.append(correlation_matrix.iloc[i, j])

            if not edge_indices:
                logger.warning("No edges found in graph. Consider lowering correlation threshold.")
                return None

            # Convert to PyTorch tensors
            edge_index = torch.tensor(edge_indices, dtype=torch.long).t().contiguous()
            edge_attr = torch.tensor(edge_weights, dtype=torch.float)

            # Node features (normalized feature values)
            node_features = torch.tensor(features_df.values, dtype=torch.float)

            # Create PyTorch Geometric Data object
            graph_data = Data(
                x=node_features,
                edge_index=edge_index,
                edge_attr=edge_attr,
                num_nodes=len(features_df.columns)
            )

            self.graph_data = graph_data
            logger.info(f"Created graph with {graph_data.num_nodes} nodes and {graph_data.num_edges} edges")

            return graph_data

        except Exception as e:
            logger.error(f"Error creating graph from features: {e}")
            return None

    def build_gnn_model(self, input_dim: int, output_dim: int) -> Optional[Any]:
        """
        Build GNN model architecture.

        Args:
            input_dim: Input feature dimension
            output_dim: Output dimension (number of classes)

        Returns:
            PyTorch GNN model or None if not available
        """
        if not self.is_available:
            logger.warning("Cannot build GNN model: PyTorch Geometric not available")
            return None

        try:
            if self.model_type == 'GCN':
                model = GCNModel(
                    input_dim=input_dim,
                    hidden_dim=self.hidden_dim,
                    output_dim=output_dim,
                    num_layers=self.num_layers,
                    dropout_rate=self.dropout_rate
                )
            elif self.model_type == 'GAT':
                model = GATModel(
                    input_dim=input_dim,
                    hidden_dim=self.hidden_dim,
                    output_dim=output_dim,
                    num_layers=self.num_layers,
                    dropout_rate=self.dropout_rate
                )
            else:  # GraphConv
                model = GraphConvModel(
                    input_dim=input_dim,
                    hidden_dim=self.hidden_dim,
                    output_dim=output_dim,
                    num_layers=self.num_layers,
                    dropout_rate=self.dropout_rate
                )

            self.model = model
            logger.info(f"Built {self.model_type} model with {input_dim} input features")

            return model

        except Exception as e:
            logger.error(f"Error building GNN model: {e}")
            return None

    def train_gnn(self, train_data: Any, val_data: Optional[Any] = None) -> Dict[str, Any]:
        """
        Train the GNN model.

        Args:
            train_data: Training graph data
            val_data: Optional validation graph data

        Returns:
            Training results dictionary
        """
        if not self.is_available or self.model is None:
            return {'error': 'GNN not available or model not built'}

        try:
            # Set up optimizer and loss function
            optimizer = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate)
            criterion = torch.nn.CrossEntropyLoss()

            # Training loop
            training_losses = []
            validation_losses = []

            self.model.train()

            for epoch in range(self.num_epochs):
                optimizer.zero_grad()

                # Forward pass
                out = self.model(train_data.x, train_data.edge_index)
                loss = criterion(out, train_data.y)

                # Backward pass
                loss.backward()
                optimizer.step()

                training_losses.append(loss.item())

                # Validation
                if val_data is not None:
                    self.model.eval()
                    with torch.no_grad():
                        val_out = self.model(val_data.x, val_data.edge_index)
                        val_loss = criterion(val_out, val_data.y)
                        validation_losses.append(val_loss.item())
                    self.model.train()

                # Log progress
                if epoch % 10 == 0:
                    logger.info(f"Epoch {epoch}/{self.num_epochs}, Loss: {loss.item():.4f}")

            training_results = {
                'training_losses': training_losses,
                'validation_losses': validation_losses,
                'final_training_loss': training_losses[-1] if training_losses else None,
                'final_validation_loss': validation_losses[-1] if validation_losses else None,
                'epochs_completed': len(training_losses)
            }

            logger.info(f"GNN training completed. Final loss: {training_losses[-1]:.4f}")
            return training_results

        except Exception as e:
            logger.error(f"Error training GNN: {e}")
            return {'error': str(e)}

    def predict(self, data: Any) -> Optional[np.ndarray]:
        """
        Make predictions using trained GNN model.

        Args:
            data: Graph data for prediction

        Returns:
            Predictions array or None if not available
        """
        if not self.is_available or self.model is None:
            logger.warning("Cannot make predictions: GNN not available or not trained")
            return None

        try:
            self.model.eval()
            with torch.no_grad():
                predictions = self.model(data.x, data.edge_index)
                predictions = torch.softmax(predictions, dim=1)
                return predictions.numpy()

        except Exception as e:
            logger.error(f"Error making GNN predictions: {e}")
            return None

    def create_simple_gnn_model(self, in_channels: int, hidden_channels: int, out_channels: int) -> Optional[Any]:
        """
        Create a simple GNN model compatible with the main file interface.

        This method provides backward compatibility with the simple GNNModel
        interface from the main file while leveraging the advanced modular architecture.

        Args:
            in_channels: Number of input channels/features
            hidden_channels: Number of hidden layer channels
            out_channels: Number of output channels/classes

        Returns:
            Simple GNN model instance or None if not available
        """
        if not self.is_available:
            logger.warning("Cannot create simple GNN model: PyTorch Geometric not available")
            return None

        try:
            # Create a simple 2-layer GCN model similar to main file
            model = SimpleGNNModel(in_channels, hidden_channels, out_channels)
            logger.info(f"Created simple GNN model: {in_channels} -> {hidden_channels} -> {out_channels}")
            return model

        except Exception as e:
            logger.error(f"Error creating simple GNN model: {e}")
            return None


# GNN Model Architectures (only available if torch_geometric is installed)
if TORCH_GEOMETRIC_AVAILABLE:

    class GCNModel(nn.Module):
        """Graph Convolutional Network model."""

        def __init__(self, input_dim: int, hidden_dim: int, output_dim: int,
                     num_layers: int = 3, dropout_rate: float = 0.2):
            super(GCNModel, self).__init__()

            self.layers = nn.ModuleList()
            self.layers.append(GCNConv(input_dim, hidden_dim))

            for _ in range(num_layers - 2):
                self.layers.append(GCNConv(hidden_dim, hidden_dim))

            self.layers.append(GCNConv(hidden_dim, output_dim))
            self.dropout = nn.Dropout(dropout_rate)

        def forward(self, x, edge_index):
            for i, layer in enumerate(self.layers[:-1]):
                x = layer(x, edge_index)
                x = F.relu(x)
                x = self.dropout(x)

            x = self.layers[-1](x, edge_index)
            return x


    class GATModel(nn.Module):
        """Graph Attention Network model."""

        def __init__(self, input_dim: int, hidden_dim: int, output_dim: int,
                     num_layers: int = 3, dropout_rate: float = 0.2, heads: int = 4):
            super(GATModel, self).__init__()

            self.layers = nn.ModuleList()
            self.layers.append(GATConv(input_dim, hidden_dim, heads=heads, dropout=dropout_rate))

            for _ in range(num_layers - 2):
                self.layers.append(GATConv(hidden_dim * heads, hidden_dim, heads=heads, dropout=dropout_rate))

            self.layers.append(GATConv(hidden_dim * heads, output_dim, heads=1, dropout=dropout_rate))
            self.dropout = nn.Dropout(dropout_rate)

        def forward(self, x, edge_index):
            for i, layer in enumerate(self.layers[:-1]):
                x = layer(x, edge_index)
                x = F.relu(x)
                x = self.dropout(x)

            x = self.layers[-1](x, edge_index)
            return x


    class GraphConvModel(nn.Module):
        """General Graph Convolution model."""

        def __init__(self, input_dim: int, hidden_dim: int, output_dim: int,
                     num_layers: int = 3, dropout_rate: float = 0.2):
            super(GraphConvModel, self).__init__()

            self.layers = nn.ModuleList()
            self.layers.append(GraphConv(input_dim, hidden_dim))

            for _ in range(num_layers - 2):
                self.layers.append(GraphConv(hidden_dim, hidden_dim))

            self.layers.append(GraphConv(hidden_dim, output_dim))
            self.dropout = nn.Dropout(dropout_rate)

        def forward(self, x, edge_index):
            for i, layer in enumerate(self.layers[:-1]):
                x = layer(x, edge_index)
                x = F.relu(x)
                x = self.dropout(x)

            x = self.layers[-1](x, edge_index)
            return x


    class SimpleGNNModel(nn.Module):
        """
        Simple GNN model compatible with main file interface.

        This is a simplified 2-layer Graph Convolutional Network that matches
        the interface and functionality of the GNNModel in the main file.
        """

        def __init__(self, in_channels: int, hidden_channels: int, out_channels: int):
            """
            Initialize simple GNN model.

            Args:
                in_channels: Number of input features
                hidden_channels: Number of hidden layer features
                out_channels: Number of output features/classes
            """
            super(SimpleGNNModel, self).__init__()
            self.conv1 = GCNConv(in_channels, hidden_channels)
            self.conv2 = GCNConv(hidden_channels, out_channels)

        def forward(self, data):
            """
            Forward pass through the simple GNN.

            Args:
                data: Graph data object with x (features) and edge_index

            Returns:
                Output tensor after graph convolutions
            """
            x, edge_index = data.x, data.edge_index
            x = self.conv1(x, edge_index)
            x = F.relu(x)
            x = F.dropout(x, p=0.5, training=self.training)
            x = self.conv2(x, edge_index)
            return x

else:
    # Placeholder classes when torch_geometric is not available
    class GCNModel:
        def __init__(self, *args, **kwargs):
            raise ImportError("PyTorch Geometric not available. Cannot create GCN model.")

    class GATModel:
        def __init__(self, *args, **kwargs):
            raise ImportError("PyTorch Geometric not available. Cannot create GAT model.")

    class GraphConvModel:
        def __init__(self, *args, **kwargs):
            raise ImportError("PyTorch Geometric not available. Cannot create GraphConv model.")

    class SimpleGNNModel:
        def __init__(self, *args, **kwargs):
            raise ImportError("PyTorch Geometric not available. Cannot create Simple GNN model.")


class ModelTrainer:
    """
    Advanced machine learning model training system for trading strategies.

    Provides comprehensive model training with hyperparameter optimization, feature
    selection, validation, and explainability analysis. Supports multiple model
    types including XGBoost, MiniRocket, and neural networks.

    Features:
    - Optuna-based hyperparameter optimization
    - Shadow set validation for robust performance estimation
    - SHAP-based feature importance and explainability
    - Multi-objective optimization with custom metrics
    - Class balancing and sample weighting
    - Early stopping and overfitting prevention
    - Comprehensive model evaluation and reporting

    Attributes:
        config: Configuration model with training parameters
        gemini_analyzer: AI analyzer for intelligent optimization
        shap_summaries: SHAP importance summaries by model
        class_weights: Computed class weights for balancing
        classification_report_str: Detailed classification performance report
        study: Optuna study object for hyperparameter optimization
    """

    def __init__(self, config: ConfigModel, gemini_analyzer: GeminiAnalyzer):
        """
        Initialize the ModelTrainer with configuration and AI analyzer.

        Args:
            config: Configuration model containing training parameters
            gemini_analyzer: AI analyzer for intelligent optimization guidance
        """
        self.config = config
        self.gemini_analyzer = gemini_analyzer
        self.shap_summaries: Dict[str, Optional[pd.DataFrame]] = {}
        self.class_weights: Optional[Dict] = None
        self.classification_report_str: str = "N/A"
        self.is_minirocket_model = 'MiniRocket' in getattr(config, 'strategy_name', '')
        self.study: Optional[optuna.study.Study] = None

        # Initialize GNN model if available
        self.gnn_model = GNNModel(config) if TORCH_GEOMETRIC_AVAILABLE else None
        self.enable_gnn = getattr(config, 'ENABLE_GNN_MODELS', False) and TORCH_GEOMETRIC_AVAILABLE

        if self.enable_gnn and self.gnn_model:
            logger.info("GNN models enabled and available")
        elif getattr(config, 'ENABLE_GNN_MODELS', False) and not TORCH_GEOMETRIC_AVAILABLE:
            logger.warning("GNN models requested but PyTorch Geometric not available")

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

    def train_all_models(self, df_train_labeled: pd.DataFrame, feature_list: List[str],
                        framework_history: Dict, cycle_directives: Dict = {},
                        regime: str = "Unknown") -> Tuple[Optional[Dict[str, Any]], Optional[str]]:
        """
        Orchestrates the training of an ENSEMBLE of models, one for each labeling horizon.

        Args:
            df_train_labeled: Labeled training data with multiple target columns
            feature_list: List of features to use for training
            framework_history: Historical performance data for optimization
            cycle_directives: Specific directives for this training cycle
            regime: Current market regime for context

        Returns:
            Tuple of (ensemble_results, failure_reason)
        """
        logger.info(f"-> Orchestrating ENSEMBLE model training for horizons: {getattr(self.config, 'LABEL_HORIZONS', [30, 60, 90])}")
        self.shap_summaries = {}

        # Get AI-defined search spaces for all horizons before training
        diagnosed_regime_for_ai = "Trending"  # This could be passed in or made more dynamic
        horizon_search_spaces = self.gemini_analyzer.propose_horizon_specific_search_spaces(
            horizons=getattr(self.config, 'LABEL_HORIZONS', [30, 60, 90]),
            strategy_name=getattr(self.config, 'strategy_name', 'Unknown'),
            diagnosed_regime=diagnosed_regime_for_ai
        )

        trained_pipelines = {}
        features_per_model = {}
        horizon_performance_metrics = {}

        for horizon in label_horizons:
            target_col = f'target_signal_pressure_class_h{horizon}'

            # Check if target column exists
            if target_col not in df_train_labeled.columns:
                logger.warning(f"Target column {target_col} not found. Skipping horizon {horizon}")
                continue

            task_name = f'primary_model_h{horizon}'
            logger.info(f"--- Training Model for Horizon: {horizon} ---")

            # Get AI-defined search space for this horizon
            search_space_for_horizon = horizon_search_spaces.get(f'h{horizon}', {})

            # Train model for this horizon with AI search space
            pipeline, best_f1, selected_features, failure_reason = self.train_single_model(
                df_train=df_train_labeled,
                feature_list=feature_list,
                target_col=target_col,
                model_type='classification',
                task_name=task_name,
                framework_history=framework_history,
                search_space=search_space_for_horizon,
                regime=regime
            )

            if pipeline is None:
                logger.error(f"Failed to train model for horizon {horizon}: {failure_reason}")
                continue

            # Store enhanced results
            trained_pipelines[task_name] = pipeline
            features_per_model[task_name] = selected_features
            horizon_performance_metrics[task_name] = {
                'f1_score': best_f1,
                'horizon': horizon,
                'target_column': target_col,
                'failure_reason': failure_reason,
                'search_space_used': search_space_for_horizon
            }

        if not trained_pipelines:
            logger.error("No models were successfully trained in the ensemble")
            return None, "NO_MODELS_TRAINED"

        # Compile enhanced ensemble results
        final_results = {
            'trained_pipelines': trained_pipelines,
            'features_per_model': features_per_model,
            'horizon_performance_metrics': horizon_performance_metrics,
            'shap_summaries': self.shap_summaries,
            'search_spaces_used': horizon_search_spaces,
            'training_summary': {
                'total_models': len(trained_pipelines),
                'successful_horizons': list(trained_pipelines.keys()),
                'regime': regime,
                'ai_guided_optimization': True
            }
        }

        logger.info(f"✓ Ensemble training completed. Successfully trained {len(trained_pipelines)} models with AI-guided optimization.")
        return final_results, None

    def train_ensemble_models(self, df_train_labeled: pd.DataFrame, feature_list: List[str],
                             regime: str = "Unknown") -> Dict[str, Any]:
        """
        Alias for train_all_models to maintain compatibility with framework_orchestrator.

        Args:
            df_train_labeled: Training data with labels
            feature_list: List of features to use
            regime: Market regime for adaptive training

        Returns:
            Training results dictionary
        """
        results, _ = self.train_all_models(df_train_labeled, feature_list, {}, {}, regime)
        return results if results else {}

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
            
            # Enhanced feature selection with multiple methods
            logger.info(f"  - Stage 1: Running Feature Selection method: '{getattr(self.config, 'FEATURE_SELECTION_METHOD', 'mutual_info')}'...")

            X_for_tuning = pd.DataFrame()
            elite_feature_names: List[str] = []
            features_for_pipeline_input: List[str] = []
            pre_fitted_transformer = None

            selection_method = getattr(self.config, 'FEATURE_SELECTION_METHOD', 'mutual_info').lower()
            if selection_method == 'trex':
                elite_feature_names = self._select_features_with_trex(X, y)
                X_for_tuning = X[elite_feature_names]
                features_for_pipeline_input = elite_feature_names
            elif selection_method == 'mutual_info':
                pruned_features = self._remove_redundant_features(X)
                X_pruned = X[pruned_features]
                elite_feature_names = self._select_elite_features_mi(X_pruned, y)
                X_for_tuning = X_pruned[elite_feature_names]
                features_for_pipeline_input = elite_feature_names
            elif selection_method == 'pca':
                X_for_tuning, elite_feature_names, pre_fitted_transformer = self._select_features_with_pca(X)
                features_for_pipeline_input = X.columns.tolist()
            else:
                elite_feature_names = feature_list
                X_for_tuning = X
                features_for_pipeline_input = feature_list

            if X_for_tuning.empty or not elite_feature_names:
                logger.error(f"  - [{task_name}] Feature selection resulted in an empty set. Aborting this model.")
                return None, None, None, "FEATURE_SELECTION_FAILED"

            logger.info(f"  - [{task_name}] Stage 2: Optimizing hyperparameters with {X_for_tuning.shape[1]} features...")

            # Sample data for optimization if needed
            sample_size = min(len(X_for_tuning), 30000)
            if len(X_for_tuning) > sample_size:
                X_sample = X_for_tuning.sample(n=sample_size, random_state=42)
                y_sample = y.loc[X_sample.index]
            else:
                X_sample, y_sample = X_for_tuning, y

            # Pass the horizon-specific search_space down to the optimization function
            study = self._optimize_hyperparameters(X_sample, y_sample, model_type, task_name, df_train, search_space)
            
            self.study = study
            if not study or not study.best_trials:
                logger.error(f"  - [{task_name}] Optuna study failed. Aborting this model.")
                return None, None, None, "OPTUNA_FAILED"

            best_trials = study.best_trials
            ai_decision = None
            try:
                logger.info(f"  - [{task_name}] Asking AI to select best trade-off from {len(best_trials)} trials...")
                current_directive = self.gemini_analyzer.establish_strategic_directive(
                    framework_history.get('historical_runs', []) if framework_history else [],
                    getattr(self.config, 'operating_state', 'BASELINE')
                )
                ai_decision = self.gemini_analyzer.select_best_tradeoff(
                    best_trials=best_trials,
                    risk_profile=getattr(self.config, 'RISK_PROFILE', 'MODERATE'),
                    strategic_directive=current_directive
                )
            except Exception as e:
                logger.warning(f"  - AI trial selection failed: {e}. Using fallback logic.")

            if ai_decision and ai_decision.get('selected_trial_number') is not None:
                selected_trial_number = ai_decision.get('selected_trial_number')
                selected_trial = next((t for t in best_trials if t.number == selected_trial_number), None)
                if 'disqualified_trials' in ai_decision:
                    current_cycle = len(framework_history.get('historical_runs', [])) + 1 if framework_history else 1
                    self._log_disqualified_trials(ai_decision['disqualified_trials'], task_name, current_cycle)
            else:
                selected_trial = max(best_trials, key=lambda t: t.values[0] if t.values else -float('inf'))
                logger.info(f"  - FALLBACK: Selected Trial #{selected_trial.number} with highest score ({selected_trial.values[0]:.3f}).")

            if selected_trial is None:
                logger.error(f"  - Could not determine a best trial. Aborting model training for task '{task_name}'.")
                return None, None, None, "OPTUNA_SELECTION_FAILED"

            best_params = selected_trial.params
            
            # Enhanced threshold finding and AI-driven F1 gate determination
            if model_type == 'classification':
                logger.info(f"  - [{task_name}] Stage 3: Finding best threshold and F1 score...")
                best_threshold, f1_at_best_thresh = self._find_best_threshold(best_params, X_for_tuning, y_sample)

                # AI-driven F1 gate determination
                optuna_summary = {"best_values": selected_trial.values, "best_params": best_params}
                label_dist_summary = self._create_label_distribution_report(df_train, target_col)

                ai_f1_gate_decision = self.gemini_analyzer.determine_dynamic_f1_gate(
                    optuna_summary,
                    label_dist_summary,
                    "BASELINE ESTABLISHMENT",
                    f1_at_best_thresh
                )

                if 'MIN_F1_SCORE_GATE' in ai_f1_gate_decision:
                    self.config.MIN_F1_SCORE_GATE = ai_f1_gate_decision['MIN_F1_SCORE_GATE']

                if f1_at_best_thresh < getattr(self.config, 'MIN_F1_SCORE_GATE', 0.3):
                    logger.error(f"  - [{task_name}] MODEL REJECTED. F1 score {f1_at_best_thresh:.3f} is below AI quality gate of {getattr(self.config, 'MIN_F1_SCORE_GATE', 0.3):.3f}.")
                    return None, None, None, "F1_SCORE_TOO_LOW"
            else:
                best_threshold = 0.5
                f1_at_best_thresh = 0.0
            
            # Train final model with enhanced features
            logger.info(f"  - [{task_name}] Stage 4: Training final model...")
            final_pipeline = self._train_final_model(
                best_params=best_params,
                X_input=df_train[features_for_pipeline_input],
                y_input=y,
                model_type=model_type,
                task_name=task_name,
                pre_fitted_transformer=pre_fitted_transformer
            )
            if final_pipeline is None:
                return None, None, None, "FINAL_TRAINING_FAILED"

            # Enhanced shadow set validation
            df_train_sorted = df_train.sort_index()
            split_date = df_train_sorted.index.max() - pd.DateOffset(months=2)
            df_shadow_val = df_train_sorted.loc[df_train_sorted.index > split_date]

            if len(df_shadow_val) < 100:
                df_shadow_val = pd.DataFrame()

            if not self._validate_on_shadow_set(final_pipeline, df_shadow_val, features_for_pipeline_input, model_type, target_col):
                return None, None, None, "SHADOW_VALIDATION_FAILURE"

            # Enhanced SHAP calculation
            if getattr(self.config, 'CALCULATE_SHAP_VALUES', True):
                X_for_fitting = df_train[features_for_pipeline_input].sample(n=min(2000, len(df_train)), random_state=42)
                transformer_step = final_pipeline.named_steps.get('transformer') or final_pipeline.named_steps.get('scaler')

                X_for_fitting_cleaned = X_for_fitting.select_dtypes(include=np.number).fillna(0)
                X_for_shap = transformer_step.transform(X_for_fitting_cleaned) if transformer_step else X_for_fitting_cleaned

                self._generate_shap_summary(final_pipeline.named_steps['model'], X_for_shap, elite_feature_names, task_name, regime)

            logger.info(f"--- [SUCCESS] Model training for '{task_name}' complete. ---")
            return final_pipeline, best_threshold, features_for_pipeline_input, None
            
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

    def _create_label_distribution_report(self, df: pd.DataFrame, target_col: str) -> Dict[str, Any]:
        """
        Create a comprehensive label distribution report for AI analysis.
        """
        try:
            if target_col not in df.columns:
                return {"error": f"Target column '{target_col}' not found"}

            target_series = df[target_col].dropna()
            if len(target_series) == 0:
                return {"error": "No valid target values found"}

            # Basic distribution statistics
            value_counts = target_series.value_counts().sort_index()
            total_samples = len(target_series)

            distribution_report = {
                "total_samples": total_samples,
                "unique_classes": len(value_counts),
                "class_distribution": value_counts.to_dict(),
                "class_percentages": (value_counts / total_samples * 100).round(2).to_dict(),
                "most_common_class": value_counts.index[0],
                "least_common_class": value_counts.index[-1],
                "imbalance_ratio": value_counts.max() / value_counts.min() if value_counts.min() > 0 else float('inf'),
                "target_column": target_col
            }

            return distribution_report

        except Exception as e:
            logger.warning(f"Failed to create label distribution report: {e}")
            return {"error": str(e)}

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

    def _select_features_with_trex(self, X: pd.DataFrame, y: pd.Series) -> List[str]:
        """
        Select features using Tree-based Recursive Extraction (TREX).
        Uses Random Forest feature importance for robust feature selection.
        """
        if not ADVANCED_SELECTION_AVAILABLE:
            logger.warning("Advanced feature selection not available. Using all features.")
            return X.columns.tolist()

        try:
            # Use Random Forest for feature importance
            rf = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=1)
            rf.fit(X.fillna(0), y)

            # Get feature importance
            importance_scores = rf.feature_importances_
            feature_importance = pd.DataFrame({
                'feature': X.columns,
                'importance': importance_scores
            }).sort_values('importance', ascending=False)

            # Select top features (configurable)
            n_features = min(getattr(self.config, 'MAX_FEATURES_TREX', 50), len(X.columns))
            selected_features = feature_importance.head(n_features)['feature'].tolist()

            logger.info(f"TREX selected {len(selected_features)} features from {len(X.columns)} total.")
            return selected_features

        except Exception as e:
            logger.warning(f"TREX feature selection failed: {e}. Using all features.")
            return X.columns.tolist()

    def _remove_redundant_features(self, X: pd.DataFrame, correlation_threshold: float = 0.95) -> List[str]:
        """
        Remove highly correlated features to reduce redundancy.
        """
        try:
            # Calculate correlation matrix
            corr_matrix = X.corr().abs()

            # Find pairs of highly correlated features
            upper_triangle = corr_matrix.where(
                np.triu(np.ones(corr_matrix.shape), k=1).astype(bool)
            )

            # Find features to drop
            to_drop = [column for column in upper_triangle.columns if any(upper_triangle[column] > correlation_threshold)]

            # Keep features not in drop list
            features_to_keep = [col for col in X.columns if col not in to_drop]

            logger.info(f"Removed {len(to_drop)} redundant features. Keeping {len(features_to_keep)} features.")
            return features_to_keep

        except Exception as e:
            logger.warning(f"Redundant feature removal failed: {e}. Using all features.")
            return X.columns.tolist()

    def _select_elite_features_mi(self, X: pd.DataFrame, y: pd.Series) -> List[str]:
        """
        Select elite features using mutual information with enhanced scoring.
        """
        try:
            # Calculate mutual information scores
            mi_scores = mutual_info_classif(X.fillna(0), y, random_state=42)

            # Create feature importance DataFrame
            feature_importance = pd.DataFrame({
                'feature': X.columns,
                'mi_score': mi_scores
            }).sort_values('mi_score', ascending=False)

            # Select top features
            n_features = min(getattr(self.config, 'MAX_FEATURES_MI', 40), len(X.columns))
            selected_features = feature_importance.head(n_features)['feature'].tolist()

            logger.info(f"Mutual Info selected {len(selected_features)} elite features.")
            return selected_features

        except Exception as e:
            logger.warning(f"Elite feature selection failed: {e}. Using all features.")
            return X.columns.tolist()

    def _select_features_with_pca(self, X: pd.DataFrame, n_components: float = 0.95) -> Tuple[pd.DataFrame, List[str], Pipeline]:
        """
        Select features using PCA for dimensionality reduction.
        Returns transformed data, component names, and fitted transformer.
        """
        if not ADVANCED_SELECTION_AVAILABLE:
            logger.warning("PCA not available. Using original features.")
            return X, X.columns.tolist(), None

        try:
            # Prepare data
            X_filled = X.fillna(0)

            # Create PCA transformer
            pca = PCA(n_components=n_components, random_state=42)
            X_transformed = pca.fit_transform(X_filled)

            # Create component names
            n_components_actual = X_transformed.shape[1]
            component_names = [f'PC_{i+1}' for i in range(n_components_actual)]

            # Convert back to DataFrame
            X_pca = pd.DataFrame(X_transformed, index=X.index, columns=component_names)

            # Create pipeline for later use
            pca_pipeline = Pipeline([('pca', pca)])

            logger.info(f"PCA reduced {X.shape[1]} features to {n_components_actual} components "
                       f"(explained variance: {pca.explained_variance_ratio_.sum():.3f})")

            return X_pca, component_names, pca_pipeline

        except Exception as e:
            logger.warning(f"PCA feature selection failed: {e}. Using original features.")
            return X, X.columns.tolist(), None

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

    def _log_disqualified_trials(self, disqualified_list: List[Dict], task_name: str, cycle_num: int):
        """
        Logs disqualified trials to a persistent JSONL file for analysis.
        Helps track which hyperparameter combinations are consistently poor.
        """
        if not disqualified_list or not isinstance(disqualified_list, list):
            return

        log_path = getattr(self.config, 'DISQUALIFIED_TRIALS_PATH', 'disqualified_trials.jsonl')
        timestamp = datetime.now().isoformat()

        try:
            with open(log_path, 'a') as f:
                for item in disqualified_list:
                    # Ensure item is a dict before processing
                    if not isinstance(item, dict):
                        continue

                    log_entry = {
                        "timestamp": timestamp,
                        "cycle_num": cycle_num,
                        "task_name": task_name,
                        "strategy_name": getattr(self.config, 'strategy_name', 'Unknown'),
                        "trial_number": item.get('trial_number'),
                        "reason": item.get('reason'),
                        "parameters": item.get('parameters', {}),
                        "score": item.get('score', None)
                    }
                    f.write(json.dumps(log_entry) + '\n')
            logger.info(f"  - Logged {len(disqualified_list)} disqualified trials to vetting memory.")
        except Exception as e:
            logger.error(f"  - Failed to log disqualified trials: {e}")

    def _optimize_hyperparameters(self, X: pd.DataFrame, y: pd.Series, model_type: str,
                                 task_name: str, df_full_for_metrics: pd.DataFrame, search_space: Dict) -> Optional[optuna.study.Study]:
        """
        Advanced hyperparameter optimization with financial metrics and AI-guided search spaces.
        Includes validation step to automatically correct AI-generated hyperparameter ranges.
        """
        VALID_TUNABLE_PARAMS = {
            'n_estimators', 'max_depth', 'learning_rate', 'subsample',
            'colsample_bytree', 'reg_alpha', 'reg_lambda', 'gamma', 'min_child_weight'
        }

        def _get_trial_param(trial, name, space_def):
            """Helper to suggest a parameter based on its definition, with validation."""
            param_type = space_def.get('type')

            if param_type == 'int':
                low = space_def['low']
                high = space_def['high']
                step = space_def.get('step', 1)

                # Validate and correct the range to prevent Optuna warnings
                if (high - low) % step != 0:
                    corrected_high = high - ((high - low) % step)
                    # Failsafe in case of weird AI inputs (e.g., high < low)
                    if corrected_high >= low:
                        logger.debug(f"Correcting Optuna int range for '{name}' from [{low}, {high}] to [{low}, {corrected_high}] to be divisible by step {step}.")
                        high = corrected_high

                return trial.suggest_int(name, low, high, step=step)

            elif param_type == 'float':
                low = space_def['low']
                is_log = space_def.get('log', False)
                if is_log and low <= 0:
                    low = 1e-9
                    logger.debug(f"Adjusted 'low' for log-scale param '{name}' from <=0 to {low}.")
                return trial.suggest_float(name, low, space_def['high'], log=is_log)

            logger.warning(f"Unsupported or missing parameter type for '{name}'. Skipping.")
            return None

        def objective(trial: optuna.Trial) -> Tuple[float, float]:
            model_params = {}
            for param_name, space_def in search_space.items():
                if param_name not in VALID_TUNABLE_PARAMS:
                    if param_name not in ['sl_multiplier', 'tp_multiplier']:
                        logger.warning(f"AI suggested a non-tunable/unsupported parameter '{param_name}'. Ignoring.")
                    continue
                if isinstance(space_def, dict) and 'type' in space_def:
                     param_value = _get_trial_param(trial, param_name, space_def)
                     if param_value is not None:
                         model_params[param_name] = param_value
                else:
                     logger.warning(f"Skipping malformed search space definition for '{param_name}'.")

            model_params['n_jobs'] = 1
            sl_multiplier = trial.suggest_float('sl_multiplier', 0.8, 3.0)
            tp_multiplier = trial.suggest_float('tp_multiplier', 1.0, 4.0)

            if model_type != 'classification':
                raise optuna.exceptions.TrialPruned("Optimization objective only supports classification.")

            if len(y.unique()) < 2:
                raise optuna.exceptions.TrialPruned("Cannot create stratified split with only one class.")

            X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.25, random_state=42, stratify=y)

            model = xgb.XGBClassifier(**model_params, objective='multi:softprob', eval_metric='mlogloss', seed=42, num_class=3)
            class_weights_train = compute_class_weight('balanced', classes=np.unique(y_train), y=y_train)
            sample_weight_train = y_train.map(dict(zip(np.unique(y_train), class_weights_train)))
            model.fit(X_train, y_train, sample_weight=sample_weight_train, verbose=False)

            val_probas = model.predict_proba(X_val)
            confidence_threshold = 0.60
            predictions = np.argmax(val_probas, axis=1)
            predictions[np.max(val_probas, axis=1) < confidence_threshold] = 1

            pnl, wins, losses, num_trades = [], 0, 0, 0
            for j, signal in enumerate(predictions):
                if signal != 1:
                    num_trades += 1
                    payoff_ratio = tp_multiplier / sl_multiplier if sl_multiplier > 0 else 0
                    true_outcome = y_val.iloc[j]
                    if true_outcome == signal:
                        pnl.append(payoff_ratio)
                        wins += 1
                    else:
                        pnl.append(-1.0)
                        losses += 1

            financial_score = 0.0
            MIN_STD_DEV_THRESHOLD = 0.01

            if pnl and len(pnl) > 1:
                pnl_series = pd.Series(pnl)
                pnl_std = pnl_series.std()

                if pnl_std < MIN_STD_DEV_THRESHOLD:
                    sharpe_score = -5.0
                else:
                    sharpe_score = pnl_series.mean() / (pnl_std + 1e-9)

                weights = getattr(self.config, 'STATE_BASED_CONFIG', {}).get(getattr(self.config, 'operating_state', 'BASELINE'), {}).get("optimization_weights", {"sharpe": 1.0})
                financial_score += weights.get("sharpe", 0.8) * sharpe_score
                financial_score += weights.get("num_trades", 0.2) * np.log1p(num_trades)

            if num_trades < 10:
                financial_score -= 5

            win_rate = wins / num_trades if num_trades > 0 else 0
            loss_rate = losses / num_trades if num_trades > 0 else 0
            avg_win = sum(p for p in pnl if p > 0) / wins if wins > 0 else 0
            avg_loss = abs(sum(p for p in pnl if p < 0) / losses) if losses > 0 else 0
            expected_payoff = (win_rate * avg_win) - (loss_rate * avg_loss)

            stability_adjusted_score = financial_score

            trial.set_user_attr('score_stability_std', 0)
            trial.set_user_attr('avg_financial_score_pre_penalty', financial_score)

            return stability_adjusted_score, expected_payoff

        try:
            study = optuna.create_study(directions=['maximize', 'maximize'], pruner=optuna.pruners.HyperbandPruner(), study_name=task_name)
            study.optimize(
                objective,
                n_trials=getattr(self.config, 'OPTUNA_TRIALS', 50),
                timeout=3600,
                n_jobs=getattr(self.config, 'OPTUNA_N_JOBS', 1),
                callbacks=[self._log_optuna_trial]
            )
            sys.stdout.write('\n')
            return study
        except Exception as e:
            sys.stdout.write('\n')
            logger.error(f"    - [{task_name}] Optuna study failed: {e}", exc_info=True)
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

    def _generate_shap_summary(self, model, X_scaled, feature_names, task_name: str, regime: str):
        """
        Generate enhanced SHAP summary using the modern SHAP API with robust aggregation.
        Prevents dimensionality errors and provides comprehensive feature importance analysis.
        """
        logger.info(f"    - Generating ENHANCED SHAP feature importance for '{task_name}'...")
        try:
            X_sample = shap.utils.sample(X_scaled, 2000) if len(X_scaled) > 2000 else X_scaled

            # Use the modern shap.Explainer for consistent output
            explainer = shap.Explainer(model, X_sample)
            shap_values_obj = explainer(X_sample)

            # The .values attribute can be 2D (samples, features) or 3D (samples, features, classes)
            shap_values_array = shap_values_obj.values

            # Robustly aggregate the SHAP values to a 1D array
            # 1. Take the absolute value of all scores
            abs_shap_values = np.abs(shap_values_array)

            # 2. If the array is 3D (multi-class output), average across the class dimension (last axis) first
            if abs_shap_values.ndim == 3:
                abs_shap_values = abs_shap_values.mean(axis=2)

            # 3. Now the array is guaranteed to be 2D (samples, features), so average across the samples
            global_importance_values = abs_shap_values.mean(axis=0)

            # Final validation checks
            if global_importance_values.ndim != 1 or len(global_importance_values) != len(feature_names):
                logger.error(f"SHAP shape mismatch after aggregation for '{task_name}'. Skipping summary.")
                self.shap_summaries[task_name] = None
                return

            summary_df = pd.DataFrame({
                'feature': feature_names,
                'SHAP_Importance': global_importance_values
            }).sort_values(by='SHAP_Importance', ascending=False).reset_index(drop=True)

            summary_df['rank'] = summary_df.index + 1
            summary_df['model_id'] = task_name
            summary_df['regime_when_trained'] = regime
            summary_df['is_top_10_percent'] = summary_df['rank'] <= max(1, len(summary_df) // 10)

            self.shap_summaries[task_name] = summary_df
            logger.info(f"    - Enhanced SHAP summary generated successfully for '{task_name}'.")

        except Exception as e:
            logger.error(f"    - Failed to generate SHAP summary for '{task_name}': {e}", exc_info=True)
            self.shap_summaries[task_name] = None

    def _calculate_shap_values(self, pipeline: Pipeline, X: pd.DataFrame, feature_names: List[str]) -> Optional[pd.DataFrame]:
        """
        Calculate SHAP values for feature importance analysis with enhanced error handling.

        Uses modern SHAP API with robust aggregation to prevent dimensionality errors
        and provides comprehensive feature importance analysis.
        """
        if not SHAP_AVAILABLE:
            logger.warning("SHAP not available. Skipping SHAP calculation.")
            return None

        try:
            # Get the model from pipeline
            model = None
            for step_name, step in pipeline.steps:
                if hasattr(step, 'predict'):
                    model = step
                    break

            if model is None:
                logger.warning("No model found in pipeline for SHAP analysis")
                return None

            # Transform data through preprocessing steps
            X_transformed = X.copy()
            for step_name, step in pipeline.steps[:-1]:  # All steps except the model
                X_transformed = step.transform(X_transformed)

            # Sample data if too large for SHAP analysis
            X_sample = X_transformed
            if len(X_transformed) > 2000:
                import shap.utils
                X_sample = shap.utils.sample(X_transformed, 2000)
                logger.info(f"Sampling {len(X_sample)} rows for SHAP analysis (from {len(X_transformed)})")

            # Use modern SHAP Explainer for consistent output
            explainer = shap.Explainer(model, X_sample)
            shap_values_obj = explainer(X_sample)

            # Handle different SHAP value formats
            shap_values_array = shap_values_obj.values

            # Handle multi-class case (3D array: samples, features, classes)
            if len(shap_values_array.shape) == 3:
                # For multi-class, take the mean across classes
                shap_values_2d = np.mean(np.abs(shap_values_array), axis=2)
            else:
                # Binary classification case (2D array: samples, features)
                shap_values_2d = np.abs(shap_values_array)

            # Calculate mean absolute SHAP importance per feature
            mean_abs_shap_per_feature = np.mean(shap_values_2d, axis=0)

            # Create comprehensive SHAP summary
            shap_summary = pd.DataFrame({
                'Feature': feature_names[:len(mean_abs_shap_per_feature)],
                'SHAP_Importance': mean_abs_shap_per_feature,
                'SHAP_Rank': range(1, len(mean_abs_shap_per_feature) + 1)
            }).sort_values('SHAP_Importance', ascending=False).reset_index(drop=True)

            # Update ranks after sorting
            shap_summary['SHAP_Rank'] = range(1, len(shap_summary) + 1)

            logger.info(f"✓ Enhanced SHAP analysis completed for {len(feature_names)} features")
            if len(shap_summary) >= 3:
                logger.info(f"  Top 3 features: {', '.join(shap_summary.head(3)['Feature'].tolist())}")

            return shap_summary

        except Exception as e:
            logger.warning(f"SHAP calculation failed: {e}")
            return None

    def train_gnn_model(self, labeled_data: pd.DataFrame, feature_list: List[str],
                       target_column: str = 'target_signal_pressure_class_h30') -> Dict[str, Any]:
        """
        Train Graph Neural Network model for feature relationship learning.

        Args:
            labeled_data: DataFrame with features and target columns
            feature_list: List of features to use for graph construction
            target_column: Target column name for prediction

        Returns:
            Dictionary containing GNN training results and model
        """
        if not self.enable_gnn or self.gnn_model is None:
            return {
                'error': 'GNN models not available or not enabled',
                'gnn_available': TORCH_GEOMETRIC_AVAILABLE,
                'gnn_enabled': self.enable_gnn
            }

        try:
            logger.info("Starting GNN model training...")

            # Prepare data for GNN
            feature_data = labeled_data[feature_list].copy()
            target_data = labeled_data[target_column].copy()

            # Handle missing values
            feature_data = feature_data.fillna(feature_data.mean())

            # Normalize features
            from sklearn.preprocessing import StandardScaler
            scaler = StandardScaler()
            feature_data_scaled = pd.DataFrame(
                scaler.fit_transform(feature_data),
                columns=feature_data.columns,
                index=feature_data.index
            )

            # Create graph from feature correlations
            graph_data = self.gnn_model.create_graph_from_features(feature_data_scaled)

            if graph_data is None:
                return {'error': 'Failed to create graph from features'}

            # Prepare target data
            if not TORCH_GEOMETRIC_AVAILABLE:
                return {'error': 'PyTorch Geometric not available'}

            import torch

            # Convert target to tensor
            target_tensor = torch.tensor(target_data.values, dtype=torch.long)
            graph_data.y = target_tensor

            # Split data for training and validation
            num_samples = len(target_data)
            train_size = int(0.8 * num_samples)

            train_mask = torch.zeros(num_samples, dtype=torch.bool)
            val_mask = torch.zeros(num_samples, dtype=torch.bool)

            train_mask[:train_size] = True
            val_mask[train_size:] = True

            graph_data.train_mask = train_mask
            graph_data.val_mask = val_mask

            # Build GNN model
            input_dim = len(feature_list)
            output_dim = len(target_data.unique())

            model = self.gnn_model.build_gnn_model(input_dim, output_dim)

            if model is None:
                return {'error': 'Failed to build GNN model'}

            # Prepare training and validation data
            train_data = graph_data.clone()
            train_data.y = train_data.y[train_mask]
            train_data.x = train_data.x[train_mask]

            val_data = graph_data.clone()
            val_data.y = val_data.y[val_mask]
            val_data.x = val_data.x[val_mask]

            # Train the model
            training_results = self.gnn_model.train_gnn(train_data, val_data)

            # Evaluate model performance
            model.eval()
            with torch.no_grad():
                # Training accuracy
                train_out = model(train_data.x, train_data.edge_index)
                train_pred = train_out.argmax(dim=1)
                train_accuracy = (train_pred == train_data.y).float().mean().item()

                # Validation accuracy
                val_out = model(val_data.x, val_data.edge_index)
                val_pred = val_out.argmax(dim=1)
                val_accuracy = (val_pred == val_data.y).float().mean().item()

            # Compile results
            gnn_results = {
                'model_type': 'GNN',
                'gnn_architecture': self.gnn_model.model_type,
                'training_results': training_results,
                'performance_metrics': {
                    'train_accuracy': train_accuracy,
                    'val_accuracy': val_accuracy,
                    'num_nodes': graph_data.num_nodes,
                    'num_edges': graph_data.num_edges,
                    'input_features': input_dim,
                    'output_classes': output_dim
                },
                'model': model,
                'graph_data': graph_data,
                'scaler': scaler,
                'feature_list': feature_list,
                'target_column': target_column
            }

            logger.info(f"GNN training completed. Train accuracy: {train_accuracy:.4f}, Val accuracy: {val_accuracy:.4f}")

            return gnn_results

        except Exception as e:
            logger.error(f"GNN training failed: {e}", exc_info=True)
            return {'error': str(e)}

    def predict_with_gnn(self, features_df: pd.DataFrame, gnn_results: Dict[str, Any]) -> Optional[np.ndarray]:
        """
        Make predictions using trained GNN model.

        Args:
            features_df: DataFrame containing features for prediction
            gnn_results: Results from GNN training containing model and preprocessing

        Returns:
            Prediction probabilities or None if failed
        """
        if not self.enable_gnn or 'model' not in gnn_results:
            logger.warning("Cannot make GNN predictions: model not available")
            return None

        try:
            # Extract components from training results
            model = gnn_results['model']
            scaler = gnn_results['scaler']
            feature_list = gnn_results['feature_list']

            # Prepare features
            feature_data = features_df[feature_list].copy()
            feature_data = feature_data.fillna(feature_data.mean())

            # Scale features using training scaler
            feature_data_scaled = pd.DataFrame(
                scaler.transform(feature_data),
                columns=feature_data.columns,
                index=feature_data.index
            )

            # Create graph for prediction
            graph_data = self.gnn_model.create_graph_from_features(feature_data_scaled)

            if graph_data is None:
                logger.warning("Failed to create graph for prediction")
                return None

            # Make predictions
            predictions = self.gnn_model.predict(graph_data)

            return predictions

        except Exception as e:
            logger.error(f"GNN prediction failed: {e}")
            return None

    def get_gnn_feature_importance(self, gnn_results: Dict[str, Any]) -> Optional[pd.DataFrame]:
        """
        Extract feature importance from trained GNN model.

        Args:
            gnn_results: Results from GNN training

        Returns:
            DataFrame with feature importance scores or None if failed
        """
        if not self.enable_gnn or 'graph_data' not in gnn_results:
            return None

        try:
            graph_data = gnn_results['graph_data']
            feature_list = gnn_results['feature_list']

            # Calculate node centrality as proxy for feature importance
            if TORCH_GEOMETRIC_AVAILABLE:
                from torch_geometric.utils import to_networkx
                import networkx as nx

                # Convert to NetworkX graph
                G = to_networkx(graph_data, to_undirected=True)

                # Calculate various centrality measures
                degree_centrality = nx.degree_centrality(G)
                betweenness_centrality = nx.betweenness_centrality(G)
                closeness_centrality = nx.closeness_centrality(G)
                eigenvector_centrality = nx.eigenvector_centrality(G, max_iter=1000)

                # Create importance DataFrame
                importance_df = pd.DataFrame({
                    'Feature': feature_list,
                    'Degree_Centrality': [degree_centrality.get(i, 0) for i in range(len(feature_list))],
                    'Betweenness_Centrality': [betweenness_centrality.get(i, 0) for i in range(len(feature_list))],
                    'Closeness_Centrality': [closeness_centrality.get(i, 0) for i in range(len(feature_list))],
                    'Eigenvector_Centrality': [eigenvector_centrality.get(i, 0) for i in range(len(feature_list))]
                })

                # Calculate composite importance score
                importance_df['Composite_Importance'] = (
                    importance_df['Degree_Centrality'] * 0.3 +
                    importance_df['Betweenness_Centrality'] * 0.3 +
                    importance_df['Closeness_Centrality'] * 0.2 +
                    importance_df['Eigenvector_Centrality'] * 0.2
                )

                # Sort by composite importance
                importance_df = importance_df.sort_values('Composite_Importance', ascending=False).reset_index(drop=True)
                importance_df['Importance_Rank'] = range(1, len(importance_df) + 1)

                logger.info(f"GNN feature importance calculated for {len(feature_list)} features")

                return importance_df

            return None

        except Exception as e:
            logger.error(f"GNN feature importance calculation failed: {e}")
            return None

    def _train_minirocket_pipeline(self, df_sample: pd.DataFrame) -> Optional[Tuple[Pipeline, float]]:
        """Train MiniRocket pipeline for time series classification."""
        logger.info("  - MiniRocket path selected. Preparing 3D panel data from sample...")

        try:
            from sktime.transformations.panel.rocket import MiniRocket
            from sklearn.linear_model import RidgeClassifierCV

            # Prepare 3D data for MiniRocket
            feature_list = [col for col in df_sample.columns if col not in ['target_signal_pressure_class_h30', 'Symbol']]
            X_3d, y, index = self._prepare_3d_data(df_sample, feature_list, lookback=10, target_col='target_signal_pressure_class_h30')

            if X_3d.shape[0] < 10:
                logger.warning("Insufficient data for MiniRocket training")
                return None

            # Create MiniRocket pipeline
            minirocket = MiniRocket(num_kernels=1000, random_state=42)
            classifier = RidgeClassifierCV(alphas=np.logspace(-3, 3, 10), cv=3)

            pipeline = Pipeline([
                ('minirocket', minirocket),
                ('classifier', classifier)
            ])

            # Train and evaluate
            pipeline.fit(X_3d, y)
            score = pipeline.score(X_3d, y)

            logger.info(f"MiniRocket training complete. Score: {score:.4f}")
            return pipeline, score

        except ImportError:
            logger.warning("MiniRocket not available. Skipping.")
            return None
        except Exception as e:
            logger.error(f"Error training MiniRocket: {e}")
            return None

    def _prepare_3d_data(self, df: pd.DataFrame, feature_list: List[str], lookback: int, target_col: str) -> Tuple[np.ndarray, np.ndarray, pd.Index]:
        """Prepare 3D data for time series models."""
        df_features = df[feature_list].fillna(0)

        # Create sliding windows
        X_list = []
        y_list = []
        indices = []

        for i in range(lookback, len(df_features)):
            X_window = df_features.iloc[i-lookback:i].values
            y_val = df[target_col].iloc[i]

            if not np.isnan(y_val):
                X_list.append(X_window)
                y_list.append(y_val)
                indices.append(df.index[i])

        X_3d = np.array(X_list)
        y = np.array(y_list)

        return X_3d, y, pd.Index(indices)


class EnsembleManager:
    """
    Manages ensemble model creation and voting strategies.
    Provides advanced ensemble methods for improved predictions.
    """

    def __init__(self, config):
        """Initialize ensemble manager."""
        self.config = config
        self.models = {}
        self.weights = {}

    def add_model(self, name: str, model, weight: float = 1.0):
        """Add model to ensemble."""
        self.models[name] = model
        self.weights[name] = weight

    def predict_ensemble(self, X: pd.DataFrame) -> np.ndarray:
        """Make ensemble predictions."""
        if not self.models:
            raise ValueError("No models in ensemble")

        predictions = []
        total_weight = sum(self.weights.values())

        for name, model in self.models.items():
            pred = model.predict_proba(X) if hasattr(model, 'predict_proba') else model.predict(X)
            weight = self.weights[name] / total_weight
            predictions.append(pred * weight)

        return np.sum(predictions, axis=0)


class HyperparameterOptimizer:
    """
    Advanced hyperparameter optimization using Optuna.
    Provides intelligent search strategies and early stopping.
    """

    def __init__(self, config):
        """Initialize hyperparameter optimizer."""
        self.config = config
        self.study = None

    def optimize(self, objective_func, n_trials: int = 100) -> Dict[str, Any]:
        """Run hyperparameter optimization."""
        import optuna

        self.study = optuna.create_study(
            direction='maximize',
            pruner=optuna.pruners.HyperbandPruner()
        )

        self.study.optimize(objective_func, n_trials=n_trials)

        return {
            'best_params': self.study.best_params,
            'best_value': self.study.best_value,
            'n_trials': len(self.study.trials)
        }


class SHAPAnalyzer:
    """
    SHAP (SHapley Additive exPlanations) analysis for model interpretability.
    Provides feature importance and interaction analysis.
    """

    def __init__(self, model, X_background: pd.DataFrame):
        """Initialize SHAP analyzer."""
        self.model = model
        self.X_background = X_background
        self.explainer = None

    def create_explainer(self):
        """Create SHAP explainer."""
        try:
            import shap
            self.explainer = shap.Explainer(self.model, self.X_background)
        except ImportError:
            raise ImportError("SHAP not available. Install with: pip install shap")

    def calculate_shap_values(self, X: pd.DataFrame) -> Dict[str, Any]:
        """Calculate SHAP values for given data."""
        if self.explainer is None:
            self.create_explainer()

        shap_values = self.explainer(X)

        return {
            'shap_values': shap_values.values,
            'base_values': shap_values.base_values,
            'feature_names': X.columns.tolist()
        }


class ModelValidator:
    """
    Comprehensive model validation and performance assessment.
    Provides various validation strategies and metrics.
    """

    def __init__(self, config):
        """Initialize model validator."""
        self.config = config

    def cross_validate(self, model, X: pd.DataFrame, y: pd.Series, cv: int = 5) -> Dict[str, Any]:
        """Perform cross-validation."""
        from sklearn.model_selection import cross_val_score, cross_validate
        from sklearn.metrics import make_scorer, f1_score

        scoring = {
            'accuracy': 'accuracy',
            'f1': make_scorer(f1_score, average='weighted'),
            'precision': 'precision_weighted',
            'recall': 'recall_weighted'
        }

        cv_results = cross_validate(model, X, y, cv=cv, scoring=scoring, return_train_score=True)

        return {
            'test_scores': {metric: scores.mean() for metric, scores in cv_results.items() if 'test_' in metric},
            'train_scores': {metric: scores.mean() for metric, scores in cv_results.items() if 'train_' in metric},
            'std_scores': {metric: scores.std() for metric, scores in cv_results.items() if 'test_' in metric}
        }

    def validate_on_holdout(self, model, X_train: pd.DataFrame, y_train: pd.Series,
                           X_holdout: pd.DataFrame, y_holdout: pd.Series) -> Dict[str, Any]:
        """Validate model on holdout set."""
        from sklearn.metrics import classification_report, confusion_matrix

        # Train on training set
        model.fit(X_train, y_train)

        # Predict on holdout
        y_pred = model.predict(X_holdout)

        return {
            'classification_report': classification_report(y_holdout, y_pred, output_dict=True),
            'confusion_matrix': confusion_matrix(y_holdout, y_pred).tolist(),
            'holdout_accuracy': (y_pred == y_holdout).mean()
        }


class TimeSeriesTransformer(nn.Module if TORCH_AVAILABLE else object):
    """
    Transformer-based neural network for time series prediction.

    Implements a transformer encoder architecture specifically designed for
    financial time series forecasting. Uses positional embeddings and
    multi-head attention to capture temporal dependencies and patterns.

    Features:
    - Multi-head self-attention mechanism
    - Positional embeddings for temporal awareness
    - Configurable encoder layers and attention heads
    - Dropout regularization for overfitting prevention
    - Flexible input/output dimensions

    Args:
        feature_size: Number of input features per timestep
        num_layers: Number of transformer encoder layers
        d_model: Dimension of the model (embedding size)
        nhead: Number of attention heads
        dim_feedforward: Dimension of feedforward network
        dropout: Dropout probability
        seq_length: Input sequence length
        prediction_length: Number of future steps to predict
    """

    def __init__(
        self,
        feature_size: int = 9,
        num_layers: int = 2,
        d_model: int = 64,
        nhead: int = 8,
        dim_feedforward: int = 256,
        dropout: float = 0.1,
        seq_length: int = 30,
        prediction_length: int = 1
    ):
        """
        Initialize TimeSeriesTransformer with specified architecture parameters.

        Args:
            feature_size: Number of input features per timestep
            num_layers: Number of transformer encoder layers
            d_model: Dimension of the model (embedding size)
            nhead: Number of attention heads
            dim_feedforward: Dimension of feedforward network
            dropout: Dropout probability for regularization
            seq_length: Expected input sequence length
            prediction_length: Number of future steps to predict
        """
        if not TORCH_AVAILABLE:
            logger.warning("PyTorch not available. TimeSeriesTransformer will not function.")
            return

        super(TimeSeriesTransformer, self).__init__()

        # Input projection layer
        self.input_fc = nn.Linear(feature_size, d_model)

        # Learnable positional embeddings
        self.pos_embedding = nn.Parameter(torch.zeros(1, seq_length, d_model))

        # Transformer encoder layers
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation="relu"
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # Output projection layer
        self.fc_out = nn.Linear(d_model, prediction_length)

        # Store parameters for reference
        self.feature_size = feature_size
        self.d_model = d_model
        self.seq_length = seq_length
        self.prediction_length = prediction_length

        logger.info(f"TimeSeriesTransformer initialized: {feature_size} features -> {d_model}d model -> {prediction_length} predictions")

    def forward(self, src: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the transformer network.

        Args:
            src: Input tensor of shape (batch_size, seq_length, feature_size)

        Returns:
            Output tensor of shape (batch_size, prediction_length)
        """
        if not TORCH_AVAILABLE:
            logger.error("PyTorch not available for TimeSeriesTransformer forward pass")
            return torch.zeros(1, self.prediction_length)

        try:
            batch_size, seq_len, _ = src.shape

            # Project input features to model dimension
            src = self.input_fc(src)

            # Add positional embeddings
            src = src + self.pos_embedding[:, :seq_len, :]

            # Transformer expects (seq_len, batch_size, d_model)
            src = src.permute(1, 0, 2)

            # Pass through transformer encoder
            encoded = self.transformer_encoder(src)

            # Use the last timestep for prediction
            last_step = encoded[-1, :, :]

            # Project to output dimension
            out = self.fc_out(last_step)

            return out

        except Exception as e:
            logger.error(f"Error in TimeSeriesTransformer forward pass: {e}")
            return torch.zeros(src.size(0), self.prediction_length)

    def get_attention_weights(self, src: torch.Tensor) -> List[torch.Tensor]:
        """
        Extract attention weights from transformer layers for interpretability.

        Args:
            src: Input tensor of shape (batch_size, seq_length, feature_size)

        Returns:
            List of attention weight tensors from each layer
        """
        if not TORCH_AVAILABLE:
            logger.error("PyTorch not available for attention weight extraction")
            return []

        try:
            attention_weights = []
            batch_size, seq_len, _ = src.shape

            # Project input features to model dimension
            src = self.input_fc(src)
            src = src + self.pos_embedding[:, :seq_len, :]
            src = src.permute(1, 0, 2)

            # Extract attention weights from each layer
            for layer in self.transformer_encoder.layers:
                # This is a simplified version - actual implementation would require
                # modifying the transformer layer to return attention weights
                # For now, return empty list
                pass

            return attention_weights

        except Exception as e:
            logger.error(f"Error extracting attention weights: {e}")
            return []

    def get_model_info(self) -> Dict[str, Any]:
        """
        Get comprehensive information about the model architecture.

        Returns:
            Dictionary containing model architecture details
        """
        try:
            total_params = sum(p.numel() for p in self.parameters())
            trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)

            return {
                'model_type': 'TimeSeriesTransformer',
                'feature_size': self.feature_size,
                'd_model': self.d_model,
                'seq_length': self.seq_length,
                'prediction_length': self.prediction_length,
                'total_parameters': total_params,
                'trainable_parameters': trainable_params,
                'num_layers': len(self.transformer_encoder.layers),
                'architecture': 'Transformer Encoder'
            }

        except Exception as e:
            logger.error(f"Error getting model info: {e}")
            return {'error': str(e)}
