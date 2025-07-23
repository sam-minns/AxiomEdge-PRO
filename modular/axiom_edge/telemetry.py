# =============================================================================
# TELEMETRY MODULE
# =============================================================================

import os
import json
import logging
from datetime import datetime
from typing import Dict, List, Any, Optional
import pandas as pd

from .config import ConfigModel
from .utils import json_serializer_default

logger = logging.getLogger(__name__)

def _recursive_sanitize(obj: Any) -> Any:
    """Recursively sanitize objects for JSON serialization."""
    if isinstance(obj, dict):
        return {k: _recursive_sanitize(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [_recursive_sanitize(item) for item in obj]
    elif isinstance(obj, (pd.Series, pd.DataFrame)):
        return obj.to_dict() if hasattr(obj, 'to_dict') else str(obj)
    elif hasattr(obj, '__dict__'):
        return _recursive_sanitize(obj.__dict__)
    elif isinstance(obj, (int, float, str, bool, type(None))):
        return obj
    else:
        return str(obj)

class TelemetryCollector:
    """
    Centralized telemetry collection system for AxiomEdge framework.
    Streams data directly to JSONL files to prevent memory accumulation
    during long backtests and provides comprehensive system monitoring.
    """
    
    def __init__(self, file_path: str):
        """
        Initialize telemetry collector.
        
        Args:
            file_path: Path to JSONL file for telemetry storage
        """
        self.file_path = file_path
        
        # Ensure directory exists
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        
        # Clear the file at the beginning of a new run
        with open(self.file_path, 'w') as f:
            f.write('')
        
        # Initialize session metadata
        self.session_start = datetime.now()
        self.session_id = self.session_start.strftime("%Y%m%d_%H%M%S")
        
        logger.info(f"Telemetry Collector initialized. Session: {self.session_id}")
        logger.info(f"Logging to: {self.file_path}")
        
        # Log session start
        self._log_event({
            "event_type": "session_start",
            "session_id": self.session_id,
            "timestamp": self.session_start.isoformat(),
            "framework_version": "2.1.1"
        })

    def log_cycle_data(self, cycle_num: int, status: str, config_snapshot: ConfigModel,
                      labeling_summary: Dict[str, Any], training_summary: Dict[str, Any],
                      backtest_metrics: Dict[str, Any], horizon_metrics: Dict[str, Any],
                      ai_notes: str = "") -> None:
        """
        Log comprehensive cycle data to telemetry.
        
        Args:
            cycle_num: Cycle number
            status: Cycle status (completed, failed, etc.)
            config_snapshot: Configuration snapshot
            labeling_summary: Data labeling summary
            training_summary: Model training summary
            backtest_metrics: Backtesting performance metrics
            horizon_metrics: Forward-looking metrics
            ai_notes: AI analysis notes
        """
        try:
            telemetry_snapshot = {
                "event_type": "cycle_complete",
                "session_id": self.session_id,
                "timestamp": datetime.now().isoformat(),
                "cycle": cycle_num,
                "status": status,
                "configuration": config_snapshot.model_dump(mode='json'),
                "data_processing": {
                    "labeling": _recursive_sanitize(labeling_summary)
                },
                "model_training": _recursive_sanitize(training_summary),
                "backtest_performance": _recursive_sanitize(backtest_metrics),
                "horizon_analysis": _recursive_sanitize(horizon_metrics),
                "ai_analysis": {
                    "notes": ai_notes,
                    "timestamp": datetime.now().isoformat()
                }
            }
            
            self._log_event(telemetry_snapshot)
            logger.info(f"Cycle {cycle_num} telemetry logged successfully")
            
        except Exception as e:
            logger.error(f"Failed to log cycle {cycle_num} telemetry: {e}")

    def log_ai_intervention(self, intervention_type: str, trigger_reason: str,
                           ai_analysis: Dict[str, Any], action_taken: Dict[str, Any],
                           performance_impact: Optional[Dict[str, Any]] = None) -> None:
        """
        Log AI intervention events.
        
        Args:
            intervention_type: Type of intervention (strategic, tactical, emergency)
            trigger_reason: Reason that triggered the intervention
            ai_analysis: AI's analysis and reasoning
            action_taken: Actions taken as a result
            performance_impact: Performance impact of the intervention
        """
        try:
            intervention_data = {
                "event_type": "ai_intervention",
                "session_id": self.session_id,
                "timestamp": datetime.now().isoformat(),
                "intervention_type": intervention_type,
                "trigger_reason": trigger_reason,
                "ai_analysis": _recursive_sanitize(ai_analysis),
                "action_taken": _recursive_sanitize(action_taken),
                "performance_impact": _recursive_sanitize(performance_impact) if performance_impact else None
            }
            
            self._log_event(intervention_data)
            logger.info(f"AI intervention logged: {intervention_type}")
            
        except Exception as e:
            logger.error(f"Failed to log AI intervention: {e}")

    def log_performance_milestone(self, milestone_type: str, metrics: Dict[str, Any],
                                 comparison_baseline: Optional[Dict[str, Any]] = None) -> None:
        """
        Log performance milestones and achievements.
        
        Args:
            milestone_type: Type of milestone (new_high, drawdown_recovery, etc.)
            metrics: Current performance metrics
            comparison_baseline: Baseline metrics for comparison
        """
        try:
            milestone_data = {
                "event_type": "performance_milestone",
                "session_id": self.session_id,
                "timestamp": datetime.now().isoformat(),
                "milestone_type": milestone_type,
                "current_metrics": _recursive_sanitize(metrics),
                "baseline_comparison": _recursive_sanitize(comparison_baseline) if comparison_baseline else None
            }
            
            self._log_event(milestone_data)
            logger.info(f"Performance milestone logged: {milestone_type}")
            
        except Exception as e:
            logger.error(f"Failed to log performance milestone: {e}")

    def log_system_health(self, component: str, health_status: str,
                         metrics: Dict[str, Any], alerts: List[str] = None) -> None:
        """
        Log system health and monitoring data.
        
        Args:
            component: System component name
            health_status: Health status (healthy, warning, critical)
            metrics: Health metrics
            alerts: List of active alerts
        """
        try:
            health_data = {
                "event_type": "system_health",
                "session_id": self.session_id,
                "timestamp": datetime.now().isoformat(),
                "component": component,
                "health_status": health_status,
                "metrics": _recursive_sanitize(metrics),
                "alerts": alerts or []
            }
            
            self._log_event(health_data)
            
            if health_status != "healthy":
                logger.warning(f"System health alert - {component}: {health_status}")
            
        except Exception as e:
            logger.error(f"Failed to log system health: {e}")

    def log_feature_importance_evolution(self, cycle_num: int, feature_rankings: Dict[str, Any],
                                       changes_from_previous: Optional[Dict[str, Any]] = None) -> None:
        """
        Log evolution of feature importance over time.
        
        Args:
            cycle_num: Current cycle number
            feature_rankings: Current feature importance rankings
            changes_from_previous: Changes from previous cycle
        """
        try:
            feature_data = {
                "event_type": "feature_evolution",
                "session_id": self.session_id,
                "timestamp": datetime.now().isoformat(),
                "cycle": cycle_num,
                "feature_rankings": _recursive_sanitize(feature_rankings),
                "changes_from_previous": _recursive_sanitize(changes_from_previous) if changes_from_previous else None
            }
            
            self._log_event(feature_data)
            logger.info(f"Feature importance evolution logged for cycle {cycle_num}")
            
        except Exception as e:
            logger.error(f"Failed to log feature evolution: {e}")

    def log_regime_detection(self, detected_regime: str, confidence: float,
                           regime_characteristics: Dict[str, Any],
                           strategy_adjustments: Optional[Dict[str, Any]] = None) -> None:
        """
        Log market regime detection and strategy adjustments.
        
        Args:
            detected_regime: Detected market regime
            confidence: Detection confidence
            regime_characteristics: Characteristics of the detected regime
            strategy_adjustments: Strategy adjustments made
        """
        try:
            regime_data = {
                "event_type": "regime_detection",
                "session_id": self.session_id,
                "timestamp": datetime.now().isoformat(),
                "detected_regime": detected_regime,
                "confidence": confidence,
                "characteristics": _recursive_sanitize(regime_characteristics),
                "strategy_adjustments": _recursive_sanitize(strategy_adjustments) if strategy_adjustments else None
            }
            
            self._log_event(regime_data)
            logger.info(f"Regime detection logged: {detected_regime} (confidence: {confidence:.3f})")
            
        except Exception as e:
            logger.error(f"Failed to log regime detection: {e}")

    def log_genetic_programming_evolution(self, generation: int, population_stats: Dict[str, Any],
                                        best_individual: Dict[str, Any],
                                        evolution_metrics: Dict[str, Any]) -> None:
        """
        Log genetic programming evolution progress.
        
        Args:
            generation: Current generation number
            population_stats: Population statistics
            best_individual: Best individual in current generation
            evolution_metrics: Evolution performance metrics
        """
        try:
            gp_data = {
                "event_type": "genetic_evolution",
                "session_id": self.session_id,
                "timestamp": datetime.now().isoformat(),
                "generation": generation,
                "population_stats": _recursive_sanitize(population_stats),
                "best_individual": _recursive_sanitize(best_individual),
                "evolution_metrics": _recursive_sanitize(evolution_metrics)
            }
            
            self._log_event(gp_data)
            
        except Exception as e:
            logger.error(f"Failed to log genetic programming evolution: {e}")

    def _log_event(self, event_data: Dict[str, Any]) -> None:
        """
        Write event data to JSONL file.
        
        Args:
            event_data: Event data to log
        """
        try:
            with open(self.file_path, 'a') as f:
                json.dump(event_data, f, default=json_serializer_default, separators=(',', ':'))
                f.write('\n')
        except Exception as e:
            logger.error(f"Failed to write telemetry event: {e}")

    def get_session_summary(self) -> Dict[str, Any]:
        """
        Get summary of current telemetry session.
        
        Returns:
            Session summary statistics
        """
        try:
            event_counts = {}
            total_events = 0
            
            with open(self.file_path, 'r') as f:
                for line in f:
                    if line.strip():
                        event = json.loads(line)
                        event_type = event.get('event_type', 'unknown')
                        event_counts[event_type] = event_counts.get(event_type, 0) + 1
                        total_events += 1
            
            session_duration = datetime.now() - self.session_start
            
            return {
                "session_id": self.session_id,
                "session_start": self.session_start.isoformat(),
                "session_duration_seconds": session_duration.total_seconds(),
                "total_events": total_events,
                "event_counts": event_counts,
                "file_path": self.file_path
            }
            
        except Exception as e:
            logger.error(f"Failed to generate session summary: {e}")
            return {"error": str(e)}

    def export_session_data(self, output_format: str = "json") -> Optional[str]:
        """
        Export session data in specified format.
        
        Args:
            output_format: Export format (json, csv, parquet)
            
        Returns:
            Path to exported file or None if failed
        """
        try:
            base_path = self.file_path.replace('.jsonl', '')
            
            if output_format.lower() == "json":
                # Already in JSONL format
                return self.file_path
            
            elif output_format.lower() == "csv":
                # Convert to CSV
                events = []
                with open(self.file_path, 'r') as f:
                    for line in f:
                        if line.strip():
                            events.append(json.loads(line))
                
                if events:
                    df = pd.json_normalize(events)
                    csv_path = f"{base_path}.csv"
                    df.to_csv(csv_path, index=False)
                    return csv_path
            
            elif output_format.lower() == "parquet":
                # Convert to Parquet
                events = []
                with open(self.file_path, 'r') as f:
                    for line in f:
                        if line.strip():
                            events.append(json.loads(line))
                
                if events:
                    df = pd.json_normalize(events)
                    parquet_path = f"{base_path}.parquet"
                    df.to_parquet(parquet_path, index=False)
                    return parquet_path
            
            return None
            
        except Exception as e:
            logger.error(f"Failed to export session data: {e}")
            return None

    def close_session(self) -> None:
        """Close the telemetry session and log final summary."""
        try:
            session_summary = self.get_session_summary()
            
            final_event = {
                "event_type": "session_end",
                "session_id": self.session_id,
                "timestamp": datetime.now().isoformat(),
                "session_summary": session_summary
            }
            
            self._log_event(final_event)
            logger.info(f"Telemetry session {self.session_id} closed")
            
        except Exception as e:
            logger.error(f"Failed to close telemetry session: {e}")

class TelemetryAnalyzer:
    """Analyze telemetry data for insights and patterns."""
    
    def __init__(self, telemetry_file: str):
        self.telemetry_file = telemetry_file
    
    def analyze_cycle_performance_trends(self) -> Dict[str, Any]:
        """Analyze performance trends across cycles."""
        try:
            cycles = []
            
            with open(self.telemetry_file, 'r') as f:
                for line in f:
                    if line.strip():
                        event = json.loads(line)
                        if event.get('event_type') == 'cycle_complete':
                            cycles.append(event)
            
            if not cycles:
                return {"error": "No cycle data found"}
            
            # Extract performance metrics
            performance_data = []
            for cycle in cycles:
                metrics = cycle.get('backtest_performance', {})
                performance_data.append({
                    'cycle': cycle.get('cycle'),
                    'sharpe_ratio': metrics.get('sharpe_ratio', 0),
                    'total_return': metrics.get('net_profit_pct', 0),
                    'max_drawdown': metrics.get('max_drawdown_pct', 0),
                    'win_rate': metrics.get('win_rate', 0)
                })
            
            df = pd.DataFrame(performance_data)
            
            return {
                "total_cycles": len(cycles),
                "avg_sharpe": df['sharpe_ratio'].mean(),
                "avg_return": df['total_return'].mean(),
                "avg_drawdown": df['max_drawdown'].mean(),
                "avg_win_rate": df['win_rate'].mean(),
                "performance_trend": df['sharpe_ratio'].diff().mean(),  # Trend direction
                "consistency": df['sharpe_ratio'].std()  # Lower is more consistent
            }
            
        except Exception as e:
            logger.error(f"Failed to analyze cycle performance trends: {e}")
            return {"error": str(e)}
    
    def get_ai_intervention_effectiveness(self) -> Dict[str, Any]:
        """Analyze effectiveness of AI interventions."""
        try:
            interventions = []
            
            with open(self.telemetry_file, 'r') as f:
                for line in f:
                    if line.strip():
                        event = json.loads(line)
                        if event.get('event_type') == 'ai_intervention':
                            interventions.append(event)
            
            if not interventions:
                return {"error": "No AI intervention data found"}
            
            # Analyze intervention types and outcomes
            intervention_types = {}
            for intervention in interventions:
                int_type = intervention.get('intervention_type', 'unknown')
                intervention_types[int_type] = intervention_types.get(int_type, 0) + 1
            
            return {
                "total_interventions": len(interventions),
                "intervention_types": intervention_types,
                "avg_interventions_per_session": len(interventions)  # Would need session count for accuracy
            }
            
        except Exception as e:
            logger.error(f"Failed to analyze AI intervention effectiveness: {e}")
            return {"error": str(e)}
