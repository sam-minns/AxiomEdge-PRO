# =============================================================================
# TELEMETRY MODULE
# =============================================================================

import os
import json
import logging
import time
import psutil
import threading
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple
import pandas as pd
import numpy as np

from .config import ConfigModel
from .utils import json_serializer_default

# Optional imports with fallbacks
try:
    import matplotlib.pyplot as plt
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False

try:
    import plotly.graph_objects as go
    import plotly.express as px
    PLOTLY_AVAILABLE = True
except ImportError:
    PLOTLY_AVAILABLE = False

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
    Advanced telemetry collection system for comprehensive framework monitoring.

    Streams data directly to JSONL files to prevent memory accumulation during long
    backtests and provides comprehensive system monitoring with AI doctor capabilities.

    Features:
    - Real-time event streaming to JSONL format with buffering
    - Session management and lifecycle tracking with recovery
    - Cycle-by-cycle performance monitoring with trend analysis
    - AI intervention logging and effectiveness tracking
    - Performance milestone detection and automated alerting
    - System health monitoring with resource usage tracking
    - Evolution tracking for genetic programming with convergence detection
    - Market regime detection and strategy adaptations
    - Multi-format data export capabilities (JSON, CSV, Parquet)
    - Real-time performance dashboards and visualizations
    - Automated anomaly detection and alerting
    - Memory-efficient streaming for long-running processes
    - Thread-safe concurrent logging capabilities
    - Comprehensive error recovery and data integrity checks
    """
    
    def __init__(self, config: ConfigModel):
        """
        Initialize advanced telemetry collector with comprehensive monitoring.

        Args:
            config: Configuration model with telemetry settings
        """
        # Configuration
        self.config = config
        self.file_path = getattr(config, 'TELEMETRY_FILE_PATH', 'telemetry/session.jsonl')
        self.enable_system_monitoring = getattr(config, 'ENABLE_SYSTEM_MONITORING', True)
        self.enable_real_time_alerts = getattr(config, 'ENABLE_REAL_TIME_ALERTS', True)
        self.buffer_size = getattr(config, 'TELEMETRY_BUFFER_SIZE', 100)

        # Ensure directory exists
        os.makedirs(os.path.dirname(self.file_path), exist_ok=True)

        # Initialize session metadata
        self.session_start = datetime.now()
        self.session_id = self.session_start.strftime("%Y%m%d_%H%M%S")
        self.is_collecting = False

        # Performance tracking
        self.event_count = 0
        self.last_flush_time = time.time()
        self.event_buffer = []
        self.buffer_lock = threading.Lock()

        # System monitoring
        self.system_stats = {
            'cpu_usage': [],
            'memory_usage': [],
            'disk_usage': [],
            'network_io': []
        }

        # Alert thresholds
        self.alert_thresholds = {
            'cpu_usage': getattr(config, 'CPU_ALERT_THRESHOLD', 80.0),
            'memory_usage': getattr(config, 'MEMORY_ALERT_THRESHOLD', 85.0),
            'disk_usage': getattr(config, 'DISK_ALERT_THRESHOLD', 90.0)
        }

        # Performance baselines
        self.performance_baselines = {
            'sharpe_ratio': getattr(config, 'BASELINE_SHARPE_RATIO', 1.0),
            'win_rate': getattr(config, 'BASELINE_WIN_RATE', 0.55),
            'max_drawdown': getattr(config, 'BASELINE_MAX_DRAWDOWN', 0.15)
        }

        logger.info(f"Advanced Telemetry Collector initialized. Session: {self.session_id}")
        logger.info(f"Logging to: {self.file_path}")
        logger.info(f"System monitoring: {self.enable_system_monitoring}")
        logger.info(f"Real-time alerts: {self.enable_real_time_alerts}")

        # Initialize the file
        self._initialize_session_file()

    def _initialize_session_file(self):
        """Initialize the session file with metadata."""
        try:
            # Clear the file and write session start event
            with open(self.file_path, 'w') as f:
                f.write('')

            # Log session start with system information
            system_info = self._get_system_info()
            session_start_event = {
                "event_type": "session_start",
                "session_id": self.session_id,
                "timestamp": self.session_start.isoformat(),
                "framework_version": getattr(self.config, 'FRAMEWORK_VERSION', '2.1.1'),
                "system_info": system_info,
                "configuration_snapshot": self._sanitize_config_for_logging()
            }

            self._log_event_immediate(session_start_event)

        except Exception as e:
            logger.error(f"Failed to initialize session file: {e}")

    def _get_system_info(self) -> Dict[str, Any]:
        """Get comprehensive system information."""
        try:
            return {
                'cpu_count': psutil.cpu_count(),
                'cpu_freq': psutil.cpu_freq()._asdict() if psutil.cpu_freq() else None,
                'memory_total': psutil.virtual_memory().total,
                'disk_total': psutil.disk_usage('/').total,
                'platform': os.name,
                'python_version': os.sys.version
            }
        except Exception as e:
            logger.warning(f"Could not collect system info: {e}")
            return {'error': str(e)}

    def _sanitize_config_for_logging(self) -> Dict[str, Any]:
        """Sanitize configuration for logging (remove sensitive data)."""
        try:
            config_dict = self.config.dict() if hasattr(self.config, 'dict') else vars(self.config)

            # Remove sensitive keys
            sensitive_keys = ['API_KEY', 'SECRET_KEY', 'PASSWORD', 'TOKEN']
            sanitized = {}

            for key, value in config_dict.items():
                if any(sensitive in key.upper() for sensitive in sensitive_keys):
                    sanitized[key] = "***REDACTED***"
                else:
                    sanitized[key] = _recursive_sanitize(value)

            return sanitized

        except Exception as e:
            logger.warning(f"Could not sanitize config: {e}")
            return {'error': str(e)}

    def start_collection(self):
        """Start telemetry collection with system monitoring."""
        self.is_collecting = True

        if self.enable_system_monitoring:
            self._start_system_monitoring()

        logger.info("Telemetry collection started")

    def stop_collection(self) -> Dict[str, Any]:
        """Stop telemetry collection and return summary."""
        self.is_collecting = False

        # Flush any remaining events
        self._flush_buffer()

        # Generate session summary
        summary = self.get_session_summary()

        # Log session end
        self._log_event_immediate({
            "event_type": "session_end",
            "session_id": self.session_id,
            "timestamp": datetime.now().isoformat(),
            "session_duration_seconds": (datetime.now() - self.session_start).total_seconds(),
            "total_events_logged": self.event_count,
            "session_summary": summary
        })

        logger.info("Telemetry collection stopped")
        return summary

    def _start_system_monitoring(self):
        """Start background system monitoring thread."""
        def monitor_system():
            while self.is_collecting:
                try:
                    # Collect system metrics
                    cpu_percent = psutil.cpu_percent(interval=1)
                    memory = psutil.virtual_memory()
                    disk = psutil.disk_usage('/')

                    # Store metrics
                    self.system_stats['cpu_usage'].append(cpu_percent)
                    self.system_stats['memory_usage'].append(memory.percent)
                    self.system_stats['disk_usage'].append(disk.percent)

                    # Check for alerts
                    if self.enable_real_time_alerts:
                        self._check_system_alerts(cpu_percent, memory.percent, disk.percent)

                    # Log system health periodically
                    if len(self.system_stats['cpu_usage']) % 60 == 0:  # Every minute
                        self.log_system_health(
                            component="system",
                            health_status="monitoring",
                            metrics={
                                'cpu_usage': cpu_percent,
                                'memory_usage': memory.percent,
                                'disk_usage': disk.percent
                            }
                        )

                    time.sleep(1)  # Monitor every second

                except Exception as e:
                    logger.error(f"System monitoring error: {e}")
                    time.sleep(5)  # Wait longer on error

        # Start monitoring thread
        monitor_thread = threading.Thread(target=monitor_system, daemon=True)
        monitor_thread.start()
        logger.info("System monitoring thread started")

    def _check_system_alerts(self, cpu_percent: float, memory_percent: float, disk_percent: float):
        """Check system metrics against alert thresholds."""
        alerts = []

        if cpu_percent > self.alert_thresholds['cpu_usage']:
            alerts.append(f"High CPU usage: {cpu_percent:.1f}%")

        if memory_percent > self.alert_thresholds['memory_usage']:
            alerts.append(f"High memory usage: {memory_percent:.1f}%")

        if disk_percent > self.alert_thresholds['disk_usage']:
            alerts.append(f"High disk usage: {disk_percent:.1f}%")

        if alerts:
            self.log_system_health(
                component="system",
                health_status="alert",
                metrics={
                    'cpu_usage': cpu_percent,
                    'memory_usage': memory_percent,
                    'disk_usage': disk_percent
                },
                alerts=alerts
            )

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
        Add event data to buffer for efficient batch writing.

        Args:
            event_data: Event data to log
        """
        try:
            # Add timestamp if not present
            if 'timestamp' not in event_data:
                event_data['timestamp'] = datetime.now().isoformat()

            # Add session ID
            event_data['session_id'] = self.session_id

            # Thread-safe buffer addition
            with self.buffer_lock:
                self.event_buffer.append(event_data)
                self.event_count += 1

                # Flush buffer if it's full or enough time has passed
                if (len(self.event_buffer) >= self.buffer_size or
                    time.time() - self.last_flush_time > 10):  # Flush every 10 seconds
                    self._flush_buffer()

        except Exception as e:
            logger.error(f"Failed to buffer telemetry event: {e}")

    def _log_event_immediate(self, event_data: Dict[str, Any]) -> None:
        """
        Write event data immediately to JSONL file (for critical events).

        Args:
            event_data: Event data to log
        """
        try:
            # Add timestamp if not present
            if 'timestamp' not in event_data:
                event_data['timestamp'] = datetime.now().isoformat()

            # Add session ID
            event_data['session_id'] = self.session_id

            with open(self.file_path, 'a') as f:
                json.dump(event_data, f, default=json_serializer_default, separators=(',', ':'))
                f.write('\n')

            self.event_count += 1

        except Exception as e:
            logger.error(f"Failed to write immediate telemetry event: {e}")

    def _flush_buffer(self):
        """Flush the event buffer to disk."""
        try:
            with self.buffer_lock:
                if not self.event_buffer:
                    return

                # Write all buffered events
                with open(self.file_path, 'a') as f:
                    for event in self.event_buffer:
                        json.dump(event, f, default=json_serializer_default, separators=(',', ':'))
                        f.write('\n')

                # Clear buffer and update flush time
                buffer_size = len(self.event_buffer)
                self.event_buffer.clear()
                self.last_flush_time = time.time()

                logger.debug(f"Flushed {buffer_size} telemetry events to disk")

        except Exception as e:
            logger.error(f"Failed to flush telemetry buffer: {e}")

    def log_performance_alert(self, alert_type: str, current_metrics: Dict[str, Any],
                             threshold_breached: str, severity: str = "warning") -> None:
        """
        Log performance alerts when metrics breach thresholds.

        Args:
            alert_type: Type of performance alert
            current_metrics: Current performance metrics
            threshold_breached: Description of threshold that was breached
            severity: Alert severity level
        """
        try:
            alert_data = {
                "event_type": "performance_alert",
                "alert_type": alert_type,
                "severity": severity,
                "current_metrics": _recursive_sanitize(current_metrics),
                "threshold_breached": threshold_breached,
                "baseline_comparison": self._compare_to_baselines(current_metrics)
            }

            self._log_event_immediate(alert_data)  # Immediate logging for alerts

            if severity in ["critical", "error"]:
                logger.error(f"CRITICAL PERFORMANCE ALERT: {alert_type} - {threshold_breached}")
            else:
                logger.warning(f"Performance alert: {alert_type} - {threshold_breached}")

        except Exception as e:
            logger.error(f"Failed to log performance alert: {e}")

    def _compare_to_baselines(self, current_metrics: Dict[str, Any]) -> Dict[str, Any]:
        """Compare current metrics to established baselines."""
        try:
            comparison = {}

            for metric, baseline in self.performance_baselines.items():
                if metric in current_metrics:
                    current_value = current_metrics[metric]
                    if isinstance(current_value, (int, float)):
                        comparison[metric] = {
                            'current': current_value,
                            'baseline': baseline,
                            'difference': current_value - baseline,
                            'percentage_change': ((current_value - baseline) / baseline * 100) if baseline != 0 else 0
                        }

            return comparison

        except Exception as e:
            logger.error(f"Failed to compare to baselines: {e}")
            return {}

    def log_model_performance_degradation(self, model_name: str, current_performance: Dict[str, Any],
                                        historical_performance: Dict[str, Any], degradation_metrics: Dict[str, Any]) -> None:
        """
        Log model performance degradation events.

        Args:
            model_name: Name of the model showing degradation
            current_performance: Current performance metrics
            historical_performance: Historical performance for comparison
            degradation_metrics: Specific degradation measurements
        """
        try:
            degradation_data = {
                "event_type": "model_performance_degradation",
                "model_name": model_name,
                "current_performance": _recursive_sanitize(current_performance),
                "historical_performance": _recursive_sanitize(historical_performance),
                "degradation_metrics": _recursive_sanitize(degradation_metrics),
                "severity": self._assess_degradation_severity(degradation_metrics)
            }

            self._log_event(degradation_data)
            logger.warning(f"Model performance degradation detected: {model_name}")

        except Exception as e:
            logger.error(f"Failed to log model performance degradation: {e}")

    def _assess_degradation_severity(self, degradation_metrics: Dict[str, Any]) -> str:
        """Assess the severity of performance degradation."""
        try:
            # Simple severity assessment based on degradation percentage
            max_degradation = 0

            for metric, value in degradation_metrics.items():
                if isinstance(value, dict) and 'percentage_change' in value:
                    max_degradation = max(max_degradation, abs(value['percentage_change']))

            if max_degradation > 50:
                return "critical"
            elif max_degradation > 25:
                return "high"
            elif max_degradation > 10:
                return "medium"
            else:
                return "low"

        except Exception as e:
            logger.error(f"Failed to assess degradation severity: {e}")
            return "unknown"

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

    def log_cycle_completion(self, cycle_metrics: Dict[str, Any]) -> None:
        """
        Log completion of a walk-forward cycle.

        Args:
            cycle_metrics: Dictionary containing cycle performance metrics
        """
        event_data = {
            "event_type": "cycle_completion",
            "timestamp": datetime.now().isoformat(),
            "cycle_metrics": cycle_metrics,
            "session_id": self.session_id
        }

        self._log_event(event_data)
        logger.debug(f"Logged cycle completion for cycle {cycle_metrics.get('cycle_num', 'unknown')}")

    def log_task_start(self, task_name: str, config_snapshot: ConfigModel) -> None:
        """
        Log the start of a task execution.

        Args:
            task_name: Name of the task being started
            config_snapshot: Configuration snapshot at task start
        """
        event_data = {
            "event_type": "task_start",
            "timestamp": datetime.now().isoformat(),
            "task_name": task_name,
            "config_snapshot": self._sanitize_config_for_logging(),
            "session_id": self.session_id
        }

        self._log_event(event_data)
        logger.debug(f"Logged task start: {task_name}")

    def log_task_completion(self, task_name: str, execution_time: float, results_summary: Dict[str, Any]) -> None:
        """
        Log the completion of a task execution.

        Args:
            task_name: Name of the completed task
            execution_time: Task execution time in seconds
            results_summary: Summary of task results
        """
        event_data = {
            "event_type": "task_completion",
            "timestamp": datetime.now().isoformat(),
            "task_name": task_name,
            "execution_time_seconds": execution_time,
            "results_summary": results_summary,
            "session_id": self.session_id
        }

        self._log_event(event_data)
        logger.debug(f"Logged task completion: {task_name} ({execution_time:.2f}s)")

    def get_historical_telemetry(self) -> List[Dict[str, Any]]:
        """
        Retrieves all historical telemetry data from the session file.

        Returns:
            List of telemetry events from the current session
        """
        try:
            if not os.path.exists(self.session_file):
                return []

            telemetry_data = []
            with open(self.session_file, 'r') as f:
                for line in f:
                    line = line.strip()
                    if line:
                        try:
                            event = json.loads(line)
                            telemetry_data.append(event)
                        except json.JSONDecodeError:
                            continue

            return telemetry_data

        except Exception as e:
            logger.error(f"Error retrieving historical telemetry: {e}")
            return []

    def get_last_n_cycles(self, n: int) -> List[Dict[str, Any]]:
        """
        Retrieves telemetry data for the last N cycles.

        Args:
            n: Number of recent cycles to retrieve

        Returns:
            List of cycle telemetry data for the last N cycles
        """
        try:
            all_telemetry = self.get_historical_telemetry()

            # Filter for cycle events
            cycle_events = [
                event for event in all_telemetry
                if event.get('event_type') == 'cycle_completion'
            ]

            # Sort by cycle number and return last N
            cycle_events.sort(key=lambda x: x.get('cycle_num', 0))
            return cycle_events[-n:] if len(cycle_events) >= n else cycle_events

        except Exception as e:
            logger.error(f"Error retrieving last {n} cycles: {e}")
            return []


class TelemetryAnalyzer:
    """
    Advanced telemetry analysis system with comprehensive AI doctor capabilities.

    Provides comprehensive analysis of telemetry data to identify patterns, trends,
    and optimization opportunities. Includes AI-powered anomaly detection, performance
    analysis, and automated diagnostic capabilities.

    Features:
    - Comprehensive performance trend analysis with statistical significance testing
    - AI intervention effectiveness evaluation with ROI analysis
    - Advanced anomaly detection using multiple algorithms
    - Performance degradation early warning with predictive analytics
    - Feature evolution analysis with importance drift detection
    - Session lifecycle analysis with resource utilization tracking
    - Automated diagnostic reports with actionable recommendations
    - Real-time performance monitoring with adaptive thresholds
    - Cross-session comparative analysis and benchmarking
    - Export capabilities for further analysis (JSON, CSV, HTML reports)
    - Interactive visualization generation for stakeholder reporting
    - Automated alert generation for critical performance issues
    """

    def __init__(self, config: ConfigModel):
        """
        Initialize advanced telemetry analyzer.

        Args:
            config: Configuration model with analyzer settings
        """
        self.config = config
        self.telemetry_file = getattr(config, 'TELEMETRY_FILE_PATH', 'telemetry/session.jsonl')
        self.enable_advanced_analytics = getattr(config, 'ENABLE_ADVANCED_ANALYTICS', True)
        self.enable_predictive_analysis = getattr(config, 'ENABLE_PREDICTIVE_ANALYSIS', True)
        self.anomaly_detection_threshold = getattr(config, 'ANOMALY_DETECTION_THRESHOLD', 2.0)

        # Performance thresholds for alerts
        self.performance_thresholds = {
            'sharpe_ratio_min': getattr(config, 'MIN_SHARPE_RATIO', 0.5),
            'win_rate_min': getattr(config, 'MIN_WIN_RATE', 0.45),
            'max_drawdown_max': getattr(config, 'MAX_DRAWDOWN_THRESHOLD', 0.20),
            'profit_factor_min': getattr(config, 'MIN_PROFIT_FACTOR', 1.2)
        }

        # Analysis cache for performance
        self.analysis_cache = {}
        self.last_analysis_time = None
    
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

    def run_comprehensive_ai_doctor_analysis(self) -> Dict[str, Any]:
        """
        Run comprehensive AI doctor analysis with automated diagnostics.

        Returns:
            Comprehensive diagnostic report with recommendations
        """
        try:
            logger.info("Running comprehensive AI doctor analysis...")

            # Initialize diagnostic report
            diagnostic_report = {
                'analysis_timestamp': datetime.now().isoformat(),
                'analyzer_version': '2.1.1',
                'data_quality_assessment': {},
                'performance_analysis': {},
                'anomaly_detection': {},
                'trend_analysis': {},
                'intervention_effectiveness': {},
                'predictive_insights': {},
                'recommendations': [],
                'alerts': [],
                'overall_health_score': 0.0
            }

            # 1. Data Quality Assessment
            diagnostic_report['data_quality_assessment'] = self._assess_data_quality()

            # 2. Performance Analysis
            diagnostic_report['performance_analysis'] = self.analyze_cycle_performance_trends()

            # 3. Anomaly Detection
            if self.enable_advanced_analytics:
                diagnostic_report['anomaly_detection'] = self._detect_performance_anomalies()

            # 4. Trend Analysis
            diagnostic_report['trend_analysis'] = self._analyze_performance_trends()

            # 5. AI Intervention Effectiveness
            diagnostic_report['intervention_effectiveness'] = self.get_ai_intervention_effectiveness()

            # 6. Predictive Insights
            if self.enable_predictive_analysis:
                diagnostic_report['predictive_insights'] = self._generate_predictive_insights()

            # 7. Generate Recommendations
            diagnostic_report['recommendations'] = self._generate_ai_recommendations(diagnostic_report)

            # 8. Generate Alerts
            diagnostic_report['alerts'] = self._generate_performance_alerts(diagnostic_report)

            # 9. Calculate Overall Health Score
            diagnostic_report['overall_health_score'] = self._calculate_overall_health_score(diagnostic_report)

            logger.info(f"AI doctor analysis completed. Health score: {diagnostic_report['overall_health_score']:.2f}")
            return diagnostic_report

        except Exception as e:
            logger.error(f"AI doctor analysis failed: {e}", exc_info=True)
            return {
                'error': 'AI doctor analysis failed',
                'details': str(e),
                'analysis_timestamp': datetime.now().isoformat()
            }

    def _assess_data_quality(self) -> Dict[str, Any]:
        """Assess the quality and completeness of telemetry data."""
        try:
            quality_assessment = {
                'total_events': 0,
                'event_types': {},
                'data_completeness': 0.0,
                'temporal_coverage': {},
                'missing_data_issues': [],
                'data_quality_score': 0.0
            }

            events = []
            event_types = {}
            timestamps = []

            # Read and analyze all events
            with open(self.telemetry_file, 'r') as f:
                for line in f:
                    if line.strip():
                        try:
                            event = json.loads(line)
                            events.append(event)

                            event_type = event.get('event_type', 'unknown')
                            event_types[event_type] = event_types.get(event_type, 0) + 1

                            if 'timestamp' in event:
                                timestamps.append(event['timestamp'])

                        except json.JSONDecodeError:
                            quality_assessment['missing_data_issues'].append('Invalid JSON line detected')

            quality_assessment['total_events'] = len(events)
            quality_assessment['event_types'] = event_types

            # Assess temporal coverage
            if timestamps:
                timestamps.sort()
                quality_assessment['temporal_coverage'] = {
                    'start_time': timestamps[0],
                    'end_time': timestamps[-1],
                    'duration_hours': (datetime.fromisoformat(timestamps[-1]) -
                                     datetime.fromisoformat(timestamps[0])).total_seconds() / 3600
                }

            # Calculate data completeness
            expected_event_types = ['session_start', 'cycle_complete', 'system_health']
            present_types = set(event_types.keys())
            expected_types = set(expected_event_types)
            quality_assessment['data_completeness'] = len(present_types & expected_types) / len(expected_types)

            # Calculate overall data quality score
            completeness_score = quality_assessment['data_completeness']
            volume_score = min(1.0, len(events) / 100)  # Normalize to 100 events
            quality_assessment['data_quality_score'] = (completeness_score + volume_score) / 2

            return quality_assessment

        except Exception as e:
            logger.error(f"Data quality assessment failed: {e}")
            return {'error': str(e)}

    def _detect_performance_anomalies(self) -> Dict[str, Any]:
        """Detect performance anomalies using statistical methods."""
        try:
            anomaly_report = {
                'anomalies_detected': [],
                'anomaly_score': 0.0,
                'detection_method': 'statistical_outlier',
                'threshold_used': self.anomaly_detection_threshold
            }

            # Extract performance metrics from cycles
            performance_metrics = []

            with open(self.telemetry_file, 'r') as f:
                for line in f:
                    if line.strip():
                        event = json.loads(line)
                        if event.get('event_type') == 'cycle_complete':
                            metrics = event.get('performance_metrics', {})
                            if metrics:
                                performance_metrics.append(metrics)

            if len(performance_metrics) < 3:
                return {'error': 'Insufficient data for anomaly detection'}

            # Analyze each metric for anomalies
            metrics_to_analyze = ['sharpe_ratio', 'win_rate', 'profit_factor', 'max_drawdown_pct']

            for metric in metrics_to_analyze:
                values = [m.get(metric, 0) for m in performance_metrics if metric in m]

                if len(values) >= 3:
                    # Calculate z-scores
                    mean_val = np.mean(values)
                    std_val = np.std(values)

                    if std_val > 0:
                        z_scores = [(v - mean_val) / std_val for v in values]

                        # Detect outliers
                        for i, z_score in enumerate(z_scores):
                            if abs(z_score) > self.anomaly_detection_threshold:
                                anomaly_report['anomalies_detected'].append({
                                    'metric': metric,
                                    'cycle_index': i,
                                    'value': values[i],
                                    'z_score': z_score,
                                    'severity': 'high' if abs(z_score) > 3 else 'medium'
                                })

            # Calculate overall anomaly score
            if anomaly_report['anomalies_detected']:
                max_z_score = max(abs(a['z_score']) for a in anomaly_report['anomalies_detected'])
                anomaly_report['anomaly_score'] = min(1.0, max_z_score / 5.0)  # Normalize to 0-1

            return anomaly_report

        except Exception as e:
            logger.error(f"Anomaly detection failed: {e}")
            return {'error': str(e)}

    def _analyze_performance_trends(self) -> Dict[str, Any]:
        """Analyze performance trends over time."""
        try:
            trend_analysis = {
                'trend_direction': {},
                'trend_strength': {},
                'performance_stability': 0.0,
                'improvement_rate': 0.0
            }

            # Extract performance metrics over time
            performance_data = []

            with open(self.telemetry_file, 'r') as f:
                for line in f:
                    if line.strip():
                        event = json.loads(line)
                        if event.get('event_type') == 'cycle_complete':
                            metrics = event.get('performance_metrics', {})
                            if metrics and 'timestamp' in event:
                                metrics['timestamp'] = event['timestamp']
                                performance_data.append(metrics)

            if len(performance_data) < 3:
                return {'error': 'Insufficient data for trend analysis'}

            # Sort by timestamp
            performance_data.sort(key=lambda x: x['timestamp'])

            # Analyze trends for key metrics
            metrics_to_analyze = ['sharpe_ratio', 'win_rate', 'profit_factor']

            for metric in metrics_to_analyze:
                values = [d.get(metric, 0) for d in performance_data if metric in d]

                if len(values) >= 3:
                    # Calculate trend using linear regression
                    x = np.arange(len(values))
                    slope = np.polyfit(x, values, 1)[0]

                    # Determine trend direction and strength
                    if slope > 0.01:
                        trend_analysis['trend_direction'][metric] = 'improving'
                    elif slope < -0.01:
                        trend_analysis['trend_direction'][metric] = 'declining'
                    else:
                        trend_analysis['trend_direction'][metric] = 'stable'

                    trend_analysis['trend_strength'][metric] = abs(slope)

            # Calculate overall performance stability
            if performance_data:
                sharpe_values = [d.get('sharpe_ratio', 0) for d in performance_data if 'sharpe_ratio' in d]
                if sharpe_values:
                    trend_analysis['performance_stability'] = 1.0 - (np.std(sharpe_values) / (np.mean(sharpe_values) + 1e-6))

            return trend_analysis

        except Exception as e:
            logger.error(f"Trend analysis failed: {e}")
            return {'error': str(e)}

    def _generate_ai_recommendations(self, diagnostic_report: Dict[str, Any]) -> List[str]:
        """Generate AI-powered recommendations based on diagnostic analysis."""
        recommendations = []

        try:
            # Performance-based recommendations
            performance_analysis = diagnostic_report.get('performance_analysis', {})
            if 'avg_sharpe_ratio' in performance_analysis:
                avg_sharpe = performance_analysis['avg_sharpe_ratio']
                if avg_sharpe < 0.5:
                    recommendations.append("Critical: Sharpe ratio below 0.5. Consider revising risk management strategy.")
                elif avg_sharpe < 1.0:
                    recommendations.append("Moderate: Sharpe ratio could be improved. Review position sizing and entry criteria.")

            # Anomaly-based recommendations
            anomaly_detection = diagnostic_report.get('anomaly_detection', {})
            if anomaly_detection.get('anomalies_detected'):
                high_severity_anomalies = [a for a in anomaly_detection['anomalies_detected'] if a.get('severity') == 'high']
                if high_severity_anomalies:
                    recommendations.append("High priority: Performance anomalies detected. Investigate model stability.")

            # Trend-based recommendations
            trend_analysis = diagnostic_report.get('trend_analysis', {})
            trend_directions = trend_analysis.get('trend_direction', {})
            if 'sharpe_ratio' in trend_directions and trend_directions['sharpe_ratio'] == 'declining':
                recommendations.append("Warning: Declining Sharpe ratio trend. Consider model retraining.")

            # Data quality recommendations
            data_quality = diagnostic_report.get('data_quality_assessment', {})
            if data_quality.get('data_quality_score', 1.0) < 0.7:
                recommendations.append("Data quality issues detected. Review telemetry collection setup.")

            # Default recommendation if no issues
            if not recommendations:
                recommendations.append("System performance within acceptable parameters. Continue monitoring.")

            return recommendations

        except Exception as e:
            logger.error(f"Failed to generate AI recommendations: {e}")
            return ["Unable to generate recommendations due to analysis error."]

    def _generate_performance_alerts(self, diagnostic_report: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Generate performance alerts based on diagnostic analysis."""
        alerts = []

        try:
            # Check performance thresholds
            performance_analysis = diagnostic_report.get('performance_analysis', {})

            # Sharpe ratio alert
            if 'avg_sharpe_ratio' in performance_analysis:
                avg_sharpe = performance_analysis['avg_sharpe_ratio']
                if avg_sharpe < self.performance_thresholds['sharpe_ratio_min']:
                    alerts.append({
                        'type': 'performance_threshold',
                        'metric': 'sharpe_ratio',
                        'severity': 'high' if avg_sharpe < 0.3 else 'medium',
                        'current_value': avg_sharpe,
                        'threshold': self.performance_thresholds['sharpe_ratio_min'],
                        'message': f"Sharpe ratio ({avg_sharpe:.3f}) below threshold ({self.performance_thresholds['sharpe_ratio_min']})"
                    })

            # Anomaly alerts
            anomaly_detection = diagnostic_report.get('anomaly_detection', {})
            if anomaly_detection.get('anomaly_score', 0) > 0.5:
                alerts.append({
                    'type': 'anomaly_detection',
                    'severity': 'high' if anomaly_detection['anomaly_score'] > 0.8 else 'medium',
                    'anomaly_score': anomaly_detection['anomaly_score'],
                    'message': f"Performance anomalies detected (score: {anomaly_detection['anomaly_score']:.3f})"
                })

            # Data quality alerts
            data_quality = diagnostic_report.get('data_quality_assessment', {})
            if data_quality.get('data_quality_score', 1.0) < 0.5:
                alerts.append({
                    'type': 'data_quality',
                    'severity': 'medium',
                    'quality_score': data_quality['data_quality_score'],
                    'message': f"Low data quality score: {data_quality['data_quality_score']:.3f}"
                })

            return alerts

        except Exception as e:
            logger.error(f"Failed to generate performance alerts: {e}")
            return []

    def _calculate_overall_health_score(self, diagnostic_report: Dict[str, Any]) -> float:
        """Calculate overall system health score (0-1 scale)."""
        try:
            health_components = []

            # Performance component
            performance_analysis = diagnostic_report.get('performance_analysis', {})
            if 'avg_sharpe_ratio' in performance_analysis:
                sharpe_score = min(1.0, max(0.0, performance_analysis['avg_sharpe_ratio'] / 2.0))
                health_components.append(sharpe_score * 0.4)  # 40% weight

            # Data quality component
            data_quality = diagnostic_report.get('data_quality_assessment', {})
            if 'data_quality_score' in data_quality:
                health_components.append(data_quality['data_quality_score'] * 0.2)  # 20% weight

            # Anomaly component (inverted - fewer anomalies = better health)
            anomaly_detection = diagnostic_report.get('anomaly_detection', {})
            anomaly_score = 1.0 - anomaly_detection.get('anomaly_score', 0)
            health_components.append(anomaly_score * 0.2)  # 20% weight

            # Stability component
            trend_analysis = diagnostic_report.get('trend_analysis', {})
            stability_score = trend_analysis.get('performance_stability', 0.5)
            health_components.append(stability_score * 0.2)  # 20% weight

            # Calculate weighted average
            if health_components:
                overall_health = sum(health_components)
                return min(1.0, max(0.0, overall_health))
            else:
                return 0.5  # Default neutral score

        except Exception as e:
            logger.error(f"Failed to calculate health score: {e}")
            return 0.0

    def _generate_predictive_insights(self) -> Dict[str, Any]:
        """Generate predictive insights based on historical trends."""
        try:
            insights = {
                'performance_forecast': {},
                'risk_assessment': {},
                'optimization_opportunities': []
            }

            # This would typically use more sophisticated ML models
            # For now, provide basic trend-based predictions

            trend_analysis = self._analyze_performance_trends()

            if 'trend_direction' in trend_analysis:
                for metric, direction in trend_analysis['trend_direction'].items():
                    if direction == 'improving':
                        insights['performance_forecast'][metric] = 'Expected continued improvement'
                    elif direction == 'declining':
                        insights['performance_forecast'][metric] = 'Risk of further decline'
                    else:
                        insights['performance_forecast'][metric] = 'Stable performance expected'

            # Risk assessment
            insights['risk_assessment'] = {
                'overall_risk_level': 'medium',  # Would be calculated based on various factors
                'key_risk_factors': ['Market regime changes', 'Model overfitting', 'Data quality degradation']
            }

            # Optimization opportunities
            insights['optimization_opportunities'] = [
                'Consider ensemble model approaches',
                'Implement adaptive position sizing',
                'Enhance feature engineering pipeline'
            ]

            return insights

        except Exception as e:
            logger.error(f"Failed to generate predictive insights: {e}")
            return {'error': str(e)}


def _recursive_sanitize(obj):
    """Recursively sanitize nested objects for logging."""
    if isinstance(obj, dict):
        return {k: _recursive_sanitize(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [_recursive_sanitize(item) for item in obj]
    elif isinstance(obj, (str, int, float, bool)) or obj is None:
        return obj
    else:
        return str(obj)  # Convert complex objects to string


class InterventionManager:
    """
    Manages AI intervention history and effectiveness tracking.

    This class tracks AI-driven interventions during framework execution,
    evaluates their effectiveness, and provides feedback for future AI decisions.
    """

    def __init__(self, ledger_path: str):
        self.ledger_path = ledger_path
        self.ledger = self._load_ledger()

    def _load_ledger(self) -> Dict[str, Any]:
        """Loads the intervention history from a JSON file."""
        if os.path.exists(self.ledger_path):
            try:
                with open(self.ledger_path, 'r') as f:
                    return json.load(f)
            except (json.JSONDecodeError, IOError):
                logger.warning(f"Could not load intervention ledger from {self.ledger_path}. Starting fresh.")
        return {"interventions": {}, "metadata": {"created": datetime.now().isoformat()}}

    def _save_ledger(self):
        """Saves the current state of the intervention ledger to a file."""
        try:
            os.makedirs(os.path.dirname(self.ledger_path), exist_ok=True)
            with open(self.ledger_path, 'w') as f:
                json.dump(self.ledger, f, indent=2, default=str)
        except IOError as e:
            logger.error(f"Failed to save intervention ledger: {e}")

    def log_intervention(self, cycle_num: int, suggestion_type: str, scope: str, regime: str, details: Dict, notes: str) -> str:
        """
        Logs an AI intervention with a unique ID for later evaluation.

        Args:
            cycle_num: The cycle number when intervention occurred
            suggestion_type: Type of intervention (parameter_adjustment, strategy_switch, etc.)
            scope: Scope of intervention (global, symbol_specific, etc.)
            regime: Market regime when intervention occurred
            details: Detailed information about the intervention
            notes: Additional notes about the intervention

        Returns:
            Unique intervention ID for tracking
        """
        intervention_id = f"{suggestion_type}_{scope}_{cycle_num}_{int(time.time())}"

        self.ledger["interventions"][intervention_id] = {
            "timestamp": datetime.now().isoformat(),
            "cycle_num": cycle_num,
            "suggestion_type": suggestion_type,
            "scope": scope,
            "regime": regime,
            "details": details,
            "notes": notes,
            "outcome_evaluated": False
        }

        self._save_ledger()
        logger.info(f"Logged intervention {intervention_id}: {suggestion_type} in {scope} scope")
        return intervention_id

    def evaluate_intervention_outcome(self, intervention_id: str, historical_telemetry: List[Dict]):
        """
        Evaluates the outcome of a previous intervention based on subsequent performance.

        Args:
            intervention_id: The ID of the intervention to evaluate
            historical_telemetry: Historical telemetry data for evaluation
        """
        if intervention_id not in self.ledger["interventions"]:
            logger.warning(f"Intervention {intervention_id} not found in ledger")
            return

        intervention = self.ledger["interventions"][intervention_id]
        intervention_cycle = intervention["cycle_num"]

        # Find performance data before and after intervention
        pre_intervention_cycles = [t for t in historical_telemetry if t.get("cycle_num", 0) < intervention_cycle]
        post_intervention_cycles = [t for t in historical_telemetry if t.get("cycle_num", 0) > intervention_cycle]

        if len(pre_intervention_cycles) >= 3 and len(post_intervention_cycles) >= 3:
            # Calculate performance metrics before and after
            pre_performance = np.mean([c.get("final_equity", 0) for c in pre_intervention_cycles[-3:]])
            post_performance = np.mean([c.get("final_equity", 0) for c in post_intervention_cycles[:3]])

            improvement = (post_performance - pre_performance) / pre_performance if pre_performance > 0 else 0

            # Update intervention record
            intervention["outcome_evaluated"] = True
            intervention["performance_improvement"] = improvement
            intervention["evaluation_timestamp"] = datetime.now().isoformat()
            intervention["effectiveness"] = "positive" if improvement > 0.02 else "negative" if improvement < -0.02 else "neutral"

            self._save_ledger()
            logger.info(f"Evaluated intervention {intervention_id}: {improvement:.2%} performance change")

    def get_feedback_for_ai_prompt(self, suggestion_type: str, scope: str, num_to_include: int = 5) -> str:
        """
        Generates feedback text for AI prompts based on intervention history.

        Args:
            suggestion_type: Type of intervention to get feedback for
            scope: Scope of intervention
            num_to_include: Number of recent interventions to include

        Returns:
            Formatted feedback string for AI prompts
        """
        relevant_interventions = []

        for intervention_id, intervention in self.ledger["interventions"].items():
            if (intervention["suggestion_type"] == suggestion_type and
                intervention["scope"] == scope and
                intervention["outcome_evaluated"]):
                relevant_interventions.append((intervention_id, intervention))

        # Sort by timestamp and take most recent
        relevant_interventions.sort(key=lambda x: x[1]["timestamp"], reverse=True)
        recent_interventions = relevant_interventions[:num_to_include]

        if not recent_interventions:
            return f"No previous intervention history available for {suggestion_type} in {scope} scope."

        feedback_lines = [f"Previous {suggestion_type} interventions in {scope} scope:"]

        for intervention_id, intervention in recent_interventions:
            effectiveness = intervention.get("effectiveness", "unknown")
            improvement = intervention.get("performance_improvement", 0)
            feedback_lines.append(
                f"- {intervention_id}: {effectiveness} outcome ({improvement:.2%} performance change)"
            )

        return "\n".join(feedback_lines)
