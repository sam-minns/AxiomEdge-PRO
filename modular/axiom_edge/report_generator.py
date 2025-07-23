# =============================================================================
# REPORT GENERATOR MODULE
# =============================================================================

import os
import logging
import textwrap
from datetime import datetime
from typing import Dict, List, Optional, Any
import pandas as pd
import numpy as np

# Optional imports with fallbacks
try:
    import matplotlib.pyplot as plt
    import seaborn as sns
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False

try:
    import plotly.graph_objects as go
    import plotly.express as px
    from plotly.subplots import make_subplots
    PLOTLY_AVAILABLE = True
except ImportError:
    PLOTLY_AVAILABLE = False

from .config import ConfigModel

logger = logging.getLogger(__name__)

class ReportGenerator:
    """
    Comprehensive report generation class for trading performance analysis.
    Creates detailed text reports, visualizations, and performance summaries.
    """
    
    def __init__(self, config: ConfigModel):
        """Initialize the ReportGenerator with configuration."""
        self.config = config
        
        # Set up matplotlib style if available
        if MATPLOTLIB_AVAILABLE:
            try:
                plt.style.use('seaborn-v0_8-darkgrid')
            except:
                try:
                    plt.style.use('seaborn-darkgrid')
                except:
                    pass  # Use default style
        
        logger.info("ReportGenerator initialized")

    def generate_full_report(self, trades_df: Optional[pd.DataFrame] = None,
                           equity_curve: Optional[pd.Series] = None,
                           cycle_metrics: Optional[List[Dict]] = None,
                           aggregated_shap: Optional[pd.DataFrame] = None,
                           framework_memory: Optional[Dict] = None,
                           aggregated_daily_dd: Optional[List[Dict]] = None,
                           last_classification_report: str = "N/A") -> Dict[str, Any]:
        """
        Generate a comprehensive performance report.
        
        Args:
            trades_df: DataFrame with individual trade data
            equity_curve: Series with equity curve over time
            cycle_metrics: List of metrics from each walk-forward cycle
            aggregated_shap: DataFrame with SHAP feature importance
            framework_memory: Historical performance data
            aggregated_daily_dd: Daily drawdown events
            last_classification_report: Classification performance report
            
        Returns:
            Dictionary with calculated performance metrics
        """
        logger.info("Generating comprehensive performance report...")
        
        # Generate visualizations if data is available
        if equity_curve is not None and len(equity_curve) > 1:
            self.plot_equity_curve(equity_curve)
        
        if aggregated_shap is not None and not aggregated_shap.empty:
            self.plot_shap_summary(aggregated_shap)
        
        if trades_df is not None and not trades_df.empty:
            self.plot_trade_analysis(trades_df)
        
        # Calculate performance metrics
        metrics = self._calculate_metrics(trades_df, equity_curve) if trades_df is not None else {}
        
        # Generate text report
        self.generate_text_report(
            metrics, cycle_metrics or [], aggregated_shap, 
            framework_memory, aggregated_daily_dd, last_classification_report
        )
        
        # Generate HTML report if possible
        if PLOTLY_AVAILABLE:
            self.generate_html_report(metrics, trades_df, equity_curve, aggregated_shap)
        
        logger.info("Report generation completed")
        return metrics

    def _calculate_metrics(self, trades_df: Optional[pd.DataFrame], 
                          equity_curve: Optional[pd.Series]) -> Dict[str, Any]:
        """Calculate comprehensive performance metrics."""
        metrics = {}
        
        try:
            # Basic metrics
            initial_capital = getattr(self.config, 'INITIAL_CAPITAL', 10000)
            metrics['initial_capital'] = initial_capital
            
            if equity_curve is not None and len(equity_curve) > 0:
                ending_capital = equity_curve.iloc[-1]
                metrics['ending_capital'] = ending_capital
                metrics['total_net_profit'] = ending_capital - initial_capital
                metrics['net_profit_pct'] = (ending_capital / initial_capital) - 1
                
                # Calculate returns
                returns = equity_curve.pct_change().dropna()
                if len(returns) > 0:
                    # Annualized metrics (assuming daily data)
                    trading_days = 252
                    total_days = len(returns)
                    years = total_days / trading_days
                    
                    # CAGR
                    if years > 0:
                        metrics['cagr'] = (ending_capital / initial_capital) ** (1/years) - 1
                    else:
                        metrics['cagr'] = 0
                    
                    # Sharpe Ratio
                    if returns.std() > 0:
                        metrics['sharpe_ratio'] = (returns.mean() * trading_days) / (returns.std() * np.sqrt(trading_days))
                    else:
                        metrics['sharpe_ratio'] = 0
                    
                    # Sortino Ratio
                    downside_returns = returns[returns < 0]
                    if len(downside_returns) > 0 and downside_returns.std() > 0:
                        metrics['sortino_ratio'] = (returns.mean() * trading_days) / (downside_returns.std() * np.sqrt(trading_days))
                    else:
                        metrics['sortino_ratio'] = 0
                    
                    # Maximum Drawdown
                    peak = equity_curve.expanding().max()
                    drawdown = (equity_curve - peak) / peak
                    metrics['max_drawdown'] = drawdown.min()
                    metrics['max_drawdown_pct'] = abs(drawdown.min())
                    
                    # Calmar/MAR Ratio
                    if metrics['max_drawdown_pct'] > 0:
                        metrics['mar_ratio'] = metrics['cagr'] / metrics['max_drawdown_pct']
                    else:
                        metrics['mar_ratio'] = 0
            
            # Trade-based metrics
            if trades_df is not None and not trades_df.empty:
                metrics['total_trades'] = len(trades_df)
                
                # Profit/Loss analysis
                if 'pnl' in trades_df.columns:
                    pnl = trades_df['pnl']
                    winning_trades = pnl[pnl > 0]
                    losing_trades = pnl[pnl < 0]
                    
                    metrics['winning_trades'] = len(winning_trades)
                    metrics['losing_trades'] = len(losing_trades)
                    metrics['win_rate'] = len(winning_trades) / len(trades_df) if len(trades_df) > 0 else 0
                    
                    metrics['avg_win'] = winning_trades.mean() if len(winning_trades) > 0 else 0
                    metrics['avg_loss'] = losing_trades.mean() if len(losing_trades) > 0 else 0
                    
                    # Profit Factor
                    gross_profit = winning_trades.sum() if len(winning_trades) > 0 else 0
                    gross_loss = abs(losing_trades.sum()) if len(losing_trades) > 0 else 0
                    metrics['profit_factor'] = gross_profit / gross_loss if gross_loss > 0 else float('inf')
                    
                    # Expected Payoff
                    metrics['expected_payoff'] = pnl.mean()
                    
                    # Largest win/loss
                    metrics['largest_win'] = pnl.max()
                    metrics['largest_loss'] = pnl.min()
            
        except Exception as e:
            logger.error(f"Error calculating metrics: {e}")
        
        return metrics

    def plot_equity_curve(self, equity_curve: pd.Series):
        """Generate equity curve plot."""
        if not MATPLOTLIB_AVAILABLE:
            logger.warning("Matplotlib not available. Skipping equity curve plot.")
            return
        
        try:
            plt.figure(figsize=(12, 8))
            
            # Plot equity curve
            plt.subplot(2, 1, 1)
            equity_curve.plot(color='blue', linewidth=2)
            plt.title(f'{getattr(self.config, "nickname", "Strategy")} - Equity Curve', fontsize=14, weight='bold')
            plt.ylabel('Portfolio Value ($)', fontsize=12)
            plt.grid(True, alpha=0.3)
            
            # Plot drawdown
            plt.subplot(2, 1, 2)
            peak = equity_curve.expanding().max()
            drawdown = (equity_curve - peak) / peak * 100
            drawdown.plot(color='red', linewidth=1, alpha=0.7)
            plt.fill_between(drawdown.index, drawdown, 0, color='red', alpha=0.3)
            plt.title('Drawdown (%)', fontsize=12)
            plt.ylabel('Drawdown (%)', fontsize=12)
            plt.xlabel('Date', fontsize=12)
            plt.grid(True, alpha=0.3)
            
            plt.tight_layout()
            
            # Save plot
            save_path = os.path.join(getattr(self.config, 'BASE_PATH', './'), 'Results', 'equity_curve.png')
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            plt.close()
            
            logger.info(f"Equity curve plot saved to: {save_path}")
            
        except Exception as e:
            logger.error(f"Error generating equity curve plot: {e}")

    def plot_shap_summary(self, shap_summary: pd.DataFrame):
        """Generate SHAP feature importance plot."""
        if not MATPLOTLIB_AVAILABLE:
            logger.warning("Matplotlib not available. Skipping SHAP plot.")
            return
        
        try:
            plt.figure(figsize=(12, 10))
            
            # Get top 20 features
            top_features = shap_summary.head(20)
            
            # Create horizontal bar plot
            if 'SHAP_Importance' in top_features.columns:
                importance_col = 'SHAP_Importance'
            elif 'mean_abs_shap' in top_features.columns:
                importance_col = 'mean_abs_shap'
            else:
                # Use the second column if available
                importance_col = top_features.columns[1] if len(top_features.columns) > 1 else top_features.columns[0]
            
            top_features = top_features.sort_values(by=importance_col)
            
            plt.barh(range(len(top_features)), top_features[importance_col], color='mediumseagreen')
            plt.yticks(range(len(top_features)), top_features.iloc[:, 0])  # First column as feature names
            
            title_str = f"{getattr(self.config, 'nickname', 'Strategy')} - Feature Importance"
            plt.title(title_str, fontsize=16, weight='bold')
            plt.xlabel('Mean Absolute SHAP Value', fontsize=12)
            plt.ylabel('Feature', fontsize=12)
            plt.tight_layout()
            
            # Save plot
            save_path = os.path.join(getattr(self.config, 'BASE_PATH', './'), 'Results', 'shap_summary.png')
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            plt.close()
            
            logger.info(f"SHAP summary plot saved to: {save_path}")
            
        except Exception as e:
            logger.error(f"Error generating SHAP plot: {e}")

    def plot_trade_analysis(self, trades_df: pd.DataFrame):
        """Generate trade analysis plots."""
        if not MATPLOTLIB_AVAILABLE:
            logger.warning("Matplotlib not available. Skipping trade analysis plots.")
            return
        
        try:
            fig, axes = plt.subplots(2, 2, figsize=(15, 10))
            
            # PnL distribution
            if 'pnl' in trades_df.columns:
                axes[0, 0].hist(trades_df['pnl'], bins=30, alpha=0.7, color='blue', edgecolor='black')
                axes[0, 0].set_title('PnL Distribution')
                axes[0, 0].set_xlabel('PnL ($)')
                axes[0, 0].set_ylabel('Frequency')
                axes[0, 0].grid(True, alpha=0.3)
            
            # Cumulative PnL
            if 'pnl' in trades_df.columns:
                cumulative_pnl = trades_df['pnl'].cumsum()
                axes[0, 1].plot(cumulative_pnl, color='green', linewidth=2)
                axes[0, 1].set_title('Cumulative PnL')
                axes[0, 1].set_xlabel('Trade Number')
                axes[0, 1].set_ylabel('Cumulative PnL ($)')
                axes[0, 1].grid(True, alpha=0.3)
            
            # Win/Loss by month (if date column exists)
            if 'entry_time' in trades_df.columns and 'pnl' in trades_df.columns:
                trades_df['month'] = pd.to_datetime(trades_df['entry_time']).dt.to_period('M')
                monthly_pnl = trades_df.groupby('month')['pnl'].sum()
                monthly_pnl.plot(kind='bar', ax=axes[1, 0], color=['red' if x < 0 else 'green' for x in monthly_pnl])
                axes[1, 0].set_title('Monthly PnL')
                axes[1, 0].set_xlabel('Month')
                axes[1, 0].set_ylabel('PnL ($)')
                axes[1, 0].tick_params(axis='x', rotation=45)
            
            # Trade duration analysis (if available)
            if 'duration' in trades_df.columns:
                axes[1, 1].hist(trades_df['duration'], bins=20, alpha=0.7, color='orange', edgecolor='black')
                axes[1, 1].set_title('Trade Duration Distribution')
                axes[1, 1].set_xlabel('Duration')
                axes[1, 1].set_ylabel('Frequency')
                axes[1, 1].grid(True, alpha=0.3)
            
            plt.tight_layout()
            
            # Save plot
            save_path = os.path.join(getattr(self.config, 'BASE_PATH', './'), 'Results', 'trade_analysis.png')
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            plt.close()
            
            logger.info(f"Trade analysis plots saved to: {save_path}")
            
        except Exception as e:
            logger.error(f"Error generating trade analysis plots: {e}")

    def generate_text_report(self, metrics: Dict[str, Any], cycle_metrics: List[Dict],
                           aggregated_shap: Optional[pd.DataFrame] = None,
                           framework_memory: Optional[Dict] = None,
                           aggregated_daily_dd: Optional[List[Dict]] = None,
                           last_classification_report: str = "N/A"):
        """Generate comprehensive text report."""
        
        WIDTH = 90  # Fixed width for A4 compatibility
        
        def _box_top(w): return f"+{'-' * (w-2)}+"
        def _box_mid(w): return f"+{'-' * (w-2)}+"
        def _box_bot(w): return f"+{'-' * (w-2)}+"
        def _box_title(title, w):
            padding = (w - len(title) - 4) // 2
            return [f"| {' ' * padding}{title}{' ' * (w - len(title) - padding - 3)}|"]
        def _box_line(text, w):
            wrapped = textwrap.fill(text, width=w-4)
            return [f"| {line:<{w-4}} |" for line in wrapped.split('\n')]
        def _box_text_kv(key, val, w):
            line = f"{key} {val}"
            return _box_line(line, w)
        
        # Build report sections
        report = [_box_top(WIDTH)]
        report.extend(_box_title('AXIOM EDGE PERFORMANCE REPORT', WIDTH))
        report.append(_box_mid(WIDTH))
        
        # Header information
        report.extend(_box_line(f"Strategy: {getattr(self.config, 'nickname', 'N/A')}", WIDTH))
        report.extend(_box_line(f"Version: {getattr(self.config, 'REPORT_LABEL', 'N/A')}", WIDTH))
        report.extend(_box_line(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", WIDTH))
        
        # Executive Summary
        if metrics:
            report.append(_box_mid(WIDTH))
            report.extend(_box_title('EXECUTIVE SUMMARY', WIDTH))
            report.append(_box_mid(WIDTH))
            
            summary_items = [
                (f"Initial Capital:", f"${metrics.get('initial_capital', 0):,.2f}"),
                (f"Ending Capital:", f"${metrics.get('ending_capital', 0):,.2f}"),
                (f"Total Net Profit:", f"${metrics.get('total_net_profit', 0):,.2f} ({metrics.get('net_profit_pct', 0):.2%})"),
                (f"Profit Factor:", f"{metrics.get('profit_factor', 0):.2f}"),
                (f"Win Rate:", f"{metrics.get('win_rate', 0):.2%}"),
                (f"Total Trades:", f"{metrics.get('total_trades', 0)}")
            ]
            
            for key, val in summary_items:
                report.extend(_box_text_kv(key, val, WIDTH))
        
        # Performance Metrics
        if metrics:
            report.append(_box_mid(WIDTH))
            report.extend(_box_title('PERFORMANCE METRICS', WIDTH))
            report.append(_box_mid(WIDTH))
            
            performance_items = [
                (f"CAGR (Annual Return):", f"{metrics.get('cagr', 0):.2%}"),
                (f"Sharpe Ratio:", f"{metrics.get('sharpe_ratio', 0):.2f}"),
                (f"Sortino Ratio:", f"{metrics.get('sortino_ratio', 0):.2f}"),
                (f"Calmar/MAR Ratio:", f"{metrics.get('mar_ratio', 0):.2f}"),
                (f"Maximum Drawdown:", f"{metrics.get('max_drawdown_pct', 0):.2%}"),
                (f"Expected Payoff:", f"${metrics.get('expected_payoff', 0):,.2f}")
            ]
            
            for key, val in performance_items:
                report.extend(_box_text_kv(key, val, WIDTH))
        
        # Trade Analysis
        if metrics and metrics.get('total_trades', 0) > 0:
            report.append(_box_mid(WIDTH))
            report.extend(_box_title('TRADE ANALYSIS', WIDTH))
            report.append(_box_mid(WIDTH))
            
            trade_items = [
                (f"Winning Trades:", f"{metrics.get('winning_trades', 0)}"),
                (f"Losing Trades:", f"{metrics.get('losing_trades', 0)}"),
                (f"Average Win:", f"${metrics.get('avg_win', 0):,.2f}"),
                (f"Average Loss:", f"${metrics.get('avg_loss', 0):,.2f}"),
                (f"Largest Win:", f"${metrics.get('largest_win', 0):,.2f}"),
                (f"Largest Loss:", f"${metrics.get('largest_loss', 0):,.2f}")
            ]
            
            for key, val in trade_items:
                report.extend(_box_text_kv(key, val, WIDTH))
        
        # Cycle Breakdown
        if cycle_metrics:
            report.append(_box_mid(WIDTH))
            report.extend(_box_title('WALK-FORWARD CYCLE BREAKDOWN', WIDTH))
            report.append(_box_mid(WIDTH))
            
            # Create cycle summary table
            cycle_data = []
            for cycle in cycle_metrics:
                cycle_metrics_data = cycle.get('metrics', {})
                cycle_data.append({
                    'Cycle': cycle.get('cycle', 'N/A'),
                    'Status': cycle.get('status', 'N/A'),
                    'Trades': cycle_metrics_data.get('total_trades', 0),
                    'PNL': f"${cycle_metrics_data.get('total_net_profit', 0):,.2f}",
                    'Win Rate': f"{cycle_metrics_data.get('win_rate', 0):.1%}",
                    'Max DD': f"{cycle_metrics_data.get('max_drawdown_pct', 0):.1f}%"
                })
            
            if cycle_data:
                cycle_df = pd.DataFrame(cycle_data)
                cycle_str = cycle_df.to_string(index=False)
                for line in cycle_str.split('\n'):
                    report.extend(_box_line(line, WIDTH))
            else:
                report.extend(_box_line("No cycle data available.", WIDTH))
        
        # Feature Importance
        if aggregated_shap is not None and not aggregated_shap.empty:
            report.append(_box_mid(WIDTH))
            report.extend(_box_title('TOP FEATURE IMPORTANCE (SHAP)', WIDTH))
            report.append(_box_mid(WIDTH))
            
            # Get top 15 features
            top_features = aggregated_shap.head(15)
            
            # Determine importance column
            if 'SHAP_Importance' in top_features.columns:
                importance_col = 'SHAP_Importance'
            elif 'mean_abs_shap' in top_features.columns:
                importance_col = 'mean_abs_shap'
            else:
                importance_col = top_features.columns[1] if len(top_features.columns) > 1 else top_features.columns[0]
            
            feature_col = top_features.columns[0]
            
            for idx, row in top_features.iterrows():
                feature_name = row[feature_col]
                importance = row[importance_col]
                report.extend(_box_line(f"{feature_name}: {importance:.4f}", WIDTH))
        
        # Classification Report
        if last_classification_report and last_classification_report != "N/A":
            report.append(_box_mid(WIDTH))
            report.extend(_box_title('MODEL PERFORMANCE (LAST CYCLE)', WIDTH))
            report.append(_box_mid(WIDTH))
            
            for line in last_classification_report.split('\n'):
                if line.strip():
                    report.extend(_box_line(line, WIDTH))
        
        report.append(_box_bot(WIDTH))
        
        # Generate final report
        final_report = "\n".join(report)
        
        # Save report
        try:
            save_path = os.path.join(getattr(self.config, 'BASE_PATH', './'), 'Results', 'performance_report.txt')
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            
            with open(save_path, 'w', encoding='utf-8') as f:
                f.write(final_report)
            
            logger.info(f"Text report saved to: {save_path}")
            
        except Exception as e:
            logger.error(f"Error saving text report: {e}")
        
        # Also log the report
        logger.info("\n" + final_report)

    def generate_html_report(self, metrics: Dict[str, Any], trades_df: Optional[pd.DataFrame] = None,
                           equity_curve: Optional[pd.Series] = None, 
                           aggregated_shap: Optional[pd.DataFrame] = None):
        """Generate interactive HTML report using Plotly."""
        if not PLOTLY_AVAILABLE:
            logger.warning("Plotly not available. Skipping HTML report generation.")
            return
        
        try:
            # Create subplots
            fig = make_subplots(
                rows=3, cols=2,
                subplot_titles=('Equity Curve', 'Drawdown', 'PnL Distribution', 
                               'Feature Importance', 'Monthly Returns', 'Trade Analysis'),
                specs=[[{"secondary_y": False}, {"secondary_y": False}],
                       [{"secondary_y": False}, {"secondary_y": False}],
                       [{"secondary_y": False}, {"secondary_y": False}]]
            )
            
            # Equity curve
            if equity_curve is not None:
                fig.add_trace(
                    go.Scatter(x=equity_curve.index, y=equity_curve.values, 
                              name='Equity', line=dict(color='blue')),
                    row=1, col=1
                )
                
                # Drawdown
                peak = equity_curve.expanding().max()
                drawdown = (equity_curve - peak) / peak * 100
                fig.add_trace(
                    go.Scatter(x=drawdown.index, y=drawdown.values, 
                              name='Drawdown', fill='tonexty', line=dict(color='red')),
                    row=1, col=2
                )
            
            # PnL distribution
            if trades_df is not None and 'pnl' in trades_df.columns:
                fig.add_trace(
                    go.Histogram(x=trades_df['pnl'], name='PnL Distribution', 
                                marker_color='lightblue'),
                    row=2, col=1
                )
            
            # Feature importance
            if aggregated_shap is not None and not aggregated_shap.empty:
                top_features = aggregated_shap.head(10)
                importance_col = 'SHAP_Importance' if 'SHAP_Importance' in top_features.columns else top_features.columns[1]
                feature_col = top_features.columns[0]
                
                fig.add_trace(
                    go.Bar(x=top_features[importance_col], y=top_features[feature_col],
                          orientation='h', name='Feature Importance', marker_color='green'),
                    row=2, col=2
                )
            
            # Update layout
            fig.update_layout(
                title=f"AxiomEdge Performance Dashboard - {getattr(self.config, 'nickname', 'Strategy')}",
                height=1200,
                showlegend=False
            )
            
            # Save HTML report
            save_path = os.path.join(getattr(self.config, 'BASE_PATH', './'), 'Results', 'performance_dashboard.html')
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            fig.write_html(save_path)
            
            logger.info(f"HTML dashboard saved to: {save_path}")
            
        except Exception as e:
            logger.error(f"Error generating HTML report: {e}")

    def generate_summary_stats(self, metrics: Dict[str, Any]) -> str:
        """Generate a concise summary of key statistics."""
        if not metrics:
            return "No performance data available."
        
        summary_lines = [
            f"📊 PERFORMANCE SUMMARY",
            f"{'='*50}",
            f"💰 Total Return: {metrics.get('net_profit_pct', 0):.2%}",
            f"📈 CAGR: {metrics.get('cagr', 0):.2%}",
            f"⚡ Sharpe Ratio: {metrics.get('sharpe_ratio', 0):.2f}",
            f"📉 Max Drawdown: {metrics.get('max_drawdown_pct', 0):.2%}",
            f"🎯 Win Rate: {metrics.get('win_rate', 0):.2%}",
            f"🔢 Total Trades: {metrics.get('total_trades', 0)}",
            f"💎 Profit Factor: {metrics.get('profit_factor', 0):.2f}",
            f"{'='*50}"
        ]
        
        return "\n".join(summary_lines)
