# =============================================================================
# REPORT GENERATOR MODULE
# =============================================================================

import os
import logging
import textwrap
import json
import base64
from io import BytesIO
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple
import pandas as pd
import numpy as np
import warnings

# Optional imports with fallbacks
try:
    import matplotlib.pyplot as plt
    import matplotlib.dates as mdates
    import seaborn as sns
    from matplotlib.patches import Rectangle
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False

try:
    import plotly.graph_objects as go
    import plotly.express as px
    from plotly.subplots import make_subplots
    import plotly.figure_factory as ff
    PLOTLY_AVAILABLE = True
except ImportError:
    PLOTLY_AVAILABLE = False

try:
    import shap
    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False

from .config import ConfigModel

logger = logging.getLogger(__name__)

class ReportGenerator:
    """
    Advanced comprehensive report generation system for trading performance analysis.

    Creates detailed text reports, interactive visualizations, and performance summaries
    with professional formatting and comprehensive analytics. Supports multiple output
    formats including text, HTML, PDF, and static plots.

    Features:
    - Professional formatted performance reports with executive summaries
    - Advanced static visualizations using Matplotlib with custom styling
    - Interactive dashboards using Plotly with drill-down capabilities
    - Comprehensive risk-adjusted analytics and regime analysis
    - Enhanced SHAP feature importance visualization with explanations
    - Detailed trade analysis and breakdown statistics
    - Multi-format output support (Text, HTML, PDF, JSON)
    - Performance comparison and benchmarking
    - Monte Carlo simulation visualization
    - Drawdown analysis with recovery periods
    - Rolling performance metrics visualization
    - Trade sequence analysis and pattern detection
    - Risk attribution and factor analysis
    - Custom branding and styling options
    """
    
    def __init__(self, config: ConfigModel):
        """Initialize the Advanced ReportGenerator with configuration and styling."""
        self.config = config

        # Advanced configuration options
        self.output_dir = getattr(config, 'REPORTS_OUTPUT_DIR', 'reports')
        self.include_interactive = getattr(config, 'INCLUDE_INTERACTIVE_PLOTS', True)
        self.include_static = getattr(config, 'INCLUDE_STATIC_PLOTS', True)
        self.report_format = getattr(config, 'REPORT_FORMAT', 'both')  # 'text', 'html', 'both'
        self.custom_branding = getattr(config, 'CUSTOM_BRANDING', {})

        # Create output directory if it doesn't exist
        os.makedirs(self.output_dir, exist_ok=True)

        # Set up matplotlib style if available
        if MATPLOTLIB_AVAILABLE:
            try:
                plt.style.use('seaborn-v0_8-darkgrid')
                # Custom color palette for professional look
                self.colors = {
                    'primary': '#2E86AB',
                    'secondary': '#A23B72',
                    'success': '#F18F01',
                    'danger': '#C73E1D',
                    'neutral': '#6C757D',
                    'background': '#F8F9FA'
                }
            except:
                try:
                    plt.style.use('seaborn-darkgrid')
                    self.colors = {
                        'primary': '#1f77b4',
                        'secondary': '#ff7f0e',
                        'success': '#2ca02c',
                        'danger': '#d62728',
                        'neutral': '#7f7f7f',
                        'background': '#ffffff'
                    }
                except:
                    # Default colors
                    self.colors = {
                        'primary': '#1f77b4',
                        'secondary': '#ff7f0e',
                        'success': '#2ca02c',
                        'danger': '#d62728',
                        'neutral': '#7f7f7f',
                        'background': '#ffffff'
                    }

        # Performance tracking
        self.generated_reports = []
        self.report_metadata = {}

        logger.info(f"Advanced ReportGenerator initialized with output dir: {self.output_dir}")

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

    def plot_enhanced_equity_curve(self, equity_curve: pd.Series, trades_df: Optional[pd.DataFrame] = None):
        """
        Generate enhanced equity curve plot with drawdown analysis and trade markers.
        """
        if not MATPLOTLIB_AVAILABLE:
            logger.warning("Matplotlib not available. Skipping enhanced equity curve plot.")
            return

        try:
            fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(15, 12),
                                               gridspec_kw={'height_ratios': [3, 1, 1]})

            # Main equity curve
            ax1.plot(equity_curve.index, equity_curve.values,
                    color=self.colors['primary'], linewidth=2, label='Equity Curve')

            # Calculate and plot drawdown
            peak = equity_curve.expanding().max()
            drawdown = (equity_curve - peak) / peak * 100

            ax2.fill_between(drawdown.index, drawdown.values, 0,
                           color=self.colors['danger'], alpha=0.3, label='Drawdown %')
            ax2.plot(drawdown.index, drawdown.values,
                    color=self.colors['danger'], linewidth=1)

            # Add trade markers if available
            if trades_df is not None and not trades_df.empty:
                # Plot winning and losing trades
                if 'entry_time' in trades_df.columns and 'pnl' in trades_df.columns:
                    winning_trades = trades_df[trades_df['pnl'] > 0]
                    losing_trades = trades_df[trades_df['pnl'] < 0]

                    if not winning_trades.empty:
                        ax1.scatter(winning_trades['entry_time'],
                                  [equity_curve.loc[t] if t in equity_curve.index else equity_curve.iloc[0]
                                   for t in winning_trades['entry_time']],
                                  color=self.colors['success'], marker='^', s=30, alpha=0.7, label='Winning Trades')

                    if not losing_trades.empty:
                        ax1.scatter(losing_trades['entry_time'],
                                  [equity_curve.loc[t] if t in equity_curve.index else equity_curve.iloc[0]
                                   for t in losing_trades['entry_time']],
                                  color=self.colors['danger'], marker='v', s=30, alpha=0.7, label='Losing Trades')

            # Rolling Sharpe ratio (if enough data)
            if len(equity_curve) > 252:
                returns = equity_curve.pct_change().dropna()
                rolling_sharpe = returns.rolling(252).mean() / returns.rolling(252).std() * np.sqrt(252)
                ax3.plot(rolling_sharpe.index, rolling_sharpe.values,
                        color=self.colors['secondary'], linewidth=1, label='Rolling Sharpe (1Y)')
                ax3.axhline(y=1.0, color=self.colors['neutral'], linestyle='--', alpha=0.5)
                ax3.axhline(y=2.0, color=self.colors['success'], linestyle='--', alpha=0.5)

            # Formatting
            ax1.set_title('Enhanced Equity Curve Analysis', fontsize=16, fontweight='bold')
            ax1.set_ylabel('Portfolio Value', fontsize=12)
            ax1.grid(True, alpha=0.3)
            ax1.legend()

            ax2.set_ylabel('Drawdown %', fontsize=12)
            ax2.grid(True, alpha=0.3)
            ax2.legend()

            ax3.set_ylabel('Rolling Sharpe', fontsize=12)
            ax3.set_xlabel('Date', fontsize=12)
            ax3.grid(True, alpha=0.3)
            ax3.legend()

            # Format x-axis
            for ax in [ax1, ax2, ax3]:
                ax.tick_params(axis='x', rotation=45)

            plt.tight_layout()

            # Save plot
            save_path = os.path.join(self.output_dir, 'enhanced_equity_curve.png')
            plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
            plt.close()

            logger.info(f"Enhanced equity curve plot saved to: {save_path}")

        except Exception as e:
            logger.error(f"Error generating enhanced equity curve plot: {e}")

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

    def plot_advanced_trade_analysis(self, trades_df: pd.DataFrame):
        """
        Generate comprehensive trade analysis with multiple visualizations.
        """
        if not MATPLOTLIB_AVAILABLE or trades_df.empty:
            logger.warning("Matplotlib not available or no trades data. Skipping advanced trade analysis.")
            return

        try:
            fig = plt.figure(figsize=(20, 16))

            # Create a complex subplot layout
            gs = fig.add_gridspec(4, 4, hspace=0.3, wspace=0.3)

            # 1. P&L Distribution
            ax1 = fig.add_subplot(gs[0, 0:2])
            if 'pnl' in trades_df.columns:
                trades_df['pnl'].hist(bins=30, alpha=0.7, color=self.colors['primary'], ax=ax1)
                ax1.axvline(trades_df['pnl'].mean(), color=self.colors['danger'],
                           linestyle='--', label=f"Mean: {trades_df['pnl'].mean():.2f}")
                ax1.set_title('P&L Distribution', fontweight='bold')
                ax1.set_xlabel('P&L')
                ax1.set_ylabel('Frequency')
                ax1.legend()
                ax1.grid(True, alpha=0.3)

            # 2. Win/Loss Ratio by Time
            ax2 = fig.add_subplot(gs[0, 2:4])
            if 'entry_time' in trades_df.columns and 'pnl' in trades_df.columns:
                trades_df['date'] = pd.to_datetime(trades_df['entry_time']).dt.date
                daily_stats = trades_df.groupby('date').agg({
                    'pnl': ['count', lambda x: (x > 0).sum(), lambda x: (x < 0).sum()]
                }).round(2)
                daily_stats.columns = ['total_trades', 'wins', 'losses']
                daily_stats['win_rate'] = daily_stats['wins'] / daily_stats['total_trades'] * 100

                ax2.plot(daily_stats.index, daily_stats['win_rate'],
                        color=self.colors['success'], marker='o', markersize=4)
                ax2.axhline(y=50, color=self.colors['neutral'], linestyle='--', alpha=0.5)
                ax2.set_title('Daily Win Rate %', fontweight='bold')
                ax2.set_ylabel('Win Rate %')
                ax2.tick_params(axis='x', rotation=45)
                ax2.grid(True, alpha=0.3)

            # 3. Trade Duration Analysis
            ax3 = fig.add_subplot(gs[1, 0:2])
            if 'entry_time' in trades_df.columns and 'exit_time' in trades_df.columns:
                trades_df['duration'] = (pd.to_datetime(trades_df['exit_time']) -
                                       pd.to_datetime(trades_df['entry_time'])).dt.total_seconds() / 3600
                trades_df['duration'].hist(bins=20, alpha=0.7, color=self.colors['secondary'], ax=ax3)
                ax3.set_title('Trade Duration Distribution (Hours)', fontweight='bold')
                ax3.set_xlabel('Duration (Hours)')
                ax3.set_ylabel('Frequency')
                ax3.grid(True, alpha=0.3)

            # 4. Monthly Performance Heatmap
            ax4 = fig.add_subplot(gs[1, 2:4])
            if 'entry_time' in trades_df.columns and 'pnl' in trades_df.columns:
                trades_df['month'] = pd.to_datetime(trades_df['entry_time']).dt.month
                trades_df['year'] = pd.to_datetime(trades_df['entry_time']).dt.year
                monthly_pnl = trades_df.groupby(['year', 'month'])['pnl'].sum().unstack(fill_value=0)

                if not monthly_pnl.empty:
                    im = ax4.imshow(monthly_pnl.values, cmap='RdYlGn', aspect='auto')
                    ax4.set_xticks(range(len(monthly_pnl.columns)))
                    ax4.set_xticklabels(monthly_pnl.columns)
                    ax4.set_yticks(range(len(monthly_pnl.index)))
                    ax4.set_yticklabels(monthly_pnl.index)
                    ax4.set_title('Monthly P&L Heatmap', fontweight='bold')
                    ax4.set_xlabel('Month')
                    ax4.set_ylabel('Year')
                    plt.colorbar(im, ax=ax4, shrink=0.8)

            # 5. Cumulative P&L by Trade
            ax5 = fig.add_subplot(gs[2, 0:2])
            if 'pnl' in trades_df.columns:
                cumulative_pnl = trades_df['pnl'].cumsum()
                ax5.plot(range(len(cumulative_pnl)), cumulative_pnl.values,
                        color=self.colors['primary'], linewidth=2)
                ax5.set_title('Cumulative P&L by Trade', fontweight='bold')
                ax5.set_xlabel('Trade Number')
                ax5.set_ylabel('Cumulative P&L')
                ax5.grid(True, alpha=0.3)

            # 6. Risk-Reward Scatter
            ax6 = fig.add_subplot(gs[2, 2:4])
            if 'pnl' in trades_df.columns and 'size' in trades_df.columns:
                colors = ['green' if pnl > 0 else 'red' for pnl in trades_df['pnl']]
                ax6.scatter(trades_df['size'], trades_df['pnl'],
                           c=colors, alpha=0.6, s=30)
                ax6.set_title('Risk vs Reward', fontweight='bold')
                ax6.set_xlabel('Position Size')
                ax6.set_ylabel('P&L')
                ax6.grid(True, alpha=0.3)

            # 7. Trade Sequence Analysis
            ax7 = fig.add_subplot(gs[3, 0:4])
            if 'pnl' in trades_df.columns:
                # Calculate streaks
                trades_df['win'] = trades_df['pnl'] > 0
                trades_df['streak_id'] = (trades_df['win'] != trades_df['win'].shift()).cumsum()
                streaks = trades_df.groupby('streak_id').agg({
                    'win': ['first', 'count'],
                    'pnl': 'sum'
                })
                streaks.columns = ['is_winning_streak', 'length', 'total_pnl']

                # Plot streak lengths
                winning_streaks = streaks[streaks['is_winning_streak']]['length']
                losing_streaks = streaks[~streaks['is_winning_streak']]['length']

                x_pos = np.arange(len(streaks))
                colors = ['green' if win else 'red' for win in streaks['is_winning_streak']]
                bars = ax7.bar(x_pos, streaks['length'], color=colors, alpha=0.7)

                ax7.set_title('Win/Loss Streak Analysis', fontweight='bold')
                ax7.set_xlabel('Streak Number')
                ax7.set_ylabel('Streak Length')
                ax7.grid(True, alpha=0.3)

                # Add legend
                from matplotlib.patches import Patch
                legend_elements = [Patch(facecolor='green', alpha=0.7, label='Winning Streaks'),
                                 Patch(facecolor='red', alpha=0.7, label='Losing Streaks')]
                ax7.legend(handles=legend_elements)

            plt.suptitle('Advanced Trade Analysis Dashboard', fontsize=20, fontweight='bold', y=0.98)

            # Save plot
            save_path = os.path.join(self.output_dir, 'advanced_trade_analysis.png')
            plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
            plt.close()

            logger.info(f"Advanced trade analysis plot saved to: {save_path}")

        except Exception as e:
            logger.error(f"Error generating advanced trade analysis: {e}")

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

    def create_interactive_dashboard(self, metrics: Dict[str, Any], trades_df: Optional[pd.DataFrame] = None,
                                   equity_curve: Optional[pd.Series] = None,
                                   aggregated_shap: Optional[pd.DataFrame] = None) -> str:
        """
        Create comprehensive interactive dashboard using Plotly.

        Returns:
            HTML string of the interactive dashboard
        """
        if not PLOTLY_AVAILABLE:
            logger.warning("Plotly not available. Cannot create interactive dashboard.")
            return "<p>Interactive dashboard requires Plotly installation.</p>"

        try:
            # Create subplots
            fig = make_subplots(
                rows=4, cols=2,
                subplot_titles=('Equity Curve', 'Drawdown Analysis',
                              'Monthly Returns Heatmap', 'Trade P&L Distribution',
                              'Rolling Sharpe Ratio', 'Win Rate Over Time',
                              'Feature Importance (SHAP)', 'Risk Metrics'),
                specs=[[{"secondary_y": False}, {"secondary_y": False}],
                       [{"type": "heatmap"}, {"type": "histogram"}],
                       [{"secondary_y": False}, {"secondary_y": False}],
                       [{"type": "bar"}, {"type": "indicator"}]],
                vertical_spacing=0.08,
                horizontal_spacing=0.1
            )

            # 1. Equity Curve
            if equity_curve is not None and len(equity_curve) > 1:
                fig.add_trace(
                    go.Scatter(x=equity_curve.index, y=equity_curve.values,
                             mode='lines', name='Equity Curve',
                             line=dict(color='#2E86AB', width=2),
                             hovertemplate='Date: %{x}<br>Value: $%{y:,.2f}<extra></extra>'),
                    row=1, col=1
                )

                # Add drawdown
                peak = equity_curve.expanding().max()
                drawdown = (equity_curve - peak) / peak * 100
                fig.add_trace(
                    go.Scatter(x=drawdown.index, y=drawdown.values,
                             mode='lines', name='Drawdown %',
                             fill='tonexty', fillcolor='rgba(199, 62, 29, 0.3)',
                             line=dict(color='#C73E1D', width=1),
                             hovertemplate='Date: %{x}<br>Drawdown: %{y:.2f}%<extra></extra>'),
                    row=1, col=2
                )

            # 2. Monthly Returns Heatmap
            if trades_df is not None and not trades_df.empty and 'entry_time' in trades_df.columns:
                trades_df['month'] = pd.to_datetime(trades_df['entry_time']).dt.month
                trades_df['year'] = pd.to_datetime(trades_df['entry_time']).dt.year
                monthly_returns = trades_df.groupby(['year', 'month'])['pnl'].sum().unstack(fill_value=0)

                if not monthly_returns.empty:
                    fig.add_trace(
                        go.Heatmap(z=monthly_returns.values,
                                 x=monthly_returns.columns,
                                 y=monthly_returns.index,
                                 colorscale='RdYlGn',
                                 hovertemplate='Month: %{x}<br>Year: %{y}<br>Return: $%{z:,.2f}<extra></extra>'),
                        row=2, col=1
                    )

            # 3. Trade P&L Distribution
            if trades_df is not None and not trades_df.empty and 'pnl' in trades_df.columns:
                fig.add_trace(
                    go.Histogram(x=trades_df['pnl'], nbinsx=30,
                               name='P&L Distribution',
                               marker_color='#2E86AB',
                               opacity=0.7,
                               hovertemplate='P&L Range: %{x}<br>Count: %{y}<extra></extra>'),
                    row=2, col=2
                )

            # 4. Rolling Sharpe Ratio
            if equity_curve is not None and len(equity_curve) > 252:
                returns = equity_curve.pct_change().dropna()
                rolling_sharpe = returns.rolling(252).mean() / returns.rolling(252).std() * np.sqrt(252)
                fig.add_trace(
                    go.Scatter(x=rolling_sharpe.index, y=rolling_sharpe.values,
                             mode='lines', name='Rolling Sharpe (1Y)',
                             line=dict(color='#A23B72', width=2),
                             hovertemplate='Date: %{x}<br>Sharpe: %{y:.2f}<extra></extra>'),
                    row=3, col=1
                )

                # Add reference lines
                fig.add_hline(y=1.0, line_dash="dash", line_color="gray",
                            annotation_text="Good (1.0)", row=3, col=1)
                fig.add_hline(y=2.0, line_dash="dash", line_color="green",
                            annotation_text="Excellent (2.0)", row=3, col=1)

            # 5. Win Rate Over Time
            if trades_df is not None and not trades_df.empty:
                trades_df['date'] = pd.to_datetime(trades_df['entry_time']).dt.date
                daily_stats = trades_df.groupby('date').agg({
                    'pnl': ['count', lambda x: (x > 0).sum()]
                })
                daily_stats.columns = ['total_trades', 'wins']
                daily_stats['win_rate'] = daily_stats['wins'] / daily_stats['total_trades'] * 100

                fig.add_trace(
                    go.Scatter(x=daily_stats.index, y=daily_stats['win_rate'],
                             mode='lines+markers', name='Daily Win Rate',
                             line=dict(color='#F18F01', width=2),
                             marker=dict(size=4),
                             hovertemplate='Date: %{x}<br>Win Rate: %{y:.1f}%<extra></extra>'),
                    row=3, col=2
                )

                fig.add_hline(y=50, line_dash="dash", line_color="gray",
                            annotation_text="Break-even (50%)", row=3, col=2)

            # 6. Feature Importance (SHAP)
            if aggregated_shap is not None and not aggregated_shap.empty:
                top_features = aggregated_shap.head(10)
                fig.add_trace(
                    go.Bar(x=top_features['SHAP_Importance'],
                          y=top_features['feature'],
                          orientation='h',
                          name='SHAP Importance',
                          marker_color='#2E86AB',
                          hovertemplate='Feature: %{y}<br>Importance: %{x:.4f}<extra></extra>'),
                    row=4, col=1
                )

            # 7. Risk Metrics Indicator
            if metrics:
                fig.add_trace(
                    go.Indicator(
                        mode="gauge+number+delta",
                        value=metrics.get('sharpe_ratio', 0),
                        domain={'x': [0, 1], 'y': [0, 1]},
                        title={'text': "Sharpe Ratio"},
                        delta={'reference': 1.0},
                        gauge={'axis': {'range': [None, 3]},
                              'bar': {'color': "#2E86AB"},
                              'steps': [{'range': [0, 1], 'color': "lightgray"},
                                       {'range': [1, 2], 'color': "yellow"},
                                       {'range': [2, 3], 'color': "green"}],
                              'threshold': {'line': {'color': "red", 'width': 4},
                                          'thickness': 0.75, 'value': 2.0}}),
                    row=4, col=2
                )

            # Update layout
            fig.update_layout(
                height=1200,
                showlegend=False,
                title_text="AxiomEdge Trading Performance Dashboard",
                title_x=0.5,
                title_font_size=24,
                template="plotly_white"
            )

            # Convert to HTML
            html_str = fig.to_html(include_plotlyjs='cdn', div_id="dashboard")

            # Add custom CSS for better styling
            custom_css = """
            <style>
                body { font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; }
                .plotly-graph-div { margin: 10px; }
                h1 { color: #2E86AB; text-align: center; }
            </style>
            """

            html_str = html_str.replace('<head>', f'<head>{custom_css}')

            return html_str

        except Exception as e:
            logger.error(f"Error creating interactive dashboard: {e}")
            return f"<p>Error creating dashboard: {str(e)}</p>"

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

    def generate_executive_summary(self, metrics: Dict[str, Any], trades_df: Optional[pd.DataFrame] = None) -> str:
        """
        Generate a concise executive summary for stakeholders.

        Args:
            metrics: Performance metrics dictionary
            trades_df: Optional trades DataFrame for additional analysis

        Returns:
            Formatted executive summary string
        """
        try:
            summary_lines = []

            # Header
            summary_lines.extend([
                "=" * 80,
                "AXIOM EDGE TRADING SYSTEM - EXECUTIVE SUMMARY",
                "=" * 80,
                f"Report Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
                ""
            ])

            # Key Performance Indicators
            summary_lines.extend([
                "KEY PERFORMANCE INDICATORS",
                "-" * 40,
                f"Total Return:           {self._format_percentage(metrics.get('net_profit_pct', 0))}",
                f"Annualized Return:      {self._format_percentage(metrics.get('cagr', 0))}",
                f"Sharpe Ratio:           {metrics.get('sharpe_ratio', 0):.2f}",
                f"Maximum Drawdown:       {self._format_percentage(metrics.get('max_drawdown_pct', 0))}",
                f"Win Rate:               {self._format_percentage(metrics.get('win_rate', 0))}",
                f"Profit Factor:          {metrics.get('profit_factor', 0):.2f}",
                ""
            ])

            # Risk Assessment
            risk_level = self._assess_risk_level(metrics)
            summary_lines.extend([
                "RISK ASSESSMENT",
                "-" * 40,
                f"Risk Level:             {risk_level['level']}",
                f"Risk Score:             {risk_level['score']:.1f}/10",
                f"Key Risk Factors:       {', '.join(risk_level['factors'])}",
                ""
            ])

            # Trading Activity
            if trades_df is not None and not trades_df.empty:
                avg_trade_duration = self._calculate_avg_trade_duration(trades_df)
                summary_lines.extend([
                    "TRADING ACTIVITY",
                    "-" * 40,
                    f"Total Trades:           {len(trades_df):,}",
                    f"Avg Trade Duration:     {avg_trade_duration}",
                    f"Best Trade:             {self._format_currency(metrics.get('largest_win', 0))}",
                    f"Worst Trade:            {self._format_currency(metrics.get('largest_loss', 0))}",
                    ""
                ])

            # Recommendations
            recommendations = self._generate_recommendations(metrics, trades_df)
            summary_lines.extend([
                "STRATEGIC RECOMMENDATIONS",
                "-" * 40,
            ])
            for i, rec in enumerate(recommendations, 1):
                summary_lines.append(f"{i}. {rec}")

            summary_lines.extend(["", "=" * 80])

            return "\n".join(summary_lines)

        except Exception as e:
            logger.error(f"Error generating executive summary: {e}")
            return "Error generating executive summary."

    def _assess_risk_level(self, metrics: Dict[str, Any]) -> Dict[str, Any]:
        """Assess overall risk level based on multiple metrics."""
        risk_factors = []
        risk_score = 0

        # Sharpe ratio assessment
        sharpe = metrics.get('sharpe_ratio', 0)
        if sharpe < 0.5:
            risk_factors.append("Low Sharpe Ratio")
            risk_score += 3
        elif sharpe < 1.0:
            risk_score += 1

        # Drawdown assessment
        max_dd = abs(metrics.get('max_drawdown_pct', 0))
        if max_dd > 0.2:
            risk_factors.append("High Drawdown")
            risk_score += 3
        elif max_dd > 0.1:
            risk_score += 1

        # Win rate assessment
        win_rate = metrics.get('win_rate', 0)
        if win_rate < 0.4:
            risk_factors.append("Low Win Rate")
            risk_score += 2

        # Profit factor assessment
        profit_factor = metrics.get('profit_factor', 0)
        if profit_factor < 1.2:
            risk_factors.append("Low Profit Factor")
            risk_score += 2

        # Determine risk level
        if risk_score <= 2:
            level = "LOW"
        elif risk_score <= 5:
            level = "MODERATE"
        elif risk_score <= 8:
            level = "HIGH"
        else:
            level = "VERY HIGH"

        if not risk_factors:
            risk_factors = ["None identified"]

        return {
            'level': level,
            'score': min(10, risk_score),
            'factors': risk_factors
        }

    def _generate_recommendations(self, metrics: Dict[str, Any], trades_df: Optional[pd.DataFrame]) -> List[str]:
        """Generate strategic recommendations based on performance analysis."""
        recommendations = []

        # Sharpe ratio recommendations
        sharpe = metrics.get('sharpe_ratio', 0)
        if sharpe < 1.0:
            recommendations.append("Consider improving risk-adjusted returns through better position sizing or entry timing")

        # Drawdown recommendations
        max_dd = abs(metrics.get('max_drawdown_pct', 0))
        if max_dd > 0.15:
            recommendations.append("Implement stricter risk management to reduce maximum drawdown")

        # Win rate recommendations
        win_rate = metrics.get('win_rate', 0)
        if win_rate < 0.45:
            recommendations.append("Focus on improving trade selection criteria to increase win rate")

        # Profit factor recommendations
        profit_factor = metrics.get('profit_factor', 0)
        if profit_factor < 1.5:
            recommendations.append("Optimize take-profit and stop-loss levels to improve profit factor")

        # Trading frequency recommendations
        if trades_df is not None and len(trades_df) < 50:
            recommendations.append("Consider increasing trading frequency for better statistical significance")
        elif trades_df is not None and len(trades_df) > 500:
            recommendations.append("Evaluate if high trading frequency is adding value or increasing costs")

        # Default recommendation if performance is good
        if not recommendations:
            recommendations.append("Performance metrics are within acceptable ranges - continue current strategy")

        return recommendations

    def _calculate_avg_trade_duration(self, trades_df: pd.DataFrame) -> str:
        """Calculate average trade duration in human-readable format."""
        try:
            if 'entry_time' in trades_df.columns and 'exit_time' in trades_df.columns:
                durations = (pd.to_datetime(trades_df['exit_time']) -
                           pd.to_datetime(trades_df['entry_time']))
                avg_duration = durations.mean()

                if avg_duration.total_seconds() < 3600:
                    return f"{avg_duration.total_seconds()/60:.0f} minutes"
                elif avg_duration.total_seconds() < 86400:
                    return f"{avg_duration.total_seconds()/3600:.1f} hours"
                else:
                    return f"{avg_duration.days:.1f} days"
            return "N/A"
        except:
            return "N/A"

    def _format_percentage(self, value: float) -> str:
        """Format a decimal as a percentage."""
        return f"{value * 100:.2f}%"

    def _format_currency(self, value: float) -> str:
        """Format a value as currency."""
        return f"${value:,.2f}"


class PerformanceAnalyzer:
    """
    Comprehensive performance analysis and reporting system from the main file.
    Generates detailed performance reports with visualizations and comparisons.
    """

    def __init__(self, config):
        """Initialize performance analyzer."""
        self.config = config

    def generate_full_report(self, trades_df: Optional[pd.DataFrame], equity_curve: Optional[pd.Series],
                           cycle_metrics: List[Dict], aggregated_shap: Optional[pd.DataFrame] = None,
                           framework_memory: Optional[Dict] = None, aggregated_daily_dd: Optional[List[Dict]] = None,
                           last_classification_report: str = "N/A") -> Dict[str, Any]:
        """
        Generate comprehensive performance report with all analysis components.

        Args:
            trades_df: DataFrame containing trade execution data
            equity_curve: Series containing equity curve progression
            cycle_metrics: List of metrics from each walk-forward cycle
            aggregated_shap: SHAP feature importance summary
            framework_memory: Historical performance memory
            aggregated_daily_dd: Daily drawdown analysis data
            last_classification_report: Model classification performance report

        Returns:
            Dictionary containing calculated performance metrics
        """
        logger.info("-> Stage 4: Generating Final Performance Report...")

        try:
            # Generate visualizations
            if equity_curve is not None and len(equity_curve) > 1:
                self.plot_equity_curve(equity_curve)

            if aggregated_shap is not None and not aggregated_shap.empty:
                self.plot_shap_summary(aggregated_shap)

            # Calculate comprehensive metrics
            metrics = self._calculate_metrics(trades_df, equity_curve) if trades_df is not None and not trades_df.empty else {}

            # Generate detailed text report
            self.generate_text_report(metrics, cycle_metrics, aggregated_shap, framework_memory,
                                    aggregated_daily_dd, last_classification_report)

            logger.info(f"[SUCCESS] Final report generated and saved to: {self.config.REPORT_SAVE_PATH}")
            return metrics

        except Exception as e:
            logger.error(f"Error generating full performance report: {e}")
            return {}

    def plot_equity_curve(self, equity_curve: pd.Series):
        """
        Generate and save equity curve visualization.

        Args:
            equity_curve: Series containing equity progression over time
        """
        try:
            import matplotlib.pyplot as plt

            plt.style.use('seaborn-v0_8-darkgrid')
            plt.figure(figsize=(16, 8))
            plt.plot(equity_curve.values, color='dodgerblue', linewidth=2)
            plt.title(f"{self.config.nickname or self.config.REPORT_LABEL} - Walk-Forward Equity Curve",
                     fontsize=16, weight='bold')
            plt.xlabel("Trade Event Number (including partial closes)", fontsize=12)
            plt.ylabel("Equity ($)", fontsize=12)
            plt.grid(True, which='both', linestyle=':')

            plt.savefig(self.config.PLOT_SAVE_PATH)
            plt.close()
            logger.info(f"  - Equity curve plot saved to: {self.config.PLOT_SAVE_PATH}")

        except Exception as e:
            logger.error(f"  - Failed to save equity curve plot: {e}")

    def plot_shap_summary(self, shap_summary: pd.DataFrame):
        """
        Generate and save SHAP feature importance visualization.

        Args:
            shap_summary: DataFrame containing SHAP importance values
        """
        try:
            import matplotlib.pyplot as plt

            plt.style.use('seaborn-v0_8-darkgrid')
            plt.figure(figsize=(12, 10))
            shap_summary.head(20).sort_values(by='SHAP_Importance').plot(kind='barh', legend=False, color='mediumseagreen')
            title_str = f"{self.config.nickname or self.config.REPORT_LABEL} ({self.config.strategy_name}) - Aggregated Feature Importance"
            plt.title(title_str, fontsize=16, weight='bold')
            plt.xlabel("Mean Absolute SHAP Value", fontsize=12)
            plt.ylabel("Feature", fontsize=12)
            plt.tight_layout()

            # Construct full path for aggregated SHAP summary
            final_shap_path = os.path.join(self.config.SHAP_PLOT_PATH, f"{self.config.run_timestamp}_aggregated_shap_summary.png")
            plt.savefig(final_shap_path)
            plt.close()
            logger.info(f"  - SHAP summary plot saved to: {final_shap_path}")

        except Exception as e:
            logger.error(f"  - Failed to save SHAP plot: {e}")

    def _calculate_metrics(self, trades_df: pd.DataFrame, equity_curve: pd.Series) -> Dict[str, Any]:
        """
        Calculate comprehensive performance metrics from trade data and equity curve.

        Args:
            trades_df: DataFrame containing trade execution data
            equity_curve: Series containing equity progression

        Returns:
            Dictionary containing calculated performance metrics
        """
        try:
            m = {}

            # Basic capital metrics
            m['initial_capital'] = self.config.INITIAL_CAPITAL
            m['ending_capital'] = equity_curve.iloc[-1]
            m['total_net_profit'] = m['ending_capital'] - m['initial_capital']
            m['net_profit_pct'] = (m['total_net_profit'] / m['initial_capital']) if m['initial_capital'] > 0 else 0

            # Profit/Loss analysis
            wins = trades_df[trades_df['PNL'] > 0]
            losses = trades_df[trades_df['PNL'] < 0]
            m['gross_profit'] = wins['PNL'].sum()
            m['gross_loss'] = abs(losses['PNL'].sum())
            m['profit_factor'] = m['gross_profit'] / m['gross_loss'] if m['gross_loss'] > 0 else np.inf

            # Trade statistics
            m['total_trade_events'] = len(trades_df)
            final_exits_df = trades_df[trades_df['ExitReason'].str.contains("Stop Loss|Take Profit", na=False)]
            m['total_trades'] = len(final_exits_df)

            m['winning_trades'] = len(final_exits_df[final_exits_df['PNL'] > 0])
            m['losing_trades'] = len(final_exits_df[final_exits_df['PNL'] < 0])
            m['win_rate'] = m['winning_trades'] / m['total_trades'] if m['total_trades'] > 0 else 0

            m['avg_win_amount'] = wins['PNL'].mean() if len(wins) > 0 else 0
            m['avg_loss_amount'] = abs(losses['PNL'].mean()) if len(losses) > 0 else 0

            # Payoff analysis
            avg_full_win = final_exits_df[final_exits_df['PNL'] > 0]['PNL'].mean() if len(final_exits_df[final_exits_df['PNL'] > 0]) > 0 else 0
            avg_full_loss = abs(final_exits_df[final_exits_df['PNL'] < 0]['PNL'].mean()) if len(final_exits_df[final_exits_df['PNL'] < 0]) > 0 else 0
            m['payoff_ratio'] = avg_full_win / avg_full_loss if avg_full_loss > 0 else np.inf
            m['expected_payoff'] = (m['win_rate'] * avg_full_win) - ((1 - m['win_rate']) * avg_full_loss) if m['total_trades'] > 0 else 0

            return m

        except Exception as e:
            logger.error(f"Error calculating performance metrics: {e}")
            return {}

    def _get_comparison_block(self, metrics: Dict, memory: Dict, ledger: Dict, width: int) -> str:
        """Generate performance comparison block."""
        try:
            champion = memory.get('champion_config')
            historical_runs = memory.get('historical_runs', [])
            previous_run = historical_runs[-1] if historical_runs else None

            def get_data(source: Optional[Dict], key: str, is_percent: bool = False) -> str:
                if not source: return "N/A"
                val = source.get(key) if isinstance(source, dict) and key in source else source.get("final_metrics", {}).get(key) if isinstance(source, dict) else None
                if val is None or not isinstance(val, (int, float)): return "N/A"
                return f"{val:.2f}%" if is_percent else f"{val:.2f}"

            def get_info(source: Optional[Dict], key: str) -> str:
                if not source: return "N/A"
                if hasattr(source, key):
                    return str(getattr(source, key, 'N/A'))
                elif isinstance(source, dict):
                    return str(source.get(key, 'N/A'))
                return "N/A"

            def get_nickname(source: Optional[Dict]) -> str:
                if not source: return "N/A"
                version_key = 'REPORT_LABEL' if hasattr(source, 'REPORT_LABEL') else 'script_version'
                version = get_info(source, version_key)
                return ledger.get(version, "N/A")

            c_nick, p_nick, champ_nick = get_nickname(self.config), get_nickname(previous_run), get_nickname(champion)
            c_strat, p_strat, champ_strat = get_info(self.config, 'strategy_name'), get_info(previous_run, 'strategy_name'), get_info(champion, 'strategy_name')
            c_mar, p_mar, champ_mar = get_data(metrics, 'mar_ratio'), get_data(previous_run, 'mar_ratio'), get_data(champion, 'mar_ratio')
            c_mdd, p_mdd, champ_mdd = get_data(metrics, 'max_drawdown_pct', True), get_data(previous_run, 'max_drawdown_pct', True), get_data(champion, 'max_drawdown_pct', True)
            c_pf, p_pf, champ_pf = get_data(metrics, 'profit_factor'), get_data(previous_run, 'profit_factor'), get_data(champion, 'profit_factor')

            col_w = (width - 5) // 4

            # Helper function to truncate text if it's too long for the column
            def _fit_text(text: str, max_width: int) -> str:
                if len(text) <= max_width:
                    return text
                else:
                    return text[:max_width-3] + "..."

            # Ensure all text fits within column widths
            c_nick = _fit_text(c_nick, col_w)
            p_nick = _fit_text(p_nick, col_w)
            champ_nick = _fit_text(champ_nick, col_w)
            c_strat = _fit_text(c_strat, col_w)
            p_strat = _fit_text(p_strat, col_w)
            champ_strat = _fit_text(champ_strat, col_w)

            header = f"| {'Metric'.ljust(col_w-1)}|{'Current Run'.center(col_w)}|{'Previous Run'.center(col_w)}|{'All-Time Champion'.center(col_w)}|"
            sep = f"+{'-'*(col_w)}+{'-'*(col_w)}+{'-'*(col_w)}+{'-'*(col_w)}+"
            rows = [
                f"| {'Run Nickname'.ljust(col_w-1)}|{c_nick.center(col_w)}|{p_nick.center(col_w)}|{champ_nick.center(col_w)}|",
                f"| {'Strategy'.ljust(col_w-1)}|{c_strat.center(col_w)}|{p_strat.center(col_w)}|{champ_strat.center(col_w)}|",
                f"| {'MAR Ratio'.ljust(col_w-1)}|{c_mar.center(col_w)}|{p_mar.center(col_w)}|{champ_mar.center(col_w)}|",
                f"| {'Max Drawdown'.ljust(col_w-1)}|{c_mdd.center(col_w)}|{p_mdd.center(col_w)}|{champ_mdd.center(col_w)}|",
                f"| {'Profit Factor'.ljust(col_w-1)}|{c_pf.center(col_w)}|{p_pf.center(col_w)}|{champ_pf.center(col_w)}|"
            ]
            return "\n".join([header, sep] + rows)

        except Exception as e:
            logger.error(f"Error generating comparison block: {e}")
            return "Comparison data unavailable"

    def generate_text_report(self, m: Dict[str, Any], cycle_metrics: List[Dict],
                           aggregated_shap: Optional[pd.DataFrame] = None,
                           framework_memory: Optional[Dict] = None,
                           aggregated_daily_dd: Optional[List[Dict]] = None,
                           last_classification_report: str = "N/A"):
        """Generate comprehensive text report with professional formatting."""
        import textwrap

        WIDTH = 90  # Fixed width for A4 compatibility

        def _box_top(w): return f"+{'-' * (w-2)}+"
        def _box_mid(w): return f"+{'-' * (w-2)}+"
        def _box_bot(w): return f"+{'-' * (w-2)}+"

        def _box_line(text, w):
            """Wraps long text across multiple lines within the border"""
            max_content_width = w - 4  # Account for "| " and " |"
            if len(text) <= max_content_width:
                padding = max_content_width - len(text)
                return [f"| {text}{' ' * padding} |"]
            else:
                # Wrap the text and return multiple lines
                wrapped_lines = textwrap.wrap(text, width=max_content_width)
                result = []
                for line in wrapped_lines:
                    padding = max_content_width - len(line)
                    result.append(f"| {line}{' ' * padding} |")
                return result

        def _box_title(title, w):
            """Centers title, wrapping if necessary"""
            max_content_width = w - 4
            if len(title) <= max_content_width:
                return [f"| {title.center(max_content_width)} |"]
            else:
                # For very long titles, just left-align and wrap
                return _box_line(title, w)

        def _box_text_kv(key, val, w):
            """Formats key-value pairs, wrapping if necessary"""
            val_str = str(val)
            full_text = f"{key} {val_str}"
            max_content_width = w - 4

            if len(full_text) <= max_content_width:
                padding = max_content_width - len(full_text)
                return [f"| {key}{' ' * max(1, padding)}{val_str} |"]
            else:
                # If too long, try to keep key and value on same line if possible
                if len(key) + len(val_str) + 1 <= max_content_width:
                    padding = max_content_width - len(key) - len(val_str)
                    return [f"| {key}{' ' * max(1, padding)}{val_str} |"]
                else:
                    # Split across lines: key on first line, value on subsequent lines
                    result = []
                    key_padding = max_content_width - len(key)
                    result.append(f"| {key}{' ' * key_padding} |")

                    # Wrap the value across multiple lines with indentation
                    wrapped_values = textwrap.wrap(val_str, width=max_content_width - 4)  # Leave space for indentation
                    for val_line in wrapped_values:
                        val_padding = max_content_width - len(val_line) - 4
                        result.append(f"|     {val_line}{' ' * val_padding} |")
                    return result

        try:
            ledger = {}
            if hasattr(self.config, 'NICKNAME_LEDGER_PATH') and self.config.NICKNAME_LEDGER_PATH and os.path.exists(self.config.NICKNAME_LEDGER_PATH):
                try:
                    with open(self.config.NICKNAME_LEDGER_PATH, 'r') as f:
                        ledger = json.load(f)
                except (json.JSONDecodeError, IOError):
                    logger.warning("Could not load nickname ledger for reporting.")

            # Build the actual report with the calculated width
            report = [_box_top(WIDTH)]
            report.extend(_box_title('ADAPTIVE WALK-FORWARD PERFORMANCE REPORT', WIDTH))
            report.append(_box_mid(WIDTH))
            report.extend(_box_line(f"Nickname: {getattr(self.config, 'nickname', 'N/A')} ({getattr(self.config, 'strategy_name', 'N/A')})", WIDTH))
            report.extend(_box_line(f"Version: {getattr(self.config, 'REPORT_LABEL', 'N/A')}", WIDTH))
            report.extend(_box_line(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", WIDTH))

            if hasattr(self.config, 'analysis_notes') and self.config.analysis_notes:
                report.extend(_box_line(f"AI Notes: {self.config.analysis_notes}", WIDTH))

            if framework_memory:
                report.append(_box_mid(WIDTH))
                report.extend(_box_title('I. PERFORMANCE vs. HISTORY', WIDTH))
                report.append(_box_mid(WIDTH))
                report.append(self._get_comparison_block(m, framework_memory, ledger, WIDTH))

            # Add main sections
            sections = {
                "II. EXECUTIVE SUMMARY": [
                    (f"Initial Capital:", f"${m.get('initial_capital', 0):,.2f}"),
                    (f"Ending Capital:", f"${m.get('ending_capital', 0):,.2f}"),
                    (f"Total Net Profit:", f"${m.get('total_net_profit', 0):,.2f} ({m.get('net_profit_pct', 0):.2%})"),
                    (f"Profit Factor:", f"{m.get('profit_factor', 0):.2f}"),
                    (f"Win Rate (Full Trades):", f"{m.get('win_rate', 0):.2%}"),
                    (f"Expected Payoff:", f"${m.get('expected_payoff', 0):,.2f}")
                ],
                "III. CORE PERFORMANCE METRICS": [
                    (f"Annual Return (CAGR):", f"{m.get('cagr', 0):.2%}"),
                    (f"Sharpe Ratio (annual):", f"{m.get('sharpe_ratio', 0):.2f}"),
                    (f"Sortino Ratio (annual):", f"{m.get('sortino_ratio', 0):.2f}"),
                    (f"Calmar Ratio / MAR:", f"{m.get('mar_ratio', 0):.2f}")
                ]
            }

            for title, data in sections.items():
                if not m: continue
                report.append(_box_mid(WIDTH))
                report.extend(_box_title(title, WIDTH))
                report.append(_box_mid(WIDTH))
                for key, val in data:
                    report.extend(_box_text_kv(key, val, WIDTH))

            report.append(_box_bot(WIDTH))
            final_report = "\n".join(report)
            logger.info("\n" + final_report)

            # Save report
            try:
                with open(self.config.REPORT_SAVE_PATH, 'w', encoding='utf-8') as f:
                    f.write(final_report)
            except IOError as e:
                logger.error(f"  - Failed to save text report: {e}", exc_info=True)

        except Exception as e:
            logger.error(f"Error generating text report: {e}")


def calculate_advanced_metrics(trades_df: pd.DataFrame, equity_curve: pd.Series,
                             initial_capital: float, benchmark_returns: pd.Series = None) -> Dict[str, Any]:
    """
    Calculate advanced performance metrics beyond basic statistics.

    Args:
        trades_df: DataFrame containing trade execution data
        equity_curve: Series containing equity progression
        initial_capital: Initial capital amount
        benchmark_returns: Optional benchmark returns for comparison

    Returns:
        Dictionary containing advanced performance metrics
    """
    try:
        metrics = {}

        if trades_df.empty or equity_curve.empty:
            logger.warning("Empty data provided for advanced metrics calculation")
            return metrics

        # Basic metrics
        ending_capital = equity_curve.iloc[-1]
        total_return = (ending_capital - initial_capital) / initial_capital

        # Time-based calculations
        if hasattr(equity_curve.index, 'to_pydatetime'):
            start_date = equity_curve.index[0]
            end_date = equity_curve.index[-1]
            days = (end_date - start_date).days
        else:
            days = len(equity_curve)

        years = max(days / 365.25, 1/365.25)

        # Advanced return metrics
        metrics['total_return'] = total_return
        metrics['annualized_return'] = (1 + total_return) ** (1/years) - 1
        metrics['cagr'] = metrics['annualized_return']

        # Risk metrics
        daily_returns = equity_curve.pct_change().fillna(0)

        if len(daily_returns) > 1:
            metrics['volatility'] = daily_returns.std() * np.sqrt(252)
            metrics['sharpe_ratio'] = (metrics['annualized_return'] / metrics['volatility']) if metrics['volatility'] > 0 else 0

            # Sortino ratio (downside deviation)
            downside_returns = daily_returns[daily_returns < 0]
            downside_std = downside_returns.std() * np.sqrt(252) if len(downside_returns) > 0 else 0
            metrics['sortino_ratio'] = (metrics['annualized_return'] / downside_std) if downside_std > 0 else np.inf

            # Maximum drawdown analysis
            running_max = equity_curve.cummax()
            drawdown = (equity_curve - running_max) / running_max
            metrics['max_drawdown'] = abs(drawdown.min())
            metrics['max_drawdown_duration'] = self._calculate_max_drawdown_duration(drawdown)

            # Calmar ratio
            metrics['calmar_ratio'] = metrics['annualized_return'] / metrics['max_drawdown'] if metrics['max_drawdown'] > 0 else np.inf

            # Value at Risk (VaR)
            metrics['var_95'] = np.percentile(daily_returns, 5)
            metrics['var_99'] = np.percentile(daily_returns, 1)

            # Expected Shortfall (Conditional VaR)
            var_95_threshold = metrics['var_95']
            tail_losses = daily_returns[daily_returns <= var_95_threshold]
            metrics['expected_shortfall'] = tail_losses.mean() if len(tail_losses) > 0 else 0

        # Trade-based metrics
        if 'PNL' in trades_df.columns:
            trade_returns = trades_df['PNL'] / initial_capital

            metrics['win_rate'] = (trades_df['PNL'] > 0).mean()
            metrics['avg_win'] = trades_df[trades_df['PNL'] > 0]['PNL'].mean() if (trades_df['PNL'] > 0).any() else 0
            metrics['avg_loss'] = trades_df[trades_df['PNL'] < 0]['PNL'].mean() if (trades_df['PNL'] < 0).any() else 0
            metrics['profit_factor'] = abs(metrics['avg_win'] / metrics['avg_loss']) if metrics['avg_loss'] != 0 else np.inf

            # Consecutive wins/losses
            pnl_signs = np.sign(trades_df['PNL'])
            metrics['max_consecutive_wins'] = self._max_consecutive(pnl_signs, 1)
            metrics['max_consecutive_losses'] = self._max_consecutive(pnl_signs, -1)

            # Trade frequency
            metrics['trades_per_year'] = len(trades_df) / years
            metrics['avg_trade_duration'] = self._calculate_avg_trade_duration(trades_df)

        # Benchmark comparison
        if benchmark_returns is not None and len(benchmark_returns) > 0:
            benchmark_total_return = (1 + benchmark_returns).prod() - 1
            metrics['alpha'] = total_return - benchmark_total_return

            # Beta calculation
            if len(daily_returns) == len(benchmark_returns):
                covariance = np.cov(daily_returns, benchmark_returns)[0, 1]
                benchmark_variance = np.var(benchmark_returns)
                metrics['beta'] = covariance / benchmark_variance if benchmark_variance > 0 else 0

            # Information ratio
            active_returns = daily_returns - benchmark_returns[:len(daily_returns)]
            tracking_error = active_returns.std() * np.sqrt(252)
            metrics['information_ratio'] = (metrics['alpha'] / tracking_error) if tracking_error > 0 else 0

        # Risk-adjusted metrics
        risk_free_rate = 0.02  # Assume 2% risk-free rate
        excess_return = metrics['annualized_return'] - risk_free_rate
        metrics['sharpe_ratio_rf'] = (excess_return / metrics.get('volatility', 1)) if metrics.get('volatility', 0) > 0 else 0

        logger.info(f"Calculated {len(metrics)} advanced performance metrics")
        return metrics

    except Exception as e:
        logger.error(f"Error calculating advanced metrics: {e}")
        return {}

    def _calculate_max_drawdown_duration(self, drawdown_series: pd.Series) -> int:
        """Calculate maximum drawdown duration in days."""
        try:
            in_drawdown = drawdown_series < 0
            drawdown_periods = []
            current_period = 0

            for is_dd in in_drawdown:
                if is_dd:
                    current_period += 1
                else:
                    if current_period > 0:
                        drawdown_periods.append(current_period)
                        current_period = 0

            if current_period > 0:
                drawdown_periods.append(current_period)

            return max(drawdown_periods) if drawdown_periods else 0

        except Exception:
            return 0

    def _max_consecutive(self, series: pd.Series, value: int) -> int:
        """Calculate maximum consecutive occurrences of a value."""
        try:
            max_count = 0
            current_count = 0

            for val in series:
                if val == value:
                    current_count += 1
                    max_count = max(max_count, current_count)
                else:
                    current_count = 0

            return max_count

        except Exception:
            return 0

    def _calculate_avg_trade_duration(self, trades_df: pd.DataFrame) -> float:
        """Calculate average trade duration in days."""
        try:
            if 'EntryTime' in trades_df.columns and 'ExitTime' in trades_df.columns:
                entry_times = pd.to_datetime(trades_df['EntryTime'])
                exit_times = pd.to_datetime(trades_df['ExitTime'])
                durations = (exit_times - entry_times).dt.total_seconds() / (24 * 3600)  # Convert to days
                return durations.mean()
            return 0.0

        except Exception:
            return 0.0


def generate_performance_summary(metrics: Dict[str, Any], trades_df: pd.DataFrame = None,
                               cycle_metrics: List[Dict] = None, strategy_name: str = "Unknown") -> Dict[str, Any]:
    """
    Generate comprehensive performance summary with key insights and recommendations.

    Args:
        metrics: Dictionary of calculated performance metrics
        trades_df: Optional DataFrame containing trade data
        cycle_metrics: Optional list of cycle-by-cycle metrics
        strategy_name: Name of the trading strategy

    Returns:
        Dictionary containing performance summary and insights
    """
    try:
        summary = {
            "strategy_name": strategy_name,
            "summary_timestamp": datetime.now().isoformat(),
            "key_metrics": {},
            "performance_grade": "N/A",
            "strengths": [],
            "weaknesses": [],
            "recommendations": [],
            "risk_assessment": {},
            "cycle_analysis": {}
        }

        # Extract key metrics
        key_metrics = {
            "total_return": metrics.get("total_return", 0),
            "annualized_return": metrics.get("annualized_return", 0),
            "sharpe_ratio": metrics.get("sharpe_ratio", 0),
            "max_drawdown": metrics.get("max_drawdown", 0),
            "win_rate": metrics.get("win_rate", 0),
            "profit_factor": metrics.get("profit_factor", 0),
            "calmar_ratio": metrics.get("calmar_ratio", 0)
        }
        summary["key_metrics"] = key_metrics

        # Performance grading
        grade_score = 0
        if key_metrics["annualized_return"] > 0.15: grade_score += 2
        elif key_metrics["annualized_return"] > 0.08: grade_score += 1

        if key_metrics["sharpe_ratio"] > 1.5: grade_score += 2
        elif key_metrics["sharpe_ratio"] > 1.0: grade_score += 1

        if key_metrics["max_drawdown"] < 0.1: grade_score += 2
        elif key_metrics["max_drawdown"] < 0.2: grade_score += 1

        if key_metrics["win_rate"] > 0.6: grade_score += 1
        if key_metrics["profit_factor"] > 1.5: grade_score += 1

        grade_map = {0: "F", 1: "D", 2: "C", 3: "C+", 4: "B-", 5: "B", 6: "B+", 7: "A-", 8: "A", 9: "A+"}
        summary["performance_grade"] = grade_map.get(min(grade_score, 9), "F")

        # Identify strengths
        if key_metrics["sharpe_ratio"] > 1.5:
            summary["strengths"].append("Excellent risk-adjusted returns (Sharpe > 1.5)")
        if key_metrics["max_drawdown"] < 0.1:
            summary["strengths"].append("Low maximum drawdown (<10%)")
        if key_metrics["win_rate"] > 0.6:
            summary["strengths"].append("High win rate (>60%)")
        if key_metrics["profit_factor"] > 2.0:
            summary["strengths"].append("Strong profit factor (>2.0)")
        if key_metrics["calmar_ratio"] > 2.0:
            summary["strengths"].append("Excellent Calmar ratio (>2.0)")

        # Identify weaknesses
        if key_metrics["sharpe_ratio"] < 0.5:
            summary["weaknesses"].append("Poor risk-adjusted returns (Sharpe < 0.5)")
        if key_metrics["max_drawdown"] > 0.25:
            summary["weaknesses"].append("High maximum drawdown (>25%)")
        if key_metrics["win_rate"] < 0.4:
            summary["weaknesses"].append("Low win rate (<40%)")
        if key_metrics["profit_factor"] < 1.2:
            summary["weaknesses"].append("Weak profit factor (<1.2)")
        if key_metrics["annualized_return"] < 0:
            summary["weaknesses"].append("Negative annualized returns")

        # Generate recommendations
        if key_metrics["max_drawdown"] > 0.2:
            summary["recommendations"].append("Consider implementing stricter risk management")
        if key_metrics["win_rate"] < 0.45:
            summary["recommendations"].append("Review entry criteria to improve win rate")
        if key_metrics["profit_factor"] < 1.3:
            summary["recommendations"].append("Optimize take-profit and stop-loss levels")
        if key_metrics["sharpe_ratio"] < 1.0:
            summary["recommendations"].append("Focus on improving risk-adjusted returns")

        # Risk assessment
        risk_level = "Low"
        if key_metrics["max_drawdown"] > 0.15 or metrics.get("volatility", 0) > 0.25:
            risk_level = "Medium"
        if key_metrics["max_drawdown"] > 0.3 or metrics.get("volatility", 0) > 0.4:
            risk_level = "High"

        summary["risk_assessment"] = {
            "overall_risk_level": risk_level,
            "max_drawdown_pct": key_metrics["max_drawdown"] * 100,
            "volatility_pct": metrics.get("volatility", 0) * 100,
            "var_95_pct": metrics.get("var_95", 0) * 100,
            "risk_score": min(grade_score, 5)  # Risk score out of 5
        }

        # Trade analysis
        if trades_df is not None and not trades_df.empty:
            trade_analysis = {
                "total_trades": len(trades_df),
                "avg_trade_pnl": trades_df["PNL"].mean() if "PNL" in trades_df.columns else 0,
                "largest_win": trades_df["PNL"].max() if "PNL" in trades_df.columns else 0,
                "largest_loss": trades_df["PNL"].min() if "PNL" in trades_df.columns else 0,
                "trade_frequency": metrics.get("trades_per_year", 0)
            }
            summary["trade_analysis"] = trade_analysis

        # Cycle analysis
        if cycle_metrics:
            cycle_returns = [c.get("net_profit_pct", 0) for c in cycle_metrics]
            cycle_sharpes = [c.get("sharpe_ratio", 0) for c in cycle_metrics]

            summary["cycle_analysis"] = {
                "total_cycles": len(cycle_metrics),
                "profitable_cycles": sum(1 for r in cycle_returns if r > 0),
                "avg_cycle_return": np.mean(cycle_returns),
                "cycle_consistency": 1 - (np.std(cycle_returns) / abs(np.mean(cycle_returns))) if np.mean(cycle_returns) != 0 else 0,
                "best_cycle_return": max(cycle_returns) if cycle_returns else 0,
                "worst_cycle_return": min(cycle_returns) if cycle_returns else 0
            }

        # Overall assessment
        if grade_score >= 7:
            summary["overall_assessment"] = "Excellent performance with strong risk-adjusted returns"
        elif grade_score >= 5:
            summary["overall_assessment"] = "Good performance with room for improvement"
        elif grade_score >= 3:
            summary["overall_assessment"] = "Moderate performance requiring optimization"
        else:
            summary["overall_assessment"] = "Poor performance requiring significant improvements"

        logger.info(f"Generated performance summary for {strategy_name} with grade {summary['performance_grade']}")
        return summary

    except Exception as e:
        logger.error(f"Error generating performance summary: {e}")
        return {
            "strategy_name": strategy_name,
            "error": str(e),
            "summary_timestamp": datetime.now().isoformat()
        }


def calculate_risk_metrics(equity_curve: pd.Series, trades_df: pd.DataFrame = None,
                         benchmark_returns: pd.Series = None, confidence_levels: List[float] = None) -> Dict[str, Any]:
    """
    Calculate comprehensive risk metrics for portfolio analysis.

    Args:
        equity_curve: Series containing equity progression over time
        trades_df: Optional DataFrame containing trade data
        benchmark_returns: Optional benchmark returns for comparison
        confidence_levels: List of confidence levels for VaR calculation

    Returns:
        Dictionary containing comprehensive risk metrics
    """
    try:
        if confidence_levels is None:
            confidence_levels = [0.95, 0.99]

        risk_metrics = {}

        if equity_curve.empty:
            logger.warning("Empty equity curve provided for risk analysis")
            return risk_metrics

        # Calculate returns
        returns = equity_curve.pct_change().fillna(0)

        # Basic risk metrics
        risk_metrics['volatility_daily'] = returns.std()
        risk_metrics['volatility_annualized'] = returns.std() * np.sqrt(252)
        risk_metrics['skewness'] = returns.skew()
        risk_metrics['kurtosis'] = returns.kurtosis()

        # Drawdown analysis
        running_max = equity_curve.cummax()
        drawdown = (equity_curve - running_max) / running_max

        risk_metrics['max_drawdown'] = abs(drawdown.min())
        risk_metrics['current_drawdown'] = abs(drawdown.iloc[-1])
        risk_metrics['avg_drawdown'] = abs(drawdown[drawdown < 0].mean()) if (drawdown < 0).any() else 0

        # Drawdown duration analysis
        drawdown_periods = self._identify_drawdown_periods(drawdown)
        if drawdown_periods:
            durations = [period['duration'] for period in drawdown_periods]
            risk_metrics['max_drawdown_duration'] = max(durations)
            risk_metrics['avg_drawdown_duration'] = np.mean(durations)
            risk_metrics['drawdown_frequency'] = len(drawdown_periods) / len(equity_curve) * 252  # Annualized
        else:
            risk_metrics['max_drawdown_duration'] = 0
            risk_metrics['avg_drawdown_duration'] = 0
            risk_metrics['drawdown_frequency'] = 0

        # Value at Risk (VaR)
        for confidence in confidence_levels:
            percentile = (1 - confidence) * 100
            var_key = f'var_{int(confidence*100)}'
            risk_metrics[var_key] = np.percentile(returns, percentile)

            # Expected Shortfall (Conditional VaR)
            es_key = f'expected_shortfall_{int(confidence*100)}'
            tail_losses = returns[returns <= risk_metrics[var_key]]
            risk_metrics[es_key] = tail_losses.mean() if len(tail_losses) > 0 else 0

        # Downside risk metrics
        target_return = 0.0
        downside_returns = returns[returns < target_return]
        risk_metrics['downside_deviation'] = np.sqrt(np.mean(np.square(downside_returns - target_return))) if len(downside_returns) > 0 else 0
        risk_metrics['downside_deviation_annualized'] = risk_metrics['downside_deviation'] * np.sqrt(252)

        # Semi-variance
        risk_metrics['semi_variance'] = np.var(downside_returns) if len(downside_returns) > 0 else 0

        # Ulcer Index (measure of downside risk)
        squared_drawdowns = drawdown ** 2
        risk_metrics['ulcer_index'] = np.sqrt(squared_drawdowns.mean())

        # Pain Index (average drawdown)
        risk_metrics['pain_index'] = abs(drawdown.mean())

        # Risk-adjusted metrics
        if risk_metrics['volatility_annualized'] > 0:
            mean_return = returns.mean() * 252
            risk_metrics['sharpe_ratio'] = mean_return / risk_metrics['volatility_annualized']

            if risk_metrics['downside_deviation_annualized'] > 0:
                risk_metrics['sortino_ratio'] = mean_return / risk_metrics['downside_deviation_annualized']

            if risk_metrics['max_drawdown'] > 0:
                risk_metrics['calmar_ratio'] = mean_return / risk_metrics['max_drawdown']

        # Trade-based risk metrics
        if trades_df is not None and not trades_df.empty and 'PNL' in trades_df.columns:
            trade_pnls = trades_df['PNL']

            risk_metrics['trade_var_95'] = np.percentile(trade_pnls, 5)
            risk_metrics['trade_var_99'] = np.percentile(trade_pnls, 1)
            risk_metrics['largest_loss'] = trade_pnls.min()
            risk_metrics['avg_loss'] = trade_pnls[trade_pnls < 0].mean() if (trade_pnls < 0).any() else 0

            # Consecutive loss analysis
            loss_streaks = self._calculate_consecutive_losses(trade_pnls)
            risk_metrics['max_consecutive_losses'] = max(loss_streaks) if loss_streaks else 0
            risk_metrics['avg_consecutive_losses'] = np.mean(loss_streaks) if loss_streaks else 0

        # Benchmark comparison
        if benchmark_returns is not None and len(benchmark_returns) > 0:
            # Align returns with benchmark
            aligned_returns = returns[:len(benchmark_returns)]
            aligned_benchmark = benchmark_returns[:len(returns)]

            if len(aligned_returns) == len(aligned_benchmark):
                # Tracking error
                active_returns = aligned_returns - aligned_benchmark
                risk_metrics['tracking_error'] = active_returns.std() * np.sqrt(252)

                # Beta
                covariance = np.cov(aligned_returns, aligned_benchmark)[0, 1]
                benchmark_variance = np.var(aligned_benchmark)
                risk_metrics['beta'] = covariance / benchmark_variance if benchmark_variance > 0 else 0

                # Correlation
                risk_metrics['correlation_with_benchmark'] = np.corrcoef(aligned_returns, aligned_benchmark)[0, 1]

        # Risk concentration metrics
        if trades_df is not None and not trades_df.empty:
            # Position size analysis
            if 'PositionSize' in trades_df.columns:
                position_sizes = trades_df['PositionSize'].abs()
                risk_metrics['max_position_size'] = position_sizes.max()
                risk_metrics['avg_position_size'] = position_sizes.mean()
                risk_metrics['position_size_std'] = position_sizes.std()

        logger.info(f"Calculated {len(risk_metrics)} risk metrics")
        return risk_metrics

    except Exception as e:
        logger.error(f"Error calculating risk metrics: {e}")
        return {}

    def _identify_drawdown_periods(self, drawdown_series: pd.Series) -> List[Dict[str, Any]]:
        """Identify individual drawdown periods."""
        try:
            periods = []
            in_drawdown = False
            start_idx = None

            for i, dd in enumerate(drawdown_series):
                if dd < 0 and not in_drawdown:
                    # Start of drawdown
                    in_drawdown = True
                    start_idx = i
                elif dd >= 0 and in_drawdown:
                    # End of drawdown
                    in_drawdown = False
                    if start_idx is not None:
                        period_dd = drawdown_series.iloc[start_idx:i]
                        periods.append({
                            'start_idx': start_idx,
                            'end_idx': i-1,
                            'duration': i - start_idx,
                            'max_drawdown': abs(period_dd.min()),
                            'recovery_time': i - start_idx
                        })

            # Handle case where drawdown continues to end
            if in_drawdown and start_idx is not None:
                period_dd = drawdown_series.iloc[start_idx:]
                periods.append({
                    'start_idx': start_idx,
                    'end_idx': len(drawdown_series) - 1,
                    'duration': len(drawdown_series) - start_idx,
                    'max_drawdown': abs(period_dd.min()),
                    'recovery_time': None  # Still in drawdown
                })

            return periods

        except Exception:
            return []

    def _calculate_consecutive_losses(self, pnl_series: pd.Series) -> List[int]:
        """Calculate consecutive loss streaks."""
        try:
            streaks = []
            current_streak = 0

            for pnl in pnl_series:
                if pnl < 0:
                    current_streak += 1
                else:
                    if current_streak > 0:
                        streaks.append(current_streak)
                        current_streak = 0

            # Add final streak if it ends with losses
            if current_streak > 0:
                streaks.append(current_streak)

            return streaks

        except Exception:
            return []


def analyze_drawdown_periods(equity_curve: pd.Series, detailed_analysis: bool = True) -> Dict[str, Any]:
    """
    Perform detailed analysis of drawdown periods including recovery patterns.

    Args:
        equity_curve: Series containing equity progression over time
        detailed_analysis: Whether to include detailed period-by-period analysis

    Returns:
        Dictionary containing comprehensive drawdown analysis
    """
    try:
        analysis = {
            "summary": {},
            "periods": [],
            "statistics": {},
            "recovery_analysis": {}
        }

        if equity_curve.empty:
            logger.warning("Empty equity curve provided for drawdown analysis")
            return analysis

        # Calculate drawdown series
        running_max = equity_curve.cummax()
        drawdown = (equity_curve - running_max) / running_max

        # Identify drawdown periods
        periods = []
        in_drawdown = False
        start_idx = None
        peak_value = None

        for i, (idx, equity_val) in enumerate(equity_curve.items()):
            dd_val = drawdown.iloc[i]

            if dd_val < -0.001 and not in_drawdown:  # Start of significant drawdown (>0.1%)
                in_drawdown = True
                start_idx = i
                peak_value = running_max.iloc[i]

            elif dd_val >= -0.001 and in_drawdown:  # End of drawdown
                in_drawdown = False
                if start_idx is not None:
                    period_data = self._analyze_single_drawdown_period(
                        equity_curve.iloc[start_idx:i+1],
                        drawdown.iloc[start_idx:i+1],
                        start_idx, i, peak_value
                    )
                    periods.append(period_data)

        # Handle ongoing drawdown
        if in_drawdown and start_idx is not None:
            period_data = self._analyze_single_drawdown_period(
                equity_curve.iloc[start_idx:],
                drawdown.iloc[start_idx:],
                start_idx, len(equity_curve)-1, peak_value, ongoing=True
            )
            periods.append(period_data)

        analysis["periods"] = periods

        # Summary statistics
        if periods:
            drawdown_depths = [p["max_drawdown_pct"] for p in periods]
            durations = [p["duration_days"] for p in periods]
            recovery_times = [p["recovery_days"] for p in periods if p["recovery_days"] is not None]

            analysis["summary"] = {
                "total_drawdown_periods": len(periods),
                "max_drawdown_pct": max(drawdown_depths),
                "avg_drawdown_pct": np.mean(drawdown_depths),
                "median_drawdown_pct": np.median(drawdown_depths),
                "max_duration_days": max(durations),
                "avg_duration_days": np.mean(durations),
                "max_recovery_days": max(recovery_times) if recovery_times else None,
                "avg_recovery_days": np.mean(recovery_times) if recovery_times else None,
                "current_drawdown_pct": abs(drawdown.iloc[-1]) * 100,
                "time_in_drawdown_pct": sum(durations) / len(equity_curve) * 100
            }

            # Severity classification
            severe_periods = [p for p in periods if p["max_drawdown_pct"] > 20]
            moderate_periods = [p for p in periods if 10 <= p["max_drawdown_pct"] <= 20]
            minor_periods = [p for p in periods if p["max_drawdown_pct"] < 10]

            analysis["statistics"] = {
                "severe_drawdowns": len(severe_periods),
                "moderate_drawdowns": len(moderate_periods),
                "minor_drawdowns": len(minor_periods),
                "avg_time_between_drawdowns": self._calculate_avg_time_between_drawdowns(periods),
                "drawdown_frequency_per_year": len(periods) / (len(equity_curve) / 252),
                "worst_period": max(periods, key=lambda x: x["max_drawdown_pct"]) if periods else None
            }

            # Recovery analysis
            if recovery_times:
                analysis["recovery_analysis"] = {
                    "avg_recovery_ratio": np.mean([p["recovery_days"] / p["duration_days"]
                                                 for p in periods if p["recovery_days"] is not None]),
                    "fastest_recovery_days": min(recovery_times),
                    "slowest_recovery_days": max(recovery_times),
                    "recovery_success_rate": len(recovery_times) / len(periods),
                    "incomplete_recoveries": len(periods) - len(recovery_times)
                }

        # Risk assessment
        current_dd = abs(drawdown.iloc[-1]) * 100
        risk_level = "Low"
        if current_dd > 5:
            risk_level = "Medium"
        if current_dd > 15:
            risk_level = "High"
        if current_dd > 25:
            risk_level = "Critical"

        analysis["risk_assessment"] = {
            "current_risk_level": risk_level,
            "days_since_peak": self._days_since_peak(equity_curve),
            "recovery_probability": self._estimate_recovery_probability(periods, current_dd)
        }

        logger.info(f"Analyzed {len(periods)} drawdown periods")
        return analysis

    except Exception as e:
        logger.error(f"Error analyzing drawdown periods: {e}")
        return {"error": str(e)}

    def _analyze_single_drawdown_period(self, equity_segment: pd.Series, drawdown_segment: pd.Series,
                                      start_idx: int, end_idx: int, peak_value: float, ongoing: bool = False) -> Dict[str, Any]:
        """Analyze a single drawdown period in detail."""
        try:
            max_dd = abs(drawdown_segment.min())
            trough_idx = drawdown_segment.idxmin()

            # Calculate duration
            if hasattr(equity_segment.index, 'to_pydatetime'):
                duration_days = (equity_segment.index[-1] - equity_segment.index[0]).days
            else:
                duration_days = len(equity_segment)

            # Recovery analysis
            recovery_days = None
            if not ongoing:
                trough_position = list(drawdown_segment.index).index(trough_idx)
                recovery_segment = drawdown_segment.iloc[trough_position:]
                if len(recovery_segment) > 1:
                    if hasattr(recovery_segment.index, 'to_pydatetime'):
                        recovery_days = (recovery_segment.index[-1] - recovery_segment.index[0]).days
                    else:
                        recovery_days = len(recovery_segment) - 1

            return {
                "start_date": equity_segment.index[0],
                "end_date": equity_segment.index[-1] if not ongoing else None,
                "peak_value": peak_value,
                "trough_value": equity_segment.min(),
                "max_drawdown_pct": max_dd * 100,
                "duration_days": duration_days,
                "recovery_days": recovery_days,
                "ongoing": ongoing,
                "trough_date": trough_idx,
                "drawdown_velocity": (max_dd * 100) / duration_days if duration_days > 0 else 0
            }

        except Exception:
            return {}

    def _calculate_avg_time_between_drawdowns(self, periods: List[Dict]) -> float:
        """Calculate average time between drawdown periods."""
        try:
            if len(periods) < 2:
                return 0

            intervals = []
            for i in range(1, len(periods)):
                prev_end = periods[i-1]["end_date"]
                curr_start = periods[i]["start_date"]

                if prev_end and curr_start:
                    if hasattr(prev_end, 'to_pydatetime'):
                        interval = (curr_start - prev_end).days
                    else:
                        interval = 1  # Default interval
                    intervals.append(interval)

            return np.mean(intervals) if intervals else 0

        except Exception:
            return 0

    def _days_since_peak(self, equity_curve: pd.Series) -> int:
        """Calculate days since last equity peak."""
        try:
            peak_idx = equity_curve.idxmax()
            last_idx = equity_curve.index[-1]

            if hasattr(equity_curve.index, 'to_pydatetime'):
                return (last_idx - peak_idx).days
            else:
                return len(equity_curve) - list(equity_curve.index).index(peak_idx) - 1

        except Exception:
            return 0

    def _estimate_recovery_probability(self, periods: List[Dict], current_dd: float) -> float:
        """Estimate probability of recovery based on historical patterns."""
        try:
            if not periods:
                return 0.5  # Default probability

            # Find similar historical drawdowns
            similar_periods = [p for p in periods if abs(p["max_drawdown_pct"] - current_dd) < current_dd * 0.3]

            if not similar_periods:
                # Use all periods if no similar ones found
                similar_periods = periods

            # Calculate recovery rate
            recovered_periods = [p for p in similar_periods if not p["ongoing"] and p["recovery_days"] is not None]
            recovery_rate = len(recovered_periods) / len(similar_periods) if similar_periods else 0.5

            # Adjust based on current drawdown severity
            if current_dd > 30:
                recovery_rate *= 0.7  # Reduce probability for severe drawdowns
            elif current_dd < 5:
                recovery_rate = min(recovery_rate * 1.2, 0.95)  # Increase for minor drawdowns

            return max(0.1, min(0.95, recovery_rate))

        except Exception:
            return 0.5


def generate_detailed_report(performance_data: Dict[str, Any], trades_df: pd.DataFrame = None,
                           equity_curve: pd.Series = None, config = None,
                           output_format: str = "html") -> str:
    """
    Generate comprehensive detailed report in specified format.

    Args:
        performance_data: Dictionary containing performance metrics and analysis
        trades_df: Optional DataFrame containing trade data
        equity_curve: Optional Series containing equity progression
        config: Optional configuration object
        output_format: Output format ("html", "markdown", "text")

    Returns:
        Formatted report string
    """
    try:
        if output_format.lower() == "html":
            return _generate_html_report(performance_data, trades_df, equity_curve, config)
        elif output_format.lower() == "markdown":
            return _generate_markdown_report(performance_data, trades_df, equity_curve, config)
        else:
            return _generate_text_report(performance_data, trades_df, equity_curve, config)

    except Exception as e:
        logger.error(f"Error generating detailed report: {e}")
        return f"Error generating report: {e}"


def _generate_html_report(performance_data: Dict, trades_df: pd.DataFrame = None,
                         equity_curve: pd.Series = None, config = None) -> str:
    """Generate HTML formatted report."""
    try:
        html_content = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>AxiomEdge Trading Performance Report</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 20px; }}
                .header {{ background-color: #2c3e50; color: white; padding: 20px; text-align: center; }}
                .section {{ margin: 20px 0; padding: 15px; border: 1px solid #ddd; }}
                .metric {{ display: inline-block; margin: 10px; padding: 10px; background-color: #f8f9fa; border-radius: 5px; }}
                .positive {{ color: #27ae60; }}
                .negative {{ color: #e74c3c; }}
                table {{ width: 100%; border-collapse: collapse; margin: 10px 0; }}
                th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
                th {{ background-color: #f2f2f2; }}
            </style>
        </head>
        <body>
            <div class="header">
                <h1>AxiomEdge Trading Performance Report</h1>
                <p>Generated on {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
            </div>
        """

        # Strategy Information
        strategy_name = performance_data.get('strategy_name', 'Unknown')
        html_content += f"""
            <div class="section">
                <h2>Strategy Information</h2>
                <p><strong>Strategy:</strong> {strategy_name}</p>
                <p><strong>Report Period:</strong> {performance_data.get('report_period', 'N/A')}</p>
            </div>
        """

        # Key Metrics
        key_metrics = performance_data.get('key_metrics', {})
        html_content += """
            <div class="section">
                <h2>Key Performance Metrics</h2>
        """

        for metric, value in key_metrics.items():
            if isinstance(value, (int, float)):
                css_class = "positive" if value > 0 else "negative"
                formatted_value = f"{value:.2%}" if 'rate' in metric.lower() or 'ratio' in metric.lower() else f"{value:.4f}"
                html_content += f'<div class="metric"><strong>{metric.replace("_", " ").title()}:</strong> <span class="{css_class}">{formatted_value}</span></div>'

        html_content += "</div>"

        # Trade Analysis
        if trades_df is not None and not trades_df.empty:
            html_content += """
                <div class="section">
                    <h2>Trade Analysis</h2>
                    <table>
                        <tr><th>Metric</th><th>Value</th></tr>
            """

            trade_metrics = {
                "Total Trades": len(trades_df),
                "Winning Trades": len(trades_df[trades_df['PNL'] > 0]) if 'PNL' in trades_df.columns else 0,
                "Losing Trades": len(trades_df[trades_df['PNL'] < 0]) if 'PNL' in trades_df.columns else 0,
                "Average PNL": trades_df['PNL'].mean() if 'PNL' in trades_df.columns else 0,
                "Largest Win": trades_df['PNL'].max() if 'PNL' in trades_df.columns else 0,
                "Largest Loss": trades_df['PNL'].min() if 'PNL' in trades_df.columns else 0
            }

            for metric, value in trade_metrics.items():
                formatted_value = f"{value:.2f}" if isinstance(value, float) else str(value)
                html_content += f"<tr><td>{metric}</td><td>{formatted_value}</td></tr>"

            html_content += "</table></div>"

        # Risk Analysis
        risk_data = performance_data.get('risk_assessment', {})
        if risk_data:
            html_content += """
                <div class="section">
                    <h2>Risk Analysis</h2>
                    <table>
                        <tr><th>Risk Metric</th><th>Value</th></tr>
            """

            for metric, value in risk_data.items():
                if isinstance(value, (int, float)):
                    formatted_value = f"{value:.2%}" if 'pct' in metric.lower() else f"{value:.4f}"
                else:
                    formatted_value = str(value)
                html_content += f"<tr><td>{metric.replace('_', ' ').title()}</td><td>{formatted_value}</td></tr>"

            html_content += "</table></div>"

        html_content += """
            <div class="section">
                <h2>Disclaimer</h2>
                <p><em>This report is for informational purposes only. Past performance does not guarantee future results.
                Trading involves substantial risk and may not be suitable for all investors.</em></p>
            </div>
        </body>
        </html>
        """

        return html_content

    except Exception as e:
        logger.error(f"Error generating HTML report: {e}")
        return f"<html><body><h1>Error generating report: {e}</h1></body></html>"


def _generate_markdown_report(performance_data: Dict, trades_df: pd.DataFrame = None,
                            equity_curve: pd.Series = None, config = None) -> str:
    """Generate Markdown formatted report."""
    try:
        md_content = f"""# AxiomEdge Trading Performance Report

**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

## Strategy Information
- **Strategy:** {performance_data.get('strategy_name', 'Unknown')}
- **Report Period:** {performance_data.get('report_period', 'N/A')}

## Key Performance Metrics

"""

        key_metrics = performance_data.get('key_metrics', {})
        for metric, value in key_metrics.items():
            if isinstance(value, (int, float)):
                formatted_value = f"{value:.2%}" if 'rate' in metric.lower() or 'ratio' in metric.lower() else f"{value:.4f}"
                md_content += f"- **{metric.replace('_', ' ').title()}:** {formatted_value}\n"

        # Trade Analysis
        if trades_df is not None and not trades_df.empty:
            md_content += "\n## Trade Analysis\n\n"
            md_content += "| Metric | Value |\n|--------|-------|\n"

            trade_metrics = {
                "Total Trades": len(trades_df),
                "Winning Trades": len(trades_df[trades_df['PNL'] > 0]) if 'PNL' in trades_df.columns else 0,
                "Losing Trades": len(trades_df[trades_df['PNL'] < 0]) if 'PNL' in trades_df.columns else 0,
                "Average PNL": trades_df['PNL'].mean() if 'PNL' in trades_df.columns else 0
            }

            for metric, value in trade_metrics.items():
                formatted_value = f"{value:.2f}" if isinstance(value, float) else str(value)
                md_content += f"| {metric} | {formatted_value} |\n"

        # Risk Analysis
        risk_data = performance_data.get('risk_assessment', {})
        if risk_data:
            md_content += "\n## Risk Analysis\n\n"
            md_content += "| Risk Metric | Value |\n|-------------|-------|\n"

            for metric, value in risk_data.items():
                if isinstance(value, (int, float)):
                    formatted_value = f"{value:.2%}" if 'pct' in metric.lower() else f"{value:.4f}"
                else:
                    formatted_value = str(value)
                md_content += f"| {metric.replace('_', ' ').title()} | {formatted_value} |\n"

        md_content += "\n---\n*This report is for informational purposes only. Past performance does not guarantee future results.*"

        return md_content

    except Exception as e:
        logger.error(f"Error generating Markdown report: {e}")
        return f"# Error\n\nError generating report: {e}"


def _generate_text_report(performance_data: Dict, trades_df: pd.DataFrame = None,
                        equity_curve: pd.Series = None, config = None) -> str:
    """Generate plain text formatted report."""
    try:
        text_content = f"""
{'='*80}
AXIOM EDGE TRADING PERFORMANCE REPORT
{'='*80}

Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
Strategy: {performance_data.get('strategy_name', 'Unknown')}
Report Period: {performance_data.get('report_period', 'N/A')}

{'='*80}
KEY PERFORMANCE METRICS
{'='*80}

"""

        key_metrics = performance_data.get('key_metrics', {})
        for metric, value in key_metrics.items():
            if isinstance(value, (int, float)):
                formatted_value = f"{value:.2%}" if 'rate' in metric.lower() or 'ratio' in metric.lower() else f"{value:.4f}"
                text_content += f"{metric.replace('_', ' ').title():<30}: {formatted_value:>15}\n"

        # Trade Analysis
        if trades_df is not None and not trades_df.empty:
            text_content += f"\n{'='*80}\nTRADE ANALYSIS\n{'='*80}\n\n"

            trade_metrics = {
                "Total Trades": len(trades_df),
                "Winning Trades": len(trades_df[trades_df['PNL'] > 0]) if 'PNL' in trades_df.columns else 0,
                "Losing Trades": len(trades_df[trades_df['PNL'] < 0]) if 'PNL' in trades_df.columns else 0,
                "Average PNL": trades_df['PNL'].mean() if 'PNL' in trades_df.columns else 0,
                "Largest Win": trades_df['PNL'].max() if 'PNL' in trades_df.columns else 0,
                "Largest Loss": trades_df['PNL'].min() if 'PNL' in trades_df.columns else 0
            }

            for metric, value in trade_metrics.items():
                formatted_value = f"{value:.2f}" if isinstance(value, float) else str(value)
                text_content += f"{metric:<30}: {formatted_value:>15}\n"

        # Risk Analysis
        risk_data = performance_data.get('risk_assessment', {})
        if risk_data:
            text_content += f"\n{'='*80}\nRISK ANALYSIS\n{'='*80}\n\n"

            for metric, value in risk_data.items():
                if isinstance(value, (int, float)):
                    formatted_value = f"{value:.2%}" if 'pct' in metric.lower() else f"{value:.4f}"
                else:
                    formatted_value = str(value)
                text_content += f"{metric.replace('_', ' ').title():<30}: {formatted_value:>15}\n"

        text_content += f"\n{'='*80}\nDISCLAIMER\n{'='*80}\n"
        text_content += "This report is for informational purposes only.\nPast performance does not guarantee future results.\n"

        return text_content

    except Exception as e:
        logger.error(f"Error generating text report: {e}")
        return f"Error generating report: {e}"


def create_visualization_data(equity_curve: pd.Series = None, trades_df: pd.DataFrame = None,
                            performance_metrics: Dict = None, drawdown_analysis: Dict = None) -> Dict[str, Any]:
    """
    Create structured data for visualization components.

    Args:
        equity_curve: Series containing equity progression over time
        trades_df: DataFrame containing trade data
        performance_metrics: Dictionary of performance metrics
        drawdown_analysis: Dictionary containing drawdown analysis

    Returns:
        Dictionary containing visualization-ready data structures
    """
    try:
        viz_data = {
            "equity_curve": {},
            "trade_distribution": {},
            "performance_metrics": {},
            "drawdown_chart": {},
            "monthly_returns": {},
            "risk_metrics": {},
            "metadata": {
                "generated_at": datetime.now().isoformat(),
                "data_points": 0
            }
        }

        # Equity curve data
        if equity_curve is not None and not equity_curve.empty:
            viz_data["equity_curve"] = {
                "dates": [d.isoformat() if hasattr(d, 'isoformat') else str(d) for d in equity_curve.index],
                "values": equity_curve.tolist(),
                "returns": equity_curve.pct_change().fillna(0).tolist(),
                "cumulative_returns": ((equity_curve / equity_curve.iloc[0]) - 1).tolist(),
                "rolling_max": equity_curve.cummax().tolist()
            }
            viz_data["metadata"]["data_points"] = len(equity_curve)

        # Trade distribution data
        if trades_df is not None and not trades_df.empty:
            if 'PNL' in trades_df.columns:
                pnl_data = trades_df['PNL']

                viz_data["trade_distribution"] = {
                    "pnl_values": pnl_data.tolist(),
                    "win_loss_counts": {
                        "wins": len(pnl_data[pnl_data > 0]),
                        "losses": len(pnl_data[pnl_data < 0]),
                        "breakeven": len(pnl_data[pnl_data == 0])
                    },
                    "pnl_histogram": self._create_histogram_data(pnl_data, bins=20),
                    "cumulative_pnl": pnl_data.cumsum().tolist(),
                    "trade_sequence": list(range(1, len(pnl_data) + 1))
                }

                # Monthly aggregation if dates available
                if 'ExitTime' in trades_df.columns:
                    try:
                        trades_with_dates = trades_df.copy()
                        trades_with_dates['ExitTime'] = pd.to_datetime(trades_with_dates['ExitTime'])
                        trades_with_dates['Month'] = trades_with_dates['ExitTime'].dt.to_period('M')
                        monthly_pnl = trades_with_dates.groupby('Month')['PNL'].sum()

                        viz_data["monthly_returns"] = {
                            "months": [str(m) for m in monthly_pnl.index],
                            "returns": monthly_pnl.tolist(),
                            "cumulative": monthly_pnl.cumsum().tolist()
                        }
                    except Exception as e:
                        logger.warning(f"Could not create monthly returns data: {e}")

        # Performance metrics visualization
        if performance_metrics:
            # Key metrics for dashboard
            key_metrics = {
                "total_return": performance_metrics.get("total_return", 0),
                "sharpe_ratio": performance_metrics.get("sharpe_ratio", 0),
                "max_drawdown": performance_metrics.get("max_drawdown", 0),
                "win_rate": performance_metrics.get("win_rate", 0),
                "profit_factor": performance_metrics.get("profit_factor", 0),
                "calmar_ratio": performance_metrics.get("calmar_ratio", 0)
            }

            viz_data["performance_metrics"] = {
                "key_metrics": key_metrics,
                "metric_names": list(key_metrics.keys()),
                "metric_values": list(key_metrics.values()),
                "benchmark_comparison": self._create_benchmark_data(key_metrics),
                "performance_radar": self._create_radar_chart_data(key_metrics)
            }

        # Drawdown visualization
        if equity_curve is not None and not equity_curve.empty:
            running_max = equity_curve.cummax()
            drawdown = (equity_curve - running_max) / running_max

            viz_data["drawdown_chart"] = {
                "dates": [d.isoformat() if hasattr(d, 'isoformat') else str(d) for d in equity_curve.index],
                "drawdown_pct": (drawdown * 100).tolist(),
                "underwater_curve": drawdown.tolist(),
                "recovery_periods": self._identify_recovery_periods(drawdown)
            }

        # Risk metrics visualization
        if performance_metrics:
            risk_metrics = {
                "volatility": performance_metrics.get("volatility", 0),
                "var_95": performance_metrics.get("var_95", 0),
                "expected_shortfall": performance_metrics.get("expected_shortfall", 0),
                "max_consecutive_losses": performance_metrics.get("max_consecutive_losses", 0)
            }

            viz_data["risk_metrics"] = {
                "risk_values": risk_metrics,
                "risk_gauge": self._create_risk_gauge_data(risk_metrics),
                "risk_breakdown": self._create_risk_breakdown_data(risk_metrics)
            }

        # Additional chart data
        if drawdown_analysis:
            viz_data["drawdown_analysis"] = {
                "periods": drawdown_analysis.get("periods", []),
                "summary_stats": drawdown_analysis.get("summary", {}),
                "severity_distribution": self._create_severity_distribution(drawdown_analysis)
            }

        logger.info(f"Created visualization data with {len(viz_data)} chart types")
        return viz_data

    except Exception as e:
        logger.error(f"Error creating visualization data: {e}")
        return {"error": str(e)}

    def _create_histogram_data(self, data: pd.Series, bins: int = 20) -> Dict[str, list]:
        """Create histogram data for visualization."""
        try:
            hist, bin_edges = np.histogram(data, bins=bins)
            bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

            return {
                "bin_centers": bin_centers.tolist(),
                "counts": hist.tolist(),
                "bin_edges": bin_edges.tolist()
            }
        except Exception:
            return {"bin_centers": [], "counts": [], "bin_edges": []}

    def _create_benchmark_data(self, metrics: Dict) -> Dict[str, Any]:
        """Create benchmark comparison data."""
        try:
            # Define benchmark values for comparison
            benchmarks = {
                "total_return": 0.08,  # 8% annual return
                "sharpe_ratio": 1.0,   # Sharpe ratio of 1.0
                "max_drawdown": 0.15,  # 15% max drawdown
                "win_rate": 0.5,       # 50% win rate
                "profit_factor": 1.5,  # 1.5 profit factor
                "calmar_ratio": 1.0    # 1.0 Calmar ratio
            }

            comparison = {}
            for metric, value in metrics.items():
                benchmark_val = benchmarks.get(metric, 0)
                comparison[metric] = {
                    "actual": value,
                    "benchmark": benchmark_val,
                    "outperformance": value - benchmark_val,
                    "outperformance_pct": ((value / benchmark_val) - 1) if benchmark_val != 0 else 0
                }

            return comparison
        except Exception:
            return {}

    def _create_radar_chart_data(self, metrics: Dict) -> Dict[str, list]:
        """Create radar chart data for performance visualization."""
        try:
            # Normalize metrics to 0-100 scale for radar chart
            normalized_metrics = {}

            # Define normalization ranges (min, max for 0-100 scale)
            ranges = {
                "total_return": (-0.2, 0.3),    # -20% to 30%
                "sharpe_ratio": (-1, 3),        # -1 to 3
                "max_drawdown": (0, 0.5),       # 0% to 50% (inverted)
                "win_rate": (0, 1),             # 0% to 100%
                "profit_factor": (0, 3),        # 0 to 3
                "calmar_ratio": (-1, 3)         # -1 to 3
            }

            for metric, value in metrics.items():
                if metric in ranges:
                    min_val, max_val = ranges[metric]
                    # Invert max_drawdown (lower is better)
                    if metric == "max_drawdown":
                        normalized = 100 * (1 - (value - min_val) / (max_val - min_val))
                    else:
                        normalized = 100 * (value - min_val) / (max_val - min_val)
                    normalized_metrics[metric] = max(0, min(100, normalized))

            return {
                "metrics": list(normalized_metrics.keys()),
                "values": list(normalized_metrics.values()),
                "max_value": 100
            }
        except Exception:
            return {"metrics": [], "values": [], "max_value": 100}

    def _identify_recovery_periods(self, drawdown: pd.Series) -> List[Dict]:
        """Identify recovery periods from drawdown data."""
        try:
            recovery_periods = []
            in_recovery = False
            recovery_start = None

            for i, dd in enumerate(drawdown):
                if dd < -0.01 and not in_recovery:  # Start of recovery
                    in_recovery = True
                    recovery_start = i
                elif dd >= -0.001 and in_recovery:  # End of recovery
                    in_recovery = False
                    if recovery_start is not None:
                        recovery_periods.append({
                            "start_idx": recovery_start,
                            "end_idx": i,
                            "duration": i - recovery_start,
                            "start_drawdown": drawdown.iloc[recovery_start],
                            "recovery_strength": abs(drawdown.iloc[recovery_start])
                        })

            return recovery_periods
        except Exception:
            return []

    def _create_risk_gauge_data(self, risk_metrics: Dict) -> Dict[str, Any]:
        """Create risk gauge visualization data."""
        try:
            # Calculate overall risk score (0-100)
            volatility = risk_metrics.get("volatility", 0)
            var_95 = abs(risk_metrics.get("var_95", 0))

            # Normalize to 0-100 scale
            vol_score = min(volatility * 100, 100)  # Assume 100% vol = max risk
            var_score = min(var_95 * 1000, 100)    # Scale VaR appropriately

            overall_risk = (vol_score + var_score) / 2

            # Determine risk level
            if overall_risk < 20:
                risk_level = "Low"
                color = "green"
            elif overall_risk < 50:
                risk_level = "Medium"
                color = "yellow"
            elif overall_risk < 80:
                risk_level = "High"
                color = "orange"
            else:
                risk_level = "Very High"
                color = "red"

            return {
                "risk_score": overall_risk,
                "risk_level": risk_level,
                "color": color,
                "components": {
                    "volatility_score": vol_score,
                    "var_score": var_score
                }
            }
        except Exception:
            return {"risk_score": 50, "risk_level": "Unknown", "color": "gray"}

    def _create_risk_breakdown_data(self, risk_metrics: Dict) -> Dict[str, Any]:
        """Create risk breakdown visualization data."""
        try:
            return {
                "categories": list(risk_metrics.keys()),
                "values": list(risk_metrics.values()),
                "normalized_values": [min(abs(v) * 100, 100) for v in risk_metrics.values()],
                "risk_contributions": self._calculate_risk_contributions(risk_metrics)
            }
        except Exception:
            return {"categories": [], "values": [], "normalized_values": []}

    def _calculate_risk_contributions(self, risk_metrics: Dict) -> Dict[str, float]:
        """Calculate relative risk contributions."""
        try:
            total_risk = sum(abs(v) for v in risk_metrics.values() if isinstance(v, (int, float)))
            if total_risk == 0:
                return {k: 0 for k in risk_metrics.keys()}

            return {k: abs(v) / total_risk for k, v in risk_metrics.items() if isinstance(v, (int, float))}
        except Exception:
            return {}

    def _create_severity_distribution(self, drawdown_analysis: Dict) -> Dict[str, Any]:
        """Create drawdown severity distribution data."""
        try:
            periods = drawdown_analysis.get("periods", [])
            if not periods:
                return {"categories": [], "counts": []}

            # Categorize drawdowns by severity
            severity_counts = {"Minor (<5%)": 0, "Moderate (5-15%)": 0, "Severe (15-25%)": 0, "Extreme (>25%)": 0}

            for period in periods:
                max_dd = period.get("max_drawdown_pct", 0)
                if max_dd < 5:
                    severity_counts["Minor (<5%)"] += 1
                elif max_dd < 15:
                    severity_counts["Moderate (5-15%)"] += 1
                elif max_dd < 25:
                    severity_counts["Severe (15-25%)"] += 1
                else:
                    severity_counts["Extreme (>25%)"] += 1

            return {
                "categories": list(severity_counts.keys()),
                "counts": list(severity_counts.values()),
                "total_periods": len(periods)
            }
        except Exception:
            return {"categories": [], "counts": [], "total_periods": 0}
