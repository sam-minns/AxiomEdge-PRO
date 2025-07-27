# =============================================================================
# BACKTESTER MODULE
# =============================================================================

import logging
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any
from datetime import datetime, timedelta
import os
import json
import warnings

from .config import ConfigModel

logger = logging.getLogger(__name__)

class PerformanceAnalyzer:
    """
    Advanced performance analysis and metrics calculation for backtesting results.

    Provides comprehensive financial metrics, risk-adjusted returns, and detailed
    trade analysis with support for multiple performance measurement methodologies.

    Features:
    - Comprehensive risk-adjusted metrics (Sharpe, Sortino, Calmar)
    - Advanced drawdown analysis with recovery periods
    - Trade-level performance analytics
    - Rolling performance windows
    - Benchmark comparison capabilities
    - Monte Carlo simulation support
    """

    def __init__(self, config: ConfigModel):
        self.config = config
    
    def _calculate_metrics(self, trades_df: Optional[pd.DataFrame], 
                          equity_curve: Optional[pd.Series]) -> Dict[str, Any]:
        """Calculate comprehensive performance metrics."""
        metrics = {}
        
        try:
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
                    # Annualized metrics
                    trading_days = 252
                    total_days = len(returns)
                    years = total_days / trading_days
                    
                    # CAGR
                    if years > 0:
                        metrics['cagr'] = (ending_capital / initial_capital) ** (1/years) - 1
                    else:
                        metrics['cagr'] = 0
                    
                    # Enhanced Risk-Adjusted Metrics
                    if returns.std() > 0:
                        # Sharpe Ratio
                        metrics['sharpe_ratio'] = (returns.mean() * trading_days) / (returns.std() * np.sqrt(trading_days))

                        # Sortino Ratio (downside deviation)
                        downside_returns = returns[returns < 0]
                        if len(downside_returns) > 0:
                            downside_std = downside_returns.std()
                            metrics['sortino_ratio'] = (returns.mean() * trading_days) / (downside_std * np.sqrt(trading_days))
                        else:
                            metrics['sortino_ratio'] = float('inf')
                    else:
                        metrics['sharpe_ratio'] = 0
                        metrics['sortino_ratio'] = 0

                    # Advanced Drawdown Analysis
                    peak = equity_curve.expanding().max()
                    drawdown = (equity_curve - peak) / peak
                    metrics['max_drawdown'] = drawdown.min()
                    metrics['max_drawdown_pct'] = abs(drawdown.min())

                    # Calmar Ratio (CAGR / Max Drawdown)
                    if abs(metrics['max_drawdown']) > 0:
                        metrics['calmar_ratio'] = metrics.get('cagr', 0) / abs(metrics['max_drawdown'])
                    else:
                        metrics['calmar_ratio'] = float('inf')

                    # MAR Ratio (Mean Annual Return / Max Drawdown)
                    mean_annual_return = returns.mean() * trading_days
                    if abs(metrics['max_drawdown']) > 0:
                        metrics['mar_ratio'] = mean_annual_return / abs(metrics['max_drawdown'])
                    else:
                        metrics['mar_ratio'] = float('inf')

                    # Volatility metrics
                    metrics['volatility'] = returns.std() * np.sqrt(trading_days)
                    metrics['downside_volatility'] = downside_returns.std() * np.sqrt(trading_days) if len(downside_returns) > 0 else 0
            
            # Trade-based metrics
            if trades_df is not None and not trades_df.empty:
                metrics['total_trades'] = len(trades_df)
                
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
                    
                    metrics['expected_payoff'] = pnl.mean()
                    metrics['largest_win'] = pnl.max()
                    metrics['largest_loss'] = pnl.min()
        
        except Exception as e:
            logger.error(f"Error calculating metrics: {e}")
        
        return metrics

    def calculate_rolling_metrics(self, equity_curve: pd.Series, window_days: int = 252) -> Dict[str, pd.Series]:
        """
        Calculate rolling performance metrics over specified windows.
        Useful for analyzing performance stability over time.
        """
        rolling_metrics = {}

        try:
            returns = equity_curve.pct_change().dropna()

            if len(returns) < window_days:
                logger.warning(f"Insufficient data for rolling metrics (need {window_days}, have {len(returns)})")
                return rolling_metrics

            # Rolling Sharpe Ratio
            rolling_mean = returns.rolling(window_days).mean() * 252
            rolling_std = returns.rolling(window_days).std() * np.sqrt(252)
            rolling_metrics['sharpe_ratio'] = rolling_mean / rolling_std

            # Rolling Maximum Drawdown
            rolling_peak = equity_curve.rolling(window_days).max()
            rolling_drawdown = (equity_curve - rolling_peak) / rolling_peak
            rolling_metrics['max_drawdown'] = rolling_drawdown.rolling(window_days).min()

            # Rolling Volatility
            rolling_metrics['volatility'] = rolling_std

        except Exception as e:
            logger.error(f"Error calculating rolling metrics: {e}")

        return rolling_metrics

    def analyze_trade_sequences(self, trades_df: pd.DataFrame) -> Dict[str, Any]:
        """
        Analyze sequences of winning and losing trades.
        Identifies streaks and patterns in trading performance.
        """
        sequence_analysis = {}

        try:
            if trades_df.empty or 'pnl' not in trades_df.columns:
                return sequence_analysis

            # Create win/loss sequence
            win_loss = (trades_df['pnl'] > 0).astype(int)

            # Calculate streaks
            streaks = []
            current_streak = 1
            current_type = win_loss.iloc[0]

            for i in range(1, len(win_loss)):
                if win_loss.iloc[i] == current_type:
                    current_streak += 1
                else:
                    streaks.append((current_type, current_streak))
                    current_streak = 1
                    current_type = win_loss.iloc[i]

            # Add final streak
            streaks.append((current_type, current_streak))

            # Analyze streaks
            winning_streaks = [length for streak_type, length in streaks if streak_type == 1]
            losing_streaks = [length for streak_type, length in streaks if streak_type == 0]

            sequence_analysis.update({
                'max_winning_streak': max(winning_streaks) if winning_streaks else 0,
                'max_losing_streak': max(losing_streaks) if losing_streaks else 0,
                'avg_winning_streak': np.mean(winning_streaks) if winning_streaks else 0,
                'avg_losing_streak': np.mean(losing_streaks) if losing_streaks else 0,
                'total_streaks': len(streaks),
                'winning_streak_count': len(winning_streaks),
                'losing_streak_count': len(losing_streaks)
            })

        except Exception as e:
            logger.error(f"Error analyzing trade sequences: {e}")

        return sequence_analysis

    def calculate_advanced_metrics(self, trades_df: pd.DataFrame, equity_curve: pd.Series,
                                  benchmark_returns: Optional[pd.Series] = None) -> Dict[str, Any]:
        """
        Calculate advanced performance metrics including risk-adjusted measures and benchmark comparisons.

        Args:
            trades_df: DataFrame containing trade information
            equity_curve: Series containing equity values over time
            benchmark_returns: Optional benchmark returns for comparison

        Returns:
            Dictionary containing advanced performance metrics
        """
        advanced_metrics = {}

        try:
            returns = equity_curve.pct_change().dropna()

            if len(returns) == 0:
                return advanced_metrics

            # Basic statistics
            advanced_metrics['total_return'] = (equity_curve.iloc[-1] / equity_curve.iloc[0]) - 1
            advanced_metrics['annualized_return'] = (1 + advanced_metrics['total_return']) ** (252 / len(returns)) - 1
            advanced_metrics['annualized_volatility'] = returns.std() * np.sqrt(252)

            # Risk-free rate (assume 2% annually)
            risk_free_rate = 0.02
            excess_returns = returns - (risk_free_rate / 252)

            # Advanced risk metrics
            if returns.std() > 0:
                # Information Ratio
                if benchmark_returns is not None and len(benchmark_returns) == len(returns):
                    active_returns = returns - benchmark_returns
                    if active_returns.std() > 0:
                        advanced_metrics['information_ratio'] = active_returns.mean() / active_returns.std() * np.sqrt(252)
                    else:
                        advanced_metrics['information_ratio'] = 0

                # Treynor Ratio (requires beta calculation)
                if benchmark_returns is not None and len(benchmark_returns) == len(returns):
                    covariance = np.cov(returns, benchmark_returns)[0][1]
                    benchmark_variance = benchmark_returns.var()
                    if benchmark_variance > 0:
                        beta = covariance / benchmark_variance
                        advanced_metrics['beta'] = beta
                        advanced_metrics['treynor_ratio'] = (advanced_metrics['annualized_return'] - risk_free_rate) / beta if beta != 0 else 0

                # Jensen's Alpha
                if 'beta' in advanced_metrics and benchmark_returns is not None:
                    benchmark_return = benchmark_returns.mean() * 252
                    expected_return = risk_free_rate + advanced_metrics['beta'] * (benchmark_return - risk_free_rate)
                    advanced_metrics['jensen_alpha'] = advanced_metrics['annualized_return'] - expected_return

            # Drawdown analysis
            peak = equity_curve.expanding().max()
            drawdown = (equity_curve - peak) / peak

            # Underwater curve (time in drawdown)
            underwater = (drawdown < 0).astype(int)
            underwater_periods = []
            current_period = 0

            for is_underwater in underwater:
                if is_underwater:
                    current_period += 1
                else:
                    if current_period > 0:
                        underwater_periods.append(current_period)
                    current_period = 0

            if current_period > 0:
                underwater_periods.append(current_period)

            advanced_metrics['avg_drawdown_duration'] = np.mean(underwater_periods) if underwater_periods else 0
            advanced_metrics['max_drawdown_duration'] = max(underwater_periods) if underwater_periods else 0
            advanced_metrics['time_underwater_pct'] = underwater.sum() / len(underwater) * 100

            # Recovery analysis
            recovery_times = []
            in_drawdown = False
            drawdown_start = None

            for i, dd in enumerate(drawdown):
                if dd < 0 and not in_drawdown:
                    in_drawdown = True
                    drawdown_start = i
                elif dd >= 0 and in_drawdown:
                    in_drawdown = False
                    if drawdown_start is not None:
                        recovery_times.append(i - drawdown_start)

            advanced_metrics['avg_recovery_time'] = np.mean(recovery_times) if recovery_times else 0
            advanced_metrics['max_recovery_time'] = max(recovery_times) if recovery_times else 0

            # Value at Risk (VaR) and Conditional VaR
            confidence_levels = [0.95, 0.99]
            for confidence in confidence_levels:
                var_level = int(confidence * 100)
                var_value = np.percentile(returns, (1 - confidence) * 100)
                advanced_metrics[f'var_{var_level}'] = var_value

                # Conditional VaR (Expected Shortfall)
                cvar_returns = returns[returns <= var_value]
                advanced_metrics[f'cvar_{var_level}'] = cvar_returns.mean() if len(cvar_returns) > 0 else 0

            # Tail ratio
            returns_sorted = returns.sort_values()
            tail_size = int(len(returns) * 0.1)  # Top and bottom 10%
            if tail_size > 0:
                top_tail = returns_sorted.tail(tail_size).mean()
                bottom_tail = returns_sorted.head(tail_size).mean()
                advanced_metrics['tail_ratio'] = abs(top_tail / bottom_tail) if bottom_tail != 0 else 0

            # Skewness and Kurtosis
            advanced_metrics['skewness'] = returns.skew()
            advanced_metrics['kurtosis'] = returns.kurtosis()

            # Trade-based advanced metrics
            if not trades_df.empty and 'pnl' in trades_df.columns:
                pnl = trades_df['pnl']

                # Profit factor components
                winning_trades = pnl[pnl > 0]
                losing_trades = pnl[pnl < 0]

                if len(winning_trades) > 0 and len(losing_trades) > 0:
                    # Expectancy
                    win_rate = len(winning_trades) / len(pnl)
                    avg_win = winning_trades.mean()
                    avg_loss = abs(losing_trades.mean())
                    advanced_metrics['expectancy'] = (win_rate * avg_win) - ((1 - win_rate) * avg_loss)

                    # Kelly Criterion
                    if avg_loss > 0:
                        kelly_fraction = (win_rate * avg_win - (1 - win_rate) * avg_loss) / avg_win
                        advanced_metrics['kelly_fraction'] = max(0, min(1, kelly_fraction))  # Clamp between 0 and 1

                # Trade efficiency metrics
                if 'entry_time' in trades_df.columns and 'exit_time' in trades_df.columns:
                    trade_durations = pd.to_datetime(trades_df['exit_time']) - pd.to_datetime(trades_df['entry_time'])
                    advanced_metrics['avg_trade_duration_hours'] = trade_durations.dt.total_seconds().mean() / 3600
                    advanced_metrics['max_trade_duration_hours'] = trade_durations.dt.total_seconds().max() / 3600

            return advanced_metrics

        except Exception as e:
            logger.error(f"Error calculating advanced metrics: {e}", exc_info=True)
            return advanced_metrics

    def generate_performance_report(self, trades_df: pd.DataFrame, equity_curve: pd.Series,
                                  benchmark_returns: Optional[pd.Series] = None) -> Dict[str, Any]:
        """
        Generate comprehensive performance report with all metrics and analysis.

        Args:
            trades_df: DataFrame containing trade information
            equity_curve: Series containing equity values over time
            benchmark_returns: Optional benchmark returns for comparison

        Returns:
            Comprehensive performance report
        """
        try:
            report = {
                'report_metadata': {
                    'generation_time': pd.Timestamp.now().isoformat(),
                    'total_trades': len(trades_df) if not trades_df.empty else 0,
                    'analysis_period_days': len(equity_curve),
                    'has_benchmark': benchmark_returns is not None
                },
                'basic_metrics': {},
                'advanced_metrics': {},
                'rolling_metrics': {},
                'trade_analysis': {},
                'risk_analysis': {},
                'benchmark_comparison': {}
            }

            # Basic metrics
            report['basic_metrics'] = self._calculate_metrics(trades_df, equity_curve)

            # Advanced metrics
            report['advanced_metrics'] = self.calculate_advanced_metrics(trades_df, equity_curve, benchmark_returns)

            # Rolling metrics (if enough data)
            if len(equity_curve) >= 252:
                report['rolling_metrics'] = self.calculate_rolling_metrics(equity_curve, 252)

            # Trade sequence analysis
            if not trades_df.empty:
                report['trade_analysis'] = self.analyze_trade_sequences(trades_df)

            # Risk analysis summary
            report['risk_analysis'] = self._generate_risk_summary(report['basic_metrics'], report['advanced_metrics'])

            # Benchmark comparison
            if benchmark_returns is not None:
                report['benchmark_comparison'] = self._compare_to_benchmark(equity_curve, benchmark_returns)

            return report

        except Exception as e:
            logger.error(f"Error generating performance report: {e}", exc_info=True)
            return {'error': str(e)}

    def _generate_risk_summary(self, basic_metrics: Dict[str, Any], advanced_metrics: Dict[str, Any]) -> Dict[str, Any]:
        """Generate risk analysis summary."""
        risk_summary = {
            'risk_level': 'Unknown',
            'risk_factors': [],
            'risk_score': 0.0,  # 0-10 scale
            'recommendations': []
        }

        try:
            risk_score = 0

            # Sharpe ratio assessment
            sharpe = basic_metrics.get('sharpe_ratio', 0)
            if sharpe < 0.5:
                risk_score += 3
                risk_summary['risk_factors'].append('Low risk-adjusted returns')
            elif sharpe < 1.0:
                risk_score += 1

            # Drawdown assessment
            max_dd = abs(basic_metrics.get('max_drawdown_pct', 0))
            if max_dd > 0.2:
                risk_score += 3
                risk_summary['risk_factors'].append('High maximum drawdown')
            elif max_dd > 0.1:
                risk_score += 1

            # Volatility assessment
            volatility = advanced_metrics.get('annualized_volatility', 0)
            if volatility > 0.3:
                risk_score += 2
                risk_summary['risk_factors'].append('High volatility')

            # VaR assessment
            var_95 = advanced_metrics.get('var_95', 0)
            if var_95 < -0.05:  # Daily VaR worse than -5%
                risk_score += 2
                risk_summary['risk_factors'].append('High Value at Risk')

            # Time underwater assessment
            time_underwater = advanced_metrics.get('time_underwater_pct', 0)
            if time_underwater > 50:
                risk_score += 2
                risk_summary['risk_factors'].append('Extended drawdown periods')

            # Determine risk level
            if risk_score <= 2:
                risk_summary['risk_level'] = 'Low'
            elif risk_score <= 5:
                risk_summary['risk_level'] = 'Moderate'
            elif risk_score <= 8:
                risk_summary['risk_level'] = 'High'
            else:
                risk_summary['risk_level'] = 'Very High'

            risk_summary['risk_score'] = min(10, risk_score)

            # Generate recommendations
            if sharpe < 1.0:
                risk_summary['recommendations'].append('Consider improving risk-adjusted returns through better position sizing')

            if max_dd > 0.15:
                risk_summary['recommendations'].append('Implement stricter risk management controls')

            if time_underwater > 40:
                risk_summary['recommendations'].append('Review strategy for faster recovery from drawdowns')

            return risk_summary

        except Exception as e:
            logger.error(f"Error generating risk summary: {e}")
            return risk_summary

    def _compare_to_benchmark(self, equity_curve: pd.Series, benchmark_returns: pd.Series) -> Dict[str, Any]:
        """Compare strategy performance to benchmark."""
        comparison = {}

        try:
            strategy_returns = equity_curve.pct_change().dropna()

            # Align series
            min_length = min(len(strategy_returns), len(benchmark_returns))
            strategy_returns = strategy_returns.tail(min_length)
            benchmark_returns = benchmark_returns.tail(min_length)

            # Calculate benchmark metrics
            benchmark_total_return = (1 + benchmark_returns).prod() - 1
            benchmark_volatility = benchmark_returns.std() * np.sqrt(252)
            benchmark_sharpe = (benchmark_returns.mean() * 252) / benchmark_volatility if benchmark_volatility > 0 else 0

            # Calculate strategy metrics
            strategy_total_return = (equity_curve.iloc[-1] / equity_curve.iloc[0]) - 1
            strategy_volatility = strategy_returns.std() * np.sqrt(252)
            strategy_sharpe = (strategy_returns.mean() * 252) / strategy_volatility if strategy_volatility > 0 else 0

            comparison = {
                'benchmark_total_return': benchmark_total_return,
                'strategy_total_return': strategy_total_return,
                'excess_return': strategy_total_return - benchmark_total_return,
                'benchmark_volatility': benchmark_volatility,
                'strategy_volatility': strategy_volatility,
                'benchmark_sharpe': benchmark_sharpe,
                'strategy_sharpe': strategy_sharpe,
                'sharpe_improvement': strategy_sharpe - benchmark_sharpe,
                'correlation': strategy_returns.corr(benchmark_returns),
                'tracking_error': (strategy_returns - benchmark_returns).std() * np.sqrt(252)
            }

            return comparison

        except Exception as e:
            logger.error(f"Error comparing to benchmark: {e}")
            return comparison

    def plot_equity_curve(self, equity_curve: pd.Series, benchmark_returns: Optional[pd.Series] = None,
                          save_path: Optional[str] = None) -> Optional[str]:
        """
        Plot equity curve with optional benchmark comparison.

        Args:
            equity_curve: Series containing equity values over time
            benchmark_returns: Optional benchmark returns for comparison
            save_path: Optional path to save the plot

        Returns:
            Path to saved plot or None if plotting failed
        """
        try:
            import matplotlib.pyplot as plt
            import matplotlib.dates as mdates

            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))

            # Plot equity curve
            ax1.plot(equity_curve.index, equity_curve.values, label='Strategy', linewidth=2, color='blue')

            if benchmark_returns is not None:
                # Convert benchmark returns to cumulative equity
                benchmark_equity = (1 + benchmark_returns).cumprod() * equity_curve.iloc[0]
                ax1.plot(benchmark_equity.index, benchmark_equity.values,
                        label='Benchmark', linewidth=2, color='red', alpha=0.7)

            ax1.set_title('Equity Curve Comparison', fontsize=14, fontweight='bold')
            ax1.set_ylabel('Portfolio Value', fontsize=12)
            ax1.legend()
            ax1.grid(True, alpha=0.3)

            # Plot drawdown
            peak = equity_curve.expanding().max()
            drawdown = (equity_curve - peak) / peak * 100

            ax2.fill_between(drawdown.index, drawdown.values, 0, alpha=0.3, color='red')
            ax2.plot(drawdown.index, drawdown.values, color='red', linewidth=1)
            ax2.set_title('Drawdown (%)', fontsize=14, fontweight='bold')
            ax2.set_ylabel('Drawdown %', fontsize=12)
            ax2.set_xlabel('Date', fontsize=12)
            ax2.grid(True, alpha=0.3)

            # Format x-axis
            for ax in [ax1, ax2]:
                ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
                ax.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
                plt.setp(ax.xaxis.get_majorticklabels(), rotation=45)

            plt.tight_layout()

            if save_path:
                plt.savefig(save_path, dpi=300, bbox_inches='tight')
                plt.close()
                return save_path
            else:
                plt.show()
                return None

        except ImportError:
            logger.warning("Matplotlib not available for plotting")
            return None
        except Exception as e:
            logger.error(f"Error plotting equity curve: {e}")
            return None

    def plot_rolling_metrics(self, equity_curve: pd.Series, window_days: int = 252,
                           save_path: Optional[str] = None) -> Optional[str]:
        """
        Plot rolling performance metrics.

        Args:
            equity_curve: Series containing equity values over time
            window_days: Rolling window size in days
            save_path: Optional path to save the plot

        Returns:
            Path to saved plot or None if plotting failed
        """
        try:
            import matplotlib.pyplot as plt

            rolling_metrics = self.calculate_rolling_metrics(equity_curve, window_days)

            if not rolling_metrics:
                logger.warning("No rolling metrics available for plotting")
                return None

            fig, axes = plt.subplots(3, 1, figsize=(12, 12))

            # Rolling Sharpe Ratio
            if 'sharpe_ratio' in rolling_metrics:
                axes[0].plot(rolling_metrics['sharpe_ratio'].index,
                           rolling_metrics['sharpe_ratio'].values,
                           color='blue', linewidth=2)
                axes[0].axhline(y=1.0, color='red', linestyle='--', alpha=0.7, label='Sharpe = 1.0')
                axes[0].set_title(f'Rolling Sharpe Ratio ({window_days} days)', fontweight='bold')
                axes[0].set_ylabel('Sharpe Ratio')
                axes[0].legend()
                axes[0].grid(True, alpha=0.3)

            # Rolling Maximum Drawdown
            if 'max_drawdown' in rolling_metrics:
                axes[1].fill_between(rolling_metrics['max_drawdown'].index,
                                   rolling_metrics['max_drawdown'].values * 100, 0,
                                   alpha=0.3, color='red')
                axes[1].plot(rolling_metrics['max_drawdown'].index,
                           rolling_metrics['max_drawdown'].values * 100,
                           color='red', linewidth=2)
                axes[1].set_title(f'Rolling Maximum Drawdown ({window_days} days)', fontweight='bold')
                axes[1].set_ylabel('Max Drawdown %')
                axes[1].grid(True, alpha=0.3)

            # Rolling Volatility
            if 'volatility' in rolling_metrics:
                axes[2].plot(rolling_metrics['volatility'].index,
                           rolling_metrics['volatility'].values * 100,
                           color='green', linewidth=2)
                axes[2].set_title(f'Rolling Volatility ({window_days} days)', fontweight='bold')
                axes[2].set_ylabel('Volatility %')
                axes[2].set_xlabel('Date')
                axes[2].grid(True, alpha=0.3)

            plt.tight_layout()

            if save_path:
                plt.savefig(save_path, dpi=300, bbox_inches='tight')
                plt.close()
                return save_path
            else:
                plt.show()
                return None

        except ImportError:
            logger.warning("Matplotlib not available for plotting")
            return None
        except Exception as e:
            logger.error(f"Error plotting rolling metrics: {e}")
            return None

    def create_performance_dashboard(self, trades_df: pd.DataFrame, equity_curve: pd.Series,
                                   benchmark_returns: Optional[pd.Series] = None,
                                   save_path: Optional[str] = None) -> Optional[str]:
        """
        Create comprehensive performance dashboard with multiple visualizations.

        Args:
            trades_df: DataFrame containing trade information
            equity_curve: Series containing equity values over time
            benchmark_returns: Optional benchmark returns for comparison
            save_path: Optional path to save the dashboard

        Returns:
            Path to saved dashboard or None if creation failed
        """
        try:
            import matplotlib.pyplot as plt
            import matplotlib.gridspec as gridspec

            # Create figure with custom layout
            fig = plt.figure(figsize=(16, 12))
            gs = gridspec.GridSpec(3, 3, figure=fig)

            # 1. Equity curve (top row, spans 2 columns)
            ax1 = fig.add_subplot(gs[0, :2])
            ax1.plot(equity_curve.index, equity_curve.values, label='Strategy', linewidth=2, color='blue')

            if benchmark_returns is not None:
                benchmark_equity = (1 + benchmark_returns).cumprod() * equity_curve.iloc[0]
                ax1.plot(benchmark_equity.index, benchmark_equity.values,
                        label='Benchmark', linewidth=2, color='red', alpha=0.7)

            ax1.set_title('Equity Curve', fontweight='bold')
            ax1.legend()
            ax1.grid(True, alpha=0.3)

            # 2. Drawdown (top right)
            ax2 = fig.add_subplot(gs[0, 2])
            peak = equity_curve.expanding().max()
            drawdown = (equity_curve - peak) / peak * 100
            ax2.fill_between(drawdown.index, drawdown.values, 0, alpha=0.3, color='red')
            ax2.plot(drawdown.index, drawdown.values, color='red', linewidth=1)
            ax2.set_title('Drawdown %', fontweight='bold')
            ax2.grid(True, alpha=0.3)

            # 3. Monthly returns heatmap (middle left)
            ax3 = fig.add_subplot(gs[1, 0])
            returns = equity_curve.pct_change().dropna()
            if len(returns) > 30:  # Only if we have enough data
                monthly_returns = returns.resample('M').apply(lambda x: (1 + x).prod() - 1)
                monthly_returns_pct = monthly_returns * 100

                # Create simple heatmap-like visualization
                colors = ['red' if x < 0 else 'green' for x in monthly_returns_pct]
                ax3.bar(range(len(monthly_returns_pct)), monthly_returns_pct, color=colors, alpha=0.7)
                ax3.set_title('Monthly Returns %', fontweight='bold')
                ax3.axhline(y=0, color='black', linestyle='-', alpha=0.5)
                ax3.grid(True, alpha=0.3)

            # 4. Trade distribution (middle center)
            ax4 = fig.add_subplot(gs[1, 1])
            if not trades_df.empty and 'pnl' in trades_df.columns:
                pnl = trades_df['pnl']
                ax4.hist(pnl, bins=30, alpha=0.7, color='blue', edgecolor='black')
                ax4.axvline(x=0, color='red', linestyle='--', alpha=0.7)
                ax4.set_title('Trade P&L Distribution', fontweight='bold')
                ax4.set_xlabel('P&L')
                ax4.set_ylabel('Frequency')
                ax4.grid(True, alpha=0.3)

            # 5. Rolling Sharpe (middle right)
            ax5 = fig.add_subplot(gs[1, 2])
            if len(equity_curve) >= 252:
                rolling_metrics = self.calculate_rolling_metrics(equity_curve, 252)
                if 'sharpe_ratio' in rolling_metrics:
                    ax5.plot(rolling_metrics['sharpe_ratio'].index,
                           rolling_metrics['sharpe_ratio'].values,
                           color='blue', linewidth=2)
                    ax5.axhline(y=1.0, color='red', linestyle='--', alpha=0.7)
                    ax5.set_title('Rolling Sharpe (252d)', fontweight='bold')
                    ax5.grid(True, alpha=0.3)

            # 6. Performance metrics table (bottom row)
            ax6 = fig.add_subplot(gs[2, :])
            ax6.axis('off')

            # Calculate key metrics for display
            basic_metrics = self._calculate_metrics(trades_df, equity_curve)
            advanced_metrics = self.calculate_advanced_metrics(trades_df, equity_curve, benchmark_returns)

            # Create metrics table
            metrics_text = f"""
            Key Performance Metrics:

            Total Return: {basic_metrics.get('net_profit_pct', 0):.2%}
            CAGR: {basic_metrics.get('cagr', 0):.2%}
            Sharpe Ratio: {basic_metrics.get('sharpe_ratio', 0):.2f}
            Sortino Ratio: {basic_metrics.get('sortino_ratio', 0):.2f}
            Max Drawdown: {basic_metrics.get('max_drawdown_pct', 0):.2%}

            Win Rate: {basic_metrics.get('win_rate', 0):.2%}
            Profit Factor: {basic_metrics.get('profit_factor', 0):.2f}
            Total Trades: {basic_metrics.get('total_trades', 0)}
            Volatility: {advanced_metrics.get('annualized_volatility', 0):.2%}
            VaR (95%): {advanced_metrics.get('var_95', 0):.2%}
            """

            ax6.text(0.1, 0.5, metrics_text, fontsize=10, verticalalignment='center',
                    bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgray", alpha=0.5))

            plt.suptitle('Performance Dashboard', fontsize=16, fontweight='bold')
            plt.tight_layout()

            if save_path:
                plt.savefig(save_path, dpi=300, bbox_inches='tight')
                plt.close()
                return save_path
            else:
                plt.show()
                return None

        except ImportError:
            logger.warning("Matplotlib not available for dashboard creation")
            return None
        except Exception as e:
            logger.error(f"Error creating performance dashboard: {e}")
            return None

    def export_results(self, trades_df: pd.DataFrame, equity_curve: pd.Series,
                      output_dir: str = "performance_results") -> Dict[str, str]:
        """
        Export comprehensive performance analysis results to files.

        Args:
            trades_df: DataFrame containing trade information
            equity_curve: Series containing equity values over time
            output_dir: Directory to save results

        Returns:
            Dictionary mapping result types to file paths
        """
        import os
        from pathlib import Path

        try:
            # Create output directory
            Path(output_dir).mkdir(parents=True, exist_ok=True)

            exported_files = {}

            # 1. Export comprehensive performance report
            report = self.generate_performance_report(trades_df, equity_curve)
            report_path = os.path.join(output_dir, "performance_report.json")

            import json
            with open(report_path, 'w') as f:
                json.dump(report, f, indent=2, default=str)
            exported_files['performance_report'] = report_path

            # 2. Export trades data
            if not trades_df.empty:
                trades_path = os.path.join(output_dir, "trades_data.csv")
                trades_df.to_csv(trades_path, index=False)
                exported_files['trades_data'] = trades_path

            # 3. Export equity curve
            equity_path = os.path.join(output_dir, "equity_curve.csv")
            equity_curve.to_csv(equity_path, header=['equity'])
            exported_files['equity_curve'] = equity_path

            # 4. Export visualizations
            dashboard_path = os.path.join(output_dir, "performance_dashboard.png")
            if self.create_performance_dashboard(trades_df, equity_curve, save_path=dashboard_path):
                exported_files['dashboard'] = dashboard_path

            equity_plot_path = os.path.join(output_dir, "equity_curve_plot.png")
            if self.plot_equity_curve(equity_curve, save_path=equity_plot_path):
                exported_files['equity_plot'] = equity_plot_path

            rolling_plot_path = os.path.join(output_dir, "rolling_metrics_plot.png")
            if len(equity_curve) >= 252:
                if self.plot_rolling_metrics(equity_curve, save_path=rolling_plot_path):
                    exported_files['rolling_plot'] = rolling_plot_path

            logger.info(f"Performance results exported to {output_dir}")
            return exported_files

        except Exception as e:
            logger.error(f"Error exporting results: {e}")
            return {}

    def generate_full_report(self, trades_df: Optional[pd.DataFrame], equity_curve: Optional[pd.Series],
                           cycle_metrics: List[Dict], aggregated_shap: Optional[pd.DataFrame] = None,
                           framework_memory: Optional[Dict] = None, aggregated_daily_dd: Optional[List[Dict]] = None,
                           last_classification_report: str = "N/A") -> Dict[str, Any]:
        """
        Generate comprehensive performance report matching the main file interface.

        This method follows the exact pattern from the main file:
        1. Plot equity curve if available
        2. Plot SHAP summary if available
        3. Calculate metrics
        4. Generate text report
        5. Return metrics
        """
        logger.info("-> Stage 4: Generating Final Performance Report...")

        # Plot equity curve if available
        if equity_curve is not None and len(equity_curve) > 1:
            self.plot_equity_curve(equity_curve)

        # Plot SHAP summary if available
        if aggregated_shap is not None and not aggregated_shap.empty:
            self.plot_shap_summary(aggregated_shap)

        # Calculate metrics
        metrics = self._calculate_metrics(trades_df, equity_curve) if trades_df is not None and not trades_df.empty else {}

        # Generate text report (delegate to report_generator module)
        try:
            from .report_generator import ReportGenerator
            report_gen = ReportGenerator(self.config)
            report_gen.generate_text_report(metrics, cycle_metrics, aggregated_shap, framework_memory, aggregated_daily_dd, last_classification_report)
        except Exception as e:
            logger.warning(f"Could not generate text report: {e}")

        logger.info(f"[SUCCESS] Final report generated and saved to: {getattr(self.config, 'REPORT_SAVE_PATH', 'N/A')}")
        return metrics

    def plot_shap_summary(self, shap_summary: pd.DataFrame):
        """
        Generate SHAP feature importance plot matching the main file implementation.
        """
        try:
            import matplotlib.pyplot as plt

            plt.style.use('seaborn-v0_8-darkgrid')
            plt.figure(figsize=(12, 10))
            shap_summary.head(20).sort_values(by='SHAP_Importance').plot(kind='barh', legend=False, color='mediumseagreen')

            title_str = f"{getattr(self.config, 'nickname', '') or getattr(self.config, 'REPORT_LABEL', '')} ({getattr(self.config, 'strategy_name', '')}) - Aggregated Feature Importance"
            plt.title(title_str, fontsize=16, weight='bold')
            plt.xlabel("Mean Absolute SHAP Value", fontsize=12)
            plt.ylabel("Feature", fontsize=12)
            plt.tight_layout()

            # Construct a full path for the final aggregated SHAP summary
            shap_plot_path = getattr(self.config, 'SHAP_PLOT_PATH', 'shap_plots')
            run_timestamp = getattr(self.config, 'run_timestamp', 'unknown')

            import os
            os.makedirs(shap_plot_path, exist_ok=True)
            final_shap_path = os.path.join(shap_plot_path, f"{run_timestamp}_aggregated_shap_summary.png")

            plt.savefig(final_shap_path)
            plt.close()
            logger.info(f"  - SHAP summary plot saved to: {final_shap_path}")

        except Exception as e:
            logger.error(f"  - Failed to save SHAP plot: {e}")


class Backtester:
    """
    Advanced backtesting engine with realistic execution simulation.

    Provides comprehensive backtesting capabilities including dynamic ensemble voting,
    adaptive confidence thresholds, realistic execution costs, and sophisticated
    risk management. Supports multiple position sizing strategies and take-profit ladders.

    Features:
    - Dynamic ensemble voting with adaptive model weighting
    - Confidence-based position sizing and filtering
    - Take-profit ladder execution for partial profit taking
    - Realistic execution simulation with slippage and spreads
    - Tiered risk management based on account size
    - Variable commission and latency simulation
    - Comprehensive trade tracking and performance analysis
    """

    def __init__(self, config: ConfigModel):
        self.config = config
        self.use_tp_ladder = getattr(config, 'USE_TP_LADDER', False)
        self.use_realistic_execution = getattr(config, 'USE_REALISTIC_EXECUTION', True)
        self.simulate_latency = getattr(config, 'SIMULATE_LATENCY', True)
        self.execution_latency_ms = getattr(config, 'EXECUTION_LATENCY_MS', 150)
        self.use_variable_slippage = getattr(config, 'USE_VARIABLE_SLIPPAGE', True)
        self.slippage_volatility_factor = getattr(config, 'SLIPPAGE_VOLATILITY_FACTOR', 1.5)
        self.commission_per_lot = getattr(config, 'COMMISSION_PER_LOT', 3.5)

        # Initialize spread configuration
        self.spread_config = getattr(config, 'SPREAD_CONFIG', {
            'default': {'normal_pips': 1.8, 'volatile_pips': 5.5}
        })

        # Initialize asset contract sizes
        self.asset_contract_sizes = getattr(config, 'ASSET_CONTRACT_SIZES', {
            'default': 100000.0
        })
        
        # Validate TP ladder configuration
        if self.use_tp_ladder:
            tp_levels = getattr(config, 'TP_LADDER_LEVELS_PCT', [])
            tp_multipliers = getattr(config, 'TP_LADDER_RISK_MULTIPLIERS', [])
            
            if not tp_levels or not tp_multipliers or len(tp_levels) != len(tp_multipliers):
                logger.error("TP Ladder config error: Invalid or mismatched lengths. Disabling.")
                self.use_tp_ladder = False
            elif not np.isclose(sum(tp_levels), 1.0):
                logger.error(f"TP Ladder config error: Levels sum ({sum(tp_levels)}) != 1.0. Disabling.")
                self.use_tp_ladder = False
            else:
                logger.info("Take-Profit Ladder is ENABLED.")
        
        logger.info("Backtester initialized")

    def _get_tiered_risk_params(self, equity: float) -> Tuple[float, int]:
        """Looks up risk percentage and max trades from the tiered config."""
        tiered_config = self.config.TIERED_RISK_CONFIG

        for tier_name, tier_params in tiered_config.items():
            min_equity = tier_params.get('min_equity', 0)
            max_equity = tier_params.get('max_equity', float('inf'))

            if min_equity <= equity < max_equity:
                risk_pct = tier_params.get('risk_pct', self.config.BASE_RISK_PER_TRADE_PCT)
                max_trades = tier_params.get('max_trades', self.config.MAX_CONCURRENT_TRADES)
                return risk_pct, max_trades

        # Default fallback
        return self.config.BASE_RISK_PER_TRADE_PCT, self.config.MAX_CONCURRENT_TRADES

    def _calculate_realistic_costs(self, candle: Dict, on_exit: bool = False) -> Tuple[float, float]:
        """Calculates dynamic spread and variable slippage (returns cost in price units)."""
        symbol = candle.get('Symbol', 'UNKNOWN')
        close_price = candle.get('Close', 1.0)

        # Get base spread from config
        spread_config = self.config.SPREAD_CONFIG.get(symbol, self.config.SPREAD_CONFIG.get('default', {}))
        base_spread = spread_config.get('normal_pips', 2.0)

        # Apply volatility adjustment
        atr = candle.get('ATR', close_price * 0.001)
        volatility_factor = min(atr / (close_price * 0.001), 3.0)  # Cap at 3x normal

        dynamic_spread = base_spread * (1 + volatility_factor * 0.5)

        # Calculate slippage (random component)
        import random
        slippage_factor = random.uniform(0.5, 1.5) if not on_exit else random.uniform(0.3, 1.0)
        slippage = dynamic_spread * slippage_factor * 0.3

        return dynamic_spread, slippage

    def _calculate_latency_cost(self, signal_candle: Dict, exec_candle: Dict) -> float:
        """Calculates a randomized, volatility-based cost (in price units) to simulate execution latency."""
        signal_price = signal_candle.get('Close', 1.0)
        exec_price = exec_candle.get('Close', 1.0)

        # Base latency cost as percentage of price movement
        price_movement = abs(exec_price - signal_price)

        # Add random latency factor
        import random
        latency_factor = random.uniform(0.1, 0.3)  # 10-30% of price movement

        return price_movement * latency_factor

    def _get_regime_adaptive_confidence_gate(self, regime: str) -> float:
        """
        Get confidence threshold based on current market regime.

        Args:
            regime: Current market regime identifier

        Returns:
            Confidence threshold for trade filtering
        """
        regime_gates = getattr(self.config, 'REGIME_CONFIDENCE_GATES', {
            'Trending': 0.65,
            'Ranging': 0.75,
            'Highvolatility': 0.70,
            'Default': 0.70
        })

        return regime_gates.get(regime, regime_gates.get('Default', 0.70))

    def _calculate_realistic_spread(self, symbol: str, volatility_rank: float = 0.5) -> float:
        """
        Calculate realistic spread based on symbol and market conditions.

        Args:
            symbol: Trading symbol
            volatility_rank: Current volatility rank (0-1)

        Returns:
            Spread in pips
        """
        # Get symbol-specific spread config
        symbol_config = self.spread_config.get(symbol, self.spread_config.get('default', {
            'normal_pips': 1.8, 'volatile_pips': 5.5
        }))

        normal_spread = symbol_config.get('normal_pips', 1.8)
        volatile_spread = symbol_config.get('volatile_pips', 5.5)

        # Interpolate based on volatility
        spread = normal_spread + (volatile_spread - normal_spread) * volatility_rank

        return spread

    def _calculate_position_size(self, equity: float, confidence: float, symbol: str) -> float:
        """
        Calculate position size based on equity, confidence, and risk parameters.

        Args:
            equity: Current account equity
            confidence: Model confidence for this trade
            symbol: Trading symbol

        Returns:
            Position size in lots
        """
        # Get base risk percentage
        base_risk_pct = getattr(self.config, 'BASE_RISK_PER_TRADE_PCT', 0.0025)

        # Apply confidence multiplier
        confidence_tiers = getattr(self.config, 'CONFIDENCE_TIERS', {
            'ultra_high': {'min': 0.80, 'risk_mult': 1.2},
            'high': {'min': 0.70, 'risk_mult': 1.0},
            'standard': {'min': 0.60, 'risk_mult': 0.8}
        })

        risk_multiplier = 0.8  # Default
        for tier, config in confidence_tiers.items():
            if confidence >= config['min']:
                risk_multiplier = config['risk_mult']
                break

        # Calculate risk amount
        risk_amount = equity * base_risk_pct * risk_multiplier

        # Apply risk cap
        risk_cap = getattr(self.config, 'RISK_CAP_PER_TRADE_USD', 1000.0)
        risk_amount = min(risk_amount, risk_cap)

        # Convert to position size (simplified)
        contract_size = self.asset_contract_sizes.get(symbol, self.asset_contract_sizes.get('default', 100000.0))
        min_lot_size = getattr(self.config, 'MIN_LOT_SIZE', 0.01)

        # Assume 1% price movement for position sizing
        position_size = risk_amount / (contract_size * 0.01)
        position_size = max(min_lot_size, round(position_size / min_lot_size) * min_lot_size)

        return position_size

    def _apply_circuit_breakers(self, current_equity: float, initial_equity: float,
                               trades_today: int, consecutive_losses: int) -> bool:
        """
        Apply circuit breaker mechanisms to halt trading under adverse conditions.

        Args:
            current_equity: Current account equity
            initial_equity: Starting equity
            trades_today: Number of trades executed today
            consecutive_losses: Number of consecutive losing trades

        Returns:
            True if trading should be halted, False otherwise
        """
        # Daily loss limit
        daily_loss_limit = getattr(self.config, 'DAILY_LOSS_LIMIT_PCT', 0.05)
        if (initial_equity - current_equity) / initial_equity > daily_loss_limit:
            logger.warning(f"Circuit breaker triggered: Daily loss limit ({daily_loss_limit*100:.1f}%) exceeded")
            return True

        # Maximum trades per day
        max_trades_per_day = getattr(self.config, 'MAX_TRADES_PER_DAY', 10)
        if trades_today >= max_trades_per_day:
            logger.warning(f"Circuit breaker triggered: Maximum trades per day ({max_trades_per_day}) reached")
            return True

        # Consecutive loss limit
        max_consecutive_losses = getattr(self.config, 'MAX_CONSECUTIVE_LOSSES', 5)
        if consecutive_losses >= max_consecutive_losses:
            logger.warning(f"Circuit breaker triggered: Maximum consecutive losses ({max_consecutive_losses}) reached")
            return True

        return False

    def _calculate_dynamic_ensemble_weights(self, models_dict: Dict, recent_performance: Dict) -> Dict[str, float]:
        """
        Calculate dynamic weights for ensemble voting based on recent performance.

        Args:
            models_dict: Dictionary of trained models
            recent_performance: Recent performance metrics for each model

        Returns:
            Dictionary of normalized weights for each model
        """
        if not models_dict or not recent_performance:
            # Equal weights if no performance data
            equal_weight = 1.0 / len(models_dict) if models_dict else 0.0
            return {model_name: equal_weight for model_name in models_dict.keys()}

        weights = {}

        for model_name in models_dict.keys():
            perf = recent_performance.get(model_name, {})

            # Base weight from F1 score
            f1_score = perf.get('f1_score', 0.5)
            base_weight = max(0.1, f1_score)  # Minimum weight of 0.1

            # Adjust for Sharpe ratio if available
            sharpe_ratio = perf.get('sharpe_ratio', 0.0)
            sharpe_adjustment = 1.0 + (sharpe_ratio * 0.2)  # 20% adjustment based on Sharpe

            # Adjust for win rate if available
            win_rate = perf.get('win_rate', 0.5)
            win_rate_adjustment = 0.8 + (win_rate * 0.4)  # Scale from 0.8 to 1.2

            # Calculate final weight
            final_weight = base_weight * sharpe_adjustment * win_rate_adjustment
            weights[model_name] = max(0.05, final_weight)  # Minimum weight of 0.05

        # Normalize weights to sum to 1.0
        total_weight = sum(weights.values())
        if total_weight > 0:
            weights = {model: weight / total_weight for model, weight in weights.items()}

        return weights

    def _generate_ensemble_signal(self, models_dict: Dict, feature_data: pd.Series,
                                 ensemble_weights: Dict[str, float], confidence_threshold: float) -> Tuple[int, float]:
        """
        Generate ensemble signal using weighted voting from multiple models.

        Args:
            models_dict: Dictionary of trained models
            feature_data: Feature data for prediction
            ensemble_weights: Weights for each model
            confidence_threshold: Minimum confidence for signal generation

        Returns:
            Tuple of (signal, confidence) where signal is 0=hold, 1=long, 2=short
        """
        if not models_dict:
            return 0, 0.0

        try:
            # Collect predictions from all models
            predictions = {}
            confidences = {}

            for model_name, pipeline in models_dict.items():
                try:
                    # Prepare feature data
                    feature_array = feature_data.values.reshape(1, -1)

                    # Get prediction probabilities
                    probabilities = pipeline.predict_proba(feature_array)[0]

                    # Get predicted class and confidence
                    predicted_class = np.argmax(probabilities)
                    confidence = np.max(probabilities)

                    predictions[model_name] = predicted_class
                    confidences[model_name] = confidence

                except Exception as e:
                    logger.debug(f"Model {model_name} prediction failed: {e}")
                    predictions[model_name] = 0  # Default to hold
                    confidences[model_name] = 0.0

            # Calculate weighted ensemble prediction
            weighted_votes = {0: 0.0, 1: 0.0, 2: 0.0}  # hold, long, short
            total_confidence = 0.0

            for model_name, prediction in predictions.items():
                weight = ensemble_weights.get(model_name, 0.0)
                confidence = confidences.get(model_name, 0.0)

                # Weight the vote by both model weight and prediction confidence
                vote_strength = weight * confidence
                weighted_votes[prediction] += vote_strength
                total_confidence += weight * confidence

            # Determine final signal
            if total_confidence == 0:
                return 0, 0.0

            # Get the class with highest weighted vote
            final_signal = max(weighted_votes, key=weighted_votes.get)
            final_confidence = weighted_votes[final_signal] / sum(ensemble_weights.values())

            # Apply confidence threshold
            if final_confidence < confidence_threshold:
                return 0, final_confidence  # Hold if confidence too low

            return final_signal, final_confidence

        except Exception as e:
            logger.error(f"Ensemble signal generation failed: {e}")
            return 0, 0.0

    def run_backtest_chunk(self, df_chunk_in: pd.DataFrame, training_results: Dict,
                          initial_equity: float, cycle_history: List[Dict], diagnosed_regime: str,
                          trade_lockout_until: Optional[pd.Timestamp] = None, cycle_directives: Dict = {}) -> Tuple[pd.DataFrame, pd.Series, bool, Optional[Dict], Dict, Dict]:
        """
        Runs a backtest on a chunk of data using a dynamic, weighted voting ensemble of models.

        This method simulates trade execution based on model predictions, applying
        realistic costs, risk management rules, and adaptive parameters based
        on the current market regime and historical performance. It now includes
        a final cleanup step to force-close any trades that remain open at the
        end of the cycle.

        Args:
            df_chunk_in: The DataFrame containing the market data for the backtest period.
            training_results: A dictionary containing the trained model pipelines and associated metadata.
            initial_equity: The starting equity for the backtest chunk.
            cycle_history: A list of telemetry data from previous cycles for context.
            diagnosed_regime: A string indicating the market regime for this period.
            trade_lockout_until: An optional timestamp until which no new trades should be opened.
            cycle_directives: A dictionary of AI-driven directives for the current cycle.

        Returns:
            A tuple containing the log of trades, the equity curve series, a boolean indicating
            if a circuit breaker was tripped, details of the breaker event, a report of daily
            metrics, and a count of rejected trade signals.
        """
        try:
            logger.info(f"Running ENHANCED backtest on {len(df_chunk_in)} samples with regime: {diagnosed_regime}")

            # Extract ensemble models from training results
            models_dict = training_results.get('trained_pipelines', {})
            if not models_dict:
                # Fallback to single model
                model = training_results.get('model')
                if model is None:
                    logger.error("No models found in training results")
                    return pd.DataFrame(), pd.Series(), False, None, {}
                models_dict = {'single_model': model}

            # Get recent performance for dynamic weighting
            recent_performance = training_results.get('horizon_performance_metrics', {})

            # Calculate dynamic ensemble weights
            ensemble_weights = self._calculate_dynamic_ensemble_weights(models_dict, recent_performance)
            logger.info(f"Dynamic ensemble weights: {ensemble_weights}")

            # Apply regime-adaptive confidence threshold
            regime_threshold = self._get_regime_adaptive_confidence_gate(diagnosed_regime)
            adaptive_threshold = max(0.5, regime_threshold)  # Default confidence threshold
            logger.info(f"Using adaptive confidence threshold: {adaptive_threshold:.3f} (regime: {diagnosed_regime})")

            # Prepare features - get from training results
            feature_list = training_results.get('feature_list', [])
            available_features = [f for f in feature_list if f in df_chunk_in.columns]
            if len(available_features) != len(feature_list):
                missing_features = set(feature_list) - set(available_features)
                logger.warning(f"Missing features: {missing_features}")

            X = df_chunk_in[available_features].fillna(0)

            # Generate ensemble signals for each row
            signals_data = []

            for idx, (timestamp, row) in enumerate(X.iterrows()):
                # Generate ensemble signal
                signal, confidence = self._generate_ensemble_signal(
                    models_dict, row, ensemble_weights, adaptive_threshold
                )

                if signal != 0:  # Only record non-hold signals
                    signals_data.append({
                        'timestamp': timestamp,
                        'signal': signal,
                        'confidence': confidence,
                        'Close': df_chunk.loc[timestamp, 'Close'],
                        'High': df_chunk.loc[timestamp, 'High'],
                        'Low': df_chunk.loc[timestamp, 'Low'],
                        'Open': df_chunk.loc[timestamp, 'Open'],
                        'regime': regime
                    })

            if not signals_data:
                logger.warning("No signals generated above confidence threshold")
                return pd.DataFrame(), pd.Series([initial_equity]), True, None, {'total_trades': 0}

            signals_df = pd.DataFrame(signals_data)
            logger.info(f"Generated {len(signals_df)} signals above threshold")

            # Enhanced trade simulation with circuit breakers
            trades_df, equity_curve = self._simulate_enhanced_trades(signals_df, initial_equity, diagnosed_regime)

            # Calculate comprehensive metrics
            analyzer = PerformanceAnalyzer(self.config)
            summary = analyzer._calculate_metrics(trades_df, equity_curve)

            # Add regime-specific metrics
            summary['regime'] = diagnosed_regime
            summary['adaptive_threshold_used'] = adaptive_threshold
            summary['ensemble_models_count'] = len(models_dict)

            # Calculate additional analysis
            if not trades_df.empty:
                sequence_analysis = analyzer.analyze_trade_sequences(trades_df)
                summary.update(sequence_analysis)

            # Daily metrics for reporting
            daily_metrics = {}
            if not trades_df.empty:
                daily_metrics = analyzer.calculate_daily_metrics(trades_df, equity_curve)

            # Rejected signals count
            rejected_signals = {"count": len(signals_data) - len(signals_df)}

            logger.info(f"Enhanced backtest completed: {len(trades_df)} trades, "
                       f"Final equity: ${equity_curve.iloc[-1]:,.2f}, "
                       f"Sharpe: {summary.get('sharpe_ratio', 0):.3f}")

            return trades_df, equity_curve, False, None, daily_metrics, rejected_signals

        except Exception as e:
            logger.error(f"Enhanced backtest failed: {e}", exc_info=True)
            return pd.DataFrame(), pd.Series([initial_equity]), False, None, {}, {}

    def _simulate_enhanced_trades(self, signals_df: pd.DataFrame, initial_equity: float, regime: str) -> Tuple[pd.DataFrame, pd.Series]:
        """
        Enhanced trade simulation with circuit breakers, realistic execution, and advanced risk management.
        """
        trades = []
        equity_values = [initial_equity]
        current_equity = initial_equity
        current_position = 0  # 0=flat, 1=long, -1=short
        position_entry_price = 0.0
        position_size = 0.0
        consecutive_losses = 0
        trades_today = 0
        last_trade_date = None

        # Risk management parameters
        max_position_risk = getattr(self.config, 'MAX_POSITION_RISK_PCT', 0.02)
        stop_loss_pct = getattr(self.config, 'STOP_LOSS_PCT', 0.02)
        take_profit_pct = getattr(self.config, 'TAKE_PROFIT_PCT', 0.04)

        logger.info(f"Starting enhanced trade simulation with {len(signals_df)} signals")

        for idx, signal_row in signals_df.iterrows():
            try:
                timestamp = signal_row['timestamp']
                signal = signal_row['signal']
                confidence = signal_row['confidence']
                current_price = signal_row['Close']

                # Reset daily trade counter
                current_date = timestamp.date() if hasattr(timestamp, 'date') else timestamp
                if last_trade_date != current_date:
                    trades_today = 0
                    last_trade_date = current_date

                # Apply circuit breakers
                if self._apply_circuit_breakers(current_equity, initial_equity, trades_today, consecutive_losses):
                    continue

                # Calculate realistic spread
                symbol = getattr(signal_row, 'symbol', 'UNKNOWN')
                spread = self._calculate_realistic_spread(symbol)

                # Position management logic
                if current_position == 0:  # No position
                    if signal in [1, 2]:  # Long or Short signal
                        # Calculate position size based on confidence and equity
                        position_size = self._calculate_position_size(current_equity, confidence, symbol)

                        # Apply spread for entry
                        if signal == 1:  # Long
                            entry_price = current_price * (1 + spread / 10000)  # Buy at ask
                            current_position = 1
                        else:  # Short
                            entry_price = current_price * (1 - spread / 10000)  # Sell at bid
                            current_position = -1

                        position_entry_price = entry_price
                        trades_today += 1

                        logger.debug(f"Opened {('LONG' if signal == 1 else 'SHORT')} position at {entry_price:.5f}, "
                                   f"size: {position_size:.2f}, confidence: {confidence:.3f}")

                elif current_position != 0:  # Have position
                    # Check for exit conditions
                    should_exit = False
                    exit_reason = ""

                    if current_position == 1:  # Long position
                        # Apply spread for exit
                        exit_price = current_price * (1 - spread / 10000)  # Sell at bid

                        # Calculate unrealized P&L
                        unrealized_pnl_pct = (exit_price - position_entry_price) / position_entry_price

                        # Exit conditions for long
                        if signal == 2:  # Reverse signal
                            should_exit = True
                            exit_reason = "REVERSE_SIGNAL"
                        elif unrealized_pnl_pct <= -stop_loss_pct:
                            should_exit = True
                            exit_reason = "STOP_LOSS"
                        elif unrealized_pnl_pct >= take_profit_pct:
                            should_exit = True
                            exit_reason = "TAKE_PROFIT"

                    else:  # Short position
                        # Apply spread for exit
                        exit_price = current_price * (1 + spread / 10000)  # Buy at ask

                        # Calculate unrealized P&L
                        unrealized_pnl_pct = (position_entry_price - exit_price) / position_entry_price

                        # Exit conditions for short
                        if signal == 1:  # Reverse signal
                            should_exit = True
                            exit_reason = "REVERSE_SIGNAL"
                        elif unrealized_pnl_pct <= -stop_loss_pct:
                            should_exit = True
                            exit_reason = "STOP_LOSS"
                        elif unrealized_pnl_pct >= take_profit_pct:
                            should_exit = True
                            exit_reason = "TAKE_PROFIT"

                    if should_exit:
                        # Calculate trade P&L
                        if current_position == 1:  # Closing long
                            trade_pnl = (exit_price - position_entry_price) * position_size
                        else:  # Closing short
                            trade_pnl = (position_entry_price - exit_price) * position_size

                        # Apply commission
                        commission = getattr(self.config, 'COMMISSION_PER_TRADE', 5.0)
                        trade_pnl -= commission * 2  # Entry + Exit

                        # Update equity
                        current_equity += trade_pnl
                        equity_values.append(current_equity)

                        # Track consecutive losses
                        if trade_pnl < 0:
                            consecutive_losses += 1
                        else:
                            consecutive_losses = 0

                        # Record trade
                        trades.append({
                            'entry_time': timestamp,  # Simplified for this example
                            'exit_time': timestamp,
                            'direction': 'LONG' if current_position == 1 else 'SHORT',
                            'entry_price': position_entry_price,
                            'exit_price': exit_price,
                            'size': position_size,
                            'pnl': trade_pnl,
                            'exit_reason': exit_reason,
                            'confidence': confidence,
                            'regime': regime,
                            'spread_cost': spread
                        })

                        logger.debug(f"Closed {('LONG' if current_position == 1 else 'SHORT')} position: "
                                   f"P&L: ${trade_pnl:.2f}, Reason: {exit_reason}")

                        # Reset position
                        current_position = 0
                        position_entry_price = 0.0
                        position_size = 0.0
                        trades_today += 1

                        # Check if we should immediately open reverse position
                        if exit_reason == "REVERSE_SIGNAL" and not self._apply_circuit_breakers(current_equity, initial_equity, trades_today, consecutive_losses):
                            # Open reverse position
                            new_position_size = self._calculate_position_size(current_equity, confidence, symbol)

                            if signal == 1:  # New long
                                new_entry_price = current_price * (1 + spread / 10000)
                                current_position = 1
                            else:  # New short
                                new_entry_price = current_price * (1 - spread / 10000)
                                current_position = -1

                            position_entry_price = new_entry_price
                            position_size = new_position_size
                            trades_today += 1

                            logger.debug(f"Opened reverse {('LONG' if signal == 1 else 'SHORT')} position at {new_entry_price:.5f}")

            except Exception as e:
                logger.error(f"Error processing signal at {timestamp}: {e}")
                continue

        # Close any remaining position at the end
        if current_position != 0 and len(signals_df) > 0:
            final_price = signals_df.iloc[-1]['Close']
            if current_position == 1:
                final_exit_price = final_price * 0.999  # Apply spread
                final_pnl = (final_exit_price - position_entry_price) * position_size
            else:
                final_exit_price = final_price * 1.001  # Apply spread
                final_pnl = (position_entry_price - final_exit_price) * position_size

            commission = getattr(self.config, 'COMMISSION_PER_TRADE', 5.0)
            final_pnl -= commission
            current_equity += final_pnl
            equity_values.append(current_equity)

            trades.append({
                'entry_time': signals_df.iloc[-1]['timestamp'],
                'exit_time': signals_df.iloc[-1]['timestamp'],
                'direction': 'LONG' if current_position == 1 else 'SHORT',
                'entry_price': position_entry_price,
                'exit_price': final_exit_price,
                'size': position_size,
                'pnl': final_pnl,
                'exit_reason': 'END_OF_DATA',
                'confidence': signals_df.iloc[-1]['confidence'],
                'regime': regime,
                'spread_cost': 0
            })

        trades_df = pd.DataFrame(trades)
        equity_curve = pd.Series(equity_values)

        logger.info(f"Enhanced simulation completed: {len(trades)} trades, "
                   f"Final equity: ${current_equity:.2f}, "
                   f"Total return: {((current_equity / initial_equity) - 1) * 100:.2f}%")

        return trades_df, equity_curve

    def _is_trade_locked_out(self, timestamp: pd.Timestamp, last_trade_time: Optional[pd.Timestamp]) -> bool:
        """
        Check if trading is locked out due to recent trade activity.
        Prevents overtrading and allows for market cooldown periods.
        """
        if last_trade_time is None:
            return False

        lockout_minutes = getattr(self.config, 'TRADE_LOCKOUT_MINUTES', 30)
        time_since_last_trade = timestamp - last_trade_time

        return time_since_last_trade.total_seconds() < (lockout_minutes * 60)

    def _calculate_tiered_risk(self, equity: float, base_risk_pct: float) -> float:
        """
        Calculate tiered risk based on account size.
        Larger accounts use smaller risk percentages for better risk management.
        """
        tier_thresholds = getattr(self.config, 'RISK_TIER_THRESHOLDS', {
            50000: 0.015,   # Above $50k: 1.5%
            25000: 0.020,   # Above $25k: 2.0%
            10000: 0.025,   # Above $10k: 2.5%
            0: 0.030        # Below $10k: 3.0%
        })

        for threshold, risk_pct in sorted(tier_thresholds.items(), reverse=True):
            if equity >= threshold:
                return risk_pct

        return base_risk_pct

    def _apply_slippage(self, price: float, direction: str, volatility_rank: float = 0.5) -> float:
        """
        Apply realistic slippage based on market conditions and trade direction.

        Args:
            price: Base price
            direction: 'BUY' or 'SELL'
            volatility_rank: Current volatility rank (0-1)

        Returns:
            Price adjusted for slippage
        """
        base_slippage_pips = getattr(self.config, 'BASE_SLIPPAGE_PIPS', 0.5)
        volatility_multiplier = 1 + (volatility_rank * 2)  # 1x to 3x based on volatility

        slippage_pips = base_slippage_pips * volatility_multiplier
        slippage_factor = slippage_pips / 10000  # Convert pips to decimal

        if direction == 'BUY':
            return price * (1 + slippage_factor)  # Pay more when buying
        else:
            return price * (1 - slippage_factor)  # Receive less when selling

    def cleanup_end_of_cycle_positions(self, current_positions: Dict) -> Dict[str, Any]:
        """
        Clean up positions at the end of a trading cycle.
        Ensures no positions are carried over between cycles.
        """
        cleanup_summary = {
            'positions_closed': 0,
            'total_pnl': 0.0,
            'cleanup_trades': []
        }

        try:
            for symbol, position_info in current_positions.items():
                if position_info.get('size', 0) != 0:
                    # Force close position
                    current_price = position_info.get('current_price', position_info.get('entry_price', 0))
                    entry_price = position_info.get('entry_price', 0)
                    size = position_info.get('size', 0)
                    direction = position_info.get('direction', 'LONG')

                    if direction == 'LONG':
                        pnl = (current_price - entry_price) * abs(size)
                    else:
                        pnl = (entry_price - current_price) * abs(size)

                    cleanup_summary['cleanup_trades'].append({
                        'symbol': symbol,
                        'direction': direction,
                        'size': size,
                        'entry_price': entry_price,
                        'exit_price': current_price,
                        'pnl': pnl,
                        'reason': 'END_OF_CYCLE_CLEANUP'
                    })

                    cleanup_summary['positions_closed'] += 1
                    cleanup_summary['total_pnl'] += pnl

            logger.info(f"End-of-cycle cleanup: {cleanup_summary['positions_closed']} positions closed, "
                       f"Total P&L: ${cleanup_summary['total_pnl']:.2f}")

        except Exception as e:
            logger.error(f"Error during position cleanup: {e}")
            cleanup_summary['error'] = str(e)

        return cleanup_summary

    def _simulate_trades(self, signals_df: pd.DataFrame, initial_equity: float) -> Tuple[pd.DataFrame, pd.Series]:
        """Simulate trade execution based on signals."""
        
        trades = []
        equity_values = [initial_equity]
        current_equity = initial_equity
        position = None  # Current position: None, 'long', or 'short'
        entry_price = None
        entry_time = None
        trade_id = 0
        
        # Risk management parameters
        risk_per_trade = getattr(self.config, 'BASE_RISK_PER_TRADE_PCT', 0.02)
        max_risk_usd = getattr(self.config, 'RISK_CAP_PER_TRADE_USD', 500)
        
        for idx, row in signals_df.iterrows():
            current_price = row['Close']
            signal = row['prediction']
            confidence = row['confidence']
            timestamp = row['timestamp']
            
            # Close existing position if signal changes
            if position is not None and (
                (position == 'long' and signal <= 0) or 
                (position == 'short' and signal >= 0)
            ):
                # Close position
                if position == 'long':
                    pnl = (current_price - entry_price) * position_size
                else:  # short
                    pnl = (entry_price - current_price) * position_size
                
                current_equity += pnl
                
                trades.append({
                    'trade_id': trade_id,
                    'entry_time': entry_time,
                    'exit_time': timestamp,
                    'side': position,
                    'entry_price': entry_price,
                    'exit_price': current_price,
                    'position_size': position_size,
                    'pnl': pnl,
                    'confidence': confidence
                })
                
                trade_id += 1
                position = None
                entry_price = None
                entry_time = None
            
            # Open new position
            if position is None and signal != 0:
                # Calculate position size based on risk
                risk_amount = min(current_equity * risk_per_trade, max_risk_usd)
                
                # Simple position sizing (can be enhanced)
                if signal > 0:  # Long
                    position = 'long'
                    position_size = risk_amount / (current_price * 0.02)  # Assume 2% stop loss
                else:  # Short
                    position = 'short'
                    position_size = risk_amount / (current_price * 0.02)
                
                entry_price = current_price
                entry_time = timestamp
            
            equity_values.append(current_equity)
        
        # Close any remaining position at the end
        if position is not None:
            final_price = signals_df.iloc[-1]['Close']
            if position == 'long':
                pnl = (final_price - entry_price) * position_size
            else:
                pnl = (entry_price - final_price) * position_size
            
            current_equity += pnl
            
            trades.append({
                'trade_id': trade_id,
                'entry_time': entry_time,
                'exit_time': signals_df.index[-1],
                'side': position,
                'entry_price': entry_price,
                'exit_price': final_price,
                'position_size': position_size,
                'pnl': pnl,
                'confidence': signals_df.iloc[-1]['confidence']
            })
            
            equity_values.append(current_equity)
        
        # Create DataFrames
        trades_df = pd.DataFrame(trades)
        equity_curve = pd.Series(equity_values, index=range(len(equity_values)))
        
        return trades_df, equity_curve

    def calculate_dynamic_weights(self, shap_summaries: Dict, performance_history: List[Dict]) -> Dict[str, float]:
        """Calculate dynamic ensemble weights based on SHAP importance and performance."""
        if not shap_summaries or not performance_history:
            # Equal weights if no data
            return {model: 1.0 for model in shap_summaries.keys()}
        
        weights = {}
        total_weight = 0
        
        for model_name in shap_summaries.keys():
            # Base weight from SHAP importance
            shap_df = shap_summaries[model_name]
            if shap_df is not None and not shap_df.empty:
                # Use mean SHAP importance as base weight
                importance_col = 'SHAP_Importance' if 'SHAP_Importance' in shap_df.columns else shap_df.columns[1]
                base_weight = shap_df[importance_col].mean()
            else:
                base_weight = 0.5
            
            # Adjust based on recent performance
            recent_performance = [p for p in performance_history if p.get('model') == model_name]
            if recent_performance:
                avg_performance = np.mean([p.get('f1_score', 0.5) for p in recent_performance[-5:]])  # Last 5 cycles
                performance_multiplier = avg_performance / 0.5  # Normalize around 0.5
            else:
                performance_multiplier = 1.0
            
            final_weight = base_weight * performance_multiplier
            weights[model_name] = max(0.1, final_weight)  # Minimum weight of 0.1
            total_weight += weights[model_name]
        
        # Normalize weights
        if total_weight > 0:
            weights = {model: weight / total_weight for model, weight in weights.items()}
        
        return weights

    @staticmethod
    def validate_holdout_performance(guardrail_val_data: pd.DataFrame, train_result: Dict, 
                                   config: ConfigModel, feature_list: List[str]) -> Dict[str, Any]:
        """Validate model performance on holdout data."""
        try:
            model = train_result.get('model')
            if model is None:
                return {'error': 'No model available for validation'}
            
            # Prepare validation data
            X_val = guardrail_val_data[feature_list].fillna(0)
            y_val = guardrail_val_data.get('target_signal_pressure_class', pd.Series())
            
            if y_val.empty:
                return {'error': 'No target variable for validation'}
            
            # Make predictions
            predictions = model.predict(X_val)
            
            # Calculate metrics
            from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
            
            metrics = {
                'accuracy': accuracy_score(y_val, predictions),
                'f1_score': f1_score(y_val, predictions, average='weighted', zero_division=0),
                'precision': precision_score(y_val, predictions, average='weighted', zero_division=0),
                'recall': recall_score(y_val, predictions, average='weighted', zero_division=0)
            }
            
            # Run simple backtest for financial metrics
            backtester = Backtester(config)
            trades_df, equity_curve, success, _, backtest_metrics = backtester.run_backtest_chunk(
                guardrail_val_data, train_result, config.INITIAL_CAPITAL, feature_list
            )
            
            # Combine metrics
            validation_results = {**metrics, **backtest_metrics}
            
            logger.info(f"Holdout validation - F1: {metrics['f1_score']:.3f}, "
                       f"PnL: {backtest_metrics.get('total_net_profit', 0):.2f}")
            
            return validation_results
            
        except Exception as e:
            logger.error(f"Holdout validation failed: {e}")
            return {'error': str(e)}

    @staticmethod
    def get_last_run_params(params_log_file: str) -> Optional[Dict]:
        """Get parameters from last run for drift analysis."""
        try:
            import json
            if not os.path.exists(params_log_file):
                return None
            
            with open(params_log_file, 'r') as f:
                lines = f.readlines()
                if lines:
                    return json.loads(lines[-1])
            return None
        except Exception as e:
            logger.warning(f"Could not read last run params: {e}")
            return None

    @staticmethod
    def calculate_parameter_drift(current_params: Dict, last_params: Dict) -> float:
        """Calculate parameter drift percentage between runs."""
        try:
            total_drift = 0
            param_count = 0
            
            for key in current_params:
                if key in last_params:
                    current_val = current_params[key]
                    last_val = last_params[key]
                    
                    if isinstance(current_val, (int, float)) and isinstance(last_val, (int, float)):
                        if last_val != 0:
                            drift = abs((current_val - last_val) / last_val) * 100
                            total_drift += drift
                            param_count += 1
            
            return total_drift / param_count if param_count > 0 else 0
            
        except Exception as e:
            logger.warning(f"Could not calculate parameter drift: {e}")
            return 0

    @staticmethod
    def validate_ai_suggestion(suggested_params: Dict, historical_performance_log: List, 
                             holdout_data: pd.DataFrame, config: ConfigModel, 
                             model_trainer, symbol: str, playbook: Dict) -> bool:
        """Validate AI suggestions against guardrails."""
        logger.info("AI SUGGESTION VALIDATION PROTOCOL INITIATED")
        
        try:
            # Check parameter drift
            last_run_params = Backtester.get_last_run_params(getattr(config, 'PARAMS_LOG_FILE', 'params.log'))
            if last_run_params:
                param_drift = Backtester.calculate_parameter_drift(suggested_params, last_run_params)
                max_drift = getattr(config, 'MAX_PARAM_DRIFT_TOLERANCE', 50)
                
                logger.info(f"Parameter drift: {param_drift:.2f}% (Tolerance: {max_drift}%)")
                if param_drift > max_drift:
                    logger.warning("GUARDRAIL FAILED: Excessive parameter drift. REJECTING.")
                    return False
            
            # Additional validation logic can be added here
            logger.info("AI suggestion validation PASSED")
            return True
            
        except Exception as e:
            logger.error(f"AI suggestion validation failed: {e}")
            return False
