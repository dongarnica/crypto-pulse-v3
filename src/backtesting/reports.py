"""
Report generation for backtesting results.
"""

import logging
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
import json
import os
from pathlib import Path
import base64
from io import BytesIO

from .results import BacktestResults, TradeResult, PortfolioSnapshot

logger = logging.getLogger(__name__)

# Set matplotlib backend for headless environments
plt.switch_backend('Agg')
sns.set_style("whitegrid")
plt.style.use('seaborn-v0_8')


class BacktestReportGenerator:
    """Generate comprehensive backtesting reports with visualizations."""
    
    def __init__(self, results: BacktestResults):
        self.results = results
        self.output_dir = Path("reports") / f"backtest_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
    def _save_plot(self, filename: str) -> str:
        """Save current plot and return the filename."""
        filepath = self.output_dir / filename
        plt.savefig(filepath, dpi=300, bbox_inches='tight', facecolor='white')
        plt.close()
        return str(filepath)
    
    def _plot_to_base64(self) -> str:
        """Convert current plot to base64 string for HTML embedding."""
        buffer = BytesIO()
        plt.savefig(buffer, format='png', dpi=150, bbox_inches='tight', facecolor='white')
        plt.close()
        buffer.seek(0)
        image_base64 = base64.b64encode(buffer.read()).decode('utf-8')
        buffer.close()
        return f"data:image/png;base64,{image_base64}"
    
    def create_portfolio_value_chart(self) -> str:
        """Create portfolio value over time chart."""
        if not self.results.portfolio_snapshots:
            return ""
        
        df = pd.DataFrame([{
            'timestamp': snap.timestamp,
            'total_value': snap.total_value,
            'cash': snap.cash
        } for snap in self.results.portfolio_snapshots])
        
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df.set_index('timestamp', inplace=True)
        
        plt.figure(figsize=(12, 6))
        plt.plot(df.index, df['total_value'], label='Total Portfolio Value', linewidth=2)
        plt.axhline(y=self.results.initial_capital, color='r', linestyle='--', 
                   label=f'Initial Capital (${self.results.initial_capital:,.0f})')
        
        plt.title('Portfolio Value Over Time', fontsize=14, fontweight='bold')
        plt.xlabel('Date')
        plt.ylabel('Portfolio Value ($)')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Format y-axis with dollar signs
        ax = plt.gca()
        ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'${x:,.0f}'))
        
        return self._plot_to_base64()
    
    def create_drawdown_chart(self) -> str:
        """Create drawdown chart."""
        if not self.results.portfolio_snapshots:
            return ""
        
        df = pd.DataFrame([{
            'timestamp': snap.timestamp,
            'total_value': snap.total_value
        } for snap in self.results.portfolio_snapshots])
        
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df.set_index('timestamp', inplace=True)
        
        # Calculate running maximum and drawdown
        df['running_max'] = df['total_value'].cummax()
        df['drawdown'] = (df['total_value'] - df['running_max']) / df['running_max']
        
        plt.figure(figsize=(12, 6))
        plt.fill_between(df.index, df['drawdown'], 0, alpha=0.3, color='red', label='Drawdown')
        plt.plot(df.index, df['drawdown'], color='red', linewidth=1)
        
        plt.title('Portfolio Drawdown Over Time', fontsize=14, fontweight='bold')
        plt.xlabel('Date')
        plt.ylabel('Drawdown (%)')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Format y-axis as percentage
        ax = plt.gca()
        ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'{x:.1%}'))
        
        return self._plot_to_base64()
    
    def create_monthly_returns_heatmap(self) -> str:
        """Create monthly returns heatmap."""
        if not self.results.portfolio_snapshots:
            return ""
        
        df = pd.DataFrame([{
            'timestamp': snap.timestamp,
            'total_value': snap.total_value
        } for snap in self.results.portfolio_snapshots])
        
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df.set_index('timestamp', inplace=True)
        
        # Calculate monthly returns
        monthly_values = df.resample('M')['total_value'].last()
        monthly_returns = monthly_values.pct_change().dropna()
        
        if len(monthly_returns) < 2:
            return ""
        
        # Create pivot table for heatmap
        monthly_returns_df = monthly_returns.to_frame('return')
        monthly_returns_df['year'] = monthly_returns_df.index.year
        monthly_returns_df['month'] = monthly_returns_df.index.month
        
        pivot_table = monthly_returns_df.pivot(index='year', columns='month', values='return')
        
        plt.figure(figsize=(12, 6))
        sns.heatmap(pivot_table, annot=True, fmt='.2%', cmap='RdYlGn', center=0,
                   cbar_kws={'label': 'Monthly Return'})
        
        plt.title('Monthly Returns Heatmap', fontsize=14, fontweight='bold')
        plt.xlabel('Month')
        plt.ylabel('Year')
        
        # Set month labels
        month_labels = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
                       'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
        plt.gca().set_xticklabels(month_labels)
        
        return self._plot_to_base64()
    
    def create_trade_analysis_chart(self) -> str:
        """Create trade analysis chart."""
        if not self.results.trades:
            return ""
        
        trades_df = pd.DataFrame([{
            'timestamp': trade.timestamp,
            'symbol': trade.symbol,
            'side': trade.side,
            'pnl': float(trade.pnl) if trade.pnl else 0.0,
            'pnl_percentage': trade.pnl_percentage or 0.0
        } for trade in self.results.trades if trade.pnl is not None])
        
        if trades_df.empty:
            return ""
        
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
        
        # Trade P&L distribution
        ax1.hist(trades_df['pnl'], bins=30, alpha=0.7, color='steelblue', edgecolor='black')
        ax1.set_title('Trade P&L Distribution')
        ax1.set_xlabel('P&L ($)')
        ax1.set_ylabel('Frequency')
        ax1.axvline(x=0, color='red', linestyle='--', alpha=0.7)
        
        # Cumulative P&L
        trades_df_sorted = trades_df.sort_values('timestamp')
        trades_df_sorted['cumulative_pnl'] = trades_df_sorted['pnl'].cumsum()
        ax2.plot(trades_df_sorted['timestamp'], trades_df_sorted['cumulative_pnl'], 
                linewidth=2, color='green')
        ax2.set_title('Cumulative P&L Over Time')
        ax2.set_xlabel('Date')
        ax2.set_ylabel('Cumulative P&L ($)')
        ax2.grid(True, alpha=0.3)
        
        # Trades by symbol
        symbol_pnl = trades_df.groupby('symbol')['pnl'].sum().sort_values(ascending=False)
        ax3.bar(range(len(symbol_pnl)), symbol_pnl.values, color='lightcoral')
        ax3.set_title('P&L by Symbol')
        ax3.set_xlabel('Symbol')
        ax3.set_ylabel('Total P&L ($)')
        ax3.set_xticks(range(len(symbol_pnl)))
        ax3.set_xticklabels(symbol_pnl.index, rotation=45)
        
        # Win/Loss analysis
        wins = trades_df[trades_df['pnl'] > 0]['pnl']
        losses = trades_df[trades_df['pnl'] < 0]['pnl']
        
        win_loss_data = [len(wins), len(losses)]
        win_loss_labels = [f'Wins ({len(wins)})', f'Losses ({len(losses)})']
        colors = ['green', 'red']
        
        ax4.pie(win_loss_data, labels=win_loss_labels, colors=colors, autopct='%1.1f%%')
        ax4.set_title('Win/Loss Ratio')
        
        plt.tight_layout()
        return self._plot_to_base64()
    
    def create_risk_metrics_chart(self) -> str:
        """Create risk metrics visualization."""
        if not self.results.portfolio_snapshots:
            return ""
        
        df = pd.DataFrame([{
            'timestamp': snap.timestamp,
            'total_value': snap.total_value
        } for snap in self.results.portfolio_snapshots])
        
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df.set_index('timestamp', inplace=True)
        df['returns'] = df['total_value'].pct_change().dropna()
        
        if len(df['returns'].dropna()) < 10:
            return ""
        
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
        
        # Returns distribution
        returns_clean = df['returns'].dropna()
        ax1.hist(returns_clean, bins=30, alpha=0.7, color='steelblue', edgecolor='black')
        ax1.set_title('Returns Distribution')
        ax1.set_xlabel('Daily Returns')
        ax1.set_ylabel('Frequency')
        ax1.axvline(x=returns_clean.mean(), color='red', linestyle='--', 
                   label=f'Mean: {returns_clean.mean():.4f}')
        ax1.legend()
        
        # Rolling volatility
        rolling_vol = returns_clean.rolling(window=30).std() * np.sqrt(252)
        ax2.plot(rolling_vol.index, rolling_vol, linewidth=2, color='orange')
        ax2.set_title('Rolling 30-Day Volatility (Annualized)')
        ax2.set_xlabel('Date')
        ax2.set_ylabel('Volatility')
        ax2.grid(True, alpha=0.3)
        
        # Rolling Sharpe ratio
        rolling_returns = returns_clean.rolling(window=30).mean() * 252
        rolling_sharpe = rolling_returns / rolling_vol
        ax3.plot(rolling_sharpe.index, rolling_sharpe, linewidth=2, color='purple')
        ax3.set_title('Rolling 30-Day Sharpe Ratio')
        ax3.set_xlabel('Date')
        ax3.set_ylabel('Sharpe Ratio')
        ax3.axhline(y=0, color='red', linestyle='--', alpha=0.7)
        ax3.grid(True, alpha=0.3)
        
        # Underwater curve (drawdown)
        running_max = df['total_value'].cummax()
        underwater = (df['total_value'] - running_max) / running_max
        ax4.fill_between(underwater.index, underwater, 0, alpha=0.3, color='red')
        ax4.plot(underwater.index, underwater, color='red', linewidth=1)
        ax4.set_title('Underwater Curve')
        ax4.set_xlabel('Date')
        ax4.set_ylabel('Drawdown')
        ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        return self._plot_to_base64()
    
    def create_correlation_matrix(self) -> str:
        """Create correlation matrix of symbol returns."""
        if not self.results.trades:
            return ""
        
        # Get trade data by symbol
        symbol_returns = {}
        
        for trade in self.results.trades:
            if trade.pnl is not None:
                if trade.symbol not in symbol_returns:
                    symbol_returns[trade.symbol] = []
                symbol_returns[trade.symbol].append(trade.pnl_percentage or 0.0)
        
        if len(symbol_returns) < 2:
            return ""
        
        # Create DataFrame
        max_len = max(len(returns) for returns in symbol_returns.values())
        
        for symbol in symbol_returns:
            # Pad shorter lists with zeros
            while len(symbol_returns[symbol]) < max_len:
                symbol_returns[symbol].append(0.0)
        
        returns_df = pd.DataFrame(symbol_returns)
        correlation_matrix = returns_df.corr()
        
        plt.figure(figsize=(10, 8))
        sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0,
                   square=True, cbar_kws={'label': 'Correlation'})
        
        plt.title('Symbol Returns Correlation Matrix', fontsize=14, fontweight='bold')
        plt.tight_layout()
        
        return self._plot_to_base64()
    
    def generate_html_report(self) -> str:
        """Generate comprehensive HTML report."""
        logger.info("Generating HTML report...")
        
        # Generate all charts
        portfolio_chart = self.create_portfolio_value_chart()
        drawdown_chart = self.create_drawdown_chart()
        monthly_heatmap = self.create_monthly_returns_heatmap()
        trade_analysis = self.create_trade_analysis_chart()
        risk_metrics = self.create_risk_metrics_chart()
        correlation_matrix = self.create_correlation_matrix()
        
        # Calculate additional statistics
        total_trades = len([t for t in self.results.trades if t.pnl is not None])
        winning_trades = len([t for t in self.results.trades if t.pnl and t.pnl > 0])
        losing_trades = len([t for t in self.results.trades if t.pnl and t.pnl < 0])
        
        html_content = f"""
        <!DOCTYPE html>
        <html lang="en">
        <head>
            <meta charset="UTF-8">
            <meta name="viewport" content="width=device-width, initial-scale=1.0">
            <title>Backtest Report - {self.results.start_date.strftime('%Y-%m-%d')} to {self.results.end_date.strftime('%Y-%m-%d')}</title>
            <style>
                body {{
                    font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
                    line-height: 1.6;
                    margin: 0;
                    padding: 20px;
                    background-color: #f5f5f5;
                }}
                .container {{
                    max-width: 1200px;
                    margin: 0 auto;
                    background-color: white;
                    padding: 30px;
                    border-radius: 10px;
                    box-shadow: 0 0 20px rgba(0,0,0,0.1);
                }}
                h1 {{
                    color: #2c3e50;
                    text-align: center;
                    border-bottom: 3px solid #3498db;
                    padding-bottom: 10px;
                }}
                h2 {{
                    color: #34495e;
                    border-left: 4px solid #3498db;
                    padding-left: 15px;
                }}
                .metrics-grid {{
                    display: grid;
                    grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
                    gap: 20px;
                    margin: 20px 0;
                }}
                .metric-card {{
                    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                    color: white;
                    padding: 20px;
                    border-radius: 10px;
                    text-align: center;
                    box-shadow: 0 4px 6px rgba(0,0,0,0.1);
                }}
                .metric-value {{
                    font-size: 2em;
                    font-weight: bold;
                    margin-bottom: 5px;
                }}
                .metric-label {{
                    font-size: 0.9em;
                    opacity: 0.9;
                }}
                .chart-container {{
                    margin: 30px 0;
                    text-align: center;
                }}
                .chart-container img {{
                    max-width: 100%;
                    height: auto;
                    border-radius: 8px;
                    box-shadow: 0 4px 8px rgba(0,0,0,0.1);
                }}
                .config-section {{
                    background-color: #ecf0f1;
                    padding: 20px;
                    border-radius: 8px;
                    margin: 20px 0;
                }}
                .config-grid {{
                    display: grid;
                    grid-template-columns: repeat(auto-fit, minmax(300px, 1fr));
                    gap: 15px;
                }}
                .positive {{ color: #27ae60; }}
                .negative {{ color: #e74c3c; }}
                .neutral {{ color: #7f8c8d; }}
                .trade-summary {{
                    background-color: #f8f9fa;
                    padding: 15px;
                    border-radius: 8px;
                    margin: 15px 0;
                }}
            </style>
        </head>
        <body>
            <div class="container">
                <h1>📊 Backtesting Report</h1>
                <p style="text-align: center; font-size: 1.1em; color: #7f8c8d;">
                    Period: {self.results.start_date.strftime('%Y-%m-%d')} to {self.results.end_date.strftime('%Y-%m-%d')}
                    | Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
                </p>
                
                <h2>📈 Performance Summary</h2>
                <div class="metrics-grid">
                    <div class="metric-card">
                        <div class="metric-value">${self.results.final_capital:,.0f}</div>
                        <div class="metric-label">Final Capital</div>
                    </div>
                    <div class="metric-card">
                        <div class="metric-value {'positive' if (self.results.total_return or 0) > 0 else 'negative'}">{(self.results.total_return or 0)*100:.2f}%</div>
                        <div class="metric-label">Total Return</div>
                    </div>
                    <div class="metric-card">
                        <div class="metric-value">{self.results.sharpe_ratio or 0:.3f}</div>
                        <div class="metric-label">Sharpe Ratio</div>
                    </div>
                    <div class="metric-card">
                        <div class="metric-value negative">{(self.results.max_drawdown or 0)*100:.2f}%</div>
                        <div class="metric-label">Max Drawdown</div>
                    </div>
                    <div class="metric-card">
                        <div class="metric-value">{(self.results.win_rate or 0)*100:.1f}%</div>
                        <div class="metric-label">Win Rate</div>
                    </div>
                    <div class="metric-card">
                        <div class="metric-value">{total_trades}</div>
                        <div class="metric-label">Total Trades</div>
                    </div>
                </div>
                
                <div class="trade-summary">
                    <h3>🔢 Trade Statistics</h3>
                    <div class="config-grid">
                        <div><strong>Winning Trades:</strong> {winning_trades}</div>
                        <div><strong>Losing Trades:</strong> {losing_trades}</div>
                        <div><strong>Profit Factor:</strong> {self.results.profit_factor or 0:.2f}</div>
                        <div><strong>Sortino Ratio:</strong> {self.results.sortino_ratio or 0:.3f}</div>
                        <div><strong>Calmar Ratio:</strong> {self.results.calmar_ratio or 0:.3f}</div>
                        <div><strong>Average Win:</strong> ${self.results.average_win or 0:.2f}</div>
                        <div><strong>Average Loss:</strong> ${self.results.average_loss or 0:.2f}</div>
                        <div><strong>Max Consecutive Wins:</strong> {self.results.max_consecutive_wins or 0}</div>
                        <div><strong>Max Consecutive Losses:</strong> {self.results.max_consecutive_losses or 0}</div>
                    </div>
                </div>
        """
        
        # Add charts
        if portfolio_chart:
            html_content += f"""
                <h2>📊 Portfolio Performance</h2>
                <div class="chart-container">
                    <img src="{portfolio_chart}" alt="Portfolio Value Chart">
                </div>
            """
        
        if drawdown_chart:
            html_content += f"""
                <h2>📉 Drawdown Analysis</h2>
                <div class="chart-container">
                    <img src="{drawdown_chart}" alt="Drawdown Chart">
                </div>
            """
        
        if monthly_heatmap:
            html_content += f"""
                <h2>🗓️ Monthly Returns</h2>
                <div class="chart-container">
                    <img src="{monthly_heatmap}" alt="Monthly Returns Heatmap">
                </div>
            """
        
        if trade_analysis:
            html_content += f"""
                <h2>💹 Trade Analysis</h2>
                <div class="chart-container">
                    <img src="{trade_analysis}" alt="Trade Analysis Charts">
                </div>
            """
        
        if risk_metrics:
            html_content += f"""
                <h2>⚠️ Risk Metrics</h2>
                <div class="chart-container">
                    <img src="{risk_metrics}" alt="Risk Metrics Charts">
                </div>
            """
        
        if correlation_matrix:
            html_content += f"""
                <h2>🔗 Symbol Correlation</h2>
                <div class="chart-container">
                    <img src="{correlation_matrix}" alt="Correlation Matrix">
                </div>
            """
        
        # Add configuration section
        html_content += f"""
                <h2>⚙️ Backtest Configuration</h2>
                <div class="config-section">
                    <div class="config-grid">
                        <div><strong>Initial Capital:</strong> ${self.results.initial_capital:,.0f}</div>
                        <div><strong>Symbols:</strong> {', '.join(self.results.config.symbols)}</div>
                        <div><strong>Max Portfolio Allocation:</strong> {self.results.config.max_portfolio_allocation*100:.1f}%</div>
                        <div><strong>Commission Rate:</strong> {self.results.config.commission_rate*100:.2f}%</div>
                        <div><strong>Slippage:</strong> {self.results.config.slippage_bps} bps</div>
                        <div><strong>Analysis Interval:</strong> {self.results.config.analysis_interval_minutes} min</div>
                    </div>
                </div>
                
                <div style="text-align: center; margin-top: 40px; padding: 20px; background-color: #ecf0f1; border-radius: 8px;">
                    <p style="color: #7f8c8d; margin: 0;">
                        Report generated by Crypto Pulse V3 Backtesting Engine<br>
                        <small>This report is for educational and analysis purposes only. Past performance does not guarantee future results.</small>
                    </p>
                </div>
            </div>
        </body>
        </html>
        """
        
        # Save HTML report
        html_file = self.output_dir / "backtest_report.html"
        with open(html_file, 'w', encoding='utf-8') as f:
            f.write(html_content)
        
        logger.info(f"HTML report saved to: {html_file}")
        return str(html_file)
    
    def generate_json_report(self) -> str:
        """Generate JSON report with all data."""
        logger.info("Generating JSON report...")
        
        report_data = {
            'metadata': {
                'generated_at': datetime.now().isoformat(),
                'backtest_period': {
                    'start': self.results.start_date.isoformat(),
                    'end': self.results.end_date.isoformat()
                },
                'total_trades': len([t for t in self.results.trades if t.pnl is not None])
            },
            'performance_metrics': {
                'initial_capital': self.results.initial_capital,
                'final_capital': self.results.final_capital,
                'total_return': self.results.total_return,
                'annualized_return': self.results.annualized_return,
                'sharpe_ratio': self.results.sharpe_ratio,
                'sortino_ratio': self.results.sortino_ratio,
                'calmar_ratio': self.results.calmar_ratio,
                'max_drawdown': self.results.max_drawdown,
                'win_rate': self.results.win_rate,
                'profit_factor': self.results.profit_factor,
                'average_win': self.results.average_win,
                'average_loss': self.results.average_loss,
                'max_consecutive_wins': self.results.max_consecutive_wins,
                'max_consecutive_losses': self.results.max_consecutive_losses
            },
            'configuration': {
                'symbols': self.results.config.symbols,
                'max_portfolio_allocation': self.results.config.max_portfolio_allocation,
                'commission_rate': self.results.config.commission_rate,
                'slippage_bps': self.results.config.slippage_bps,
                'analysis_interval_minutes': self.results.config.analysis_interval_minutes
            },
            'trades': [trade.to_dict() for trade in self.results.trades if trade.pnl is not None],
            'portfolio_snapshots': [
                {
                    'timestamp': snap.timestamp.isoformat(),
                    'total_value': snap.total_value,
                    'cash': snap.cash,
                    'positions': snap.positions,
                    'position_values': snap.position_values,
                    'unrealized_pnl': snap.unrealized_pnl
                } for snap in self.results.portfolio_snapshots
            ]
        }
        
        json_file = self.output_dir / "backtest_results.json"
        with open(json_file, 'w', encoding='utf-8') as f:
            json.dump(report_data, f, indent=2, default=str)
        
        logger.info(f"JSON report saved to: {json_file}")
        return str(json_file)
    
    def generate_csv_export(self) -> str:
        """Generate CSV files for further analysis."""
        logger.info("Generating CSV exports...")
        
        # Export trades
        if self.results.trades:
            trades_data = []
            for trade in self.results.trades:
                if trade.pnl is not None:
                    trades_data.append({
                        'trade_id': trade.trade_id,
                        'timestamp': trade.timestamp,
                        'symbol': trade.symbol,
                        'side': trade.side,
                        'quantity': float(trade.quantity),
                        'price': float(trade.price),
                        'commission': float(trade.commission),
                        'slippage': float(trade.slippage),
                        'pnl': float(trade.pnl),
                        'pnl_percentage': trade.pnl_percentage,
                        'holding_period_hours': trade.holding_period_hours,
                        'signal_confidence': trade.signal_confidence
                    })
            
            trades_df = pd.DataFrame(trades_data)
            trades_csv = self.output_dir / "trades.csv"
            trades_df.to_csv(trades_csv, index=False)
            logger.info(f"Trades CSV saved to: {trades_csv}")
        
        # Export portfolio snapshots
        if self.results.portfolio_snapshots:
            snapshots_data = []
            for snap in self.results.portfolio_snapshots:
                snapshots_data.append({
                    'timestamp': snap.timestamp,
                    'total_value': snap.total_value,
                    'cash': snap.cash,
                    'unrealized_pnl': snap.unrealized_pnl
                })
            
            snapshots_df = pd.DataFrame(snapshots_data)
            snapshots_csv = self.output_dir / "portfolio_snapshots.csv"
            snapshots_df.to_csv(snapshots_csv, index=False)
            logger.info(f"Portfolio snapshots CSV saved to: {snapshots_csv}")
        
        return str(self.output_dir)
    
    def generate_full_report(self) -> Dict[str, str]:
        """Generate all report formats."""
        logger.info("Generating complete backtest report...")
        
        report_files = {
            'html_report': self.generate_html_report(),
            'json_report': self.generate_json_report(),
            'csv_export_dir': self.generate_csv_export()
        }
        
        logger.info(f"All reports generated in: {self.output_dir}")
        return report_files
