"""
Backtesting results analysis and reporting.
"""

import pandas as pd
import numpy as np
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any
from decimal import Decimal
import json
import logging

logger = logging.getLogger(__name__)


@dataclass
class TradeResult:
    """Individual trade result for backtesting."""
    
    trade_id: str
    symbol: str
    timestamp: datetime
    side: str  # 'BUY' or 'SELL'
    quantity: Decimal
    price: Decimal
    commission: Decimal
    slippage: Decimal
    
    # P&L information (for closing trades)
    pnl: Optional[Decimal] = None
    pnl_percentage: Optional[float] = None
    holding_period_hours: Optional[float] = None
    
    # Signal information
    signal_confidence: Optional[float] = None
    ml_prediction: Optional[Dict[str, float]] = None
    technical_indicators: Optional[Dict[str, float]] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert trade result to dictionary."""
        return {
            'trade_id': self.trade_id,
            'symbol': self.symbol,
            'timestamp': self.timestamp.isoformat(),
            'side': self.side,
            'quantity': float(self.quantity),
            'price': float(self.price),
            'commission': float(self.commission),
            'slippage': float(self.slippage),
            'pnl': float(self.pnl) if self.pnl else None,
            'pnl_percentage': self.pnl_percentage,
            'holding_period_hours': self.holding_period_hours,
            'signal_confidence': self.signal_confidence,
            'ml_prediction': self.ml_prediction,
            'technical_indicators': self.technical_indicators
        }


@dataclass
class PortfolioSnapshot:
    """Portfolio state snapshot during backtesting."""
    
    timestamp: datetime
    total_value: Decimal
    cash_balance: Decimal
    positions_value: Decimal
    
    # Performance metrics
    daily_return: float
    cumulative_return: float
    drawdown: float
    
    # Risk metrics
    volatility: float
    sharpe_ratio: float
    
    # Position details
    positions: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert portfolio snapshot to dictionary."""
        return {
            'timestamp': self.timestamp.isoformat(),
            'total_value': float(self.total_value),
            'cash_balance': float(self.cash_balance),
            'positions_value': float(self.positions_value),
            'daily_return': self.daily_return,
            'cumulative_return': self.cumulative_return,
            'drawdown': self.drawdown,
            'volatility': self.volatility,
            'sharpe_ratio': self.sharpe_ratio,
            'positions': self.positions
        }


@dataclass
class BacktestResults:
    """Comprehensive backtesting results."""
    
    # Configuration
    config: 'BacktestConfig'
    start_time: datetime
    end_time: datetime
    duration_seconds: float
    
    # Trade history
    trades: List[TradeResult] = field(default_factory=list)
    portfolio_snapshots: List[PortfolioSnapshot] = field(default_factory=list)
    
    # Final metrics
    initial_capital: Decimal = Decimal('0')
    final_value: Decimal = Decimal('0')
    total_return: float = 0.0
    annualized_return: float = 0.0
    
    # Risk metrics
    sharpe_ratio: float = 0.0
    sortino_ratio: float = 0.0
    calmar_ratio: float = 0.0
    max_drawdown: float = 0.0
    volatility: float = 0.0
    var_95: float = 0.0
    
    # Trading statistics
    total_trades: int = 0
    winning_trades: int = 0
    losing_trades: int = 0
    win_rate: float = 0.0
    profit_factor: float = 0.0
    average_win: float = 0.0
    average_loss: float = 0.0
    
    # Benchmark comparison
    benchmark_return: float = 0.0
    alpha: float = 0.0
    beta: float = 0.0
    information_ratio: float = 0.0
    
    # Additional metrics
    recovery_factor: float = 0.0
    ulcer_index: float = 0.0
    sterling_ratio: float = 0.0
    
    def calculate_metrics(self):
        """Calculate all performance metrics from trades and snapshots."""
        try:
            if not self.portfolio_snapshots or not self.trades:
                logger.warning("No data available for metrics calculation")
                return
            
            # Basic metrics
            self.initial_capital = self.portfolio_snapshots[0].total_value
            self.final_value = self.portfolio_snapshots[-1].total_value
            self.total_return = float((self.final_value - self.initial_capital) / self.initial_capital)
            
            # Annualized return
            days = (self.end_time - self.start_time).days
            if days > 0:
                self.annualized_return = (1 + self.total_return) ** (365.0 / days) - 1
            
            # Calculate returns series
            returns = self._calculate_returns_series()
            
            # Risk metrics
            self.sharpe_ratio = self._calculate_sharpe_ratio(returns)
            self.sortino_ratio = self._calculate_sortino_ratio(returns)
            self.max_drawdown = self._calculate_max_drawdown()
            self.volatility = float(np.std(returns) * np.sqrt(252)) if len(returns) > 1 else 0.0
            self.var_95 = float(np.percentile(returns, 5)) if len(returns) > 0 else 0.0
            
            # Trading statistics
            self._calculate_trading_statistics()
            
            # Additional metrics
            self.recovery_factor = abs(self.total_return / self.max_drawdown) if self.max_drawdown != 0 else 0.0
            self.ulcer_index = self._calculate_ulcer_index()
            
            logger.info("Backtest metrics calculated successfully")
            
        except Exception as e:
            logger.error(f"Error calculating backtest metrics: {e}")
    
    def _calculate_returns_series(self) -> np.ndarray:
        """Calculate daily returns series from portfolio snapshots."""
        values = [float(snap.total_value) for snap in self.portfolio_snapshots]
        returns = np.diff(values) / values[:-1]
        return returns
    
    def _calculate_sharpe_ratio(self, returns: np.ndarray, risk_free_rate: float = 0.02) -> float:
        """Calculate Sharpe ratio."""
        if len(returns) == 0:
            return 0.0
        
        excess_returns = returns - (risk_free_rate / 252)
        return float(np.mean(excess_returns) / np.std(excess_returns) * np.sqrt(252)) if np.std(excess_returns) > 0 else 0.0
    
    def _calculate_sortino_ratio(self, returns: np.ndarray, risk_free_rate: float = 0.02) -> float:
        """Calculate Sortino ratio (downside deviation)."""
        if len(returns) == 0:
            return 0.0
        
        excess_returns = returns - (risk_free_rate / 252)
        downside_returns = excess_returns[excess_returns < 0]
        
        if len(downside_returns) == 0:
            return float('inf')
        
        downside_deviation = np.std(downside_returns)
        return float(np.mean(excess_returns) / downside_deviation * np.sqrt(252)) if downside_deviation > 0 else 0.0
    
    def _calculate_max_drawdown(self) -> float:
        """Calculate maximum drawdown from portfolio snapshots."""
        values = [float(snap.total_value) for snap in self.portfolio_snapshots]
        peak = values[0]
        max_dd = 0.0
        
        for value in values:
            if value > peak:
                peak = value
            drawdown = (peak - value) / peak
            max_dd = max(max_dd, drawdown)
        
        return max_dd
    
    def _calculate_trading_statistics(self):
        """Calculate trading performance statistics."""
        closed_trades = [trade for trade in self.trades if trade.pnl is not None]
        
        self.total_trades = len(closed_trades)
        
        if self.total_trades == 0:
            return
        
        winning_trades = [trade for trade in closed_trades if trade.pnl > 0]
        losing_trades = [trade for trade in closed_trades if trade.pnl < 0]
        
        self.winning_trades = len(winning_trades)
        self.losing_trades = len(losing_trades)
        self.win_rate = self.winning_trades / self.total_trades
        
        if winning_trades:
            self.average_win = float(np.mean([float(trade.pnl) for trade in winning_trades]))
        
        if losing_trades:
            self.average_loss = float(abs(np.mean([float(trade.pnl) for trade in losing_trades])))
        
        # Profit factor
        total_wins = sum(float(trade.pnl) for trade in winning_trades)
        total_losses = abs(sum(float(trade.pnl) for trade in losing_trades))
        
        self.profit_factor = total_wins / total_losses if total_losses > 0 else float('inf')
    
    def _calculate_ulcer_index(self) -> float:
        """Calculate Ulcer Index (measure of downside risk)."""
        values = [float(snap.total_value) for snap in self.portfolio_snapshots]
        running_max = np.maximum.accumulate(values)
        drawdowns = (running_max - values) / running_max * 100
        ulcer_index = np.sqrt(np.mean(drawdowns ** 2))
        return float(ulcer_index)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert results to dictionary for serialization."""
        return {
            'start_time': self.start_time.isoformat(),
            'end_time': self.end_time.isoformat(),
            'duration_seconds': self.duration_seconds,
            'initial_capital': float(self.initial_capital),
            'final_value': float(self.final_value),
            'total_return': self.total_return,
            'annualized_return': self.annualized_return,
            'sharpe_ratio': self.sharpe_ratio,
            'sortino_ratio': self.sortino_ratio,
            'calmar_ratio': self.calmar_ratio,
            'max_drawdown': self.max_drawdown,
            'volatility': self.volatility,
            'var_95': self.var_95,
            'total_trades': self.total_trades,
            'winning_trades': self.winning_trades,
            'losing_trades': self.losing_trades,
            'win_rate': self.win_rate,
            'profit_factor': self.profit_factor,
            'average_win': self.average_win,
            'average_loss': self.average_loss,
            'benchmark_return': self.benchmark_return,
            'alpha': self.alpha,
            'beta': self.beta,
            'information_ratio': self.information_ratio,
            'recovery_factor': self.recovery_factor,
            'ulcer_index': self.ulcer_index,
            'sterling_ratio': self.sterling_ratio,
            'trades': [trade.to_dict() for trade in self.trades],
            'portfolio_snapshots': [snap.to_dict() for snap in self.portfolio_snapshots]
        }


class BacktestReport:
    """Generate comprehensive backtesting reports."""
    
    def __init__(self, results: BacktestResults):
        self.results = results
    
    def generate_summary(self) -> str:
        """Generate a text summary of backtest results."""
        summary = f"""
CRYPTO PULSE V3 - BACKTEST RESULTS SUMMARY
==========================================

Period: {self.results.start_time.strftime('%Y-%m-%d')} to {self.results.end_time.strftime('%Y-%m-%d')}
Duration: {self.results.duration_seconds/3600:.1f} hours

PERFORMANCE METRICS
-------------------
Initial Capital: ${self.results.initial_capital:,.2f}
Final Value: ${self.results.final_value:,.2f}
Total Return: {self.results.total_return:.2%}
Annualized Return: {self.results.annualized_return:.2%}

RISK METRICS
------------
Sharpe Ratio: {self.results.sharpe_ratio:.3f}
Sortino Ratio: {self.results.sortino_ratio:.3f}
Maximum Drawdown: {self.results.max_drawdown:.2%}
Volatility: {self.results.volatility:.2%}
VaR (95%): {self.results.var_95:.2%}

TRADING STATISTICS
------------------
Total Trades: {self.results.total_trades}
Win Rate: {self.results.win_rate:.2%}
Profit Factor: {self.results.profit_factor:.2f}
Average Win: ${self.results.average_win:.2f}
Average Loss: ${self.results.average_loss:.2f}

ADDITIONAL METRICS
------------------
Recovery Factor: {self.results.recovery_factor:.2f}
Ulcer Index: {self.results.ulcer_index:.2f}
"""
        return summary
    
    def save_to_json(self, filepath: str):
        """Save results to JSON file."""
        try:
            with open(filepath, 'w') as f:
                json.dump(self.results.to_dict(), f, indent=2, default=str)
            logger.info(f"Backtest results saved to {filepath}")
        except Exception as e:
            logger.error(f"Error saving results to JSON: {e}")
    
    def save_trades_to_csv(self, filepath: str):
        """Save trade history to CSV file."""
        try:
            trades_data = [trade.to_dict() for trade in self.results.trades]
            df = pd.DataFrame(trades_data)
            df.to_csv(filepath, index=False)
            logger.info(f"Trade history saved to {filepath}")
        except Exception as e:
            logger.error(f"Error saving trades to CSV: {e}")
    
    def save_portfolio_snapshots_to_csv(self, filepath: str):
        """Save portfolio snapshots to CSV file."""
        try:
            snapshots_data = [snap.to_dict() for snap in self.results.portfolio_snapshots]
            df = pd.DataFrame(snapshots_data)
            df.to_csv(filepath, index=False)
            logger.info(f"Portfolio snapshots saved to {filepath}")
        except Exception as e:
            logger.error(f"Error saving portfolio snapshots to CSV: {e}")
