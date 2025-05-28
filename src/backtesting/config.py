"""
Backtesting configuration and parameter management.
"""

from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
from decimal import Decimal


@dataclass
class BacktestConfig:
    """Configuration for backtesting parameters."""
    
    # Time period
    start_date: datetime
    end_date: datetime
    
    # Portfolio settings
    initial_capital: Decimal = Decimal('100000')
    
    # Trading symbols
    symbols: List[str] = field(default_factory=lambda: [
        'BTC/USD', 'ETH/USD', 'ADA/USD', 'DOT/USD', 'LINK/USD',
        'UNI/USD', 'AAVE/USD', 'SUSHI/USD', 'COMP/USD', 'MKR/USD'
    ])
    
    # Trading parameters
    max_portfolio_allocation: float = 0.15
    min_portfolio_allocation: float = 0.08
    max_drawdown_threshold: float = 0.15
    atr_stop_multiplier: float = 3.5
    max_correlation_threshold: float = 0.7
    
    # Execution simulation
    commission_rate: float = 0.001  # 0.1% per trade
    slippage_bps: int = 5  # 5 basis points slippage
    market_impact_threshold: Decimal = Decimal('10000')  # $10k order size threshold
    
    # Analysis settings
    analysis_interval_minutes: int = 30
    sentiment_interval_hours: int = 2
    rebalance_threshold: float = 0.05  # 5% drift before rebalancing
    
    # ML model settings
    model_retrain_days: int = 30
    feature_lookback_days: int = 90
    min_prediction_confidence: float = 0.65
    
    # Walk-forward optimization
    walk_forward_enabled: bool = False
    walk_forward_period_days: int = 90
    walk_forward_step_days: int = 30
    
    # Performance tracking
    benchmark_symbol: str = 'BTC/USD'
    risk_free_rate: float = 0.02
    
    # Output settings
    generate_detailed_logs: bool = True
    save_trade_history: bool = True
    save_portfolio_snapshots: bool = True
    output_format: str = 'html'  # 'html', 'pdf', 'json'


@dataclass
class OptimizationConfig:
    """Configuration for parameter optimization."""
    
    # Parameters to optimize
    optimize_params: Dict[str, Dict[str, Any]] = field(default_factory=lambda: {
        'max_portfolio_allocation': {
            'min': 0.08,
            'max': 0.25,
            'step': 0.02
        },
        'atr_stop_multiplier': {
            'min': 2.0,
            'max': 5.0,
            'step': 0.5
        },
        'min_prediction_confidence': {
            'min': 0.55,
            'max': 0.80,
            'step': 0.05
        }
    })
    
    # Optimization method
    optimization_method: str = 'grid_search'  # 'grid_search', 'random_search', 'bayesian'
    max_iterations: int = 100
    
    # Evaluation criteria
    primary_metric: str = 'sharpe_ratio'
    secondary_metrics: List[str] = field(default_factory=lambda: [
        'total_return', 'max_drawdown', 'win_rate', 'profit_factor'
    ])
    
    # Cross-validation
    cross_validation_folds: int = 5
    validation_method: str = 'time_series_split'


def create_default_config(
    start_date: Optional[datetime] = None,
    end_date: Optional[datetime] = None,
    initial_capital: Optional[Decimal] = None
) -> BacktestConfig:
    """Create a default backtesting configuration."""
    
    if end_date is None:
        end_date = datetime.utcnow()
    
    if start_date is None:
        start_date = end_date - timedelta(days=365)  # 1 year backtest
    
    if initial_capital is None:
        initial_capital = Decimal('100000')
    
    return BacktestConfig(
        start_date=start_date,
        end_date=end_date,
        initial_capital=initial_capital
    )


def create_optimization_config(
    params_to_optimize: Optional[List[str]] = None
) -> OptimizationConfig:
    """Create a default optimization configuration."""
    
    config = OptimizationConfig()
    
    if params_to_optimize:
        # Filter to only optimize specified parameters
        filtered_params = {
            param: config.optimize_params[param] 
            for param in params_to_optimize 
            if param in config.optimize_params
        }
        config.optimize_params = filtered_params
    
    return config
