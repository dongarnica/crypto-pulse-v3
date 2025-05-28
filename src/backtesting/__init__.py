"""
Backtesting module for Crypto Pulse V3 trading system.
Provides comprehensive historical strategy validation and optimization.
"""

from .engine import BacktestEngine, BacktestPortfolio
from .config import BacktestConfig, OptimizationConfig
from .results import BacktestResults, BacktestReport, TradeResult, PortfolioSnapshot
from .optimization import ParameterOptimizer, WalkForwardOptimizer, OptimizationResults
from .reports import BacktestReportGenerator
from .integration import (
    DatabaseDataProvider, 
    LiveStrategyIntegration, 
    IntegratedBacktestEngine,
    BacktestValidator,
    database_data_provider,
    live_strategy_integration,
    backtest_validator
)
from .advanced_features import (
    MonteCarloSimulator,
    MonteCarloResult,
    MonteCarloSummary,
    MultiTimeframeAnalyzer,
    MultiTimeframeResult,
    BenchmarkComparator,
    BenchmarkComparison,
    PerformanceOptimizer,
    create_monte_carlo_simulator,
    create_multi_timeframe_analyzer,
    benchmark_comparator,
    performance_optimizer
)

__all__ = [
    'BacktestEngine', 
    'BacktestPortfolio',
    'BacktestConfig', 
    'OptimizationConfig',
    'BacktestResults', 
    'BacktestReport', 
    'TradeResult', 
    'PortfolioSnapshot',
    'ParameterOptimizer',
    'WalkForwardOptimizer',
    'OptimizationResults',
    'BacktestReportGenerator',
    'DatabaseDataProvider',
    'LiveStrategyIntegration',
    'IntegratedBacktestEngine',
    'BacktestValidator',
    'database_data_provider',
    'live_strategy_integration',
    'backtest_validator',
    # Advanced features
    'MonteCarloSimulator',
    'MonteCarloResult',
    'MonteCarloSummary',
    'MultiTimeframeAnalyzer',
    'MultiTimeframeResult',
    'BenchmarkComparator',
    'BenchmarkComparison',
    'PerformanceOptimizer',
    'create_monte_carlo_simulator',
    'create_multi_timeframe_analyzer',
    'benchmark_comparator',
    'performance_optimizer'
]
