"""
Advanced backtesting features including Monte Carlo simulation, multi-timeframe analysis,
benchmark comparison, and performance optimization tools.
"""

import asyncio
import logging
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any, Union
from datetime import datetime, timedelta
from decimal import Decimal
import concurrent.futures
from dataclasses import dataclass
import multiprocessing as mp
from pathlib import Path

from .config import BacktestConfig
from .results import BacktestResults, TradeResult
from .engine import BacktestEngine
from .integration import IntegratedBacktestEngine, DatabaseDataProvider
from src.core.database import db_manager
from src.core.models import MarketData
from config.settings import settings

logger = logging.getLogger(__name__)


@dataclass
class MonteCarloResult:
    """Container for Monte Carlo simulation results."""
    scenario_id: int
    total_return: float
    sharpe_ratio: float
    max_drawdown: float
    final_portfolio_value: float
    total_trades: int
    win_rate: float
    volatility: float
    var_95: float
    cvar_95: float


@dataclass
class MonteCarloSummary:
    """Summary statistics from Monte Carlo simulation."""
    total_scenarios: int
    mean_return: float
    median_return: float
    std_return: float
    best_case_return: float
    worst_case_return: float
    probability_positive: float
    probability_target: float
    confidence_intervals: Dict[str, Tuple[float, float]]
    risk_metrics: Dict[str, float]
    scenario_results: List[MonteCarloResult]


@dataclass
class MultiTimeframeResult:
    """Results from multi-timeframe analysis."""
    timeframe: str
    backtest_results: BacktestResults
    timeframe_specific_metrics: Dict[str, float]


@dataclass
class BenchmarkComparison:
    """Comparison against benchmark performance."""
    benchmark_name: str
    strategy_return: float
    benchmark_return: float
    excess_return: float
    tracking_error: float
    information_ratio: float
    beta: float
    alpha: float
    correlation: float
    up_capture: float
    down_capture: float


class MonteCarloSimulator:
    """Monte Carlo simulation for backtesting robustness testing."""
    
    def __init__(self, base_config: BacktestConfig):
        self.base_config = base_config
        self.data_provider = DatabaseDataProvider()
        
    async def run_simulation(
        self,
        num_scenarios: int = 1000,
        randomization_methods: List[str] = None,
        confidence_levels: List[float] = None,
        parallel_workers: int = None
    ) -> MonteCarloSummary:
        """
        Run Monte Carlo simulation with multiple randomization methods.
        
        Args:
            num_scenarios: Number of simulation scenarios
            randomization_methods: List of randomization methods to apply
            confidence_levels: Confidence levels for intervals (default: [0.9, 0.95, 0.99])
            parallel_workers: Number of parallel workers (default: CPU count)
        """
        if randomization_methods is None:
            randomization_methods = ['bootstrap_trades', 'shuffle_returns', 'parameter_variation']
        
        if confidence_levels is None:
            confidence_levels = [0.9, 0.95, 0.99]
            
        if parallel_workers is None:
            parallel_workers = min(mp.cpu_count(), 8)
        
        logger.info(f"Starting Monte Carlo simulation with {num_scenarios} scenarios")
        logger.info(f"Randomization methods: {randomization_methods}")
        logger.info(f"Using {parallel_workers} parallel workers")
        
        # Generate scenario parameters
        scenario_configs = await self._generate_scenario_configs(
            num_scenarios, randomization_methods
        )
        
        # Run scenarios in parallel
        scenario_results = await self._run_scenarios_parallel(
            scenario_configs, parallel_workers
        )
        
        # Calculate summary statistics
        summary = self._calculate_summary_statistics(
            scenario_results, confidence_levels
        )
        
        logger.info(f"Monte Carlo simulation completed: {len(scenario_results)} scenarios")
        return summary
    
    async def _generate_scenario_configs(
        self,
        num_scenarios: int,
        randomization_methods: List[str]
    ) -> List[Dict[str, Any]]:
        """Generate configuration for each scenario."""
        configs = []
        
        for i in range(num_scenarios):
            scenario_config = {
                'scenario_id': i,
                'base_config': self.base_config,
                'randomization_seed': np.random.randint(0, 2**32),
                'methods': randomization_methods.copy()
            }
            
            # Add parameter variations
            if 'parameter_variation' in randomization_methods:
                scenario_config.update(await self._generate_parameter_variations())
            
            configs.append(scenario_config)
        
        return configs
    
    async def _generate_parameter_variations(self) -> Dict[str, Any]:
        """Generate random parameter variations."""
        variations = {}
        
        # Commission rate variation (±20%)
        base_commission = float(self.base_config.commission_rate)
        commission_multiplier = np.random.uniform(0.8, 1.2)
        variations['commission_rate'] = Decimal(str(base_commission * commission_multiplier))
        
        # Slippage variation (±50%)
        base_slippage = float(self.base_config.slippage_rate)
        slippage_multiplier = np.random.uniform(0.5, 1.5)
        variations['slippage_rate'] = Decimal(str(base_slippage * slippage_multiplier))
        
        # Stop loss variation (±30%)
        if hasattr(self.base_config, 'stop_loss_pct') and self.base_config.stop_loss_pct:
            base_stop_loss = float(self.base_config.stop_loss_pct)
            stop_loss_multiplier = np.random.uniform(0.7, 1.3)
            variations['stop_loss_pct'] = Decimal(str(base_stop_loss * stop_loss_multiplier))
        
        # Take profit variation (±30%)
        if hasattr(self.base_config, 'take_profit_pct') and self.base_config.take_profit_pct:
            base_take_profit = float(self.base_config.take_profit_pct)
            take_profit_multiplier = np.random.uniform(0.7, 1.3)
            variations['take_profit_pct'] = Decimal(str(base_take_profit * take_profit_multiplier))
        
        return variations
    
    async def _run_scenarios_parallel(
        self,
        scenario_configs: List[Dict[str, Any]],
        parallel_workers: int
    ) -> List[MonteCarloResult]:
        """Run scenarios in parallel."""
        results = []
        
        # Split scenarios into batches for parallel processing
        batch_size = max(1, len(scenario_configs) // parallel_workers)
        batches = [
            scenario_configs[i:i + batch_size]
            for i in range(0, len(scenario_configs), batch_size)
        ]
        
        # Process batches
        with concurrent.futures.ThreadPoolExecutor(max_workers=parallel_workers) as executor:
            future_to_batch = {
                executor.submit(self._run_scenario_batch, batch): batch
                for batch in batches
            }
            
            for future in concurrent.futures.as_completed(future_to_batch):
                try:
                    batch_results = future.result()
                    results.extend(batch_results)
                    
                    progress = len(results) / len(scenario_configs) * 100
                    if len(results) % 100 == 0:
                        logger.info(f"Monte Carlo progress: {progress:.1f}% ({len(results)}/{len(scenario_configs)})")
                        
                except Exception as e:
                    logger.error(f"Error in scenario batch: {e}")
                    continue
        
        return results
    
    def _run_scenario_batch(self, batch_configs: List[Dict[str, Any]]) -> List[MonteCarloResult]:
        """Run a batch of scenarios in a single thread."""
        batch_results = []
        
        for config in batch_configs:
            try:
                result = asyncio.run(self._run_single_scenario(config))
                if result:
                    batch_results.append(result)
            except Exception as e:
                logger.error(f"Error in scenario {config['scenario_id']}: {e}")
                continue
        
        return batch_results
    
    async def _run_single_scenario(self, scenario_config: Dict[str, Any]) -> Optional[MonteCarloResult]:
        """Run a single Monte Carlo scenario."""
        try:
            np.random.seed(scenario_config['randomization_seed'])
            
            # Create modified config for this scenario
            config = BacktestConfig(
                start_date=self.base_config.start_date,
                end_date=self.base_config.end_date,
                initial_capital=self.base_config.initial_capital,
                symbols=self.base_config.symbols,
                max_position_size=self.base_config.max_position_size,
                commission_rate=scenario_config.get('commission_rate', self.base_config.commission_rate),
                slippage_rate=scenario_config.get('slippage_rate', self.base_config.slippage_rate),
                stop_loss_pct=scenario_config.get('stop_loss_pct', getattr(self.base_config, 'stop_loss_pct', None)),
                take_profit_pct=scenario_config.get('take_profit_pct', getattr(self.base_config, 'take_profit_pct', None))
            )
            
            # Create engine and apply randomization
            engine = IntegratedBacktestEngine(config)
            
            # Apply randomization methods
            for method in scenario_config['methods']:
                await self._apply_randomization_method(engine, method, scenario_config['randomization_seed'])
            
            # Run backtest
            results = await engine.run_backtest()
            
            # Convert to Monte Carlo result
            return MonteCarloResult(
                scenario_id=scenario_config['scenario_id'],
                total_return=results.total_return,
                sharpe_ratio=results.sharpe_ratio,
                max_drawdown=results.max_drawdown,
                final_portfolio_value=float(results.final_portfolio_value),
                total_trades=results.total_trades,
                win_rate=results.win_rate,
                volatility=results.volatility,
                var_95=results.var_95,
                cvar_95=results.cvar_95
            )
            
        except Exception as e:
            logger.error(f"Error running scenario {scenario_config['scenario_id']}: {e}")
            return None
    
    async def _apply_randomization_method(
        self,
        engine: IntegratedBacktestEngine,
        method: str,
        seed: int
    ):
        """Apply a specific randomization method to the engine."""
        np.random.seed(seed)
        
        if method == 'bootstrap_trades':
            # Bootstrap historical trade sequences
            await self._bootstrap_trades(engine)
        elif method == 'shuffle_returns':
            # Shuffle return sequences while maintaining statistical properties
            await self._shuffle_returns(engine)
        elif method == 'parameter_variation':
            # Parameter variations are already applied in config
            pass
        else:
            logger.warning(f"Unknown randomization method: {method}")
    
    async def _bootstrap_trades(self, engine: IntegratedBacktestEngine):
        """Bootstrap historical trade sequences."""
        # This would involve resampling historical trades with replacement
        # Implementation depends on the specific approach needed
        pass
    
    async def _shuffle_returns(self, engine: IntegratedBacktestEngine):
        """Shuffle return sequences while maintaining statistical properties."""
        # This would involve shuffling daily returns while preserving correlations
        # Implementation depends on the specific approach needed
        pass
    
    def _calculate_summary_statistics(
        self,
        scenario_results: List[MonteCarloResult],
        confidence_levels: List[float]
    ) -> MonteCarloSummary:
        """Calculate summary statistics from scenario results."""
        if not scenario_results:
            raise ValueError("No scenario results to analyze")
        
        returns = [r.total_return for r in scenario_results]
        
        # Basic statistics
        mean_return = np.mean(returns)
        median_return = np.median(returns)
        std_return = np.std(returns)
        best_case = np.max(returns)
        worst_case = np.min(returns)
        
        # Probabilities
        prob_positive = np.mean([r > 0 for r in returns])
        prob_target = np.mean([r > 0.1 for r in returns])  # 10% target
        
        # Confidence intervals
        confidence_intervals = {}
        for level in confidence_levels:
            alpha = 1 - level
            lower = np.percentile(returns, alpha/2 * 100)
            upper = np.percentile(returns, (1 - alpha/2) * 100)
            confidence_intervals[f"{level:.0%}"] = (lower, upper)
        
        # Risk metrics
        var_95 = np.percentile(returns, 5)
        cvar_95 = np.mean([r for r in returns if r <= var_95])
        
        risk_metrics = {
            'var_95': var_95,
            'cvar_95': cvar_95,
            'skewness': stats.skew(returns),
            'kurtosis': stats.kurtosis(returns),
            'tail_ratio': np.percentile(returns, 95) / abs(np.percentile(returns, 5))
        }
        
        return MonteCarloSummary(
            total_scenarios=len(scenario_results),
            mean_return=mean_return,
            median_return=median_return,
            std_return=std_return,
            best_case_return=best_case,
            worst_case_return=worst_case,
            probability_positive=prob_positive,
            probability_target=prob_target,
            confidence_intervals=confidence_intervals,
            risk_metrics=risk_metrics,
            scenario_results=scenario_results
        )


class MultiTimeframeAnalyzer:
    """Analyzes strategy performance across multiple timeframes."""
    
    def __init__(self, base_config: BacktestConfig):
        self.base_config = base_config
        self.data_provider = DatabaseDataProvider()
    
    async def analyze_timeframes(
        self,
        timeframes: List[str] = None,
        alignment_method: str = 'resample'
    ) -> List[MultiTimeframeResult]:
        """
        Analyze strategy across multiple timeframes.
        
        Args:
            timeframes: List of timeframes to analyze (e.g., ['1h', '4h', '1d'])
            alignment_method: Method to align data across timeframes
        """
        if timeframes is None:
            timeframes = ['1h', '4h', '1d']
        
        logger.info(f"Starting multi-timeframe analysis: {timeframes}")
        
        results = []
        
        for timeframe in timeframes:
            try:
                logger.info(f"Analyzing {timeframe} timeframe...")
                
                # Create timeframe-specific config
                tf_config = await self._create_timeframe_config(timeframe)
                
                # Run backtest for this timeframe
                engine = IntegratedBacktestEngine(tf_config)
                backtest_results = await engine.run_backtest()
                
                # Calculate timeframe-specific metrics
                tf_metrics = await self._calculate_timeframe_metrics(
                    timeframe, backtest_results
                )
                
                results.append(MultiTimeframeResult(
                    timeframe=timeframe,
                    backtest_results=backtest_results,
                    timeframe_specific_metrics=tf_metrics
                ))
                
            except Exception as e:
                logger.error(f"Error analyzing {timeframe} timeframe: {e}")
                continue
        
        logger.info(f"Multi-timeframe analysis completed: {len(results)} timeframes")
        return results
    
    async def _create_timeframe_config(self, timeframe: str) -> BacktestConfig:
        """Create configuration adapted for specific timeframe."""
        # Adjust date range based on timeframe
        if timeframe == '1h':
            # Use original date range
            start_date = self.base_config.start_date
            end_date = self.base_config.end_date
        elif timeframe == '4h':
            # Extend date range for sufficient data points
            duration = self.base_config.end_date - self.base_config.start_date
            start_date = self.base_config.start_date - duration
            end_date = self.base_config.end_date
        elif timeframe == '1d':
            # Further extend for daily timeframe
            duration = self.base_config.end_date - self.base_config.start_date
            start_date = self.base_config.start_date - duration * 3
            end_date = self.base_config.end_date
        else:
            start_date = self.base_config.start_date
            end_date = self.base_config.end_date
        
        return BacktestConfig(
            start_date=start_date,
            end_date=end_date,
            initial_capital=self.base_config.initial_capital,
            symbols=self.base_config.symbols,
            max_position_size=self.base_config.max_position_size,
            commission_rate=self.base_config.commission_rate,
            slippage_rate=self.base_config.slippage_rate,
            stop_loss_pct=getattr(self.base_config, 'stop_loss_pct', None),
            take_profit_pct=getattr(self.base_config, 'take_profit_pct', None)
        )
    
    async def _calculate_timeframe_metrics(
        self,
        timeframe: str,
        results: BacktestResults
    ) -> Dict[str, float]:
        """Calculate metrics specific to timeframe."""
        metrics = {}
        
        # Trade frequency metrics
        total_days = (self.base_config.end_date - self.base_config.start_date).days
        
        if timeframe == '1h':
            expected_signals_per_day = 24
        elif timeframe == '4h':
            expected_signals_per_day = 6
        elif timeframe == '1d':
            expected_signals_per_day = 1
        else:
            expected_signals_per_day = 1
        
        metrics['trades_per_day'] = results.total_trades / max(total_days, 1)
        metrics['signal_frequency'] = metrics['trades_per_day'] / expected_signals_per_day
        
        # Timeframe-adjusted metrics
        if timeframe == '1h':
            metrics['holding_period_hours'] = 1.0  # Average holding period
        elif timeframe == '4h':
            metrics['holding_period_hours'] = 4.0
        elif timeframe == '1d':
            metrics['holding_period_hours'] = 24.0
        else:
            metrics['holding_period_hours'] = 1.0
        
        # Efficiency metrics
        metrics['return_per_trade'] = results.total_return / max(results.total_trades, 1)
        metrics['timeframe_efficiency'] = metrics['return_per_trade'] / metrics['holding_period_hours']
        
        return metrics


class BenchmarkComparator:
    """Compares strategy performance against benchmarks."""
    
    def __init__(self):
        self.data_provider = DatabaseDataProvider()
    
    async def compare_to_benchmarks(
        self,
        backtest_results: BacktestResults,
        benchmarks: List[str] = None
    ) -> List[BenchmarkComparison]:
        """
        Compare strategy results to benchmark performance.
        
        Args:
            backtest_results: Results from strategy backtest
            benchmarks: List of benchmark symbols (default: ['BTCUSDT', 'ETHUSDT'])
        """
        if benchmarks is None:
            benchmarks = ['BTCUSDT', 'ETHUSDT']  # Crypto market benchmarks
        
        logger.info(f"Comparing strategy to benchmarks: {benchmarks}")
        
        comparisons = []
        
        for benchmark in benchmarks:
            try:
                comparison = await self._compare_to_single_benchmark(
                    backtest_results, benchmark
                )
                if comparison:
                    comparisons.append(comparison)
                    
            except Exception as e:
                logger.error(f"Error comparing to benchmark {benchmark}: {e}")
                continue
        
        return comparisons
    
    async def _compare_to_single_benchmark(
        self,
        backtest_results: BacktestResults,
        benchmark_symbol: str
    ) -> Optional[BenchmarkComparison]:
        """Compare strategy to a single benchmark."""
        try:
            # Get benchmark data for the same period
            start_date = min(snapshot.timestamp for snapshot in backtest_results.portfolio_snapshots)
            end_date = max(snapshot.timestamp for snapshot in backtest_results.portfolio_snapshots)
            
            benchmark_data = await self.data_provider.get_historical_data(
                benchmark_symbol, start_date, end_date
            )
            
            if benchmark_data.empty:
                logger.warning(f"No benchmark data available for {benchmark_symbol}")
                return None
            
            # Calculate benchmark returns
            strategy_returns = self._calculate_strategy_returns(backtest_results)
            benchmark_returns = self._calculate_benchmark_returns(benchmark_data, strategy_returns.index)
            
            if len(strategy_returns) != len(benchmark_returns):
                logger.warning(f"Mismatched return series lengths for {benchmark_symbol}")
                return None
            
            # Calculate comparison metrics
            strategy_total_return = backtest_results.total_return
            benchmark_total_return = (benchmark_data['close'].iloc[-1] / benchmark_data['close'].iloc[0]) - 1
            
            excess_return = strategy_total_return - benchmark_total_return
            
            # Calculate additional metrics
            tracking_error = np.std(strategy_returns - benchmark_returns) * np.sqrt(252)
            information_ratio = excess_return / tracking_error if tracking_error > 0 else 0
            
            # Beta and Alpha
            covariance = np.cov(strategy_returns, benchmark_returns)[0, 1]
            benchmark_variance = np.var(benchmark_returns)
            beta = covariance / benchmark_variance if benchmark_variance > 0 else 0
            
            risk_free_rate = 0.02  # Assume 2% risk-free rate
            alpha = strategy_total_return - (risk_free_rate + beta * (benchmark_total_return - risk_free_rate))
            
            # Correlation
            correlation = np.corrcoef(strategy_returns, benchmark_returns)[0, 1]
            
            # Up/Down capture
            up_capture, down_capture = self._calculate_capture_ratios(
                strategy_returns, benchmark_returns
            )
            
            return BenchmarkComparison(
                benchmark_name=benchmark_symbol,
                strategy_return=strategy_total_return,
                benchmark_return=benchmark_total_return,
                excess_return=excess_return,
                tracking_error=tracking_error,
                information_ratio=information_ratio,
                beta=beta,
                alpha=alpha,
                correlation=correlation,
                up_capture=up_capture,
                down_capture=down_capture
            )
            
        except Exception as e:
            logger.error(f"Error in benchmark comparison for {benchmark_symbol}: {e}")
            return None
    
    def _calculate_strategy_returns(self, backtest_results: BacktestResults) -> pd.Series:
        """Calculate daily returns from portfolio snapshots."""
        snapshots = sorted(backtest_results.portfolio_snapshots, key=lambda x: x.timestamp)
        
        dates = [s.timestamp for s in snapshots]
        values = [float(s.total_value) for s in snapshots]
        
        portfolio_series = pd.Series(values, index=dates)
        return portfolio_series.pct_change().dropna()
    
    def _calculate_benchmark_returns(self, benchmark_data: pd.DataFrame, return_index: pd.Index) -> pd.Series:
        """Calculate benchmark returns aligned with strategy returns."""
        # Resample benchmark data to match strategy return frequency
        benchmark_close = benchmark_data['close'].resample('D').last()
        benchmark_returns = benchmark_close.pct_change().dropna()
        
        # Align with strategy returns
        aligned_returns = benchmark_returns.reindex(return_index, method='ffill').fillna(0)
        return aligned_returns
    
    def _calculate_capture_ratios(
        self,
        strategy_returns: pd.Series,
        benchmark_returns: pd.Series
    ) -> Tuple[float, float]:
        """Calculate up and down capture ratios."""
        # Up capture: strategy performance when benchmark is positive
        up_periods = benchmark_returns > 0
        if up_periods.sum() > 0:
            up_capture = strategy_returns[up_periods].mean() / benchmark_returns[up_periods].mean()
        else:
            up_capture = 0.0
        
        # Down capture: strategy performance when benchmark is negative
        down_periods = benchmark_returns < 0
        if down_periods.sum() > 0:
            down_capture = strategy_returns[down_periods].mean() / benchmark_returns[down_periods].mean()
        else:
            down_capture = 0.0
        
        return up_capture, down_capture


class PerformanceOptimizer:
    """Tools for optimizing backtesting performance."""
    
    def __init__(self):
        self.cache_dir = Path("backtest_cache")
        self.cache_dir.mkdir(exist_ok=True)
    
    async def optimize_data_loading(
        self,
        symbols: List[str],
        start_date: datetime,
        end_date: datetime,
        use_cache: bool = True
    ) -> Dict[str, pd.DataFrame]:
        """Optimized data loading with caching and parallel processing."""
        cache_key = f"{'-'.join(symbols)}_{start_date.date()}_{end_date.date()}"
        cache_file = self.cache_dir / f"{cache_key}.pkl"
        
        # Check cache first
        if use_cache and cache_file.exists():
            try:
                return pd.read_pickle(cache_file)
            except Exception as e:
                logger.warning(f"Cache load failed: {e}")
        
        # Load data in parallel
        data_provider = DatabaseDataProvider()
        
        async def load_symbol_data(symbol: str) -> Tuple[str, pd.DataFrame]:
            df = await data_provider.get_historical_data(symbol, start_date, end_date)
            return symbol, df
        
        # Use asyncio.gather for parallel loading
        tasks = [load_symbol_data(symbol) for symbol in symbols]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Process results
        data_cache = {}
        for result in results:
            if isinstance(result, Exception):
                logger.error(f"Error loading symbol data: {result}")
                continue
            
            symbol, df = result
            if not df.empty:
                data_cache[symbol] = df
        
        # Cache results
        if use_cache and data_cache:
            try:
                pd.to_pickle(data_cache, cache_file)
            except Exception as e:
                logger.warning(f"Cache save failed: {e}")
        
        return data_cache
    
    def clear_cache(self):
        """Clear all cached data."""
        for cache_file in self.cache_dir.glob("*.pkl"):
            try:
                cache_file.unlink()
            except Exception as e:
                logger.warning(f"Failed to delete cache file {cache_file}: {e}")
        
        logger.info("Backtesting cache cleared")


# Global instances
monte_carlo_simulator = None
multi_timeframe_analyzer = None
benchmark_comparator = BenchmarkComparator()
performance_optimizer = PerformanceOptimizer()


def create_monte_carlo_simulator(config: BacktestConfig) -> MonteCarloSimulator:
    """Create Monte Carlo simulator instance."""
    return MonteCarloSimulator(config)


def create_multi_timeframe_analyzer(config: BacktestConfig) -> MultiTimeframeAnalyzer:
    """Create multi-timeframe analyzer instance."""
    return MultiTimeframeAnalyzer(config)
