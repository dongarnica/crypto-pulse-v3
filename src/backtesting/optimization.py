"""
Parameter optimization and walk-forward analysis for backtesting.
"""

import asyncio
import logging
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any, Union
from decimal import Decimal
import itertools
from concurrent.futures import ThreadPoolExecutor, as_completed
import json
from dataclasses import dataclass, asdict
import random

from .config import BacktestConfig, OptimizationConfig
from .engine import BacktestEngine
from .results import BacktestResults

logger = logging.getLogger(__name__)


@dataclass
class ParameterSet:
    """A set of parameters for optimization."""
    parameters: Dict[str, Any]
    score: Optional[float] = None
    results: Optional[BacktestResults] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            'parameters': self.parameters,
            'score': self.score,
            'results_summary': self.results.to_dict() if self.results else None
        }


@dataclass
class OptimizationResults:
    """Results from parameter optimization."""
    best_parameters: Dict[str, Any]
    best_score: float
    all_results: List[ParameterSet]
    optimization_time_seconds: float
    total_backtests_run: int
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            'best_parameters': self.best_parameters,
            'best_score': self.best_score,
            'total_backtests_run': self.total_backtests_run,
            'optimization_time_seconds': self.optimization_time_seconds,
            'top_10_results': [result.to_dict() for result in 
                             sorted(self.all_results, key=lambda x: x.score or -np.inf, reverse=True)[:10]]
        }


class ParameterOptimizer:
    """Parameter optimization for backtesting strategies."""
    
    def __init__(self, base_config: BacktestConfig, optimization_config: OptimizationConfig):
        self.base_config = base_config
        self.optimization_config = optimization_config
        self.executor = ThreadPoolExecutor(max_workers=optimization_config.max_parallel_jobs)
        
    def generate_parameter_combinations(self) -> List[Dict[str, Any]]:
        """Generate parameter combinations for optimization."""
        if self.optimization_config.method == 'grid_search':
            return self._generate_grid_search_combinations()
        elif self.optimization_config.method == 'random_search':
            return self._generate_random_search_combinations()
        elif self.optimization_config.method == 'bayesian':
            return self._generate_bayesian_combinations()
        else:
            raise ValueError(f"Unknown optimization method: {self.optimization_config.method}")
    
    def _generate_grid_search_combinations(self) -> List[Dict[str, Any]]:
        """Generate all combinations for grid search."""
        param_names = list(self.optimization_config.parameter_ranges.keys())
        param_values = [self.optimization_config.parameter_ranges[name] for name in param_names]
        
        combinations = []
        for combo in itertools.product(*param_values):
            param_dict = dict(zip(param_names, combo))
            combinations.append(param_dict)
        
        logger.info(f"Generated {len(combinations)} parameter combinations for grid search")
        return combinations
    
    def _generate_random_search_combinations(self) -> List[Dict[str, Any]]:
        """Generate random combinations for random search."""
        combinations = []
        max_iterations = min(
            self.optimization_config.max_iterations,
            # Calculate total possible combinations
            np.prod([len(values) for values in self.optimization_config.parameter_ranges.values()])
        )
        
        for _ in range(max_iterations):
            param_dict = {}
            for param_name, param_values in self.optimization_config.parameter_ranges.items():
                param_dict[param_name] = random.choice(param_values)
            combinations.append(param_dict)
        
        logger.info(f"Generated {len(combinations)} parameter combinations for random search")
        return combinations
    
    def _generate_bayesian_combinations(self) -> List[Dict[str, Any]]:
        """Generate combinations using Bayesian optimization (simplified)."""
        # For now, implement a simplified version that starts with random search
        # and then focuses on promising regions
        combinations = self._generate_random_search_combinations()
        
        # TODO: Implement proper Bayesian optimization with acquisition functions
        # This would require additional dependencies like scikit-optimize
        
        logger.info(f"Generated {len(combinations)} parameter combinations for Bayesian optimization")
        return combinations
    
    def create_config_with_parameters(self, parameters: Dict[str, Any]) -> BacktestConfig:
        """Create a backtest config with specific parameters."""
        config_dict = asdict(self.base_config)
        
        # Update configuration with optimization parameters
        for param_name, param_value in parameters.items():
            if hasattr(self.base_config, param_name):
                config_dict[param_name] = param_value
            else:
                logger.warning(f"Unknown parameter: {param_name}")
        
        return BacktestConfig(**config_dict)
    
    def calculate_objective_score(self, results: BacktestResults) -> float:
        """Calculate objective score for optimization."""
        if self.optimization_config.objective == 'sharpe_ratio':
            return results.sharpe_ratio or -np.inf
        elif self.optimization_config.objective == 'total_return':
            return results.total_return or -np.inf
        elif self.optimization_config.objective == 'sortino_ratio':
            return results.sortino_ratio or -np.inf
        elif self.optimization_config.objective == 'calmar_ratio':
            return results.calmar_ratio or -np.inf
        elif self.optimization_config.objective == 'profit_factor':
            return results.profit_factor or -np.inf
        else:
            # Custom weighted score
            score = 0.0
            if results.sharpe_ratio:
                score += results.sharpe_ratio * 0.3
            if results.total_return:
                score += results.total_return * 0.3
            if results.max_drawdown:
                score -= abs(results.max_drawdown) * 0.2  # Penalty for large drawdowns
            if results.win_rate:
                score += results.win_rate * 0.2
            
            return score
    
    async def run_single_backtest(self, parameters: Dict[str, Any]) -> ParameterSet:
        """Run a single backtest with given parameters."""
        try:
            config = self.create_config_with_parameters(parameters)
            engine = BacktestEngine(config)
            results = await engine.run_backtest()
            score = self.calculate_objective_score(results)
            
            return ParameterSet(
                parameters=parameters,
                score=score,
                results=results
            )
            
        except Exception as e:
            logger.error(f"Error running backtest with parameters {parameters}: {e}")
            return ParameterSet(
                parameters=parameters,
                score=-np.inf,
                results=None
            )
    
    async def optimize(self) -> OptimizationResults:
        """Run parameter optimization."""
        start_time = datetime.now()
        logger.info("Starting parameter optimization...")
        
        # Generate parameter combinations
        parameter_combinations = self.generate_parameter_combinations()
        
        # Limit the number of combinations if needed
        if len(parameter_combinations) > self.optimization_config.max_iterations:
            logger.info(f"Limiting combinations to {self.optimization_config.max_iterations}")
            parameter_combinations = parameter_combinations[:self.optimization_config.max_iterations]
        
        # Run backtests
        all_results = []
        
        # Process in batches to avoid overwhelming the system
        batch_size = self.optimization_config.max_parallel_jobs
        
        for i in range(0, len(parameter_combinations), batch_size):
            batch = parameter_combinations[i:i + batch_size]
            
            # Run batch of backtests
            tasks = [self.run_single_backtest(params) for params in batch]
            batch_results = await asyncio.gather(*tasks, return_exceptions=True)
            
            # Filter out exceptions and add to results
            for result in batch_results:
                if isinstance(result, ParameterSet):
                    all_results.append(result)
                else:
                    logger.error(f"Backtest failed with exception: {result}")
            
            # Progress logging
            completed = min(i + batch_size, len(parameter_combinations))
            logger.info(f"Completed {completed}/{len(parameter_combinations)} backtests")
        
        # Find best parameters
        valid_results = [r for r in all_results if r.score is not None and r.score > -np.inf]
        
        if not valid_results:
            raise ValueError("No valid backtest results obtained")
        
        best_result = max(valid_results, key=lambda x: x.score)
        
        end_time = datetime.now()
        optimization_time = (end_time - start_time).total_seconds()
        
        results = OptimizationResults(
            best_parameters=best_result.parameters,
            best_score=best_result.score,
            all_results=all_results,
            optimization_time_seconds=optimization_time,
            total_backtests_run=len(valid_results)
        )
        
        logger.info(f"Optimization completed in {optimization_time:.2f} seconds. "
                   f"Best score: {best_result.score:.4f}")
        
        return results


class WalkForwardOptimizer:
    """Walk-forward optimization for time series validation."""
    
    def __init__(self, base_config: BacktestConfig, optimization_config: OptimizationConfig):
        self.base_config = base_config
        self.optimization_config = optimization_config
        self.parameter_optimizer = ParameterOptimizer(base_config, optimization_config)
        
    def generate_walk_forward_periods(self) -> List[Tuple[datetime, datetime, datetime, datetime]]:
        """Generate walk-forward optimization periods."""
        periods = []
        
        total_days = (self.base_config.end_date - self.base_config.start_date).days
        train_days = self.optimization_config.train_period_days
        test_days = self.optimization_config.test_period_days
        
        current_start = self.base_config.start_date
        
        while current_start + timedelta(days=train_days + test_days) <= self.base_config.end_date:
            train_start = current_start
            train_end = train_start + timedelta(days=train_days)
            test_start = train_end
            test_end = test_start + timedelta(days=test_days)
            
            periods.append((train_start, train_end, test_start, test_end))
            
            # Move forward by step size
            current_start += timedelta(days=self.optimization_config.step_days)
        
        logger.info(f"Generated {len(periods)} walk-forward periods")
        return periods
    
    async def run_walk_forward_optimization(self) -> Dict[str, Any]:
        """Run walk-forward optimization."""
        logger.info("Starting walk-forward optimization...")
        
        periods = self.generate_walk_forward_periods()
        
        if not periods:
            raise ValueError("No valid walk-forward periods generated")
        
        all_results = []
        cumulative_returns = []
        
        for i, (train_start, train_end, test_start, test_end) in enumerate(periods):
            logger.info(f"Walk-forward period {i+1}/{len(periods)}: "
                       f"Train: {train_start.date()} to {train_end.date()}, "
                       f"Test: {test_start.date()} to {test_end.date()}")
            
            try:
                # Create training config
                train_config = BacktestConfig(
                    start_date=train_start,
                    end_date=train_end,
                    initial_capital=self.base_config.initial_capital,
                    symbols=self.base_config.symbols,
                    # Copy other parameters
                    **{k: v for k, v in asdict(self.base_config).items() 
                       if k not in ['start_date', 'end_date']}
                )
                
                # Optimize on training period
                train_optimizer = ParameterOptimizer(train_config, self.optimization_config)
                train_optimization = await train_optimizer.optimize()
                
                # Test on out-of-sample period
                test_config = BacktestConfig(
                    start_date=test_start,
                    end_date=test_end,
                    initial_capital=self.base_config.initial_capital,
                    symbols=self.base_config.symbols,
                    **{k: v for k, v in asdict(self.base_config).items() 
                       if k not in ['start_date', 'end_date']}
                )
                
                # Apply best parameters from training
                test_config_dict = asdict(test_config)
                test_config_dict.update(train_optimization.best_parameters)
                final_test_config = BacktestConfig(**test_config_dict)
                
                # Run test backtest
                test_engine = BacktestEngine(final_test_config)
                test_results = await test_engine.run_backtest()
                
                period_result = {
                    'period': i + 1,
                    'train_start': train_start,
                    'train_end': train_end,
                    'test_start': test_start,
                    'test_end': test_end,
                    'best_train_score': train_optimization.best_score,
                    'best_parameters': train_optimization.best_parameters,
                    'test_results': test_results.to_dict(),
                    'test_return': test_results.total_return,
                    'test_sharpe': test_results.sharpe_ratio,
                    'test_max_drawdown': test_results.max_drawdown
                }
                
                all_results.append(period_result)
                cumulative_returns.append(test_results.total_return or 0.0)
                
            except Exception as e:
                logger.error(f"Error in walk-forward period {i+1}: {e}")
                continue
        
        # Calculate overall statistics
        avg_return = np.mean(cumulative_returns) if cumulative_returns else 0.0
        std_return = np.std(cumulative_returns) if cumulative_returns else 0.0
        sharpe_oos = avg_return / std_return if std_return > 0 else 0.0
        
        summary = {
            'periods_completed': len(all_results),
            'average_oos_return': avg_return,
            'oos_return_std': std_return,
            'oos_sharpe_ratio': sharpe_oos,
            'cumulative_oos_return': np.sum(cumulative_returns),
            'win_rate': sum(1 for r in cumulative_returns if r > 0) / len(cumulative_returns) if cumulative_returns else 0.0,
            'all_period_results': all_results
        }
        
        logger.info(f"Walk-forward optimization completed. "
                   f"Average OOS return: {avg_return:.4f}, "
                   f"OOS Sharpe: {sharpe_oos:.4f}")
        
        return summary
