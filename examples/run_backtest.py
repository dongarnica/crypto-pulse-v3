"""
Example script demonstrating how to run backtests with the Crypto Pulse V3 system.
"""

import asyncio
import logging
from datetime import datetime, timedelta
from decimal import Decimal
import sys
import os

# Add the src directory to the path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from backtesting.config import BacktestConfig, OptimizationConfig
from backtesting.engine import BacktestEngine
from backtesting.optimization import ParameterOptimizer, WalkForwardOptimizer
from backtesting.reports import BacktestReportGenerator

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


async def run_simple_backtest():
    """Run a simple backtest example."""
    logger.info("=== Running Simple Backtest ===")
    
    # Create backtest configuration
    config = BacktestConfig(
        start_date=datetime(2023, 1, 1),
        end_date=datetime(2023, 6, 30),
        initial_capital=Decimal('100000'),
        symbols=['BTC/USD', 'ETH/USD', 'ADA/USD'],
        max_portfolio_allocation=0.20,
        commission_rate=0.001,
        analysis_interval_minutes=60
    )
    
    # Run backtest
    engine = BacktestEngine(config)
    results = await engine.run_backtest()
    
    # Generate reports
    report_generator = BacktestReportGenerator(results)
    report_files = report_generator.generate_full_report()
    
    # Print summary
    logger.info("=== Backtest Results Summary ===")
    logger.info(f"Initial Capital: ${results.initial_capital:,.2f}")
    logger.info(f"Final Capital: ${results.final_capital:,.2f}")
    logger.info(f"Total Return: {(results.total_return or 0)*100:.2f}%")
    logger.info(f"Sharpe Ratio: {results.sharpe_ratio or 0:.3f}")
    logger.info(f"Max Drawdown: {(results.max_drawdown or 0)*100:.2f}%")
    logger.info(f"Win Rate: {(results.win_rate or 0)*100:.1f}%")
    logger.info(f"Total Trades: {len([t for t in results.trades if t.pnl is not None])}")
    
    for report_type, file_path in report_files.items():
        logger.info(f"{report_type}: {file_path}")
    
    return results


async def run_parameter_optimization():
    """Run parameter optimization example."""
    logger.info("=== Running Parameter Optimization ===")
    
    # Base configuration
    base_config = BacktestConfig(
        start_date=datetime(2023, 1, 1),
        end_date=datetime(2023, 3, 31),  # Shorter period for optimization
        initial_capital=Decimal('100000'),
        symbols=['BTC/USD', 'ETH/USD'],
        analysis_interval_minutes=60
    )
    
    # Optimization configuration
    optimization_config = OptimizationConfig(
        method='random_search',
        objective='sharpe_ratio',
        max_iterations=20,  # Limited for example
        parameter_ranges={
            'max_portfolio_allocation': [0.10, 0.15, 0.20, 0.25],
            'commission_rate': [0.0005, 0.001, 0.0015],
            'atr_stop_multiplier': [2.0, 3.0, 4.0],
            'min_prediction_confidence': [0.60, 0.65, 0.70]
        },
        max_parallel_jobs=2
    )
    
    # Run optimization
    optimizer = ParameterOptimizer(base_config, optimization_config)
    optimization_results = await optimizer.optimize()
    
    # Print results
    logger.info("=== Optimization Results ===")
    logger.info(f"Best Score: {optimization_results.best_score:.4f}")
    logger.info(f"Best Parameters: {optimization_results.best_parameters}")
    logger.info(f"Total Backtests Run: {optimization_results.total_backtests_run}")
    logger.info(f"Optimization Time: {optimization_results.optimization_time_seconds:.2f} seconds")
    
    return optimization_results


async def run_walk_forward_analysis():
    """Run walk-forward analysis example."""
    logger.info("=== Running Walk-Forward Analysis ===")
    
    # Base configuration
    base_config = BacktestConfig(
        start_date=datetime(2023, 1, 1),
        end_date=datetime(2023, 12, 31),
        initial_capital=Decimal('100000'),
        symbols=['BTC/USD', 'ETH/USD'],
        analysis_interval_minutes=60
    )
    
    # Walk-forward configuration
    optimization_config = OptimizationConfig(
        method='random_search',
        objective='sharpe_ratio',
        max_iterations=10,  # Limited for example
        parameter_ranges={
            'max_portfolio_allocation': [0.15, 0.20, 0.25],
            'commission_rate': [0.001, 0.0015]
        },
        train_period_days=90,
        test_period_days=30,
        step_days=30,
        max_parallel_jobs=2
    )
    
    # Run walk-forward optimization
    wf_optimizer = WalkForwardOptimizer(base_config, optimization_config)
    wf_results = await wf_optimizer.run_walk_forward_optimization()
    
    # Print results
    logger.info("=== Walk-Forward Results ===")
    logger.info(f"Periods Completed: {wf_results['periods_completed']}")
    logger.info(f"Average OOS Return: {wf_results['average_oos_return']:.4f}")
    logger.info(f"OOS Sharpe Ratio: {wf_results['oos_sharpe_ratio']:.4f}")
    logger.info(f"Cumulative OOS Return: {wf_results['cumulative_oos_return']:.4f}")
    logger.info(f"Win Rate: {wf_results['win_rate']*100:.1f}%")
    
    return wf_results


async def run_comprehensive_analysis():
    """Run a comprehensive backtesting analysis."""
    logger.info("=== Running Comprehensive Analysis ===")
    
    # Configuration for longer backtest
    config = BacktestConfig(
        start_date=datetime(2022, 1, 1),
        end_date=datetime(2023, 12, 31),
        initial_capital=Decimal('500000'),
        symbols=[
            'BTC/USD', 'ETH/USD', 'ADA/USD', 'DOT/USD', 'LINK/USD',
            'UNI/USD', 'AAVE/USD', 'SUSHI/USD'
        ],
        max_portfolio_allocation=0.15,
        min_portfolio_allocation=0.08,
        commission_rate=0.001,
        slippage_bps=5,
        analysis_interval_minutes=30,
        sentiment_interval_hours=2,
        model_retrain_days=30,
        min_prediction_confidence=0.65
    )
    
    # Run backtest
    logger.info("Running comprehensive backtest...")
    engine = BacktestEngine(config)
    results = await engine.run_backtest()
    
    # Generate detailed reports
    logger.info("Generating comprehensive reports...")
    report_generator = BacktestReportGenerator(results)
    report_files = report_generator.generate_full_report()
    
    # Print detailed summary
    logger.info("=== Comprehensive Analysis Results ===")
    logger.info(f"Period: {config.start_date.date()} to {config.end_date.date()}")
    logger.info(f"Initial Capital: ${results.initial_capital:,.2f}")
    logger.info(f"Final Capital: ${results.final_capital:,.2f}")
    logger.info(f"Absolute Return: ${results.final_capital - results.initial_capital:,.2f}")
    logger.info(f"Total Return: {(results.total_return or 0)*100:.2f}%")
    logger.info(f"Annualized Return: {(results.annualized_return or 0)*100:.2f}%")
    logger.info("")
    logger.info("Risk Metrics:")
    logger.info(f"  Sharpe Ratio: {results.sharpe_ratio or 0:.3f}")
    logger.info(f"  Sortino Ratio: {results.sortino_ratio or 0:.3f}")
    logger.info(f"  Calmar Ratio: {results.calmar_ratio or 0:.3f}")
    logger.info(f"  Maximum Drawdown: {(results.max_drawdown or 0)*100:.2f}%")
    logger.info("")
    logger.info("Trading Statistics:")
    logger.info(f"  Total Trades: {len([t for t in results.trades if t.pnl is not None])}")
    logger.info(f"  Win Rate: {(results.win_rate or 0)*100:.1f}%")
    logger.info(f"  Profit Factor: {results.profit_factor or 0:.2f}")
    logger.info(f"  Average Win: ${results.average_win or 0:.2f}")
    logger.info(f"  Average Loss: ${results.average_loss or 0:.2f}")
    logger.info(f"  Max Consecutive Wins: {results.max_consecutive_wins or 0}")
    logger.info(f"  Max Consecutive Losses: {results.max_consecutive_losses or 0}")
    logger.info("")
    logger.info("Generated Reports:")
    for report_type, file_path in report_files.items():
        logger.info(f"  {report_type}: {file_path}")
    
    return results


async def main():
    """Main function to run backtest examples."""
    logger.info("Starting Crypto Pulse V3 Backtesting Examples")
    
    try:
        # Example 1: Simple backtest
        await run_simple_backtest()
        
        print("\n" + "="*80 + "\n")
        
        # Example 2: Parameter optimization
        await run_parameter_optimization()
        
        print("\n" + "="*80 + "\n")
        
        # Example 3: Walk-forward analysis
        await run_walk_forward_analysis()
        
        print("\n" + "="*80 + "\n")
        
        # Example 4: Comprehensive analysis
        await run_comprehensive_analysis()
        
        logger.info("All backtesting examples completed successfully!")
        
    except Exception as e:
        logger.error(f"Error running backtesting examples: {e}")
        raise


if __name__ == "__main__":
    asyncio.run(main())
