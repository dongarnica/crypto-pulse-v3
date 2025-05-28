#!/usr/bin/env python3
"""
Advanced Backtesting CLI - Comprehensive tool for advanced backtesting features
including Monte Carlo simulation, multi-timeframe analysis, and benchmark comparison.
"""

import asyncio
import argparse
import sys
import logging
from pathlib import Path
from datetime import datetime, timedelta
from decimal import Decimal

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from src.backtesting import (
    BacktestConfig,
    create_monte_carlo_simulator,
    create_multi_timeframe_analyzer,
    benchmark_comparator,
    performance_optimizer,
    IntegratedBacktestEngine,
    BacktestReportGenerator
)
from src.core.database import db_manager
from config.settings import settings

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


async def check_database_connection():
    """Check database connectivity."""
    try:
        await db_manager.initialize()
        return True
    except Exception as e:
        logger.error(f"Database connection failed: {e}")
        return False


async def run_monte_carlo(args):
    """Run Monte Carlo simulation."""
    logger.info("Starting Monte Carlo Simulation")
    
    if not await check_database_connection():
        logger.error("Cannot proceed without database connection")
        return False
    
    # Parse dates
    if args.start_date:
        start_date = datetime.strptime(args.start_date, '%Y-%m-%d')
    else:
        start_date = datetime.utcnow() - timedelta(days=args.days)
    
    if args.end_date:
        end_date = datetime.strptime(args.end_date, '%Y-%m-%d')
    else:
        end_date = datetime.utcnow() - timedelta(days=1)
    
    # Create base configuration
    config = BacktestConfig(
        start_date=start_date,
        end_date=end_date,
        initial_capital=Decimal(str(args.capital)),
        symbols=args.symbols.split(',') if args.symbols else settings.trading.trading_pairs[:args.max_symbols],
        max_position_size=Decimal(str(args.max_position_size)),
        commission_rate=Decimal(str(args.commission_rate)),
        slippage_rate=Decimal(str(args.slippage_rate))
    )
    
    logger.info(f"Monte Carlo Configuration:")
    logger.info(f"  Scenarios: {args.scenarios}")
    logger.info(f"  Period: {start_date} to {end_date}")
    logger.info(f"  Symbols: {len(config.symbols)}")
    logger.info(f"  Randomization: {args.randomization_methods}")
    
    try:
        # Create and run Monte Carlo simulation
        simulator = create_monte_carlo_simulator(config)
        
        randomization_methods = args.randomization_methods.split(',') if args.randomization_methods else None
        
        summary = await simulator.run_simulation(
            num_scenarios=args.scenarios,
            randomization_methods=randomization_methods,
            parallel_workers=args.workers
        )
        
        # Display results
        print("\n" + "="*60)
        print("MONTE CARLO SIMULATION RESULTS")
        print("="*60)
        
        print(f"Total Scenarios: {summary.total_scenarios}")
        print(f"Mean Return: {summary.mean_return:.2%}")
        print(f"Median Return: {summary.median_return:.2%}")
        print(f"Standard Deviation: {summary.std_return:.2%}")
        print(f"Best Case: {summary.best_case_return:.2%}")
        print(f"Worst Case: {summary.worst_case_return:.2%}")
        
        print(f"\nProbabilities:")
        print(f"  Positive Return: {summary.probability_positive:.2%}")
        print(f"  Target Return (10%): {summary.probability_target:.2%}")
        
        print(f"\nConfidence Intervals:")
        for level, (lower, upper) in summary.confidence_intervals.items():
            print(f"  {level}: [{lower:.2%}, {upper:.2%}]")
        
        print(f"\nRisk Metrics:")
        print(f"  VaR (95%): {summary.risk_metrics['var_95']:.2%}")
        print(f"  CVaR (95%): {summary.risk_metrics['cvar_95']:.2%}")
        print(f"  Skewness: {summary.risk_metrics['skewness']:.3f}")
        print(f"  Kurtosis: {summary.risk_metrics['kurtosis']:.3f}")
        print(f"  Tail Ratio: {summary.risk_metrics['tail_ratio']:.3f}")
        
        # Save detailed results if requested
        if args.output_file:
            output_file = args.output_file
            if not output_file.endswith('.json'):
                output_file += '.json'
            
            # Export results
            import json
            
            results_dict = {
                'summary': {
                    'total_scenarios': summary.total_scenarios,
                    'mean_return': summary.mean_return,
                    'median_return': summary.median_return,
                    'std_return': summary.std_return,
                    'best_case_return': summary.best_case_return,
                    'worst_case_return': summary.worst_case_return,
                    'probability_positive': summary.probability_positive,
                    'probability_target': summary.probability_target,
                    'confidence_intervals': summary.confidence_intervals,
                    'risk_metrics': summary.risk_metrics
                },
                'scenarios': [
                    {
                        'scenario_id': r.scenario_id,
                        'total_return': r.total_return,
                        'sharpe_ratio': r.sharpe_ratio,
                        'max_drawdown': r.max_drawdown,
                        'total_trades': r.total_trades,
                        'win_rate': r.win_rate
                    }
                    for r in summary.scenario_results
                ]
            }
            
            with open(output_file, 'w') as f:
                json.dump(results_dict, f, indent=2)
            
            print(f"\nDetailed results saved to: {output_file}")
        
        print("\n" + "="*60)
        print("MONTE CARLO SIMULATION COMPLETED")
        print("="*60)
        
        return True
        
    except Exception as e:
        logger.error(f"Error running Monte Carlo simulation: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return False


async def run_multi_timeframe(args):
    """Run multi-timeframe analysis."""
    logger.info("Starting Multi-Timeframe Analysis")
    
    if not await check_database_connection():
        logger.error("Cannot proceed without database connection")
        return False
    
    # Parse dates
    if args.start_date:
        start_date = datetime.strptime(args.start_date, '%Y-%m-%d')
    else:
        start_date = datetime.utcnow() - timedelta(days=args.days)
    
    if args.end_date:
        end_date = datetime.strptime(args.end_date, '%Y-%m-%d')
    else:
        end_date = datetime.utcnow() - timedelta(days=1)
    
    # Create base configuration
    config = BacktestConfig(
        start_date=start_date,
        end_date=end_date,
        initial_capital=Decimal(str(args.capital)),
        symbols=args.symbols.split(',') if args.symbols else settings.trading.trading_pairs[:args.max_symbols],
        max_position_size=Decimal(str(args.max_position_size)),
        commission_rate=Decimal(str(args.commission_rate)),
        slippage_rate=Decimal(str(args.slippage_rate))
    )
    
    timeframes = args.timeframes.split(',') if args.timeframes else ['1h', '4h', '1d']
    
    logger.info(f"Multi-Timeframe Configuration:")
    logger.info(f"  Timeframes: {timeframes}")
    logger.info(f"  Period: {start_date} to {end_date}")
    logger.info(f"  Symbols: {len(config.symbols)}")
    
    try:
        # Create and run multi-timeframe analysis
        analyzer = create_multi_timeframe_analyzer(config)
        
        results = await analyzer.analyze_timeframes(timeframes)
        
        # Display results
        print("\n" + "="*80)
        print("MULTI-TIMEFRAME ANALYSIS RESULTS")
        print("="*80)
        
        print(f"{'Timeframe':<10} {'Return':<10} {'Sharpe':<8} {'Drawdown':<10} {'Trades':<8} {'Win Rate':<10} {'Efficiency':<12}")
        print("-" * 80)
        
        for result in results:
            tf = result.timeframe
            br = result.backtest_results
            metrics = result.timeframe_specific_metrics
            
            print(f"{tf:<10} {br.total_return:<9.2%} {br.sharpe_ratio:<7.2f} "
                  f"{br.max_drawdown:<9.2%} {br.total_trades:<7d} {br.win_rate:<9.2%} "
                  f"{metrics.get('timeframe_efficiency', 0):<11.4f}")
        
        # Detailed analysis
        print(f"\nDetailed Timeframe Analysis:")
        for result in results:
            print(f"\n{result.timeframe} Timeframe:")
            print(f"  Total Return: {result.backtest_results.total_return:.2%}")
            print(f"  Sharpe Ratio: {result.backtest_results.sharpe_ratio:.2f}")
            print(f"  Max Drawdown: {result.backtest_results.max_drawdown:.2%}")
            print(f"  Total Trades: {result.backtest_results.total_trades}")
            print(f"  Win Rate: {result.backtest_results.win_rate:.2%}")
            
            metrics = result.timeframe_specific_metrics
            print(f"  Trades per Day: {metrics.get('trades_per_day', 0):.2f}")
            print(f"  Signal Frequency: {metrics.get('signal_frequency', 0):.2f}")
            print(f"  Return per Trade: {metrics.get('return_per_trade', 0):.4f}")
            print(f"  Timeframe Efficiency: {metrics.get('timeframe_efficiency', 0):.6f}")
        
        # Find best timeframe
        if results:
            best_result = max(results, key=lambda x: x.backtest_results.sharpe_ratio)
            print(f"\nBest Performing Timeframe: {best_result.timeframe}")
            print(f"  Sharpe Ratio: {best_result.backtest_results.sharpe_ratio:.2f}")
            print(f"  Total Return: {best_result.backtest_results.total_return:.2%}")
        
        print("\n" + "="*80)
        print("MULTI-TIMEFRAME ANALYSIS COMPLETED")
        print("="*80)
        
        return True
        
    except Exception as e:
        logger.error(f"Error running multi-timeframe analysis: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return False


async def run_benchmark_comparison(args):
    """Run benchmark comparison analysis."""
    logger.info("Starting Benchmark Comparison")
    
    if not await check_database_connection():
        logger.error("Cannot proceed without database connection")
        return False
    
    # First run a regular backtest
    if args.start_date:
        start_date = datetime.strptime(args.start_date, '%Y-%m-%d')
    else:
        start_date = datetime.utcnow() - timedelta(days=args.days)
    
    if args.end_date:
        end_date = datetime.strptime(args.end_date, '%Y-%m-%d')
    else:
        end_date = datetime.utcnow() - timedelta(days=1)
    
    config = BacktestConfig(
        start_date=start_date,
        end_date=end_date,
        initial_capital=Decimal(str(args.capital)),
        symbols=args.symbols.split(',') if args.symbols else settings.trading.trading_pairs[:args.max_symbols],
        max_position_size=Decimal(str(args.max_position_size)),
        commission_rate=Decimal(str(args.commission_rate)),
        slippage_rate=Decimal(str(args.slippage_rate))
    )
    
    benchmarks = args.benchmarks.split(',') if args.benchmarks else ['BTCUSDT', 'ETHUSDT']
    
    logger.info(f"Benchmark Comparison Configuration:")
    logger.info(f"  Benchmarks: {benchmarks}")
    logger.info(f"  Period: {start_date} to {end_date}")
    logger.info(f"  Symbols: {len(config.symbols)}")
    
    try:
        # Run strategy backtest
        engine = IntegratedBacktestEngine(config)
        strategy_results = await engine.run_backtest()
        
        # Compare against benchmarks
        comparisons = await benchmark_comparator.compare_to_benchmarks(
            strategy_results, benchmarks
        )
        
        # Display results
        print("\n" + "="*80)
        print("BENCHMARK COMPARISON RESULTS")
        print("="*80)
        
        print(f"Strategy Performance:")
        print(f"  Total Return: {strategy_results.total_return:.2%}")
        print(f"  Sharpe Ratio: {strategy_results.sharpe_ratio:.2f}")
        print(f"  Max Drawdown: {strategy_results.max_drawdown:.2%}")
        print(f"  Win Rate: {strategy_results.win_rate:.2%}")
        
        print(f"\nBenchmark Comparisons:")
        print(f"{'Benchmark':<12} {'Return':<10} {'Excess':<10} {'Beta':<8} {'Alpha':<10} {'Info Ratio':<12} {'Correlation':<12}")
        print("-" * 80)
        
        for comp in comparisons:
            print(f"{comp.benchmark_name:<12} {comp.benchmark_return:<9.2%} {comp.excess_return:<9.2%} "
                  f"{comp.beta:<7.2f} {comp.alpha:<9.2%} {comp.information_ratio:<11.2f} {comp.correlation:<11.2f}")
        
        # Detailed comparison
        print(f"\nDetailed Benchmark Analysis:")
        for comp in comparisons:
            print(f"\n{comp.benchmark_name} Comparison:")
            print(f"  Strategy Return: {comp.strategy_return:.2%}")
            print(f"  Benchmark Return: {comp.benchmark_return:.2%}")
            print(f"  Excess Return: {comp.excess_return:.2%}")
            print(f"  Tracking Error: {comp.tracking_error:.2%}")
            print(f"  Information Ratio: {comp.information_ratio:.2f}")
            print(f"  Beta: {comp.beta:.2f}")
            print(f"  Alpha: {comp.alpha:.2%}")
            print(f"  Correlation: {comp.correlation:.2f}")
            print(f"  Up Capture: {comp.up_capture:.2f}")
            print(f"  Down Capture: {comp.down_capture:.2f}")
        
        print("\n" + "="*80)
        print("BENCHMARK COMPARISON COMPLETED")
        print("="*80)
        
        return True
        
    except Exception as e:
        logger.error(f"Error running benchmark comparison: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return False


async def optimize_performance(args):
    """Run performance optimization tools."""
    logger.info("Running Performance Optimization")
    
    if args.clear_cache:
        performance_optimizer.clear_cache()
        print("Cache cleared successfully")
        return True
    
    if args.preload_data:
        if not await check_database_connection():
            logger.error("Cannot proceed without database connection")
            return False
        
        # Parse parameters
        symbols = args.symbols.split(',') if args.symbols else settings.trading.trading_pairs[:args.max_symbols]
        
        if args.start_date:
            start_date = datetime.strptime(args.start_date, '%Y-%m-%d')
        else:
            start_date = datetime.utcnow() - timedelta(days=args.days)
        
        if args.end_date:
            end_date = datetime.strptime(args.end_date, '%Y-%m-%d')
        else:
            end_date = datetime.utcnow() - timedelta(days=1)
        
        logger.info(f"Preloading data for {len(symbols)} symbols from {start_date} to {end_date}")
        
        try:
            data_cache = await performance_optimizer.optimize_data_loading(
                symbols, start_date, end_date, use_cache=True
            )
            
            print(f"Successfully preloaded data for {len(data_cache)} symbols")
            for symbol, df in data_cache.items():
                print(f"  {symbol}: {len(df)} records")
            
            return True
            
        except Exception as e:
            logger.error(f"Error preloading data: {e}")
            return False
    
    print("No optimization operation specified. Use --clear-cache or --preload-data")
    return False


def main():
    """Main CLI function."""
    parser = argparse.ArgumentParser(description='Advanced Backtesting CLI')
    subparsers = parser.add_subparsers(dest='command', help='Available commands')
    
    # Monte Carlo simulation
    mc_parser = subparsers.add_parser('monte-carlo', help='Run Monte Carlo simulation')
    mc_parser.add_argument('--scenarios', type=int, default=1000, help='Number of scenarios (default: 1000)')
    mc_parser.add_argument('--start-date', type=str, help='Start date (YYYY-MM-DD)')
    mc_parser.add_argument('--end-date', type=str, help='End date (YYYY-MM-DD)')
    mc_parser.add_argument('--days', type=int, default=30, help='Number of days back from today (default: 30)')
    mc_parser.add_argument('--symbols', type=str, help='Comma-separated list of symbols')
    mc_parser.add_argument('--max-symbols', type=int, default=5, help='Maximum number of symbols (default: 5)')
    mc_parser.add_argument('--capital', type=float, default=100000, help='Initial capital (default: 100000)')
    mc_parser.add_argument('--max-position-size', type=float, default=0.15, help='Max position size (default: 0.15)')
    mc_parser.add_argument('--commission-rate', type=float, default=0.001, help='Commission rate (default: 0.001)')
    mc_parser.add_argument('--slippage-rate', type=float, default=0.0005, help='Slippage rate (default: 0.0005)')
    mc_parser.add_argument('--randomization-methods', type=str, default='bootstrap_trades,shuffle_returns,parameter_variation', 
                          help='Comma-separated randomization methods')
    mc_parser.add_argument('--workers', type=int, default=4, help='Number of parallel workers (default: 4)')
    mc_parser.add_argument('--output-file', type=str, help='Output file for detailed results')
    
    # Multi-timeframe analysis
    mt_parser = subparsers.add_parser('multi-timeframe', help='Run multi-timeframe analysis')
    mt_parser.add_argument('--timeframes', type=str, default='1h,4h,1d', help='Comma-separated timeframes (default: 1h,4h,1d)')
    mt_parser.add_argument('--start-date', type=str, help='Start date (YYYY-MM-DD)')
    mt_parser.add_argument('--end-date', type=str, help='End date (YYYY-MM-DD)')
    mt_parser.add_argument('--days', type=int, default=30, help='Number of days back from today (default: 30)')
    mt_parser.add_argument('--symbols', type=str, help='Comma-separated list of symbols')
    mt_parser.add_argument('--max-symbols', type=int, default=5, help='Maximum number of symbols (default: 5)')
    mt_parser.add_argument('--capital', type=float, default=100000, help='Initial capital (default: 100000)')
    mt_parser.add_argument('--max-position-size', type=float, default=0.15, help='Max position size (default: 0.15)')
    mt_parser.add_argument('--commission-rate', type=float, default=0.001, help='Commission rate (default: 0.001)')
    mt_parser.add_argument('--slippage-rate', type=float, default=0.0005, help='Slippage rate (default: 0.0005)')
    
    # Benchmark comparison
    bm_parser = subparsers.add_parser('benchmark', help='Run benchmark comparison')
    bm_parser.add_argument('--benchmarks', type=str, default='BTCUSDT,ETHUSDT', help='Comma-separated benchmark symbols')
    bm_parser.add_argument('--start-date', type=str, help='Start date (YYYY-MM-DD)')
    bm_parser.add_argument('--end-date', type=str, help='End date (YYYY-MM-DD)')
    bm_parser.add_argument('--days', type=int, default=30, help='Number of days back from today (default: 30)')
    bm_parser.add_argument('--symbols', type=str, help='Comma-separated list of symbols')
    bm_parser.add_argument('--max-symbols', type=int, default=5, help='Maximum number of symbols (default: 5)')
    bm_parser.add_argument('--capital', type=float, default=100000, help='Initial capital (default: 100000)')
    bm_parser.add_argument('--max-position-size', type=float, default=0.15, help='Max position size (default: 0.15)')
    bm_parser.add_argument('--commission-rate', type=float, default=0.001, help='Commission rate (default: 0.001)')
    bm_parser.add_argument('--slippage-rate', type=float, default=0.0005, help='Slippage rate (default: 0.0005)')
    
    # Performance optimization
    opt_parser = subparsers.add_parser('optimize', help='Performance optimization tools')
    opt_parser.add_argument('--clear-cache', action='store_true', help='Clear performance cache')
    opt_parser.add_argument('--preload-data', action='store_true', help='Preload and cache data')
    opt_parser.add_argument('--start-date', type=str, help='Start date for preloading (YYYY-MM-DD)')
    opt_parser.add_argument('--end-date', type=str, help='End date for preloading (YYYY-MM-DD)')
    opt_parser.add_argument('--days', type=int, default=30, help='Number of days for preloading (default: 30)')
    opt_parser.add_argument('--symbols', type=str, help='Symbols to preload')
    opt_parser.add_argument('--max-symbols', type=int, default=10, help='Maximum symbols to preload (default: 10)')
    
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        return
    
    # Run appropriate command
    if args.command == 'monte-carlo':
        success = asyncio.run(run_monte_carlo(args))
    elif args.command == 'multi-timeframe':
        success = asyncio.run(run_multi_timeframe(args))
    elif args.command == 'benchmark':
        success = asyncio.run(run_benchmark_comparison(args))
    elif args.command == 'optimize':
        success = asyncio.run(optimize_performance(args))
    else:
        logger.error(f"Unknown command: {args.command}")
        success = False
    
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
