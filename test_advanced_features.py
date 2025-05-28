#!/usr/bin/env python3
"""
Comprehensive test suite for advanced backtesting features.
Tests Monte Carlo simulation, multi-timeframe analysis, benchmark comparison,
and performance optimization with real database data.
"""

import asyncio
import logging
import sys
from pathlib import Path
from datetime import datetime, timedelta
from decimal import Decimal

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from src.backtesting import (
    BacktestConfig,
    MonteCarloSimulator,
    MultiTimeframeAnalyzer,
    BenchmarkComparator,
    PerformanceOptimizer,
    IntegratedBacktestEngine,
    create_monte_carlo_simulator,
    create_multi_timeframe_analyzer,
    benchmark_comparator,
    performance_optimizer
)
from src.core.database import db_manager
from config.settings import settings

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


async def test_database_connection():
    """Test database connectivity and data availability."""
    logger.info("Testing database connection...")
    
    try:
        await db_manager.initialize()
        
        # Check for market data
        session = db_manager.get_session()
        from src.core.models import MarketData
        
        # Get recent data count
        recent_data = session.query(MarketData).filter(
            MarketData.timestamp > datetime.utcnow() - timedelta(days=30)
        ).count()
        
        logger.info(f"Found {recent_data} market data records in last 30 days")
        
        if recent_data == 0:
            logger.warning("No recent market data found - tests may not work properly")
            return False
        
        # Get available symbols
        symbols = session.query(MarketData.symbol).distinct().limit(5).all()
        symbols = [s[0] for s in symbols]
        logger.info(f"Available symbols: {symbols}")
        
        session.close()
        return True
        
    except Exception as e:
        logger.error(f"Database test failed: {e}")
        return False


async def test_monte_carlo_simulation():
    """Test Monte Carlo simulation with small scenario count."""
    logger.info("\n" + "="*50)
    logger.info("TESTING MONTE CARLO SIMULATION")
    logger.info("="*50)
    
    try:
        # Create test configuration
        config = BacktestConfig(
            start_date=datetime.utcnow() - timedelta(days=14),
            end_date=datetime.utcnow() - timedelta(days=1),
            initial_capital=Decimal('10000'),
            symbols=['BTCUSD', 'ETHUSD'],  # Test with common crypto pairs
            max_position_size=Decimal('0.3'),
            commission_rate=Decimal('0.001'),
            slippage_rate=Decimal('0.0005')
        )
        
        logger.info(f"Config: {config.start_date} to {config.end_date}")
        logger.info(f"Symbols: {config.symbols}")
        
        # Create simulator
        simulator = create_monte_carlo_simulator(config)
        
        # Run small simulation
        logger.info("Running 10 Monte Carlo scenarios...")
        summary = await simulator.run_simulation(
            num_scenarios=10,
            randomization_methods=['bootstrap_trades', 'shuffle_returns'],
            parallel_workers=2
        )
        
        # Display results
        logger.info(f"Monte Carlo Results:")
        logger.info(f"  Mean Return: {summary.mean_return:.2%}")
        logger.info(f"  Std Deviation: {summary.std_return:.2%}")
        logger.info(f"  95% VaR: {summary.var_95:.2%}")
        logger.info(f"  95% CVaR: {summary.cvar_95:.2%}")
        logger.info(f"  Prob > 0: {summary.prob_positive:.2%}")
        logger.info(f"  Best Case: {summary.best_case_return:.2%}")
        logger.info(f"  Worst Case: {summary.worst_case_return:.2%}")
        
        return True
        
    except Exception as e:
        logger.error(f"Monte Carlo test failed: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return False


async def test_multi_timeframe_analysis():
    """Test multi-timeframe analysis."""
    logger.info("\n" + "="*50)
    logger.info("TESTING MULTI-TIMEFRAME ANALYSIS")
    logger.info("="*50)
    
    try:
        # Create test configuration
        config = BacktestConfig(
            start_date=datetime.utcnow() - timedelta(days=7),
            end_date=datetime.utcnow() - timedelta(days=1),
            initial_capital=Decimal('10000'),
            symbols=['BTCUSD'],
            max_position_size=Decimal('0.5'),
            commission_rate=Decimal('0.001'),
            slippage_rate=Decimal('0.0005')
        )
        
        # Create analyzer
        analyzer = create_multi_timeframe_analyzer(config)
        
        # Run analysis
        logger.info("Running multi-timeframe analysis...")
        result = await analyzer.analyze_timeframes(
            timeframes=['1h', '4h', '1d'],
            use_parallel=True
        )
        
        # Display results
        logger.info(f"Multi-Timeframe Results:")
        for tf, metrics in result.timeframe_results.items():
            logger.info(f"  {tf}:")
            logger.info(f"    Total Return: {metrics.total_return:.2%}")
            logger.info(f"    Sharpe Ratio: {metrics.sharpe_ratio:.2f}")
            logger.info(f"    Max Drawdown: {metrics.max_drawdown:.2%}")
            logger.info(f"    Total Trades: {metrics.total_trades}")
        
        logger.info(f"Timeframe Efficiency:")
        for tf, eff in result.timeframe_efficiency.items():
            logger.info(f"  {tf}: {eff:.3f}")
        
        return True
        
    except Exception as e:
        logger.error(f"Multi-timeframe test failed: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return False


async def test_benchmark_comparison():
    """Test benchmark comparison."""
    logger.info("\n" + "="*50)
    logger.info("TESTING BENCHMARK COMPARISON")
    logger.info("="*50)
    
    try:
        # Create test configuration
        config = BacktestConfig(
            start_date=datetime.utcnow() - timedelta(days=14),
            end_date=datetime.utcnow() - timedelta(days=1),
            initial_capital=Decimal('10000'),
            symbols=['BTCUSD', 'ETHUSD'],
            max_position_size=Decimal('0.4'),
            commission_rate=Decimal('0.001'),
            slippage_rate=Decimal('0.0005')
        )
        
        # Run base backtest first
        engine = IntegratedBacktestEngine(config)
        results = await engine.run_backtest()
        
        # Create comparator
        comparator = benchmark_comparator
        
        # Run comparison
        logger.info("Running benchmark comparison...")
        comparison = await comparator.compare_to_benchmark(
            strategy_results=results,
            benchmark_symbol='BTCUSD',
            config=config
        )
        
        # Display results
        logger.info(f"Benchmark Comparison Results:")
        logger.info(f"  Strategy Return: {comparison.strategy_return:.2%}")
        logger.info(f"  Benchmark Return: {comparison.benchmark_return:.2%}")
        logger.info(f"  Alpha: {comparison.alpha:.2%}")
        logger.info(f"  Beta: {comparison.beta:.2f}")
        logger.info(f"  Tracking Error: {comparison.tracking_error:.2%}")
        logger.info(f"  Information Ratio: {comparison.information_ratio:.2f}")
        logger.info(f"  Up Capture: {comparison.up_capture:.2%}")
        logger.info(f"  Down Capture: {comparison.down_capture:.2%}")
        
        return True
        
    except Exception as e:
        logger.error(f"Benchmark comparison test failed: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return False


async def test_performance_optimization():
    """Test performance optimization features."""
    logger.info("\n" + "="*50)
    logger.info("TESTING PERFORMANCE OPTIMIZATION")
    logger.info("="*50)
    
    try:
        optimizer = performance_optimizer
        
        # Test cache status
        logger.info("Checking cache status...")
        cache_info = optimizer.get_cache_info()
        logger.info(f"Cache entries: {cache_info['entries']}")
        logger.info(f"Cache size: {cache_info['size']}")
        
        # Test data preloading
        logger.info("Testing data preloading...")
        symbols = ['BTCUSD', 'ETHUSD']
        start_date = datetime.utcnow() - timedelta(days=7)
        end_date = datetime.utcnow() - timedelta(days=1)
        
        await optimizer.preload_data(
            symbols=symbols,
            start_date=start_date,
            end_date=end_date
        )
        
        logger.info("Data preloading completed")
        
        # Check cache again
        cache_info = optimizer.get_cache_info()
        logger.info(f"Cache entries after preload: {cache_info['entries']}")
        
        return True
        
    except Exception as e:
        logger.error(f"Performance optimization test failed: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return False


async def run_all_tests():
    """Run all advanced features tests."""
    logger.info("Starting Advanced Features Test Suite")
    logger.info("="*60)
    
    # Test database first
    if not await test_database_connection():
        logger.error("Database connection failed - aborting tests")
        return False
    
    test_results = {}
    
    # Run individual tests
    test_results['monte_carlo'] = await test_monte_carlo_simulation()
    test_results['multi_timeframe'] = await test_multi_timeframe_analysis()
    test_results['benchmark'] = await test_benchmark_comparison()
    test_results['performance'] = await test_performance_optimization()
    
    # Summary
    logger.info("\n" + "="*60)
    logger.info("TEST SUMMARY")
    logger.info("="*60)
    
    passed = sum(test_results.values())
    total = len(test_results)
    
    for test_name, result in test_results.items():
        status = "PASSED" if result else "FAILED"
        logger.info(f"{test_name:20s}: {status}")
    
    logger.info(f"\nOverall: {passed}/{total} tests passed")
    
    if passed == total:
        logger.info("All advanced features are working correctly!")
        return True
    else:
        logger.warning("Some tests failed - check logs for details")
        return False


async def main():
    """Main test runner."""
    try:
        success = await run_all_tests()
        return 0 if success else 1
    except KeyboardInterrupt:
        logger.info("Tests interrupted by user")
        return 1
    except Exception as e:
        logger.error(f"Test suite failed: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return 1
    finally:
        try:
            await db_manager.close()
        except:
            pass


if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code)
