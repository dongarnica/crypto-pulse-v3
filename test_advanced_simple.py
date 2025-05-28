#!/usr/bin/env python3
"""
Simple test for advanced backtesting features to check basic functionality.
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

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


async def test_basic_imports():
    """Test basic imports of advanced features."""
    try:
        logger.info("Testing basic imports...")
        
        # Test engine imports
        from src.backtesting.engine import BacktestEngine
        from src.backtesting.config import BacktestConfig
        from src.backtesting.results import BacktestResults
        logger.info("✓ Basic backtesting imports successful")
        
        # Test integration imports
        from src.backtesting.integration import DatabaseDataProvider, IntegratedBacktestEngine
        logger.info("✓ Integration imports successful")
        
        # Test advanced features imports
        from src.backtesting.advanced_features import (
            MonteCarloSimulator,
            MultiTimeframeAnalyzer,
            BenchmarkComparator,
            PerformanceOptimizer
        )
        logger.info("✓ Advanced features imports successful")
        
        return True
        
    except Exception as e:
        logger.error(f"Import test failed: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return False


async def test_database_connection():
    """Test database connectivity."""
    try:
        logger.info("Testing database connection...")
        
        from src.core.database import db_manager
        await db_manager.initialize()
        
        # Check for market data
        session = db_manager.get_session()
        from src.core.models import MarketData
        
        recent_data = session.query(MarketData).filter(
            MarketData.timestamp > datetime.utcnow() - timedelta(days=7)
        ).count()
        
        logger.info(f"Found {recent_data} market data records in last 7 days")
        session.close()
        
        return recent_data > 0
        
    except Exception as e:
        logger.error(f"Database test failed: {e}")
        return False


async def test_monte_carlo_creation():
    """Test Monte Carlo simulator creation."""
    try:
        logger.info("Testing Monte Carlo simulator creation...")
        
        from src.backtesting.config import BacktestConfig
        from src.backtesting.advanced_features import MonteCarloSimulator
        
        config = BacktestConfig(
            start_date=datetime.utcnow() - timedelta(days=7),
            end_date=datetime.utcnow() - timedelta(days=1),
            initial_capital=Decimal('10000'),
            symbols=['BTCUSD'],
            max_position_size=Decimal('0.5')
        )
        
        simulator = MonteCarloSimulator(config)
        logger.info("✓ Monte Carlo simulator created successfully")
        
        return True
        
    except Exception as e:
        logger.error(f"Monte Carlo creation test failed: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return False


async def test_mini_monte_carlo():
    """Test a very small Monte Carlo simulation."""
    try:
        logger.info("Testing mini Monte Carlo simulation...")
        
        from src.backtesting.config import BacktestConfig
        from src.backtesting.advanced_features import MonteCarloSimulator
        
        config = BacktestConfig(
            start_date=datetime.utcnow() - timedelta(days=3),
            end_date=datetime.utcnow() - timedelta(days=1),
            initial_capital=Decimal('1000'),
            symbols=['BTCUSD'],
            max_position_size=Decimal('0.5')
        )
        
        simulator = MonteCarloSimulator(config)
        
        # Run just 3 scenarios
        summary = await simulator.run_simulation(
            num_scenarios=3,
            randomization_methods=['bootstrap_trades'],
            parallel_workers=1
        )
        
        logger.info(f"✓ Mini Monte Carlo completed:")
        logger.info(f"  Scenarios: {summary.total_scenarios}")
        logger.info(f"  Mean Return: {summary.mean_return:.2%}")
        
        return True
        
    except Exception as e:
        logger.error(f"Mini Monte Carlo test failed: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return False


async def test_performance_optimizer():
    """Test performance optimizer."""
    try:
        logger.info("Testing performance optimizer...")
        
        from src.backtesting.advanced_features import PerformanceOptimizer
        
        optimizer = PerformanceOptimizer()
        cache_info = optimizer.get_cache_info()
        
        logger.info(f"✓ Performance optimizer working:")
        logger.info(f"  Cache entries: {cache_info['entries']}")
        logger.info(f"  Cache size: {cache_info['size']}")
        
        return True
        
    except Exception as e:
        logger.error(f"Performance optimizer test failed: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return False


async def run_simple_tests():
    """Run simplified test suite."""
    logger.info("Starting Simple Advanced Features Test")
    logger.info("="*50)
    
    tests = [
        ("Basic Imports", test_basic_imports),
        ("Database Connection", test_database_connection),
        ("Monte Carlo Creation", test_monte_carlo_creation),
        ("Performance Optimizer", test_performance_optimizer),
        ("Mini Monte Carlo", test_mini_monte_carlo),
    ]
    
    results = {}
    
    for test_name, test_func in tests:
        logger.info(f"\n--- {test_name} ---")
        try:
            results[test_name] = await test_func()
        except Exception as e:
            logger.error(f"{test_name} failed with exception: {e}")
            results[test_name] = False
    
    # Summary
    logger.info("\n" + "="*50)
    logger.info("TEST SUMMARY")
    logger.info("="*50)
    
    passed = sum(results.values())
    total = len(results)
    
    for test_name, result in results.items():
        status = "PASSED" if result else "FAILED"
        logger.info(f"{test_name:25s}: {status}")
    
    logger.info(f"\nOverall: {passed}/{total} tests passed")
    
    return passed == total


async def main():
    """Main test runner."""
    try:
        success = await run_simple_tests()
        return 0 if success else 1
    except Exception as e:
        logger.error(f"Test suite failed: {e}")
        return 1
    finally:
        try:
            from src.core.database import db_manager
            await db_manager.close()
        except:
            pass


if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code)
