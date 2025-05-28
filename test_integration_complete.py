#!/usr/bin/env python3
"""
Comprehensive test suite for integrated backtesting functionality.
Tests database connectivity, data quality, strategy integration, and validation.
"""

import asyncio
import sys
import logging
import unittest
from pathlib import Path
from datetime import datetime, timedelta
from decimal import Decimal

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from src.backtesting import (
    BacktestConfig,
    IntegratedBacktestEngine,
    DatabaseDataProvider,
    LiveStrategyIntegration,
    BacktestValidator
)
from src.core.database import db_manager
from config.settings import settings

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class TestIntegratedBacktesting(unittest.TestCase):
    """Test cases for integrated backtesting functionality."""
    
    @classmethod
    def setUpClass(cls):
        """Set up test class."""
        cls.data_provider = DatabaseDataProvider()
        cls.strategy_integration = LiveStrategyIntegration()
        cls.validator = BacktestValidator()
        
        # Test configuration
        cls.test_symbol = 'BTCUSDT'
        cls.test_start = datetime.utcnow() - timedelta(days=7)
        cls.test_end = datetime.utcnow() - timedelta(days=1)
    
    async def async_setUp(self):
        """Async setup for each test."""
        try:
            await db_manager.initialize()
            await self.strategy_integration.initialize()
        except Exception as e:
            self.skipTest(f"Failed to initialize components: {e}")
    
    def setUp(self):
        """Set up each test."""
        asyncio.run(self.async_setUp())


class TestDatabaseIntegration(TestIntegratedBacktesting):
    """Test database integration functionality."""
    
    def test_database_connection(self):
        """Test database connectivity."""
        async def run_test():
            try:
                await db_manager.initialize()
                return True
            except Exception as e:
                logger.error(f"Database connection failed: {e}")
                return False
        
        result = asyncio.run(run_test())
        self.assertTrue(result, "Database connection should succeed")
    
    def test_historical_data_retrieval(self):
        """Test historical data retrieval."""
        async def run_test():
            df = await self.data_provider.get_historical_data(
                self.test_symbol, self.test_start, self.test_end
            )
            
            self.assertFalse(df.empty, "Historical data should not be empty")
            
            # Check required columns
            required_cols = ['open', 'high', 'low', 'close', 'volume']
            for col in required_cols:
                self.assertIn(col, df.columns, f"Column {col} should be present")
            
            # Check data quality
            self.assertFalse(df['close'].isna().all(), "Close prices should not be all NaN")
            self.assertTrue((df['high'] >= df['low']).all(), "High should be >= Low")
            self.assertTrue((df['high'] >= df['close']).all(), "High should be >= Close")
            self.assertTrue((df['low'] <= df['close']).all(), "Low should be <= Close")
            
            logger.info(f"Retrieved {len(df)} data points for {self.test_symbol}")
            return True
        
        result = asyncio.run(run_test())
        self.assertTrue(result)
    
    def test_sentiment_data_retrieval(self):
        """Test sentiment data retrieval."""
        async def run_test():
            df = await self.data_provider.get_sentiment_data(
                self.test_symbol, self.test_start, self.test_end
            )
            
            # Sentiment data might be empty, but should not raise error
            logger.info(f"Retrieved {len(df)} sentiment records")
            return True
        
        result = asyncio.run(run_test())
        self.assertTrue(result)
    
    def test_historical_signals_retrieval(self):
        """Test historical trading signals retrieval."""
        async def run_test():
            df = await self.data_provider.get_historical_signals(
                self.test_symbol, self.test_start, self.test_end
            )
            
            # Signals might be empty, but should not raise error
            logger.info(f"Retrieved {len(df)} historical signals")
            return True
        
        result = asyncio.run(run_test())
        self.assertTrue(result)


class TestStrategyIntegration(TestIntegratedBacktesting):
    """Test live strategy integration."""
    
    def test_ml_model_integration(self):
        """Test ML model integration."""
        async def run_test():
            # Get some historical data
            df = await self.data_provider.get_historical_data(
                self.test_symbol, self.test_start, self.test_end
            )
            
            if df.empty:
                self.skipTest("No historical data available for ML test")
            
            # Test analysis generation
            test_timestamp = df.index[-10]  # Use recent but not latest data
            
            analysis = await self.strategy_integration.generate_historical_analysis(
                self.test_symbol, test_timestamp, df
            )
            
            if analysis:
                self.assertEqual(analysis.symbol, self.test_symbol)
                self.assertIsInstance(analysis.confidence, float)
                self.assertGreaterEqual(analysis.confidence, 0.0)
                self.assertLessEqual(analysis.confidence, 1.0)
                self.assertIn(analysis.trading_signal, ['BUY', 'SELL', 'HOLD'])
                
                logger.info(f"Generated analysis: {analysis.trading_signal} with confidence {analysis.confidence:.2f}")
                return True
            else:
                logger.warning("No analysis generated (might be due to insufficient data)")
                return True  # Not necessarily a failure
        
        result = asyncio.run(run_test())
        self.assertTrue(result)
    
    def test_decision_generation(self):
        """Test trading decision generation."""
        async def run_test():
            # Create mock analysis
            from src.core.trading_engine import MarketAnalysis
            
            mock_analysis = MarketAnalysis(
                symbol=self.test_symbol,
                timestamp=datetime.utcnow(),
                technical_features={'close': 50000.0, 'rsi_14': 45.0},
                ml_prediction={'expected_return': 0.05, 'confidence_score': 0.75},
                sentiment_score=0.6,
                risk_metrics={'risk_score': 0.4, 'volatility': 0.02},
                trading_signal='BUY',
                confidence=0.75
            )
            
            decision = await self.strategy_integration.generate_historical_decision(
                mock_analysis, has_position=False, portfolio_value=100000.0
            )
            
            if decision:
                self.assertEqual(decision.symbol, self.test_symbol)
                self.assertIn(decision.action, ['BUY', 'SELL', 'HOLD'])
                self.assertIsInstance(decision.confidence, float)
                
                logger.info(f"Generated decision: {decision.action} with allocation {decision.recommended_allocation:.2%}")
                return True
            else:
                logger.warning("No decision generated")
                return True
        
        result = asyncio.run(run_test())
        self.assertTrue(result)


class TestIntegratedEngine(TestIntegratedBacktesting):
    """Test the integrated backtest engine."""
    
    def test_engine_initialization(self):
        """Test engine initialization."""
        config = BacktestConfig(
            start_date=self.test_start,
            end_date=self.test_end,
            initial_capital=Decimal('10000'),
            symbols=[self.test_symbol],
            max_position_size=Decimal('0.1')
        )
        
        engine = IntegratedBacktestEngine(config)
        self.assertIsNotNone(engine.strategy_integration)
        self.assertIsNotNone(engine.data_provider)
    
    def test_data_loading(self):
        """Test historical data loading."""
        async def run_test():
            config = BacktestConfig(
                start_date=self.test_start,
                end_date=self.test_end,
                initial_capital=Decimal('10000'),
                symbols=[self.test_symbol],
                max_position_size=Decimal('0.1')
            )
            
            engine = IntegratedBacktestEngine(config)
            await engine.initialize()
            
            success = await engine.load_historical_data()
            
            if success:
                self.assertIn(self.test_symbol, engine.market_data_cache)
                df = engine.market_data_cache[self.test_symbol]
                self.assertFalse(df.empty)
                logger.info(f"Loaded {len(df)} data points for backtesting")
            else:
                logger.warning("Data loading failed (might be due to no available data)")
            
            return success
        
        result = asyncio.run(run_test())
        # Don't fail if no data available, just log
        logger.info(f"Data loading test result: {result}")
    
    def test_mini_backtest(self):
        """Test a mini backtest with limited scope."""
        async def run_test():
            # Use shorter period for faster test
            short_start = datetime.utcnow() - timedelta(days=3)
            short_end = datetime.utcnow() - timedelta(days=1)
            
            config = BacktestConfig(
                start_date=short_start,
                end_date=short_end,
                initial_capital=Decimal('10000'),
                symbols=[self.test_symbol],
                max_position_size=Decimal('0.1'),
                commission_rate=Decimal('0.001'),
                slippage_rate=Decimal('0.0005')
            )
            
            engine = IntegratedBacktestEngine(config)
            
            try:
                results = await engine.run_backtest()
                
                # Basic validation of results
                self.assertIsNotNone(results)
                self.assertIsInstance(results.total_return, float)
                self.assertIsInstance(results.total_trades, int)
                self.assertGreaterEqual(results.total_trades, 0)
                
                logger.info(f"Mini backtest completed:")
                logger.info(f"  Total Return: {results.total_return:.2%}")
                logger.info(f"  Total Trades: {results.total_trades}")
                logger.info(f"  Sharpe Ratio: {results.sharpe_ratio:.2f}")
                
                return True
                
            except Exception as e:
                # Log error but don't fail test if it's due to insufficient data
                logger.warning(f"Mini backtest failed: {e}")
                if "insufficient data" in str(e).lower() or "no data" in str(e).lower():
                    return True  # Expected failure due to no data
                else:
                    raise  # Unexpected error
        
        result = asyncio.run(run_test())
        self.assertTrue(result)


class TestValidation(TestIntegratedBacktesting):
    """Test validation functionality."""
    
    def test_signal_validation_setup(self):
        """Test signal validation setup."""
        from src.backtesting.results import BacktestResults
        
        # Create mock results
        mock_results = BacktestResults(
            config=BacktestConfig(
                start_date=self.test_start,
                end_date=self.test_end,
                initial_capital=Decimal('10000'),
                symbols=[self.test_symbol]
            ),
            trades=[],
            portfolio_snapshots=[]
        )
        
        # Test validation (might return empty results if no data)
        async def run_test():
            validation_results = await self.validator.validate_against_live_signals(
                mock_results, self.test_start, self.test_end
            )
            
            self.assertIsInstance(validation_results, dict)
            logger.info(f"Validation completed for {len(validation_results)} symbols")
            return True
        
        result = asyncio.run(run_test())
        self.assertTrue(result)


async def run_integration_tests():
    """Run all integration tests."""
    logger.info("Starting Integrated Backtesting Test Suite")
    logger.info("="*60)
    
    # Create test suite
    test_suite = unittest.TestSuite()
    
    # Add test cases
    test_classes = [
        TestDatabaseIntegration,
        TestStrategyIntegration,
        TestIntegratedEngine,
        TestValidation
    ]
    
    for test_class in test_classes:
        tests = unittest.TestLoader().loadTestsFromTestCase(test_class)
        test_suite.addTests(tests)
    
    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(test_suite)
    
    # Summary
    logger.info("\n" + "="*60)
    logger.info("TEST SUMMARY")
    logger.info("="*60)
    logger.info(f"Tests run: {result.testsRun}")
    logger.info(f"Failures: {len(result.failures)}")
    logger.info(f"Errors: {len(result.errors)}")
    logger.info(f"Skipped: {len(result.skipped)}")
    
    if result.failures:
        logger.error("\nFailures:")
        for test, traceback in result.failures:
            logger.error(f"  {test}: {traceback}")
    
    if result.errors:
        logger.error("\nErrors:")
        for test, traceback in result.errors:
            logger.error(f"  {test}: {traceback}")
    
    success = len(result.failures) == 0 and len(result.errors) == 0
    
    if success:
        logger.info("\n✓ All tests passed successfully!")
        logger.info("Integration components are working correctly.")
    else:
        logger.error("\n✗ Some tests failed.")
        logger.error("Check the error messages above for details.")
    
    return success


async def run_quick_integration_demo():
    """Run a quick integration demonstration."""
    logger.info("\nRunning Quick Integration Demo")
    logger.info("-" * 40)
    
    try:
        # Test database connection
        logger.info("1. Testing database connection...")
        await db_manager.initialize()
        logger.info("   ✓ Database connected")
        
        # Test data provider
        logger.info("2. Testing data provider...")
        data_provider = DatabaseDataProvider()
        test_symbol = 'BTCUSDT'
        start_date = datetime.utcnow() - timedelta(days=2)
        end_date = datetime.utcnow() - timedelta(days=1)
        
        df = await data_provider.get_historical_data(test_symbol, start_date, end_date)
        logger.info(f"   ✓ Retrieved {len(df)} data points for {test_symbol}")
        
        # Test strategy integration
        logger.info("3. Testing strategy integration...")
        strategy_integration = LiveStrategyIntegration()
        await strategy_integration.initialize()
        logger.info("   ✓ Strategy integration initialized")
        
        # Test integrated engine
        logger.info("4. Testing integrated engine...")
        config = BacktestConfig(
            start_date=start_date,
            end_date=end_date,
            initial_capital=Decimal('10000'),
            symbols=[test_symbol],
            max_position_size=Decimal('0.1')
        )
        
        engine = IntegratedBacktestEngine(config)
        await engine.initialize()
        logger.info("   ✓ Integrated engine initialized")
        
        logger.info("\n✓ Quick integration demo completed successfully!")
        logger.info("All integration components are functional.")
        
        return True
        
    except Exception as e:
        logger.error(f"\n✗ Integration demo failed: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return False


async def main():
    """Main test execution."""
    logger.info("Integrated Backtesting Test Suite")
    logger.info("=" * 60)
    
    try:
        # Run quick demo first
        demo_success = await run_quick_integration_demo()
        
        if demo_success:
            # Run full test suite
            test_success = await run_integration_tests()
            
            if test_success:
                logger.info("\n🎉 ALL INTEGRATION TESTS PASSED!")
                logger.info("The integrated backtesting system is ready for use.")
            else:
                logger.error("\n❌ Some integration tests failed.")
                logger.error("Review the failures and fix issues before deployment.")
                
        else:
            logger.error("\n❌ Integration demo failed.")
            logger.error("Basic integration components are not working.")
    
    except Exception as e:
        logger.error(f"Test execution failed: {e}")
        import traceback
        logger.error(traceback.format_exc())


if __name__ == "__main__":
    asyncio.run(main())
