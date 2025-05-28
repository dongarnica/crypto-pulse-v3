"""
Quick test script for the backtesting system.
"""

import asyncio
import logging
import sys
import os
from datetime import datetime, timedelta
from decimal import Decimal

# Add the src directory to the path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from backtesting.config import BacktestConfig
from backtesting.engine import BacktestEngine, BacktestPortfolio
from backtesting.results import TradeResult

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def test_portfolio():
    """Test portfolio functionality."""
    logger.info("Testing BacktestPortfolio...")
    
    portfolio = BacktestPortfolio(Decimal('100000'))
    
    # Test initial state
    assert portfolio.cash == Decimal('100000')
    assert len(portfolio.positions) == 0
    
    # Test buy trade
    buy_trade = TradeResult(
        trade_id="test_buy_1",
        symbol="BTC/USD",
        timestamp=datetime.now(),
        side="BUY",
        quantity=Decimal('1.0'),
        price=Decimal('50000'),
        commission=Decimal('50'),
        slippage=Decimal('25')
    )
    
    success = portfolio.execute_trade(buy_trade, 50000.0)
    assert success
    assert portfolio.positions["BTC/USD"] == Decimal('1.0')
    assert portfolio.cash == Decimal('49925')  # 100000 - 50000 - 50 - 25
    
    # Test sell trade
    sell_trade = TradeResult(
        trade_id="test_sell_1",
        symbol="BTC/USD",
        timestamp=datetime.now(),
        side="SELL",
        quantity=Decimal('0.5'),
        price=Decimal('55000'),
        commission=Decimal('27.5'),
        slippage=Decimal('13.75')
    )
    
    success = portfolio.execute_trade(sell_trade, 55000.0)
    assert success
    assert portfolio.positions["BTC/USD"] == Decimal('0.5')
    
    # Test portfolio value calculation
    current_prices = {"BTC/USD": 60000.0}
    total_value = portfolio.get_total_value(current_prices)
    expected_value = portfolio.cash + Decimal('0.5') * Decimal('60000')
    assert abs(total_value - expected_value) < Decimal('0.01')
    
    logger.info("Portfolio tests passed!")


def test_backtest_config():
    """Test backtest configuration."""
    logger.info("Testing BacktestConfig...")
    
    config = BacktestConfig(
        start_date=datetime(2023, 1, 1),
        end_date=datetime(2023, 6, 30),
        initial_capital=Decimal('100000'),
        symbols=['BTC/USD', 'ETH/USD'],
        max_portfolio_allocation=0.20
    )
    
    assert config.start_date == datetime(2023, 1, 1)
    assert config.end_date == datetime(2023, 6, 30)
    assert config.initial_capital == Decimal('100000')
    assert len(config.symbols) == 2
    assert config.max_portfolio_allocation == 0.20
    
    logger.info("Configuration tests passed!")


async def test_backtest_engine_initialization():
    """Test backtest engine initialization."""
    logger.info("Testing BacktestEngine initialization...")
    
    config = BacktestConfig(
        start_date=datetime(2023, 1, 1),
        end_date=datetime(2023, 1, 31),
        initial_capital=Decimal('100000'),
        symbols=['BTC/USD'],
        analysis_interval_minutes=60
    )
    
    engine = BacktestEngine(config)
    
    assert engine.config == config
    assert engine.portfolio.initial_capital == Decimal('100000')
    assert engine.current_date == config.start_date
    assert engine.signals_generated == 0
    assert engine.trades_executed == 0
    
    logger.info("Engine initialization tests passed!")


async def main():
    """Run all tests."""
    logger.info("Starting backtesting system tests...")
    
    try:
        test_portfolio()
        test_backtest_config()
        await test_backtest_engine_initialization()
        
        logger.info("All tests passed! Backtesting system is ready.")
        
    except Exception as e:
        logger.error(f"Test failed: {e}")
        raise


if __name__ == "__main__":
    asyncio.run(main())
