#!/usr/bin/env python3
"""
Integration script for backtesting with the main trading system.
"""

import asyncio
import logging
import sys
import os
from datetime import datetime, timedelta
from decimal import Decimal

# Add the src directory to the path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_imports():
    """Test all backtesting imports."""
    try:
        from backtesting.config import BacktestConfig, OptimizationConfig
        from backtesting.engine import BacktestEngine, BacktestPortfolio
        from backtesting.results import BacktestResults, TradeResult
        logger.info("✓ All backtesting imports successful")
        return True
    except Exception as e:
        logger.error(f"✗ Import error: {e}")
        return False

if __name__ == "__main__":
    logger.info("Testing backtesting system integration...")
    if test_imports():
        logger.info("Backtesting system is ready for integration!")
    else:
        logger.error("Backtesting system has import issues")
