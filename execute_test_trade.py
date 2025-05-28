#!/usr/bin/env python3
"""
Test script to execute a real trade using the crypto-pulse-v3 trading system.
This will place a small market order to validate live trading functionality.
"""

import asyncio
import logging
from decimal import Decimal
from datetime import datetime

from src.execution.alpaca_executor import AlpacaExecutor, OrderRequest
from src.risk.manager import RiskManager
from config.settings import settings

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


async def execute_test_trade():
    """Execute a small test trade to validate the system."""
    logger.info("Starting test trade execution...")
    
    try:
        # Initialize executor
        executor = AlpacaExecutor()
        await executor.initialize()
        
        # Get account info to verify connection
        account_info = await executor.get_account_info()
        logger.info(f"Account connected successfully:")
        logger.info(f"  Account ID: {account_info.get('account_id')}")
        logger.info(f"  Buying Power: ${account_info.get('buying_power', 0):,.2f}")
        logger.info(f"  Cash: ${account_info.get('cash', 0):,.2f}")
        
        # Check if we have sufficient buying power
        buying_power = float(account_info.get('buying_power', 0))
        if buying_power < 50:  # Need at least $50 for test trade
            logger.error(f"Insufficient buying power: ${buying_power:,.2f}")
            return False
        
        # Define a small test trade
        # Using SPY (S&P 500 ETF) - highly liquid and commonly traded
        test_order = OrderRequest(
            symbol="SPY",
            side="BUY",
            quantity=Decimal("1"),  # Buy 1 share
            order_type="MARKET",
            time_in_force="DAY"
        )
        
        logger.info(f"Placing test order:")
        logger.info(f"  Symbol: {test_order.symbol}")
        logger.info(f"  Side: {test_order.side}")
        logger.info(f"  Quantity: {test_order.quantity}")
        logger.info(f"  Type: {test_order.order_type}")
        
        # Place the order
        result = await executor.place_order(test_order)
        
        if result.success:
            logger.info("✅ TEST TRADE SUCCESSFUL!")
            logger.info(f"  Order ID: {result.order_id}")
            logger.info(f"  Status: {result.status}")
            if result.filled_quantity > 0:
                logger.info(f"  Filled Quantity: {result.filled_quantity}")
                logger.info(f"  Average Fill Price: ${result.average_fill_price}")
            
            # Wait a moment for potential fill
            await asyncio.sleep(5)
            
            # Check order status
            order_status = await executor.get_order_status(result.order_id)
            logger.info(f"Final Order Status: {order_status}")
            
            return True
        else:
            logger.error("❌ TEST TRADE FAILED!")
            logger.error(f"  Error: {result.error_message}")
            return False
            
    except Exception as e:
        logger.error(f"Error during test trade execution: {e}")
        return False
    finally:
        if 'executor' in locals():
            await executor.cleanup()


async def main():
    """Main execution function."""
    logger.info("=" * 60)
    logger.info("CRYPTO-PULSE-V3 LIVE TRADING TEST")
    logger.info("=" * 60)
    
    success = await execute_test_trade()
    
    if success:
        logger.info("=" * 60)
        logger.info("✅ LIVE TRADING VALIDATION COMPLETE - SYSTEM READY")
        logger.info("=" * 60)
    else:
        logger.error("=" * 60)
        logger.error("❌ LIVE TRADING VALIDATION FAILED")
        logger.error("=" * 60)


if __name__ == "__main__":
    asyncio.run(main())
