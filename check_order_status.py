#!/usr/bin/env python3
"""
Check the status of the recently placed order and monitor for fills.
"""

import asyncio
import logging
from datetime import datetime

from src.execution.alpaca_executor import AlpacaExecutor
from config.settings import settings

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


async def check_order_status():
    """Check the status of the recent order."""
    logger.info("Checking order status...")
    
    try:
        # Initialize executor
        executor = AlpacaExecutor()
        await executor.initialize()
        
        # The order ID from the previous execution
        order_id = "7cf49087-6a31-4cde-90fc-7973d215f10c"
        
        # Check order status
        order_status = await executor.get_order_status(order_id)
        logger.info(f"Order Status: {order_status}")
        
        # Get current positions to see if we now own SPY
        positions = await executor.get_positions()
        logger.info(f"Current positions: {len(positions)}")
        
        for pos in positions:
            logger.info(f"Position: {pos.symbol} - Qty: {pos.quantity} - Value: ${pos.market_value} - P&L: ${pos.unrealized_pnl}")
        
        # Get account info to see updated cash balance
        account_info = await executor.get_account_info()
        logger.info(f"Updated account info:")
        logger.info(f"  Cash: ${account_info.get('cash', 0):,.2f}")
        logger.info(f"  Portfolio Value: ${account_info.get('portfolio_value', 0):,.2f}")
        
    except Exception as e:
        logger.error(f"Error checking order status: {e}")
    finally:
        if 'executor' in locals():
            await executor.cleanup()


if __name__ == "__main__":
    asyncio.run(check_order_status())
