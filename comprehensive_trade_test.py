#!/usr/bin/env python3
"""
Check market status and place a small trade during market hours if needed.
"""

import asyncio
import logging
from datetime import datetime
from decimal import Decimal

from src.execution.alpaca_executor import AlpacaExecutor, OrderRequest
from alpaca.trading.client import TradingClient
from config.settings import settings

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


async def check_market_and_trade():
    """Check market status and place a test trade if market is open."""
    logger.info("Checking market status and executing comprehensive test...")
    
    try:
        # Initialize executor
        executor = AlpacaExecutor()
        await executor.initialize()
        
        # Check market status directly with Alpaca API
        trading_client = TradingClient(
            api_key=settings.alpaca_api_key,
            secret_key=settings.alpaca_secret_key,
            paper=settings.environment != 'production'
        )
        
        # Get market calendar
        calendar = trading_client.get_calendar()
        logger.info(f"Market calendar retrieved: {len(calendar)} days")
        
        # Get clock info
        clock = trading_client.get_clock()
        logger.info(f"Market clock: {clock}")
        logger.info(f"Market is open: {clock.is_open}")
        logger.info(f"Next open: {clock.next_open}")
        logger.info(f"Next close: {clock.next_close}")
        
        # Get account info
        account_info = await executor.get_account_info()
        logger.info(f"Account Status:")
        logger.info(f"  Account ID: {account_info.get('account_id')}")
        logger.info(f"  Buying Power: ${account_info.get('buying_power', 0):,.2f}")
        logger.info(f"  Cash: ${account_info.get('cash', 0):,.2f}")
        logger.info(f"  Portfolio Value: ${account_info.get('portfolio_value', 0):,.2f}")
        
        # Check if we have any pending orders
        orders = trading_client.get_orders()
        logger.info(f"Pending orders: {len(orders)}")
        for order in orders:
            logger.info(f"  Order {order.id}: {order.symbol} {order.side} {order.qty} @ {order.status}")
        
        # Get current positions
        positions = trading_client.get_all_positions()
        logger.info(f"Current positions: {len(positions)}")
        for pos in positions:
            logger.info(f"  Position: {pos.symbol} - Qty: {pos.qty} - Value: ${pos.market_value} - P&L: ${pos.unrealized_pl}")
        
        # If market is closed, let's try a crypto trade (24/7 market)
        if not clock.is_open:
            logger.info("Stock market is closed. Testing with crypto (BTCUSD)...")
            
            crypto_order = OrderRequest(
                symbol="BTCUSD",
                side="BUY",
                quantity=Decimal("0.001"),  # Buy $100 worth (approximately)
                order_type="MARKET",
                time_in_force="GTC"
            )
            
            logger.info(f"Placing crypto test order:")
            logger.info(f"  Symbol: {crypto_order.symbol}")
            logger.info(f"  Side: {crypto_order.side}")
            logger.info(f"  Quantity: {crypto_order.quantity}")
            logger.info(f"  Type: {crypto_order.order_type}")
            
            # Place the crypto order
            result = await executor.place_order(crypto_order)
            
            if result.success:
                logger.info("✅ CRYPTO TEST TRADE SUCCESSFUL!")
                logger.info(f"  Order ID: {result.order_id}")
                logger.info(f"  Status: {result.status}")
                
                # Wait and check status
                await asyncio.sleep(10)
                crypto_status = await executor.get_order_status(result.order_id)
                logger.info(f"Crypto Order Final Status: {crypto_status}")
                
            else:
                logger.error("❌ CRYPTO TEST TRADE FAILED!")
                logger.error(f"  Error: {result.error_message}")
        
        else:
            logger.info("Stock market is open - previous SPY order should execute soon")
        
        # Final account status
        final_account = await executor.get_account_info()
        logger.info(f"Final Account Status:")
        logger.info(f"  Cash: ${final_account.get('cash', 0):,.2f}")
        logger.info(f"  Portfolio Value: ${final_account.get('portfolio_value', 0):,.2f}")
        
        return True
        
    except Exception as e:
        logger.error(f"Error during market check and trade: {e}")
        return False
    finally:
        if 'executor' in locals():
            await executor.cleanup()


if __name__ == "__main__":
    asyncio.run(check_market_and_trade())
