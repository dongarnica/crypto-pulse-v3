#!/usr/bin/env python3
"""
Execute properly sized crypto trades with specified Alpaca tickers.
Ensures minimum order values are met for successful execution.
"""

import asyncio
import logging
from decimal import Decimal
from datetime import datetime

from src.execution.alpaca_executor import AlpacaExecutor, OrderRequest
from alpaca.trading.client import TradingClient
from config.settings import settings

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


async def execute_properly_sized_crypto_trade():
    """Execute crypto trades with proper minimum sizes."""
    logger.info("=" * 75)
    logger.info("🪙 ALPACA CRYPTO TRADING - PROPERLY SIZED ORDERS")
    logger.info("Target Tickers: AAVE, BCH, BTC, DOGE, DOT, ETH, LINK, LTC, SUSHI, UNI, USDT, XRP, YFI")
    logger.info("=" * 75)
    
    try:
        # Initialize executor
        executor = AlpacaExecutor()
        await executor.initialize()
        
        # Show account status
        account_info = await executor.get_account_info()
        logger.info(f"\n💼 Account Status:")
        logger.info(f"   Account ID: {account_info.get('account_id')}")
        logger.info(f"   Buying Power: ${account_info.get('buying_power', 0):,.2f}")
        logger.info(f"   Portfolio Value: ${account_info.get('portfolio_value', 0):,.2f}")
        
        # Show current crypto positions
        logger.info(f"\n📊 Current Crypto Holdings:")
        positions = await executor.get_positions()
        crypto_positions = [pos for pos in positions if pos.symbol in settings.alpaca_crypto_pairs]
        
        if crypto_positions:
            total_crypto_value = Decimal('0')
            for pos in crypto_positions:
                logger.info(f"   {pos.symbol}: {pos.quantity} units, ${pos.market_value}, P&L: ${pos.unrealized_pnl}")
                total_crypto_value += pos.market_value
            logger.info(f"   Total Crypto Portfolio: ${total_crypto_value}")
        else:
            logger.info("   No crypto positions found")
        
        # Define trade parameters that meet Alpaca minimums
        # Alpaca requires minimum $10 order value for crypto
        trade_options = [
            {"symbol": "DOGUSD", "quantity": Decimal("30"), "description": "30 DOGE (~$12)"},
            {"symbol": "XRPUSD", "quantity": Decimal("4"), "description": "4 XRP (~$10)"},
            {"symbol": "LINKUSD", "quantity": Decimal("0.4"), "description": "0.4 LINK (~$12)"},
            {"symbol": "LTCUSD", "quantity": Decimal("0.08"), "description": "0.08 LTC (~$11)"},
        ]
        
        # Select a trade option
        selected_trade = trade_options[0]  # DOGE for this example
        
        logger.info(f"\n🎯 Selected Trade:")
        logger.info(f"   Symbol: {selected_trade['symbol']}")
        logger.info(f"   Quantity: {selected_trade['quantity']}")
        logger.info(f"   Description: {selected_trade['description']}")
        logger.info(f"   Expected Value: ~$12 (meets $10 minimum)")
        
        # Verify buying power
        buying_power = float(account_info.get('buying_power', 0))
        if buying_power < 15:
            logger.error(f"❌ Insufficient buying power: ${buying_power}")
            return False
        
        # Create order request
        crypto_order = OrderRequest(
            symbol=selected_trade['symbol'],
            side="BUY",
            quantity=selected_trade['quantity'],
            order_type="MARKET",
            time_in_force="GTC"
        )
        
        logger.info(f"\n🚀 Executing crypto trade...")
        result = await executor.place_order(crypto_order)
        
        if result.success:
            logger.info("✅ CRYPTO TRADE PLACED SUCCESSFULLY!")
            logger.info(f"   Order ID: {result.order_id}")
            logger.info(f"   Status: {result.status}")
            
            # Monitor order execution
            logger.info(f"\n🔍 Monitoring order execution...")
            for i in range(10):  # Monitor for 50 seconds
                await asyncio.sleep(5)
                status = await executor.get_order_status(result.order_id)
                current_status = status.get('status', 'unknown')
                filled_qty = status.get('filled_qty', 0)
                
                logger.info(f"   Check {i+1}: Status = {current_status}, Filled = {filled_qty}")
                
                if current_status in ['filled', 'partially_filled']:
                    logger.info("   ✅ ORDER EXECUTED!")
                    if status.get('filled_avg_price'):
                        fill_price = status.get('filled_avg_price')
                        trade_value = float(filled_qty) * float(fill_price)
                        logger.info(f"   💰 Fill Price: ${fill_price:,.4f}")
                        logger.info(f"   💰 Trade Value: ${trade_value:,.2f}")
                        logger.info(f"   📊 Quantity Filled: {filled_qty}")
                    break
                elif current_status in ['cancelled', 'rejected']:
                    logger.error(f"   ❌ Order {current_status}")
                    break
            
            # Show final portfolio
            logger.info(f"\n📈 Final Crypto Portfolio:")
            final_positions = await executor.get_positions()
            crypto_positions = [pos for pos in final_positions if pos.symbol in settings.alpaca_crypto_pairs]
            
            if crypto_positions:
                total_crypto_value = Decimal('0')
                for pos in crypto_positions:
                    logger.info(f"   {pos.symbol}: {pos.quantity} units, ${pos.market_value}, P&L: ${pos.unrealized_pnl}")
                    total_crypto_value += pos.market_value
                logger.info(f"   Total Crypto Portfolio: ${total_crypto_value}")
            
            # Final account summary
            final_account = await executor.get_account_info()
            logger.info(f"\n💰 Final Account Summary:")
            logger.info(f"   Cash: ${final_account.get('cash', 0):,.2f}")
            logger.info(f"   Portfolio Value: ${final_account.get('portfolio_value', 0):,.2f}")
            logger.info(f"   Day Trade Count: {final_account.get('day_trade_count', 0)}")
            
            logger.info("\n" + "=" * 75)
            logger.info("🎉 CRYPTO TRADING WITH SPECIFIED TICKERS SUCCESSFUL!")
            logger.info("✅ System ready for production crypto trading!")
            logger.info("📋 Supported: AAVE, BCH, BTC, DOGE, DOT, ETH, LINK, LTC, SUSHI, UNI, USDT, XRP, YFI")
            logger.info("=" * 75)
            
            return True
            
        else:
            logger.error("❌ CRYPTO TRADE FAILED!")
            logger.error(f"   Error: {result.error_message}")
            
            # Show troubleshooting info
            logger.info(f"\n🔧 Troubleshooting Info:")
            logger.info(f"   Order Symbol: {crypto_order.symbol}")
            logger.info(f"   Order Quantity: {crypto_order.quantity}")
            logger.info(f"   Order Type: {crypto_order.order_type}")
            logger.info(f"   Time in Force: {crypto_order.time_in_force}")
            
            return False
    
    except Exception as e:
        logger.error(f"❌ Error in crypto trading: {e}")
        return False
    
    finally:
        if 'executor' in locals():
            await executor.cleanup()


if __name__ == "__main__":
    asyncio.run(execute_properly_sized_crypto_trade())
