#!/usr/bin/env python3
"""
Execute a focused crypto trade using one of the specified Alpaca crypto tickers.
Target tickers: AAVE, BCH, BTC, DOGE, DOT, ETH, LINK, LTC, SUSHI, UNI, USDT, XRP, YFI
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


async def execute_focused_crypto_trade():
    """Execute a crypto trade with specified tickers."""
    logger.info("=" * 70)
    logger.info("🪙 FOCUSED ALPACA CRYPTO TRADING")
    logger.info("=" * 70)
    
    try:
        # Initialize executor
        executor = AlpacaExecutor()
        await executor.initialize()
        
        # Show account status
        account_info = await executor.get_account_info()
        logger.info(f"\n💼 Account Status:")
        logger.info(f"   Account ID: {account_info.get('account_id')}")
        logger.info(f"   Buying Power: ${account_info.get('buying_power', 0):,.2f}")
        logger.info(f"   Cash: ${account_info.get('cash', 0):,.2f}")
        
        # Display available crypto tickers
        logger.info(f"\n🪙 Available Crypto Tickers ({len(settings.alpaca_crypto_pairs)}):")
        for i, ticker in enumerate(settings.alpaca_crypto_pairs, 1):
            logger.info(f"   {i:2d}. {ticker}")
        
        # Check current crypto positions
        logger.info(f"\n📊 Current Crypto Positions:")
        positions = await executor.get_positions()
        crypto_positions = [pos for pos in positions if pos.symbol in settings.alpaca_crypto_pairs]
        
        if crypto_positions:
            for pos in crypto_positions:
                logger.info(f"   {pos.symbol}: {pos.quantity} units, ${pos.market_value}, P&L: ${pos.unrealized_pnl}")
        else:
            logger.info("   No crypto positions found")
        
        # Select Ethereum for this test (good liquidity, reasonable price)
        selected_crypto = "ETHUSD"
        trade_quantity = Decimal("0.003")  # Small amount for testing
        
        logger.info(f"\n🎯 Selected Crypto: {selected_crypto}")
        logger.info(f"📝 Trade Details:")
        logger.info(f"   Symbol: {selected_crypto}")
        logger.info(f"   Side: BUY")
        logger.info(f"   Quantity: {trade_quantity} ETH")
        logger.info(f"   Order Type: MARKET")
        
        # Create order request
        crypto_order = OrderRequest(
            symbol=selected_crypto,
            side="BUY",
            quantity=trade_quantity,
            order_type="MARKET",
            time_in_force="GTC"
        )
        
        # Execute the trade
        logger.info(f"\n🚀 Executing crypto trade...")
        result = await executor.place_order(crypto_order)
        
        if result.success:
            logger.info("✅ CRYPTO TRADE SUCCESSFUL!")
            logger.info(f"   Order ID: {result.order_id}")
            logger.info(f"   Status: {result.status}")
            
            # Monitor order execution
            logger.info(f"\n🔍 Monitoring order execution...")
            for i in range(8):  # Monitor for 40 seconds
                await asyncio.sleep(5)
                status = await executor.get_order_status(result.order_id)
                current_status = status.get('status', 'unknown')
                filled_qty = status.get('filled_qty', 0)
                
                logger.info(f"   Check {i+1}: Status = {current_status}, Filled = {filled_qty}")
                
                if current_status in ['filled', 'partially_filled']:
                    logger.info("   ✅ ORDER FILLED!")
                    if status.get('filled_avg_price'):
                        fill_price = status.get('filled_avg_price')
                        trade_value = float(filled_qty) * float(fill_price)
                        logger.info(f"   💰 Fill Price: ${fill_price:,.2f}")
                        logger.info(f"   💰 Trade Value: ${trade_value:,.2f}")
                    break
            
            # Show updated positions
            logger.info(f"\n📈 Updated Crypto Positions:")
            updated_positions = await executor.get_positions()
            crypto_positions = [pos for pos in updated_positions if pos.symbol in settings.alpaca_crypto_pairs]
            
            if crypto_positions:
                total_crypto_value = Decimal('0')
                for pos in crypto_positions:
                    logger.info(f"   {pos.symbol}: {pos.quantity} units, ${pos.market_value}, P&L: ${pos.unrealized_pnl}")
                    total_crypto_value += pos.market_value
                logger.info(f"   Total Crypto Value: ${total_crypto_value}")
            else:
                logger.info("   No crypto positions found")
            
            # Final account status
            final_account = await executor.get_account_info()
            logger.info(f"\n💰 Final Account Status:")
            logger.info(f"   Cash: ${final_account.get('cash', 0):,.2f}")
            logger.info(f"   Portfolio Value: ${final_account.get('portfolio_value', 0):,.2f}")
            
            logger.info("\n" + "=" * 70)
            logger.info("🎉 FOCUSED CRYPTO TRADING SUCCESSFUL!")
            logger.info("✅ System validated with specified crypto tickers!")
            logger.info("=" * 70)
            
            return True
            
        else:
            logger.error("❌ CRYPTO TRADE FAILED!")
            logger.error(f"   Error: {result.error_message}")
            return False
    
    except Exception as e:
        logger.error(f"❌ Error in crypto trading: {e}")
        return False
    
    finally:
        if 'executor' in locals():
            await executor.cleanup()


if __name__ == "__main__":
    asyncio.run(execute_focused_crypto_trade())
