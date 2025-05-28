#!/usr/bin/env python3
"""
Show all available crypto assets on Alpaca and execute a trade with BTC.
"""

import asyncio
import logging
from decimal import Decimal

from src.execution.alpaca_executor import AlpacaExecutor, OrderRequest
from alpaca.trading.client import TradingClient
from alpaca.trading.requests import GetAssetsRequest
from alpaca.trading.enums import AssetClass, AssetStatus
from config.settings import settings

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


async def show_all_crypto_and_trade():
    """Show all available crypto assets and execute a BTC trade."""
    logger.info("=" * 80)
    logger.info("🔍 ALPACA CRYPTO ASSETS DISCOVERY & TRADING")
    logger.info("=" * 80)
    
    try:
        # Initialize executor
        executor = AlpacaExecutor()
        await executor.initialize()
        
        # Initialize trading client
        trading_client = TradingClient(
            api_key=settings.alpaca_api_key,
            secret_key=settings.alpaca_secret_key,
            paper=settings.environment != 'production'
        )
        
        # Get all crypto assets
        logger.info("📋 Fetching all crypto assets from Alpaca...")
        
        crypto_request = GetAssetsRequest(
            asset_class=AssetClass.CRYPTO,
            status=AssetStatus.ACTIVE
        )
        crypto_assets = trading_client.get_all_assets(crypto_request)
        
        logger.info(f"✅ Found {len(crypto_assets)} active crypto assets:")
        
        # Show first 20 crypto assets to see the pattern
        logger.info("\n📝 Sample of available crypto assets:")
        for i, asset in enumerate(crypto_assets[:20]):
            logger.info(f"   {i+1:2d}. {asset.symbol} - {asset.name} (Tradable: {asset.tradable})")
        
        if len(crypto_assets) > 20:
            logger.info(f"   ... and {len(crypto_assets) - 20} more")
        
        # Look for Bitcoin specifically
        btc_assets = [asset for asset in crypto_assets if 'BTC' in asset.symbol and asset.tradable]
        
        logger.info(f"\n🪙 Bitcoin-related assets ({len(btc_assets)}):")
        for asset in btc_assets:
            logger.info(f"   {asset.symbol} - {asset.name}")
        
        # Look for our target cryptos
        target_symbols = ['AAVE', 'BCH', 'BTC', 'DOGE', 'DOT', 'ETH', 'LINK', 'LTC', 'SUSHI', 'UNI', 'USDT', 'XRP', 'YFI']
        found_targets = []
        
        logger.info(f"\n🎯 Searching for target cryptos:")
        for symbol in target_symbols:
            matching_assets = [asset for asset in crypto_assets if symbol in asset.symbol and asset.tradable]
            if matching_assets:
                found_targets.extend(matching_assets)
                for asset in matching_assets:
                    logger.info(f"   ✅ {asset.symbol} - {asset.name}")
            else:
                logger.info(f"   ❌ {symbol} - Not found")
        
        # Select BTC for trading (should be available)
        if btc_assets:
            selected_btc = btc_assets[0]  # Use first BTC asset
            
            logger.info(f"\n🎯 Selected for trading: {selected_btc.symbol}")
            
            # Show account status
            account_info = await executor.get_account_info()
            logger.info(f"\n💼 Account Status:")
            logger.info(f"   Buying Power: ${account_info.get('buying_power', 0):,.2f}")
            
            # Show current positions
            positions = await executor.get_positions()
            if positions:
                logger.info(f"\n📊 Current Positions:")
                for pos in positions:
                    logger.info(f"   {pos.symbol}: {pos.quantity} units, ${pos.market_value}, P&L: ${pos.unrealized_pnl}")
            
            # Execute BTC trade
            btc_quantity = Decimal("0.0001")  # Small amount for testing
            
            logger.info(f"\n🚀 Executing {selected_btc.symbol} trade:")
            logger.info(f"   Quantity: {btc_quantity}")
            logger.info(f"   Expected Value: ~$11")
            
            btc_order = OrderRequest(
                symbol=selected_btc.symbol,
                side="BUY",
                quantity=btc_quantity,
                order_type="MARKET",
                time_in_force="GTC"
            )
            
            result = await executor.place_order(btc_order)
            
            if result.success:
                logger.info("✅ CRYPTO TRADE SUCCESSFUL!")
                logger.info(f"   Order ID: {result.order_id}")
                logger.info(f"   Status: {result.status}")
                
                # Monitor execution
                logger.info(f"\n🔍 Monitoring execution...")
                for i in range(6):
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
                
                # Show final summary
                logger.info("\n" + "=" * 80)
                logger.info("🎉 ALPACA CRYPTO TRADING SUCCESSFUL!")
                logger.info(f"✅ Successfully traded {selected_btc.symbol}")
                logger.info(f"📋 Total crypto assets available: {len(crypto_assets)}")
                logger.info(f"🎯 Target cryptos found: {len(found_targets)}")
                logger.info("=" * 80)
                
                return True
            else:
                logger.error("❌ TRADE FAILED!")
                logger.error(f"   Error: {result.error_message}")
                return False
        else:
            logger.error("❌ No Bitcoin assets found!")
            return False
    
    except Exception as e:
        logger.error(f"❌ Error: {e}")
        return False
    
    finally:
        if 'executor' in locals():
            await executor.cleanup()


if __name__ == "__main__":
    asyncio.run(show_all_crypto_and_trade())
