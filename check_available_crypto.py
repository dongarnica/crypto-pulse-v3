#!/usr/bin/env python3
"""
Check available crypto assets on Alpaca and execute a trade with a valid symbol.
"""

import asyncio
import logging
from decimal import Decimal
from datetime import datetime

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


async def check_available_crypto_and_trade():
    """Check available crypto assets and execute a valid trade."""
    logger.info("=" * 80)
    logger.info("🔍 CHECKING AVAILABLE ALPACA CRYPTO ASSETS")
    logger.info("=" * 80)
    
    try:
        # Initialize executor
        executor = AlpacaExecutor()
        await executor.initialize()
        
        # Initialize trading client directly
        trading_client = TradingClient(
            api_key=settings.alpaca_api_key,
            secret_key=settings.alpaca_secret_key,
            paper=settings.environment != 'production'
        )
        
        # Get all crypto assets
        logger.info("📋 Fetching available crypto assets from Alpaca...")
        
        try:
            # Get crypto assets
            crypto_request = GetAssetsRequest(
                asset_class=AssetClass.CRYPTO,
                status=AssetStatus.ACTIVE
            )
            crypto_assets = trading_client.get_all_assets(crypto_request)
            
            logger.info(f"✅ Found {len(crypto_assets)} active crypto assets:")
            
            # Filter and display relevant crypto assets
            target_cryptos = ['AAVE', 'BCH', 'BTC', 'DOGE', 'DOT', 'ETH', 'LINK', 'LTC', 'SUSHI', 'UNI', 'USDT', 'XRP', 'YFI']
            available_targets = []
            
            for asset in crypto_assets:
                # Check if this is one of our target cryptos
                crypto_name = asset.symbol.replace('USD', '').replace('USDT', '')
                if crypto_name in target_cryptos:
                    available_targets.append(asset)
                    logger.info(f"   ✅ {asset.symbol} - {asset.name} (Tradable: {asset.tradable})")
            
            logger.info(f"\n🎯 Available Target Cryptos: {len(available_targets)}")
            
            if not available_targets:
                logger.error("❌ No target crypto assets found!")
                return False
            
            # Select Bitcoin (most liquid and reliable)
            btc_asset = None
            for asset in available_targets:
                if 'BTC' in asset.symbol and asset.tradable:
                    btc_asset = asset
                    break
            
            if not btc_asset:
                logger.error("❌ BTC not found in available assets!")
                return False
            
            logger.info(f"\n🎯 Selected Asset: {btc_asset.symbol}")
            logger.info(f"   Name: {btc_asset.name}")
            logger.info(f"   Tradable: {btc_asset.tradable}")
            logger.info(f"   Min Trade Increment: {getattr(btc_asset, 'min_trade_increment', 'N/A')}")
            
            # Show account status
            account_info = await executor.get_account_info()
            logger.info(f"\n💼 Account Status:")
            logger.info(f"   Buying Power: ${account_info.get('buying_power', 0):,.2f}")
            logger.info(f"   Portfolio Value: ${account_info.get('portfolio_value', 0):,.2f}")
            
            # Show current positions
            positions = await executor.get_positions()
            if positions:
                logger.info(f"\n📊 Current Positions:")
                for pos in positions:
                    logger.info(f"   {pos.symbol}: {pos.quantity} units, ${pos.market_value}, P&L: ${pos.unrealized_pnl}")
            
            # Create a small BTC order
            btc_quantity = Decimal("0.0001")  # ~$11 worth
            
            logger.info(f"\n🚀 Executing BTC trade:")
            logger.info(f"   Symbol: {btc_asset.symbol}")
            logger.info(f"   Quantity: {btc_quantity} BTC")
            logger.info(f"   Expected Value: ~$11")
            
            btc_order = OrderRequest(
                symbol=btc_asset.symbol,
                side="BUY",
                quantity=btc_quantity,
                order_type="MARKET",
                time_in_force="GTC"
            )
            
            # Execute the trade
            result = await executor.place_order(btc_order)
            
            if result.success:
                logger.info("✅ BTC TRADE SUCCESSFUL!")
                logger.info(f"   Order ID: {result.order_id}")
                logger.info(f"   Status: {result.status}")
                
                # Monitor execution
                logger.info(f"\n🔍 Monitoring order execution...")
                for i in range(8):
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
                
                # Final portfolio
                final_positions = await executor.get_positions()
                crypto_positions = [pos for pos in final_positions if 'USD' in pos.symbol]
                
                logger.info(f"\n📈 Final Crypto Portfolio:")
                total_crypto_value = Decimal('0')
                for pos in crypto_positions:
                    logger.info(f"   {pos.symbol}: {pos.quantity} units, ${pos.market_value}, P&L: ${pos.unrealized_pnl}")
                    total_crypto_value += pos.market_value
                
                logger.info(f"   Total Crypto Value: ${total_crypto_value}")
                
                # Final account
                final_account = await executor.get_account_info()
                logger.info(f"\n💰 Final Account:")
                logger.info(f"   Cash: ${final_account.get('cash', 0):,.2f}")
                logger.info(f"   Portfolio Value: ${final_account.get('portfolio_value', 0):,.2f}")
                
                logger.info("\n" + "=" * 80)
                logger.info("🎉 ALPACA CRYPTO TRADING VALIDATION COMPLETE!")
                logger.info("✅ Successfully traded with available crypto assets!")
                logger.info("=" * 80)
                
                return True
            else:
                logger.error("❌ BTC TRADE FAILED!")
                logger.error(f"   Error: {result.error_message}")
                return False
        
        except Exception as e:
            logger.error(f"❌ Error fetching crypto assets: {e}")
            logger.info("📝 Will proceed with known working symbol (BTCUSD)...")
            
            # Fallback to BTCUSD which we know works
            btc_order = OrderRequest(
                symbol="BTCUSD",
                side="BUY",
                quantity=Decimal("0.0001"),
                order_type="MARKET",
                time_in_force="GTC"
            )
            
            result = await executor.place_order(btc_order)
            
            if result.success:
                logger.info("✅ FALLBACK BTC TRADE SUCCESSFUL!")
                logger.info(f"   Order ID: {result.order_id}")
                return True
            else:
                logger.error(f"❌ Fallback trade failed: {result.error_message}")
                return False
    
    except Exception as e:
        logger.error(f"❌ Error in crypto validation: {e}")
        return False
    
    finally:
        if 'executor' in locals():
            await executor.cleanup()


if __name__ == "__main__":
    asyncio.run(check_available_crypto_and_trade())
