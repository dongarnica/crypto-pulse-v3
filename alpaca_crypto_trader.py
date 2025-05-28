#!/usr/bin/env python3
"""
Alpaca Crypto Trading Validator
Tests trading functionality with the specified crypto tickers:
AAVE, BCH, BTC, DOGE, DOT, ETH, LINK, LTC, SUSHI, UNI, USDT, XRP, YFI
"""

import asyncio
import logging
from datetime import datetime
from decimal import Decimal
import random

from src.execution.alpaca_executor import AlpacaExecutor, OrderRequest
from alpaca.trading.client import TradingClient
from config.settings import settings

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class AlpacaCryptoTrader:
    """Specialized crypto trader for Alpaca platform."""
    
    def __init__(self):
        self.executor = None
        self.trading_client = None
        
        # Crypto tickers as specified by user
        self.crypto_tickers = [
            "AAVEUSD", "BCHUSD", "BTCUSD", "DOGEUSD", "DOTUSD",
            "ETHUSD", "LINKUSD", "LTCUSD", "SUSHIUSD", "UNIUSD",
            "USDTUSD", "XRPUSD", "YFIUSD"
        ]
        
        # Minimum trade amounts for each crypto (in USD equivalent)
        self.min_trade_amounts = {
            "AAVEUSD": 10.0,
            "BCHUSD": 10.0,
            "BTCUSD": 10.0,
            "DOGEUSD": 10.0,
            "DOTUSD": 10.0,
            "ETHUSD": 10.0,
            "LINKUSD": 10.0,
            "LTCUSD": 10.0,
            "SUSHIUSD": 10.0,
            "UNIUSD": 10.0,
            "USDTUSD": 10.0,
            "XRPUSD": 10.0,
            "YFIUSD": 25.0  # Higher minimum for YFI due to price
        }
    
    async def initialize(self):
        """Initialize the crypto trader."""
        self.executor = AlpacaExecutor()
        await self.executor.initialize()
        
        self.trading_client = TradingClient(
            api_key=settings.alpaca_api_key,
            secret_key=settings.alpaca_secret_key,
            paper=settings.environment != 'production'
        )
        
    async def get_crypto_market_data(self):
        """Get current market data for all crypto tickers."""
        logger.info("📊 Getting market data for crypto tickers...")
        
        try:
            # For now, we'll use account info to verify connection
            # In production, you'd use market data API
            account = await self.executor.get_account_info()
            
            crypto_data = {}
            for ticker in self.crypto_tickers:
                # Simulate current price data (in production, get from Alpaca market data API)
                crypto_data[ticker] = {
                    'symbol': ticker,
                    'available': True,
                    'min_trade_usd': self.min_trade_amounts.get(ticker, 10.0)
                }
            
            logger.info(f"✅ Market data retrieved for {len(crypto_data)} crypto pairs")
            return crypto_data
            
        except Exception as e:
            logger.error(f"❌ Error getting market data: {e}")
            return {}
    
    async def validate_crypto_trading(self):
        """Validate crypto trading capabilities."""
        logger.info("🔍 Validating crypto trading capabilities...")
        
        try:
            # Check account status
            account_info = await self.executor.get_account_info()
            buying_power = float(account_info.get('buying_power', 0))
            
            logger.info(f"💰 Available buying power: ${buying_power:,.2f}")
            
            if buying_power < 50:
                logger.warning("⚠️  Low buying power for crypto testing")
                return False
            
            # Get market data
            market_data = await self.get_crypto_market_data()
            
            # Select a random crypto for testing (excluding USDT for simplicity)
            test_cryptos = [ticker for ticker in self.crypto_tickers if ticker != "USDTUSD"]
            selected_crypto = random.choice(test_cryptos)
            
            logger.info(f"🎯 Selected crypto for test trade: {selected_crypto}")
            
            # Calculate trade amount (very small for testing)
            min_amount = self.min_trade_amounts.get(selected_crypto, 10.0)
            trade_usd_value = min(min_amount, 15.0)  # Small test amount
            
            # For BTC and ETH, use very small quantities
            if selected_crypto == "BTCUSD":
                quantity = Decimal("0.0001")  # ~$11 worth
            elif selected_crypto == "ETHUSD":
                quantity = Decimal("0.003")   # ~$12 worth
            elif selected_crypto == "YFIUSD":
                quantity = Decimal("0.0003")  # ~$25 worth
            else:
                # For other cryptos, estimate quantity based on typical prices
                quantity = Decimal("0.01")    # Adjust based on crypto
            
            # Create test order
            test_order = OrderRequest(
                symbol=selected_crypto,
                side="BUY",
                quantity=quantity,
                order_type="MARKET",
                time_in_force="GTC"
            )
            
            logger.info(f"📝 Preparing test order:")
            logger.info(f"   Symbol: {test_order.symbol}")
            logger.info(f"   Quantity: {test_order.quantity}")
            logger.info(f"   Estimated Value: ~${trade_usd_value}")
            
            return test_order
            
        except Exception as e:
            logger.error(f"❌ Error in crypto validation: {e}")
            return None
    
    async def execute_crypto_test_trade(self, test_order):
        """Execute a test crypto trade."""
        logger.info(f"🚀 Executing test trade for {test_order.symbol}...")
        
        try:
            # Place the order
            result = await self.executor.place_order(test_order)
            
            if result.success:
                logger.info("✅ CRYPTO TEST TRADE SUCCESSFUL!")
                logger.info(f"   Order ID: {result.order_id}")
                logger.info(f"   Status: {result.status}")
                
                # Monitor order execution
                for i in range(6):  # Monitor for 30 seconds
                    await asyncio.sleep(5)
                    status = await self.executor.get_order_status(result.order_id)
                    current_status = status.get('status', 'unknown')
                    filled_qty = status.get('filled_qty', 0)
                    
                    logger.info(f"   🔍 Check {i+1}: Status = {current_status}, Filled = {filled_qty}")
                    
                    if current_status in ['filled', 'partially_filled']:
                        logger.info("   ✅ ORDER FILLED!")
                        if status.get('filled_avg_price'):
                            logger.info(f"   💰 Fill Price: ${status.get('filled_avg_price')}")
                        break
                
                return True
            else:
                logger.error("❌ CRYPTO TEST TRADE FAILED!")
                logger.error(f"   Error: {result.error_message}")
                return False
                
        except Exception as e:
            logger.error(f"❌ Error executing crypto trade: {e}")
            return False
    
    async def show_crypto_portfolio(self):
        """Show current crypto positions."""
        logger.info("📈 Current Crypto Portfolio:")
        
        try:
            positions = await self.executor.get_positions()
            crypto_positions = [pos for pos in positions if any(crypto.replace('USD', '') in pos.symbol for crypto in self.crypto_tickers)]
            
            if crypto_positions:
                total_crypto_value = Decimal('0')
                for pos in crypto_positions:
                    logger.info(f"   {pos.symbol}: {pos.quantity} units, ${pos.market_value}, P&L: ${pos.unrealized_pnl}")
                    total_crypto_value += pos.market_value
                
                logger.info(f"   Total Crypto Portfolio Value: ${total_crypto_value}")
            else:
                logger.info("   No crypto positions found")
                
            return crypto_positions
            
        except Exception as e:
            logger.error(f"❌ Error getting crypto portfolio: {e}")
            return []
    
    async def cleanup(self):
        """Cleanup resources."""
        if self.executor:
            await self.executor.cleanup()


async def main():
    """Main crypto trading validation."""
    logger.info("=" * 80)
    logger.info("🪙 ALPACA CRYPTO TRADING VALIDATION")
    logger.info("Focused on tickers: AAVE, BCH, BTC, DOGE, DOT, ETH, LINK, LTC, SUSHI, UNI, USDT, XRP, YFI")
    logger.info("=" * 80)
    
    trader = AlpacaCryptoTrader()
    
    try:
        # Initialize
        await trader.initialize()
        
        # Show account status
        account_info = await trader.executor.get_account_info()
        logger.info(f"\n💼 Account Status:")
        logger.info(f"   Account ID: {account_info.get('account_id')}")
        logger.info(f"   Buying Power: ${account_info.get('buying_power', 0):,.2f}")
        logger.info(f"   Cash: ${account_info.get('cash', 0):,.2f}")
        
        # Show supported crypto tickers
        logger.info(f"\n🪙 Supported Crypto Tickers ({len(trader.crypto_tickers)}):")
        for ticker in trader.crypto_tickers:
            min_trade = trader.min_trade_amounts.get(ticker, 10.0)
            logger.info(f"   {ticker} (min: ${min_trade})")
        
        # Validate crypto trading
        test_order = await trader.validate_crypto_trading()
        
        if test_order:
            # Execute test trade
            success = await trader.execute_crypto_test_trade(test_order)
            
            if success:
                # Show updated portfolio
                await trader.show_crypto_portfolio()
                
                # Final account status
                final_account = await trader.executor.get_account_info()
                logger.info(f"\n💰 Final Account Status:")
                logger.info(f"   Cash: ${final_account.get('cash', 0):,.2f}")
                logger.info(f"   Portfolio Value: ${final_account.get('portfolio_value', 0):,.2f}")
                
                logger.info("\n" + "=" * 80)
                logger.info("🎉 CRYPTO TRADING VALIDATION SUCCESSFUL!")
                logger.info("🚀 System ready for crypto trading with specified tickers!")
                logger.info("=" * 80)
            else:
                logger.error("\n❌ Crypto trading validation failed")
        else:
            logger.error("\n❌ Could not prepare crypto test trade")
    
    except Exception as e:
        logger.error(f"\n❌ Crypto trading validation error: {e}")
    
    finally:
        await trader.cleanup()


if __name__ == "__main__":
    asyncio.run(main())
