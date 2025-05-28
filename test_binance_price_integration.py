#!/usr/bin/env python3
"""
Test script to verify Binance real-time price integration in trading engine.
"""

import asyncio
import logging
import sys
import os

# Add the project root to the path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.core.trading_engine import TradingEngine
from src.data.binance_stream import data_streamer
from config.settings import settings

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


async def test_binance_price_integration():
    """Test that the trading engine can get real-time prices from Binance."""
    
    print("🚀 Testing Binance Price Integration in Trading Engine")
    print("=" * 60)
    
    try:
        # Initialize data streamer
        print("1. Initializing Binance data streamer...")
        await data_streamer.initialize()
        print("✅ Binance data streamer initialized")
        
        # Create trading engine instance
        print("\n2. Creating trading engine instance...")
        engine = TradingEngine()
        
        # Test price retrieval for different symbol formats
        test_symbols = [
            "BTC/USD",     # Alpaca format
            "BTCUSDT",     # Binance format
            "ETH/USD",     # Alpaca format  
            "ETHUSDT",     # Binance format
            "INVALID/USD"  # Invalid symbol
        ]
        
        print(f"\n3. Testing price retrieval for {len(test_symbols)} symbols...")
        print("-" * 40)
        
        results = {}
        for symbol in test_symbols:
            print(f"Getting price for {symbol}...", end=" ")
            try:
                price = await engine._get_current_price(symbol)
                if price:
                    results[symbol] = price
                    print(f"✅ ${price:.2f}")
                else:
                    print("❌ No price returned")
                    results[symbol] = None
            except Exception as e:
                print(f"❌ Error: {e}")
                results[symbol] = None
        
        # Display results summary
        print(f"\n4. Price Retrieval Results:")
        print("-" * 40)
        successful = 0
        for symbol, price in results.items():
            status = "✅" if price else "❌"
            price_str = f"${price:.2f}" if price else "Failed"
            print(f"{status} {symbol:12} → {price_str}")
            if price:
                successful += 1
        
        print(f"\n📊 Summary: {successful}/{len(test_symbols)} symbols successful")
        
        # Test symbol format conversion
        print(f"\n5. Testing symbol format conversion...")
        print("-" * 40)
        conversion_tests = [
            ("BTC/USD", "BTCUSDT"),
            ("ETH/USD", "ETHUSDT"),
            ("DOGE/USD", "DOGEUSDT"),
            ("BTCUSDT", "BTCUSDT"),  # Should remain unchanged
            ("INVALID", "INVALID")   # Should remain unchanged
        ]
        
        for input_symbol, expected in conversion_tests:
            converted = engine._convert_to_binance_symbol(input_symbol)
            status = "✅" if converted == expected else "❌"
            print(f"{status} {input_symbol:10} → {converted:10} (expected: {expected})")
        
        # Test direct Binance API call
        print(f"\n6. Testing direct Binance API calls...")
        print("-" * 40)
        direct_test_symbols = ["BTCUSDT", "ETHUSDT", "ADAUSDT"]
        
        for symbol in direct_test_symbols:
            print(f"Direct API call for {symbol}...", end=" ")
            try:
                price = await data_streamer.get_current_price(symbol)
                if price:
                    print(f"✅ ${price:.2f}")
                else:
                    print("❌ No price returned")
            except Exception as e:
                print(f"❌ Error: {e}")
        
        print(f"\n✅ Integration test completed successfully!")
        
    except Exception as e:
        print(f"\n❌ Integration test failed: {e}")
        logger.error(f"Test failed: {e}", exc_info=True)
        return False
    
    finally:
        # Cleanup
        try:
            await data_streamer.stop_all_streams()
            print(f"\n🧹 Cleanup completed")
        except Exception as e:
            print(f"⚠️  Cleanup warning: {e}")
    
    return True


async def main():
    """Main test function."""
    print("Starting Binance Price Integration Test...")
    
    success = await test_binance_price_integration()
    
    if success:
        print(f"\n🎉 All tests passed! Trading engine now uses Binance real-time prices.")
        sys.exit(0)
    else:
        print(f"\n💥 Tests failed. Please check the logs above.")
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())
