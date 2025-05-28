#!/usr/bin/env python3
"""
Test the updated _get_current_price method in trading engine.
"""

import sys
import os
import asyncio
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def write_test_result(message):
    """Write test result to a file so we can see it."""
    with open("/workspaces/crypto-pulse-v3/test_result.txt", "a") as f:
        f.write(f"{message}\n")

async def test_price_methods():
    write_test_result("Starting price method tests...")
    
    try:
        from src.core.trading_engine import TradingEngine
        from src.data.binance_stream import data_streamer
        
        write_test_result("✅ Imports successful")
        
        # Initialize data streamer
        await data_streamer.initialize()
        write_test_result("✅ Data streamer initialized")
        
        # Create trading engine
        engine = TradingEngine()
        write_test_result("✅ Trading engine created")
        
        # Test symbol conversion
        test_conversions = [
            ("BTC/USD", "BTCUSDT"),
            ("ETH/USD", "ETHUSDT"),
            ("BTCUSDT", "BTCUSDT")
        ]
        
        for input_sym, expected in test_conversions:
            result = engine._convert_to_binance_symbol(input_sym)
            status = "✅" if result == expected else "❌"
            write_test_result(f"{status} {input_sym} → {result} (expected: {expected})")
        
        # Test price retrieval
        test_symbols = ["BTCUSDT", "ETHUSDT"]
        
        for symbol in test_symbols:
            try:
                price = await engine._get_current_price(symbol)
                if price and price > 0:
                    write_test_result(f"✅ {symbol}: ${price:.2f}")
                else:
                    write_test_result(f"❌ {symbol}: No valid price returned")
            except Exception as e:
                write_test_result(f"❌ {symbol}: Error - {e}")
        
        write_test_result("✅ Test completed successfully!")
        
    except Exception as e:
        write_test_result(f"❌ Test failed: {e}")
        import traceback
        write_test_result(f"Traceback: {traceback.format_exc()}")
    
    finally:
        try:
            await data_streamer.stop_all_streams()
        except:
            pass

if __name__ == "__main__":
    # Clear previous results
    with open("/workspaces/crypto-pulse-v3/test_result.txt", "w") as f:
        f.write("Test Results:\n" + "="*50 + "\n")
    
    asyncio.run(test_price_methods())
