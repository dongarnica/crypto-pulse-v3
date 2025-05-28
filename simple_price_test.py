#!/usr/bin/env python3

import asyncio
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

async def simple_test():
    try:
        from src.data.binance_stream import data_streamer
        print("Initializing...")
        await data_streamer.initialize()
        print("Getting price...")
        price = await data_streamer.get_current_price('BTCUSDT')
        print(f"BTC price: {price}")
        return price
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        return None

if __name__ == "__main__":
    result = asyncio.run(simple_test())
    print(f"Final result: {result}")
