#!/usr/bin/env python3
"""
Simple test to verify crypto ticker configuration and basic functionality.
"""

import asyncio
import logging
from src.execution.alpaca_executor import AlpacaExecutor
from config.settings import settings

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

async def test_crypto_config():
    """Test crypto configuration."""
    print("=" * 60)
    print("CRYPTO TRADING CONFIGURATION TEST")
    print("=" * 60)
    
    print(f"Alpaca Crypto Pairs ({len(settings.alpaca_crypto_pairs)}):")
    for i, ticker in enumerate(settings.alpaca_crypto_pairs, 1):
        print(f"  {i:2d}. {ticker}")
    
    print(f"\nBinance Trading Pairs ({len(settings.trading_pairs)}):")
    for i, ticker in enumerate(settings.trading_pairs, 1):
        print(f"  {i:2d}. {ticker}")
    
    # Test Alpaca connection
    print("\nTesting Alpaca connection...")
    try:
        executor = AlpacaExecutor()
        await executor.initialize()
        
        account_info = await executor.get_account_info()
        print(f"✅ Connected to Alpaca successfully!")
        print(f"   Account ID: {account_info.get('account_id')}")
        print(f"   Buying Power: ${account_info.get('buying_power', 0):,.2f}")
        
        await executor.cleanup()
        
    except Exception as e:
        print(f"❌ Alpaca connection failed: {e}")
    
    print("\n" + "=" * 60)
    print("CONFIGURATION TEST COMPLETE")
    print("=" * 60)

if __name__ == "__main__":
    asyncio.run(test_crypto_config())
