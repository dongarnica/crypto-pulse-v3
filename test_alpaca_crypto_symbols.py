#!/usr/bin/env python3
"""
Test script to check actual Alpaca crypto symbol formats
"""

import asyncio
import os
from alpaca.trading.client import TradingClient
from alpaca.data import CryptoHistoricalDataClient
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

async def test_alpaca_crypto_symbols():
    """Test what crypto symbols Alpaca actually uses."""
    
    try:
        # Initialize trading client
        trading_client = TradingClient(
            api_key=os.getenv('ALPACA_API_KEY'),
            secret_key=os.getenv('ALPACA_SECRET_KEY'),
            paper=True  # Use paper trading for testing
        )
        
        # Initialize data client for crypto
        crypto_client = CryptoHistoricalDataClient(
            api_key=os.getenv('ALPACA_API_KEY'),
            secret_key=os.getenv('ALPACA_SECRET_KEY')
        )
        
        print("=== Testing Alpaca Trading Client ===")
        
        # Get account info
        account = trading_client.get_account()
        print(f"Account ID: {account.id}")
        print(f"Buying Power: ${account.buying_power}")
        
        # Try to get assets (this might show us available trading symbols)
        print("\n=== Testing Asset Retrieval ===")
        try:
            assets = trading_client.get_all_assets()
            crypto_assets = [asset for asset in assets if asset.asset_class == 'crypto']
            
            print(f"Found {len(crypto_assets)} crypto assets:")
            
            # Filter for our target cryptos
            target_symbols = ['BTC', 'ETH', 'DOGE', 'AAVE', 'BCH', 'DOT', 'LINK', 'LTC', 'SUSHI', 'UNI', 'USDT', 'XRP', 'YFI']
            
            relevant_assets = []
            for asset in crypto_assets:
                # Check if any of our target symbols are in the asset symbol
                for target in target_symbols:
                    if target in asset.symbol and 'USD' in asset.symbol:
                        relevant_assets.append(asset)
                        break
            
            print(f"\nRelevant crypto assets for our targets:")
            for asset in relevant_assets[:20]:  # Show first 20
                print(f"  Symbol: {asset.symbol}, Name: {asset.name}, Tradable: {asset.tradable}")
                
        except Exception as e:
            print(f"Error getting assets: {e}")
        
        # Test some specific symbol formats
        print(f"\n=== Testing Specific Symbol Formats ===")
        
        test_symbols = [
            "BTC/USD",
            "BTCUSD", 
            "ETH/USD",
            "ETHUSD",
            "DOGE/USD",
            "DOGEUSD"
        ]
        
        for symbol in test_symbols:
            try:
                # Try to get recent trades for this symbol
                from alpaca.data.requests import CryptoBarsRequest
                from alpaca.data.timeframe import TimeFrame
                from datetime import datetime, timedelta
                
                request = CryptoBarsRequest(
                    symbol_or_symbols=[symbol],
                    timeframe=TimeFrame.Hour,
                    start=datetime.now() - timedelta(hours=24),
                    end=datetime.now()
                )
                
                bars = crypto_client.get_crypto_bars(request)
                if bars and len(bars) > 0:
                    print(f"✓ Symbol '{symbol}' is valid - found {len(bars)} bars")
                else:
                    print(f"✗ Symbol '{symbol}' returned no data")
                    
            except Exception as e:
                print(f"✗ Symbol '{symbol}' failed: {e}")
                
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    asyncio.run(test_alpaca_crypto_symbols())
