#!/usr/bin/env python3
"""
Test script to validate corrected crypto trading with proper Alpaca symbol format
"""

import asyncio
import sys
import os
from decimal import Decimal
from datetime import datetime

# Add project root to path
sys.path.append('/workspaces/crypto-pulse-v3')

from src.execution.alpaca_executor import AlpacaExecutor, OrderRequest
from config.settings import settings
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

async def test_corrected_crypto_trading():
    """Test crypto trading with corrected Alpaca symbol format."""
    
    print("=== Testing Corrected Crypto Trading with Alpaca ===")
    print(f"Environment: {settings.environment}")
    print(f"Using Alpaca Paper Trading: {settings.environment != 'production'}")
    
    # Initialize executor
    executor = AlpacaExecutor()
    
    try:
        # Initialize connection
        print("\n1. Initializing Alpaca connection...")
        await executor.initialize()
        
        # Get account info
        print("\n2. Getting account information...")
        account_info = await executor.get_account_info()
        print(f"Account ID: {account_info.get('account_id', 'N/A')}")
        print(f"Buying Power: ${account_info.get('buying_power', 0):,.2f}")
        print(f"Portfolio Value: ${account_info.get('portfolio_value', 0):,.2f}")
        
        # Test the corrected symbol format
        print(f"\n3. Testing corrected crypto symbol formats...")
        print(f"Configured Alpaca crypto pairs: {settings.alpaca_crypto_pairs}")
        
        # Test order creation with correct symbol format
        test_symbols = ["BTC/USD", "ETH/USD", "DOGE/USD"]
        
        for symbol in test_symbols:
            print(f"\n--- Testing {symbol} ---")
            
            # Create a small test order
            order_request = OrderRequest(
                symbol=symbol,
                side='BUY',
                quantity=Decimal('0.0001'),  # Very small amount for testing
                order_type='MARKET',
                time_in_force='GTC'
            )
            
            print(f"Order Details:")
            print(f"  Symbol: {order_request.symbol}")
            print(f"  Side: {order_request.side}")
            print(f"  Quantity: {order_request.quantity}")
            print(f"  Order Type: {order_request.order_type}")
            print(f"  Time in Force: {order_request.time_in_force}")
            
            # Test symbol detection logic
            is_crypto = '/' in order_request.symbol and any(
                order_request.symbol.upper() in pair 
                for pair in settings.alpaca_crypto_pairs
            )
            print(f"  Detected as crypto: {is_crypto}")
            
            # Place the order (only BTC for actual execution to avoid multiple trades)
            if symbol == "BTC/USD":
                print(f"\n  Placing real order for {symbol}...")
                result = await executor.place_order(order_request)
                
                print(f"  Order Result:")
                print(f"    Success: {result.success}")
                print(f"    Order ID: {result.order_id}")
                print(f"    Status: {result.status}")
                print(f"    Error: {result.error_message}")
                
                if result.success and result.order_id:
                    # Wait a moment and check order status
                    print(f"  Waiting 3 seconds and checking order status...")
                    await asyncio.sleep(3)
                    
                    order_status = await executor.get_order_status(result.order_id)
                    print(f"  Order Status Update:")
                    print(f"    Status: {order_status.get('status', 'N/A')}")
                    print(f"    Filled Qty: {order_status.get('filled_qty', 0)}")
                    print(f"    Filled Price: ${order_status.get('filled_avg_price', 0)}")
                    
                    # If filled, this confirms our fix worked!
                    if order_status.get('status') == 'FILLED':
                        print(f"  🎉 SUCCESS! Order filled - crypto trading fix confirmed!")
                        filled_qty = order_status.get('filled_qty', 0)
                        filled_price = order_status.get('filled_avg_price', 0)
                        trade_value = filled_qty * filled_price
                        print(f"  Trade Details: {filled_qty} {symbol.split('/')[0]} @ ${filled_price:,.2f} = ${trade_value:.2f}")
            else:
                print(f"  Skipping actual order placement for {symbol} (testing BTC/USD only)")
        
        # Check current positions
        print(f"\n4. Checking current positions...")
        positions = await executor.get_positions()
        if positions:
            print(f"Current positions ({len(positions)}):")
            for pos in positions:
                print(f"  {pos.symbol}: {pos.quantity} @ ${pos.current_price:.4f} (PnL: ${pos.unrealized_pnl:.2f})")
        else:
            print("No current positions")
        
    except Exception as e:
        print(f"Error during testing: {e}")
        import traceback
        traceback.print_exc()
    finally:
        # Cleanup
        if hasattr(executor, 'cleanup'):
            await executor.cleanup()

if __name__ == "__main__":
    asyncio.run(test_corrected_crypto_trading())
