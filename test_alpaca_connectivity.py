#!/usr/bin/env python3
"""
Test script to verify Alpaca executor connectivity and functionality.
"""

import asyncio
import logging
from decimal import Decimal
from datetime import datetime

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

async def test_alpaca_connectivity():
    """Test Alpaca executor connectivity and basic functionality."""
    print("=" * 60)
    print("ALPACA EXECUTOR CONNECTIVITY TEST")
    print("=" * 60)
    
    try:
        # Import the executor
        print("\n1. Importing Alpaca executor...")
        from src.execution.alpaca_executor import alpaca_executor, OrderRequest
        print("✅ Alpaca executor imported successfully")
        
        # Test initialization
        print("\n2. Testing Alpaca client initialization...")
        await alpaca_executor.initialize()
        print("✅ Alpaca client initialized successfully")
        
        # Test account info retrieval
        print("\n3. Testing account information retrieval...")
        account_info = await alpaca_executor.get_account_info()
        if account_info:
            print("✅ Account info retrieved successfully:")
            print(f"   Account ID: {account_info.get('account_id', 'N/A')}")
            print(f"   Buying Power: ${account_info.get('buying_power', 0):,.2f}")
            print(f"   Cash: ${account_info.get('cash', 0):,.2f}")
            print(f"   Portfolio Value: ${account_info.get('portfolio_value', 0):,.2f}")
            print(f"   Pattern Day Trader: {account_info.get('pattern_day_trader', False)}")
            print(f"   Trading Blocked: {account_info.get('trading_blocked', False)}")
        else:
            print("❌ Failed to retrieve account info")
            return False
        
        # Test positions retrieval
        print("\n4. Testing positions retrieval...")
        positions = await alpaca_executor.get_positions()
        print(f"✅ Retrieved {len(positions)} positions")
        
        if positions:
            print("   Current positions:")
            for pos in positions[:5]:  # Show first 5 positions
                print(f"   - {pos.symbol}: {pos.side} {pos.quantity} shares @ ${pos.current_price}")
                print(f"     Market Value: ${pos.market_value}, P&L: ${pos.unrealized_pnl}")
        else:
            print("   No active positions found")
        
        # Test order validation (without placing actual orders)
        print("\n5. Testing order risk validation...")
        test_order = OrderRequest(
            symbol="AAPL",
            side="BUY",
            quantity=Decimal('1'),
            order_type="MARKET"
        )
        
        risk_validation = await alpaca_executor._validate_order_risk(test_order)
        if risk_validation.get('allowed'):
            print("✅ Order risk validation passed")
            print(f"   Reason: {risk_validation.get('reason', 'N/A')}")
        else:
            print("⚠️ Order risk validation failed (this is expected for testing):")
            print(f"   Reason: {risk_validation.get('reason', 'N/A')}")
        
        # Test order conversion
        print("\n6. Testing order format conversion...")
        try:
            alpaca_order = alpaca_executor._convert_to_alpaca_order(test_order)
            print("✅ Order conversion successful")
            print(f"   Order type: {type(alpaca_order).__name__}")
            print(f"   Symbol: {alpaca_order.symbol}")
            print(f"   Quantity: {alpaca_order.qty}")
            print(f"   Side: {alpaca_order.side}")
        except Exception as e:
            print(f"❌ Order conversion failed: {e}")
            return False
        
        print("\n" + "=" * 60)
        print("✅ ALL TESTS PASSED - Alpaca executor is ready for trading!")
        print("=" * 60)
        
        return True
        
    except Exception as e:
        print(f"\n❌ TEST FAILED: {e}")
        print("=" * 60)
        import traceback
        traceback.print_exc()
        return False

async def test_database_integration():
    """Test database integration with Alpaca executor."""
    print("\n" + "=" * 60)
    print("DATABASE INTEGRATION TEST")
    print("=" * 60)
    
    try:
        from src.core.database import db_manager
        from src.core.models import Trade, Position
        
        print("\n1. Testing database connection...")
        with db_manager.get_session() as session:
            # Test query
            trade_count = session.query(Trade).count()
            position_count = session.query(Position).count()
            print(f"✅ Database connected successfully")
            print(f"   Trades in database: {trade_count}")
            print(f"   Positions in database: {position_count}")
        
        return True
        
    except Exception as e:
        print(f"❌ Database test failed: {e}")
        return False

if __name__ == "__main__":
    async def main():
        print("Starting Alpaca Executor Test Suite...")
        print(f"Test started at: {datetime.now()}")
        
        # Test Alpaca connectivity
        alpaca_success = await test_alpaca_connectivity()
        
        # Test database integration
        db_success = await test_database_integration()
        
        print("\n" + "=" * 60)
        print("TEST SUMMARY")
        print("=" * 60)
        print(f"Alpaca Connectivity: {'✅ PASS' if alpaca_success else '❌ FAIL'}")
        print(f"Database Integration: {'✅ PASS' if db_success else '❌ FAIL'}")
        
        if alpaca_success and db_success:
            print("\n🎉 All systems ready for trading!")
        else:
            print("\n⚠️ Some systems need attention before trading can begin.")
        
        print("=" * 60)
    
    asyncio.run(main())
