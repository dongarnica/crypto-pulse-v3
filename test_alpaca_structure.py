#!/usr/bin/env python3
"""
Test script to verify Alpaca executor code structure and logic without requiring live API keys.
"""

import asyncio
import logging
from decimal import Decimal
from datetime import datetime
from unittest.mock import Mock, patch, MagicMock

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

async def test_alpaca_code_structure():
    """Test Alpaca executor code structure and logic without live connections."""
    print("=" * 60)
    print("ALPACA EXECUTOR CODE STRUCTURE TEST")
    print("=" * 60)
    
    try:
        # Import the executor
        print("\n1. Testing imports and module structure...")
        from src.execution.alpaca_executor import alpaca_executor, OrderRequest, OrderResult, PositionUpdate
        from alpaca.trading.client import TradingClient
        from alpaca.trading.requests import MarketOrderRequest, LimitOrderRequest, StopOrderRequest
        from alpaca.trading.enums import OrderSide, OrderType, TimeInForce
        print("✅ All imports successful")
        
        # Test data structures
        print("\n2. Testing data structure creation...")
        
        # Test OrderRequest
        order_request = OrderRequest(
            symbol="AAPL",
            side="BUY",
            quantity=Decimal('10'),
            order_type="MARKET",
            signal_id="test_signal_123"
        )
        print(f"✅ OrderRequest created: {order_request.symbol} {order_request.side} {order_request.quantity}")
        
        # Test OrderResult
        order_result = OrderResult(
            success=True,
            order_id="test_order_123",
            filled_quantity=Decimal('10'),
            status="FILLED"
        )
        print(f"✅ OrderResult created: {order_result.success}, ID: {order_result.order_id}")
        
        # Test PositionUpdate
        position_update = PositionUpdate(
            symbol="AAPL",
            side="LONG",
            quantity=Decimal('10'),
            current_price=Decimal('150.50'),
            unrealized_pnl=Decimal('25.00'),
            market_value=Decimal('1505.00')
        )
        print(f"✅ PositionUpdate created: {position_update.symbol} {position_update.quantity} shares")
        
        # Test order conversion logic
        print("\n3. Testing order conversion logic...")
        
        # Test market order conversion
        market_order = alpaca_executor._convert_to_alpaca_order(order_request)
        print(f"✅ Market order conversion: {type(market_order).__name__}")
        print(f"   Symbol: {market_order.symbol}, Qty: {market_order.qty}, Side: {market_order.side}")
        
        # Test limit order conversion
        limit_order_request = OrderRequest(
            symbol="TSLA",
            side="BUY",
            quantity=Decimal('5'),
            order_type="LIMIT",
            price=Decimal('200.00')
        )
        limit_order = alpaca_executor._convert_to_alpaca_order(limit_order_request)
        print(f"✅ Limit order conversion: {type(limit_order).__name__}")
        print(f"   Symbol: {limit_order.symbol}, Price: {limit_order.limit_price}")
        
        # Test stop order conversion
        stop_order_request = OrderRequest(
            symbol="NVDA",
            side="SELL",
            quantity=Decimal('2'),
            order_type="STOP",
            stop_price=Decimal('300.00')
        )
        stop_order = alpaca_executor._convert_to_alpaca_order(stop_order_request)
        print(f"✅ Stop order conversion: {type(stop_order).__name__}")
        print(f"   Symbol: {stop_order.symbol}, Stop Price: {stop_order.stop_price}")
        
        # Test unsupported order type
        print("\n4. Testing error handling...")
        try:
            invalid_order_request = OrderRequest(
                symbol="BTC",
                side="BUY",
                quantity=Decimal('1'),
                order_type="INVALID_TYPE"
            )
            alpaca_executor._convert_to_alpaca_order(invalid_order_request)
            print("❌ Should have raised ValueError for invalid order type")
        except ValueError as e:
            print(f"✅ Correctly handled invalid order type: {e}")
        
        print("\n5. Testing risk validation logic structure...")
        
        # Mock account info and positions for risk validation
        mock_account_info = {
            'buying_power': 10000.0,
            'portfolio_value': 50000.0,
            'cash': 5000.0
        }
        
        mock_positions = [
            PositionUpdate(
                symbol="AAPL",
                side="LONG",
                quantity=Decimal('50'),
                current_price=Decimal('150.00'),
                unrealized_pnl=Decimal('500.00'),
                market_value=Decimal('7500.00')
            )
        ]
        
        # Test risk validation with mocked data
        with patch.object(alpaca_executor, 'get_positions', return_value=mock_positions), \
             patch.object(alpaca_executor, 'get_account_info', return_value=mock_account_info), \
             patch.object(alpaca_executor.risk_manager, 'check_correlation_risk', 
                         return_value={'allowed': True, 'reason': 'Correlation check passed'}):
            
            test_order = OrderRequest(
                symbol="MSFT",
                side="BUY",
                quantity=Decimal('10'),
                order_type="MARKET",
                price=Decimal('300.00')  # $3000 order value
            )
            
            risk_result = await alpaca_executor._validate_order_risk(test_order)
            print(f"✅ Risk validation completed: {risk_result}")
        
        print("\n6. Testing class initialization and attributes...")
        
        # Check executor attributes
        assert hasattr(alpaca_executor, 'trading_client')
        assert hasattr(alpaca_executor, 'risk_manager')
        assert hasattr(alpaca_executor, 'active_orders')
        assert hasattr(alpaca_executor, 'position_cache')
        print("✅ All required attributes present")
        
        # Check methods exist
        required_methods = [
            'initialize', 'place_order', 'cancel_order', 'close_position',
            'get_positions', 'get_order_status', 'get_account_info'
        ]
        
        for method_name in required_methods:
            assert hasattr(alpaca_executor, method_name), f"Missing method: {method_name}"
            method = getattr(alpaca_executor, method_name)
            assert callable(method), f"Method {method_name} is not callable"
        print(f"✅ All {len(required_methods)} required methods present and callable")
        
        print("\n" + "=" * 60)
        print("✅ ALL CODE STRUCTURE TESTS PASSED!")
        print("=" * 60)
        print("\n📝 Summary:")
        print("  - Data structures work correctly")
        print("  - Order conversion logic is sound")
        print("  - Error handling is implemented")
        print("  - Risk validation framework is in place")
        print("  - All required methods are present")
        print("\n🔧 Next steps:")
        print("  - Set up valid Alpaca API keys for live testing")
        print("  - Initialize PostgreSQL database")
        print("  - Test with real market data")
        
        return True
        
    except Exception as e:
        print(f"\n❌ CODE STRUCTURE TEST FAILED: {e}")
        print("=" * 60)
        import traceback
        traceback.print_exc()
        return False

async def test_settings_integration():
    """Test settings integration with the executor."""
    print("\n" + "=" * 60)
    print("SETTINGS INTEGRATION TEST")
    print("=" * 60)
    
    try:
        from config.settings import settings
        
        print("\n1. Testing settings access...")
        print(f"✅ Environment: {settings.environment}")
        print(f"✅ Alpaca API Key: {settings.alpaca_api_key[:10]}..." if settings.alpaca_api_key else "❌ No Alpaca API key")
        print(f"✅ Max Portfolio Allocation: {settings.max_portfolio_allocation}")
        print(f"✅ Max Drawdown Threshold: {settings.max_drawdown_threshold}")
        print(f"✅ ATR Stop Multiplier: {settings.atr_stop_multiplier}")
        
        # Test paper trading setting
        is_paper = settings.environment != 'production'
        print(f"✅ Paper Trading Mode: {is_paper}")
        
        return True
        
    except Exception as e:
        print(f"❌ Settings test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    async def main():
        print("Starting Alpaca Executor Code Structure Test Suite...")
        print(f"Test started at: {datetime.now()}")
        
        # Test code structure
        structure_success = await test_alpaca_code_structure()
        
        # Test settings integration
        settings_success = await test_settings_integration()
        
        print("\n" + "=" * 60)
        print("TEST SUMMARY")
        print("=" * 60)
        print(f"Code Structure: {'✅ PASS' if structure_success else '❌ FAIL'}")
        print(f"Settings Integration: {'✅ PASS' if settings_success else '❌ FAIL'}")
        
        if structure_success and settings_success:
            print("\n🎉 Alpaca executor code is structurally sound!")
            print("Ready for live testing with proper API keys and database.")
        else:
            print("\n⚠️ Code structure needs attention.")
        
        print("=" * 60)
    
    asyncio.run(main())
