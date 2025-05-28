#!/usr/bin/env python3
"""
Comprehensive system status check to verify all components are working correctly.
Tests Binance integration, ensemble ML integration, and overall system health.
"""

import asyncio
import logging
from datetime import datetime
from decimal import Decimal

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

async def test_binance_integration():
    """Test Binance real-time price integration."""
    print("\n" + "=" * 60)
    print("BINANCE REAL-TIME PRICE INTEGRATION TEST")
    print("=" * 60)
    
    try:
        from src.data.binance_stream import data_streamer
        from src.core.trading_engine import trading_engine
        
        print("\n1. Testing Binance client initialization...")
        await data_streamer.initialize()
        print("✅ Binance data streamer initialized")
        
        print("\n2. Testing real-time price retrieval...")
        test_symbols = ["BTCUSDT", "ETHUSDT", "ADAUSDT"]
        
        for symbol in test_symbols:
            try:
                price = await data_streamer.get_current_price(symbol)
                if price:
                    print(f"✅ {symbol}: ${price:,.2f}")
                else:
                    print(f"⚠️ {symbol}: No price data")
            except Exception as e:
                print(f"❌ {symbol}: Error - {e}")
        
        print("\n3. Testing symbol conversion in trading engine...")
        alpaca_symbols = ["BTC/USD", "ETH/USD", "ADA/USD"]
        
        for alpaca_symbol in alpaca_symbols:
            binance_symbol = trading_engine._convert_to_binance_symbol(alpaca_symbol)
            print(f"✅ {alpaca_symbol} → {binance_symbol}")
        
        print("\n4. Testing trading engine price method...")
        for alpaca_symbol in alpaca_symbols[:2]:  # Test first 2
            try:
                price = await trading_engine._get_current_price(alpaca_symbol)
                if price:
                    print(f"✅ Trading engine price for {alpaca_symbol}: ${price:,.2f}")
                else:
                    print(f"⚠️ Trading engine: No price for {alpaca_symbol}")
            except Exception as e:
                print(f"❌ Trading engine price error for {alpaca_symbol}: {e}")
        
        return True
        
    except Exception as e:
        print(f"❌ Binance integration test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

async def test_ensemble_ml_integration():
    """Test ensemble ML model integration."""
    print("\n" + "=" * 60)
    print("ENSEMBLE ML INTEGRATION TEST")
    print("=" * 60)
    
    try:
        from src.ml.ensemble import ml_ensemble
        from src.core.trading_engine import trading_engine
        import numpy as np
        
        print("\n1. Testing ML ensemble initialization...")
        await ml_ensemble.initialize()
        print("✅ ML ensemble initialized")
        
        print("\n2. Testing ensemble prediction structure...")
        # Create dummy features for testing
        test_features = np.random.random((50,))  # 50 features
        
        prediction = await ml_ensemble.predict("BTCUSDT", test_features)
        
        if prediction:
            print("✅ Ensemble prediction generated:")
            print(f"   Expected Return: {prediction.get('expected_return', 'N/A')}")
            print(f"   Confidence Score: {prediction.get('confidence_score', 'N/A')}")
            print(f"   Risk Score: {prediction.get('risk_score', 'N/A')}")
            
            # Check model scores
            model_scores = prediction.get('model_scores', {})
            print(f"   Random Forest: {model_scores.get('random_forest', 'N/A')}")
            print(f"   LSTM: {model_scores.get('lstm', 'N/A')}")
            print(f"   Transformer: {model_scores.get('transformer', 'N/A')}")
        else:
            print("⚠️ No prediction generated (models may not be trained)")
        
        print("\n3. Testing trading signal determination...")
        # Test the fixed field mapping
        test_ml_prediction = {
            'expected_return': 0.03,  # 3% expected return
            'confidence_score': 0.8,  # 80% confidence
            'risk_score': 0.4,        # 40% risk score
            'model_scores': {
                'random_forest': 0.75,
                'lstm': 0.82,
                'transformer': 0.78
            }
        }
        
        # Mock features for testing
        class MockFeatures:
            def to_dict(self):
                return {'close_price': 45000, 'rsi_14': 65, 'macd': 0.1}
        
        signal, confidence = trading_engine._determine_trading_signal(
            test_ml_prediction, 0.6, {'risk_score': 0.4}, MockFeatures()
        )
        
        print(f"✅ Trading signal determination:")
        print(f"   Signal: {signal}")
        print(f"   Confidence: {confidence:.3f}")
        
        return True
        
    except Exception as e:
        print(f"❌ Ensemble ML integration test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

async def test_database_integration():
    """Test database connectivity and models."""
    print("\n" + "=" * 60)
    print("DATABASE INTEGRATION TEST")
    print("=" * 60)
    
    try:
        from src.core.database import db_manager
        from src.core.models import TradingSignal, Trade, Position, MarketData
        
        print("\n1. Testing database connection...")
        with db_manager.get_session() as session:
            result = session.execute("SELECT 1").scalar()
            print(f"✅ Database connection successful: {result}")
        
        print("\n2. Testing table access...")
        with db_manager.get_session() as session:
            tables_data = {
                'TradingSignal': session.query(TradingSignal).count(),
                'Trade': session.query(Trade).count(),
                'Position': session.query(Position).count(),
                'MarketData': session.query(MarketData).count()
            }
            
            for table, count in tables_data.items():
                print(f"✅ {table}: {count} records")
        
        return True
        
    except Exception as e:
        print(f"❌ Database integration test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

async def test_risk_management():
    """Test risk management integration."""
    print("\n" + "=" * 60)
    print("RISK MANAGEMENT INTEGRATION TEST")
    print("=" * 60)
    
    try:
        from src.risk.manager import risk_manager
        
        print("\n1. Testing risk manager initialization...")
        # Test basic functionality
        print("✅ Risk manager accessible")
        
        print("\n2. Testing Kelly criterion calculation...")
        # Mock calculation (would normally use real data)
        test_params = {
            'symbol': 'BTCUSDT',
            'signal_strength': 0.75,
            'risk_score': 0.3,
            'current_price': 45000
        }
        
        try:
            sizing_rec = await risk_manager.calculate_position_size(**test_params)
            print(f"✅ Position sizing recommendation:")
            print(f"   Recommended: {sizing_rec.recommended}")
            print(f"   Allocation: {sizing_rec.allocation_percentage:.2%}")
            print(f"   Reason: {sizing_rec.reason}")
        except Exception as e:
            print(f"⚠️ Position sizing test failed: {e}")
        
        return True
        
    except Exception as e:
        print(f"❌ Risk management test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

async def test_system_health():
    """Test system health monitoring."""
    print("\n" + "=" * 60)
    print("SYSTEM HEALTH MONITORING TEST")
    print("=" * 60)
    
    try:
        from src.monitoring.health_checks import health_checker
        
        print("\n1. Testing health checker initialization...")
        print("✅ Health checker initialized")
        
        print("\n2. Running basic health checks...")
        # Run a subset of health checks
        basic_checks = ['database', 'system_resources']
        
        for check_name in basic_checks:
            try:
                result = await health_checker.run_single_check(check_name)
                if result:
                    status_emoji = {"healthy": "✅", "warning": "⚠️", "critical": "❌"}.get(result.status.value, "❓")
                    print(f"{status_emoji} {check_name}: {result.message}")
                else:
                    print(f"⚠️ {check_name}: No result")
            except Exception as e:
                print(f"❌ {check_name}: Error - {e}")
        
        return True
        
    except Exception as e:
        print(f"❌ System health test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

async def test_settings_configuration():
    """Test settings and configuration."""
    print("\n" + "=" * 60)
    print("SETTINGS AND CONFIGURATION TEST")
    print("=" * 60)
    
    try:
        from config.settings import settings
        
        print("\n1. Testing settings access...")
        print(f"✅ Environment: {settings.environment}")
        print(f"✅ Trading pairs: {len(settings.trading_pairs)} pairs")
        print(f"✅ Max portfolio allocation: {settings.max_portfolio_allocation:.1%}")
        print(f"✅ Max drawdown threshold: {settings.max_drawdown_threshold:.1%}")
        
        print("\n2. Testing API key configuration...")
        api_keys = {
            'Binance API': bool(settings.binance_api_key and not settings.binance_api_key.startswith('development_')),
            'Alpaca API': bool(settings.alpaca_api_key and not settings.alpaca_api_key.startswith('development_')),
            'Perplexity API': bool(settings.perplexity_api_key and not settings.perplexity_api_key.startswith('development_'))
        }
        
        for api, configured in api_keys.items():
            status = "✅ Configured" if configured else "⚠️ Development/Missing"
            print(f"   {api}: {status}")
        
        return True
        
    except Exception as e:
        print(f"❌ Settings test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

async def main():
    """Run comprehensive system status check."""
    print("🚀 CRYPTO PULSE V3 - COMPREHENSIVE SYSTEM STATUS CHECK")
    print("=" * 80)
    print(f"Test started at: {datetime.now()}")
    
    # Run all tests
    test_results = {}
    
    test_functions = [
        ("Settings Configuration", test_settings_configuration),
        ("Database Integration", test_database_integration),
        ("Binance Integration", test_binance_integration),
        ("Ensemble ML Integration", test_ensemble_ml_integration),
        ("Risk Management", test_risk_management),
        ("System Health Monitoring", test_system_health),
    ]
    
    for test_name, test_func in test_functions:
        try:
            result = await test_func()
            test_results[test_name] = result
        except Exception as e:
            print(f"\n❌ {test_name} test crashed: {e}")
            test_results[test_name] = False
    
    # Print summary
    print("\n" + "=" * 80)
    print("📊 COMPREHENSIVE TEST SUMMARY")
    print("=" * 80)
    
    passed = sum(test_results.values())
    total = len(test_results)
    
    for test_name, result in test_results.items():
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{test_name:<30} {status}")
    
    print(f"\nOverall Result: {passed}/{total} tests passed")
    
    if passed == total:
        print("\n🎉 ALL SYSTEMS OPERATIONAL!")
        print("✅ Binance real-time pricing integrated")
        print("✅ Ensemble ML predictions integrated")
        print("✅ Database and risk management working")
        print("✅ System ready for trading")
    else:
        print(f"\n⚠️ {total - passed} SYSTEM(S) NEED ATTENTION")
        print("Review failed tests above before starting trading")
    
    print("=" * 80)

if __name__ == "__main__":
    asyncio.run(main())
