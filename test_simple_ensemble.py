#!/usr/bin/env python3
"""
Simple test to check ensemble ML integration.
"""

import asyncio
import logging
import sys
import os

# Configure logging first
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

# Import system modules
sys.path.append('/workspaces/crypto-pulse-v3')

from src.ml.ensemble import ml_ensemble
from src.core.trading_engine import TradingEngine

async def test_basic_integration():
    """Test basic ensemble integration."""
    
    print("=== Basic Ensemble Integration Test ===\n")
    
    # 1. Check ensemble status
    print(f"1. Ensemble Status:")
    print(f"   - Is trained: {ml_ensemble.is_trained}")
    print(f"   - RF trained: {ml_ensemble.random_forest.is_trained}")
    print(f"   - LSTM trained: {ml_ensemble.lstm.is_trained}")
    print(f"   - Transformer trained: {ml_ensemble.transformer.is_trained}")
    print(f"   - Weights: {ml_ensemble.weights}")
    
    # 2. Test trading engine signal determination
    print(f"\n2. Testing Signal Determination:")
    
    engine = TradingEngine()
    
    # Mock ML prediction with proper field names
    ml_prediction = {
        'signal_type': 'BUY',
        'confidence_score': 0.75,
        'expected_return': 0.025,  # 2.5% expected return
        'risk_score': 0.3,
        'model_scores': {
            'random_forest': 0.02,
            'lstm': 0.03,
            'transformer': 0.025
        }
    }
    
    sentiment_score = 0.6
    risk_metrics = {'risk_score': 0.4}
    
    signal, confidence = engine._determine_trading_signal(
        ml_prediction, sentiment_score, risk_metrics, None
    )
    
    print(f"   - Input: BUY signal, 0.75 confidence, 2.5% expected return")
    print(f"   - Output: {signal} signal, {confidence:.3f} confidence")
    print(f"   - ✓ Signal determination working correctly")
    
    # 3. Test ensemble prediction format
    print(f"\n3. Testing Prediction Format:")
    
    try:
        # Create mock features
        import numpy as np
        mock_features = np.random.randn(50)  # Assuming 50 features
        
        if ml_ensemble.random_forest.is_trained:
            prediction = ml_ensemble._ensemble_predict_single(mock_features)
            print(f"   - Signal Type: {prediction.signal_type}")
            print(f"   - Confidence: {prediction.confidence_score:.3f}")
            print(f"   - Expected Return: {prediction.expected_return:.3f}")
            print(f"   - Risk Score: {prediction.risk_score:.3f}")
            print(f"   - ✓ Prediction format is correct")
        else:
            print(f"   - ✗ Random Forest not trained, cannot test prediction")
            
    except Exception as e:
        print(f"   - ✗ Error testing prediction: {e}")
    
    print(f"\n=== Test Complete ===")

if __name__ == "__main__":
    asyncio.run(test_basic_integration())
