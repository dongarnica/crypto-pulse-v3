#!/usr/bin/env python3
"""
Test script to verify ensemble ML predictions are properly integrated into trading decisions.
"""

import asyncio
import logging
import traceback
from datetime import datetime
import pandas as pd
import numpy as np

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

# Import after logging configuration
import sys
import os
sys.path.append('/workspaces/crypto-pulse-v3')

from src.core.trading_engine import TradingEngine
from src.ml.ensemble import ml_ensemble
from src.data.technical_analysis import technical_analyzer
from src.core.database import db_manager
from config.settings import settings

async def test_ensemble_integration():
    """Test ensemble ML integration with trading decisions."""
    
    print("\n=== Testing Ensemble ML Integration ===\n")
    
    # 1. Test ML Ensemble Prediction
    print("1. Testing ML Ensemble Prediction...")
    
    try:
        # Check if ensemble is trained
        print(f"Ensemble trained: {ml_ensemble.is_trained}")
        print(f"Random Forest trained: {ml_ensemble.random_forest.is_trained}")
        print(f"LSTM trained: {ml_ensemble.lstm.is_trained}")
        print(f"Transformer trained: {ml_ensemble.transformer.is_trained}")
        print(f"Ensemble weights: {ml_ensemble.weights}")
        
        # Test with a symbol
        symbol = "BTC/USD"
        print(f"\nTesting prediction for {symbol}...")
        
        # Get features
        features = technical_analyzer.extract_features(symbol, '4h')
        if features:
            print(f"✓ Features extracted successfully for {symbol}")
            print(f"  Features timestamp: {features.timestamp}")
            print(f"  Sample features: RSI={features.rsi_14:.2f}, MACD={features.macd:.6f}")
            
            # Test ensemble prediction
            prediction = await ml_ensemble.predict(symbol, features)
            print(f"✓ Ensemble prediction successful:")
            print(f"  Signal Type: {prediction['signal_type']}")
            print(f"  Confidence Score: {prediction['confidence_score']:.3f}")
            print(f"  Expected Return: {prediction['expected_return']:.3f}")
            print(f"  Risk Score: {prediction['risk_score']:.3f}")
            print(f"  Model Scores: {prediction['model_scores']}")
            
        else:
            print(f"✗ Failed to extract features for {symbol}")
            
    except Exception as e:
        print(f"✗ Error testing ensemble prediction: {e}")
        print(traceback.format_exc())
    
    # 2. Test Trading Engine Signal Determination
    print(f"\n2. Testing Trading Engine Signal Determination...")
    
    try:
        engine = TradingEngine()
        
        # Create mock ML prediction
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
        
        sentiment_score = 0.6  # Neutral to positive sentiment
        risk_metrics = {'risk_score': 0.4}
        features = None  # Mock
        
        # Test signal determination
        signal, confidence = engine._determine_trading_signal(
            ml_prediction, sentiment_score, risk_metrics, features
        )
        
        print(f"✓ Trading signal determination successful:")
        print(f"  Input ML prediction: {ml_prediction['signal_type']} (confidence: {ml_prediction['confidence_score']:.3f})")
        print(f"  Input expected return: {ml_prediction['expected_return']:.3f}")
        print(f"  Input sentiment: {sentiment_score:.3f}")
        print(f"  Output signal: {signal}")
        print(f"  Output confidence: {confidence:.3f}")
        
        # Test with different scenarios
        scenarios = [
            {
                'name': 'Strong Buy Signal',
                'prediction': {
                    'signal_type': 'BUY',
                    'confidence_score': 0.85,
                    'expected_return': 0.04,
                    'risk_score': 0.2,
                    'model_scores': {'random_forest': 0.04, 'lstm': 0.038, 'transformer': 0.042}
                },
                'sentiment': 0.7,
                'risk': {'risk_score': 0.3}
            },
            {
                'name': 'Weak Signal',
                'prediction': {
                    'signal_type': 'BUY',
                    'confidence_score': 0.55,
                    'expected_return': 0.008,
                    'risk_score': 0.6,
                    'model_scores': {'random_forest': 0.01, 'lstm': 0.005, 'transformer': 0.009}
                },
                'sentiment': 0.45,
                'risk': {'risk_score': 0.7}
            },
            {
                'name': 'Sell Signal',
                'prediction': {
                    'signal_type': 'SELL',
                    'confidence_score': 0.78,
                    'expected_return': -0.025,
                    'risk_score': 0.4,
                    'model_scores': {'random_forest': -0.03, 'lstm': -0.02, 'transformer': -0.025}
                },
                'sentiment': 0.3,
                'risk': {'risk_score': 0.5}
            }
        ]
        
        print(f"\n  Testing different scenarios:")
        for scenario in scenarios:
            signal, confidence = engine._determine_trading_signal(
                scenario['prediction'], scenario['sentiment'], scenario['risk'], features
            )
            print(f"    {scenario['name']}: {signal} (conf: {confidence:.3f})")
        
    except Exception as e:
        print(f"✗ Error testing trading signal determination: {e}")
        print(traceback.format_exc())
    
    # 3. Test Individual Model Predictions
    print(f"\n3. Testing Individual Model Predictions...")
    
    try:
        if ml_ensemble.random_forest.is_trained:
            # Test Random Forest
            feature_vector = technical_analyzer.get_feature_vector(features) if features else np.random.randn(50)
            rf_direction, rf_confidence, rf_return = ml_ensemble.random_forest.predict(feature_vector)
            print(f"✓ Random Forest prediction:")
            print(f"  Direction: {rf_direction}, Confidence: {rf_confidence:.3f}, Return: {rf_return:.3f}")
        else:
            print(f"! Random Forest not trained")
        
        # Check LSTM and Transformer (may not be trained)
        print(f"  LSTM trained: {ml_ensemble.lstm.is_trained}")
        print(f"  Transformer trained: {ml_ensemble.transformer.is_trained}")
        
    except Exception as e:
        print(f"✗ Error testing individual models: {e}")
        print(traceback.format_exc())
    
    # 4. Test Ensemble Weight Optimization
    print(f"\n4. Testing Ensemble Weight Optimization...")
    
    try:
        print(f"Current ensemble weights: {ml_ensemble.weights}")
        
        # Verify weights sum to approximately 1
        weight_sum = sum(ml_ensemble.weights.values())
        print(f"Weight sum: {weight_sum:.3f} (should be close to 1.0)")
        
        if abs(weight_sum - 1.0) > 0.1:
            print(f"! Warning: Weights don't sum to 1.0")
        else:
            print(f"✓ Weights are properly normalized")
            
    except Exception as e:
        print(f"✗ Error checking ensemble weights: {e}")
    
    # 5. Test Database Storage of Ensemble Scores
    print(f"\n5. Testing Database Storage Integration...")
    
    try:
        # Check if TradingSignal model has ensemble score fields
        from src.core.models import TradingSignal
        
        # Get recent trading signals
        with db_manager.get_session() as session:
            recent_signals = session.query(TradingSignal).order_by(
                TradingSignal.timestamp.desc()
            ).limit(5).all()
            
            print(f"Found {len(recent_signals)} recent trading signals")
            
            for signal in recent_signals:
                print(f"  Signal: {signal.symbol} - {signal.signal_type}")
                print(f"    Ensemble Score: {signal.ensemble_score}")
                print(f"    RF Score: {signal.random_forest_score}")
                print(f"    LSTM Score: {signal.lstm_score}")
                print(f"    Transformer Score: {signal.transformer_score}")
                print(f"    Expected Return: {signal.expected_return}")
                print(f"    Risk Score: {signal.risk_score}")
                print(f"    Confidence: {signal.confidence_score}")
                print()
                
    except Exception as e:
        print(f"✗ Error testing database storage: {e}")
        print(traceback.format_exc())
    
    print("\n=== Ensemble Integration Test Complete ===\n")

async def test_training_status():
    """Check if models are trained and train if necessary."""
    
    print("=== Checking Training Status ===\n")
    
    if not ml_ensemble.is_trained:
        print("Ensemble not trained. Testing training process...")
        
        try:
            # Test training with a few symbols
            training_symbols = settings.trading.trading_pairs[:3]  # Use first 3 symbols
            print(f"Training with symbols: {training_symbols}")
            
            ml_ensemble.train(training_symbols, days_back=60)  # Use shorter period for testing
            print("✓ Training completed successfully")
            
        except Exception as e:
            print(f"✗ Training failed: {e}")
            print(traceback.format_exc())
    else:
        print("✓ Ensemble is already trained")

if __name__ == "__main__":
    async def main():
        try:
            await test_training_status()
            await test_ensemble_integration()
            
        except Exception as e:
            print(f"Error in main: {e}")
            print(traceback.format_exc())
    
    asyncio.run(main())
