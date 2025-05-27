"""
Machine Learning ensemble for trading signal generation.
Combines Random Forest, LSTM, and Transformer models.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any
from datetime import datetime, timedelta
import logging
import pickle
import json
from dataclasses import dataclass
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import classification_report, mean_squared_error, accuracy_score
import tensorflow as tf
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import LSTM, Dense, Dropout, Attention, MultiHeadAttention, LayerNormalization
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
import joblib

from src.core.database import db_manager
from src.core.models import TradingSignal, ModelPerformance
from src.data.technical_analysis import technical_analyzer, TechnicalFeatures
from config.settings import settings

logger = logging.getLogger(__name__)


@dataclass
class ModelPrediction:
    """Container for model predictions."""
    symbol: str
    timestamp: datetime
    signal_type: str  # 'BUY', 'SELL', 'HOLD'
    confidence_score: float
    expected_return: float
    risk_score: float
    model_scores: Dict[str, float]  # Individual model scores


@dataclass
class TrainingData:
    """Container for training data."""
    features: np.ndarray
    labels: np.ndarray
    timestamps: List[datetime]
    symbols: List[str]


class RandomForestPredictor:
    """Random Forest model for pattern recognition."""
    
    def __init__(self):
        self.classifier = RandomForestClassifier(
            n_estimators=200,
            max_depth=15,
            min_samples_split=10,
            min_samples_leaf=5,
            random_state=42,
            n_jobs=-1
        )
        self.regressor = RandomForestRegressor(
            n_estimators=200,
            max_depth=15,
            min_samples_split=10,
            min_samples_leaf=5,
            random_state=42,
            n_jobs=-1
        )
        self.scaler = StandardScaler()
        self.label_encoder = LabelEncoder()
        self.is_trained = False
        self.feature_importance = None
    
    def train(self, training_data: TrainingData):
        """Train Random Forest models."""
        try:
            logger.info("Training Random Forest models...")
            
            # Scale features
            X_scaled = self.scaler.fit_transform(training_data.features)
            
            # Prepare classification labels (direction)
            y_direction = np.where(training_data.labels > 0.005, 1,  # BUY
                         np.where(training_data.labels < -0.005, -1, 0))  # SELL, HOLD
            
            # Train classification model
            self.classifier.fit(X_scaled, y_direction)
            
            # Train regression model for return prediction
            self.regressor.fit(X_scaled, training_data.labels)
            
            # Store feature importance
            self.feature_importance = self.classifier.feature_importances_
            
            # Evaluate models
            direction_score = cross_val_score(self.classifier, X_scaled, y_direction, cv=5).mean()
            return_score = cross_val_score(self.regressor, X_scaled, training_data.labels, cv=5).mean()
            
            logger.info(f"Random Forest - Direction accuracy: {direction_score:.3f}, Return R²: {return_score:.3f}")
            
            self.is_trained = True
            
        except Exception as e:
            logger.error(f"Error training Random Forest: {e}")
            raise
    
    def predict(self, features: np.ndarray) -> Tuple[int, float, float]:
        """
        Make predictions using Random Forest models.
        
        Returns:
            direction: -1 (SELL), 0 (HOLD), 1 (BUY)
            confidence: Probability of prediction
            expected_return: Predicted return
        """
        if not self.is_trained:
            raise ValueError("Model must be trained before making predictions")
        
        try:
            # Scale features
            X_scaled = self.scaler.transform(features.reshape(1, -1))
            
            # Get direction prediction and probabilities
            direction = self.classifier.predict(X_scaled)[0]
            probabilities = self.classifier.predict_proba(X_scaled)[0]
            confidence = np.max(probabilities)
            
            # Get return prediction
            expected_return = self.regressor.predict(X_scaled)[0]
            
            return direction, confidence, expected_return
            
        except Exception as e:
            logger.error(f"Error in Random Forest prediction: {e}")
            return 0, 0.0, 0.0


class LSTMPredictor:
    """LSTM model for sequential pattern recognition."""
    
    def __init__(self, sequence_length: int = 30):
        self.sequence_length = sequence_length
        self.model = None
        self.scaler = StandardScaler()
        self.is_trained = False
    
    def _build_model(self, input_shape: Tuple[int, int]) -> Sequential:
        """Build LSTM model architecture."""
        model = Sequential([
            LSTM(128, return_sequences=True, input_shape=input_shape),
            Dropout(0.2),
            LSTM(64, return_sequences=True),
            Dropout(0.2),
            LSTM(32, return_sequences=False),
            Dropout(0.2),
            Dense(16, activation='relu'),
            Dense(1, activation='tanh')  # Output range [-1, 1]
        ])
        
        model.compile(
            optimizer=Adam(learning_rate=0.001),
            loss='mse',
            metrics=['mae']
        )
        
        return model
    
    def _prepare_sequences(self, features: np.ndarray, labels: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Prepare sequential data for LSTM training."""
        X_seq, y_seq = [], []
        
        for i in range(self.sequence_length, len(features)):
            X_seq.append(features[i-self.sequence_length:i])
            y_seq.append(labels[i])
        
        return np.array(X_seq), np.array(y_seq)
    
    def train(self, training_data: TrainingData):
        """Train LSTM model."""
        try:
            logger.info("Training LSTM model...")
            
            # Scale features
            X_scaled = self.scaler.fit_transform(training_data.features)
            
            # Prepare sequences
            X_seq, y_seq = self._prepare_sequences(X_scaled, training_data.labels)
            
            if len(X_seq) < 100:
                logger.warning("Insufficient data for LSTM training")
                return
            
            # Build model
            self.model = self._build_model((self.sequence_length, X_scaled.shape[1]))
            
            # Train/validation split
            X_train, X_val, y_train, y_val = train_test_split(
                X_seq, y_seq, test_size=0.2, random_state=42
            )
            
            # Training callbacks
            callbacks = [
                EarlyStopping(patience=10, restore_best_weights=True),
                ReduceLROnPlateau(patience=5, factor=0.5, min_lr=1e-6)
            ]
            
            # Train model
            history = self.model.fit(
                X_train, y_train,
                validation_data=(X_val, y_val),
                epochs=50,
                batch_size=32,
                callbacks=callbacks,
                verbose=0
            )
            
            # Evaluate
            val_loss = min(history.history['val_loss'])
            logger.info(f"LSTM - Validation loss: {val_loss:.4f}")
            
            self.is_trained = True
            
        except Exception as e:
            logger.error(f"Error training LSTM: {e}")
            raise
    
    def predict(self, features_sequence: np.ndarray) -> Tuple[float, float]:
        """
        Make prediction using LSTM model.
        
        Args:
            features_sequence: Recent feature history (sequence_length x n_features)
            
        Returns:
            predicted_return: Predicted return
            confidence: Prediction confidence (based on uncertainty)
        """
        if not self.is_trained or self.model is None:
            return 0.0, 0.0
        
        try:
            # Scale features
            X_scaled = self.scaler.transform(features_sequence)
            X_seq = X_scaled.reshape(1, self.sequence_length, -1)
            
            # Get prediction
            prediction = self.model.predict(X_seq, verbose=0)[0][0]
            
            # Estimate confidence (simplified approach)
            confidence = min(abs(prediction), 1.0)
            
            return float(prediction), float(confidence)
            
        except Exception as e:
            logger.error(f"Error in LSTM prediction: {e}")
            return 0.0, 0.0


class TransformerPredictor:
    """Transformer model for attention-based pattern recognition."""
    
    def __init__(self, sequence_length: int = 30):
        self.sequence_length = sequence_length
        self.model = None
        self.scaler = StandardScaler()
        self.is_trained = False
    
    def _build_transformer_model(self, input_shape: Tuple[int, int]) -> Model:
        """Build Transformer model with multi-head attention."""
        inputs = tf.keras.Input(shape=input_shape)
        
        # Multi-head attention
        attention_output = MultiHeadAttention(
            num_heads=8,
            key_dim=64
        )(inputs, inputs)
        
        # Add & Norm
        attention_output = LayerNormalization()(inputs + attention_output)
        
        # Feed forward
        ff_output = Dense(128, activation='relu')(attention_output)
        ff_output = Dropout(0.1)(ff_output)
        ff_output = Dense(input_shape[1])(ff_output)
        
        # Add & Norm
        ff_output = LayerNormalization()(attention_output + ff_output)
        
        # Global pooling and output
        pooled = tf.keras.layers.GlobalAveragePooling1D()(ff_output)
        outputs = Dense(32, activation='relu')(pooled)
        outputs = Dropout(0.1)(outputs)
        outputs = Dense(1, activation='tanh')(outputs)
        
        model = Model(inputs, outputs)
        model.compile(
            optimizer=Adam(learning_rate=0.001),
            loss='mse',
            metrics=['mae']
        )
        
        return model
    
    def train(self, training_data: TrainingData):
        """Train Transformer model."""
        try:
            logger.info("Training Transformer model...")
            
            # Scale features
            X_scaled = self.scaler.fit_transform(training_data.features)
            
            # Prepare sequences
            X_seq, y_seq = self._prepare_sequences(X_scaled, training_data.labels)
            
            if len(X_seq) < 100:
                logger.warning("Insufficient data for Transformer training")
                return
            
            # Build model
            self.model = self._build_transformer_model((self.sequence_length, X_scaled.shape[1]))
            
            # Train/validation split
            X_train, X_val, y_train, y_val = train_test_split(
                X_seq, y_seq, test_size=0.2, random_state=42
            )
            
            # Training callbacks
            callbacks = [
                EarlyStopping(patience=15, restore_best_weights=True),
                ReduceLROnPlateau(patience=7, factor=0.5, min_lr=1e-6)
            ]
            
            # Train model
            history = self.model.fit(
                X_train, y_train,
                validation_data=(X_val, y_val),
                epochs=100,
                batch_size=32,
                callbacks=callbacks,
                verbose=0
            )
            
            # Evaluate
            val_loss = min(history.history['val_loss'])
            logger.info(f"Transformer - Validation loss: {val_loss:.4f}")
            
            self.is_trained = True
            
        except Exception as e:
            logger.error(f"Error training Transformer: {e}")
            raise
    
    def _prepare_sequences(self, features: np.ndarray, labels: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Prepare sequential data for Transformer training."""
        X_seq, y_seq = [], []
        
        for i in range(self.sequence_length, len(features)):
            X_seq.append(features[i-self.sequence_length:i])
            y_seq.append(labels[i])
        
        return np.array(X_seq), np.array(y_seq)
    
    def predict(self, features_sequence: np.ndarray) -> Tuple[float, float]:
        """Make prediction using Transformer model."""
        if not self.is_trained or self.model is None:
            return 0.0, 0.0
        
        try:
            # Scale features
            X_scaled = self.scaler.transform(features_sequence)
            X_seq = X_scaled.reshape(1, self.sequence_length, -1)
            
            # Get prediction
            prediction = self.model.predict(X_seq, verbose=0)[0][0]
            
            # Estimate confidence
            confidence = min(abs(prediction), 1.0)
            
            return float(prediction), float(confidence)
            
        except Exception as e:
            logger.error(f"Error in Transformer prediction: {e}")
            return 0.0, 0.0


class MLEnsemble:
    """
    Machine Learning ensemble combining Random Forest, LSTM, and Transformer models.
    """
    
    def __init__(self):
        self.random_forest = RandomForestPredictor()
        self.lstm = LSTMPredictor()
        self.transformer = TransformerPredictor()
        
        # Ensemble weights (will be optimized during training)
        self.weights = {
            'random_forest': 0.4,
            'lstm': 0.3,
            'transformer': 0.3
        }
        
        self.is_trained = False
        self.model_version = "1.0.0"
    
    def prepare_training_data(self, symbols: List[str], days_back: int = 180) -> TrainingData:
        """Prepare training data from historical market data."""
        try:
            logger.info(f"Preparing training data for {len(symbols)} symbols...")
            
            all_features = []
            all_labels = []
            all_timestamps = []
            all_symbols = []
            
            for symbol in symbols:
                # Get historical market data
                df = technical_analyzer.get_market_data(symbol, '4h', days_back * 6)  # 6 periods per day for 4h data
                
                if df.empty or len(df) < 50:
                    logger.warning(f"Insufficient data for {symbol}")
                    continue
                
                # Calculate technical indicators
                df = technical_analyzer.calculate_basic_indicators(df)
                df = technical_analyzer.calculate_advanced_indicators(df)
                df = technical_analyzer.calculate_market_structure(df)
                df = technical_analyzer.calculate_statistical_features(df)
                
                # Calculate forward returns (labels)
                df['forward_return_4h'] = df['close'].shift(-1) / df['close'] - 1
                df['forward_return_12h'] = df['close'].shift(-3) / df['close'] - 1
                df['forward_return_24h'] = df['close'].shift(-6) / df['close'] - 1
                
                # Use 12h forward return as primary label
                df['label'] = df['forward_return_12h']
                
                # Remove rows with NaN values
                df = df.dropna()
                
                if len(df) < 30:
                    continue
                
                # Extract features for each timestamp
                for idx, row in df.iterrows():
                    features = technical_analyzer.extract_features(symbol, '4h')
                    if features is not None:
                        feature_vector = technical_analyzer.get_feature_vector(features)
                        
                        all_features.append(feature_vector)
                        all_labels.append(row['label'])
                        all_timestamps.append(idx)
                        all_symbols.append(symbol)
            
            if not all_features:
                raise ValueError("No training data could be prepared")
            
            training_data = TrainingData(
                features=np.array(all_features),
                labels=np.array(all_labels),
                timestamps=all_timestamps,
                symbols=all_symbols
            )
            
            logger.info(f"Prepared {len(all_features)} training samples")
            return training_data
            
        except Exception as e:
            logger.error(f"Error preparing training data: {e}")
            raise
    
    def train(self, symbols: List[str], days_back: int = 180):
        """Train all models in the ensemble."""
        try:
            logger.info("Starting ensemble training...")
            
            # Prepare training data
            training_data = self.prepare_training_data(symbols, days_back)
            
            # Train individual models
            self.random_forest.train(training_data)
            self.lstm.train(training_data)
            self.transformer.train(training_data)
            
            # Optimize ensemble weights (simplified approach)
            self._optimize_weights(training_data)
            
            self.is_trained = True
            logger.info("Ensemble training completed successfully")
            
        except Exception as e:
            logger.error(f"Error training ensemble: {e}")
            raise
    
    def _optimize_weights(self, training_data: TrainingData):
        """Optimize ensemble weights based on validation performance."""
        # This is a simplified weight optimization
        # In practice, you might use more sophisticated methods
        try:
            # Split data for weight optimization
            split_idx = int(len(training_data.features) * 0.8)
            
            val_features = training_data.features[split_idx:]
            val_labels = training_data.labels[split_idx:]
            
            # Test different weight combinations
            best_score = float('inf')
            best_weights = self.weights.copy()
            
            weight_combinations = [
                {'random_forest': 0.5, 'lstm': 0.25, 'transformer': 0.25},
                {'random_forest': 0.4, 'lstm': 0.3, 'transformer': 0.3},
                {'random_forest': 0.3, 'lstm': 0.4, 'transformer': 0.3},
                {'random_forest': 0.3, 'lstm': 0.3, 'transformer': 0.4},
            ]
            
            for weights in weight_combinations:
                # Evaluate ensemble with these weights
                predictions = []
                for i, features in enumerate(val_features):
                    pred = self._ensemble_predict_single(features, weights)
                    predictions.append(pred.expected_return)
                
                # Calculate MSE
                mse = mean_squared_error(val_labels, predictions)
                
                if mse < best_score:
                    best_score = mse
                    best_weights = weights
            
            self.weights = best_weights
            logger.info(f"Optimized ensemble weights: {self.weights}")
            
        except Exception as e:
            logger.warning(f"Error optimizing weights, using default: {e}")
    
    def _ensemble_predict_single(self, features: np.ndarray, weights: Dict[str, float] = None) -> ModelPrediction:
        """Make prediction using ensemble of models."""
        if weights is None:
            weights = self.weights
        
        model_scores = {}
        
        # Random Forest prediction
        rf_direction, rf_confidence, rf_return = self.random_forest.predict(features)
        model_scores['random_forest'] = rf_return * rf_confidence
        
        # LSTM prediction (requires sequence)
        lstm_return, lstm_confidence = 0.0, 0.0
        if self.lstm.is_trained:
            # For single prediction, create dummy sequence
            sequence = np.tile(features, (self.lstm.sequence_length, 1))
            lstm_return, lstm_confidence = self.lstm.predict(sequence)
        model_scores['lstm'] = lstm_return * lstm_confidence
        
        # Transformer prediction (requires sequence)
        transformer_return, transformer_confidence = 0.0, 0.0
        if self.transformer.is_trained:
            # For single prediction, create dummy sequence
            sequence = np.tile(features, (self.transformer.sequence_length, 1))
            transformer_return, transformer_confidence = self.transformer.predict(sequence)
        model_scores['transformer'] = transformer_return * transformer_confidence
        
        # Ensemble prediction
        ensemble_return = (
            weights['random_forest'] * model_scores['random_forest'] +
            weights['lstm'] * model_scores['lstm'] +
            weights['transformer'] * model_scores['transformer']
        )
        
        # Determine signal type
        if ensemble_return > 0.01:  # 1% threshold
            signal_type = 'BUY'
        elif ensemble_return < -0.01:
            signal_type = 'SELL'
        else:
            signal_type = 'HOLD'
        
        # Calculate overall confidence
        confidence = np.mean([rf_confidence, lstm_confidence, transformer_confidence])
        
        # Calculate risk score (based on prediction volatility)
        risk_score = np.std(list(model_scores.values()))
        
        return ModelPrediction(
            symbol="",  # Will be set by caller
            timestamp=datetime.utcnow(),
            signal_type=signal_type,
            confidence_score=confidence,
            expected_return=ensemble_return,
            risk_score=risk_score,
            model_scores=model_scores
        )
    
    def predict(self, symbol: str) -> Optional[ModelPrediction]:
        """Generate trading signal for a symbol."""
        if not self.is_trained:
            logger.warning("Ensemble not trained, cannot make predictions")
            return None
        
        try:
            # Extract current features
            features = technical_analyzer.extract_features(symbol, '4h')
            if features is None:
                logger.warning(f"Could not extract features for {symbol}")
                return None
            
            feature_vector = technical_analyzer.get_feature_vector(features)
            
            # Make ensemble prediction
            prediction = self._ensemble_predict_single(feature_vector)
            prediction.symbol = symbol
            prediction.timestamp = features.timestamp
            
            return prediction
            
        except Exception as e:
            logger.error(f"Error making prediction for {symbol}: {e}")
            return None
    
    def save_models(self, filepath: str):
        """Save trained models to disk."""
        try:
            models_data = {
                'random_forest_classifier': self.random_forest.classifier,
                'random_forest_regressor': self.random_forest.regressor,
                'random_forest_scaler': self.random_forest.scaler,
                'lstm_scaler': self.lstm.scaler,
                'transformer_scaler': self.transformer.scaler,
                'weights': self.weights,
                'model_version': self.model_version,
                'is_trained': self.is_trained
            }
            
            # Save sklearn models
            joblib.dump(models_data, f"{filepath}_sklearn.pkl")
            
            # Save Keras models separately
            if self.lstm.model is not None:
                self.lstm.model.save(f"{filepath}_lstm.h5")
            
            if self.transformer.model is not None:
                self.transformer.model.save(f"{filepath}_transformer.h5")
            
            logger.info(f"Models saved to {filepath}")
            
        except Exception as e:
            logger.error(f"Error saving models: {e}")
    
    def load_models(self, filepath: str):
        """Load trained models from disk."""
        try:
            # Load sklearn models and metadata
            models_data = joblib.load(f"{filepath}_sklearn.pkl")
            
            self.random_forest.classifier = models_data['random_forest_classifier']
            self.random_forest.regressor = models_data['random_forest_regressor']
            self.random_forest.scaler = models_data['random_forest_scaler']
            self.random_forest.is_trained = True
            
            self.lstm.scaler = models_data['lstm_scaler']
            self.transformer.scaler = models_data['transformer_scaler']
            
            self.weights = models_data['weights']
            self.model_version = models_data['model_version']
            self.is_trained = models_data['is_trained']
            
            # Load Keras models
            try:
                self.lstm.model = tf.keras.models.load_model(f"{filepath}_lstm.h5")
                self.lstm.is_trained = True
            except:
                logger.warning("Could not load LSTM model")
            
            try:
                self.transformer.model = tf.keras.models.load_model(f"{filepath}_transformer.h5")
                self.transformer.is_trained = True
            except:
                logger.warning("Could not load Transformer model")
            
            logger.info(f"Models loaded from {filepath}")
            
        except Exception as e:
            logger.error(f"Error loading models: {e}")
    
    async def initialize(self):
        """Initialize the ML ensemble (async wrapper)."""
        try:
            # Load any existing trained models
            model_path = "models/ensemble_model"
            try:
                self.load_models(model_path)
                logger.info("Loaded existing ML models")
            except:
                logger.info("No existing models found, will train new ones")
            
            logger.info("ML ensemble initialized")
        except Exception as e:
            logger.error(f"Error initializing ML ensemble: {e}")
    
    async def predict(self, symbol: str, features) -> Dict[str, float]:
        """
        Async wrapper for prediction that handles TechnicalFeatures input.
        
        Args:
            symbol: Trading pair symbol
            features: TechnicalFeatures object or feature vector
            
        Returns:
            Dictionary with prediction results
        """
        try:
            # Convert TechnicalFeatures to numpy array if needed
            if hasattr(features, 'to_dict'):
                # It's a TechnicalFeatures object
                feature_vector = technical_analyzer.get_feature_vector(features)
            else:
                # Assume it's already a numpy array
                feature_vector = features
            
            # Get prediction
            prediction = self._ensemble_predict_single(feature_vector)
            
            if prediction:
                return {
                    'signal_type': prediction.signal_type,
                    'confidence_score': prediction.confidence_score,
                    'expected_return': prediction.expected_return,
                    'risk_score': prediction.risk_score,
                    'model_scores': prediction.model_scores
                }
            else:
                return {
                    'signal_type': 'HOLD',
                    'confidence_score': 0.0,
                    'expected_return': 0.0,
                    'risk_score': 0.5,
                    'model_scores': {'random_forest': 0.0, 'lstm': 0.0, 'transformer': 0.0}
                }
                
        except Exception as e:
            logger.error(f"Error in async predict for {symbol}: {e}")
            return {
                'signal_type': 'HOLD',
                'confidence_score': 0.0,
                'expected_return': 0.0,
                'risk_score': 0.5,
                'model_scores': {'random_forest': 0.0, 'lstm': 0.0, 'transformer': 0.0}
            }


# Global ML ensemble instance
ml_ensemble = MLEnsemble()
