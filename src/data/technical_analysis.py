"""
Technical analysis and feature engineering for trading signals.
Implements comprehensive technical indicators and market analysis.
"""

import pandas as pd
import numpy as np
from finta import TA
from typing import Dict, List, Optional, Tuple
from datetime import datetime, timedelta
import logging
from dataclasses import dataclass

from src.core.database import db_manager
from src.core.models import MarketData

logger = logging.getLogger(__name__)


@dataclass
class TechnicalFeatures:
    """Container for technical analysis features."""
    symbol: str
    timestamp: datetime
    
    # Price features
    close_price: float
    high_low_pct: float
    open_close_pct: float
    price_position: float  # Position within high-low range
    
    # Trend indicators
    sma_20: float
    sma_50: float
    ema_12: float
    ema_26: float
    price_above_sma20: bool
    price_above_sma50: bool
    sma_slope_20: float
    
    # Momentum indicators
    rsi_14: float
    rsi_30: float
    rsi_oversold: bool
    rsi_overbought: bool
    macd_line: float
    macd_signal: float
    macd_histogram: float
    macd_bullish: bool
    
    # Volatility indicators
    bb_upper: float
    bb_middle: float
    bb_lower: float
    bb_width: float
    bb_position: float  # Position within bands
    atr_14: float
    atr_normalized: float
    
    # Volume indicators
    volume: float
    volume_sma_20: float
    volume_ratio: float
    volume_surge: bool
    vwap: float
    price_vs_vwap: float
    
    # Support/Resistance
    pivot_point: float
    resistance_1: float
    support_1: float
    near_resistance: bool
    near_support: bool
    
    # Market structure
    higher_high: bool
    higher_low: bool
    lower_high: bool
    lower_low: bool
    trend_direction: str  # 'bullish', 'bearish', 'sideways'
    
    # Statistical features
    volatility_20: float
    skewness_20: float
    kurtosis_20: float
    returns_1h: float
    returns_4h: float
    returns_24h: float

    def to_dict(self) -> Dict[str, float]:
        """Convert TechnicalFeatures to dictionary for serialization."""
        return {
            'close_price': self.close_price,
            'high_low_pct': self.high_low_pct,
            'open_close_pct': self.open_close_pct,
            'price_position': self.price_position,
            'sma_20': self.sma_20,
            'sma_50': self.sma_50,
            'ema_12': self.ema_12,
            'ema_26': self.ema_26,
            'price_above_sma20': float(self.price_above_sma20),
            'price_above_sma50': float(self.price_above_sma50),
            'sma_slope_20': self.sma_slope_20,
            'rsi_14': self.rsi_14,
            'rsi_30': self.rsi_30,
            'rsi_oversold': float(self.rsi_oversold),
            'rsi_overbought': float(self.rsi_overbought),
            'macd_line': self.macd_line,
            'macd_signal': self.macd_signal,
            'macd_histogram': self.macd_histogram,
            'macd_bullish': float(self.macd_bullish),
            'bb_upper': self.bb_upper,
            'bb_middle': self.bb_middle,
            'bb_lower': self.bb_lower,
            'bb_width': self.bb_width,
            'bb_position': self.bb_position,
            'atr_14': self.atr_14,
            'atr_normalized': self.atr_normalized,
            'volume': self.volume,
            'volume_sma_20': self.volume_sma_20,
            'volume_ratio': self.volume_ratio,
            'volume_surge': float(self.volume_surge),
            'vwap': self.vwap,
            'price_vs_vwap': self.price_vs_vwap,
            'pivot_point': self.pivot_point,
            'resistance_1': self.resistance_1,
            'support_1': self.support_1,
            'near_resistance': float(self.near_resistance),
            'near_support': float(self.near_support),
            'higher_high': float(self.higher_high),
            'higher_low': float(self.higher_low),
            'lower_high': float(self.lower_high),
            'lower_low': float(self.lower_low),
            'trend_direction_bullish': float(self.trend_direction == 'bullish'),
            'trend_direction_bearish': float(self.trend_direction == 'bearish'),
            'volatility_20': self.volatility_20,
            'skewness_20': self.skewness_20,
            'kurtosis_20': self.kurtosis_20,
            'returns_1h': self.returns_1h,
            'returns_4h': self.returns_4h,
            'returns_24h': self.returns_24h
        }


class TechnicalAnalyzer:
    """
    Comprehensive technical analysis engine.
    Calculates indicators and extracts features for ML models.
    """
    
    def __init__(self):
        self.lookback_periods = {
            'short': 50,    # For immediate indicators
            'medium': 200,  # For trend analysis
            'long': 500     # For statistical features
        }
    
    def get_market_data(
        self, 
        symbol: str, 
        timeframe: str = '1h',
        limit: int = 500
    ) -> pd.DataFrame:
        """
        Retrieve market data from database.
        
        Args:
            symbol: Trading pair symbol
            timeframe: Data timeframe (1h, 4h, 1d)
            limit: Number of periods to retrieve
        """
        try:
            with db_manager.get_session() as session:
                # Calculate start time based on limit and timeframe
                timeframe_minutes = {
                    '1h': 60,
                    '4h': 240,
                    '1d': 1440
                }
                
                if timeframe not in timeframe_minutes:
                    raise ValueError(f"Unsupported timeframe: {timeframe}")
                
                minutes = timeframe_minutes[timeframe]
                start_time = datetime.utcnow() - timedelta(minutes=minutes * limit)
                
                # Query market data
                query = session.query(MarketData).filter(
                    MarketData.symbol == symbol,
                    MarketData.timestamp >= start_time
                ).order_by(MarketData.timestamp.asc())
                
                data = query.all()
                
                if not data:
                    logger.warning(f"No market data found for {symbol}")
                    return pd.DataFrame()
                
                # Convert to DataFrame
                df = pd.DataFrame([{
                    'timestamp': d.timestamp,
                    'open': float(d.open_price),
                    'high': float(d.high_price),
                    'low': float(d.low_price),
                    'close': float(d.close_price),
                    'volume': float(d.volume),
                    'quote_volume': float(d.quote_volume) if d.quote_volume else 0,
                    'trade_count': d.trade_count or 0
                } for d in data])
                
                df.set_index('timestamp', inplace=True)
                df = df.sort_index()
                
                return df
                
        except Exception as e:
            logger.error(f"Error retrieving market data for {symbol}: {e}")
            return pd.DataFrame()
    
    def calculate_basic_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate basic technical indicators."""
        if df.empty or len(df) < 50:
            return df
        
        try:
            # Moving averages
            df['sma_20'] = TA.SMA(df, 20)
            df['sma_50'] = TA.SMA(df, 50)
            df['ema_12'] = TA.EMA(df, 12)
            df['ema_26'] = TA.EMA(df, 26)
            
            # RSI
            df['rsi_14'] = TA.RSI(df, 14)
            df['rsi_30'] = TA.RSI(df, 30)
            
            # MACD
            macd_line = TA.MACD(df)['MACD']
            macd_signal = TA.MACD(df)['SIGNAL']
            df['macd_line'] = macd_line
            df['macd_signal'] = macd_signal
            df['macd_histogram'] = macd_line - macd_signal
            
            # Bollinger Bands
            bb_data = TA.BBANDS(df)
            df['bb_upper'] = bb_data['BB_UPPER']
            df['bb_middle'] = bb_data['BB_MIDDLE']
            df['bb_lower'] = bb_data['BB_LOWER']
            
            # ATR
            df['atr_14'] = TA.ATR(df, 14)
            
            # Volume indicators
            df['volume_sma_20'] = df['volume'].rolling(window=20).mean()
            
            return df
            
        except Exception as e:
            logger.error(f"Error calculating basic indicators: {e}")
            return df
    
    def calculate_advanced_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate advanced technical indicators."""
        if df.empty or len(df) < 50:
            return df
        
        try:
            # Stochastic
            stoch_data = TA.STOCH(df)
            df['stoch_k'] = stoch_data
            df['stoch_d'] = stoch_data.rolling(window=3).mean()  # Simple approximation for %D
            
            # Williams %R
            df['williams_r'] = TA.WILLIAMS(df)
            
            # Commodity Channel Index
            df['cci'] = TA.CCI(df)
            
            # Money Flow Index (use simple approximation if not available)
            try:
                df['mfi'] = TA.MFI(df)
            except:
                # Simple MFI approximation
                typical_price = (df['high'] + df['low'] + df['close']) / 3
                money_flow = typical_price * df['volume']
                df['mfi'] = money_flow.rolling(window=14).mean()
            
            # Average Directional Index
            try:
                df['adx'] = TA.ADX(df)
            except:
                df['adx'] = 25.0  # Default neutral value
            
            # Parabolic SAR (use simple approximation if not available)
            try:
                df['sar'] = TA.SAR(df)
            except:
                df['sar'] = df['close']  # Fallback to close price
            
            # VWAP (Volume Weighted Average Price)
            try:
                df['vwap'] = TA.VWAP(df)
            except:
                # Simple VWAP calculation
                typical_price = (df['high'] + df['low'] + df['close']) / 3
                df['vwap'] = (typical_price * df['volume']).cumsum() / df['volume'].cumsum()
            
            # Price channels
            df['high_20'] = df['high'].rolling(window=20).max()
            df['low_20'] = df['low'].rolling(window=20).min()
            
            return df
            
        except Exception as e:
            logger.error(f"Error calculating advanced indicators: {e}")
            return df
    
    def calculate_market_structure(self, df: pd.DataFrame) -> pd.DataFrame:
        """Analyze market structure patterns."""
        if df.empty or len(df) < 20:
            return df
        
        try:
            # Pivot points
            df['pivot'] = (df['high'] + df['low'] + df['close']) / 3
            df['resistance_1'] = 2 * df['pivot'] - df['low']
            df['support_1'] = 2 * df['pivot'] - df['high']
            
            # Swing highs and lows
            df['swing_high'] = df['high'] == df['high'].rolling(window=5, center=True).max()
            df['swing_low'] = df['low'] == df['low'].rolling(window=5, center=True).min()
            
            # Trend analysis
            df['price_above_sma20'] = df['close'] > df['sma_20']
            df['price_above_sma50'] = df['close'] > df['sma_50']
            df['sma20_above_sma50'] = df['sma_20'] > df['sma_50']
            
            # Price position within range
            df['price_position'] = (df['close'] - df['low_20']) / (df['high_20'] - df['low_20'])
            df['bb_position'] = (df['close'] - df['bb_lower']) / (df['bb_upper'] - df['bb_lower'])
            
            return df
            
        except Exception as e:
            logger.error(f"Error calculating market structure: {e}")
            return df
    
    def calculate_statistical_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate statistical features for ML models."""
        if df.empty or len(df) < 50:
            return df
        
        try:
            # Returns
            df['returns_1'] = df['close'].pct_change()
            df['returns_4'] = df['close'].pct_change(periods=4)
            df['returns_24'] = df['close'].pct_change(periods=24)
            
            # Volatility
            df['volatility_20'] = df['returns_1'].rolling(window=20).std()
            
            # Statistical moments
            df['skewness_20'] = df['returns_1'].rolling(window=20).skew()
            df['kurtosis_20'] = df['returns_1'].rolling(window=20).kurt()
            
            # Price ratios
            df['high_low_ratio'] = df['high'] / df['low']
            df['open_close_ratio'] = df['open'] / df['close']
            
            # Volume analysis
            df['volume_ratio'] = df['volume'] / df['volume_sma_20']
            df['volume_surge'] = df['volume_ratio'] > 2.0
            
            # Normalized ATR
            df['atr_normalized'] = df['atr_14'] / df['close']
            
            return df
            
        except Exception as e:
            logger.error(f"Error calculating statistical features: {e}")
            return df
    
    def calculate_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate all technical indicators for a DataFrame.
        This is a wrapper method that calls all indicator calculation methods.
        """
        if df.empty:
            return df
        
        try:
            # Calculate basic indicators
            df = self.calculate_basic_indicators(df)
            
            # Calculate advanced indicators
            df = self.calculate_advanced_indicators(df)
            
            # Calculate market structure
            df = self.calculate_market_structure(df)
            
            # Calculate statistical features
            df = self.calculate_statistical_features(df)
            
            return df
            
        except Exception as e:
            logger.error(f"Error calculating indicators: {e}")
            return df
    
    def extract_features(self, symbol: str, timeframe: str = '4h') -> Optional[TechnicalFeatures]:
        """
        Extract comprehensive technical features for a symbol.
        
        Args:
            symbol: Trading pair symbol
            timeframe: Data timeframe for analysis
            
        Returns:
            TechnicalFeatures object or None if insufficient data
        """
        try:
            # Get market data
            df = self.get_market_data(symbol, timeframe, self.lookback_periods['long'])
            
            if df.empty or len(df) < 50:
                logger.warning(f"Insufficient data for {symbol}")
                return None
            
            # Calculate all indicators
            df = self.calculate_basic_indicators(df)
            df = self.calculate_advanced_indicators(df)
            df = self.calculate_market_structure(df)
            df = self.calculate_statistical_features(df)
            
            # Get latest values
            latest = df.iloc[-1]
            prev = df.iloc[-2] if len(df) > 1 else latest
            
            # Extract features
            features = TechnicalFeatures(
                symbol=symbol,
                timestamp=latest.name,
                
                # Price features
                close_price=latest['close'],
                high_low_pct=(latest['high'] - latest['low']) / latest['low'],
                open_close_pct=(latest['close'] - latest['open']) / latest['open'],
                price_position=latest.get('price_position', 0.5),
                
                # Trend indicators
                sma_20=latest.get('sma_20', latest['close']),
                sma_50=latest.get('sma_50', latest['close']),
                ema_12=latest.get('ema_12', latest['close']),
                ema_26=latest.get('ema_26', latest['close']),
                price_above_sma20=latest.get('price_above_sma20', False),
                price_above_sma50=latest.get('price_above_sma50', False),
                sma_slope_20=(latest.get('sma_20', 0) - prev.get('sma_20', 0)) / prev.get('sma_20', 1),
                
                # Momentum indicators
                rsi_14=latest.get('rsi_14', 50),
                rsi_30=latest.get('rsi_30', 50),
                rsi_oversold=latest.get('rsi_14', 50) < 30,
                rsi_overbought=latest.get('rsi_14', 50) > 70,
                macd_line=latest.get('macd_line', 0),
                macd_signal=latest.get('macd_signal', 0),
                macd_histogram=latest.get('macd_histogram', 0),
                macd_bullish=latest.get('macd_line', 0) > latest.get('macd_signal', 0),
                
                # Volatility indicators
                bb_upper=latest.get('bb_upper', latest['close']),
                bb_middle=latest.get('bb_middle', latest['close']),
                bb_lower=latest.get('bb_lower', latest['close']),
                bb_width=(latest.get('bb_upper', 0) - latest.get('bb_lower', 0)) / latest.get('bb_middle', 1),
                bb_position=latest.get('bb_position', 0.5),
                atr_14=latest.get('atr_14', 0),
                atr_normalized=latest.get('atr_normalized', 0),
                
                # Volume indicators
                volume=latest['volume'],
                volume_sma_20=latest.get('volume_sma_20', latest['volume']),
                volume_ratio=latest.get('volume_ratio', 1.0),
                volume_surge=latest.get('volume_surge', False),
                vwap=latest.get('vwap', latest['close']),
                price_vs_vwap=(latest['close'] - latest.get('vwap', latest['close'])) / latest.get('vwap', latest['close']),
                
                # Support/Resistance
                pivot_point=latest.get('pivot', latest['close']),
                resistance_1=latest.get('resistance_1', latest['close']),
                support_1=latest.get('support_1', latest['close']),
                near_resistance=abs(latest['close'] - latest.get('resistance_1', latest['close'])) / latest['close'] < 0.02,
                near_support=abs(latest['close'] - latest.get('support_1', latest['close'])) / latest['close'] < 0.02,
                
                # Market structure
                higher_high=latest['high'] > prev['high'] and prev['high'] > df.iloc[-3]['high'] if len(df) > 2 else False,
                higher_low=latest['low'] > prev['low'] and prev['low'] > df.iloc[-3]['low'] if len(df) > 2 else False,
                lower_high=latest['high'] < prev['high'] and prev['high'] < df.iloc[-3]['high'] if len(df) > 2 else False,
                lower_low=latest['low'] < prev['low'] and prev['low'] < df.iloc[-3]['low'] if len(df) > 2 else False,
                trend_direction=self._determine_trend(df),
                
                # Statistical features
                volatility_20=latest.get('volatility_20', 0),
                skewness_20=latest.get('skewness_20', 0),
                kurtosis_20=latest.get('kurtosis_20', 0),
                returns_1h=latest.get('returns_1', 0),
                returns_4h=latest.get('returns_4', 0),
                returns_24h=latest.get('returns_24', 0)
            )
            
            return features
            
        except Exception as e:
            logger.error(f"Error extracting features for {symbol}: {e}")
            return None
    
    def _determine_trend(self, df: pd.DataFrame) -> str:
        """Determine overall trend direction."""
        if df.empty or len(df) < 20:
            return 'sideways'
        
        try:
            latest = df.iloc[-1]
            sma_20 = latest.get('sma_20', latest['close'])
            sma_50 = latest.get('sma_50', latest['close'])
            
            price_above_sma20 = latest['close'] > sma_20
            sma20_above_sma50 = sma_20 > sma_50
            
            # Simple trend determination
            if price_above_sma20 and sma20_above_sma50:
                return 'bullish'
            elif not price_above_sma20 and not sma20_above_sma50:
                return 'bearish'
            else:
                return 'sideways'
                
        except Exception:
            return 'sideways'
    
    def get_feature_vector(self, features: TechnicalFeatures) -> np.ndarray:
        """Convert TechnicalFeatures to numpy array for ML models."""
        try:
            vector = np.array([
                features.high_low_pct,
                features.open_close_pct,
                features.price_position,
                features.sma_slope_20,
                features.rsi_14,
                features.rsi_30,
                features.macd_histogram,
                features.bb_width,
                features.bb_position,
                features.atr_normalized,
                features.volume_ratio,
                features.price_vs_vwap,
                features.volatility_20,
                features.skewness_20,
                features.kurtosis_20,
                features.returns_1h,
                features.returns_4h,
                features.returns_24h,
                1.0 if features.rsi_oversold else 0.0,
                1.0 if features.rsi_overbought else 0.0,
                1.0 if features.macd_bullish else 0.0,
                1.0 if features.volume_surge else 0.0,
                1.0 if features.near_resistance else 0.0,
                1.0 if features.near_support else 0.0,
                1.0 if features.trend_direction == 'bullish' else 0.0,
                1.0 if features.trend_direction == 'bearish' else 0.0
            ])
            
            # Replace NaN/inf values
            vector = np.nan_to_num(vector, nan=0.0, posinf=1.0, neginf=-1.0)
            
            return vector
            
        except Exception as e:
            logger.error(f"Error creating feature vector: {e}")
            return np.zeros(26)  # Return zero vector on error


# Global technical analyzer instance
technical_analyzer = TechnicalAnalyzer()
