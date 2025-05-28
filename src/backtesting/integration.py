"""
Integration module for connecting backtesting with the live trading system.
Provides real data access, strategy integration, and validation capabilities.
"""

import asyncio
import logging
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any, Union
from decimal import Decimal
import traceback

from .config import BacktestConfig
from .results import TradeResult, PortfolioSnapshot, BacktestResults
from .engine import BacktestEngine
from src.core.database import db_manager
from src.core.models import (
    MarketData, TradingSignal, Trade, Position, 
    PortfolioSnapshot as DBPortfolioSnapshot, SentimentData
)
from src.core.trading_engine import TradingEngine, MarketAnalysis, TradingDecision
from src.data.technical_analysis import technical_analyzer
from src.data.sentiment import sentiment_analyzer
from src.ml.ensemble import ml_ensemble
from src.risk.manager import risk_manager
from config.settings import settings

logger = logging.getLogger(__name__)


class DatabaseDataProvider:
    """Provides historical market data from the database for backtesting."""
    
    def __init__(self):
        self.cache = {}
        self.cache_ttl = 3600  # 1 hour cache
        
    async def get_historical_data(
        self, 
        symbol: str, 
        start_date: datetime, 
        end_date: datetime,
        timeframe: str = '1h'
    ) -> pd.DataFrame:
        """Get historical market data from database."""
        try:
            cache_key = f"{symbol}_{start_date}_{end_date}_{timeframe}"
            
            # Check cache first
            if cache_key in self.cache:
                cached_data, cached_time = self.cache[cache_key]
                if (datetime.utcnow() - cached_time).seconds < self.cache_ttl:
                    logger.debug(f"Using cached data for {symbol}")
                    return cached_data
            
            with db_manager.get_session() as session:
                query = session.query(MarketData).filter(
                    MarketData.symbol == symbol,
                    MarketData.timestamp >= start_date,
                    MarketData.timestamp <= end_date
                ).order_by(MarketData.timestamp.asc())
                
                data = query.all()
                
                if not data:
                    logger.warning(f"No historical data found for {symbol} from {start_date} to {end_date}")
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
                    'trade_count': d.trade_count or 0,
                    # Technical indicators if available
                    'rsi_14': d.rsi_14,
                    'macd_line': d.macd_line,
                    'macd_signal': d.macd_signal,
                    'bb_upper': d.bb_upper,
                    'bb_middle': d.bb_middle,
                    'bb_lower': d.bb_lower,
                    'atr_14': d.atr_14
                } for d in data])
                
                df.set_index('timestamp', inplace=True)
                df = df.sort_index()
                
                # Cache the result
                self.cache[cache_key] = (df, datetime.utcnow())
                
                logger.info(f"Loaded {len(df)} data points for {symbol}")
                return df
                
        except Exception as e:
            logger.error(f"Error loading historical data for {symbol}: {e}")
            return pd.DataFrame()
    
    async def get_sentiment_data(
        self, 
        symbol: Optional[str], 
        start_date: datetime, 
        end_date: datetime
    ) -> pd.DataFrame:
        """Get historical sentiment data."""
        try:
            with db_manager.get_session() as session:
                query = session.query(SentimentData).filter(
                    SentimentData.timestamp >= start_date,
                    SentimentData.timestamp <= end_date
                )
                
                if symbol:
                    query = query.filter(
                        (SentimentData.symbol == symbol) | 
                        (SentimentData.symbol.is_(None))
                    )
                
                data = query.order_by(SentimentData.timestamp.asc()).all()
                
                if not data:
                    return pd.DataFrame()
                
                df = pd.DataFrame([{
                    'timestamp': d.timestamp,
                    'symbol': d.symbol,
                    'news_sentiment': d.news_sentiment,
                    'social_sentiment': d.social_sentiment,
                    'fear_greed_index': d.fear_greed_index,
                    'sentiment_trend': d.sentiment_trend,
                    'confidence_score': d.confidence_score
                } for d in data])
                
                df.set_index('timestamp', inplace=True)
                return df
                
        except Exception as e:
            logger.error(f"Error loading sentiment data: {e}")
            return pd.DataFrame()
    
    async def get_historical_signals(
        self, 
        symbol: str, 
        start_date: datetime, 
        end_date: datetime
    ) -> pd.DataFrame:
        """Get historical trading signals."""
        try:
            with db_manager.get_session() as session:
                query = session.query(TradingSignal).filter(
                    TradingSignal.symbol == symbol,
                    TradingSignal.timestamp >= start_date,
                    TradingSignal.timestamp <= end_date
                ).order_by(TradingSignal.timestamp.asc())
                
                data = query.all()
                
                if not data:
                    return pd.DataFrame()
                
                df = pd.DataFrame([{
                    'timestamp': d.timestamp,
                    'signal_type': d.signal_type,
                    'confidence_score': d.confidence_score,
                    'expected_return': d.expected_return,
                    'ensemble_score': d.ensemble_score,
                    'random_forest_score': d.random_forest_score,
                    'lstm_score': d.lstm_score,
                    'transformer_score': d.transformer_score,
                    'sentiment_score': d.sentiment_score,
                    'risk_score': d.risk_score,
                    'recommended_allocation': d.recommended_allocation
                } for d in data])
                
                df.set_index('timestamp', inplace=True)
                return df
                
        except Exception as e:
            logger.error(f"Error loading historical signals: {e}")
            return pd.DataFrame()


class LiveStrategyIntegration:
    """Integrates live trading strategies with backtesting engine."""
    
    def __init__(self):
        self.trading_engine = None
        self.data_provider = DatabaseDataProvider()
        
    async def initialize(self):
        """Initialize the strategy integration."""
        try:
            # Initialize ML models
            await ml_ensemble.initialize()
            logger.info("Live strategy integration initialized")
        except Exception as e:
            logger.error(f"Error initializing strategy integration: {e}")
            raise
    
    async def generate_historical_analysis(
        self, 
        symbol: str, 
        timestamp: datetime,
        market_data: pd.DataFrame
    ) -> Optional[MarketAnalysis]:
        """Generate market analysis for a historical point in time."""
        try:
            # Get data slice up to the current timestamp
            historical_slice = market_data[market_data.index <= timestamp].copy()
            
            if len(historical_slice) < 100:  # Need sufficient history
                return None
            
            # Calculate technical indicators if not present
            if 'rsi_14' not in historical_slice.columns or historical_slice['rsi_14'].isna().all():
                historical_slice = technical_analyzer.calculate_indicators(historical_slice)
            
            # Extract features for ML
            features = technical_analyzer.extract_features(historical_slice)
            if features is None:
                return None
            
            # Get ML prediction
            ml_prediction = await ml_ensemble.predict(symbol, features)
            
            # Get sentiment (use cached or fallback)
            sentiment_score = await self._get_historical_sentiment(symbol, timestamp)
            
            # Calculate risk metrics
            risk_metrics = await self._calculate_historical_risk_metrics(symbol, historical_slice)
            
            # Determine trading signal
            signal, confidence = self._determine_trading_signal(
                ml_prediction, sentiment_score, risk_metrics, features
            )
            
            return MarketAnalysis(
                symbol=symbol,
                timestamp=timestamp,
                technical_features=features.to_dict() if hasattr(features, 'to_dict') else dict(features),
                ml_prediction=ml_prediction,
                sentiment_score=sentiment_score,
                risk_metrics=risk_metrics,
                trading_signal=signal,
                confidence=confidence
            )
            
        except Exception as e:
            logger.error(f"Error generating historical analysis for {symbol} at {timestamp}: {e}")
            return None
    
    async def generate_historical_decision(
        self, 
        analysis: MarketAnalysis, 
        has_position: bool,
        portfolio_value: float
    ) -> Optional[TradingDecision]:
        """Generate trading decision based on historical analysis."""
        try:
            signal = analysis.trading_signal
            confidence = analysis.confidence
            
            if confidence < 0.6:  # Minimum confidence threshold
                return None
            
            # Get position sizing recommendation
            sizing_rec = await risk_manager.calculate_position_size(
                symbol=analysis.symbol,
                signal_strength=confidence,
                risk_score=analysis.risk_metrics.get('risk_score', 0.5),
                current_price=analysis.technical_features.get('close', 0)
            )
            
            if not sizing_rec.recommended:
                return TradingDecision(
                    symbol=analysis.symbol,
                    action='HOLD',
                    confidence=confidence,
                    recommended_allocation=0.0,
                    reason="Risk manager declined position"
                )
            
            # Determine action based on signal and position status
            if signal == 'BUY' and not has_position:
                return TradingDecision(
                    symbol=analysis.symbol,
                    action='BUY',
                    confidence=confidence,
                    recommended_allocation=sizing_rec.allocation_percentage,
                    stop_loss=self._calculate_stop_loss(analysis),
                    take_profit=self._calculate_take_profit(analysis),
                    reason=f"ML Signal: {analysis.ml_prediction.get('signal', 'N/A')}, Sentiment: {analysis.sentiment_score:.2f}"
                )
            
            elif signal == 'SELL' and has_position:
                return TradingDecision(
                    symbol=analysis.symbol,
                    action='SELL',
                    confidence=confidence,
                    recommended_allocation=1.0,  # Full position
                    reason=f"Exit signal: {analysis.ml_prediction.get('signal', 'N/A')}"
                )
            
            elif signal == 'HOLD':
                return TradingDecision(
                    symbol=analysis.symbol,
                    action='HOLD',
                    confidence=confidence,
                    recommended_allocation=0.0,
                    reason="Hold signal from analysis"
                )
            
            return None
            
        except Exception as e:
            logger.error(f"Error generating historical decision for {analysis.symbol}: {e}")
            return None
    
    async def _get_historical_sentiment(self, symbol: str, timestamp: datetime) -> float:
        """Get sentiment score for historical timestamp."""
        try:
            # Look for sentiment data within 2 hours of timestamp
            start_time = timestamp - timedelta(hours=2)
            end_time = timestamp + timedelta(hours=2)
            
            sentiment_df = await self.data_provider.get_sentiment_data(symbol, start_time, end_time)
            
            if not sentiment_df.empty:
                # Get the closest sentiment reading
                closest_idx = sentiment_df.index.get_indexer([timestamp], method='nearest')[0]
                if closest_idx != -1:
                    row = sentiment_df.iloc[closest_idx]
                    # Combine different sentiment scores
                    news_score = row.get('news_sentiment', 0.5) or 0.5
                    social_score = row.get('social_sentiment', 0.5) or 0.5
                    fear_greed = row.get('fear_greed_index', 0.5) or 0.5
                    
                    # Weighted average
                    combined_score = (news_score * 0.4 + social_score * 0.3 + fear_greed * 0.3)
                    return max(0.0, min(1.0, combined_score))
            
            return 0.5  # Neutral sentiment as fallback
            
        except Exception as e:
            logger.error(f"Error getting historical sentiment: {e}")
            return 0.5
    
    async def _calculate_historical_risk_metrics(self, symbol: str, df: pd.DataFrame) -> Dict[str, float]:
        """Calculate risk metrics from historical data."""
        try:
            if len(df) < 20:
                return {'volatility': 0.02, 'risk_score': 0.5, 'expected_return': 0.05}
            
            # Calculate volatility
            returns = df['close'].pct_change().dropna()
            volatility = returns.std() * np.sqrt(24)  # Annualized hourly volatility
            
            # Calculate other risk metrics
            max_drawdown = self._calculate_max_drawdown(df['close'])
            var_95 = np.percentile(returns, 5) if len(returns) > 20 else -0.05
            
            risk_score = min(1.0, max(0.0, (volatility * 10 + abs(max_drawdown) * 5) / 2))
            
            return {
                'volatility': float(volatility),
                'max_drawdown': float(max_drawdown),
                'var_95': float(var_95),
                'risk_score': float(risk_score),
                'expected_return': 0.05  # Placeholder
            }
            
        except Exception as e:
            logger.error(f"Error calculating risk metrics: {e}")
            return {'volatility': 0.02, 'risk_score': 0.5, 'expected_return': 0.05}
    
    def _calculate_max_drawdown(self, prices: pd.Series) -> float:
        """Calculate maximum drawdown."""
        try:
            cumulative = (1 + prices.pct_change()).cumprod()
            running_max = cumulative.expanding().max()
            drawdown = (cumulative - running_max) / running_max
            return drawdown.min()
        except:
            return 0.0
    
    def _determine_trading_signal(
        self, 
        ml_prediction: Dict, 
        sentiment_score: float, 
        risk_metrics: Dict, 
        features
    ) -> Tuple[str, float]:
        """Determine final trading signal from all inputs."""
        try:
            # Get ensemble prediction
            expected_return = ml_prediction.get('expected_return', 0.0)
            confidence = ml_prediction.get('confidence_score', 0.0)
            
            # Convert expected return to score (0-1 scale)
            ensemble_score = max(0.0, min(1.0, (expected_return + 0.05) / 0.1))
            
            # Adjust for sentiment
            sentiment_weight = 0.2
            adjusted_score = ensemble_score * (1 - sentiment_weight) + sentiment_score * sentiment_weight
            
            # Adjust for risk
            risk_score = risk_metrics.get('risk_score', 0.5)
            if risk_score > 0.7:  # High risk
                confidence *= 0.8
            
            # Determine signal
            if adjusted_score > 0.6 and confidence > 0.65:
                return 'BUY', confidence
            elif adjusted_score < 0.4 and confidence > 0.65:
                return 'SELL', confidence
            else:
                return 'HOLD', confidence
                
        except Exception as e:
            logger.error(f"Error determining trading signal: {e}")
            return 'HOLD', 0.0
    
    def _calculate_stop_loss(self, analysis: MarketAnalysis) -> Optional[float]:
        """Calculate stop loss price."""
        try:
            current_price = analysis.technical_features.get('close', 0)
            atr = analysis.technical_features.get('atr_14', current_price * 0.02)
            
            # Use ATR-based stop loss
            stop_loss = current_price - (atr * settings.trading.atr_stop_multiplier)
            return max(stop_loss, current_price * 0.95)  # Maximum 5% stop loss
            
        except Exception:
            return None
    
    def _calculate_take_profit(self, analysis: MarketAnalysis) -> Optional[float]:
        """Calculate take profit price."""
        try:
            current_price = analysis.technical_features.get('close', 0)
            expected_return = analysis.risk_metrics.get('expected_return', 0.05)
            
            return current_price * (1 + max(expected_return, 0.08))  # Minimum 8% target
            
        except Exception:
            return None


class IntegratedBacktestEngine(BacktestEngine):
    """Enhanced backtest engine with live system integration."""
    
    def __init__(self, config: BacktestConfig):
        super().__init__(config)
        self.strategy_integration = LiveStrategyIntegration()
        self.data_provider = DatabaseDataProvider()
        
    async def initialize(self):
        """Initialize the integrated backtest engine."""
        try:
            await self.strategy_integration.initialize()
            logger.info("Integrated backtest engine initialized")
        except Exception as e:
            logger.error(f"Error initializing integrated engine: {e}")
            raise
    
    async def load_historical_data(self) -> bool:
        """Load historical data from database."""
        logger.info("Loading historical data from database...")
        
        try:
            for symbol in self.config.symbols:
                df = await self.data_provider.get_historical_data(
                    symbol, 
                    self.config.start_date, 
                    self.config.end_date
                )
                
                if df.empty:
                    logger.warning(f"No data available for {symbol}")
                    continue
                
                self.market_data_cache[symbol] = df
                logger.info(f"Loaded {len(df)} records for {symbol}")
            
            if not self.market_data_cache:
                logger.error("No market data loaded for any symbols")
                return False
            
            return True
            
        except Exception as e:
            logger.error(f"Error loading historical data: {e}")
            return False
    
    async def analyze_market_conditions(self, current_date: datetime) -> List[MarketAnalysis]:
        """Analyze market conditions using live strategies."""
        analyses = []
        
        for symbol in self.config.symbols:
            try:
                if symbol not in self.market_data_cache:
                    continue
                
                market_data = self.market_data_cache[symbol]
                
                # Generate analysis using live strategy
                analysis = await self.strategy_integration.generate_historical_analysis(
                    symbol, current_date, market_data
                )
                
                if analysis:
                    analyses.append(analysis)
                    self.signals_generated += 1
                
            except Exception as e:
                logger.error(f"Error analyzing {symbol} at {current_date}: {e}")
                self.analysis_errors += 1
                continue
        
        return analyses
    
    async def generate_signals(
        self, 
        analyses: List[MarketAnalysis], 
        current_date: datetime
    ) -> List[TradingDecision]:
        """Generate trading signals from market analyses."""
        decisions = []
        
        # Get current portfolio state
        portfolio_value = float(self.portfolio.get_total_value(
            {symbol: analyses_dict.technical_features.get('close', 0) 
             for analyses_dict in analyses for symbol in [analyses_dict.symbol]}
        ))
        
        for analysis in analyses:
            try:
                has_position = analysis.symbol in self.portfolio.positions and \
                              self.portfolio.positions[analysis.symbol] > 0
                
                decision = await self.strategy_integration.generate_historical_decision(
                    analysis, has_position, portfolio_value
                )
                
                if decision and decision.action != 'HOLD':
                    decisions.append(decision)
                
            except Exception as e:
                logger.error(f"Error generating signal for {analysis.symbol}: {e}")
                continue
        
        return decisions
    
    async def run_backtest(self) -> BacktestResults:
        """Run integrated backtest with live strategies."""
        logger.info(f"Starting integrated backtest from {self.config.start_date} to {self.config.end_date}")
        
        try:
            # Initialize
            await self.initialize()
            
            # Load data
            if not await self.load_historical_data():
                raise ValueError("Failed to load historical data")
            
            # Get all timestamps
            all_timestamps = set()
            for df in self.market_data_cache.values():
                all_timestamps.update(df.index)
            
            timestamps = sorted(all_timestamps)
            total_steps = len(timestamps)
            
            logger.info(f"Processing {total_steps} time steps")
            
            # Main backtest loop
            for i, timestamp in enumerate(timestamps):
                try:
                    if i % 100 == 0:
                        progress = (i / total_steps) * 100
                        logger.info(f"Progress: {progress:.1f}% ({i}/{total_steps})")
                    
                    self.current_date = timestamp
                    
                    # Analyze market conditions
                    analyses = await self.analyze_market_conditions(timestamp)
                    
                    if not analyses:
                        continue
                    
                    # Generate trading signals
                    decisions = await self.generate_signals(analyses, timestamp)
                    
                    # Execute trades
                    await self.execute_trades(decisions, timestamp)
                    
                    # Take portfolio snapshot
                    current_prices = {
                        analysis.symbol: analysis.technical_features.get('close', 0)
                        for analysis in analyses
                    }
                    self.portfolio.take_snapshot(timestamp, current_prices)
                    
                except Exception as e:
                    logger.error(f"Error processing timestamp {timestamp}: {e}")
                    continue
            
            # Calculate final results
            results = await self.calculate_results()
            
            logger.info("Integrated backtest completed successfully")
            return results
            
        except Exception as e:
            logger.error(f"Error in integrated backtest: {e}")
            logger.error(traceback.format_exc())
            raise
    
    async def calculate_results(self) -> BacktestResults:
        """Calculate backtest results with enhanced metrics."""
        try:
            # Use parent class method and enhance with integration-specific metrics
            results = await super().calculate_results()
            
            # Add integration-specific statistics
            if hasattr(results, 'metadata'):
                results.metadata.update({
                    'integration_type': 'live_strategy',
                    'data_source': 'database',
                    'strategy_components': ['technical_analysis', 'ml_ensemble', 'sentiment', 'risk_management']
                })
            
            return results
            
        except Exception as e:
            logger.error(f"Error calculating integrated results: {e}")
            raise


class BacktestValidator:
    """Validates backtest results against live trading performance."""
    
    def __init__(self):
        self.data_provider = DatabaseDataProvider()
    
    async def validate_against_live_signals(
        self, 
        backtest_results: BacktestResults,
        start_date: datetime,
        end_date: datetime
    ) -> Dict[str, Any]:
        """Compare backtest signals with actual live trading signals."""
        try:
            validation_results = {}
            
            for symbol in backtest_results.symbols:
                # Get live signals from database
                live_signals = await self.data_provider.get_historical_signals(
                    symbol, start_date, end_date
                )
                
                # Get backtest signals
                backtest_signals = [
                    trade for trade in backtest_results.trades 
                    if trade.symbol == symbol
                ]
                
                # Compare signals
                signal_comparison = self._compare_signals(live_signals, backtest_signals)
                validation_results[symbol] = signal_comparison
            
            return validation_results
            
        except Exception as e:
            logger.error(f"Error validating against live signals: {e}")
            return {}
    
    def _compare_signals(self, live_signals: pd.DataFrame, backtest_trades: List[TradeResult]) -> Dict[str, float]:
        """Compare live signals with backtest trades."""
        try:
            if live_signals.empty or not backtest_trades:
                return {'signal_accuracy': 0.0, 'timing_accuracy': 0.0, 'signal_count_ratio': 0.0}
            
            # Convert backtest trades to signals format
            backtest_df = pd.DataFrame([{
                'timestamp': trade.timestamp,
                'signal_type': trade.side,
                'confidence_score': getattr(trade, 'confidence', 0.7)
            } for trade in backtest_trades])
            
            backtest_df.set_index('timestamp', inplace=True)
            
            # Calculate signal accuracy (within 1 hour windows)
            matches = 0
            total_signals = len(live_signals)
            
            for idx, live_signal in live_signals.iterrows():
                window_start = idx - timedelta(hours=1)
                window_end = idx + timedelta(hours=1)
                
                matching_backtest = backtest_df[
                    (backtest_df.index >= window_start) & 
                    (backtest_df.index <= window_end) &
                    (backtest_df['signal_type'] == live_signal['signal_type'])
                ]
                
                if not matching_backtest.empty:
                    matches += 1
            
            signal_accuracy = matches / total_signals if total_signals > 0 else 0.0
            signal_count_ratio = len(backtest_trades) / total_signals if total_signals > 0 else 0.0
            
            return {
                'signal_accuracy': signal_accuracy,
                'timing_accuracy': signal_accuracy,  # Simplified for now
                'signal_count_ratio': signal_count_ratio,
                'live_signals_count': total_signals,
                'backtest_signals_count': len(backtest_trades)
            }
            
        except Exception as e:
            logger.error(f"Error comparing signals: {e}")
            return {'signal_accuracy': 0.0, 'timing_accuracy': 0.0, 'signal_count_ratio': 0.0}


# Global instances
database_data_provider = DatabaseDataProvider()
live_strategy_integration = LiveStrategyIntegration()
backtest_validator = BacktestValidator()
