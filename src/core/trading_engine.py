"""
Core trading engine - Main coordinator for the Crypto Pulse trading system.
Orchestrates data ingestion, analysis, signal generation, and trade execution.
"""

import asyncio
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any
import pandas as pd
from dataclasses import dataclass
import traceback

from src.core.database import db_manager
from src.core.models import TradingSignal, Position, PortfolioSnapshot
from src.data.binance_stream import data_streamer
from src.data.technical_analysis import technical_analyzer
from src.data.sentiment import sentiment_analyzer
from src.ml.ensemble import ml_ensemble
from src.risk.manager import risk_manager
from src.execution.alpaca_executor import alpaca_executor, OrderRequest
from config.settings import settings

logger = logging.getLogger(__name__)


@dataclass
class MarketAnalysis:
    """Container for market analysis results."""
    symbol: str
    timestamp: datetime
    technical_features: Dict[str, float]
    ml_prediction: Dict[str, float]
    sentiment_score: float
    risk_metrics: Dict[str, float]
    trading_signal: Optional[str] = None
    confidence: float = 0.0


@dataclass
class TradingDecision:
    """Container for trading decisions."""
    symbol: str
    action: str  # 'BUY', 'SELL', 'HOLD', 'CLOSE'
    confidence: float
    recommended_allocation: float
    stop_loss: Optional[float] = None
    take_profit: Optional[float] = None
    reason: str = ""


class TradingEngine:
    """
    Main trading engine coordinating all system components.
    """
    
    def __init__(self):
        self.is_running = False
        self.analysis_cycle_minutes = 30
        self.sentiment_update_minutes = 120
        self.last_sentiment_update = datetime.utcnow() - timedelta(hours=3)
        self.active_signals: Dict[str, TradingSignal] = {}
        self.market_analysis_cache: Dict[str, MarketAnalysis] = {}
        
        # Performance tracking
        self.cycle_count = 0
        self.successful_analyses = 0
        self.errors_count = 0
        
    async def initialize(self):
        """Initialize all trading system components."""
        try:
            logger.info("Initializing Crypto Pulse Trading Engine...")
            
            # Initialize database
            await db_manager.initialize()
            
            # Initialize data streaming
            await data_streamer.initialize()
            
            # Initialize ML models
            await ml_ensemble.initialize()
            
            # Initialize trade execution
            await alpaca_executor.initialize()
            
            # Load any existing active signals
            await self._load_active_signals()
            
            logger.info("Trading engine initialization complete")
            
        except Exception as e:
            logger.error(f"Failed to initialize trading engine: {e}")
            raise
    
    async def start(self):
        """Start the main trading loop."""
        if self.is_running:
            logger.warning("Trading engine is already running")
            return
        
        try:
            self.is_running = True
            logger.info("Starting Crypto Pulse Trading Engine")
            
            # Start data streams
            await self._start_data_streams()
            
            # Start main trading loop
            await self._run_trading_loop()
            
        except Exception as e:
            logger.error(f"Trading engine error: {e}")
            logger.error(traceback.format_exc())
        finally:
            self.is_running = False
            await self.stop()
    
    async def stop(self):
        """Stop the trading engine gracefully."""
        logger.info("Stopping Crypto Pulse Trading Engine")
        self.is_running = False
        
        # Stop data streams
        await data_streamer.stop_all_streams()
        
        # Close any pending orders (implementation specific)
        await self._handle_shutdown_orders()
        
        logger.info("Trading engine stopped")
    
    async def _run_trading_loop(self):
        """Main trading loop with 30-minute analysis cycles."""
        logger.info(f"Starting trading loop with {self.analysis_cycle_minutes}-minute cycles")
        
        while self.is_running:
            try:
                cycle_start = datetime.utcnow()
                self.cycle_count += 1
                
                logger.info(f"Starting analysis cycle #{self.cycle_count}")
                
                # Update sentiment if needed (every 2 hours)
                if self._should_update_sentiment():
                    await self._update_market_sentiment()
                
                # Analyze all trading pairs
                market_analyses = await self._analyze_all_markets()
                
                # Generate trading decisions
                trading_decisions = await self._generate_trading_decisions(market_analyses)
                
                # Execute trades
                await self._execute_trading_decisions(trading_decisions)
                
                # Update portfolio tracking
                await self._update_portfolio_metrics()
                
                # Risk monitoring
                await self._monitor_risk_limits()
                
                # Clean up old data
                await self._cleanup_old_data()
                
                cycle_duration = (datetime.utcnow() - cycle_start).total_seconds()
                self.successful_analyses += 1
                
                logger.info(f"Completed cycle #{self.cycle_count} in {cycle_duration:.1f}s")
                
                # Wait for next cycle
                await self._wait_for_next_cycle(cycle_start)
                
            except Exception as e:
                self.errors_count += 1
                logger.error(f"Error in trading cycle #{self.cycle_count}: {e}")
                logger.error(traceback.format_exc())
                
                # Wait before retrying
                await asyncio.sleep(60)
    
    async def _start_data_streams(self):
        """Start real-time data streams for all trading pairs."""
        try:
            trading_pairs = settings.trading.trading_pairs
            
            # Start kline streams (1-hour candles)
            await data_streamer.start_kline_streams(trading_pairs, "1h")
            
            # Start order book streams for top pairs
            priority_pairs = trading_pairs[:10]  # Top 10 pairs
            await data_streamer.start_orderbook_streams(priority_pairs, depth=20)
            
            logger.info(f"Started data streams for {len(trading_pairs)} pairs")
            
        except Exception as e:
            logger.error(f"Error starting data streams: {e}")
            raise
    
    async def _analyze_all_markets(self) -> List[MarketAnalysis]:
        """Analyze all trading pairs and generate market insights."""
        analyses = []
        
        for symbol in settings.trading.trading_pairs:
            try:
                analysis = await self._analyze_single_market(symbol)
                if analysis:
                    analyses.append(analysis)
                    self.market_analysis_cache[symbol] = analysis
                    
            except Exception as e:
                logger.error(f"Error analyzing {symbol}: {e}")
                continue
        
        logger.info(f"Completed analysis for {len(analyses)} markets")
        return analyses
    
    async def _analyze_single_market(self, symbol: str) -> Optional[MarketAnalysis]:
        """Perform comprehensive analysis for a single market."""
        try:
            # Get market data
            df = technical_analyzer.get_market_data(symbol, timeframe='1h', limit=500)
            if df.empty or len(df) < 100:
                logger.warning(f"Insufficient data for {symbol}")
                return None
            
            # Calculate technical indicators
            df_with_indicators = technical_analyzer.calculate_indicators(df)
            
            # Extract features for ML
            features = technical_analyzer.extract_features(df_with_indicators)
            if not features:
                logger.warning(f"Failed to extract features for {symbol}")
                return None
            
            # Get ML predictions
            ml_prediction = await ml_ensemble.predict(symbol, features)
            
            # Get sentiment score
            sentiment_score = await self._get_symbol_sentiment(symbol)
            
            # Calculate risk metrics
            risk_metrics = await self._calculate_risk_metrics(symbol, df_with_indicators)
            
            # Determine trading signal
            signal, confidence = self._determine_trading_signal(
                ml_prediction, sentiment_score, risk_metrics, features
            )
            
            return MarketAnalysis(
                symbol=symbol,
                timestamp=datetime.utcnow(),
                technical_features=features.to_dict(),
                ml_prediction=ml_prediction,
                sentiment_score=sentiment_score,
                risk_metrics=risk_metrics,
                trading_signal=signal,
                confidence=confidence
            )
            
        except Exception as e:
            logger.error(f"Error in single market analysis for {symbol}: {e}")
            return None
    
    async def _generate_trading_decisions(self, analyses: List[MarketAnalysis]) -> List[TradingDecision]:
        """Generate concrete trading decisions from market analyses."""
        decisions = []
        
        # Get current positions
        current_positions = await alpaca_executor.get_positions()
        position_symbols = {pos.symbol for pos in current_positions}
        
        for analysis in analyses:
            try:
                decision = await self._make_trading_decision(analysis, analysis.symbol in position_symbols)
                if decision:
                    decisions.append(decision)
                    
            except Exception as e:
                logger.error(f"Error generating decision for {analysis.symbol}: {e}")
                continue
        
        # Sort by confidence and filter
        decisions = sorted(decisions, key=lambda x: x.confidence, reverse=True)
        decisions = [d for d in decisions if d.confidence >= 0.65]  # Minimum confidence threshold
        
        logger.info(f"Generated {len(decisions)} trading decisions")
        return decisions
    
    async def _make_trading_decision(self, analysis: MarketAnalysis, has_position: bool) -> Optional[TradingDecision]:
        """Make a trading decision for a single market."""
        try:
            signal = analysis.trading_signal
            confidence = analysis.confidence
            
            if confidence < 0.6:  # Minimum confidence for any action
                return None
            
            # Get position sizing recommendation
            sizing_rec = await risk_manager.calculate_position_size(
                symbol=analysis.symbol,
                signal_strength=confidence,
                risk_score=analysis.risk_metrics.get('risk_score', 0.5),
                current_price=analysis.technical_features.get('close_price', 0)
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
            logger.error(f"Error making trading decision for {analysis.symbol}: {e}")
            return None
    
    async def _execute_trading_decisions(self, decisions: List[TradingDecision]):
        """Execute trading decisions through Alpaca."""
        executed_count = 0
        
        for decision in decisions:
            try:
                if decision.action == 'HOLD':
                    continue
                
                # Get current market price
                current_price = await self._get_current_price(decision.symbol)
                if not current_price:
                    logger.warning(f"Could not get current price for {decision.symbol}")
                    continue
                
                # Calculate quantity
                account_info = await alpaca_executor.get_account_info()
                portfolio_value = account_info.get('portfolio_value', 100000)
                
                if decision.action == 'BUY':
                    order_value = portfolio_value * decision.recommended_allocation
                    quantity = order_value / current_price
                    
                    order_request = OrderRequest(
                        symbol=decision.symbol,
                        side='BUY',
                        quantity=round(quantity, 8),
                        order_type='MARKET'
                    )
                    
                elif decision.action == 'SELL':
                    # Get position quantity
                    positions = await alpaca_executor.get_positions()
                    position = next((p for p in positions if p.symbol == decision.symbol), None)
                    
                    if not position:
                        logger.warning(f"No position found for {decision.symbol} to sell")
                        continue
                    
                    order_request = OrderRequest(
                        symbol=decision.symbol,
                        side='SELL',
                        quantity=position.quantity,
                        order_type='MARKET'
                    )
                
                # Execute the order
                result = await alpaca_executor.place_order(order_request)
                
                if result.success:
                    executed_count += 1
                    logger.info(f"Executed {decision.action} order for {decision.symbol}")
                    
                    # Store trading signal
                    await self._store_trading_signal(decision, result)
                else:
                    logger.error(f"Failed to execute {decision.action} for {decision.symbol}: {result.error_message}")
                
            except Exception as e:
                logger.error(f"Error executing decision for {decision.symbol}: {e}")
                continue
        
        logger.info(f"Executed {executed_count} trading decisions")
    
    async def _update_market_sentiment(self):
        """Update market sentiment data."""
        try:
            logger.info("Updating market sentiment")
            
            # Update sentiment for all trading pairs
            for symbol in settings.trading.trading_pairs:
                await sentiment_analyzer.analyze_symbol_sentiment(symbol)
            
            # Update general market sentiment
            await sentiment_analyzer.update_fear_greed_index()
            
            self.last_sentiment_update = datetime.utcnow()
            logger.info("Market sentiment update completed")
            
        except Exception as e:
            logger.error(f"Error updating market sentiment: {e}")
    
    async def _update_portfolio_metrics(self):
        """Update portfolio performance metrics."""
        try:
            # Get current positions and account info
            positions = await alpaca_executor.get_positions()
            account_info = await alpaca_executor.get_account_info()
            
            # Calculate portfolio metrics
            total_value = account_info.get('portfolio_value', 0)
            total_pnl = sum(float(pos.unrealized_pnl) for pos in positions)
            
            # Store portfolio snapshot
            with db_manager.get_session() as session:
                snapshot = PortfolioSnapshot(
                    timestamp=datetime.utcnow(),
                    total_value=total_value,
                    cash_balance=account_info.get('cash', 0),
                    total_positions=len(positions),
                    unrealized_pnl=total_pnl,
                    # Additional metrics would be calculated here
                    sharpe_ratio=0.0,  # Placeholder
                    max_drawdown=0.0,  # Placeholder
                    win_rate=0.0  # Placeholder
                )
                session.add(snapshot)
                session.commit()
            
            logger.debug(f"Updated portfolio metrics: Value=${total_value:.2f}, PnL=${total_pnl:.2f}")
            
        except Exception as e:
            logger.error(f"Error updating portfolio metrics: {e}")
    
    async def _monitor_risk_limits(self):
        """Monitor and enforce risk limits."""
        try:
            # Get current portfolio state
            positions = await alpaca_executor.get_positions()
            account_info = await alpaca_executor.get_account_info()
            
            # Check drawdown limits
            portfolio_value = account_info.get('portfolio_value', 100000)
            
            # Check individual position sizes
            for position in positions:
                position_value = float(position.market_value)
                allocation = position_value / portfolio_value
                
                if allocation > settings.trading.max_portfolio_allocation * 1.2:  # 20% buffer
                    logger.warning(f"Position {position.symbol} exceeds allocation limit: {allocation:.2%}")
                    # Could implement automatic position reduction here
            
            # Check correlation limits
            await risk_manager.update_correlation_matrix()
            
        except Exception as e:
            logger.error(f"Error monitoring risk limits: {e}")
    
    def _should_update_sentiment(self) -> bool:
        """Check if sentiment should be updated."""
        return (datetime.utcnow() - self.last_sentiment_update).total_seconds() >= (self.sentiment_update_minutes * 60)
    
    def _determine_trading_signal(self, ml_prediction: Dict, sentiment_score: float, 
                                 risk_metrics: Dict, features) -> Tuple[str, float]:
        """Determine final trading signal from all inputs."""
        try:
            # Get ensemble prediction
            expected_return = ml_prediction.get('expected_return', 0.0)
            confidence = ml_prediction.get('confidence_score', 0.0)
            
            # Convert expected return to score (0-1 scale)
            # Map returns to probability scale: -5% to +5% return maps to 0-1 score
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
            current_price = analysis.technical_features.get('close_price', 0)
            atr = analysis.technical_features.get('atr_14', current_price * 0.02)
            
            # Use ATR-based stop loss
            stop_loss = current_price - (atr * settings.trading.atr_stop_multiplier)
            return max(stop_loss, current_price * 0.95)  # Maximum 5% stop loss
            
        except Exception:
            return None
    
    def _calculate_take_profit(self, analysis: MarketAnalysis) -> Optional[float]:
        """Calculate take profit price."""
        try:
            current_price = analysis.technical_features.get('close_price', 0)
            expected_return = analysis.risk_metrics.get('expected_return', 0.05)
            
            return current_price * (1 + max(expected_return, 0.08))  # Minimum 8% target
            
        except Exception:
            return None
    
    async def _get_current_price(self, symbol: str) -> Optional[float]:
        """Get current market price for symbol from Binance with database fallback."""
        try:
            # Convert symbol format if needed (Alpaca format BTC/USD -> Binance format BTCUSDT)
            binance_symbol = self._convert_to_binance_symbol(symbol)
            
            # First try to get real-time price from Binance
            current_price = await data_streamer.get_current_price(binance_symbol)
            
            if current_price:
                logger.debug(f"Got real-time price for {symbol} ({binance_symbol}): ${current_price}")
                return current_price
            
            logger.warning(f"Could not get real-time price for {symbol}, falling back to database")
            
            # Fallback to database data
            df = technical_analyzer.get_market_data(binance_symbol, timeframe='1h', limit=1)
            if not df.empty:
                fallback_price = float(df['close'].iloc[-1])
                logger.info(f"Using database fallback price for {symbol}: ${fallback_price}")
                return fallback_price
                
            logger.error(f"No price data available for {symbol} from any source")
            return None
            
        except Exception as e:
            logger.error(f"Error getting current price for {symbol}: {e}")
            return None
    
    def _convert_to_binance_symbol(self, symbol: str) -> str:
        """Convert symbol from Alpaca format (BTC/USD) to Binance format (BTCUSDT)."""
        try:
            # If already in Binance format, return as-is
            if '/' not in symbol and symbol.endswith('USDT'):
                return symbol
            
            # Convert from Alpaca format
            if '/' in symbol:
                base, quote = symbol.split('/')
                if quote == 'USD':
                    return f"{base}USDT"
                else:
                    return f"{base}{quote}"
            
            # If symbol doesn't match expected format, try to find it in trading pairs
            for trading_pair in settings.trading_pairs:
                if symbol.replace('/', '').replace('USD', 'USDT') == trading_pair.replace('USDT', 'USDT'):
                    return trading_pair
            
            # Default fallback
            return symbol
            
        except Exception as e:
            logger.warning(f"Error converting symbol format for {symbol}: {e}")
            return symbol
    
    async def _get_symbol_sentiment(self, symbol: str) -> float:
        """Get sentiment score for symbol."""
        try:
            return await sentiment_analyzer.get_symbol_sentiment(symbol)
        except Exception:
            return 0.5  # Neutral sentiment as fallback
    
    async def _calculate_risk_metrics(self, symbol: str, df: pd.DataFrame) -> Dict[str, float]:
        """Calculate risk metrics for symbol."""
        try:
            volatility = df['close'].pct_change().std() * (24 ** 0.5)  # Daily volatility
            
            return {
                'volatility': volatility,
                'risk_score': min(volatility * 10, 1.0),  # Normalize to 0-1
                'expected_return': 0.05  # Placeholder
            }
        except Exception:
            return {'volatility': 0.02, 'risk_score': 0.5, 'expected_return': 0.05}
    
    async def _store_trading_signal(self, decision: TradingDecision, order_result):
        """Store trading signal in database."""
        try:
            analysis = self.market_analysis_cache.get(decision.symbol)
            if not analysis:
                return
            
            with db_manager.get_session() as session:
                signal = TradingSignal(
                    symbol=decision.symbol,
                    timestamp=datetime.utcnow(),
                    signal_type=decision.action,
                    confidence_score=decision.confidence,
                    timeframe='1h',
                    ensemble_score=analysis.ml_prediction.get('expected_return', 0.0),
                    random_forest_score=analysis.ml_prediction.get('model_scores', {}).get('random_forest', 0.0),
                    lstm_score=analysis.ml_prediction.get('model_scores', {}).get('lstm', 0.0),
                    transformer_score=analysis.ml_prediction.get('model_scores', {}).get('transformer', 0.0),
                    expected_return=analysis.ml_prediction.get('expected_return', 0.0),
                    risk_score=analysis.ml_prediction.get('risk_score', 0.5),
                    technical_features=analysis.technical_features,
                    recommended_allocation=decision.recommended_allocation
                )
                session.add(signal)
                session.commit()
                
        except Exception as e:
            logger.error(f"Error storing trading signal: {e}")
    
    async def _load_active_signals(self):
        """Load active trading signals from database."""
        try:
            with db_manager.get_session() as session:
                signals = session.query(TradingSignal).filter(
                    TradingSignal.is_active == True,
                    TradingSignal.timestamp >= datetime.utcnow() - timedelta(hours=24)
                ).all()
                
                for signal in signals:
                    self.active_signals[signal.symbol] = signal
                
                logger.info(f"Loaded {len(signals)} active signals")
                
        except Exception as e:
            logger.error(f"Error loading active signals: {e}")
    
    async def _wait_for_next_cycle(self, cycle_start: datetime):
        """Wait for the next analysis cycle."""
        cycle_duration = (datetime.utcnow() - cycle_start).total_seconds()
        wait_time = max(0, (self.analysis_cycle_minutes * 60) - cycle_duration)
        
        if wait_time > 0:
            logger.debug(f"Waiting {wait_time:.1f}s for next cycle")
            await asyncio.sleep(wait_time)
    
    async def _cleanup_old_data(self):
        """Clean up old data to maintain performance."""
        try:
            # This would implement data cleanup logic
            # For now, just log the action
            logger.debug("Performing data cleanup")
        except Exception as e:
            logger.error(f"Error in data cleanup: {e}")
    
    async def _handle_shutdown_orders(self):
        """Handle any pending orders during shutdown."""
        try:
            # Implementation would cancel pending orders if needed
            logger.info("Handling shutdown procedures")
        except Exception as e:
            logger.error(f"Error handling shutdown: {e}")


# Global trading engine instance
trading_engine = TradingEngine()
