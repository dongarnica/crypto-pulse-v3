"""
Main backtesting engine for historical strategy validation.
"""

import asyncio
import logging
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any
from decimal import Decimal
import uuid
from collections import defaultdict

from .config import BacktestConfig
from .results import TradeResult, PortfolioSnapshot, BacktestResults
from src.core.database import db_manager
from src.core.models import MarketData, Position
from src.data.technical_analysis import technical_analyzer
from src.data.sentiment import sentiment_analyzer
from src.ml.ensemble import ml_ensemble
from src.risk.manager import risk_manager

logger = logging.getLogger(__name__)


class BacktestPortfolio:
    """Portfolio state management for backtesting."""
    
    def __init__(self, initial_capital: Decimal):
        self.initial_capital = initial_capital
        self.cash = initial_capital
        self.positions: Dict[str, Decimal] = {}  # symbol -> quantity
        self.position_costs: Dict[str, Decimal] = {}  # symbol -> avg cost basis
        self.trade_history: List[TradeResult] = []
        self.snapshots: List[PortfolioSnapshot] = []
        
    def get_total_value(self, prices: Dict[str, float]) -> Decimal:
        """Calculate total portfolio value."""
        position_value = sum(
            self.positions.get(symbol, Decimal('0')) * Decimal(str(price))
            for symbol, price in prices.items()
        )
        return self.cash + position_value
    
    def get_position_value(self, symbol: str, price: float) -> Decimal:
        """Get position value for a symbol."""
        quantity = self.positions.get(symbol, Decimal('0'))
        return quantity * Decimal(str(price))
    
    def execute_trade(self, trade: TradeResult, current_price: float) -> bool:
        """Execute a trade and update portfolio state."""
        try:
            if trade.side == 'BUY':
                cost = trade.quantity * trade.price + trade.commission + trade.slippage
                if cost <= self.cash:
                    self.cash -= cost
                    current_quantity = self.positions.get(trade.symbol, Decimal('0'))
                    current_cost = self.position_costs.get(trade.symbol, Decimal('0'))
                    
                    # Update position and average cost basis
                    total_quantity = current_quantity + trade.quantity
                    total_cost = (current_quantity * current_cost + 
                                trade.quantity * trade.price)
                    
                    self.positions[trade.symbol] = total_quantity
                    if total_quantity > 0:
                        self.position_costs[trade.symbol] = total_cost / total_quantity
                    
                    self.trade_history.append(trade)
                    return True
                else:
                    logger.warning(f"Insufficient cash for trade: {trade.trade_id}")
                    return False
                    
            elif trade.side == 'SELL':
                current_quantity = self.positions.get(trade.symbol, Decimal('0'))
                if trade.quantity <= current_quantity:
                    proceeds = trade.quantity * trade.price - trade.commission - trade.slippage
                    self.cash += proceeds
                    
                    # Calculate P&L
                    avg_cost = self.position_costs.get(trade.symbol, Decimal('0'))
                    trade.pnl = (trade.price - avg_cost) * trade.quantity - trade.commission - trade.slippage
                    trade.pnl_percentage = float(trade.pnl / (avg_cost * trade.quantity)) if avg_cost > 0 else 0.0
                    
                    # Update position
                    new_quantity = current_quantity - trade.quantity
                    self.positions[trade.symbol] = new_quantity
                    if new_quantity == 0:
                        del self.position_costs[trade.symbol]
                    
                    self.trade_history.append(trade)
                    return True
                else:
                    logger.warning(f"Insufficient position for trade: {trade.trade_id}")
                    return False
                    
        except Exception as e:
            logger.error(f"Error executing trade {trade.trade_id}: {e}")
            return False
    
    def take_snapshot(self, timestamp: datetime, prices: Dict[str, float]) -> PortfolioSnapshot:
        """Take a portfolio snapshot."""
        position_values = {
            symbol: float(self.get_position_value(symbol, prices.get(symbol, 0.0)))
            for symbol in self.positions.keys()
        }
        
        total_value = float(self.get_total_value(prices))
        
        snapshot = PortfolioSnapshot(
            timestamp=timestamp,
            total_value=total_value,
            cash=float(self.cash),
            positions=dict(self.positions),
            position_values=position_values,
            unrealized_pnl=total_value - float(self.initial_capital) - sum(
                float(trade.pnl or 0) for trade in self.trade_history
            )
        )
        
        self.snapshots.append(snapshot)
        return snapshot


class BacktestEngine:
    """Main backtesting engine for historical strategy validation."""
    
    def __init__(self, config: BacktestConfig):
        self.config = config
        self.portfolio = BacktestPortfolio(config.initial_capital)
        self.current_date = config.start_date
        
        # Market data cache
        self.market_data_cache: Dict[str, pd.DataFrame] = {}
        
        # Analysis components
        self.technical_analyzer = technical_analyzer
        self.sentiment_analyzer = sentiment_analyzer
        self.ml_ensemble = ml_ensemble
        self.risk_manager = risk_manager
        
        # Tracking
        self.signals_generated = 0
        self.trades_executed = 0
        self.analysis_errors = 0
        
    async def load_historical_data(self) -> bool:
        """Load historical market data for all symbols."""
        logger.info("Loading historical market data...")
        
        try:
            for symbol in self.config.symbols:
                # Load data with some buffer for technical indicators
                start_buffer = self.config.start_date - timedelta(days=100)
                
                data = await db_manager.get_historical_data(
                    symbol=symbol,
                    start_date=start_buffer,
                    end_date=self.config.end_date,
                    interval_minutes=self.config.analysis_interval_minutes
                )
                
                if data.empty:
                    logger.warning(f"No data found for {symbol}")
                    continue
                    
                self.market_data_cache[symbol] = data
                logger.info(f"Loaded {len(data)} data points for {symbol}")
            
            return len(self.market_data_cache) > 0
            
        except Exception as e:
            logger.error(f"Error loading historical data: {e}")
            return False
    
    def get_market_data_at_time(self, timestamp: datetime) -> Dict[str, Dict[str, float]]:
        """Get market data for all symbols at a specific timestamp."""
        market_data = {}
        
        for symbol, data in self.market_data_cache.items():
            # Find the most recent data point before or at the timestamp
            filtered_data = data[data.index <= timestamp]
            if not filtered_data.empty:
                latest = filtered_data.iloc[-1]
                market_data[symbol] = {
                    'open': latest['open'],
                    'high': latest['high'],
                    'low': latest['low'],
                    'close': latest['close'],
                    'volume': latest['volume']
                }
        
        return market_data
    
    async def analyze_market_at_time(self, timestamp: datetime) -> Dict[str, Dict[str, Any]]:
        """Perform market analysis for all symbols at a specific time."""
        analysis_results = {}
        
        for symbol in self.config.symbols:
            try:
                if symbol not in self.market_data_cache:
                    continue
                
                # Get data up to current timestamp
                data = self.market_data_cache[symbol]
                current_data = data[data.index <= timestamp]
                
                if len(current_data) < 50:  # Need enough data for analysis
                    continue
                
                # Technical analysis
                technical_features = await self.technical_analyzer.analyze_symbol(
                    symbol, current_data.tail(100)
                )
                
                # ML prediction (simulate using historical data)
                ml_prediction = await self.ml_ensemble.predict_batch([{
                    'symbol': symbol,
                    'features': technical_features.__dict__ if technical_features else {}
                }])
                
                # Risk analysis
                current_price = current_data.iloc[-1]['close']
                risk_metrics = self.risk_manager.calculate_position_risk(
                    symbol, float(current_price), current_data.tail(30)
                )
                
                analysis_results[symbol] = {
                    'technical_features': technical_features,
                    'ml_prediction': ml_prediction[0] if ml_prediction else {},
                    'risk_metrics': risk_metrics,
                    'current_price': current_price
                }
                
            except Exception as e:
                logger.error(f"Error analyzing {symbol} at {timestamp}: {e}")
                self.analysis_errors += 1
                continue
        
        return analysis_results
    
    def generate_trading_signals(self, analysis_results: Dict[str, Dict[str, Any]], 
                               timestamp: datetime) -> List[Dict[str, Any]]:
        """Generate trading signals based on analysis results."""
        signals = []
        
        for symbol, analysis in analysis_results.items():
            try:
                technical = analysis.get('technical_features')
                ml_pred = analysis.get('ml_prediction', {})
                risk_metrics = analysis.get('risk_metrics', {})
                current_price = analysis.get('current_price', 0.0)
                
                if not technical or current_price <= 0:
                    continue
                
                # Signal generation logic (simplified version of main engine logic)
                signal_strength = 0.0
                signal_type = 'HOLD'
                
                # Technical signal components
                tech_score = 0.0
                if hasattr(technical, 'rsi_14'):
                    if technical.rsi_14 < 30:  # Oversold
                        tech_score += 0.3
                    elif technical.rsi_14 > 70:  # Overbought
                        tech_score -= 0.3
                
                if hasattr(technical, 'macd_bullish'):
                    if technical.macd_bullish:
                        tech_score += 0.2
                    else:
                        tech_score -= 0.2
                
                # ML signal component
                ml_score = ml_pred.get('signal_strength', 0.0)
                
                # Combined signal
                signal_strength = (tech_score * 0.4 + ml_score * 0.6)
                
                # Apply thresholds
                if signal_strength > 0.3:
                    signal_type = 'BUY'
                elif signal_strength < -0.3:
                    signal_type = 'SELL'
                
                # Risk-based filtering
                max_drawdown = risk_metrics.get('max_drawdown_1w', 0.0)
                if abs(max_drawdown) > self.config.max_drawdown_threshold:
                    signal_type = 'HOLD'  # Too risky
                
                if signal_type != 'HOLD':
                    signals.append({
                        'symbol': symbol,
                        'signal': signal_type,
                        'strength': abs(signal_strength),
                        'price': current_price,
                        'timestamp': timestamp,
                        'technical_score': tech_score,
                        'ml_score': ml_score,
                        'risk_metrics': risk_metrics
                    })
                    self.signals_generated += 1
                    
            except Exception as e:
                logger.error(f"Error generating signal for {symbol}: {e}")
                continue
        
        return signals
    
    def calculate_position_size(self, signal: Dict[str, Any], 
                              current_portfolio_value: float) -> Decimal:
        """Calculate position size based on signal strength and risk management."""
        try:
            base_allocation = (self.config.min_portfolio_allocation + 
                             (self.config.max_portfolio_allocation - self.config.min_portfolio_allocation) * 
                             signal['strength'])
            
            # Apply Kelly Criterion if available
            risk_metrics = signal.get('risk_metrics', {})
            kelly_fraction = risk_metrics.get('kelly_fraction', base_allocation)
            
            # Use more conservative of the two
            final_allocation = min(base_allocation, kelly_fraction, self.config.max_portfolio_allocation)
            
            # Calculate dollar amount
            dollar_amount = current_portfolio_value * final_allocation
            position_size = Decimal(str(dollar_amount / signal['price']))
            
            return position_size
            
        except Exception as e:
            logger.error(f"Error calculating position size: {e}")
            return Decimal('0')
    
    def create_trade(self, signal: Dict[str, Any], position_size: Decimal) -> TradeResult:
        """Create a trade result from a signal."""
        price = Decimal(str(signal['price']))
        commission = price * position_size * Decimal(str(self.config.commission_rate))
        slippage = price * position_size * Decimal(str(self.config.slippage_bps / 10000))
        
        return TradeResult(
            trade_id=str(uuid.uuid4()),
            symbol=signal['symbol'],
            timestamp=signal['timestamp'],
            side=signal['signal'],
            quantity=position_size,
            price=price,
            commission=commission,
            slippage=slippage,
            signal_confidence=signal['strength'],
            ml_prediction=signal.get('ml_score'),
            technical_indicators=signal.get('technical_score')
        )
    
    async def run_backtest(self) -> BacktestResults:
        """Run the complete backtest."""
        logger.info(f"Starting backtest from {self.config.start_date} to {self.config.end_date}")
        
        # Load historical data
        if not await self.load_historical_data():
            raise ValueError("Failed to load historical data")
        
        # Generate time steps
        current_time = self.config.start_date
        interval = timedelta(minutes=self.config.analysis_interval_minutes)
        
        while current_time <= self.config.end_date:
            try:
                # Get current market prices for portfolio valuation
                market_data = self.get_market_data_at_time(current_time)
                current_prices = {symbol: data['close'] for symbol, data in market_data.items()}
                
                if not current_prices:
                    current_time += interval
                    continue
                
                # Take portfolio snapshot
                portfolio_value = float(self.portfolio.get_total_value(current_prices))
                self.portfolio.take_snapshot(current_time, current_prices)
                
                # Perform market analysis
                analysis_results = await self.analyze_market_at_time(current_time)
                
                if not analysis_results:
                    current_time += interval
                    continue
                
                # Generate trading signals
                signals = self.generate_trading_signals(analysis_results, current_time)
                
                # Execute trades based on signals
                for signal in signals:
                    if signal['signal'] == 'BUY':
                        position_size = self.calculate_position_size(signal, portfolio_value)
                        if position_size > 0:
                            trade = self.create_trade(signal, position_size)
                            if self.portfolio.execute_trade(trade, signal['price']):
                                self.trades_executed += 1
                                logger.debug(f"Executed BUY trade: {trade.symbol} {trade.quantity}")
                    
                    elif signal['signal'] == 'SELL':
                        # Check if we have a position to sell
                        current_position = self.portfolio.positions.get(signal['symbol'], Decimal('0'))
                        if current_position > 0:
                            # Sell entire position for simplicity
                            trade = self.create_trade(signal, current_position)
                            if self.portfolio.execute_trade(trade, signal['price']):
                                self.trades_executed += 1
                                logger.debug(f"Executed SELL trade: {trade.symbol} {trade.quantity}")
                
                # Progress logging
                if current_time.day % 7 == 0:  # Weekly progress
                    logger.info(f"Backtest progress: {current_time.date()}, "
                              f"Portfolio value: ${portfolio_value:,.2f}")
                
                current_time += interval
                
            except Exception as e:
                logger.error(f"Error at timestamp {current_time}: {e}")
                current_time += interval
                continue
        
        # Final portfolio snapshot
        final_prices = self.get_market_data_at_time(self.config.end_date)
        if final_prices:
            final_prices_dict = {symbol: data['close'] for symbol, data in final_prices.items()}
            self.portfolio.take_snapshot(self.config.end_date, final_prices_dict)
        
        # Generate results
        results = BacktestResults(
            config=self.config,
            trades=self.portfolio.trade_history,
            portfolio_snapshots=self.portfolio.snapshots,
            start_date=self.config.start_date,
            end_date=self.config.end_date,
            initial_capital=float(self.config.initial_capital),
            final_capital=float(self.portfolio.get_total_value(final_prices_dict)) if final_prices_dict else 0.0
        )
        
        # Calculate all metrics
        results.calculate_metrics()
        
        logger.info(f"Backtest completed. Generated {self.signals_generated} signals, "
                   f"executed {self.trades_executed} trades, "
                   f"final portfolio value: ${results.final_capital:,.2f}")
        
        return results
