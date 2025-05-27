"""
Database models for the Crypto Pulse trading system.
Optimized for time-series data with TimescaleDB.
"""

from sqlalchemy import (
    Column, Integer, String, Float, DateTime, Boolean, 
    Text, Index, ForeignKey, Numeric, BigInteger
)
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import relationship
from sqlalchemy.dialects.postgresql import UUID, JSONB
from datetime import datetime
import uuid

Base = declarative_base()


class MarketData(Base):
    """
    Time-series market data for cryptocurrency pairs.
    Hypertable for high-frequency ingestion.
    """
    __tablename__ = 'market_data'
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    symbol = Column(String(20), nullable=False, index=True)
    timestamp = Column(DateTime, nullable=False, index=True)
    
    # OHLCV data
    open_price = Column(Numeric(20, 8), nullable=False)
    high_price = Column(Numeric(20, 8), nullable=False)
    low_price = Column(Numeric(20, 8), nullable=False)
    close_price = Column(Numeric(20, 8), nullable=False)
    volume = Column(Numeric(20, 8), nullable=False)
    
    # Additional market metrics
    quote_volume = Column(Numeric(20, 8))
    trade_count = Column(Integer)
    taker_buy_volume = Column(Numeric(20, 8))
    taker_buy_quote_volume = Column(Numeric(20, 8))
    
    # Technical indicators (calculated)
    rsi_14 = Column(Float)
    rsi_30 = Column(Float)
    macd_line = Column(Float)
    macd_signal = Column(Float)
    macd_histogram = Column(Float)
    bb_upper = Column(Float)
    bb_middle = Column(Float)
    bb_lower = Column(Float)
    atr_14 = Column(Float)
    
    # Volume indicators
    volume_sma_20 = Column(Float)
    volume_weighted_price = Column(Float)
    
    __table_args__ = (
        Index('idx_market_data_symbol_timestamp', 'symbol', 'timestamp'),
        Index('idx_market_data_timestamp', 'timestamp'),
    )


class OrderBookSnapshot(Base):
    """
    Level 2 order book snapshots for deeper market analysis.
    """
    __tablename__ = 'order_book_snapshots'
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    symbol = Column(String(20), nullable=False, index=True)
    timestamp = Column(DateTime, nullable=False, index=True)
    
    # Order book data as JSON
    bids = Column(JSONB)  # Array of [price, quantity] pairs
    asks = Column(JSONB)  # Array of [price, quantity] pairs
    
    # Calculated metrics
    bid_ask_spread = Column(Float)
    mid_price = Column(Numeric(20, 8))
    total_bid_volume = Column(Numeric(20, 8))
    total_ask_volume = Column(Numeric(20, 8))
    imbalance_ratio = Column(Float)  # (bid_vol - ask_vol) / (bid_vol + ask_vol)
    
    __table_args__ = (
        Index('idx_orderbook_symbol_timestamp', 'symbol', 'timestamp'),
    )


class TradingSignal(Base):
    """
    Generated trading signals from ML models and technical analysis.
    """
    __tablename__ = 'trading_signals'
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    symbol = Column(String(20), nullable=False, index=True)
    timestamp = Column(DateTime, nullable=False, default=datetime.utcnow)
    
    # Signal information
    signal_type = Column(String(20), nullable=False)  # 'BUY', 'SELL', 'HOLD'
    confidence_score = Column(Float, nullable=False)  # 0.0 to 1.0
    timeframe = Column(String(10), nullable=False)  # '1h', '4h', etc.
    
    # Model predictions
    random_forest_score = Column(Float)
    lstm_score = Column(Float)
    transformer_score = Column(Float)
    ensemble_score = Column(Float, nullable=False)
    
    # Risk metrics
    expected_return = Column(Float)
    risk_score = Column(Float)
    volatility_forecast = Column(Float)
    
    # Technical analysis features
    technical_features = Column(JSONB)  # Store feature vector
    market_regime = Column(String(20))  # 'trending', 'sideways', 'volatile'
    
    # Position sizing recommendation
    kelly_allocation = Column(Float)
    recommended_allocation = Column(Float)
    
    is_active = Column(Boolean, default=True)
    
    __table_args__ = (
        Index('idx_signals_symbol_timestamp', 'symbol', 'timestamp'),
        Index('idx_signals_active', 'is_active'),
    )


class Trade(Base):
    """
    Executed trades with detailed execution information.
    """
    __tablename__ = 'trades'
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    
    # Trade identification
    symbol = Column(String(20), nullable=False, index=True)
    exchange_order_id = Column(String(100), unique=True)
    signal_id = Column(UUID(as_uuid=True), ForeignKey('trading_signals.id'))
    
    # Trade details
    side = Column(String(10), nullable=False)  # 'BUY', 'SELL'
    order_type = Column(String(20), nullable=False)  # 'MARKET', 'LIMIT', 'STOP'
    status = Column(String(20), nullable=False)  # 'PENDING', 'FILLED', 'CANCELLED'
    
    # Quantities and prices
    quantity = Column(Numeric(20, 8), nullable=False)
    price = Column(Numeric(20, 8))  # NULL for market orders initially
    filled_quantity = Column(Numeric(20, 8), default=0)
    average_fill_price = Column(Numeric(20, 8))
    
    # Timestamps
    created_at = Column(DateTime, nullable=False, default=datetime.utcnow)
    executed_at = Column(DateTime)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    # Fees and costs
    commission = Column(Numeric(20, 8))
    commission_asset = Column(String(10))
    
    # Risk management
    stop_loss_price = Column(Numeric(20, 8))
    take_profit_price = Column(Numeric(20, 8))
    
    # Relationship
    signal = relationship("TradingSignal", backref="trades")
    
    __table_args__ = (
        Index('idx_trades_symbol_created', 'symbol', 'created_at'),
        Index('idx_trades_status', 'status'),
    )


class Position(Base):
    """
    Current and historical position tracking.
    """
    __tablename__ = 'positions'
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    symbol = Column(String(20), nullable=False, index=True)
    
    # Position details
    side = Column(String(10), nullable=False)  # 'LONG', 'SHORT'
    quantity = Column(Numeric(20, 8), nullable=False)
    average_entry_price = Column(Numeric(20, 8), nullable=False)
    current_price = Column(Numeric(20, 8))
    
    # P&L calculations
    unrealized_pnl = Column(Numeric(20, 8))
    realized_pnl = Column(Numeric(20, 8), default=0)
    total_pnl = Column(Numeric(20, 8))
    
    # Risk management
    stop_loss_price = Column(Numeric(20, 8))
    take_profit_price = Column(Numeric(20, 8))
    risk_amount = Column(Numeric(20, 8))
    
    # Timestamps
    opened_at = Column(DateTime, nullable=False, default=datetime.utcnow)
    closed_at = Column(DateTime)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    # Status
    is_open = Column(Boolean, default=True, index=True)
    
    __table_args__ = (
        Index('idx_positions_symbol_open', 'symbol', 'is_open'),
        Index('idx_positions_opened_at', 'opened_at'),
    )


class PortfolioSnapshot(Base):
    """
    Regular snapshots of portfolio performance and risk metrics.
    """
    __tablename__ = 'portfolio_snapshots'
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    timestamp = Column(DateTime, nullable=False, default=datetime.utcnow, index=True)
    
    # Portfolio values
    total_value = Column(Numeric(20, 8), nullable=False)
    cash_balance = Column(Numeric(20, 8), nullable=False)
    positions_value = Column(Numeric(20, 8), nullable=False)
    
    # Performance metrics
    daily_return = Column(Float)
    cumulative_return = Column(Float)
    sharpe_ratio = Column(Float)
    max_drawdown = Column(Float)
    current_drawdown = Column(Float)
    
    # Risk metrics
    var_95 = Column(Float)  # Value at Risk 95%
    portfolio_beta = Column(Float)
    total_exposure = Column(Float)
    leverage = Column(Float)
    
    # Position counts
    open_positions = Column(Integer)
    winning_positions = Column(Integer)
    losing_positions = Column(Integer)
    
    # Correlation analysis
    avg_correlation = Column(Float)
    max_correlation = Column(Float)


class SentimentData(Base):
    """
    Market sentiment data from various sources.
    """
    __tablename__ = 'sentiment_data'
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    timestamp = Column(DateTime, nullable=False, default=datetime.utcnow)
    symbol = Column(String(20), index=True)  # NULL for general market sentiment
    
    # Sentiment scores (-1 to 1)
    news_sentiment = Column(Float)
    social_sentiment = Column(Float)
    fear_greed_index = Column(Float)
    
    # Source data
    source = Column(String(50), nullable=False)  # 'perplexity', 'twitter', etc.
    raw_data = Column(JSONB)  # Store raw sentiment data
    
    # Processed metrics
    sentiment_trend = Column(Float)  # Rate of change
    sentiment_volatility = Column(Float)
    confidence_score = Column(Float)
    
    __table_args__ = (
        Index('idx_sentiment_timestamp', 'timestamp'),
        Index('idx_sentiment_symbol_timestamp', 'symbol', 'timestamp'),
    )


class ModelPerformance(Base):
    """
    Track performance of individual ML models over time.
    """
    __tablename__ = 'model_performance'
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    model_name = Column(String(50), nullable=False, index=True)
    model_version = Column(String(20), nullable=False)
    timestamp = Column(DateTime, nullable=False, default=datetime.utcnow)
    
    # Performance metrics
    accuracy = Column(Float)
    precision = Column(Float)
    recall = Column(Float)
    f1_score = Column(Float)
    auc_roc = Column(Float)
    
    # Trading-specific metrics
    directional_accuracy = Column(Float)
    profit_factor = Column(Float)
    win_rate = Column(Float)
    avg_return_per_signal = Column(Float)
    
    # Model metadata
    training_data_size = Column(Integer)
    feature_count = Column(Integer)
    hyperparameters = Column(JSONB)
    
    __table_args__ = (
        Index('idx_model_perf_name_timestamp', 'model_name', 'timestamp'),
    )
