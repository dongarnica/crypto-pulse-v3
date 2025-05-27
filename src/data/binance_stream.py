"""
Real-time data ingestion from Binance WebSocket streams.
Handles market data, order book updates, and trade feeds.
"""

import asyncio
import json
import websockets
import logging
from datetime import datetime, timezone
from decimal import Decimal
from typing import Dict, List, Callable, Optional
from dataclasses import dataclass
import pandas as pd

from binance import AsyncClient, BinanceSocketManager
from config.settings import settings
from src.core.database import db_manager
from src.core.models import MarketData, OrderBookSnapshot

logger = logging.getLogger(__name__)


@dataclass
class KlineData:
    """Standardized kline/candlestick data structure."""
    symbol: str
    timestamp: datetime
    open_price: Decimal
    high_price: Decimal
    low_price: Decimal
    close_price: Decimal
    volume: Decimal
    quote_volume: Decimal
    trade_count: int
    taker_buy_volume: Decimal
    taker_buy_quote_volume: Decimal
    is_closed: bool


@dataclass
class OrderBookData:
    """Standardized order book data structure."""
    symbol: str
    timestamp: datetime
    bids: List[List[str]]  # [[price, quantity], ...]
    asks: List[List[str]]  # [[price, quantity], ...]
    last_update_id: int


class BinanceDataStreamer:
    """
    Manages real-time data streams from Binance WebSocket API.
    """
    
    def __init__(self):
        self.client: Optional[AsyncClient] = None
        self.socket_manager: Optional[BinanceSocketManager] = None
        self.active_streams: Dict[str, asyncio.Task] = {}
        self.callbacks: Dict[str, List[Callable]] = {
            'kline': [],
            'orderbook': [],
            'trade': []
        }
        
    async def initialize(self):
        """Initialize Binance client and socket manager."""
        try:
            self.client = await AsyncClient.create(
                api_key=settings.api.binance_api_key,
                api_secret=settings.api.binance_secret_key
            )
            self.socket_manager = BinanceSocketManager(self.client)
            logger.info("Binance client initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize Binance client: {e}")
            raise
    
    async def start_kline_streams(self, symbols: List[str], interval: str = "1h"):
        """
        Start kline/candlestick streams for multiple symbols.
        
        Args:
            symbols: List of trading pairs (e.g., ['BTCUSDT', 'ETHUSDT'])
            interval: Kline interval (1m, 5m, 15m, 1h, 4h, 1d)
        """
        if not self.socket_manager:
            await self.initialize()
        
        for symbol in symbols:
            stream_name = f"kline_{symbol}_{interval}"
            if stream_name not in self.active_streams:
                task = asyncio.create_task(
                    self._handle_kline_stream(symbol, interval)
                )
                self.active_streams[stream_name] = task
                logger.info(f"Started kline stream for {symbol} ({interval})")
    
    async def start_orderbook_streams(self, symbols: List[str], depth: int = 20):
        """
        Start order book depth streams for multiple symbols.
        
        Args:
            symbols: List of trading pairs
            depth: Order book depth (5, 10, 20)
        """
        if not self.socket_manager:
            await self.initialize()
        
        for symbol in symbols:
            stream_name = f"depth_{symbol}_{depth}"
            if stream_name not in self.active_streams:
                task = asyncio.create_task(
                    self._handle_orderbook_stream(symbol, depth)
                )
                self.active_streams[stream_name] = task
                logger.info(f"Started order book stream for {symbol} (depth {depth})")
    
    async def _handle_kline_stream(self, symbol: str, interval: str):
        """Handle individual kline stream."""
        try:
            ts = self.socket_manager.kline_socket(symbol, interval)
            async with ts as stream:
                while True:
                    msg = await stream.recv()
                    await self._process_kline_message(msg)
        except Exception as e:
            logger.error(f"Error in kline stream for {symbol}: {e}")
            # Restart stream after delay
            await asyncio.sleep(5)
            await self.start_kline_streams([symbol], interval)
    
    async def _handle_orderbook_stream(self, symbol: str, depth: int):
        """Handle individual order book stream."""
        try:
            ts = self.socket_manager.depth_socket(symbol, depth)
            async with ts as stream:
                while True:
                    msg = await stream.recv()
                    await self._process_orderbook_message(msg)
        except Exception as e:
            logger.error(f"Error in order book stream for {symbol}: {e}")
            # Restart stream after delay
            await asyncio.sleep(5)
            await self.start_orderbook_streams([symbol], depth)
    
    async def _process_kline_message(self, msg: dict):
        """Process incoming kline message and store to database."""
        try:
            kline_data = msg['k']
            
            # Parse kline data
            kline = KlineData(
                symbol=kline_data['s'],
                timestamp=datetime.fromtimestamp(
                    kline_data['t'] / 1000, tz=timezone.utc
                ).replace(tzinfo=None),
                open_price=Decimal(kline_data['o']),
                high_price=Decimal(kline_data['h']),
                low_price=Decimal(kline_data['l']),
                close_price=Decimal(kline_data['c']),
                volume=Decimal(kline_data['v']),
                quote_volume=Decimal(kline_data['q']),
                trade_count=int(kline_data['n']),
                taker_buy_volume=Decimal(kline_data['V']),
                taker_buy_quote_volume=Decimal(kline_data['Q']),
                is_closed=kline_data['x']
            )
            
            # Only process closed candles for database storage
            if kline.is_closed:
                await self._store_market_data(kline)
            
            # Call registered callbacks
            for callback in self.callbacks['kline']:
                await callback(kline)
                
        except Exception as e:
            logger.error(f"Error processing kline message: {e}")
    
    async def _process_orderbook_message(self, msg: dict):
        """Process incoming order book message."""
        try:
            orderbook = OrderBookData(
                symbol=msg['s'],
                timestamp=datetime.fromtimestamp(
                    msg['E'] / 1000, tz=timezone.utc
                ).replace(tzinfo=None),
                bids=msg['b'],
                asks=msg['a'],
                last_update_id=msg['u']
            )
            
            # Store order book snapshot
            await self._store_orderbook_data(orderbook)
            
            # Call registered callbacks
            for callback in self.callbacks['orderbook']:
                await callback(orderbook)
                
        except Exception as e:
            logger.error(f"Error processing order book message: {e}")
    
    async def _store_market_data(self, kline: KlineData):
        """Store market data to database."""
        try:
            with db_manager.get_session() as session:
                # Check if record already exists
                existing = session.query(MarketData).filter(
                    MarketData.symbol == kline.symbol,
                    MarketData.timestamp == kline.timestamp
                ).first()
                
                if existing:
                    # Update existing record
                    existing.open_price = kline.open_price
                    existing.high_price = kline.high_price
                    existing.low_price = kline.low_price
                    existing.close_price = kline.close_price
                    existing.volume = kline.volume
                    existing.quote_volume = kline.quote_volume
                    existing.trade_count = kline.trade_count
                    existing.taker_buy_volume = kline.taker_buy_volume
                    existing.taker_buy_quote_volume = kline.taker_buy_quote_volume
                else:
                    # Create new record
                    market_data = MarketData(
                        symbol=kline.symbol,
                        timestamp=kline.timestamp,
                        open_price=kline.open_price,
                        high_price=kline.high_price,
                        low_price=kline.low_price,
                        close_price=kline.close_price,
                        volume=kline.volume,
                        quote_volume=kline.quote_volume,
                        trade_count=kline.trade_count,
                        taker_buy_volume=kline.taker_buy_volume,
                        taker_buy_quote_volume=kline.taker_buy_quote_volume
                    )
                    session.add(market_data)
                
                session.commit()
                logger.debug(f"Stored market data for {kline.symbol} at {kline.timestamp}")
                
        except Exception as e:
            logger.error(f"Error storing market data: {e}")
    
    async def _store_orderbook_data(self, orderbook: OrderBookData):
        """Store order book data to database."""
        try:
            # Calculate metrics
            best_bid = float(orderbook.bids[0][0]) if orderbook.bids else 0
            best_ask = float(orderbook.asks[0][0]) if orderbook.asks else 0
            bid_ask_spread = best_ask - best_bid if best_bid and best_ask else 0
            mid_price = (best_bid + best_ask) / 2 if best_bid and best_ask else 0
            
            # Calculate volume imbalance
            total_bid_volume = sum(float(bid[1]) for bid in orderbook.bids[:10])
            total_ask_volume = sum(float(ask[1]) for ask in orderbook.asks[:10])
            total_volume = total_bid_volume + total_ask_volume
            imbalance_ratio = (
                (total_bid_volume - total_ask_volume) / total_volume
                if total_volume > 0 else 0
            )
            
            with db_manager.get_session() as session:
                orderbook_snapshot = OrderBookSnapshot(
                    symbol=orderbook.symbol,
                    timestamp=orderbook.timestamp,
                    bids=orderbook.bids,
                    asks=orderbook.asks,
                    bid_ask_spread=bid_ask_spread,
                    mid_price=Decimal(str(mid_price)),
                    total_bid_volume=Decimal(str(total_bid_volume)),
                    total_ask_volume=Decimal(str(total_ask_volume)),
                    imbalance_ratio=imbalance_ratio
                )
                session.add(orderbook_snapshot)
                session.commit()
                
                logger.debug(f"Stored order book for {orderbook.symbol}")
                
        except Exception as e:
            logger.error(f"Error storing order book data: {e}")
    
    def register_callback(self, stream_type: str, callback: Callable):
        """Register callback for stream data."""
        if stream_type in self.callbacks:
            self.callbacks[stream_type].append(callback)
            logger.info(f"Registered callback for {stream_type} stream")
    
    async def stop_all_streams(self):
        """Stop all active streams."""
        for stream_name, task in self.active_streams.items():
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                pass
            logger.info(f"Stopped stream: {stream_name}")
        
        self.active_streams.clear()
        
        if self.client:
            await self.client.close_connection()
            logger.info("Closed Binance client connection")


class DataStreamer:
    """Main data streaming coordinator."""
    
    def __init__(self):
        self.binance_streamer = BinanceDataStreamer()
        self.is_running = False
    
    async def start(self):
        """Start all data streams."""
        if self.is_running:
            logger.warning("Data streamer is already running")
            return
        
        try:
            logger.info("Starting data streams...")
            
            # Initialize Binance streamer
            await self.binance_streamer.initialize()
            
            # Start 1-hour kline streams for all trading pairs
            await self.binance_streamer.start_kline_streams(
                settings.trading.trading_pairs, "1h"
            )
            
            # Start 4-hour kline streams for ML model training
            await self.binance_streamer.start_kline_streams(
                settings.trading.trading_pairs, "4h"
            )
            
            # Start order book streams for major pairs
            major_pairs = settings.trading.trading_pairs[:10]  # Top 10 pairs
            await self.binance_streamer.start_orderbook_streams(major_pairs)
            
            self.is_running = True
            logger.info(f"Data streams started for {len(settings.trading.trading_pairs)} pairs")
            
        except Exception as e:
            logger.error(f"Failed to start data streams: {e}")
            raise
    
    async def stop(self):
        """Stop all data streams."""
        if not self.is_running:
            return
        
        logger.info("Stopping data streams...")
        await self.binance_streamer.stop_all_streams()
        self.is_running = False
        logger.info("Data streams stopped")
    
    async def initialize(self):
        """Initialize all data stream components."""
        await self.binance_streamer.initialize()
    
    async def start_kline_streams(self, symbols: List[str], interval: str = "1h"):
        """Start kline streams for specified symbols and interval."""
        await self.binance_streamer.start_kline_streams(symbols, interval)
    
    async def start_orderbook_streams(self, symbols: List[str], depth: int = 20):
        """Start order book streams for specified symbols and depth."""
        await self.binance_streamer.start_orderbook_streams(symbols, depth)
    
    async def stop_all_streams(self):
        """Stop all active data streams."""
        await self.binance_streamer.stop_all_streams()
        self.is_running = False

    def get_binance_streamer(self) -> BinanceDataStreamer:
        """Get Binance streamer instance for callback registration."""
        return self.binance_streamer


# Global data streamer instance
data_streamer = DataStreamer()
