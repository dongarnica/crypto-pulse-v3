"""
Trade execution system using Alpaca Markets.
Handles order placement, management, and position tracking.
"""

import asyncio
import logging
from decimal import Decimal
from typing import Dict, List, Optional, Tuple, Any
from datetime import datetime, timedelta
from dataclasses import dataclass, asdict
import pandas as pd

from alpaca.trading.client import TradingClient
from alpaca.trading.requests import (
    MarketOrderRequest, LimitOrderRequest, StopOrderRequest,
    GetOrdersRequest, ClosePositionRequest
)
from alpaca.trading.enums import (
    OrderSide, OrderType, TimeInForce, OrderStatus, PositionSide
)
from alpaca.common.exceptions import APIError

from src.core.database import db_manager
from src.core.models import Trade, Position, TradingSignal
from src.risk.manager import RiskManager, PositionSizeRecommendation
from config.settings import settings

logger = logging.getLogger(__name__)


@dataclass
class OrderRequest:
    """Standardized order request structure."""
    symbol: str
    side: str  # 'BUY', 'SELL'
    quantity: Decimal
    order_type: str  # 'MARKET', 'LIMIT', 'STOP'
    price: Optional[Decimal] = None
    stop_price: Optional[Decimal] = None
    time_in_force: str = 'GTC'
    signal_id: Optional[str] = None


@dataclass
class OrderResult:
    """Order execution result."""
    success: bool
    order_id: Optional[str] = None
    filled_quantity: Decimal = Decimal('0')
    average_fill_price: Optional[Decimal] = None
    status: str = 'PENDING'
    error_message: Optional[str] = None
    commission: Optional[Decimal] = None


@dataclass
class PositionUpdate:
    """Position update information."""
    symbol: str
    side: str
    quantity: Decimal
    current_price: Decimal
    unrealized_pnl: Decimal
    market_value: Decimal


class AlpacaExecutor:
    """
    Alpaca Markets trade execution system.
    """
    
    def __init__(self):
        self.trading_client = None
        self.risk_manager = RiskManager()
        self.active_orders: Dict[str, Any] = {}
        self.position_cache: Dict[str, Position] = {}
        self.last_position_update = datetime.utcnow()
        
    async def initialize(self):
        """Initialize Alpaca trading client."""
        try:
            self.trading_client = TradingClient(
                api_key=settings.alpaca_api_key,
                secret_key=settings.alpaca_secret_key,
                paper=settings.environment != 'production'
            )
            
            # Verify connection
            account = self.trading_client.get_account()
            logger.info(f"Connected to Alpaca - Account: {account.id}")
            logger.info(f"Buying power: ${account.buying_power}")
            
            # Load existing positions
            await self._load_positions()
            
        except Exception as e:
            logger.error(f"Failed to initialize Alpaca client: {e}")
            raise
    
    async def place_order(self, order_request: OrderRequest) -> OrderResult:
        """
        Place a trade order with comprehensive risk checks.
        """
        try:
            # Pre-trade risk validation
            risk_check = await self._validate_order_risk(order_request)
            if not risk_check['allowed']:
                return OrderResult(
                    success=False,
                    error_message=f"Risk validation failed: {risk_check['reason']}"
                )
            
            # Convert to Alpaca order format
            alpaca_order = self._convert_to_alpaca_order(order_request)
            
            # Submit order
            submitted_order = self.trading_client.submit_order(alpaca_order)
            
            # Store order in database
            trade_record = await self._create_trade_record(order_request, submitted_order)
            
            # Monitor order execution
            asyncio.create_task(self._monitor_order_execution(submitted_order.id))
            
            return OrderResult(
                success=True,
                order_id=submitted_order.id,
                status=submitted_order.status.value,
                filled_quantity=Decimal(str(submitted_order.filled_qty or 0)),
                average_fill_price=Decimal(str(submitted_order.filled_avg_price or 0)) if submitted_order.filled_avg_price else None
            )
            
        except APIError as e:
            logger.error(f"Alpaca API error placing order: {e}")
            return OrderResult(
                success=False,
                error_message=f"API Error: {str(e)}"
            )
        except Exception as e:
            logger.error(f"Unexpected error placing order: {e}")
            return OrderResult(
                success=False,
                error_message=f"System Error: {str(e)}"
            )
    
    async def cancel_order(self, order_id: str) -> bool:
        """Cancel an active order."""
        try:
            self.trading_client.cancel_order_by_id(order_id)
            
            # Update database
            await self._update_trade_status(order_id, 'CANCELLED')
            
            logger.info(f"Successfully cancelled order {order_id}")
            return True
            
        except Exception as e:
            logger.error(f"Error cancelling order {order_id}: {e}")
            return False
    
    async def close_position(self, symbol: str, percentage: float = 100.0) -> OrderResult:
        """Close a position partially or fully."""
        try:
            close_request = ClosePositionRequest(
                qty=percentage / 100.0 if percentage < 100 else None,
                percentage=percentage if percentage < 100 else None
            )
            
            closed_position = self.trading_client.close_position(symbol, close_request)
            
            # Update position records
            await self._update_position_after_close(symbol, percentage)
            
            return OrderResult(
                success=True,
                order_id=closed_position.id if hasattr(closed_position, 'id') else None,
                filled_quantity=Decimal(str(closed_position.qty if hasattr(closed_position, 'qty') else 0))
            )
            
        except Exception as e:
            logger.error(f"Error closing position for {symbol}: {e}")
            return OrderResult(
                success=False,
                error_message=str(e)
            )
    
    async def get_positions(self) -> List[PositionUpdate]:
        """Get current positions from Alpaca."""
        try:
            positions = self.trading_client.get_all_positions()
            
            position_updates = []
            for pos in positions:
                update = PositionUpdate(
                    symbol=pos.symbol,
                    side='LONG' if pos.side == PositionSide.LONG else 'SHORT',
                    quantity=Decimal(str(pos.qty)),
                    current_price=Decimal(str(pos.current_price)),
                    unrealized_pnl=Decimal(str(pos.unrealized_pl)),
                    market_value=Decimal(str(pos.market_value))
                )
                position_updates.append(update)
            
            # Update position cache
            await self._update_position_cache(position_updates)
            
            return position_updates
            
        except Exception as e:
            logger.error(f"Error retrieving positions: {e}")
            return []
    
    async def get_order_status(self, order_id: str) -> Dict[str, Any]:
        """Get order status and execution details."""
        try:
            order = self.trading_client.get_order_by_id(order_id)
            
            return {
                'id': order.id,
                'status': order.status.value,
                'filled_qty': float(order.filled_qty or 0),
                'filled_avg_price': float(order.filled_avg_price or 0),
                'created_at': order.created_at,
                'updated_at': order.updated_at,
                'commission': 0  # Alpaca has commission-free trading
            }
            
        except Exception as e:
            logger.error(f"Error getting order status for {order_id}: {e}")
            return {}
    
    async def get_account_info(self) -> Dict[str, Any]:
        """Get account information and buying power."""
        try:
            account = self.trading_client.get_account()
            
            return {
                'account_id': account.id,
                'buying_power': float(account.buying_power),
                'cash': float(account.cash),
                'portfolio_value': float(account.portfolio_value),
                'day_trade_count': getattr(account, 'day_trade_count', 0),
                'pattern_day_trader': getattr(account, 'pattern_day_trader', False),
                'trading_blocked': getattr(account, 'trading_blocked', False),
                'account_blocked': getattr(account, 'account_blocked', False)
            }
            
        except Exception as e:
            logger.error(f"Error getting account info: {e}")
            return {}
    
    async def _validate_order_risk(self, order_request: OrderRequest) -> Dict[str, Any]:
        """Validate order against risk management rules."""
        try:
            # Get current portfolio state
            positions = await self.get_positions()
            account_info = await self.get_account_info()
            
            # Calculate order value
            order_value = order_request.quantity * (order_request.price or Decimal('0'))
            
            # Check buying power
            if order_request.side == 'BUY':
                available_cash = Decimal(str(account_info.get('buying_power', 0)))
                if order_value > available_cash:
                    return {
                        'allowed': False,
                        'reason': f"Insufficient buying power: ${available_cash} < ${order_value}"
                    }
            
            # Check position size limits
            portfolio_value = Decimal(str(account_info.get('portfolio_value', 1)))
            allocation_percentage = order_value / portfolio_value
            
            if allocation_percentage > Decimal(str(settings.max_portfolio_allocation)):
                return {
                    'allowed': False,
                    'reason': f"Allocation too large: {allocation_percentage:.2%} > {settings.max_portfolio_allocation:.2%}"
                }
            
            # Check correlation limits
            correlation_check = await self.risk_manager.check_correlation_risk(
                order_request.symbol, float(allocation_percentage)
            )
            
            if not correlation_check['allowed']:
                return {
                    'allowed': False,
                    'reason': correlation_check['reason']
                }
            
            return {'allowed': True, 'reason': 'Risk checks passed'}
            
        except Exception as e:
            logger.error(f"Error in risk validation: {e}")
            return {
                'allowed': False,
                'reason': f"Risk validation error: {str(e)}"
            }
    
    def _convert_to_alpaca_order(self, order_request: OrderRequest):
        """Convert OrderRequest to Alpaca order format."""
        side = OrderSide.BUY if order_request.side == 'BUY' else OrderSide.SELL
        
        # Determine if this is a crypto trade based on symbol format
        # Alpaca crypto symbols use slash format (e.g., "BTC/USD")
        is_crypto = '/' in order_request.symbol and any(
            order_request.symbol.upper() in pair 
            for pair in settings.alpaca_crypto_pairs
        )
        
        # Crypto orders require GTC, stock orders can use DAY
        default_tif = TimeInForce.GTC if is_crypto else TimeInForce.DAY
        
        if order_request.order_type == 'MARKET':
            return MarketOrderRequest(
                symbol=order_request.symbol,
                qty=float(order_request.quantity),
                side=side,
                time_in_force=default_tif
            )
        elif order_request.order_type == 'LIMIT':
            return LimitOrderRequest(
                symbol=order_request.symbol,
                qty=float(order_request.quantity),
                side=side,
                limit_price=float(order_request.price),
                time_in_force=TimeInForce.GTC
            )
        elif order_request.order_type == 'STOP':
            return StopOrderRequest(
                symbol=order_request.symbol,
                qty=float(order_request.quantity),
                side=side,
                stop_price=float(order_request.stop_price),
                time_in_force=TimeInForce.GTC
            )
        else:
            raise ValueError(f"Unsupported order type: {order_request.order_type}")
    
    async def _create_trade_record(self, order_request: OrderRequest, submitted_order) -> Trade:
        """Create trade record in database."""
        try:
            with db_manager.get_session() as session:
                trade = Trade(
                    symbol=order_request.symbol,
                    exchange_order_id=submitted_order.id,
                    signal_id=order_request.signal_id,
                    side=order_request.side,
                    order_type=order_request.order_type,
                    status='PENDING',
                    quantity=order_request.quantity,
                    price=order_request.price,
                    filled_quantity=Decimal('0'),
                    created_at=datetime.utcnow()
                )
                session.add(trade)
                session.commit()
                session.refresh(trade)
                
                logger.info(f"Created trade record for order {submitted_order.id}")
                return trade
                
        except Exception as e:
            logger.error(f"Error creating trade record: {e}")
            raise
    
    async def _monitor_order_execution(self, order_id: str):
        """Monitor order execution and update database."""
        max_monitoring_time = 300  # 5 minutes
        check_interval = 5  # 5 seconds
        start_time = datetime.utcnow()
        
        while (datetime.utcnow() - start_time).seconds < max_monitoring_time:
            try:
                order_status = await self.get_order_status(order_id)
                
                if order_status.get('status') in ['FILLED', 'CANCELLED', 'REJECTED']:
                    # Update trade record
                    await self._update_trade_from_order_status(order_id, order_status)
                    
                    # Update positions if filled
                    if order_status.get('status') == 'FILLED':
                        await self._update_positions_after_fill(order_id, order_status)
                    
                    break
                
                await asyncio.sleep(check_interval)
                
            except Exception as e:
                logger.error(f"Error monitoring order {order_id}: {e}")
                break
    
    async def _update_trade_from_order_status(self, order_id: str, order_status: Dict[str, Any]):
        """Update trade record from order status."""
        try:
            with db_manager.get_session() as session:
                trade = session.query(Trade).filter(
                    Trade.exchange_order_id == order_id
                ).first()
                
                if trade:
                    trade.status = order_status['status']
                    trade.filled_quantity = Decimal(str(order_status.get('filled_qty', 0)))
                    trade.average_fill_price = Decimal(str(order_status.get('filled_avg_price', 0))) if order_status.get('filled_avg_price') else None
                    trade.executed_at = order_status.get('updated_at', datetime.utcnow())
                    trade.commission = Decimal(str(order_status.get('commission', 0)))
                    trade.updated_at = datetime.utcnow()
                    
                    session.commit()
                    logger.info(f"Updated trade record for order {order_id}")
                
        except Exception as e:
            logger.error(f"Error updating trade record: {e}")
    
    async def _update_trade_status(self, order_id: str, status: str):
        """Update trade status in database."""
        try:
            with db_manager.get_session() as session:
                trade = session.query(Trade).filter(
                    Trade.exchange_order_id == order_id
                ).first()
                
                if trade:
                    trade.status = status
                    trade.updated_at = datetime.utcnow()
                    session.commit()
                
        except Exception as e:
            logger.error(f"Error updating trade status: {e}")
    
    async def _load_positions(self):
        """Load existing positions from database."""
        try:
            with db_manager.get_session() as session:
                positions = session.query(Position).filter(
                    Position.is_open == True
                ).all()
                
                for pos in positions:
                    self.position_cache[pos.symbol] = pos
                
                logger.info(f"Loaded {len(positions)} active positions")
                
        except Exception as e:
            logger.error(f"Error loading positions: {e}")
    
    async def _update_position_cache(self, position_updates: List[PositionUpdate]):
        """Update local position cache."""
        for update in position_updates:
            self.position_cache[update.symbol] = update
        
        self.last_position_update = datetime.utcnow()
    
    async def _update_positions_after_fill(self, order_id: str, order_status: Dict[str, Any]):
        """Update position records after order fill."""
        # Implementation would update Position table based on filled order
        # This is a complex operation that needs to handle:
        # - New positions
        # - Adding to existing positions
        # - Reducing positions
        # - Closing positions
        pass
    
    async def _update_position_after_close(self, symbol: str, percentage: float):
        """Update position record after closing."""
        # Implementation would update Position table after closing
        pass
    
    async def cleanup(self):
        """Cleanup resources."""
        try:
            # Clear caches
            self.position_cache.clear()
            self.active_orders.clear()
            logger.info("AlpacaExecutor cleanup completed")
        except Exception as e:
            logger.error(f"Error during cleanup: {e}")


# Global executor instance
alpaca_executor = AlpacaExecutor()
