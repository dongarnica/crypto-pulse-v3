"""
Dashboard metrics aggregator for real-time monitoring interface.
Provides structured data for monitoring dashboards and real-time updates.
"""

import asyncio
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple
import json
from dataclasses import dataclass, asdict
from collections import defaultdict, deque
import pandas as pd
import numpy as np

from src.monitoring.performance_monitor import performance_monitor, PerformanceMetrics, SystemHealthMetrics
from src.monitoring.notifications import notification_manager
from src.core.database import db_manager
from src.core.models import Trade, Position, MarketData, TradingSignal
from config.settings import settings

logger = logging.getLogger(__name__)


@dataclass
class DashboardSnapshot:
    """Complete dashboard snapshot with all metrics."""
    timestamp: datetime
    
    # Portfolio summary
    portfolio: Dict[str, Any]
    
    # Performance metrics
    performance: Dict[str, Any]
    
    # Risk metrics
    risk: Dict[str, Any]
    
    # Trading activity
    trading: Dict[str, Any]
    
    # System health
    system: Dict[str, Any]
    
    # Recent activity
    activity: Dict[str, Any]


@dataclass
class RealTimeMetric:
    """Real-time metric point for streaming."""
    timestamp: datetime
    metric_name: str
    value: float
    symbol: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None


class MetricsBuffer:
    """Circular buffer for real-time metrics."""
    
    def __init__(self, max_size: int = 1000):
        self.buffer = deque(maxlen=max_size)
        self.max_size = max_size
    
    def add_metric(self, metric: RealTimeMetric):
        """Add a metric to the buffer."""
        self.buffer.append(metric)
    
    def get_metrics(self, minutes: int = 60) -> List[RealTimeMetric]:
        """Get metrics from the last N minutes."""
        cutoff = datetime.utcnow() - timedelta(minutes=minutes)
        return [m for m in self.buffer if m.timestamp > cutoff]
    
    def get_latest(self, metric_name: str) -> Optional[RealTimeMetric]:
        """Get the latest value for a specific metric."""
        for metric in reversed(self.buffer):
            if metric.metric_name == metric_name:
                return metric
        return None


class DashboardAggregator:
    """
    Aggregates all monitoring data for dashboard consumption.
    """
    
    def __init__(self):
        self.metrics_buffer = MetricsBuffer()
        self.last_snapshot: Optional[DashboardSnapshot] = None
        self.snapshot_interval = 30  # seconds
        self.is_running = False
        
        # Real-time metric streams
        self.metric_streams = {
            'portfolio_value': deque(maxlen=100),
            'pnl': deque(maxlen=100),
            'sharpe_ratio': deque(maxlen=100),
            'drawdown': deque(maxlen=100),
            'win_rate': deque(maxlen=100)
        }
    
    async def initialize(self):
        """Initialize dashboard aggregator."""
        try:
            logger.info("Initializing dashboard aggregator")
            
            # Generate initial snapshot
            await self.generate_snapshot()
            
            logger.info("Dashboard aggregator initialized")
            
        except Exception as e:
            logger.error(f"Failed to initialize dashboard aggregator: {e}")
            raise
    
    async def start_streaming(self):
        """Start real-time metrics streaming."""
        if self.is_running:
            logger.warning("Dashboard streaming already running")
            return
        
        try:
            self.is_running = True
            logger.info("Starting dashboard metrics streaming")
            
            # Start streaming tasks
            await asyncio.gather(
                self._snapshot_generation_loop(),
                self._real_time_metrics_loop()
            )
            
        except Exception as e:
            logger.error(f"Error in dashboard streaming: {e}")
        finally:
            self.is_running = False
    
    async def stop_streaming(self):
        """Stop metrics streaming."""
        logger.info("Stopping dashboard streaming")
        self.is_running = False
    
    async def generate_snapshot(self) -> DashboardSnapshot:
        """Generate a complete dashboard snapshot."""
        try:
            timestamp = datetime.utcnow()
            
            # Get current performance metrics
            performance_metrics = await performance_monitor.get_current_performance()
            system_health = await performance_monitor.get_system_health()
            
            # Aggregate portfolio data
            portfolio_data = await self._aggregate_portfolio_data()
            
            # Aggregate performance data
            performance_data = await self._aggregate_performance_data(performance_metrics)
            
            # Aggregate risk data
            risk_data = await self._aggregate_risk_data(performance_metrics)
            
            # Aggregate trading activity
            trading_data = await self._aggregate_trading_data()
            
            # Aggregate system health
            system_data = await self._aggregate_system_data(system_health)
            
            # Get recent activity
            activity_data = await self._aggregate_activity_data()
            
            snapshot = DashboardSnapshot(
                timestamp=timestamp,
                portfolio=portfolio_data,
                performance=performance_data,
                risk=risk_data,
                trading=trading_data,
                system=system_data,
                activity=activity_data
            )
            
            self.last_snapshot = snapshot
            return snapshot
            
        except Exception as e:
            logger.error(f"Error generating dashboard snapshot: {e}")
            return self._get_default_snapshot()
    
    async def get_real_time_metrics(self, minutes: int = 60) -> Dict[str, List[RealTimeMetric]]:
        """Get real-time metrics grouped by type."""
        try:
            metrics = self.metrics_buffer.get_metrics(minutes)
            
            grouped_metrics = defaultdict(list)
            for metric in metrics:
                grouped_metrics[metric.metric_name].append(metric)
            
            return dict(grouped_metrics)
            
        except Exception as e:
            logger.error(f"Error getting real-time metrics: {e}")
            return {}
    
    async def get_portfolio_chart_data(self, hours: int = 24) -> Dict[str, Any]:
        """Get portfolio value chart data."""
        try:
            with db_manager.get_session() as session:
                from src.core.models import PortfolioSnapshot
                
                start_time = datetime.utcnow() - timedelta(hours=hours)
                snapshots = session.query(PortfolioSnapshot).filter(
                    PortfolioSnapshot.timestamp >= start_time
                ).order_by(PortfolioSnapshot.timestamp).all()
                
                if not snapshots:
                    return {"timestamps": [], "values": [], "pnl": []}
                
                timestamps = [s.timestamp.isoformat() for s in snapshots]
                values = [float(s.total_value) for s in snapshots]
                pnl = [float(s.unrealized_pnl) for s in snapshots]
                
                return {
                    "timestamps": timestamps,
                    "values": values,
                    "pnl": pnl,
                    "start_value": values[0],
                    "current_value": values[-1],
                    "change": values[-1] - values[0],
                    "change_percent": ((values[-1] - values[0]) / values[0]) * 100 if values[0] > 0 else 0
                }
                
        except Exception as e:
            logger.error(f"Error getting portfolio chart data: {e}")
            return {"timestamps": [], "values": [], "pnl": []}
    
    async def get_performance_breakdown(self) -> Dict[str, Any]:
        """Get detailed performance breakdown by timeframes."""
        try:
            performance_data = {}
            
            # Different timeframes
            timeframes = {
                '1h': 1,
                '4h': 4,
                '24h': 24,
                '7d': 168,
                '30d': 720
            }
            
            for label, hours in timeframes.items():
                data = await self.get_portfolio_chart_data(hours)
                if data.get('timestamps'):
                    performance_data[label] = {
                        'return': data.get('change_percent', 0),
                        'pnl': data.get('change', 0),
                        'start_value': data.get('start_value', 0),
                        'end_value': data.get('current_value', 0)
                    }
                else:
                    performance_data[label] = {
                        'return': 0,
                        'pnl': 0,
                        'start_value': 0,
                        'end_value': 0
                    }
            
            return performance_data
            
        except Exception as e:
            logger.error(f"Error getting performance breakdown: {e}")
            return {}
    
    async def get_trading_heatmap(self, days: int = 30) -> Dict[str, Any]:
        """Get trading activity heatmap data."""
        try:
            with db_manager.get_session() as session:
                start_date = datetime.utcnow() - timedelta(days=days)
                
                trades = session.query(Trade).filter(
                    Trade.executed_at >= start_date,
                    Trade.status == 'FILLED'
                ).all()
                
                # Group by symbol and calculate metrics
                symbol_metrics = defaultdict(lambda: {
                    'trade_count': 0,
                    'total_volume': 0,
                    'total_pnl': 0,
                    'win_rate': 0
                })
                
                for trade in trades:
                    symbol = trade.symbol
                    symbol_metrics[symbol]['trade_count'] += 1
                    symbol_metrics[symbol]['total_volume'] += float(trade.quantity * trade.average_fill_price)
                    # Note: PnL calculation would need proper implementation
                
                # Convert to list format for heatmap
                heatmap_data = []
                for symbol, metrics in symbol_metrics.items():
                    heatmap_data.append({
                        'symbol': symbol,
                        'trade_count': metrics['trade_count'],
                        'total_volume': metrics['total_volume'],
                        'avg_trade_size': metrics['total_volume'] / metrics['trade_count'] if metrics['trade_count'] > 0 else 0
                    })
                
                # Sort by total volume
                heatmap_data.sort(key=lambda x: x['total_volume'], reverse=True)
                
                return {
                    'data': heatmap_data[:20],  # Top 20 symbols
                    'total_symbols': len(symbol_metrics),
                    'total_trades': len(trades),
                    'period_days': days
                }
                
        except Exception as e:
            logger.error(f"Error getting trading heatmap: {e}")
            return {'data': [], 'total_symbols': 0, 'total_trades': 0, 'period_days': days}
    
    async def _snapshot_generation_loop(self):
        """Loop for generating regular snapshots."""
        while self.is_running:
            try:
                await self.generate_snapshot()
                await asyncio.sleep(self.snapshot_interval)
                
            except Exception as e:
                logger.error(f"Error in snapshot generation loop: {e}")
                await asyncio.sleep(30)
    
    async def _real_time_metrics_loop(self):
        """Loop for collecting real-time metrics."""
        while self.is_running:
            try:
                # Get current performance metrics
                metrics = await performance_monitor.get_current_performance()
                timestamp = datetime.utcnow()
                
                # Add metrics to buffer
                real_time_metrics = [
                    RealTimeMetric(timestamp, 'portfolio_value', metrics.total_value),
                    RealTimeMetric(timestamp, 'total_pnl', metrics.total_pnl),
                    RealTimeMetric(timestamp, 'unrealized_pnl', metrics.unrealized_pnl),
                    RealTimeMetric(timestamp, 'realized_pnl', metrics.realized_pnl),
                    RealTimeMetric(timestamp, 'sharpe_ratio', metrics.sharpe_ratio),
                    RealTimeMetric(timestamp, 'max_drawdown', metrics.max_drawdown),
                    RealTimeMetric(timestamp, 'current_drawdown', metrics.current_drawdown),
                    RealTimeMetric(timestamp, 'win_rate', metrics.win_rate),
                    RealTimeMetric(timestamp, 'volatility', metrics.volatility)
                ]
                
                for metric in real_time_metrics:
                    self.metrics_buffer.add_metric(metric)
                
                # Update metric streams
                self.metric_streams['portfolio_value'].append((timestamp, metrics.total_value))
                self.metric_streams['pnl'].append((timestamp, metrics.total_pnl))
                self.metric_streams['sharpe_ratio'].append((timestamp, metrics.sharpe_ratio))
                self.metric_streams['drawdown'].append((timestamp, metrics.current_drawdown))
                self.metric_streams['win_rate'].append((timestamp, metrics.win_rate))
                
                await asyncio.sleep(10)  # 10 second intervals
                
            except Exception as e:
                logger.error(f"Error in real-time metrics loop: {e}")
                await asyncio.sleep(30)
    
    async def _aggregate_portfolio_data(self) -> Dict[str, Any]:
        """Aggregate portfolio summary data."""
        try:
            from src.execution.alpaca_executor import alpaca_executor
            
            # Get current positions
            positions = await alpaca_executor.get_positions()
            account_info = await alpaca_executor.get_account_info()
            
            total_value = account_info.get('portfolio_value', 0)
            cash_balance = account_info.get('cash', 0)
            
            position_data = []
            total_market_value = 0
            
            for position in positions:
                market_value = float(position.market_value) if hasattr(position, 'market_value') else 0
                unrealized_pnl = float(position.unrealized_pnl) if hasattr(position, 'unrealized_pnl') else 0
                
                position_data.append({
                    'symbol': position.symbol,
                    'quantity': float(position.qty),
                    'market_value': market_value,
                    'unrealized_pnl': unrealized_pnl,
                    'allocation': (market_value / total_value) * 100 if total_value > 0 else 0
                })
                
                total_market_value += market_value
            
            # Sort by allocation
            position_data.sort(key=lambda x: abs(x['allocation']), reverse=True)
            
            return {
                'total_value': total_value,
                'cash_balance': cash_balance,
                'total_positions': len(positions),
                'invested_value': total_market_value,
                'cash_allocation': (cash_balance / total_value) * 100 if total_value > 0 else 0,
                'positions': position_data[:10]  # Top 10 positions
            }
            
        except Exception as e:
            logger.error(f"Error aggregating portfolio data: {e}")
            return {
                'total_value': 0,
                'cash_balance': 0,
                'total_positions': 0,
                'invested_value': 0,
                'cash_allocation': 0,
                'positions': []
            }
    
    async def _aggregate_performance_data(self, metrics: PerformanceMetrics) -> Dict[str, Any]:
        """Aggregate performance metrics."""
        try:
            # Get performance breakdown
            breakdown = await self.get_performance_breakdown()
            
            return {
                'daily_return': metrics.daily_return,
                'cumulative_return': metrics.cumulative_return,
                'sharpe_ratio': metrics.sharpe_ratio,
                'sortino_ratio': metrics.sortino_ratio,
                'calmar_ratio': metrics.calmar_ratio,
                'total_pnl': metrics.total_pnl,
                'unrealized_pnl': metrics.unrealized_pnl,
                'realized_pnl': metrics.realized_pnl,
                'breakdown': breakdown,
                'target_progress': {
                    'return': metrics.target_return_progress,
                    'sharpe': metrics.target_sharpe_progress,
                    'winrate': metrics.target_winrate_progress
                }
            }
            
        except Exception as e:
            logger.error(f"Error aggregating performance data: {e}")
            return {}
    
    async def _aggregate_risk_data(self, metrics: PerformanceMetrics) -> Dict[str, Any]:
        """Aggregate risk metrics."""
        try:
            return {
                'max_drawdown': metrics.max_drawdown,
                'current_drawdown': metrics.current_drawdown,
                'volatility': metrics.volatility,
                'var_95': metrics.var_95,
                'risk_score': self._calculate_risk_score(metrics),
                'risk_level': self._get_risk_level(metrics)
            }
            
        except Exception as e:
            logger.error(f"Error aggregating risk data: {e}")
            return {}
    
    async def _aggregate_trading_data(self) -> Dict[str, Any]:
        """Aggregate trading activity data."""
        try:
            with db_manager.get_session() as session:
                # Get recent trades
                recent_trades = session.query(Trade).filter(
                    Trade.executed_at >= datetime.utcnow() - timedelta(hours=24)
                ).order_by(Trade.executed_at.desc()).limit(10).all()
                
                # Get trading stats
                all_trades = session.query(Trade).filter(
                    Trade.status == 'FILLED',
                    Trade.executed_at >= datetime.utcnow() - timedelta(days=30)
                ).all()
                
                trade_data = []
                for trade in recent_trades:
                    trade_data.append({
                        'symbol': trade.symbol,
                        'side': trade.side,
                        'quantity': float(trade.quantity),
                        'price': float(trade.average_fill_price) if trade.average_fill_price else 0,
                        'value': float(trade.quantity * (trade.average_fill_price or 0)),
                        'timestamp': trade.executed_at.isoformat() if trade.executed_at else None
                    })
                
                return {
                    'recent_trades': trade_data,
                    'total_trades_today': len([t for t in all_trades if t.executed_at >= datetime.utcnow() - timedelta(days=1)]),
                    'total_trades_week': len([t for t in all_trades if t.executed_at >= datetime.utcnow() - timedelta(days=7)]),
                    'total_trades_month': len(all_trades),
                    'avg_trades_per_day': len(all_trades) / 30 if all_trades else 0
                }
                
        except Exception as e:
            logger.error(f"Error aggregating trading data: {e}")
            return {
                'recent_trades': [],
                'total_trades_today': 0,
                'total_trades_week': 0,
                'total_trades_month': 0,
                'avg_trades_per_day': 0
            }
    
    async def _aggregate_system_data(self, health: SystemHealthMetrics) -> Dict[str, Any]:
        """Aggregate system health data."""
        try:
            return {
                'database': {
                    'status': health.db_connection_status,
                    'latency_ms': health.db_query_latency_ms,
                    'connections': health.db_active_connections
                },
                'apis': {
                    'alpaca': {
                        'status': health.alpaca_api_status,
                        'latency_ms': health.alpaca_api_latency_ms
                    },
                    'binance': {
                        'status': health.binance_api_status,
                        'latency_ms': health.binance_api_latency_ms
                    }
                },
                'ml_models': {
                    'last_update': health.model_last_update.isoformat(),
                    'accuracy': health.model_prediction_accuracy,
                    'confidence': health.ensemble_confidence_avg
                },
                'data_pipeline': {
                    'active': health.data_stream_active,
                    'last_update': health.last_data_update.isoformat(),
                    'missing_count': health.missing_data_count
                },
                'trading_engine': {
                    'uptime_hours': health.engine_uptime_hours,
                    'success_rate': health.cycle_success_rate,
                    'successful_cycles': health.successful_cycles,
                    'failed_cycles': health.failed_cycles
                },
                'overall_status': self._calculate_overall_health(health)
            }
            
        except Exception as e:
            logger.error(f"Error aggregating system data: {e}")
            return {}
    
    async def _aggregate_activity_data(self) -> Dict[str, Any]:
        """Aggregate recent activity data."""
        try:
            # Get recent notifications
            notifications = notification_manager.get_notification_history(hours=24)
            
            activity_items = []
            for notification in notifications[-10:]:  # Last 10 notifications
                activity_items.append({
                    'timestamp': notification.timestamp.isoformat(),
                    'type': notification.message_type.value,
                    'priority': notification.priority.value,
                    'title': notification.title,
                    'message': notification.message[:100] + '...' if len(notification.message) > 100 else notification.message
                })
            
            # Sort by timestamp (newest first)
            activity_items.sort(key=lambda x: x['timestamp'], reverse=True)
            
            return {
                'recent_activity': activity_items,
                'total_notifications_today': len([n for n in notifications if n.timestamp >= datetime.utcnow() - timedelta(days=1)]),
                'high_priority_alerts': len([n for n in notifications if n.priority.value in ['high', 'critical']])
            }
            
        except Exception as e:
            logger.error(f"Error aggregating activity data: {e}")
            return {
                'recent_activity': [],
                'total_notifications_today': 0,
                'high_priority_alerts': 0
            }
    
    def _calculate_risk_score(self, metrics: PerformanceMetrics) -> float:
        """Calculate overall risk score (0-100)."""
        try:
            # Weighted risk factors
            drawdown_factor = min(metrics.current_drawdown * 500, 50)  # 50 max for drawdown
            volatility_factor = min(metrics.volatility * 100, 30)      # 30 max for volatility
            var_factor = min(abs(metrics.var_95) * 200, 20)            # 20 max for VaR
            
            risk_score = drawdown_factor + volatility_factor + var_factor
            return min(risk_score, 100)
            
        except Exception:
            return 0
    
    def _get_risk_level(self, metrics: PerformanceMetrics) -> str:
        """Get risk level classification."""
        risk_score = self._calculate_risk_score(metrics)
        
        if risk_score >= 75:
            return "HIGH"
        elif risk_score >= 50:
            return "MEDIUM"
        elif risk_score >= 25:
            return "LOW"
        else:
            return "MINIMAL"
    
    def _calculate_overall_health(self, health: SystemHealthMetrics) -> str:
        """Calculate overall system health status."""
        try:
            if not health.db_connection_status:
                return "CRITICAL"
            
            if not health.alpaca_api_status or not health.binance_api_status:
                return "DEGRADED"
            
            if health.cycle_success_rate < 0.8:
                return "WARNING"
            
            if not health.data_stream_active:
                return "WARNING"
            
            return "HEALTHY"
            
        except Exception:
            return "UNKNOWN"
    
    def _get_default_snapshot(self) -> DashboardSnapshot:
        """Get default snapshot for error cases."""
        return DashboardSnapshot(
            timestamp=datetime.utcnow(),
            portfolio={},
            performance={},
            risk={},
            trading={},
            system={},
            activity={}
        )


# Global dashboard aggregator instance
dashboard_aggregator = DashboardAggregator()
