"""
Real-time metrics streaming and WebSocket server for live dashboard updates.
Provides WebSocket endpoints for streaming performance data to frontend dashboards.
"""

import asyncio
import json
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Set, Any, Optional
from dataclasses import asdict
import websockets
from websockets.server import serve
from websockets import WebSocketServerProtocol

from src.monitoring.dashboard import dashboard_aggregator, RealTimeMetric
from src.monitoring.performance_monitor import performance_monitor
from src.monitoring.health_checks import health_checker
from config.settings import settings

logger = logging.getLogger(__name__)


class MetricsWebSocketServer:
    """
    WebSocket server for streaming real-time metrics to dashboard clients.
    """
    
    def __init__(self, host: str = "0.0.0.0", port: int = 8765):
        self.host = host
        self.port = port
        self.server = None
        self.clients: Set[WebSocketServerProtocol] = set()
        self.is_running = False
        
        # Streaming intervals (seconds)
        self.metrics_interval = 5      # Performance metrics every 5 seconds
        self.portfolio_interval = 10   # Portfolio updates every 10 seconds
        self.health_interval = 30      # Health checks every 30 seconds
        
        # Message types
        self.MESSAGE_TYPES = {
            'PERFORMANCE_UPDATE': 'performance_update',
            'PORTFOLIO_UPDATE': 'portfolio_update',
            'HEALTH_UPDATE': 'health_update',
            'TRADE_ALERT': 'trade_alert',
            'SYSTEM_ALERT': 'system_alert',
            'SNAPSHOT': 'snapshot'
        }
    
    async def start_server(self):
        """Start the WebSocket server."""
        if self.is_running:
            logger.warning("WebSocket server already running")
            return
        
        try:
            logger.info(f"Starting WebSocket server on {self.host}:{self.port}")
            
            self.server = await serve(
                self.handle_client,
                self.host,
                self.port,
                ping_interval=30,
                ping_timeout=10
            )
            
            self.is_running = True
            
            # Start streaming tasks
            await asyncio.gather(
                self.stream_performance_metrics(),
                self.stream_portfolio_updates(),
                self.stream_health_updates()
            )
            
        except Exception as e:
            logger.error(f"Failed to start WebSocket server: {e}")
            raise
    
    async def stop_server(self):
        """Stop the WebSocket server."""
        if not self.is_running:
            return
        
        logger.info("Stopping WebSocket server")
        self.is_running = False
        
        # Close all client connections
        if self.clients:
            await asyncio.gather(
                *[client.close() for client in self.clients],
                return_exceptions=True
            )
        
        # Stop the server
        if self.server:
            self.server.close()
            await self.server.wait_closed()
    
    async def handle_client(self, websocket: WebSocketServerProtocol, path: str):
        """Handle new WebSocket client connection."""
        logger.info(f"New WebSocket client connected: {websocket.remote_address}")
        self.clients.add(websocket)
        
        try:
            # Send initial snapshot
            await self.send_initial_snapshot(websocket)
            
            # Handle client messages
            async for message in websocket:
                await self.handle_client_message(websocket, message)
                
        except websockets.exceptions.ConnectionClosed:
            logger.info(f"WebSocket client disconnected: {websocket.remote_address}")
        except Exception as e:
            logger.error(f"Error handling WebSocket client: {e}")
        finally:
            self.clients.discard(websocket)
    
    async def handle_client_message(self, websocket: WebSocketServerProtocol, message: str):
        """Handle incoming message from client."""
        try:
            data = json.loads(message)
            message_type = data.get('type')
            
            if message_type == 'REQUEST_SNAPSHOT':
                await self.send_initial_snapshot(websocket)
            
            elif message_type == 'SUBSCRIBE':
                # Handle subscription to specific metrics
                subscriptions = data.get('subscriptions', [])
                # Store client subscriptions (implementation would extend this)
                logger.debug(f"Client subscribed to: {subscriptions}")
            
            elif message_type == 'PING':
                await self.send_to_client(websocket, {'type': 'PONG', 'timestamp': datetime.utcnow().isoformat()})
            
        except json.JSONDecodeError:
            logger.warning(f"Invalid JSON from client: {websocket.remote_address}")
        except Exception as e:
            logger.error(f"Error handling client message: {e}")
    
    async def send_initial_snapshot(self, websocket: WebSocketServerProtocol):
        """Send initial dashboard snapshot to new client."""
        try:
            snapshot = await dashboard_aggregator.generate_snapshot()
            message = {
                'type': self.MESSAGE_TYPES['SNAPSHOT'],
                'timestamp': datetime.utcnow().isoformat(),
                'data': asdict(snapshot)
            }
            await self.send_to_client(websocket, message)
            
        except Exception as e:
            logger.error(f"Error sending initial snapshot: {e}")
    
    async def send_to_client(self, websocket: WebSocketServerProtocol, message: Dict[str, Any]):
        """Send message to a specific client."""
        try:
            await websocket.send(json.dumps(message))
        except websockets.exceptions.ConnectionClosed:
            self.clients.discard(websocket)
        except Exception as e:
            logger.error(f"Error sending message to client: {e}")
    
    async def broadcast_message(self, message: Dict[str, Any]):
        """Broadcast message to all connected clients."""
        if not self.clients:
            return
        
        # Send to all clients concurrently
        tasks = [self.send_to_client(client, message) for client in self.clients.copy()]
        await asyncio.gather(*tasks, return_exceptions=True)
    
    async def stream_performance_metrics(self):
        """Stream real-time performance metrics."""
        while self.is_running:
            try:
                # Get current performance metrics
                metrics = await performance_monitor.get_current_performance()
                
                message = {
                    'type': self.MESSAGE_TYPES['PERFORMANCE_UPDATE'],
                    'timestamp': datetime.utcnow().isoformat(),
                    'data': asdict(metrics)
                }
                
                await self.broadcast_message(message)
                await asyncio.sleep(self.metrics_interval)
                
            except Exception as e:
                logger.error(f"Error streaming performance metrics: {e}")
                await asyncio.sleep(10)
    
    async def stream_portfolio_updates(self):
        """Stream portfolio value and position updates."""
        while self.is_running:
            try:
                # Get portfolio chart data
                chart_data = await dashboard_aggregator.get_portfolio_chart_data(hours=24)
                
                message = {
                    'type': self.MESSAGE_TYPES['PORTFOLIO_UPDATE'],
                    'timestamp': datetime.utcnow().isoformat(),
                    'data': chart_data
                }
                
                await self.broadcast_message(message)
                await asyncio.sleep(self.portfolio_interval)
                
            except Exception as e:
                logger.error(f"Error streaming portfolio updates: {e}")
                await asyncio.sleep(15)
    
    async def stream_health_updates(self):
        """Stream system health updates."""
        while self.is_running:
            try:
                # Get overall health status
                health_result = await health_checker.get_overall_health()
                
                message = {
                    'type': self.MESSAGE_TYPES['HEALTH_UPDATE'],
                    'timestamp': datetime.utcnow().isoformat(),
                    'data': asdict(health_result)
                }
                
                await self.broadcast_message(message)
                await asyncio.sleep(self.health_interval)
                
            except Exception as e:
                logger.error(f"Error streaming health updates: {e}")
                await asyncio.sleep(20)
    
    async def send_trade_alert(self, trade_data: Dict[str, Any]):
        """Send real-time trade execution alert."""
        message = {
            'type': self.MESSAGE_TYPES['TRADE_ALERT'],
            'timestamp': datetime.utcnow().isoformat(),
            'data': trade_data
        }
        await self.broadcast_message(message)
    
    async def send_system_alert(self, alert_data: Dict[str, Any]):
        """Send system alert to all clients."""
        message = {
            'type': self.MESSAGE_TYPES['SYSTEM_ALERT'],
            'timestamp': datetime.utcnow().isoformat(),
            'data': alert_data
        }
        await self.broadcast_message(message)
    
    def get_client_count(self) -> int:
        """Get number of connected clients."""
        return len(self.clients)


class MetricsLogger:
    """
    Logger for metrics to files and external systems.
    """
    
    def __init__(self, log_directory: str = "/tmp/crypto-pulse-metrics"):
        self.log_directory = log_directory
        self.file_handles = {}
        self.is_logging = False
        
        # Create log directory
        import os
        os.makedirs(log_directory, exist_ok=True)
    
    async def start_logging(self):
        """Start metrics logging."""
        if self.is_logging:
            return
        
        self.is_logging = True
        logger.info(f"Starting metrics logging to {self.log_directory}")
        
        await asyncio.gather(
            self.log_performance_metrics(),
            self.log_trading_activity(),
            self.log_system_health()
        )
    
    async def stop_logging(self):
        """Stop metrics logging."""
        self.is_logging = False
        
        # Close all file handles
        for handle in self.file_handles.values():
            handle.close()
        self.file_handles.clear()
    
    async def log_performance_metrics(self):
        """Log performance metrics to file."""
        while self.is_logging:
            try:
                metrics = await performance_monitor.get_current_performance()
                
                log_entry = {
                    'timestamp': datetime.utcnow().isoformat(),
                    'total_value': metrics.total_value,
                    'total_pnl': metrics.total_pnl,
                    'sharpe_ratio': metrics.sharpe_ratio,
                    'max_drawdown': metrics.max_drawdown,
                    'win_rate': metrics.win_rate
                }
                
                await self.write_log_entry('performance', log_entry)
                await asyncio.sleep(300)  # Every 5 minutes
                
            except Exception as e:
                logger.error(f"Error logging performance metrics: {e}")
                await asyncio.sleep(60)
    
    async def log_trading_activity(self):
        """Log trading activity to file."""
        while self.is_logging:
            try:
                # Get recent trades
                trading_data = await dashboard_aggregator._aggregate_trading_data()
                
                if trading_data.get('recent_trades'):
                    for trade in trading_data['recent_trades']:
                        await self.write_log_entry('trades', trade)
                
                await asyncio.sleep(60)  # Every minute
                
            except Exception as e:
                logger.error(f"Error logging trading activity: {e}")
                await asyncio.sleep(120)
    
    async def log_system_health(self):
        """Log system health to file."""
        while self.is_logging:
            try:
                health_results = await health_checker.run_all_checks()
                
                log_entry = {
                    'timestamp': datetime.utcnow().isoformat(),
                    'overall_status': 'healthy',
                    'component_count': len(health_results),
                    'critical_issues': len([r for r in health_results.values() if r.status.value == 'critical'])
                }
                
                await self.write_log_entry('health', log_entry)
                await asyncio.sleep(600)  # Every 10 minutes
                
            except Exception as e:
                logger.error(f"Error logging system health: {e}")
                await asyncio.sleep(300)
    
    async def write_log_entry(self, log_type: str, entry: Dict[str, Any]):
        """Write log entry to file."""
        try:
            import os
            date_str = datetime.utcnow().strftime('%Y-%m-%d')
            filename = f"{log_type}_{date_str}.jsonl"
            filepath = os.path.join(self.log_directory, filename)
            
            # Open file if not already open
            if filepath not in self.file_handles:
                self.file_handles[filepath] = open(filepath, 'a')
            
            # Write entry
            self.file_handles[filepath].write(json.dumps(entry) + '\n')
            self.file_handles[filepath].flush()
            
        except Exception as e:
            logger.error(f"Error writing log entry: {e}")


class AlertThresholdManager:
    """
    Manages dynamic alert thresholds based on market conditions and performance.
    """
    
    def __init__(self):
        self.thresholds = {
            'drawdown_warning': 0.10,     # 10%
            'drawdown_critical': 0.20,    # 20%
            'volatility_warning': 0.30,   # 30%
            'volatility_critical': 0.50,  # 50%
            'sharpe_warning': 1.0,
            'sharpe_critical': 0.5,
            'win_rate_warning': 0.50,     # 50%
            'win_rate_critical': 0.30     # 30%
        }
        
        self.adaptive_thresholds = True
        self.market_volatility_factor = 1.0
    
    async def update_thresholds(self):
        """Update thresholds based on current market conditions."""
        if not self.adaptive_thresholds:
            return
        
        try:
            # Get current market volatility
            metrics = await performance_monitor.get_current_performance()
            current_volatility = metrics.volatility
            
            # Adjust thresholds based on market volatility
            if current_volatility > 0.40:  # High volatility market
                self.market_volatility_factor = 1.5
            elif current_volatility > 0.25:  # Medium volatility
                self.market_volatility_factor = 1.2
            else:  # Low volatility
                self.market_volatility_factor = 1.0
            
            # Update volatility thresholds
            base_vol_warning = 0.30
            base_vol_critical = 0.50
            
            self.thresholds['volatility_warning'] = base_vol_warning * self.market_volatility_factor
            self.thresholds['volatility_critical'] = base_vol_critical * self.market_volatility_factor
            
            logger.debug(f"Updated alert thresholds (market factor: {self.market_volatility_factor})")
            
        except Exception as e:
            logger.error(f"Error updating alert thresholds: {e}")
    
    def get_threshold(self, metric: str, level: str = 'warning') -> float:
        """Get threshold value for a metric."""
        key = f"{metric}_{level}"
        return self.thresholds.get(key, 0)
    
    def check_threshold_breach(self, metric: str, value: float) -> Optional[str]:
        """Check if a value breaches any threshold."""
        critical_threshold = self.get_threshold(metric, 'critical')
        warning_threshold = self.get_threshold(metric, 'warning')
        
        if metric in ['drawdown', 'volatility']:
            # Higher values are worse
            if value >= critical_threshold:
                return 'critical'
            elif value >= warning_threshold:
                return 'warning'
        else:
            # Lower values are worse (sharpe, win_rate)
            if value <= critical_threshold:
                return 'critical'
            elif value <= warning_threshold:
                return 'warning'
        
        return None


# Global instances
websocket_server = MetricsWebSocketServer()
metrics_logger = MetricsLogger()
threshold_manager = AlertThresholdManager()
