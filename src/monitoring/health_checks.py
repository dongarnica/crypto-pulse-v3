"""
Comprehensive health check system for monitoring all trading system components.
Provides detailed diagnostics and automated health monitoring.
"""

import asyncio
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Callable
from dataclasses import dataclass
from enum import Enum
import time
import psutil
import httpx

from src.core.database import db_manager
from src.core.models import MarketData, Trade, Position
from src.execution.alpaca_executor import alpaca_executor
from src.data.binance_stream import data_streamer
from src.ml.ensemble import ml_ensemble
from config.settings import settings

logger = logging.getLogger(__name__)


class HealthStatus(Enum):
    """Health status levels."""
    HEALTHY = "healthy"
    WARNING = "warning"
    CRITICAL = "critical"
    UNKNOWN = "unknown"


@dataclass
class HealthCheckResult:
    """Result of a health check."""
    component: str
    status: HealthStatus
    message: str
    timestamp: datetime
    latency_ms: Optional[float] = None
    details: Optional[Dict[str, Any]] = None
    error: Optional[str] = None


@dataclass
class SystemResources:
    """System resource usage metrics."""
    cpu_percent: float
    memory_percent: float
    memory_total_gb: float
    memory_available_gb: float
    disk_usage_percent: float
    disk_free_gb: float
    network_sent_mb: float
    network_recv_mb: float
    load_average: List[float]


class HealthChecker:
    """
    Comprehensive health checking system for all components.
    """
    
    def __init__(self):
        self.health_checks: Dict[str, Callable] = {}
        self.last_results: Dict[str, HealthCheckResult] = {}
        self.monitoring_active = False
        
        # Register health checks
        self._register_health_checks()
    
    def _register_health_checks(self):
        """Register all health check functions."""
        self.health_checks = {
            'database': self._check_database_health,
            'alpaca_api': self._check_alpaca_api_health,
            'binance_api': self._check_binance_api_health,
            'data_stream': self._check_data_stream_health,
            'ml_models': self._check_ml_models_health,
            'system_resources': self._check_system_resources,
            'trading_engine': self._check_trading_engine_health,
            'risk_management': self._check_risk_management_health,
            'notifications': self._check_notifications_health
        }
    
    async def run_all_checks(self) -> Dict[str, HealthCheckResult]:
        """Run all registered health checks."""
        results = {}
        
        for component, check_func in self.health_checks.items():
            try:
                result = await check_func()
                results[component] = result
                self.last_results[component] = result
                
            except Exception as e:
                logger.error(f"Health check failed for {component}: {e}")
                results[component] = HealthCheckResult(
                    component=component,
                    status=HealthStatus.CRITICAL,
                    message=f"Health check failed: {str(e)}",
                    timestamp=datetime.utcnow(),
                    error=str(e)
                )
        
        return results
    
    async def run_single_check(self, component: str) -> Optional[HealthCheckResult]:
        """Run a single health check."""
        if component not in self.health_checks:
            return None
        
        try:
            result = await self.health_checks[component]()
            self.last_results[component] = result
            return result
            
        except Exception as e:
            logger.error(f"Health check failed for {component}: {e}")
            return HealthCheckResult(
                component=component,
                status=HealthStatus.CRITICAL,
                message=f"Health check failed: {str(e)}",
                timestamp=datetime.utcnow(),
                error=str(e)
            )
    
    async def get_overall_health(self) -> HealthCheckResult:
        """Get overall system health summary."""
        results = await self.run_all_checks()
        
        # Calculate overall status
        critical_count = sum(1 for r in results.values() if r.status == HealthStatus.CRITICAL)
        warning_count = sum(1 for r in results.values() if r.status == HealthStatus.WARNING)
        
        if critical_count > 0:
            status = HealthStatus.CRITICAL
            message = f"System critical: {critical_count} critical issues detected"
        elif warning_count > 0:
            status = HealthStatus.WARNING
            message = f"System warning: {warning_count} warnings detected"
        else:
            status = HealthStatus.HEALTHY
            message = "All systems operational"
        
        return HealthCheckResult(
            component="system_overall",
            status=status,
            message=message,
            timestamp=datetime.utcnow(),
            details={
                'total_checks': len(results),
                'healthy': sum(1 for r in results.values() if r.status == HealthStatus.HEALTHY),
                'warning': warning_count,
                'critical': critical_count,
                'components': {k: v.status.value for k, v in results.items()}
            }
        )
    
    async def start_monitoring(self, interval_seconds: int = 300):
        """Start continuous health monitoring."""
        if self.monitoring_active:
            logger.warning("Health monitoring already active")
            return
        
        self.monitoring_active = True
        logger.info(f"Starting health monitoring (interval: {interval_seconds}s)")
        
        try:
            while self.monitoring_active:
                await self.run_all_checks()
                
                # Check for critical issues
                critical_issues = [r for r in self.last_results.values() 
                                 if r.status == HealthStatus.CRITICAL]
                
                if critical_issues:
                    logger.error(f"Critical health issues detected: {len(critical_issues)}")
                    for issue in critical_issues:
                        logger.error(f"  - {issue.component}: {issue.message}")
                
                await asyncio.sleep(interval_seconds)
                
        except Exception as e:
            logger.error(f"Error in health monitoring loop: {e}")
        finally:
            self.monitoring_active = False
    
    async def stop_monitoring(self):
        """Stop health monitoring."""
        logger.info("Stopping health monitoring")
        self.monitoring_active = False
    
    async def _check_database_health(self) -> HealthCheckResult:
        """Check database connectivity and performance."""
        start_time = time.time()
        
        try:
            with db_manager.get_session() as session:
                # Test basic connectivity
                session.execute("SELECT 1")
                
                # Test table access
                market_data_count = session.query(MarketData).count()
                
                # Test write performance
                test_start = time.time()
                session.execute("SELECT pg_sleep(0.001)")  # 1ms test query
                write_latency = (time.time() - test_start) * 1000
                
            latency = (time.time() - start_time) * 1000
            
            # Determine status based on latency
            if latency > 1000:  # > 1 second
                status = HealthStatus.CRITICAL
                message = f"Database responding slowly ({latency:.1f}ms)"
            elif latency > 500:  # > 500ms
                status = HealthStatus.WARNING
                message = f"Database latency elevated ({latency:.1f}ms)"
            else:
                status = HealthStatus.HEALTHY
                message = f"Database operational ({latency:.1f}ms)"
            
            return HealthCheckResult(
                component="database",
                status=status,
                message=message,
                timestamp=datetime.utcnow(),
                latency_ms=latency,
                details={
                    'market_data_records': market_data_count,
                    'write_latency_ms': write_latency,
                    'connection_pool_size': db_manager.engine.pool.size() if hasattr(db_manager.engine, 'pool') else 'unknown'
                }
            )
            
        except Exception as e:
            return HealthCheckResult(
                component="database",
                status=HealthStatus.CRITICAL,
                message=f"Database connection failed: {str(e)}",
                timestamp=datetime.utcnow(),
                error=str(e)
            )
    
    async def _check_alpaca_api_health(self) -> HealthCheckResult:
        """Check Alpaca API connectivity and status."""
        start_time = time.time()
        
        try:
            # Test account access
            account_info = await alpaca_executor.get_account_info()
            
            # Test positions access
            positions = await alpaca_executor.get_positions()
            
            latency = (time.time() - start_time) * 1000
            
            if not account_info:
                return HealthCheckResult(
                    component="alpaca_api",
                    status=HealthStatus.CRITICAL,
                    message="Failed to retrieve account information",
                    timestamp=datetime.utcnow(),
                    latency_ms=latency
                )
            
            # Check account status
            account_status = account_info.get('status', 'unknown')
            if account_status != 'ACTIVE':
                status = HealthStatus.WARNING
                message = f"Account status: {account_status}"
            elif latency > 5000:  # > 5 seconds
                status = HealthStatus.WARNING
                message = f"API responding slowly ({latency:.1f}ms)"
            else:
                status = HealthStatus.HEALTHY
                message = f"API operational ({latency:.1f}ms)"
            
            return HealthCheckResult(
                component="alpaca_api",
                status=status,
                message=message,
                timestamp=datetime.utcnow(),
                latency_ms=latency,
                details={
                    'account_status': account_status,
                    'buying_power': account_info.get('buying_power', 0),
                    'portfolio_value': account_info.get('portfolio_value', 0),
                    'positions_count': len(positions) if positions else 0
                }
            )
            
        except Exception as e:
            return HealthCheckResult(
                component="alpaca_api",
                status=HealthStatus.CRITICAL,
                message=f"API connection failed: {str(e)}",
                timestamp=datetime.utcnow(),
                error=str(e)
            )
    
    async def _check_binance_api_health(self) -> HealthCheckResult:
        """Check Binance API connectivity and data stream."""
        start_time = time.time()
        
        try:
            # Test basic connectivity
            async with httpx.AsyncClient() as client:
                response = await client.get("https://api.binance.com/api/v3/ping", timeout=10)
                
            latency = (time.time() - start_time) * 1000
            
            if response.status_code != 200:
                return HealthCheckResult(
                    component="binance_api",
                    status=HealthStatus.CRITICAL,
                    message=f"API ping failed: HTTP {response.status_code}",
                    timestamp=datetime.utcnow(),
                    latency_ms=latency
                )
            
            # Check data freshness
            with db_manager.get_session() as session:
                latest_data = session.query(MarketData).order_by(
                    MarketData.timestamp.desc()
                ).first()
                
                if latest_data:
                    data_age = datetime.utcnow() - latest_data.timestamp
                    data_age_minutes = data_age.total_seconds() / 60
                else:
                    data_age_minutes = float('inf')
            
            if data_age_minutes > 10:  # No data in 10 minutes
                status = HealthStatus.CRITICAL
                message = f"Data stream stale ({data_age_minutes:.1f} minutes)"
            elif data_age_minutes > 5:  # No data in 5 minutes
                status = HealthStatus.WARNING
                message = f"Data stream delayed ({data_age_minutes:.1f} minutes)"
            elif latency > 2000:  # > 2 seconds
                status = HealthStatus.WARNING
                message = f"API responding slowly ({latency:.1f}ms)"
            else:
                status = HealthStatus.HEALTHY
                message = f"API operational ({latency:.1f}ms)"
            
            return HealthCheckResult(
                component="binance_api",
                status=status,
                message=message,
                timestamp=datetime.utcnow(),
                latency_ms=latency,
                details={
                    'data_age_minutes': data_age_minutes,
                    'latest_data_timestamp': latest_data.timestamp.isoformat() if latest_data else None
                }
            )
            
        except Exception as e:
            return HealthCheckResult(
                component="binance_api",
                status=HealthStatus.CRITICAL,
                message=f"API check failed: {str(e)}",
                timestamp=datetime.utcnow(),
                error=str(e)
            )
    
    async def _check_data_stream_health(self) -> HealthCheckResult:
        """Check data stream status and data quality."""
        try:
            # Check if stream is active
            stream_active = hasattr(data_streamer, 'is_running') and data_streamer.is_running
            
            # Check data quality and freshness
            with db_manager.get_session() as session:
                # Check recent data for each trading pair
                pair_status = {}
                for pair in settings.trading.trading_pairs:
                    latest = session.query(MarketData).filter(
                        MarketData.symbol == pair
                    ).order_by(MarketData.timestamp.desc()).first()
                    
                    if latest:
                        age_minutes = (datetime.utcnow() - latest.timestamp).total_seconds() / 60
                        pair_status[pair] = age_minutes
                    else:
                        pair_status[pair] = float('inf')
                
                # Calculate overall data health
                max_age = max(pair_status.values()) if pair_status else float('inf')
                avg_age = sum(pair_status.values()) / len(pair_status) if pair_status else float('inf')
            
            if not stream_active:
                status = HealthStatus.CRITICAL
                message = "Data stream not active"
            elif max_age > 10:
                status = HealthStatus.CRITICAL
                message = f"Data severely stale (max age: {max_age:.1f} min)"
            elif avg_age > 5:
                status = HealthStatus.WARNING
                message = f"Data moderately stale (avg age: {avg_age:.1f} min)"
            else:
                status = HealthStatus.HEALTHY
                message = f"Data stream healthy (avg age: {avg_age:.1f} min)"
            
            return HealthCheckResult(
                component="data_stream",
                status=status,
                message=message,
                timestamp=datetime.utcnow(),
                details={
                    'stream_active': stream_active,
                    'max_data_age_minutes': max_age,
                    'avg_data_age_minutes': avg_age,
                    'monitored_pairs': len(settings.trading.trading_pairs),
                    'pair_status': {k: f"{v:.1f}m" for k, v in pair_status.items()}
                }
            )
            
        except Exception as e:
            return HealthCheckResult(
                component="data_stream",
                status=HealthStatus.CRITICAL,
                message=f"Data stream check failed: {str(e)}",
                timestamp=datetime.utcnow(),
                error=str(e)
            )
    
    async def _check_ml_models_health(self) -> HealthCheckResult:
        """Check ML models status and performance."""
        try:
            # Check if models are loaded
            models_loaded = hasattr(ml_ensemble, 'random_forest') and ml_ensemble.random_forest.is_trained
            
            if not models_loaded:
                return HealthCheckResult(
                    component="ml_models",
                    status=HealthStatus.CRITICAL,
                    message="ML models not loaded",
                    timestamp=datetime.utcnow()
                )
            
            # Test prediction capability
            try:
                # Create dummy features for testing
                import numpy as np
                test_features = np.random.random((1, 50))  # Assuming 50 features
                
                start_time = time.time()
                prediction = await ml_ensemble.predict_async('BTCUSDT', test_features)
                prediction_latency = (time.time() - start_time) * 1000
                
                if prediction is None:
                    status = HealthStatus.CRITICAL
                    message = "Model prediction failed"
                elif prediction_latency > 1000:  # > 1 second
                    status = HealthStatus.WARNING
                    message = f"Model prediction slow ({prediction_latency:.1f}ms)"
                else:
                    status = HealthStatus.HEALTHY
                    message = f"Models operational ({prediction_latency:.1f}ms)"
                
            except Exception as pred_error:
                status = HealthStatus.WARNING
                message = f"Model prediction test failed: {str(pred_error)}"
                prediction_latency = None
            
            return HealthCheckResult(
                component="ml_models",
                status=status,
                message=message,
                timestamp=datetime.utcnow(),
                latency_ms=prediction_latency,
                details={
                    'models_loaded': models_loaded,
                    'prediction_latency_ms': prediction_latency
                }
            )
            
        except Exception as e:
            return HealthCheckResult(
                component="ml_models",
                status=HealthStatus.CRITICAL,
                message=f"ML models check failed: {str(e)}",
                timestamp=datetime.utcnow(),
                error=str(e)
            )
    
    async def _check_system_resources(self) -> HealthCheckResult:
        """Check system resource usage."""
        try:
            # Get CPU usage
            cpu_percent = psutil.cpu_percent(interval=1)
            
            # Get memory usage
            memory = psutil.virtual_memory()
            memory_percent = memory.percent
            memory_total_gb = memory.total / (1024**3)
            memory_available_gb = memory.available / (1024**3)
            
            # Get disk usage
            disk = psutil.disk_usage('/')
            disk_usage_percent = (disk.used / disk.total) * 100
            disk_free_gb = disk.free / (1024**3)
            
            # Get network stats
            network = psutil.net_io_counters()
            network_sent_mb = network.bytes_sent / (1024**2)
            network_recv_mb = network.bytes_recv / (1024**2)
            
            # Get load average (Unix only)
            try:
                load_average = list(psutil.getloadavg())
            except (AttributeError, OSError):
                load_average = [0, 0, 0]
            
            # Determine status
            if cpu_percent > 90 or memory_percent > 90 or disk_usage_percent > 95:
                status = HealthStatus.CRITICAL
                message = "System resources critically low"
            elif cpu_percent > 75 or memory_percent > 80 or disk_usage_percent > 85:
                status = HealthStatus.WARNING
                message = "System resources elevated"
            else:
                status = HealthStatus.HEALTHY
                message = "System resources normal"
            
            return HealthCheckResult(
                component="system_resources",
                status=status,
                message=message,
                timestamp=datetime.utcnow(),
                details={
                    'cpu_percent': cpu_percent,
                    'memory_percent': memory_percent,
                    'memory_total_gb': memory_total_gb,
                    'memory_available_gb': memory_available_gb,
                    'disk_usage_percent': disk_usage_percent,
                    'disk_free_gb': disk_free_gb,
                    'network_sent_mb': network_sent_mb,
                    'network_recv_mb': network_recv_mb,
                    'load_average': load_average
                }
            )
            
        except Exception as e:
            return HealthCheckResult(
                component="system_resources",
                status=HealthStatus.CRITICAL,
                message=f"System resources check failed: {str(e)}",
                timestamp=datetime.utcnow(),
                error=str(e)
            )
    
    async def _check_trading_engine_health(self) -> HealthCheckResult:
        """Check trading engine status and recent activity."""
        try:
            # Check recent trading activity
            with db_manager.get_session() as session:
                recent_trades = session.query(Trade).filter(
                    Trade.executed_at >= datetime.utcnow() - timedelta(hours=24)
                ).count()
                
                # Check for stuck orders
                pending_orders = session.query(Trade).filter(
                    Trade.status.in_(['PENDING', 'SUBMITTED']),
                    Trade.submitted_at <= datetime.utcnow() - timedelta(hours=1)
                ).count()
                
                # Check recent signals
                from src.core.models import TradingSignal
                recent_signals = session.query(TradingSignal).filter(
                    TradingSignal.timestamp >= datetime.utcnow() - timedelta(hours=1)
                ).count()
            
            # Determine status
            if pending_orders > 5:
                status = HealthStatus.CRITICAL
                message = f"Many stuck orders detected ({pending_orders})"
            elif recent_signals == 0:
                status = HealthStatus.WARNING
                message = "No recent trading signals generated"
            else:
                status = HealthStatus.HEALTHY
                message = f"Trading engine operational ({recent_trades} trades today)"
            
            return HealthCheckResult(
                component="trading_engine",
                status=status,
                message=message,
                timestamp=datetime.utcnow(),
                details={
                    'recent_trades_24h': recent_trades,
                    'pending_orders': pending_orders,
                    'recent_signals_1h': recent_signals
                }
            )
            
        except Exception as e:
            return HealthCheckResult(
                component="trading_engine",
                status=HealthStatus.CRITICAL,
                message=f"Trading engine check failed: {str(e)}",
                timestamp=datetime.utcnow(),
                error=str(e)
            )
    
    async def _check_risk_management_health(self) -> HealthCheckResult:
        """Check risk management system status."""
        try:
            from src.risk.manager import risk_manager
            
            # Check current portfolio risk
            account_info = await alpaca_executor.get_account_info()
            portfolio_value = account_info.get('portfolio_value', 0) if account_info else 0
            
            # Check position concentration
            positions = await alpaca_executor.get_positions()
            if positions and portfolio_value > 0:
                max_position_value = max(float(pos.market_value) for pos in positions if hasattr(pos, 'market_value'))
                max_concentration = (max_position_value / portfolio_value) * 100
            else:
                max_concentration = 0
            
            # Determine status
            if max_concentration > 50:
                status = HealthStatus.CRITICAL
                message = f"Excessive position concentration ({max_concentration:.1f}%)"
            elif max_concentration > 30:
                status = HealthStatus.WARNING
                message = f"High position concentration ({max_concentration:.1f}%)"
            else:
                status = HealthStatus.HEALTHY
                message = f"Risk management operational (max concentration: {max_concentration:.1f}%)"
            
            return HealthCheckResult(
                component="risk_management",
                status=status,
                message=message,
                timestamp=datetime.utcnow(),
                details={
                    'portfolio_value': portfolio_value,
                    'max_position_concentration': max_concentration,
                    'total_positions': len(positions) if positions else 0
                }
            )
            
        except Exception as e:
            return HealthCheckResult(
                component="risk_management",
                status=HealthStatus.CRITICAL,
                message=f"Risk management check failed: {str(e)}",
                timestamp=datetime.utcnow(),
                error=str(e)
            )
    
    async def _check_notifications_health(self) -> HealthCheckResult:
        """Check notification system health."""
        try:
            from src.monitoring.notifications import notification_manager
            
            # Test notification capability
            if hasattr(notification_manager.telegram, 'enabled') and notification_manager.telegram.enabled:
                status = HealthStatus.HEALTHY
                message = "Notification system operational"
            else:
                status = HealthStatus.WARNING
                message = "Telegram notifications disabled"
            
            # Check recent notification activity
            recent_notifications = notification_manager.get_notification_history(hours=24)
            
            return HealthCheckResult(
                component="notifications",
                status=status,
                message=message,
                timestamp=datetime.utcnow(),
                details={
                    'telegram_enabled': notification_manager.telegram.enabled,
                    'notifications_24h': len(recent_notifications)
                }
            )
            
        except Exception as e:
            return HealthCheckResult(
                component="notifications",
                status=HealthStatus.CRITICAL,
                message=f"Notifications check failed: {str(e)}",
                timestamp=datetime.utcnow(),
                error=str(e)
            )


# Global health checker instance
health_checker = HealthChecker()
