"""
Comprehensive monitoring package for the Crypto Pulse V3 trading system.

This package provides:
- Real-time performance monitoring and metrics tracking
- System health checks and diagnostics  
- Telegram notifications and alerting
- Dashboard data aggregation and real-time streaming
- Monitoring orchestration and coordination

Key Components:
- PerformanceMonitor: Real-time P&L, risk metrics, trading statistics
- HealthChecker: Component health checks and system diagnostics
- NotificationManager: Telegram alerts and scheduled reports
- DashboardAggregator: Real-time metrics and dashboard data
- MonitoringOrchestrator: Central coordination of all monitoring

Usage:
    from src.monitoring import monitoring_orchestrator
    
    # Initialize and start monitoring
    await monitoring_orchestrator.initialize()
    await monitoring_orchestrator.start()
"""

from .performance_monitor import (
    performance_monitor,
    PerformanceMetrics,
    SystemHealthMetrics,
    AlertMetrics
)

from .notifications import (
    notification_manager,
    NotificationManager,
    TelegramNotifier,
    Notification,
    NotificationType,
    Priority
)

from .dashboard import (
    dashboard_aggregator,
    DashboardAggregator,
    DashboardSnapshot,
    RealTimeMetric,
    MetricsBuffer
)

from .health_checks import (
    health_checker,
    HealthChecker,
    HealthCheckResult,
    HealthStatus,
    SystemResources
)

# TODO: Fix orchestrator.py corruption
# from .orchestrator import (
#     monitoring_orchestrator,
#     MonitoringOrchestrator
# )

# Export main monitoring interface
__all__ = [
    # Performance monitoring
    'performance_monitor',
    'PerformanceMetrics',
    'SystemHealthMetrics',
    'AlertMetrics',
    
    # Notifications
    'notification_manager',
    'NotificationManager',
    'TelegramNotifier',
    'Notification',
    'NotificationType',
    'Priority',
    
    # Dashboard and real-time data
    'dashboard_aggregator',
    'DashboardAggregator',
    'DashboardSnapshot',
    'RealTimeMetric',
    'MetricsBuffer',
    
    # Health checks
    'health_checker',
    'HealthChecker',
    'HealthCheckResult',
    'HealthStatus',
    'SystemResources'
]

# Package version
__version__ = '3.0.0'

# Package information
__author__ = 'Crypto Pulse Development Team'
__description__ = 'Comprehensive monitoring system for cryptocurrency trading'
