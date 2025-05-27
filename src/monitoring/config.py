"""
Monitoring system configuration and settings.
Centralizes all monitoring-related configuration parameters.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any
from datetime import time


@dataclass
class PerformanceMonitoringConfig:
    """Performance monitoring configuration."""
    
    # Monitoring intervals (seconds)
    update_interval: int = 300          # 5 minutes
    snapshot_interval: int = 60         # 1 minute
    health_check_interval: int = 600    # 10 minutes
    
    # Performance thresholds
    target_annual_return: float = 0.28   # 28%
    target_sharpe_ratio: float = 1.8
    target_win_rate: float = 0.65        # 65%
    max_drawdown_threshold: float = 0.15 # 15%
    
    # Risk thresholds
    volatility_warning: float = 0.30     # 30%
    volatility_critical: float = 0.50    # 50%
    var_confidence: float = 0.95         # 95% VaR
    
    # Trading metrics
    min_trades_for_metrics: int = 10
    lookback_days: int = 30
    
    # Performance history retention
    max_snapshots: int = 10000
    cleanup_days: int = 90


@dataclass
class NotificationConfig:
    """Notification system configuration."""
    
    # Telegram settings
    enabled: bool = True
    bot_token: Optional[str] = None
    chat_id: Optional[str] = None
    
    # Rate limiting
    rate_limit_seconds: int = 3
    burst_limit: int = 5
    
    # Notification thresholds
    min_trade_value: float = 100.0       # Minimum trade value to notify
    critical_drawdown: float = 0.20      # 20%
    warning_drawdown: float = 0.10       # 10%
    
    # Scheduled reports
    daily_report_time: time = time(9, 0)  # 09:00 UTC
    weekly_report_day: int = 0            # Monday
    send_daily_summary: bool = True
    send_weekly_report: bool = True
    
    # Message formatting
    use_html: bool = True
    include_charts: bool = False
    max_message_length: int = 4096


@dataclass
class DashboardConfig:
    """Dashboard and streaming configuration."""
    
    # WebSocket settings
    websocket_enabled: bool = True
    websocket_host: str = "0.0.0.0"
    websocket_port: int = 8765
    
    # Streaming intervals (seconds)
    metrics_stream_interval: int = 5
    portfolio_stream_interval: int = 10
    health_stream_interval: int = 30
    
    # Data retention
    realtime_buffer_size: int = 1000
    chart_data_hours: int = 24
    metrics_history_hours: int = 168     # 1 week
    
    # Dashboard features
    enable_real_time_charts: bool = True
    enable_trade_alerts: bool = True
    enable_system_alerts: bool = True
    
    # Performance optimization
    max_concurrent_clients: int = 100
    client_timeout_seconds: int = 60


@dataclass
class HealthCheckConfig:
    """Health check system configuration."""
    
    # Check intervals (seconds)
    database_check_interval: int = 300   # 5 minutes
    api_check_interval: int = 180        # 3 minutes
    system_check_interval: int = 600     # 10 minutes
    
    # Timeout settings
    database_timeout: float = 5.0        # seconds
    api_timeout: float = 10.0            # seconds
    ping_timeout: float = 3.0            # seconds
    
    # Thresholds
    database_latency_warning: float = 500.0    # ms
    database_latency_critical: float = 1000.0  # ms
    api_latency_warning: float = 2000.0        # ms
    api_latency_critical: float = 5000.0       # ms
    
    # System resource thresholds
    cpu_warning: float = 75.0            # %
    cpu_critical: float = 90.0           # %
    memory_warning: float = 80.0         # %
    memory_critical: float = 90.0        # %
    disk_warning: float = 85.0           # %
    disk_critical: float = 95.0          # %
    
    # Data freshness thresholds
    data_warning_minutes: int = 5
    data_critical_minutes: int = 10
    
    # Health check retention
    history_retention_hours: int = 72    # 3 days


@dataclass
class LoggingConfig:
    """Metrics logging configuration."""
    
    # File logging
    enable_file_logging: bool = True
    log_directory: str = "/tmp/crypto-pulse-metrics"
    log_rotation_hours: int = 24
    max_log_files: int = 30
    
    # Log levels
    performance_log_interval: int = 300  # 5 minutes
    trade_log_interval: int = 60         # 1 minute
    health_log_interval: int = 600       # 10 minutes
    
    # External logging
    enable_remote_logging: bool = False
    remote_endpoint: Optional[str] = None
    remote_api_key: Optional[str] = None
    
    # Log formats
    timestamp_format: str = "%Y-%m-%d %H:%M:%S.%f"
    include_metadata: bool = True


@dataclass
class AlertConfig:
    """Alert and threshold configuration."""
    
    # Adaptive thresholds
    enable_adaptive_thresholds: bool = True
    volatility_adjustment_factor: float = 1.2
    
    # Alert escalation
    escalation_threshold: int = 10       # alerts per hour
    cooldown_period: int = 3600          # 1 hour
    
    # Performance alerts
    performance_alert_types: List[str] = field(default_factory=lambda: [
        'drawdown', 'low_sharpe', 'high_volatility', 'low_win_rate'
    ])
    
    # Risk alerts
    risk_alert_types: List[str] = field(default_factory=lambda: [
        'position_concentration', 'correlation_risk', 'var_breach'
    ])
    
    # System alerts
    system_alert_types: List[str] = field(default_factory=lambda: [
        'database_down', 'api_failure', 'data_stale', 'high_resource_usage'
    ])
    
    # Alert priorities
    critical_alerts: List[str] = field(default_factory=lambda: [
        'database_down', 'api_failure', 'excessive_drawdown'
    ])
    
    # Suppression rules
    suppress_duplicates: bool = True
    duplicate_window_minutes: int = 15


@dataclass
class MonitoringConfig:
    """Complete monitoring system configuration."""
    
    # Component configurations
    performance: PerformanceMonitoringConfig = field(default_factory=PerformanceMonitoringConfig)
    notifications: NotificationConfig = field(default_factory=NotificationConfig)
    dashboard: DashboardConfig = field(default_factory=DashboardConfig)
    health_checks: HealthCheckConfig = field(default_factory=HealthCheckConfig)
    logging: LoggingConfig = field(default_factory=LoggingConfig)
    alerts: AlertConfig = field(default_factory=AlertConfig)
    
    # Global settings
    enabled: bool = True
    debug_mode: bool = False
    max_concurrent_tasks: int = 50
    
    # Environment-specific overrides
    environment_overrides: Dict[str, Any] = field(default_factory=dict)
    
    def apply_environment_overrides(self, environment: str):
        """Apply environment-specific configuration overrides."""
        if environment in self.environment_overrides:
            overrides = self.environment_overrides[environment]
            
            for section, settings in overrides.items():
                if hasattr(self, section):
                    section_config = getattr(self, section)
                    for key, value in settings.items():
                        if hasattr(section_config, key):
                            setattr(section_config, key, value)
    
    def validate_config(self) -> List[str]:
        """Validate configuration and return any errors."""
        errors = []
        
        # Validate notification settings
        if self.notifications.enabled and not self.notifications.bot_token:
            errors.append("Telegram bot token required when notifications are enabled")
        
        # Validate intervals
        if self.performance.update_interval < 30:
            errors.append("Performance update interval should be at least 30 seconds")
        
        if self.dashboard.metrics_stream_interval < 1:
            errors.append("Metrics stream interval should be at least 1 second")
        
        # Validate thresholds
        if self.performance.max_drawdown_threshold <= 0 or self.performance.max_drawdown_threshold >= 1:
            errors.append("Max drawdown threshold should be between 0 and 1")
        
        if self.performance.target_sharpe_ratio <= 0:
            errors.append("Target Sharpe ratio should be positive")
        
        # Validate resource thresholds
        if self.health_checks.cpu_warning >= self.health_checks.cpu_critical:
            errors.append("CPU warning threshold should be less than critical threshold")
        
        if self.health_checks.memory_warning >= self.health_checks.memory_critical:
            errors.append("Memory warning threshold should be less than critical threshold")
        
        return errors
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary."""
        return {
            'performance': self.performance.__dict__,
            'notifications': self.notifications.__dict__,
            'dashboard': self.dashboard.__dict__,
            'health_checks': self.health_checks.__dict__,
            'logging': self.logging.__dict__,
            'alerts': self.alerts.__dict__,
            'enabled': self.enabled,
            'debug_mode': self.debug_mode,
            'max_concurrent_tasks': self.max_concurrent_tasks
        }
    
    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> 'MonitoringConfig':
        """Create configuration from dictionary."""
        config = cls()
        
        for section_name, section_data in config_dict.items():
            if hasattr(config, section_name) and isinstance(section_data, dict):
                section_config = getattr(config, section_name)
                for key, value in section_data.items():
                    if hasattr(section_config, key):
                        setattr(section_config, key, value)
        
        return config


# Default configuration instance
default_monitoring_config = MonitoringConfig()

# Environment-specific configurations
ENVIRONMENT_CONFIGS = {
    'development': {
        'performance': {
            'update_interval': 60,  # More frequent updates in dev
            'snapshot_interval': 30
        },
        'notifications': {
            'enabled': False,  # Disable notifications in dev
            'send_daily_summary': False,
            'send_weekly_report': False
        },
        'dashboard': {
            'websocket_port': 8766,  # Different port for dev
            'max_concurrent_clients': 10
        },
        'logging': {
            'log_directory': './logs/dev-metrics'
        }
    },
    
    'testing': {
        'performance': {
            'update_interval': 30,
            'max_snapshots': 100  # Reduced for testing
        },
        'notifications': {
            'enabled': False
        },
        'health_checks': {
            'database_check_interval': 60,  # More frequent in testing
            'api_check_interval': 60
        },
        'logging': {
            'enable_file_logging': False  # Disable file logging in tests
        }
    },
    
    'production': {
        'performance': {
            'update_interval': 300,
            'cleanup_days': 365  # Keep more history in production
        },
        'notifications': {
            'enabled': True,
            'send_daily_summary': True,
            'send_weekly_report': True
        },
        'dashboard': {
            'websocket_host': "0.0.0.0",
            'max_concurrent_clients': 100
        },
        'health_checks': {
            'history_retention_hours': 168  # 1 week
        },
        'logging': {
            'enable_file_logging': True,
            'enable_remote_logging': True,
            'max_log_files': 90
        }
    }
}

# Apply environment-specific overrides to default config
default_monitoring_config.environment_overrides = ENVIRONMENT_CONFIGS
