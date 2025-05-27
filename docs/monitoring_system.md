# Monitoring System Documentation

## Overview

The Crypto Pulse V3 monitoring system provides comprehensive real-time monitoring, alerting, and performance tracking for the cryptocurrency trading system. It consists of multiple integrated components that work together to ensure system health, track performance, and provide actionable insights.

## Architecture

### Core Components

1. **Performance Monitor** (`performance_monitor.py`)
   - Real-time P&L tracking
   - Risk metrics calculation (Sharpe ratio, drawdown, VaR)
   - Trading statistics (win rate, profit factor)
   - Target achievement tracking

2. **Health Checker** (`health_checks.py`)
   - Database connectivity monitoring
   - API health checks (Alpaca, Binance)
   - System resource monitoring
   - Data pipeline health
   - ML model status

3. **Notification Manager** (`notifications.py`)
   - Telegram bot integration
   - Priority-based alerting
   - Rate limiting and throttling
   - Scheduled reports (daily/weekly)

4. **Dashboard Aggregator** (`dashboard.py`)
   - Real-time metrics streaming
   - Portfolio analytics
   - Trading heatmaps
   - Performance breakdowns

5. **Monitoring Orchestrator** (`orchestrator.py`)
   - Central coordination
   - Task scheduling
   - Alert management
   - Report generation

6. **WebSocket Streaming** (`streaming.py`)
   - Real-time data streaming
   - Client connection management
   - Live dashboard updates

## Key Features

### Real-time Performance Tracking
- Portfolio value and P&L monitoring
- Risk-adjusted returns (Sharpe, Sortino, Calmar ratios)
- Drawdown tracking and alerts
- Volatility monitoring
- Value at Risk (VaR) calculations

### Comprehensive Health Monitoring
- Database performance and connectivity
- API latency and availability
- System resource usage (CPU, memory, disk)
- Data freshness validation
- ML model performance

### Intelligent Alerting
- Priority-based notification system
- Adaptive thresholds based on market conditions
- Rate limiting to prevent spam
- Escalation for critical issues
- Scheduled performance reports

### Dashboard Integration
- Real-time WebSocket streaming
- Interactive performance charts
- Trading activity heatmaps
- System health status
- Historical performance analysis

## Configuration

The monitoring system is highly configurable through the `config.py` file:

```python
from src.monitoring.config import MonitoringConfig

# Create custom configuration
config = MonitoringConfig()
config.performance.target_annual_return = 0.30  # 30%
config.notifications.enabled = True
config.dashboard.websocket_port = 8765
```

### Environment-Specific Configuration

Different environments (development, testing, production) can have customized settings:

```python
# Apply production configuration
config.apply_environment_overrides('production')
```

## Usage

### Basic Setup

```python
from src.monitoring import monitoring_orchestrator

# Initialize and start monitoring
await monitoring_orchestrator.initialize()
await monitoring_orchestrator.start()
```

### Individual Component Usage

```python
from src.monitoring import (
    performance_monitor,
    health_checker,
    notification_manager,
    dashboard_aggregator
)

# Get current performance metrics
metrics = await performance_monitor.get_current_performance()

# Run health checks
health_results = await health_checker.run_all_checks()

# Send custom notification
await notification_manager.send_trade_notification(
    symbol="BTCUSDT",
    side="BUY",
    quantity=0.1,
    price=45000,
    confidence=0.85
)

# Generate dashboard snapshot
snapshot = await dashboard_aggregator.generate_snapshot()
```

### WebSocket Streaming

```python
from src.monitoring.streaming import websocket_server

# Start WebSocket server for real-time updates
await websocket_server.start_server()
```

## Performance Metrics

### Core Metrics Tracked

1. **Portfolio Metrics**
   - Total portfolio value
   - Cash balance
   - Unrealized P&L
   - Realized P&L
   - Position count

2. **Performance Ratios**
   - Daily return
   - Cumulative return
   - Sharpe ratio
   - Sortino ratio
   - Calmar ratio

3. **Risk Metrics**
   - Volatility (annualized)
   - Maximum drawdown
   - Current drawdown
   - Value at Risk (95% confidence)

4. **Trading Statistics**
   - Win rate
   - Profit factor
   - Average win/loss
   - Total trades
   - Trading frequency

### Target Achievement Tracking

The system tracks progress towards performance targets:
- Annual return target: 28%
- Sharpe ratio target: 1.8
- Win rate target: 65%
- Maximum drawdown limit: 15%

## Health Checks

### Database Health
- Connection status
- Query latency
- Active connections
- Table accessibility

### API Health
- Alpaca API connectivity
- Binance API connectivity
- Response times
- Error rates

### System Resources
- CPU usage
- Memory usage
- Disk space
- Network activity
- Load average

### Data Pipeline
- Data freshness
- Stream status
- Missing data detection
- Quality validation

### ML Models
- Model loading status
- Prediction capability
- Performance metrics
- Update timestamps

## Alerting System

### Alert Types

1. **Performance Alerts**
   - High drawdown
   - Low Sharpe ratio
   - Excessive volatility
   - Poor win rate

2. **Risk Alerts**
   - Position concentration
   - Correlation limits
   - VaR breaches
   - Leverage warnings

3. **System Alerts**
   - Database failures
   - API outages
   - Resource exhaustion
   - Data pipeline issues

4. **Trade Notifications**
   - Order executions
   - Position changes
   - Significant trades

### Priority Levels

- **CRITICAL**: Immediate attention required
- **HIGH**: Important but not urgent
- **MEDIUM**: Normal operational alerts
- **LOW**: Informational notifications

### Rate Limiting

The system implements intelligent rate limiting:
- Maximum 1 message per 3 seconds
- Burst protection
- Priority-based queuing
- Duplicate suppression

## Dashboard Features

### Real-time Charts
- Portfolio value over time
- P&L tracking
- Drawdown visualization
- Performance metrics

### Trading Analytics
- Activity heatmaps
- Symbol performance
- Trade distribution
- Volume analysis

### System Status
- Component health
- Resource usage
- API status
- Data quality

## WebSocket API

### Message Types

1. **PERFORMANCE_UPDATE**: Real-time performance metrics
2. **PORTFOLIO_UPDATE**: Portfolio value and positions
3. **HEALTH_UPDATE**: System health status
4. **TRADE_ALERT**: Trade execution notifications
5. **SYSTEM_ALERT**: System status alerts
6. **SNAPSHOT**: Complete dashboard snapshot

### Client Connection

```javascript
const ws = new WebSocket('ws://localhost:8765');

ws.onmessage = function(event) {
    const data = JSON.parse(event.data);
    
    switch(data.type) {
        case 'PERFORMANCE_UPDATE':
            updatePerformanceDisplay(data.data);
            break;
        case 'TRADE_ALERT':
            showTradeNotification(data.data);
            break;
    }
};

// Request initial snapshot
ws.send(JSON.stringify({type: 'REQUEST_SNAPSHOT'}));
```

## Telegram Integration

### Bot Setup

1. Create a Telegram bot through @BotFather
2. Get bot token and chat ID
3. Configure in settings:

```python
TELEGRAM_BOT_TOKEN = "your_bot_token"
TELEGRAM_CHAT_ID = "your_chat_id"
```

### Message Formatting

Messages include:
- Priority indicators (emojis)
- Timestamps
- Detailed metrics
- Formatted numbers (currency, percentages)

### Scheduled Reports

- **Daily Summary**: Portfolio performance, trades, top performers
- **Weekly Report**: Comprehensive analysis, risk metrics, recommendations

## Logging and Persistence

### Database Storage
- Performance snapshots stored in `portfolio_snapshots` table
- Trading activity in `trades` table
- System health metrics in dedicated tables

### File Logging
- JSON Lines format for easy parsing
- Daily rotation
- Separate files for different metric types
- Configurable retention periods

### External Integration
- Support for remote logging endpoints
- Metrics export to external systems
- API endpoints for data retrieval

## Monitoring Best Practices

### Performance Optimization
- Efficient database queries
- Connection pooling
- Async processing
- Resource monitoring

### Reliability
- Error handling and recovery
- Graceful degradation
- Redundancy for critical components
- Health check validation

### Security
- Secure WebSocket connections
- API key protection
- Access control
- Data encryption

## Troubleshooting

### Common Issues

1. **Database Connection Failures**
   - Check connection string
   - Verify database accessibility
   - Check network connectivity

2. **API Timeouts**
   - Increase timeout settings
   - Check API status pages
   - Verify credentials

3. **High Resource Usage**
   - Monitor system resources
   - Optimize query performance
   - Scale infrastructure

4. **Missing Notifications**
   - Verify Telegram configuration
   - Check rate limiting
   - Review notification settings

### Debug Mode

Enable debug mode for detailed logging:

```python
config.debug_mode = True
```

### Health Check Validation

Run manual health checks:

```python
results = await health_checker.run_all_checks()
for component, result in results.items():
    print(f"{component}: {result.status.value} - {result.message}")
```

## API Reference

### Performance Monitor

```python
# Get current metrics
metrics = await performance_monitor.get_current_performance()

# Generate performance report
report = await performance_monitor.generate_performance_report(days=30)

# Start/stop monitoring
await performance_monitor.start_monitoring()
await performance_monitor.stop_monitoring()
```

### Health Checker

```python
# Run all health checks
results = await health_checker.run_all_checks()

# Run specific check
result = await health_checker.run_single_check('database')

# Get overall health
overall = await health_checker.get_overall_health()
```

### Notification Manager

```python
# Send trade notification
await notification_manager.send_trade_notification(
    symbol="BTCUSDT", side="BUY", quantity=0.1, 
    price=45000, confidence=0.85
)

# Send performance alert
await notification_manager.send_performance_alert(
    "High Drawdown", 0.15, 0.10, "Drawdown exceeded threshold"
)

# Send system alert
await notification_manager.send_system_alert(
    "database", "CRITICAL", "Connection lost"
)
```

### Dashboard Aggregator

```python
# Generate snapshot
snapshot = await dashboard_aggregator.generate_snapshot()

# Get portfolio chart data
chart_data = await dashboard_aggregator.get_portfolio_chart_data(hours=24)

# Get trading heatmap
heatmap = await dashboard_aggregator.get_trading_heatmap(days=30)
```

## Extension Points

### Custom Metrics

Add custom metrics by extending the PerformanceMetrics dataclass:

```python
@dataclass
class CustomMetrics(PerformanceMetrics):
    custom_ratio: float = 0.0
    custom_indicator: float = 0.0
```

### Custom Health Checks

Add custom health checks:

```python
health_checker.health_checks['custom_check'] = custom_check_function
```

### Custom Notifications

Extend notification types:

```python
class CustomNotificationType(Enum):
    CUSTOM_ALERT = "custom_alert"
```

### Custom Dashboard Data

Add custom dashboard sections:

```python
async def custom_dashboard_section():
    return {"custom_data": "value"}

dashboard_aggregator.custom_sections['custom'] = custom_dashboard_section
```

## Performance Tuning

### Database Optimization
- Use database indexes
- Optimize query patterns
- Implement connection pooling
- Regular maintenance

### Memory Management
- Monitor memory usage
- Implement cleanup routines
- Use efficient data structures
- Limit history retention

### Network Optimization
- Compress WebSocket messages
- Batch database operations
- Implement caching
- Use connection pooling

### Concurrency
- Use async/await patterns
- Limit concurrent operations
- Implement proper locking
- Monitor task queues

This monitoring system provides a robust foundation for tracking and managing the performance of your cryptocurrency trading system. It can be easily extended and customized to meet specific requirements while maintaining high performance and reliability.
