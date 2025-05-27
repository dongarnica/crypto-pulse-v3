"""
FastAPI application for Crypto Pulse V3 monitoring and control.
Provides RESTful API endpoints for system management and monitoring.
"""

from fastapi import FastAPI, HTTPException, Depends, BackgroundTasks, Query, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
from typing import Dict, List, Optional, Any, TYPE_CHECKING
from datetime import datetime, timedelta
import asyncio
import logging

from src.monitoring.performance_monitor import performance_monitor, PerformanceMetrics, SystemHealthMetrics
from src.monitoring.notifications import notification_manager, NotificationType, Priority
from config.settings import settings

# Type checking imports to avoid circular dependencies
if TYPE_CHECKING:
    from main import CryptoPulseSystem

logger = logging.getLogger(__name__)


# Pydantic models for API responses
class TradingSystemStatus(BaseModel):
    """Trading system status response."""
    is_running: bool
    uptime_minutes: int
    cycle_count: int
    successful_analyses: int
    errors_count: int
    last_analysis: Optional[datetime] = None
    next_analysis: Optional[datetime] = None

class PortfolioSummary(BaseModel):
    """Portfolio summary response."""
    total_value: float
    cash_balance: float
    positions_value: float
    daily_return: float
    total_return: float
    unrealized_pnl: float
    realized_pnl: float
    open_positions: int
    win_rate: float

class Position(BaseModel):
    """Position information."""
    symbol: str
    side: str
    quantity: float
    entry_price: float
    current_price: float
    unrealized_pnl: float
    unrealized_pnl_pct: float
    market_value: float

class Trade(BaseModel):
    """Trade information."""
    id: str
    symbol: str
    side: str
    quantity: float
    price: float
    timestamp: datetime
    status: str
    commission: Optional[float] = None

class RiskMetrics(BaseModel):
    """Risk metrics response."""
    portfolio_value: float
    total_exposure: float
    leverage: float
    max_position_size: float
    correlation_risk: float

class AlertRequest(BaseModel):
    """Alert/notification request."""
    message: str
    priority: str = "medium"
    component: str = "manual"


def create_app(system_instance: Optional['CryptoPulseSystem'] = None) -> FastAPI:
    """Create FastAPI application with system integration."""
    
    # FastAPI app instance
    app = FastAPI(
        title="Crypto Pulse V3 API",
        description="Advanced cryptocurrency trading system with ML ensemble and risk management",
        version="3.0.0",
        docs_url="/docs",
        redoc_url="/redoc"
    )

    # CORS middleware
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["http://localhost:3000", "http://localhost:8080"],  # Add your frontend URLs
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    
    # Store system reference for endpoints
    app.state.system = system_instance

    # Helper function to get system components
    def get_system_component(component_name: str):
        """Get a system component safely."""
        if not app.state.system:
            return None
        return app.state.system.components.get(component_name)

    # Health check endpoint
    @app.get("/health")
    async def health_check():
        """Get system health status."""
        try:
            system_status = "unknown"
            components_status = {}
            
            if app.state.system:
                system_status = "running" if app.state.system.running else "stopped"
                
                # Check individual components
                for name, component in app.state.system.components.items():
                    try:
                        if hasattr(component, 'is_running'):
                            components_status[name] = "running" if component.is_running else "stopped"
                        elif hasattr(component, 'is_connected'):
                            components_status[name] = "connected" if component.is_connected else "disconnected"
                        else:
                            components_status[name] = "active" if component else "inactive"
                    except Exception:
                        components_status[name] = "error"
            
            # Get performance monitor health
            try:
                health_metrics = await performance_monitor.get_system_health()
                monitoring_status = "healthy"
            except Exception as e:
                health_metrics = None
                monitoring_status = f"error: {str(e)}"
            
            overall_health = "healthy" if system_status == "running" and monitoring_status == "healthy" else "degraded"
            
            return {
                "status": overall_health,
                "system": system_status,
                "monitoring": monitoring_status,
                "components": components_status,
                "health_metrics": health_metrics,
                "timestamp": datetime.utcnow().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Health check failed: {e}")
            return JSONResponse(
                status_code=500,
                content={"status": "error", "message": str(e)}
            )

    # System control endpoints
    @app.post("/system/{action}")
    async def control_system(action: str):
        """Control system operations (start/stop/restart)."""
        if not app.state.system:
            raise HTTPException(status_code=503, detail="System not available")
        
        try:
            if action == "status":
                return app.state.system.get_system_status()
            elif action == "restart":
                # Note: Restart would need to be handled at a higher level
                return {"message": "Restart initiated", "action": action}
            else:
                raise HTTPException(status_code=400, detail=f"Unknown action: {action}")
                
        except Exception as e:
            logger.error(f"System control failed: {e}")
            raise HTTPException(status_code=500, detail=str(e))

    # Trading system status
    @app.get("/trading/status", response_model=TradingSystemStatus)
    async def get_trading_status():
        """Get trading engine status."""
        try:
            trading_engine = get_system_component('trading_engine')
            
            if not trading_engine:
                return TradingSystemStatus(
                    is_running=False,
                    uptime_minutes=0,
                    cycle_count=0,
                    successful_analyses=0,
                    errors_count=0
                )
            
            # Get status from trading engine
            if hasattr(trading_engine, 'get_status'):
                status = await trading_engine.get_status()
                return TradingSystemStatus(**status)
            else:
                return TradingSystemStatus(
                    is_running=getattr(trading_engine, 'is_running', False),
                    uptime_minutes=0,
                    cycle_count=0,
                    successful_analyses=0,
                    errors_count=0
                )
                
        except Exception as e:
            logger.error(f"Failed to get trading status: {e}")
            raise HTTPException(status_code=500, detail=str(e))

    # Portfolio endpoints
    @app.get("/portfolio/summary", response_model=PortfolioSummary)
    async def get_portfolio_summary():
        """Get portfolio summary."""
        try:
            # Get latest performance metrics
            metrics = await performance_monitor.get_latest_metrics()
            
            if not metrics:
                # Return default values if no metrics available
                return PortfolioSummary(
                    total_value=0.0,
                    cash_balance=0.0,
                    positions_value=0.0,
                    daily_return=0.0,
                    total_return=0.0,
                    unrealized_pnl=0.0,
                    realized_pnl=0.0,
                    open_positions=0,
                    win_rate=0.0
                )
            
            return PortfolioSummary(
                total_value=float(metrics.portfolio_value),
                cash_balance=float(metrics.cash_balance),
                positions_value=float(metrics.positions_value),
                daily_return=float(metrics.daily_return or 0),
                total_return=float(metrics.total_return or 0),
                unrealized_pnl=float(metrics.unrealized_pnl or 0),
                realized_pnl=float(metrics.realized_pnl or 0),
                open_positions=int(metrics.open_positions or 0),
                win_rate=float(metrics.win_rate or 0)
            )
            
        except Exception as e:
            logger.error(f"Failed to get portfolio summary: {e}")
            raise HTTPException(status_code=500, detail=str(e))

    @app.get("/portfolio/positions", response_model=List[Position])
    async def get_positions():
        """Get current positions."""
        try:
            # Try to get positions from trading engine or executor
            trading_engine = get_system_component('trading_engine')
            
            if trading_engine and hasattr(trading_engine, 'get_positions'):
                positions = await trading_engine.get_positions()
                return [
                    Position(
                        symbol=pos.symbol,
                        side=pos.side,
                        quantity=float(pos.quantity),
                        entry_price=float(pos.entry_price),
                        current_price=float(pos.current_price or 0),
                        unrealized_pnl=float(pos.unrealized_pnl or 0),
                        unrealized_pnl_pct=float(pos.unrealized_pnl_pct or 0),
                        market_value=float(pos.market_value or 0)
                    )
                    for pos in positions
                ]
            
            return []
            
        except Exception as e:
            logger.error(f"Failed to get positions: {e}")
            return []

    @app.get("/portfolio/trades", response_model=List[Trade])
    async def get_recent_trades(limit: int = Query(50, ge=1, le=500)):
        """Get recent trades."""
        try:
            trading_engine = get_system_component('trading_engine')
            
            if trading_engine and hasattr(trading_engine, 'get_recent_trades'):
                trades = await trading_engine.get_recent_trades(limit=limit)
                return [
                    Trade(
                        id=str(trade.id),
                        symbol=trade.symbol,
                        side=trade.side,
                        quantity=float(trade.quantity),
                        price=float(trade.price or 0),
                        timestamp=trade.timestamp,
                        status=trade.status,
                        commission=float(trade.commission) if trade.commission else None
                    )
                    for trade in trades
                ]
            
            return []
            
        except Exception as e:
            logger.error(f"Failed to get trades: {e}")
            return []

    # Risk management endpoints
    @app.get("/risk/metrics", response_model=RiskMetrics)
    async def get_risk_metrics():
        """Get current risk metrics."""
        try:
            metrics = await performance_monitor.get_latest_metrics()
            
            if not metrics:
                return RiskMetrics(
                    portfolio_value=0.0,
                    total_exposure=0.0,
                    leverage=0.0,
                    max_position_size=0.0,
                    correlation_risk=0.0,
                    var_95=0.0,
                    sharpe_ratio=0.0,
                    max_drawdown=0.0,
                    current_drawdown=0.0
                )
            
            return RiskMetrics(
                portfolio_value=float(metrics.portfolio_value),
                total_exposure=float(metrics.total_exposure or 0),
                leverage=float(metrics.leverage or 0),
                max_position_size=float(metrics.max_position_size or 0),
                correlation_risk=float(metrics.correlation_risk or 0),
                var_95=float(metrics.var_95 or 0),
                sharpe_ratio=float(metrics.sharpe_ratio or 0),
                max_drawdown=float(metrics.max_drawdown or 0),
                current_drawdown=float(metrics.current_drawdown or 0)
            )
            
        except Exception as e:
            logger.error(f"Failed to get risk metrics: {e}")
            raise HTTPException(status_code=500, detail=str(e))

    # Performance monitoring endpoints
    @app.get("/performance/metrics")
    async def get_performance_metrics(hours: int = Query(24, ge=1, le=168)):
        """Get performance metrics for the specified time period."""
        try:
            end_time = datetime.utcnow()
            start_time = end_time - timedelta(hours=hours)
            
            metrics = await performance_monitor.get_metrics_history(start_time, end_time)
            
            return {
                "period_hours": hours,
                "start_time": start_time.isoformat(),
                "end_time": end_time.isoformat(),
                "metrics": metrics
            }
            
        except Exception as e:
            logger.error(f"Failed to get performance metrics: {e}")
            raise HTTPException(status_code=500, detail=str(e))

    # Notification endpoints
    @app.post("/notifications/send")
    async def send_notification(alert: AlertRequest):
        """Send a manual notification."""
        try:
            priority_map = {
                "low": Priority.LOW,
                "medium": Priority.MEDIUM,
                "high": Priority.HIGH,
                "critical": Priority.CRITICAL
            }
            
            priority = priority_map.get(alert.priority.lower(), Priority.MEDIUM)
            
            await notification_manager.send_system_alert(
                component=alert.component,
                level=alert.priority.upper(),
                message=alert.message
            )
            
            return {"message": "Notification sent successfully"}
            
        except Exception as e:
            logger.error(f"Failed to send notification: {e}")
            raise HTTPException(status_code=500, detail=str(e))

    @app.get("/notifications/settings")
    async def get_notification_settings():
        """Get current notification settings."""
        try:
            return {
                "telegram_enabled": bool(settings.TELEGRAM_BOT_TOKEN),
                "rate_limit_seconds": 3,
                "priorities": ["low", "medium", "high", "critical"],
                "components": ["system", "trading", "risk", "data", "ml", "manual"]
            }
            
        except Exception as e:
            logger.error(f"Failed to get notification settings: {e}")
            raise HTTPException(status_code=500, detail=str(e))

    # System information endpoints
    @app.get("/system/info")
    async def get_system_info():
        """Get system information and configuration."""
        try:
            system_info = {
                "version": "3.0.0",
                "environment": settings.ENVIRONMENT,
                "start_time": app.state.system.start_time if app.state.system else None,
                "uptime_seconds": (datetime.utcnow().timestamp() - app.state.system.start_time) if app.state.system and app.state.system.start_time else 0,
                "configuration": {
                    "trading_pairs": settings.TRADING_PAIRS,
                    "api_host": settings.API_HOST,
                    "api_port": settings.API_PORT,
                    "database_configured": bool(settings.DB_NAME),
                    "telegram_configured": bool(settings.TELEGRAM_BOT_TOKEN),
                }
            }
            
            return system_info
            
        except Exception as e:
            logger.error(f"Failed to get system info: {e}")
            raise HTTPException(status_code=500, detail=str(e))

    # ML model endpoints
    @app.get("/ml/status")
    async def get_ml_status():
        """Get ML models status."""
        try:
            # This would be implemented when ML components are available
            return {
                "models": {
                    "random_forest": {"status": "loaded", "last_trained": None},
                    "lstm": {"status": "loaded", "last_trained": None},
                    "transformer": {"status": "loaded", "last_trained": None}
                },
                "ensemble": {"status": "active", "last_prediction": None}
            }
            
        except Exception as e:
            logger.error(f"Failed to get ML status: {e}")
            raise HTTPException(status_code=500, detail=str(e))

    @app.post("/ml/retrain")
    async def retrain_models(background_tasks: BackgroundTasks):
        """Trigger model retraining."""
        try:
            # This would trigger actual model retraining
            background_tasks.add_task(lambda: logger.info("Model retraining would be triggered here"))
            return {"message": "Model retraining initiated"}
            
        except Exception as e:
            logger.error(f"Failed to initiate model retraining: {e}")
            raise HTTPException(status_code=500, detail=str(e))

    # WebSocket endpoint for real-time updates
    @app.websocket("/ws/metrics")
    async def websocket_metrics(websocket):
        """WebSocket endpoint for real-time metrics streaming."""
        await websocket.accept()
        try:
            while True:
                # Get latest metrics
                metrics = await performance_monitor.get_latest_metrics()
                
                if metrics:
                    await websocket.send_json({
                        "type": "metrics",
                        "data": {
                            "portfolio_value": float(metrics.portfolio_value),
                            "daily_return": float(metrics.daily_return or 0),
                            "sharpe_ratio": float(metrics.sharpe_ratio or 0),
                            "current_drawdown": float(metrics.current_drawdown or 0),
                            "timestamp": datetime.utcnow().isoformat()
                        }
                    })
                
                await asyncio.sleep(5)  # Update every 5 seconds
                
        except Exception as e:
            logger.error(f"WebSocket error: {e}")
        finally:
            await websocket.close()

    return app


# Create a default app instance for development
app = create_app()


class PortfolioSummary(BaseModel):
    """Portfolio summary response."""
    total_value: float
    cash_balance: float
    total_positions: int
    unrealized_pnl: float
    realized_pnl: float
    daily_return: float
    total_return: float


class TradingPosition(BaseModel):
    """Trading position response."""
    symbol: str
    side: str
    quantity: float
    current_price: float
    unrealized_pnl: float
    market_value: float


class TradingSignalResponse(BaseModel):
    """Trading signal response."""
    symbol: str
    signal_type: str
    confidence_score: float
    timestamp: datetime
    ensemble_score: float
    recommended_allocation: float


class SystemCommand(BaseModel):
    """System command request."""
    action: str = Field(..., description="Action to perform: start, stop, restart, pause")
    parameters: Optional[Dict[str, Any]] = None


class NotificationRequest(BaseModel):
    """Notification request."""
    title: str
    message: str
    priority: str = "medium"
    notification_type: str = "system_alert"


class PerformanceReportRequest(BaseModel):
    """Performance report request."""
    period_days: int = Field(30, ge=1, le=365)
    include_trades: bool = True
    include_signals: bool = True


# Startup and shutdown events
@app.on_event("startup")
async def startup_event():
    """Initialize system components on startup."""
    try:
        logger.info("Starting Crypto Pulse V3 API")
        
        # Initialize components
        await trading_engine.initialize()
        await performance_monitor.initialize()
        await notification_manager.initialize()
        
        # Send startup notification
        await notification_manager.send_startup_notification()
        
        logger.info("Crypto Pulse V3 API started successfully")
        
    except Exception as e:
        logger.error(f"Failed to start API: {e}")
        raise


@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup on shutdown."""
    try:
        logger.info("Shutting down Crypto Pulse V3 API")
        
        # Stop trading engine
        await trading_engine.stop()
        
        # Stop monitoring
        await performance_monitor.stop_monitoring()
        
        # Send shutdown notification
        await notification_manager.send_shutdown_notification("API shutdown")
        
        logger.info("Crypto Pulse V3 API shutdown complete")
        
    except Exception as e:
        logger.error(f"Error during shutdown: {e}")


# System Status Endpoints
@app.get("/status", response_model=TradingSystemStatus)
async def get_system_status():
    """Get current trading system status."""
    try:
        return TradingSystemStatus(
            is_running=trading_engine.is_running,
            uptime_minutes=0,  # Would calculate actual uptime
            cycle_count=trading_engine.cycle_count,
            successful_analyses=trading_engine.successful_analyses,
            errors_count=trading_engine.errors_count,
            success_rate=trading_engine.successful_analyses / max(trading_engine.cycle_count, 1)
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    try:
        health = await performance_monitor.get_system_health()
        
        # Determine overall health
        overall_health = "healthy"
        if not health.db_connection_status:
            overall_health = "unhealthy"
        elif health.cycle_success_rate < 0.9:
            overall_health = "degraded"
        
        return {
            "status": overall_health,
            "timestamp": datetime.utcnow().isoformat(),
            "components": {
                "database": "healthy" if health.db_connection_status else "unhealthy",
                "alpaca_api": "healthy" if health.alpaca_api_status else "unhealthy",
                "binance_api": "healthy" if health.binance_api_status else "unhealthy",
                "data_stream": "healthy" if health.data_stream_active else "unhealthy",
                "trading_engine": "healthy" if health.cycle_success_rate > 0.9 else "degraded"
            }
        }
    except Exception as e:
        return JSONResponse(
            status_code=503,
            content={"status": "unhealthy", "error": str(e)}
        )


# System Control Endpoints
@app.post("/control")
async def system_control(command: SystemCommand, background_tasks: BackgroundTasks):
    """Control trading system operations."""
    try:
        action = command.action.lower()
        
        if action == "start":
            if trading_engine.is_running:
                raise HTTPException(status_code=400, detail="Trading engine is already running")
            
            background_tasks.add_task(trading_engine.start)
            background_tasks.add_task(performance_monitor.start_monitoring)
            
            await notification_manager.send_system_alert(
                "Trading Engine", "STARTED", "Trading engine started via API"
            )
            
            return {"message": "Trading engine start initiated"}
        
        elif action == "stop":
            if not trading_engine.is_running:
                raise HTTPException(status_code=400, detail="Trading engine is not running")
            
            await trading_engine.stop()
            await performance_monitor.stop_monitoring()
            
            await notification_manager.send_system_alert(
                "Trading Engine", "STOPPED", "Trading engine stopped via API"
            )
            
            return {"message": "Trading engine stopped"}
        
        elif action == "restart":
            await trading_engine.stop()
            await performance_monitor.stop_monitoring()
            
            background_tasks.add_task(trading_engine.start)
            background_tasks.add_task(performance_monitor.start_monitoring)
            
            await notification_manager.send_system_alert(
                "Trading Engine", "RESTARTED", "Trading engine restarted via API"
            )
            
            return {"message": "Trading engine restart initiated"}
        
        else:
            raise HTTPException(status_code=400, detail=f"Unknown action: {action}")
            
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# Portfolio Endpoints
@app.get("/portfolio", response_model=PortfolioSummary)
async def get_portfolio_summary():
    """Get current portfolio summary."""
    try:
        metrics = await performance_monitor.get_current_performance()
        
        return PortfolioSummary(
            total_value=metrics.total_value,
            cash_balance=metrics.cash_balance,
            total_positions=metrics.total_positions,
            unrealized_pnl=metrics.unrealized_pnl,
            realized_pnl=metrics.realized_pnl,
            daily_return=metrics.daily_return,
            total_return=metrics.cumulative_return
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/positions", response_model=List[TradingPosition])
async def get_positions():
    """Get current trading positions."""
    try:
        positions = await alpaca_executor.get_positions()
        
        return [
            TradingPosition(
                symbol=pos.symbol,
                side=pos.side,
                quantity=float(pos.quantity),
                current_price=float(pos.current_price),
                unrealized_pnl=float(pos.unrealized_pnl),
                market_value=float(pos.market_value)
            )
            for pos in positions
        ]
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/performance", response_model=PerformanceMetrics)
async def get_performance_metrics():
    """Get detailed performance metrics."""
    try:
        return await performance_monitor.get_current_performance()
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/performance/report")
async def get_performance_report(request: PerformanceReportRequest = Depends()):
    """Generate performance report for specified period."""
    try:
        report = await performance_monitor.generate_performance_report(request.period_days)
        return report
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# Trading Signal Endpoints
@app.get("/signals", response_model=List[TradingSignalResponse])
async def get_recent_signals(limit: int = Query(20, ge=1, le=100)):
    """Get recent trading signals."""
    try:
        # This would query the database for recent signals
        # For now, return empty list
        return []
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/signals/{symbol}")
async def get_symbol_signals(symbol: str, hours: int = Query(24, ge=1, le=168)):
    """Get trading signals for specific symbol."""
    try:
        # Implementation would query signals for specific symbol
        return {"symbol": symbol, "signals": []}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# Risk Management Endpoints
@app.get("/risk/metrics")
async def get_risk_metrics():
    """Get current risk metrics."""
    try:
        # Get portfolio risk assessment
        positions = await alpaca_executor.get_positions()
        account_info = await alpaca_executor.get_account_info()
        
        portfolio_value = account_info.get('portfolio_value', 0)
        
        # Calculate risk metrics
        total_exposure = sum(float(pos.market_value) for pos in positions)
        max_position_size = max([float(pos.market_value) for pos in positions]) if positions else 0
        
        return {
            "portfolio_value": portfolio_value,
            "total_exposure": total_exposure,
            "exposure_ratio": total_exposure / portfolio_value if portfolio_value > 0 else 0,
            "max_position_size": max_position_size,
            "max_position_ratio": max_position_size / portfolio_value if portfolio_value > 0 else 0,
            "position_count": len(positions),
            "max_allowed_allocation": settings.trading.max_portfolio_allocation
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/risk/correlation")
async def get_correlation_matrix():
    """Get current correlation matrix."""
    try:
        # This would return the correlation matrix from risk manager
        return {"message": "Correlation matrix endpoint - implementation pending"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# System Health Endpoints
@app.get("/system/health", response_model=SystemHealthMetrics)
async def get_detailed_health():
    """Get detailed system health metrics."""
    try:
        return await performance_monitor.get_system_health()
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/system/logs")
async def get_system_logs(lines: int = Query(100, ge=10, le=1000)):
    """Get recent system logs."""
    try:
        # This would return recent log entries
        # For now, return placeholder
        return {"logs": [], "message": "Log retrieval not implemented"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# Notification Endpoints
@app.post("/notifications/send")
async def send_notification(request: NotificationRequest):
    """Send custom notification."""
    try:
        # Parse priority and type
        priority_map = {
            "low": Priority.LOW,
            "medium": Priority.MEDIUM,
            "high": Priority.HIGH,
            "critical": Priority.CRITICAL
        }
        
        type_map = {
            "trade": NotificationType.TRADE_EXECUTED,
            "performance": NotificationType.PERFORMANCE_ALERT,
            "risk": NotificationType.RISK_ALERT,
            "system": NotificationType.SYSTEM_ALERT
        }
        
        priority = priority_map.get(request.priority.lower(), Priority.MEDIUM)
        notification_type = type_map.get(request.notification_type.lower(), NotificationType.SYSTEM_ALERT)
        
        # Send notification
        success = await notification_manager.send_notification(
            type_=notification_type,
            priority=priority,
            title=request.title,
            message=request.message,
            timestamp=datetime.utcnow()
        )
        
        return {"success": success, "message": "Notification sent" if success else "Failed to send notification"}
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/notifications/history")
async def get_notification_history(hours: int = Query(24, ge=1, le=168)):
    """Get notification history."""
    try:
        history = notification_manager.get_notification_history(hours)
        
        return {
            "count": len(history),
            "notifications": [
                {
                    "timestamp": n.timestamp.isoformat(),
                    "type": n.message_type.value,
                    "priority": n.priority.value,
                    "title": n.title,
                    "message": n.message
                }
                for n in history
            ]
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# Configuration Endpoints
@app.get("/config")
async def get_configuration():
    """Get current system configuration."""
    try:
        return {
            "trading": {
                "environment": settings.trading.environment,
                "max_portfolio_allocation": settings.trading.max_portfolio_allocation,
                "min_portfolio_allocation": settings.trading.min_portfolio_allocation,
                "max_drawdown_threshold": settings.trading.max_drawdown_threshold,
                "trading_pairs_count": len(settings.trading.trading_pairs)
            },
            "risk": {
                "atr_stop_multiplier": settings.risk.atr_stop_multiplier,
                "max_correlation_threshold": settings.risk.max_correlation_threshold,
                "volatility_scaling_factor": settings.risk.volatility_scaling_factor
            },
            "performance_targets": {
                "annual_return": settings.performance.target_annual_return,
                "sharpe_ratio": settings.performance.target_sharpe_ratio,
                "win_rate": settings.performance.target_win_rate,
                "profit_factor": settings.performance.target_profit_factor
            }
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# Market Data Endpoints
@app.get("/market/data/{symbol}")
async def get_market_data(symbol: str, timeframe: str = "1h", limit: int = Query(100, ge=1, le=1000)):
    """Get market data for symbol."""
    try:
        # This would retrieve market data from the database
        return {"symbol": symbol, "timeframe": timeframe, "data": []}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/market/sentiment")
async def get_market_sentiment():
    """Get current market sentiment data."""
    try:
        # This would return current sentiment metrics
        return {"sentiment": "neutral", "fear_greed_index": 50}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ML Model Endpoints
@app.get("/ml/status")
async def get_ml_model_status():
    """Get ML model status and performance."""
    try:
        # This would return ML model metrics
        return {
            "models": {
                "random_forest": {"loaded": True, "accuracy": 0.72},
                "lstm": {"loaded": True, "accuracy": 0.68},
                "transformer": {"loaded": True, "accuracy": 0.75}
            },
            "ensemble_performance": {"accuracy": 0.73, "confidence_avg": 0.71}
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/ml/retrain")
async def trigger_model_retraining(background_tasks: BackgroundTasks):
    """Trigger ML model retraining."""
    try:
        # This would trigger model retraining
        background_tasks.add_task(ml_ensemble.retrain_models)
        
        await notification_manager.send_system_alert(
            "ML Models", "RETRAINING", "Model retraining initiated via API"
        )
        
        return {"message": "Model retraining initiated"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "src.api.main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )
