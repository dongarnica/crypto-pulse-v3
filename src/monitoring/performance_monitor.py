"""
Performance monitoring system for tracking trading metrics and system health.
Provides real-time P&L tracking, risk metrics, and performance analytics.
"""

import asyncio
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any
import pandas as pd
import numpy as np
from dataclasses import dataclass, asdict
from sqlalchemy import func, and_, desc

from src.core.database import db_manager
from src.core.models import (
    Trade, Position, PortfolioSnapshot, TradingSignal, 
    MarketData, ModelPerformance
)
from src.execution.alpaca_executor import alpaca_executor
from config.settings import settings

logger = logging.getLogger(__name__)


@dataclass
class PerformanceMetrics:
    """Container for performance metrics."""
    timestamp: datetime
    
    # Portfolio metrics
    total_value: float
    cash_balance: float
    total_positions: int
    unrealized_pnl: float
    realized_pnl: float
    total_pnl: float
    
    # Performance ratios
    daily_return: float
    cumulative_return: float
    sharpe_ratio: float
    sortino_ratio: float
    calmar_ratio: float
    
    # Risk metrics
    volatility: float
    max_drawdown: float
    current_drawdown: float
    var_95: float
    
    # Trading metrics
    win_rate: float
    profit_factor: float
    average_win: float
    average_loss: float
    total_trades: int
    
    # Target achievement
    target_return_progress: float
    target_sharpe_progress: float
    target_winrate_progress: float


@dataclass
class SystemHealthMetrics:
    """Container for system health metrics."""
    timestamp: datetime
    
    # Database health
    db_connection_status: bool
    db_query_latency_ms: float
    db_active_connections: int
    
    # API health
    alpaca_api_status: bool
    alpaca_api_latency_ms: float
    binance_api_status: bool
    binance_api_latency_ms: float
    
    # ML model health
    model_last_update: datetime
    model_prediction_accuracy: float
    ensemble_confidence_avg: float
    
    # Data pipeline health
    data_stream_active: bool
    last_data_update: datetime
    missing_data_count: int
    
    # Trading engine health
    engine_uptime_hours: float
    successful_cycles: int
    failed_cycles: int
    cycle_success_rate: float


@dataclass
class AlertMetrics:
    """Container for alert conditions."""
    timestamp: datetime
    alert_type: str  # 'PERFORMANCE', 'RISK', 'SYSTEM', 'TRADE'
    severity: str    # 'LOW', 'MEDIUM', 'HIGH', 'CRITICAL'
    message: str
    current_value: float
    threshold_value: float
    symbol: Optional[str] = None


class PerformanceMonitor:
    """
    Comprehensive performance monitoring system.
    """
    
    def __init__(self):
        self.monitoring_active = False
        self.last_metrics_update = datetime.utcnow()
        self.performance_cache: Dict[str, Any] = {}
        self.alert_history: List[AlertMetrics] = []
        
        # Performance tracking
        self.initial_portfolio_value = 100000.0  # Default starting value
        self.high_water_mark = 100000.0
        self.daily_returns: List[float] = []
        self.portfolio_history: List[Tuple[datetime, float]] = []
        
    async def initialize(self):
        """Initialize performance monitoring system."""
        try:
            logger.info("Initializing performance monitor")
            
            # Load historical performance data
            await self._load_historical_data()
            
            # Initialize baseline metrics
            await self._initialize_baseline_metrics()
            
            logger.info("Performance monitor initialized")
            
        except Exception as e:
            logger.error(f"Failed to initialize performance monitor: {e}")
            raise
    
    async def start_monitoring(self):
        """Start continuous performance monitoring."""
        if self.monitoring_active:
            logger.warning("Performance monitoring already active")
            return
        
        try:
            self.monitoring_active = True
            logger.info("Starting performance monitoring")
            
            # Start monitoring tasks
            await asyncio.gather(
                self._performance_monitoring_loop(),
                self._system_health_monitoring_loop(),
                self._alert_monitoring_loop()
            )
            
        except Exception as e:
            logger.error(f"Error in performance monitoring: {e}")
        finally:
            self.monitoring_active = False
    
    async def stop_monitoring(self):
        """Stop performance monitoring."""
        logger.info("Stopping performance monitoring")
        self.monitoring_active = False
    
    async def get_current_performance(self) -> PerformanceMetrics:
        """Get current performance metrics."""
        try:
            # Get current portfolio state
            positions = await alpaca_executor.get_positions()
            account_info = await alpaca_executor.get_account_info()
            
            # Calculate basic metrics
            total_value = account_info.get('portfolio_value', 0)
            cash_balance = account_info.get('cash', 0)
            unrealized_pnl = sum(float(pos.unrealized_pnl) for pos in positions)
            
            # Get realized P&L from trades
            realized_pnl = await self._calculate_realized_pnl()
            total_pnl = unrealized_pnl + realized_pnl
            
            # Calculate returns
            daily_return = await self._calculate_daily_return(total_value)
            cumulative_return = (total_value - self.initial_portfolio_value) / self.initial_portfolio_value
            
            # Calculate risk metrics
            sharpe_ratio = await self._calculate_sharpe_ratio()
            sortino_ratio = await self._calculate_sortino_ratio()
            max_drawdown = await self._calculate_max_drawdown(total_value)
            current_drawdown = (self.high_water_mark - total_value) / self.high_water_mark
            volatility = await self._calculate_volatility()
            var_95 = await self._calculate_var(confidence=0.95)
            
            # Calculate trading metrics
            trading_metrics = await self._calculate_trading_metrics()
            
            # Calculate target progress
            target_progress = self._calculate_target_progress(
                cumulative_return, sharpe_ratio, trading_metrics['win_rate']
            )
            
            return PerformanceMetrics(
                timestamp=datetime.utcnow(),
                total_value=total_value,
                cash_balance=cash_balance,
                total_positions=len(positions),
                unrealized_pnl=unrealized_pnl,
                realized_pnl=realized_pnl,
                total_pnl=total_pnl,
                daily_return=daily_return,
                cumulative_return=cumulative_return,
                sharpe_ratio=sharpe_ratio,
                sortino_ratio=sortino_ratio,
                calmar_ratio=cumulative_return / max_drawdown if max_drawdown > 0 else 0,
                volatility=volatility,
                max_drawdown=max_drawdown,
                current_drawdown=current_drawdown,
                var_95=var_95,
                win_rate=trading_metrics['win_rate'],
                profit_factor=trading_metrics['profit_factor'],
                average_win=trading_metrics['average_win'],
                average_loss=trading_metrics['average_loss'],
                total_trades=trading_metrics['total_trades'],
                target_return_progress=target_progress['return'],
                target_sharpe_progress=target_progress['sharpe'],
                target_winrate_progress=target_progress['winrate']
            )
            
        except Exception as e:
            logger.error(f"Error calculating performance metrics: {e}")
            return self._get_default_performance_metrics()
    
    async def get_system_health(self) -> SystemHealthMetrics:
        """Get current system health metrics."""
        try:
            # Database health
            db_health = await self._check_database_health()
            
            # API health
            api_health = await self._check_api_health()
            
            # ML model health
            model_health = await self._check_model_health()
            
            # Data pipeline health
            data_health = await self._check_data_pipeline_health()
            
            # Trading engine health
            engine_health = await self._check_trading_engine_health()
            
            return SystemHealthMetrics(
                timestamp=datetime.utcnow(),
                db_connection_status=db_health['connected'],
                db_query_latency_ms=db_health['latency_ms'],
                db_active_connections=db_health['active_connections'],
                alpaca_api_status=api_health['alpaca']['status'],
                alpaca_api_latency_ms=api_health['alpaca']['latency_ms'],
                binance_api_status=api_health['binance']['status'],
                binance_api_latency_ms=api_health['binance']['latency_ms'],
                model_last_update=model_health['last_update'],
                model_prediction_accuracy=model_health['accuracy'],
                ensemble_confidence_avg=model_health['confidence_avg'],
                data_stream_active=data_health['stream_active'],
                last_data_update=data_health['last_update'],
                missing_data_count=data_health['missing_count'],
                engine_uptime_hours=engine_health['uptime_hours'],
                successful_cycles=engine_health['successful_cycles'],
                failed_cycles=engine_health['failed_cycles'],
                cycle_success_rate=engine_health['success_rate']
            )
            
        except Exception as e:
            logger.error(f"Error checking system health: {e}")
            return self._get_default_health_metrics()
    
    async def generate_performance_report(self, period_days: int = 30) -> Dict[str, Any]:
        """Generate comprehensive performance report."""
        try:
            end_date = datetime.utcnow()
            start_date = end_date - timedelta(days=period_days)
            
            # Get portfolio snapshots for the period
            with db_manager.get_session() as session:
                snapshots = session.query(PortfolioSnapshot).filter(
                    PortfolioSnapshot.timestamp >= start_date
                ).order_by(PortfolioSnapshot.timestamp).all()
            
            if not snapshots:
                return {"error": "No performance data available for the specified period"}
            
            # Convert to DataFrame for analysis
            df = pd.DataFrame([{
                'timestamp': s.timestamp,
                'total_value': float(s.total_value),
                'total_pnl': float(s.unrealized_pnl),
                'sharpe_ratio': s.sharpe_ratio,
                'max_drawdown': s.max_drawdown
            } for s in snapshots])
            
            df.set_index('timestamp', inplace=True)
            
            # Calculate performance statistics
            total_return = (df['total_value'].iloc[-1] - df['total_value'].iloc[0]) / df['total_value'].iloc[0]
            daily_returns = df['total_value'].pct_change().dropna()
            
            # Risk-adjusted metrics
            sharpe = daily_returns.mean() / daily_returns.std() * np.sqrt(252) if daily_returns.std() > 0 else 0
            max_dd = (df['total_value'] / df['total_value'].cummax() - 1).min()
            
            # Trading statistics
            trades_stats = await self._get_trades_statistics(start_date, end_date)
            
            return {
                'period': {
                    'start_date': start_date.isoformat(),
                    'end_date': end_date.isoformat(),
                    'days': period_days
                },
                'returns': {
                    'total_return': total_return,
                    'annualized_return': (1 + total_return) ** (365 / period_days) - 1,
                    'daily_avg_return': daily_returns.mean(),
                    'best_day': daily_returns.max(),
                    'worst_day': daily_returns.min()
                },
                'risk_metrics': {
                    'volatility': daily_returns.std() * np.sqrt(252),
                    'sharpe_ratio': sharpe,
                    'max_drawdown': max_dd,
                    'var_95': daily_returns.quantile(0.05),
                    'var_99': daily_returns.quantile(0.01)
                },
                'trading_stats': trades_stats,
                'target_progress': {
                    'annual_return_target': settings.performance.target_annual_return,
                    'sharpe_target': settings.performance.target_sharpe_ratio,
                    'current_annual_return': (1 + total_return) ** (365 / period_days) - 1,
                    'current_sharpe': sharpe
                }
            }
            
        except Exception as e:
            logger.error(f"Error generating performance report: {e}")
            return {"error": f"Failed to generate report: {str(e)}"}
    
    async def _performance_monitoring_loop(self):
        """Main performance monitoring loop."""
        while self.monitoring_active:
            try:
                # Calculate current metrics
                metrics = await self.get_current_performance()
                
                # Store metrics snapshot
                await self._store_performance_snapshot(metrics)
                
                # Update portfolio history
                self.portfolio_history.append((metrics.timestamp, metrics.total_value))
                
                # Update high water mark
                if metrics.total_value > self.high_water_mark:
                    self.high_water_mark = metrics.total_value
                
                # Check for alerts
                await self._check_performance_alerts(metrics)
                
                # Log key metrics
                logger.info(f"Portfolio: ${metrics.total_value:.2f}, "
                          f"P&L: ${metrics.total_pnl:.2f}, "
                          f"Sharpe: {metrics.sharpe_ratio:.2f}, "
                          f"DD: {metrics.current_drawdown:.2%}")
                
                # Wait before next update
                await asyncio.sleep(300)  # 5 minutes
                
            except Exception as e:
                logger.error(f"Error in performance monitoring loop: {e}")
                await asyncio.sleep(60)
    
    async def _system_health_monitoring_loop(self):
        """System health monitoring loop."""
        while self.monitoring_active:
            try:
                # Check system health
                health = await self.get_system_health()
                
                # Check for system alerts
                await self._check_system_alerts(health)
                
                # Log system status
                logger.debug(f"System Health - DB: {health.db_connection_status}, "
                           f"APIs: Alpaca={health.alpaca_api_status}/Binance={health.binance_api_status}, "
                           f"Engine Success Rate: {health.cycle_success_rate:.1%}")
                
                # Wait before next check
                await asyncio.sleep(180)  # 3 minutes
                
            except Exception as e:
                logger.error(f"Error in system health monitoring: {e}")
                await asyncio.sleep(60)
    
    async def _alert_monitoring_loop(self):
        """Alert monitoring and notification loop."""
        while self.monitoring_active:
            try:
                # Process pending alerts
                await self._process_alerts()
                
                # Clean up old alerts
                await self._cleanup_old_alerts()
                
                await asyncio.sleep(60)  # 1 minute
                
            except Exception as e:
                logger.error(f"Error in alert monitoring: {e}")
                await asyncio.sleep(30)
    
    async def _calculate_realized_pnl(self) -> float:
        """Calculate realized P&L from completed trades."""
        try:
            with db_manager.get_session() as session:
                # Get completed trades
                completed_trades = session.query(Trade).filter(
                    Trade.status == 'FILLED'
                ).all()
                
                # Calculate P&L for each trade pair (buy/sell)
                pnl = 0.0
                # Implementation would calculate actual P&L
                # This is a simplified version
                
                return pnl
                
        except Exception as e:
            logger.error(f"Error calculating realized P&L: {e}")
            return 0.0
    
    async def _calculate_daily_return(self, current_value: float) -> float:
        """Calculate daily return."""
        try:
            if len(self.portfolio_history) < 2:
                return 0.0
            
            yesterday_value = None
            yesterday = datetime.utcnow() - timedelta(days=1)
            
            # Find closest value to 24 hours ago
            for timestamp, value in reversed(self.portfolio_history):
                if timestamp <= yesterday:
                    yesterday_value = value
                    break
            
            if yesterday_value:
                return (current_value - yesterday_value) / yesterday_value
            
            return 0.0
            
        except Exception:
            return 0.0
    
    async def _calculate_sharpe_ratio(self) -> float:
        """Calculate Sharpe ratio."""
        try:
            if len(self.portfolio_history) < 30:  # Need at least 30 data points
                return 0.0
            
            # Calculate daily returns
            values = [v for _, v in self.portfolio_history[-30:]]
            returns = pd.Series(values).pct_change().dropna()
            
            if len(returns) == 0 or returns.std() == 0:
                return 0.0
            
            # Assume risk-free rate of 2% annually
            risk_free_rate = 0.02 / 252  # Daily risk-free rate
            excess_returns = returns - risk_free_rate
            
            return excess_returns.mean() / returns.std() * np.sqrt(252)
            
        except Exception:
            return 0.0
    
    async def _calculate_sortino_ratio(self) -> float:
        """Calculate Sortino ratio (downside deviation)."""
        try:
            if len(self.portfolio_history) < 30:
                return 0.0
            
            values = [v for _, v in self.portfolio_history[-30:]]
            returns = pd.Series(values).pct_change().dropna()
            
            if len(returns) == 0:
                return 0.0
            
            downside_returns = returns[returns < 0]
            if len(downside_returns) == 0:
                return float('inf')  # No downside
            
            downside_deviation = downside_returns.std()
            return returns.mean() / downside_deviation * np.sqrt(252)
            
        except Exception:
            return 0.0
    
    async def _calculate_max_drawdown(self, current_value: float) -> float:
        """Calculate maximum drawdown."""
        try:
            if len(self.portfolio_history) < 2:
                return 0.0
            
            values = [v for _, v in self.portfolio_history] + [current_value]
            peak = values[0]
            max_dd = 0.0
            
            for value in values:
                if value > peak:
                    peak = value
                drawdown = (peak - value) / peak
                max_dd = max(max_dd, drawdown)
            
            return max_dd
            
        except Exception:
            return 0.0
    
    async def _calculate_volatility(self) -> float:
        """Calculate portfolio volatility."""
        try:
            if len(self.portfolio_history) < 30:
                return 0.0
            
            values = [v for _, v in self.portfolio_history[-30:]]
            returns = pd.Series(values).pct_change().dropna()
            
            return returns.std() * np.sqrt(252)  # Annualized volatility
            
        except Exception:
            return 0.0
    
    async def _calculate_var(self, confidence: float = 0.95) -> float:
        """Calculate Value at Risk."""
        try:
            if len(self.portfolio_history) < 30:
                return 0.0
            
            values = [v for _, v in self.portfolio_history[-30:]]
            returns = pd.Series(values).pct_change().dropna()
            
            return returns.quantile(1 - confidence)
            
        except Exception:
            return 0.0
    
    async def _calculate_trading_metrics(self) -> Dict[str, float]:
        """Calculate trading performance metrics."""
        try:
            with db_manager.get_session() as session:
                # Get completed trades from last 30 days
                start_date = datetime.utcnow() - timedelta(days=30)
                trades = session.query(Trade).filter(
                    Trade.status == 'FILLED',
                    Trade.executed_at >= start_date
                ).all()
                
                if not trades:
                    return {
                        'win_rate': 0.0,
                        'profit_factor': 0.0,
                        'average_win': 0.0,
                        'average_loss': 0.0,
                        'total_trades': 0
                    }
                
                # Calculate trade P&L (simplified)
                winning_trades = []
                losing_trades = []
                
                # This would need proper P&L calculation based on entry/exit prices
                # For now, using placeholder logic
                
                win_rate = len(winning_trades) / len(trades) if trades else 0.0
                avg_win = np.mean(winning_trades) if winning_trades else 0.0
                avg_loss = abs(np.mean(losing_trades)) if losing_trades else 0.0
                profit_factor = (avg_win * len(winning_trades)) / (avg_loss * len(losing_trades)) if losing_trades else 0.0
                
                return {
                    'win_rate': win_rate,
                    'profit_factor': profit_factor,
                    'average_win': avg_win,
                    'average_loss': avg_loss,
                    'total_trades': len(trades)
                }
                
        except Exception as e:
            logger.error(f"Error calculating trading metrics: {e}")
            return {
                'win_rate': 0.0,
                'profit_factor': 0.0,
                'average_win': 0.0,
                'average_loss': 0.0,
                'total_trades': 0
            }
    
    def _calculate_target_progress(self, cumulative_return: float, sharpe_ratio: float, win_rate: float) -> Dict[str, float]:
        """Calculate progress towards performance targets."""
        try:
            return {
                'return': min(cumulative_return / settings.performance.target_annual_return, 1.0),
                'sharpe': min(sharpe_ratio / settings.performance.target_sharpe_ratio, 1.0),
                'winrate': min(win_rate / settings.performance.target_win_rate, 1.0)
            }
        except Exception:
            return {'return': 0.0, 'sharpe': 0.0, 'winrate': 0.0}
    
    async def _store_performance_snapshot(self, metrics: PerformanceMetrics):
        """Store performance snapshot to database."""
        try:
            with db_manager.get_session() as session:
                snapshot = PortfolioSnapshot(
                    timestamp=metrics.timestamp,
                    total_value=metrics.total_value,
                    cash_balance=metrics.cash_balance,
                    total_positions=metrics.total_positions,
                    unrealized_pnl=metrics.unrealized_pnl,
                    realized_pnl=metrics.realized_pnl,
                    sharpe_ratio=metrics.sharpe_ratio,
                    max_drawdown=metrics.max_drawdown,
                    win_rate=metrics.win_rate,
                    volatility=metrics.volatility,
                    var_95=metrics.var_95
                )
                session.add(snapshot)
                session.commit()
                
        except Exception as e:
            logger.error(f"Error storing performance snapshot: {e}")
    
    async def _check_performance_alerts(self, metrics: PerformanceMetrics):
        """Check for performance-related alerts."""
        alerts = []
        
        # Drawdown alerts
        if metrics.current_drawdown > 0.10:  # 10% drawdown
            severity = 'HIGH' if metrics.current_drawdown > 0.15 else 'MEDIUM'
            alerts.append(AlertMetrics(
                timestamp=datetime.utcnow(),
                alert_type='RISK',
                severity=severity,
                message=f"Portfolio drawdown exceeds threshold",
                current_value=metrics.current_drawdown,
                threshold_value=0.10
            ))
        
        # Low Sharpe ratio alert
        if metrics.sharpe_ratio < 1.0 and len(self.portfolio_history) > 30:
            alerts.append(AlertMetrics(
                timestamp=datetime.utcnow(),
                alert_type='PERFORMANCE',
                severity='MEDIUM',
                message=f"Sharpe ratio below target",
                current_value=metrics.sharpe_ratio,
                threshold_value=settings.performance.target_sharpe_ratio
            ))
        
        # Store alerts
        for alert in alerts:
            self.alert_history.append(alert)
    
    async def _check_database_health(self) -> Dict[str, Any]:
        """Check database connection health."""
        try:
            start_time = datetime.utcnow()
            
            with db_manager.get_session() as session:
                # Simple query to test connection
                session.execute("SELECT 1")
                
            latency = (datetime.utcnow() - start_time).total_seconds() * 1000
            
            return {
                'connected': True,
                'latency_ms': latency,
                'active_connections': 1  # Placeholder
            }
            
        except Exception as e:
            logger.error(f"Database health check failed: {e}")
            return {
                'connected': False,
                'latency_ms': 0,
                'active_connections': 0
            }
    
    async def _check_api_health(self) -> Dict[str, Any]:
        """Check API health status."""
        # Placeholder implementation
        return {
            'alpaca': {'status': True, 'latency_ms': 100},
            'binance': {'status': True, 'latency_ms': 150}
        }
    
    async def _check_model_health(self) -> Dict[str, Any]:
        """Check ML model health."""
        # Placeholder implementation
        return {
            'last_update': datetime.utcnow(),
            'accuracy': 0.7,
            'confidence_avg': 0.75
        }
    
    async def _check_data_pipeline_health(self) -> Dict[str, Any]:
        """Check data pipeline health."""
        # Placeholder implementation
        return {
            'stream_active': True,
            'last_update': datetime.utcnow(),
            'missing_count': 0
        }
    
    async def _check_trading_engine_health(self) -> Dict[str, Any]:
        """Check trading engine health."""
        # Placeholder implementation
        return {
            'uptime_hours': 24,
            'successful_cycles': 100,
            'failed_cycles': 5,
            'success_rate': 0.95
        }
    
    async def _check_system_alerts(self, health: SystemHealthMetrics):
        """Check for system health alerts."""
        alerts = []
        
        if not health.db_connection_status:
            alerts.append(AlertMetrics(
                timestamp=datetime.utcnow(),
                alert_type='SYSTEM',
                severity='CRITICAL',
                message="Database connection lost",
                current_value=0,
                threshold_value=1
            ))
        
        if health.cycle_success_rate < 0.9:
            alerts.append(AlertMetrics(
                timestamp=datetime.utcnow(),
                alert_type='SYSTEM',
                severity='HIGH',
                message="Trading engine success rate below threshold",
                current_value=health.cycle_success_rate,
                threshold_value=0.9
            ))
        
        for alert in alerts:
            self.alert_history.append(alert)
    
    async def _process_alerts(self):
        """Process and potentially send alerts."""
        # Implementation would send notifications
        # For now, just log critical alerts
        critical_alerts = [a for a in self.alert_history[-10:] if a.severity == 'CRITICAL']
        for alert in critical_alerts:
            logger.warning(f"CRITICAL ALERT: {alert.message}")
    
    async def _cleanup_old_alerts(self):
        """Clean up old alerts."""
        cutoff_time = datetime.utcnow() - timedelta(hours=24)
        self.alert_history = [a for a in self.alert_history if a.timestamp > cutoff_time]
    
    async def _load_historical_data(self):
        """Load historical performance data."""
        try:
            with db_manager.get_session() as session:
                snapshots = session.query(PortfolioSnapshot).order_by(
                    PortfolioSnapshot.timestamp
                ).limit(1000).all()
                
                for snapshot in snapshots:
                    self.portfolio_history.append((
                        snapshot.timestamp, 
                        float(snapshot.total_value)
                    ))
                
                if snapshots:
                    self.initial_portfolio_value = float(snapshots[0].total_value)
                    self.high_water_mark = max(float(s.total_value) for s in snapshots)
                
        except Exception as e:
            logger.error(f"Error loading historical data: {e}")
    
    async def _initialize_baseline_metrics(self):
        """Initialize baseline performance metrics."""
        try:
            account_info = await alpaca_executor.get_account_info()
            if account_info:
                self.initial_portfolio_value = account_info.get('portfolio_value', 100000)
                self.high_water_mark = self.initial_portfolio_value
                
        except Exception as e:
            logger.error(f"Error initializing baseline metrics: {e}")
    
    def _get_default_performance_metrics(self) -> PerformanceMetrics:
        """Get default performance metrics for error cases."""
        return PerformanceMetrics(
            timestamp=datetime.utcnow(),
            total_value=0.0, cash_balance=0.0, total_positions=0,
            unrealized_pnl=0.0, realized_pnl=0.0, total_pnl=0.0,
            daily_return=0.0, cumulative_return=0.0,
            sharpe_ratio=0.0, sortino_ratio=0.0, calmar_ratio=0.0,
            volatility=0.0, max_drawdown=0.0, current_drawdown=0.0, var_95=0.0,
            win_rate=0.0, profit_factor=0.0, average_win=0.0, average_loss=0.0, total_trades=0,
            target_return_progress=0.0, target_sharpe_progress=0.0, target_winrate_progress=0.0
        )
    
    def _get_default_health_metrics(self) -> SystemHealthMetrics:
        """Get default health metrics for error cases."""
        return SystemHealthMetrics(
            timestamp=datetime.utcnow(),
            db_connection_status=False, db_query_latency_ms=0, db_active_connections=0,
            alpaca_api_status=False, alpaca_api_latency_ms=0,
            binance_api_status=False, binance_api_latency_ms=0,
            model_last_update=datetime.utcnow(), model_prediction_accuracy=0.0, ensemble_confidence_avg=0.0,
            data_stream_active=False, last_data_update=datetime.utcnow(), missing_data_count=0,
            engine_uptime_hours=0.0, successful_cycles=0, failed_cycles=0, cycle_success_rate=0.0
        )
    
    async def _get_trades_statistics(self, start_date: datetime, end_date: datetime) -> Dict[str, Any]:
        """Get trading statistics for a period."""
        try:
            with db_manager.get_session() as session:
                trades = session.query(Trade).filter(
                    Trade.executed_at >= start_date,
                    Trade.executed_at <= end_date,
                    Trade.status == 'FILLED'
                ).all()
                
                return {
                    'total_trades': len(trades),
                    'buy_trades': len([t for t in trades if t.side == 'BUY']),
                    'sell_trades': len([t for t in trades if t.side == 'SELL']),
                    'avg_trade_size': np.mean([float(t.quantity * t.average_fill_price) for t in trades]) if trades else 0,
                    'largest_trade': max([float(t.quantity * t.average_fill_price) for t in trades]) if trades else 0
                }
                
        except Exception as e:
            logger.error(f"Error getting trades statistics: {e}")
            return {'total_trades': 0, 'buy_trades': 0, 'sell_trades': 0, 'avg_trade_size': 0, 'largest_trade': 0}


# Global performance monitor instance
performance_monitor = PerformanceMonitor()
