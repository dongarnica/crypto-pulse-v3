#!/usr/bin/env python3
"""
Main entry point for Crypto Pulse V3 trading system.
Coordinates all system components and provides unified lifecycle management.
"""

import asyncio
import signal
import sys
import logging
from pathlib import Path
from typing import Optional, List
from contextlib import AsyncExitStack
import uvloop

# Add the project root to the Python path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from config.settings import settings
from src.core.trading_engine import TradingEngine
from src.core.database import get_async_database, init_database
from src.data.binance_stream import BinanceDataStream
from src.monitoring.orchestrator import MonitoringOrchestrator
from src.monitoring.performance_monitor import PerformanceMonitor
from src.monitoring.notifications import NotificationManager
from src.api.main import create_app
import uvicorn

logger = logging.getLogger(__name__)


class CryptoPulseSystem:
    """
    Main system orchestrator for Crypto Pulse V3.
    Manages the lifecycle of all system components.
    """
    
    def __init__(self):
        self.components = {}
        self.running = False
        self.shutdown_event = asyncio.Event()
        self.exit_stack = AsyncExitStack()
        
        # Performance tracking
        self.start_time = None
        self.startup_duration = None
        
    async def initialize_database(self):
        """Initialize database connections and verify setup."""
        logger.info("Initializing database connections...")
        
        try:
            # Initialize async database
            await init_database()
            
            # Verify database health
            async_engine = get_async_database()
            async with async_engine.begin() as conn:
                from sqlalchemy import text
                result = await conn.execute(text("SELECT 1"))
                if result.scalar() == 1:
                    logger.info("✅ Database connection verified")
                else:
                    raise Exception("Database health check failed")
                    
        except Exception as e:
            logger.error(f"❌ Database initialization failed: {e}")
            raise
    
    async def start_data_streams(self):
        """Start all data ingestion streams."""
        logger.info("Starting data streams...")
        
        try:
            # Initialize Binance data stream
            binance_stream = BinanceDataStream()
            self.components['binance_stream'] = binance_stream
            
            # Start the data stream
            await binance_stream.start()
            logger.info("✅ Binance data stream started")
            
            # Add to exit stack for cleanup
            self.exit_stack.push_async_callback(binance_stream.stop)
            
        except Exception as e:
            logger.error(f"❌ Data stream initialization failed: {e}")
            raise
    
    async def start_monitoring(self):
        """Start monitoring and alerting systems."""
        logger.info("Starting monitoring systems...")
        
        try:
            # Initialize monitoring orchestrator
            monitoring = MonitoringOrchestrator()
            self.components['monitoring'] = monitoring
            
            # Start monitoring services
            await monitoring.start()
            logger.info("✅ Monitoring systems started")
            
            # Add to exit stack for cleanup
            self.exit_stack.push_async_callback(monitoring.stop)
            
        except Exception as e:
            logger.error(f"❌ Monitoring initialization failed: {e}")
            raise
    
    async def start_trading_engine(self):
        """Start the core trading engine."""
        logger.info("Starting trading engine...")
        
        try:
            # Initialize trading engine
            trading_engine = TradingEngine()
            self.components['trading_engine'] = trading_engine
            
            # Start the trading engine
            await trading_engine.start()
            logger.info("✅ Trading engine started")
            
            # Add to exit stack for cleanup
            self.exit_stack.push_async_callback(trading_engine.stop)
            
        except Exception as e:
            logger.error(f"❌ Trading engine initialization failed: {e}")
            raise
    
    async def start_api_server(self):
        """Start the FastAPI server for system control."""
        logger.info("Starting API server...")
        
        try:
            # Create FastAPI app with system reference
            app = create_app(self)
            
            # Configure Uvicorn
            config = uvicorn.Config(
                app=app,
                host=settings.API_HOST,
                port=settings.API_PORT,
                log_level="info",
                access_log=True,
                loop="asyncio"
            )
            
            # Start server
            server = uvicorn.Server(config)
            self.components['api_server'] = server
            
            # Start in background
            asyncio.create_task(server.serve())
            logger.info(f"✅ API server started on {settings.API_HOST}:{settings.API_PORT}")
            
        except Exception as e:
            logger.error(f"❌ API server initialization failed: {e}")
            raise
    
    async def setup_signal_handlers(self):
        """Set up signal handlers for graceful shutdown."""
        def signal_handler(signum, frame):
            logger.info(f"Received signal {signum}, initiating shutdown...")
            self.shutdown_event.set()
        
        # Register signal handlers
        signal.signal(signal.SIGINT, signal_handler)
        signal.signal(signal.SIGTERM, signal_handler)
        
        if hasattr(signal, 'SIGHUP'):
            signal.signal(signal.SIGHUP, signal_handler)
    
    async def wait_for_startup_verification(self):
        """Wait for all components to be fully operational."""
        logger.info("Verifying system startup...")
        
        max_wait_time = 60  # seconds
        check_interval = 5  # seconds
        elapsed_time = 0
        
        while elapsed_time < max_wait_time:
            try:
                # Check if all critical components are running
                checks = {
                    'database': self._check_database_health(),
                    'data_stream': self._check_data_stream_health(),
                    'monitoring': self._check_monitoring_health(),
                    'trading_engine': self._check_trading_engine_health()
                }
                
                results = await asyncio.gather(*checks.values(), return_exceptions=True)
                
                all_healthy = all(
                    not isinstance(result, Exception) and result 
                    for result in results
                )
                
                if all_healthy:
                    logger.info("✅ All systems operational")
                    return True
                
                # Log which components are not ready
                for component, result in zip(checks.keys(), results):
                    if isinstance(result, Exception) or not result:
                        logger.warning(f"⚠️  {component} not ready: {result}")
                
                await asyncio.sleep(check_interval)
                elapsed_time += check_interval
                
            except Exception as e:
                logger.warning(f"Health check failed: {e}")
                await asyncio.sleep(check_interval)
                elapsed_time += check_interval
        
        logger.error("❌ System startup verification failed")
        return False
    
    async def _check_database_health(self) -> bool:
        """Check database connectivity."""
        try:
            async_engine = get_async_database()
            async with async_engine.begin() as conn:
                from sqlalchemy import text
                result = await conn.execute(text("SELECT 1"))
                return result.scalar() == 1
        except Exception:
            return False
    
    async def _check_data_stream_health(self) -> bool:
        """Check data stream status."""
        try:
            stream = self.components.get('binance_stream')
            return stream and stream.is_connected if hasattr(stream, 'is_connected') else False
        except Exception:
            return False
    
    async def _check_monitoring_health(self) -> bool:
        """Check monitoring system status."""
        try:
            monitoring = self.components.get('monitoring')
            return monitoring and monitoring.is_running if hasattr(monitoring, 'is_running') else False
        except Exception:
            return False
    
    async def _check_trading_engine_health(self) -> bool:
        """Check trading engine status."""
        try:
            engine = self.components.get('trading_engine')
            return engine and engine.is_running if hasattr(engine, 'is_running') else False
        except Exception:
            return False
    
    async def send_startup_notification(self):
        """Send notification that system has started."""
        try:
            if 'monitoring' in self.components:
                monitoring = self.components['monitoring']
                notification_manager = getattr(monitoring, 'notification_manager', None)
                
                if notification_manager:
                    await notification_manager.send_system_alert(
                        component="system",
                        level="INFO",
                        message=f"🚀 Crypto Pulse V3 started successfully in {self.startup_duration:.1f}s",
                        details={
                            "start_time": self.start_time.isoformat(),
                            "components": list(self.components.keys()),
                            "version": "3.0.0",
                            "environment": settings.ENVIRONMENT
                        }
                    )
        except Exception as e:
            logger.warning(f"Failed to send startup notification: {e}")
    
    async def start(self):
        """Start the complete trading system."""
        import time
        self.start_time = time.time()
        
        logger.info("🚀 Starting Crypto Pulse V3 Trading System...")
        logger.info(f"Environment: {settings.ENVIRONMENT}")
        logger.info(f"Version: 3.0.0")
        
        try:
            async with self.exit_stack:
                # Set up signal handlers
                await self.setup_signal_handlers()
                
                # Initialize components in order
                await self.initialize_database()
                await self.start_monitoring()  # Start monitoring early
                await self.start_data_streams()
                await self.start_trading_engine()
                await self.start_api_server()
                
                # Verify startup
                startup_success = await self.wait_for_startup_verification()
                if not startup_success:
                    raise Exception("System startup verification failed")
                
                self.running = True
                self.startup_duration = time.time() - self.start_time
                
                # Send startup notification
                await self.send_startup_notification()
                
                logger.info(f"🎉 Crypto Pulse V3 is now running! (startup time: {self.startup_duration:.1f}s)")
                logger.info("System is ready for trading operations")
                logger.info(f"API available at: http://{settings.API_HOST}:{settings.API_PORT}")
                logger.info("Press Ctrl+C to shutdown gracefully")
                
                # Wait for shutdown signal
                await self.shutdown_event.wait()
                
        except Exception as e:
            logger.error(f"❌ System startup failed: {e}")
            raise
        finally:
            await self.shutdown()
    
    async def shutdown(self):
        """Gracefully shutdown all system components."""
        if not self.running:
            return
        
        logger.info("🛑 Initiating graceful shutdown...")
        self.running = False
        
        try:
            # Send shutdown notification
            if 'monitoring' in self.components:
                monitoring = self.components['monitoring']
                notification_manager = getattr(monitoring, 'notification_manager', None)
                
                if notification_manager:
                    await notification_manager.send_system_alert(
                        component="system",
                        level="INFO",
                        message="🛑 Crypto Pulse V3 shutting down gracefully",
                        details={"shutdown_time": asyncio.get_event_loop().time()}
                    )
            
            # Components will be cleaned up by exit_stack
            logger.info("✅ System shutdown completed")
            
        except Exception as e:
            logger.error(f"Error during shutdown: {e}")
    
    def get_system_status(self) -> dict:
        """Get current system status for API endpoints."""
        return {
            "running": self.running,
            "start_time": self.start_time,
            "startup_duration": self.startup_duration,
            "components": {
                name: {
                    "status": "running" if component else "stopped",
                    "type": type(component).__name__ if component else None
                }
                for name, component in self.components.items()
            },
            "version": "3.0.0",
            "environment": settings.ENVIRONMENT
        }


async def main():
    """Main entry point."""
    # Configure logging
    logging.basicConfig(
        level=getattr(logging, settings.LOG_LEVEL),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler('crypto-pulse.log') if settings.ENVIRONMENT == 'production' else logging.NullHandler()
        ]
    )
    
    # Use uvloop for better performance on Unix systems
    if sys.platform != 'win32':
        try:
            import uvloop
            asyncio.set_event_loop_policy(uvloop.EventLoopPolicy())
            logger.info("Using uvloop for enhanced performance")
        except ImportError:
            logger.info("uvloop not available, using default event loop")
    
    # Validate environment configuration
    try:
        settings.validate_config()
        logger.info("✅ Configuration validation passed")
    except Exception as e:
        logger.error(f"❌ Configuration validation failed: {e}")
        sys.exit(1)
    
    # Create and start the system
    system = CryptoPulseSystem()
    
    try:
        await system.start()
    except KeyboardInterrupt:
        logger.info("Received keyboard interrupt")
    except Exception as e:
        logger.error(f"System error: {e}")
        sys.exit(1)
    finally:
        logger.info("Crypto Pulse V3 stopped")


if __name__ == "__main__":
    # Handle Windows-specific event loop policy
    if sys.platform == 'win32':
        asyncio.set_event_loop_policy(asyncio.WindowsProactorEventLoopPolicy())
    
    asyncio.run(main())
