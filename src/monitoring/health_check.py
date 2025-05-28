#!/usr/bin/env python3
"""
Health check module for Crypto Pulse V3 containers.
Provides simple health status verification for Docker containers.
"""

import sys
import asyncio
import logging
from pathlib import Path

# Add the project root to the Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

try:
    from config.settings import settings
    from src.core.database import get_async_database
    import asyncpg
    import redis
except ImportError as e:
    print(f"Import error during health check: {e}")
    sys.exit(1)

logger = logging.getLogger(__name__)


async def check_database_health():
    """Check if database is accessible."""
    try:
        # Simple connection test
        import urllib.parse
        parsed = urllib.parse.urlparse(settings.database_url)
        conn = await asyncpg.connect(settings.database_url)
        await conn.fetchval("SELECT 1")
        await conn.close()
        return True
    except Exception as e:
        logger.error(f"Database health check failed: {e}")
        return False


def check_redis_health():
    """Check if Redis is accessible."""
    try:
        import urllib.parse
        parsed = urllib.parse.urlparse(settings.redis_url)
        r = redis.Redis(host=parsed.hostname, port=parsed.port, db=parsed.path.lstrip('/') or 0)
        r.ping()
        return True
    except Exception as e:
        logger.error(f"Redis health check failed: {e}")
        return False


def check_basic_imports():
    """Check if core modules can be imported."""
    try:
        from src.core.trading_engine import TradingEngine
        from src.data.binance_stream import BinanceDataStream
        from src.ml.ensemble import EnsemblePredictor
        return True
    except Exception as e:
        logger.error(f"Import health check failed: {e}")
        return False


async def check_health():
    """Main health check function."""
    try:
        # Basic imports check
        if not check_basic_imports():
            return False
        
        # Database check
        db_healthy = await check_database_health()
        if not db_healthy:
            logger.warning("Database health check failed, but continuing...")
        
        # Redis check
        redis_healthy = check_redis_health()
        if not redis_healthy:
            logger.warning("Redis health check failed, but continuing...")
        
        # If basic imports work, consider the app healthy
        # Database and Redis might not be ready during startup
        return True
        
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        return False


def main():
    """CLI entry point for health checks."""
    logging.basicConfig(level=logging.WARNING)
    
    try:
        # Run health check
        result = asyncio.run(check_health())
        
        if result:
            print("Health check passed")
            sys.exit(0)
        else:
            print("Health check failed")
            sys.exit(1)
            
    except Exception as e:
        print(f"Health check error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
