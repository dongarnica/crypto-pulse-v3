#!/usr/bin/env python3
"""
Database initialization script for Crypto Pulse V3.
Creates the PostgreSQL database, TimescaleDB extension, and all required tables.
"""

import asyncio
import sys
import logging
from pathlib import Path

# Add the project root to the Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import asyncpg
from sqlalchemy import create_engine, text
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession
from sqlalchemy.orm import sessionmaker

from config.settings import settings
from src.core.models import Base
from src.core.database import get_database, get_async_database

logger = logging.getLogger(__name__)


async def create_database_if_not_exists():
    """Create the main database if it doesn't exist."""
    # Connect to PostgreSQL server (not to our specific database)
    server_url = f"postgresql://{settings.DB_USER}:{settings.DB_PASSWORD}@{settings.DB_HOST}:{settings.DB_PORT}/postgres"
    
    try:
        conn = await asyncpg.connect(server_url)
        
        # Check if database exists
        result = await conn.fetchval(
            "SELECT 1 FROM pg_database WHERE datname = $1", 
            settings.DB_NAME
        )
        
        if not result:
            logger.info(f"Creating database '{settings.DB_NAME}'...")
            await conn.execute(f'CREATE DATABASE "{settings.DB_NAME}"')
            logger.info("Database created successfully")
        else:
            logger.info(f"Database '{settings.DB_NAME}' already exists")
            
        await conn.close()
        
    except Exception as e:
        logger.error(f"Error creating database: {e}")
        raise


async def create_timescaledb_extension():
    """Create TimescaleDB extension for time-series optimization."""
    try:
        async_engine = get_async_database()
        
        async with async_engine.begin() as conn:
            # Create TimescaleDB extension
            await conn.execute(text("CREATE EXTENSION IF NOT EXISTS timescaledb;"))
            logger.info("TimescaleDB extension created/verified")
            
    except Exception as e:
        logger.error(f"Error creating TimescaleDB extension: {e}")
        raise


async def create_tables():
    """Create all database tables."""
    try:
        # Use synchronous engine for table creation
        engine = create_engine(
            f"postgresql://{settings.DB_USER}:{settings.DB_PASSWORD}@{settings.DB_HOST}:{settings.DB_PORT}/{settings.DB_NAME}"
        )
        
        logger.info("Creating database tables...")
        Base.metadata.create_all(engine)
        logger.info("Database tables created successfully")
        
        engine.dispose()
        
    except Exception as e:
        logger.error(f"Error creating tables: {e}")
        raise


async def create_hypertables():
    """Convert regular tables to TimescaleDB hypertables for time-series optimization."""
    try:
        async_engine = get_async_database()
        
        async with async_engine.begin() as conn:
            # Create hypertables for time-series data
            hypertables = [
                ("market_data", "timestamp"),
                ("order_book_snapshots", "timestamp"),
                ("trading_signals", "timestamp"),
                ("trades", "created_at"),
                ("portfolio_snapshots", "timestamp"),
                ("sentiment_data", "timestamp"),
                ("model_performance", "timestamp")
            ]
            
            for table_name, time_column in hypertables:
                try:
                    # Check if hypertable already exists
                    result = await conn.execute(text(
                        "SELECT COUNT(*) FROM timescaledb_information.hypertables WHERE hypertable_name = :table_name"
                    ), {"table_name": table_name})
                    
                    count = result.scalar()
                    
                    if count == 0:
                        await conn.execute(text(
                            f"SELECT create_hypertable('{table_name}', '{time_column}', if_not_exists => TRUE);"
                        ))
                        logger.info(f"Created hypertable for {table_name}")
                    else:
                        logger.info(f"Hypertable {table_name} already exists")
                        
                except Exception as e:
                    logger.warning(f"Could not create hypertable for {table_name}: {e}")
                    
    except Exception as e:
        logger.error(f"Error creating hypertables: {e}")
        raise


async def create_indexes():
    """Create additional indexes for performance optimization."""
    try:
        async_engine = get_async_database()
        
        async with async_engine.begin() as conn:
            indexes = [
                # Market data performance indexes
                "CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_market_data_symbol_timestamp_desc ON market_data (symbol, timestamp DESC);",
                "CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_market_data_close_price ON market_data (close_price) WHERE timestamp > NOW() - INTERVAL '30 days';",
                
                # Trading signals optimization
                "CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_signals_confidence ON trading_signals (confidence_score DESC) WHERE is_active = true;",
                "CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_signals_symbol_recent ON trading_signals (symbol, timestamp DESC) WHERE timestamp > NOW() - INTERVAL '7 days';",
                
                # Trade execution indexes
                "CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_trades_execution_time ON trades (executed_at DESC) WHERE status = 'FILLED';",
                "CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_trades_pnl ON trades (symbol, average_fill_price) WHERE status = 'FILLED';",
                
                # Position management
                "CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_positions_open_pnl ON positions (unrealized_pnl DESC) WHERE is_open = true;",
                "CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_positions_risk ON positions (risk_amount DESC) WHERE is_open = true;",
                
                # Portfolio performance
                "CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_portfolio_performance ON portfolio_snapshots (timestamp DESC, total_value DESC);",
                "CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_portfolio_drawdown ON portfolio_snapshots (current_drawdown DESC) WHERE current_drawdown > 0;",
                
                # Sentiment analysis
                "CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_sentiment_recent ON sentiment_data (timestamp DESC, symbol) WHERE timestamp > NOW() - INTERVAL '24 hours';",
            ]
            
            for index_sql in indexes:
                try:
                    await conn.execute(text(index_sql))
                    logger.info(f"Created index: {index_sql.split('idx_')[1].split(' ')[0] if 'idx_' in index_sql else 'custom'}")
                except Exception as e:
                    logger.warning(f"Could not create index: {e}")
                    
    except Exception as e:
        logger.error(f"Error creating indexes: {e}")
        raise


async def setup_compression():
    """Set up compression policies for TimescaleDB to optimize storage."""
    try:
        async_engine = get_async_database()
        
        async with async_engine.begin() as conn:
            # Add compression policies for old data
            compression_policies = [
                ("market_data", "7 days"),
                ("order_book_snapshots", "3 days"),
                ("trading_signals", "30 days"),
                ("portfolio_snapshots", "30 days"),
                ("sentiment_data", "14 days"),
                ("model_performance", "30 days")
            ]
            
            for table_name, compress_after in compression_policies:
                try:
                    # Enable compression
                    await conn.execute(text(
                        f"ALTER TABLE {table_name} SET (timescaledb.compress, timescaledb.compress_segmentby = 'symbol');"
                    ))
                    
                    # Add compression policy
                    await conn.execute(text(
                        f"SELECT add_compression_policy('{table_name}', INTERVAL '{compress_after}');"
                    ))
                    
                    logger.info(f"Enabled compression for {table_name} after {compress_after}")
                    
                except Exception as e:
                    logger.warning(f"Could not enable compression for {table_name}: {e}")
                    
    except Exception as e:
        logger.error(f"Error setting up compression: {e}")
        # Don't raise - compression is optional


async def setup_retention_policies():
    """Set up data retention policies to manage storage."""
    try:
        async_engine = get_async_database()
        
        async with async_engine.begin() as conn:
            # Set up retention policies
            retention_policies = [
                ("order_book_snapshots", "30 days"),  # Keep order book data for 30 days
                ("trading_signals", "1 year"),         # Keep signals for 1 year
                ("model_performance", "2 years"),      # Keep model performance for 2 years
                ("sentiment_data", "6 months"),        # Keep sentiment data for 6 months
            ]
            
            for table_name, retain_for in retention_policies:
                try:
                    await conn.execute(text(
                        f"SELECT add_retention_policy('{table_name}', INTERVAL '{retain_for}');"
                    ))
                    logger.info(f"Added retention policy for {table_name}: {retain_for}")
                    
                except Exception as e:
                    logger.warning(f"Could not add retention policy for {table_name}: {e}")
                    
    except Exception as e:
        logger.error(f"Error setting up retention policies: {e}")
        # Don't raise - retention is optional


async def insert_initial_data():
    """Insert initial reference data."""
    try:
        async_engine = get_async_database()
        async_session = sessionmaker(async_engine, class_=AsyncSession, expire_on_commit=False)
        
        async with async_session() as session:
            # Insert initial trading pairs if not exists
            initial_symbols = [
                'BTCUSDT', 'ETHUSDT', 'BNBUSDT', 'ADAUSDT', 'DOTUSDT',
                'LINKUSDT', 'LTCUSDT', 'BCHUSDT', 'XLMUSDT', 'UNIUSDT',
                'AAVEUSDT', 'SUSHIUSDT', 'COMPUSDT', 'MKRUSDT', 'YFIUSDT'
            ]
            
            # You could insert initial configuration data here
            # For now, we'll just log the symbols that will be tracked
            logger.info(f"System configured to track {len(initial_symbols)} trading pairs")
            logger.info(f"Trading pairs: {', '.join(initial_symbols)}")
            
            await session.commit()
            
    except Exception as e:
        logger.error(f"Error inserting initial data: {e}")
        raise


async def verify_setup():
    """Verify that the database setup was successful."""
    try:
        async_engine = get_async_database()
        
        async with async_engine.begin() as conn:
            # Check that TimescaleDB is working
            result = await conn.execute(text("SELECT extname FROM pg_extension WHERE extname = 'timescaledb';"))
            timescale_exists = result.scalar()
            
            if timescale_exists:
                logger.info("✓ TimescaleDB extension verified")
            else:
                logger.warning("✗ TimescaleDB extension not found")
            
            # Check hypertables
            result = await conn.execute(text("SELECT COUNT(*) FROM timescaledb_information.hypertables;"))
            hypertable_count = result.scalar()
            logger.info(f"✓ {hypertable_count} hypertables created")
            
            # Check tables
            result = await conn.execute(text(
                "SELECT COUNT(*) FROM information_schema.tables WHERE table_schema = 'public';"
            ))
            table_count = result.scalar()
            logger.info(f"✓ {table_count} tables created")
            
            logger.info("Database setup verification completed successfully")
            
    except Exception as e:
        logger.error(f"Error verifying setup: {e}")
        raise


async def main():
    """Main initialization function."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    logger.info("Starting Crypto Pulse V3 database initialization...")
    
    try:
        # Step 1: Create database if it doesn't exist
        await create_database_if_not_exists()
        
        # Step 2: Create TimescaleDB extension
        await create_timescaledb_extension()
        
        # Step 3: Create all tables
        await create_tables()
        
        # Step 4: Convert to hypertables
        await create_hypertables()
        
        # Step 5: Create performance indexes
        await create_indexes()
        
        # Step 6: Set up compression (optional)
        await setup_compression()
        
        # Step 7: Set up retention policies (optional)
        await setup_retention_policies()
        
        # Step 8: Insert initial data
        await insert_initial_data()
        
        # Step 9: Verify setup
        await verify_setup()
        
        logger.info("🚀 Database initialization completed successfully!")
        logger.info("The Crypto Pulse V3 trading system is ready to start.")
        
    except Exception as e:
        logger.error(f"❌ Database initialization failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())
