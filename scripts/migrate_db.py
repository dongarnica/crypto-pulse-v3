#!/usr/bin/env python3
"""
Database migration utilities for Crypto Pulse V3.
Handles schema versioning and migrations.
"""

import asyncio
import sys
import logging
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Any

# Add the project root to the Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import asyncpg
from sqlalchemy import text
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession
from sqlalchemy.orm import sessionmaker

from config.settings import settings
from src.core.database import get_async_database

logger = logging.getLogger(__name__)


class DatabaseMigration:
    """Handles database schema migrations."""
    
    def __init__(self):
        self.engine = get_async_database()
        self.migrations_dir = Path(__file__).parent / "migrations"
        self.migrations_dir.mkdir(exist_ok=True)
    
    async def create_migration_table(self):
        """Create the migrations tracking table."""
        async with self.engine.begin() as conn:
            await conn.execute(text("""
                CREATE TABLE IF NOT EXISTS schema_migrations (
                    id SERIAL PRIMARY KEY,
                    version VARCHAR(50) NOT NULL UNIQUE,
                    name VARCHAR(255) NOT NULL,
                    applied_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    checksum VARCHAR(64)
                );
            """))
            
            # Create index for version lookups
            await conn.execute(text("""
                CREATE INDEX IF NOT EXISTS idx_migrations_version 
                ON schema_migrations (version);
            """))
    
    async def get_applied_migrations(self) -> List[str]:
        """Get list of applied migration versions."""
        try:
            async with self.engine.begin() as conn:
                result = await conn.execute(text(
                    "SELECT version FROM schema_migrations ORDER BY applied_at"
                ))
                return [row[0] for row in result.fetchall()]
        except Exception:
            # If table doesn't exist yet, return empty list
            return []
    
    async def apply_migration(self, version: str, name: str, sql: str):
        """Apply a single migration."""
        try:
            async with self.engine.begin() as conn:
                # Execute the migration SQL
                await conn.execute(text(sql))
                
                # Record the migration as applied
                await conn.execute(text("""
                    INSERT INTO schema_migrations (version, name, checksum)
                    VALUES (:version, :name, :checksum)
                """), {
                    "version": version,
                    "name": name,
                    "checksum": hash(sql) % (10**16)  # Simple checksum
                })
                
                logger.info(f"Applied migration {version}: {name}")
                
        except Exception as e:
            logger.error(f"Failed to apply migration {version}: {e}")
            raise
    
    async def run_migrations(self):
        """Run all pending migrations."""
        await self.create_migration_table()
        applied_migrations = await self.get_applied_migrations()
        
        # Get all migration files
        migration_files = sorted(self.migrations_dir.glob("*.sql"))
        
        for migration_file in migration_files:
            version = migration_file.stem
            
            if version not in applied_migrations:
                name = version.replace("_", " ").title()
                sql = migration_file.read_text()
                
                logger.info(f"Running migration: {version}")
                await self.apply_migration(version, name, sql)
            else:
                logger.debug(f"Migration {version} already applied")
        
        logger.info("All migrations completed")
    
    def create_migration_file(self, name: str, sql: str) -> str:
        """Create a new migration file."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        version = f"{timestamp}_{name.lower().replace(' ', '_')}"
        
        migration_file = self.migrations_dir / f"{version}.sql"
        migration_file.write_text(sql)
        
        logger.info(f"Created migration file: {migration_file}")
        return version


# Predefined migrations for common operations
MIGRATION_TEMPLATES = {
    "add_column": """
-- Add column {column_name} to {table_name}
ALTER TABLE {table_name} 
ADD COLUMN IF NOT EXISTS {column_name} {data_type};
""",
    
    "create_index": """
-- Create index {index_name}
CREATE INDEX CONCURRENTLY IF NOT EXISTS {index_name} 
ON {table_name} ({columns});
""",
    
    "add_constraint": """
-- Add constraint {constraint_name} to {table_name}
ALTER TABLE {table_name}
ADD CONSTRAINT {constraint_name} {constraint_definition};
""",
    
    "create_hypertable": """
-- Convert {table_name} to hypertable
SELECT create_hypertable('{table_name}', '{time_column}', if_not_exists => TRUE);
""",
    
    "add_compression": """
-- Enable compression for {table_name}
ALTER TABLE {table_name} SET (
    timescaledb.compress,
    timescaledb.compress_segmentby = '{segment_by}'
);

SELECT add_compression_policy('{table_name}', INTERVAL '{compress_after}');
""",
    
    "add_retention": """
-- Add retention policy for {table_name}
SELECT add_retention_policy('{table_name}', INTERVAL '{retain_for}');
"""
}


async def create_performance_optimizations():
    """Create additional performance optimization migrations."""
    migrator = DatabaseMigration()
    
    # Advanced indexing for frequently queried patterns
    advanced_indexes = """
-- Advanced performance indexes for Crypto Pulse V3

-- Partial index for recent active signals
CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_signals_recent_active
ON trading_signals (symbol, confidence_score DESC, timestamp DESC)
WHERE is_active = true AND timestamp > NOW() - INTERVAL '1 day';

-- Covering index for portfolio performance queries
CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_portfolio_perf_covering
ON portfolio_snapshots (timestamp DESC) 
INCLUDE (total_value, daily_return, sharpe_ratio, max_drawdown);

-- Multi-column index for trade analysis
CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_trades_analysis
ON trades (symbol, status, side, executed_at DESC)
WHERE status = 'FILLED';

-- Sentiment correlation index
CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_sentiment_correlation
ON sentiment_data (symbol, timestamp DESC, news_sentiment, social_sentiment)
WHERE timestamp > NOW() - INTERVAL '7 days';

-- Position risk management index
CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_positions_risk_mgmt
ON positions (symbol, is_open, risk_amount DESC, unrealized_pnl)
WHERE is_open = true;
"""
    
    migrator.create_migration_file("advanced_performance_indexes", advanced_indexes)
    
    # Materialized views for common aggregations
    materialized_views = """
-- Materialized views for performance optimization

-- Daily trading summary
CREATE MATERIALIZED VIEW IF NOT EXISTS daily_trading_summary AS
SELECT 
    DATE(executed_at) as trade_date,
    symbol,
    COUNT(*) as trade_count,
    SUM(CASE WHEN side = 'BUY' THEN quantity ELSE 0 END) as total_bought,
    SUM(CASE WHEN side = 'SELL' THEN quantity ELSE 0 END) as total_sold,
    AVG(average_fill_price) as avg_price,
    SUM(commission) as total_fees
FROM trades 
WHERE status = 'FILLED' AND executed_at >= CURRENT_DATE - INTERVAL '30 days'
GROUP BY DATE(executed_at), symbol
ORDER BY trade_date DESC, symbol;

CREATE UNIQUE INDEX ON daily_trading_summary (trade_date, symbol);

-- Hourly market data summary
CREATE MATERIALIZED VIEW IF NOT EXISTS hourly_market_summary AS
SELECT 
    DATE_TRUNC('hour', timestamp) as hour_timestamp,
    symbol,
    FIRST(open_price, timestamp) as open_price,
    MAX(high_price) as high_price,
    MIN(low_price) as low_price,
    LAST(close_price, timestamp) as close_price,
    SUM(volume) as total_volume,
    AVG(rsi_14) as avg_rsi,
    LAST(bb_upper, timestamp) as bb_upper,
    LAST(bb_lower, timestamp) as bb_lower
FROM market_data 
WHERE timestamp >= CURRENT_TIMESTAMP - INTERVAL '7 days'
GROUP BY DATE_TRUNC('hour', timestamp), symbol
ORDER BY hour_timestamp DESC, symbol;

CREATE UNIQUE INDEX ON hourly_market_summary (hour_timestamp, symbol);

-- Model performance summary
CREATE MATERIALIZED VIEW IF NOT EXISTS model_performance_summary AS
SELECT 
    model_name,
    DATE(timestamp) as performance_date,
    AVG(accuracy) as avg_accuracy,
    AVG(precision) as avg_precision,
    AVG(recall) as avg_recall,
    AVG(f1_score) as avg_f1,
    AVG(directional_accuracy) as avg_directional_accuracy,
    AVG(win_rate) as avg_win_rate
FROM model_performance 
WHERE timestamp >= CURRENT_DATE - INTERVAL '30 days'
GROUP BY model_name, DATE(timestamp)
ORDER BY performance_date DESC, model_name;

CREATE UNIQUE INDEX ON model_performance_summary (performance_date, model_name);
"""
    
    migrator.create_migration_file("materialized_views", materialized_views)
    
    # Functions for common calculations
    sql_functions = """
-- Custom SQL functions for trading calculations

-- Calculate portfolio Sharpe ratio
CREATE OR REPLACE FUNCTION calculate_sharpe_ratio(
    returns NUMERIC[],
    risk_free_rate NUMERIC DEFAULT 0.02
) RETURNS NUMERIC AS $$
DECLARE
    avg_return NUMERIC;
    std_dev NUMERIC;
BEGIN
    SELECT AVG(unnest), STDDEV(unnest) 
    INTO avg_return, std_dev 
    FROM unnest(returns);
    
    IF std_dev = 0 OR std_dev IS NULL THEN
        RETURN 0;
    END IF;
    
    RETURN (avg_return - risk_free_rate / 252) / std_dev * SQRT(252);
END;
$$ LANGUAGE plpgsql;

-- Calculate maximum drawdown
CREATE OR REPLACE FUNCTION calculate_max_drawdown(portfolio_values NUMERIC[])
RETURNS NUMERIC AS $$
DECLARE
    max_dd NUMERIC := 0;
    peak NUMERIC := 0;
    current_dd NUMERIC;
    value NUMERIC;
BEGIN
    FOREACH value IN ARRAY portfolio_values LOOP
        IF value > peak THEN
            peak := value;
        END IF;
        
        current_dd := (peak - value) / peak;
        
        IF current_dd > max_dd THEN
            max_dd := current_dd;
        END IF;
    END LOOP;
    
    RETURN max_dd;
END;
$$ LANGUAGE plpgsql;

-- Calculate correlation between two price series
CREATE OR REPLACE FUNCTION calculate_correlation(
    series1 NUMERIC[],
    series2 NUMERIC[]
) RETURNS NUMERIC AS $$
BEGIN
    RETURN CORR(unnest_1, unnest_2)
    FROM (
        SELECT unnest(series1) as unnest_1, unnest(series2) as unnest_2
    ) AS correlation_data;
END;
$$ LANGUAGE plpgsql;

-- Calculate Kelly Criterion optimal position size
CREATE OR REPLACE FUNCTION kelly_criterion(
    win_rate NUMERIC,
    avg_win NUMERIC,
    avg_loss NUMERIC
) RETURNS NUMERIC AS $$
DECLARE
    b NUMERIC; -- odds ratio
    p NUMERIC; -- probability of win
    q NUMERIC; -- probability of loss
BEGIN
    IF avg_loss = 0 OR avg_loss IS NULL THEN
        RETURN 0;
    END IF;
    
    b := avg_win / ABS(avg_loss);
    p := win_rate;
    q := 1 - win_rate;
    
    -- Kelly fraction: f = (bp - q) / b
    RETURN GREATEST(0, LEAST(0.25, (b * p - q) / b)); -- Cap at 25% for safety
END;
$$ LANGUAGE plpgsql;
"""
    
    migrator.create_migration_file("trading_functions", sql_functions)
    
    return migrator


async def main():
    """Main migration function."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    logger.info("Starting database migrations...")
    
    try:
        # Create performance optimization migrations
        migrator = await create_performance_optimizations()
        
        # Run all pending migrations
        await migrator.run_migrations()
        
        logger.info("✅ All migrations completed successfully!")
        
    except Exception as e:
        logger.error(f"❌ Migration failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())
