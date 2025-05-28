-- Database initialization script for Crypto Pulse V3
-- This script sets up the initial database schema and configuration

-- Set timezone to UTC for consistent timestamp handling
SET timezone = 'UTC';

-- Create necessary extensions
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";
CREATE EXTENSION IF NOT EXISTS "pg_stat_statements";
CREATE EXTENSION IF NOT EXISTS "btree_gin";
CREATE EXTENSION IF NOT EXISTS "pg_trgm";

-- Create custom types
DO $$ BEGIN
    CREATE TYPE trade_status AS ENUM ('pending', 'executed', 'cancelled', 'failed');
EXCEPTION
    WHEN duplicate_object THEN null;
END $$;

DO $$ BEGIN
    CREATE TYPE order_type AS ENUM ('market', 'limit', 'stop_loss', 'take_profit');
EXCEPTION
    WHEN duplicate_object THEN null;
END $$;

DO $$ BEGIN
    CREATE TYPE trade_side AS ENUM ('buy', 'sell');
EXCEPTION
    WHEN duplicate_object THEN null;
END $$;

-- Create a function to update updated_at timestamp
CREATE OR REPLACE FUNCTION update_updated_at_column()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = CURRENT_TIMESTAMP;
    RETURN NEW;
END;
$$ language 'plpgsql';

-- Create a function for trade PnL calculation
CREATE OR REPLACE FUNCTION calculate_trade_pnl(
    entry_price DECIMAL,
    exit_price DECIMAL,
    quantity DECIMAL,
    side trade_side
)
RETURNS DECIMAL AS $$
BEGIN
    IF side = 'buy' THEN
        RETURN (exit_price - entry_price) * quantity;
    ELSE
        RETURN (entry_price - exit_price) * quantity;
    END IF;
END;
$$ language 'plpgsql';

-- Create indexes for common queries (will be applied after tables are created)
-- These are stored procedures to be called after table creation

CREATE OR REPLACE FUNCTION create_performance_indexes()
RETURNS void AS $$
BEGIN
    -- Market data indexes
    IF EXISTS (SELECT 1 FROM information_schema.tables WHERE table_name = 'market_data') THEN
        CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_market_data_symbol_timestamp 
        ON market_data(symbol, timestamp DESC);
        
        CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_market_data_timestamp 
        ON market_data(timestamp DESC);
    END IF;

    -- Trades indexes
    IF EXISTS (SELECT 1 FROM information_schema.tables WHERE table_name = 'trades') THEN
        CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_trades_symbol_timestamp 
        ON trades(symbol, created_at DESC);
        
        CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_trades_status 
        ON trades(status);
        
        CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_trades_entry_time 
        ON trades(entry_time DESC);
    END IF;

    -- Portfolio positions indexes
    IF EXISTS (SELECT 1 FROM information_schema.tables WHERE table_name = 'portfolio_positions') THEN
        CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_portfolio_positions_symbol 
        ON portfolio_positions(symbol);
        
        CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_portfolio_positions_updated_at 
        ON portfolio_positions(updated_at DESC);
    END IF;

    -- Backtesting results indexes
    IF EXISTS (SELECT 1 FROM information_schema.tables WHERE table_name = 'backtest_results') THEN
        CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_backtest_results_strategy 
        ON backtest_results(strategy_name);
        
        CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_backtest_results_created_at 
        ON backtest_results(created_at DESC);
    END IF;

    RAISE NOTICE 'Performance indexes created successfully';
END;
$$ language 'plpgsql';

-- Database configuration optimizations
ALTER SYSTEM SET shared_preload_libraries = 'pg_stat_statements';
ALTER SYSTEM SET log_statement = 'all';
ALTER SYSTEM SET log_duration = on;
ALTER SYSTEM SET track_activities = on;
ALTER SYSTEM SET track_counts = on;

-- Set optimal memory settings for trading workload
ALTER SYSTEM SET effective_cache_size = '256MB';
ALTER SYSTEM SET shared_buffers = '64MB';
ALTER SYSTEM SET work_mem = '16MB';
ALTER SYSTEM SET maintenance_work_mem = '32MB';

-- Connection and performance settings
ALTER SYSTEM SET max_connections = 100;
ALTER SYSTEM SET checkpoint_completion_target = 0.7;
ALTER SYSTEM SET wal_buffers = '16MB';

-- Reload configuration
SELECT pg_reload_conf();

-- Log successful initialization
DO $$
BEGIN
    RAISE NOTICE 'Crypto Pulse V3 database initialized successfully at %', NOW();
    RAISE NOTICE 'Database name: %', current_database();
    RAISE NOTICE 'Database version: %', version();
END
$$;
