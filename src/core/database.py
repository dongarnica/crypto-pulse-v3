"""
Database connection and session management.
"""

from sqlalchemy import create_engine, text
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession, async_sessionmaker
from sqlalchemy.orm import sessionmaker, Session
from sqlalchemy.pool import QueuePool
from contextlib import contextmanager, asynccontextmanager
from typing import Generator, AsyncGenerator
import logging
import asyncio

from config.settings import settings
from src.core.models import Base

logger = logging.getLogger(__name__)


class DatabaseManager:
    """Manages database connections and sessions."""
    
    def __init__(self):
        # Synchronous engine
        self.engine = create_engine(
            settings.database_url,
            poolclass=QueuePool,
            pool_size=20,
            max_overflow=30,
            pool_pre_ping=True,
            pool_recycle=3600,
            echo=settings.log_level == "DEBUG"
        )
        self.SessionLocal = sessionmaker(
            autocommit=False,
            autoflush=False,
            bind=self.engine
        )
        
        # Async engine
        async_url = settings.database_url.replace('postgresql://', 'postgresql+asyncpg://')
        self.async_engine = create_async_engine(
            async_url,
            pool_size=20,
            max_overflow=30,
            pool_pre_ping=True,
            pool_recycle=3600,
            echo=settings.log_level == "DEBUG"
        )
        self.AsyncSessionLocal = async_sessionmaker(
            autocommit=False,
            autoflush=False,
            bind=self.async_engine,
            class_=AsyncSession
        )

    def create_tables(self):
        """Create all database tables."""
        try:
            Base.metadata.create_all(bind=self.engine)
            logger.info("Database tables created successfully")
        except Exception as e:
            logger.error(f"Error creating tables: {e}")
            raise
    
    def setup_timescaledb(self):
        """Set up TimescaleDB hypertables for time-series optimization."""
        hypertables = [
            {
                'table': 'market_data',
                'time_column': 'timestamp',
                'chunk_interval': "'1 hour'"
            },
            {
                'table': 'order_book_snapshots',
                'time_column': 'timestamp',
                'chunk_interval': "'30 minutes'"
            },
            {
                'table': 'portfolio_snapshots',
                'time_column': 'timestamp',
                'chunk_interval': "'1 day'"
            },
            {
                'table': 'sentiment_data',
                'time_column': 'timestamp',
                'chunk_interval': "'2 hours'"
            }
        ]
        
        with self.get_session() as session:
            # Enable TimescaleDB extension
            try:
                session.execute(text("CREATE EXTENSION IF NOT EXISTS timescaledb;"))
                session.commit()
                logger.info("TimescaleDB extension enabled")
            except Exception as e:
                logger.warning(f"Could not enable TimescaleDB extension: {e}")
                return
            
            # Create hypertables
            for ht in hypertables:
                try:
                    sql = f"""
                    SELECT create_hypertable(
                        '{ht['table']}', 
                        '{ht['time_column']}',
                        chunk_time_interval => INTERVAL {ht['chunk_interval']},
                        if_not_exists => TRUE
                    );
                    """
                    session.execute(text(sql))
                    session.commit()
                    logger.info(f"Hypertable created for {ht['table']}")
                except Exception as e:
                    logger.warning(f"Could not create hypertable for {ht['table']}: {e}")
    
    def setup_compression(self):
        """Set up compression policies for historical data."""
        compression_policies = [
            {
                'table': 'market_data',
                'compress_after': "'7 days'"
            },
            {
                'table': 'order_book_snapshots',
                'compress_after': "'2 days'"
            },
            {
                'table': 'portfolio_snapshots',
                'compress_after': "'30 days'"
            }
        ]
        
        with self.get_session() as session:
            for cp in compression_policies:
                try:
                    sql = f"""
                    ALTER TABLE {cp['table']} SET (
                        timescaledb.compress,
                        timescaledb.compress_segmentby = 'symbol'
                    );
                    
                    SELECT add_compression_policy(
                        '{cp['table']}', 
                        INTERVAL {cp['compress_after']}
                    );
                    """
                    session.execute(text(sql))
                    session.commit()
                    logger.info(f"Compression policy set for {cp['table']}")
                except Exception as e:
                    logger.warning(f"Could not set compression for {cp['table']}: {e}")
    
    def setup_retention_policies(self):
        """Set up data retention policies."""
        retention_policies = [
            {
                'table': 'order_book_snapshots',
                'drop_after': "'30 days'"
            },
            {
                'table': 'sentiment_data',
                'drop_after': "'90 days'"
            }
        ]
        
        with self.get_session() as session:
            for rp in retention_policies:
                try:
                    sql = f"""
                    SELECT add_retention_policy(
                        '{rp['table']}',
                        INTERVAL {rp['drop_after']}
                    );
                    """
                    session.execute(text(sql))
                    session.commit()
                    logger.info(f"Retention policy set for {rp['table']}")
                except Exception as e:
                    logger.warning(f"Could not set retention for {rp['table']}: {e}")
    
    @contextmanager
    def get_session(self) -> Generator[Session, None, None]:
        """Get database session with automatic cleanup."""
        session = self.SessionLocal()
        try:
            yield session
        except Exception:
            session.rollback()
            raise
        finally:
            session.close()
    
    def get_session_sync(self) -> Session:
        """Get synchronous database session."""
        return self.SessionLocal()
    
    async def get_session_async(self) -> AsyncGenerator[AsyncSession, None]:
        """Get asynchronous database session."""
        async with self.AsyncSessionLocal() as session:
            yield session
    
    @asynccontextmanager
    async def get_async_session(self) -> AsyncGenerator[AsyncSession, None]:
        """Get async database session with automatic cleanup."""
        session = self.AsyncSessionLocal()
        try:
            yield session
            await session.commit()
        except Exception:
            await session.rollback()
            raise
        finally:
            await session.close()

    def health_check(self) -> bool:
        """Check database connectivity."""
        try:
            with self.get_session() as session:
                session.execute(text("SELECT 1"))
                return True
        except Exception as e:
            logger.error(f"Database health check failed: {e}")
            return False

    async def health_check_async(self) -> bool:
        """Check database connectivity asynchronously."""
        try:
            async with self.get_async_session() as session:
                await session.execute(text("SELECT 1"))
                return True
        except Exception as e:
            logger.error(f"Async database health check failed: {e}")
            return False


# Global database manager instance
db_manager = DatabaseManager()


def get_db() -> Generator[Session, None, None]:
    """Dependency for FastAPI to get database session."""
    with db_manager.get_session() as session:
        yield session


async def get_async_db() -> AsyncGenerator[AsyncSession, None]:
    """Dependency for FastAPI to get async database session."""
    async with db_manager.get_async_session() as session:
        yield session


def get_async_database():
    """Get the async database engine."""
    return db_manager.async_engine


async def init_database():
    """Initialize database with tables and TimescaleDB setup (async version)."""
    logger.info("Initializing database...")
    
    # Create tables (async)
    async with db_manager.async_engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    
    # Set up TimescaleDB features using sync connection
    db_manager.setup_timescaledb()
    db_manager.setup_compression()
    db_manager.setup_retention_policies()
    
    logger.info("Database initialization completed")


def init_database_sync():
    """Initialize database with tables and TimescaleDB setup (sync version)."""
    logger.info("Initializing database...")
    
    # Create tables
    db_manager.create_tables()
    
    # Set up TimescaleDB features
    db_manager.setup_timescaledb()
    db_manager.setup_compression()
    db_manager.setup_retention_policies()
    
    logger.info("Database initialization completed")


if __name__ == "__main__":
    init_database_sync()
