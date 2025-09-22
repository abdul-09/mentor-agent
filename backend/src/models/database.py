"""
Database Configuration for AI Code Mentor
Implements production-grade database setup with connection pooling and monitoring.

Compliance:
- RULE DB-001: Database normalization (3NF minimum)
- RULE DB-002: UUID primary keys and proper indexing
- RULE PERF-002: Connection pooling (5-20 connections)
- RULE BACKUP-001: Backup-ready configuration
"""

import asyncio
from typing import AsyncGenerator

import structlog
from sqlalchemy import event, MetaData
from sqlalchemy.ext.asyncio import (
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
    AsyncEngine
)
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.pool import NullPool, QueuePool

from src.config.settings import get_settings

logger = structlog.get_logger(__name__)
settings = get_settings()

# Database metadata with naming convention for constraints
metadata = MetaData(
    naming_convention={
        "ix": "ix_%(column_0_label)s",
        "uq": "uq_%(table_name)s_%(column_0_name)s",
        "ck": "ck_%(table_name)s_%(constraint_name)s",
        "fk": "fk_%(table_name)s_%(column_0_name)s_%(referred_table_name)s",
        "pk": "pk_%(table_name)s"
    }
)

# Create declarative base
Base = declarative_base(metadata=metadata)

# Global engine and session maker
engine: AsyncEngine = None
async_session_maker: async_sessionmaker = None


def get_database_url() -> str:
    """Get database URL with environment-specific configuration."""
    if settings.TESTING:
        return settings.TEST_DATABASE_URL or "sqlite+aiosqlite:///:memory:"
    return settings.DATABASE_URL


async def create_database_engine() -> AsyncEngine:
    """
    Create database engine with production-grade configuration.
    
    Implements RULE PERF-002 connection pooling requirements.
    """
    database_url = get_database_url()
    
    # Engine configuration based on database type
    engine_kwargs = {
        "url": database_url,
        "echo": settings.DEBUG and not settings.TESTING,  # SQL logging in debug mode
        "future": True,  # Use SQLAlchemy 2.0 style
    }
    
    if "sqlite" not in database_url:
        # PostgreSQL configuration with connection pooling
        engine_kwargs.update({
            "pool_size": settings.DB_POOL_SIZE,
            "max_overflow": settings.DB_MAX_OVERFLOW,
            "pool_timeout": settings.DB_POOL_TIMEOUT,
            "pool_recycle": settings.DB_POOL_RECYCLE,
            "pool_pre_ping": True,  # Validate connections
            "poolclass": QueuePool,
        })
        
        logger.info(
            "Configuring PostgreSQL connection pool",
            pool_size=settings.DB_POOL_SIZE,
            max_overflow=settings.DB_MAX_OVERFLOW,
            timeout=settings.DB_POOL_TIMEOUT
        )
    else:
        # SQLite configuration (testing)
        engine_kwargs.update({
            "poolclass": NullPool,
            "connect_args": {"check_same_thread": False}
        })
        
        logger.info("Configuring SQLite for testing")
    
    engine = create_async_engine(**engine_kwargs)
    
    # Add connection event listeners for monitoring
    @event.listens_for(engine.sync_engine, "connect")
    def on_connect(dbapi_connection, connection_record):
        logger.debug("Database connection established")
    
    @event.listens_for(engine.sync_engine, "checkout")
    def on_checkout(dbapi_connection, connection_record, connection_proxy):
        logger.debug("Database connection checked out from pool")
    
    @event.listens_for(engine.sync_engine, "checkin")
    def on_checkin(dbapi_connection, connection_record):
        logger.debug("Database connection checked back into pool")
    
    return engine


async def init_database() -> None:
    """
    Initialize database connection and create tables.
    
    This function should be called during application startup.
    """
    global engine, async_session_maker
    
    try:
        logger.info("Initializing database connection")
        
        # Create engine
        engine = await create_database_engine()
        
        # Create session maker
        async_session_maker = async_sessionmaker(
            engine,
            class_=AsyncSession,
            expire_on_commit=False,
            autoflush=False,
            autocommit=False
        )
        
        # Test connection
        async with engine.begin() as conn:
            # Import all models to ensure they're registered
            from src.models import user, file, analysis, session as session_model
            
            # Create all tables
            await conn.run_sync(Base.metadata.create_all)
            
            logger.info("Database tables created successfully")
        
        logger.info("Database initialization completed")
        
    except Exception as e:
        logger.error("Database initialization failed", error=str(e))
        raise


async def close_database() -> None:
    """
    Close database connections gracefully.
    
    This function should be called during application shutdown.
    """
    global engine
    
    if engine:
        logger.info("Closing database connections")
        await engine.dispose()
        engine = None
        logger.info("Database connections closed")


async def get_async_session() -> AsyncGenerator[AsyncSession, None]:
    """
    Get async database session with proper error handling.
    
    Yields:
        AsyncSession: Database session
        
    Raises:
        RuntimeError: If database is not initialized
    """
    if not async_session_maker:
        raise RuntimeError("Database not initialized. Call init_database() first.")
    
    async with async_session_maker() as session:
        try:
            yield session
            await session.commit()
        except Exception as e:
            await session.rollback()
            logger.error("Database session error", error=str(e))
            raise
        finally:
            await session.close()


async def get_db_health() -> dict:
    """
    Check database health for monitoring.
    
    Returns:
        dict: Health status information
    """
    if not engine:
        return {
            "status": "unhealthy",
            "error": "Database not initialized"
        }
    
    try:
        async with engine.begin() as conn:
            result = await conn.execute("SELECT 1")
            await result.fetchone()
        
        # Get pool status
        pool_status = {
            "size": engine.pool.size(),
            "checked_in": engine.pool.checkedin(),
            "checked_out": engine.pool.checkedout(),
            "overflow": engine.pool.overflow(),
        }
        
        return {
            "status": "healthy",
            "pool_status": pool_status,
            "database_url": get_database_url().split("@")[-1] if "@" in get_database_url() else "sqlite"
        }
        
    except Exception as e:
        return {
            "status": "unhealthy",
            "error": str(e)
        }