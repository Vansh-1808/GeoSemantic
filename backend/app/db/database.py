"""
SQLAlchemy async engine and session factory.
PostGIS extension is assumed to be enabled on the database.
"""
from __future__ import annotations

from collections.abc import AsyncGenerator

from sqlalchemy.ext.asyncio import (
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)
from sqlalchemy.orm import DeclarativeBase

from app.core.config import settings
from app.core.logging import get_logger

logger = get_logger(__name__)

from sqlalchemy.pool import NullPool

# Async engine — used by FastAPI routes
engine_kwargs: dict = {
    "echo": False,
    "pool_pre_ping": True,
}
if settings.app_env == "test":
    engine_kwargs["poolclass"] = NullPool
else:
    engine_kwargs["pool_size"] = 10
    engine_kwargs["max_overflow"] = 20

engine = create_async_engine(
    str(settings.database_url),
    **engine_kwargs,
)

# Session factory
AsyncSessionLocal = async_sessionmaker(
    engine,
    class_=AsyncSession,
    expire_on_commit=False,
    autoflush=False,
    autocommit=False,
)


class Base(DeclarativeBase):
    """Base class for all ORM models."""
    pass


async def get_session() -> AsyncGenerator[AsyncSession, None]:
    """FastAPI dependency — yields an async database session."""
    async with AsyncSessionLocal() as session:
        try:
            yield session
            await session.commit()
        except Exception:
            await session.rollback()
            raise
        finally:
            await session.close()


async def check_database_connection() -> bool:
    """Health check — verifies database reachability."""
    try:
        from sqlalchemy import text
        async with engine.connect() as conn:
            await conn.execute(text("SELECT 1"))
        return True
    except Exception as exc:
        logger.error("database_connection_failed", error=str(exc))
        return False


async def enable_postgis(conn) -> None:
    """Enable PostGIS extension if not already enabled."""
    from sqlalchemy import text
    await conn.execute(text("CREATE EXTENSION IF NOT EXISTS postgis"))
    await conn.execute(text("CREATE EXTENSION IF NOT EXISTS postgis_topology"))
    await conn.execute(text('CREATE EXTENSION IF NOT EXISTS "uuid-ossp"'))
