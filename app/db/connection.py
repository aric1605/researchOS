"""Database connection management and migrations for PostgreSQL (Neon)."""

import os
import time
import logging
from typing import Optional, Dict, Any
from pathlib import Path
from contextlib import asynccontextmanager

import psycopg
from psycopg.rows import dict_row
from psycopg_pool import AsyncConnectionPool

from app.config import settings

logger = logging.getLogger("researchos.db")

_pool: Optional[AsyncConnectionPool] = None


def get_connection_string() -> Optional[str]:
    """Returns cleaned DATABASE_URL with sslmode if neon/cloud."""
    url = settings.DATABASE_URL
    if not url:
        return None
    # Strip any problematic channel_binding parameters
    if "channel_binding=" in url:
        import re
        url = re.sub(r"[?&]channel_binding=[^&]*", "", url)
        if "?" not in url and "&" in url:
            url = url.replace("&", "?", 1)
    return url


async def get_db_pool() -> Optional[AsyncConnectionPool]:
    """Retrieves or creates the global async connection pool."""
    global _pool
    conn_str = get_connection_string()
    if not conn_str:
        return None

    if _pool is None:
        try:
            _pool = AsyncConnectionPool(
                conninfo=conn_str,
                min_size=1,
                max_size=10,
                open=False,
                timeout=10.0,
                kwargs={"row_factory": dict_row},
            )
            await _pool.open()
            logger.info("Database connection pool initialized.")
        except Exception as e:
            logger.error(f"Failed to initialize connection pool: {e}")
            _pool = None
    return _pool


async def close_db_pool() -> None:
    """Closes the async connection pool."""
    global _pool
    if _pool is not None:
        await _pool.close()
        _pool = None
        logger.info("Database connection pool closed.")


async def run_migrations() -> bool:
    """Applies SQL migrations from migrations/ directory."""
    conn_str = get_connection_string()
    if not conn_str:
        logger.warning("DATABASE_URL not set; skipping migrations.")
        return False

    migrations_dir = Path(__file__).resolve().parent.parent.parent / "migrations"
    migration_files = sorted(migrations_dir.glob("*.sql"))

    if not migration_files:
        logger.info("No migration files found.")
        return True

    try:
        async with await psycopg.AsyncConnection.connect(conn_str) as conn:
            async with conn.cursor() as cur:
                for file_path in migration_files:
                    logger.info(f"Applying migration: {file_path.name}")
                    sql = file_path.read_text(encoding="utf-8")
                    await cur.execute(sql)
            await conn.commit()
            logger.info("All migrations executed successfully.")
        return True
    except Exception as e:
        logger.error(f"Migration error: {e}")
        return False


async def check_postgres_health() -> Dict[str, Any]:
    """Health check for PostgreSQL database."""
    conn_str = get_connection_string()
    if not conn_str:
        return {"status": "unconfigured", "message": "DATABASE_URL not set"}

    start_time = time.time()
    try:
        async with await psycopg.AsyncConnection.connect(conn_str, timeout=5.0) as conn:
            async with conn.cursor() as cur:
                await cur.execute("SELECT 1 AS alive;")
                row = await cur.fetchone()
                latency_ms = round((time.time() - start_time) * 1000, 2)
                if row and row[0] == 1:
                    return {"status": "healthy", "latency_ms": latency_ms}
                return {"status": "degraded", "latency_ms": latency_ms}
    except Exception as e:
        latency_ms = round((time.time() - start_time) * 1000, 2)
        return {"status": "unhealthy", "latency_ms": latency_ms, "error": str(e)}
