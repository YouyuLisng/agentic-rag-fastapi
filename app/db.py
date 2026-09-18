from pgvector.psycopg import register_vector_async
from psycopg import AsyncConnection
from psycopg_pool import AsyncConnectionPool

from app.config import get_settings

_pool: AsyncConnectionPool | None = None


async def _configure(conn: AsyncConnection) -> None:
    # Lets psycopg adapt Python lists <-> pgvector's `vector` column type
    # directly, on every connection the pool hands out.
    await register_vector_async(conn)


async def init_pool() -> AsyncConnectionPool:
    global _pool
    settings = get_settings()
    _pool = AsyncConnectionPool(conninfo=settings.database_url, open=False, configure=_configure)
    await _pool.open()
    return _pool


async def close_pool() -> None:
    if _pool is not None:
        await _pool.close()


def get_pool() -> AsyncConnectionPool:
    if _pool is None:
        raise RuntimeError(
            "Connection pool not initialized -- app startup must call init_pool() first."
        )
    return _pool
