from psycopg_pool import AsyncConnectionPool

from app.config import get_settings

_pool: AsyncConnectionPool | None = None


async def init_pool() -> AsyncConnectionPool:
    global _pool
    settings = get_settings()
    _pool = AsyncConnectionPool(conninfo=settings.database_url, open=False)
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
