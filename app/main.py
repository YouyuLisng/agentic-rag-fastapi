from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from typing import cast

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from slowapi import _rate_limit_exceeded_handler
from slowapi.errors import RateLimitExceeded
from starlette.types import ExceptionHandler

from app.config import get_settings
from app.db import close_pool, get_pool, init_pool
from app.rate_limit import limiter
from app.routers.chat import router as chat_router
from app.routers.data import router as data_router
from app.routers.documents import router as documents_router
from app.routers.eval import router as eval_router


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    await init_pool()
    yield
    await close_pool()


app = FastAPI(title="Agentic RAG Travel Assistant", lifespan=lifespan)

app.state.limiter = limiter
# slowapi's handler is typed against its own narrower signature, not
# Starlette's generic ExceptionHandler -- the mismatch is a typing-only
# issue (the runtime signature is compatible), so cast rather than wrap.
app.add_exception_handler(RateLimitExceeded, cast(ExceptionHandler, _rate_limit_exceeded_handler))

# The frontend runs as a separate origin -- CORS_ORIGINS (comma-
# separated) controls what's allowed, so the deployed frontend's real
# origin can be added via an env var without a code change/redeploy.
app.add_middleware(
    CORSMiddleware,
    allow_origins=get_settings().cors_origins_list,
    allow_methods=["*"],
    allow_headers=["*"],
)


app.include_router(chat_router)
app.include_router(documents_router)
app.include_router(data_router)
app.include_router(eval_router)


@app.get("/health")
async def health() -> dict[str, str]:
    pool = get_pool()
    async with pool.connection() as conn:
        await conn.execute("SELECT 1")
    return {"status": "ok"}
