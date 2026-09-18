from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.db import close_pool, get_pool, init_pool
from app.routers.chat import router as chat_router
from app.routers.data import router as data_router
from app.routers.documents import router as documents_router


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    await init_pool()
    yield
    await close_pool()


app = FastAPI(title="Agentic RAG Travel Assistant", lifespan=lifespan)

# The frontend (Next.js) runs as a separate origin during local dev;
# tighten this to the deployed frontend's real origin before shipping.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://localhost:3001"],
    allow_methods=["*"],
    allow_headers=["*"],
)


app.include_router(chat_router)
app.include_router(documents_router)
app.include_router(data_router)


@app.get("/health")
async def health() -> dict[str, str]:
    pool = get_pool()
    async with pool.connection() as conn:
        await conn.execute("SELECT 1")
    return {"status": "ok"}
