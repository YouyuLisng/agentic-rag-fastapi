"""Read-only endpoints exposing the actual seed data -- so a viewer
(an interviewer, say) can independently check whether the agent's
answers actually match what's really in the database, rather than
having to trust them."""

from typing import Any

from fastapi import APIRouter
from pydantic import BaseModel

from app.db import get_pool

router = APIRouter()


class TourOut(BaseModel):
    id: str
    title: str
    country: str
    location: str
    days: int
    budget_twd: int
    suitable_for: list[str]
    summary: str
    itinerary: list[dict[str, Any]]
    capacity: int
    enrolled_count: int


class PolicyDocumentOut(BaseModel):
    document_slug: str
    title: str
    content: str
    chunk_count: int


@router.get("/tours")
async def list_tours() -> list[TourOut]:
    pool = get_pool()
    async with pool.connection() as conn, conn.cursor() as cur:
        await cur.execute(
            "select id, title, country, location, days, budget_twd, suitable_for, "
            "summary, itinerary, capacity, enrolled_count from tours order by title"
        )
        rows = await cur.fetchall()
        assert cur.description is not None
        columns = [desc[0] for desc in cur.description]

    tours = []
    for row in rows:
        data = dict(zip(columns, row, strict=True))
        data["id"] = str(data["id"])  # psycopg returns a uuid.UUID, not a str
        tours.append(TourOut(**data))
    return tours


@router.get("/policies")
async def list_policies() -> list[PolicyDocumentOut]:
    pool = get_pool()
    async with pool.connection() as conn, conn.cursor() as cur:
        await cur.execute(
            "select document_slug, title, chunk_index, content from policy_chunks "
            "order by document_slug, chunk_index"
        )
        rows = await cur.fetchall()

    # Reassemble each document from its chunks in order -- a viewer
    # cares about what the policy actually says, not where RAG happened
    # to draw chunk boundaries.
    documents: dict[str, PolicyDocumentOut] = {}
    for slug, title, _chunk_index, content in rows:
        if slug not in documents:
            documents[slug] = PolicyDocumentOut(document_slug=slug, title=title, content=content, chunk_count=1)
        else:
            existing = documents[slug]
            documents[slug] = existing.model_copy(
                update={"content": f"{existing.content}\n\n{content}", "chunk_count": existing.chunk_count + 1}
            )

    return sorted(documents.values(), key=lambda d: d.document_slug)
