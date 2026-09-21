from app.db import get_pool
from app.rag.embeddings import embed_query


async def search_knowledge(query: str, match_count: int = 5, tags: list[str] | None = None) -> list[dict]:
    """Semantic search over the policy knowledge base via the
    match_policy_chunks Postgres function (cosine similarity, HNSW
    index). This is the RAG half of the agentic loop's tool set.

    `tags`, when given, pre-filters to chunks sharing at least one tag
    (SQL array overlap) before ranking -- a caller that already knows
    the topic bucket (from upstream context, not inferred here) can use
    it to keep a near-miss semantic match out of contention entirely,
    rather than hoping embedding similarity alone ranks it correctly.
    """
    query_embedding = await embed_query(query)

    pool = get_pool()
    async with pool.connection() as conn, conn.cursor() as cur:
        await cur.execute(
            "select document_slug, title, content, similarity "
            "from match_policy_chunks(%s::vector, %s, %s::text[])",
            (query_embedding, match_count, tags),
        )
        rows = await cur.fetchall()
        assert cur.description is not None
        columns = [desc[0] for desc in cur.description]

    return [dict(zip(columns, row, strict=True)) for row in rows]
