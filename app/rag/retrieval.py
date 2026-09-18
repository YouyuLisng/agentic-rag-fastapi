from app.db import get_pool
from app.rag.embeddings import embed_query


async def search_knowledge(query: str, match_count: int = 5) -> list[dict]:
    """Semantic search over the policy knowledge base via the
    match_policy_chunks Postgres function (cosine similarity, HNSW
    index). This is the RAG half of the agentic loop's tool set."""
    query_embedding = await embed_query(query)

    pool = get_pool()
    async with pool.connection() as conn, conn.cursor() as cur:
        await cur.execute(
            "select document_slug, title, content, similarity "
            "from match_policy_chunks(%s::vector, %s)",
            (query_embedding, match_count),
        )
        rows = await cur.fetchall()
        assert cur.description is not None
        columns = [desc[0] for desc in cur.description]

    return [dict(zip(columns, row, strict=True)) for row in rows]
