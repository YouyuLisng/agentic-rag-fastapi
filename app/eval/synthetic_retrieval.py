"""Two-stage retrieval against the synthetic benchmark table. Mirrors
app/rag/retrieval.py's vector->rerank flow but targets
match_synthetic_policy_chunks, and -- crucially -- reuses the exact same
rerank_documents() the production path uses, so the benchmark measures the
real reranker behavior, not a reimplementation of it."""

from app.config import get_settings
from app.db import get_pool
from app.rag.embeddings import embed_query
from app.rag.rerank import rerank_documents


async def _vector_search(query_embedding: list[float], match_count: int) -> list[dict]:
    pool = get_pool()
    async with pool.connection() as conn, conn.cursor() as cur:
        await cur.execute(
            "select document_slug, cluster, title, content, similarity "
            "from match_synthetic_policy_chunks(%s::vector, %s)",
            (query_embedding, match_count),
        )
        rows = await cur.fetchall()
        assert cur.description is not None
        columns = [desc[0] for desc in cur.description]
    return [dict(zip(columns, row, strict=True)) for row in rows]


async def search_synthetic(query: str, match_count: int = 5, rerank: bool | None = None) -> list[dict]:
    settings = get_settings()
    use_rerank = settings.rerank_enabled if rerank is None else rerank

    query_embedding = await embed_query(query)

    if not use_rerank:
        return await _vector_search(query_embedding, match_count)

    candidate_count = max(settings.rerank_candidate_count, match_count)
    candidates = await _vector_search(query_embedding, candidate_count)
    if not candidates:
        return []

    ranked = await rerank_documents(query, [c["content"] for c in candidates], top_k=match_count)
    results: list[dict] = []
    for index, score in ranked:
        hit = dict(candidates[index])
        hit["rerank_score"] = score
        results.append(hit)
    return results
