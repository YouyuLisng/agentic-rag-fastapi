from app.config import get_settings
from app.db import get_pool
from app.rag.embeddings import embed_query
from app.rag.rerank import rerank_documents


async def _vector_search(query_embedding: list[float], match_count: int, tags: list[str] | None) -> list[dict]:
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


async def search_knowledge(
    query: str, match_count: int = 5, tags: list[str] | None = None, rerank: bool | None = None
) -> list[dict]:
    """Semantic search over the policy knowledge base.

    Two-stage when reranking is on: the vector search (cosine similarity,
    HNSW index, via match_policy_chunks) pulls a wide candidate set, then
    a Voyage cross-encoder reranks it down to match_count. The returned
    dicts always carry the first-stage `similarity`; reranked results
    additionally carry `rerank_score`. With reranking off it's a plain
    single-stage top-match_count vector search -- same shape, no
    rerank_score.

    `tags`, when given, pre-filters to chunks sharing at least one tag
    (SQL array overlap) before ranking -- a caller that already knows the
    topic bucket can keep a near-miss semantic match out of contention
    entirely, rather than hoping embedding similarity alone ranks it
    correctly. The pre-filter runs first either way, so it composes with
    reranking (filter narrows the candidate pool, rerank reorders it).

    `rerank` overrides the settings default per call (None = use
    settings.rerank_enabled) -- the eval harness uses this to measure the
    same queries with and without reranking in one run.
    """
    settings = get_settings()
    use_rerank = settings.rerank_enabled if rerank is None else rerank

    query_embedding = await embed_query(query)

    if not use_rerank:
        return await _vector_search(query_embedding, match_count, tags)

    # Retrieve wide, then rerank down. Never fetch fewer candidates than
    # the caller asked to keep, even if candidate_count is misconfigured
    # below match_count.
    candidate_count = max(settings.rerank_candidate_count, match_count)
    candidates = await _vector_search(query_embedding, candidate_count, tags)
    if not candidates:
        return []

    ranked = await rerank_documents(query, [c["content"] for c in candidates], top_k=match_count)

    results: list[dict] = []
    for index, score in ranked:
        hit = dict(candidates[index])
        hit["rerank_score"] = score
        results.append(hit)
    return results
