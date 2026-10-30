import asyncio

from app.config import get_settings
from app.rag.embeddings import _get_client


def _rerank_sync(query: str, documents: list[str], top_k: int) -> list[tuple[int, float]]:
    settings = get_settings()
    result = _get_client().rerank(
        query=query,
        documents=documents,
        model=settings.voyage_rerank_model,
        top_k=top_k,
    )
    # RerankingResult.index points back into the input `documents` list;
    # results come back sorted by relevance_score descending.
    return [(r.index, float(r.relevance_score)) for r in result.results]


async def rerank_documents(query: str, documents: list[str], top_k: int) -> list[tuple[int, float]]:
    """Second-stage reranking via Voyage's cross-encoder. Returns
    (original_index, relevance_score) pairs, best first -- the caller
    maps original_index back onto the candidate list it passed in.

    Reuses embeddings.py's Voyage client (same key, same singleton) and
    runs the blocking SDK call in a thread, same as embed_query/
    embed_documents, so it doesn't stall the event loop."""
    if not documents:
        return []
    return await asyncio.to_thread(_rerank_sync, query, documents, top_k)
