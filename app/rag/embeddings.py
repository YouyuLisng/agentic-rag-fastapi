import asyncio

from voyageai.client import Client

from app.config import get_settings

_client: Client | None = None


def _get_client() -> Client:
    global _client
    if _client is None:
        _client = Client(api_key=get_settings().voyage_api_key)
    return _client


def _embed_documents_sync(texts: list[str]) -> list[list[float]]:
    settings = get_settings()
    result = _get_client().embed(
        texts, model=settings.voyage_embedding_model, input_type="document"
    )
    return [[float(x) for x in vec] for vec in result.embeddings]


def _embed_query_sync(text: str) -> list[float]:
    settings = get_settings()
    result = _get_client().embed(
        [text], model=settings.voyage_embedding_model, input_type="query"
    )
    return [float(x) for x in result.embeddings[0]]


async def embed_documents(texts: list[str]) -> list[list[float]]:
    """Embed chunks for storage. Asymmetric with embed_query -- Voyage's
    input_type="document" vs "query" improves retrieval accuracy over
    embedding both sides the same way.

    Voyage's SDK is synchronous (blocking HTTP call) -- run it in a
    thread so it doesn't stall the event loop the connection pool's
    background tasks also depend on."""
    return await asyncio.to_thread(_embed_documents_sync, texts)


async def embed_query(text: str) -> list[float]:
    return await asyncio.to_thread(_embed_query_sync, text)
