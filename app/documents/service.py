import uuid
from typing import Any

from app.db import get_pool
from app.documents.extract import extract_text
from app.rag.chunking import chunk_markdown
from app.rag.embeddings import embed_documents, embed_query


async def upload_document(filename: str, content: bytes) -> dict[str, Any]:
    """Extract -> chunk -> embed -> store, scoped under a fresh
    document_id. Reuses the same chunker/embedder as the policy
    knowledge base -- the pipeline shape doesn't change, only the
    source and the table it lands in."""
    text = await extract_text(filename, content)
    chunks = chunk_markdown(text)

    if not chunks:
        raise ValueError("No extractable text content found in this document.")

    embeddings = await embed_documents(chunks)
    document_id = uuid.uuid4()

    pool = get_pool()
    async with pool.connection() as conn, conn.cursor() as cur:
        for idx, (chunk, embedding) in enumerate(zip(chunks, embeddings, strict=True)):
            await cur.execute(
                "insert into document_chunks (document_id, filename, chunk_index, content, embedding) "
                "values (%s, %s, %s, %s, %s)",
                (document_id, filename, idx, chunk, embedding),
            )
        await conn.commit()

    return {"document_id": str(document_id), "filename": filename, "chunk_count": len(chunks)}


async def search_document(document_id: str, query: str, match_count: int = 5) -> list[dict[str, Any]]:
    query_embedding = await embed_query(query)

    pool = get_pool()
    async with pool.connection() as conn, conn.cursor() as cur:
        await cur.execute(
            "select id, chunk_index, content, similarity "
            "from match_document_chunks(%s::uuid, %s::vector, %s)",
            (document_id, query_embedding, match_count),
        )
        rows = await cur.fetchall()
        assert cur.description is not None
        columns = [desc[0] for desc in cur.description]

    return [dict(zip(columns, row, strict=True)) for row in rows]
