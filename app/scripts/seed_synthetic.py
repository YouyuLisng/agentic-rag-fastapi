"""Seeds the synthetic benchmark corpus (app/eval/synthetic_corpus.py)
into synthetic_policy_chunks -- a separate table from the real knowledge
base, never served by search_knowledge. Embeds each doc with Voyage
(cheap: ~160 document embeddings, no agent turns, no Claude calls).

Idempotent: clears the synthetic table before reinserting. Run:
    uv run python -m app.scripts.seed_synthetic
"""

import asyncio

from dotenv import load_dotenv

load_dotenv()

from app.db import close_pool, get_pool, init_pool  # noqa: E402
from app.eval.synthetic_corpus import generate_corpus  # noqa: E402
from app.rag.embeddings import embed_documents  # noqa: E402


async def main() -> None:
    docs = generate_corpus()
    embeddings = await embed_documents([d.content for d in docs])

    await init_pool()
    pool = get_pool()
    try:
        async with pool.connection() as conn:
            async with conn.cursor() as cur:
                await cur.execute("delete from synthetic_policy_chunks")
                for doc, embedding in zip(docs, embeddings, strict=True):
                    await cur.execute(
                        """
                        insert into synthetic_policy_chunks
                            (document_slug, cluster, title, content, embedding)
                        values (%s, %s, %s, %s, %s)
                        """,
                        (doc.slug, doc.cluster, doc.title, doc.content, embedding),
                    )
            await conn.commit()
    finally:
        await close_pool()

    clusters = {d.cluster for d in docs}
    print(f"Seeded {len(docs)} synthetic docs across {len(clusters)} clusters: {', '.join(sorted(clusters))}")


if __name__ == "__main__":
    asyncio.run(main())
