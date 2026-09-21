"""Seeds `tours` (structured, unembedded) and `policy_chunks` (chunked +
embedded) from data/tours.json and data/policies/*.md.

Idempotent: clears each table before reinserting, so reruns don't pile
up duplicates. Run directly: `uv run python -m app.scripts.seed`.
"""

import asyncio
import json
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

from app.db import close_pool, get_pool, init_pool  # noqa: E402
from app.rag.chunking import chunk_markdown  # noqa: E402
from app.rag.embeddings import embed_documents  # noqa: E402

DATA_DIR = Path(__file__).resolve().parent.parent.parent / "data"


async def seed_tours(conn) -> None:
    tours = json.loads((DATA_DIR / "tours.json").read_text(encoding="utf-8"))

    async with conn.cursor() as cur:
        await cur.execute("delete from tours")
        for tour in tours:
            await cur.execute(
                """
                insert into tours
                    (title, country, location, days, budget_twd, suitable_for, summary,
                     itinerary, capacity, enrolled_count, cost_price_twd)
                values (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                """,
                (
                    tour["title"],
                    tour["country"],
                    tour["location"],
                    tour["days"],
                    tour["budget_twd"],
                    tour["suitable_for"],
                    tour["summary"],
                    json.dumps(tour["itinerary"]),
                    tour["capacity"],
                    tour["enrolled_count"],
                    tour["cost_price_twd"],
                ),
            )
    await conn.commit()
    print(f"Seeded {len(tours)} tours.")


# Curated per-document topic tags -- deliberately allowed to overlap
# (force-majeure is tagged 退款 too, since it genuinely discusses refund
# handling) rather than being a 1:1 relabeling of document_slug. See
# db/schema.sql's comment on policy_chunks.tags for why this exists.
POLICY_TAGS: dict[str, list[str]] = {
    "cancellation": ["退訂", "退款"],
    "force-majeure": ["不可抗力", "退款", "天災", "疫情"],
    "insurance": ["保險", "理賠"],
    "payment": ["付款", "訂金"],
    "preparation": ["行前準備", "證件"],
    "special-needs": ["特殊需求", "無障礙"],
    "visa": ["簽證", "證件"],
    "packing": ["打包", "穿著"],
}


async def seed_policies(conn) -> None:
    md_files = sorted((DATA_DIR / "policies").glob("*.md"))

    records: list[tuple[str, str, str, int]] = []  # slug, title, content, chunk_index
    for path in md_files:
        text = path.read_text(encoding="utf-8").strip()
        lines = text.splitlines()
        if lines and lines[0].startswith("#"):
            title = lines[0].lstrip("#").strip()
            body = "\n".join(lines[1:]).strip()
        else:
            title = path.stem
            body = text

        for i, chunk in enumerate(chunk_markdown(body)):
            records.append((path.stem, title, chunk, i))

    embeddings = await embed_documents([r[2] for r in records])

    async with conn.cursor() as cur:
        await cur.execute("delete from policy_chunks")
        for (slug, title, content, idx), embedding in zip(records, embeddings, strict=True):
            await cur.execute(
                """
                insert into policy_chunks (document_slug, title, content, chunk_index, embedding, tags)
                values (%s, %s, %s, %s, %s, %s)
                """,
                (slug, title, content, idx, embedding, POLICY_TAGS.get(slug, [])),
            )
    await conn.commit()
    print(f"Seeded {len(records)} policy chunks from {len(md_files)} documents.")


async def main() -> None:
    await init_pool()
    pool = get_pool()
    try:
        async with pool.connection() as conn:
            await seed_tours(conn)
            await seed_policies(conn)
    finally:
        await close_pool()


if __name__ == "__main__":
    asyncio.run(main())
