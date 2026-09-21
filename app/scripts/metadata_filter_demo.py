"""Demonstrates why policy_chunks.tags exists: a real near-miss from
the retrieval eval set (app/eval/dataset.py), fixed with a metadata
pre-filter rather than by touching the embeddings at all.

"因為疫情取消行程,費用會退嗎?" (expected doc: force-majeure) ranks
`cancellation` first on pure vector search, because both documents
genuinely discuss refunds and "取消行程,費用會退" reads very close to
the general cancellation policy's own wording. No amount of chunking or
prompt tuning fixes a ranking problem like this -- but if a caller
already knows (from upstream context, not inferred here) that this is
a force-majeure-flavored question, filtering to tags=['不可抗力'] keeps
the near-miss out of contention entirely.

Run: `uv run python -m app.scripts.metadata_filter_demo`
"""

import asyncio

from dotenv import load_dotenv

load_dotenv()

from app.db import close_pool, init_pool  # noqa: E402
from app.rag.retrieval import search_knowledge  # noqa: E402

QUERY = "因為疫情取消行程,費用會退嗎?"
EXPECTED_SLUG = "force-majeure"


def _rank_of(results: list[dict], slug: str) -> int | None:
    slugs = [r["document_slug"] for r in results]
    return slugs.index(slug) + 1 if slug in slugs else None


async def main() -> None:
    await init_pool()
    try:
        unfiltered = await search_knowledge(QUERY, match_count=5)
        filtered = await search_knowledge(QUERY, match_count=5, tags=["不可抗力"])

        print(f"查詢:{QUERY!r}(預期文件:{EXPECTED_SLUG})\n")

        print("不加 tags 過濾(純向量檢索):")
        for i, r in enumerate(unfiltered, start=1):
            print(f"  {i}. {r['document_slug']:<15} similarity={r['similarity']:.3f}")
        rank_before = _rank_of(unfiltered, EXPECTED_SLUG)
        print(f"  -> force-majeure 排名: {rank_before}\n")

        print("加 tags=['不可抗力'] 過濾:")
        for i, r in enumerate(filtered, start=1):
            print(f"  {i}. {r['document_slug']:<15} similarity={r['similarity']:.3f}")
        rank_after = _rank_of(filtered, EXPECTED_SLUG)
        print(f"  -> force-majeure 排名: {rank_after}\n")

        if rank_after == 1 and (rank_before is None or rank_before > 1):
            print(f"[PASS] metadata 過濾把 force-majeure 從第 {rank_before} 名修正到第 1 名")
        else:
            print(f"[FAIL] 過濾前後排名分別是 {rank_before} / {rank_after},未達預期效果")
    finally:
        await close_pool()


if __name__ == "__main__":
    asyncio.run(main())
