"""Runs the RAG retrieval eval against the real Voyage/pgvector
pipeline and prints a report. Run: `uv run python -m app.scripts.run_eval`
"""

import asyncio

from dotenv import load_dotenv

load_dotenv()

from app.db import close_pool, init_pool  # noqa: E402
from app.eval.run import run_retrieval_eval  # noqa: E402


async def main() -> None:
    await init_pool()
    try:
        report = await run_retrieval_eval()
    finally:
        await close_pool()

    print(f"{'':2}{'查詢':<30}{'預期文件':<16}{'命中排名':<10}{'Top similarity':<15}")
    print("-" * 75)
    for case in report.cases:
        marker = "✓" if case.rank == 1 else ("△" if case.rank else "✗")
        rank_str = str(case.rank) if case.rank else "未命中"
        print(f"{marker} {case.query[:26]:<28}{case.expected_slug:<16}{rank_str:<10}{case.top_similarity:.3f}")

    m = report.metrics
    print()
    print(f"樣本數: {m.n}")
    print(f"Accuracy@1: {m.accuracy_at_1:.1%}")
    print(f"Accuracy@3: {m.accuracy_at_3:.1%}")
    print(f"Accuracy@5: {m.accuracy_at_5:.1%}")
    print(f"MRR: {m.mrr:.3f}")


if __name__ == "__main__":
    asyncio.run(main())
