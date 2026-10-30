"""CI gate for the retrieval eval layer -- unlike app/scripts/run_eval.py
(human-readable report, always exits 0), this asserts the metrics
haven't regressed past a floor and exits non-zero if they have, so a
broken retrieval query/embedding/schema change fails the build instead
of shipping silently. Floors sit well below the documented baseline
(93.8% / 100% / 100% / 0.969 MRR on the 16-case eval set, see README)
so normal fluctuation doesn't false-positive -- only a real regression
should trip this.

Retrieval is the one eval layer cheap and deterministic enough (pure
embedding calls, no agent turns) to gate every push/PR on -- see
app/scripts/ci_full_eval_report.py for why generation/tool-selection
stay report-only. Run: `uv run python -m app.scripts.ci_retrieval_eval`
"""

import asyncio
import os
import sys

from dotenv import load_dotenv

load_dotenv()

from app.db import close_pool, init_pool  # noqa: E402
from app.eval.run import run_retrieval_eval  # noqa: E402

ACCURACY_AT_1_FLOOR = 0.80
ACCURACY_AT_3_FLOOR = 0.90
MRR_FLOOR = 0.85


async def main() -> int:
    await init_pool()
    try:
        report = await run_retrieval_eval()
    finally:
        await close_pool()

    m = report.metrics
    table = (
        "| Metric | Value | Floor |\n"
        "|---|---|---|\n"
        f"| Accuracy@1 | {m.accuracy_at_1:.1%} | {ACCURACY_AT_1_FLOOR:.0%} |\n"
        f"| Accuracy@3 | {m.accuracy_at_3:.1%} | {ACCURACY_AT_3_FLOOR:.0%} |\n"
        f"| Accuracy@5 | {m.accuracy_at_5:.1%} | - |\n"
        f"| MRR | {m.mrr:.3f} | {MRR_FLOOR:.2f} |\n"
        f"| n | {m.n} | - |\n"
    )
    print(table)

    summary_path = os.environ.get("GITHUB_STEP_SUMMARY")
    if summary_path:
        with open(summary_path, "a") as f:
            f.write("## Retrieval eval\n\n" + table + "\n")

    failures = []
    if m.accuracy_at_1 < ACCURACY_AT_1_FLOOR:
        failures.append(f"Accuracy@1 {m.accuracy_at_1:.1%} < floor {ACCURACY_AT_1_FLOOR:.0%}")
    if m.accuracy_at_3 < ACCURACY_AT_3_FLOOR:
        failures.append(f"Accuracy@3 {m.accuracy_at_3:.1%} < floor {ACCURACY_AT_3_FLOOR:.0%}")
    if m.mrr < MRR_FLOOR:
        failures.append(f"MRR {m.mrr:.3f} < floor {MRR_FLOOR:.2f}")

    if failures:
        print("FAILED:")
        for failure in failures:
            print(f" - {failure}")
        return 1

    print("OK -- retrieval eval within floor")
    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
