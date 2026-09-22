"""Ragas-style LLM-as-a-Judge scoring for the generation stage.

Faithfulness: decompose the answer into atomic claims, then judge each
claim against the retrieved context -- score is the fraction supported.
Answer Relevancy: have the judge reverse-generate questions the answer
would suit, then embed and compare them to the original question --
score is the average cosine similarity. An answer that dodges the
question entirely (noncommittal) scores 0 rather than being embedded.

Both steps use the fast model as judge (cheap, and judging is an easier
task than the generation it's grading) -- combining claim-extraction and
verification into one call each, rather than Ragas' separate calls per
step, trades a little precision for roughly half the judge-side cost.
"""

import asyncio
import math
from typing import Any

from pydantic import BaseModel

from app.llm_json import extract_json
from app.rag.embeddings import embed_query

# Typed as Any rather than AsyncAnthropic: AsyncMessages.create is
# @overload-heavy (streaming vs. non-streaming variants), which doesn't
# structurally match a narrower Protocol -- and only .messages.create()
# is actually used here, so tests pass a lightweight fake instead of a
# real client.
_JudgeClient = Any

FAITHFULNESS_PROMPT = """你是一個嚴謹的事實查核員,任務是檢查一段回答是否忠實於提供的參考資料。

步驟:
1. 把「回答」拆解成獨立的事實陳述(claims),每個陳述只包含一個可查核的事實或主張。
   問候語、反問、免責聲明等非事實內容不算陳述。
2. 針對每個陳述,判斷它能不能從「參考資料」中得到支持。

只輸出以下格式的 JSON,不要有任何其他文字或說明:
{{"claims": [{{"claim": "...", "supported": true}}, ...]}}

若回答中沒有任何可查核的事實陳述,回傳 {{"claims": []}}。

參考資料:
{context}

回答:
{answer}"""

ANSWER_RELEVANCY_PROMPT = """你是評估助理。根據下面這段「回答」,反推出 3 個這個回答最適合拿來回答的問題,
問題之間語意要有差異,分別涵蓋回答內容的不同重點,並使用繁體中文。

如果這段回答迴避、答非所問、或沒有實際回應任何具體問題(例如反問使用者、純粹問候語、要求更多資訊),
把 noncommittal 設為 true,questions 留空陣列即可。

只輸出以下格式的 JSON,不要有任何其他文字或說明:
{{"questions": ["...", "...", "..."], "noncommittal": false}}

回答:
{answer}"""


class ClaimVerdict(BaseModel):
    claim: str
    supported: bool


class FaithfulnessJudgment(BaseModel):
    claims: list[ClaimVerdict]


class RelevancyJudgment(BaseModel):
    questions: list[str]
    noncommittal: bool


def _cosine_similarity(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b, strict=True))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(y * y for y in b))
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot / (norm_a * norm_b)


async def score_faithfulness(
    client: _JudgeClient, judge_model: str, answer: str, context: list[str]
) -> tuple[float, list[ClaimVerdict]]:
    context_text = "\n\n---\n\n".join(context) if context else "(這次對話沒有檢索到任何資料)"
    response = await client.messages.create(
        model=judge_model,
        max_tokens=2048,
        messages=[{"role": "user", "content": FAITHFULNESS_PROMPT.format(context=context_text, answer=answer)}],
    )
    raw = "".join(b.text for b in response.content if b.type == "text")
    judgment = FaithfulnessJudgment.model_validate_json(extract_json(raw))

    if not judgment.claims:
        # Nothing factual to check (e.g. a pure greeting) is vacuously
        # faithful, not unscored -- consistent with there being no way
        # for it to contradict the context.
        return 1.0, []

    score = sum(1 for c in judgment.claims if c.supported) / len(judgment.claims)
    return score, judgment.claims


async def score_answer_relevancy(
    client: _JudgeClient, judge_model: str, original_query: str, answer: str
) -> tuple[float, list[str], bool]:
    response = await client.messages.create(
        model=judge_model,
        max_tokens=1024,
        messages=[{"role": "user", "content": ANSWER_RELEVANCY_PROMPT.format(answer=answer)}],
    )
    raw = "".join(b.text for b in response.content if b.type == "text")
    judgment = RelevancyJudgment.model_validate_json(extract_json(raw))

    if judgment.noncommittal or not judgment.questions:
        return 0.0, judgment.questions, True

    original_embedding, generated_embeddings = await asyncio.gather(
        embed_query(original_query),
        asyncio.gather(*(embed_query(q) for q in judgment.questions)),
    )
    similarities = [_cosine_similarity(original_embedding, emb) for emb in generated_embeddings]
    return sum(similarities) / len(similarities), judgment.questions, False
