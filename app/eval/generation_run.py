import asyncio
from collections.abc import Sequence

from anthropic import AsyncAnthropic
from pydantic import BaseModel

from app.agent.events import AgentEvent, FinalAnswerEvent, MaxTurnsExceededEvent, RefusalEvent, ToolResultEvent
from app.agent.loop import run_agent
from app.config import get_settings
from app.eval.generation_dataset import GENERATION_EVAL_QUERIES
from app.eval.judge import ClaimVerdict, cosine_similarity, score_answer_relevancy, score_faithfulness
from app.rag.embeddings import embed_query


class GenerationEvalCase(BaseModel):
    query: str
    answer: str
    context: list[str]  # every non-error tool_result this run produced
    faithfulness: float
    faithfulness_claims: list[ClaimVerdict]
    answer_relevancy: float
    relevancy_questions: list[str]
    is_noncommittal: bool


class GenerationEvalMetrics(BaseModel):
    n: int
    avg_faithfulness: float
    avg_answer_relevancy: float
    # Raw cosine similarity between short Chinese sentences is
    # compressed well above 0 even for genuinely unrelated pairs (an
    # embedding-space anisotropy artifact, not a bug) -- so
    # avg_answer_relevancy alone reads as "low" even when the method is
    # working correctly. This baseline re-scores each case's reverse-
    # generated questions against a deliberately *wrong* original query
    # (circular-shifted by one) to show what "unrelated" actually scores
    # on this embedding model, so the real number can be read relative
    # to it instead of against an uninformative 0..1 scale.
    avg_answer_relevancy_baseline: float


class GenerationEvalReport(BaseModel):
    metrics: GenerationEvalMetrics
    cases: list[GenerationEvalCase]


def _collect_answer_and_context(events: Sequence[AgentEvent]) -> tuple[str, list[str]]:
    context = [e.result for e in events if isinstance(e, ToolResultEvent) and not e.is_error]

    for e in events:
        if isinstance(e, FinalAnswerEvent):
            return e.text, context
        if isinstance(e, RefusalEvent):
            return e.explanation or "(拒絕回答)", context
        if isinstance(e, MaxTurnsExceededEvent):
            return e.text, context

    return "(沒有產生回答)", context


async def _run_case(client: AsyncAnthropic, judge_model: str, query: str) -> GenerationEvalCase:
    events = [event async for event in run_agent(query)]
    answer, context = _collect_answer_and_context(events)

    faithfulness, claims = await score_faithfulness(client, judge_model, answer, context)
    relevancy, questions, noncommittal = await score_answer_relevancy(client, judge_model, query, answer)

    return GenerationEvalCase(
        query=query,
        answer=answer,
        context=context,
        faithfulness=faithfulness,
        faithfulness_claims=claims,
        answer_relevancy=relevancy,
        relevancy_questions=questions,
        is_noncommittal=noncommittal,
    )


async def _compute_relevancy_baseline(cases: list[GenerationEvalCase]) -> float:
    """Average cosine similarity between each case's reverse-generated
    questions and a *mismatched* original query (every case shifted to
    pair with its neighbor's query instead of its own) -- an "obviously
    unrelated" floor to compare avg_answer_relevancy against, since raw
    cosine similarity between short Chinese sentences doesn't approach 0
    just because the sentences are unrelated.
    """
    scorable = [c for c in cases if c.relevancy_questions]
    if len(scorable) < 2:
        return 0.0

    mismatched_queries = [c.query for c in scorable[1:]] + [scorable[0].query]

    mismatched_embeddings, generated_embeddings = await asyncio.gather(
        asyncio.gather(*(embed_query(q) for q in mismatched_queries)),
        asyncio.gather(*(asyncio.gather(*(embed_query(q) for q in c.relevancy_questions)) for c in scorable)),
    )

    similarities = [
        cosine_similarity(mismatched_embedding, generated_embedding)
        for mismatched_embedding, case_embeddings in zip(mismatched_embeddings, generated_embeddings, strict=True)
        for generated_embedding in case_embeddings
    ]
    return sum(similarities) / len(similarities)


async def run_generation_eval() -> GenerationEvalReport:
    settings = get_settings()
    client = AsyncAnthropic(api_key=settings.anthropic_api_key)
    judge_model = settings.claude_model_fast

    # Sequential, not gathered -- this already fires ~4 LLM calls per
    # case, and 24 cases running concurrently would spike straight into
    # Anthropic's own per-minute rate limits for no real benefit here.
    cases = [await _run_case(client, judge_model, query) for query in GENERATION_EVAL_QUERIES]
    baseline = await _compute_relevancy_baseline(cases)

    n = len(cases)
    metrics = GenerationEvalMetrics(
        n=n,
        avg_faithfulness=sum(c.faithfulness for c in cases) / n,
        avg_answer_relevancy=sum(c.answer_relevancy for c in cases) / n,
        avg_answer_relevancy_baseline=baseline,
    )
    return GenerationEvalReport(metrics=metrics, cases=cases)
