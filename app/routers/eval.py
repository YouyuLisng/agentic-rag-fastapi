from fastapi import APIRouter, Request

from app.config import get_settings
from app.eval.generation_run import GenerationEvalReport, run_generation_eval
from app.eval.run import EvalReport, run_retrieval_eval
from app.eval.tool_selection_run import ToolSelectionReport, run_tool_selection_eval
from app.rate_limit import limiter

router = APIRouter()


@router.get("/eval/retrieval")
@limiter.limit(get_settings().rate_limit_eval)
async def get_retrieval_eval(request: Request) -> EvalReport:
    """Runs the RAG retrieval eval live (real embeddings, real pgvector
    query) rather than serving a cached/precomputed result -- so what a
    viewer sees always reflects the current knowledge base, not a
    snapshot from whenever this was last run manually."""
    return await run_retrieval_eval()


@router.get("/eval/generation")
@limiter.limit(get_settings().rate_limit_eval_generation)
async def get_generation_eval(request: Request) -> GenerationEvalReport:
    """Ragas-style LLM-as-a-Judge scoring of the generation stage
    (Faithfulness, Answer Relevancy) -- runs the real hand-rolled agent
    loop per case, not a canned answer, so this is far more expensive
    per hit than /eval/retrieval and is capped accordingly."""
    return await run_generation_eval()


@router.get("/eval/tool-selection")
@limiter.limit(get_settings().rate_limit_eval_generation)
async def get_tool_selection_eval(request: Request) -> ToolSelectionReport:
    """The third scoring layer alongside retrieval and generation: did
    the agent call the *right* tool(s), not just produce a plausible
    answer -- catches routing mistakes (wrong tool, or no tool when one
    was needed) that Faithfulness/Answer Relevancy can't see. Runs the
    real hand-rolled agent loop per case, same cost profile as
    /eval/generation."""
    return await run_tool_selection_eval()
