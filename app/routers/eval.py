from fastapi import APIRouter, Request

from app.config import get_settings
from app.eval.run import EvalReport, run_retrieval_eval
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
