from fastapi import APIRouter

from app.eval.run import EvalReport, run_retrieval_eval

router = APIRouter()


@router.get("/eval/retrieval")
async def get_retrieval_eval() -> EvalReport:
    """Runs the RAG retrieval eval live (real embeddings, real pgvector
    query) rather than serving a cached/precomputed result -- so what a
    viewer sees always reflects the current knowledge base, not a
    snapshot from whenever this was last run manually."""
    return await run_retrieval_eval()
