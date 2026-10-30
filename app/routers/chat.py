from collections.abc import AsyncIterator
from typing import Literal

from fastapi import APIRouter, Request
from fastapi.responses import StreamingResponse
from langfuse import get_client
from pydantic import BaseModel

from app.agent.events import AgentEvent, ErrorEvent, FinalAnswerEvent
from app.agent.history import HistoryMessage
from app.agent.langchain_loop import run_agent_langchain
from app.agent.loop import run_agent
from app.config import get_settings
from app.rate_limit import limiter

router = APIRouter()


class ChatRequest(BaseModel):
    message: str
    impl: Literal["handrolled", "langchain"] = "handrolled"
    document_id: str | None = None  # Mode B -- set once /documents has been uploaded
    # Prior turns' final text, replayed by the client -- this backend
    # keeps no server-side session, so conversational continuity only
    # exists if the caller resends it. See app.agent.history.
    history: list[HistoryMessage] = []


def _run(
    message: str, impl: Literal["handrolled", "langchain"], document_id: str | None, history: list[HistoryMessage]
) -> AsyncIterator[AgentEvent]:
    if impl == "langchain":
        return run_agent_langchain(message, document_id=document_id, history=history)
    return run_agent(message, document_id=document_id, history=history)


async def _event_stream(
    message: str, impl: Literal["handrolled", "langchain"], document_id: str | None, history: list[HistoryMessage]
) -> AsyncIterator[str]:
    langfuse = get_client()
    # Parent span for this request -- every Claude call made while this
    # `with` block is active (both the hand-rolled loop's manual
    # generations and the LangChain loop's CallbackHandler spans) nests
    # under it via OTEL context propagation, so one /chat request shows
    # up as one trace regardless of which impl handled it.
    with langfuse.start_as_current_observation(
        as_type="span",
        name="chat",
        input={"message": message, "impl": impl, "document_id": document_id},
    ) as span:
        final_text = None
        try:
            async for event in _run(message, impl, document_id, history):
                if isinstance(event, FinalAnswerEvent):
                    final_text = event.text
                yield f"data: {event.model_dump_json()}\n\n"
            span.update(output=final_text)
        except Exception as e:
            # A raw exception here would just silently truncate the HTTP
            # response -- the frontend would see a stream that stopped
            # without a final_answer event and have nothing to show the
            # user. Surface it as a real event instead.
            span.update(output=str(e), level="ERROR")
            yield f"data: {ErrorEvent(message=str(e)).model_dump_json()}\n\n"


@router.post("/chat")
@limiter.limit(get_settings().rate_limit_chat)
async def chat(request: Request, body: ChatRequest) -> StreamingResponse:
    return StreamingResponse(
        _event_stream(body.message, body.impl, body.document_id, body.history),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
        },
    )
