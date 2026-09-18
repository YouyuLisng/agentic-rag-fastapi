from collections.abc import AsyncIterator
from typing import Literal

from fastapi import APIRouter
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from app.agent.events import AgentEvent, ErrorEvent
from app.agent.langchain_loop import run_agent_langchain
from app.agent.loop import run_agent

router = APIRouter()


class ChatRequest(BaseModel):
    message: str
    impl: Literal["handrolled", "langchain"] = "handrolled"
    document_id: str | None = None  # Mode B -- set once /documents has been uploaded


def _run(
    message: str, impl: Literal["handrolled", "langchain"], document_id: str | None
) -> AsyncIterator[AgentEvent]:
    if impl == "langchain":
        return run_agent_langchain(message, document_id=document_id)
    return run_agent(message, document_id=document_id)


async def _event_stream(
    message: str, impl: Literal["handrolled", "langchain"], document_id: str | None
) -> AsyncIterator[str]:
    try:
        async for event in _run(message, impl, document_id):
            yield f"data: {event.model_dump_json()}\n\n"
    except Exception as e:
        # A raw exception here would just silently truncate the HTTP
        # response -- the frontend would see a stream that stopped
        # without a final_answer event and have nothing to show the
        # user. Surface it as a real event instead.
        yield f"data: {ErrorEvent(message=str(e)).model_dump_json()}\n\n"


@router.post("/chat")
async def chat(request: ChatRequest) -> StreamingResponse:
    return StreamingResponse(
        _event_stream(request.message, request.impl, request.document_id),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
        },
    )
