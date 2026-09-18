from collections.abc import AsyncIterator

from fastapi import APIRouter
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from app.agent.events import ErrorEvent
from app.agent.loop import run_agent

router = APIRouter()


class ChatRequest(BaseModel):
    message: str


async def _event_stream(message: str) -> AsyncIterator[str]:
    try:
        async for event in run_agent(message):
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
        _event_stream(request.message),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
        },
    )
