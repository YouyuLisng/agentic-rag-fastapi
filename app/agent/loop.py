from collections.abc import AsyncIterator
from typing import Any, cast

from anthropic import AsyncAnthropic
from anthropic.types import MessageParam, ToolResultBlockParam

from app.agent.events import (
    AgentEvent,
    FinalAnswerEvent,
    MaxTurnsExceededEvent,
    RefusalEvent,
    ToolCallEvent,
    ToolResultEvent,
)
from app.agent.tools import TOOLS, execute_tool
from app.config import get_settings

SYSTEM_PROMPT = """你是一個旅行社的 AI 助理,使用繁體中文協助使用者了解旅遊政策與尋找合適的行程。

你有兩種查詢能力,依問題性質選擇:
- 政策/規則類問題(退訂、保險、簽證、打包建議等)-- 用 search_knowledge 語意檢索知識庫。
- 尋找/篩選行程(依地區、預算、天數、適合對象)-- 用 search_tours 查行程列表;
  需要某個行程的完整每日行程時,再用 get_tour_detail(需要先從 search_tours 拿到該行程的 id)。

複合問題可以拆成多次工具呼叫,依序或同時查詢都可以,查完再統整成一個回答。
不要憑空編造政策內容或行程細節 -- 如果查詢結果不足以回答問題(例如查無符合條件的行程),
誠實告知使用者,不要臆測或硬湊一個答案。

回答時可以簡短說明資訊依據(例如「根據退訂政策...」「目前符合條件的行程有...」),讓使用者知道這是有根據的答案。"""

MAX_TOKENS = 4096


async def run_agent(user_message: str, max_turns: int | None = None) -> AsyncIterator[AgentEvent]:
    """The hand-rolled agentic loop: call Claude, check stop_reason, run
    whatever tools it asked for, feed results back, repeat. Yields
    structured events as they happen so a caller (CLI logger now, SSE
    endpoint later) can observe each decision, not just the final text."""
    settings = get_settings()
    client = AsyncAnthropic(api_key=settings.anthropic_api_key)
    turns = max_turns if max_turns is not None else settings.max_agent_turns

    messages: list[MessageParam] = [{"role": "user", "content": user_message}]

    for turn in range(turns):
        response = await client.messages.create(
            model=settings.claude_model,
            max_tokens=MAX_TOKENS,
            system=SYSTEM_PROMPT,
            tools=TOOLS,
            messages=messages,
        )

        if response.stop_reason == "end_turn":
            text = "".join(block.text for block in response.content if block.type == "text")
            yield FinalAnswerEvent(text=text)
            return

        if response.stop_reason == "refusal":
            details = response.stop_details
            yield RefusalEvent(
                category=details.category if details else None,
                explanation=details.explanation if details else None,
            )
            return

        if response.stop_reason == "max_tokens":
            # Ran out of budget mid-turn (mid-thinking or mid-tool-call) --
            # nothing coherent to return, so stop rather than feed a
            # truncated response back into the next turn.
            yield FinalAnswerEvent(
                text="回答時超過長度限制,請試著把問題拆得更簡短明確一點。"
            )
            return

        if response.stop_reason != "tool_use":
            # pause_turn only happens with server-side tools, which this
            # agent doesn't use -- treat any other stop_reason as done
            # rather than looping forever on an unhandled case.
            text = "".join(block.text for block in response.content if block.type == "text")
            yield FinalAnswerEvent(text=text)
            return

        # response.content holds response-side ContentBlock objects, not
        # the request-side ContentBlockParam types MessageParam expects --
        # passing them straight back is the SDK's documented round-trip
        # pattern for continuing a tool-use turn, just stricter than the
        # static types express.
        messages.append({"role": "assistant", "content": cast(Any, response.content)})

        tool_result_blocks: list[ToolResultBlockParam] = []
        for block in response.content:
            if block.type != "tool_use":
                continue

            yield ToolCallEvent(
                turn=turn, tool_use_id=block.id, name=block.name, input=cast(dict[str, Any], block.input)
            )
            result_json, is_error = await execute_tool(block.name, cast(dict[str, Any], block.input))
            yield ToolResultEvent(
                turn=turn, tool_use_id=block.id, name=block.name, result=result_json, is_error=is_error
            )

            tool_result_blocks.append(
                ToolResultBlockParam(
                    type="tool_result",
                    tool_use_id=block.id,
                    content=result_json,
                    is_error=is_error,
                )
            )

        messages.append({"role": "user", "content": tool_result_blocks})

    yield MaxTurnsExceededEvent(
        text="這個問題需要的查詢步驟比較多,已達單次對話的查詢上限,請試著把問題拆得更明確一點。"
    )
