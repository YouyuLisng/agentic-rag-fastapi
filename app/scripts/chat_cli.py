"""Terminal harness for the agentic loop -- logs every tool call/result/
decision per turn so multi-step behavior is actually observable, not just
the final answer. Run: `uv run python -m app.scripts.chat_cli "<question>"`
"""

import asyncio
import sys

from dotenv import load_dotenv

load_dotenv()

from app.agent.loop import run_agent  # noqa: E402
from app.db import close_pool, init_pool  # noqa: E402


async def main() -> None:
    question = sys.argv[1] if len(sys.argv) > 1 else "退訂政策是什麼?另外我想知道去峇里島要辦簽證嗎?"

    print(f"問題: {question}\n{'-' * 60}")

    await init_pool()
    try:
        async for event in run_agent(question):
            match event.type:
                case "tool_call":
                    print(f"[turn {event.turn}] ({event.model}) 呼叫工具: {event.name}({event.input})")
                case "tool_result":
                    status = "ERROR" if event.is_error else "OK"
                    preview = event.result[:200] + ("..." if len(event.result) > 200 else "")
                    print(f"[turn {event.turn}] 工具結果 ({status}): {preview}")
                case "final_answer":
                    print(f"\n{'-' * 60}\n最終回答 (by {event.model}):\n{event.text}")
                case "refusal":
                    print(f"\n模型拒絕回答: {event.category} -- {event.explanation}")
                case "max_turns_exceeded":
                    print(f"\n{event.text}")
    finally:
        await close_pool()


if __name__ == "__main__":
    asyncio.run(main())
