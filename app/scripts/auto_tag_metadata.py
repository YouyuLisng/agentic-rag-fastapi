"""AI-assisted metadata tagging vs. the manually-curated POLICY_TAGS
dict in app/scripts/seed.py -- the real production answer to "chunk +
metadata: hand-done or handed to AI?" for the metadata half. Chunking
itself stays rule-based even at scale (most business documents have a
fixed structure a regex/paragraph splitter handles fine); metadata
tagging is the step that doesn't scale by hand once document count
moves from "8, read once" to "hundreds, arriving continuously" --
that's the actual argument this script is here to make concrete, not
just assert.

Tags at document granularity (not per-chunk), matching how
POLICY_TAGS is actually applied in seed.py -- every chunk of a
document gets that document's tags, so the fair comparison is
"document in, document-level tags out" on both sides.

Costs real money (one Claude call per document, 8 total -- cheap).
Run: `uv run python -m app.scripts.auto_tag_metadata`
"""

import asyncio
from pathlib import Path

from anthropic import AsyncAnthropic
from dotenv import load_dotenv
from pydantic import BaseModel

load_dotenv()

from app.config import get_settings  # noqa: E402
from app.llm_json import extract_json  # noqa: E402
from app.scripts.seed import POLICY_TAGS  # noqa: E402 -- the manual ground truth being compared against

DATA_DIR = Path(__file__).resolve().parent.parent.parent / "data"

TAG_PROMPT = """你是文件分類助理。請閱讀以下政策文件,產生 2-4 個最能代表這份文件主題的標籤。

標籤要求:
- 用繁體中文,每個標籤 2-4 個字
- 具體到可以拿來做檢索過濾用(例如「退款」「簽證」),不要「政策」「規定」這種太籠統的字
- 如果文件內容橫跨多個主題,標籤也要涵蓋到(例如同時討論退款跟不可抗力的文件,兩個主題都要有對應標籤)

只輸出以下格式的 JSON,不要有任何其他文字:
{{"tags": ["...", "...", ...]}}

文件標題:{title}

文件內容:
{content}"""


class TagResult(BaseModel):
    tags: list[str]


async def auto_tag(client: AsyncAnthropic, model: str, title: str, content: str) -> list[str]:
    response = await client.messages.create(
        model=model,
        max_tokens=256,
        messages=[{"role": "user", "content": TAG_PROMPT.format(title=title, content=content)}],
    )
    raw = "".join(b.text for b in response.content if b.type == "text")
    return TagResult.model_validate_json(extract_json(raw)).tags


def _split_title_and_body(path: Path) -> tuple[str, str]:
    text = path.read_text(encoding="utf-8").strip()
    lines = text.splitlines()
    if lines and lines[0].startswith("#"):
        return lines[0].lstrip("#").strip(), "\n".join(lines[1:]).strip()
    return path.stem, text


async def main() -> None:
    settings = get_settings()
    client = AsyncAnthropic(api_key=settings.anthropic_api_key)

    md_files = sorted((DATA_DIR / "policies").glob("*.md"))
    total_manual = 0
    total_exact_overlap = 0

    for path in md_files:
        title, body = _split_title_and_body(path)
        manual_tags = POLICY_TAGS.get(path.stem, [])
        ai_tags = await auto_tag(client, settings.claude_model_fast, title, body)

        exact_overlap = set(manual_tags) & set(ai_tags)
        total_manual += len(manual_tags)
        total_exact_overlap += len(exact_overlap)

        print(f"{path.stem} ({title})")
        print(f"  人工標註: {manual_tags}")
        print(f"  AI 標註:  {ai_tags}")
        print(f"  完全一致的字: {sorted(exact_overlap) or '(無)'}")
        print()

    print(f"完全字面相符率: {total_exact_overlap}/{total_manual}")
    print(
        "注意:這是「字面完全相同」的比率,不是語意相似度 -- AI 可能用『退費』"
        "而人工標的是『退款』,語意一樣但這個統計會算不一致。實際要用的話,"
        "應該人工抽查而不是只看字面比對。"
    )


if __name__ == "__main__":
    asyncio.run(main())
