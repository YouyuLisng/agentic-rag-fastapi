import base64
import io
from typing import Any, Literal, cast

from anthropic import AsyncAnthropic
from anthropic.types import (
    Base64ImageSourceParam,
    Base64PDFSourceParam,
    DocumentBlockParam,
    ImageBlockParam,
    MessageParam,
    TextBlockParam,
)
from docx import Document as DocxDocument

from app.config import get_settings

TRANSCRIBE_PROMPT = (
    "請完整轉錄這份文件裡的所有文字內容,保持原有段落結構,不要摘要、不要省略,"
    "也不要加上任何額外說明或評論 -- 只要純文字內容。"
)

TRANSCRIBE_MAX_TOKENS = 8192

# PDF and images both go through Claude's vision/document understanding
# rather than a dedicated OCR library -- this handles scanned PDFs and
# photographed text uniformly, without needing to first detect whether
# a given PDF actually has a text layer.
_IMAGE_MEDIA_TYPES: dict[str, Literal["image/jpeg", "image/png"]] = {
    ".jpg": "image/jpeg",
    ".jpeg": "image/jpeg",
    ".png": "image/png",
}

SUPPORTED_EXTENSIONS = {".docx", ".pdf", *_IMAGE_MEDIA_TYPES.keys()}


def _extract_docx(content: bytes) -> str:
    # .docx already has real, machine-readable text in its XML -- no
    # need to involve the model at all, unlike PDF/images.
    doc = DocxDocument(io.BytesIO(content))
    return "\n\n".join(p.text for p in doc.paragraphs if p.text.strip())


async def _transcribe_via_claude(block: DocumentBlockParam | ImageBlockParam) -> str:
    settings = get_settings()
    client = AsyncAnthropic(api_key=settings.anthropic_api_key)

    text_block: TextBlockParam = {"type": "text", "text": TRANSCRIBE_PROMPT}
    messages: list[MessageParam] = [{"role": "user", "content": [cast(Any, block), text_block]}]

    response = await client.messages.create(
        model=settings.claude_model_smart,
        max_tokens=TRANSCRIBE_MAX_TOKENS,
        messages=messages,
    )
    return "".join(b.text for b in response.content if b.type == "text")


async def extract_text(filename: str, content: bytes) -> str:
    ext = "." + filename.rsplit(".", 1)[-1].lower() if "." in filename else ""
    b64 = base64.standard_b64encode(content).decode("utf-8")

    if ext == ".docx":
        return _extract_docx(content)

    if ext == ".pdf":
        pdf_source: Base64PDFSourceParam = {"type": "base64", "media_type": "application/pdf", "data": b64}
        document_block: DocumentBlockParam = {"type": "document", "source": pdf_source}
        return await _transcribe_via_claude(document_block)

    image_media_type = _IMAGE_MEDIA_TYPES.get(ext)
    if image_media_type is not None:
        image_source: Base64ImageSourceParam = {"type": "base64", "media_type": image_media_type, "data": b64}
        image_block: ImageBlockParam = {"type": "image", "source": image_source}
        return await _transcribe_via_claude(image_block)

    raise ValueError(f"Unsupported file type: {ext or filename!r}")
