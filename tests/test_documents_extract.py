import io

import pytest
from docx import Document as DocxDocument

from app.documents.extract import SUPPORTED_EXTENSIONS, _extract_docx, extract_text


def _make_docx_bytes(paragraphs: list[str]) -> bytes:
    doc = DocxDocument()
    for p in paragraphs:
        doc.add_paragraph(p)
    buf = io.BytesIO()
    doc.save(buf)
    return buf.getvalue()


def test_extract_docx_joins_nonempty_paragraphs():
    content = _make_docx_bytes(["第一段文字", "", "第二段文字"])
    assert _extract_docx(content) == "第一段文字\n\n第二段文字"


def test_extract_docx_empty_document_returns_empty_string():
    content = _make_docx_bytes([])
    assert _extract_docx(content) == ""


def test_supported_extensions_cover_docx_pdf_and_images():
    assert SUPPORTED_EXTENSIONS == {".docx", ".pdf", ".jpg", ".jpeg", ".png"}


async def test_extract_text_rejects_unsupported_extension():
    with pytest.raises(ValueError, match="Unsupported file type"):
        await extract_text("resume.txt", b"whatever")


async def test_extract_text_dispatches_docx_without_calling_claude():
    content = _make_docx_bytes(["Hello world"])
    result = await extract_text("upload.docx", content)
    assert result == "Hello world"
