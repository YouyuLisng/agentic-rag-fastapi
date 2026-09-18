from typing import cast

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from slowapi import _rate_limit_exceeded_handler
from slowapi.errors import RateLimitExceeded
from starlette.types import ExceptionHandler

from app.rate_limit import limiter
from app.routers import documents as documents_module
from app.routers.documents import router as documents_router


@pytest.fixture
def client() -> TestClient:
    # A bare app with only the documents router mounted -- no lifespan,
    # so no real DB pool is opened. upload_document itself is monkeypatched
    # per-test, so nothing here touches the network or the database.
    app = FastAPI()
    app.state.limiter = limiter
    app.add_exception_handler(RateLimitExceeded, cast(ExceptionHandler, _rate_limit_exceeded_handler))
    app.include_router(documents_router)
    return TestClient(app)


def test_rejects_unsupported_extension(client: TestClient):
    res = client.post("/documents", files={"file": ("resume.txt", b"hello", "text/plain")})
    assert res.status_code == 400
    assert "不支援的檔案格式" in res.json()["detail"]


def test_rejects_file_over_size_cap(client: TestClient, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(documents_module, "MAX_UPLOAD_BYTES", 10)
    res = client.post("/documents", files={"file": ("small.docx", b"x" * 20, "application/octet-stream")})
    assert res.status_code == 413
    assert "檔案過大" in res.json()["detail"]


def test_accepts_supported_extension_and_returns_service_result(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
):
    async def fake_upload_document(filename: str, content: bytes):
        assert filename == "notes.docx"
        return {"document_id": "doc-1", "filename": filename, "chunk_count": 3}

    monkeypatch.setattr(documents_module, "upload_document", fake_upload_document)

    res = client.post("/documents", files={"file": ("notes.docx", b"real content", "application/octet-stream")})

    assert res.status_code == 200
    assert res.json() == {"document_id": "doc-1", "filename": "notes.docx", "chunk_count": 3}


def test_service_value_error_becomes_400(client: TestClient, monkeypatch: pytest.MonkeyPatch):
    async def failing_upload_document(filename: str, content: bytes):
        raise ValueError("無法解析文件內容")

    monkeypatch.setattr(documents_module, "upload_document", failing_upload_document)

    res = client.post("/documents", files={"file": ("bad.pdf", b"not really a pdf", "application/pdf")})

    assert res.status_code == 400
    assert res.json()["detail"] == "無法解析文件內容"
